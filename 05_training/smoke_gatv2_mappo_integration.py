from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv


class GATv2Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, edge_dim: Optional[int] = None):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=2, concat=True, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=1, concat=True, edge_dim=edge_dim)

    def forward(self, data):
        edge_attr = getattr(data, "edge_attr", None)
        x = self.conv1(data.x, data.edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index, edge_attr=edge_attr)
        return F.relu(x)


class TinyGATv2MAPPO(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, action_dim: int, y_dim: int, edge_dim: Optional[int]):
        super().__init__()
        self.encoder = GATv2Encoder(in_channels, hidden_channels, edge_dim=edge_dim)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_channels, action_dim),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_channels, 1),
        )
        self.target_head = torch.nn.Linear(hidden_channels, y_dim)

    def forward(self, batch, num_agents: int):
        node_embeddings = self.encoder(batch)
        agent_embeddings = node_embeddings[: min(num_agents, node_embeddings.size(0))]
        graph_embedding = node_embeddings.mean(dim=0, keepdim=True)
        return {
            "node_embeddings": node_embeddings,
            "agent_logits": self.actor(agent_embeddings),
            "value": self.critic(graph_embedding).reshape(-1),
            "target_pred": self.target_head(node_embeddings),
        }


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_pt_files(dataset_dir: Path, split: Optional[str], max_files: int) -> List[Path]:
    search_dir = dataset_dir / split if split else dataset_dir
    files = sorted(Path(p) for p in glob.glob(str(search_dir / "*.pt")))
    if not files and split is None:
        for child in ("train", "val", "test"):
            files.extend(sorted(Path(p) for p in glob.glob(str(dataset_dir / child / "*.pt"))))
    if not files:
        raise FileNotFoundError(f"No .pt files found under {search_dir}")
    return files[:max_files]


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false.")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(requested)


def get_mask(batch):
    mask = getattr(batch, "train_mask", None)
    if mask is None:
        mask = getattr(batch, "node_mask", None)
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.bool()
    return mask


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    files = resolve_pt_files(dataset_dir, args.split, args.max_files)
    data_list = [torch_load(path) for path in files]
    sample = data_list[0]

    y = getattr(sample, "y", None)
    if y is None or y.ndim != 2:
        raise ValueError("Expected sample.y as a node-level 2D target tensor.")

    edge_attr = getattr(sample, "edge_attr", None)
    edge_dim = edge_attr.size(1) if edge_attr is not None else None
    device = resolve_device(args.device)

    model = TinyGATv2MAPPO(
        in_channels=sample.x.size(1),
        hidden_channels=args.hidden_channels,
        action_dim=args.action_dim,
        y_dim=y.size(1),
        edge_dim=edge_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(data_list, batch_size=1, shuffle=False, num_workers=0)

    losses: List[float] = []
    action_samples: List[List[int]] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch, args.num_agents)

        dist = Categorical(logits=out["agent_logits"])
        actions = dist.sample()
        entropy = dist.entropy().mean()

        target = batch.y.float()
        mask = get_mask(batch)
        target_pred = out["target_pred"][mask] if mask is not None else out["target_pred"]
        target_loss = F.mse_loss(target_pred, target[mask] if mask is not None else target)

        reward_proxy = -target_loss.detach()
        actor_loss = -(dist.log_prob(actions).mean() * reward_proxy)
        critic_loss = F.mse_loss(out["value"], reward_proxy.reshape_as(out["value"]))
        loss = actor_loss + critic_loss + target_loss - args.entropy_coef * entropy

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        action_samples.append([int(v) for v in actions.detach().cpu().tolist()])

    return {
        "status": "PASS",
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "files_loaded": len(files),
        "first_file": str(files[0]),
        "device": str(device),
        "losses": losses,
        "action_samples": action_samples,
        "note": "This is a gradient-flow smoke test for GATv2 encoder + MAPPO-style actor/critic heads.",
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny GATv2 + MAPPO-style integration smoke test.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing .pt files or train/val/test subdirectories.")
    parser.add_argument("--split", default=None, choices=[None, "train", "val", "test"])
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, cuda:0, or auto.")
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    return parser.parse_args(argv)


def main() -> None:
    try:
        report = run(parse_args())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        sys.exit(2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
