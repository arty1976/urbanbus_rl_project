from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv


class NodeLevelGATv2(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, edge_dim: Optional[int] = None):
        super().__init__()
        self.conv1 = GATv2Conv(
            in_channels,
            hidden_channels,
            heads=2,
            concat=True,
            edge_dim=edge_dim,
        )
        self.conv2 = GATv2Conv(
            hidden_channels * 2,
            hidden_channels,
            heads=1,
            concat=True,
            edge_dim=edge_dim,
        )
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        edge_attr = getattr(data, "edge_attr", None)
        x = self.conv1(data.x, data.edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        return self.lin(x)


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


def data_summary(data) -> Dict[str, Any]:
    edge_attr = getattr(data, "edge_attr", None)
    y = getattr(data, "y", None)
    return {
        "x_shape": list(data.x.shape),
        "edge_index_shape": list(data.edge_index.shape),
        "edge_attr_shape": list(edge_attr.shape) if edge_attr is not None else None,
        "y_shape": list(y.shape) if y is not None else None,
        "has_train_mask": getattr(data, "train_mask", None) is not None,
        "has_node_mask": getattr(data, "node_mask", None) is not None,
    }


def build_model_from_sample(sample, hidden_channels: int) -> NodeLevelGATv2:
    y = getattr(sample, "y", None)
    if y is None or y.ndim != 2:
        raise ValueError("Expected sample.y as a node-level 2D target tensor.")
    edge_attr = getattr(sample, "edge_attr", None)
    edge_dim = edge_attr.size(1) if edge_attr is not None else None
    return NodeLevelGATv2(
        in_channels=sample.x.size(1),
        hidden_channels=hidden_channels,
        out_channels=y.size(1),
        edge_dim=edge_dim,
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    files = resolve_pt_files(dataset_dir, args.split, args.max_files)
    data_list = [torch_load(path) for path in files]
    sample = data_list[0]

    report: Dict[str, Any] = {
        "mode": args.mode,
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "files_loaded": len(files),
        "first_file": str(files[0]),
        "last_file": str(files[-1]),
        "sample": data_summary(sample),
    }

    if args.mode == "load":
        return report

    device = resolve_device(args.device)
    loader = DataLoader(data_list, batch_size=args.batch_size, shuffle=args.mode == "train", num_workers=0)
    model = build_model_from_sample(sample, args.hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    report["device"] = str(device)

    max_epochs = args.epochs if args.mode == "train" else 1
    step_limit = 1 if args.mode in {"forward", "backward"} else None
    losses: List[float] = []

    model.train(args.mode in {"backward", "train"})
    for epoch in range(max_epochs):
        for step_idx, batch in enumerate(loader, start=1):
            batch = batch.to(device)
            pred = model(batch)
            target = batch.y.float()
            if pred.shape != target.shape:
                raise ValueError(f"Prediction/target shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")

            mask = get_mask(batch)
            loss_pred = pred[mask] if mask is not None else pred
            loss_target = target[mask] if mask is not None else target
            loss = F.mse_loss(loss_pred, loss_target)
            losses.append(float(loss.detach().cpu().item()))

            if args.mode in {"backward", "train"}:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if step_limit is not None and step_idx >= step_limit:
                break

    report["losses"] = losses
    report["status"] = "PASS"
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mac smoke runner for UrbanBus GATv2 PyG tensors.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing .pt files or train/val/test subdirectories.")
    parser.add_argument("--split", default=None, choices=[None, "train", "val", "test"], help="Optional split subdirectory.")
    parser.add_argument("--mode", default="load", choices=["load", "forward", "backward", "train"])
    parser.add_argument("--device", default="cpu", help="cpu, mps, cuda, cuda:0, or auto.")
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
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
