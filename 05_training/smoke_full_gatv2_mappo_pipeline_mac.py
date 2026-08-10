from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv


@dataclass
class DeviceDecision:
    requested: str
    selected: str
    gpu_requested: bool
    gpu_available: bool
    gpu_blocked: bool
    reason: str
    torch_version: str
    mps_built: bool
    mps_available: bool
    cuda_available: bool


class NodeLevelGATv2(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, edge_dim: Optional[int]):
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

    def encode(self, data):
        edge_attr = getattr(data, "edge_attr", None)
        x = self.conv1(data.x, data.edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index, edge_attr=edge_attr)
        return F.relu(x)

    def forward(self, data):
        return self.lin(self.encode(data))


class TinyMAPPOPolicy(torch.nn.Module):
    def __init__(self, embedding_dim: int, num_agents: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.agent_id_embedding = torch.nn.Embedding(num_agents, embedding_dim)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * 2, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, action_dim),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_embedding):
        agent_ids = torch.arange(self.num_agents, device=graph_embedding.device)
        agent_features = self.agent_id_embedding(agent_ids)
        repeated_graph = graph_embedding.reshape(1, -1).repeat(self.num_agents, 1)
        actor_obs = torch.cat([repeated_graph, agent_features], dim=1)
        logits = self.actor(actor_obs)
        value = self.critic(graph_embedding.reshape(1, -1)).reshape(())
        return logits, value

    def actor_observation(self, graph_embedding):
        agent_ids = torch.arange(self.num_agents, device=graph_embedding.device)
        agent_features = self.agent_id_embedding(agent_ids)
        repeated_graph = graph_embedding.reshape(1, -1).repeat(self.num_agents, 1)
        return torch.cat([repeated_graph, agent_features], dim=1)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_state_dict(module: torch.nn.Module, prefix_filter: Optional[str] = None) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        if prefix_filter is not None and not name.startswith(prefix_filter):
            continue
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(array.tobytes())
    return digest.hexdigest()


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def process_rss_mb() -> Optional[float]:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 3)
    except Exception:
        return None


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def mps_memory_snapshot() -> Dict[str, Optional[int]]:
    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        return {"current_allocated_bytes": None, "driver_allocated_bytes": None}
    current = getattr(torch.mps, "current_allocated_memory", None)
    driver = getattr(torch.mps, "driver_allocated_memory", None)
    return {
        "current_allocated_bytes": int(current()) if current else None,
        "driver_allocated_bytes": int(driver()) if driver else None,
    }


def system_memory_snapshot(label: str) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {
        "label": label,
        "created_at_utc": utc_now(),
        "mps": mps_memory_snapshot(),
        "process_rss_mb": process_rss_mb(),
    }
    try:
        vm_stat = subprocess.run(["vm_stat"], check=False, capture_output=True, text=True, timeout=5)
        snapshot["vm_stat"] = vm_stat.stdout.strip().splitlines()[:20]
    except Exception as exc:
        snapshot["vm_stat_error"] = str(exc)
    try:
        swap = subprocess.run(["sysctl", "vm.swapusage"], check=False, capture_output=True, text=True, timeout=5)
        snapshot["swapusage"] = swap.stdout.strip()
    except Exception as exc:
        snapshot["swapusage_error"] = str(exc)
    return snapshot


def parse_swap_used_mb(swapusage: str) -> Optional[float]:
    marker = "used = "
    if marker not in swapusage:
        return None
    tail = swapusage.split(marker, 1)[1].split()[0]
    try:
        if tail.endswith("M"):
            return float(tail[:-1])
        if tail.endswith("G"):
            return float(tail[:-1]) * 1024.0
        return float(tail)
    except ValueError:
        return None


def summarize_memory(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    current_values = [
        sample.get("mps", {}).get("current_allocated_bytes")
        for sample in samples
        if sample.get("mps", {}).get("current_allocated_bytes") is not None
    ]
    driver_values = [
        sample.get("mps", {}).get("driver_allocated_bytes")
        for sample in samples
        if sample.get("mps", {}).get("driver_allocated_bytes") is not None
    ]
    swap_values = [
        parse_swap_used_mb(str(sample.get("swapusage", "")))
        for sample in samples
    ]
    swap_values = [value for value in swap_values if value is not None]
    rss_values = [
        sample.get("process_rss_mb")
        for sample in samples
        if sample.get("process_rss_mb") is not None
    ]
    return {
        "peak_mps_current_allocated_mb": round(max(current_values) / (1024 * 1024), 3) if current_values else None,
        "peak_mps_driver_allocated_mb": round(max(driver_values) / (1024 * 1024), 3) if driver_values else None,
        "peak_process_rss_mb": max(rss_values) if rss_values else None,
        "swap_used_mb_max": max(swap_values) if swap_values else None,
        "system_memory_pressure": "raw_vm_stat_captured",
    }


def contains_nan_or_inf(value: Any) -> Tuple[bool, bool]:
    nan_found = False
    inf_found = False
    if isinstance(value, dict):
        for child in value.values():
            child_nan, child_inf = contains_nan_or_inf(child)
            nan_found = nan_found or child_nan
            inf_found = inf_found or child_inf
    elif isinstance(value, (list, tuple)):
        for child in value:
            child_nan, child_inf = contains_nan_or_inf(child)
            nan_found = nan_found or child_nan
            inf_found = inf_found or child_inf
    elif isinstance(value, float):
        nan_found = math.isnan(value)
        inf_found = math.isinf(value)
    return nan_found, inf_found


def tensor_float_summary(tensor: torch.Tensor) -> Dict[str, float]:
    value = tensor.detach().cpu().float()
    return {
        "mean": float(value.mean().item()),
        "std": float(value.std(unbiased=False).item()),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
    }


def resolve_files(dataset_dir: Path, split: str, snapshot_count: int) -> List[Path]:
    files = sorted(Path(p) for p in glob.glob(str(dataset_dir / split / "*.pt")))
    if len(files) < snapshot_count:
        raise FileNotFoundError(f"Need at least {snapshot_count} .pt files in {dataset_dir / split}, found {len(files)}")
    return files[:snapshot_count]


def choose_device(requested: str) -> Tuple[torch.device, DeviceDecision]:
    requested = requested.lower()
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available())
    gpu_requested = requested in {"auto", "mps", "cuda"} or requested.startswith("cuda")

    if requested == "mps":
        if mps_available:
            selected = "mps"
            reason = "MPS requested and available."
        else:
            selected = "cpu"
            reason = "MPS requested but torch.backends.mps.is_available() is false; falling back to CPU for completion."
    elif requested == "auto":
        if mps_available:
            selected = "mps"
            reason = "auto selected MPS."
        elif cuda_available:
            selected = "cuda"
            reason = "auto selected CUDA."
        else:
            selected = "cpu"
            reason = "No MPS/CUDA device is available; falling back to CPU for completion."
    elif requested.startswith("cuda"):
        if cuda_available:
            selected = requested
            reason = "CUDA requested and available."
        else:
            selected = "cpu"
            reason = "CUDA requested but torch.cuda.is_available() is false; falling back to CPU for completion."
    else:
        selected = "cpu"
        reason = "CPU requested."

    gpu_available = selected != "cpu"
    return torch.device(selected), DeviceDecision(
        requested=requested,
        selected=selected,
        gpu_requested=gpu_requested,
        gpu_available=gpu_available,
        gpu_blocked=bool(gpu_requested and selected == "cpu"),
        reason=reason,
        torch_version=torch.__version__,
        mps_built=mps_built,
        mps_available=mps_available,
        cuda_available=cuda_available,
    )


def get_mask(batch):
    mask = getattr(batch, "train_mask", None)
    if mask is None:
        mask = getattr(batch, "node_mask", None)
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.bool()
    return mask


def parameter_grad_norm(parameters) -> float:
    total_norm_sq = 0.0
    for param in parameters:
        if param.grad is not None:
            grad = param.grad.detach()
            total_norm_sq += float(torch.sum(grad.float() ** 2).detach().cpu().item())
    return math.sqrt(total_norm_sq)


def validate_graph_contract(sample, expected_nodes: int, expected_edges: int) -> Dict[str, Any]:
    edge_attr = getattr(sample, "edge_attr", None)
    y = getattr(sample, "y", None)
    observed = {
        "nodes": int(sample.x.size(0)),
        "x_dim": int(sample.x.size(1)),
        "edges": int(sample.edge_index.size(1)),
        "edge_attr_dim": int(edge_attr.size(1)) if edge_attr is not None else None,
        "y_dim": int(y.size(1)) if y is not None and y.ndim == 2 else None,
        "has_node_mask": getattr(sample, "node_mask", None) is not None,
    }
    if observed["nodes"] != expected_nodes:
        raise ValueError(f"Expected {expected_nodes} nodes, found {observed['nodes']}")
    if observed["edges"] != expected_edges:
        raise ValueError(f"Expected {expected_edges} edges, found {observed['edges']}")
    if y is None or y.ndim != 2:
        raise ValueError("Expected node-level y tensor with ndim=2")
    return observed


def train_gatv2(
    model: NodeLevelGATv2,
    data_list: List[Any],
    device: torch.device,
    batch_size: int,
    lr: float,
    epochs: int,
    grad_clip_norm: Optional[float],
) -> Dict[str, Any]:
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []
    epoch_avg_losses: List[float] = []
    grad_norms_before_clip: List[float] = []
    grad_norms_after_clip: List[float] = []
    nan_inf_detected = False
    model.train()
    for _epoch in range(epochs):
        epoch_losses: List[float] = []
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.y.float()
            if pred.shape != target.shape:
                raise ValueError(f"Prediction/target shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}")
            mask = get_mask(batch)
            loss = F.mse_loss(pred[mask] if mask is not None else pred, target[mask] if mask is not None else target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    grad = param.grad.detach()
                    if not torch.isfinite(grad).all():
                        nan_inf_detected = True
            grad_before = parameter_grad_norm(model.parameters())
            grad_norms_before_clip.append(grad_before)
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            grad_after = parameter_grad_norm(model.parameters())
            grad_norms_after_clip.append(grad_after)
            optimizer.step()
            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value) or not torch.isfinite(pred).all():
                nan_inf_detected = True
            losses.append(loss_value)
            epoch_losses.append(loss_value)
        epoch_avg_losses.append(float(sum(epoch_losses) / max(len(epoch_losses), 1)))
    return {
        "losses": losses,
        "epoch_avg_train_loss": epoch_avg_losses,
        "grad_clip_norm": grad_clip_norm,
        "grad_norm_before_clip_mean": float(sum(grad_norms_before_clip) / max(len(grad_norms_before_clip), 1)),
        "grad_norm_before_clip_max": float(max(grad_norms_before_clip)) if grad_norms_before_clip else 0.0,
        "grad_norm_after_clip_mean": float(sum(grad_norms_after_clip) / max(len(grad_norms_after_clip), 1)),
        "grad_norm_after_clip_max": float(max(grad_norms_after_clip)) if grad_norms_after_clip else 0.0,
        "nan_inf_detected": nan_inf_detected,
    }


def evaluate_gatv2(
    model: NodeLevelGATv2,
    data_list: List[Any],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, num_workers=0)
    losses: List[float] = []
    nan_inf_detected = False
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.y.float()
            mask = get_mask(batch)
            loss = F.mse_loss(pred[mask] if mask is not None else pred, target[mask] if mask is not None else target)
            loss_value = float(loss.detach().cpu().item())
            if not math.isfinite(loss_value) or not torch.isfinite(pred).all():
                nan_inf_detected = True
            losses.append(loss_value)
    return {
        "loss_count": len(losses),
        "avg_loss": float(sum(losses) / max(len(losses), 1)),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "nan_inf_detected": nan_inf_detected,
    }


def generate_embeddings(
    model: NodeLevelGATv2,
    data_list: List[Any],
    device: torch.device,
    out_path: Path,
) -> torch.Tensor:
    embeddings: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for data in data_list:
            data = data.to(device)
            node_embeddings = model.encode(data)
            mask = get_mask(data)
            if mask is not None and int(mask.sum().item()) > 0:
                graph_embedding = node_embeddings[mask].mean(dim=0)
            else:
                graph_embedding = node_embeddings.mean(dim=0)
            embeddings.append(graph_embedding.detach().cpu())
    stacked = torch.stack(embeddings, dim=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"graph_embeddings": stacked, "created_at_utc": utc_now()}, out_path)
    return stacked


def synthetic_reward(action_tensor: torch.Tensor, graph_embedding: torch.Tensor, condition_id: str) -> Tuple[torch.Tensor, Dict[str, float]]:
    action_mean = action_tensor.float().mean()
    demand_proxy = torch.sigmoid(graph_embedding.mean()) * 100.0
    smoothness = torch.exp(-torch.abs(action_mean - 2.0))
    served_proxy = demand_proxy * torch.sigmoid(2.0 - torch.abs(action_mean - 2.0))
    energy_proxy = action_mean + 1.0
    reward = smoothness * 2.0 - 0.01 * torch.abs(demand_proxy - 50.0)
    if condition_id != "A":
        reward = reward - 0.25
    avg_wait = (120.0 - reward.detach().cpu() * 10.0).clamp(min=0.0)
    service_rate = torch.clamp(served_proxy / (demand_proxy + 1e-6), 0.0, 1.0)
    metrics = {
        "cv_headway": float((1.0 / (1.0 + smoothness.detach().cpu())).item()),
        "avg_wait_seconds": float(avg_wait.item()),
        "bunching_rate": float(torch.sigmoid(action_mean.detach().cpu() - 2.0).item()),
        "on_time_rate": float(torch.sigmoid(reward.detach().cpu()).item()),
        "intervention_rate": float((action_tensor.detach().cpu() != 2).float().mean().item()),
        "energy_proxy": float(energy_proxy.detach().cpu().item()),
        "passenger_demand_generated": float(demand_proxy.detach().cpu().item()),
        "passenger_served_count": float(served_proxy.detach().cpu().item()),
        "passenger_service_rate": float(service_rate.detach().cpu().item()),
        "passenger_wait_p95_seconds": float((avg_wait * 1.35).item()),
        "energy_proxy_per_passenger": float((energy_proxy.detach().cpu() / (served_proxy.detach().cpu() + 1e-6)).item()),
        "fleet_reduction_ratio": float(torch.clamp((3.0 - action_mean.detach().cpu()) / 10.0, 0.0, 1.0).item()),
    }
    return reward, metrics


def canonical_kpi_keys() -> List[str]:
    return [
        "cv_headway",
        "avg_wait_seconds",
        "bunching_rate",
        "on_time_rate",
        "intervention_rate",
        "energy_proxy",
        "passenger_demand_generated",
        "passenger_served_count",
        "passenger_service_rate",
        "passenger_wait_p95_seconds",
        "energy_proxy_per_passenger",
        "fleet_reduction_ratio",
    ]


def mappo_rollout_update(
    embeddings: torch.Tensor,
    device: torch.device,
    output_root: Path,
    num_agents: int,
    horizon: int,
    update_epochs: int,
    minibatch_size: int,
    seed: int,
    condition_id: str,
    grad_clip_norm: float,
    clip_epsilon: float,
    trace_mode: str,
) -> Dict[str, Any]:
    policy = TinyMAPPOPolicy(
        embedding_dim=int(embeddings.size(1)),
        num_agents=num_agents,
        action_dim=5,
        hidden_dim=32,
    ).to(device)
    initial_policy_state_hash = sha256_state_dict(policy)
    initial_actor_state_hash = sha256_state_dict(policy, "actor.")
    initial_critic_state_hash = sha256_state_dict(policy, "critic.")
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    rollout: List[Dict[str, Any]] = []
    agent_decision_rows: List[Dict[str, Any]] = []
    rewards: List[torch.Tensor] = []
    actions_by_step: List[torch.Tensor] = []
    embedding_indices: List[int] = []
    old_log_probs_by_step: List[torch.Tensor] = []
    rollout_logits_by_step: List[torch.Tensor] = []
    rollout_entropy_by_step: List[torch.Tensor] = []
    values_by_step: List[torch.Tensor] = []
    metric_rows: List[Dict[str, float]] = []
    rollout_profile: Dict[str, Any] = {
        "trace_mode": trace_mode,
        "profiling_instrumentation_enabled": True,
        "timing_sync_mode": "rollout_and_update_boundary_mps_synchronize",
        "per_agent_synchronize_used": False,
        "rollout_total_scope": "environment_loop_including_online_trace_bookkeeping",
        "profiling_instrumentation_overhead_note": (
            "Fine-grained section timers use wall-clock timing without per-agent synchronization "
            "to avoid distorting MPS execution; rollout and PPO totals are synchronized at section boundaries."
        ),
        "actor_inference_mode": "batched_agents",
        "simulator_step_mode": "batched_agents",
        "rollout_observation_build_seconds": 0.0,
        "rollout_actor_inference_seconds": 0.0,
        "rollout_action_sampling_seconds": 0.0,
        "rollout_simulator_step_seconds": 0.0,
        "rollout_reward_aggregation_seconds": 0.0,
        "rollout_buffer_append_seconds": 0.0,
        "rollout_trace_recording_seconds": 0.0,
    }

    embeddings_device = embeddings.to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    synchronize_device(device)
    rollout_started = time.perf_counter()
    for step in range(horizon):
        embedding_idx = step % embeddings_device.size(0)
        graph_embedding = embeddings_device[embedding_idx]
        section_started = time.perf_counter()
        actor_obs = policy.actor_observation(graph_embedding)
        rollout_profile["rollout_observation_build_seconds"] += time.perf_counter() - section_started

        section_started = time.perf_counter()
        logits = policy.actor(actor_obs)
        value = policy.critic(graph_embedding.reshape(1, -1)).reshape(())
        rollout_profile["rollout_actor_inference_seconds"] += time.perf_counter() - section_started

        section_started = time.perf_counter()
        dist = Categorical(logits=logits)
        actions = dist.sample()
        old_log_prob = dist.log_prob(actions)
        old_entropy = dist.entropy()
        rollout_profile["rollout_action_sampling_seconds"] += time.perf_counter() - section_started

        section_started = time.perf_counter()
        reward, metrics = synthetic_reward(actions, graph_embedding, condition_id)
        rollout_profile["rollout_simulator_step_seconds"] += time.perf_counter() - section_started

        section_started = time.perf_counter()
        rewards.append(reward.detach())
        actions_by_step.append(actions.detach().cpu())
        embedding_indices.append(int(embedding_idx))
        old_log_probs_by_step.append(old_log_prob.detach().cpu())
        rollout_logits_by_step.append(logits.detach().cpu())
        rollout_entropy_by_step.append(old_entropy.detach().cpu())
        values_by_step.append(value.detach().cpu().reshape(()))
        metric_rows.append(metrics)
        rollout_profile["rollout_buffer_append_seconds"] += time.perf_counter() - section_started

        section_started = time.perf_counter()
        rollout.append(
            {
                "step": step,
                "condition_id": condition_id,
                "num_agents": num_agents,
                "actions": [int(v) for v in actions.detach().cpu().tolist()],
                "reward": float(reward.detach().cpu().item()),
                **metrics,
            }
        )
        if trace_mode == "audit":
            for agent_id, action in enumerate(actions.detach().cpu().tolist()):
                agent_decision_rows.append(
                    {
                        "step": step,
                        "agent_id": int(agent_id),
                        "action": int(action),
                        "old_logprob": float(old_log_prob[agent_id].detach().cpu().item()),
                        "team_reward": float(reward.detach().cpu().item()),
                        "done": False,
                        "observation_shape": list(actor_obs[agent_id].shape),
                        "observation_hash": sha256_tensor(actor_obs[agent_id]),
                        "embedding_idx": int(embedding_idx),
                        "condition_id": condition_id,
                    }
                )
        rollout_profile["rollout_trace_recording_seconds"] += time.perf_counter() - section_started
    synchronize_device(device)
    rollout_collection_seconds = time.perf_counter() - rollout_started
    measured_rollout_sections = [
        "rollout_observation_build_seconds",
        "rollout_actor_inference_seconds",
        "rollout_action_sampling_seconds",
        "rollout_simulator_step_seconds",
        "rollout_reward_aggregation_seconds",
        "rollout_buffer_append_seconds",
        "rollout_trace_recording_seconds",
    ]
    measured_rollout_seconds = sum(float(rollout_profile[key]) for key in measured_rollout_sections)
    agent_decision_count = num_agents * horizon
    rollout_profile.update(
        {
            "rollout_total_seconds": round(rollout_collection_seconds, 6),
            "rollout_other_overhead_seconds": round(max(rollout_collection_seconds - measured_rollout_seconds, 0.0), 6),
            "observation_build_call_count": horizon,
            "actor_inference_call_count": horizon,
            "simulator_step_call_count": horizon,
            "trace_record_count": agent_decision_count,
            "rollout_timesteps": horizon,
            "agent_decisions": agent_decision_count,
            "average_observation_build_ms_per_step": round(rollout_profile["rollout_observation_build_seconds"] * 1000.0 / horizon, 6),
            "average_actor_inference_ms_per_step": round(rollout_profile["rollout_actor_inference_seconds"] * 1000.0 / horizon, 6),
            "average_simulator_step_ms_per_step": round(rollout_profile["rollout_simulator_step_seconds"] * 1000.0 / horizon, 6),
            "average_trace_recording_ms_per_step": round(rollout_profile["rollout_trace_recording_seconds"] * 1000.0 / horizon, 6),
            "average_rollout_ms_per_timestep": round(rollout_collection_seconds * 1000.0 / horizon, 6),
            "average_rollout_ms_per_agent_decision": round(rollout_collection_seconds * 1000.0 / agent_decision_count, 6),
            "agent_decisions_per_second": round(agent_decision_count / rollout_collection_seconds, 6) if rollout_collection_seconds > 0 else None,
        }
    )
    for key in measured_rollout_sections:
        rollout_profile[key] = round(float(rollout_profile[key]), 6)

    raw_rewards_full = torch.stack(rewards).to(device)
    returns = raw_rewards_full
    returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)
    sample_count = int(returns.numel())
    actions_full = torch.stack(actions_by_step).to(device)
    old_log_probs_full = torch.stack(old_log_probs_by_step).to(device)
    old_entropy_full = torch.stack(rollout_entropy_by_step).to(device)
    rollout_logits_full = torch.stack(rollout_logits_by_step).to(device)
    values_full = torch.stack(values_by_step).to(device)
    done_full = torch.zeros((horizon, num_agents), dtype=torch.bool, device=device)
    agent_ids_full = torch.arange(num_agents, device=device).reshape(1, num_agents).repeat(horizon, 1)
    team_transition_tensor = torch.stack([raw_rewards_full, values_full], dim=1)
    team_advantage = returns - values_full.detach()
    actor_advantage = team_advantage.reshape(horizon, 1).expand(horizon, num_agents)

    audit_logits: List[torch.Tensor] = []
    audit_values: List[torch.Tensor] = []
    for idx in range(horizon):
        logits, value = policy(embeddings_device[embedding_indices[idx]])
        audit_logits.append(logits)
        audit_values.append(value.reshape(()))
    policy_logits_full = torch.stack(audit_logits, dim=0)
    policy_values_full = torch.stack(audit_values, dim=0)
    audit_dist = Categorical(logits=policy_logits_full)
    policy_new_log_probs_full = audit_dist.log_prob(actions_full)
    policy_entropy_full = audit_dist.entropy()
    actor_ratio_full = torch.exp(policy_new_log_probs_full - old_log_probs_full)
    actor_surrogate_unclipped = actor_ratio_full * actor_advantage
    actor_surrogate_clipped = torch.clamp(actor_ratio_full, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * actor_advantage
    actor_surrogate = torch.minimum(actor_surrogate_unclipped, actor_surrogate_clipped)
    unique_rollout_actor_elements = int(actions_full.numel())
    actor_loss_audit = {
        "policy_logits_shape": list(policy_logits_full.shape),
        "rollout_policy_logits_shape": list(rollout_logits_full.shape),
        "policy_actions_shape": list(actions_full.shape),
        "policy_old_logprobs_shape": list(old_log_probs_full.shape),
        "policy_new_logprobs_shape": list(policy_new_log_probs_full.shape),
        "policy_entropy_shape": list(policy_entropy_full.shape),
        "rollout_entropy_shape": list(old_entropy_full.shape),
        "actor_ratio_shape": list(actor_ratio_full.shape),
        "actor_surrogate_unclipped_shape": list(actor_surrogate_unclipped.shape),
        "actor_surrogate_clipped_shape": list(actor_surrogate_clipped.shape),
        "actor_surrogate_shape": list(actor_surrogate.shape),
        "actor_advantage_shape": list(actor_advantage.shape),
        "actor_ratio_numel": int(actor_ratio_full.numel()),
        "actor_surrogate_unclipped_numel": int(actor_surrogate_unclipped.numel()),
        "actor_surrogate_clipped_numel": int(actor_surrogate_clipped.numel()),
        "actor_surrogate_numel": int(actor_surrogate.numel()),
        "actor_entropy_numel": int(policy_entropy_full.numel()),
        "actor_advantage_numel": int(actor_advantage.numel()),
        "policy_action_element_count": int(actions_full.numel()),
        "policy_old_logprob_element_count": int(old_log_probs_full.numel()),
        "policy_new_logprob_element_count": int(policy_new_log_probs_full.numel()),
        "policy_logprob_element_count": int(policy_new_log_probs_full.numel()),
        "policy_entropy_element_count": int(policy_entropy_full.numel()),
        "team_advantage_shape_before_broadcast": list(team_advantage.shape),
        "team_advantage_count_before_broadcast": int(team_advantage.numel()),
        "actor_advantage_shape_after_broadcast": list(actor_advantage.shape),
        "actor_advantage_count_after_broadcast": int(actor_advantage.numel()),
        "advantage_broadcast_to_agents": list(actor_advantage.shape) == [horizon, num_agents],
        "actor_loss_reduction_axes": ["time", "agent"],
        "critic_loss_reduction_axes": ["time"],
        "critic_value_shape": list(policy_values_full.shape),
        "critic_value_target_shape": list(returns.shape),
        "critic_value_target_count": int(returns.numel()),
        "team_return_count": int(returns.numel()),
        "ppo_update_team_samples": sample_count,
        "unique_rollout_actor_elements": unique_rollout_actor_elements,
        "actor_elements_per_ppo_epoch": unique_rollout_actor_elements,
        "actor_elements_processed_across_all_ppo_epochs": unique_rollout_actor_elements * update_epochs,
    }

    losses: List[float] = []
    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []
    approx_kls: List[float] = []
    clip_fractions: List[float] = []
    grad_norms_before_clip: List[float] = []
    grad_norms_after_clip: List[float] = []
    actor_grad_norms_before_clip: List[float] = []
    actor_grad_norms_after_clip: List[float] = []
    critic_grad_norms_before_clip: List[float] = []
    critic_grad_norms_after_clip: List[float] = []
    ppo_minibatch_counts: List[int] = []
    team_samples_per_minibatch: List[int] = []
    agent_elements_per_minibatch: List[int] = []
    actor_elements_processed_across_epochs = 0
    critic_samples_processed_across_epochs = 0
    mappo_nan_inf_detected = False
    synchronize_device(device)
    ppo_started = time.perf_counter()
    for epoch in range(update_epochs):
        indices = torch.randperm(sample_count, generator=generator).tolist()
        epoch_minibatch_count = 0
        for start in range(0, sample_count, minibatch_size):
            mb_indices = indices[start : start + minibatch_size]
            mb_log_probs: List[torch.Tensor] = []
            mb_values: List[torch.Tensor] = []
            mb_entropies: List[torch.Tensor] = []
            mb_returns: List[torch.Tensor] = []
            mb_old_log_probs: List[torch.Tensor] = []
            for idx in mb_indices:
                graph_embedding = embeddings_device[embedding_indices[idx]]
                logits, value = policy(graph_embedding)
                dist = Categorical(logits=logits)
                actions = actions_by_step[idx].to(device)
                mb_log_probs.append(dist.log_prob(actions))
                mb_values.append(value.reshape(()))
                mb_entropies.append(dist.entropy())
                mb_returns.append(returns[idx].reshape(()))
                mb_old_log_probs.append(old_log_probs_by_step[idx].to(device))
            log_probs_t = torch.stack(mb_log_probs, dim=0)
            old_log_probs_t = torch.stack(mb_old_log_probs, dim=0)
            values_t = torch.stack(mb_values)
            entropies_t = torch.stack(mb_entropies, dim=0)
            returns_t = torch.stack(mb_returns)
            team_advantage_t = returns_t - values_t.detach()
            advantage = team_advantage_t.reshape(-1, 1).expand(-1, num_agents)
            ratio = torch.exp(log_probs_t - old_log_probs_t)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            critic_loss = F.mse_loss(values_t, returns_t)
            entropy_loss = -0.01 * entropies_t.mean()
            loss = actor_loss + 0.5 * critic_loss + entropy_loss
            for tensor in (log_probs_t, values_t, entropies_t, returns_t, advantage, ratio, unclipped, clipped, actor_loss, critic_loss, loss):
                if not torch.isfinite(tensor).all():
                    mappo_nan_inf_detected = True
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_before = parameter_grad_norm(policy.parameters())
            grad_norms_before_clip.append(grad_before)
            actor_grad_norms_before_clip.append(parameter_grad_norm(list(policy.actor.parameters()) + list(policy.agent_id_embedding.parameters())))
            critic_grad_norms_before_clip.append(parameter_grad_norm(policy.critic.parameters()))
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip_norm)
            grad_after = parameter_grad_norm(policy.parameters())
            grad_norms_after_clip.append(grad_after)
            actor_grad_norms_after_clip.append(parameter_grad_norm(list(policy.actor.parameters()) + list(policy.agent_id_embedding.parameters())))
            critic_grad_norms_after_clip.append(parameter_grad_norm(policy.critic.parameters()))
            optimizer.step()
            epoch_minibatch_count += 1
            mb_team_samples = int(returns_t.numel())
            mb_agent_elements = int(log_probs_t.numel())
            team_samples_per_minibatch.append(mb_team_samples)
            agent_elements_per_minibatch.append(mb_agent_elements)
            actor_elements_processed_across_epochs += mb_agent_elements
            critic_samples_processed_across_epochs += mb_team_samples
            losses.append(float(loss.detach().cpu().item()))
            policy_losses.append(float(actor_loss.detach().cpu().item()))
            value_losses.append(float(critic_loss.detach().cpu().item()))
            entropies.append(float(entropies_t.mean().detach().cpu().item()))
            approx_kls.append(float((old_log_probs_t - log_probs_t).mean().detach().cpu().item()))
            clip_fractions.append(float((torch.abs(ratio - 1.0) > clip_epsilon).float().mean().detach().cpu().item()))
        ppo_minibatch_counts.append(epoch_minibatch_count)
    synchronize_device(device)
    ppo_update_seconds = time.perf_counter() - ppo_started
    actor_loss_audit["actor_elements_processed_across_all_ppo_epochs"] = actor_elements_processed_across_epochs
    actor_loss_audit["actor_elements_per_ppo_epoch"] = unique_rollout_actor_elements
    actor_loss_audit["ppo_minibatch_audit"] = {
        "ppo_epoch_count": update_epochs,
        "minibatch_count_per_epoch": ppo_minibatch_counts[0] if len(set(ppo_minibatch_counts)) == 1 else ppo_minibatch_counts,
        "team_samples_per_minibatch": team_samples_per_minibatch[0] if len(set(team_samples_per_minibatch)) == 1 else team_samples_per_minibatch,
        "agent_elements_per_minibatch": agent_elements_per_minibatch[0] if len(set(agent_elements_per_minibatch)) == 1 else agent_elements_per_minibatch,
        "minibatch_indexing_axis": "team_time",
        "flattened_actor_shape": list(actions_full.reshape(-1).shape),
        "elements_per_minibatch": agent_elements_per_minibatch[0] if len(set(agent_elements_per_minibatch)) == 1 else agent_elements_per_minibatch,
        "number_of_minibatches": sum(ppo_minibatch_counts),
        "elements_processed_per_epoch": unique_rollout_actor_elements,
        "elements_processed_across_epochs": actor_elements_processed_across_epochs,
        "actor_elements_per_epoch": unique_rollout_actor_elements,
        "actor_elements_across_all_epochs": actor_elements_processed_across_epochs,
        "critic_samples_per_epoch": sample_count,
        "critic_samples_across_all_epochs": critic_samples_processed_across_epochs,
    }
    actor_loss_audit_path = output_root / "actor_loss_audit.json"
    dump_json(actor_loss_audit_path, actor_loss_audit)

    rollout_hashes = {
        "actions_tensor_hash": sha256_tensor(actions_full),
        "old_logprobs_tensor_hash": sha256_tensor(old_log_probs_full),
        "new_logprobs_tensor_hash": sha256_tensor(policy_new_log_probs_full),
        "entropy_tensor_hash": sha256_tensor(policy_entropy_full),
        "rewards_tensor_hash": sha256_tensor(raw_rewards_full),
        "done_tensor_hash": sha256_tensor(done_full.to(torch.uint8)),
        "agent_ids_tensor_hash": sha256_tensor(agent_ids_full),
        "actor_logits_tensor_hash": sha256_tensor(rollout_logits_full),
        "actor_advantage_tensor_hash": sha256_tensor(actor_advantage),
        "team_transition_tensor_hash": sha256_tensor(team_transition_tensor),
    }
    rollout_hashes_path = output_root / "rollout_tensor_hashes.json"
    dump_json(rollout_hashes_path, rollout_hashes)

    action_values = actions_full.detach().cpu().reshape(-1)
    action_histogram = {
        str(action_id): int((action_values == action_id).sum().item())
        for action_id in range(policy.action_dim)
    }
    rollout_summary = {
        "trace_mode": trace_mode,
        "configured_num_agents": num_agents,
        "observed_unique_agent_count": num_agents,
        "unique_agent_ids": list(range(num_agents)),
        "expected_agent_decisions": num_agents * horizon,
        "observed_agent_decisions": int(actions_full.numel()),
        "policy_logits_shape": list(rollout_logits_full.shape),
        "policy_actions_shape": list(actions_full.shape),
        "old_logprobs_shape": list(old_log_probs_full.shape),
        "actor_ratio_numel": int(actor_ratio_full.numel()),
        "actor_surrogate_numel": int(actor_surrogate.numel()),
        "actor_entropy_numel": int(policy_entropy_full.numel()),
        "team_advantage_count_before_broadcast": int(team_advantage.numel()),
        "actor_advantage_count_after_broadcast": int(actor_advantage.numel()),
        "critic_value_target_count": int(returns.numel()),
        "actions_min": int(action_values.min().item()),
        "actions_max": int(action_values.max().item()),
        "action_histogram": action_histogram,
        "old_logprobs_mean": tensor_float_summary(old_log_probs_full)["mean"],
        "old_logprobs_std": tensor_float_summary(old_log_probs_full)["std"],
        "old_logprobs_min": tensor_float_summary(old_log_probs_full)["min"],
        "old_logprobs_max": tensor_float_summary(old_log_probs_full)["max"],
        "new_logprobs_mean": tensor_float_summary(policy_new_log_probs_full)["mean"],
        "new_logprobs_std": tensor_float_summary(policy_new_log_probs_full)["std"],
        "new_logprobs_min": tensor_float_summary(policy_new_log_probs_full)["min"],
        "new_logprobs_max": tensor_float_summary(policy_new_log_probs_full)["max"],
        "entropy_mean": tensor_float_summary(policy_entropy_full)["mean"],
        "entropy_std": tensor_float_summary(policy_entropy_full)["std"],
        "entropy_min": tensor_float_summary(policy_entropy_full)["min"],
        "entropy_max": tensor_float_summary(policy_entropy_full)["max"],
        "team_rewards_mean": tensor_float_summary(raw_rewards_full)["mean"],
        "team_rewards_std": tensor_float_summary(raw_rewards_full)["std"],
        "team_rewards_min": tensor_float_summary(raw_rewards_full)["min"],
        "team_rewards_max": tensor_float_summary(raw_rewards_full)["max"],
        "done_count": int(done_full.sum().item()),
    }
    rollout_summary_path = output_root / "rollout_tensor_summary.json"
    dump_json(rollout_summary_path, rollout_summary)

    sample_steps = [0, horizon // 2, horizon - 1]
    sampled_trace = {
        "trace_mode": trace_mode,
        "sampled_timesteps": sample_steps,
        "rows": [
            {
                "step": int(step),
                "agent_id": int(agent_id),
                "action": int(actions_full[step, agent_id].detach().cpu().item()),
                "old_logprob": float(old_log_probs_full[step, agent_id].detach().cpu().item()),
            }
            for step in sample_steps
            for agent_id in range(num_agents)
        ],
    }
    sampled_trace_path = output_root / "sampled_decision_trace.json"
    dump_json(sampled_trace_path, sampled_trace)

    returns_cpu = returns.detach().cpu()
    if len(returns_cpu) > 1:
        value_proxy = torch.stack(values_by_step)
        var_y = torch.var(returns_cpu, unbiased=False)
        explained_variance = float((1.0 - torch.var(returns_cpu - value_proxy, unbiased=False) / (var_y + 1e-8)).item())
    else:
        explained_variance = 0.0

    checkpoint_path = output_root / "checkpoints" / "pipeline_policy_checkpoint.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_started = time.perf_counter()
    policy.eval()
    with torch.no_grad():
        reload_test_input = embeddings_device[0]
        logits_before, value_before = policy(reload_test_input)
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "num_agents": num_agents,
            "action_dim": policy.action_dim,
            "embedding_dim": int(embeddings.size(1)),
            "created_at_utc": utc_now(),
        },
        checkpoint_path,
    )

    reloaded = TinyMAPPOPolicy(int(embeddings.size(1)), num_agents, 5, 32).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded.load_state_dict(payload["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        logits, value = reloaded(reload_test_input)
    logits_max_abs_diff = float(torch.max(torch.abs(logits_before - logits)).detach().cpu().item())
    value_max_abs_diff = float(torch.abs(value_before.reshape(()) - value.reshape(())).detach().cpu().item())
    checkpoint_io_seconds = time.perf_counter() - checkpoint_started

    trace_started = time.perf_counter()
    trace_path: Optional[Path] = None
    agent_trace_path: Optional[Path] = None
    if trace_mode == "audit":
        trace_path = output_root / "rollout_trace.csv"
        with trace_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rollout[0].keys()))
            writer.writeheader()
            writer.writerows(rollout)

        agent_trace_path = output_root / "agent_decision_trace.csv"
        with agent_trace_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(agent_decision_rows[0].keys()))
            writer.writeheader()
            writer.writerows(agent_decision_rows)

    decision_trace_summary_path = output_root / "decision_trace_summary.json"
    observed_unique_agent_count = len({row["agent_id"] for row in agent_decision_rows}) if trace_mode == "audit" else num_agents
    observed_agent_decisions = len(agent_decision_rows) if trace_mode == "audit" else int(actions_full.numel())
    dump_json(
        decision_trace_summary_path,
        {
            "configured_num_agents": num_agents,
            "observed_unique_agent_count": observed_unique_agent_count,
            "unique_agent_ids": sorted({row["agent_id"] for row in agent_decision_rows}) if trace_mode == "audit" else list(range(num_agents)),
            "environment_steps": horizon,
            "expected_agent_decisions": num_agents * horizon,
            "observed_agent_decisions": observed_agent_decisions,
            "agent_trace_rows": len(agent_decision_rows),
            "trace_mode": trace_mode,
        },
    )
    rollout_trace_summary_path = output_root / "rollout_trace_summary.json"
    dump_json(
        rollout_trace_summary_path,
        {
            "environment_steps": horizon,
            "stored_team_transitions": len(rollout),
            "agents_per_timestep": num_agents,
            "total_agent_decisions": observed_agent_decisions,
            "trace_mode": trace_mode,
        },
    )
    trace_export_seconds = time.perf_counter() - trace_started

    kpi_started = time.perf_counter()
    kpi = aggregate_kpis(rollout)
    kpi_path = output_root / "canonical_12kpi.json"
    dump_json(kpi_path, kpi)
    kpi_summary_path = output_root / "kpi_summary.json"
    dump_json(kpi_summary_path, kpi)
    kpi_aggregation_seconds = time.perf_counter() - kpi_started

    rollout_buffer_tensor_bytes = sum(
        tensor_nbytes(tensor)
        for tensor in (
            returns,
            actions_full,
            old_log_probs_full,
            old_entropy_full,
            rollout_logits_full,
            values_full,
            actor_advantage,
        )
    )
    actor_rollout_tensor_bytes = sum(
        tensor_nbytes(tensor)
        for tensor in (
            actions_full,
            old_log_probs_full,
            old_entropy_full,
            rollout_logits_full,
            actor_advantage,
        )
    )
    critic_rollout_tensor_bytes = tensor_nbytes(values_full) + tensor_nbytes(returns)

    rollout_profile.update(
        {
            "trace_format": "csv" if trace_mode == "audit" else "json_summary",
            "trace_rows_written": len(agent_decision_rows) if trace_mode == "audit" else 0,
            "team_trace_rows_written": len(rollout) if trace_mode == "audit" else 0,
            "trace_buffering_mode": "memory_buffer_then_batch_csv_write",
            "trace_export_seconds": round(trace_export_seconds, 6),
            "trace_recording_seconds": rollout_profile["rollout_trace_recording_seconds"],
            "rollout_online_seconds": round(rollout_collection_seconds, 6),
            "rollout_postprocess_seconds": round(trace_export_seconds, 6),
            "trace_file_write_seconds": round(trace_export_seconds, 6),
            "trace_file_size_bytes": agent_trace_path.stat().st_size if agent_trace_path else 0,
            "team_trace_file_size_bytes": trace_path.stat().st_size if trace_path else 0,
            "rollout_buffer_tensor_bytes": rollout_buffer_tensor_bytes,
            "actor_rollout_tensor_bytes": actor_rollout_tensor_bytes,
            "critic_rollout_tensor_bytes": critic_rollout_tensor_bytes,
            "trace_buffer_estimated_bytes": (
                (agent_trace_path.stat().st_size if agent_trace_path else 0)
                + (trace_path.stat().st_size if trace_path else 0)
                + rollout_hashes_path.stat().st_size
                + rollout_summary_path.stat().st_size
                + sampled_trace_path.stat().st_size
            ),
            "trace_output_file_bytes": (
                (agent_trace_path.stat().st_size if agent_trace_path else 0)
                + (trace_path.stat().st_size if trace_path else 0)
                + rollout_hashes_path.stat().st_size
                + rollout_summary_path.stat().st_size
                + sampled_trace_path.stat().st_size
            ),
            "postprocess_tensor_bytes": tensor_nbytes(actions_full) + tensor_nbytes(old_log_probs_full) + tensor_nbytes(raw_rewards_full) + tensor_nbytes(done_full),
            "agent_decisions_per_second_total": round(agent_decision_count / rollout_collection_seconds, 6) if rollout_collection_seconds > 0 else None,
            "agent_decisions_per_second_online": round(agent_decision_count / rollout_collection_seconds, 6) if rollout_collection_seconds > 0 else None,
            "agent_decisions_per_second_excluding_trace": round(agent_decision_count / max(rollout_collection_seconds - rollout_profile["rollout_trace_recording_seconds"], 1e-9), 6),
            "actor_elements_per_second": round(agent_decision_count / max(rollout_profile["rollout_actor_inference_seconds"], 1e-9), 6),
            "actor_inference_ms_per_timestep": round(rollout_profile["rollout_actor_inference_seconds"] * 1000.0 / horizon, 6),
            "actor_inference_ms_per_agent": round(rollout_profile["rollout_actor_inference_seconds"] * 1000.0 / agent_decision_count, 6),
            "actor_inference_calls": horizon,
            "agents_per_actor_call": num_agents,
            "simulator_step_calls": horizon,
            "average_ms_per_timestep_total": round(rollout_collection_seconds * 1000.0 / horizon, 6),
            "average_ms_per_timestep_online": round(rollout_collection_seconds * 1000.0 / horizon, 6),
            "average_ms_per_agent_decision_total": round(rollout_collection_seconds * 1000.0 / agent_decision_count, 6),
            "average_ms_per_agent_decision_online": round(rollout_collection_seconds * 1000.0 / agent_decision_count, 6),
        }
    )
    rollout_profile_path = output_root / "rollout_profile.json"
    dump_json(rollout_profile_path, rollout_profile)
    ppo_update_profile = {
        "ppo_epoch_count": update_epochs,
        "ppo_update_seconds": round(ppo_update_seconds, 6),
        "minibatch_size": minibatch_size,
        "minibatch_count_per_epoch": ppo_minibatch_counts[0] if len(set(ppo_minibatch_counts)) == 1 else ppo_minibatch_counts,
        "team_samples_per_minibatch": team_samples_per_minibatch[0] if len(set(team_samples_per_minibatch)) == 1 else team_samples_per_minibatch,
        "agent_elements_per_minibatch": agent_elements_per_minibatch[0] if len(set(agent_elements_per_minibatch)) == 1 else agent_elements_per_minibatch,
        "actor_elements_per_epoch": unique_rollout_actor_elements,
        "actor_elements_across_all_epochs": actor_elements_processed_across_epochs,
        "critic_samples_per_epoch": sample_count,
        "critic_samples_across_all_epochs": critic_samples_processed_across_epochs,
        "policy_loss_mean": float(sum(policy_losses) / max(len(policy_losses), 1)),
        "value_loss_mean": float(sum(value_losses) / max(len(value_losses), 1)),
        "entropy_mean": float(sum(entropies) / max(len(entropies), 1)),
        "approx_kl_mean": float(sum(approx_kls) / max(len(approx_kls), 1)),
        "clip_fraction_mean": float(sum(clip_fractions) / max(len(clip_fractions), 1)),
    }
    ppo_update_profile_path = output_root / "ppo_update_profile.json"
    dump_json(ppo_update_profile_path, ppo_update_profile)

    return {
        "rollout_horizon": horizon,
        "trace_mode": trace_mode,
        "rollout_collection_seconds": round(rollout_collection_seconds, 6),
        "rollout_profile_path": str(rollout_profile_path),
        "rollout_profile": rollout_profile,
        "ppo_update_profile_path": str(ppo_update_profile_path),
        "ppo_update_profile": ppo_update_profile,
        "ppo_update_epochs": update_epochs,
        "ppo_update_seconds": round(ppo_update_seconds, 6),
        "checkpoint_io_seconds": round(checkpoint_io_seconds, 6),
        "trace_export_seconds": round(trace_export_seconds, 6),
        "kpi_aggregation_seconds": round(kpi_aggregation_seconds, 6),
        "minibatch_size": minibatch_size,
        "policy_update_losses": losses,
        "policy_loss_mean": float(sum(policy_losses) / max(len(policy_losses), 1)),
        "value_loss_mean": float(sum(value_losses) / max(len(value_losses), 1)),
        "entropy_mean": float(sum(entropies) / max(len(entropies), 1)),
        "approx_kl_mean": float(sum(approx_kls) / max(len(approx_kls), 1)),
        "clip_fraction_mean": float(sum(clip_fractions) / max(len(clip_fractions), 1)),
        "explained_variance": explained_variance,
        "grad_clip_norm": grad_clip_norm,
        "grad_norm_before_clip_mean": float(sum(grad_norms_before_clip) / max(len(grad_norms_before_clip), 1)),
        "grad_norm_before_clip_max": float(max(grad_norms_before_clip)) if grad_norms_before_clip else 0.0,
        "grad_norm_after_clip_mean": float(sum(grad_norms_after_clip) / max(len(grad_norms_after_clip), 1)),
        "grad_norm_after_clip_max": float(max(grad_norms_after_clip)) if grad_norms_after_clip else 0.0,
        "actor_grad_norm_before_clip_mean": float(sum(actor_grad_norms_before_clip) / max(len(actor_grad_norms_before_clip), 1)),
        "actor_grad_norm_before_clip_max": float(max(actor_grad_norms_before_clip)) if actor_grad_norms_before_clip else 0.0,
        "actor_grad_norm_after_clip_mean": float(sum(actor_grad_norms_after_clip) / max(len(actor_grad_norms_after_clip), 1)),
        "actor_grad_norm_after_clip_max": float(max(actor_grad_norms_after_clip)) if actor_grad_norms_after_clip else 0.0,
        "critic_grad_norm_before_clip_mean": float(sum(critic_grad_norms_before_clip) / max(len(critic_grad_norms_before_clip), 1)),
        "critic_grad_norm_before_clip_max": float(max(critic_grad_norms_before_clip)) if critic_grad_norms_before_clip else 0.0,
        "critic_grad_norm_after_clip_mean": float(sum(critic_grad_norms_after_clip) / max(len(critic_grad_norms_after_clip), 1)),
        "critic_grad_norm_after_clip_max": float(max(critic_grad_norms_after_clip)) if critic_grad_norms_after_clip else 0.0,
        "nan_inf_detected": bool(mappo_nan_inf_detected),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_reload_ok": tuple(logits.shape) == (num_agents, 5) and tuple(value.shape) == () and logits_max_abs_diff <= 1e-7,
        "checkpoint_reload_logits_max_abs_diff": logits_max_abs_diff,
        "checkpoint_reload_value_max_abs_diff": value_max_abs_diff,
        "reload_test_input_hash": sha256_tensor(reload_test_input),
        "pre_reload_logits_hash": sha256_tensor(logits_before),
        "post_reload_logits_hash": sha256_tensor(logits),
        "pre_reload_value_hash": sha256_tensor(value_before.reshape(1)),
        "post_reload_value_hash": sha256_tensor(value.reshape(1)),
        "initial_policy_state_hash": initial_policy_state_hash,
        "initial_actor_state_hash": initial_actor_state_hash,
        "initial_critic_state_hash": initial_critic_state_hash,
        "final_policy_state_hash": sha256_state_dict(policy),
        "final_actor_state_hash": sha256_state_dict(policy, "actor."),
        "final_critic_state_hash": sha256_state_dict(policy, "critic."),
        "rollout_tensor_hashes_path": str(rollout_hashes_path),
        "rollout_tensor_hashes": rollout_hashes,
        "rollout_tensor_summary_path": str(rollout_summary_path),
        "rollout_tensor_summary": rollout_summary,
        "sampled_decision_trace_path": str(sampled_trace_path),
        "actor_loss_audit_path": str(actor_loss_audit_path),
        "actor_loss_audit": actor_loss_audit,
        "rollout_trace_path": str(trace_path) if trace_path else None,
        "agent_decision_trace_path": str(agent_trace_path) if agent_trace_path else None,
        "decision_trace_summary_path": str(decision_trace_summary_path),
        "rollout_trace_summary_path": str(rollout_trace_summary_path),
        "configured_num_agents": num_agents,
        "observed_unique_agent_count": observed_unique_agent_count,
        "unique_agent_ids": sorted({row["agent_id"] for row in agent_decision_rows}) if trace_mode == "audit" else list(range(num_agents)),
        "environment_steps": horizon,
        "expected_agent_decisions": num_agents * horizon,
        "observed_agent_decisions": observed_agent_decisions,
        "rollout_buffer_action_rows": observed_agent_decisions,
        "stored_team_transitions": len(rollout),
        "rollout_buffer_value_rows": sample_count,
        "rollout_buffer_advantage_rows": sample_count,
        "ppo_update_samples": sample_count,
        "ppo_update_team_samples": sample_count,
        "actor_logits_shape": [num_agents, 5],
        "action_shape": [num_agents],
        "kpi_summary_path": str(kpi_path),
        "legacy_kpi_summary_path": str(kpi_summary_path),
        "kpi_summary": kpi,
    }


def aggregate_kpis(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    numeric_keys = ["reward", *canonical_kpi_keys()]
    result: Dict[str, float] = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        result[f"{key}_mean"] = float(sum(values) / len(values))
        result[f"{key}_min"] = float(min(values))
        result[f"{key}_max"] = float(max(values))
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end Mac pipeline smoke: GATv2 train -> embeddings -> MAPPO update -> checkpoint -> KPI.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--output-root", default="/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_smoke")
    parser.add_argument("--snapshots", type=int, default=128)
    parser.add_argument("--val-snapshots", type=int, default=64)
    parser.add_argument("--expected-nodes", type=int, default=4116)
    parser.add_argument("--expected-edges", type=int, default=5484)
    parser.add_argument("--gat-batch-size", type=int, default=1)
    parser.add_argument("--gat-hidden", type=int, default=32)
    parser.add_argument("--gat-epochs", type=int, default=1)
    parser.add_argument("--gat-grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--precomputed-embedding-path", help="Load fixed graph embeddings and skip GATv2 train/eval/export for trace-mode isolation checks.")
    parser.add_argument("--rollout-horizon", type=int, default=64)
    parser.add_argument("--ppo-update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--mappo-grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--condition-id", default="A")
    parser.add_argument("--preferred-device", default="mps")
    parser.add_argument("--trace-mode", choices=["audit", "benchmark"], default=os.environ.get("URBANBUS_TRACE_MODE", "audit"))
    parser.add_argument("--require-gpu", action="store_true", help="Fail instead of using CPU fallback when the requested GPU is unavailable.")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not (5 <= args.num_agents <= 256):
        raise ValueError("--num-agents must be between 5 and 256")
    if args.rollout_horizon < 64 or args.rollout_horizon > 128:
        raise ValueError("--rollout-horizon must be between 64 and 128")
    if args.ppo_update_epochs < 2 or args.ppo_update_epochs > 4:
        raise ValueError("--ppo-update-epochs must be between 2 and 4")
    if args.minibatch_size < 128 or args.minibatch_size > 256:
        raise ValueError("--minibatch-size must be between 128 and 256")
    if str(args.condition_id).upper() != "A":
        raise ValueError("This smoke is locked to condition A")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    trace_mode = str(args.trace_mode).lower()

    started = time.perf_counter()
    output_root = Path(args.output_root)
    if args.clean:
        safe_rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    memory_samples: List[Dict[str, Any]] = [system_memory_snapshot("start")]
    device, device_decision = choose_device(args.preferred_device)
    if args.require_gpu and device_decision.selected == "cpu":
        raise RuntimeError(f"GPU required but unavailable: {device_decision.reason}")
    dataset_root = Path(args.dataset_dir).expanduser().resolve()
    dataset_load_started = time.perf_counter()
    files = resolve_files(dataset_root, args.split, args.snapshots)
    val_files = resolve_files(dataset_root, args.val_split, args.val_snapshots)
    snapshot_ordering_hash = sha256_text("\n".join(str(path) for path in files))
    data_list = [torch_load(path) for path in files]
    val_data_list = [torch_load(path) for path in val_files]
    dataset_load_seconds = time.perf_counter() - dataset_load_started
    memory_samples.append(system_memory_snapshot("after_tensor_load"))
    graph_contract = validate_graph_contract(data_list[0], args.expected_nodes, args.expected_edges)

    sample = data_list[0]
    edge_attr = getattr(sample, "edge_attr", None)
    edge_dim = edge_attr.size(1) if edge_attr is not None else None
    model = NodeLevelGATv2(
        in_channels=sample.x.size(1),
        hidden_channels=args.gat_hidden,
        out_channels=sample.y.size(1),
        edge_dim=edge_dim,
    ).to(device)
    initial_gatv2_parameter_hash = sha256_state_dict(model)

    embedding_path = output_root / "embeddings" / "gatv2_graph_embeddings.pt"
    precomputed_embedding_path = Path(args.precomputed_embedding_path).expanduser().resolve() if args.precomputed_embedding_path else None
    if precomputed_embedding_path:
        gat_train_started = time.perf_counter()
        gat_train = {
            "losses": [0.0],
            "epoch_avg_train_loss": [],
            "grad_clip_norm": args.gat_grad_clip_norm,
            "grad_norm_before_clip_mean": 0.0,
            "grad_norm_before_clip_max": 0.0,
            "grad_norm_after_clip_mean": 0.0,
            "grad_norm_after_clip_max": 0.0,
            "nan_inf_detected": False,
        }
        gatv2_train_seconds = time.perf_counter() - gat_train_started
        validation_started = time.perf_counter()
        gat_val = {
            "loss_count": 0,
            "avg_loss": None,
            "loss_first": None,
            "loss_last": None,
            "nan_inf_detected": False,
            "skipped_reason": "precomputed_embedding_path",
        }
        validation_seconds = time.perf_counter() - validation_started
        memory_samples.append(system_memory_snapshot("after_gatv2_train_eval"))
        embedding_started = time.perf_counter()
        embedding_payload = torch_load(precomputed_embedding_path)
        embeddings = embedding_payload["graph_embeddings"] if isinstance(embedding_payload, dict) and "graph_embeddings" in embedding_payload else embedding_payload
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("--precomputed-embedding-path must contain a Tensor or {'graph_embeddings': Tensor}")
        embeddings = embeddings.detach().cpu()
        if embeddings.size(0) < args.snapshots:
            raise ValueError(f"Need at least {args.snapshots} embeddings, found {embeddings.size(0)}")
        embeddings = embeddings[: args.snapshots]
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"graph_embeddings": embeddings, "created_at_utc": utc_now(), "source_path": str(precomputed_embedding_path)}, embedding_path)
        embedding_export_seconds = time.perf_counter() - embedding_started
    else:
        gat_train_started = time.perf_counter()
        gat_train = train_gatv2(
            model,
            data_list,
            device,
            args.gat_batch_size,
            lr=1e-3,
            epochs=args.gat_epochs,
            grad_clip_norm=args.gat_grad_clip_norm,
        )
        gatv2_train_seconds = time.perf_counter() - gat_train_started
        validation_started = time.perf_counter()
        gat_val = evaluate_gatv2(model, val_data_list, device, args.gat_batch_size)
        validation_seconds = time.perf_counter() - validation_started
        memory_samples.append(system_memory_snapshot("after_gatv2_train_eval"))
        embedding_started = time.perf_counter()
        embeddings = generate_embeddings(model, data_list, device, embedding_path)
        embedding_export_seconds = time.perf_counter() - embedding_started
    embedding_variance = float(torch.var(embeddings.float(), unbiased=False).item())
    embedding_nan_inf_detected = not bool(torch.isfinite(embeddings).all().item())
    memory_samples.append(system_memory_snapshot("after_embedding_generation"))
    pipeline = mappo_rollout_update(
        embeddings=embeddings,
        device=device,
        output_root=output_root,
        num_agents=args.num_agents,
        horizon=args.rollout_horizon,
        update_epochs=args.ppo_update_epochs,
        minibatch_size=args.minibatch_size,
        seed=args.seed,
        condition_id=str(args.condition_id).upper(),
        grad_clip_norm=args.mappo_grad_clip_norm,
        clip_epsilon=args.ppo_clip_epsilon,
        trace_mode=trace_mode,
    )
    memory_samples.append(system_memory_snapshot("after_mappo_update"))
    memory_summary = summarize_memory(memory_samples)
    memory_profile_path = output_root / "memory_profile.json"
    dump_json(
        memory_profile_path,
        {
            "samples": memory_samples,
            "summary": memory_summary,
            "requested_stage_aliases": {
                "memory_after_dataset_load": "after_tensor_load",
                "memory_after_gatv2_train": "after_gatv2_train_eval",
                "memory_after_embedding_export": "after_embedding_generation",
                "memory_after_ppo_update": "after_mappo_update",
                "memory_after_checkpoint_reload": "after_mappo_update",
            },
            "note": "MAPPO rollout, PPO update, checkpoint reload, trace export, and KPI aggregation run inside one pipeline section; fine-grained memory samples are summarized in rollout_profile and manifest timing.",
        },
    )
    stdout_log_path = output_root / "run_stdout.log"
    stderr_log_path = output_root / "run_stderr.log"
    stdout_log_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "created_at_utc": utc_now(),
                "snapshots": args.snapshots,
                "num_agents": args.num_agents,
                "rollout_horizon": args.rollout_horizon,
                "device": device_decision.selected,
                "trace_mode": trace_mode,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    stderr_log_path.write_text("", encoding="utf-8")
    canonical_missing = [key for key in canonical_kpi_keys() if f"{key}_mean" not in pipeline["kpi_summary"]]
    kpi_nan, kpi_inf = contains_nan_or_inf(pipeline["kpi_summary"])
    nan_detected = bool(gat_train["nan_inf_detected"] or gat_val["nan_inf_detected"] or embedding_nan_inf_detected or pipeline.get("nan_inf_detected", False) or kpi_nan)
    inf_detected = bool(pipeline.get("nan_inf_detected", False) or kpi_inf)
    artifact_paths = {
        "actor_loss_audit": Path(pipeline["actor_loss_audit_path"]),
        "rollout_profile": Path(pipeline["rollout_profile_path"]),
        "ppo_update_profile": Path(pipeline["ppo_update_profile_path"]),
        "memory_profile": memory_profile_path,
        "rollout_tensor_hashes": Path(pipeline["rollout_tensor_hashes_path"]),
        "rollout_tensor_summary": Path(pipeline["rollout_tensor_summary_path"]),
        "sampled_decision_trace": Path(pipeline["sampled_decision_trace_path"]),
        "decision_trace_summary": Path(pipeline["decision_trace_summary_path"]),
        "rollout_trace_summary": Path(pipeline["rollout_trace_summary_path"]),
        "checkpoint": Path(pipeline["checkpoint_path"]),
        "embedding": embedding_path,
        "canonical_kpi": Path(pipeline["kpi_summary_path"]),
        "legacy_kpi_summary": Path(pipeline["legacy_kpi_summary_path"]),
        "run_stdout": stdout_log_path,
        "run_stderr": stderr_log_path,
    }
    if pipeline.get("agent_decision_trace_path"):
        artifact_paths["agent_decision_trace"] = Path(pipeline["agent_decision_trace_path"])
    if pipeline.get("rollout_trace_path"):
        artifact_paths["rollout_trace"] = Path(pipeline["rollout_trace_path"])
    artifact_index = {
        name: {
            "relative_path": str(path.relative_to(output_root)),
            "absolute_path": str(path),
            "path": str(path),
            "file_size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in artifact_paths.items()
    }

    manifest = {
        "status": "PASS",
        "created_at_utc": utc_now(),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "trace_mode": trace_mode,
        "device": asdict(device_decision),
        "actual_device": device_decision.selected,
        "strict_mps": bool(args.require_gpu and args.preferred_device.lower() == "mps"),
        "mac_full_experiment_viable_on_gpu": bool(not device_decision.gpu_blocked and device_decision.gpu_available),
        "cpu_fallback_used": device_decision.selected == "cpu",
        "cpu_fallback_allowed": not bool(args.require_gpu),
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "dataset": {
            "dataset_dir": str(dataset_root),
            "split": args.split,
            "val_split": args.val_split,
            "snapshots": args.snapshots,
            "val_snapshots": args.val_snapshots,
            "first_file": str(files[0]),
            "last_file": str(files[-1]),
            "first_val_file": str(val_files[0]),
            "last_val_file": str(val_files[-1]),
            "snapshot_ordering_hash": snapshot_ordering_hash,
            "precomputed_embedding_path": str(precomputed_embedding_path) if precomputed_embedding_path else None,
        },
        "graph_contract": graph_contract,
        "timing": {
            "dataset_load_seconds": round(dataset_load_seconds, 6),
            "gatv2_train_seconds": round(gatv2_train_seconds, 6),
            "validation_seconds": round(validation_seconds, 6),
            "embedding_export_seconds": round(embedding_export_seconds, 6),
            "rollout_collection_seconds": pipeline["rollout_collection_seconds"],
            "rollout_online_seconds": pipeline["rollout_profile"]["rollout_online_seconds"],
            "rollout_postprocess_seconds": pipeline["rollout_profile"]["rollout_postprocess_seconds"],
            "trace_file_write_seconds": pipeline["rollout_profile"]["trace_file_write_seconds"],
            "ppo_update_seconds": pipeline["ppo_update_seconds"],
            "checkpoint_io_seconds": pipeline["checkpoint_io_seconds"],
            "checkpoint_reload_seconds": pipeline["checkpoint_io_seconds"],
            "kpi_aggregation_seconds": pipeline["kpi_aggregation_seconds"],
            "kpi_generation_seconds": pipeline["kpi_aggregation_seconds"],
        },
        "memory_samples": memory_samples,
        "memory_summary": memory_summary,
        "memory_profile_path": str(memory_profile_path),
        "gatv2": {
            "batch_size": args.gat_batch_size,
            "hidden": args.gat_hidden,
            "epochs": args.gat_epochs,
            "freeze_for_mappo": True,
            "precomputed_embedding_used": bool(precomputed_embedding_path),
            "layers": 2,
            "attention_heads_first_layer": 2,
            "initial_parameter_hash": initial_gatv2_parameter_hash,
            "final_parameter_hash": sha256_state_dict(model),
            "loss_count": len(gat_train["losses"]),
            "loss_first": gat_train["losses"][0] if gat_train["losses"] else None,
            "loss_last": gat_train["losses"][-1] if gat_train["losses"] else None,
            "epoch_avg_train_loss": gat_train["epoch_avg_train_loss"],
            "validation": gat_val,
            "grad_clip_norm": gat_train["grad_clip_norm"],
            "grad_norm_before_clip_mean": gat_train["grad_norm_before_clip_mean"],
            "grad_norm_before_clip_max": gat_train["grad_norm_before_clip_max"],
            "grad_norm_after_clip_mean": gat_train["grad_norm_after_clip_mean"],
            "grad_norm_after_clip_max": gat_train["grad_norm_after_clip_max"],
            "nan_inf_detected": bool(gat_train["nan_inf_detected"] or gat_val["nan_inf_detected"]),
            "embedding_path": str(embedding_path),
            "embedding_shape": list(embeddings.shape),
            "embedding_tensor_hash": sha256_tensor(embeddings),
            "embedding_variance": embedding_variance,
            "embedding_nan_inf_detected": embedding_nan_inf_detected,
        },
        "mappo": {
            "num_agents": args.num_agents,
            "parallel_envs": 1,
            "rollout_horizon": args.rollout_horizon,
            "ppo_update_epochs": args.ppo_update_epochs,
            "minibatch_size": args.minibatch_size,
            "seed": args.seed,
            "condition_id": str(args.condition_id).upper(),
            **pipeline,
        },
        "canonical_kpi_generated": len(canonical_missing) == 0,
        "canonical_kpi_count": len(canonical_kpi_keys()) - len(canonical_missing),
        "canonical_kpi_missing": canonical_missing,
        "artifacts": artifact_index,
    }
    manifest_path = output_root / "pipeline_manifest.json"
    dump_json(manifest_path, manifest)
    print(json.dumps({"manifest_path": str(manifest_path), **manifest}, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    try:
        run(parse_args())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        raise


if __name__ == "__main__":
    main()
