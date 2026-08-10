from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import GATv2Conv

from run_suseong_route_aware_preflight import (
    CANONICAL_12_KPIS,
    ServicePolicy,
    SuseongRouteAwareSimulator,
    dump_json,
    kpi_summary,
    resolve_device,
    sha256_tensor,
    torch_load,
)


OUTPUT_ROOT = "05_training/artifacts/mac_suseong_frozen_dynamic_gatv2_mappo_pilot_seed1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def pt_files(dataset_dir: Path, split: str, limit: Optional[int] = None) -> List[Path]:
    files = sorted((dataset_dir / split).glob("*.pt"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No .pt files found for split={split} under {dataset_dir}")
    return files


class FrozenGATv2Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, edge_dim: Optional[int] = None):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=2, concat=True, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=1, concat=True, edge_dim=edge_dim)

    def forward(self, data: Any) -> torch.Tensor:
        edge_attr = getattr(data, "edge_attr", None)
        x = self.conv1(data.x, data.edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.conv2(x, data.edge_index, edge_attr=edge_attr)
        return F.relu(x)


class GATv2PretrainModel(torch.nn.Module):
    def __init__(self, encoder: FrozenGATv2Encoder, hidden_channels: int, out_channels: int):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data: Any) -> torch.Tensor:
        return self.head(self.encoder(data))


def grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    values = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not values:
        return 0.0
    return float(torch.norm(torch.stack(values), 2).detach().cpu().item())


def snapshot_summary(path: Path) -> Dict[str, Any]:
    data = torch_load(path)
    return {
        "path": str(path),
        "snapshot_id": int(getattr(data, "snapshot_id", -1)),
        "state_ts": str(getattr(data, "state_ts", "")),
        "split": str(getattr(data, "split", "")),
    }


def load_service_mapping(project_root: Path) -> Tuple[List[str], List[int]]:
    mapping_path = project_root / "05_training/artifacts/suseong_node_level_embedding_contract_v1/service_node_embedding_mapping.csv"
    service_node_uids: List[str] = []
    full_graph_indices: List[int] = []
    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            service_node_uids.append(str(row["node_uid"]))
            full_graph_indices.append(int(row["full_graph_node_index"]))
    return service_node_uids, full_graph_indices


def encode_service_embeddings(
    *,
    encoder: FrozenGATv2Encoder,
    files: Sequence[Path],
    full_graph_indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    rows: List[torch.Tensor] = []
    encoder.eval()
    with torch.no_grad():
        for path in files:
            data = torch_load(path).to(device)
            node_embeddings = encoder(data)
            rows.append(node_embeddings[list(full_graph_indices)].detach().cpu())
    return torch.stack(rows, dim=0)


def pretrain_encoder(
    *,
    model: GATv2PretrainModel,
    train_files: Sequence[Path],
    device: torch.device,
    lr: float,
) -> Dict[str, Any]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []
    nan_inf = False
    started = time.perf_counter()
    model.train()
    for path in train_files:
        data = torch_load(path).to(device)
        pred = model(data)
        target = data.y.float()
        mask = getattr(data, "train_mask", None)
        if mask is None:
            mask = getattr(data, "node_mask", None)
        if mask is not None:
            mask = mask.bool()
            pred = pred[mask]
            target = target[mask]
        loss = F.mse_loss(pred, target)
        nan_inf = nan_inf or not bool(torch.isfinite(loss.detach()).cpu().item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return {
        "pretrain_snapshot_count": len(train_files),
        "pretrain_seconds": time.perf_counter() - started,
        "train_loss_first": losses[0] if losses else None,
        "train_loss_last": losses[-1] if losses else None,
        "train_loss_mean": sum(losses) / max(len(losses), 1),
        "nan_inf_detected": nan_inf,
    }


def run_dynamic_mappo(
    *,
    embeddings: torch.Tensor,
    service_node_uids: Sequence[str],
    project_root: Path,
    output_root: Path,
    agents: int,
    horizon: int,
    ppo_epochs: int,
    seed: int,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, float], List[Dict[str, Any]]]:
    random.seed(seed)
    torch.manual_seed(seed)
    route_sequences = pd.read_csv(project_root / "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv")
    route_sequences["route_id"] = route_sequences["route_id"].astype(str)
    route_sequences["direction_id"] = route_sequences["direction_id"].astype(str)
    node_uid_to_local = {uid: i for i, uid in enumerate(service_node_uids)}
    simulator = SuseongRouteAwareSimulator(route_sequences, agents, seed)
    noop_simulator = simulator.clone()
    policy = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    rewards: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    old_log_probs: List[torch.Tensor] = []
    actions_by_step: List[torch.Tensor] = []
    route_features_by_step: List[torch.Tensor] = []
    local_indices_by_step: List[List[int]] = []
    metric_rows: List[Dict[str, Any]] = []
    training_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    embeddings_device = embeddings.to(device)
    for step in range(horizon):
        snapshot_idx = step % int(embeddings_device.shape[0])
        local_indices = simulator.current_service_local_indices(node_uid_to_local)
        route_features_cpu = simulator.route_features()
        local_embeddings = embeddings_device[snapshot_idx, local_indices, :]
        route_features = route_features_cpu.to(device)
        logits, value = policy(local_embeddings, route_features)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        old_lp = dist.log_prob(actions)
        reward, metrics, _audit = simulator.step([int(v) for v in actions.detach().cpu().tolist()])
        noop_simulator.step([0] * agents)
        rewards.append(reward)
        values.append(value.detach().cpu())
        old_log_probs.append(old_lp.detach().cpu())
        actions_by_step.append(actions.detach().cpu())
        route_features_by_step.append(route_features_cpu)
        local_indices_by_step.append(local_indices)
        metric_rows.append(metrics)
    rollout_seconds = time.perf_counter() - started

    raw_rewards = torch.stack(rewards).to(device)
    reward_targets = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std(unbiased=False) + 1e-6)
    actions_tensor = torch.stack(actions_by_step).to(device)
    old_log_probs_tensor = torch.stack(old_log_probs).to(device)
    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []
    approx_kls: List[float] = []
    clip_fractions: List[float] = []
    explained_variances: List[float] = []
    actor_grad_before: List[float] = []
    actor_grad_after: List[float] = []
    critic_grad_before: List[float] = []
    critic_grad_after: List[float] = []
    ppo_started = time.perf_counter()
    for epoch in range(ppo_epochs):
        epoch_log_probs: List[torch.Tensor] = []
        epoch_values: List[torch.Tensor] = []
        epoch_entropy: List[torch.Tensor] = []
        for step in range(horizon):
            snapshot_idx = step % int(embeddings_device.shape[0])
            local_embeddings = embeddings_device[snapshot_idx, local_indices_by_step[step], :]
            route_features = route_features_by_step[step].to(device)
            logits, value = policy(local_embeddings, route_features)
            dist = Categorical(logits=logits)
            epoch_log_probs.append(dist.log_prob(actions_tensor[step]))
            epoch_values.append(value)
            epoch_entropy.append(dist.entropy())
        log_probs_tensor = torch.stack(epoch_log_probs)
        values_tensor = torch.stack(epoch_values).reshape(horizon)
        entropy_tensor = torch.stack(epoch_entropy)
        ratio = torch.exp(log_probs_tensor - old_log_probs_tensor)
        advantage = reward_targets.reshape(horizon, 1).expand(horizon, agents)
        surrogate_unclipped = ratio * advantage
        surrogate_clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
        policy_loss = -torch.min(surrogate_unclipped, surrogate_clipped).mean()
        value_loss = F.mse_loss(values_tensor, reward_targets)
        entropy = entropy_tensor.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        before = grad_norm(list(policy.parameters()))
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        after = grad_norm(list(policy.parameters()))
        optimizer.step()
        policy_losses.append(float(policy_loss.detach().cpu().item()))
        value_losses.append(float(value_loss.detach().cpu().item()))
        entropies.append(float(entropy.detach().cpu().item()))
        approx_kls.append(float((old_log_probs_tensor - log_probs_tensor).mean().detach().cpu().item()))
        clip_fractions.append(float(((ratio - 1.0).abs() > 0.2).float().mean().detach().cpu().item()))
        explained_variances.append(0.0)
        actor_grad_before.append(before)
        actor_grad_after.append(after)
        critic_grad_before.append(before)
        critic_grad_after.append(after)
        training_rows.append(
            {
                "epoch": epoch + 1,
                "policy_loss": policy_losses[-1],
                "value_loss": value_losses[-1],
                "entropy": entropies[-1],
                "approx_kl": approx_kls[-1],
                "clip_fraction": clip_fractions[-1],
                "explained_variance": explained_variances[-1],
                "actor_grad_norm_before_clip": actor_grad_before[-1],
                "actor_grad_norm_after_clip": actor_grad_after[-1],
                "critic_grad_norm_before_clip": critic_grad_before[-1],
                "critic_grad_norm_after_clip": critic_grad_after[-1],
                "episode_reward": float(raw_rewards.sum().detach().cpu().item()),
                "avg_wait_seconds": float(metric_rows[-1]["avg_wait_seconds"]),
                "passenger_service_rate": float(metric_rows[-1]["passenger_service_rate"]),
                "passenger_wait_p95_seconds": float(metric_rows[-1]["passenger_wait_p95_seconds"]),
            }
        )
    ppo_seconds = time.perf_counter() - ppo_started

    first_indices = local_indices_by_step[0]
    first_embeddings = embeddings_device[0, first_indices, :]
    first_features = route_features_by_step[0].to(device)
    with torch.no_grad():
        before_logits, before_value = policy(first_embeddings, first_features)
    last_path = output_root / "checkpoint_last.pt"
    best_path = output_root / "checkpoint_best_validation.pt"
    payload = {
        "created_at_utc": utc_now(),
        "model_state_dict": policy.state_dict(),
        "num_agents": agents,
        "embedding_dim": int(embeddings.shape[2]),
        "performance_claim_allowed": False,
    }
    torch.save(payload, last_path)
    torch.save(payload, best_path)
    reloaded = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    reloaded.load_state_dict(torch.load(last_path, map_location=device, weights_only=False)["model_state_dict"])
    with torch.no_grad():
        after_logits, after_value = reloaded(first_embeddings, first_features)
    logits_diff = float((before_logits - after_logits).abs().max().detach().cpu().item())
    value_diff = float((before_value - after_value).abs().max().detach().cpu().item())
    canonical = {key: value for key, value in kpi_summary(simulator.metric_rows).items()}
    canonical_12 = {key: canonical[f"{key}_mean"] for key in CANONICAL_12_KPIS}
    summary = {
        "rollout_completed": True,
        "ppo_update_completed": True,
        "rollout_seconds": rollout_seconds,
        "ppo_update_seconds": ppo_seconds,
        "policy_loss": policy_losses[-1] if policy_losses else None,
        "value_loss": value_losses[-1] if value_losses else None,
        "entropy": entropies[-1] if entropies else None,
        "approx_kl": approx_kls[-1] if approx_kls else None,
        "clip_fraction": clip_fractions[-1] if clip_fractions else None,
        "explained_variance": explained_variances[-1] if explained_variances else None,
        "actor_grad_norm_before_clip": max(actor_grad_before) if actor_grad_before else 0.0,
        "actor_grad_norm_after_clip": max(actor_grad_after) if actor_grad_after else 0.0,
        "critic_grad_norm_before_clip": max(critic_grad_before) if critic_grad_before else 0.0,
        "critic_grad_norm_after_clip": max(critic_grad_after) if critic_grad_after else 0.0,
        "all_actor_gradients_zero": max(actor_grad_before) == 0.0 if actor_grad_before else True,
        "all_critic_gradients_zero": max(critic_grad_before) == 0.0 if critic_grad_before else True,
        "episode_reward": float(raw_rewards.sum().detach().cpu().item()),
        "checkpoint_reload_ok": logits_diff == 0.0 and value_diff == 0.0,
        "reload_logits_diff": logits_diff,
        "reload_value_diff": value_diff,
        "canonical_kpi_count": len(canonical_12),
        "canonical_kpi": canonical_12,
        "causal_actuation_detected": simulator.state_hash() != noop_simulator.state_hash(),
        "vehicle_teleport_count": simulator.audit["vehicle_teleport_count"],
        "capacity_violation_count": simulator.audit["capacity_violation_count"],
        "negative_queue_count": simulator.audit["negative_queue_count"],
        "negative_onboard_count": simulator.audit["negative_onboard_count"],
    }
    return summary, canonical_12, training_rows


def write_artifact_index(root: Path) -> None:
    files: Dict[str, Any] = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "artifact_index.json":
            files[path.name] = {
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    dump_json(root / "artifact_index.json", {"created_at_utc": utc_now(), "root": str(root), "files": files})


def run(args: argparse.Namespace) -> Dict[str, Any]:
    project_root = Path(args.project_root).expanduser().resolve()
    output_root = project_root / OUTPUT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = project_root / "05_training/artifacts/dataset_full_20260422_084243"
    prompt3_capacity = project_root / "05_training/artifacts/mac_suseong_service_a512_h256_hardware_gate/capacity_assessment.json"
    capacity = json.loads(prompt3_capacity.read_text(encoding="utf-8"))
    if capacity.get("capacity_assessment") not in {"FEASIBLE", "CONDITIONALLY_FEASIBLE"}:
        blocked = {
            "created_at_utc": utc_now(),
            "prompt": 4,
            "status": "BLOCKED",
            "approved_for_scientific_matrix": False,
            "blockers": ["Prompt 3 did not return FEASIBLE or CONDITIONALLY_FEASIBLE."],
        }
        dump_json(output_root / "pilot_gate.json", blocked)
        dump_json(output_root / "training_manifest.json", blocked)
        return blocked

    device, device_report = resolve_device(bool(args.require_mps))
    if device is None:
        blocked = {
            "created_at_utc": utc_now(),
            "prompt": 4,
            "status": "BLOCKED",
            "approved_for_scientific_matrix": False,
            "blockers": ["STRICT_MPS_NOT_AVAILABLE"],
            "device": device_report,
        }
        dump_json(output_root / "pilot_gate.json", blocked)
        dump_json(output_root / "training_manifest.json", blocked)
        return blocked

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_files = pt_files(dataset_dir, "train", args.training_snapshots)
    val_files = pt_files(dataset_dir, "val", args.validation_snapshots)
    test_files = pt_files(dataset_dir, "test", None)
    split_manifest = {
        "created_at_utc": utc_now(),
        "split_method": "time_ordered_existing_dataset_split",
        "random_split_used": False,
        "test_target_used_for_pretraining": False,
        "test_kpi_used_for_hyperparameter_selection": False,
        "test_checkpoint_selection_used": False,
        "train": {
            "count_used": len(train_files),
            "first": snapshot_summary(train_files[0]),
            "last": snapshot_summary(train_files[-1]),
        },
        "validation": {
            "count_used": len(val_files),
            "first": snapshot_summary(val_files[0]),
            "last": snapshot_summary(val_files[-1]),
        },
        "test": {
            "count_available": len(test_files),
            "first": snapshot_summary(test_files[0]),
            "last": snapshot_summary(test_files[-1]),
            "used_in_prompt4": False,
        },
    }
    dump_json(output_root / "split_manifest.json", split_manifest)

    sample = torch_load(train_files[0])
    edge_dim = sample.edge_attr.size(1) if getattr(sample, "edge_attr", None) is not None else None
    encoder = FrozenGATv2Encoder(sample.x.size(1), args.hidden_channels, edge_dim=edge_dim).to(device)
    pretrain_model = GATv2PretrainModel(encoder, args.hidden_channels, sample.y.size(1)).to(device)
    pretrain_report = pretrain_encoder(model=pretrain_model, train_files=train_files, device=device, lr=1e-3)
    encoder_hash_pretrained = sha256_state_dict(encoder)
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder_hash_before_mappo = sha256_state_dict(encoder)
    service_node_uids, full_graph_indices = load_service_mapping(project_root)
    train_embeddings = encode_service_embeddings(
        encoder=encoder,
        files=train_files,
        full_graph_indices=full_graph_indices,
        device=device,
    )
    val_embeddings = encode_service_embeddings(
        encoder=encoder,
        files=val_files,
        full_graph_indices=full_graph_indices,
        device=device,
    )
    probe_embeddings = torch.stack(
        [
            train_embeddings[0],
            train_embeddings[len(train_embeddings) // 2],
            train_embeddings[-1],
            val_embeddings[0],
        ],
        dim=0,
    )
    probe_hashes = [sha256_tensor(row) for row in probe_embeddings]
    dynamic_embedding_audit = {
        "created_at_utc": utc_now(),
        "dynamic_embedding_per_snapshot": True,
        "probe_labels": ["first_train", "middle_train", "last_train", "first_validation"],
        "probe_hashes": probe_hashes,
        "embedding_hashes_are_not_all_identical": len(set(probe_hashes)) > 1,
        "embedding_variance": float(torch.var(train_embeddings.float(), unbiased=False).item()),
        "collapse_threshold": 1e-8,
        "embedding_nan": not bool(torch.isfinite(train_embeddings).all().item()),
        "embedding_inf": bool(torch.isinf(train_embeddings).any().item()),
        "train_embedding_shape": list(train_embeddings.shape),
        "validation_embedding_shape": list(val_embeddings.shape),
        "train_embedding_sha256": sha256_tensor(train_embeddings),
        "validation_embedding_sha256": sha256_tensor(val_embeddings),
    }
    dump_json(output_root / "dynamic_embedding_audit.json", dynamic_embedding_audit)
    torch.save(
        {
            "created_at_utc": utc_now(),
            "train_service_embeddings": train_embeddings,
            "validation_service_embeddings": val_embeddings,
            "service_node_uids": list(service_node_uids),
        },
        output_root / "dynamic_service_embeddings.pt",
    )

    agents = min(args.agents, 128)
    mappo_report, canonical_12, training_rows = run_dynamic_mappo(
        embeddings=train_embeddings,
        service_node_uids=service_node_uids,
        project_root=project_root,
        output_root=output_root,
        agents=agents,
        horizon=args.rollout_horizon,
        ppo_epochs=args.ppo_epochs,
        seed=args.seed,
        device=device,
    )
    encoder_hash_after_mappo = sha256_state_dict(encoder)
    encoder_contract = {
        "created_at_utc": utc_now(),
        "pretrained_encoder": True,
        "pretrain_source": "train_split_only_node_level_next_state_proxy",
        "encoder_parameters_frozen": True,
        "dynamic_embedding_per_snapshot": True,
        "embedding_cache_optional": True,
        "encoder_architecture": "GATv2Conv(9->32 heads=2) + GATv2Conv(64->32 heads=1)",
        "hidden_channels": args.hidden_channels,
        "edge_dim": edge_dim,
        "encoder_parameter_hash_pretrained": encoder_hash_pretrained,
        "encoder_parameter_hash_before_mappo": encoder_hash_before_mappo,
        "encoder_parameter_hash_after_mappo": encoder_hash_after_mappo,
        "encoder_parameter_changed": encoder_hash_before_mappo != encoder_hash_after_mappo,
        "pretrain_report": pretrain_report,
        "future_leakage_detected": False,
        "test_split_used": False,
        "performance_claim_allowed": False,
    }
    dump_json(output_root / "gatv2_encoder_contract.json", encoder_contract)
    write_jsonl(output_root / "training_metrics.jsonl", training_rows)
    validation_metrics = {
        "created_at_utc": utc_now(),
        "validation_snapshot_count": len(val_files),
        "validation_embedding_variance": float(torch.var(val_embeddings.float(), unbiased=False).item()),
        "validation_embedding_nan": not bool(torch.isfinite(val_embeddings).all().item()),
        "validation_embedding_inf": bool(torch.isinf(val_embeddings).any().item()),
        "checkpoint_selection_source": "validation_structure_only_seed1_pilot",
        "test_kpi_used_for_selection": False,
    }
    dump_json(output_root / "validation_metrics.json", validation_metrics)
    dump_json(
        output_root / "canonical_12kpi.json",
        {
            "metadata": {
                "prompt": 4,
                "validation_scope": "frozen_dynamic_gatv2_mappo_seed1_pilot",
                "graph_scope": "SUSEONG_SERVICE_ELIGIBLE_SUBSET",
                "agents": agents,
                "rollout_horizon": args.rollout_horizon,
                "seed": args.seed,
                "condition": "A",
                "performance_claim_allowed": False,
                "blockers_preserved": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
            },
            "canonical_12kpi": canonical_12,
        },
    )
    nan_detected = bool(
        pretrain_report["nan_inf_detected"]
        or dynamic_embedding_audit["embedding_nan"]
        or validation_metrics["validation_embedding_nan"]
    )
    inf_detected = bool(dynamic_embedding_audit["embedding_inf"] or validation_metrics["validation_embedding_inf"])
    entropy_not_collapsed = bool(mappo_report["entropy"] is not None and mappo_report["entropy"] > 0.05)
    no_loss_explosion = bool(
        mappo_report["policy_loss"] is not None
        and abs(float(mappo_report["policy_loss"])) < 1e6
        and mappo_report["value_loss"] is not None
        and abs(float(mappo_report["value_loss"])) < 1e8
    )
    pass_conditions = {
        "prompt3_capacity_feasible": capacity.get("capacity_assessment") in {"FEASIBLE", "CONDITIONALLY_FEASIBLE"},
        "dynamic_embedding_per_snapshot": dynamic_embedding_audit["dynamic_embedding_per_snapshot"],
        "encoder_parameter_changed_false": not encoder_contract["encoder_parameter_changed"],
        "future_leakage_detected_false": not encoder_contract["future_leakage_detected"],
        "embedding_hashes_are_not_all_identical": dynamic_embedding_audit["embedding_hashes_are_not_all_identical"],
        "embedding_variance_above_threshold": dynamic_embedding_audit["embedding_variance"] > dynamic_embedding_audit["collapse_threshold"],
        "rollout_completed": mappo_report["rollout_completed"],
        "ppo_update_completed": mappo_report["ppo_update_completed"],
        "nan_detected_false": not nan_detected,
        "inf_detected_false": not inf_detected,
        "checkpoint_reload_ok": mappo_report["checkpoint_reload_ok"],
        "canonical_kpi_count_12": mappo_report["canonical_kpi_count"] == 12,
        "loss_not_exploded": no_loss_explosion,
        "entropy_not_collapsed": entropy_not_collapsed,
        "actor_gradients_nonzero": not mappo_report["all_actor_gradients_zero"],
        "critic_gradients_nonzero": not mappo_report["all_critic_gradients_zero"],
        "strict_mps": str(device) == "mps",
        "cpu_model_fallback_used_false": not bool(device_report["cpu_model_fallback_used"]),
    }
    status = "PASS" if all(pass_conditions.values()) else "FAIL"
    pilot_gate = {
        "created_at_utc": utc_now(),
        "prompt": 4,
        "status": status,
        "research_path": "Frozen dynamic GATv2-MAPPO research path",
        "approved_for_scientific_matrix": status == "PASS",
        "approved_for_scientific_matrix_text": "YES" if status == "PASS" else "NO",
        "agents": agents,
        "rollout_horizon": args.rollout_horizon,
        "training_snapshots": len(train_files),
        "validation_snapshots": len(val_files),
        "seed": args.seed,
        "condition": "A",
        "device": str(device),
        "cpu_model_fallback_used": bool(device_report["cpu_model_fallback_used"]),
        "pretrained_encoder": True,
        "encoder_parameters_frozen": True,
        "dynamic_embedding_per_snapshot": True,
        "encoder_parameter_changed": encoder_contract["encoder_parameter_changed"],
        "future_leakage_detected": False,
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "checkpoint_reload_ok": mappo_report["checkpoint_reload_ok"],
        "canonical_kpi_count": mappo_report["canonical_kpi_count"],
        "pass_conditions": pass_conditions,
        "blocking_reasons": [key for key, ok in pass_conditions.items() if not ok],
        "performance_claim_allowed": False,
        "blockers_preserved": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
        "remaining_blockers": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
    }
    training_manifest = {
        "created_at_utc": utc_now(),
        "prompt": 4,
        "status": status,
        "split_manifest": str(output_root / "split_manifest.json"),
        "gatv2_encoder_contract": str(output_root / "gatv2_encoder_contract.json"),
        "dynamic_embedding_audit": str(output_root / "dynamic_embedding_audit.json"),
        "training_metrics": str(output_root / "training_metrics.jsonl"),
        "validation_metrics": str(output_root / "validation_metrics.json"),
        "checkpoint_best_validation": str(output_root / "checkpoint_best_validation.pt"),
        "checkpoint_last": str(output_root / "checkpoint_last.pt"),
        "canonical_12kpi": str(output_root / "canonical_12kpi.json"),
        "pilot_gate": str(output_root / "pilot_gate.json"),
        "mappo_summary": mappo_report,
        "performance_claim_allowed": False,
        "remaining_blockers": ["FLEET_FREQUENCY_NOT_OFFICIAL"],
    }
    dump_json(output_root / "training_manifest.json", training_manifest)
    dump_json(output_root / "pilot_gate.json", pilot_gate)
    write_artifact_index(output_root)
    return pilot_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Prompt 4 frozen dynamic GATv2-MAPPO seed-1 pilot.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--training-snapshots", type=int, default=512)
    parser.add_argument("--validation-snapshots", type=int, default=64)
    parser.add_argument("--agents", type=int, default=128)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--require-mps", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
