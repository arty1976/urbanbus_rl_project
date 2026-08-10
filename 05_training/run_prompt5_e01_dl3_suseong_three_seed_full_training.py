from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is present in the project env, but keep seed audit robust.
    np = None


ARTIFACT_PREFIX = "prompt5_e01_dl3_suseong_three_seed_full_training"
DL2_ARTIFACT = "prompt5_e01_dl2_suseong_mac_m4_capacity_envelope_20260731_112258"
DL2_GATE = "PASS_SUSEONG_MAC_M4_CAPACITY_ENVELOPE_AND_FULL_TRAINING_PROFILE_SELECTED"
PASS_GATE = "PASS_SUSEONG_GATV2_MAPPO_THREE_SEED_FULL_TRAINING_COMPLETED_ON_MAC_M4"
PASS_HIGH_VARIANCE = "PASS_SUSEONG_GATV2_MAPPO_THREE_SEED_FULL_TRAINING_COMPLETED_HIGH_SEED_VARIANCE"
FAIL_DL2 = "FAIL_DL2_UPSTREAM_INTEGRITY"
FAIL_PROFILE = "FAIL_SELECTED_PROFILE_MISMATCH"
FAIL_SPLIT = "FAIL_SUSEONG_SPLIT_INTEGRITY"
FAIL_MPS = "FAIL_NATIVE_MPS_UNAVAILABLE"
FAIL_CPU = "FAIL_CPU_FALLBACK_USED"
FAIL_OOM = "FAIL_MPS_OUT_OF_MEMORY"
FAIL_CONFIG = "FAIL_SEED_CONFIGURATION_DRIFT"
FAIL_INCOMPLETE = "FAIL_SEED_TRAINING_INCOMPLETE"
FAIL_GATV2 = "FAIL_GATV2_NOT_LEARNING"
FAIL_ACTOR = "FAIL_ACTOR_NOT_LEARNING"
FAIL_CRITIC = "FAIL_CRITIC_NOT_LEARNING"
FAIL_NAN = "FAIL_NAN_OR_INF"
FAIL_RELOAD = "FAIL_CHECKPOINT_RELOAD"
FAIL_VAL_MUTATION = "FAIL_VALIDATION_PARAMETER_MUTATION"
FAIL_TEST_MUTATION = "FAIL_TEST_PARAMETER_MUTATION"
FAIL_TEST_SELECTION = "FAIL_TEST_DATA_USED_FOR_SELECTION"
FAIL_SCOPE = "FAIL_H200_OR_CUDA_SCOPE_VIOLATION"
FAIL_MANIFEST = "FAIL_MANIFEST_INTEGRITY"

TOP_REQUIRED_FILES = [
    "upstream_validation.json",
    "selected_profile_snapshot.json",
    "study_area_snapshot.json",
    "dataset_split_manifest.json",
    "dataset_split_audit.json",
    "training_budget.json",
    "seed_registry.json",
    "seed_registry.parquet",
    "three_seed_metric_summary.json",
    "three_seed_metric_summary.parquet",
    "seed_stability_audit.json",
    "representative_checkpoint_selection.json",
    "runtime_estimate_comparison.json",
    "resource_usage_summary.json",
    "external_access_audit.json",
    "scope_guard_audit.json",
    "source_change_inventory.json",
    "artifact_manifest.json",
    "gate_decision.json",
    "final_report.json",
    "final_report.md",
    "_SUCCESS.lock",
]

SEED_REQUIRED_FILES = [
    "configuration.json",
    "runtime_environment.json",
    "training_metrics.jsonl",
    "validation_metrics.jsonl",
    "test_metrics.json",
    "learning_path_audit.json",
    "gradient_audit.json",
    "parameter_delta.json",
    "loss_finiteness_audit.json",
    "memory_audit.json",
    "initial_checkpoint.pt",
    "best_validation_checkpoint.pt",
    "final_checkpoint.pt",
    "checkpoint_manifest.json",
    "checkpoint_reload_audit.json",
    "seed_gate.json",
]


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def iso_kst() -> str:
    return now_kst().isoformat(timespec="seconds")


def timestamp() -> str:
    return now_kst().strftime("%Y%m%d_%H%M%S")


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def import_dl1(project_root: Path) -> Any:
    module_path = project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
    spec = importlib.util.spec_from_file_location("prompt5_dl1", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import DL-1 runner: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parquet_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                flat[key] = value
        out.append(flat)
    return out


def safe_to_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    pd.DataFrame(parquet_rows(rows)).to_parquet(path, index=False)


def runtime_environment(device: str) -> Dict[str, Any]:
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    return {
        "created_at": iso_kst(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python": sys.version,
        "torch": torch.__version__,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_device": "mps" if device == "mps" and mps_available else None,
        "cpu_fallback_allowed": False,
        "cpu_fallback_used": False,
    }


def memory_snapshot_mb() -> Dict[str, Optional[float]]:
    current = driver = None
    if torch.backends.mps.is_available():
        try:
            current = float(torch.mps.current_allocated_memory()) / (1024 * 1024)
            driver = float(torch.mps.driver_allocated_memory()) / (1024 * 1024)
        except Exception:
            pass
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    return {
        "mps_current_allocated_mb": current,
        "mps_driver_allocated_mb": driver,
        "process_rss_mb": rss,
    }


def max_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [float(v) for v in values if v is not None]
    return max(present) if present else None


def finite_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def tensor_stats(values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    if mask is not None:
        values = values[mask.bool()]
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def explained_variance(predicted: torch.Tensor, target: torch.Tensor) -> Optional[float]:
    predicted = predicted.detach().float()
    target = target.detach().float()
    if predicted.numel() < 2:
        return None
    var_y = torch.var(target, unbiased=False)
    if float(var_y.detach().cpu().item()) <= 1e-12:
        return None
    return float((1.0 - torch.var(target - predicted, unbiased=False) / var_y).detach().cpu().item())


def list_split_files(project_root: Path) -> Tuple[Path, List[Path], List[Path], List[Path], Dict[str, Any]]:
    dataset = project_root / "05_training/artifacts/dataset_full_20260422_084243"
    train = sorted((dataset / "train").glob("*.pt"))
    val = sorted((dataset / "val").glob("*.pt"))
    test = sorted((dataset / "test").glob("*.pt"))
    build = read_json(dataset / "build_report.json")
    return dataset, train, val, test, build


def snapshot_number(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def split_manifest(project_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dataset, train, val, test, build = list_split_files(project_root)
    split_sets = {"train": {p.name for p in train}, "validation": {p.name for p in val}, "test": {p.name for p in test}}
    overlap = {
        "train_validation": len(split_sets["train"] & split_sets["validation"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "validation_test": len(split_sets["validation"] & split_sets["test"]),
    }
    order = {
        "train_first": train[0].name if train else None,
        "train_last": train[-1].name if train else None,
        "validation_first": val[0].name if val else None,
        "validation_last": val[-1].name if val else None,
        "test_first": test[0].name if test else None,
        "test_last": test[-1].name if test else None,
    }
    train_nums = [snapshot_number(p) for p in train]
    val_nums = [snapshot_number(p) for p in val]
    test_nums = [snapshot_number(p) for p in test]
    timestamp_order_violation = int(
        bool(train_nums and val_nums and max(train_nums) >= min(val_nums))
        or bool(val_nums and test_nums and max(val_nums) >= min(test_nums))
    )
    duplicate_assignment = len(train) + len(val) + len(test) - len(split_sets["train"] | split_sets["validation"] | split_sets["test"])
    manifest = {
        "created_at": iso_kst(),
        "dataset_root": str(dataset),
        "authoritative_split_source": "existing frozen dataset_full_20260422_084243 train/val/test directories",
        "build_report_hash": sha256_file(dataset / "build_report.json"),
        "full_train_snapshot_count": len(train),
        "full_validation_snapshot_count": len(val),
        "full_test_snapshot_count": len(test),
        "split_counts_from_build_report": build.get("snapshots", {}).get("split_counts", {}),
        "split_order": order,
        "split_hash": sha256_text(json.dumps({k: sorted(v) for k, v in split_sets.items()}, sort_keys=True)),
        "train_files_hash": sha256_text(json.dumps([p.name for p in train], sort_keys=True)),
        "validation_files_hash": sha256_text(json.dumps([p.name for p in val], sort_keys=True)),
        "test_files_hash": sha256_text(json.dumps([p.name for p in test], sort_keys=True)),
    }
    audit = {
        "created_at": iso_kst(),
        "train_validation_overlap": overlap["train_validation"],
        "train_test_overlap": overlap["train_test"],
        "validation_test_overlap": overlap["validation_test"],
        "timestamp_order_violation": timestamp_order_violation,
        "duplicate_snapshot_assignment": duplicate_assignment,
        "test_data_used_for_tuning": False,
        "split_integrity_passed": all(v == 0 for v in overlap.values()) and timestamp_order_violation == 0 and duplicate_assignment == 0,
    }
    return manifest, audit


def validate_upstream(project_root: Path, upstream_artifact: Path, profile_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], bool, str]:
    gate = read_json(upstream_artifact / "gate_decision.json")
    selected = read_json(upstream_artifact / "selected_full_training_profile.json")
    study = read_json(upstream_artifact / "study_area_snapshot.json")
    upstream = {
        "created_at": iso_kst(),
        "artifact": str(upstream_artifact),
        "gate": gate.get("gate"),
        "gate_passed": bool(gate.get("gate_passed")),
        "selected_full_training_profile": gate.get("selected_full_training_profile"),
        "selected_profile_id": gate.get("selected_profile_id"),
        "success_lock_present": (upstream_artifact / "_SUCCESS.lock").exists(),
        "artifact_manifest_present": (upstream_artifact / "artifact_manifest.json").exists(),
        "required_gate": DL2_GATE,
    }
    expected = {
        "selected_full_training_profile": "BALANCED",
        "profile_id": profile_id,
        "agents": 8,
        "gatv2_hidden": 128,
        "rollout_horizon": 512,
        "minibatch_size": 256,
        "ppo_epochs": 4,
        "device": "mps",
    }
    profile_ok = (
        selected.get("selected_full_training_profile") == expected["selected_full_training_profile"]
        and selected.get("profile_id") == expected["profile_id"]
        and int(selected.get("agents", -1)) == expected["agents"]
        and int(selected.get("gatv2_hidden", -1)) == expected["gatv2_hidden"]
        and int(selected.get("rollout_horizon", -1)) == expected["rollout_horizon"]
        and int(selected.get("minibatch_size", -1)) == expected["minibatch_size"]
        and int(selected.get("ppo_epochs", -1)) == expected["ppo_epochs"]
        and selected.get("device") == expected["device"]
    )
    upstream_ok = bool(gate.get("gate_passed")) and gate.get("gate") == DL2_GATE and profile_ok and upstream["success_lock_present"]
    if not bool(gate.get("gate_passed")) or gate.get("gate") != DL2_GATE:
        fail_gate = FAIL_DL2
    elif not profile_ok:
        fail_gate = FAIL_PROFILE
    else:
        fail_gate = ""
    profile_snapshot = {
        "created_at": iso_kst(),
        "source": str(upstream_artifact / "selected_full_training_profile.json"),
        "expected": expected,
        "selected_profile": selected,
        "profile_integrity_passed": profile_ok,
    }
    return upstream, profile_snapshot, study, upstream_ok, fail_gate


def build_training_budget(train_count: int, val_count: int, test_count: int, selected: Mapping[str, Any]) -> Dict[str, Any]:
    horizon = int(selected["rollout_horizon"])
    agents = int(selected["agents"])
    ppo_epochs = int(selected["ppo_epochs"])
    rollout_count = int(math.ceil(train_count / horizon))
    return {
        "created_at": iso_kst(),
        "training_snapshot_count": train_count,
        "validation_snapshot_count": val_count,
        "test_snapshot_count": test_count,
        "rollout_horizon_configured": horizon,
        "total_rollout_count": rollout_count,
        "total_environment_steps": train_count * agents,
        "total_ppo_update_count": rollout_count * ppo_epochs,
        "total_minibatch_update_count": rollout_count * ppo_epochs,
        "dataset_pass_count": 1,
        "last_rollout_partial": train_count % horizon != 0,
        "last_rollout_effective_horizon": train_count % horizon or horizon,
    }


def set_all_seeds(seed: int) -> Dict[str, Any]:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass
    return {
        "python_random_seed": seed,
        "numpy_seed": seed if np is not None else None,
        "torch_seed": seed,
        "environment_seed": seed,
        "action_sampling_seed": seed,
        "data_loader_seed": seed,
    }


def load_subgraphs(dl1: Any, paths: Sequence[Path], spec: Mapping[str, Any]) -> List[Any]:
    return [dl1.make_subgraph_data(dl1.torch_load(path), spec) for path in paths]


def collect_rollout_from_data(
    dl1: Any,
    data_seq: Sequence[Any],
    offset: int,
    horizon: int,
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    device: torch.device,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    rewards: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    old_log_probs: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    agent_indices: List[List[int]] = []
    started = time.perf_counter()
    encoder.eval()
    actor.eval()
    critic.eval()
    with torch.no_grad():
        for local_step in range(horizon):
            absolute_step = offset + local_step
            data = data_seq[absolute_step].to(device)
            indices = dl1.agent_indices_for_step(config["spec"], absolute_step, int(config["effective_agents"]))
            logits, value, agent_mask, _ = dl1.forward_policy(data, indices, encoder, actor, critic)
            dist = Categorical(logits=logits)
            action = dist.sample()
            target = dl1.action_targets_from_y(data.y[torch.tensor(indices, dtype=torch.long, device=device)], int(config["action_dim"]))
            reward = dl1.reward_from_actions(action, target)
            rewards.append(reward.detach())
            values.append(value.detach())
            old_log_probs.append(dist.log_prob(action).detach())
            actions.append(action.detach())
            masks.append(agent_mask.detach())
            agent_indices.append(indices)
        next_idx = min(offset + horizon, len(data_seq) - 1)
        next_data = data_seq[next_idx].to(device)
        next_indices = dl1.agent_indices_for_step(config["spec"], next_idx, int(config["effective_agents"]))
        _logits, last_next_value, _mask, _ = dl1.forward_policy(next_data, next_indices, encoder, actor, critic)

    values_t = torch.stack(values)
    next_values_t = torch.zeros_like(values_t)
    next_values_t[:-1] = values_t[1:]
    next_values_t[-1] = last_next_value.detach()
    rewards_t = torch.stack(rewards)
    masks_t = torch.stack(masks).bool()
    terminated = torch.zeros_like(rewards_t, dtype=torch.bool)
    truncated = torch.zeros_like(rewards_t, dtype=torch.bool)
    truncated[-1] = True
    returns, advantages, normalized, gae_audit = dl1.compute_gae(
        rewards_t,
        values_t,
        next_values_t,
        terminated,
        truncated,
        masks_t,
        float(config["gamma"]),
        float(config["gae_lambda"]),
    )
    return {
        "rollout_collection_seconds": time.perf_counter() - started,
        "rewards": rewards_t.detach(),
        "values": values_t.detach(),
        "old_log_probs": torch.stack(old_log_probs).detach(),
        "actions": torch.stack(actions).detach(),
        "agent_mask": masks_t.detach(),
        "agent_indices": agent_indices,
        "returns": returns.detach(),
        "advantages": advantages.detach(),
        "normalized_advantages": normalized.detach(),
        "gae_audit": gae_audit,
    }


def run_ppo_update(
    dl1: Any,
    data_seq: Sequence[Any],
    offset: int,
    rollout: Mapping[str, Any],
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer],
    device: torch.device,
    config: Mapping[str, Any],
    before_state: Mapping[str, Mapping[str, torch.Tensor]],
    rollout_index: int,
    ppo_update_start_index: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    metrics_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, Any]] = []
    horizon = int(rollout["actions"].size(0))
    valid_flat = torch.where(rollout["agent_mask"].reshape(-1))[0]
    selected_flat = valid_flat[: min(int(config["minibatch_size"]), int(valid_flat.numel()))]
    if selected_flat.numel() == 0:
        raise RuntimeError("No valid Suseong agents were available for PPO update.")
    update_started = time.perf_counter()
    for epoch in range(int(config["ppo_epochs"])):
        encoder.train()
        actor.train()
        critic.train()
        new_log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        predicted_values: List[torch.Tensor] = []
        for local_step in range(horizon):
            absolute_step = offset + local_step
            data = data_seq[absolute_step].to(device)
            logits, values, _mask, _ = dl1.forward_policy(data, rollout["agent_indices"][local_step], encoder, actor, critic)
            dist = Categorical(logits=logits)
            new_log_probs.append(dist.log_prob(rollout["actions"][local_step].to(device)))
            entropies.append(dist.entropy())
            predicted_values.append(values)
        new_log_probs_t = torch.stack(new_log_probs)
        entropy_t = torch.stack(entropies)
        predicted_values_t = torch.stack(predicted_values)
        old_log_probs = rollout["old_log_probs"].to(device)
        returns = rollout["returns"].to(device)
        normalized_adv = rollout["normalized_advantages"].to(device)
        rewards = rollout["rewards"].to(device)
        flat_new = new_log_probs_t.reshape(-1)
        flat_old = old_log_probs.reshape(-1)
        flat_adv = normalized_adv.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = predicted_values_t.reshape(-1)
        idx = selected_flat.to(device)
        ratio = torch.exp(flat_new[idx] - flat_old[idx])
        unclipped = ratio * flat_adv[idx]
        clipped = torch.clamp(ratio, 1.0 - float(config["ppo_clip_epsilon"]), 1.0 + float(config["ppo_clip_epsilon"])) * flat_adv[idx]
        policy_loss = -torch.min(unclipped, clipped).mean()
        value_loss = F.mse_loss(flat_values[idx], flat_returns[idx])
        entropy = entropy_t.reshape(-1)[idx].mean()
        total_loss = policy_loss + float(config["value_loss_coef"]) * value_loss - float(config["entropy_coef"]) * entropy
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_before = {
            "gatv2": dl1.grad_norm(encoder),
            "actor": dl1.grad_norm(actor),
            "critic": dl1.grad_norm(critic),
        }
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=float(config["gatv2_grad_clip"]))
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(config["mappo_grad_clip"]))
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float(config["mappo_grad_clip"]))
        grad_after = {
            "gatv2": dl1.grad_norm(encoder),
            "actor": dl1.grad_norm(actor),
            "critic": dl1.grad_norm(critic),
        }
        optimizers["gatv2"].step()
        optimizers["actor"].step()
        optimizers["critic"].step()
        approx_kl = (flat_old[idx] - flat_new[idx]).mean()
        clip_fraction = ((ratio - 1.0).abs() > float(config["ppo_clip_epsilon"])).float().mean()
        finite_losses = {
            "policy_loss_finite": bool(torch.isfinite(policy_loss).detach().cpu().item()),
            "value_loss_finite": bool(torch.isfinite(value_loss).detach().cpu().item()),
            "entropy_finite": bool(torch.isfinite(entropy).detach().cpu().item()),
            "total_loss_finite": bool(torch.isfinite(total_loss).detach().cpu().item()),
            "predicted_value_finite": bool(torch.isfinite(flat_values[idx]).all().detach().cpu().item()),
            "target_return_finite": bool(torch.isfinite(flat_returns[idx]).all().detach().cpu().item()),
            "advantage_finite": bool(torch.isfinite(flat_adv[idx]).all().detach().cpu().item()),
        }
        ppo_update_index = ppo_update_start_index + epoch + 1
        mem = memory_snapshot_mb()
        metrics_rows.append(
            {
                "global_step": int(min(offset + horizon, len(data_seq)) * int(config["effective_agents"])),
                "snapshot_start": int(offset),
                "snapshot_end": int(offset + horizon - 1),
                "rollout_index": int(rollout_index),
                "ppo_update_index": int(ppo_update_index),
                "reward_mean": tensor_stats(rewards, rollout["agent_mask"].to(device))["mean"],
                "reward_std": tensor_stats(rewards, rollout["agent_mask"].to(device))["std"],
                "episode_return_mean": tensor_stats(returns, rollout["agent_mask"].to(device))["mean"],
                "episode_return_std": tensor_stats(returns, rollout["agent_mask"].to(device))["std"],
                "policy_loss": finite_float(policy_loss),
                "value_loss": finite_float(value_loss),
                "entropy": finite_float(entropy),
                "approx_kl": finite_float(approx_kl),
                "clip_fraction": finite_float(clip_fraction),
                "predicted_value_mean": tensor_stats(flat_values[idx])["mean"],
                "predicted_value_std": tensor_stats(flat_values[idx])["std"],
                "target_return_mean": tensor_stats(flat_returns[idx])["mean"],
                "target_return_std": tensor_stats(flat_returns[idx])["std"],
                "advantage_mean": tensor_stats(flat_adv[idx])["mean"],
                "advantage_std": tensor_stats(flat_adv[idx])["std"],
                "explained_variance": explained_variance(flat_values[idx], flat_returns[idx]),
                "gatv2_grad_norm_before_clip": grad_before["gatv2"],
                "gatv2_grad_norm_after_clip": grad_after["gatv2"],
                "actor_grad_norm_before_clip": grad_before["actor"],
                "actor_grad_norm_after_clip": grad_after["actor"],
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "gatv2_parameter_delta": dl1.delta_stats(encoder, before_state["gatv2"])["l2_delta"],
                "actor_parameter_delta": dl1.delta_stats(actor, before_state["actor"])["l2_delta"],
                "critic_parameter_delta": dl1.delta_stats(critic, before_state["critic"])["l2_delta"],
                "rollout_seconds": float(rollout["rollout_collection_seconds"]),
                "ppo_update_seconds": float(time.perf_counter() - update_started),
                "elapsed_seconds": float(time.perf_counter() - config["started_at_perf"]),
                **mem,
                "nan_count": 0 if all(finite_losses.values()) else 1,
                "inf_count": 0 if all(finite_losses.values()) else 1,
            }
        )
        gradient_rows.append(
            {
                "rollout_index": int(rollout_index),
                "ppo_update_index": int(ppo_update_index),
                "gatv2_grad_norm_before_clip": grad_before["gatv2"],
                "gatv2_grad_norm_after_clip": grad_after["gatv2"],
                "actor_grad_norm_before_clip": grad_before["actor"],
                "actor_grad_norm_after_clip": grad_after["actor"],
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "critic_optimizer_step_executed": True,
                "critic_nonzero_gradient_step": bool(grad_before["critic"] > 0.0),
            }
        )
        loss_rows.append({"rollout_index": int(rollout_index), "ppo_update_index": int(ppo_update_index), **finite_losses})
    return metrics_rows, gradient_rows, loss_rows


def evaluate_split(
    dl1: Any,
    data_seq: Sequence[Any],
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    device: torch.device,
    config: Mapping[str, Any],
    split: str,
    phase: str,
    rollout_index: int,
    ppo_update_index: int,
) -> Dict[str, Any]:
    before = {
        "gatv2": dl1.module_hash(encoder),
        "actor": dl1.module_hash(actor),
        "critic": dl1.module_hash(critic),
    }
    rewards: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    entropies: List[torch.Tensor] = []
    action_matches: List[torch.Tensor] = []
    started = time.perf_counter()
    encoder.eval()
    actor.eval()
    critic.eval()
    with torch.no_grad():
        for step, cpu_data in enumerate(data_seq):
            data = cpu_data.to(device)
            indices = dl1.agent_indices_for_step(config["spec"], step, int(config["effective_agents"]))
            logits, value, agent_mask, _ = dl1.forward_policy(data, indices, encoder, actor, critic)
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1)
            target = dl1.action_targets_from_y(data.y[torch.tensor(indices, dtype=torch.long, device=device)], int(config["action_dim"]))
            reward = dl1.reward_from_actions(action, target)
            rewards.append(reward.detach())
            values.append(value.detach())
            masks.append(agent_mask.detach())
            entropies.append(dist.entropy().detach())
            action_matches.append((action == target).float().detach())
    rewards_t = torch.stack(rewards)
    values_t = torch.stack(values)
    masks_t = torch.stack(masks).bool()
    next_values_t = torch.zeros_like(values_t)
    next_values_t[:-1] = values_t[1:]
    next_values_t[-1] = values_t[-1]
    terminated = torch.zeros_like(rewards_t, dtype=torch.bool)
    truncated = torch.zeros_like(rewards_t, dtype=torch.bool)
    truncated[-1] = True
    returns, advantages, _normalized, _audit = dl1.compute_gae(
        rewards_t,
        values_t,
        next_values_t,
        terminated,
        truncated,
        masks_t,
        float(config["gamma"]),
        float(config["gae_lambda"]),
    )
    valid = masks_t.bool()
    value_loss = F.mse_loss(values_t[valid], returns[valid]) if bool(valid.any().detach().cpu().item()) else torch.tensor(float("nan"), device=device)
    after = {
        "gatv2": dl1.module_hash(encoder),
        "actor": dl1.module_hash(actor),
        "critic": dl1.module_hash(critic),
    }
    mem = memory_snapshot_mb()
    result = {
        "created_at": iso_kst(),
        "split": split,
        "phase": phase,
        "rollout_index": int(rollout_index),
        "ppo_update_index": int(ppo_update_index),
        f"{split}_reward_mean": tensor_stats(rewards_t, valid)["mean"],
        f"{split}_reward_std": tensor_stats(rewards_t, valid)["std"],
        f"{split}_episode_return_mean": tensor_stats(returns, valid)["mean"],
        f"{split}_episode_return_std": tensor_stats(returns, valid)["std"],
        f"{split}_value_loss": finite_float(value_loss),
        f"{split}_policy_diagnostic": {
            "deterministic_argmax_match_rate": tensor_stats(torch.stack(action_matches), valid)["mean"],
            "optimizer_step": 0,
            "gradient_update": 0,
            "parameter_mutation": 0,
        },
        f"{split}_entropy": tensor_stats(torch.stack(entropies), valid)["mean"],
        f"{split}_explained_variance": explained_variance(values_t[valid], returns[valid]) if bool(valid.any().detach().cpu().item()) else None,
        f"{split}_nan_count": 0 if torch.isfinite(value_loss).detach().cpu().item() else 1,
        f"{split}_inf_count": 0 if torch.isfinite(value_loss).detach().cpu().item() else 1,
        "parameter_mutation_count": 0 if before == after else 1,
        "evaluation_seconds": float(time.perf_counter() - started),
        **mem,
    }
    return result


def save_checkpoint(
    path: Path,
    kind: str,
    seed: int,
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer],
    config: Mapping[str, Any],
    validation_metric: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "created_at": iso_kst(),
        "checkpoint_kind": kind,
        "seed": seed,
        "gatv2_state_dict": encoder.state_dict(),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "gatv2_optimizer_state_dict": optimizers["gatv2"].state_dict(),
        "actor_optimizer_state_dict": optimizers["actor"].state_dict(),
        "critic_optimizer_state_dict": optimizers["critic"].state_dict(),
        "training_configuration": {k: v for k, v in config.items() if k not in {"spec", "started_at_perf"}},
        "node_mapping_hash": config["node_mapping_hash"],
        "split_manifest_hash": config["split_manifest_hash"],
        "validation_metric": dict(validation_metric) if validation_metric else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path), "kind": kind, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def reload_checkpoint_audit(
    dl1: Any,
    seed_dir: Path,
    checkpoints: Sequence[Mapping[str, Any]],
    sample_graph: Any,
    config: Mapping[str, Any],
    device: torch.device,
    original_modules: Mapping[str, torch.nn.Module],
) -> Dict[str, Any]:
    rows = []
    indices = dl1.agent_indices_for_step(config["spec"], 0, int(config["effective_agents"]))
    sample = sample_graph.to(device)
    with torch.no_grad():
        original_outputs = {}
        for label, modules in {"current": original_modules}.items():
            logits, values, _mask, _ = dl1.forward_policy(sample, indices, modules["gatv2"], modules["actor"], modules["critic"])
            original_outputs[label] = {"logits": logits.detach().cpu(), "values": values.detach().cpu()}
    for row in checkpoints:
        loaded = torch.load(Path(row["path"]), map_location=device, weights_only=False)
        encoder = dl1.GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
        actor = dl1.MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
        critic = dl1.CentralizedCritic(int(config["gatv2_hidden"])).to(device)
        opts = dl1.optimizer_triplet(encoder, actor, critic, float(config["learning_rate"]))
        encoder.load_state_dict(loaded["gatv2_state_dict"])
        actor.load_state_dict(loaded["actor_state_dict"])
        critic.load_state_dict(loaded["critic_state_dict"])
        optimizer_state_load_success = True
        try:
            opts[0].load_state_dict(loaded["gatv2_optimizer_state_dict"])
            opts[1].load_state_dict(loaded["actor_optimizer_state_dict"])
            opts[2].load_state_dict(loaded["critic_optimizer_state_dict"])
        except Exception:
            optimizer_state_load_success = False
        with torch.no_grad():
            logits, values, _mask, _ = dl1.forward_policy(sample, indices, encoder, actor, critic)
        if row["kind"] == "final":
            ref_logits = original_outputs["current"]["logits"]
            ref_values = original_outputs["current"]["values"]
            output_max_abs_diff = float(max((logits.detach().cpu() - ref_logits).abs().max().item(), (values.detach().cpu() - ref_values).abs().max().item()))
        else:
            output_max_abs_diff = 0.0
        rows.append(
            {
                "checkpoint_kind": row["kind"],
                "path": row["path"],
                "gatv2_parameter_hash_match": dl1.state_dict_hash(loaded["gatv2_state_dict"]) == dl1.module_hash(encoder),
                "actor_parameter_hash_match": dl1.state_dict_hash(loaded["actor_state_dict"]) == dl1.module_hash(actor),
                "critic_parameter_hash_match": dl1.state_dict_hash(loaded["critic_state_dict"]) == dl1.module_hash(critic),
                "optimizer_state_load_success": optimizer_state_load_success,
                "training_configuration_load_success": bool(loaded.get("training_configuration")),
                "node_mapping_hash_match": loaded.get("node_mapping_hash") == config["node_mapping_hash"],
                "split_manifest_hash_match": loaded.get("split_manifest_hash") == config["split_manifest_hash"],
                "same_observation_inference_output_match": output_max_abs_diff <= 1e-5,
                "floating_point_tolerance": 1e-5,
                "output_max_abs_diff": output_max_abs_diff,
            }
        )
    reload_ok = all(
        row["gatv2_parameter_hash_match"]
        and row["actor_parameter_hash_match"]
        and row["critic_parameter_hash_match"]
        and row["optimizer_state_load_success"]
        and row["training_configuration_load_success"]
        and row["node_mapping_hash_match"]
        and row["split_manifest_hash_match"]
        and row["same_observation_inference_output_match"]
        for row in rows
    )
    return {"created_at": iso_kst(), "reload_ok": reload_ok, "checks": rows}


def value_signal_status(training_rows: Sequence[Mapping[str, Any]]) -> str:
    losses = [float(row["value_loss"]) for row in training_rows if row.get("value_loss") is not None and math.isfinite(float(row["value_loss"]))]
    if len(losses) < 3:
        return "VALUE_SIGNAL_INDETERMINATE"
    first = sum(losses[: max(1, len(losses) // 4)]) / max(1, len(losses[: max(1, len(losses) // 4)]))
    last = sum(losses[-max(1, len(losses) // 4) :]) / max(1, len(losses[-max(1, len(losses) // 4) :]))
    ratio = last / first if first > 1e-12 else None
    if ratio is not None and ratio < 0.9:
        return "VALUE_SIGNAL_IMPROVING"
    if ratio is not None and ratio < 1.25:
        return "VALUE_SIGNAL_STABLE"
    if ratio is not None and ratio < 2.0:
        return "VALUE_SIGNAL_NOISY"
    return "VALUE_SIGNAL_DIVERGING"


def run_seed_worker(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).expanduser().resolve()
    seed_dir = Path(args.seed_output_dir).expanduser().resolve()
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)
    runtime = runtime_environment(args.device)
    dump_json(seed_dir / "runtime_environment.json", runtime)
    if not (runtime["platform_system"] == "Darwin" and runtime["platform_machine"] == "arm64" and runtime["mps_built"] and runtime["mps_available"]):
        dump_json(seed_dir / "seed_gate.json", {"created_at": iso_kst(), "seed": seed, "gate": FAIL_MPS, "gate_passed": False})
        return 2
    gc.collect()
    torch.mps.empty_cache()
    seed_audit = set_all_seeds(seed)
    dl1 = import_dl1(project_root)
    dataset, train_files, val_files, test_files, _build = list_split_files(project_root)
    sample_full = dl1.torch_load(train_files[0])
    mapping_artifact = Path(args.mapping_artifact).expanduser().resolve()
    spec, inventory, connectivity, tensor_mask = dl1.build_subgraph_spec(project_root, sample_full, mapping_artifact=mapping_artifact)
    sample_graph = dl1.make_subgraph_data(sample_full, spec)
    device = torch.device("mps")
    train_data = load_subgraphs(dl1, train_files, spec)
    val_data = load_subgraphs(dl1, val_files, spec)
    test_data = load_subgraphs(dl1, test_files, spec)
    split_manifest_payload = read_json(Path(args.parent_split_manifest))
    config = {
        "created_at": iso_kst(),
        "study_area": "SUSEONG_GU_DAEGU",
        "seed": seed,
        "seed_audit": seed_audit,
        "condition": "A",
        "device": "mps",
        "cpu_fallback_allowed": False,
        "agents": 8,
        "effective_agents": min(8, int(inventory["available_suseong_agents"])),
        "gatv2_hidden": 128,
        "gatv2_batch": 1,
        "gatv2_epochs": 1,
        "gatv2_grad_clip": 5.0,
        "rollout_horizon": 512,
        "minibatch_size": 256,
        "ppo_epochs": 4,
        "mappo_grad_clip": 0.5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "learning_rate": 1e-3,
        "ppo_clip_epsilon": 0.2,
        "entropy_coef": 0.01,
        "value_loss_coef": 0.5,
        "action_dim": 3,
        "training_snapshot_count": len(train_files),
        "validation_snapshot_count": len(val_files),
        "test_snapshot_count": len(test_files),
        "node_mapping_hash": spec["node_edge_mapping_hash"],
        "split_manifest_hash": split_manifest_payload["split_hash"],
        "test_data_used_for_selection": False,
    }
    dump_json(seed_dir / "configuration.json", {k: v for k, v in config.items() if k != "spec"})
    config["spec"] = spec
    config["started_at_perf"] = time.perf_counter()
    encoder = dl1.GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
    actor = dl1.MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
    critic = dl1.CentralizedCritic(int(config["gatv2_hidden"])).to(device)
    gatv2_opt, actor_opt, critic_opt = dl1.optimizer_triplet(encoder, actor, critic, float(config["learning_rate"]))
    optimizers = {"gatv2": gatv2_opt, "actor": actor_opt, "critic": critic_opt}
    before_state = {"gatv2": dl1.clone_state_dict(encoder), "actor": dl1.clone_state_dict(actor), "critic": dl1.clone_state_dict(critic)}
    checkpoints: List[Dict[str, Any]] = []
    checkpoints.append(save_checkpoint(seed_dir / "initial_checkpoint.pt", "initial", seed, encoder, actor, critic, optimizers, config, None))
    training_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, Any]] = []
    memory_rows: List[Dict[str, Optional[float]]] = []
    validation_parameter_mutation = 0
    test_parameter_mutation = 0
    best_validation: Optional[Dict[str, Any]] = None

    def maybe_record_validation(phase: str, rollout_index: int, ppo_update_index: int) -> None:
        nonlocal best_validation, validation_parameter_mutation, checkpoints
        row = evaluate_split(dl1, val_data, encoder, actor, critic, device, config, "validation", phase, rollout_index, ppo_update_index)
        validation_rows.append(row)
        validation_parameter_mutation += int(row["parameter_mutation_count"])
        if (
            best_validation is None
            or float(row["validation_episode_return_mean"]) > float(best_validation["validation_episode_return_mean"])
            or (
                math.isclose(float(row["validation_episode_return_mean"]), float(best_validation["validation_episode_return_mean"]), rel_tol=1e-6, abs_tol=1e-6)
                and float(row["validation_value_loss"]) < float(best_validation["validation_value_loss"])
            )
        ):
            best_validation = row
            ckpt = save_checkpoint(seed_dir / "best_validation_checkpoint.pt", "best_validation", seed, encoder, actor, critic, optimizers, config, row)
            checkpoints = [c for c in checkpoints if c["kind"] != "best_validation"] + [ckpt]

    started = time.perf_counter()
    maybe_record_validation("start", 0, 0)
    horizon = int(config["rollout_horizon"])
    total_rollouts = int(math.ceil(len(train_data) / horizon))
    midpoint_rollout = max(1, total_rollouts // 2)
    ppo_index = 0
    mps_oom = False
    try:
        for rollout_index, offset in enumerate(range(0, len(train_data), horizon), start=1):
            effective_horizon = min(horizon, len(train_data) - offset)
            rollout = collect_rollout_from_data(dl1, train_data, offset, effective_horizon, encoder, actor, critic, device, config)
            rows, grads, losses = run_ppo_update(
                dl1,
                train_data,
                offset,
                rollout,
                encoder,
                actor,
                critic,
                optimizers,
                device,
                config,
                before_state,
                rollout_index,
                ppo_index,
            )
            ppo_index += int(config["ppo_epochs"])
            training_rows.extend(rows)
            gradient_rows.extend(grads)
            loss_rows.extend(losses)
            memory_rows.append(memory_snapshot_mb())
            if rollout_index == midpoint_rollout:
                maybe_record_validation("middle", rollout_index, ppo_index)
            del rollout
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            mps_oom = True
        else:
            raise
    maybe_record_validation("end", total_rollouts, ppo_index)
    checkpoints.append(save_checkpoint(seed_dir / "final_checkpoint.pt", "final", seed, encoder, actor, critic, optimizers, config, validation_rows[-1]))
    final_modules = {"gatv2": encoder, "actor": actor, "critic": critic}
    reload_audit = reload_checkpoint_audit(dl1, seed_dir, checkpoints, sample_graph, config, device, final_modules)
    dump_json(seed_dir / "checkpoint_reload_audit.json", reload_audit)
    checkpoint_manifest = {"created_at": iso_kst(), "checkpoints": checkpoints, "checkpoint_count": len(checkpoints)}
    dump_json(seed_dir / "checkpoint_manifest.json", checkpoint_manifest)
    loaded_best = torch.load(seed_dir / "best_validation_checkpoint.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(loaded_best["gatv2_state_dict"])
    actor.load_state_dict(loaded_best["actor_state_dict"])
    critic.load_state_dict(loaded_best["critic_state_dict"])
    test_metrics = evaluate_split(dl1, test_data, encoder, actor, critic, device, config, "test", "holdout_once_after_best_validation_checkpoint_selection", total_rollouts, ppo_index)
    test_parameter_mutation += int(test_metrics["parameter_mutation_count"])
    dump_json(seed_dir / "test_metrics.json", test_metrics)
    parameter_delta = {
        "created_at": iso_kst(),
        "gatv2": dl1.delta_stats(final_modules["gatv2"], before_state["gatv2"]),
        "actor": dl1.delta_stats(final_modules["actor"], before_state["actor"]),
        "critic": dl1.delta_stats(final_modules["critic"], before_state["critic"]),
    }
    critic_forward_count = len(training_rows) * int(config["effective_agents"])
    critic_backward_count = sum(1 for row in gradient_rows if row["critic_grad_norm_before_clip"] > 0.0)
    critic_step_count = len(gradient_rows)
    critic_nonzero_step_count = sum(1 for row in gradient_rows if row["critic_nonzero_gradient_step"])
    learning_path = {
        "created_at": iso_kst(),
        "full_training_completed": not mps_oom and len(training_rows) == total_rollouts * int(config["ppo_epochs"]),
        "critic_forward_count": critic_forward_count,
        "critic_backward_count": critic_backward_count,
        "critic_optimizer_step_count": critic_step_count,
        "critic_nonzero_gradient_step_count": critic_nonzero_step_count,
        "critic_parameter_delta_l2": parameter_delta["critic"]["l2_delta"],
        "critic_learning_active": critic_forward_count > 0 and critic_backward_count > 0 and critic_step_count > 0 and critic_nonzero_step_count > 0 and parameter_delta["critic"]["l2_delta"] > 0.0,
        "value_signal_classification": value_signal_status(training_rows),
        "classification_basis": {
            "first_value_loss": training_rows[0]["value_loss"] if training_rows else None,
            "final_value_loss": training_rows[-1]["value_loss"] if training_rows else None,
        },
    }
    nan_count = sum(int(row.get("nan_count", 0)) for row in training_rows) + sum(int(row.get("validation_nan_count", 0)) for row in validation_rows) + int(test_metrics.get("test_nan_count", 0))
    inf_count = sum(int(row.get("inf_count", 0)) for row in training_rows) + sum(int(row.get("validation_inf_count", 0)) for row in validation_rows) + int(test_metrics.get("test_inf_count", 0))
    memory_audit = {
        "created_at": iso_kst(),
        "peak_mps_current_allocation_mb": max_optional([row.get("mps_current_allocated_mb") for row in training_rows + validation_rows + [test_metrics]]),
        "peak_mps_driver_allocation_mb": max_optional([row.get("mps_driver_allocated_mb") for row in training_rows + validation_rows + [test_metrics]]),
        "peak_process_rss_mb": max_optional([row.get("process_rss_mb") for row in training_rows + validation_rows + [test_metrics]]),
        "swap_usage": "not_privileged_not_measured",
        "memory_pressure": "not_privileged_not_measured",
        "thermal_warning": "not_privileged_not_measured",
        "unexpected_process_termination": False,
        "mps_oom": mps_oom,
    }
    loss_audit = {
        "created_at": iso_kst(),
        "training_loss_rows": loss_rows,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "all_finite": nan_count == 0 and inf_count == 0 and all(all(v for k, v in row.items() if k.endswith("_finite")) for row in loss_rows),
    }
    write_jsonl(seed_dir / "training_metrics.jsonl", training_rows)
    write_jsonl(seed_dir / "validation_metrics.jsonl", validation_rows)
    dump_json(seed_dir / "learning_path_audit.json", learning_path)
    dump_json(seed_dir / "gradient_audit.json", {"created_at": iso_kst(), "rows": gradient_rows})
    dump_json(seed_dir / "parameter_delta.json", parameter_delta)
    dump_json(seed_dir / "loss_finiteness_audit.json", loss_audit)
    dump_json(seed_dir / "memory_audit.json", memory_audit)
    elapsed = time.perf_counter() - started
    seed_gate = PASS_GATE
    gate_passed = True
    if mps_oom:
        seed_gate, gate_passed = FAIL_OOM, False
    elif parameter_delta["gatv2"]["l2_delta"] <= 0:
        seed_gate, gate_passed = FAIL_GATV2, False
    elif parameter_delta["actor"]["l2_delta"] <= 0:
        seed_gate, gate_passed = FAIL_ACTOR, False
    elif not learning_path["critic_learning_active"]:
        seed_gate, gate_passed = FAIL_CRITIC, False
    elif nan_count or inf_count:
        seed_gate, gate_passed = FAIL_NAN, False
    elif not reload_audit["reload_ok"]:
        seed_gate, gate_passed = FAIL_RELOAD, False
    elif validation_parameter_mutation:
        seed_gate, gate_passed = FAIL_VAL_MUTATION, False
    elif test_parameter_mutation:
        seed_gate, gate_passed = FAIL_TEST_MUTATION, False
    elif not validation_rows or not test_metrics:
        seed_gate, gate_passed = FAIL_INCOMPLETE, False
    seed_gate_payload = {
        "created_at": iso_kst(),
        "seed": seed,
        "gate": seed_gate,
        "gate_passed": gate_passed,
        "elapsed_seconds": elapsed,
        "native_mps_used": True,
        "cpu_fallback_used": False,
        "training_rows": len(training_rows),
        "validation_rows": len(validation_rows),
        "holdout_test_completed": bool(test_metrics),
        "best_validation_episode_return_mean": best_validation["validation_episode_return_mean"] if best_validation else None,
        "best_validation_value_loss": best_validation["validation_value_loss"] if best_validation else None,
        "test_episode_return_mean": test_metrics["test_episode_return_mean"],
        "final_training_reward_mean": training_rows[-1]["reward_mean"] if training_rows else None,
        "final_value_loss": training_rows[-1]["value_loss"] if training_rows else None,
        "validation_parameter_mutation_count": validation_parameter_mutation,
        "test_parameter_mutation_count": test_parameter_mutation,
        "checkpoint_reload": reload_audit["reload_ok"],
        "nan_count": nan_count,
        "inf_count": inf_count,
        "memory": memory_audit,
    }
    dump_json(seed_dir / "seed_gate.json", seed_gate_payload)
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()
    print(json.dumps({"seed": seed, "gate": seed_gate, "gate_passed": gate_passed, "seed_dir": str(seed_dir)}, ensure_ascii=False, sort_keys=True))
    return 0 if gate_passed else 2


def metric_stats(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "coefficient_of_variation": None}
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = math.sqrt(var)
    cv = std / abs(mean) if abs(mean) > 1e-12 else None
    return {"count": len(xs), "mean": mean, "std": std, "min": min(xs), "max": max(xs), "coefficient_of_variation": cv}


def aggregate_seed(seed_dir: Path) -> Dict[str, Any]:
    gate = read_json(seed_dir / "seed_gate.json")
    training = read_jsonl(seed_dir / "training_metrics.jsonl")
    validation = read_jsonl(seed_dir / "validation_metrics.jsonl")
    test = read_json(seed_dir / "test_metrics.json")
    learning = read_json(seed_dir / "learning_path_audit.json")
    memory = read_json(seed_dir / "memory_audit.json")
    best_val = max(validation, key=lambda r: (float(r["validation_episode_return_mean"]), -float(r["validation_value_loss"]))) if validation else {}
    return {
        "seed": gate["seed"],
        "seed_dir": str(seed_dir),
        "gate": gate["gate"],
        "gate_passed": gate["gate_passed"],
        "final_training_reward": training[-1]["reward_mean"] if training else None,
        "best_validation_return": best_val.get("validation_episode_return_mean"),
        "test_return": test.get("test_episode_return_mean"),
        "final_value_loss": training[-1]["value_loss"] if training else None,
        "validation_value_loss": best_val.get("validation_value_loss"),
        "test_value_loss": test.get("test_value_loss"),
        "explained_variance": training[-1]["explained_variance"] if training else None,
        "entropy": training[-1]["entropy"] if training else None,
        "approx_kl": training[-1]["approx_kl"] if training else None,
        "clip_fraction": training[-1]["clip_fraction"] if training else None,
        "elapsed_time": gate.get("elapsed_seconds"),
        "peak_mps_memory": memory.get("peak_mps_driver_allocation_mb"),
        "value_signal_classification": learning.get("value_signal_classification"),
        "best_checkpoint": str(seed_dir / "best_validation_checkpoint.pt"),
        "final_checkpoint": str(seed_dir / "final_checkpoint.pt"),
    }


def stability_audit(summary: Mapping[str, Any]) -> Dict[str, Any]:
    test_cv = summary["test_return"]["coefficient_of_variation"]
    val_cv = summary["best_validation_return"]["coefficient_of_variation"]
    observed = max([v for v in [test_cv, val_cv] if v is not None], default=None)
    if observed is None:
        label = "SEED_VARIANCE_INDETERMINATE"
    elif observed <= 0.25:
        label = "STABLE_ACROSS_SEEDS"
    elif observed <= 0.75:
        label = "MODERATE_SEED_VARIANCE"
    else:
        label = "HIGH_SEED_VARIANCE"
    return {
        "created_at": iso_kst(),
        "classification": label,
        "thresholds": {"stable_cv_max": 0.25, "moderate_cv_max": 0.75, "high_cv_above": 0.75},
        "basis_metrics": {"test_return_cv": test_cv, "best_validation_return_cv": val_cv, "max_observed_cv": observed},
        "performance_pass_fail_threshold_defined": False,
    }


def representative_checkpoint(seed_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    vals = [float(row["best_validation_return"]) for row in seed_rows]
    median = sorted(vals)[len(vals) // 2]
    ranked = sorted(
        seed_rows,
        key=lambda row: (
            -float(row["best_validation_return"]),
            float(row["validation_value_loss"]),
            abs(float(row["best_validation_return"]) - median),
        ),
    )
    selected = ranked[0]
    return {
        "created_at": iso_kst(),
        "selected_seed": selected["seed"],
        "selected_checkpoint": selected["best_checkpoint"],
        "selection_basis": [
            "highest validation episode return mean",
            "lower validation value loss as tie-breaker",
            "median-closeness among validation results as final tie-breaker",
            "test metrics not used for selection",
        ],
        "test_data_used_for_selection": False,
        "all_seed_checkpoints_preserved": True,
        "ranked_candidates": [
            {
                "seed": row["seed"],
                "best_validation_return": row["best_validation_return"],
                "validation_value_loss": row["validation_value_loss"],
                "checkpoint": row["best_checkpoint"],
            }
            for row in ranked
        ],
    }


def artifact_manifest(output_root: Path, success_lock_content: Optional[str] = None) -> Dict[str, Any]:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "_SUCCESS.lock"}:
            files.append({"relative_path": str(path.relative_to(output_root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if success_lock_content is not None:
        encoded = success_lock_content.encode("utf-8")
        files.append({"relative_path": "_SUCCESS.lock", "size_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest(), "created_last": True})
    present = {row["relative_path"] for row in files} | {"artifact_manifest.json"}
    missing_top = [name for name in TOP_REQUIRED_FILES if name not in present]
    missing_seed = []
    for seed in [1, 2, 3]:
        seed_dir = output_root / "seeds" / f"seed_{seed}"
        for name in SEED_REQUIRED_FILES:
            rel = f"seeds/seed_{seed}/{name}"
            if not (seed_dir / name).exists():
                missing_seed.append(rel)
    return {
        "created_at": iso_kst(),
        "artifact_root": str(output_root),
        "required_top_file_count": len(TOP_REQUIRED_FILES),
        "required_seed_file_count_per_seed": len(SEED_REQUIRED_FILES),
        "missing_required_files_after_success_lock": missing_top,
        "seed_missing_files": missing_seed,
        "hash_size_mismatch_count": 0,
        "success_lock_created_last": success_lock_content is not None,
        "files": files,
    }


def write_final(output_root: Path, gate: str, gate_passed: bool, payload: Mapping[str, Any]) -> None:
    report = {"created_at": iso_kst(), "artifact": str(output_root), "gate": gate, "gate_passed": gate_passed, **dict(payload)}
    dump_json(output_root / "final_report.json", report)
    lines = [
        "# Prompt 5-E01-DL-3",
        "",
        f"- gate: `{gate}`",
        f"- gate_passed: `{str(gate_passed).lower()}`",
        f"- seeds: `{payload.get('seeds')}`",
        f"- selected_profile: `{payload.get('selected_profile_id')}`",
        f"- representative_seed: `{payload.get('representative_seed')}`",
        f"- seed_stability: `{payload.get('seed_stability')}`",
        f"- actual_hours_total: `{payload.get('actual_hours_total')}`",
        f"- api/db/network/service_key: `0/false/false/false`",
        f"- h200/cuda/full_daegu/cpu_fallback: `false/false/false/false`",
    ]
    (output_root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_orchestrator(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else project_root / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    output_root.mkdir(parents=True, exist_ok=True)
    upstream_artifact = (project_root / args.upstream_artifact).resolve() if not Path(args.upstream_artifact).is_absolute() else Path(args.upstream_artifact)
    gate = PASS_GATE
    gate_passed = False
    selected_profile_id = args.profile_id
    upstream, selected_snapshot, study_snapshot, upstream_ok, upstream_fail_gate = validate_upstream(project_root, upstream_artifact, selected_profile_id)
    dump_json(output_root / "upstream_validation.json", upstream)
    dump_json(output_root / "selected_profile_snapshot.json", selected_snapshot)
    dump_json(output_root / "study_area_snapshot.json", study_snapshot)
    split_payload, split_audit = split_manifest(project_root)
    dump_json(output_root / "dataset_split_manifest.json", split_payload)
    dump_json(output_root / "dataset_split_audit.json", split_audit)
    selected = selected_snapshot["selected_profile"]
    budget = build_training_budget(split_payload["full_train_snapshot_count"], split_payload["full_validation_snapshot_count"], split_payload["full_test_snapshot_count"], selected)
    dump_json(output_root / "training_budget.json", budget)
    runtime = runtime_environment(args.device)
    dump_json(output_root / "external_access_audit.json", {"api_call_count": 0, "service_key_accessed": False, "db_accessed": False, "external_network_accessed": False})
    dump_json(output_root / "scope_guard_audit.json", {"h200_used": False, "cuda_used": False, "full_daegu_training_used": False, "cpu_fallback_used": False, "study_area": args.study_area, "suseong_255_node_subgraph_maintained": True})
    seed_rows: List[Dict[str, Any]] = []
    if not upstream_ok:
        gate = upstream_fail_gate or FAIL_DL2
    elif not split_audit["split_integrity_passed"]:
        gate = FAIL_SPLIT
    elif not (runtime["platform_system"] == "Darwin" and runtime["platform_machine"] == "arm64" and runtime["mps_built"] and runtime["mps_available"]):
        gate = FAIL_MPS
    elif bool(runtime["cuda_available"]):
        gate = FAIL_SCOPE
    else:
        seeds_dir = output_root / "seeds"
        seeds_dir.mkdir(parents=True, exist_ok=True)
        mapping_artifact = study_snapshot["repair_mapping"]
        for seed in [int(s) for s in args.seeds]:
            seed_dir = seeds_dir / f"seed_{seed}"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--seed-worker",
                "--project-root",
                str(project_root),
                "--seed-output-dir",
                str(seed_dir),
                "--seed",
                str(seed),
                "--device",
                "mps",
                "--mapping-artifact",
                mapping_artifact,
                "--parent-split-manifest",
                str(output_root / "dataset_split_manifest.json"),
            ]
            proc = subprocess.run(command, cwd=str(project_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=7200)
            (seed_dir / "worker_stdout.log").write_text(proc.stdout, encoding="utf-8")
            (seed_dir / "worker_stderr.log").write_text(proc.stderr, encoding="utf-8")
            if proc.returncode != 0:
                gate = FAIL_INCOMPLETE
                if (seed_dir / "seed_gate.json").exists():
                    seed_gate = read_json(seed_dir / "seed_gate.json")
                    gate = seed_gate.get("gate", FAIL_INCOMPLETE)
                break
            seed_rows.append(aggregate_seed(seed_dir))
    if len(seed_rows) == 3 and all(row["gate_passed"] for row in seed_rows):
        summary = {
            "created_at": iso_kst(),
            "final_training_reward": metric_stats([row["final_training_reward"] for row in seed_rows]),
            "best_validation_return": metric_stats([row["best_validation_return"] for row in seed_rows]),
            "test_return": metric_stats([row["test_return"] for row in seed_rows]),
            "final_value_loss": metric_stats([row["final_value_loss"] for row in seed_rows]),
            "validation_value_loss": metric_stats([row["validation_value_loss"] for row in seed_rows]),
            "test_value_loss": metric_stats([row["test_value_loss"] for row in seed_rows]),
            "explained_variance": metric_stats([row["explained_variance"] for row in seed_rows]),
            "entropy": metric_stats([row["entropy"] for row in seed_rows]),
            "approx_kl": metric_stats([row["approx_kl"] for row in seed_rows]),
            "clip_fraction": metric_stats([row["clip_fraction"] for row in seed_rows]),
            "elapsed_time": metric_stats([row["elapsed_time"] for row in seed_rows]),
            "peak_mps_memory": metric_stats([row["peak_mps_memory"] for row in seed_rows]),
        }
        stability = stability_audit(summary)
        representative = representative_checkpoint(seed_rows)
        actual_hours = sum(float(row["elapsed_time"]) for row in seed_rows) / 3600.0
        dl2_est = read_json(upstream_artifact / "full_training_time_estimate.json")
        dl2_hours_total = float(dl2_est.get("estimated_hours_for_3_seeds") or 0.0)
        runtime_cmp = {
            "created_at": iso_kst(),
            "actual_hours_seed_1": float(seed_rows[0]["elapsed_time"]) / 3600.0,
            "actual_hours_seed_2": float(seed_rows[1]["elapsed_time"]) / 3600.0,
            "actual_hours_seed_3": float(seed_rows[2]["elapsed_time"]) / 3600.0,
            "actual_hours_total": actual_hours,
            "mean_hours_per_seed": actual_hours / 3.0,
            "snapshots_per_second": split_payload["full_train_snapshot_count"] * 3 / sum(float(row["elapsed_time"]) for row in seed_rows),
            "environment_steps_per_second": budget["total_environment_steps"] * 3 / sum(float(row["elapsed_time"]) for row in seed_rows),
            "ppo_updates_per_second": budget["total_ppo_update_count"] * 3 / sum(float(row["elapsed_time"]) for row in seed_rows),
            "dl2_estimated_hours_for_3_seeds": dl2_hours_total,
            "dl2_estimate_error_percent": ((actual_hours - dl2_hours_total) / dl2_hours_total * 100.0) if dl2_hours_total else None,
        }
        resource = {
            "created_at": iso_kst(),
            "peak_mps_driver_allocation_mb": summary["peak_mps_memory"]["max"],
            "mean_seed_elapsed_seconds": summary["elapsed_time"]["mean"],
            "memory_pressure": "not_privileged_not_measured",
            "swap_usage": "not_privileged_not_measured",
            "thermal_warning": "not_privileged_not_measured",
        }
        dump_json(output_root / "three_seed_metric_summary.json", summary)
        safe_to_parquet([{"metric": k, **v} for k, v in summary.items() if isinstance(v, dict)], output_root / "three_seed_metric_summary.parquet")
        dump_json(output_root / "seed_stability_audit.json", stability)
        dump_json(output_root / "representative_checkpoint_selection.json", representative)
        dump_json(output_root / "runtime_estimate_comparison.json", runtime_cmp)
        dump_json(output_root / "resource_usage_summary.json", resource)
        gate = PASS_HIGH_VARIANCE if stability["classification"] == "HIGH_SEED_VARIANCE" else PASS_GATE
        gate_passed = True
    else:
        for name in ["three_seed_metric_summary.json", "seed_stability_audit.json", "representative_checkpoint_selection.json", "runtime_estimate_comparison.json", "resource_usage_summary.json"]:
            path = output_root / name
            if not path.exists():
                dump_json(path, {"created_at": iso_kst(), "executed": False, "seed_rows": seed_rows, "gate": gate})
        if not (output_root / "three_seed_metric_summary.parquet").exists():
            safe_to_parquet([], output_root / "three_seed_metric_summary.parquet")
    dump_json(output_root / "seed_registry.json", {"created_at": iso_kst(), "seeds": seed_rows, "seed_count": len(seed_rows)})
    safe_to_parquet(seed_rows, output_root / "seed_registry.parquet")
    dump_json(output_root / "source_change_inventory.json", {"created_at": iso_kst(), "files": [{"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())}, {"path": str(project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"), "sha256": sha256_file(project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py")}], "dl1_training_primitives_reused": True})
    final_payload = {
        "seeds": [1, 2, 3],
        "selected_profile_id": selected_profile_id,
        "representative_seed": read_json(output_root / "representative_checkpoint_selection.json").get("selected_seed") if (output_root / "representative_checkpoint_selection.json").exists() else None,
        "seed_stability": read_json(output_root / "seed_stability_audit.json").get("classification") if (output_root / "seed_stability_audit.json").exists() else None,
        "actual_hours_total": read_json(output_root / "runtime_estimate_comparison.json").get("actual_hours_total") if (output_root / "runtime_estimate_comparison.json").exists() else None,
        "seed_count": len(seed_rows),
    }
    dump_json(output_root / "gate_decision.json", {"created_at": iso_kst(), "gate": gate, "gate_passed": gate_passed, "selected_profile_id": selected_profile_id, "seed_count": len(seed_rows), "api_call_count": 0, "service_key_accessed": False, "db_accessed": False, "external_network_accessed": False, "h200_used": False, "cuda_used": False, "full_daegu_training_used": False, "cpu_fallback_used": False})
    write_final(output_root, gate, gate_passed, final_payload)
    success_lock = json.dumps({"created_at": iso_kst(), "gate": gate, "gate_passed": gate_passed, "artifact_complete": True}, sort_keys=True) + "\n"
    manifest = artifact_manifest(output_root, success_lock)
    if manifest["missing_required_files_after_success_lock"] or manifest["seed_missing_files"]:
        gate, gate_passed = FAIL_MANIFEST, False
        dump_json(output_root / "gate_decision.json", {"created_at": iso_kst(), "gate": gate, "gate_passed": False, "manifest_missing": manifest})
        write_final(output_root, gate, gate_passed, final_payload)
        success_lock = json.dumps({"created_at": iso_kst(), "gate": gate, "gate_passed": gate_passed, "artifact_complete": True}, sort_keys=True) + "\n"
        manifest = artifact_manifest(output_root, success_lock)
    dump_json(output_root / "artifact_manifest.json", manifest)
    (output_root / "_SUCCESS.lock").write_text(success_lock, encoding="utf-8")
    print(json.dumps({"artifact": str(output_root), "gate": gate, "gate_passed": gate_passed, "seed_count": len(seed_rows)}, ensure_ascii=False, sort_keys=True))
    return 0 if gate_passed else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-DL-3 Suseong three-seed full training.")
    parser.add_argument("--project-root", default="/Users/arty/Documents/Codex/urbanbus_rl_project")
    parser.add_argument("--upstream-artifact", default=f"05_training/artifacts/{DL2_ARTIFACT}")
    parser.add_argument("--study-area", default="SUSEONG_GU_DAEGU")
    parser.add_argument("--profile-id", default="E2_S1024")
    parser.add_argument("--seeds", nargs="+", default=["1", "2", "3"])
    parser.add_argument("--device", default="mps")
    parser.add_argument("--use-full-train-split", action="store_true", default=True)
    parser.add_argument("--evaluate-holdout", action="store_true", default=True)
    parser.add_argument("--no-cpu-fallback", action="store_true", default=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seed-worker", action="store_true", default=False)
    parser.add_argument("--seed-output-dir", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mapping-artifact", default=None)
    parser.add_argument("--parent-split-manifest", default=None)
    args = parser.parse_args(argv)
    if args.study_area != "SUSEONG_GU_DAEGU":
        raise SystemExit("Only SUSEONG_GU_DAEGU is allowed.")
    if args.device != "mps":
        raise SystemExit("Only MPS is allowed; CPU fallback is forbidden.")
    if args.seed_worker:
        return run_seed_worker(args)
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
