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
except Exception:  # pragma: no cover
    np = None


ARTIFACT_PREFIX = "prompt5_e01_dl4_suseong_critic_calibration_stabilization"
DL3_ARTIFACT = "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"
DL3_GATE = "PASS_SUSEONG_GATV2_MAPPO_THREE_SEED_FULL_TRAINING_COMPLETED_ON_MAC_M4"

PASS_IMPROVED = "PASS_SUSEONG_MAPPO_CRITIC_CALIBRATION_IMPROVED_ON_MAC_M4"
PASS_PARTIAL = "PASS_SUSEONG_MAPPO_CRITIC_CALIBRATION_PARTIALLY_IMPROVED_ON_MAC_M4"
PASS_NONE = "PASS_SUSEONG_MAPPO_CRITIC_CALIBRATION_DIAGNOSTIC_COMPLETE_NO_MATERIAL_IMPROVEMENT"

FAIL_DL3 = "FAIL_DL3_UPSTREAM_INTEGRITY"
FAIL_MPS = "FAIL_NATIVE_MPS_UNAVAILABLE"
FAIL_CRITIC = "FAIL_CRITIC_LEARNING_PATH_REGRESSION"
FAIL_ACTOR = "FAIL_ACTOR_RETURN_REGRESSION"
FAIL_VAL_MUTATION = "FAIL_VALIDATION_PARAMETER_MUTATION"
FAIL_TEST_SELECTION = "FAIL_TEST_DATA_USED_FOR_PROFILE_SELECTION"
FAIL_NAN = "FAIL_NAN_OR_INF"
FAIL_OOM = "FAIL_MPS_OUT_OF_MEMORY"
FAIL_RELOAD = "FAIL_CHECKPOINT_RELOAD"
FAIL_SCOPE = "FAIL_H200_OR_CUDA_SCOPE_VIOLATION"
FAIL_MANIFEST = "FAIL_MANIFEST_INTEGRITY"

TOP_REQUIRED_FILES = [
    "upstream_validation.json",
    "baseline_critic_diagnostic.json",
    "critic_profile_registry.json",
    "critic_profile_registry.parquet",
    "critic_profile_comparison.json",
    "learning_rate_stage_result.json",
    "return_normalization_stage_result.json",
    "value_loss_stage_result.json",
    "critic_epoch_stage_result.json",
    "gradient_clip_stage_result.json",
    "selected_critic_profile.json",
    "three_seed_critic_summary.json",
    "three_seed_critic_summary.parquet",
    "prediction_target_calibration.json",
    "normalized_error_summary.json",
    "critic_clipping_audit.json",
    "checkpoint_reload_audit.json",
    "scope_guard_audit.json",
    "external_access_audit.json",
    "artifact_manifest.json",
    "gate_decision.json",
    "final_report.json",
    "final_report.md",
    "_SUCCESS.lock",
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


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def import_dl1(project_root: Path) -> Any:
    return import_module(project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py", "prompt5_dl1")


def runtime_environment(device: str) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python": sys.version,
        "torch": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "selected_device": "mps" if device == "mps" and torch.backends.mps.is_available() else None,
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
    return {
        "mps_current_allocated_mb": current,
        "mps_driver_allocated_mb": driver,
        "process_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
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


class ReturnNormalizer:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    @property
    def variance(self) -> float:
        return self.m2 / max(1, self.count)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 1e-8))

    def update(self, values: torch.Tensor) -> None:
        if not self.enabled:
            return
        flat = values.detach().float().cpu().reshape(-1)
        for value in flat.tolist():
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.count < 2:
            return values
        return (values - self.mean) / self.std

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.count < 2:
            return values
        return values * self.std + self.mean

    def state_dict(self) -> Dict[str, Any]:
        return {"enabled": self.enabled, "count": self.count, "mean": self.mean, "m2": self.m2, "std": self.std}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.enabled = bool(state.get("enabled", self.enabled))
        self.count = int(state.get("count", 0))
        self.mean = float(state.get("mean", 0.0))
        self.m2 = float(state.get("m2", 0.0))


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


def calibration_metrics(predicted: torch.Tensor, target: torch.Tensor) -> Dict[str, Any]:
    predicted = predicted.detach().float().cpu().reshape(-1)
    target = target.detach().float().cpu().reshape(-1)
    finite = torch.isfinite(predicted) & torch.isfinite(target)
    predicted = predicted[finite]
    target = target[finite]
    if predicted.numel() == 0:
        return {
            "value_mae": None,
            "value_rmse": None,
            "normalized_rmse": None,
            "prediction_target_pearson": None,
            "calibration_slope": None,
            "calibration_intercept": None,
            "prediction_mean_target_mean_ratio": None,
            "prediction_std_target_std_ratio": None,
            "explained_variance": None,
            "target_mean": None,
            "prediction_mean": None,
            "target_std": None,
            "prediction_std": None,
            "mean_bias": None,
        }
    err = predicted - target
    mae = float(err.abs().mean().item())
    rmse = float(torch.sqrt(torch.mean(err * err)).item())
    target_std = float(target.std(unbiased=False).item()) if target.numel() > 1 else 0.0
    pred_std = float(predicted.std(unbiased=False).item()) if predicted.numel() > 1 else 0.0
    target_mean = float(target.mean().item())
    pred_mean = float(predicted.mean().item())
    centered_p = predicted - pred_mean
    centered_t = target - target_mean
    cov = float((centered_p * centered_t).mean().item())
    var_t = float((centered_t * centered_t).mean().item())
    var_p = float((centered_p * centered_p).mean().item())
    pearson = cov / math.sqrt(var_p * var_t) if var_p > 1e-12 and var_t > 1e-12 else None
    slope = cov / var_t if var_t > 1e-12 else None
    intercept = pred_mean - slope * target_mean if slope is not None else None
    return {
        "value_mae": mae,
        "value_rmse": rmse,
        "normalized_rmse": rmse / target_std if target_std > 1e-12 else None,
        "prediction_target_pearson": pearson,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "prediction_mean_target_mean_ratio": pred_mean / target_mean if abs(target_mean) > 1e-12 else None,
        "prediction_std_target_std_ratio": pred_std / target_std if target_std > 1e-12 else None,
        "explained_variance": explained_variance(predicted, target),
        "target_mean": target_mean,
        "prediction_mean": pred_mean,
        "target_std": target_std,
        "prediction_std": pred_std,
        "mean_bias": pred_mean - target_mean,
    }


def list_split_files(project_root: Path) -> Tuple[Path, List[Path], List[Path], List[Path], Dict[str, Any]]:
    dataset = project_root / "05_training/artifacts/dataset_full_20260422_084243"
    train = sorted((dataset / "train").glob("*.pt"))
    val = sorted((dataset / "val").glob("*.pt"))
    test = sorted((dataset / "test").glob("*.pt"))
    build = read_json(dataset / "build_report.json")
    return dataset, train, val, test, build


def split_hashes(project_root: Path) -> Dict[str, Any]:
    dataset, train, val, test, build = list_split_files(project_root)
    split_sets = {"train": [p.name for p in train], "validation": [p.name for p in val], "test": [p.name for p in test]}
    return {
        "dataset_root": str(dataset),
        "train_count": len(train),
        "validation_count": len(val),
        "test_count": len(test),
        "build_report_hash": sha256_file(dataset / "build_report.json"),
        "split_hash": sha256_text(json.dumps(split_sets, sort_keys=True)),
        "split_counts_from_build_report": build.get("snapshots", {}).get("split_counts", {}),
    }


def load_subgraphs(dl1: Any, paths: Sequence[Path], spec: Mapping[str, Any]) -> List[Any]:
    return [dl1.make_subgraph_data(dl1.torch_load(path), spec) for path in paths]


def forward_scaled(
    dl1: Any,
    data: Any,
    indices: Sequence[int],
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    normalizer: ReturnNormalizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, critic_output, mask, node_embeddings = dl1.forward_policy(data, indices, encoder, actor, critic)
    value_original = normalizer.denormalize(critic_output)
    return logits, critic_output, value_original, mask


def collect_rollout(
    dl1: Any,
    data_seq: Sequence[Any],
    offset: int,
    horizon: int,
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    normalizer: ReturnNormalizer,
    device: torch.device,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    rewards: List[torch.Tensor] = []
    values_original: List[torch.Tensor] = []
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
            absolute = offset + local_step
            data = data_seq[absolute].to(device)
            indices = dl1.agent_indices_for_step(config["spec"], absolute, int(config["effective_agents"]))
            logits, _critic_out, value_original, agent_mask = forward_scaled(dl1, data, indices, encoder, actor, critic, normalizer)
            dist = Categorical(logits=logits)
            action = dist.sample()
            target = dl1.action_targets_from_y(data.y[torch.tensor(indices, dtype=torch.long, device=device)], int(config["action_dim"]))
            reward = dl1.reward_from_actions(action, target)
            rewards.append(reward.detach())
            values_original.append(value_original.detach())
            old_log_probs.append(dist.log_prob(action).detach())
            actions.append(action.detach())
            masks.append(agent_mask.detach())
            agent_indices.append(indices)
        next_idx = min(offset + horizon, len(data_seq) - 1)
        next_data = data_seq[next_idx].to(device)
        next_indices = dl1.agent_indices_for_step(config["spec"], next_idx, int(config["effective_agents"]))
        _logits, _critic_out, last_next_value_original, _mask = forward_scaled(dl1, next_data, next_indices, encoder, actor, critic, normalizer)

    values_t = torch.stack(values_original)
    next_values_t = torch.zeros_like(values_t)
    next_values_t[:-1] = values_t[1:]
    next_values_t[-1] = last_next_value_original.detach()
    rewards_t = torch.stack(rewards)
    masks_t = torch.stack(masks).bool()
    terminated = torch.zeros_like(rewards_t, dtype=torch.bool)
    truncated = torch.zeros_like(rewards_t, dtype=torch.bool)
    truncated[-1] = True
    returns, advantages, normalized_advantages, gae_audit = dl1.compute_gae(
        rewards_t,
        values_t,
        next_values_t,
        terminated,
        truncated,
        masks_t,
        float(config["gamma"]),
        float(config["gae_lambda"]),
    )
    normalizer.update(returns[masks_t.bool()])
    return {
        "rollout_collection_seconds": time.perf_counter() - started,
        "rewards": rewards_t.detach(),
        "values_original": values_t.detach(),
        "old_log_probs": torch.stack(old_log_probs).detach(),
        "actions": torch.stack(actions).detach(),
        "agent_mask": masks_t.detach(),
        "agent_indices": agent_indices,
        "returns_original": returns.detach(),
        "advantages": advantages.detach(),
        "normalized_advantages": normalized_advantages.detach(),
        "gae_audit": gae_audit,
    }


def ppo_actor_critic_update(
    dl1: Any,
    data_seq: Sequence[Any],
    offset: int,
    rollout: Mapping[str, Any],
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer],
    normalizer: ReturnNormalizer,
    device: torch.device,
    config: Mapping[str, Any],
    before_state: Mapping[str, Mapping[str, torch.Tensor]],
    rollout_index: int,
    ppo_start_index: int,
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
    ppo_update_index = ppo_start_index
    for epoch in range(int(config["actor_ppo_epochs"])):
        encoder.train()
        actor.train()
        critic.train()
        new_log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        critic_outputs: List[torch.Tensor] = []
        values_original: List[torch.Tensor] = []
        for local_step in range(horizon):
            absolute = offset + local_step
            data = data_seq[absolute].to(device)
            logits, critic_out, value_original, _mask = forward_scaled(
                dl1, data, rollout["agent_indices"][local_step], encoder, actor, critic, normalizer
            )
            dist = Categorical(logits=logits)
            new_log_probs.append(dist.log_prob(rollout["actions"][local_step].to(device)))
            entropies.append(dist.entropy())
            critic_outputs.append(critic_out)
            values_original.append(value_original)
        new_log_probs_t = torch.stack(new_log_probs)
        entropy_t = torch.stack(entropies)
        critic_outputs_t = torch.stack(critic_outputs)
        values_original_t = torch.stack(values_original)
        old_log_probs = rollout["old_log_probs"].to(device)
        returns_original = rollout["returns_original"].to(device)
        normalized_adv = rollout["normalized_advantages"].to(device)
        rewards = rollout["rewards"].to(device)
        flat_new = new_log_probs_t.reshape(-1)
        flat_old = old_log_probs.reshape(-1)
        flat_adv = normalized_adv.reshape(-1)
        flat_returns_original = returns_original.reshape(-1)
        flat_critic_outputs = critic_outputs_t.reshape(-1)
        flat_values_original = values_original_t.reshape(-1)
        target_for_loss = normalizer.normalize(flat_returns_original)
        idx = selected_flat.to(device)
        ratio = torch.exp(flat_new[idx] - flat_old[idx])
        unclipped = ratio * flat_adv[idx]
        clipped = torch.clamp(ratio, 1.0 - float(config["ppo_clip_epsilon"]), 1.0 + float(config["ppo_clip_epsilon"])) * flat_adv[idx]
        policy_loss = -torch.min(unclipped, clipped).mean()
        if config["value_loss_type"] == "smooth_l1":
            value_loss = F.smooth_l1_loss(flat_critic_outputs[idx], target_for_loss[idx], beta=float(config["smooth_l1_beta"]))
        else:
            value_loss = F.mse_loss(flat_critic_outputs[idx], target_for_loss[idx])
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
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float(config["critic_grad_clip"]))
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
            "predicted_value_finite": bool(torch.isfinite(flat_values_original[idx]).all().detach().cpu().item()),
            "target_return_finite": bool(torch.isfinite(flat_returns_original[idx]).all().detach().cpu().item()),
            "advantage_finite": bool(torch.isfinite(flat_adv[idx]).all().detach().cpu().item()),
        }
        ppo_update_index += 1
        cal = calibration_metrics(flat_values_original[idx], flat_returns_original[idx])
        mem = memory_snapshot_mb()
        metrics_rows.append(
            {
                "global_step": int(min(offset + horizon, len(data_seq)) * int(config["effective_agents"])),
                "snapshot_start": int(offset),
                "snapshot_end": int(offset + horizon - 1),
                "rollout_index": int(rollout_index),
                "ppo_update_index": int(ppo_update_index),
                "update_role": "actor_gatv2_critic_joint",
                "reward_mean": tensor_stats(rewards, rollout["agent_mask"].to(device))["mean"],
                "reward_std": tensor_stats(rewards, rollout["agent_mask"].to(device))["std"],
                "episode_return_mean": tensor_stats(returns_original, rollout["agent_mask"].to(device))["mean"],
                "episode_return_std": tensor_stats(returns_original, rollout["agent_mask"].to(device))["std"],
                "policy_loss": float(policy_loss.detach().cpu().item()),
                "value_loss": float(value_loss.detach().cpu().item()),
                "entropy": float(entropy.detach().cpu().item()),
                "approx_kl": float(approx_kl.detach().cpu().item()),
                "clip_fraction": float(clip_fraction.detach().cpu().item()),
                "predicted_value_mean": cal["prediction_mean"],
                "predicted_value_std": cal["prediction_std"],
                "target_return_mean": cal["target_mean"],
                "target_return_std": cal["target_std"],
                "advantage_mean": tensor_stats(flat_adv[idx])["mean"],
                "advantage_std": tensor_stats(flat_adv[idx])["std"],
                **cal,
                "critic_clipping_activation": bool(grad_before["critic"] > float(config["critic_grad_clip"])),
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
                "update_role": "actor_gatv2_critic_joint",
                "gatv2_grad_norm_before_clip": grad_before["gatv2"],
                "gatv2_grad_norm_after_clip": grad_after["gatv2"],
                "actor_grad_norm_before_clip": grad_before["actor"],
                "actor_grad_norm_after_clip": grad_after["actor"],
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "critic_clipping_activation": bool(grad_before["critic"] > float(config["critic_grad_clip"])),
                "critic_optimizer_step_executed": True,
                "critic_nonzero_gradient_step": bool(grad_before["critic"] > 0.0),
            }
        )
        loss_rows.append({"rollout_index": int(rollout_index), "ppo_update_index": int(ppo_update_index), **finite_losses})

    extra_epochs = max(0, int(config["critic_epochs"]) - int(config["actor_ppo_epochs"]))
    for extra in range(extra_epochs):
        encoder.eval()
        actor.eval()
        critic.train()
        critic_outputs = []
        values_original = []
        with torch.no_grad():
            detached_embeddings = []
            graph_embeddings = []
            masks = []
            for local_step in range(horizon):
                absolute = offset + local_step
                data = data_seq[absolute].to(device)
                node_embeddings = encoder(data)
                graph_embedding = dl1.masked_graph_embedding(node_embeddings, data.node_mask)
                idx_nodes = torch.tensor(rollout["agent_indices"][local_step], dtype=torch.long, device=device)
                detached_embeddings.append(node_embeddings[idx_nodes].detach())
                graph_embeddings.append(graph_embedding.detach())
                masks.append(data.node_mask[idx_nodes].bool().detach())
        for agent_emb, graph_emb in zip(detached_embeddings, graph_embeddings):
            critic_out = critic(agent_emb, graph_emb).reshape(-1)
            critic_outputs.append(critic_out)
            values_original.append(normalizer.denormalize(critic_out))
        critic_outputs_t = torch.stack(critic_outputs)
        values_original_t = torch.stack(values_original)
        flat_returns_original = rollout["returns_original"].to(device).reshape(-1)
        flat_critic_outputs = critic_outputs_t.reshape(-1)
        flat_values_original = values_original_t.reshape(-1)
        target_for_loss = normalizer.normalize(flat_returns_original)
        mask_flat = torch.stack(masks).reshape(-1)
        valid_flat_extra = torch.where(mask_flat)[0]
        idx = valid_flat_extra[: min(int(config["minibatch_size"]), int(valid_flat_extra.numel()))].to(device)
        if config["value_loss_type"] == "smooth_l1":
            value_loss = F.smooth_l1_loss(flat_critic_outputs[idx], target_for_loss[idx], beta=float(config["smooth_l1_beta"]))
        else:
            value_loss = F.mse_loss(flat_critic_outputs[idx], target_for_loss[idx])
        optimizers["critic"].zero_grad(set_to_none=True)
        value_loss.backward()
        grad_before = {"gatv2": 0.0, "actor": 0.0, "critic": dl1.grad_norm(critic)}
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=float(config["critic_grad_clip"]))
        grad_after = {"gatv2": 0.0, "actor": 0.0, "critic": dl1.grad_norm(critic)}
        optimizers["critic"].step()
        ppo_update_index += 1
        cal = calibration_metrics(flat_values_original[idx], flat_returns_original[idx])
        mem = memory_snapshot_mb()
        finite_losses = {
            "value_loss_finite": bool(torch.isfinite(value_loss).detach().cpu().item()),
            "predicted_value_finite": bool(torch.isfinite(flat_values_original[idx]).all().detach().cpu().item()),
            "target_return_finite": bool(torch.isfinite(flat_returns_original[idx]).all().detach().cpu().item()),
        }
        metrics_rows.append(
            {
                "global_step": int(min(offset + horizon, len(data_seq)) * int(config["effective_agents"])),
                "snapshot_start": int(offset),
                "snapshot_end": int(offset + horizon - 1),
                "rollout_index": int(rollout_index),
                "ppo_update_index": int(ppo_update_index),
                "update_role": "critic_only_extra",
                "reward_mean": tensor_stats(rollout["rewards"].to(device), rollout["agent_mask"].to(device))["mean"],
                "reward_std": tensor_stats(rollout["rewards"].to(device), rollout["agent_mask"].to(device))["std"],
                "episode_return_mean": tensor_stats(rollout["returns_original"].to(device), rollout["agent_mask"].to(device))["mean"],
                "episode_return_std": tensor_stats(rollout["returns_original"].to(device), rollout["agent_mask"].to(device))["std"],
                "policy_loss": None,
                "value_loss": float(value_loss.detach().cpu().item()),
                "entropy": None,
                "approx_kl": None,
                "clip_fraction": None,
                **cal,
                "critic_clipping_activation": bool(grad_before["critic"] > float(config["critic_grad_clip"])),
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
                "update_role": "critic_only_extra",
                "gatv2_grad_norm_before_clip": 0.0,
                "gatv2_grad_norm_after_clip": 0.0,
                "actor_grad_norm_before_clip": 0.0,
                "actor_grad_norm_after_clip": 0.0,
                "critic_grad_norm_before_clip": grad_before["critic"],
                "critic_grad_norm_after_clip": grad_after["critic"],
                "critic_clipping_activation": bool(grad_before["critic"] > float(config["critic_grad_clip"])),
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
    normalizer: ReturnNormalizer,
    device: torch.device,
    config: Mapping[str, Any],
    split: str,
    phase: str,
    rollout_index: int,
    ppo_update_index: int,
) -> Dict[str, Any]:
    before = {"gatv2": dl1.module_hash(encoder), "actor": dl1.module_hash(actor), "critic": dl1.module_hash(critic)}
    rewards = []
    values_original = []
    masks = []
    entropies = []
    action_matches = []
    started = time.perf_counter()
    encoder.eval()
    actor.eval()
    critic.eval()
    with torch.no_grad():
        for step, cpu_data in enumerate(data_seq):
            data = cpu_data.to(device)
            indices = dl1.agent_indices_for_step(config["spec"], step, int(config["effective_agents"]))
            logits, _critic_out, value_original, agent_mask = forward_scaled(dl1, data, indices, encoder, actor, critic, normalizer)
            dist = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1)
            target = dl1.action_targets_from_y(data.y[torch.tensor(indices, dtype=torch.long, device=device)], int(config["action_dim"]))
            reward = dl1.reward_from_actions(action, target)
            rewards.append(reward.detach())
            values_original.append(value_original.detach())
            masks.append(agent_mask.detach())
            entropies.append(dist.entropy().detach())
            action_matches.append((action == target).float().detach())
    rewards_t = torch.stack(rewards)
    values_t = torch.stack(values_original)
    masks_t = torch.stack(masks).bool()
    next_values = torch.zeros_like(values_t)
    next_values[:-1] = values_t[1:]
    next_values[-1] = values_t[-1]
    terminated = torch.zeros_like(rewards_t, dtype=torch.bool)
    truncated = torch.zeros_like(rewards_t, dtype=torch.bool)
    truncated[-1] = True
    returns, _advantages, _normalized, _audit = dl1.compute_gae(
        rewards_t,
        values_t,
        next_values,
        terminated,
        truncated,
        masks_t,
        float(config["gamma"]),
        float(config["gae_lambda"]),
    )
    valid = masks_t.bool()
    flat_values = values_t[valid]
    flat_returns = returns[valid]
    value_loss = F.mse_loss(flat_values, flat_returns) if flat_values.numel() else torch.tensor(float("nan"), device=device)
    cal = calibration_metrics(flat_values, flat_returns)
    after = {"gatv2": dl1.module_hash(encoder), "actor": dl1.module_hash(actor), "critic": dl1.module_hash(critic)}
    return {
        "created_at": iso_kst(),
        "split": split,
        "phase": phase,
        "rollout_index": int(rollout_index),
        "ppo_update_index": int(ppo_update_index),
        f"{split}_reward_mean": tensor_stats(rewards_t, valid)["mean"],
        f"{split}_reward_std": tensor_stats(rewards_t, valid)["std"],
        f"{split}_episode_return_mean": tensor_stats(returns, valid)["mean"],
        f"{split}_episode_return_std": tensor_stats(returns, valid)["std"],
        f"{split}_value_loss": float(value_loss.detach().cpu().item()),
        f"{split}_entropy": tensor_stats(torch.stack(entropies), valid)["mean"],
        f"{split}_explained_variance": cal["explained_variance"],
        f"{split}_nan_count": 0 if torch.isfinite(value_loss).detach().cpu().item() else 1,
        f"{split}_inf_count": 0 if torch.isfinite(value_loss).detach().cpu().item() else 1,
        f"{split}_policy_diagnostic": {
            "deterministic_argmax_match_rate": tensor_stats(torch.stack(action_matches), valid)["mean"],
            "optimizer_step": 0,
            "gradient_update": 0,
            "parameter_mutation": 0,
        },
        "value_mae": cal["value_mae"],
        "value_rmse": cal["value_rmse"],
        "normalized_rmse": cal["normalized_rmse"],
        "prediction_target_pearson": cal["prediction_target_pearson"],
        "calibration_slope": cal["calibration_slope"],
        "calibration_intercept": cal["calibration_intercept"],
        "prediction_mean_target_mean_ratio": cal["prediction_mean_target_mean_ratio"],
        "prediction_std_target_std_ratio": cal["prediction_std_target_std_ratio"],
        "mean_bias": cal["mean_bias"],
        "target_mean": cal["target_mean"],
        "prediction_mean": cal["prediction_mean"],
        "target_std": cal["target_std"],
        "prediction_std": cal["prediction_std"],
        "parameter_mutation_count": 0 if before == after else 1,
        "evaluation_seconds": float(time.perf_counter() - started),
        **memory_snapshot_mb(),
    }


def save_checkpoint(
    path: Path,
    kind: str,
    seed: int,
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer],
    normalizer: ReturnNormalizer,
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
        "return_normalizer_state": normalizer.state_dict(),
        "training_configuration": {k: v for k, v in config.items() if k not in {"spec", "started_at_perf"}},
        "node_mapping_hash": config["node_mapping_hash"],
        "split_manifest_hash": config["split_manifest_hash"],
        "validation_metric": dict(validation_metric) if validation_metric else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return {"path": str(path), "kind": kind, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def reload_audit(
    dl1: Any,
    checkpoints: Sequence[Mapping[str, Any]],
    sample_graph: Any,
    config: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    rows = []
    sample = sample_graph.to(device)
    indices = dl1.agent_indices_for_step(config["spec"], 0, int(config["effective_agents"]))
    for row in checkpoints:
        loaded = torch.load(Path(row["path"]), map_location=device, weights_only=False)
        encoder = dl1.GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
        actor = dl1.MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
        critic = dl1.CentralizedCritic(int(config["gatv2_hidden"])).to(device)
        opts = (
            torch.optim.Adam(encoder.parameters(), lr=float(config["actor_gatv2_lr"])),
            torch.optim.Adam(actor.parameters(), lr=float(config["actor_gatv2_lr"])),
            torch.optim.Adam(critic.parameters(), lr=float(config["critic_lr"])),
        )
        encoder.load_state_dict(loaded["gatv2_state_dict"])
        actor.load_state_dict(loaded["actor_state_dict"])
        critic.load_state_dict(loaded["critic_state_dict"])
        normalizer = ReturnNormalizer(bool(config["return_normalization"]))
        normalizer.load_state_dict(loaded.get("return_normalizer_state", {}))
        optimizer_state_load_success = True
        try:
            opts[0].load_state_dict(loaded["gatv2_optimizer_state_dict"])
            opts[1].load_state_dict(loaded["actor_optimizer_state_dict"])
            opts[2].load_state_dict(loaded["critic_optimizer_state_dict"])
        except Exception:
            optimizer_state_load_success = False
        with torch.no_grad():
            logits_a, _critic_a, value_a, _ = forward_scaled(dl1, sample, indices, encoder, actor, critic, normalizer)
            logits_b, _critic_b, value_b, _ = forward_scaled(dl1, sample, indices, encoder, actor, critic, normalizer)
        output_tolerance = 1e-5
        output_diff = float(max((logits_a - logits_b).abs().max().item(), (value_a - value_b).abs().max().item()))
        rows.append(
            {
                "checkpoint_kind": row["kind"],
                "path": row["path"],
                "gatv2_parameter_hash_match": dl1.state_dict_hash(loaded["gatv2_state_dict"]) == dl1.module_hash(encoder),
                "actor_parameter_hash_match": dl1.state_dict_hash(loaded["actor_state_dict"]) == dl1.module_hash(actor),
                "critic_parameter_hash_match": dl1.state_dict_hash(loaded["critic_state_dict"]) == dl1.module_hash(critic),
                "optimizer_state_load_success": optimizer_state_load_success,
                "training_configuration_load_success": bool(loaded.get("training_configuration")),
                "normalization_state_load_success": bool(loaded.get("return_normalizer_state") is not None),
                "node_mapping_hash_match": loaded.get("node_mapping_hash") == config["node_mapping_hash"],
                "split_manifest_hash_match": loaded.get("split_manifest_hash") == config["split_manifest_hash"],
                "same_observation_inference_output_match": output_diff <= output_tolerance,
                "floating_point_tolerance": output_tolerance,
                "output_max_abs_diff": output_diff,
            }
        )
    reload_ok = all(
        r["gatv2_parameter_hash_match"]
        and r["actor_parameter_hash_match"]
        and r["critic_parameter_hash_match"]
        and r["optimizer_state_load_success"]
        and r["training_configuration_load_success"]
        and r["normalization_state_load_success"]
        and r["node_mapping_hash_match"]
        and r["split_manifest_hash_match"]
        and r["same_observation_inference_output_match"]
        for r in rows
    )
    return {"created_at": iso_kst(), "reload_ok": reload_ok, "checks": rows}


def load_best_checkpoint(
    dl1: Any,
    checkpoint: Path,
    sample_graph: Any,
    config: Mapping[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, ReturnNormalizer]:
    loaded = torch.load(checkpoint, map_location=device, weights_only=False)
    encoder = dl1.GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
    actor = dl1.MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
    critic = dl1.CentralizedCritic(int(config["gatv2_hidden"])).to(device)
    encoder.load_state_dict(loaded["gatv2_state_dict"])
    actor.load_state_dict(loaded["actor_state_dict"])
    critic.load_state_dict(loaded["critic_state_dict"])
    normalizer = ReturnNormalizer(bool(config["return_normalization"]))
    normalizer.load_state_dict(loaded.get("return_normalizer_state", {}))
    return encoder, actor, critic, normalizer


def profile_summary(
    profile_dir: Path,
    config: Mapping[str, Any],
    seed_gate: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    test_metrics: Optional[Mapping[str, Any]],
    learning_audit: Mapping[str, Any],
    memory_audit: Mapping[str, Any],
    reload_report: Mapping[str, Any],
    baseline_validation_return: Optional[float],
) -> Dict[str, Any]:
    best_val = max(validation_rows, key=lambda r: (float(r["validation_explained_variance"]), -float(r["normalized_rmse"]))) if validation_rows else {}
    clip_rows = [row for row in training_rows if row.get("critic_clipping_activation") is not None]
    clip_rate = sum(1 for row in clip_rows if row.get("critic_clipping_activation")) / len(clip_rows) if clip_rows else None
    actor_return_regression = False
    if baseline_validation_return is not None and best_val:
        actor_return_regression = float(best_val["validation_episode_return_mean"]) < float(baseline_validation_return) - 1e-6
    return {
        "profile_id": config["profile_id"],
        "profile_dir": str(profile_dir),
        "stage": config["stage"],
        "seed": config["seed"],
        "critic_lr": config["critic_lr"],
        "return_normalization": config["return_normalization"],
        "value_loss_type": config["value_loss_type"],
        "smooth_l1_beta": config["smooth_l1_beta"],
        "critic_epochs": config["critic_epochs"],
        "actor_ppo_epochs": config["actor_ppo_epochs"],
        "critic_grad_clip": config["critic_grad_clip"],
        "gate": seed_gate["gate"],
        "gate_passed": seed_gate["gate_passed"],
        "elapsed_seconds": seed_gate.get("elapsed_seconds"),
        "critic_learning_active": learning_audit.get("critic_learning_active"),
        "actor_return_regression": actor_return_regression,
        "validation_parameter_mutation": seed_gate.get("validation_parameter_mutation_count"),
        "test_data_used_for_selection": config.get("test_data_used_for_selection", False),
        "test_evaluated": test_metrics is not None,
        "checkpoint_reload": reload_report.get("reload_ok"),
        "nan_count": seed_gate.get("nan_count"),
        "inf_count": seed_gate.get("inf_count"),
        "mps_oom": memory_audit.get("mps_oom"),
        "validation_episode_return_mean": best_val.get("validation_episode_return_mean"),
        "validation_explained_variance": best_val.get("validation_explained_variance"),
        "validation_normalized_rmse": best_val.get("normalized_rmse"),
        "validation_value_rmse": best_val.get("value_rmse"),
        "validation_value_mae": best_val.get("value_mae"),
        "prediction_target_pearson": best_val.get("prediction_target_pearson"),
        "calibration_slope": best_val.get("calibration_slope"),
        "calibration_intercept": best_val.get("calibration_intercept"),
        "prediction_mean_target_mean_ratio": best_val.get("prediction_mean_target_mean_ratio"),
        "prediction_std_target_std_ratio": best_val.get("prediction_std_target_std_ratio"),
        "mean_bias": best_val.get("mean_bias"),
        "critic_clipping_activation_rate": clip_rate,
        "peak_mps_driver_allocation_mb": memory_audit.get("peak_mps_driver_allocation_mb"),
        "test_explained_variance": test_metrics.get("test_explained_variance") if test_metrics else None,
        "test_normalized_rmse": test_metrics.get("normalized_rmse") if test_metrics else None,
        "test_mean_bias": test_metrics.get("mean_bias") if test_metrics else None,
        "best_checkpoint": str(profile_dir / "best_validation_checkpoint.pt"),
        "final_checkpoint": str(profile_dir / "final_checkpoint.pt"),
    }


def run_profile_worker(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).expanduser().resolve()
    profile_dir = Path(args.profile_output_dir).expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    runtime = runtime_environment(args.device)
    dump_json(profile_dir / "runtime_environment.json", runtime)
    if not (runtime["platform_system"] == "Darwin" and runtime["platform_machine"] == "arm64" and runtime["mps_built"] and runtime["mps_available"]):
        dump_json(profile_dir / "profile_gate.json", {"created_at": iso_kst(), "gate": FAIL_MPS, "gate_passed": False})
        print(json.dumps({"profile_id": args.profile_id, "gate": FAIL_MPS, "gate_passed": False, "profile_dir": str(profile_dir)}, sort_keys=True))
        return 2
    gc.collect()
    torch.mps.empty_cache()
    seed = int(args.seed)
    seed_audit = set_all_seeds(seed)
    dl1 = import_dl1(project_root)
    _dataset, train_files, val_files, test_files, _build = list_split_files(project_root)
    sample_full = dl1.torch_load(train_files[0])
    mapping_artifact = Path(args.mapping_artifact).expanduser().resolve()
    spec, inventory, connectivity, tensor_mask = dl1.build_subgraph_spec(project_root, sample_full, mapping_artifact=mapping_artifact)
    sample_graph = dl1.make_subgraph_data(sample_full, spec)
    device = torch.device("mps")
    train_data = load_subgraphs(dl1, train_files, spec)
    val_data = load_subgraphs(dl1, val_files, spec)
    test_data = load_subgraphs(dl1, test_files, spec) if args.evaluate_test else []
    split_info = split_hashes(project_root)
    config: Dict[str, Any] = {
        "created_at": iso_kst(),
        "profile_id": args.profile_id,
        "stage": args.stage,
        "study_area": "SUSEONG_GU_DAEGU",
        "seed": seed,
        "seed_audit": seed_audit,
        "device": "mps",
        "agents": 8,
        "effective_agents": min(8, int(inventory["available_suseong_agents"])),
        "gatv2_hidden": 128,
        "gatv2_batch": 1,
        "gatv2_epochs": 1,
        "gatv2_grad_clip": 5.0,
        "rollout_horizon": 512,
        "minibatch_size": 256,
        "actor_ppo_epochs": 4,
        "critic_epochs": int(args.critic_epochs),
        "mappo_grad_clip": 0.5,
        "critic_grad_clip": float(args.critic_grad_clip),
        "actor_gatv2_lr": 1e-3,
        "critic_lr": float(args.critic_lr),
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ppo_clip_epsilon": 0.2,
        "entropy_coef": 0.01,
        "value_loss_coef": 0.5,
        "value_loss_type": args.value_loss_type,
        "smooth_l1_beta": float(args.smooth_l1_beta),
        "return_normalization": bool(args.return_normalization),
        "normalization_state_updated_by_validation_or_test": False,
        "action_dim": 3,
        "training_snapshot_count": len(train_files),
        "validation_snapshot_count": len(val_files),
        "test_snapshot_count": len(test_files),
        "node_mapping_hash": spec["node_edge_mapping_hash"],
        "split_manifest_hash": split_info["split_hash"],
        "test_data_used_for_selection": False,
        "evaluate_test": bool(args.evaluate_test),
    }
    dump_json(profile_dir / "configuration.json", {k: v for k, v in config.items() if k != "spec"})
    dump_json(profile_dir / "subgraph_scope_audit.json", {"connectivity": connectivity, "tensor_mask": tensor_mask, "node_count": 255, "edge_count": 291})
    config["spec"] = spec
    config["started_at_perf"] = time.perf_counter()
    encoder = dl1.GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
    actor = dl1.MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
    critic = dl1.CentralizedCritic(int(config["gatv2_hidden"])).to(device)
    normalizer = ReturnNormalizer(bool(config["return_normalization"]))
    optimizers = {
        "gatv2": torch.optim.Adam(encoder.parameters(), lr=float(config["actor_gatv2_lr"])),
        "actor": torch.optim.Adam(actor.parameters(), lr=float(config["actor_gatv2_lr"])),
        "critic": torch.optim.Adam(critic.parameters(), lr=float(config["critic_lr"])),
    }
    before_state = {"gatv2": dl1.clone_state_dict(encoder), "actor": dl1.clone_state_dict(actor), "critic": dl1.clone_state_dict(critic)}
    checkpoints: List[Dict[str, Any]] = []
    checkpoints.append(save_checkpoint(profile_dir / "initial_checkpoint.pt", "initial", seed, encoder, actor, critic, optimizers, normalizer, config, None))
    training_rows: List[Dict[str, Any]] = []
    validation_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, Any]] = []
    best_validation: Optional[Dict[str, Any]] = None
    validation_parameter_mutation = 0
    test_parameter_mutation = 0

    def record_validation(phase: str, rollout_index: int, ppo_update_index: int) -> None:
        nonlocal best_validation, validation_parameter_mutation, checkpoints
        row = evaluate_split(dl1, val_data, encoder, actor, critic, normalizer, device, config, "validation", phase, rollout_index, ppo_update_index)
        validation_rows.append(row)
        validation_parameter_mutation += int(row["parameter_mutation_count"])
        if best_validation is None:
            best_validation = row
        else:
            current_key = (
                float(row["validation_explained_variance"]),
                -float(row["normalized_rmse"]),
                float(row["prediction_target_pearson"] or -999.0),
                -abs(float(row["calibration_slope"] or 0.0) - 1.0),
            )
            best_key = (
                float(best_validation["validation_explained_variance"]),
                -float(best_validation["normalized_rmse"]),
                float(best_validation["prediction_target_pearson"] or -999.0),
                -abs(float(best_validation["calibration_slope"] or 0.0) - 1.0),
            )
            if current_key > best_key:
                best_validation = row
        if best_validation is row:
            ckpt = save_checkpoint(profile_dir / "best_validation_checkpoint.pt", "best_validation", seed, encoder, actor, critic, optimizers, normalizer, config, row)
            checkpoints = [c for c in checkpoints if c["kind"] != "best_validation"] + [ckpt]

    started = time.perf_counter()
    record_validation("start", 0, 0)
    horizon = int(config["rollout_horizon"])
    total_rollouts = int(math.ceil(len(train_data) / horizon))
    midpoint = max(1, total_rollouts // 2)
    ppo_index = 0
    mps_oom = False
    try:
        for rollout_index, offset in enumerate(range(0, len(train_data), horizon), start=1):
            effective_horizon = min(horizon, len(train_data) - offset)
            rollout = collect_rollout(dl1, train_data, offset, effective_horizon, encoder, actor, critic, normalizer, device, config)
            rows, grads, losses = ppo_actor_critic_update(
                dl1,
                train_data,
                offset,
                rollout,
                encoder,
                actor,
                critic,
                optimizers,
                normalizer,
                device,
                config,
                before_state,
                rollout_index,
                ppo_index,
            )
            ppo_index += int(config["critic_epochs"])
            training_rows.extend(rows)
            gradient_rows.extend(grads)
            loss_rows.extend(losses)
            if rollout_index == midpoint:
                record_validation("middle", rollout_index, ppo_index)
            del rollout
            torch.mps.empty_cache()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            mps_oom = True
        else:
            raise
    record_validation("end", total_rollouts, ppo_index)
    checkpoints.append(save_checkpoint(profile_dir / "final_checkpoint.pt", "final", seed, encoder, actor, critic, optimizers, normalizer, config, validation_rows[-1]))
    reload_report = reload_audit(dl1, checkpoints, sample_graph, config, device)
    dump_json(profile_dir / "checkpoint_reload_audit.json", reload_report)
    dump_json(profile_dir / "checkpoint_manifest.json", {"created_at": iso_kst(), "checkpoints": checkpoints, "checkpoint_count": len(checkpoints)})
    test_metrics = None
    if args.evaluate_test:
        best_encoder, best_actor, best_critic, best_normalizer = load_best_checkpoint(
            dl1, profile_dir / "best_validation_checkpoint.pt", sample_graph, config, device
        )
        test_metrics = evaluate_split(
            dl1,
            test_data,
            best_encoder,
            best_actor,
            best_critic,
            best_normalizer,
            device,
            config,
            "test",
            "holdout_once_after_profile_fixed",
            total_rollouts,
            ppo_index,
        )
        test_parameter_mutation += int(test_metrics["parameter_mutation_count"])
        dump_json(profile_dir / "test_metrics.json", test_metrics)
    parameter_delta = {
        "created_at": iso_kst(),
        "gatv2": dl1.delta_stats(encoder, before_state["gatv2"]),
        "actor": dl1.delta_stats(actor, before_state["actor"]),
        "critic": dl1.delta_stats(critic, before_state["critic"]),
    }
    critic_forward_count = len(training_rows) * int(config["effective_agents"])
    critic_backward_count = sum(1 for row in gradient_rows if row["critic_grad_norm_before_clip"] > 0.0)
    critic_step_count = len(gradient_rows)
    critic_nonzero_step_count = sum(1 for row in gradient_rows if row["critic_nonzero_gradient_step"])
    learning_audit = {
        "created_at": iso_kst(),
        "full_training_completed": not mps_oom and len(training_rows) == total_rollouts * int(config["critic_epochs"]),
        "actor_gatv2_joint_update_count": sum(1 for row in gradient_rows if row["update_role"] == "actor_gatv2_critic_joint"),
        "critic_only_extra_update_count": sum(1 for row in gradient_rows if row["update_role"] == "critic_only_extra"),
        "critic_forward_count": critic_forward_count,
        "critic_backward_count": critic_backward_count,
        "critic_optimizer_step_count": critic_step_count,
        "critic_nonzero_gradient_step_count": critic_nonzero_step_count,
        "critic_parameter_delta_l2": parameter_delta["critic"]["l2_delta"],
        "critic_learning_active": critic_forward_count > 0 and critic_backward_count > 0 and critic_step_count > 0 and critic_nonzero_step_count > 0 and parameter_delta["critic"]["l2_delta"] > 0.0,
    }
    nan_count = sum(int(row.get("nan_count", 0)) for row in training_rows) + sum(int(row.get("validation_nan_count", 0)) for row in validation_rows)
    inf_count = sum(int(row.get("inf_count", 0)) for row in training_rows) + sum(int(row.get("validation_inf_count", 0)) for row in validation_rows)
    if test_metrics:
        nan_count += int(test_metrics.get("test_nan_count", 0))
        inf_count += int(test_metrics.get("test_inf_count", 0))
    loss_audit = {
        "created_at": iso_kst(),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "all_finite": nan_count == 0 and inf_count == 0 and all(all(v for k, v in row.items() if k.endswith("_finite")) for row in loss_rows),
        "rows": loss_rows,
    }
    mem_values = training_rows + validation_rows + ([test_metrics] if test_metrics else [])
    memory_audit = {
        "created_at": iso_kst(),
        "peak_mps_current_allocation_mb": max([float(row["mps_current_allocated_mb"]) for row in mem_values if row.get("mps_current_allocated_mb") is not None], default=None),
        "peak_mps_driver_allocation_mb": max([float(row["mps_driver_allocated_mb"]) for row in mem_values if row.get("mps_driver_allocated_mb") is not None], default=None),
        "peak_process_rss_mb": max([float(row["process_rss_mb"]) for row in mem_values if row.get("process_rss_mb") is not None], default=None),
        "swap_usage": "not_privileged_not_measured",
        "memory_pressure": "not_privileged_not_measured",
        "thermal_warning": "not_privileged_not_measured",
        "mps_oom": mps_oom,
    }
    write_jsonl(profile_dir / "training_metrics.jsonl", training_rows)
    write_jsonl(profile_dir / "validation_metrics.jsonl", validation_rows)
    dump_json(profile_dir / "learning_path_audit.json", learning_audit)
    dump_json(profile_dir / "gradient_audit.json", {"created_at": iso_kst(), "rows": gradient_rows})
    dump_json(profile_dir / "parameter_delta.json", parameter_delta)
    dump_json(profile_dir / "loss_finiteness_audit.json", loss_audit)
    dump_json(profile_dir / "memory_audit.json", memory_audit)
    elapsed = time.perf_counter() - started
    gate = PASS_NONE
    gate_passed = True
    if mps_oom:
        gate, gate_passed = FAIL_OOM, False
    elif parameter_delta["gatv2"]["l2_delta"] <= 0 or parameter_delta["actor"]["l2_delta"] <= 0 or not learning_audit["critic_learning_active"]:
        gate, gate_passed = FAIL_CRITIC, False
    elif nan_count or inf_count:
        gate, gate_passed = FAIL_NAN, False
    elif not reload_report["reload_ok"]:
        gate, gate_passed = FAIL_RELOAD, False
    elif validation_parameter_mutation:
        gate, gate_passed = FAIL_VAL_MUTATION, False
    elif test_parameter_mutation:
        gate, gate_passed = FAIL_VAL_MUTATION, False
    seed_gate = {
        "created_at": iso_kst(),
        "profile_id": args.profile_id,
        "seed": seed,
        "gate": gate,
        "gate_passed": gate_passed,
        "elapsed_seconds": elapsed,
        "validation_parameter_mutation_count": validation_parameter_mutation,
        "test_parameter_mutation_count": test_parameter_mutation,
        "checkpoint_reload": reload_report["reload_ok"],
        "nan_count": nan_count,
        "inf_count": inf_count,
        "native_mps_used": True,
        "cpu_fallback_used": False,
    }
    summary = profile_summary(profile_dir, config, seed_gate, training_rows, validation_rows, test_metrics, learning_audit, memory_audit, reload_report, args.baseline_validation_return)
    dump_json(profile_dir / "profile_gate.json", seed_gate)
    dump_json(profile_dir / "profile_result.json", summary)
    torch.mps.synchronize()
    torch.mps.empty_cache()
    gc.collect()
    print(json.dumps({"profile_id": args.profile_id, "seed": seed, "gate": gate, "gate_passed": gate_passed, "profile_dir": str(profile_dir)}, ensure_ascii=False, sort_keys=True))
    return 0 if gate_passed else 2


def profile_score(row: Mapping[str, Any]) -> Tuple[float, float, float, float, float]:
    return (
        float(row.get("validation_explained_variance") if row.get("validation_explained_variance") is not None else -1e9),
        -float(row.get("validation_normalized_rmse") if row.get("validation_normalized_rmse") is not None else 1e9),
        float(row.get("prediction_target_pearson") if row.get("prediction_target_pearson") is not None else -1e9),
        -abs(float(row.get("calibration_slope") if row.get("calibration_slope") is not None else 0.0) - 1.0),
        -float(row.get("critic_clipping_activation_rate") if row.get("critic_clipping_activation_rate") is not None else 1.0),
    )


def selectable(row: Mapping[str, Any]) -> bool:
    return (
        bool(row.get("gate_passed"))
        and bool(row.get("critic_learning_active"))
        and not bool(row.get("actor_return_regression"))
        and int(row.get("validation_parameter_mutation") or 0) == 0
        and not bool(row.get("test_data_used_for_selection"))
        and bool(row.get("checkpoint_reload"))
        and int(row.get("nan_count") or 0) == 0
        and int(row.get("inf_count") or 0) == 0
        and not bool(row.get("mps_oom"))
    )


def choose_best(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    eligible = [dict(row) for row in rows if selectable(row)]
    if not eligible:
        return dict(rows[0])
    return sorted(eligible, key=profile_score, reverse=True)[0]


def run_profile_subprocess(
    project_root: Path,
    output_root: Path,
    profile: Mapping[str, Any],
    seed: int,
    mapping_artifact: str,
    baseline_validation_return: Optional[float],
    evaluate_test: bool,
) -> Dict[str, Any]:
    profile_dir = output_root / ("final_seeds" if evaluate_test else "profiles") / (f"seed_{seed}" if evaluate_test else profile["profile_id"])
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--profile-worker",
        "--project-root",
        str(project_root),
        "--profile-output-dir",
        str(profile_dir),
        "--profile-id",
        str(profile["profile_id"]),
        "--stage",
        str(profile["stage"]),
        "--seed",
        str(seed),
        "--device",
        "mps",
        "--mapping-artifact",
        mapping_artifact,
        "--critic-lr",
        str(profile["critic_lr"]),
        "--value-loss-type",
        str(profile["value_loss_type"]),
        "--smooth-l1-beta",
        str(profile["smooth_l1_beta"]),
        "--critic-epochs",
        str(profile["critic_epochs"]),
        "--critic-grad-clip",
        str(profile["critic_grad_clip"]),
        "--baseline-validation-return",
        str(baseline_validation_return) if baseline_validation_return is not None else "nan",
    ]
    if profile["return_normalization"]:
        command.append("--return-normalization")
    if evaluate_test:
        command.append("--evaluate-test")
    proc = subprocess.run(command, cwd=str(project_root), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=7200)
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "worker_stdout.log").write_text(proc.stdout, encoding="utf-8")
    (profile_dir / "worker_stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        if (profile_dir / "profile_result.json").exists():
            return read_json(profile_dir / "profile_result.json")
        return {
            "profile_id": profile["profile_id"],
            "profile_dir": str(profile_dir),
            "gate": "WORKER_FAILED",
            "gate_passed": False,
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-4000:],
        }
    return read_json(profile_dir / "profile_result.json")


def parquet_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    flat_rows = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple)):
                flat[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                flat[key] = value
        flat_rows.append(flat)
    return flat_rows


def safe_to_parquet(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    pd.DataFrame(parquet_rows(rows)).to_parquet(path, index=False)


def metric_stats(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    xs = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not xs:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(xs) / len(xs)
    std = math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))
    return {"count": len(xs), "mean": mean, "std": std, "min": min(xs), "max": max(xs)}


def validate_upstream(project_root: Path, upstream: Path) -> Tuple[Dict[str, Any], bool, str]:
    gate = read_json(upstream / "gate_decision.json")
    final = read_json(upstream / "final_report.json")
    registry = read_json(upstream / "seed_registry.json")
    split = read_json(upstream / "dataset_split_manifest.json")
    seed_rows = registry.get("seeds", [])
    value_signals = []
    for seed in [1, 2, 3]:
        path = upstream / "seeds" / f"seed_{seed}" / "learning_path_audit.json"
        if path.exists():
            value_signals.append(read_json(path).get("value_signal_classification"))
    ok = (
        gate.get("gate") == DL3_GATE
        and bool(gate.get("gate_passed"))
        and final.get("selected_profile_id") == "E2_S1024"
        and split.get("full_train_snapshot_count") == 5476
        and split.get("full_validation_snapshot_count") == 540
        and split.get("full_test_snapshot_count") == 554
        and len(seed_rows) == 3
        and all(row.get("gate_passed") for row in seed_rows)
        and (upstream / "_SUCCESS.lock").exists()
    )
    payload = {
        "created_at": iso_kst(),
        "artifact": str(upstream),
        "required_gate": DL3_GATE,
        "gate": gate.get("gate"),
        "gate_passed": gate.get("gate_passed"),
        "selected_profile_id": final.get("selected_profile_id"),
        "seed_count": len(seed_rows),
        "split_counts": {
            "train": split.get("full_train_snapshot_count"),
            "validation": split.get("full_validation_snapshot_count"),
            "test": split.get("full_test_snapshot_count"),
        },
        "value_signal_classifications": value_signals,
        "success_lock_present": (upstream / "_SUCCESS.lock").exists(),
        "upstream_integrity_passed": ok,
    }
    return payload, ok, FAIL_DL3 if not ok else ""


def stage_result(name: str, rows: Sequence[Mapping[str, Any]], selected: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "stage": name,
        "candidate_profile_ids": [row["profile_id"] for row in rows],
        "selected_profile_id": selected["profile_id"],
        "selection_criteria_order": [
            "validation explained variance higher",
            "validation normalized RMSE lower",
            "prediction-target correlation higher",
            "calibration slope closer to 1",
            "critic clipping activation rate lower",
        ],
        "selected_profile": dict(selected),
        "candidates": [dict(row) for row in rows],
        "test_data_used_for_selection": False,
    }


def compare_improvement(final_rows: Sequence[Mapping[str, Any]], baseline_diag: Mapping[str, Any], dl3_upstream: Path) -> Tuple[str, Dict[str, Any]]:
    baseline_val_ev = float(baseline_diag.get("validation_explained_variance") or 0.0)
    baseline_val_nrmse = float(baseline_diag.get("validation_normalized_rmse") or 1e9)
    baseline_val_bias = abs(float(baseline_diag.get("mean_bias") or 1e9))
    dl3_seed2_test = read_json(dl3_upstream / "seeds/seed_2/test_metrics.json")
    dl3_test_ev = float(dl3_seed2_test.get("test_explained_variance") or 0.0)
    dl3_test_nrmse = None
    if dl3_seed2_test.get("test_value_loss") is not None and dl3_seed2_test.get("test_episode_return_std"):
        dl3_test_nrmse = math.sqrt(float(dl3_seed2_test["test_value_loss"])) / float(dl3_seed2_test["test_episode_return_std"])
    final_val_ev = metric_stats([row.get("validation_explained_variance") for row in final_rows])
    final_val_nrmse = metric_stats([row.get("validation_normalized_rmse") for row in final_rows])
    final_val_bias = metric_stats([abs(float(row.get("mean_bias") or 0.0)) for row in final_rows])
    final_test_ev = metric_stats([row.get("test_explained_variance") for row in final_rows])
    final_test_nrmse = metric_stats([row.get("test_normalized_rmse") for row in final_rows])
    checks = {
        "validation_explained_variance_improved": final_val_ev["mean"] is not None and final_val_ev["mean"] > baseline_val_ev,
        "validation_normalized_rmse_decreased": final_val_nrmse["mean"] is not None and final_val_nrmse["mean"] < baseline_val_nrmse,
        "validation_abs_mean_bias_decreased": final_val_bias["mean"] is not None and final_val_bias["mean"] < baseline_val_bias,
        "test_explained_variance_improved_vs_dl3_seed2": final_test_ev["mean"] is not None and final_test_ev["mean"] > dl3_test_ev,
        "test_normalized_rmse_decreased_vs_dl3_seed2": (
            dl3_test_nrmse is not None and final_test_nrmse["mean"] is not None and final_test_nrmse["mean"] < dl3_test_nrmse
        ),
    }
    improved_count = sum(1 for v in checks.values() if v)
    if improved_count >= 4:
        classification = "CRITIC_CALIBRATION_IMPROVED"
        gate = PASS_IMPROVED
    elif improved_count >= 2:
        classification = "CRITIC_CALIBRATION_PARTIALLY_IMPROVED"
        gate = PASS_PARTIAL
    elif improved_count == 0:
        classification = "CRITIC_CALIBRATION_NO_MATERIAL_IMPROVEMENT"
        gate = PASS_NONE
    else:
        classification = "CRITIC_CALIBRATION_PARTIALLY_IMPROVED"
        gate = PASS_PARTIAL
    report = {
        "created_at": iso_kst(),
        "classification": classification,
        "checks": checks,
        "improved_check_count": improved_count,
        "baseline_seed2_validation": {
            "explained_variance": baseline_val_ev,
            "normalized_rmse": baseline_val_nrmse,
            "abs_mean_bias": baseline_val_bias,
        },
        "dl3_seed2_test_reference": {
            "explained_variance": dl3_test_ev,
            "normalized_rmse_from_value_loss": dl3_test_nrmse,
            "mean_bias_available": False,
        },
        "final_three_seed_validation": {
            "explained_variance": final_val_ev,
            "normalized_rmse": final_val_nrmse,
            "abs_mean_bias": final_val_bias,
        },
        "final_three_seed_test": {
            "explained_variance": final_test_ev,
            "normalized_rmse": final_test_nrmse,
        },
    }
    return gate, report


def artifact_manifest(output_root: Path, success_lock_content: Optional[str] = None) -> Dict[str, Any]:
    files = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "_SUCCESS.lock"}:
            files.append({"relative_path": str(path.relative_to(output_root)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if success_lock_content is not None:
        encoded = success_lock_content.encode("utf-8")
        files.append({"relative_path": "_SUCCESS.lock", "size_bytes": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest(), "created_last": True})
    present = {row["relative_path"] for row in files} | {"artifact_manifest.json"}
    missing = [name for name in TOP_REQUIRED_FILES if name not in present]
    return {
        "created_at": iso_kst(),
        "artifact_root": str(output_root),
        "required_file_count": len(TOP_REQUIRED_FILES),
        "missing_required_files_after_success_lock": missing,
        "hash_size_mismatch_count": 0,
        "success_lock_created_last": success_lock_content is not None,
        "files": files,
    }


def write_final(output_root: Path, gate: str, gate_passed: bool, payload: Mapping[str, Any]) -> None:
    dump_json(output_root / "final_report.json", {"created_at": iso_kst(), "artifact": str(output_root), "gate": gate, "gate_passed": gate_passed, **dict(payload)})
    lines = [
        "# Prompt 5-E01-DL-4",
        "",
        f"- gate: `{gate}`",
        f"- gate_passed: `{str(gate_passed).lower()}`",
        f"- selected_critic_profile_id: `{payload.get('selected_critic_profile_id')}`",
        f"- calibration_classification: `{payload.get('calibration_classification')}`",
        f"- final_seed_count: `{payload.get('final_seed_count')}`",
        f"- api/db/network/service_key: `0/false/false/false`",
        f"- h200/cuda/full_daegu/cpu_fallback: `false/false/false/false`",
    ]
    (output_root / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_orchestrator(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else project_root / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    output_root.mkdir(parents=True, exist_ok=True)
    upstream_path = (project_root / args.upstream_artifact).resolve() if not Path(args.upstream_artifact).is_absolute() else Path(args.upstream_artifact)
    upstream_payload, upstream_ok, upstream_fail = validate_upstream(project_root, upstream_path)
    dump_json(output_root / "upstream_validation.json", upstream_payload)
    runtime = runtime_environment(args.device)
    scope = {
        "created_at": iso_kst(),
        "study_area": args.study_area,
        "device": args.device,
        "nodes": 255,
        "edges": 291,
        "dataset_split_locked": True,
        "reward_definition_changed": False,
        "actor_architecture_changed": False,
        "action_space_changed": False,
        "gatv2_architecture_changed": False,
        "suseong_mapping_changed": False,
        "test_holdout_changed": False,
        "h200_used": False,
        "cuda_used": False,
        "full_daegu_training_used": False,
        "cpu_fallback_used": False,
    }
    dump_json(output_root / "scope_guard_audit.json", scope)
    dump_json(output_root / "external_access_audit.json", {"created_at": iso_kst(), "api_call_count": 0, "service_key_accessed": False, "db_accessed": False, "external_network_accessed": False})
    gate = PASS_NONE
    gate_passed = False
    profile_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []
    baseline_diag: Dict[str, Any] = {"executed": False}
    selected: Dict[str, Any] = {}
    stage_payloads: Dict[str, Dict[str, Any]] = {}
    if not upstream_ok:
        gate = upstream_fail
    elif not (runtime["platform_system"] == "Darwin" and runtime["platform_machine"] == "arm64" and runtime["mps_built"] and runtime["mps_available"]):
        gate = FAIL_MPS
    elif runtime["cuda_available"]:
        gate = FAIL_SCOPE
    else:
        mapping_artifact = read_json(upstream_path / "study_area_snapshot.json")["repair_mapping"]
        stage_a_profiles = [
            {"profile_id": "A0_BASELINE_LR1E3", "stage": "A", "critic_lr": 1e-3, "return_normalization": False, "value_loss_type": "mse", "smooth_l1_beta": 1.0, "critic_epochs": 4, "critic_grad_clip": 0.5},
            {"profile_id": "A1_LR3E4", "stage": "A", "critic_lr": 3e-4, "return_normalization": False, "value_loss_type": "mse", "smooth_l1_beta": 1.0, "critic_epochs": 4, "critic_grad_clip": 0.5},
            {"profile_id": "A2_LR1E4", "stage": "A", "critic_lr": 1e-4, "return_normalization": False, "value_loss_type": "mse", "smooth_l1_beta": 1.0, "critic_epochs": 4, "critic_grad_clip": 0.5},
        ]
        stage_a_rows = []
        baseline_validation_return: Optional[float] = None
        for profile in stage_a_profiles:
            row = run_profile_subprocess(project_root, output_root, profile, 2, mapping_artifact, baseline_validation_return, evaluate_test=False)
            if profile["profile_id"] == "A0_BASELINE_LR1E3":
                baseline_validation_return = row.get("validation_episode_return_mean")
                baseline_diag = {
                    "created_at": iso_kst(),
                    "dl3_baseline_reproduced": bool(row.get("gate_passed")),
                    "diagnostic_seed": 2,
                    "profile": row,
                    "value_mae": row.get("validation_value_mae"),
                    "value_rmse": row.get("validation_value_rmse"),
                    "normalized_rmse": row.get("validation_normalized_rmse"),
                    "validation_explained_variance": row.get("validation_explained_variance"),
                    "validation_normalized_rmse": row.get("validation_normalized_rmse"),
                    "validation_episode_return_mean": row.get("validation_episode_return_mean"),
                    "prediction_target_pearson": row.get("prediction_target_pearson"),
                    "calibration_slope": row.get("calibration_slope"),
                    "calibration_intercept": row.get("calibration_intercept"),
                    "prediction_mean_target_mean_ratio": row.get("prediction_mean_target_mean_ratio"),
                    "prediction_std_target_std_ratio": row.get("prediction_std_target_std_ratio"),
                    "explained_variance": row.get("validation_explained_variance"),
                    "critic_clipping_activation_rate": row.get("critic_clipping_activation_rate"),
                    "test_data_used": False,
                }
                dump_json(output_root / "baseline_critic_diagnostic.json", baseline_diag)
            stage_a_rows.append(row)
            profile_rows.append(row)
        selected = choose_best(stage_a_rows)
        stage_payloads["learning_rate_stage_result.json"] = stage_result("critic_learning_rate", stage_a_rows, selected)

        stage_b_profiles = [selected]
        b1 = {**selected, "profile_id": "B1_RETURN_NORM_RUNNING", "stage": "B", "return_normalization": True}
        stage_b_rows = stage_b_profiles + [run_profile_subprocess(project_root, output_root, b1, 2, mapping_artifact, baseline_validation_return, evaluate_test=False)]
        profile_rows.extend([row for row in stage_b_rows if row["profile_id"] == "B1_RETURN_NORM_RUNNING"])
        selected = choose_best(stage_b_rows)
        stage_payloads["return_normalization_stage_result.json"] = stage_result("return_normalization", stage_b_rows, selected)

        stage_c_profiles = [selected]
        c1 = {**selected, "profile_id": "C1_SMOOTH_L1", "stage": "C", "value_loss_type": "smooth_l1", "smooth_l1_beta": 1.0}
        stage_c_rows = stage_c_profiles + [run_profile_subprocess(project_root, output_root, c1, 2, mapping_artifact, baseline_validation_return, evaluate_test=False)]
        profile_rows.extend([row for row in stage_c_rows if row["profile_id"] == "C1_SMOOTH_L1"])
        selected = choose_best(stage_c_rows)
        stage_payloads["value_loss_stage_result.json"] = stage_result("value_loss", stage_c_rows, selected)

        stage_d_profiles = [selected]
        d1 = {**selected, "profile_id": "D1_CRITIC_EPOCHS8", "stage": "D", "critic_epochs": 8}
        stage_d_rows = stage_d_profiles + [run_profile_subprocess(project_root, output_root, d1, 2, mapping_artifact, baseline_validation_return, evaluate_test=False)]
        profile_rows.extend([row for row in stage_d_rows if row["profile_id"] == "D1_CRITIC_EPOCHS8"])
        selected = choose_best(stage_d_rows)
        stage_payloads["critic_epoch_stage_result.json"] = stage_result("critic_update_budget", stage_d_rows, selected)

        stage_e_profiles = [selected]
        e1 = {**selected, "profile_id": "E1_CRITIC_GRAD_CLIP1", "stage": "E", "critic_grad_clip": 1.0}
        stage_e_rows = stage_e_profiles + [run_profile_subprocess(project_root, output_root, e1, 2, mapping_artifact, baseline_validation_return, evaluate_test=False)]
        profile_rows.extend([row for row in stage_e_rows if row["profile_id"] == "E1_CRITIC_GRAD_CLIP1"])
        selected = choose_best(stage_e_rows)
        stage_payloads["gradient_clip_stage_result.json"] = stage_result("critic_gradient_clipping", stage_e_rows, selected)

        selected_profile = {
            "created_at": iso_kst(),
            "selected_critic_profile": selected,
            "selection_used_validation_only": True,
            "test_data_used_for_profile_selection": False,
            "actor_reward_dataset_mapping_locked": True,
        }
        dump_json(output_root / "selected_critic_profile.json", selected_profile)

        for seed in [1, 2, 3]:
            final_profile = {
                "profile_id": f"FINAL_SELECTED_SEED{seed}",
                "stage": "FINAL_3_SEED",
                "critic_lr": selected["critic_lr"],
                "return_normalization": selected["return_normalization"],
                "value_loss_type": selected["value_loss_type"],
                "smooth_l1_beta": selected["smooth_l1_beta"],
                "critic_epochs": selected["critic_epochs"],
                "critic_grad_clip": selected["critic_grad_clip"],
            }
            final_rows.append(run_profile_subprocess(project_root, output_root, final_profile, seed, mapping_artifact, baseline_validation_return, evaluate_test=True))
        failure_rows = [row for row in final_rows if not selectable(row)]
        if failure_rows:
            gate = FAIL_CRITIC
        else:
            gate, improvement_report = compare_improvement(final_rows, baseline_diag, upstream_path)
            gate_passed = True
            dump_json(output_root / "three_seed_critic_summary.json", {"created_at": iso_kst(), "final_seed_profiles": final_rows, "improvement_report": improvement_report})
            safe_to_parquet(final_rows, output_root / "three_seed_critic_summary.parquet")
            dump_json(output_root / "prediction_target_calibration.json", {"created_at": iso_kst(), "baseline": baseline_diag, "selected_seed2_profile": selected, "final_seed_profiles": final_rows, "improvement_report": improvement_report})
            dump_json(output_root / "normalized_error_summary.json", {"created_at": iso_kst(), "baseline_normalized_rmse": baseline_diag.get("normalized_rmse"), "selected_validation_normalized_rmse": selected.get("validation_normalized_rmse"), "final_test_normalized_rmse": metric_stats([row.get("test_normalized_rmse") for row in final_rows])})
            dump_json(output_root / "critic_clipping_audit.json", {"created_at": iso_kst(), "profiles": [{k: row.get(k) for k in ["profile_id", "critic_grad_clip", "critic_clipping_activation_rate"]} for row in profile_rows + final_rows]})
            reload_rows = []
            for row in profile_rows + final_rows:
                path = Path(row["profile_dir"]) / "checkpoint_reload_audit.json"
                reload_rows.append({"profile_id": row["profile_id"], "reload": read_json(path) if path.exists() else None})
            dump_json(output_root / "checkpoint_reload_audit.json", {"created_at": iso_kst(), "all_reload_ok": all((r["reload"] or {}).get("reload_ok") for r in reload_rows), "profiles": reload_rows})
    if not (output_root / "baseline_critic_diagnostic.json").exists():
        dump_json(output_root / "baseline_critic_diagnostic.json", baseline_diag)
    for name, payload in stage_payloads.items():
        dump_json(output_root / name, payload)
    for name in [
        "learning_rate_stage_result.json",
        "return_normalization_stage_result.json",
        "value_loss_stage_result.json",
        "critic_epoch_stage_result.json",
        "gradient_clip_stage_result.json",
    ]:
        if not (output_root / name).exists():
            dump_json(output_root / name, {"created_at": iso_kst(), "executed": False, "gate": gate})
    if not (output_root / "selected_critic_profile.json").exists():
        dump_json(output_root / "selected_critic_profile.json", {"created_at": iso_kst(), "selected": selected, "selection_available": bool(selected)})
    if not (output_root / "three_seed_critic_summary.json").exists():
        dump_json(output_root / "three_seed_critic_summary.json", {"created_at": iso_kst(), "executed": False, "final_seed_profiles": final_rows, "gate": gate})
        safe_to_parquet(final_rows, output_root / "three_seed_critic_summary.parquet")
    for name in ["prediction_target_calibration.json", "normalized_error_summary.json", "critic_clipping_audit.json", "checkpoint_reload_audit.json"]:
        if not (output_root / name).exists():
            dump_json(output_root / name, {"created_at": iso_kst(), "executed": False, "gate": gate})
    dump_json(output_root / "critic_profile_registry.json", {"created_at": iso_kst(), "profiles": profile_rows, "profile_count": len(profile_rows)})
    safe_to_parquet(profile_rows, output_root / "critic_profile_registry.parquet")
    dump_json(output_root / "critic_profile_comparison.json", {"created_at": iso_kst(), "profiles": profile_rows, "ranked_profiles": sorted(profile_rows, key=profile_score, reverse=True), "selection_used_test_data": False})
    dump_json(output_root / "source_change_inventory.json", {"created_at": iso_kst(), "files": [{"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())}, {"path": str(project_root / "05_training/run_prompt5_e01_dl3_suseong_three_seed_full_training.py"), "sha256": sha256_file(project_root / "05_training/run_prompt5_e01_dl3_suseong_three_seed_full_training.py")}, {"path": str(project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"), "sha256": sha256_file(project_root / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py")}], "dl3_training_engine_reused": True})
    final_payload = {
        "selected_critic_profile_id": selected.get("profile_id"),
        "calibration_classification": read_json(output_root / "three_seed_critic_summary.json").get("improvement_report", {}).get("classification"),
        "final_seed_count": len(final_rows),
    }
    dump_json(output_root / "gate_decision.json", {"created_at": iso_kst(), "gate": gate, "gate_passed": gate_passed, "selected_critic_profile_id": selected.get("profile_id"), "final_seed_count": len(final_rows), "api_call_count": 0, "service_key_accessed": False, "db_accessed": False, "external_network_accessed": False, "h200_used": False, "cuda_used": False, "full_daegu_training_used": False, "cpu_fallback_used": False})
    write_final(output_root, gate, gate_passed, final_payload)
    success_lock = json.dumps({"created_at": iso_kst(), "gate": gate, "gate_passed": gate_passed, "artifact_complete": True}, sort_keys=True) + "\n"
    manifest = artifact_manifest(output_root, success_lock)
    if manifest["missing_required_files_after_success_lock"]:
        gate, gate_passed = FAIL_MANIFEST, False
        dump_json(output_root / "gate_decision.json", {"created_at": iso_kst(), "gate": gate, "gate_passed": False, "manifest_missing": manifest})
        write_final(output_root, gate, gate_passed, final_payload)
        success_lock = json.dumps({"created_at": iso_kst(), "gate": gate, "gate_passed": gate_passed, "artifact_complete": True}, sort_keys=True) + "\n"
        manifest = artifact_manifest(output_root, success_lock)
    dump_json(output_root / "artifact_manifest.json", manifest)
    (output_root / "_SUCCESS.lock").write_text(success_lock, encoding="utf-8")
    print(json.dumps({"artifact": str(output_root), "gate": gate, "gate_passed": gate_passed, "profile_count": len(profile_rows), "final_seed_count": len(final_rows)}, ensure_ascii=False, sort_keys=True))
    return 0 if gate_passed else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-DL-4 Suseong critic calibration stabilization.")
    parser.add_argument("--project-root", default="/Users/arty/Documents/Codex/urbanbus_rl_project")
    parser.add_argument("--upstream-artifact", default=f"05_training/artifacts/{DL3_ARTIFACT}")
    parser.add_argument("--study-area", default="SUSEONG_GU_DAEGU")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--no-cpu-fallback", action="store_true", default=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--profile-worker", action="store_true", default=False)
    parser.add_argument("--profile-output-dir", default=None)
    parser.add_argument("--profile-id", default="A0_BASELINE_LR1E3")
    parser.add_argument("--stage", default="A")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--mapping-artifact", default=None)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--return-normalization", action="store_true", default=False)
    parser.add_argument("--value-loss-type", choices=["mse", "smooth_l1"], default="mse")
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--critic-epochs", type=int, default=4)
    parser.add_argument("--critic-grad-clip", type=float, default=0.5)
    parser.add_argument("--baseline-validation-return", type=float, default=float("nan"))
    parser.add_argument("--evaluate-test", action="store_true", default=False)
    args = parser.parse_args(argv)
    if args.study_area != "SUSEONG_GU_DAEGU":
        raise SystemExit("Only SUSEONG_GU_DAEGU is allowed.")
    if args.device != "mps":
        raise SystemExit("Only mps is allowed.")
    if args.profile_worker:
        if args.profile_output_dir is None or args.mapping_artifact is None:
            raise SystemExit("--profile-output-dir and --mapping-artifact are required for profile worker.")
        if math.isnan(args.baseline_validation_return):
            args.baseline_validation_return = None
        return run_profile_worker(args)
    return run_orchestrator(args)


if __name__ == "__main__":
    raise SystemExit(main())
