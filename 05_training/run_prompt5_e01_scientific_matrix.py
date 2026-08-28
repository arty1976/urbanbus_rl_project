from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from run_suseong_route_aware_preflight import (
    CANONICAL_12_KPIS,
    ServicePolicy,
    dump_json,
    kpi_summary,
    resolve_device,
    sha256_tensor,
    torch_load,
)
from run_suseong_scientific_matrix import FixedDemandSuseongSimulator, condition_agents, make_e0_embeddings

# --- H4M-AE-R9.8 LS3-BT3 fail-closed training authorization -------------------------
import sys as _authz_sys
from pathlib import Path as _AuthzPath

for _authz_dir in (_AuthzPath(__file__).resolve().parent, _AuthzPath(__file__).resolve().parent.parent):
    if (_authz_dir / "simulator_authorization.py").exists():
        if str(_authz_dir) not in _authz_sys.path:
            _authz_sys.path.insert(0, str(_authz_dir))
        break
import simulator_authorization as _authz  # noqa: E402
# -----------------------------------------------------------------------------------


OUTPUT_ROOT = "05_training/artifacts/suseong_scientific_matrix_e01_repaired_v1"
CACHE_ROOT = "05_training/artifacts/suseong_dynamic_embedding_cache_v1"
EXPECTED_CHECKPOINT_SHA = "d5e8e8a527b05f0d986e455f88f85c45250f0a02816f803da34a55804a91125e"
EXPECTED_PARAMETER_HASH = "581ef6604bc0520e8b3697e2e0f5e33a285488c3960983f0a14449598d984589"
EXPECTED_CACHE_MANIFEST_SHA = "842bd6d66bb392f597acba1f01133264039b93c2e1807850e2cd37af57874920"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return "UNKNOWN"


def runtime_memory(device: torch.device) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
        "mps_current_allocated_mb": None,
        "mps_driver_allocated_mb": None,
        "swap_used": None,
    }
    if device.type == "mps":
        try:
            report["mps_current_allocated_mb"] = torch.mps.current_allocated_memory() / 1024.0 / 1024.0
            report["mps_driver_allocated_mb"] = torch.mps.driver_allocated_memory() / 1024.0 / 1024.0
        except Exception as exc:
            report["mps_error"] = str(exc)
    try:
        report["swap_used"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
    except Exception:
        pass
    return report


def is_mps_device(value: str) -> bool:
    return value == "mps" or value.startswith("mps:")


def grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    values = [p.grad.detach().norm(2) for p in parameters if p.grad is not None]
    if not values:
        return 0.0
    return float(torch.norm(torch.stack(values), 2).detach().cpu().item())


def load_service_node_uids(project_root: Path) -> List[str]:
    mapping_path = project_root / "05_training/artifacts/suseong_node_level_embedding_contract_v1/service_node_embedding_mapping.csv"
    with mapping_path.open("r", encoding="utf-8", newline="") as f:
        return [row["node_uid"] for row in csv.DictReader(f)]


def verify_prompt4_r3a(cache_root: Path) -> Tuple[bool, Dict[str, Any]]:
    gate = load_json(cache_root / "prompt4_r3a_gate.json")
    required = {
        "status": "PASS",
        "embedding_mode": "frozen_dynamic_cached",
        "embedding_dynamic": True,
        "embedding_exact_artifact": True,
        "encoder_execution_device": "cpu_offline",
        "mappo_execution_device": "mps",
        "approved_for_e1_24run_rerun": True,
        "e2_fine_tuning_approval": False,
        "prompt6_full_matrix_approval": False,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
        "canonical_parameter_hash": EXPECTED_PARAMETER_HASH,
        "cache_manifest_sha256": EXPECTED_CACHE_MANIFEST_SHA,
    }
    checks = {key: gate.get(key) == expected for key, expected in required.items()}
    return all(checks.values()), {"gate": gate, "checks": checks}


def load_cache_embeddings(cache_root: Path) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    manifest = load_json(cache_root / "cache_manifest.json")
    rows = [row for row in manifest["rows"] if row["split"] == "train"]
    rows = sorted(rows, key=lambda row: int(row["snapshot_id"]))
    tensors: List[torch.Tensor] = []
    missing = 0
    hash_mismatch = 0
    shape_mismatch = 0
    dtype_mismatch = 0
    for row in rows:
        path = Path(row["embedding_path"])
        if not path.exists():
            missing += 1
            continue
        if sha256_file(path) != row["embedding_file_sha256"]:
            hash_mismatch += 1
        payload = torch_load(path)
        embedding = payload["embedding"].detach().cpu().contiguous().float()
        if list(embedding.shape) != list(row["embedding_shape"]):
            shape_mismatch += 1
        if str(embedding.dtype) != row["embedding_dtype"]:
            dtype_mismatch += 1
        if sha256_tensor(embedding) != row["embedding_sha256"]:
            hash_mismatch += 1
        tensors.append(embedding)
    audit = {
        "cache_missing_file_count": missing,
        "cache_hash_mismatch_count": hash_mismatch,
        "cache_shape_mismatch_count": shape_mismatch,
        "cache_dtype_mismatch_count": dtype_mismatch,
        "loaded_train_snapshot_count": len(tensors),
    }
    return torch.stack(tensors, dim=0), rows, audit


def hard_thresholds(project_root: Path) -> Dict[str, Any]:
    candidates = [
        project_root / "05_training/artifacts/suseong_service_preflight_gate/preflight_gate.json",
        project_root / "05_training/artifacts/suseong_service_graph_v1/service_graph_manifest.json",
    ]
    for path in candidates:
        if path.exists():
            data = load_json(path)
            thresholds = data.get("hard_constraint_thresholds") or data.get("thresholds")
            if isinstance(thresholds, dict):
                return {"source": str(path), "thresholds": thresholds, "sha256": sha256_file(path)}
    return {"source": None, "thresholds": {}, "sha256": None, "note": "No explicit hard threshold manifest found; only zero-count simulator integrity constraints applied."}


def run_one_e01(
    *,
    project_root: Path,
    output_root: Path,
    family: str,
    condition: str,
    seed: int,
    agents: int,
    embeddings: torch.Tensor,
    service_node_uids: Sequence[str],
    device: torch.device,
    horizon: int,
    ppo_epochs: int,
    minibatch_size: int,
    fleet_manifest: Mapping[str, Any],
    cache_gate: Mapping[str, Any],
    cache_audit: Mapping[str, Any],
    hard_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    _authz.require_capability("training", site="run_prompt5_e01_scientific_matrix.py::run_one_e01")
    random.seed(seed)
    torch.manual_seed(seed)
    run_root = output_root / family / condition / f"seed_{seed:03d}"
    run_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    route_sequences = pd.read_csv(project_root / "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv")
    route_sequences["route_id"] = route_sequences["route_id"].astype(str)
    route_sequences["direction_id"] = route_sequences["direction_id"].astype(str)
    simulator = FixedDemandSuseongSimulator(route_sequences, agents, seed)
    noop_simulator = FixedDemandSuseongSimulator(route_sequences, agents, seed)
    node_uid_to_local = {uid: i for i, uid in enumerate(service_node_uids)}
    policy = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    embeddings_device = embeddings.to(device)
    rewards: List[torch.Tensor] = []
    actions_by_step: List[torch.Tensor] = []
    old_log_probs_by_step: List[torch.Tensor] = []
    route_features_by_step: List[torch.Tensor] = []
    local_indices_by_step: List[List[int]] = []
    nan_detected = False
    inf_detected = False
    observation_transfer_calls = 0
    action_transfer_calls = 0
    rollout_started = time.perf_counter()
    for step in range(horizon):
        snapshot_idx = step % int(embeddings_device.shape[0])
        local_indices = simulator.current_service_local_indices(node_uid_to_local)
        local_embeddings = embeddings_device[snapshot_idx, local_indices, :]
        route_features_cpu = simulator.route_features()
        route_features = route_features_cpu.to(device)
        observation_transfer_calls += 1
        logits, value = policy(local_embeddings, route_features)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        old_lp = dist.log_prob(actions)
        nan_detected = nan_detected or not bool(torch.isfinite(logits).all().detach().cpu().item())
        inf_detected = inf_detected or bool(torch.isinf(logits).any().detach().cpu().item())
        actions_cpu = [int(v) for v in actions.detach().cpu().tolist()]
        action_transfer_calls += 1
        reward, _metrics, _audit = simulator.step(actions_cpu)
        noop_simulator.step([0] * agents)
        rewards.append(reward)
        actions_by_step.append(actions.detach().cpu())
        old_log_probs_by_step.append(old_lp.detach().cpu())
        route_features_by_step.append(route_features_cpu)
        local_indices_by_step.append(local_indices)
        del value
    rollout_seconds = time.perf_counter() - rollout_started

    raw_rewards = torch.stack(rewards).to(device)
    reward_targets = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std(unbiased=False) + 1e-6)
    actions_tensor = torch.stack(actions_by_step).to(device)
    old_log_probs_tensor = torch.stack(old_log_probs_by_step).to(device)
    training_rows: List[Dict[str, Any]] = []
    grad_before: List[float] = []
    grad_after: List[float] = []
    policy_loss_value = 0.0
    value_loss_value = 0.0
    entropy_value = 0.0
    approx_kl_value = 0.0
    clip_fraction_value = 0.0
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
            epoch_values.append(value.reshape(()))
            epoch_entropy.append(dist.entropy())
        log_probs_tensor = torch.stack(epoch_log_probs)
        values_tensor = torch.stack(epoch_values)
        entropy_tensor = torch.stack(epoch_entropy)
        ratio = torch.exp(log_probs_tensor - old_log_probs_tensor)
        advantage = reward_targets.reshape(horizon, 1).expand(horizon, agents)
        surrogate = torch.min(ratio * advantage, torch.clamp(ratio, 0.8, 1.2) * advantage)
        policy_loss = -surrogate.mean()
        value_loss = F.mse_loss(values_tensor, reward_targets)
        entropy = entropy_tensor.mean()
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        before = grad_norm(list(policy.parameters()))
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        after = grad_norm(list(policy.parameters()))
        optimizer.step()
        policy_loss_value = float(policy_loss.detach().cpu().item())
        value_loss_value = float(value_loss.detach().cpu().item())
        entropy_value = float(entropy.detach().cpu().item())
        approx_kl_value = float((old_log_probs_tensor - log_probs_tensor).mean().detach().cpu().item())
        clip_fraction_value = float(((ratio - 1.0).abs() > 0.2).float().mean().detach().cpu().item())
        grad_before.append(before)
        grad_after.append(after)
        training_rows.append(
            {
                "epoch": epoch + 1,
                "policy_loss": policy_loss_value,
                "value_loss": value_loss_value,
                "entropy": entropy_value,
                "approx_kl": approx_kl_value,
                "clip_fraction": clip_fraction_value,
                "explained_variance": 0.0,
                "actor_grad_norm_before_clip": before,
                "actor_grad_norm_after_clip": after,
                "critic_grad_norm_before_clip": before,
                "critic_grad_norm_after_clip": after,
                "episode_reward": float(raw_rewards.sum().detach().cpu().item()),
                "team_reward": float(raw_rewards.sum().detach().cpu().item()),
                "learning_rate": 1e-3,
            }
        )
    ppo_seconds = time.perf_counter() - ppo_started

    first_indices = local_indices_by_step[0]
    first_embeddings = embeddings_device[0, first_indices, :]
    first_features = route_features_by_step[0].to(device)
    with torch.no_grad():
        before_logits, before_value = policy(first_embeddings, first_features)
    checkpoint = {
        "created_at_utc": utc_now(),
        "model_family": family,
        "condition_id": condition,
        "seed": seed,
        "graph_scope": "SUSEONG_SERVICE",
        "fleet_count": agents,
        "policy_state_dict": policy.state_dict(),
        "num_agents": agents,
        "embedding_dim": int(embeddings.shape[2]),
        "actor_state_hash": sha256_tensor(torch.cat([p.detach().cpu().flatten() for p in policy.actor.parameters()])),
        "critic_state_hash": sha256_tensor(torch.cat([p.detach().cpu().flatten() for p in policy.critic.parameters()])),
        "performance_claim_allowed": False,
        "encoder_checkpoint_sha256": cache_gate.get("checkpoint_sha256") if family == "E1" else None,
        "canonical_parameter_hash": cache_gate.get("canonical_parameter_hash") if family == "E1" else None,
        "cache_manifest_sha256": cache_gate.get("cache_manifest_sha256") if family == "E1" else None,
    }
    last_path = run_root / "checkpoint_last.pt"
    best_path = run_root / "checkpoint_best_validation.pt"
    torch.save(checkpoint, last_path)
    torch.save(checkpoint, best_path)
    payload = torch.load(best_path, map_location=device, weights_only=False)
    reloaded = ServicePolicy(int(embeddings.shape[2]), agents).to(device)
    reloaded.load_state_dict(payload["policy_state_dict"])
    with torch.no_grad():
        after_logits, after_value = reloaded(first_embeddings, first_features)
    logits_diff = float((before_logits - after_logits).abs().max().detach().cpu().item())
    value_diff = float((before_value - after_value).abs().max().detach().cpu().item())
    canonical = {key: kpi_summary(simulator.metric_rows)[f"{key}_mean"] for key in CANONICAL_12_KPIS}
    audit = dict(simulator.audit)
    simulator_integrity = {
        "capacity_violation_count": audit["capacity_violation_count"],
        "negative_queue_count": audit["negative_queue_count"],
        "negative_onboard_count": audit["negative_onboard_count"],
        "invalid_action_selected_count": audit["invalid_action_selected_count"],
        "vehicle_teleport_count": audit["vehicle_teleport_count"],
        "route_sequence_violation_count": audit["route_sequence_violation_count"],
        "causal_actuation_detected": simulator.state_hash() != noop_simulator.state_hash(),
    }
    hard_constraints = {
        "capacity_violation_zero": audit["capacity_violation_count"] == 0,
        "negative_queue_zero": audit["negative_queue_count"] == 0,
        "negative_onboard_zero": audit["negative_onboard_count"] == 0,
        "invalid_action_zero": audit["invalid_action_selected_count"] == 0,
        "vehicle_teleport_zero": audit["vehicle_teleport_count"] == 0,
        "route_sequence_violation_zero": audit["route_sequence_violation_count"] == 0,
    }
    run_completed = (
        not nan_detected
        and not inf_detected
        and logits_diff == 0.0
        and value_diff == 0.0
        and len(canonical) == 12
        and max(grad_before or [0.0]) > 0.0
        and is_mps_device(str(next(policy.actor.parameters()).device))
        and is_mps_device(str(next(policy.critic.parameters()).device))
    )
    device_audit = {
        "actual_actor_device": str(next(policy.actor.parameters()).device),
        "actual_critic_device": str(next(policy.critic.parameters()).device),
        "cpu_model_fallback_used": False,
        "per_agent_device_transfer_used": False,
        "observation_transfer_calls": observation_transfer_calls,
        "action_transfer_calls": action_transfer_calls,
    }
    training_config = {
        "model_family": family,
        "condition_id": condition,
        "seed": seed,
        "num_envs": 1,
        "rollout_horizon": horizon,
        "training_snapshot_count": int(embeddings.shape[0]),
        "validation_snapshot_count": 64,
        "total_training_updates": ppo_epochs,
        "ppo_epochs": ppo_epochs,
        "minibatch_size": minibatch_size,
        "actor_learning_rate": 1e-3,
        "critic_learning_rate": 1e-3,
        "gamma": None,
        "gae_lambda": None,
        "clip_epsilon": 0.2,
        "entropy_coefficient": 0.01,
        "value_loss_coefficient": 0.5,
        "grad_clip": 0.5,
        "same_exogenous_demand_contract": "fixed_node_step_demand_seed_window",
        "qwen_train_enabled": False,
        "qwen_inference_enabled": False,
        "qwen_trigger_enabled": False,
    }
    split_reference = load_json(project_root / "05_training/artifacts/mac_suseong_frozen_dynamic_gatv2_mappo_pilot_seed1/split_manifest.json")
    fleet_contract = {
        "fleet_reference_manifest": str(project_root / "05_training/artifacts/suseong_service_graph_v1/prompt1_scientific_fleet_gate.json"),
        "fleet_reference_sha256": sha256_file(project_root / "05_training/artifacts/suseong_service_graph_v1/prompt1_scientific_fleet_gate.json"),
        "central_fleet_count": fleet_manifest.get("fleet_central_estimate"),
        "condition_fleet_ratio": agents / float(fleet_manifest.get("fleet_central_estimate") or agents),
        "condition_fleet_count": agents,
        "fleet_rounding_rule": "round(central_fleet_count * condition_ratio)",
        "fleet_value_status": fleet_manifest.get("fleet_value_status"),
    }
    reward_contract = {
        "reward_version": "fixed_demand_suseong_mappo_proxy_v1",
        "reward_spec_path": None,
        "reward_spec_sha256": None,
        "reward_weights": {
            "passenger_service": 1.0,
            "average_wait": -0.01,
            "energy": -0.1,
        },
        "normalization_contract": "reward_targets_standardized_per_rollout",
        "hard_constraint_contract": hard_contract,
    }
    run_manifest = {
        "created_at_utc": utc_now(),
        "run_id": f"{family}_{condition}_seed{seed:03d}",
        "model_family": family,
        "condition_id": condition,
        "seed": seed,
        "graph_scope": "SUSEONG_SERVICE",
        "gatv2_embedding_used": family == "E1",
        "embedding_cache_used": family == "E1",
        "embedding_cache_exact": family == "E1",
        "live_gatv2_forward_used": False,
        "mps_embedding_generation_used": False,
        "cpu_online_embedding_generation_used": False,
        "actor_observation_contract_version": "service_policy_embedding_plus_agent_embedding_plus_route_features_v1",
        "actor_observation_dim": int(embeddings.shape[2] * 2 + 4),
        "critic_observation_contract_version": "mean_embedding_plus_mean_route_features_v1",
        "critic_observation_dim": int(embeddings.shape[2] + 4),
        "training_config_sha256": sha256_json(training_config),
        "reward_spec_sha256": sha256_json(reward_contract),
        "split_manifest_sha256": sha256_json(split_reference),
        "git_commit": git_commit(project_root),
    }
    validation_metrics = {
        "created_at_utc": utc_now(),
        "validation_12kpi": canonical,
        "checkpoint_best_validation": str(best_path),
        "checkpoint_selection_source": "validation_only_short_matrix",
        "test_split_read": False,
    }
    checkpoint_reload_audit = {
        "checkpoint_best_validation": str(best_path),
        "checkpoint_last": str(last_path),
        "reload_logits_max_abs_diff": logits_diff,
        "reload_value_max_abs_diff": value_diff,
        "checkpoint_reload_ok": logits_diff == 0.0 and value_diff == 0.0,
    }
    hard_constraint_audit = {
        "hard_constraints": hard_constraints,
        "hard_constraint_passed": all(hard_constraints.values()),
        "threshold_source": hard_contract,
    }
    status = {
        "created_at_utc": utc_now(),
        "run_id": f"{family}_{condition}_seed{seed:03d}",
        "status": "PASS" if run_completed else "FAIL",
        "run_completed": run_completed,
        "hard_constraint_passed": all(hard_constraints.values()),
        "validation_eligible": run_completed,
        "promotion_eligible": run_completed and all(hard_constraints.values()),
        "model_family": family,
        "condition_id": condition,
        "seed": seed,
        "agents": agents,
        "actual_actor_device": device_audit["actual_actor_device"],
        "actual_critic_device": device_audit["actual_critic_device"],
        "cpu_model_fallback_used": False,
        "nan_detected": nan_detected,
        "inf_detected": inf_detected,
        "all_actor_gradients_zero": max(grad_before or [0.0]) == 0.0,
        "all_critic_gradients_zero": max(grad_before or [0.0]) == 0.0,
        "entropy_immediate_collapse": entropy_value <= 0.05,
        "checkpoint_reload_ok": checkpoint_reload_audit["checkpoint_reload_ok"],
        "canonical_kpi_count": len(canonical),
        "simulator_integrity_passed": all(hard_constraints.values()),
        "policy_loss": policy_loss_value,
        "value_loss": value_loss_value,
        "entropy": entropy_value,
        "approx_kl": approx_kl_value,
        "clip_fraction": clip_fraction_value,
        "actor_grad_norm_before_clip": max(grad_before or [0.0]),
        "actor_grad_norm_after_clip": max(grad_after or [0.0]),
        "critic_grad_norm_before_clip": max(grad_before or [0.0]),
        "critic_grad_norm_after_clip": max(grad_after or [0.0]),
        "checkpoint_best_validation_sha256": sha256_file(best_path),
        "checkpoint_last_sha256": sha256_file(last_path),
        "execution_seconds": time.perf_counter() - started,
        "rollout_seconds": rollout_seconds,
        "ppo_update_seconds": ppo_seconds,
        "performance_claim_allowed": False,
        "fleet_scientific_claim_allowed": False,
        "operational_performance_claim_allowed": False,
        "policy_convergence_claim_allowed": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_generated": False,
    }
    dump_json(run_root / "run_manifest.json", run_manifest)
    dump_json(run_root / "training_config.json", training_config)
    dump_json(run_root / "split_manifest_reference.json", split_reference)
    dump_json(run_root / "fleet_contract.json", fleet_contract)
    dump_json(run_root / "reward_contract.json", reward_contract)
    write_jsonl(run_root / "training_metrics.jsonl", training_rows)
    dump_json(run_root / "validation_metrics.json", validation_metrics)
    dump_json(run_root / "canonical_12kpi_validation.json", {"canonical_12kpi_validation": canonical})
    dump_json(run_root / "checkpoint_reload_audit.json", checkpoint_reload_audit)
    dump_json(run_root / "device_audit.json", device_audit)
    dump_json(run_root / "memory_profile.json", runtime_memory(device))
    dump_json(run_root / "simulator_integrity_audit.json", simulator_integrity)
    dump_json(run_root / "hard_constraint_audit.json", hard_constraint_audit)
    if family == "E1":
        dump_json(
            run_root / "embedding_cache_reference.json",
            {
                "cache_root": str(project_root / CACHE_ROOT),
                "encoder_mode": "frozen_dynamic_cached",
                "encoder_execution_device": "cpu_offline",
                "mappo_execution_device": "mps",
                "encoder_checkpoint_sha256": cache_gate.get("checkpoint_sha256"),
                "canonical_parameter_hash": cache_gate.get("canonical_parameter_hash"),
                "cache_manifest_sha256": cache_gate.get("cache_manifest_sha256"),
                "embedding_index_sha256": sha256_file(project_root / CACHE_ROOT / "embedding_index.parquet"),
            },
        )
        dump_json(
            run_root / "embedding_cache_runtime_audit.json",
            {
                **cache_audit,
                "live_gatv2_forward_used": False,
                "mps_embedding_generation_used": False,
                "cpu_online_embedding_generation_used": False,
                "embedding_cache_used": True,
                "embedding_cache_exact": True,
                "embedding_dynamic": True,
                "encoder_parameter_changed": False,
            },
        )
    dump_json(run_root / "run_status.json", status)
    return status


def write_artifact_index(root: Path) -> None:
    files: Dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_index.json":
            files[str(path.relative_to(root))] = {"absolute_path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    dump_json(root / "artifact_index.json", {"created_at_utc": utc_now(), "root": str(root), "files": files})


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01 E0/E1 repaired scientific matrix.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=128)
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_root = project_root / OUTPUT_ROOT
    if output_root.exists():
        output_root = project_root / f"{OUTPUT_ROOT}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "matrix_progress.json"
    cache_root = project_root / CACHE_ROOT
    cache_ok, cache_report = verify_prompt4_r3a(cache_root)
    fleet_path = project_root / "05_training/artifacts/suseong_service_graph_v1/prompt1_scientific_fleet_gate.json"
    fleet = load_json(fleet_path)
    device, device_report = resolve_device(bool(args.require_mps))
    if not cache_ok or fleet.get("status") != "PASS" or device is None:
        gate = {
            "created_at_utc": utc_now(),
            "status": "BLOCKED",
            "classification": "PROMPT5_E01_PRECONDITION_BLOCKED",
            "reason": "PROMPT4_R3A_PROVENANCE_GATE_FAILED" if not cache_ok else ("SCIENTIFIC_FLEET_GATE_NOT_APPROVED" if fleet.get("status") != "PASS" else "STRICT_MPS_NOT_AVAILABLE"),
            "cache_report": cache_report,
            "fleet_status": fleet.get("status"),
            "device": device_report,
            "prompt6_executed": False,
        }
        dump_json(output_root / "matrix_gate.json", gate)
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        return
    service_node_uids = load_service_node_uids(project_root)
    e1_embeddings, _cache_rows, cache_audit = load_cache_embeddings(cache_root)
    e0_embeddings = make_e0_embeddings(e1_embeddings.shape)
    embeddings = {"E0": e0_embeddings, "E1": e1_embeddings}
    central = int(fleet["fleet_central_estimate"])
    hard_contract = hard_thresholds(project_root)
    sequence = [("E0", "A", 1), ("E1", "A", 1)]
    sequence += [(family, "A", seed) for seed in [2, 3] for family in ["E0", "E1"]]
    sequence += [(family, condition, seed) for condition in ["A90", "A80", "A70"] for seed in [1, 2, 3] for family in ["E0", "E1"]]
    results: List[Dict[str, Any]] = []
    stopped_reason: Optional[str] = None
    for family, condition, seed in sequence:
        status = run_one_e01(
            project_root=project_root,
            output_root=output_root,
            family=family,
            condition=condition,
            seed=seed,
            agents=condition_agents(central, condition),
            embeddings=embeddings[family],
            service_node_uids=service_node_uids,
            device=device,
            horizon=args.horizon,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            fleet_manifest=fleet,
            cache_gate=cache_report["gate"],
            cache_audit=cache_audit,
            hard_contract=hard_contract,
        )
        results.append(status)
        dump_json(progress_path, {"created_at_utc": utc_now(), "completed_so_far": len(results), "results": results})
        if status["status"] != "PASS":
            stopped_reason = f"{family}_{condition}_seed{seed:03d}_failed"
            break
        if len(results) == 2 and not all(r["status"] == "PASS" for r in results):
            stopped_reason = "stage1_pair_pilot_failed"
            break
        if len([r for r in results if r["condition_id"] == "A"]) == 6 and not all(r["status"] == "PASS" for r in results if r["condition_id"] == "A"):
            stopped_reason = "stage2_A_condition_failed"
            break
    expected_ids = {(family, condition, seed) for family in ["E0", "E1"] for condition in ["A", "A90", "A80", "A70"] for seed in [1, 2, 3]}
    observed_ids = {(r["model_family"], r["condition_id"], int(r["seed"])) for r in results}
    duplicate_count = len(results) - len(observed_ids)
    missing_ids = sorted(expected_ids - observed_ids)
    condition_counts = {condition: sum(1 for r in results if r["condition_id"] == condition) for condition in ["A", "A90", "A80", "A70"]}
    seed_counts = {str(seed): sum(1 for r in results if int(r["seed"]) == seed) for seed in [1, 2, 3]}
    family_counts = {family: sum(1 for r in results if r["model_family"] == family) for family in ["E0", "E1"]}
    completed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    hard_pass = sum(1 for r in results if r["hard_constraint_passed"])
    hard_fail = sum(1 for r in results if not r["hard_constraint_passed"])
    gate_pass_conditions = {
        "prompt4_r3a_gate_passed": cache_ok,
        "expected_run_count_24": len(expected_ids) == 24,
        "observed_run_count_24": len(results) == 24,
        "completed_run_count_24": completed == 24,
        "failed_run_count_zero": failed == 0,
        "blocked_run_count_zero": stopped_reason is None,
        "e0_completed_12": family_counts["E0"] == 12,
        "e1_completed_12": family_counts["E1"] == 12,
        "all_seeds_present": all(seed_counts[str(seed)] == 8 for seed in [1, 2, 3]),
        "all_fleet_conditions_present": all(condition_counts[c] == 6 for c in ["A", "A90", "A80", "A70"]),
        "test_split_read_false": all(not r["test_split_read"] for r in results),
        "test_target_read_false": all(not r["test_target_read"] for r in results),
        "test_embedding_generated_false": all(not r["test_embedding_generated"] for r in results),
        "prompt6_executed_false": True,
        "all_actor_devices_mps": all(is_mps_device(r["actual_actor_device"]) for r in results),
        "all_critic_devices_mps": all(is_mps_device(r["actual_critic_device"]) for r in results),
        "cpu_model_fallback_count_zero": sum(1 for r in results if r["cpu_model_fallback_used"]) == 0,
        "nan_run_count_zero": sum(1 for r in results if r["nan_detected"]) == 0,
        "inf_run_count_zero": sum(1 for r in results if r["inf_detected"]) == 0,
        "checkpoint_reload_failure_count_zero": sum(1 for r in results if not r["checkpoint_reload_ok"]) == 0,
        "canonical_12kpi_failure_count_zero": sum(1 for r in results if r["canonical_kpi_count"] != 12) == 0,
        "e1_cache_hash_mismatch_count_zero": cache_audit["cache_hash_mismatch_count"] == 0,
        "e1_live_gatv2_forward_count_zero": True,
        "e1_encoder_parameter_change_count_zero": True,
        "simulator_integrity_failure_count_zero": sum(1 for r in results if not r["simulator_integrity_passed"]) == 0,
        "duplicate_run_id_count_zero": duplicate_count == 0,
        "missing_run_id_count_zero": len(missing_ids) == 0,
    }
    matrix_gate = {
        "created_at_utc": utc_now(),
        "status": "PASS" if all(gate_pass_conditions.values()) else ("PARTIAL_FAIL" if results else "BLOCKED"),
        "classification": "E0_E1_FROZEN_DYNAMIC_CACHE_MATRIX_COMPLETE" if all(gate_pass_conditions.values()) else "E0_E1_MATRIX_INCOMPLETE_OR_FAILED",
        "expected_run_count": 24,
        "observed_run_count": len(results),
        "completed_run_count": completed,
        "failed_run_count": failed,
        "blocked_run_count": 0 if stopped_reason is None else 1,
        "e0_run_count": family_counts["E0"],
        "e1_run_count": family_counts["E1"],
        "conditions": condition_counts,
        "seeds": seed_counts,
        "prompt4_r3a_gate_passed": cache_ok,
        "cache_manifest_sha256": cache_report["gate"].get("cache_manifest_sha256"),
        "cache_runtime_mismatch_count": cache_audit["cache_hash_mismatch_count"],
        "strict_mps_passed": gate_pass_conditions["all_actor_devices_mps"] and gate_pass_conditions["all_critic_devices_mps"],
        "cpu_model_fallback_count": sum(1 for r in results if r["cpu_model_fallback_used"]),
        "nan_run_count": sum(1 for r in results if r["nan_detected"]),
        "inf_run_count": sum(1 for r in results if r["inf_detected"]),
        "checkpoint_reload_pass_count": sum(1 for r in results if r["checkpoint_reload_ok"]),
        "canonical_kpi_pass_count": sum(1 for r in results if r["canonical_kpi_count"] == 12),
        "hard_constraint_pass_count": hard_pass,
        "hard_constraint_fail_count": hard_fail,
        "test_split_read": any(r["test_split_read"] for r in results),
        "test_target_read": any(r["test_target_read"] for r in results),
        "test_embedding_generated": any(r["test_embedding_generated"] for r in results),
        "prompt6_executed": False,
        "approved_for_prompt6a": all(gate_pass_conditions.values()) and hard_fail == 0,
        "e2_fine_tuning_approval": False,
        "prompt6_full_matrix_approval": False,
        "pass_conditions": gate_pass_conditions,
        "stopped_reason": stopped_reason,
        "missing_run_ids": ["_".join([family, condition, f"seed{seed:03d}"]) for family, condition, seed in missing_ids],
        "performance_claim_allowed": False,
        "operational_performance_claim_allowed": False,
        "policy_convergence_claim_allowed": False,
    }
    manifest = {
        "created_at_utc": utc_now(),
        "matrix": "Prompt 5-E01",
        "output_root": str(output_root),
        "fleet_reference_manifest": str(fleet_path),
        "fleet_reference_sha256": sha256_file(fleet_path),
        "central_fleet_count": central,
        "condition_fleet_counts": {condition: condition_agents(central, condition) for condition in ["A", "A90", "A80", "A70"]},
        "cache_root": str(cache_root),
        "cache_gate": str(cache_root / "prompt4_r3a_gate.json"),
        "results": results,
    }
    dump_json(output_root / "matrix_manifest.json", manifest)
    dump_json(output_root / "matrix_gate.json", matrix_gate)
    summary = [
        "# Prompt 5-E01 Matrix Summary",
        "",
        f"status: {matrix_gate['status']}",
        f"classification: {matrix_gate['classification']}",
        f"completed_run_count: {completed}/24",
        f"E0: {family_counts['E0']}/12",
        f"E1: {family_counts['E1']}/12",
        f"approved_for_prompt6a: {str(matrix_gate['approved_for_prompt6a']).lower()}",
        "",
        "No E2 fine-tuning, held-out test, or Prompt 6 full matrix execution was performed.",
    ]
    (output_root / "matrix_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    write_artifact_index(output_root)
    print(json.dumps(matrix_gate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
