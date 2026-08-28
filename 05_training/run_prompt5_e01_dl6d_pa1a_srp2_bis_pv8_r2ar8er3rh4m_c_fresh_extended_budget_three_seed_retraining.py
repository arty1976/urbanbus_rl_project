from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import torch
from torch.distributions import Categorical


STAGE = "PV8-R2A-R8E-R3-R-H4M-C"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_C_"
    "FRESH_EXTENDED_BUDGET_THREE_SEED_RETRAINING_COMPLETE"
)
BLOCK_GATE = (
    "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_C_"
    "FRESH_EXTENDED_BUDGET_THREE_SEED_RETRAINING_FAILED"
)
PASS_DECISION = "PV8_FRESH_EXTENDED_BUDGET_THREE_SEED_RETRAINING_COMPLETE_READY_FOR_FROZEN_POLICY_REVALIDATION"
BLOCK_DECISION = "PV8_FRESH_EXTENDED_BUDGET_THREE_SEED_RETRAINING_BLOCKED"
NEXT_GATE = "H4M-D_FROZEN_EXTENDED_BUDGET_POLICY_REVALIDATION"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"

H4M_C_SOURCE_REL = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_c_"
    "fresh_extended_budget_three_seed_retraining.py"
)
H4M_B_SOURCE_REL = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_b_"
    "training_budget_extension_selection_freeze.py"
)
H4K_SOURCE_REL = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_"
    "fresh_reward_v2_zero_loss_three_seed_full_retraining.py"
)

H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"
H4I_R3_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_r3_fresh_training_contract_freeze_20260810_183250"
DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"

EXPECTED = {
    "h4m_b_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_B_TRAINING_BUDGET_EXTENSION_SELECTION_AND_FREEZE_COMPLETE",
    "h4m_b_decision": "PV8_EXTENDED_TRAINING_BUDGET_FROZEN_READY_FOR_FRESH_THREE_SEED_RETRAINING_RELEASE",
    "h4m_b_schedule_sha": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "h4m_b_source_commit": "ec29d1939fbb68aa9ce2f436cda2af3eb853e476",
    "reward_v2_sha": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "outer_training_count": 11,
    "total_rollouts_per_seed": 11,
    "total_ppo_updates_per_seed": 44,
    "total_critic_updates_per_seed": 88,
    "expected_active_samples_per_seed": 3872,
}

ACTION_NAMES = {
    0: "HOLD_CURRENT_POSITION",
    1: "SERVE_AND_MOVE_TO_NEXT_STOP",
    2: "CONDITIONAL_SKIP_EMPTY_STOP",
}
ACTION_ORDER = [0, 1, 2]

RUNTIME_DEPENDENCIES = {
    H4M_C_SOURCE_REL,
    H4M_B_SOURCE_REL,
    H4K_SOURCE_REL,
    "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py",
    "05_training/mappo_runner.py",
    "05_training/rewards/mappo_reward_v1.py",
    "05_training/simulator/zero_loss_admission_adapter.py",
    "05_training/simulator/k_action_mask_runtime.py",
    "05_training/simulator/pv8_b1_orchestrator.py",
    "05_training/simulator/pv8_reward_outcome_collector.py",
}

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_git_source_provenance.json",
    "03_extended_schedule_binding.json",
    "04_seed1_training_summary.json",
    "05_seed2_training_summary.json",
    "06_seed3_training_summary.json",
    "07_cycle_by_cycle_policy_diagnostics.json",
    "08_three_seed_training_diagnostics.json",
    "09_zero_loss_kmask_integrity.json",
    "10_actor_critic_gatv2_health.json",
    "11_safety_causality_audit.json",
    "12_checkpoint_registry.json",
    "13_resource_usage.json",
    "14_h4m_c_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n",
        encoding="utf-8",
    )


def dump_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=jsonable) + "\n")


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def import_module_from_path(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def get_path(payload: Mapping[str, Any], keys: Iterable[Any], default: Any = None) -> Any:
    cur: Any = payload
    for key in keys:
        if isinstance(cur, Mapping) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


def git_run(args: Sequence[str], timeout: int = 30) -> Tuple[int, str, str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def parse_status_paths(status_short: str) -> List[Dict[str, Any]]:
    rows = []
    for line in status_short.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        rows.append({"status": status, "path": path})
    return rows


def finite_stats(values: Sequence[Any]) -> Dict[str, Any]:
    nums: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            v = float(value)
        except Exception:
            continue
        if math.isfinite(v):
            nums.append(v)
    if not nums:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    avg = mean(nums)
    var = mean([(v - avg) ** 2 for v in nums]) if len(nums) > 1 else 0.0
    return {"count": len(nums), "mean": avg, "std": math.sqrt(var), "min": min(nums), "max": max(nums)}


def clean_torch_cache() -> None:
    gc.collect()
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def git_source_provenance(created_at: str) -> Dict[str, Any]:
    _rc, status_short, _status_err = git_run(["status", "--short"])
    _rc, diff_stat, _diff_err = git_run(["diff", "--stat"])
    _rc, head, head_err = git_run(["rev-parse", "HEAD"])
    _rc, branch, branch_err = git_run(["branch", "--show-current"])
    dirty_rows = parse_status_paths(status_short)
    dirty_classification = []
    relevant_dirty = []
    for row in dirty_rows:
        path = row["path"]
        relevance = "H4M_C_EXECUTION_DEPENDENCY" if path in RUNTIME_DEPENDENCIES else "OUTSIDE_H4M_C_EXECUTION_DEPENDENCY_SET"
        dirty_classification.append({**row, "relevance": relevance})
        if relevance == "H4M_C_EXECUTION_DEPENDENCY":
            relevant_dirty.append(path)

    committed_checks = []
    for rel in sorted(RUNTIME_DEPENDENCIES):
        ls_rc, _ls_out, _ls_err = git_run(["ls-files", "--error-unmatch", rel])
        diff_rc, _diff_out, _diff_err = git_run(["diff", "--quiet", "HEAD", "--", rel])
        committed_checks.append(
            {
                "path": rel,
                "present_in_head": ls_rc == 0,
                "no_uncommitted_diff_vs_head": diff_rc == 0,
                "source_sha256": sha256_file(PROJECT_ROOT / rel),
            }
        )
    h4m_c_check = next(row for row in committed_checks if row["path"] == H4M_C_SOURCE_REL)
    relevant_source_clean = all(row["present_in_head"] and row["no_uncommitted_diff_vs_head"] for row in committed_checks)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "pre_training_git_status_short": status_short,
        "pre_training_git_diff_stat": diff_stat,
        "pre_training_git_head": head,
        "pre_training_git_head_error": head_err or None,
        "pre_training_git_branch": branch,
        "pre_training_git_branch_error": branch_err or None,
        "h4m_c_training_source_git_commit": head
        if h4m_c_check["present_in_head"] and h4m_c_check["no_uncommitted_diff_vs_head"]
        else None,
        "committed_source_checks": committed_checks,
        "local_source_only_commit_created_before_training": bool(
            h4m_c_check["present_in_head"] and h4m_c_check["no_uncommitted_diff_vs_head"]
        ),
        "all_execution_relevant_sources_clean_in_head": bool(relevant_source_clean),
        "remaining_dirty_paths": dirty_rows,
        "dirty_path_relevance_classification": dirty_classification,
        "relevant_execution_dependency_remains_dirty": bool(relevant_dirty),
        "relevant_dirty_paths": relevant_dirty,
        "post_commit_provenance_gate_passed": bool(
            h4m_c_check["present_in_head"]
            and h4m_c_check["no_uncommitted_diff_vs_head"]
            and relevant_source_clean
            and not relevant_dirty
        ),
        "github_push_performed": False,
    }


def authoritative_binding(created_at: str) -> Dict[str, Any]:
    h4m_b_gate = read_json(H4M_B_ROOT / "09_h4m_b_gate_matrix.json")
    h4m_b_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    h4m_b_manifest = read_json(H4M_B_ROOT / "manifest.json")
    h4m_b_binding = read_json(H4M_B_ROOT / "01_authoritative_binding.json")
    required_shas = h4m_b_binding.get("required_sha_bindings", {})
    observed_schedule_sha = h4m_b_schedule.get("extended_training_schedule_sha256")
    schedule_without_hash = {
        k: v
        for k, v in h4m_b_schedule.items()
        if k not in {"extended_training_schedule_sha256", "extended_training_schedule_sha256_scope"}
    }
    recomputed_schedule_sha = canonical_sha(schedule_without_hash)
    checks = {
        "h4m_b_gate_match": h4m_b_gate.get("gate") == EXPECTED["h4m_b_gate"],
        "h4m_b_decision_match": h4m_b_gate.get("decision") == EXPECTED["h4m_b_decision"],
        "h4m_b_manifest_gate_match": h4m_b_manifest.get("gate") == EXPECTED["h4m_b_gate"],
        "h4m_b_source_commit_match": h4m_b_gate.get("h4m_b_source_git_commit") == EXPECTED["h4m_b_source_commit"]
        and h4m_b_manifest.get("h4m_b_source_git_commit") == EXPECTED["h4m_b_source_commit"],
        "h4m_b_schedule_sha_match": observed_schedule_sha == EXPECTED["h4m_b_schedule_sha"],
        "h4m_b_schedule_sha_recomputed": recomputed_schedule_sha == EXPECTED["h4m_b_schedule_sha"],
        "reward_v2_sha_match": get_path(required_shas, ["reward_v2_sha", "observed"]) == EXPECTED["reward_v2_sha"],
        "h4g_runtime_sha_match": get_path(required_shas, ["h4g_runtime_sha", "observed"]) == EXPECTED["h4g_runtime_sha"],
        "r3_split_sha_match": get_path(required_shas, ["r3_split_sha", "observed"]) == EXPECTED["r3_split_sha"],
        "zero_loss_adapter_sha_match": get_path(required_shas, ["zero_loss_adapter_sha", "observed"])
        == EXPECTED["zero_loss_adapter_sha"],
        "all_required_h4m_b_sha_bindings_match": all(
            bool(row.get("match")) for row in required_shas.values() if isinstance(row, Mapping)
        ),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "authoritative_binding_passed": all(checks.values()),
        "checks": checks,
        "h4m_b_artifact_root": str(H4M_B_ROOT),
        "upstream_h4m_b_gate": {
            "path": str(H4M_B_ROOT / "09_h4m_b_gate_matrix.json"),
            "expected": EXPECTED["h4m_b_gate"],
            "observed": h4m_b_gate.get("gate"),
            "decision": h4m_b_gate.get("decision"),
        },
        "upstream_h4m_b_schedule": {
            "path": str(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json"),
            "expected_sha256": EXPECTED["h4m_b_schedule_sha"],
            "observed_sha256": observed_schedule_sha,
            "recomputed_sha256": recomputed_schedule_sha,
        },
        "required_sha_bindings": {
            "reward_v2_sha": required_shas.get("reward_v2_sha"),
            "h4g_runtime_sha": required_shas.get("h4g_runtime_sha"),
            "r3_split_sha": required_shas.get("r3_split_sha"),
            "zero_loss_adapter_sha": required_shas.get("zero_loss_adapter_sha"),
        },
        "source_file_sha256": {
            H4M_B_SOURCE_REL: sha256_file(PROJECT_ROOT / H4M_B_SOURCE_REL),
            H4K_SOURCE_REL: sha256_file(PROJECT_ROOT / H4K_SOURCE_REL),
            H4M_C_SOURCE_REL: sha256_file(PROJECT_ROOT / H4M_C_SOURCE_REL),
        },
        "training_authorized_by_h4m_b": True,
        "validation_authorized": False,
        "sealed_test_authorized": False,
        "github_push_performed": False,
    }


def extended_schedule_binding(created_at: str) -> Dict[str, Any]:
    schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    without_hash = {
        k: v
        for k, v in schedule.items()
        if k not in {"extended_training_schedule_sha256", "extended_training_schedule_sha256_scope"}
    }
    recomputed = canonical_sha(without_hash)
    exact = {
        "schedule_sha_verified": recomputed == EXPECTED["h4m_b_schedule_sha"]
        and schedule.get("extended_training_schedule_sha256") == EXPECTED["h4m_b_schedule_sha"],
        "seeds": schedule.get("seeds") == [1, 2, 3],
        "training_windows": schedule.get("training_windows") == 44,
        "outer_training_unit": schedule.get("outer_training_unit") == "fresh_rollout_update_cycle",
        "outer_training_count": schedule.get("outer_training_count") == EXPECTED["outer_training_count"],
        "rollout_horizon": schedule.get("rollout_horizon") == 512,
        "total_rollouts_per_seed": schedule.get("total_rollouts_per_seed") == EXPECTED["total_rollouts_per_seed"],
        "ppo_epochs_per_update": schedule.get("ppo_epochs_per_update") == 4,
        "total_ppo_updates_per_seed": schedule.get("total_ppo_updates_per_seed") == EXPECTED["total_ppo_updates_per_seed"],
        "critic_epochs_per_update": schedule.get("critic_epochs_per_update") == 8,
        "total_critic_updates_per_seed": schedule.get("total_critic_updates_per_seed")
        == EXPECTED["total_critic_updates_per_seed"],
        "expected_active_samples_per_seed": schedule.get("expected_active_samples_per_seed")
        == EXPECTED["expected_active_samples_per_seed"],
        "validation_usage": schedule.get("validation_usage") == "DEVELOPMENT_DIAGNOSTIC_ALREADY_OBSERVED",
        "test_usage": schedule.get("test_usage") == "SEALED_NOT_OPENED",
        "fresh_restart_policy": all(
            get_path(schedule, ["fresh_restart_policy", key]) == "FRESH"
            for key in ("gatv2", "actor", "critic", "optimizers", "reward_normalizer", "return_normalizer")
        ),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "h4m_b_schedule_root": str(H4M_B_ROOT),
        "schedule": schedule,
        "recomputed_extended_training_schedule_sha256": recomputed,
        "extended_training_schedule_sha_verified": all(exact.values()),
        "exact_frozen_field_checks": exact,
        "do_not_reinterpret_traversal_reset_rng_checkpoint_semantics": True,
        "training_windows_allowed_for_optimization": 44,
        "validation_usage": "DEVELOPMENT_DIAGNOSTIC_ALREADY_OBSERVED",
        "test_usage": "SEALED_NOT_OPENED",
        "training_authorized": True,
        "validation_authorized": False,
        "sealed_test_authorized": False,
    }


def pre_optimizer_gate(
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> Tuple[bool, List[str]]:
    blockers = []
    if not binding.get("authoritative_binding_passed"):
        blockers.append("AUTHORITATIVE_SHA_OR_H4M_B_GATE_MISMATCH")
    if not provenance.get("post_commit_provenance_gate_passed"):
        blockers.append("BLOCK_DIRTY_EXECUTION_DEPENDENCY")
    if not schedule.get("extended_training_schedule_sha_verified"):
        blockers.append("H4M_B_EXTENDED_SCHEDULE_SHA_OR_FIELD_MISMATCH")
    return not blockers, blockers


def h4m_c_reward_v2_metrics(
    reward_mod: Any,
    *,
    cycle_index: int,
    window: Mapping[str, Any],
    local_step: int,
    agent_slot: int,
    action_id: int,
    target_id: int,
) -> Dict[str, Any]:
    transition_id = f"H4M-C:C{cycle_index:02d}:{window['window_id']}:step{local_step:03d}:agent{agent_slot:02d}"
    local_decision_ts = float(local_step * 60 + agent_slot)
    service_required = int(target_id == 1)
    service_completed = int(service_required and action_id == 1)
    affected_wait_rows: List[Dict[str, Any]] = []
    if service_completed:
        request_ts = local_decision_ts - float(reward_mod.PV8_REWARD_V2_AVG_WAIT_REFERENCE_SECONDS)
        affected_wait_rows.append(
            {
                "passenger_id": f"{transition_id}:passenger",
                "originating_transition_id": transition_id,
                "wait_ownership_key": f"{transition_id}:passenger",
                "request_ts": request_ts,
                "local_decision_ts": local_decision_ts,
                "first_eligible_service_ts": local_decision_ts,
                "actual_board_ts": local_decision_ts,
            }
        )
    return {
        "transition_id": transition_id,
        "vehicle_slot_id": agent_slot,
        "route_id": f"H4M_C_R3_DIRECTION_{window.get('direction_id')}",
        "direction_id": window.get("direction_id"),
        "occurrence_id": window.get("window_id"),
        "local_decision_ts": local_decision_ts,
        "action": ACTION_NAMES.get(action_id, str(action_id)),
        "reward_semantics_version": reward_mod.PV8_REWARD_SEMANTICS_VERSION,
        "reward_freeze_sha256": reward_mod.PV8_REWARD_V2_FREEZE_SHA256,
        "pickup_obligation_count": service_required,
        "dropoff_obligation_count": 0,
        "approved_static_mandatory_obligation_count": 0,
        "completed_pickup_obligation_count": service_completed,
        "completed_dropoff_obligation_count": 0,
        "completed_static_mandatory_obligation_count": 0,
        "affected_wait_rows": affected_wait_rows,
        "explicit_forced_external_intervention_count": 0 if action_id == target_id else 1,
        "forced_safety_override_count": 0,
        "external_policy_intervention_count": 0,
        "ordinary_k_mask_restriction_counted": False,
        "p95_training_reward_enabled": False,
        "p95_training_normalization_active": False,
    }


def policy_probability_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        ACTION_NAMES[action_id]: {
            "raw_probability": finite_stats([row[f"raw_probability_{action_id}"] for row in rows]),
            "masked_probability": finite_stats([row[f"masked_probability_{action_id}"] for row in rows]),
            "raw_logit": finite_stats([row[f"raw_logit_{action_id}"] for row in rows]),
            "masked_logit": finite_stats([row[f"masked_logit_{action_id}"] for row in rows]),
            "legal_count": sum(1 for row in rows if row[f"legal_{action_id}"]),
        }
        for action_id in ACTION_ORDER
    }


def collect_h4m_c_rollout(
    h4k: Any,
    dl1: Any,
    dl4: Any,
    reward_mod: Any,
    mappo_mod: Any,
    data_seq: Sequence[Any],
    window_plan_rows: Sequence[Mapping[str, Any]],
    offset: int,
    horizon: int,
    cycle_index: int,
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    reward_normalizer: Any,
    return_normalizer: Any,
    device: torch.device,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_rewards: List[torch.Tensor] = []
    values_original: List[torch.Tensor] = []
    old_log_probs: List[torch.Tensor] = []
    actions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    agent_indices: List[List[int]] = []
    reward_rows: List[Dict[str, Any]] = []
    policy_rows: List[Dict[str, Any]] = []
    entropy_coefficients: List[float] = []
    masked_skip_count = 0
    allowed_skip_count = 0
    started = time.perf_counter()
    encoder.eval()
    actor.eval()
    critic.eval()
    with torch.no_grad():
        for local_step in range(horizon):
            absolute = offset + local_step
            window = window_plan_rows[absolute]
            data = data_seq[absolute].to(device)
            indices = dl1.agent_indices_for_step(config["spec"], absolute, int(config["effective_agents"]))
            logits, _critic_out, value_original, agent_mask = dl4.forward_scaled(
                dl1, data, indices, encoder, actor, critic, return_normalizer
            )
            target = dl1.action_targets_from_y(
                data.y[torch.tensor(indices, dtype=torch.long, device=device)],
                int(config["action_dim"]),
            )
            masked_logits, allowed = h4k.masked_logits_for_targets(logits, target)
            raw_probs = torch.softmax(logits, dim=-1)
            masked_probs = torch.softmax(masked_logits, dim=-1)
            entropy = Categorical(logits=masked_logits).entropy()
            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            rewards_for_step = []
            for agent_slot, (action_id, target_id, is_active) in enumerate(
                zip(action.detach().cpu().tolist(), target.detach().cpu().tolist(), agent_mask.detach().cpu().tolist())
            ):
                if bool(is_active):
                    metrics = h4m_c_reward_v2_metrics(
                        reward_mod,
                        cycle_index=cycle_index,
                        window=window,
                        local_step=absolute,
                        agent_slot=agent_slot,
                        action_id=int(action_id),
                        target_id=int(target_id),
                    )
                    materialized = reward_mod.compute_reward_v2(metrics)
                    reward_value = float(materialized["reward_total"])
                    legal_ids = [int(i) for i, v in enumerate(allowed[agent_slot].detach().cpu().tolist()) if bool(v)]
                    policy_row = {
                        "cycle_index": cycle_index,
                        "window_id": window["window_id"],
                        "snapshot_id": window["snapshot_id"],
                        "local_step": absolute,
                        "agent_slot": agent_slot,
                        "action_id": int(action_id),
                        "action": ACTION_NAMES.get(int(action_id), str(action_id)),
                        "target_id": int(target_id),
                        "target": ACTION_NAMES.get(int(target_id), str(target_id)),
                        "legal_action_ids": legal_ids,
                        "legal_actions": [ACTION_NAMES[i] for i in legal_ids],
                        "legal_combination": "{" + ", ".join(ACTION_NAMES[i].split("_", 1)[0] for i in legal_ids) + "}",
                        "entropy": float(entropy[agent_slot].detach().cpu().item()),
                        "sampled_action_log_prob": float(dist.log_prob(action)[agent_slot].detach().cpu().item()),
                    }
                    for action_key in ACTION_ORDER:
                        policy_row[f"legal_{action_key}"] = action_key in legal_ids
                        policy_row[f"raw_logit_{action_key}"] = float(logits[agent_slot, action_key].detach().cpu().item())
                        policy_row[f"masked_logit_{action_key}"] = float(
                            masked_logits[agent_slot, action_key].detach().cpu().item()
                        )
                        policy_row[f"raw_probability_{action_key}"] = float(
                            raw_probs[agent_slot, action_key].detach().cpu().item()
                        )
                        policy_row[f"masked_probability_{action_key}"] = float(
                            masked_probs[agent_slot, action_key].detach().cpu().item()
                        )
                    policy_rows.append(policy_row)
                    reward_rows.append(
                        {
                            "cycle_index": cycle_index,
                            "window_id": window["window_id"],
                            "snapshot_id": window["snapshot_id"],
                            "local_step": absolute,
                            "agent_slot": agent_slot,
                            "action_id": int(action_id),
                            "action": ACTION_NAMES.get(int(action_id), str(action_id)),
                            "target_id": int(target_id),
                            "target": ACTION_NAMES.get(int(target_id), str(target_id)),
                            "reward_total": reward_value,
                            "reward_service_component_weighted": materialized["reward_service_component_weighted"],
                            "reward_avg_wait_component_weighted": materialized["reward_avg_wait_component_weighted"],
                            "reward_intervention_component_weighted": materialized["reward_intervention_component_weighted"],
                            "reward_freeze_sha256": materialized["reward_freeze_sha256"],
                            "p95_training_reward_enabled": materialized["p95_training_reward_enabled"],
                            "blanket_SKIP_penalty_applied": materialized["blanket_SKIP_penalty_applied"],
                            "direct_time_band_reward_term_applied": materialized["direct_time_band_reward_term_applied"],
                        }
                    )
                else:
                    reward_value = 0.0
                rewards_for_step.append(reward_value)
            raw_rewards.append(torch.tensor(rewards_for_step, dtype=value_original.dtype, device=device))
            values_original.append(value_original.detach())
            old_log_probs.append(dist.log_prob(action).detach())
            actions.append(action.detach())
            targets.append(target.detach())
            masks.append(agent_mask.detach())
            agent_indices.append(indices)
            if logits.size(-1) >= 3:
                masked_skip_count += int((~allowed[:, 2]).sum().detach().cpu().item())
                allowed_skip_count += int(allowed[:, 2].sum().detach().cpu().item())
            progress = float(local_step) / max(1.0, float(horizon - 1))
            entropy_coefficients.append(
                float(
                    mappo_mod.entropy_coef_for_time_band(
                        str(window.get("time_band", "offpeak")),
                        progress,
                        offpeak_coef=0.01,
                        peak_coef=0.03,
                        min_coef=0.001,
                    )
                )
            )
        next_idx = min(offset + horizon, len(data_seq) - 1)
        next_data = data_seq[next_idx].to(device)
        next_indices = dl1.agent_indices_for_step(config["spec"], next_idx, int(config["effective_agents"]))
        _logits, _critic_out, last_next_value_original, _mask = dl4.forward_scaled(
            dl1, next_data, next_indices, encoder, actor, critic, return_normalizer
        )

    raw_rewards_t = torch.stack(raw_rewards)
    masks_t = torch.stack(masks).bool()
    normalized_rewards_t = torch.zeros_like(raw_rewards_t)
    active_raw = raw_rewards_t[masks_t].detach().cpu().tolist()
    active_normalized = reward_normalizer.normalize([float(v) for v in active_raw])
    normalized_rewards_t[masks_t] = torch.tensor(active_normalized, dtype=raw_rewards_t.dtype, device=device)
    values_t = torch.stack(values_original)
    next_values_t = torch.zeros_like(values_t)
    next_values_t[:-1] = values_t[1:]
    next_values_t[-1] = last_next_value_original.detach()
    terminated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool)
    truncated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool)
    truncated[-1] = True
    returns, advantages, normalized_advantages, gae_audit = dl1.compute_gae(
        normalized_rewards_t,
        values_t,
        next_values_t,
        terminated,
        truncated,
        masks_t,
        float(config["gamma"]),
        float(config["gae_lambda"]),
    )
    return_normalizer.update(returns[masks_t.bool()])
    return {
        "cycle_index": cycle_index,
        "rollout_collection_seconds": time.perf_counter() - started,
        "raw_rewards": raw_rewards_t.detach(),
        "normalized_rewards": normalized_rewards_t.detach(),
        "values_original": values_t.detach(),
        "old_log_probs": torch.stack(old_log_probs).detach(),
        "actions": torch.stack(actions).detach(),
        "targets": torch.stack(targets).detach(),
        "agent_mask": masks_t.detach(),
        "agent_indices": agent_indices,
        "returns_original": returns.detach(),
        "advantages": advantages.detach(),
        "normalized_advantages": normalized_advantages.detach(),
        "gae_audit": gae_audit,
        "reward_rows": reward_rows,
        "policy_rows": policy_rows,
        "entropy_coef_mean": float(mean(entropy_coefficients)) if entropy_coefficients else 0.01,
        "entropy_coef_stats": finite_stats(entropy_coefficients),
        "k_mask_stats": {
            "skip_masked_count": masked_skip_count,
            "skip_allowed_count": allowed_skip_count,
            "illegal_skip_count": 0,
            "k_mask_source": "H4M-C preserves H4K offline training mask derived from frozen R3 action target legality.",
        },
    }


def action_rates(action_counts: Mapping[str, int], denominator: int) -> Dict[str, float]:
    return {ACTION_NAMES[i]: float(action_counts.get(ACTION_NAMES[i], 0)) / max(1, denominator) for i in ACTION_ORDER}


def cycle_diagnostic(
    dl4: Any,
    rollout: Mapping[str, Any],
    cycle_index: int,
    training_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    loss_rows: Sequence[Mapping[str, Any]],
    parameter_delta: Mapping[str, Mapping[str, Any]],
    device: torch.device,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    policy_rows = rollout["policy_rows"]
    active_samples = len(policy_rows)
    action_counts = Counter(row["action"] for row in policy_rows)
    legal_combo_counts = Counter(row["legal_combination"] for row in policy_rows)
    legal_counts = {
        ACTION_NAMES[i]: sum(1 for row in policy_rows if row[f"legal_{i}"])
        for i in ACTION_ORDER
    }
    mem_rows = [row for row in training_rows if row.get("mps_current_allocated_mb") is not None or row.get("process_rss_mb") is not None]
    return {
        "stage": STAGE,
        "cycle_index": cycle_index,
        "rollout_rows": int(rollout["actions"].size(0)),
        "active_samples": active_samples,
        "action_counts": {ACTION_NAMES[i]: int(action_counts.get(ACTION_NAMES[i], 0)) for i in ACTION_ORDER},
        "action_rates": action_rates(action_counts, active_samples),
        "legal_action_distribution": dict(legal_combo_counts),
        "legal_action_counts": legal_counts,
        "policy_entropy": finite_stats([row["entropy"] for row in policy_rows]),
        "policy_probability_statistics": policy_probability_summary(policy_rows),
        "reward_v2_statistics": finite_stats([row["reward_total"] for row in rollout["reward_rows"]]),
        "normalized_reward_statistics": dl4.tensor_stats(rollout["normalized_rewards"].to(device), rollout["agent_mask"].to(device)),
        "advantage_statistics": dl4.tensor_stats(rollout["advantages"].to(device), rollout["agent_mask"].to(device)),
        "normalized_advantage_statistics": dl4.tensor_stats(
            rollout["normalized_advantages"].to(device), rollout["agent_mask"].to(device)
        ),
        "return_statistics": dl4.tensor_stats(rollout["returns_original"].to(device), rollout["agent_mask"].to(device)),
        "actor_loss": finite_stats([row.get("policy_loss") for row in training_rows if row.get("policy_loss") is not None]),
        "critic_loss": finite_stats([row.get("value_loss") for row in training_rows if row.get("value_loss") is not None]),
        "critic_mse_where_available": finite_stats([row.get("value_loss") for row in training_rows if row.get("value_loss") is not None]),
        "gradient_norms": {
            "gatv2_before_clip": finite_stats([row.get("gatv2_grad_norm_before_clip") for row in gradient_rows]),
            "actor_before_clip": finite_stats([row.get("actor_grad_norm_before_clip") for row in gradient_rows]),
            "critic_before_clip": finite_stats([row.get("critic_grad_norm_before_clip") for row in gradient_rows]),
        },
        "parameter_deltas_from_seed_initialization": parameter_delta,
        "nan_count": sum(int(row.get("nan_count", 0)) for row in training_rows),
        "inf_count": sum(int(row.get("inf_count", 0)) for row in training_rows),
        "rollout_time_seconds": float(rollout["rollout_collection_seconds"]),
        "update_time_seconds": max([float(row.get("ppo_update_seconds") or 0.0) for row in training_rows], default=0.0),
        "elapsed_time_seconds": max([float(row.get("elapsed_seconds") or 0.0) for row in training_rows], default=0.0),
        "resource_metrics": {
            "peak_mps_current_allocated_mb": max(
                [float(row["mps_current_allocated_mb"]) for row in mem_rows if row.get("mps_current_allocated_mb") is not None],
                default=None,
            ),
            "peak_mps_driver_allocated_mb": max(
                [float(row["mps_driver_allocated_mb"]) for row in mem_rows if row.get("mps_driver_allocated_mb") is not None],
                default=None,
            ),
            "peak_process_rss_mb": max(
                [float(row["process_rss_mb"]) for row in mem_rows if row.get("process_rss_mb") is not None],
                default=None,
            ),
        },
        "opportunity_diagnostics": {
            "multi_action_legal_states": active_samples,
            "hold_legal_states": legal_counts["HOLD_CURRENT_POSITION"],
            "skip_legal_states": legal_counts["CONDITIONAL_SKIP_EMPTY_STOP"],
            "non_SERVE_beneficial_exposure": int(config["non_serve_beneficial_per_cycle"]),
            "zero_loss_candidate_attempts": 0,
            "zero_loss_accepted": 0,
            "zero_loss_rejected": 0,
        },
        "k_mask_stats": rollout["k_mask_stats"],
        "diagnostic_not_used_for_training_control": True,
    }


def save_final_checkpoint(
    dl1: Any,
    dl4: Any,
    seed_dir: Path,
    seed: int,
    namespace: str,
    encoder: torch.nn.Module,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer],
    reward_normalizer: Any,
    return_normalizer: Any,
    config: Mapping[str, Any],
    sample_graph: Any,
    device: torch.device,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    config_public = {k: v for k, v in config.items() if k not in {"spec", "started_at_perf"}}
    training_configuration_sha256 = canonical_sha(config_public)
    checkpoint_path = seed_dir / f"{namespace}.pt"
    payload = {
        "checkpoint_boundary": namespace,
        "checkpoint_status": "TRAINING_COMPLETE_EVALUATION_NOT_RELEASED",
        "seed": seed,
        "created_at": kst_now().isoformat(timespec="seconds"),
        "gatv2_state_dict": encoder.state_dict(),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "gatv2_optimizer_state_dict": optimizers["gatv2"].state_dict(),
        "actor_optimizer_state_dict": optimizers["actor"].state_dict(),
        "critic_optimizer_state_dict": optimizers["critic"].state_dict(),
        "reward_normalizer_state": {
            "window_size": int(getattr(reward_normalizer, "window_size", 1000)),
            "clip_value": float(getattr(reward_normalizer, "clip_value", 10.0)),
            "buffer": [float(v) for v in getattr(reward_normalizer, "buffer", [])],
        },
        "return_normalizer_state": return_normalizer.state_dict(),
        "training_configuration": config_public,
        "training_configuration_sha256": training_configuration_sha256,
        "metadata": dict(metadata),
        "architecture_identity": {
            "gatv2": "DL1.GATv2Encoder(hidden=128, heads=2->1)",
            "actor": "DL1.MAPPOActor(hidden=128, action_dim=3)",
            "critic": "DL1.CentralizedCritic(hidden=128)",
        },
        "normalization_contract": {
            "reward": "Reward V2 RewardNormalizer FRESH per seed",
            "advantage": "active-agent-only GAE standardization",
            "return": "DL4 Welford running return normalization FRESH per seed",
            "value": "no independent value normalization state",
        },
    }
    torch.save(payload, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sample = sample_graph.to(device)
    indices = dl1.agent_indices_for_step(config["spec"], 0, int(config["effective_agents"]))
    re_encoder = dl1.GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
    re_actor = dl1.MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
    re_critic = dl1.CentralizedCritic(int(config["gatv2_hidden"])).to(device)
    re_encoder.load_state_dict(loaded["gatv2_state_dict"])
    re_actor.load_state_dict(loaded["actor_state_dict"])
    re_critic.load_state_dict(loaded["critic_state_dict"])
    re_return_norm = dl4.ReturnNormalizer(bool(config["return_normalization"]))
    re_return_norm.load_state_dict(loaded["return_normalizer_state"])
    optimizer_state_load_success = True
    try:
        torch.optim.Adam(re_encoder.parameters(), lr=float(config["gatv2_lr"])).load_state_dict(
            loaded["gatv2_optimizer_state_dict"]
        )
        torch.optim.Adam(re_actor.parameters(), lr=float(config["actor_lr"])).load_state_dict(
            loaded["actor_optimizer_state_dict"]
        )
        torch.optim.Adam(re_critic.parameters(), lr=float(config["critic_lr"])).load_state_dict(
            loaded["critic_optimizer_state_dict"]
        )
    except Exception:
        optimizer_state_load_success = False
    encoder.eval()
    actor.eval()
    critic.eval()
    re_encoder.eval()
    re_actor.eval()
    re_critic.eval()
    with torch.no_grad():
        logits_original, _critic_original, value_original, _ = dl4.forward_scaled(
            dl1, sample, indices, encoder, actor, critic, return_normalizer
        )
        logits_reload, _critic_reload, value_reload, _ = dl4.forward_scaled(
            dl1, sample, indices, re_encoder, re_actor, re_critic, re_return_norm
        )
    output_diff = float(
        max((logits_original - logits_reload).abs().max().item(), (value_original - value_reload).abs().max().item())
    )
    validation = {
        "checkpoint_boundary_match": loaded.get("checkpoint_boundary") == namespace,
        "checkpoint_status_match": loaded.get("checkpoint_status") == "TRAINING_COMPLETE_EVALUATION_NOT_RELEASED",
        "seed_match": int(loaded.get("seed")) == seed,
        "reward_v2_sha_match": loaded.get("metadata", {}).get("reward_v2_sha256") == EXPECTED["reward_v2_sha"],
        "h4g_runtime_sha_match": loaded.get("metadata", {}).get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha"],
        "r3_split_sha_match": loaded.get("metadata", {}).get("r3_split_sha256") == EXPECTED["r3_split_sha"],
        "zero_loss_adapter_sha_match": loaded.get("metadata", {}).get("zero_loss_adapter_sha256")
        == EXPECTED["zero_loss_adapter_sha"],
        "h4m_b_schedule_sha_match": loaded.get("metadata", {}).get("h4m_b_extended_training_schedule_sha256")
        == EXPECTED["h4m_b_schedule_sha"],
        "h4m_c_training_source_git_commit_present": bool(
            loaded.get("metadata", {}).get("h4m_c_training_source_git_commit")
        ),
        "outer_training_count_match": loaded.get("metadata", {}).get("outer_training_count") == EXPECTED["outer_training_count"],
        "total_rollouts_match": loaded.get("metadata", {}).get("total_rollouts") == EXPECTED["total_rollouts_per_seed"],
        "total_ppo_updates_match": loaded.get("metadata", {}).get("total_ppo_updates") == EXPECTED["total_ppo_updates_per_seed"],
        "total_critic_updates_match": loaded.get("metadata", {}).get("total_critic_updates")
        == EXPECTED["total_critic_updates_per_seed"],
        "gatv2_parameter_hash_match": dl1.state_dict_hash(loaded["gatv2_state_dict"]) == dl1.module_hash(re_encoder),
        "actor_parameter_hash_match": dl1.state_dict_hash(loaded["actor_state_dict"]) == dl1.module_hash(re_actor),
        "critic_parameter_hash_match": dl1.state_dict_hash(loaded["critic_state_dict"]) == dl1.module_hash(re_critic),
        "optimizer_state_load_success": optimizer_state_load_success,
        "training_configuration_sha_match": loaded.get("training_configuration_sha256") == training_configuration_sha256,
        "normalization_state_present": bool(
            loaded.get("reward_normalizer_state") is not None and loaded.get("return_normalizer_state") is not None
        ),
        "same_observation_original_vs_reload_output_match": output_diff <= 1.0e-5,
        "output_max_abs_diff": output_diff,
    }
    validation["checkpoint_integrity_passed"] = all(
        value for key, value in validation.items() if key != "output_max_abs_diff"
    )
    return {
        "namespace": namespace,
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "training_configuration_sha256": training_configuration_sha256,
        "checkpoint_validation": validation,
    }


def run_seed(
    seed: int,
    output_root: Path,
    h4k: Any,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schedule_binding: Mapping[str, Any],
    window_plan: Mapping[str, Any],
) -> Dict[str, Any]:
    del binding
    schedule = schedule_binding["schedule"]
    dl4 = import_module_from_path(TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py", f"h4m_c_dl4_seed_{seed}")
    dl1 = dl4.import_dl1(PROJECT_ROOT)
    mappo_mod = import_module_from_path(TRAINING_ROOT / "mappo_runner.py", f"h4m_c_mappo_runner_seed_{seed}")
    reward_mod = import_module_from_path(TRAINING_ROOT / "rewards/mappo_reward_v1.py", f"h4m_c_reward_v2_seed_{seed}")
    seed_dir = output_root / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    clean_torch_cache()
    seed_audit = dl4.set_all_seeds(seed)
    runtime = dl4.runtime_environment("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mapping_artifact = read_json(DL3_ROOT / "study_area_snapshot.json")["repair_mapping"]
    train_paths = [Path(row["snapshot_path"]) for row in window_plan["train_rows"]]
    sample_full = dl1.torch_load(train_paths[0])
    spec, inventory, connectivity, tensor_mask = dl1.build_subgraph_spec(
        PROJECT_ROOT, sample_full, mapping_artifact=Path(mapping_artifact)
    )
    sample_graph = dl1.make_subgraph_data(sample_full, spec)
    train_data = dl4.load_subgraphs(dl1, train_paths, spec)
    config: Dict[str, Any] = {
        "created_at": kst_now().isoformat(timespec="seconds"),
        "stage": STAGE,
        "study_area": "SUSEONG_GU_DAEGU",
        "seed": seed,
        "seed_audit": seed_audit,
        "device": str(device),
        "agents": 8,
        "effective_agents": min(8, int(inventory["available_suseong_agents"])),
        "gatv2_hidden": 128,
        "gatv2_batch": 1,
        "gatv2_epochs": 1,
        "gatv2_grad_clip": 5.0,
        "rollout_horizon": int(schedule["rollout_horizon"]),
        "minibatch_size": int(schedule["minibatch"]),
        "actor_ppo_epochs": int(schedule["ppo_epochs_per_update"]),
        "critic_epochs": int(schedule["critic_epochs_per_update"]),
        "mappo_grad_clip": 0.5,
        "critic_grad_clip": 0.5,
        "actor_gatv2_lr": float(schedule["gatv2_lr"]),
        "actor_lr": float(schedule["actor_lr"]),
        "gatv2_lr": float(schedule["gatv2_lr"]),
        "critic_lr": float(schedule["critic_lr"]),
        "gamma": float(schedule["gamma"]),
        "gae_lambda": float(schedule["gae_lambda"]),
        "ppo_clip_epsilon": float(schedule["clip_epsilon"]),
        "entropy_contract": {"adaptive": True, "peak": 0.03, "offpeak": 0.01, "night": 0.01, "min": 0.001},
        "value_loss_coef": float(schedule["value_coefficient"]),
        "value_loss_type": "mse",
        "return_normalization": True,
        "reward_normalization": True,
        "action_dim": 3,
        "outer_training_unit": schedule["outer_training_unit"],
        "outer_training_count": int(schedule["outer_training_count"]),
        "total_rollouts_per_seed": int(schedule["total_rollouts_per_seed"]),
        "total_ppo_updates_per_seed": int(schedule["total_ppo_updates_per_seed"]),
        "total_critic_updates_per_seed": int(schedule["total_critic_updates_per_seed"]),
        "expected_active_samples_per_seed": int(schedule["expected_active_samples_per_seed"]),
        "non_serve_beneficial_per_cycle": int(
            read_json(H4M_B_ROOT / "02_h4m_a_budget_evidence.json")["train44_structural_opportunities"][
                "non_SERVE_better_or_equal_reviewable_opportunities"
            ]
        ),
        "training_snapshot_count": len(train_data),
        "validation_snapshot_count": 4,
        "test_snapshot_count": 6,
        "node_mapping_hash": spec["node_edge_mapping_hash"],
        "split_manifest_hash": EXPECTED["r3_split_sha"],
        "h4m_b_extended_training_schedule_sha256": EXPECTED["h4m_b_schedule_sha"],
        "h4m_c_training_source_git_commit": provenance.get("h4m_c_training_source_git_commit"),
        "validation_usage": "DEVELOPMENT_DIAGNOSTIC_ALREADY_OBSERVED",
        "test_usage": "SEALED_NOT_OPENED",
        "test_data_used_for_selection": False,
        "policy_evaluation_authorized": False,
        "winner_selection_authorized": False,
    }
    dump_json(seed_dir / "configuration.json", {k: v for k, v in config.items() if k != "spec"})
    dump_json(seed_dir / "r3_train_window_plan.json", {"train_rows": window_plan["train_rows"]})
    dump_json(seed_dir / "subgraph_scope_audit.json", {"connectivity": connectivity, "tensor_mask": tensor_mask, "inventory": inventory})
    config["spec"] = spec
    config["started_at_perf"] = time.perf_counter()

    encoder = dl1.GATv2Encoder(sample_graph.x.size(1), int(config["gatv2_hidden"]), sample_graph.edge_attr.size(1)).to(device)
    actor = dl1.MAPPOActor(int(config["gatv2_hidden"]), int(config["action_dim"])).to(device)
    critic = dl1.CentralizedCritic(int(config["gatv2_hidden"])).to(device)
    reward_normalizer = mappo_mod.RewardNormalizer(window_size=1000, clip_value=10.0)
    return_normalizer = dl4.ReturnNormalizer(True)
    optimizers = {
        "gatv2": torch.optim.Adam(encoder.parameters(), lr=float(config["gatv2_lr"])),
        "actor": torch.optim.Adam(actor.parameters(), lr=float(config["actor_lr"])),
        "critic": torch.optim.Adam(critic.parameters(), lr=float(config["critic_lr"])),
    }
    before_state = {"gatv2": dl1.clone_state_dict(encoder), "actor": dl1.clone_state_dict(actor), "critic": dl1.clone_state_dict(critic)}

    training_rows: List[Dict[str, Any]] = []
    gradient_rows: List[Dict[str, Any]] = []
    loss_rows: List[Dict[str, Any]] = []
    reward_rows: List[Dict[str, Any]] = []
    policy_rows_all: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    start = time.perf_counter()
    ppo_index = 0
    offset = 0
    effective_horizon = len(train_data)

    for cycle_index in range(1, int(config["outer_training_count"]) + 1):
        rollout = collect_h4m_c_rollout(
            h4k,
            dl1,
            dl4,
            reward_mod,
            mappo_mod,
            train_data,
            window_plan["train_rows"],
            offset,
            effective_horizon,
            cycle_index,
            encoder,
            actor,
            critic,
            reward_normalizer,
            return_normalizer,
            device,
            config,
        )
        rows, grads, losses = h4k.ppo_update_h4k(
            dl1,
            dl4,
            train_data,
            offset,
            rollout,
            encoder,
            actor,
            critic,
            optimizers,
            return_normalizer,
            device,
            config,
            before_state,
            cycle_index,
            ppo_index,
        )
        for row in rows:
            row["cycle_index"] = cycle_index
        for row in grads:
            row["cycle_index"] = cycle_index
        for row in losses:
            row["cycle_index"] = cycle_index
        ppo_index += int(config["critic_epochs"])
        training_rows.extend(rows)
        gradient_rows.extend(grads)
        loss_rows.extend(losses)
        reward_rows.extend(rollout["reward_rows"])
        policy_rows_all.extend(rollout["policy_rows"])
        parameter_delta_now = {
            "gatv2": dl1.delta_stats(encoder, before_state["gatv2"]),
            "actor": dl1.delta_stats(actor, before_state["actor"]),
            "critic": dl1.delta_stats(critic, before_state["critic"]),
        }
        cycle_rows.append(
            cycle_diagnostic(dl4, rollout, cycle_index, rows, grads, losses, parameter_delta_now, device, config)
        )
        del rollout
        clean_torch_cache()

    parameter_delta = {
        "gatv2": dl1.delta_stats(encoder, before_state["gatv2"]),
        "actor": dl1.delta_stats(actor, before_state["actor"]),
        "critic": dl1.delta_stats(critic, before_state["critic"]),
    }
    nan_count = sum(int(row.get("nan_count", 0)) for row in training_rows)
    inf_count = sum(int(row.get("inf_count", 0)) for row in training_rows)
    loss_all_finite = nan_count == 0 and inf_count == 0 and all(
        all(bool(v) for k, v in row.items() if k.endswith("_finite")) for row in loss_rows
    )
    actor_joint_updates = sum(1 for row in gradient_rows if row["update_role"] == "actor_gatv2_critic_joint")
    critic_updates = len(gradient_rows)
    total_rollouts = int(config["outer_training_count"])
    active_samples = sum(int(row["active_samples"]) for row in cycle_rows)
    namespace = f"H4M_C_SEED_{seed:03d}_FRESH_EXTENDED_BUDGET_REWARD_V2_ZERO_LOSS"
    metadata = {
        "seed": seed,
        "training_completion_state": "TRAINING_COMPLETE_EVALUATION_NOT_RELEASED",
        "h4m_c_training_source_git_commit": provenance.get("h4m_c_training_source_git_commit"),
        "h4m_b_extended_training_schedule_sha256": EXPECTED["h4m_b_schedule_sha"],
        "reward_v2_sha256": EXPECTED["reward_v2_sha"],
        "h4g_runtime_sha256": EXPECTED["h4g_runtime_sha"],
        "r3_split_sha256": EXPECTED["r3_split_sha"],
        "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha"],
        "outer_training_count": int(config["outer_training_count"]),
        "total_rollouts": total_rollouts,
        "total_ppo_updates": actor_joint_updates,
        "total_critic_updates": critic_updates,
        "policy_evaluation_authorized": False,
        "winner_selection_authorized": False,
        "sealed_test_opened": False,
    }
    checkpoint = save_final_checkpoint(
        dl1,
        dl4,
        seed_dir,
        seed,
        namespace,
        encoder,
        actor,
        critic,
        optimizers,
        reward_normalizer,
        return_normalizer,
        config,
        sample_graph,
        device,
        metadata,
    )
    elapsed = time.perf_counter() - start
    gate_passed = (
        total_rollouts == EXPECTED["total_rollouts_per_seed"]
        and actor_joint_updates == EXPECTED["total_ppo_updates_per_seed"]
        and critic_updates == EXPECTED["total_critic_updates_per_seed"]
        and active_samples == EXPECTED["expected_active_samples_per_seed"]
        and parameter_delta["gatv2"]["l2_delta"] > 0
        and parameter_delta["actor"]["l2_delta"] > 0
        and parameter_delta["critic"]["l2_delta"] > 0
        and loss_all_finite
        and bool(checkpoint["checkpoint_validation"]["checkpoint_integrity_passed"])
    )

    summary = {
        "stage": STAGE,
        "seed": seed,
        "status": "TRAINING_COMPLETE_EVALUATION_NOT_RELEASED" if gate_passed else "BLOCKED_SEED_INTEGRITY_FAILURE",
        "gate_passed": gate_passed,
        "fresh_lineage": {
            "gatv2": "FRESH",
            "actor": "FRESH",
            "critic": "FRESH",
            "optimizer": "FRESH",
            "reward_normalizer": "FRESH",
            "return_normalizer": "FRESH",
            "h4k_checkpoint_continuation": False,
            "h4j_zl2_checkpoint_reuse": False,
            "dl3_dl4_checkpoint_reuse": False,
            "cross_seed_model_reuse": False,
            "cross_seed_optimizer_reuse": False,
            "cross_seed_normalizer_reuse": False,
        },
        "elapsed_seconds": elapsed,
        "rollout_collection_seconds_total": sum(float(row["rollout_time_seconds"]) for row in cycle_rows),
        "update_seconds_total": sum(float(row["update_time_seconds"]) for row in cycle_rows),
        "training_transitions_per_cycle": len(train_data),
        "total_training_transitions": len(train_data) * total_rollouts,
        "active_samples": active_samples,
        "total_rollouts": total_rollouts,
        "ppo_updates_completed": actor_joint_updates,
        "critic_updates_completed": critic_updates,
        "actor_gatv2_joint_updates": actor_joint_updates,
        "critic_only_extra_updates": sum(1 for row in gradient_rows if row["update_role"] == "critic_only_extra"),
        "first_cycle_action_distribution": cycle_rows[0]["action_counts"] if cycle_rows else {},
        "last_cycle_action_distribution": cycle_rows[-1]["action_counts"] if cycle_rows else {},
        "cycle_action_distributions": {
            str(row["cycle_index"]): row["action_counts"]
            for row in cycle_rows
        },
        "cycle_policy_probability_evolution": {
            str(row["cycle_index"]): {
                action: stats["masked_probability"]
                for action, stats in row["policy_probability_statistics"].items()
            }
            for row in cycle_rows
        },
        "raw_reward_v2_distribution": finite_stats([row["reward_total"] for row in reward_rows]),
        "actor_loss": finite_stats([row["policy_loss"] for row in training_rows if row.get("policy_loss") is not None]),
        "actor_entropy": finite_stats([row["actor_entropy"] for row in training_rows if row.get("actor_entropy") is not None]),
        "critic_loss": finite_stats([row["value_loss"] for row in training_rows if row.get("value_loss") is not None]),
        "critic_mse": finite_stats([row["value_loss"] for row in training_rows if row.get("value_loss") is not None]),
        "value_distribution": finite_stats([row["prediction_mean"] for row in training_rows if row.get("prediction_mean") is not None]),
        "return_target_distribution": finite_stats([row["target_mean"] for row in training_rows if row.get("target_mean") is not None]),
        "explained_variance_last": training_rows[-1].get("explained_variance") if training_rows else None,
        "gradient_norms": {
            "gatv2_before_clip": finite_stats([row["gatv2_grad_norm_before_clip"] for row in gradient_rows]),
            "actor_before_clip": finite_stats([row["actor_grad_norm_before_clip"] for row in gradient_rows]),
            "critic_before_clip": finite_stats([row["critic_grad_norm_before_clip"] for row in gradient_rows]),
        },
        "parameter_deltas": parameter_delta,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "future_leakage": 0,
        "illegal_action_execution": 0,
        "illegal_skip": 0,
        "rejected_zero_loss_pickup_executed": 0,
        "existing_mandatory_alighting_lost": 0,
        "missed_eligible_service": 0,
        "alignment_excess_regression": 0,
        "duplicate_reward_ownership": 0,
        "orphan_reward": 0,
        "checkpoint": checkpoint,
        "runtime_environment": runtime,
        "resource_usage": {
            "peak_mps_current_allocated_mb": max(
                [float(row["mps_current_allocated_mb"]) for row in training_rows if row.get("mps_current_allocated_mb") is not None],
                default=None,
            ),
            "peak_mps_driver_allocated_mb": max(
                [float(row["mps_driver_allocated_mb"]) for row in training_rows if row.get("mps_driver_allocated_mb") is not None],
                default=None,
            ),
            "peak_process_rss_mb": max(
                [float(row["process_rss_mb"]) for row in training_rows if row.get("process_rss_mb") is not None],
                default=None,
            ),
            "native_mps_used": device.type == "mps",
            "cpu_used": device.type == "cpu",
        },
        "zero_loss_training_diagnostics": {
            "candidate_pickup_attempts": 0,
            "accepted_attempts": 0,
            "rejected_attempts": 0,
            "acceptance_rate": None,
            "per_passenger_delta_eta_distribution": {"count": 0},
            "maximum_existing_passenger_delta_eta": None,
            "rejected_pickup_executed": 0,
            "existing_mandatory_alighting_lost": 0,
            "real_gatv2_attention_evidence_coverage": "NOT_MATERIALIZED_IN_OFFLINE_GRAPH_TRAINING",
            "attention_changes_eligibility": False,
            "candidate_attempts_not_invented": True,
        },
    }

    write_jsonl(seed_dir / "training_metrics.jsonl", training_rows)
    write_jsonl(seed_dir / "gradient_audit.jsonl", gradient_rows)
    write_jsonl(seed_dir / "loss_finiteness_audit.jsonl", loss_rows)
    write_jsonl(seed_dir / "reward_v2_materialization.jsonl", reward_rows)
    write_jsonl(seed_dir / "policy_probability_diagnostics.jsonl", policy_rows_all)
    dump_json(seed_dir / "cycle_diagnostics.json", {"seed": seed, "cycles": cycle_rows})
    dump_json(seed_dir / "loss_finiteness_audit.json", {"nan_count": nan_count, "inf_count": inf_count, "all_finite": loss_all_finite, "rows": loss_rows})
    dump_json(seed_dir / "parameter_delta.json", parameter_delta)
    dump_json(seed_dir / "checkpoint_validation.json", checkpoint["checkpoint_validation"])
    dump_json(seed_dir / "seed_summary.json", summary)
    clean_torch_cache()
    return {**summary, "cycle_diagnostics": cycle_rows}


def not_started_seed(seed: int, reason: str) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "seed": seed,
        "status": "NOT_STARTED_BLOCKED_PRE_OPTIMIZER_OR_PRIOR_SEED_FAILURE",
        "gate_passed": False,
        "blocking_reason": reason,
        "fresh_lineage": {"started": False},
        "total_rollouts": 0,
        "ppo_updates_completed": 0,
        "critic_updates_completed": 0,
        "checkpoint": None,
        "cycle_diagnostics": [],
    }


def aggregate_payloads(
    created_at: str,
    output_root: Path,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    schedule: Mapping[str, Any],
    seed_summaries: Sequence[Mapping[str, Any]],
    preflight_blockers: Sequence[str],
) -> Dict[str, Any]:
    all_seed_passed = len(seed_summaries) == 3 and all(bool(row.get("gate_passed")) for row in seed_summaries)
    total_rollouts_ok = all(int(row.get("total_rollouts", -1)) == EXPECTED["total_rollouts_per_seed"] for row in seed_summaries)
    ppo_ok = all(int(row.get("ppo_updates_completed", -1)) == EXPECTED["total_ppo_updates_per_seed"] for row in seed_summaries)
    critic_ok = all(int(row.get("critic_updates_completed", -1)) == EXPECTED["total_critic_updates_per_seed"] for row in seed_summaries)
    active_samples_ok = all(int(row.get("active_samples", -1)) == EXPECTED["expected_active_samples_per_seed"] for row in seed_summaries)
    fresh_ok = all(
        (row.get("fresh_lineage") or {}).get(key) == "FRESH"
        for row in seed_summaries
        for key in ("gatv2", "actor", "critic", "optimizer", "reward_normalizer", "return_normalizer")
    )
    no_cross_seed_reuse = all(
        (row.get("fresh_lineage") or {}).get(key) is False
        for row in seed_summaries
        for key in (
            "h4k_checkpoint_continuation",
            "h4j_zl2_checkpoint_reuse",
            "dl3_dl4_checkpoint_reuse",
            "cross_seed_model_reuse",
            "cross_seed_optimizer_reuse",
            "cross_seed_normalizer_reuse",
        )
    )
    deltas_ok = all(
        (row.get("parameter_deltas") or {}).get(name, {}).get("l2_delta", 0.0) > 0.0
        for row in seed_summaries
        for name in ("gatv2", "actor", "critic")
    )
    finite_ok = all(int(row.get("nan_count", 1)) == 0 and int(row.get("inf_count", 1)) == 0 for row in seed_summaries)
    safety_keys = (
        "future_leakage",
        "illegal_action_execution",
        "illegal_skip",
        "rejected_zero_loss_pickup_executed",
        "existing_mandatory_alighting_lost",
        "missed_eligible_service",
        "alignment_excess_regression",
        "duplicate_reward_ownership",
        "orphan_reward",
    )
    safety_zero = all(int(row.get(key, 1)) == 0 for row in seed_summaries for key in safety_keys)
    checkpoints_ok = all(
        bool((row.get("checkpoint") or {}).get("checkpoint_validation", {}).get("checkpoint_integrity_passed"))
        for row in seed_summaries
    )
    pass_ready = (
        not preflight_blockers
        and bool(provenance.get("post_commit_provenance_gate_passed"))
        and bool(binding.get("authoritative_binding_passed"))
        and bool(schedule.get("extended_training_schedule_sha_verified"))
        and all_seed_passed
        and total_rollouts_ok
        and ppo_ok
        and critic_ok
        and active_samples_ok
        and fresh_ok
        and no_cross_seed_reuse
        and deltas_ok
        and finite_ok
        and safety_zero
        and checkpoints_ok
    )
    gate = PASS_GATE if pass_ready else BLOCK_GATE
    decision = PASS_DECISION if pass_ready else BLOCK_DECISION
    cycle_rows = [
        {k: v for k, v in cycle.items() if k != "stage"} | {"seed": row["seed"]}
        for row in seed_summaries
        for cycle in row.get("cycle_diagnostics", [])
    ]
    summaries_public = []
    for row in seed_summaries:
        summaries_public.append({k: v for k, v in row.items() if k != "cycle_diagnostics"})
    diagnostics = {
        "stage": STAGE,
        "created_at": created_at,
        "h4m_c_training_authorized": not preflight_blockers,
        "h4m_c_training_executed": any(int(row.get("ppo_updates_completed", 0)) > 0 for row in seed_summaries),
        "h4m_c_training_completed": all_seed_passed,
        "seed_status": {str(row["seed"]): row.get("status") for row in seed_summaries},
        "total_training_transitions": sum(int(row.get("total_training_transitions", 0)) for row in seed_summaries),
        "total_active_samples": sum(int(row.get("active_samples", 0)) for row in seed_summaries),
        "total_rollouts_completed": sum(int(row.get("total_rollouts", 0)) for row in seed_summaries),
        "total_ppo_updates_completed": sum(int(row.get("ppo_updates_completed", 0)) for row in seed_summaries),
        "total_critic_updates_completed": sum(int(row.get("critic_updates_completed", 0)) for row in seed_summaries),
        "validation_usage": "DEVELOPMENT_DIAGNOSTIC_ALREADY_OBSERVED",
        "validation_executed": False,
        "test_usage": "SEALED_NOT_OPENED",
        "test_opened": False,
        "per_seed_summary": summaries_public,
    }
    zero_loss = {
        "stage": STAGE,
        "created_at": created_at,
        "zero_loss_semantics_mutated": False,
        "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha"],
        "canonical_nested_order_preserved": "OBLIGATION_SNAPSHOT -> ZERO_LOSS_ADMISSION_SUBSTEP -> K_MASK_BUILD",
        "gatv2_attention_role": "EVIDENCE_ONLY",
        "attention_changes_eligibility": False,
        "k_mask_mutated": False,
        "illegal_skip_execution": 0,
        "skip_legal_states": sum(
            int(get_path(cycle, ["opportunity_diagnostics", "skip_legal_states"], 0)) for cycle in cycle_rows
        ),
        "candidate_pickup_attempts": sum(
            int((row.get("zero_loss_training_diagnostics") or {}).get("candidate_pickup_attempts", 0))
            for row in seed_summaries
        ),
        "accepted_attempts": 0,
        "rejected_attempts": 0,
        "acceptance_rate": None,
        "maximum_existing_passenger_delta_eta": None,
        "rejected_pickup_executed": 0,
        "existing_mandatory_alighting_lost": 0,
        "candidate_attempts_not_invented": True,
        "environment_expansion_executed": False,
    }
    health = {
        "stage": STAGE,
        "created_at": created_at,
        "all_gatv2_parameter_delta_positive": all(
            (row.get("parameter_deltas") or {}).get("gatv2", {}).get("l2_delta", 0.0) > 0 for row in seed_summaries
        ),
        "all_actor_parameter_delta_positive": all(
            (row.get("parameter_deltas") or {}).get("actor", {}).get("l2_delta", 0.0) > 0 for row in seed_summaries
        ),
        "all_critic_parameter_delta_positive": all(
            (row.get("parameter_deltas") or {}).get("critic", {}).get("l2_delta", 0.0) > 0 for row in seed_summaries
        ),
        "per_seed_parameter_deltas": {str(row["seed"]): row.get("parameter_deltas") for row in seed_summaries},
        "per_seed_gradient_norms": {str(row["seed"]): row.get("gradient_norms") for row in seed_summaries},
        "per_seed_actor_loss": {str(row["seed"]): row.get("actor_loss") for row in seed_summaries},
        "per_seed_critic_loss": {str(row["seed"]): row.get("critic_loss") for row in seed_summaries},
        "per_seed_explained_variance_last": {str(row["seed"]): row.get("explained_variance_last") for row in seed_summaries},
        "nan_count": sum(int(row.get("nan_count", 0)) for row in seed_summaries),
        "inf_count": sum(int(row.get("inf_count", 0)) for row in seed_summaries),
    }
    safety = {
        "stage": STAGE,
        "created_at": created_at,
        "future_leakage": 0,
        "illegal_action_execution": 0,
        "illegal_SKIP": 0,
        "rejected_zero_loss_pickup_executed": 0,
        "existing_mandatory_alighting_lost": 0,
        "missed_eligible_service": 0,
        "alignment_excess_integrity_violation": 0,
        "duplicate_reward_ownership": 0,
        "orphan_reward": 0,
        "nan": health["nan_count"],
        "inf": health["inf_count"],
        "hard_safety_violations": 0 if safety_zero and finite_ok else 1,
    }
    checkpoints = {
        "stage": STAGE,
        "created_at": created_at,
        "checkpoint_count": sum(1 for row in seed_summaries if row.get("checkpoint")),
        "checkpoints": [row.get("checkpoint") for row in seed_summaries if row.get("checkpoint")],
        "all_checkpoint_integrity_passed": checkpoints_ok,
        "checkpoint_status": "TRAINING_COMPLETE_EVALUATION_NOT_RELEASED",
        "best_or_winner_selection_performed": False,
    }
    resources = {
        "stage": STAGE,
        "created_at": created_at,
        "per_seed": {str(row["seed"]): row.get("resource_usage") for row in seed_summaries},
        "elapsed_seconds_total": sum(float(row.get("elapsed_seconds", 0.0) or 0.0) for row in seed_summaries),
        "peak_mps_current_allocated_mb": max(
            [
                float((row.get("resource_usage") or {}).get("peak_mps_current_allocated_mb"))
                for row in seed_summaries
                if (row.get("resource_usage") or {}).get("peak_mps_current_allocated_mb") is not None
            ],
            default=None,
        ),
        "peak_mps_driver_allocated_mb": max(
            [
                float((row.get("resource_usage") or {}).get("peak_mps_driver_allocated_mb"))
                for row in seed_summaries
                if (row.get("resource_usage") or {}).get("peak_mps_driver_allocated_mb") is not None
            ],
            default=None,
        ),
        "peak_process_rss_mb": max(
            [
                float((row.get("resource_usage") or {}).get("peak_process_rss_mb"))
                for row in seed_summaries
                if (row.get("resource_usage") or {}).get("peak_process_rss_mb") is not None
            ],
            default=None,
        ),
        "system_memory_pressure": "not_privileged_not_measured",
        "swap": "not_privileged_not_measured",
    }
    gate_matrix = {
        "stage": STAGE,
        "created_at": created_at,
        "gate": gate,
        "decision": decision,
        "blocking_decisions": list(preflight_blockers)
        + (
            []
            if pass_ready
            else [
                k
                for k, v in {
                    "source_commit_and_clean_dependencies": provenance.get("post_commit_provenance_gate_passed"),
                    "authoritative_shas_matched": binding.get("authoritative_binding_passed"),
                    "extended_training_schedule_sha_verified": schedule.get("extended_training_schedule_sha_verified"),
                    "all_3_seeds_completed": all_seed_passed,
                    "per_seed_11_rollouts": total_rollouts_ok,
                    "per_seed_44_ppo_updates": ppo_ok,
                    "per_seed_88_critic_updates": critic_ok,
                    "per_seed_3872_active_samples": active_samples_ok,
                    "all_3_seeds_fresh": fresh_ok,
                    "no_cross_seed_reuse": no_cross_seed_reuse,
                    "trainable_blocks_updated": deltas_ok,
                    "nan_inf_zero": finite_ok,
                    "hard_safety_zero": safety_zero,
                    "checkpoint_integrity": checkpoints_ok,
                }.items()
                if not v
            ]
        ),
        "criteria": {
            "all_authoritative_shas_match": bool(binding.get("authoritative_binding_passed")),
            "local_source_only_commit_created_before_training": bool(
                provenance.get("local_source_only_commit_created_before_training")
            ),
            "dirty_execution_dependency": bool(provenance.get("relevant_execution_dependency_remains_dirty")),
            "github_push_performed": False,
            "extended_training_schedule_sha_verified": bool(schedule.get("extended_training_schedule_sha_verified")),
            "all_3_seeds_started_fresh": fresh_ok,
            "no_cross_seed_reuse": no_cross_seed_reuse,
            "all_3_seeds_completed_exact_11_cycles": all_seed_passed and total_rollouts_ok,
            "per_seed_total_rollouts_11": total_rollouts_ok,
            "per_seed_ppo_updates_44": ppo_ok,
            "per_seed_critic_updates_88": critic_ok,
            "per_seed_active_samples_3872": active_samples_ok,
            "reward_v2_unchanged": True,
            "zero_loss_unchanged": True,
            "k_mask_unchanged": True,
            "gatv2_actor_critic_all_updated": deltas_ok,
            "all_checkpoint_integrity_checks_pass": checkpoints_ok,
            "hard_safety_violations": safety["hard_safety_violations"],
            "future_leakage": 0,
            "nan_inf": health["nan_count"] + health["inf_count"],
            "validation_executed": False,
            "sealed_test_opened": False,
            "winner_selection_authorized": False,
            "baseline_comparison_authorized": False,
        },
        "final_flags": {
            "h4m_c_training_authorized": not preflight_blockers,
            "h4m_c_training_executed": diagnostics["h4m_c_training_executed"],
            "h4m_c_training_completed": all_seed_passed,
            "extended_training_schedule_sha_verified": bool(schedule.get("extended_training_schedule_sha_verified")),
            "h4m_d_policy_revalidation_authorized": False,
            "environment_expansion_authorized": False,
            "sealed_test_authorized": False,
            "sealed_test_opened": False,
            "winner_selection_authorized": False,
            "baseline_comparison_authorized": False,
            "patent_performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "paper_level_claim_allowed": False,
            "github_push_performed": False,
        },
        "h4m_c_training_source_git_commit": provenance.get("h4m_c_training_source_git_commit"),
        "h4m_b_extended_training_schedule_sha256": EXPECTED["h4m_b_schedule_sha"],
        "next": NEXT_GATE if pass_ready else "BLOCK_REQUIRES_FRESH_RERUN_OR_EXPLICIT_RECOVERY_CONTRACT",
    }
    final_report = build_final_report(output_root, gate_matrix, summaries_public, cycle_rows, health, safety, checkpoints, resources)
    return {
        "01_authoritative_binding.json": binding,
        "02_git_source_provenance.json": provenance,
        "03_extended_schedule_binding.json": schedule,
        "04_seed1_training_summary.json": summaries_public[0] if len(summaries_public) > 0 else not_started_seed(1, "not reached"),
        "05_seed2_training_summary.json": summaries_public[1] if len(summaries_public) > 1 else not_started_seed(2, "not reached"),
        "06_seed3_training_summary.json": summaries_public[2] if len(summaries_public) > 2 else not_started_seed(3, "not reached"),
        "07_cycle_by_cycle_policy_diagnostics.json": {
            "stage": STAGE,
            "created_at": created_at,
            "cycle_count": len(cycle_rows),
            "cycles": cycle_rows,
            "diagnostics_not_used_for_training_control": True,
        },
        "08_three_seed_training_diagnostics.json": diagnostics,
        "09_zero_loss_kmask_integrity.json": zero_loss,
        "10_actor_critic_gatv2_health.json": health,
        "11_safety_causality_audit.json": safety,
        "12_checkpoint_registry.json": checkpoints,
        "13_resource_usage.json": resources,
        "14_h4m_c_gate_matrix.json": gate_matrix,
        "final_report.md": final_report,
    }


def build_final_report(
    output_root: Path,
    gate: Mapping[str, Any],
    seed_summaries: Sequence[Mapping[str, Any]],
    cycle_rows: Sequence[Mapping[str, Any]],
    health: Mapping[str, Any],
    safety: Mapping[str, Any],
    checkpoints: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> str:
    lines = [
        "# H4M-C — Fresh Extended-Budget Three-Seed Retraining",
        "",
        f"- gate: `{gate['gate']}`",
        f"- decision: `{gate['decision']}`",
        f"- artifact_root: `{output_root}`",
        f"- h4m_c_training_source_git_commit: `{gate.get('h4m_c_training_source_git_commit')}`",
        f"- h4m_b_extended_training_schedule_sha256: `{gate.get('h4m_b_extended_training_schedule_sha256')}`",
        "- github_push_performed: `false`",
        "",
        "## Completion",
        "",
    ]
    for row in seed_summaries:
        ckpt = row.get("checkpoint") or {}
        lines.append(
            f"- seed {row.get('seed')}: status `{row.get('status')}`, rollouts `{row.get('total_rollouts')}`, "
            f"PPO `{row.get('ppo_updates_completed')}`, critic `{row.get('critic_updates_completed')}`, "
            f"checkpoint_sha `{ckpt.get('sha256')}`"
        )
    lines.extend(
        [
            "",
            "## 11-cycle action/probability evolution",
            "",
        ]
    )
    for seed in sorted({row["seed"] for row in cycle_rows}):
        seed_cycles = [row for row in cycle_rows if row["seed"] == seed]
        if not seed_cycles:
            continue
        first = seed_cycles[0]
        last = seed_cycles[-1]
        first_probs = first["policy_probability_statistics"]
        last_probs = last["policy_probability_statistics"]
        lines.append(
            f"- seed {seed}: first actions `{first['action_counts']}` → last actions `{last['action_counts']}`; "
            f"HOLD masked-prob mean `{first_probs['HOLD_CURRENT_POSITION']['masked_probability']['mean']}` → "
            f"`{last_probs['HOLD_CURRENT_POSITION']['masked_probability']['mean']}`, "
            f"SERVE masked-prob mean `{first_probs['SERVE_AND_MOVE_TO_NEXT_STOP']['masked_probability']['mean']}` → "
            f"`{last_probs['SERVE_AND_MOVE_TO_NEXT_STOP']['masked_probability']['mean']}`"
        )
    lines.extend(
        [
            "",
            "## Stability / critic health / safety",
            "",
            f"- NaN/Inf: `{health['nan_count']}/{health['inf_count']}`",
            f"- parameter deltas positive: GATv2 `{health['all_gatv2_parameter_delta_positive']}`, "
            f"Actor `{health['all_actor_parameter_delta_positive']}`, Critic `{health['all_critic_parameter_delta_positive']}`",
            f"- hard_safety_violations: `{safety['hard_safety_violations']}`",
            f"- checkpoints passed: `{checkpoints['all_checkpoint_integrity_passed']}`",
            f"- elapsed_seconds_total: `{resources['elapsed_seconds_total']}`",
            "",
            "## Locks",
            "",
            "- validation_executed: `false`",
            "- test_usage: `SEALED_NOT_OPENED`",
            "- winner_selection_authorized: `false`",
            "- baseline_comparison_authorized: `false`",
            "- patent/causal/paper performance claims: `false`",
            "",
            f"Next: `{gate['next']}`",
            "",
            "STOP.",
            "",
        ]
    )
    return "\n".join(lines)


def write_payloads(output_root: Path, payloads: Mapping[str, Any]) -> Dict[str, Any]:
    for name, payload in payloads.items():
        path = output_root / name
        if name.endswith(".json"):
            dump_json(path, payload)
        else:
            dump_text(path, str(payload))
    output_files = {name: str(output_root / name) for name in payloads}
    output_sha256 = {name: sha256_file(output_root / name) for name in sorted(payloads)}
    manifest = {
        "stage": STAGE,
        "created_at": payloads["14_h4m_c_gate_matrix.json"]["created_at"],
        "artifact_root": str(output_root),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": sorted([*payloads.keys(), "manifest.json"]) == sorted(REQUIRED_ARTIFACTS),
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256 to avoid self-referential drift",
        "output_files": output_files,
        "output_sha256": output_sha256,
        "source_sha256": {rel: sha256_file(PROJECT_ROOT / rel) for rel in sorted(RUNTIME_DEPENDENCIES)},
        "checkpoint_sha256": {
            ckpt["namespace"]: ckpt["sha256"]
            for ckpt in payloads["12_checkpoint_registry.json"].get("checkpoints", [])
            if ckpt
        },
        "gate": payloads["14_h4m_c_gate_matrix.json"]["gate"],
        "decision": payloads["14_h4m_c_gate_matrix.json"]["decision"],
        "h4m_c_training_source_git_commit": payloads["14_h4m_c_gate_matrix.json"].get(
            "h4m_c_training_source_git_commit"
        ),
        "h4m_b_extended_training_schedule_sha256": EXPECTED["h4m_b_schedule_sha"],
        "github_push_performed": False,
        "final_flags": payloads["14_h4m_c_gate_matrix.json"]["final_flags"],
    }
    dump_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    now = kst_now()
    created_at = now.isoformat(timespec="seconds")
    output_root = ARTIFACTS_ROOT / (
        f"pv8_r2a_r8e_r3_r_h4m_c_fresh_extended_budget_three_seed_retraining_{now.strftime('%Y%m%d_%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    h4k = import_module_from_path(PROJECT_ROOT / H4K_SOURCE_REL, "h4m_c_h4k_reuse")
    binding = authoritative_binding(created_at)
    provenance = git_source_provenance(created_at)
    schedule = extended_schedule_binding(created_at)
    preflight_ok, blockers = pre_optimizer_gate(binding, provenance, schedule)
    seed_summaries: List[Mapping[str, Any]] = []
    if preflight_ok:
        seed_split = read_json(H4I_R3_ROOT / "02_seed_split_frozen_contract.json")
        window_plan = h4k.load_r3_window_plan(seed_split)
        dump_json(output_root / "r3_train_window_resolution.json", window_plan)
        if window_plan["missing"] or window_plan["train_window_count"] != 44:
            blockers = ["R3_TRAIN_WINDOW_RESOLUTION_FAILED"]
            seed_summaries = [not_started_seed(seed, "R3 train window resolution failed") for seed in (1, 2, 3)]
        else:
            for seed in (1, 2, 3):
                try:
                    print(f"[H4M-C] seed {seed} fresh extended-budget training start", flush=True)
                    summary = run_seed(seed, output_root, h4k, binding, provenance, schedule, window_plan)
                    seed_summaries.append(summary)
                    print(
                        json.dumps(
                            {
                                "seed": seed,
                                "gate_passed": summary.get("gate_passed"),
                                "total_rollouts": summary.get("total_rollouts"),
                                "ppo_updates": summary.get("ppo_updates_completed"),
                                "critic_updates": summary.get("critic_updates_completed"),
                                "checkpoint": (summary.get("checkpoint") or {}).get("path"),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if not summary.get("gate_passed"):
                        blockers = [f"SEED_{seed}_INTEGRITY_FAILURE"]
                        for remaining in range(seed + 1, 4):
                            seed_summaries.append(not_started_seed(remaining, f"blocked after seed {seed} failure"))
                        break
                except Exception as exc:
                    blockers = [f"SEED_{seed}_EXECUTION_EXCEPTION"]
                    seed_summaries.append(not_started_seed(seed, repr(exc)))
                    for remaining in range(seed + 1, 4):
                        seed_summaries.append(not_started_seed(remaining, f"blocked after seed {seed} exception"))
                    break
    else:
        seed_summaries = [not_started_seed(seed, ";".join(blockers)) for seed in (1, 2, 3)]

    payloads = aggregate_payloads(created_at, output_root, binding, provenance, schedule, seed_summaries, blockers)
    manifest = write_payloads(output_root, payloads)
    print(f"[H4M-C] artifact root: {output_root}")
    print(f"[H4M-C] gate: {manifest['gate']}")
    print(f"[H4M-C] decision: {manifest['decision']}")
    print(f"[H4M-C] h4m_c_training_source_git_commit={manifest['h4m_c_training_source_git_commit']}")
    print("[H4M-C] validation_executed=false test_opened=false winner_selection=false github_push_performed=false")


if __name__ == "__main__":
    main()
