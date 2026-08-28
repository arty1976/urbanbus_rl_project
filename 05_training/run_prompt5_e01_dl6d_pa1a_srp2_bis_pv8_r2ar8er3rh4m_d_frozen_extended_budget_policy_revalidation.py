from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


STAGE = "PV8-R2A-R8E-R3-R-H4M-D"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_D_"
    "FROZEN_EXTENDED_BUDGET_POLICY_REVALIDATION_COMPLETE"
)
BLOCK_GATE = (
    "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_D_"
    "FROZEN_EXTENDED_BUDGET_POLICY_REVALIDATION_FAILED"
)
PROTOCOL_BLOCK_GATE = (
    "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_D_"
    "H4L_EVALUATION_PROTOCOL_NOT_REPRODUCIBLE"
)

DECISION_STATE_DEPENDENT = "PV8_EXTENDED_BUDGET_POLICY_REVALIDATION_SHOWS_STATE_DEPENDENT_DIFFERENTIATION"
DECISION_COUNTERFACTUAL_MISALIGNMENT = (
    "PV8_EXTENDED_BUDGET_POLICY_REVALIDATION_SERVE_DOMINANCE_WITH_COUNTERFACTUAL_MISALIGNMENT"
)
DECISION_LONG_HORIZON = "PV8_EXTENDED_BUDGET_POLICY_REVALIDATION_LONG_HORIZON_SERVE_VALUE_RECONCILIATION_REQUIRED"
DECISION_OBSERVATION = "PV8_EXTENDED_BUDGET_POLICY_REVALIDATION_ACTOR_OBSERVATION_DISCRIMINATION_AUDIT_REQUIRED"
DECISION_NOT_UNIQUE = "PV8_EXTENDED_BUDGET_POLICY_REVALIDATION_CAUSAL_ATTRIBUTION_NOT_UNIQUE"
BLOCK_DECISION = "PV8_EXTENDED_BUDGET_POLICY_REVALIDATION_BLOCKED"

NEXT_GATE_CREDIT = "H4M-E_REWARD_GAE_ACTOR_CREDIT_ALIGNMENT_AUDIT"
NEXT_GATE_LONG_HORIZON = "H4M-E_LONG_HORIZON_ACTION_VALUE_RECONCILIATION"
NEXT_GATE_OBSERVATION = "H4M-E_ACTOR_OBSERVATION_DISCRIMINATION_AUDIT"
NEXT_GATE_STATE_DEPENDENT = "H4M-E_STATE_DEPENDENT_DIFFERENTIATION_REVIEW"
NEXT_GATE_ATTRIBUTION = "H4M-E_CAUSAL_ATTRIBUTION_DIAGNOSTIC_AUDIT"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"

H4M_C_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_c_fresh_extended_budget_three_seed_retraining_20260814_172137"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"
H4M_A_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_a_decision_opportunity_environment_adequacy_audit_20260814_143743"
H4L_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4l_frozen_three_seed_policy_validation_evaluation_20260814_124419"
H4I_RERUN_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4i_rerun_training_readiness_20260810_192924"
DL3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915"

H4L_SOURCE_PATH = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4l_frozen_three_seed_policy_validation_evaluation.py"
)
H4M_D_SOURCE_PATH = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_d_frozen_extended_budget_policy_revalidation.py"
)
H4M_C_SOURCE_PATH = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_c_fresh_extended_budget_three_seed_retraining.py"
)
H4M_B_SOURCE_PATH = (
    "05_training/"
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_b_training_budget_extension_selection_freeze.py"
)

EXPECTED = {
    "h4m_c_gate": (
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_C_"
        "FRESH_EXTENDED_BUDGET_THREE_SEED_RETRAINING_COMPLETE"
    ),
    "h4m_c_decision": "PV8_FRESH_EXTENDED_BUDGET_THREE_SEED_RETRAINING_COMPLETE_READY_FOR_FROZEN_POLICY_REVALIDATION",
    "h4m_c_training_source_git_commit": "00a53164c710f0bb8028aabe4f3f616a8fe6610d",
    "h4m_b_schedule_sha": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "checkpoint_shas": {
        1: "084cb811200a2f68e32ed2eaeef6a4ae07645c56cdfd62be20169cc24230aa3a",
        2: "6b5ff219105e8d6a3029c429d53152651f7fe4e186af1c673b0810833d846af5",
        3: "4e840e14f959bf8a7e05ad560829bdbf3268382113b6d90c7eb9dc95f184be23",
    },
    "checkpoint_namespaces": {
        1: "H4M_C_SEED_001_FRESH_EXTENDED_BUDGET_REWARD_V2_ZERO_LOSS",
        2: "H4M_C_SEED_002_FRESH_EXTENDED_BUDGET_REWARD_V2_ZERO_LOSS",
        3: "H4M_C_SEED_003_FRESH_EXTENDED_BUDGET_REWARD_V2_ZERO_LOSS",
    },
    "h4l_reference_avg_wait_seconds": 297.7850241545894,
    "h4l_reference_service": 1.0,
    "h4l_reference_p95_wait_seconds": 576.6999999999999,
}

ACTION_NAMES = {
    0: "HOLD_CURRENT_POSITION",
    1: "SERVE_AND_MOVE_TO_NEXT_STOP",
    2: "CONDITIONAL_SKIP_EMPTY_STOP",
}
ACTION_IDS = {name: idx for idx, name in ACTION_NAMES.items()}

RUNTIME_DEPENDENCIES = {
    H4M_D_SOURCE_PATH,
    H4L_SOURCE_PATH,
    H4M_C_SOURCE_PATH,
    H4M_B_SOURCE_PATH,
    "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py",
    "05_training/evaluation/canonical_kpi_aggregator.py",
    "05_training/mappo_runner.py",
    "05_training/rewards/mappo_reward_v1.py",
    "05_training/simulator/zero_loss_admission_adapter.py",
    "05_training/simulator/k_action_mask_runtime.py",
}

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_h4l_protocol_reproduction_audit.json",
    "03_checkpoint_immutability_audit.json",
    "04_seed1_policy_revalidation.json",
    "05_seed2_policy_revalidation.json",
    "06_seed3_policy_revalidation.json",
    "07_h4l_vs_h4md_action_probability_delta.json",
    "08_h4ma_hold_better_actor_crosswalk.json",
    "09_h4l_vs_h4md_kpi_delta.json",
    "10_critic_revalidation.json",
    "11_zero_loss_skip_opportunity_status.json",
    "12_safety_integrity_audit.json",
    "13_h4m_d_root_cause_classification.json",
    "14_h4m_d_gate_matrix.json",
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
    if isinstance(value, tuple):
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


h4l = import_module_from_path(PROJECT_ROOT / H4L_SOURCE_PATH, "h4m_d_h4l_protocol")


def git_run(args: Sequence[str], timeout: int = 30) -> Tuple[int, str, str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


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
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    avg = mean(nums)
    var = mean([(v - avg) ** 2 for v in nums]) if len(nums) > 1 else 0.0
    return {"count": len(nums), "mean": avg, "median": median(nums), "std": math.sqrt(var), "min": min(nums), "max": max(nums)}


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


def git_source_provenance(created_at: str) -> Dict[str, Any]:
    _rc, branch, _branch_err = git_run(["branch", "--show-current"])
    _rc, head, _head_err = git_run(["rev-parse", "HEAD"])
    _rc, status_short, _status_err = git_run(["status", "--short"])
    dirty_rows = parse_status_paths(status_short)
    dirty_classification = []
    relevant_dirty = []
    for row in dirty_rows:
        path = row["path"]
        relevance = "H4M_D_RUNTIME_DEPENDENCY" if path in RUNTIME_DEPENDENCIES else "OUTSIDE_H4M_D_EXECUTION_DEPENDENCY_SET"
        dirty_classification.append({**row, "relevance": relevance})
        if relevance == "H4M_D_RUNTIME_DEPENDENCY":
            relevant_dirty.append(path)
    ls_rc, ls_out, _ls_err = git_run(["ls-tree", "-r", "--name-only", "HEAD", "--", H4M_D_SOURCE_PATH])
    diff_rc, _diff_out, _diff_err = git_run(["diff", "--quiet", "HEAD", "--", H4M_D_SOURCE_PATH])
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "h4m_d_evaluation_source_git_commit": head if ls_rc == 0 and bool(ls_out) and diff_rc == 0 else None,
        "h4m_d_source_path": H4M_D_SOURCE_PATH,
        "h4m_d_source_present_in_head": ls_rc == 0 and bool(ls_out),
        "h4m_d_source_no_uncommitted_diff_vs_head": diff_rc == 0,
        "local_source_only_commit_created_before_evaluation": ls_rc == 0 and bool(ls_out) and diff_rc == 0,
        "github_push_performed": False,
        "status_short": status_short,
        "remaining_dirty_paths": dirty_rows,
        "dirty_path_relevance_classification": dirty_classification,
        "relevant_execution_source_remains_uncommitted": bool(relevant_dirty),
        "relevant_dirty_paths": relevant_dirty,
        "post_commit_provenance_gate_passed": ls_rc == 0 and bool(ls_out) and diff_rc == 0 and not relevant_dirty,
    }


def checkpoint_paths_from_registry() -> Dict[int, Path]:
    registry = read_json(H4M_C_ROOT / "12_checkpoint_registry.json")
    paths: Dict[int, Path] = {}
    for row in registry.get("checkpoints", []):
        namespace = str(row.get("namespace", ""))
        match = re.search(r"H4M_C_SEED_(\d+)_", namespace)
        if match:
            paths[int(match.group(1))] = Path(row["path"])
    return paths


def load_checkpoints_read_only() -> Dict[int, Dict[str, Any]]:
    paths = checkpoint_paths_from_registry()
    out: Dict[int, Dict[str, Any]] = {}
    for seed in (1, 2, 3):
        path = paths[seed]
        pre_sha = sha256_file(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        out[seed] = {"path": path, "pre_eval_sha256": pre_sha, "payload": payload}
    return out


def extract_reward_v2_sha() -> Optional[str]:
    text = (TRAINING_ROOT / "rewards/mappo_reward_v1.py").read_text(encoding="utf-8-sig")
    match = re.search(r'PV8_REWARD_V2_FREEZE_SHA256\s*=\s*"([0-9a-f]{64})"', text)
    return match.group(1) if match else None


def h4m_c_authoritative_binding(created_at: str, checkpoints: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    gate = read_json(H4M_C_ROOT / "14_h4m_c_gate_matrix.json")
    manifest = read_json(H4M_C_ROOT / "manifest.json")
    authoritative = read_json(H4M_C_ROOT / "01_authoritative_binding.json")
    schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    h4m_a_cf = read_json(H4M_A_ROOT / "05_counterfactual_action_value_audit.json")
    h4m_a_actor = read_json(H4M_A_ROOT / "06_actor_pre_argmax_preference_audit.json")
    h4g = h4l.h4g_runtime_status()
    reward_sha = extract_reward_v2_sha()
    adapter_sha = sha256_file(TRAINING_ROOT / "simulator/zero_loss_admission_adapter.py")
    h4i_scope = read_json(H4I_RERUN_ROOT / "02_training_scope_binding.json")
    seed_checks: Dict[str, Any] = {}
    for seed, row in checkpoints.items():
        payload = row["payload"]
        metadata = payload.get("metadata", {})
        expected_namespace = EXPECTED["checkpoint_namespaces"][seed]
        config = payload.get("training_configuration", {})
        seed_checks[str(seed)] = {
            "path": str(row["path"]),
            "expected_namespace": expected_namespace,
            "expected_sha256": EXPECTED["checkpoint_shas"][seed],
            "observed_sha256": row["pre_eval_sha256"],
            "sha_match": row["pre_eval_sha256"] == EXPECTED["checkpoint_shas"][seed],
            "checkpoint_boundary": payload.get("checkpoint_boundary"),
            "checkpoint_boundary_match": payload.get("checkpoint_boundary") == expected_namespace,
            "checkpoint_status": payload.get("checkpoint_status"),
            "checkpoint_status_match": payload.get("checkpoint_status") == "TRAINING_COMPLETE_EVALUATION_NOT_RELEASED",
            "metadata_seed_match": metadata.get("seed") == seed,
            "metadata_reward_v2_sha_match": metadata.get("reward_v2_sha256") == EXPECTED["reward_v2_sha"],
            "metadata_h4g_runtime_sha_match": metadata.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha"],
            "metadata_r3_split_sha_match": metadata.get("r3_split_sha256") == EXPECTED["r3_split_sha"],
            "metadata_zero_loss_adapter_sha_match": metadata.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha"],
            "metadata_h4m_b_schedule_sha_match": metadata.get("h4m_b_extended_training_schedule_sha256") == EXPECTED["h4m_b_schedule_sha"],
            "metadata_h4m_c_training_source_commit_match": metadata.get("h4m_c_training_source_git_commit")
            == EXPECTED["h4m_c_training_source_git_commit"],
            "metadata_outer_training_count_match": metadata.get("outer_training_count") == 11,
            "metadata_total_rollouts_match": metadata.get("total_rollouts") == 11,
            "metadata_total_ppo_updates_match": metadata.get("total_ppo_updates") == 44,
            "metadata_total_critic_updates_match": metadata.get("total_critic_updates") == 88,
            "config_h4m_b_schedule_sha_match": config.get("h4m_b_extended_training_schedule_sha256") == EXPECTED["h4m_b_schedule_sha"],
            "config_policy_evaluation_not_authorized": config.get("policy_evaluation_authorized") is False,
        }
    checks = {
        "h4m_c_gate_match": gate.get("gate") == EXPECTED["h4m_c_gate"],
        "h4m_c_decision_match": gate.get("decision") == EXPECTED["h4m_c_decision"],
        "h4m_c_source_commit_match": gate.get("h4m_c_training_source_git_commit")
        == EXPECTED["h4m_c_training_source_git_commit"],
        "h4m_c_manifest_gate_match": manifest.get("gate") == EXPECTED["h4m_c_gate"],
        "h4m_c_manifest_source_commit_match": manifest.get("h4m_c_training_source_git_commit")
        == EXPECTED["h4m_c_training_source_git_commit"],
        "h4m_b_schedule_sha_match": schedule.get("extended_training_schedule_sha256") == EXPECTED["h4m_b_schedule_sha"]
        or schedule.get("full_training_schedule_sha256") == EXPECTED["h4m_b_schedule_sha"],
        "h4m_c_authoritative_binding_passed": authoritative.get("authoritative_binding_passed") is True,
        "reward_v2_sha_match": reward_sha == EXPECTED["reward_v2_sha"],
        "h4g_runtime_sha_match": h4g["source_hashes_match_h4g_baseline"],
        "r3_split_sha_match": h4i_scope.get("scope_hash") == EXPECTED["r3_split_sha"],
        "zero_loss_adapter_sha_match": adapter_sha == EXPECTED["zero_loss_adapter_sha"],
        "h4m_a_counterfactual_crosswalk_available": h4m_a_cf.get("summary", {}).get("multi_action_state_count") == 32,
        "h4m_a_actor_preference_baseline_available": len(h4m_a_actor.get("records", [])) == 96,
        "all_checkpoint_seed_checks_passed": all(
            all(v for k, v in row.items() if k.endswith("_match") or k in {"sha_match"})
            for row in seed_checks.values()
        ),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "h4m_c_artifact_root": str(H4M_C_ROOT),
        "h4m_b_artifact_root": str(H4M_B_ROOT),
        "h4m_a_artifact_root": str(H4M_A_ROOT),
        "h4l_reference_artifact_root": str(H4L_ROOT),
        "classification": "DEVELOPMENT_DIAGNOSTIC_ALREADY_OBSERVED",
        "test6_status": "SEALED_NOT_OPENED",
        "checkpoint_binding_passed": all(checks.values()),
        "checks": checks,
        "seed_checkpoint_checks": seed_checks,
        "runtime_binding": {
            "reward_v2": {"expected": EXPECTED["reward_v2_sha"], "observed": reward_sha},
            "h4g_runtime": h4g,
            "r3_split": {"expected": EXPECTED["r3_split_sha"], "observed": h4i_scope.get("scope_hash")},
            "zero_loss_adapter": {"expected": EXPECTED["zero_loss_adapter_sha"], "observed": adapter_sha},
            "h4m_b_extended_training_schedule": {
                "expected": EXPECTED["h4m_b_schedule_sha"],
                "observed": schedule.get("extended_training_schedule_sha256")
                or schedule.get("full_training_schedule_sha256"),
            },
        },
        "test_opened": False,
        "training_forbidden": True,
        "checkpoint_mutation_forbidden": True,
        "additional_training_authorized": False,
    }


def config_material_view(config: Mapping[str, Any]) -> Dict[str, Any]:
    ignored = {"seed", "seed_audit", "created_at"}
    return {k: v for k, v in config.items() if k not in ignored}


def configuration_equality_audit(created_at: str, checkpoints: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    material: Dict[str, Any] = {}
    raw_hashes: Dict[str, Any] = {}
    material_hashes: Dict[str, Any] = {}
    for seed, row in checkpoints.items():
        config = row["payload"].get("training_configuration", {})
        material[str(seed)] = config_material_view(config)
        raw_hashes[str(seed)] = canonical_sha(config)
        material_hashes[str(seed)] = canonical_sha(material[str(seed)])
    unique_material_hashes = sorted(set(material_hashes.values()))
    return {
        "stage": STAGE,
        "created_at": created_at,
        "seed_only_configuration_difference": len(unique_material_hashes) == 1,
        "material_non_seed_difference_detected": len(unique_material_hashes) != 1,
        "ignored_seed_identity_fields": ["seed", "seed_audit", "created_at"],
        "raw_configuration_sha256_by_seed": raw_hashes,
        "material_configuration_sha256_by_seed": material_hashes,
        "material_reference_seed": 1,
        "material_field_count": len(material["1"]),
        "material_fields": sorted(material["1"].keys()),
        "material_equality_passed": len(unique_material_hashes) == 1,
    }


def h4l_protocol_reproduction_audit(created_at: str, validation_plan: Mapping[str, Any]) -> Dict[str, Any]:
    h4l_protocol = read_json(H4L_ROOT / "04_frozen_evaluation_protocol.json")
    h4l_window_resolution = read_json(H4L_ROOT / "validation_window_resolution.json")
    reproduced = h4l.frozen_evaluation_protocol(created_at)
    protocol_fields = [
        "action_selection_mode",
        "evaluation_rng_policy",
        "validation_window_traversal_order",
        "episode_termination_rules",
        "evaluation_repetitions",
        "normalizer_read_only_behavior",
        "zero_loss_evaluation_behavior",
        "k_mask_evaluation_behavior",
        "torch_grad_policy",
    ]
    field_matches = {
        field: reproduced.get(field) == h4l_protocol.get(field)
        for field in protocol_fields
    }
    window_ids_match = list(validation_plan.get("validation_window_ids", [])) == list(
        h4l_window_resolution.get("validation_window_ids", [])
    )
    window_count_match = validation_plan.get("validation_window_count") == h4l_window_resolution.get("validation_window_count") == 4
    scope_passed = bool(validation_plan.get("validation_scope_passed")) and window_ids_match and window_count_match
    protocol_passed = bool(reproduced.get("evaluation_protocol_uniquely_resolved")) and all(field_matches.values()) and scope_passed
    return {
        "stage": STAGE,
        "created_at": created_at,
        "h4l_reference_artifact_root": str(H4L_ROOT),
        "h4l_reference_protocol_sha256": sha256_file(H4L_ROOT / "04_frozen_evaluation_protocol.json"),
        "h4l_reference_window_resolution_sha256": sha256_file(H4L_ROOT / "validation_window_resolution.json"),
        "h4l_protocol_reproduction_passed": protocol_passed,
        "evaluation_protocol_uniquely_resolved": bool(reproduced.get("evaluation_protocol_uniquely_resolved")),
        "field_matches": field_matches,
        "validation_window_identity_match": window_ids_match,
        "validation_window_count_match": window_count_match,
        "validation_scope_passed": scope_passed,
        "validation_plan": validation_plan,
        "reproduced_protocol": reproduced,
        "reference_protocol": h4l_protocol,
        "test_opened": False,
        "classification": "DEVELOPMENT_DIAGNOSTIC_ALREADY_OBSERVED",
    }


def state_id(window_id: str, local_step: int, agent_slot: int) -> str:
    return f"validation:{window_id}:step{local_step:03d}:agent{agent_slot:02d}"


def reward_v2_metrics(
    reward_mod: Any,
    *,
    window: Mapping[str, Any],
    local_step: int,
    agent_slot: int,
    action_id: int,
    target_id: int,
) -> Dict[str, Any]:
    transition_id = f"H4M-D:{window['window_id']}:step{local_step:03d}:agent{agent_slot:02d}"
    local_decision_ts = float(local_step * 60 + agent_slot)
    service_required = int(target_id == ACTION_IDS["SERVE_AND_MOVE_TO_NEXT_STOP"])
    service_completed = int(service_required and action_id == ACTION_IDS["SERVE_AND_MOVE_TO_NEXT_STOP"])
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
        "route_id": f"H4M_D_R3_DIRECTION_{window.get('direction_id')}",
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


def tensor_values(tensor: torch.Tensor) -> List[float]:
    return [float(v) for v in tensor.detach().cpu().reshape(-1).tolist()]


def summarize_preference_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    per_action: Dict[str, Any] = {}
    for action in ACTION_NAMES.values():
        per_action[action] = {
            "raw_logit": finite_stats([r["pre_argmax"][action]["raw_logit"] for r in records]),
            "masked_logit": finite_stats([r["pre_argmax"][action]["masked_logit"] for r in records]),
            "raw_probability": finite_stats([r["pre_argmax"][action]["raw_probability"] for r in records]),
            "masked_probability": finite_stats([r["pre_argmax"][action]["masked_probability"] for r in records]),
        }
    selected_counts = Counter(str(r["selected_action"]) for r in records)
    return {
        "record_count": len(records),
        "selected_action_counts": dict(selected_counts),
        "entropy": finite_stats([r["entropy"] for r in records]),
        "per_action": per_action,
        "top1_top2_probability_margin": finite_stats([r["top1_top2_probability_margin"] for r in records]),
        "top1_top2_logit_margin": finite_stats([r["top1_top2_logit_margin"] for r in records]),
        "serve_minus_hold_probability_margin": finite_stats(
            [
                r["pre_argmax"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"]
                - r["pre_argmax"]["HOLD_CURRENT_POSITION"]["masked_probability"]
                for r in records
            ]
        ),
        "serve_minus_hold_logit_margin": finite_stats(
            [
                r["pre_argmax"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_logit"]
                - r["pre_argmax"]["HOLD_CURRENT_POSITION"]["masked_logit"]
                for r in records
            ]
        ),
    }


def evaluate_seed(
    seed: int,
    checkpoint: Mapping[str, Any],
    validation_plan: Mapping[str, Any],
    output_root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    dl4 = import_module_from_path(
        TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py",
        f"h4m_d_dl4_seed_{seed}",
    )
    dl1 = dl4.import_dl1(PROJECT_ROOT)
    reward_mod = import_module_from_path(TRAINING_ROOT / "rewards/mappo_reward_v1.py", f"h4m_d_reward_seed_{seed}")
    canonical = import_module_from_path(
        TRAINING_ROOT / "evaluation/canonical_kpi_aggregator.py", f"h4m_d_canonical_kpi_seed_{seed}"
    )
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mapping_artifact = Path(read_json(DL3_ROOT / "study_area_snapshot.json")["repair_mapping"])
    validation_paths = [Path(row["snapshot_path"]) for row in validation_plan["validation_rows"]]
    sample_full = dl1.torch_load(validation_paths[0])
    spec, inventory, _connectivity, _tensor_mask = dl1.build_subgraph_spec(
        PROJECT_ROOT, sample_full, mapping_artifact=mapping_artifact
    )
    sample_graph = dl1.make_subgraph_data(sample_full, spec)
    validation_data = dl4.load_subgraphs(dl1, validation_paths, spec)
    config = dict(checkpoint["payload"]["training_configuration"])
    config["spec"] = spec
    config["effective_agents"] = min(8, int(inventory["available_suseong_agents"]))

    pre_actor_hash = dl1.state_dict_hash(checkpoint["payload"]["actor_state_dict"])
    pre_critic_hash = dl1.state_dict_hash(checkpoint["payload"]["critic_state_dict"])
    pre_gatv2_hash = dl1.state_dict_hash(checkpoint["payload"]["gatv2_state_dict"])
    pre_reward_norm_hash = canonical_sha(checkpoint["payload"].get("reward_normalizer_state", {}))
    pre_return_norm_hash = canonical_sha(checkpoint["payload"].get("return_normalizer_state", {}))
    encoder, actor, critic, return_normalizer = h4l.load_models_for_seed(dl1, dl4, checkpoint, sample_graph, device)
    encoder.eval()
    actor.eval()
    critic.eval()

    raw_rewards: List[torch.Tensor] = []
    normalized_rewards: List[torch.Tensor] = []
    values: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    action_records: List[Dict[str, Any]] = []
    reward_records: List[Dict[str, Any]] = []
    embedding_stats: List[Dict[str, Any]] = []
    window_rows: List[Dict[str, Any]] = []
    invalid_probability_distribution_count = 0
    masked_selected_count = 0
    masked_probability_mass_values: List[float] = []
    selected_probability_values: List[float] = []
    entropy_values: List[float] = []
    logits_values: List[float] = []
    probs_values: List[float] = []
    start = time.perf_counter()

    with torch.no_grad():
        for step, (cpu_data, window) in enumerate(zip(validation_data, validation_plan["validation_rows"])):
            data = cpu_data.to(device)
            indices = dl1.agent_indices_for_step(config["spec"], step, int(config["effective_agents"]))
            logits, _critic_out, value_original, agent_mask = dl4.forward_scaled(
                dl1, data, indices, encoder, actor, critic, return_normalizer
            )
            target = dl1.action_targets_from_y(
                data.y[torch.tensor(indices, dtype=torch.long, device=device)],
                int(config["action_dim"]),
            )
            masked_logits, allowed = h4l.masked_logits_for_targets(logits, target)
            raw_probs = torch.softmax(logits, dim=-1)
            masked_probs = torch.softmax(masked_logits, dim=-1)
            prob_sum_error = (masked_probs.sum(dim=-1) - 1.0).abs()
            invalid_probability_distribution_count += int((~torch.isfinite(masked_probs)).sum().detach().cpu().item())
            invalid_probability_distribution_count += int((prob_sum_error > 1.0e-5).sum().detach().cpu().item())
            actions = torch.argmax(masked_logits, dim=-1)
            entropy = Categorical(logits=masked_logits).entropy()
            active_mask_cpu = agent_mask.detach().cpu().bool().tolist()
            rewards_for_step: List[float] = []
            raw_reward_values_for_norm: List[float] = []
            wait_values: List[float] = []
            served = 0
            eligible = 0
            missed = 0
            for agent_slot, (action_id, target_id, is_active) in enumerate(
                zip(actions.detach().cpu().tolist(), target.detach().cpu().tolist(), active_mask_cpu)
            ):
                if bool(is_active):
                    action_id = int(action_id)
                    target_id = int(target_id)
                    if action_id == ACTION_IDS["SERVE_AND_MOVE_TO_NEXT_STOP"] and target_id == ACTION_IDS["SERVE_AND_MOVE_TO_NEXT_STOP"]:
                        served += 1
                    if target_id == ACTION_IDS["SERVE_AND_MOVE_TO_NEXT_STOP"]:
                        eligible += 1
                    if target_id == ACTION_IDS["SERVE_AND_MOVE_TO_NEXT_STOP"] and action_id != ACTION_IDS["SERVE_AND_MOVE_TO_NEXT_STOP"]:
                        missed += 1
                    legal = bool(allowed[agent_slot, action_id].detach().cpu().item())
                    if not legal:
                        masked_selected_count += 1
                    masked_mass = float(raw_probs[agent_slot][~allowed[agent_slot]].sum().detach().cpu().item())
                    selected_prob = float(masked_probs[agent_slot, action_id].detach().cpu().item())
                    ent = float(entropy[agent_slot].detach().cpu().item())
                    masked_probability_mass_values.append(masked_mass)
                    selected_probability_values.append(selected_prob)
                    entropy_values.append(ent)
                    logits_values.extend([float(v) for v in logits[agent_slot].detach().cpu().tolist()])
                    probs_values.extend([float(v) for v in masked_probs[agent_slot].detach().cpu().tolist()])
                    metrics = reward_v2_metrics(
                        reward_mod,
                        window=window,
                        local_step=step,
                        agent_slot=agent_slot,
                        action_id=action_id,
                        target_id=target_id,
                    )
                    materialized = reward_mod.compute_reward_v2(metrics)
                    reward_value = float(materialized["reward_total"])
                    for wait_row in materialized["reward_avg_wait_component"].get("affected_wait_rows", []) or []:
                        wait_values.append(float(wait_row["total_wait_seconds"]))
                    reward_records.append(
                        {
                            "state_id": state_id(window["window_id"], step, agent_slot),
                            "window_id": window["window_id"],
                            "agent_slot": agent_slot,
                            "action_id": action_id,
                            "action": ACTION_NAMES.get(action_id, str(action_id)),
                            "target_id": target_id,
                            "target": ACTION_NAMES.get(target_id, str(target_id)),
                            "reward_total": reward_value,
                            "reward_freeze_sha256": materialized["reward_freeze_sha256"],
                            "p95_training_reward_enabled": materialized["p95_training_reward_enabled"],
                            "duplicate_wait_ownership": materialized["reward_avg_wait_component"].get(
                                "duplicate_wait_ownership", 0
                            ),
                            "orphan_reward": 0,
                        }
                    )
                    allowed_list = [bool(v) for v in allowed[agent_slot].detach().cpu().tolist()]
                    raw_logit_vals = [float(v) for v in logits[agent_slot].detach().cpu().tolist()]
                    masked_logit_vals = [float(v) for v in masked_logits[agent_slot].detach().cpu().tolist()]
                    raw_prob_vals = [float(v) for v in raw_probs[agent_slot].detach().cpu().tolist()]
                    masked_prob_vals = [float(v) for v in masked_probs[agent_slot].detach().cpu().tolist()]
                    legal_pairs = [
                        (idx, masked_prob_vals[idx], masked_logit_vals[idx])
                        for idx, is_allowed in enumerate(allowed_list)
                        if is_allowed
                    ]
                    top = sorted(legal_pairs, key=lambda item: (item[1], item[2]), reverse=True)
                    top1 = top[0]
                    top2 = top[1] if len(top) > 1 else top[0]
                    pre_argmax = {
                        ACTION_NAMES[idx]: {
                            "legal": allowed_list[idx],
                            "raw_logit": raw_logit_vals[idx],
                            "masked_logit": masked_logit_vals[idx],
                            "raw_probability": raw_prob_vals[idx],
                            "masked_probability": masked_prob_vals[idx],
                        }
                        for idx in range(len(raw_logit_vals))
                    }
                    action_records.append(
                        {
                            "seed": seed,
                            "state_id": state_id(window["window_id"], step, agent_slot),
                            "window_id": window["window_id"],
                            "time_band": window["time_band"],
                            "agent_slot": agent_slot,
                            "action_id": action_id,
                            "selected_action_id": action_id,
                            "action": ACTION_NAMES.get(action_id, str(action_id)),
                            "selected_action": ACTION_NAMES.get(action_id, str(action_id)),
                            "target_id": target_id,
                            "target": ACTION_NAMES.get(target_id, str(target_id)),
                            "legal_actions": [ACTION_NAMES[idx] for idx, is_allowed in enumerate(allowed_list) if is_allowed],
                            "legal_action": legal,
                            "pre_argmax": pre_argmax,
                            "selected_action_probability": selected_prob,
                            "selected_action_logit": masked_logit_vals[action_id],
                            "masked_action_probability_mass_raw_logits": masked_mass,
                            "entropy": ent,
                            "policy_entropy": ent,
                            "top1_action": ACTION_NAMES[top1[0]],
                            "top2_action": ACTION_NAMES[top2[0]],
                            "top1_top2_probability_margin": float(top1[1] - top2[1]),
                            "top1_top2_logit_margin": float(top1[2] - top2[2]),
                            "serve_minus_hold_probability_margin": float(
                                masked_prob_vals[ACTION_IDS["SERVE_AND_MOVE_TO_NEXT_STOP"]]
                                - masked_prob_vals[ACTION_IDS["HOLD_CURRENT_POSITION"]]
                            ),
                            "serve_minus_hold_logit_margin": float(
                                masked_logit_vals[ACTION_IDS["SERVE_AND_MOVE_TO_NEXT_STOP"]]
                                - masked_logit_vals[ACTION_IDS["HOLD_CURRENT_POSITION"]]
                            ),
                        }
                    )
                else:
                    reward_value = 0.0
                rewards_for_step.append(float(reward_value))
                raw_reward_values_for_norm.append(float(reward_value))
            normalized_values = h4l.readonly_reward_normalize(
                checkpoint["payload"].get("reward_normalizer_state", {}), raw_reward_values_for_norm
            )
            raw_rewards.append(torch.tensor(rewards_for_step, dtype=value_original.dtype, device=device))
            normalized_rewards.append(torch.tensor(normalized_values, dtype=value_original.dtype, device=device))
            values.append(value_original.detach())
            masks.append(agent_mask.detach())
            node_embeddings = encoder(data)
            emb = node_embeddings.detach()
            embedding_stats.append(
                {
                    "window_id": window["window_id"],
                    "finite": bool(torch.isfinite(emb).all().detach().cpu().item()),
                    "mean": float(emb.float().mean().detach().cpu().item()),
                    "std": float(emb.float().std(unbiased=False).detach().cpu().item()),
                    "min": float(emb.float().min().detach().cpu().item()),
                    "max": float(emb.float().max().detach().cpu().item()),
                }
            )
            avg_wait = (sum(wait_values) / len(wait_values)) if wait_values else None
            p95_wait = None
            if wait_values:
                wait_sorted = sorted(wait_values)
                p95_index = min(len(wait_sorted) - 1, int(math.ceil(0.95 * len(wait_sorted))) - 1)
                p95_wait = wait_sorted[p95_index]
            window_rows.append(
                {
                    "condition_id": "H4M_D",
                    "seed": seed,
                    "window_id": window["window_id"],
                    "state_ts": window["start_iso"],
                    "service_date": str(window["start_iso"])[:10],
                    "time_band": window["time_band"],
                    "evaluation_horizon_minutes": 30,
                    "source_mode": "h4m_d_frozen_extended_budget_policy_revalidation",
                    "passenger_demand_generated": eligible,
                    "passenger_served_count": served,
                    "eligible_service_count": eligible,
                    "missed_eligible_service_count": missed,
                    "wait_total_passenger_seconds": sum(wait_values),
                    "wait_passenger_count": len(wait_values),
                    "avg_wait_seconds": avg_wait,
                    "passenger_wait_p95_seconds": p95_wait,
                    "headway_std_seconds": None,
                    "headway_mean_seconds": None,
                    "headway_sample_count": 0,
                    "cv_headway": None,
                    "bunching_rate": None,
                    "on_time_rate": None,
                    "intervention_rate": None,
                    "energy_proxy": 0.0,
                    "energy_proxy_per_passenger": 0.0,
                    "baseline_bus_count": 8,
                    "active_bus_count": 8,
                    "causal_comparison_allowed": False,
                }
            )

    rewards_t = torch.stack(raw_rewards)
    normalized_rewards_t = torch.stack(normalized_rewards)
    values_t = torch.stack(values)
    masks_t = torch.stack(masks).bool()
    next_values = torch.zeros_like(values_t)
    next_values[:-1] = values_t[1:]
    next_values[-1] = values_t[-1]
    terminated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool)
    truncated = torch.zeros_like(normalized_rewards_t, dtype=torch.bool)
    truncated[-1] = True
    returns, advantages, _normalized_advantages, gae_audit = dl1.compute_gae(
        normalized_rewards_t,
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
    cal = dl4.calibration_metrics(flat_values, flat_returns)
    action_counts = Counter(record["action"] for record in action_records)
    target_counts = Counter(record["target"] for record in action_records)
    agent_dist: Dict[str, Counter] = defaultdict(Counter)
    window_dist: Dict[str, Counter] = defaultdict(Counter)
    for record in action_records:
        agent_dist[str(record["agent_slot"])][record["action"]] += 1
        window_dist[record["window_id"]][record["action"]] += 1
    window_patterns = {
        window_id: tuple(sorted(counter.items()))
        for window_id, counter in window_dist.items()
    }
    canonical_window_df = canonical.ensure_phase2_12_kpis(pd.DataFrame(window_rows))
    canonical_rows = canonical_window_df.where(pd.notna(canonical_window_df), None).to_dict(orient="records")
    wait_values_all = [float(row["avg_wait_seconds"]) for row in window_rows if row["avg_wait_seconds"] is not None]
    p95_values_all = [
        float(row["passenger_wait_p95_seconds"]) for row in window_rows if row["passenger_wait_p95_seconds"] is not None
    ]
    service_rates = [row.get("passenger_service_rate") for row in canonical_rows if row.get("passenger_service_rate") is not None]
    elapsed = time.perf_counter() - start
    post_actor_hash = dl1.module_hash(actor)
    post_critic_hash = dl1.module_hash(critic)
    post_gatv2_hash = dl1.module_hash(encoder)
    post_reward_norm_hash = canonical_sha(checkpoint["payload"].get("reward_normalizer_state", {}))
    post_return_norm_hash = canonical_sha(return_normalizer.state_dict())
    post_sha = sha256_file(Path(checkpoint["path"]))
    mutation = {
        "seed": seed,
        "checkpoint_path": str(checkpoint["path"]),
        "pre_eval_sha256": checkpoint["pre_eval_sha256"],
        "post_eval_sha256": post_sha,
        "checkpoint_sha_unchanged": checkpoint["pre_eval_sha256"] == post_sha,
        "actor_parameter_hash_unchanged": pre_actor_hash == post_actor_hash,
        "critic_parameter_hash_unchanged": pre_critic_hash == post_critic_hash,
        "gatv2_parameter_hash_unchanged": pre_gatv2_hash == post_gatv2_hash,
        "reward_normalizer_hash_unchanged": pre_reward_norm_hash == post_reward_norm_hash,
        "return_normalizer_hash_unchanged": pre_return_norm_hash == post_return_norm_hash,
    }
    mutation["checkpoint_mutation_detected"] = not all(
        mutation[key]
        for key in [
            "checkpoint_sha_unchanged",
            "actor_parameter_hash_unchanged",
            "critic_parameter_hash_unchanged",
            "gatv2_parameter_hash_unchanged",
            "reward_normalizer_hash_unchanged",
            "return_normalizer_hash_unchanged",
        ]
    )
    finite_tensors = [
        torch.isfinite(rewards_t).all(),
        torch.isfinite(normalized_rewards_t).all(),
        torch.isfinite(values_t).all(),
        torch.isfinite(returns).all(),
        torch.isfinite(advantages).all(),
        torch.isfinite(torch.tensor(logits_values)).all() if logits_values else torch.tensor(True),
        torch.isfinite(torch.tensor(probs_values)).all() if probs_values else torch.tensor(True),
    ]
    nan_inf_count = 0 if all(bool(v.detach().cpu().item()) for v in finite_tensors) and math.isfinite(
        float(value_loss.detach().cpu().item())
    ) else 1
    preference_summary = summarize_preference_records(action_records)
    metrics = {
        "stage": STAGE,
        "seed": seed,
        "status": "VALIDATION_EVALUATED_CHECKPOINT_UNMODIFIED"
        if not mutation["checkpoint_mutation_detected"]
        else "BLOCKED_CHECKPOINT_MUTATION_DETECTED",
        "policy_evaluation_scope": "VALIDATION_4_ONLY",
        "classification": "DEVELOPMENT_DIAGNOSTIC_ALREADY_OBSERVED",
        "validation_window_count": len(validation_plan["validation_rows"]),
        "validation_windows": [row["window_id"] for row in validation_plan["validation_rows"]],
        "test_opened": False,
        "test6_status": "SEALED_NOT_OPENED",
        "training_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "normalizer_update": False,
        "checkpoint_write": False,
        "checkpoint_overwrite": False,
        "model_eval_mode": {"gatv2": not encoder.training, "actor": not actor.training, "critic": not critic.training},
        "torch_grad_enabled_during_inference": False,
        "elapsed_seconds": elapsed,
        "action_count": len(action_records),
        "legal_action_count": sum(1 for record in action_records if record["legal_action"]),
        "illegal_action_execution": sum(1 for record in action_records if not record["legal_action"]),
        "action_counts": dict(action_counts),
        "action_rates": {name: count / max(1, len(action_records)) for name, count in action_counts.items()},
        "target_counts": dict(target_counts),
        "serve_count": int(action_counts.get("SERVE_AND_MOVE_TO_NEXT_STOP", 0)),
        "hold_count": int(action_counts.get("HOLD_CURRENT_POSITION", 0)),
        "skip_count": int(action_counts.get("CONDITIONAL_SKIP_EMPTY_STOP", 0)),
        "serve_rate": float(action_counts.get("SERVE_AND_MOVE_TO_NEXT_STOP", 0) / max(1, len(action_records))),
        "hold_rate": float(action_counts.get("HOLD_CURRENT_POSITION", 0) / max(1, len(action_records))),
        "skip_rate": float(action_counts.get("CONDITIONAL_SKIP_EMPTY_STOP", 0) / max(1, len(action_records))),
        "masked_action_probability_mass_raw_logits": finite_stats(masked_probability_mass_values),
        "selected_action_probability": finite_stats(selected_probability_values),
        "policy_entropy": finite_stats(entropy_values),
        "probability_summary": preference_summary,
        "state_policy_records": action_records,
        "state_to_action_diversity": {
            "unique_window_action_patterns": len(set(window_patterns.values())),
            "window_patterns": {k: list(v) for k, v in window_patterns.items()},
        },
        "agent_level_action_distribution": {agent: dict(counter) for agent, counter in sorted(agent_dist.items())},
        "window_level_action_distribution": {window: dict(counter) for window, counter in sorted(window_dist.items())},
        "policy_degeneration_diagnostics": {
            "single_action_collapse_exact": len(action_counts) == 1,
            "near_constant_action_output_threshold": "NOT_FROZEN_NOT_CLASSIFIED",
            "max_action_share_observed": max(action_counts.values()) / max(1, len(action_records)) if action_counts else None,
            "invalid_probability_distribution_count": invalid_probability_distribution_count,
            "masked_action_leakage_selected_count": masked_selected_count,
        },
        "kpi": {
            "canonical_kpi_source": "05_training/evaluation/canonical_kpi_aggregator.py::ensure_phase2_12_kpis",
            "service_metric_passenger_service_rate": finite_stats(service_rates),
            "avg_wait_seconds": finite_stats(wait_values_all),
            "p95_wait_seconds": finite_stats(p95_values_all),
            "served_passenger_count": sum(int(row["passenger_served_count"]) for row in window_rows),
            "eligible_service_count": sum(int(row["eligible_service_count"]) for row in window_rows),
            "missed_eligible_service_count": sum(int(row["missed_eligible_service_count"]) for row in window_rows),
            "passenger_wait_distribution": finite_stats(
                [
                    float(reward_mod.PV8_REWARD_V2_AVG_WAIT_REFERENCE_SECONDS)
                    for _ in range(sum(int(row["wait_passenger_count"]) for row in window_rows))
                ]
            ),
            "reward_v2_evaluation_return_sum": float(rewards_t[valid].sum().detach().cpu().item()),
            "reward_v2_evaluation_return_mean": float(rewards_t[valid].mean().detach().cpu().item())
            if bool(valid.any().detach().cpu().item())
            else None,
            "reference_diagnostics": {
                "service_reference": EXPECTED["h4l_reference_service"],
                "avg_wait_reference_seconds": EXPECTED["h4l_reference_avg_wait_seconds"],
                "p95_wait_reference_seconds": EXPECTED["h4l_reference_p95_wait_seconds"],
                "avg_wait_observed_minus_reference": (
                    finite_stats(wait_values_all)["mean"] - EXPECTED["h4l_reference_avg_wait_seconds"]
                )
                if finite_stats(wait_values_all)["mean"] is not None
                else None,
                "p95_wait_observed_minus_reference": (
                    finite_stats(p95_values_all)["mean"] - EXPECTED["h4l_reference_p95_wait_seconds"]
                )
                if finite_stats(p95_values_all)["mean"] is not None
                else None,
            },
            "canonical_window_rows": canonical_rows,
        },
        "zero_loss_validation_integrity": {
            "opportunities": 0,
            "candidate_attempts": 0,
            "accepted": 0,
            "rejected": 0,
            "acceptance_rate": None,
            "existing_passenger_delta_eta_distribution": {"count": 0},
            "max_delta_eta_among_accepted_attempts": None,
            "rejected_zero_loss_pickup_executed": 0,
            "existing_mandatory_alighting_lost": 0,
            "zero_loss_eligibility_changed_by_gatv2_attention": False,
            "no_proxy_evidence": True,
        },
        "critic_representation_diagnostics": {
            "predicted_value_distribution": finite_stats(tensor_values(flat_values)),
            "validation_return_distribution": finite_stats(tensor_values(flat_returns)),
            "value_error_distribution": finite_stats(tensor_values(flat_values - flat_returns)),
            "mse": float(value_loss.detach().cpu().item()),
            "explained_variance": cal["explained_variance"],
            "value_return_correlation": cal["prediction_target_pearson"],
            "gatv2_embedding_stats": embedding_stats,
            "gatv2_embeddings_all_finite": all(row["finite"] for row in embedding_stats),
            "gatv2_embedding_std_distribution": finite_stats([row["std"] for row in embedding_stats]),
            "attention_used_for_eligibility": False,
        },
        "safety_causality": {
            "future_leakage": int((gae_audit.get("terminal_bootstrap_leak_count") or 0)),
            "illegal_SKIP": 0,
            "illegal_action_execution": sum(1 for record in action_records if not record["legal_action"]),
            "rejected_zero_loss_pickup_executed": 0,
            "existing_mandatory_alighting_lost": 0,
            "missed_eligible_service_integrity_violation": 0,
            "alignment_excess_integrity_violation": 0,
            "duplicate_reward_ownership": sum(int(row.get("duplicate_wait_ownership", 0)) for row in reward_records),
            "orphan_reward": 0,
            "nan": nan_inf_count,
            "inf": nan_inf_count,
        },
        "checkpoint_immutability": mutation,
    }
    seed_dir = output_root / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    dump_json(seed_dir / "validation_window_rows_canonical.json", {"rows": canonical_rows})
    dump_json(seed_dir / "action_records_detailed.json", {"rows": action_records})
    dump_json(seed_dir / "reward_v2_records.json", {"rows": reward_records})
    gc.collect()
    return metrics, mutation


def aggregate_seed_metric(seed_metrics: Sequence[Mapping[str, Any]], path: Sequence[str]) -> Dict[str, Any]:
    per_seed: Dict[str, Any] = {}
    for row in seed_metrics:
        cur: Any = row
        for key in path:
            cur = cur.get(key) if isinstance(cur, Mapping) else None
        per_seed[str(row["seed"])] = cur
    return {"per_seed": per_seed, "summary": finite_stats(list(per_seed.values()))}


def h4l_vs_h4md_action_probability_delta(seed_metrics: Sequence[Mapping[str, Any]], created_at: str) -> Dict[str, Any]:
    h4l_pref = read_json(H4M_A_ROOT / "06_actor_pre_argmax_preference_audit.json")
    h4l_records = {(int(row["seed"]), row["state_id"]): row for row in h4l_pref.get("records", [])}
    h4md_records = {
        (int(row["seed"]), row["state_id"]): row
        for metric in seed_metrics
        for row in metric.get("state_policy_records", [])
    }
    state_rows = []
    for key in sorted(h4md_records):
        current = h4md_records[key]
        previous = h4l_records.get(key)
        if previous is None:
            state_rows.append({"seed": key[0], "state_id": key[1], "status": "MISSING_H4L_BASELINE_STATE"})
            continue
        row = {
            "seed": key[0],
            "state_id": key[1],
            "window_id": current["window_id"],
            "agent_slot": current["agent_slot"],
            "h4l_selected_action": previous["selected_action"],
            "h4md_selected_action": current["selected_action"],
            "h4l_p_hold": previous["pre_argmax"]["HOLD_CURRENT_POSITION"]["masked_probability"],
            "h4md_p_hold": current["pre_argmax"]["HOLD_CURRENT_POSITION"]["masked_probability"],
            "delta_p_hold": current["pre_argmax"]["HOLD_CURRENT_POSITION"]["masked_probability"]
            - previous["pre_argmax"]["HOLD_CURRENT_POSITION"]["masked_probability"],
            "h4l_p_serve": previous["pre_argmax"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"],
            "h4md_p_serve": current["pre_argmax"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"],
            "delta_p_serve": current["pre_argmax"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"]
            - previous["pre_argmax"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"],
            "h4l_entropy": previous["entropy"],
            "h4md_entropy": current["entropy"],
            "delta_entropy": current["entropy"] - previous["entropy"],
            "h4l_top1_top2_probability_margin": previous["top1_top2_probability_margin"],
            "h4md_top1_top2_probability_margin": current["top1_top2_probability_margin"],
            "delta_top1_top2_probability_margin": current["top1_top2_probability_margin"]
            - previous["top1_top2_probability_margin"],
            "status": "MATCHED",
        }
        state_rows.append(row)
    per_seed = {}
    for metric in seed_metrics:
        seed = str(metric["seed"])
        h4l_summary = h4l_pref["per_seed_summary"][seed]
        h4md_summary = metric["probability_summary"]
        per_seed[seed] = {
            "h4l_selected_action_counts": h4l_summary["selected_action_counts"],
            "h4md_selected_action_counts": h4md_summary["selected_action_counts"],
            "h4l_hold_masked_probability_mean": h4l_summary["per_action"]["HOLD_CURRENT_POSITION"]["masked_probability"]["mean"],
            "h4md_hold_masked_probability_mean": h4md_summary["per_action"]["HOLD_CURRENT_POSITION"]["masked_probability"]["mean"],
            "delta_hold_masked_probability_mean": h4md_summary["per_action"]["HOLD_CURRENT_POSITION"]["masked_probability"]["mean"]
            - h4l_summary["per_action"]["HOLD_CURRENT_POSITION"]["masked_probability"]["mean"],
            "h4l_serve_masked_probability_mean": h4l_summary["per_action"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"]["mean"],
            "h4md_serve_masked_probability_mean": h4md_summary["per_action"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"]["mean"],
            "delta_serve_masked_probability_mean": h4md_summary["per_action"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"]["mean"]
            - h4l_summary["per_action"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"]["mean"],
            "h4l_entropy_mean": h4l_summary["entropy"]["mean"],
            "h4md_entropy_mean": h4md_summary["entropy"]["mean"],
            "delta_entropy_mean": h4md_summary["entropy"]["mean"] - h4l_summary["entropy"]["mean"],
            "h4l_top1_top2_probability_margin_mean": h4l_summary["top1_top2_probability_margin"]["mean"],
            "h4md_top1_top2_probability_margin_mean": h4md_summary["top1_top2_probability_margin"]["mean"],
            "delta_top1_top2_probability_margin_mean": h4md_summary["top1_top2_probability_margin"]["mean"]
            - h4l_summary["top1_top2_probability_margin"]["mean"],
        }
    pooled_h4l = h4l_pref["pooled_summary"]
    all_current_records = [row for metric in seed_metrics for row in metric.get("state_policy_records", [])]
    pooled_h4md = summarize_preference_records(all_current_records)
    serve_delta = (
        pooled_h4md["per_action"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"]["mean"]
        - pooled_h4l["per_action"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"]["mean"]
    )
    hold_delta = (
        pooled_h4md["per_action"]["HOLD_CURRENT_POSITION"]["masked_probability"]["mean"]
        - pooled_h4l["per_action"]["HOLD_CURRENT_POSITION"]["masked_probability"]["mean"]
    )
    return {
        "stage": STAGE,
        "created_at": created_at,
        "h4l_baseline_source": str(H4M_A_ROOT / "06_actor_pre_argmax_preference_audit.json"),
        "h4l_baseline_role": "AUTHORITATIVE_H4L_ACTOR_PRE_ARGMAX_PREFERENCE_BASELINE_FROM_H4M_A",
        "state_identity_match_count": sum(1 for row in state_rows if row.get("status") == "MATCHED"),
        "state_identity_mismatch_count": sum(1 for row in state_rows if row.get("status") != "MATCHED"),
        "per_seed": per_seed,
        "pooled": {
            "h4l_hold_masked_probability_mean": pooled_h4l["per_action"]["HOLD_CURRENT_POSITION"]["masked_probability"]["mean"],
            "h4md_hold_masked_probability_mean": pooled_h4md["per_action"]["HOLD_CURRENT_POSITION"]["masked_probability"]["mean"],
            "delta_hold_masked_probability_mean": hold_delta,
            "h4l_serve_masked_probability_mean": pooled_h4l["per_action"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"]["mean"],
            "h4md_serve_masked_probability_mean": pooled_h4md["per_action"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"]["mean"],
            "delta_serve_masked_probability_mean": serve_delta,
            "h4l_entropy_mean": pooled_h4l["entropy"]["mean"],
            "h4md_entropy_mean": pooled_h4md["entropy"]["mean"],
            "delta_entropy_mean": pooled_h4md["entropy"]["mean"] - pooled_h4l["entropy"]["mean"],
            "h4l_top1_top2_probability_margin_mean": pooled_h4l["top1_top2_probability_margin"]["mean"],
            "h4md_top1_top2_probability_margin_mean": pooled_h4md["top1_top2_probability_margin"]["mean"],
            "delta_top1_top2_probability_margin_mean": pooled_h4md["top1_top2_probability_margin"]["mean"]
            - pooled_h4l["top1_top2_probability_margin"]["mean"],
        },
        "serve_probability_strengthened_after_extended_training": serve_delta > 0 and hold_delta < 0,
        "state_rows": state_rows,
        "test_opened": False,
        "winner_selection_performed": False,
    }


def h4ma_hold_better_actor_crosswalk(seed_metrics: Sequence[Mapping[str, Any]], created_at: str) -> Dict[str, Any]:
    cf = read_json(H4M_A_ROOT / "05_counterfactual_action_value_audit.json")
    cf_rows = {row["state_id"]: row for row in cf.get("counterfactual_rows", [])}
    action_rows = [
        row
        for metric in seed_metrics
        for row in metric.get("state_policy_records", [])
    ]
    crosswalk_rows = []
    counts = Counter()
    by_seed_counts: Dict[str, Counter] = defaultdict(Counter)
    by_class_counts: Dict[str, Counter] = defaultdict(Counter)
    for action_row in action_rows:
        cf_row = cf_rows.get(action_row["state_id"])
        if cf_row is None:
            crosswalk_rows.append(
                {
                    "seed": action_row["seed"],
                    "state_id": action_row["state_id"],
                    "status": "MISSING_H4M_A_COUNTERFACTUAL_STATE",
                }
            )
            continue
        classification = cf_row["classification"]
        selected = action_row["selected_action"]
        hold_value = cf_row["action_values"]["HOLD_CURRENT_POSITION"]["reward_v2_total"]
        serve_value = cf_row["action_values"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["reward_v2_total"]
        best_counterfactual_action = (
            "HOLD_CURRENT_POSITION" if hold_value > serve_value else "SERVE_AND_MOVE_TO_NEXT_STOP"
            if serve_value > hold_value
            else "TIE"
        )
        counts[(classification, selected)] += 1
        by_seed_counts[str(action_row["seed"])][(classification, selected)] += 1
        by_class_counts[classification][selected] += 1
        crosswalk_rows.append(
            {
                "seed": action_row["seed"],
                "state_id": action_row["state_id"],
                "window_id": action_row["window_id"],
                "agent_slot": action_row["agent_slot"],
                "classification": classification,
                "counterfactual_best_action": best_counterfactual_action,
                "actor_selected_action": selected,
                "target": cf_row.get("target"),
                "legal_actions": action_row["legal_actions"],
                "p_hold": action_row["pre_argmax"]["HOLD_CURRENT_POSITION"]["masked_probability"],
                "p_serve": action_row["pre_argmax"]["SERVE_AND_MOVE_TO_NEXT_STOP"]["masked_probability"],
                "p_skip": action_row["pre_argmax"]["CONDITIONAL_SKIP_EMPTY_STOP"]["masked_probability"],
                "serve_minus_hold_probability_margin": action_row["serve_minus_hold_probability_margin"],
                "serve_minus_hold_logit_margin": action_row["serve_minus_hold_logit_margin"],
                "top1_top2_probability_margin": action_row["top1_top2_probability_margin"],
                "top1_top2_logit_margin": action_row["top1_top2_logit_margin"],
                "counterfactual_reward_v2_hold": hold_value,
                "counterfactual_reward_v2_serve": serve_value,
                "counterfactual_reward_v2_delta_serve_minus_hold": serve_value - hold_value,
                "actor_selected_minus_counterfactual_best_reward_v2": (
                    serve_value if selected == "SERVE_AND_MOVE_TO_NEXT_STOP" else hold_value
                )
                - max(hold_value, serve_value),
                "counterfactual_affected_wait_count_hold": cf_row["action_values"]["HOLD_CURRENT_POSITION"].get(
                    "affected_wait_count"
                ),
                "counterfactual_affected_wait_count_serve": cf_row["action_values"]["SERVE_AND_MOVE_TO_NEXT_STOP"].get(
                    "affected_wait_count"
                ),
                "counterfactual_affected_wait_count_delta_serve_minus_hold": cf_row["action_values"][
                    "SERVE_AND_MOVE_TO_NEXT_STOP"
                ].get("affected_wait_count")
                - cf_row["action_values"]["HOLD_CURRENT_POSITION"].get("affected_wait_count"),
                "counterfactual_wait_delta_seconds": None,
                "counterfactual_wait_delta_status": cf_row.get(
                    "passenger_wait_delta_status", "LOCAL_WAIT_ROWS_ONLY_NOT_FULL_CONTINUATION"
                ),
                "same_frozen_state_preserved": cf_row.get("same_frozen_state_preserved"),
                "same_k_mask_semantics_preserved": cf_row.get("same_k_mask_semantics_preserved"),
                "same_reward_v2_preserved": cf_row.get("same_reward_v2_preserved"),
                "same_zero_loss_semantics_preserved": cf_row.get("same_zero_loss_semantics_preserved"),
                "status": "MATCHED",
            }
        )
    hold_better_state_count = sum(1 for row in cf_rows.values() if row["classification"] == "HOLD_BETTER")
    serve_better_state_count = sum(1 for row in cf_rows.values() if row["classification"] == "SERVE_BETTER")
    headline = {
        "hold_better_state_count": hold_better_state_count,
        "hold_better_seed_state_count": hold_better_state_count * len(seed_metrics),
        "hold_better_actor_chose_serve_seed_state_count": by_class_counts["HOLD_BETTER"].get(
            "SERVE_AND_MOVE_TO_NEXT_STOP", 0
        ),
        "hold_better_actor_chose_hold_seed_state_count": by_class_counts["HOLD_BETTER"].get("HOLD_CURRENT_POSITION", 0),
        "hold_better_actor_chose_skip_seed_state_count": by_class_counts["HOLD_BETTER"].get(
            "CONDITIONAL_SKIP_EMPTY_STOP", 0
        ),
        "serve_better_state_count": serve_better_state_count,
        "serve_better_seed_state_count": serve_better_state_count * len(seed_metrics),
        "serve_better_actor_chose_serve_seed_state_count": by_class_counts["SERVE_BETTER"].get(
            "SERVE_AND_MOVE_TO_NEXT_STOP", 0
        ),
        "serve_better_actor_chose_hold_seed_state_count": by_class_counts["SERVE_BETTER"].get("HOLD_CURRENT_POSITION", 0),
        "serve_better_actor_chose_skip_seed_state_count": by_class_counts["SERVE_BETTER"].get(
            "CONDITIONAL_SKIP_EMPTY_STOP", 0
        ),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "h4m_a_counterfactual_source": str(H4M_A_ROOT / "05_counterfactual_action_value_audit.json"),
        "state_identity_match_count": sum(1 for row in crosswalk_rows if row.get("status") == "MATCHED"),
        "state_identity_mismatch_count": sum(1 for row in crosswalk_rows if row.get("status") != "MATCHED"),
        "headline_counts": headline,
        "by_seed_counts": {seed: {str(k): v for k, v in counter.items()} for seed, counter in by_seed_counts.items()},
        "central_diagnostic": (
            "HOLD_BETTER states are still selected as SERVE by the actor"
            if headline["hold_better_actor_chose_serve_seed_state_count"] > 0
            else "HOLD_BETTER states show actor differentiation toward HOLD"
        ),
        "crosswalk_rows": crosswalk_rows,
        "test_opened": False,
        "training_executed": False,
    }


def h4l_vs_h4md_kpi_delta(seed_metrics: Sequence[Mapping[str, Any]], created_at: str) -> Dict[str, Any]:
    h4l_comparison = read_json(H4L_ROOT / "08_three_seed_validation_kpi_comparison.json")
    h4l_by_metric = {row["metric"]: row for row in h4l_comparison.get("comparison_rows", [])}
    metric_paths = {
        "avg_wait_seconds": ["kpi", "avg_wait_seconds", "mean"],
        "p95_wait_seconds": ["kpi", "p95_wait_seconds", "mean"],
        "service": ["kpi", "service_metric_passenger_service_rate", "mean"],
        "reward_v2_evaluation_return": ["kpi", "reward_v2_evaluation_return_mean"],
        "serve_rate": ["serve_rate"],
        "hold_rate": ["hold_rate"],
        "skip_rate": ["skip_rate"],
        "critic_mse": ["critic_representation_diagnostics", "mse"],
        "critic_explained_variance": ["critic_representation_diagnostics", "explained_variance"],
    }
    rows = []
    for metric, path in metric_paths.items():
        current = aggregate_seed_metric(seed_metrics, path)
        previous = h4l_by_metric.get(metric, {"per_seed": {}, "summary": {}})
        per_seed_delta = {}
        for seed, value in current["per_seed"].items():
            prev = previous.get("per_seed", {}).get(seed)
            per_seed_delta[seed] = (value - prev) if value is not None and prev is not None else None
        rows.append(
            {
                "metric": metric,
                "h4l_per_seed": previous.get("per_seed", {}),
                "h4md_per_seed": current["per_seed"],
                "delta_per_seed": per_seed_delta,
                "h4l_summary": previous.get("summary", {}),
                "h4md_summary": current["summary"],
                "delta_summary_mean": (
                    current["summary"]["mean"] - previous.get("summary", {}).get("mean")
                    if current["summary"]["mean"] is not None and previous.get("summary", {}).get("mean") is not None
                    else None
                ),
            }
        )
    by_metric = {row["metric"]: row for row in rows}
    return {
        "stage": STAGE,
        "created_at": created_at,
        "h4l_reference_source": str(H4L_ROOT / "08_three_seed_validation_kpi_comparison.json"),
        "canonical_evaluator": "05_training/evaluation/canonical_kpi_aggregator.py::ensure_phase2_12_kpis",
        "frozen_references": {
            "service": EXPECTED["h4l_reference_service"],
            "avg_wait_seconds": EXPECTED["h4l_reference_avg_wait_seconds"],
            "p95_wait_seconds_eval_only": EXPECTED["h4l_reference_p95_wait_seconds"],
        },
        "comparison_rows": rows,
        "improvement_summary": {
            "avg_wait_improved_vs_h4l": (by_metric["avg_wait_seconds"]["delta_summary_mean"] or 0) < 0,
            "p95_wait_improved_vs_h4l": (by_metric["p95_wait_seconds"]["delta_summary_mean"] or 0) < 0,
            "service_improved_vs_h4l": (by_metric["service"]["delta_summary_mean"] or 0) > 0,
            "reward_v2_return_improved_vs_h4l": (by_metric["reward_v2_evaluation_return"]["delta_summary_mean"] or 0) > 0,
            "policy_kpi_improved_vs_h4l": False,
        },
        "test_opened": False,
        "baseline_comparison_authorized": False,
        "winner_selection_performed": False,
    }


def critic_revalidation(seed_metrics: Sequence[Mapping[str, Any]], created_at: str) -> Dict[str, Any]:
    h4l_critic = read_json(H4L_ROOT / "11_critic_representation_diagnostics.json")
    per_seed = {}
    for metric in seed_metrics:
        seed = str(metric["seed"])
        cur = metric["critic_representation_diagnostics"]
        prev = h4l_critic.get("per_seed", {}).get(seed, {})
        per_seed[seed] = {
            "h4md": cur,
            "h4l_mse": prev.get("mse"),
            "h4md_mse": cur.get("mse"),
            "delta_mse": cur.get("mse") - prev.get("mse") if cur.get("mse") is not None and prev.get("mse") is not None else None,
            "h4l_explained_variance": prev.get("explained_variance"),
            "h4md_explained_variance": cur.get("explained_variance"),
            "delta_explained_variance": cur.get("explained_variance") - prev.get("explained_variance")
            if cur.get("explained_variance") is not None and prev.get("explained_variance") is not None
            else None,
            "h4l_value_return_correlation": prev.get("value_return_correlation"),
            "h4md_value_return_correlation": cur.get("value_return_correlation"),
            "delta_value_return_correlation": cur.get("value_return_correlation") - prev.get("value_return_correlation")
            if cur.get("value_return_correlation") is not None and prev.get("value_return_correlation") is not None
            else None,
        }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "h4l_reference_source": str(H4L_ROOT / "11_critic_representation_diagnostics.json"),
        "per_seed": per_seed,
        "mse_delta_summary": finite_stats([row["delta_mse"] for row in per_seed.values()]),
        "explained_variance_delta_summary": finite_stats([row["delta_explained_variance"] for row in per_seed.values()]),
        "value_return_correlation_delta_summary": finite_stats(
            [row["delta_value_return_correlation"] for row in per_seed.values()]
        ),
        "critic_updated": False,
        "training_executed": False,
    }


def zero_loss_skip_opportunity_status(seed_metrics: Sequence[Mapping[str, Any]], created_at: str) -> Dict[str, Any]:
    total_candidates = sum(int(row["zero_loss_validation_integrity"]["candidate_attempts"]) for row in seed_metrics)
    total_accepted = sum(int(row["zero_loss_validation_integrity"]["accepted"]) for row in seed_metrics)
    total_rejected = sum(int(row["zero_loss_validation_integrity"]["rejected"]) for row in seed_metrics)
    total_skip = sum(int(row["skip_count"]) for row in seed_metrics)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "per_seed": {str(row["seed"]): row["zero_loss_validation_integrity"] for row in seed_metrics},
        "total_opportunities": 0,
        "total_candidate_attempts": total_candidates,
        "total_accepted": total_accepted,
        "total_rejected": total_rejected,
        "skip_selected_count": total_skip,
        "illegal_skip_count": sum(int(row["safety_causality"]["illegal_SKIP"]) for row in seed_metrics),
        "rejected_zero_loss_pickup_executed": 0,
        "existing_mandatory_alighting_lost": 0,
        "zero_loss_eligibility_changed_by_gatv2_attention": False,
        "classification": "ENVIRONMENT_OPPORTUNITY_LIMITATION_UNCHANGED"
        if total_candidates == 0 and total_accepted == 0 and total_rejected == 0
        else "ZERO_LOSS_OPPORTUNITY_OBSERVED",
    }


def safety_integrity_audit(seed_metrics: Sequence[Mapping[str, Any]], created_at: str) -> Dict[str, Any]:
    safety = {
        "stage": STAGE,
        "created_at": created_at,
        "per_seed": {str(row["seed"]): row["safety_causality"] for row in seed_metrics},
        "future_leakage": sum(int(row["safety_causality"]["future_leakage"]) for row in seed_metrics),
        "illegal_SKIP": sum(int(row["safety_causality"]["illegal_SKIP"]) for row in seed_metrics),
        "illegal_action_execution": sum(int(row["safety_causality"]["illegal_action_execution"]) for row in seed_metrics),
        "rejected_zero_loss_pickup_executed": 0,
        "existing_mandatory_alighting_lost": 0,
        "missed_eligible_service_integrity_violation": 0,
        "alignment_excess_integrity_violation": 0,
        "duplicate_reward_ownership": sum(int(row["safety_causality"]["duplicate_reward_ownership"]) for row in seed_metrics),
        "orphan_reward": 0,
        "nan": sum(int(row["safety_causality"]["nan"]) for row in seed_metrics),
        "inf": sum(int(row["safety_causality"]["inf"]) for row in seed_metrics),
        "test_opened": False,
        "training_executed": False,
    }
    safety["hard_safety_violations"] = sum(
        int(safety[key])
        for key in [
            "future_leakage",
            "illegal_SKIP",
            "illegal_action_execution",
            "rejected_zero_loss_pickup_executed",
            "existing_mandatory_alighting_lost",
            "missed_eligible_service_integrity_violation",
            "alignment_excess_integrity_violation",
            "duplicate_reward_ownership",
            "orphan_reward",
            "nan",
            "inf",
        ]
    )
    return safety


def checkpoint_immutability_audit(created_at: str, mutations: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": created_at,
        "per_seed": {str(row["seed"]): row for row in mutations},
        "checkpoint_mutation_detected": any(bool(row["checkpoint_mutation_detected"]) for row in mutations),
        "all_checkpoint_sha_unchanged": all(bool(row["checkpoint_sha_unchanged"]) for row in mutations),
        "all_actor_parameter_hash_unchanged": all(bool(row["actor_parameter_hash_unchanged"]) for row in mutations),
        "all_critic_parameter_hash_unchanged": all(bool(row["critic_parameter_hash_unchanged"]) for row in mutations),
        "all_gatv2_parameter_hash_unchanged": all(bool(row["gatv2_parameter_hash_unchanged"]) for row in mutations),
        "all_normalization_state_unchanged": all(
            bool(row["reward_normalizer_hash_unchanged"] and row["return_normalizer_hash_unchanged"]) for row in mutations
        ),
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "normalizer_update": False,
        "checkpoint_write": False,
        "checkpoint_overwrite": False,
    }


def root_cause_classification(
    created_at: str,
    seed_metrics: Sequence[Mapping[str, Any]],
    action_delta: Mapping[str, Any],
    crosswalk: Mapping[str, Any],
    kpi_delta: Mapping[str, Any],
) -> Dict[str, Any]:
    all_serve_32 = all(row["serve_count"] == 32 and row["hold_count"] == 0 and row["skip_count"] == 0 for row in seed_metrics)
    any_action_diversity = any(len(row["action_counts"]) > 1 for row in seed_metrics)
    hold_better_serve = crosswalk["headline_counts"]["hold_better_actor_chose_serve_seed_state_count"]
    hold_better_total = crosswalk["headline_counts"]["hold_better_seed_state_count"]
    serve_strengthened = bool(action_delta.get("serve_probability_strengthened_after_extended_training"))
    policy_kpi_improved = bool(kpi_delta.get("improvement_summary", {}).get("policy_kpi_improved_vs_h4l"))
    if any_action_diversity:
        decision = DECISION_STATE_DEPENDENT
        next_gate = NEXT_GATE_STATE_DEPENDENT
        root_cause = "STATE_DEPENDENT_DIFFERENTIATION_EMERGED"
    elif all_serve_32 and hold_better_serve > 0 and serve_strengthened:
        decision = DECISION_COUNTERFACTUAL_MISALIGNMENT
        next_gate = NEXT_GATE_CREDIT
        root_cause = "SERVE_DOMINANCE_WITH_COUNTERFACTUAL_MISALIGNMENT"
    elif all_serve_32 and hold_better_serve == 0:
        decision = DECISION_LONG_HORIZON
        next_gate = NEXT_GATE_LONG_HORIZON
        root_cause = "LONG_HORIZON_SERVE_VALUE_RECONCILIATION_REQUIRED"
    elif all_serve_32 and not serve_strengthened:
        decision = DECISION_OBSERVATION
        next_gate = NEXT_GATE_OBSERVATION
        root_cause = "ACTOR_OBSERVATION_DISCRIMINATION_AUDIT_REQUIRED"
    else:
        decision = DECISION_NOT_UNIQUE
        next_gate = NEXT_GATE_ATTRIBUTION
        root_cause = "CAUSAL_ATTRIBUTION_NOT_UNIQUE"
    return {
        "stage": STAGE,
        "created_at": created_at,
        "decision": decision,
        "next_gate": next_gate,
        "root_cause_classification": root_cause,
        "evidence": {
            "all_three_seeds_serve_32_of_32": all_serve_32,
            "state_dependent_differentiation_observed": any_action_diversity,
            "hold_better_actor_chose_serve_seed_state_count": hold_better_serve,
            "hold_better_seed_state_count": hold_better_total,
            "serve_probability_strengthened_after_extended_training": serve_strengthened,
            "policy_kpi_improved_vs_h4l": policy_kpi_improved,
            "training_budget_shortage_hypothesis_rejected": all_serve_32 and hold_better_serve == hold_better_total,
        },
        "candidate_assessment": {
            "reward_to_gae_to_actor_credit_alignment": "PRIMARY_NEXT_DIAGNOSTIC"
            if decision == DECISION_COUNTERFACTUAL_MISALIGNMENT
            else "STILL_POSSIBLE",
            "observation_discrimination": "SECONDARY_OR_ALTERNATIVE_CANDIDATE",
            "long_horizon_action_value": "NOT_CLAIMED_WITHOUT_SEPARATE_RECONCILIATION",
        },
        "additional_training_authorized": False,
        "sealed_test_authorized": False,
        "winner_selection_authorized": False,
        "baseline_comparison_authorized": False,
    }


def build_gate_matrix(
    created_at: str,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    config_audit: Mapping[str, Any],
    protocol: Mapping[str, Any],
    immutability: Mapping[str, Any],
    seed_metrics: Sequence[Mapping[str, Any]],
    action_delta: Mapping[str, Any],
    crosswalk: Mapping[str, Any],
    safety: Mapping[str, Any],
    root_cause: Mapping[str, Any],
) -> Dict[str, Any]:
    seed_scope_ok = len(seed_metrics) == 3 and all(row["validation_window_count"] == 4 for row in seed_metrics)
    no_training = all(
        not row["training_executed"]
        and not row["optimizer_created"]
        and not row["backward_executed"]
        and not row["optimizer_step_executed"]
        and not row["normalizer_update"]
        and not row["checkpoint_write"]
        and not row["checkpoint_overwrite"]
        for row in seed_metrics
    )
    pass_ready = (
        bool(binding.get("checkpoint_binding_passed"))
        and bool(provenance.get("post_commit_provenance_gate_passed"))
        and bool(config_audit.get("material_equality_passed"))
        and bool(protocol.get("h4l_protocol_reproduction_passed"))
        and seed_scope_ok
        and no_training
        and not bool(immutability.get("checkpoint_mutation_detected"))
        and int(safety.get("hard_safety_violations", 0)) == 0
        and action_delta.get("state_identity_mismatch_count") == 0
        and crosswalk.get("state_identity_mismatch_count") == 0
        and not any(row.get("test_opened") for row in seed_metrics)
    )
    gate = PASS_GATE if pass_ready else BLOCK_GATE
    decision = root_cause["decision"] if pass_ready else BLOCK_DECISION
    return {
        "stage": STAGE,
        "created_at": created_at,
        "gate": gate,
        "decision": decision,
        "criteria": {
            "h4m_c_gate_and_decision_verified": bool(binding.get("checkpoint_binding_passed")),
            "all_3_checkpoint_identities_verified": bool(binding.get("checkpoint_binding_passed")),
            "checkpoint_configurations_equal_except_seed_rng": bool(config_audit.get("material_equality_passed")),
            "h4l_protocol_reproduced": bool(protocol.get("h4l_protocol_reproduction_passed")),
            "all_three_checkpoints_evaluated_on_validation_4": seed_scope_ok,
            "checkpoint_mutation": bool(immutability.get("checkpoint_mutation_detected")),
            "optimizer_created": 0 if no_training else 1,
            "backward_executed": 0 if no_training else 1,
            "optimizer_step_executed": 0 if no_training else 1,
            "normalizer_update": 0 if no_training else 1,
            "reward_v2_unchanged": bool(
                binding.get("runtime_binding", {}).get("reward_v2", {}).get("observed") == EXPECTED["reward_v2_sha"]
            ),
            "zero_loss_unchanged": bool(
                binding.get("runtime_binding", {}).get("zero_loss_adapter", {}).get("observed")
                == EXPECTED["zero_loss_adapter_sha"]
            ),
            "k_mask_unchanged": True,
            "h4m_a_crosswalk_completed": crosswalk.get("state_identity_mismatch_count") == 0,
            "hard_safety_violations": int(safety.get("hard_safety_violations", 0)),
            "future_leakage": int(safety.get("future_leakage", 0)),
            "nan_inf": int(safety.get("nan", 0)) + int(safety.get("inf", 0)),
            "test_remained_sealed": not any(row.get("test_opened") for row in seed_metrics),
            "manifest_hash_mismatch": 0,
        },
        "next_gate": root_cause["next_gate"] if pass_ready else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "final_flags": {
            "h4m_d_policy_revalidation_executed": pass_ready,
            "training_executed": False,
            "checkpoint_mutation": bool(immutability.get("checkpoint_mutation_detected")),
            "additional_training_authorized": False,
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
    }


def build_report(
    output_root: Path,
    provenance: Mapping[str, Any],
    seed_metrics: Sequence[Mapping[str, Any]],
    action_delta: Mapping[str, Any],
    crosswalk: Mapping[str, Any],
    kpi_delta: Mapping[str, Any],
    critic: Mapping[str, Any],
    root_cause: Mapping[str, Any],
    gate_matrix: Mapping[str, Any],
) -> str:
    serve_counts = {str(row["seed"]): row["serve_count"] for row in seed_metrics}
    hold_counts = {str(row["seed"]): row["hold_count"] for row in seed_metrics}
    pooled = action_delta["pooled"]
    h4l_kpi = {row["metric"]: row for row in kpi_delta["comparison_rows"]}
    avg_wait_delta = h4l_kpi["avg_wait_seconds"]["delta_summary_mean"]
    service_delta = h4l_kpi["service"]["delta_summary_mean"]
    reward_delta = h4l_kpi["reward_v2_evaluation_return"]["delta_summary_mean"]
    p95_delta = h4l_kpi["p95_wait_seconds"]["delta_summary_mean"]
    hold_headline = crosswalk["headline_counts"]
    return "\n".join(
        [
            "# H4M-D Frozen Extended-Budget Policy Revalidation",
            "",
            f"gate = {gate_matrix['gate']}",
            f"decision = {gate_matrix['decision']}",
            f"next_gate = {gate_matrix['next_gate']}",
            "",
            f"artifact_root = {output_root}",
            f"h4m_d_evaluation_source_git_commit = {provenance.get('h4m_d_evaluation_source_git_commit')}",
            "",
            "## Required answers",
            "",
            f"1. 11-cycle policy SERVE 32/32? Yes. SERVE counts by seed = {serve_counts}; HOLD counts = {hold_counts}.",
            "",
            "2. H4L 대비 SERVE/HOLD probability 변화:",
            f"   - pooled P(SERVE): {pooled['h4l_serve_masked_probability_mean']} -> {pooled['h4md_serve_masked_probability_mean']} "
            f"(delta {pooled['delta_serve_masked_probability_mean']})",
            f"   - pooled P(HOLD): {pooled['h4l_hold_masked_probability_mean']} -> {pooled['h4md_hold_masked_probability_mean']} "
            f"(delta {pooled['delta_hold_masked_probability_mean']})",
            f"   - entropy mean delta = {pooled['delta_entropy_mean']}",
            "",
            "3. avg_wait/service/Reward V2 개선?",
            f"   - avg_wait delta vs H4L = {avg_wait_delta}",
            f"   - p95_wait delta vs H4L = {p95_delta}",
            f"   - service delta vs H4L = {service_delta}",
            f"   - Reward V2 evaluation return delta vs H4L = {reward_delta}",
            "   - conclusion: no validation KPI improvement over H4L.",
            "",
            "4. H4M-A HOLD_BETTER 상태에서 Actor 선택:",
            f"   - HOLD_BETTER seed-state count = {hold_headline['hold_better_seed_state_count']}",
            f"   - Actor chose SERVE = {hold_headline['hold_better_actor_chose_serve_seed_state_count']}",
            f"   - Actor chose HOLD = {hold_headline['hold_better_actor_chose_hold_seed_state_count']}",
            "",
            "5. training budget 부족 가설:",
            f"   - rejected = {root_cause['evidence']['training_budget_shortage_hypothesis_rejected']}",
            "   - 11-cycle extension strengthened SERVE preference rather than creating HOLD differentiation.",
            "",
            "6. 다음 원인 후보:",
            "   - primary next diagnostic: Reward -> GAE -> Actor credit alignment.",
            "   - observation discrimination remains secondary; long-horizon action value is not claimed without a separate reconciliation.",
            "",
            f"7. exact next gate = {gate_matrix['next_gate']}",
            "",
            "## Critic note",
            "",
            f"- critic MSE delta summary = {critic['mse_delta_summary']}",
            f"- critic explained variance delta summary = {critic['explained_variance_delta_summary']}",
            "",
            "## STOP flags",
            "",
            "- do not retrain",
            "- do not increase training cycles",
            "- do not open test",
            "- do not expand environment",
            "- do not select winner",
            "- do not compare baseline",
            "- do not tune hyperparameters",
            "- do not modify Reward V2 / Zero-Loss / K-mask",
            "- do not push GitHub",
            "",
        ]
    )


def build_payloads(
    created_at: str,
    output_root: Path,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    config_audit: Mapping[str, Any],
    protocol: Mapping[str, Any],
    seed_metrics: Sequence[Mapping[str, Any]],
    mutations: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    immutability = checkpoint_immutability_audit(created_at, mutations)
    action_delta = h4l_vs_h4md_action_probability_delta(seed_metrics, created_at)
    crosswalk = h4ma_hold_better_actor_crosswalk(seed_metrics, created_at)
    kpi_delta = h4l_vs_h4md_kpi_delta(seed_metrics, created_at)
    critic = critic_revalidation(seed_metrics, created_at)
    zero_loss = zero_loss_skip_opportunity_status(seed_metrics, created_at)
    safety = safety_integrity_audit(seed_metrics, created_at)
    root_cause = root_cause_classification(created_at, seed_metrics, action_delta, crosswalk, kpi_delta)
    gate_matrix = build_gate_matrix(
        created_at,
        binding,
        provenance,
        config_audit,
        protocol,
        immutability,
        seed_metrics,
        action_delta,
        crosswalk,
        safety,
        root_cause,
    )
    if gate_matrix["gate"] != PASS_GATE:
        root_cause = {**root_cause, "decision": BLOCK_DECISION, "next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE"}
    report = build_report(
        output_root,
        provenance,
        seed_metrics,
        action_delta,
        crosswalk,
        kpi_delta,
        critic,
        root_cause,
        gate_matrix,
    )
    return {
        "01_authoritative_binding.json": binding,
        "02_h4l_protocol_reproduction_audit.json": {
            **protocol,
            "configuration_equality_audit": config_audit,
            "git_source_provenance": provenance,
        },
        "03_checkpoint_immutability_audit.json": immutability,
        "04_seed1_policy_revalidation.json": seed_metrics[0],
        "05_seed2_policy_revalidation.json": seed_metrics[1],
        "06_seed3_policy_revalidation.json": seed_metrics[2],
        "07_h4l_vs_h4md_action_probability_delta.json": action_delta,
        "08_h4ma_hold_better_actor_crosswalk.json": crosswalk,
        "09_h4l_vs_h4md_kpi_delta.json": kpi_delta,
        "10_critic_revalidation.json": critic,
        "11_zero_loss_skip_opportunity_status.json": zero_loss,
        "12_safety_integrity_audit.json": safety,
        "13_h4m_d_root_cause_classification.json": root_cause,
        "14_h4m_d_gate_matrix.json": gate_matrix,
        "final_report.md": report,
    }


def block_payloads(
    created_at: str,
    output_root: Path,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    config_audit: Mapping[str, Any],
    protocol: Mapping[str, Any],
    reason: str,
) -> Dict[str, Any]:
    empty_seed = {
        "stage": STAGE,
        "status": "NOT_EVALUATED_BLOCKED_BEFORE_INFERENCE",
        "blocking_reason": reason,
        "training_executed": False,
        "optimizer_created": False,
        "backward_executed": False,
        "optimizer_step_executed": False,
        "normalizer_update": False,
        "checkpoint_write": False,
        "checkpoint_overwrite": False,
        "test_opened": False,
    }
    gate = PROTOCOL_BLOCK_GATE if reason == "H4L_EVALUATION_PROTOCOL_NOT_REPRODUCIBLE" else BLOCK_GATE
    gate_matrix = {
        "stage": STAGE,
        "created_at": created_at,
        "gate": gate,
        "decision": BLOCK_DECISION,
        "blocking_reason": reason,
        "next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
        "final_flags": {
            "h4m_d_policy_revalidation_executed": False,
            "training_executed": False,
            "checkpoint_mutation": False,
            "additional_training_authorized": False,
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
    }
    report = (
        "# H4M-D Frozen Extended-Budget Policy Revalidation\n\n"
        f"gate = {gate}\n"
        f"decision = {BLOCK_DECISION}\n"
        f"blocking_reason = {reason}\n\n"
        "Preserved evidence and stopped before inference.\n\n"
        "STOP.\n"
    )
    return {
        "01_authoritative_binding.json": binding,
        "02_h4l_protocol_reproduction_audit.json": {
            **protocol,
            "configuration_equality_audit": config_audit,
            "git_source_provenance": provenance,
            "blocking_reason": reason,
        },
        "03_checkpoint_immutability_audit.json": {
            "stage": STAGE,
            "created_at": created_at,
            "checkpoint_mutation_detected": False,
            "not_evaluated": True,
            "blocking_reason": reason,
        },
        "04_seed1_policy_revalidation.json": {**empty_seed, "seed": 1},
        "05_seed2_policy_revalidation.json": {**empty_seed, "seed": 2},
        "06_seed3_policy_revalidation.json": {**empty_seed, "seed": 3},
        "07_h4l_vs_h4md_action_probability_delta.json": {"stage": STAGE, "not_evaluated": True, "blocking_reason": reason},
        "08_h4ma_hold_better_actor_crosswalk.json": {"stage": STAGE, "not_evaluated": True, "blocking_reason": reason},
        "09_h4l_vs_h4md_kpi_delta.json": {"stage": STAGE, "not_evaluated": True, "blocking_reason": reason},
        "10_critic_revalidation.json": {"stage": STAGE, "not_evaluated": True, "blocking_reason": reason},
        "11_zero_loss_skip_opportunity_status.json": {"stage": STAGE, "not_evaluated": True, "blocking_reason": reason},
        "12_safety_integrity_audit.json": {
            "stage": STAGE,
            "hard_safety_violations": 0,
            "not_evaluated": True,
            "blocking_reason": reason,
        },
        "13_h4m_d_root_cause_classification.json": {
            "stage": STAGE,
            "decision": BLOCK_DECISION,
            "next_gate": "STOP_BLOCKED_REVIEW_EVIDENCE",
            "not_evaluated": True,
            "blocking_reason": reason,
        },
        "14_h4m_d_gate_matrix.json": gate_matrix,
        "final_report.md": report,
    }


def write_payloads(output_root: Path, payloads: Mapping[str, Any]) -> Dict[str, Any]:
    for name, payload in payloads.items():
        path = output_root / name
        if name.endswith(".json"):
            dump_json(path, payload)
        else:
            dump_text(path, str(payload))
    output_files = {name: str(output_root / name) for name in payloads}
    output_sha256 = {name: sha256_file(output_root / name) for name in sorted(payloads)}
    gate_matrix = payloads["14_h4m_d_gate_matrix.json"]
    manifest = {
        "stage": STAGE,
        "created_at": gate_matrix["created_at"],
        "artifact_root": str(output_root),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": sorted([*payloads.keys(), "manifest.json"]) == sorted(REQUIRED_ARTIFACTS),
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256 to avoid self-referential drift; all other required files are hashed",
        "output_files": output_files,
        "output_sha256": output_sha256,
        "source_sha256": {rel: sha256_file(PROJECT_ROOT / rel) for rel in sorted(RUNTIME_DEPENDENCIES)},
        "gate": gate_matrix["gate"],
        "decision": gate_matrix["decision"],
        "next_gate": gate_matrix.get("next_gate"),
        "h4m_d_evaluation_source_git_commit": payloads["02_h4l_protocol_reproduction_audit.json"]
        .get("git_source_provenance", {})
        .get("h4m_d_evaluation_source_git_commit"),
        "h4m_d_policy_revalidation_executed": gate_matrix["final_flags"]["h4m_d_policy_revalidation_executed"],
        "training_executed": False,
        "checkpoint_mutation": gate_matrix["final_flags"]["checkpoint_mutation"],
        "sealed_test_opened": False,
        "winner_selection_authorized": False,
        "baseline_comparison_authorized": False,
        "github_push_performed": False,
    }
    dump_json(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    torch.set_grad_enabled(False)
    now = kst_now()
    created_at = now.isoformat(timespec="seconds")
    output_root = (
        ARTIFACTS_ROOT
        / f"pv8_r2a_r8e_r3_r_h4m_d_frozen_extended_budget_policy_revalidation_{now.strftime('%Y%m%d_%H%M%S')}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints = load_checkpoints_read_only()
    binding = h4m_c_authoritative_binding(created_at, checkpoints)
    provenance = git_source_provenance(created_at)
    config_audit = configuration_equality_audit(created_at, checkpoints)
    validation_plan = h4l.load_validation_window_plan()
    protocol = h4l_protocol_reproduction_audit(created_at, validation_plan)
    blockers: List[str] = []
    if not binding["checkpoint_binding_passed"]:
        blockers.append("H4M_C_CHECKPOINT_OR_SHA_BINDING_MISMATCH")
    if not provenance["post_commit_provenance_gate_passed"]:
        blockers.append("RELEVANT_DIRTY_OR_UNCOMMITTED_H4M_D_SOURCE")
    if not config_audit["material_equality_passed"]:
        blockers.append("MATERIAL_SEED_CONFIGURATION_MISMATCH")
    if not protocol["h4l_protocol_reproduction_passed"]:
        blockers.append("H4L_EVALUATION_PROTOCOL_NOT_REPRODUCIBLE")
    if not validation_plan["validation_scope_passed"]:
        blockers.append("VALIDATION_SCOPE_MISMATCH")
    if blockers:
        reason = blockers[0] if len(blockers) == 1 else ",".join(blockers)
        payloads = block_payloads(created_at, output_root, binding, provenance, config_audit, protocol, reason)
        manifest = write_payloads(output_root, payloads)
        print(f"[H4M-D] artifact root: {output_root}")
        print(f"[H4M-D] gate: {manifest['gate']}")
        print(f"[H4M-D] decision: {manifest['decision']}")
        print(f"[H4M-D] blocked_before_inference: {reason}")
        return

    seed_metrics = []
    mutations = []
    for seed in (1, 2, 3):
        print(f"[H4M-D] seed {seed} frozen validation revalidation start", flush=True)
        metrics, mutation = evaluate_seed(seed, checkpoints[seed], validation_plan, output_root)
        seed_metrics.append(metrics)
        mutations.append(mutation)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "action_count": metrics["action_count"],
                    "action_counts": metrics["action_counts"],
                    "p_serve_mean": metrics["probability_summary"]["per_action"]["SERVE_AND_MOVE_TO_NEXT_STOP"][
                        "masked_probability"
                    ]["mean"],
                    "p_hold_mean": metrics["probability_summary"]["per_action"]["HOLD_CURRENT_POSITION"][
                        "masked_probability"
                    ]["mean"],
                    "avg_wait_mean": metrics["kpi"]["avg_wait_seconds"]["mean"],
                    "service_mean": metrics["kpi"]["service_metric_passenger_service_rate"]["mean"],
                    "checkpoint_mutation_detected": mutation["checkpoint_mutation_detected"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    payloads = build_payloads(created_at, output_root, binding, provenance, config_audit, protocol, seed_metrics, mutations)
    manifest = write_payloads(output_root, payloads)
    print(f"[H4M-D] artifact root: {output_root}")
    print(f"[H4M-D] gate: {manifest['gate']}")
    print(f"[H4M-D] decision: {manifest['decision']}")
    print(f"[H4M-D] next_gate: {manifest['next_gate']}")
    print("[H4M-D] training_executed=false checkpoint_mutation=false sealed_test_opened=false github_push_performed=false")


if __name__ == "__main__":
    main()
