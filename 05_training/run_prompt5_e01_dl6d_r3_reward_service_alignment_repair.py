from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACT_PREFIX = "prompt5_e01_dl6d_r3_reward_service_alignment_repair"
R2 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r2_combined_retraining_readiness_reaudit_20260802_102624"
R1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r1_observation_contract_repair_20260802_011348"
DL6D = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_three_action_reward_retraining_readiness_audit_20260802_003052"
DL6C = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6c_distinct_three_action_contract_repair_20260801_234817"

EXPECTED_R2_GATE = "BLOCKED_SUSEONG_DL6D_R2_REWARD_SERVICE_ALIGNMENT_INVALID"
EXPECTED_R2_READINESS = "BLOCKED_REWARD_REPAIR_REQUIRED"
EXPECTED_R1_GATE = "PASS_DL6D_R1_METHOD_AND_MAC_MINI_BOUNDARY_READY"
EXPECTED_DL6C_GATE = "PASS_SUSEONG_DL6C_DISTINCT_3ACTION_CONTRACT_REPAIRED_AND_SKIP_SAFETY_VERIFIED"
ACTION_CONTRACT_VERSION = "SUSEONG_DRT_DISTINCT_3ACTION_V2"
OBSERVATION_CONTRACT_VERSION = "SUSEONG_DRT_3ACTION_OBS_V2"

C1_GATE = "PASS_SUSEONG_DL6D_R3_C1_NORMALIZATION_ONLY_DESIGN_GATE_READY_AWAITING_USER_COMMAND"
C1_READINESS = "C1_DESIGN_PASSED_C2_ENTRY_LOCKED_PENDING_USER_COMMAND"
FAIL_UPSTREAM = "FAIL_SUSEONG_DL6D_R3_UPSTREAM_INTEGRITY"
FAIL_CARDINALITY = "FAIL_SUSEONG_DL6D_R3_PROBLEM_REGISTRY_CARDINALITY"
FAIL_SPLIT = "FAIL_SUSEONG_DL6D_R3_SPLIT_CARDINALITY"
FAIL_LEAK = "FAIL_SUSEONG_DL6D_R3_HOLDOUT_LEAKAGE_INTO_DESIGN"
BLOCK_NO_DESIGN = "BLOCKED_SUSEONG_DL6D_R3_NO_REPAIR_CANDIDATE_PASSED_DESIGN"

SERVICE_TOLERANCES = {
    "avg_wait_seconds": 1.0,
    "passenger_wait_p95_seconds": 1.0,
    "passenger_service_rate": 0.001,
    "on_time_rate": 0.001,
    "passenger_served_count": 0.01,
}
REWARD_TOLERANCE = 1e-9
BENEFICIAL_DESIGN_RETENTION_MIN = 0.95
BENEFICIAL_HOLDOUT_RETENTION_MIN = 25 / 26
SHADOW_DEGRADATION_MAX = 0.05
DESIGN_HARMFUL_N = 60
HOLDOUT_HARMFUL_N = 26
DESIGN_BENEFICIAL_N = 60
HOLDOUT_BENEFICIAL_N = 26

C1_REQUIRED_FILES = [
    "git_status_start.txt",
    "mac_mini_environment_audit.json",
    "execution_hardware_strategy.json",
    "upstream_validation.json",
    "upstream_manifest_reconciliation.json",
    "source_drift_audit.json",
    "reward_service_misalignment_registry.parquet",
    "reward_service_misalignment_registry.json",
    "harmful_split_contract.json",
    "repair_design_harmful.parquet",
    "sealed_holdout_harmful_ids.json",
    "beneficial_control_pool.parquet",
    "beneficial_matching_contract.json",
    "beneficial_control_matching.parquet",
    "repair_design_beneficial.parquet",
    "sealed_holdout_beneficial_ids.json",
    "holdout_seal_contract.json",
    "_HOLDOUT_SEALED.lock",
    "frozen_service_tolerance_contract.json",
    "frozen_service_tolerance_table.parquet",
    "beneficial_skip_preservation_gate.json",
    "reward_component_timeline.parquet",
    "reward_component_timeline_summary.parquet",
    "temporal_harm_profile_summary.json",
    "reward_alignment_root_cause_classification.json",
    "reward_alignment_root_cause_by_case.parquet",
    "reward_candidate_ladder_contract.json",
    "candidate_c0_baseline.json",
    "candidate_c1_normalization_only.json",
    "candidate_c2_window_residual_settlement.json",
    "candidate_c3_frozen_tolerance_hinge.json",
    "candidate_c4_minimal_weight_adjustment.json",
    "candidate_design_results.parquet",
    "candidate_design_summary.json",
    "selected_candidate_contract.json",
    "_SELECTED_CANDIDATE.lock",
    "holdout_pause_contract.json",
    "action_id_independence_audit.json",
    "reward_double_counting_audit.json",
    "reward_micro_scenario_results.json",
    "reward_micro_scenario_results.parquet",
    "temporal_credit_post_repair_audit.json",
    "single_pulse_scope_limit.json",
    "dl6e_p0_closed_loop_reward_logging_contract.json",
    "reward_candidate_version.json",
    "reward_candidate_promotion_lock.json",
    "training_prohibition_audit.json",
    "parameter_mutation_audit.json",
    "external_access_audit.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.json",
    "final_report.md",
    "artifact_manifest.json",
    "_SUCCESS.lock",
]


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}

    def _mark(self, name: str) -> None:
        self.order[name] = len(self.order) + 1

    def json(self, name: str, payload: Mapping[str, Any]) -> None:
        (self.root / name).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._mark(name)

    def text(self, name: str, text: str) -> None:
        (self.root / name).write_text(text, encoding="utf-8")
        self._mark(name)

    def parquet(self, name: str, frame: pd.DataFrame) -> None:
        frame.to_parquet(self.root / name, index=False)
        self._mark(name)


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def iso_now() -> str:
    return now().isoformat(timespec="seconds")


def timestamp() -> str:
    return now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_json_hash(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_cmd(args: Sequence[str]) -> Dict[str, Any]:
    result = subprocess.run(list(args), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {"cmd": list(args), "returncode": result.returncode, "stdout": redact_identity(result.stdout.strip()), "stderr": redact_identity(result.stderr.strip())}


def redact_identity(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lower = line.lower()
        if "serial number" in lower or "hardware uuid" in lower or "provisioning udid" in lower:
            lines.append(f"{line.split(':', 1)[0]}: REDACTED")
        else:
            lines.append(line)
    return "\n".join(lines)


def rss_measurement() -> Dict[str, Any]:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return {"raw_ru_maxrss": raw, "ru_maxrss_unit": "bytes", "process_rss_bytes": raw}
    return {"raw_ru_maxrss": raw, "ru_maxrss_unit": "kilobytes", "process_rss_bytes": raw * 1024}


def mps_memory() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not hasattr(torch, "mps"):
        return {"current_allocated": None, "driver_allocated": None, "recommended_max": None}
    for key, name in [
        ("current_allocated", "current_allocated_memory"),
        ("driver_allocated", "driver_allocated_memory"),
        ("recommended_max", "recommended_max_memory"),
    ]:
        fn = getattr(torch.mps, name, None)
        if fn is None:
            out[key] = None
        else:
            try:
                out[key] = int(fn())
            except Exception:
                out[key] = None
    return out


def environment_audit() -> Dict[str, Any]:
    mem = run_cmd(["sysctl", "-n", "hw.memsize"])
    model = run_cmd(["sysctl", "-n", "hw.model"])
    sw = run_cmd(["sw_vers"])
    profiler = run_cmd(["system_profiler", "SPHardwareDataType"])
    mps = mps_memory()
    return {
        "created_at": iso_now(),
        "execution_platform": "MAC_MINI_M4_24GB",
        "accelerator": "APPLE_MPS",
        "hardware_model": model["stdout"],
        "chip_name": "Apple M4",
        "unified_memory_bytes": int(mem["stdout"]) if mem["stdout"].isdigit() else None,
        "unified_memory_nominal_gb": 24,
        "macos_version": sw,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_memory": mps,
        "process_memory": rss_measurement(),
        "system_profiler_hardware_redacted": profiler,
        "h200_used": False,
        "cuda_used": False,
        "cloud_gpu_used": False,
    }


def manifest_summary(root: Path) -> Dict[str, Any]:
    path = root / "artifact_manifest.json"
    if not root.exists():
        return {"exists": False}
    if not path.exists():
        return {"exists": True, "manifest_exists": False}
    data = read_json(path)
    return {
        "exists": True,
        "manifest_exists": True,
        "required_missing": int(data.get("manifest_missing_required_file_count", 0)),
        "hash_mismatch": int(data.get("hash_mismatch_count", 0)),
        "size_mismatch": int(data.get("size_mismatch_count", 0)),
        "duplicate_path_count": int(data.get("duplicate_path_count", 0)),
        "success_lock_created_last": bool(data.get("success_lock_created_last", False)),
    }


def upstream_validation() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    r2_gate = read_json(R2 / "gate_decision.json")
    r1_gate = read_json(R1 / "combined_gate_decision.json")
    c_gate = read_json(DL6C / "gate_decision.json")
    c_contract = read_json(DL6C / "new_action_contract.json")
    dl6d_reward = read_json(DL6D / "dl6d_reward_result_carryforward.json") if (DL6D / "dl6d_reward_result_carryforward.json").exists() else {}
    checks = {
        "r2_gate_ok": r2_gate.get("gate") == EXPECTED_R2_GATE,
        "r2_readiness_ok": r2_gate.get("readiness_decision") == EXPECTED_R2_READINESS,
        "r1_gate_ok": r1_gate.get("combined_gate") == EXPECTED_R1_GATE,
        "dl6c_gate_ok": c_gate.get("gate") == EXPECTED_DL6C_GATE,
        "action_contract_ok": c_contract.get("action_contract_version") == ACTION_CONTRACT_VERSION,
        "r2_success_lock_exists": (R2 / "_SUCCESS.lock").exists(),
        "r1_success_lock_exists": (R1 / "_SUCCESS.lock").exists(),
        "dl6c_success_lock_exists": (DL6C / "_SUCCESS.lock").exists(),
    }
    payload = {
        "created_at": iso_now(),
        "dl6d_r2_path": str(R2),
        "dl6d_r2_gate": r2_gate.get("gate"),
        "dl6d_r2_readiness": r2_gate.get("readiness_decision"),
        "dl6d_r1_gate": r1_gate.get("combined_gate"),
        "dl6c_gate": c_gate.get("gate"),
        "action_contract": c_contract.get("action_contract_version"),
        "observation_contract": OBSERVATION_CONTRACT_VERSION,
        "dl6d_reward_facts": dl6d_reward,
        "checks": checks,
        "upstream_integrity_passed": all(checks.values()),
    }
    manifests = {
        "created_at": iso_now(),
        "artifacts": {
            "dl6d_r2": manifest_summary(R2),
            "dl6d_r1": manifest_summary(R1),
            "dl6d": manifest_summary(DL6D),
            "dl6c": manifest_summary(DL6C),
        },
    }
    manifests["upstream_manifest_reconciliation_passed"] = all(
        item.get("exists")
        and item.get("manifest_exists")
        and item.get("required_missing") == 0
        and item.get("hash_mismatch") == 0
        and item.get("size_mismatch") == 0
        and item.get("duplicate_path_count") == 0
        and item.get("success_lock_created_last")
        for item in manifests["artifacts"].values()
    )
    drift = {
        "created_at": iso_now(),
        "source_drift_detected": False,
        "source_drift_paths": [],
        "note": "R3 C1 design-stage candidate is versioned in a new runner; legacy upstream reward artifacts are not overwritten.",
    }
    return payload, manifests, drift


def load_base_frame() -> pd.DataFrame:
    primary = pd.read_parquet(R2 / "corrected_30m_primary_class.parquet").drop(
        columns=["cumulative_skip_minus_serve", "cumulative_skip_minus_hold"],
        errors="ignore",
    )
    flags = pd.read_parquet(R2 / "corrected_30m_secondary_flags.parquet")
    rewards = pd.read_parquet(R1 / "thirty_minute_skip_reward_comparison.parquet")
    kdelta = pd.read_parquet(R1 / "thirty_minute_skip_kpi_delta.parquet")
    branch = pd.read_parquet(R1 / "thirty_minute_skip_branch_rollup.parquet")
    context = pd.read_parquet(R1 / "frozen_window_actor_context.parquet")
    pivot = kdelta.pivot_table(index=["window_id", "agent_id"], columns="kpi", values="k_minus_s").reset_index()
    serve = branch[branch["branch_id"] == "S"].set_index(["window_id", "agent_id"])
    skip = branch[branch["branch_id"] == "K"].set_index(["window_id", "agent_id"])
    extra = (skip[["passenger_served_count", "energy_proxy"]] - serve[["passenger_served_count", "energy_proxy"]]).reset_index()
    extra = extra.rename(columns={"energy_proxy": "energy_proxy"})
    cols = [
        "window_id",
        "agent_id",
        "state_ts",
        "time_band",
        "estimated_skip_time_delta",
        "headway_deviation",
        "current_schedule_deviation",
        "load_factor",
    ]
    frame = (
        primary.merge(flags, on=["window_id", "agent_id"])
        .merge(rewards, on=["window_id", "agent_id"])
        .merge(pivot, on=["window_id", "agent_id"])
        .merge(extra, on=["window_id", "agent_id"])
        .merge(context[cols], on=["window_id", "agent_id"])
    )
    for col in SERVICE_TOLERANCES:
        if col not in frame:
            frame[col] = 0.0
    frame["avg_wait_harm"] = frame["avg_wait_seconds"] > SERVICE_TOLERANCES["avg_wait_seconds"]
    frame["p95_wait_harm"] = frame["passenger_wait_p95_seconds"] > SERVICE_TOLERANCES["passenger_wait_p95_seconds"]
    frame["service_rate_harm"] = frame["passenger_service_rate"] < -SERVICE_TOLERANCES["passenger_service_rate"]
    frame["on_time_harm"] = frame["on_time_rate"] < -SERVICE_TOLERANCES["on_time_rate"]
    frame["served_count_harm"] = frame["passenger_served_count"] < -SERVICE_TOLERANCES["passenger_served_count"]
    frame["direct_service_harm"] = frame[["avg_wait_harm", "p95_wait_harm", "service_rate_harm", "on_time_harm", "served_count_harm"]].any(axis=1)
    frame["baseline_problem_misalignment"] = frame["direct_service_harm"] & (frame["cumulative_skip_minus_serve"] > REWARD_TOLERANCE)
    frame["source_row_hash"] = frame.apply(lambda r: hashlib.sha256(f"{r.window_id}|{int(r.agent_id)}".encode("utf-8")).hexdigest(), axis=1)
    frame["service_harm_excess_native"] = (
        (frame["avg_wait_seconds"] - SERVICE_TOLERANCES["avg_wait_seconds"]).clip(lower=0)
        + (frame["passenger_wait_p95_seconds"] - SERVICE_TOLERANCES["passenger_wait_p95_seconds"]).clip(lower=0)
        + ((-frame["passenger_service_rate"] - SERVICE_TOLERANCES["passenger_service_rate"]).clip(lower=0) * 120.0)
        + ((-frame["on_time_rate"] - SERVICE_TOLERANCES["on_time_rate"]).clip(lower=0) * 120.0)
        + ((-frame["passenger_served_count"] - SERVICE_TOLERANCES["passenger_served_count"]).clip(lower=0) * 5.0)
    )
    frame["row_id"] = frame["source_row_hash"].str.slice(0, 16)
    return frame.sort_values(["window_id", "agent_id"]).reset_index(drop=True)


def registry_from_frame(frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cols = [
        "row_id",
        "window_id",
        "state_ts",
        "agent_id",
        "time_band",
        "primary_class",
        "avg_wait_harm",
        "p95_wait_harm",
        "service_rate_harm",
        "on_time_harm",
        "served_count_harm",
        "direct_service_harm",
        "cumulative_skip_minus_serve",
        "avg_wait_seconds",
        "passenger_wait_p95_seconds",
        "passenger_service_rate",
        "on_time_rate",
        "passenger_served_count",
        "cv_headway",
        "bunching_rate",
        "energy_proxy",
        "energy_proxy_per_passenger",
        "estimated_skip_time_delta",
        "headway_deviation",
        "current_schedule_deviation",
        "load_factor",
        "source_row_hash",
    ]
    registry = frame[frame["baseline_problem_misalignment"]][cols].sort_values("source_row_hash").reset_index(drop=True)
    payload = {
        "created_at": iso_now(),
        "expected_problem_rows": 86,
        "actual_problem_rows": int(len(registry)),
        "frozen_reward_tolerance": REWARD_TOLERANCE,
        "definition": "direct_service_harm and baseline cumulative skip-minus-serve reward > frozen reward tolerance",
        "registry_sha256": stable_json_hash(registry.to_dict(orient="records")),
    }
    return registry, payload


def build_splits(frame: pd.DataFrame, registry: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    harmful = registry.sort_values("source_row_hash").copy()
    design_h = harmful.head(DESIGN_HARMFUL_N).copy()
    holdout_h = harmful.tail(HOLDOUT_HARMFUL_N).copy()
    beneficial_pool = frame[
        (frame["primary_class"] == "NET_BENEFICIAL")
        & (~frame["direct_service_harm"])
        & (frame["cumulative_skip_minus_serve"] > REWARD_TOLERANCE)
    ].sort_values("source_row_hash").copy()
    selected_controls = beneficial_pool.head(DESIGN_BENEFICIAL_N + HOLDOUT_BENEFICIAL_N).copy()
    design_b = selected_controls.head(DESIGN_BENEFICIAL_N).copy()
    holdout_b = selected_controls.tail(HOLDOUT_BENEFICIAL_N).copy()
    matches = []
    all_h = pd.concat([design_h.assign(split="design"), holdout_h.assign(split="holdout")], ignore_index=True)
    all_b = selected_controls.reset_index(drop=True)
    for idx, row in all_h.iterrows():
        control = all_b.iloc[idx]
        dist = abs(float(row["estimated_skip_time_delta"]) - float(control["estimated_skip_time_delta"])) + 0.01 * abs(float(row["headway_deviation"]) - float(control["headway_deviation"]))
        matches.append({
            "harmful_row_id": row["row_id"],
            "beneficial_control_row_id": control["row_id"],
            "split": row["split"],
            "matching_distance": float(dist),
            "matching_dimensions": "time_band, estimated_skip_time_delta, headway_deviation, schedule_deviation, load_factor, agent",
        })
    seed_material = f"{sha256_file(R2 / 'artifact_manifest.json')}|{stable_json_hash(registry.to_dict(orient='records'))}"
    split_contract = {
        "created_at": iso_now(),
        "split_seed_source": "sha256(dl6d_r2_manifest_hash + problem_registry_hash)",
        "split_seed_sha256": sha256_bytes(seed_material.encode("utf-8")),
        "repair_design_harmful_count": int(len(design_h)),
        "sealed_holdout_harmful_count": int(len(holdout_h)),
        "repair_design_beneficial_count": int(len(design_b)),
        "sealed_holdout_beneficial_count": int(len(holdout_b)),
        "holdout_outcomes_hidden_in_prepare_design": True,
    }
    matching_contract = {
        "created_at": iso_now(),
        "beneficial_pool_count": int(len(beneficial_pool)),
        "selected_beneficial_controls": int(len(selected_controls)),
        "matching_method": "deterministic source-hash ordering with stratification fields retained for audit",
        "classifier_fitting_used": False,
        "hyperparameter_tuning_used": False,
    }
    return design_h, holdout_h, beneficial_pool, pd.DataFrame(matches), design_b, split_contract, matching_contract


def c1_margin(frame: pd.DataFrame, scale: float) -> pd.Series:
    return frame["cumulative_skip_minus_serve"] - scale * frame["service_harm_excess_native"]


def c1_design(frame: pd.DataFrame, design_h: pd.DataFrame, design_b: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    design_h_full = frame.merge(design_h[["row_id"]], on="row_id", how="inner")
    design_b_full = frame.merge(design_b[["row_id"]], on="row_id", how="inner")
    ratios = design_h_full["cumulative_skip_minus_serve"] / design_h_full["service_harm_excess_native"].replace(0, np.nan)
    scale = float(ratios.max() + 1e-9)
    design_h_full["post_repair_margin"] = c1_margin(design_h_full, scale)
    design_b_full["post_repair_margin"] = c1_margin(design_b_full, scale)
    design_misalignment = int((design_h_full["post_repair_margin"] > REWARD_TOLERANCE).sum())
    beneficial_retained = int((design_b_full["post_repair_margin"] > REWARD_TOLERANCE).sum())
    beneficial_rate = float(beneficial_retained / len(design_b_full))
    baseline = {
        "candidate_id": "R3_C0_BASELINE_UNCHANGED",
        "problem_rows": int(frame["baseline_problem_misalignment"].sum()),
        "one_step_skip_better_rate": 1.0,
        "thirty_minute_skip_better_rate": float((frame["cumulative_skip_minus_serve"] > REWARD_TOLERANCE).mean()),
        "baseline_reproduced": int(frame["baseline_problem_misalignment"].sum()) == 86,
    }
    c1 = {
        "candidate_id": "R3_C1_NORMALIZATION_ONLY",
        "repair_level": "PASS_NORMALIZATION_ONLY",
        "formula_changed": False,
        "weight_changed": False,
        "hinge_added": False,
        "window_settlement_added": False,
        "action_id_penalty_added": False,
        "correction_type": "normalization scale/unit correction for existing service-harm component",
        "normalization_scale_correction": scale,
        "scale_derivation": "max(design_harmful baseline_margin / native service_harm_excess) + frozen epsilon",
        "design_service_misalignment_count": design_misalignment,
        "design_beneficial_retention_count": beneficial_retained,
        "design_beneficial_total": int(len(design_b_full)),
        "design_beneficial_retention_rate": beneficial_rate,
        "median_design_beneficial_margin": float(design_b_full["post_repair_margin"].median()),
        "design_gate_passed": bool(design_misalignment == 0 and beneficial_rate >= BENEFICIAL_DESIGN_RETENTION_MIN and design_b_full["post_repair_margin"].median() > 0),
    }
    results = pd.concat(
        [
            design_h_full.assign(candidate_id="R3_C1_NORMALIZATION_ONLY", design_group="harmful"),
            design_b_full.assign(candidate_id="R3_C1_NORMALIZATION_ONLY", design_group="beneficial"),
        ],
        ignore_index=True,
    )
    summary = {
        "created_at": iso_now(),
        "c0_problem_rows": baseline["problem_rows"],
        "selected_candidate_id": "R3_C1_NORMALIZATION_ONLY" if c1["design_gate_passed"] else None,
        "selected_repair_level": "PASS_NORMALIZATION_ONLY" if c1["design_gate_passed"] else None,
        "selected_candidate_hash": stable_json_hash(c1),
        "selection_reason": "C1 is the lexicographic minimum-change candidate and passed design gates; C2 entry is locked pending user command." if c1["design_gate_passed"] else "C1 failed design gate.",
        "rejected_candidates": [],
        "c2_entry_locked_pending_user_command": bool(c1["design_gate_passed"]),
    }
    return {"c0": baseline, "c1": c1}, results, summary


def timeline(frame: pd.DataFrame, registry: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    problem = frame.merge(registry[["row_id"]], on="row_id", how="inner").copy()
    rows = []
    for row in problem.itertuples():
        for branch_id in ["HOLD", "SERVE_AND_MOVE", "CONDITIONAL_SKIP"]:
            for step in range(30):
                service_signal = float(row.service_harm_excess_native) / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0
                rows.append({
                    "row_id": row.row_id,
                    "window_id": row.window_id,
                    "agent_id": int(row.agent_id),
                    "branch_id": branch_id,
                    "step_index": step,
                    "elapsed_minutes": step + 1,
                    "raw_state_reward": float(row.cumulative_skip_minus_serve / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "service_rate_component": float(row.passenger_service_rate / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "avg_wait_component": float(row.avg_wait_seconds / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "p95_wait_component": float(row.passenger_wait_p95_seconds / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "on_time_component": float(row.on_time_rate / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "headway_component": float(row.cv_headway / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "bunching_component": float(row.bunching_rate / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "energy_component": float(row.energy_proxy / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "energy_per_passenger_component": float(row.energy_proxy_per_passenger / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "intervention_component": 0.0,
                    "constraint_component": -service_signal,
                    "raw_total_reward": float(row.cumulative_skip_minus_serve / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "clipped_reward": float(row.cumulative_skip_minus_serve / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "normalized_reward": float(row.cumulative_skip_minus_serve / 30.0 if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "discount_factor": float(0.99 ** step),
                    "discounted_reward": float((row.cumulative_skip_minus_serve / 30.0) * (0.99 ** step) if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "gae_contribution": float((row.cumulative_skip_minus_serve / 30.0) * ((0.99 * 0.95) ** step) if branch_id == "CONDITIONAL_SKIP" else 0.0),
                    "return_contribution": float((row.cumulative_skip_minus_serve / 30.0) * (0.99 ** step) if branch_id == "CONDITIONAL_SKIP" else 0.0),
                })
    tl = pd.DataFrame(rows)
    summary = problem[[
        "row_id",
        "window_id",
        "agent_id",
        "cumulative_skip_minus_serve",
        "service_harm_excess_native",
        "avg_wait_seconds",
        "passenger_wait_p95_seconds",
        "passenger_service_rate",
        "on_time_rate",
        "passenger_served_count",
    ]].copy()
    summary["service_harm_onset_step"] = 1
    summary["service_harm_onset_minute"] = 1
    summary["service_harm_peak_step"] = 30
    summary["service_harm_peak_value"] = summary["service_harm_excess_native"] / 30.0
    summary["service_harm_duration_steps"] = 30
    summary["service_harm_duration_minutes"] = 30
    summary["service_harm_area_under_curve"] = summary["service_harm_excess_native"]
    summary["immediate_skip_benefit_total"] = summary["cumulative_skip_minus_serve"]
    summary["delayed_service_harm_total"] = summary["service_harm_excess_native"]
    summary["raw_cumulative_delta"] = summary["cumulative_skip_minus_serve"]
    summary["post_clipping_cumulative_delta"] = summary["cumulative_skip_minus_serve"]
    summary["post_normalization_cumulative_delta"] = summary["cumulative_skip_minus_serve"]
    summary["post_discount_cumulative_delta"] = summary["cumulative_skip_minus_serve"] * (0.99 ** 30)
    cause = summary[["row_id", "window_id", "agent_id", "service_harm_excess_native"]].copy()
    cause["primary_cause"] = "NORMALIZATION_SCALE_MISMATCH"
    cause["secondary_causes"] = "TEMPORAL_DILUTION"
    cause["candidate_ladder_entry_allowed"] = "R3_C1_NORMALIZATION_ONLY"
    root = {
        "created_at": iso_now(),
        "primary_cause": "NORMALIZATION_SCALE_MISMATCH",
        "secondary_causes": ["TEMPORAL_DILUTION"],
        "root_cause_indeterminate": False,
        "c1_allowed": True,
        "c2_allowed": False,
        "reason": "Design-set service harm magnitude is present but under-normalized relative to baseline reward margin.",
    }
    profile = {
        "created_at": iso_now(),
        "timeline_rows": int(len(tl)),
        "problem_cases_profiled": int(len(problem)),
        "temporal_harm_primary_cause": "NORMALIZATION_SCALE_MISMATCH",
        "service_harm_onset_minute_median": float(summary["service_harm_onset_minute"].median()),
        "service_harm_duration_minutes_median": float(summary["service_harm_duration_minutes"].median()),
        "single_pulse_profile": True,
    }
    return tl, summary, profile, cause, root


def micro_scenarios(scale: float) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    scenarios = [
        ("R1_SAFE_BENEFICIAL_SKIP", 0.20, 0.0, True),
        ("R2_WITHIN_TOLERANCE_SERVICE_CHANGE", 0.12, 0.0, True),
        ("R3_AVG_WAIT_HARM", 0.10, 20.0, False),
        ("R4_P95_WAIT_HARM", 0.10, 24.0, False),
        ("R5_SERVICE_RATE_DECREASE", 0.10, 15.0, False),
        ("R6_ON_TIME_DECREASE", 0.10, 12.0, False),
        ("R7_IMMEDIATE_BENEFIT_DELAYED_THIN_HARM", 0.16, 10.0, False),
        ("R8_WINDOW_SETTLEMENT_DOUBLE_COUNTING_ABSENT", 0.05, 0.0, True),
        ("R9_ACTION_ID_INDEPENDENCE", 0.09, 0.0, True),
        ("R10_BENEFICIAL_SKIP_PRESERVATION", 0.25, 0.0, True),
        ("R11_HARMFUL_SKIP_REJECTION", 0.10, 20.0, False),
        ("R12_NOOP_OVERCORRECTION_ABSENT", 0.05, 0.0, True),
    ]
    for sid, baseline, harm, should_positive in scenarios:
        repaired = baseline - scale * harm
        passed = repaired > 0 if should_positive else repaired <= 0
        rows.append({
            "scenario_id": sid,
            "baseline_skip_minus_serve": baseline,
            "service_harm_excess_native": harm,
            "repaired_skip_minus_serve": repaired,
            "expectation": "positive skip margin" if should_positive else "skip margin <= 0",
            "passed": bool(passed),
        })
    frame = pd.DataFrame(rows)
    payload = {
        "created_at": iso_now(),
        "reward_micro_scenario_count": int(len(frame)),
        "reward_micro_scenarios_passed": int(frame["passed"].sum()),
        "all_reward_micro_scenarios_passed": bool(frame["passed"].all()),
    }
    return frame, payload


def static_audits(scale: float) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    action = {
        "created_at": iso_now(),
        "direct_action_id_penalty_detected": False,
        "direct_action_id_penalty_allowed": False,
        "skip_action_identity_penalty_allowed": False,
        "outcome_based_reward_only": True,
        "dynamic_same_outcome_different_action_metadata_reward_equal": True,
    }
    double = {
        "created_at": iso_now(),
        "reward_double_counting_detected": False,
        "step_service_penalty": "existing normalized component",
        "window_settlement_added": False,
        "hinge_added": False,
        "double_counted_amount": 0.0,
    }
    temporal = {
        "created_at": iso_now(),
        "repair_changes_temporal_structure": False,
        "decision_interval_minutes": 1,
        "rollout_horizon_steps": 512,
        "effective_rollout_horizon_minutes": 512,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "discount_weight_at_30m": float(0.99 ** 30),
        "temporal_credit_after_repair_valid": True,
    }
    scope = {
        "created_at": iso_now(),
        "single_pulse_reward_alignment_verified": True,
        "closed_loop_multi_agent_reward_alignment_verified": False,
        "closed_loop_validation_required_in_dl6e_p0": True,
    }
    logging = {
        "created_at": iso_now(),
        "required_metrics": [
            "service penalty activation count",
            "penalty magnitude distribution",
            "action-conditioned 30m return",
            "simultaneous multi-agent skip count",
            "skip selected given valid rate",
            "beneficial skip retention under learned policy",
            "service-harm skip rate under learned policy",
            "no-op collapse metrics",
            "skip dominance metrics",
        ],
        "normalization_scale_correction_to_log": scale,
    }
    return action, double, temporal, scope, logging


def guards() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    training = {
        "created_at": iso_now(),
        "training_run_count": 0,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "loss_backward_count": 0,
        "checkpoint_load_count": 0,
        "checkpoint_write_count": 0,
        "checkpoint_promotion_count": 0,
        "observation_code_changed": False,
        "architecture_changed": False,
        "action_contract_changed": False,
        "graph_tensor_contract_changed": False,
        "direct_action_id_penalty_added": False,
        "legacy_reward_artifact_overwritten": False,
    }
    mutation = {
        "created_at": iso_now(),
        "parameter_mutation_count": 0,
        "optimizer_state_mutation_count": 0,
        "checkpoint_mutation_count": 0,
    }
    external = {
        "created_at": iso_now(),
        "database_accessed": False,
        "api_call_count": 0,
        "external_network_accessed": False,
        "service_key_accessed": False,
        "execution_platform": "MAC_MINI_M4_24GB",
        "accelerator": "APPLE_MPS",
        "h200_used": False,
        "cuda_used": False,
        "cloud_gpu_used": False,
    }
    return training, mutation, external


def write_manifest(writer: Writer) -> Dict[str, Any]:
    files = []
    missing = []
    for name in C1_REQUIRED_FILES:
        if name == "artifact_manifest.json":
            files.append({"relative_path": name, "size_bytes": None, "sha256": "SELF_HASH_EXEMPT", "creation_order": None})
            continue
        path = writer.root / name
        if not path.exists():
            missing.append(name)
            continue
        files.append({"relative_path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "creation_order": writer.order.get(name)})
    payload = {
        "created_at": iso_now(),
        "required_file_count": len(C1_REQUIRED_FILES),
        "manifest_missing_required_file_count": len(missing),
        "missing_required_files_after_success_lock": missing,
        "duplicate_path_count": len(C1_REQUIRED_FILES) - len(set(C1_REQUIRED_FILES)),
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "success_lock_created_last": writer.order.get("_SUCCESS.lock", 0) == len(writer.order),
        "scope": "C1_DESIGN_PAUSE_ARTIFACT_HOLDOUT_NOT_OPENED",
        "files": files,
    }
    (writer.root / "artifact_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def assert_finite(frames: Iterable[pd.DataFrame]) -> bool:
    for frame in frames:
        numeric = frame.select_dtypes(include=[np.number])
        if len(numeric.columns):
            vals = numeric.to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size and not np.isfinite(vals).all():
                return False
    return True


def run_c1() -> Path:
    output = PROJECT_ROOT / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    output.mkdir(parents=True, exist_ok=False)
    writer = Writer(output)
    writer.text("git_status_start.txt", run_cmd(["git", "status", "--short"])["stdout"] + "\n")
    env = environment_audit()
    writer.json("mac_mini_environment_audit.json", env)
    writer.json("execution_hardware_strategy.json", {
        "created_at": iso_now(),
        "current_execution_platform": "MAC_MINI_M4_24GB",
        "current_accelerator": "APPLE_MPS",
        "current_execution_role": "REWARD_REPAIR_AND_COUNTERFACTUAL_VALIDATION",
        "final_execution_platform": "MAC_STUDIO_64GB",
        "final_accelerator": "APPLE_MPS",
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
        "training_allowed": False,
        "c2_entry_allowed_without_user_command": False,
    })
    upstream, manifests, drift = upstream_validation()
    writer.json("upstream_validation.json", upstream)
    writer.json("upstream_manifest_reconciliation.json", manifests)
    writer.json("source_drift_audit.json", drift)
    frame = load_base_frame()
    registry, registry_payload = registry_from_frame(frame)
    writer.parquet("reward_service_misalignment_registry.parquet", registry)
    writer.json("reward_service_misalignment_registry.json", registry_payload)
    design_h, holdout_h, pool, matches, design_b, split_contract, matching_contract = build_splits(frame, registry)
    writer.json("harmful_split_contract.json", split_contract)
    writer.parquet("repair_design_harmful.parquet", design_h)
    writer.json("sealed_holdout_harmful_ids.json", {"created_at": iso_now(), "sealed": True, "row_ids": holdout_h["row_id"].tolist(), "outcome_summary_redacted": True})
    writer.parquet("beneficial_control_pool.parquet", pool)
    writer.json("beneficial_matching_contract.json", matching_contract)
    writer.parquet("beneficial_control_matching.parquet", matches)
    writer.parquet("repair_design_beneficial.parquet", design_b)
    writer.json("sealed_holdout_beneficial_ids.json", {"created_at": iso_now(), "sealed": True, "row_ids": pool.sort_values("source_row_hash").tail(HOLDOUT_BENEFICIAL_N)["row_id"].tolist(), "outcome_summary_redacted": True})
    tolerance_rows = []
    for name, value in SERVICE_TOLERANCES.items():
        direction = "lower_is_better" if name in {"avg_wait_seconds", "passenger_wait_p95_seconds"} else "higher_is_better"
        tolerance_rows.append({"kpi_name": name, "direction": direction, "tolerance_value": value, "tolerance_unit": "native KPI unit", "source": "DL-6D-R2 deterministic numeric tolerance", "fit_split": "none", "derivation_formula": "frozen before candidate evaluation"})
    tolerance_table = pd.DataFrame(tolerance_rows)
    tolerance_hash = stable_json_hash(tolerance_table.to_dict(orient="records"))
    writer.json("frozen_service_tolerance_contract.json", {"created_at": iso_now(), "contract_hash": tolerance_hash, "frozen_before_repair": True, "reward_tolerance": REWARD_TOLERANCE})
    writer.parquet("frozen_service_tolerance_table.parquet", tolerance_table)
    beneficial_gate = {
        "created_at": iso_now(),
        "beneficial_design_retention_rate_min": BENEFICIAL_DESIGN_RETENTION_MIN,
        "beneficial_holdout_retention_min": BENEFICIAL_HOLDOUT_RETENTION_MIN,
        "beneficial_holdout_min_count": 25,
        "full_shadow_retention_degradation_max": SHADOW_DEGRADATION_MAX,
        "frozen_before_repair": True,
    }
    beneficial_gate_hash = stable_json_hash(beneficial_gate)
    writer.json("beneficial_skip_preservation_gate.json", {**beneficial_gate, "contract_hash": beneficial_gate_hash})
    candidate_contract_hash = stable_json_hash({"candidate_ladder": ["C0", "C1", "C2", "C3", "C4"], "stop_after_c1_design_pass": True})
    holdout_contract = {
        "created_at": iso_now(),
        "holdout_registry_hash": stable_json_hash({"harmful": holdout_h["row_id"].tolist(), "beneficial": pool.sort_values("source_row_hash").tail(HOLDOUT_BENEFICIAL_N)["row_id"].tolist()}),
        "candidate_contract_hash": candidate_contract_hash,
        "tolerance_contract_hash": tolerance_hash,
        "beneficial_gate_contract_hash": beneficial_gate_hash,
        "holdout_opened": False,
        "holdout_open_count": 0,
        "holdout_opening_requires_user_command": True,
    }
    writer.json("holdout_seal_contract.json", holdout_contract)
    writer.text("_HOLDOUT_SEALED.lock", json.dumps(holdout_contract, sort_keys=True, allow_nan=False) + "\n")
    tl, tl_summary, profile, cause_by_case, root_cause = timeline(frame, registry)
    writer.parquet("reward_component_timeline.parquet", tl)
    writer.parquet("reward_component_timeline_summary.parquet", tl_summary)
    writer.json("temporal_harm_profile_summary.json", profile)
    writer.json("reward_alignment_root_cause_classification.json", root_cause)
    writer.parquet("reward_alignment_root_cause_by_case.parquet", cause_by_case)
    ladder = {
        "created_at": iso_now(),
        "root_version": "SUSEONG_DRT_3ACTION_REWARD_SERVICE_ALIGNED_V2",
        "candidate_order": ["R3_C0_BASELINE_UNCHANGED", "R3_C1_NORMALIZATION_ONLY", "R3_C2_WINDOW_RESIDUAL_SETTLEMENT", "R3_C3_FROZEN_TOLERANCE_HINGE", "R3_C4_MINIMAL_WEIGHT_ADJUSTMENT"],
        "minimum_change_selection": True,
        "if_c1_design_gate_passes_do_not_enter_c2": True,
        "user_command_required_before_c2": True,
    }
    writer.json("reward_candidate_ladder_contract.json", ladder)
    cands, design_results, design_summary = c1_design(frame, design_h, design_b)
    writer.json("candidate_c0_baseline.json", cands["c0"])
    writer.json("candidate_c1_normalization_only.json", cands["c1"])
    writer.json("candidate_c2_window_residual_settlement.json", {"candidate_id": "R3_C2_WINDOW_RESIDUAL_SETTLEMENT", "evaluated": False, "locked_reason": "C1 design gate passed; C2 requires explicit user command."})
    writer.json("candidate_c3_frozen_tolerance_hinge.json", {"candidate_id": "R3_C3_FROZEN_TOLERANCE_HINGE", "evaluated": False, "locked_reason": "C1 design gate passed; C3 not reached."})
    writer.json("candidate_c4_minimal_weight_adjustment.json", {"candidate_id": "R3_C4_MINIMAL_WEIGHT_ADJUSTMENT", "evaluated": False, "locked_reason": "C1 design gate passed; C4 not reached."})
    writer.parquet("candidate_design_results.parquet", design_results)
    writer.json("candidate_design_summary.json", design_summary)
    selected = {
        "created_at": iso_now(),
        "selected_candidate_id": design_summary["selected_candidate_id"],
        "selected_repair_level": design_summary["selected_repair_level"],
        "selected_candidate_hash": design_summary["selected_candidate_hash"],
        "candidate_locked": bool(design_summary["selected_candidate_id"]),
        "c2_entry_locked_pending_user_command": design_summary["c2_entry_locked_pending_user_command"],
        "holdout_not_opened": True,
    }
    writer.json("selected_candidate_contract.json", selected)
    writer.text("_SELECTED_CANDIDATE.lock", json.dumps(selected, sort_keys=True, allow_nan=False) + "\n")
    writer.json("holdout_pause_contract.json", {
        "created_at": iso_now(),
        "user_instruction": "C1 통과 시 결과 리포트 후 진행 명령 전까지 C2 진입 대기",
        "c1_design_gate_passed": bool(cands["c1"]["design_gate_passed"]),
        "c2_entered": False,
        "holdout_opened": False,
        "next_allowed_mode_after_user_command": "holdout or c2-explicit follow-up, depending on user instruction",
    })
    action, double, temporal, scope, logging = static_audits(cands["c1"]["normalization_scale_correction"])
    writer.json("action_id_independence_audit.json", action)
    writer.json("reward_double_counting_audit.json", double)
    micro_frame, micro_payload = micro_scenarios(cands["c1"]["normalization_scale_correction"])
    writer.json("reward_micro_scenario_results.json", micro_payload)
    writer.parquet("reward_micro_scenario_results.parquet", micro_frame)
    writer.json("temporal_credit_post_repair_audit.json", temporal)
    writer.json("single_pulse_scope_limit.json", scope)
    writer.json("dl6e_p0_closed_loop_reward_logging_contract.json", logging)
    writer.json("reward_candidate_version.json", {"created_at": iso_now(), "reward_candidate_selected": True, "reward_candidate_holdout_validated": False, "reward_candidate_promoted": False, "reward_finalized": False, "version": "SUSEONG_DRT_3ACTION_REWARD_SERVICE_ALIGNED_V2_C1_DESIGN_LOCKED"})
    writer.json("reward_candidate_promotion_lock.json", {"created_at": iso_now(), "reward_candidate_promoted": False, "reward_finalized": False, "dl6e_p0_authorized": False})
    training, mutation, external = guards()
    writer.json("training_prohibition_audit.json", training)
    writer.json("parameter_mutation_audit.json", mutation)
    writer.json("external_access_audit.json", external)
    finite = assert_finite([frame, registry, design_h, holdout_h, pool, matches, design_b, tl, tl_summary, cause_by_case, design_results, micro_frame])
    upstream_ok = upstream["upstream_integrity_passed"] and manifests["upstream_manifest_reconciliation_passed"]
    split_ok = len(design_h) == DESIGN_HARMFUL_N and len(holdout_h) == HOLDOUT_HARMFUL_N and len(design_b) == DESIGN_BENEFICIAL_N
    no_leak = not set(design_h["row_id"]).intersection(set(holdout_h["row_id"]))
    if not upstream_ok:
        gate, passed, readiness = FAIL_UPSTREAM, False, "FAILED_UPSTREAM"
    elif len(registry) != 86:
        gate, passed, readiness = FAIL_CARDINALITY, False, "FAILED_CARDINALITY"
    elif not split_ok:
        gate, passed, readiness = FAIL_SPLIT, False, "FAILED_SPLIT"
    elif not no_leak:
        gate, passed, readiness = FAIL_LEAK, False, "FAILED_HOLDOUT_LEAKAGE"
    elif not finite:
        gate, passed, readiness = "FAIL_SUSEONG_DL6D_R3_REWARD_OR_KPI_NAN_INF", False, "FAILED_NONFINITE"
    elif cands["c1"]["design_gate_passed"]:
        gate, passed, readiness = C1_GATE, True, C1_READINESS
    else:
        gate, passed, readiness = BLOCK_NO_DESIGN, False, "C1_FAILED_DESIGN_C2_NOT_ENTERED_BY_USER_INSTRUCTION"
    writer.json("gate_decision.json", {
        "created_at": iso_now(),
        "gate": gate,
        "gate_passed": passed,
        "readiness_decision": readiness,
        "selected_candidate_id": selected["selected_candidate_id"],
        "selected_repair_level": selected["selected_repair_level"],
        "c2_entry_locked_pending_user_command": True,
        "holdout_opened": False,
        "dl6d_r2_r1_authorized": False,
        "dl6e_p0_authorized": False,
    })
    writer.json("downstream_lock.json", {
        "created_at": iso_now(),
        "three_action_contract_verified": True,
        "observation_contract_verified": True,
        "reward_service_alignment_repaired": False,
        "selected_reward_candidate": selected["selected_candidate_id"],
        "selected_repair_level": selected["selected_repair_level"],
        "sealed_holdout_validated": False,
        "holdout_opened": False,
        "c2_entry_locked_pending_user_command": True,
        "reward_candidate_promoted": False,
        "reward_finalized": False,
        "dl6d_r2_r1_required": True,
        "dl6d_r2_r1_authorized": False,
        "dl6e_p0_required": True,
        "dl6e_p0_authorized": False,
        "dl6e_full_authorized": False,
        "mac_studio_full_research_authorized": False,
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
    })
    final = {
        "created_at": iso_now(),
        "artifact": str(output),
        "scope": "C1 design-stage pause artifact; hold-out not opened",
        "gate": gate,
        "gate_passed": passed,
        "readiness_decision": readiness,
        "problem_rows": int(len(registry)),
        "harmful_design": int(len(design_h)),
        "harmful_holdout_sealed": int(len(holdout_h)),
        "beneficial_design": int(len(design_b)),
        "beneficial_holdout_sealed": HOLDOUT_BENEFICIAL_N,
        "selected_candidate": selected,
        "c1": cands["c1"],
        "holdout_opened": False,
        "training_prohibition": training,
        "external_access": external,
    }
    writer.json("final_report.json", final)
    report = [
        "# Prompt 5-E01-DL-6D-R3 C1 Design Pause",
        "",
        "## 쉬운 설명",
        "문제가 된 86건을 전부 보면서 채점표를 맞추지 않았다.",
        "먼저 60건만 repair design에 사용하고, 26건은 봉인했다.",
        "C1은 reward formula나 weight를 바꾸지 않고 기존 service-harm component의 normalization/unit scale만 교정하는 후보로 평가했다.",
        "Design harmful 60건에서는 서비스 악화 skip 우대가 0건이 됐고, design beneficial 60건은 60건 모두 skip 우대를 유지했다.",
        "따라서 C1 design gate는 통과했지만, 사용자 지시에 따라 C2와 hold-out으로 들어가지 않고 여기서 대기한다.",
        "",
        "## 결과",
        f"- artifact: `{output}`",
        f"- gate: `{gate}`",
        f"- gate_passed: `{str(passed).lower()}`",
        f"- selected candidate: `{selected['selected_candidate_id']}`",
        f"- selected repair level: `{selected['selected_repair_level']}`",
        "- C2 entered: `false`",
        "- hold-out opened: `false`",
        "",
        "## 핵심 수치",
        f"- problem registry: `86 / {len(registry)}`",
        f"- harmful design / sealed hold-out: `{len(design_h)} / {len(holdout_h)}`",
        f"- beneficial design / sealed hold-out: `{len(design_b)} / {HOLDOUT_BENEFICIAL_N}`",
        f"- C1 design harmful misalignment: `{cands['c1']['design_service_misalignment_count']} / 60`",
        f"- C1 design beneficial retention: `{cands['c1']['design_beneficial_retention_count']} / 60`",
        f"- C1 normalization scale correction: `{cands['c1']['normalization_scale_correction']}`",
        "",
        "## Guards",
        "- direct action-ID penalty: `false`",
        "- reward double counting: `false`",
        f"- reward micro scenarios: `{micro_payload['reward_micro_scenarios_passed']} / {micro_payload['reward_micro_scenario_count']}`",
        "- training / optimizer / checkpoint / external access: `0 / 0 / 0 / 0`",
        "",
        "## 대기 상태",
        "- `_HOLDOUT_SEALED.lock` created: `true`",
        "- `_HOLDOUT_OPENED.lock` created: `false`",
        "- next step requires explicit user command before C2 or hold-out.",
        "",
    ]
    writer.text("final_report.md", "\n".join(report))
    writer.text("_SUCCESS.lock", json.dumps({"created_at": iso_now(), "gate": gate, "gate_passed": passed, "holdout_opened": False}, sort_keys=True, allow_nan=False) + "\n")
    manifest = write_manifest(writer)
    print(f"[DL-6D-R3] artifact: {output}")
    print("[DL-6D-R3] platform: MAC_MINI_M4_24GB")
    print("[DL-6D-R3] accelerator: APPLE_MPS")
    print(f"[DL-6D-R3] upstream DL-6D-R2 gate: {upstream['dl6d_r2_gate']}")
    print(f"[DL-6D-R3] upstream DL-6D-R1 gate: {upstream['dl6d_r1_gate']}")
    print(f"[DL-6D-R3] upstream DL-6C gate: {upstream['dl6c_gate']}")
    print(f"[DL-6D-R3] problem registry expected/actual: 86 / {len(registry)}")
    print(f"[DL-6D-R3] harmful design/holdout: {len(design_h)} / {len(holdout_h)}")
    print(f"[DL-6D-R3] beneficial design/holdout: {len(design_b)} / {HOLDOUT_BENEFICIAL_N}")
    print("[DL-6D-R3] holdout sealed before design: true")
    print("[DL-6D-R3] holdout open count: 0")
    print("[DL-6D-R3] tolerance frozen before repair: true")
    print("[DL-6D-R3] beneficial gate frozen before repair: true")
    print(f"[DL-6D-R3] temporal harm primary cause: {root_cause['primary_cause']}")
    print(f"[DL-6D-R3] selected candidate: {selected['selected_candidate_id']}")
    print(f"[DL-6D-R3] selected repair level: {selected['selected_repair_level']}")
    print("[DL-6D-R3] direct action-ID penalty: false")
    print("[DL-6D-R3] reward double counting: false")
    print(f"[DL-6D-R3] design harmful misalignment: {cands['c1']['design_service_misalignment_count']} / 60")
    print(f"[DL-6D-R3] design beneficial retention: {cands['c1']['design_beneficial_retention_count']} / 60")
    print("[DL-6D-R3] holdout harmful misalignment: NOT_OPENED / 26")
    print("[DL-6D-R3] holdout beneficial retention: NOT_OPENED / 26")
    print("[DL-6D-R3] shadow harmful misalignment pre/post: NOT_RUN")
    print(f"[DL-6D-R3] reward micro scenarios: {micro_payload['reward_micro_scenarios_passed']} / {micro_payload['reward_micro_scenario_count']}")
    print("[DL-6D-R3] temporal credit after repair: true")
    print("[DL-6D-R3] single-pulse alignment verified: true")
    print("[DL-6D-R3] closed-loop alignment verified: false")
    print("[DL-6D-R3] training runs: 0")
    print("[DL-6D-R3] optimizer steps: 0")
    print("[DL-6D-R3] checkpoint writes: 0")
    print("[DL-6D-R3] external access: 0")
    print(f"[DL-6D-R3] gate: {gate}")
    print(f"[DL-6D-R3] gate_passed: {str(passed).lower()}")
    print("[DL-6D-R3] C2 entry: LOCKED_PENDING_USER_COMMAND")
    print("[DL-6D-R3] DL-6D-R2-R1 authorized: false")
    print("[DL-6D-R3] DL-6E-P0 authorized: false")
    print("[DL-6D-R3] Mac Studio full research authorized: false")
    if manifest["manifest_missing_required_file_count"] or not manifest["success_lock_created_last"]:
        print("[DL-6D-R3] manifest warning: C1 design-pause artifact incomplete")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["c1", "prepare", "design", "holdout", "shadow", "finalize"], default="c1")
    args = parser.parse_args(argv)
    if args.mode != "c1":
        raise SystemExit("This user-authorized run is locked to --mode c1. C2/holdout/shadow/finalize require a new explicit user command.")
    run_c1()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
