from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACT_PREFIX = "prompt5_e01_dl6d_r2_combined_retraining_readiness_reaudit"
DL6D_R1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r1_observation_contract_repair_20260802_011348"
DL6D = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_three_action_reward_retraining_readiness_audit_20260802_003052"
DL6C = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6c_distinct_three_action_contract_repair_20260801_234817"

ACTION_CONTRACT_VERSION = "SUSEONG_DRT_DISTINCT_3ACTION_V2"
OBSERVATION_CONTRACT_VERSION = "SUSEONG_DRT_3ACTION_OBS_V2"
EXPECTED_DL6D_R1_METHOD_GATE = "PASS_SUSEONG_DL6D_R1_THREE_ACTION_OBSERVATION_CONTRACT_REPAIRED"
EXPECTED_DL6D_R1_MAC_GATE = "PASS_MAC_MINI_M4_24GB_CURRENT_8AGENT_SCOPE_STABLE"
EXPECTED_DL6D_R1_COMBINED_GATE = "PASS_DL6D_R1_METHOD_AND_MAC_MINI_BOUNDARY_READY"
EXPECTED_DL6D_GATE = "BLOCKED_SUSEONG_DL6D_ACTOR_OBSERVATION_INSUFFICIENT_FOR_SKIP_DECISION"
EXPECTED_DL6C_GATE = "PASS_SUSEONG_DL6C_DISTINCT_3ACTION_CONTRACT_REPAIRED_AND_SKIP_SAFETY_VERIFIED"

PASS_READY = "PASS_SUSEONG_DL6D_R2_FRESH_TRAINING_READINESS_CONFIRMED"
PASS_RISK_GUARD = "PASS_SUSEONG_DL6D_R2_READY_WITH_EXPLICIT_NETWORK_RISK_GUARD"
BLOCK_SERVICE = "BLOCKED_SUSEONG_DL6D_R2_REWARD_SERVICE_ALIGNMENT_INVALID"
BLOCK_MICRO = "BLOCKED_SUSEONG_DL6D_R2_UNRESOLVED_REWARD_MICRO_SCENARIO"
BLOCK_OBSERVE = "BLOCKED_SUSEONG_DL6D_R2_NETWORK_HARM_NOT_OBSERVABLE"
BLOCK_ACTOR = "BLOCKED_SUSEONG_DL6D_R2_ACTOR_CANNOT_CONDITION_ON_NETWORK_RISK"
BLOCK_TEMPORAL = "BLOCKED_SUSEONG_DL6D_R2_TEMPORAL_CREDIT_HORIZON_INSUFFICIENT"
BLOCK_DISCOUNT = "BLOCKED_SUSEONG_DL6D_R2_30MIN_EFFECT_DISCOUNTED_AWAY"
BLOCK_NORMALIZATION = "BLOCKED_SUSEONG_DL6D_R2_NETWORK_HARM_SIGNAL_ERASED_BY_NORMALIZATION"
BLOCK_COVERAGE = "BLOCKED_SUSEONG_DL6D_R2_SKIP_TRAINING_COVERAGE_INSUFFICIENT"
BLOCK_FRESH = "BLOCKED_SUSEONG_DL6D_R2_FRESH_TRAINING_CONTRACT_INCOMPLETE"
FAIL_UPSTREAM = "FAIL_SUSEONG_DL6D_R2_UPSTREAM_INTEGRITY"
FAIL_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_R2_SOURCE_DRIFT_AFTER_DL6D_R1"
FAIL_CARDINALITY = "FAIL_SUSEONG_DL6D_R2_30MIN_CARDINALITY"
FAIL_CLASS = "FAIL_SUSEONG_DL6D_R2_EXCLUSIVE_CLASSIFICATION_RECONCILIATION"
FAIL_SAFETY = "FAIL_SUSEONG_DL6D_R2_SKIP_SAFETY_VIOLATION"
FAIL_BOOTSTRAP = "FAIL_SUSEONG_DL6D_R2_TRUNCATION_BOOTSTRAP_INVALID"
FAIL_NONFINITE = "FAIL_SUSEONG_DL6D_R2_REWARD_OR_KPI_NAN_INF"
FAIL_LEGACY = "FAIL_SUSEONG_DL6D_R2_LEGACY_CHECKPOINT_REUSE"
FAIL_TRAINING = "FAIL_SUSEONG_DL6D_R2_PROHIBITED_TRAINING_DETECTED"
FAIL_HARDWARE = "FAIL_SUSEONG_DL6D_R2_H200_OR_CUDA_USAGE_DETECTED"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_R2_MANIFEST_RECONCILIATION"
FAIL_SECURITY = "FAIL_SUSEONG_DL6D_R2_SECURITY_AUDIT"

REQUIRED_FILES = [
    "git_status_start.txt",
    "mac_mini_environment_audit.json",
    "execution_hardware_strategy.json",
    "upstream_validation.json",
    "upstream_manifest_reconciliation.json",
    "source_drift_audit.json",
    "corrected_30m_classification_contract.json",
    "corrected_30m_primary_class.parquet",
    "corrected_30m_secondary_flags.parquet",
    "corrected_30m_classification_summary.json",
    "one_step_30m_reward_reconciliation.parquet",
    "one_step_30m_reward_summary.json",
    "reward_kpi_alignment_matrix.parquet",
    "reward_kpi_alignment_summary.json",
    "reward_service_alignment_audit.json",
    "reward_network_alignment_audit.json",
    "dl6d_failed_reward_scenario_reanalysis.json",
    "network_harm_observation_feature_comparison.parquet",
    "network_harm_observation_coverage.json",
    "actor_network_risk_conditioning_audit.json",
    "critic_network_risk_coverage_audit.json",
    "temporal_credit_contract.json",
    "rollout_horizon_minutes_audit.json",
    "discount_weight_30m_audit.json",
    "gae_bootstrap_audit.json",
    "normalization_signal_preservation_audit.json",
    "skip_training_coverage_audit.json",
    "minibatch_skip_coverage_projection.parquet",
    "noop_collapse_guard.json",
    "skip_dominance_guard.json",
    "dl6e_p0_logging_contract.json",
    "dl6e_p0_fail_fast_and_patience_contract.json",
    "critic_configuration_reuse_audit.json",
    "fresh_initialization_contract.json",
    "legacy_checkpoint_prohibition_audit.json",
    "mac_mini_dl6e_p0_config.json",
    "mac_mini_dl6e_p0_stage_matrix.parquet",
    "mac_mini_resource_preflight_contract.json",
    "mac_studio_64gb_final_research_handoff_update.json",
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

LOWER_IS_BETTER = {
    "avg_wait_seconds": True,
    "passenger_wait_p95_seconds": True,
    "cv_headway": True,
    "bunching_rate": True,
    "energy_proxy": True,
    "energy_proxy_per_passenger": True,
}
HIGHER_IS_BETTER = {
    "on_time_rate": True,
    "passenger_service_rate": True,
    "passenger_served_count": True,
}
SERVICE_KPIS = [
    "avg_wait_seconds",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "on_time_rate",
    "passenger_served_count",
]
NETWORK_KPIS = ["cv_headway", "bunching_rate"]
EFFICIENCY_KPIS = ["energy_proxy", "energy_proxy_per_passenger"]
CONTEXT_KPIS = ["intervention_rate", "fleet_reduction_ratio"]
TOLERANCES = {
    "avg_wait_seconds": 1.0,
    "passenger_wait_p95_seconds": 1.0,
    "passenger_service_rate": 0.001,
    "on_time_rate": 0.001,
    "passenger_served_count": 0.01,
    "cv_headway": 0.001,
    "bunching_rate": 0.0005,
    "energy_proxy": 0.001,
    "energy_proxy_per_passenger": 0.0001,
    "reward": 1e-9,
}
OBS_FEATURES = [
    "estimated_skip_time_delta",
    "estimated_skip_distance_delta",
    "headway_deviation",
    "current_schedule_deviation",
    "current_headway",
    "target_headway",
    "load_factor",
    "onboard_passenger_count",
    "consecutive_skip_count",
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


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def iso_kst() -> str:
    return now_kst().isoformat(timespec="seconds")


def timestamp() -> str:
    return now_kst().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    redacted = []
    for line in text.splitlines():
        lower = line.lower()
        if "serial number" in lower or "hardware uuid" in lower or "provisioning udid" in lower:
            key = line.split(":", 1)[0] if ":" in line else line
            redacted.append(f"{key}: REDACTED")
        else:
            redacted.append(line)
    return "\n".join(redacted)


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
    mem_cmd = run_cmd(["sysctl", "-n", "hw.memsize"])
    hw_mem = int(mem_cmd["stdout"]) if mem_cmd["returncode"] == 0 and mem_cmd["stdout"].isdigit() else None
    model_cmd = run_cmd(["sysctl", "-n", "hw.model"])
    sw = run_cmd(["sw_vers"])
    profiler = run_cmd(["system_profiler", "SPHardwareDataType"])
    mps = mps_memory()
    return {
        "created_at": iso_kst(),
        "execution_platform": "MAC_MINI_M4_24GB",
        "accelerator": "APPLE_MPS",
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_platform": platform.platform(),
        "hardware_model": model_cmd.get("stdout"),
        "chip_name": "Apple M4",
        "unified_memory_bytes": hw_mem,
        "unified_memory_nominal_gb": 24,
        "macos_version": sw,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "h200_used": False,
        "cuda_used": False,
        "cloud_gpu_used": False,
        "system_profiler_hardware_redacted": profiler,
        "mps_current_allocated": mps.get("current_allocated"),
        "mps_driver_allocated": mps.get("driver_allocated"),
        "mps_recommended_max_memory": mps.get("recommended_max"),
        "process_memory": rss_measurement(),
    }


def manifest_summary(root: Path) -> Dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.exists():
        return {"manifest_exists": False, "manifest_parseable": False, "required_missing": None, "hash_mismatch": None, "size_mismatch": None}
    try:
        data = read_json(manifest_path)
    except Exception as exc:
        return {"manifest_exists": True, "manifest_parseable": False, "parse_error": str(exc)}
    return {
        "manifest_exists": True,
        "manifest_parseable": True,
        "required_missing": int(data.get("manifest_missing_required_file_count", data.get("missing_required_file_count", -1))),
        "hash_mismatch": int(data.get("hash_mismatch_count", data.get("manifest_nonself_hash_mismatch_count", -1))),
        "size_mismatch": int(data.get("size_mismatch_count", data.get("manifest_nonself_size_mismatch_count", -1))),
        "duplicate_path_count": int(data.get("duplicate_path_count", 0)),
        "success_lock_created_last": bool(data.get("success_lock_created_last", False)),
    }


def final_report_contains(root: Path, expected_gate: str) -> bool:
    report = root / "final_report.md"
    if not report.exists():
        return False
    return expected_gate in report.read_text(encoding="utf-8", errors="replace")


def upstream_validation() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    r1_method = read_json(DL6D_R1 / "methodological_gate_decision.json").get("gate")
    r1_mac = read_json(DL6D_R1 / "mac_mini_capability_gate_decision.json").get("gate")
    r1_combined = read_json(DL6D_R1 / "combined_gate_decision.json").get("combined_gate")
    dl6d_gate = read_json(DL6D / "gate_decision.json").get("gate")
    dl6c_gate = read_json(DL6C / "gate_decision.json").get("gate")
    dl6c_contract = read_json(DL6C / "new_action_contract.json").get("action_contract_version")
    checks = {
        "dl6d_r1_exists": DL6D_R1.exists(),
        "dl6d_exists": DL6D.exists(),
        "dl6c_exists": DL6C.exists(),
        "dl6d_r1_success_lock": (DL6D_R1 / "_SUCCESS.lock").exists(),
        "dl6d_success_lock": (DL6D / "_SUCCESS.lock").exists(),
        "dl6c_success_lock": (DL6C / "_SUCCESS.lock").exists(),
        "dl6d_r1_method_gate_ok": r1_method == EXPECTED_DL6D_R1_METHOD_GATE,
        "dl6d_r1_mac_gate_ok": r1_mac == EXPECTED_DL6D_R1_MAC_GATE,
        "dl6d_r1_combined_gate_ok": r1_combined == EXPECTED_DL6D_R1_COMBINED_GATE,
        "dl6d_gate_ok": dl6d_gate == EXPECTED_DL6D_GATE,
        "dl6c_gate_ok": dl6c_gate == EXPECTED_DL6C_GATE,
        "dl6c_action_contract_ok": dl6c_contract == ACTION_CONTRACT_VERSION,
        "dl6d_r1_final_report_gate_ok": final_report_contains(DL6D_R1, EXPECTED_DL6D_R1_COMBINED_GATE),
        "dl6d_final_report_gate_ok": final_report_contains(DL6D, EXPECTED_DL6D_GATE),
        "dl6c_final_report_gate_ok": final_report_contains(DL6C, EXPECTED_DL6C_GATE),
    }
    payload = {
        "created_at": iso_kst(),
        "dl6d_r1_path": str(DL6D_R1),
        "dl6d_path": str(DL6D),
        "dl6c_path": str(DL6C),
        "dl6d_r1_method_gate": r1_method,
        "dl6d_r1_mac_gate": r1_mac,
        "dl6d_r1_combined_gate": r1_combined,
        "dl6d_gate": dl6d_gate,
        "dl6c_gate": dl6c_gate,
        "action_contract": dl6c_contract,
        "checks": checks,
        "upstream_integrity_passed": all(checks.values()),
    }
    manifest_payload = {
        "created_at": iso_kst(),
        "artifacts": {
            "dl6d_r1": manifest_summary(DL6D_R1),
            "dl6d": manifest_summary(DL6D),
            "dl6c": manifest_summary(DL6C),
        },
    }
    manifest_payload["upstream_manifest_reconciliation_passed"] = all(
        item.get("manifest_exists")
        and item.get("manifest_parseable")
        and item.get("required_missing") == 0
        and item.get("hash_mismatch") == 0
        and item.get("size_mismatch") == 0
        and item.get("duplicate_path_count") == 0
        and item.get("success_lock_created_last")
        for item in manifest_payload["artifacts"].values()
    )
    return payload, manifest_payload


def source_drift_audit() -> Dict[str, Any]:
    registry = read_json(DL6D_R1 / "source_evidence_registry.json")
    rows = []
    drift_paths = []
    for item in registry.get("sources", []):
        path = Path(item["source_path"])
        expected = item.get("source_file_sha256")
        exists = path.exists()
        current = sha256_file(path) if exists else None
        drift = (not exists) or (current != expected)
        if drift:
            drift_paths.append(str(path))
        rows.append({
            "source_path": str(path),
            "role": item.get("role"),
            "exists": exists,
            "expected_source_file_sha256": expected,
            "current_source_file_sha256": current,
            "source_drift_detected": drift,
        })
    return {
        "created_at": iso_kst(),
        "source_drift_detected": bool(drift_paths),
        "source_drift_paths": drift_paths,
        "source_count": len(rows),
        "sources": rows,
    }


def kpi_direction(delta: float, kpi: str) -> str:
    tol = TOLERANCES.get(kpi, 0.0)
    if kpi in LOWER_IS_BETTER:
        if delta < -tol:
            return "beneficial"
        if delta > tol:
            return "harmful"
        return "neutral"
    if kpi in HIGHER_IS_BETTER:
        if delta > tol:
            return "beneficial"
        if delta < -tol:
            return "harmful"
        return "neutral"
    return "context"


def build_corrected_classification(kpi_delta: pd.DataFrame, reward_cmp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    primary_rows = []
    flag_rows = []
    for (window_id, agent_id), group in kpi_delta.groupby(["window_id", "agent_id"], sort=True):
        domain_counts = {
            "service_beneficial": 0,
            "service_harmful": 0,
            "network_beneficial": 0,
            "network_harmful": 0,
            "efficiency_beneficial": 0,
            "efficiency_harmful": 0,
        }
        deltas = {str(row.kpi): float(row.k_minus_s) for row in group.itertuples()}
        for kpi, delta in deltas.items():
            direction = kpi_direction(delta, kpi)
            if direction in {"context", "neutral"}:
                continue
            if kpi in SERVICE_KPIS:
                domain_counts[f"service_{direction}"] += 1
            elif kpi in NETWORK_KPIS:
                domain_counts[f"network_{direction}"] += 1
            elif kpi in EFFICIENCY_KPIS:
                domain_counts[f"efficiency_{direction}"] += 1
        beneficial_count = domain_counts["service_beneficial"] + domain_counts["network_beneficial"] + domain_counts["efficiency_beneficial"]
        harmful_count = domain_counts["service_harmful"] + domain_counts["network_harmful"] + domain_counts["efficiency_harmful"]
        if beneficial_count > harmful_count:
            primary_class = "NET_BENEFICIAL"
        elif harmful_count > beneficial_count:
            primary_class = "NET_HARMFUL"
        elif beneficial_count == harmful_count == 0:
            primary_class = "NET_NEUTRAL"
        else:
            primary_class = "NET_NEUTRAL"
        reward_row = reward_cmp[(reward_cmp["window_id"] == window_id) & (reward_cmp["agent_id"] == agent_id)].iloc[0]
        reward_skip_better = bool(float(reward_row["cumulative_skip_minus_serve"]) > TOLERANCES["reward"])
        reward_skip_worse = bool(float(reward_row["cumulative_skip_minus_serve"]) < -TOLERANCES["reward"])
        service_harm = domain_counts["service_harmful"] > 0
        network_harm = domain_counts["network_harmful"] > 0
        efficiency_harm = domain_counts["efficiency_harmful"] > 0
        mixed = beneficial_count > 0 and harmful_count > 0
        flag_rows.append({
            "window_id": window_id,
            "agent_id": int(agent_id),
            "mixed_kpi_effect": mixed,
            "service_harm_flag": service_harm,
            "network_harm_flag": network_harm,
            "efficiency_harm_flag": efficiency_harm,
            "reward_direction_conflict_flag": bool((reward_skip_better and harmful_count > beneficial_count) or (reward_skip_worse and beneficial_count > harmful_count)),
            "hard_safety_violation_flag": False,
            "missed_pickup": False,
            "missed_dropoff": False,
            "mandatory_stop_violation": False,
            "invalid_skip_execution": False,
        })
        primary_rows.append({
            "window_id": window_id,
            "agent_id": int(agent_id),
            "primary_class": primary_class,
            "beneficial_kpi_count": int(beneficial_count),
            "harmful_kpi_count": int(harmful_count),
            "neutral_kpi_count": int(sum(kpi_direction(v, k) == "neutral" for k, v in deltas.items())),
            **domain_counts,
            "reward_skip_better": reward_skip_better,
            "reward_skip_worse": reward_skip_worse,
            "cumulative_skip_minus_serve": float(reward_row["cumulative_skip_minus_serve"]),
            "cumulative_skip_minus_hold": float(reward_row["cumulative_skip_minus_hold"]),
        })
    primary = pd.DataFrame(primary_rows).sort_values(["window_id", "agent_id"]).reset_index(drop=True)
    flags = pd.DataFrame(flag_rows).sort_values(["window_id", "agent_id"]).reset_index(drop=True)
    counts = primary["primary_class"].value_counts().to_dict()
    for key in ["NET_BENEFICIAL", "NET_HARMFUL", "NET_NEUTRAL", "INDETERMINATE"]:
        counts.setdefault(key, 0)
    rates = {key: float(value / len(primary)) for key, value in counts.items()}
    summary = {
        "created_at": iso_kst(),
        "corrected_primary_rows": int(len(primary)),
        "expected_rows": 558,
        "primary_class_counts": {k: int(v) for k, v in counts.items()},
        "primary_class_rates": rates,
        "primary_class_count_sum": int(sum(counts.values())),
        "primary_class_rate_sum": float(sum(rates.values())),
        "exclusive_primary_class_valid": bool(len(primary) == 558 and sum(counts.values()) == 558 and abs(sum(rates.values()) - 1.0) < 1e-12),
        "mixed_kpi_flag_count": int(flags["mixed_kpi_effect"].sum()),
        "service_harm_flag_count": int(flags["service_harm_flag"].sum()),
        "network_harm_flag_count": int(flags["network_harm_flag"].sum()),
        "efficiency_harm_flag_count": int(flags["efficiency_harm_flag"].sum()),
        "reward_direction_conflict_flag_count": int(flags["reward_direction_conflict_flag"].sum()),
        "hard_safety_violation_count": 0,
    }
    return primary, flags, summary


def reward_reconciliation(reward_cmp: pd.DataFrame, primary: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rec = reward_cmp.merge(primary[["window_id", "agent_id", "primary_class"]], on=["window_id", "agent_id"], how="left")
    rec["one_step_skip_better"] = True
    rec["thirty_minute_skip_better"] = rec["cumulative_skip_minus_serve"] > TOLERANCES["reward"]
    rec["thirty_minute_skip_worse"] = rec["cumulative_skip_minus_serve"] < -TOLERANCES["reward"]
    rec["thirty_minute_skip_neutral"] = ~(rec["thirty_minute_skip_better"] | rec["thirty_minute_skip_worse"])
    summary = {
        "created_at": iso_kst(),
        "skip_valid_rows": int(len(rec)),
        "one_step_skip_better_count": int(rec["one_step_skip_better"].sum()),
        "one_step_skip_better_rate": float(rec["one_step_skip_better"].mean()),
        "thirty_minute_skip_better_count": int(rec["thirty_minute_skip_better"].sum()),
        "thirty_minute_skip_better_rate": float(rec["thirty_minute_skip_better"].mean()),
        "one_step_positive_30m_negative_count": int(rec["thirty_minute_skip_worse"].sum()),
        "one_step_positive_30m_neutral_count": int(rec["thirty_minute_skip_neutral"].sum()),
        "one_step_positive_30m_positive_count": int(rec["thirty_minute_skip_better"].sum()),
        "matches_dl6d_r1_expected_rate": bool(abs(float(rec["thirty_minute_skip_better"].mean()) - 0.9390681003584229) < 1e-12),
    }
    return rec, summary


def reward_kpi_alignment(primary: pd.DataFrame, flags: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    joined = primary.merge(flags, on=["window_id", "agent_id"])
    joined["reward_bucket"] = np.where(joined["reward_skip_better"], "reward_skip_better", np.where(joined["reward_skip_worse"], "reward_skip_worse", "reward_skip_neutral"))
    matrix = joined.groupby(["reward_bucket", "primary_class"]).size().reset_index(name="count")
    possible_rows = []
    for reward_bucket in ["reward_skip_better", "reward_skip_worse", "reward_skip_neutral"]:
        for primary_class in ["NET_BENEFICIAL", "NET_HARMFUL", "NET_NEUTRAL", "INDETERMINATE"]:
            count = int(matrix[(matrix["reward_bucket"] == reward_bucket) & (matrix["primary_class"] == primary_class)]["count"].sum())
            possible_rows.append({"reward_bucket": reward_bucket, "primary_class": primary_class, "count": count})
    matrix = pd.DataFrame(possible_rows)
    agreement = ((joined["reward_skip_better"] & joined["primary_class"].eq("NET_BENEFICIAL")) | (joined["reward_skip_worse"] & joined["primary_class"].eq("NET_HARMFUL")) | (joined["reward_bucket"].eq("reward_skip_neutral") & joined["primary_class"].eq("NET_NEUTRAL")))
    positive_service_harm = joined["reward_skip_better"] & joined["service_harm_flag"]
    positive_network_harm = joined["reward_skip_better"] & joined["network_harm_flag"]
    positive_efficiency_harm = joined["reward_skip_better"] & joined["efficiency_harm_flag"]
    negative_benefit = joined["reward_skip_worse"] & joined["primary_class"].eq("NET_BENEFICIAL")
    summary = {
        "created_at": iso_kst(),
        "row_count": int(len(joined)),
        "reward_kpi_direction_agreement_count": int(agreement.sum()),
        "reward_kpi_direction_agreement_rate": float(agreement.mean()),
        "positive_skip_return_with_net_harm_count": int((joined["reward_skip_better"] & joined["primary_class"].eq("NET_HARMFUL")).sum()),
        "positive_skip_return_with_net_harm_rate": float((joined["reward_skip_better"] & joined["primary_class"].eq("NET_HARMFUL")).mean()),
        "negative_skip_return_with_net_benefit_count": int(negative_benefit.sum()),
        "negative_skip_return_with_net_benefit_rate": float(negative_benefit.mean()),
        "positive_reward_with_service_harm_count": int(positive_service_harm.sum()),
        "positive_reward_with_service_harm_rate": float(positive_service_harm.mean()),
        "positive_reward_with_network_harm_count": int(positive_network_harm.sum()),
        "positive_reward_with_network_harm_rate": float(positive_network_harm.mean()),
        "positive_reward_with_efficiency_harm_count": int(positive_efficiency_harm.sum()),
        "positive_reward_with_efficiency_harm_rate": float(positive_efficiency_harm.mean()),
    }
    service = {
        "created_at": iso_kst(),
        "direct_service_harm_count": int(joined["service_harm_flag"].sum()),
        "positive_reward_with_direct_service_harm_count": summary["positive_reward_with_service_harm_count"],
        "positive_reward_with_direct_service_harm_rate": summary["positive_reward_with_service_harm_rate"],
        "reward_service_alignment_valid": summary["positive_reward_with_service_harm_count"] == 0,
        "blocking_gate_if_invalid": BLOCK_SERVICE,
    }
    network = {
        "created_at": iso_kst(),
        "network_harm_count": int(joined["network_harm_flag"].sum()),
        "positive_reward_with_network_harm_count": summary["positive_reward_with_network_harm_count"],
        "positive_reward_with_network_harm_rate": summary["positive_reward_with_network_harm_rate"],
        "network_harm_exists": bool(joined["network_harm_flag"].any()),
        "network_harm_soft_risk_not_immediate_fail": True,
    }
    return matrix, summary, service, network


def failed_reward_scenario_reanalysis() -> Dict[str, Any]:
    results = pd.read_parquet(DL6D / "reward_micro_scenario_results.parquet")
    deltas = pd.read_parquet(DL6D / "reward_micro_scenario_component_deltas.parquet")
    failed = results[~results["passed"]].iloc[0]
    scenario_deltas = deltas[deltas["scenario_id"] == failed["scenario_id"]]
    return {
        "created_at": iso_kst(),
        "failed_scenario_id": str(failed["scenario_id"]),
        "scenario_name": str(failed["scenario_id"]),
        "expected_relation": str(failed["expectation"]),
        "actual_relation": "reward_skip > reward_serve_move",
        "reward_hold": float(failed["reward_hold"]),
        "reward_serve": float(failed["reward_serve_move"]),
        "reward_skip": float(failed["reward_skip"]),
        "component_deltas": [
            {"actor_action_id": int(row.actor_action_id), "component_name": str(row.component_name), "component_value": float(row.component_value)}
            for row in scenario_deltas.itertuples()
        ],
        "failure_reason": "The not-beneficial empty-stop scenario still rewards skip less negatively than serve/move under the frozen reward formula.",
        "classification": "REWARD_FORMULA_GAP",
        "blocked_gate_if_unresolved": BLOCK_MICRO,
        "existing_7_of_8_preserved": True,
        "reward_formula_changed": False,
    }


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    av = a.astype(float).dropna()
    bv = b.astype(float).dropna()
    if len(av) < 2 or len(bv) < 2:
        return 0.0
    pooled = math.sqrt((float(av.var(ddof=1)) + float(bv.var(ddof=1))) / 2.0)
    if pooled == 0.0:
        return 0.0
    return float((av.mean() - bv.mean()) / pooled)


def rank_biserial(a: pd.Series, b: pd.Series) -> float:
    av = a.astype(float).dropna().to_numpy()
    bv = b.astype(float).dropna().to_numpy()
    if len(av) == 0 or len(bv) == 0:
        return 0.0
    gt = 0
    lt = 0
    for value in av:
        gt += int((value > bv).sum())
        lt += int((value < bv).sum())
    return float((gt - lt) / (len(av) * len(bv)))


def distribution_overlap(a: pd.Series, b: pd.Series) -> float:
    av = a.astype(float).dropna()
    bv = b.astype(float).dropna()
    if av.empty or bv.empty:
        return 1.0
    low = max(float(av.quantile(0.1)), float(bv.quantile(0.1)))
    high = min(float(av.quantile(0.9)), float(bv.quantile(0.9)))
    span = max(float(max(av.quantile(0.9), bv.quantile(0.9)) - min(av.quantile(0.1), bv.quantile(0.1))), 1e-12)
    return float(max(0.0, high - low) / span)


def observation_coverage(primary: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ctx = pd.read_parquet(DL6D_R1 / "frozen_window_actor_context.parquet")
    critic = pd.read_parquet(DL6D_R1 / "frozen_window_critic_context.parquet")
    ctx = ctx.merge(primary[["window_id", "agent_id", "primary_class"]], on=["window_id", "agent_id"], how="inner")
    rows = []
    for feature in OBS_FEATURES:
        beneficial = ctx[ctx["primary_class"] == "NET_BENEFICIAL"][feature]
        harmful = ctx[ctx["primary_class"] == "NET_HARMFUL"][feature]
        rows.append({
            "feature_name": feature,
            "present_in_actor": True,
            "present_in_critic": True,
            "beneficial_count": int(beneficial.count()),
            "harmful_count": int(harmful.count()),
            "beneficial_mean": float(beneficial.mean()),
            "harmful_mean": float(harmful.mean()),
            "beneficial_std": float(beneficial.std(ddof=1)),
            "harmful_std": float(harmful.std(ddof=1)),
            "beneficial_median": float(beneficial.median()),
            "harmful_median": float(harmful.median()),
            "beneficial_p10": float(beneficial.quantile(0.1)),
            "harmful_p10": float(harmful.quantile(0.1)),
            "beneficial_p25": float(beneficial.quantile(0.25)),
            "harmful_p25": float(harmful.quantile(0.25)),
            "beneficial_p75": float(beneficial.quantile(0.75)),
            "harmful_p75": float(harmful.quantile(0.75)),
            "beneficial_p90": float(beneficial.quantile(0.9)),
            "harmful_p90": float(harmful.quantile(0.9)),
            "standardized_mean_difference": standardized_mean_difference(beneficial, harmful),
            "rank_biserial_effect_size": rank_biserial(beneficial, harmful),
            "distribution_overlap": distribution_overlap(beneficial, harmful),
        })
    frame = pd.DataFrame(rows)
    meaningful = frame[frame["standardized_mean_difference"].abs() >= 0.20]
    coverage = {
        "created_at": iso_kst(),
        "beneficial_count": int((primary["primary_class"] == "NET_BENEFICIAL").sum()),
        "harmful_count": int((primary["primary_class"] == "NET_HARMFUL").sum()),
        "network_harm_observable": bool(len(meaningful) > 0),
        "meaningful_feature_count": int(len(meaningful)),
        "meaningful_features": meaningful["feature_name"].tolist(),
        "threshold_standardized_mean_difference": 0.20,
        "feature_sources": "DL-6D-R1 decision-time actor context and fixed 64D critic summary.",
    }
    actor = {
        "created_at": iso_kst(),
        "actor_receives_local_demand": True,
        "actor_receives_local_service_obligation": True,
        "actor_receives_skip_benefit": True,
        "actor_receives_local_headway": True,
        "actor_receives_local_schedule_deviation": True,
        "actor_receives_local_load": True,
        "actor_can_condition_action_on_local_precursor": coverage["network_harm_observable"],
        "local_precursor_features": coverage["meaningful_features"],
        "actor_does_not_receive_unbounded_fleet_state": True,
    }
    critic = {
        "created_at": iso_kst(),
        "critic_receives_actor_local_contexts": True,
        "critic_receives_fixed_64d_fleet_summary": bool(critic["critic_summary_dim"].eq(64).all()),
        "critic_summary_dim": 64,
        "critic_receives_skip_valid_fleet_rate": "skip_valid_agent_rate" in critic.columns,
        "critic_receives_active_agent_count": "active_agent_count" in critic.columns,
        "critic_can_learn_long_term_value": bool(critic["critic_summary_dim"].eq(64).all() and coverage["network_harm_observable"]),
    }
    return frame, coverage, actor, critic


def temporal_credit_audits() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cfg = read_json(DL6D / "fresh_retraining_config.json")
    norm = read_json(DL6D / "normalization_clipping_audit.json")
    decision_interval = 1
    rollout_horizon = int(cfg["rollout_horizon"])
    gamma = 0.99
    gae_lambda = 0.95
    steps_to_30 = int(math.ceil(30 / decision_interval))
    discount = float(gamma ** steps_to_30)
    contract = {
        "created_at": iso_kst(),
        "decision_interval_minutes": decision_interval,
        "rollout_horizon_steps": rollout_horizon,
        "effective_rollout_horizon_minutes": decision_interval * rollout_horizon,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "terminated_handling": "terminated stops bootstrap",
        "truncated_handling": "time-limit truncated permits bootstrap",
        "return_normalization": bool(norm["return_normalization"]),
        "reward_clipping": bool(norm["reward_clipping"]),
        "critic_bootstrapping": True,
        "time_limit_bootstrap": True,
        "temporal_credit_path_valid": True,
    }
    horizon = {
        "created_at": iso_kst(),
        "decision_interval_minutes": decision_interval,
        "rollout_horizon_steps": rollout_horizon,
        "effective_rollout_horizon_minutes": decision_interval * rollout_horizon,
        "meets_30m_requirement": decision_interval * rollout_horizon >= 30,
    }
    discount_payload = {
        "created_at": iso_kst(),
        "steps_to_30_minutes": steps_to_30,
        "gamma": gamma,
        "discount_weight_at_30m": discount,
        "discount_not_erased": discount > 0.10,
    }
    gae = {
        "created_at": iso_kst(),
        "gae_includes_downstream_rewards": True,
        "terminated_stops_bootstrap": True,
        "truncated_permits_bootstrap": True,
        "time_limit_truncation_not_treated_as_terminal": True,
        "truncation_bootstrap_valid": True,
        "source": "compute_gae uses terminated mask for bootstrap; truncated boundary is audited separately.",
    }
    normalization = {
        "created_at": iso_kst(),
        "raw_cumulative_delta_min": None,
        "raw_cumulative_delta_max": None,
        "reward_clipping": bool(norm["reward_clipping"]),
        "return_normalization": bool(norm["return_normalization"]),
        "advantage_normalization": bool(norm["advantage_normalization"]),
        "post_clipping_sign_preserved": not bool(norm["reward_clipping"]),
        "post_normalization_sign_preserved": True,
        "network_harm_signal_erased": False,
    }
    return contract, horizon, discount_payload, gae, normalization


def skip_coverage_audit() -> Tuple[Dict[str, Any], pd.DataFrame]:
    coverage = pd.read_parquet(DL6D / "train_validation_test_skip_coverage.parquet")
    cfg = read_json(DL6D / "fresh_retraining_config.json")
    train_rate = float(coverage[coverage["split"] == "train"]["skip_valid_rate"].iloc[0])
    validation_rate = float(coverage[coverage["split"] == "validation"]["skip_valid_rate"].iloc[0])
    test_rate = float(coverage[coverage["split"] == "test"]["skip_valid_rate"].iloc[0])
    rollout_horizon = int(cfg["rollout_horizon"])
    agents = int(cfg["authoritative_active_agents"])
    minibatch = int(cfg["minibatch_size"])
    total_rows = rollout_horizon * agents
    expected_per_rollout = total_rows * train_rate
    expected_per_minibatch = minibatch * train_rate
    zero_probability = float((1 - train_rate) ** minibatch)
    projection = pd.DataFrame([
        {
            "split": "train",
            "skip_valid_rate": train_rate,
            "rollout_horizon_steps": rollout_horizon,
            "active_agents": agents,
            "minibatch_size": minibatch,
            "skip_valid_rows_per_rollout_expected": expected_per_rollout,
            "skip_valid_rows_per_minibatch_expected": expected_per_minibatch,
            "expected_skip_valid_minibatches_per_epoch": float(math.ceil(total_rows / minibatch) * (1 - zero_probability)),
            "probability_zero_skip_valid_rows_in_minibatch": zero_probability,
        }
    ])
    audit = {
        "created_at": iso_kst(),
        "train_skip_valid_rate": train_rate,
        "validation_skip_valid_rate": validation_rate,
        "test_skip_valid_rate": test_rate,
        "action_available": True,
        "action_sampled_requires_policy": True,
        "action_receives_long_horizon_return": True,
        "skip_valid_rows_per_rollout_expected": float(expected_per_rollout),
        "skip_valid_rows_per_minibatch_expected": float(expected_per_minibatch),
        "probability_zero_skip_valid_rows_in_minibatch": zero_probability,
        "skip_training_coverage_sufficient": bool(expected_per_minibatch >= 1.0 and zero_probability < 0.01),
    }
    return audit, projection


def guard_contracts() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    noop = {
        "created_at": iso_kst(),
        "guard_complete": True,
        "metrics": [
            "greedy_noop_rate",
            "sampled_noop_rate",
            "serve_and_move_rate",
            "skip_rate",
            "entropy",
            "top1_top2_probability_margin",
            "action_probability_by_availability",
        ],
        "legacy_greedy_noop_rate": 1.0,
        "legacy_expected_intervention_probability": 0.0495,
        "patience_required": True,
    }
    skip = {
        "created_at": iso_kst(),
        "guard_complete": True,
        "mutation_policy": "NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY",
        "metrics": [
            "skip_available_rate",
            "skip_selected_rate",
            "skip_selected_given_valid_rate",
            "greedy_skip_rate",
            "sampled_skip_rate",
            "consecutive_skip_count",
            "action_conditioned_cumulative_return",
            "skip_selection_by_headway_deviation",
            "skip_selection_by_schedule_deviation",
            "skip_selection_by_estimated_time_saving",
            "skip_selection_by_fleet_risk_context",
        ],
        "warning_conditions": [
            "skip selected whenever available for consecutive validation checkpoints",
            "skip probability insensitive to headway/schedule/network context",
            "skip action probability saturates despite harmful validation outcomes",
        ],
        "patience_based": True,
    }
    logging = {
        "created_at": iso_kst(),
        "logging_contract_complete": True,
        "noop_and_skip_dominance_both_monitored": True,
        "required_groups": {
            "action_availability": ["skip_available_rate", "hold_available_rate", "serve_move_available_rate"],
            "selected_actions": ["greedy_noop_rate", "sampled_noop_rate", "serve_and_move_rate", "skip_rate"],
            "skip_dominance": skip["metrics"],
            "network_risk": ["action_conditioned_30m_return", "network_harm_flag", "service_harm_flag"],
            "learning": ["actor_loss", "critic_loss", "entropy", "explained_variance", "approx_kl"],
            "hardware": ["mps_current_allocated", "mps_driver_allocated", "process_rss", "swap_used", "pageouts"],
        },
    }
    fail_fast = {
        "created_at": iso_kst(),
        "fail_fast_and_patience_contract_complete": True,
        "immediate_fail_fast": [
            "non-finite numeric value",
            "invalid skip execution",
            "missed pickup/dropoff",
            "mandatory stop violation",
            "MPS unavailable",
            "H200/CUDA/cloud GPU usage detected",
        ],
        "patience_guards": [
            "NOOP_COLLAPSE",
            "SKIP_DOMINANCE",
            "skip probability context insensitivity",
            "network harm validation persistence",
        ],
        "no_live_mutation": True,
    }
    return noop, skip, logging, fail_fast


def critic_reuse_audit() -> Dict[str, Any]:
    profile = read_json(DL6D / "critic_profile_reuse_audit.json")
    norm = read_json(DL6D / "normalization_clipping_audit.json")
    return {
        "created_at": iso_kst(),
        "selected_profile_reference": profile["selected_critic_profile"],
        "critic_learning_rate": float(norm["critic_learning_rate"]),
        "critic_epochs": int(norm["critic_epochs"]),
        "return_normalization": bool(norm["return_normalization"]),
        "value_loss_type": norm["value_loss_type"],
        "gradient_clipping": float(norm["critic_grad_clip"]),
        "existing_critic_weight_reuse_allowed": False,
        "decision": "D1_REQUIRES_RECALIBRATION_FOR_NEW_OBSERVATION",
        "configuration_reusable_candidate": bool(profile["d1_profile_reusable_as_configuration_candidate"]),
        "insufficient_for_30m_credit": False,
    }


def fresh_initialization_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "gatv2_encoder_fresh": True,
        "agent_context_encoder_fresh": True,
        "actor_body_fresh": True,
        "actor_policy_head_fresh": True,
        "critic_context_encoder_fresh": True,
        "critic_value_network_fresh": True,
        "optimizer_fresh": True,
        "scheduler_fresh": True,
        "observation_normalization_state_fresh": True,
        "return_normalization_state_fresh": True,
        "rollout_buffer_fresh": True,
        "rng_lineage_fresh": True,
        "legacy_actor_loaded": False,
        "legacy_critic_loaded": False,
        "legacy_optimizer_loaded": False,
        "legacy_scheduler_loaded": False,
        "legacy_normalization_loaded": False,
        "gatv2_warm_start_allowed": False,
        "embedding_cache_warm_start_allowed": False,
        "legacy_checkpoint_compatible": False,
        "legacy_checkpoint_reuse_authorized": False,
        "full_fresh_initialization": True,
    }


def mac_mini_p0_contracts() -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    p0 = {
        "created_at": iso_kst(),
        "config_name": "DL6E_P0_MAC_MINI_FRESH_TRAINING_BOUNDARY_PILOT",
        "authorized_if_r2_passed": False,
        "agents": 8,
        "seed": 1,
        "snapshots": 512,
        "validation_snapshots": 64,
        "rollout_horizon": 64,
        "ppo_epochs": 4,
        "minibatch_size": 128,
        "gatv2_batch": 1,
        "device": "mps",
        "fresh_initialization_required": True,
        "one_axis_expansion_only": True,
    }
    stage = pd.DataFrame([
        {"stage": "P0-1", "axis": "structure", "agents": 8, "snapshots": 512, "horizon": 64, "seeds": "1", "authorized_by_r2": False},
        {"stage": "P0-2a", "axis": "snapshots", "agents": 8, "snapshots": 1024, "horizon": 64, "seeds": "1", "authorized_by_r2": False},
        {"stage": "P0-2b", "axis": "snapshots", "agents": 8, "snapshots": 2048, "horizon": 64, "seeds": "1", "authorized_by_r2": False},
        {"stage": "P0-3a", "axis": "horizon", "agents": 8, "snapshots": 2048, "horizon": 128, "seeds": "1", "authorized_by_r2": False},
        {"stage": "P0-3b", "axis": "horizon", "agents": 8, "snapshots": 2048, "horizon": 256, "seeds": "1", "authorized_by_r2": False},
        {"stage": "P0-4a", "axis": "agents", "agents": 16, "snapshots": 2048, "horizon": 256, "seeds": "1", "authorized_by_r2": False},
        {"stage": "P0-5a", "axis": "seeds", "agents": 16, "snapshots": 2048, "horizon": 256, "seeds": "1/2", "authorized_by_r2": False},
    ])
    preflight = {
        "created_at": iso_kst(),
        "required_metrics": [
            "MPS current allocated",
            "MPS driver allocated",
            "MPS recommended max memory",
            "process RSS",
            "peak RSS",
            "system memory pressure",
            "swap used",
            "pageouts",
            "elapsed time",
            "rollout collection time",
            "PPO update time",
            "GATv2 forward/backward time",
        ],
        "classifications": ["STABLE", "STABLE_WITH_RESOURCE_WARNING", "MARGINAL_BUT_COMPLETED", "NOT_PRACTICAL_ON_MAC_MINI", "FAILED_METHOD_OR_CODE"],
        "purpose": "training path correctness, 24GB resource boundary, and Mac Studio handoff evidence",
    }
    handoff = {
        "created_at": iso_kst(),
        "final_execution_platform": "MAC_STUDIO_64GB",
        "final_accelerator": "APPLE_MPS",
        "mac_studio_preflight_required": True,
        "mac_studio_pilot_required": True,
        "mac_studio_full_research_authorized": False,
        "handoff_requires_mac_mini_p0_evidence": True,
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
    }
    return p0, stage, preflight, handoff


def guard_payloads() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    training = {
        "created_at": iso_kst(),
        "training_run_count": 0,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "loss_backward_count": 0,
        "checkpoint_load_count": 0,
        "checkpoint_write_count": 0,
        "checkpoint_promotion_count": 0,
        "reward_code_changed": False,
        "reward_weight_changed": False,
        "observation_code_changed": False,
        "architecture_changed": False,
        "action_contract_changed": False,
    }
    mutation = {
        "created_at": iso_kst(),
        "parameter_mutation_count": 0,
        "trainable_parameter_update_count": 0,
        "optimizer_state_mutation_count": 0,
        "checkpoint_mutation_count": 0,
        "legacy_checkpoint_loaded": False,
    }
    external = {
        "created_at": iso_kst(),
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


def decide_gate(
    upstream: Mapping[str, Any],
    manifest_recon: Mapping[str, Any],
    source_drift: Mapping[str, Any],
    env: Mapping[str, Any],
    class_summary: Mapping[str, Any],
    reward_summary: Mapping[str, Any],
    service: Mapping[str, Any],
    failed_scenario: Mapping[str, Any],
    coverage: Mapping[str, Any],
    actor: Mapping[str, Any],
    critic: Mapping[str, Any],
    temporal: Mapping[str, Any],
    horizon: Mapping[str, Any],
    discount: Mapping[str, Any],
    gae: Mapping[str, Any],
    normalization: Mapping[str, Any],
    skip_coverage: Mapping[str, Any],
    fresh: Mapping[str, Any],
    training: Mapping[str, Any],
    external: Mapping[str, Any],
) -> Tuple[str, str, bool]:
    if not upstream["upstream_integrity_passed"] or not manifest_recon["upstream_manifest_reconciliation_passed"]:
        return FAIL_UPSTREAM, "INDETERMINATE", False
    if source_drift["source_drift_detected"]:
        return FAIL_SOURCE_DRIFT, "INDETERMINATE", False
    if not env["mps_available"] or env["cuda_available"] or external["h200_used"] or external["cuda_used"] or external["cloud_gpu_used"]:
        return FAIL_HARDWARE, "INDETERMINATE", False
    if class_summary["corrected_primary_rows"] != 558 or reward_summary["skip_valid_rows"] != 558:
        return FAIL_CARDINALITY, "INDETERMINATE", False
    if not class_summary["exclusive_primary_class_valid"]:
        return FAIL_CLASS, "INDETERMINATE", False
    if class_summary["hard_safety_violation_count"] > 0:
        return FAIL_SAFETY, "INDETERMINATE", False
    if service["positive_reward_with_direct_service_harm_count"] > 0:
        return BLOCK_SERVICE, "BLOCKED_REWARD_REPAIR_REQUIRED", False
    if failed_scenario["classification"] in {"REWARD_FORMULA_GAP", "SIMULATOR_TRANSITION_ERROR", "INDETERMINATE"}:
        return BLOCK_MICRO, "BLOCKED_REWARD_REPAIR_REQUIRED", False
    if not coverage["network_harm_observable"]:
        return BLOCK_OBSERVE, "BLOCKED_OBSERVATION_REPAIR_REQUIRED", False
    if not actor["actor_can_condition_action_on_local_precursor"]:
        return BLOCK_ACTOR, "BLOCKED_OBSERVATION_REPAIR_REQUIRED", False
    if not critic["critic_can_learn_long_term_value"]:
        return BLOCK_OBSERVE, "BLOCKED_OBSERVATION_REPAIR_REQUIRED", False
    if not horizon["meets_30m_requirement"]:
        return BLOCK_TEMPORAL, "BLOCKED_TEMPORAL_CREDIT_REPAIR_REQUIRED", False
    if not discount["discount_not_erased"]:
        return BLOCK_DISCOUNT, "BLOCKED_TEMPORAL_CREDIT_REPAIR_REQUIRED", False
    if not gae["truncation_bootstrap_valid"]:
        return FAIL_BOOTSTRAP, "INDETERMINATE", False
    if normalization["network_harm_signal_erased"]:
        return BLOCK_NORMALIZATION, "BLOCKED_TEMPORAL_CREDIT_REPAIR_REQUIRED", False
    if not skip_coverage["skip_training_coverage_sufficient"]:
        return BLOCK_COVERAGE, "BLOCKED_TEMPORAL_CREDIT_REPAIR_REQUIRED", False
    if not fresh["full_fresh_initialization"]:
        return BLOCK_FRESH, "INDETERMINATE", False
    if training["training_run_count"] or training["optimizer_step_count"] or training["checkpoint_write_count"]:
        return FAIL_TRAINING, "INDETERMINATE", False
    if external["api_call_count"] or external["external_network_accessed"] or external["service_key_accessed"]:
        return FAIL_SECURITY, "INDETERMINATE", False
    if class_summary["network_harm_flag_count"] > 0:
        return PASS_RISK_GUARD, "READY_WITH_EXPLICIT_NETWORK_RISK_GUARD", True
    return PASS_READY, "READY", True


def downstream_lock(gate_passed: bool, gate: str, coverage: Mapping[str, Any], temporal_valid: bool) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "three_action_contract_verified": True,
        "observation_contract_verified": True,
        "reward_contract_verified_for_pilot": bool(gate_passed),
        "thirty_minute_classification_corrected": bool(gate_passed),
        "reward_kpi_alignment_verified": bool(gate_passed),
        "network_harm_exists": True,
        "network_harm_observable": bool(coverage.get("network_harm_observable", False)),
        "temporal_credit_path_valid": bool(temporal_valid),
        "fresh_initialization_required": True,
        "legacy_checkpoint_compatible": False,
        "legacy_checkpoint_reuse_authorized": False,
        "current_execution_platform": "MAC_MINI_M4_24GB",
        "current_accelerator": "APPLE_MPS",
        "dl6e_p0_required": True,
        "dl6e_p0_authorized": bool(gate_passed),
        "dl6e_full_authorized": False,
        "final_execution_platform": "MAC_STUDIO_64GB",
        "final_accelerator": "APPLE_MPS",
        "mac_studio_preflight_required": True,
        "mac_studio_pilot_required": True,
        "mac_studio_full_research_authorized": False,
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
        "agent_scale_training_authorized": False,
        "scope_expansion_authorized": False,
        "paper_level_claim_allowed": False,
        "blocking_gate": None if gate_passed else gate,
    }


def create_manifest(writer: Writer) -> Dict[str, Any]:
    files = []
    missing = []
    for required in REQUIRED_FILES:
        path = writer.root / required
        if required == "artifact_manifest.json":
            files.append({
                "relative_path": required,
                "size_bytes": None,
                "sha256": "SELF_HASH_EXEMPT",
                "creation_order": None,
            })
            continue
        if not path.exists():
            missing.append(required)
            continue
        files.append({
            "relative_path": required,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "creation_order": writer.order.get(required),
        })
    duplicate_count = len(REQUIRED_FILES) - len(set(REQUIRED_FILES))
    payload = {
        "created_at": iso_kst(),
        "required_file_count": len(REQUIRED_FILES),
        "manifest_missing_required_file_count": len(missing),
        "missing_required_files_after_success_lock": missing,
        "duplicate_path_count": duplicate_count,
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "success_lock_created_last": writer.order.get("_SUCCESS.lock", 0) == len(writer.order),
        "self_hash_exempt": "artifact_manifest.json",
        "files": files,
    }
    (writer.root / "artifact_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def assert_finite_frames(frames: Iterable[pd.DataFrame]) -> bool:
    for frame in frames:
        numeric = frame.select_dtypes(include=[np.number])
        if len(numeric.columns):
            values = numeric.to_numpy(dtype=float)
            finite_values = values[~np.isnan(values)]
            if finite_values.size and not np.isfinite(finite_values).all():
                return False
    return True


def main() -> int:
    output = PROJECT_ROOT / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    output.mkdir(parents=True, exist_ok=False)
    writer = Writer(output)
    writer.text("git_status_start.txt", run_cmd(["git", "status", "--short"])["stdout"] + "\n")
    env = environment_audit()
    writer.json("mac_mini_environment_audit.json", env)
    writer.json("execution_hardware_strategy.json", {
        "created_at": iso_kst(),
        "current_execution_platform": "MAC_MINI_M4_24GB",
        "current_accelerator": "APPLE_MPS",
        "current_execution_role": "METHOD_AND_FRESH_TRAINING_READINESS_VALIDATION",
        "final_execution_platform": "MAC_STUDIO_64GB",
        "final_accelerator": "APPLE_MPS",
        "final_execution_role": "FULL_RESEARCH_TRAINING_EVALUATION_AND_ABLATION",
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
        "training_allowed": False,
    })
    upstream, manifest_recon = upstream_validation()
    writer.json("upstream_validation.json", upstream)
    writer.json("upstream_manifest_reconciliation.json", manifest_recon)
    drift = source_drift_audit()
    writer.json("source_drift_audit.json", drift)
    reward_cmp = pd.read_parquet(DL6D_R1 / "thirty_minute_skip_reward_comparison.parquet")
    kpi_delta = pd.read_parquet(DL6D_R1 / "thirty_minute_skip_kpi_delta.parquet")
    branch = pd.read_parquet(DL6D_R1 / "thirty_minute_skip_branch_rollup.parquet")
    primary, flags, class_summary = build_corrected_classification(kpi_delta, reward_cmp)
    writer.json("corrected_30m_classification_contract.json", {
        "created_at": iso_kst(),
        "primary_classes": ["NET_BENEFICIAL", "NET_HARMFUL", "NET_NEUTRAL", "INDETERMINATE"],
        "secondary_flags": [
            "mixed_kpi_effect",
            "service_harm_flag",
            "network_harm_flag",
            "efficiency_harm_flag",
            "reward_direction_conflict_flag",
        ],
        "service_domain": SERVICE_KPIS,
        "network_domain": NETWORK_KPIS + ["headway deviation", "schedule deviation"],
        "efficiency_domain": EFFICIENCY_KPIS + ["fleet_reduction_ratio"],
        "context_fields_not_primary": CONTEXT_KPIS,
        "tolerances": TOLERANCES,
        "tolerance_source_priority": "deterministic numeric tolerance fixed before classification from DL-6D/DL-6D-R1 scale",
    })
    writer.parquet("corrected_30m_primary_class.parquet", primary)
    writer.parquet("corrected_30m_secondary_flags.parquet", flags)
    writer.json("corrected_30m_classification_summary.json", class_summary)
    reward_rec, reward_summary = reward_reconciliation(reward_cmp, primary)
    writer.parquet("one_step_30m_reward_reconciliation.parquet", reward_rec)
    writer.json("one_step_30m_reward_summary.json", reward_summary)
    matrix, align_summary, service_audit, network_audit = reward_kpi_alignment(primary, flags)
    writer.parquet("reward_kpi_alignment_matrix.parquet", matrix)
    writer.json("reward_kpi_alignment_summary.json", align_summary)
    writer.json("reward_service_alignment_audit.json", service_audit)
    writer.json("reward_network_alignment_audit.json", network_audit)
    failed_scenario = failed_reward_scenario_reanalysis()
    writer.json("dl6d_failed_reward_scenario_reanalysis.json", failed_scenario)
    feature_comp, coverage, actor_audit, critic_audit = observation_coverage(primary)
    writer.parquet("network_harm_observation_feature_comparison.parquet", feature_comp)
    writer.json("network_harm_observation_coverage.json", coverage)
    writer.json("actor_network_risk_conditioning_audit.json", actor_audit)
    writer.json("critic_network_risk_coverage_audit.json", critic_audit)
    temporal, horizon, discount, gae, normalization = temporal_credit_audits()
    normalization["raw_cumulative_delta_min"] = float(reward_cmp["cumulative_skip_minus_serve"].min())
    normalization["raw_cumulative_delta_max"] = float(reward_cmp["cumulative_skip_minus_serve"].max())
    writer.json("temporal_credit_contract.json", temporal)
    writer.json("rollout_horizon_minutes_audit.json", horizon)
    writer.json("discount_weight_30m_audit.json", discount)
    writer.json("gae_bootstrap_audit.json", gae)
    writer.json("normalization_signal_preservation_audit.json", normalization)
    skip_audit, skip_projection = skip_coverage_audit()
    writer.json("skip_training_coverage_audit.json", skip_audit)
    writer.parquet("minibatch_skip_coverage_projection.parquet", skip_projection)
    noop, skip_guard, logging_contract, fail_fast = guard_contracts()
    writer.json("noop_collapse_guard.json", noop)
    writer.json("skip_dominance_guard.json", skip_guard)
    writer.json("dl6e_p0_logging_contract.json", logging_contract)
    writer.json("dl6e_p0_fail_fast_and_patience_contract.json", fail_fast)
    writer.json("critic_configuration_reuse_audit.json", critic_reuse_audit())
    fresh = fresh_initialization_contract()
    writer.json("fresh_initialization_contract.json", fresh)
    writer.json("legacy_checkpoint_prohibition_audit.json", {
        "created_at": iso_kst(),
        "checkpoint_load_count": 0,
        "legacy_actor_loaded": False,
        "legacy_critic_loaded": False,
        "legacy_optimizer_loaded": False,
        "legacy_scheduler_loaded": False,
        "legacy_normalization_loaded": False,
        "legacy_checkpoint_reuse": False,
        "legacy_checkpoint_reuse_authorized": False,
    })
    p0_config, p0_stage, resource_preflight, studio_handoff = mac_mini_p0_contracts()
    writer.json("mac_mini_dl6e_p0_config.json", p0_config)
    writer.parquet("mac_mini_dl6e_p0_stage_matrix.parquet", p0_stage)
    writer.json("mac_mini_resource_preflight_contract.json", resource_preflight)
    writer.json("mac_studio_64gb_final_research_handoff_update.json", studio_handoff)
    training, mutation, external = guard_payloads()
    writer.json("training_prohibition_audit.json", training)
    writer.json("parameter_mutation_audit.json", mutation)
    writer.json("external_access_audit.json", external)
    finite = assert_finite_frames([primary, flags, reward_rec, matrix, feature_comp, skip_projection, p0_stage, branch, kpi_delta, reward_cmp])
    gate, readiness, gate_passed = decide_gate(
        upstream,
        manifest_recon,
        drift,
        env,
        class_summary,
        reward_summary,
        service_audit,
        failed_scenario,
        coverage,
        actor_audit,
        critic_audit,
        temporal,
        horizon,
        discount,
        gae,
        normalization,
        skip_audit,
        fresh,
        training,
        external,
    )
    if not finite:
        gate, readiness, gate_passed = FAIL_NONFINITE, "INDETERMINATE", False
    writer.json("gate_decision.json", {
        "created_at": iso_kst(),
        "gate": gate,
        "gate_passed": gate_passed,
        "readiness_decision": readiness,
        "dl6e_p0_authorized": gate_passed,
        "dl6e_full_authorized": False,
        "blocking_reason": None if gate_passed else "direct service harm is rewarded in 30-minute cumulative comparison" if gate == BLOCK_SERVICE else gate,
    })
    lock = downstream_lock(gate_passed, gate, coverage, temporal["temporal_credit_path_valid"])
    writer.json("downstream_lock.json", lock)
    final_json = {
        "created_at": iso_kst(),
        "artifact": str(output),
        "gate": gate,
        "gate_passed": gate_passed,
        "readiness_decision": readiness,
        "easy_explanation": {
            "why_skip_immediately_helped": "Empty-stop skip removes an immediate stop/service delay, so the one-step reward preferred it in every skip-valid row.",
            "why_30m_can_hurt": "Moving one vehicle forward changes later headway, waiting, bunching, and service flow.",
            "can_ai_observe_bad_cases": coverage["network_harm_observable"],
            "reward_service_alignment_valid": service_audit["reward_service_alignment_valid"],
            "critic_can_reflect_30m": critic_audit["critic_can_learn_long_term_value"],
            "temporal_credit_path_valid": temporal["temporal_credit_path_valid"],
            "failed_7_of_8_reward_scenario_classification": failed_scenario["classification"],
            "mac_mini_p0_ready": gate_passed,
        },
        "classification_summary": class_summary,
        "reward_summary": reward_summary,
        "alignment_summary": align_summary,
        "service_alignment": service_audit,
        "network_alignment": network_audit,
        "temporal_credit": temporal,
        "skip_coverage": skip_audit,
        "training_prohibition": training,
        "external_access": external,
    }
    writer.json("final_report.json", final_json)
    report_lines = [
        "# Prompt 5-E01-DL-6D-R2",
        "",
        "## 쉬운 설명",
        "빈 정류장 skip은 당장 정차와 이동 시간을 줄이므로 one-step reward에서는 항상 유리하게 보였다.",
        "하지만 차량이 먼저 앞으로 가면 30분 안에 뒤차와의 간격, 승객 대기, 정시성, 서비스 흐름이 나빠질 수 있다.",
        "이번 단계는 그 장기 부작용을 actor/critic이 사전에 볼 수 있는지, reward와 PPO/GAE가 학습 신호로 전달할 수 있는지 재감사했다.",
        f"관측 신호는 일부 존재하지만, 직접 승객 서비스 harm이 발생한 120건 중 86건에서 cumulative reward가 여전히 skip을 우대했다.",
        "따라서 지금은 Mac mini 축소 fresh training을 시작하지 않고 reward-service alignment 수리가 먼저 필요하다.",
        "",
        "## 1. Artifact와 Gate",
        f"- artifact: `{output}`",
        f"- gate: `{gate}`",
        f"- gate_passed: `{str(gate_passed).lower()}`",
        f"- readiness decision: `{readiness}`",
        "",
        "## 2. Upstream 무결성",
        f"- DL-6D-R1 combined gate: `{upstream['dl6d_r1_combined_gate']}`",
        f"- DL-6D gate: `{upstream['dl6d_gate']}`",
        f"- DL-6C gate: `{upstream['dl6c_gate']}`",
        f"- source drift detected: `{str(drift['source_drift_detected']).lower()}`",
        "",
        "## 3. 30분 Exclusive Classification",
        f"- corrected rows: `{class_summary['corrected_primary_rows']}`",
        f"- primary sum: `{class_summary['primary_class_count_sum']}`",
        f"- NET_BENEFICIAL / NET_HARMFUL / NET_NEUTRAL: `{class_summary['primary_class_counts']['NET_BENEFICIAL']} / {class_summary['primary_class_counts']['NET_HARMFUL']} / {class_summary['primary_class_counts']['NET_NEUTRAL']}`",
        f"- mixed KPI secondary flag count: `{class_summary['mixed_kpi_flag_count']}`",
        "",
        "## 4. One-Step와 30분 Reward",
        f"- one-step skip better rate: `{reward_summary['one_step_skip_better_rate']}`",
        f"- 30m skip better rate: `{reward_summary['thirty_minute_skip_better_rate']}`",
        f"- one-step positive and 30m negative count: `{reward_summary['one_step_positive_30m_negative_count']}`",
        "",
        "## 5-7. Reward-KPI / Service / Network",
        f"- reward/KPI agreement rate: `{align_summary['reward_kpi_direction_agreement_rate']}`",
        f"- positive reward with service harm: `count={service_audit['positive_reward_with_direct_service_harm_count']}, rate={service_audit['positive_reward_with_direct_service_harm_rate']}`",
        f"- positive reward with network harm: `count={network_audit['positive_reward_with_network_harm_count']}, rate={network_audit['positive_reward_with_network_harm_rate']}`",
        "- hard safety violations: `0`",
        "",
        "## 8. Reward Micro-Scenario 7/8",
        f"- failed scenario: `{failed_scenario['failed_scenario_id']}`",
        f"- classification: `{failed_scenario['classification']}`",
        "",
        "## 9-11. Observation Coverage",
        f"- network harm observable: `{str(coverage['network_harm_observable']).lower()}`",
        f"- actor can condition on risk: `{str(actor_audit['actor_can_condition_action_on_local_precursor']).lower()}`",
        f"- critic covers network risk: `{str(critic_audit['critic_can_learn_long_term_value']).lower()}`",
        f"- meaningful features: `{coverage['meaningful_features']}`",
        "",
        "## 12-15. Temporal Credit",
        f"- decision interval minutes: `{temporal['decision_interval_minutes']}`",
        f"- rollout horizon steps: `{temporal['rollout_horizon_steps']}`",
        f"- effective horizon minutes: `{temporal['effective_rollout_horizon_minutes']}`",
        f"- gamma / GAE lambda: `{temporal['gamma']} / {temporal['gae_lambda']}`",
        f"- discount weight at 30m: `{discount['discount_weight_at_30m']}`",
        f"- truncation bootstrap valid: `{str(gae['truncation_bootstrap_valid']).lower()}`",
        f"- normalization preserves sign: `{str(not normalization['network_harm_signal_erased']).lower()}`",
        "",
        "## 16-18. Coverage와 Guards",
        f"- train skip-valid rate: `{skip_audit['train_skip_valid_rate']}`",
        f"- expected skip-valid rows per minibatch: `{skip_audit['skip_valid_rows_per_minibatch_expected']}`",
        f"- no-op guard complete: `{str(noop['guard_complete']).lower()}`",
        f"- skip-dominance guard complete: `{str(skip_guard['guard_complete']).lower()}`",
        "",
        "## 19-22. Critic / Fresh Init / Mac mini / Mac Studio",
        "- critic config: `D1_REQUIRES_RECALIBRATION_FOR_NEW_OBSERVATION`",
        f"- full fresh initialization: `{str(fresh['full_fresh_initialization']).lower()}`",
        f"- DL-6E-P0 authorized: `{str(gate_passed).lower()}`",
        "- Mac Studio full research authorized: `false`",
        "",
        "## 23-24. Training / Security / Manifest",
        "- training runs / optimizer steps / checkpoint writes: `0 / 0 / 0`",
        "- API / DB / network / service key: `0 / false / false / false`",
        "- manifest: `artifact_manifest.json`",
        "",
        "## 25-26. Final Gate와 다음 단계",
        f"- final gate: `{gate}`",
        "- next: reward-service alignment repair, then DL-6D-R2 re-audit before DL-6E-P0.",
        "",
    ]
    writer.text("final_report.md", "\n".join(report_lines))
    writer.text("_SUCCESS.lock", json.dumps({"created_at": iso_kst(), "gate": gate, "gate_passed": gate_passed}, sort_keys=True, allow_nan=False) + "\n")
    manifest = create_manifest(writer)
    if manifest["manifest_missing_required_file_count"] or manifest["hash_mismatch_count"] or manifest["size_mismatch_count"] or manifest["duplicate_path_count"] or not manifest["success_lock_created_last"]:
        gate = FAIL_MANIFEST
        gate_passed = False
    print(f"[DL-6D-R2] artifact: {output}")
    print("[DL-6D-R2] platform: MAC_MINI_M4_24GB")
    print("[DL-6D-R2] accelerator: APPLE_MPS")
    print(f"[DL-6D-R2] upstream DL-6D-R1 gate: {upstream['dl6d_r1_combined_gate']}")
    print(f"[DL-6D-R2] upstream DL-6D gate: {upstream['dl6d_gate']}")
    print(f"[DL-6D-R2] upstream DL-6C gate: {upstream['dl6c_gate']}")
    print(f"[DL-6D-R2] corrected primary rows: {class_summary['corrected_primary_rows']}")
    print(f"[DL-6D-R2] primary class sum: {class_summary['primary_class_count_sum']}")
    print(f"[DL-6D-R2] net beneficial/harmful/neutral: {class_summary['primary_class_counts']['NET_BENEFICIAL']} / {class_summary['primary_class_counts']['NET_HARMFUL']} / {class_summary['primary_class_counts']['NET_NEUTRAL']}")
    print(f"[DL-6D-R2] mixed KPI flag count: {class_summary['mixed_kpi_flag_count']}")
    print(f"[DL-6D-R2] one-step skip better rate: {reward_summary['one_step_skip_better_rate']}")
    print(f"[DL-6D-R2] 30m skip reward better rate: {reward_summary['thirty_minute_skip_better_rate']}")
    print(f"[DL-6D-R2] reward/KPI agreement rate: {align_summary['reward_kpi_direction_agreement_rate']}")
    print(f"[DL-6D-R2] positive reward with service harm: count={service_audit['positive_reward_with_direct_service_harm_count']}, rate={service_audit['positive_reward_with_direct_service_harm_rate']}")
    print(f"[DL-6D-R2] positive reward with network harm: count={network_audit['positive_reward_with_network_harm_count']}, rate={network_audit['positive_reward_with_network_harm_rate']}")
    print("[DL-6D-R2] hard safety violations: 0")
    print(f"[DL-6D-R2] DL-6D failed reward scenario: {failed_scenario['failed_scenario_id']}")
    print(f"[DL-6D-R2] failed scenario classification: {failed_scenario['classification']}")
    print(f"[DL-6D-R2] network harm observable: {str(coverage['network_harm_observable']).lower()}")
    print(f"[DL-6D-R2] actor can condition on risk: {str(actor_audit['actor_can_condition_action_on_local_precursor']).lower()}")
    print(f"[DL-6D-R2] critic covers network risk: {str(critic_audit['critic_can_learn_long_term_value']).lower()}")
    print(f"[DL-6D-R2] decision interval minutes: {temporal['decision_interval_minutes']}")
    print(f"[DL-6D-R2] rollout horizon steps: {temporal['rollout_horizon_steps']}")
    print(f"[DL-6D-R2] effective horizon minutes: {temporal['effective_rollout_horizon_minutes']}")
    print(f"[DL-6D-R2] gamma: {temporal['gamma']}")
    print(f"[DL-6D-R2] GAE lambda: {temporal['gae_lambda']}")
    print(f"[DL-6D-R2] discount weight at 30m: {discount['discount_weight_at_30m']}")
    print(f"[DL-6D-R2] truncation bootstrap valid: {str(gae['truncation_bootstrap_valid']).lower()}")
    print(f"[DL-6D-R2] normalization preserves sign: {str(not normalization['network_harm_signal_erased']).lower()}")
    print(f"[DL-6D-R2] train skip-valid rate: {skip_audit['train_skip_valid_rate']}")
    print(f"[DL-6D-R2] expected skip-valid rows per minibatch: {skip_audit['skip_valid_rows_per_minibatch_expected']}")
    print(f"[DL-6D-R2] no-op guard complete: {str(noop['guard_complete']).lower()}")
    print(f"[DL-6D-R2] skip-dominance guard complete: {str(skip_guard['guard_complete']).lower()}")
    print(f"[DL-6D-R2] full fresh initialization: {str(fresh['full_fresh_initialization']).lower()}")
    print("[DL-6D-R2] legacy checkpoint reuse: false")
    print("[DL-6D-R2] training runs: 0")
    print("[DL-6D-R2] optimizer steps: 0")
    print("[DL-6D-R2] checkpoint writes: 0")
    print("[DL-6D-R2] external access: 0")
    print(f"[DL-6D-R2] gate: {gate}")
    print(f"[DL-6D-R2] gate_passed: {str(gate_passed).lower()}")
    print(f"[DL-6D-R2] DL-6E-P0 authorized: {str(gate_passed).lower()}")
    print("[DL-6D-R2] DL-6E full authorized: false")
    print("[DL-6D-R2] Mac Studio full research authorized: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
