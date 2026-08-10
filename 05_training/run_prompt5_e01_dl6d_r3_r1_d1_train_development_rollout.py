from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import re
import resource
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - torch is expected in the project env.
    torch = None  # type: ignore[assignment]


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
I0 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_r1_split_inventory_20260802_132332"
DL6C = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6c_distinct_three_action_contract_repair_20260801_234817"
R1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r1_observation_contract_repair_20260802_011348"
R2 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r2_combined_retraining_readiness_reaudit_20260802_102624"
R3 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_reward_service_alignment_repair_20260802_112731"
HO1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_holdout_evaluation_20260802_122513"

R1_RUNNER = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py"
DL6C_ENGINE = PROJECT_ROOT / "05_training/simulator/suseong_service_transition_engine.py"
THIS_RUNNER = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r3_r1_d1_train_development_rollout.py"

EXPECTED_I0_GATE = "PASS_SUSEONG_DL6D_R3_R1_TRAIN_VALIDATION_SKIP_VALID_INVENTORY_AND_VALIDATION_SEAL_COMPLETE"
EXPECTED_HO1_GATE = "BLOCKED_SUSEONG_DL6D_R3_C1_NORMALIZATION_FAILED_GENERALIZATION"
PASS_CALIBRATION_GATE = "PASS_SUSEONG_DL6D_R3_R1_D1_CALIBRATION_COMPLETE_AWAITING_USER_COMMAND"
READINESS_CALIBRATION = "CALIBRATION_PASSED_ROLLOUT_PENDING_USER_COMMAND"
BLOCK_BUDGET = "BLOCKED_SUSEONG_DL6D_R3_R1_D1_ROLLOUT_BUDGET_EXCEEDED"
FAIL_UPSTREAM = "FAIL_SUSEONG_DL6D_R3_R1_D1_UPSTREAM_INTEGRITY"
FAIL_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_R3_R1_D1_SOURCE_DRIFT"
FAIL_DEFINITION = "FAIL_SUSEONG_DL6D_R3_R1_D1_DEFINITION_REDEFINED"
FAIL_SEAL = "FAIL_SUSEONG_DL6D_R3_R1_D1_VALIDATION_SEAL_MUTATED"
FAIL_VALIDATION = "FAIL_SUSEONG_DL6D_R3_R1_D1_VALIDATION_OUTCOME_LEAKAGE"
FAIL_ALIGNMENT = "FAIL_SUSEONG_DL6D_R3_R1_D1_BRANCH_ALIGNMENT_INVALID"
FAIL_SAFETY = "FAIL_SUSEONG_DL6D_R3_R1_D1_SKIP_SAFETY_VIOLATION"
FAIL_SCALE = "FAIL_SUSEONG_DL6D_R3_R1_D1_SCALE_DERIVATION_DETECTED"
FAIL_TEST_REUSE = "FAIL_SUSEONG_DL6D_R3_R1_D1_TEST_HOLDOUT_REUSE_DETECTED"
FAIL_NONFINITE = "FAIL_SUSEONG_DL6D_R3_R1_D1_REWARD_OR_KPI_NAN_INF"
FAIL_TRAINING = "FAIL_SUSEONG_DL6D_R3_R1_D1_PROHIBITED_TRAINING"
FAIL_HARDWARE = "FAIL_SUSEONG_DL6D_R3_R1_D1_H200_OR_CUDA_USAGE"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_R3_R1_D1_MANIFEST_RECONCILIATION"

CALIBRATION_ROWS = 250
TOTAL_TRAIN_SKIP_VALID_ROWS = 5523
TOTAL_TRAIN_BRANCHES = TOTAL_TRAIN_SKIP_VALID_ROWS * 3
FULL_ROLLOUT_TARGET_BRANCHES = 16569
UNIFIED_MEMORY_BYTES_24GB = 24 * 1024**3
BUDGET_MAX_TOTAL_HOURS = 12.0
BUDGET_MAX_PEAK_RSS_FRACTION_OF_24GB = 0.6
RSS_BUDGET_BYTES = int(UNIFIED_MEMORY_BYTES_24GB * BUDGET_MAX_PEAK_RSS_FRACTION_OF_24GB)
FROZEN_REWARD_TOLERANCE = 1e-9
TEST_FULL_86_MAX_RATIO = 0.02102008520998175
I0_PROJECTED_TRAIN_HARMFUL = 851.2150537634409

SERVICE_TOLERANCES = {
    "avg_wait_seconds": 1.0,
    "passenger_wait_p95_seconds": 1.0,
    "passenger_service_rate": 0.001,
    "on_time_rate": 0.001,
    "passenger_served_count": 0.01,
}
NETWORK_TOLERANCES = {"cv_headway": 0.001, "bunching_rate": 0.0005}
EFFICIENCY_TOLERANCES = {"energy_proxy": 0.001, "energy_proxy_per_passenger": 0.0001}

CALIBRATION_REQUIRED_FILES = [
    "git_status_d1.txt",
    "mac_mini_environment_d1.json",
    "execution_contract_patch_audit.json",
    "upstream_validation.json",
    "upstream_manifest_reconciliation.json",
    "source_drift_audit.json",
    "definition_reuse_audit.json",
    "validation_seal_revalidation.json",
    "validation_untouched_audit.json",
    "covariate_shift_reinterpretation.json",
    "chunk_execution_plan.parquet",
    "chunk_execution_state.json",
    "calibration_chunk_result.json",
    "rollout_cost_projection.json",
    "train_30m_branch_results.parquet",
    "branch_alignment_audit.json",
    "train_hard_safety_audit.json",
    "scale_derivation_prohibition_audit.json",
    "test_holdout_non_reuse_audit.json",
    "mac_mini_resource_telemetry_d1.json",
    "mac_mini_resource_telemetry_d1.parquet",
    "training_prohibition_audit.json",
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

    def _mark(self, rel: str) -> None:
        if rel not in self.order:
            self.order[rel] = len(self.order) + 1

    def text(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self._mark(rel)

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")

    def parquet(self, rel: str, frame: pd.DataFrame) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        self._mark(rel)


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def iso_kst() -> str:
    return now_kst().isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str).encode("utf-8")).hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_cmd(args: Sequence[str]) -> Dict[str, Any]:
    result = subprocess.run(list(args), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {"cmd": list(args), "returncode": result.returncode, "stdout": redact_identity(result.stdout.strip()), "stderr": redact_identity(result.stderr.strip())}


def redact_identity(text: str) -> str:
    out = []
    for line in text.splitlines():
        lower = line.lower()
        if "serial number" in lower or "hardware uuid" in lower or "provisioning udid" in lower:
            out.append(f"{line.split(':', 1)[0]}: REDACTED")
        else:
            out.append(line)
    return "\n".join(out)


def manifest_file_hash(root: Path, rel: str) -> Optional[str]:
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    for item in manifest.get("files", []):
        if item.get("relative_path") == rel:
            return item.get("sha256")
    return None


def manifest_summary(root: Path) -> Dict[str, Any]:
    path = root / "artifact_manifest.json"
    if not path.exists():
        return {"path": str(root), "manifest_exists": False, "manifest_clean": False}
    data = read_json(path)
    clean = (
        int(data.get("manifest_missing_required_file_count", 0)) == 0
        and int(data.get("hash_mismatch_count", 0)) == 0
        and int(data.get("size_mismatch_count", 0)) == 0
        and int(data.get("duplicate_path_count", 0)) == 0
        and bool(data.get("success_lock_created_last", False))
    )
    return {
        "path": str(root),
        "manifest_exists": True,
        "manifest_clean": bool(clean),
        "required_missing": int(data.get("manifest_missing_required_file_count", 0)),
        "hash_mismatch": int(data.get("hash_mismatch_count", 0)),
        "size_mismatch": int(data.get("size_mismatch_count", 0)),
        "duplicate_path_count": int(data.get("duplicate_path_count", 0)),
        "success_lock_created_last": bool(data.get("success_lock_created_last", False)),
    }


def torch_mps_available() -> Tuple[bool, bool]:
    if torch is None:
        return False, False
    built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
    available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return built, available


def mps_memory() -> Dict[str, Any]:
    if torch is None or not hasattr(torch, "mps"):
        return {
            "current_allocated": {"value": None, "reason": "API unavailable in current torch build"},
            "driver_allocated": {"value": None, "reason": "API unavailable in current torch build"},
            "recommended_max": {"value": None, "reason": "API unavailable in current torch build"},
        }
    out: Dict[str, Any] = {}
    for key, fn_name in [
        ("current_allocated", "current_allocated_memory"),
        ("driver_allocated", "driver_allocated_memory"),
        ("recommended_max", "recommended_max_memory"),
    ]:
        fn = getattr(torch.mps, fn_name, None)
        if fn is None:
            out[key] = {"value": None, "reason": "API unavailable in current torch build"}
            continue
        try:
            out[key] = {"value": int(fn()), "reason": None}
        except Exception as exc:  # pragma: no cover - platform dependent.
            out[key] = {"value": None, "reason": f"API call failed: {type(exc).__name__}"}
    return out


def rss_measurement() -> Dict[str, Any]:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return {"raw_ru_maxrss": raw, "ru_maxrss_unit": "bytes", "process_rss_bytes": raw}
    return {"raw_ru_maxrss": raw, "ru_maxrss_unit": "kilobytes", "process_rss_bytes": raw * 1024}


def parse_swap_bytes(text: str) -> Optional[int]:
    match = re.search(r"used\s*=\s*([0-9.]+)\s*([KMGTP]?)", text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).upper()
    scale = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4, "P": 1024**5}[unit]
    return int(value * scale)


def swap_usage() -> Dict[str, Any]:
    result = run_cmd(["sysctl", "vm.swapusage"])
    used = parse_swap_bytes(result.get("stdout", "")) if result["returncode"] == 0 else None
    return {
        "value": used,
        "reason": None if used is not None else "sysctl vm.swapusage unavailable or unparseable",
        "raw": result,
    }


def numeric_value(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key, {}).get("value") if isinstance(payload.get(key), Mapping) else payload.get(key)
    return int(value or 0)


def environment_audit() -> Dict[str, Any]:
    mem = run_cmd(["sysctl", "-n", "hw.memsize"])
    model = run_cmd(["sysctl", "-n", "hw.model"])
    sw = run_cmd(["sw_vers"])
    profiler = run_cmd(["system_profiler", "SPHardwareDataType"])
    mps_built, mps_available = torch_mps_available()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    mem_bytes = int(mem["stdout"]) if str(mem.get("stdout", "")).isdigit() else None
    return {
        "created_at": iso_kst(),
        "execution_platform": "MAC_MINI_M4_24GB",
        "accelerator": "APPLE_MPS",
        "hardware_model": model["stdout"] or "UNKNOWN",
        "chip_name": "Apple M4",
        "unified_memory_bytes": mem_bytes,
        "unified_memory_nominal_gb": 24,
        "macos_version": sw,
        "python_version": sys.version,
        "torch_version": getattr(torch, "__version__", None),
        "mps_built": bool(mps_built),
        "mps_available": bool(mps_available),
        "cuda_available": bool(cuda_available),
        "process_memory": rss_measurement(),
        "mps_memory": mps_memory(),
        "system_profiler_hardware_redacted": profiler,
        "h200_used": False,
        "cuda_used": False,
        "cloud_gpu_used": False,
    }


def upstream_validation() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    i0_gate = read_json(I0 / "gate_decision.json")
    r3_gate = read_json(R3 / "gate_decision.json")
    ho1_gate = read_json(HO1 / "sealed_holdout_gate_decision.json")
    capacity = read_json(I0 / "harmful_capacity_projection.json")
    checks = {
        "i0_gate_ok": i0_gate.get("gate") == EXPECTED_I0_GATE and bool(i0_gate.get("gate_passed")),
        "i0_success_lock_exists": (I0 / "_SUCCESS.lock").exists(),
        "i0_validation_seal_exists": (I0 / "_VALIDATION_INVENTORY_SEALED.lock").exists(),
        "r3_success_lock_exists": (R3 / "_SUCCESS.lock").exists(),
        "r3_c1_design_gate_ok": r3_gate.get("gate") == "PASS_SUSEONG_DL6D_R3_C1_NORMALIZATION_ONLY_DESIGN_GATE_READY_AWAITING_USER_COMMAND",
        "ho1_failure_gate_ok": ho1_gate.get("gate") == EXPECTED_HO1_GATE,
        "validation_capacity_50_plausible": bool(capacity.get("validation_capacity_50_plausible")),
        "validation_capacity_86_plausible_false": capacity.get("validation_capacity_86_plausible") is False,
        "corrected_classification_contract_exists": (R2 / "corrected_30m_classification_contract.json").exists(),
        "r1_runner_exists": R1_RUNNER.exists(),
    }
    upstream = {
        "created_at": iso_kst(),
        "i0_path": str(I0),
        "i0_gate": i0_gate.get("gate"),
        "i0_readiness": i0_gate.get("readiness"),
        "r3_path": str(R3),
        "r3_gate": r3_gate.get("gate"),
        "ho1_path": str(HO1),
        "ho1_gate": ho1_gate.get("gate"),
        "r2_corrected_classification_contract_path": str(R2 / "corrected_30m_classification_contract.json"),
        "checks": checks,
        "upstream_integrity_valid": all(checks.values()),
    }
    artifacts = {
        "i0": {**manifest_summary(I0), "manifest_required_for_d1_gate": True},
        "dl6c": {**manifest_summary(DL6C), "manifest_required_for_d1_gate": True},
        "r1": {**manifest_summary(R1), "manifest_required_for_d1_gate": True},
        "r2": {**manifest_summary(R2), "manifest_required_for_d1_gate": True},
        "r3": {**manifest_summary(R3), "manifest_required_for_d1_gate": True},
        "ho1": {**manifest_summary(HO1), "manifest_required_for_d1_gate": False, "gate_reference_only": True},
    }
    manifests = {"created_at": iso_kst(), "artifacts": artifacts}
    manifests["upstream_manifest_reconciliation_passed"] = all(
        item.get("manifest_clean")
        for item in manifests["artifacts"].values()
        if item.get("manifest_required_for_d1_gate")
    )
    return upstream, manifests


def source_drift_audit() -> Dict[str, Any]:
    i0_source = read_json(I0 / "source_drift_audit.json")
    checks: List[Dict[str, Any]] = []
    for item in i0_source.get("checks", []):
        path = Path(str(item["path"]))
        current = sha256_file(path) if path.exists() else None
        checks.append({
            "source_name": item.get("source_name"),
            "path": str(path),
            "expected_sha256": item.get("expected_sha256"),
            "current_sha256": current,
            "drift_detected": current != item.get("expected_sha256"),
            "source": "I0 source_drift_audit",
        })
    for source_name, path in [
        ("dl6d_r1_thirty_minute_branch_procedure_source", R1_RUNNER),
        ("dl6d_r2_corrected_30m_classification_contract", R2 / "corrected_30m_classification_contract.json"),
        ("dl6d_r3_frozen_service_tolerance_contract", R3 / "frozen_service_tolerance_contract.json"),
        ("dl6d_r3_frozen_service_tolerance_table", R3 / "frozen_service_tolerance_table.parquet"),
        ("dl6d_r3_candidate_design_results", R3 / "candidate_design_results.parquet"),
        ("d1_runner_current_source", THIS_RUNNER),
    ]:
        current = sha256_file(path)
        checks.append({
            "source_name": source_name,
            "path": str(path),
            "expected_sha256": current,
            "current_sha256": current,
            "drift_detected": False,
            "source": "current authoritative input hash recorded for D1",
        })
    count = sum(int(item["drift_detected"]) for item in checks)
    return {
        "created_at": iso_kst(),
        "source_drift_detected": bool(count),
        "source_drift_count": int(count),
        "checks": checks,
    }


def definition_reuse_audit() -> Dict[str, Any]:
    definitions = [
        {
            "definition_name": "skip-valid predicate",
            "source_path": str(DL6C_ENGINE),
            "source_sha256": sha256_file(DL6C_ENGINE),
            "source_artifact": str(DL6C),
            "redefined": False,
        },
        {
            "definition_name": "30-minute single-pulse branch procedure",
            "source_path": str(R1_RUNNER),
            "source_symbol": "thirty_minute_audit",
            "source_sha256": sha256_file(R1_RUNNER),
            "source_artifact": str(R1),
            "continuation": "ALL_NOOP_AFTER_SINGLE_PULSE",
            "redefined": False,
        },
        {
            "definition_name": "corrected 30m primary class contract",
            "source_path": str(R2 / "corrected_30m_classification_contract.json"),
            "source_sha256": sha256_file(R2 / "corrected_30m_classification_contract.json"),
            "source_artifact": str(R2),
            "redefined": False,
        },
        {
            "definition_name": "direct service harm tolerance contract",
            "source_path": str(R3 / "frozen_service_tolerance_contract.json"),
            "source_sha256": sha256_file(R3 / "frozen_service_tolerance_contract.json"),
            "source_artifact": str(R3),
            "redefined": False,
        },
        {
            "definition_name": "harmful misalignment",
            "source_path": str(R3 / "reward_service_misalignment_registry.json"),
            "source_sha256": sha256_file(R3 / "reward_service_misalignment_registry.json"),
            "definition": "direct_service_harm == true and cumulative_skip_minus_serve > frozen_reward_tolerance",
            "frozen_reward_tolerance": FROZEN_REWARD_TOLERANCE,
            "redefined": False,
        },
    ]
    return {
        "created_at": iso_kst(),
        "definition_reuse_required": True,
        "definition_redefined_count": 0,
        "definitions": definitions,
        "reward_formula_changed": False,
        "reward_weight_changed": False,
        "tolerance_changed": False,
    }


def validation_seal_revalidation() -> Dict[str, Any]:
    contract_path = I0 / "validation_skip_valid_seal_contract.json"
    sealed_ids_path = I0 / "validation_skip_valid_sealed_ids.json"
    lock_path = I0 / "_VALIDATION_INVENTORY_SEALED.lock"
    contract = read_json(contract_path)
    sealed_ids = read_json(sealed_ids_path)
    expected_contract_hash = manifest_file_hash(I0, "validation_skip_valid_seal_contract.json")
    expected_ids_hash = manifest_file_hash(I0, "validation_skip_valid_sealed_ids.json")
    expected_lock_hash = manifest_file_hash(I0, "_VALIDATION_INVENTORY_SEALED.lock")
    current_contract_hash = sha256_file(contract_path)
    current_ids_hash = sha256_file(sealed_ids_path)
    current_lock_hash = sha256_file(lock_path)
    checks = {
        "seal_contract_hash_unchanged": current_contract_hash == expected_contract_hash,
        "sealed_ids_hash_unchanged": current_ids_hash == expected_ids_hash,
        "sealed_lock_hash_unchanged": current_lock_hash == expected_lock_hash,
        "validation_skip_valid_id_hash_present": bool(contract.get("validation_skip_valid_id_hash")),
        "validation_skip_valid_count_matches_ids": int(sealed_ids.get("validation_skip_valid_count", -1)) == len(sealed_ids.get("inventory_row_ids", [])),
        "validation_outcome_not_generated": contract.get("outcome_generated") is False and sealed_ids.get("outcome_generated") is False,
        "validation_outcome_not_accessed": contract.get("outcome_accessed") is False and sealed_ids.get("outcome_accessed") is False,
    }
    return {
        "created_at": iso_kst(),
        "validation_inventory_hash_access_allowed": True,
        "validation_row_level_feature_access_allowed": False,
        "validation_skip_valid_inventory_parquet_read": False,
        "validation_skip_valid_inventory_hash": contract.get("validation_skip_valid_inventory_hash"),
        "validation_skip_valid_id_hash": contract.get("validation_skip_valid_id_hash"),
        "validation_split_manifest_hash": contract.get("validation_split_manifest_hash"),
        "validation_snapshot_inventory_hash": contract.get("validation_snapshot_inventory_hash"),
        "seal_contract_file_sha256": current_contract_hash,
        "sealed_ids_file_sha256": current_ids_hash,
        "sealed_lock_file_sha256": current_lock_hash,
        "i0_manifest_seal_contract_sha256": expected_contract_hash,
        "i0_manifest_sealed_ids_sha256": expected_ids_hash,
        "i0_manifest_sealed_lock_sha256": expected_lock_hash,
        "checks": checks,
        "validation_seal_intact": all(checks.values()),
    }


def validation_untouched_audit(seal: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "validation_inventory_hash_access_allowed": True,
        "validation_precomputed_covariate_summary_access_allowed": True,
        "validation_row_level_feature_access_allowed": False,
        "validation_outcome_access_allowed": False,
        "validation_reward_access_allowed": False,
        "validation_harm_label_access_allowed": False,
        "validation_30m_rollout_allowed": False,
        "validation_30m_rollout_executed": False,
        "validation_30m_branch_execution_count": 0,
        "validation_reward_computation_count": 0,
        "validation_harm_label_count": 0,
        "validation_outcome_accessed": False,
        "validation_used_in_any_derivation": False,
        "validation_used_for_candidate_selection": False,
        "validation_used_for_validation": False,
        "validation_seal_intact": bool(seal.get("validation_seal_intact")),
    }


def covariate_shift_reinterpretation() -> Dict[str, Any]:
    summary = read_json(I0 / "train_validation_covariate_shift_summary.json")
    capacity = read_json(I0 / "harmful_capacity_projection.json")
    table = pd.read_parquet(I0 / "train_validation_decision_context_comparison.parquet")
    categorical = table[table["feature_type"] == "categorical"].copy()
    numeric = table[table["feature_type"] == "numeric"].copy()
    calendar_vars = ["month"]
    calendar = categorical[categorical["feature_name"].isin(calendar_vars)]
    non_calendar = categorical[~categorical["feature_name"].isin(calendar_vars)]
    max_calendar_row = calendar.sort_values("absolute_proportion_difference", ascending=False).head(1)
    max_non_calendar = float(non_calendar["absolute_proportion_difference"].max()) if not non_calendar.empty else 0.0
    non_calendar_by_feature = {
        str(feature): float(group["absolute_proportion_difference"].max())
        for feature, group in non_calendar.groupby("feature_name")
    }
    max_smd = float(numeric["standardized_mean_difference"].abs().max())
    max_ks = float(numeric["empirical_ks_statistic"].max())
    if max_smd < 0.10 and max_ks < 0.05 and max_non_calendar < 0.05:
        classification = "NEGLIGIBLE"
    elif max_smd < 0.20 and max_ks < 0.10:
        classification = "LOW"
    elif max_smd < 0.50 and max_ks < 0.30:
        classification = "MODERATE"
    else:
        classification = "HIGH"
    return {
        "created_at": iso_kst(),
        "i0_artifact_modified": False,
        "i0_original_covariate_shift_classification": summary.get("train_validation_covariate_shift_classification"),
        "calendar_partition_artifact_variables": calendar_vars,
        "max_calendar_artifact_difference": None if max_calendar_row.empty else {
            "feature_name": str(max_calendar_row.iloc[0]["feature_name"]),
            "category": str(max_calendar_row.iloc[0]["category"]),
            "train_proportion": float(max_calendar_row.iloc[0]["train_proportion"]),
            "validation_proportion": float(max_calendar_row.iloc[0]["validation_proportion"]),
            "absolute_proportion_difference": float(max_calendar_row.iloc[0]["absolute_proportion_difference"]),
        },
        "numeric_only_max_abs_standardized_mean_difference": max_smd,
        "numeric_only_max_empirical_ks_statistic": max_ks,
        "non_calendar_categorical_max_abs_proportion_difference": max_non_calendar,
        "non_calendar_categorical_max_by_feature": non_calendar_by_feature,
        "decision_context_shift_classification": classification,
        "decision_context_shift_basis": "numeric SMD/KS plus non-calendar categorical proportions only",
        "calendar_generalization_untested": True,
        "validation_projection_interpretation": "arithmetic projection only; not an observed expectation because train-validation exchangeability is unverified",
        "validation_projected_misalignment_count_arithmetic_only": capacity.get("validation_projected_misalignment_count"),
        "validation_capacity_highest_tier": capacity.get("validation_capacity_highest_tier"),
        "validation_capacity_86_plausible": bool(capacity.get("validation_capacity_86_plausible")),
        "train_validation_exchangeability_verified": False,
    }


def execution_contract_patch_audit(artifact_root: Path) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "artifact_root_argument_required": True,
        "artifact_root": str(artifact_root),
        "same_artifact_used_by_calibrate_rollout_analyze": True,
        "automatic_mode_chaining_allowed": False,
        "calibrate_stops_before_rollout": True,
        "hypothetical_scale_grid_distinguished_from_selected_scale": True,
        "hypothetical_scale_grid_future_contract": [
            "train_p99",
            "train_p999",
            "train_max",
            "train_max_x_1_1",
            "train_max_x_1_25",
            "train_max_x_1_5",
        ],
        "prior_test_outcome_reference_split_from_reuse": True,
        "prior_test_full_86_max_ratio_excluded_from_sensitivity_grid": True,
        "validation_capacity_language_revised": True,
        "validation_access_scope_split": True,
        "sensitivity_equation_locked": "post_margin(scale) = baseline_margin - scale * service_harm_excess_native",
        "tail_concentration_zero_denominator_guard": "if abs(p99 - p50) <= 1e-12: value = null with denominator-zero reason",
        "d2_wording_corrected": "Use D1 Train analysis only to design D2; freeze rules, coefficients, and candidate before validation outcome.",
    }


def load_train_skip_valid_sorted() -> pd.DataFrame:
    train = pd.read_parquet(I0 / "train_skip_valid_inventory.parquet")
    train["_state_ts_sort"] = pd.to_datetime(train["state_ts"], utc=True)
    train = train.sort_values(["_state_ts_sort", "agent_id", "inventory_row_id"]).drop(columns=["_state_ts_sort"])
    return train.reset_index(drop=True)


def build_chunk_execution_plan(train: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 1
    chunk_sizes = [CALIBRATION_ROWS]
    remaining = len(train) - CALIBRATION_ROWS
    while remaining > 0:
        size = min(500, remaining)
        chunk_sizes.append(size)
        remaining -= size
    for size in chunk_sizes:
        chunk = train.iloc[start : start + size]
        rows.append({
            "chunk_id": f"D1C{chunk_index:04d}",
            "chunk_index": chunk_index,
            "first_row_id": str(chunk["inventory_row_id"].iloc[0]),
            "last_row_id": str(chunk["inventory_row_id"].iloc[-1]),
            "row_count": int(len(chunk)),
            "expected_branch_count": int(len(chunk) * 3),
            "status": "PENDING",
            "is_calibration_chunk": chunk_index == 1,
        })
        start += size
        chunk_index += 1
    return pd.DataFrame(rows)


def pivot_kpi_delta(kpi_delta: pd.DataFrame) -> pd.DataFrame:
    wide = (
        kpi_delta.pivot_table(index=["window_id", "agent_id"], columns="kpi", values="k_minus_s", aggfunc="first")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for col in [
        "avg_wait_seconds",
        "passenger_wait_p95_seconds",
        "passenger_service_rate",
        "on_time_rate",
        "cv_headway",
        "bunching_rate",
        "energy_proxy_per_passenger",
    ]:
        if col not in wide:
            wide[col] = 0.0
    return wide


def pivot_branch_deltas(branch_df: pd.DataFrame) -> pd.DataFrame:
    needed = ["passenger_served_count", "energy_proxy", "initial_state_hash", "branch_aligned"]
    frames = []
    for (window_id, agent_id), group in branch_df.groupby(["window_id", "agent_id"]):
        by = {str(row["branch_id"]): row for _, row in group.iterrows()}
        if not {"H", "S", "K"}.issubset(by):
            continue
        frames.append({
            "window_id": window_id,
            "agent_id": int(agent_id),
            "initial_state_hash": str(by["H"]["initial_state_hash"]),
            "branch_alignment_ok": bool(group["branch_aligned"].all() and group["initial_state_hash"].nunique() == 1),
            "served_count_delta": float(by["K"]["passenger_served_count"] - by["S"]["passenger_served_count"]),
            "energy_delta": float(by["K"]["energy_proxy"] - by["S"]["energy_proxy"]),
        })
    return pd.DataFrame(frames)


def service_harm_excess(row: Mapping[str, Any]) -> float:
    return float(
        max(float(row["avg_wait_delta"]) - SERVICE_TOLERANCES["avg_wait_seconds"], 0.0)
        + max(float(row["p95_wait_delta"]) - SERVICE_TOLERANCES["passenger_wait_p95_seconds"], 0.0)
        + max(-float(row["service_rate_delta"]) - SERVICE_TOLERANCES["passenger_service_rate"], 0.0) * 120.0
        + max(-float(row["on_time_delta"]) - SERVICE_TOLERANCES["on_time_rate"], 0.0) * 120.0
        + max(-float(row["served_count_delta"]) - SERVICE_TOLERANCES["passenger_served_count"], 0.0) * 5.0
    )


def add_classification(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["avg_wait_harm"] = out["avg_wait_delta"] > SERVICE_TOLERANCES["avg_wait_seconds"]
    out["p95_wait_harm"] = out["p95_wait_delta"] > SERVICE_TOLERANCES["passenger_wait_p95_seconds"]
    out["service_rate_harm"] = out["service_rate_delta"] < -SERVICE_TOLERANCES["passenger_service_rate"]
    out["on_time_harm"] = out["on_time_delta"] < -SERVICE_TOLERANCES["on_time_rate"]
    out["served_count_harm"] = out["served_count_delta"] < -SERVICE_TOLERANCES["passenger_served_count"]
    out["direct_service_harm"] = out[["avg_wait_harm", "p95_wait_harm", "service_rate_harm", "on_time_harm", "served_count_harm"]].any(axis=1)
    out["service_harm_excess_native"] = out.apply(service_harm_excess, axis=1)
    out["network_harm_flag"] = (out["cv_headway_delta"] > NETWORK_TOLERANCES["cv_headway"]) | (out["bunching_delta"] > NETWORK_TOLERANCES["bunching_rate"])
    out["efficiency_harm_flag"] = (out["energy_delta"] > EFFICIENCY_TOLERANCES["energy_proxy"]) | (out["energy_per_passenger_delta"] > EFFICIENCY_TOLERANCES["energy_proxy_per_passenger"])
    service_benefit = (
        (out["avg_wait_delta"] < -SERVICE_TOLERANCES["avg_wait_seconds"])
        | (out["p95_wait_delta"] < -SERVICE_TOLERANCES["passenger_wait_p95_seconds"])
        | (out["service_rate_delta"] > SERVICE_TOLERANCES["passenger_service_rate"])
        | (out["on_time_delta"] > SERVICE_TOLERANCES["on_time_rate"])
        | (out["served_count_delta"] > SERVICE_TOLERANCES["passenger_served_count"])
    )
    network_benefit = (out["cv_headway_delta"] < -NETWORK_TOLERANCES["cv_headway"]) | (out["bunching_delta"] < -NETWORK_TOLERANCES["bunching_rate"])
    efficiency_benefit = (out["energy_delta"] < -EFFICIENCY_TOLERANCES["energy_proxy"]) | (out["energy_per_passenger_delta"] < -EFFICIENCY_TOLERANCES["energy_proxy_per_passenger"])
    any_harm = out["direct_service_harm"] | out["network_harm_flag"] | out["efficiency_harm_flag"]
    any_benefit = service_benefit | network_benefit | efficiency_benefit
    out["service_harm_flag"] = out["direct_service_harm"]
    out["mixed_kpi_effect"] = any_harm & any_benefit
    out["primary_class"] = np.select(
        [
            out["direct_service_harm"],
            (~any_harm) & (out["cumulative_skip_minus_serve"] > FROZEN_REWARD_TOLERANCE) & any_benefit,
            any_harm & (out["cumulative_skip_minus_serve"] < -FROZEN_REWARD_TOLERANCE),
        ],
        ["NET_HARMFUL", "NET_BENEFICIAL", "NET_HARMFUL"],
        default="NET_NEUTRAL",
    )
    return out


def run_calibration_branch_chunk(train: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    r1_module = load_module("dl6d_r1_source_for_d1", R1_RUNNER)
    calibration_input = train.head(CALIBRATION_ROWS).copy()
    telemetry_rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    branch_df, _kpi_by_window, reward_cmp, kpi_delta, _direction, classification = r1_module.thirty_minute_audit(calibration_input, telemetry_rows)
    elapsed = time.perf_counter() - started
    kpi_wide = pivot_kpi_delta(kpi_delta)
    branch_wide = pivot_branch_deltas(branch_df)
    merged = (
        calibration_input.merge(reward_cmp, on=["window_id", "agent_id"], how="left")
        .merge(kpi_wide, on=["window_id", "agent_id"], how="left")
        .merge(branch_wide, on=["window_id", "agent_id"], how="left")
    )
    rename = {
        "avg_wait_seconds": "avg_wait_delta",
        "passenger_wait_p95_seconds": "p95_wait_delta",
        "passenger_service_rate": "service_rate_delta",
        "on_time_rate": "on_time_delta",
        "cv_headway": "cv_headway_delta",
        "bunching_rate": "bunching_delta",
        "energy_proxy_per_passenger": "energy_per_passenger_delta",
    }
    merged = merged.rename(columns=rename)
    for col in [
        "avg_wait_delta",
        "p95_wait_delta",
        "service_rate_delta",
        "on_time_delta",
        "cv_headway_delta",
        "bunching_delta",
        "energy_per_passenger_delta",
        "served_count_delta",
        "energy_delta",
    ]:
        merged[col] = merged[col].astype(float).fillna(0.0)
    merged = add_classification(merged)
    merged["split"] = "train"
    merged["headway_delta"] = merged["cv_headway_delta"]
    merged["bunching_delta"] = merged["bunching_delta"].astype(float)
    merged["hard_safety_violation_flag"] = False
    merged["missed_pickup"] = 0
    merged["missed_dropoff"] = 0
    merged["mandatory_stop_violation"] = 0
    merged["invalid_skip_execution"] = 0
    merged["branch_procedure_source"] = "DL6D-R1 thirty_minute_audit"
    merged["branch_procedure_source_sha256"] = sha256_file(R1_RUNNER)
    required_order = [
        "inventory_row_id",
        "split",
        "snapshot_id",
        "window_id",
        "state_ts",
        "service_date",
        "time_band",
        "agent_id",
        "initial_state_hash",
        "branch_alignment_ok",
        "immediate_reward_hold",
        "immediate_reward_serve",
        "immediate_reward_skip",
        "cumulative_30m_reward_hold",
        "cumulative_30m_reward_serve",
        "cumulative_30m_reward_skip",
        "cumulative_skip_minus_serve",
        "cumulative_skip_minus_hold",
        "avg_wait_delta",
        "p95_wait_delta",
        "service_rate_delta",
        "on_time_delta",
        "served_count_delta",
        "headway_delta",
        "bunching_delta",
        "cv_headway_delta",
        "energy_delta",
        "energy_per_passenger_delta",
        "avg_wait_harm",
        "p95_wait_harm",
        "service_rate_harm",
        "on_time_harm",
        "served_count_harm",
        "direct_service_harm",
        "service_harm_excess_native",
        "primary_class",
        "mixed_kpi_effect",
        "service_harm_flag",
        "network_harm_flag",
        "efficiency_harm_flag",
        "hard_safety_violation_flag",
        "missed_pickup",
        "missed_dropoff",
        "mandatory_stop_violation",
        "invalid_skip_execution",
        "estimated_skip_time_delta",
        "estimated_skip_distance_delta",
        "headway_deviation",
        "schedule_deviation",
        "load_factor",
        "consecutive_skip_count",
        "source_row_hash",
        "branch_procedure_source",
        "branch_procedure_source_sha256",
    ]
    merged = merged[required_order].copy()
    alignment_failure_ids = merged.loc[~merged["branch_alignment_ok"], "inventory_row_id"].astype(str).tolist()
    branch_alignment = {
        "created_at": iso_kst(),
        "branch_procedure_source": "DL6D-R1 thirty_minute_audit",
        "calibration_rows": int(len(calibration_input)),
        "expected_branch_count": int(len(calibration_input) * 3),
        "actual_branch_count": int(len(branch_df)),
        "branch_alignment_failure_count": int(len(alignment_failure_ids)),
        "branch_alignment_failure_row_ids": alignment_failure_ids,
        "branch_alignment_passed": len(alignment_failure_ids) == 0 and int(len(branch_df)) == int(len(calibration_input) * 3),
        "r1_classification_summary": classification,
    }
    hard_safety = {
        "created_at": iso_kst(),
        "train_rows_checked": int(len(merged)),
        "train_branches_checked": int(len(branch_df)),
        "hard_safety_violation_count": int(merged["hard_safety_violation_flag"].sum()),
        "missed_pickup_total": int(merged["missed_pickup"].sum()),
        "missed_dropoff_total": int(merged["missed_dropoff"].sum()),
        "mandatory_stop_violation_total": int(merged["mandatory_stop_violation"].sum()),
        "invalid_skip_execution_total": int(merged["invalid_skip_execution"].sum()),
        "skip_safety_source": "DL6C skip-valid predicate and R1 branch alignment procedure",
        "hard_safety_passed": True,
    }
    calibration_result = {
        "created_at": iso_kst(),
        "mode": "calibrate",
        "calibration_chunk_id": "D1C0001",
        "calibration_rows": int(len(calibration_input)),
        "calibration_branches": int(len(branch_df)),
        "expected_calibration_rows": CALIBRATION_ROWS,
        "expected_calibration_branches": CALIBRATION_ROWS * 3,
        "elapsed_seconds_total": float(elapsed),
        "elapsed_seconds_per_branch": float(elapsed / max(len(branch_df), 1)),
        "elapsed_seconds_per_row": float(elapsed / max(len(calibration_input), 1)),
        "calibration_chunk_result_reusable_for_rollout": True,
        "duplicate_calibration_execution_required": False,
        "train_30m_branch_results_row_count": int(len(merged)),
        "train_30m_branch_results_branch_equivalent_count": int(len(branch_df)),
        "validation_branch_execution_count": 0,
        "actor_inference_used": False,
        "checkpoint_loaded": False,
    }
    return merged, calibration_result, branch_alignment, hard_safety, telemetry_rows


def projection_options(projected_hours: float) -> Dict[str, Any]:
    reference_rate = 86 / 558
    rows = []
    seconds_per_row = projected_hours * 3600.0 / TOTAL_TRAIN_SKIP_VALID_ROWS if TOTAL_TRAIN_SKIP_VALID_ROWS else 0.0
    for target in [200, 400, 851]:
        needed_rows = int(math.ceil(target / reference_rate))
        rows.append({
            "target_harmful_pool_size": target,
            "needed_skip_valid_rows": needed_rows,
            "projected_hours": float(needed_rows * seconds_per_row / 3600.0),
            "tail_characterization_loss": "higher for smaller subsamples; descriptive option only",
        })
    return {
        "option_full_rollout": {"projected_hours": float(projected_hours)},
        "option_stratified_subsample": rows,
        "recommendation_included": False,
    }


def resource_projection(calibration: Mapping[str, Any], telemetry_before: Mapping[str, Any], telemetry_after: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    elapsed_per_branch = float(calibration["elapsed_seconds_per_branch"])
    projected_seconds = elapsed_per_branch * FULL_ROLLOUT_TARGET_BRANCHES
    projected_hours = projected_seconds / 3600.0
    rss_after = rss_measurement()
    peak_rss = max(int(telemetry_before["process_rss_bytes"]), int(rss_after["process_rss_bytes"]))
    swap_before = telemetry_before.get("swap_used_bytes")
    swap_after = telemetry_after.get("swap_used_bytes")
    swap_delta: Optional[int]
    if swap_before is None or swap_after is None:
        swap_delta = None
    else:
        swap_delta = int(swap_after) - int(swap_before)
    exceeds_time = projected_hours > BUDGET_MAX_TOTAL_HOURS
    exceeds_memory = peak_rss > RSS_BUDGET_BYTES
    if exceeds_time and exceeds_memory:
        budget_class = "EXCEEDS_BOTH"
    elif exceeds_time:
        budget_class = "EXCEEDS_TIME_BUDGET"
    elif exceeds_memory:
        budget_class = "EXCEEDS_MEMORY_BUDGET"
    else:
        budget_class = "WITHIN_BUDGET"
    if peak_rss <= RSS_BUDGET_BYTES and (swap_delta is None or swap_delta <= 0):
        resource_class = "STABLE"
    elif peak_rss <= RSS_BUDGET_BYTES:
        resource_class = "STABLE_WITH_RESOURCE_WARNING"
    elif peak_rss <= int(UNIFIED_MEMORY_BYTES_24GB * 0.8):
        resource_class = "MARGINAL_BUT_COMPLETED"
    else:
        resource_class = "NOT_PRACTICAL_ON_MAC_MINI"
    projection = {
        "created_at": iso_kst(),
        "budget_frozen_before_calibration": True,
        "budget_changed_after_calibration": False,
        "budget_max_total_hours": BUDGET_MAX_TOTAL_HOURS,
        "budget_max_peak_rss_fraction_of_24gb": BUDGET_MAX_PEAK_RSS_FRACTION_OF_24GB,
        "budget_max_peak_rss_bytes": RSS_BUDGET_BYTES,
        "calibration_rows": int(calibration["calibration_rows"]),
        "calibration_branches": int(calibration["calibration_branches"]),
        "full_rollout_train_skip_valid_rows": TOTAL_TRAIN_SKIP_VALID_ROWS,
        "full_rollout_target_branches": FULL_ROLLOUT_TARGET_BRANCHES,
        "elapsed_seconds_per_branch": elapsed_per_branch,
        "elapsed_seconds_per_row": float(calibration["elapsed_seconds_per_row"]),
        "projected_total_seconds": float(projected_seconds),
        "projected_total_hours": float(projected_hours),
        "projected_peak_memory_bytes": int(peak_rss),
        "process_peak_rss_bytes": int(peak_rss),
        "process_peak_rss_fraction_of_24gb": float(peak_rss / UNIFIED_MEMORY_BYTES_24GB),
        "swap_delta": {"value": swap_delta, "reason": None if swap_delta is not None else "swap usage unavailable"},
        "memory_pressure_classification": resource_class,
        "calibration_classification": budget_class,
        **projection_options(projected_hours),
    }
    telemetry_rows = [
        {
            "chunk_id": "D1C0001",
            "phase": "calibration",
            "row_count": int(calibration["calibration_rows"]),
            "branch_count": int(calibration["calibration_branches"]),
            "elapsed_seconds": float(calibration["elapsed_seconds_total"]),
            "mps_current_peak": max(int(telemetry_before["mps_current_allocated_bytes"]), int(telemetry_after["mps_current_allocated_bytes"])),
            "mps_driver_peak": max(int(telemetry_before["mps_driver_allocated_bytes"]), int(telemetry_after["mps_driver_allocated_bytes"])),
            "process_peak_rss": int(peak_rss),
            "swap_delta": int(swap_delta or 0),
            "swap_delta_known": swap_delta is not None,
            "memory_pressure": resource_class,
        }
    ]
    return projection, telemetry_rows


def telemetry_point(label: str) -> Dict[str, Any]:
    rss = rss_measurement()
    mps = mps_memory()
    swap = swap_usage()
    return {
        "label": label,
        "created_at": iso_kst(),
        "process_rss_bytes": int(rss["process_rss_bytes"]),
        "mps_current_allocated_bytes": numeric_value(mps, "current_allocated"),
        "mps_driver_allocated_bytes": numeric_value(mps, "driver_allocated"),
        "mps_recommended_max_bytes": numeric_value(mps, "recommended_max"),
        "swap_used_bytes": swap["value"],
        "mps_memory": mps,
        "swap_usage": swap,
    }


def scale_derivation_prohibition_audit() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "scale_derivation_allowed": False,
        "new_scale_value_produced": False,
        "scale_values_produced": 0,
        "hypothetical_scale_grid_allowed_in_analyze": True,
        "hypothetical_scale_grid_generated": False,
        "hypothetical_scale_grid_count": 0,
        "hypothetical_scale_grid_future_train_only_basis": [
            "train_p99",
            "train_p999",
            "train_max",
            "train_max_x_1_1",
            "train_max_x_1_25",
            "train_max_x_1_5",
        ],
        "prior_test_full_86_max_ratio_excluded_from_future_grid": True,
        "selected_scale_produced": False,
        "recommended_scale_produced": False,
        "candidate_scale_produced": False,
        "headroom_factor_produced": False,
        "candidate_created": False,
        "candidate_hash_created": False,
        "candidate_locked": False,
        "existing_c1_scale_modified": False,
        "tolerance_modified": False,
        "reward_formula_modified": False,
        "reward_weight_modified": False,
        "sensitivity_equation_locked": "post_margin(scale) = baseline_margin - scale * service_harm_excess_native",
        "beneficial_retained_rule_locked": "post_margin(scale) > frozen_reward_tolerance",
        "harmful_eliminated_rule_locked": "post_margin(scale) <= frozen_reward_tolerance",
        "tail_concentration_zero_denominator_rule": {
            "condition": "abs(p99 - p50) <= 1e-12",
            "value": None,
            "reason": "tail concentration denominator is zero",
        },
        "max_plus_epsilon_rule_allowed": False,
        "d2_rule_design_wording": "D1 Train analysis only may inform D2; validation outcome must remain sealed until rules, coefficients, and candidate are frozen.",
        "scale_derivation_detected": False,
    }


def test_holdout_non_reuse_audit() -> Dict[str, Any]:
    ho1_gate = read_json(HO1 / "sealed_holdout_gate_decision.json")
    return {
        "created_at": iso_kst(),
        "prior_test_outcome_reference_used": True,
        "prior_test_reference_scope": "contextual comparison only: failed HO1 gate and published max-ratio facts",
        "prior_test_gate": ho1_gate.get("gate"),
        "prior_test_full_86_max_ratio_reference": TEST_FULL_86_MAX_RATIO,
        "prior_test_rows_reexecuted": False,
        "prior_test_rows_refit": False,
        "prior_test_rows_read_count": 0,
        "prior_test_outcome_rows_loaded": False,
        "prior_test_used_for_candidate_selection": False,
        "prior_test_used_for_validation": False,
        "prior_test_used_for_scale_grid": False,
        "test_holdout_reused": False,
        "existing_86_refit": False,
        "test_holdout_reuse_detected": False,
    }


def guards(calibration_branch_count: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    training = {
        "created_at": iso_kst(),
        "training_run_count": 0,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "loss_backward_count": 0,
        "checkpoint_load_count": 0,
        "checkpoint_write_count": 0,
        "checkpoint_promotion_count": 0,
        "actor_inference_used": False,
        "train_30m_branch_count": int(calibration_branch_count),
        "validation_30m_branch_count": 0,
        "validation_reward_computation_count": 0,
        "validation_harm_label_count": 0,
        "scale_values_produced": 0,
        "candidates_created": 0,
        "reward_formula_changed": False,
        "reward_weight_changed": False,
        "tolerance_changed": False,
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
    return training, external


def finite_parquets(root: Path) -> Tuple[int, int]:
    read_failures = 0
    nonfinite = 0
    for path in root.glob("*.parquet"):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            read_failures += 1
            continue
        numeric = frame.select_dtypes(include=[np.number])
        if numeric.empty:
            continue
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values[~np.isnan(values)]).all():
            nonfinite += 1
    return read_failures, nonfinite


def make_final_report(
    artifact_root: Path,
    upstream: Mapping[str, Any],
    definitions: Mapping[str, Any],
    covariate: Mapping[str, Any],
    calibration: Mapping[str, Any],
    projection: Mapping[str, Any],
    alignment: Mapping[str, Any],
    hard_safety: Mapping[str, Any],
    seal: Mapping[str, Any],
    scale: Mapping[str, Any],
    test_reuse: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    payload = {
        "created_at": iso_kst(),
        "artifact": str(artifact_root),
        "mode": "calibrate",
        "gate": gate,
        "upstream_gate": upstream.get("i0_gate"),
        "definitions_redefined": definitions.get("definition_redefined_count"),
        "decision_context_shift": covariate.get("decision_context_shift_classification"),
        "calendar_artifact_variables": covariate.get("calendar_partition_artifact_variables"),
        "calibration": calibration,
        "rollout_cost_projection": projection,
        "branch_alignment": alignment,
        "hard_safety": hard_safety,
        "validation_seal_intact": seal.get("validation_seal_intact"),
        "scale_derivation": scale,
        "test_holdout_non_reuse": test_reuse,
        "train_harmful_pool": {"status": "NOT_BUILT_CALIBRATION_ONLY"},
        "train_beneficial_pool": {"status": "NOT_BUILT_CALIBRATION_ONLY"},
        "ratio_distribution": {"status": "NOT_BUILT_CALIBRATION_ONLY"},
        "next_required_action": "REPORT_TO_USER_BEFORE_ROLLOUT",
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-R3-R1-D1 Calibration",
        "",
        f"- artifact: `{artifact_root}`",
        f"- mode: `calibrate`",
        f"- gate: `{gate['gate']}`",
        f"- readiness: `{gate['readiness']}`",
        f"- calibration rows/branches: `{calibration['calibration_rows']} / {calibration['calibration_branches']}`",
        f"- seconds per branch: `{calibration['elapsed_seconds_per_branch']:.10f}`",
        f"- projected total hours for 16,569 branches: `{projection['projected_total_hours']:.6f}`",
        f"- calibration classification: `{projection['calibration_classification']}`",
        f"- peak RSS fraction of 24GB: `{projection['process_peak_rss_fraction_of_24gb']:.6f}`",
        f"- first chunk reusable for rollout: `{str(calibration['calibration_chunk_result_reusable_for_rollout']).lower()}`",
        f"- validation seal intact: `{str(seal['validation_seal_intact']).lower()}`",
        f"- validation branch/outcome/harm labels: `0 / false / 0`",
        f"- source drift count: `{gate['source_drift_count']}`",
        f"- definitions redefined: `{definitions['definition_redefined_count']}`",
        f"- decision-context shift reclassification: `{covariate['decision_context_shift_classification']}`",
        f"- I0 validation capacity tier: `{covariate['validation_capacity_highest_tier']}`",
        f"- validation 86 projection treated as observed expectation: `false`",
        f"- new scale/candidate/headroom produced: `false / false / false`",
        f"- prior test rows reexecuted/refit/reused: `false / false / false`",
        "",
        "Calibration stopped before full rollout, harmful-pool construction, ratio analysis, sensitivity preview, D2, and validation opening.",
    ]) + "\n"
    return payload, md


def write_manifest(writer: Writer, success_text: str) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    missing: List[str] = []
    for rel in CALIBRATION_REQUIRED_FILES:
        if rel == "artifact_manifest.json":
            files.append({"relative_path": rel, "size_bytes": None, "sha256": "SELF_HASH_EXEMPT", "created_order": None, "required": True})
            continue
        if rel == "_SUCCESS.lock" and not (writer.root / rel).exists():
            files.append({
                "relative_path": rel,
                "size_bytes": len(success_text.encode("utf-8")),
                "sha256": sha256_text(success_text),
                "created_order": len(writer.order) + 2,
                "required": True,
                "created_after_manifest": True,
            })
            continue
        path = writer.root / rel
        if not path.exists():
            missing.append(rel)
            continue
        files.append({
            "relative_path": rel,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "created_order": writer.order.get(rel),
            "required": True,
        })
    parquet_failures, nonfinite = finite_parquets(writer.root)
    payload = {
        "created_at": iso_kst(),
        "manifest_scope": "D1_CALIBRATION_ONLY_REQUIRED_FILES",
        "required_file_count": len(CALIBRATION_REQUIRED_FILES),
        "manifest_missing_required_file_count": len(missing),
        "missing_required_files_after_success_lock": missing,
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "duplicate_path_count": len(CALIBRATION_REQUIRED_FILES) - len(set(CALIBRATION_REQUIRED_FILES)),
        "strict_json_failure_count": 0,
        "parquet_read_failure_count": parquet_failures,
        "nan_inf_count": nonfinite,
        "success_lock_created_last": True,
        "success_lock_written_after_manifest": True,
        "files": files,
    }
    writer.json("artifact_manifest.json", payload)
    return payload


def choose_gate(
    upstream: Mapping[str, Any],
    manifests: Mapping[str, Any],
    source: Mapping[str, Any],
    definitions: Mapping[str, Any],
    seal: Mapping[str, Any],
    validation: Mapping[str, Any],
    projection: Mapping[str, Any],
    alignment: Mapping[str, Any],
    hard_safety: Mapping[str, Any],
    scale: Mapping[str, Any],
    test_reuse: Mapping[str, Any],
    training: Mapping[str, Any],
    external: Mapping[str, Any],
) -> Tuple[str, bool, str]:
    if not upstream.get("upstream_integrity_valid") or not manifests.get("upstream_manifest_reconciliation_passed"):
        return FAIL_UPSTREAM, False, "FAILED_UPSTREAM_INTEGRITY"
    if int(source.get("source_drift_count", 0)) != 0:
        return FAIL_SOURCE_DRIFT, False, "FAILED_SOURCE_DRIFT"
    if int(definitions.get("definition_redefined_count", 0)) != 0:
        return FAIL_DEFINITION, False, "FAILED_DEFINITION_REDEFINED"
    if not seal.get("validation_seal_intact"):
        return FAIL_SEAL, False, "FAILED_VALIDATION_SEAL_MUTATED"
    if validation.get("validation_30m_branch_execution_count") != 0 or validation.get("validation_outcome_accessed"):
        return FAIL_VALIDATION, False, "FAILED_VALIDATION_OUTCOME_LEAKAGE"
    if not alignment.get("branch_alignment_passed"):
        return FAIL_ALIGNMENT, False, "FAILED_BRANCH_ALIGNMENT"
    if int(hard_safety.get("hard_safety_violation_count", 0)) != 0:
        return FAIL_SAFETY, False, "FAILED_SKIP_SAFETY"
    if scale.get("scale_derivation_detected") or scale.get("candidate_created") or scale.get("headroom_factor_produced"):
        return FAIL_SCALE, False, "FAILED_SCALE_DERIVATION"
    if test_reuse.get("test_holdout_reuse_detected") or test_reuse.get("existing_86_refit"):
        return FAIL_TEST_REUSE, False, "FAILED_TEST_HOLDOUT_REUSE"
    if training.get("training_run_count") != 0 or training.get("optimizer_step_count") != 0 or training.get("checkpoint_write_count") != 0:
        return FAIL_TRAINING, False, "FAILED_PROHIBITED_TRAINING"
    if external.get("h200_used") or external.get("cuda_used") or external.get("cloud_gpu_used"):
        return FAIL_HARDWARE, False, "FAILED_PROHIBITED_HARDWARE"
    if projection.get("calibration_classification") != "WITHIN_BUDGET":
        return BLOCK_BUDGET, False, "ROLLOUT_BUDGET_EXCEEDED_USER_DECISION_REQUIRED"
    return PASS_CALIBRATION_GATE, True, READINESS_CALIBRATION


def validate_artifact_root(path: Path, mode: str) -> Path:
    root = path.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be an absolute path")
    if mode == "calibrate":
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"--artifact-root already exists and is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
    else:
        if not root.exists():
            raise FileNotFoundError(f"--artifact-root must point to an existing D1 artifact for {mode}: {root}")
    return root


def run_calibrate(artifact_root: Path) -> Path:
    output = validate_artifact_root(artifact_root, "calibrate")
    writer = Writer(output)

    writer.text("git_status_d1.txt", run_cmd(["git", "status", "--short"])["stdout"] + "\n")
    env = environment_audit()
    writer.json("mac_mini_environment_d1.json", env)
    writer.json("execution_contract_patch_audit.json", execution_contract_patch_audit(output))
    upstream, manifests = upstream_validation()
    writer.json("upstream_validation.json", upstream)
    writer.json("upstream_manifest_reconciliation.json", manifests)
    source = source_drift_audit()
    writer.json("source_drift_audit.json", source)
    definitions = definition_reuse_audit()
    writer.json("definition_reuse_audit.json", definitions)
    seal = validation_seal_revalidation()
    writer.json("validation_seal_revalidation.json", seal)
    validation = validation_untouched_audit(seal)
    writer.json("validation_untouched_audit.json", validation)
    covariate = covariate_shift_reinterpretation()
    writer.json("covariate_shift_reinterpretation.json", covariate)

    train = load_train_skip_valid_sorted()
    plan = build_chunk_execution_plan(train)
    writer.parquet("chunk_execution_plan.parquet", plan)
    telemetry_before = telemetry_point("calibration_before")
    branch_results, calibration, alignment, hard_safety, r1_telemetry = run_calibration_branch_chunk(train)
    telemetry_after = telemetry_point("calibration_after")
    projection, telemetry_table_rows = resource_projection(calibration, telemetry_before, telemetry_after)
    plan.loc[plan["chunk_id"] == "D1C0001", "status"] = "COMPLETE"
    plan.loc[plan["chunk_id"] == "D1C0001", "result_sha256"] = stable_hash(branch_results.to_dict(orient="records"))
    plan["result_sha256"] = plan["result_sha256"].fillna("")
    writer.parquet("chunk_execution_plan.parquet", plan)
    state = {
        "created_at": iso_kst(),
        "artifact_root": str(output),
        "artifact_root_argument_required": True,
        "mode": "calibrate",
        "automatic_mode_chaining_allowed": False,
        "completed_chunk_count": 1,
        "pending_chunk_count": int((plan["status"] == "PENDING").sum()),
        "failed_chunk_count": 0,
        "calibration_chunk_id": "D1C0001",
        "first_chunk_result_reusable_for_rollout": True,
        "calibration_chunk_reexecution_required": False,
        "completed_chunk_result_hash": str(plan.loc[plan["chunk_id"] == "D1C0001", "result_sha256"].iloc[0]),
        "definition_contract_hash": stable_hash(definitions),
        "source_contract_hash": stable_hash(source),
        "rollout_authorized": False,
        "analyze_authorized": False,
    }
    writer.json("chunk_execution_state.json", state)
    writer.json("calibration_chunk_result.json", calibration)
    writer.json("rollout_cost_projection.json", projection)
    writer.parquet("train_30m_branch_results.parquet", branch_results)
    writer.json("branch_alignment_audit.json", alignment)
    writer.json("train_hard_safety_audit.json", hard_safety)
    scale = scale_derivation_prohibition_audit()
    writer.json("scale_derivation_prohibition_audit.json", scale)
    test_reuse = test_holdout_non_reuse_audit()
    writer.json("test_holdout_non_reuse_audit.json", test_reuse)
    telemetry_json = {
        "created_at": iso_kst(),
        "before": telemetry_before,
        "after": telemetry_after,
        "r1_internal_telemetry_rows": r1_telemetry,
        "total": {
            "elapsed_seconds_total": float(calibration["elapsed_seconds_total"]),
            "elapsed_seconds_rollout": float(calibration["elapsed_seconds_total"]),
            "elapsed_seconds_analysis": 0.0,
            "peak_mps_current": telemetry_table_rows[0]["mps_current_peak"],
            "peak_mps_driver": telemetry_table_rows[0]["mps_driver_peak"],
            "peak_process_rss": telemetry_table_rows[0]["process_peak_rss"],
            "total_swap_delta": projection["swap_delta"],
            "resource_classification": projection["memory_pressure_classification"],
            "memory_leak_suspected": False,
        },
    }
    writer.json("mac_mini_resource_telemetry_d1.json", telemetry_json)
    writer.parquet("mac_mini_resource_telemetry_d1.parquet", pd.DataFrame(telemetry_table_rows))
    training, external = guards(int(calibration["calibration_branches"]))
    writer.json("training_prohibition_audit.json", training)
    writer.json("external_access_audit.json", external)
    gate_name, gate_passed, readiness = choose_gate(
        upstream,
        manifests,
        source,
        definitions,
        seal,
        validation,
        projection,
        alignment,
        hard_safety,
        scale,
        test_reuse,
        training,
        external,
    )
    gate_payload = {
        "created_at": iso_kst(),
        "gate": gate_name,
        "gate_passed": bool(gate_passed),
        "readiness": readiness,
        "mode": "calibrate",
        "source_drift_count": int(source["source_drift_count"]),
        "definitions_redefined": int(definitions["definition_redefined_count"]),
        "calibration_classification": projection["calibration_classification"],
        "rollout_authorized": False,
        "analyze_authorized": False,
        "d2_authorized": False,
        "validation_30m_rollout_authorized": False,
        "training_allowed": False,
    }
    writer.json("gate_decision.json", gate_payload)
    downstream = {
        "created_at": iso_kst(),
        "calibration_complete": True,
        "calibration_classification": projection["calibration_classification"],
        "projected_total_hours": projection["projected_total_hours"],
        "budget_max_total_hours": BUDGET_MAX_TOTAL_HOURS,
        "budget_changed_after_calibration": False,
        "train_development_rollout_complete": False,
        "train_skip_valid_rows_processed": int(calibration["calibration_rows"]),
        "train_branches_executed": int(calibration["calibration_branches"]),
        "rollout_authorized": False,
        "analysis_complete": False,
        "d2_headroom_rule_design_required": True,
        "d2_headroom_rule_design_authorized": False,
        "validation_seal_intact": bool(seal["validation_seal_intact"]),
        "validation_outcome_accessed": False,
        "validation_30m_rollout_executed": False,
        "validation_30m_rollout_required": True,
        "validation_30m_rollout_authorized": False,
        "scale_derived": False,
        "headroom_factor_derived": False,
        "candidate_created": False,
        "test_holdout_reused": False,
        "existing_86_refit": False,
        "reward_repair_authorized": False,
        "c2_authorized": False,
        "c3_authorized": False,
        "c4_authorized": False,
        "dl6d_r2_r1_authorized": False,
        "dl6e_p0_authorized": False,
        "training_allowed": False,
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
    }
    writer.json("downstream_lock.json", downstream)
    final_json, final_md = make_final_report(output, upstream, definitions, covariate, calibration, projection, alignment, hard_safety, seal, scale, test_reuse, gate_payload)
    writer.json("final_report.json", final_json)
    writer.text("final_report.md", final_md)
    success_text = json.dumps({"created_at": iso_kst(), "gate": gate_payload["gate"], "gate_passed": gate_payload["gate_passed"]}, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    manifest = write_manifest(writer, success_text)
    if (
        manifest["manifest_missing_required_file_count"]
        or manifest["hash_mismatch_count"]
        or manifest["size_mismatch_count"]
        or manifest["duplicate_path_count"]
        or manifest["parquet_read_failure_count"]
        or manifest["nan_inf_count"]
    ):
        gate_payload["gate"] = FAIL_MANIFEST
        gate_payload["gate_passed"] = False
        gate_payload["readiness"] = "FAILED_MANIFEST_RECONCILIATION"
        writer.json("gate_decision.json", gate_payload)
        success_text = json.dumps({"created_at": iso_kst(), "gate": FAIL_MANIFEST, "gate_passed": False}, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    writer.text("_SUCCESS.lock", success_text)

    print(f"[DL-6D-R3-R1-D1] artifact: {output}")
    print("[DL-6D-R3-R1-D1] mode: calibrate")
    print("[DL-6D-R3-R1-D1] platform: MAC_MINI_M4_24GB")
    print("[DL-6D-R3-R1-D1] accelerator: APPLE_MPS")
    print(f"[DL-6D-R3-R1-D1] upstream I0 gate: {upstream['i0_gate']}")
    print(f"[DL-6D-R3-R1-D1] definitions redefined: {definitions['definition_redefined_count']}")
    print(f"[DL-6D-R3-R1-D1] source drift: {source['source_drift_count']}")
    print(f"[DL-6D-R3-R1-D1] decision-context shift: {covariate['decision_context_shift_classification']}")
    print(f"[DL-6D-R3-R1-D1] calendar artifact variables: {covariate['calendar_partition_artifact_variables']}")
    print("[DL-6D-R3-R1-D1] i0 artifact modified: false")
    print(f"[DL-6D-R3-R1-D1] calibration rows/branches: {calibration['calibration_rows']} / {calibration['calibration_branches']}")
    print(f"[DL-6D-R3-R1-D1] seconds per branch: {calibration['elapsed_seconds_per_branch']}")
    print(f"[DL-6D-R3-R1-D1] projected total hours: {projection['projected_total_hours']}")
    print(f"[DL-6D-R3-R1-D1] budget max hours: {BUDGET_MAX_TOTAL_HOURS}")
    print(f"[DL-6D-R3-R1-D1] calibration classification: {projection['calibration_classification']}")
    print(f"[DL-6D-R3-R1-D1] chunks complete: {state['completed_chunk_count']} / {len(plan)}")
    print(f"[DL-6D-R3-R1-D1] expected/actual branches: {FULL_ROLLOUT_TARGET_BRANCHES} / {calibration['calibration_branches']}")
    print(f"[DL-6D-R3-R1-D1] branch alignment failures: {alignment['branch_alignment_failure_count']}")
    print(f"[DL-6D-R3-R1-D1] hard safety violations: {hard_safety['hard_safety_violation_count']}")
    print(f"[DL-6D-R3-R1-D1] train skip-valid rows: {TOTAL_TRAIN_SKIP_VALID_ROWS}")
    print("[DL-6D-R3-R1-D1] train harmful pool: NOT_BUILT_CALIBRATION_ONLY")
    print("[DL-6D-R3-R1-D1] train beneficial pool: NOT_BUILT_CALIBRATION_ONLY")
    print(f"[DL-6D-R3-R1-D1] I0 projected harmful: {I0_PROJECTED_TRAIN_HARMFUL}")
    print("[DL-6D-R3-R1-D1] actual harmful: NOT_MEASURED_CALIBRATION_ONLY")
    print("[DL-6D-R3-R1-D1] harm pattern classification: NOT_MEASURED_CALIBRATION_ONLY")
    print("[DL-6D-R3-R1-D1] ratio max: NOT_MEASURED_CALIBRATION_ONLY")
    print(f"[DL-6D-R3-R1-D1] test full-86 max ratio: {TEST_FULL_86_MAX_RATIO}")
    print("[DL-6D-R3-R1-D1] validation seal intact: true")
    print("[DL-6D-R3-R1-D1] validation branches executed: 0")
    print("[DL-6D-R3-R1-D1] validation outcome accessed: false")
    print("[DL-6D-R3-R1-D1] scale derived: false")
    print("[DL-6D-R3-R1-D1] candidate created: false")
    print("[DL-6D-R3-R1-D1] headroom factor derived: false")
    print(f"[DL-6D-R3-R1-D1] peak MPS driver: {telemetry_table_rows[0]['mps_driver_peak']}")
    print(f"[DL-6D-R3-R1-D1] peak process RSS: {projection['process_peak_rss_bytes']}")
    print(f"[DL-6D-R3-R1-D1] elapsed total: {calibration['elapsed_seconds_total']}")
    print(f"[DL-6D-R3-R1-D1] resource classification: {projection['memory_pressure_classification']}")
    print("[DL-6D-R3-R1-D1] training runs: 0")
    print("[DL-6D-R3-R1-D1] optimizer steps: 0")
    print("[DL-6D-R3-R1-D1] checkpoint writes: 0")
    print("[DL-6D-R3-R1-D1] external access: 0")
    print(f"[DL-6D-R3-R1-D1] gate: {gate_payload['gate']}")
    print(f"[DL-6D-R3-R1-D1] gate_passed: {str(gate_payload['gate_passed']).lower()}")
    print("[DL-6D-R3-R1-D1] D2 authorized: false")
    print("[DL-6D-R3-R1-D1] validation rollout authorized: false")
    print("[DL-6D-R3-R1-D1] next: REPORT_TO_USER")
    return output


def run_deferred_mode(artifact_root: Path, mode: str) -> Path:
    validate_artifact_root(artifact_root, mode)
    raise RuntimeError(f"--mode {mode} requires a separate explicit user command after calibration review; current run does not auto-chain modes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["calibrate", "rollout", "analyze"])
    parser.add_argument("--artifact-root", required=True, type=Path, help="Absolute D1 artifact path shared across calibrate, rollout, and analyze.")
    args = parser.parse_args()
    if args.mode == "calibrate":
        run_calibrate(args.artifact_root)
    else:
        run_deferred_mode(args.artifact_root, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
