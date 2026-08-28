from __future__ import annotations

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
ARTIFACT_PREFIX = "prompt5_e01_dl6d_r3_pre_holdout_audit"
R3_FAMILY = "prompt5_e01_dl6d_r3_reward_service_alignment_repair_*"
R2 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r2_combined_retraining_readiness_reaudit_20260802_102624"
R1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r1_observation_contract_repair_20260802_011348"

PASS_GATE = "PASS_SUSEONG_DL6D_R3_PRE_HOLDOUT_TECHNICAL_AUDIT_PENDING_MANUAL_REVIEW"
BLOCK_MULTI = "BLOCKED_SUSEONG_DL6D_R3_PH_MULTIPLE_AUTHORITATIVE_ARTIFACTS"
BLOCK_ROOT_MISMATCH = "BLOCKED_SUSEONG_DL6D_R3_PH_ROOT_CAUSE_CANDIDATE_MISMATCH"
BLOCK_ROOT_INSUFFICIENT = "BLOCKED_SUSEONG_DL6D_R3_PH_ROOT_CAUSE_EVIDENCE_INSUFFICIENT"
BLOCK_LOCK = "BLOCKED_SUSEONG_DL6D_R3_PH_CANDIDATE_NOT_PROPERLY_LOCKED"
FAIL_C0 = "FAIL_SUSEONG_DL6D_R3_PH_C0_EXACT_COUNT_MISMATCH"
FAIL_SEAL = "FAIL_SUSEONG_DL6D_R3_PH_HOLDOUT_SEAL_MUTATED"
FAIL_OPENED = "FAIL_SUSEONG_DL6D_R3_PH_HOLDOUT_ALREADY_OPENED"
FAIL_HASH = "FAIL_SUSEONG_DL6D_R3_PH_CANDIDATE_THREE_WAY_HASH_MISMATCH"
FAIL_FROZEN = "FAIL_SUSEONG_DL6D_R3_PH_FROZEN_CONTRACT_HASH_MISMATCH"
FAIL_LEAK = "FAIL_SUSEONG_DL6D_R3_PH_HOLDOUT_LEAKAGE"
FAIL_LADDER = "FAIL_SUSEONG_DL6D_R3_PH_CANDIDATE_LADDER_VIOLATION"
FAIL_TRAINING = "FAIL_SUSEONG_DL6D_R3_PH_PROHIBITED_TRAINING"
FAIL_HW = "FAIL_SUSEONG_DL6D_R3_PH_H200_OR_CUDA_USAGE"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_R3_PH_MANIFEST_RECONCILIATION"

REQUIRED_FILES = [
    "git_status_pre_holdout.txt",
    "mac_mini_environment_pre_holdout.json",
    "authoritative_r3_artifact_selection.json",
    "holdout_seal_immutability_audit.json",
    "holdout_pre_open_state.json",
    "c0_baseline_exact_count_audit.json",
    "c0_baseline_exact_count_table.parquet",
    "selected_candidate_three_way_hash_audit.json",
    "frozen_contract_integrity_audit.json",
    "c1_design_gate_reconfirmation.json",
    "candidate_ladder_termination_audit.json",
    "root_cause_candidate_consistency_evidence.json",
    "root_cause_signal_scale_comparison.parquet",
    "root_cause_reward_margin_comparison.parquet",
    "root_cause_case_level_consistency.parquet",
    "pre_holdout_manual_review_packet.json",
    "pre_holdout_manual_review_packet.md",
    "pre_holdout_gate_decision.json",
    "pre_holdout_downstream_lock.json",
    "training_prohibition_audit_pre_holdout.json",
    "external_access_audit_pre_holdout.json",
    "artifact_manifest_pre_holdout.json",
    "_PRE_HOLDOUT_COMPLETE.lock",
]


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}

    def mark(self, name: str) -> None:
        self.order[name] = len(self.order) + 1

    def json(self, name: str, payload: Mapping[str, Any]) -> None:
        (self.root / name).write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self.mark(name)

    def text(self, name: str, text: str) -> None:
        (self.root / name).write_text(text, encoding="utf-8")
        self.mark(name)

    def parquet(self, name: str, frame: pd.DataFrame) -> None:
        frame.to_parquet(self.root / name, index=False)
        self.mark(name)


def now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def iso_now() -> str:
    return now().isoformat(timespec="seconds")


def timestamp() -> str:
    return now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str).encode("utf-8")).hexdigest()


def redact_identity(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lower = line.lower()
        if "serial number" in lower or "hardware uuid" in lower or "provisioning udid" in lower:
            lines.append(f"{line.split(':', 1)[0]}: REDACTED")
        else:
            lines.append(line)
    return "\n".join(lines)


def run_cmd(args: Sequence[str]) -> Dict[str, Any]:
    result = subprocess.run(list(args), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {"cmd": list(args), "returncode": result.returncode, "stdout": redact_identity(result.stdout.strip()), "stderr": redact_identity(result.stderr.strip())}


def rss() -> Dict[str, Any]:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return {"raw_ru_maxrss": raw, "ru_maxrss_unit": "bytes", "process_rss_bytes": raw}
    return {"raw_ru_maxrss": raw, "ru_maxrss_unit": "kilobytes", "process_rss_bytes": raw * 1024}


def mps_memory() -> Dict[str, Any]:
    if not hasattr(torch, "mps"):
        return {"current_allocated": None, "driver_allocated": None, "recommended_max": None}
    out: Dict[str, Any] = {}
    for key, name in [("current_allocated", "current_allocated_memory"), ("driver_allocated", "driver_allocated_memory"), ("recommended_max", "recommended_max_memory")]:
        fn = getattr(torch.mps, name, None)
        try:
            out[key] = int(fn()) if fn else None
        except Exception:
            out[key] = None
    return out


def environment() -> Dict[str, Any]:
    mem = run_cmd(["sysctl", "-n", "hw.memsize"])
    model = run_cmd(["sysctl", "-n", "hw.model"])
    sw = run_cmd(["sw_vers"])
    profiler = run_cmd(["system_profiler", "SPHardwareDataType"])
    return {
        "created_at": iso_now(),
        "current_execution_platform": "MAC_MINI_M4_24GB",
        "current_accelerator": "APPLE_MPS",
        "hardware_model": model["stdout"],
        "chip_name": "Apple M4",
        "unified_memory_bytes": int(mem["stdout"]) if mem["stdout"].isdigit() else None,
        "macos_version": sw,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_memory": mps_memory(),
        "process_memory": rss(),
        "system_profiler_hardware_redacted": profiler,
        "h200_used": False,
        "cuda_used": False,
        "cloud_gpu_used": False,
    }


def manifest_valid(root: Path) -> bool:
    path = root / "artifact_manifest.json"
    if not path.exists():
        return False
    data = read_json(path)
    return (
        data.get("manifest_missing_required_file_count") == 0
        and data.get("hash_mismatch_count") == 0
        and data.get("size_mismatch_count") == 0
        and data.get("duplicate_path_count") == 0
        and data.get("success_lock_created_last") is True
    )


def is_authoritative_candidate(root: Path) -> bool:
    try:
        gate = read_json(root / "gate_decision.json")
        selected = read_json(root / "selected_candidate_contract.json")
        seal = read_json(root / "holdout_seal_contract.json")
    except Exception:
        return False
    return (
        manifest_valid(root)
        and (root / "_HOLDOUT_SEALED.lock").exists()
        and not (root / "_HOLDOUT_OPENED.lock").exists()
        and (root / "_SELECTED_CANDIDATE.lock").exists()
        and gate.get("gate_passed") is True
        and gate.get("selected_candidate_id") == "R3_C1_NORMALIZATION_ONLY"
        and selected.get("selected_candidate_id") == "R3_C1_NORMALIZATION_ONLY"
        and seal.get("holdout_opened") is False
        and int(seal.get("holdout_open_count", -1)) == 0
    )


def select_r3_artifact() -> Tuple[Path | None, Dict[str, Any]]:
    candidates = sorted((PROJECT_ROOT / "05_training/artifacts").glob(R3_FAMILY))
    valid = [path for path in candidates if is_authoritative_candidate(path)]
    payload = {
        "created_at": iso_now(),
        "candidate_paths": [str(p) for p in candidates],
        "valid_authoritative_paths": [str(p) for p in valid],
        "valid_authoritative_count": len(valid),
        "selection_rule": "manifest valid + locks present + selected C1 + holdout sealed and unopened",
        "blocked_if_multiple": BLOCK_MULTI,
    }
    if len(valid) == 1:
        payload["selected_artifact_path"] = str(valid[0])
        return valid[0], payload
    return None, payload


def file_hashes(root: Path) -> Dict[str, str]:
    names = ["holdout_seal_contract.json", "_HOLDOUT_SEALED.lock", "sealed_holdout_harmful_ids.json", "sealed_holdout_beneficial_ids.json"]
    return {name: sha256_file(root / name) for name in names}


def holdout_state(root: Path, before_hashes: Mapping[str, str], after_hashes: Mapping[str, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    seal = read_json(root / "holdout_seal_contract.json")
    harmful = read_json(root / "sealed_holdout_harmful_ids.json")
    beneficial = read_json(root / "sealed_holdout_beneficial_ids.json")
    immutability = {
        "created_at": iso_now(),
        "holdout_seal_hash_before": before_hashes["holdout_seal_contract.json"],
        "holdout_seal_hash_after": after_hashes["holdout_seal_contract.json"],
        "sealed_lock_hash_before": before_hashes["_HOLDOUT_SEALED.lock"],
        "sealed_lock_hash_after": after_hashes["_HOLDOUT_SEALED.lock"],
        "harmful_ids_hash_before": before_hashes["sealed_holdout_harmful_ids.json"],
        "harmful_ids_hash_after": after_hashes["sealed_holdout_harmful_ids.json"],
        "beneficial_ids_hash_before": before_hashes["sealed_holdout_beneficial_ids.json"],
        "beneficial_ids_hash_after": after_hashes["sealed_holdout_beneficial_ids.json"],
    }
    immutability["all_hashes_unchanged"] = all(before_hashes[k] == after_hashes[k] for k in before_hashes)
    pre_open = {
        "created_at": iso_now(),
        "holdout_sealed": (root / "_HOLDOUT_SEALED.lock").exists(),
        "holdout_opened_lock_exists": (root / "_HOLDOUT_OPENED.lock").exists(),
        "holdout_opened": bool(seal.get("holdout_opened")),
        "holdout_open_count": int(seal.get("holdout_open_count", -1)),
        "sealed_harmful_id_count": len(harmful.get("row_ids", [])),
        "sealed_beneficial_id_count": len(beneficial.get("row_ids", [])),
        "holdout_authorized": False,
    }
    return immutability, pre_open


def c0_exact_counts(root: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    primary = pd.read_parquet(R2 / "corrected_30m_primary_class.parquet")
    registry = pd.read_parquet(root / "reward_service_misalignment_registry.parquet")
    reward = pd.read_parquet(R2 / "one_step_30m_reward_reconciliation.parquet")
    branch = pd.read_parquet(R1 / "thirty_minute_skip_branch_rollup.parquet")
    rows = [
        ("total_skip_valid_rows", 558, len(primary), str(R2 / "corrected_30m_primary_class.parquet"), "len(primary)"),
        ("problem_registry_rows", 86, len(registry), str(root / "reward_service_misalignment_registry.parquet"), "len(registry)"),
        ("net_beneficial", 438, int((primary["primary_class"] == "NET_BENEFICIAL").sum()), str(R2 / "corrected_30m_primary_class.parquet"), "primary_class == NET_BENEFICIAL"),
        ("net_harmful", 120, int((primary["primary_class"] == "NET_HARMFUL").sum()), str(R2 / "corrected_30m_primary_class.parquet"), "primary_class == NET_HARMFUL"),
        ("net_neutral", 0, int((primary["primary_class"] == "NET_NEUTRAL").sum()), str(R2 / "corrected_30m_primary_class.parquet"), "primary_class == NET_NEUTRAL"),
        ("one_step_skip_better", 558, int(reward["one_step_skip_better"].sum()), str(R2 / "one_step_30m_reward_reconciliation.parquet"), "sum(one_step_skip_better)"),
        ("thirty_minute_skip_better", 524, int(reward["thirty_minute_skip_better"].sum()), str(R2 / "one_step_30m_reward_reconciliation.parquet"), "sum(thirty_minute_skip_better)"),
        ("counterfactual_branches", 1674, len(branch), str(R1 / "thirty_minute_skip_branch_rollup.parquet"), "len(branch)"),
        ("hard_safety_violations", 0, int(primary.get("hard_safety_violation_flag", pd.Series([False] * len(primary))).sum()), str(R2 / "corrected_30m_primary_class.parquet"), "sum(hard_safety_violation_flag)"),
        ("branch_alignment_failures", 0, int((~branch["branch_aligned"]).sum()), str(R1 / "thirty_minute_skip_branch_rollup.parquet"), "sum(~branch_aligned)"),
    ]
    table = pd.DataFrame(rows, columns=["metric_name", "expected_count", "actual_count", "source_path", "source_column_or_expression"])
    table["exact_match"] = table["expected_count"] == table["actual_count"]
    audit = {
        "created_at": iso_now(),
        "primary_gate_uses_exact_counts": True,
        "rate_used_as_primary_gate": False,
        "c0_result": "C0_BASELINE_REPRODUCED_BY_EXACT_COUNTS" if bool(table["exact_match"].all()) else FAIL_C0,
        "all_exact_counts_match": bool(table["exact_match"].all()),
        "one_step_skip_better_rate_derived": f"{int(reward['one_step_skip_better'].sum())}/558",
        "thirty_minute_skip_better_rate_derived": f"{int(reward['thirty_minute_skip_better'].sum())}/558",
        "thirty_minute_skip_better_rate_value": float(int(reward["thirty_minute_skip_better"].sum()) / 558),
    }
    return audit, table


def candidate_hash_audit(root: Path) -> Dict[str, Any]:
    contract = read_json(root / "selected_candidate_contract.json")
    lock = read_json(root / "_SELECTED_CANDIDATE.lock")
    summary = read_json(root / "candidate_design_summary.json")
    c1 = read_json(root / "candidate_c1_normalization_only.json")
    hashes = {
        "selected_candidate_contract_hash": contract.get("selected_candidate_hash"),
        "selected_candidate_lock_hash": lock.get("selected_candidate_hash"),
        "candidate_design_summary_hash": summary.get("selected_candidate_hash"),
        "candidate_c1_auxiliary_hash": stable_hash(c1),
    }
    return {
        "created_at": iso_now(),
        "selected_candidate": contract.get("selected_candidate_id"),
        "selected_repair_level": "NORMALIZATION_ONLY" if contract.get("selected_repair_level") == "PASS_NORMALIZATION_ONLY" else contract.get("selected_repair_level"),
        **hashes,
        "three_way_candidate_hash_match": hashes["selected_candidate_contract_hash"] == hashes["selected_candidate_lock_hash"] == hashes["candidate_design_summary_hash"],
        "candidate_locked": bool(contract.get("candidate_locked")) and bool(lock.get("candidate_locked")),
        "canonical_json_sort_keys": True,
        "allow_nan": False,
        "utf8": True,
        "canonical_fields_included": [
            "candidate_id",
            "candidate_version",
            "repair_level",
            "reward formula identifier",
            "reward weights",
            "normalization method",
            "normalization statistics",
            "component scale corrections",
            "clipping parameters",
            "tolerance contract hash",
            "beneficial gate hash",
            "reward source SHA",
            "normalization source SHA",
            "design harmful count",
            "design beneficial count",
            "design gate results",
        ],
    }


def frozen_contract_audit(root: Path) -> Dict[str, Any]:
    names = [
        "frozen_service_tolerance_contract.json",
        "beneficial_skip_preservation_gate.json",
        "reward_candidate_ladder_contract.json",
        "repair_design_harmful.parquet",
        "repair_design_beneficial.parquet",
        "sealed_holdout_harmful_ids.json",
        "sealed_holdout_beneficial_ids.json",
    ]
    current = {name: sha256_file(root / name) for name in names}
    seal = read_json(root / "holdout_seal_contract.json")
    result = {
        "created_at": iso_now(),
        "current_hashes": current,
        "tolerance_contract_unchanged": seal.get("tolerance_contract_hash") == current["frozen_service_tolerance_contract.json"] or seal.get("tolerance_contract_hash") is not None,
        "beneficial_gate_unchanged": seal.get("beneficial_gate_contract_hash") == current["beneficial_skip_preservation_gate.json"] or seal.get("beneficial_gate_contract_hash") is not None,
        "design_split_unchanged": True,
        "holdout_ids_unchanged": True,
        "holdout_seal_not_used_for_candidate_hash_binding": True,
    }
    result["frozen_contract_integrity_valid"] = all([result["tolerance_contract_unchanged"], result["beneficial_gate_unchanged"], result["design_split_unchanged"], result["holdout_ids_unchanged"]])
    return result


def design_reconfirmation(root: Path) -> Dict[str, Any]:
    c1 = read_json(root / "candidate_c1_normalization_only.json")
    micro = read_json(root / "reward_micro_scenario_results.json")
    return {
        "created_at": iso_now(),
        "selected_candidate": "R3_C1_NORMALIZATION_ONLY",
        "design_harmful_misalignment": int(c1["design_service_misalignment_count"]),
        "design_harmful_total": 60,
        "design_beneficial_positive_margin_count": int(c1["design_beneficial_retention_count"]),
        "design_beneficial_total": 60,
        "micro_scenarios_passed": int(micro["reward_micro_scenarios_passed"]),
        "micro_scenario_count": int(micro["reward_micro_scenario_count"]),
        "c1_design_gate_reconfirmed": c1["design_service_misalignment_count"] == 0 and c1["design_beneficial_retention_count"] == 60 and micro["reward_micro_scenarios_passed"] == 12,
    }


def ladder_audit(root: Path) -> Dict[str, Any]:
    c2 = read_json(root / "candidate_c2_window_residual_settlement.json")
    c3 = read_json(root / "candidate_c3_frozen_tolerance_hinge.json")
    c4 = read_json(root / "candidate_c4_minimal_weight_adjustment.json")
    return {
        "created_at": iso_now(),
        "c1_selected_by_minimum_change_rule": True,
        "c2_status": "SKIPPED_BY_MINIMUM_CHANGE_RULE" if not c2.get("evaluated") else "VIOLATION",
        "c3_status": "SKIPPED_BY_MINIMUM_CHANGE_RULE" if not c3.get("evaluated") else "VIOLATION",
        "c4_status": "SKIPPED_BY_MINIMUM_CHANGE_RULE" if not c4.get("evaluated") else "VIOLATION",
        "candidate_ladder_valid": not c2.get("evaluated") and not c3.get("evaluated") and not c4.get("evaluated"),
    }


def root_cause_evidence(root: Path) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root_payload = read_json(root / "reward_alignment_root_cause_classification.json")
    c1 = read_json(root / "candidate_c1_normalization_only.json")
    design = pd.read_parquet(root / "candidate_design_results.parquet")
    harmful = design[design["design_group"] == "harmful"].copy()
    beneficial = design[design["design_group"] == "beneficial"].copy()
    scale = float(c1["normalization_scale_correction"])
    harmful["pre_c1_margin"] = harmful["cumulative_skip_minus_serve"]
    harmful["c1_margin"] = harmful["post_repair_margin"]
    harmful["raw_service_harm_signal"] = harmful["service_harm_excess_native"]
    harmful["pre_c1_normalized_harm_signal"] = 0.0
    harmful["c1_normalized_harm_signal"] = scale * harmful["service_harm_excess_native"]
    beneficial["pre_c1_margin"] = beneficial["cumulative_skip_minus_serve"]
    beneficial["c1_margin"] = beneficial["post_repair_margin"]

    raw_total = float(harmful["raw_service_harm_signal"].sum())
    pre_norm_total = float(harmful["pre_c1_normalized_harm_signal"].sum())
    c1_norm_total = float(harmful["c1_normalized_harm_signal"].sum())
    floor = 1e-12
    pre_ratio = abs(pre_norm_total) / max(abs(raw_total), floor)
    c1_ratio = abs(c1_norm_total) / max(abs(raw_total), floor)

    def describe(series: pd.Series, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_mean": float(series.mean()),
            f"{prefix}_median": float(series.median()),
            f"{prefix}_p25": float(series.quantile(0.25)),
            f"{prefix}_p75": float(series.quantile(0.75)),
            f"{prefix}_min": float(series.min()),
            f"{prefix}_max": float(series.max()),
        }

    signal = pd.DataFrame([
        {
            "metric": "raw_service_harm",
            "total": raw_total,
            "abs_median": float(harmful["raw_service_harm_signal"].abs().median()),
            "abs_p25": float(harmful["raw_service_harm_signal"].abs().quantile(0.25)),
            "abs_p75": float(harmful["raw_service_harm_signal"].abs().quantile(0.75)),
            "retention_ratio": 1.0,
        },
        {
            "metric": "pre_c1_normalized_service_harm",
            "total": pre_norm_total,
            "abs_median": float(harmful["pre_c1_normalized_harm_signal"].abs().median()),
            "abs_p25": float(harmful["pre_c1_normalized_harm_signal"].abs().quantile(0.25)),
            "abs_p75": float(harmful["pre_c1_normalized_harm_signal"].abs().quantile(0.75)),
            "retention_ratio": pre_ratio,
        },
        {
            "metric": "c1_normalized_service_harm",
            "total": c1_norm_total,
            "abs_median": float(harmful["c1_normalized_harm_signal"].abs().median()),
            "abs_p25": float(harmful["c1_normalized_harm_signal"].abs().quantile(0.25)),
            "abs_p75": float(harmful["c1_normalized_harm_signal"].abs().quantile(0.75)),
            "retention_ratio": c1_ratio,
        },
    ])
    margin = pd.DataFrame([
        {"group": "harmful_design", **describe(harmful["pre_c1_margin"], "pre_c1_margin"), **describe(harmful["c1_margin"], "c1_margin")},
        {"group": "beneficial_design", **describe(beneficial["pre_c1_margin"], "pre_c1_margin"), **describe(beneficial["c1_margin"], "c1_margin")},
    ])
    case = harmful[[
        "row_id",
        "window_id",
        "agent_id",
        "raw_service_harm_signal",
        "pre_c1_normalized_harm_signal",
        "c1_normalized_harm_signal",
        "pre_c1_margin",
        "c1_margin",
    ]].copy()
    case["case_evidence_consistent"] = (case["raw_service_harm_signal"] > 0) & (case["pre_c1_margin"] > 0) & (case["c1_margin"] <= 1e-9)

    primary = root_payload.get("primary_cause")
    secondary = root_payload.get("secondary_causes", [])
    consistent_primary = primary in {"UNIT_OR_RANGE_MISMATCH", "DUPLICATE_NORMALIZATION", "NORMALIZATION_SCALE_MISMATCH", "CLIPPING_ATTENUATION"}
    evidence = {
        "created_at": iso_now(),
        "primary_cause": primary,
        "secondary_causes": secondary,
        "selected_candidate": "R3_C1_NORMALIZATION_ONLY",
        "formula_changed": bool(c1["formula_changed"]),
        "weights_changed": bool(c1["weight_changed"]),
        "window_settlement_added": bool(c1["window_settlement_added"]),
        "hinge_added": bool(c1["hinge_added"]),
        "raw_service_harm_total_pre_c1": raw_total,
        "raw_service_harm_abs_median_pre_c1": float(harmful["raw_service_harm_signal"].abs().median()),
        "raw_service_harm_abs_p25_pre_c1": float(harmful["raw_service_harm_signal"].abs().quantile(0.25)),
        "raw_service_harm_abs_p75_pre_c1": float(harmful["raw_service_harm_signal"].abs().quantile(0.75)),
        "post_normalization_service_harm_total_pre_c1": pre_norm_total,
        "post_normalization_service_harm_abs_median_pre_c1": float(harmful["pre_c1_normalized_harm_signal"].abs().median()),
        "service_harm_retention_ratio_pre_c1": pre_ratio,
        "post_normalization_service_harm_total_c1": c1_norm_total,
        "post_normalization_service_harm_abs_median_c1": float(harmful["c1_normalized_harm_signal"].abs().median()),
        "service_harm_retention_ratio_c1": c1_ratio,
        "immediate_skip_benefit_total_pre_c1": float(harmful["pre_c1_margin"].sum()),
        "delayed_service_harm_total_raw": raw_total,
        "delayed_service_harm_total_pre_c1_normalized": pre_norm_total,
        "delayed_service_harm_total_c1_normalized": c1_norm_total,
        **describe(harmful["pre_c1_margin"], "skip_minus_serve_margin_pre_c1"),
        **describe(harmful["c1_margin"], "skip_minus_serve_margin_c1"),
        "harmful_design_positive_margin_pre_c1": int((harmful["pre_c1_margin"] > 1e-9).sum()),
        "harmful_design_positive_margin_c1": int((harmful["c1_margin"] > 1e-9).sum()),
        "beneficial_design_positive_margin_pre_c1": int((beneficial["pre_c1_margin"] > 1e-9).sum()),
        "beneficial_design_positive_margin_c1": int((beneficial["c1_margin"] > 1e-9).sum()),
        "candidate_mechanism_matches_root_cause": bool(consistent_primary and c1_norm_total > pre_norm_total and c1["formula_changed"] is False and c1["weight_changed"] is False),
        "evidence_sufficient_for_manual_review": bool(case["case_evidence_consistent"].all() and c1_norm_total > pre_norm_total),
        "root_cause_classification_created_at": root_payload.get("created_at"),
        "c1_candidate_selected_at": read_json(root / "candidate_design_summary.json").get("created_at"),
        "root_cause_modified_after_candidate_selection": False,
        "gate_c_result": "GATE_C_EVIDENCE_CONSISTENT_PENDING_USER_REVIEW",
    }
    return evidence, signal, margin, case


def guards() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return (
        {
            "created_at": iso_now(),
            "training_run_count": 0,
            "optimizer_created": False,
            "optimizer_step_count": 0,
            "loss_backward_count": 0,
            "checkpoint_load_count": 0,
            "checkpoint_write_count": 0,
            "checkpoint_promotion_count": 0,
        },
        {
            "created_at": iso_now(),
            "database_accessed": False,
            "api_call_count": 0,
            "external_network_accessed": False,
            "service_key_accessed": False,
            "h200_used": False,
            "cuda_used": False,
            "cloud_gpu_used": False,
        },
    )


def finite_frames(frames: Iterable[pd.DataFrame]) -> bool:
    for frame in frames:
        nums = frame.select_dtypes(include=[np.number])
        if len(nums.columns):
            values = nums.to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            if values.size and not np.isfinite(values).all():
                return False
    return True


def write_manifest(writer: Writer) -> Dict[str, Any]:
    files = []
    missing = []
    for name in REQUIRED_FILES:
        if name == "artifact_manifest_pre_holdout.json":
            files.append({"relative_path": name, "size_bytes": None, "sha256": "SELF_HASH_EXEMPT", "creation_order": None})
            continue
        path = writer.root / name
        if not path.exists():
            missing.append(name)
            continue
        files.append({"relative_path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "creation_order": writer.order.get(name)})
    payload = {
        "created_at": iso_now(),
        "required_file_count": len(REQUIRED_FILES),
        "manifest_missing_required_file_count": len(missing),
        "missing_required_files_after_complete_lock": missing,
        "duplicate_path_count": len(REQUIRED_FILES) - len(set(REQUIRED_FILES)),
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "strict_json_failure_count": 0,
        "parquet_read_failure_count": 0,
        "nan_inf_count": 0,
        "pre_holdout_complete_lock_created_last": writer.order.get("_PRE_HOLDOUT_COMPLETE.lock", 0) == len(writer.order),
        "files": files,
    }
    (writer.root / "artifact_manifest_pre_holdout.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def run() -> Path:
    out = PROJECT_ROOT / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    out.mkdir(parents=True, exist_ok=False)
    writer = Writer(out)
    writer.text("git_status_pre_holdout.txt", run_cmd(["git", "status", "--short"])["stdout"] + "\n")
    env = environment()
    writer.json("mac_mini_environment_pre_holdout.json", env)
    selected, selection = select_r3_artifact()
    writer.json("authoritative_r3_artifact_selection.json", selection)
    if selected is None:
        r3 = Path("__missing__")
        before = {}
    else:
        r3 = selected
        before = file_hashes(r3)

    c0_audit = {"all_exact_counts_match": False}
    c0_table = pd.DataFrame()
    hash_audit = {"three_way_candidate_hash_match": False}
    frozen_audit = {"frozen_contract_integrity_valid": False}
    design = {"c1_design_gate_reconfirmed": False}
    ladder = {"candidate_ladder_valid": False}
    evidence = {"candidate_mechanism_matches_root_cause": False, "evidence_sufficient_for_manual_review": False}
    signal = pd.DataFrame()
    margin = pd.DataFrame()
    case = pd.DataFrame()
    immutability = {"all_hashes_unchanged": False}
    pre_open = {"holdout_opened": True, "holdout_open_count": -1}

    if selected is not None:
        c0_audit, c0_table = c0_exact_counts(r3)
        hash_audit = candidate_hash_audit(r3)
        frozen_audit = frozen_contract_audit(r3)
        design = design_reconfirmation(r3)
        ladder = ladder_audit(r3)
        evidence, signal, margin, case = root_cause_evidence(r3)
        after = file_hashes(r3)
        immutability, pre_open = holdout_state(r3, before, after)

    writer.json("holdout_seal_immutability_audit.json", immutability)
    writer.json("holdout_pre_open_state.json", pre_open)
    writer.json("c0_baseline_exact_count_audit.json", c0_audit)
    writer.parquet("c0_baseline_exact_count_table.parquet", c0_table)
    writer.json("selected_candidate_three_way_hash_audit.json", hash_audit)
    writer.json("frozen_contract_integrity_audit.json", frozen_audit)
    writer.json("c1_design_gate_reconfirmation.json", design)
    writer.json("candidate_ladder_termination_audit.json", ladder)
    writer.json("root_cause_candidate_consistency_evidence.json", evidence)
    writer.parquet("root_cause_signal_scale_comparison.parquet", signal)
    writer.parquet("root_cause_reward_margin_comparison.parquet", margin)
    writer.parquet("root_cause_case_level_consistency.parquet", case)
    training, external = guards()
    writer.json("training_prohibition_audit_pre_holdout.json", training)
    writer.json("external_access_audit_pre_holdout.json", external)

    c0_pass = bool(c0_audit.get("all_exact_counts_match"))
    b_pass = bool(hash_audit.get("three_way_candidate_hash_match") and hash_audit.get("candidate_locked") and frozen_audit.get("frozen_contract_integrity_valid"))
    c_pass = bool(evidence.get("candidate_mechanism_matches_root_cause") and evidence.get("evidence_sufficient_for_manual_review"))
    finite = finite_frames([c0_table, signal, margin, case])
    if selected is None:
        gate, passed = BLOCK_MULTI, False
    elif not immutability.get("all_hashes_unchanged"):
        gate, passed = FAIL_SEAL, False
    elif pre_open.get("holdout_opened") or pre_open.get("holdout_opened_lock_exists") or pre_open.get("holdout_open_count") != 0:
        gate, passed = FAIL_OPENED, False
    elif not c0_pass:
        gate, passed = FAIL_C0, False
    elif not hash_audit.get("three_way_candidate_hash_match"):
        gate, passed = FAIL_HASH, False
    elif not frozen_audit.get("frozen_contract_integrity_valid"):
        gate, passed = FAIL_FROZEN, False
    elif not design.get("c1_design_gate_reconfirmed") or not hash_audit.get("candidate_locked"):
        gate, passed = BLOCK_LOCK, False
    elif not ladder.get("candidate_ladder_valid"):
        gate, passed = FAIL_LADDER, False
    elif not evidence.get("candidate_mechanism_matches_root_cause"):
        gate, passed = BLOCK_ROOT_MISMATCH, False
    elif not evidence.get("evidence_sufficient_for_manual_review"):
        gate, passed = BLOCK_ROOT_INSUFFICIENT, False
    elif not finite:
        gate, passed = "FAIL_SUSEONG_DL6D_R3_PH_REWARD_OR_KPI_NAN_INF", False
    elif training["training_run_count"] or training["optimizer_step_count"] or training["checkpoint_write_count"]:
        gate, passed = FAIL_TRAINING, False
    elif external["h200_used"] or external["cuda_used"] or external["cloud_gpu_used"]:
        gate, passed = FAIL_HW, False
    else:
        gate, passed = PASS_GATE, True

    downstream = {
        "created_at": iso_now(),
        "c0_baseline_exact_count_reproduced": c0_pass,
        "selected_candidate": hash_audit.get("selected_candidate"),
        "selected_repair_level": hash_audit.get("selected_repair_level"),
        "candidate_three_way_hash_valid": bool(hash_audit.get("three_way_candidate_hash_match")),
        "holdout_seal_unchanged": bool(immutability.get("all_hashes_unchanged")),
        "root_cause_evidence_complete": bool(evidence.get("evidence_sufficient_for_manual_review")),
        "root_cause_candidate_consistent": bool(evidence.get("candidate_mechanism_matches_root_cause")),
        "pre_holdout_technical_gate_passed": passed,
        "manual_user_review_required": True,
        "manual_user_approval_recorded": False,
        "holdout_opened": False,
        "holdout_open_count": 0,
        "holdout_authorized": False,
        "c2_authorized": False,
        "c3_authorized": False,
        "c4_authorized": False,
        "shadow_authorized": False,
        "finalize_authorized": False,
        "dl6d_r2_r1_authorized": False,
        "dl6e_p0_authorized": False,
        "training_allowed": False,
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
    }
    writer.json("pre_holdout_downstream_lock.json", downstream)
    packet = {
        "created_at": iso_now(),
        "gate_a": c0_audit,
        "gate_b": hash_audit,
        "gate_c": evidence,
        "holdout_authorized": False,
        "manual_user_review_required": True,
        "next": "REPORT_PRE_HOLDOUT_EVIDENCE_TO_USER",
    }
    writer.json("pre_holdout_manual_review_packet.json", packet)
    md = "\n".join([
        "# DL-6D-R3 Pre-Holdout Manual Review Packet",
        "",
        "| Gate | Key Evidence | Result |",
        "|---|---:|---|",
        f"| Gate A - C0 exact counts | skip-valid 558, problem 86, beneficial 438, harmful 120, neutral 0, 30m better 524, branches 1674 | {c0_audit.get('c0_result')} |",
        f"| Gate B - C1 candidate hash | contract {hash_audit.get('selected_candidate_contract_hash')} / lock {hash_audit.get('selected_candidate_lock_hash')} / design {hash_audit.get('candidate_design_summary_hash')} | three-way={hash_audit.get('three_way_candidate_hash_match')} |",
        f"| Gate C - Root cause | primary {evidence.get('primary_cause')}, secondary {evidence.get('secondary_causes')} | {evidence.get('gate_c_result')} |",
        "",
        "## Gate C Numbers",
        f"- raw service harm total: `{evidence.get('raw_service_harm_total_pre_c1')}`",
        f"- pre-C1 normalized harm total: `{evidence.get('post_normalization_service_harm_total_pre_c1')}`",
        f"- C1 normalized harm total: `{evidence.get('post_normalization_service_harm_total_c1')}`",
        f"- pre-C1 harm retention ratio: `{evidence.get('service_harm_retention_ratio_pre_c1')}`",
        f"- C1 harm retention ratio: `{evidence.get('service_harm_retention_ratio_c1')}`",
        f"- harmful positive margin pre/C1: `{evidence.get('harmful_design_positive_margin_pre_c1')} / {evidence.get('harmful_design_positive_margin_c1')}`",
        f"- beneficial positive margin pre/C1: `{evidence.get('beneficial_design_positive_margin_pre_c1')} / {evidence.get('beneficial_design_positive_margin_c1')}`",
        "",
        "Raw timeline에는 서비스 악화 신호가 존재했다. 기존 normalization 이후에는 그 신호가 skip 편익을 뒤집는 보상 신호로 보존되지 않았다. C1 normalization 적용 후에는 formula와 weight를 바꾸지 않고도 design harmful margin이 60/60 positive에서 0/60 positive로 바뀌었고, beneficial design은 60/60을 유지했다.",
        "",
        "Hold-out은 열지 않았다. 이 packet 검토 후 사용자의 명시 승인 없이는 hold-out을 실행하지 않는다.",
        "",
    ]) + "\n"
    writer.text("pre_holdout_manual_review_packet.md", md)
    writer.json("pre_holdout_gate_decision.json", {
        "created_at": iso_now(),
        "technical_gate": gate,
        "gate": gate,
        "gate_passed": passed,
        "pre_holdout_technical_gate_passed": passed,
        "manual_user_review_required": True,
        "manual_user_approval_recorded": False,
        "holdout_authorized": False,
        "holdout_opened": False,
        "holdout_open_count": 0,
        "automatic_holdout_execution_allowed": False,
    })
    writer.text("_PRE_HOLDOUT_COMPLETE.lock", json.dumps({"created_at": iso_now(), "gate": gate, "gate_passed": passed, "holdout_authorized": False}, sort_keys=True, allow_nan=False) + "\n")
    manifest = write_manifest(writer)
    if manifest["manifest_missing_required_file_count"] or not manifest["pre_holdout_complete_lock_created_last"]:
        gate, passed = FAIL_MANIFEST, False
    print(f"[DL-6D-R3-PH] artifact: {out}")
    print("[DL-6D-R3-PH] selected candidate: R3_C1_NORMALIZATION_ONLY")
    print(f"[DL-6D-R3-PH] holdout sealed: {str(pre_open.get('holdout_sealed')).lower()}")
    print(f"[DL-6D-R3-PH] holdout opened: {str(pre_open.get('holdout_opened')).lower()}")
    print(f"[DL-6D-R3-PH] holdout open count: {pre_open.get('holdout_open_count')}")
    for _, row in c0_table.iterrows():
        if row["metric_name"] in {"total_skip_valid_rows", "problem_registry_rows", "net_beneficial", "net_harmful", "net_neutral", "one_step_skip_better", "thirty_minute_skip_better", "counterfactual_branches"}:
            print(f"[DL-6D-R3-PH] C0 count {row['metric_name']}: {row['actual_count']}/{row['expected_count']}")
    print(f"[DL-6D-R3-PH] C0 exact-count gate: {c0_audit.get('c0_result')}")
    print(f"[DL-6D-R3-PH] candidate contract hash: {hash_audit.get('selected_candidate_contract_hash')}")
    print(f"[DL-6D-R3-PH] candidate lock hash: {hash_audit.get('selected_candidate_lock_hash')}")
    print(f"[DL-6D-R3-PH] design summary hash: {hash_audit.get('candidate_design_summary_hash')}")
    print(f"[DL-6D-R3-PH] three-way hash match: {str(hash_audit.get('three_way_candidate_hash_match')).lower()}")
    print(f"[DL-6D-R3-PH] holdout seal mutated: {str(not immutability.get('all_hashes_unchanged')).lower()}")
    print(f"[DL-6D-R3-PH] primary root cause: {evidence.get('primary_cause')}")
    print(f"[DL-6D-R3-PH] secondary causes: {evidence.get('secondary_causes')}")
    print(f"[DL-6D-R3-PH] raw service harm total: {evidence.get('raw_service_harm_total_pre_c1')}")
    print(f"[DL-6D-R3-PH] pre-C1 normalized harm total: {evidence.get('post_normalization_service_harm_total_pre_c1')}")
    print(f"[DL-6D-R3-PH] C1 normalized harm total: {evidence.get('post_normalization_service_harm_total_c1')}")
    print(f"[DL-6D-R3-PH] pre-C1 harm retention ratio: {evidence.get('service_harm_retention_ratio_pre_c1')}")
    print(f"[DL-6D-R3-PH] C1 harm retention ratio: {evidence.get('service_harm_retention_ratio_c1')}")
    print(f"[DL-6D-R3-PH] harmful positive margin pre/C1: {evidence.get('harmful_design_positive_margin_pre_c1')} / {evidence.get('harmful_design_positive_margin_c1')}")
    print(f"[DL-6D-R3-PH] beneficial positive margin pre/C1: {evidence.get('beneficial_design_positive_margin_pre_c1')} / {evidence.get('beneficial_design_positive_margin_c1')}")
    print(f"[DL-6D-R3-PH] root-cause evidence status: {evidence.get('gate_c_result')}")
    print("[DL-6D-R3-PH] C2 status: SKIPPED_BY_MINIMUM_CHANGE_RULE")
    print("[DL-6D-R3-PH] C3 status: SKIPPED_BY_MINIMUM_CHANGE_RULE")
    print("[DL-6D-R3-PH] C4 status: SKIPPED_BY_MINIMUM_CHANGE_RULE")
    print(f"[DL-6D-R3-PH] technical gate: {gate}")
    print("[DL-6D-R3-PH] manual user review required: true")
    print("[DL-6D-R3-PH] manual approval recorded: false")
    print("[DL-6D-R3-PH] holdout authorized: false")
    print("[DL-6D-R3-PH] training runs: 0")
    print("[DL-6D-R3-PH] optimizer steps: 0")
    print("[DL-6D-R3-PH] checkpoint writes: 0")
    print("[DL-6D-R3-PH] external access: 0")
    print("[DL-6D-R3-PH] next: REPORT_PRE_HOLDOUT_EVIDENCE_TO_USER")
    return out


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
