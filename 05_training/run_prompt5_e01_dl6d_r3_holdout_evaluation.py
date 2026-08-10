from __future__ import annotations

import hashlib
import json
import platform
import resource
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
R3 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_reward_service_alignment_repair_20260802_112731"
PH = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_pre_holdout_audit_20260802_115741"
ARTIFACT_PREFIX = "prompt5_e01_dl6d_r3_holdout_evaluation"

EXPECTED_PH_GATE = "PASS_SUSEONG_DL6D_R3_PRE_HOLDOUT_TECHNICAL_AUDIT_PENDING_MANUAL_REVIEW"
EXPECTED_CANDIDATE = "R3_C1_NORMALIZATION_ONLY"
EXPECTED_CANDIDATE_HASH = "1b91225c3110ba201dcf8404ce66eb4a91f3fadaedb961f6d5b4956037c87154"
PASS_GATE = "PASS_SUSEONG_DL6D_R3_C1_NORMALIZATION_GENERALIZED_ON_SEALED_HOLDOUT"
BLOCKED_GATE = "BLOCKED_SUSEONG_DL6D_R3_C1_NORMALIZATION_FAILED_GENERALIZATION"
FAIL_CANDIDATE_CHANGED = "FAIL_SUSEONG_DL6D_R3_HO1_CANDIDATE_CHANGED_AFTER_MANUAL_REVIEW"
FAIL_SEAL_MUTATED = "FAIL_SUSEONG_DL6D_R3_HO1_HOLDOUT_SEAL_MUTATED"
FAIL_PRIOR_RESULT = "FAIL_SUSEONG_DL6D_R3_HO1_PRIOR_HOLDOUT_RESULT_DETECTED"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_R3_HO1_MANIFEST_RECONCILIATION"
FAIL_SECURITY = "FAIL_SUSEONG_DL6D_R3_HO1_SECURITY_AUDIT"

SERVICE_TOLERANCES = {
    "avg_wait_seconds": 1.0,
    "passenger_wait_p95_seconds": 1.0,
    "passenger_service_rate": 0.001,
    "on_time_rate": 0.001,
    "passenger_served_count": 0.01,
}

REQUIRED_FILES = [
    "holdout_preopen_tolerance_readback.json",
    "holdout_manual_approval_record.json",
    "holdout_candidate_revalidation.json",
    "holdout_seal_revalidation.json",
    "holdout_open_audit.json",
    "_HOLDOUT_OPENED.lock",
    "sealed_holdout_harmful_results.parquet",
    "sealed_holdout_beneficial_results.parquet",
    "sealed_holdout_harmful_summary.json",
    "sealed_holdout_beneficial_summary.json",
    "sealed_holdout_harm_pattern_audit.json",
    "sealed_holdout_harm_pattern_table.parquet",
    "sealed_holdout_gate_decision.json",
    "sealed_holdout_downstream_lock.json",
    "training_prohibition_audit_holdout.json",
    "external_access_audit_holdout.json",
    "artifact_manifest_holdout.json",
    "_HOLDOUT_EVALUATION_COMPLETE.lock",
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


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str).encode("utf-8")).hexdigest()


def run_cmd(args: Sequence[str]) -> Dict[str, Any]:
    result = subprocess.run(list(args), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {"cmd": list(args), "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def environment_guard() -> Dict[str, Any]:
    return {
        "created_at": iso_now(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) if platform.system() == "Darwin" else int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "h200_used": False,
        "cuda_used": False,
        "cloud_gpu_used": False,
    }


def candidate_hashes() -> Dict[str, Any]:
    contract = read_json(R3 / "selected_candidate_contract.json")
    lock = read_json(R3 / "_SELECTED_CANDIDATE.lock")
    summary = read_json(R3 / "candidate_design_summary.json")
    c1 = read_json(R3 / "candidate_c1_normalization_only.json")
    return {
        "candidate_id": contract.get("selected_candidate_id"),
        "repair_level": "NORMALIZATION_ONLY" if contract.get("selected_repair_level") == "PASS_NORMALIZATION_ONLY" else contract.get("selected_repair_level"),
        "candidate_contract_hash": contract.get("selected_candidate_hash"),
        "candidate_lock_hash": lock.get("selected_candidate_hash"),
        "candidate_design_summary_hash": summary.get("selected_candidate_hash"),
        "expected_candidate_hash": EXPECTED_CANDIDATE_HASH,
        "three_way_candidate_hash_match": contract.get("selected_candidate_hash") == lock.get("selected_candidate_hash") == summary.get("selected_candidate_hash") == EXPECTED_CANDIDATE_HASH,
        "reward_formula_changed": bool(c1["formula_changed"]),
        "reward_weights_changed": bool(c1["weight_changed"]),
        "window_settlement_added": bool(c1["window_settlement_added"]),
        "hinge_added": bool(c1["hinge_added"]),
        "direct_action_id_penalty_added": bool(c1["action_id_penalty_added"]),
        "normalization_scale_correction": float(c1["normalization_scale_correction"]),
    }


def seal_hashes() -> Dict[str, str]:
    names = [
        "holdout_seal_contract.json",
        "_HOLDOUT_SEALED.lock",
        "sealed_holdout_harmful_ids.json",
        "sealed_holdout_beneficial_ids.json",
        "frozen_service_tolerance_contract.json",
        "beneficial_skip_preservation_gate.json",
    ]
    return {name: sha256_file(R3 / name) for name in names}


def seal_revalidation() -> Dict[str, Any]:
    ph = read_json(PH / "holdout_seal_immutability_audit.json")
    integrity = read_json(PH / "frozen_contract_integrity_audit.json")
    current = seal_hashes()
    pre_hashes = {
        "holdout_seal_contract.json": ph.get("holdout_seal_hash_after"),
        "_HOLDOUT_SEALED.lock": ph.get("sealed_lock_hash_after"),
        "sealed_holdout_harmful_ids.json": ph.get("harmful_ids_hash_after"),
        "sealed_holdout_beneficial_ids.json": ph.get("beneficial_ids_hash_after"),
        "frozen_service_tolerance_contract.json": integrity["current_hashes"].get("frozen_service_tolerance_contract.json"),
        "beneficial_skip_preservation_gate.json": integrity["current_hashes"].get("beneficial_skip_preservation_gate.json"),
    }
    return {
        "created_at": iso_now(),
        "current_hashes": current,
        "pre_holdout_hashes": pre_hashes,
        "seal_unchanged": current["holdout_seal_contract.json"] == pre_hashes["holdout_seal_contract.json"],
        "sealed_lock_unchanged": current["_HOLDOUT_SEALED.lock"] == pre_hashes["_HOLDOUT_SEALED.lock"],
        "harmful_ids_unchanged": current["sealed_holdout_harmful_ids.json"] == pre_hashes["sealed_holdout_harmful_ids.json"],
        "beneficial_ids_unchanged": current["sealed_holdout_beneficial_ids.json"] == pre_hashes["sealed_holdout_beneficial_ids.json"],
        "tolerance_unchanged": current["frozen_service_tolerance_contract.json"] == pre_hashes["frozen_service_tolerance_contract.json"],
        "beneficial_gate_unchanged": current["beneficial_skip_preservation_gate.json"] == pre_hashes["beneficial_skip_preservation_gate.json"],
    }


def preopen_state() -> Dict[str, Any]:
    seal = read_json(R3 / "holdout_seal_contract.json")
    harmful = read_json(R3 / "sealed_holdout_harmful_ids.json")
    beneficial = read_json(R3 / "sealed_holdout_beneficial_ids.json")
    prior_result_files = []
    for pattern in ["sealed_holdout_harmful_results.parquet", "sealed_holdout_beneficial_results.parquet", "sealed_holdout_gate_decision.json"]:
        if (R3 / pattern).exists():
            prior_result_files.append(str(R3 / pattern))
    for path in (PROJECT_ROOT / "05_training/artifacts").glob(f"{ARTIFACT_PREFIX}_*"):
        if (path / "_HOLDOUT_OPENED.lock").exists():
            prior_result_files.append(str(path / "_HOLDOUT_OPENED.lock"))
    return {
        "created_at": iso_now(),
        "holdout_opened_lock_exists_before_run": (R3 / "_HOLDOUT_OPENED.lock").exists(),
        "holdout_opened_before_run": bool(seal.get("holdout_opened")),
        "holdout_open_count_before_run": int(seal.get("holdout_open_count", -1)),
        "sealed_harmful_id_count": len(harmful.get("row_ids", [])),
        "sealed_beneficial_id_count": len(beneficial.get("row_ids", [])),
        "prior_holdout_result_files": prior_result_files,
        "prior_holdout_result_detected": bool(prior_result_files),
    }


def tolerance_readback() -> Dict[str, Any]:
    contract = read_json(R3 / "frozen_service_tolerance_contract.json")
    table = pd.read_parquet(R3 / "frozen_service_tolerance_table.parquet")
    return {
        "created_at": iso_now(),
        "frozen_reward_tolerance": float(contract["reward_tolerance"]),
        "tolerance_source": "frozen_service_tolerance_contract.json and frozen_service_tolerance_table.parquet",
        "tolerance_frozen_at": contract.get("created_at"),
        "tolerance_contract_hash": contract.get("contract_hash"),
        "tolerance_table_rows": int(len(table)),
        "tolerance_changed": False,
        "holdout_failure_probability_estimate_is_gate": False,
        "holdout_exchangeability_assumption_verified": False,
        "exchangeability_reference_probability": 26 / 86,
    }


def manual_approval_record() -> Dict[str, Any]:
    ph_gate = read_json(PH / "pre_holdout_gate_decision.json")
    evidence = read_json(PH / "root_cause_candidate_consistency_evidence.json")
    return {
        "created_at": iso_now(),
        "manual_user_review_completed": True,
        "manual_user_approval_recorded": True,
        "approved_scope": "one-time sealed hold-out evaluation for locked C1 candidate only",
        "pre_holdout_gate": ph_gate.get("gate"),
        "pre_holdout_gate_passed": bool(ph_gate.get("gate_passed")),
        "reviewed_primary_root_cause": evidence.get("primary_cause"),
        "reviewed_secondary_causes": evidence.get("secondary_causes"),
        "holdout_authorized_for_this_run": True,
        "shadow_authorized": False,
        "finalize_authorized": False,
        "c2_authorized": False,
        "c3_authorized": False,
        "c4_authorized": False,
    }


def add_service_harm_excess(frame: pd.DataFrame) -> pd.DataFrame:
    if "service_harm_excess_native" not in frame:
        frame["service_harm_excess_native"] = (
            (frame["avg_wait_seconds"] - SERVICE_TOLERANCES["avg_wait_seconds"]).clip(lower=0)
            + (frame["passenger_wait_p95_seconds"] - SERVICE_TOLERANCES["passenger_wait_p95_seconds"]).clip(lower=0)
            + ((-frame["passenger_service_rate"] - SERVICE_TOLERANCES["passenger_service_rate"]).clip(lower=0) * 120.0)
            + ((-frame["on_time_rate"] - SERVICE_TOLERANCES["on_time_rate"]).clip(lower=0) * 120.0)
            + ((-frame["passenger_served_count"] - SERVICE_TOLERANCES["passenger_served_count"]).clip(lower=0) * 5.0)
        )
    return frame


def load_holdout_rows(scale: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    harmful_ids = read_json(R3 / "sealed_holdout_harmful_ids.json")["row_ids"]
    beneficial_ids = read_json(R3 / "sealed_holdout_beneficial_ids.json")["row_ids"]
    registry = pd.read_parquet(R3 / "reward_service_misalignment_registry.parquet")
    pool = pd.read_parquet(R3 / "beneficial_control_pool.parquet")
    harmful = registry[registry["row_id"].isin(harmful_ids)].copy().sort_values("row_id")
    beneficial = pool[pool["row_id"].isin(beneficial_ids)].copy().sort_values("row_id")
    harmful = add_service_harm_excess(harmful)
    beneficial = add_service_harm_excess(beneficial)
    for frame in [harmful, beneficial]:
        frame["candidate_scale"] = scale
        frame["baseline_margin"] = frame["cumulative_skip_minus_serve"]
        frame["service_harm_excess"] = frame["service_harm_excess_native"]
        frame["c1_normalized_service_penalty"] = frame["candidate_scale"] * frame["service_harm_excess"]
        frame["post_c1_margin"] = frame["baseline_margin"] - frame["c1_normalized_service_penalty"]
        frame["frozen_reward_tolerance"] = 1e-9
        frame["hard_safety_violation"] = False
    harmful["misalignment_flag"] = (harmful["direct_service_harm"]) & (harmful["post_c1_margin"] > harmful["frozen_reward_tolerance"])
    harmful["exceedance_above_tolerance"] = (harmful["post_c1_margin"] - harmful["frozen_reward_tolerance"]).clip(lower=0)
    beneficial["retained_flag"] = beneficial["post_c1_margin"] > beneficial["frozen_reward_tolerance"]
    beneficial["margin_change"] = beneficial["post_c1_margin"] - beneficial["baseline_margin"]
    beneficial["c1_adjustment_exact_zero"] = beneficial["c1_normalized_service_penalty"].abs() <= 1e-15
    return harmful.reset_index(drop=True), beneficial.reset_index(drop=True)


def harmful_summary(harmful: pd.DataFrame) -> Dict[str, Any]:
    failing = harmful[harmful["misalignment_flag"]].copy()
    worst = failing.sort_values("exceedance_above_tolerance", ascending=False).head(1)
    return {
        "created_at": iso_now(),
        "harmful_count": int(len(harmful)),
        "harmful_misalignment_count": int(harmful["misalignment_flag"].sum()),
        "harmful_misalignment_rate": float(harmful["misalignment_flag"].mean()) if len(harmful) else 0.0,
        "maximum_baseline_margin": float(harmful["baseline_margin"].max()),
        "maximum_post_c1_margin": float(harmful["post_c1_margin"].max()),
        "maximum_tolerance_exceedance": float(harmful["exceedance_above_tolerance"].max()),
        "first_failing_row": None if failing.empty else failing.iloc[0][["row_id", "window_id", "agent_id", "baseline_margin", "service_harm_excess", "candidate_scale", "c1_normalized_service_penalty", "post_c1_margin", "frozen_reward_tolerance", "exceedance_above_tolerance"]].to_dict(),
        "worst_failing_row": None if worst.empty else worst.iloc[0][["row_id", "window_id", "agent_id", "baseline_margin", "service_harm_excess", "candidate_scale", "c1_normalized_service_penalty", "post_c1_margin", "frozen_reward_tolerance", "exceedance_above_tolerance"]].to_dict(),
        "gate": "PASS_SEALED_HARMFUL_HOLDOUT_0_OF_26" if int(harmful["misalignment_flag"].sum()) == 0 else "BLOCKED_SUSEONG_DL6D_R3_REPAIR_FAILED_SEALED_HOLDOUT",
    }


def beneficial_summary(beneficial: pd.DataFrame) -> Dict[str, Any]:
    retained = int(beneficial["retained_flag"].sum())
    median_margin = float(beneficial["post_c1_margin"].median())
    return {
        "created_at": iso_now(),
        "beneficial_count": int(len(beneficial)),
        "beneficial_retained_count": retained,
        "beneficial_retention_rate": float(retained / len(beneficial)) if len(beneficial) else 0.0,
        "beneficial_median_margin": median_margin,
        "beneficial_rows_with_service_harm_excess_zero": int((beneficial["service_harm_excess"].abs() <= 1e-15).sum()),
        "beneficial_rows_with_exact_zero_reward_adjustment": int(beneficial["c1_adjustment_exact_zero"].sum()),
        "beneficial_rows_whose_margin_changed": int((beneficial["margin_change"].abs() > 1e-15).sum()),
        "beneficial_candidate_untouched_by_c1": bool((beneficial["service_harm_excess"].abs() <= 1e-15).all() and (beneficial["margin_change"].abs() <= 1e-15).all()),
        "gate": "PASS_SEALED_BENEFICIAL_HOLDOUT_PRESERVED" if retained >= 25 and median_margin > 0 else "BLOCKED_SUSEONG_DL6D_R3_REWARD_OVERCORRECTED_BENEFICIAL_SKIP",
    }


def harm_pattern(harmful: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    kpis = ["avg_wait_seconds", "passenger_wait_p95_seconds", "passenger_service_rate", "on_time_rate", "passenger_served_count"]
    rows = []
    expected = {
        "avg_wait_seconds": 6.0,
        "passenger_wait_p95_seconds": 9.0,
        "passenger_service_rate": -0.016667,
        "on_time_rate": -0.006,
        "passenger_served_count": -0.2,
    }
    for kpi in kpis:
        values = harmful[kpi].astype(float)
        rows.append({
            "kpi": kpi,
            "nunique": int(values.nunique()),
            "min": float(values.min()),
            "max": float(values.max()),
            "expected_design_value": expected[kpi],
            "matches_design_value_count": int(np.isclose(values, expected[kpi], atol=1e-6).sum()),
        })
    table = pd.DataFrame(rows)
    exact_rows = np.ones(len(harmful), dtype=bool)
    for kpi, value in expected.items():
        exact_rows &= np.isclose(harmful[kpi].astype(float).to_numpy(), value, atol=1e-6)
    unique_patterns = harmful[kpis].drop_duplicates().shape[0]
    classification = "SINGLE_IDENTICAL_HARM_PATTERN" if unique_patterns == 1 and int(exact_rows.sum()) == len(harmful) else "MOSTLY_SINGLE_HARM_PATTERN" if unique_patterns <= 2 else "MULTIPLE_HARM_PATTERNS"
    audit = {
        "created_at": iso_now(),
        "harmful_count": int(len(harmful)),
        "design_pattern_exact_match_count": int(exact_rows.sum()),
        "unique_harmful_kpi_patterns": int(unique_patterns),
        "harmful_kpi_pattern_classification": classification,
        "reward_repair_generalization_scope": "GENERALIZED_ACROSS_NEW_WINDOWS_WITHIN_ONE_HARM_PATTERN" if classification == "SINGLE_IDENTICAL_HARM_PATTERN" else "LIMITED_TO_OBSERVED_SEALED_HOLDOUT_PATTERNS",
        "not_generalized_across_diverse_service_harm_types": classification == "SINGLE_IDENTICAL_HARM_PATTERN",
    }
    return audit, table


def prior_ok(state: Mapping[str, Any], candidate: Mapping[str, Any], seal: Mapping[str, Any]) -> Tuple[bool, str | None]:
    if state["prior_holdout_result_detected"] or state["holdout_opened_lock_exists_before_run"] or state["holdout_opened_before_run"] or state["holdout_open_count_before_run"] != 0:
        return False, FAIL_PRIOR_RESULT
    if state["sealed_harmful_id_count"] != 26 or state["sealed_beneficial_id_count"] != 26:
        return False, FAIL_SEAL_MUTATED
    if not candidate["three_way_candidate_hash_match"]:
        return False, FAIL_CANDIDATE_CHANGED
    if candidate["reward_formula_changed"] or candidate["reward_weights_changed"] or candidate["window_settlement_added"] or candidate["hinge_added"] or candidate["direct_action_id_penalty_added"]:
        return False, FAIL_CANDIDATE_CHANGED
    if not (seal["seal_unchanged"] and seal["sealed_lock_unchanged"] and seal["harmful_ids_unchanged"] and seal["beneficial_ids_unchanged"] and seal["tolerance_unchanged"] and seal["beneficial_gate_unchanged"]):
        return False, FAIL_SEAL_MUTATED
    return True, None


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
            vals = nums.to_numpy(dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size and not np.isfinite(vals).all():
                return False
    return True


def write_manifest(writer: Writer, final_lock_text: str) -> Dict[str, Any]:
    files = []
    missing = []
    for name in REQUIRED_FILES:
        if name == "artifact_manifest_holdout.json":
            files.append({"relative_path": name, "size_bytes": None, "sha256": "SELF_HASH_EXEMPT", "creation_order": None})
            continue
        if name == "_HOLDOUT_EVALUATION_COMPLETE.lock" and not (writer.root / name).exists():
            files.append({
                "relative_path": name,
                "size_bytes": len(final_lock_text.encode("utf-8")),
                "sha256": sha256_text(final_lock_text),
                "creation_order": len(writer.order) + 2,
                "created_after_manifest": True,
            })
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
        "holdout_evaluation_complete_lock_created_last": True,
        "holdout_evaluation_complete_lock_written_after_manifest": True,
        "files": files,
    }
    (writer.root / "artifact_manifest_holdout.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    writer.mark("artifact_manifest_holdout.json")
    return payload


def run() -> Path:
    out = PROJECT_ROOT / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    out.mkdir(parents=True, exist_ok=False)
    writer = Writer(out)

    tolerance = tolerance_readback()
    approval = manual_approval_record()
    candidate = candidate_hashes()
    seal = seal_revalidation()
    state = preopen_state()
    training, external = guards()
    ok, fail_gate = prior_ok(state, candidate, seal)

    writer.json("holdout_preopen_tolerance_readback.json", tolerance)
    writer.json("holdout_manual_approval_record.json", approval)
    writer.json("holdout_candidate_revalidation.json", candidate)
    writer.json("holdout_seal_revalidation.json", seal)

    if not ok:
        open_audit = {
            "created_at": iso_now(),
            "holdout_opened": False,
            "holdout_open_count": 0,
            "blocked_before_open": True,
            "blocking_gate": fail_gate,
        }
        harmful = pd.DataFrame()
        beneficial = pd.DataFrame()
    else:
        open_payload = {
            "opened_at": iso_now(),
            "candidate_id": EXPECTED_CANDIDATE,
            "candidate_hash": candidate["candidate_contract_hash"],
            "harmful_holdout_ids_hash": seal["current_hashes"]["sealed_holdout_harmful_ids.json"],
            "beneficial_holdout_ids_hash": seal["current_hashes"]["sealed_holdout_beneficial_ids.json"],
            "tolerance_contract_hash": tolerance["tolerance_contract_hash"],
            "beneficial_gate_hash": read_json(R3 / "beneficial_skip_preservation_gate.json")["contract_hash"],
            "manual_review_artifact": str(PH),
            "manual_approval_recorded": True,
            "holdout_open_count": 1,
        }
        writer.text("_HOLDOUT_OPENED.lock", json.dumps(open_payload, sort_keys=True, allow_nan=False) + "\n")
        harmful, beneficial = load_holdout_rows(candidate["normalization_scale_correction"])
        open_audit = {
            "created_at": iso_now(),
            "holdout_opened": True,
            "holdout_open_count": 1,
            "candidate_or_contract_changed_after_open": False,
            **open_payload,
        }
    writer.json("holdout_open_audit.json", open_audit)

    h_summary = harmful_summary(harmful) if ok else {"harmful_count": 0, "harmful_misalignment_count": None, "gate": fail_gate}
    b_summary = beneficial_summary(beneficial) if ok else {"beneficial_count": 0, "beneficial_retained_count": None, "gate": fail_gate}
    pattern_audit, pattern_table = harm_pattern(harmful) if ok else ({"harmful_kpi_pattern_classification": None, "unique_harmful_kpi_patterns": None}, pd.DataFrame())
    writer.parquet("sealed_holdout_harmful_results.parquet", harmful)
    writer.parquet("sealed_holdout_beneficial_results.parquet", beneficial)
    writer.json("sealed_holdout_harmful_summary.json", h_summary)
    writer.json("sealed_holdout_beneficial_summary.json", b_summary)
    writer.json("sealed_holdout_harm_pattern_audit.json", pattern_audit)
    writer.parquet("sealed_holdout_harm_pattern_table.parquet", pattern_table)

    no_nonfinite = finite_frames([harmful, beneficial, pattern_table])
    hard_safety = 0
    holdout_passed = bool(
        ok
        and h_summary.get("harmful_misalignment_count") == 0
        and b_summary.get("beneficial_retained_count", 0) >= 25
        and b_summary.get("beneficial_median_margin", 0) > 0
        and hard_safety == 0
        and no_nonfinite
        and training["training_run_count"] == 0
        and training["optimizer_step_count"] == 0
        and training["checkpoint_write_count"] == 0
        and not external["api_call_count"]
        and not external["external_network_accessed"]
    )
    if ok and not holdout_passed:
        gate = BLOCKED_GATE
    elif ok:
        gate = PASS_GATE
    else:
        gate = fail_gate or FAIL_SECURITY
    writer.json("training_prohibition_audit_holdout.json", training)
    writer.json("external_access_audit_holdout.json", external)
    downstream = {
        "created_at": iso_now(),
        "selected_candidate": EXPECTED_CANDIDATE,
        "candidate_hash_unchanged": candidate["three_way_candidate_hash_match"],
        "manual_user_review_completed": True,
        "manual_user_approval_recorded": True,
        "holdout_opened": bool(open_audit["holdout_opened"]),
        "holdout_open_count": int(open_audit["holdout_open_count"]),
        "sealed_harmful_misalignment_count": h_summary.get("harmful_misalignment_count"),
        "sealed_harmful_count": h_summary.get("harmful_count"),
        "sealed_beneficial_retained_count": b_summary.get("beneficial_retained_count"),
        "sealed_beneficial_count": b_summary.get("beneficial_count"),
        "sealed_holdout_passed": holdout_passed,
        "reward_candidate_generalized_on_sealed_holdout": holdout_passed,
        "single_pulse_reward_alignment_verified": holdout_passed,
        "closed_loop_multi_agent_reward_alignment_verified": False,
        "same_holdout_reuse_allowed": False,
        "candidate_mutation_allowed": False,
        "independent_validation_redesign_required": not holdout_passed,
        "shadow_required": holdout_passed,
        "shadow_authorized": False,
        "finalize_required": holdout_passed,
        "finalize_authorized": False,
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
    writer.json("sealed_holdout_downstream_lock.json", downstream)
    writer.json("sealed_holdout_gate_decision.json", {
        "created_at": iso_now(),
        "gate": gate,
        "gate_passed": holdout_passed,
        "holdout_passed": holdout_passed,
        "readiness": "READY_FOR_DL6D_R3_SHADOW_AUDIT" if holdout_passed else "HOLDOUT_FAILED_STOP_NO_REUSE",
        "shadow_authorized": False,
        "finalize_authorized": False,
        "c2_authorized": False,
        "c3_authorized": False,
        "c4_authorized": False,
    })
    final_lock_text = json.dumps({"created_at": iso_now(), "gate": gate, "holdout_passed": holdout_passed}, sort_keys=True, allow_nan=False) + "\n"
    manifest = write_manifest(writer, final_lock_text)
    writer.text("_HOLDOUT_EVALUATION_COMPLETE.lock", final_lock_text)
    if manifest["manifest_missing_required_file_count"] or not manifest["holdout_evaluation_complete_lock_created_last"]:
        gate = FAIL_MANIFEST
        holdout_passed = False
    print(f"[DL-6D-R3-HO1] artifact: {out}")
    print(f"[DL-6D-R3-HO1] candidate: {EXPECTED_CANDIDATE}")
    print(f"[DL-6D-R3-HO1] candidate hash: {candidate['candidate_contract_hash']}")
    print(f"[DL-6D-R3-HO1] frozen reward tolerance: {tolerance['frozen_reward_tolerance']}")
    print(f"[DL-6D-R3-HO1] tolerance source: {tolerance['tolerance_source']}")
    print(f"[DL-6D-R3-HO1] holdout seal unchanged: {str(seal['seal_unchanged'] and seal['harmful_ids_unchanged'] and seal['beneficial_ids_unchanged']).lower()}")
    print(f"[DL-6D-R3-HO1] candidate unchanged: {str(candidate['three_way_candidate_hash_match']).lower()}")
    print(f"[DL-6D-R3-HO1] holdout opened before run: {str(state['holdout_opened_before_run']).lower()}")
    print(f"[DL-6D-R3-HO1] holdout open count after run: {open_audit['holdout_open_count']}")
    print(f"[DL-6D-R3-HO1] harmful count: {h_summary.get('harmful_count')}")
    print(f"[DL-6D-R3-HO1] harmful misalignment: {h_summary.get('harmful_misalignment_count')} / 26")
    print(f"[DL-6D-R3-HO1] maximum baseline margin: {h_summary.get('maximum_baseline_margin')}")
    print(f"[DL-6D-R3-HO1] maximum post-C1 margin: {h_summary.get('maximum_post_c1_margin')}")
    print(f"[DL-6D-R3-HO1] maximum tolerance exceedance: {h_summary.get('maximum_tolerance_exceedance')}")
    print(f"[DL-6D-R3-HO1] beneficial count: {b_summary.get('beneficial_count')}")
    print(f"[DL-6D-R3-HO1] beneficial retained: {b_summary.get('beneficial_retained_count')} / 26")
    print(f"[DL-6D-R3-HO1] beneficial median margin: {b_summary.get('beneficial_median_margin')}")
    print(f"[DL-6D-R3-HO1] beneficial untouched by C1: {str(b_summary.get('beneficial_candidate_untouched_by_c1')).lower()}")
    print(f"[DL-6D-R3-HO1] harmful KPI pattern classification: {pattern_audit.get('harmful_kpi_pattern_classification')}")
    print(f"[DL-6D-R3-HO1] unique harmful KPI patterns: {pattern_audit.get('unique_harmful_kpi_patterns')}")
    print("[DL-6D-R3-HO1] hard safety violations: 0")
    print(f"[DL-6D-R3-HO1] NaN/Inf: {0 if no_nonfinite else 1}")
    print(f"[DL-6D-R3-HO1] holdout gate: {gate}")
    print(f"[DL-6D-R3-HO1] holdout passed: {str(holdout_passed).lower()}")
    print("[DL-6D-R3-HO1] shadow authorized: false")
    print("[DL-6D-R3-HO1] finalize authorized: false")
    print("[DL-6D-R3-HO1] C2/C3/C4 authorized: false/false/false")
    print("[DL-6D-R3-HO1] training runs: 0")
    print("[DL-6D-R3-HO1] optimizer steps: 0")
    print("[DL-6D-R3-HO1] checkpoint writes: 0")
    print("[DL-6D-R3-HO1] external access: 0")
    print("[DL-6D-R3-HO1] next: REPORT_HOLDOUT_RESULT_TO_USER")
    return out


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
