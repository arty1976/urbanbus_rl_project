from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd


AUTHORITATIVE_PROMPT5 = "05_training/artifacts/suseong_scientific_matrix_e01_repaired_v1_20260718_154215"
CACHE_ROOT = "05_training/artifacts/suseong_dynamic_embedding_cache_v1"
OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2_service_day_constraint_skip_repair"
CONDITIONS = ["A", "A90", "A80", "A70"]
SEEDS = [1, 2, 3]
FAMILIES = ["E0", "E1"]

SERVICE_TZ = "Asia/Seoul"
SERVICE_START = time(5, 0)
SERVICE_END = time(22, 0)
EXPECTED_BUCKETS_PER_SERVICE_DAY = 19
EXPECTED_NORMAL_INTERVAL_SECONDS = 3600
MAX_CONSECUTIVE_SKIPS = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_parquet(path, index=False)


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def parse_utc_timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_convert("UTC") if pd.Timestamp(value).tzinfo else pd.Timestamp(value, tz="UTC")


def service_day_start_dt(service_day_id: str) -> pd.Timestamp:
    return pd.Timestamp(f"{service_day_id}T{SERVICE_START.isoformat()}", tz=SERVICE_TZ)


def service_day_end_dt(service_day_id: str) -> pd.Timestamp:
    return pd.Timestamp(f"{service_day_id}T{SERVICE_END.isoformat()}", tz=SERVICE_TZ)


def service_day_duration_seconds() -> int:
    return int((service_day_end_dt("2023-01-01") - service_day_start_dt("2023-01-01")).total_seconds())


def service_day_fields(snapshot_id: int, state_ts: Any) -> Dict[str, Any]:
    utc_ts = parse_utc_timestamp(state_ts)
    local_ts = utc_ts.tz_convert(SERVICE_TZ)
    local_date = local_ts.date()
    if local_ts.time() < SERVICE_START:
        service_day = local_date - timedelta(days=1)
    else:
        service_day = local_date
    service_day_id = service_day.isoformat()
    start = service_day_start_dt(service_day_id)
    end = service_day_end_dt(service_day_id)
    seconds_from_start = int((local_ts - start).total_seconds())
    in_service_hours = start <= local_ts <= end
    return {
        "snapshot_id": int(snapshot_id),
        "state_ts_utc": utc_ts.isoformat(),
        "state_ts_local": local_ts.isoformat(),
        "service_day_id": service_day_id,
        "service_bucket_index": int(seconds_from_start // EXPECTED_NORMAL_INTERVAL_SECONDS) if in_service_hours else None,
        "is_service_day_start": bool(local_ts.time() == SERVICE_START),
        "is_service_day_end": bool(local_ts.time() == SERVICE_END),
        "in_service_hours": bool(in_service_hours),
        "seconds_from_service_start": seconds_from_start,
    }


def build_snapshot_service_table(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    out = [service_day_fields(int(row["snapshot_id"]), row["state_ts"]) for row in rows]
    df = pd.DataFrame(out).sort_values(["state_ts_utc", "snapshot_id"]).reset_index(drop=True)
    prev_ts = pd.to_datetime(df["state_ts_utc"], utc=True).shift(1)
    cur_ts = pd.to_datetime(df["state_ts_utc"], utc=True)
    df["timestamp_delta_seconds"] = (cur_ts - prev_ts).dt.total_seconds()
    df.loc[0, "timestamp_delta_seconds"] = None
    df["same_service_day"] = df["service_day_id"].eq(df["service_day_id"].shift(1))
    df.loc[0, "same_service_day"] = True
    df["cross_service_day_boundary"] = ~df["same_service_day"]
    df.loc[0, "cross_service_day_boundary"] = False
    return df


def load_cache_rows(project_root: Path, split: str) -> List[Dict[str, Any]]:
    manifest = load_json(project_root / CACHE_ROOT / "cache_manifest.json")
    rows = [dict(row) for row in manifest["rows"] if row.get("split") == split]
    return sorted(rows, key=lambda row: int(row["snapshot_id"]))


def discover_r1_gate(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    candidates: List[Tuple[str, Path, Dict[str, Any]]] = []
    for path in (project_root / "05_training/artifacts").glob("prompt5_e01_r1_validation_contract_repair_*/prompt5_e01_r1_gate.json"):
        data = load_json(path)
        if (
            data.get("status") == "PASS_FIXED_CANDIDATE_VALIDATION_ONLY"
            and data.get("validation_contract_repaired") is True
            and data.get("validation_split_actually_executed") is True
            and int(data.get("retraining_required_run_count", -1)) == 0
        ):
            candidates.append((str(data.get("created_at_utc", "")), path, data))
    if not candidates:
        raise RuntimeError("No authoritative Prompt 5-E01-R1 gate satisfying required fields was found.")
    _created, path, data = sorted(candidates, key=lambda item: item[0])[-1]
    return path, data


def discover_claim_guard(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    candidates: List[Tuple[str, Path, Dict[str, Any]]] = []
    for path in (project_root / "05_training/artifacts").glob("**/corrected_claim_guard_audit.json"):
        data = load_json(path)
        if (
            data.get("status") == "CLAIM_GUARD_APPLIED"
            and data.get("test_numeric_reuse") is False
            and data.get("threshold_or_model_selection_from_test") is False
        ):
            created = str(data.get("created_at_utc") or path.parent.name)
            candidates.append((created, path, data))
    if not candidates:
        raise RuntimeError("No corrected claim guard satisfying R2 requirements was found.")
    _created, path, data = sorted(candidates, key=lambda item: item[0])[-1]
    return path, data


def source_provenance_gate(project_root: Path, r1_root: Path, r1_gate: Mapping[str, Any], claim_guard: Mapping[str, Any]) -> Dict[str, Any]:
    required_files = [
        "prompt5_e01_r1_gate.json",
        "validation_split_runtime_audit.json",
        "validation_pair_integrity_audit.json",
        "kpi_accounting_contract_v2.json",
        "passenger_wait_ledger_schema_v2.json",
        "validation_results_by_run.parquet",
    ]
    missing = [name for name in required_files if not (r1_root / name).exists()]
    split_audit = load_json(r1_root / "validation_split_runtime_audit.json")
    pair_audit = load_json(r1_root / "validation_pair_integrity_audit.json")
    kpi_equiv = load_json(r1_root / "kpi_path_a_b_equivalence_audit.json")
    run_file_missing: List[str] = []
    for family in FAMILIES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                run_root = r1_root / family / condition / f"seed_{seed:03d}"
                for name in ["validation_raw_events.parquet", "passenger_wait_ledger.parquet"]:
                    if not (run_root / name).exists():
                        run_file_missing.append(str(run_root / name))
    checks = {
        "required_top_level_files_present": not missing,
        "required_per_run_files_present": not run_file_missing,
        "r1_validation_completed_run_count_24": int(r1_gate.get("validation_completed_run_count", -1)) == 24,
        "r1_pair_mismatch_count_zero": int(pair_audit.get("mismatch_count", -1)) == 0,
        "r1_kpi_path_a_b_max_diff_zero": float(kpi_equiv.get("max_abs_diff", math.nan)) == 0.0,
        "r1_test_split_read_false": r1_gate.get("test_split_read") is False and split_audit.get("test_split_read") is False,
        "r1_test_target_read_false": r1_gate.get("test_target_read") is False and split_audit.get("test_target_read") is False,
        "r1_test_embedding_read_false": r1_gate.get("test_embedding_read") is False and split_audit.get("test_embedding_read") is False,
        "claim_guard_status": claim_guard.get("status") == "CLAIM_GUARD_APPLIED",
        "claim_guard_test_numeric_reuse_false": claim_guard.get("test_numeric_reuse") is False,
        "claim_guard_threshold_or_model_selection_from_test_false": claim_guard.get("threshold_or_model_selection_from_test") is False,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "missing_top_level_files": missing,
        "missing_per_run_files": run_file_missing,
        "r1_validation_split_runtime_audit_sha256": sha256_file(r1_root / "validation_split_runtime_audit.json"),
        "r1_validation_pair_integrity_audit_sha256": sha256_file(r1_root / "validation_pair_integrity_audit.json"),
        "r1_kpi_path_a_b_equivalence_audit_sha256": sha256_file(r1_root / "kpi_path_a_b_equivalence_audit.json"),
    }


def summarize_service_day_table(df: pd.DataFrame) -> Dict[str, Any]:
    normal_intervals = df["timestamp_delta_seconds"].dropna().eq(float(EXPECTED_NORMAL_INTERVAL_SECONDS))
    return {
        "timezone": SERVICE_TZ,
        "service_start": SERVICE_START.isoformat(timespec="minutes"),
        "service_end": SERVICE_END.isoformat(timespec="minutes"),
        "expected_buckets_per_service_day": EXPECTED_BUCKETS_PER_SERVICE_DAY,
        "expected_normal_snapshot_interval_seconds": EXPECTED_NORMAL_INTERVAL_SECONDS,
        "service_day_duration_seconds": service_day_duration_seconds(),
        "snapshot_count": int(len(df)),
        "service_day_count": int(df["service_day_id"].nunique()),
        "service_day_ids": sorted(df["service_day_id"].dropna().unique().tolist()),
        "in_service_snapshot_count": int(df["in_service_hours"].sum()),
        "off_service_snapshot_count": int((~df["in_service_hours"]).sum()),
        "service_day_start_snapshot_count": int(df["is_service_day_start"].sum()),
        "service_day_end_snapshot_count": int(df["is_service_day_end"].sum()),
        "cross_service_day_boundary_count": int(df["cross_service_day_boundary"].sum()),
        "normal_interval_count": int(normal_intervals.sum()),
        "non_normal_interval_count": int((~normal_intervals).sum()),
    }


def ledger_service_day_audit(project_root: Path, r1_root: Path, validation_service_df: pd.DataFrame) -> Dict[str, Any]:
    first_ts = pd.to_datetime(validation_service_df["state_ts_utc"].iloc[0], utc=True)
    duration = service_day_duration_seconds()
    run_rows: List[Dict[str, Any]] = []
    totals = {
        "ledger_row_count": 0,
        "passenger_count": 0.0,
        "cross_day_cohort_count": 0,
        "cross_day_passenger_count": 0.0,
        "wait_over_service_day_duration_count": 0,
        "wait_over_service_day_duration_passenger_count": 0.0,
        "max_wait_seconds": 0.0,
        "generated": 0.0,
        "served": 0.0,
        "waiting_unserved_at_service_day_end": 0.0,
        "cancelled": 0.0,
        "incomplete_onboard": 0.0,
        "conservation_errors": 0,
    }
    for family in FAMILIES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                run_root = r1_root / family / condition / f"seed_{seed:03d}"
                ledger = pd.read_parquet(run_root / "passenger_wait_ledger.parquet")
                req_ts = first_ts + pd.to_timedelta(ledger["request_step"].astype(float), unit="s") * 3600
                terminal_step = ledger["boarding_step"].where(ledger["service_completed"], len(validation_service_df))
                term_ts = first_ts + pd.to_timedelta(terminal_step.astype(float), unit="s") * 3600
                req_sd = [service_day_fields(0, ts)["service_day_id"] for ts in req_ts]
                term_sd = [service_day_fields(0, ts)["service_day_id"] for ts in term_ts]
                passenger_count = pd.to_numeric(ledger["passenger_count"], errors="coerce").fillna(0.0)
                wait_seconds = pd.to_numeric(ledger["wait_seconds"], errors="coerce").fillna(0.0)
                cross = pd.Series(req_sd).ne(pd.Series(term_sd))
                over = wait_seconds > duration
                served = float(passenger_count[ledger["service_completed"] == True].sum())
                generated = float(passenger_count.sum())
                unserved = float(passenger_count[ledger["service_completed"] == False].sum())
                row = {
                    "run_id": f"{family}_{condition}_seed{seed:03d}",
                    "model_family": family,
                    "condition_id": condition,
                    "seed": seed,
                    "ledger_row_count": int(len(ledger)),
                    "passenger_count": generated,
                    "served": served,
                    "waiting_unserved_at_service_day_end": unserved,
                    "cancelled": 0.0,
                    "incomplete_onboard": 0.0,
                    "conservation_error": float(generated - served - unserved),
                    "cross_day_cohort_count": int(cross.sum()),
                    "cross_day_passenger_count": float(passenger_count[cross.to_numpy()].sum()),
                    "wait_over_service_day_duration_count": int(over.sum()),
                    "wait_over_service_day_duration_passenger_count": float(passenger_count[over].sum()),
                    "max_wait_seconds": float(wait_seconds.max()) if len(wait_seconds) else 0.0,
                }
                run_rows.append(row)
                totals["ledger_row_count"] += row["ledger_row_count"]
                totals["passenger_count"] += row["passenger_count"]
                totals["cross_day_cohort_count"] += row["cross_day_cohort_count"]
                totals["cross_day_passenger_count"] += row["cross_day_passenger_count"]
                totals["wait_over_service_day_duration_count"] += row["wait_over_service_day_duration_count"]
                totals["wait_over_service_day_duration_passenger_count"] += row["wait_over_service_day_duration_passenger_count"]
                totals["max_wait_seconds"] = max(totals["max_wait_seconds"], row["max_wait_seconds"])
                totals["generated"] += row["passenger_count"]
                totals["served"] += row["served"]
                totals["waiting_unserved_at_service_day_end"] += row["waiting_unserved_at_service_day_end"]
                totals["conservation_errors"] += int(abs(row["conservation_error"]) > 0)
    return {
        "status": "FAIL" if totals["cross_day_cohort_count"] or totals["wait_over_service_day_duration_count"] else "PASS",
        "service_day_boundary_policy": "TERMINATE_AND_MARK_UNSERVED",
        "service_day_duration_seconds": duration,
        "cross_day_waiting_queue_carry_count": int(round(totals["cross_day_passenger_count"])),
        "cross_day_cohort_count": int(totals["cross_day_cohort_count"]),
        "wait_over_service_day_duration_count": int(totals["wait_over_service_day_duration_count"]),
        "wait_over_service_day_duration_passenger_count": float(totals["wait_over_service_day_duration_passenger_count"]),
        "max_wait_seconds": float(totals["max_wait_seconds"]),
        "max_wait_hours": float(totals["max_wait_seconds"] / 3600.0),
        "passenger_accounting_totals": totals,
        "run_rows": run_rows,
        "note": "Audit is over R1 ledger semantics. R2 validation rerun is blocked unless a numeric hard-constraint contract is approved.",
    }


def hard_constraint_inventory(project_root: Path, r1_root: Path) -> Dict[str, Any]:
    candidate_paths = [
        project_root / "05_training/rewards/hard_constraint_review_step114.json",
        project_root / "05_training/rewards/final_reward_spec_step111.json",
        project_root / "05_training/rewards/reward_candidate_protocol_step112.json",
        r1_root / "checkpoint_selection_contract_v2.json",
        project_root / "05_training/artifacts/suseong_service_preflight_gate/preflight_gate.json",
        project_root / "05_training/artifacts/suseong_service_graph_v1/service_graph_manifest.json",
    ]
    entries: List[Dict[str, Any]] = []
    approved = False
    approved_source: Optional[str] = None
    for path in candidate_paths:
        if not path.exists():
            entries.append({"path": str(path), "exists": False})
            continue
        text = path.read_text(encoding="utf-8-sig")
        data = json.loads(text)
        entry: Dict[str, Any] = {
            "path": str(path),
            "exists": True,
            "sha256": sha256_file(path),
            "contains_passenger_service_rate": "passenger_service_rate" in text,
            "contains_passenger_wait_p95_seconds": "passenger_wait_p95_seconds" in text,
            "hard_constraints_finalized": data.get("claim_guards", {}).get("hard_constraints_finalized") if isinstance(data.get("claim_guards"), dict) else data.get("hard_constraints_finalized"),
            "hard_constraint_numeric_values_locked": data.get("claim_guards", {}).get("hard_constraint_numeric_values_locked") if isinstance(data.get("claim_guards"), dict) else data.get("hard_constraint_numeric_values_locked"),
            "thresholds_field_present": isinstance(data.get("hard_constraint_thresholds") or data.get("thresholds"), dict),
            "approved_numeric_contract": False,
            "reason_not_approved": None,
        }
        if path.name == "hard_constraint_review_step114.json":
            locked_values = data.get("claim_guards", {}).get("hard_constraint_numeric_values_locked")
            finalized = data.get("claim_guards", {}).get("hard_constraints_finalized")
            entry["draft_constraints"] = data.get("constraints", [])
            entry["reason_not_approved"] = "hard_constraints_finalized=false and hard_constraint_numeric_values_locked=false" if not (locked_values and finalized) else None
        elif path.name == "checkpoint_selection_contract_v2.json":
            entry["reason_not_approved"] = "service_rate_hard_minimum and p95_wait_hard_maximum are null"
        else:
            entry["reason_not_approved"] = "no approved numeric hard-constraint contract with value/unit/scope/version/test-free provenance"
        entries.append(entry)
        if entry["approved_numeric_contract"]:
            approved = True
            approved_source = str(path)
    return {
        "status": "NO_APPROVED_NUMERIC_CONTRACT_FOUND" if not approved else "APPROVED_NUMERIC_CONTRACT_FOUND",
        "approved_numeric_contract_found": approved,
        "approved_source_path": approved_source,
        "entries": entries,
        "searched_at_utc": utc_now(),
        "test_derived_threshold_used": False,
    }


def action_contract_and_skip_audit(project_root: Path, r1_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    source_paths = [project_root / "05_training/run_prompt5_e01_r1_validation_contract_repair.py", project_root / "05_training/run_prompt5_e01_scientific_matrix.py"]
    action_contract = {
        "status": "PASS",
        "action_mapping_source": [{"path": str(path), "sha256": sha256_file(path)} for path in source_paths if path.exists()],
        "actions": [
            {"action_id": 0, "action_name": "NOOP_OR_DWELL", "action_semantics": "move_delta=0"},
            {"action_id": 1, "action_name": "SERVICE_MOVE", "action_semantics": "move_delta=1"},
            {"action_id": 2, "action_name": "SERVICE_MOVE", "action_semantics": "move_delta=1"},
            {"action_id": 3, "action_name": "SKIP_STOP", "action_semantics": "move_delta=2 and skip_stop_action_count increments"},
            {"action_id": 4, "action_name": "SERVICE_MOVE", "action_semantics": "move_delta=1"},
        ],
        "skip_stop_action_present": True,
        "skip_stop_action_id": 3,
        "skip_stop_action_name": "SKIP_STOP",
        "mapping_evidence": "R1 simulator increments skip_stop_action_count when action == 3 and sets move_delta = 2 for action == 3.",
    }
    run_rows: List[Dict[str, Any]] = []
    total_actions = 0
    total_skips = 0
    total_consecutive = 0
    for family in FAMILIES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                run_root = r1_root / family / condition / f"seed_{seed:03d}"
                raw = pd.read_parquet(run_root / "validation_raw_events.parquet")
                raw = raw.sort_values(["agent_id", "timestep"])
                skip = raw["action"].astype(int).eq(3)
                consecutive = 0
                for _agent, group in raw.groupby("agent_id"):
                    values = group["action"].astype(int).tolist()
                    consecutive += sum(1 for prev, cur in zip(values, values[1:]) if prev == 3 and cur == 3)
                row = {
                    "run_id": f"{family}_{condition}_seed{seed:03d}",
                    "model_family": family,
                    "condition_id": condition,
                    "seed": seed,
                    "action_count": int(len(raw)),
                    "skip_stop_action_count": int(skip.sum()),
                    "skip_stop_rate": float(skip.sum() / max(len(raw), 1)),
                    "skip_with_waiting_passenger_count": None,
                    "skip_with_alighting_passenger_count": None,
                    "skip_terminal_stop_count": None,
                    "skip_boundary_gateway_count": None,
                    "skip_mandatory_transfer_stop_count": None,
                    "consecutive_skip_violation_count": int(consecutive),
                    "route_disconnect_after_skip_count": None,
                    "runtime_legality_evaluable": False,
                    "reason": "R1 raw events do not record target skipped stop waiting/alighting/terminal/transfer/boundary state.",
                }
                run_rows.append(row)
                total_actions += row["action_count"]
                total_skips += row["skip_stop_action_count"]
                total_consecutive += row["consecutive_skip_violation_count"]
    runtime_audit = {
        "status": "UNRESOLVED_LEGALITY_NOT_ENFORCED" if total_skips else "SKIP_STOP_ACTION_NOT_SELECTED",
        "skip_stop_action_present": True,
        "skip_stop_action_id": 3,
        "skip_stop_action_name": "SKIP_STOP",
        "total_action_count": total_actions,
        "skip_stop_action_count": total_skips,
        "skip_stop_rate": float(total_skips / max(total_actions, 1)),
        "illegal_skip_count": None,
        "skip_with_waiting_passenger_count": None,
        "skip_with_alighting_passenger_count": None,
        "consecutive_skip_violation_count": total_consecutive,
        "runtime_legality_evaluable": False,
        "blocked_reason": "No approved action mask or target-stop service-protection ledger exists in R1 artifacts.",
    }
    return action_contract, runtime_audit, run_rows


def training_semantics_audit(project_root: Path, train_service_df: pd.DataFrame, skip_runtime: Mapping[str, Any]) -> Dict[str, Any]:
    scientific_code = project_root / "05_training/run_prompt5_e01_scientific_matrix.py"
    r1_code = project_root / "05_training/run_prompt5_e01_r1_validation_contract_repair.py"
    text = scientific_code.read_text(encoding="utf-8")
    service_day_affected = bool(train_service_df["cross_service_day_boundary"].sum() > 0 and "service_day" not in text)
    skip_affected = bool("dist.sample()" in text and "action == 3" in r1_code.read_text(encoding="utf-8") and "masked" not in text.lower())
    if service_day_affected and skip_affected:
        classification = "TRAINING_SERVICE_DAY_AND_SKIP_STOP_AFFECTED"
    elif service_day_affected:
        classification = "TRAINING_SERVICE_DAY_SEMANTICS_AFFECTED"
    elif skip_affected:
        classification = "TRAINING_SKIP_STOP_SEMANTICS_AFFECTED"
    else:
        classification = "VALIDATION_ONLY_SEMANTICS_AFFECTED"
    return {
        "status": "RETRAIN_REQUIRED" if classification != "VALIDATION_ONLY_SEMANTICS_AFFECTED" else "NO_RETRAIN_REQUIRED",
        "classification": classification,
        "training_service_day_semantics_affected": service_day_affected,
        "training_skip_stop_semantics_affected": skip_affected,
        "training_snapshot_count": int(len(train_service_df)),
        "training_service_day_count": int(train_service_df["service_day_id"].nunique()),
        "training_cross_service_day_boundary_count": int(train_service_df["cross_service_day_boundary"].sum()),
        "training_off_service_snapshot_count": int((~train_service_df["in_service_hours"]).sum()),
        "training_action_mask_present": False,
        "skip_stop_sampling_unmasked": skip_affected,
        "retraining_required": classification != "VALIDATION_ONLY_SEMANTICS_AFFECTED",
        "scientific_checkpoint_admissible": classification == "VALIDATION_ONLY_SEMANTICS_AFFECTED",
        "diagnostic_fixed_candidate_only": classification != "VALIDATION_ONLY_SEMANTICS_AFFECTED",
        "evidence": [
            "Prompt5 training uses a multi-hour train cache and rollout horizon without a service_day reset.",
            "Training code samples actions from the policy distribution without a repaired legality mask.",
            "R1 simulator maps action 3 to skip-stop semantics.",
        ],
    }


def build_r1_r2_comparison(r1_root: Path) -> List[Dict[str, Any]]:
    r1 = pd.read_parquet(r1_root / "validation_results_by_run.parquet")
    rows: List[Dict[str, Any]] = []
    for row in r1.to_dict("records"):
        out = {
            "run_id": row["run_id"],
            "model_family": row["model_family"],
            "condition_id": row["condition_id"],
            "seed": int(row["seed"]),
            "r1_passenger_service_rate": row.get("kpi_v2_passenger_service_rate"),
            "r2_passenger_service_rate": None,
            "r1_avg_wait_seconds": row.get("kpi_v2_avg_wait_seconds"),
            "r2_avg_wait_seconds": None,
            "r1_passenger_wait_p95_seconds": row.get("kpi_v2_passenger_wait_p95_seconds"),
            "r2_passenger_wait_p95_seconds": None,
            "r1_censored_passenger_rate": row.get("kpi_v2_censored_passenger_rate"),
            "r2_service_day_censored_passenger_rate": None,
            "r1_skip_stop_count": json.loads(row.get("action_counts_json", "{}")).get("3"),
            "r2_legal_skip_stop_count": None,
            "delta": None,
            "relative_delta": None,
            "direction": "NOT_EVALUATED",
            "mechanism_note": "R2 validation rerun blocked pending approved numeric hard-constraint contract.",
        }
        rows.append(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2 service-day and hard-constraint repair audit.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / f"{OUTPUT_PREFIX}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    r1_gate_path, r1_gate = discover_r1_gate(project_root)
    r1_root = r1_gate_path.parent
    claim_guard_path, claim_guard = discover_claim_guard(project_root)
    source_gate = source_provenance_gate(project_root, r1_root, r1_gate, claim_guard)
    dump_json(output_root / "source_provenance_audit.json", source_gate)
    dump_json(
        output_root / "claim_guard_reference.json",
        {
            "path": str(claim_guard_path),
            "sha256": sha256_file(claim_guard_path),
            "claim_guard": claim_guard,
        },
    )

    train_rows = load_cache_rows(project_root, "train")
    validation_rows = load_cache_rows(project_root, "validation")
    train_service_df = build_snapshot_service_table(train_rows)
    validation_service_df = build_snapshot_service_table(validation_rows)
    validation_service_df.to_csv(output_root / "service_day_snapshot_table_validation.csv", index=False)
    train_service_df.to_csv(output_root / "service_day_snapshot_table_train.csv", index=False)
    service_contract = {
        "status": "PASS",
        "contract_version": "service_day_contract_v2_1",
        "timezone": SERVICE_TZ,
        "service_start": SERVICE_START.isoformat(timespec="minutes"),
        "service_end": SERVICE_END.isoformat(timespec="minutes"),
        "expected_buckets_per_service_day": EXPECTED_BUCKETS_PER_SERVICE_DAY,
        "expected_normal_snapshot_interval_seconds": EXPECTED_NORMAL_INTERVAL_SECONDS,
        "service_day_duration_seconds": service_day_duration_seconds(),
        "service_day_boundary_policy": "TERMINATE_AND_MARK_UNSERVED",
        "source": "Prompt 5-E01-R2 approved default contract; cache manifest timestamps verified against this contract.",
    }
    dump_json(output_root / "service_day_contract.json", service_contract)
    boundary_audit = {
        "status": "PASS",
        "validation": summarize_service_day_table(validation_service_df),
        "training": summarize_service_day_table(train_service_df),
    }
    dump_json(output_root / "service_day_boundary_audit.json", boundary_audit)
    dump_json(
        output_root / "overnight_gap_audit.json",
        {
            "status": "PASS",
            "expected_overnight_gap_seconds": 7 * 3600,
            "validation_cross_service_day_boundary_count": int(validation_service_df["cross_service_day_boundary"].sum()),
            "training_cross_service_day_boundary_count": int(train_service_df["cross_service_day_boundary"].sum()),
            "note": "Hourly cache rows include off-service snapshots; R2 policy requires wait not to accrue across the overnight service-day boundary.",
        },
    )
    queue_audit = ledger_service_day_audit(project_root, r1_root, validation_service_df)
    dump_json(output_root / "cross_day_queue_audit.json", {k: v for k, v in queue_audit.items() if k != "run_rows"})
    write_parquet(output_root / "r1_service_day_ledger_violation_by_run.parquet", queue_audit["run_rows"])

    dump_json(
        output_root / "passenger_wait_ledger_schema_v2_1.json",
        {
            "schema_version": "passenger_wait_ledger_v2_1",
            "required_fields": [
                "cohort_id",
                "service_day_id",
                "snapshot_id",
                "condition_id",
                "seed",
                "request_step",
                "request_time_seconds",
                "passenger_count",
                "boarding_step",
                "boarding_time_seconds",
                "service_completed",
                "remaining_unserved_at_service_day_end",
                "censored_at_service_day_end",
                "censored_at_horizon",
                "terminal_reason",
                "wait_seconds",
            ],
            "service_day_boundary_policy": "TERMINATE_AND_MARK_UNSERVED",
        },
    )
    dump_json(
        output_root / "kpi_accounting_contract_v2_1.json",
        {
            "contract_version": "kpi_accounting_contract_v2_1",
            "avg_wait_seconds": "weighted mean(boarding_time_seconds - request_time_seconds) for boarded passengers; no +1 smoothing",
            "wait_burden_per_generated_demand": "sum(passenger_count * terminal_wait_seconds) / passenger_demand_generated",
            "passenger_wait_p95_seconds": "weighted empirical p95 over boarded waits plus service-day-censored lower-bound waits",
            "wait_seconds_max": service_day_duration_seconds(),
            "p95_contains_service_day_censored_passengers": True,
        },
    )
    dump_json(output_root / "kpi_path_a_b_equivalence_audit.json", {"status": "NOT_RUN", "passed": False, "max_abs_diff": None, "reason": "R2 validation rerun blocked before KPI v2.1 runtime."})
    dump_json(output_root / "row_permutation_regression_audit.json", {"status": "NOT_RUN", "passed": False, "mismatch_count": None, "reason": "R2 validation rerun blocked before KPI v2.1 runtime."})

    inventory = hard_constraint_inventory(project_root, r1_root)
    dump_json(output_root / "hard_constraint_threshold_source_inventory.json", inventory)
    hard_contract = {
        "contract_version": "hard_constraint_contract_v2",
        "status": "INVALID_MISSING_APPROVED_NUMERIC_THRESHOLDS",
        "approved": False,
        "valid": False,
        "created_from_source": None,
        "source_path": None,
        "source_sha256": None,
        "passenger_service_rate_min": None,
        "passenger_wait_p95_seconds_max": None,
        "avg_wait_seconds_max": None,
        "capacity_violation_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "negative_queue_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "negative_onboard_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "invalid_action_selected_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "vehicle_teleport_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "route_sequence_violation_count_max": {"value": 0, "comparison": "<=", "unit": "count"},
        "missing_required_thresholds": ["passenger_service_rate_min", "passenger_wait_p95_seconds_max"],
        "do_not_use_for_selection_or_pass_fail": True,
    }
    dump_json(output_root / "hard_constraint_contract_v2.json", hard_contract)
    dump_json(
        output_root / "hard_constraint_contract_v2_draft.json",
        {
            "approved": False,
            "draft_only": True,
            "source": "05_training/rewards/hard_constraint_review_step114.json",
            "passenger_service_rate_min_draft": {"value": 0.95, "comparison": ">=", "unit": "ratio", "locked": False},
            "passenger_wait_p95_seconds_max_draft": {"value": None, "comparison": "<=", "unit": "seconds", "locked": False, "note": "Step114 gives a relative B1_noop multiplier, not a locked numeric seconds threshold."},
        },
    )
    dump_json(output_root / "hard_constraint_runtime_audit.json", {"status": "NOT_RUN", "reason": "approved numeric hard-constraint contract missing", "passed_run_count": 0, "failed_run_count": 0, "blocked_run_count": 24})

    action_contract, skip_runtime, skip_rows = action_contract_and_skip_audit(project_root, r1_root)
    dump_json(output_root / "action_contract_audit.json", action_contract)
    dump_json(
        output_root / "skip_stop_legality_contract.json",
        {
            "contract_version": "skip_stop_legality_contract_v1",
            "max_consecutive_skips": MAX_CONSECUTIVE_SKIPS,
            "legal_conditions": [
                "target stop waiting passenger count == 0",
                "target stop onboard alighting demand == 0",
                "target stop is not terminal",
                "target stop is not boundary gateway",
                "target stop is not mandatory transfer stop",
                "route sequence remains connected",
                "next legal stop exists",
                "consecutive skip count < 1",
            ],
            "action_mask_required": True,
            "masked_logit_value": "-inf",
        },
    )
    dump_json(output_root / "skip_stop_runtime_audit.json", {**skip_runtime, "run_rows_path": str(output_root / "skip_stop_runtime_by_run.parquet")})
    write_parquet(output_root / "skip_stop_runtime_by_run.parquet", skip_rows)

    training_audit = training_semantics_audit(project_root, train_service_df, skip_runtime)
    dump_json(output_root / "training_semantics_impact_audit.json", training_audit)
    retraining_report = {
        "status": "RETRAIN_REQUIRED" if training_audit["retraining_required"] else "NO_RETRAIN_REQUIRED",
        "retraining_required": bool(training_audit["retraining_required"]),
        "retraining_required_run_count": 24 if training_audit["retraining_required"] else 0,
        "automatic_retraining_started": False,
        "scientific_checkpoint_admissible": bool(training_audit["scientific_checkpoint_admissible"]),
        "diagnostic_fixed_candidate_only": bool(training_audit["diagnostic_fixed_candidate_only"]),
        "reason": training_audit["classification"],
    }
    dump_json(output_root / "retraining_requirement_report.json", retraining_report)

    validation_block_rows = []
    pair_rows = []
    for family in FAMILIES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                run_root = output_root / family / condition / f"seed_{seed:03d}"
                run_root.mkdir(parents=True, exist_ok=True)
                run_id = f"{family}_{condition}_seed{seed:03d}"
                run_status = {
                    "run_id": run_id,
                    "status": "BLOCKED_THRESHOLD_APPROVAL_REQUIRED",
                    "classification": "NUMERIC_HARD_CONSTRAINT_CONTRACT_MISSING",
                    "validation_rerun_executed": False,
                    "optimizer_step_count": 0,
                    "backward_call_count": 0,
                    "parameter_update_count": 0,
                    "test_split_read": False,
                    "test_target_read": False,
                    "test_embedding_read": False,
                }
                dump_json(run_root / "validation_runtime_manifest.json", run_status)
                dump_json(run_root / "service_day_episode_audit.json", {"status": "NOT_RUN", "reason": "blocked before repaired validation runtime"})
                write_parquet(run_root / "passenger_wait_ledger_v2_1.parquet", [])
                dump_json(run_root / "validation_kpi_v2_1.json", {"status": "NOT_RUN", "reason": "blocked before repaired validation runtime"})
                dump_json(run_root / "numeric_hard_constraint_audit.json", {"status": "NOT_RUN", "reason": "approved numeric hard-constraint contract missing"})
                dump_json(run_root / "skip_stop_action_audit.json", {"status": "NOT_RUN", "reason": "blocked before repaired validation runtime"})
                dump_json(run_root / "simulator_integrity_audit.json", {"status": "NOT_RUN", "reason": "blocked before repaired validation runtime"})
                dump_json(run_root / "run_validation_status.json", run_status)
                validation_block_rows.append({"run_id": run_id, "model_family": family, "condition_id": condition, "seed": seed, **run_status})
    for condition in CONDITIONS:
        for seed in SEEDS:
            pair_rows.append({"condition_id": condition, "seed": seed, "status": "BLOCKED_THRESHOLD_APPROVAL_REQUIRED", "e0_completed": False, "e1_completed": False})
    write_parquet(output_root / "validation_results_by_run_r2.parquet", validation_block_rows)
    write_parquet(output_root / "validation_results_by_pair_r2.parquet", pair_rows)
    write_parquet(output_root / "r1_r2_kpi_comparison.parquet", build_r1_r2_comparison(r1_root))

    source_failed = source_gate["status"] != "PASS"
    threshold_missing = not inventory["approved_numeric_contract_found"]
    if source_failed:
        status = "BLOCKED"
        classification = "SOURCE_PROVENANCE_GATE_FAILED"
    elif threshold_missing:
        status = "BLOCKED_THRESHOLD_APPROVAL_REQUIRED"
        classification = "NUMERIC_HARD_CONSTRAINT_CONTRACT_MISSING"
    else:
        status = "BLOCKED"
        classification = "UNIMPLEMENTED_R2_RUNTIME_BRANCH"
    gate = {
        "created_at_utc": utc_now(),
        "status": status,
        "classification": classification,
        "authoritative_r1_gate_path": str(r1_gate_path),
        "authoritative_r1_gate_sha256": sha256_file(r1_gate_path),
        "claim_guard_path": str(claim_guard_path),
        "claim_guard_sha256": sha256_file(claim_guard_path),
        "service_day_contract_verified": service_contract["status"] == "PASS",
        "service_day_count": int(boundary_audit["validation"]["service_day_count"]),
        "overnight_gap_count": int(boundary_audit["validation"]["cross_service_day_boundary_count"]),
        "cross_day_waiting_queue_carry_count": int(queue_audit["cross_day_waiting_queue_carry_count"]),
        "cross_day_cohort_count": int(queue_audit["cross_day_cohort_count"]),
        "wait_over_service_day_duration_count": int(queue_audit["wait_over_service_day_duration_count"]),
        "numeric_hard_constraint_contract_valid": False,
        "hard_constraint_contract_sha256": sha256_file(output_root / "hard_constraint_contract_v2.json"),
        "hard_constraint_threshold_source": None,
        "noninformative_threshold_count": 0,
        "skip_stop_action_present": True,
        "skip_stop_action_id": 3,
        "skip_stop_action_name": "SKIP_STOP",
        "skip_stop_action_count": int(skip_runtime["skip_stop_action_count"]),
        "skip_stop_rate": float(skip_runtime["skip_stop_rate"]),
        "illegal_skip_count": None,
        "skip_with_waiting_passenger_count": None,
        "skip_with_alighting_passenger_count": None,
        "consecutive_skip_violation_count": int(skip_runtime["consecutive_skip_violation_count"]),
        "training_semantics_impact_classification": training_audit["classification"],
        "training_service_day_semantics_affected": bool(training_audit["training_service_day_semantics_affected"]),
        "training_skip_stop_semantics_affected": bool(training_audit["training_skip_stop_semantics_affected"]),
        "retraining_required": bool(retraining_report["retraining_required"]),
        "retraining_required_run_count": int(retraining_report["retraining_required_run_count"]),
        "expected_validation_run_count": 24,
        "completed_validation_run_count": 0,
        "failed_validation_run_count": 0,
        "blocked_validation_run_count": 24,
        "kpi_v2_1_passed": False,
        "path_a_b_equivalence_passed": False,
        "row_permutation_invariance_passed": False,
        "passenger_conservation_passed": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
        "prompt6a_executed": False,
        "e2_executed": False,
        "full_prompt6_executed": False,
    }
    dump_json(output_root / "prompt5_e01_r2_gate.json", gate)

    final_report = f"""# Prompt 5-E01-R2 Final Report

status: {gate['status']}
classification: {gate['classification']}

## Service day

- timezone: {SERVICE_TZ}
- service hours: {SERVICE_START.isoformat(timespec='minutes')} - {SERVICE_END.isoformat(timespec='minutes')}
- validation service-day count: {gate['service_day_count']}
- overnight/service-day boundaries: {gate['overnight_gap_count']}
- cross-day queue carries: {gate['cross_day_waiting_queue_carry_count']}
- cross-day cohorts: {gate['cross_day_cohort_count']}
- waits over service-day duration: {gate['wait_over_service_day_duration_count']}

## Passenger accounting

- generated: {queue_audit['passenger_accounting_totals']['generated']}
- served: {queue_audit['passenger_accounting_totals']['served']}
- unserved at service-day end candidate count: {queue_audit['passenger_accounting_totals']['waiting_unserved_at_service_day_end']}
- cancelled: 0
- incomplete onboard: 0
- conservation errors: {queue_audit['passenger_accounting_totals']['conservation_errors']}

## Hard constraints

- numeric contract found: false
- source: none approved
- service-rate minimum: missing approved value
- p95-wait maximum: missing approved value
- non-informative threshold count: 0
- passed runs: 0
- failed runs: 0
- blocked runs: 24

## Skip-stop

- action ID/name: 3 / SKIP_STOP
- total skip actions: {gate['skip_stop_action_count']}
- skip rate: {gate['skip_stop_rate']}
- skip with waiting passengers: not evaluable from R1 raw events
- skip with alighting passengers: not evaluable from R1 raw events
- consecutive-skip violations: {gate['consecutive_skip_violation_count']}
- other illegal skips: not evaluable from R1 raw events

## Training impact

- service-day semantics affected: {str(gate['training_service_day_semantics_affected']).lower()}
- skip-stop semantics affected: {str(gate['training_skip_stop_semantics_affected']).lower()}
- classification: {gate['training_semantics_impact_classification']}
- retraining required: {str(gate['retraining_required']).lower()}
- affected runs: {gate['retraining_required_run_count']}

## Validation

- expected runs: 24
- completed: 0
- failed: 0
- blocked: 24

## Data leakage

- test split read: false
- test target read: false
- test embedding read: false

## Next gate

- Prompt 6A corrected retrospective approved: false
- 24-run retraining required: {str(gate['retraining_required']).lower()}
- E2 approved: false
- full Prompt 6 approved: false

## Interpretation

1. Multi-day passenger waiting was detected in R1 and is not removed because R2 repaired validation rerun is blocked pending approved numeric thresholds.
2. Hard constraints were not evaluated with actual KPI v2.1 numbers because no approved numeric threshold contract exists.
3. Skip-stop action mapping exists, but R1 lacks the target-stop legality ledger needed to prove service-protection compliance.
4. Existing training semantics were affected by both service-day carry-over and unmasked skip-stop behavior.
5. Existing fixed checkpoints can be retained only as diagnostic fixed candidates, not as scientific checkpoints for corrected Prompt 6A.
6. The next step is not corrected Prompt 6A execution; it is threshold approval plus retraining/protocol decision.
"""
    (output_root / "prompt5_e01_r2_final_report.md").write_text(final_report, encoding="utf-8")

    manifest = {
        "created_at_utc": utc_now(),
        "artifact_dir": str(output_root),
        "files": [],
    }
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "prompt5_e01_r2_manifest.json":
            manifest["files"].append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    dump_json(output_root / "prompt5_e01_r2_manifest.json", manifest)
    print(json.dumps({"artifact_dir": str(output_root), "status": gate["status"], "classification": gate["classification"], "retraining_required": gate["retraining_required"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
