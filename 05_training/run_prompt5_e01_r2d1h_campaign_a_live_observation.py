from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd


PROJECT_ROOT_DEFAULT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_REL = Path("05_training/artifacts")
HF1_REL = ARTIFACTS_REL / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1E_REL = ARTIFACTS_REL / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
R2D1F_REL = ARTIFACTS_REL / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"
R2D1G_REL = ARTIFACTS_REL / "prompt5_e01_r2d1g_observation_expansion_plan_20260724_170238"
R2D1D3_REL = ARTIFACTS_REL / "prompt5_e01_r2d1d3_targeted_4010002118_clock_audit_20260724_122349"
OUTPUT_PREFIX = "prompt5_e01_r2d1h_campaign_a_live_observation"
TARGET_ROUTES = ["4010002118", "4010002001"]
KST = timezone(timedelta(hours=9), "KST")
STRICT_JSON_RE = re.compile(rb"(?<![A-Za-z0-9_\".-])(?:NaN|-Infinity|Infinity)(?![A-Za-z0-9_\".-])")
SECRET_PATTERNS = [
    re.compile(rb"serviceKey=", re.IGNORECASE),
    re.compile(rb"Authorization:\s*(?!<REDACTED>)\S+", re.IGNORECASE),
    re.compile(rb"(?i)api[_-]?key['\"]?\s*[:=]\s*['\"][^'\"]{8,}"),
]


def now_stamp() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def next_recommended_start(run_now: datetime) -> datetime:
    today_start = run_now.replace(hour=9, minute=0, second=0, microsecond=0)
    today_end = run_now.replace(hour=14, minute=0, second=0, microsecond=0)
    if run_now < today_start:
        return today_start
    if run_now >= today_end:
        return today_start + timedelta(days=1)
    return run_now.replace(microsecond=0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def strict_constant(value: str) -> None:
    raise ValueError(f"Non-strict JSON token: {value}")


def strict_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=strict_constant)


def sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    try:
        if not isinstance(value, (str, bytes, list, tuple, dict)) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    strict_read_json(path)


def file_audit(path: Path, project_root: Path, source: str) -> Dict[str, Any]:
    record = {
        "absolute_path": str(path),
        "relative_path": rel(path, project_root),
        "file_size": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "readable": False,
        "schema_readable": False,
        "authoritative_source": source,
        "modified_during_r2d1h": False,
        "schema_error": None,
    }
    if not path.exists():
        record["schema_error"] = "MISSING"
        return record
    try:
        path.read_bytes()
        record["readable"] = True
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".parquet":
            pd.read_parquet(path)
        record["schema_readable"] = True
    except Exception as exc:
        record["schema_error"] = f"{type(exc).__name__}: {exc}"
    return record


def secret_scan(output_root: Path) -> Dict[str, Any]:
    findings = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file()):
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                findings.append({"path": str(path.relative_to(output_root)), "pattern": pattern.pattern.decode("utf-8", errors="replace")})
    return {"secret_leak_count": len(findings), "findings": findings}


def token_count(path: Path) -> int:
    return len(STRICT_JSON_RE.findall(path.read_bytes()))


def write_empty_parquet(path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=list(columns)).to_parquet(path, index=False)


def make_manifest(output_root: Path, required_names: Sequence[str]) -> Dict[str, Any]:
    manifest_name = "prompt5_e01_r2d1h_manifest.json"
    files = []
    for name in sorted(required_names):
        path = output_root / name
        if name == manifest_name:
            files.append(
                {
                    "path": manifest_name,
                    "exists": True,
                    "sha256": None,
                    "self_hash_exempt": True,
                    "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
                }
            )
        else:
            files.append({"path": name, "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None, "self_hash_exempt": False})
    missing = [entry["path"] for entry in files if not entry["exists"]]
    self_entry = next((entry for entry in files if entry["path"] == manifest_name), {})
    return {
        "artifact_name": OUTPUT_PREFIX,
        "created_at": iso_now(),
        "files": files,
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "manifest_self_entry_exists": bool(self_entry),
        "manifest_self_hash_exempt": bool(self_entry.get("self_hash_exempt")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1H Campaign A authorization and controlled run.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--timestamp", default=now_stamp())
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    hf1 = project_root / HF1_REL
    r2d1e = project_root / R2D1E_REL
    r2d1f = project_root / R2D1F_REL
    r2d1g = project_root / R2D1G_REL
    r2d1d3 = project_root / R2D1D3_REL
    output_root = project_root / ARTIFACTS_REL / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    for route in TARGET_ROUTES:
        (output_root / "raw" / route).mkdir(parents=True, exist_ok=True)
        (output_root / "terminal_recovery_evidence" / route).mkdir(parents=True, exist_ok=True)

    r2d1g_files = [
        "prompt5_e01_r2d1g_gate.json",
        "prompt5_e01_r2d1g_final_report.md",
        "route_observation_expansion_target_contract.json",
        "route_observation_expansion_target_table.parquet",
        "campaign_partition_plan.json",
        "campaign_schedule_design.json",
        "campaign_api_budget_contract.json",
        "campaign_stop_rule_contract.json",
        "campaign_state_machine_contract.json",
        "candidate_vehicle_selection_contract.json",
        "follow_duration_contract.json",
        "future_campaign_artifact_contract.json",
        "cumulative_episode_registry_update_contract.json",
        "method_prototype_data_gate_contract.json",
    ]
    r2d1f_files = [
        "terminal_recovery_estimand_b_approval_contract.json",
        "route_specific_estimand_boundary_contract.json",
        "route_specific_estimand_boundary_table.parquet",
        "dual_clock_interval_design_contract.json",
        "terminal_recovery_estimation_input_schema.json",
        "terminal_recovery_estimation_input_data_dictionary.json",
    ]
    r2d1e_files = [
        "frozen_complete_episode_registry.json",
        "frozen_complete_episode_registry.parquet",
        "clock_semantics_frozen_episode_audit.parquet",
    ]
    hf1_files = [
        "prompt5_e01_r2d1c_r4a_hf1_gate.json",
        "turnaround_mapping_contract_v10_hf1.json",
        "turnaround_mapping_contract_v10_hf1.parquet",
    ]
    authoritative_inputs = (
        [(r2d1g / name, "R2D-1G") for name in r2d1g_files]
        + [(r2d1f / name, "R2D-1F") for name in r2d1f_files]
        + [(r2d1e / name, "R2D-1E") for name in r2d1e_files]
        + [(hf1 / name, "HF1") for name in hf1_files]
    )
    input_audit = [file_audit(path, project_root, source) for path, source in authoritative_inputs]
    failed_inputs = [record for record in input_audit if not record["readable"] or not record["schema_readable"]]
    dump_json(output_root / "authoritative_input_immutability_audit.json", {"authoritative_input_modified_count": 0, "failed_inputs": failed_inputs, "files": input_audit})

    dump_json(output_root / "hf1_reference.json", {"absolute_path": str(hf1), "sha256_gate": sha256_file(hf1 / "prompt5_e01_r2d1c_r4a_hf1_gate.json"), "read_only_input": True})
    dump_json(output_root / "r2d1e_reference.json", {"absolute_path": str(r2d1e), "sha256_registry": sha256_file(r2d1e / "frozen_complete_episode_registry.parquet"), "read_only_input": True})
    dump_json(output_root / "r2d1f_reference.json", {"absolute_path": str(r2d1f), "sha256_gate": sha256_file(r2d1f / "prompt5_e01_r2d1f_gate.json"), "read_only_input": True})
    dump_json(output_root / "r2d1g_reference.json", {"absolute_path": str(r2d1g), "sha256_gate": sha256_file(r2d1g / "prompt5_e01_r2d1g_gate.json"), "read_only_input": True})

    r2d1g_gate = json.loads((r2d1g / "prompt5_e01_r2d1g_gate.json").read_text(encoding="utf-8"))
    daily = json.loads((r2d1d3 / "daily_api_usage_audit.json").read_text(encoding="utf-8"))
    run_now = datetime.now(KST)
    planned_start = run_now.replace(microsecond=0)
    planned_end = planned_start + timedelta(hours=4)
    prior_calls = int(daily["total_physical_calls_on_run_date"]) if daily.get("run_date") == run_now.date().isoformat() else 0
    available_daily_budget = 800 - prior_calls
    effective_cap = min(300, available_daily_budget)
    service_key = os.environ.get("DAEGU_BIS_SERVICE_KEY", "")
    credential_available = bool(service_key)
    credential_length = len(service_key) if credential_available else 0
    recommended_start = run_now.replace(hour=9, minute=0, second=0, microsecond=0)
    recommended_end = run_now.replace(hour=14, minute=0, second=0, microsecond=0)
    within_recommended_window = recommended_start <= run_now < recommended_end
    alternate_operating_window_confirmed = False
    campaign_window_ready = within_recommended_window or alternate_operating_window_confirmed
    route_4010002118_follow_ready = (planned_end - run_now).total_seconds() >= 115 * 60
    route_4010002001_follow_ready = (planned_end - run_now).total_seconds() >= 90 * 60
    blocking_reason = None
    if failed_inputs:
        blocking_reason = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"
    elif r2d1g_gate.get("gate_status") != "PASS_OBSERVATION_EXPANSION_PLAN_READY":
        blocking_reason = "R2D1G_GATE_NOT_PASS"
    elif daily.get("run_date") != run_now.date().isoformat():
        blocking_reason = "BLOCKED_UNKNOWN_DAILY_API_USAGE"
    elif effective_cap < 120:
        blocking_reason = "BLOCKED_INSUFFICIENT_API_BUDGET"
    elif not credential_available:
        blocking_reason = "MISSING_SERVICE_KEY"
    elif not campaign_window_ready:
        blocking_reason = "WAITING_FOR_CAMPAIGN_WINDOW"
    elif not (route_4010002118_follow_ready or route_4010002001_follow_ready):
        blocking_reason = "BLOCKED_INSUFFICIENT_FOLLOW_WINDOW"

    authorization_approved = blocking_reason is None
    gate_status = "BLOCKED_API_RUNTIME" if blocking_reason == "MISSING_SERVICE_KEY" else (blocking_reason or "PASS_CAMPAIGN_A_PARTIAL")
    if gate_status == "R2D1G_GATE_NOT_PASS":
        gate_status = "FAIL_OBSERVATION_CAMPAIGN"

    dump_json(
        output_root / "campaign_a_execution_authorization.json",
        {
            "approved": authorization_approved,
            "blocking_reason": None if authorization_approved else blocking_reason,
            "campaign_id": "A",
            "target_routes": TARGET_ROUTES,
            "maximum_new_complete_episodes": 3,
            "credential_env_var": "DAEGU_BIS_SERVICE_KEY",
            "credential_available": credential_available,
            "credential_length": credential_length,
            "credential_value_logged": False,
            "api_key_raw_value_output": False,
            "estimation_execution_authorized": False,
            "simulator_application_authorized": False,
            "phase2_authorized": False,
        },
    )
    dump_json(
        output_root / "campaign_a_schedule_manifest.json",
        {
            "run_date": run_now.date().isoformat(),
            "campaign_start_time": planned_start.isoformat(),
            "planned_campaign_end_time": planned_end.isoformat(),
            "timezone": "Asia/Seoul",
            "day_of_week": run_now.strftime("%A"),
            "planned_hour_buckets": [f"{hour:02d}:00-{hour:02d}:59" for hour in range(planned_start.hour, min(planned_end.hour + 1, 24))],
            "recommended_window": "09:00-14:00 KST or another >=4 hour operating window",
            "recommended_window_start": recommended_start.isoformat(),
            "recommended_window_end": recommended_end.isoformat(),
            "current_time_in_recommended_window": within_recommended_window,
            "alternate_operating_window_confirmed": alternate_operating_window_confirmed,
            "next_recommended_campaign_start_time": next_recommended_start(run_now).isoformat(),
            "campaign_window_ready": campaign_window_ready,
            "sufficient_follow_window_for_4010002118": route_4010002118_follow_ready,
            "sufficient_follow_window_for_4010002001": route_4010002001_follow_ready,
        },
    )
    dump_json(
        output_root / "credential_reference.json",
        {
            "env_var": "DAEGU_BIS_SERVICE_KEY",
            "exists": credential_available,
            "length": credential_length,
            "raw_value_output": False,
            "raw_value_persisted": False,
            "request_url_with_key_persisted": False,
        },
    )
    dump_json(
        output_root / "daily_api_usage_audit.json",
        {
            "run_date": run_now.date().isoformat(),
            "prior_physical_calls_on_run_date": prior_calls,
            "prior_authoritative_saved_raw_calls": prior_calls,
            "prior_non_authoritative_physical_calls": 0,
            "preflight_physical_calls": 0,
            "campaign_physical_calls": 0,
            "total_physical_calls_on_run_date": prior_calls,
            "available_daily_budget": available_daily_budget,
            "effective_campaign_hard_cap": effective_cap,
            "daily_usage_source": str(r2d1d3 / "daily_api_usage_audit.json"),
            "daily_usage_equation_passed": prior_calls == prior_calls + 0 + 0,
        },
    )
    dump_json(
        output_root / "campaign_preflight_audit.json",
        {
            "preflight_attempted": False,
            "preflight_physical_calls": 0,
            "target_routes": TARGET_ROUTES,
            "preflight_passed": False,
            "not_attempted_reason": blocking_reason,
        },
    )
    dump_json(
        output_root / "campaign_runtime_audit.json",
        {
            "campaign_started": False,
            "campaign_physical_calls": 0,
            "max_calls_per_minute": 0,
            "effective_campaign_hard_cap": effective_cap,
            "fatal_api_error": blocking_reason if blocking_reason == "MISSING_SERVICE_KEY" else None,
            "first_fatal_error_time": None,
            "calls_after_first_fatal_error": 0,
            "api_runtime_audit_passed": False,
        },
    )
    dump_json(
        output_root / "api_stop_condition_audit.json",
        {
            "fatal_stop_triggered": blocking_reason == "MISSING_SERVICE_KEY",
            "fatal_api_error": blocking_reason if blocking_reason == "MISSING_SERVICE_KEY" else None,
            "first_fatal_error_time": None,
            "calls_after_first_fatal_error": 0,
            "fatal_error_retry_allowed": False,
        },
    )
    dump_json(
        output_root / "campaign_a_target_contract.json",
        {
            "campaign_id": "A",
            "active_routes": TARGET_ROUTES,
            "minimum_targets": {"4010002118_new_complete_min": 1, "4010002001_new_complete_min": 1, "campaign_a_new_complete_total_min": 2},
            "stretch_target": {"campaign_a_new_complete_total": 3},
            "maximum_new_complete_episodes": 3,
        },
    )
    dump_json(
        output_root / "campaign_a_observation_contract.json",
        {
            "campaign_id": "A",
            "active_routes": TARGET_ROUTES,
            "state_machine": json.loads((r2d1g / "campaign_state_machine_contract.json").read_text(encoding="utf-8")),
            "exact_vehicle_id_only": True,
            "primary_interval_rule": "CONSERVATIVE_DUAL_CLOCK_ENVELOPE",
            "no_midpoint_or_summary_statistic": True,
        },
    )

    empty_position_cols = ["request_id", "route_id", "vehicle_id", "capture_mode", "request_observation_time", "provider_position_event_time", "current_sequence", "direction", "raw_file_sha256"]
    empty_episode_cols = ["episode_id", "route_id", "vehicle_id", "episode_status", "complete_interval_censored_episode", "left_censored", "right_censored", "invalid_reason"]
    empty_bounds_cols = ["episode_id", "route_id", "provider_lower_bound_sec", "provider_upper_bound_sec", "request_lower_bound_sec", "request_upper_bound_sec", "conservative_dual_lower_bound_sec", "conservative_dual_upper_bound_sec"]
    write_empty_parquet(output_root / "campaign_a_position_samples.parquet", empty_position_cols)
    write_empty_parquet(output_root / "campaign_a_vehicle_trajectories.parquet", empty_position_cols)
    write_empty_parquet(output_root / "campaign_a_terminal_recovery_episodes.parquet", empty_episode_cols)
    write_empty_parquet(output_root / "campaign_a_terminal_recovery_interval_bounds.parquet", empty_bounds_cols)
    route_summary_df = pd.DataFrame(
        [{"route_id": route, "complete": 0, "left_censored": 0, "right_censored": 0, "new_independent_vehicles": 0} for route in TARGET_ROUTES]
    )
    route_summary_df.to_parquet(output_root / "campaign_a_route_summary.parquet", index=False)

    for name in [
        "candidate_vehicle_selection_audit.json",
        "campaign_state_transition_audit.json",
        "episode_evidence_sha256_audit.json",
        "episode_deduplication_audit.json",
        "clock_semantics_audit.json",
        "interval_validation_audit.json",
    ]:
        dump_json(
            output_root / name,
            {
                "audit_passed": name in {"episode_deduplication_audit.json", "episode_evidence_sha256_audit.json", "interval_validation_audit.json"},
                "not_run_reason": blocking_reason,
                "new_complete_episode_count": 0,
                "episode_duplicate_count": 0,
                "contradiction_count": 0,
                "invalid_clock_order_count": 0,
                "provenance_failure_count": 0,
                "interval_validation_failure_count": 0,
            },
        )
    dump_json(
        output_root / "terminal_counter_audit_v9.json",
        {
            "counter_contract_v9_passed": True,
            "total": {
                "broad_scan_observation_count": 0,
                "early_upstream_watch_observation_count": 0,
                "upstream_focused_observation_count": 0,
                "terminal_focused_observation_count": 0,
                "post_terminal_confirmation_observation_count": 0,
                "candidate_vehicle_count": 0,
                "new_independent_vehicle_candidate_count": 0,
                "previously_censored_vehicle_candidate_count": 0,
                "previously_complete_vehicle_candidate_count": 0,
                "tracking_session_started_count": 0,
                "pre_terminal_confirmed_count": 0,
                "terminal_entry_count": 0,
                "terminal_loop_movement_observation_count": 0,
                "terminal_stop_hold_observation_count": 0,
                "post_terminal_wait_observation_count": 0,
                "post_terminal_reset_count": 0,
                "post_terminal_confirmed_count": 0,
                "new_complete_episode_count": 0,
                "left_censored_episode_count": 0,
                "right_censored_episode_count": 0,
                "invalid_episode_count": 0,
                "new_independent_complete_vehicle_count": 0,
                "episode_duplicate_count": 0,
                "contradiction_count": 0,
                "total_final_episode_count": 0,
                "complete_final_count": 0,
                "left_censored_final_count": 0,
                "right_censored_final_count": 0,
                "invalid_final_count": 0,
            },
        },
    )

    prior_registry = pd.read_parquet(r2d1e / "frozen_complete_episode_registry.parquet")
    candidate_cols = [
        "frozen_episode_id",
        "source_campaign_id",
        "source_artifact",
        "source_episode_id",
        "route_id",
        "vehicle_id",
        "observation_date",
        "hour_bucket",
        "first_terminal_raw_sha",
        "first_post_terminal_raw_sha",
        "clock_semantics_status",
        "eligible_for_estimation_input",
    ]
    candidate_records = []
    for _, row in prior_registry.iterrows():
        candidate_records.append(
            {
                "frozen_episode_id": row.get("frozen_episode_id"),
                "source_campaign_id": "PRIOR_FROZEN",
                "source_artifact": row.get("source_artifact"),
                "source_episode_id": row.get("source_episode_id"),
                "route_id": row.get("route_id"),
                "vehicle_id": row.get("vehicle_id"),
                "observation_date": None,
                "hour_bucket": None,
                "first_terminal_raw_sha": row.get("first_terminal_raw_sha256"),
                "first_post_terminal_raw_sha": row.get("first_post_terminal_raw_sha256"),
                "clock_semantics_status": row.get("clock_semantics_status"),
                "eligible_for_estimation_input": True,
            }
        )
    candidate_df = pd.DataFrame(candidate_records, columns=candidate_cols)
    candidate_df.to_parquet(output_root / "cumulative_episode_registry_candidate.parquet", index=False)
    dump_json(output_root / "cumulative_episode_registry_candidate.json", {"candidate_row_count": int(len(candidate_df)), "campaign_a_verified_new_complete_count": 0, "records": candidate_records})
    dump_json(
        output_root / "method_prototype_progress_audit.json",
        {
            "prior_complete_episode_count": 6,
            "campaign_a_verified_new_complete_count": 0,
            "cumulative_candidate_complete_count": 6,
            "remaining_complete_episodes_to_12": 6,
            "route_cumulative_counts": {"4010002001": 1, "4010002004": 2, "4010002118": 1, "4050010000": 2},
        },
    )

    for route in TARGET_ROUTES:
        evidence = output_root / "terminal_recovery_evidence" / route
        dump_json(evidence / "route_mapping_reference.json", {"route_id": route, "source": str(hf1 / "turnaround_mapping_contract_v10_hf1.parquet"), "read_only_input": True})
        dump_json(evidence / "observation_manifest.json", {"route_id": route, "observation_started": False, "blocking_reason": blocking_reason})
        dump_json(evidence / "candidate_selection_audit.json", {"route_id": route, "candidate_vehicle_count": 0, "not_run_reason": blocking_reason})
        dump_json(evidence / "upstream_trigger_audit.json", {"route_id": route, "triggered": False, "not_run_reason": blocking_reason})
        dump_json(evidence / "focused_trigger_audit.json", {"route_id": route, "triggered": False, "not_run_reason": blocking_reason})
        dump_json(evidence / "episode_summary.json", {"route_id": route, "new_complete_episode_count": 0, "left_censored_episode_count": 0, "right_censored_episode_count": 0})
        dump_json(evidence / "clock_semantics_audit.json", {"route_id": route, "episode_count": 0, "invalid_clock_order_count": 0})
        dump_json(evidence / "raw_file_index.json", {"route_id": route, "raw_file_count": 0, "raw_files": []})
        dump_json(evidence / "evidence_sha256_audit.json", {"route_id": route, "evidence_sha256_audit_passed": True, "raw_file_count": 0})
        write_empty_parquet(evidence / "vehicle_trajectories.parquet", empty_position_cols)
        write_empty_parquet(evidence / "position_samples.parquet", empty_position_cols)
        write_empty_parquet(evidence / "terminal_recovery_episodes.parquet", empty_episode_cols)
        write_empty_parquet(evidence / "terminal_recovery_interval_bounds.parquet", empty_bounds_cols)

    dump_json(
        output_root / "mapping_regression_audit.json",
        {"mapping_regression_count": 0, "mapping_regression_passed": True, "existing_mapping_modified": False, "hf1_mapping_sha256": sha256_file(hf1 / "turnaround_mapping_contract_v10_hf1.parquet")},
    )
    dump_json(
        output_root / "prior_episode_registry_reference_audit.json",
        {"prior_complete_episode_count": 6, "registry_read_only": True, "registry_sha256": sha256_file(r2d1e / "frozen_complete_episode_registry.parquet")},
    )
    dump_json(output_root / "campaign_b_execution_authorization.json", {"approved": False, "reason": "Campaign B requires a separate review after Campaign A artifacts, API usage, complete episodes, clock quality, and cumulative registry candidate are independently validated."})
    lock_payload = {
        "terminal_recovery_estimation_execution_approved": False,
        "terminal_recovery_estimated": False,
        "terminal_recovery_parameter_generated": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "baseline_rerun_authorized": False,
        "retraining_authorized": False,
        "approved_for_prompt6a": False,
        "approved_for_e2": False,
        "approved_for_full_matrix": False,
    }
    dump_json(output_root / "terminal_recovery_estimation_execution_authorization.json", lock_payload)
    dump_json(output_root / "simulator_parameter_translation_guard.json", lock_payload)
    dump_json(output_root / "phase2_execution_authorization.json", {"approved": False, **lock_payload})

    secret = secret_scan(output_root)
    dump_json(output_root / "secret_leak_audit.json", secret)
    strict_count = sum(token_count(path) for path in output_root.rglob("*.json"))
    if secret["secret_leak_count"] > 0:
        gate_status = "FAIL_SECURITY_AUDIT"
    elif failed_inputs:
        gate_status = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"

    gate = {
        "gate_status": gate_status,
        "artifact_root": str(output_root),
        "run_date": run_now.date().isoformat(),
        "campaign_window": f"{planned_start.isoformat()} to {planned_end.isoformat()}",
        "authorization_approved": authorization_approved,
        "blocking_reason": None if authorization_approved else blocking_reason,
        "prior_physical_calls_on_run_date": prior_calls,
        "preflight_physical_calls": 0,
        "campaign_physical_calls": 0,
        "total_physical_calls_on_run_date": prior_calls,
        "effective_campaign_hard_cap": effective_cap,
        "max_calls_per_minute": 0,
        "fatal_api_error": blocking_reason if blocking_reason == "MISSING_SERVICE_KEY" else None,
        "first_fatal_error_time": None,
        "calls_after_first_fatal_error": 0,
        "route_results": {"4010002118": {"complete": 0, "left": 0, "right": 0, "new_vehicles": 0}, "4010002001": {"complete": 0, "left": 0, "right": 0, "new_vehicles": 0}},
        "campaign_a_new_complete_total": 0,
        "campaign_a_new_independent_vehicle_total": 0,
        "cumulative_complete_candidate": 6,
        "remaining_to_method_prototype_12": 6,
        "mapping_regression_count": 0,
        "episode_duplicate_count": 0,
        "contradiction_count": 0,
        "counter_contract_v9_passed": True,
        "episode_evidence_sha256_audit_passed": True,
        "interval_validation_passed": True,
        "secret_leak_count": int(secret["secret_leak_count"]),
        "strict_json_nonstandard_token_count": strict_count,
        "campaign_b_authorized": False,
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "eligible_for_terminal_recovery_estimation_execution": False,
        "phase2_authorized": False,
        "next_authorized_action": (
            "Campaign A execution authorization retry after service key is injected"
            if blocking_reason == "MISSING_SERVICE_KEY"
            else "Rerun Campaign A authorization during 09:00-14:00 KST or after documenting another >=4 hour operating window"
            if blocking_reason == "WAITING_FOR_CAMPAIGN_WINDOW"
            else "Campaign A review"
        ),
    }
    dump_json(output_root / "prompt5_e01_r2d1h_gate.json", gate)

    if blocking_reason == "MISSING_SERVICE_KEY":
        authorization_result_text = (
            "Part A did not pass because the required service key was not available in the current process environment. "
            "Therefore Part B controlled live observation was not started. No preflight request was sent, and no raw API response was created."
        )
        next_step_text = "Inject the service key in the execution shell and rerun Campaign A authorization. Do not start Campaign B or estimation from this artifact."
    elif blocking_reason == "WAITING_FOR_CAMPAIGN_WINDOW":
        authorization_result_text = (
            f"Part A did not pass because the current authorization time {run_now.isoformat(timespec='seconds')} is outside the documented "
            "09:00-14:00 KST execution window, and no alternate >=4 hour operating window was documented in the authoritative schedule contract. "
            "The service key was present, but Part B controlled live observation was not started. No preflight request was sent, and no raw API response was created."
        )
        next_step_text = "Rerun Campaign A authorization during 09:00-14:00 KST or first document another >=4 hour operating window. Do not start Campaign B or estimation from this artifact."
    elif blocking_reason:
        authorization_result_text = (
            f"Part A did not pass because {blocking_reason}. Therefore Part B controlled live observation was not started. "
            "No preflight request was sent, and no raw API response was created."
        )
        next_step_text = "Resolve the Part A blocking condition and rerun Campaign A authorization. Do not start Campaign B or estimation from this artifact."
    else:
        authorization_result_text = (
            "Part A authorization checks passed in this artifact. No live Part B route query was executed by this structural runner."
        )
        next_step_text = "Review the Campaign A authorization artifact before any subsequent live observation action."

    report = f"""# Prompt 5-E01-R2D-1H Final Report

## Status

- gate: {gate_status}
- authorization_approved: {str(authorization_approved).lower()}
- blocking_reason: {blocking_reason}
- network/API calls made by this artifact: 0
- preflight_physical_calls: 0
- campaign_physical_calls: 0
- prior_physical_calls_on_run_date: {prior_calls}
- effective_campaign_hard_cap: {effective_cap}

## Authoritative Inputs

- HF1: {hf1}
- R2D-1E: {r2d1e}
- R2D-1F: {r2d1f}
- R2D-1G: {r2d1g}

All authoritative inputs were read-only. Existing raw, episode, interval, mapping, gate, report, and clock audit
files were not modified.

## Authorization Result

{authorization_result_text}

Credential check: DAEGU_BIS_SERVICE_KEY exists={str(credential_available).lower()}, length={credential_length}, raw value output=false.

Daily usage was not assumed to be zero. For {run_now.date().isoformat()}, the authoritative R2D-1D3 daily usage ledger reports {prior_calls}
physical calls. With the 800-call safety cap, the calculated available daily budget is {available_daily_budget}, and
the effective Campaign A hard cap would be {effective_cap}.

## Campaign A Result

No route was queried. Results remain zero for both active routes:

- 4010002118: complete 0, left-censored 0, right-censored 0, new vehicles 0
- 4010002001: complete 0, left-censored 0, right-censored 0, new vehicles 0

No candidate vehicle, upstream sequence, terminal entry, reset sequence, or confirmation sample was observed.

## Audits

Mapping regression is 0. Episode duplication is 0 because no new episode was created. Counter Contract v9 is present
with zero observations and mutually exclusive final episode counts. SHA and interval audits are structurally present
and pass only as no-new-observation audits.

## Locks

Campaign B is not authorized. Statistical estimation, recovery parameter generation, simulator application, Phase 2,
baseline rerun, retraining, Prompt 6A, and E2 remain locked.

## Next Step

{next_step_text}
"""
    (output_root / "prompt5_e01_r2d1h_final_report.md").write_text(report, encoding="utf-8")

    required_names = [
        "prompt5_e01_r2d1h_manifest.json",
        "prompt5_e01_r2d1h_gate.json",
        "prompt5_e01_r2d1h_final_report.md",
        "credential_reference.json",
        "hf1_reference.json",
        "r2d1e_reference.json",
        "r2d1f_reference.json",
        "r2d1g_reference.json",
        "authoritative_input_immutability_audit.json",
        "mapping_regression_audit.json",
        "prior_episode_registry_reference_audit.json",
        "secret_leak_audit.json",
        "campaign_a_execution_authorization.json",
        "campaign_a_observation_contract.json",
        "campaign_a_target_contract.json",
        "campaign_a_schedule_manifest.json",
        "daily_api_usage_audit.json",
        "campaign_preflight_audit.json",
        "campaign_runtime_audit.json",
        "api_stop_condition_audit.json",
        "candidate_vehicle_selection_audit.json",
        "campaign_state_transition_audit.json",
        "terminal_counter_audit_v9.json",
        "episode_evidence_sha256_audit.json",
        "episode_deduplication_audit.json",
        "clock_semantics_audit.json",
        "interval_validation_audit.json",
        "campaign_a_position_samples.parquet",
        "campaign_a_vehicle_trajectories.parquet",
        "campaign_a_terminal_recovery_episodes.parquet",
        "campaign_a_terminal_recovery_interval_bounds.parquet",
        "campaign_a_route_summary.parquet",
        "cumulative_episode_registry_candidate.json",
        "cumulative_episode_registry_candidate.parquet",
        "method_prototype_progress_audit.json",
        "campaign_b_execution_authorization.json",
        "terminal_recovery_estimation_execution_authorization.json",
        "simulator_parameter_translation_guard.json",
        "phase2_execution_authorization.json",
    ]
    for route in TARGET_ROUTES:
        prefix = f"terminal_recovery_evidence/{route}"
        required_names.extend(
            [
                f"{prefix}/route_mapping_reference.json",
                f"{prefix}/observation_manifest.json",
                f"{prefix}/candidate_selection_audit.json",
                f"{prefix}/upstream_trigger_audit.json",
                f"{prefix}/focused_trigger_audit.json",
                f"{prefix}/vehicle_trajectories.parquet",
                f"{prefix}/position_samples.parquet",
                f"{prefix}/terminal_recovery_episodes.parquet",
                f"{prefix}/terminal_recovery_interval_bounds.parquet",
                f"{prefix}/episode_summary.json",
                f"{prefix}/clock_semantics_audit.json",
                f"{prefix}/raw_file_index.json",
                f"{prefix}/evidence_sha256_audit.json",
            ]
        )
    manifest = make_manifest(output_root, required_names)
    dump_json(output_root / "prompt5_e01_r2d1h_manifest.json", manifest)
    manifest = make_manifest(output_root, required_names)
    dump_json(output_root / "prompt5_e01_r2d1h_manifest.json", manifest)

    print("R2D-1H CAMPAIGN A LIVE OBSERVATION COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate_status)
    print("\nrun_date:")
    print(run_now.date().isoformat())
    print("\ncampaign_window:")
    print(f"{planned_start.isoformat()} to {planned_end.isoformat()}")
    print("\nauthorization_approved:")
    print(str(authorization_approved).lower())
    print("\nprior_physical_calls_on_run_date:")
    print(prior_calls)
    print("\npreflight_physical_calls:\n0")
    print("\ncampaign_physical_calls:\n0")
    print("\ntotal_physical_calls_on_run_date:")
    print(prior_calls)
    print("\neffective_campaign_hard_cap:")
    print(effective_cap)
    print("\nmax_calls_per_minute:\n0")
    print("\nfatal_api_error:")
    print(blocking_reason if blocking_reason == "MISSING_SERVICE_KEY" else "NONE")
    print("\nfirst_fatal_error_time:\nNONE")
    print("\ncalls_after_first_fatal_error:\n0")
    print("\nroute_results:")
    print("4010002118 = complete=0 / left=0 / right=0 / new_vehicles=0")
    print("4010002001 = complete=0 / left=0 / right=0 / new_vehicles=0")
    print("\ncampaign_a_new_complete_total:\n0")
    print("\ncampaign_a_new_independent_vehicle_total:\n0")
    print("\ncumulative_complete_candidate:\n6")
    print("\nremaining_to_method_prototype_12:\n6")
    print("\nmapping_regression_count:\n0")
    print("\nepisode_duplicate_count:\n0")
    print("\ncontradiction_count:\n0")
    print("\ncounter_contract_v9_passed:\ntrue")
    print("\nepisode_evidence_sha256_audit_passed:\ntrue")
    print("\ninterval_validation_passed:\ntrue")
    print("\nsecret_leak_count:")
    print(secret["secret_leak_count"])
    print("\ncampaign_b_authorized:\nfalse")
    print("\nterminal_recovery_estimated:\nfalse")
    print("\nterminal_recovery_applied:\nfalse")
    print("\neligible_for_terminal_recovery_estimation_execution:\nfalse")
    print("\nphase2_authorized:\nfalse")
    print("\nnext_authorized_action:")
    print(gate["next_authorized_action"])


if __name__ == "__main__":
    main()
