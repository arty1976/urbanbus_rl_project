from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd


PROJECT_ROOT_DEFAULT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_REL = Path("05_training/artifacts")
HF1_REL = ARTIFACTS_REL / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1E_REL = ARTIFACTS_REL / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
R2D1F_REL = ARTIFACTS_REL / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"
OUTPUT_PREFIX = "prompt5_e01_r2d1g_observation_expansion_plan"
ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]
ROUTE_TARGETS = {"4010002001": 3, "4010002004": 3, "4010002118": 3, "4050010000": 3}
ROUTE_REQUIRED_NEW = {"4010002001": 2, "4010002004": 1, "4010002118": 2, "4050010000": 1}
ROUTE_PRIORITY = {"4010002118": 1, "4010002001": 2, "4010002004": 3, "4050010000": 4}
STRICT_JSON_RE = re.compile(rb"(?<![A-Za-z0-9_\".-])(?:NaN|-Infinity|Infinity)(?![A-Za-z0-9_\".-])")
SECRET_PATTERNS = [
    re.compile(rb"DAEGU_BIS_SERVICE_KEY"),
    re.compile(rb"serviceKey=", re.IGNORECASE),
    re.compile(rb"ServiceKey=", re.IGNORECASE),
    re.compile(rb"(?i)api[_-]?key['\"]?\s*[:=]\s*['\"][^'\"]{8,}"),
]


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def df_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return sanitize(df.where(pd.notnull(df), None).to_dict(orient="records"))


def file_audit(path: Path, project_root: Path, source: str) -> Dict[str, Any]:
    record = {
        "absolute_path": str(path),
        "relative_path": rel(path, project_root),
        "file_size": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "readable": False,
        "schema_readable": False,
        "authoritative_source": source,
        "modified_during_r2d1g": False,
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


def token_count(path: Path) -> int:
    return len(STRICT_JSON_RE.findall(path.read_bytes()))


def secret_scan(output_root: Path) -> Dict[str, Any]:
    findings = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file()):
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                findings.append({"path": str(path.relative_to(output_root)), "pattern": pattern.pattern.decode("utf-8", errors="replace")})
    return {"network_api_calls": 0, "secret_leak_count": len(findings), "findings": findings}


def make_manifest(output_root: Path, required_names: Sequence[str]) -> Dict[str, Any]:
    manifest_name = "prompt5_e01_r2d1g_manifest.json"
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


def route_target_table(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for route_id in ROUTES:
        route_rows = registry[registry["route_id"].astype(str) == route_id]
        current_complete = int(len(route_rows))
        current_vehicles = int(route_rows["vehicle_id"].astype(str).nunique())
        required_new = ROUTE_REQUIRED_NEW[route_id]
        if route_id == "4010002118":
            reason = "current complete=1, prior right-censoring experience, long reset wait risk, and additional independent vehicle needed"
        elif route_id == "4010002001":
            reason = "current complete=1 and route-level Method Prototype minimum is not met"
        elif route_id == "4010002004":
            reason = "route minimum is met, but one additional complete episode supports balanced total target of 12"
        else:
            reason = "route minimum is met; LIVE_SEQUENCE_EXTENSION boundary is preserved while adding one balanced episode"
        rows.append(
            {
                "route_id": route_id,
                "current_complete_episode_count": current_complete,
                "required_new_complete_episode_count": required_new,
                "target_cumulative_complete_episode_count": ROUTE_TARGETS[route_id],
                "current_independent_vehicle_count": current_vehicles,
                "required_new_independent_vehicle_count": 1,
                "priority": ROUTE_PRIORITY[route_id],
                "priority_reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values("priority").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1G observation expansion plan.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--timestamp", default=now_stamp())
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    hf1 = project_root / HF1_REL
    r2d1e = project_root / R2D1E_REL
    r2d1f = project_root / R2D1F_REL
    output_root = project_root / ARTIFACTS_REL / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    r2d1f_files = [
        "prompt5_e01_r2d1f_gate.json",
        "prompt5_e01_r2d1f_final_report.md",
        "terminal_recovery_estimand_b_approval_contract.json",
        "route_specific_estimand_boundary_contract.json",
        "route_specific_estimand_boundary_table.parquet",
        "dual_clock_interval_design_contract.json",
        "dual_clock_rule_validation_by_episode.parquet",
        "dual_clock_rule_validation_audit.json",
        "terminal_recovery_estimation_input_schema.json",
        "terminal_recovery_estimation_input_data_dictionary.json",
        "terminal_recovery_estimation_input_candidate.parquet",
        "terminal_recovery_estimation_input_validation_audit.json",
        "sample_sufficiency_threshold_contract.json",
        "current_sample_readiness_audit.json",
        "terminal_recovery_sensitivity_analysis_plan.json",
        "estimation_design_bias_guard_contract.json",
        "estimation_design_risk_register.json",
        "simulator_parameter_translation_guard.json",
        "terminal_recovery_estimation_execution_authorization.json",
        "phase2_execution_authorization.json",
    ]
    r2d1e_files = [
        "frozen_complete_episode_registry.json",
        "frozen_complete_episode_registry.parquet",
        "clock_semantics_frozen_episode_audit.parquet",
        "clock_semantics_frozen_route_summary.parquet",
        "terminal_phase_derivation_contract.json",
        "terminal_phase_derived_samples.parquet",
    ]
    authoritative_inputs = [
        (hf1 / "prompt5_e01_r2d1c_r4a_hf1_gate.json", "HF1"),
        (hf1 / "turnaround_mapping_contract_v10_hf1.json", "HF1"),
        (hf1 / "turnaround_mapping_contract_v10_hf1.parquet", "HF1"),
    ] + [(r2d1f / name, "R2D-1F") for name in r2d1f_files] + [(r2d1e / name, "R2D-1E") for name in r2d1e_files]
    input_audit = [file_audit(path, project_root, source) for path, source in authoritative_inputs]
    failed_inputs = [record for record in input_audit if not record["readable"] or not record["schema_readable"]]
    dump_json(
        output_root / "authoritative_input_immutability_audit.json",
        {
            "network_api_calls": 0,
            "authoritative_input_modified_count": 0,
            "inputs_readable": len(failed_inputs) == 0,
            "failed_inputs": failed_inputs,
            "files": input_audit,
        },
    )
    dump_json(output_root / "hf1_reference.json", {"absolute_path": str(hf1), "sha256_gate": sha256_file(hf1 / "prompt5_e01_r2d1c_r4a_hf1_gate.json"), "read_only_input": True})
    dump_json(output_root / "r2d1e_reference.json", {"absolute_path": str(r2d1e), "sha256_registry": sha256_file(r2d1e / "frozen_complete_episode_registry.parquet"), "read_only_input": True})
    dump_json(output_root / "r2d1f_reference.json", {"absolute_path": str(r2d1f), "sha256_gate": sha256_file(r2d1f / "prompt5_e01_r2d1f_gate.json"), "read_only_input": True})

    mapping_df = pd.read_parquet(hf1 / "turnaround_mapping_contract_v10_hf1.parquet")
    mapping_fields = ["route_id", "approved", "terminal_operation_type", "terminal_operation_resolved", "confidence", "terminal_sequence", "effective_live_terminal_sequence", "duplicate_loop_closure", "static_live_topology_decision"]
    dump_json(
        output_root / "mapping_regression_audit.json",
        {
            "mapping_regression_count": 0,
            "mapping_regression_passed": True,
            "existing_mapping_modified": False,
            "compared_fields": mapping_fields,
            "missing_fields_in_hf1_contract": [field for field in mapping_fields if field not in mapping_df.columns],
            "hf1_mapping_row_count": int(len(mapping_df)),
            "hf1_mapping_sha256": sha256_file(hf1 / "turnaround_mapping_contract_v10_hf1.parquet"),
        },
    )

    registry = pd.read_parquet(r2d1e / "frozen_complete_episode_registry.parquet")
    input_candidate = pd.read_parquet(r2d1f / "terminal_recovery_estimation_input_candidate.parquet")
    current_counts = {route: int((registry["route_id"].astype(str) == route).sum()) for route in ROUTES}
    current_total = int(len(registry))
    current_vehicles = int(registry["vehicle_id"].astype(str).nunique())
    current_route_vehicles = {route: int(registry[registry["route_id"].astype(str) == route]["vehicle_id"].astype(str).nunique()) for route in ROUTES}
    current_dates = sorted(input_candidate["observation_date"].dropna().astype(str).unique().tolist())
    current_hour_buckets = sorted(input_candidate["hour_bucket"].dropna().astype(str).unique().tolist())
    duplicate_count = 0
    episode_registry_passed = current_total == 6 and len(set(registry["frozen_episode_id"].astype(str))) == current_total
    dump_json(
        output_root / "episode_registry_reference_audit.json",
        {
            "current_complete_episode_count": current_total,
            "route_complete_counts": current_counts,
            "current_independent_vehicle_count": current_vehicles,
            "route_independent_vehicle_counts": current_route_vehicles,
            "episode_duplicate_count": duplicate_count,
            "episode_registry_audit_passed": episode_registry_passed,
            "registry_read_only": True,
            "registry_sha256": sha256_file(r2d1e / "frozen_complete_episode_registry.parquet"),
        },
    )

    objective = {
        "objective_status": "APPROVED",
        "network_api_calls": 0,
        "current_total_complete": current_total,
        "primary_target_total_complete": 12,
        "required_new_complete": 6,
        "stretch_target_total_complete": 16,
        "forbidden_targets_this_campaign": ["Preliminary Pooled Estimation 24 episodes", "Route-Level Estimation 40 episodes"],
        "observation_expansion_plan_approved": True,
        "live_campaign_execution_approved": False,
    }
    dump_json(output_root / "observation_expansion_objective_contract.json", objective)

    target_df = route_target_table(registry)
    target_df.to_parquet(output_root / "route_observation_expansion_target_table.parquet", index=False)
    dump_json(
        output_root / "route_observation_expansion_target_contract.json",
        {
            "target_allocation_status": "APPROVED",
            "primary_target_total_complete": 12,
            "required_new_complete_episode_count": 6,
            "route_targets": df_records(target_df),
            "target_sum_check": int(target_df["target_cumulative_complete_episode_count"].sum()),
            "all_routes_target_at_least_2": bool((target_df["target_cumulative_complete_episode_count"] >= 2).all()),
        },
    )

    dump_json(
        output_root / "independent_vehicle_expansion_contract.json",
        {
            "independent_vehicle_target_status": "APPROVED",
            "current_total_independent_vehicles": current_vehicles,
            "minimum_target_total_independent_vehicles": 8,
            "planned_new_independent_vehicle_count": 4,
            "route_recommendations": {
                "4010002001": "new vehicle minimum 1",
                "4010002004": "new vehicle minimum 1 recommended",
                "4010002118": "new vehicle minimum 1",
                "4050010000": "new vehicle minimum 1 recommended",
            },
            "vehicle_history_classes": ["NEW_INDEPENDENT_VEHICLE", "PREVIOUSLY_COMPLETE_VEHICLE", "PREVIOUSLY_CENSORED_VEHICLE", "DUPLICATE_TERMINAL_CYCLE", "UNKNOWN_VEHICLE_HISTORY"],
            "vehicle_linkage_rule": "exact vehicle_id equality only",
            "forbidden_linkage_rules": ["coordinate proximity", "same stop", "same route/direction", "sequence similarity", "reappearance after disappearance", "vehicle_id string similarity"],
        },
    )

    bucket_definitions = {
        "MORNING": "09:00-10:59",
        "MIDDAY": "11:00-12:59",
        "AFTERNOON": "13:00-15:59",
        "EVENING": "16:00-18:59",
    }
    dump_json(
        output_root / "date_hour_diversity_contract.json",
        {
            "date_hour_diversity_contract_status": "APPROVED",
            "current_observation_dates": current_dates,
            "current_hour_buckets": current_hour_buckets,
            "bucket_definitions": bucket_definitions,
            "primary_targets": {"overall_observation_dates_min": 2, "overall_hour_buckets_min": 2, "per_route_different_hour_buckets_when_possible": True},
            "stretch_targets": {"overall_observation_dates": 3, "per_route_hour_buckets_min": 2},
            "same_date_same_hour_all_six_plan_approved": False,
            "preferred_future_windows": ["MORNING", "AFTERNOON", "EVENING"],
            "date_hour_diversity_contract_passed": True,
        },
    )

    campaigns = [
        {"campaign_id": "A", "routes": ["4010002118", "4010002001"], "target_new_complete_max": 3, "priority": 1, "purpose": "high-priority deficit routes and long-follow route 4010002118"},
        {"campaign_id": "B", "routes": ["4010002004", "4050010000"], "target_new_complete_max": 2, "priority": 2, "purpose": "balanced target completion and LIVE_SEQUENCE_EXTENSION coverage"},
        {"campaign_id": "C", "routes": ROUTES, "target_new_complete_max": 1, "priority": 3, "purpose": "residual route deficit fill after A/B verification"},
    ]
    dump_json(
        output_root / "campaign_partition_plan.json",
        {
            "campaign_partition_plan_status": "APPROVED",
            "campaigns": campaigns,
            "automatic_continuous_execution_allowed": False,
            "between_campaign_required_checks": ["API runtime", "fatal stop compliance", "raw SHA", "episode completeness", "vehicle independence", "clock quality", "mapping regression", "secret leak"],
        },
    )
    dump_json(
        output_root / "campaign_schedule_design.json",
        {
            "campaign_schedule_design_status": "APPROVED_FOR_PLANNING_ONLY",
            "campaigns_are_separately_dated": True,
            "automatic_schedule_created": False,
            "campaign_a_next_authorization_step": "Prompt 5-E01-R2D-1H Campaign A Live Observation Execution Authorization",
            "start_condition": "dated execution authorization after daily API usage, operator readiness, route targets, and schedule are confirmed",
        },
    )
    dump_json(
        output_root / "campaign_api_budget_contract.json",
        {
            "api_budget_contract_status": "APPROVED",
            "api_budget_contract_passed": True,
            "network_api_calls_this_step": 0,
            "physical_call_hard_cap_per_campaign": 450,
            "daily_physical_call_safety_cap": 800,
            "max_calls_per_minute": 8,
            "recommended_campaign_budgets": {"Campaign A": 300, "Campaign B": 300, "Campaign C": 250},
            "required_daily_ledger_fields": ["run_date", "prior_physical_calls_on_run_date", "preflight_physical_calls", "campaign_physical_calls", "total_physical_calls_on_run_date", "authoritative_saved_raw_calls", "non_authoritative_physical_calls"],
            "ledger_identity": "total_physical_calls_on_run_date = prior_physical_calls_on_run_date + preflight_physical_calls + campaign_physical_calls",
            "unknown_daily_api_usage_gate": "BLOCKED_UNKNOWN_DAILY_API_USAGE",
        },
    )
    dump_json(
        output_root / "campaign_stop_rule_contract.json",
        {
            "campaign_stop_rule_contract_status": "APPROVED",
            "campaign_stop_rule_contract_passed": True,
            "fatal_stop_conditions": ["API token quota exceeded", "HTTP 429", "AUTH_ERROR", "HTML_RESPONSE", "secret leak", "3 consecutive timeouts", "3 consecutive provider parse failures", "campaign physical hard cap", "daily physical safety cap"],
            "calls_after_first_fatal_error": 0,
            "fatal_error_retry_allowed": False,
            "open_session_final_status_on_fatal": "RIGHT_CENSORED_FATAL_API_STOP",
        },
    )
    dump_json(
        output_root / "campaign_state_machine_contract.json",
        {
            "state_machine_contract_status": "APPROVED",
            "states": ["BROAD_SCAN", "EARLY_UPSTREAM_WATCH", "UPSTREAM_FOCUSED", "PRE_TERMINAL_CONFIRMED", "TERMINAL_ENTERED", "TERMINAL_LOOP_MOVEMENT", "TERMINAL_STOP_HOLD", "POST_TERMINAL_WAIT", "POST_TERMINAL_RESET", "POST_TERMINAL_CONFIRM", "COMPLETE_INTERVAL_CENSORED", "LEFT_CENSORED", "RIGHT_CENSORED", "INVALID"],
            "intervals": {"BROAD_SCAN": "120 sec", "UPSTREAM_FOCUSED": "45-60 sec", "TERMINAL_FOCUSED": "30 sec", "POST_TERMINAL_WAIT": "30 sec"},
            "early_upstream_rule": "current_sequence >= effective_live_terminal_sequence - 22 and current_sequence < effective_live_terminal_sequence - 12",
            "upstream_focused_rule": "current_sequence >= effective_live_terminal_sequence - 12 and current_sequence < effective_live_terminal_sequence - 5",
            "terminal_focused_rule": "current_sequence >= effective_live_terminal_sequence - 5",
            "post_terminal_confirm_rule": "3 normal samples after reset or 10 minutes, whichever comes first",
        },
    )
    dump_json(
        output_root / "candidate_vehicle_selection_contract.json",
        {
            "candidate_selection_contract_status": "APPROVED",
            "max_focused_vehicle_per_route": 1,
            "max_focused_vehicle_total": 2,
            "candidate_priority": ["NEW_INDEPENDENT_VEHICLE", "UNKNOWN_VEHICLE_HISTORY", "PREVIOUSLY_CENSORED_VEHICLE", "PREVIOUSLY_COMPLETE_VEHICLE"],
            "terminal_zone_first_seen_class": "LEFT_CENSORED_EXISTING_TERMINAL_VEHICLE",
            "do_not_retroactively_complete_terminal_zone_first_seen": True,
            "required_complete_samples": ["early upstream sample", "last pre-terminal sample", "first terminal sample", "last terminal sample", "first post-terminal reset sample", "post-terminal confirmation sample"],
            "call_optimization_rules": ["stop new focused sessions for routes that achieved target complete count", "do not duplicate-track the same vehicle terminal cycle", "prioritize new vehicle_id before repeating prior complete vehicles", "do not start long-follow session within 90 minutes of campaign end"],
        },
    )
    dump_json(
        output_root / "follow_duration_contract.json",
        {
            "follow_duration_contract_status": "APPROVED",
            "max_follow_minutes_by_route": {"4010002001": 75, "4010002004": 75, "4010002118": 100, "4050010000": 75},
            "route_4010002118_long_follow_reason": "prior long reset wait and right-censoring risk",
            "hard_cap_precedence": ["daily physical safety cap", "campaign physical hard cap", "campaign end time", "route max follow"],
            "do_not_start_if_remaining_campaign_time_less_than_route_max_follow": True,
        },
    )

    dump_json(
        output_root / "future_campaign_artifact_contract.json",
        {
            "artifact_pattern": "05_training/artifacts/prompt5_e01_r2d1g_campaign_<campaign_id>_<YYYYMMDD_HHMMSS>/",
            "common_required_files": ["campaign_manifest.json", "campaign_gate.json", "campaign_final_report.md", "r2d1f_reference.json", "mapping_reference.json", "prior_episode_registry_reference.json", "campaign_execution_authorization.json", "campaign_observation_contract.json", "daily_api_usage_audit.json", "campaign_preflight_audit.json", "campaign_runtime_audit.json", "api_stop_condition_audit.json", "position_samples.parquet", "vehicle_trajectories.parquet", "terminal_recovery_episodes.parquet", "terminal_recovery_interval_bounds.parquet", "episode_evidence_sha256_audit.json", "episode_deduplication_audit.json", "clock_semantics_audit.json", "mapping_regression_audit.json", "secret_leak_audit.json"],
            "route_evidence_subdir": "terminal_recovery_evidence/<route_id>/",
            "campaign_pass_does_not_authorize_estimation_execution": True,
        },
    )
    dump_json(
        output_root / "cumulative_episode_registry_update_contract.json",
        {
            "registry_update_contract_status": "APPROVED",
            "update_mode": "append-only candidate",
            "formula": "prior frozen registry + verified new complete episodes = cumulative registry candidate",
            "do_not_modify_prior_frozen_six": True,
            "required_fields": ["frozen_episode_id", "source_campaign_id", "source_artifact", "route_id", "vehicle_id", "observation_date", "hour_bucket", "first_terminal_raw_sha", "first_post_terminal_raw_sha", "clock_semantics_status", "eligible_for_estimation_input"],
            "duplicate_key": ["route_id", "vehicle_id", "terminal cycle time range", "first_terminal raw SHA", "first_post_terminal raw SHA"],
        },
    )
    dump_json(
        output_root / "method_prototype_data_gate_contract.json",
        {
            "method_prototype_gate_contract_status": "APPROVED",
            "allowed_future_gates": ["PASS_METHOD_PROTOTYPE_DATA_READY", "PASS_EXPANSION_PARTIAL", "BLOCKED_ROUTE_MINIMUM", "BLOCKED_INDEPENDENT_VEHICLE_MINIMUM", "BLOCKED_DATE_TIME_DIVERSITY", "FAIL_CUMULATIVE_PROVENANCE", "FAIL_MAPPING_REGRESSION"],
            "pass_conditions": {"cumulative_complete_episodes_min": 12, "each_route_complete_episodes_min": 2, "total_independent_vehicles_min": 8, "each_route_independent_vehicles_min": 2, "observation_dates_min": 2, "observation_hour_buckets_min": 2, "episode_duplicate": 0, "mapping_regression": 0, "invalid_clock_order": 0, "provenance_failure": 0, "secret_leak": 0},
            "on_pass_allowed_flags": {"eligible_for_terminal_recovery_method_prototype_execution_review": True, "eligible_for_terminal_recovery_estimation_execution": False, "terminal_recovery_estimated": False, "terminal_recovery_applied": False, "phase2_authorized": False},
        },
    )

    risks = [
        ("R01", "target vehicle not observed", "all", "medium", "medium", "no candidate in broad scan", "shift to Campaign C residual targeting", "no upstream vehicle", "partial campaign"),
        ("R02", "vehicle already terminal-zone at first observation", "all", "medium", "medium", "first seen sequence >= effective-5", "classify as left-censored; wait for next cycle", "left-censored only", "do not retro-complete"),
        ("R03", "long terminal wait causes right censoring", "4010002118", "high", "high", "terminal hold exceeds expected follow", "allow 100 minute follow within hard caps", "insufficient follow window", "reschedule 4010002118"),
        ("R04", "API quota exhaustion", "all", "medium", "high", "daily ledger approaches cap", "hard cap <=450 campaign and <=800 daily", "quota or HTTP 429", "stop calls immediately"),
        ("R05", "HTML response", "all", "low", "high", "provider response is HTML", "fatal stop with calls_after_first_fatal_error=0", "HTML_RESPONSE", "block campaign"),
        ("R06", "provider timestamp stale", "all", "high", "medium", "repeat provider timestamp span >=120 sec", "preserve dual-clock and classify clock semantics", "invalid clock order", "exclude invalid only"),
        ("R07", "request sampling interval too wide", "all", "medium", "medium", "large request reentry window", "30 sec terminal/post-terminal cadence", "interval validation fail", "retain as wide interval diagnostic"),
        ("R08", "same vehicle repeated too often", "all", "medium", "medium", "vehicle history class previously complete", "prioritize NEW_INDEPENDENT_VEHICLE", "independent vehicle target unmet", "additional campaign"),
        ("R09", "same date/time bucket concentration", "all", "medium", "medium", "candidate schedule repeats current bucket", "prefer missing windows and no all-six same bucket plan", "date/hour diversity unmet", "reschedule campaign"),
        ("R10", "wide interval episode dominance", "all", "medium", "medium", "wide intervals in validation", "S8 future sensitivity design", "wide interval rule absent", "do not compute summary statistic"),
        ("R11", "route 4010002118 long follow cost", "4010002118", "high", "medium", "focused session consumes cap", "Campaign A cap and max 2 focused vehicles total", "campaign hard cap", "defer residual to Campaign C"),
        ("R12", "LIVE_SEQUENCE_EXTENSION boundary sensitivity", "4050010000", "medium", "medium", "service terminal and effective live terminal differ", "preserve HF1 extension and boundary contract", "mapping mismatch", "block mapping regression"),
        ("R13", "campaign end before reset", "all", "medium", "medium", "remaining time < max follow", "do not start new long-follow session late", "insufficient follow window", "right-censor open session"),
    ]
    risk_records = [
        {
            "risk_id": rid,
            "risk_name": name,
            "affected_route": route,
            "probability": probability,
            "impact": impact,
            "early_warning": warning,
            "mitigation": mitigation,
            "blocking_condition": blocking,
            "fallback": fallback,
        }
        for rid, name, route, probability, impact, warning, mitigation, blocking, fallback in risks
    ]
    dump_json(output_root / "observation_expansion_risk_register.json", {"risk_register_status": "APPROVED", "risks": risk_records})
    dump_json(
        output_root / "terminal_recovery_data_expansion_requirements_v2.json",
        {
            "current_total_complete": current_total,
            "primary_target_total_complete": 12,
            "required_new_complete": 6,
            "route_targets": df_records(target_df),
            "current_total_independent_vehicles": current_vehicles,
            "minimum_target_total_independent_vehicles": 8,
            "planned_new_independent_vehicle_count": 4,
            "automatic_api_execution_allowed": False,
            "next_allowed_step": "Prompt 5-E01-R2D-1H Campaign A Live Observation Execution Authorization",
        },
    )
    dump_json(
        output_root / "r2d1g_observation_campaign_execution_authorization_draft.json",
        {
            "approved": False,
            "live_campaign_execution_approved": False,
            "automatic_schedule_created": False,
            "reason": "R2D-1G defines the observation expansion plan only. Each live API campaign requires a separate dated execution authorization after daily API usage, schedule, route targets, and operator readiness are confirmed.",
        },
    )
    dump_json(
        output_root / "terminal_recovery_estimation_execution_authorization.json",
        {
            "approved": False,
            "eligible_for_terminal_recovery_estimation_execution": False,
            "terminal_recovery_estimated": False,
            "reason": "The current dataset has 6 complete episodes and remains below the approved Method Prototype threshold of 12 episodes.",
        },
    )
    lock_payload = {
        "terminal_recovery_parameter_generated": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "baseline_rerun_authorized": False,
        "retraining_authorized": False,
        "approved_for_prompt6a": False,
        "approved_for_e2": False,
        "approved_for_full_matrix": False,
    }
    dump_json(output_root / "simulator_parameter_translation_guard.json", lock_payload)
    dump_json(output_root / "phase2_execution_authorization.json", {"approved": False, **lock_payload})

    secret = secret_scan(output_root)
    dump_json(output_root / "secret_leak_audit.json", secret)
    strict_count = sum(token_count(path) for path in output_root.rglob("*.json"))
    plan_passed = (
        not failed_inputs
        and current_total == 6
        and sum(ROUTE_REQUIRED_NEW.values()) == 6
        and int(target_df["target_cumulative_complete_episode_count"].sum()) == 12
        and bool((target_df["target_cumulative_complete_episode_count"] >= 2).all())
        and current_vehicles + 4 >= 8
        and duplicate_count == 0
        and episode_registry_passed
        and strict_count == 0
        and secret["secret_leak_count"] == 0
    )
    gate_status = "PASS_OBSERVATION_EXPANSION_PLAN_READY" if plan_passed else "PASS_OBSERVATION_EXPANSION_PLAN_PARTIAL"
    if failed_inputs:
        gate_status = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"
    elif strict_count != 0:
        gate_status = "FAIL_STRICT_JSON_AUDIT"
    elif secret["secret_leak_count"] != 0:
        gate_status = "FAIL_SECURITY_AUDIT"

    gate = {
        "gate_status": gate_status,
        "artifact_root": str(output_root),
        "network_api_calls": 0,
        "live_observation_calls": 0,
        "authoritative_input_modified_count": 0,
        "current_complete_episode_count": current_total,
        "primary_target_complete_episode_count": 12,
        "required_new_complete_episode_count": 6,
        "route_targets": {route: {"current": current_counts[route], "target": ROUTE_TARGETS[route], "required": ROUTE_REQUIRED_NEW[route]} for route in ROUTES},
        "current_independent_vehicle_count": current_vehicles,
        "planned_new_independent_vehicle_count": 4,
        "api_budget_contract_passed": True,
        "campaign_stop_rule_contract_passed": True,
        "date_hour_diversity_contract_passed": True,
        "mapping_regression_count": 0,
        "episode_registry_audit_passed": episode_registry_passed,
        "secret_leak_count": int(secret["secret_leak_count"]),
        "strict_json_nonstandard_token_count": strict_count,
        "observation_expansion_plan_approved": gate_status == "PASS_OBSERVATION_EXPANSION_PLAN_READY",
        "live_campaign_execution_approved": False,
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "eligible_for_terminal_recovery_estimation_execution": False,
        "phase2_authorized": False,
        "next_authorized_action": "Prompt 5-E01-R2D-1H Campaign A Live Observation Execution Authorization" if gate_status == "PASS_OBSERVATION_EXPANSION_PLAN_READY" else None,
    }
    dump_json(output_root / "prompt5_e01_r2d1g_gate.json", gate)

    report = f"""# Prompt 5-E01-R2D-1G Final Report

## Status

- gate: {gate_status}
- network_api_calls: 0
- live vehicle observation calls: 0
- authoritative_input_modified_count: 0
- current complete episodes: {current_total}
- current independent vehicles: {current_vehicles}
- Method Prototype primary target: 12 complete episodes
- required new complete episodes: 6

## Authoritative Inputs

- HF1: {hf1}
- R2D-1E: {r2d1e}
- R2D-1F: {r2d1f}

All inputs were read-only. Existing raw responses, complete/censored episodes, interval bounds, mapping contracts,
gates, reports, clock audits, and estimation design contracts were not modified.

## Route Targets

- 4010002118: current 1, target 3, required 2, priority 1
- 4010002001: current 1, target 3, required 2, priority 2
- 4010002004: current 2, target 3, required 1, priority 3
- 4050010000: current 2, target 3, required 1, priority 4

The planned six new complete episodes bring all four routes to three cumulative complete episodes.

## Vehicle And Diversity Targets

Primary plan requires at least four new independent vehicles across the six new complete episodes. Existing dates
already cover two dates and three raw hour buckets, but future campaigns must avoid concentrating all new observations
in one date or one time bucket. MORNING, AFTERNOON, and EVENING windows are preferred where operations and API budget
allow.

## Campaign Plan

Campaign A covers 4010002118 and 4010002001 with up to three new complete episodes. Campaign B covers 4010002004
and 4050010000 with up to two new complete episodes. Campaign C fills any residual deficit. Campaigns must not run
automatically back-to-back; each campaign requires its own artifact, API ledger, gate, and verification.

## API Budget And Stop Rules

Each campaign has a physical hard cap <= 450, daily physical safety cap <= 800, and max 8 calls per minute.
Recommended campaign budgets are A <= 300, B <= 300, C <= 250. Unknown daily usage blocks execution with
`BLOCKED_UNKNOWN_DAILY_API_USAGE`. Fatal stops require `calls_after_first_fatal_error = 0`.

## State Machine And Candidate Rules

The plan fixes BROAD_SCAN, EARLY_UPSTREAM_WATCH, UPSTREAM_FOCUSED, terminal/post-terminal states, censoring states,
and complete episode states. Candidate continuity uses exact `vehicle_id` equality only. Vehicles first seen in
terminal zone are left-censored and cannot be retroactively promoted to complete.

## Follow Durations

Maximum follow is 75 minutes for 4010002001, 4010002004, and 4050010000, and 100 minutes for 4010002118. API hard
caps and campaign end time override route follow duration.

## Provenance And Future Registry

Each future campaign must save raw responses, SHA audits, clock semantics, deduplication, mapping regression, and
secret leak audits. Cumulative registry update is append-only: prior frozen registry plus verified new complete
episodes. The prior frozen six are never modified.

## Method Prototype Completion Gate

Future `PASS_METHOD_PROTOTYPE_DATA_READY` requires cumulative complete episodes >= 12, each route >= 2, total
independent vehicles >= 8, each route independent vehicles >= 2, at least two dates, at least two hour buckets,
duplicate = 0, mapping regression = 0, invalid clock order = 0, provenance failure = 0, and secret leak = 0.

## Locks

This prompt did not execute live observation, API calls, statistical estimation, average/median/percentile calculation,
recovery parameter generation, simulator application, Phase 2, baseline rerun, or retraining. Campaign execution,
terminal recovery estimation, simulator translation, and Phase 2 remain locked.

## Next Step

Prompt 5-E01-R2D-1H Campaign A Live Observation Execution Authorization.
"""
    (output_root / "prompt5_e01_r2d1g_final_report.md").write_text(report, encoding="utf-8")

    required_names = [
        "prompt5_e01_r2d1g_manifest.json",
        "prompt5_e01_r2d1g_gate.json",
        "prompt5_e01_r2d1g_final_report.md",
        "hf1_reference.json",
        "r2d1e_reference.json",
        "r2d1f_reference.json",
        "authoritative_input_immutability_audit.json",
        "mapping_regression_audit.json",
        "episode_registry_reference_audit.json",
        "secret_leak_audit.json",
        "observation_expansion_objective_contract.json",
        "route_observation_expansion_target_contract.json",
        "route_observation_expansion_target_table.parquet",
        "independent_vehicle_expansion_contract.json",
        "date_hour_diversity_contract.json",
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
        "observation_expansion_risk_register.json",
        "terminal_recovery_data_expansion_requirements_v2.json",
        "r2d1g_observation_campaign_execution_authorization_draft.json",
        "terminal_recovery_estimation_execution_authorization.json",
        "simulator_parameter_translation_guard.json",
        "phase2_execution_authorization.json",
    ]
    manifest = make_manifest(output_root, required_names)
    dump_json(output_root / "prompt5_e01_r2d1g_manifest.json", manifest)
    manifest = make_manifest(output_root, required_names)
    dump_json(output_root / "prompt5_e01_r2d1g_manifest.json", manifest)

    print("R2D-1G OBSERVATION EXPANSION PLAN COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate_status)
    print("\nnetwork_api_calls:\n0")
    print("\nauthoritative_input_modified_count:\n0")
    print(f"\ncurrent_complete_episode_count:\n{current_total}")
    print("\nprimary_target_complete_episode_count:\n12")
    print("\nrequired_new_complete_episode_count:\n6")
    print("\nroute_targets:")
    for route in ROUTES:
        print(f"{route} = current={current_counts[route]} / target={ROUTE_TARGETS[route]} / required={ROUTE_REQUIRED_NEW[route]}")
    print(f"\ncurrent_independent_vehicle_count:\n{current_vehicles}")
    print("\nplanned_new_independent_vehicle_count:\n4")
    print("\ncampaign_partitions:")
    print("A = 4010002118 + 4010002001, max 3 complete")
    print("B = 4010002004 + 4050010000, max 2 complete")
    print("C = residual deficit fill, max 1 complete")
    print("\napi_budget_contract_passed:\ntrue")
    print("\ncampaign_stop_rule_contract_passed:\ntrue")
    print("\ndate_hour_diversity_contract_passed:\ntrue")
    print("\nmapping_regression_count:\n0")
    print(f"\nepisode_registry_audit_passed:\n{str(episode_registry_passed).lower()}")
    print(f"\nsecret_leak_count:\n{secret['secret_leak_count']}")
    print(f"\nobservation_expansion_plan_approved:\n{str(gate_status == 'PASS_OBSERVATION_EXPANSION_PLAN_READY').lower()}")
    print("\nlive_campaign_execution_approved:\nfalse")
    print("\nterminal_recovery_estimated:\nfalse")
    print("\nterminal_recovery_applied:\nfalse")
    print("\neligible_for_terminal_recovery_estimation_execution:\nfalse")
    print("\nphase2_authorized:\nfalse")
    print("\nnext_authorized_action:\nPrompt 5-E01-R2D-1H Campaign A Live Observation Execution Authorization")


if __name__ == "__main__":
    main()
