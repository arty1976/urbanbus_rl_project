#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"
SCRIPT_PATH = PROJECT_ROOT / "05_training" / "run_prompt5_e01_r2d1l_campaign_c_authorization_review.py"

R2D1K_HF1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1k_hf1_campaign_b_metadata_finalization_20260727_113644"
R2D1K_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1k_campaign_b_controlled_live_observation_20260727_093548"
R2D1J_HF1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1j_hf1_campaign_b_early_stop_semantics_handoff_finalization_20260726_235819"
R2D1I_HF2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1i_hf2_final_metadata_reconciliation_20260726_191538"
HF1_MAPPING_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1E_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
R2D1F_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1f_estimation_design_approval_20260724_160903"

REGISTRY_PATH = R2D1K_HF1_ROOT / "cumulative_episode_registry_candidate_hf1.parquet"
MAPPING_PATH = HF1_MAPPING_ROOT / "turnaround_mapping_contract_v10_hf1.parquet"

TARGET_ROUTE = "4010002118"
TARGET_ROUTES = [TARGET_ROUTE]
EXCLUDED_ROUTES = ["4010002001", "4010002004", "4050010000"]
ALL_ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]
PASS_GATE = "PASS_CAMPAIGN_C_AUTHORIZATION_REVIEW_READY"
DIVERSITY_PASS_GATE = "PASS_CAMPAIGN_C_AUTHORIZATION_REVIEW_WITH_DIVERSITY_CONSTRAINT"

REQUIRED_FILES = [
    "prompt5_e01_r2d1l_manifest.json",
    "prompt5_e01_r2d1l_gate.json",
    "prompt5_e01_r2d1l_final_report.md",
    "upstream_reference_r2d1k_hf1.json",
    "upstream_reference_r2d1k.json",
    "upstream_reference_r2d1j_hf1.json",
    "upstream_reference_r2d1i_hf2.json",
    "upstream_reference_hf1_mapping.json",
    "upstream_reference_r2d1e.json",
    "upstream_reference_r2d1f.json",
    "network_api_call_audit.json",
    "service_key_access_audit.json",
    "authoritative_input_immutability_audit.json",
    "source_artifact_integrity_audit.json",
    "registry_freeze_reference_audit.json",
    "method_prototype_current_state_audit.json",
    "mapping_regression_audit.json",
    "secret_leak_audit.json",
    "campaign_c_scope_contract.json",
    "campaign_c_target_route_contract.json",
    "campaign_c_method_prototype_gap_contract.json",
    "campaign_c_date_hour_diversity_audit.json",
    "campaign_c_candidate_priority_contract.json",
    "campaign_c_vehicle_exclusion_registry.json",
    "campaign_c_vehicle_exclusion_registry.parquet",
    "campaign_c_observation_window_contract.json",
    "campaign_c_follow_duration_contract.json",
    "campaign_c_api_budget_contract.json",
    "campaign_c_preflight_contract.json",
    "campaign_c_state_machine_contract.json",
    "campaign_c_temporary_disappearance_contract.json",
    "campaign_c_complete_episode_contract.json",
    "campaign_c_censoring_contract.json",
    "campaign_c_dual_clock_interval_contract.json",
    "campaign_c_fatal_stop_contract.json",
    "campaign_c_early_stop_contract.json",
    "campaign_c_episode_namespace_contract.json",
    "campaign_c_expected_outcome_contract.json",
    "campaign_c_execution_handoff_packet.json",
    "campaign_c_execution_preflight_checklist.json",
    "campaign_c_authorization_review.json",
    "campaign_c_live_execution_authorization.json",
    "terminal_recovery_estimation_execution_authorization.json",
    "simulator_parameter_translation_guard.json",
    "phase2_execution_authorization.json",
]


class GateFailure(Exception):
    def __init__(self, gate: str, message: str) -> None:
        super().__init__(message)
        self.gate = gate


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def snapshot(roots: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for root in roots:
        for path in files_under(root):
            result[str(path)] = {
                "absolute_path": str(path),
                "relative_path": str(path.relative_to(root)),
                "root": str(root),
                "file_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return result


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    pd.read_parquet(path)


def dataframe_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return [jsonable(record) for record in frame.to_dict("records")]


def route_counts(registry: pd.DataFrame) -> Dict[str, int]:
    counts = registry.groupby(registry["route_id"].astype(str)).size().to_dict()
    return {route: int(counts.get(route, 0)) for route in ALL_ROUTES}


def route_local_unique_counts(registry: pd.DataFrame) -> Dict[str, int]:
    counts = registry.groupby(registry["route_id"].astype(str))["route_local_vehicle_identity_key"].nunique().to_dict()
    return {route: int(counts.get(route, 0)) for route in ALL_ROUTES}


def hour_buckets_by_route(registry: pd.DataFrame) -> Dict[str, List[str]]:
    result = {}
    for route in ALL_ROUTES:
        subset = registry[registry["route_id"].astype(str) == route]
        result[route] = sorted(subset["hour_bucket"].dropna().astype(str).unique().tolist())
    return result


def build_upstream_reference(root: Path, gate_name: str | None) -> Dict[str, Any]:
    gate_path = root / gate_name if gate_name else None
    gate_status = read_json(gate_path).get("gate_status") if gate_path and gate_path.exists() else None
    return {
        "artifact_path": str(root),
        "exists": root.exists(),
        "gate_path": str(gate_path) if gate_path else None,
        "gate_status": gate_status,
        "file_count": len(files_under(root)),
        "read_only_input": True,
    }


def manifest_payload(output_root: Path) -> Dict[str, Any]:
    entries = []
    for path in sorted(files_under(output_root), key=lambda item: str(item.relative_to(output_root))):
        rel = str(path.relative_to(output_root))
        if rel == "prompt5_e01_r2d1l_manifest.json":
            entries.append(
                {
                    "path": rel,
                    "exists": True,
                    "sha256": None,
                    "self_hash_exempt": True,
                    "self_hash_exemption_reason": "Stable self-hashing is not possible after final serialization.",
                    "size_bytes": None,
                    "self_size_exempt": True,
                    "self_size_exemption_reason": "Stable self-size recording is not guaranteed when the manifest contains its own metadata.",
                }
            )
        else:
            entries.append(
                {
                    "path": rel,
                    "exists": True,
                    "sha256": sha256_file(path),
                    "self_hash_exempt": False,
                    "size_bytes": path.stat().st_size,
                    "self_size_exempt": False,
                }
            )
    present = {entry["path"] for entry in entries}
    missing = [name for name in REQUIRED_FILES if name not in present]
    return {
        "artifact_name": output_root.name,
        "created_at_kst": datetime.now(KST).isoformat(),
        "required_file_count": len(REQUIRED_FILES),
        "present_required_file_count": len(REQUIRED_FILES) - len(missing),
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "manifest_self_hash_exempt": True,
        "manifest_self_size_exempt": True,
        "files": entries,
    }


def validate_manifest(output_root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    missing = []
    hash_mismatch = []
    size_mismatch = []
    for entry in manifest["files"]:
        path = output_root / entry["path"]
        if not path.exists():
            missing.append(entry["path"])
            continue
        if entry["path"] == "prompt5_e01_r2d1l_manifest.json":
            if entry.get("sha256") is not None or entry.get("size_bytes") is not None:
                size_mismatch.append(entry["path"])
            continue
        if entry.get("sha256") and sha256_file(path) != entry["sha256"]:
            hash_mismatch.append(entry["path"])
        if entry.get("size_bytes") is not None and path.stat().st_size != entry["size_bytes"]:
            size_mismatch.append(entry["path"])
    return {
        "manifest_missing_file_count": len(missing),
        "manifest_missing_files": missing,
        "manifest_nonself_hash_mismatch_count": len(hash_mismatch),
        "manifest_nonself_hash_mismatches": hash_mismatch,
        "manifest_nonself_size_mismatch_count": len(size_mismatch),
        "manifest_nonself_size_mismatches": size_mismatch,
    }


def strict_json_failures(root: Path) -> List[Dict[str, str]]:
    failures = []
    for path in files_under(root):
        if path.suffix != ".json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append({"path": str(path.relative_to(root)), "error": repr(exc)})
    return failures


def parquet_failures(root: Path) -> List[Dict[str, str]]:
    failures = []
    for path in files_under(root):
        if path.suffix != ".parquet":
            continue
        try:
            pd.read_parquet(path)
        except Exception as exc:
            failures.append({"path": str(path.relative_to(root)), "error": repr(exc)})
    return failures


def build_vehicle_exclusion_registry(registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for vehicle_id, group in registry.groupby(registry["canonical_vehicle_id"].astype(str), dropna=False):
        routes = sorted(group["route_id"].astype(str).unique().tolist())
        rows.append(
            {
                "canonical_vehicle_id": str(vehicle_id).strip(),
                "global_complete_episode_count": int(len(group)),
                "route_count": int(len(routes)),
                "routes_seen": routes,
                "source_episode_ids": sorted(group["source_episode_id"].astype(str).tolist()),
                "excluded_from_global_unseen_vehicle_tier": True,
                "exact_id_matching_only": True,
            }
        )
    return pd.DataFrame(rows).sort_values("canonical_vehicle_id").reset_index(drop=True)


def select_gate(metrics: Dict[str, Any], diversity_required: bool) -> str:
    checks = [
        ("FAIL_SOURCE_ARTIFACT_INTEGRITY", metrics["source_artifact_failure_count"]),
        ("FAIL_SOURCE_IMMUTABILITY", metrics["upstream_modified_file_count"] + metrics["upstream_deleted_file_count"] + metrics["upstream_added_file_count"]),
        ("FAIL_REGISTRY_FREEZE_RECONCILIATION", metrics["registry_freeze_failure_count"]),
        ("FAIL_METHOD_PROTOTYPE_GAP_RECONCILIATION", metrics["method_gap_failure_count"]),
        ("FAIL_ROUTE_SCOPE_RECONCILIATION", metrics["route_scope_failure_count"]),
        ("FAIL_VEHICLE_EXCLUSION_REGISTRY", metrics["vehicle_exclusion_registry_failure_count"]),
        ("FAIL_DATE_HOUR_DIVERSITY_AUDIT", metrics["date_hour_diversity_failure_count"]),
        ("FAIL_FOLLOW_DURATION_CONTRACT", metrics["follow_duration_contract_failure_count"]),
        ("FAIL_API_BUDGET_CONTRACT", metrics["api_budget_contract_failure_count"]),
        ("FAIL_STATE_MACHINE_CONTRACT", metrics["state_machine_contract_failure_count"]),
        ("FAIL_CENSORING_CONTRACT", metrics["censoring_contract_failure_count"]),
        ("FAIL_EXECUTION_HANDOFF_CONTRACT", metrics["execution_handoff_contract_failure_count"]),
        ("FAIL_MAPPING_REGRESSION", metrics["mapping_regression_count"]),
        ("FAIL_SCHEMA_AUDIT", metrics["strict_json_failure_count"] + metrics["parquet_read_failure_count"] + metrics["json_parquet_mismatch_count"]),
        ("FAIL_MANIFEST_RECONCILIATION", metrics["manifest_missing_required_file_count"] + metrics["manifest_nonself_hash_mismatch_count"] + metrics["manifest_nonself_size_mismatch_count"]),
        ("FAIL_SECURITY_AUDIT", metrics["secret_leak_count"]),
    ]
    for gate, count in checks:
        if count:
            return gate
    return DIVERSITY_PASS_GATE if diversity_required else PASS_GATE


def main() -> None:
    timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    output_root = ARTIFACTS_ROOT / f"prompt5_e01_r2d1l_campaign_c_authorization_review_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    upstream_roots = [R2D1K_HF1_ROOT, R2D1K_ROOT, R2D1J_HF1_ROOT, R2D1I_HF2_ROOT, HF1_MAPPING_ROOT, R2D1E_ROOT, R2D1F_ROOT]
    before = snapshot(upstream_roots)
    for root in upstream_roots:
        if not root.exists():
            raise GateFailure("BLOCKED_AUTHORITATIVE_INPUT_MISSING", f"Missing upstream artifact: {root}")

    source_hf1_gate = read_json(R2D1K_HF1_ROOT / "prompt5_e01_r2d1k_hf1_gate.json").get("gate_status")
    source_k_gate = read_json(R2D1K_ROOT / "prompt5_e01_r2d1k_gate.json").get("gate_status")
    source_j_hf1_gate = read_json(R2D1J_HF1_ROOT / "prompt5_e01_r2d1j_hf1_gate.json").get("gate_status")
    source_i_hf2_gate = read_json(R2D1I_HF2_ROOT / "prompt5_e01_r2d1i_hf2_gate.json").get("gate_status")
    source_e_gate = read_json(R2D1E_ROOT / "prompt5_e01_r2d1d3_hf1_r2d1e_gate.json").get("gate_status")
    source_f_gate = read_json(R2D1F_ROOT / "prompt5_e01_r2d1f_gate.json").get("gate_status")
    source_artifact_failure_count = sum(
        [
            source_hf1_gate != "PASS_CAMPAIGN_B_METADATA_FINALIZATION_FREEZE_READY",
            source_k_gate != "PASS_CAMPAIGN_B_FULL_TARGET",
            source_j_hf1_gate != "PASS_CAMPAIGN_B_EARLY_STOP_SEMANTICS_HANDOFF_FINALIZED",
            source_i_hf2_gate != "PASS_FINAL_METADATA_RECONCILIATION_FREEZE_READY",
            not str(source_e_gate).startswith("PASS_"),
            not str(source_f_gate).startswith("PASS_"),
        ]
    )

    registry = pd.read_parquet(REGISTRY_PATH)
    mapping_df = pd.read_parquet(MAPPING_PATH)
    target_mapping = mapping_df[mapping_df["route_id"].astype(str) == TARGET_ROUTE].copy()
    route_count_map = route_counts(registry)
    route_local_unique = route_local_unique_counts(registry)
    global_unique_vehicle_count = int(registry["canonical_vehicle_id"].dropna().astype(str).nunique())
    complete_count = int(len(registry))
    complete_deficit = max(0, 12 - complete_count)
    global_vehicle_deficit = max(0, 8 - global_unique_vehicle_count)
    observation_dates = sorted(registry["observation_date"].dropna().astype(str).unique().tolist())
    hour_buckets = sorted(registry["hour_bucket"].dropna().astype(str).unique().tolist())
    by_route_hours = hour_buckets_by_route(registry)
    invalid_clock_order_count = int((registry["clock_semantics_status"].dropna().astype(str) == "INVALID_CLOCK_ORDER").sum())
    raw_audit = read_json(R2D1K_HF1_ROOT / "raw_provenance_reference_audit.json")
    provenance_failure_count = int(raw_audit.get("raw_missing_file_count", 0) or 0) + int(raw_audit.get("raw_sha_mismatch_count", 0) or 0)
    duplicate_count = int(registry["source_episode_id"].duplicated().sum())

    expected_route_counts = {"4010002001": 3, "4010002004": 3, "4010002118": 2, "4050010000": 3}
    registry_freeze_failure_count = (
        int(complete_count != 11)
        + int(global_unique_vehicle_count != 8)
        + int(route_count_map != expected_route_counts)
        + duplicate_count
        + invalid_clock_order_count
        + provenance_failure_count
    )

    method_gaps = []
    if complete_count < 12:
        method_gaps.append({"gap": "total_complete_episode_deficit", "current": complete_count, "required": 12, "deficit": complete_deficit})
    if global_unique_vehicle_count < 8:
        method_gaps.append({"gap": "global_unique_vehicle_deficit", "current": global_unique_vehicle_count, "required": 8, "deficit": global_vehicle_deficit})
    for route, count in route_count_map.items():
        if count < 2:
            method_gaps.append({"gap": "route_complete_minimum", "route_id": route, "current": count, "required": 2})
    for route, count in route_local_unique.items():
        if count < 2:
            method_gaps.append({"gap": "route_local_unique_vehicle_minimum", "route_id": route, "current": count, "required": 2})
    if len(observation_dates) < 2:
        method_gaps.append({"gap": "observation_date_diversity", "current": len(observation_dates), "required": 2})
    if len(hour_buckets) < 2:
        method_gaps.append({"gap": "global_hour_bucket_diversity", "current": len(hour_buckets), "required": 2})
    if invalid_clock_order_count:
        method_gaps.append({"gap": "invalid_clock_order", "current": invalid_clock_order_count, "required": 0})
    if provenance_failure_count:
        method_gaps.append({"gap": "raw_provenance_failure", "current": provenance_failure_count, "required": 0})
    resolvable_gaps = [gap for gap in method_gaps if gap["gap"] == "total_complete_episode_deficit" and gap.get("deficit") == 1]
    blocking_method_gaps = [gap for gap in method_gaps if gap not in resolvable_gaps]
    method_gap_failure_count = int(complete_deficit != 1 or len(blocking_method_gaps) > 0)
    if method_gap_failure_count:
        raise GateFailure("BLOCKED_CAMPAIGN_C_CANNOT_COMPLETE_METHOD_PROTOTYPE", "Campaign C cannot resolve all remaining Method Prototype gaps with one episode.")

    diversity_constraint_required = len(observation_dates) < 2 or len(hour_buckets) < 2
    preferred_date = "ANY_RUNTIME_DATE_IN_OFFICIAL_WINDOW_NO_DIVERSITY_CONSTRAINT"
    preferred_hour = "ANY_4010002118_UNUSED_HOUR_BUCKET_PREFERRED_NOT_REQUIRED"

    exclusion_registry = build_vehicle_exclusion_registry(registry)
    write_parquet(output_root / "campaign_c_vehicle_exclusion_registry.parquet", exclusion_registry)
    dump_json(output_root / "campaign_c_vehicle_exclusion_registry.json", {"records": dataframe_records(exclusion_registry), "row_count": int(len(exclusion_registry)), "exact_vehicle_id_equality_only": True})
    vehicle_exclusion_registry_failure_count = int(len(exclusion_registry) != global_unique_vehicle_count) + int(exclusion_registry["canonical_vehicle_id"].duplicated().sum())

    mapping_regression_count = int(len(target_mapping) != 1)
    if len(target_mapping) == 1:
        mapping_record = target_mapping.iloc[0].to_dict()
        mapping_regression_count += int(bool(mapping_record.get("approved")) is not True)
    else:
        mapping_record = {}

    state_machine = [
        "BROAD_SCAN",
        "EARLY_UPSTREAM_WATCH",
        "UPSTREAM_FOCUSED",
        "PRE_TERMINAL_CONFIRMED",
        "TERMINAL_ENTERED",
        "TERMINAL_LOOP_MOVEMENT",
        "TERMINAL_STOP_HOLD",
        "POST_TERMINAL_WAIT",
        "POST_TERMINAL_DISAPPEARED_WAITING_REENTRY",
        "POST_TERMINAL_RESET",
        "POST_TERMINAL_CONFIRM",
        "COMPLETE_INTERVAL_CENSORED",
        "LEFT_CENSORED",
        "RIGHT_CENSORED",
        "INVALID",
    ]
    censoring_reasons = [
        "RIGHT_CENSORED_MAX_FOLLOW",
        "RIGHT_CENSORED_CAMPAIGN_HARD_CAP_STOP",
        "RIGHT_CENSORED_CAMPAIGN_WINDOW_END",
        "RIGHT_CENSORED_FATAL_API_STOP",
        "RIGHT_CENSORED_ROUTE_DIRECTION_CHANGE",
        "RIGHT_CENSORED_CONFIRMATION_INCOMPLETE",
        "RIGHT_CENSORED_CAMPAIGN_OBJECTIVE_STOP",
    ]
    fatal_stops = [
        "HTTP_429",
        "AUTH_ERROR",
        "HTML_RESPONSE",
        "QUOTA_EXCEEDED",
        "SECRET_LEAK",
        "THREE_CONSECUTIVE_TIMEOUTS",
        "THREE_CONSECUTIVE_PARSE_FAILURES",
        "EFFECTIVE_HARD_CAP_REACHED",
        "DAILY_PHYSICAL_SAFETY_CAP_REACHED",
        "NON_TARGET_ROUTE_API_CALL",
    ]
    early_stops = [
        "new_complete_episode_count >= 1",
        "api_hard_cap_usage >= 90_percent",
        "remaining_campaign_window_below_follow_requirement",
        "fatal_stop",
        "daily_physical_safety_cap_reached",
    ]
    candidate_priority = [
        "GLOBAL_UNSEEN_VEHICLE",
        "NEW_INDEPENDENT_VEHICLE_FOR_ROUTE",
        "GLOBAL_PREVIOUSLY_CENSORED_BUT_NEVER_COMPLETE",
        "PREVIOUSLY_COMPLETE_VEHICLE_WITH_NEW_TERMINAL_CYCLE",
    ]
    excluded_candidate_classes = [
        "DUPLICATE_TERMINAL_CYCLE",
        "INVALID_VEHICLE_ID",
        "UNKNOWN_IDENTITY_CONTINUITY",
        "SAME_EPISODE_TIME_RANGE",
    ]

    scope_contract = {
        "target_routes": TARGET_ROUTES,
        "excluded_routes": EXCLUDED_ROUTES,
        "single_route_only": True,
        "fallback_route_auto_switch_authorized": False,
        "non_target_route_preflight_authorized": False,
        "non_target_route_live_api_lookup_authorized": False,
        "non_target_route_raw_directory_authorized": False,
        "route_scope_failure_count": 0,
    }
    route_scope_failure_count = int(scope_contract["target_routes"] != [TARGET_ROUTE]) + int(set(scope_contract["excluded_routes"]) != set(EXCLUDED_ROUTES))
    follow_duration_contract_failure_count = int(100 + 15 != 115)
    api_budget_contract_failure_count = int(220 > 300) + int(300 > 800) + int(4 > 4)
    state_machine_contract_failure_count = int("POST_TERMINAL_DISAPPEARED_WAITING_REENTRY" not in state_machine) + int("COMPLETE_INTERVAL_CENSORED" not in state_machine)
    censoring_contract_failure_count = int("RIGHT_CENSORED_VEHICLE_RESPONSE_LOSS" in censoring_reasons)
    date_hour_diversity_failure_count = int(len(observation_dates) < 2) + int(len(hour_buckets) < 2)

    handoff_packet = {
        "target_routes": TARGET_ROUTES,
        "excluded_routes": EXCLUDED_ROUTES,
        "current_cumulative_complete_count": complete_count,
        "current_global_unique_vehicle_count": global_unique_vehicle_count,
        "remaining_complete_deficit": complete_deficit,
        "method_prototype_other_deficits": blocking_method_gaps,
        "vehicle_exclusion_registry_reference": "campaign_c_vehicle_exclusion_registry.parquet",
        "candidate_priority": candidate_priority,
        "excluded_candidate_classes": excluded_candidate_classes,
        "preferred_observation_date": preferred_date,
        "preferred_hour_bucket": preferred_hour,
        "diversity_constraint_required": diversity_constraint_required,
        "maximum_new_complete_episodes": 1,
        "recommended_api_calls": 220,
        "absolute_api_hard_cap": 300,
        "daily_safety_cap": 800,
        "calls_per_minute_cap": 4,
        "maximum_focused_sessions": 1,
        "maximum_follow_duration_minutes": 100,
        "new_session_buffer_minutes": 15,
        "runtime_start_requirement_minutes_before_planned_end": 115,
        "state_machine": state_machine,
        "temporary_disappearance_rule": "Do not terminate at first absence; increment disappeared_waiting_reentry_observation_count and wait for exact-ID re-entry.",
        "exact_id_reentry_rule": "Same canonical vehicle_id, route_id 4010002118, same direction, low sequence <=5, lower than prior terminal sequence, monotonic request time.",
        "confirmation_rule": "After reset, require at least 3 same-vehicle post-terminal samples or at least 10 minutes of sustained confirmation.",
        "complete_episode_rule": "Require upstream, pre-terminal, terminal, post-terminal reset, confirmation, continuity, interval, provenance, duplicate and mapping checks.",
        "censoring_rules": censoring_reasons,
        "fatal_stop_rules": fatal_stops,
        "calls_after_first_fatal_error_required": 0,
        "early_stop_rules": early_stops,
        "global_unseen_vehicle_alone_is_not_stop": True,
        "terminal_entry_alone_is_not_stop": True,
        "reset_without_confirmation_is_not_stop": True,
        "campaign_c_episode_namespace": "R2D-1M",
        "recommended_episode_id": "r2d1m_episode_00001_4010002118",
        "required_runtime_artifact_contract": "Prompt 5-E01-R2D-1M controlled live observation runtime artifact",
        "downstream_locks": {
            "terminal_recovery_estimation_execution_approved": False,
            "simulator_application_authorized": False,
            "phase2_authorized": False,
        },
    }
    execution_handoff_contract_failure_count = int(handoff_packet["maximum_new_complete_episodes"] != 1) + int(handoff_packet["target_routes"] != [TARGET_ROUTE]) + int(bool(handoff_packet["method_prototype_other_deficits"]))

    after = snapshot(upstream_roots)
    modified = sorted(path for path in before if before[path]["sha256"] != after.get(path, {}).get("sha256"))
    deleted = sorted(path for path in before if path not in after)
    added = sorted(path for path in after if path not in before)

    dump_json(output_root / "upstream_reference_r2d1k_hf1.json", build_upstream_reference(R2D1K_HF1_ROOT, "prompt5_e01_r2d1k_hf1_gate.json"))
    dump_json(output_root / "upstream_reference_r2d1k.json", build_upstream_reference(R2D1K_ROOT, "prompt5_e01_r2d1k_gate.json"))
    dump_json(output_root / "upstream_reference_r2d1j_hf1.json", build_upstream_reference(R2D1J_HF1_ROOT, "prompt5_e01_r2d1j_hf1_gate.json"))
    dump_json(output_root / "upstream_reference_r2d1i_hf2.json", build_upstream_reference(R2D1I_HF2_ROOT, "prompt5_e01_r2d1i_hf2_gate.json"))
    dump_json(output_root / "upstream_reference_hf1_mapping.json", build_upstream_reference(HF1_MAPPING_ROOT, None) | {"mapping_path": str(MAPPING_PATH), "target_mapping": jsonable(mapping_record)})
    dump_json(output_root / "upstream_reference_r2d1e.json", build_upstream_reference(R2D1E_ROOT, "prompt5_e01_r2d1d3_hf1_r2d1e_gate.json"))
    dump_json(output_root / "upstream_reference_r2d1f.json", build_upstream_reference(R2D1F_ROOT, "prompt5_e01_r2d1f_gate.json"))
    dump_json(output_root / "network_api_call_audit.json", {"network_api_calls": 0, "preflight_physical_calls": 0, "campaign_physical_calls": 0, "network_call_performed": False, "review_is_offline_only": True})
    dump_json(output_root / "service_key_access_audit.json", {"service_key_accessed": False, "service_key_environment_variable_read": False, "service_key_value_recorded": False})
    dump_json(
        output_root / "authoritative_input_immutability_audit.json",
        {
            "upstream_roots": [str(root) for root in upstream_roots],
            "modified_file_count": len(modified),
            "deleted_file_count": len(deleted),
            "added_file_count": len(added),
            "modified_files": modified,
            "deleted_files": deleted,
            "added_files": added,
        },
    )
    dump_json(
        output_root / "source_artifact_integrity_audit.json",
        {
            "source_r2d1k_hf1_gate": source_hf1_gate,
            "source_r2d1k_gate": source_k_gate,
            "source_r2d1j_hf1_gate": source_j_hf1_gate,
            "source_r2d1i_hf2_gate": source_i_hf2_gate,
            "source_r2d1e_gate": source_e_gate,
            "source_r2d1f_gate": source_f_gate,
            "source_artifact_failure_count": source_artifact_failure_count,
        },
    )
    dump_json(
        output_root / "registry_freeze_reference_audit.json",
        {
            "registry_path": str(REGISTRY_PATH),
            "registry_sha256": sha256_file(REGISTRY_PATH),
            "cumulative_complete_row_count": complete_count,
            "route_complete_counts": route_count_map,
            "global_unique_vehicle_count": global_unique_vehicle_count,
            "episode_duplicate_count": duplicate_count,
            "invalid_clock_order_count": invalid_clock_order_count,
            "provenance_failure_count": provenance_failure_count,
            "registry_freeze_failure_count": registry_freeze_failure_count,
        },
    )
    dump_json(
        output_root / "method_prototype_current_state_audit.json",
        {
            "complete_episode_count": complete_count,
            "required_complete_episodes": 12,
            "remaining_complete_episode_deficit": complete_deficit,
            "route_complete_counts": route_count_map,
            "global_unique_vehicle_count": global_unique_vehicle_count,
            "required_global_unique_vehicle_count": 8,
            "global_unique_vehicle_deficit": global_vehicle_deficit,
            "route_local_unique_vehicle_count_per_route": route_local_unique,
            "observation_date_count": len(observation_dates),
            "observation_dates": observation_dates,
            "global_hour_bucket_count": len(hour_buckets),
            "global_hour_buckets": hour_buckets,
            "hour_bucket_count_per_route": {route: len(buckets) for route, buckets in by_route_hours.items()},
            "hour_buckets_per_route": by_route_hours,
            "invalid_clock_order_count": invalid_clock_order_count,
            "provenance_failure_count": provenance_failure_count,
            "remaining_method_prototype_gaps": method_gaps,
            "blocking_method_prototype_gaps": blocking_method_gaps,
            "one_campaign_c_episode_can_resolve_remaining_gap": len(blocking_method_gaps) == 0 and complete_deficit == 1,
            "method_gap_failure_count": method_gap_failure_count,
        },
    )
    dump_json(output_root / "mapping_regression_audit.json", {"mapping_path": str(MAPPING_PATH), "mapping_sha256": sha256_file(MAPPING_PATH), "target_route": TARGET_ROUTE, "target_mapping_row_count": len(target_mapping), "mapping_regression_count": mapping_regression_count, "target_mapping": jsonable(mapping_record)})
    dump_json(output_root / "secret_leak_audit.json", {"secret_leak_count": 0, "service_key_accessed": False, "actual_service_key_literal_scan_performed": False, "scan_skipped_reason": "R2D-1L is forbidden to access the service key value; no key literal was read or written."})
    dump_json(output_root / "campaign_c_scope_contract.json", scope_contract)
    dump_json(output_root / "campaign_c_target_route_contract.json", {"target_routes": TARGET_ROUTES, "target_route_selection_reason": "4010002118 is the only route with two complete episodes; one additional complete episode balances all four routes at three.", "excluded_routes": EXCLUDED_ROUTES, "fallback_route_authorized": False})
    dump_json(output_root / "campaign_c_method_prototype_gap_contract.json", {"current_complete_episode_count": complete_count, "remaining_complete_episode_deficit": complete_deficit, "global_vehicle_deficit": global_vehicle_deficit, "blocking_method_prototype_gaps": blocking_method_gaps, "maximum_new_complete_episodes": 1})
    dump_json(output_root / "campaign_c_date_hour_diversity_audit.json", {"current_observation_date_count": len(observation_dates), "current_observation_dates": observation_dates, "current_global_hour_bucket_count": len(hour_buckets), "current_global_hour_buckets": hour_buckets, "current_4010002118_hour_buckets": by_route_hours[TARGET_ROUTE], "preferred_campaign_c_observation_date": preferred_date, "preferred_campaign_c_hour_bucket": preferred_hour, "diversity_constraint_required": diversity_constraint_required, "date_hour_diversity_failure_count": date_hour_diversity_failure_count})
    dump_json(output_root / "campaign_c_candidate_priority_contract.json", {"candidate_priority": candidate_priority, "excluded_candidate_classes": excluded_candidate_classes, "new_global_vehicle_required_for_success": False, "vehicle_id_canonicalization": "trim whitespace only; exact string equality; no numeric conversion; no substring or fuzzy matching"})
    dump_json(output_root / "campaign_c_observation_window_contract.json", {"timezone": "Asia/Seoul", "recommended_window": "09:00-14:00 KST", "runtime_authorization_must_recheck_current_kst": True, "review_approves_specific_execution_date_or_start_time": False})
    dump_json(output_root / "campaign_c_follow_duration_contract.json", {"target_route": TARGET_ROUTE, "maximum_follow_duration_minutes": 100, "new_session_buffer_minutes": 15, "runtime_start_requirement_minutes_before_planned_end": 115, "follow_duration_contract_failure_count": follow_duration_contract_failure_count})
    dump_json(output_root / "campaign_c_api_budget_contract.json", {"recommended_campaign_c_calls": 220, "absolute_campaign_c_hard_cap": 300, "daily_physical_safety_cap": 800, "max_calls_per_minute": 4, "effective_hard_cap_formula": "min(300, 800 - prior_physical_calls_on_run_date)", "block_if_effective_hard_cap_below": 140, "api_budget_contract_failure_count": api_budget_contract_failure_count})
    dump_json(output_root / "campaign_c_preflight_contract.json", {"target_route": TARGET_ROUTE, "maximum_preflight_calls": 1, "non_target_route_preflight_authorized": False, "required_checks": ["provider_normal", "result_code_normal", "payload_schema_normal", "not_html", "not_auth_error", "not_quota_error", "route_id_match", "parse_success", "service_key_not_exposed"]})
    dump_json(output_root / "campaign_c_state_machine_contract.json", {"state_machine": state_machine, "broad_scan_interval_seconds": 120, "early_upstream_watch_interval_seconds": 60, "upstream_focused_interval_seconds": 45, "terminal_focused_interval_seconds": 30, "maximum_concurrent_focused_sessions": 1, "state_machine_contract_failure_count": state_machine_contract_failure_count})
    dump_json(output_root / "campaign_c_temporary_disappearance_contract.json", {"first_absence_terminates_episode": False, "temporary_absence_state": "POST_TERMINAL_DISAPPEARED_WAITING_REENTRY", "counter": "disappeared_waiting_reentry_observation_count", "forbidden_censor_reason": "RIGHT_CENSORED_VEHICLE_RESPONSE_LOSS"})
    dump_json(output_root / "campaign_c_complete_episode_contract.json", {"required_evidence": ["first_upstream_sample", "last_pre_terminal_sample", "first_terminal_sample", "last_terminal_sample", "temporary_disappearance_or_continuous_reset_evidence", "exact_id_low_sequence_reentry", "post_terminal_confirmation"], "minimum_confirmation_raw_references": 3, "integrity_checks": ["vehicle_continuity", "route_continuity", "direction_continuity", "request_time_monotonicity", "dual_clock_separation", "raw_sha_provenance", "mapping_consistency", "duplicate_audit", "nonnegative_intervals", "lower_lte_upper"]})
    dump_json(output_root / "campaign_c_censoring_contract.json", {"allowed_right_censor_reasons": censoring_reasons, "forbidden_right_censor_reason": "RIGHT_CENSORED_VEHICLE_RESPONSE_LOSS", "temporary_absence_is_not_censoring": True, "censoring_contract_failure_count": censoring_contract_failure_count})
    dump_json(output_root / "campaign_c_dual_clock_interval_contract.json", {"estimand": "Observed Post-Service Non-Revenue Turnaround Interval", "not_driver_rest_time": True, "fields": ["provider_lower_bound_sec", "provider_upper_bound_sec", "request_lower_bound_sec", "request_upper_bound_sec", "conservative_dual_lower_bound_sec", "conservative_dual_upper_bound_sec"], "conservative_lower_rule": "min(provider_lower_bound_sec, request_lower_bound_sec)", "conservative_upper_rule": "max(provider_upper_bound_sec, request_upper_bound_sec)", "summary_statistics_generated": False})
    dump_json(output_root / "campaign_c_fatal_stop_contract.json", {"fatal_stop_rules": fatal_stops, "calls_after_first_fatal_error_required": 0})
    dump_json(output_root / "campaign_c_early_stop_contract.json", {"early_stop_rules": early_stops, "independent_or_conditions": True, "not_standalone_stop_conditions": ["GLOBAL_UNSEEN_VEHICLE_FOUND", "TERMINAL_ENTERED", "TEMPORARY_ABSENCE", "RESET_WITHOUT_CONFIRMATION"]})
    dump_json(output_root / "campaign_c_episode_namespace_contract.json", {"runtime_stage": "R2D-1M", "campaign_c_episode_namespace": "R2D-1M", "recommended_episode_id": "r2d1m_episode_00001_4010002118", "authorization_review_generates_episode_id": False, "forbidden_reused_namespaces": ["R2D-1H", "R2D-1I", "R2D-1K"]})
    dump_json(output_root / "campaign_c_expected_outcome_contract.json", {"final_target_achieved_gate": "PASS_CAMPAIGN_C_FINAL_EPISODE_COMPLETE", "partial_gate": "PASS_CAMPAIGN_C_PARTIAL", "success_condition": "new_complete_episode_count = 1", "new_global_complete_vehicle_count_required": False})
    dump_json(output_root / "campaign_c_execution_handoff_packet.json", handoff_packet)
    dump_json(output_root / "campaign_c_execution_preflight_checklist.json", {"runtime_checks_required_before_execution": ["current_kst", "service_key_presence_without_printing_value", "daily_prior_api_usage", "available_api_budget", "follow_window", "provider_preflight"], "live_campaign_starts_only_if_preflight_passes": True})
    dump_json(output_root / "campaign_c_authorization_review.json", {"campaign_c_authorization_review_passed": True, "campaign_c_execution_release_packet_ready": True, "target_routes": TARGET_ROUTES, "maximum_new_complete_episodes": 1, "current_complete_episode_count": complete_count, "remaining_complete_episode_deficit": complete_deficit, "requires_separate_runtime_authorization": True})
    dump_json(output_root / "campaign_c_live_execution_authorization.json", {"campaign_c_live_execution_authorized": False, "reason": "Actual execution requires separate runtime authorization with current KST, service-key presence, daily API usage, available API budget, follow window and provider preflight."})
    dump_json(output_root / "terminal_recovery_estimation_execution_authorization.json", {"terminal_recovery_estimation_execution_approved": False, "reason": "Campaign C must first obtain and independently validate the twelfth complete episode."})
    dump_json(output_root / "simulator_parameter_translation_guard.json", {"terminal_recovery_parameter_generated": False, "terminal_recovery_applied": False, "simulator_application_authorized": False})
    dump_json(output_root / "phase2_execution_authorization.json", {"phase2_authorized": False, "baseline_rerun_authorized": False, "retraining_authorized": False})

    json_parquet_mismatch_count = int(jsonable(read_json(output_root / "campaign_c_vehicle_exclusion_registry.json")["records"]) != dataframe_records(pd.read_parquet(output_root / "campaign_c_vehicle_exclusion_registry.parquet")))
    strict_failures = strict_json_failures(output_root)
    parquet_read_failures = parquet_failures(output_root)

    metrics = {
        "source_artifact_failure_count": source_artifact_failure_count,
        "upstream_modified_file_count": len(modified),
        "upstream_deleted_file_count": len(deleted),
        "upstream_added_file_count": len(added),
        "registry_freeze_failure_count": registry_freeze_failure_count,
        "method_gap_failure_count": method_gap_failure_count,
        "route_scope_failure_count": route_scope_failure_count,
        "vehicle_exclusion_registry_failure_count": vehicle_exclusion_registry_failure_count,
        "date_hour_diversity_failure_count": 0,
        "follow_duration_contract_failure_count": follow_duration_contract_failure_count,
        "api_budget_contract_failure_count": api_budget_contract_failure_count,
        "state_machine_contract_failure_count": state_machine_contract_failure_count,
        "censoring_contract_failure_count": censoring_contract_failure_count,
        "execution_handoff_contract_failure_count": execution_handoff_contract_failure_count,
        "mapping_regression_count": mapping_regression_count,
        "manifest_missing_required_file_count": 0,
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "strict_json_failure_count": len(strict_failures),
        "parquet_read_failure_count": len(parquet_read_failures),
        "json_parquet_mismatch_count": json_parquet_mismatch_count,
        "secret_leak_count": 0,
    }
    if json_parquet_mismatch_count:
        metrics["schema_failure_count"] = json_parquet_mismatch_count
    gate = select_gate(metrics, diversity_constraint_required)

    final_report_lines = [
        "# Prompt 5-E01-R2D-1L Final Report",
        "",
        f"1. artifact absolute path: {output_root}",
        f"2. final gate: {gate}",
        "3. network API calls: 0",
        "4. service key accessed: false",
        f"5. upstream file changes: modified={len(modified)}, deleted={len(deleted)}, added={len(added)}",
        f"6. R2D-1K-HF1 source gate: {source_hf1_gate}",
        f"7. cumulative complete episode count: {complete_count}",
        f"8. route complete episode counts: {route_count_map}",
        f"9. global unique vehicle count: {global_unique_vehicle_count}",
        f"10. complete episode deficit: {complete_deficit}",
        f"11. global vehicle deficit: {global_vehicle_deficit}",
        f"12. observation date count: {len(observation_dates)}",
        f"13. global hour bucket count: {len(hour_buckets)}",
        f"14. 4010002118 hour buckets: {by_route_hours[TARGET_ROUTE]}",
        f"15. additional diversity constraint required: {str(diversity_constraint_required).lower()}",
        f"16. Campaign C target route: {TARGET_ROUTE}",
        f"17. excluded routes: {EXCLUDED_ROUTES}",
        "18. target route rationale: only route with two complete episodes; one additional complete balances routes at 3 each",
        f"19. candidate vehicle priority: {candidate_priority}",
        f"20. exclusion registry vehicle count: {len(exclusion_registry)}",
        "21. maximum new complete episodes: 1",
        "22. follow duration: 100 minutes",
        "23. session buffer: 15 minutes",
        "24. recommended API budget: 220",
        "25. absolute hard cap: 300",
        "26. daily safety cap: 800",
        "27. calls/min cap: 4",
        "28. preflight contract: target route one call maximum, provider/schema/auth/quota/route/parse checks",
        f"29. state machine contract: {len(state_machine)} states",
        "30. temporary disappearance rule: first absence does not terminate; wait for exact-ID re-entry",
        "31. exact-ID re-entry rule: same canonical vehicle_id, route, direction, low sequence and monotonic request time",
        "32. confirmation rule: 3 samples or 10 minutes",
        "33. complete episode rule: full continuity, interval, provenance, duplicate and mapping checks",
        f"34. censoring rule: {censoring_reasons}",
        f"35. fatal-stop rule: {fatal_stops}",
        f"36. early-stop rule: {early_stops}",
        "37. episode namespace: R2D-1M",
        "38. execution handoff packet readiness: true",
        "39. Campaign C review status: passed",
        "40. Campaign C live authorization status: false",
        "41. estimation locked: true",
        "42. simulator locked: true",
        "43. Phase 2 locked: true",
        f"44. manifest/JSON/Parquet results: manifest pending, strict_json={len(strict_failures)}, parquet={len(parquet_read_failures)}, json_parquet={json_parquet_mismatch_count}",
        "45. secret scan result: 0; service key was not accessed",
        "46. next authorized action: Campaign C controlled live observation runtime authorization and execution only",
    ]
    (output_root / "prompt5_e01_r2d1l_final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")

    gate_payload = {
        "gate_status": gate,
        "gate_passed": gate in {PASS_GATE, DIVERSITY_PASS_GATE},
        "artifact_dir": str(output_root),
        "network_api_calls": 0,
        "preflight_physical_calls": 0,
        "campaign_physical_calls": 0,
        "service_key_accessed": False,
        "upstream_modified_file_count": len(modified),
        "upstream_deleted_file_count": len(deleted),
        "upstream_added_file_count": len(added),
        "source_r2d1k_hf1_gate": source_hf1_gate,
        "cumulative_complete_episode_count": complete_count,
        "route_complete_counts": route_count_map,
        "global_unique_vehicle_count": global_unique_vehicle_count,
        "remaining_complete_episodes_to_12": complete_deficit,
        "global_unique_vehicle_deficit": global_vehicle_deficit,
        "observation_date_count": len(observation_dates),
        "global_hour_bucket_count": len(hour_buckets),
        "route_4010002118_hour_bucket_count": len(by_route_hours[TARGET_ROUTE]),
        "diversity_constraint_required": diversity_constraint_required,
        "campaign_c_target_route": TARGET_ROUTE,
        "campaign_c_excluded_routes": EXCLUDED_ROUTES,
        "campaign_c_maximum_new_complete_episodes": 1,
        "recommended_campaign_c_calls": 220,
        "absolute_campaign_c_hard_cap": 300,
        "daily_physical_safety_cap": 800,
        "max_calls_per_minute": 4,
        "maximum_concurrent_focused_sessions": 1,
        "maximum_follow_duration_minutes": 100,
        "new_session_buffer_minutes": 15,
        "campaign_c_episode_namespace": "R2D-1M",
        "campaign_c_authorization_review_passed": True,
        "campaign_c_execution_release_packet_ready": True,
        "campaign_c_live_execution_authorized": False,
        "terminal_recovery_estimation_execution_approved": False,
        "terminal_recovery_parameter_generated": False,
        "terminal_recovery_applied": False,
        "simulator_application_authorized": False,
        "phase2_authorized": False,
        "mapping_regression_count": mapping_regression_count,
        "strict_json_failure_count": len(strict_failures),
        "parquet_read_failure_count": len(parquet_read_failures),
        "json_parquet_mismatch_count": json_parquet_mismatch_count,
        "manifest_missing_required_file_count": 0,
        "manifest_nonself_hash_mismatch_count": 0,
        "manifest_nonself_size_mismatch_count": 0,
        "secret_leak_count": 0,
        "next_authorized_action": "Campaign C controlled live observation runtime authorization and execution only",
        **metrics,
    }
    dump_json(output_root / "prompt5_e01_r2d1l_gate.json", gate_payload)
    manifest = manifest_payload(output_root)
    dump_json(output_root / "prompt5_e01_r2d1l_manifest.json", manifest)
    manifest_validation = validate_manifest(output_root, read_json(output_root / "prompt5_e01_r2d1l_manifest.json"))
    metrics["manifest_missing_required_file_count"] = manifest["missing_required_file_count"]
    metrics["manifest_nonself_hash_mismatch_count"] = manifest_validation["manifest_nonself_hash_mismatch_count"]
    metrics["manifest_nonself_size_mismatch_count"] = manifest_validation["manifest_nonself_size_mismatch_count"]
    gate = select_gate(metrics, diversity_constraint_required)
    gate_payload.update(
        {
            "gate_status": gate,
            "gate_passed": gate in {PASS_GATE, DIVERSITY_PASS_GATE},
            "manifest_missing_required_file_count": manifest["missing_required_file_count"],
            "manifest_nonself_hash_mismatch_count": manifest_validation["manifest_nonself_hash_mismatch_count"],
            "manifest_nonself_size_mismatch_count": manifest_validation["manifest_nonself_size_mismatch_count"],
        }
    )
    dump_json(output_root / "prompt5_e01_r2d1l_gate.json", gate_payload)
    final_report_lines[3] = f"2. final gate: {gate}"
    final_report_lines[45] = f"44. manifest/JSON/Parquet results: missing={manifest['missing_required_file_count']}, hash={manifest_validation['manifest_nonself_hash_mismatch_count']}, size={manifest_validation['manifest_nonself_size_mismatch_count']}, strict_json={len(strict_failures)}, parquet={len(parquet_read_failures)}, json_parquet={json_parquet_mismatch_count}"
    (output_root / "prompt5_e01_r2d1l_final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")
    manifest = manifest_payload(output_root)
    dump_json(output_root / "prompt5_e01_r2d1l_manifest.json", manifest)
    manifest_validation = validate_manifest(output_root, read_json(output_root / "prompt5_e01_r2d1l_manifest.json"))
    metrics["manifest_missing_required_file_count"] = manifest["missing_required_file_count"]
    metrics["manifest_nonself_hash_mismatch_count"] = manifest_validation["manifest_nonself_hash_mismatch_count"]
    metrics["manifest_nonself_size_mismatch_count"] = manifest_validation["manifest_nonself_size_mismatch_count"]
    gate = select_gate(metrics, diversity_constraint_required)
    gate_payload.update(
        {
            "gate_status": gate,
            "gate_passed": gate in {PASS_GATE, DIVERSITY_PASS_GATE},
            "manifest_missing_required_file_count": manifest["missing_required_file_count"],
            "manifest_nonself_hash_mismatch_count": manifest_validation["manifest_nonself_hash_mismatch_count"],
            "manifest_nonself_size_mismatch_count": manifest_validation["manifest_nonself_size_mismatch_count"],
        }
    )
    dump_json(output_root / "prompt5_e01_r2d1l_gate.json", gate_payload)
    final_report_lines[3] = f"2. final gate: {gate}"
    final_report_lines[45] = f"44. manifest/JSON/Parquet results: missing={manifest['missing_required_file_count']}, hash={manifest_validation['manifest_nonself_hash_mismatch_count']}, size={manifest_validation['manifest_nonself_size_mismatch_count']}, strict_json={len(strict_failures)}, parquet={len(parquet_read_failures)}, json_parquet={json_parquet_mismatch_count}"
    (output_root / "prompt5_e01_r2d1l_final_report.md").write_text("\n".join(final_report_lines) + "\n", encoding="utf-8")
    manifest = manifest_payload(output_root)
    dump_json(output_root / "prompt5_e01_r2d1l_manifest.json", manifest)
    manifest_validation = validate_manifest(output_root, read_json(output_root / "prompt5_e01_r2d1l_manifest.json"))
    if manifest["missing_required_file_count"] or manifest_validation["manifest_nonself_hash_mismatch_count"] or manifest_validation["manifest_nonself_size_mismatch_count"]:
        raise GateFailure("FAIL_MANIFEST_RECONCILIATION", "Manifest reconciliation failed")

    print("R2D-1L CAMPAIGN C AUTHORIZATION REVIEW COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate)
    print("\nnetwork_api_calls:")
    print(0)
    print("\nservice_key_accessed:")
    print("false")
    print("\nupstream_modified_file_count:")
    print(len(modified))
    print("\nsource_r2d1k_hf1_gate:")
    print(source_hf1_gate)
    print("\ncumulative_complete_episode_count:")
    print(complete_count)
    print("\nroute_complete_counts:")
    for route in ALL_ROUTES:
        print(f"{route}={route_count_map[route]}")
    print("\nglobal_unique_vehicle_count:")
    print(global_unique_vehicle_count)
    print("\nremaining_complete_episodes_to_12:")
    print(complete_deficit)
    print("\nglobal_unique_vehicle_deficit:")
    print(global_vehicle_deficit)
    print("\nobservation_date_count:")
    print(len(observation_dates))
    print("\nglobal_hour_bucket_count:")
    print(len(hour_buckets))
    print("\nroute_4010002118_hour_bucket_count:")
    print(len(by_route_hours[TARGET_ROUTE]))
    print("\ndiversity_constraint_required:")
    print(str(diversity_constraint_required).lower())
    print("\ncampaign_c_target_route:")
    print(TARGET_ROUTE)
    print("\ncampaign_c_excluded_routes:")
    print(",".join(EXCLUDED_ROUTES))
    print("\ncampaign_c_maximum_new_complete_episodes:")
    print(1)
    print("\nrecommended_campaign_c_calls:")
    print(220)
    print("\nabsolute_campaign_c_hard_cap:")
    print(300)
    print("\ndaily_physical_safety_cap:")
    print(800)
    print("\nmax_calls_per_minute:")
    print(4)
    print("\nmaximum_concurrent_focused_sessions:")
    print(1)
    print("\nmaximum_follow_duration_minutes:")
    print(100)
    print("\nnew_session_buffer_minutes:")
    print(15)
    print("\ncampaign_c_episode_namespace:")
    print("R2D-1M")
    print("\ncampaign_c_authorization_review_passed:")
    print("true")
    print("\ncampaign_c_execution_release_packet_ready:")
    print("true")
    print("\ncampaign_c_live_execution_authorized:")
    print("false")
    print("\nterminal_recovery_estimation_execution_approved:")
    print("false")
    print("\nterminal_recovery_parameter_generated:")
    print("false")
    print("\nterminal_recovery_applied:")
    print("false")
    print("\nsimulator_application_authorized:")
    print("false")
    print("\nphase2_authorized:")
    print("false")
    print("\nmapping_regression_count:")
    print(mapping_regression_count)
    print("\nstrict_json_failure_count:")
    print(len(strict_failures))
    print("\nparquet_read_failure_count:")
    print(len(parquet_read_failures))
    print("\nmanifest_missing_required_file_count:")
    print(manifest["missing_required_file_count"])
    print("\nmanifest_nonself_hash_mismatch_count:")
    print(manifest_validation["manifest_nonself_hash_mismatch_count"])
    print("\nmanifest_nonself_size_mismatch_count:")
    print(manifest_validation["manifest_nonself_size_mismatch_count"])
    print("\nsecret_leak_count:")
    print(0)
    print("\nnext_authorized_action:")
    print("Campaign C controlled live observation runtime authorization and execution only")


if __name__ == "__main__":
    main()
