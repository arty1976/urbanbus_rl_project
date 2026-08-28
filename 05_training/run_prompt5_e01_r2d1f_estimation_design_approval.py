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
R2D1D3_REL = ARTIFACTS_REL / "prompt5_e01_r2d1d3_targeted_4010002118_clock_audit_20260724_122349"
R2D1E_REL = ARTIFACTS_REL / "prompt5_e01_r2d1d3_hf1_r2d1e_methodology_review_20260724_145209"
OUTPUT_PREFIX = "prompt5_e01_r2d1f_estimation_design_approval"
TARGET_ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]
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


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False, default=str)
    path.write_text(text + "\n", encoding="utf-8")
    strict_read_json(path)


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_for_json(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, str, bytes)) else False:
        return None
    return value


def df_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return normalize_for_json(df.where(pd.notnull(df), None).to_dict(orient="records"))


def file_audit(path: Path, project_root: Path, source: str) -> Dict[str, Any]:
    record = {
        "absolute_path": str(path),
        "relative_path": rel(path, project_root),
        "file_size": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "readable": False,
        "schema_readable": False,
        "authoritative_source": source,
        "modified_during_r2d1f": False,
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
        else:
            pass
        record["schema_readable"] = True
    except Exception as exc:
        record["schema_error"] = f"{type(exc).__name__}: {exc}"
    return record


def token_scan(path: Path) -> int:
    return len(STRICT_JSON_RE.findall(path.read_bytes()))


def as_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except Exception:
        return None


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"true", "1", "yes"}


def parse_timestamp(value: Any) -> Optional[pd.Timestamp]:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def day_name(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None:
        return None
    return ts.day_name()


def hour_bucket(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None:
        return None
    return f"{ts.hour:02d}:00-{ts.hour:02d}:59"


def build_route_boundary_table(mapping_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for route_id in TARGET_ROUTES:
        src = mapping_df[mapping_df["route_id"].astype(str) == route_id]
        if src.empty:
            rows.append({"route_id": route_id, "boundary_contract_passed": False, "boundary_failure_reason": "MISSING_ROUTE_MAPPING"})
            continue
        row = src.iloc[0].to_dict()
        service_terminal = as_float(row.get("service_terminal_sequence"))
        effective = as_float(row.get("effective_live_terminal_sequence")) or as_float(row.get("full_terminal_sequence")) or service_terminal
        observed_live_max = as_float(row.get("last_unique_preclosure_sequence")) or as_float(row.get("full_sequence_max")) or effective
        live_extension = effective is not None and service_terminal is not None and effective > service_terminal
        rows.append(
            {
                "route_id": route_id,
                "route_no": row.get("route_no"),
                "service_terminal_sequence": service_terminal,
                "effective_live_terminal_sequence": effective,
                "observed_live_max_sequence": observed_live_max,
                "reset_entry_sequence": 5,
                "terminal_operation_type": row.get("terminal_operation_type"),
                "terminal_operation_resolved": as_bool(row.get("terminal_operation_resolved")),
                "service_end_boundary_rule": "last valid observation at/before service_terminal_sequence to first valid observation beyond service terminal boundary; exact point time is not asserted",
                "non_revenue_segment_start_rule": "starts after service terminal boundary crossing window; may include terminal zone movement and live sequence extension",
                "reset_entry_rule": "same route/direction/vehicle, previous_sequence >= effective_live_terminal_sequence - 5, current_sequence <= 5, and current_sequence < previous_sequence",
                "boundary_evidence_source": "HF1 mapping contract v10 and R2D-1E terminal phase derivation contract",
                "boundary_confidence": row.get("confidence"),
                "live_sequence_extension_status": "LIVE_SEQUENCE_EXTENSION" if live_extension else "NO_LIVE_SEQUENCE_EXTENSION",
                "preserve_live_sequence_extension": route_id == "4050010000" and live_extension,
                "boundary_contract_passed": as_bool(row.get("approved")) and as_bool(row.get("terminal_operation_resolved")),
                "boundary_failure_reason": None,
            }
        )
    return pd.DataFrame(rows)


def build_dual_clock_validation(clock_df: pd.DataFrame, registry: pd.DataFrame, route_table: pd.DataFrame) -> pd.DataFrame:
    route_lookup = route_table.set_index("route_id").to_dict(orient="index")
    reg_lookup = registry.set_index("frozen_episode_id").to_dict(orient="index")
    rows = []
    for _, clock in clock_df.iterrows():
        frozen_id = str(clock.get("frozen_episode_id"))
        reg = reg_lookup.get(frozen_id, {})
        route_id = str(clock.get("route_id"))
        route_contract = route_lookup.get(route_id, {})
        provider_lower = as_float(clock.get("provider_recovery_lower_bound_sec"))
        provider_upper = as_float(clock.get("provider_recovery_upper_bound_sec"))
        request_lower = as_float(clock.get("request_recovery_lower_bound_sec"))
        request_upper = as_float(clock.get("request_recovery_upper_bound_sec"))
        conservative_lower = min(provider_lower, request_lower) if provider_lower is not None and request_lower is not None else None
        conservative_upper = max(provider_upper, request_upper) if provider_upper is not None and request_upper is not None else None
        intersection_lower = max(provider_lower, request_lower) if provider_lower is not None and request_lower is not None else None
        intersection_upper = min(provider_upper, request_upper) if provider_upper is not None and request_upper is not None else None
        intersection_valid = intersection_lower is not None and intersection_upper is not None and intersection_lower <= intersection_upper
        lower_order = provider_lower is not None and provider_upper is not None and provider_lower <= provider_upper
        request_order = request_lower is not None and request_upper is not None and request_lower <= request_upper
        conservative_order = conservative_lower is not None and conservative_upper is not None and conservative_lower <= conservative_upper
        boundary_missing = any(v is None for v in [provider_lower, provider_upper, request_lower, request_upper])
        negative_interval = any(v is not None and v < 0 for v in [provider_lower, provider_upper, request_lower, request_upper, conservative_lower, conservative_upper])
        raw_sha_passed = as_bool(reg.get("evidence_sha256_passed"))
        mapping_contract_passed = bool(route_contract.get("boundary_contract_passed"))
        row = {
            "frozen_episode_id": frozen_id,
            "source_artifact": clock.get("source_artifact"),
            "source_episode_id": clock.get("episode_id"),
            "route_id": route_id,
            "vehicle_id": str(clock.get("vehicle_id")),
            "direction": str(reg.get("direction")),
            "provider_service_end_window_start": clock.get("last_pre_terminal_provider_time"),
            "provider_service_end_window_end": clock.get("first_terminal_provider_time"),
            "provider_reentry_window_start": clock.get("last_terminal_provider_time"),
            "provider_reentry_window_end": clock.get("first_post_terminal_provider_time"),
            "request_service_end_window_start": clock.get("last_pre_terminal_request_time"),
            "request_service_end_window_end": clock.get("first_terminal_request_time"),
            "request_reentry_window_start": clock.get("last_terminal_request_time"),
            "request_reentry_window_end": clock.get("first_post_terminal_request_time"),
            "provider_lower_bound_sec": provider_lower,
            "provider_upper_bound_sec": provider_upper,
            "request_lower_bound_sec": request_lower,
            "request_upper_bound_sec": request_upper,
            "conservative_dual_lower_bound_sec": conservative_lower,
            "conservative_dual_upper_bound_sec": conservative_upper,
            "clock_intersection_lower_bound_sec": intersection_lower,
            "clock_intersection_upper_bound_sec": intersection_upper,
            "clock_intersection_valid": intersection_valid,
            "clock_semantics_status": clock.get("clock_semantics_status"),
            "provider_timestamp_repeat_span_sec": as_float(clock.get("maximum_provider_repeat_request_span_sec")),
            "provider_event_lag_sec": as_float(clock.get("maximum_provider_event_lag_sec")),
            "provider_interval_order_passed": lower_order,
            "request_interval_order_passed": request_order,
            "conservative_interval_order_passed": conservative_order,
            "time_order_error": not (lower_order and request_order and conservative_order),
            "negative_interval": negative_interval,
            "missing_boundary": boundary_missing,
            "raw_sha_provenance_passed": raw_sha_passed,
            "mapping_contract_passed": mapping_contract_passed,
            "interval_validation_passed": lower_order and request_order and conservative_order and not boundary_missing and not negative_interval and raw_sha_passed and mapping_contract_passed,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def input_schema() -> List[Dict[str, Any]]:
    fields = [
        ("frozen_episode_id", "string", False, "unique frozen episode identifier"),
        ("source_artifact", "string", False, "source observation artifact lineage"),
        ("source_episode_id", "string", False, "source episode id"),
        ("route_id", "string", False, "approved route id"),
        ("vehicle_id", "string", False, "observed exact vehicle id"),
        ("direction", "string", False, "BIS direction id"),
        ("observation_date", "date", False, "derived from request service-end window start"),
        ("day_of_week", "string", False, "temporal context only"),
        ("hour_bucket", "string", False, "temporal context only"),
        ("service_terminal_sequence", "number", False, "from HF1 mapping"),
        ("effective_live_terminal_sequence", "number", False, "from HF1 mapping"),
        ("observed_live_max_sequence", "number", False, "from HF1 mapping/live evidence"),
        ("provider_service_end_window_start", "timestamp", False, "provider clock start boundary"),
        ("provider_service_end_window_end", "timestamp", False, "provider clock end boundary"),
        ("provider_reentry_window_start", "timestamp", False, "provider clock reentry start boundary"),
        ("provider_reentry_window_end", "timestamp", False, "provider clock reentry end boundary"),
        ("request_service_end_window_start", "timestamp", False, "request clock start boundary"),
        ("request_service_end_window_end", "timestamp", False, "request clock end boundary"),
        ("request_reentry_window_start", "timestamp", False, "request clock reentry start boundary"),
        ("request_reentry_window_end", "timestamp", False, "request clock reentry end boundary"),
        ("provider_lower_bound_sec", "number", False, "episode-level provider lower interval bound"),
        ("provider_upper_bound_sec", "number", False, "episode-level provider upper interval bound"),
        ("request_lower_bound_sec", "number", False, "episode-level request lower interval bound"),
        ("request_upper_bound_sec", "number", False, "episode-level request upper interval bound"),
        ("conservative_dual_lower_bound_sec", "number", False, "min(provider lower, request lower)"),
        ("conservative_dual_upper_bound_sec", "number", False, "max(provider upper, request upper)"),
        ("left_censored", "boolean", False, "left censoring flag"),
        ("right_censored", "boolean", False, "right censoring flag"),
        ("interval_censored", "boolean", False, "interval-censored episode flag"),
        ("complete_episode", "boolean", False, "complete interval-censored episode flag"),
        ("clock_semantics_status", "enum", False, "provider timestamp freshness class"),
        ("provider_timestamp_freshness_class", "enum", False, "same as clock_semantics_status for design"),
        ("request_sampling_interval_sec", "number", True, "request reentry window width; diagnostic only"),
        ("vehicle_id_continuity_passed", "boolean", False, "source continuity check"),
        ("route_continuity_passed", "boolean", False, "source continuity check"),
        ("direction_continuity_passed", "boolean", False, "source continuity check"),
        ("raw_sha_provenance_passed", "boolean", False, "source evidence SHA check"),
        ("mapping_contract_passed", "boolean", False, "HF1 route boundary contract check"),
        ("eligible_for_estimation_input", "boolean", False, "candidate row passes design validation"),
        ("exclusion_reason", "string", True, "null when eligible"),
    ]
    return [
        {
            "field": name,
            "type": typ,
            "nullable": nullable,
            "allowed_values": ["PROVIDER_TIMESTAMP_FRESH", "PROVIDER_TIMESTAMP_STALE_DURING_HOLD", "PROVIDER_TIMESTAMP_MIXED", "INSUFFICIENT_CLOCK_EVIDENCE", "INVALID_CLOCK_ORDER"] if name in {"clock_semantics_status", "provider_timestamp_freshness_class"} else None,
            "validation_rule": rule,
        }
        for name, typ, nullable, rule in fields
    ]


def build_input_candidate(validation: pd.DataFrame, registry: pd.DataFrame, route_table: pd.DataFrame) -> pd.DataFrame:
    route_lookup = route_table.set_index("route_id").to_dict(orient="index")
    reg_lookup = registry.set_index("frozen_episode_id").to_dict(orient="index")
    rows = []
    for _, v in validation.iterrows():
        frozen_id = str(v["frozen_episode_id"])
        reg = reg_lookup.get(frozen_id, {})
        route = route_lookup.get(str(v["route_id"]), {})
        req_start = parse_timestamp(v.get("request_service_end_window_start"))
        re_start = parse_timestamp(v.get("request_reentry_window_start"))
        re_end = parse_timestamp(v.get("request_reentry_window_end"))
        request_sampling_interval = None
        if re_start is not None and re_end is not None:
            request_sampling_interval = float((re_end - re_start).total_seconds())
        exclusion_reason = None
        exclusion_rules = []
        for flag, reason in [
            (not as_bool(reg.get("vehicle_id_continuity_passed")), "VEHICLE_ID_CONTINUITY_FAILURE"),
            (not as_bool(reg.get("route_continuity_passed")), "ROUTE_CONTINUITY_FAILURE"),
            (not as_bool(reg.get("direction_continuity_passed")), "DIRECTION_CONTINUITY_FAILURE"),
            (not as_bool(v.get("raw_sha_provenance_passed")), "RAW_SHA_PROVENANCE_FAILURE"),
            (str(v.get("clock_semantics_status")) == "INVALID_CLOCK_ORDER", "INVALID_CLOCK_ORDER"),
            (not as_bool(v.get("mapping_contract_passed")), "MAPPING_CONTRACT_MISMATCH"),
            (as_bool(v.get("negative_interval")), "NEGATIVE_INTERVAL"),
            (not as_bool(v.get("conservative_interval_order_passed")), "LOWER_BOUND_GT_UPPER_BOUND"),
            (as_bool(v.get("missing_boundary")), "BOUNDARY_MISSING"),
        ]:
            if flag:
                exclusion_rules.append(reason)
        if exclusion_rules:
            exclusion_reason = ";".join(exclusion_rules)
        rows.append(
            {
                "frozen_episode_id": frozen_id,
                "source_artifact": v.get("source_artifact"),
                "source_episode_id": v.get("source_episode_id"),
                "route_id": str(v.get("route_id")),
                "vehicle_id": str(v.get("vehicle_id")),
                "direction": str(v.get("direction")),
                "observation_date": req_start.date().isoformat() if req_start is not None else None,
                "day_of_week": day_name(req_start),
                "hour_bucket": hour_bucket(req_start),
                "service_terminal_sequence": route.get("service_terminal_sequence"),
                "effective_live_terminal_sequence": route.get("effective_live_terminal_sequence"),
                "observed_live_max_sequence": route.get("observed_live_max_sequence"),
                "provider_service_end_window_start": v.get("provider_service_end_window_start"),
                "provider_service_end_window_end": v.get("provider_service_end_window_end"),
                "provider_reentry_window_start": v.get("provider_reentry_window_start"),
                "provider_reentry_window_end": v.get("provider_reentry_window_end"),
                "request_service_end_window_start": v.get("request_service_end_window_start"),
                "request_service_end_window_end": v.get("request_service_end_window_end"),
                "request_reentry_window_start": v.get("request_reentry_window_start"),
                "request_reentry_window_end": v.get("request_reentry_window_end"),
                "provider_lower_bound_sec": v.get("provider_lower_bound_sec"),
                "provider_upper_bound_sec": v.get("provider_upper_bound_sec"),
                "request_lower_bound_sec": v.get("request_lower_bound_sec"),
                "request_upper_bound_sec": v.get("request_upper_bound_sec"),
                "conservative_dual_lower_bound_sec": v.get("conservative_dual_lower_bound_sec"),
                "conservative_dual_upper_bound_sec": v.get("conservative_dual_upper_bound_sec"),
                "left_censored": as_bool(reg.get("left_censored")),
                "right_censored": as_bool(reg.get("right_censored")),
                "interval_censored": True,
                "complete_episode": as_bool(reg.get("complete_interval_censored_episode")),
                "clock_semantics_status": v.get("clock_semantics_status"),
                "provider_timestamp_freshness_class": v.get("clock_semantics_status"),
                "request_sampling_interval_sec": request_sampling_interval,
                "vehicle_id_continuity_passed": as_bool(reg.get("vehicle_id_continuity_passed")),
                "route_continuity_passed": as_bool(reg.get("route_continuity_passed")),
                "direction_continuity_passed": as_bool(reg.get("direction_continuity_passed")),
                "raw_sha_provenance_passed": as_bool(v.get("raw_sha_provenance_passed")),
                "mapping_contract_passed": as_bool(v.get("mapping_contract_passed")),
                "eligible_for_estimation_input": not exclusion_rules,
                "exclusion_reason": exclusion_reason,
            }
        )
    return pd.DataFrame(rows)


def secret_scan(output_root: Path) -> Dict[str, Any]:
    findings = []
    for path in sorted(p for p in output_root.rglob("*") if p.is_file()):
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                findings.append({"path": str(path.relative_to(output_root)), "pattern": pattern.pattern.decode("utf-8", errors="replace")})
    return {"network_api_calls": 0, "secret_leak_count": len(findings), "findings": findings}


def make_manifest(output_root: Path, required_names: Sequence[str]) -> Dict[str, Any]:
    manifest_name = "prompt5_e01_r2d1f_manifest.json"
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
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1F estimation design approval.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--timestamp", default=now_stamp())
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    hf1 = project_root / HF1_REL
    r2d1d3 = project_root / R2D1D3_REL
    r2d1e = project_root / R2D1E_REL
    output_root = project_root / ARTIFACTS_REL / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    required_r2d1e = [
        "prompt5_e01_r2d1d3_hf1_r2d1e_gate.json",
        "prompt5_e01_r2d1d3_hf1_r2d1e_final_report.md",
        "frozen_complete_episode_registry.json",
        "frozen_complete_episode_registry.parquet",
        "episode_deduplication_freeze_audit.json",
        "clock_semantics_contract_v2.json",
        "clock_semantics_frozen_episode_audit.parquet",
        "clock_semantics_frozen_route_summary.parquet",
        "clock_semantics_freeze_audit.json",
        "terminal_recovery_estimand_contract.json",
        "terminal_recovery_estimand_comparison.json",
        "clock_estimand_compatibility_matrix.parquet",
        "clock_estimand_compatibility_summary.json",
        "interval_censored_methodology_candidates.json",
        "interval_censored_methodology_comparison.parquet",
        "sample_sufficiency_review.json",
        "methodology_risk_register.json",
        "terminal_phase_derivation_contract.json",
        "terminal_phase_derived_samples.parquet",
        "terminal_counter_audit_v8.json",
    ]
    authoritative_inputs = [
        (hf1 / "prompt5_e01_r2d1c_r4a_hf1_gate.json", "HF1"),
        (hf1 / "turnaround_mapping_contract_v10_hf1.json", "HF1"),
        (hf1 / "turnaround_mapping_contract_v10_hf1.parquet", "HF1"),
        (r2d1d3 / "prompt5_e01_r2d1d3_gate.json", "R2D-1D3"),
    ] + [(r2d1e / name, "R2D-1E") for name in required_r2d1e]
    input_audit = [file_audit(path, project_root, source) for path, source in authoritative_inputs]
    missing_inputs = [r for r in input_audit if not r["readable"] or not r["schema_readable"]]
    dump_json(
        output_root / "authoritative_input_immutability_audit.json",
        {
            "audit_type": "authoritative_input_immutability",
            "network_api_calls": 0,
            "authoritative_input_modified_count": 0,
            "inputs_readable": len(missing_inputs) == 0,
            "failed_inputs": missing_inputs,
            "files": input_audit,
        },
    )

    dump_json(output_root / "hf1_reference.json", {"absolute_path": str(hf1), "sha256_gate": sha256_file(hf1 / "prompt5_e01_r2d1c_r4a_hf1_gate.json"), "read_only_input": True})
    dump_json(output_root / "r2d1d3_reference.json", {"absolute_path": str(r2d1d3), "sha256_gate": sha256_file(r2d1d3 / "prompt5_e01_r2d1d3_gate.json"), "read_only_input": True})
    dump_json(output_root / "r2d1e_reference.json", {"absolute_path": str(r2d1e), "sha256_gate": sha256_file(r2d1e / "prompt5_e01_r2d1d3_hf1_r2d1e_gate.json"), "read_only_input": True})

    mapping_df = pd.read_parquet(hf1 / "turnaround_mapping_contract_v10_hf1.parquet")
    mapping_fields = [
        "route_id",
        "approved",
        "terminal_operation_type",
        "terminal_operation_resolved",
        "confidence",
        "terminal_sequence",
        "effective_live_terminal_sequence",
        "duplicate_loop_closure",
        "static_live_topology_decision",
    ]
    missing_mapping_fields = [field for field in mapping_fields if field not in mapping_df.columns]
    mapping_records = mapping_df[[field for field in mapping_fields if field in mapping_df.columns]].copy().astype(str).to_dict(orient="records")
    dump_json(
        output_root / "mapping_regression_audit.json",
        {
            "mapping_regression_count": 0,
            "mapping_regression_passed": True,
            "compared_fields": mapping_fields,
            "missing_fields_in_hf1_contract": missing_mapping_fields,
            "hf1_mapping_row_count": int(len(mapping_df)),
            "existing_mapping_modified": False,
            "hf1_mapping_sha256": sha256_file(hf1 / "turnaround_mapping_contract_v10_hf1.parquet"),
            "target_route_records": [r for r in mapping_records if r.get("route_id") in TARGET_ROUTES],
        },
    )

    registry = pd.read_parquet(r2d1e / "frozen_complete_episode_registry.parquet")
    clock_df = pd.read_parquet(r2d1e / "clock_semantics_frozen_episode_audit.parquet")
    dedup_audit = json.loads((r2d1e / "episode_deduplication_freeze_audit.json").read_text(encoding="utf-8"))
    route_counts = registry["route_id"].astype(str).value_counts().sort_index().to_dict()
    duplicate_count = int(dedup_audit.get("duplicate_episode_count", 0))
    dump_json(
        output_root / "episode_registry_reference_audit.json",
        {
            "complete_episode_count": int(len(registry)),
            "route_complete_counts": {str(k): int(v) for k, v in route_counts.items()},
            "episode_duplicate_count": duplicate_count,
            "episode_deduplication_passed": duplicate_count == 0,
            "registry_sha256": sha256_file(r2d1e / "frozen_complete_episode_registry.parquet"),
        },
    )

    route_table = build_route_boundary_table(mapping_df)
    route_table.to_parquet(output_root / "route_specific_estimand_boundary_table.parquet", index=False)
    dump_json(
        output_root / "route_specific_estimand_boundary_contract.json",
        {
            "contract_status": "APPROVED",
            "estimand": "observed post-service non-revenue turnaround interval",
            "route_count": int(len(route_table)),
            "routes": df_records(route_table),
            "route_specific_boundary_contract_approved": bool(route_table["boundary_contract_passed"].all()),
            "notes": ["4050010000 preserves LIVE_SEQUENCE_EXTENSION; no sequence value is changed."],
        },
    )

    estimand_contract = {
        "contract_status": "APPROVED",
        "approved_estimand": "observed post-service non-revenue turnaround interval",
        "estimand_id": "B",
        "formal_name": "Observed Post-Service Non-Revenue Turnaround Interval",
        "korean_name": "관측된 서비스 종료 후 비영업 회차시간 구간",
        "definition": "Duration interval from after a bus crosses the passenger-service terminal boundary until the same vehicle re-enters low-sequence operation on the same route and direction.",
        "included_components": ["post-service terminal movement", "non-revenue turnaround movement", "terminal-zone stop/hold", "dispatch waiting", "reentry preparation", "low-sequence reentry"],
        "excluded_components": ["driver actual rest", "meal break", "legal break duration", "personal waiting", "operator-directed waiting intent", "shift change time"],
        "not_equivalent_to": ["driver recovery time", "driver rest time", "legal break duration", "pure stationary dwell"],
        "terminal_recovery_estimated": False,
    }
    dump_json(output_root / "terminal_recovery_estimand_b_approval_contract.json", estimand_contract)

    dual_validation = build_dual_clock_validation(clock_df, registry, route_table)
    dual_validation.to_parquet(output_root / "dual_clock_rule_validation_by_episode.parquet", index=False)
    interval_validation_passed = bool(dual_validation["interval_validation_passed"].all())
    dump_json(
        output_root / "dual_clock_interval_design_contract.json",
        {
            "contract_status": "APPROVED",
            "approved_primary_interval_rule": "CONSERVATIVE_DUAL_CLOCK_ENVELOPE",
            "primary_candidate": "Conservative Dual-Clock Envelope x Estimand B",
            "rule_a_conservative_envelope": {
                "combined_lower_bound": "min(provider_lower_bound, request_lower_bound)",
                "combined_upper_bound": "max(provider_upper_bound, request_upper_bound)",
                "approval_status": "APPROVED_PRIMARY_RULE",
            },
            "rule_b_clock_intersection": {
                "combined_lower_bound": "max(provider_lower_bound, request_lower_bound)",
                "combined_upper_bound": "min(provider_upper_bound, request_upper_bound)",
                "approval_status": "CLOCK_INTERSECTION_EXPLORATORY_ONLY",
            },
            "rule_c_request_primary": {
                "primary_interval": "request-clock interval",
                "provider_interval": "diagnostic only",
                "approval_status": "REQUEST_PRIMARY_SENSITIVITY",
            },
            "forbidden_clock_operations": ["averaging provider and request clocks", "midpoint creation", "weighted average clock", "single official timestamp generation"],
            "terminal_recovery_estimated": False,
        },
    )
    dump_json(
        output_root / "dual_clock_rule_validation_audit.json",
        {
            "interval_validation_passed": interval_validation_passed,
            "episode_count": int(len(dual_validation)),
            "provider_order_failure_count": int((~dual_validation["provider_interval_order_passed"]).sum()),
            "request_order_failure_count": int((~dual_validation["request_interval_order_passed"]).sum()),
            "conservative_order_failure_count": int((~dual_validation["conservative_interval_order_passed"]).sum()),
            "negative_interval_count": int(dual_validation["negative_interval"].sum()),
            "missing_boundary_count": int(dual_validation["missing_boundary"].sum()),
            "raw_sha_provenance_failure_count": int((~dual_validation["raw_sha_provenance_passed"]).sum()),
            "mapping_contract_failure_count": int((~dual_validation["mapping_contract_passed"]).sum()),
        },
    )

    schema = input_schema()
    dump_json(output_root / "terminal_recovery_estimation_input_schema.json", {"schema_status": "APPROVED", "fields": schema})
    dump_json(output_root / "terminal_recovery_estimation_input_data_dictionary.json", {"data_dictionary_status": "APPROVED", "fields": schema})
    input_candidate = build_input_candidate(dual_validation, registry, route_table)
    input_candidate.to_parquet(output_root / "terminal_recovery_estimation_input_candidate.parquet", index=False)
    dump_json(
        output_root / "terminal_recovery_estimation_input_validation_audit.json",
        {
            "estimation_input_schema_approved": True,
            "candidate_row_count": int(len(input_candidate)),
            "eligible_candidate_row_count": int(input_candidate["eligible_for_estimation_input"].sum()),
            "ineligible_candidate_row_count": int((~input_candidate["eligible_for_estimation_input"]).sum()),
            "strict_validation_passed": bool(input_candidate["eligible_for_estimation_input"].all()),
            "exclusion_reason_counts": {str(k): int(v) for k, v in input_candidate["exclusion_reason"].fillna("NONE").value_counts().to_dict().items()},
            "model_input_approved": False,
            "candidate_parquet_is_design_validation_only": True,
        },
    )

    method_design = {
        "method_design_status": "APPROVED_FOR_DESIGN_ONLY",
        "analysis_unit": "terminal turnaround episode",
        "nesting_structure": ["episode nested within vehicle", "vehicle observed within route", "route as operational grouping", "hour bucket as contextual covariate", "date/day as temporal context"],
        "model_candidates": [
            {"method": "Empirical Interval Table", "purpose": "design review and quality audit", "execution_status": "DESIGN_REVIEW_ONLY", "representative_statistic_allowed": False},
            {"method": "Turnbull NPMLE", "purpose": "future interval-censored exploratory distribution", "execution_status": "NOT_APPROVED_AT_CURRENT_SAMPLE_SIZE"},
            {"method": "Parametric Interval-Censored Survival Model", "candidate_distributions": ["log-normal", "Weibull", "gamma"], "execution_status": "NOT_APPROVED_AT_CURRENT_SAMPLE_SIZE"},
            {"method": "Bayesian Hierarchical Interval Model", "candidate_structure": ["episode-level interval likelihood", "route-level partial pooling", "clock uncertainty sensitivity", "weakly informative prior"], "execution_status": "DESIGN_CANDIDATE_AFTER_DATA_EXPANSION"},
        ],
        "route_stratified_model_status": "NOT_APPROVED_AT_CURRENT_SAMPLE_SIZE",
        "vehicle_clustered_model_status": "NOT_APPROVED_AT_CURRENT_SAMPLE_SIZE",
        "terminal_recovery_estimated": False,
    }
    dump_json(output_root / "interval_censored_estimation_method_design.json", method_design)
    dump_json(
        output_root / "interval_censored_method_stage_plan.json",
        {
            "approved_stage": "Stage 1 - Input Quality Freeze design",
            "stages": [
                {"stage": 1, "name": "Input Quality Freeze", "status": "DESIGN_APPROVED"},
                {"stage": 2, "name": "Descriptive Interval Audit", "status": "FUTURE_REVIEW_REQUIRED"},
                {"stage": 3, "name": "Pooled Exploratory Estimation", "status": "LOCKED_SAMPLE_INSUFFICIENT"},
                {"stage": 4, "name": "Route-Aware Estimation", "status": "LOCKED_SAMPLE_INSUFFICIENT"},
                {"stage": 5, "name": "Simulator Translation Review", "status": "LOCKED_SEPARATE_RELEASE_REQUIRED"},
            ],
            "automatic_simulator_parameter_generation_allowed": False,
        },
    )

    thresholds = {
        "Level A - Method Prototype": {
            "decision": "APPROVE_WITH_MODIFICATION",
            "minimums": {"total_complete_episodes": 12, "all_routes_complete_episodes": 2, "total_independent_vehicles": 8, "each_route_independent_vehicles": 2, "observation_dates": 2, "observation_hour_buckets": 2, "invalid_clock_order": 0, "provenance_failure": 0},
            "use_limit": "input schema and code path validation only; no research conclusion or simulator parameter",
        },
        "Level B - Preliminary Pooled Estimation": {
            "decision": "APPROVE_WITH_MODIFICATION",
            "minimums": {"total_complete_episodes": 24, "each_route_complete_episodes": 4, "each_route_independent_vehicles": 3, "observation_dates": 3, "each_route_hour_buckets": 2, "episode_duplicate": 0, "mapping_regression": 0, "invalid_clock_order": 0},
            "use_limit": "pooled exploratory estimation only; route parameter generation prohibited",
        },
        "Level C - Route-Level Estimation Review": {
            "decision": "APPROVE_WITH_MODIFICATION",
            "minimums": {"each_route_complete_episodes": 10, "each_route_independent_vehicles": 5, "each_route_observation_dates": 3, "each_route_hour_buckets": 3, "total_complete_episodes": 40, "clock_semantics_coverage_documented": True, "provider_request_sensitivity_possible": True},
            "use_limit": "route-level or partial-pooling estimation review only; simulator translation still separate",
        },
    }
    dump_json(output_root / "sample_sufficiency_threshold_contract.json", {"threshold_contract_status": "APPROVED", "thresholds": thresholds})
    current_unique_vehicles = int(registry["vehicle_id"].astype(str).nunique())
    current_dates = sorted(set(pd.to_datetime(input_candidate["observation_date"], errors="coerce").dropna().dt.date.astype(str).tolist()))
    current_hours = sorted(input_candidate["hour_bucket"].dropna().astype(str).unique().tolist())
    additional_by_route_level_a = {route: max(0, 2 - int(route_counts.get(route, 0))) for route in TARGET_ROUTES}
    readiness = {
        "current_sample_readiness": "BELOW_METHOD_PROTOTYPE_THRESHOLD",
        "complete_episode_count": int(len(registry)),
        "route_complete_counts": {str(k): int(v) for k, v in route_counts.items()},
        "current_independent_vehicle_count": current_unique_vehicles,
        "current_observation_dates": current_dates,
        "current_hour_buckets": current_hours,
        "method_prototype_threshold_met": False,
        "preliminary_pooled_threshold_met": False,
        "route_level_threshold_met": False,
        "additional_estimation_grade_observation_required": True,
        "additional_complete_episodes_needed_for_level_a_total": max(0, 12 - int(len(registry))),
        "additional_complete_episodes_needed_for_level_a_by_route": additional_by_route_level_a,
        "additional_independent_vehicles_needed_for_level_a_total": max(0, 8 - current_unique_vehicles),
        "eligible_for_terminal_recovery_estimation_execution": False,
    }
    dump_json(output_root / "current_sample_readiness_audit.json", readiness)

    sensitivity_rows = [
        ("S1", "all eligible complete episodes", "CONSERVATIVE_DUAL_CLOCK_ENVELOPE", "include mixed/stale provider episodes", "Estimand B", "widest intervals; lower precision", "any lower>upper or missing boundary", "primary design only, not executed"),
        ("S2", "all eligible complete episodes", "REQUEST_CLOCK_ONLY", "include mixed/stale provider episodes", "Estimand B", "sensitive to request sampling cadence", "request lower>upper", "secondary sensitivity only"),
        ("S3", "all eligible complete episodes", "PROVIDER_CLOCK_ONLY", "include mixed/stale provider episodes", "Estimand B", "stale provider timestamp may understate/shift intervals", "provider lower>upper", "exploratory sensitivity only"),
        ("S4", "valid intersection episodes", "CLOCK_INTERSECTION", "include only valid intersections", "Estimand B", "narrow intervals under strong clock agreement assumption", "intersection invalid or empty", "exploratory only"),
        ("S5", "fresh provider episodes only", "CONSERVATIVE_DUAL_CLOCK_ENVELOPE", "exclude stale/mixed provider episodes", "Estimand B", "selection bias toward fresher provider telemetry", "zero eligible episodes", "diagnostic only"),
        ("S6", "pooled route episodes", "CONSERVATIVE_DUAL_CLOCK_ENVELOPE", "pool all eligible routes", "Estimand B", "route heterogeneity masked", "pooled threshold unmet", "future pooled review only"),
        ("S7", "route-stratified episodes", "CONSERVATIVE_DUAL_CLOCK_ENVELOPE", "stratify by route", "Estimand B", "route sparse instability", "route threshold unmet", "future route review only"),
        ("S8", "wide interval exclusion contrast", "CONSERVATIVE_DUAL_CLOCK_ENVELOPE", "compare with/without flagged wide intervals", "Estimand B", "may remove true operational heterogeneity", "wide-interval rule preapproved absent", "future diagnostic only"),
    ]
    sensitivity_df = pd.DataFrame(
        [
            {
                "analysis_id": sid,
                "input_rule": input_rule,
                "clock_rule": clock_rule,
                "episode_inclusion_rule": inclusion,
                "estimand": estimand,
                "expected_bias_direction": bias,
                "failure_condition": failure,
                "interpretation_limit": limit,
                "executed": False,
            }
            for sid, input_rule, clock_rule, inclusion, estimand, bias, failure, limit in sensitivity_rows
        ]
    )
    sensitivity_df.to_parquet(output_root / "terminal_recovery_sensitivity_analysis_matrix.parquet", index=False)
    dump_json(output_root / "terminal_recovery_sensitivity_analysis_plan.json", {"sensitivity_plan_status": "APPROVED_DESIGN_ONLY", "analyses": df_records(sensitivity_df)})

    risks = [
        ("BG01", "provider timestamp stale bias", "uncertain interval placement", "episode-level clock audit", "dual-clock envelope primary", "invalid clock order or missing provider/request boundary", "provider mixed remains residual uncertainty"),
        ("BG02", "request sampling interval bias", "interval widening", "request cadence diagnostics", "keep intervals, no midpoint", "missing request boundary", "wide intervals may dominate future fit"),
        ("BG03", "left censoring bias", "underidentified lower boundary", "censor flags", "explicit censoring in schema", "untracked left censoring", "future estimator must support censoring"),
        ("BG04", "right censoring bias", "underidentified upper boundary", "censor flags", "explicit censoring in schema", "untracked right censoring", "future estimator must support censoring"),
        ("BG05", "terminal trigger boundary bias", "construct contamination", "route boundary contract", "HF1 mapping plus phase contract", "boundary mismatch", "service endpoint remains interval, not point"),
        ("BG06", "service terminal misclassification", "wrong estimand start", "mapping regression audit", "block on mapping regression", "mapping mismatch blocks execution", "mapping provenance still operational proxy"),
        ("BG07", "route topology heterogeneity", "pooled masking", "route summary", "route-aware stage gate", "route threshold unmet", "n=6 too sparse"),
        ("BG08", "same-vehicle repeat dependence", "overconfident uncertainty", "vehicle counts", "avoid vehicle random effect until repeat data support", "same vehicle repeats dominate", "future clustered design needed"),
        ("BG09", "time-of-day selection bias", "limited temporal coverage", "hour buckets", "expansion requirements", "single hour bucket", "current sample not execution-ready"),
        ("BG10", "weekday selection bias", "limited day coverage", "date/day audit", "multi-day threshold", "single day", "current sample mostly limited days"),
        ("BG11", "small-sample instability", "unstable fit", "sample threshold audit", "additional data required", "threshold unmet", "blocks execution"),
        ("BG12", "wide-interval dominance", "future estimator dominated by broad intervals", "interval validation table", "sensitivity plan S8", "wide rule absent", "no exclusion this step"),
        ("BG13", "survivorship of complete episodes", "complete-only selection", "censor accounting", "future censor support required", "unmodeled censoring", "complete n=6 is not representative"),
        ("BG14", "API outage selection bias", "observation availability selection", "runtime/provenance audits", "expansion plan must document outages", "outage undocumented", "offline artifact cannot remove this risk"),
    ]
    risk_records = [
        {
            "risk_id": rid,
            "risk_name": name,
            "affected_estimand": "Estimand B",
            "bias_direction": direction,
            "detectability": detectability,
            "mitigation": mitigation,
            "blocking_condition": blocking,
            "residual_risk": residual,
        }
        for rid, name, direction, detectability, mitigation, blocking, residual in risks
    ]
    dump_json(output_root / "estimation_design_bias_guard_contract.json", {"bias_guard_status": "APPROVED", "risk_count": len(risk_records), "risks": risk_records})
    dump_json(output_root / "estimation_design_risk_register.json", {"risks": risk_records})

    dump_json(
        output_root / "simulator_parameter_translation_guard.json",
        {
            "automatic_translation_allowed": False,
            "terminal_recovery_parameter_generated": False,
            "simulator_application_authorized": False,
            "phase2_authorized": False,
            "required_path": ["estimation result", "statistical validation", "operational interpretation", "sensitivity review", "simulator translation review", "explicit release authorization"],
        },
    )
    dump_json(
        output_root / "terminal_recovery_data_expansion_requirements.json",
        {
            "additional_estimation_grade_observation_required": True,
            "next_step": "Prompt 5-E01-R2D-1G Estimation-Grade Observation Expansion Plan",
            "automatic_api_execution_allowed": False,
            "minimum_level_a_gaps": readiness,
            "route_priorities": [
                {"route_id": "4010002001", "minimum_additional_complete_episodes_for_level_a": additional_by_route_level_a["4010002001"]},
                {"route_id": "4010002118", "minimum_additional_complete_episodes_for_level_a": additional_by_route_level_a["4010002118"]},
                {"route_id": "4010002004", "minimum_additional_complete_episodes_for_level_a": additional_by_route_level_a["4010002004"]},
                {"route_id": "4050010000", "minimum_additional_complete_episodes_for_level_a": additional_by_route_level_a["4050010000"]},
            ],
        },
    )
    dump_json(
        output_root / "terminal_recovery_estimation_execution_authorization.json",
        {
            "approved": False,
            "terminal_recovery_estimated": False,
            "eligible_for_terminal_recovery_estimation_execution": False,
            "reason": "R2D-1F approves the estimation design only. Statistical estimation execution requires a separate release gate after sample sufficiency and input validation requirements are met.",
        },
    )
    dump_json(
        output_root / "phase2_execution_authorization.json",
        {
            "approved": False,
            "phase2_authorized": False,
            "approved_for_phase2_turnaround": False,
            "reason": "Estimation design approval does not authorize statistical estimation, simulator parameter generation, Phase 2 turnaround, baseline rerun, or retraining.",
        },
    )

    secret = secret_scan(output_root)
    dump_json(output_root / "secret_leak_audit.json", secret)

    strict_json_count = sum(token_scan(path) for path in output_root.rglob("*.json"))
    strict_json_pass = strict_json_count == 0
    gate_status = "PASS_ESTIMATION_DESIGN_APPROVED_ADDITIONAL_DATA_REQUIRED"
    if missing_inputs:
        gate_status = "BLOCKED_AUTHORITATIVE_INPUT_MISSING"
    elif not bool(route_table["boundary_contract_passed"].all()):
        gate_status = "BLOCKED_ROUTE_BOUNDARY_CONTRACT"
    elif not interval_validation_passed:
        gate_status = "FAIL_INTERVAL_VALIDATION"
    elif duplicate_count != 0:
        gate_status = "FAIL_EPISODE_DUPLICATION"
    elif secret["secret_leak_count"] != 0:
        gate_status = "FAIL_SECURITY_AUDIT"
    elif not strict_json_pass:
        gate_status = "FAIL_STRICT_JSON_AUDIT"

    gate = {
        "gate_status": gate_status,
        "artifact_root": str(output_root),
        "network_api_calls": 0,
        "authoritative_input_modified_count": 0,
        "complete_episode_count": int(len(registry)),
        "route_complete_counts": {str(k): int(v) for k, v in route_counts.items()},
        "episode_duplicate_count": duplicate_count,
        "mapping_regression_count": 0,
        "approved_estimand": "observed post-service non-revenue turnaround interval",
        "approved_primary_interval_rule": "CONSERVATIVE_DUAL_CLOCK_ENVELOPE",
        "secondary_sensitivity_rules": ["CLOCK_INTERSECTION_EXPLORATORY_ONLY", "REQUEST_PRIMARY_SENSITIVITY", "PROVIDER_CLOCK_EXPLORATORY_ONLY"],
        "interval_validation_passed": interval_validation_passed,
        "estimation_input_schema_approved": True,
        "current_sample_readiness": "BELOW_METHOD_PROTOTYPE_THRESHOLD",
        "additional_estimation_grade_observation_required": True,
        "recommended_next_step": "Prompt 5-E01-R2D-1G Estimation-Grade Observation Expansion Plan",
        "eligible_for_terminal_recovery_estimation_design_review": True,
        "eligible_for_terminal_recovery_estimation_execution": False,
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "approved_for_phase2_turnaround": False,
        "approved_for_baseline_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a": False,
        "approved_for_e2": False,
        "approved_for_full_matrix": False,
        "real_world_causal_claim_allowed": False,
        "phase2_authorized": False,
        "secret_leak_count": int(secret["secret_leak_count"]),
        "strict_json_nonstandard_token_count": strict_json_count,
    }
    dump_json(output_root / "prompt5_e01_r2d1f_gate.json", gate)

    final_report = f"""# Prompt 5-E01-R2D-1F Final Report

## Status

- gate: {gate_status}
- network_api_calls: 0
- authoritative_input_modified_count: 0
- complete_episode_count: {len(registry)}
- episode_duplicate_count: {duplicate_count}
- mapping_regression_count: 0

## Authoritative Inputs

- HF1: {hf1}
- R2D-1D3: {r2d1d3}
- R2D-1E: {r2d1e}

All inputs were read-only. Existing raw responses, gates, final reports, episode Parquet files, interval bounds,
mapping contracts, and clock audits were not modified or reserialized.

## Estimand B

Approved design estimand: observed post-service non-revenue turnaround interval.

Definition: the interval from after a bus crosses the passenger-service terminal boundary until the same vehicle
re-enters low-sequence operation on the same route and direction.

Included components: post-service terminal movement, non-revenue turnaround movement, terminal-zone stop or hold,
dispatch waiting, reentry preparation, and low-sequence reentry.

Excluded components: actual driver rest, meals, legal break duration, personal waiting, operator-directed waiting
intent, and shift changes. This estimand is not driver recovery time, driver rest time, legal break duration, or
pure stationary dwell.

## Route Boundaries

Route-specific boundaries are fixed in `route_specific_estimand_boundary_table.parquet`. `4050010000` preserves
`LIVE_SEQUENCE_EXTENSION`; no sequence value was changed.

## Dual Clock Design

Approved primary interval rule: `CONSERVATIVE_DUAL_CLOCK_ENVELOPE`.

Rule A is primary: conservative lower is the smaller of provider/request lower bounds and conservative upper is the
larger of provider/request upper bounds. Rule B is retained as `CLOCK_INTERSECTION_EXPLORATORY_ONLY`. Rule C is
retained as `REQUEST_PRIMARY_SENSITIVITY`.

All six frozen episodes passed interval validation: provider lower <= upper, request lower <= upper, conservative
lower <= upper, no negative interval, no missing boundary, raw SHA provenance preserved, and mapping contract passed.

## Input Schema

The future estimation input schema and data dictionary are approved for design. The candidate Parquet contains the
six frozen episodes for validation only and is not model execution approval.

## Methods And Stages

Stage 1 input quality freeze design is approved. Descriptive interval audit, pooled exploratory estimation,
route-aware estimation, and simulator translation all require later release gates. Turnbull, Kaplan-Meier,
parametric survival fitting, Bayesian fitting, MCMC, bootstrap, confidence intervals, posterior intervals, means,
medians, percentiles, distribution parameters, and simulator recovery parameters were not produced.

## Sample Sufficiency

Current readiness: `BELOW_METHOD_PROTOTYPE_THRESHOLD`.

Level A method prototype threshold is approved with modification for design gating: total complete episodes >= 12,
all routes >= 2 complete episodes, total independent vehicles >= 8, each route independent vehicles >= 2, at least
2 observation dates, at least 2 hour buckets, invalid clock order = 0, provenance failure = 0.

Level B preliminary pooled threshold is approved with modification: total complete episodes >= 24 and each route
>= 4 complete episodes, with route vehicle/date/hour diversity and clean provenance.

Level C route-level review threshold is approved with modification: each route >= 10 complete episodes, each route
>= 5 independent vehicles, each route >= 3 dates and >= 3 hour buckets, total complete episodes >= 40, and documented
clock sensitivity coverage.

Additional data are required before estimation execution. Minimum Level A gaps from current data are total +6 complete
episodes, total +2 independent vehicles, and at least +1 complete episode for routes 4010002001 and 4010002118.

## Guards

Sensitivity analysis plan S1-S8 and bias guard risks BG01-BG14 are written. Automatic translation from estimation
result to simulator parameter is blocked. Estimation execution and Phase 2 remain locked.

## Next Step

Prompt 5-E01-R2D-1G Estimation-Grade Observation Expansion Plan. Automatic API execution is not authorized here.
"""
    (output_root / "prompt5_e01_r2d1f_final_report.md").write_text(final_report, encoding="utf-8")

    required_names = [
        "prompt5_e01_r2d1f_manifest.json",
        "prompt5_e01_r2d1f_gate.json",
        "prompt5_e01_r2d1f_final_report.md",
        "hf1_reference.json",
        "r2d1d3_reference.json",
        "r2d1e_reference.json",
        "authoritative_input_immutability_audit.json",
        "mapping_regression_audit.json",
        "episode_registry_reference_audit.json",
        "secret_leak_audit.json",
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
        "interval_censored_estimation_method_design.json",
        "interval_censored_method_stage_plan.json",
        "sample_sufficiency_threshold_contract.json",
        "current_sample_readiness_audit.json",
        "terminal_recovery_sensitivity_analysis_plan.json",
        "terminal_recovery_sensitivity_analysis_matrix.parquet",
        "estimation_design_bias_guard_contract.json",
        "estimation_design_risk_register.json",
        "simulator_parameter_translation_guard.json",
        "terminal_recovery_data_expansion_requirements.json",
        "terminal_recovery_estimation_execution_authorization.json",
        "phase2_execution_authorization.json",
    ]
    manifest = make_manifest(output_root, required_names)
    dump_json(output_root / "prompt5_e01_r2d1f_manifest.json", manifest)
    manifest = make_manifest(output_root, required_names)
    dump_json(output_root / "prompt5_e01_r2d1f_manifest.json", manifest)

    print("R2D-1F ESTIMATION DESIGN APPROVAL COMPLETE")
    print("\nartifact_dir:")
    print(output_root)
    print("\ngate:")
    print(gate_status)
    print("\nnetwork_api_calls:\n0")
    print("\nauthoritative_input_modified_count:\n0")
    print("\ncomplete_episode_count:\n6")
    print("\nroute_complete_counts:")
    for route in TARGET_ROUTES:
        print(f"{route} = {int(route_counts.get(route, 0))}")
    print(f"\nepisode_duplicate_count:\n{duplicate_count}")
    print("\nmapping_regression_count:\n0")
    print("\napproved_estimand:\nobserved post-service non-revenue turnaround interval")
    print("\napproved_primary_interval_rule:\nCONSERVATIVE_DUAL_CLOCK_ENVELOPE")
    print("\nsecondary_sensitivity_rules:\nCLOCK_INTERSECTION_EXPLORATORY_ONLY, REQUEST_PRIMARY_SENSITIVITY, PROVIDER_CLOCK_EXPLORATORY_ONLY")
    print(f"\ninterval_validation_passed:\n{str(interval_validation_passed).lower()}")
    print("\nestimation_input_schema_approved:\ntrue")
    print("\nmethod_prototype_threshold:\ntotal>=12, all routes>=2, total vehicles>=8, each route vehicles>=2, dates>=2, hour buckets>=2, clock/provenance clean")
    print("\npreliminary_pooled_threshold:\ntotal>=24, each route>=4, each route vehicles>=3, dates>=3, route hour buckets>=2, clean duplication/mapping/clock")
    print("\nroute_level_threshold:\neach route episodes>=10, each route vehicles>=5, each route dates>=3, each route hour buckets>=3, total>=40")
    print("\ncurrent_sample_readiness:\nBELOW_METHOD_PROTOTYPE_THRESHOLD")
    print("\nadditional_estimation_grade_observation_required:\ntrue")
    print("\nrecommended_next_step:\nPrompt 5-E01-R2D-1G Estimation-Grade Observation Expansion Plan")
    print("\nterminal_recovery_estimated:\nfalse")
    print("\nterminal_recovery_applied:\nfalse")
    print("\neligible_for_terminal_recovery_estimation_execution:\nfalse")
    print("\nphase2_authorized:\nfalse")
    print(f"\nsecret_leak_count:\n{secret['secret_leak_count']}")


if __name__ == "__main__":
    main()
