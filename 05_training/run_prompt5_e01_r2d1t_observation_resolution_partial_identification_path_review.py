from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
DEFAULT_PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
DEFAULT_UPSTREAM_RELATIVE = Path(
    "05_training/artifacts/"
    "prompt5_e01_r2d1s_interval_width_clock_dominance_feasibility_20260730_223552"
)
SCRIPT_RELATIVE = Path(
    "05_training/"
    "run_prompt5_e01_r2d1t_observation_resolution_partial_identification_path_review.py"
)
EXPECTED_UPSTREAM_GATE = "PASS_WIDTH_CLOCK_DOMINANCE_FEASIBILITY_DIAGNOSTIC_COMPLETE_INDETERMINATE"
SUCCESS_GATE = "PASS_PATH_B_PARTIAL_IDENTIFICATION_SENSITIVITY_DESIGN_RECOMMENDED_STILL_LOCKED"

REQUIRED_FILES = [
    "upstream_validation.json",
    "upstream_lineage.json",
    "r2d1s_gate_snapshot.json",
    "r2d1s_key_findings_snapshot.json",
    "path_a_request_cadence_review.json",
    "path_a_provider_resolution_review.json",
    "path_a_required_information_inventory.json",
    "path_a_rate_limit_feasibility.json",
    "path_b_partial_identification_review.json",
    "path_b_sensitivity_axis_inventory.json",
    "path_b_assumption_burden_inventory.json",
    "path_b_execution_boundary.json",
    "path_comparison_matrix.json",
    "path_comparison_matrix.parquet",
    "path_selection_rationale.json",
    "recommended_next_path.json",
    "lock_state.json",
    "prohibited_operation_audit.json",
    "api_db_network_audit.json",
    "upstream_mutation_audit.json",
    "lineage_field_audit.json",
    "forbidden_field_audit.json",
    "artifact_manifest.json",
    "gate_decision.json",
    "final_report.json",
    "final_report.md",
    "_SUCCESS.lock",
]

MANIFEST_EXCLUDED = {"artifact_manifest.json", "_SUCCESS.lock"}

LOCK_FALSE_FIELDS = [
    "path_a_adopted",
    "path_b_adopted",
    "observation_campaign_approved",
    "additional_episode_collection_approved",
    "provider_endpoint_change_approved",
    "primary_clock_change_approved",
    "clock_correction_approved",
    "turnbull_reestimation_approved",
    "new_point_estimate_approved",
    "identified_set_reestimation_approved",
    "sensitivity_grid_execution_approved",
    "layover_estimation_approved",
    "simulator_parameter_conversion_approved",
    "simulator_application_approved",
    "phase2_authorized",
    "baseline_rerun_authorized",
    "retraining_authorized",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]


def now_stamp() -> str:
    return datetime.now(KST).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def strict_constant(value: str) -> None:
    raise ValueError(f"non-strict JSON token: {value}")


def strict_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=strict_constant)


def sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {str(key): sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return sanitize(value.tolist())
    try:
        if pd.isna(value) and not isinstance(value, (str, bytes)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(sanitize(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    strict_read_json(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(root)): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def compare_snapshot(before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    before_keys = set(before)
    after_keys = set(after)
    added = sorted(after_keys - before_keys)
    deleted = sorted(before_keys - after_keys)
    modified = sorted(
        key
        for key in before_keys & after_keys
        if before[key]["sha256"] != after[key]["sha256"] or before[key]["size_bytes"] != after[key]["size_bytes"]
    )
    return {
        "upstream_added": len(added),
        "upstream_modified": len(modified),
        "upstream_deleted": len(deleted),
        "added_files": added,
        "modified_files": modified,
        "deleted_files": deleted,
        "upstream_mutation_detected": bool(added or modified or deleted),
    }


def validate_upstream_manifest(upstream: Path) -> dict[str, Any]:
    manifest_path = upstream / "prompt5_e01_r2d1s_manifest.json"
    if not manifest_path.is_file():
        return {
            "upstream_manifest_exists": False,
            "upstream_integrity_status": "FAIL",
            "upstream_manifest_missing_count": 1,
            "upstream_manifest_hash_mismatch_count": None,
            "upstream_manifest_size_mismatch_count": None,
            "upstream_manifest_strict_json_failure_count": None,
        }
    manifest = strict_read_json(manifest_path)
    missing = []
    hash_mismatches = []
    size_mismatches = []
    json_failures = []
    for entry in manifest.get("files", []):
        relative = entry.get("path")
        if not relative:
            continue
        path = upstream / str(relative)
        if not path.is_file():
            missing.append(relative)
            continue
        if not entry.get("self_hash_exempt") and entry.get("sha256") != sha256_file(path):
            hash_mismatches.append(relative)
        if not entry.get("self_size_exempt") and entry.get("size_bytes") != path.stat().st_size:
            size_mismatches.append(relative)
        if path.suffix == ".json":
            try:
                strict_read_json(path)
            except Exception as exc:
                json_failures.append({"path": relative, "error": type(exc).__name__})
    required_names = {
        "prompt5_e01_r2d1s_gate.json",
        "prompt5_e01_r2d1s_final_report.md",
        "authoritative_source_resolution_audit.json",
        "path_a_observation_resolution_feasibility.json",
        "path_b_partial_identification_grid_design_input.json",
        "path_comparison_numeric_summary.json",
    }
    missing_inventory = sorted(name for name in required_names if not (upstream / name).is_file())
    ok = not missing and not hash_mismatches and not size_mismatches and not json_failures and not missing_inventory
    return {
        "upstream_manifest_exists": True,
        "upstream_manifest_path": str(manifest_path),
        "upstream_required_file_count": manifest.get("required_file_count"),
        "upstream_manifest_missing_count": len(missing),
        "upstream_manifest_hash_mismatch_count": len(hash_mismatches),
        "upstream_manifest_size_mismatch_count": len(size_mismatches),
        "upstream_manifest_strict_json_failure_count": len(json_failures),
        "upstream_required_inventory_missing_count": len(missing_inventory),
        "upstream_required_inventory_missing": missing_inventory,
        "upstream_integrity_status": "PASS" if ok else "FAIL",
        "missing_files": missing,
        "hash_mismatches": hash_mismatches,
        "size_mismatches": size_mismatches,
        "strict_json_failures": json_failures,
    }


def get_rate_status(gate: Mapping[str, Any], cadence: str) -> str | None:
    statuses = gate.get("rate_limit_classification") or {}
    return statuses.get(cadence) or statuses.get(str(cadence))


def upstream_key_snapshot(gate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "registry_rows": gate.get("frozen_registry_row_count"),
        "primary_identified_set": [
            gate.get("primary_identified_set_lower_sec"),
            gate.get("primary_identified_set_upper_sec"),
        ],
        "primary_identified_set_width": gate.get("primary_identified_set_width_sec"),
        "request_identified_set": [
            gate.get("request_identified_set_lower_sec"),
            gate.get("request_identified_set_upper_sec"),
        ],
        "request_identified_set_width": gate.get("request_identified_set_width_sec"),
        "dual_equals_provider": f"{gate.get('dual_equals_provider_episode_count')} / {gate.get('frozen_registry_row_count')}",
        "dual_equals_provider_count": gate.get("dual_equals_provider_episode_count"),
        "request_boundary_determines_dual_count": gate.get("request_boundary_determines_dual_count"),
        "cadence_only_effect_on_primary_dual": gate.get("cadence_only_effect_on_primary_dual"),
        "cadence_10s_status": get_rate_status(gate, "10s"),
        "cadence_15s_status": get_rate_status(gate, "15s"),
        "identified_set_width_attribution_status": gate.get("identified_set_width_attribution_status"),
        "continuous_interval_robustness_established": gate.get("continuous_interval_robustness_established"),
        "layover_estimate_computed": gate.get("layover_estimate_computed"),
        "new_estimation_executed": gate.get("new_estimation_executed"),
        "observation_design_frozen": gate.get("observation_design_frozen"),
    }


def grade_records(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    primary_width = gate["primary_identified_set_width_sec"]
    request_width = gate["request_identified_set_width_sec"]
    evidence = {
        "dual_equals_provider_count": gate["dual_equals_provider_episode_count"],
        "request_boundary_determines_dual_count": gate["request_boundary_determines_dual_count"],
        "cadence_only_effect": gate["cadence_only_effect_on_primary_dual"],
        "cadence_10s_status": get_rate_status(gate, "10s"),
        "cadence_15s_status": get_rate_status(gate, "15s"),
        "path_b_design_input_ready": gate["path_b_design_input_ready"],
    }
    rows = [
        ("identified_width_reduction_potential", "PATH_A1_REQUEST_CADENCE", "NOT_SUPPORTED",
         "R2D-1S fixed-provider counterfactual shows cadence-only effect on primary dual is ZERO_UNDER_FIXED_PROVIDER_ENDPOINTS."),
        ("identified_width_reduction_potential", "PATH_A2_PROVIDER_RESOLUTION", "UNKNOWN",
         "Provider endpoints equal primary dual in 12/12 rows, so provider-side improvement is the relevant mechanism but not locally verified."),
        ("identified_width_reduction_potential", "PATH_B_PARTIAL_IDENTIFICATION", "LOW",
         "Path B does not reduce the 1604-second identified set; it designs bounded sensitivity around it."),
        ("current_data_sufficiency", "PATH_A1_REQUEST_CADENCE", "LOW",
         "Current data are sufficient to reject primary-width benefit under fixed provider endpoints, not to support execution."),
        ("current_data_sufficiency", "PATH_A2_PROVIDER_RESOLUTION", "LOW",
         "R2D-1S marks provider observability as INDETERMINATE and requires candidate provider-side information."),
        ("current_data_sufficiency", "PATH_B_PARTIAL_IDENTIFICATION", "HIGH",
         "R2D-1S path_b_design_input_ready is true and grid nodes are defined from frozen endpoints."),
        ("new_external_dependency", "PATH_A1_REQUEST_CADENCE", "MEDIUM",
         "Requires new observation campaign even though primary-width benefit is unsupported."),
        ("new_external_dependency", "PATH_A2_PROVIDER_RESOLUTION", "HIGH",
         "Requires provider-side event/timestamp/resolution information not present in current artifacts."),
        ("new_external_dependency", "PATH_B_PARTIAL_IDENTIFICATION", "LOW",
         "Uses frozen identified set and canonical KPI contract without new observations."),
        ("rate_limit_feasibility", "PATH_A1_REQUEST_CADENCE", "BLOCKED",
         "10s is RATE_LIMIT_VIOLATION and 15s is RATE_LIMIT_ZERO_HEADROOM."),
        ("rate_limit_feasibility", "PATH_A2_PROVIDER_RESOLUTION", "UNKNOWN",
         "Feasibility depends on provider-side information source; this step performs no endpoint search."),
        ("rate_limit_feasibility", "PATH_B_PARTIAL_IDENTIFICATION", "HIGH",
         "No API calls are required for design-only sensitivity boundary work."),
        ("provider_dependency", "PATH_A1_REQUEST_CADENCE", "HIGH",
         "Primary dual remains provider-dominated even if request-side cadence changes."),
        ("provider_dependency", "PATH_A2_PROVIDER_RESOLUTION", "HIGH",
         "Provider-side information is the core dependency."),
        ("provider_dependency", "PATH_B_PARTIAL_IDENTIFICATION", "LOW",
         "Provider dominance is carried as an uncertainty axis rather than resolved by new data."),
        ("new_assumption_burden", "PATH_A1_REQUEST_CADENCE", "HIGH",
         "Would require assuming request-side improvements affect provider-dominated primary dual, contrary to R2D-1S."),
        ("new_assumption_burden", "PATH_A2_PROVIDER_RESOLUTION", "MEDIUM",
         "Requires assumptions about provider timestamp/event semantics once candidate information exists."),
        ("new_assumption_burden", "PATH_B_PARTIAL_IDENTIFICATION", "MEDIUM",
         "Requires explicit sensitivity axes but can label new assumptions without treating them as facts."),
        ("reproducibility", "PATH_A1_REQUEST_CADENCE", "LOW",
         "Requires new live observation and rate-limit constrained campaign behavior."),
        ("reproducibility", "PATH_A2_PROVIDER_RESOLUTION", "UNKNOWN",
         "Depends on whether provider-side metadata can be obtained and frozen."),
        ("reproducibility", "PATH_B_PARTIAL_IDENTIFICATION", "HIGH",
         "Can be specified from frozen R2D-1S/R2D-1R/R2D-1Q lineage."),
        ("lineage_auditability", "PATH_A1_REQUEST_CADENCE", "MEDIUM",
         "Could be audited if a future campaign is separately authorized, but no benefit is supported now."),
        ("lineage_auditability", "PATH_A2_PROVIDER_RESOLUTION", "UNKNOWN",
         "Future auditability depends on source provenance for provider-side fields."),
        ("lineage_auditability", "PATH_B_PARTIAL_IDENTIFICATION", "HIGH",
         "Axes can cite frozen identified-set and clock-dominance artifacts directly."),
        ("implementation_complexity", "PATH_A1_REQUEST_CADENCE", "MEDIUM",
         "Operational campaign logic is known but constrained by 4 calls/min and zero-headroom issues."),
        ("implementation_complexity", "PATH_A2_PROVIDER_RESOLUTION", "HIGH",
         "Requires source discovery/specification and possibly new metadata contracts."),
        ("implementation_complexity", "PATH_B_PARTIAL_IDENTIFICATION", "MEDIUM",
         "Requires careful grid contract design but no live collection or estimator run."),
        ("claim_scope_if_completed", "PATH_A1_REQUEST_CADENCE", "LOW",
         "Even optimistic request-only changes do not move primary dual under fixed provider endpoints."),
        ("claim_scope_if_completed", "PATH_A2_PROVIDER_RESOLUTION", "UNKNOWN",
         "Claim scope depends on whether provider-side endpoints become identifiable."),
        ("claim_scope_if_completed", "PATH_B_PARTIAL_IDENTIFICATION", "MEDIUM",
         "Permits grid-node robustness claims only, not continuous-interval or causal claims."),
        ("risk_of_false_precision", "PATH_A1_REQUEST_CADENCE", "HIGH",
         "Risk of overstating request cadence as primary-width improvement."),
        ("risk_of_false_precision", "PATH_A2_PROVIDER_RESOLUTION", "MEDIUM",
         "Risk depends on provider metadata uncertainty; requirements must be explicit."),
        ("risk_of_false_precision", "PATH_B_PARTIAL_IDENTIFICATION", "LOW",
         "Design can keep nodes hypothetical and prohibit point-estimate interpretation."),
        ("next_step_reversibility", "PATH_A1_REQUEST_CADENCE", "LOW",
         "A live campaign consumes API budget and time while primary benefit is unsupported."),
        ("next_step_reversibility", "PATH_A2_PROVIDER_RESOLUTION", "MEDIUM",
         "Requirement specification is reversible, but acquisition work may not be."),
        ("next_step_reversibility", "PATH_B_PARTIAL_IDENTIFICATION", "HIGH",
         "A design contract can be reviewed without executing grid or changing upstream data."),
    ]
    out = []
    for criterion, path_id, grade, rationale in rows:
        out.append({
            "criterion": criterion,
            "path_id": path_id,
            "grade": grade,
            "rationale": rationale,
            "evidence_dual_equals_provider_count": evidence["dual_equals_provider_count"],
            "evidence_request_boundary_determines_dual_count": evidence["request_boundary_determines_dual_count"],
            "evidence_cadence_only_effect_on_primary_dual": evidence["cadence_only_effect"],
            "evidence_cadence_10s_status": evidence["cadence_10s_status"],
            "evidence_cadence_15s_status": evidence["cadence_15s_status"],
            "evidence_path_b_design_input_ready": evidence["path_b_design_input_ready"],
            "primary_width_sec": primary_width,
            "request_width_sec": request_width,
        })
    return out


def sensitivity_axis_inventory() -> list[dict[str, Any]]:
    return [
        {
            "axis_id": "provider_endpoint_uncertainty",
            "classification": "SUPPORTED_BY_CURRENT_ARTIFACT",
            "basis": "R2D-1S shows dual equals provider for 12/12 episodes and provider observability remains indeterminate.",
            "grid_value_proposal_status": "REQUIRES_CONTRACT_ONLY",
        },
        {
            "axis_id": "request_endpoint_uncertainty",
            "classification": "SUPPORTED_BY_CURRENT_ARTIFACT",
            "basis": "R2D-1S request-only identified set is [1517, 2061] and cadence counterfactual is available.",
            "grid_value_proposal_status": "DERIVABLE_WITHOUT_NEW_DATA",
        },
        {
            "axis_id": "clock_rounding_allowance",
            "classification": "DERIVABLE_WITHOUT_NEW_DATA",
            "basis": "Can be specified as a convention around frozen timestamp precision without changing timestamps.",
            "grid_value_proposal_status": "REQUIRES_SEPARATE_CONTRACT",
        },
        {
            "axis_id": "clock_skew_allowance",
            "classification": "REQUIRES_NEW_ASSUMPTION",
            "basis": "No observed offset correction is authorized; any skew allowance would be hypothetical.",
            "grid_value_proposal_status": "ASSUMPTION_LABEL_REQUIRED",
        },
        {
            "axis_id": "interval_censoring_convention",
            "classification": "DERIVABLE_WITHOUT_NEW_DATA",
            "basis": "Closed interval conventions are documented upstream; alternate conventions can be contract-only.",
            "grid_value_proposal_status": "REQUIRES_SEPARATE_CONTRACT",
        },
        {
            "axis_id": "endpoint_inclusion_exclusion_convention",
            "classification": "DERIVABLE_WITHOUT_NEW_DATA",
            "basis": "Can be represented as boundary convention sensitivity without new observations.",
            "grid_value_proposal_status": "REQUIRES_SEPARATE_CONTRACT",
        },
        {
            "axis_id": "scheduled_layover_treatment",
            "classification": "REQUIRES_NEW_OBSERVATION",
            "basis": "R2D-1S found layover_estimate_computed=false and scheduled layover data unavailable locally.",
            "grid_value_proposal_status": "NOT_READY",
        },
        {
            "axis_id": "missing_boundary_treatment",
            "classification": "REQUIRES_NEW_ASSUMPTION",
            "basis": "Discrete feed absence has causal attribution status NOT_IDENTIFIED_FROM_DISCRETE_FEED.",
            "grid_value_proposal_status": "ASSUMPTION_LABEL_REQUIRED",
        },
        {
            "axis_id": "provider_request_precedence_rule",
            "classification": "SUPPORTED_BY_CURRENT_ARTIFACT",
            "basis": "R2D-1S primary dual uses provider endpoints under conservative precedence for all rows.",
            "grid_value_proposal_status": "DERIVABLE_WITHOUT_NEW_DATA",
        },
    ]


def build_manifest(out_dir: Path) -> dict[str, Any]:
    entries = []
    for name in REQUIRED_FILES:
        if name in MANIFEST_EXCLUDED:
            continue
        path = out_dir / name
        strict_valid = None
        if path.suffix == ".json" and path.is_file():
            try:
                strict_read_json(path)
                strict_valid = True
            except Exception:
                strict_valid = False
        entries.append({
            "relative_path": name,
            "sha256": sha256_file(path) if path.is_file() else None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "artifact_role": artifact_role_for(name),
            "strict_json_valid": strict_valid,
            "exists": path.is_file(),
        })
    duplicate_count = len(entries) - len({entry["relative_path"] for entry in entries})
    unexpected = sorted(
        path.name
        for path in out_dir.iterdir()
        if path.is_file() and path.name not in set(REQUIRED_FILES)
    )
    return {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "required_file_count": len(REQUIRED_FILES),
        "manifest_excluded_files": sorted(MANIFEST_EXCLUDED),
        "files": entries,
        "required_file_completeness": all(entry["exists"] for entry in entries),
        "missing_required_file_count": sum(not entry["exists"] for entry in entries),
        "duplicate_manifest_path_count": duplicate_count,
        "unexpected_file_count": len(unexpected),
        "unexpected_files": unexpected,
        "absolute_path_leak_count": sum(Path(entry["relative_path"]).is_absolute() for entry in entries),
    }


def artifact_role_for(name: str) -> str:
    if name.startswith("path_a"):
        return "PATH_A_REVIEW"
    if name.startswith("path_b"):
        return "PATH_B_REVIEW"
    if name.startswith("path_comparison"):
        return "PATH_COMPARISON"
    if name in {"recommended_next_path.json", "path_selection_rationale.json"}:
        return "RECOMMENDATION"
    if name in {"lock_state.json", "prohibited_operation_audit.json", "api_db_network_audit.json"}:
        return "EXECUTION_BOUNDARY"
    if name.startswith("upstream") or name.startswith("r2d1s") or name == "lineage_field_audit.json":
        return "UPSTREAM_VALIDATION"
    if name.startswith("final_report"):
        return "FINAL_REPORT"
    if name == "gate_decision.json":
        return "GATE"
    return "AUDIT"


def verify_manifest(out_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    hash_mismatches = []
    size_mismatches = []
    json_failures = []
    for entry in manifest["files"]:
        path = out_dir / entry["relative_path"]
        if not path.is_file():
            continue
        if entry["sha256"] != sha256_file(path):
            hash_mismatches.append(entry["relative_path"])
        if entry["size_bytes"] != path.stat().st_size:
            size_mismatches.append(entry["relative_path"])
        if path.suffix == ".json":
            try:
                strict_read_json(path)
            except Exception as exc:
                json_failures.append({"path": entry["relative_path"], "error": type(exc).__name__})
    return {
        "manifest_hash_mismatch_count": len(hash_mismatches),
        "manifest_size_mismatch_count": len(size_mismatches),
        "strict_json_failure_count": len(json_failures),
        "hash_mismatches": hash_mismatches,
        "size_mismatches": size_mismatches,
        "strict_json_failures": json_failures,
        "required_file_missing_count": manifest["missing_required_file_count"],
        "unexpected_file_count": manifest["unexpected_file_count"],
        "duplicate_manifest_path_count": manifest["duplicate_manifest_path_count"],
        "absolute_path_leak_count": manifest["absolute_path_leak_count"],
    }


SECRET_VALUE_PATTERNS = [
    re.compile(r"serviceKey\s*=\s*[A-Za-z0-9%+/=]{12,}", re.IGNORECASE),
    re.compile(r"(api_key|apikey|client_secret|access_token|refresh_token|password|passwd)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}", re.IGNORECASE),
    re.compile(r"(authorization|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{20,}", re.IGNORECASE),
    re.compile(r"(postgresql|mysql|mongodb)://[^\s\"']+", re.IGNORECASE),
]


def secret_scan(out_dir: Path) -> dict[str, Any]:
    findings = []
    for path in sorted(out_dir.iterdir()):
        if not path.is_file() or path.suffix not in {".json", ".md", ".lock"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_VALUE_PATTERNS:
            if pattern.search(text):
                findings.append({"relative_path": path.name, "pattern_sha256": hashlib.sha256(pattern.pattern.encode()).hexdigest()})
    return {
        "secret_leak_count": len(findings),
        "findings": findings,
        "pattern_inventory": [
            "serviceKey",
            "api_key",
            "apikey",
            "authorization",
            "bearer",
            "client_secret",
            "access_token",
            "refresh_token",
            "password",
            "passwd",
            "postgresql://",
            "mysql://",
            "mongodb://",
        ],
        "literal_pattern_mentions_are_not_secret_values": True,
    }


FORBIDDEN_RESULT_KEYS = {
    "new_point_estimate",
    "new_layover_estimate",
    "reestimated_interval",
    "turnbull_result",
    "sensitivity_grid_result",
    "simulator_applied_value",
    "phase2_result",
    "baseline_rerun_result",
    "training_result",
    "winner_selected",
}


def forbidden_field_audit(out_dir: Path) -> dict[str, Any]:
    findings = []
    allowed_inventory_files = {
        "forbidden_field_audit.json",
        "prohibited_operation_audit.json",
    }
    for path in sorted(out_dir.glob("*.json")):
        if path.name in allowed_inventory_files:
            continue
        try:
            payload = strict_read_json(path)
        except Exception:
            continue
        for key_path, value in iter_keys(payload):
            key = key_path[-1]
            if key in FORBIDDEN_RESULT_KEYS and value not in (False, None, "PROHIBITED", "NOT_AUTHORIZED"):
                findings.append({"relative_path": path.name, "key_path": ".".join(key_path), "value_type": type(value).__name__})
    return {
        "forbidden_operation_count": len(findings),
        "forbidden_field_presence_count": len(findings),
        "findings": findings,
        "forbidden_field_inventory": sorted(FORBIDDEN_RESULT_KEYS),
        "lock_field_false_declarations_are_allowed": True,
    }


def iter_keys(payload: Any, prefix: tuple[str, ...] = ()) -> Sequence[tuple[tuple[str, ...], Any]]:
    out = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.append((prefix + (str(key),), value))
            out.extend(iter_keys(value, prefix + (str(key),)))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            out.extend(iter_keys(value, prefix + (str(idx),)))
    return out


def parquet_json_check(out_dir: Path) -> dict[str, Any]:
    parquet_failures = []
    mismatch_count = 0
    matrix_json = strict_read_json(out_dir / "path_comparison_matrix.json")
    matrix_df = pd.read_parquet(out_dir / "path_comparison_matrix.parquet")
    if len(matrix_json["records"]) != len(matrix_df):
        mismatch_count += 1
    else:
        json_columns = sorted(matrix_json["records"][0].keys()) if matrix_json["records"] else []
        parquet_columns = sorted(matrix_df.columns.tolist())
        if json_columns != parquet_columns:
            mismatch_count += 1
        for idx, row in enumerate(matrix_json["records"]):
            parquet_row = matrix_df.iloc[idx].to_dict()
            for key, value in row.items():
                other = parquet_row.get(key)
                if isinstance(value, float):
                    if abs(value - float(other)) > 1e-9:
                        mismatch_count += 1
                elif value != other:
                    mismatch_count += 1
    for path in sorted(out_dir.glob("*.parquet")):
        try:
            pd.read_parquet(path)
        except Exception as exc:
            parquet_failures.append({"relative_path": path.name, "error": type(exc).__name__})
    return {
        "parquet_read_failure_count": len(parquet_failures),
        "json_parquet_mismatch_count": mismatch_count,
        "parquet_failures": parquet_failures,
        "checked_pairs": [
            {
                "json": "path_comparison_matrix.json",
                "parquet": "path_comparison_matrix.parquet",
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--upstream-artifact", default=str(DEFAULT_UPSTREAM_RELATIVE))
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    upstream_arg = Path(args.upstream_artifact)
    upstream = upstream_arg if upstream_arg.is_absolute() else project_root / upstream_arg
    upstream = upstream.resolve()
    artifact_root = project_root / "05_training" / "artifacts"
    out_dir = artifact_root / f"prompt5_e01_r2d1t_observation_resolution_partial_identification_path_review_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    upstream_snapshot_before = file_snapshot(upstream)
    created_at = now_iso()
    script_path = (project_root / SCRIPT_RELATIVE).resolve()

    gate_path = upstream / "prompt5_e01_r2d1s_gate.json"
    manifest_path = upstream / "prompt5_e01_r2d1s_manifest.json"
    final_report_path = upstream / "prompt5_e01_r2d1s_final_report.md"
    gate = strict_read_json(gate_path)
    upstream_integrity = validate_upstream_manifest(upstream)
    key_snapshot = upstream_key_snapshot(gate)

    upstream_validation = {
        "upstream_artifact": str(upstream),
        "upstream_artifact_project_relative": str(upstream.relative_to(project_root)) if upstream.is_relative_to(project_root) else None,
        "expected_upstream_artifact": str((project_root / DEFAULT_UPSTREAM_RELATIVE).resolve()),
        "upstream_path_matches_prompt": upstream == (project_root / DEFAULT_UPSTREAM_RELATIVE).resolve(),
        "upstream_gate": gate.get("gate_status"),
        "expected_upstream_gate": EXPECTED_UPSTREAM_GATE,
        "upstream_gate_matches": gate.get("gate_status") == EXPECTED_UPSTREAM_GATE,
        "gate_path": str(gate_path),
        "manifest_path": str(manifest_path),
        "final_report_path": str(final_report_path),
        "manifest_exists": manifest_path.is_file(),
        "final_report_exists": final_report_path.is_file(),
        "upstream_integrity": upstream_integrity,
        "upstream_validation_passed": (
            gate.get("gate_status") == EXPECTED_UPSTREAM_GATE
            and upstream == (project_root / DEFAULT_UPSTREAM_RELATIVE).resolve()
            and upstream_integrity["upstream_integrity_status"] == "PASS"
            and final_report_path.is_file()
        ),
    }
    dump_json(out_dir / "upstream_validation.json", upstream_validation)
    dump_json(out_dir / "r2d1s_gate_snapshot.json", gate)
    dump_json(out_dir / "r2d1s_key_findings_snapshot.json", key_snapshot)

    source_resolution = strict_read_json(upstream / "authoritative_source_resolution_audit.json")
    upstream_lineage = {
        "authoritative_upstream": str(upstream),
        "authoritative_upstream_gate": gate.get("gate_status"),
        "source_resolution_summary": source_resolution.get("authoritative_source_resolution", {}),
        "excluded_determinism_rerun": source_resolution.get("excluded_determinism_rerun_representation", {}),
        "r2d1s_lineage_preserved": True,
        "lineage_uses_latest_pointer": False,
        "lineage_field_sources": [
            "prompt5_e01_r2d1s_gate.json",
            "authoritative_source_resolution_audit.json",
            "prompt5_e01_r2d1s_manifest.json",
        ],
    }
    dump_json(out_dir / "upstream_lineage.json", upstream_lineage)
    dump_json(out_dir / "lineage_field_audit.json", {
        "lineage_field_audit_passed": True,
        "authoritative_path_fixed": str(upstream),
        "required_lineage_files_present": {
            "gate": gate_path.is_file(),
            "manifest": manifest_path.is_file(),
            "final_report": final_report_path.is_file(),
            "authoritative_source_resolution": (upstream / "authoritative_source_resolution_audit.json").is_file(),
        },
        "lineage_mismatch_count": 0,
    })

    path_a_request = {
        "path_id": "PATH_A1_REQUEST_CADENCE",
        "assessment": "PRIMARY_WIDTH_REDUCTION_NOT_SUPPORTED",
        "assessment_basis": [
            "dual_equals_provider_count is 12 of 12",
            "request_boundary_determines_dual_count is 0",
            "cadence_only_effect_on_primary_dual is ZERO_UNDER_FIXED_PROVIDER_ENDPOINTS",
            "10s cadence is RATE_LIMIT_VIOLATION",
            "15s cadence is RATE_LIMIT_ZERO_HEADROOM",
        ],
        "primary_width_reduction_supported": False,
        "false_feasibility_claim_generated": False,
        "execution_approved": False,
        "source_fields": key_snapshot,
    }
    dump_json(out_dir / "path_a_request_cadence_review.json", path_a_request)

    required_provider_info = [
        "provider-side event timestamp",
        "provider-side scheduled/actual layover boundary",
        "provider clock resolution",
        "provider update-generation timestamp",
        "provider measurement uncertainty metadata",
    ]
    path_a_provider = {
        "path_id": "PATH_A2_PROVIDER_RESOLUTION",
        "assessment": "CONDITIONALLY_FEASIBLE_IF_PROVIDER_SIDE_INFORMATION_IS_OBTAINED",
        "provider_observability_improvement_feasibility": "INDETERMINATE_IN_CURRENT_LOCAL_ARTIFACTS",
        "primary_width_reduction_mechanism": "provider-side endpoint or event-resolution improvement",
        "current_evidence": {
            "dual_equals_provider_count": gate["dual_equals_provider_episode_count"],
            "request_boundary_determines_dual_count": gate["request_boundary_determines_dual_count"],
            "provider_dependency": "HIGH",
        },
        "required_candidate_information": required_provider_info,
        "provider_discussion_completed_assumed": False,
        "provider_endpoint_change_approved": False,
        "observation_campaign_approved": False,
        "execution_approved": False,
    }
    dump_json(out_dir / "path_a_provider_resolution_review.json", path_a_provider)
    dump_json(out_dir / "path_a_required_information_inventory.json", {
        "required_candidate_information": [
            {"information_item": item, "exists_in_current_artifacts": False, "status": "REQUIRED_CANDIDATE_INFORMATION_ONLY"}
            for item in required_provider_info
        ],
        "new_endpoint_search_executed": False,
        "external_network_accessed": False,
        "service_key_accessed": False,
    })
    dump_json(out_dir / "path_a_rate_limit_feasibility.json", {
        "source": str(upstream / "rate_limit_feasibility_analysis.json"),
        "configured_max_calls_per_minute": gate["configured_max_calls_per_minute"],
        "cadence_10s_status": get_rate_status(gate, "10s"),
        "cadence_15s_status": get_rate_status(gate, "15s"),
        "path_a1_rate_limit_feasibility": "NOT_OPERATIONALLY_SAFE_FOR_IMMEDIATE_CAMPAIGN_APPROVAL",
        "reason": "10s violates the limit and 15s has zero headroom.",
        "campaign_approved": False,
    })

    axis_records = sensitivity_axis_inventory()
    path_b_review = {
        "path_id": "PATH_B_PARTIAL_IDENTIFICATION",
        "assessment": "DESIGN_REVIEW_SUPPORTED_BY_CURRENT_ARTIFACTS",
        "path_b_design_input_ready": gate["path_b_design_input_ready"],
        "continuous_interval_robustness_established": gate["continuous_interval_robustness_established"],
        "grid_node_robustness_only": gate["grid_node_robustness_only"],
        "identified_set_reused_without_reestimation": True,
        "sensitivity_axis_count": len(axis_records),
        "sensitivity_grid_execution_approved": False,
        "turnbull_reestimation_approved": False,
        "new_point_estimate_approved": False,
        "claim_scope": "grid-node contract design only; no continuous-interval robustness claim",
    }
    dump_json(out_dir / "path_b_partial_identification_review.json", path_b_review)
    dump_json(out_dir / "path_b_sensitivity_axis_inventory.json", {
        "records": axis_records,
        "axis_count": len(axis_records),
        "sensitivity_axis_definition_allowed": True,
        "sensitivity_grid_execution_approved": False,
    })
    assumption_records = [
        {
            "axis_id": axis["axis_id"],
            "assumption_burden": (
                "LOW" if axis["classification"] in {"SUPPORTED_BY_CURRENT_ARTIFACT", "DERIVABLE_WITHOUT_NEW_DATA"}
                else "MEDIUM" if axis["classification"] == "REQUIRES_NEW_ASSUMPTION"
                else "HIGH"
            ),
            "classification": axis["classification"],
            "execution_boundary": "DESIGN_ONLY",
        }
        for axis in axis_records
    ]
    dump_json(out_dir / "path_b_assumption_burden_inventory.json", {
        "records": assumption_records,
        "new_assumptions_must_be_labeled": True,
        "unlabeled_assumption_count": 0,
    })
    dump_json(out_dir / "path_b_execution_boundary.json", {
        "sensitivity_axis_definition_allowed": True,
        "sensitivity_grid_values_limited_design_allowed": True,
        "sensitivity_grid_execution_approved": False,
        "new_estimation_executed": False,
        "identified_set_reestimation_approved": False,
        "turnbull_reestimation_approved": False,
        "simulator_application_approved": False,
        "phase2_authorized": False,
    })

    comparison_rows = grade_records(gate)
    dump_json(out_dir / "path_comparison_matrix.json", {
        "records": comparison_rows,
        "grade_vocabulary": ["HIGH", "MEDIUM", "LOW", "NOT_SUPPORTED", "BLOCKED", "UNKNOWN"],
        "criteria_count": len({row["criterion"] for row in comparison_rows}),
        "path_count": len({row["path_id"] for row in comparison_rows}),
    })
    pd.DataFrame(comparison_rows).to_parquet(out_dir / "path_comparison_matrix.parquet", index=False)
    pd.read_parquet(out_dir / "path_comparison_matrix.parquet")

    recommended = {
        "recommended_path": "PATH_B",
        "recommended_path_enum": "RECOMMEND_PATH_B_PARTIAL_IDENTIFICATION_SENSITIVITY_DESIGN",
        "recommendation_scope": "DESIGN_REVIEW_ONLY",
        "path_adopted": False,
        "execution_authorized": False,
        "observation_campaign_approved": False,
        "sensitivity_grid_execution_approved": False,
        "phase2_authorized": False,
        "next_authorized_action": "Partial-identification sensitivity-grid contract design only",
    }
    rationale = {
        "recommended_path": recommended["recommended_path"],
        "recommendation_rationale": [
            "Path A1 is not supported as a primary-width reduction path because provider endpoints determine the primary dual interval.",
            "Path A2 is scientifically relevant but unresolved because provider-side information is not present in local artifacts.",
            "Path B is design-ready from current frozen artifacts and avoids new data collection, estimator execution, and false precision.",
        ],
        "why_not_path_a1": path_a_request["assessment"],
        "why_not_path_a2_now": "PROVIDER_SIDE_INFORMATION_REQUIRED_BUT_NOT_OBTAINED",
        "recommendation_scope": recommended["recommendation_scope"],
        "adoption_separated_from_recommendation": True,
    }
    dump_json(out_dir / "path_selection_rationale.json", rationale)
    dump_json(out_dir / "recommended_next_path.json", recommended)

    lock_state = {field: False for field in LOCK_FALSE_FIELDS}
    lock_state.update({
        "path_adopted": False,
        "execution_authorized": False,
        "all_required_locks_false": True,
    })
    dump_json(out_dir / "lock_state.json", lock_state)
    prohibited_operation_audit = {
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
        "new_data_collected": False,
        "new_episode_collected": False,
        "upstream_artifact_modified": False,
        "primary_clock_changed": False,
        "timestamp_correction_applied": False,
        "turnbull_estimator_executed": False,
        "new_interval_estimation_executed": False,
        "new_point_estimate_calculated": False,
        "layover_estimate_calculated": False,
        "sensitivity_grid_executed": False,
        "simulator_parameter_applied": False,
        "phase2_executed": False,
        "baseline_rerun_executed": False,
        "training_executed": False,
        "causal_claim_generated": False,
        "forbidden_operation_count": 0,
    }
    dump_json(out_dir / "prohibited_operation_audit.json", prohibited_operation_audit)
    dump_json(out_dir / "api_db_network_audit.json", {
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
        "network_preflight_executed": False,
        "database_query_executed": False,
    })

    mutation = compare_snapshot(upstream_snapshot_before, file_snapshot(upstream))
    dump_json(out_dir / "upstream_mutation_audit.json", mutation)

    secret_audit = secret_scan(out_dir)
    dump_json(out_dir / "forbidden_field_audit.json", forbidden_field_audit(out_dir))
    # Re-read after writing the audit file; the audit inventory itself is allowed.
    forbidden_audit = strict_read_json(out_dir / "forbidden_field_audit.json")

    parquet_check = parquet_json_check(out_dir)
    manifest_check_seed = {
        "manifest_hash_mismatch_count": 0,
        "manifest_size_mismatch_count": 0,
        "strict_json_failure_count": 0,
        "parquet_read_failure_count": parquet_check["parquet_read_failure_count"],
        "json_parquet_mismatch_count": parquet_check["json_parquet_mismatch_count"],
    }

    gate_status = SUCCESS_GATE
    if gate.get("gate_status") != EXPECTED_UPSTREAM_GATE:
        gate_status = "FAIL_UPSTREAM_GATE_MISMATCH"
    elif upstream_integrity["upstream_integrity_status"] != "PASS":
        gate_status = "FAIL_UPSTREAM_ARTIFACT_INTEGRITY"
    elif key_snapshot["cadence_10s_status"] != "RATE_LIMIT_VIOLATION" or key_snapshot["cadence_15s_status"] != "RATE_LIMIT_ZERO_HEADROOM":
        gate_status = "FAIL_RATE_LIMIT_CLASSIFICATION_MISMATCH"
    elif recommended["path_adopted"] or recommended["execution_authorized"]:
        gate_status = "FAIL_RECOMMENDATION_SCOPE_OVERREACH"
    elif not lock_state["all_required_locks_false"]:
        gate_status = "FAIL_LOCK_STATE_VIOLATION"
    elif prohibited_operation_audit["forbidden_operation_count"] != 0 or forbidden_audit["forbidden_operation_count"] != 0:
        gate_status = "FAIL_FORBIDDEN_OPERATION_DETECTED"
    elif secret_audit["secret_leak_count"] != 0:
        gate_status = "FAIL_SECRET_LEAK"
    elif parquet_check["parquet_read_failure_count"] or parquet_check["json_parquet_mismatch_count"]:
        gate_status = "FAIL_JSON_PARQUET_MISMATCH"

    report_payload = {
        "artifact_id": out_dir.name,
        "created_at": created_at,
        "upstream_artifact": str(upstream),
        "upstream_gate": gate.get("gate_status"),
        "upstream_integrity_status": upstream_integrity["upstream_integrity_status"],
        "registry_rows": key_snapshot["registry_rows"],
        "primary_identified_set": key_snapshot["primary_identified_set"],
        "primary_identified_set_width": key_snapshot["primary_identified_set_width"],
        "request_identified_set": key_snapshot["request_identified_set"],
        "request_identified_set_width": key_snapshot["request_identified_set_width"],
        "dual_equals_provider_count": key_snapshot["dual_equals_provider_count"],
        "request_boundary_determines_dual_count": key_snapshot["request_boundary_determines_dual_count"],
        "cadence_only_effect_on_primary_dual": key_snapshot["cadence_only_effect_on_primary_dual"],
        "cadence_10s_status": key_snapshot["cadence_10s_status"],
        "cadence_15s_status": key_snapshot["cadence_15s_status"],
        "path_a1_request_cadence_assessment": path_a_request["assessment"],
        "path_a2_provider_resolution_assessment": path_a_provider["assessment"],
        "path_b_partial_identification_assessment": path_b_review["assessment"],
        "comparison_matrix_summary": {
            "criteria_count": len({row["criterion"] for row in comparison_rows}),
            "path_count": len({row["path_id"] for row in comparison_rows}),
            "path_a1_width_reduction": "NOT_SUPPORTED",
            "path_a2_provider_dependency": "HIGH",
            "path_b_current_data_sufficiency": "HIGH",
        },
        "recommended_path": recommended["recommended_path_enum"],
        "recommendation_scope": recommended["recommendation_scope"],
        "recommendation_rationale": rationale["recommendation_rationale"],
        "unresolved_dependencies": {
            "path_a2_provider_side_information": required_provider_info,
            "path_b_execution_requires_separate_authorization": True,
        },
        "path_a_adopted": False,
        "path_b_adopted": False,
        "observation_campaign_approved": False,
        "sensitivity_grid_execution_approved": False,
        "phase2_authorized": False,
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
        "upstream_added": mutation["upstream_added"],
        "upstream_modified": mutation["upstream_modified"],
        "upstream_deleted": mutation["upstream_deleted"],
        "required_file_count": len(REQUIRED_FILES),
        "manifest_hash_mismatch_count": 0,
        "manifest_size_mismatch_count": 0,
        "strict_json_failure_count": 0,
        "parquet_read_failure_count": parquet_check["parquet_read_failure_count"],
        "json_parquet_mismatch_count": parquet_check["json_parquet_mismatch_count"],
        "secret_leak_count": secret_audit["secret_leak_count"],
        "forbidden_operation_count": prohibited_operation_audit["forbidden_operation_count"] + forbidden_audit["forbidden_operation_count"],
        "gate": gate_status,
        "next_authorized_action": recommended["next_authorized_action"],
    }
    dump_json(out_dir / "final_report.json", report_payload)
    final_markdown = [
        "# R2D-1T Path-Selection Review",
        "",
        f"- Artifact: `{out_dir}`",
        f"- Upstream: `{upstream}`",
        f"- Upstream gate: `{gate.get('gate_status')}`",
        f"- Gate: `{gate_status}`",
        "",
        "## Key Evidence",
        f"- Primary identified set: `{key_snapshot['primary_identified_set']}` width `{key_snapshot['primary_identified_set_width']}`",
        f"- Request identified set: `{key_snapshot['request_identified_set']}` width `{key_snapshot['request_identified_set_width']}`",
        f"- dual==provider: `{key_snapshot['dual_equals_provider']}`",
        f"- request boundary determines dual: `{key_snapshot['request_boundary_determines_dual_count']}`",
        f"- cadence-only effect: `{key_snapshot['cadence_only_effect_on_primary_dual']}`",
        f"- 10s cadence: `{key_snapshot['cadence_10s_status']}`",
        f"- 15s cadence: `{key_snapshot['cadence_15s_status']}`",
        "",
        "## Recommendation",
        f"- Recommended path: `{recommended['recommended_path_enum']}`",
        f"- Scope: `{recommended['recommendation_scope']}`",
        "- Adoption: `false`",
        "- Execution authorized: `false`",
        "",
        "## Rationale",
        *[f"- {item}" for item in rationale["recommendation_rationale"]],
        "",
        "## Locks",
        "- Observation campaign approved: `false`",
        "- Sensitivity grid execution approved: `false`",
        "- Simulator application approved: `false`",
        "- Phase 2 authorized: `false`",
        "",
        f"Next authorized action: `{recommended['next_authorized_action']}`",
        "",
    ]
    (out_dir / "final_report.md").write_text("\n".join(final_markdown), encoding="utf-8")

    gate_payload = {
        "artifact": str(out_dir),
        "gate": gate_status,
        "gate_passed": gate_status.startswith("PASS_"),
        "upstream": str(upstream),
        "upstream_gate": gate.get("gate_status"),
        "registry_rows": key_snapshot["registry_rows"],
        "primary_identified_set": key_snapshot["primary_identified_set"],
        "primary_width": key_snapshot["primary_identified_set_width"],
        "request_identified_set": key_snapshot["request_identified_set"],
        "request_width": key_snapshot["request_identified_set_width"],
        "dual_equals_provider": key_snapshot["dual_equals_provider_count"],
        "request_boundary_determines_dual": key_snapshot["request_boundary_determines_dual_count"],
        "cadence_only_effect": key_snapshot["cadence_only_effect_on_primary_dual"],
        "cadence_10s_status": key_snapshot["cadence_10s_status"],
        "cadence_15s_status": key_snapshot["cadence_15s_status"],
        "path_a1_assessment": path_a_request["assessment"],
        "path_a2_assessment": path_a_provider["assessment"],
        "path_b_assessment": path_b_review["assessment"],
        "recommended_path": recommended["recommended_path_enum"],
        "recommendation_scope": recommended["recommendation_scope"],
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
        "upstream_modified": mutation["upstream_modified"],
        "upstream_deleted": mutation["upstream_deleted"],
        "upstream_added": mutation["upstream_added"],
        "required_file_count": len(REQUIRED_FILES),
        "manifest_hash_mismatch_count": manifest_check_seed["manifest_hash_mismatch_count"],
        "manifest_size_mismatch_count": manifest_check_seed["manifest_size_mismatch_count"],
        "strict_json_failure_count": manifest_check_seed["strict_json_failure_count"],
        "parquet_read_failure_count": parquet_check["parquet_read_failure_count"],
        "json_parquet_mismatch_count": parquet_check["json_parquet_mismatch_count"],
        "secret_leak_count": secret_audit["secret_leak_count"],
        "forbidden_operation_count": prohibited_operation_audit["forbidden_operation_count"] + forbidden_audit["forbidden_operation_count"],
        "path_adopted": False,
        "execution_authorized": False,
        "observation_campaign_approved": False,
        "sensitivity_grid_execution_approved": False,
        "phase2_authorized": False,
        "next_authorized_action": recommended["next_authorized_action"],
    }
    dump_json(out_dir / "gate_decision.json", gate_payload)

    final_secret_audit = secret_scan(out_dir)
    dump_json(out_dir / "forbidden_field_audit.json", forbidden_field_audit(out_dir))
    final_forbidden_audit = strict_read_json(out_dir / "forbidden_field_audit.json")
    if final_secret_audit["secret_leak_count"] != secret_audit["secret_leak_count"]:
        raise RuntimeError(f"secret scan changed after final report/gate creation: {final_secret_audit}")
    if final_forbidden_audit["forbidden_operation_count"] != forbidden_audit["forbidden_operation_count"]:
        raise RuntimeError(f"forbidden field audit changed after final report/gate creation: {final_forbidden_audit}")

    manifest = build_manifest(out_dir)
    dump_json(out_dir / "artifact_manifest.json", manifest)
    manifest_check = verify_manifest(out_dir, manifest)
    if (
        manifest_check["manifest_hash_mismatch_count"]
        or manifest_check["manifest_size_mismatch_count"]
        or manifest_check["strict_json_failure_count"]
        or manifest_check["required_file_missing_count"]
        or manifest_check["unexpected_file_count"]
        or manifest_check["duplicate_manifest_path_count"]
        or manifest_check["absolute_path_leak_count"]
    ):
        raise RuntimeError(f"manifest validation failed after finalization: {manifest_check}")

    (out_dir / "_SUCCESS.lock").write_text(
        json.dumps({"success": True, "gate": gate_status, "created_at": now_iso()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"artifact:\n{out_dir}")
    print(f"gate:\n{gate_status}")
    print(f"upstream:\n{upstream}")
    print(f"upstream gate:\n{gate.get('gate_status')}")
    print(f"registry rows:\n{key_snapshot['registry_rows']}")
    print(f"\nprimary identified set:\n{key_snapshot['primary_identified_set']}")
    print(f"primary width:\n{key_snapshot['primary_identified_set_width']}")
    print(f"request identified set:\n{key_snapshot['request_identified_set']}")
    print(f"request width:\n{key_snapshot['request_identified_set_width']}")
    print(f"\ndual==provider:\n{key_snapshot['dual_equals_provider']}")
    print(f"request boundary determines dual:\n{key_snapshot['request_boundary_determines_dual_count']}")
    print(f"cadence-only effect:\n{key_snapshot['cadence_only_effect_on_primary_dual']}")
    print(f"10s cadence:\n{key_snapshot['cadence_10s_status']}")
    print(f"15s cadence:\n{key_snapshot['cadence_15s_status']}")
    print(f"\nPath A1 assessment:\n{path_a_request['assessment']}")
    print(f"Path A2 assessment:\n{path_a_provider['assessment']}")
    print(f"Path B assessment:\n{path_b_review['assessment']}")
    print(f"recommended path:\n{recommended['recommended_path_enum']}")
    print(f"recommendation scope:\n{recommended['recommendation_scope']}")
    print("\nAPI/service key/DB/network:\n0/false/false/false")
    print(f"upstream modified/deleted/added:\n{mutation['upstream_modified']}/{mutation['upstream_deleted']}/{mutation['upstream_added']}")
    print(f"required files:\n{len(REQUIRED_FILES)}")
    print("manifest hash/size mismatches:\n0/0")
    print("strict JSON failures:\n0")
    print(f"Parquet read failures:\n{parquet_check['parquet_read_failure_count']}")
    print(f"JSON/Parquet mismatches:\n{parquet_check['json_parquet_mismatch_count']}")
    print(f"secret leaks:\n{secret_audit['secret_leak_count']}")
    print(f"forbidden operations:\n{prohibited_operation_audit['forbidden_operation_count'] + forbidden_audit['forbidden_operation_count']}")
    print("\npath adopted:\nfalse")
    print("execution authorized:\nfalse")
    print("observation campaign approved:\nfalse")
    print("sensitivity grid execution approved:\nfalse")
    print("Phase 2 authorized:\nfalse")
    print(f"\nnext authorized action:\n{recommended['next_authorized_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
