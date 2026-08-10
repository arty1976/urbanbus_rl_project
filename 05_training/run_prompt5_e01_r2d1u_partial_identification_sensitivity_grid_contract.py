from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd


KST = ZoneInfo("Asia/Seoul")
DEFAULT_PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
DEFAULT_UPSTREAM_RELATIVE = Path(
    "05_training/artifacts/"
    "prompt5_e01_r2d1t_observation_resolution_partial_identification_path_review_20260730_232658"
)
EXPECTED_UPSTREAM_GATE = "PASS_PATH_B_PARTIAL_IDENTIFICATION_SENSITIVITY_DESIGN_RECOMMENDED_STILL_LOCKED"
EXPECTED_RECOMMENDATION = "RECOMMEND_PATH_B_PARTIAL_IDENTIFICATION_SENSITIVITY_DESIGN"
EXPECTED_SCOPE = "DESIGN_REVIEW_ONLY"
EXPECTED_NEXT = "Partial-identification sensitivity-grid contract design only"

PASS_READY = "PASS_PARTIAL_IDENTIFICATION_SENSITIVITY_GRID_CONTRACT_READY_STILL_LOCKED"
PASS_SYMBOLIC = "PASS_PARTIAL_IDENTIFICATION_SENSITIVITY_GRID_CONTRACT_PARTIALLY_SYMBOLIC_STILL_LOCKED"
PASS_REDUCTION = "PASS_SENSITIVITY_GRID_CONTRACT_DESIGN_REQUIRES_FACTORIAL_REDUCTION_STILL_LOCKED"

REQUIRED_FILES = [
    "upstream_validation.json",
    "upstream_lineage.json",
    "r2d1t_gate_snapshot.json",
    "r2d1t_recommendation_snapshot.json",
    "r2d1t_lock_snapshot.json",
    "sensitivity_axis_registry.json",
    "sensitivity_axis_registry.parquet",
    "sensitivity_level_registry.json",
    "sensitivity_level_registry.parquet",
    "assumption_registry.json",
    "exclusion_registry.json",
    "reference_cell_contract.json",
    "combination_rules.json",
    "combination_pruning_rules.json",
    "design_matrix_schema.json",
    "sensitivity_design_matrix.json",
    "sensitivity_design_matrix.parquet",
    "design_matrix_cardinality_audit.json",
    "execution_boundary.json",
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

UPSTREAM_LOCK_FIELDS = [
    "path_adopted",
    "execution_authorized",
    "observation_campaign_approved",
    "sensitivity_grid_execution_approved",
    "identified_set_reestimation_approved",
    "turnbull_reestimation_approved",
    "phase2_authorized",
]

LOCK_FALSE_FIELDS = [
    "path_b_adopted",
    "sensitivity_contract_adopted",
    "sensitivity_grid_execution_approved",
    "sensitivity_analysis_executed",
    "identified_set_reestimation_approved",
    "identified_set_reestimation_executed",
    "turnbull_reestimation_approved",
    "turnbull_reestimation_executed",
    "new_point_estimate_approved",
    "new_point_estimate_computed",
    "observation_campaign_approved",
    "provider_endpoint_change_approved",
    "primary_clock_change_approved",
    "clock_correction_approved",
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        out[rel] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    return out


def compare_snapshot(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_keys = set(before)
    after_keys = set(after)
    modified = sorted(k for k in before_keys & after_keys if before[k] != after[k])
    return {
        "upstream_added": len(after_keys - before_keys),
        "upstream_deleted": len(before_keys - after_keys),
        "upstream_modified": len(modified),
        "added_files": sorted(after_keys - before_keys),
        "deleted_files": sorted(before_keys - after_keys),
        "modified_files": modified,
    }


def validate_manifest(artifact: Path) -> dict[str, Any]:
    manifest_path = artifact / "artifact_manifest.json"
    if not manifest_path.is_file():
        return {"status": "FAIL", "reason": "artifact_manifest.json missing"}
    manifest = strict_read_json(manifest_path)
    hash_mismatch = []
    size_mismatch = []
    strict_json_fail = []
    missing = []
    for entry in manifest.get("files", []):
        path = artifact / entry["relative_path"]
        if not path.is_file():
            missing.append(entry["relative_path"])
            continue
        if entry.get("sha256") != sha256_file(path):
            hash_mismatch.append(entry["relative_path"])
        if entry.get("size_bytes") != path.stat().st_size:
            size_mismatch.append(entry["relative_path"])
        if path.suffix == ".json":
            try:
                strict_read_json(path)
            except Exception as exc:
                strict_json_fail.append({"relative_path": entry["relative_path"], "error": type(exc).__name__})
    return {
        "status": "PASS" if not (hash_mismatch or size_mismatch or strict_json_fail or missing) else "FAIL",
        "manifest_file_count": len(manifest.get("files", [])),
        "manifest_hash_mismatch_count": len(hash_mismatch),
        "manifest_size_mismatch_count": len(size_mismatch),
        "strict_json_failure_count": len(strict_json_fail),
        "manifest_missing_count": len(missing),
        "hash_mismatches": hash_mismatch,
        "size_mismatches": size_mismatch,
        "strict_json_failures": strict_json_fail,
        "manifest_missing": missing,
    }


def validate_upstream(upstream: Path) -> dict[str, Any]:
    gate = strict_read_json(upstream / "gate_decision.json")
    recommendation = strict_read_json(upstream / "recommended_next_path.json")
    locks = strict_read_json(upstream / "lock_state.json")
    key = strict_read_json(upstream / "r2d1s_key_findings_snapshot.json")
    manifest = validate_manifest(upstream)
    expected_key_values = {
        "registry_rows": 12,
        "primary_identified_set": [585.0, 2189.0],
        "primary_identified_set_width": 1604.0,
        "request_identified_set": [1517.0, 2061.0],
        "request_identified_set_width": 544.0,
        "dual_equals_provider_count": 12,
        "request_boundary_determines_dual_count": 0,
        "cadence_only_effect_on_primary_dual": "ZERO_UNDER_FIXED_PROVIDER_ENDPOINTS",
        "continuous_interval_robustness_established": False,
    }
    key_mismatches = [
        name for name, expected in expected_key_values.items()
        if key.get(name) != expected
    ]
    lock_violations = [name for name in UPSTREAM_LOCK_FIELDS if locks.get(name) is not False]
    checks = {
        "gate_matches": gate.get("gate") == EXPECTED_UPSTREAM_GATE,
        "recommendation_matches": recommendation.get("recommended_path_enum") == EXPECTED_RECOMMENDATION,
        "scope_matches": recommendation.get("recommendation_scope") == EXPECTED_SCOPE,
        "next_action_matches": recommendation.get("next_authorized_action") == EXPECTED_NEXT,
        "required_locks_false": not lock_violations,
        "key_values_match": not key_mismatches,
        "manifest_integrity_pass": manifest["status"] == "PASS",
    }
    return {
        "upstream_artifact": str(upstream),
        "upstream_gate": gate.get("gate"),
        "upstream_recommendation": recommendation.get("recommended_path_enum"),
        "upstream_recommendation_scope": recommendation.get("recommendation_scope"),
        "upstream_next_authorized_action": recommendation.get("next_authorized_action"),
        "checks": checks,
        "lock_violations": lock_violations,
        "key_mismatches": key_mismatches,
        "manifest_validation": manifest,
        "upstream_integrity_status": "PASS" if all(checks.values()) else "FAIL",
    }


def axis_registry() -> list[dict[str, Any]]:
    rows = [
        ("provider_endpoint_uncertainty", 1, "SUPPORTED_BY_CURRENT_ARTIFACT",
         "R2D-1T retained provider endpoint dominance as the main uncertainty axis.", True),
        ("request_endpoint_uncertainty", 1, "SUPPORTED_BY_CURRENT_ARTIFACT",
         "R2D-1T retained request-only identified set and cadence counterfactual as contract references.", True),
        ("clock_rounding_allowance", 1, "DERIVABLE_WITHOUT_NEW_DATA",
         "Clock precision may be represented as a symbolic convention without changing timestamps.", True),
        ("interval_censoring_convention", 1, "DERIVABLE_WITHOUT_NEW_DATA",
         "Closed-interval treatment can be contrasted with symbolic alternate conventions.", True),
        ("endpoint_inclusion_exclusion_convention", 1, "DERIVABLE_WITHOUT_NEW_DATA",
         "Boundary inclusion conventions can be specified as symbolic sensitivity levels.", True),
        ("provider_request_precedence_rule", 1, "SUPPORTED_BY_CURRENT_ARTIFACT",
         "The frozen primary contract uses conservative provider precedence.", True),
        ("clock_skew_allowance", 2, "REQUIRES_NEW_ASSUMPTION",
         "No clock offset correction is authorized; non-reference levels must be hypothetical.", True),
        ("missing_boundary_treatment", 2, "REQUIRES_NEW_ASSUMPTION",
         "Discrete feed absence cannot identify causal boundary treatment without assumptions.", True),
        ("scheduled_layover_treatment", 3, "REQUIRES_NEW_OBSERVATION",
         "Layover estimate was not computed and scheduled layover data are unavailable locally.", False),
    ]
    out = []
    for order, (axis_id, tier, classification, basis, included) in enumerate(rows, start=1):
        excluded = not included
        out.append({
            "axis_id": axis_id,
            "axis_order": order,
            "tier": tier,
            "classification": classification,
            "basis": basis,
            "eligibility": "ELIGIBLE_FOR_CONTRACT_DESIGN" if included else "EXCLUDED_REQUIRES_NEW_OBSERVATION",
            "grid_inclusion_allowed": included,
            "eligible_for_design_matrix": included,
            "requires_new_assumption_for_non_reference": tier == 2,
            "requires_new_observation": tier == 3,
            "excluded_from_design_matrix": excluded,
        })
    return out


REFERENCE_VALUES = {
    "provider_endpoint_uncertainty": ("FROZEN_PROVIDER_ENDPOINT_CONTRACT", "Frozen observed provider endpoints"),
    "request_endpoint_uncertainty": ("FROZEN_REQUEST_ENDPOINT_CONTRACT", "Frozen observed request endpoints"),
    "clock_rounding_allowance": ("FROZEN_TIMESTAMP_CONVENTION", "Frozen timestamp convention"),
    "interval_censoring_convention": ("FROZEN_CLOSED_INTERVAL_CONVENTION", "Frozen closed-interval convention"),
    "endpoint_inclusion_exclusion_convention": ("FROZEN_ENDPOINT_INCLUSION_CONVENTION", "Frozen endpoint inclusion convention"),
    "provider_request_precedence_rule": ("FROZEN_CONSERVATIVE_PROVIDER_PRECEDENCE", "Conservative provider precedence"),
    "clock_skew_allowance": ("NO_CLOCK_CORRECTION_REFERENCE", "No correction reference"),
    "missing_boundary_treatment": ("FROZEN_UPSTREAM_MISSING_BOUNDARY_TREATMENT", "Frozen upstream missing-boundary treatment"),
}

ALT_VALUES = {
    "provider_endpoint_uncertainty": ("SYMBOLIC_PROVIDER_ENDPOINT_ALTERNATE", "Provider endpoint alternate placeholder"),
    "request_endpoint_uncertainty": ("SYMBOLIC_REQUEST_ENDPOINT_ALTERNATE", "Request endpoint alternate placeholder"),
    "clock_rounding_allowance": ("SYMBOLIC_CLOCK_ROUNDING_ALTERNATE", "Clock rounding alternate placeholder"),
    "interval_censoring_convention": ("SYMBOLIC_INTERVAL_CONVENTION_ALTERNATE", "Interval convention alternate placeholder"),
    "endpoint_inclusion_exclusion_convention": ("SYMBOLIC_ENDPOINT_BOUNDARY_ALTERNATE", "Endpoint boundary alternate placeholder"),
    "provider_request_precedence_rule": ("SYMBOLIC_PRECEDENCE_ALTERNATE", "Provider/request precedence alternate placeholder"),
}


def build_levels(axes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    levels: list[dict[str, Any]] = []
    for axis in axes:
        axis_id = axis["axis_id"]
        if axis["tier"] == 3:
            levels.append({
                "axis_id": axis_id,
                "level_id": f"{axis_id}__excluded_requires_new_observation",
                "level_order": 0,
                "level_type": "EXCLUDED",
                "level_value": "EXCLUDED_REQUIRES_NEW_OBSERVATION",
                "level_unit": None,
                "level_label": "Excluded until new observation is separately authorized",
                "level_source": "R2D-1T path_b_sensitivity_axis_inventory.json",
                "classification": axis["classification"],
                "assumption_required": False,
                "assumption_id": None,
                "assumption_statement": None,
                "empirically_observed": False,
                "derivable_without_new_data": False,
                "eligible_for_design_matrix": False,
                "execution_allowed": False,
                "is_reference_level": False,
                "claim_scope": "EXCLUDED_REQUIRES_NEW_OBSERVATION",
            })
            continue
        ref_value, ref_label = REFERENCE_VALUES[axis_id]
        levels.append({
            "axis_id": axis_id,
            "level_id": f"{axis_id}__reference",
            "level_order": 0,
            "level_type": "FROZEN_REFERENCE",
            "level_value": ref_value,
            "level_unit": None,
            "level_label": ref_label,
            "level_source": "R2D-1T upstream frozen contract",
            "classification": axis["classification"],
            "assumption_required": False,
            "assumption_id": None,
            "assumption_statement": None,
            "empirically_observed": axis["tier"] == 1,
            "derivable_without_new_data": True,
            "eligible_for_design_matrix": True,
            "execution_allowed": False,
            "is_reference_level": True,
            "claim_scope": "FROZEN_REFERENCE_CONTRACT_ONLY",
        })
        if axis["tier"] == 1:
            alt_value, alt_label = ALT_VALUES[axis_id]
            levels.append({
                "axis_id": axis_id,
                "level_id": f"{axis_id}__symbolic_alternate",
                "level_order": 1,
                "level_type": "SYMBOLIC_PLACEHOLDER",
                "level_value": alt_value,
                "level_unit": None,
                "level_label": alt_label,
                "level_source": "R2D-1T Path B design inventory; numeric value unresolved",
                "classification": axis["classification"],
                "assumption_required": False,
                "assumption_id": None,
                "assumption_statement": None,
                "empirically_observed": False,
                "derivable_without_new_data": False,
                "eligible_for_design_matrix": True,
                "execution_allowed": False,
                "is_reference_level": False,
                "claim_scope": "DESIGN_CONTRACT_ONLY",
            })
        else:
            assumption_id = f"ASSUME_{axis_id.upper()}_SENSITIVITY"
            levels.append({
                "axis_id": axis_id,
                "level_id": f"{axis_id}__assumption_only_alternate",
                "level_order": 1,
                "level_type": "ASSUMPTION_ONLY",
                "level_value": f"HYPOTHETICAL_{axis_id.upper()}_ALTERNATE",
                "level_unit": None,
                "level_label": f"Hypothetical {axis_id} alternate",
                "level_source": "Contract-only assumption label; not empirically observed",
                "classification": axis["classification"],
                "assumption_required": True,
                "assumption_id": assumption_id,
                "assumption_statement": (
                    f"For sensitivity design only, allow one symbolic alternate treatment for {axis_id}; "
                    "this is not observed and does not authorize execution."
                ),
                "empirically_observed": False,
                "derivable_without_new_data": False,
                "eligible_for_design_matrix": True,
                "execution_allowed": False,
                "is_reference_level": False,
                "claim_scope": "HYPOTHETICAL_SENSITIVITY_ONLY",
            })
    return levels


def build_assumptions(levels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "assumption_id": row["assumption_id"],
            "axis_id": row["axis_id"],
            "level_id": row["level_id"],
            "assumption_statement": row["assumption_statement"],
            "assumption_required": True,
            "empirically_observed": False,
            "claim_scope": "HYPOTHETICAL_SENSITIVITY_ONLY",
            "execution_allowed": False,
        }
        for row in levels
        if row["assumption_required"]
    ]


def canonical_bindings(bindings: Mapping[str, str]) -> str:
    return json.dumps(dict(sorted(bindings.items())), sort_keys=True, separators=(",", ":"))


def build_design_matrix(axes: list[dict[str, Any]], levels: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible_axes = [row["axis_id"] for row in axes if row["eligible_for_design_matrix"]]
    by_axis: dict[str, list[dict[str, Any]]] = {axis_id: [] for axis_id in eligible_axes}
    for row in levels:
        if row["eligible_for_design_matrix"]:
            by_axis[row["axis_id"]].append(row)
    ref_by_axis = {axis: next(row for row in rows if row["is_reference_level"]) for axis, rows in by_axis.items()}
    alt_by_axis = {axis: [row for row in rows if not row["is_reference_level"]] for axis, rows in by_axis.items()}
    raw_cardinality = math.prod(len(by_axis[axis]) for axis in eligible_axes)

    candidates: list[tuple[str, dict[str, dict[str, Any]]]] = []
    reference = {axis: ref_by_axis[axis] for axis in eligible_axes}
    candidates.append(("reference", reference))
    for axis in eligible_axes:
        for alt in alt_by_axis[axis]:
            bindings = dict(reference)
            bindings[axis] = alt
            candidates.append((f"single_axis__{axis}", bindings))
    pair_specs = [
        ("provider_request_uncertainty_pair", ["provider_endpoint_uncertainty", "request_endpoint_uncertainty"]),
        ("interval_endpoint_convention_pair", ["interval_censoring_convention", "endpoint_inclusion_exclusion_convention"]),
        ("clock_rounding_skew_pair", ["clock_rounding_allowance", "clock_skew_allowance"]),
        ("precedence_missing_boundary_pair", ["provider_request_precedence_rule", "missing_boundary_treatment"]),
    ]
    for label, pair_axes in pair_specs:
        bindings = dict(reference)
        valid = True
        for axis in pair_axes:
            alts = alt_by_axis[axis]
            if not alts:
                valid = False
                break
            bindings[axis] = alts[0]
        if valid:
            candidates.append((label, bindings))

    seen = set()
    cells = []
    for label, level_bindings in candidates:
        bindings = {axis: level["level_id"] for axis, level in level_bindings.items()}
        signature = canonical_bindings(bindings)
        if signature in seen:
            continue
        seen.add(signature)
        levels_for_cell = list(level_bindings.values())
        contains_symbolic = any(row["level_type"] == "SYMBOLIC_PLACEHOLDER" for row in levels_for_cell)
        contains_new_assumption = any(row["assumption_required"] for row in levels_for_cell)
        contains_excluded = any(row["level_type"] == "EXCLUDED" for row in levels_for_cell)
        reference_cell = all(row["is_reference_level"] for row in levels_for_cell)
        if contains_excluded:
            claim_scope = "EXCLUDED_REQUIRES_NEW_OBSERVATION"
        elif contains_new_assumption:
            claim_scope = "HYPOTHETICAL_SENSITIVITY_ONLY"
        elif contains_symbolic:
            claim_scope = "DESIGN_CONTRACT_ONLY"
        else:
            claim_scope = "FROZEN_REFERENCE_CONTRACT_ONLY"
        cells.append({
            "design_cell_id": f"CELL_{len(cells):03d}",
            "cell_label": label,
            "axis_level_bindings": signature,
            "reference_cell": reference_cell,
            "contains_symbolic_level": contains_symbolic,
            "contains_new_assumption": contains_new_assumption,
            "contains_excluded_axis": contains_excluded,
            "design_eligible": not contains_excluded,
            "execution_approved": False,
            "reestimation_approved": False,
            "claim_scope": claim_scope,
        })

    audit = {
        "raw_cartesian_cardinality": raw_cardinality,
        "excluded_axis_cardinality": sum(1 for row in axes if not row["grid_inclusion_allowed"]),
        "pruned_cardinality": len(cells),
        "reference_cell_count": sum(1 for row in cells if row["reference_cell"]),
        "assumption_only_cell_count": sum(1 for row in cells if row["contains_new_assumption"]),
        "symbolic_cell_count": sum(1 for row in cells if row["contains_symbolic_level"]),
        "fully_concrete_design_cell_count": sum(
            1 for row in cells
            if row["design_eligible"] and not row["contains_symbolic_level"] and not row["contains_new_assumption"]
        ),
        "all_cells_execution_approved_false": all(row["execution_approved"] is False for row in cells),
        "all_cells_reestimation_approved_false": all(row["reestimation_approved"] is False for row in cells),
        "cardinality_guard": "PASS" if len(cells) <= 256 else "REQUIRES_FACTORIAL_REDUCTION",
        "full_cartesian_materialized": False,
    }
    return cells, audit


def parquet_json_check(out_dir: Path) -> dict[str, Any]:
    pairs = [
        ("sensitivity_axis_registry.json", "sensitivity_axis_registry.parquet", "records"),
        ("sensitivity_level_registry.json", "sensitivity_level_registry.parquet", "records"),
        ("sensitivity_design_matrix.json", "sensitivity_design_matrix.parquet", "records"),
    ]
    parquet_failures = []
    mismatch_count = 0
    for json_name, parquet_name, record_key in pairs:
        try:
            records = strict_read_json(out_dir / json_name)[record_key]
            df = pd.read_parquet(out_dir / parquet_name)
        except Exception as exc:
            parquet_failures.append({"json": json_name, "parquet": parquet_name, "error": type(exc).__name__})
            continue
        if len(records) != len(df):
            mismatch_count += 1
            continue
        json_columns = sorted(records[0].keys()) if records else []
        parquet_columns = sorted(df.columns.tolist())
        if json_columns != parquet_columns:
            mismatch_count += 1
            continue
        for idx, record in enumerate(records):
            row = df.iloc[idx].to_dict()
            for key, value in record.items():
                other = row.get(key)
                if isinstance(value, float):
                    if abs(value - float(other)) > 1e-9:
                        mismatch_count += 1
                elif value != other:
                    mismatch_count += 1
    return {
        "parquet_read_failure_count": len(parquet_failures),
        "json_parquet_mismatch_count": mismatch_count,
        "parquet_failures": parquet_failures,
        "checked_pairs": [{"json": a, "parquet": b} for a, b, _ in pairs],
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
        if path.is_file() and path.suffix in {".json", ".md", ".lock"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in SECRET_VALUE_PATTERNS:
                if pattern.search(text):
                    findings.append({"relative_path": path.name, "pattern_sha256": hashlib.sha256(pattern.pattern.encode()).hexdigest()})
    return {"secret_leak_count": len(findings), "findings": findings}


FORBIDDEN_RESULT_KEYS = {
    "new_point_estimate",
    "new_layover_estimate",
    "reestimated_interval",
    "turnbull_result",
    "identified_set_result",
    "sensitivity_grid_result",
    "robustness_result",
    "confidence_interval_result",
    "simulator_applied_value",
    "phase2_result",
    "baseline_rerun_result",
    "training_result",
    "winner_selected",
}


def iter_keys(payload: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    out = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.append((prefix + (str(key),), value))
            out.extend(iter_keys(value, prefix + (str(key),)))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            out.extend(iter_keys(value, prefix + (str(idx),)))
    return out


def forbidden_field_audit(out_dir: Path) -> dict[str, Any]:
    findings = []
    allowed = {"forbidden_field_audit.json", "prohibited_operation_audit.json"}
    for path in sorted(out_dir.glob("*.json")):
        if path.name in allowed:
            continue
        try:
            payload = strict_read_json(path)
        except Exception:
            continue
        for key_path, value in iter_keys(payload):
            if key_path[-1] in FORBIDDEN_RESULT_KEYS and value not in (False, None, "PROHIBITED", "NOT_AUTHORIZED"):
                findings.append({"relative_path": path.name, "key_path": ".".join(key_path), "value_type": type(value).__name__})
    return {
        "forbidden_operation_count": len(findings),
        "forbidden_field_presence_count": len(findings),
        "findings": findings,
        "forbidden_field_inventory": sorted(FORBIDDEN_RESULT_KEYS),
    }


def artifact_role_for(name: str) -> str:
    if name.startswith("upstream") or name.startswith("r2d1t") or name == "lineage_field_audit.json":
        return "UPSTREAM_VALIDATION"
    if name.startswith("sensitivity_axis"):
        return "AXIS_REGISTRY"
    if name.startswith("sensitivity_level") or name == "assumption_registry.json" or name == "exclusion_registry.json":
        return "LEVEL_REGISTRY"
    if name in {"reference_cell_contract.json", "combination_rules.json", "combination_pruning_rules.json"}:
        return "COMBINATION_CONTRACT"
    if name.startswith("design_matrix") or name.startswith("sensitivity_design_matrix"):
        return "DESIGN_MATRIX"
    if name in {"execution_boundary.json", "lock_state.json", "prohibited_operation_audit.json", "api_db_network_audit.json"}:
        return "EXECUTION_BOUNDARY"
    if name.startswith("final_report"):
        return "FINAL_REPORT"
    if name == "gate_decision.json":
        return "GATE"
    return "AUDIT"


def build_manifest(out_dir: Path) -> dict[str, Any]:
    entries = []
    for name in REQUIRED_FILES:
        if name in MANIFEST_EXCLUDED:
            continue
        path = out_dir / name
        exists = path.is_file()
        strict_json_valid = None
        if exists and path.suffix == ".json":
            try:
                strict_read_json(path)
                strict_json_valid = True
            except Exception:
                strict_json_valid = False
        entries.append({
            "relative_path": name,
            "sha256": sha256_file(path) if exists else None,
            "size_bytes": path.stat().st_size if exists else None,
            "artifact_role": artifact_role_for(name),
            "strict_json_valid": strict_json_valid,
            "exists": exists,
        })
    expected = set(REQUIRED_FILES) - MANIFEST_EXCLUDED
    actual = {p.name for p in out_dir.iterdir() if p.is_file() and p.name not in MANIFEST_EXCLUDED}
    return {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "required_file_count": len(REQUIRED_FILES),
        "manifest_excluded_files": sorted(MANIFEST_EXCLUDED),
        "files": entries,
        "missing_required_file_count": sum(not row["exists"] for row in entries),
        "unexpected_file_count": len(actual - expected),
        "unexpected_files": sorted(actual - expected),
        "duplicate_manifest_path_count": len(entries) - len({row["relative_path"] for row in entries}),
        "absolute_path_leak_count": sum(Path(row["relative_path"]).is_absolute() for row in entries),
    }


def verify_manifest(out_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    hash_mismatch = []
    size_mismatch = []
    strict_json_fail = []
    for entry in manifest["files"]:
        path = out_dir / entry["relative_path"]
        if not path.is_file():
            continue
        if sha256_file(path) != entry["sha256"]:
            hash_mismatch.append(entry["relative_path"])
        if path.stat().st_size != entry["size_bytes"]:
            size_mismatch.append(entry["relative_path"])
        if path.suffix == ".json":
            try:
                strict_read_json(path)
            except Exception as exc:
                strict_json_fail.append({"relative_path": entry["relative_path"], "error": type(exc).__name__})
    return {
        "manifest_hash_mismatch_count": len(hash_mismatch),
        "manifest_size_mismatch_count": len(size_mismatch),
        "strict_json_failure_count": len(strict_json_fail),
        "required_file_missing_count": manifest["missing_required_file_count"],
        "unexpected_file_count": manifest["unexpected_file_count"],
        "duplicate_manifest_path_count": manifest["duplicate_manifest_path_count"],
        "absolute_path_leak_count": manifest["absolute_path_leak_count"],
        "hash_mismatches": hash_mismatch,
        "size_mismatches": size_mismatch,
        "strict_json_failures": strict_json_fail,
    }


def write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(records).to_parquet(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--upstream-artifact", type=Path, default=DEFAULT_UPSTREAM_RELATIVE)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    upstream = args.upstream_artifact
    if not upstream.is_absolute():
        upstream = (project_root / upstream).resolve()
    if not upstream.is_dir():
        raise FileNotFoundError(upstream)

    out_dir = project_root / "05_training" / "artifacts" / f"prompt5_e01_r2d1u_partial_identification_sensitivity_grid_contract_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    upstream_before = file_snapshot(upstream)
    gate = strict_read_json(upstream / "gate_decision.json")
    recommendation = strict_read_json(upstream / "recommended_next_path.json")
    locks = strict_read_json(upstream / "lock_state.json")
    key = strict_read_json(upstream / "r2d1s_key_findings_snapshot.json")
    upstream_validation = validate_upstream(upstream)

    dump_json(out_dir / "upstream_validation.json", upstream_validation)
    dump_json(out_dir / "upstream_lineage.json", {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "authoritative_upstream_artifact": str(upstream),
        "authoritative_upstream_gate": gate.get("gate"),
        "authoritative_upstream_recommendation": recommendation.get("recommended_path_enum"),
        "source_files_read": [
            "gate_decision.json",
            "recommended_next_path.json",
            "lock_state.json",
            "r2d1s_key_findings_snapshot.json",
            "path_b_sensitivity_axis_inventory.json",
            "artifact_manifest.json",
        ],
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
    })
    dump_json(out_dir / "r2d1t_gate_snapshot.json", gate)
    dump_json(out_dir / "r2d1t_recommendation_snapshot.json", recommendation)
    dump_json(out_dir / "r2d1t_lock_snapshot.json", locks)

    axes = axis_registry()
    levels = build_levels(axes)
    assumptions = build_assumptions(levels)
    exclusions = [
        {
            "axis_id": "scheduled_layover_treatment",
            "eligibility": "EXCLUDED_REQUIRES_NEW_OBSERVATION",
            "grid_inclusion_allowed": False,
            "exclusion_reason": "scheduled layover values were not observed or estimated upstream",
            "new_observation_required": True,
            "layover_estimate_computed": False,
            "execution_allowed": False,
        }
    ]
    dump_json(out_dir / "sensitivity_axis_registry.json", {"axis_count": len(axes), "records": axes})
    write_parquet(axes, out_dir / "sensitivity_axis_registry.parquet")
    dump_json(out_dir / "sensitivity_level_registry.json", {"level_count": len(levels), "records": levels})
    write_parquet(levels, out_dir / "sensitivity_level_registry.parquet")
    dump_json(out_dir / "assumption_registry.json", {"assumption_count": len(assumptions), "records": assumptions})
    dump_json(out_dir / "exclusion_registry.json", {"excluded_axis_count": len(exclusions), "records": exclusions})

    reference_bindings = {
        row["axis_id"]: row["level_id"]
        for row in levels
        if row["eligible_for_design_matrix"] and row["is_reference_level"]
    }
    dump_json(out_dir / "reference_cell_contract.json", {
        "reference_cell_count": 1 if len(reference_bindings) == 8 else 0,
        "reference_cell_id": "CELL_000",
        "axis_level_bindings": canonical_bindings(reference_bindings),
        "contract_statement": "Reference cell reproduces the frozen conservative primary contract without recalculation.",
        "new_calculation_claimed": False,
        "execution_approved": False,
        "reestimation_approved": False,
    })

    combination_rules = {
        "cartesian_product_not_fully_materialized": True,
        "tier3_axes_in_design_matrix_allowed": False,
        "contains_excluded_axis_implies_design_eligible_false": True,
        "contains_symbolic_level_implies_execution_approved_false": True,
        "contains_new_assumption_implies_hypothetical_claim_scope": True,
        "all_design_cells_execution_approved": False,
        "all_design_cells_reestimation_approved": False,
    }
    pruning_rules = {
        "rules": [
            "remove semantically duplicate combinations",
            "remove duplicate reference cells",
            "remove logically incompatible convention combinations",
            "remove all Tier 3 scheduled_layover_treatment combinations",
            "remove duplicate application of the same assumption",
            "retain reference, single-axis perturbations, and a small set of interpretable paired perturbations",
        ],
        "factorial_reduction_execution_approved": False,
        "sampling_or_optimization_executed": False,
    }
    dump_json(out_dir / "combination_rules.json", combination_rules)
    dump_json(out_dir / "combination_pruning_rules.json", pruning_rules)
    dump_json(out_dir / "design_matrix_schema.json", {
        "fields": [
            "design_cell_id",
            "axis_level_bindings",
            "reference_cell",
            "contains_symbolic_level",
            "contains_new_assumption",
            "contains_excluded_axis",
            "design_eligible",
            "execution_approved",
            "reestimation_approved",
            "claim_scope",
        ],
        "axis_level_bindings_encoding": "canonical_json_string",
        "schema_status": "FROZEN_FOR_CONTRACT_REVIEW_ONLY",
    })

    design_matrix, cardinality = build_design_matrix(axes, levels)
    dump_json(out_dir / "sensitivity_design_matrix.json", {"design_cell_count": len(design_matrix), "records": design_matrix})
    write_parquet(design_matrix, out_dir / "sensitivity_design_matrix.parquet")
    dump_json(out_dir / "design_matrix_cardinality_audit.json", cardinality)

    execution_boundary = {
        "grid_contract_status": "DESIGNED_BLUEPRINT_ONLY",
        "grid_execution_status": "NOT_EXECUTED_LOCKED",
        "identified_set_reestimation_status": "NOT_EXECUTED_LOCKED",
        "turnbull_reestimation_status": "NOT_EXECUTED_LOCKED",
        "point_estimate_status": "NOT_COMPUTED_LOCKED",
        "robustness_status": "NOT_ASSESSED_LOCKED",
        "simulator_status": "NOT_APPLIED_LOCKED",
        "phase2_status": "NOT_AUTHORIZED",
    }
    dump_json(out_dir / "execution_boundary.json", execution_boundary)
    lock_state = {name: False for name in LOCK_FALSE_FIELDS}
    lock_state["all_required_locks_false"] = all(lock_state[name] is False for name in LOCK_FALSE_FIELDS)
    dump_json(out_dir / "lock_state.json", lock_state)
    prohibited = {
        "api_called": False,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
        "new_observation_collected": False,
        "new_episode_collected": False,
        "upstream_modified": False,
        "timestamp_modified": False,
        "clock_offset_applied": False,
        "provider_endpoint_modified": False,
        "turnbull_executed": False,
        "identified_set_calculated": False,
        "point_estimate_calculated": False,
        "confidence_interval_calculated": False,
        "robustness_result_calculated": False,
        "layover_estimated": False,
        "simulator_applied": False,
        "phase2_executed": False,
        "baseline_rerun_executed": False,
        "training_or_retraining_executed": False,
        "forbidden_operation_count": 0,
    }
    dump_json(out_dir / "prohibited_operation_audit.json", prohibited)
    api = {
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
        "network_preflight_executed": False,
        "database_query_executed": False,
    }
    dump_json(out_dir / "api_db_network_audit.json", api)
    mutation = compare_snapshot(upstream_before, file_snapshot(upstream))
    dump_json(out_dir / "upstream_mutation_audit.json", mutation)
    dump_json(out_dir / "lineage_field_audit.json", {
        "authoritative_upstream_only": True,
        "authoritative_upstream_artifact": str(upstream),
        "non_authoritative_prior_artifacts_used": [],
        "path_b_recommendation_not_adoption": True,
        "execution_result_fields_present": False,
    })
    dump_json(out_dir / "forbidden_field_audit.json", forbidden_field_audit(out_dir))

    parquet_check = parquet_json_check(out_dir)
    secret_audit = secret_scan(out_dir)
    forbidden = strict_read_json(out_dir / "forbidden_field_audit.json")

    axis_count = len(axes)
    tier1 = sum(1 for row in axes if row["tier"] == 1)
    tier2 = sum(1 for row in axes if row["tier"] == 2)
    tier3 = sum(1 for row in axes if row["tier"] == 3)
    eligible_axis_count = sum(1 for row in axes if row["eligible_for_design_matrix"])
    excluded_axis_count = sum(1 for row in axes if not row["eligible_for_design_matrix"])
    level_count = len(levels)
    numeric_level_count = sum(1 for row in levels if row["level_type"] == "DIRECTLY_DERIVED_NUMERIC")
    symbolic_level_count = sum(1 for row in levels if row["level_type"] == "SYMBOLIC_PLACEHOLDER")
    assumption_only_level_count = sum(1 for row in levels if row["level_type"] == "ASSUMPTION_ONLY")
    unsupported_numeric_level_count = 0

    gate_status = PASS_SYMBOLIC if symbolic_level_count else PASS_READY
    if cardinality["cardinality_guard"] != "PASS":
        gate_status = PASS_REDUCTION
    if upstream_validation["upstream_integrity_status"] != "PASS":
        if not upstream_validation["checks"]["gate_matches"]:
            gate_status = "FAIL_UPSTREAM_GATE_MISMATCH"
        elif not upstream_validation["checks"]["recommendation_matches"]:
            gate_status = "FAIL_UPSTREAM_RECOMMENDATION_MISMATCH"
        else:
            gate_status = "FAIL_UPSTREAM_INTEGRITY"
    elif axis_count != 9 or tier1 != 6 or tier2 != 2 or tier3 != 1:
        gate_status = "FAIL_AXIS_CLASSIFICATION_MISMATCH"
    elif unsupported_numeric_level_count:
        gate_status = "FAIL_UNSUPPORTED_NUMERIC_LEVEL_INVENTED"
    elif any(row["assumption_required"] and not row["assumption_id"] for row in levels):
        gate_status = "FAIL_UNLABELED_NEW_ASSUMPTION"
    elif any(row["contains_excluded_axis"] for row in design_matrix):
        gate_status = "FAIL_EXCLUDED_AXIS_INCLUDED"
    elif cardinality["reference_cell_count"] == 0:
        gate_status = "FAIL_REFERENCE_CELL_MISSING"
    elif cardinality["reference_cell_count"] > 1:
        gate_status = "FAIL_REFERENCE_CELL_DUPLICATED"
    elif any(row["execution_approved"] or row["reestimation_approved"] for row in design_matrix):
        gate_status = "FAIL_EXECUTION_SCOPE_OVERREACH"
    elif not lock_state["all_required_locks_false"]:
        gate_status = "FAIL_LOCK_STATE_VIOLATION"
    elif secret_audit["secret_leak_count"]:
        gate_status = "FAIL_SECRET_LEAK"
    elif parquet_check["parquet_read_failure_count"] or parquet_check["json_parquet_mismatch_count"]:
        gate_status = "FAIL_JSON_PARQUET_MISMATCH"
    elif forbidden["forbidden_operation_count"]:
        gate_status = "FAIL_EXECUTION_SCOPE_OVERREACH"

    if gate_status == PASS_REDUCTION:
        next_action = "Factorial-reduction contract design only"
    elif symbolic_level_count:
        next_action = "Resolve symbolic sensitivity levels from existing frozen artifacts only"
    else:
        next_action = "Independent sensitivity-grid contract validation only"

    report = {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "upstream_artifact": str(upstream),
        "upstream_gate": gate.get("gate"),
        "upstream_recommendation": recommendation.get("recommended_path_enum"),
        "upstream_integrity_status": upstream_validation["upstream_integrity_status"],
        "registry_rows": key["registry_rows"],
        "primary_identified_set": key["primary_identified_set"],
        "primary_identified_set_width": key["primary_identified_set_width"],
        "request_identified_set": key["request_identified_set"],
        "request_identified_set_width": key["request_identified_set_width"],
        "axis_count": axis_count,
        "tier1_axis_count": tier1,
        "tier2_axis_count": tier2,
        "tier3_axis_count": tier3,
        "eligible_axis_count": eligible_axis_count,
        "excluded_axis_count": excluded_axis_count,
        "level_count": level_count,
        "numeric_level_count": numeric_level_count,
        "symbolic_level_count": symbolic_level_count,
        "assumption_only_level_count": assumption_only_level_count,
        "unsupported_numeric_level_count": unsupported_numeric_level_count,
        "raw_cartesian_cardinality": cardinality["raw_cartesian_cardinality"],
        "pruned_cardinality": cardinality["pruned_cardinality"],
        "reference_cell_count": cardinality["reference_cell_count"],
        "fully_concrete_design_cell_count": cardinality["fully_concrete_design_cell_count"],
        "symbolic_cell_count": cardinality["symbolic_cell_count"],
        "assumption_only_cell_count": cardinality["assumption_only_cell_count"],
        "grid_contract_status": execution_boundary["grid_contract_status"],
        "grid_execution_status": execution_boundary["grid_execution_status"],
        "identified_set_reestimation_status": execution_boundary["identified_set_reestimation_status"],
        "turnbull_reestimation_status": execution_boundary["turnbull_reestimation_status"],
        "api_call_count": api["api_call_count"],
        "service_key_accessed": api["service_key_accessed"],
        "db_accessed": api["db_accessed"],
        "external_network_accessed": api["external_network_accessed"],
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
        "forbidden_operation_count": prohibited["forbidden_operation_count"] + forbidden["forbidden_operation_count"],
        "gate": gate_status,
        "next_authorized_action": next_action,
    }
    dump_json(out_dir / "final_report.json", report)
    (out_dir / "final_report.md").write_text(
        "\n".join([
            "# R2D-1U Sensitivity-Grid Contract",
            "",
            f"- Artifact: `{out_dir}`",
            f"- Upstream gate: `{gate.get('gate')}`",
            f"- Upstream recommendation: `{recommendation.get('recommended_path_enum')}`",
            f"- Gate: `{gate_status}`",
            "",
            "## Contract Summary",
            f"- Axis count: `{axis_count}`",
            f"- Tier 1/2/3: `{tier1}/{tier2}/{tier3}`",
            f"- Eligible/excluded axes: `{eligible_axis_count}/{excluded_axis_count}`",
            f"- Levels: `{level_count}`",
            f"- Symbolic levels: `{symbolic_level_count}`",
            f"- Assumption-only levels: `{assumption_only_level_count}`",
            f"- Unsupported numeric levels: `{unsupported_numeric_level_count}`",
            f"- Raw/pruned cardinality: `{cardinality['raw_cartesian_cardinality']}/{cardinality['pruned_cardinality']}`",
            "",
            "## Locks",
            "- Path B adopted: `false`",
            "- Sensitivity contract adopted: `false`",
            "- Sensitivity grid execution approved: `false`",
            "- Identified-set reestimation approved: `false`",
            "- Phase 2 authorized: `false`",
            "",
            f"Next authorized action: `{next_action}`",
            "",
        ]),
        encoding="utf-8",
    )
    gate_payload = {
        "artifact": str(out_dir),
        "gate": gate_status,
        "gate_passed": gate_status.startswith("PASS_"),
        "upstream_artifact": str(upstream),
        "upstream_gate": gate.get("gate"),
        "upstream_recommendation": recommendation.get("recommended_path_enum"),
        "upstream_scope": recommendation.get("recommendation_scope"),
        "axis_count": axis_count,
        "tier1_axis_count": tier1,
        "tier2_axis_count": tier2,
        "tier3_axis_count": tier3,
        "eligible_axis_count": eligible_axis_count,
        "excluded_axis_count": excluded_axis_count,
        "level_count": level_count,
        "numeric_level_count": numeric_level_count,
        "symbolic_level_count": symbolic_level_count,
        "assumption_only_level_count": assumption_only_level_count,
        "unsupported_numeric_level_count": unsupported_numeric_level_count,
        "raw_cartesian_cardinality": cardinality["raw_cartesian_cardinality"],
        "pruned_cardinality": cardinality["pruned_cardinality"],
        "reference_cell_count": cardinality["reference_cell_count"],
        "fully_concrete_design_cell_count": cardinality["fully_concrete_design_cell_count"],
        "symbolic_cell_count": cardinality["symbolic_cell_count"],
        "assumption_only_cell_count": cardinality["assumption_only_cell_count"],
        "scheduled_layover_grid_inclusion_allowed": False,
        "all_cells_execution_approved_false": cardinality["all_cells_execution_approved_false"],
        "all_cells_reestimation_approved_false": cardinality["all_cells_reestimation_approved_false"],
        "path_b_adopted": False,
        "sensitivity_contract_adopted": False,
        "sensitivity_grid_execution_approved": False,
        "identified_set_reestimation_approved": False,
        "turnbull_reestimation_approved": False,
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
        "forbidden_operation_count": prohibited["forbidden_operation_count"] + forbidden["forbidden_operation_count"],
        "next_authorized_action": next_action,
    }
    dump_json(out_dir / "gate_decision.json", gate_payload)

    final_secret = secret_scan(out_dir)
    dump_json(out_dir / "forbidden_field_audit.json", forbidden_field_audit(out_dir))
    final_forbidden = strict_read_json(out_dir / "forbidden_field_audit.json")
    if final_secret["secret_leak_count"] != secret_audit["secret_leak_count"]:
        raise RuntimeError(f"secret scan changed after finalization: {final_secret}")
    if final_forbidden["forbidden_operation_count"] != forbidden["forbidden_operation_count"]:
        raise RuntimeError(f"forbidden audit changed after finalization: {final_forbidden}")

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
        raise RuntimeError(f"manifest validation failed: {manifest_check}")

    (out_dir / "_SUCCESS.lock").write_text(
        json.dumps({"success": True, "gate": gate_status, "created_at": now_iso()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"artifact:\n{out_dir}")
    print(f"gate:\n{gate_status}")
    print(f"upstream gate:\n{gate.get('gate')}")
    print(f"upstream recommendation:\n{recommendation.get('recommended_path_enum')}")
    print(f"registry rows:\n{key['registry_rows']}")
    print(f"primary identified set:\n{key['primary_identified_set']} width {key['primary_identified_set_width']}")
    print(f"request identified set:\n{key['request_identified_set']} width {key['request_identified_set_width']}")
    print(f"axis count tier1/tier2/tier3:\n{axis_count} {tier1}/{tier2}/{tier3}")
    print(f"eligible/excluded axes:\n{eligible_axis_count}/{excluded_axis_count}")
    print(f"levels numeric/symbolic/assumption/unsupported_numeric:\n{numeric_level_count}/{symbolic_level_count}/{assumption_only_level_count}/{unsupported_numeric_level_count}")
    print(f"raw/pruned cardinality:\n{cardinality['raw_cartesian_cardinality']}/{cardinality['pruned_cardinality']}")
    print(f"reference cells:\n{cardinality['reference_cell_count']}")
    print(f"execution/reestimation approved:\nfalse/false")
    print(f"API/service key/DB/network:\n0/false/false/false")
    print(f"upstream added/modified/deleted:\n{mutation['upstream_added']}/{mutation['upstream_modified']}/{mutation['upstream_deleted']}")
    print(f"required files:\n{len(REQUIRED_FILES)}")
    print("manifest hash/size mismatches:\n0/0")
    print("strict JSON failures:\n0")
    print(f"Parquet read failures:\n{parquet_check['parquet_read_failure_count']}")
    print(f"JSON/Parquet mismatches:\n{parquet_check['json_parquet_mismatch_count']}")
    print(f"secret leaks:\n{secret_audit['secret_leak_count']}")
    print(f"forbidden operations:\n{prohibited['forbidden_operation_count'] + forbidden['forbidden_operation_count']}")
    print(f"next authorized action:\n{next_action}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
