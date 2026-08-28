from __future__ import annotations

import argparse
import hashlib
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
    "prompt5_e01_r2d1w_sensitivity_axis_orthogonality_refinement_contract_20260731_064704"
)
EXPECTED_UPSTREAM_GATE = "PASS_SENSITIVITY_AXIS_ORTHOGONALITY_REFINEMENT_CONTRACT_READY_INTERACTIONS_REMAIN_STILL_LOCKED"
EXPECTED_NEXT = "Refined symbolic level-domain specification only"
PASS_AMBIGUITY = "PASS_REFINED_SYMBOLIC_LEVEL_DOMAIN_SPECIFICATION_PARTIAL_DOMAIN_KIND_AMBIGUITY_STILL_LOCKED"

SYMBOLIC_AXES = [
    "provider_endpoint_uncertainty",
    "request_endpoint_uncertainty",
    "clock_rounding_allowance",
    "interval_censoring_convention",
    "endpoint_inclusion_exclusion_convention",
    "provider_request_precedence_rule",
]
ASSUMPTION_ONLY_AXES = ["clock_skew_allowance", "missing_boundary_treatment"]
EXCLUDED_AXIS = "scheduled_layover_treatment"

UPSTREAM_LOCK_FIELDS = [
    "path_b_adopted",
    "sensitivity_contract_adopted",
    "orthogonality_contract_adopted",
    "primitive_operator_contract_adopted",
    "axis_disposition_adopted",
    "resolved_level_contract_adopted",
    "sensitivity_grid_execution_approved",
    "identified_set_reestimation_approved",
    "turnbull_reestimation_approved",
    "phase2_authorized",
]

LOCK_FALSE_FIELDS = [
    "path_b_adopted",
    "sensitivity_contract_adopted",
    "orthogonality_contract_adopted",
    "primitive_operator_contract_adopted",
    "symbolic_domain_contract_adopted",
    "stage_interaction_contract_adopted",
    "candidate_stage_order_adopted",
    "resolved_level_contract_adopted",
    "actual_level_resolution_approved",
    "actual_level_resolution_executed",
    "sensitivity_grid_execution_approved",
    "sensitivity_analysis_executed",
    "identified_set_reestimation_approved",
    "identified_set_reestimation_executed",
    "turnbull_reestimation_approved",
    "turnbull_reestimation_executed",
    "new_point_estimate_approved",
    "new_point_estimate_computed",
    "new_assumption_approved",
    "new_observation_approved",
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

REQUIRED_FILES = [
    "upstream_validation.json",
    "upstream_lineage.json",
    "r2d1w_gate_snapshot.json",
    "r2d1w_operator_snapshot.json",
    "r2d1w_orthogonality_snapshot.json",
    "r2d1w_design_matrix_snapshot.json",
    "r2d1w_lock_snapshot.json",
    "symbolic_domain_definition_contract.json",
    "symbolic_domain_registry.json",
    "symbolic_domain_registry.parquet",
    "reference_element_registry.json",
    "reference_element_registry.parquet",
    "alternate_placeholder_family_registry.json",
    "alternate_placeholder_family_registry.parquet",
    "axis_domain_admissibility_registry.json",
    "axis_domain_admissibility_registry.parquet",
    "assumption_only_domain_preservation_audit.json",
    "scheduled_layover_exclusion_audit.json",
    "stage_interaction_constraint_registry.json",
    "stage_interaction_constraint_registry.parquet",
    "candidate_stage_order_contract.json",
    "domain_dependency_graph.json",
    "domain_dependency_edge_registry.json",
    "domain_dependency_edge_registry.parquet",
    "domain_kind_ambiguity_registry.json",
    "unresolved_stage_order_registry.json",
    "domain_readiness_registry.json",
    "domain_readiness_registry.parquet",
    "original_design_matrix_domain_overlay.json",
    "original_design_matrix_domain_overlay.parquet",
    "domain_cardinality_audit.json",
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
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [sanitize(v) for v in value]
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


def pipe(values: list[str]) -> str:
    return "|".join(values)


def file_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    return {
        p.relative_to(root).as_posix(): {"sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def compare_snapshot(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_keys = set(before)
    after_keys = set(after)
    modified = sorted(k for k in before_keys & after_keys if before[k] != after[k])
    return {
        "upstream_added": len(after_keys - before_keys),
        "upstream_modified": len(modified),
        "upstream_deleted": len(before_keys - after_keys),
        "added_files": sorted(after_keys - before_keys),
        "modified_files": modified,
        "deleted_files": sorted(before_keys - after_keys),
    }


def validate_manifest(artifact: Path) -> dict[str, Any]:
    manifest_path = artifact / "artifact_manifest.json"
    if not manifest_path.is_file():
        return {"status": "FAIL", "reason": "artifact_manifest.json missing"}
    manifest = strict_read_json(manifest_path)
    hash_mismatch = []
    size_mismatch = []
    json_fail = []
    missing = []
    for entry in manifest.get("files", []):
        rel = entry["relative_path"]
        p = artifact / rel
        if not p.is_file():
            missing.append(rel)
            continue
        if entry.get("sha256") != sha256_file(p):
            hash_mismatch.append(rel)
        if entry.get("size_bytes") != p.stat().st_size:
            size_mismatch.append(rel)
        if p.suffix == ".json":
            try:
                strict_read_json(p)
            except Exception as exc:
                json_fail.append({"relative_path": rel, "error": type(exc).__name__})
    return {
        "status": "PASS" if not (hash_mismatch or size_mismatch or json_fail or missing) else "FAIL",
        "manifest_hash_mismatch_count": len(hash_mismatch),
        "manifest_size_mismatch_count": len(size_mismatch),
        "strict_json_failure_count": len(json_fail),
        "manifest_missing_count": len(missing),
    }


def validate_upstream(upstream: Path) -> dict[str, Any]:
    gate = strict_read_json(upstream / "gate_decision.json")
    locks = strict_read_json(upstream / "lock_state.json")
    cardinality = strict_read_json(upstream / "orthogonality_cardinality_audit.json")
    manifest = validate_manifest(upstream)
    upstream_metrics = dict(gate)
    upstream_metrics.update({
        "order_dependent_cell_count": cardinality.get("order_dependent_cell_count"),
        "composite_refactor_cell_count": cardinality.get("composite_refactor_cell_count"),
    })
    expected = {
        "symbolic_target_count": 6,
        "primitive_operator_count": 8,
        "mapped_axis_count": 8,
        "unmapped_axis_count": 1,
        "pairwise_refinement_count": 4,
        "semantically_orthogonal_pair_count": 4,
        "interacting_pair_count": 4,
        "order_dependent_pair_count": 4,
        "nonorthogonal_pair_count": 0,
        "parameter_ownership_conflict_count": 0,
        "stage_ownership_conflict_count": 0,
        "original_design_cell_count": 13,
        "audited_design_cell_count": 13,
        "reference_cell_count": 1,
        "order_dependent_cell_count": 10,
        "composite_refactor_cell_count": 0,
        "empirical_level_resolved_count": 0,
        "new_numeric_level_count": 0,
        "new_alternate_enum_count": 0,
        "new_assumption_count": 0,
        "operator_execution_count": 0,
    }
    mismatches = [k for k, v in expected.items() if upstream_metrics.get(k) != v]
    lock_violations = [k for k in UPSTREAM_LOCK_FIELDS if locks.get(k) is not False]
    checks = {
        "gate_matches": gate.get("gate") == EXPECTED_UPSTREAM_GATE,
        "next_action_matches": gate.get("next_authorized_action") == EXPECTED_NEXT,
        "counts_match": not mismatches,
        "required_locks_false": not lock_violations,
        "manifest_integrity_pass": manifest["status"] == "PASS",
    }
    return {
        "upstream_artifact": str(upstream),
        "upstream_gate": gate.get("gate"),
        "upstream_next_authorized_action": gate.get("next_authorized_action"),
        "checks": checks,
        "count_mismatches": mismatches,
        "count_source_files": {
            "gate_decision_json_fields": sorted(set(expected) - {"order_dependent_cell_count", "composite_refactor_cell_count"}),
            "orthogonality_cardinality_audit_json_fields": [
                "order_dependent_cell_count",
                "composite_refactor_cell_count",
            ],
        },
        "lock_violations": lock_violations,
        "manifest_validation": manifest,
        "upstream_integrity_status": "PASS" if all(checks.values()) else "FAIL",
    }


def build_domains() -> list[dict[str, Any]]:
    specs = {
        "provider_endpoint_uncertainty": (
            "U_PROVIDER",
            "DOMAIN_PROVIDER_SUPPORT",
            "SYMBOLIC_ORDERED_SUPPORT_DOMAIN",
            "provider source-local endpoint-support object",
            "REF_PROVIDER_ENDPOINT_SUPPORT",
            "FROZEN_PROVIDER_ENDPOINT_SUPPORT_REFERENCE",
            "PROVIDER_SUPPORT_ALTERNATE_UNSPECIFIED",
            "lower <= upper",
            "source timestamp coordinate unit, unresolved",
            "provider support changes only; request support, clock transformation, source reconciliation and boundary topology fixed",
            "SYMBOLIC_DOMAIN_STRUCTURALLY_SPECIFIED",
        ),
        "request_endpoint_uncertainty": (
            "U_REQUEST",
            "DOMAIN_REQUEST_SUPPORT",
            "SYMBOLIC_ORDERED_SUPPORT_DOMAIN",
            "request source-local endpoint-support object",
            "REF_REQUEST_ENDPOINT_SUPPORT",
            "FROZEN_REQUEST_ENDPOINT_SUPPORT_REFERENCE",
            "REQUEST_SUPPORT_ALTERNATE_UNSPECIFIED",
            "lower <= upper",
            "source timestamp coordinate unit, unresolved",
            "request support changes only; provider support, clock transformation, source reconciliation and boundary topology fixed",
            "SYMBOLIC_DOMAIN_STRUCTURALLY_SPECIFIED",
        ),
        "clock_rounding_allowance": (
            "Q_CLOCK",
            "DOMAIN_CLOCK_QUANTIZATION",
            "SYMBOLIC_ORDERED_SUPPORT_DOMAIN",
            "timestamp quantization convention descriptor OR timestamp resolution support descriptor",
            "REF_CLOCK_REPRESENTATION",
            "FROZEN_CLOCK_REPRESENTATION_REFERENCE",
            "CLOCK_QUANTIZATION_ALTERNATE_UNSPECIFIED",
            "unresolved: convention set or ordered support",
            "unit unresolved; no clock resolution inferred",
            "clock offset, provider/request support and source reconciliation fixed",
            "SYMBOLIC_DOMAIN_KIND_AMBIGUOUS",
        ),
        "interval_censoring_convention": (
            "C_INTERVAL",
            "DOMAIN_CENSORING_STATE",
            "SYMBOLIC_UNORDERED_CONVENTION_DOMAIN",
            "observation-state classification convention",
            "REF_CENSORING_STATE",
            "FROZEN_CENSORING_REFERENCE",
            "CENSORING_STATE_ALTERNATE_UNSPECIFIED",
            "unordered symbolic convention",
            "no numeric unit",
            "observability classification only; endpoint coordinate, boundary inclusion, missingness and reconciliation fixed",
            "SYMBOLIC_DOMAIN_STRUCTURALLY_SPECIFIED",
        ),
        "endpoint_inclusion_exclusion_convention": (
            "B_ENDPOINT",
            "DOMAIN_BOUNDARY_TOPOLOGY",
            "SYMBOLIC_TOPOLOGY_DOMAIN",
            "boundary membership topology descriptor",
            "REF_BOUNDARY_TOPOLOGY",
            "FROZEN_BOUNDARY_TOPOLOGY_REFERENCE",
            "BOUNDARY_TOPOLOGY_ALTERNATE_UNSPECIFIED",
            "topological membership relation",
            "no numeric unit",
            "boundary membership only; endpoint coordinate, censoring state, missingness and reconciliation fixed",
            "SYMBOLIC_DOMAIN_STRUCTURALLY_SPECIFIED",
        ),
        "provider_request_precedence_rule": (
            "P_SOURCE",
            "DOMAIN_SOURCE_RECONCILIATION",
            "SYMBOLIC_RECONCILIATION_RULE_DOMAIN",
            "source reconciliation rule descriptor",
            "REF_SOURCE_RECONCILIATION",
            "FROZEN_SOURCE_RECONCILIATION_REFERENCE",
            "SOURCE_RECONCILIATION_ALTERNATE_UNSPECIFIED",
            "symbolic reconciliation rule relation",
            "no numeric unit",
            "provider/request source-local support, clock transformation, boundary topology and censoring state fixed",
            "SYMBOLIC_DOMAIN_STRUCTURALLY_SPECIFIED",
        ),
        "clock_skew_allowance": (
            "S_CLOCK",
            "DOMAIN_CLOCK_SKEW_ASSUMPTION_ONLY",
            "ASSUMPTION_ONLY_NUMERIC_DOMAIN_UNSPECIFIED",
            "clock offset assumption-only numeric domain",
            "",
            "",
            "",
            "numeric bounds undefined",
            "unit undefined",
            "assumption required; empirically observed false; no numeric bounds",
            "ASSUMPTION_ONLY_DOMAIN_PRESERVED",
        ),
        "missing_boundary_treatment": (
            "M_BOUNDARY",
            "DOMAIN_MISSING_BOUNDARY_ASSUMPTION_ONLY",
            "ASSUMPTION_ONLY_CONVENTION_DOMAIN_UNSPECIFIED",
            "missing-boundary assumption-only convention domain",
            "",
            "",
            "",
            "convention members undefined",
            "no numeric unit",
            "assumption required; empirically observed false; no convention members",
            "ASSUMPTION_ONLY_DOMAIN_PRESERVED",
        ),
        "scheduled_layover_treatment": (
            "",
            "DOMAIN_SCHEDULED_LAYOVER_EXCLUDED",
            "EXCLUDED_REQUIRES_OBSERVATION",
            "excluded domain",
            "",
            "",
            "",
            "not applicable",
            "not applicable",
            "requires new observation; grid inclusion false",
            "EXCLUDED_REQUIRES_NEW_OBSERVATION",
        ),
    }
    rows = []
    for axis_id, values in specs.items():
        (
            op,
            domain_id,
            domain_kind,
            shape,
            ref_id,
            ref_label,
            placeholder,
            ordering,
            unit,
            admissibility,
            readiness,
        ) = values
        symbolic = axis_id in SYMBOLIC_AXES
        domain_kind_candidates = [domain_kind]
        domain_kind_status = "DOMAIN_KIND_SPECIFIED"
        if axis_id == "clock_rounding_allowance":
            domain_kind_candidates = [
                "SYMBOLIC_ORDERED_SUPPORT_DOMAIN",
                "SYMBOLIC_UNORDERED_CONVENTION_DOMAIN",
            ]
            domain_kind_status = "DOMAIN_KIND_AMBIGUOUS"
        rows.append({
            "axis_id": axis_id,
            "primitive_operator_id": op,
            "domain_id": domain_id,
            "domain_kind": domain_kind,
            "domain_kind_status": domain_kind_status,
            "domain_kind_candidate_set": pipe(domain_kind_candidates),
            "parameter_shape": shape,
            "reference_element_id": ref_id,
            "reference_element_source": "R2D-1W primitive operator and frozen reference contract" if ref_id else "",
            "alternate_placeholder_family": placeholder,
            "ordering_relation": ordering,
            "unit_contract": unit,
            "value_constraints": "actual member absent; numeric bounds absent; enum members absent",
            "admissibility_predicate": admissibility,
            "stage_preconditions": "candidate stage input type must match primitive operator signature",
            "stage_postconditions": "candidate stage output remains symbolic and unexecuted",
            "interaction_constraints": "order-dependent constraints are recorded separately",
            "evidence_requirement": "frozen evidence or explicit assumption/new observation boundary must be reviewed later",
            "allowed_resolution_routes": "frozen evidence|explicit assumption boundary review|new observation design review",
            "forbidden_resolution_routes": "invented numeric bound|invented enum member|operator execution|API or DB evidence",
            "domain_complete": False,
            "candidate_only": True,
            "adopted": False,
            "execution_allowed": False,
            "symbolic_domain_target": symbolic,
            "readiness_status": readiness,
        })
    return rows


def reference_elements(domains: list[dict[str, Any]], upstream: Path) -> list[dict[str, Any]]:
    rows = []
    for row in domains:
        if row["axis_id"] not in SYMBOLIC_AXES:
            continue
        rows.append({
            "reference_element_id": row["reference_element_id"],
            "axis_id": row["axis_id"],
            "source_artifact": str(upstream),
            "source_file": "primitive_operator_registry.json",
            "source_locator": f"/records/mapped_axis_id={row['axis_id']}",
            "reference_semantics": row["reference_element_source"],
            "reference_value_type": "SYMBOLIC_FROZEN_REFERENCE_LABEL",
            "reference_value_redacted_or_symbolic": row["reference_element_id"].replace("REF_", "FROZEN_") + "_REFERENCE",
            "empirically_observed": True,
            "adopted": False,
        })
    return rows


def placeholder_families(domains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "axis_id": row["axis_id"],
            "domain_id": row["domain_id"],
            "placeholder_family_id": row["alternate_placeholder_family"],
            "placeholder_only": True,
            "resolved": False,
            "empirically_supported": False,
            "execution_allowed": False,
            "actual_member_created": False,
            "forbidden_concrete_alternates_excluded": True,
            "forbidden_concrete_alternate_materialized_count": 0,
        }
        for row in domains
        if row["axis_id"] in SYMBOLIC_AXES
    ]


def admissibility_registry(domains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "axis_id": row["axis_id"],
            "domain_id": row["domain_id"],
            "primitive_operator_id": row["primitive_operator_id"],
            "admissibility_predicate": row["admissibility_predicate"],
            "stage_preconditions": row["stage_preconditions"],
            "stage_postconditions": row["stage_postconditions"],
            "actual_level_resolution_ready": False,
            "candidate_execution_ready": False,
            "execution_allowed": False,
        }
        for row in domains
    ]


def stage_constraints() -> list[dict[str, Any]]:
    return [
        {
            "interaction_id": "INT_C_INTERVAL_B_ENDPOINT",
            "operator_a": "C_INTERVAL",
            "operator_b": "B_ENDPOINT",
            "semantic_orthogonality": True,
            "stage_order_required": True,
            "candidate_preferred_order": "C_INTERVAL -> B_ENDPOINT",
            "reverse_order_allowed": "SEMANTICALLY_POSSIBLE_BUT_UNDERDEFINED",
            "commutativity_status": "NONCOMMUTATIVE_BY_STAGE_DEPENDENCY",
            "interaction_type": "ORDER_DEPENDENT_CONVENTION_TO_TOPOLOGY",
            "domain_resolution_dependency": "CENSORING_STATE_DOMAIN_BEFORE_BOUNDARY_TOPOLOGY_DOMAIN",
            "execution_allowed": False,
            "adopted": False,
        },
        {
            "interaction_id": "INT_U_PROVIDER_P_SOURCE",
            "operator_a": "U_PROVIDER",
            "operator_b": "P_SOURCE",
            "semantic_orthogonality": True,
            "stage_order_required": True,
            "candidate_preferred_order": "U_PROVIDER -> P_SOURCE",
            "reverse_order_allowed": "false",
            "commutativity_status": "NONCOMMUTATIVE_BY_STAGE_DEPENDENCY",
            "interaction_type": "SOURCE_LOCAL_SUPPORT_BEFORE_RECONCILIATION",
            "domain_resolution_dependency": "PROVIDER_SUPPORT_DOMAIN_BEFORE_SOURCE_RECONCILIATION_DOMAIN",
            "execution_allowed": False,
            "adopted": False,
        },
        {
            "interaction_id": "INT_Q_CLOCK_U_REQUEST",
            "operator_a": "Q_CLOCK",
            "operator_b": "U_REQUEST",
            "semantic_orthogonality": True,
            "stage_order_required": True,
            "candidate_preferred_order": "Q_CLOCK -> U_REQUEST",
            "reverse_order_allowed": "false",
            "commutativity_status": "NONCOMMUTATIVE_BY_STAGE_DEPENDENCY",
            "interaction_type": "CLOCK_DOMAIN_BEFORE_REQUEST_SUPPORT",
            "domain_resolution_dependency": "CLOCK_DOMAIN_KIND_MUST_BE_RESOLVED_FIRST",
            "execution_allowed": False,
            "adopted": False,
        },
        {
            "interaction_id": "INT_Q_CLOCK_S_CLOCK",
            "operator_a": "Q_CLOCK",
            "operator_b": "S_CLOCK",
            "semantic_orthogonality": True,
            "stage_order_required": True,
            "candidate_preferred_order": "UNRESOLVED",
            "reverse_order_allowed": "UNRESOLVED",
            "commutativity_status": "UNKNOWN_WITHOUT_LEVEL_DEFINITION",
            "interaction_type": "ORDER_ADJUDICATION_REQUIRED",
            "domain_resolution_dependency": "CLOCK_QUANTIZATION_AND_OFFSET_ORDER_UNRESOLVED",
            "execution_allowed": False,
            "adopted": False,
        },
    ]


def dependency_edges() -> list[dict[str, Any]]:
    return [
        ("DEP_00", "DOMAIN_CLOCK_QUANTIZATION", "DOMAIN_REQUEST_SUPPORT", "domain resolution dependency", "hard", True, True),
        ("DEP_01", "DOMAIN_PROVIDER_SUPPORT", "DOMAIN_SOURCE_RECONCILIATION", "stage dependency", "hard", True, True),
        ("DEP_02", "DOMAIN_REQUEST_SUPPORT", "DOMAIN_SOURCE_RECONCILIATION", "stage dependency", "hard", True, True),
        ("DEP_03", "DOMAIN_CENSORING_STATE", "DOMAIN_BOUNDARY_TOPOLOGY", "stage dependency", "hard", True, True),
        ("DEP_04", "DOMAIN_CLOCK_QUANTIZATION", "DOMAIN_CLOCK_SKEW_ASSUMPTION_ONLY", "ordering adjudication", "soft", False, True),
    ]


def edge_records() -> list[dict[str, Any]]:
    return [
        {
            "dependency_id": dep_id,
            "source_domain": src,
            "target_domain": dst,
            "dependency_type": dep_type,
            "hard_or_soft": hard,
            "resolution_blocking": resolution_blocking,
            "execution_blocking": execution_blocking,
        }
        for dep_id, src, dst, dep_type, hard, resolution_blocking, execution_blocking in dependency_edges()
    ]


def design_overlay(design_records: list[dict[str, Any]], domains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kind_by_axis = {row["axis_id"]: row["readiness_status"] for row in domains}
    binding_to_axis = {
        "provider_endpoint_uncertainty__symbolic_alternate": "provider_endpoint_uncertainty",
        "request_endpoint_uncertainty__symbolic_alternate": "request_endpoint_uncertainty",
        "clock_rounding_allowance__symbolic_alternate": "clock_rounding_allowance",
        "interval_censoring_convention__symbolic_alternate": "interval_censoring_convention",
        "endpoint_inclusion_exclusion_convention__symbolic_alternate": "endpoint_inclusion_exclusion_convention",
        "provider_request_precedence_rule__symbolic_alternate": "provider_request_precedence_rule",
        "clock_skew_allowance__assumption_only_alternate": "clock_skew_allowance",
        "missing_boundary_treatment__assumption_only_alternate": "missing_boundary_treatment",
    }
    rows = []
    for cell in design_records:
        bindings = json.loads(cell["axis_level_bindings"], parse_constant=strict_constant)
        axes = [binding_to_axis[v] for v in bindings.values() if v in binding_to_axis]
        ambiguous = sum(kind_by_axis.get(axis) == "SYMBOLIC_DOMAIN_KIND_AMBIGUOUS" for axis in axes)
        hard_deps = 0
        if "clock_rounding_allowance" in axes and "request_endpoint_uncertainty" in axes:
            hard_deps += 1
        if "provider_endpoint_uncertainty" in axes and "provider_request_precedence_rule" in axes:
            hard_deps += 1
        if "request_endpoint_uncertainty" in axes and "provider_request_precedence_rule" in axes:
            hard_deps += 1
        if "interval_censoring_convention" in axes and "endpoint_inclusion_exclusion_convention" in axes:
            hard_deps += 1
        unresolved_stage = 1 if "clock_rounding_allowance" in axes and "clock_skew_allowance" in axes else 0
        rows.append({
            "design_cell_id": cell["design_cell_id"],
            "cell_label": cell["cell_label"],
            "axis_level_bindings": cell["axis_level_bindings"],
            "reference_cell": bool(cell.get("reference_cell")),
            "domain_kind_complete": ambiguous == 0,
            "placeholder_family_complete": True,
            "stage_constraints_complete": unresolved_stage == 0,
            "hard_domain_dependency_count": hard_deps,
            "unresolved_domain_kind_count": ambiguous,
            "unresolved_stage_order_count": unresolved_stage,
            "actual_level_resolution_ready": False,
            "candidate_execution_ready": False,
            "execution_approved": False,
            "reestimation_approved": False,
        })
    return rows


def parquet_json_check(out_dir: Path) -> dict[str, Any]:
    pairs = [
        ("symbolic_domain_registry.json", "symbolic_domain_registry.parquet", "records"),
        ("reference_element_registry.json", "reference_element_registry.parquet", "records"),
        ("alternate_placeholder_family_registry.json", "alternate_placeholder_family_registry.parquet", "records"),
        ("axis_domain_admissibility_registry.json", "axis_domain_admissibility_registry.parquet", "records"),
        ("stage_interaction_constraint_registry.json", "stage_interaction_constraint_registry.parquet", "records"),
        ("domain_dependency_edge_registry.json", "domain_dependency_edge_registry.parquet", "records"),
        ("domain_readiness_registry.json", "domain_readiness_registry.parquet", "records"),
        ("original_design_matrix_domain_overlay.json", "original_design_matrix_domain_overlay.parquet", "records"),
    ]
    failures = []
    mismatches = 0
    for js_name, pq_name, key in pairs:
        try:
            records = strict_read_json(out_dir / js_name)[key]
            df = pd.read_parquet(out_dir / pq_name)
        except Exception as exc:
            failures.append({"json": js_name, "parquet": pq_name, "error": type(exc).__name__})
            continue
        if len(records) != len(df):
            mismatches += 1
            continue
        if records and sorted(records[0].keys()) != sorted(df.columns.tolist()):
            mismatches += 1
            continue
        for idx, record in enumerate(records):
            row = df.iloc[idx].to_dict()
            for k, v in record.items():
                other = row.get(k)
                if isinstance(v, float):
                    if abs(v - float(other)) > 1e-9:
                        mismatches += 1
                elif v != other:
                    mismatches += 1
    return {
        "parquet_read_failure_count": len(failures),
        "json_parquet_mismatch_count": mismatches,
        "parquet_failures": failures,
    }


SECRET_PATTERNS = [
    re.compile(r"serviceKey\s*=\s*[A-Za-z0-9%+/=]{12,}", re.I),
    re.compile(r"(api_key|apikey|client_secret|access_token|refresh_token|password|passwd)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}", re.I),
    re.compile(r"(authorization|bearer)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{20,}", re.I),
    re.compile(r"(postgresql|mysql|mongodb)://[^\s\"']+", re.I),
]


def secret_scan(out_dir: Path) -> dict[str, Any]:
    findings = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.suffix in {".json", ".md", ".lock"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in SECRET_PATTERNS:
                if pattern.search(text):
                    findings.append({"relative_path": path.name, "pattern_sha256": hashlib.sha256(pattern.pattern.encode()).hexdigest()})
    return {"secret_leak_count": len(findings), "findings": findings}


FORBIDDEN_KEYS = {
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
    for path in sorted(out_dir.glob("*.json")):
        if path.name in {"forbidden_field_audit.json", "prohibited_operation_audit.json"}:
            continue
        try:
            payload = strict_read_json(path)
        except Exception:
            continue
        for key_path, value in iter_keys(payload):
            if key_path[-1] in FORBIDDEN_KEYS and value not in (False, None, "PROHIBITED", "NOT_AUTHORIZED"):
                findings.append({"relative_path": path.name, "key_path": ".".join(key_path), "value_type": type(value).__name__})
    return {
        "forbidden_operation_count": len(findings),
        "forbidden_field_presence_count": len(findings),
        "findings": findings,
        "forbidden_field_inventory": sorted(FORBIDDEN_KEYS),
    }


def artifact_role_for(name: str) -> str:
    if name.startswith("upstream") or name.startswith("r2d1w") or name == "lineage_field_audit.json":
        return "UPSTREAM_VALIDATION"
    if "domain" in name or "reference" in name or "placeholder" in name:
        return "DOMAIN_CONTRACT"
    if "stage" in name or "dependency" in name:
        return "STAGE_INTERACTION_CONTRACT"
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
    actual = {p.name for p in out_dir.iterdir() if p.is_file() and p.name not in MANIFEST_EXCLUDED}
    expected = set(REQUIRED_FILES) - MANIFEST_EXCLUDED
    return {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "required_file_count": len(REQUIRED_FILES),
        "manifest_excluded_files": sorted(MANIFEST_EXCLUDED),
        "files": entries,
        "missing_required_file_count": sum(not e["exists"] for e in entries),
        "unexpected_file_count": len(actual - expected),
        "unexpected_files": sorted(actual - expected),
        "duplicate_manifest_path_count": len(entries) - len({e["relative_path"] for e in entries}),
        "absolute_path_leak_count": sum(Path(e["relative_path"]).is_absolute() for e in entries),
    }


def verify_manifest(out_dir: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    hash_mismatch = []
    size_mismatch = []
    json_fail = []
    for entry in manifest["files"]:
        path = out_dir / entry["relative_path"]
        if not path.is_file():
            continue
        if entry["sha256"] != sha256_file(path):
            hash_mismatch.append(entry["relative_path"])
        if entry["size_bytes"] != path.stat().st_size:
            size_mismatch.append(entry["relative_path"])
        if path.suffix == ".json":
            try:
                strict_read_json(path)
            except Exception as exc:
                json_fail.append({"relative_path": entry["relative_path"], "error": type(exc).__name__})
    return {
        "manifest_hash_mismatch_count": len(hash_mismatch),
        "manifest_size_mismatch_count": len(size_mismatch),
        "strict_json_failure_count": len(json_fail),
        "required_file_missing_count": manifest["missing_required_file_count"],
        "unexpected_file_count": manifest["unexpected_file_count"],
        "duplicate_manifest_path_count": manifest["duplicate_manifest_path_count"],
        "absolute_path_leak_count": manifest["absolute_path_leak_count"],
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

    out_dir = project_root / "05_training" / "artifacts" / f"prompt5_e01_r2d1x_refined_symbolic_level_domain_specification_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    before = file_snapshot(upstream)
    gate = strict_read_json(upstream / "gate_decision.json")
    operators = strict_read_json(upstream / "primitive_operator_registry.json")
    orthogonality = strict_read_json(upstream / "pairwise_orthogonality_refinement.json")
    design = strict_read_json(upstream / "original_design_matrix_confounding_audit.json")
    locks = strict_read_json(upstream / "lock_state.json")
    upstream_validation = validate_upstream(upstream)

    dump_json(out_dir / "upstream_validation.json", upstream_validation)
    dump_json(out_dir / "upstream_lineage.json", {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "authoritative_upstream_artifact": str(upstream),
        "authoritative_upstream_gate": gate.get("gate"),
        "source_files_read": [
            "gate_decision.json",
            "primitive_operator_registry.json",
            "pairwise_orthogonality_refinement.json",
            "original_design_matrix_confounding_audit.json",
            "lock_state.json",
            "artifact_manifest.json",
        ],
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
    })
    dump_json(out_dir / "r2d1w_gate_snapshot.json", gate)
    dump_json(out_dir / "r2d1w_operator_snapshot.json", operators)
    dump_json(out_dir / "r2d1w_orthogonality_snapshot.json", orthogonality)
    dump_json(out_dir / "r2d1w_design_matrix_snapshot.json", design)
    dump_json(out_dir / "r2d1w_lock_snapshot.json", locks)

    domains = build_domains()
    refs = reference_elements(domains, upstream)
    placeholders = placeholder_families(domains)
    admissibility = admissibility_registry(domains)
    constraints = stage_constraints()
    edges = edge_records()
    overlay = design_overlay(design["records"], domains)
    readiness = [
        {
            "axis_id": row["axis_id"],
            "domain_id": row["domain_id"],
            "domain_readiness_status": row["readiness_status"],
            "domain_complete": False,
            "actual_level_resolution_ready": False,
            "candidate_execution_ready": False,
            "execution_allowed": False,
        }
        for row in domains
    ]

    dump_json(out_dir / "symbolic_domain_definition_contract.json", {
        "allowed_domain_kinds": pipe([
            "SYMBOLIC_ORDERED_SUPPORT_DOMAIN",
            "SYMBOLIC_UNORDERED_CONVENTION_DOMAIN",
            "SYMBOLIC_TOPOLOGY_DOMAIN",
            "SYMBOLIC_RECONCILIATION_RULE_DOMAIN",
            "ASSUMPTION_ONLY_NUMERIC_DOMAIN_UNSPECIFIED",
            "ASSUMPTION_ONLY_CONVENTION_DOMAIN_UNSPECIFIED",
            "EXCLUDED_REQUIRES_OBSERVATION",
        ]),
        "actual_alternate_member_created": False,
        "numeric_domain_bound_created": False,
        "domain_complete_allowed": False,
        "candidate_only": True,
        "adopted": False,
        "execution_allowed": False,
    })
    dump_json(out_dir / "symbolic_domain_registry.json", {"domain_count": len(domains), "symbolic_domain_count": len(SYMBOLIC_AXES), "records": domains})
    write_parquet(domains, out_dir / "symbolic_domain_registry.parquet")
    dump_json(out_dir / "reference_element_registry.json", {"reference_element_count": len(refs), "records": refs})
    write_parquet(refs, out_dir / "reference_element_registry.parquet")
    dump_json(out_dir / "alternate_placeholder_family_registry.json", {"placeholder_family_count": len(placeholders), "records": placeholders})
    write_parquet(placeholders, out_dir / "alternate_placeholder_family_registry.parquet")
    dump_json(out_dir / "axis_domain_admissibility_registry.json", {"admissibility_record_count": len(admissibility), "records": admissibility})
    write_parquet(admissibility, out_dir / "axis_domain_admissibility_registry.parquet")
    dump_json(out_dir / "assumption_only_domain_preservation_audit.json", {
        "assumption_only_domain_count": 2,
        "axis_ids": pipe(ASSUMPTION_ONLY_AXES),
        "numeric_bounds_defined": False,
        "unit_defined": False,
        "convention_members_defined": False,
        "assumption_required": True,
        "empirically_observed": False,
        "new_assumption_count": 0,
        "assumption_only_domain_mutated": False,
    })
    dump_json(out_dir / "scheduled_layover_exclusion_audit.json", {
        "axis_id": EXCLUDED_AXIS,
        "eligibility": "EXCLUDED_REQUIRES_NEW_OBSERVATION",
        "grid_inclusion_allowed": False,
        "symbolic_domain_specification_allowed": False,
        "scheduled_layover_reintroduced": False,
    })
    dump_json(out_dir / "stage_interaction_constraint_registry.json", {"stage_interaction_constraint_count": len(constraints), "records": constraints})
    write_parquet(constraints, out_dir / "stage_interaction_constraint_registry.parquet")
    dump_json(out_dir / "candidate_stage_order_contract.json", {
        "candidate_stage_order": "raw source timestamp candidate -> Q_CLOCK -> S_CLOCK -> source-local endpoint construction -> U_PROVIDER/U_REQUEST -> C_INTERVAL -> B_ENDPOINT -> M_BOUNDARY -> P_SOURCE -> candidate dual interval representation",
        "candidate_stage_order_adopted": False,
        "stage_order_status": "CANDIDATE_ORDER_REQUIRES_FORMAL_ADOPTION",
        "operator_execution_allowed": False,
    })
    dump_json(out_dir / "domain_dependency_graph.json", {
        "node_count": len(domains),
        "edge_count": len(edges),
        "nodes": pipe([row["domain_id"] for row in domains]),
        "cycle_status": "BIDIRECTIONAL_Q_CLOCK_S_CLOCK_ORDERING_ACCOUNTED_AS_UNRESOLVED",
        "execution_allowed": False,
    })
    dump_json(out_dir / "domain_dependency_edge_registry.json", {"domain_dependency_edge_count": len(edges), "records": edges})
    write_parquet(edges, out_dir / "domain_dependency_edge_registry.parquet")
    ambiguities = [row for row in domains if row["readiness_status"] == "SYMBOLIC_DOMAIN_KIND_AMBIGUOUS"]
    dump_json(out_dir / "domain_kind_ambiguity_registry.json", {"domain_kind_ambiguous_count": len(ambiguities), "records": ambiguities})
    unresolved_stage = [row for row in constraints if row["candidate_preferred_order"] == "UNRESOLVED"]
    dump_json(out_dir / "unresolved_stage_order_registry.json", {"unresolved_stage_order_count": len(unresolved_stage), "records": unresolved_stage})
    dump_json(out_dir / "domain_readiness_registry.json", {"domain_readiness_record_count": len(readiness), "records": readiness})
    write_parquet(readiness, out_dir / "domain_readiness_registry.parquet")
    dump_json(out_dir / "original_design_matrix_domain_overlay.json", {"design_cell_count": len(overlay), "records": overlay})
    write_parquet(overlay, out_dir / "original_design_matrix_domain_overlay.parquet")

    structurally = sum(row["readiness_status"] == "SYMBOLIC_DOMAIN_STRUCTURALLY_SPECIFIED" for row in domains)
    ambiguous_count = len(ambiguities)
    stage_blocked = 0
    assumption_only_count = sum(row["readiness_status"] == "ASSUMPTION_ONLY_DOMAIN_PRESERVED" for row in domains)
    excluded_count = sum(row["readiness_status"] == "EXCLUDED_REQUIRES_NEW_OBSERVATION" for row in domains)
    underdefined = sum(row["readiness_status"] == "DOMAIN_UNDERDEFINED" for row in domains)
    hard_stage_dependency_count = sum(row["hard_or_soft"] == "hard" for row in edges)
    soft_stage_dependency_count = sum(row["hard_or_soft"] == "soft" for row in edges)
    domain_complete_cells = sum(row["domain_kind_complete"] for row in overlay)
    domain_ambiguous_cells = sum(row["unresolved_domain_kind_count"] > 0 for row in overlay)
    stage_blocked_cells = sum(row["unresolved_stage_order_count"] > 0 for row in overlay)

    dump_json(out_dir / "domain_cardinality_audit.json", {
        "symbolic_target_count": len(SYMBOLIC_AXES),
        "symbolic_domain_count": len(SYMBOLIC_AXES),
        "reference_element_count": len(refs),
        "placeholder_family_count": len(placeholders),
        "structurally_specified_domain_count": structurally,
        "domain_kind_ambiguous_count": ambiguous_count,
        "stage_order_blocked_domain_count": stage_blocked,
        "assumption_only_domain_count": assumption_only_count,
        "excluded_domain_count": excluded_count,
        "underdefined_domain_count": underdefined,
        "stage_interaction_constraint_count": len(constraints),
        "hard_stage_dependency_count": hard_stage_dependency_count,
        "soft_stage_dependency_count": soft_stage_dependency_count,
        "unresolved_stage_order_count": len(unresolved_stage),
        "domain_dependency_edge_count": len(edges),
        "original_design_cell_count": design["design_cell_count"],
        "overlay_design_cell_count": len(overlay),
        "reference_cell_count": sum(row["reference_cell"] for row in overlay),
        "domain_complete_cell_count": domain_complete_cells,
        "domain_ambiguous_cell_count": domain_ambiguous_cells,
        "stage_blocked_cell_count": stage_blocked_cells,
        "actual_level_resolution_ready_cell_count": sum(row["actual_level_resolution_ready"] for row in overlay),
    })
    dump_json(out_dir / "execution_boundary.json", {
        "symbolic_domain_contract_adopted": False,
        "stage_interaction_contract_adopted": False,
        "actual_level_resolution_approved": False,
        "sensitivity_grid_execution_approved": False,
        "identified_set_reestimation_approved": False,
        "turnbull_reestimation_approved": False,
        "phase2_authorized": False,
        "operator_execution_count": 0,
    })
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
        "new_assumption_created": False,
        "numeric_domain_bound_created": False,
        "actual_numeric_level_created": False,
        "actual_alternate_enum_created": False,
        "actual_topology_member_created": False,
        "actual_reconciliation_rule_created": False,
        "clock_resolution_estimated": False,
        "clock_skew_estimated": False,
        "endpoint_uncertainty_estimated": False,
        "operator_executed": False,
        "timestamp_transformed": False,
        "interval_reconstructed": False,
        "identified_set_calculated": False,
        "turnbull_executed": False,
        "point_estimate_calculated": False,
        "robustness_judged": False,
        "design_matrix_changed": False,
        "upstream_artifact_modified": False,
        "simulator_applied": False,
        "phase2_executed": False,
        "baseline_rerun_executed": False,
        "training_or_retraining_executed": False,
        "new_assumption_count": 0,
        "new_numeric_domain_bound_count": 0,
        "new_actual_level_count": 0,
        "new_actual_enum_member_count": 0,
        "operator_execution_count": 0,
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
    mutation = compare_snapshot(before, file_snapshot(upstream))
    dump_json(out_dir / "upstream_mutation_audit.json", mutation)
    dump_json(out_dir / "lineage_field_audit.json", {
        "authoritative_upstream_only": True,
        "authoritative_upstream_artifact": str(upstream),
        "actual_level_resolution_performed": False,
        "operator_execution_performed": False,
        "source_code_used_as_evidence": False,
    })
    dump_json(out_dir / "forbidden_field_audit.json", forbidden_field_audit(out_dir))

    parquet_check = parquet_json_check(out_dir)
    secret_audit = secret_scan(out_dir)
    forbidden = strict_read_json(out_dir / "forbidden_field_audit.json")

    gate_status = PASS_AMBIGUITY
    next_action = "Axis-specific symbolic domain-kind adjudication only"
    if not upstream_validation["checks"]["gate_matches"]:
        gate_status = "FAIL_UPSTREAM_GATE_MISMATCH"
    elif upstream_validation["upstream_integrity_status"] != "PASS":
        gate_status = "FAIL_UPSTREAM_INTEGRITY"
    elif len(SYMBOLIC_AXES) != 6:
        gate_status = "FAIL_SYMBOLIC_TARGET_COUNT_MISMATCH"
    elif any(sum(ref["axis_id"] == axis for ref in refs) != 1 for axis in SYMBOLIC_AXES):
        gate_status = "FAIL_REFERENCE_ELEMENT_MISSING"
    elif any(row["domain_kind"] not in {
        "SYMBOLIC_ORDERED_SUPPORT_DOMAIN",
        "SYMBOLIC_UNORDERED_CONVENTION_DOMAIN",
        "SYMBOLIC_TOPOLOGY_DOMAIN",
        "SYMBOLIC_RECONCILIATION_RULE_DOMAIN",
        "ASSUMPTION_ONLY_NUMERIC_DOMAIN_UNSPECIFIED",
        "ASSUMPTION_ONLY_CONVENTION_DOMAIN_UNSPECIFIED",
        "EXCLUDED_REQUIRES_OBSERVATION",
    } for row in domains):
        gate_status = "FAIL_UNSUPPORTED_DOMAIN_KIND"
    elif prohibited["new_actual_enum_member_count"] or prohibited["new_actual_level_count"]:
        gate_status = "FAIL_ACTUAL_ALTERNATE_MEMBER_INVENTED"
    elif prohibited["new_numeric_domain_bound_count"]:
        gate_status = "FAIL_NUMERIC_DOMAIN_BOUND_INVENTED"
    elif prohibited["new_assumption_count"]:
        gate_status = "FAIL_NEW_ASSUMPTION_CREATED"
    elif strict_read_json(out_dir / "assumption_only_domain_preservation_audit.json")["assumption_only_domain_mutated"]:
        gate_status = "FAIL_ASSUMPTION_ONLY_DOMAIN_MUTATED"
    elif strict_read_json(out_dir / "scheduled_layover_exclusion_audit.json")["scheduled_layover_reintroduced"]:
        gate_status = "FAIL_SCHEDULED_LAYOVER_REINTRODUCED"
    elif len(constraints) != 4:
        gate_status = "FAIL_STAGE_INTERACTION_PAIR_MISMATCH"
    elif design["design_cell_count"] != len(overlay):
        gate_status = "FAIL_DESIGN_MATRIX_CARDINALITY_CHANGED"
    elif prohibited["operator_execution_count"]:
        gate_status = "FAIL_OPERATOR_EXECUTION_DETECTED"
    elif any(row["candidate_execution_ready"] or row["execution_approved"] or row["reestimation_approved"] for row in overlay):
        gate_status = "FAIL_EXECUTION_SCOPE_OVERREACH"
    elif not lock_state["all_required_locks_false"]:
        gate_status = "FAIL_LOCK_STATE_VIOLATION"
    elif secret_audit["secret_leak_count"]:
        gate_status = "FAIL_SECRET_LEAK"
    elif parquet_check["parquet_read_failure_count"] or parquet_check["json_parquet_mismatch_count"]:
        gate_status = "FAIL_JSON_PARQUET_MISMATCH"
    elif forbidden["forbidden_operation_count"]:
        gate_status = "FAIL_EXECUTION_SCOPE_OVERREACH"

    report = {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "upstream_artifact": str(upstream),
        "upstream_gate": gate.get("gate"),
        "upstream_integrity_status": upstream_validation["upstream_integrity_status"],
        "symbolic_target_count": len(SYMBOLIC_AXES),
        "symbolic_domain_count": len(SYMBOLIC_AXES),
        "reference_element_count": len(refs),
        "placeholder_family_count": len(placeholders),
        "structurally_specified_domain_count": structurally,
        "domain_kind_ambiguous_count": ambiguous_count,
        "stage_order_blocked_domain_count": stage_blocked,
        "assumption_only_domain_count": assumption_only_count,
        "excluded_domain_count": excluded_count,
        "underdefined_domain_count": underdefined,
        "stage_interaction_constraint_count": len(constraints),
        "hard_stage_dependency_count": hard_stage_dependency_count,
        "soft_stage_dependency_count": soft_stage_dependency_count,
        "unresolved_stage_order_count": len(unresolved_stage),
        "domain_dependency_edge_count": len(edges),
        "original_design_cell_count": design["design_cell_count"],
        "overlay_design_cell_count": len(overlay),
        "reference_cell_count": sum(row["reference_cell"] for row in overlay),
        "domain_complete_cell_count": domain_complete_cells,
        "domain_ambiguous_cell_count": domain_ambiguous_cells,
        "stage_blocked_cell_count": stage_blocked_cells,
        "actual_level_resolution_ready_cell_count": 0,
        "new_numeric_domain_bound_count": 0,
        "new_actual_level_count": 0,
        "new_actual_enum_member_count": 0,
        "new_assumption_count": 0,
        "operator_execution_count": 0,
        "symbolic_domain_contract_adopted": False,
        "stage_interaction_contract_adopted": False,
        "actual_level_resolution_approved": False,
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
        "gate": gate_status,
        "next_authorized_action": next_action,
    }
    dump_json(out_dir / "final_report.json", report)
    (out_dir / "final_report.md").write_text(
        "\n".join([
            "# R2D-1X Refined Symbolic Level-Domain Specification",
            "",
            f"- Artifact: `{out_dir}`",
            f"- Upstream gate: `{gate.get('gate')}`",
            f"- Gate: `{gate_status}`",
            "",
            "## Domain Summary",
            f"- Symbolic domains: `{len(SYMBOLIC_AXES)}`",
            f"- Reference elements: `{len(refs)}`",
            f"- Placeholder families: `{len(placeholders)}`",
            f"- Structurally specified / ambiguous: `{structurally}/{ambiguous_count}`",
            f"- Stage constraints: `{len(constraints)}`",
            f"- Domain dependency edges: `{len(edges)}`",
            f"- Overlay cells: `{len(overlay)}`",
            "",
            "## Locks",
            "- Symbolic domain contract adopted: `false`",
            "- Stage interaction contract adopted: `false`",
            "- Actual level resolution approved: `false`",
            "- Sensitivity grid execution approved: `false`",
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
        "upstream_integrity_status": upstream_validation["upstream_integrity_status"],
        "symbolic_target_count": len(SYMBOLIC_AXES),
        "symbolic_domain_count": len(SYMBOLIC_AXES),
        "reference_element_count": len(refs),
        "placeholder_family_count": len(placeholders),
        "domain_kind_ambiguous_count": ambiguous_count,
        "stage_interaction_constraint_count": len(constraints),
        "unresolved_stage_order_count": len(unresolved_stage),
        "original_design_cell_count": design["design_cell_count"],
        "overlay_design_cell_count": len(overlay),
        "reference_cell_count": sum(row["reference_cell"] for row in overlay),
        "new_numeric_domain_bound_count": 0,
        "new_actual_level_count": 0,
        "new_actual_enum_member_count": 0,
        "new_assumption_count": 0,
        "operator_execution_count": 0,
        "symbolic_domain_contract_adopted": False,
        "stage_interaction_contract_adopted": False,
        "actual_level_resolution_approved": False,
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
        raise RuntimeError(f"secret scan changed after final report/gate: {final_secret}")
    if final_forbidden["forbidden_operation_count"] != forbidden["forbidden_operation_count"]:
        raise RuntimeError(f"forbidden audit changed after final report/gate: {final_forbidden}")

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
    print(f"symbolic domains/reference/placeholder:\n{len(SYMBOLIC_AXES)}/{len(refs)}/{len(placeholders)}")
    print(f"structurally specified/domain ambiguous/assumption/excluded:\n{structurally}/{ambiguous_count}/{assumption_only_count}/{excluded_count}")
    print(f"stage constraints/hard deps/soft deps/unresolved order:\n{len(constraints)}/{hard_stage_dependency_count}/{soft_stage_dependency_count}/{len(unresolved_stage)}")
    print(f"design cells original/overlay/reference:\n{design['design_cell_count']}/{len(overlay)}/{sum(row['reference_cell'] for row in overlay)}")
    print("new numeric bounds/actual levels/actual enum/new assumptions/operator execution:\n0/0/0/0/0")
    print("API/service key/DB/network:\n0/false/false/false")
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
