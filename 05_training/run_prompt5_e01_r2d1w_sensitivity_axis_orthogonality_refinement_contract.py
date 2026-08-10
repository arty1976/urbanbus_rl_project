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
    "prompt5_e01_r2d1v_frozen_artifact_symbolic_level_resolution_audit_20260731_060755"
)
EXPECTED_UPSTREAM_GATE = "PASS_FROZEN_ARTIFACT_SYMBOLIC_LEVEL_ORTHOGONALITY_UNRESOLVED_STILL_LOCKED"
EXPECTED_NEXT = "Sensitivity-axis orthogonality refinement contract only"
SUCCESS_GATE = "PASS_SENSITIVITY_AXIS_ORTHOGONALITY_REFINEMENT_CONTRACT_READY_INTERACTIONS_REMAIN_STILL_LOCKED"

SYMBOLIC_TARGETS = [
    "provider_endpoint_uncertainty__symbolic_alternate",
    "request_endpoint_uncertainty__symbolic_alternate",
    "clock_rounding_allowance__symbolic_alternate",
    "interval_censoring_convention__symbolic_alternate",
    "endpoint_inclusion_exclusion_convention__symbolic_alternate",
    "provider_request_precedence_rule__symbolic_alternate",
]

UPSTREAM_LOCK_FIELDS = [
    "path_b_adopted",
    "sensitivity_contract_adopted",
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
    "axis_disposition_adopted",
    "composite_axis_adopted",
    "resolved_level_contract_adopted",
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
    "r2d1v_gate_snapshot.json",
    "r2d1v_resolution_snapshot.json",
    "r2d1v_orthogonality_snapshot.json",
    "r2d1v_design_matrix_snapshot.json",
    "r2d1v_lock_snapshot.json",
    "orthogonality_definition_contract.json",
    "primitive_operator_registry.json",
    "primitive_operator_registry.parquet",
    "original_axis_to_primitive_mapping.json",
    "original_axis_to_primitive_mapping.parquet",
    "candidate_processing_stage_contract.json",
    "operator_type_signature_registry.json",
    "operator_type_signature_registry.parquet",
    "axis_parameter_ownership_registry.json",
    "axis_parameter_ownership_registry.parquet",
    "pairwise_orthogonality_refinement.json",
    "pairwise_orthogonality_refinement.parquet",
    "operator_interaction_graph.json",
    "operator_commutativity_matrix.json",
    "operator_commutativity_matrix.parquet",
    "stage_dependency_graph.json",
    "axis_disposition_registry.json",
    "axis_disposition_registry.parquet",
    "composite_axis_candidate_registry.json",
    "assumption_only_preservation_audit.json",
    "scheduled_layover_exclusion_audit.json",
    "original_design_matrix_confounding_audit.json",
    "original_design_matrix_confounding_audit.parquet",
    "candidate_refactoring_impact_preview.json",
    "orthogonality_cardinality_audit.json",
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


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pipe(values: list[str]) -> str:
    return "|".join(values)


def file_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        out[path.relative_to(root).as_posix()] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return out


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


def validate_artifact_manifest(artifact: Path) -> dict[str, Any]:
    manifest_path = artifact / "artifact_manifest.json"
    if not manifest_path.is_file():
        return {"status": "FAIL", "reason": "artifact_manifest.json missing"}
    manifest = strict_read_json(manifest_path)
    hash_mismatch = []
    size_mismatch = []
    strict_json_fail = []
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
                strict_json_fail.append({"relative_path": rel, "error": type(exc).__name__})
    return {
        "status": "PASS" if not (hash_mismatch or size_mismatch or strict_json_fail or missing) else "FAIL",
        "manifest_hash_mismatch_count": len(hash_mismatch),
        "manifest_size_mismatch_count": len(size_mismatch),
        "strict_json_failure_count": len(strict_json_fail),
        "manifest_missing_count": len(missing),
        "hash_mismatches": hash_mismatch,
        "size_mismatches": size_mismatch,
        "strict_json_failures": strict_json_fail,
        "missing": missing,
    }


def validate_upstream(upstream: Path) -> dict[str, Any]:
    gate = strict_read_json(upstream / "gate_decision.json")
    final_report = strict_read_json(upstream / "final_report.json")
    locks = strict_read_json(upstream / "lock_state.json")
    manifest = validate_artifact_manifest(upstream)
    expected = {
        "symbolic_target_count": 6,
        "resolved_discrete_count": 0,
        "resolved_numeric_count": 0,
        "resolved_enum_set_count": 0,
        "conflicting_count": 0,
        "nonorthogonal_count": 6,
        "evidence_hit_count": 41,
        "direct_explicit_evidence_count": 0,
        "direct_derived_evidence_count": 0,
        "assumption_only_preserved_count": 2,
        "scheduled_layover_excluded_count": 1,
        "original_design_cell_count": 13,
        "candidate_design_cell_count": 13,
        "reference_cell_count": 1,
    }
    combined = dict(final_report)
    combined.update(gate)
    mismatches = [k for k, v in expected.items() if combined.get(k) != v]
    lock_violations = [k for k in UPSTREAM_LOCK_FIELDS if locks.get(k) is not False]
    checks = {
        "gate_matches": gate.get("gate") == EXPECTED_UPSTREAM_GATE,
        "next_action_matches": gate.get("next_authorized_action") == EXPECTED_NEXT,
        "required_counts_match": not mismatches,
        "required_locks_false": not lock_violations,
        "manifest_integrity_pass": manifest["status"] == "PASS",
    }
    return {
        "upstream_artifact": str(upstream),
        "upstream_gate": gate.get("gate"),
        "upstream_next_authorized_action": gate.get("next_authorized_action"),
        "checks": checks,
        "count_mismatches": mismatches,
        "lock_violations": lock_violations,
        "manifest_validation": manifest,
        "upstream_integrity_status": "PASS" if all(checks.values()) else "FAIL",
    }


def primitive_operators() -> list[dict[str, Any]]:
    return [
        {
            "operator_id": "U_PROVIDER",
            "operator_name": "Provider endpoint support operator",
            "mapped_axis_id": "provider_endpoint_uncertainty",
            "operator_stage": "source-local endpoint support treatment",
            "input_object_type": "provider_source_endpoint_candidate",
            "output_object_type": "provider_source_local_interval_contract",
            "parameter_namespace": "provider_endpoint_support",
            "parameter_domain_type": "SYMBOLIC_NUMERIC_DOMAIN_UNSPECIFIED",
            "changed_fields": pipe(["provider_endpoint_support", "provider_lower_endpoint_candidate", "provider_upper_endpoint_candidate"]),
            "frozen_fields": pipe(["request_endpoint", "clock_rounding_rule", "clock_skew_rule", "source_reconciliation_rule", "boundary_topology", "censoring_class"]),
            "required_preconditions": "provider source endpoint candidates already exist in frozen input contract",
            "postconditions": "provider source-local support is typed before source reconciliation",
            "assumption_required": False,
            "new_observation_required": False,
            "symbolic_level_resolution_supported": False,
            "execution_allowed": False,
            "adopted": False,
        },
        {
            "operator_id": "U_REQUEST",
            "operator_name": "Request endpoint support operator",
            "mapped_axis_id": "request_endpoint_uncertainty",
            "operator_stage": "source-local endpoint support treatment",
            "input_object_type": "request_source_endpoint_candidate",
            "output_object_type": "request_source_local_interval_contract",
            "parameter_namespace": "request_endpoint_support",
            "parameter_domain_type": "SYMBOLIC_NUMERIC_DOMAIN_UNSPECIFIED",
            "changed_fields": pipe(["request_endpoint_support", "request_lower_endpoint_candidate", "request_upper_endpoint_candidate"]),
            "frozen_fields": pipe(["provider_endpoint", "clock_rounding_rule", "clock_skew_rule", "source_reconciliation_rule", "boundary_topology", "censoring_class"]),
            "required_preconditions": "request source endpoint candidates already exist in frozen input contract",
            "postconditions": "request source-local support is typed before source reconciliation",
            "assumption_required": False,
            "new_observation_required": False,
            "symbolic_level_resolution_supported": False,
            "execution_allowed": False,
            "adopted": False,
        },
        {
            "operator_id": "Q_CLOCK",
            "operator_name": "Timestamp quantization operator",
            "mapped_axis_id": "clock_rounding_allowance",
            "operator_stage": "timestamp quantization",
            "input_object_type": "raw_timestamp_or_timestamp_candidate",
            "output_object_type": "quantized_timestamp_representation_contract",
            "parameter_namespace": "clock_quantization_rule",
            "parameter_domain_type": "SYMBOLIC_DISCRETE_DOMAIN_UNSPECIFIED",
            "changed_fields": pipe(["quantized_timestamp_representation", "timestamp_bin_assignment"]),
            "frozen_fields": pipe(["clock_offset", "source_uncertainty_support", "source_reconciliation_rule", "missing_boundary_rule"]),
            "required_preconditions": "raw timestamp candidate exists; no precision value is invented",
            "postconditions": "timestamp representation is typed before endpoint support treatment",
            "assumption_required": False,
            "new_observation_required": False,
            "symbolic_level_resolution_supported": False,
            "execution_allowed": False,
            "adopted": False,
        },
        {
            "operator_id": "S_CLOCK",
            "operator_name": "Clock offset operator",
            "mapped_axis_id": "clock_skew_allowance",
            "operator_stage": "clock offset treatment",
            "input_object_type": "timestamp_coordinate_contract",
            "output_object_type": "offset_timestamp_coordinate_contract",
            "parameter_namespace": "clock_offset_rule",
            "parameter_domain_type": "ASSUMPTION_ONLY_DOMAIN",
            "changed_fields": pipe(["timestamp_coordinate_offset", "source_specific_clock_offset"]),
            "frozen_fields": pipe(["clock_quantization_rule", "source_uncertainty_support", "source_reconciliation_rule"]),
            "required_preconditions": "assumption-only level remains unobserved and unexecuted",
            "postconditions": "offset treatment remains candidate-only and locked",
            "assumption_required": True,
            "new_observation_required": False,
            "symbolic_level_resolution_supported": False,
            "execution_allowed": False,
            "adopted": False,
        },
        {
            "operator_id": "C_INTERVAL",
            "operator_name": "Censoring-state operator",
            "mapped_axis_id": "interval_censoring_convention",
            "operator_stage": "censoring-state classification",
            "input_object_type": "source_local_endpoint_observability_contract",
            "output_object_type": "observation_state_contract",
            "parameter_namespace": "censoring_state_classification",
            "parameter_domain_type": "SYMBOLIC_DISCRETE_DOMAIN_UNSPECIFIED",
            "changed_fields": pipe(["observation_state_type", "boundary_observability_state"]),
            "frozen_fields": pipe(["endpoint_numeric_coordinate", "endpoint_inclusion_topology", "source_reconciliation", "clock_transformation"]),
            "required_preconditions": "endpoint coordinates are fixed for this operator",
            "postconditions": "observation-state classification is typed separately from boundary membership",
            "assumption_required": False,
            "new_observation_required": False,
            "symbolic_level_resolution_supported": False,
            "execution_allowed": False,
            "adopted": False,
        },
        {
            "operator_id": "B_ENDPOINT",
            "operator_name": "Boundary-topology operator",
            "mapped_axis_id": "endpoint_inclusion_exclusion_convention",
            "operator_stage": "boundary topology",
            "input_object_type": "endpoint_coordinate_and_observation_state_contract",
            "output_object_type": "interval_topology_contract",
            "parameter_namespace": "endpoint_boundary_membership",
            "parameter_domain_type": "SYMBOLIC_DISCRETE_DOMAIN_UNSPECIFIED",
            "changed_fields": pipe(["lower_boundary_membership", "upper_boundary_membership", "interval_topology"]),
            "frozen_fields": pipe(["endpoint_numeric_coordinate", "censoring_state", "missingness_state", "source_precedence"]),
            "required_preconditions": "endpoint coordinates and censoring state are held fixed when topology changes",
            "postconditions": "boundary membership is typed separately from censoring-state classification",
            "assumption_required": False,
            "new_observation_required": False,
            "symbolic_level_resolution_supported": False,
            "execution_allowed": False,
            "adopted": False,
        },
        {
            "operator_id": "P_SOURCE",
            "operator_name": "Source-reconciliation operator",
            "mapped_axis_id": "provider_request_precedence_rule",
            "operator_stage": "provider/request reconciliation",
            "input_object_type": "provider_and_request_source_local_interval_contracts",
            "output_object_type": "candidate_dual_interval_representation_contract",
            "parameter_namespace": "source_reconciliation_rule",
            "parameter_domain_type": "SYMBOLIC_DISCRETE_DOMAIN_UNSPECIFIED",
            "changed_fields": pipe(["source_reconciliation_rule", "dual_interval_construction_rule"]),
            "frozen_fields": pipe(["provider_source_local_interval", "request_source_local_interval", "clock_transformation", "boundary_topology", "censoring_state"]),
            "required_preconditions": "source-local intervals are fixed before reconciliation changes",
            "postconditions": "dual interval construction rule is typed without creating a new enum",
            "assumption_required": False,
            "new_observation_required": False,
            "symbolic_level_resolution_supported": False,
            "execution_allowed": False,
            "adopted": False,
        },
        {
            "operator_id": "M_BOUNDARY",
            "operator_name": "Missing-boundary operator",
            "mapped_axis_id": "missing_boundary_treatment",
            "operator_stage": "missing-boundary treatment",
            "input_object_type": "boundary_observability_contract",
            "output_object_type": "missing_boundary_treatment_contract",
            "parameter_namespace": "missing_boundary_rule",
            "parameter_domain_type": "ASSUMPTION_ONLY_DOMAIN",
            "changed_fields": pipe(["missing_boundary_rule", "boundary_absence_treatment"]),
            "frozen_fields": pipe(["observed_boundary_coordinates", "clock_transformation", "source_precedence"]),
            "required_preconditions": "missing boundary treatment remains assumption-only",
            "postconditions": "missingness handling remains locked and unexecuted",
            "assumption_required": True,
            "new_observation_required": False,
            "symbolic_level_resolution_supported": False,
            "execution_allowed": False,
            "adopted": False,
        },
    ]


def stage_contract() -> list[dict[str, Any]]:
    stages = [
        ("STAGE_00", "raw source timestamp candidate", "", "Q_CLOCK"),
        ("STAGE_01", "Q_CLOCK timestamp quantization", "STAGE_00", "S_CLOCK"),
        ("STAGE_02", "S_CLOCK offset treatment", "STAGE_01", "source-local endpoint construction"),
        ("STAGE_03", "source-local endpoint construction", "STAGE_02", "U_PROVIDER|U_REQUEST"),
        ("STAGE_04", "U_PROVIDER / U_REQUEST endpoint support treatment", "STAGE_03", "C_INTERVAL"),
        ("STAGE_05", "C_INTERVAL censoring-state classification", "STAGE_04", "B_ENDPOINT"),
        ("STAGE_06", "B_ENDPOINT boundary topology", "STAGE_05", "M_BOUNDARY"),
        ("STAGE_07", "M_BOUNDARY missing-boundary treatment", "STAGE_06", "P_SOURCE"),
        ("STAGE_08", "P_SOURCE provider/request reconciliation", "STAGE_07", "candidate dual interval representation"),
        ("STAGE_09", "candidate dual interval representation", "STAGE_08", ""),
    ]
    return [
        {
            "stage_id": stage_id,
            "stage_order": idx,
            "stage_name": name,
            "depends_on_stage_id": depends,
            "feeds": feeds,
            "candidate_stage_order_adopted": False,
            "operator_execution_allowed": False,
            "stage_status": "CANDIDATE_ORDER_REQUIRES_FORMAL_ADOPTION",
        }
        for idx, (stage_id, name, depends, feeds) in enumerate(stages)
    ]


def axis_mapping(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_axis = {op["mapped_axis_id"]: op for op in ops}
    axes = [
        "provider_endpoint_uncertainty",
        "request_endpoint_uncertainty",
        "clock_rounding_allowance",
        "interval_censoring_convention",
        "endpoint_inclusion_exclusion_convention",
        "provider_request_precedence_rule",
        "clock_skew_allowance",
        "missing_boundary_treatment",
        "scheduled_layover_treatment",
    ]
    out = []
    for axis in axes:
        op = by_axis.get(axis)
        out.append({
            "axis_id": axis,
            "primitive_operator_id": op["operator_id"] if op else "",
            "mapped": op is not None,
            "mapping_status": "MAPPED_TO_PRIMITIVE_OPERATOR" if op else "UNMAPPED_EXCLUDED_REQUIRES_NEW_OBSERVATION",
            "candidate_only": True,
            "adopted": False,
            "execution_allowed": False,
        })
    return out


def ownership_registry(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for op in ops:
        axis = op["mapped_axis_id"]
        interaction_axes = {
            "provider_endpoint_uncertainty": "provider_request_precedence_rule",
            "request_endpoint_uncertainty": "clock_rounding_allowance",
            "clock_rounding_allowance": "request_endpoint_uncertainty|clock_skew_allowance",
            "interval_censoring_convention": "endpoint_inclusion_exclusion_convention",
            "endpoint_inclusion_exclusion_convention": "interval_censoring_convention",
            "provider_request_precedence_rule": "provider_endpoint_uncertainty|missing_boundary_treatment",
            "clock_skew_allowance": "clock_rounding_allowance",
            "missing_boundary_treatment": "provider_request_precedence_rule",
        }.get(axis, "")
        rows.append({
            "axis_id": axis,
            "owned_parameter_fields": op["changed_fields"],
            "owned_transformation_stage": op["operator_stage"],
            "allowed_output_fields": op["output_object_type"],
            "must_not_modify_fields": op["frozen_fields"],
            "requires_fixed_axes": op["frozen_fields"],
            "interaction_axes": interaction_axes,
            "ownership_conflict_count": 0,
            "ownership_decision": "NO_OWNERSHIP_CONFLICT",
            "candidate_only": True,
            "adopted": False,
            "execution_allowed": False,
        })
    rows.append({
        "axis_id": "scheduled_layover_treatment",
        "owned_parameter_fields": "",
        "owned_transformation_stage": "excluded",
        "allowed_output_fields": "",
        "must_not_modify_fields": "all_existing_fields",
        "requires_fixed_axes": "all_axes",
        "interaction_axes": "",
        "ownership_conflict_count": 0,
        "ownership_decision": "NO_OWNERSHIP_CONFLICT",
        "candidate_only": True,
        "adopted": False,
        "execution_allowed": False,
    })
    return rows


def pairwise_refinement() -> list[dict[str, Any]]:
    rows = [
        (
            "PAIR_00",
            "interval_censoring_convention",
            "endpoint_inclusion_exclusion_convention",
            "C_INTERVAL",
            "B_ENDPOINT",
            "C_INTERVAL classifies which boundary is observed; B_ENDPOINT defines membership of existing boundary coordinates.",
            "SEMANTICALLY_ORTHOGONAL_BUT_ORDER_DEPENDENT",
            "NONCOMMUTATIVE_BY_STAGE_DEPENDENCY",
        ),
        (
            "PAIR_01",
            "provider_endpoint_uncertainty",
            "provider_request_precedence_rule",
            "U_PROVIDER",
            "P_SOURCE",
            "U_PROVIDER changes provider source-local support before reconciliation; P_SOURCE changes the fixed-source reconciliation rule.",
            "SEMANTICALLY_ORTHOGONAL_BUT_ORDER_DEPENDENT",
            "NONCOMMUTATIVE_BY_STAGE_DEPENDENCY",
        ),
        (
            "PAIR_02",
            "request_endpoint_uncertainty",
            "clock_rounding_allowance",
            "U_REQUEST",
            "Q_CLOCK",
            "Q_CLOCK changes timestamp representation; U_REQUEST changes request endpoint support after timestamp treatment.",
            "SEMANTICALLY_ORTHOGONAL_BUT_ORDER_DEPENDENT",
            "NONCOMMUTATIVE_BY_STAGE_DEPENDENCY",
        ),
        (
            "PAIR_03",
            "clock_rounding_allowance",
            "clock_skew_allowance",
            "Q_CLOCK",
            "S_CLOCK",
            "Q_CLOCK quantizes coordinates; S_CLOCK applies an offset. They are semantically distinct but order dependent.",
            "SEMANTICALLY_ORTHOGONAL_BUT_ORDER_DEPENDENT",
            "NONCOMMUTATIVE_BY_STAGE_DEPENDENCY",
        ),
    ]
    return [
        {
            "pair_id": pair_id,
            "axis_a": axis_a,
            "axis_b": axis_b,
            "operator_a": op_a,
            "operator_b": op_b,
            "refined_semantic_basis": basis,
            "semantic_orthogonality_status": status,
            "semantically_orthogonal": True,
            "operators_commute": False,
            "commutativity_status": commute,
            "operational_interaction": True,
            "order_dependent": True,
            "empirical_level_resolvability_changed": False,
            "alternate_level_resolved": False,
            "new_observation_required": False,
            "requires_composite_axis": False,
            "underdefined": False,
            "candidate_only": True,
            "adopted": False,
            "execution_allowed": False,
        }
        for pair_id, axis_a, axis_b, op_a, op_b, basis, status, commute in rows
    ]


def axis_dispositions(mapping: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dispositions = {
        "provider_endpoint_uncertainty": ("KEEP_SEPARATE_ORDER_DEPENDENT", "Provider source-local support is distinct from reconciliation but feeds downstream bounds."),
        "request_endpoint_uncertainty": ("KEEP_SEPARATE_ORDER_DEPENDENT", "Request source-local support is distinct from clock quantization but depends on timestamp treatment."),
        "clock_rounding_allowance": ("KEEP_SEPARATE_ORDER_DEPENDENT", "Timestamp quantization is distinct from endpoint support and clock offset but order-dependent."),
        "interval_censoring_convention": ("KEEP_SEPARATE_ORDER_DEPENDENT", "Observation-state classification is distinct from boundary topology but precedes it."),
        "endpoint_inclusion_exclusion_convention": ("KEEP_SEPARATE_ORDER_DEPENDENT", "Boundary membership is distinct from censoring state but depends on fixed state typing."),
        "provider_request_precedence_rule": ("KEEP_SEPARATE_ORDER_DEPENDENT", "Source reconciliation is distinct from source-local support but downstream of it."),
        "clock_skew_allowance": ("RECLASSIFY_AS_ASSUMPTION_ONLY", "Clock offset remains assumption-only and unobserved."),
        "missing_boundary_treatment": ("RECLASSIFY_AS_ASSUMPTION_ONLY", "Missing-boundary treatment remains assumption-only and unobserved."),
        "scheduled_layover_treatment": ("EXCLUDE_REQUIRES_NEW_OBSERVATION", "Scheduled layover remains excluded without new observation."),
    }
    out = []
    for row in mapping:
        axis = row["axis_id"]
        disposition, reason = dispositions[axis]
        out.append({
            "axis_id": axis,
            "primitive_operator_id": row["primitive_operator_id"],
            "disposition": disposition,
            "disposition_reason": reason,
            "semantic_orthogonality_established": disposition.startswith("KEEP_SEPARATE"),
            "empirical_level_resolved": False,
            "alternate_level_available": False,
            "candidate_only": True,
            "adopted": False,
            "execution_allowed": False,
        })
    return out


def original_design_matrix_audit(candidate_matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_axis_for_level = {
        "provider_endpoint_uncertainty__symbolic_alternate": "provider_endpoint_uncertainty",
        "request_endpoint_uncertainty__symbolic_alternate": "request_endpoint_uncertainty",
        "clock_rounding_allowance__symbolic_alternate": "clock_rounding_allowance",
        "interval_censoring_convention__symbolic_alternate": "interval_censoring_convention",
        "endpoint_inclusion_exclusion_convention__symbolic_alternate": "endpoint_inclusion_exclusion_convention",
        "provider_request_precedence_rule__symbolic_alternate": "provider_request_precedence_rule",
    }
    rows = []
    for cell in candidate_matrix:
        bindings = json.loads(cell["axis_level_bindings"], parse_constant=strict_constant)
        symbolic_axes = sorted(target_axis_for_level[v] for v in bindings.values() if v in target_axis_for_level)
        contains_order_dependent = bool(symbolic_axes)
        row = {
            "design_cell_id": cell["design_cell_id"],
            "cell_label": cell["cell_label"],
            "axis_level_bindings": cell["axis_level_bindings"],
            "contains_semantically_nonorthogonal_bindings": False,
            "contains_order_dependent_bindings": contains_order_dependent,
            "contains_parameter_ownership_conflict": False,
            "contains_assumption_only_binding": bool(cell.get("contains_new_assumption")),
            "contains_excluded_binding": bool(cell.get("contains_excluded_axis")),
            "requires_composite_axis_refactor": False,
            "candidate_execution_ready": False,
            "execution_approved": False,
            "reestimation_approved": False,
            "reference_cell": bool(cell.get("reference_cell")),
            "order_dependent_axes": pipe(symbolic_axes),
        }
        rows.append(row)
    return rows


def parquet_json_check(out_dir: Path) -> dict[str, Any]:
    pairs = [
        ("primitive_operator_registry.json", "primitive_operator_registry.parquet", "records"),
        ("original_axis_to_primitive_mapping.json", "original_axis_to_primitive_mapping.parquet", "records"),
        ("operator_type_signature_registry.json", "operator_type_signature_registry.parquet", "records"),
        ("axis_parameter_ownership_registry.json", "axis_parameter_ownership_registry.parquet", "records"),
        ("pairwise_orthogonality_refinement.json", "pairwise_orthogonality_refinement.parquet", "records"),
        ("operator_commutativity_matrix.json", "operator_commutativity_matrix.parquet", "records"),
        ("axis_disposition_registry.json", "axis_disposition_registry.parquet", "records"),
        ("original_design_matrix_confounding_audit.json", "original_design_matrix_confounding_audit.parquet", "records"),
    ]
    failures = []
    mismatches = 0
    for json_name, parquet_name, key in pairs:
        try:
            records = strict_read_json(out_dir / json_name)[key]
            df = pd.read_parquet(out_dir / parquet_name)
        except Exception as exc:
            failures.append({"json": json_name, "parquet": parquet_name, "error": type(exc).__name__})
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
    allowed = {"forbidden_field_audit.json", "prohibited_operation_audit.json"}
    for path in sorted(out_dir.glob("*.json")):
        if path.name in allowed:
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
    if name.startswith("upstream") or name.startswith("r2d1v") or name == "lineage_field_audit.json":
        return "UPSTREAM_VALIDATION"
    if "operator" in name or "stage" in name or "ownership" in name:
        return "OPERATOR_CONTRACT"
    if "orthogonality" in name or "commutativity" in name or "interaction" in name:
        return "ORTHOGONALITY_CONTRACT"
    if "disposition" in name or "composite" in name or "refactoring" in name:
        return "AXIS_DISPOSITION"
    if "design_matrix" in name:
        return "DESIGN_MATRIX_AUDIT"
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
        if entry["sha256"] != sha256_file(path):
            hash_mismatch.append(entry["relative_path"])
        if entry["size_bytes"] != path.stat().st_size:
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

    out_dir = project_root / "05_training" / "artifacts" / f"prompt5_e01_r2d1w_sensitivity_axis_orthogonality_refinement_contract_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    upstream_before = file_snapshot(upstream)
    gate_snapshot = strict_read_json(upstream / "gate_decision.json")
    resolution_snapshot = strict_read_json(upstream / "resolution_cardinality_audit.json")
    orthogonality_snapshot = strict_read_json(upstream / "axis_orthogonality_audit.json")
    design_snapshot = strict_read_json(upstream / "candidate_resolved_design_matrix.json")
    lock_snapshot = strict_read_json(upstream / "lock_state.json")
    upstream_validation = validate_upstream(upstream)

    dump_json(out_dir / "upstream_validation.json", upstream_validation)
    dump_json(out_dir / "upstream_lineage.json", {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "authoritative_upstream_artifact": str(upstream),
        "authoritative_upstream_gate": gate_snapshot.get("gate"),
        "source_files_read": [
            "gate_decision.json",
            "resolution_cardinality_audit.json",
            "axis_orthogonality_audit.json",
            "candidate_resolved_design_matrix.json",
            "lock_state.json",
            "artifact_manifest.json",
        ],
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
    })
    dump_json(out_dir / "r2d1v_gate_snapshot.json", gate_snapshot)
    dump_json(out_dir / "r2d1v_resolution_snapshot.json", resolution_snapshot)
    dump_json(out_dir / "r2d1v_orthogonality_snapshot.json", orthogonality_snapshot)
    dump_json(out_dir / "r2d1v_design_matrix_snapshot.json", design_snapshot)
    dump_json(out_dir / "r2d1v_lock_snapshot.json", lock_snapshot)

    orthogonality_contract = {
        "semantic_orthogonality_conditions": [
            "different primitive operators",
            "different parameter namespaces",
            "axis A can change while axis B parameters remain fixed",
            "axis B can change while axis A parameters remain fixed",
            "changed-field contracts are not identical",
            "neither axis definition contains the other",
        ],
        "not_required_conditions": [
            "operators commute",
            "effects are additive",
            "no statistical interaction exists",
            "output fields never overlap",
        ],
        "semantic_orthogonality_distinguished_from_commutativity": True,
        "empirical_level_resolution_performed": False,
        "contract_adopted": False,
        "execution_allowed": False,
    }
    dump_json(out_dir / "orthogonality_definition_contract.json", orthogonality_contract)

    ops = primitive_operators()
    mapping = axis_mapping(ops)
    stages = stage_contract()
    ownership = ownership_registry(ops)
    pairs = pairwise_refinement()
    dispositions = axis_dispositions(mapping)
    design_audit = original_design_matrix_audit(design_snapshot["records"])

    dump_json(out_dir / "primitive_operator_registry.json", {"primitive_operator_count": len(ops), "records": ops})
    write_parquet(ops, out_dir / "primitive_operator_registry.parquet")
    dump_json(out_dir / "original_axis_to_primitive_mapping.json", {"axis_count": len(mapping), "records": mapping})
    write_parquet(mapping, out_dir / "original_axis_to_primitive_mapping.parquet")
    dump_json(out_dir / "candidate_processing_stage_contract.json", {
        "stage_count": len(stages),
        "candidate_stage_order_adopted": False,
        "operator_execution_allowed": False,
        "stage_order_status": "CANDIDATE_ORDER_REQUIRES_FORMAL_ADOPTION",
        "records": stages,
    })
    dump_json(out_dir / "operator_type_signature_registry.json", {"operator_type_signature_count": len(ops), "records": ops})
    write_parquet(ops, out_dir / "operator_type_signature_registry.parquet")
    dump_json(out_dir / "axis_parameter_ownership_registry.json", {"axis_count": len(ownership), "records": ownership})
    write_parquet(ownership, out_dir / "axis_parameter_ownership_registry.parquet")
    dump_json(out_dir / "pairwise_orthogonality_refinement.json", {"pairwise_refinement_count": len(pairs), "records": pairs})
    write_parquet(pairs, out_dir / "pairwise_orthogonality_refinement.parquet")
    dump_json(out_dir / "operator_interaction_graph.json", {
        "node_count": len(ops),
        "edge_count": len(pairs),
        "nodes": [op["operator_id"] for op in ops],
        "edges": [
            {
                "source": row["operator_a"],
                "target": row["operator_b"],
                "interaction_type": row["commutativity_status"],
                "execution_performed": False,
            }
            for row in pairs
        ],
    })
    commutativity = [
        {
            "axis_a": row["axis_a"],
            "axis_b": row["axis_b"],
            "operator_a": row["operator_a"],
            "operator_b": row["operator_b"],
            "semantically_orthogonal": row["semantically_orthogonal"],
            "operators_commute": row["operators_commute"],
            "commutativity_status": row["commutativity_status"],
            "orthogonality_commutativity_conflated": False,
            "execution_performed": False,
        }
        for row in pairs
    ]
    dump_json(out_dir / "operator_commutativity_matrix.json", {"pair_count": len(commutativity), "records": commutativity})
    write_parquet(commutativity, out_dir / "operator_commutativity_matrix.parquet")
    dump_json(out_dir / "stage_dependency_graph.json", {
        "stage_count": len(stages),
        "operator_execution_count": 0,
        "records": stages,
        "dependency_edges": [
            {"source": row["depends_on_stage_id"], "target": row["stage_id"]}
            for row in stages
            if row["depends_on_stage_id"]
        ],
    })
    dump_json(out_dir / "axis_disposition_registry.json", {"axis_count": len(dispositions), "records": dispositions})
    write_parquet(dispositions, out_dir / "axis_disposition_registry.parquet")
    dump_json(out_dir / "composite_axis_candidate_registry.json", {
        "composite_axis_candidate_count": 0,
        "records": [],
        "composite_axis_adopted": False,
        "execution_allowed": False,
    })

    dump_json(out_dir / "assumption_only_preservation_audit.json", {
        "assumption_only_level_count": 2,
        "preserved_axis_ids": ["clock_skew_allowance", "missing_boundary_treatment"],
        "assumption_only_level_mutated": False,
        "new_assumption_count": 0,
        "execution_allowed": False,
    })
    dump_json(out_dir / "scheduled_layover_exclusion_audit.json", {
        "axis_id": "scheduled_layover_treatment",
        "grid_inclusion_allowed": False,
        "eligibility": "EXCLUDED_REQUIRES_NEW_OBSERVATION",
        "scheduled_layover_reintroduced": False,
        "execution_allowed": False,
    })
    dump_json(out_dir / "original_design_matrix_confounding_audit.json", {
        "design_cell_count": len(design_audit),
        "records": design_audit,
    })
    write_parquet(design_audit, out_dir / "original_design_matrix_confounding_audit.parquet")
    dump_json(out_dir / "candidate_refactoring_impact_preview.json", {
        "original_design_cell_count": design_snapshot["design_cell_count"],
        "candidate_design_cell_count": len(design_audit),
        "cell_added_count": 0,
        "cell_deleted_count": 0,
        "cell_reordered_count": 0,
        "composite_axis_candidate_count": 0,
        "refactoring_adopted": False,
        "execution_allowed": False,
    })

    parameter_conflicts = sum(row["ownership_conflict_count"] for row in ownership)
    stage_conflicts = 0
    shared_output_distinct = sum(row["ownership_decision"] == "SHARED_OUTPUT_BUT_DISTINCT_PARAMETER_OWNERSHIP" for row in ownership)
    semantically_orthogonal_pair_count = sum(row["semantically_orthogonal"] for row in pairs)
    interacting_pair_count = sum(row["operational_interaction"] for row in pairs)
    order_dependent_pair_count = sum(row["order_dependent"] for row in pairs)
    nonorthogonal_pair_count = sum(not row["semantically_orthogonal"] for row in pairs)
    new_observation_required_pair_count = sum(row["new_observation_required"] for row in pairs)
    underdefined_pair_count = sum(row["underdefined"] for row in pairs)
    keep_separate_axis_count = sum(row["disposition"].startswith("KEEP_SEPARATE") for row in dispositions)
    interacting_axis_count = sum(row["disposition"] in {"KEEP_SEPARATE_INTERACTING", "KEEP_SEPARATE_ORDER_DEPENDENT"} for row in dispositions)
    order_dependent_axis_count = sum(row["disposition"] == "KEEP_SEPARATE_ORDER_DEPENDENT" for row in dispositions)
    assumption_only_axis_count = sum(row["disposition"] == "RECLASSIFY_AS_ASSUMPTION_ONLY" for row in dispositions)
    excluded_axis_count = sum(row["disposition"] == "EXCLUDE_REQUIRES_NEW_OBSERVATION" for row in dispositions)
    confounded_cell_count = sum(row["contains_semantically_nonorthogonal_bindings"] for row in design_audit)
    order_dependent_cell_count = sum(row["contains_order_dependent_bindings"] for row in design_audit)
    composite_refactor_cell_count = sum(row["requires_composite_axis_refactor"] for row in design_audit)
    empirical_level_resolved_count = 0
    new_numeric_level_count = 0
    new_alternate_enum_count = 0
    new_assumption_count = 0
    operator_execution_count = 0

    dump_json(out_dir / "orthogonality_cardinality_audit.json", {
        "symbolic_target_count": len(SYMBOLIC_TARGETS),
        "primitive_operator_count": len(ops),
        "mapped_axis_count": sum(row["mapped"] for row in mapping),
        "unmapped_axis_count": sum(not row["mapped"] for row in mapping),
        "pairwise_refinement_count": len(pairs),
        "semantically_orthogonal_pair_count": semantically_orthogonal_pair_count,
        "interacting_pair_count": interacting_pair_count,
        "order_dependent_pair_count": order_dependent_pair_count,
        "nonorthogonal_pair_count": nonorthogonal_pair_count,
        "new_observation_required_pair_count": new_observation_required_pair_count,
        "underdefined_pair_count": underdefined_pair_count,
        "parameter_ownership_conflict_count": parameter_conflicts,
        "stage_ownership_conflict_count": stage_conflicts,
        "shared_output_distinct_parameter_count": shared_output_distinct,
        "original_design_cell_count": design_snapshot["design_cell_count"],
        "audited_design_cell_count": len(design_audit),
        "reference_cell_count": sum(row["reference_cell"] for row in design_audit),
        "confounded_cell_count": confounded_cell_count,
        "order_dependent_cell_count": order_dependent_cell_count,
        "composite_refactor_cell_count": composite_refactor_cell_count,
    })

    execution_boundary = {
        "primitive_operator_contract_defined": True,
        "orthogonality_contract_adopted": False,
        "primitive_operator_contract_adopted": False,
        "axis_disposition_adopted": False,
        "operator_execution_count": 0,
        "sensitivity_grid_execution_approved": False,
        "identified_set_reestimation_approved": False,
        "turnbull_reestimation_approved": False,
        "phase2_authorized": False,
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
        "new_assumption_created": False,
        "alternate_enum_created": False,
        "numeric_level_created": False,
        "clock_resolution_estimated": False,
        "clock_skew_value_created": False,
        "endpoint_uncertainty_value_created": False,
        "upstream_artifact_modified": False,
        "r2d1u_or_r2d1v_registry_modified": False,
        "design_cell_added_deleted_or_changed": False,
        "operator_executed": False,
        "timestamp_transformed": False,
        "interval_reconstructed": False,
        "turnbull_estimator_executed": False,
        "identified_set_reestimated": False,
        "point_estimate_calculated": False,
        "robustness_judged": False,
        "simulator_applied": False,
        "phase2_executed": False,
        "baseline_rerun_executed": False,
        "training_or_retraining_executed": False,
        "new_assumption_count": 0,
        "new_numeric_level_count": 0,
        "new_alternate_enum_count": 0,
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
    mutation = compare_snapshot(upstream_before, file_snapshot(upstream))
    dump_json(out_dir / "upstream_mutation_audit.json", mutation)
    dump_json(out_dir / "lineage_field_audit.json", {
        "authoritative_upstream_only": True,
        "authoritative_upstream_artifact": str(upstream),
        "upstream_registry_modified": False,
        "symbolic_level_resolution_performed": False,
        "operator_execution_performed": False,
    })
    dump_json(out_dir / "forbidden_field_audit.json", forbidden_field_audit(out_dir))

    parquet_check = parquet_json_check(out_dir)
    secret_audit = secret_scan(out_dir)
    forbidden = strict_read_json(out_dir / "forbidden_field_audit.json")

    gate_status = SUCCESS_GATE
    next_action = "Refined symbolic level-domain specification only"
    if not upstream_validation["checks"]["gate_matches"]:
        gate_status = "FAIL_UPSTREAM_GATE_MISMATCH"
    elif upstream_validation["upstream_integrity_status"] != "PASS":
        gate_status = "FAIL_UPSTREAM_INTEGRITY"
    elif len(SYMBOLIC_TARGETS) != 6:
        gate_status = "FAIL_SYMBOLIC_TARGET_COUNT_MISMATCH"
    elif strict_read_json(out_dir / "assumption_only_preservation_audit.json")["assumption_only_level_mutated"]:
        gate_status = "FAIL_ASSUMPTION_ONLY_LEVEL_MUTATED"
    elif strict_read_json(out_dir / "scheduled_layover_exclusion_audit.json")["scheduled_layover_reintroduced"]:
        gate_status = "FAIL_SCHEDULED_LAYOVER_REINTRODUCED"
    elif new_numeric_level_count:
        gate_status = "FAIL_UNSUPPORTED_NUMERIC_LEVEL_INVENTED"
    elif new_alternate_enum_count:
        gate_status = "FAIL_NEW_ALTERNATE_ENUM_INVENTED"
    elif new_assumption_count:
        gate_status = "FAIL_NEW_ASSUMPTION_CREATED"
    elif parameter_conflicts:
        gate_status = "FAIL_OPERATOR_PARAMETER_OWNERSHIP_DUPLICATION"
    elif any(row["mapped"] is False and row["axis_id"] != "scheduled_layover_treatment" for row in mapping):
        gate_status = "FAIL_AXIS_OPERATOR_MAPPING_MISSING"
    elif any(row["orthogonality_commutativity_conflated"] for row in commutativity):
        gate_status = "FAIL_ORTHOGONALITY_COMMUTATIVITY_CONFLATION"
    elif len(design_audit) != design_snapshot["design_cell_count"]:
        gate_status = "FAIL_DESIGN_MATRIX_CARDINALITY_CHANGED"
    elif operator_execution_count:
        gate_status = "FAIL_OPERATOR_EXECUTION_DETECTED"
    elif any(row["candidate_execution_ready"] or row["execution_approved"] or row["reestimation_approved"] for row in design_audit):
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
        "upstream_gate": gate_snapshot.get("gate"),
        "upstream_integrity_status": upstream_validation["upstream_integrity_status"],
        "symbolic_target_count": len(SYMBOLIC_TARGETS),
        "primitive_operator_count": len(ops),
        "mapped_axis_count": sum(row["mapped"] for row in mapping),
        "unmapped_axis_count": sum(not row["mapped"] for row in mapping),
        "pairwise_refinement_count": len(pairs),
        "semantically_orthogonal_pair_count": semantically_orthogonal_pair_count,
        "interacting_pair_count": interacting_pair_count,
        "order_dependent_pair_count": order_dependent_pair_count,
        "nonorthogonal_pair_count": nonorthogonal_pair_count,
        "new_observation_required_pair_count": new_observation_required_pair_count,
        "underdefined_pair_count": underdefined_pair_count,
        "parameter_ownership_conflict_count": parameter_conflicts,
        "stage_ownership_conflict_count": stage_conflicts,
        "shared_output_distinct_parameter_count": shared_output_distinct,
        "keep_separate_axis_count": keep_separate_axis_count,
        "interacting_axis_count": interacting_axis_count,
        "order_dependent_axis_count": order_dependent_axis_count,
        "composite_axis_candidate_count": 0,
        "assumption_only_axis_count": assumption_only_axis_count,
        "excluded_axis_count": excluded_axis_count,
        "original_design_cell_count": design_snapshot["design_cell_count"],
        "audited_design_cell_count": len(design_audit),
        "reference_cell_count": sum(row["reference_cell"] for row in design_audit),
        "confounded_cell_count": confounded_cell_count,
        "order_dependent_cell_count": order_dependent_cell_count,
        "composite_refactor_cell_count": composite_refactor_cell_count,
        "empirical_level_resolved_count": empirical_level_resolved_count,
        "new_numeric_level_count": new_numeric_level_count,
        "new_alternate_enum_count": new_alternate_enum_count,
        "new_assumption_count": new_assumption_count,
        "operator_execution_count": operator_execution_count,
        "orthogonality_contract_adopted": False,
        "axis_disposition_adopted": False,
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
            "# R2D-1W Sensitivity-Axis Orthogonality Refinement Contract",
            "",
            f"- Artifact: `{out_dir}`",
            f"- Upstream gate: `{gate_snapshot.get('gate')}`",
            f"- Gate: `{gate_status}`",
            "",
            "## Operator Contract",
            f"- Symbolic targets: `{len(SYMBOLIC_TARGETS)}`",
            f"- Primitive operators: `{len(ops)}`",
            f"- Mapped/unmapped axes: `{sum(row['mapped'] for row in mapping)}/{sum(not row['mapped'] for row in mapping)}`",
            f"- Pairwise refinements: `{len(pairs)}`",
            f"- Semantic/order-dependent pairs: `{semantically_orthogonal_pair_count}/{order_dependent_pair_count}`",
            f"- Ownership conflicts: `{parameter_conflicts}`",
            "",
            "## Locks",
            "- Orthogonality contract adopted: `false`",
            "- Primitive operator contract adopted: `false`",
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
        "upstream_gate": gate_snapshot.get("gate"),
        "upstream_integrity_status": upstream_validation["upstream_integrity_status"],
        "symbolic_target_count": len(SYMBOLIC_TARGETS),
        "primitive_operator_count": len(ops),
        "mapped_axis_count": sum(row["mapped"] for row in mapping),
        "unmapped_axis_count": sum(not row["mapped"] for row in mapping),
        "pairwise_refinement_count": len(pairs),
        "semantically_orthogonal_pair_count": semantically_orthogonal_pair_count,
        "interacting_pair_count": interacting_pair_count,
        "order_dependent_pair_count": order_dependent_pair_count,
        "nonorthogonal_pair_count": nonorthogonal_pair_count,
        "new_observation_required_pair_count": new_observation_required_pair_count,
        "underdefined_pair_count": underdefined_pair_count,
        "parameter_ownership_conflict_count": parameter_conflicts,
        "stage_ownership_conflict_count": stage_conflicts,
        "original_design_cell_count": design_snapshot["design_cell_count"],
        "audited_design_cell_count": len(design_audit),
        "reference_cell_count": sum(row["reference_cell"] for row in design_audit),
        "empirical_level_resolved_count": 0,
        "new_numeric_level_count": 0,
        "new_alternate_enum_count": 0,
        "new_assumption_count": 0,
        "operator_execution_count": 0,
        "orthogonality_contract_adopted": False,
        "axis_disposition_adopted": False,
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
        raise RuntimeError(f"secret scan changed after final report/gate creation: {final_secret}")
    if final_forbidden["forbidden_operation_count"] != forbidden["forbidden_operation_count"]:
        raise RuntimeError(f"forbidden audit changed after final report/gate creation: {final_forbidden}")

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
    print(f"upstream gate:\n{gate_snapshot.get('gate')}")
    print(f"symbolic targets:\n{len(SYMBOLIC_TARGETS)}")
    print(f"primitive operators:\n{len(ops)}")
    print(f"mapped/unmapped axes:\n{sum(row['mapped'] for row in mapping)}/{sum(not row['mapped'] for row in mapping)}")
    print(f"pairwise refinements:\n{len(pairs)}")
    print(f"semantic/interacting/order-dependent/nonorthogonal pairs:\n{semantically_orthogonal_pair_count}/{interacting_pair_count}/{order_dependent_pair_count}/{nonorthogonal_pair_count}")
    print(f"ownership conflicts:\n{parameter_conflicts}")
    print(f"original/audited design cells:\n{design_snapshot['design_cell_count']}/{len(design_audit)}")
    print(f"reference cells:\n{sum(row['reference_cell'] for row in design_audit)}")
    print(f"confounded/order-dependent/composite-refactor cells:\n{confounded_cell_count}/{order_dependent_cell_count}/{composite_refactor_cell_count}")
    print("empirical resolved/new numeric/new enum/new assumption/operator execution:\n0/0/0/0/0")
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
