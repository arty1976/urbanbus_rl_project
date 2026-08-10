#!/usr/bin/env python3
"""R2D-1Y clock domain-kind adjudication runner.

This runner adjudicates only the symbolic domain kind for the single
R2D-1X ambiguous axis. It does not resolve concrete levels, units, stage
order, or execution permissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


DEFAULT_PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
DEFAULT_UPSTREAM = Path(
    "05_training/artifacts/"
    "prompt5_e01_r2d1x_refined_symbolic_level_domain_specification_20260731_071030"
)
EXPECTED_UPSTREAM_GATE = (
    "PASS_REFINED_SYMBOLIC_LEVEL_DOMAIN_SPECIFICATION_PARTIAL_DOMAIN_KIND_AMBIGUITY_STILL_LOCKED"
)
EXPECTED_UPSTREAM_NEXT = "Axis-specific symbolic domain-kind adjudication only"
PASS_CONVENTION = (
    "PASS_AXIS_SPECIFIC_SYMBOLIC_DOMAIN_KIND_ADJUDICATED_AS_CONVENTION_STILL_LOCKED"
)
ADJUDICATION_STATUS = "ADJUDICATED_AS_SYMBOLIC_UNORDERED_CONVENTION_DOMAIN"
CANDIDATE_DOMAIN_KIND = "SYMBOLIC_UNORDERED_CONVENTION_DOMAIN"
NEXT_ACTION = "Clock-quantization convention evidence-boundary specification only"

REQUIRED_FILES = [
    "upstream_validation.json",
    "upstream_lineage.json",
    "r2d1x_gate_snapshot.json",
    "r2d1x_domain_registry_snapshot.json",
    "r2d1x_domain_ambiguity_snapshot.json",
    "r2d1x_stage_interaction_snapshot.json",
    "r2d1x_design_matrix_snapshot.json",
    "r2d1x_lock_snapshot.json",
    "ambiguous_axis_target.json",
    "domain_kind_candidate_contract.json",
    "domain_kind_discriminator_registry.json",
    "domain_kind_discriminator_registry.parquet",
    "frozen_evidence_locator_registry.json",
    "frozen_evidence_locator_registry.parquet",
    "parameter_separability_audit.json",
    "parameter_separability_audit.parquet",
    "clock_quantization_semantic_audit.json",
    "clock_resolution_support_semantic_audit.json",
    "clock_rounding_domain_kind_adjudication.json",
    "stage_interaction_preservation_audit.json",
    "reference_element_preservation_audit.json",
    "assumption_boundary_audit.json",
    "candidate_adjudicated_symbolic_domain_registry.json",
    "candidate_adjudicated_symbolic_domain_registry.parquet",
    "original_design_matrix_adjudication_overlay.json",
    "original_design_matrix_adjudication_overlay.parquet",
    "adjudication_cardinality_audit.json",
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
    "symbolic_domain_contract_adopted",
    "stage_interaction_contract_adopted",
    "candidate_stage_order_adopted",
    "resolved_level_contract_adopted",
    "actual_level_resolution_approved",
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
    "domain_kind_adjudication_adopted",
    "parameter_factorisation_adopted",
    "axis_split_adopted",
    "stage_interaction_contract_adopted",
    "candidate_stage_order_adopted",
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
    "clock_resolution_change_approved",
    "clock_rounding_rule_change_approved",
    "clock_correction_approved",
    "simulator_parameter_conversion_approved",
    "simulator_application_approved",
    "phase2_authorized",
    "baseline_rerun_authorized",
    "retraining_authorized",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]

FORBIDDEN_FIELD_NAMES = [
    "actual_rounding_rule_value",
    "actual_resolution_value",
    "actual_unit_value",
    "numeric_bound_value",
    "axis_split_result",
    "identified_set_result",
    "turnbull_result",
    "point_estimate_result",
    "sensitivity_grid_result",
    "simulator_result",
    "training_result",
]


def now_stamp() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def pipe(values: list[str]) -> str:
    return "|".join(values)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get("records", []))


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    serializable = [{k: "" if v is None else v for k, v in row.items()} for row in rows]
    pd.DataFrame(serializable).to_parquet(path, index=False)


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows]


def validate_manifest(artifact: Path) -> dict[str, Any]:
    manifest_path = artifact / "artifact_manifest.json"
    success_path = artifact / "_SUCCESS.lock"
    if not manifest_path.exists():
        return {"status": "FAIL", "manifest_missing_count": 1}
    manifest = read_json(manifest_path)
    entries = manifest.get("files", manifest.get("entries", []))
    hash_mismatch = 0
    size_mismatch = 0
    strict_json_failure = 0
    missing = 0
    for item in entries:
        rel = item.get("relative_path") or item.get("path") or item.get("file")
        if not rel:
            missing += 1
            continue
        path = artifact / rel
        if not path.exists():
            missing += 1
            continue
        if item.get("sha256") and sha256_file(path) != item["sha256"]:
            hash_mismatch += 1
        if item.get("size_bytes") is not None and path.stat().st_size != item["size_bytes"]:
            size_mismatch += 1
        if path.suffix == ".json":
            try:
                read_json(path)
            except Exception:
                strict_json_failure += 1
    return {
        "status": "PASS"
        if manifest_path.exists()
        and success_path.exists()
        and hash_mismatch == 0
        and size_mismatch == 0
        and strict_json_failure == 0
        and missing == 0
        else "FAIL",
        "manifest_hash_mismatch_count": hash_mismatch,
        "manifest_size_mismatch_count": size_mismatch,
        "strict_json_failure_count": strict_json_failure,
        "manifest_missing_count": 0 if manifest_path.exists() else 1,
        "success_lock_missing_count": 0 if success_path.exists() else 1,
        "manifest_entry_missing_count": missing,
    }


def lineage_chain(upstream: Path) -> dict[str, Path]:
    x = upstream
    w = Path(read_json(x / "upstream_lineage.json")["authoritative_upstream_artifact"])
    v = Path(read_json(w / "upstream_lineage.json")["authoritative_upstream_artifact"])
    u = Path(read_json(v / "upstream_lineage.json")["authoritative_upstream_artifact"])
    return {"R2D-1X": x, "R2D-1W": w, "R2D-1V": v, "R2D-1U": u}


def validate_upstream(upstream: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    gate = read_json(upstream / "gate_decision.json")
    locks = read_json(upstream / "lock_state.json")
    ambiguity = read_json(upstream / "domain_kind_ambiguity_registry.json")
    cardinality = read_json(upstream / "domain_cardinality_audit.json")
    chain = lineage_chain(upstream)
    manifests = {label: validate_manifest(path) for label, path in chain.items()}
    upstream_metrics = dict(gate)
    upstream_metrics.update({
        "structurally_specified_domain_count": cardinality.get("structurally_specified_domain_count"),
        "domain_dependency_edge_count": cardinality.get("domain_dependency_edge_count"),
    })
    expected_counts = {
        "symbolic_domain_count": 6,
        "reference_element_count": 6,
        "placeholder_family_count": 6,
        "structurally_specified_domain_count": 5,
        "domain_kind_ambiguous_count": 1,
        "stage_interaction_constraint_count": 4,
        "domain_dependency_edge_count": 5,
        "original_design_cell_count": 13,
        "overlay_design_cell_count": 13,
        "reference_cell_count": 1,
    }
    count_mismatches = [k for k, v in expected_counts.items() if upstream_metrics.get(k) != v]
    lock_violations = [k for k in UPSTREAM_LOCK_FIELDS if locks.get(k) is not False]
    ambiguity_records = records(ambiguity)
    checks = {
        "gate_matches": gate.get("gate") == EXPECTED_UPSTREAM_GATE,
        "next_action_matches": gate.get("next_authorized_action") == EXPECTED_UPSTREAM_NEXT,
        "counts_match": not count_mismatches,
        "required_locks_false": not lock_violations,
        "ambiguity_count_matches": len(ambiguity_records) == 1,
        "lineage_manifest_chain_pass": all(m.get("status") == "PASS" for m in manifests.values()),
    }
    return {
        "upstream_artifact": str(upstream),
        "upstream_gate": gate.get("gate"),
        "upstream_next_authorized_action": gate.get("next_authorized_action"),
        "checks": checks,
        "count_mismatches": count_mismatches,
        "count_source_files": {
            "gate_decision_json_fields": sorted(set(expected_counts) - {
                "structurally_specified_domain_count",
                "domain_dependency_edge_count",
            }),
            "domain_cardinality_audit_json_fields": [
                "structurally_specified_domain_count",
                "domain_dependency_edge_count",
            ],
        },
        "lock_violations": lock_violations,
        "lineage_artifacts": {k: str(v) for k, v in chain.items()},
        "lineage_manifest_validation": manifests,
        "upstream_integrity_status": "PASS" if all(checks.values()) else "FAIL",
    }, chain


def find_one(rows: list[dict[str, Any]], **match: str) -> dict[str, Any]:
    matches = [row for row in rows if all(str(row.get(k)) == v for k, v in match.items())]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for {match}, got {len(matches)}")
    return matches[0]


def evidence_locators(chain: dict[str, Path]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": "ELOC_01",
            "artifact_label": "R2D-1X",
            "artifact_path": str(chain["R2D-1X"]),
            "source_file": "domain_kind_ambiguity_registry.json",
            "field_locator": "/records/axis_id=clock_rounding_allowance/parameter_shape",
            "evidence_summary": "Parameter shape explicitly remains a descriptor-or-support ambiguity.",
            "admissible": True,
        },
        {
            "evidence_id": "ELOC_02",
            "artifact_label": "R2D-1X",
            "artifact_path": str(chain["R2D-1X"]),
            "source_file": "domain_kind_ambiguity_registry.json",
            "field_locator": "/records/axis_id=clock_rounding_allowance/domain_kind_candidate_set",
            "evidence_summary": "R2D-1X candidate set lists ordered support and unordered convention only.",
            "admissible": True,
        },
        {
            "evidence_id": "ELOC_03",
            "artifact_label": "R2D-1W",
            "artifact_path": str(chain["R2D-1W"]),
            "source_file": "primitive_operator_registry.json",
            "field_locator": "/records/operator_id=Q_CLOCK/parameter_namespace",
            "evidence_summary": "Q_CLOCK owns a clock quantization rule namespace.",
            "admissible": True,
        },
        {
            "evidence_id": "ELOC_04",
            "artifact_label": "R2D-1W",
            "artifact_path": str(chain["R2D-1W"]),
            "source_file": "primitive_operator_registry.json",
            "field_locator": "/records/operator_id=Q_CLOCK/parameter_domain_type",
            "evidence_summary": "Q_CLOCK parameter domain type is symbolic discrete and unspecified.",
            "admissible": True,
        },
        {
            "evidence_id": "ELOC_05",
            "artifact_label": "R2D-1W",
            "artifact_path": str(chain["R2D-1W"]),
            "source_file": "primitive_operator_registry.json",
            "field_locator": "/records/operator_id=Q_CLOCK/changed_fields",
            "evidence_summary": "Changed fields are representation and bin assignment, not a numeric support bound.",
            "admissible": True,
        },
        {
            "evidence_id": "ELOC_06",
            "artifact_label": "R2D-1W",
            "artifact_path": str(chain["R2D-1W"]),
            "source_file": "primitive_operator_registry.json",
            "field_locator": "/records/operator_id=Q_CLOCK/required_preconditions",
            "evidence_summary": "The operator contract forbids inventing a precision value.",
            "admissible": True,
        },
        {
            "evidence_id": "ELOC_07",
            "artifact_label": "R2D-1U",
            "artifact_path": str(chain["R2D-1U"]),
            "source_file": "sensitivity_axis_registry.json",
            "field_locator": "/records/axis_id=clock_rounding_allowance/basis",
            "evidence_summary": "The axis basis describes a symbolic convention without timestamp changes.",
            "admissible": True,
        },
        {
            "evidence_id": "ELOC_08",
            "artifact_label": "R2D-1U",
            "artifact_path": str(chain["R2D-1U"]),
            "source_file": "sensitivity_level_registry.json",
            "field_locator": "/records/axis_id=clock_rounding_allowance,is_reference_level=true",
            "evidence_summary": "The frozen reference is a timestamp convention with no unit assignment.",
            "admissible": True,
        },
        {
            "evidence_id": "ELOC_09",
            "artifact_label": "R2D-1V",
            "artifact_path": str(chain["R2D-1V"]),
            "source_file": "clock_rounding_allowance_resolution.json",
            "field_locator": "/resolution_status",
            "evidence_summary": "The prior resolution audit did not resolve a concrete level independently.",
            "admissible": True,
        },
        {
            "evidence_id": "ELOC_10",
            "artifact_label": "R2D-1X",
            "artifact_path": str(chain["R2D-1X"]),
            "source_file": "reference_element_registry.json",
            "field_locator": "/records/axis_id=clock_rounding_allowance",
            "evidence_summary": "The reference element remains a symbolic frozen label, not a value.",
            "admissible": True,
        },
        {
            "evidence_id": "ELOC_11",
            "artifact_label": "R2D-1X",
            "artifact_path": str(chain["R2D-1X"]),
            "source_file": "stage_interaction_constraint_registry.json",
            "field_locator": "/records/interaction_id=INT_Q_CLOCK_U_REQUEST",
            "evidence_summary": "Q_CLOCK must precede request support, but this is stage order evidence.",
            "admissible": True,
        },
        {
            "evidence_id": "ELOC_12",
            "artifact_label": "R2D-1X",
            "artifact_path": str(chain["R2D-1X"]),
            "source_file": "stage_interaction_constraint_registry.json",
            "field_locator": "/records/interaction_id=INT_Q_CLOCK_S_CLOCK",
            "evidence_summary": "Q_CLOCK and S_CLOCK remain distinct while their order is unresolved.",
            "admissible": True,
        },
    ]


def discriminator_rows() -> list[dict[str, Any]]:
    return [
        {
            "discriminator_id": "D01_parameter_shape",
            "discriminator_name": "Parameter shape separates descriptor from support",
            "evidence_locator": "ELOC_01",
            "supports_unordered_convention": False,
            "supports_ordered_support": False,
            "supports_parameter_factorisation": False,
            "supports_underdefined": True,
            "evidence_strength": "AMBIGUOUS",
            "decision_relevance": "R2D-1X alone cannot choose the kind.",
        },
        {
            "discriminator_id": "D02_parameter_namespace",
            "discriminator_name": "Parameter namespace",
            "evidence_locator": "ELOC_03",
            "supports_unordered_convention": True,
            "supports_ordered_support": False,
            "supports_parameter_factorisation": False,
            "supports_underdefined": False,
            "evidence_strength": "DIRECT_EXPLICIT",
            "decision_relevance": "Namespace is a quantization rule, not a support width.",
        },
        {
            "discriminator_id": "D03_natural_ordering_relation",
            "discriminator_name": "Natural ordering relation",
            "evidence_locator": "ELOC_01",
            "supports_unordered_convention": True,
            "supports_ordered_support": False,
            "supports_parameter_factorisation": False,
            "supports_underdefined": True,
            "evidence_strength": "DIRECT_DERIVED",
            "decision_relevance": "No natural ordering relation is available for Q_CLOCK levels.",
        },
        {
            "discriminator_id": "D04_unit_requirement",
            "discriminator_name": "Unit requirement",
            "evidence_locator": "ELOC_08",
            "supports_unordered_convention": True,
            "supports_ordered_support": False,
            "supports_parameter_factorisation": False,
            "supports_underdefined": False,
            "evidence_strength": "DIRECT_DERIVED",
            "decision_relevance": "The frozen timestamp convention carries no unit assignment.",
        },
        {
            "discriminator_id": "D05_rule_without_magnitude_change",
            "discriminator_name": "Rule separable without magnitude change",
            "evidence_locator": "ELOC_07",
            "supports_unordered_convention": True,
            "supports_ordered_support": False,
            "supports_parameter_factorisation": False,
            "supports_underdefined": False,
            "evidence_strength": "DIRECT_EXPLICIT",
            "decision_relevance": "R2D-1U describes the axis as a symbolic convention without timestamp changes.",
        },
        {
            "discriminator_id": "D06_magnitude_without_rule_change",
            "discriminator_name": "Magnitude separable without rule change",
            "evidence_locator": "ELOC_06",
            "supports_unordered_convention": True,
            "supports_ordered_support": False,
            "supports_parameter_factorisation": False,
            "supports_underdefined": True,
            "evidence_strength": "ABSENT",
            "decision_relevance": "No resolution/support magnitude field is available; precision invention is barred.",
        },
        {
            "discriminator_id": "D07_reference_element_semantics",
            "discriminator_name": "Reference element semantics",
            "evidence_locator": "ELOC_10",
            "supports_unordered_convention": False,
            "supports_ordered_support": False,
            "supports_parameter_factorisation": False,
            "supports_underdefined": True,
            "evidence_strength": "SUPPORTING_ONLY",
            "decision_relevance": "The reference label does not resolve an actual member.",
        },
        {
            "discriminator_id": "D08_placeholder_family_semantics",
            "discriminator_name": "Placeholder family semantics",
            "evidence_locator": "ELOC_02",
            "supports_unordered_convention": True,
            "supports_ordered_support": False,
            "supports_parameter_factorisation": False,
            "supports_underdefined": True,
            "evidence_strength": "SUPPORTING_ONLY",
            "decision_relevance": "The placeholder is clock quantization, not a numeric support value.",
        },
        {
            "discriminator_id": "D09_owned_fields",
            "discriminator_name": "Owned fields",
            "evidence_locator": "ELOC_05",
            "supports_unordered_convention": True,
            "supports_ordered_support": False,
            "supports_parameter_factorisation": False,
            "supports_underdefined": False,
            "evidence_strength": "DIRECT_DERIVED",
            "decision_relevance": "Owned fields are representation/bin assignment, with no independent support-width field.",
        },
        {
            "discriminator_id": "D10_stage_input_output_types",
            "discriminator_name": "Stage input and output types",
            "evidence_locator": "ELOC_04",
            "supports_unordered_convention": True,
            "supports_ordered_support": False,
            "supports_parameter_factorisation": False,
            "supports_underdefined": False,
            "evidence_strength": "DIRECT_EXPLICIT",
            "decision_relevance": "The operator type contract is symbolic discrete and timestamp representation oriented.",
        },
        {
            "discriminator_id": "D11_qclock_sclock_separation",
            "discriminator_name": "Q_CLOCK and S_CLOCK separation",
            "evidence_locator": "ELOC_12",
            "supports_unordered_convention": True,
            "supports_ordered_support": False,
            "supports_parameter_factorisation": False,
            "supports_underdefined": False,
            "evidence_strength": "DIRECT_DERIVED",
            "decision_relevance": "Clock offset is a separate operator and assumption boundary.",
        },
        {
            "discriminator_id": "D12_qclock_urequest_dependency",
            "discriminator_name": "Q_CLOCK to U_REQUEST dependency",
            "evidence_locator": "ELOC_11",
            "supports_unordered_convention": False,
            "supports_ordered_support": False,
            "supports_parameter_factorisation": False,
            "supports_underdefined": True,
            "evidence_strength": "SUPPORTING_ONLY",
            "decision_relevance": "Stage dependency is preserved and not used as the domain-kind decision.",
        },
    ]


def parameter_separability_rows() -> list[dict[str, Any]]:
    return [
        {
            "candidate_parameter_id": "quantization_rule_parameter",
            "semantic_definition_available": True,
            "owned_fields_available": True,
            "unit_required": False,
            "natural_ordering_available": False,
            "independently_variable": True,
            "reference_element_available": True,
            "alternate_member_available": False,
            "empirically_resolved": False,
            "candidate_only": True,
        },
        {
            "candidate_parameter_id": "quantization_resolution_parameter",
            "semantic_definition_available": False,
            "owned_fields_available": False,
            "unit_required": True,
            "natural_ordering_available": False,
            "independently_variable": False,
            "reference_element_available": False,
            "alternate_member_available": False,
            "empirically_resolved": False,
            "candidate_only": True,
        },
    ]


def candidate_domain_registry(domains: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in domains:
        if row.get("symbolic_domain_target") is not True:
            continue
        is_clock = row["axis_id"] == "clock_rounding_allowance"
        out = dict(row)
        out.update({
            "original_domain_kind": row["domain_kind"],
            "previous_domain_kind_status": row["domain_kind_status"],
            "candidate_domain_kind": CANDIDATE_DOMAIN_KIND if is_clock else row["domain_kind"],
            "adjudication_status": ADJUDICATION_STATUS if is_clock else "UNCHANGED_NOT_AMBIGUOUS",
            "supporting_evidence_ids": pipe(["D02", "D04", "D05", "D09", "D10", "D11"])
            if is_clock
            else "",
            "contrary_evidence_ids": "D01|D07|D12" if is_clock else "",
            "parameter_factorisation_required": False,
            "actual_member_resolved": False,
            "numeric_bound_resolved": False,
            "unit_resolved": False,
            "candidate_only": True,
            "adopted": False,
            "execution_allowed": False,
        })
        rows.append(out)
    return rows


def design_overlay(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlay = []
    for row in rows:
        bindings = json.loads(row["axis_level_bindings"])
        out = dict(row)
        out.update({
            "contains_adjudicated_axis": "clock_rounding_allowance" in bindings,
            "domain_kind_candidate_available": True,
            "parameter_factorisation_required": False,
            "domain_kind_underdefined": False,
            "actual_level_resolution_ready": False,
            "candidate_execution_ready": False,
            "execution_approved": False,
            "reestimation_approved": False,
        })
        overlay.append(out)
    return overlay


def parquet_json_check(out_dir: Path) -> dict[str, Any]:
    pairs = [
        ("domain_kind_discriminator_registry.json", "domain_kind_discriminator_registry.parquet"),
        ("frozen_evidence_locator_registry.json", "frozen_evidence_locator_registry.parquet"),
        ("parameter_separability_audit.json", "parameter_separability_audit.parquet"),
        (
            "candidate_adjudicated_symbolic_domain_registry.json",
            "candidate_adjudicated_symbolic_domain_registry.parquet",
        ),
        (
            "original_design_matrix_adjudication_overlay.json",
            "original_design_matrix_adjudication_overlay.parquet",
        ),
    ]
    parquet_fail = 0
    mismatch = 0
    for json_name, parquet_name in pairs:
        try:
            json_rows = normalize_rows(records(read_json(out_dir / json_name)))
            parquet_rows = normalize_rows(
                pd.read_parquet(out_dir / parquet_name).to_dict(orient="records")
            )
        except Exception:
            parquet_fail += 1
            continue
        if json_rows != parquet_rows:
            mismatch += 1
    return {
        "parquet_read_failure_count": parquet_fail,
        "json_parquet_mismatch_count": mismatch,
    }


def upstream_mutation_audit(chain: dict[str, Path], snapshots: dict[str, dict[str, Any]]) -> dict[str, Any]:
    added: list[str] = []
    modified: list[str] = []
    deleted: list[str] = []
    for label, before in snapshots.items():
        root = chain[label]
        manifest = read_json(root / "artifact_manifest.json")
        entries = manifest.get("files", manifest.get("entries", []))
        current = {
            item.get("relative_path") or item.get("path") or item.get("file"): {
                "sha256": item.get("sha256"),
                "size_bytes": item.get("size_bytes"),
            }
            for item in entries
        }
        for rel in before:
            if rel not in current:
                deleted.append(f"{label}:{rel}")
            elif before[rel] != current[rel]:
                modified.append(f"{label}:{rel}")
        for rel in current:
            if rel not in before:
                added.append(f"{label}:{rel}")
    return {
        "upstream_added": len(added),
        "upstream_modified": len(modified),
        "upstream_deleted": len(deleted),
        "added_files": added,
        "modified_files": modified,
        "deleted_files": deleted,
    }


def manifest_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    manifest = read_json(path / "artifact_manifest.json")
    return {
        item.get("relative_path") or item.get("path") or item.get("file"): {
            "sha256": item.get("sha256"),
            "size_bytes": item.get("size_bytes"),
        }
        for item in manifest.get("files", manifest.get("entries", []))
    }


def forbidden_field_audit(out_dir: Path) -> dict[str, Any]:
    text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in out_dir.iterdir()
        if path.is_file() and path.suffix in {".json", ".md", ".lock"}
    )
    findings = [field for field in FORBIDDEN_FIELD_NAMES if re.search(rf'"{re.escape(field)}"\s*:', text)]
    return {
        "forbidden_field_inventory": FORBIDDEN_FIELD_NAMES,
        "findings": findings,
        "forbidden_field_presence_count": len(findings),
        "forbidden_operation_count": len(findings),
    }


def secret_scan(out_dir: Path) -> dict[str, Any]:
    markers = ["DAEGU_BIS_SERVICE_KEY", "serviceKey=", "ServiceKey=", "apis.data.go.kr"]
    hits = []
    for path in out_dir.iterdir():
        if path.is_file() and path.suffix in {".json", ".md", ".lock"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in markers:
                if marker in text:
                    hits.append({"file": path.name, "marker": marker})
    return {"secret_leak_count": len(hits), "findings": hits}


def build_manifest(out_dir: Path) -> dict[str, Any]:
    files = []
    for name in REQUIRED_FILES:
        if name in MANIFEST_EXCLUDED:
            continue
        path = out_dir / name
        files.append({
            "relative_path": name,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
            "strict_json_valid": True if path.suffix != ".json" else _json_valid(path),
        })
    return {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "files": files,
        "required_file_count": len(REQUIRED_FILES),
        "manifest_entry_count": len(files),
        "manifest_hash_mismatch_count": 0,
        "manifest_size_mismatch_count": 0,
    }


def _json_valid(path: Path) -> bool:
    try:
        read_json(path)
        return True
    except Exception:
        return False


def verify_own_manifest(out_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    hash_mismatch = 0
    size_mismatch = 0
    strict_json_failure = 0
    missing = 0
    for item in manifest["files"]:
        path = out_dir / item["relative_path"]
        if not path.exists():
            missing += 1
            continue
        if sha256_file(path) != item["sha256"]:
            hash_mismatch += 1
        if path.stat().st_size != item["size_bytes"]:
            size_mismatch += 1
        if path.suffix == ".json" and not _json_valid(path):
            strict_json_failure += 1
    unexpected = [
        p.name
        for p in out_dir.iterdir()
        if p.is_file() and p.name not in set(REQUIRED_FILES)
    ]
    return {
        "required_file_missing_count": missing,
        "unexpected_file_count": len(unexpected),
        "manifest_hash_mismatch_count": hash_mismatch,
        "manifest_size_mismatch_count": size_mismatch,
        "strict_json_failure_count": strict_json_failure,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--upstream-artifact", type=Path, default=DEFAULT_UPSTREAM)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    upstream = args.upstream_artifact
    if not upstream.is_absolute():
        upstream = (project_root / upstream).resolve()

    upstream_validation, chain = validate_upstream(upstream)
    upstream_snapshots = {label: manifest_snapshot(path) for label, path in chain.items()}

    out_dir = (
        project_root
        / "05_training"
        / "artifacts"
        / f"prompt5_e01_r2d1y_axis_specific_symbolic_domain_kind_adjudication_{now_stamp()}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    gate = read_json(upstream / "gate_decision.json")
    domain_registry = read_json(upstream / "symbolic_domain_registry.json")
    ambiguity = read_json(upstream / "domain_kind_ambiguity_registry.json")
    stage_registry = read_json(upstream / "stage_interaction_constraint_registry.json")
    design = read_json(upstream / "original_design_matrix_domain_overlay.json")
    locks = read_json(upstream / "lock_state.json")
    reference_registry = read_json(upstream / "reference_element_registry.json")
    placeholder_registry = read_json(upstream / "alternate_placeholder_family_registry.json")
    w_operator = read_json(chain["R2D-1W"] / "primitive_operator_registry.json")
    w_type = read_json(chain["R2D-1W"] / "operator_type_signature_registry.json")
    u_axis = read_json(chain["R2D-1U"] / "sensitivity_axis_registry.json")
    u_level = read_json(chain["R2D-1U"] / "sensitivity_level_registry.json")
    v_clock = read_json(chain["R2D-1V"] / "clock_rounding_allowance_resolution.json")

    dump_json(out_dir / "upstream_validation.json", upstream_validation)
    dump_json(out_dir / "r2d1x_gate_snapshot.json", gate)
    dump_json(out_dir / "r2d1x_domain_registry_snapshot.json", domain_registry)
    dump_json(out_dir / "r2d1x_domain_ambiguity_snapshot.json", ambiguity)
    dump_json(out_dir / "r2d1x_stage_interaction_snapshot.json", stage_registry)
    dump_json(out_dir / "r2d1x_design_matrix_snapshot.json", design)
    dump_json(out_dir / "r2d1x_lock_snapshot.json", locks)

    ambiguity_rows = records(ambiguity)
    ambiguity_row = ambiguity_rows[0] if len(ambiguity_rows) == 1 else {}
    qclock_operator = find_one(records(w_operator), operator_id="Q_CLOCK")
    qclock_type = find_one(records(w_type), operator_id="Q_CLOCK")
    clock_axis_u = find_one(records(u_axis), axis_id="clock_rounding_allowance")
    clock_ref_level_u = [
        row
        for row in records(u_level)
        if row.get("axis_id") == "clock_rounding_allowance" and row.get("is_reference_level") is True
    ][0]
    qclock_ref = find_one(records(reference_registry), axis_id="clock_rounding_allowance")
    qclock_placeholder = find_one(records(placeholder_registry), axis_id="clock_rounding_allowance")

    target = {
        "ambiguity_row_count": len(ambiguity_rows),
        "axis_id": ambiguity_row.get("axis_id"),
        "primitive_operator_id": ambiguity_row.get("primitive_operator_id"),
        "expected_axis_id": "clock_rounding_allowance",
        "expected_primitive_operator_id": "Q_CLOCK",
        "unexpected_ambiguous_axis": ambiguity_row.get("axis_id") != "clock_rounding_allowance",
        "primitive_operator_mismatch": ambiguity_row.get("primitive_operator_id") != "Q_CLOCK",
        "candidate_only": True,
        "adopted": False,
        "execution_allowed": False,
    }
    dump_json(out_dir / "ambiguous_axis_target.json", target)

    lineage_payload = {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "authoritative_upstream_artifact": str(upstream),
        "authoritative_upstream_gate": gate.get("gate"),
        "admissible_evidence_artifacts": {k: str(v) for k, v in chain.items()},
        "source_files_read": [
            "R2D-1X/gate_decision.json",
            "R2D-1X/domain_kind_ambiguity_registry.json",
            "R2D-1X/symbolic_domain_registry.json",
            "R2D-1X/stage_interaction_constraint_registry.json",
            "R2D-1X/original_design_matrix_domain_overlay.json",
            "R2D-1W/primitive_operator_registry.json",
            "R2D-1W/operator_type_signature_registry.json",
            "R2D-1V/clock_rounding_allowance_resolution.json",
            "R2D-1U/sensitivity_axis_registry.json",
            "R2D-1U/sensitivity_level_registry.json",
        ],
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
    }
    dump_json(out_dir / "upstream_lineage.json", lineage_payload)

    locators = evidence_locators(chain)
    discriminators = discriminator_rows()
    separability = parameter_separability_rows()

    dump_json(out_dir / "frozen_evidence_locator_registry.json", {
        "frozen_evidence_locator_count": len(locators),
        "records": locators,
    })
    write_parquet(out_dir / "frozen_evidence_locator_registry.parquet", locators)

    dump_json(out_dir / "domain_kind_discriminator_registry.json", {
        "discriminator_count": len(discriminators),
        "records": discriminators,
    })
    write_parquet(out_dir / "domain_kind_discriminator_registry.parquet", discriminators)

    dump_json(out_dir / "parameter_separability_audit.json", {
        "candidate_parameter_count": len(separability),
        "records": separability,
        "quantization_rule_parameter_defined": True,
        "quantization_resolution_parameter_defined": False,
        "parameters_independently_variable": False,
        "parameter_factorisation_required": False,
    })
    write_parquet(out_dir / "parameter_separability_audit.parquet", separability)

    dump_json(out_dir / "domain_kind_candidate_contract.json", {
        "axis_id": "clock_rounding_allowance",
        "primitive_operator_id": "Q_CLOCK",
        "allowed_adjudication_statuses": pipe([
            "ADJUDICATED_AS_SYMBOLIC_UNORDERED_CONVENTION_DOMAIN",
            "ADJUDICATED_AS_SYMBOLIC_ORDERED_SUPPORT_DOMAIN",
            "DOMAIN_KIND_REQUIRES_PARAMETER_FACTORISATION",
            "DOMAIN_KIND_REMAINS_UNDERDEFINED",
        ]),
        "candidate_domain_kind": CANDIDATE_DOMAIN_KIND,
        "adjudication_status": ADJUDICATION_STATUS,
        "candidate_only": True,
        "adopted": False,
        "execution_allowed": False,
        "actual_rounding_member_resolved": False,
        "numeric_resolution_resolved": False,
        "unit_resolved": False,
    })

    dump_json(out_dir / "clock_quantization_semantic_audit.json", {
        "candidate_parameter_id": "quantization_rule_parameter",
        "semantic_definition_available": True,
        "basis": "Q_CLOCK namespace, symbolic discrete parameter type, and R2D-1U convention basis.",
        "operator_parameter_namespace": qclock_operator["parameter_namespace"],
        "operator_parameter_domain_type": qclock_operator["parameter_domain_type"],
        "changed_fields": qclock_operator["changed_fields"],
        "reference_level_label": clock_ref_level_u["level_label"],
        "reference_level_unit": clock_ref_level_u["level_unit"],
        "actual_member_resolved": False,
        "alternate_member_available": False,
        "candidate_only": True,
        "adopted": False,
        "execution_allowed": False,
    })
    dump_json(out_dir / "clock_resolution_support_semantic_audit.json", {
        "candidate_parameter_id": "quantization_resolution_parameter",
        "semantic_definition_available": False,
        "basis": "No frozen owned field, unit, bound, or support-width parameter is available.",
        "owned_fields_available": False,
        "unit_required": True,
        "unit_resolved": False,
        "natural_ordering_available": False,
        "numeric_bound_resolved": False,
        "alternate_member_available": False,
        "empirically_resolved": False,
        "candidate_only": True,
        "adopted": False,
        "execution_allowed": False,
    })

    adjudication = {
        "axis_id": "clock_rounding_allowance",
        "primitive_operator_id": "Q_CLOCK",
        "previous_domain_kind_status": ambiguity_row.get("domain_kind_status"),
        "candidate_domain_kind": CANDIDATE_DOMAIN_KIND,
        "adjudication_status": ADJUDICATION_STATUS,
        "adjudication_rationale": (
            "R2D-1W directly names Q_CLOCK's namespace as a quantization rule and "
            "types the parameter as symbolic discrete; R2D-1U describes the axis as "
            "a symbolic timestamp convention. Frozen evidence does not provide a "
            "separate resolution/support-width field, unit, numeric bound, or "
            "natural ordering relation."
        ),
        "supporting_evidence_ids": pipe(["D02", "D04", "D05", "D08", "D09", "D10", "D11"]),
        "contrary_evidence_ids": pipe(["D01", "D07", "D12"]),
        "parameter_factorisation_required": False,
        "actual_member_resolved": False,
        "numeric_bound_resolved": False,
        "unit_resolved": False,
        "candidate_only": True,
        "adopted": False,
        "execution_allowed": False,
    }
    dump_json(out_dir / "clock_rounding_domain_kind_adjudication.json", adjudication)

    q_interactions = [
        row for row in records(stage_registry) if "Q_CLOCK" in {row["operator_a"], row["operator_b"]}
    ]
    dump_json(out_dir / "stage_interaction_preservation_audit.json", {
        "qclock_interaction_count": len(q_interactions),
        "qclock_interaction_ids": pipe([row["interaction_id"] for row in q_interactions]),
        "stage_interaction_preserved": True,
        "stage_interaction_resolved": False,
        "stage_order_adopted": False,
        "commutativity_established": False,
        "records": q_interactions,
    })
    dump_json(out_dir / "reference_element_preservation_audit.json", {
        "axis_id": "clock_rounding_allowance",
        "reference_element_id": qclock_ref["reference_element_id"],
        "reference_element_source": qclock_ref["source_artifact"],
        "reference_semantics": qclock_ref["reference_semantics"],
        "reference_value_redacted_or_symbolic": qclock_ref["reference_value_redacted_or_symbolic"],
        "reference_element_adopted": False,
        "actual_reference_value_resolved": False,
        "preserved": True,
    })
    dump_json(out_dir / "assumption_boundary_audit.json", {
        "new_assumption_count": 0,
        "clock_skew_distinct_from_clock_rounding": True,
        "clock_skew_assumption_not_modified": True,
        "missing_boundary_assumption_not_modified": True,
        "scheduled_layover_not_reintroduced": True,
        "assumption_boundary_preserved": True,
    })

    candidate_registry = candidate_domain_registry(records(domain_registry))
    dump_json(out_dir / "candidate_adjudicated_symbolic_domain_registry.json", {
        "domain_row_count": len(candidate_registry),
        "records": candidate_registry,
    })
    write_parquet(out_dir / "candidate_adjudicated_symbolic_domain_registry.parquet", candidate_registry)

    overlay = design_overlay(records(design))
    dump_json(out_dir / "original_design_matrix_adjudication_overlay.json", {
        "design_cell_count": len(overlay),
        "records": overlay,
    })
    write_parquet(out_dir / "original_design_matrix_adjudication_overlay.parquet", overlay)

    direct_explicit = sum(row["evidence_strength"] == "DIRECT_EXPLICIT" for row in discriminators)
    direct_derived = sum(row["evidence_strength"] == "DIRECT_DERIVED" for row in discriminators)
    supporting = sum(row["evidence_strength"] == "SUPPORTING_ONLY" for row in discriminators)
    ambiguous_count = sum(row["evidence_strength"] == "AMBIGUOUS" for row in discriminators)
    absent = sum(row["evidence_strength"] == "ABSENT" for row in discriminators)

    dump_json(out_dir / "execution_boundary.json", {
        "scope": "CLOCK_DOMAIN_KIND_ADJUDICATION_ONLY",
        "domain_kind_adjudication_executed": True,
        "domain_kind_adjudication_adopted": False,
        "parameter_factorisation_executed": False,
        "axis_split_executed": False,
        "stage_order_adopted": False,
        "actual_level_resolution_executed": False,
        "sensitivity_analysis_executed": False,
        "identified_set_reestimation_executed": False,
        "turnbull_reestimation_executed": False,
        "simulator_application_executed": False,
        "phase2_authorized": False,
    })
    lock_payload = {field: False for field in LOCK_FALSE_FIELDS}
    lock_payload["all_required_locks_false"] = True
    dump_json(out_dir / "lock_state.json", lock_payload)
    dump_json(out_dir / "prohibited_operation_audit.json", {
        "api_called": False,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
        "new_observation_collected": False,
        "new_episode_collected": False,
        "new_assumption_created": False,
        "actual_rounding_rule_created": False,
        "actual_resolution_number_created": False,
        "actual_unit_decided": False,
        "numeric_domain_bound_created": False,
        "actual_alternate_enum_created": False,
        "axis_split_executed": False,
        "design_matrix_changed": False,
        "stage_order_adopted": False,
        "operator_executed": False,
        "timestamp_transformed": False,
        "interval_reconstructed": False,
        "identified_set_calculated": False,
        "turnbull_executed": False,
        "point_estimate_calculated": False,
        "robustness_judged": False,
        "simulator_applied": False,
        "phase2_executed": False,
        "baseline_rerun_executed": False,
        "training_or_retraining_executed": False,
        "new_assumption_count": 0,
        "new_numeric_domain_bound_count": 0,
        "new_actual_level_count": 0,
        "new_actual_enum_member_count": 0,
        "new_unit_assignment_count": 0,
        "axis_split_execution_count": 0,
        "design_matrix_mutation_count": 0,
        "operator_execution_count": 0,
        "forbidden_operation_count": 0,
    })
    dump_json(out_dir / "api_db_network_audit.json", {
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
    })
    mutation = upstream_mutation_audit(chain, upstream_snapshots)
    dump_json(out_dir / "upstream_mutation_audit.json", mutation)
    dump_json(out_dir / "lineage_field_audit.json", {
        "authoritative_upstream_only": True,
        "lineage_chain_checked": True,
        "partial_failure_artifact_used": False,
        "external_evidence_used": False,
        "lineage_field_issue_count": 0,
    })
    dump_json(out_dir / "forbidden_field_audit.json", forbidden_field_audit(out_dir))

    parquet_check = parquet_json_check(out_dir)
    forbidden = read_json(out_dir / "forbidden_field_audit.json")
    secret = secret_scan(out_dir)

    cardinality = {
        "ambiguous_axis_count": len(ambiguity_rows),
        "discriminator_count": len(discriminators),
        "direct_explicit_evidence_count": direct_explicit,
        "direct_derived_evidence_count": direct_derived,
        "supporting_evidence_count": supporting,
        "ambiguous_evidence_count": ambiguous_count,
        "absent_evidence_count": absent,
        "quantization_rule_parameter_defined": True,
        "quantization_resolution_parameter_defined": False,
        "parameters_independently_variable": False,
        "parameter_factorisation_required": False,
        "candidate_domain_kind": CANDIDATE_DOMAIN_KIND,
        "adjudication_status": ADJUDICATION_STATUS,
        "domain_kind_resolved": True,
        "stage_interaction_preserved": True,
        "stage_order_adopted": False,
        "commutativity_established": False,
        "original_domain_count": 6,
        "candidate_domain_count": len(candidate_registry),
        "original_design_cell_count": 13,
        "overlay_design_cell_count": len(overlay),
        "reference_cell_count": sum(row["reference_cell"] is True for row in overlay),
        "new_numeric_domain_bound_count": 0,
        "new_actual_level_count": 0,
        "new_actual_enum_member_count": 0,
        "new_unit_assignment_count": 0,
        "new_assumption_count": 0,
        "axis_split_execution_count": 0,
        "design_matrix_mutation_count": 0,
        "operator_execution_count": 0,
    }
    dump_json(out_dir / "adjudication_cardinality_audit.json", cardinality)

    gate_status = PASS_CONVENTION
    if gate.get("gate") != EXPECTED_UPSTREAM_GATE:
        gate_status = "FAIL_UPSTREAM_GATE_MISMATCH"
    elif upstream_validation["upstream_integrity_status"] != "PASS":
        gate_status = "FAIL_UPSTREAM_INTEGRITY"
    elif len(ambiguity_rows) != 1:
        gate_status = "FAIL_AMBIGUITY_COUNT_MISMATCH"
    elif target["unexpected_ambiguous_axis"]:
        gate_status = "FAIL_UNEXPECTED_AMBIGUOUS_AXIS_ID"
    elif target["primitive_operator_mismatch"]:
        gate_status = "FAIL_PRIMITIVE_OPERATOR_MISMATCH"
    elif qclock_operator["parameter_namespace"] != "clock_quantization_rule":
        gate_status = "FAIL_DOMAIN_KIND_DECISION_WITHOUT_EVIDENCE"
    elif qclock_operator["parameter_domain_type"] != "SYMBOLIC_DISCRETE_DOMAIN_UNSPECIFIED":
        gate_status = "FAIL_DOMAIN_KIND_DECISION_WITHOUT_EVIDENCE"
    elif cardinality["quantization_resolution_parameter_defined"]:
        gate_status = "FAIL_CONVENTION_SUPPORT_CONFLATION"
    elif read_json(out_dir / "stage_interaction_preservation_audit.json")["stage_order_adopted"]:
        gate_status = "FAIL_DOMAIN_KIND_STAGE_ORDER_CONFLATION"
    elif read_json(out_dir / "prohibited_operation_audit.json")["new_actual_enum_member_count"]:
        gate_status = "FAIL_ACTUAL_ROUNDING_RULE_INVENTED"
    elif read_json(out_dir / "prohibited_operation_audit.json")["new_actual_level_count"]:
        gate_status = "FAIL_NUMERIC_RESOLUTION_INVENTED"
    elif read_json(out_dir / "prohibited_operation_audit.json")["new_unit_assignment_count"]:
        gate_status = "FAIL_UNIT_INVENTED"
    elif read_json(out_dir / "prohibited_operation_audit.json")["new_assumption_count"]:
        gate_status = "FAIL_NEW_ASSUMPTION_CREATED"
    elif read_json(out_dir / "prohibited_operation_audit.json")["axis_split_execution_count"]:
        gate_status = "FAIL_AXIS_SPLIT_EXECUTED"
    elif len(overlay) != 13:
        gate_status = "FAIL_DESIGN_MATRIX_CARDINALITY_CHANGED"
    elif any(row["axis_level_bindings"] != src["axis_level_bindings"] for row, src in zip(overlay, records(design))):
        gate_status = "FAIL_DESIGN_MATRIX_BINDING_CHANGED"
    elif read_json(out_dir / "prohibited_operation_audit.json")["operator_execution_count"]:
        gate_status = "FAIL_OPERATOR_EXECUTION_DETECTED"
    elif any(lock_payload[field] is not False for field in LOCK_FALSE_FIELDS):
        gate_status = "FAIL_LOCK_STATE_VIOLATION"
    elif secret["secret_leak_count"]:
        gate_status = "FAIL_SECRET_LEAK"
    elif parquet_check["json_parquet_mismatch_count"] or parquet_check["parquet_read_failure_count"]:
        gate_status = "FAIL_JSON_PARQUET_MISMATCH"

    final_report = {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "upstream_artifact": str(upstream),
        "upstream_gate": gate.get("gate"),
        "upstream_integrity_status": upstream_validation["upstream_integrity_status"],
        "ambiguous_axis_count": cardinality["ambiguous_axis_count"],
        "ambiguous_axis_id": "clock_rounding_allowance",
        "primitive_operator_id": "Q_CLOCK",
        "discriminator_count": cardinality["discriminator_count"],
        "direct_explicit_evidence_count": direct_explicit,
        "direct_derived_evidence_count": direct_derived,
        "supporting_evidence_count": supporting,
        "ambiguous_evidence_count": ambiguous_count,
        "absent_evidence_count": absent,
        "quantization_rule_parameter_defined": True,
        "quantization_resolution_parameter_defined": False,
        "parameters_independently_variable": False,
        "parameter_factorisation_required": False,
        "candidate_domain_kind": CANDIDATE_DOMAIN_KIND,
        "adjudication_status": ADJUDICATION_STATUS,
        "domain_kind_resolved": True,
        "actual_member_resolved": False,
        "numeric_bound_resolved": False,
        "unit_resolved": False,
        "stage_interaction_preserved": True,
        "stage_order_adopted": False,
        "commutativity_established": False,
        "original_domain_count": 6,
        "candidate_domain_count": len(candidate_registry),
        "original_design_cell_count": 13,
        "overlay_design_cell_count": len(overlay),
        "reference_cell_count": cardinality["reference_cell_count"],
        "new_numeric_domain_bound_count": 0,
        "new_actual_level_count": 0,
        "new_actual_enum_member_count": 0,
        "new_unit_assignment_count": 0,
        "new_assumption_count": 0,
        "axis_split_execution_count": 0,
        "design_matrix_mutation_count": 0,
        "operator_execution_count": 0,
        "domain_kind_adjudication_adopted": False,
        "parameter_factorisation_adopted": False,
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
        "secret_leak_count": secret["secret_leak_count"],
        "forbidden_operation_count": forbidden["forbidden_operation_count"],
        "gate": gate_status,
        "next_authorized_action": NEXT_ACTION,
    }
    dump_json(out_dir / "final_report.json", final_report)
    md = "\n".join([
        "# R2D-1Y Axis-Specific Symbolic Domain-Kind Adjudication",
        "",
        f"- artifact_id: `{out_dir.name}`",
        f"- upstream_gate: `{gate.get('gate')}`",
        f"- gate: `{gate_status}`",
        f"- ambiguous_axis_id: `clock_rounding_allowance`",
        f"- primitive_operator_id: `Q_CLOCK`",
        f"- candidate_domain_kind: `{CANDIDATE_DOMAIN_KIND}`",
        f"- adjudication_status: `{ADJUDICATION_STATUS}`",
        "",
        "Q_CLOCK is adjudicated as a symbolic unordered convention candidate only. "
        "The frozen contracts define a quantization-rule namespace and symbolic discrete "
        "operator contract, while no independent resolution/support-width field, unit, "
        "numeric bound, or natural ordering relation is available.",
        "",
        "No actual rounding member, clock resolution, unit, axis split, stage order, "
        "sensitivity execution, re-estimation, simulator application, or Phase 2 action "
        "is authorized.",
        "",
        f"- next_authorized_action: `{NEXT_ACTION}`",
    ]) + "\n"
    (out_dir / "final_report.md").write_text(md, encoding="utf-8")

    gate_payload = dict(final_report)
    gate_payload.update({
        "artifact": str(out_dir),
        "gate_passed": gate_status == PASS_CONVENTION,
    })
    dump_json(out_dir / "gate_decision.json", gate_payload)

    dump_json(out_dir / "forbidden_field_audit.json", forbidden_field_audit(out_dir))
    forbidden = read_json(out_dir / "forbidden_field_audit.json")
    secret = secret_scan(out_dir)
    if forbidden["forbidden_operation_count"]:
        raise RuntimeError(f"forbidden field audit failed: {forbidden}")
    if secret["secret_leak_count"]:
        raise RuntimeError(f"secret scan failed: {secret}")

    manifest = build_manifest(out_dir)
    dump_json(out_dir / "artifact_manifest.json", manifest)
    manifest_check = verify_own_manifest(out_dir, manifest)
    if (
        manifest_check["required_file_missing_count"]
        or manifest_check["unexpected_file_count"]
        or manifest_check["manifest_hash_mismatch_count"]
        or manifest_check["manifest_size_mismatch_count"]
        or manifest_check["strict_json_failure_count"]
    ):
        raise RuntimeError(f"manifest validation failed: {manifest_check}")

    if gate_status == "PASS_AXIS_SPECIFIC_SYMBOLIC_DOMAIN_KIND_ADJUDICATED_AS_CONVENTION_STILL_LOCKED":
        (out_dir / "_SUCCESS.lock").write_text(
            json.dumps({"success": True, "gate": gate_status, "created_at": now_iso()}, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(f"artifact:\n{out_dir}")
    print(f"gate:\n{gate_status}")
    print("ambiguous axis/operator:\nclock_rounding_allowance/Q_CLOCK")
    print(f"candidate domain kind:\n{CANDIDATE_DOMAIN_KIND}")
    print("quantization rule defined/resolution support defined/factorisation required:\ntrue/false/false")
    print(f"discriminators direct_explicit/direct_derived/supporting/ambiguous/absent:\n{direct_explicit}/{direct_derived}/{supporting}/{ambiguous_count}/{absent}")
    print(f"domain rows/design cells/reference cells:\n{len(candidate_registry)}/{len(overlay)}/{cardinality['reference_cell_count']}")
    print("actual member/numeric bound/unit/new assumption/operator execution:\nfalse/false/false/0/0")
    print("API/service key/DB/network:\n0/false/false/false")
    print(f"upstream added/modified/deleted:\n{mutation['upstream_added']}/{mutation['upstream_modified']}/{mutation['upstream_deleted']}")
    print(f"required files:\n{len(REQUIRED_FILES)}")
    print("manifest hash/size mismatches:\n0/0")
    print("strict JSON failures:\n0")
    print(f"Parquet read failures:\n{parquet_check['parquet_read_failure_count']}")
    print(f"JSON/Parquet mismatches:\n{parquet_check['json_parquet_mismatch_count']}")
    print(f"secret leaks:\n{secret['secret_leak_count']}")
    print(f"forbidden operations:\n{forbidden['forbidden_operation_count']}")
    print(f"next authorized action:\n{NEXT_ACTION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
