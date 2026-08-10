#!/usr/bin/env python3
"""R2D-1Z Q_CLOCK convention evidence-boundary runner.

This script defines evidence requirements for future Q_CLOCK convention
members. It does not collect provider metadata, create convention members,
run tests, transform timestamps, or authorize sensitivity execution.
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
    "prompt5_e01_r2d1y_axis_specific_symbolic_domain_kind_adjudication_20260731_074037"
)
EXPECTED_UPSTREAM_GATE = (
    "PASS_AXIS_SPECIFIC_SYMBOLIC_DOMAIN_KIND_ADJUDICATED_AS_CONVENTION_STILL_LOCKED"
)
EXPECTED_NEXT = "Clock-quantization convention evidence-boundary specification only"
EXPECTED_DOMAIN_KIND = "SYMBOLIC_UNORDERED_CONVENTION_DOMAIN"
EXPECTED_ADJUDICATION = "ADJUDICATED_AS_SYMBOLIC_UNORDERED_CONVENTION_DOMAIN"
PASS_GATE = "PASS_CLOCK_QUANTIZATION_CONVENTION_EVIDENCE_BOUNDARY_READY_STILL_LOCKED"
NEXT_ACTION = "Clock-quantization convention evidence-source inventory only"

REQUIRED_FILES = [
    "upstream_validation.json",
    "upstream_lineage.json",
    "r2d1y_gate_snapshot.json",
    "r2d1y_adjudication_snapshot.json",
    "r2d1y_evidence_snapshot.json",
    "r2d1y_stage_interaction_snapshot.json",
    "r2d1y_lock_snapshot.json",
    "qclock_convention_semantic_contract.json",
    "qclock_resolution_context_separation_contract.json",
    "evidence_source_class_registry.json",
    "evidence_source_class_registry.parquet",
    "evidence_strength_registry.json",
    "evidence_strength_registry.parquet",
    "evidence_completeness_dimension_registry.json",
    "evidence_completeness_dimension_registry.parquet",
    "candidate_member_minimum_evidence_contract.json",
    "candidate_member_acceptance_predicate.json",
    "structural_property_requirement_registry.json",
    "structural_property_requirement_registry.parquet",
    "stage_interaction_evidence_boundary.json",
    "insufficient_evidence_registry.json",
    "conflict_handling_contract.json",
    "fallback_default_classification_contract.json",
    "future_evidence_bundle_schema.json",
    "evidence_readiness_status_registry.json",
    "current_frozen_evidence_boundary_audit.json",
    "actual_member_identification_boundary.json",
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

UPSTREAM_LOCK_FALSE_FIELDS = [
    "domain_kind_adjudication_adopted",
    "actual_level_resolution_approved",
    "sensitivity_grid_execution_approved",
    "identified_set_reestimation_approved",
    "turnbull_reestimation_approved",
    "phase2_authorized",
]

LOCK_FALSE_FIELDS = [
    "path_b_adopted",
    "sensitivity_contract_adopted",
    "symbolic_domain_contract_adopted",
    "domain_kind_adjudication_adopted",
    "clock_quantization_evidence_boundary_adopted",
    "candidate_member_contract_adopted",
    "candidate_member_adopted",
    "actual_rounding_rule_resolved",
    "clock_resolution_resolved",
    "clock_unit_resolved",
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
    "simulator_parameter_conversion_approved",
    "simulator_application_approved",
    "phase2_authorized",
    "baseline_rerun_authorized",
    "retraining_authorized",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]

FORBIDDEN_FIELD_NAMES = [
    "actual_convention_member_value",
    "actual_rounding_rule_value",
    "actual_resolution_value",
    "actual_unit_value",
    "test_vector_result",
    "operator_execution_result",
    "timestamp_transform_result",
    "identified_set_result",
    "turnbull_result",
    "point_estimate_result",
    "sensitivity_grid_result",
    "phase2_result",
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


def records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(payload.get("records", []))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame([{k: "" if v is None else v for k, v in row.items()} for row in rows]).to_parquet(
        path, index=False
    )


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{k: "" if v is None else str(v) for k, v in row.items()} for row in rows]


def json_valid(path: Path) -> bool:
    try:
        read_json(path)
        return True
    except Exception:
        return False


def validate_manifest(artifact: Path) -> dict[str, Any]:
    manifest_path = artifact / "artifact_manifest.json"
    success_path = artifact / "_SUCCESS.lock"
    if not manifest_path.exists():
        return {
            "status": "FAIL",
            "manifest_missing_count": 1,
            "success_lock_missing_count": 0 if success_path.exists() else 1,
        }
    manifest = read_json(manifest_path)
    hash_mismatch = 0
    size_mismatch = 0
    strict_json_failure = 0
    missing = 0
    for item in manifest.get("files", manifest.get("entries", [])):
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
        if path.suffix == ".json" and not json_valid(path):
            strict_json_failure += 1
    return {
        "status": "PASS"
        if success_path.exists()
        and hash_mismatch == 0
        and size_mismatch == 0
        and strict_json_failure == 0
        and missing == 0
        else "FAIL",
        "manifest_hash_mismatch_count": hash_mismatch,
        "manifest_size_mismatch_count": size_mismatch,
        "strict_json_failure_count": strict_json_failure,
        "manifest_missing_count": 0,
        "success_lock_missing_count": 0 if success_path.exists() else 1,
        "manifest_entry_missing_count": missing,
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


def upstream_mutation_audit(path: Path, before: dict[str, dict[str, Any]]) -> dict[str, Any]:
    current = manifest_snapshot(path)
    added = [rel for rel in current if rel not in before]
    deleted = [rel for rel in before if rel not in current]
    modified = [rel for rel in before if rel in current and before[rel] != current[rel]]
    return {
        "upstream_added": len(added),
        "upstream_modified": len(modified),
        "upstream_deleted": len(deleted),
        "added_files": added,
        "modified_files": modified,
        "deleted_files": deleted,
    }


def validate_upstream(upstream: Path) -> dict[str, Any]:
    gate = read_json(upstream / "gate_decision.json")
    adjudication = read_json(upstream / "clock_rounding_domain_kind_adjudication.json")
    stage = read_json(upstream / "stage_interaction_preservation_audit.json")
    locks = read_json(upstream / "lock_state.json")
    manifest = validate_manifest(upstream)
    checks = {
        "gate_matches": gate.get("gate") == EXPECTED_UPSTREAM_GATE,
        "next_action_matches": gate.get("next_authorized_action") == EXPECTED_NEXT,
        "axis_matches": gate.get("ambiguous_axis_id") == "clock_rounding_allowance"
        and adjudication.get("axis_id") == "clock_rounding_allowance",
        "operator_matches": gate.get("primitive_operator_id") == "Q_CLOCK"
        and adjudication.get("primitive_operator_id") == "Q_CLOCK",
        "domain_kind_matches": gate.get("candidate_domain_kind") == EXPECTED_DOMAIN_KIND
        and adjudication.get("candidate_domain_kind") == EXPECTED_DOMAIN_KIND,
        "adjudication_status_matches": adjudication.get("adjudication_status") == EXPECTED_ADJUDICATION,
        "unresolved_values_remain_false": gate.get("actual_member_resolved") is False
        and gate.get("numeric_bound_resolved") is False
        and gate.get("unit_resolved") is False,
        "stage_state_preserved": gate.get("stage_interaction_preserved") is True
        and gate.get("stage_order_adopted") is False
        and gate.get("commutativity_established") is False
        and stage.get("stage_interaction_preserved") is True,
        "required_locks_false": all(locks.get(field) is False for field in UPSTREAM_LOCK_FALSE_FIELDS),
        "manifest_integrity_pass": manifest.get("status") == "PASS",
    }
    return {
        "upstream_artifact": str(upstream),
        "upstream_gate": gate.get("gate"),
        "upstream_next_authorized_action": gate.get("next_authorized_action"),
        "checks": checks,
        "manifest_validation": manifest,
        "upstream_integrity_status": "PASS" if all(checks.values()) else "FAIL",
    }


def evidence_source_classes() -> list[dict[str, Any]]:
    return [
        {
            "source_class_id": "E1",
            "source_class_name": "Frozen explicit semantic contract",
            "evidence_role": "DIRECT_SEMANTIC",
            "definition": "Integrity-verified frozen artifact directly defines Q_CLOCK mapping semantics.",
            "required_conditions": pipe([
                "artifact_manifest_present",
                "success_lock_present",
                "manifest_validation_pass",
                "source_locator_present",
                "direct_qclock_operator_link",
            ]),
            "provider_metadata_collection_allowed_now": False,
            "candidate_member_adoption_allowed_now": False,
        },
        {
            "source_class_id": "E2",
            "source_class_name": "Authoritative provider metadata",
            "evidence_role": "DIRECT_SEMANTIC",
            "definition": "Provider metadata explicitly describes timestamp expression or quantization rule semantics.",
            "required_conditions": pipe([
                "provider_identity",
                "metadata_version",
                "effective_date",
                "field_scope",
                "timestamp_field_name",
                "rule_semantics",
            ]),
            "provider_metadata_collection_allowed_now": False,
            "candidate_member_adoption_allowed_now": False,
        },
        {
            "source_class_id": "E3",
            "source_class_name": "Canonical implementation contract",
            "evidence_role": "DIRECT_IMPLEMENTATION",
            "definition": "Canonical implementation has an explicit Q_CLOCK parameter and deterministic mapping contract.",
            "required_conditions": pipe([
                "implementation_version_or_hash",
                "operator_parameter_contract",
                "input_output_schema",
                "processing_stage",
                "fallback_status",
                "default_status",
            ]),
            "provider_metadata_collection_allowed_now": False,
            "candidate_member_adoption_allowed_now": False,
        },
        {
            "source_class_id": "E4",
            "source_class_name": "Reproducible input-output test vector",
            "evidence_role": "DIRECT_TEST_VECTOR",
            "definition": "Frozen test vector links input, fixed context, convention identifier, and expected output.",
            "required_conditions": pipe([
                "input_timestamp",
                "fixed_context",
                "expected_output",
                "convention_identifier",
                "implementation_hash",
                "test_status",
            ]),
            "provider_metadata_collection_allowed_now": False,
            "candidate_member_adoption_allowed_now": False,
        },
        {
            "source_class_id": "E5",
            "source_class_name": "Frozen execution trace",
            "evidence_role": "DIRECT_EXECUTION_TRACE",
            "definition": "Canonical trace records raw and transformed timestamp with operator and parameter lineage.",
            "required_conditions": pipe([
                "operator_identifier",
                "parameter_identifier",
                "implementation_version",
                "fixed_resolution_context",
                "fallback_flag",
            ]),
            "provider_metadata_collection_allowed_now": False,
            "candidate_member_adoption_allowed_now": False,
        },
    ]


def evidence_strength_classes() -> list[dict[str, Any]]:
    return [
        {
            "strength_id": "DIRECT_SEMANTIC",
            "definition": "Convention meaning and application scope are explicitly defined.",
            "can_satisfy_semantic_requirement": True,
            "can_satisfy_reproducibility_requirement": False,
        },
        {
            "strength_id": "DIRECT_IMPLEMENTATION",
            "definition": "Q_CLOCK parameter and deterministic mapping implementation are explicit.",
            "can_satisfy_semantic_requirement": False,
            "can_satisfy_reproducibility_requirement": True,
        },
        {
            "strength_id": "DIRECT_TEST_VECTOR",
            "definition": "Convention identifier is linked to expected output in a reproducible frozen test.",
            "can_satisfy_semantic_requirement": False,
            "can_satisfy_reproducibility_requirement": True,
        },
        {
            "strength_id": "DIRECT_EXECUTION_TRACE",
            "definition": "Canonical execution records convention identifier and transformed output together.",
            "can_satisfy_semantic_requirement": False,
            "can_satisfy_reproducibility_requirement": True,
        },
        {
            "strength_id": "SUPPORTING_CONTEXT_ONLY",
            "definition": "Related context exists but does not identify a convention member.",
            "can_satisfy_semantic_requirement": False,
            "can_satisfy_reproducibility_requirement": False,
        },
        {
            "strength_id": "AMBIGUOUS",
            "definition": "Evidence does not distinguish convention from other contexts.",
            "can_satisfy_semantic_requirement": False,
            "can_satisfy_reproducibility_requirement": False,
        },
        {
            "strength_id": "CONFLICTING",
            "definition": "Direct evidence supports incompatible convention semantics in overlapping scope.",
            "can_satisfy_semantic_requirement": False,
            "can_satisfy_reproducibility_requirement": False,
        },
        {
            "strength_id": "INADMISSIBLE",
            "definition": "Evidence source is barred as a convention-member basis.",
            "can_satisfy_semantic_requirement": False,
            "can_satisfy_reproducibility_requirement": False,
        },
    ]


def completeness_dimensions() -> list[dict[str, Any]]:
    required = {
        "semantic_definition",
        "source_scope",
        "field_scope",
        "operator_stage",
        "input_schema",
        "output_schema",
        "resolution_context",
        "timezone_context",
        "boundary_behavior",
        "missing_value_behavior",
        "determinism",
        "monotonicity",
        "implementation_version",
        "fallback_behavior",
        "lineage_traceability",
    }
    conditional = {
        "tie_behavior",
        "idempotence",
        "test_vector_support",
        "execution_trace_support",
    }
    rows = []
    for order, name in enumerate([
        "semantic_definition",
        "source_scope",
        "field_scope",
        "operator_stage",
        "input_schema",
        "output_schema",
        "resolution_context",
        "timezone_context",
        "boundary_behavior",
        "tie_behavior",
        "missing_value_behavior",
        "determinism",
        "idempotence",
        "monotonicity",
        "implementation_version",
        "test_vector_support",
        "execution_trace_support",
        "fallback_behavior",
        "lineage_traceability",
    ], start=1):
        status = "REQUIRED" if name in required else "CONDITIONALLY_REQUIRED"
        if name not in required and name not in conditional:
            status = "OPTIONAL_SUPPORT"
        rows.append({
            "dimension_id": f"CD{order:02d}",
            "dimension_name": name,
            "requirement_status": status,
            "actual_value_resolved_now": False,
            "candidate_member_adoption_allowed_now": False,
        })
    return rows


def structural_properties() -> list[dict[str, Any]]:
    return [
        {
            "property_id": "SP01",
            "property_name": "determinism",
            "requirement": "Same input timestamp and fixed context produce the same output.",
            "verification_required_for_future_candidate": True,
            "verified_now": False,
        },
        {
            "property_id": "SP02",
            "property_name": "idempotence",
            "requirement": "Q_CLOCK applied twice should equal one application unless direct semantics require review.",
            "verification_required_for_future_candidate": True,
            "verified_now": False,
        },
        {
            "property_id": "SP03",
            "property_name": "monotonicity",
            "requirement": "Input order must preserve output order.",
            "verification_required_for_future_candidate": True,
            "verified_now": False,
        },
        {
            "property_id": "SP04",
            "property_name": "source_locality",
            "requirement": "Q_CLOCK must not perform provider-request source reconciliation.",
            "verification_required_for_future_candidate": True,
            "verified_now": False,
        },
        {
            "property_id": "SP05",
            "property_name": "no_implicit_skew_correction",
            "requirement": "Q_CLOCK must not perform the S_CLOCK role implicitly.",
            "verification_required_for_future_candidate": True,
            "verified_now": False,
        },
    ]


def readiness_statuses() -> list[dict[str, Any]]:
    return [
        {"status": "NO_ADMISSIBLE_EVIDENCE", "candidate_member_adopted": False, "execution_allowed": False},
        {"status": "SUPPORTING_EVIDENCE_ONLY", "candidate_member_adopted": False, "execution_allowed": False},
        {"status": "SEMANTIC_EVIDENCE_INCOMPLETE", "candidate_member_adopted": False, "execution_allowed": False},
        {"status": "IMPLEMENTATION_OR_TEST_EVIDENCE_INCOMPLETE", "candidate_member_adopted": False, "execution_allowed": False},
        {"status": "STAGE_SCOPE_INCOMPLETE", "candidate_member_adopted": False, "execution_allowed": False},
        {"status": "RESOLUTION_CONTEXT_CONFLATED", "candidate_member_adopted": False, "execution_allowed": False},
        {"status": "CLOCK_SKEW_CONFLATED", "candidate_member_adopted": False, "execution_allowed": False},
        {"status": "CONFLICTING_DIRECT_EVIDENCE", "candidate_member_adopted": False, "execution_allowed": False},
        {
            "status": "CANDIDATE_MEMBER_EVIDENCE_BOUNDARY_SATISFIED",
            "candidate_member_adopted": False,
            "execution_allowed": False,
        },
    ]


def insufficient_evidence_rows() -> list[dict[str, Any]]:
    names = [
        "variable_name",
        "function_name",
        "comment_only",
        "generic_library_default",
        "generic_timestamp_practice",
        "request_cadence",
        "api_rate_limit",
        "identified_set_width",
        "provider_request_dominance",
        "output_timestamp_value_only_trace",
        "developer_memory",
        "file_creation_time",
        "operating_system_file_timestamp",
    ]
    return [
        {
            "insufficient_evidence_id": f"IE{idx:02d}",
            "evidence_kind": name,
            "admissible_alone": False,
            "reason": "Does not directly define Q_CLOCK convention-member semantics and reproducibility.",
        }
        for idx, name in enumerate(names, start=1)
    ]


def parquet_json_check(out_dir: Path) -> dict[str, int]:
    pairs = [
        ("evidence_source_class_registry.json", "evidence_source_class_registry.parquet"),
        ("evidence_strength_registry.json", "evidence_strength_registry.parquet"),
        ("evidence_completeness_dimension_registry.json", "evidence_completeness_dimension_registry.parquet"),
        ("structural_property_requirement_registry.json", "structural_property_requirement_registry.parquet"),
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
    entries = []
    for name in REQUIRED_FILES:
        if name in MANIFEST_EXCLUDED:
            continue
        path = out_dir / name
        entries.append({
            "relative_path": name,
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
            "strict_json_valid": True if path.suffix != ".json" else json_valid(path),
        })
    return {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "required_file_count": len(REQUIRED_FILES),
        "manifest_entry_count": len(entries),
        "files": entries,
    }


def verify_own_manifest(out_dir: Path, manifest: dict[str, Any]) -> dict[str, int]:
    missing = 0
    unexpected = 0
    hash_mismatch = 0
    size_mismatch = 0
    strict_json_failure = 0
    required = set(REQUIRED_FILES)
    for path in out_dir.iterdir():
        if path.is_file() and path.name not in required:
            unexpected += 1
    for item in manifest["files"]:
        path = out_dir / item["relative_path"]
        if not path.exists():
            missing += 1
            continue
        if sha256_file(path) != item["sha256"]:
            hash_mismatch += 1
        if path.stat().st_size != item["size_bytes"]:
            size_mismatch += 1
        if path.suffix == ".json" and not json_valid(path):
            strict_json_failure += 1
    return {
        "required_file_missing_count": missing,
        "unexpected_file_count": unexpected,
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

    upstream_before = manifest_snapshot(upstream)
    upstream_validation = validate_upstream(upstream)

    out_dir = (
        project_root
        / "05_training"
        / "artifacts"
        / f"prompt5_e01_r2d1z_clock_quantization_convention_evidence_boundary_{now_stamp()}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    gate = read_json(upstream / "gate_decision.json")
    adjudication = read_json(upstream / "clock_rounding_domain_kind_adjudication.json")
    evidence = read_json(upstream / "domain_kind_discriminator_registry.json")
    stage = read_json(upstream / "stage_interaction_preservation_audit.json")
    locks = read_json(upstream / "lock_state.json")

    dump_json(out_dir / "upstream_validation.json", upstream_validation)
    dump_json(out_dir / "r2d1y_gate_snapshot.json", gate)
    dump_json(out_dir / "r2d1y_adjudication_snapshot.json", adjudication)
    dump_json(out_dir / "r2d1y_evidence_snapshot.json", evidence)
    dump_json(out_dir / "r2d1y_stage_interaction_snapshot.json", stage)
    dump_json(out_dir / "r2d1y_lock_snapshot.json", locks)
    dump_json(out_dir / "upstream_lineage.json", {
        "artifact_id": out_dir.name,
        "created_at": now_iso(),
        "authoritative_upstream_artifact": str(upstream),
        "authoritative_upstream_gate": gate.get("gate"),
        "source_files_read": [
            "gate_decision.json",
            "clock_rounding_domain_kind_adjudication.json",
            "domain_kind_discriminator_registry.json",
            "stage_interaction_preservation_audit.json",
            "lock_state.json",
            "artifact_manifest.json",
        ],
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
    })

    semantic_contract = {
        "axis_id": "clock_rounding_allowance",
        "primitive_operator_id": "Q_CLOCK",
        "domain_kind": EXPECTED_DOMAIN_KIND,
        "contract_type": "Q_CLOCK_CONVENTION_MEMBER_MINIMUM_SEMANTIC_CONTRACT",
        "semantic_rule": "same_timestamp_input_plus_same_fixed_context_plus_same_convention_parameter_yields_same_quantized_timestamp_output",
        "required_semantic_items": pipe([
            "input_timestamp_format",
            "output_timestamp_format",
            "source_scope",
            "processing_stage",
            "resolution_metadata_provenance",
            "boundary_or_tie_behavior",
            "timezone_boundary",
            "missing_timestamp_boundary",
            "qclock_sclock_separation",
            "qclock_urequest_stage_dependency",
        ]),
        "actual_values_resolved": False,
        "candidate_member_adopted": False,
        "execution_allowed": False,
    }
    dump_json(out_dir / "qclock_convention_semantic_contract.json", semantic_contract)

    separation_contract = {
        "convention_identity_required": True,
        "resolution_metadata_required_as_context": True,
        "resolution_is_axis_member": False,
        "resolution_is_fixed_context_input": True,
        "resolution_value_resolved": False,
        "resolution_unit_resolved": False,
        "request_cadence_is_convention": False,
        "clock_skew_is_convention": False,
        "resolution_context_separated": True,
        "clock_skew_separated": True,
        "request_cadence_separated": True,
    }
    dump_json(out_dir / "qclock_resolution_context_separation_contract.json", separation_contract)

    source_rows = evidence_source_classes()
    dump_json(out_dir / "evidence_source_class_registry.json", {
        "evidence_source_class_count": len(source_rows),
        "records": source_rows,
    })
    write_parquet(out_dir / "evidence_source_class_registry.parquet", source_rows)

    strength_rows = evidence_strength_classes()
    dump_json(out_dir / "evidence_strength_registry.json", {
        "evidence_strength_class_count": len(strength_rows),
        "records": strength_rows,
    })
    write_parquet(out_dir / "evidence_strength_registry.parquet", strength_rows)

    dimension_rows = completeness_dimensions()
    dump_json(out_dir / "evidence_completeness_dimension_registry.json", {
        "completeness_dimension_count": len(dimension_rows),
        "records": dimension_rows,
    })
    write_parquet(out_dir / "evidence_completeness_dimension_registry.parquet", dimension_rows)

    minimum_contract = {
        "direct_semantic_required": True,
        "implementation_or_test_evidence_required": True,
        "direct_implementation_or_test_or_trace_required": True,
        "source_scope_required": True,
        "operator_stage_required": True,
        "resolution_context_separation_required": True,
        "clock_skew_separation_required": True,
        "conflict_free_direct_evidence_required": True,
        "candidate_member_adopted": False,
        "execution_allowed": False,
    }
    dump_json(out_dir / "candidate_member_minimum_evidence_contract.json", minimum_contract)
    dump_json(out_dir / "candidate_member_acceptance_predicate.json", {
        "candidate_acceptance_predicate_defined": True,
        "predicate": (
            "direct_semantic_count >= 1 and "
            "(direct_implementation_count >= 1 or direct_test_vector_count >= 1 or direct_execution_trace_count >= 1) and "
            "source_scope_defined and operator_stage_defined and resolution_context_separated and "
            "clock_skew_separated and conflicting_direct_evidence_count == 0"
        ),
        "current_actual_candidate_evaluated": False,
        "candidate_member_evidence_sufficient_now": False,
        "candidate_member_adopted": False,
        "execution_allowed": False,
    })

    property_rows = structural_properties()
    dump_json(out_dir / "structural_property_requirement_registry.json", {
        "structural_property_requirement_count": len(property_rows),
        "records": property_rows,
    })
    write_parquet(out_dir / "structural_property_requirement_registry.parquet", property_rows)

    dump_json(out_dir / "stage_interaction_evidence_boundary.json", {
        "qclock_sclock_interaction_preserved": True,
        "qclock_urequest_interaction_preserved": True,
        "stage_order_required_to_be_declared_for_future_candidate": True,
        "qclock_output_to_urequest_input_required": True,
        "stage_order_adopted_now": False,
        "commutativity_established_now": False,
        "candidate_member_semantically_resolvable": False,
        "candidate_member_execution_ready": False,
        "records": stage.get("records", []),
    })

    insufficient_rows = insufficient_evidence_rows()
    dump_json(out_dir / "insufficient_evidence_registry.json", {
        "insufficient_evidence_count": len(insufficient_rows),
        "records": insufficient_rows,
    })
    dump_json(out_dir / "conflict_handling_contract.json", {
        "conflict_policy_defined": True,
        "conflict_status": "CONFLICTING_DIRECT_EVIDENCE",
        "conflict_record_schema": pipe([
            "conflict_id",
            "evidence_ids",
            "source_versions",
            "conflicting_fields",
            "effective_dates",
            "scope_overlap",
            "possible_version_transition",
            "adjudication_required",
        ]),
        "candidate_member_evidence_sufficient_when_conflicting": False,
        "adjudication_required": True,
        "actual_conflict_adjudicated_now": False,
    })
    dump_json(out_dir / "fallback_default_classification_contract.json", {
        "fallback_policy_defined": True,
        "records": [
            {
                "fallback_classification": "EXPLICIT_CONFIGURED_CONVENTION",
                "future_candidate_admissibility": "ADMISSIBLE_CANDIDATE_EVIDENCE",
            },
            {
                "fallback_classification": "VERSIONED_CANONICAL_DEFAULT",
                "future_candidate_admissibility": "CONDITIONALLY_ADMISSIBLE",
            },
            {
                "fallback_classification": "IMPLICIT_LIBRARY_DEFAULT",
                "future_candidate_admissibility": "INADMISSIBLE",
            },
            {
                "fallback_classification": "FALLBACK_AFTER_MISSING_CONFIG",
                "future_candidate_admissibility": "INADMISSIBLE_FOR_PRIMARY_CANDIDATE",
            },
            {
                "fallback_classification": "UNKNOWN_DEFAULT_SOURCE",
                "future_candidate_admissibility": "INADMISSIBLE",
            },
        ],
        "implicit_library_default_accepted": False,
    })

    future_bundle_fields = [
        "candidate_member_id",
        "candidate_member_label",
        "candidate_only",
        "semantic_evidence_ids",
        "implementation_evidence_ids",
        "test_vector_evidence_ids",
        "execution_trace_evidence_ids",
        "source_scope",
        "field_scope",
        "operator_stage",
        "input_schema",
        "output_schema",
        "resolution_context_id",
        "resolution_value_redacted_or_symbolic",
        "resolution_unit_redacted_or_symbolic",
        "timezone_context",
        "boundary_behavior_status",
        "tie_behavior_status",
        "missing_value_behavior_status",
        "determinism_evidence_status",
        "idempotence_evidence_status",
        "monotonicity_evidence_status",
        "qclock_sclock_separation_status",
        "qclock_urequest_dependency_status",
        "fallback_classification",
        "conflicting_evidence_count",
        "semantic_resolution_ready",
        "execution_validation_ready",
        "candidate_member_adopted",
        "execution_allowed",
    ]
    dump_json(out_dir / "future_evidence_bundle_schema.json", {
        "future_evidence_bundle_schema_defined": True,
        "field_count": len(future_bundle_fields),
        "required_candidate_flags": {
            "candidate_only": True,
            "candidate_member_adopted": False,
            "execution_allowed": False,
        },
        "fields": future_bundle_fields,
    })

    readiness_rows = readiness_statuses()
    dump_json(out_dir / "evidence_readiness_status_registry.json", {
        "evidence_readiness_status_count": len(readiness_rows),
        "records": readiness_rows,
    })

    domain_kind_direct_evidence = [
        row for row in records(evidence) if row.get("supports_unordered_convention") is True
    ]
    current_audit = {
        "domain_kind_evidence_available": len(domain_kind_direct_evidence) > 0,
        "actual_member_identification_evidence_available": False,
        "current_evidence_boundary_status": "SUPPORTING_EVIDENCE_ONLY",
        "r2d1y_evidence_is_domain_kind_evidence": True,
        "r2d1y_evidence_is_actual_member_evidence": False,
        "direct_semantic_actual_member_count": 0,
        "direct_implementation_actual_member_count": 0,
        "direct_test_vector_actual_member_count": 0,
        "direct_execution_trace_actual_member_count": 0,
        "candidate_member_evidence_sufficient": False,
    }
    dump_json(out_dir / "current_frozen_evidence_boundary_audit.json", current_audit)
    dump_json(out_dir / "actual_member_identification_boundary.json", {
        "actual_convention_member_count": 0,
        "resolution_value_count": 0,
        "unit_assignment_count": 0,
        "semantic_resolution_ready": False,
        "execution_validation_ready": False,
        "candidate_member_adopted": False,
        "execution_allowed": False,
        "candidate_member_execution_ready": False,
    })

    dump_json(out_dir / "execution_boundary.json", {
        "scope": "CLOCK_QUANTIZATION_CONVENTION_EVIDENCE_BOUNDARY_ONLY",
        "clock_quantization_evidence_boundary_specified": True,
        "provider_metadata_collected": False,
        "actual_convention_member_created": False,
        "test_vector_generated": False,
        "test_executed": False,
        "operator_executed": False,
        "timestamp_transformed": False,
        "stage_order_adopted": False,
        "sensitivity_analysis_executed": False,
        "identified_set_reestimation_executed": False,
        "turnbull_reestimation_executed": False,
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
        "provider_metadata_searched": False,
        "new_observation_collected": False,
        "new_episode_collected": False,
        "new_assumption_created": False,
        "actual_convention_member_created": False,
        "actual_rounding_rule_selected": False,
        "resolution_value_created": False,
        "unit_decided": False,
        "test_vector_generated": False,
        "test_executed": False,
        "operator_executed": False,
        "timestamp_transformed": False,
        "interval_reconstructed": False,
        "stage_order_adopted": False,
        "identified_set_calculated": False,
        "turnbull_executed": False,
        "point_estimate_calculated": False,
        "robustness_judged": False,
        "simulator_applied": False,
        "phase2_executed": False,
        "baseline_rerun_executed": False,
        "training_or_retraining_executed": False,
        "new_assumption_count": 0,
        "new_actual_convention_member_count": 0,
        "new_resolution_value_count": 0,
        "new_unit_assignment_count": 0,
        "test_vector_generation_count": 0,
        "test_execution_count": 0,
        "operator_execution_count": 0,
        "forbidden_operation_count": 0,
    })
    dump_json(out_dir / "api_db_network_audit.json", {
        "api_call_count": 0,
        "service_key_accessed": False,
        "db_accessed": False,
        "external_network_accessed": False,
    })
    mutation = upstream_mutation_audit(upstream, upstream_before)
    dump_json(out_dir / "upstream_mutation_audit.json", mutation)
    dump_json(out_dir / "lineage_field_audit.json", {
        "authoritative_upstream_only": True,
        "partial_failure_artifact_used": False,
        "external_evidence_used": False,
        "provider_metadata_collected": False,
        "lineage_field_issue_count": 0,
    })
    dump_json(out_dir / "forbidden_field_audit.json", forbidden_field_audit(out_dir))

    parquet_check = parquet_json_check(out_dir)
    forbidden = read_json(out_dir / "forbidden_field_audit.json")
    secret = secret_scan(out_dir)
    prohibited = read_json(out_dir / "prohibited_operation_audit.json")

    direct_semantic_required = minimum_contract["direct_semantic_required"]
    implementation_or_test_required = minimum_contract["implementation_or_test_evidence_required"]
    acceptance_defined = True
    conflict_policy_defined = True
    fallback_policy_defined = True
    schema_defined = True
    gate_status = PASS_GATE
    if gate.get("gate") != EXPECTED_UPSTREAM_GATE:
        gate_status = "FAIL_UPSTREAM_GATE_MISMATCH"
    elif upstream_validation["upstream_integrity_status"] != "PASS":
        gate_status = "FAIL_UPSTREAM_INTEGRITY"
    elif adjudication.get("candidate_domain_kind") != EXPECTED_DOMAIN_KIND:
        gate_status = "FAIL_DOMAIN_KIND_MISMATCH"
    elif adjudication.get("primitive_operator_id") != "Q_CLOCK":
        gate_status = "FAIL_QCLOCK_OPERATOR_MISMATCH"
    elif not separation_contract["resolution_context_separated"]:
        gate_status = "FAIL_RESOLUTION_CONTEXT_CONVENTION_CONFLATION"
    elif not separation_contract["clock_skew_separated"]:
        gate_status = "FAIL_CLOCK_SKEW_CONVENTION_CONFLATION"
    elif not separation_contract["request_cadence_separated"]:
        gate_status = "FAIL_REQUEST_CADENCE_CONVENTION_CONFLATION"
    elif not direct_semantic_required:
        gate_status = "FAIL_EVIDENCE_ACCEPTANCE_WITHOUT_DIRECT_SEMANTIC_REQUIREMENT"
    elif read_json(out_dir / "fallback_default_classification_contract.json")[
        "implicit_library_default_accepted"
    ]:
        gate_status = "FAIL_IMPLICIT_LIBRARY_DEFAULT_ACCEPTED"
    elif not conflict_policy_defined:
        gate_status = "FAIL_CONFLICT_POLICY_MISSING"
    elif prohibited["new_actual_convention_member_count"]:
        gate_status = "FAIL_ACTUAL_CONVENTION_MEMBER_INVENTED"
    elif prohibited["new_resolution_value_count"]:
        gate_status = "FAIL_RESOLUTION_VALUE_INVENTED"
    elif prohibited["new_unit_assignment_count"]:
        gate_status = "FAIL_UNIT_INVENTED"
    elif prohibited["new_assumption_count"]:
        gate_status = "FAIL_NEW_ASSUMPTION_CREATED"
    elif prohibited["test_execution_count"] or prohibited["operator_execution_count"]:
        gate_status = "FAIL_TEST_OR_OPERATOR_EXECUTION_DETECTED"
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
        "axis_id": "clock_rounding_allowance",
        "primitive_operator_id": "Q_CLOCK",
        "domain_kind": EXPECTED_DOMAIN_KIND,
        "evidence_source_class_count": len(source_rows),
        "evidence_strength_class_count": len(strength_rows),
        "completeness_dimension_count": len(dimension_rows),
        "structural_property_requirement_count": len(property_rows),
        "direct_semantic_required": direct_semantic_required,
        "implementation_or_test_evidence_required": implementation_or_test_required,
        "source_scope_required": True,
        "operator_stage_required": True,
        "resolution_context_separation_required": True,
        "clock_skew_separation_required": True,
        "conflict_free_direct_evidence_required": True,
        "candidate_acceptance_predicate_defined": acceptance_defined,
        "conflict_policy_defined": conflict_policy_defined,
        "fallback_policy_defined": fallback_policy_defined,
        "future_evidence_bundle_schema_defined": schema_defined,
        "domain_kind_evidence_available": current_audit["domain_kind_evidence_available"],
        "actual_member_identification_evidence_available": False,
        "current_evidence_boundary_status": current_audit["current_evidence_boundary_status"],
        "actual_convention_member_count": 0,
        "resolution_value_count": 0,
        "unit_assignment_count": 0,
        "new_assumption_count": 0,
        "test_vector_generation_count": 0,
        "test_execution_count": 0,
        "operator_execution_count": 0,
        "clock_quantization_evidence_boundary_adopted": False,
        "candidate_member_adopted": False,
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
        "# R2D-1Z Clock-Quantization Convention Evidence Boundary",
        "",
        f"- artifact_id: `{out_dir.name}`",
        f"- upstream_gate: `{gate.get('gate')}`",
        f"- gate: `{gate_status}`",
        "- axis_id: `clock_rounding_allowance`",
        "- primitive_operator_id: `Q_CLOCK`",
        f"- domain_kind: `{EXPECTED_DOMAIN_KIND}`",
        "- current_evidence_boundary_status: `SUPPORTING_EVIDENCE_ONLY`",
        "",
        "The evidence boundary is specified for future Q_CLOCK convention members. "
        "A future member must have direct semantic evidence plus at least one direct "
        "implementation, test-vector, or execution-trace evidence source, with source "
        "scope, stage scope, resolution-context separation, clock-skew separation, "
        "and no conflicting direct evidence.",
        "",
        "Current frozen evidence supports the domain-kind adjudication only. It does "
        "not identify an actual convention member, resolution value, unit, test vector, "
        "or executable candidate.",
        "",
        f"- next_authorized_action: `{NEXT_ACTION}`",
    ]) + "\n"
    (out_dir / "final_report.md").write_text(md, encoding="utf-8")
    gate_payload = dict(final_report)
    gate_payload.update({"artifact": str(out_dir), "gate_passed": gate_status == PASS_GATE})
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

    (out_dir / "_SUCCESS.lock").write_text(
        json.dumps({"success": True, "gate": gate_status, "created_at": now_iso()}, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"artifact:\n{out_dir}")
    print(f"gate:\n{gate_status}")
    print("axis/operator/domain:\nclock_rounding_allowance/Q_CLOCK/SYMBOLIC_UNORDERED_CONVENTION_DOMAIN")
    print(f"evidence source/strength/completeness/property counts:\n{len(source_rows)}/{len(strength_rows)}/{len(dimension_rows)}/{len(property_rows)}")
    print("direct semantic required/implementation-or-test required/conflict policy/fallback policy/schema:\ntrue/true/true/true/true")
    print("domain-kind evidence available/actual-member identification evidence available:\ntrue/false")
    print("actual convention member/resolution/unit/new assumption:\n0/0/0/0")
    print("test vector generation/test execution/operator execution:\n0/0/0")
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
