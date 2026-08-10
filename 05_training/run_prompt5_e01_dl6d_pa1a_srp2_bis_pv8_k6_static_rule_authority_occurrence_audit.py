#!/usr/bin/env python3
"""PV8-K6 static rule authority and repeated-stop occurrence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit.py"
ROUTE_SEQUENCE_PATH = ARTIFACTS_ROOT / "suseong_source_pack_v1" / "route_stop_sequences.parquet"

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
C3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c3_mappo_interface_compatibility_20260808_100118"
K1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k1_skip_safety_evidence_audit_20260808_101303"
K2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k2_safety_state_contract_20260808_102318"
K3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k3_safety_source_state_feasibility_20260808_112054"
K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"
K5_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness_20260808_121540"

UPSTREAMS = {
    "PV8-C2": (C2_ROOT, "artifact_manifest_srp2_bis_pv8_c2.json", "_PV8_C2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"),
    "PV8-C3": (C3_ROOT, "artifact_manifest_srp2_bis_pv8_c3.json", "_PV8_C3_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_COMPLETE"),
    "PV8-K1": (K1_ROOT, "artifact_manifest_srp2_bis_pv8_k1.json", "_PV8_K1_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K1_SKIP_SAFETY_EVIDENCE_AUDIT_COMPLETE"),
    "PV8-K2": (K2_ROOT, "artifact_manifest_srp2_bis_pv8_k2.json", "_PV8_K2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K2_SAFETY_STATE_CONTRACT_COMPLETE"),
    "PV8-K3": (K3_ROOT, "artifact_manifest_srp2_bis_pv8_k3.json", "_PV8_K3_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K3_SAFETY_SOURCE_AND_STATE_FEASIBILITY_COMPLETE"),
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
    "PV8-K5": (K5_ROOT, "artifact_manifest_srp2_bis_pv8_k5.json", "_PV8_K5_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K5_STATIC_RULEBOOK_AND_K_ACTION_MASK_READINESS_COMPLETE"),
}

R2D_TURNAROUND_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D_TURNAROUND_JSON = R2D_TURNAROUND_ROOT / "turnaround_mapping_contract_v10_hf1.json"
R2D_TURNAROUND_GATE = R2D_TURNAROUND_ROOT / "prompt5_e01_r2d1c_r4a_hf1_gate.json"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K6_STATIC_RULE_AUTHORITY_AND_OCCURRENCE_AUDIT_COMPLETE"
FAIL_GATE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K6_STATIC_RULE_AUTHORITY_AND_OCCURRENCE_AUDIT_INCOMPLETE"
PASS_READINESS = "SRP2_BIS_PV8_K6_COMPLETE_RESEARCH_CONTRACT_APPROVAL_REQUIRED_K7_LOCKED"

DECISION_REAL = "STATIC_RULEBOOK_REAL_NETWORK_READY"
DECISION_RESEARCH = "STATIC_RULEBOOK_RESEARCH_CONTRACT_APPROVAL_REQUIRED"
DECISION_EXTERNAL = "STATIC_RULEBOOK_EXTERNAL_SOURCE_REQUIRED"

CLASSIFICATIONS = {
    "OBSERVED",
    "DERIVED_SAFE",
    "CONTRACT_FIXED_RESEARCH_RULE",
    "UNKNOWN_EXTERNAL_APPROVAL_REQUIRED",
}
STATIC_FIELDS = (
    "mandatory_stop",
    "protected_stop",
    "planned_itinerary_allows_skip",
    "terminal_or_turnaround_stop",
    "charging_or_driver_relief_stop",
)

PAYLOADS = [
    "k6_occurrence_reconciliation.json",
    "k6_route_stop_occurrence_master.parquet",
    "k6_static_rule_authority_matrix.json",
    "k6_real_network_rulebook.parquet",
    "k6_research_rule_template.json",
    "k6_remaining_approval_gaps.json",
    "k6_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class K6Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def verify_upstreams() -> Dict[str, Any]:
    verified: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed_gate != expected_gate or not k5.manifest_ok(checks):
            raise K6Error(f"{label} upstream integrity failure: gate={observed_gate}, checks={checks}")
        verified[label] = {
            "artifact_root": str(root),
            "gate": observed_gate,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return verified


def stable_occurrence_id(route_id: str, direction_id: str, stop_sequence: int, occurrence_index: int, stop_id: str) -> str:
    payload = json.dumps(
        ["PV8_K6_ROUTE_STOP_OCCURRENCE_V1", route_id, direction_id, int(stop_sequence), int(occurrence_index), stop_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "RSO1_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


OCCURRENCE_COLUMNS = [
    "route_stop_occurrence_id", "occurrence_key_version", "route_id", "direction_id", "stop_sequence",
    "occurrence_index", "stop_id", "same_stop_occurrence_ordinal", "same_stop_occurrence_count",
    "is_repeated_stop_occurrence", "is_interior_repeated_stop_occurrence", "route_direction_endpoint",
    "endpoint_role", "operational_turnaround_stop", "operational_turnaround_classification",
    "prior_stop_id", "post_skip_target_exists", "post_skip_target_stop_id", "post_skip_target_sequence",
    "node_uid", "route_no", "route_type", "stop_name", "x_pos", "y_pos", "source_classification",
    "source_dataset_path", "source_dataset_sha256",
]


def build_occurrence_master() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source = pd.read_parquet(ROUTE_SEQUENCE_PATH).copy()
    required = {"route_id", "direction_id", "stop_order", "stop_id", "node_uid", "route_no", "route_type", "stop_name"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise K6Error(f"route sequence columns missing: {missing}")
    for column in ("route_id", "direction_id", "stop_id", "node_uid", "route_no", "route_type", "stop_name", "source_classification"):
        source[column] = source[column].astype(str)
    source["stop_order"] = pd.to_numeric(source["stop_order"], errors="raise").astype(int)
    source = source.sort_values(["route_id", "direction_id", "stop_order", "stop_id"], kind="mergesort").reset_index(drop=True)
    source_sha = k5.sha256_file(ROUTE_SEQUENCE_PATH)
    source["same_stop_occurrence_count"] = source.groupby(["route_id", "direction_id", "stop_id"])["stop_id"].transform("size").astype(int)
    source["same_stop_occurrence_ordinal"] = source.groupby(["route_id", "direction_id", "stop_id"]).cumcount().add(1).astype(int)

    rows: List[Dict[str, Any]] = []
    for (route_id, direction_id), group in source.groupby(["route_id", "direction_id"], sort=True):
        ordered = group.sort_values(["stop_order", "stop_id"], kind="mergesort").reset_index(drop=True)
        for offset, source_row in ordered.iterrows():
            occurrence_index = offset + 1
            is_start = occurrence_index == 1
            is_end = occurrence_index == len(ordered)
            endpoint_role = "START_AND_END" if is_start and is_end else "START" if is_start else "END" if is_end else "NONE"
            has_post = offset + 1 < len(ordered)
            prior = ordered.iloc[offset - 1] if offset > 0 else None
            post = ordered.iloc[offset + 1] if has_post else None
            repeated = int(source_row["same_stop_occurrence_count"]) > 1
            row = {
                "route_stop_occurrence_id": stable_occurrence_id(str(route_id), str(direction_id), int(source_row["stop_order"]), occurrence_index, str(source_row["stop_id"])),
                "occurrence_key_version": "PV8_K6_ROUTE_STOP_OCCURRENCE_V1",
                "route_id": str(route_id),
                "direction_id": str(direction_id),
                "stop_sequence": int(source_row["stop_order"]),
                "occurrence_index": occurrence_index,
                "stop_id": str(source_row["stop_id"]),
                "same_stop_occurrence_ordinal": int(source_row["same_stop_occurrence_ordinal"]),
                "same_stop_occurrence_count": int(source_row["same_stop_occurrence_count"]),
                "is_repeated_stop_occurrence": repeated,
                "is_interior_repeated_stop_occurrence": bool(repeated and not (is_start or is_end)),
                "route_direction_endpoint": bool(is_start or is_end),
                "endpoint_role": endpoint_role,
                "operational_turnaround_stop": None,
                "operational_turnaround_classification": "UNKNOWN_EXTERNAL_APPROVAL_REQUIRED",
                "prior_stop_id": str(prior["stop_id"]) if prior is not None else None,
                "post_skip_target_exists": has_post,
                "post_skip_target_stop_id": str(post["stop_id"]) if post is not None else None,
                "post_skip_target_sequence": int(post["stop_order"]) if post is not None else None,
                "node_uid": str(source_row["node_uid"]),
                "route_no": str(source_row["route_no"]),
                "route_type": str(source_row["route_type"]),
                "stop_name": str(source_row["stop_name"]),
                "x_pos": float(source_row["x_pos"]) if pd.notna(source_row.get("x_pos")) else None,
                "y_pos": float(source_row["y_pos"]) if pd.notna(source_row.get("y_pos")) else None,
                "source_classification": str(source_row["source_classification"]),
                "source_dataset_path": str(ROUTE_SEQUENCE_PATH),
                "source_dataset_sha256": source_sha,
            }
            rows.append(row)

    key = lambda row: (row["route_id"], row["direction_id"], row["stop_sequence"], row["occurrence_index"], row["stop_id"])
    triplet_counts = Counter((row["route_id"], row["direction_id"], row["stop_id"]) for row in rows)
    repeated_groups = {group_key: count for group_key, count in triplet_counts.items() if count > 1}
    count_distribution = Counter(repeated_groups.values())
    reconciliation = {
        "created_at": iso_kst(),
        "source_row_count": len(source),
        "k5_canonical_row_count": int(k5.read_json(K5_ROOT / "k5_rulebook_completeness.json")["canonical_rulebook_row_count"]),
        "row_difference": len(source) - int(k5.read_json(K5_ROOT / "k5_rulebook_completeness.json")["canonical_rulebook_row_count"]),
        "difference_cause": "K5 grouped route_id+direction_id+stop_id, collapsing additional occurrences of stops repeated within the same ordered route-direction sequence",
        "repeated_route_direction_stop_group_count": len(repeated_groups),
        "repeated_stop_occurrence_row_count": sum(repeated_groups.values()),
        "extra_occurrence_count": sum(count - 1 for count in repeated_groups.values()),
        "repeated_count_distribution": {str(count): groups for count, groups in sorted(count_distribution.items())},
        "maximum_occurrences_per_route_direction_stop": max(repeated_groups.values()),
        "route_direction_count": len({(row["route_id"], row["direction_id"]) for row in rows}),
        "route_direction_stop_sequence_duplicate_count": len(rows) - len({(row["route_id"], row["direction_id"], row["stop_sequence"]) for row in rows}),
        "occurrence_key_duplicate_count": len(rows) - len({key(row) for row in rows}),
        "route_stop_occurrence_id_duplicate_count": len(rows) - len({row["route_stop_occurrence_id"] for row in rows}),
        "occurrence_id_recomputation_mismatch_count": sum(
            row["route_stop_occurrence_id"]
            != stable_occurrence_id(row["route_id"], row["direction_id"], row["stop_sequence"], row["occurrence_index"], row["stop_id"])
            for row in rows
        ),
        "source_rows_preserved_one_to_one": len(rows) == len(source),
        "endpoint_occurrence_count": sum(bool(row["route_direction_endpoint"]) for row in rows),
        "interior_repeated_stop_occurrence_count": sum(bool(row["is_interior_repeated_stop_occurrence"]) for row in rows),
        "occurrence_key_contract": ["route_id", "direction_id", "stop_sequence", "occurrence_index", "stop_id"],
        "route_stop_occurrence_id_algorithm": "RSO1_ + SHA256(canonical JSON array of version, route_id, direction_id, stop_sequence, occurrence_index, stop_id)",
    }
    return rows, reconciliation


RULEBOOK_COLUMNS = [
    "rule_scope", "route_stop_occurrence_id", "route_id", "direction_id", "stop_sequence", "occurrence_index", "stop_id",
    "same_stop_occurrence_ordinal", "same_stop_occurrence_count", "is_repeated_stop_occurrence",
    "is_interior_repeated_stop_occurrence", "route_direction_endpoint", "endpoint_role", "post_skip_target_exists",
    "post_skip_target_stop_id", "mandatory_stop", "mandatory_stop_classification", "mandatory_stop_provenance",
    "protected_stop", "protected_stop_classification", "protected_stop_provenance",
    "planned_itinerary_allows_skip", "planned_itinerary_allows_skip_classification", "planned_itinerary_allows_skip_provenance",
    "terminal_or_turnaround_stop", "terminal_or_turnaround_stop_classification", "terminal_or_turnaround_stop_provenance",
    "operational_turnaround_stop", "operational_turnaround_classification",
    "charging_or_driver_relief_stop", "charging_or_driver_relief_stop_classification", "charging_or_driver_relief_stop_provenance",
    "effective_scope", "effective_from", "effective_to", "rule_authority_complete", "unknown_fields_json",
    "positive_static_skip_clearance", "research_rule_mixed_into_real_network", "source_dataset_sha256",
]


def build_real_network_rulebook(occurrences: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for occurrence in occurrences:
        has_post = bool(occurrence["post_skip_target_exists"])
        endpoint = bool(occurrence["route_direction_endpoint"])
        values: Dict[str, Optional[bool]] = {
            "mandatory_stop": None,
            "protected_stop": None,
            "planned_itinerary_allows_skip": None if has_post else False,
            "terminal_or_turnaround_stop": True if endpoint else None,
            "charging_or_driver_relief_stop": None,
        }
        classes = {field: "UNKNOWN_EXTERNAL_APPROVAL_REQUIRED" for field in STATIC_FIELDS}
        if not has_post:
            classes["planned_itinerary_allows_skip"] = "DERIVED_SAFE"
        if endpoint:
            classes["terminal_or_turnaround_stop"] = "DERIVED_SAFE"
        unknown = [field for field in STATIC_FIELDS if values[field] is None]
        provenance = {
            "mandatory_stop": "no operator-owned mandatory-service rule source found; missing is not false",
            "protected_stop": "no operator-owned protected-stop rule source found; geography is not authority",
            "planned_itinerary_allows_skip": "ordered occurrence has no post-skip target; one-sided block" if not has_post else "topology does not grant skip permission",
            "terminal_or_turnaround_stop": "ordered route-direction start/end occurrence; one-sided endpoint block" if endpoint else "interior operational turnaround absence is not proven",
            "charging_or_driver_relief_stop": "no charging, depot, block-duty, or driver-relief authority source found",
        }
        row: Dict[str, Any] = {
            "rule_scope": "REAL_NETWORK_RULE",
            **{column: occurrence[column] for column in (
                "route_stop_occurrence_id", "route_id", "direction_id", "stop_sequence", "occurrence_index", "stop_id",
                "same_stop_occurrence_ordinal", "same_stop_occurrence_count", "is_repeated_stop_occurrence",
                "is_interior_repeated_stop_occurrence", "route_direction_endpoint", "endpoint_role",
                "post_skip_target_exists", "post_skip_target_stop_id",
            )},
            "operational_turnaround_stop": None,
            "operational_turnaround_classification": "UNKNOWN_EXTERNAL_APPROVAL_REQUIRED",
            "effective_scope": "ROUTE_DIRECTION_STOP_OCCURRENCE",
            "effective_from": None,
            "effective_to": None,
            "rule_authority_complete": False,
            "unknown_fields_json": json.dumps(unknown),
            "positive_static_skip_clearance": False,
            "research_rule_mixed_into_real_network": False,
            "source_dataset_sha256": occurrence["source_dataset_sha256"],
        }
        for field in STATIC_FIELDS:
            row[field] = values[field]
            row[f"{field}_classification"] = classes[field]
            row[f"{field}_provenance"] = provenance[field]
        rows.append(row)
    return rows


def database_authority_audit() -> Dict[str, Any]:
    import psycopg2

    queries = [
        (
            "exact_static_rule_columns",
            """SELECT table_name, column_name, data_type
               FROM information_schema.columns
               WHERE table_schema='public'
                 AND lower(column_name) ~ 'mandatory|protected|terminal|turnaround|charging|driver_relief|relief|allows_skip|skip_allowed|skip_legal|itinerary'
               ORDER BY table_name, ordinal_position""",
        ),
        (
            "candidate_operational_rule_relations",
            """SELECT table_name, table_type
               FROM information_schema.tables
               WHERE table_schema='public'
                 AND lower(table_name) ~ 'mandatory|protected|turnaround|charging|relief|duty|depot|skip|itinerary|operation_rule|stop_rule'
               ORDER BY table_name""",
        ),
        (
            "route_stop_master_columns",
            """SELECT table_name, column_name, data_type
               FROM information_schema.columns
               WHERE table_schema='public'
                 AND table_name IN ('stg_daegu_routes','dim_stop','graph_edge_master','graph_node_master','route_link_sequence')
               ORDER BY table_name, ordinal_position""",
        ),
    ]
    connection = psycopg2.connect(dbname="urbanbus")
    connection.set_session(readonly=True, autocommit=True)
    registry: List[Dict[str, Any]] = []
    results: Dict[str, List[Dict[str, Any]]] = {}
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET default_transaction_read_only = on")
            cursor.execute("SET statement_timeout = '30s'")
        for query_id, sql in queries:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                values = cursor.fetchall()
                columns = [item.name for item in cursor.description]
            results[query_id] = [dict(zip(columns, value)) for value in values]
            registry.append({"query_id": query_id, "normalized_sql": " ".join(sql.split()), "read_only_verified": True, "returned_row_count": len(values), "db_write_count": 0})
    finally:
        connection.close()
    excluded_temporal = [
        {**row, "accepted_as_operational_rule": False, "reason": "temporal model transition label has no route-stop rule provenance"}
        for row in results["exact_static_rule_columns"]
        if row["column_name"] == "is_terminal_transition"
    ]
    return {
        "connection": "local urbanbus PostgreSQL read-only session",
        "direct_postgresql_query_count": len(queries),
        "db_write_count": 0,
        "query_registry": registry,
        "exact_static_rule_column_candidates": results["exact_static_rule_columns"],
        "candidate_operational_rule_relations": results["candidate_operational_rule_relations"],
        "candidate_operational_rule_relation_count": len(results["candidate_operational_rule_relations"]),
        "route_stop_master_columns": results["route_stop_master_columns"],
        "excluded_temporal_feature_columns": excluded_temporal,
        "approved_static_rule_authority_relation_count": 0,
    }


def scan_project_contracts() -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    for path in sorted(TRAINING_ROOT.rglob("*")):
        if not path.is_file() or "artifacts" in path.parts or path.suffix.lower() not in {".py", ".json", ".yaml", ".yml", ".toml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        matched = [field for field in STATIC_FIELDS if field in text]
        if not matched:
            continue
        relative = str(path.relative_to(PROJECT_ROOT))
        if path.name.startswith("test_"):
            category = "TEST_FIXTURE"
        elif path.name.startswith("run_prompt"):
            category = "AUDIT_OR_RESEARCH_RUNNER"
        elif "simulator" in path.parts:
            category = "SIMULATOR_SCHEMA_OR_PREDICATE"
        else:
            category = "UNCLASSIFIED_NON_AUTHORITY_REFERENCE"
        records.append(
            {
                "relative_path": relative,
                "sha256": k5.sha256_file(path),
                "matched_fields": matched,
                "category": category,
                "accepted_as_real_network_rule_authority": False,
            }
        )
    return {
        "scanned_root": str(TRAINING_ROOT),
        "artifact_directories_excluded": True,
        "matching_file_count": len(records),
        "real_network_rule_authority_file_count": 0,
        "records": records,
    }


def turnaround_candidate_audit(occurrences: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    contract = k5.read_json(R2D_TURNAROUND_JSON)
    gate = k5.read_json(R2D_TURNAROUND_GATE)
    records = contract.get("rows", [])
    occurrence_keys = {
        (row["route_id"], row["direction_id"], int(row["stop_sequence"]), row["stop_id"])
        for row in occurrences
    }
    exact_matches = 0
    finite_live = 0
    sequence_conflicts = 0
    for record in records:
        direction = str(record.get("service_direction_id") or record.get("direction_id"))
        sequence = record.get("service_terminal_sequence")
        stop_id = str(record.get("service_terminal_stop_id") or record.get("terminal_stop_id"))
        if sequence is not None and (str(record.get("route_id")), direction, int(sequence), stop_id) in occurrence_keys:
            exact_matches += 1
        effective = record.get("effective_live_terminal_sequence")
        if isinstance(effective, (int, float)) and math.isfinite(float(effective)):
            finite_live += 1
            if sequence is None or float(effective) != float(sequence):
                sequence_conflicts += 1
    return {
        "artifact_root": str(R2D_TURNAROUND_ROOT),
        "contract_sha256": k5.sha256_file(R2D_TURNAROUND_JSON),
        "gate_sha256": k5.sha256_file(R2D_TURNAROUND_GATE),
        "mapping_row_count": len(records),
        "approved_mapping_row_count": sum(bool(row.get("approved")) for row in records),
        "passed_mapping_row_count": sum(bool(row.get("passed")) for row in records),
        "service_terminal_exact_occurrence_match_count": exact_matches,
        "finite_effective_live_sequence_count": finite_live,
        "effective_live_vs_service_sequence_conflict_count": sequence_conflicts,
        "terminal_recovery_applied": bool(gate.get("terminal_recovery_applied")),
        "approved_for_phase2_turnaround": bool(gate.get("approved_for_phase2_turnaround")),
        "real_world_causal_claim_allowed": bool(gate.get("real_world_causal_claim_allowed")),
        "promoted_to_k6_operational_rule_count": 0,
        "non_promotion_reason": "historical turnaround mapping was not applied or approved for Phase 2, and effective live sequence often differs from service-terminal sequence; it cannot automatically bind an effective-time occurrence-level K-safety rule",
    }


def authority_matrix(rulebook: Sequence[Mapping[str, Any]], db_audit: Mapping[str, Any], project_scan: Mapping[str, Any], turnaround_audit: Mapping[str, Any]) -> Dict[str, Any]:
    records = []
    for field in STATIC_FIELDS:
        counts = Counter(row[f"{field}_classification"] for row in rulebook)
        for classification in CLASSIFICATIONS:
            counts.setdefault(classification, 0)
        if field == "planned_itinerary_allows_skip":
            proven = "false only where the occurrence has no post-skip target"
            needed = "operator-approved skip permission and valid path rule for every eligible occurrence"
        elif field == "terminal_or_turnaround_stop":
            proven = "true for ordered route-direction start/end occurrences only"
            needed = "effective-time operational turnaround list, including interior and repeated occurrences"
        elif field == "mandatory_stop":
            proven = "none"
            needed = "operator-owned mandatory service-stop rules with route/direction/occurrence scope"
        elif field == "protected_stop":
            proven = "none"
            needed = "operator/regulator protected-stop designation and effective period"
        else:
            proven = "none"
            needed = "vehicle block, depot, charging, and driver-relief stop schedule authority"
        records.append(
            {
                "predicate": field,
                "overall_classification": "UNKNOWN_EXTERNAL_APPROVAL_REQUIRED",
                "real_network_observed_row_count": counts["OBSERVED"],
                "one_sided_derived_safe_row_count": counts["DERIVED_SAFE"],
                "research_contract_fixed_row_count": counts["CONTRACT_FIXED_RESEARCH_RULE"],
                "unknown_external_approval_row_count": counts["UNKNOWN_EXTERNAL_APPROVAL_REQUIRED"],
                "actually_proven_now": proven,
                "authority_required_for_closure": needed,
                "missing_interpreted_as_false": False,
            }
        )
    return {
        "created_at": iso_kst(),
        "classification_domain": sorted(CLASSIFICATIONS),
        "records": records,
        "database_authority_audit": db_audit,
        "project_contract_scan": project_scan,
        "historical_turnaround_candidate_audit": turnaround_audit,
        "real_network_predicates_actually_proven": [
            "route_direction_endpoint=true from authoritative ordered sequence",
            "planned_itinerary_allows_skip=false where no post-skip target exists",
            "interior_repeated_stop_occurrence identity from ordered occurrence position",
        ],
        "observed_complete_predicate_count": 0,
        "real_network_rulebook_complete": False,
        "real_network_and_research_provenance_mixed": False,
        "new_bis_api_call_count": 0,
        "db_write_count": 0,
    }


def research_rule_template() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "template_name": "PV8_K6_RESEARCH_ONLY_STATIC_STOP_RULE_V1_DRAFT",
        "scope": "RESEARCH_RULE_TEMPLATE_NOT_REAL_NETWORK_OBSERVATION",
        "activated": False,
        "approval_status": "USER_OR_RESEARCH_GOVERNANCE_APPROVAL_REQUIRED",
        "classification_after_approval": "CONTRACT_FIXED_RESEARCH_RULE",
        "may_be_mixed_into_real_network_rulebook": False,
        "complete_research_rulebook_freezable_without_real_world_observation_claim": True,
        "required_key": ["route_stop_occurrence_id", "route_id", "direction_id", "stop_sequence", "occurrence_index", "stop_id"],
        "fields": [
            {"predicate": "mandatory_stop", "approved_value_or_occurrence_list": None, "approval_required": True, "default_allowed": False},
            {"predicate": "protected_stop", "approved_value_or_occurrence_list": None, "approval_required": True, "default_allowed": False},
            {"predicate": "planned_itinerary_allows_skip", "approved_research_policy": None, "approval_required": True, "default_allowed": False},
            {"predicate": "terminal_or_turnaround_stop", "derived_endpoints_retained": True, "approved_interior_turnaround_occurrence_list": None, "approval_required": True},
            {"predicate": "charging_or_driver_relief_stop", "approved_value_or_occurrence_list": None, "approval_required": True, "default_allowed": False},
        ],
        "freeze_prerequisites": [
            "approve every unresolved occurrence or approve a deterministic research-only rule that covers it",
            "record approver, rationale, effective version, and SHA-256 of the occurrence master",
            "label every filled value CONTRACT_FIXED_RESEARCH_RULE",
            "forbid real-network observation, operator-policy, causal, and deployment claims",
            "keep K_action_mask and policy execution locked until a separate K7 integration authorization",
        ],
    }


def approval_gaps() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "occurrence_identity_gap_closed": True,
        "real_network_external_source_required": True,
        "research_contract_approval_path_available": True,
        "gaps": [
            {"predicate": "mandatory_stop", "exact_missing_authority": "effective-time operator mandatory-service stop list keyed to route-direction occurrence"},
            {"predicate": "protected_stop", "exact_missing_authority": "effective-time operator/regulator protected-stop designation"},
            {"predicate": "planned_itinerary_allows_skip", "exact_missing_authority": "operator-approved skip permission plus path-validity semantics for each occurrence"},
            {"predicate": "terminal_or_turnaround_stop", "exact_missing_authority": "effective-time operational turnaround list resolving interior and repeated occurrences"},
            {"predicate": "charging_or_driver_relief_stop", "exact_missing_authority": "vehicle block/depot/charging/driver-relief schedule mapped to route-stop occurrence"},
        ],
        "minimum_research_approval": [
            "approve a research-only value or deterministic rule for every unresolved field and occurrence",
            "approve strict claim label separating CONTRACT_FIXED_RESEARCH_RULE from OBSERVED/DERIVED_SAFE",
            "freeze occurrence-master hash, rule version, approver, and effective scope before K7",
        ],
        "absence_of_evidence_interpreted_as_false": False,
    }


def final_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# PV8-K6 Static Rule Authority and Occurrence Audit Final Report",
            "",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- gate: `{summary['gate']}`",
            f"- final decision: `{summary['final_decision']}`",
            f"- source rows: `{summary['source_rows']}`",
            f"- K5 canonical rows: `{summary['k5_rows']}`",
            f"- collapsed difference: `{summary['row_difference']}`",
            f"- repeated route-direction-stop groups: `{summary['repeated_groups']}`",
            f"- repeated occurrence rows: `{summary['repeated_occurrence_rows']}`",
            f"- extra occurrences previously collapsed: `{summary['extra_occurrences']}`",
            f"- occurrence master rows / unique IDs: `{summary['occurrence_rows']} / {summary['unique_occurrence_ids']}`",
            f"- real-network rulebook rows: `{summary['rulebook_rows']}`",
            f"- complete real-network rule rows: `{summary['complete_real_rows']}`",
            f"- real-network positive SKIP rows: `{summary['positive_real_rows']}`",
            f"- K_action_mask_available: `{summary['K_action_mask_available']}`",
            "",
            "## 20,508 to 20,326",
            "",
            "K5 grouped by route_id, direction_id, and stop_id. There are 170 repeated keys containing 352 occurrence rows: 158 keys occur twice and 12 occur three times. The 182 additional occurrences were therefore collapsed. K6 preserves every source row with a stable occurrence-aware key and route_stop_occurrence_id.",
            "",
            "## Proven real-network predicates",
            "",
            "The authoritative ordered sequence proves route-direction endpoints and proves a skip impossible where no post-skip target exists. These are one-sided DERIVED_SAFE blockers. No full positive-clearance predicate is observed.",
            "",
            "## Approval and external-source gaps",
            "",
            "Mandatory, protected, operational-turnaround, and charging/driver-relief rules remain unavailable. Path existence does not grant planned skip permission. Historical turnaround observations were not applied or approved for Phase 2 and generally do not bind the same effective live sequence, so they were not promoted into occurrence rules.",
            "",
            "A complete research-only rulebook can be frozen after explicit approval, with every assumption labeled CONTRACT_FIXED_RESEARCH_RULE and with no real-world observation or deployment claim. The template is separate and inactive.",
            "",
            "## Minimum K7 scope",
            "",
            "Choose and approve either an operator-authoritative real-network rule source or the research-only contract branch; freeze the occurrence master and rule hashes; bind occurrence lookup plus K4 dynamic state to all eight fixed vehicle decision paths; serialize versions into snapshots; and run prospective fail-closed integration tests. Policy, checkpoint, evaluation, and training authorization remain separate.",
            "",
            "No DB write, new BIS/API call, K7 execution, policy evaluation, checkpoint reuse, or MAPPO retraining was performed.",
            "",
        ]
    )


def write_manifest_and_lock(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for relative_path in PAYLOADS:
        path = writer.root / relative_path
        rows.append({"relative_path": relative_path, "size_bytes": path.stat().st_size if path.exists() else None, "sha256": k5.sha256_file(path) if path.exists() else None, "required": True, "artifact_role": Path(relative_path).stem, "exists": path.exists()})
    jsonl_name = "artifact_manifest_srp2_bis_pv8_k6.jsonl"
    writer.text(jsonl_name, "".join(json.dumps(k5.json_clean(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    jsonl_path = writer.root / jsonl_name
    rows.append({"relative_path": jsonl_name, "size_bytes": jsonl_path.stat().st_size, "sha256": k5.sha256_file(jsonl_path), "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest_srp2_bis_pv8_k6.json"
    writer.json(manifest_name, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": sum(not row["exists"] for row in rows), "files": rows})
    manifest_path = writer.root / manifest_name
    writer.json("_PV8_K6_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest_name, "final_manifest_sha256": k5.sha256_file(manifest_path), "manifest_size_bytes": manifest_path.stat().st_size, "created_at": iso_kst()})


def run_audit(root: Path) -> Path:
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    upstreams = verify_upstreams()
    occurrences, reconciliation = build_occurrence_master()
    rulebook = build_real_network_rulebook(occurrences)
    db_audit = database_authority_audit()
    project_scan = scan_project_contracts()
    turnaround_audit = turnaround_candidate_audit(occurrences)
    matrix = authority_matrix(rulebook, db_audit, project_scan, turnaround_audit)
    template = research_rule_template()
    gaps = approval_gaps()

    complete_rows = sum(bool(row["rule_authority_complete"]) for row in rulebook)
    positive_rows = sum(bool(row["positive_static_skip_clearance"]) for row in rulebook)
    all_classes_valid = all(row[f"{field}_classification"] in CLASSIFICATIONS for row in rulebook for field in STATIC_FIELDS)
    occurrence_ids = {row["route_stop_occurrence_id"] for row in occurrences}
    rulebook_ids = {row["route_stop_occurrence_id"] for row in rulebook}
    audit_passed = bool(
        reconciliation["source_rows_preserved_one_to_one"]
        and reconciliation["row_difference"] == reconciliation["extra_occurrence_count"] == 182
        and reconciliation["occurrence_key_duplicate_count"] == 0
        and reconciliation["route_stop_occurrence_id_duplicate_count"] == 0
        and reconciliation["occurrence_id_recomputation_mismatch_count"] == 0
        and len(occurrences) == len(rulebook) == 20508
        and occurrence_ids == rulebook_ids
        and all_classes_valid
        and db_audit["db_write_count"] == 0
        and matrix["new_bis_api_call_count"] == 0
    )
    final_decision = DECISION_RESEARCH
    readiness = {
        "created_at": iso_kst(),
        "final_decision": final_decision,
        "K_action_mask_available": False,
        "conditional_skip_policy_enabled": False,
        "occurrence_identity_reconciled": True,
        "real_network_rulebook_complete": False,
        "research_rule_template_complete": True,
        "research_rule_template_approved": False,
        "complete_research_rulebook_freezable_without_real_world_observation_claim": True,
        "real_network_external_source_still_required": True,
        "minimum_k7_scope": [
            "obtain operator-authoritative rules or explicitly approve the isolated research-only contract branch",
            "freeze occurrence-master and rulebook hashes plus rule authority and effective version",
            "bind occurrence-aware static lookup and K4 dynamic snapshots into all eight fixed vehicle decision paths",
            "serialize occurrence/rule versions into clone-reset snapshots and run prospective no-future tests",
            "keep policy, checkpoint, evaluation, and training authorization separately locked",
        ],
    }
    guards = {
        "created_at": iso_kst(),
        "K_action_mask_available": False,
        "conditional_skip_policy_enabled": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_evaluation_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "research_rule_activation_authorized": False,
        "automatic_k7_execution_authorized": False,
        "automatic_mappo_retraining_authorized": False,
    }
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE if audit_passed else FAIL_GATE,
        "terminal_gate": PASS_GATE if audit_passed else FAIL_GATE,
        "readiness": PASS_READINESS if audit_passed else "SRP2_BIS_PV8_K6_AUDIT_FAILED",
        "gate_passed": audit_passed,
        "final_decision": final_decision,
        "failure_reasons": [] if audit_passed else ["upstream, occurrence identity, row preservation, classification, DB safety, or artifact condition failed"],
    }
    summary = {
        "artifact_root": str(root),
        "gate": gate["gate"],
        "final_decision": final_decision,
        "source_rows": reconciliation["source_row_count"],
        "k5_rows": reconciliation["k5_canonical_row_count"],
        "row_difference": reconciliation["row_difference"],
        "repeated_groups": reconciliation["repeated_route_direction_stop_group_count"],
        "repeated_occurrence_rows": reconciliation["repeated_stop_occurrence_row_count"],
        "extra_occurrences": reconciliation["extra_occurrence_count"],
        "occurrence_rows": len(occurrences),
        "unique_occurrence_ids": len(occurrence_ids),
        "rulebook_rows": len(rulebook),
        "complete_real_rows": complete_rows,
        "positive_real_rows": positive_rows,
        "K_action_mask_available": False,
    }

    writer.json("k6_occurrence_reconciliation.json", reconciliation)
    writer.parquet("k6_route_stop_occurrence_master.parquet", occurrences, OCCURRENCE_COLUMNS)
    writer.json("k6_static_rule_authority_matrix.json", matrix)
    writer.parquet("k6_real_network_rulebook.parquet", rulebook, RULEBOOK_COLUMNS)
    writer.json("k6_research_rule_template.json", template)
    writer.json("k6_remaining_approval_gaps.json", gaps)
    writer.json("k6_readiness_decision.json", readiness)
    writer.json("claim_guard_status.json", guards)
    writer.json(
        "run_manifest.json",
        {
            "created_at": iso_kst(),
            "artifact_family": ARTIFACT_PREFIX,
            "mode": "audit",
            "runner_path": str(RUNNER_PATH),
            "runner_sha256": k5.sha256_file(RUNNER_PATH),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "upstream_integrity": upstreams,
            "direct_postgresql_query_count": db_audit["direct_postgresql_query_count"],
            "db_write_count": 0,
            "new_bis_api_call_count": 0,
            "policy_evaluation_count": 0,
            "checkpoint_reuse_count": 0,
            "k7_execution_count": 0,
            "mappo_training_count": 0,
        },
    )
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": gate["gate"], "readiness": gate["readiness"], "final_decision": final_decision})
    writer.text("final_report.md", final_report(summary))
    write_manifest_and_lock(writer, gate)
    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_k6.json", "_PV8_K6_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise K6Error(f"K6 artifact integrity failure: {own}")
    if not audit_passed:
        raise K6Error(f"K6 audit failed: {gate['failure_reasons']}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"readiness: {gate['readiness']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run_audit(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
