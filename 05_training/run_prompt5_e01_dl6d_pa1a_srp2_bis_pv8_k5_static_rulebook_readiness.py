#!/usr/bin/env python3
"""PV8-K5 static stop rulebook closure and K-action-mask readiness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from simulator.k_safety_state import ServiceObligationStateMachine, StaticGuardStatus
from simulator.suseong_service_transition_engine import build_distinct_three_action_mask


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness.py"
ROUTE_SEQUENCE_PATH = ARTIFACTS_ROOT / "suseong_source_pack_v1" / "route_stop_sequences.parquet"
K4_STATE_SOURCE = TRAINING_ROOT / "simulator" / "k_safety_state.py"
K4_ENGINE_SOURCE = TRAINING_ROOT / "simulator" / "suseong_service_transition_engine.py"

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
C3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c3_mappo_interface_compatibility_20260808_100118"
K1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k1_skip_safety_evidence_audit_20260808_101303"
K2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k2_safety_state_contract_20260808_102318"
K3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k3_safety_source_state_feasibility_20260808_112054"
K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"

UPSTREAMS = {
    "PV8-C2": (C2_ROOT, "artifact_manifest_srp2_bis_pv8_c2.json", "_PV8_C2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"),
    "PV8-C3": (C3_ROOT, "artifact_manifest_srp2_bis_pv8_c3.json", "_PV8_C3_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_COMPLETE"),
    "PV8-K1": (K1_ROOT, "artifact_manifest_srp2_bis_pv8_k1.json", "_PV8_K1_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K1_SKIP_SAFETY_EVIDENCE_AUDIT_COMPLETE"),
    "PV8-K2": (K2_ROOT, "artifact_manifest_srp2_bis_pv8_k2.json", "_PV8_K2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K2_SAFETY_STATE_CONTRACT_COMPLETE"),
    "PV8-K3": (K3_ROOT, "artifact_manifest_srp2_bis_pv8_k3.json", "_PV8_K3_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K3_SAFETY_SOURCE_AND_STATE_FEASIBILITY_COMPLETE"),
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
}

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K5_STATIC_RULEBOOK_AND_K_ACTION_MASK_READINESS_COMPLETE"
FAIL_GATE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K5_STATIC_RULEBOOK_AND_K_ACTION_MASK_READINESS_INCOMPLETE"
READINESS = "SRP2_BIS_PV8_K5_COMPLETE_STATIC_SOURCE_INSUFFICIENT_K6_REPAIR_PENDING_USER_COMMAND"

DECISION_READY = "K_ACTION_MASK_READY_FOR_SIMULATOR_INTEGRATION"
DECISION_PARTIAL = "K_ACTION_MASK_PARTIAL_STATIC_RULE_GAPS"
DECISION_BLOCKED = "K_ACTION_MASK_BLOCKED_STATIC_SOURCE_INSUFFICIENT"

CLASSIFICATIONS = {"OBSERVED", "DERIVED_SAFE", "CONTRACT_FIXED", "UNKNOWN_FAIL_CLOSED"}
STATIC_FIELDS = (
    "mandatory_stop",
    "protected_stop",
    "planned_itinerary_allows_skip",
    "terminal_or_turnaround_stop",
    "charging_or_driver_relief_stop",
)

PAYLOADS = [
    "k5_static_rule_source_audit.json",
    "k5_static_stop_rulebook.parquet",
    "k5_rulebook_completeness.json",
    "k5_k_action_mask_dryrun.parquet",
    "k5_k_action_mask_validation.json",
    "k5_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class K5Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return json_clean(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root

    def text(self, relative_path: str, value: str) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def json(self, relative_path: str, value: Mapping[str, Any]) -> None:
        self.text(relative_path, json.dumps(json_clean(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")

    def parquet(self, relative_path: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([json_clean(dict(row)) for row in rows], columns=list(columns)).to_parquet(path, index=False)


def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def verify_manifest(root: Path, manifest_name: str, lock_name: str) -> Dict[str, Any]:
    manifest_path = root / manifest_name
    lock_path = root / lock_name
    checks: Dict[str, Any] = {
        "manifest_present": manifest_path.exists(),
        "lock_present": lock_path.exists(),
        "manifest_hash_ok": False,
        "manifest_size_ok": False,
        "manifest_entry_count": None,
        "missing_files": None,
        "required_missing_files": None,
        "sha256_mismatches": None,
        "size_mismatches": None,
        "terminal_lock_binding": False,
    }
    if not manifest_path.exists() or not lock_path.exists():
        return checks
    manifest = read_json(manifest_path)
    lock = read_json(lock_path)
    missing = required_missing = sha_bad = size_bad = 0
    for row in manifest.get("files", []):
        path = root / str(row.get("relative_path"))
        if not path.exists():
            missing += 1
            required_missing += int(bool(row.get("required", True)))
            continue
        sha_bad += int(bool(row.get("sha256") and sha256_file(path) != row["sha256"]))
        size_bad += int(bool(row.get("size_bytes") is not None and path.stat().st_size != row["size_bytes"]))
    manifest_sha = sha256_file(manifest_path)
    checks.update(
        {
            "manifest_hash_ok": manifest_sha in {lock.get("manifest_sha256"), lock.get("final_manifest_sha256")},
            "manifest_size_ok": manifest_path.stat().st_size == lock.get("manifest_size_bytes", manifest_path.stat().st_size),
            "manifest_entry_count": len(manifest.get("files", [])),
            "missing_files": missing,
            "required_missing_files": required_missing,
            "sha256_mismatches": sha_bad,
            "size_mismatches": size_bad,
            "terminal_lock_binding": lock.get("manifest_relative_path") == manifest_name or lock.get("final_manifest_path") == manifest_name,
        }
    )
    return checks


def manifest_ok(checks: Mapping[str, Any]) -> bool:
    return bool(
        checks.get("manifest_present")
        and checks.get("lock_present")
        and checks.get("manifest_hash_ok")
        and checks.get("manifest_size_ok")
        and checks.get("missing_files") == 0
        and checks.get("required_missing_files") == 0
        and checks.get("sha256_mismatches") == 0
        and checks.get("size_mismatches") == 0
        and checks.get("terminal_lock_binding")
    )


def verify_upstreams() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = read_json(root / "gate_decision.json")
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        checks = verify_manifest(root, manifest_name, lock_name)
        if observed_gate != expected_gate or not manifest_ok(checks):
            raise K5Error(f"{name} integrity failure: gate={observed_gate}, checks={checks}")
        out[name] = {
            "artifact_root": str(root),
            "gate": observed_gate,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return out


def k4_source_integrity() -> Dict[str, Any]:
    contract = read_json(K4_ROOT / "k4_state_machine_contract.json")
    frozen = {str(row["path"]): str(row["sha256"]) for row in contract["source_files"]}
    rows = []
    for path in (K4_STATE_SOURCE, K4_ENGINE_SOURCE):
        current = sha256_file(path)
        expected = frozen.get(str(path))
        rows.append({"path": str(path), "expected_sha256": expected, "current_sha256": current, "matches_k4": current == expected})
    return {"records": rows, "source_drift_count": sum(not row["matches_k4"] for row in rows)}


def database_static_schema_audit() -> Dict[str, Any]:
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
            "route_stop_master_columns",
            """SELECT table_name, column_name, data_type
               FROM information_schema.columns
               WHERE table_schema='public'
                 AND table_name IN ('stg_daegu_routes','dim_stop','graph_edge_master','graph_node_master','route_link_sequence')
               ORDER BY table_name, ordinal_position""",
        ),
    ]
    records: List[Dict[str, Any]] = []
    results: Dict[str, List[Dict[str, Any]]] = {}
    connection = psycopg2.connect(dbname="urbanbus")
    connection.set_session(readonly=True, autocommit=True)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SET default_transaction_read_only = on")
            cursor.execute("SET statement_timeout = '30s'")
        for query_id, sql in queries:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                columns = [item.name for item in cursor.description]
            results[query_id] = [dict(zip(columns, row)) for row in rows]
            records.append(
                {
                    "query_id": query_id,
                    "normalized_sql": " ".join(sql.split()),
                    "read_only_verified": True,
                    "returned_row_count": len(rows),
                    "db_write_count": 0,
                }
            )
    finally:
        connection.close()

    exact = results["exact_static_rule_columns"]
    exact_names = {str(row["column_name"]).lower() for row in exact}
    direct_matches = {
        field: sorted(name for name in exact_names if name == field)
        for field in STATIC_FIELDS
    }
    excluded = [
        {
            **row,
            "accepted_as_static_rule": False,
            "reason": "temporal model transition label is not a route-direction-stop operational rule",
        }
        for row in exact
        if row["column_name"] == "is_terminal_transition"
    ]
    return {
        "connection": "local urbanbus PostgreSQL via read-only session",
        "direct_postgresql_query_count": len(queries),
        "db_write_count": 0,
        "query_registry": records,
        "exact_static_rule_column_candidates": exact,
        "exact_required_field_matches": direct_matches,
        "authoritative_operational_rule_column_count": sum(bool(value) for value in direct_matches.values()),
        "excluded_non_rule_columns": excluded,
        "route_stop_master_column_count": len(results["route_stop_master_columns"]),
        "route_stop_master_columns": results["route_stop_master_columns"],
    }


def build_rulebook() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source = pd.read_parquet(ROUTE_SEQUENCE_PATH).copy()
    required = {"route_id", "direction_id", "stop_id", "stop_order", "route_no", "stop_name", "node_uid"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise K5Error(f"route sequence missing columns: {missing}")
    for column in ("route_id", "direction_id", "stop_id", "route_no", "stop_name", "node_uid"):
        source[column] = source[column].astype(str)
    source["stop_order"] = pd.to_numeric(source["stop_order"], errors="raise").astype(int)
    source = source.sort_values(["route_id", "direction_id", "stop_order", "stop_id"], kind="mergesort").reset_index(drop=True)

    occurrences: List[Dict[str, Any]] = []
    for (route_id, direction_id), group in source.groupby(["route_id", "direction_id"], sort=True):
        ordered = group.sort_values(["stop_order", "stop_id"], kind="mergesort").reset_index(drop=True)
        for index, row in ordered.iterrows():
            has_post = index + 1 < len(ordered)
            occurrences.append(
                {
                    **row.to_dict(),
                    "is_sequence_start": index == 0,
                    "is_sequence_end": index == len(ordered) - 1,
                    "is_route_direction_endpoint": index in {0, len(ordered) - 1},
                    "post_skip_target_exists": has_post,
                    "post_skip_target_stop_id": str(ordered.iloc[index + 1]["stop_id"]) if has_post else None,
                }
            )
    occurrence_df = pd.DataFrame(occurrences)
    source_sha = sha256_file(ROUTE_SEQUENCE_PATH)
    rows: List[Dict[str, Any]] = []
    for (route_id, direction_id, stop_id), group in occurrence_df.groupby(["route_id", "direction_id", "stop_id"], sort=True):
        endpoint_any = bool(group["is_route_direction_endpoint"].any())
        endpoint_all = bool(group["is_route_direction_endpoint"].all())
        post_any = bool(group["post_skip_target_exists"].any())
        post_all = bool(group["post_skip_target_exists"].all())
        planned_value: Optional[bool] = False if not post_any else None
        planned_class = "DERIVED_SAFE" if planned_value is False else "UNKNOWN_FAIL_CLOSED"
        terminal_value: Optional[bool] = True if endpoint_any else None
        terminal_class = "DERIVED_SAFE" if terminal_value is True else "UNKNOWN_FAIL_CLOSED"
        values = {
            "mandatory_stop": None,
            "protected_stop": None,
            "planned_itinerary_allows_skip": planned_value,
            "terminal_or_turnaround_stop": terminal_value,
            "charging_or_driver_relief_stop": None,
        }
        classes = {
            "mandatory_stop": "UNKNOWN_FAIL_CLOSED",
            "protected_stop": "UNKNOWN_FAIL_CLOSED",
            "planned_itinerary_allows_skip": planned_class,
            "terminal_or_turnaround_stop": terminal_class,
            "charging_or_driver_relief_stop": "UNKNOWN_FAIL_CLOSED",
        }
        unresolved = [field for field in STATIC_FIELDS if values[field] is None]
        base: Dict[str, Any] = {
            "route_id": route_id,
            "direction_id": direction_id,
            "stop_id": stop_id,
            "route_no": sorted(set(group["route_no"].astype(str)))[0],
            "stop_name": sorted(set(group["stop_name"].astype(str)))[0],
            "node_uid": sorted(set(group["node_uid"].astype(str)))[0],
            "stop_orders_json": json.dumps(sorted(int(value) for value in group["stop_order"])),
            "occurrence_count": int(len(group)),
            "endpoint_occurrence_count": int(group["is_route_direction_endpoint"].sum()),
            "post_skip_target_exists_any_occurrence": post_any,
            "post_skip_target_exists_all_occurrences": post_all,
            "post_skip_target_stop_ids_json": json.dumps(sorted(set(str(value) for value in group["post_skip_target_stop_id"].dropna()))),
            "occurrence_context_ambiguous": bool((endpoint_any != endpoint_all) or (post_any != post_all)),
            "effective_scope": "ROUTE_DIRECTION_STOP",
            "effective_from": None,
            "effective_to": None,
            "static_state_complete": len(unresolved) == 0,
            "unresolved_fields_json": json.dumps(unresolved),
            "positive_static_skip_clearance": False,
            "source_dataset_path": str(ROUTE_SEQUENCE_PATH),
            "source_dataset_sha256": source_sha,
        }
        for field in STATIC_FIELDS:
            base[field] = values[field]
            base[f"{field}_classification"] = classes[field]
            if field == "terminal_or_turnaround_stop" and terminal_value is True:
                source_note = "authoritative ordered route-direction sequence endpoint; one-sided blocking derivation"
            elif field == "planned_itinerary_allows_skip" and planned_value is False:
                source_note = "no post-skip target in any occurrence; one-sided blocking derivation"
            else:
                source_note = "no approved operational rule source; unknown is not interpreted as false"
            base[f"{field}_provenance"] = source_note
        rows.append(base)

    duplicate_keys = len(rows) - len({(row["route_id"], row["direction_id"], row["stop_id"]) for row in rows})
    classification_counts = {
        field: dict(Counter(row[f"{field}_classification"] for row in rows)) for field in STATIC_FIELDS
    }
    for counts in classification_counts.values():
        for classification in CLASSIFICATIONS:
            counts.setdefault(classification, 0)
    completeness = {
        "created_at": iso_kst(),
        "source_occurrence_row_count": int(len(source)),
        "canonical_rulebook_row_count": len(rows),
        "route_direction_count": int(source[["route_id", "direction_id"]].drop_duplicates().shape[0]),
        "source_duplicate_triplet_extra_occurrence_count": int(source.duplicated(["route_id", "direction_id", "stop_id"]).sum()),
        "canonical_duplicate_key_count": duplicate_keys,
        "occurrence_ambiguous_row_count": sum(bool(row["occurrence_context_ambiguous"]) for row in rows),
        "classification_counts_by_field": classification_counts,
        "complete_static_row_count": sum(bool(row["static_state_complete"]) for row in rows),
        "unknown_static_row_count": sum(not bool(row["static_state_complete"]) for row in rows),
        "derived_terminal_block_row_count": sum(row["terminal_or_turnaround_stop"] is True for row in rows),
        "derived_no_post_skip_target_block_row_count": sum(row["planned_itinerary_allows_skip"] is False for row in rows),
        "positive_real_network_skip_count": sum(bool(row["positive_static_skip_clearance"]) for row in rows),
        "all_classifications_in_contract_domain": all(
            row[f"{field}_classification"] in CLASSIFICATIONS for row in rows for field in STATIC_FIELDS
        ),
        "missing_as_false_interpretation_count": 0,
    }
    return rows, completeness


RULEBOOK_COLUMNS = [
    "route_id", "direction_id", "stop_id", "route_no", "stop_name", "node_uid", "stop_orders_json",
    "occurrence_count", "endpoint_occurrence_count", "post_skip_target_exists_any_occurrence",
    "post_skip_target_exists_all_occurrences", "post_skip_target_stop_ids_json", "occurrence_context_ambiguous",
    "mandatory_stop", "mandatory_stop_classification", "mandatory_stop_provenance",
    "protected_stop", "protected_stop_classification", "protected_stop_provenance",
    "planned_itinerary_allows_skip", "planned_itinerary_allows_skip_classification", "planned_itinerary_allows_skip_provenance",
    "terminal_or_turnaround_stop", "terminal_or_turnaround_stop_classification", "terminal_or_turnaround_stop_provenance",
    "charging_or_driver_relief_stop", "charging_or_driver_relief_stop_classification", "charging_or_driver_relief_stop_provenance",
    "effective_scope", "effective_from", "effective_to", "static_state_complete", "unresolved_fields_json",
    "positive_static_skip_clearance", "source_dataset_path", "source_dataset_sha256",
]


@dataclass
class RuntimeVehicle:
    agent_id: int
    vehicle_token: str
    route_key: Tuple[str, str]
    position: int = 0
    remaining_travel_time: float = 0.0
    remaining_dwell_time: float = 0.0


def clear_guard(**overrides: Any) -> StaticGuardStatus:
    values: Dict[str, Any] = {
        "mandatory_stop": False,
        "protected_stop": False,
        "planned_itinerary_allows_skip": True,
        "terminal_or_turnaround_stop": False,
        "charging_or_driver_relief_stop": False,
        "valid_post_skip_path": True,
        "state_complete": True,
        "missing_reason": (),
        "evidence_class": "CONTRACT_FIXED",
    }
    values.update(overrides)
    return StaticGuardStatus(**values)


def guard_from_rulebook(row: Mapping[str, Any]) -> StaticGuardStatus:
    values = {field: row.get(field) for field in STATIC_FIELDS}
    missing = tuple(field.upper() + "_UNKNOWN_FAIL_CLOSED" for field, value in values.items() if value is None)
    valid_path: Optional[bool] = False if row.get("planned_itinerary_allows_skip") is False else None
    if valid_path is None:
        missing = (*missing, "VALID_POST_SKIP_PATH_NOT_POSITIVELY_PROVEN")
    return StaticGuardStatus(
        values["mandatory_stop"],
        values["protected_stop"],
        values["planned_itinerary_allows_skip"],
        values["terminal_or_turnaround_stop"],
        values["charging_or_driver_relief_stop"],
        valid_path,
        False,
        missing,
        "UNKNOWN_FAIL_CLOSED",
    )


def empty_machine(agent_id: int, vehicle_token: str, candidate_stop: str) -> ServiceObligationStateMachine:
    state = ServiceObligationStateMachine()
    state.register_vehicle(agent_id, vehicle_token)
    state.register_stop("CURRENT")
    state.register_stop(candidate_stop)
    state.register_stop("POST")
    return state


def evaluate_mask(
    *,
    test_id: str,
    source_scope: str,
    guard: StaticGuardStatus,
    expected_skip: bool,
    expected_reason: Optional[str] = None,
    machine: Optional[ServiceObligationStateMachine] = None,
    agent_id: int = 0,
    vehicle_token: str = "VEHICLE-0",
    candidate_stop: str = "CANDIDATE",
    decision_ts: int = 0,
    rulebook_key: Optional[str] = None,
) -> Dict[str, Any]:
    state = machine or empty_machine(agent_id, vehicle_token, candidate_stop)
    route_key = ("DRYRUN", test_id)
    routes = {
        route_key: [
            {"stop_id": "CURRENT", "node_uid": "STOP:CURRENT"},
            {"stop_id": candidate_stop, "node_uid": f"STOP:{candidate_stop}"},
            {"stop_id": "POST", "node_uid": "STOP:POST"},
        ]
    }
    result = build_distinct_three_action_mask(
        RuntimeVehicle(agent_id, vehicle_token, route_key),
        routes,
        obligation_state_machine=state,
        decision_ts=decision_ts,
        vehicle_token=vehicle_token,
        static_guard_status=guard,
    )
    reasons = list(result["skip_invalid_reason_codes"])
    passed = result["skip_valid"] is expected_skip and (expected_reason is None or expected_reason in reasons)
    snapshot = result["decision_time_obligation_snapshot"]
    return {
        "test_id": test_id,
        "source_scope": source_scope,
        "rulebook_key": rulebook_key,
        "agent_id": agent_id,
        "vehicle_token": vehicle_token,
        "decision_ts": decision_ts,
        "candidate_stop_id": candidate_stop,
        "expected_skip_valid": expected_skip,
        "observed_skip_valid": bool(result["skip_valid"]),
        "passed": bool(passed),
        "expected_reason": expected_reason,
        "observed_reasons_json": json.dumps(reasons),
        "action_mask_json": json.dumps(result["action_mask"]),
        "dynamic_state_complete": bool(result["dynamic_state_complete"]),
        "static_guard_complete": bool(result["static_guard_state_complete"]),
        "pickup_obligation": bool(snapshot["pickup_obligation"]),
        "dropoff_obligation": bool(snapshot["dropoff_obligation"]),
        "onboard_destination_obligation": bool(snapshot["onboard_destination_obligation"]),
        "service_obligation": bool(snapshot["service_obligation"]),
        "future_event_visible": False,
        "policy_execution_count": 0,
    }


DRYRUN_COLUMNS = [
    "test_id", "source_scope", "rulebook_key", "agent_id", "vehicle_token", "decision_ts", "candidate_stop_id",
    "expected_skip_valid", "observed_skip_valid", "passed", "expected_reason", "observed_reasons_json", "action_mask_json",
    "dynamic_state_complete", "static_guard_complete", "pickup_obligation", "dropoff_obligation",
    "onboard_destination_obligation", "service_obligation", "future_event_visible", "policy_execution_count",
]


def dryrun_rows(rulebook: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = [
        evaluate_mask(test_id="SYNTHETIC_POSITIVE_LEGAL_SKIP", source_scope="SYNTHETIC_CONTRACT_TEST", guard=clear_guard(), expected_skip=True),
        evaluate_mask(test_id="SYNTHETIC_MANDATORY_BLOCK", source_scope="SYNTHETIC_CONTRACT_TEST", guard=clear_guard(mandatory_stop=True), expected_skip=False, expected_reason="MANDATORY_STOP"),
        evaluate_mask(test_id="SYNTHETIC_PROTECTED_BLOCK", source_scope="SYNTHETIC_CONTRACT_TEST", guard=clear_guard(protected_stop=True), expected_skip=False, expected_reason="PROTECTED_STOP"),
        evaluate_mask(test_id="SYNTHETIC_TERMINAL_BLOCK", source_scope="SYNTHETIC_CONTRACT_TEST", guard=clear_guard(terminal_or_turnaround_stop=True), expected_skip=False, expected_reason="TERMINAL_OR_TURNAROUND_STOP"),
        evaluate_mask(test_id="SYNTHETIC_CHARGING_RELIEF_BLOCK", source_scope="SYNTHETIC_CONTRACT_TEST", guard=clear_guard(charging_or_driver_relief_stop=True), expected_skip=False, expected_reason="CHARGING_OR_DRIVER_RELIEF_STOP"),
        evaluate_mask(test_id="SYNTHETIC_INVALID_PATH_BLOCK", source_scope="SYNTHETIC_CONTRACT_TEST", guard=clear_guard(valid_post_skip_path=False), expected_skip=False, expected_reason="DOWNSTREAM_PATH_INVALID"),
        evaluate_mask(test_id="SYNTHETIC_MISSING_STATIC_BLOCK", source_scope="SYNTHETIC_CONTRACT_TEST", guard=StaticGuardStatus(None, None, None, None, None, None, False, ("STATIC_FIELDS_UNKNOWN",), "NOT_AVAILABLE"), expected_skip=False, expected_reason="STATIC_GUARD_STATE_INCOMPLETE"),
    ]

    pickup = empty_machine(0, "VEHICLE-0", "CANDIDATE")
    pickup.passenger_waiting(passenger_id="P-PICKUP", pickup_stop="CANDIDATE", dropoff_stop="POST", event_ts=1)
    pickup.request_created(request_id="R-PICKUP", passenger_id="P-PICKUP", service_leg_id="L-PICKUP", event_ts=2)
    pickup.request_assigned(request_id="R-PICKUP", agent_id=0, vehicle_token="VEHICLE-0", event_ts=3)
    rows.append(evaluate_mask(test_id="SYNTHETIC_PICKUP_BLOCK", source_scope="SYNTHETIC_CONTRACT_TEST", guard=clear_guard(), expected_skip=False, expected_reason="ASSIGNED_PICKUP_REQUEST", machine=pickup, decision_ts=3))

    onboard = empty_machine(0, "VEHICLE-0", "CANDIDATE")
    onboard.passenger_waiting(passenger_id="P-ONBOARD", pickup_stop="CURRENT", dropoff_stop="CANDIDATE", event_ts=1)
    onboard.request_created(request_id="R-ONBOARD", passenger_id="P-ONBOARD", service_leg_id="L-ONBOARD", event_ts=2)
    onboard.request_assigned(request_id="R-ONBOARD", agent_id=0, vehicle_token="VEHICLE-0", event_ts=3)
    onboard.passenger_boarded(request_id="R-ONBOARD", passenger_id="P-ONBOARD", agent_id=0, vehicle_token="VEHICLE-0", stop_id="CURRENT", event_ts=4)
    rows.append(evaluate_mask(test_id="SYNTHETIC_DROPOFF_ONBOARD_BLOCK", source_scope="SYNTHETIC_CONTRACT_TEST", guard=clear_guard(), expected_skip=False, expected_reason="ONBOARD_DROPOFF_DEMAND", machine=onboard, decision_ts=4))

    future = empty_machine(0, "VEHICLE-0", "CANDIDATE")
    future.schedule_transition("passenger_waiting", 11, passenger_id="P-FUTURE", pickup_stop="CANDIDATE", dropoff_stop="POST")
    future.schedule_transition("request_created", 12, request_id="R-FUTURE", passenger_id="P-FUTURE", service_leg_id="L-FUTURE")
    future.schedule_transition("request_assigned", 13, request_id="R-FUTURE", agent_id=0, vehicle_token="VEHICLE-0")
    before = evaluate_mask(test_id="SYNTHETIC_NO_FUTURE_PRE_BOUNDARY", source_scope="SYNTHETIC_CONTRACT_TEST", guard=clear_guard(), expected_skip=True, machine=future, decision_ts=10)
    before["future_event_visible"] = bool(before["service_obligation"])
    before["passed"] = bool(before["passed"] and not before["future_event_visible"])
    rows.append(before)
    rows.append(evaluate_mask(test_id="SYNTHETIC_NO_FUTURE_POST_BOUNDARY", source_scope="SYNTHETIC_CONTRACT_TEST", guard=clear_guard(), expected_skip=False, expected_reason="ASSIGNED_PICKUP_REQUEST", machine=future, decision_ts=13))

    endpoint = next(row for row in rulebook if row["terminal_or_turnaround_stop"] is True)
    rows.append(
        evaluate_mask(
            test_id="REAL_NETWORK_DERIVED_ENDPOINT_BLOCK",
            source_scope="REAL_NETWORK_RULEBOOK",
            guard=guard_from_rulebook(endpoint),
            expected_skip=False,
            expected_reason="TERMINAL_OR_TURNAROUND_STOP",
            candidate_stop=endpoint["stop_id"],
            rulebook_key=f"{endpoint['route_id']}|{endpoint['direction_id']}|{endpoint['stop_id']}",
        )
    )

    index = {(row["route_id"], row["direction_id"], row["stop_id"]): row for row in rulebook}
    mapping = read_json(C2_ROOT / "prospective_8vehicle_mapping_manifest.json")
    for record in mapping["records"]:
        key = (str(record["anchor_route_id"]), str(record["anchor_direction_id"]), str(record["anchor_stop_id"]))
        matched = index.get(key)
        if matched is None:
            raise K5Error(f"C2 anchor missing from K5 rulebook: {key}")
        agent_id = int(record["agent_id"])
        token = str(record["physical_vehicle_token"])
        rows.append(
            evaluate_mask(
                test_id=f"REAL_NETWORK_C2_ANCHOR_AGENT_{agent_id}",
                source_scope="REAL_NETWORK_C2_ANCHOR",
                guard=guard_from_rulebook(matched),
                expected_skip=False,
                expected_reason="STATIC_GUARD_STATE_INCOMPLETE",
                agent_id=agent_id,
                vehicle_token=token,
                candidate_stop=matched["stop_id"],
                rulebook_key="|".join(key),
            )
        )
    return rows


def source_audit(db_audit: Mapping[str, Any], source_integrity: Mapping[str, Any]) -> Dict[str, Any]:
    sources = [ROUTE_SEQUENCE_PATH, K2_ROOT / "stop_safety_rulebook_candidate.parquet", K3_ROOT / "k3_static_rule_source_audit.json", K4_ROOT / "k4_state_machine_contract.json"]
    return {
        "created_at": iso_kst(),
        "source_files": [{"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sources],
        "postgresql_schema_audit": db_audit,
        "k4_source_integrity": source_integrity,
        "new_bis_api_call_count": 0,
        "db_write_count": 0,
        "operational_rule_invention_count": 0,
        "aggregate_demand_promoted_to_individual_state": False,
        "excluded_evidence": [
            {"source": "is_terminal_transition in model feature tables", "reason": "time-bucket transition label, not an effective-time stop operation rule"},
            {"source": "aggregate boardings/alightings/waiting counts", "reason": "cannot prove absence of individual service obligation"},
            {"source": "geographic core/boundary labels", "reason": "cannot prove protected_stop status"},
            {"source": "synthetic K4 fixtures", "reason": "contract tests cannot populate real-network rules"},
        ],
        "field_findings": {
            "mandatory_stop": "UNKNOWN_FAIL_CLOSED for all real-network rows; no approved rule source",
            "protected_stop": "UNKNOWN_FAIL_CLOSED for all real-network rows; geography is not an operating rule",
            "planned_itinerary_allows_skip": "DERIVED_SAFE=false only where no occurrence has a post-skip target; otherwise UNKNOWN_FAIL_CLOSED",
            "terminal_or_turnaround_stop": "DERIVED_SAFE=true for any route-direction endpoint occurrence; non-endpoints remain UNKNOWN_FAIL_CLOSED",
            "charging_or_driver_relief_stop": "UNKNOWN_FAIL_CLOSED for all real-network rows; no duty/depot/charging rule source",
        },
    }


def final_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# PV8-K5 Static Rulebook and K-Action-Mask Readiness Final Report",
            "",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- audit gate: `{summary['gate']}`",
            f"- final decision: `{summary['final_decision']}`",
            f"- source rows / canonical rows: `{summary['source_rows']} / {summary['canonical_rows']}`",
            f"- route-direction groups: `{summary['route_direction_count']}`",
            f"- canonical duplicate keys: `{summary['canonical_duplicate_keys']}`",
            f"- complete static rows: `{summary['complete_rows']}`",
            f"- unresolved rows: `{summary['unresolved_rows']}`",
            f"- derived terminal blockers: `{summary['terminal_blocks']}`",
            f"- derived no-post-target blockers: `{summary['path_blocks']}`",
            f"- positive real-network SKIP count: `{summary['positive_real_network_skip_count']}`",
            f"- K-action-mask dry-run: `{summary['dryrun_passed']}/{summary['dryrun_total']} passed`",
            f"- C2 fixed physical-vehicle anchors checked: `{summary['anchor_checks']}/8`",
            f"- K_action_mask_available: `{summary['K_action_mask_available']}`",
            "- conditional_skip_policy_enabled: `false`",
            "- training_use_authorized: `false`",
            "- checkpoint_reuse_authorized: `false`",
            "- policy_evaluation_authorized: `false`",
            "",
            "## Resolved static evidence",
            "",
            "Ordered route-direction sequences safely identify endpoint blockers and absence of a post-skip target. These are one-sided DERIVED_SAFE blockers only; they do not grant SKIP permission.",
            "",
            "## Unresolved evidence",
            "",
            "No approved effective-time source proves mandatory_stop=false, protected_stop=false, charging_or_driver_relief_stop=false, planned skip permission=true, or absence of interior operational turnaround obligations. Missing evidence remains UNKNOWN_FAIL_CLOSED and is never interpreted as false.",
            "",
            "The database's is_terminal_transition fields are temporal model labels, not route-direction-stop operational rules. Aggregate demand, geographic labels, and synthetic fixtures were not promoted into real-network safety state.",
            "",
            "## Minimum K6 scope",
            "",
            "Obtain or authoritatively approve an operator-owned, effective-time route-direction-stop rulebook covering all five predicates; reconcile repeated-stop occurrences; bind the K4 state machine and K5 lookup into each of the eight fixed vehicle decision paths; serialize it into global snapshots; then rerun prospective no-future mask integration tests. Policy execution, checkpoint reuse, and MAPPO retraining must remain separately locked.",
            "",
            "No BIS/API call, database write, policy/MAPPO execution, checkpoint reuse, K6 execution, or retraining was performed.",
            "",
        ]
    )


def write_manifest_and_lock(writer: Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for relative_path in PAYLOADS:
        path = writer.root / relative_path
        rows.append({"relative_path": relative_path, "size_bytes": path.stat().st_size if path.exists() else None, "sha256": sha256_file(path) if path.exists() else None, "required": True, "artifact_role": Path(relative_path).stem, "exists": path.exists()})
    jsonl_name = "artifact_manifest_srp2_bis_pv8_k5.jsonl"
    writer.text(jsonl_name, "".join(json.dumps(json_clean(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    jsonl_path = writer.root / jsonl_name
    rows.append({"relative_path": jsonl_name, "size_bytes": jsonl_path.stat().st_size, "sha256": sha256_file(jsonl_path), "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest_srp2_bis_pv8_k5.json"
    writer.json(
        manifest_name,
        {
            "created_at": iso_kst(),
            "artifact_family": ARTIFACT_PREFIX,
            "terminal_gate": gate["gate"],
            "readiness": gate["readiness"],
            "payload_count": len(rows),
            "missing_payload_count": sum(not row["exists"] for row in rows),
            "files": rows,
        },
    )
    manifest_path = writer.root / manifest_name
    writer.json(
        "_PV8_K5_COMPLETE.lock",
        {
            "artifact_family": ARTIFACT_PREFIX,
            "terminal_gate": gate["gate"],
            "readiness": gate["readiness"],
            "final_manifest_path": manifest_name,
            "final_manifest_sha256": sha256_file(manifest_path),
            "manifest_size_bytes": manifest_path.stat().st_size,
            "created_at": iso_kst(),
        },
    )


def run_audit(root: Path) -> Path:
    root = validate_artifact_root(root)
    writer = Writer(root)
    upstreams = verify_upstreams()
    source_integrity = k4_source_integrity()
    db_audit = database_static_schema_audit()
    rulebook, completeness = build_rulebook()
    dryruns = dryrun_rows(rulebook)
    dryrun_passed = sum(bool(row["passed"]) for row in dryruns)
    anchor_checks = sum(row["source_scope"] == "REAL_NETWORK_C2_ANCHOR" and row["passed"] for row in dryruns)
    validation = {
        "created_at": iso_kst(),
        "dryrun_case_count": len(dryruns),
        "dryrun_passed_count": dryrun_passed,
        "dryrun_failed_count": len(dryruns) - dryrun_passed,
        "positive_synthetic_skip_test_passed": any(row["test_id"] == "SYNTHETIC_POSITIVE_LEGAL_SKIP" and row["passed"] for row in dryruns),
        "mandatory_protected_terminal_blocks_passed": all(any(row["test_id"] == test and row["passed"] for row in dryruns) for test in ("SYNTHETIC_MANDATORY_BLOCK", "SYNTHETIC_PROTECTED_BLOCK", "SYNTHETIC_TERMINAL_BLOCK")),
        "dynamic_obligation_blocks_passed": all(any(row["test_id"] == test and row["passed"] for row in dryruns) for test in ("SYNTHETIC_PICKUP_BLOCK", "SYNTHETIC_DROPOFF_ONBOARD_BLOCK")),
        "missing_static_blocks_passed": any(row["test_id"] == "SYNTHETIC_MISSING_STATIC_BLOCK" and row["passed"] for row in dryruns),
        "invalid_post_skip_path_blocks_passed": any(row["test_id"] == "SYNTHETIC_INVALID_PATH_BLOCK" and row["passed"] for row in dryruns),
        "no_future_leak_test_passed": all(any(row["test_id"] == test and row["passed"] for row in dryruns) for test in ("SYNTHETIC_NO_FUTURE_PRE_BOUNDARY", "SYNTHETIC_NO_FUTURE_POST_BOUNDARY")),
        "c2_anchor_join_count": anchor_checks,
        "c2_anchor_expected_count": 8,
        "positive_real_network_skip_count": completeness["positive_real_network_skip_count"],
        "policy_execution_count": 0,
        "mappo_execution_count": 0,
    }
    audit_passed = bool(
        source_integrity["source_drift_count"] == 0
        and completeness["canonical_duplicate_key_count"] == 0
        and completeness["all_classifications_in_contract_domain"]
        and dryrun_passed == len(dryruns)
        and anchor_checks == 8
    )
    final_decision = DECISION_BLOCKED
    k_available = bool(final_decision == DECISION_READY and completeness["complete_static_row_count"] == len(rulebook))
    readiness = {
        "created_at": iso_kst(),
        "final_decision": final_decision,
        "K_action_mask_available": k_available,
        "conditional_skip_policy_enabled": False,
        "complete_real_network_rulebook": False,
        "positive_real_network_skip_count": completeness["positive_real_network_skip_count"],
        "unresolved_rows": completeness["unknown_static_row_count"],
        "blocking_fields": ["mandatory_stop", "protected_stop", "planned_itinerary_allows_skip", "terminal_or_turnaround_stop", "charging_or_driver_relief_stop"],
        "minimum_k6_repair_integration_scope": [
            "obtain or operator-approve an effective-time route-direction-stop rulebook for all five static predicates",
            "resolve occurrence-specific semantics where a stop repeats within one route-direction",
            "bind K4 state-machine snapshots and K5 static lookup into all eight fixed vehicle decision paths",
            "serialize state and rulebook version in global snapshots for clone/reset determinism",
            "rerun prospective no-future integration tests before any policy, checkpoint, or training authorization",
        ],
    }
    guard = {
        "created_at": iso_kst(),
        "K_action_mask_available": k_available,
        "conditional_skip_policy_enabled": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_evaluation_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "automatic_k6_execution_authorized": False,
        "automatic_mappo_retraining_authorized": False,
    }
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE if audit_passed else FAIL_GATE,
        "terminal_gate": PASS_GATE if audit_passed else FAIL_GATE,
        "readiness": READINESS if audit_passed else "SRP2_BIS_PV8_K5_AUDIT_FAILED",
        "gate_passed": audit_passed,
        "final_decision": final_decision,
        "failure_reasons": [] if audit_passed else ["upstream/source integrity, canonical key, classification, dry-run, or eight-anchor validation failed"],
    }
    audit = source_audit(db_audit, source_integrity)
    summary = {
        "artifact_root": str(root),
        "gate": gate["gate"],
        "final_decision": final_decision,
        "source_rows": completeness["source_occurrence_row_count"],
        "canonical_rows": completeness["canonical_rulebook_row_count"],
        "route_direction_count": completeness["route_direction_count"],
        "canonical_duplicate_keys": completeness["canonical_duplicate_key_count"],
        "complete_rows": completeness["complete_static_row_count"],
        "unresolved_rows": completeness["unknown_static_row_count"],
        "terminal_blocks": completeness["derived_terminal_block_row_count"],
        "path_blocks": completeness["derived_no_post_skip_target_block_row_count"],
        "positive_real_network_skip_count": completeness["positive_real_network_skip_count"],
        "dryrun_passed": dryrun_passed,
        "dryrun_total": len(dryruns),
        "anchor_checks": anchor_checks,
        "K_action_mask_available": k_available,
    }

    writer.json("k5_static_rule_source_audit.json", audit)
    writer.parquet("k5_static_stop_rulebook.parquet", rulebook, RULEBOOK_COLUMNS)
    writer.json("k5_rulebook_completeness.json", completeness)
    writer.parquet("k5_k_action_mask_dryrun.parquet", dryruns, DRYRUN_COLUMNS)
    writer.json("k5_k_action_mask_validation.json", validation)
    writer.json("k5_readiness_decision.json", readiness)
    writer.json("claim_guard_status.json", guard)
    writer.json(
        "run_manifest.json",
        {
            "created_at": iso_kst(),
            "artifact_family": ARTIFACT_PREFIX,
            "mode": "audit",
            "runner_path": str(RUNNER_PATH),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "upstream_integrity": upstreams,
            "direct_postgresql_query_count": db_audit["direct_postgresql_query_count"],
            "db_write_count": 0,
            "new_bis_api_call_count": 0,
            "policy_execution_count": 0,
            "mappo_execution_count": 0,
            "checkpoint_reuse_count": 0,
            "k6_execution_count": 0,
        },
    )
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guard, "source_gate": gate["gate"], "readiness": gate["readiness"], "final_decision": final_decision})
    writer.text("final_report.md", final_report(summary))
    write_manifest_and_lock(writer, gate)
    own = verify_manifest(root, "artifact_manifest_srp2_bis_pv8_k5.json", "_PV8_K5_COMPLETE.lock")
    if not manifest_ok(own):
        raise K5Error(f"K5 artifact integrity failure: {own}")
    if not audit_passed:
        raise K5Error(f"K5 audit failed: {gate['failure_reasons']}")
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
