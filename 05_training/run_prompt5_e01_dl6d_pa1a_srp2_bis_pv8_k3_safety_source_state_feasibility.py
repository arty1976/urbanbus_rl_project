#!/usr/bin/env python3
"""PV8-K3 K-safety source closure and simulator-state feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k3_safety_source_state_feasibility.py"

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
C3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c3_mappo_interface_compatibility_20260808_100118"
K1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k1_skip_safety_evidence_audit_20260808_101303"
K2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k2_safety_state_contract_20260808_102318"

UPSTREAMS = {
    "PV8-C2": (
        C2_ROOT,
        "artifact_manifest_srp2_bis_pv8_c2.json",
        "_PV8_C2_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED",
    ),
    "PV8-C3": (
        C3_ROOT,
        "artifact_manifest_srp2_bis_pv8_c3.json",
        "_PV8_C3_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_COMPLETE",
    ),
    "PV8-K1": (
        K1_ROOT,
        "artifact_manifest_srp2_bis_pv8_k1.json",
        "_PV8_K1_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K1_SKIP_SAFETY_EVIDENCE_AUDIT_COMPLETE",
    ),
    "PV8-K2": (
        K2_ROOT,
        "artifact_manifest_srp2_bis_pv8_k2.json",
        "_PV8_K2_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K2_SAFETY_STATE_CONTRACT_COMPLETE",
    ),
}

SOURCE_PACK_ROOT = ARTIFACTS_ROOT / "suseong_source_pack_v1"
SERVICE_GRAPH_ROOT = ARTIFACTS_ROOT / "suseong_service_graph_v1"
ROUTE_SEQUENCE_PATH = SERVICE_GRAPH_ROOT / "service_route_sequences.csv"
FULL_ROUTE_SEQUENCE_PATH = SOURCE_PACK_ROOT / "route_stop_sequences.parquet"
SERVICE_EDGES_PATH = SERVICE_GRAPH_ROOT / "service_edges.csv"
ENGINE_PATH = TRAINING_ROOT / "simulator" / "suseong_service_transition_engine.py"
SNAPSHOT_PATH = TRAINING_ROOT / "simulator" / "dynamics_state_snapshot.py"
REPLAY_PATH = TRAINING_ROOT / "simulator" / "dynamics_replay_contract.py"
ORCHESTRATOR_PATH = TRAINING_ROOT / "simulator" / "dynamics_multiagent_orchestrator.py"
EVENT_TRACE_PATH = TRAINING_ROOT / "simulator" / "dynamics_event_trace.py"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k3_safety_source_state_feasibility"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K3_SAFETY_SOURCE_AND_STATE_FEASIBILITY_COMPLETE"
READINESS = "SRP2_BIS_PV8_K3_COMPLETE_DYNAMIC_STATE_FEASIBLE_STATIC_RULE_GAPS_REMAIN_K4_LOCKED"
FINAL_DECISION = "K_SAFETY_IMPLEMENTATION_PARTIAL_STATIC_RULE_GAPS"
FIELD_CLASSES = {"OBSERVED", "DERIVED_SAFE", "SIMULATOR_OWNED_EXACT", "PROXY_GUARDED", "NOT_AVAILABLE"}

PAYLOADS = [
    "k3_static_rule_source_audit.json",
    "k3_dynamic_state_ownership_matrix.json",
    "k3_simulator_state_gap_audit.json",
    "k3_service_obligation_state_machine_contract.json",
    "k3_skip_predicate_v2.json",
    "k3_positive_skip_feasibility.json",
    "k3_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class K3Error(RuntimeError):
    pass


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root

    def text(self, relative_path: str, value: str) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def json(self, relative_path: str, value: Mapping[str, Any]) -> None:
        self.text(
            relative_path,
            json.dumps(json_clean(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        )


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
            if row.get("required", True):
                required_missing += 1
            continue
        if row.get("sha256") and sha256_file(path) != row["sha256"]:
            sha_bad += 1
        if row.get("size_bytes") is not None and path.stat().st_size != row["size_bytes"]:
            size_bad += 1
    manifest_sha = sha256_file(manifest_path)
    checks.update(
        {
            "manifest_hash_ok": manifest_sha == lock.get("manifest_sha256")
            or manifest_sha == lock.get("final_manifest_sha256"),
            "manifest_size_ok": manifest_path.stat().st_size
            == lock.get("manifest_size_bytes", manifest_path.stat().st_size),
            "manifest_entry_count": len(manifest.get("files", [])),
            "missing_files": missing,
            "required_missing_files": required_missing,
            "sha256_mismatches": sha_bad,
            "size_mismatches": size_bad,
            "terminal_lock_binding": lock.get("manifest_relative_path") == manifest_name
            or lock.get("final_manifest_path") == manifest_name,
        }
    )
    return checks


def manifest_ok(checks: Mapping[str, Any]) -> bool:
    return (
        bool(checks.get("manifest_present"))
        and bool(checks.get("lock_present"))
        and bool(checks.get("manifest_hash_ok"))
        and bool(checks.get("manifest_size_ok"))
        and checks.get("missing_files") == 0
        and checks.get("required_missing_files") == 0
        and checks.get("sha256_mismatches") == 0
        and checks.get("size_mismatches") == 0
        and bool(checks.get("terminal_lock_binding"))
    )


def verify_upstreams() -> Dict[str, Any]:
    verified: Dict[str, Any] = {}
    for name, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = read_json(root / "gate_decision.json")
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        checks = verify_manifest(root, manifest_name, lock_name)
        if observed_gate != expected_gate or not manifest_ok(checks):
            raise K3Error(f"{name} integrity failure: gate={observed_gate}, checks={checks}")
        verified[name] = {
            "artifact_root": str(root),
            "gate": observed_gate,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return verified


def source_record(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def classified_field(
    field_name: str,
    classification: str,
    source: str,
    current_value_materialized: bool,
    positive_clearance_capable: bool,
    rationale: str,
    one_sided_guard: str = "",
) -> Dict[str, Any]:
    if classification not in FIELD_CLASSES:
        raise ValueError(f"invalid field classification: {classification}")
    return {
        "field": field_name,
        "classification": classification,
        "source": source,
        "current_value_materialized": bool(current_value_materialized),
        "positive_skip_clearance_capable_now": bool(positive_clearance_capable),
        "rationale": rationale,
        "one_sided_guard": one_sided_guard,
        "unknown_forces_skip_false": True,
    }


def static_rule_source_audit(upstreams: Mapping[str, Any]) -> Dict[str, Any]:
    sequence = pd.read_csv(ROUTE_SEQUENCE_PATH, dtype={"route_id": str, "direction_id": str, "stop_id": str})
    keys = ["route_id", "direction_id"]
    ordered = sequence.sort_values(keys + ["stop_order", "stop_id"]).reset_index(drop=True)
    endpoint_rows = ordered.groupby(keys, dropna=False).tail(1)
    startpoint_rows = ordered.groupby(keys, dropna=False).head(1)
    safety_columns = {
        name: name in sequence.columns
        for name in [
            "mandatory_stop",
            "protected_stop",
            "planned_itinerary_allows_skip",
            "terminal_or_turnaround_stop",
            "charging_or_driver_relief_stop",
        ]
    }
    fields = [
        classified_field(
            "mandatory_stop",
            "NOT_AVAILABLE",
            "no authoritative stop-safety or operation rulebook in existing evidence",
            False,
            False,
            "route and stop master data do not encode a mandatory-service rule",
        ),
        classified_field(
            "protected_stop",
            "NOT_AVAILABLE",
            "no authoritative protected-stop rulebook in existing evidence",
            False,
            False,
            "core/boundary stop classifications are geographic and cannot be promoted to protected service status",
        ),
        classified_field(
            "planned_itinerary_allows_skip",
            "PROXY_GUARDED",
            "ordered route sequence plus service graph path",
            False,
            False,
            "topology can show that a downstream target exists but cannot grant operational permission to skip",
            "missing post-skip target or path blocks skip; path existence does not clear skip",
        ),
        classified_field(
            "terminal_or_turnaround_stop",
            "PROXY_GUARDED",
            "route_stop_sequences route-direction endpoints",
            False,
            False,
            "route endpoints can be positively identified as terminal guards, but interior operational turnaround points cannot be ruled out",
            "known route endpoints block skip; non-endpoint status does not clear skip",
        ),
        classified_field(
            "charging_or_driver_relief_stop",
            "NOT_AVAILABLE",
            "no charging, depot, duty, or driver-relief schedule source in existing evidence",
            False,
            False,
            "neither route topology nor vehicle position proves absence of an operational relief obligation",
        ),
    ]
    return {
        "created_at": iso_kst(),
        "upstream_integrity": upstreams,
        "source_files": [
            source_record(ROUTE_SEQUENCE_PATH),
            source_record(FULL_ROUTE_SEQUENCE_PATH),
            source_record(SERVICE_EDGES_PATH),
            source_record(K2_ROOT / "stop_safety_rulebook_candidate.parquet"),
        ],
        "route_sequence_reader": "service_route_sequences.csv; no optional parquet engine or network install required",
        "k2_full_rulebook_candidate_row_count": int(read_json(K2_ROOT / "run_manifest.json")["rulebook_candidate_row_count"]),
        "route_stop_sequence_row_count": int(len(sequence)),
        "route_direction_count": int(sequence.groupby(keys, dropna=False).ngroups),
        "derived_route_endpoint_row_count": int(len(endpoint_rows)),
        "derived_route_startpoint_row_count": int(len(startpoint_rows)),
        "route_master_columns": [str(column) for column in sequence.columns],
        "required_safety_columns_present": safety_columns,
        "records": fields,
        "classification_counts": {
            classification: sum(row["classification"] == classification for row in fields)
            for classification in sorted(FIELD_CLASSES)
        },
        "static_fields_recovered_for_positive_skip_clearance": [],
        "one_sided_static_guards_recoverable": [
            "route-direction endpoint => terminal guard true",
            "missing post-skip target or graph path => planned itinerary/path guard false",
        ],
        "remaining_static_gaps": [row["field"] for row in fields],
        "aggregate_or_geographic_classification_promoted_to_rulebook_flag": False,
        "direct_postgresql_query_count": 0,
        "db_write_count": 0,
        "new_bis_api_call_count": 0,
    }


def dynamic_state_ownership_matrix() -> Dict[str, Any]:
    records = []
    specs = [
        (
            "onboard_destination_obligation",
            "vehicle obligation ledger keyed by passenger_id/request_id/service_leg_id and destination_stop_id",
            "ReplayEvent has passenger and destination identity; current engine reads optional vehicle attributes but no canonical lifecycle reducer materializes them",
        ),
        (
            "assigned_pickup",
            "request ownership ledger keyed by request_id and pickup_stop_id",
            "orchestrator can arbitrate request ownership; assignment must be persisted into decision-time simulator state",
        ),
        (
            "assigned_dropoff",
            "onboard service-leg ledger keyed by request_id/service_leg_id and destination_stop_id",
            "canonical service-leg identity exists; dropoff obligation still needs state transitions",
        ),
        (
            "boarding_obligation",
            "exact stop queue plus assigned-pickup ledger at decision_ts",
            "must be derived only from simulator-owned passenger/request entities, never aggregate demand",
        ),
        (
            "alighting_obligation",
            "exact onboard destination ledger at decision_ts",
            "must be derived from onboard passenger/service-leg state and destination equality",
        ),
        (
            "service_obligation",
            "logical OR over exact pickup, boarding, dropoff, alighting, onboard destination, and static obligations",
            "becomes exact only after all component ledgers and evidence completeness flags are materialized",
        ),
    ]
    for field_name, target_state, rationale in specs:
        records.append(
            {
                "field": field_name,
                "classification": "SIMULATOR_OWNED_EXACT",
                "ownership_target": "CAUSAL_SIMULATOR_DECISION_TIME_STATE",
                "target_state": target_state,
                "current_value_materialized": False,
                "implementation_required": True,
                "external_observation_claim_allowed": False,
                "aggregate_proxy_promotion_allowed": False,
                "decision_time_exact_after_k4": True,
                "unknown_before_k4_forces_skip_false": True,
                "rationale": rationale,
            }
        )
    return {
        "created_at": iso_kst(),
        "records": records,
        "classification_counts": {"SIMULATOR_OWNED_EXACT": len(records)},
        "all_required_dynamic_fields_simulator_ownable": True,
        "all_required_dynamic_fields_currently_materialized": False,
        "external_passenger_observation_fabricated": False,
        "decision_time_state_semantics": "events with timestamp <= decision_ts are reduced before mask evaluation; later events are invisible",
    }


def source_line_numbers(path: Path, needles: Sequence[str]) -> Dict[str, List[int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {needle: [index for index, line in enumerate(lines, start=1) if needle in line] for needle in needles}


def simulator_state_gap_audit(dynamic_matrix: Mapping[str, Any]) -> Dict[str, Any]:
    source_files = [ENGINE_PATH, SNAPSHOT_PATH, REPLAY_PATH, ORCHESTRATOR_PATH, EVENT_TRACE_PATH]
    engine_refs = source_line_numbers(
        ENGINE_PATH,
        [
            "waiting_pickup_count",
            "assigned_pickup_request_count",
            "assigned_dropoff_request_count",
            "onboard_destination_stop_ids",
            "planned_itinerary_allows_skip",
            "def evaluate_conditional_skip_safety",
        ],
    )
    snapshot_refs = source_line_numbers(
        SNAPSHOT_PATH,
        ["waiting_passengers", "assigned_pickups", "assigned_dropoffs", "onboard_passengers", "mandatory_stop_state"],
    )
    replay_refs = source_line_numbers(
        REPLAY_PATH,
        ["PASSENGER_ARRIVAL", "PICKUP_REQUEST", "PASSENGER_DESTINATION", "PASSENGER_CANCELLATION"],
    )
    orchestrator_refs = source_line_numbers(
        ORCHESTRATOR_PATH,
        ["request_ownership_by_request_id", "CanonicalServiceUnitIdentity", "PASSENGER_BOARD", "ONBOARD_ASSIGNMENT"],
    )
    return {
        "created_at": iso_kst(),
        "source_files": [source_record(path) for path in source_files],
        "current_capabilities": {
            "snapshot_declares_passenger_and_assignment_containers": all(snapshot_refs[key] for key in snapshot_refs),
            "replay_contract_has_passenger_request_identity_events": all(replay_refs[key] for key in replay_refs),
            "orchestrator_has_request_arbitration": bool(orchestrator_refs["request_ownership_by_request_id"]),
            "orchestrator_has_canonical_service_unit_identity": bool(orchestrator_refs["CanonicalServiceUnitIdentity"]),
            "event_trace_can_carry_board_and_alight_identity": True,
            "transition_engine_has_fail_closed_missing_count_reasons": True,
        },
        "current_gaps": {
            "canonical_passenger_request_state_reducer_present": False,
            "replay_events_mutate_snapshot_passenger_ledgers": False,
            "assignment_winner_persisted_to_assigned_pickups": False,
            "board_event_moves_identity_to_onboard_ledger": False,
            "alight_event_closes_service_leg_and_removes_onboard_identity": False,
            "decision_time_obligation_snapshot_materialized_before_action_mask": False,
            "static_rule_evidence_completeness_checked_by_predicate": False,
            "engine_reads_ad_hoc_stop_counts_and_vehicle_attributes": True,
            "missing_static_boolean_uses_false_default_in_engine": True,
        },
        "source_line_evidence": {
            "transition_engine": engine_refs,
            "state_snapshot": snapshot_refs,
            "replay_contract": replay_refs,
            "multiagent_orchestrator": orchestrator_refs,
        },
        "feasibility_judgment": "FEASIBLE_WITH_EXPLICIT_STATE_MACHINE_AND_RULE_EVIDENCE_REPAIR",
        "dynamic_state_can_be_simulator_owned_exact": dynamic_matrix["all_required_dynamic_fields_simulator_ownable"],
        "simulator_execution_count": 0,
        "simulator_source_modified": False,
    }


def state_machine_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract_name": "PV8_K3_SIMULATOR_OWNED_SERVICE_OBLIGATION_STATE_MACHINE_V1",
        "implementation_status": "CONTRACT_ONLY_K4_REQUIRED",
        "entities": {
            "Passenger": ["passenger_id", "arrival_ts", "pickup_stop_id", "destination_stop_id", "status"],
            "ServiceRequest": ["request_id", "passenger_id", "service_leg_id", "pickup_stop_id", "dropoff_stop_id", "assigned_agent_id", "status"],
            "StopQueue": ["stop_id", "waiting_passenger_ids", "assigned_request_ids"],
            "VehicleObligationLedger": ["agent_id", "vehicle_token", "onboard_passenger_ids", "active_service_leg_ids", "destination_by_service_leg"],
            "StaticRuleEvidence": ["route_id", "direction_id", "stop_id", "field", "value", "evidence_class", "effective_from", "effective_to", "source_hash"],
        },
        "state_enums": {
            "Passenger.status": ["NOT_ARRIVED", "WAITING", "ASSIGNED", "ONBOARD", "SERVED", "CANCELLED"],
            "ServiceRequest.status": ["CREATED", "ASSIGNED", "BOARDED", "COMPLETED", "CANCELLED"],
        },
        "transitions": [
            {"event": "PASSENGER_ARRIVAL", "pre": "identity absent", "post": "Passenger=WAITING; add passenger_id to StopQueue"},
            {"event": "PICKUP_REQUEST", "pre": "Passenger in WAITING", "post": "ServiceRequest=CREATED; pickup obligation exact true"},
            {"event": "REQUEST_ASSIGNED", "pre": "ServiceRequest=CREATED", "post": "ServiceRequest=ASSIGNED; persist agent ownership; assigned pickup exact true"},
            {"event": "PASSENGER_BOARD", "pre": "assigned request and matching stop/vehicle", "post": "remove queue/pickup obligation; add onboard identity and destination; ServiceRequest=BOARDED"},
            {"event": "PASSENGER_ALIGHT", "pre": "matching onboard service leg and destination stop", "post": "remove onboard/dropoff obligation; ServiceRequest=COMPLETED; Passenger=SERVED"},
            {"event": "PASSENGER_CANCELLATION", "pre": "not completed", "post": "remove waiting/assigned obligation; mark cancelled; never remove onboard leg silently"},
        ],
        "decision_boundary": {
            "event_inclusion_rule": "apply only events with event_timestamp_seconds <= decision_ts",
            "snapshot_order": ["reduce eligible exogenous events", "complete due service transitions", "freeze obligation snapshot", "build action mask", "choose action"],
            "future_event_access_allowed": False,
            "post_action_outcome_used_for_current_mask": False,
        },
        "derived_exact_fields": {
            "onboard_destination_obligation": "any onboard service leg has dropoff_stop_id == candidate_stop_id",
            "assigned_pickup": "any ASSIGNED request has pickup_stop_id == candidate_stop_id",
            "assigned_dropoff": "any BOARDED service leg has dropoff_stop_id == candidate_stop_id",
            "boarding_obligation": "waiting or assigned pickup identity exists at candidate_stop_id",
            "alighting_obligation": "onboard destination identity exists at candidate_stop_id",
            "service_obligation": "OR(boarding, alighting, assigned_pickup, assigned_dropoff, onboard_destination, mandatory, protected, charging_or_relief)",
        },
        "invariants": [
            "passenger_id, request_id, and service_leg_id are stable and non-empty",
            "one active service leg belongs to at most one vehicle",
            "boarding requires assignment and removes the same pickup obligation exactly once",
            "alighting requires onboard identity and closes the same service leg exactly once",
            "negative obligation is valid only when ledger completeness is true",
            "aggregate demand never creates, clears, or substitutes an individual passenger identity",
            "unknown or inconsistent state raises a safety fault and masks CONDITIONAL_SKIP",
        ],
    }


def skip_predicate_v2() -> Dict[str, Any]:
    static_predicates = [
        "mandatory_stop == false",
        "protected_stop == false",
        "planned_itinerary_allows_skip == true",
        "terminal_or_turnaround_stop == false",
        "charging_or_driver_relief_stop == false",
    ]
    dynamic_predicates = [
        "assigned_pickup == false",
        "assigned_dropoff == false",
        "boarding_obligation == false",
        "alighting_obligation == false",
        "onboard_destination_obligation == false",
        "service_obligation == false",
    ]
    topology_predicates = ["next_stop_exists", "post_skip_target_exists", "graph_edge_or_path_valid"]
    return {
        "created_at": iso_kst(),
        "contract_name": "PV8_K3_FAIL_CLOSED_SKIP_PREDICATE_V2",
        "conditional_skip_enabled": False,
        "K_action_mask_available": False,
        "expression": {
            "all": topology_predicates + static_predicates + dynamic_predicates,
            "all_critical_evidence_classes_in": ["OBSERVED", "DERIVED_SAFE", "SIMULATOR_OWNED_EXACT"],
            "all_required_state_complete": True,
            "if_any_unknown_missing_proxy_or_inconsistent": False,
        },
        "static_predicates": static_predicates,
        "dynamic_predicates": dynamic_predicates,
        "topology_predicates": topology_predicates,
        "fail_closed_rule": "any unknown, missing, proxy-only, stale, future-derived, or inconsistent critical predicate => CONDITIONAL_SKIP=false",
        "missing_is_false_or_zero_interpretation_allowed": False,
        "proxy_can_clear_skip": False,
        "future_leakage_allowed": False,
        "mask_evaluation_time": "decision_ts after eligible event reduction and before action selection",
    }


def positive_skip_feasibility() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "positive_legal_skip_examples_available_now": False,
        "positive_legal_skip_examples_constructible_after_k4_without_future_leakage": True,
        "conditions": [
            "K4 exact passenger/request/service-leg state reducer is implemented and verified",
            "every static predicate has explicit safe evidence or a fixture-owned contract value before decision_ts",
            "all candidate-stop obligation ledgers are complete and exactly empty at decision_ts",
            "topology and post-skip path are valid using only state available at decision_ts",
        ],
        "prospective_test_scenarios": [
            "positive: exact empty ledgers plus fully cleared static rule evidence",
            "negative: waiting or assigned pickup exists",
            "negative: onboard destination or assigned dropoff exists",
            "negative: mandatory/protected/terminal/charging rule is true",
            "negative: each critical field independently unknown or proxy-only",
            "temporal: a passenger arrives after decision_ts and must not affect the current mask",
            "temporal: a passenger arrives at decision_ts and must affect the current mask",
        ],
        "observed_network_positive_skip_claim_allowed": False,
        "synthetic_contract_fixture_claim_scope": "implementation correctness only",
        "future_vehicle_survival_or_post_decision_outcome_used": False,
    }


def readiness_decision(
    static_audit: Mapping[str, Any],
    dynamic_matrix: Mapping[str, Any],
    gap_audit: Mapping[str, Any],
    feasibility: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "final_decision": FINAL_DECISION,
        "static_fields_recovered_for_positive_clearance": static_audit["static_fields_recovered_for_positive_skip_clearance"],
        "one_sided_static_guards_recoverable": static_audit["one_sided_static_guards_recoverable"],
        "remaining_static_gaps": static_audit["remaining_static_gaps"],
        "fields_to_become_simulator_owned_exact": [row["field"] for row in dynamic_matrix["records"]],
        "dynamic_state_implementation_feasible": gap_audit["dynamic_state_can_be_simulator_owned_exact"],
        "current_simulator_state_sufficient_for_skip_enablement": False,
        "minimum_k4_implementation_scope": [
            "implement canonical Passenger, ServiceRequest, StopQueue, and VehicleObligationLedger state",
            "reduce passenger/request events into those ledgers before each decision-time mask",
            "persist request ownership and perform identity-preserving board/alight/cancel transitions",
            "replace ad-hoc stop-row count reads with a typed decision-time obligation snapshot and completeness flags",
            "materialize static rule evidence with provenance/effective time; retain fail-closed unknowns where sources remain absent",
            "implement predicate V2 and tests for positive, each blocking obligation, missing evidence, and decision-time boundaries",
        ],
        "positive_legal_skip_scenarios_constructible_without_future_leakage": feasibility[
            "positive_legal_skip_examples_constructible_after_k4_without_future_leakage"
        ],
        "positive_skip_requires_static_rule_closure_or_explicit_fixture_contract": True,
        "K_action_mask_available": False,
        "conditional_skip_enabled": False,
    }


def claim_guard() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "K_action_mask_available": False,
        "conditional_skip_enabled": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_evaluation_authorized": False,
        "policy_performance_evaluation": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "k4_implementation_authorized": False,
        "automatic_k4_execution_authorized": False,
        "automatic_retraining_authorized": False,
        "observed_passenger_state_claim_allowed": False,
        "new_bis_api_collection_authorized": False,
        "db_write_authorized": False,
    }


def final_report_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# PV8-K3 Safety Source and State Feasibility Final Report",
            "",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- final_decision: `{summary['final_decision']}`",
            f"- static fields recovered for positive clearance: `{summary['static_fields_recovered_for_positive_clearance']}`",
            f"- one-sided static guards recoverable: `{summary['one_sided_static_guards_recoverable']}`",
            f"- remaining static gaps: `{summary['remaining_static_gaps']}`",
            f"- fields that should become simulator-owned exact state: `{summary['fields_to_become_simulator_owned_exact']}`",
            f"- dynamic state implementation feasible: `{summary['dynamic_state_implementation_feasible']}`",
            f"- minimal K4 implementation scope: `{summary['minimum_k4_implementation_scope']}`",
            f"- positive legal-SKIP scenarios constructible without future leakage after K4: `{summary['positive_legal_skip_scenarios_constructible_without_future_leakage']}`",
            f"- K_action_mask_available: `{summary['K_action_mask_available']}`",
            f"- conditional_skip_enabled: `{summary['conditional_skip_enabled']}`",
            f"- training_use_authorized: `{summary['training_use_authorized']}`",
            f"- checkpoint_reuse_authorized: `{summary['checkpoint_reuse_authorized']}`",
            f"- policy_evaluation_authorized: `{summary['policy_evaluation_authorized']}`",
            f"- gate: `{summary['gate']}`",
            f"- readiness: `{summary['readiness']}`",
            "",
            "No complete static field can currently clear a real-network skip. Route endpoints and path existence are retained only as one-sided blocking guards.",
            "",
            "The simulator already has identity/event/snapshot scaffolding, but it does not yet reduce passenger and request events into an exact decision-time obligation ledger. K4 must implement that state machine before any skip action can be enabled.",
            "",
            "No new BIS call, DB query/write, simulator execution, policy evaluation, K4 implementation, retraining, or SKIP enablement was performed.",
            "",
        ]
    )


def write_manifest_and_lock(writer: Writer, gate: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for relative_path in PAYLOADS:
        path = writer.root / relative_path
        rows.append(
            {
                "relative_path": relative_path,
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": sha256_file(path) if path.exists() else None,
                "required": True,
                "artifact_role": Path(relative_path).stem,
                "exists": path.exists(),
            }
        )
    jsonl_name = "artifact_manifest_srp2_bis_pv8_k3.jsonl"
    writer.text(jsonl_name, "".join(json.dumps(json_clean(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    jsonl_path = writer.root / jsonl_name
    rows.append(
        {
            "relative_path": jsonl_name,
            "size_bytes": jsonl_path.stat().st_size,
            "sha256": sha256_file(jsonl_path),
            "required": True,
            "artifact_role": "manifest_jsonl",
            "exists": True,
        }
    )
    manifest = {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "payload_count": len(rows),
        "missing_payload_count": sum(1 for row in rows if not row["exists"]),
        "files": rows,
    }
    manifest_name = "artifact_manifest_srp2_bis_pv8_k3.json"
    writer.json(manifest_name, manifest)
    manifest_path = writer.root / manifest_name
    writer.json(
        "_PV8_K3_COMPLETE.lock",
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
    return manifest


def run_validate(root: Path) -> Path:
    root = validate_artifact_root(root)
    writer = Writer(root)
    upstreams = verify_upstreams()
    static_audit = static_rule_source_audit(upstreams)
    dynamic_matrix = dynamic_state_ownership_matrix()
    gap_audit = simulator_state_gap_audit(dynamic_matrix)
    state_machine = state_machine_contract()
    predicate = skip_predicate_v2()
    feasibility = positive_skip_feasibility()
    decision = readiness_decision(static_audit, dynamic_matrix, gap_audit, feasibility)
    guard = claim_guard()
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": READINESS,
        "gate_passed": True,
        "final_decision": decision["final_decision"],
        "failure_reasons": [],
    }
    summary = {
        "artifact_root": str(root),
        **decision,
        "training_use_authorized": guard["training_use_authorized"],
        "checkpoint_reuse_authorized": guard["checkpoint_reuse_authorized"],
        "policy_evaluation_authorized": guard["policy_evaluation_authorized"],
        "gate": gate["gate"],
        "readiness": gate["readiness"],
    }

    writer.json("k3_static_rule_source_audit.json", static_audit)
    writer.json("k3_dynamic_state_ownership_matrix.json", dynamic_matrix)
    writer.json("k3_simulator_state_gap_audit.json", gap_audit)
    writer.json("k3_service_obligation_state_machine_contract.json", state_machine)
    writer.json("k3_skip_predicate_v2.json", predicate)
    writer.json("k3_positive_skip_feasibility.json", feasibility)
    writer.json("k3_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guard)
    writer.json(
        "run_manifest.json",
        {
            "created_at": iso_kst(),
            "artifact_family": ARTIFACT_PREFIX,
            "mode": "validate",
            "runner_path": str(RUNNER_PATH),
            "runner_sha256": sha256_file(RUNNER_PATH),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "new_bis_api_call_count": 0,
            "direct_postgresql_query_count": 0,
            "db_write_count": 0,
            "simulator_execution_count": 0,
            "policy_evaluation_count": 0,
            "simulator_source_modified": False,
            "k4_implementation_count": 0,
            "training_use_authorized": False,
            "checkpoint_reuse_authorized": False,
        },
    )
    writer.json("gate_decision.json", gate)
    writer.json(
        "downstream_lock.json",
        {
            "created_at": iso_kst(),
            "source_gate": gate["gate"],
            "readiness": gate["readiness"],
            "final_decision": decision["final_decision"],
            "K_action_mask_available": False,
            "conditional_skip_enabled": False,
            "training_use_authorized": False,
            "checkpoint_reuse_authorized": False,
            "policy_evaluation_authorized": False,
            "causal_performance_claim_allowed": False,
            "paper_level_claim_allowed": False,
            "automatic_k4_execution_authorized": False,
            "automatic_retraining_authorized": False,
        },
    )
    writer.text("final_report.md", final_report_text(summary))
    write_manifest_and_lock(writer, gate)
    own_checks = verify_manifest(root, "artifact_manifest_srp2_bis_pv8_k3.json", "_PV8_K3_COMPLETE.lock")
    if not manifest_ok(own_checks):
        raise K3Error(f"ARTIFACT_INTEGRITY_FAILURE: {own_checks}")

    for key in [
        "artifact_root",
        "final_decision",
        "static_fields_recovered_for_positive_clearance",
        "one_sided_static_guards_recoverable",
        "remaining_static_gaps",
        "fields_to_become_simulator_owned_exact",
        "dynamic_state_implementation_feasible",
        "positive_legal_skip_scenarios_constructible_without_future_leakage",
        "K_action_mask_available",
        "conditional_skip_enabled",
        "training_use_authorized",
        "checkpoint_reuse_authorized",
        "policy_evaluation_authorized",
        "gate",
        "readiness",
    ]:
        print(f"{key}: {summary[key]}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run_validate(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
