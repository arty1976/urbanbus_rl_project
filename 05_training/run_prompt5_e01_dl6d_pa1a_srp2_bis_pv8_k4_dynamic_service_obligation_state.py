#!/usr/bin/env python3
"""PV8-K4 simulator-owned passenger and service-obligation state implementation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import resource
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from simulator.k_safety_state import (
    DecisionTimeObligationSnapshot,
    KSafetyChronologyError,
    Passenger,
    PassengerStatus,
    RequestStatus,
    ServiceObligationStateMachine,
    ServiceRequest,
    StaticGuardStatus,
    StopQueue,
    VehicleObligationLedger,
)
from simulator.suseong_service_transition_engine import (
    ENGINE_ACTION_CONDITIONAL_SKIP,
    InvalidConditionalSkipError,
    StopServiceResult,
    TransitionConfig,
    advance_vehicle_time_budget,
    build_distinct_three_action_mask,
)


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state.py"
STATE_SOURCE = TRAINING_ROOT / "simulator" / "k_safety_state.py"
ENGINE_SOURCE = TRAINING_ROOT / "simulator" / "suseong_service_transition_engine.py"
K4_TEST_SOURCE = TRAINING_ROOT / "simulator" / "test_pv8_k4_service_obligation_state.py"
DL6C_TEST_SOURCE = TRAINING_ROOT / "simulator" / "test_dl6c_conditional_skip_safety.py"
BASE_ENGINE_TEST_SOURCE = TRAINING_ROOT / "simulator" / "test_suseong_service_transition_engine.py"
ER1_ENGINE_TEST_SOURCE = TRAINING_ROOT / "test_dl6d_pa1a_er1_engine_contract_repair.py"
PROVENANCE_TEST_SOURCE = TRAINING_ROOT / "test_dl6d_pa1a_provenance_and_dynamics_feasibility.py"
SPLIT_INVENTORY_TEST_SOURCE = TRAINING_ROOT / "test_dl6d_r3_r1_split_inventory.py"

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
C3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c3_mappo_interface_compatibility_20260808_100118"
K1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k1_skip_safety_evidence_audit_20260808_101303"
K2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k2_safety_state_contract_20260808_102318"
K3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k3_safety_source_state_feasibility_20260808_112054"

UPSTREAMS = {
    "PV8-C2": (C2_ROOT, "artifact_manifest_srp2_bis_pv8_c2.json", "_PV8_C2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"),
    "PV8-C3": (C3_ROOT, "artifact_manifest_srp2_bis_pv8_c3.json", "_PV8_C3_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_COMPLETE"),
    "PV8-K1": (K1_ROOT, "artifact_manifest_srp2_bis_pv8_k1.json", "_PV8_K1_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K1_SKIP_SAFETY_EVIDENCE_AUDIT_COMPLETE"),
    "PV8-K2": (K2_ROOT, "artifact_manifest_srp2_bis_pv8_k2.json", "_PV8_K2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K2_SAFETY_STATE_CONTRACT_COMPLETE"),
    "PV8-K3": (K3_ROOT, "artifact_manifest_srp2_bis_pv8_k3.json", "_PV8_K3_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K3_SAFETY_SOURCE_AND_STATE_FEASIBILITY_COMPLETE"),
}

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"
FAIL_GATE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_INCOMPLETE"
PASS_READINESS = "SRP2_BIS_PV8_K4_COMPLETE_DYNAMIC_STATE_IMPLEMENTED_STATIC_RULES_GATED_K5_PENDING"
FAIL_READINESS = "SRP2_BIS_PV8_K4_BLOCKED_IMPLEMENTATION_OR_TEST_FAILURE"
PASS_DECISION = "K_SAFETY_DYNAMIC_STATE_IMPLEMENTED_STATIC_RULES_GATED"
FAIL_DECISION = "K_SAFETY_IMPLEMENTATION_INCOMPLETE"

PAYLOADS = [
    "k4_state_machine_contract.json",
    "k4_transition_test_results.parquet",
    "k4_decision_snapshot_samples.parquet",
    "k4_skip_predicate_test_results.parquet",
    "k4_state_integrity_audit.json",
    "k4_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class K4Error(RuntimeError):
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
        self.text(relative_path, json.dumps(json_clean(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")

    def parquet(self, relative_path: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([json_clean(dict(row)) for row in rows], columns=list(columns)).to_parquet(path, index=False)


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
            "manifest_hash_ok": manifest_sha == lock.get("manifest_sha256") or manifest_sha == lock.get("final_manifest_sha256"),
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
            raise K4Error(f"{name} integrity failure: gate={observed_gate}, checks={checks}")
        out[name] = {
            "artifact_root": str(root),
            "gate": observed_gate,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return out


@dataclass
class RuntimeVehicle:
    agent_id: int = 0
    vehicle_token: str = "VEHICLE-0"
    route_key: Tuple[str, str] = ("R_TEST", "1")
    position: int = 0
    remaining_travel_time: float = 0.0
    remaining_dwell_time: float = 0.0


def clear_stop(stop_id: str, **overrides: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "stop_id": stop_id,
        "node_uid": f"STOP:{stop_id}",
        "mandatory_stop": False,
        "protected_stop": False,
        "planned_itinerary_allows_skip": True,
        "terminal_or_turnaround_stop": False,
        "charging_or_driver_relief_stop": False,
        "downstream_path_valid": True,
        "graph_edge_or_path_valid": True,
    }
    row.update(overrides)
    return row


def fixture_routes(next_stop: Optional[Mapping[str, Any]] = None) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    return {
        ("R_TEST", "1"): [
            {"stop_id": "S0", "node_uid": "STOP:S0"},
            dict(next_stop) if next_stop is not None else clear_stop("S1"),
            {"stop_id": "S2", "node_uid": "STOP:S2"},
        ]
    }


def empty_machine(*, register_candidate: bool = True) -> ServiceObligationStateMachine:
    state = ServiceObligationStateMachine()
    state.register_vehicle(0, "VEHICLE-0")
    state.register_stop("S0")
    if register_candidate:
        state.register_stop("S1")
    state.register_stop("S2")
    return state


def add_assigned_pickup(state: ServiceObligationStateMachine, *, pickup_stop: str = "S1", dropoff_stop: str = "S2") -> None:
    state.passenger_waiting(passenger_id="P1", pickup_stop=pickup_stop, dropoff_stop=dropoff_stop, event_ts=1)
    state.request_created(request_id="R1", passenger_id="P1", service_leg_id="L1", event_ts=2)
    state.request_assigned(request_id="R1", agent_id=0, vehicle_token="VEHICLE-0", event_ts=3)


def add_onboard_dropoff(state: ServiceObligationStateMachine, *, dropoff_stop: str = "S1") -> None:
    add_assigned_pickup(state, pickup_stop="S0", dropoff_stop=dropoff_stop)
    state.passenger_boarded(
        request_id="R1",
        passenger_id="P1",
        agent_id=0,
        vehicle_token="VEHICLE-0",
        stop_id="S0",
        event_ts=4,
    )


def make_mask(
    state: Optional[ServiceObligationStateMachine],
    *,
    next_stop: Optional[Mapping[str, Any]] = None,
    decision_ts: int = 0,
    vehicle: Optional[RuntimeVehicle] = None,
) -> Dict[str, Any]:
    runtime_vehicle = vehicle or RuntimeVehicle()
    return build_distinct_three_action_mask(
        runtime_vehicle,
        fixture_routes(next_stop),
        obligation_state_machine=state,
        decision_ts=int(decision_ts),
        vehicle_token=runtime_vehicle.vehicle_token,
    )


def transition_test_rows() -> Tuple[List[Dict[str, Any]], List[ServiceObligationStateMachine]]:
    rows: List[Dict[str, Any]] = []
    machines: List[ServiceObligationStateMachine] = []
    state = empty_machine()
    machines.append(state)

    def add_row(
        test_id: str,
        transition: str,
        event_ts: int,
        expected_passenger: Optional[str],
        expected_request: Optional[str],
        *,
        passed: bool = True,
        details: str = "",
    ) -> None:
        passenger = state.passengers.get("P1")
        request = state.requests.get("R1")
        observed_passenger = passenger.passenger_status.value if passenger is not None else None
        observed_request = request.request_status.value if request is not None else None
        identity_preserved = bool(
            passenger is None
            or request is None
            or (
                request.passenger_id == passenger.passenger_id
                and request.service_leg_id == "L1"
                and passenger.active_request_id in {None, request.request_id}
            )
        )
        rows.append(
            {
                "test_id": test_id,
                "transition": transition,
                "passed": bool(passed and observed_passenger == expected_passenger and observed_request == expected_request and identity_preserved),
                "passenger_id": "P1" if passenger is not None else None,
                "request_id": "R1" if request is not None else None,
                "service_leg_id": request.service_leg_id if request is not None else None,
                "event_ts": int(event_ts),
                "expected_passenger_status": expected_passenger,
                "observed_passenger_status": observed_passenger,
                "expected_request_status": expected_request,
                "observed_request_status": observed_request,
                "identity_preserved": identity_preserved,
                "chronology_preserved": bool(not state.event_log or all(state.event_log[index]["event_ts"] <= state.event_log[index + 1]["event_ts"] for index in range(len(state.event_log) - 1))),
                "details": details,
            }
        )

    state.passenger_waiting(passenger_id="P1", pickup_stop="S1", dropoff_stop="S2", event_ts=1)
    add_row("T01", "passenger_waiting", 1, "WAITING", None)
    state.request_created(request_id="R1", passenger_id="P1", service_leg_id="L1", event_ts=2)
    add_row("T02", "request_created", 2, "WAITING", "CREATED")
    state.request_assigned(request_id="R1", agent_id=0, vehicle_token="VEHICLE-0", event_ts=3)
    add_row("T03", "request_assigned", 3, "ASSIGNED", "ASSIGNED")
    state.passenger_boarded(request_id="R1", passenger_id="P1", agent_id=0, vehicle_token="VEHICLE-0", stop_id="S1", event_ts=4)
    add_row("T04", "passenger_boarded", 4, "ONBOARD", "BOARDED")
    state.passenger_alighted(request_id="R1", passenger_id="P1", agent_id=0, vehicle_token="VEHICLE-0", stop_id="S2", event_ts=5)
    add_row("T05", "passenger_alighted", 5, "ALIGHTED", "ALIGHTED")
    state.request_completed(request_id="R1", event_ts=6)
    add_row("T06", "request_completed", 6, "SERVED", "COMPLETED")

    cancelled = empty_machine()
    machines.append(cancelled)
    add_assigned_pickup(cancelled)
    cancelled.request_cancelled(request_id="R1", event_ts=4)
    rows.append(
        {
            "test_id": "T07",
            "transition": "request_cancelled",
            "passed": bool(
                cancelled.passengers["P1"].passenger_status == PassengerStatus.CANCELLED
                and cancelled.requests["R1"].request_status == RequestStatus.CANCELLED
                and not cancelled.vehicle_ledgers[0].assigned_pickup
                and not cancelled.stop_queues["S1"].waiting_queue
            ),
            "passenger_id": "P1",
            "request_id": "R1",
            "service_leg_id": "L1",
            "event_ts": 4,
            "expected_passenger_status": "CANCELLED",
            "observed_passenger_status": cancelled.passengers["P1"].passenger_status.value,
            "expected_request_status": "CANCELLED",
            "observed_request_status": cancelled.requests["R1"].request_status.value,
            "identity_preserved": cancelled.requests["R1"].passenger_id == "P1",
            "chronology_preserved": True,
            "details": "cancel removes waiting and assigned pickup without fabricating completion",
        }
    )

    reassigned = empty_machine()
    machines.append(reassigned)
    reassigned.register_vehicle(1, "VEHICLE-1")
    add_assigned_pickup(reassigned)
    reassigned.request_reassigned(request_id="R1", agent_id=1, vehicle_token="VEHICLE-1", event_ts=4)
    rows.append(
        {
            "test_id": "T08",
            "transition": "request_reassigned",
            "passed": bool("R1" not in reassigned.vehicle_ledgers[0].assigned_pickup and "R1" in reassigned.vehicle_ledgers[1].assigned_pickup and reassigned.requests["R1"].passenger_id == "P1"),
            "passenger_id": "P1",
            "request_id": "R1",
            "service_leg_id": "L1",
            "event_ts": 4,
            "expected_passenger_status": "ASSIGNED",
            "observed_passenger_status": reassigned.passengers["P1"].passenger_status.value,
            "expected_request_status": "ASSIGNED",
            "observed_request_status": reassigned.requests["R1"].request_status.value,
            "identity_preserved": reassigned.requests["R1"].passenger_id == "P1",
            "chronology_preserved": True,
            "details": "ownership moves from VEHICLE-0 to VEHICLE-1 exactly once",
        }
    )
    chronology_blocked = False
    event_count_before = len(reassigned.event_log)
    try:
        reassigned.request_cancelled(request_id="R1", event_ts=3)
    except KSafetyChronologyError:
        chronology_blocked = True
    rows.append(
        {
            "test_id": "T09",
            "transition": "chronology_guard",
            "passed": bool(chronology_blocked and len(reassigned.event_log) == event_count_before),
            "passenger_id": "P1",
            "request_id": "R1",
            "service_leg_id": "L1",
            "event_ts": 3,
            "expected_passenger_status": "ASSIGNED",
            "observed_passenger_status": reassigned.passengers["P1"].passenger_status.value,
            "expected_request_status": "ASSIGNED",
            "observed_request_status": reassigned.requests["R1"].request_status.value,
            "identity_preserved": True,
            "chronology_preserved": chronology_blocked,
            "details": "past transition rejected before mutation",
        }
    )
    return rows, machines


def snapshot_row(sample_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    static = payload["static_guard_status"]
    return {
        "sample_id": sample_id,
        "agent_id": payload["agent_id"],
        "vehicle_token": payload["vehicle_token"],
        "decision_ts": payload["decision_ts"],
        "current_stop_id": payload["current_stop_id"],
        "candidate_stop_id": payload["candidate_stop_id"],
        "pickup_obligation": payload["pickup_obligation"],
        "dropoff_obligation": payload["dropoff_obligation"],
        "boarding_obligation": payload["boarding_obligation"],
        "alighting_obligation": payload["alighting_obligation"],
        "onboard_destination_obligation": payload["onboard_destination_obligation"],
        "service_obligation": payload["service_obligation"],
        "assigned_pickup": payload["assigned_pickup"],
        "assigned_dropoff": payload["assigned_dropoff"],
        "dynamic_state_complete": payload["dynamic_state_complete"],
        "static_guard_complete": static["state_complete"],
        "static_guard_clear": static["positively_clear"],
        "state_complete": payload["state_complete"],
        "skip_allowed": payload["skip_allowed"],
        "waiting_queue": json.dumps(payload["waiting_queue"], ensure_ascii=False),
        "missing_reason": json.dumps(payload["missing_reason"], ensure_ascii=False),
    }


def predicate_and_snapshot_tests() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    predicate_rows: List[Dict[str, Any]] = []
    snapshot_rows: List[Dict[str, Any]] = []
    engine_execution_count = 0

    def record(
        test_id: str,
        state: Optional[ServiceObligationStateMachine],
        *,
        expected_skip: bool,
        expected_reason: str = "",
        next_stop: Optional[Mapping[str, Any]] = None,
        decision_ts: int = 0,
        future_event_visible: Optional[bool] = None,
        execute: bool = False,
    ) -> Dict[str, Any]:
        nonlocal engine_execution_count
        vehicle = RuntimeVehicle()
        result = make_mask(state, next_stop=next_stop, decision_ts=decision_ts, vehicle=vehicle)
        reasons = list(result["skip_invalid_reason_codes"])
        snapshot = dict(result["decision_time_obligation_snapshot"])
        execution_attempted = False
        execution_succeeded = False
        if execute:
            execution_attempted = True
            engine_execution_count += 1
            try:
                trace = advance_vehicle_time_budget(
                    vehicle=vehicle,
                    routes=fixture_routes(next_stop),
                    delta_t_seconds=1.0,
                    action=ENGINE_ACTION_CONDITIONAL_SKIP,
                    stop_service=lambda _vehicle, _stop: StopServiceResult(),
                    config=TransitionConfig(edge_travel_seconds=0.0, dwell_seconds=0.0),
                    obligation_state_machine=state,
                    decision_ts=decision_ts,
                    vehicle_token=vehicle.vehicle_token,
                )
                execution_succeeded = bool(trace.skip_executed)
            except InvalidConditionalSkipError:
                execution_succeeded = False
        passed = bool(result["skip_valid"] is expected_skip)
        if expected_reason:
            passed = bool(passed and expected_reason in reasons)
        if execute:
            passed = bool(passed and execution_succeeded is expected_skip)
        row = {
            "test_id": test_id,
            "expected_skip_valid": expected_skip,
            "observed_skip_valid": bool(result["skip_valid"]),
            "passed": passed,
            "expected_reason": expected_reason or None,
            "observed_reasons": json.dumps(reasons, ensure_ascii=False),
            "dynamic_state_complete": bool(result["dynamic_state_complete"]),
            "static_guard_complete": bool(result["static_guard_state_complete"]),
            "snapshot_present": result["decision_time_obligation_snapshot"] is not None,
            "future_event_visible": future_event_visible,
            "engine_execution_attempted": execution_attempted,
            "engine_execution_succeeded": execution_succeeded,
            "position_after": int(vehicle.position),
        }
        predicate_rows.append(row)
        snapshot_rows.append(snapshot_row(test_id, snapshot))
        return result

    record("P01_POSITIVE_LEGAL_SKIP", empty_machine(), expected_skip=True, execute=True)

    assigned = empty_machine()
    add_assigned_pickup(assigned)
    record("P02_PICKUP_BLOCK", assigned, expected_skip=False, expected_reason="ASSIGNED_PICKUP_REQUEST", decision_ts=3)

    waiting = empty_machine()
    waiting.passenger_waiting(passenger_id="P1", pickup_stop="S1", dropoff_stop="S2", event_ts=1)
    record("P03_BOARDING_BLOCK", waiting, expected_skip=False, expected_reason="WAITING_PICKUP_DEMAND", decision_ts=1)

    onboard = empty_machine()
    add_onboard_dropoff(onboard)
    record("P04_DROPOFF_BLOCK", onboard, expected_skip=False, expected_reason="ASSIGNED_DROPOFF_REQUEST", decision_ts=4)
    record("P05_ONBOARD_DESTINATION_BLOCK", onboard, expected_skip=False, expected_reason="ONBOARD_DROPOFF_DEMAND", decision_ts=4)
    record("P06_ALIGHTING_BLOCK", onboard, expected_skip=False, expected_reason="ONBOARD_DROPOFF_DEMAND", decision_ts=4)

    record("P07_MISSING_STATE_BLOCK", empty_machine(register_candidate=False), expected_skip=False, expected_reason="DYNAMIC_STATE_INCOMPLETE")
    record("P08_MISSING_MACHINE_BLOCK", None, expected_skip=False, expected_reason="DYNAMIC_STATE_MACHINE_MISSING")
    record("P09_TERMINAL_BLOCK", empty_machine(), expected_skip=False, expected_reason="TERMINAL_OR_TURNAROUND_STOP", next_stop=clear_stop("S1", terminal_or_turnaround_stop=True))
    record("P10_PATH_BLOCK", empty_machine(), expected_skip=False, expected_reason="DOWNSTREAM_PATH_INVALID", next_stop=clear_stop("S1", downstream_path_valid=False))
    record("P11_UNKNOWN_STATIC_BLOCK", empty_machine(), expected_skip=False, expected_reason="STATIC_GUARD_STATE_INCOMPLETE", next_stop={"stop_id": "S1", "node_uid": "STOP:S1"})

    future = empty_machine()
    future.schedule_transition("passenger_waiting", 101, passenger_id="P_FUTURE", pickup_stop="S1", dropoff_stop="S2")
    before = record("P12_NO_FUTURE_BEFORE_BOUNDARY", future, expected_skip=True, decision_ts=100, future_event_visible=False)
    at = record("P13_EVENT_VISIBLE_AT_BOUNDARY", future, expected_skip=False, expected_reason="WAITING_PICKUP_DEMAND", decision_ts=101, future_event_visible=True)
    predicate_rows[-2]["passed"] = bool(predicate_rows[-2]["passed"] and before["decision_time_obligation_snapshot"]["waiting_queue"] == [])
    predicate_rows[-1]["passed"] = bool(predicate_rows[-1]["passed"] and at["decision_time_obligation_snapshot"]["waiting_queue"] == ["P_FUTURE"])
    return predicate_rows, snapshot_rows, engine_execution_count


def run_pytest_regression() -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        str(BASE_ENGINE_TEST_SOURCE),
        str(DL6C_TEST_SOURCE),
        str(K4_TEST_SOURCE),
        str(ER1_ENGINE_TEST_SOURCE),
        str(PROVENANCE_TEST_SOURCE),
        str(SPLIT_INVENTORY_TEST_SOURCE),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(TRAINING_ROOT / "simulator"), str(TRAINING_ROOT), env.get("PYTHONPATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = "/private/tmp/pv8_k4_pycache"
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False)
    output = (result.stdout + result.stderr).strip()
    match = re.search(r"(\d+) passed", output)
    return {
        "command": command,
        "returncode": int(result.returncode),
        "passed": result.returncode == 0,
        "passed_test_count": int(match.group(1)) if match else None,
        "output": output,
    }


def state_machine_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract_name": "PV8_K4_SIMULATOR_OWNED_SERVICE_OBLIGATION_STATE_MACHINE_V2",
        "implementation_status": "IMPLEMENTED_AND_TESTED",
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in [STATE_SOURCE, ENGINE_SOURCE, K4_TEST_SOURCE, DL6C_TEST_SOURCE]
        ],
        "entities": {
            "Passenger": list(Passenger.__dataclass_fields__),
            "ServiceRequest": list(ServiceRequest.__dataclass_fields__),
            "StopQueue": list(StopQueue.__dataclass_fields__),
            "VehicleObligationLedger": list(VehicleObligationLedger.__dataclass_fields__),
            "DecisionTimeObligationSnapshot": list(DecisionTimeObligationSnapshot.__dataclass_fields__),
        },
        "required_state_mapping": {
            "passenger_id": "Passenger.passenger_id",
            "request_id": "ServiceRequest.request_id",
            "pickup_stop": "Passenger/ServiceRequest.pickup_stop",
            "dropoff_stop": "Passenger/ServiceRequest.dropoff_stop",
            "assigned_vehicle": "Passenger/ServiceRequest.assigned_vehicle plus VehicleObligationLedger",
            "passenger_status": "Passenger.passenger_status",
            "waiting_queue": "StopQueue.waiting_queue",
            "assigned_pickup": "StopQueue and VehicleObligationLedger assigned_pickup",
            "assigned_dropoff": "VehicleObligationLedger.assigned_dropoff",
            "onboard_destination": "VehicleObligationLedger.onboard_destination",
            "boarding_obligation": "DecisionTimeObligationSnapshot.boarding_obligation",
            "alighting_obligation": "DecisionTimeObligationSnapshot.alighting_obligation",
            "service_obligation": "DecisionTimeObligationSnapshot.service_obligation",
            "state_complete": "StopQueue/VehicleObligationLedger/DecisionTimeObligationSnapshot state_complete",
            "missing_reason": "typed state and snapshot missing_reason",
        },
        "implemented_transitions": [
            "passenger_waiting",
            "request_created",
            "request_assigned",
            "request_reassigned",
            "passenger_boarded",
            "passenger_alighted",
            "request_completed",
            "request_cancelled",
        ],
        "decision_time_order": [
            "apply scheduled transitions where event_ts <= decision_ts",
            "freeze typed DecisionTimeObligationSnapshot",
            "evaluate dynamic completeness and obligations",
            "evaluate static guard completeness and positive clearance",
            "build fail-closed action mask",
        ],
        "future_information_access_allowed": False,
        "aggregate_demand_promoted_to_passenger_identity": False,
        "ad_hoc_dynamic_stop_row_count_fallback_allowed": False,
        "missing_state_interpreted_as_zero": False,
    }


def integrity_audit(
    transition_rows: Sequence[Mapping[str, Any]],
    predicate_rows: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    machines: Sequence[ServiceObligationStateMachine],
    pytest_result: Mapping[str, Any],
) -> Dict[str, Any]:
    engine_text = ENGINE_SOURCE.read_text(encoding="utf-8")
    forbidden_dynamic_reads = [
        r'next_stop\.get\("waiting_pickup_count"',
        r'next_stop\.get\("scheduled_alighting_count"',
        r'next_stop\.get\("assigned_pickup_request_count"',
        r'next_stop\.get\("assigned_dropoff_request_count"',
        r'getattr\(vehicle, "onboard_destination_stop_ids"',
        r'getattr\(vehicle, "passenger_destinations"',
    ]
    matches = {pattern: bool(re.search(pattern, engine_text)) for pattern in forbidden_dynamic_reads}
    machine_audits = [machine.audit_integrity() for machine in machines]
    transition_failures = [row["test_id"] for row in transition_rows if not row["passed"]]
    predicate_failures = [row["test_id"] for row in predicate_rows if not row["passed"]]
    snapshot_missing = [row["sample_id"] for row in snapshots if row.get("agent_id") is None]
    return {
        "created_at": iso_kst(),
        "entity_classes_present": all([Passenger, ServiceRequest, StopQueue, VehicleObligationLedger, DecisionTimeObligationSnapshot]),
        "transition_test_count": len(transition_rows),
        "transition_failure_count": len(transition_failures),
        "transition_failures": transition_failures,
        "skip_predicate_test_count": len(predicate_rows),
        "skip_predicate_failure_count": len(predicate_failures),
        "skip_predicate_failures": predicate_failures,
        "snapshot_sample_count": len(snapshots),
        "snapshot_missing_count": len(snapshot_missing),
        "snapshot_missing_samples": snapshot_missing,
        "all_state_machine_integrity_audits_passed": all(row["passed"] for row in machine_audits),
        "state_machine_integrity_audits": machine_audits,
        "forbidden_ad_hoc_dynamic_read_matches": matches,
        "ad_hoc_dynamic_stop_row_read_count": sum(matches.values()),
        "typed_snapshot_generated_for_every_mask_test": all(row["snapshot_present"] for row in predicate_rows),
        "positive_legal_skip_test_passed": any(row["test_id"] == "P01_POSITIVE_LEGAL_SKIP" and row["passed"] for row in predicate_rows),
        "no_future_leak_tests_passed": all(row["passed"] for row in predicate_rows if row["test_id"] in {"P12_NO_FUTURE_BEFORE_BOUNDARY", "P13_EVENT_VISIBLE_AT_BOUNDARY"}),
        "identity_chronology_tests_passed": all(row["identity_preserved"] and row["chronology_preserved"] for row in transition_rows),
        "pytest_regression": pytest_result,
        "simulator_source_modified_for_k4": True,
        "policy_use_enabled": False,
    }


def claim_guard() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "K_action_mask_available": False,
        "conditional_skip_policy_enabled": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_evaluation_authorized": False,
        "policy_performance_evaluation": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "observed_passenger_state_claim_allowed": False,
        "automatic_k5_execution_authorized": False,
        "automatic_mappo_retraining_authorized": False,
        "new_bis_api_collection_authorized": False,
        "db_write_authorized": False,
    }


def readiness_decision(integrity: Mapping[str, Any]) -> Dict[str, Any]:
    k3 = read_json(K3_ROOT / "k3_readiness_decision.json")
    passed = bool(
        integrity["transition_failure_count"] == 0
        and integrity["skip_predicate_failure_count"] == 0
        and integrity["snapshot_missing_count"] == 0
        and integrity["all_state_machine_integrity_audits_passed"]
        and integrity["ad_hoc_dynamic_stop_row_read_count"] == 0
        and integrity["typed_snapshot_generated_for_every_mask_test"]
        and integrity["positive_legal_skip_test_passed"]
        and integrity["no_future_leak_tests_passed"]
        and integrity["identity_chronology_tests_passed"]
        and integrity["pytest_regression"]["passed"]
    )
    return {
        "created_at": iso_kst(),
        "final_decision": PASS_DECISION if passed else FAIL_DECISION,
        "dynamic_state_implemented": passed,
        "dynamic_state_complete_in_valid_fixture": passed,
        "positive_legal_skip_synthetic_test_passed": integrity["positive_legal_skip_test_passed"],
        "remaining_static_rule_gaps": list(k3["remaining_static_gaps"]),
        "static_rules_positively_clear_for_real_network": False,
        "minimum_k5_scope": [
            "materialize an authoritative static rulebook for mandatory/protected/planned-skip/terminal/charging-relief fields with effective-time provenance",
            "bind one state-machine instance to each branch/global simulation state and reduce ReplayEvent passenger/request transitions before every agent mask",
            "serialize passenger, request, stop-queue, vehicle-ledger, pending-event, and completeness state into DynamicsStateSnapshot for clone/reset determinism",
            "validate all eight fixed physical-vehicle snapshots and static joins prospectively without future leakage",
            "enable K_action_mask only after static positive-clearance and end-to-end orchestrator tests pass; keep policy and training locked separately",
        ],
        "K_action_mask_available": False,
        "conditional_skip_policy_enabled": False,
    }


def final_report_text(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# PV8-K4 Dynamic Service-Obligation State Final Report",
            "",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- final_decision: `{summary['final_decision']}`",
            f"- implemented entities: `{summary['implemented_entities']}`",
            f"- implemented transitions: `{summary['implemented_transitions']}`",
            f"- transition tests passed: `{summary['transition_passed']}/{summary['transition_total']}`",
            f"- skip predicate tests passed: `{summary['predicate_passed']}/{summary['predicate_total']}`",
            f"- relevant pytest regression: `{summary['pytest_passed_count']} passed`",
            f"- dynamic-state completeness in valid fixtures: `{summary['dynamic_state_complete']}`",
            f"- positive legal-SKIP synthetic test: `{summary['positive_skip_test_passed']}`",
            f"- no-future-leak boundary tests: `{summary['no_future_leak_tests_passed']}`",
            f"- remaining static-rule gaps: `{summary['remaining_static_rule_gaps']}`",
            f"- minimum K5 scope: `{summary['minimum_k5_scope']}`",
            f"- K_action_mask_available: `{summary['K_action_mask_available']}`",
            f"- conditional_skip_policy_enabled: `{summary['conditional_skip_policy_enabled']}`",
            f"- training_use_authorized: `{summary['training_use_authorized']}`",
            f"- checkpoint_reuse_authorized: `{summary['checkpoint_reuse_authorized']}`",
            f"- policy_evaluation_authorized: `{summary['policy_evaluation_authorized']}`",
            f"- gate: `{summary['gate']}`",
            f"- readiness: `{summary['readiness']}`",
            "",
            "Dynamic passenger/request/service-leg identity and decision-time obligations are implemented. The action mask now consumes a typed snapshot; absent or incomplete dynamic state is not treated as zero.",
            "",
            "The positive SKIP result is a synthetic contract test only. Real-network SKIP remains unavailable because all five static rule fields still lack positive clearance evidence.",
            "",
            "No new BIS call, DB query/write, policy evaluation, K5 execution, MAPPO retraining, checkpoint reuse, or policy SKIP enablement was performed.",
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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_k4.jsonl"
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
    manifest_name = "artifact_manifest_srp2_bis_pv8_k4.json"
    writer.json(manifest_name, manifest)
    manifest_path = writer.root / manifest_name
    writer.json(
        "_PV8_K4_COMPLETE.lock",
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


TRANSITION_COLUMNS = [
    "test_id", "transition", "passed", "passenger_id", "request_id", "service_leg_id", "event_ts",
    "expected_passenger_status", "observed_passenger_status", "expected_request_status", "observed_request_status",
    "identity_preserved", "chronology_preserved", "details",
]
SNAPSHOT_COLUMNS = [
    "sample_id", "agent_id", "vehicle_token", "decision_ts", "current_stop_id", "candidate_stop_id",
    "pickup_obligation", "dropoff_obligation", "boarding_obligation", "alighting_obligation",
    "onboard_destination_obligation", "service_obligation", "assigned_pickup", "assigned_dropoff",
    "dynamic_state_complete", "static_guard_complete", "static_guard_clear", "state_complete", "skip_allowed",
    "waiting_queue", "missing_reason",
]
PREDICATE_COLUMNS = [
    "test_id", "expected_skip_valid", "observed_skip_valid", "passed", "expected_reason", "observed_reasons",
    "dynamic_state_complete", "static_guard_complete", "snapshot_present", "future_event_visible",
    "engine_execution_attempted", "engine_execution_succeeded", "position_after",
]


def run_validate(root: Path) -> Path:
    root = validate_artifact_root(root)
    writer = Writer(root)
    upstreams = verify_upstreams()
    contract = state_machine_contract()
    transition_rows, machines = transition_test_rows()
    predicate_rows, snapshot_rows, engine_execution_count = predicate_and_snapshot_tests()
    pytest_result = run_pytest_regression()
    integrity = integrity_audit(transition_rows, predicate_rows, snapshot_rows, machines, pytest_result)
    decision = readiness_decision(integrity)
    guard = claim_guard()
    passed = decision["final_decision"] == PASS_DECISION
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE if passed else FAIL_GATE,
        "terminal_gate": PASS_GATE if passed else FAIL_GATE,
        "readiness": PASS_READINESS if passed else FAIL_READINESS,
        "gate_passed": passed,
        "final_decision": decision["final_decision"],
        "failure_reasons": [] if passed else [
            "transition, snapshot, predicate, source-integrity, or pytest regression condition failed"
        ],
    }
    summary = {
        "artifact_root": str(root),
        "final_decision": decision["final_decision"],
        "implemented_entities": list(contract["entities"]),
        "implemented_transitions": contract["implemented_transitions"],
        "transition_passed": sum(bool(row["passed"]) for row in transition_rows),
        "transition_total": len(transition_rows),
        "predicate_passed": sum(bool(row["passed"]) for row in predicate_rows),
        "predicate_total": len(predicate_rows),
        "pytest_passed_count": pytest_result["passed_test_count"],
        "dynamic_state_complete": decision["dynamic_state_complete_in_valid_fixture"],
        "positive_skip_test_passed": decision["positive_legal_skip_synthetic_test_passed"],
        "no_future_leak_tests_passed": integrity["no_future_leak_tests_passed"],
        "remaining_static_rule_gaps": decision["remaining_static_rule_gaps"],
        "minimum_k5_scope": decision["minimum_k5_scope"],
        "K_action_mask_available": guard["K_action_mask_available"],
        "conditional_skip_policy_enabled": guard["conditional_skip_policy_enabled"],
        "training_use_authorized": guard["training_use_authorized"],
        "checkpoint_reuse_authorized": guard["checkpoint_reuse_authorized"],
        "policy_evaluation_authorized": guard["policy_evaluation_authorized"],
        "gate": gate["gate"],
        "readiness": gate["readiness"],
    }

    writer.json("k4_state_machine_contract.json", contract)
    writer.parquet("k4_transition_test_results.parquet", transition_rows, TRANSITION_COLUMNS)
    writer.parquet("k4_decision_snapshot_samples.parquet", snapshot_rows, SNAPSHOT_COLUMNS)
    writer.parquet("k4_skip_predicate_test_results.parquet", predicate_rows, PREDICATE_COLUMNS)
    writer.json("k4_state_integrity_audit.json", integrity)
    writer.json("k4_readiness_decision.json", decision)
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
            "upstream_integrity": upstreams,
            "new_bis_api_call_count": 0,
            "direct_postgresql_query_count": 0,
            "db_write_count": 0,
            "engine_validation_execution_count": engine_execution_count,
            "relevant_pytest_case_count": pytest_result["passed_test_count"],
            "policy_evaluation_count": 0,
            "k5_execution_count": 0,
            "mappo_training_count": 0,
            "checkpoint_reuse_count": 0,
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
            "conditional_skip_policy_enabled": False,
            "training_use_authorized": False,
            "checkpoint_reuse_authorized": False,
            "policy_evaluation_authorized": False,
            "causal_performance_claim_allowed": False,
            "paper_level_claim_allowed": False,
            "automatic_k5_execution_authorized": False,
            "automatic_mappo_retraining_authorized": False,
        },
    )
    writer.text("final_report.md", final_report_text(summary))
    write_manifest_and_lock(writer, gate)
    own_checks = verify_manifest(root, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock")
    if not manifest_ok(own_checks):
        raise K4Error(f"ARTIFACT_INTEGRITY_FAILURE: {own_checks}")
    if not passed:
        raise K4Error(f"K4 validation failed: {integrity}")

    for key in [
        "artifact_root",
        "final_decision",
        "implemented_entities",
        "implemented_transitions",
        "transition_passed",
        "transition_total",
        "predicate_passed",
        "predicate_total",
        "pytest_passed_count",
        "dynamic_state_complete",
        "positive_skip_test_passed",
        "no_future_leak_tests_passed",
        "remaining_static_rule_gaps",
        "K_action_mask_available",
        "conditional_skip_policy_enabled",
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
