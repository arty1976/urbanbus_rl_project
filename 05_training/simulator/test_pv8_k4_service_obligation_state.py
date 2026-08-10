from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest

from simulator.k_safety_state import (
    KSafetyChronologyError,
    PassengerStatus,
    RequestStatus,
    ServiceObligationStateMachine,
    StaticGuardStatus,
)
from simulator.suseong_service_transition_engine import (
    ENGINE_ACTION_CONDITIONAL_SKIP,
    InvalidConditionalSkipError,
    StopServiceResult,
    TransitionConfig,
    advance_vehicle_time_budget,
    build_distinct_three_action_mask,
)


@dataclass
class Vehicle:
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


def fixture_routes(next_stop: Optional[Dict[str, Any]] = None) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    return {
        ("R_TEST", "1"): [
            {"stop_id": "S0", "node_uid": "STOP:S0"},
            next_stop or clear_stop("S1"),
            {"stop_id": "S2", "node_uid": "STOP:S2"},
        ]
    }


def machine(*, register_candidate: bool = True) -> ServiceObligationStateMachine:
    state = ServiceObligationStateMachine()
    state.register_vehicle(0, "VEHICLE-0")
    if register_candidate:
        state.register_stop("S1")
    state.register_stop("S2")
    return state


def mask(state: ServiceObligationStateMachine, next_stop: Optional[Dict[str, Any]] = None, decision_ts: int = 0):
    return build_distinct_three_action_mask(
        Vehicle(),
        fixture_routes(next_stop),
        obligation_state_machine=state,
        decision_ts=decision_ts,
        vehicle_token="VEHICLE-0",
    )


def add_assigned_pickup(state: ServiceObligationStateMachine) -> None:
    state.passenger_waiting(passenger_id="P1", pickup_stop="S1", dropoff_stop="S2", event_ts=1)
    state.request_created(request_id="R1", passenger_id="P1", service_leg_id="L1", event_ts=2)
    state.request_assigned(request_id="R1", agent_id=0, vehicle_token="VEHICLE-0", event_ts=3)


def test_positive_legal_skip_synthetic_case_executes_two_stop_move() -> None:
    state = machine()
    vehicle = Vehicle()
    routes = fixture_routes()
    result = build_distinct_three_action_mask(
        vehicle,
        routes,
        obligation_state_machine=state,
        decision_ts=0,
        vehicle_token=vehicle.vehicle_token,
    )
    assert result["action_mask"] == [True, True, True]
    trace = advance_vehicle_time_budget(
        vehicle=vehicle,
        routes=routes,
        delta_t_seconds=1.0,
        action=ENGINE_ACTION_CONDITIONAL_SKIP,
        stop_service=lambda _vehicle, _stop: StopServiceResult(),
        config=TransitionConfig(edge_travel_seconds=0.0, dwell_seconds=0.0),
        obligation_state_machine=state,
        decision_ts=0,
        vehicle_token=vehicle.vehicle_token,
    )
    assert trace.skip_executed is True
    assert vehicle.position == 2


def test_pickup_and_boarding_obligations_block_skip() -> None:
    state = machine()
    add_assigned_pickup(state)
    result = mask(state, decision_ts=3)
    assert result["skip_valid"] is False
    assert result["pickup_obligation"] is True
    assert "ASSIGNED_PICKUP_REQUEST" in result["skip_invalid_reason_codes"]
    assert "WAITING_PICKUP_DEMAND" in result["skip_invalid_reason_codes"]


def test_waiting_passenger_boarding_obligation_blocks_without_assignment() -> None:
    state = machine()
    state.passenger_waiting(passenger_id="P1", pickup_stop="S1", dropoff_stop="S2", event_ts=1)
    result = mask(state, decision_ts=1)
    snapshot = result["decision_time_obligation_snapshot"]
    assert snapshot["boarding_obligation"] is True
    assert snapshot["assigned_pickup"] is False
    assert result["skip_valid"] is False


def test_boarded_identity_creates_dropoff_alighting_and_onboard_destination_obligations() -> None:
    state = machine()
    add_assigned_pickup(state)
    state.passenger_boarded(
        request_id="R1",
        passenger_id="P1",
        agent_id=0,
        vehicle_token="VEHICLE-0",
        stop_id="S1",
        event_ts=4,
    )
    snapshot = state.snapshot_for_vehicle(
        agent_id=0,
        vehicle_token="VEHICLE-0",
        decision_ts=4,
        current_stop_id="S1",
        candidate_stop_id="S2",
        static_guard_status=StaticGuardStatus(False, False, True, False, False, True, True),
    )
    assert snapshot.dropoff_obligation is True
    assert snapshot.alighting_obligation is True
    assert snapshot.onboard_destination_obligation is True
    assert snapshot.assigned_dropoff is True
    assert snapshot.skip_allowed is False


def test_missing_dynamic_state_and_missing_machine_fail_closed() -> None:
    incomplete = machine(register_candidate=False)
    result = mask(incomplete)
    assert result["dynamic_state_complete"] is False
    assert "CANDIDATE_STOP_QUEUE_MISSING" in result["skip_invalid_reason_codes"]
    no_machine = build_distinct_three_action_mask(Vehicle(), fixture_routes())
    assert no_machine["skip_valid"] is False
    assert "DYNAMIC_STATE_MACHINE_MISSING" in no_machine["skip_invalid_reason_codes"]


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"terminal_or_turnaround_stop": True}, "TERMINAL_OR_TURNAROUND_STOP"),
        ({"downstream_path_valid": False}, "DOWNSTREAM_PATH_INVALID"),
        ({"mandatory_stop": True}, "MANDATORY_STOP"),
    ],
)
def test_static_terminal_path_and_mandatory_guards_block(overrides: Dict[str, Any], reason: str) -> None:
    result = mask(machine(), clear_stop("S1", **overrides))
    assert result["skip_valid"] is False
    assert reason in result["skip_invalid_reason_codes"]


def test_unknown_static_rule_blocks_skip() -> None:
    result = mask(machine(), {"stop_id": "S1", "node_uid": "STOP:S1"})
    assert result["skip_valid"] is False
    assert result["static_guard_state_complete"] is False
    assert "STATIC_GUARD_STATE_INCOMPLETE" in result["skip_invalid_reason_codes"]


def test_cancel_transition_removes_waiting_and_assigned_pickup() -> None:
    state = machine()
    add_assigned_pickup(state)
    state.request_cancelled(request_id="R1", event_ts=4)
    snapshot = state.snapshot_for_vehicle(
        agent_id=0,
        vehicle_token="VEHICLE-0",
        decision_ts=4,
        current_stop_id="S0",
        candidate_stop_id="S1",
        static_guard_status=StaticGuardStatus(False, False, True, False, False, True, True),
    )
    assert snapshot.service_obligation is False
    assert state.requests["R1"].request_status == RequestStatus.CANCELLED
    assert state.passengers["P1"].passenger_status == PassengerStatus.CANCELLED
    assert state.audit_integrity()["passed"] is True


def test_board_onboard_alight_complete_identity_chain() -> None:
    state = machine()
    add_assigned_pickup(state)
    state.passenger_boarded(request_id="R1", passenger_id="P1", agent_id=0, vehicle_token="VEHICLE-0", stop_id="S1", event_ts=4)
    state.passenger_alighted(request_id="R1", passenger_id="P1", agent_id=0, vehicle_token="VEHICLE-0", stop_id="S2", event_ts=5)
    state.request_completed(request_id="R1", event_ts=6)
    transitions = [row["transition"] for row in state.event_log]
    assert transitions == [
        "passenger_waiting",
        "request_created",
        "request_assigned",
        "passenger_boarded",
        "passenger_alighted",
        "request_completed",
    ]
    assert state.passengers["P1"].passenger_status == PassengerStatus.SERVED
    assert state.requests["R1"].request_status == RequestStatus.COMPLETED
    assert state.audit_integrity()["passed"] is True


def test_decision_time_boundary_does_not_read_future_event() -> None:
    state = machine()
    state.schedule_transition(
        "passenger_waiting",
        101,
        passenger_id="P_FUTURE",
        pickup_stop="S1",
        dropoff_stop="S2",
    )
    before = mask(state, decision_ts=100)
    assert before["skip_valid"] is True
    assert before["decision_time_obligation_snapshot"]["waiting_queue"] == []
    at_boundary = mask(state, decision_ts=101)
    assert at_boundary["skip_valid"] is False
    assert at_boundary["decision_time_obligation_snapshot"]["waiting_queue"] == ["P_FUTURE"]


def test_chronology_and_reassignment_preserve_identity() -> None:
    state = machine()
    state.register_vehicle(1, "VEHICLE-1")
    add_assigned_pickup(state)
    state.request_reassigned(request_id="R1", agent_id=1, vehicle_token="VEHICLE-1", event_ts=4)
    assert "R1" not in state.vehicle_ledgers[0].assigned_pickup
    assert "R1" in state.vehicle_ledgers[1].assigned_pickup
    assert state.requests["R1"].passenger_id == "P1"
    with pytest.raises(KSafetyChronologyError):
        state.request_cancelled(request_id="R1", event_ts=3)


def test_engine_rejects_skip_without_typed_snapshot_before_mutation() -> None:
    vehicle = Vehicle()
    with pytest.raises(InvalidConditionalSkipError):
        advance_vehicle_time_budget(
            vehicle=vehicle,
            routes=fixture_routes(),
            delta_t_seconds=1.0,
            action=ENGINE_ACTION_CONDITIONAL_SKIP,
            stop_service=lambda _vehicle, _stop: StopServiceResult(),
            config=TransitionConfig(edge_travel_seconds=0.0, dwell_seconds=0.0),
        )
    assert vehicle.position == 0
