from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest

from k_safety_state import ServiceObligationStateMachine
from suseong_service_transition_engine import (
    ACTOR_TO_ENGINE_ACTION,
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
    onboard_count: int = 0
    capacity: int = 80
    remaining_travel_time: float = 0.0
    remaining_dwell_time: float = 0.0
    onboard_destination_stop_ids: List[str] = None

    def __post_init__(self) -> None:
        if self.onboard_destination_stop_ids is None:
            self.onboard_destination_stop_ids = []


def stop(
    name: str,
    *,
    waiting: int = 0,
    assigned_pickup: int = 0,
    assigned_dropoff: int = 0,
    scheduled_alighting: int = 0,
    mandatory: bool = False,
    path_valid: bool = True,
    has_safety_fields: bool = True,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"stop_id": name, "node_uid": f"STOP:{name}"}
    if has_safety_fields:
        row.update(
            {
                "waiting_pickup_count": waiting,
                "scheduled_alighting_count": scheduled_alighting,
                "assigned_pickup_request_count": assigned_pickup,
                "assigned_dropoff_request_count": assigned_dropoff,
                "mandatory_stop": mandatory,
                "protected_stop": False,
                "terminal_or_turnaround_stop": False,
                "charging_or_driver_relief_stop": False,
                "planned_itinerary_allows_skip": True,
                "downstream_path_valid": path_valid,
                "graph_edge_or_path_valid": path_valid,
                "max_consecutive_skip_constraint_satisfied": True,
                "service_fairness_constraint_satisfied": True,
            }
        )
    return row


def routes(next_stop: Dict[str, Any], post_stop: Optional[Dict[str, Any]] = None) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    rows = [stop("S0"), next_stop]
    if post_stop is not None:
        rows.append(post_stop)
    return {("R_TEST", "1"): rows}


def no_service(_vehicle: Vehicle, _stop_row: Dict[str, Any]) -> StopServiceResult:
    return StopServiceResult(boardings=0, alightings=0, dwell_required=False)


def state(case: str = "empty") -> ServiceObligationStateMachine:
    result = ServiceObligationStateMachine()
    result.register_vehicle(0, "VEHICLE-0")
    for stop_id in ["S0", "S1", "S2"]:
        result.register_stop(stop_id)
    if case in {"waiting", "assigned_pickup"}:
        result.passenger_waiting(passenger_id="P1", pickup_stop="S1", dropoff_stop="S2", event_ts=1)
        if case == "assigned_pickup":
            result.request_created(request_id="R1", passenger_id="P1", event_ts=2)
            result.request_assigned(request_id="R1", agent_id=0, vehicle_token="VEHICLE-0", event_ts=3)
    if case in {"assigned_dropoff", "onboard_dropoff"}:
        result.passenger_waiting(passenger_id="P1", pickup_stop="S0", dropoff_stop="S1", event_ts=1)
        result.request_created(request_id="R1", passenger_id="P1", event_ts=2)
        result.request_assigned(request_id="R1", agent_id=0, vehicle_token="VEHICLE-0", event_ts=3)
        result.passenger_boarded(
            request_id="R1",
            passenger_id="P1",
            agent_id=0,
            vehicle_token="VEHICLE-0",
            stop_id="S0",
            event_ts=4,
        )
    return result


def mask(vehicle: Vehicle, fixture_routes: Dict[Tuple[str, str], List[Dict[str, Any]]], state_machine: ServiceObligationStateMachine):
    return build_distinct_three_action_mask(
        vehicle,
        fixture_routes,
        obligation_state_machine=state_machine,
        decision_ts=state_machine.current_ts,
        vehicle_token=vehicle.vehicle_token,
    )


def step(
    vehicle: Vehicle,
    fixture_routes: Dict[Tuple[str, str], List[Dict[str, Any]]],
    action: int,
    state_machine: Optional[ServiceObligationStateMachine] = None,
):
    return advance_vehicle_time_budget(
        vehicle=vehicle,
        routes=fixture_routes,
        delta_t_seconds=45.0,
        action=action,
        stop_service=no_service,
        config=TransitionConfig(edge_travel_seconds=45.0, dwell_seconds=0.0, allow_turnaround=False),
        obligation_state_machine=state_machine,
        decision_ts=state_machine.current_ts if state_machine is not None else 0,
        vehicle_token=vehicle.vehicle_token,
    )


def test_dl6c_actor_to_engine_mapping_uses_skip_engine_action() -> None:
    assert ACTOR_TO_ENGINE_ACTION == {0: 0, 1: 1, 2: 3}


def test_empty_stop_skip_is_valid_and_distinct_from_move_one() -> None:
    fixture = routes(stop("S1"), stop("S2"))
    skip_vehicle = Vehicle()
    move_vehicle = Vehicle()
    state_machine = state()
    action_mask = mask(skip_vehicle, fixture, state_machine)
    assert action_mask["action_mask"] == [True, True, True]
    skip_trace = step(skip_vehicle, fixture, ENGINE_ACTION_CONDITIONAL_SKIP, state_machine)
    move_trace = step(move_vehicle, fixture, 1)
    assert skip_vehicle.position == 2
    assert move_vehicle.position == 1
    assert skip_trace.skip_executed is True
    assert skip_trace.skipped_stop_id == "S1"
    assert skip_trace.post_skip_target_stop_id == "S2"
    assert skip_trace.missed_pickup_due_to_skip == 0
    assert skip_trace.missed_dropoff_due_to_skip == 0


@pytest.mark.parametrize(
    ("case", "next_stop", "reason"),
    [
        ("waiting", stop("S1"), "WAITING_PICKUP_DEMAND"),
        ("assigned_pickup", stop("S1"), "ASSIGNED_PICKUP_REQUEST"),
        ("assigned_dropoff", stop("S1"), "ASSIGNED_DROPOFF_REQUEST"),
        ("empty", stop("S1", mandatory=True), "MANDATORY_STOP"),
        ("empty", stop("S1", path_valid=False), "DOWNSTREAM_PATH_INVALID"),
    ],
)
def test_skip_block_reasons(case: str, next_stop: Dict[str, Any], reason: str) -> None:
    fixture = routes(next_stop, stop("S2"))
    vehicle = Vehicle()
    state_machine = state(case)
    action_mask = mask(vehicle, fixture, state_machine)
    assert action_mask["skip_valid"] is False
    assert reason in action_mask["skip_invalid_reason_codes"]
    with pytest.raises(InvalidConditionalSkipError):
        step(vehicle, fixture, ENGINE_ACTION_CONDITIONAL_SKIP, state_machine)


def test_onboard_dropoff_blocks_skip() -> None:
    fixture = routes(stop("S1"), stop("S2"))
    vehicle = Vehicle()
    action_mask = mask(vehicle, fixture, state("onboard_dropoff"))
    assert action_mask["skip_valid"] is False
    assert "ONBOARD_DROPOFF_DEMAND" in action_mask["skip_invalid_reason_codes"]


def test_no_post_skip_target_blocks_skip() -> None:
    fixture = routes(stop("S1"), None)
    vehicle = Vehicle()
    action_mask = mask(vehicle, fixture, state())
    assert action_mask["skip_valid"] is False
    assert "NO_POST_SKIP_TARGET" in action_mask["skip_invalid_reason_codes"]


def test_missing_safety_fields_fail_closed() -> None:
    fixture = routes(stop("S1", has_safety_fields=False), stop("S2"))
    vehicle = Vehicle()
    action_mask = mask(vehicle, fixture, state())
    assert action_mask["skip_valid"] is False
    assert "STATIC_GUARD_STATE_INCOMPLETE" in action_mask["skip_invalid_reason_codes"]


def test_three_actions_are_pairwise_distinct() -> None:
    fixture = routes(stop("S1"), stop("S2"))
    vehicles = [Vehicle(), Vehicle(), Vehicle()]
    traces = [step(vehicles[0], fixture, 0), step(vehicles[1], fixture, 1), step(vehicles[2], fixture, 3, state())]
    states = {(vehicle.position, trace.last_action_semantic, trace.skip_executed) for vehicle, trace in zip(vehicles, traces)}
    assert len(states) == 3
