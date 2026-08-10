from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pytest

from suseong_service_transition_engine import (
    StopServiceResult,
    TransitionConfig,
    advance_vehicle_time_budget,
    compute_counter_semantics_v3,
)


@dataclass
class Vehicle:
    agent_id: int = 0
    route_key: Tuple[str, str] = ("R_TEST", "1")
    position: int = 0
    onboard_count: int = 0
    capacity: int = 80
    remaining_travel_time: float = 0.0
    remaining_dwell_time: float = 0.0


def fixture_routes() -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    return {
        ("R_TEST", "1"): [
            {"stop_id": "S0", "node_uid": "STOP:S0"},
            {"stop_id": "S1", "node_uid": "STOP:S1"},
            {"stop_id": "S2", "node_uid": "STOP:S2"},
        ]
    }


def no_demand_service(vehicle: Vehicle, _stop_row: Dict[str, Any]) -> StopServiceResult:
    return StopServiceResult(boardings=0, alightings=0, dwell_required=vehicle.position > 0)


@pytest.mark.parametrize(
    ("delta_t", "expected_position", "expected_travel", "expected_dwell"),
        [
            (60.0, 0, 240.0, 0.0),
            (300.0, 1, 0.0, 0.0),
            (600.0, 1, 60.0, 0.0),
            (3600.0, 2, 0.0, 0.0),
        ],
)
def test_time_budget_exact_fixture(delta_t: float, expected_position: int, expected_travel: float, expected_dwell: float) -> None:
    vehicle = Vehicle()
    trace = advance_vehicle_time_budget(
        vehicle=vehicle,
        routes=fixture_routes(),
        delta_t_seconds=delta_t,
        action=1,
        stop_service=no_demand_service,
        config=TransitionConfig(edge_travel_seconds=300.0, dwell_seconds=60.0, allow_turnaround=False),
    )
    assert vehicle.position == expected_position
    assert vehicle.remaining_travel_time == expected_travel
    assert vehicle.remaining_dwell_time == expected_dwell
    assert trace.time_budget_conservation_error_seconds == 0.0
    assert not trace.transition_guard_triggered


def test_zero_delta_consumes_nothing() -> None:
    vehicle = Vehicle()
    trace = advance_vehicle_time_budget(
        vehicle=vehicle,
        routes=fixture_routes(),
        delta_t_seconds=0.0,
        action=1,
        stop_service=no_demand_service,
        config=TransitionConfig(edge_travel_seconds=300.0, dwell_seconds=60.0, allow_turnaround=False),
    )
    assert vehicle.position == 0
    assert trace.events == []
    assert trace.time_budget_conservation_error_seconds == 0.0


def test_negative_delta_rejected() -> None:
    with pytest.raises(ValueError):
        advance_vehicle_time_budget(
            vehicle=Vehicle(),
            routes=fixture_routes(),
            delta_t_seconds=-1.0,
            action=1,
            stop_service=no_demand_service,
            config=TransitionConfig(edge_travel_seconds=300.0, dwell_seconds=60.0, allow_turnaround=False),
        )


def test_zero_travel_and_zero_dwell_are_explicit() -> None:
    vehicle = Vehicle()
    trace = advance_vehicle_time_budget(
        vehicle=vehicle,
        routes=fixture_routes(),
        delta_t_seconds=1.0,
        action=1,
        stop_service=lambda _vehicle, _stop: StopServiceResult(),
        config=TransitionConfig(edge_travel_seconds=0.0, dwell_seconds=0.0, allow_turnaround=False),
    )
    assert vehicle.position == 2
    assert trace.edges_traversed == 2
    assert trace.valid_idle_seconds == 1.0
    assert trace.time_budget_conservation_error_seconds == 0.0


def test_counter_v3_terminal_idle_is_not_trip() -> None:
    vehicle = Vehicle()
    events: List[Dict[str, Any]] = []
    for snapshot_id in range(5):
        trace = advance_vehicle_time_budget(
            vehicle=vehicle,
            routes=fixture_routes(),
            delta_t_seconds=1.0,
            action=1,
            stop_service=lambda _vehicle, _stop: StopServiceResult(),
            config=TransitionConfig(edge_travel_seconds=0.0, dwell_seconds=0.0, allow_turnaround=False),
            run_id="fixture",
            service_day_id="day0",
            snapshot_id=snapshot_id,
            snapshot_start_time_seconds=float(snapshot_id),
        )
        events.extend(trace.events)
    audit = compute_counter_semantics_v3(events, instantiated_vehicle_count=1, validation_snapshot_count=5)
    assert audit["terminal_arrival_event_count"] == 1
    assert audit["completed_trip_count"] == 1
    assert audit["unique_terminal_arriving_vehicle_count"] == 1
    assert audit["terminal_idle_observation_count"] == 5
    assert audit["repeated_terminal_idle_observation_count"] == 4
    assert audit["terminal_stuck_unique_vehicle_count"] == 1
    assert audit["repeated_terminal_observation_counted_as_trip"] is False
    assert audit["counter_semantics_passed"] is True


def test_counter_v3_terminal_not_reached() -> None:
    vehicle = Vehicle()
    trace = advance_vehicle_time_budget(
        vehicle=vehicle,
        routes=fixture_routes(),
        delta_t_seconds=10.0,
        action=1,
        stop_service=lambda _vehicle, _stop: StopServiceResult(),
        config=TransitionConfig(edge_travel_seconds=100.0, dwell_seconds=0.0, allow_turnaround=False),
    )
    audit = compute_counter_semantics_v3(trace.events, instantiated_vehicle_count=1, validation_snapshot_count=1)
    assert audit["terminal_arrival_event_count"] == 0
    assert audit["completed_trip_count"] == 0
    assert audit["terminal_idle_observation_count"] == 0
    assert audit["terminal_stuck_unique_vehicle_count"] == 0


def test_counter_v3_two_vehicles_reach_terminal() -> None:
    events: List[Dict[str, Any]] = []
    for agent_id in [1, 2]:
        vehicle = Vehicle(agent_id=agent_id)
        trace = advance_vehicle_time_budget(
            vehicle=vehicle,
            routes=fixture_routes(),
            delta_t_seconds=1.0,
            action=1,
            stop_service=lambda _vehicle, _stop: StopServiceResult(),
            config=TransitionConfig(edge_travel_seconds=0.0, dwell_seconds=0.0, allow_turnaround=False),
        )
        events.extend(trace.events)
    audit = compute_counter_semantics_v3(events, instantiated_vehicle_count=2, validation_snapshot_count=1)
    assert audit["terminal_arrival_event_count"] == 2
    assert audit["unique_terminal_arriving_vehicle_count"] == 2
    assert audit["terminal_stuck_unique_vehicle_count"] == 2


def test_counter_v3_same_vehicle_reentry_counts_multiple_trip() -> None:
    vehicle = Vehicle()
    first = advance_vehicle_time_budget(
        vehicle=vehicle,
        routes=fixture_routes(),
        delta_t_seconds=1.0,
        action=1,
        stop_service=lambda _vehicle, _stop: StopServiceResult(),
        config=TransitionConfig(edge_travel_seconds=0.0, dwell_seconds=0.0, allow_turnaround=False),
    )
    vehicle.position = 0
    vehicle.vehicle_state = "IN_SERVICE"
    second = advance_vehicle_time_budget(
        vehicle=vehicle,
        routes=fixture_routes(),
        delta_t_seconds=1.0,
        action=1,
        stop_service=lambda _vehicle, _stop: StopServiceResult(),
        config=TransitionConfig(edge_travel_seconds=0.0, dwell_seconds=0.0, allow_turnaround=False),
    )
    audit = compute_counter_semantics_v3([*first.events, *second.events], instantiated_vehicle_count=1, validation_snapshot_count=2)
    assert audit["terminal_arrival_event_count"] == 2
    assert audit["multiple_trip_vehicle_count"] == 1
