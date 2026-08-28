from __future__ import annotations

import pytest
from typing import Optional

from simulator.k_safety_state import ServiceObligationStateMachine
from simulator.pv8_reward_outcome_collector import (
    CausalOutcomeCollector,
    DecisionOutcomeBinding,
    DuplicateRewardEventError,
    RewardEventBoundaryError,
    RewardIdentityError,
    RewardOutcomeEvent,
    RewardOutcomeEventType,
    aggregate_team_outcomes,
    map_k4_event_log,
)


def binding(decision_id: str = "D0", *, active: bool = True, agent_id: int = 0) -> DecisionOutcomeBinding:
    return DecisionOutcomeBinding(
        decision_id=decision_id,
        decision_ts=100,
        agent_id=agent_id,
        vehicle_token=f"PV8-{agent_id}",
        action_t="SERVE" if active else None,
        outcome_start_ts=100,
        outcome_end_ts=200,
        next_decision_ts=200,
        terminal_ts=None,
        active_bus=active,
        observation_hash=f"obs-{agent_id}",
        k_mask_hash=f"mask-{agent_id}",
    )


def event(
    event_id: str,
    event_ts: int,
    event_type: RewardOutcomeEventType,
    *,
    agent_id: int = 0,
    request_id: Optional[str] = None,
    passenger_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> RewardOutcomeEvent:
    return RewardOutcomeEvent(
        event_id=event_id,
        event_ts=event_ts,
        event_type=event_type,
        agent_id=agent_id,
        vehicle_token=f"PV8-{agent_id}",
        request_id=request_id,
        passenger_id=passenger_id,
        route_id="R1" if event_type == RewardOutcomeEventType.STOP_ARRIVAL else None,
        direction_id="0" if event_type == RewardOutcomeEventType.STOP_ARRIVAL else None,
        route_stop_occurrence_id="R1:0:1:S1" if event_type == RewardOutcomeEventType.STOP_ARRIVAL else None,
        metadata=metadata or {},
    )


def test_exact_service_wait_and_movement_metrics_without_reward_value() -> None:
    collector = CausalOutcomeCollector()
    collector.register_decision(binding())
    events = [
        event("e1", 110, RewardOutcomeEventType.PASSENGER_GENERATED, request_id="R1", passenger_id="P1"),
        event("e2", 140, RewardOutcomeEventType.PASSENGER_BOARDED, request_id="R1", passenger_id="P1"),
        event("e3", 180, RewardOutcomeEventType.PASSENGER_ALIGHTED, request_id="R1", passenger_id="P1"),
        event("e4", 181, RewardOutcomeEventType.REQUEST_COMPLETED, request_id="R1", passenger_id="P1"),
        event("e5", 150, RewardOutcomeEventType.VEHICLE_MOVEMENT, metadata={"distance_m": 500.0, "travel_seconds": 40.0, "dwell_seconds": 10.0}),
    ]
    for row in events:
        collector.record_event("D0", row)
    result = collector.finalize("D0")
    assert result["eligible_request_count"] == 1
    assert result["passenger_served_count"] == 1
    assert result["service_rate"] == 1.0
    assert result["avg_wait_seconds"] == 30.0
    assert result["p95_wait_seconds"] == 30.0
    assert result["vehicle_distance_m"] == 500.0
    assert result["reward_value_materialized"] is False


def test_unserved_request_is_retained_as_censored_wait() -> None:
    collector = CausalOutcomeCollector()
    collector.register_decision(binding())
    collector.record_event("D0", event("e1", 120, RewardOutcomeEventType.PASSENGER_GENERATED, request_id="R1", passenger_id="P1"))
    result = collector.finalize("D0")
    assert result["service_rate"] == 0.0
    assert result["avg_wait_seconds"] == 80.0
    assert result["censored_wait_count"] == 1


def test_inactive_slot_is_bypassed_and_rejects_events() -> None:
    collector = CausalOutcomeCollector()
    collector.register_decision(binding(active=False))
    with pytest.raises(RewardIdentityError):
        collector.record_event("D0", event("e1", 150, RewardOutcomeEventType.VEHICLE_MOVEMENT))
    result = collector.finalize("D0")
    assert result["learning_sample"] is False
    assert result["event_count"] == 0


def test_boundary_and_duplicate_guards_precede_mutation() -> None:
    collector = CausalOutcomeCollector()
    collector.register_decision(binding())
    with pytest.raises(RewardEventBoundaryError):
        collector.record_event("D0", event("pre", 100, RewardOutcomeEventType.VEHICLE_MOVEMENT))
    with pytest.raises(RewardEventBoundaryError):
        collector.record_event("D0", event("future", 201, RewardOutcomeEventType.VEHICLE_MOVEMENT))
    accepted = event("ok", 150, RewardOutcomeEventType.VEHICLE_MOVEMENT)
    collector.record_event("D0", accepted)
    with pytest.raises(DuplicateRewardEventError):
        collector.record_event("D0", accepted)
    assert collector.finalize("D0")["event_count"] == 1


def test_team_headway_and_intervention_semantics() -> None:
    results = []
    for agent_id, arrival_ts in enumerate([110, 140, 180]):
        collector = CausalOutcomeCollector()
        decision_id = f"D{agent_id}"
        collector.register_decision(binding(decision_id, agent_id=agent_id))
        collector.record_event(decision_id, event(f"a{agent_id}", arrival_ts, RewardOutcomeEventType.STOP_ARRIVAL, agent_id=agent_id))
        if agent_id == 0:
            collector.record_event(decision_id, event("override", 150, RewardOutcomeEventType.FORCED_SAFETY_OVERRIDE, agent_id=agent_id))
        results.append(collector.finalize(decision_id))
    team = aggregate_team_outcomes(results, bunching_threshold_seconds=35.0, reference_fleet_size=4)
    assert team["headway_values_seconds"] == [30.0, 40.0]
    assert team["headway_cv"] == pytest.approx(1.0 / 7.0)
    assert team["bunching_rate"] == 0.5
    assert team["fleet_reduction"] == 0.25
    assert team["intervention_count_candidate"] == 1


def test_k4_identity_log_maps_to_causal_reward_events() -> None:
    state = ServiceObligationStateMachine()
    state.register_vehicle(0, "PV8-0")
    state.register_stop("S0")
    state.register_stop("S1")
    state.passenger_waiting(passenger_id="P1", pickup_stop="S0", dropoff_stop="S1", event_ts=110)
    state.request_created(request_id="R1", passenger_id="P1", service_leg_id="L1", event_ts=111)
    state.request_assigned(request_id="R1", agent_id=0, vehicle_token="PV8-0", event_ts=112)
    state.passenger_boarded(request_id="R1", passenger_id="P1", agent_id=0, vehicle_token="PV8-0", stop_id="S0", event_ts=140)
    state.passenger_alighted(request_id="R1", passenger_id="P1", agent_id=0, vehicle_token="PV8-0", stop_id="S1", event_ts=180)
    state.request_completed(request_id="R1", event_ts=181)
    mapped = map_k4_event_log(state.event_log, agent_id=0, vehicle_token="PV8-0", event_id_prefix="K4")
    collector = CausalOutcomeCollector()
    collector.register_decision(binding())
    for row in mapped:
        collector.record_event("D0", row)
    result = collector.finalize("D0")
    assert result["passenger_generated_count"] == 1
    assert result["boarding_completion_count"] == 1
    assert result["alighting_completion_count"] == 1
    assert result["passenger_served_count"] == 1
    assert result["avg_wait_seconds"] == 30.0
