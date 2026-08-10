from __future__ import annotations

import pytest

from simulator.k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime, RuntimeVersionBinding
from simulator.pv8_b1_orchestrator import (
    PassengerRequestScheduleRow,
    PV8B1OrchestratorError,
    TypedPV8B1Orchestrator,
)


def fixture_components():
    bindings = {agent: f"PV8-{agent}" for agent in range(8)}
    rules = []
    for direction in ("0", "1"):
        for sequence in range(5):
            stop_id = f"S{direction}-{sequence}"
            rules.append({
                "route_stop_occurrence_id": f"R:{direction}:{sequence}:{stop_id}",
                "route_id": "R",
                "direction_id": direction,
                "stop_sequence": sequence,
                "occurrence_index": sequence,
                "stop_id": stop_id,
                "post_skip_target_stop_id": f"S{direction}-{sequence + 2}" if sequence < 3 else None,
                "rule_version": "TEST_RULE_V1",
                "rule_class": "CONTRACT_FIXED_RESEARCH_RULE",
                "static_rule_complete": True,
                "mandatory_stop": False,
                "protected_stop": False,
                "planned_itinerary_allows_skip": sequence < 3,
                "terminal_or_turnaround_stop": sequence == 4,
                "charging_or_driver_relief_stop": False,
                "valid_post_skip_path": sequence < 3,
            })
    version = RuntimeVersionBinding(
        k_safety_state_version="K4",
        static_rulebook_version="TEST_RULE_V1",
        static_rulebook_sha256="rule-hash",
        occurrence_master_sha256="occurrence-hash",
        dynamic_state_contract_version="K4_DYNAMIC",
        mask_predicate_version="K4_FAIL_CLOSED_V2",
        experiment_version="R7_TEST",
    )
    approval = {
        "approval_valid": True,
        "research_rule_approved": True,
        "real_network_rule_claim_allowed": False,
        "static_rulebook_sha256": "rule-hash",
        "occurrence_master_sha256": "occurrence-hash",
    }
    runtime = FixedVehicleOccurrenceMaskRuntime(
        rule_rows=rules,
        fixed_vehicle_bindings=bindings,
        approval_record=approval,
        version_binding=version,
        actual_rulebook_sha256="rule-hash",
        actual_occurrence_master_sha256="occurrence-hash",
    )
    return TypedPV8B1Orchestrator(
        runtime=runtime,
        version_binding=version,
        rule_rows=rules,
        fixed_vehicle_bindings=bindings,
    ), rules


def schedule(*, cancellation_ts=None, request_ts=110):
    return PassengerRequestScheduleRow(
        passenger_id="P1",
        request_id="Q1",
        request_ts=request_ts,
        origin_stop="S0-1",
        destination_stop="S0-2",
        route_id="R",
        direction_id="0",
        origin_stop_sequence=1,
        destination_stop_sequence=2,
        agent_id=0,
        vehicle_token="PV8-0",
        cancellation_ts=cancellation_ts,
    )


def test_full_identity_service_chain_and_h4_metrics() -> None:
    orchestrator, rules = fixture_components()
    state = orchestrator.new_state()
    orchestrator.apply_schedule(state, [schedule()])
    result = orchestrator.run_h4_fixture(
        fixture_id="FULL",
        state=state,
        focal_agent_id=0,
        route_segment=[row for row in rules if row["direction_id"] == "0"][:4],
        decision_ts=100,
    )
    outcome = result["focal_outcome"]
    assert result["snapshot_agent_count"] == 8
    assert result["inactive_learning_sample_count"] == 7
    assert result["conditional_skip_selected"] is False
    assert result["state_integrity"]["passed"] is True
    assert outcome["passenger_generated_count"] == 1
    assert outcome["passenger_served_count"] == 1
    assert outcome["service_rate"] == 1.0
    assert outcome["avg_wait_seconds"] == 21.0
    assert outcome["p95_wait_seconds"] == 21.0
    assert all(result["lifecycle_checks"].values())


def test_cancellation_preserves_identity_without_service() -> None:
    orchestrator, rules = fixture_components()
    state = orchestrator.new_state()
    orchestrator.apply_schedule(state, [schedule(cancellation_ts=120)])
    result = orchestrator.run_h4_fixture(
        fixture_id="CANCEL",
        state=state,
        focal_agent_id=0,
        route_segment=[row for row in rules if row["direction_id"] == "0"][:4],
        decision_ts=100,
    )
    outcome = result["focal_outcome"]
    assert result["state_integrity"]["passed"] is True
    assert outcome["passenger_generated_count"] == 1
    assert outcome["passenger_cancelled_count"] == 1
    assert outcome["passenger_served_count"] == 0


def test_future_request_does_not_enter_decision_mask_or_h4() -> None:
    orchestrator, rules = fixture_components()
    state = orchestrator.new_state()
    orchestrator.apply_schedule(state, [schedule(request_ts=341)])
    result = orchestrator.run_h4_fixture(
        fixture_id="FUTURE",
        state=state,
        focal_agent_id=0,
        route_segment=[row for row in rules if row["direction_id"] == "0"][:4],
        decision_ts=100,
    )
    assert result["future_information_violation"] is False
    assert result["focal_outcome"]["passenger_generated_count"] == 0
    assert result["pending_future_transition_count"] == 3


def test_reappearance_and_direction_transition_preserve_slot_and_token() -> None:
    orchestrator, rules = fixture_components()
    state = orchestrator.new_state()
    direction0 = next(row for row in rules if row["direction_id"] == "0" and row["stop_sequence"] == 0)
    direction1 = next(row for row in rules if row["direction_id"] == "1" and row["stop_sequence"] == 0)
    inactive = orchestrator.capture_snapshot(
        state=state,
        cycle_index=1,
        decision_ts=10,
        occurrence_by_agent={},
        active_by_agent={},
    )
    reappeared = orchestrator.capture_snapshot(
        state=state,
        cycle_index=2,
        decision_ts=20,
        occurrence_by_agent={0: direction0},
        active_by_agent={0: True},
        identity_transition_by_agent={0: "REAPPEARED_SAME_IDENTITY"},
    )
    transitioned = orchestrator.capture_snapshot(
        state=state,
        cycle_index=3,
        decision_ts=30,
        occurrence_by_agent={0: direction1},
        active_by_agent={0: True},
        identity_transition_by_agent={0: "DIRECTION_TRANSITION_SAME_IDENTITY"},
    )
    inactive_row = inactive.to_payload()["agent_snapshots"][0]
    reappeared_row = reappeared.to_payload()["agent_snapshots"][0]
    transitioned_row = transitioned.to_payload()["agent_snapshots"][0]
    assert inactive_row["vehicle_token"] == reappeared_row["vehicle_token"] == transitioned_row["vehicle_token"]
    assert transitioned_row["direction_id"] == "1"
    assert inactive_row["actor_sampling_bypassed"] is True


def test_schedule_identity_and_claim_boundary_fail_closed() -> None:
    orchestrator, _ = fixture_components()
    state = orchestrator.new_state()
    with pytest.raises(PV8B1OrchestratorError):
        orchestrator.apply_schedule(state, [schedule(), schedule()])
    with pytest.raises(PV8B1OrchestratorError):
        PassengerRequestScheduleRow(
            passenger_id="P2",
            request_id="Q2",
            request_ts=100,
            origin_stop="S0-1",
            destination_stop="S0-2",
            route_id="R",
            direction_id="0",
            origin_stop_sequence=1,
            destination_stop_sequence=2,
            agent_id=0,
            vehicle_token="PV8-0",
            source_class="OBSERVED_EXACT",
        )
