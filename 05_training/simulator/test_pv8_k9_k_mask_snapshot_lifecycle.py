from __future__ import annotations

import copy

import pytest

from simulator.k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime, RuntimeVersionBinding
from simulator.k_mask_snapshot_lifecycle import (
    GlobalKMaskSnapshot,
    GlobalKMaskSnapshotLifecycle,
    KMaskAgentContext,
    KMaskSnapshotInvariantError,
    bind_global_k_mask_to_dynamics_state,
    clone_global_k_mask_snapshot,
    deserialize_global_k_mask_snapshot,
    extract_global_k_mask_from_dynamics_state,
    serialize_global_k_mask_snapshot,
)
from simulator.k_safety_state import ServiceObligationStateMachine
from simulator.dynamics_state_snapshot import (
    clone_dynamics_state,
    deserialize_dynamics_state,
    reset_runtime_state_from_snapshot,
    serialize_dynamics_state,
)


def rule(agent_id: int):
    return {
        "route_stop_occurrence_id": f"RSO1_TEST_{agent_id}",
        "route_id": "R1",
        "direction_id": "0",
        "stop_sequence": agent_id + 1,
        "stop_id": f"S{agent_id}",
        "post_skip_target_stop_id": f"P{agent_id}",
        "rule_version": "RULE_V1",
        "rule_class": "CONTRACT_FIXED_RESEARCH_RULE",
        "static_rule_complete": True,
        "mandatory_stop": False,
        "protected_stop": False,
        "planned_itinerary_allows_skip": True,
        "terminal_or_turnaround_stop": False,
        "charging_or_driver_relief_stop": False,
        "valid_post_skip_path": True,
    }


def lifecycle():
    binding = RuntimeVersionBinding("STATE_V1", "RULE_V1", "a" * 64, "b" * 64, "DYNAMIC_V1", "MASK_V1", "EXP_V1")
    approval = {
        "approval_valid": True,
        "research_rule_approved": True,
        "real_network_rule_claim_allowed": False,
        "static_rulebook_sha256": "a" * 64,
        "occurrence_master_sha256": "b" * 64,
    }
    runtime = FixedVehicleOccurrenceMaskRuntime(
        rule_rows=[rule(agent) for agent in range(8)],
        fixed_vehicle_bindings={agent: f"V{agent}" for agent in range(8)},
        approval_record=approval,
        version_binding=binding,
        actual_rulebook_sha256="a" * 64,
        actual_occurrence_master_sha256="b" * 64,
    )
    return GlobalKMaskSnapshotLifecycle(runtime, binding)


def context(agent_id: int, *, active: bool = True, decision_ts: int = 0):
    row = rule(agent_id)
    state = ServiceObligationStateMachine()
    state.register_vehicle(agent_id, f"V{agent_id}")
    state.register_stop("__K8_PREVIOUS__")
    state.register_stop(row["stop_id"])
    state.register_stop(row["post_skip_target_stop_id"])
    return KMaskAgentContext(
        agent_id=agent_id,
        vehicle_token=f"V{agent_id}",
        active_bus_mask=active,
        decision_ts=decision_ts,
        obligation_state_machine=state,
        route_id=row["route_id"] if active else None,
        direction_id=row["direction_id"] if active else None,
        stop_sequence=row["stop_sequence"] if active else None,
        stop_id=row["stop_id"] if active else None,
        route_stop_occurrence_id=row["route_stop_occurrence_id"] if active else None,
    )


def snapshot():
    contexts = [context(agent, active=agent != 7) for agent in range(8)]
    return lifecycle().capture_cycle(cycle_index=1, cycle_timestamp="2026-08-08T00:00:00Z", contexts=contexts)


def test_capture_enforces_active_and_inactive_mask_invariants():
    rows = snapshot().to_payload()["agent_snapshots"]
    assert all(row["k_action_mask"] == [True, True, True] for row in rows[:7])
    assert rows[7]["k_action_mask"] == [False, False, False]
    assert rows[7]["actor_sampling_bypassed"] is True


def test_serialize_clone_restore_recompute_is_deterministic():
    original = snapshot()
    serialized = serialize_global_k_mask_snapshot(original)
    restored = deserialize_global_k_mask_snapshot(serialized)
    cloned = clone_global_k_mask_snapshot(restored)
    recomputed = lifecycle().recompute(cloned)
    assert original.snapshot_hash == restored.snapshot_hash == cloned.snapshot_hash == recomputed.snapshot_hash


def test_pending_future_obligation_does_not_leak_before_decision_boundary():
    contexts = [context(agent) for agent in range(8)]
    state = contexts[0].obligation_state_machine
    state.schedule_transition("passenger_waiting", 11, passenger_id="P0", pickup_stop="S0", dropoff_stop="P0")
    state.schedule_transition("request_created", 12, request_id="R0", passenger_id="P0", service_leg_id="L0")
    state.schedule_transition("request_assigned", 13, request_id="R0", agent_id=0, vehicle_token="V0")
    contexts[0] = KMaskAgentContext(**{**contexts[0].__dict__, "decision_ts": 10})
    original = lifecycle().capture_cycle(cycle_index=2, cycle_timestamp=None, contexts=contexts)
    recomputed = lifecycle().recompute(original)
    row = recomputed.to_payload()["agent_snapshots"][0]
    assert row["conditional_skip_valid"] is True
    assert row["decision_time_obligation_snapshot"]["service_obligation"] is False


def test_existing_global_dynamics_snapshot_lifecycle_preserves_k_mask_binding():
    original = snapshot()
    dynamics = bind_global_k_mask_to_dynamics_state(original)
    serialized = serialize_dynamics_state(dynamics)
    restored = deserialize_dynamics_state(serialized)
    cloned = clone_dynamics_state(restored)
    reset = reset_runtime_state_from_snapshot(cloned)
    extracted = extract_global_k_mask_from_dynamics_state(cloned)
    recomputed = lifecycle().recompute(extracted)
    assert dynamics.state_hash == restored.state_hash == cloned.state_hash
    assert reset["hash_match"] is True
    assert original.snapshot_hash == extracted.snapshot_hash == recomputed.snapshot_hash


def test_snapshot_rejects_active_all_false_mask():
    payload = snapshot().to_payload()
    payload["agent_snapshots"][0]["k_action_mask"] = [False, False, False]
    with pytest.raises(KMaskSnapshotInvariantError):
        GlobalKMaskSnapshot(copy.deepcopy(payload))


def test_snapshot_rejects_version_drift():
    payload = snapshot().to_payload()
    payload["agent_snapshots"][0]["static_rulebook_sha256"] = "c" * 64
    with pytest.raises(Exception):
        GlobalKMaskSnapshot(payload)
