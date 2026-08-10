from __future__ import annotations

import copy

from simulator.k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime, RuntimeVersionBinding
from simulator.k_safety_state import ServiceObligationStateMachine
from simulator.zero_loss_admission_adapter import (
    ZERO_LOSS_ELIGIBLE,
    ZERO_LOSS_INELIGIBLE,
    ZeroLossAdmissionAdapter,
    ZeroLossKMaskAdmissionRuntime,
    canonical_hash,
)


def route_rows():
    rows = []
    for idx in range(6):
        rows.append(
            {
                "route_stop_occurrence_id": f"R:0:{idx}:S{idx}",
                "route_id": "R",
                "direction_id": "0",
                "stop_sequence": idx,
                "stop_id": f"S{idx}",
                "travel_seconds_to_next": 30.0,
            }
        )
    return rows


def candidate(attempt_id: str = "ATT_ZL1"):
    return {
        "attempt_id": attempt_id,
        "candidate_passenger_id": "PNEW",
        "candidate_request_id": "RNEW",
        "candidate_pickup_stop_id": "S1",
        "candidate_dropoff_stop_id": "S4",
        "route_id": "R",
        "direction_id": "0",
    }


def attention(attempt_id: str = "ATT_ZL1", *, weight_shift: float = 0.0):
    return [
        {
            "attempt_id": attempt_id,
            "state_ts": "10",
            "layer_id": 1,
            "head_id": 0,
            "src_node": "S0",
            "dst_node": "S1",
            "attention_weight": 0.70 + weight_shift,
            "path_segment": "candidate_pickup_path",
            "filter_match_type": "edge_overlap",
            "attention_source": "gatv2conv_return_attention_weights",
            "attention_connection_mode": "attempt_route_path_edge_filter_step165",
            "real_gatv2conv_attention_extracted": True,
        },
        {
            "attempt_id": attempt_id,
            "state_ts": "10",
            "layer_id": 1,
            "head_id": 1,
            "src_node": "S1",
            "dst_node": "S2",
            "attention_weight": 0.55 + weight_shift,
            "path_segment": "existing_passenger_path",
            "filter_match_type": "edge_overlap",
            "attention_source": "gatv2conv_return_attention_weights",
            "attention_connection_mode": "attempt_route_path_edge_filter_step165",
            "real_gatv2conv_attention_extracted": True,
        },
    ]


def base_state(dropoffs, *, include_candidate: bool = False):
    state = ServiceObligationStateMachine()
    state.register_vehicle(0, "V0")
    state.register_stop("__K8_PREVIOUS__")
    for row in route_rows():
        state.register_stop(row["stop_id"])

    for idx, dropoff in enumerate(dropoffs):
        passenger_id = f"P{idx}"
        request_id = f"R{idx}"
        state.passenger_waiting(passenger_id=passenger_id, pickup_stop="S0", dropoff_stop=dropoff, event_ts=0)
        state.request_created(request_id=request_id, passenger_id=passenger_id, service_leg_id=f"L{idx}", event_ts=0)
        state.request_assigned(request_id=request_id, agent_id=0, vehicle_token="V0", event_ts=0)
    for idx, _dropoff in enumerate(dropoffs):
        state.passenger_boarded(
            request_id=f"R{idx}",
            passenger_id=f"P{idx}",
            agent_id=0,
            vehicle_token="V0",
            stop_id="S0",
            event_ts=1,
        )

    if include_candidate:
        state.passenger_waiting(passenger_id="PNEW", pickup_stop="S1", dropoff_stop="S4", event_ts=2)
        state.request_created(request_id="RNEW", passenger_id="PNEW", service_leg_id="LNEW", event_ts=2)
        state.request_assigned(request_id="RNEW", agent_id=0, vehicle_token="V0", event_ts=2)
    return state


def adapter_result(dropoffs, *, attempt_id: str = "ATT_ZL1", include_candidate: bool = False, weight_shift: float = 0.0):
    state = base_state(dropoffs, include_candidate=include_candidate)
    return ZeroLossAdmissionAdapter().evaluate(
        obligation_state_machine=state,
        agent_id=0,
        vehicle_token="V0",
        decision_ts=10,
        current_stop_id="S0",
        route_rows=route_rows(),
        candidate=candidate(attempt_id),
        attention_evidence=attention(attempt_id, weight_shift=weight_shift),
    )


def runtime():
    binding = RuntimeVersionBinding("STATE_V1", "RULE_V1", "a" * 64, "b" * 64, "DYNAMIC_V1", "MASK_V1", "EXP_V1")
    approval = {
        "approval_valid": True,
        "research_rule_approved": True,
        "real_network_rule_claim_allowed": False,
        "static_rulebook_sha256": "a" * 64,
        "occurrence_master_sha256": "b" * 64,
    }
    rule = {
        "route_stop_occurrence_id": "R:0:1:S1",
        "route_id": "R",
        "direction_id": "0",
        "stop_sequence": 1,
        "stop_id": "S1",
        "post_skip_target_stop_id": "S2",
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
    base = FixedVehicleOccurrenceMaskRuntime(
        rule_rows=[rule],
        fixed_vehicle_bindings={0: "V0"},
        approval_record=approval,
        version_binding=binding,
        actual_rulebook_sha256="a" * 64,
        actual_occurrence_master_sha256="b" * 64,
    )
    return ZeroLossKMaskAdmissionRuntime(base_runtime=base)


def run_runtime(state):
    return runtime().evaluate(
        agent_id=0,
        vehicle_token="V0",
        active_bus_mask=True,
        route_id="R",
        direction_id="0",
        stop_sequence=1,
        stop_id="S1",
        route_stop_occurrence_id="R:0:1:S1",
        obligation_state_machine=state,
        decision_ts=10,
        current_stop_id="S0",
        route_rows=route_rows(),
        candidate=candidate(),
        attention_evidence=attention(),
    )


def test_a_all_onboard_passengers_delta_le_zero_accepts():
    result = adapter_result(["S0"])
    assert result["candidate_admission_state"] == ZERO_LOSS_ELIGIBLE
    assert result["zero_loss_accept"] is True
    assert result["per_passenger"][0]["delta_eta_sec"] == 0


def test_b_one_passenger_positive_delta_rejects():
    result = adapter_result(["S3"])
    assert result["candidate_admission_state"] == ZERO_LOSS_INELIGIBLE
    assert result["zero_loss_accept"] is False
    assert result["per_passenger"][0]["delta_eta_sec"] > 0


def test_c_multiple_passengers_with_mixed_deltas_rejects():
    result = adapter_result(["S0", "S3"])
    assert result["zero_loss_accept"] is False
    assert [row["threshold_pass"] for row in result["per_passenger"]] == [True, False]


def test_d_exact_zero_delta_accepts():
    result = adapter_result(["S1"])
    assert result["zero_loss_accept"] is True
    assert result["per_passenger"][0]["delta_eta_sec"] == 0


def test_e_rejected_candidate_preserves_existing_mandatory_alighting():
    state = base_state(["S1", "S3"], include_candidate=True)
    result = run_runtime(state)
    snapshot = result["decision_time_obligation_snapshot"]
    assert result["zero_loss_candidate_pickup_executable"] is False
    assert "R0" in snapshot["onboard_destination_request_ids"]
    assert snapshot["dropoff_obligation"] is True
    assert snapshot["alighting_obligation"] is True


def test_f_zero_loss_rejection_removes_candidate_before_action():
    state = base_state(["S3"], include_candidate=True)
    state_hash = canonical_hash(state.to_payload())
    result = run_runtime(state)
    snapshot = result["decision_time_obligation_snapshot"]
    assert result["zero_loss_candidate_removed_before_action"] is True
    assert "RNEW" not in snapshot["assigned_pickup_request_ids"]
    assert "PNEW" not in snapshot["waiting_queue"]
    assert canonical_hash(state.to_payload()) == state_hash


def test_g_attention_changes_do_not_change_eligibility_or_eta():
    base = adapter_result(["S3"], attempt_id="ATT_A", weight_shift=0.0)
    changed = adapter_result(["S3"], attempt_id="ATT_A", weight_shift=0.1)
    assert changed["zero_loss_accept"] == base["zero_loss_accept"]
    assert changed["per_passenger"] == base["per_passenger"]
    assert changed["attention_evidence"]["attention_mass_total"] != base["attention_evidence"]["attention_mass_total"]


def test_h_identical_input_replay_is_deterministic_for_eta_mask_and_evidence():
    state = base_state(["S3"], include_candidate=True)
    first = run_runtime(copy.deepcopy(state))
    second = run_runtime(copy.deepcopy(state))
    assert canonical_hash(first["zero_loss_admission"]) == canonical_hash(second["zero_loss_admission"])
    assert first["action_mask"] == second["action_mask"]
    assert first["zero_loss_kmask_state_hash"] == second["zero_loss_kmask_state_hash"]
    assert first["zero_loss_admission"]["future_leakage_count"] == 0
    assert first["zero_loss_admission"]["nan_inf_count"] == 0
    assert first["zero_loss_candidate_pickup_executable"] is False
    assert first["zero_loss_candidate_removed_before_action"] is True
