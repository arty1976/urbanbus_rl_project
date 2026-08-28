from __future__ import annotations

from simulator.k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime, RuntimeVersionBinding
from simulator.k_safety_state import ServiceObligationStateMachine


def rule(*, occurrence_id: str = "RSO1_TEST", terminal: bool = False, post: bool = True):
    return {
        "route_stop_occurrence_id": occurrence_id,
        "route_id": "R1",
        "direction_id": "0",
        "stop_sequence": 2,
        "stop_id": "S1",
        "post_skip_target_stop_id": "S2" if post else None,
        "rule_version": "RULE_V1",
        "rule_class": "CONTRACT_FIXED_RESEARCH_RULE",
        "static_rule_complete": True,
        "mandatory_stop": False,
        "protected_stop": False,
        "planned_itinerary_allows_skip": post,
        "terminal_or_turnaround_stop": terminal,
        "charging_or_driver_relief_stop": False,
        "valid_post_skip_path": post,
    }


def runtime(row=None):
    binding = RuntimeVersionBinding("STATE_V1", "RULE_V1", "a" * 64, "b" * 64, "DYNAMIC_V1", "MASK_V1", "EXP_V1")
    approval = {
        "approval_valid": True,
        "research_rule_approved": True,
        "real_network_rule_claim_allowed": False,
        "static_rulebook_sha256": "a" * 64,
        "occurrence_master_sha256": "b" * 64,
    }
    return FixedVehicleOccurrenceMaskRuntime(
        rule_rows=[row or rule()],
        fixed_vehicle_bindings={0: "V0"},
        approval_record=approval,
        version_binding=binding,
        actual_rulebook_sha256="a" * 64,
        actual_occurrence_master_sha256="b" * 64,
    )


def machine():
    state = ServiceObligationStateMachine()
    state.register_vehicle(0, "V0")
    state.register_stop("__K8_PREVIOUS__")
    state.register_stop("S1")
    state.register_stop("S2")
    return state


def evaluate(adapter, *, active=True, token="V0", state=None):
    return adapter.evaluate(
        agent_id=0,
        vehicle_token=token,
        active_bus_mask=active,
        route_id="R1" if active else None,
        direction_id="0" if active else None,
        stop_sequence=2 if active else None,
        stop_id="S1" if active else None,
        route_stop_occurrence_id="RSO1_TEST" if active else None,
        obligation_state_machine=state,
        decision_ts=0,
    )


def test_clear_approved_research_rule_allows_skip_without_policy_execution():
    result = evaluate(runtime(), state=machine())
    assert result["action_mask"] == [True, True, True]
    assert result["policy_execution_count"] == 0


def test_inactive_agent_has_no_action_leakage():
    result = evaluate(runtime(), active=False)
    assert result["action_mask"] == [False, False, False]
    assert result["inactive_action_leakage"] is False


def test_identity_mismatch_fails_closed_before_mask_evaluation():
    result = evaluate(runtime(), token="WRONG", state=machine())
    assert result["skip_valid"] is False
    assert result["identity_failure"] is True


def test_terminal_and_no_target_rule_fails_closed():
    result = evaluate(runtime(rule(terminal=True, post=False)), state=machine())
    assert result["skip_valid"] is False
    assert "TERMINAL_OR_TURNAROUND_STOP" in result["skip_invalid_reason_codes"]
    assert "NO_POST_SKIP_TARGET" in result["skip_invalid_reason_codes"]
