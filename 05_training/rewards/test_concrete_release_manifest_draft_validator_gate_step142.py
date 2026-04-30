from __future__ import annotations

from validate_concrete_release_manifest_draft_validator_gate_step142 import validate_payload


def base_payload() -> dict:
    return {
        "artifact_version": "concrete_release_manifest_draft_validator_gate_step142_v1",
        "step": 142,
        "audit_status": "PASS",
        "gate_status": "A_RELEASE_DRAFT_VALIDATED_AND_B_GROUP_BASELINES_READY_ACTUAL_STILL_LOCKED",
        "a_release_draft_validated": True,
        "b_group_baseline_ready": True,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "actual_executed": False,
        "actual_results": False,
        "reward_result_written": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "final_reward_design_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }


def test_pass_payload() -> None:
    result = validate_payload(base_payload(), require_pass=True)
    assert result["validation_status"] == "PASS", result


def test_actual_execution_true_fails() -> None:
    payload = base_payload()
    payload["actual_execution_allowed"] = True
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_execution_allowed" in e for e in result["errors"])


def test_missing_a_validation_fails() -> None:
    payload = base_payload()
    payload["a_release_draft_validated"] = False
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("a_release_draft_not_validated" in e for e in result["errors"])


def test_missing_b_baseline_fails() -> None:
    payload = base_payload()
    payload["b_group_baseline_ready"] = False
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("b_group_baseline_not_ready" in e for e in result["errors"])


def test_blocked_payload_can_pass_with_allow_blocked() -> None:
    payload = base_payload()
    payload["audit_status"] = "BLOCKED"
    payload["gate_status"] = "RELEASE_DRAFT_VALIDATOR_GATE_BLOCKED"
    payload["a_release_draft_validated"] = False
    result = validate_payload(payload, require_pass=False)
    assert result["validation_status"] == "PASS", result


def main() -> None:
    test_pass_payload()
    test_actual_execution_true_fails()
    test_missing_a_validation_fails()
    test_missing_b_baseline_fails()
    test_blocked_payload_can_pass_with_allow_blocked()
    print("[OK] Step 142 concrete release manifest draft validator gate self-test PASS")


if __name__ == "__main__":
    main()
