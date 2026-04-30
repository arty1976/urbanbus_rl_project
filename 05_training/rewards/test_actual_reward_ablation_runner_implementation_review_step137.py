from __future__ import annotations

from validate_actual_reward_ablation_runner_implementation_review_step137 import validate_payload


def base_payload() -> dict:
    return {
        "artifact_version": "actual_reward_ablation_runner_implementation_review_step137_v1",
        "step": 137,
        "audit_status": "PASS",
        "review_status": "RUNNER_GUARD_REVIEW_PASS_ACTUAL_STILL_LOCKED",
        "static_review": {"violations": []},
        "dynamic_review": {
            "violations": [],
            "actual_attempt": {
                "returncode": 2,
                "audit_status": "BLOCKED",
            },
        },
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
    payload = base_payload()
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "PASS", result


def test_actual_execution_allowed_fails() -> None:
    payload = base_payload()
    payload["actual_execution_allowed"] = True
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_execution_allowed" in e for e in result["errors"])


def test_actual_attempt_zero_return_fails() -> None:
    payload = base_payload()
    payload["dynamic_review"]["actual_attempt"]["returncode"] = 0
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_attempt_must_return_nonzero" in e for e in result["errors"])


def test_actual_attempt_pass_status_fails() -> None:
    payload = base_payload()
    payload["dynamic_review"]["actual_attempt"]["audit_status"] = "PASS"
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_attempt_audit_status_must_be_blocked" in e for e in result["errors"])


def test_blocked_payload_can_validate_with_allow_blocked() -> None:
    payload = base_payload()
    payload["audit_status"] = "BLOCKED"
    payload["review_status"] = "RUNNER_GUARD_REVIEW_BLOCKED"
    result = validate_payload(payload, require_pass=False)
    assert result["validation_status"] == "PASS", result


def main() -> None:
    test_pass_payload()
    test_actual_execution_allowed_fails()
    test_actual_attempt_zero_return_fails()
    test_actual_attempt_pass_status_fails()
    test_blocked_payload_can_validate_with_allow_blocked()
    print("[OK] Step 137 runner implementation review self-test PASS")


if __name__ == "__main__":
    main()
