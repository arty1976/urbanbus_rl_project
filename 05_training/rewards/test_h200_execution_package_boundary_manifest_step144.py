from __future__ import annotations

from validate_h200_execution_package_boundary_manifest_step144 import validate_payload


def base_payload_and_boundary() -> tuple[dict, dict]:
    payload = {
        "artifact_version": "h200_execution_package_boundary_manifest_step144_v1",
        "step": 144,
        "audit_status": "PASS",
        "package_status": "H200_EXECUTION_PACKAGE_BOUNDARY_READY_ACTUAL_STILL_LOCKED",
        "planned_run_count": 72,
        "conditions": ["A", "A90", "A80", "A70"],
        "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
        "seeds": [1, 2, 3],
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
    boundary = {
        "package_boundary_version": "h200_execution_package_boundary_v1",
        "created_from_step": 144,
        "matrix_source": {
            "planned_run_count": 72,
            "conditions": ["A", "A90", "A80", "A70"],
            "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
            "seeds": [1, 2, 3],
        },
        "execution_boundary": {
            "actual_execution_allowed": False,
            "actual_execution_released": False,
            "release_ready": False,
        },
        "non_claim_guards": {
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "train_with_this_reward_allowed": False,
            "actual_training_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    }
    return payload, boundary


def test_pass_payload_and_boundary() -> None:
    payload, boundary = base_payload_and_boundary()
    result = validate_payload(payload, boundary, require_pass=True)
    assert result["validation_status"] == "PASS", result


def test_actual_execution_true_fails() -> None:
    payload, boundary = base_payload_and_boundary()
    payload["actual_execution_allowed"] = True
    result = validate_payload(payload, boundary, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_execution_allowed" in e for e in result["errors"])


def test_wrong_run_count_fails() -> None:
    payload, boundary = base_payload_and_boundary()
    boundary["matrix_source"]["planned_run_count"] = 18
    result = validate_payload(payload, boundary, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("planned_run_count" in e for e in result["errors"])


def test_release_ready_true_fails() -> None:
    payload, boundary = base_payload_and_boundary()
    boundary["execution_boundary"]["release_ready"] = True
    result = validate_payload(payload, boundary, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("release_ready" in e for e in result["errors"])


def test_blocked_payload_can_pass_with_allow_blocked() -> None:
    payload, boundary = base_payload_and_boundary()
    payload["audit_status"] = "BLOCKED"
    payload["package_status"] = "H200_EXECUTION_PACKAGE_BOUNDARY_BLOCKED"
    result = validate_payload(payload, boundary, require_pass=False)
    assert result["validation_status"] == "PASS", result


def main() -> None:
    test_pass_payload_and_boundary()
    test_actual_execution_true_fails()
    test_wrong_run_count_fails()
    test_release_ready_true_fails()
    test_blocked_payload_can_pass_with_allow_blocked()
    print("[OK] Step 144 H200 execution package boundary manifest self-test PASS")


if __name__ == "__main__":
    main()
