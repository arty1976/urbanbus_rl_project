from __future__ import annotations

from validate_reward_ablation_actual_execution_release_request_step136 import validate_payload


def base_payload() -> dict:
    return {
        "artifact_version": "reward_ablation_actual_execution_release_request_step136_v1",
        "step": 136,
        "audit_status": "PASS",
        "release_request_status": "REQUEST_PACKAGE_CREATED_PENDING_OPERATOR_APPROVAL",
        "requested_matrix": {
            "conditions": ["A"],
            "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
            "seeds": [1, 2, 3],
            "planned_run_count": 18,
        },
        "operator_checklist": [
            {"item_id": "OP-001", "confirmed": False},
            {"item_id": "OP-002", "confirmed": False},
        ],
        "release_manifest_required_next": True,
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


def test_actual_execution_release_is_forbidden() -> None:
    payload = base_payload()
    payload["actual_execution_released"] = True
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_execution_released" in e for e in result["errors"])


def test_operator_confirmation_is_not_allowed_yet() -> None:
    payload = base_payload()
    payload["operator_checklist"][0]["confirmed"] = True
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("operator_checklist_must_not_be_confirmed_yet" in e for e in result["errors"])


def test_wrong_matrix_fails() -> None:
    payload = base_payload()
    payload["requested_matrix"]["planned_run_count"] = 17
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("planned_run_count" in e for e in result["errors"])


def test_blocked_payload_can_validate_with_allow_blocked() -> None:
    payload = base_payload()
    payload["audit_status"] = "BLOCKED"
    payload["release_request_status"] = "REQUEST_PACKAGE_BLOCKED"
    result = validate_payload(payload, require_pass=False)
    assert result["validation_status"] == "PASS", result


def main() -> None:
    test_pass_payload()
    test_actual_execution_release_is_forbidden()
    test_operator_confirmation_is_not_allowed_yet()
    test_wrong_matrix_fails()
    test_blocked_payload_can_validate_with_allow_blocked()
    print("[OK] Step 136 release request package self-test PASS")


if __name__ == "__main__":
    main()
