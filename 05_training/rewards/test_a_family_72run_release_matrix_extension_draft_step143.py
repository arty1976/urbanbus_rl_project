from __future__ import annotations

from a_family_72run_release_matrix_extension_draft_step143 import build_run_matrix
from validate_a_family_72run_release_matrix_extension_draft_step143 import validate_payload


def base_payload_and_draft() -> tuple[dict, dict]:
    matrix = build_run_matrix()
    payload = {
        "artifact_version": "a_family_72run_release_matrix_extension_draft_step143_v1",
        "step": 143,
        "audit_status": "PASS",
        "matrix_status": "A_FAMILY_72RUN_RELEASE_MATRIX_DRAFT_READY_ACTUAL_STILL_LOCKED",
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
    draft = {
        "release_matrix_version": "a_family_72run_release_matrix_v1",
        "created_from_step": 143,
        "conditions": ["A", "A90", "A80", "A70"],
        "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
        "seeds": [1, 2, 3],
        "planned_run_count": 72,
        "run_matrix": matrix,
        "release_decision": {
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
    return payload, draft


def test_pass_payload_and_draft() -> None:
    payload, draft = base_payload_and_draft()
    result = validate_payload(payload, draft, require_pass=True)
    assert result["validation_status"] == "PASS", result


def test_run_matrix_count_is_72() -> None:
    matrix = build_run_matrix()
    assert len(matrix) == 72
    assert len({r["run_id"] for r in matrix}) == 72


def test_actual_execution_true_fails() -> None:
    payload, draft = base_payload_and_draft()
    payload["actual_execution_allowed"] = True
    result = validate_payload(payload, draft, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_execution_allowed" in e for e in result["errors"])


def test_draft_missing_condition_fails() -> None:
    payload, draft = base_payload_and_draft()
    draft["run_matrix"] = [r for r in draft["run_matrix"] if r["condition_id"] != "A70"]
    result = validate_payload(payload, draft, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("run_matrix_count" in e or "condition_count" in e for e in result["errors"])


def test_blocked_payload_can_pass_with_allow_blocked() -> None:
    payload, draft = base_payload_and_draft()
    payload["audit_status"] = "BLOCKED"
    payload["matrix_status"] = "A_FAMILY_72RUN_RELEASE_MATRIX_DRAFT_BLOCKED"
    result = validate_payload(payload, draft, require_pass=False)
    assert result["validation_status"] == "PASS", result


def main() -> None:
    test_pass_payload_and_draft()
    test_run_matrix_count_is_72()
    test_actual_execution_true_fails()
    test_draft_missing_condition_fails()
    test_blocked_payload_can_pass_with_allow_blocked()
    print("[OK] Step 143 A-family 72-run release matrix extension draft self-test PASS")


if __name__ == "__main__":
    main()
