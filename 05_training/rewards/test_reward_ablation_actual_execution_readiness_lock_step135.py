from __future__ import annotations

from validate_reward_ablation_actual_execution_readiness_lock_step135 import validate_payload


def base_payload() -> dict:
    return {
        "artifact_version": "reward_ablation_actual_execution_readiness_lock_step135_v1",
        "step": 135,
        "audit_status": "PASS",
        "readiness_lock_status": "LOCKED_READY_FOR_EXPLICIT_RELEASE",
        "upstream_checks": [
            {"name": "step131_environment_preflight", "violations": []},
            {"name": "step132_command_dry_run", "violations": []},
            {"name": "step134_command_runner_integration", "violations": []},
        ],
        "release_requirements": [
            {"requirement_id": "REQ-001", "satisfied": False},
            {"requirement_id": "REQ-002", "satisfied": False},
        ],
        "actual_execution_allowed": False,
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


def test_actual_execution_allowed_is_blocked() -> None:
    payload = base_payload()
    payload["actual_execution_allowed"] = True
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_execution_allowed" in e for e in result["errors"])


def test_release_requirement_cannot_be_satisfied_yet() -> None:
    payload = base_payload()
    payload["release_requirements"][0]["satisfied"] = True
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("release_requirement_must_not_be_satisfied_yet" in e for e in result["errors"])


def test_blocked_payload_can_validate_with_allow_blocked() -> None:
    payload = base_payload()
    payload["audit_status"] = "BLOCKED"
    payload["readiness_lock_status"] = "LOCKED_BLOCKED"
    result = validate_payload(payload, require_pass=False)
    assert result["validation_status"] == "PASS", result


def main() -> None:
    test_pass_payload()
    test_actual_execution_allowed_is_blocked()
    test_release_requirement_cannot_be_satisfied_yet()
    test_blocked_payload_can_validate_with_allow_blocked()
    print("[OK] Step 135 actual execution readiness lock self-test PASS")


if __name__ == "__main__":
    main()
