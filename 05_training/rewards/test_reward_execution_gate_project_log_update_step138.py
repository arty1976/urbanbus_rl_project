from __future__ import annotations

from validate_reward_execution_gate_project_log_update_step138 import validate_payload


def base_payload() -> dict:
    return {
        "artifact_version": "reward_execution_gate_project_log_update_step138_v1",
        "step": 138,
        "audit_status": "PASS",
        "project_log_updated": True,
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


def base_log() -> str:
    return """
<!-- STEP138_REWARD_EXECUTION_GATE_UPDATE_START -->
## Step 138 - Reward execution gate project log update
- Step 131: actual reward ablation execution environment preflight
- Step 132: reward ablation command dry-run executor
- Step 133: actual reward ablation runner guard
- Step 134: command-to-guarded-runner integration
- Step 135: actual execution readiness lock
- Step 136: actual execution release request package
- Step 137: actual runner implementation review gate
- actual_execution_allowed = false
- actual_results = false
- winner_selected = false
- train_with_this_reward_allowed = false
- paper_level_claim_allowed = false
- causal_performance_claim_allowed = false
<!-- STEP138_REWARD_EXECUTION_GATE_UPDATE_END -->
"""


def test_pass_payload_and_log() -> None:
    result = validate_payload(base_payload(), base_log(), require_pass=True)
    assert result["validation_status"] == "PASS", result


def test_true_guard_fails() -> None:
    payload = base_payload()
    payload["actual_results"] = True
    result = validate_payload(payload, base_log(), require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_results" in e for e in result["errors"])


def test_missing_log_token_fails() -> None:
    log = base_log().replace("Step 137: actual runner implementation review gate", "")
    result = validate_payload(base_payload(), log, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("project_log_required_token_missing" in e for e in result["errors"])


def test_blocked_payload_can_validate_with_allow_blocked() -> None:
    payload = base_payload()
    payload["audit_status"] = "BLOCKED"
    payload["project_log_updated"] = False
    result = validate_payload(payload, base_log(), require_pass=False)
    assert result["validation_status"] == "PASS", result


def main() -> None:
    test_pass_payload_and_log()
    test_true_guard_fails()
    test_missing_log_token_fails()
    test_blocked_payload_can_validate_with_allow_blocked()
    print("[OK] Step 138 project log update self-test PASS")


if __name__ == "__main__":
    main()
