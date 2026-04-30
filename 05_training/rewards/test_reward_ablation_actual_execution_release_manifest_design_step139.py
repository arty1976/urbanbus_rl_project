from __future__ import annotations

from reward_ablation_actual_execution_release_manifest_design_step139 import build_release_manifest_template
from validate_reward_ablation_actual_execution_release_manifest_design_step139 import validate_payload


def base_payload_and_template() -> tuple[dict, dict]:
    upstream = [
        {"name": "step135_readiness_lock", "manifest_path": "m135.json"},
        {"name": "step136_release_request_package", "manifest_path": "m136.json"},
        {"name": "step137_runner_review", "manifest_path": "m137.json"},
        {"name": "step138_project_log_update", "manifest_path": "m138.json"},
    ]
    template = build_release_manifest_template(upstream)
    payload = {
        "artifact_version": "reward_ablation_actual_execution_release_manifest_design_step139_v1",
        "step": 139,
        "audit_status": "PASS",
        "design_status": "RELEASE_MANIFEST_SCHEMA_DESIGNED_ACTUAL_STILL_LOCKED",
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
    return payload, template


def test_pass_payload_and_template() -> None:
    payload, template = base_payload_and_template()
    result = validate_payload(payload, template, require_pass=True)
    assert result["validation_status"] == "PASS", result


def test_payload_actual_execution_true_fails() -> None:
    payload, template = base_payload_and_template()
    payload["actual_execution_allowed"] = True
    result = validate_payload(payload, template, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_execution_allowed" in e for e in result["errors"])


def test_template_release_ready_true_fails() -> None:
    payload, template = base_payload_and_template()
    template["release_decision"]["release_ready"] = True
    result = validate_payload(payload, template, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("release_ready" in e for e in result["errors"])


def test_wrong_matrix_fails() -> None:
    payload, template = base_payload_and_template()
    template["requested_matrix"]["planned_run_count"] = 17
    result = validate_payload(payload, template, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("planned_run_count" in e for e in result["errors"])


def main() -> None:
    test_pass_payload_and_template()
    test_payload_actual_execution_true_fails()
    test_template_release_ready_true_fails()
    test_wrong_matrix_fails()
    print("[OK] Step 139 release manifest design self-test PASS")


if __name__ == "__main__":
    main()
