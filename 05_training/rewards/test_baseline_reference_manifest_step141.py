from __future__ import annotations

from validate_baseline_reference_manifest_step141 import validate_payload


def base_payload() -> dict:
    refs = []
    for baseline_id in ["B0R", "B1", "B2"]:
        refs.append({
            "baseline_id": baseline_id,
            "reference_status": "READY_AS_NONCAUSAL_BASELINE_REFERENCE",
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "best_reward_claim_allowed": False,
            "train_with_this_reward_allowed": False,
            "actual_training_allowed": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
        })
    return {
        "artifact_version": "baseline_reference_manifest_step141_v1",
        "step": 141,
        "audit_status": "PASS",
        "baseline_reference_status": "B_GROUP_BASELINE_REFERENCE_READY",
        "ready_baseline_count": 3,
        "required_baseline_count": 3,
        "baseline_references": refs,
        "causal_comparison_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
    }


def test_ready_payload_passes() -> None:
    result = validate_payload(base_payload(), require_all_ready=True)
    assert result["validation_status"] == "PASS", result


def test_incomplete_payload_can_pass_without_require_all_ready() -> None:
    payload = base_payload()
    payload["audit_status"] = "PASS_WITH_INCOMPLETE_BASELINES"
    payload["baseline_reference_status"] = "B_GROUP_BASELINE_REFERENCE_INCOMPLETE"
    payload["ready_baseline_count"] = 2
    payload["baseline_references"][2]["reference_status"] = "PARTIAL_REFERENCE_HAS_WINDOW_KPI"
    result = validate_payload(payload, require_all_ready=False)
    assert result["validation_status"] == "PASS", result


def test_incomplete_payload_fails_when_require_all_ready() -> None:
    payload = base_payload()
    payload["audit_status"] = "PASS_WITH_INCOMPLETE_BASELINES"
    payload["baseline_reference_status"] = "B_GROUP_BASELINE_REFERENCE_INCOMPLETE"
    payload["ready_baseline_count"] = 2
    payload["baseline_references"][2]["reference_status"] = "PARTIAL_REFERENCE_HAS_WINDOW_KPI"
    result = validate_payload(payload, require_all_ready=True)
    assert result["validation_status"] == "FAIL"
    assert any("not_all_baselines_ready" in e for e in result["errors"])


def test_causal_true_fails() -> None:
    payload = base_payload()
    payload["baseline_references"][0]["causal_comparison_allowed"] = True
    result = validate_payload(payload, require_all_ready=False)
    assert result["validation_status"] == "FAIL"
    assert any("causal_comparison_allowed" in e for e in result["errors"])


def test_wrong_baseline_set_fails() -> None:
    payload = base_payload()
    payload["baseline_references"][0]["baseline_id"] = "B0"
    result = validate_payload(payload, require_all_ready=False)
    assert result["validation_status"] == "FAIL"
    assert any("baseline_ids_mismatch" in e for e in result["errors"])


def main() -> None:
    test_ready_payload_passes()
    test_incomplete_payload_can_pass_without_require_all_ready()
    test_incomplete_payload_fails_when_require_all_ready()
    test_causal_true_fails()
    test_wrong_baseline_set_fails()
    print("[OK] Step 141 B-group baseline reference manifest self-test PASS")


if __name__ == "__main__":
    main()
