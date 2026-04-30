from __future__ import annotations

from validate_h200_transfer_package_integrity_verifier_step146 import validate_payload


def base_payload() -> dict:
    entries = [
        {
            "relative_path": "05_training/rewards/final_reward_spec_step111.md",
            "checked_path": "/tmp/urbanbus_rl_project/05_training/rewards/final_reward_spec_step111.md",
            "exists": True,
            "expected_size_bytes": 10,
            "actual_size_bytes": 10,
            "expected_sha256": "a" * 64,
            "actual_sha256": "a" * 64,
            "integrity_status": "PASS",
            "reasons": [],
        }
    ]
    return {
        "artifact_version": "h200_transfer_package_integrity_verifier_step146_v1",
        "step": 146,
        "audit_status": "PASS",
        "integrity_status": "H200_TRANSFER_PACKAGE_INTEGRITY_VERIFIED_ACTUAL_STILL_LOCKED",
        "planned_run_count": 72,
        "conditions": ["A", "A90", "A80", "A70"],
        "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
        "seeds": [1, 2, 3],
        "verified_file_count": len(entries),
        "verified_entries": entries,
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


def test_missing_file_fails() -> None:
    payload = base_payload()
    payload["verified_entries"][0]["exists"] = False
    payload["verified_entries"][0]["integrity_status"] = "FAIL"
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("exists_false" in e or "integrity_status_not_pass" in e for e in result["errors"])


def test_sha_mismatch_fails() -> None:
    payload = base_payload()
    payload["verified_entries"][0]["actual_sha256"] = "b" * 64
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("sha256_mismatch" in e for e in result["errors"])


def test_size_mismatch_fails() -> None:
    payload = base_payload()
    payload["verified_entries"][0]["actual_size_bytes"] = 11
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("size_mismatch" in e for e in result["errors"])


def test_blocked_payload_can_pass_with_allow_blocked() -> None:
    payload = base_payload()
    payload["audit_status"] = "BLOCKED"
    payload["integrity_status"] = "H200_TRANSFER_PACKAGE_INTEGRITY_BLOCKED"
    result = validate_payload(payload, require_pass=False)
    assert result["validation_status"] == "PASS", result


def main() -> None:
    test_pass_payload()
    test_actual_execution_true_fails()
    test_missing_file_fails()
    test_sha_mismatch_fails()
    test_size_mismatch_fails()
    test_blocked_payload_can_pass_with_allow_blocked()
    print("[OK] Step 146 H200 transfer package integrity verifier self-test PASS")


if __name__ == "__main__":
    main()
