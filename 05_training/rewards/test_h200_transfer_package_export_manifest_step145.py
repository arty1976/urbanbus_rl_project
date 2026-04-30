from __future__ import annotations

from validate_h200_transfer_package_export_manifest_step145 import validate_payload


def base_payload() -> dict:
    entries = [
        {
            "relative_path": "05_training/rewards/final_reward_spec_step111.md",
            "exists": True,
            "size_bytes": 10,
            "sha256": "a" * 64,
        },
        {
            "relative_path": "05_training/rewards/a_family_72run_release_matrix_extension_draft_step143.py",
            "exists": True,
            "size_bytes": 10,
            "sha256": "b" * 64,
        },
    ]
    return {
        "artifact_version": "h200_transfer_package_export_manifest_step145_v1",
        "step": 145,
        "audit_status": "PASS",
        "export_status": "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_READY_ACTUAL_STILL_LOCKED",
        "planned_run_count": 72,
        "conditions": ["A", "A90", "A80", "A70"],
        "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
        "seeds": [1, 2, 3],
        "transfer_file_count": len(entries),
        "transfer_entries": entries,
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


def test_forbidden_artifact_transfer_fails() -> None:
    payload = base_payload()
    payload["transfer_entries"][0]["relative_path"] = "artifacts/rewards/foo.json"
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("forbidden_transfer_prefix" in e for e in result["errors"])


def test_latest_pointer_transfer_fails() -> None:
    payload = base_payload()
    payload["transfer_entries"][0]["relative_path"] = "05_training/rewards/foo.latest.json"
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("forbidden_transfer_suffix" in e for e in result["errors"])


def test_bad_sha_fails() -> None:
    payload = base_payload()
    payload["transfer_entries"][0]["sha256"] = "not-a-sha"
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("invalid_sha256" in e for e in result["errors"])


def test_blocked_payload_can_pass_with_allow_blocked() -> None:
    payload = base_payload()
    payload["audit_status"] = "BLOCKED"
    payload["export_status"] = "H200_TRANSFER_PACKAGE_EXPORT_MANIFEST_BLOCKED"
    result = validate_payload(payload, require_pass=False)
    assert result["validation_status"] == "PASS", result


def main() -> None:
    test_pass_payload()
    test_actual_execution_true_fails()
    test_forbidden_artifact_transfer_fails()
    test_latest_pointer_transfer_fails()
    test_bad_sha_fails()
    test_blocked_payload_can_pass_with_allow_blocked()
    print("[OK] Step 145 H200 transfer package export manifest self-test PASS")


if __name__ == "__main__":
    main()
