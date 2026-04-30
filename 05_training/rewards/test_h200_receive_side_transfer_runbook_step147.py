from __future__ import annotations

from validate_h200_receive_side_transfer_runbook_step147 import validate_payload


def base_payload() -> dict:
    return {
        "artifact_version": "h200_receive_side_transfer_runbook_step147_v1",
        "step": 147,
        "audit_status": "PASS",
        "receive_plan_status": "H200_RECEIVE_SIDE_TRANSFER_RUNBOOK_READY_ACTUAL_STILL_LOCKED",
        "pinned_git_commit": "a" * 40,
        "planned_run_count": 72,
        "conditions": ["A", "A90", "A80", "A70"],
        "reward_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
        "seeds": [1, 2, 3],
        "transfer_file_count": 10,
        "receive_command_plan": {
            "local_before_transfer": ["git status --short"],
            "h200_git_receive": ["git pull --ff-only origin main"],
            "h200_metadata_receive_note": ["# copy metadata"],
            "h200_integrity_verify": ["python 05_training/rewards/h200_transfer_package_integrity_verifier_step146.py --project-root . --transfer-root ."],
            "blocked_execution_reminder": ["# do not run actual execution"],
        },
        "verification_metadata_files": [
            {"relative_path": "artifacts/rewards/h200_transfer_package_export_manifest_step145/h200_transfer_package_export_manifest_step145.json"},
            {"relative_path": "artifacts/rewards/h200_transfer_package_export_manifest_step145/h200_transfer_package_filelist_step145.txt"},
        ],
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
    assert validate_payload(base_payload(), True)["validation_status"] == "PASS"


def test_actual_execution_true_fails() -> None:
    p = base_payload(); p["actual_execution_allowed"] = True
    r = validate_payload(p, True)
    assert r["validation_status"] == "FAIL"
    assert any("actual_execution_allowed" in e for e in r["errors"])


def test_missing_command_plan_fails() -> None:
    p = base_payload(); p["receive_command_plan"].pop("h200_integrity_verify")
    r = validate_payload(p, True)
    assert r["validation_status"] == "FAIL"
    assert any("command_plan_missing_or_empty" in e for e in r["errors"])


def test_wrong_transfer_count_fails() -> None:
    p = base_payload(); p["transfer_file_count"] = 9
    r = validate_payload(p, True)
    assert r["validation_status"] == "FAIL"
    assert any("transfer_file_count" in e for e in r["errors"])


def test_blocked_payload_can_pass_with_allow_blocked() -> None:
    p = base_payload(); p["audit_status"] = "BLOCKED"; p["receive_plan_status"] = "H200_RECEIVE_SIDE_TRANSFER_RUNBOOK_BLOCKED"
    assert validate_payload(p, False)["validation_status"] == "PASS"


def main() -> None:
    test_pass_payload(); test_actual_execution_true_fails(); test_missing_command_plan_fails(); test_wrong_transfer_count_fails(); test_blocked_payload_can_pass_with_allow_blocked()
    print("[OK] Step 147 H200 receive-side transfer runbook self-test PASS")


if __name__ == "__main__":
    main()
