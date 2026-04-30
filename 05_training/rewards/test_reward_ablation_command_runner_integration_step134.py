from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from reward_ablation_command_runner_integration_step134 import (
    extract_planned_commands_from_step132,
    reconstruct_expected_commands,
    validate_command_matrix,
)
from validate_reward_ablation_command_runner_integration_step134 import validate_payload


def test_extract_from_dict_commands() -> None:
    payload = {
        "step": 132,
        "planned_commands": [
            {"condition": "A", "reward_id": "R0", "seed": 1},
            {"condition": "A", "reward_id": "R0", "seed": 2},
        ],
    }
    commands, warnings = extract_planned_commands_from_step132(payload)
    assert len(commands) == 2
    assert commands[0]["reward_id"] == "R0"
    assert warnings == []


def test_reconstructed_matrix_is_complete() -> None:
    commands = reconstruct_expected_commands()
    result = validate_command_matrix(commands)
    assert result["errors"] == [], result
    assert result["actual_count"] == 18


def test_missing_matrix_entry_fails() -> None:
    commands = reconstruct_expected_commands()
    commands = commands[:-1]
    result = validate_command_matrix(commands)
    assert result["errors"], result
    assert any("missing_expected_commands" in e for e in result["errors"])


def test_validator_catches_actual_results() -> None:
    payload = {
        "artifact_version": "reward_ablation_command_runner_integration_step134_v1",
        "step": 134,
        "audit_status": "PASS",
        "summary": {
            "planned_command_count": 18,
            "step133_invocation_count": 18,
            "step133_guard_pass_count": 18,
            "step133_guard_fail_count": 0,
        },
        "planned_commands": reconstruct_expected_commands(),
        "actual_executed": False,
        "actual_results": True,
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
    result = validate_payload(payload, require_pass=True)
    assert result["validation_status"] == "FAIL"
    assert any("actual_results" in e for e in result["errors"])


def main() -> None:
    test_extract_from_dict_commands()
    test_reconstructed_matrix_is_complete()
    test_missing_matrix_entry_fails()
    test_validator_catches_actual_results()
    print("[OK] Step 134 command-runner integration self-test PASS")


if __name__ == "__main__":
    main()
