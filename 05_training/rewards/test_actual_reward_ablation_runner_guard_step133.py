from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from run_actual_reward_ablation_candidate import build_runner_guard_manifest
from validate_actual_reward_ablation_runner_guard_step133 import validate_payload


def test_dry_run_pass() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step133_pass_"))
    try:
        payload = build_runner_guard_manifest(
            project_root=tmp,
            reward_id="R0",
            seed=1,
            condition="A",
            mode="dry-run",
            output_root=tmp / "out",
            execute_actual=False,
            step132_manifest=None,
        )
        assert payload["audit_status"] == "PASS", payload
        assert payload["actual_executed"] is False
        assert payload["actual_results"] is False
        result = validate_payload(payload, require_pass=True)
        assert result["validation_status"] == "PASS", result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_actual_attempt_blocks() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step133_actual_block_"))
    try:
        payload = build_runner_guard_manifest(
            project_root=tmp,
            reward_id="R0",
            seed=1,
            condition="A",
            mode="actual",
            output_root=tmp / "out",
            execute_actual=False,
            step132_manifest=None,
        )
        assert payload["audit_status"] == "BLOCKED"
        assert "actual_execution_blocked_by_step133_guard" in payload["blocking_reasons"]
        result = validate_payload(payload, require_pass=False)
        assert result["validation_status"] == "PASS", result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_invalid_reward_blocks() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step133_invalid_reward_"))
    try:
        payload = build_runner_guard_manifest(
            project_root=tmp,
            reward_id="R9",
            seed=1,
            condition="A",
            mode="dry-run",
            output_root=tmp / "out",
            execute_actual=False,
            step132_manifest=None,
        )
        assert payload["audit_status"] == "BLOCKED"
        assert any("invalid_reward_id" in r for r in payload["blocking_reasons"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_validator_catches_tampered_actual_results() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step133_tamper_"))
    try:
        payload = build_runner_guard_manifest(
            project_root=tmp,
            reward_id="R0",
            seed=1,
            condition="A",
            mode="dry-run",
            output_root=tmp / "out",
            execute_actual=False,
            step132_manifest=None,
        )
        payload["actual_results"] = True
        result = validate_payload(payload, require_pass=True)
        assert result["validation_status"] == "FAIL"
        assert any("actual_results" in e for e in result["errors"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    test_dry_run_pass()
    test_actual_attempt_blocks()
    test_invalid_reward_blocks()
    test_validator_catches_tampered_actual_results()
    print("[OK] Step 133 actual reward ablation runner guard self-test PASS")


if __name__ == "__main__":
    main()
