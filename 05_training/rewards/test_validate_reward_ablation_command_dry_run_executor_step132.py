from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from reward_ablation_command_dry_run_executor_step132 import (
    DEFAULT_CANDIDATES,
    DEFAULT_CONDITIONS,
    DEFAULT_SEEDS,
    FORBIDDEN_TRUE_GUARDS,
    run_step132,
)
from validate_reward_ablation_command_dry_run_executor_step132 import validate_payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_step131_fixture(root: Path, pass_status: bool = True) -> Path:
    manifest = (
        root
        / "artifacts"
        / "rewards"
        / "reward_ablation_execution_environment_preflight_step131"
        / "reward_ablation_execution_environment_preflight_step131_manifest.json"
    )
    guards = {key: False for key in FORBIDDEN_TRUE_GUARDS}
    payload = {
        "artifact_version": "reward_ablation_execution_environment_preflight_step131_v1",
        "step": 131,
        "audit_status": "PASS" if pass_status else "BLOCKED",
        "decision": {
            "actual_reward_ablation_execution_allowed": bool(pass_status),
        },
        "non_claim_guards": guards,
    }
    write_json(manifest, payload)
    pointer = root / "05_training" / "rewards" / "reward_ablation_execution_environment_preflight_step131.latest.json"
    write_json(pointer, {"manifest_path": str(manifest), "audit_status": payload["audit_status"]})
    return manifest


def test_pass_fixture() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step132_pass_"))
    try:
        build_step131_fixture(tmp, pass_status=True)
        payload = run_step132(
            project_root=tmp,
            output_root=tmp / "artifacts" / "rewards" / "step132",
            step131_manifest=None,
            candidates=["R0", "R1"],
            conditions=["A"],
            seeds=[1, 2],
            runner_path="05_training/rewards/run_actual_reward_ablation_candidate.py",
            require_runner_exists=False,
        )
        assert payload["audit_status"] == "PASS", payload
        assert payload["planned_command_count"] == 4, payload
        assert payload["decision"]["actual_execution_performed"] is False
        result = validate_payload(payload, require_pass=True)
        assert result["validation_status"] == "PASS", result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_step131_blocked_blocks_step132() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step132_blocked_"))
    try:
        build_step131_fixture(tmp, pass_status=False)
        payload = run_step132(
            project_root=tmp,
            output_root=tmp / "artifacts" / "rewards" / "step132",
            step131_manifest=None,
            candidates=["R0"],
            conditions=["A"],
            seeds=[1],
            runner_path="05_training/rewards/run_actual_reward_ablation_candidate.py",
            require_runner_exists=False,
        )
        assert payload["audit_status"] == "BLOCKED", payload
        assert any("step131" in x for x in payload["blocking_reasons"]), payload
        result = validate_payload(payload, require_pass=False)
        assert result["validation_status"] == "PASS", result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_runner_required_blocks_when_missing() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step132_runner_missing_"))
    try:
        build_step131_fixture(tmp, pass_status=True)
        payload = run_step132(
            project_root=tmp,
            output_root=tmp / "artifacts" / "rewards" / "step132",
            step131_manifest=None,
            candidates=["R0"],
            conditions=["A"],
            seeds=[1],
            runner_path="05_training/rewards/run_actual_reward_ablation_candidate.py",
            require_runner_exists=True,
        )
        assert payload["audit_status"] == "BLOCKED", payload
        assert any("runner_path_missing" in x for x in payload["blocking_reasons"]), payload
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    test_pass_fixture()
    test_step131_blocked_blocks_step132()
    test_runner_required_blocks_when_missing()
    print("[OK] Step 132 reward ablation command dry-run executor self-test PASS")


if __name__ == "__main__":
    main()
