from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from reward_ablation_execution_environment_preflight_step131 import (
    FORBIDDEN_TRUE_GUARDS,
    run_preflight,
)
from validate_reward_ablation_execution_environment_preflight_step131 import validate_payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_fixture(root: Path) -> None:
    rewards = root / "05_training" / "rewards"
    rewards.mkdir(parents=True, exist_ok=True)

    for step in range(111, 131):
        (rewards / f"step{step}_marker.txt").write_text("ok", encoding="utf-8")

    guards = {key: False for key in FORBIDDEN_TRUE_GUARDS}
    write_json(rewards / "reward_pipeline_project_log_update_step130.json", guards)

    (root / "project_log.md").write_text("Step 130\n", encoding="utf-8")


def test_pass_fixture() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step131_pass_"))
    try:
        build_fixture(tmp)
        payload = run_preflight(tmp, tmp / "artifacts" / "rewards" / "step131")
        assert payload["audit_status"] == "PASS"
        assert payload["decision"]["actual_reward_ablation_execution_allowed"] is True
        result = validate_payload(payload, require_pass=True)
        assert result["validation_status"] == "PASS", result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_step_blocks() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step131_missing_"))
    try:
        build_fixture(tmp)
        (tmp / "05_training" / "rewards" / "step119_marker.txt").unlink()
        payload = run_preflight(tmp, tmp / "artifacts" / "rewards" / "step131")
        assert payload["audit_status"] == "BLOCKED"
        assert any("missing_step_markers" in x for x in payload["blocking_reasons"])
        result = validate_payload(payload, require_pass=False)
        assert result["validation_status"] == "PASS", result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_guard_violation_blocks() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="step131_guard_"))
    try:
        build_fixture(tmp)
        guards = {key: False for key in FORBIDDEN_TRUE_GUARDS}
        guards["actual_results"] = True
        write_json(
            tmp / "05_training" / "rewards" / "reward_pipeline_project_log_update_step130.json",
            guards,
        )
        payload = run_preflight(tmp, tmp / "artifacts" / "rewards" / "step131")
        assert payload["audit_status"] == "BLOCKED"
        assert any("forbidden_true_guard: actual_results" in x for x in payload["blocking_reasons"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    test_pass_fixture()
    test_missing_step_blocks()
    test_guard_violation_blocks()
    print("[OK] Step 131 reward ablation environment preflight self-test PASS")


if __name__ == "__main__":
    main()
