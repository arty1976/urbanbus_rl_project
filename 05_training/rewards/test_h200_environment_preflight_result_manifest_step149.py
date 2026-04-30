from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h200_environment_preflight_result_manifest_step149 as step149


def make_step148_manifest(root: Path) -> None:
    path = root / step149.STEP148_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_version": "h200_receive_side_preflight_gate_step148_v1",
        "step_id": 148,
        "gate_status": "READY_FOR_H200_RECEIVE_SIDE_PREFLIGHT_REVIEW",
        "hard_failures": [],
        "h200_project_root": "/workspace/urbanbus_rl_project",
        "required_source_file_count": 23,
        "execution_locks": {
            "actual_execution_allowed": False,
            "actual_execution_released": False,
            "train_allowed": False,
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_local_manifest_passes_without_h200_assertion() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_step148_manifest(root)
        payload = step149.build_manifest(
            project_root=root,
            output_root=root / "artifacts" / "step149",
            expect_h200=False,
            min_gpu_count=1,
        )

        assert payload["audit_status"] == "PASS"
        assert payload["environment_status"] == "LOCAL_ENVIRONMENT_PREFLIGHT_RECORDED_H200_NOT_ASSERTED_ACTUAL_STILL_LOCKED"
        assert payload["execution_locks"]["actual_execution_allowed"] is False
        assert Path(payload["manifest_path"]).exists()


def test_missing_step148_blocks() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        payload = step149.build_manifest(
            project_root=root,
            output_root=root / "artifacts" / "step149",
            expect_h200=False,
            min_gpu_count=1,
        )

        assert payload["audit_status"] == "BLOCKED"
        assert any("step148_manifest_missing" in x for x in payload["hard_failures"])


def test_bad_step148_lock_blocks() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_step148_manifest(root)
        path = root / step149.STEP148_MANIFEST
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["execution_locks"]["train_allowed"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")

        out = step149.build_manifest(
            project_root=root,
            output_root=root / "artifacts" / "step149",
            expect_h200=False,
            min_gpu_count=1,
        )

        assert out["audit_status"] == "BLOCKED"
        assert any("step148_lock_not_false" in x for x in out["hard_failures"])


def main() -> None:
    test_local_manifest_passes_without_h200_assertion()
    test_missing_step148_blocks()
    test_bad_step148_lock_blocks()
    print("[OK] Step 149 H200 environment preflight result manifest self-test PASS")


if __name__ == "__main__":
    main()
