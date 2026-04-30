from __future__ import annotations

import json
import tempfile
from pathlib import Path

import h200_receive_side_preflight_gate_step148 as step148


def make_required_files(root: Path) -> None:
    for rel in step148.REQUIRED_SOURCE_FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"dummy file for {rel}\n", encoding="utf-8")


def test_ready_manifest_with_all_required_files() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_required_files(root)
        manifest = step148.build_manifest(
            project_root=root,
            output_root=root / "artifacts" / "step148",
            h200_project_root="/workspace/urbanbus_rl_project",
        )

        assert manifest["gate_status"] == "READY_FOR_H200_RECEIVE_SIDE_PREFLIGHT_REVIEW"
        assert manifest["hard_failures"] == []
        assert manifest["execution_locks"]["actual_execution_allowed"] is False
        assert manifest["execution_locks"]["train_allowed"] is False
        assert manifest["execution_locks"]["paper_level_claim_allowed"] is False
        assert manifest["required_source_file_count"] == len(step148.REQUIRED_SOURCE_FILES)

        manifest_path = Path(manifest["manifest_path"])
        assert manifest_path.exists()
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert saved["step_id"] == 148


def test_missing_required_file_blocks_gate() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_required_files(root)
        missing_path = root / step148.REQUIRED_SOURCE_FILES[0]
        missing_path.unlink()

        manifest = step148.build_manifest(
            project_root=root,
            output_root=root / "artifacts" / "step148",
            h200_project_root="/workspace/urbanbus_rl_project",
        )

        assert manifest["gate_status"] == "BLOCKED"
        assert manifest["missing_required_source_files"]
        assert any("missing_required_source_files" in x for x in manifest["hard_failures"])


def test_windows_style_h200_path_blocks_gate() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        make_required_files(root)

        manifest = step148.build_manifest(
            project_root=root,
            output_root=root / "artifacts" / "step148",
            h200_project_root="C:\\workspace\\urbanbus_rl_project",
        )

        assert manifest["gate_status"] == "BLOCKED"
        assert any("linux_style" in x for x in manifest["hard_failures"])


def main() -> None:
    test_ready_manifest_with_all_required_files()
    test_missing_required_file_blocks_gate()
    test_windows_style_h200_path_blocks_gate()
    print("[OK] Step 148 H200 receive-side preflight gate self-test PASS")


if __name__ == "__main__":
    main()
