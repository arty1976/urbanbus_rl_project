from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(cmd, cwd: Path, expect_ok: bool = True) -> subprocess.CompletedProcess:
    cp = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )

    if expect_ok and cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command failed: {cmd}")

    if (not expect_ok) and cp.returncode == 0:
        print(cp.stdout)
        print(cp.stderr)
        raise RuntimeError(f"command unexpectedly passed: {cmd}")

    return cp


def good_step150_manifest() -> dict:
    return {
        "artifact_version": "actual_reward_ablation_operator_release_checklist_step150_v1",
        "step": 150,
        "checklist_status": "READY_FOR_STEP151_EXPLICIT_OPERATOR_RELEASE_MANIFEST_DRAFT",
        "decision": "CHECKLIST_ONLY_NOT_RELEASED",
        "hard_failures": 0,
        "warnings": 0,
        "actual_execution_allowed": False,
        "train_allowed": False,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reward_dir = project_root / "05_training" / "rewards"
    gen = reward_dir / "explicit_operator_release_manifest_draft_step151a.py"
    val = reward_dir / "validate_explicit_operator_release_manifest_draft_step151a.py"

    selftest_root = project_root / "artifacts" / "rewards" / "explicit_operator_release_manifest_draft_step151a_selftest"
    input_root = selftest_root / "input"

    good = input_root / "step150_good.json"
    bad_status = input_root / "step150_bad_status.json"
    bad_release = input_root / "step150_bad_actual_allowed.json"

    dump_json(good, good_step150_manifest())

    bad_status_payload = good_step150_manifest()
    bad_status_payload["checklist_status"] = "BLOCKED"
    dump_json(bad_status, bad_status_payload)

    bad_release_payload = good_step150_manifest()
    bad_release_payload["actual_execution_allowed"] = True
    dump_json(bad_release, bad_release_payload)

    positive_output = selftest_root / "positive_output"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step150-manifest",
            str(good),
            "--output-root",
            str(positive_output),
        ],
        cwd=project_root,
        expect_ok=True,
    )

    positive_manifest = positive_output / "explicit_operator_release_manifest_draft_step151a_manifest.json"

    run(
        [
            sys.executable,
            str(val),
            "--manifest",
            str(positive_manifest),
        ],
        cwd=project_root,
        expect_ok=True,
    )

    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step150-manifest",
            str(bad_status),
            "--output-root",
            str(selftest_root / "negative_status_output"),
        ],
        cwd=project_root,
        expect_ok=False,
    )

    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step150-manifest",
            str(bad_release),
            "--output-root",
            str(selftest_root / "negative_release_output"),
        ],
        cwd=project_root,
        expect_ok=False,
    )

    print("[OK] Step 151-A explicit operator release manifest draft self-test PASS")
    print(f"[OK] selftest_root: {selftest_root}")


if __name__ == "__main__":
    main()
