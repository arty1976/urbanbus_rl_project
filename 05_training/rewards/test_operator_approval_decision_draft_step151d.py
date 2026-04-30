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


def waiting_step151c_manifest() -> dict:
    return {
        "artifact_version": "h200_actual_preflight_rerun_result_intake_placeholder_step151c_v1",
        "step": "151-C",
        "intake_status": "WAITING_FOR_H200_EXPECTED_PREFLIGHT_RESULT_STILL_LOCKED",
        "decision": "INTAKE_PLACEHOLDER_ONLY_EXECUTION_STILL_LOCKED",
        "hard_failures": 0,
        "warnings": 0,
        "h200_expected_preflight_manifest_present": False,
        "h200_expected_preflight_passed": False,
        "operator_approval_recorded": False,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
    }


def validated_step151c_manifest() -> dict:
    p = waiting_step151c_manifest()
    p["intake_status"] = "H200_EXPECTED_PREFLIGHT_RESULT_RECEIVED_AND_VALIDATED_STILL_LOCKED"
    p["h200_expected_preflight_manifest_present"] = True
    p["h200_expected_preflight_passed"] = True
    return p


def bad_step151c_manifest() -> dict:
    p = waiting_step151c_manifest()
    p["intake_status"] = "BLOCKED"
    return p


def bad_unlocked_step151c_manifest() -> dict:
    p = waiting_step151c_manifest()
    p["actual_execution_allowed"] = True
    return p


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reward_dir = project_root / "05_training" / "rewards"
    gen = reward_dir / "operator_approval_decision_draft_step151d.py"
    val = reward_dir / "validate_operator_approval_decision_draft_step151d.py"

    selftest_root = project_root / "artifacts" / "rewards" / "operator_approval_decision_draft_step151d_selftest"
    input_root = selftest_root / "input"

    waiting = input_root / "step151c_waiting.json"
    validated = input_root / "step151c_validated.json"
    bad = input_root / "step151c_bad.json"
    bad_unlocked = input_root / "step151c_bad_unlocked.json"

    dump_json(waiting, waiting_step151c_manifest())
    dump_json(validated, validated_step151c_manifest())
    dump_json(bad, bad_step151c_manifest())
    dump_json(bad_unlocked, bad_unlocked_step151c_manifest())

    waiting_output = selftest_root / "waiting_output"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step151c-manifest",
            str(waiting),
            "--output-root",
            str(waiting_output),
        ],
        cwd=project_root,
        expect_ok=True,
    )

    waiting_manifest = waiting_output / "operator_approval_decision_draft_step151d_manifest.json"
    run([sys.executable, str(val), "--manifest", str(waiting_manifest)], cwd=project_root, expect_ok=True)

    validated_output = selftest_root / "validated_output"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step151c-manifest",
            str(validated),
            "--output-root",
            str(validated_output),
        ],
        cwd=project_root,
        expect_ok=True,
    )

    validated_manifest = validated_output / "operator_approval_decision_draft_step151d_manifest.json"
    run([sys.executable, str(val), "--manifest", str(validated_manifest)], cwd=project_root, expect_ok=True)

    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step151c-manifest",
            str(bad),
            "--output-root",
            str(selftest_root / "bad_output"),
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
            "--step151c-manifest",
            str(bad_unlocked),
            "--output-root",
            str(selftest_root / "bad_unlocked_output"),
        ],
        cwd=project_root,
        expect_ok=False,
    )

    print("[OK] Step 151-D operator approval decision draft self-test PASS")
    print(f"[OK] selftest_root: {selftest_root}")


if __name__ == "__main__":
    main()
