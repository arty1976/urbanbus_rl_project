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


def good_step151b_manifest() -> dict:
    return {
        "artifact_version": "h200_expected_preflight_rerun_checklist_step151b_v1",
        "step": "151-B",
        "checklist_status": "H200_EXPECTED_PREFLIGHT_RERUN_CHECKLIST_READY_STILL_LOCKED",
        "decision": "CHECKLIST_ONLY_H200_RERUN_NOT_EXECUTED",
        "hard_failures": 0,
        "warnings": 0,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "h200_expected_preflight_rerun_executed": False,
        "h200_expected_preflight_passed": False,
    }


def good_h200_step149_manifest() -> dict:
    return {
        "artifact_version": "h200_environment_preflight_result_manifest_step149_v1",
        "step": 149,
        "audit_status": "PASS",
        "environment_status": "H200_ENVIRONMENT_PREFLIGHT_PASS_ACTUAL_STILL_LOCKED",
        "expect_h200": True,
        "torch_import_ok": True,
        "cuda_available": True,
        "gpu_count": 8,
        "hard_failures": 0,
        "warnings": 0,
        "actual_execution_allowed": False,
        "train_allowed": False,
    }


def bad_h200_step149_manifest() -> dict:
    p = good_h200_step149_manifest()
    p["cuda_available"] = False
    p["gpu_count"] = 0
    return p


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reward_dir = project_root / "05_training" / "rewards"
    gen = reward_dir / "h200_actual_preflight_rerun_result_intake_placeholder_step151c.py"
    val = reward_dir / "validate_h200_actual_preflight_rerun_result_intake_placeholder_step151c.py"

    selftest_root = project_root / "artifacts" / "rewards" / "h200_actual_preflight_rerun_result_intake_placeholder_step151c_selftest"
    input_root = selftest_root / "input"

    step151b_good = input_root / "step151b_good.json"
    step151b_bad = input_root / "step151b_bad.json"
    h200_good = input_root / "h200_step149_good.json"
    h200_bad = input_root / "h200_step149_bad.json"
    h200_missing = input_root / "h200_step149_missing.json"

    dump_json(step151b_good, good_step151b_manifest())

    bad_151b = good_step151b_manifest()
    bad_151b["checklist_status"] = "BLOCKED"
    dump_json(step151b_bad, bad_151b)

    dump_json(h200_good, good_h200_step149_manifest())
    dump_json(h200_bad, bad_h200_step149_manifest())

    waiting_output = selftest_root / "waiting_output"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step151b-manifest",
            str(step151b_good),
            "--h200-step149-manifest",
            str(h200_missing),
            "--output-root",
            str(waiting_output),
        ],
        cwd=project_root,
        expect_ok=True,
    )

    waiting_manifest = waiting_output / "h200_actual_preflight_rerun_result_intake_placeholder_step151c_manifest.json"
    run([sys.executable, str(val), "--manifest", str(waiting_manifest)], cwd=project_root, expect_ok=True)

    received_output = selftest_root / "received_output"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step151b-manifest",
            str(step151b_good),
            "--h200-step149-manifest",
            str(h200_good),
            "--output-root",
            str(received_output),
        ],
        cwd=project_root,
        expect_ok=True,
    )

    received_manifest = received_output / "h200_actual_preflight_rerun_result_intake_placeholder_step151c_manifest.json"
    run([sys.executable, str(val), "--manifest", str(received_manifest)], cwd=project_root, expect_ok=True)

    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step151b-manifest",
            str(step151b_good),
            "--h200-step149-manifest",
            str(h200_bad),
            "--output-root",
            str(selftest_root / "bad_h200_output"),
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
            "--step151b-manifest",
            str(step151b_bad),
            "--h200-step149-manifest",
            str(h200_missing),
            "--output-root",
            str(selftest_root / "bad_151b_output"),
        ],
        cwd=project_root,
        expect_ok=False,
    )

    print("[OK] Step 151-C H200 actual preflight rerun result intake placeholder self-test PASS")
    print(f"[OK] selftest_root: {selftest_root}")


if __name__ == "__main__":
    main()
