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



def good_step151a_manifest() -> dict:
    return {
        "artifact_version": "explicit_operator_release_manifest_draft_step151a_v1",
        "step": "151-A",
        "release_manifest_status": "RELEASE_MANIFEST_DRAFT_STILL_LOCKED",
        "decision": "DRAFT_ONLY_NOT_RELEASED",
        "hard_failures": 0,
        "warnings": 0,
        "actual_execution_allowed": False,
        "actual_execution_released": False,
        "train_allowed": False,
        "h200_expected_preflight_required": True,
        "h200_expected_preflight_required_args": "--expect-h200 --min-gpu-count 1",
    }



def good_step149_local_manifest() -> dict:
    return {
        "artifact_version": "h200_environment_preflight_result_manifest_step149_v1",
        "step": 149,
        "audit_status": "PASS",
        "environment_status": "LOCAL_ENVIRONMENT_PREFLIGHT_RECORDED_H200_NOT_ASSERTED_ACTUAL_STILL_LOCKED",
        "expect_h200": False,
        "torch_import_ok": True,
        "cuda_available": False,
        "gpu_count": 0,
        "hard_failures": 0,
        "warnings": 3,
        "actual_execution_allowed": False,
        "train_allowed": False,
    }



def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    reward_dir = project_root / "05_training" / "rewards"
    gen = reward_dir / "h200_expected_preflight_rerun_checklist_step151b.py"
    val = reward_dir / "validate_h200_expected_preflight_rerun_checklist_step151b.py"

    selftest_root = project_root / "artifacts" / "rewards" / "h200_expected_preflight_rerun_checklist_step151b_selftest"
    input_root = selftest_root / "input"

    good151a = input_root / "step151a_good.json"
    good149 = input_root / "step149_local.json"
    bad_status = input_root / "step151a_bad_status.json"
    bad_release = input_root / "step151a_bad_actual_allowed.json"
    bad_h200_args = input_root / "step151a_bad_h200_args.json"

    dump_json(good151a, good_step151a_manifest())
    dump_json(good149, good_step149_local_manifest())

    bad_status_payload = good_step151a_manifest()
    bad_status_payload["release_manifest_status"] = "BLOCKED"
    dump_json(bad_status, bad_status_payload)

    bad_release_payload = good_step151a_manifest()
    bad_release_payload["actual_execution_allowed"] = True
    dump_json(bad_release, bad_release_payload)

    bad_h200_args_payload = good_step151a_manifest()
    bad_h200_args_payload["h200_expected_preflight_required_args"] = "--expect-h200"
    dump_json(bad_h200_args, bad_h200_args_payload)

    positive_output = selftest_root / "positive_output"
    run(
        [
            sys.executable,
            str(gen),
            "--project-root",
            str(project_root),
            "--step151a-manifest",
            str(good151a),
            "--step149-local-manifest",
            str(good149),
            "--output-root",
            str(positive_output),
        ],
        cwd=project_root,
        expect_ok=True,
    )

    positive_manifest = positive_output / "h200_expected_preflight_rerun_checklist_step151b_manifest.json"

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
            "--step151a-manifest",
            str(bad_status),
            "--step149-local-manifest",
            str(good149),
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
            "--step151a-manifest",
            str(bad_release),
            "--step149-local-manifest",
            str(good149),
            "--output-root",
            str(selftest_root / "negative_release_output"),
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
            "--step151a-manifest",
            str(bad_h200_args),
            "--step149-local-manifest",
            str(good149),
            "--output-root",
            str(selftest_root / "negative_h200_args_output"),
        ],
        cwd=project_root,
        expect_ok=False,
    )

    print("[OK] Step 151-B H200 expected preflight rerun checklist self-test PASS")
    print(f"[OK] selftest_root: {selftest_root}")



if __name__ == "__main__":
    main()
