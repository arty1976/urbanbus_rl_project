from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def unique_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    updater = project_root / "05_training" / "rewards" / "update_project_log_reward_pipeline_step130.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_pipeline_project_log_update_step130.py"
    status = project_root / "05_training" / "rewards" / "reward_pipeline_project_log_update_step130.json"

    suffix = unique_suffix()
    output_root = project_root / "artifacts" / "rewards" / f"reward_pipeline_project_log_update_step130_selftest_{suffix}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_pipeline_project_log_update_step130_validation_{suffix}"

    update_cmd = [
        sys.executable,
        str(updater),
        "--status",
        str(status),
        "--project-log",
        str(project_root / "project_log.md"),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(update_cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 130 project log updater failed")

    manifest_path = output_root / "reward_pipeline_project_log_update_step130_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("[FAIL] Step 130 manifest missing")

    val_cmd = [
        sys.executable,
        str(validator),
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(validation_root),
    ]
    val = subprocess.run(val_cmd, cwd=project_root, text=True, capture_output=True)
    print(val.stdout)
    if val.returncode != 0:
        print(val.stderr)
        raise SystemExit("[FAIL] Step 130 validator failed")

    report = load_json(validation_root / "reward_pipeline_project_log_update_step130_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_PROJECT_LOG_UPDATED_REWARD_PIPELINE_WAITING",
        "next_status": "READY_FOR_STEP131_ACTUAL_ABLATION_EXECUTION_ENV_PREFLIGHT_OR_PUSH_SYNC",
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    print("[OK] Step 130 reward pipeline project log update self-test PASS")
    print("[DONE] Step 130 reward pipeline project log update complete.")


if __name__ == "__main__":
    main()
