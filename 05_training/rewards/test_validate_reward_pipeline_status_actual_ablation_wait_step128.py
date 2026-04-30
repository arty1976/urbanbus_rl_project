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


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def unique_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    validator = project_root / "05_training" / "rewards" / "validate_reward_pipeline_status_actual_ablation_wait_step128.py"
    status = project_root / "05_training" / "rewards" / "reward_pipeline_status_actual_ablation_wait_step128.json"

    suffix = unique_suffix()
    validation_root = project_root / "artifacts" / "rewards" / f"reward_pipeline_status_actual_ablation_wait_step128_validation_{suffix}"

    cmd = [
        sys.executable,
        str(validator),
        "--status",
        str(status),
        "--output-root",
        str(validation_root),
    ]
    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 128 validator failed")

    report = load_json(validation_root / "reward_pipeline_status_actual_ablation_wait_step128_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_REWARD_PIPELINE_DOCUMENTED_WAITING_FOR_ACTUAL_ABLATION_DATA",
        "next_status": "WAIT_FOR_ACTUAL_ABLATION_DATA_OR_PREPARE_ACTUAL_ABLATION_RUNBOOK",
        "covered_step_count": 17,
        "expected_actual_ablation_rows": 72,
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

    # Negative test: training allowed must fail.
    bad = load_json(status)
    bad["guard_flags"]["train_with_this_reward_allowed"] = True
    bad_path = validation_root / "bad_train_allowed_status_step128.json"
    dump_json(bad_path, bad)

    bad_root = validation_root / "bad_case"
    bad_cmd = [
        sys.executable,
        str(validator),
        "--status",
        str(bad_path),
        "--output-root",
        str(bad_root),
    ]
    bad_result = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] bad train_allowed status unexpectedly passed")

    bad_report = load_json(bad_root / "reward_pipeline_status_actual_ablation_wait_step128_validation_report.json")
    failed_ids = {
        c["check_id"]
        for c in bad_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "guard_train_with_this_reward_allowed" not in failed_ids:
        raise SystemExit("[FAIL] train_with_this_reward_allowed guard did not fail")

    print("[OK] Step 128 reward pipeline status self-test PASS")
    print("[DONE] Step 128 reward pipeline status actual ablation wait complete.")


if __name__ == "__main__":
    main()
