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
    validator = project_root / "05_training" / "rewards" / "validate_actual_reward_ablation_runbook_step129.py"
    runbook = project_root / "05_training" / "rewards" / "actual_reward_ablation_runbook_step129.json"

    suffix = unique_suffix()
    validation_root = project_root / "artifacts" / "rewards" / f"actual_reward_ablation_runbook_step129_validation_{suffix}"

    cmd = [
        sys.executable,
        str(validator),
        "--runbook",
        str(runbook),
        "--output-root",
        str(validation_root),
    ]
    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 129 validator failed")

    report = load_json(validation_root / "actual_reward_ablation_runbook_step129_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_ACTUAL_ABLATION_RUNBOOK_READY_NOT_EXECUTED",
        "next_status": "READY_FOR_STEP130_PROJECT_LOG_UPDATE_OR_ACTUAL_ABLATION_EXECUTION_ENV_PREFLIGHT",
        "expected_actual_runs": 72,
        "actual_execution_started": False,
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

    # Negative test: actual_execution_started=true must fail.
    bad = load_json(runbook)
    bad["guard_flags"]["actual_execution_started"] = True
    bad_path = validation_root / "bad_actual_started_runbook_step129.json"
    dump_json(bad_path, bad)

    bad_root = validation_root / "bad_case"
    bad_cmd = [
        sys.executable,
        str(validator),
        "--runbook",
        str(bad_path),
        "--output-root",
        str(bad_root),
    ]
    bad_result = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] bad actual_execution_started runbook unexpectedly passed")

    bad_report = load_json(bad_root / "actual_reward_ablation_runbook_step129_validation_report.json")
    failed_ids = {
        c["check_id"]
        for c in bad_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "guard_actual_execution_started" not in failed_ids:
        raise SystemExit("[FAIL] actual_execution_started guard did not fail")

    print("[OK] Step 129 actual reward ablation runbook self-test PASS")
    print("[DONE] Step 129 actual reward ablation runbook complete.")


if __name__ == "__main__":
    main()
