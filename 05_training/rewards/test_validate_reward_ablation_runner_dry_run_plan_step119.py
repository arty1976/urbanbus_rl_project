from __future__ import annotations

import csv
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


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    planner = project_root / "05_training" / "rewards" / "reward_ablation_runner_dry_run_plan_step119.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_runner_dry_run_plan_step119.py"
    spec = project_root / "05_training" / "rewards" / "reward_ablation_runner_dry_run_plan_step119.json"

    unique = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"
    output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_runner_dry_run_plan_step119_selftest_{unique}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_runner_dry_run_plan_step119_validation_{unique}"

    run = subprocess.run(
        [sys.executable, str(planner), "--plan", str(spec), "--output-root", str(output_root)],
        cwd=project_root,
        text=True,
        capture_output=True,
    )
    print(run.stdout)
    if run.returncode != 0:
        print(run.stderr)
        raise SystemExit("[FAIL] Step 119 planner failed")

    manifest = output_root / "reward_ablation_runner_dry_run_manifest.json"
    if not manifest.exists():
        raise SystemExit("[FAIL] Step 119 manifest missing")

    val = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--plan-spec",
            str(spec),
            "--manifest",
            str(manifest),
            "--output-root",
            str(validation_root),
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
    )
    print(val.stdout)
    if val.returncode != 0:
        print(val.stderr)
        raise SystemExit("[FAIL] Step 119 validator failed")

    report = load_json(validation_root / "reward_ablation_runner_dry_run_plan_step119_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_REWARD_ABLATION_RUNNER_DRY_RUN_PLAN_NOT_EXECUTED",
        "next_status": "READY_FOR_STEP120_REWARD_ABLATION_EXECUTION_PREFLIGHT",
        "planned_run_count": 72,
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={report.get(key)!r}")

    # Negative test: corrupt one command by adding --execute. It must fail.
    bad_root = project_root / "artifacts" / "rewards" / f"reward_ablation_runner_dry_run_plan_step119_badcase_{unique}"
    bad_root.mkdir(parents=True, exist_ok=True)
    manifest_payload = load_json(manifest)
    csv_src = Path(manifest_payload["output_files"]["plan_csv"])
    rows = []
    with open(csv_src, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["planned_command_text"] = rows[0]["planned_command_text"] + " --execute"
    bad_csv = bad_root / "reward_ablation_runner_dry_run_plan.csv"
    with open(bad_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bad_manifest = dict(manifest_payload)
    bad_manifest["output_root"] = str(bad_root)
    bad_manifest["output_files"] = dict(manifest_payload["output_files"])
    bad_manifest["output_files"]["plan_csv"] = str(bad_csv)
    bad_manifest_path = bad_root / "reward_ablation_runner_dry_run_manifest.json"
    dump_json(bad_manifest_path, bad_manifest)

    bad_validation = project_root / "artifacts" / "rewards" / f"reward_ablation_runner_dry_run_plan_step119_bad_validation_{unique}"
    bad = subprocess.run(
        [
            sys.executable,
            str(validator),
            "--plan-spec",
            str(spec),
            "--manifest",
            str(bad_manifest_path),
            "--output-root",
            str(bad_validation),
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
    )
    if bad.returncode == 0:
        print(bad.stdout)
        raise SystemExit("[FAIL] corrupted dry-run command unexpectedly passed")

    bad_report = load_json(bad_validation / "reward_ablation_runner_dry_run_plan_step119_validation_report.json")
    failed_ids = {c["check_id"] for c in bad_report.get("checks", []) if not c.get("passed", False)}
    if "row_000_forbid___execute" not in failed_ids:
        raise SystemExit("[FAIL] forbidden --execute guard did not fail")

    print("[OK] Step 119 reward ablation runner dry-run plan self-test PASS")
    print("[DONE] Step 119 reward ablation runner dry-run plan complete.")


if __name__ == "__main__":
    main()

