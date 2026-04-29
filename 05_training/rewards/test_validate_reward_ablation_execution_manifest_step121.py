from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def write_mock_step119_plan(root: Path) -> Path:
    candidates = ["R0", "R1", "R2", "R3", "R4", "R5"]
    conditions = ["A", "A90", "A80", "A70"]
    seeds = [1, 2, 3]

    rows = []
    for candidate in candidates:
        for condition in conditions:
            for seed in seeds:
                run_id = f"{candidate}_{condition}_seed_{seed:03d}"
                rows.append({
                    "run_id": run_id,
                    "candidate_id": candidate,
                    "condition_id": condition,
                    "seed": seed,
                    "planned_output_root": str(root / "planned_outputs" / run_id),
                    "planned_command_text": (
                        f"python 05_training/rewards/reward_ablation_result_writer_guard_step118.py "
                        f"--candidate-id {candidate} --condition-id {condition} --seed {seed} "
                        f"--output-root artifacts/rewards/ablation/{run_id} --dry-run-plan"
                    ),
                    "execute_allowed": False,
                    "actual_training_allowed": False,
                    "train_with_this_reward_allowed": False,
                    "actual_results": False,
                    "winner_selected": False,
                })

    csv_path = root / "reward_ablation_runner_dry_run_plan.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = root / "reward_ablation_runner_dry_run_manifest.json"
    manifest = {
        "artifact_version": "reward_ablation_runner_dry_run_plan_step119_v1",
        "planner_status": "DRY_RUN_ROWS_WRITTEN_NOT_EXECUTED",
        "planned_runs": len(rows),
        "execute_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "output_files": {
            "dry_run_plan_csv": str(csv_path)
        },
    }
    dump_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    tag = utc_tag()

    materializer = project_root / "05_training" / "rewards" / "reward_ablation_execution_manifest_step121.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_execution_manifest_step121.py"

    source_root = project_root / "artifacts" / "rewards" / f"step121_mock_step119_source_{tag}"
    output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_manifest_step121_selftest_{tag}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_manifest_step121_validation_{tag}"

    dry_manifest = write_mock_step119_plan(source_root)

    cmd = [
        sys.executable,
        str(materializer),
        "--dry-run-manifest",
        str(dry_manifest),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 121 materializer failed")

    manifest_path = output_root / "reward_ablation_execution_manifest_step121_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("[FAIL] Step 121 manifest missing")

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
        raise SystemExit("[FAIL] Step 121 validator failed")

    report = load_json(validation_root / "reward_ablation_execution_manifest_step121_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_REWARD_ABLATION_EXECUTION_MANIFEST_NOT_EXECUTABLE",
        "next_status": "READY_FOR_STEP122_REWARD_ABLATION_SANDBOX_NOOP_RUNNER_GUARD",
        "row_count": 72,
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Bad case: add --execute to one command and ensure validation fails.
    bad_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_manifest_step121_bad_case_{tag}"
    shutil.copytree(output_root, bad_root)
    bad_csv = bad_root / "reward_ablation_execution_manifest_step121.csv"

    with open(bad_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["planned_command_text"] = rows[0]["planned_command_text"] + " --execute"
    with open(bad_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bad_manifest_path = bad_root / "reward_ablation_execution_manifest_step121_manifest.json"
    bad_manifest = load_json(bad_manifest_path)
    bad_manifest["output_files"]["execution_manifest_csv"] = str(bad_csv)
    dump_json(bad_manifest_path, bad_manifest)

    bad_val_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_manifest_step121_bad_validation_{tag}"
    bad_val_cmd = [
        sys.executable,
        str(validator),
        "--manifest",
        str(bad_manifest_path),
        "--output-root",
        str(bad_val_root),
    ]
    bad_val = subprocess.run(bad_val_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_val.returncode == 0:
        print(bad_val.stdout)
        raise SystemExit("[FAIL] bad Step 121 manifest unexpectedly passed")

    print("[OK] Step 121 reward ablation execution manifest self-test PASS")
    print("[DONE] Step 121 reward ablation execution manifest complete.")


if __name__ == "__main__":
    main()
