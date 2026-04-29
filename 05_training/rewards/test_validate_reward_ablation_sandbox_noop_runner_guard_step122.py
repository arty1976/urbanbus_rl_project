from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
CONDITIONS = ["A", "A90", "A80", "A70"]
SEEDS = [1, 2, 3]


def unique_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_mock_step121_manifest(project_root: Path, root: Path, unsafe: bool = False) -> Path:
    rows = []
    for candidate_id in CANDIDATES:
        for condition_id in CONDITIONS:
            for seed in SEEDS:
                command = (
                    "python 05_training/rewards/reward_ablation_result_writer_guard_step118.py "
                    f"--candidate-id {candidate_id} --condition-id {condition_id} --seed {seed} --dry-run-plan"
                )
                if unsafe and candidate_id == "R0" and condition_id == "A" and seed == 1:
                    command += " --execute"
                rows.append({
                    "manifest_run_id": f"{candidate_id}_{condition_id}_seed{seed:03d}",
                    "candidate_id": candidate_id,
                    "condition_id": condition_id,
                    "seed": seed,
                    "planned_output_root": str(root / "planned" / candidate_id / condition_id / f"seed_{seed:03d}"),
                    "planned_command_text": command,
                    "command_hash": f"hash_{candidate_id}_{condition_id}_{seed}",
                    "execute_allowed": False,
                    "actual_training_allowed": False,
                    "train_with_this_reward_allowed": False,
                    "actual_results": False,
                    "winner_selected": False,
                })

    plan_csv = root / "reward_ablation_execution_manifest_rows.csv"
    write_csv(plan_csv, rows)
    manifest = {
        "artifact_version": "mock_step121_execution_manifest_v1",
        "row_count": len(rows),
        "output_files": {
            "execution_manifest_csv": str(plan_csv),
        },
    }
    manifest_path = root / "reward_ablation_execution_manifest_step121_manifest.json"
    dump_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    runner = project_root / "05_training" / "rewards" / "reward_ablation_sandbox_noop_runner_guard_step122.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_sandbox_noop_runner_guard_step122.py"
    tag = unique_tag()

    source_root = project_root / "artifacts" / "rewards" / f"step122_mock_step121_source_{tag}"
    output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_sandbox_noop_runner_guard_step122_selftest_{tag}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_sandbox_noop_runner_guard_step122_validation_{tag}"

    manifest_path = write_mock_step121_manifest(project_root, source_root)

    run_cmd = [
        sys.executable,
        str(runner),
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(output_root),
    ]
    run_result = subprocess.run(run_cmd, cwd=project_root, text=True, capture_output=True)
    print(run_result.stdout)
    if run_result.returncode != 0:
        print(run_result.stderr)
        raise SystemExit("[FAIL] Step 122 runner failed")

    out_manifest = output_root / "reward_ablation_sandbox_noop_runner_guard_manifest.json"
    if not out_manifest.exists():
        raise SystemExit("[FAIL] Step 122 output manifest missing")

    val_cmd = [
        sys.executable,
        str(validator),
        "--manifest",
        str(out_manifest),
        "--output-root",
        str(validation_root),
    ]
    val_result = subprocess.run(val_cmd, cwd=project_root, text=True, capture_output=True)
    print(val_result.stdout)
    if val_result.returncode != 0:
        print(val_result.stderr)
        raise SystemExit("[FAIL] Step 122 validator unexpectedly failed")

    report = load_json(validation_root / "reward_ablation_sandbox_noop_runner_guard_step122_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_REWARD_ABLATION_SANDBOX_NOOP_RUNNER_GUARD_NOT_EXECUTED",
        "next_status": "READY_FOR_STEP123_REWARD_ABLATION_RESULT_INGESTION_GUARD",
        "row_count": 72,
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

    # Negative input test: unsafe command with --execute must be rejected by runner.
    bad_source_root = project_root / "artifacts" / "rewards" / f"step122_bad_input_source_{tag}"
    bad_manifest = write_mock_step121_manifest(project_root, bad_source_root, unsafe=True)
    bad_result = subprocess.run([
        sys.executable,
        str(runner),
        "--manifest",
        str(bad_manifest),
        "--output-root",
        str(project_root / "artifacts" / "rewards" / f"step122_bad_input_output_{tag}"),
    ], cwd=project_root, text=True, capture_output=True)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] unsafe --execute command unexpectedly passed runner")

    # Negative output test: if a status row says command_executed=True, validator must fail.
    bad_output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_sandbox_noop_runner_guard_step122_bad_output_{tag}"
    shutil.copytree(output_root, bad_output_root)
    bad_manifest_path = bad_output_root / "reward_ablation_sandbox_noop_runner_guard_manifest.json"
    bad_manifest_payload = load_json(bad_manifest_path)
    bad_status_csv = bad_output_root / "reward_ablation_sandbox_noop_status.csv"
    rows = read_csv(bad_status_csv)
    rows[0]["command_executed"] = "True"
    write_csv(bad_status_csv, rows)
    bad_manifest_payload["output_root"] = str(bad_output_root)
    bad_manifest_payload["output_files"]["status_csv"] = str(bad_status_csv)
    bad_manifest_payload["output_files"]["status_json"] = str(bad_output_root / "reward_ablation_sandbox_noop_status.json")
    bad_manifest_payload["output_files"]["manifest_json"] = str(bad_manifest_path)
    dump_json(bad_manifest_path, bad_manifest_payload)

    bad_val_root = project_root / "artifacts" / "rewards" / f"reward_ablation_sandbox_noop_runner_guard_step122_bad_validation_{tag}"
    bad_val_result = subprocess.run([
        sys.executable,
        str(validator),
        "--manifest",
        str(bad_manifest_path),
        "--output-root",
        str(bad_val_root),
    ], cwd=project_root, text=True, capture_output=True)
    if bad_val_result.returncode == 0:
        print(bad_val_result.stdout)
        raise SystemExit("[FAIL] corrupted command_executed=True output unexpectedly passed")

    bad_report = load_json(bad_val_root / "reward_ablation_sandbox_noop_runner_guard_step122_validation_report.json")
    failed_ids = {
        c["check_id"]
        for c in bad_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "all_command_not_executed" not in failed_ids:
        raise SystemExit("[FAIL] command_executed corruption did not trigger all_command_not_executed")

    print("[OK] Step 122 reward ablation sandbox no-op runner guard self-test PASS")
    print("[DONE] Step 122 reward ablation sandbox no-op runner guard complete.")


if __name__ == "__main__":
    main()
