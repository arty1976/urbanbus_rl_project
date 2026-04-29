from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
CONDITIONS = ["A", "A90", "A80", "A70"]
SEEDS = [1, 2, 3]


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_step119_fixture(root: Path, corrupt_execute: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for cand in CANDIDATES:
        for cond in CONDITIONS:
            for seed in SEEDS:
                cmd = (
                    "python 05_training/rewards/reward_ablation_result_writer_guard_step118.py "
                    f"--candidate-id {cand} --condition-id {cond} --seed {seed} "
                    f"--output-root artifacts/rewards/ablation/{cand}/{cond}/seed_{seed:03d} "
                    "--dry-run-plan"
                )
                if corrupt_execute and cand == "R0" and cond == "A" and seed == 1:
                    cmd += " --execute"
                rows.append({
                    "run_id": f"{cand}_{cond}_seed_{seed:03d}",
                    "candidate_id": cand,
                    "condition_id": cond,
                    "seed": seed,
                    "command_type": "dry_run_plan_only",
                    "execute_allowed": "false",
                    "actual_training_allowed": "false",
                    "train_with_this_reward_allowed": "false",
                    "actual_results": "false",
                    "winner_selected": "false",
                    "best_reward_claim_allowed": "false",
                    "planned_output_root": f"artifacts/rewards/ablation/{cand}/{cond}/seed_{seed:03d}",
                    "planned_command_text": cmd,
                })

    plan_csv = root / "reward_ablation_runner_dry_run_plan.csv"
    with open(plan_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plan_json = root / "reward_ablation_runner_dry_run_plan.json"
    dump_json(plan_json, {"rows": rows})

    manifest = {
        "artifact_version": "reward_ablation_runner_dry_run_plan_step119_manifest_fixture_v1",
        "planner_status": "DRY_RUN_ROWS_WRITTEN_NOT_EXECUTED",
        "planned_runs": len(rows),
        "execute_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "output_root": str(root),
        "output_files": {
            "plan_csv": str(plan_csv),
            "plan_json": str(plan_json)
        }
    }
    dump_json(root / "reward_ablation_runner_dry_run_manifest.json", manifest)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    unique = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"

    preflight = project_root / "05_training" / "rewards" / "reward_ablation_execution_preflight_step120.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_execution_preflight_step120.py"

    fixture_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_preflight_step120_fixture_{unique}"
    report_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_preflight_step120_selftest_{unique}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_preflight_step120_validation_{unique}"

    write_step119_fixture(fixture_root, corrupt_execute=False)

    cmd = [sys.executable, str(preflight), "--step119-plan-root", str(fixture_root), "--output-root", str(report_root)]
    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 120 valid preflight unexpectedly failed")

    report_path = report_root / "reward_ablation_execution_preflight_step120_report.json"
    report = load_json(report_path)
    if report.get("audit_status") != "PASS":
        raise SystemExit("[FAIL] Step 120 report audit_status must be PASS")
    if report.get("planned_runs") != 72:
        raise SystemExit("[FAIL] Step 120 planned_runs must be 72")
    if report.get("execute_allowed") is not False:
        raise SystemExit("[FAIL] Step 120 execute_allowed must be false")

    vcmd = [sys.executable, str(validator), "--report", str(report_path), "--output-root", str(validation_root)]
    vresult = subprocess.run(vcmd, cwd=project_root, text=True, capture_output=True)
    print(vresult.stdout)
    if vresult.returncode != 0:
        print(vresult.stderr)
        raise SystemExit("[FAIL] Step 120 report validator unexpectedly failed")

    # Bad case: command contains --execute and must fail at preflight.
    bad_fixture_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_preflight_step120_bad_fixture_{unique}"
    bad_report_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_preflight_step120_bad_selftest_{unique}"
    write_step119_fixture(bad_fixture_root, corrupt_execute=True)
    bad_cmd = [sys.executable, str(preflight), "--step119-plan-root", str(bad_fixture_root), "--output-root", str(bad_report_root)]
    bad_result = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] corrupted Step 119 plan unexpectedly passed Step 120 preflight")

    bad_report = load_json(bad_report_root / "reward_ablation_execution_preflight_step120_report.json")
    failed_ids = {c["check_id"] for c in bad_report.get("checks", []) if not c.get("passed")}
    if "forbidden_command_tokens_absent" not in failed_ids:
        raise SystemExit("[FAIL] forbidden command token guard did not fail")

    print("[OK] Step 120 reward ablation execution preflight self-test PASS")
    print("[DONE] Step 120 reward ablation execution preflight complete.")


if __name__ == "__main__":
    main()
