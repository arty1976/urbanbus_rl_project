$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

# ---------------------------------------------------------------------
# Step 119 dry-run plan JSON
# ---------------------------------------------------------------------
$PlanJson = @'
{
  "artifact_version": "reward_ablation_runner_dry_run_plan_step119_v1",
  "step": 119,
  "plan_status": "DRY_RUN_PLAN_NOT_EXECUTABLE",
  "purpose": "Create a dry-run-only execution plan for reward ablation candidates R0-R5 without running training or selecting a winner.",
  "required_previous_steps": [
    "Step111 final reward specification draft",
    "Step112 reward candidate protocol",
    "Step113 reward normalization baseline lock",
    "Step114 hard constraint review",
    "Step115 reward ablation matrix",
    "Step116 trainable reward promotion gate",
    "Step117 reward ablation result schema",
    "Step118 reward ablation result writer guard"
  ],
  "required_input_artifacts": [
    "05_training/rewards/reward_ablation_matrix_step115.json",
    "05_training/rewards/reward_ablation_result_schema_step117.json",
    "05_training/rewards/reward_ablation_result_writer_guard_step118.py",
    "05_training/rewards/validate_reward_ablation_result_writer_guard_step118.py"
  ],
  "candidate_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
  "condition_ids": ["A", "A90", "A80", "A70"],
  "seeds": [1, 2, 3],
  "planned_run_count": 72,
  "execution_guards": {
    "dry_run_only": true,
    "execute_allowed": false,
    "actual_training_allowed": false,
    "train_with_this_reward_allowed": false,
    "actual_results": false,
    "winner_selected": false,
    "best_reward_claim_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false
  },
  "command_contract": {
    "entrypoint": "05_training/rewards/reward_ablation_result_writer_guard_step118.py",
    "command_mode": "template_generation_only",
    "required_flag": "--dry-run-plan",
    "forbidden_flags": ["--execute", "--train", "--promote", "--select-winner"],
    "note": "Step 119 commands are placeholders for operator review. They must not execute MAPPO training."
  },
  "output_contract": {
    "plan_json": "artifacts/rewards/reward_ablation_runner_dry_run_plan_step119/reward_ablation_runner_dry_run_plan.json",
    "plan_csv": "artifacts/rewards/reward_ablation_runner_dry_run_plan_step119/reward_ablation_runner_dry_run_plan.csv",
    "manifest_json": "artifacts/rewards/reward_ablation_runner_dry_run_plan_step119/reward_ablation_runner_dry_run_manifest.json"
  },
  "next_step": "Step120 reward ablation runner guard or Step120 reward ablation execution preflight"
}
'@

$PlanJsonPath = ".\05_training\rewards\reward_ablation_runner_dry_run_plan_step119.json"
$PlanJson | Set-Content -Path $PlanJsonPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 119 Markdown
# ---------------------------------------------------------------------
$PlanMd = @'
# Step 119 — Reward Ablation Runner Dry-Run Plan

## Purpose

Step 119 creates a dry-run-only run plan for reward ablation candidates.

It does not run MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) training, does not generate actual reward ablation results, and does not select a best reward.

## Planned Grid

The dry-run plan covers:

- reward candidates: `R0`, `R1`, `R2`, `R3`, `R4`, `R5`
- conditions: `A`, `A90`, `A80`, `A70`
- seeds: `1`, `2`, `3`

Total planned run rows:

```text
6 candidates × 4 conditions × 3 seeds = 72 planned rows
```

## Execution Guards

The following must remain false:

- `execute_allowed`
- `actual_training_allowed`
- `train_with_this_reward_allowed`
- `actual_results`
- `winner_selected`
- `best_reward_claim_allowed`
- `paper_level_claim_allowed`
- `causal_performance_claim_allowed`

## Command Rule

Every planned command must be review-only or dry-run-only.

Required:

```text
--dry-run-plan
```

Forbidden:

```text
--execute
--train
--promote
--select-winner
```

## Meaning

Step 119 prepares the runner grid so that later steps can decide how to execute reward candidate comparison safely. It is still not a training approval step.

## Next Step

Recommended next step:

```text
Step120 reward ablation runner guard / execution preflight
```
'@

$PlanMdPath = ".\05_training\rewards\reward_ablation_runner_dry_run_plan_step119.md"
$PlanMd | Set-Content -Path $PlanMdPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 119 planner script
# ---------------------------------------------------------------------
$PlannerPy = @'
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "reward_ablation_runner_dry_run_plan_step119_v1"
EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_SEEDS = [1, 2, 3]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def build_rows(plan: Dict[str, Any], project_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    entrypoint = str(plan.get("command_contract", {}).get("entrypoint", ""))

    for candidate_id in plan.get("candidate_ids", []):
        for condition_id in plan.get("condition_ids", []):
            for seed in plan.get("seeds", []):
                run_id = f"step119_{candidate_id}_{condition_id}_seed_{int(seed):03d}"
                planned_output_root = output_root / "planned_runs" / candidate_id / condition_id / f"seed_{int(seed):03d}"
                command = [
                    "python",
                    entrypoint,
                    "--candidate-id",
                    str(candidate_id),
                    "--condition-id",
                    str(condition_id),
                    "--seed",
                    str(int(seed)),
                    "--output-root",
                    str(planned_output_root),
                    "--dry-run-plan",
                ]
                rows.append({
                    "run_id": run_id,
                    "candidate_id": str(candidate_id),
                    "condition_id": str(condition_id),
                    "seed": int(seed),
                    "command_type": "dry_run_plan_only",
                    "execute_allowed": False,
                    "actual_training_allowed": False,
                    "train_with_this_reward_allowed": False,
                    "actual_results": False,
                    "winner_selected": False,
                    "planned_output_root": str(planned_output_root),
                    "planned_command_json": json.dumps(command, ensure_ascii=False),
                    "planned_command_text": " ".join(command),
                })

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("no rows to write")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="05_training/rewards/reward_ablation_runner_dry_run_plan_step119.json")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_runner_dry_run_plan_step119_selftest")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    plan_path = resolve_path(project_root, args.plan)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not plan_path.exists():
        raise SystemExit(f"plan not found: {plan_path}")

    plan = load_json_any_encoding(plan_path)
    rows = build_rows(plan, project_root, output_root)

    plan_payload = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "planner_status": "DRY_RUN_ROWS_WRITTEN_NOT_EXECUTED",
        "source_plan": str(plan_path),
        "candidate_ids": plan.get("candidate_ids", []),
        "condition_ids": plan.get("condition_ids", []),
        "seeds": plan.get("seeds", []),
        "planned_run_count": len(rows),
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "best_reward_claim_allowed": False,
        "rows": rows,
    }

    plan_json_path = output_root / "reward_ablation_runner_dry_run_plan.json"
    plan_csv_path = output_root / "reward_ablation_runner_dry_run_plan.csv"
    manifest_path = output_root / "reward_ablation_runner_dry_run_manifest.json"

    dump_json(plan_json_path, plan_payload)
    write_csv(plan_csv_path, rows)

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "writer_status": "DRY_RUN_PLAN_WRITTEN_NOT_EXECUTED",
        "output_root": str(output_root),
        "output_files": {
            "plan_json": str(plan_json_path),
            "plan_csv": str(plan_csv_path),
            "manifest_json": str(manifest_path),
        },
        "candidate_count": len(plan.get("candidate_ids", [])),
        "condition_count": len(plan.get("condition_ids", [])),
        "seed_count": len(plan.get("seeds", [])),
        "planned_run_count": len(rows),
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
    }
    dump_json(manifest_path, manifest)

    print("[OK] Step 119 reward ablation runner dry-run plan written")
    print(f"[OK] planner_status : {plan_payload['planner_status']}")
    print(f"[OK] candidate_count: {manifest['candidate_count']}")
    print(f"[OK] condition_count: {manifest['condition_count']}")
    print(f"[OK] seed_count     : {manifest['seed_count']}")
    print(f"[OK] planned_runs   : {manifest['planned_run_count']}")
    print(f"[OK] execute_allowed: {manifest['execute_allowed']}")
    print(f"[OK] train_allowed  : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] output_root    : {output_root}")
    print(f"[OK] manifest       : {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@

$PlannerPath = ".\05_training\rewards\reward_ablation_runner_dry_run_plan_step119.py"
$PlannerPy | Set-Content -Path $PlannerPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 119 validator
# ---------------------------------------------------------------------
$ValidatorPy = @'
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "validate_reward_ablation_runner_dry_run_plan_step119_v1"
EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_SEEDS = [1, 2, 3]
EXPECTED_RUN_COUNT = len(EXPECTED_CANDIDATES) * len(EXPECTED_CONDITIONS) * len(EXPECTED_SEEDS)
FORBIDDEN_FLAGS = ["--execute", "--train", "--promote", "--select-winner"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read csv: {path}")


def add_check(checks: List[Dict[str, Any]], check_id: str, expected: Any, actual: Any, description: str) -> None:
    checks.append({
        "check_id": check_id,
        "description": description,
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
        "blocking": True,
    })


def validate(plan_spec: Dict[str, Any], manifest: Dict[str, Any], csv_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(checks, "plan_artifact_version", "reward_ablation_runner_dry_run_plan_step119_v1", plan_spec.get("artifact_version"), "Plan spec artifact version must match.")
    add_check(checks, "plan_status", "DRY_RUN_PLAN_NOT_EXECUTABLE", plan_spec.get("plan_status"), "Plan must remain dry-run-only.")
    add_check(checks, "candidates_exact", EXPECTED_CANDIDATES, plan_spec.get("candidate_ids"), "Candidate IDs must be R0-R5.")
    add_check(checks, "conditions_set", sorted(EXPECTED_CONDITIONS), sorted(plan_spec.get("condition_ids", [])), "Condition set must match A-family.")
    add_check(checks, "seeds_exact", EXPECTED_SEEDS, plan_spec.get("seeds"), "Seeds must match expected list.")
    add_check(checks, "planned_run_count_spec", EXPECTED_RUN_COUNT, int(plan_spec.get("planned_run_count", -1)), "Spec planned run count must be 72.")

    guards = plan_spec.get("execution_guards", {})
    for key in [
        "execute_allowed",
        "actual_training_allowed",
        "train_with_this_reward_allowed",
        "actual_results",
        "winner_selected",
        "best_reward_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
    ]:
        add_check(checks, f"guard_{key}", False, guards.get(key), f"{key} must remain false.")
    add_check(checks, "guard_dry_run_only", True, guards.get("dry_run_only"), "dry_run_only must be true.")

    add_check(checks, "manifest_writer_status", "DRY_RUN_PLAN_WRITTEN_NOT_EXECUTED", manifest.get("writer_status"), "Manifest writer status must be dry-run only.")
    add_check(checks, "manifest_planned_run_count", EXPECTED_RUN_COUNT, int(manifest.get("planned_run_count", -1)), "Manifest run count must be 72.")
    add_check(checks, "csv_row_count", EXPECTED_RUN_COUNT, len(csv_rows), "CSV row count must be 72.")

    combos = set()
    for i, row in enumerate(csv_rows):
        candidate_id = row.get("candidate_id")
        condition_id = row.get("condition_id")
        seed = int(row.get("seed", -999))
        combos.add((candidate_id, condition_id, seed))
        prefix = f"row_{i:03d}"
        add_check(checks, f"{prefix}_candidate_known", True, candidate_id in EXPECTED_CANDIDATES, "Candidate must be known.")
        add_check(checks, f"{prefix}_condition_known", True, condition_id in EXPECTED_CONDITIONS, "Condition must be known.")
        add_check(checks, f"{prefix}_seed_known", True, seed in EXPECTED_SEEDS, "Seed must be known.")
        add_check(checks, f"{prefix}_execute_false", "False", str(row.get("execute_allowed")), "execute_allowed must be false in row.")
        add_check(checks, f"{prefix}_train_false", "False", str(row.get("train_with_this_reward_allowed")), "train flag must be false in row.")
        cmd_text = str(row.get("planned_command_text", ""))
        add_check(checks, f"{prefix}_has_dry_run_flag", True, "--dry-run-plan" in cmd_text, "Command must include --dry-run-plan.")
        for flag in FORBIDDEN_FLAGS:
            add_check(checks, f"{prefix}_forbid_{flag.replace('-', '_')}", False, flag in cmd_text, f"Command must not include {flag}.")

    expected_combos = {
        (c, cond, s)
        for c in EXPECTED_CANDIDATES
        for cond in EXPECTED_CONDITIONS
        for s in EXPECTED_SEEDS
    }
    add_check(checks, "combo_coverage", expected_combos, combos, "All candidate-condition-seed combinations must exist exactly once.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": "PASS_REWARD_ABLATION_RUNNER_DRY_RUN_PLAN_NOT_EXECUTED" if audit_status == "PASS" else "FAIL_REWARD_ABLATION_RUNNER_DRY_RUN_PLAN",
        "next_status": "READY_FOR_STEP120_REWARD_ABLATION_EXECUTION_PREFLIGHT" if audit_status == "PASS" else "BLOCKED_FIX_STEP119_PLAN",
        "candidate_count": len(EXPECTED_CANDIDATES),
        "condition_count": len(EXPECTED_CONDITIONS),
        "seed_count": len(EXPECTED_SEEDS),
        "planned_run_count": len(csv_rows),
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "failure_count": len(failures),
        "check_count": len(checks),
        "checks": checks,
    }


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-spec", default="05_training/rewards/reward_ablation_runner_dry_run_plan_step119.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_runner_dry_run_plan_step119_validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    spec_path = resolve_path(project_root, args.plan_spec)
    manifest_path = resolve_path(project_root, args.manifest)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    plan_spec = load_json_any_encoding(spec_path)
    manifest = load_json_any_encoding(manifest_path)
    csv_path = Path(manifest["output_files"]["plan_csv"])
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path
    csv_rows = read_csv_rows(csv_path)

    report = validate(plan_spec, manifest, csv_rows)
    report["plan_spec"] = str(spec_path)
    report["manifest"] = str(manifest_path)
    report["plan_csv"] = str(csv_path)

    report_path = output_root / "reward_ablation_runner_dry_run_plan_step119_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 119 reward ablation runner dry-run plan validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] planned_runs  : {report['planned_run_count']}")
    print(f"[OK] execute_allowed: {report['execute_allowed']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] actual_results: {report['actual_results']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'@

$ValidatorPath = ".\05_training\rewards\validate_reward_ablation_runner_dry_run_plan_step119.py"
$ValidatorPy | Set-Content -Path $ValidatorPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 119 self-test
# ---------------------------------------------------------------------
$TestPy = @'
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
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

    unique = datetime.utcnow().strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"
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
'@

$TestPath = ".\05_training\rewards\test_validate_reward_ablation_runner_dry_run_plan_step119.py"
$TestPy | Set-Content -Path $TestPath -Encoding UTF8

# ---------------------------------------------------------------------
# Run Step 119 self-test
# ---------------------------------------------------------------------
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 119 reward ablation runner dry-run plan self-test failed"
}

Write-Host ""
Write-Host "[DONE] Step 119 reward ablation runner dry-run plan files created and tested."
Write-Host "[NEXT] Review git status, then commit the Step 119 dry-run plan files."
