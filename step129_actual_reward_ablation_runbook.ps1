$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

# ---------------------------------------------------------------------
# Step 129 runbook JSON
# ---------------------------------------------------------------------
$RunbookJson = @'
{
  "artifact_version": "actual_reward_ablation_runbook_step129_v1",
  "step": 129,
  "runbook_status": "ACTUAL_ABLATION_RUNBOOK_READY_NOT_EXECUTED",
  "purpose": "Prepare the operator runbook for future actual reward ablation execution while preserving all no-execution and no-claim guards.",
  "source_steps": [
    "Step111 final reward specification draft",
    "Step112 reward candidate protocol",
    "Step113 reward normalization baseline lock",
    "Step114 hard constraint review",
    "Step115 reward ablation matrix",
    "Step116 trainable reward promotion gate",
    "Step117 reward ablation result schema",
    "Step118 reward ablation result writer guard",
    "Step119 reward ablation runner dry-run plan",
    "Step120 reward ablation execution preflight",
    "Step121 reward ablation execution manifest",
    "Step122 reward ablation sandbox no-op runner guard",
    "Step123 reward ablation no-op result ingestion guard",
    "Step124 reward ablation actual result schema bridge",
    "Step125 reward ablation selection criteria gate",
    "Step126 reward ablation actual result ingestion preflight",
    "Step127 trainable reward promotion decision package",
    "Step128 reward pipeline actual ablation wait state"
  ],
  "candidate_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
  "condition_ids": ["A", "A90", "A80", "A70"],
  "seeds": [1, 2, 3],
  "expected_actual_runs": 72,
  "runbook_sections": [
    "operator_preflight",
    "source_code_and_git_state",
    "environment_and_hardware_record",
    "input_artifact_requirements",
    "checkpoint_and_reward_contract",
    "execution_order",
    "per_run_output_requirements",
    "failure_handling",
    "post_run_ingestion",
    "claim_and_promotion_guards"
  ],
  "operator_preflight_required": [
    "git status must be clean or explicitly documented",
    "Step 111-128 artifacts must be committed",
    "Step 119 dry-run plan must exist",
    "Step 120 execution preflight must pass",
    "Step 121 execution manifest must exist",
    "Step 126 actual-result ingestion preflight must be available",
    "H200 or approved execution environment must be documented",
    "reward candidate IDs R0-R5 must match Step 115",
    "B1_noop normalization reference must be preserved",
    "hard constraints from Step 114 must be active"
  ],
  "execution_matrix": {
    "candidate_count": 6,
    "condition_count": 4,
    "seed_count": 3,
    "total_runs": 72,
    "conditions": ["A", "A90", "A80", "A70"],
    "candidates": ["R0", "R1", "R2", "R3", "R4", "R5"],
    "seeds": [1, 2, 3]
  },
  "required_outputs_per_run": [
    "run_config.json",
    "training_or_evaluation_log.txt",
    "canonical_eval_path",
    "reward_candidate_id",
    "condition_id",
    "seed",
    "checkpoint_path",
    "checkpoint_sha256",
    "trained_model",
    "actual_result",
    "hard_constraint_report.json",
    "kpi_by_window.parquet",
    "kpi_by_seed.parquet",
    "kpi_overall.json"
  ],
  "hard_stop_conditions": [
    "any template/no-op/dry-run/smoke-only marker in actual result",
    "missing checkpoint_path or checkpoint_sha256",
    "trained_model=false for an actual result row",
    "actual_result=false for an actual result row",
    "hard constraint failure for selected candidate",
    "candidate/condition/seed combination missing",
    "paper_level_claim_allowed=true before paper evidence gate",
    "causal_performance_claim_allowed=true before causal evidence gate",
    "winner_selected=true before Step 125 selection criteria are applied to actual results"
  ],
  "post_run_required_sequence": [
    "collect all 72 actual run outputs",
    "build actual result bundle following Step 126",
    "run Step 126 actual-result ingestion preflight",
    "populate Step 125 selection scorecard from actual results",
    "prepare a new promotion decision package",
    "only then consider trainable reward promotion"
  ],
  "guard_flags": {
    "actual_execution_started": false,
    "actual_results": false,
    "actual_ablation_data_available": false,
    "winner_selected": false,
    "trainable_reward_promoted": false,
    "train_with_this_reward_allowed": false,
    "actual_training_allowed_by_this_step": false,
    "final_reward_design_claim_allowed": false,
    "best_reward_claim_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false
  },
  "next_status": "READY_FOR_STEP130_PROJECT_LOG_UPDATE_OR_ACTUAL_ABLATION_EXECUTION_ENV_PREFLIGHT"
}
'@

$RunbookJsonPath = ".\05_training\rewards\actual_reward_ablation_runbook_step129.json"
$RunbookJson | Set-Content -Path $RunbookJsonPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 129 markdown
# ---------------------------------------------------------------------
$RunbookMd = @'
# Step 129 — Actual Reward Ablation Runbook

## Purpose

Step 129 prepares the runbook for future actual reward ablation execution.

This step does not execute reward ablation, does not start MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) training, does not select a winner, and does not promote any reward.

## Current Status

```text
ACTUAL_ABLATION_RUNBOOK_READY_NOT_EXECUTED
```

Current guards:

```text
actual_execution_started = false
actual_results = false
winner_selected = false
trainable_reward_promoted = false
train_with_this_reward_allowed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```

## Execution Matrix

Future actual ablation requires:

```text
R0/R1/R2/R3/R4/R5
× A/A90/A80/A70
× seed 1,2,3
= 72 runs
```

## Operator Preflight

Before actual execution:

1. Git state must be clean or explicitly documented.
2. Step 111~128 artifacts must be committed.
3. Step 119 dry-run plan must exist.
4. Step 120 execution preflight must pass.
5. Step 121 execution manifest must exist.
6. Step 126 actual-result ingestion preflight must be available.
7. H200 or approved execution environment must be documented.
8. R0~R5 candidate definitions must match Step 115.
9. B1_noop normalization reference must be preserved.
10. Step 114 hard constraints must be active.

## Per-Run Required Outputs

Each future actual run must produce:

- `run_config.json`
- training/evaluation log
- canonical KPI output path
- reward candidate ID
- condition ID
- seed
- checkpoint path
- checkpoint SHA256
- trained model flag
- actual result flag
- hard constraint report
- `kpi_by_window.parquet`
- `kpi_by_seed.parquet`
- `kpi_overall.json`

## Hard Stop Conditions

Stop immediately if:

- template/no-op/dry-run/smoke-only markers appear in actual results
- checkpoint path or checkpoint hash is missing
- `trained_model=false`
- `actual_result=false`
- any selected candidate violates hard constraints
- any candidate/condition/seed row is missing
- paper or causal claim flags become true before later evidence gates
- winner is selected before actual-result scorecard evaluation

## Post-Run Required Sequence

After future actual execution:

1. collect all 72 actual outputs
2. build actual result bundle
3. run Step 126 ingestion preflight
4. populate Step 125 selection scorecard
5. prepare a new promotion decision package
6. only then consider trainable reward promotion

## Important

This runbook prepares the execution procedure only. It is not permission to execute.
'@

$RunbookMdPath = ".\05_training\rewards\actual_reward_ablation_runbook_step129.md"
$RunbookMd | Set-Content -Path $RunbookMdPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 129 validator
# ---------------------------------------------------------------------
$ValidatorPy = @'
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_SEEDS = [1, 2, 3]
EXPECTED_SOURCE_STEPS = list(range(111, 129))


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


def add_check(checks: List[Dict[str, Any]], check_id: str, expected: Any, actual: Any, description: str) -> None:
    checks.append({
        "check_id": check_id,
        "description": description,
        "expected": expected,
        "actual": actual,
        "passed": expected == actual,
        "severity": "blocker",
        "blocking": True,
    })


def validate_runbook(runbook: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(checks, "artifact_version", "actual_reward_ablation_runbook_step129_v1", runbook.get("artifact_version"), "Artifact version must match Step 129.")
    add_check(checks, "runbook_status", "ACTUAL_ABLATION_RUNBOOK_READY_NOT_EXECUTED", runbook.get("runbook_status"), "Runbook must be ready but not executed.")
    add_check(checks, "candidate_ids", EXPECTED_CANDIDATES, runbook.get("candidate_ids"), "Candidate IDs must be R0-R5.")
    add_check(checks, "condition_ids", EXPECTED_CONDITIONS, runbook.get("condition_ids"), "Conditions must be A/A90/A80/A70.")
    add_check(checks, "seeds", EXPECTED_SEEDS, [int(x) for x in runbook.get("seeds", [])], "Seeds must be 1/2/3.")
    add_check(checks, "expected_actual_runs", 72, int(runbook.get("expected_actual_runs", -1)), "Expected actual runs must be 72.")

    source_steps = runbook.get("source_steps", [])
    for step in EXPECTED_SOURCE_STEPS:
        found = any(f"Step{step}" in str(item) or f"Step {step}" in str(item) for item in source_steps)
        add_check(checks, f"source_step_{step}", True, found, f"Step {step} must be referenced.")

    matrix = runbook.get("execution_matrix", {})
    add_check(checks, "matrix_total_runs", 72, int(matrix.get("total_runs", -1)), "Execution matrix total must be 72.")
    add_check(checks, "matrix_candidate_count", 6, int(matrix.get("candidate_count", -1)), "Matrix candidate count must be 6.")
    add_check(checks, "matrix_condition_count", 4, int(matrix.get("condition_count", -1)), "Matrix condition count must be 4.")
    add_check(checks, "matrix_seed_count", 3, int(matrix.get("seed_count", -1)), "Matrix seed count must be 3.")

    sections = runbook.get("runbook_sections", [])
    for required in [
        "operator_preflight",
        "execution_order",
        "per_run_output_requirements",
        "failure_handling",
        "post_run_ingestion",
        "claim_and_promotion_guards",
    ]:
        add_check(checks, f"section_{required}", True, required in sections, f"Runbook section {required} must exist.")

    preflight = runbook.get("operator_preflight_required", [])
    add_check(checks, "operator_preflight_count_min", True, len(preflight) >= 10, "Operator preflight must be explicit.")

    outputs = runbook.get("required_outputs_per_run", [])
    for required in [
        "run_config.json",
        "checkpoint_path",
        "checkpoint_sha256",
        "trained_model",
        "actual_result",
        "hard_constraint_report.json",
        "kpi_by_window.parquet",
        "kpi_by_seed.parquet",
        "kpi_overall.json",
    ]:
        add_check(checks, f"required_output_{required}", True, required in outputs, f"Required output {required} must be listed.")

    hard_stops = runbook.get("hard_stop_conditions", [])
    add_check(checks, "hard_stop_count_min", True, len(hard_stops) >= 8, "Hard stop list must be explicit.")

    guards = runbook.get("guard_flags", {})
    false_guards = [
        "actual_execution_started",
        "actual_results",
        "actual_ablation_data_available",
        "winner_selected",
        "trainable_reward_promoted",
        "train_with_this_reward_allowed",
        "actual_training_allowed_by_this_step",
        "final_reward_design_claim_allowed",
        "best_reward_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
    ]
    for key in false_guards:
        add_check(checks, f"guard_{key}", False, bool(guards.get(key)), f"{key} must remain false.")

    post = runbook.get("post_run_required_sequence", [])
    for phrase in ["collect all 72", "Step 126", "Step 125", "promotion decision"]:
        found = any(phrase.lower() in str(item).lower() for item in post)
        add_check(checks, f"post_sequence_{phrase.replace(' ', '_')}", True, found, f"Post-run sequence must include {phrase}.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_actual_reward_ablation_runbook_step129_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": (
            "PASS_ACTUAL_ABLATION_RUNBOOK_READY_NOT_EXECUTED"
            if audit_status == "PASS"
            else "FAIL_ACTUAL_ABLATION_RUNBOOK"
        ),
        "next_status": (
            "READY_FOR_STEP130_PROJECT_LOG_UPDATE_OR_ACTUAL_ABLATION_EXECUTION_ENV_PREFLIGHT"
            if audit_status == "PASS"
            else "BLOCKED_FIX_STEP129_RUNBOOK"
        ),
        "expected_actual_runs": int(runbook.get("expected_actual_runs", 0)),
        "actual_execution_started": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "failure_count": len(failures),
        "checks": checks,
    }


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runbook", default="05_training/rewards/actual_reward_ablation_runbook_step129.json")
    parser.add_argument("--output-root", default="artifacts/rewards/actual_reward_ablation_runbook_step129_validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    runbook_path = resolve_path(project_root, args.runbook)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    runbook = load_json_any_encoding(runbook_path)
    report = validate_runbook(runbook)
    report["runbook_path"] = str(runbook_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "actual_reward_ablation_runbook_step129_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 129 actual reward ablation runbook validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] expected_runs : {report['expected_actual_runs']}")
    print(f"[OK] actual_started: {report['actual_execution_started']}")
    print(f"[OK] actual_results: {report['actual_results']}")
    print(f"[OK] winner_selected: {report['winner_selected']}")
    print(f"[OK] promoted      : {report['trainable_reward_promoted']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'@

$ValidatorPath = ".\05_training\rewards\validate_actual_reward_ablation_runbook_step129.py"
$ValidatorPy | Set-Content -Path $ValidatorPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 129 self-test
# ---------------------------------------------------------------------
$TestPy = @'
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
'@

$TestPath = ".\05_training\rewards\test_validate_actual_reward_ablation_runbook_step129.py"
$TestPy | Set-Content -Path $TestPath -Encoding UTF8

# ---------------------------------------------------------------------
# Run Step 129 self-test
# ---------------------------------------------------------------------
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 129 actual reward ablation runbook self-test failed"
}

Write-Host ""
Write-Host "[DONE] Step 129 actual reward ablation runbook files created and tested."
Write-Host "[NEXT] Review git status, then commit the Step 129 actual reward ablation runbook files."
