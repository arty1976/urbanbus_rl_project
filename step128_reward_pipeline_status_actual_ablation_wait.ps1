$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

# ---------------------------------------------------------------------
# Step 128 status JSON
# ---------------------------------------------------------------------
$StatusJson = @'
{
  "artifact_version": "reward_pipeline_status_actual_ablation_wait_step128_v1",
  "step": 128,
  "pipeline_status": "REWARD_PIPELINE_DOCUMENTED_WAITING_FOR_ACTUAL_ABLATION_DATA",
  "purpose": "Document the completed reward-design gate pipeline from Step 111 through Step 127 and explicitly hold the system in a waiting state until actual reward ablation data exists.",
  "covered_steps": [
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
    "Step127 trainable reward promotion decision package"
  ],
  "candidate_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
  "condition_ids": ["A", "A90", "A80", "A70"],
  "seeds": [1, 2, 3],
  "expected_actual_ablation_rows": 72,
  "current_artifact_class": {
    "reward_spec": "draft",
    "candidate_protocol": "defined",
    "baseline_reference": "B1_noop_locked_as_reference_name_only",
    "hard_constraints": "reviewed_not_final_training_unlock",
    "ablation_matrix": "draft_not_trainable",
    "dry_run_plan": "not_executed",
    "execution_manifest": "not_executable",
    "sandbox_runner": "noop_only",
    "result_ingestion": "noop_or_template_only",
    "actual_result_ingestion": "not_ingested",
    "promotion_decision": "not_promoted"
  },
  "required_actual_evidence_before_next_promotion_attempt": [
    "actual_reward_ablation_bundle",
    "72 actual result rows for R0-R5 x A/A90/A80/A70 x seeds 1/2/3",
    "trained_model=true evidence",
    "checkpoint_path and checkpoint_sha256 for every actual row",
    "canonical KPI outputs for every row",
    "hard constraint pass/fail outputs",
    "selection criteria scorecard populated from actual results",
    "no no-op/template/dry-run/smoke-only rows",
    "separate paper-level and causal-claim gates"
  ],
  "guard_flags": {
    "actual_results": false,
    "actual_ablation_data_available": false,
    "winner_selected": false,
    "trainable_reward_promoted": false,
    "train_with_this_reward_allowed": false,
    "actual_training_allowed": false,
    "final_reward_design_claim_allowed": false,
    "best_reward_claim_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false
  },
  "allowed_next_actions": [
    "commit Step 128 documentation",
    "push local commits to origin/main",
    "wait for actual reward ablation result bundle",
    "prepare H200 or simulator runbook for actual ablation execution in a separate guarded step"
  ],
  "prohibited_next_actions": [
    "do not select R0-R5 winner from template/no-op/dry-run outputs",
    "do not train MAPPO with any Step 111-127 reward candidate yet",
    "do not claim best reward",
    "do not make paper-level performance claims",
    "do not make causal performance claims"
  ],
  "next_status": "WAIT_FOR_ACTUAL_ABLATION_DATA_OR_PREPARE_ACTUAL_ABLATION_RUNBOOK"
}
'@

$StatusJsonPath = ".\05_training\rewards\reward_pipeline_status_actual_ablation_wait_step128.json"
$StatusJson | Set-Content -Path $StatusJsonPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 128 markdown
# ---------------------------------------------------------------------
$StatusMd = @'
# Step 128 — Reward Pipeline Status and Actual Ablation Data Wait

## Purpose

Step 128 closes the current reward-design gate sequence by documenting the status of Step 111 through Step 127.

This step does not run reward ablation, does not choose a reward winner, and does not allow MAPPO training.

## Current Pipeline Status

```text
REWARD_PIPELINE_DOCUMENTED_WAITING_FOR_ACTUAL_ABLATION_DATA
```

## Completed Gate Chain

- Step 111 — final reward specification draft
- Step 112 — reward candidate protocol
- Step 113 — B1_noop normalization baseline reference lock
- Step 114 — hard constraint review
- Step 115 — R0~R5 reward ablation matrix
- Step 116 — trainable reward promotion gate
- Step 117 — reward ablation result schema
- Step 118 — result writer guard
- Step 119 — dry-run runner plan
- Step 120 — execution preflight
- Step 121 — execution manifest
- Step 122 — sandbox no-op runner guard
- Step 123 — no-op result ingestion guard
- Step 124 — actual result schema bridge
- Step 125 — selection criteria gate
- Step 126 — actual result ingestion preflight
- Step 127 — trainable reward promotion decision package

## Current Decision

No reward is promoted.

```text
actual_results = false
winner_selected = false
trainable_reward_promoted = false
train_with_this_reward_allowed = false
```

## Why Promotion Is Blocked

The system has created specifications, schemas, dry-run plans, execution manifests, no-op guards, ingestion gates, and decision-package templates.

However, no actual reward ablation result bundle exists yet.

Therefore, R0~R5 cannot be ranked, and no candidate can be selected.

## Required Actual Evidence

Future promotion requires 72 actual rows:

```text
R0/R1/R2/R3/R4/R5
× A/A90/A80/A70
× seed 1,2,3
= 72 rows
```

Each row must come from real actual execution, not no-op/template/dry-run/smoke output.

## Prohibited Claims

Until actual evidence is ingested and later gates pass:

- no best reward claim
- no trainable reward promotion
- no MAPPO training with these candidates
- no paper-level performance claim
- no causal performance claim

## Next Options

1. Commit and push Step 111~128 reward pipeline work.
2. Wait for actual reward ablation data.
3. Prepare an actual reward ablation runbook or H200 execution plan in a separate guarded step.
'@

$StatusMdPath = ".\05_training\rewards\reward_pipeline_status_actual_ablation_wait_step128.md"
$StatusMd | Set-Content -Path $StatusMdPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 128 validator
# ---------------------------------------------------------------------
$ValidatorPy = @'
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_STEPS = list(range(111, 128))
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


def add_check(checks: List[Dict[str, Any]], check_id: str, expected: Any, actual: Any, description: str) -> None:
    checks.append({
        "check_id": check_id,
        "description": description,
        "expected": expected,
        "actual": actual,
        "passed": expected == actual,
        "severity": "blocker",
        "blocking": True
    })


def validate_status(status: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(
        checks,
        "artifact_version",
        "reward_pipeline_status_actual_ablation_wait_step128_v1",
        status.get("artifact_version"),
        "Artifact version must match Step 128."
    )
    add_check(
        checks,
        "pipeline_status",
        "REWARD_PIPELINE_DOCUMENTED_WAITING_FOR_ACTUAL_ABLATION_DATA",
        status.get("pipeline_status"),
        "Pipeline must wait for actual ablation data."
    )

    covered = status.get("covered_steps", [])
    for step in EXPECTED_STEPS:
        found = any(f"Step{step}" in str(item) or f"Step {step}" in str(item) for item in covered)
        add_check(checks, f"covered_step_{step}", True, found, f"Step {step} must be covered.")

    add_check(checks, "candidate_ids", EXPECTED_CANDIDATES, status.get("candidate_ids"), "Candidate IDs must be R0-R5.")
    add_check(checks, "condition_ids", EXPECTED_CONDITIONS, status.get("condition_ids"), "Conditions must be A/A90/A80/A70.")
    add_check(checks, "seeds", EXPECTED_SEEDS, [int(x) for x in status.get("seeds", [])], "Seeds must be 1/2/3.")
    add_check(checks, "expected_actual_rows", 72, int(status.get("expected_actual_ablation_rows", -1)), "Expected actual ablation rows must be 72.")

    guards = status.get("guard_flags", {})
    false_guards = [
        "actual_results",
        "actual_ablation_data_available",
        "winner_selected",
        "trainable_reward_promoted",
        "train_with_this_reward_allowed",
        "actual_training_allowed",
        "final_reward_design_claim_allowed",
        "best_reward_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
    ]
    for key in false_guards:
        add_check(checks, f"guard_{key}", False, bool(guards.get(key)), f"{key} must remain false.")

    required_evidence = status.get("required_actual_evidence_before_next_promotion_attempt", [])
    add_check(checks, "required_evidence_count_min", True, len(required_evidence) >= 8, "Required actual evidence list must be explicit.")

    prohibited = status.get("prohibited_next_actions", [])
    for phrase in ["do not select", "do not train", "do not claim"]:
        found = any(phrase in str(item).lower() for item in prohibited)
        add_check(checks, f"prohibited_phrase_{phrase.replace(' ', '_')}", True, found, f"Prohibited actions must include '{phrase}'.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_reward_pipeline_status_actual_ablation_wait_step128_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": (
            "PASS_REWARD_PIPELINE_DOCUMENTED_WAITING_FOR_ACTUAL_ABLATION_DATA"
            if audit_status == "PASS"
            else "FAIL_REWARD_PIPELINE_STATUS"
        ),
        "next_status": (
            "WAIT_FOR_ACTUAL_ABLATION_DATA_OR_PREPARE_ACTUAL_ABLATION_RUNBOOK"
            if audit_status == "PASS"
            else "BLOCKED_FIX_STEP128_STATUS"
        ),
        "covered_step_count": len(EXPECTED_STEPS),
        "expected_actual_ablation_rows": int(status.get("expected_actual_ablation_rows", 0)),
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "train_with_this_reward_allowed": False,
        "failure_count": len(failures),
        "checks": checks
    }


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", default="05_training/rewards/reward_pipeline_status_actual_ablation_wait_step128.json")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_pipeline_status_actual_ablation_wait_step128_validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    status_path = resolve_path(project_root, args.status)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    status = load_json_any_encoding(status_path)
    report = validate_status(status)
    report["status_path"] = str(status_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "reward_pipeline_status_actual_ablation_wait_step128_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 128 reward pipeline status validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] covered_steps : {report['covered_step_count']}")
    print(f"[OK] expected_rows : {report['expected_actual_ablation_rows']}")
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

$ValidatorPath = ".\05_training\rewards\validate_reward_pipeline_status_actual_ablation_wait_step128.py"
$ValidatorPy | Set-Content -Path $ValidatorPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 128 self-test
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
'@

$TestPath = ".\05_training\rewards\test_validate_reward_pipeline_status_actual_ablation_wait_step128.py"
$TestPy | Set-Content -Path $TestPath -Encoding UTF8

# ---------------------------------------------------------------------
# Run Step 128 self-test
# ---------------------------------------------------------------------
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 128 reward pipeline status self-test failed"
}

Write-Host ""
Write-Host "[DONE] Step 128 reward pipeline status files created and tested."
Write-Host "[NEXT] Review git status, then commit the Step 128 reward pipeline status files."
