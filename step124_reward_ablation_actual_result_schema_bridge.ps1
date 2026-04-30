$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

# ---------------------------------------------------------------------
# Step 124 spec JSON
# ---------------------------------------------------------------------
$SpecJson = @'
{
  "artifact_version": "reward_ablation_actual_result_schema_bridge_step124_v1",
  "step": 124,
  "bridge_status": "ACTUAL_RESULT_SCHEMA_BRIDGE_NOT_ACTUAL_RESULTS",
  "purpose": "Define the schema bridge between Step 123 no-op ingestion outputs and future actual reward ablation result ingestion.",
  "source_steps": [
    "Step117 reward ablation result schema",
    "Step118 reward ablation result writer guard",
    "Step119 reward ablation runner dry-run plan",
    "Step120 reward ablation execution preflight",
    "Step121 reward ablation execution manifest",
    "Step122 reward ablation sandbox no-op runner guard",
    "Step123 reward ablation no-op result ingestion guard"
  ],
  "candidate_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
  "condition_ids": ["A", "A90", "A80", "A70"],
  "seeds": [1, 2, 3],
  "expected_actual_detail_rows": 72,
  "canonical_12_kpis": [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio"
  ],
  "required_actual_detail_columns": [
    "run_id",
    "candidate_id",
    "condition_id",
    "seed",
    "reward_candidate_version",
    "execution_status",
    "actual_result",
    "trained_model",
    "checkpoint_path",
    "checkpoint_sha256",
    "source_mode",
    "adapter_mode",
    "canonical_eval_path",
    "reward_total_mean",
    "reward_total_std",
    "constraint_violation_count",
    "hard_constraint_passed",
    "cv_headway_mean",
    "avg_wait_seconds_mean",
    "bunching_rate_mean",
    "on_time_rate_mean",
    "intervention_rate_mean",
    "energy_proxy_mean",
    "passenger_demand_generated_mean",
    "passenger_served_count_mean",
    "passenger_service_rate_mean",
    "passenger_wait_p95_seconds_mean",
    "energy_proxy_per_passenger_mean",
    "fleet_reduction_ratio_mean",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "winner_eligible"
  ],
  "actual_result_required_upgrade_conditions": [
    "actual_result must be true only after a real ablation run has completed.",
    "trained_model must be true only for real trained checkpoints.",
    "checkpoint_path and checkpoint_sha256 must be non-empty for actual results.",
    "execution_status must be COMPLETED_ACTUAL for actual results.",
    "canonical_eval_path must point to a real canonical KPI output for actual results.",
    "hard_constraint_passed must be true for winner eligibility.",
    "paper_level_claim_allowed and causal_performance_claim_allowed remain false until later evidence gates."
  ],
  "guard_flags": {
    "actual_results": false,
    "winner_selected": false,
    "winner_eligible": false,
    "train_with_this_reward_allowed": false,
    "actual_training_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false,
    "best_reward_claim_allowed": false
  },
  "next_required_steps": [
    "Step125 reward ablation selection criteria gate",
    "Step126 reward ablation actual-result ingestion preflight",
    "Step127 trainable reward promotion decision package"
  ]
}
'@

$SpecJsonPath = ".\05_training\rewards\reward_ablation_actual_result_schema_bridge_step124.json"
$SpecJson | Set-Content -Path $SpecJsonPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 124 markdown
# ---------------------------------------------------------------------
$SpecMd = @'
# Step 124 — Reward Ablation Actual Result Schema Bridge

## Purpose

Step 124 defines the bridge between no-op/template reward ablation outputs and future actual reward ablation result ingestion.

This step does not run reward ablation, does not ingest actual results, and does not select a winner.

## Current Status

- `bridge_status = ACTUAL_RESULT_SCHEMA_BRIDGE_NOT_ACTUAL_RESULTS`
- `actual_results = false`
- `winner_selected = false`
- `train_with_this_reward_allowed = false`
- `actual_training_allowed = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`

## Expected Matrix

The future actual result table must cover:

```text
R0/R1/R2/R3/R4/R5
× A/A90/A80/A70
× seeds 1,2,3
= 72 rows
```

## Required Actual Result Meaning

A row may become an actual result only when all of the following are true:

1. The reward candidate was actually run.
2. The checkpoint is a real trained checkpoint.
3. The checkpoint path and SHA256 are recorded.
4. Canonical KPI outputs exist.
5. Hard-constraint results are recorded.
6. The result is not from no-op, template, mock, smoke-only, or dry-run output.

## Important Guard

This bridge only prepares the shape of future actual result files.

It must not promote any reward candidate and must not allow training.

## Next Step

Step 125 should define winner selection criteria without selecting a winner.
'@

$SpecMdPath = ".\05_training\rewards\reward_ablation_actual_result_schema_bridge_step124.md"
$SpecMd | Set-Content -Path $SpecMdPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 124 materializer
# ---------------------------------------------------------------------
$MaterializerPy = @'
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stable_run_id(candidate_id: str, condition_id: str, seed: int) -> str:
    raw = f"{candidate_id}|{condition_id}|{seed}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"step124_{candidate_id}_{condition_id}_seed{int(seed):03d}_{h}"


def materialize(spec: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    candidate_ids = list(spec["candidate_ids"])
    condition_ids = list(spec["condition_ids"])
    seeds = [int(x) for x in spec["seeds"]]
    fieldnames = list(spec["required_actual_detail_columns"])

    detail_rows: List[Dict[str, Any]] = []
    for candidate_id in candidate_ids:
        for condition_id in condition_ids:
            for seed in seeds:
                row = {k: "" for k in fieldnames}
                row.update({
                    "run_id": stable_run_id(candidate_id, condition_id, seed),
                    "candidate_id": candidate_id,
                    "condition_id": condition_id,
                    "seed": int(seed),
                    "reward_candidate_version": "step115_candidate_matrix_v1",
                    "execution_status": "WAITING_FOR_ACTUAL_RESULTS",
                    "actual_result": False,
                    "trained_model": False,
                    "checkpoint_path": "",
                    "checkpoint_sha256": "",
                    "source_mode": "not_executed",
                    "adapter_mode": "not_executed",
                    "canonical_eval_path": "",
                    "reward_total_mean": "",
                    "reward_total_std": "",
                    "constraint_violation_count": "",
                    "hard_constraint_passed": False,
                    "paper_level_claim_allowed": False,
                    "causal_performance_claim_allowed": False,
                    "winner_eligible": False,
                })
                for kpi in spec["canonical_12_kpis"]:
                    row[f"{kpi}_mean"] = ""
                detail_rows.append(row)

    summary_fieldnames = [
        "candidate_id",
        "planned_condition_count",
        "planned_seed_count",
        "planned_run_count",
        "actual_result_count",
        "winner_eligible_count",
        "winner_selected",
        "train_with_this_reward_allowed",
    ]
    summary_rows = []
    for candidate_id in candidate_ids:
        summary_rows.append({
            "candidate_id": candidate_id,
            "planned_condition_count": len(condition_ids),
            "planned_seed_count": len(seeds),
            "planned_run_count": len(condition_ids) * len(seeds),
            "actual_result_count": 0,
            "winner_eligible_count": 0,
            "winner_selected": False,
            "train_with_this_reward_allowed": False,
        })

    output_root.mkdir(parents=True, exist_ok=True)
    detail_path = output_root / "reward_ablation_actual_result_detail_template_step124.csv"
    summary_path = output_root / "reward_ablation_actual_result_summary_template_step124.csv"
    selection_path = output_root / "reward_ablation_actual_result_selection_input_template_step124.json"
    manifest_path = output_root / "reward_ablation_actual_result_schema_bridge_manifest_step124.json"

    write_csv(detail_path, detail_rows, fieldnames)
    write_csv(summary_path, summary_rows, summary_fieldnames)

    selection_input = {
        "artifact_version": "reward_ablation_actual_result_selection_input_template_step124_v1",
        "created_at_utc": utc_now(),
        "actual_results": False,
        "winner_selected": False,
        "winner_eligible_candidates": [],
        "selection_ready": False,
        "reason": "Template only. No actual ablation result has been ingested.",
    }
    dump_json(selection_path, selection_input)

    manifest = {
        "artifact_version": "reward_ablation_actual_result_schema_bridge_manifest_step124_v1",
        "created_at_utc": utc_now(),
        "bridge_status": spec["bridge_status"],
        "candidate_count": len(candidate_ids),
        "condition_count": len(condition_ids),
        "seed_count": len(seeds),
        "detail_rows": len(detail_rows),
        "expected_actual_detail_rows": int(spec["expected_actual_detail_rows"]),
        "actual_results": False,
        "winner_selected": False,
        "winner_eligible": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "output_root": str(output_root),
        "output_files": {
            "actual_detail_template_csv": str(detail_path),
            "actual_summary_template_csv": str(summary_path),
            "selection_input_template_json": str(selection_path),
            "manifest": str(manifest_path),
        },
        "source_spec_snapshot": spec,
    }
    dump_json(manifest_path, manifest)
    return manifest


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="05_training/rewards/reward_ablation_actual_result_schema_bridge_step124.json")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_actual_result_schema_bridge_step124")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    spec_path = resolve_path(project_root, args.spec)
    output_root = resolve_path(project_root, args.output_root)

    spec = load_json_any_encoding(spec_path)
    manifest = materialize(spec, output_root)

    print("[OK] Step 124 reward ablation actual result schema bridge materialized")
    print(f"[OK] bridge_status : {manifest['bridge_status']}")
    print(f"[OK] detail_rows   : {manifest['detail_rows']}")
    print(f"[OK] actual_results: {manifest['actual_results']}")
    print(f"[OK] winner_selected: {manifest['winner_selected']}")
    print(f"[OK] train_allowed : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] output_root   : {output_root}")
    print(f"[OK] manifest      : {manifest['output_files']['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@

$MaterializerPath = ".\05_training\rewards\reward_ablation_actual_result_schema_bridge_step124.py"
$MaterializerPy | Set-Content -Path $MaterializerPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 124 validator
# ---------------------------------------------------------------------
$ValidatorPy = @'
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_SEEDS = [1, 2, 3]
EXPECTED_ROWS = 72


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


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


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


def validate_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(checks, "bridge_status", "ACTUAL_RESULT_SCHEMA_BRIDGE_NOT_ACTUAL_RESULTS", manifest.get("bridge_status"), "Bridge must remain not-actual-results.")
    add_check(checks, "detail_rows", EXPECTED_ROWS, int(manifest.get("detail_rows", -1)), "Detail template must have 72 rows.")
    add_check(checks, "expected_rows", EXPECTED_ROWS, int(manifest.get("expected_actual_detail_rows", -1)), "Expected row count must be 72.")

    for key in [
        "actual_results",
        "winner_selected",
        "winner_eligible",
        "train_with_this_reward_allowed",
        "actual_training_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "best_reward_claim_allowed",
    ]:
        add_check(checks, f"guard_{key}", False, bool(manifest.get(key)), f"{key} must remain false.")

    output_files = manifest.get("output_files", {})
    detail_path = Path(output_files.get("actual_detail_template_csv", ""))
    summary_path = Path(output_files.get("actual_summary_template_csv", ""))
    selection_path = Path(output_files.get("selection_input_template_json", ""))

    add_check(checks, "detail_file_exists", True, detail_path.exists(), "Detail CSV must exist.")
    add_check(checks, "summary_file_exists", True, summary_path.exists(), "Summary CSV must exist.")
    add_check(checks, "selection_file_exists", True, selection_path.exists(), "Selection input JSON must exist.")

    detail_rows: List[Dict[str, Any]] = []
    if detail_path.exists():
        detail_rows = read_csv(detail_path)

    add_check(checks, "detail_csv_row_count", EXPECTED_ROWS, len(detail_rows), "Detail CSV must contain 72 rows.")

    candidates = sorted({r.get("candidate_id") for r in detail_rows})
    conditions = sorted({r.get("condition_id") for r in detail_rows})
    seeds = sorted({int(r.get("seed")) for r in detail_rows if str(r.get("seed")).strip()})

    add_check(checks, "candidate_set", sorted(EXPECTED_CANDIDATES), candidates, "Candidate set must be R0-R5.")
    add_check(checks, "condition_set", sorted(EXPECTED_CONDITIONS), conditions, "Condition set must be A/A90/A80/A70.")
    add_check(checks, "seed_set", EXPECTED_SEEDS, seeds, "Seed set must be 1/2/3.")

    combos = {
        (r.get("candidate_id"), r.get("condition_id"), int(r.get("seed")))
        for r in detail_rows
        if str(r.get("seed")).strip()
    }
    expected_combos = {
        (c, cond, s)
        for c in EXPECTED_CANDIDATES
        for cond in EXPECTED_CONDITIONS
        for s in EXPECTED_SEEDS
    }
    add_check(checks, "combination_completeness", len(expected_combos), len(combos), "All candidate/condition/seed combos must exist exactly once.")
    add_check(checks, "combination_set_exact", True, combos == expected_combos, "Combination set must exactly match expected matrix.")

    required_cols = set(manifest.get("source_spec_snapshot", {}).get("required_actual_detail_columns", []))
    if detail_rows:
        actual_cols = set(detail_rows[0].keys())
        missing_cols = sorted(required_cols - actual_cols)
    else:
        missing_cols = sorted(required_cols)
    add_check(checks, "required_detail_columns_present", [], missing_cols, "Required actual detail columns must be present.")

    bad_actual_rows = [
        r for r in detail_rows
        if as_bool(r.get("actual_result")) or as_bool(r.get("trained_model")) or as_bool(r.get("winner_eligible"))
    ]
    add_check(checks, "no_actual_result_rows", 0, len(bad_actual_rows), "Template bridge must not contain actual result rows.")

    bad_status_rows = [
        r for r in detail_rows
        if str(r.get("execution_status")) != "WAITING_FOR_ACTUAL_RESULTS"
    ]
    add_check(checks, "waiting_status_only", 0, len(bad_status_rows), "All template rows must wait for actual results.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_reward_ablation_actual_result_schema_bridge_step124_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": (
            "PASS_ACTUAL_RESULT_SCHEMA_BRIDGE_TEMPLATE_ONLY"
            if audit_status == "PASS"
            else "FAIL_ACTUAL_RESULT_SCHEMA_BRIDGE"
        ),
        "next_status": (
            "READY_FOR_STEP125_REWARD_ABLATION_SELECTION_CRITERIA_GATE"
            if audit_status == "PASS"
            else "BLOCKED_FIX_STEP124_BRIDGE"
        ),
        "detail_rows": len(detail_rows),
        "actual_results": False,
        "winner_selected": False,
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_actual_result_schema_bridge_step124_validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    manifest_path = resolve_path(project_root, args.manifest)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_json_any_encoding(manifest_path)
    report = validate_manifest(manifest)
    report["manifest_path"] = str(manifest_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "reward_ablation_actual_result_schema_bridge_step124_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 124 reward ablation actual result schema bridge validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] detail_rows   : {report['detail_rows']}")
    print(f"[OK] actual_results: {report['actual_results']}")
    print(f"[OK] winner_selected: {report['winner_selected']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'@

$ValidatorPath = ".\05_training\rewards\validate_reward_ablation_actual_result_schema_bridge_step124.py"
$ValidatorPy | Set-Content -Path $ValidatorPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 124 self-test
# ---------------------------------------------------------------------
$TestPy = @'
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


def write_csv(path: Path, rows, fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def unique_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    materializer = project_root / "05_training" / "rewards" / "reward_ablation_actual_result_schema_bridge_step124.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_actual_result_schema_bridge_step124.py"
    spec = project_root / "05_training" / "rewards" / "reward_ablation_actual_result_schema_bridge_step124.json"

    suffix = unique_suffix()
    output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_actual_result_schema_bridge_step124_selftest_{suffix}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_actual_result_schema_bridge_step124_validation_{suffix}"

    run_cmd = [
        sys.executable,
        str(materializer),
        "--spec",
        str(spec),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(run_cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 124 materializer failed")

    manifest_path = output_root / "reward_ablation_actual_result_schema_bridge_manifest_step124.json"
    if not manifest_path.exists():
        raise SystemExit("[FAIL] Step 124 manifest missing")

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
        raise SystemExit("[FAIL] Step 124 validator failed")

    report = load_json(validation_root / "reward_ablation_actual_result_schema_bridge_step124_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_ACTUAL_RESULT_SCHEMA_BRIDGE_TEMPLATE_ONLY",
        "next_status": "READY_FOR_STEP125_REWARD_ABLATION_SELECTION_CRITERIA_GATE",
        "detail_rows": 72,
        "actual_results": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Negative test: setting one row to actual_result=true must fail.
    manifest = load_json(manifest_path)
    detail_path = Path(manifest["output_files"]["actual_detail_template_csv"])
    with open(detail_path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    rows[0]["execution_status"] = "COMPLETED_ACTUAL"
    rows[0]["actual_result"] = "True"
    rows[0]["trained_model"] = "True"
    write_csv(detail_path, rows, fieldnames)

    bad_root = project_root / "artifacts" / "rewards" / f"reward_ablation_actual_result_schema_bridge_step124_bad_validation_{suffix}"
    bad_val_cmd = [
        sys.executable,
        str(validator),
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(bad_root),
    ]
    bad_val = subprocess.run(bad_val_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_val.returncode == 0:
        print(bad_val.stdout)
        raise SystemExit("[FAIL] bad actual_result row unexpectedly passed")

    bad_report = load_json(bad_root / "reward_ablation_actual_result_schema_bridge_step124_validation_report.json")
    failed_ids = {
        c["check_id"]
        for c in bad_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "no_actual_result_rows" not in failed_ids:
        raise SystemExit("[FAIL] actual_result guard did not fail")

    print("[OK] Step 124 reward ablation actual result schema bridge self-test PASS")
    print("[DONE] Step 124 reward ablation actual result schema bridge complete.")


if __name__ == "__main__":
    main()
'@

$TestPath = ".\05_training\rewards\test_validate_reward_ablation_actual_result_schema_bridge_step124.py"
$TestPy | Set-Content -Path $TestPath -Encoding UTF8

# ---------------------------------------------------------------------
# Run Step 124 self-test
# ---------------------------------------------------------------------
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 124 reward ablation actual result schema bridge self-test failed"
}

Write-Host ""
Write-Host "[DONE] Step 124 reward ablation actual result schema bridge files created and tested."
Write-Host "[NEXT] Review git status, then commit the Step 124 actual result schema bridge files."
