$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

# ---------------------------------------------------------------------
# Step 127 decision package JSON
# ---------------------------------------------------------------------
$SpecJson = @'
{
  "artifact_version": "trainable_reward_promotion_decision_package_step127_v1",
  "step": 127,
  "decision_status": "PROMOTION_DECISION_PACKAGE_CREATED_NOT_PROMOTED",
  "purpose": "Package all reward-design gates into a trainable reward promotion decision record without promoting any reward candidate.",
  "source_steps": [
    "Step111 final reward specification draft",
    "Step112 reward candidate protocol",
    "Step113 reward normalization baseline lock",
    "Step114 hard constraint review",
    "Step115 reward ablation matrix",
    "Step116 trainable reward promotion gate",
    "Step117 reward ablation result schema",
    "Step118 result writer guard",
    "Step119 dry-run plan",
    "Step120 execution preflight",
    "Step121 execution manifest",
    "Step122 sandbox no-op runner guard",
    "Step123 no-op result ingestion guard",
    "Step124 actual result schema bridge",
    "Step125 selection criteria gate",
    "Step126 actual result ingestion preflight"
  ],
  "candidate_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
  "condition_ids": ["A", "A90", "A80", "A70"],
  "seeds": [1, 2, 3],
  "required_actual_result_rows": 72,
  "decision_inputs_required_before_promotion": [
    "actual_ablation_result_bundle_passed_step126",
    "all_72_candidate_condition_seed_rows_present",
    "actual_result_true_for_all_rows",
    "trained_model_true_for_all_rows",
    "checkpoint_hashes_present",
    "canonical_kpi_outputs_present",
    "hard_constraints_passed_for_selected_candidate",
    "selection_criteria_step125_passed",
    "no_template_noop_dryrun_smoke_rows",
    "paper_and_causal_claims_still_separately_gated"
  ],
  "current_evidence_status": {
    "actual_ablation_result_bundle_present": false,
    "actual_results_ingested": false,
    "all_72_rows_present": false,
    "winner_selected": false,
    "selected_candidate_id": null,
    "hard_constraints_verified_on_actual_results": false,
    "promotion_evidence_complete": false
  },
  "decision": {
    "trainable_reward_promoted": false,
    "selected_candidate_id": null,
    "promotion_allowed_now": false,
    "promotion_block_reason": "No actual reward ablation results have been ingested. Current pipeline is limited to templates, no-op guards, dry-run plans, and preflight gates.",
    "train_with_this_reward_allowed": false,
    "final_reward_design_claim_allowed": false,
    "best_reward_claim_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false
  },
  "guard_flags": {
    "actual_results": false,
    "actual_results_ingested": false,
    "winner_selected": false,
    "trainable_reward_promoted": false,
    "train_with_this_reward_allowed": false,
    "actual_training_allowed": false,
    "final_reward_design_claim_allowed": false,
    "best_reward_claim_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false
  },
  "next_status": "READY_FOR_STEP128_REWARD_PIPELINE_DOCUMENTATION_OR_ACTUAL_ABLATION_DATA_WAIT"
}
'@

$SpecJsonPath = ".\05_training\rewards\trainable_reward_promotion_decision_package_step127.json"
$SpecJson | Set-Content -Path $SpecJsonPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 127 markdown
# ---------------------------------------------------------------------
$SpecMd = @'
# Step 127 — Trainable Reward Promotion Decision Package

## Purpose

Step 127 packages the reward-design gates into a formal decision record.

This step does not promote a reward candidate.

## Current Decision

```text
trainable_reward_promoted = false
selected_candidate_id = null
train_with_this_reward_allowed = false
```

## Reason

No actual reward ablation results have been ingested.

The current pipeline has produced templates, no-op guards, dry-run plans, execution manifests, and actual-result preflight gates. These are necessary safeguards, but they are not evidence that any reward candidate is better.

## Promotion Requires

A future promotion decision requires:

1. actual ablation result bundle
2. all 72 R0~R5 × A/A90/A80/A70 × seed rows
3. actual trained checkpoint evidence
4. canonical KPI outputs
5. hard-constraint verification
6. Step 125 selection criteria pass
7. no template/no-op/dry-run/smoke rows
8. separate paper-level and causal-claim gates

## Guard

Even after this package exists, training remains blocked until a later decision package has actual evidence.
'@

$SpecMdPath = ".\05_training\rewards\trainable_reward_promotion_decision_package_step127.md"
$SpecMd | Set-Content -Path $SpecMdPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 127 materializer
# ---------------------------------------------------------------------
$MaterializerPy = @'
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


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


def materialize(spec: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    decision_path = output_root / "trainable_reward_promotion_decision_step127.json"
    evidence_path = output_root / "trainable_reward_promotion_missing_evidence_step127.json"
    manifest_path = output_root / "trainable_reward_promotion_decision_package_manifest_step127.json"

    decision = {
        "artifact_version": "trainable_reward_promotion_decision_step127_v1",
        "created_at_utc": utc_now(),
        "decision_status": spec["decision_status"],
        "decision": spec["decision"],
        "current_evidence_status": spec["current_evidence_status"],
        "guard_flags": spec["guard_flags"],
    }
    dump_json(decision_path, decision)

    missing_evidence = {
        "artifact_version": "trainable_reward_promotion_missing_evidence_step127_v1",
        "created_at_utc": utc_now(),
        "promotion_evidence_complete": False,
        "missing_items": list(spec["decision_inputs_required_before_promotion"]),
        "reason": spec["decision"]["promotion_block_reason"],
    }
    dump_json(evidence_path, missing_evidence)

    manifest = {
        "artifact_version": "trainable_reward_promotion_decision_package_manifest_step127_v1",
        "created_at_utc": utc_now(),
        "decision_status": spec["decision_status"],
        "candidate_count": len(spec["candidate_ids"]),
        "condition_count": len(spec["condition_ids"]),
        "seed_count": len(spec["seeds"]),
        "required_actual_result_rows": int(spec["required_actual_result_rows"]),
        "actual_results": False,
        "actual_results_ingested": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "selected_candidate_id": None,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "final_reward_design_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "output_root": str(output_root),
        "output_files": {
            "decision_json": str(decision_path),
            "missing_evidence_json": str(evidence_path),
            "manifest": str(manifest_path)
        },
        "source_spec_snapshot": spec
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
    parser.add_argument("--spec", default="05_training/rewards/trainable_reward_promotion_decision_package_step127.json")
    parser.add_argument("--output-root", default="artifacts/rewards/trainable_reward_promotion_decision_package_step127")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    spec_path = resolve_path(project_root, args.spec)
    output_root = resolve_path(project_root, args.output_root)

    spec = load_json_any_encoding(spec_path)
    manifest = materialize(spec, output_root)

    print("[OK] Step 127 trainable reward promotion decision package materialized")
    print(f"[OK] decision_status: {manifest['decision_status']}")
    print(f"[OK] promoted       : {manifest['trainable_reward_promoted']}")
    print(f"[OK] selected       : {manifest['selected_candidate_id']}")
    print(f"[OK] train_allowed  : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] actual_results : {manifest['actual_results']}")
    print(f"[OK] manifest       : {manifest['output_files']['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@

$MaterializerPath = ".\05_training\rewards\trainable_reward_promotion_decision_package_step127.py"
$MaterializerPy | Set-Content -Path $MaterializerPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 127 validator
# ---------------------------------------------------------------------
$ValidatorPy = @'
from __future__ import annotations

import argparse
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


def validate_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    spec = manifest.get("source_spec_snapshot", {})
    decision = spec.get("decision", {})
    evidence = spec.get("current_evidence_status", {})
    guards = spec.get("guard_flags", {})

    add_check(checks, "decision_status", "PROMOTION_DECISION_PACKAGE_CREATED_NOT_PROMOTED", manifest.get("decision_status"), "Decision package must be not promoted.")
    add_check(checks, "candidate_count", 6, int(manifest.get("candidate_count", -1)), "Candidate count must be 6.")
    add_check(checks, "condition_count", 4, int(manifest.get("condition_count", -1)), "Condition count must be 4.")
    add_check(checks, "seed_count", 3, int(manifest.get("seed_count", -1)), "Seed count must be 3.")
    add_check(checks, "required_rows", 72, int(manifest.get("required_actual_result_rows", -1)), "Required future actual row count must be 72.")

    false_keys = [
        "actual_results",
        "actual_results_ingested",
        "winner_selected",
        "trainable_reward_promoted",
        "train_with_this_reward_allowed",
        "actual_training_allowed",
        "final_reward_design_claim_allowed",
        "best_reward_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed"
    ]

    for key in false_keys:
        add_check(checks, f"manifest_guard_{key}", False, bool(manifest.get(key)), f"{key} must remain false in manifest.")
        add_check(checks, f"spec_guard_{key}", False, bool(guards.get(key)), f"{key} must remain false in spec.")

    add_check(checks, "selected_candidate_none", None, manifest.get("selected_candidate_id"), "No candidate may be selected.")
    add_check(checks, "decision_trainable_promoted_false", False, bool(decision.get("trainable_reward_promoted")), "Decision must not promote trainable reward.")
    add_check(checks, "decision_selected_candidate_none", None, decision.get("selected_candidate_id"), "Decision selected candidate must be null.")
    add_check(checks, "decision_promotion_allowed_false", False, bool(decision.get("promotion_allowed_now")), "Promotion must not be allowed now.")

    for key in [
        "actual_ablation_result_bundle_present",
        "actual_results_ingested",
        "all_72_rows_present",
        "winner_selected",
        "hard_constraints_verified_on_actual_results",
        "promotion_evidence_complete"
    ]:
        add_check(checks, f"evidence_{key}", False, bool(evidence.get(key)), f"{key} must be false.")

    missing = spec.get("decision_inputs_required_before_promotion", [])
    add_check(checks, "missing_evidence_count_min", True, len(missing) >= 8, "Decision package must list missing evidence.")

    output_files = manifest.get("output_files", {})
    for label in ["decision_json", "missing_evidence_json", "manifest"]:
        path = Path(output_files.get(label, ""))
        add_check(checks, f"output_file_{label}", True, path.exists(), f"{label} must exist.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_trainable_reward_promotion_decision_package_step127_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": (
            "PASS_PROMOTION_DECISION_PACKAGE_NOT_PROMOTED"
            if audit_status == "PASS"
            else "FAIL_PROMOTION_DECISION_PACKAGE"
        ),
        "next_status": (
            "READY_FOR_STEP128_REWARD_PIPELINE_DOCUMENTATION_OR_ACTUAL_ABLATION_DATA_WAIT"
            if audit_status == "PASS"
            else "BLOCKED_FIX_STEP127_DECISION_PACKAGE"
        ),
        "trainable_reward_promoted": False,
        "selected_candidate_id": None,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="artifacts/rewards/trainable_reward_promotion_decision_package_step127_validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    manifest_path = resolve_path(project_root, args.manifest)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_json_any_encoding(manifest_path)
    report = validate_manifest(manifest)
    report["manifest_path"] = str(manifest_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "trainable_reward_promotion_decision_package_step127_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 127 trainable reward promotion decision package validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] promoted      : {report['trainable_reward_promoted']}")
    print(f"[OK] selected      : {report['selected_candidate_id']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] actual_results: {report['actual_results']}")
    print(f"[OK] winner_selected: {report['winner_selected']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'@

$ValidatorPath = ".\05_training\rewards\validate_trainable_reward_promotion_decision_package_step127.py"
$ValidatorPy | Set-Content -Path $ValidatorPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 127 self-test
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
    materializer = project_root / "05_training" / "rewards" / "trainable_reward_promotion_decision_package_step127.py"
    validator = project_root / "05_training" / "rewards" / "validate_trainable_reward_promotion_decision_package_step127.py"
    spec = project_root / "05_training" / "rewards" / "trainable_reward_promotion_decision_package_step127.json"

    suffix = unique_suffix()
    output_root = project_root / "artifacts" / "rewards" / f"trainable_reward_promotion_decision_package_step127_selftest_{suffix}"
    validation_root = project_root / "artifacts" / "rewards" / f"trainable_reward_promotion_decision_package_step127_validation_{suffix}"

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
        raise SystemExit("[FAIL] Step 127 materializer failed")

    manifest_path = output_root / "trainable_reward_promotion_decision_package_manifest_step127.json"
    if not manifest_path.exists():
        raise SystemExit("[FAIL] Step 127 manifest missing")

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
        raise SystemExit("[FAIL] Step 127 validator failed")

    report = load_json(validation_root / "trainable_reward_promotion_decision_package_step127_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_PROMOTION_DECISION_PACKAGE_NOT_PROMOTED",
        "next_status": "READY_FOR_STEP128_REWARD_PIPELINE_DOCUMENTATION_OR_ACTUAL_ABLATION_DATA_WAIT",
        "trainable_reward_promoted": False,
        "selected_candidate_id": None,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Negative test: promoted=true must fail.
    bad_manifest = load_json(manifest_path)
    bad_manifest["trainable_reward_promoted"] = True
    bad_manifest["source_spec_snapshot"]["guard_flags"]["trainable_reward_promoted"] = True
    bad_manifest["source_spec_snapshot"]["decision"]["trainable_reward_promoted"] = True
    bad_path = output_root / "bad_promoted_manifest_step127.json"
    dump_json(bad_path, bad_manifest)

    bad_root = project_root / "artifacts" / "rewards" / f"trainable_reward_promotion_decision_package_step127_bad_validation_{suffix}"
    bad_cmd = [
        sys.executable,
        str(validator),
        "--manifest",
        str(bad_path),
        "--output-root",
        str(bad_root),
    ]
    bad = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad.returncode == 0:
        print(bad.stdout)
        raise SystemExit("[FAIL] bad promoted manifest unexpectedly passed")

    bad_report = load_json(bad_root / "trainable_reward_promotion_decision_package_step127_validation_report.json")
    failed_ids = {
        c["check_id"]
        for c in bad_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "manifest_guard_trainable_reward_promoted" not in failed_ids:
        raise SystemExit("[FAIL] promoted guard did not fail")

    print("[OK] Step 127 trainable reward promotion decision package self-test PASS")
    print("[DONE] Step 127 trainable reward promotion decision package complete.")


if __name__ == "__main__":
    main()
'@

$TestPath = ".\05_training\rewards\test_validate_trainable_reward_promotion_decision_package_step127.py"
$TestPy | Set-Content -Path $TestPath -Encoding UTF8

# ---------------------------------------------------------------------
# Run Step 127 self-test
# ---------------------------------------------------------------------
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 127 trainable reward promotion decision package self-test failed"
}

Write-Host ""
Write-Host "[DONE] Step 127 trainable reward promotion decision package files created and tested."
Write-Host "[NEXT] Review git status, then commit the Step 127 decision package files."
