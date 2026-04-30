$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) { throw "[STOP] Project root not found: $ProjectRoot" }
Set-Location $ProjectRoot

$RewardsDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

$SpecJson = @'
{
  "artifact_version": "reward_ablation_actual_result_ingestion_preflight_step126_v1",
  "step": 126,
  "preflight_status": "ACTUAL_RESULT_INGESTION_PREFLIGHT_DEFINED_NOT_INGESTED",
  "purpose": "Define preflight rules for future actual reward ablation result bundles before ingestion.",
  "candidate_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
  "condition_ids": ["A", "A90", "A80", "A70"],
  "seeds": [1, 2, 3],
  "expected_actual_result_rows": 72,
  "required_bundle_files": [
    "actual_result_detail_csv",
    "actual_result_summary_csv",
    "hard_constraint_violations_csv",
    "canonical_eval_manifest_json",
    "checkpoint_validation_manifest_json",
    "bundle_manifest_json"
  ],
  "required_detail_columns": [
    "run_id", "candidate_id", "condition_id", "seed", "execution_status",
    "actual_result", "trained_model", "checkpoint_path", "checkpoint_sha256",
    "canonical_eval_path", "hard_constraint_passed", "constraint_violation_count",
    "reward_total_mean", "reward_total_std", "passenger_service_rate_mean",
    "avg_wait_seconds_mean", "passenger_wait_p95_seconds_mean",
    "energy_proxy_per_passenger_mean", "fleet_reduction_ratio_mean",
    "winner_eligible", "paper_level_claim_allowed", "causal_performance_claim_allowed"
  ],
  "required_bundle_manifest_flags": {
    "actual_results": true,
    "template_only": false,
    "noop_only": false,
    "dry_run_only": false,
    "smoke_only": false,
    "winner_selected": false,
    "train_with_this_reward_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false
  },
  "ingestion_blockers": [
    "missing_required_bundle_file",
    "missing_required_detail_column",
    "row_count_not_72",
    "missing_candidate_condition_seed_combination",
    "actual_result_false",
    "trained_model_false",
    "checkpoint_path_empty",
    "checkpoint_sha256_empty",
    "canonical_eval_path_empty",
    "template_or_noop_or_dry_run_or_smoke_flag_true",
    "winner_selected_true_before_selection_gate",
    "paper_or_causal_claim_true_before_later_evidence_gate"
  ],
  "guard_flags": {
    "ingestion_allowed_now": false,
    "actual_results_ingested": false,
    "winner_selected": false,
    "train_with_this_reward_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false,
    "best_reward_claim_allowed": false
  },
  "next_status": "READY_FOR_STEP127_TRAINABLE_REWARD_PROMOTION_DECISION_PACKAGE"
}
'@
$SpecJson | Set-Content -Path ".\05_training\rewards\reward_ablation_actual_result_ingestion_preflight_step126.json" -Encoding UTF8

$SpecMd = @'
# Step 126 — Reward Ablation Actual Result Ingestion Preflight

Step 126 defines the preflight rules for future actual reward ablation result bundles.

This step does not ingest actual results, does not select a winner, and does not allow training.

Required future matrix:

```text
R0/R1/R2/R3/R4/R5 × A/A90/A80/A70 × seeds 1,2,3 = 72 rows
```

The future actual bundle must contain actual result detail CSV, summary CSV, hard-constraint violations CSV, canonical evaluation manifest, checkpoint validation manifest, and bundle manifest.

Current guards remain:

- `ingestion_allowed_now = false`
- `actual_results_ingested = false`
- `winner_selected = false`
- `train_with_this_reward_allowed = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`
'@
$SpecMd | Set-Content -Path ".\05_training\rewards\reward_ablation_actual_result_ingestion_preflight_step126.md" -Encoding UTF8

$MaterializerPy = @'
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
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


def resolve(project_root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else project_root / p


def materialize(spec: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    checklist: List[Dict[str, Any]] = []
    for name in spec["required_bundle_files"]:
        checklist.append({
            "check_id": f"required_file_{name}",
            "category": "bundle_file",
            "required_item": name,
            "current_status": "not_ingested",
            "passed": False,
        })
    for name in spec["required_detail_columns"]:
        checklist.append({
            "check_id": f"required_column_{name}",
            "category": "detail_column",
            "required_item": name,
            "current_status": "not_ingested",
            "passed": False,
        })

    output_root.mkdir(parents=True, exist_ok=True)
    checklist_path = output_root / "actual_result_ingestion_preflight_checklist_step126.csv"
    bundle_template_path = output_root / "actual_result_bundle_manifest_template_step126.json"
    manifest_path = output_root / "reward_ablation_actual_result_ingestion_preflight_manifest_step126.json"

    write_csv(checklist_path, checklist, ["check_id", "category", "required_item", "current_status", "passed"])

    bundle_template = {
        "artifact_version": "actual_result_bundle_manifest_template_step126_v1",
        "created_at_utc": utc_now(),
        "actual_results": True,
        "template_only": False,
        "noop_only": False,
        "dry_run_only": False,
        "smoke_only": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "required_row_count": int(spec["expected_actual_result_rows"]),
        "bundle_files": {name: "" for name in spec["required_bundle_files"]},
        "note": "Template only. It describes a future actual result bundle.",
    }
    dump_json(bundle_template_path, bundle_template)

    manifest = {
        "artifact_version": "reward_ablation_actual_result_ingestion_preflight_manifest_step126_v1",
        "created_at_utc": utc_now(),
        "preflight_status": spec["preflight_status"],
        "candidate_count": len(spec["candidate_ids"]),
        "condition_count": len(spec["condition_ids"]),
        "seed_count": len(spec["seeds"]),
        "expected_actual_result_rows": int(spec["expected_actual_result_rows"]),
        "required_bundle_file_count": len(spec["required_bundle_files"]),
        "required_detail_column_count": len(spec["required_detail_columns"]),
        "checklist_rows": len(checklist),
        "ingestion_allowed_now": False,
        "actual_results_ingested": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "output_root": str(output_root),
        "output_files": {
            "preflight_checklist_csv": str(checklist_path),
            "actual_result_bundle_manifest_template_json": str(bundle_template_path),
            "manifest": str(manifest_path),
        },
        "source_spec_snapshot": spec,
    }
    dump_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="05_training/rewards/reward_ablation_actual_result_ingestion_preflight_step126.json")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_actual_result_ingestion_preflight_step126")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    spec = load_json(resolve(project_root, args.spec))
    output_root = resolve(project_root, args.output_root)
    manifest = materialize(spec, output_root)

    print("[OK] Step 126 reward ablation actual result ingestion preflight materialized")
    print(f"[OK] preflight_status: {manifest['preflight_status']}")
    print(f"[OK] expected_rows   : {manifest['expected_actual_result_rows']}")
    print(f"[OK] checklist_rows  : {manifest['checklist_rows']}")
    print(f"[OK] ingestion_allowed: {manifest['ingestion_allowed_now']}")
    print(f"[OK] actual_ingested : {manifest['actual_results_ingested']}")
    print(f"[OK] winner_selected: {manifest['winner_selected']}")
    print(f"[OK] train_allowed  : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] manifest       : {manifest['output_files']['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@
$MaterializerPy | Set-Content -Path ".\05_training\rewards\reward_ablation_actual_result_ingestion_preflight_step126.py" -Encoding UTF8

$ValidatorPy = @'
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
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


def resolve(project_root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else project_root / p


def validate_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    spec = manifest.get("source_spec_snapshot", {})

    add_check(checks, "preflight_status", "ACTUAL_RESULT_INGESTION_PREFLIGHT_DEFINED_NOT_INGESTED", manifest.get("preflight_status"), "Preflight must be defined but not ingested.")
    add_check(checks, "candidate_count", 6, int(manifest.get("candidate_count", -1)), "Candidate count must be 6.")
    add_check(checks, "condition_count", 4, int(manifest.get("condition_count", -1)), "Condition count must be 4.")
    add_check(checks, "seed_count", 3, int(manifest.get("seed_count", -1)), "Seed count must be 3.")
    add_check(checks, "expected_actual_rows", 72, int(manifest.get("expected_actual_result_rows", -1)), "Future actual result rows must be 72.")

    false_guards = [
        "ingestion_allowed_now",
        "actual_results_ingested",
        "winner_selected",
        "train_with_this_reward_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
        "best_reward_claim_allowed",
    ]
    for key in false_guards:
        add_check(checks, f"manifest_guard_{key}", False, bool(manifest.get(key)), f"{key} must remain false.")
        add_check(checks, f"spec_guard_{key}", False, bool(spec.get("guard_flags", {}).get(key)), f"{key} must remain false in spec.")

    required_files = spec.get("required_bundle_files", [])
    required_cols = spec.get("required_detail_columns", [])
    add_check(checks, "required_bundle_file_count", 6, len(required_files), "Six required bundle files must be listed.")
    add_check(checks, "required_detail_column_count_min", True, len(required_cols) >= 20, "Required detail columns must be complete enough.")

    output_files = manifest.get("output_files", {})
    checklist_path = Path(output_files.get("preflight_checklist_csv", ""))
    bundle_template_path = Path(output_files.get("actual_result_bundle_manifest_template_json", ""))
    add_check(checks, "checklist_exists", True, checklist_path.exists(), "Checklist CSV must exist.")
    add_check(checks, "bundle_template_exists", True, bundle_template_path.exists(), "Bundle template JSON must exist.")

    checklist_rows = read_csv(checklist_path) if checklist_path.exists() else []
    add_check(checks, "checklist_row_count", len(required_files) + len(required_cols), len(checklist_rows), "Checklist must cover required files and columns.")

    bundle_template = load_json(bundle_template_path) if bundle_template_path.exists() else {}
    add_check(checks, "bundle_template_actual_results", True, bool(bundle_template.get("actual_results")), "Future bundle template must require actual results.")
    for key in ["template_only", "noop_only", "dry_run_only", "smoke_only", "winner_selected", "train_with_this_reward_allowed", "paper_level_claim_allowed", "causal_performance_claim_allowed"]:
        add_check(checks, f"bundle_template_flag_{key}", False, bool(bundle_template.get(key)), f"{key} must be false in bundle template.")

    blockers = spec.get("ingestion_blockers", [])
    add_check(checks, "blocker_count_min", True, len(blockers) >= 10, "Ingestion blockers should be explicit.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_reward_ablation_actual_result_ingestion_preflight_step126_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": "PASS_ACTUAL_RESULT_INGESTION_PREFLIGHT_DEFINED_NOT_INGESTED" if audit_status == "PASS" else "FAIL_ACTUAL_RESULT_INGESTION_PREFLIGHT",
        "next_status": "READY_FOR_STEP127_TRAINABLE_REWARD_PROMOTION_DECISION_PACKAGE" if audit_status == "PASS" else "BLOCKED_FIX_STEP126_PREFLIGHT",
        "expected_actual_result_rows": int(manifest.get("expected_actual_result_rows", 0)),
        "ingestion_allowed_now": False,
        "actual_results_ingested": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "failure_count": len(failures),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_actual_result_ingestion_preflight_step126_validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    manifest_path = resolve(project_root, args.manifest)
    output_root = resolve(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    report = validate_manifest(load_json(manifest_path))
    report["manifest_path"] = str(manifest_path)
    report["output_root"] = str(output_root)
    report_path = output_root / "reward_ablation_actual_result_ingestion_preflight_step126_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 126 reward ablation actual result ingestion preflight validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] expected_rows : {report['expected_actual_result_rows']}")
    print(f"[OK] ingestion_allowed: {report['ingestion_allowed_now']}")
    print(f"[OK] actual_ingested: {report['actual_results_ingested']}")
    print(f"[OK] winner_selected: {report['winner_selected']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")
    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'@
$ValidatorPy | Set-Content -Path ".\05_training\rewards\validate_reward_ablation_actual_result_ingestion_preflight_step126.py" -Encoding UTF8

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
    materializer = project_root / "05_training" / "rewards" / "reward_ablation_actual_result_ingestion_preflight_step126.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_actual_result_ingestion_preflight_step126.py"
    spec = project_root / "05_training" / "rewards" / "reward_ablation_actual_result_ingestion_preflight_step126.json"

    suffix = unique_suffix()
    output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_actual_result_ingestion_preflight_step126_selftest_{suffix}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_actual_result_ingestion_preflight_step126_validation_{suffix}"

    run_cmd = [sys.executable, str(materializer), "--spec", str(spec), "--output-root", str(output_root)]
    result = subprocess.run(run_cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 126 materializer failed")

    manifest_path = output_root / "reward_ablation_actual_result_ingestion_preflight_manifest_step126.json"
    if not manifest_path.exists():
        raise SystemExit("[FAIL] Step 126 manifest missing")

    val_cmd = [sys.executable, str(validator), "--manifest", str(manifest_path), "--output-root", str(validation_root)]
    val = subprocess.run(val_cmd, cwd=project_root, text=True, capture_output=True)
    print(val.stdout)
    if val.returncode != 0:
        print(val.stderr)
        raise SystemExit("[FAIL] Step 126 validator failed")

    report = load_json(validation_root / "reward_ablation_actual_result_ingestion_preflight_step126_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_ACTUAL_RESULT_INGESTION_PREFLIGHT_DEFINED_NOT_INGESTED",
        "next_status": "READY_FOR_STEP127_TRAINABLE_REWARD_PROMOTION_DECISION_PACKAGE",
        "expected_actual_result_rows": 72,
        "ingestion_allowed_now": False,
        "actual_results_ingested": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    bad_manifest = load_json(manifest_path)
    bad_manifest["winner_selected"] = True
    bad_path = output_root / "bad_winner_selected_preflight_manifest_step126.json"
    dump_json(bad_path, bad_manifest)

    bad_root = project_root / "artifacts" / "rewards" / f"reward_ablation_actual_result_ingestion_preflight_step126_bad_validation_{suffix}"
    bad_cmd = [sys.executable, str(validator), "--manifest", str(bad_path), "--output-root", str(bad_root)]
    bad = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad.returncode == 0:
        print(bad.stdout)
        raise SystemExit("[FAIL] bad winner_selected manifest unexpectedly passed")

    bad_report = load_json(bad_root / "reward_ablation_actual_result_ingestion_preflight_step126_validation_report.json")
    failed_ids = {c["check_id"] for c in bad_report.get("checks", []) if not c.get("passed", False)}
    if "manifest_guard_winner_selected" not in failed_ids:
        raise SystemExit("[FAIL] winner_selected guard did not fail")

    print("[OK] Step 126 reward ablation actual result ingestion preflight self-test PASS")
    print("[DONE] Step 126 reward ablation actual result ingestion preflight complete.")


if __name__ == "__main__":
    main()
'@
$TestPy | Set-Content -Path ".\05_training\rewards\test_validate_reward_ablation_actual_result_ingestion_preflight_step126.py" -Encoding UTF8

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py ".\05_training\rewards\test_validate_reward_ablation_actual_result_ingestion_preflight_step126.py"

if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 126 reward ablation actual result ingestion preflight self-test failed" }

Write-Host ""
Write-Host "[DONE] Step 126 reward ablation actual result ingestion preflight files created and tested."
Write-Host "[NEXT] Review git status, then commit the Step 126 ingestion preflight files."
