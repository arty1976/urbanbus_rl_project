$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) { throw "[STOP] Project root not found: $ProjectRoot" }
Set-Location $ProjectRoot

$RewardsDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

$SpecJson = @'
{
  "artifact_version": "reward_ablation_selection_criteria_gate_step125_v1",
  "step": 125,
  "gate_status": "SELECTION_CRITERIA_DEFINED_NO_WINNER_SELECTED",
  "purpose": "Define how a future reward ablation winner may be selected, without selecting any winner in this step.",
  "candidate_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
  "condition_ids": ["A", "A90", "A80", "A70"],
  "seeds": [1, 2, 3],
  "required_actual_result_rows": 72,
  "selection_readiness": {
    "actual_results_required": true,
    "noop_results_allowed_for_selection": false,
    "template_results_allowed_for_selection": false,
    "dry_run_results_allowed_for_selection": false,
    "winner_selected": false,
    "winner_selection_allowed_now": false,
    "train_with_selected_reward_allowed": false
  },
  "primary_selection_principles": {
    "service_quality_first": true,
    "hard_constraints_must_pass": true,
    "no_energy_saving_by_stranding_passengers": true,
    "long_wait_fairness_required": true,
    "fleet_reduction_bonus_only": true,
    "energy_fleet_secondary": true
  },
  "hard_constraint_requirements": [
    "passenger_service_rate_floor_passed",
    "avg_wait_regression_cap_passed",
    "p95_wait_regression_cap_passed",
    "qwen_off_for_A_family_passed",
    "no_actual_observation_overclaim_passed"
  ],
  "selection_tiers": [
    {"tier": 1, "name": "eligibility_filter", "description": "Only actual, complete, hard-constraint-passing results can be eligible."},
    {"tier": 2, "name": "service_quality_rank", "description": "Prioritize passenger service rate, wait, p95 wait, bunching, headway CV, and on-time rate."},
    {"tier": 3, "name": "efficiency_secondary_rank", "description": "Use energy per passenger and fleet reduction only as secondary tie-breakers."},
    {"tier": 4, "name": "stability_and_seed_robustness", "description": "Prefer lower seed variance and no condition-specific collapse."}
  ],
  "disqualifying_conditions": [
    "Any hard constraint failure",
    "Any missing candidate/condition/seed result row",
    "Any actual_result=false row",
    "Any trained_model=false row",
    "Any paper_level_claim_allowed=true before later evidence gate",
    "Any causal_performance_claim_allowed=true before later evidence gate",
    "Any reward candidate that improves energy by reducing passenger_service_rate below floor",
    "Any winner chosen from no-op, template, mock, smoke-only, or dry-run outputs"
  ],
  "scorecard_columns_required_for_future_selection": [
    "candidate_id", "eligible", "actual_result_count", "hard_constraint_failure_count",
    "passenger_service_rate_mean", "avg_wait_seconds_mean", "passenger_wait_p95_seconds_mean",
    "bunching_rate_mean", "cv_headway_mean", "on_time_rate_mean",
    "energy_proxy_per_passenger_mean", "fleet_reduction_ratio_mean",
    "seed_variance_summary", "selection_rank", "selection_reason", "winner_selected"
  ],
  "guard_flags": {
    "actual_results": false,
    "winner_selected": false,
    "winner_selection_allowed_now": false,
    "train_with_this_reward_allowed": false,
    "actual_training_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false,
    "best_reward_claim_allowed": false
  },
  "next_status": "READY_FOR_STEP126_REWARD_ABLATION_ACTUAL_RESULT_INGESTION_PREFLIGHT"
}
'@
$SpecJson | Set-Content -Path ".\05_training\rewards\reward_ablation_selection_criteria_gate_step125.json" -Encoding UTF8

$SpecMd = @'
# Step 125 — Reward Ablation Selection Criteria Gate

Step 125 defines how a future reward ablation winner may be selected.

This step does not select a winner, does not promote a reward, and does not allow MAPPO training.

Current guards:

- `actual_results = false`
- `winner_selected = false`
- `winner_selection_allowed_now = false`
- `train_with_this_reward_allowed = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`

The selection rule is service-quality first. Energy and fleet reduction can only be used after hard constraints and service quality are protected.

A future winner may only be selected from actual results covering R0/R1/R2/R3/R4/R5 × A/A90/A80/A70 × seeds 1,2,3 = 72 rows.

Ranking tiers:

1. Eligibility filter
2. Service quality rank
3. Efficiency secondary rank
4. Stability and seed robustness

A candidate is disqualified if it violates hard constraints, has missing rows, uses no-op/template/dry-run outputs, or improves energy by harming passenger service.
'@
$SpecMd | Set-Content -Path ".\05_training\rewards\reward_ablation_selection_criteria_gate_step125.md" -Encoding UTF8

$MaterializerPy = @'
from __future__ import annotations
import argparse, csv, json
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
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

def materialize(spec: Dict[str, Any], output_root: Path) -> Dict[str, Any]:
    cols = list(spec["scorecard_columns_required_for_future_selection"])
    rows = []
    for candidate_id in spec["candidate_ids"]:
        row = {c: "" for c in cols}
        row.update({
            "candidate_id": candidate_id,
            "eligible": False,
            "actual_result_count": 0,
            "seed_variance_summary": "not_evaluated",
            "selection_rank": "",
            "selection_reason": "No actual reward ablation results have been ingested.",
            "winner_selected": False,
        })
        rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    scorecard = output_root / "reward_ablation_selection_scorecard_template_step125.csv"
    manifest_path = output_root / "reward_ablation_selection_criteria_gate_manifest_step125.json"
    write_csv(scorecard, rows, cols)
    manifest = {
        "artifact_version": "reward_ablation_selection_criteria_gate_manifest_step125_v1",
        "created_at_utc": utc_now(),
        "gate_status": spec["gate_status"],
        "candidate_count": len(spec["candidate_ids"]),
        "condition_count": len(spec["condition_ids"]),
        "seed_count": len(spec["seeds"]),
        "required_actual_result_rows": int(spec["required_actual_result_rows"]),
        "scorecard_rows": len(rows),
        "actual_results": False,
        "winner_selected": False,
        "winner_selection_allowed_now": False,
        "train_with_this_reward_allowed": False,
        "actual_training_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "best_reward_claim_allowed": False,
        "output_root": str(output_root),
        "output_files": {"scorecard_template_csv": str(scorecard), "manifest": str(manifest_path)},
        "source_spec_snapshot": spec,
    }
    dump_json(manifest_path, manifest)
    return manifest

def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="05_training/rewards/reward_ablation_selection_criteria_gate_step125.json")
    ap.add_argument("--output-root", default="artifacts/rewards/reward_ablation_selection_criteria_gate_step125")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = materialize(load_json(resolve_path(root, args.spec)), resolve_path(root, args.output_root))
    print("[OK] Step 125 reward ablation selection criteria gate materialized")
    print(f"[OK] gate_status   : {manifest['gate_status']}")
    print(f"[OK] candidate_count: {manifest['candidate_count']}")
    print(f"[OK] scorecard_rows: {manifest['scorecard_rows']}")
    print(f"[OK] actual_results: {manifest['actual_results']}")
    print(f"[OK] winner_selected: {manifest['winner_selected']}")
    print(f"[OK] train_allowed : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] manifest      : {manifest['output_files']['manifest']}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
'@
$MaterializerPy | Set-Content -Path ".\05_training\rewards\reward_ablation_selection_criteria_gate_step125.py" -Encoding UTF8

$ValidatorPy = @'
from __future__ import annotations
import argparse, csv, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_SEEDS = [1, 2, 3]

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

def add(checks, check_id, expected, actual, desc):
    checks.append({"check_id": check_id, "description": desc, "expected": expected, "actual": actual, "passed": expected == actual, "severity": "blocker", "blocking": True})

def validate_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    checks = []
    spec = manifest.get("source_spec_snapshot", {})
    readiness = spec.get("selection_readiness", {})
    principles = spec.get("primary_selection_principles", {})
    guards = spec.get("guard_flags", {})
    add(checks, "gate_status", "SELECTION_CRITERIA_DEFINED_NO_WINNER_SELECTED", manifest.get("gate_status"), "Gate must define criteria without winner.")
    add(checks, "candidate_count", 6, int(manifest.get("candidate_count", -1)), "Candidate count must be 6.")
    add(checks, "condition_count", 4, int(manifest.get("condition_count", -1)), "Condition count must be 4.")
    add(checks, "seed_count", 3, int(manifest.get("seed_count", -1)), "Seed count must be 3.")
    add(checks, "required_rows", 72, int(manifest.get("required_actual_result_rows", -1)), "Future actual rows must be 72.")
    add(checks, "candidate_set", EXPECTED_CANDIDATES, list(spec.get("candidate_ids", [])), "Candidate IDs must be R0-R5.")
    add(checks, "condition_set", EXPECTED_CONDITIONS, list(spec.get("condition_ids", [])), "Condition IDs must be A/A90/A80/A70.")
    add(checks, "seed_set", EXPECTED_SEEDS, [int(x) for x in spec.get("seeds", [])], "Seeds must be 1/2/3.")
    for key in ["actual_results", "winner_selected", "winner_selection_allowed_now", "train_with_this_reward_allowed", "actual_training_allowed", "paper_level_claim_allowed", "causal_performance_claim_allowed", "best_reward_claim_allowed"]:
        add(checks, f"manifest_guard_{key}", False, bool(manifest.get(key)), f"{key} must remain false in manifest.")
        add(checks, f"spec_guard_{key}", False, bool(guards.get(key)), f"{key} must remain false in spec.")
    add(checks, "actual_results_required", True, bool(readiness.get("actual_results_required")), "Actual results must be required for selection.")
    for key in ["noop_results_allowed_for_selection", "template_results_allowed_for_selection", "dry_run_results_allowed_for_selection", "winner_selected", "winner_selection_allowed_now", "train_with_selected_reward_allowed"]:
        add(checks, f"readiness_{key}", False, bool(readiness.get(key)), f"{key} must be false.")
    for key in ["service_quality_first", "hard_constraints_must_pass", "no_energy_saving_by_stranding_passengers", "long_wait_fairness_required", "fleet_reduction_bonus_only", "energy_fleet_secondary"]:
        add(checks, f"principle_{key}", True, bool(principles.get(key)), f"{key} must be true.")
    out = manifest.get("output_files", {})
    scorecard = Path(out.get("scorecard_template_csv", ""))
    add(checks, "scorecard_file_exists", True, scorecard.exists(), "Scorecard template must exist.")
    rows = read_csv(scorecard) if scorecard.exists() else []
    add(checks, "scorecard_rows", 6, len(rows), "Scorecard must have one row per candidate.")
    add(checks, "scorecard_candidate_set", EXPECTED_CANDIDATES, [r.get("candidate_id") for r in rows], "Scorecard candidate order must match.")
    add(checks, "no_scorecard_winner", 0, sum(str(r.get("winner_selected")).lower() == "true" for r in rows), "No scorecard row may select winner.")
    add(checks, "selection_tier_count", 4, len(spec.get("selection_tiers", [])), "Four selection tiers are required.")
    add(checks, "disqualifying_condition_count_min", True, len(spec.get("disqualifying_conditions", [])) >= 8, "At least eight disqualifying conditions should be documented.")
    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    ok = not failures
    return {"artifact_version": "validate_reward_ablation_selection_criteria_gate_step125_v1", "created_at_utc": utc_now(), "audit_status": "PASS" if ok else "FAIL", "gate_status": "PASS_SELECTION_CRITERIA_DEFINED_NO_WINNER_SELECTED" if ok else "FAIL_SELECTION_CRITERIA_GATE", "next_status": "READY_FOR_STEP126_REWARD_ABLATION_ACTUAL_RESULT_INGESTION_PREFLIGHT" if ok else "BLOCKED_FIX_STEP125_SELECTION_CRITERIA", "candidate_count": len(rows), "actual_results": False, "winner_selected": False, "train_with_this_reward_allowed": False, "failure_count": len(failures), "checks": checks}

def resolve_path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-root", default="artifacts/rewards/reward_ablation_selection_criteria_gate_step125_validation")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = resolve_path(root, args.output_root); out.mkdir(parents=True, exist_ok=True)
    report = validate_manifest(load_json(resolve_path(root, args.manifest)))
    path = out / "reward_ablation_selection_criteria_gate_step125_validation_report.json"
    dump_json(path, report)
    print("[OK] Step 125 reward ablation selection criteria gate validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] candidate_count: {report['candidate_count']}")
    print(f"[OK] actual_results: {report['actual_results']}")
    print(f"[OK] winner_selected: {report['winner_selected']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {path}")
    return 0 if report["audit_status"] == "PASS" else 1
if __name__ == "__main__":
    raise SystemExit(main())
'@
$ValidatorPy | Set-Content -Path ".\05_training\rewards\validate_reward_ablation_selection_criteria_gate_step125.py" -Encoding UTF8

$TestPy = @'
from __future__ import annotations
import json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

def load_json(path: Path):
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f: return json.load(f)
        except UnicodeDecodeError: continue
    raise RuntimeError(f"failed to read json: {path}")

def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f: json.dump(payload, f, ensure_ascii=False, indent=2)

def suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    materializer = root / "05_training" / "rewards" / "reward_ablation_selection_criteria_gate_step125.py"
    validator = root / "05_training" / "rewards" / "validate_reward_ablation_selection_criteria_gate_step125.py"
    spec = root / "05_training" / "rewards" / "reward_ablation_selection_criteria_gate_step125.json"
    s = suffix()
    output_root = root / "artifacts" / "rewards" / f"reward_ablation_selection_criteria_gate_step125_selftest_{s}"
    validation_root = root / "artifacts" / "rewards" / f"reward_ablation_selection_criteria_gate_step125_validation_{s}"
    res = subprocess.run([sys.executable, str(materializer), "--spec", str(spec), "--output-root", str(output_root)], cwd=root, text=True, capture_output=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr); raise SystemExit("[FAIL] Step 125 materializer failed")
    manifest = output_root / "reward_ablation_selection_criteria_gate_manifest_step125.json"
    val = subprocess.run([sys.executable, str(validator), "--manifest", str(manifest), "--output-root", str(validation_root)], cwd=root, text=True, capture_output=True)
    print(val.stdout)
    if val.returncode != 0:
        print(val.stderr); raise SystemExit("[FAIL] Step 125 validator failed")
    report = load_json(validation_root / "reward_ablation_selection_criteria_gate_step125_validation_report.json")
    expected = {"audit_status": "PASS", "gate_status": "PASS_SELECTION_CRITERIA_DEFINED_NO_WINNER_SELECTED", "next_status": "READY_FOR_STEP126_REWARD_ABLATION_ACTUAL_RESULT_INGESTION_PREFLIGHT", "candidate_count": 6, "actual_results": False, "winner_selected": False, "train_with_this_reward_allowed": False, "failure_count": 0}
    for k, v in expected.items():
        if report.get(k) != v: raise SystemExit(f"[FAIL] {k}: expected={v!r}, actual={report.get(k)!r}")
    bad = load_json(manifest)
    bad["winner_selected"] = True
    bad["source_spec_snapshot"]["guard_flags"]["winner_selected"] = True
    bad_path = output_root / "bad_winner_selected_manifest_step125.json"
    dump_json(bad_path, bad)
    bad_root = root / "artifacts" / "rewards" / f"reward_ablation_selection_criteria_gate_step125_bad_validation_{s}"
    bad_res = subprocess.run([sys.executable, str(validator), "--manifest", str(bad_path), "--output-root", str(bad_root)], cwd=root, text=True, capture_output=True)
    if bad_res.returncode == 0:
        print(bad_res.stdout); raise SystemExit("[FAIL] bad winner_selected manifest unexpectedly passed")
    bad_report = load_json(bad_root / "reward_ablation_selection_criteria_gate_step125_validation_report.json")
    failed = {c["check_id"] for c in bad_report.get("checks", []) if not c.get("passed", False)}
    if "manifest_guard_winner_selected" not in failed: raise SystemExit("[FAIL] winner_selected guard did not fail")
    print("[OK] Step 125 reward ablation selection criteria gate self-test PASS")
    print("[DONE] Step 125 reward ablation selection criteria gate complete.")
if __name__ == "__main__": main()
'@
$TestPy | Set-Content -Path ".\05_training\rewards\test_validate_reward_ablation_selection_criteria_gate_step125.py" -Encoding UTF8

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else { $py = "python" }

& $py ".\05_training\rewards\test_validate_reward_ablation_selection_criteria_gate_step125.py"
if ($LASTEXITCODE -ne 0) { throw "[FAIL] Step 125 reward ablation selection criteria gate self-test failed" }

Write-Host ""
Write-Host "[DONE] Step 125 reward ablation selection criteria gate files created and tested."
Write-Host "[NEXT] Review git status, then commit the Step 125 selection criteria gate files."
