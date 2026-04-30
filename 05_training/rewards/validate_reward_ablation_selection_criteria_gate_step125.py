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
