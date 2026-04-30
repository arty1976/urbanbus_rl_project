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
