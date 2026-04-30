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
