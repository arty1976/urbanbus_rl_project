from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


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
        "passed": actual == expected,
        "severity": "blocker",
        "blocking": True,
    })


def validate_report(report: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    add_check(checks, "audit_status", "PASS", report.get("audit_status"), "Preflight audit must pass.")
    add_check(checks, "gate_status", "PASS_REWARD_ABLATION_EXECUTION_PREFLIGHT_NOT_EXECUTABLE", report.get("gate_status"), "Gate must pass in non-executable mode.")
    add_check(checks, "planned_runs", 72, int(report.get("planned_runs", -1)), "Planned runs must be 72.")
    add_check(checks, "candidate_count", 6, int(report.get("candidate_count", -1)), "Candidate count must be 6.")
    add_check(checks, "condition_count", 4, int(report.get("condition_count", -1)), "Condition count must be 4.")
    add_check(checks, "seed_count", 3, int(report.get("seed_count", -1)), "Seed count must be 3.")
    add_check(checks, "execute_allowed_false", False, bool(report.get("execute_allowed")), "execute_allowed must be false.")
    add_check(checks, "actual_training_allowed_false", False, bool(report.get("actual_training_allowed")), "actual_training_allowed must be false.")
    add_check(checks, "train_allowed_false", False, bool(report.get("train_with_this_reward_allowed")), "train_with_this_reward_allowed must be false.")
    add_check(checks, "actual_results_false", False, bool(report.get("actual_results")), "actual_results must be false.")
    add_check(checks, "winner_selected_false", False, bool(report.get("winner_selected")), "winner_selected must be false.")
    add_check(checks, "best_reward_claim_allowed_false", False, bool(report.get("best_reward_claim_allowed")), "best_reward_claim_allowed must be false.")
    add_check(checks, "failure_count_zero", 0, int(report.get("failure_count", -1)), "failure_count must be 0.")

    failures = [c for c in checks if not c["passed"]]
    return {
        "artifact_version": "validate_reward_ablation_execution_preflight_step120_v1",
        "audit_status": "PASS" if not failures else "FAIL",
        "gate_status": "PASS_STEP120_PREFLIGHT_REPORT_VALIDATED" if not failures else "FAIL_STEP120_PREFLIGHT_REPORT_VALIDATION",
        "failure_count": len(failures),
        "check_count": len(checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_execution_preflight_step120_validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = project_root / report_path
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    report = load_json_any_encoding(report_path)
    validation = validate_report(report)
    validation["report_path"] = str(report_path)

    out_path = output_root / "reward_ablation_execution_preflight_step120_validation_report.json"
    dump_json(out_path, validation)

    print("[OK] Step 120 reward ablation execution preflight report validation completed")
    print(f"[OK] audit_status  : {validation['audit_status']}")
    print(f"[OK] gate_status   : {validation['gate_status']}")
    print(f"[OK] failure_count : {validation['failure_count']}")
    print(f"[OK] report_json   : {out_path}")

    return 0 if validation["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
