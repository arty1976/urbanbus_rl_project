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
