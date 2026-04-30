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
