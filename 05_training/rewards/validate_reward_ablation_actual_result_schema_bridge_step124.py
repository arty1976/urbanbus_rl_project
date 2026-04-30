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
