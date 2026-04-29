from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "validate_reward_ablation_result_writer_guard_step118_v1"

EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_SEEDS = [1, 2, 3]

CORE_12_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
]

REQUIRED_DETAIL_COLUMNS = [
    "candidate_id",
    "condition_id",
    "seed",
    "window_count",
    "evaluation_horizon_minutes",
    "source_mode",
    "actual_results",
    "winner_selected",
    "train_with_this_reward_allowed",
    "hard_constraint_violation_count",
    "hard_constraint_pass",
    *CORE_12_KPIS,
]

REQUIRED_SUMMARY_COLUMNS = [
    "candidate_id",
    "condition_count",
    "seed_count",
    "window_count_total",
    "actual_results",
    "winner_selected",
    "best_claim_allowed",
    "train_with_this_reward_allowed",
    "hard_constraint_violation_count_total",
    "hard_constraint_pass_all",
    "selection_eligible",
    "selection_block_reason",
]

REQUIRED_VIOLATION_COLUMNS = [
    "candidate_id",
    "condition_id",
    "seed",
    "constraint_id",
    "metric",
    "threshold",
    "observed_value",
    "violation",
    "severity",
    "notes",
]


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


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def add_check(
    checks: List[Dict[str, Any]],
    check_id: str,
    expected: Any,
    actual: Any,
    description: str,
    severity: str = "blocker",
) -> None:
    checks.append({
        "check_id": check_id,
        "description": description,
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
        "severity": severity,
        "blocking": severity == "blocker",
    })


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def validate_output(output_root: Path) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    manifest_path = output_root / "reward_ablation_result_writer_guard_manifest.json"
    add_check(checks, "manifest_exists", True, manifest_path.exists(), "Manifest must exist.")

    if not manifest_path.exists():
        return {
            "artifact_version": ARTIFACT_VERSION,
            "created_at_utc": utc_now(),
            "audit_status": "FAIL",
            "gate_status": "FAIL_REWARD_ABLATION_RESULT_WRITER_GUARD",
            "next_status": "BLOCKED_MISSING_MANIFEST",
            "check_count": len(checks),
            "failure_count": 1,
            "checks": checks,
        }

    manifest = load_json_any_encoding(manifest_path)

    add_check(checks, "manifest_artifact_version", "reward_ablation_result_writer_guard_step118_v1", manifest.get("artifact_version"), "Manifest artifact version must match Step 118.")
    add_check(checks, "writer_status", "TEMPLATE_RESULTS_WRITTEN_NOT_EVALUATED", manifest.get("writer_status"), "Writer status must remain template/not evaluated.")
    add_check(checks, "candidate_ids", EXPECTED_CANDIDATES, manifest.get("candidate_ids"), "R0-R5 candidates must be included.")
    add_check(checks, "condition_ids", EXPECTED_CONDITIONS, manifest.get("condition_ids"), "A-family conditions must be included.")
    add_check(checks, "seeds", EXPECTED_SEEDS, manifest.get("seeds"), "Seeds must be [1,2,3].")
    add_check(checks, "manifest_actual_results_false", False, manifest.get("actual_results"), "Actual results must be false.")
    add_check(checks, "manifest_winner_selected_false", False, manifest.get("winner_selected"), "Winner must not be selected.")
    add_check(checks, "manifest_best_claim_false", False, manifest.get("best_claim_allowed"), "Best claim must not be allowed.")
    add_check(checks, "manifest_train_allowed_false", False, manifest.get("train_with_this_reward_allowed"), "Training must remain disallowed.")
    add_check(checks, "manifest_promoted_false", False, manifest.get("trainable_reward_promoted"), "Reward must not be promoted.")

    files = manifest.get("output_files", {})
    detail_path = Path(files.get("detail_template_csv", ""))
    summary_path = Path(files.get("summary_template_csv", ""))
    violation_path = Path(files.get("hard_constraint_violations_template_csv", ""))
    selection_path = Path(files.get("selection_summary_template_json", ""))

    for name, path in [
        ("detail_template_csv", detail_path),
        ("summary_template_csv", summary_path),
        ("hard_constraint_violations_template_csv", violation_path),
        ("selection_summary_template_json", selection_path),
    ]:
        add_check(checks, f"{name}_exists", True, path.exists(), f"{name} must exist.")

    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    violation_rows: List[Dict[str, Any]] = []
    selection: Dict[str, Any] = {}

    if detail_path.exists():
        detail_rows = read_csv_rows(detail_path)
        detail_cols = list(detail_rows[0].keys()) if detail_rows else []
        for col in REQUIRED_DETAIL_COLUMNS:
            add_check(checks, f"detail_col_{col}", True, col in detail_cols, f"Detail template must include {col}.")
        add_check(checks, "detail_row_count", len(EXPECTED_CANDIDATES) * len(EXPECTED_CONDITIONS) * len(EXPECTED_SEEDS), len(detail_rows), "Detail template must have candidate x condition x seed rows.")
        add_check(checks, "detail_candidate_set", EXPECTED_CANDIDATES, sorted({r.get("candidate_id") for r in detail_rows}), "Detail template must contain R0-R5.")
        add_check(
            checks,
            "detail_condition_set",
            sorted(EXPECTED_CONDITIONS),
            sorted({r.get("condition_id") for r in detail_rows}),
            "Detail template must contain A-family conditions.",
        )
        add_check(checks, "detail_seed_set", [str(x) for x in EXPECTED_SEEDS], sorted({str(r.get("seed")) for r in detail_rows}), "Detail template must contain seeds 1,2,3.")
        add_check(checks, "detail_actual_results_all_false", True, all(not as_bool(r.get("actual_results")) for r in detail_rows), "Detail rows must be template-only.")
        add_check(checks, "detail_winner_selected_all_false", True, all(not as_bool(r.get("winner_selected")) for r in detail_rows), "Winner must not be selected in detail rows.")
        add_check(checks, "detail_train_allowed_all_false", True, all(not as_bool(r.get("train_with_this_reward_allowed")) for r in detail_rows), "Training must be false in detail rows.")

    if summary_path.exists():
        summary_rows = read_csv_rows(summary_path)
        summary_cols = list(summary_rows[0].keys()) if summary_rows else []
        for col in REQUIRED_SUMMARY_COLUMNS:
            add_check(checks, f"summary_col_{col}", True, col in summary_cols, f"Summary template must include {col}.")
        add_check(checks, "summary_row_count", len(EXPECTED_CANDIDATES), len(summary_rows), "Summary template must have one row per candidate.")
        add_check(checks, "summary_candidate_set", EXPECTED_CANDIDATES, sorted({r.get("candidate_id") for r in summary_rows}), "Summary template must contain R0-R5.")
        add_check(checks, "summary_actual_results_all_false", True, all(not as_bool(r.get("actual_results")) for r in summary_rows), "Summary rows must be template-only.")
        add_check(checks, "summary_selection_eligible_all_false", True, all(not as_bool(r.get("selection_eligible")) for r in summary_rows), "No candidate can be selection eligible in template.")
        add_check(checks, "summary_train_allowed_all_false", True, all(not as_bool(r.get("train_with_this_reward_allowed")) for r in summary_rows), "Training must be false in summary rows.")

    if violation_path.exists():
        violation_rows = read_csv_rows(violation_path)
        violation_cols = list(violation_rows[0].keys()) if violation_rows else []
        for col in REQUIRED_VIOLATION_COLUMNS:
            add_check(checks, f"violation_col_{col}", True, col in violation_cols, f"Violation template must include {col}.")
        add_check(checks, "violation_row_count", len(EXPECTED_CANDIDATES), len(violation_rows), "Violation template must have one template row per candidate.")
        add_check(checks, "violation_candidate_set", EXPECTED_CANDIDATES, sorted({r.get("candidate_id") for r in violation_rows}), "Violation template must contain R0-R5.")

    if selection_path.exists():
        selection = load_json_any_encoding(selection_path)
        add_check(checks, "selection_actual_results_false", False, selection.get("actual_results_available"), "Selection summary must not claim actual results.")
        add_check(checks, "selection_winner_selected_false", False, selection.get("winner_selected"), "Selection summary must not select winner.")
        add_check(checks, "selection_best_claim_false", False, selection.get("best_claim_allowed"), "Selection summary must not allow best claim.")
        add_check(checks, "selection_train_allowed_false", False, selection.get("train_with_this_reward_allowed"), "Selection summary must not allow training.")
        add_check(checks, "selection_promoted_false", False, selection.get("trainable_reward_promoted"), "Selection summary must not promote reward.")
        add_check(checks, "selection_candidate_ids", EXPECTED_CANDIDATES, selection.get("candidate_ids"), "Selection summary must list R0-R5.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": (
            "PASS_REWARD_ABLATION_RESULT_WRITER_GUARD_TEMPLATE_ONLY"
            if audit_status == "PASS"
            else "FAIL_REWARD_ABLATION_RESULT_WRITER_GUARD"
        ),
        "next_status": (
            "READY_FOR_STEP119_REWARD_ABLATION_RUNNER_DRY_RUN_PLAN"
            if audit_status == "PASS"
            else "BLOCKED_FIX_STEP118_OUTPUTS"
        ),
        "candidate_count": len(EXPECTED_CANDIDATES),
        "detail_row_count": len(detail_rows),
        "summary_row_count": len(summary_rows),
        "violation_row_count": len(violation_rows),
        "actual_results": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "best_claim_allowed": False,
        "trainable_reward_promoted": False,
        "check_count": len(checks),
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
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/reward_ablation_result_writer_guard_step118",
    )
    parser.add_argument(
        "--report-root",
        default="artifacts/rewards/reward_ablation_result_writer_guard_step118_validation",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    output_root = resolve_path(project_root, args.output_root)
    report_root = resolve_path(project_root, args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    report = validate_output(output_root)
    report["output_root"] = str(output_root)
    report["report_root"] = str(report_root)

    report_path = report_root / "reward_ablation_result_writer_guard_step118_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 118 reward ablation result writer guard validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] candidate_count: {report['candidate_count']}")
    print(f"[OK] detail_rows    : {report['detail_row_count']}")
    print(f"[OK] summary_rows   : {report['summary_row_count']}")
    print(f"[OK] actual_results : {report['actual_results']}")
    print(f"[OK] winner_selected: {report['winner_selected']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
