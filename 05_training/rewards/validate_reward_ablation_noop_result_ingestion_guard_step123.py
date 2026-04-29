from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_ROW_COUNT = 72


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
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read csv: {path}")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n", "", "none", "null"}:
        return False
    raise ValueError(f"cannot parse bool: {value!r}")


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


def validate_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(checks, "source_audit_status", "PASS", manifest.get("audit_status"), "Step 123 ingestion manifest must pass.")
    add_check(checks, "row_count", EXPECTED_ROW_COUNT, int(manifest.get("row_count", -1)), "No-op ingested row count must be 72.")
    add_check(checks, "actual_results_false", False, bool(manifest.get("actual_results")), "No-op ingestion cannot be actual results.")
    add_check(checks, "winner_selected_false", False, bool(manifest.get("winner_selected")), "No winner can be selected from no-op results.")
    add_check(checks, "execute_allowed_false", False, bool(manifest.get("execute_allowed")), "Execution must remain disabled.")
    add_check(checks, "actual_training_allowed_false", False, bool(manifest.get("actual_training_allowed")), "Actual training must remain disabled.")
    add_check(checks, "train_with_reward_false", False, bool(manifest.get("train_with_this_reward_allowed")), "Train-with-reward must remain disabled.")
    add_check(checks, "best_claim_false", False, bool(manifest.get("best_reward_claim_allowed")), "Best reward claim must remain disabled.")
    add_check(checks, "paper_claim_false", False, bool(manifest.get("paper_level_claim_allowed")), "Paper-level claim must remain disabled.")
    add_check(checks, "causal_claim_false", False, bool(manifest.get("causal_performance_claim_allowed")), "Causal performance claim must remain disabled.")

    output_files = manifest.get("output_files", {})
    summary_path = Path(output_files.get("noop_ingestion_summary_csv", ""))
    ingested_path = Path(output_files.get("noop_ingested_rows_csv", ""))
    add_check(checks, "summary_exists", True, summary_path.exists(), "Summary CSV must exist.")
    add_check(checks, "ingested_exists", True, ingested_path.exists(), "Ingested CSV must exist.")

    if summary_path.exists():
        summary_rows = read_csv(summary_path)
        add_check(checks, "summary_candidate_count", 6, len(summary_rows), "Summary must include six candidates.")
        summary_candidates = sorted(str(r.get("candidate_id")) for r in summary_rows)
        add_check(checks, "summary_candidates", EXPECTED_CANDIDATES, summary_candidates, "Summary candidates must be R0-R5.")
        for row in summary_rows:
            cid = row.get("candidate_id")
            add_check(checks, f"summary_{cid}_row_count", 12, int(row.get("row_count", -1)), f"Candidate {cid} must have 12 rows.")
            add_check(checks, f"summary_{cid}_conditions", 4, int(row.get("condition_count", -1)), f"Candidate {cid} must cover four conditions.")
            add_check(checks, f"summary_{cid}_seeds", 3, int(row.get("seed_count", -1)), f"Candidate {cid} must cover three seeds.")
            add_check(checks, f"summary_{cid}_noop_rows", 12, int(row.get("noop_rows", -1)), f"Candidate {cid} must have 12 no-op rows.")
            for col in ["execute_allowed_any", "actual_training_allowed_any", "train_allowed_any", "actual_results_any", "winner_selected_any"]:
                add_check(checks, f"summary_{cid}_{col}_false", False, as_bool(row.get(col)), f"{cid} {col} must remain false.")

    failures = [c for c in checks if not c["passed"] and c["blocking"]]
    audit_status = "PASS" if not failures else "FAIL"
    return {
        "artifact_version": "validate_reward_ablation_noop_result_ingestion_guard_step123_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": "PASS_NOOP_RESULT_INGESTION_GUARD_NOT_ACTUAL_RESULTS" if audit_status == "PASS" else "FAIL_NOOP_RESULT_INGESTION_GUARD",
        "next_status": "READY_FOR_STEP124_REWARD_ABLATION_SELECTION_CRITERIA_GATE" if audit_status == "PASS" else "BLOCKED_FIX_STEP123_INGESTION",
        "row_count": int(manifest.get("row_count", -1)),
        "actual_results": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
    }


def resolve_path(project_root: Path, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingestion-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    manifest_path = resolve_path(project_root, args.ingestion_manifest)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_json_any_encoding(manifest_path)
    report = validate_manifest(manifest)
    report["ingestion_manifest"] = str(manifest_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "reward_ablation_noop_result_ingestion_guard_step123_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 123 reward ablation no-op result ingestion validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] row_count     : {report['row_count']}")
    print(f"[OK] actual_results: {report['actual_results']}")
    print(f"[OK] winner_selected: {report['winner_selected']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
