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
EXPECTED_ROW_COUNT = 72
FORBIDDEN_TOKENS = [
    "--execute",
    "--train",
    "--promote",
    "--select-winner",
    "--allow-training",
    "--actual-results",
    "--winner-selected",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


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
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def add_check(checks: List[Dict[str, Any]], check_id: str, expected: Any, actual: Any, description: str) -> None:
    checks.append({
        "check_id": check_id,
        "description": description,
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
        "blocking": True,
    })


def validate_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(checks, "artifact_version", "reward_ablation_execution_manifest_step121_v1", manifest.get("artifact_version"), "Artifact version must match.")
    add_check(checks, "manifest_status", "EXECUTION_MANIFEST_MATERIALIZED_NOT_EXECUTABLE", manifest.get("manifest_status"), "Manifest must not be executable.")
    add_check(checks, "row_count", EXPECTED_ROW_COUNT, int(manifest.get("row_count", -1)), "Manifest must contain 72 rows.")
    add_check(checks, "execute_allowed_false", False, bool_value(manifest.get("execute_allowed")), "Execution must remain disabled.")
    add_check(checks, "actual_training_allowed_false", False, bool_value(manifest.get("actual_training_allowed")), "Training must remain disabled.")
    add_check(checks, "train_with_this_reward_allowed_false", False, bool_value(manifest.get("train_with_this_reward_allowed")), "Reward training must remain disabled.")
    add_check(checks, "actual_results_false", False, bool_value(manifest.get("actual_results")), "Actual results must be false.")
    add_check(checks, "winner_selected_false", False, bool_value(manifest.get("winner_selected")), "Winner selection must be false.")

    output_files = manifest.get("output_files", {})
    csv_path = Path(str(output_files.get("execution_manifest_csv", "")))
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path

    add_check(checks, "execution_manifest_csv_exists", True, csv_path.exists(), "Execution manifest CSV must exist.")

    rows: List[Dict[str, Any]] = []
    if csv_path.exists():
        rows = read_csv_rows(csv_path)

    add_check(checks, "csv_row_count", EXPECTED_ROW_COUNT, len(rows), "CSV row count must be 72.")

    combos = set()
    run_ids = []
    hashes = []
    orders = []
    for row in rows:
        candidate = str(row.get("candidate_id", "")).strip()
        condition = str(row.get("condition_id", "")).strip()
        seed = int(row.get("seed", 0))
        command = str(row.get("planned_command_text", ""))
        combos.add((candidate, condition, seed))
        run_ids.append(str(row.get("run_id", "")))
        hashes.append(str(row.get("run_hash", "")))
        orders.append(int(row.get("execution_order", 0)))

        for key in [
            "manifest_only",
        ]:
            add_check(checks, f"row_{candidate}_{condition}_{seed}_{key}_true", True, bool_value(row.get(key)), f"{key} must be true.")

        for key in [
            "execute_allowed",
            "actual_training_allowed",
            "train_with_this_reward_allowed",
            "actual_results",
            "winner_selected",
        ]:
            add_check(checks, f"row_{candidate}_{condition}_{seed}_{key}_false", False, bool_value(row.get(key)), f"{key} must be false.")

        add_check(checks, f"row_{candidate}_{condition}_{seed}_dry_run_token", True, "--dry-run-plan" in command, "Command must include --dry-run-plan.")
        for token in FORBIDDEN_TOKENS:
            add_check(checks, f"row_{candidate}_{condition}_{seed}_forbid_{token}", False, token in command, f"Command must not include {token}.")

    expected_combos = {
        (candidate, condition, seed)
        for candidate in EXPECTED_CANDIDATES
        for condition in EXPECTED_CONDITIONS
        for seed in EXPECTED_SEEDS
    }
    add_check(checks, "combination_set_exact", expected_combos, combos, "Candidate/condition/seed combinations must match expected matrix.")
    add_check(checks, "unique_run_ids", len(run_ids), len(set(run_ids)), "run_id values must be unique.")
    add_check(checks, "unique_run_hashes", len(hashes), len(set(hashes)), "run_hash values must be unique.")
    add_check(checks, "execution_order_sequence", list(range(1, EXPECTED_ROW_COUNT + 1)), sorted(orders), "execution_order must be 1..72.")

    failures = [c for c in checks if not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_reward_ablation_execution_manifest_step121_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": (
            "PASS_REWARD_ABLATION_EXECUTION_MANIFEST_NOT_EXECUTABLE"
            if audit_status == "PASS"
            else "FAIL_REWARD_ABLATION_EXECUTION_MANIFEST"
        ),
        "next_status": (
            "READY_FOR_STEP122_REWARD_ABLATION_SANDBOX_NOOP_RUNNER_GUARD"
            if audit_status == "PASS"
            else "BLOCKED_FIX_STEP121_EXECUTION_MANIFEST"
        ),
        "row_count": len(rows),
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
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
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_execution_manifest_step121_validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    manifest_path = resolve_path(project_root, args.manifest)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = load_json(manifest_path)
    report = validate_manifest(manifest)
    report["manifest_path"] = str(manifest_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "reward_ablation_execution_manifest_step121_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 121 reward ablation execution manifest validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] row_count     : {report['row_count']}")
    print(f"[OK] execute_allowed: {report['execute_allowed']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] actual_results: {report['actual_results']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")
    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
