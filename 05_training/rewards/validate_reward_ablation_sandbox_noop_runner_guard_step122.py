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
PROHIBITED_COMMAND_TOKENS = ["--execute", "--train", "--promote", "--select-winner", "--allow-training"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


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
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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

    add_check(
        checks,
        "artifact_version",
        "reward_ablation_sandbox_noop_runner_guard_step122_manifest_v1",
        manifest.get("artifact_version"),
        "Manifest version must match Step 122.",
    )
    add_check(
        checks,
        "runner_status",
        "SANDBOX_NOOP_STATUS_WRITTEN_NOT_EXECUTED",
        manifest.get("runner_status"),
        "Runner must be no-op and not executed.",
    )
    add_check(checks, "row_count", EXPECTED_ROW_COUNT, int(manifest.get("row_count", -1)), "Manifest row count must be 72.")

    for flag in (
        "execute_allowed",
        "actual_training_allowed",
        "train_with_this_reward_allowed",
        "actual_results",
        "winner_selected",
        "best_reward_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
    ):
        add_check(checks, f"flag_{flag}", False, as_bool(manifest.get(flag, False)), f"{flag} must remain false.")

    output_files = manifest.get("output_files", {})
    status_csv = Path(output_files.get("status_csv", ""))
    if not status_csv.exists():
        checks.append({
            "check_id": "status_csv_exists",
            "description": "status_csv must exist.",
            "expected": True,
            "actual": False,
            "passed": False,
            "blocking": True,
        })
        rows: List[Dict[str, Any]] = []
    else:
        add_check(checks, "status_csv_exists", True, True, "status_csv exists.")
        rows = read_csv_rows(status_csv)

    add_check(checks, "status_row_count", EXPECTED_ROW_COUNT, len(rows), "Status CSV must have 72 rows.")

    combos = set()
    all_noop_status = True
    all_command_not_executed = True
    all_process_not_spawned = True
    all_safe_flags_false = True
    no_prohibited_tokens = True
    all_hash_present = True

    for row in rows:
        candidate_id = str(row.get("candidate_id", "")).strip()
        condition_id = str(row.get("condition_id", "")).strip().upper()
        seed = int(row.get("seed", -999))
        combos.add((candidate_id, condition_id, seed))

        all_noop_status = all_noop_status and row.get("sandbox_status") == "NOOP_RECORDED_NOT_EXECUTED"
        all_command_not_executed = all_command_not_executed and not as_bool(row.get("command_executed"))
        all_process_not_spawned = all_process_not_spawned and not as_bool(row.get("process_spawned"))
        all_hash_present = all_hash_present and bool(str(row.get("command_hash", "")).strip())

        for flag in (
            "execute_allowed",
            "actual_training_allowed",
            "train_with_this_reward_allowed",
            "actual_results",
            "winner_selected",
            "best_reward_claim_allowed",
            "paper_level_claim_allowed",
            "causal_performance_claim_allowed",
        ):
            if as_bool(row.get(flag, False)):
                all_safe_flags_false = False

        command_text = str(row.get("planned_command_text", ""))
        for token in PROHIBITED_COMMAND_TOKENS:
            if token in command_text:
                no_prohibited_tokens = False

    expected_combos = {
        (candidate_id, condition_id, seed)
        for candidate_id in EXPECTED_CANDIDATES
        for condition_id in EXPECTED_CONDITIONS
        for seed in EXPECTED_SEEDS
    }

    add_check(checks, "combo_complete", expected_combos, combos, "All 72 candidate/condition/seed combos must exist.")
    add_check(checks, "all_noop_status", True, all_noop_status, "All rows must be NOOP status.")
    add_check(checks, "all_command_not_executed", True, all_command_not_executed, "No command may be executed.")
    add_check(checks, "all_process_not_spawned", True, all_process_not_spawned, "No process may be spawned.")
    add_check(checks, "all_safe_flags_false", True, all_safe_flags_false, "All execution/training/result flags must be false.")
    add_check(checks, "no_prohibited_tokens", True, no_prohibited_tokens, "No prohibited command token may appear.")
    add_check(checks, "all_hash_present", True, all_hash_present, "All rows need command hash.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "validate_reward_ablation_sandbox_noop_runner_guard_step122_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": (
            "PASS_REWARD_ABLATION_SANDBOX_NOOP_RUNNER_GUARD_NOT_EXECUTED"
            if audit_status == "PASS"
            else "FAIL_REWARD_ABLATION_SANDBOX_NOOP_RUNNER_GUARD"
        ),
        "next_status": (
            "READY_FOR_STEP123_REWARD_ABLATION_RESULT_INGESTION_GUARD"
            if audit_status == "PASS"
            else "BLOCKED_FIX_STEP122_OUTPUTS"
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
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/reward_ablation_sandbox_noop_runner_guard_step122_validation",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    manifest_path = resolve_path(project_root, args.manifest)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    manifest = load_json_any_encoding(manifest_path)
    report = validate_manifest(manifest)
    report["manifest_path"] = str(manifest_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "reward_ablation_sandbox_noop_runner_guard_step122_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 122 reward ablation sandbox no-op runner guard validation completed")
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
