from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "validate_reward_ablation_runner_dry_run_plan_step119_v1"
EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_SEEDS = [1, 2, 3]
EXPECTED_RUN_COUNT = len(EXPECTED_CANDIDATES) * len(EXPECTED_CONDITIONS) * len(EXPECTED_SEEDS)
FORBIDDEN_FLAGS = ["--execute", "--train", "--promote", "--select-winner"]


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


def json_default_for_report(obj):
    """Make validation reports JSON-serializable.

    The validator may place sets in expected/actual check payloads.
    JSON cannot serialize Python set objects, so convert them to sorted lists.
    Also handle a few scalar-like objects defensively.
    """
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)

def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default_for_report)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read csv: {path}")


def add_check(checks: List[Dict[str, Any]], check_id: str, expected: Any, actual: Any, description: str) -> None:
    checks.append({
        "check_id": check_id,
        "description": description,
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
        "blocking": True,
    })


def validate(plan_spec: Dict[str, Any], manifest: Dict[str, Any], csv_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    add_check(checks, "plan_artifact_version", "reward_ablation_runner_dry_run_plan_step119_v1", plan_spec.get("artifact_version"), "Plan spec artifact version must match.")
    add_check(checks, "plan_status", "DRY_RUN_PLAN_NOT_EXECUTABLE", plan_spec.get("plan_status"), "Plan must remain dry-run-only.")
    add_check(checks, "candidates_exact", EXPECTED_CANDIDATES, plan_spec.get("candidate_ids"), "Candidate IDs must be R0-R5.")
    add_check(checks, "conditions_set", sorted(EXPECTED_CONDITIONS), sorted(plan_spec.get("condition_ids", [])), "Condition set must match A-family.")
    add_check(checks, "seeds_exact", EXPECTED_SEEDS, plan_spec.get("seeds"), "Seeds must match expected list.")
    add_check(checks, "planned_run_count_spec", EXPECTED_RUN_COUNT, int(plan_spec.get("planned_run_count", -1)), "Spec planned run count must be 72.")

    guards = plan_spec.get("execution_guards", {})
    for key in [
        "execute_allowed",
        "actual_training_allowed",
        "train_with_this_reward_allowed",
        "actual_results",
        "winner_selected",
        "best_reward_claim_allowed",
        "paper_level_claim_allowed",
        "causal_performance_claim_allowed",
    ]:
        add_check(checks, f"guard_{key}", False, guards.get(key), f"{key} must remain false.")
    add_check(checks, "guard_dry_run_only", True, guards.get("dry_run_only"), "dry_run_only must be true.")

    add_check(checks, "manifest_writer_status", "DRY_RUN_PLAN_WRITTEN_NOT_EXECUTED", manifest.get("writer_status"), "Manifest writer status must be dry-run only.")
    add_check(checks, "manifest_planned_run_count", EXPECTED_RUN_COUNT, int(manifest.get("planned_run_count", -1)), "Manifest run count must be 72.")
    add_check(checks, "csv_row_count", EXPECTED_RUN_COUNT, len(csv_rows), "CSV row count must be 72.")

    combos = set()
    for i, row in enumerate(csv_rows):
        candidate_id = row.get("candidate_id")
        condition_id = row.get("condition_id")
        seed = int(row.get("seed", -999))
        combos.add((candidate_id, condition_id, seed))
        prefix = f"row_{i:03d}"
        add_check(checks, f"{prefix}_candidate_known", True, candidate_id in EXPECTED_CANDIDATES, "Candidate must be known.")
        add_check(checks, f"{prefix}_condition_known", True, condition_id in EXPECTED_CONDITIONS, "Condition must be known.")
        add_check(checks, f"{prefix}_seed_known", True, seed in EXPECTED_SEEDS, "Seed must be known.")
        add_check(checks, f"{prefix}_execute_false", "False", str(row.get("execute_allowed")), "execute_allowed must be false in row.")
        add_check(checks, f"{prefix}_train_false", "False", str(row.get("train_with_this_reward_allowed")), "train flag must be false in row.")
        cmd_text = str(row.get("planned_command_text", ""))
        add_check(checks, f"{prefix}_has_dry_run_flag", True, "--dry-run-plan" in cmd_text, "Command must include --dry-run-plan.")
        for flag in FORBIDDEN_FLAGS:
            add_check(checks, f"{prefix}_forbid_{flag.replace('-', '_')}", False, flag in cmd_text, f"Command must not include {flag}.")

    expected_combos = {
        (c, cond, s)
        for c in EXPECTED_CANDIDATES
        for cond in EXPECTED_CONDITIONS
        for s in EXPECTED_SEEDS
    }
    add_check(checks, "combo_coverage", expected_combos, combos, "All candidate-condition-seed combinations must exist exactly once.")

    failures = [c for c in checks if c["blocking"] and not c["passed"]]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": "PASS_REWARD_ABLATION_RUNNER_DRY_RUN_PLAN_NOT_EXECUTED" if audit_status == "PASS" else "FAIL_REWARD_ABLATION_RUNNER_DRY_RUN_PLAN",
        "next_status": "READY_FOR_STEP120_REWARD_ABLATION_EXECUTION_PREFLIGHT" if audit_status == "PASS" else "BLOCKED_FIX_STEP119_PLAN",
        "candidate_count": len(EXPECTED_CANDIDATES),
        "condition_count": len(EXPECTED_CONDITIONS),
        "seed_count": len(EXPECTED_SEEDS),
        "planned_run_count": len(csv_rows),
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "failure_count": len(failures),
        "check_count": len(checks),
        "checks": checks,
    }


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-spec", default="05_training/rewards/reward_ablation_runner_dry_run_plan_step119.json")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_runner_dry_run_plan_step119_validation")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    spec_path = resolve_path(project_root, args.plan_spec)
    manifest_path = resolve_path(project_root, args.manifest)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    plan_spec = load_json_any_encoding(spec_path)
    manifest = load_json_any_encoding(manifest_path)
    csv_path = Path(manifest["output_files"]["plan_csv"])
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path
    csv_rows = read_csv_rows(csv_path)

    report = validate(plan_spec, manifest, csv_rows)
    report["plan_spec"] = str(spec_path)
    report["manifest"] = str(manifest_path)
    report["plan_csv"] = str(csv_path)

    report_path = output_root / "reward_ablation_runner_dry_run_plan_step119_validation_report.json"
    dump_json(report_path, report)

    print("[OK] Step 119 reward ablation runner dry-run plan validation completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] planned_runs  : {report['planned_run_count']}")
    print(f"[OK] execute_allowed: {report['execute_allowed']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] actual_results: {report['actual_results']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

