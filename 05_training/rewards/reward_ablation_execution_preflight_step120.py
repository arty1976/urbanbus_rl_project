from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


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


def load_csv_rows(path: Path) -> List[Dict[str, Any]]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read csv: {path}")


def resolve_path(project_root: Path, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


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


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def discover_plan_files(plan_root: Path, manifest: Dict[str, Any]) -> Tuple[Path, Path]:
    output_files = manifest.get("output_files", {}) if isinstance(manifest.get("output_files", {}), dict) else {}

    candidate_csvs = [
        output_files.get("plan_csv"),
        output_files.get("dry_run_plan_csv"),
        output_files.get("reward_ablation_runner_dry_run_plan_csv"),
        str(plan_root / "reward_ablation_runner_dry_run_plan.csv"),
    ]
    candidate_jsons = [
        output_files.get("plan_json"),
        output_files.get("dry_run_plan_json"),
        output_files.get("reward_ablation_runner_dry_run_plan_json"),
        str(plan_root / "reward_ablation_runner_dry_run_plan.json"),
    ]

    plan_csv = None
    for value in candidate_csvs:
        if not value:
            continue
        p = Path(value)
        if not p.is_absolute():
            p = plan_root / p.name
        if p.exists():
            plan_csv = p
            break

    plan_json = None
    for value in candidate_jsons:
        if not value:
            continue
        p = Path(value)
        if not p.is_absolute():
            p = plan_root / p.name
        if p.exists():
            plan_json = p
            break

    if plan_csv is None:
        raise RuntimeError(f"plan csv not found under {plan_root}")
    if plan_json is None:
        raise RuntimeError(f"plan json not found under {plan_root}")

    return plan_json, plan_csv


def validate_preflight(spec: Dict[str, Any], manifest: Dict[str, Any], rows: List[Dict[str, Any]], project_root: Path) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    expected_candidates = list(spec["expected_candidate_ids"])
    expected_conditions = list(spec["expected_condition_ids"])
    expected_seeds = [int(x) for x in spec["expected_seeds"]]
    expected_run_count = int(spec["expected_planned_run_count"])
    forbidden_tokens = list(spec.get("forbidden_command_tokens", []))
    required_tokens = list(spec.get("required_command_tokens", []))

    add_check(checks, "row_count", expected_run_count, len(rows), "Dry-run plan row count must be 72.")

    candidates = sorted({str(r.get("candidate_id", "")) for r in rows})
    conditions = sorted({str(r.get("condition_id", "")) for r in rows})
    seeds = sorted({int(r.get("seed")) for r in rows if str(r.get("seed", "")).strip()})

    add_check(checks, "candidate_set", sorted(expected_candidates), candidates, "Candidate set must be R0~R5.")
    add_check(checks, "condition_set", sorted(expected_conditions), conditions, "Condition set must be A/A90/A80/A70.")
    add_check(checks, "seed_set", expected_seeds, seeds, "Seed set must be 1,2,3.")

    combos = [(str(r.get("candidate_id")), str(r.get("condition_id")), int(r.get("seed"))) for r in rows]
    expected_combos = [(c, cond, s) for c in expected_candidates for cond in expected_conditions for s in expected_seeds]
    add_check(checks, "unique_combo_count", expected_run_count, len(set(combos)), "Each candidate/condition/seed combo must appear once.")
    add_check(checks, "combo_set_complete", sorted(expected_combos), sorted(set(combos)), "All dry-run combinations must be present.")

    for flag in [
        "execute_allowed",
        "actual_training_allowed",
        "train_with_this_reward_allowed",
        "actual_results",
        "winner_selected",
        "best_reward_claim_allowed",
    ]:
        bad = [i for i, r in enumerate(rows) if as_bool(r.get(flag, False))]
        add_check(checks, f"all_{flag}_false", [], bad, f"{flag} must be false for every row.")

    missing_required = []
    forbidden_found = []
    for i, r in enumerate(rows):
        cmd = str(r.get("planned_command_text", ""))
        if not cmd:
            cmd = json.dumps(r.get("planned_command_json", ""), ensure_ascii=False)
        for token in required_tokens:
            if token not in cmd:
                missing_required.append({"row_index": i, "token": token})
        for token in forbidden_tokens:
            if token in cmd:
                forbidden_found.append({"row_index": i, "token": token})

    add_check(checks, "required_command_tokens_present", [], missing_required, "Every command must include required dry-run token.")
    add_check(checks, "forbidden_command_tokens_absent", [], forbidden_found, "No command may include execution tokens.")

    missing_artifacts = []
    for rel in spec.get("required_upstream_artifacts", []):
        p = resolve_path(project_root, rel)
        if not p.exists():
            missing_artifacts.append(str(p))
    add_check(checks, "required_upstream_artifacts_exist", [], missing_artifacts, "Required upstream Step 115/117/118/119 artifacts must exist.")

    manifest_status = str(manifest.get("planner_status", manifest.get("writer_status", "")))
    add_check(
        checks,
        "manifest_not_executed_status",
        True,
        "NOT_EXECUTED" in manifest_status or "DRY_RUN" in manifest_status or manifest_status == "",
        "Step 119 manifest must indicate dry-run/not-executed status.",
    )

    failures = [c for c in checks if not c["passed"] and c.get("blocking", False)]
    audit_status = "PASS" if not failures else "FAIL"

    return {
        "artifact_version": "reward_ablation_execution_preflight_step120_report_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "gate_status": "PASS_REWARD_ABLATION_EXECUTION_PREFLIGHT_NOT_EXECUTABLE" if audit_status == "PASS" else "FAIL_REWARD_ABLATION_EXECUTION_PREFLIGHT",
        "next_status": "READY_FOR_STEP121_REWARD_ABLATION_EXECUTION_SANDBOX_GUARD" if audit_status == "PASS" else "BLOCKED_FIX_STEP120_PREFLIGHT",
        "planned_runs": len(rows),
        "candidate_count": len(candidates),
        "condition_count": len(conditions),
        "seed_count": len(seeds),
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "best_reward_claim_allowed": False,
        "failure_count": len(failures),
        "check_count": len(checks),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="05_training/rewards/reward_ablation_execution_preflight_step120.json")
    parser.add_argument("--step119-plan-root", required=True)
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_execution_preflight_step120")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    spec_path = resolve_path(project_root, args.spec)
    plan_root = resolve_path(project_root, args.step119_plan_root)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not spec_path.exists():
        raise SystemExit(f"spec not found: {spec_path}")
    if not plan_root.exists():
        raise SystemExit(f"step119 plan root not found: {plan_root}")

    manifest_path = plan_root / "reward_ablation_runner_dry_run_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Step 119 manifest not found: {manifest_path}")

    spec = load_json_any_encoding(spec_path)
    manifest = load_json_any_encoding(manifest_path)
    plan_json_path, plan_csv_path = discover_plan_files(plan_root, manifest)
    rows = load_csv_rows(plan_csv_path)

    report = validate_preflight(spec=spec, manifest=manifest, rows=rows, project_root=project_root)
    report["spec_path"] = str(spec_path)
    report["step119_plan_root"] = str(plan_root)
    report["step119_manifest"] = str(manifest_path)
    report["plan_json"] = str(plan_json_path)
    report["plan_csv"] = str(plan_csv_path)
    report["output_root"] = str(output_root)

    report_path = output_root / "reward_ablation_execution_preflight_step120_report.json"
    dump_json(report_path, report)

    print("[OK] Step 120 reward ablation execution preflight completed")
    print(f"[OK] audit_status  : {report['audit_status']}")
    print(f"[OK] gate_status   : {report['gate_status']}")
    print(f"[OK] next_status   : {report['next_status']}")
    print(f"[OK] planned_runs  : {report['planned_runs']}")
    print(f"[OK] execute_allowed: {report['execute_allowed']}")
    print(f"[OK] train_allowed : {report['train_with_this_reward_allowed']}")
    print(f"[OK] actual_results: {report['actual_results']}")
    print(f"[OK] failure_count : {report['failure_count']}")
    print(f"[OK] report_json   : {report_path}")

    return 0 if report["audit_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
