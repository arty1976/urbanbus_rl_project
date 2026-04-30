$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

# ---------------------------------------------------------------------
# Step 120 preflight spec JSON
# ---------------------------------------------------------------------
$SpecJson = @'
{
  "artifact_version": "reward_ablation_execution_preflight_step120_v1",
  "step": 120,
  "preflight_status": "DRY_RUN_PLAN_PREFLIGHT_ONLY_NOT_EXECUTABLE",
  "purpose": "Preflight the Step 119 reward ablation dry-run plan before any execution path is allowed.",
  "expected_candidate_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
  "expected_condition_ids": ["A", "A90", "A80", "A70"],
  "expected_seeds": [1, 2, 3],
  "expected_planned_run_count": 72,
  "required_upstream_artifacts": [
    "05_training/rewards/reward_ablation_matrix_step115.json",
    "05_training/rewards/reward_ablation_result_schema_step117.json",
    "05_training/rewards/reward_ablation_result_writer_guard_step118.py",
    "05_training/rewards/reward_ablation_runner_dry_run_plan_step119.json",
    "05_training/rewards/reward_ablation_runner_dry_run_plan_step119.py",
    "05_training/rewards/validate_reward_ablation_runner_dry_run_plan_step119.py"
  ],
  "execution_guards": {
    "dry_run_plan_required": true,
    "execute_allowed": false,
    "actual_training_allowed": false,
    "train_with_this_reward_allowed": false,
    "actual_results": false,
    "winner_selected": false,
    "best_reward_claim_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false,
    "promotion_allowed": false
  },
  "forbidden_command_tokens": [
    "--execute",
    "--train",
    "--promote",
    "--select-winner",
    "--allow-actual-results"
  ],
  "required_command_tokens": [
    "--dry-run-plan"
  ],
  "next_required_gate": "Step121 reward ablation execution sandbox guard or manual approval checklist"
}
'@
$SpecJsonPath = ".\05_training\rewards\reward_ablation_execution_preflight_step120.json"
$SpecJson | Set-Content -Path $SpecJsonPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 120 markdown
# ---------------------------------------------------------------------
$SpecMd = @'
# Step 120 — Reward Ablation Execution Preflight

## Purpose

Step 120 checks the Step 119 reward ablation dry-run plan before any execution path is allowed.

This step is still not an execution step. It does not run MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화), does not train a policy, does not evaluate a winner, and does not promote a reward.

## Expected Plan Shape

The preflight expects:

```text
R0~R5 reward candidates 6
× A/A90/A80/A70 conditions 4
× seeds 1,2,3
= 72 planned dry-run rows
```

## Required Guards

All rows must preserve:

- `execute_allowed = false`
- `actual_training_allowed = false`
- `train_with_this_reward_allowed = false`
- `actual_results = false`
- `winner_selected = false`
- `best_reward_claim_allowed = false`

## Command Safety

Every planned command must include:

- `--dry-run-plan`

No planned command may include:

- `--execute`
- `--train`
- `--promote`
- `--select-winner`
- `--allow-actual-results`

## Interpretation

A PASS means the dry-run plan is complete and still non-executable.

A PASS does not mean the reward ablation has run. It only means the plan is safe enough to be reviewed by the next gate.
'@
$SpecMdPath = ".\05_training\rewards\reward_ablation_execution_preflight_step120.md"
$SpecMd | Set-Content -Path $SpecMdPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 120 preflight script
# ---------------------------------------------------------------------
$PreflightPy = @'
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
'@
$PreflightPath = ".\05_training\rewards\reward_ablation_execution_preflight_step120.py"
$PreflightPy | Set-Content -Path $PreflightPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 120 report validator
# ---------------------------------------------------------------------
$ValidatorPy = @'
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
'@
$ValidatorPath = ".\05_training\rewards\validate_reward_ablation_execution_preflight_step120.py"
$ValidatorPy | Set-Content -Path $ValidatorPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 120 self-test
# ---------------------------------------------------------------------
$TestPy = @'
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
CONDITIONS = ["A", "A90", "A80", "A70"]
SEEDS = [1, 2, 3]


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_step119_fixture(root: Path, corrupt_execute: bool = False) -> None:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for cand in CANDIDATES:
        for cond in CONDITIONS:
            for seed in SEEDS:
                cmd = (
                    "python 05_training/rewards/reward_ablation_result_writer_guard_step118.py "
                    f"--candidate-id {cand} --condition-id {cond} --seed {seed} "
                    f"--output-root artifacts/rewards/ablation/{cand}/{cond}/seed_{seed:03d} "
                    "--dry-run-plan"
                )
                if corrupt_execute and cand == "R0" and cond == "A" and seed == 1:
                    cmd += " --execute"
                rows.append({
                    "run_id": f"{cand}_{cond}_seed_{seed:03d}",
                    "candidate_id": cand,
                    "condition_id": cond,
                    "seed": seed,
                    "command_type": "dry_run_plan_only",
                    "execute_allowed": "false",
                    "actual_training_allowed": "false",
                    "train_with_this_reward_allowed": "false",
                    "actual_results": "false",
                    "winner_selected": "false",
                    "best_reward_claim_allowed": "false",
                    "planned_output_root": f"artifacts/rewards/ablation/{cand}/{cond}/seed_{seed:03d}",
                    "planned_command_text": cmd,
                })

    plan_csv = root / "reward_ablation_runner_dry_run_plan.csv"
    with open(plan_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plan_json = root / "reward_ablation_runner_dry_run_plan.json"
    dump_json(plan_json, {"rows": rows})

    manifest = {
        "artifact_version": "reward_ablation_runner_dry_run_plan_step119_manifest_fixture_v1",
        "planner_status": "DRY_RUN_ROWS_WRITTEN_NOT_EXECUTED",
        "planned_runs": len(rows),
        "execute_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "output_root": str(root),
        "output_files": {
            "plan_csv": str(plan_csv),
            "plan_json": str(plan_json)
        }
    }
    dump_json(root / "reward_ablation_runner_dry_run_manifest.json", manifest)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    unique = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"

    preflight = project_root / "05_training" / "rewards" / "reward_ablation_execution_preflight_step120.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_execution_preflight_step120.py"

    fixture_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_preflight_step120_fixture_{unique}"
    report_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_preflight_step120_selftest_{unique}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_preflight_step120_validation_{unique}"

    write_step119_fixture(fixture_root, corrupt_execute=False)

    cmd = [sys.executable, str(preflight), "--step119-plan-root", str(fixture_root), "--output-root", str(report_root)]
    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 120 valid preflight unexpectedly failed")

    report_path = report_root / "reward_ablation_execution_preflight_step120_report.json"
    report = load_json(report_path)
    if report.get("audit_status") != "PASS":
        raise SystemExit("[FAIL] Step 120 report audit_status must be PASS")
    if report.get("planned_runs") != 72:
        raise SystemExit("[FAIL] Step 120 planned_runs must be 72")
    if report.get("execute_allowed") is not False:
        raise SystemExit("[FAIL] Step 120 execute_allowed must be false")

    vcmd = [sys.executable, str(validator), "--report", str(report_path), "--output-root", str(validation_root)]
    vresult = subprocess.run(vcmd, cwd=project_root, text=True, capture_output=True)
    print(vresult.stdout)
    if vresult.returncode != 0:
        print(vresult.stderr)
        raise SystemExit("[FAIL] Step 120 report validator unexpectedly failed")

    # Bad case: command contains --execute and must fail at preflight.
    bad_fixture_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_preflight_step120_bad_fixture_{unique}"
    bad_report_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_preflight_step120_bad_selftest_{unique}"
    write_step119_fixture(bad_fixture_root, corrupt_execute=True)
    bad_cmd = [sys.executable, str(preflight), "--step119-plan-root", str(bad_fixture_root), "--output-root", str(bad_report_root)]
    bad_result = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] corrupted Step 119 plan unexpectedly passed Step 120 preflight")

    bad_report = load_json(bad_report_root / "reward_ablation_execution_preflight_step120_report.json")
    failed_ids = {c["check_id"] for c in bad_report.get("checks", []) if not c.get("passed")}
    if "forbidden_command_tokens_absent" not in failed_ids:
        raise SystemExit("[FAIL] forbidden command token guard did not fail")

    print("[OK] Step 120 reward ablation execution preflight self-test PASS")
    print("[DONE] Step 120 reward ablation execution preflight complete.")


if __name__ == "__main__":
    main()
'@
$TestPath = ".\05_training\rewards\test_validate_reward_ablation_execution_preflight_step120.py"
$TestPy | Set-Content -Path $TestPath -Encoding UTF8

# ---------------------------------------------------------------------
# Run Step 120 self-test
# ---------------------------------------------------------------------
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 120 reward ablation execution preflight self-test failed"
}

Write-Host ""
Write-Host "[DONE] Step 120 reward ablation execution preflight files created and tested."
Write-Host "[NEXT] Review git status, then commit the Step 120 preflight files."
