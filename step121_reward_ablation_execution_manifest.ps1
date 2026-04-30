$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

# ---------------------------------------------------------------------
# Step 121 spec JSON
# ---------------------------------------------------------------------
$SpecJson = @'
{
  "artifact_version": "reward_ablation_execution_manifest_step121_v1",
  "step": 121,
  "manifest_status": "EXECUTION_MANIFEST_MATERIALIZED_NOT_EXECUTABLE",
  "purpose": "Materialize the 72-row Step 119 reward ablation dry-run plan into a stable execution manifest while preserving execution and training guards.",
  "expected_candidate_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
  "expected_condition_ids": ["A", "A90", "A80", "A70"],
  "expected_seeds": [1, 2, 3],
  "expected_manifest_rows": 72,
  "required_source_gates": [
    "Step115 reward_ablation_matrix",
    "Step117 reward_ablation_result_schema",
    "Step118 reward_ablation_result_writer_guard",
    "Step119 reward_ablation_runner_dry_run_plan",
    "Step120 reward_ablation_execution_preflight"
  ],
  "execution_guards": {
    "manifest_only": true,
    "execute_allowed": false,
    "actual_training_allowed": false,
    "train_with_this_reward_allowed": false,
    "actual_results": false,
    "winner_selected": false,
    "best_reward_claim_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false
  },
  "dangerous_command_tokens_forbidden": [
    "--execute",
    "--train",
    "--promote",
    "--select-winner",
    "--allow-training",
    "--actual-results",
    "--winner-selected"
  ],
  "required_command_tokens": [
    "--dry-run-plan"
  ],
  "required_manifest_columns": [
    "execution_manifest_id",
    "execution_order",
    "run_id",
    "run_hash",
    "candidate_id",
    "condition_id",
    "seed",
    "planned_output_root",
    "planned_command_text",
    "command_sha256",
    "manifest_only",
    "execute_allowed",
    "actual_training_allowed",
    "train_with_this_reward_allowed",
    "actual_results",
    "winner_selected"
  ],
  "next_step": "Step122 reward ablation sandbox/no-op runner guard"
}
'@

$SpecPath = ".\05_training\rewards\reward_ablation_execution_manifest_step121.json"
$SpecJson | Set-Content -Path $SpecPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 121 markdown
# ---------------------------------------------------------------------
$SpecMd = @'
# Step 121 — Reward Ablation Execution Manifest

## Purpose

Step 121 converts the Step 119 reward ablation dry-run plan into a stable execution manifest.

This is still not execution. It is the final packaging layer before any sandbox/no-op runner guard.

## Expected Matrix

- Reward candidates: `R0`, `R1`, `R2`, `R3`, `R4`, `R5`
- Conditions: `A`, `A90`, `A80`, `A70`
- Seeds: `1`, `2`, `3`
- Expected rows: `6 × 4 × 3 = 72`

## Manifest Fields

Each row must include:

- `execution_manifest_id`
- `execution_order`
- `run_id`
- `run_hash`
- `candidate_id`
- `condition_id`
- `seed`
- `planned_output_root`
- `planned_command_text`
- `command_sha256`
- `manifest_only`
- `execute_allowed`
- `actual_training_allowed`
- `train_with_this_reward_allowed`
- `actual_results`
- `winner_selected`

## Execution Guards

The manifest must preserve:

- `manifest_only = true`
- `execute_allowed = false`
- `actual_training_allowed = false`
- `train_with_this_reward_allowed = false`
- `actual_results = false`
- `winner_selected = false`
- `best_reward_claim_allowed = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`

## Forbidden Tokens

Commands must not include:

- `--execute`
- `--train`
- `--promote`
- `--select-winner`
- `--allow-training`
- `--actual-results`
- `--winner-selected`

Commands must include:

- `--dry-run-plan`

## Interpretation

Step 121 produces a reproducible command manifest. It does not run reward ablation, does not select a winner, and does not allow MAPPO training.
'@

$SpecMdPath = ".\05_training\rewards\reward_ablation_execution_manifest_step121.md"
$SpecMd | Set-Content -Path $SpecMdPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 121 materializer
# ---------------------------------------------------------------------
$MaterializerPy = @'
from __future__ import annotations

import argparse
import csv
import hashlib
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


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def find_plan_csv_from_manifest(manifest: Dict[str, Any], manifest_path: Path) -> Path:
    output_files = manifest.get("output_files", {})
    candidates = [
        output_files.get("dry_run_plan_csv"),
        output_files.get("plan_csv"),
        output_files.get("reward_ablation_runner_dry_run_plan_csv"),
        output_files.get("csv"),
    ]
    for item in candidates:
        if item:
            p = Path(str(item))
            if not p.is_absolute():
                p = manifest_path.parent / p
            if p.exists():
                return p

    # Fallback: find the first likely CSV next to the manifest.
    likely = sorted(manifest_path.parent.glob("*dry_run*plan*.csv"))
    if likely:
        return likely[0]

    raise RuntimeError("could not locate Step 119 dry-run plan CSV from manifest")


def validate_source_rows(rows: List[Dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_ROW_COUNT:
        raise RuntimeError(f"expected {EXPECTED_ROW_COUNT} source rows, got {len(rows)}")

    combos = set()
    for row in rows:
        cid = str(row.get("candidate_id", "")).strip()
        cond = str(row.get("condition_id", "")).strip()
        seed = int(row.get("seed", 0))
        command = str(row.get("planned_command_text", ""))

        if cid not in EXPECTED_CANDIDATES:
            raise RuntimeError(f"unexpected candidate_id: {cid}")
        if cond not in EXPECTED_CONDITIONS:
            raise RuntimeError(f"unexpected condition_id: {cond}")
        if seed not in EXPECTED_SEEDS:
            raise RuntimeError(f"unexpected seed: {seed}")

        if "--dry-run-plan" not in command:
            raise RuntimeError(f"missing --dry-run-plan in command: {command}")

        for token in FORBIDDEN_TOKENS:
            if token in command:
                raise RuntimeError(f"forbidden token {token} in command: {command}")

        for flag in [
            "execute_allowed",
            "actual_training_allowed",
            "train_with_this_reward_allowed",
            "actual_results",
            "winner_selected",
        ]:
            if bool_value(row.get(flag, False)):
                raise RuntimeError(f"{flag} must be false in source row")

        combos.add((cid, cond, seed))

    expected = {
        (candidate, condition, seed)
        for candidate in EXPECTED_CANDIDATES
        for condition in EXPECTED_CONDITIONS
        for seed in EXPECTED_SEEDS
    }

    if combos != expected:
        missing = sorted(expected - combos)
        extra = sorted(combos - expected)
        raise RuntimeError(f"combination mismatch. missing={missing}, extra={extra}")


def materialize_manifest(source_rows: List[Dict[str, Any]], output_root: Path) -> Dict[str, Any]:
    execution_manifest_id = "reward_ablation_exec_manifest_step121_" + sha256_text(
        utc_now() + "|" + str(output_root)
    )[:12]

    ordered = sorted(
        source_rows,
        key=lambda r: (
            EXPECTED_CANDIDATES.index(str(r["candidate_id"]).strip()),
            EXPECTED_CONDITIONS.index(str(r["condition_id"]).strip()),
            int(r["seed"]),
        ),
    )

    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(ordered, start=1):
        candidate = str(row["candidate_id"]).strip()
        condition = str(row["condition_id"]).strip()
        seed = int(row["seed"])
        run_id = str(row.get("run_id") or f"{candidate}_{condition}_seed_{seed:03d}")
        command = str(row.get("planned_command_text", ""))
        planned_output_root = str(row.get("planned_output_root", ""))

        command_hash = sha256_text(command)
        run_hash = sha256_text(f"{execution_manifest_id}|{run_id}|{candidate}|{condition}|{seed}|{command}")[:16]

        rows.append({
            "execution_manifest_id": execution_manifest_id,
            "execution_order": idx,
            "run_id": run_id,
            "run_hash": run_hash,
            "candidate_id": candidate,
            "condition_id": condition,
            "seed": seed,
            "planned_output_root": planned_output_root,
            "planned_command_text": command,
            "command_sha256": command_hash,
            "manifest_only": True,
            "execute_allowed": False,
            "actual_training_allowed": False,
            "train_with_this_reward_allowed": False,
            "actual_results": False,
            "winner_selected": False,
        })

    fieldnames = [
        "execution_manifest_id",
        "execution_order",
        "run_id",
        "run_hash",
        "candidate_id",
        "condition_id",
        "seed",
        "planned_output_root",
        "planned_command_text",
        "command_sha256",
        "manifest_only",
        "execute_allowed",
        "actual_training_allowed",
        "train_with_this_reward_allowed",
        "actual_results",
        "winner_selected",
    ]

    csv_path = output_root / "reward_ablation_execution_manifest_step121.csv"
    json_path = output_root / "reward_ablation_execution_manifest_step121.json"
    manifest_path = output_root / "reward_ablation_execution_manifest_step121_manifest.json"

    write_csv_rows(csv_path, rows, fieldnames)
    dump_json(json_path, {"rows": rows})

    manifest = {
        "artifact_version": "reward_ablation_execution_manifest_step121_v1",
        "created_at_utc": utc_now(),
        "execution_manifest_id": execution_manifest_id,
        "manifest_status": "EXECUTION_MANIFEST_MATERIALIZED_NOT_EXECUTABLE",
        "row_count": len(rows),
        "candidate_count": len(EXPECTED_CANDIDATES),
        "condition_count": len(EXPECTED_CONDITIONS),
        "seed_count": len(EXPECTED_SEEDS),
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "output_files": {
            "execution_manifest_csv": str(csv_path),
            "execution_manifest_json": str(json_path),
            "execution_manifest_manifest": str(manifest_path),
        },
    }
    dump_json(manifest_path, manifest)
    return manifest


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run-manifest", required=True)
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_execution_manifest_step121")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dry_manifest_path = resolve_path(project_root, args.dry_run_manifest)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_manifest = load_json(dry_manifest_path)
    plan_csv = find_plan_csv_from_manifest(source_manifest, dry_manifest_path)
    source_rows = read_csv_rows(plan_csv)
    validate_source_rows(source_rows)
    manifest = materialize_manifest(source_rows, output_root)
    manifest["source_dry_run_manifest"] = str(dry_manifest_path)
    manifest["source_dry_run_plan_csv"] = str(plan_csv)
    dump_json(Path(manifest["output_files"]["execution_manifest_manifest"]), manifest)

    print("[OK] Step 121 reward ablation execution manifest materialized")
    print(f"[OK] manifest_status: {manifest['manifest_status']}")
    print(f"[OK] row_count      : {manifest['row_count']}")
    print(f"[OK] execute_allowed: {manifest['execute_allowed']}")
    print(f"[OK] train_allowed  : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] actual_results : {manifest['actual_results']}")
    print(f"[OK] winner_selected: {manifest['winner_selected']}")
    print(f"[OK] output_root    : {output_root}")
    print(f"[OK] manifest       : {manifest['output_files']['execution_manifest_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@

$MaterializerPath = ".\05_training\rewards\reward_ablation_execution_manifest_step121.py"
$MaterializerPy | Set-Content -Path $MaterializerPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 121 validator
# ---------------------------------------------------------------------
$ValidatorPy = @'
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
'@

$ValidatorPath = ".\05_training\rewards\validate_reward_ablation_execution_manifest_step121.py"
$ValidatorPy | Set-Content -Path $ValidatorPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 121 self-test
# ---------------------------------------------------------------------
$TestPy = @'
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def write_mock_step119_plan(root: Path) -> Path:
    candidates = ["R0", "R1", "R2", "R3", "R4", "R5"]
    conditions = ["A", "A90", "A80", "A70"]
    seeds = [1, 2, 3]

    rows = []
    for candidate in candidates:
        for condition in conditions:
            for seed in seeds:
                run_id = f"{candidate}_{condition}_seed_{seed:03d}"
                rows.append({
                    "run_id": run_id,
                    "candidate_id": candidate,
                    "condition_id": condition,
                    "seed": seed,
                    "planned_output_root": str(root / "planned_outputs" / run_id),
                    "planned_command_text": (
                        f"python 05_training/rewards/reward_ablation_result_writer_guard_step118.py "
                        f"--candidate-id {candidate} --condition-id {condition} --seed {seed} "
                        f"--output-root artifacts/rewards/ablation/{run_id} --dry-run-plan"
                    ),
                    "execute_allowed": False,
                    "actual_training_allowed": False,
                    "train_with_this_reward_allowed": False,
                    "actual_results": False,
                    "winner_selected": False,
                })

    csv_path = root / "reward_ablation_runner_dry_run_plan.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = root / "reward_ablation_runner_dry_run_manifest.json"
    manifest = {
        "artifact_version": "reward_ablation_runner_dry_run_plan_step119_v1",
        "planner_status": "DRY_RUN_ROWS_WRITTEN_NOT_EXECUTED",
        "planned_runs": len(rows),
        "execute_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "output_files": {
            "dry_run_plan_csv": str(csv_path)
        },
    }
    dump_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    tag = utc_tag()

    materializer = project_root / "05_training" / "rewards" / "reward_ablation_execution_manifest_step121.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_execution_manifest_step121.py"

    source_root = project_root / "artifacts" / "rewards" / f"step121_mock_step119_source_{tag}"
    output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_manifest_step121_selftest_{tag}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_manifest_step121_validation_{tag}"

    dry_manifest = write_mock_step119_plan(source_root)

    cmd = [
        sys.executable,
        str(materializer),
        "--dry-run-manifest",
        str(dry_manifest),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 121 materializer failed")

    manifest_path = output_root / "reward_ablation_execution_manifest_step121_manifest.json"
    if not manifest_path.exists():
        raise SystemExit("[FAIL] Step 121 manifest missing")

    val_cmd = [
        sys.executable,
        str(validator),
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(validation_root),
    ]
    val = subprocess.run(val_cmd, cwd=project_root, text=True, capture_output=True)
    print(val.stdout)
    if val.returncode != 0:
        print(val.stderr)
        raise SystemExit("[FAIL] Step 121 validator failed")

    report = load_json(validation_root / "reward_ablation_execution_manifest_step121_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_REWARD_ABLATION_EXECUTION_MANIFEST_NOT_EXECUTABLE",
        "next_status": "READY_FOR_STEP122_REWARD_ABLATION_SANDBOX_NOOP_RUNNER_GUARD",
        "row_count": 72,
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Bad case: add --execute to one command and ensure validation fails.
    bad_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_manifest_step121_bad_case_{tag}"
    shutil.copytree(output_root, bad_root)
    bad_csv = bad_root / "reward_ablation_execution_manifest_step121.csv"

    with open(bad_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["planned_command_text"] = rows[0]["planned_command_text"] + " --execute"
    with open(bad_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bad_manifest_path = bad_root / "reward_ablation_execution_manifest_step121_manifest.json"
    bad_manifest = load_json(bad_manifest_path)
    bad_manifest["output_files"]["execution_manifest_csv"] = str(bad_csv)
    dump_json(bad_manifest_path, bad_manifest)

    bad_val_root = project_root / "artifacts" / "rewards" / f"reward_ablation_execution_manifest_step121_bad_validation_{tag}"
    bad_val_cmd = [
        sys.executable,
        str(validator),
        "--manifest",
        str(bad_manifest_path),
        "--output-root",
        str(bad_val_root),
    ]
    bad_val = subprocess.run(bad_val_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_val.returncode == 0:
        print(bad_val.stdout)
        raise SystemExit("[FAIL] bad Step 121 manifest unexpectedly passed")

    print("[OK] Step 121 reward ablation execution manifest self-test PASS")
    print("[DONE] Step 121 reward ablation execution manifest complete.")


if __name__ == "__main__":
    main()
'@

$TestPath = ".\05_training\rewards\test_validate_reward_ablation_execution_manifest_step121.py"
$TestPy | Set-Content -Path $TestPath -Encoding UTF8

# ---------------------------------------------------------------------
# Run Step 121 self-test
# ---------------------------------------------------------------------
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 121 reward ablation execution manifest self-test failed"
}

Write-Host ""
Write-Host "[DONE] Step 121 reward ablation execution manifest files created and tested."
Write-Host "[NEXT] Review git status, then commit the Step 121 execution manifest files."
