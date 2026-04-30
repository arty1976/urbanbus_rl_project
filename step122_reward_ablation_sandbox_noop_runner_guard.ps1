$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

# ---------------------------------------------------------------------
# Step 122 guard spec JSON
# ---------------------------------------------------------------------
$SpecJson = @'
{
  "artifact_version": "reward_ablation_sandbox_noop_runner_guard_step122_v1",
  "step": 122,
  "guard_status": "SANDBOX_NOOP_RUNNER_GUARD_NOT_EXECUTING",
  "purpose": "Read the Step 121 reward ablation execution manifest and write no-op sandbox status rows without executing training commands.",
  "source_steps": [
    "Step119 reward ablation runner dry-run plan",
    "Step120 reward ablation execution preflight",
    "Step121 reward ablation execution manifest"
  ],
  "expected_candidates": ["R0", "R1", "R2", "R3", "R4", "R5"],
  "expected_conditions": ["A", "A90", "A80", "A70"],
  "expected_seeds": [1, 2, 3],
  "expected_row_count": 72,
  "runner_policy": {
    "sandbox_noop_only": true,
    "execute_commands": false,
    "spawn_processes": false,
    "write_status_rows_only": true,
    "no_training_side_effects": true
  },
  "prohibited_command_tokens": [
    "--execute",
    "--train",
    "--promote",
    "--select-winner",
    "--allow-training"
  ],
  "claim_guards": {
    "execute_allowed": false,
    "actual_training_allowed": false,
    "train_with_this_reward_allowed": false,
    "actual_results": false,
    "winner_selected": false,
    "best_reward_claim_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false
  },
  "next_status": "READY_FOR_STEP123_REWARD_ABLATION_RESULT_INGESTION_GUARD"
}
'@

$SpecJsonPath = ".\05_training\rewards\reward_ablation_sandbox_noop_runner_guard_step122.json"
$SpecJson | Set-Content -Path $SpecJsonPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 122 markdown document
# ---------------------------------------------------------------------
$SpecMd = @'
# Step 122 — Reward Ablation Sandbox No-op Runner Guard

## Purpose

Step 122 reads the Step 121 reward ablation execution manifest and records sandbox no-op status rows.

It does **not** execute reward ablation, MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) training, or any command in the manifest.

## Required Input

The input is a Step 121 execution manifest containing 72 planned rows:

```text
R0~R5 reward candidates × A/A90/A80/A70 conditions × seeds 1/2/3 = 72 rows
```

## Output

The runner writes no-op status artifacts:

```text
reward_ablation_sandbox_noop_status.csv
reward_ablation_sandbox_noop_status.json
reward_ablation_sandbox_noop_runner_guard_manifest.json
```

## Guarded Status

The following must remain false:

- `execute_allowed`
- `actual_training_allowed`
- `train_with_this_reward_allowed`
- `actual_results`
- `winner_selected`
- `best_reward_claim_allowed`
- `paper_level_claim_allowed`
- `causal_performance_claim_allowed`

## Prohibited Command Tokens

The no-op runner must reject any manifest command containing:

- `--execute`
- `--train`
- `--promote`
- `--select-winner`
- `--allow-training`

## Interpretation

Step 122 is a runner shell validation step. It proves that a 72-row manifest can be read and converted into status rows without accidentally executing anything.

It is not an execution step and does not produce actual ablation results.
'@

$SpecMdPath = ".\05_training\rewards\reward_ablation_sandbox_noop_runner_guard_step122.md"
$SpecMd | Set-Content -Path $SpecMdPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 122 runner
# ---------------------------------------------------------------------
$RunnerPy = @'
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_SEEDS = [1, 2, 3]
EXPECTED_ROW_COUNT = len(EXPECTED_CANDIDATES) * len(EXPECTED_CONDITIONS) * len(EXPECTED_SEEDS)
PROHIBITED_COMMAND_TOKENS = ["--execute", "--train", "--promote", "--select-winner", "--allow-training"]


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


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("cannot write empty csv")
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def get_first(row: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def resolve_manifest_rows(manifest_path: Path) -> List[Dict[str, Any]]:
    if manifest_path.suffix.lower() == ".csv":
        return read_csv_rows(manifest_path)

    payload = load_json_any_encoding(manifest_path)

    for key in ("rows", "manifest_rows", "execution_rows", "planned_rows", "dry_run_rows"):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(x) for x in value]

    output_files = payload.get("output_files", {})
    candidate_paths: List[Path] = []
    for key in (
        "execution_manifest_csv",
        "reward_ablation_execution_manifest_csv",
        "dry_run_plan_csv",
        "reward_ablation_runner_dry_run_plan_csv",
        "plan_csv",
    ):
        value = output_files.get(key)
        if value:
            candidate_paths.append(Path(value))

    for key in (
        "execution_manifest_json",
        "reward_ablation_execution_manifest_json",
        "dry_run_plan_json",
        "reward_ablation_runner_dry_run_plan_json",
        "plan_json",
    ):
        value = output_files.get(key)
        if value:
            p = Path(value)
            if p.exists():
                nested = load_json_any_encoding(p)
                for rows_key in ("rows", "manifest_rows", "execution_rows", "planned_rows", "dry_run_rows"):
                    rows = nested.get(rows_key)
                    if isinstance(rows, list):
                        return [dict(x) for x in rows]

    for p in candidate_paths:
        if p.exists():
            return read_csv_rows(p)

    raise RuntimeError(f"could not resolve execution rows from manifest: {manifest_path}")


def validate_input_rows(rows: List[Dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_ROW_COUNT:
        raise RuntimeError(f"expected {EXPECTED_ROW_COUNT} rows, got {len(rows)}")

    combos = set()
    for i, row in enumerate(rows):
        candidate_id = str(get_first(row, ["candidate_id", "reward_candidate_id"])).strip()
        condition_id = str(get_first(row, ["condition_id"])).strip().upper()
        seed = int(get_first(row, ["seed", "training_seed"]))
        command_text = str(get_first(row, ["planned_command_text", "command_text", "dry_run_command", "command"], ""))

        if candidate_id not in EXPECTED_CANDIDATES:
            raise RuntimeError(f"invalid candidate_id at row {i}: {candidate_id}")
        if condition_id not in EXPECTED_CONDITIONS:
            raise RuntimeError(f"invalid condition_id at row {i}: {condition_id}")
        if seed not in EXPECTED_SEEDS:
            raise RuntimeError(f"invalid seed at row {i}: {seed}")
        if not command_text:
            raise RuntimeError(f"missing command text at row {i}")
        if "--dry-run-plan" not in command_text and "--dry-run" not in command_text:
            raise RuntimeError(f"row {i} command must contain --dry-run-plan or --dry-run")
        for token in PROHIBITED_COMMAND_TOKENS:
            if token in command_text:
                raise RuntimeError(f"row {i} command contains prohibited token: {token}")

        for flag_key in ("execute_allowed", "actual_training_allowed", "train_with_this_reward_allowed"):
            if as_bool(row.get(flag_key, False)):
                raise RuntimeError(f"row {i} has unsafe true flag: {flag_key}")

        combos.add((candidate_id, condition_id, seed))

    expected = {
        (candidate_id, condition_id, seed)
        for candidate_id in EXPECTED_CANDIDATES
        for condition_id in EXPECTED_CONDITIONS
        for seed in EXPECTED_SEEDS
    }
    if combos != expected:
        missing = sorted(expected - combos)
        extra = sorted(combos - expected)
        raise RuntimeError(f"combo mismatch. missing={missing[:5]}, extra={extra[:5]}")


def build_status_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        candidate_id = str(get_first(row, ["candidate_id", "reward_candidate_id"])).strip()
        condition_id = str(get_first(row, ["condition_id"])).strip().upper()
        seed = int(get_first(row, ["seed", "training_seed"]))
        command_text = str(get_first(row, ["planned_command_text", "command_text", "dry_run_command", "command"], ""))
        command_hash = str(get_first(row, ["command_hash", "planned_command_hash"], stable_hash(command_text)))
        run_id = str(get_first(row, ["manifest_run_id", "run_id"], f"{candidate_id}_{condition_id}_seed{seed:03d}"))
        planned_output_root = str(get_first(row, ["planned_output_root", "output_root"], ""))

        out.append({
            "sandbox_status": "NOOP_RECORDED_NOT_EXECUTED",
            "row_index": idx,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "condition_id": condition_id,
            "seed": seed,
            "command_hash": command_hash,
            "planned_output_root": planned_output_root,
            "planned_command_text": command_text,
            "command_executed": False,
            "process_spawned": False,
            "exit_code": "",
            "execute_allowed": False,
            "actual_training_allowed": False,
            "train_with_this_reward_allowed": False,
            "actual_results": False,
            "winner_selected": False,
            "best_reward_claim_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "created_at_utc": utc_now(),
        })

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--output-root",
        default="artifacts/rewards/reward_ablation_sandbox_noop_runner_guard_step122",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    manifest_path = resolve_path(project_root, args.manifest)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    rows = resolve_manifest_rows(manifest_path)
    validate_input_rows(rows)
    status_rows = build_status_rows(rows)

    status_csv = output_root / "reward_ablation_sandbox_noop_status.csv"
    status_json = output_root / "reward_ablation_sandbox_noop_status.json"
    manifest_json = output_root / "reward_ablation_sandbox_noop_runner_guard_manifest.json"

    write_csv(status_csv, status_rows)
    dump_json(status_json, {"rows": status_rows})

    manifest = {
        "artifact_version": "reward_ablation_sandbox_noop_runner_guard_step122_manifest_v1",
        "created_at_utc": utc_now(),
        "runner_status": "SANDBOX_NOOP_STATUS_WRITTEN_NOT_EXECUTED",
        "source_manifest": str(manifest_path),
        "output_root": str(output_root),
        "candidate_count": len(EXPECTED_CANDIDATES),
        "condition_count": len(EXPECTED_CONDITIONS),
        "seed_count": len(EXPECTED_SEEDS),
        "row_count": len(status_rows),
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "best_reward_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "output_files": {
            "status_csv": str(status_csv),
            "status_json": str(status_json),
            "manifest_json": str(manifest_json)
        },
    }
    dump_json(manifest_json, manifest)

    print("[OK] Step 122 reward ablation sandbox no-op runner guard completed")
    print("[OK] runner_status : SANDBOX_NOOP_STATUS_WRITTEN_NOT_EXECUTED")
    print(f"[OK] row_count     : {len(status_rows)}")
    print("[OK] command_exec  : False")
    print("[OK] process_spawn : False")
    print("[OK] execute_allowed: False")
    print("[OK] train_allowed : False")
    print(f"[OK] output_root   : {output_root}")
    print(f"[OK] manifest      : {manifest_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'@

$RunnerPath = ".\05_training\rewards\reward_ablation_sandbox_noop_runner_guard_step122.py"
$RunnerPy | Set-Content -Path $RunnerPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 122 validator
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
'@

$ValidatorPath = ".\05_training\rewards\validate_reward_ablation_sandbox_noop_runner_guard_step122.py"
$ValidatorPy | Set-Content -Path $ValidatorPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 122 self-test
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


CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
CONDITIONS = ["A", "A90", "A80", "A70"]
SEEDS = [1, 2, 3]


def unique_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_mock_step121_manifest(project_root: Path, root: Path, unsafe: bool = False) -> Path:
    rows = []
    for candidate_id in CANDIDATES:
        for condition_id in CONDITIONS:
            for seed in SEEDS:
                command = (
                    "python 05_training/rewards/reward_ablation_result_writer_guard_step118.py "
                    f"--candidate-id {candidate_id} --condition-id {condition_id} --seed {seed} --dry-run-plan"
                )
                if unsafe and candidate_id == "R0" and condition_id == "A" and seed == 1:
                    command += " --execute"
                rows.append({
                    "manifest_run_id": f"{candidate_id}_{condition_id}_seed{seed:03d}",
                    "candidate_id": candidate_id,
                    "condition_id": condition_id,
                    "seed": seed,
                    "planned_output_root": str(root / "planned" / candidate_id / condition_id / f"seed_{seed:03d}"),
                    "planned_command_text": command,
                    "command_hash": f"hash_{candidate_id}_{condition_id}_{seed}",
                    "execute_allowed": False,
                    "actual_training_allowed": False,
                    "train_with_this_reward_allowed": False,
                    "actual_results": False,
                    "winner_selected": False,
                })

    plan_csv = root / "reward_ablation_execution_manifest_rows.csv"
    write_csv(plan_csv, rows)
    manifest = {
        "artifact_version": "mock_step121_execution_manifest_v1",
        "row_count": len(rows),
        "output_files": {
            "execution_manifest_csv": str(plan_csv),
        },
    }
    manifest_path = root / "reward_ablation_execution_manifest_step121_manifest.json"
    dump_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    runner = project_root / "05_training" / "rewards" / "reward_ablation_sandbox_noop_runner_guard_step122.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_sandbox_noop_runner_guard_step122.py"
    tag = unique_tag()

    source_root = project_root / "artifacts" / "rewards" / f"step122_mock_step121_source_{tag}"
    output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_sandbox_noop_runner_guard_step122_selftest_{tag}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_sandbox_noop_runner_guard_step122_validation_{tag}"

    manifest_path = write_mock_step121_manifest(project_root, source_root)

    run_cmd = [
        sys.executable,
        str(runner),
        "--manifest",
        str(manifest_path),
        "--output-root",
        str(output_root),
    ]
    run_result = subprocess.run(run_cmd, cwd=project_root, text=True, capture_output=True)
    print(run_result.stdout)
    if run_result.returncode != 0:
        print(run_result.stderr)
        raise SystemExit("[FAIL] Step 122 runner failed")

    out_manifest = output_root / "reward_ablation_sandbox_noop_runner_guard_manifest.json"
    if not out_manifest.exists():
        raise SystemExit("[FAIL] Step 122 output manifest missing")

    val_cmd = [
        sys.executable,
        str(validator),
        "--manifest",
        str(out_manifest),
        "--output-root",
        str(validation_root),
    ]
    val_result = subprocess.run(val_cmd, cwd=project_root, text=True, capture_output=True)
    print(val_result.stdout)
    if val_result.returncode != 0:
        print(val_result.stderr)
        raise SystemExit("[FAIL] Step 122 validator unexpectedly failed")

    report = load_json(validation_root / "reward_ablation_sandbox_noop_runner_guard_step122_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_REWARD_ABLATION_SANDBOX_NOOP_RUNNER_GUARD_NOT_EXECUTED",
        "next_status": "READY_FOR_STEP123_REWARD_ABLATION_RESULT_INGESTION_GUARD",
        "row_count": 72,
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={report.get(key)!r}")

    # Negative input test: unsafe command with --execute must be rejected by runner.
    bad_source_root = project_root / "artifacts" / "rewards" / f"step122_bad_input_source_{tag}"
    bad_manifest = write_mock_step121_manifest(project_root, bad_source_root, unsafe=True)
    bad_result = subprocess.run([
        sys.executable,
        str(runner),
        "--manifest",
        str(bad_manifest),
        "--output-root",
        str(project_root / "artifacts" / "rewards" / f"step122_bad_input_output_{tag}"),
    ], cwd=project_root, text=True, capture_output=True)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] unsafe --execute command unexpectedly passed runner")

    # Negative output test: if a status row says command_executed=True, validator must fail.
    bad_output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_sandbox_noop_runner_guard_step122_bad_output_{tag}"
    shutil.copytree(output_root, bad_output_root)
    bad_manifest_path = bad_output_root / "reward_ablation_sandbox_noop_runner_guard_manifest.json"
    bad_manifest_payload = load_json(bad_manifest_path)
    bad_status_csv = bad_output_root / "reward_ablation_sandbox_noop_status.csv"
    rows = read_csv(bad_status_csv)
    rows[0]["command_executed"] = "True"
    write_csv(bad_status_csv, rows)
    bad_manifest_payload["output_root"] = str(bad_output_root)
    bad_manifest_payload["output_files"]["status_csv"] = str(bad_status_csv)
    bad_manifest_payload["output_files"]["status_json"] = str(bad_output_root / "reward_ablation_sandbox_noop_status.json")
    bad_manifest_payload["output_files"]["manifest_json"] = str(bad_manifest_path)
    dump_json(bad_manifest_path, bad_manifest_payload)

    bad_val_root = project_root / "artifacts" / "rewards" / f"reward_ablation_sandbox_noop_runner_guard_step122_bad_validation_{tag}"
    bad_val_result = subprocess.run([
        sys.executable,
        str(validator),
        "--manifest",
        str(bad_manifest_path),
        "--output-root",
        str(bad_val_root),
    ], cwd=project_root, text=True, capture_output=True)
    if bad_val_result.returncode == 0:
        print(bad_val_result.stdout)
        raise SystemExit("[FAIL] corrupted command_executed=True output unexpectedly passed")

    bad_report = load_json(bad_val_root / "reward_ablation_sandbox_noop_runner_guard_step122_validation_report.json")
    failed_ids = {
        c["check_id"]
        for c in bad_report.get("checks", [])
        if not c.get("passed", False)
    }
    if "all_command_not_executed" not in failed_ids:
        raise SystemExit("[FAIL] command_executed corruption did not trigger all_command_not_executed")

    print("[OK] Step 122 reward ablation sandbox no-op runner guard self-test PASS")
    print("[DONE] Step 122 reward ablation sandbox no-op runner guard complete.")


if __name__ == "__main__":
    main()
'@

$TestPath = ".\05_training\rewards\test_validate_reward_ablation_sandbox_noop_runner_guard_step122.py"
$TestPy | Set-Content -Path $TestPath -Encoding UTF8

# ---------------------------------------------------------------------
# Run self-test
# ---------------------------------------------------------------------
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 122 reward ablation sandbox no-op runner guard self-test failed"
}

Write-Host ""
Write-Host "[DONE] Step 122 reward ablation sandbox no-op runner guard files created and tested."
Write-Host "[NEXT] Review git status, then commit the Step 122 sandbox no-op runner guard files."
