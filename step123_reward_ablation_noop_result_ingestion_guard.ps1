$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$RewardsDir = ".\05_training\rewards"
New-Item -ItemType Directory -Force -Path $RewardsDir | Out-Null

# ---------------------------------------------------------------------
# Step 123 config JSON
# ---------------------------------------------------------------------
$SpecJson = @'
{
  "artifact_version": "reward_ablation_noop_result_ingestion_guard_step123_v1",
  "step": 123,
  "guard_status": "NOOP_RESULT_INGESTION_GUARD_NOT_ACTUAL_RESULTS",
  "purpose": "Ingest Step 122 sandbox no-op runner outputs and verify that they are complete, schema-safe, and not mistaken for actual reward ablation results.",
  "expected_candidate_ids": ["R0", "R1", "R2", "R3", "R4", "R5"],
  "expected_condition_ids": ["A", "A90", "A80", "A70"],
  "expected_seeds": [1, 2, 3],
  "expected_row_count": 72,
  "required_noop_status_values": ["NOOP_RECORDED_NOT_EXECUTED"],
  "claim_guards": {
    "actual_results": false,
    "winner_selected": false,
    "best_reward_claim_allowed": false,
    "execute_allowed": false,
    "actual_training_allowed": false,
    "train_with_this_reward_allowed": false,
    "paper_level_claim_allowed": false,
    "causal_performance_claim_allowed": false
  },
  "required_input_columns": [
    "run_id",
    "candidate_id",
    "condition_id",
    "seed",
    "noop_status",
    "execute_allowed",
    "actual_training_allowed",
    "train_with_this_reward_allowed",
    "actual_results",
    "winner_selected",
    "command_hash",
    "planned_output_root"
  ],
  "output_tables": [
    "noop_ingested_rows_csv",
    "noop_ingestion_summary_csv",
    "noop_ingestion_manifest_json"
  ],
  "next_step": "Step124 reward ablation actual result schema bridge or selection criteria gate"
}
'@
$SpecJsonPath = ".\05_training\rewards\reward_ablation_noop_result_ingestion_guard_step123.json"
$SpecJson | Set-Content -Path $SpecJsonPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 123 markdown
# ---------------------------------------------------------------------
$SpecMd = @'
# Step 123 — Reward Ablation No-op Result Ingestion Guard

## Purpose

Step 123 reads Step 122 sandbox no-op runner outputs and verifies that the result ingestion path is safe.

This step does not ingest actual reward ablation results. It only proves that the no-op output path can be read, summarized, and guarded against accidental promotion.

## Required State

- `actual_results = false`
- `winner_selected = false`
- `best_reward_claim_allowed = false`
- `execute_allowed = false`
- `actual_training_allowed = false`
- `train_with_this_reward_allowed = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`

## Expected Matrix

The no-op ingestion guard expects:

- Reward candidates: `R0`, `R1`, `R2`, `R3`, `R4`, `R5`
- Conditions: `A`, `A90`, `A80`, `A70`
- Seeds: `1`, `2`, `3`
- Total rows: `72`

## What This Step Validates

1. The Step 122 no-op output manifest is readable.
2. The no-op result CSV exists.
3. All 72 candidate-condition-seed rows exist.
4. No duplicate `(candidate_id, condition_id, seed)` rows exist.
5. All rows remain no-op only.
6. No actual result, winner, execution, or training flags are enabled.
7. The output is summarized into ingestion-safe tables.

## What This Step Does Not Do

- It does not execute MAPPO training.
- It does not run reward ablation.
- It does not select the best reward.
- It does not promote any reward candidate.
- It does not allow paper-level or causal performance claims.

## Next Step

Step 124 should either define an actual-result schema bridge or define a reward selection criteria gate. Both must keep actual promotion blocked until real ablation results exist.
'@
$SpecMdPath = ".\05_training\rewards\reward_ablation_noop_result_ingestion_guard_step123.md"
$SpecMd | Set-Content -Path $SpecMdPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 123 ingestion script
# ---------------------------------------------------------------------
$IngestPy = @'
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_SEEDS = [1, 2, 3]
EXPECTED_ROW_COUNT = len(EXPECTED_CANDIDATES) * len(EXPECTED_CONDITIONS) * len(EXPECTED_SEEDS)
REQUIRED_COLUMNS = [
    "run_id",
    "candidate_id",
    "condition_id",
    "seed",
    "noop_status",
    "execute_allowed",
    "actual_training_allowed",
    "train_with_this_reward_allowed",
    "actual_results",
    "winner_selected",
    "command_hash",
    "planned_output_root",
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
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)


def json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n", "", "none", "null"}:
        return False
    raise ValueError(f"cannot parse bool: {value!r}")


def read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read csv: {path}")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def resolve_path(project_root: Path, value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def discover_noop_csv(manifest: Dict[str, Any], manifest_path: Path) -> Path:
    output_files = manifest.get("output_files", {})
    candidates: List[Any] = []
    for key in [
        "noop_results_csv",
        "sandbox_noop_results_csv",
        "runner_noop_results_csv",
        "noop_run_results_csv",
        "result_rows_csv",
    ]:
        if key in output_files:
            candidates.append(output_files[key])

    # Fallback names under manifest directory.
    candidates.extend([
        manifest_path.parent / "reward_ablation_sandbox_noop_results.csv",
        manifest_path.parent / "sandbox_noop_results.csv",
        manifest_path.parent / "noop_results.csv",
        manifest_path.parent / "reward_ablation_noop_results.csv",
    ])

    for c in candidates:
        p = Path(c)
        if not p.is_absolute():
            p = manifest_path.parent / p
        if p.exists():
            return p

    raise FileNotFoundError("No Step 122 no-op result CSV found from manifest output_files or fallback names")


def validate_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    failures: List[str] = []

    if not rows:
        failures.append("noop result rows are empty")
        return rows, failures

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in rows[0].keys()]
    if missing_cols:
        failures.append(f"missing required columns: {missing_cols}")
        return rows, failures

    if len(rows) != EXPECTED_ROW_COUNT:
        failures.append(f"row_count expected {EXPECTED_ROW_COUNT}, got {len(rows)}")

    seen = set()
    duplicate_keys = []
    for row in rows:
        try:
            seed = int(row.get("seed", ""))
        except Exception:
            failures.append(f"invalid seed value: {row.get('seed')!r}")
            seed = -1
        key = (str(row.get("candidate_id", "")), str(row.get("condition_id", "")), seed)
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)

    if duplicate_keys:
        failures.append(f"duplicate candidate/condition/seed keys: {duplicate_keys[:5]}")

    expected_keys = {
        (candidate, condition, seed)
        for candidate in EXPECTED_CANDIDATES
        for condition in EXPECTED_CONDITIONS
        for seed in EXPECTED_SEEDS
    }
    missing_keys = sorted(expected_keys - seen)
    extra_keys = sorted(seen - expected_keys)
    if missing_keys:
        failures.append(f"missing matrix keys: {missing_keys[:8]}")
    if extra_keys:
        failures.append(f"unexpected matrix keys: {extra_keys[:8]}")

    bad_noop = [r for r in rows if str(r.get("noop_status", "")) != "NOOP_RECORDED_NOT_EXECUTED"]
    if bad_noop:
        failures.append(f"non-noop statuses found: {sorted({r.get('noop_status') for r in bad_noop})}")

    flag_columns = [
        "execute_allowed",
        "actual_training_allowed",
        "train_with_this_reward_allowed",
        "actual_results",
        "winner_selected",
    ]
    for col in flag_columns:
        bad = []
        for r in rows:
            try:
                if as_bool(r.get(col)):
                    bad.append(r)
            except Exception:
                failures.append(f"invalid boolean in column {col}: {r.get(col)!r}")
        if bad:
            failures.append(f"{col} must be false for all rows, true_count={len(bad)}")

    return rows, failures


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        candidate = str(row.get("candidate_id"))
        if candidate not in grouped:
            grouped[candidate] = {
                "candidate_id": candidate,
                "row_count": 0,
                "condition_count": set(),
                "seed_count": set(),
                "noop_rows": 0,
                "execute_allowed_any": False,
                "actual_training_allowed_any": False,
                "train_allowed_any": False,
                "actual_results_any": False,
                "winner_selected_any": False,
            }
        g = grouped[candidate]
        g["row_count"] += 1
        g["condition_count"].add(str(row.get("condition_id")))
        try:
            g["seed_count"].add(int(row.get("seed")))
        except Exception:
            pass
        if str(row.get("noop_status")) == "NOOP_RECORDED_NOT_EXECUTED":
            g["noop_rows"] += 1
        g["execute_allowed_any"] = bool(g["execute_allowed_any"] or as_bool(row.get("execute_allowed")))
        g["actual_training_allowed_any"] = bool(g["actual_training_allowed_any"] or as_bool(row.get("actual_training_allowed")))
        g["train_allowed_any"] = bool(g["train_allowed_any"] or as_bool(row.get("train_with_this_reward_allowed")))
        g["actual_results_any"] = bool(g["actual_results_any"] or as_bool(row.get("actual_results")))
        g["winner_selected_any"] = bool(g["winner_selected_any"] or as_bool(row.get("winner_selected")))

    out: List[Dict[str, Any]] = []
    for candidate in EXPECTED_CANDIDATES:
        g = grouped.get(candidate, {
            "candidate_id": candidate,
            "row_count": 0,
            "condition_count": set(),
            "seed_count": set(),
            "noop_rows": 0,
            "execute_allowed_any": False,
            "actual_training_allowed_any": False,
            "train_allowed_any": False,
            "actual_results_any": False,
            "winner_selected_any": False,
        })
        out.append({
            "candidate_id": candidate,
            "row_count": int(g["row_count"]),
            "condition_count": int(len(g["condition_count"])),
            "seed_count": int(len(g["seed_count"])),
            "noop_rows": int(g["noop_rows"]),
            "execute_allowed_any": bool(g["execute_allowed_any"]),
            "actual_training_allowed_any": bool(g["actual_training_allowed_any"]),
            "train_allowed_any": bool(g["train_allowed_any"]),
            "actual_results_any": bool(g["actual_results_any"]),
            "winner_selected_any": bool(g["winner_selected_any"]),
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noop-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    noop_manifest_path = resolve_path(project_root, args.noop_manifest)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    noop_manifest = load_json_any_encoding(noop_manifest_path)
    noop_csv = discover_noop_csv(noop_manifest, noop_manifest_path)
    rows = read_csv_dicts(noop_csv)
    rows, failures = validate_rows(rows)
    summary_rows = summarize(rows)

    ingested_csv = output_root / "reward_ablation_noop_ingested_rows_step123.csv"
    summary_csv = output_root / "reward_ablation_noop_ingestion_summary_step123.csv"
    manifest_json = output_root / "reward_ablation_noop_result_ingestion_guard_step123_manifest.json"

    write_csv(ingested_csv, rows, REQUIRED_COLUMNS)
    write_csv(summary_csv, summary_rows, [
        "candidate_id",
        "row_count",
        "condition_count",
        "seed_count",
        "noop_rows",
        "execute_allowed_any",
        "actual_training_allowed_any",
        "train_allowed_any",
        "actual_results_any",
        "winner_selected_any",
    ])

    audit_status = "PASS" if not failures else "FAIL"
    manifest = {
        "artifact_version": "reward_ablation_noop_result_ingestion_guard_step123_manifest_v1",
        "created_at_utc": utc_now(),
        "audit_status": audit_status,
        "guard_status": "PASS_NOOP_RESULT_INGESTION_GUARD_NOT_ACTUAL_RESULTS" if audit_status == "PASS" else "FAIL_NOOP_RESULT_INGESTION_GUARD",
        "next_status": "READY_FOR_STEP124_REWARD_ABLATION_SELECTION_CRITERIA_GATE" if audit_status == "PASS" else "BLOCKED_FIX_STEP123_INGESTION",
        "source_noop_manifest": str(noop_manifest_path),
        "source_noop_csv": str(noop_csv),
        "output_root": str(output_root),
        "row_count": int(len(rows)),
        "candidate_count": int(len({r.get('candidate_id') for r in rows})),
        "condition_count": int(len({r.get('condition_id') for r in rows})),
        "seed_count": int(len({str(r.get('seed')) for r in rows})),
        "actual_results": False,
        "winner_selected": False,
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "best_reward_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "failure_count": int(len(failures)),
        "failures": failures,
        "output_files": {
            "noop_ingested_rows_csv": str(ingested_csv),
            "noop_ingestion_summary_csv": str(summary_csv),
            "noop_ingestion_manifest_json": str(manifest_json),
        },
    }
    dump_json(manifest_json, manifest)

    print("[OK] Step 123 reward ablation no-op result ingestion guard completed")
    print(f"[OK] audit_status  : {audit_status}")
    print(f"[OK] guard_status  : {manifest['guard_status']}")
    print(f"[OK] next_status   : {manifest['next_status']}")
    print(f"[OK] row_count     : {manifest['row_count']}")
    print(f"[OK] actual_results: {manifest['actual_results']}")
    print(f"[OK] winner_selected: {manifest['winner_selected']}")
    print(f"[OK] train_allowed : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] failure_count : {manifest['failure_count']}")
    print(f"[OK] manifest      : {manifest_json}")

    return 0 if audit_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
'@
$IngestPyPath = ".\05_training\rewards\reward_ablation_noop_result_ingestion_guard_step123.py"
$IngestPy | Set-Content -Path $IngestPyPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 123 validator
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
'@
$ValidatorPath = ".\05_training\rewards\validate_reward_ablation_noop_result_ingestion_guard_step123.py"
$ValidatorPy | Set-Content -Path $ValidatorPath -Encoding UTF8

# ---------------------------------------------------------------------
# Step 123 self-test
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


def unique_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f") + f"_{os.getpid()}"


def load_json(path: Path):
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_mock_step122_noop_output(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "reward_ablation_sandbox_noop_results.csv"
    manifest_path = root / "reward_ablation_sandbox_noop_runner_guard_step122_manifest.json"

    fieldnames = [
        "run_id",
        "candidate_id",
        "condition_id",
        "seed",
        "noop_status",
        "execute_allowed",
        "actual_training_allowed",
        "train_with_this_reward_allowed",
        "actual_results",
        "winner_selected",
        "command_hash",
        "planned_output_root",
    ]

    rows = []
    for candidate in CANDIDATES:
        for condition in CONDITIONS:
            for seed in SEEDS:
                run_id = f"{candidate}_{condition}_seed{seed:03d}"
                rows.append({
                    "run_id": run_id,
                    "candidate_id": candidate,
                    "condition_id": condition,
                    "seed": seed,
                    "noop_status": "NOOP_RECORDED_NOT_EXECUTED",
                    "execute_allowed": False,
                    "actual_training_allowed": False,
                    "train_with_this_reward_allowed": False,
                    "actual_results": False,
                    "winner_selected": False,
                    "command_hash": f"hash_{run_id}",
                    "planned_output_root": str(root / "planned" / run_id),
                })

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "artifact_version": "mock_step122_noop_runner_manifest_v1",
        "audit_status": "PASS",
        "row_count": 72,
        "actual_results": False,
        "winner_selected": False,
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "output_files": {
            "noop_results_csv": str(csv_path)
        }
    }
    dump_json(manifest_path, manifest)
    return manifest_path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    tag = unique_tag()
    source_root = project_root / "artifacts" / "rewards" / f"step123_mock_step122_noop_source_{tag}"
    output_root = project_root / "artifacts" / "rewards" / f"reward_ablation_noop_ingestion_step123_selftest_{tag}"
    validation_root = project_root / "artifacts" / "rewards" / f"reward_ablation_noop_ingestion_step123_validation_{tag}"

    ingest = project_root / "05_training" / "rewards" / "reward_ablation_noop_result_ingestion_guard_step123.py"
    validator = project_root / "05_training" / "rewards" / "validate_reward_ablation_noop_result_ingestion_guard_step123.py"

    noop_manifest = write_mock_step122_noop_output(source_root)

    cmd = [
        sys.executable,
        str(ingest),
        "--noop-manifest",
        str(noop_manifest),
        "--output-root",
        str(output_root),
    ]
    result = subprocess.run(cmd, cwd=project_root, text=True, capture_output=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("[FAIL] Step 123 ingestion script failed")

    ingestion_manifest = output_root / "reward_ablation_noop_result_ingestion_guard_step123_manifest.json"
    if not ingestion_manifest.exists():
        raise SystemExit("[FAIL] ingestion manifest missing")

    vcmd = [
        sys.executable,
        str(validator),
        "--ingestion-manifest",
        str(ingestion_manifest),
        "--output-root",
        str(validation_root),
    ]
    vresult = subprocess.run(vcmd, cwd=project_root, text=True, capture_output=True)
    print(vresult.stdout)
    if vresult.returncode != 0:
        print(vresult.stderr)
        raise SystemExit("[FAIL] Step 123 validator failed")

    report = load_json(validation_root / "reward_ablation_noop_result_ingestion_guard_step123_validation_report.json")
    expected = {
        "audit_status": "PASS",
        "gate_status": "PASS_NOOP_RESULT_INGESTION_GUARD_NOT_ACTUAL_RESULTS",
        "next_status": "READY_FOR_STEP124_REWARD_ABLATION_SELECTION_CRITERIA_GATE",
        "row_count": 72,
        "actual_results": False,
        "winner_selected": False,
        "train_with_this_reward_allowed": False,
        "failure_count": 0,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if actual != value:
            raise SystemExit(f"[FAIL] {key}: expected={value!r}, actual={actual!r}")

    # Negative test: a corrupted actual_results=true row must fail ingestion validation.
    bad_root = project_root / "artifacts" / "rewards" / f"step123_bad_step122_noop_source_{tag}"
    bad_manifest = write_mock_step122_noop_output(bad_root)
    bad_payload = load_json(bad_manifest)
    bad_csv = Path(bad_payload["output_files"]["noop_results_csv"])
    with open(bad_csv, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0]["actual_results"] = "True"
    with open(bad_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bad_output = project_root / "artifacts" / "rewards" / f"reward_ablation_noop_ingestion_step123_bad_case_{tag}"
    bad_cmd = [
        sys.executable,
        str(ingest),
        "--noop-manifest",
        str(bad_manifest),
        "--output-root",
        str(bad_output),
    ]
    bad_result = subprocess.run(bad_cmd, cwd=project_root, text=True, capture_output=True)
    if bad_result.returncode == 0:
        print(bad_result.stdout)
        raise SystemExit("[FAIL] corrupted actual_results row unexpectedly passed ingestion")

    print("[OK] Step 123 reward ablation no-op result ingestion guard self-test PASS")
    print("[DONE] Step 123 reward ablation no-op result ingestion guard complete.")


if __name__ == "__main__":
    main()
'@
$TestPath = ".\05_training\rewards\test_validate_reward_ablation_noop_result_ingestion_guard_step123.py"
$TestPy | Set-Content -Path $TestPath -Encoding UTF8

# ---------------------------------------------------------------------
# Run Step 123 self-test
# ---------------------------------------------------------------------
if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py $TestPath

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 123 reward ablation no-op result ingestion guard self-test failed"
}

Write-Host ""
Write-Host "[DONE] Step 123 reward ablation no-op result ingestion guard files created and tested."
Write-Host "[NEXT] Review git status, then commit the Step 123 no-op ingestion guard files."
