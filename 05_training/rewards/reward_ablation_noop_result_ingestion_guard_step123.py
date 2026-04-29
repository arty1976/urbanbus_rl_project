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
