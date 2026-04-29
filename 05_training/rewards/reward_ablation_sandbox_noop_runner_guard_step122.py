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
