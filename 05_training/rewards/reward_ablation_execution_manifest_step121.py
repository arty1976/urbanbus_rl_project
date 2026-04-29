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
