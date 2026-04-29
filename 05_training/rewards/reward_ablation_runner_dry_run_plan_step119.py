from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ARTIFACT_VERSION = "reward_ablation_runner_dry_run_plan_step119_v1"
EXPECTED_CANDIDATES = ["R0", "R1", "R2", "R3", "R4", "R5"]
EXPECTED_CONDITIONS = ["A", "A90", "A80", "A70"]
EXPECTED_SEEDS = [1, 2, 3]


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


def resolve_path(project_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return project_root / p


def build_rows(plan: Dict[str, Any], project_root: Path, output_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    entrypoint = str(plan.get("command_contract", {}).get("entrypoint", ""))

    for candidate_id in plan.get("candidate_ids", []):
        for condition_id in plan.get("condition_ids", []):
            for seed in plan.get("seeds", []):
                run_id = f"step119_{candidate_id}_{condition_id}_seed_{int(seed):03d}"
                planned_output_root = output_root / "planned_runs" / candidate_id / condition_id / f"seed_{int(seed):03d}"
                command = [
                    "python",
                    entrypoint,
                    "--candidate-id",
                    str(candidate_id),
                    "--condition-id",
                    str(condition_id),
                    "--seed",
                    str(int(seed)),
                    "--output-root",
                    str(planned_output_root),
                    "--dry-run-plan",
                ]
                rows.append({
                    "run_id": run_id,
                    "candidate_id": str(candidate_id),
                    "condition_id": str(condition_id),
                    "seed": int(seed),
                    "command_type": "dry_run_plan_only",
                    "execute_allowed": False,
                    "actual_training_allowed": False,
                    "train_with_this_reward_allowed": False,
                    "actual_results": False,
                    "winner_selected": False,
                    "planned_output_root": str(planned_output_root),
                    "planned_command_json": json.dumps(command, ensure_ascii=False),
                    "planned_command_text": " ".join(command),
                })

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("no rows to write")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="05_training/rewards/reward_ablation_runner_dry_run_plan_step119.json")
    parser.add_argument("--output-root", default="artifacts/rewards/reward_ablation_runner_dry_run_plan_step119_selftest")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    plan_path = resolve_path(project_root, args.plan)
    output_root = resolve_path(project_root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not plan_path.exists():
        raise SystemExit(f"plan not found: {plan_path}")

    plan = load_json_any_encoding(plan_path)
    rows = build_rows(plan, project_root, output_root)

    plan_payload = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "planner_status": "DRY_RUN_ROWS_WRITTEN_NOT_EXECUTED",
        "source_plan": str(plan_path),
        "candidate_ids": plan.get("candidate_ids", []),
        "condition_ids": plan.get("condition_ids", []),
        "seeds": plan.get("seeds", []),
        "planned_run_count": len(rows),
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "best_reward_claim_allowed": False,
        "rows": rows,
    }

    plan_json_path = output_root / "reward_ablation_runner_dry_run_plan.json"
    plan_csv_path = output_root / "reward_ablation_runner_dry_run_plan.csv"
    manifest_path = output_root / "reward_ablation_runner_dry_run_manifest.json"

    dump_json(plan_json_path, plan_payload)
    write_csv(plan_csv_path, rows)

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": utc_now(),
        "writer_status": "DRY_RUN_PLAN_WRITTEN_NOT_EXECUTED",
        "output_root": str(output_root),
        "output_files": {
            "plan_json": str(plan_json_path),
            "plan_csv": str(plan_csv_path),
            "manifest_json": str(manifest_path),
        },
        "candidate_count": len(plan.get("candidate_ids", [])),
        "condition_count": len(plan.get("condition_ids", [])),
        "seed_count": len(plan.get("seeds", [])),
        "planned_run_count": len(rows),
        "execute_allowed": False,
        "actual_training_allowed": False,
        "train_with_this_reward_allowed": False,
        "actual_results": False,
        "winner_selected": False,
    }
    dump_json(manifest_path, manifest)

    print("[OK] Step 119 reward ablation runner dry-run plan written")
    print(f"[OK] planner_status : {plan_payload['planner_status']}")
    print(f"[OK] candidate_count: {manifest['candidate_count']}")
    print(f"[OK] condition_count: {manifest['condition_count']}")
    print(f"[OK] seed_count     : {manifest['seed_count']}")
    print(f"[OK] planned_runs   : {manifest['planned_run_count']}")
    print(f"[OK] execute_allowed: {manifest['execute_allowed']}")
    print(f"[OK] train_allowed  : {manifest['train_with_this_reward_allowed']}")
    print(f"[OK] output_root    : {output_root}")
    print(f"[OK] manifest       : {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
