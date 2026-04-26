"""
run_a_family_bridge_smoke_v1.py
===============================

Smoke runner for A/A90/A80/A70 bridge integration.

This runner does not run the causal simulator.
It verifies that the Step 25 bridge can generate extended window_rollup rows
for all A-family conditions.

It checks:
- A/A90/A80/A70 use the same passenger_demand_generated
- A90/A80/A70 reduce active_bus_count only
- qwen_trigger_rate remains 0.0
- reward fields are attached
- rollout schema validation passes
- placeholder and actual MAPPO source_mode remain separated

MAPPO means Multi-Agent Proximal Policy Optimization.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


THIS = Path(__file__).resolve()
TRAINING_DIR = THIS.parents[1]
ROLLOUTS_DIR = TRAINING_DIR / "rollouts"
POLICIES_DIR = TRAINING_DIR / "policies"
REWARDS_DIR = TRAINING_DIR / "rewards"

for p in (ROLLOUTS_DIR, POLICIES_DIR, REWARDS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


from a_family_rollout_bridge_v1 import (  # noqa: E402
    build_a_family_rollout_row,
    sample_base_row,
)
from rollout_schema_v1 import (  # noqa: E402
    validate_rollout_rows,
    validate_same_demand_by_window,
)


A_FAMILY = ["A", "A90", "A80", "A70"]


def make_rows(
    policy_kind: str,
    checkpoint_path: Optional[str],
    require_existing_checkpoint: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = sample_base_row()

    for condition_id in A_FAMILY:
        row = build_a_family_rollout_row(
            base,
            condition_id=condition_id,
            policy_kind=policy_kind,
            checkpoint_path=checkpoint_path,
            require_existing_checkpoint=require_existing_checkpoint,
            strict_claim_validation=(policy_kind == "mappo"),
        )
        rows.append(row)

    return rows


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    schema_summary = validate_rollout_rows(rows, strict=True)
    demand_summary = validate_same_demand_by_window(rows)

    fleet = {
        row["condition_id"]: {
            "baseline_bus_count": row.get("baseline_bus_count"),
            "active_bus_count": row.get("active_bus_count"),
            "fleet_reduction_ratio": row.get("fleet_reduction_ratio"),
        }
        for row in rows
    }

    source_modes = {
        row["condition_id"]: row.get("source_mode")
        for row in rows
    }

    qwen_rates = {
        row["condition_id"]: row.get("qwen_trigger_rate")
        for row in rows
    }

    reward_total = {
        row["condition_id"]: row.get("reward_total")
        for row in rows
    }

    passed = bool(schema_summary["passed"]) and bool(demand_summary["passed"])

    return {
        "artifact_version": "a_family_bridge_smoke_v1",
        "condition_ids": [row["condition_id"] for row in rows],
        "row_count": len(rows),
        "passed": passed,
        "schema_summary": schema_summary,
        "demand_fairness_summary": demand_summary,
        "fleet": fleet,
        "source_modes": source_modes,
        "qwen_trigger_rates": qwen_rates,
        "reward_total": reward_total,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            safe_row = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    safe_row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    safe_row[key] = value
            writer.writerow(safe_row)


def try_write_parquet(path: Path, rows: List[Dict[str, Any]]) -> bool:
    try:
        import pandas as pd
    except Exception:
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def run_smoke(
    output_root: Path,
    policy_kind: str,
    checkpoint_path: Optional[str],
    require_existing_checkpoint: bool,
    write_parquet: bool,
) -> Dict[str, Any]:
    rows = make_rows(
        policy_kind=policy_kind,
        checkpoint_path=checkpoint_path,
        require_existing_checkpoint=require_existing_checkpoint,
    )
    summary = summarize_rows(rows)

    output_root.mkdir(parents=True, exist_ok=True)

    rows_jsonl = output_root / "a_family_rows.jsonl"
    rows_csv = output_root / "a_family_rows.csv"
    summary_json = output_root / "summary.json"
    rows_parquet = output_root / "a_family_rows.parquet"

    write_jsonl(rows_jsonl, rows)
    write_csv(rows_csv, rows)
    write_json(summary_json, summary)

    parquet_written = False
    if write_parquet:
        parquet_written = try_write_parquet(rows_parquet, rows)

    manifest = {
        "artifact_version": "a_family_bridge_smoke_manifest_v1",
        "output_root": str(output_root),
        "policy_kind": policy_kind,
        "checkpoint_path": checkpoint_path,
        "require_existing_checkpoint": require_existing_checkpoint,
        "write_parquet_requested": bool(write_parquet),
        "parquet_written": bool(parquet_written),
        "files": {
            "rows_jsonl": str(rows_jsonl),
            "rows_csv": str(rows_csv),
            "rows_parquet": str(rows_parquet) if parquet_written else None,
            "summary_json": str(summary_json),
        },
        "passed": bool(summary["passed"]),
    }

    write_json(output_root / "manifest.json", manifest)

    if not summary["passed"]:
        raise RuntimeError(f"A-family bridge smoke failed: {summary}")

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="artifacts/step26_a_family_bridge_smoke",
    )
    parser.add_argument(
        "--policy-kind",
        choices=["placeholder", "mappo"],
        default="placeholder",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="artifacts/experiment_A_v1/checkpoints/best.pt",
    )
    parser.add_argument(
        "--require-existing-checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    if args.policy_kind == "placeholder":
        checkpoint_path = None
    else:
        checkpoint_path = args.checkpoint_path

    output_root = Path(args.output_root)

    manifest = run_smoke(
        output_root=output_root,
        policy_kind=args.policy_kind,
        checkpoint_path=checkpoint_path,
        require_existing_checkpoint=bool(args.require_existing_checkpoint),
        write_parquet=bool(args.write_parquet),
    )

    print("[OK] A-family bridge smoke runner completed")
    print(f"[OK] output_root: {output_root}")
    print(f"[OK] manifest   : {output_root / 'manifest.json'}")
    print(f"[OK] summary    : {output_root / 'summary.json'}")
    print(f"[OK] rows_jsonl : {output_root / 'a_family_rows.jsonl'}")
    print(f"[OK] rows_csv   : {output_root / 'a_family_rows.csv'}")
    print(f"[OK] passed     : {manifest['passed']}")

    if args.self_test:
        print("[OK] run_a_family_bridge_smoke_v1 self-test passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
