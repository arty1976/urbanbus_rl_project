"""
run_causal_rollout_a_family_bridge_v1.py
========================================

Root-level entrypoint for A/A90/A80/A70 rollout bridge.

This is Step 28.

Purpose
-------
This file provides a root-level execution path similar to run_causal_rollout.py,
but it delegates to the Step 27 A-family scenario rollout writer.

It does not modify run_causal_rollout.py yet.

MAPPO means Multi-Agent Proximal Policy Optimization.
Qwen remains disabled for A/A90/A80/A70 in the current phase.

Modes
-----
1. placeholder:
   Contract/smoke mode. Not valid for paper-level performance claims.

2. mappo:
   Actual MAPPO boundary mode. Requires checkpoint_path.
   If --require-existing-checkpoint is used, the checkpoint file must exist.

Outputs
-------
Default output root:

    artifacts/experiment_A_family_bridge_v1

The delegated writer creates:

- combined_a_family_window_rollup.csv
- combined_a_family_window_rollup.jsonl
- combined_a_family_window_rollup.parquet, if requested
- per-condition/per-seed window_rollup files
- summary.json
- manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


THIS = Path(__file__).resolve()
TRAINING_DIR = THIS.parent
ROLLOUTS_DIR = TRAINING_DIR / "rollouts"

if str(ROLLOUTS_DIR) not in sys.path:
    sys.path.insert(0, str(ROLLOUTS_DIR))


from run_a_family_scenario_rollout_writer_v1 import (  # noqa: E402
    parse_csv_list,
    parse_seed_list,
    run_writer,
)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_entry_manifest(
    *,
    delegated_manifest: Dict[str, Any],
    entry_args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "artifact_version": "run_causal_rollout_a_family_bridge_entry_v1",
        "entrypoint": "05_training/run_causal_rollout_a_family_bridge_v1.py",
        "delegated_writer": "05_training/rollouts/run_a_family_scenario_rollout_writer_v1.py",
        "mode_note": (
            "This is a root-level bridge entrypoint. It does not run neural MAPPO inference yet. "
            "It verifies scenario-to-window_rollup integration before patching run_causal_rollout.py."
        ),
        "policy_kind": entry_args.policy_kind,
        "checkpoint_path": None if entry_args.policy_kind == "placeholder" else entry_args.checkpoint_path,
        "require_existing_checkpoint": bool(entry_args.require_existing_checkpoint),
        "conditions": parse_csv_list(entry_args.conditions),
        "seeds": parse_seed_list(entry_args.seeds),
        "limit": int(entry_args.limit),
        "output_root": str(entry_args.output_root),
        "delegated_manifest": delegated_manifest,
        "passed": bool(delegated_manifest.get("passed", False)),
    }


def run_entry(args: argparse.Namespace) -> Dict[str, Any]:
    if args.self_test:
        scenario_index_path = None
    else:
        scenario_index_path = Path(args.scenario_index)

    delegated_manifest = run_writer(
        scenario_index_path=scenario_index_path,
        output_root=Path(args.output_root),
        limit=int(args.limit),
        conditions=parse_csv_list(args.conditions),
        seeds=parse_seed_list(args.seeds),
        policy_kind=args.policy_kind,
        checkpoint_path=args.checkpoint_path,
        require_existing_checkpoint=bool(args.require_existing_checkpoint),
        baseline_bus_count=float(args.baseline_bus_count),
        eval_horizon_minutes=int(args.eval_horizon_minutes),
        write_parquet_files=bool(args.write_parquet),
        self_test=bool(args.self_test),
    )

    entry_manifest = build_entry_manifest(
        delegated_manifest=delegated_manifest,
        entry_args=args,
    )

    entry_manifest_path = Path(args.output_root) / "entry_manifest.json"
    write_json(entry_manifest_path, entry_manifest)

    if not entry_manifest["passed"]:
        raise RuntimeError("A-family bridge entrypoint failed")

    return entry_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Root-level A-family rollout bridge entrypoint"
    )

    parser.add_argument(
        "--scenario-index",
        default="artifacts/baseline_v1/B1_noop/scenario_index.parquet",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/experiment_A_family_bridge_v1",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--conditions",
        default="A,A90,A80,A70",
    )
    parser.add_argument(
        "--seeds",
        default="1,2,3",
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
        "--baseline-bus-count",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--eval-horizon-minutes",
        type=int,
        default=30,
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

    manifest = run_entry(args)

    print("[OK] root A-family bridge entrypoint completed")
    print(f"[OK] output_root     : {args.output_root}")
    print(f"[OK] entry_manifest  : {Path(args.output_root) / 'entry_manifest.json'}")
    print(f"[OK] writer_manifest : {Path(args.output_root) / 'manifest.json'}")
    print(f"[OK] summary         : {Path(args.output_root) / 'summary.json'}")
    print(f"[OK] passed          : {manifest['passed']}")

    if args.self_test:
        print("[OK] run_causal_rollout_a_family_bridge_v1 self-test passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
