"""
run_a_family_policy_interface_scenario_writer_v1.py
===================================================

Step 37 scenario writer with MAPPO policy interface actions.

MAPPO means Multi-Agent Proximal Policy Optimization.
KPI means Key Performance Indicator.
Qwen remains disabled for A/A90/A80/A70 in the current phase.

Purpose
-------
Step 36 proved that a single A-family rollout row can receive an action dict
from mappo_policy_interface_v1.

Step 37 extends that to scenario_index based multi-window generation:

scenario_index
-> seed
-> A/A90/A80/A70
-> MAPPO policy interface action
-> extended window_rollup rows

Important
---------
This still does not run a trained neural MAPPO model.
It uses contract metadata and conservative mock action from Step 35.
No performance claim is allowed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


THIS = Path(__file__).resolve()
TRAINING_DIR = THIS.parents[1]
ROLLOUTS_DIR = TRAINING_DIR / "rollouts"
POLICIES_DIR = TRAINING_DIR / "policies"
REWARDS_DIR = TRAINING_DIR / "rewards"

for p in (ROLLOUTS_DIR, POLICIES_DIR, REWARDS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


from a_family_policy_interface_bridge_v1 import (  # noqa: E402
    build_a_family_row_with_policy_interface,
    write_contract_checkpoint,
)
from run_a_family_scenario_rollout_writer_v1 import (  # noqa: E402
    build_base_row_from_scenario,
    load_scenario_rows,
    parse_csv_list,
    parse_seed_list,
    summarize,
    write_condition_seed_outputs,
    write_csv,
    write_json,
    write_jsonl,
    write_parquet,
)
from rollout_schema_v1 import validate_rollout_rows, validate_same_demand_by_window  # noqa: E402


A_FAMILY = ["A", "A90", "A80", "A70"]

REQUIRED_ACTION_FIELDS = [
    "action_version",
    "dispatch_delta",
    "hold_seconds",
    "skip_stop_flag",
    "target_headway_ratio",
    "policy_debug",
    "policy_action_debug",
    "policy_interface_version",
    "performance_claim_allowed",
]


def load_policy_debug(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)

    if value is None:
        return {}

    try:
        return json.loads(str(value))
    except Exception:
        return {}


def validate_policy_interface_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    violations: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        condition_id = str(row.get("condition_id"))
        policy_source = str(row.get("policy_source"))
        source_mode = str(row.get("source_mode"))
        qwen_trigger_rate = float(row.get("qwen_trigger_rate", 0.0))

        problems: List[str] = []

        for field in REQUIRED_ACTION_FIELDS:
            if field not in row:
                problems.append(f"missing action/interface field: {field}")

        if condition_id not in A_FAMILY:
            problems.append(f"unexpected condition_id: {condition_id}")

        if policy_source != "mappo_policy":
            problems.append(f"policy_source must be mappo_policy, got {policy_source}")

        if not source_mode.startswith("causal_"):
            problems.append(f"source_mode must start with causal_, got {source_mode}")

        if "_mappo_policy_v1" not in source_mode:
            problems.append(f"source_mode must contain _mappo_policy_v1, got {source_mode}")

        lower_source_mode = source_mode.lower()
        if "placeholder" in lower_source_mode or "stub" in lower_source_mode or "smoke" in lower_source_mode:
            problems.append(f"source_mode contains placeholder/stub/smoke marker: {source_mode}")

        if abs(qwen_trigger_rate) > 1e-12:
            problems.append(f"qwen_trigger_rate must be 0.0, got {qwen_trigger_rate}")

        if bool(row.get("performance_claim_allowed")) is not False:
            problems.append("performance_claim_allowed must be False for mock interface rows")

        policy_debug = load_policy_debug(row.get("policy_action_debug"))
        if not policy_debug.get("mock_action", False):
            problems.append("policy_action_debug.mock_action must be True for Step 37")

        if policy_debug.get("performance_claim_allowed", True) is not False:
            problems.append("policy_action_debug.performance_claim_allowed must be False")

        if problems:
            violations.append(
                {
                    "row_index": idx,
                    "condition_id": condition_id,
                    "window_id": row.get("window_id"),
                    "problems": problems,
                }
            )

    condition_ids = sorted(set(str(row.get("condition_id")) for row in rows))
    source_modes = sorted(set(str(row.get("source_mode")) for row in rows))
    policy_sources = sorted(set(str(row.get("policy_source")) for row in rows))

    return {
        "row_count": len(rows),
        "condition_ids": condition_ids,
        "source_modes": source_modes,
        "policy_sources": policy_sources,
        "violation_count": len(violations),
        "violations": violations,
        "passed": len(violations) == 0,
    }


def build_rows_with_policy_interface(
    *,
    scenario_rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[str],
    seeds: Sequence[int],
    checkpoint_path: Path,
    baseline_bus_count: float,
    eval_horizon_minutes: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        for scenario_row in scenario_rows:
            base_row = build_base_row_from_scenario(
                scenario_row,
                seed=seed,
                baseline_bus_count=baseline_bus_count,
                eval_horizon_minutes=eval_horizon_minutes,
            )

            for condition_id in conditions:
                row = build_a_family_row_with_policy_interface(
                    base_row=base_row,
                    condition_id=condition_id,
                    checkpoint_path=checkpoint_path,
                    require_existing_checkpoint=True,
                )
                rows.append(row)

    return rows


def prepare_checkpoint(
    *,
    output_root: Path,
    checkpoint_path: Optional[Path],
    use_existing_checkpoint: bool,
) -> Path:
    if use_existing_checkpoint:
        if checkpoint_path is None:
            raise FileNotFoundError("--checkpoint-path is required when --use-existing-checkpoint is set")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"MAPPO checkpoint metadata does not exist: {checkpoint_path}")
        return checkpoint_path

    out = output_root / "mock_checkpoints" / "sample_mappo_checkpoint_metadata.json"
    return write_contract_checkpoint(out)


def run_writer(
    *,
    scenario_index_path: Optional[Path],
    output_root: Path,
    limit: int,
    conditions: Sequence[str],
    seeds: Sequence[int],
    checkpoint_path: Optional[Path],
    use_existing_checkpoint: bool,
    baseline_bus_count: float,
    eval_horizon_minutes: int,
    write_parquet_files: bool,
    self_test: bool,
) -> Dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)

    effective_checkpoint = prepare_checkpoint(
        output_root=output_root,
        checkpoint_path=checkpoint_path,
        use_existing_checkpoint=use_existing_checkpoint,
    )

    scenario_rows = load_scenario_rows(
        scenario_index_path=scenario_index_path,
        limit=limit,
        self_test=self_test,
    )

    rows = build_rows_with_policy_interface(
        scenario_rows=scenario_rows,
        conditions=conditions,
        seeds=seeds,
        checkpoint_path=effective_checkpoint,
        baseline_bus_count=baseline_bus_count,
        eval_horizon_minutes=eval_horizon_minutes,
    )

    schema_summary = validate_rollout_rows(rows, strict=True)
    demand_summary = validate_same_demand_by_window(rows)
    base_summary = summarize(rows)
    action_summary = validate_policy_interface_rows(rows)

    if not schema_summary["passed"]:
        raise RuntimeError(f"schema validation failed: {schema_summary}")

    if not demand_summary["passed"]:
        raise RuntimeError(f"demand fairness validation failed: {demand_summary}")

    if not action_summary["passed"]:
        raise RuntimeError(f"policy interface action validation failed: {action_summary}")

    combined_csv = output_root / "combined_policy_interface_window_rollup.csv"
    combined_jsonl = output_root / "combined_policy_interface_window_rollup.jsonl"
    combined_parquet = output_root / "combined_policy_interface_window_rollup.parquet"

    write_csv(combined_csv, rows)
    write_jsonl(combined_jsonl, rows)

    combined_parquet_written = False
    if write_parquet_files:
        combined_parquet_written = write_parquet(combined_parquet, rows)

    condition_seed_files = write_condition_seed_outputs(
        output_root=output_root,
        rows=rows,
        write_parquet_files=write_parquet_files,
    )

    manifest = {
        "artifact_version": "a_family_policy_interface_scenario_writer_v1",
        "important": "This is not a performance claim. Step 37 uses conservative mock policy actions.",
        "scenario_index_path": str(scenario_index_path) if scenario_index_path else None,
        "output_root": str(output_root),
        "checkpoint_path": str(effective_checkpoint),
        "use_existing_checkpoint": bool(use_existing_checkpoint),
        "limit": limit,
        "conditions": list(conditions),
        "seeds": list(seeds),
        "baseline_bus_count": baseline_bus_count,
        "eval_horizon_minutes": eval_horizon_minutes,
        "write_parquet_files": write_parquet_files,
        "combined_files": {
            "csv": str(combined_csv),
            "jsonl": str(combined_jsonl),
            "parquet": str(combined_parquet) if combined_parquet_written else None,
        },
        "condition_seed_files": condition_seed_files,
        "schema_summary": schema_summary,
        "demand_fairness_summary": demand_summary,
        "base_summary": base_summary,
        "action_summary": action_summary,
        "passed": bool(schema_summary["passed"]) and bool(demand_summary["passed"]) and bool(action_summary["passed"]),
    }

    write_json(output_root / "summary.json", base_summary)
    write_json(output_root / "action_summary.json", action_summary)
    write_json(output_root / "manifest.json", manifest)

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario-index",
        default="artifacts/baseline_v1/B1_noop/scenario_index.parquet",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/step37_a_family_policy_interface_scenario_writer",
    )
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--conditions", default="A,A90,A80,A70")
    parser.add_argument("--seeds", default="1")
    parser.add_argument(
        "--checkpoint-path",
        default="",
    )
    parser.add_argument("--use-existing-checkpoint", action="store_true")
    parser.add_argument("--baseline-bus-count", type=float, default=10.0)
    parser.add_argument("--eval-horizon-minutes", type=int, default=30)
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--self-test", action="store_true")

    args = parser.parse_args()

    scenario_index_path: Optional[Path]
    if args.self_test:
        scenario_index_path = None
    else:
        scenario_index_path = Path(args.scenario_index)

    checkpoint_path: Optional[Path]
    if str(args.checkpoint_path).strip():
        checkpoint_path = Path(args.checkpoint_path)
    else:
        checkpoint_path = None

    manifest = run_writer(
        scenario_index_path=scenario_index_path,
        output_root=Path(args.output_root),
        limit=int(args.limit),
        conditions=parse_csv_list(args.conditions),
        seeds=parse_seed_list(args.seeds),
        checkpoint_path=checkpoint_path,
        use_existing_checkpoint=bool(args.use_existing_checkpoint),
        baseline_bus_count=float(args.baseline_bus_count),
        eval_horizon_minutes=int(args.eval_horizon_minutes),
        write_parquet_files=bool(args.write_parquet),
        self_test=bool(args.self_test),
    )

    print("[OK] Step 37 A-family policy interface scenario writer completed")
    print(f"[OK] output_root : {args.output_root}")
    print(f"[OK] manifest    : {Path(args.output_root) / 'manifest.json'}")
    print(f"[OK] action_sum  : {Path(args.output_root) / 'action_summary.json'}")
    print(f"[OK] passed      : {manifest['passed']}")

    if args.self_test:
        print("[OK] run_a_family_policy_interface_scenario_writer_v1 self-test passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
