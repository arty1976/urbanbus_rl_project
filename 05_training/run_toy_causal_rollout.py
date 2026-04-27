from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from adapters.causal_simulator_adapter import (  # noqa: E402
    ACTION_DISPATCH,
    ACTION_HOLD,
    ACTION_SKIP,
    CausalSimulatorAdapter,
)


VALID_CONDITIONS = {"A", "A90", "A80", "A70"}
VALID_TIME_BANDS = {"peak", "offpeak", "night"}

REQUIRED_RAW_EVENT_COLUMNS = [
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "time_band",
    "agent_id",
    "action",
    "action_name",
    "position_before",
    "position_after",
    "waiting_passenger_cnt",
    "passenger_served_count",
    "distance_m",
    "hold_seconds",
    "acceleration_event_count",
    "energy_proxy_total",
    "intervention_applied",
    "terminated",
    "truncated",
    "source_mode",
    "causal_comparison_allowed",
]

REQUIRED_WINDOW_ROLLUP_COLUMNS = [
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "service_date",
    "time_band",
    "evaluation_horizon_minutes",
    "qwen_trigger_rate",
    "effective_replay_step_minutes",
    "headway_mean_seconds",
    "headway_std_seconds",
    "headway_sample_count",
    "bunching_event_count",
    "headway_event_count",
    "wait_total_passenger_seconds",
    "wait_passenger_count",
    "ontime_event_count",
    "schedulable_arrival_count",
    "intervention_count",
    "decision_step_count",
    "energy_proxy_total",
    "source_mode",
]


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(x) for x in parse_csv(value)]


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def stable_bucket(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    total = 0
    for ch in text:
        total = (total * 131 + ord(ch)) % 1_000_003
    return total


def scenario_rows(time_bands: Iterable[str], windows_per_time_band: int) -> List[Dict[str, Any]]:
    base_ts = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)
    rows: List[Dict[str, Any]] = []

    window_no = 0
    for time_band in time_bands:
        tb = str(time_band).strip().lower()
        for idx in range(windows_per_time_band):
            ts = base_ts + timedelta(minutes=30 * window_no)
            window_no += 1
            rows.append(
                {
                    "window_id": f"toy_{tb}_{idx + 1:03d}",
                    "state_ts": ts.isoformat().replace("+00:00", "Z"),
                    "service_date": ts.date().isoformat(),
                    "time_band": tb,
                }
            )

    return rows


def build_actions(condition_id: str, seed: int, window_id: str, step_idx: int, num_agents: int) -> Dict[int, int]:
    cid = str(condition_id).upper()
    actions: Dict[int, int] = {}

    # Toy policies are deterministic and intentionally simple.
    # They are not performance claims. They only create causal file outputs.
    for agent_id in range(num_agents):
        b = stable_bucket(cid, seed, window_id, step_idx, agent_id)

        if cid == "A":
            if b % 11 == 0:
                actions[agent_id] = ACTION_SKIP
            elif b % 7 == 0:
                actions[agent_id] = ACTION_HOLD
            else:
                actions[agent_id] = ACTION_DISPATCH

        elif cid == "A90":
            if agent_id == num_agents - 1:
                actions[agent_id] = ACTION_HOLD
            elif b % 13 == 0:
                actions[agent_id] = ACTION_SKIP
            else:
                actions[agent_id] = ACTION_DISPATCH

        elif cid == "A80":
            if agent_id >= max(1, int(num_agents * 0.8)):
                actions[agent_id] = ACTION_HOLD
            elif b % 9 == 0:
                actions[agent_id] = ACTION_SKIP
            else:
                actions[agent_id] = ACTION_DISPATCH

        elif cid == "A70":
            if agent_id >= max(1, int(num_agents * 0.7)):
                actions[agent_id] = ACTION_HOLD
            elif b % 8 == 0:
                actions[agent_id] = ACTION_SKIP
            else:
                actions[agent_id] = ACTION_DISPATCH

        else:
            raise RuntimeError(f"unsupported condition_id: {condition_id}")

    return actions


def validate_output_frames(raw_df: pd.DataFrame, rollup_df: pd.DataFrame) -> None:
    missing_raw = [c for c in REQUIRED_RAW_EVENT_COLUMNS if c not in raw_df.columns]
    if missing_raw:
        raise RuntimeError(f"raw_events missing columns: {missing_raw}")

    missing_rollup = [c for c in REQUIRED_WINDOW_ROLLUP_COLUMNS if c not in rollup_df.columns]
    if missing_rollup:
        raise RuntimeError(f"window_rollup missing columns: {missing_rollup}")

    if raw_df.empty:
        raise RuntimeError("raw_events is empty")

    if rollup_df.empty:
        raise RuntimeError("window_rollup is empty")

    if not bool(raw_df["causal_comparison_allowed"].all()):
        raise RuntimeError("raw_events must have causal_comparison_allowed=true for all rows")

    if "causal_comparison_allowed" not in rollup_df.columns:
        raise RuntimeError("window_rollup must include causal_comparison_allowed metadata column")

    if not bool(rollup_df["causal_comparison_allowed"].all()):
        raise RuntimeError("window_rollup must have causal_comparison_allowed=true for all rows")

    if not all(str(x).startswith("causal_") for x in raw_df["source_mode"].dropna().unique()):
        raise RuntimeError("raw_events source_mode must start with causal_")

    if not all(str(x).startswith("causal_") for x in rollup_df["source_mode"].dropna().unique()):
        raise RuntimeError("window_rollup source_mode must start with causal_")

    bad_horizon = pd.to_numeric(rollup_df["evaluation_horizon_minutes"], errors="coerce") != 30
    if bool(bad_horizon.any()):
        raise RuntimeError("evaluation_horizon_minutes must be 30")

    bad_decisions = pd.to_numeric(rollup_df["decision_step_count"], errors="coerce") <= 0
    if bool(bad_decisions.any()):
        raise RuntimeError("decision_step_count must be positive")


def run_rollouts(args: argparse.Namespace) -> Dict[str, Any]:
    conditions = [x.upper() for x in parse_csv(args.conditions)]
    seeds = parse_int_csv(args.seeds)
    time_bands = [x.lower() for x in parse_csv(args.time_bands)]

    bad_conditions = sorted(set(conditions) - VALID_CONDITIONS)
    if bad_conditions:
        raise SystemExit(f"invalid conditions: {bad_conditions}")

    bad_time_bands = sorted(set(time_bands) - VALID_TIME_BANDS)
    if bad_time_bands:
        raise SystemExit(f"invalid time_bands: {bad_time_bands}")

    if args.num_agents < 5 or args.num_agents > 10:
        raise SystemExit("--num-agents must be between 5 and 10")

    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    if args.windows_per_time_band < 1:
        raise SystemExit("--windows-per-time-band must be >= 1")

    output_root = Path(args.output_root)
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    scenarios = scenario_rows(time_bands=time_bands, windows_per_time_band=args.windows_per_time_band)
    scenario_index = pd.DataFrame(scenarios)
    scenario_index_path = output_root / "scenario_index.parquet"
    scenario_index.to_parquet(scenario_index_path, index=False)

    total_raw_rows = 0
    total_rollup_rows = 0
    written_files: List[str] = []

    for condition_id in conditions:
        condition_root = output_root / condition_id
        condition_root.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            seed_dir = condition_root / "rollouts" / f"seed_{seed:03d}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            raw_frames: List[pd.DataFrame] = []
            rollup_rows: List[Dict[str, Any]] = []

            for scenario in scenarios:
                adapter = CausalSimulatorAdapter(
                    {
                        "condition_id": condition_id,
                        "seed": seed,
                        "num_agents": args.num_agents,
                        "max_steps": args.steps,
                        "control_step_minutes": 30,
                        "evaluation_horizon_minutes": 30,
                    }
                )

                adapter.reset(seed=seed, scenario_config=scenario)

                for step_idx in range(args.steps):
                    actions = build_actions(
                        condition_id=condition_id,
                        seed=seed,
                        window_id=str(scenario["window_id"]),
                        step_idx=step_idx,
                        num_agents=args.num_agents,
                    )
                    adapter.step(actions)

                raw_frames.append(pd.DataFrame(adapter.get_raw_events()))
                rollup_rows.append(adapter.build_window_rollup_row())
                adapter.close()

            raw_df = pd.concat(raw_frames, ignore_index=True)
            rollup_df = pd.DataFrame(rollup_rows)

            validate_output_frames(raw_df=raw_df, rollup_df=rollup_df)

            raw_path = seed_dir / "raw_events.parquet"
            rollup_path = seed_dir / "window_rollup.parquet"
            manifest_path = seed_dir / "run_manifest.json"

            raw_df.to_parquet(raw_path, index=False)
            rollup_df.to_parquet(rollup_path, index=False)

            manifest = {
                "artifact_version": "toy_causal_rollout_writer_v1_step78",
                "condition_id": condition_id,
                "seed": int(seed),
                "num_agents": int(args.num_agents),
                "steps": int(args.steps),
                "control_step_minutes": 30,
                "evaluation_horizon_minutes": 30,
                "scenario_count": int(len(scenarios)),
                "raw_event_rows": int(len(raw_df)),
                "window_rollup_rows": int(len(rollup_df)),
                "source_modes": sorted(raw_df["source_mode"].astype(str).unique().tolist()),
                "causal_comparison_allowed": bool(raw_df["causal_comparison_allowed"].all()),
                "qwen_trigger_rate": 0.0,
                "outputs": {
                    "raw_events": str(raw_path),
                    "window_rollup": str(rollup_path),
                    "run_manifest": str(manifest_path),
                },
                "note": (
                    "Step 78 writes toy causal simulator outputs. "
                    "This is causal-dynamics validation, not a paper-level performance claim."
                ),
            }
            dump_json(manifest_path, manifest)

            total_raw_rows += int(len(raw_df))
            total_rollup_rows += int(len(rollup_df))
            written_files.extend([str(raw_path), str(rollup_path), str(manifest_path)])

    root_manifest = {
        "artifact_version": "toy_causal_rollout_writer_v1_step78",
        "output_root": str(output_root),
        "conditions": conditions,
        "seeds": seeds,
        "time_bands": time_bands,
        "windows_per_time_band": int(args.windows_per_time_band),
        "num_agents": int(args.num_agents),
        "steps": int(args.steps),
        "control_step_minutes": 30,
        "evaluation_horizon_minutes": 30,
        "scenario_index": str(scenario_index_path),
        "scenario_count": int(len(scenarios)),
        "total_raw_event_rows": int(total_raw_rows),
        "total_window_rollup_rows": int(total_rollup_rows),
        "causal_comparison_allowed": True,
        "written_files": written_files,
    }

    root_manifest_path = output_root / "run_manifest.json"
    dump_json(root_manifest_path, root_manifest)

    print("[OK] Step 78 toy causal rollout writer completed")
    print(f"[OK] output_root        : {output_root}")
    print(f"[OK] scenario_index     : {scenario_index_path}")
    print(f"[OK] conditions         : {conditions}")
    print(f"[OK] seeds              : {seeds}")
    print(f"[OK] scenario_count     : {len(scenarios)}")
    print(f"[OK] raw_event_rows     : {total_raw_rows}")
    print(f"[OK] window_rollup_rows : {total_rollup_rows}")
    print(f"[OK] manifest           : {root_manifest_path}")

    return root_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="artifacts/phase2_toy_causal_v1")
    parser.add_argument("--conditions", default="A")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--time-bands", default="peak,offpeak,night")
    parser.add_argument("--windows-per-time-band", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    run_rollouts(args)


if __name__ == "__main__":
    main()
