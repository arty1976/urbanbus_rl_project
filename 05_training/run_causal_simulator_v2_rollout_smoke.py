from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


TRAINING_DIR = Path(__file__).resolve().parent
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from adapters.causal_simulator_v2_adapter import (  # noqa: E402
    CausalSimulatorV2Adapter,
    FORBIDDEN_DYNAMIC_SIGNAL_FIELDS,
)


SOURCE_MODE = "static_signal_causal_v2_scaffold_smoke_nonperformance_v1"


REQUIRED_WINDOW_ROLLUP_COLUMNS = [
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "service_date",
    "time_band",
    "evaluation_horizon_minutes",
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


def parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_seed_arg(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_no_forbidden_columns(df: pd.DataFrame, name: str) -> None:
    bad = [c for c in FORBIDDEN_DYNAMIC_SIGNAL_FIELDS if c in df.columns]
    if bad:
        raise RuntimeError(f"{name} contains forbidden dynamic signal fields: {bad}")


def select_action(condition_id: str, agent_id: int, step_idx: int) -> int:
    """Deterministic scaffold policy for smoke-only rollout writing."""
    cid = str(condition_id).upper()

    if "NOOP" in cid:
        return 0
    if "HOLD" in cid:
        return 1
    if "SERVE" in cid:
        return 2

    # Mixed deterministic smoke policy.
    return int((agent_id + step_idx) % 3)


def get_float_from_obs(obs: Dict[str, Any], agent_id: int, col_idx: int) -> float:
    arr = obs["actor_obs"]
    try:
        return float(arr[agent_id, col_idx].item())
    except AttributeError:
        return float(arr[agent_id, col_idx])


def build_raw_event_rows(
    condition_id: str,
    seed: int,
    adapter: CausalSimulatorV2Adapter,
    episode_steps: int,
    evaluation_horizon_minutes: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    obs = adapter.reset(seed=seed)
    window_id = f"{condition_id}_seed{seed:03d}_w000"
    service_date = "2026-01-01"

    for step_idx in range(episode_steps):
        actions = {
            agent_id: select_action(condition_id, agent_id, step_idx)
            for agent_id in range(adapter.num_agents)
        }
        result = adapter.step(actions)
        next_obs = result.obs

        for agent_id in range(adapter.num_agents):
            rows.append({
                "condition_id": condition_id,
                "seed": int(seed),
                "window_id": window_id,
                "state_ts": f"step_{step_idx:04d}",
                "service_date": service_date,
                "time_band": "offpeak",
                "evaluation_horizon_minutes": int(evaluation_horizon_minutes),
                "step_idx": int(step_idx),
                "agent_id": int(agent_id),
                "action": int(actions[agent_id]),
                "reward": float(result.rewards[agent_id]),
                "terminated": bool(result.terminated),
                "source_mode": SOURCE_MODE,
                "demand_pressure_proxy": get_float_from_obs(next_obs, agent_id, 0),
                "signal_count_250m_scaled": get_float_from_obs(next_obs, agent_id, 1),
                "nearest_signal_distance_scaled": get_float_from_obs(next_obs, agent_id, 2),
                "pedestrian_signal_count_250m_scaled": get_float_from_obs(next_obs, agent_id, 3),
                "blink_signal_ratio_250m": get_float_from_obs(next_obs, agent_id, 4),
                "controlled_signal_ratio_250m": get_float_from_obs(next_obs, agent_id, 5),
                "signal_delay_risk_proxy": get_float_from_obs(next_obs, agent_id, 6),
                "intersection_complexity_proxy": get_float_from_obs(next_obs, agent_id, 7),
                "trained_model": False,
                "performance_claim_allowed": False,
                "causal_performance_claim_allowed": False,
                "dynamic_signal_phase_claim_allowed": False,
                "red_light_delay_claim_allowed": False,
                "green_time_claim_allowed": False,
                "cycle_length_claim_allowed": False,
            })

        obs = next_obs
        if result.terminated:
            break

    return rows


def summarize_window(raw_df: pd.DataFrame, condition_id: str, seed: int, num_agents: int, baseline_bus_count: int) -> Dict[str, Any]:
    if raw_df.empty:
        raise RuntimeError("raw_df is empty")

    demand = pd.to_numeric(raw_df["demand_pressure_proxy"], errors="coerce").fillna(0.0)
    reward = pd.to_numeric(raw_df["reward"], errors="coerce").fillna(0.0)
    actions = pd.to_numeric(raw_df["action"], errors="coerce").fillna(0).astype(int)

    mean_demand = float(demand.mean())
    std_demand = float(demand.std(ddof=0))
    p95_demand = float(demand.quantile(0.95))

    headway_mean_seconds = float(420.0 + 160.0 * mean_demand)
    headway_std_seconds = float(max(1.0, 45.0 + 120.0 * std_demand))
    headway_event_count = int(max(1, raw_df["step_idx"].nunique()))
    headway_sample_count = int(headway_event_count + 1)

    wait_passenger_count = int(max(1, num_agents * raw_df["step_idx"].nunique()))
    avg_wait_seconds = float(120.0 + 220.0 * mean_demand)
    wait_total_passenger_seconds = float(avg_wait_seconds * wait_passenger_count)

    passenger_demand_generated = int(wait_passenger_count)
    passenger_served_count = int(max(0, round(passenger_demand_generated * min(1.0, 0.88 + 0.04 * (actions == 2).mean()))))
    passenger_service_rate = float(passenger_served_count / max(1, passenger_demand_generated))
    passenger_wait_p95_seconds = float(120.0 + 260.0 * p95_demand)

    schedulable_arrival_count = int(max(1, headway_event_count))
    ontime_event_count = int(max(0, min(schedulable_arrival_count, round(schedulable_arrival_count * max(0.0, 0.92 - 0.10 * mean_demand)))))
    bunching_event_count = int(max(0, min(headway_event_count, round(headway_event_count * min(0.5, 0.05 + std_demand)))))

    decision_step_count = int(len(raw_df))
    intervention_count = int((actions != 0).sum())

    signal_complexity = pd.to_numeric(raw_df["intersection_complexity_proxy"], errors="coerce").fillna(0.0)
    energy_proxy_total = float(
        50.0
        + 3.0 * float((actions == 2).sum())
        + 1.5 * float((actions == 1).sum())
        + 0.25 * float(signal_complexity.sum())
    )
    energy_proxy_per_passenger = float(energy_proxy_total / max(1, passenger_served_count))

    active_bus_count = int(num_agents)
    baseline_bus_count = int(max(baseline_bus_count, active_bus_count))
    fleet_reduction_ratio = float(max(0.0, (baseline_bus_count - active_bus_count) / max(1, baseline_bus_count)))

    return {
        "condition_id": condition_id,
        "seed": int(seed),
        "window_id": raw_df["window_id"].iloc[0],
        "state_ts": raw_df["state_ts"].iloc[0],
        "service_date": raw_df["service_date"].iloc[0],
        "time_band": raw_df["time_band"].iloc[0],
        "evaluation_horizon_minutes": int(raw_df["evaluation_horizon_minutes"].iloc[0]),
        "headway_mean_seconds": headway_mean_seconds,
        "headway_std_seconds": headway_std_seconds,
        "headway_sample_count": headway_sample_count,
        "bunching_event_count": bunching_event_count,
        "headway_event_count": headway_event_count,
        "wait_total_passenger_seconds": wait_total_passenger_seconds,
        "wait_passenger_count": wait_passenger_count,
        "ontime_event_count": ontime_event_count,
        "schedulable_arrival_count": schedulable_arrival_count,
        "intervention_count": intervention_count,
        "decision_step_count": decision_step_count,
        "energy_proxy_total": energy_proxy_total,
        "source_mode": SOURCE_MODE,
        "qwen_trigger_rate": 0.0,
        "effective_replay_step_minutes": 60.0,
        "passenger_demand_generated": passenger_demand_generated,
        "passenger_served_count": passenger_served_count,
        "passenger_service_rate": passenger_service_rate,
        "passenger_wait_p95_seconds": passenger_wait_p95_seconds,
        "energy_proxy_per_passenger": energy_proxy_per_passenger,
        "active_bus_count": active_bus_count,
        "baseline_bus_count": baseline_bus_count,
        "fleet_reduction_ratio": fleet_reduction_ratio,
        "scaffold_avg_reward": float(reward.mean()),
        "static_signal_context_loaded": True,
        "trained_model": False,
        "performance_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "dynamic_signal_phase_claim_allowed": False,
        "red_light_delay_claim_allowed": False,
        "green_time_claim_allowed": False,
        "cycle_length_claim_allowed": False,
    }


def validate_window_rollup(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_WINDOW_ROLLUP_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"window_rollup missing required columns: {missing}")

    ensure_no_forbidden_columns(df, "window_rollup")

    for c in [
        "headway_mean_seconds",
        "headway_std_seconds",
        "wait_total_passenger_seconds",
        "energy_proxy_total",
    ]:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().any() or (s < 0).any():
            raise RuntimeError(f"window_rollup column must be nonnegative numeric: {c}")

    for flag in [
        "trained_model",
        "performance_claim_allowed",
        "causal_performance_claim_allowed",
        "dynamic_signal_phase_claim_allowed",
        "red_light_delay_claim_allowed",
        "green_time_claim_allowed",
        "cycle_length_claim_allowed",
    ]:
        if bool(df[flag].any()):
            raise RuntimeError(f"guardrail flag must remain false: {flag}")


def run_one(
    project_root: Path,
    output_root: Path,
    condition_id: str,
    seed: int,
    num_agents: int,
    episode_steps: int,
    evaluation_horizon_minutes: int,
    baseline_bus_count: int,
    contract_path: str | None,
    node_signal_features_path: str | None,
    edge_signal_features_path: str | None,
) -> Dict[str, Any]:
    if contract_path and node_signal_features_path and edge_signal_features_path:
        adapter = CausalSimulatorV2Adapter(
            contract_path=contract_path,
            node_signal_features_path=node_signal_features_path,
            edge_signal_features_path=edge_signal_features_path,
            num_agents=num_agents,
            episode_steps=episode_steps,
            seed=seed,
        )
    else:
        adapter = CausalSimulatorV2Adapter.from_step100_artifacts(
            root=project_root,
            num_agents=num_agents,
            episode_steps=episode_steps,
            seed=seed,
        )

    raw_rows = build_raw_event_rows(
        condition_id=condition_id,
        seed=seed,
        adapter=adapter,
        episode_steps=episode_steps,
        evaluation_horizon_minutes=evaluation_horizon_minutes,
    )
    adapter.close()

    raw_df = pd.DataFrame(raw_rows)
    ensure_no_forbidden_columns(raw_df, "raw_events")

    window_row = summarize_window(
        raw_df=raw_df,
        condition_id=condition_id,
        seed=seed,
        num_agents=num_agents,
        baseline_bus_count=baseline_bus_count,
    )
    window_df = pd.DataFrame([window_row])
    validate_window_rollup(window_df)

    seed_dir = output_root / condition_id / "rollouts" / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    raw_path = seed_dir / "raw_events.parquet"
    rollup_path = seed_dir / "window_rollup.parquet"
    raw_df.to_parquet(raw_path, index=False)
    window_df.to_parquet(rollup_path, index=False)

    return {
        "condition_id": condition_id,
        "seed": int(seed),
        "raw_events": str(raw_path),
        "window_rollup": str(rollup_path),
        "raw_event_rows": int(len(raw_df)),
        "window_rollup_rows": int(len(window_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--conditions", default="C2_STATIC_SIGNAL")
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=4)
    parser.add_argument("--evaluation-horizon-minutes", type=int, default=30)
    parser.add_argument("--baseline-bus-count", type=int, default=8)
    parser.add_argument("--output-root", default="artifacts/causal_simulator_v2_rollout_smoke")
    parser.add_argument("--contract-path", default="")
    parser.add_argument("--node-signal-features", default="")
    parser.add_argument("--edge-signal-features", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    conditions = parse_csv_arg(args.conditions)
    seeds = parse_seed_arg(args.seeds)

    results = []
    for condition_id in conditions:
        for seed in seeds:
            results.append(
                run_one(
                    project_root=project_root,
                    output_root=output_root,
                    condition_id=condition_id,
                    seed=seed,
                    num_agents=args.num_agents,
                    episode_steps=args.episode_steps,
                    evaluation_horizon_minutes=args.evaluation_horizon_minutes,
                    baseline_bus_count=args.baseline_bus_count,
                    contract_path=args.contract_path or None,
                    node_signal_features_path=args.node_signal_features or None,
                    edge_signal_features_path=args.edge_signal_features or None,
                )
            )

    manifest = {
        "artifact_version": "causal_simulator_v2_rollout_smoke_manifest_step102",
        "created_at_utc": now_utc_iso(),
        "source_mode": SOURCE_MODE,
        "conditions": conditions,
        "seeds": seeds,
        "num_agents": int(args.num_agents),
        "episode_steps": int(args.episode_steps),
        "evaluation_horizon_minutes": int(args.evaluation_horizon_minutes),
        "run_count": int(len(results)),
        "runs": results,
        "claim_guardrails": {
            "trained_model": False,
            "performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "dynamic_signal_phase_claim_allowed": False,
            "red_light_delay_claim_allowed": False,
            "green_time_claim_allowed": False,
            "cycle_length_claim_allowed": False,
        },
        "forbidden_dynamic_signal_fields": list(FORBIDDEN_DYNAMIC_SIGNAL_FIELDS),
        "next_step": {
            "step": 103,
            "title": "Canonical KPI aggregation smoke for CausalSimulatorAdapter v2 rollout",
        },
    }

    manifest_path = output_root / "rollout_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] Step 102 CausalSimulatorAdapter v2 rollout smoke complete")
    print(f"[OK] output_root : {output_root}")
    print(f"[OK] manifest    : {manifest_path}")
    print(f"[OK] run_count   : {len(results)}")
    print(f"[OK] raw_rows    : {sum(r['raw_event_rows'] for r in results)}")
    print(f"[OK] rollup_rows : {sum(r['window_rollup_rows'] for r in results)}")
    print("[OK] performance_claim_allowed: False")


if __name__ == "__main__":
    main()
