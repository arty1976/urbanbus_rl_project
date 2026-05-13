from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


KPI_12 = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def stable_int(*parts: str) -> int:
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def base_by_time_band(time_band: str) -> Dict[str, float]:
    tb = str(time_band).strip().lower()

    if tb == "peak":
        return {
            "headway_mean": 540.0,
            "headway_std": 150.0,
            "headway_events": 6,
            "wait_avg": 360.0,
            "sched_arrivals": 6,
            "energy": 120.0,
            "demand_base": 54,
        }

    if tb == "night":
        return {
            "headway_mean": 900.0,
            "headway_std": 90.0,
            "headway_events": 3,
            "wait_avg": 180.0,
            "sched_arrivals": 3,
            "energy": 70.0,
            "demand_base": 24,
        }

    return {
        "headway_mean": 660.0,
        "headway_std": 110.0,
        "headway_events": 5,
        "wait_avg": 240.0,
        "sched_arrivals": 5,
        "energy": 90.0,
        "demand_base": 36,
    }


def build_window_rollup(
    scenario_df: pd.DataFrame,
    seed: int,
    eval_horizon_minutes: int,
    num_agents: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for r in scenario_df.itertuples(index=False):
        tb = str(r.time_band).strip().lower()
        base = base_by_time_band(tb)
        h = stable_int("B0C", seed, str(r.window_id))

        headway_events = int(base["headway_events"])
        headway_sample_count = headway_events + 1

        headway_mean_seconds = float(base["headway_mean"] + (h % 31) - 15 + seed * 2)
        headway_std_seconds = float(max(1.0, base["headway_std"] + (h % 21) - 10))

        cv_headway = float(headway_std_seconds / max(headway_mean_seconds, 1.0))

        passenger_demand_generated = int(base["demand_base"] + (h % 19))
        service_rate_target = 0.965 + ((h % 5) / 1000.0)
        passenger_served_count = int(round(passenger_demand_generated * service_rate_target))
        passenger_served_count = max(0, min(passenger_demand_generated, passenger_served_count))

        passenger_service_rate = float(
            passenger_served_count / passenger_demand_generated
            if passenger_demand_generated > 0 else 0.0
        )

        avg_wait_seconds = float(max(30.0, base["wait_avg"] + (h % 31) - 15 + seed * 2))
        wait_total_passenger_seconds = float(avg_wait_seconds * passenger_demand_generated)
        passenger_wait_p95_seconds = float(avg_wait_seconds * 1.65 + (h % 17))

        schedulable_arrival_count = int(base["sched_arrivals"])
        ontime_ratio = 0.72 + ((h % 16) / 100.0)
        ontime_event_count = int(
            max(0, min(schedulable_arrival_count, round(schedulable_arrival_count * ontime_ratio)))
        )
        on_time_rate = float(ontime_event_count / schedulable_arrival_count)

        if tb == "peak":
            bunching_event_count = int(h % 3)
        elif tb == "offpeak":
            bunching_event_count = int(h % 2)
        else:
            bunching_event_count = 0

        bunching_event_count = int(min(bunching_event_count, headway_events))
        bunching_rate = float(bunching_event_count / max(headway_events, 1))

        intervention_count = 0
        decision_step_count = 1
        intervention_rate = 0.0

        energy_proxy_total = float(base["energy"] + (h % 23))
        energy_proxy_per_passenger = float(
            energy_proxy_total / max(passenger_served_count, 1)
        )

        rows.append({
            "condition_id": "B0C",
            "seed": int(seed),
            "window_id": r.window_id,
            "state_ts": str(r.state_ts),
            "service_date": str(r.service_date),
            "time_band": tb,
            "evaluation_horizon_minutes": int(eval_horizon_minutes),

            "headway_mean_seconds": headway_mean_seconds,
            "headway_std_seconds": headway_std_seconds,
            "headway_sample_count": int(headway_sample_count),
            "bunching_event_count": int(bunching_event_count),
            "headway_event_count": int(headway_events),
            "wait_total_passenger_seconds": wait_total_passenger_seconds,
            "wait_passenger_count": int(passenger_demand_generated),
            "ontime_event_count": int(ontime_event_count),
            "schedulable_arrival_count": int(schedulable_arrival_count),
            "intervention_count": int(intervention_count),
            "decision_step_count": int(decision_step_count),
            "energy_proxy_total": energy_proxy_total,

            "cv_headway": cv_headway,
            "avg_wait_seconds": avg_wait_seconds,
            "bunching_rate": bunching_rate,
            "on_time_rate": on_time_rate,
            "intervention_rate": intervention_rate,
            "energy_proxy": energy_proxy_total,
            "passenger_demand_generated": int(passenger_demand_generated),
            "passenger_served_count": int(passenger_served_count),
            "passenger_service_rate": passenger_service_rate,
            "passenger_wait_p95_seconds": passenger_wait_p95_seconds,
            "energy_proxy_per_passenger": energy_proxy_per_passenger,
            "fleet_reduction_ratio": 0.0,

            "policy_source": "historical_shadow_policy",
            "source_mode": "noncausal_B0C_causal_shadow_scaffold_not_validated_v1",
            "source_mode_target_after_validation": "causal_B0C_historical_shadow_v1",
            "policy_action_source_mode": "maintain_no_learned_intervention",
            "qwen_train": False,
            "qwen_inference": False,
            "qwen_trigger_rate": 0.0,
            "active_bus_ratio": 1.0,
            "learned_policy": False,
            "rule_based_policy": False,
            "causal_simulator_target": True,
            "simulator_validation_status": "not_validated",
            "strict_canonical": False,
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
            "num_agents": int(num_agents),
        })

    return pd.DataFrame(rows)


def build_raw_events(window_df: pd.DataFrame, num_agents: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for r in window_df.itertuples(index=False):
        for agent_id in range(num_agents):
            rows.append({
                "condition_id": "B0C",
                "seed": int(r.seed),
                "window_id": r.window_id,
                "state_ts": r.state_ts,
                "service_date": r.service_date,
                "time_band": r.time_band,
                "agent_id": int(agent_id),
                "action": "maintain",
                "action_code": 0,
                "intervention_applied": False,
                "hold_seconds": 0.0,
                "skip_applied": False,
                "terminated": False,
                "truncated": False,
                "waiting_passenger_cnt": float(r.passenger_demand_generated) / max(num_agents, 1),
                "reward_proxy": -float(r.avg_wait_seconds),
                "policy_source": "historical_shadow_policy",
                "source_mode": "noncausal_B0C_causal_shadow_scaffold_not_validated_v1",
                "qwen_trigger_rate": 0.0,
                "causal_comparison_allowed": False,
                "paper_level_claim_allowed": False,
            })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--b0c-contract", required=True)
    parser.add_argument("--scenario-index", required=True)
    parser.add_argument("--b0c-root", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-agents", type=int, default=10)
    args = parser.parse_args()

    base_contract_path = Path(args.contract)
    b0c_contract_path = Path(args.b0c_contract)
    scenario_index_path = Path(args.scenario_index)
    b0c_root = Path(args.b0c_root)

    if not base_contract_path.exists():
        raise SystemExit(f"[STOP] missing baseline contract: {base_contract_path}")

    if not b0c_contract_path.exists():
        raise SystemExit(f"[STOP] missing B0C contract: {b0c_contract_path}")

    if not scenario_index_path.exists():
        raise SystemExit(f"[STOP] missing scenario index: {scenario_index_path}")

    base_contract = load_json(base_contract_path)
    b0c_contract = load_json(b0c_contract_path)

    eval_horizon = int(base_contract.get("evaluation_horizon_minutes", 30))

    if b0c_contract.get("condition_id") != "B0C":
        raise SystemExit("[STOP] B0C contract condition_id must be B0C")

    scenario_df = pd.read_parquet(scenario_index_path).copy()

    required = ["window_id", "state_ts", "service_date", "time_band"]
    missing = [c for c in required if c not in scenario_df.columns]
    if missing:
        raise SystemExit(f"[STOP] scenario_index missing columns: {missing}")

    if args.limit and args.limit > 0:
        scenario_df = scenario_df.head(args.limit).copy()

    out_dir = b0c_root / "rollouts" / f"seed_{args.seed:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    window_df = build_window_rollup(
        scenario_df=scenario_df,
        seed=args.seed,
        eval_horizon_minutes=eval_horizon,
        num_agents=args.num_agents,
    )
    raw_df = build_raw_events(window_df, num_agents=args.num_agents)

    raw_path = out_dir / "raw_events.parquet"
    window_path = out_dir / "window_rollup.parquet"
    status_path = out_dir / "status.json"
    manifest_path = out_dir / "run_manifest.json"

    raw_df.to_parquet(raw_path, index=False)
    window_df.to_parquet(window_path, index=False)

    manifest = {
        "artifact_version": "b0c_causal_shadow_rollout_writer_v1",
        "condition_id": "B0C",
        "baseline_id": "B0C_causal_shadow_v1",
        "seed": int(args.seed),
        "scenario_index": str(scenario_index_path),
        "scenario_count": int(len(scenario_df)),
        "num_agents": int(args.num_agents),
        "evaluation_horizon_minutes": int(eval_horizon),
        "output_files": {
            "raw_events": str(raw_path),
            "window_rollup": str(window_path),
            "status": str(status_path),
            "run_manifest": str(manifest_path),
        },
        "policy": b0c_contract.get("policy", {}),
        "claim_guards": {
            "causal_simulator_target": True,
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "actual_results": False,
        },
    }

    status = {
        "status": "B0C_CAUSAL_SHADOW_ROLLOUT_SCAFFOLD_WRITTEN",
        "warning": "Not simulator-validated. Do not use as causal performance evidence.",
        "raw_event_rows": int(len(raw_df)),
        "window_rollup_rows": int(len(window_df)),
        "kpi_schema": KPI_12,
        "causal_comparison_allowed": False,
        "paper_level_claim_allowed": False,
    }

    dump_json(manifest_path, manifest)
    dump_json(status_path, status)

    print(f"[OK] B0C seed={args.seed:03d} raw_events     : {raw_path} rows={len(raw_df)}")
    print(f"[OK] B0C seed={args.seed:03d} window_rollup  : {window_path} rows={len(window_df)}")
    print("[WARN] B0C scaffold is not simulator-validated; do not use as causal performance evidence.")


if __name__ == "__main__":
    main()
