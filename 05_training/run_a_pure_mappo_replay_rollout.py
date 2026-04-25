import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


DEFAULT_SIMULATOR_ADAPTER = "adapters.historical_replay_adapter.HistoricalReplayAdapter"
REQUIRED_SCENARIO_COLUMNS = ["window_id", "state_ts", "service_date", "time_band"]
SOURCE_MODE = "replay_A_pure_mappo_noncausal"


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


def stable_int(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def base_by_time_band(time_band: str) -> Dict[str, float]:
    tb = str(time_band).strip().lower()
    if tb == "peak":
        return {
            "headway_mean": 555.0,
            "headway_std": 138.0,
            "headway_events": 6.0,
            "wait_avg": 335.0,
            "sched_arrivals": 6.0,
            "energy": 116.0,
            "base_waiting_cnt": 18.0,
            "base_ontime_ratio": 0.73,
            "action_threshold": 0.84,
        }
    if tb == "night":
        return {
            "headway_mean": 915.0,
            "headway_std": 84.0,
            "headway_events": 3.0,
            "wait_avg": 168.0,
            "sched_arrivals": 3.0,
            "energy": 67.0,
            "base_waiting_cnt": 5.0,
            "base_ontime_ratio": 0.81,
            "action_threshold": 0.97,
        }
    return {
        "headway_mean": 675.0,
        "headway_std": 103.0,
        "headway_events": 5.0,
        "wait_avg": 222.0,
        "sched_arrivals": 5.0,
        "energy": 87.0,
        "base_waiting_cnt": 9.0,
        "base_ontime_ratio": 0.77,
        "action_threshold": 0.91,
    }


def proxy_headway_seconds(seed: int, window_id: Any, time_band: str) -> float:
    base = base_by_time_band(time_band)
    h = stable_int("a_headway", seed, window_id, time_band)
    jitter = float((h % 151) - 75)
    return float(max(60.0, base["headway_mean"] + jitter + seed * 1.5))


def proxy_waiting_passenger_count(seed: int, window_id: Any, agent_id: int, time_band: str) -> int:
    base = base_by_time_band(time_band)
    h = stable_int("a_waiting", seed, window_id, agent_id, time_band)
    return int(max(0, round(base["base_waiting_cnt"] + (h % 13) - 4)))


def choose_mappo_stub_actions(
    *,
    seed: int,
    window_id: Any,
    time_band: str,
    agent_ids: List[int],
    policy_mode: str,
) -> Tuple[Dict[int, int], Dict[int, float], float]:
    """
    Phase-1 pure MAPPO action placeholder.

    action=0: no-op
    action=1: policy intervention requested

    This runner does not claim trained MAPPO performance. It only produces
    replay-backed A-condition artifacts with qwen_trigger_rate=0.0 so that the
    canonical KPI path can be validated under the same schema as B1/B2.
    """
    if policy_mode == "noop":
        return {int(a): 0 for a in agent_ids}, {int(a): 0.0 for a in agent_ids}, proxy_headway_seconds(seed, window_id, time_band)

    headway = proxy_headway_seconds(seed, window_id, time_band)
    base = base_by_time_band(time_band)
    threshold = float(base["action_threshold"])

    actions: Dict[int, int] = {}
    scores: Dict[int, float] = {}
    for agent_id in agent_ids:
        h = stable_int("a_policy_score", seed, window_id, time_band, agent_id)
        score = (h % 10000) / 10000.0
        scores[int(agent_id)] = float(score)
        actions[int(agent_id)] = 1 if score >= threshold else 0

    # Keep at least one deterministic intervention for severe peak windows.
    if str(time_band).strip().lower() == "peak" and headway > 900 and not any(actions.values()):
        chosen = sorted(agent_ids, key=lambda a: stable_int("a_force", seed, window_id, a))[0]
        actions[int(chosen)] = 1
        scores[int(chosen)] = 1.0

    return actions, scores, headway


def load_adapter_class(import_path: str) -> type:
    try:
        from simulator_adapter_interface import load_adapter_class as project_loader
        return project_loader(import_path)
    except Exception:
        pass

    module_path, class_name = import_path.rsplit(".", 1)
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def validate_inputs(df: pd.DataFrame, contract: Dict[str, Any]) -> pd.DataFrame:
    missing = [c for c in REQUIRED_SCENARIO_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"scenario_index missing columns: {missing}")

    out = df.copy()
    out["time_band"] = out["time_band"].astype(str).str.strip().str.lower()
    valid_time_bands = {str(x).strip().lower() for x in contract.get("time_bands", [])}
    bad = sorted(set(out["time_band"]) - valid_time_bands)
    if bad:
        raise SystemExit(f"invalid time_band values in scenario_index: {bad}")

    if str(contract.get("condition_id", "")).upper() == "A":
        if bool(contract.get("qwen_train", False)):
            raise SystemExit("Experiment A contract must have qwen_train=false")
        if bool(contract.get("qwen_inference", False)):
            raise SystemExit("Experiment A contract must have qwen_inference=false")

    return out


def build_window_rollup_row(
    *,
    scenario: Any,
    seed: int,
    eval_horizon_minutes: int,
    raw_events: pd.DataFrame,
    headway_proxy: float,
    effective_replay_step_minutes: float,
) -> Dict[str, Any]:
    time_band = str(scenario.time_band).strip().lower()
    base = base_by_time_band(time_band)
    h = stable_int("a_rollup", seed, scenario.window_id, time_band)

    intervention_count = int(raw_events["intervention_applied"].sum())
    decision_step_count = int(len(raw_events))
    intervention_ratio = intervention_count / max(decision_step_count, 1)

    # Proxy-only KPI shaping for contract validation. Because HistoricalReplayAdapter
    # is non-causal, these values must not be interpreted as performance evidence.
    headway_mean_seconds = float(max(60.0, headway_proxy - 25.0 * intervention_ratio))
    headway_std_seconds = float(max(1.0, base["headway_std"] + (h % 21) - 10 - 32.0 * intervention_ratio))
    headway_event_count = int(base["headway_events"])
    headway_sample_count = int(headway_event_count + 1)

    wait_passenger_count = int(max(1, raw_events["waiting_passenger_cnt"].sum()))
    avg_wait_seconds = float(max(30.0, base["wait_avg"] + (h % 31) - 15 - 42.0 * intervention_ratio))
    wait_total_passenger_seconds = float(avg_wait_seconds * wait_passenger_count)

    if time_band == "peak":
        base_bunching = int(h % 3)
    elif time_band == "offpeak":
        base_bunching = int(h % 2)
    else:
        base_bunching = 0
    bunching_event_count = int(max(0, min(headway_event_count, base_bunching - (1 if intervention_count else 0))))

    schedulable_arrival_count = int(base["sched_arrivals"])
    ontime_ratio = min(0.98, float(base["base_ontime_ratio"] + 0.075 * intervention_ratio + (h % 8) / 100.0))
    ontime_event_count = int(max(0, min(schedulable_arrival_count, round(schedulable_arrival_count * ontime_ratio))))

    energy_proxy_total = float(max(0.0, base["energy"] + (h % 23) + intervention_count * 1.5))

    return {
        "condition_id": "A",
        "seed": int(seed),
        "window_id": scenario.window_id,
        "state_ts": str(scenario.state_ts),
        "service_date": str(scenario.service_date),
        "time_band": time_band,
        "evaluation_horizon_minutes": int(eval_horizon_minutes),
        "qwen_trigger_rate": 0.0,
        "effective_replay_step_minutes": float(effective_replay_step_minutes),
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
    }


def run_one_seed(
    *,
    df: pd.DataFrame,
    contract: Dict[str, Any],
    a_root: Path,
    seed: int,
    adapter_cls: type,
    simulator_adapter_path: str,
    num_agents: int,
    policy_mode: str,
) -> None:
    seed_dir = a_root / "rollouts" / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    eval_horizon_minutes = int(contract["evaluation_horizon_minutes"])
    raw_rows: List[Dict[str, Any]] = []
    rollup_rows: List[Dict[str, Any]] = []

    adapter = adapter_cls({
        "condition_id": "A",
        "experiment_id": "A_pure_mappo",
        "num_agents": int(num_agents),
        "qwen_trigger_rate": 0.0,
    })

    try:
        graph_skeleton = adapter.get_graph_skeleton()

        for scenario in df.itertuples(index=False):
            scenario_config = {
                "window_id": scenario.window_id,
                "state_ts": str(scenario.state_ts),
                "service_date": str(scenario.service_date),
                "time_band": str(scenario.time_band).strip().lower(),
                "evaluation_horizon_minutes": eval_horizon_minutes,
                "effective_replay_step_minutes": getattr(adapter, "effective_replay_step_minutes", 60),
            }

            obs = adapter.reset(seed=seed, scenario_config=scenario_config)
            agent_ids = [int(x) for x in obs.get("agent_ids", [])]
            if not agent_ids:
                raise RuntimeError(f"adapter returned no agent_ids for window_id={scenario.window_id}")

            actions, policy_scores, headway_proxy = choose_mappo_stub_actions(
                seed=seed,
                window_id=scenario.window_id,
                time_band=scenario_config["time_band"],
                agent_ids=agent_ids,
                policy_mode=policy_mode,
            )

            step_result = adapter.step(actions)
            effective_step = float(step_result.info.get(
                "effective_replay_step_minutes",
                getattr(adapter, "effective_replay_step_minutes", 60),
            ))

            window_event_rows = []
            for agent_id in agent_ids:
                waiting_cnt = proxy_waiting_passenger_count(
                    seed=seed,
                    window_id=scenario.window_id,
                    agent_id=agent_id,
                    time_band=scenario_config["time_band"],
                )
                event = {
                    "experiment_id": "A_pure_mappo",
                    "condition_id": "A",
                    "seed": int(seed),
                    "window_id": scenario.window_id,
                    "state_ts": str(scenario.state_ts),
                    "service_date": str(scenario.service_date),
                    "time_band": scenario_config["time_band"],
                    "agent_id": int(agent_id),
                    "action": int(actions[agent_id]),
                    "policy_mode": policy_mode,
                    "policy_score_proxy": float(policy_scores[agent_id]),
                    "headway_proxy_seconds": float(headway_proxy),
                    "waiting_passenger_cnt": int(waiting_cnt),
                    "intervention_applied": bool(actions[agent_id] != 0),
                    "reward": float(step_result.rewards.get(agent_id, 0.0)),
                    "terminated": bool(step_result.terminated),
                    "truncated": bool(step_result.truncated),
                    "effective_replay_step_minutes": effective_step,
                    "qwen_trigger_rate": 0.0,
                    "source_mode": SOURCE_MODE,
                    "causal_comparison_allowed": False,
                }
                raw_rows.append(event)
                window_event_rows.append(event)

            rollup_rows.append(build_window_rollup_row(
                scenario=scenario,
                seed=seed,
                eval_horizon_minutes=eval_horizon_minutes,
                raw_events=pd.DataFrame(window_event_rows),
                headway_proxy=headway_proxy,
                effective_replay_step_minutes=effective_step,
            ))

        raw_df = pd.DataFrame(raw_rows)
        rollup_df = pd.DataFrame(rollup_rows)

        raw_path = seed_dir / "raw_events.parquet"
        rollup_path = seed_dir / "window_rollup.parquet"
        raw_df.to_parquet(raw_path, index=False)
        rollup_df.to_parquet(rollup_path, index=False)

        manifest = {
            "experiment_id": "A_pure_mappo",
            "condition_id": "A",
            "seed": int(seed),
            "status": "replay_backed_rollout_completed",
            "scenario_count_used": int(len(df)),
            "raw_event_rows": int(len(raw_df)),
            "window_rollup_rows": int(len(rollup_df)),
            "raw_events_path": str(raw_path),
            "window_rollup_path": str(rollup_path),
            "simulator_adapter": simulator_adapter_path,
            "adapter_class": adapter_cls.__name__,
            "policy_mode": policy_mode,
            "num_agents": int(getattr(adapter, "num_agents", num_agents)),
            "observation_space": getattr(adapter, "observation_space", None),
            "action_space": getattr(adapter, "action_space", None),
            "graph_skeleton_summary": {
                "num_nodes": getattr(graph_skeleton, "num_nodes", None),
                "num_edges": getattr(graph_skeleton, "num_edges", None),
                "node_features_dim": getattr(graph_skeleton, "node_features_dim", None),
                "edge_features_dim": getattr(graph_skeleton, "edge_features_dim", None),
            },
            "qwen_train": False,
            "qwen_inference": False,
            "qwen_trigger_rate": 0.0,
            "causal_comparison_allowed": False,
            "source_mode": SOURCE_MODE,
            "note": "HistoricalReplayAdapter is non-causal. This is a Phase-1 replay-backed A-condition smoke artifact, not trained MAPPO causal performance evidence.",
        }
        dump_json(seed_dir / "run_manifest.json", manifest)
        dump_json(seed_dir / "status.json", {
            "status": "completed",
            "experiment_id": "A_pure_mappo",
            "condition_id": "A",
            "seed": int(seed),
            "scenario_count_used": int(len(df)),
            "raw_event_rows": int(len(raw_df)),
            "window_rollup_rows": int(len(rollup_df)),
            "causal_comparison_allowed": False,
            "qwen_trigger_rate": 0.0,
            "source_mode": SOURCE_MODE,
        })

        print(f"[OK] A seed={seed:03d} raw_events     : {raw_path} rows={len(raw_df)}")
        print(f"[OK] A seed={seed:03d} window_rollup  : {rollup_path} rows={len(rollup_df)}")

    finally:
        try:
            adapter.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="A pure MAPPO replay-backed rollout runner")
    parser.add_argument("--contract", required=True, help="Path to experiment_A_contract.json or baseline_contract.json")
    parser.add_argument("--scenario-index", required=True, help="Path to scenario_index.parquet")
    parser.add_argument("--a-root", required=True, help="Root directory for artifacts/experiment_A_v1")
    parser.add_argument("--simulator-adapter", default=DEFAULT_SIMULATOR_ADAPTER)
    parser.add_argument("--seed", type=int, default=0, help="Run one seed only. Default 0 means all contract seeds")
    parser.add_argument("--limit", type=int, default=0, help="Use the first N scenarios for smoke validation")
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--policy-mode", choices=["deterministic_stub", "noop"], default="deterministic_stub")
    args = parser.parse_args()

    contract_path = Path(args.contract)
    scenario_index_path = Path(args.scenario_index)
    a_root = Path(args.a_root)

    if not contract_path.exists():
        raise SystemExit(f"contract not found: {contract_path}")
    if not scenario_index_path.exists():
        raise SystemExit(f"scenario_index not found: {scenario_index_path}")

    contract = load_json_any_encoding(contract_path)
    df = pd.read_parquet(scenario_index_path)
    df = validate_inputs(df, contract)

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()
    if df.empty:
        raise SystemExit("scenario_index is empty after applying limit")

    adapter_cls = load_adapter_class(args.simulator_adapter)

    contract_seeds = [int(x) for x in contract.get("seeds", [])]
    seeds = [int(args.seed)] if args.seed else contract_seeds
    if not seeds:
        raise SystemExit("no seeds found. Provide --seed or contract['seeds']")

    a_root.mkdir(parents=True, exist_ok=True)
    dump_json(a_root / "a_replay_runner_config.json", {
        "experiment_id": "A_pure_mappo",
        "condition_id": "A",
        "contract": str(contract_path),
        "scenario_index": str(scenario_index_path),
        "scenario_count_used": int(len(df)),
        "simulator_adapter": args.simulator_adapter,
        "seeds": seeds,
        "limit": int(args.limit),
        "num_agents": int(args.num_agents),
        "policy_mode": args.policy_mode,
        "qwen_train": False,
        "qwen_inference": False,
        "qwen_trigger_rate": 0.0,
        "source_mode": SOURCE_MODE,
        "causal_comparison_allowed": False,
    })

    total_windows = 0
    for seed in seeds:
        run_one_seed(
            df=df,
            contract=contract,
            a_root=a_root,
            seed=seed,
            adapter_cls=adapter_cls,
            simulator_adapter_path=args.simulator_adapter,
            num_agents=int(args.num_agents),
            policy_mode=args.policy_mode,
        )
        total_windows += len(df)

    print(f"[OK] A replay-backed rollout complete. seeds={seeds} total_window_rows={total_windows}")
    print("[WARN] Non-causal replay only: do not use these outputs as causal performance evidence.")


if __name__ == "__main__":
    main()
