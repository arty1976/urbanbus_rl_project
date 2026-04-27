from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def append_training_dir_to_path() -> Path:
    training_dir = Path(__file__).resolve().parent
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    return training_dir


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_first_scenario(
    scenario_index_path: str,
    *,
    row_index: int,
) -> Dict[str, Any]:
    if not scenario_index_path:
        return {
            "window_id": "step52_a_family_policy_rollout_smoke",
            "state_ts": "2024-01-01T08:00:00Z",
            "service_date": "2024-01-01",
            "time_band": "offpeak",
            "snapshot_path": None,
            "effective_replay_step_minutes": 60,
        }

    import pandas as pd

    path = Path(scenario_index_path)
    if not path.exists():
        raise FileNotFoundError(f"scenario_index not found: {path}")

    df = pd.read_parquet(path)
    if df.empty:
        raise RuntimeError(f"scenario_index is empty: {path}")

    idx = int(row_index)
    if idx < 0 or idx >= len(df):
        raise RuntimeError(f"scenario_row_index out of range: {idx}, rows={len(df)}")

    row = df.iloc[idx].to_dict()

    state_ts = str(row.get("state_ts", "2024-01-01T08:00:00Z"))
    service_date = str(row.get("service_date", state_ts[:10]))
    time_band = str(row.get("time_band", "offpeak")).strip().lower()

    return {
        "window_id": str(row.get("window_id", f"scenario_row_{idx}")),
        "state_ts": state_ts,
        "service_date": service_date,
        "time_band": time_band,
        "snapshot_path": row.get("snapshot_path", None),
        "effective_replay_step_minutes": int(row.get("effective_replay_step_minutes", 60)),
    }


def make_mock_actions(obs: Dict[str, Any]) -> tuple[Dict[int, int], Dict[str, Any]]:
    agent_ids = [int(x) for x in obs.get("agent_ids", [])]
    actions = {agent_id: 0 for agent_id in agent_ids}

    metadata = {
        "policy_metadata_version": "policy_source_metadata_v1",
        "policy_source": "mock_policy",
        "policy_action_source_version": "mock_policy_action_source_v1",
        "policy_action_source_mode": "mock_smoke_v1",
        "checkpoint_path": "",
        "checkpoint_validation_mode": "",
        "checkpoint_validator_ran": False,
        "checkpoint_loaded": False,
        "trained_model": False,
        "performance_claim_allowed": False,
        "placeholder_fallback_used": True,
        "mock_action_used": True,
        "qwen_train": False,
        "qwen_inference": False,
        "qwen_trigger_rate": 0.0,
        "reward_version": "mappo_reward_v1",
        "energy_proxy_model_version": "daegu_energy_proxy_v1",
        "energy_proxy_constants": {
            "k_dist_kwh_per_m": 0.0012,
            "k_acc_kwh_per_event": 0.1800,
            "k_idle_kwh_per_sec": 0.0080,
        },
        "action_count": len(actions),
        "actions_preview": [
            {"agent_id": int(agent_id), "action": 0}
            for agent_id in agent_ids[:20]
        ],
        "note": "mock_smoke development action path; never use for performance claims",
    }

    return actions, metadata


def build_mappo_actions(
    obs: Dict[str, Any],
    *,
    checkpoint_path: str,
    checkpoint_validation_mode: str,
    device: str,
    seed: int,
    validation_report_path: Path,
) -> tuple[Dict[int, int], Dict[str, Any]]:
    from policies.mappo_policy_action_source import build_mappo_policy_action_source

    source = build_mappo_policy_action_source(
        str(checkpoint_path),
        validation_mode=str(checkpoint_validation_mode),
        device=str(device),
        deterministic=True,
        require_checkpoint=True,
        hidden_dim=128,
        seed=int(seed),
        validation_report_path=str(validation_report_path),
    )

    result = source.select_actions(obs)
    return result.actions, result.metadata


def build_rollout_metadata(
    *,
    condition_id: str,
    policy_source_mode: str,
    action_metadata: Dict[str, Any],
    causal_simulator: bool,
    expected_source_mode: str,
) -> Dict[str, Any]:
    if str(policy_source_mode).strip().lower() == "mock_smoke":
        constants = action_metadata.get("energy_proxy_constants", {})
        return {
            "policy_metadata_version": "policy_source_metadata_v1",
            "condition_id": str(condition_id).upper(),
            "source_mode": str(expected_source_mode),
            "policy_source": "mock_policy",
            "policy_action_source_version": "mock_policy_action_source_v1",
            "policy_action_source_mode": "mock_smoke_v1",
            "checkpoint_path": "",
            "checkpoint_validation_mode": "",
            "checkpoint_validator_ran": False,
            "checkpoint_loaded": False,
            "trained_model": False,
            "performance_claim_allowed": False,
            "placeholder_fallback_used": True,
            "mock_action_used": True,
            "qwen_train": False,
            "qwen_inference": False,
            "qwen_trigger_rate": 0.0,
            "reward_version": "mappo_reward_v1",
            "energy_proxy_model_version": "daegu_energy_proxy_v1",
            "k_dist_kwh_per_m": float(constants.get("k_dist_kwh_per_m", 0.0012)),
            "k_acc_kwh_per_event": float(constants.get("k_acc_kwh_per_event", 0.1800)),
            "k_idle_kwh_per_sec": float(constants.get("k_idle_kwh_per_sec", 0.0080)),
            "actual_policy_claim_ready": False,
            "causal_policy_claim_ready": False,
        }

    from policies.policy_source_metadata import (
        build_rollout_policy_metadata,
        validate_rollout_policy_metadata,
    )

    rollout_metadata = build_rollout_policy_metadata(
        action_metadata,
        condition_id=str(condition_id).upper(),
        causal_simulator=bool(causal_simulator),
        source_mode=str(expected_source_mode),
    )

    validation = validate_rollout_policy_metadata(
        rollout_metadata,
        require_actual_ready=False,
        require_causal_ready=False,
    )

    if not validation.valid:
        raise RuntimeError(f"rollout policy metadata validation failed: {validation.errors}")

    return rollout_metadata


def base_by_time_band(time_band: str) -> Dict[str, float]:
    tb = str(time_band).strip().lower()
    if tb == "peak":
        return {
            "headway_mean": 540.0,
            "headway_std": 150.0,
            "headway_events": 6.0,
            "wait_avg": 360.0,
            "sched_arrivals": 6.0,
            "distance_m": 4200.0,
            "passenger_count": 55.0,
        }
    if tb == "night":
        return {
            "headway_mean": 900.0,
            "headway_std": 90.0,
            "headway_events": 3.0,
            "wait_avg": 180.0,
            "sched_arrivals": 3.0,
            "distance_m": 1900.0,
            "passenger_count": 18.0,
        }
    return {
        "headway_mean": 660.0,
        "headway_std": 110.0,
        "headway_events": 5.0,
        "wait_avg": 240.0,
        "sched_arrivals": 5.0,
        "distance_m": 3000.0,
        "passenger_count": 35.0,
    }


def build_window_rollup_row(
    *,
    condition_id: str,
    seed: int,
    scenario: Dict[str, Any],
    actions: Dict[int, int],
    rollout_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    from rewards.energy_proxy_model_v1 import compute_energy_proxy_from_components

    tb = str(scenario["time_band"]).strip().lower()
    base = base_by_time_band(tb)

    action_values = [int(v) for v in actions.values()]
    action_count = len(action_values)
    nonzero_action_count = sum(1 for v in action_values if int(v) != 0)

    decision_step_count = max(1, action_count)
    intervention_count = int(nonzero_action_count)

    acceleration_event_count = float(max(1, nonzero_action_count))
    hold_seconds = float(max(0, action_count - nonzero_action_count) * 1.0)
    passenger_count = float(base["passenger_count"])

    energy = compute_energy_proxy_from_components(
        distance_m=float(base["distance_m"]),
        acceleration_event_count=acceleration_event_count,
        hold_seconds=hold_seconds,
        passenger_served_count=passenger_count,
    )

    headway_events = int(base["headway_events"])
    wait_avg = float(max(30.0, base["wait_avg"] - min(30, nonzero_action_count)))
    headway_mean = float(max(60.0, base["headway_mean"] - min(20, nonzero_action_count)))
    headway_std = float(max(1.0, base["headway_std"] - min(10, nonzero_action_count)))
    wait_passenger_count = int(passenger_count)

    schedulable_arrival_count = int(base["sched_arrivals"])
    ontime_event_count = int(max(0, min(schedulable_arrival_count, round(schedulable_arrival_count * 0.75))))

    bunching_event_count = 1 if tb in {"peak", "offpeak"} else 0
    bunching_event_count = min(bunching_event_count, headway_events)

    row: Dict[str, Any] = {
        "condition_id": str(condition_id).upper(),
        "seed": int(seed),
        "window_id": str(scenario["window_id"]),
        "state_ts": str(scenario["state_ts"]),
        "service_date": str(scenario.get("service_date", str(scenario["state_ts"])[:10])),
        "time_band": tb,
        "evaluation_horizon_minutes": 30,
        "qwen_trigger_rate": 0.0,
        "effective_replay_step_minutes": float(scenario.get("effective_replay_step_minutes", 60)),
        "headway_mean_seconds": headway_mean,
        "headway_std_seconds": headway_std,
        "headway_sample_count": int(headway_events + 1),
        "bunching_event_count": int(bunching_event_count),
        "headway_event_count": int(headway_events),
        "wait_total_passenger_seconds": float(wait_avg * wait_passenger_count),
        "wait_passenger_count": int(wait_passenger_count),
        "ontime_event_count": int(ontime_event_count),
        "schedulable_arrival_count": int(schedulable_arrival_count),
        "intervention_count": int(intervention_count),
        "decision_step_count": int(decision_step_count),
        "energy_proxy_total": float(energy["energy_kwh_equiv"]),
        "source_mode": str(rollout_metadata["source_mode"]),
        "policy_action_count": int(action_count),
        "policy_nonzero_action_count": int(nonzero_action_count),
        "distance_m": float(base["distance_m"]),
        "acceleration_event_count": float(acceleration_event_count),
        "hold_seconds": float(hold_seconds),
        "passenger_served_count": float(passenger_count),
        "energy_proxy_per_passenger": float(energy["energy_proxy_per_passenger"]),
    }

    row.update(rollout_metadata)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A-family policy-source selectable rollout smoke writer"
    )
    parser.add_argument("--condition-id", choices=["A", "A90", "A80", "A70"], default="A")
    parser.add_argument(
        "--policy-source-mode",
        choices=["mock_smoke", "mappo_smoke", "mappo_actual"],
        required=True,
    )
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--checkpoint-validation-mode", default="")
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument(
        "--simulator-adapter",
        default="adapters.historical_replay_adapter.HistoricalReplayAdapter",
    )
    parser.add_argument("--scenario-index", default="")
    parser.add_argument("--scenario-row-index", type=int, default=0)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--causal-simulator", action="store_true")
    return parser.parse_args()


def main() -> int:
    training_dir = append_training_dir_to_path()
    project_root = training_dir.parent

    import pandas as pd
    from simulator_adapter_interface import load_adapter_class
    from policies.a_family_rollout_policy_integration import (
        AFamilyPolicyIntegrationConfig,
        build_a_family_policy_integration_plan,
        plan_to_dict,
    )

    args = parse_args()

    output_root = (
        Path(args.output_root)
        if args.output_root
        else project_root / "artifacts" / "experiment_A_v1" / "a_family_policy_rollout_smoke"
    )

    run_name = f"{args.condition_id}_{args.policy_source_mode}_seed_{int(args.seed):03d}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    status_path = run_dir / "status.json"
    manifest_path = run_dir / "run_manifest.json"
    actions_path = run_dir / "actions.json"
    metadata_path = run_dir / "policy_metadata.json"
    rollup_path = run_dir / "window_rollup.parquet"

    try:
        plan = build_a_family_policy_integration_plan(
            AFamilyPolicyIntegrationConfig(
                condition_id=str(args.condition_id),
                policy_source_mode=str(args.policy_source_mode),
                checkpoint_path=str(args.checkpoint_path or ""),
                checkpoint_validation_mode=str(args.checkpoint_validation_mode or ""),
                causal_simulator=bool(args.causal_simulator),
                qwen_train=False,
                qwen_inference=False,
                qwen_trigger_rate=0.0,
                allow_mock=bool(args.allow_mock),
            )
        )
        plan_payload = plan_to_dict(plan)

        scenario = load_first_scenario(
            str(args.scenario_index or ""),
            row_index=int(args.scenario_row_index),
        )

        adapter_cls = load_adapter_class(str(args.simulator_adapter))
        adapter = adapter_cls(
            {
                "condition_id": str(args.condition_id).upper(),
                "qwen_trigger_rate": 0.0,
            }
        )

        try:
            obs = adapter.reset(seed=int(args.seed), scenario_config=scenario)

            validation_report_path = run_dir / "checkpoint_validation_report.json"

            if str(args.policy_source_mode) == "mock_smoke":
                actions, action_metadata = make_mock_actions(obs)
            else:
                actions, action_metadata = build_mappo_actions(
                    obs,
                    checkpoint_path=str(args.checkpoint_path),
                    checkpoint_validation_mode=str(plan.checkpoint_validation_mode),
                    device=str(args.device),
                    seed=int(args.seed),
                    validation_report_path=validation_report_path,
                )

            rollout_metadata = build_rollout_metadata(
                condition_id=str(args.condition_id),
                policy_source_mode=str(args.policy_source_mode),
                action_metadata=action_metadata,
                causal_simulator=bool(args.causal_simulator),
                expected_source_mode=str(plan.expected_source_mode),
            )

            step_result = adapter.step(actions)

            row = build_window_rollup_row(
                condition_id=str(args.condition_id),
                seed=int(args.seed),
                scenario=scenario,
                actions=actions,
                rollout_metadata=rollout_metadata,
            )

            pd.DataFrame([row]).to_parquet(rollup_path, index=False)

            actions_payload = {
                "condition_id": str(args.condition_id).upper(),
                "policy_source_mode": str(args.policy_source_mode),
                "action_count": len(actions),
                "actions_first_50": [
                    {"agent_id": int(agent_id), "action": int(action)}
                    for agent_id, action in list(actions.items())[:50]
                ],
            }

            manifest = {
                "artifact_version": "a_family_policy_rollout_smoke_v1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "condition_id": str(args.condition_id).upper(),
                "seed": int(args.seed),
                "policy_source_mode": str(args.policy_source_mode),
                "checkpoint_path": str(args.checkpoint_path or ""),
                "checkpoint_validation_mode": str(plan.checkpoint_validation_mode),
                "allow_mock": bool(args.allow_mock),
                "causal_simulator": bool(args.causal_simulator),
                "integration_plan": plan_payload,
                "scenario": scenario,
                "simulator_adapter": str(args.simulator_adapter),
                "adapter_class": adapter_cls.__name__,
                "step_result": {
                    "terminated": bool(step_result.terminated),
                    "truncated": bool(step_result.truncated),
                    "reward_agent_count": int(len(step_result.rewards)),
                    "info": step_result.info,
                },
                "policy_metadata": rollout_metadata,
                "output_files": {
                    "status": str(status_path),
                    "manifest": str(manifest_path),
                    "actions": str(actions_path),
                    "policy_metadata": str(metadata_path),
                    "window_rollup": str(rollup_path),
                },
                "limitations": {
                    "historical_replay_is_noncausal": not bool(args.causal_simulator),
                    "smoke_output_not_for_performance_claims": str(args.policy_source_mode) != "mappo_actual",
                },
            }

            status = {
                "status": "PASS",
                "step": 52,
                "condition_id": str(args.condition_id).upper(),
                "policy_source_mode": str(args.policy_source_mode),
                "source_mode": str(rollout_metadata["source_mode"]),
                "policy_source": str(rollout_metadata["policy_source"]),
                "checkpoint_loaded": bool(rollout_metadata["checkpoint_loaded"]),
                "checkpoint_validator_ran": bool(rollout_metadata["checkpoint_validator_ran"]),
                "mock_action_used": bool(rollout_metadata["mock_action_used"]),
                "placeholder_fallback_used": bool(rollout_metadata["placeholder_fallback_used"]),
                "qwen_trigger_rate": float(rollout_metadata["qwen_trigger_rate"]),
                "actual_policy_claim_ready": bool(rollout_metadata["actual_policy_claim_ready"]),
                "causal_policy_claim_ready": bool(rollout_metadata["causal_policy_claim_ready"]),
                "window_rollup": str(rollup_path),
            }

            dump_json(actions_path, actions_payload)
            dump_json(metadata_path, rollout_metadata)
            dump_json(manifest_path, manifest)
            dump_json(status_path, status)

            print("[OK] Step 52 A-family policy rollout smoke completed")
            print(f"[OK] run_dir          : {run_dir}")
            print(f"[OK] status           : {status_path}")
            print(f"[OK] manifest         : {manifest_path}")
            print(f"[OK] policy_metadata  : {metadata_path}")
            print(f"[OK] window_rollup    : {rollup_path}")
            print(f"[OK] policy_source    : {status['policy_source']}")
            print(f"[OK] source_mode      : {status['source_mode']}")
            print("SMOKE PASS")
            return 0
        finally:
            try:
                adapter.close()
            except Exception:
                pass

    except Exception as exc:
        status = {
            "status": "FAIL",
            "step": 52,
            "condition_id": str(args.condition_id).upper(),
            "policy_source_mode": str(args.policy_source_mode),
            "error": str(exc),
        }
        dump_json(status_path, status)
        print("[FAIL] Step 52 A-family policy rollout smoke failed")
        print(f"[FAIL] run_dir: {run_dir}")
        print(f"[FAIL] error  : {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
