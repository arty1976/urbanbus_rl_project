from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


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


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def validate_experiment_A_contract(contract: Dict[str, Any]) -> None:
    if contract.get("condition_id") != "A":
        raise RuntimeError(f"condition_id must be A, got {contract.get('condition_id')}")
    if bool(contract.get("qwen_train", False)):
        raise RuntimeError("Experiment A must have qwen_train=false")
    if bool(contract.get("qwen_inference", False)):
        raise RuntimeError("Experiment A must have qwen_inference=false")
    if int(contract.get("evaluation_horizon_minutes", -1)) != 30:
        raise RuntimeError("evaluation_horizon_minutes must be 30")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 40 neural MAPPO inference adapter smoke runner"
    )
    parser.add_argument("--contract", default="", help="experiment_A_contract.json path")
    parser.add_argument(
        "--output-root",
        default="",
        help="output root for Step 40 smoke artifacts",
    )
    parser.add_argument(
        "--simulator-adapter",
        default="adapters.historical_replay_adapter.HistoricalReplayAdapter",
    )
    parser.add_argument("--checkpoint-path", default="", help="optional MAPPO checkpoint path")
    parser.add_argument(
        "--require-checkpoint",
        action="store_true",
        help="fail if checkpoint is missing",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    parser.add_argument("--hidden-dim", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    training_dir = Path(__file__).resolve().parent
    project_root = training_dir.parent

    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))

    from simulator_adapter_interface import load_adapter_class
    from policies.mappo_neural_inference_adapter import NeuralMAPPOInferenceAdapter

    contract_path: Optional[Path]
    if args.contract:
        contract_path = Path(args.contract)
    else:
        contract_path = first_existing(
            [
                project_root / "artifacts" / "experiment_A_v1" / "experiment_A_contract.json",
                training_dir / "artifacts" / "experiment_A_v1" / "experiment_A_contract.json",
            ]
        )

    if contract_path is None or not contract_path.exists():
        raise SystemExit("[STOP] experiment_A_contract.json not found. Pass --contract explicitly.")

    contract = load_json_any_encoding(contract_path)
    validate_experiment_A_contract(contract)

    output_root = (
        Path(args.output_root)
        if args.output_root
        else project_root / "artifacts" / "experiment_A_v1" / "neural_inference_smoke"
    )

    run_dir = output_root / f"seed_{int(args.seed):03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter_cls = load_adapter_class(args.simulator_adapter)
    env = adapter_cls(
        {
            "condition_id": "A",
            "qwen_trigger_rate": 0.0,
        }
    )

    scenario_config = {
        "window_id": "step40_neural_inference_smoke",
        "state_ts": "2024-01-01T08:00:00Z",
        "time_band": "offpeak",
        "snapshot_path": None,
        "effective_replay_step_minutes": 60,
    }

    try:
        obs = env.reset(seed=int(args.seed), scenario_config=scenario_config)

        policy = NeuralMAPPOInferenceAdapter.from_obs(
            obs,
            checkpoint_path=str(args.checkpoint_path or ""),
            device=str(args.device),
            deterministic=not bool(args.stochastic),
            require_checkpoint=bool(args.require_checkpoint),
            hidden_dim=int(args.hidden_dim),
            seed=int(args.seed),
        )

        actions, policy_info = policy.select_actions(obs)
        step_result = env.step(actions)

        actions_preview = {
            "actions_first_20": [
                {"agent_id": int(agent_id), "action": int(action)}
                for agent_id, action in list(actions.items())[:20]
            ],
            "policy_info": policy_info,
        }

        manifest = {
            "artifact_version": "step40_neural_mappo_inference_adapter_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "condition_id": "A",
            "qwen_train": False,
            "qwen_inference": False,
            "qwen_trigger_rate": 0.0,
            "seed": int(args.seed),
            "contract_path": str(contract_path),
            "simulator_adapter": str(args.simulator_adapter),
            "simulator_adapter_class": adapter_cls.__name__,
            "scenario_config": scenario_config,
            "checkpoint_path": str(args.checkpoint_path or ""),
            "require_checkpoint": bool(args.require_checkpoint),
            "device_requested": str(args.device),
            "policy_adapter": "policies.mappo_neural_inference_adapter.NeuralMAPPOInferenceAdapter",
            "status": "neural_inference_adapter_smoke_completed",
            "phase1_limitations": {
                "historical_replay_is_non_causal": True,
                "actions_do_not_change_future_state": True,
                "kpi_comparison_allowed": False,
            },
            "obs_summary": {
                "agent_count": len(obs.get("agent_ids", [])),
                "time_band": obs.get("time_band"),
                "state_ts": obs.get("state_ts"),
            },
            "step_result_summary": {
                "terminated": bool(step_result.terminated),
                "truncated": bool(step_result.truncated),
                "reward_agent_count": len(step_result.rewards),
                "info": step_result.info,
            },
            "output_files": {
                "manifest": str(run_dir / "neural_inference_manifest.json"),
                "actions_preview": str(run_dir / "actions_preview.json"),
                "status": str(run_dir / "status.json"),
            },
        }

        status = {
            "status": "PASS",
            "step": 40,
            "message": "Neural MAPPO inference adapter placement smoke completed.",
            "checkpoint_loaded": bool(
                policy_info["checkpoint_status"].get("checkpoint_loaded", False)
            ),
            "checkpoint_status": policy_info["checkpoint_status"],
            "qwen_trigger_rate": 0.0,
            "causal_comparison_allowed": False,
        }

        dump_json(run_dir / "actions_preview.json", actions_preview)
        dump_json(run_dir / "neural_inference_manifest.json", manifest)
        dump_json(run_dir / "status.json", status)

        print("[OK] Step 40 neural MAPPO inference adapter smoke completed")
        print(f"[OK] run_dir          : {run_dir}")
        print(f"[OK] manifest         : {run_dir / 'neural_inference_manifest.json'}")
        print(f"[OK] actions_preview  : {run_dir / 'actions_preview.json'}")
        print(f"[OK] status           : {run_dir / 'status.json'}")
        print(f"[OK] checkpoint_loaded: {status['checkpoint_loaded']}")
        print("SMOKE PASS")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
