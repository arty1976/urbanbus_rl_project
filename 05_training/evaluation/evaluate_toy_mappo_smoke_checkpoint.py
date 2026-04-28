from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


PHASE2_12_KPIS = [
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

VALID_CONDITIONS = {"A", "A90", "A80", "A70"}
REWARD_VERSION = "mappo_reward_v1"
REWARD_CLAIM_BOUNDARY = "toy_causal_training_reward_contract_not_paper_performance_claim"


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def scenario_rows(time_bands: List[str], windows_per_time_band: int) -> List[Dict[str, Any]]:
    base_ts = pd.Timestamp("2024-01-01T08:00:00Z")
    rows: List[Dict[str, Any]] = []
    idx = 0
    for time_band in time_bands:
        tb = str(time_band).lower()
        for window_idx in range(windows_per_time_band):
            ts = base_ts + pd.Timedelta(minutes=30 * idx)
            idx += 1
            rows.append(
                {
                    "window_id": f"step91_{tb}_{window_idx + 1:03d}",
                    "state_ts": ts.isoformat().replace("+00:00", "Z"),
                    "service_date": ts.date().isoformat(),
                    "time_band": tb,
                }
            )
    return rows


def import_eval_deps():
    try:
        import numpy as np
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"[STOP] missing dependency: {exc.name}. Step 91 evaluator requires numpy and torch."
        ) from exc

    from adapters.causal_simulator_adapter import CausalSimulatorAdapter

    return np, torch, nn, CausalSimulatorAdapter


def build_eval_model(torch, nn, actor_obs_dim: int, critic_obs_dim: int, hidden_dim: int, action_dim: int):
    class TinyMAPPOModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(actor_obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(critic_obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        def forward_actor(self, actor_obs):
            return self.actor(actor_obs)

        def forward_critic(self, critic_obs):
            return self.critic(critic_obs)

    return TinyMAPPOModel()


def load_checkpoint(checkpoint_path: Path, torch) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise RuntimeError(f"checkpoint missing: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint must be dict")

    required = [
        "artifact_version",
        "actor_obs_dim",
        "critic_obs_dim",
        "action_dim",
        "hidden_dim",
        "model_state_dict",
        "reward_version",
        "reward_claim_boundary",
        "trained_model",
        "performance_claim_allowed",
        "smoke_training_only",
    ]
    missing = [k for k in required if k not in checkpoint]
    if missing:
        raise RuntimeError(f"checkpoint missing keys: {missing}")

    if checkpoint["reward_version"] != REWARD_VERSION:
        raise RuntimeError("checkpoint reward_version mismatch")
    if checkpoint["reward_claim_boundary"] != REWARD_CLAIM_BOUNDARY:
        raise RuntimeError("checkpoint reward_claim_boundary mismatch")
    if checkpoint["trained_model"] is not False:
        raise RuntimeError("checkpoint trained_model must be false")
    if checkpoint["performance_claim_allowed"] is not False:
        raise RuntimeError("checkpoint performance_claim_allowed must be false")
    if checkpoint["smoke_training_only"] is not True:
        raise RuntimeError("checkpoint smoke_training_only must be true")

    return checkpoint


def validate_reward_info(result) -> None:
    info = result.info
    if info.get("reward_version") != REWARD_VERSION:
        raise RuntimeError("StepResult reward_version mismatch")
    if info.get("reward_claim_boundary") != REWARD_CLAIM_BOUNDARY:
        raise RuntimeError("StepResult reward_claim_boundary mismatch")
    metrics = info.get("reward_metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("StepResult reward_metrics must be dict")
    missing = [k for k in PHASE2_12_KPIS if k not in metrics]
    if missing:
        raise RuntimeError(f"StepResult reward_metrics missing 12-KPI keys: {missing}")

    reward_total = float(info["reward_total"])
    if not math.isfinite(reward_total):
        raise RuntimeError("reward_total must be finite")
    for agent_id, value in result.rewards.items():
        if abs(float(value) - reward_total) > 1e-8:
            raise RuntimeError(f"agent {agent_id} reward is not team reward broadcast")


def run_cmd(cmd: List[str], cwd: Path) -> str:
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if completed.returncode != 0:
        print("[STDOUT]")
        print(completed.stdout)
        print("[STDERR]")
        print(completed.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    print(completed.stdout)
    return completed.stdout


def build_toy_eval_contract(contract_path: Path, condition_id: str, seeds: List[int], time_bands: List[str]) -> None:
    payload = {
        "artifact_version": "toy_mappo_smoke_eval_contract_v1_step91",
        "phase": "Phase-2",
        "condition_id": "PHASE2_TOY_MAPPO_SMOKE_EVAL",
        "condition_name": "toy_mappo_smoke_checkpoint_evaluation",
        "qwen_train": False,
        "qwen_inference": False,
        "evaluation_horizon_minutes": 30,
        "seeds": seeds,
        "time_bands": time_bands,
        "shared_kpis": PHASE2_12_KPIS,
        "fairness_constraints": {
            "same_initial_state": True,
            "same_exogenous_events": True,
            "same_eval_window": True,
        },
        "evaluated_condition_id": condition_id,
        "trained_model": False,
        "performance_claim_allowed": False,
        "smoke_evaluation_only": True,
        "claim_boundary": "toy causal smoke checkpoint evaluation only; not a paper-level performance claim",
    }
    dump_json(contract_path, payload)


def rollout_checkpoint(args: argparse.Namespace) -> Dict[str, Any]:
    np, torch, nn, CausalSimulatorAdapter = import_eval_deps()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, torch=torch)

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    condition_id = str(args.condition_id).upper()
    if condition_id not in VALID_CONDITIONS:
        raise SystemExit(f"--condition-id must be one of {sorted(VALID_CONDITIONS)}")

    output_root = Path(args.output_root)
    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    model = build_eval_model(
        torch=torch,
        nn=nn,
        actor_obs_dim=int(checkpoint["actor_obs_dim"]),
        critic_obs_dim=int(checkpoint["critic_obs_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        action_dim=int(checkpoint["action_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    time_bands = [x.lower() for x in parse_csv(args.time_bands)]
    scenarios = scenario_rows(time_bands=time_bands, windows_per_time_band=int(args.windows_per_time_band))
    seeds = [int(args.seed)]

    rollouts_root = output_root / "rollouts"
    seed_dir = rollouts_root / condition_id / "rollouts" / f"seed_{int(args.seed):03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    scenario_index_path = rollouts_root / "scenario_index.parquet"
    scenario_df = pd.DataFrame(scenarios)
    scenario_df.to_parquet(scenario_index_path, index=False)

    raw_frames: List[pd.DataFrame] = []
    rollup_rows: List[Dict[str, Any]] = []
    reward_trace_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for scenario in scenarios:
            adapter = CausalSimulatorAdapter(
                {
                    "condition_id": condition_id,
                    "seed": int(args.seed),
                    "num_agents": int(args.num_agents),
                    "max_steps": int(args.steps),
                    "control_step_minutes": 30,
                    "evaluation_horizon_minutes": 30,
                }
            )
            obs = adapter.reset(seed=int(args.seed), scenario_config=scenario)
            last_reward_metrics: Dict[str, Any] | None = None

            for step_idx in range(int(args.steps)):
                actor_obs = torch.as_tensor(obs["actor_obs"], dtype=torch.float32, device=device)
                logits = model.forward_actor(actor_obs)

                if args.action_mode == "argmax":
                    actions_tensor = torch.argmax(logits, dim=-1)
                elif args.action_mode == "sample":
                    dist = torch.distributions.Categorical(logits=logits)
                    actions_tensor = dist.sample()
                else:
                    raise SystemExit("--action-mode must be argmax or sample")

                actions = {
                    int(agent_id): int(actions_tensor[agent_id].detach().cpu().item())
                    for agent_id in range(int(args.num_agents))
                }

                result = adapter.step(actions)
                validate_reward_info(result)

                metrics = result.info["reward_metrics"]
                last_reward_metrics = dict(metrics)
                reward_trace_rows.append(
                    {
                        "window_id": scenario["window_id"],
                        "time_band": scenario["time_band"],
                        "step": int(step_idx),
                        "reward_total": float(result.info["reward_total"]),
                        "passenger_service_rate": float(metrics["passenger_service_rate"]),
                        "passenger_wait_p95_seconds": float(metrics["passenger_wait_p95_seconds"]),
                        "energy_proxy_per_passenger": float(metrics["energy_proxy_per_passenger"]),
                        "fleet_reduction_ratio": float(metrics["fleet_reduction_ratio"]),
                    }
                )

                obs = result.obs
                if bool(result.terminated) or bool(result.truncated):
                    break

            raw_frames.append(pd.DataFrame(adapter.get_raw_events()))
            rollup_row = adapter.build_window_rollup_row()
            if last_reward_metrics is None:
                raise RuntimeError("no reward metrics captured for scenario before rollup write")
            for kpi in PHASE2_12_KPIS:
                if kpi not in rollup_row:
                    rollup_row[kpi] = float(last_reward_metrics[kpi])
            if "baseline_bus_count" in last_reward_metrics and "baseline_bus_count" not in rollup_row:
                rollup_row["baseline_bus_count"] = float(last_reward_metrics["baseline_bus_count"])
            if "active_bus_count" in last_reward_metrics and "active_bus_count" not in rollup_row:
                rollup_row["active_bus_count"] = float(last_reward_metrics["active_bus_count"])
            rollup_rows.append(rollup_row)
            adapter.close()

    raw_df = pd.concat(raw_frames, ignore_index=True)
    rollup_df = pd.DataFrame(rollup_rows)

    raw_path = seed_dir / "raw_events.parquet"
    rollup_path = seed_dir / "window_rollup.parquet"
    reward_trace_path = seed_dir / "reward_trace.csv"
    run_manifest_path = seed_dir / "run_manifest.json"

    raw_df.to_parquet(raw_path, index=False)
    rollup_df.to_parquet(rollup_path, index=False)
    pd.DataFrame(reward_trace_rows).to_csv(reward_trace_path, index=False, encoding="utf-8-sig")

    run_manifest = {
        "artifact_version": "toy_mappo_smoke_checkpoint_eval_rollout_v1_step91",
        "condition_id": condition_id,
        "seed": int(args.seed),
        "num_agents": int(args.num_agents),
        "steps": int(args.steps),
        "scenario_count": int(len(scenarios)),
        "raw_event_rows": int(len(raw_df)),
        "window_rollup_rows": int(len(rollup_df)),
        "reward_trace_rows": int(len(reward_trace_rows)),
        "checkpoint_path": str(checkpoint_path),
        "reward_version": REWARD_VERSION,
        "reward_claim_boundary": REWARD_CLAIM_BOUNDARY,
        "trained_model": False,
        "performance_claim_allowed": False,
        "smoke_evaluation_only": True,
        "causal_comparison_allowed": True,
        "outputs": {
            "raw_events": str(raw_path),
            "window_rollup": str(rollup_path),
            "reward_trace": str(reward_trace_path),
            "run_manifest": str(run_manifest_path),
        },
        "note": "toy causal MAPPO smoke checkpoint evaluation only; not a paper-level performance claim",
    }
    dump_json(run_manifest_path, run_manifest)

    contract_path = output_root / "toy_mappo_smoke_eval_contract.json"
    build_toy_eval_contract(contract_path, condition_id=condition_id, seeds=seeds, time_bands=time_bands)

    canonical_root = output_root / "canonical_eval"
    run_cmd(
        [
            sys.executable,
            str(TRAINING_DIR / "evaluation" / "canonical_kpi_aggregator.py"),
            "--mode",
            "official_rollup",
            "--contract",
            str(contract_path),
            "--input-root",
            str(rollouts_root),
            "--scenario-index",
            str(scenario_index_path),
            "--output-root",
            str(canonical_root),
            "--smoke",
        ],
        cwd=TRAINING_DIR.parent,
    )

    inspection_root = output_root / "inspection"
    inspection_ran = False
    if args.skip_inspector:
        inspection_root.mkdir(parents=True, exist_ok=True)
        dump_json(
            inspection_root / "inspection_skipped.json",
            {
                "artifact_version": "toy_mappo_smoke_checkpoint_inspection_skipped_v1_step93",
                "condition_id": condition_id,
                "reason": "single non-baseline condition evaluation inside Step 93 matrix; combined matrix inspector runs later",
                "performance_claim_allowed": False,
                "smoke_evaluation_only": True,
                "created_at_utc": utc_now(),
            },
        )
    else:
        run_cmd(
            [
                sys.executable,
                str(TRAINING_DIR / "evaluation" / "inspect_toy_causal_a_family_kpis.py"),
                "--canonical-root",
                str(canonical_root),
                "--output-root",
                str(inspection_root),
                "--conditions",
                condition_id,
            ],
            cwd=TRAINING_DIR.parent,
        )
        inspection_ran = True

    evaluation_manifest_path = output_root / "evaluation_manifest.json"
    reward_values = [float(r["reward_total"]) for r in reward_trace_rows]
    evaluation_manifest = {
        "artifact_version": "toy_mappo_smoke_checkpoint_evaluation_v1_step91",
        "output_root": str(output_root),
        "checkpoint_path": str(checkpoint_path),
        "rollouts_root": str(rollouts_root),
        "canonical_root": str(canonical_root),
        "inspection_root": str(inspection_root),
        "inspection_ran": bool(inspection_ran),
        "scenario_index": str(scenario_index_path),
        "condition_id": condition_id,
        "seed": int(args.seed),
        "num_agents": int(args.num_agents),
        "steps": int(args.steps),
        "time_bands": time_bands,
        "windows_per_time_band": int(args.windows_per_time_band),
        "action_mode": str(args.action_mode),
        "reward_version": REWARD_VERSION,
        "reward_claim_boundary": REWARD_CLAIM_BOUNDARY,
        "trained_model": False,
        "performance_claim_allowed": False,
        "smoke_evaluation_only": True,
        "causal_comparison_allowed": True,
        "reward_metric_keys": PHASE2_12_KPIS,
        "raw_event_rows": int(len(raw_df)),
        "window_rollup_rows": int(len(rollup_df)),
        "reward_trace_rows": int(len(reward_trace_rows)),
        "reward_summary": {
            "reward_total_mean": float(sum(reward_values) / len(reward_values)),
            "reward_total_min": float(min(reward_values)),
            "reward_total_max": float(max(reward_values)),
        },
        "outputs": {
            "raw_events": str(raw_path),
            "window_rollup": str(rollup_path),
            "reward_trace": str(reward_trace_path),
            "canonical_eval": str(canonical_root),
            "inspection": str(inspection_root),
        },
        "created_at_utc": utc_now(),
        "note": "toy causal MAPPO smoke checkpoint evaluation only; not a paper-level performance claim",
    }
    dump_json(evaluation_manifest_path, evaluation_manifest)

    print("[OK] Step 91 toy MAPPO smoke checkpoint evaluation completed")
    print(f"[OK] output_root               : {output_root}")
    print(f"[OK] evaluation_manifest       : {evaluation_manifest_path}")
    print(f"[OK] raw_events                : {raw_path}")
    print(f"[OK] window_rollup             : {rollup_path}")
    print(f"[OK] canonical_root            : {canonical_root}")
    print(f"[OK] inspection_root           : {inspection_root}")
    print(f"[OK] reward_version            : {REWARD_VERSION}")
    print(f"[OK] performance_claim_allowed : False")
    print(f"[OK] smoke_evaluation_only     : True")

    return evaluation_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", default="artifacts/phase2_toy_mappo_smoke_checkpoint_eval")
    parser.add_argument("--condition-id", default="A")
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--time-bands", default="peak,offpeak,night")
    parser.add_argument("--windows-per-time-band", type=int, default=1)
    parser.add_argument("--action-mode", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-inspector", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_checkpoint(args)


if __name__ == "__main__":
    main()
