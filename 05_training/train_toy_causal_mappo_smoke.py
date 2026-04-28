from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


TRAINING_DIR = Path(__file__).resolve().parent
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


def safe_rmtree(path: Path) -> None:
    if not path.exists():
        return

    def on_error(func, target, exc_info):
        try:
            import os
            os.chmod(target, 0o700)
            func(target)
        except Exception:
            pass

    for attempt in range(5):
        try:
            try:
                shutil.rmtree(path, onexc=on_error)
            except TypeError:
                shutil.rmtree(path, onerror=on_error)
            return
        except PermissionError:
            time.sleep(0.5 + 0.5 * attempt)

    raise PermissionError(f"could not remove locked directory: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_args(args: argparse.Namespace) -> None:
    if str(args.condition_id).upper() not in VALID_CONDITIONS:
        raise SystemExit(f"--condition-id must be one of {sorted(VALID_CONDITIONS)}")
    if int(args.num_agents) < 5 or int(args.num_agents) > 10:
        raise SystemExit("--num-agents must be between 5 and 10")
    if int(args.steps) < 1:
        raise SystemExit("--steps must be >= 1")
    if int(args.epochs) < 1:
        raise SystemExit("--epochs must be >= 1")
    if not (str(args.device) == "cpu" or str(args.device).startswith("cuda")):
        raise SystemExit("--device must be cpu or cuda*")


def write_dry_run_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = Path(args.output_root)
    if args.clean:
        safe_rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "artifact_version": "toy_causal_mappo_smoke_dry_run_v1_step88",
        "output_root": str(output_root),
        "condition_id": str(args.condition_id).upper(),
        "seed": int(args.seed),
        "num_agents": int(args.num_agents),
        "steps": int(args.steps),
        "epochs": int(args.epochs),
        "device": str(args.device),
        "reward_version": REWARD_VERSION,
        "reward_claim_boundary": REWARD_CLAIM_BOUNDARY,
        "trained_model": False,
        "performance_claim_allowed": False,
        "causal_comparison_allowed": True,
        "smoke_training_only": True,
        "dry_run_contract": True,
        "created_at_utc": utc_now(),
        "note": "dry-run contract only; no rollout or optimizer update executed",
    }

    manifest_path = output_root / "training_manifest_dry_run.json"
    dump_json(manifest_path, manifest)
    print("[OK] Step 88 dry-run contract completed")
    print(f"[OK] manifest: {manifest_path}")
    return manifest


def import_training_deps():
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.distributions import Categorical
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"[STOP] missing training dependency: {exc.name}. "
            "Install numpy/torch in 05_training\\.venv or use --dry-run-contract."
        ) from exc

    from adapters.causal_simulator_adapter import CausalSimulatorAdapter

    return np, torch, nn, F, Categorical, CausalSimulatorAdapter


def build_model(torch, nn, actor_obs_dim: int, critic_obs_dim: int, hidden_dim: int, action_dim: int):
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


def set_all_seeds(seed: int, np, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_reward_info(result) -> None:
    info = result.info
    if info.get("reward_version") != REWARD_VERSION:
        raise RuntimeError(f"reward_version mismatch: {info.get('reward_version')}")
    if info.get("reward_claim_boundary") != REWARD_CLAIM_BOUNDARY:
        raise RuntimeError("reward_claim_boundary mismatch")
    metrics = info.get("reward_metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("StepResult.info.reward_metrics must be dict")

    missing = [k for k in PHASE2_12_KPIS if k not in metrics]
    if missing:
        raise RuntimeError(f"reward_metrics missing 12-KPI keys: {missing}")

    reward_total = float(info["reward_total"])
    if not math.isfinite(reward_total):
        raise RuntimeError("reward_total is not finite")

    for agent_id, value in result.rewards.items():
        if abs(float(value) - reward_total) > 1e-8:
            raise RuntimeError(f"agent {agent_id} reward is not broadcast reward_total")


def run_one_epoch(
    args: argparse.Namespace,
    epoch: int,
    model,
    optimizer,
    device,
    np,
    torch,
    F,
    Categorical,
    CausalSimulatorAdapter,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    adapter = CausalSimulatorAdapter(
        {
            "condition_id": str(args.condition_id).upper(),
            "seed": int(args.seed),
            "num_agents": int(args.num_agents),
            "max_steps": int(args.steps),
            "control_step_minutes": 30,
            "evaluation_horizon_minutes": 30,
        }
    )

    obs = adapter.reset(
        seed=int(args.seed),
        scenario_config={
            "window_id": "step88_toy_causal_mappo_smoke",
            "state_ts": "2024-01-01T08:00:00Z",
            "time_band": "peak",
        },
    )

    log_probs = []
    entropies = []
    values = []
    rewards = []
    trace_rows: List[Dict[str, Any]] = []

    for step_idx in range(int(args.steps)):
        actor_obs = torch.as_tensor(obs["actor_obs"], dtype=torch.float32, device=device)
        critic_obs = torch.as_tensor(obs["critic_obs"], dtype=torch.float32, device=device)

        logits = model.forward_actor(actor_obs)
        dist = Categorical(logits=logits)
        actions_tensor = dist.sample()
        log_prob = dist.log_prob(actions_tensor).mean()
        entropy = dist.entropy().mean()

        value = model.forward_critic(critic_obs).reshape(-1).mean()

        actions = {
            int(agent_id): int(actions_tensor[agent_id].detach().cpu().item())
            for agent_id in range(int(args.num_agents))
        }

        result = adapter.step(actions)
        assert_reward_info(result)

        reward_total = float(result.info["reward_total"])
        metrics = result.info["reward_metrics"]

        log_probs.append(log_prob)
        entropies.append(entropy)
        values.append(value)
        rewards.append(reward_total)

        trace_rows.append(
            {
                "epoch": int(epoch),
                "step": int(step_idx),
                "reward_total": reward_total,
                "passenger_service_rate": float(metrics["passenger_service_rate"]),
                "passenger_wait_p95_seconds": float(metrics["passenger_wait_p95_seconds"]),
                "energy_proxy_per_passenger": float(metrics["energy_proxy_per_passenger"]),
                "fleet_reduction_ratio": float(metrics["fleet_reduction_ratio"]),
            }
        )

        obs = result.obs
        if bool(result.terminated) or bool(result.truncated):
            break

    gamma = 0.99
    running = 0.0
    returns = []
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()

    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    values_t = torch.stack(values)
    log_probs_t = torch.stack(log_probs)
    entropies_t = torch.stack(entropies)

    advantages = returns_t - values_t.detach()
    policy_loss = -(log_probs_t * advantages).mean()
    value_loss = F.mse_loss(values_t, returns_t)
    entropy_bonus = entropies_t.mean()
    total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

    if not torch.isfinite(total_loss):
        raise RuntimeError("total_loss is not finite")

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()

    grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm += float(param.grad.detach().norm().cpu().item())

    if not math.isfinite(grad_norm) or grad_norm <= 0.0:
        raise RuntimeError(f"invalid gradient norm: {grad_norm}")

    optimizer.step()

    loss_summary = {
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "value_loss": float(value_loss.detach().cpu().item()),
        "entropy": float(entropy_bonus.detach().cpu().item()),
        "total_loss": float(total_loss.detach().cpu().item()),
        "grad_norm": float(grad_norm),
    }

    for row in trace_rows:
        row.update(loss_summary)

    adapter.close()
    return trace_rows, loss_summary


def run_smoke_training(args: argparse.Namespace) -> Dict[str, Any]:
    np, torch, nn, F, Categorical, CausalSimulatorAdapter = import_training_deps()

    device = torch.device(str(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")

    output_root = Path(args.output_root)
    if args.clean:
        safe_rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(int(args.seed), np, torch)

    probe_adapter = CausalSimulatorAdapter(
        {
            "condition_id": str(args.condition_id).upper(),
            "seed": int(args.seed),
            "num_agents": int(args.num_agents),
            "max_steps": int(args.steps),
        }
    )
    probe_obs = probe_adapter.reset(
        seed=int(args.seed),
        scenario_config={
            "window_id": "step88_probe",
            "state_ts": "2024-01-01T08:00:00Z",
            "time_band": "peak",
        },
    )
    actor_obs_dim = int(probe_obs["actor_obs"].shape[1])
    critic_obs_dim = int(probe_obs["critic_obs"].shape[1])
    probe_adapter.close()

    action_dim = 3
    model = build_model(
        torch=torch,
        nn=nn,
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        hidden_dim=int(args.hidden_dim),
        action_dim=action_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    all_trace_rows: List[Dict[str, Any]] = []
    loss_summaries: List[Dict[str, float]] = []

    for epoch in range(int(args.epochs)):
        epoch_rows, loss_summary = run_one_epoch(
            args=args,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            device=device,
            np=np,
            torch=torch,
            F=F,
            Categorical=Categorical,
            CausalSimulatorAdapter=CausalSimulatorAdapter,
        )
        all_trace_rows.extend(epoch_rows)
        loss_summaries.append(loss_summary)

    if not all_trace_rows:
        raise RuntimeError("no training trace rows generated")

    rewards = [float(r["reward_total"]) for r in all_trace_rows]
    reward_summary = {
        "reward_total_mean": float(sum(rewards) / len(rewards)),
        "reward_total_min": float(min(rewards)),
        "reward_total_max": float(max(rewards)),
        "reward_total_count": int(len(rewards)),
    }

    last_loss = loss_summaries[-1]
    checkpoint_path = checkpoint_dir / "toy_mappo_smoke_checkpoint.pt"

    checkpoint = {
        "artifact_version": "toy_causal_mappo_smoke_checkpoint_v1_step88",
        "condition_id": str(args.condition_id).upper(),
        "seed": int(args.seed),
        "num_agents": int(args.num_agents),
        "steps": int(args.steps),
        "epochs": int(args.epochs),
        "hidden_dim": int(args.hidden_dim),
        "actor_obs_dim": int(actor_obs_dim),
        "critic_obs_dim": int(critic_obs_dim),
        "action_dim": int(action_dim),
        "reward_version": REWARD_VERSION,
        "reward_claim_boundary": REWARD_CLAIM_BOUNDARY,
        "trained_model": False,
        "performance_claim_allowed": False,
        "causal_comparison_allowed": True,
        "smoke_training_only": True,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_summary": {
            "total_transitions": int(len(all_trace_rows) * int(args.num_agents)),
            "loss_summary": last_loss,
            "reward_summary": reward_summary,
        },
        "created_at_utc": utc_now(),
    }

    torch.save(checkpoint, checkpoint_path)

    trace_path = output_root / "training_trace.csv"
    with open(trace_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "epoch",
            "step",
            "reward_total",
            "policy_loss",
            "value_loss",
            "entropy",
            "total_loss",
            "grad_norm",
            "passenger_service_rate",
            "passenger_wait_p95_seconds",
            "energy_proxy_per_passenger",
            "fleet_reduction_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_trace_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    manifest_path = output_root / "training_manifest.json"
    manifest = {
        "artifact_version": "toy_causal_mappo_smoke_v1_step88",
        "output_root": str(output_root),
        "checkpoint_path": str(checkpoint_path),
        "condition_id": str(args.condition_id).upper(),
        "seed": int(args.seed),
        "num_agents": int(args.num_agents),
        "steps": int(args.steps),
        "epochs": int(args.epochs),
        "device": str(device),
        "reward_version": REWARD_VERSION,
        "reward_claim_boundary": REWARD_CLAIM_BOUNDARY,
        "trained_model": False,
        "performance_claim_allowed": False,
        "causal_comparison_allowed": True,
        "smoke_training_only": True,
        "total_transitions": int(len(all_trace_rows) * int(args.num_agents)),
        "loss_summary": last_loss,
        "reward_summary": reward_summary,
        "reward_metric_keys": PHASE2_12_KPIS,
        "training_trace": str(trace_path),
        "created_at_utc": utc_now(),
        "note": "toy causal MAPPO smoke scaffold only; not a paper-level performance claim",
    }
    dump_json(manifest_path, manifest)

    print("[OK] Step 88 toy causal MAPPO smoke training completed")
    print(f"[OK] output_root               : {output_root}")
    print(f"[OK] checkpoint_path           : {checkpoint_path}")
    print(f"[OK] training_manifest         : {manifest_path}")
    print(f"[OK] training_trace            : {trace_path}")
    print(f"[OK] total_transitions         : {manifest['total_transitions']}")
    print(f"[OK] reward_version            : {manifest['reward_version']}")
    print(f"[OK] performance_claim_allowed : {manifest['performance_claim_allowed']}")
    print(f"[OK] smoke_training_only       : {manifest['smoke_training_only']}")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="artifacts/phase2_toy_causal_mappo_smoke")
    parser.add_argument("--condition-id", default="A")
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--num-agents", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--dry-run-contract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.condition_id = str(args.condition_id).upper()
    validate_args(args)

    if args.dry_run_contract:
        write_dry_run_manifest(args)
    else:
        run_smoke_training(args)


if __name__ == "__main__":
    main()
