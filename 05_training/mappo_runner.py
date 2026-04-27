import json
import random
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from torch_geometric.data import Batch
except Exception:
    Batch = None


EXPECTED_SHARED_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
]


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            pass
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_rng_state() -> Dict[str, Any]:
    state = {
        "python_random_state": repr(random.getstate()),
        "numpy_random_state": repr(np.random.get_state()),
    }
    if torch is not None:
        state["torch_rng_state"] = torch.get_rng_state().cpu().tolist()
        if torch.cuda.is_available():
            state["torch_cuda_rng_state_all"] = [
                x.cpu().tolist() for x in torch.cuda.get_rng_state_all()
            ]
    return state


class RewardNormalizer:
    def __init__(self, window_size: int = 1000, clip_value: float = 10.0) -> None:
        self.window_size = window_size
        self.clip_value = clip_value
        self.buffer: deque[float] = deque(maxlen=window_size)

    def normalize(self, rewards: List[float]) -> List[float]:
        clipped = [float(np.clip(r, -self.clip_value, self.clip_value)) for r in rewards]
        for r in clipped:
            self.buffer.append(r)

        arr = np.array(self.buffer, dtype=np.float32)
        mean = float(arr.mean()) if len(arr) else 0.0
        std = float(arr.std()) if len(arr) else 1.0
        std = max(std, 1e-6)

        return [float((r - mean) / std) for r in clipped]


def compute_gae(
    rewards: List[float],
    values: List[float],
    next_values: List[float],
    terminated: List[bool],
    truncated: List[bool],
    gamma: float,
    gae_lambda: float,
) -> Dict[str, List[float]]:
    n = len(rewards)
    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0

    for t in reversed(range(n)):
        # terminated=True means no bootstrap from next state.
        # truncated=True still allows bootstrap because the state boundary
        # is artificial (for example, horizon cutoff) rather than terminal.
        bootstrap_mask = 0.0 if terminated[t] else 1.0
        delta = rewards[t] + gamma * next_values[t] * bootstrap_mask - values[t]
        gae = delta + gamma * gae_lambda * bootstrap_mask * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return {"advantages": advantages, "returns": returns}


def entropy_coef_for_time_band(
    time_band: str,
    progress: float,
    offpeak_coef: float = 0.01,
    peak_coef: float = 0.03,
    min_coef: float = 0.001,
) -> float:
    base = peak_coef if str(time_band).lower() == "peak" else offpeak_coef
    decay = max(0.1, 1.0 - progress)
    return float(max(min_coef, base * decay))


def maybe_pyg_batch(data_list: List[Any]) -> Optional[Any]:
    if Batch is None:
        return None
    return Batch.from_data_list(data_list)


@dataclass
class FreezePhaseConfig:
    phase_a_end_env_steps: int = 20000
    phase_b_end_env_steps: int = 50000
    phase_b_lr_scale: float = 0.1


@dataclass
class RewardNormConfig:
    enabled: bool = True
    window_size: int = 1000
    clip_value: float = 10.0


@dataclass
class DistributedConfig:
    executor_backend: str = "single_process"
    distributed_enabled: bool = False


@dataclass
class RunnerConfig:
    root_dir: str
    experiment_contract_path: str
    run_root_dir: str
    simulator_adapter_path: str = ""
    shared_policy: bool = True
    use_ctde: bool = True
    use_active_bus_mask: bool = True
    store_edge_index_in_rollout_buffer: bool = False
    grad_norm_clip: float = 0.5
    checkpoint_save_rng_state: bool = True
    qwen_trigger_rate_expected: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    reward_norm: RewardNormConfig = field(default_factory=RewardNormConfig)
    freeze_schedule: FreezePhaseConfig = field(default_factory=FreezePhaseConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


class MAPPOExperimentRunner:
    def __init__(self, config: RunnerConfig) -> None:
        self.config = config
        self.reward_normalizer = RewardNormalizer(
            window_size=config.reward_norm.window_size,
            clip_value=config.reward_norm.clip_value,
        )

    def validate_contract(self, contract: Dict[str, Any]) -> None:
        if contract.get("condition_id") != "A":
            raise RuntimeError(f"condition_id must be A, got {contract.get('condition_id')}")

        if bool(contract.get("qwen_train", False)) or bool(contract.get("qwen_inference", False)):
            raise RuntimeError("Experiment A must have qwen_train=false and qwen_inference=false")

        if int(contract.get("evaluation_horizon_minutes", -1)) != 30:
            raise RuntimeError("evaluation_horizon_minutes must be 30")

        if list(contract.get("shared_kpis", [])) != EXPECTED_SHARED_KPIS:
            raise RuntimeError("shared_kpis do not match expected baseline contract")

        fairness = contract.get("fairness_constraints", {})
        if not fairness.get("same_initial_state", False):
            raise RuntimeError("same_initial_state must be true")
        if not fairness.get("same_exogenous_events", False):
            raise RuntimeError("same_exogenous_events must be true")
        if not fairness.get("same_eval_window", False):
            raise RuntimeError("same_eval_window must be true")

    def freeze_phase(self, total_env_steps: int) -> Dict[str, Any]:
        fs = self.config.freeze_schedule
        if total_env_steps < fs.phase_a_end_env_steps:
            return {"phase": "A", "gat_frozen": True, "encoder_lr_scale": 0.0}
        if total_env_steps < fs.phase_b_end_env_steps:
            return {"phase": "B", "gat_frozen": False, "encoder_lr_scale": fs.phase_b_lr_scale}
        return {"phase": "C", "gat_frozen": False, "encoder_lr_scale": 1.0}

    def build_run_manifest(self, contract: Dict[str, Any], seed: int) -> Dict[str, Any]:
        return {
            "condition_id": "A",
            "condition_name": "pure_mappo_baseline",
            "seed": seed,
            "shared_policy": self.config.shared_policy,
            "ctde_enabled": self.config.use_ctde,
            "use_active_bus_mask": self.config.use_active_bus_mask,
            "store_edge_index_in_rollout_buffer": self.config.store_edge_index_in_rollout_buffer,
            "grad_norm_clip": self.config.grad_norm_clip,
            "checkpoint_save_rng_state": self.config.checkpoint_save_rng_state,
            "qwen_trigger_rate_expected": self.config.qwen_trigger_rate_expected,
            "evaluation_horizon_minutes": contract["evaluation_horizon_minutes"],
            "shared_kpis": contract["shared_kpis"],
            "reward_normalization": asdict(self.config.reward_norm),
            "freeze_schedule": asdict(self.config.freeze_schedule),
            "distributed": asdict(self.config.distributed),
            "gae_contract": {
                "terminated_truncated_separated": True,
                "shared_critic_individual_advantage": True,
            },
            "entropy_contract": {
                "adaptive_peak_entropy": True,
                "offpeak_entropy_coef": 0.01,
                "peak_entropy_coef": 0.03,
            },
            "logging_contract": {
                "log_qwen_trigger_rate": True,
                "log_approx_kl": True,
                "log_clip_fraction": True,
                "log_explained_var": True,
            },
        }

    def write_checkpoint_stub(self, run_dir: Path, seed: int) -> None:
        """
        Write a smoke checkpoint stub plus a contract preview.

        This does not create an actual trained MAPPO checkpoint.
        It records the exact checkpoint metadata contract that future H200
        training checkpoints must satisfy before neural inference can accept them.
        """
        payload = {
            "seed": seed,
            "rng_state_included": self.config.checkpoint_save_rng_state,
            "rng_state": get_rng_state() if self.config.checkpoint_save_rng_state else None,
            "checkpoint_contract_version": "mappo_checkpoint_contract_v1",
            "checkpoint_artifact_version": "mappo_policy_checkpoint_v1",
            "trained_model": False,
            "performance_claim_allowed": False,
            "note": (
                "Smoke checkpoint stub only. Actual MAPPO checkpoints must include "
                "model_state_dict and pass policies/validate_mappo_checkpoint.py."
            ),
        }
        dump_json(run_dir / "checkpoint_stub.json", payload)

        try:
            from policies.mappo_checkpoint_builder import write_checkpoint_contract_preview_json

            write_checkpoint_contract_preview_json(
                run_dir / "checkpoint_contract_preview.json",
                project_root=Path(self.config.root_dir),
                seed=seed,
                trained_model=False,
                performance_claim_allowed=False,
                extra_metadata={
                    "runner": "MAPPOExperimentRunner",
                    "runner_condition_id": "A",
                    "runner_shared_policy": self.config.shared_policy,
                    "runner_ctde_enabled": self.config.use_ctde,
                    "reward_norm": asdict(self.config.reward_norm),
                    "freeze_schedule": asdict(self.config.freeze_schedule),
                    "distributed": asdict(self.config.distributed),
                },
            )
        except Exception as exc:
            dump_json(
                run_dir / "checkpoint_contract_preview_error.json",
                {
                    "status": "checkpoint_contract_preview_failed",
                    "error": str(exc),
                    "note": (
                        "Runner smoke can continue, but Step 45 contract preview "
                        "should be fixed before actual H200 checkpoint generation."
                    ),
                },
            )

    def _build_stub_scenario_config(self) -> Dict[str, Any]:
        return {
            "window_id": None,
            "state_ts": None,
            "time_band": "offpeak",
            "snapshot_path": None,
            "effective_replay_step_minutes": 60,
        }

    def run_seed(self, contract: Dict[str, Any], seed: int) -> None:
        run_dir = Path(self.config.run_root_dir) / f"seed_{seed:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        set_all_seeds(seed)

        dump_json(run_dir / "run_manifest.json", self.build_run_manifest(contract, seed))
        dump_json(
            run_dir / "status.json",
            {
                "status": "starting",
                "seed": seed,
                "qwen_trigger_rate": 0.0,
            },
        )
        self.write_checkpoint_stub(run_dir, seed)

        if not self.config.simulator_adapter_path:
            dump_json(
                run_dir / "status.json",
                {
                    "status": "adapter_missing",
                    "seed": seed,
                    "reason": "simulator adapter not configured",
                    "qwen_trigger_rate": 0.0,
                    "approx_kl": None,
                    "clip_fraction": None,
                    "explained_var": None,
                },
            )
            return

        try:
            from simulator_adapter_interface import load_adapter_class
            adapter_cls = load_adapter_class(self.config.simulator_adapter_path)
        except Exception as exc:
            dump_json(
                run_dir / "status.json",
                {
                    "status": "adapter_load_failed",
                    "seed": seed,
                    "simulator_adapter_path": self.config.simulator_adapter_path,
                    "error": str(exc),
                    "qwen_trigger_rate": 0.0,
                },
            )
            return

        scenario_config = self._build_stub_scenario_config()

        dump_json(
            run_dir / "status.json",
            {
                "status": "smoke_running",
                "seed": seed,
                "simulator_adapter_path": self.config.simulator_adapter_path,
                "adapter_class": adapter_cls.__name__,
                "scenario_config": scenario_config,
                "qwen_trigger_rate": 0.0,
            },
        )

        adapter = None
        try:
            adapter = adapter_cls(
                {
                    "condition_id": "A",
                    "qwen_trigger_rate": 0.0,
                }
            )

            obs = adapter.reset(seed=seed, scenario_config=scenario_config)
            graph_skeleton = adapter.get_graph_skeleton()

            smoke_step_info = None
            try:
                dummy_actions = {agent_id: 0 for agent_id in obs["agent_ids"]}
                step_result = adapter.step(dummy_actions)
                smoke_step_info = {
                    "terminated": step_result.terminated,
                    "truncated": step_result.truncated,
                    "reward_agent_count": len(step_result.rewards),
                    "effective_replay_step_minutes": step_result.info.get("effective_replay_step_minutes", 60),
                }
            except Exception as step_exc:
                smoke_step_info = {
                    "step_skipped": True,
                    "step_error": str(step_exc),
                }

            dump_json(
                run_dir / "status.json",
                {
                    "status": "smoke_completed",
                    "seed": seed,
                    "simulator_adapter_path": self.config.simulator_adapter_path,
                    "adapter_class": adapter_cls.__name__,
                    "num_agents": getattr(adapter, "num_agents", None),
                    "observation_space": getattr(adapter, "observation_space", None),
                    "action_space": getattr(adapter, "action_space", None),
                    "graph_skeleton_summary": {
                        "num_nodes": getattr(graph_skeleton, "num_nodes", None),
                        "num_edges": getattr(graph_skeleton, "num_edges", None),
                        "node_features_dim": getattr(graph_skeleton, "node_features_dim", None),
                        "edge_features_dim": getattr(graph_skeleton, "edge_features_dim", None),
                    },
                    "obs_summary": {
                        "time_band": obs.get("time_band"),
                        "state_ts": obs.get("state_ts"),
                        "agent_count": len(obs.get("agent_ids", [])),
                    },
                    "smoke_step_info": smoke_step_info,
                    "effective_replay_step_minutes": scenario_config["effective_replay_step_minutes"],
                    "note": "adapter instantiated; reset and get_graph_skeleton succeeded. replay mode does not causally apply actions. causal evaluation is not supported in Phase 1.",
                    "qwen_trigger_rate": 0.0,
                    "approx_kl": None,
                    "clip_fraction": None,
                    "explained_var": None,
                },
            )
        except Exception as exc:
            dump_json(
                run_dir / "status.json",
                {
                    "status": "causal_not_supported",
                    "seed": seed,
                    "simulator_adapter_path": self.config.simulator_adapter_path,
                    "error": str(exc),
                    "note": "adapter loaded but Phase 1 smoke execution could not complete; causal evaluation remains unsupported in replay mode.",
                    "qwen_trigger_rate": 0.0,
                },
            )
        finally:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:
                    pass
