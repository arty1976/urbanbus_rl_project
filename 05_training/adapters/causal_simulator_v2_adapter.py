
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# --- H4M-AE-R9.8 fail-closed simulator authorization -------------------------------
import sys as _authz_sys
from pathlib import Path as _AuthzPath

for _authz_dir in (_AuthzPath(__file__).resolve().parent, _AuthzPath(__file__).resolve().parent.parent):
    if (_authz_dir / "simulator_authorization.py").exists():
        if str(_authz_dir) not in _authz_sys.path:
            _authz_sys.path.insert(0, str(_authz_dir))
        break
import simulator_authorization as _authz  # noqa: E402
# -----------------------------------------------------------------------------------

try:
    import torch
except Exception:
    torch = None

TRAINING_DIR = Path(__file__).resolve().parents[1]
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

try:
    from simulator_adapter_interface import SimulatorAdapterInterface, GraphSkeleton, StepResult
except Exception:
    class SimulatorAdapterInterface:
        pass

    @dataclass
    class GraphSkeleton:
        num_nodes: int
        num_edges: int
        node_features_dim: int
        edge_features_dim: int
        edge_index: Any
        edge_attr: Any

    @dataclass
    class StepResult:
        obs: Dict[str, Any]
        rewards: Dict[int, float]
        terminated: bool
        truncated: bool
        info: Dict[str, Any]


FORBIDDEN_DYNAMIC_SIGNAL_FIELDS = [
    "red_light_delay_seconds",
    "green_time_seconds",
    "cycle_length_seconds",
    "phase_sequence",
    "signal_offset_seconds",
    "real_time_signal_state",
    "queue_discharge_rate",
]

REQ_NODE = [
    "node_uid", "node_index", "signal_count_250m", "nearest_signal_distance_m",
    "pedestrian_signal_count_250m", "blink_signal_ratio_250m",
    "controlled_signal_ratio_250m", "signal_delay_risk_proxy",
    "intersection_complexity_proxy",
]

REQ_EDGE = [
    "src_idx", "dst_idx", "distance_m", "edge_signal_count",
    "edge_signal_density_per_km", "edge_nearest_signal_distance_m",
    "edge_control_complexity_proxy",
]


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _tensor(x: np.ndarray) -> Any:
    if torch is None:
        return x
    if x.dtype == np.bool_:
        return torch.from_numpy(x)
    if np.issubdtype(x.dtype, np.integer):
        return torch.from_numpy(x.astype(np.int64))
    return torch.from_numpy(x.astype(np.float32))


def _shape(x: Any) -> tuple:
    return tuple(x.shape)


def _require(df: pd.DataFrame, cols: List[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"{name} missing columns: {miss}")


def _reject_forbidden(df: pd.DataFrame, name: str) -> None:
    bad = [c for c in FORBIDDEN_DYNAMIC_SIGNAL_FIELDS if c in df.columns]
    if bad:
        raise RuntimeError(f"{name} contains forbidden dynamic signal fields: {bad}")


class CausalSimulatorV2Adapter(SimulatorAdapterInterface):
    """Step 101 static-signal-aware scaffold. Not a real signal phase simulator."""

    actor_obs_dim = 9
    critic_obs_dim = 8
    action_dim = 3

    def __init__(
        self,
        contract_path: str | Path,
        node_signal_features_path: str | Path,
        edge_signal_features_path: str | Path,
        num_agents: int = 8,
        episode_steps: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        self.contract_path = Path(contract_path)
        self.node_signal_features_path = Path(node_signal_features_path)
        self.edge_signal_features_path = Path(edge_signal_features_path)
        self._num_agents = int(num_agents)
        self.episode_steps = int(episode_steps)
        self.rng = np.random.default_rng(seed)
        self.current_step = 0
        self.closed = False

        self.contract = _json(self.contract_path)
        self._validate_contract(self.contract)

        self.node_df = pd.read_parquet(self.node_signal_features_path).copy()
        self.edge_df = pd.read_parquet(self.edge_signal_features_path).copy()
        _require(self.node_df, REQ_NODE, "node_signal_features")
        _require(self.edge_df, REQ_EDGE, "edge_signal_features")
        _reject_forbidden(self.node_df, "node_signal_features")
        _reject_forbidden(self.edge_df, "edge_signal_features")

        self.node_df["node_index"] = pd.to_numeric(self.node_df["node_index"], errors="raise").astype(int)
        self.node_df = self.node_df.sort_values("node_index").reset_index(drop=True)

        self.edge_df["src_idx"] = pd.to_numeric(self.edge_df["src_idx"], errors="raise").astype(int)
        self.edge_df["dst_idx"] = pd.to_numeric(self.edge_df["dst_idx"], errors="raise").astype(int)

        self.edge_index_np = self.edge_df[["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T
        self.edge_attr_np = self.edge_df[
            ["distance_m", "edge_signal_count", "edge_signal_density_per_km",
             "edge_nearest_signal_distance_m", "edge_control_complexity_proxy"]
        ].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        self.agent_node_indices = np.zeros((self._num_agents,), dtype=np.int64)
        self.demand_pressure = np.zeros((self._num_agents,), dtype=np.float32)

    @staticmethod
    def _validate_contract(contract: Dict[str, Any]) -> None:
        guard = contract.get("claim_guardrails", {})
        for key in [
            "trained_model", "performance_claim_allowed",
            "causal_performance_claim_allowed", "dynamic_signal_phase_claim_allowed",
            "red_light_delay_claim_allowed", "green_time_claim_allowed",
            "cycle_length_claim_allowed",
        ]:
            if bool(guard.get(key, False)):
                raise RuntimeError(f"Step 101 requires claim_guardrails.{key}=false")

        dyn = contract.get("feature_classes", {}).get("dynamic_signal_phase_features", {})
        if str(dyn.get("admission", "rejected")).lower() != "rejected":
            raise RuntimeError("dynamic_signal_phase_features must be rejected")

    @classmethod
    def from_step100_artifacts(
        cls, root: str | Path = ".", num_agents: int = 8,
        episode_steps: int = 8, seed: Optional[int] = None
    ) -> "CausalSimulatorV2Adapter":
        root = Path(root)
        return cls(
            contract_path=root / "artifacts/causal_simulator_v2_contract/causal_simulator_v2_input_contract.json",
            node_signal_features_path=root / "artifacts/signal_features_v2/node_signal_features.parquet",
            edge_signal_features_path=root / "artifacts/signal_features_v2/edge_signal_features.parquet",
            num_agents=num_agents,
            episode_steps=episode_steps,
            seed=seed,
        )

    def reset(self, seed: Optional[int] = None, scenario_config: Optional[dict] = None) -> Dict[str, Any]:
        _authz.require_capability("simulator_execution", site="adapters/causal_simulator_v2_adapter.py::reset")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_step = 0

        cfg = scenario_config or {}
        if "agent_node_indices" in cfg:
            idx = np.asarray(cfg["agent_node_indices"], dtype=np.int64)
            if len(idx) != self._num_agents:
                raise ValueError("agent_node_indices length must match num_agents")
            self.agent_node_indices = idx
        else:
            nodes = self.node_df["node_index"].to_numpy(dtype=np.int64)
            if len(nodes) < self._num_agents:
                raise RuntimeError("not enough node rows")
            pos = np.linspace(0, len(nodes) - 1, self._num_agents).round().astype(int)
            self.agent_node_indices = nodes[pos]

        lookup = self.node_df.set_index("node_index", drop=False)
        vals = []
        for node_idx in self.agent_node_indices:
            r = lookup.loc[int(node_idx)]
            vals.append(
                0.20
                + 0.03 * float(r["signal_count_250m"])
                + 0.05 * float(r["signal_delay_risk_proxy"])
                + float(self.rng.uniform(0.0, 0.02))
            )
        self.demand_pressure = np.asarray(vals, dtype=np.float32)
        return self._obs()

    def step(self, actions: Dict[int, Any]) -> StepResult:
        _authz.require_capability("simulator_execution", site="adapters/causal_simulator_v2_adapter.py::step")
        lookup = self.node_df.set_index("node_index", drop=False)
        rewards = {}
        for agent_id in range(self._num_agents):
            action = int(actions.get(agent_id, 0))
            action = max(0, min(action, self.action_dim - 1))
            r = lookup.loc[int(self.agent_node_indices[agent_id])]
            risk = float(r["signal_delay_risk_proxy"])
            comp = float(r["intersection_complexity_proxy"])

            if action == 0:
                delta, penalty = 0.025 + 0.010 * risk, 0.0
            elif action == 1:
                delta, penalty = -0.015 + 0.004 * comp, 0.03
            else:
                delta, penalty = -0.055 + 0.006 * comp, 0.05

            self.demand_pressure[agent_id] = np.clip(self.demand_pressure[agent_id] + delta, 0.0, 3.0)
            rewards[agent_id] = float(-self.demand_pressure[agent_id] - 0.02 * risk - penalty)

        self.current_step += 1
        return StepResult(
            obs=self._obs(),
            rewards=rewards,
            terminated=bool(self.current_step >= self.episode_steps),
            truncated=False,
            info={
                "adapter_version": "causal_simulator_v2_scaffold_step101",
                "source_mode": "static_signal_scaffold_nonperformance_v1",
                "performance_claim_allowed": False,
                "causal_performance_claim_allowed": False,
                "dynamic_signal_phase_claim_allowed": False,
                "red_light_delay_claim_allowed": False,
                "green_time_claim_allowed": False,
                "cycle_length_claim_allowed": False,
                "forbidden_dynamic_signal_fields": list(FORBIDDEN_DYNAMIC_SIGNAL_FIELDS),
                "state_transition_scope": "toy_scaffold_static_signal_context_only",
            },
        )

    def _obs(self) -> Dict[str, Any]:
        lookup = self.node_df.set_index("node_index", drop=False)
        max_count = max(float(pd.to_numeric(self.node_df["signal_count_250m"], errors="coerce").max()), 1.0)
        max_ped = max(float(pd.to_numeric(self.node_df["pedestrian_signal_count_250m"], errors="coerce").max()), 1.0)

        rows = []
        for i, node_idx in enumerate(self.agent_node_indices):
            r = lookup.loc[int(node_idx)]
            rows.append([
                float(self.demand_pressure[i]),
                float(r["signal_count_250m"]) / max_count,
                min(float(r["nearest_signal_distance_m"]), 1000.0) / 1000.0,
                float(r["pedestrian_signal_count_250m"]) / max_ped,
                float(r["blink_signal_ratio_250m"]),
                float(r["controlled_signal_ratio_250m"]),
                float(r["signal_delay_risk_proxy"]),
                float(r["intersection_complexity_proxy"]),
                float(self.current_step / max(1, self.episode_steps)),
            ])

        actor = np.asarray(rows, dtype=np.float32)
        critic = np.asarray([[
            float(actor[:, 0].mean()),
            float(actor[:, 0].max()),
            float(actor[:, 1].mean()),
            float(actor[:, 2].mean()),
            float(actor[:, 4].mean()),
            float(actor[:, 5].mean()),
            float(actor[:, 6].mean()),
            float(self.current_step / max(1, self.episode_steps)),
        ]], dtype=np.float32)

        return {
            "actor_obs": _tensor(actor),
            "critic_obs": _tensor(critic),
            "edge_index": _tensor(self.edge_index_np),
            "edge_attr": _tensor(self.edge_attr_np),
            "active_bus_mask": _tensor(np.ones((self._num_agents,), dtype=bool)),
            "action_mask": _tensor(np.ones((self._num_agents, self.action_dim), dtype=bool)),
            "time_band": "scaffold",
            "state_ts": f"step_{self.current_step:04d}",
            "agent_ids": list(range(self._num_agents)),
        }

    def get_graph_skeleton(self) -> GraphSkeleton:
        return GraphSkeleton(
            num_nodes=int(self.node_df["node_index"].nunique()),
            num_edges=int(len(self.edge_df)),
            node_features_dim=self.actor_obs_dim,
            edge_features_dim=int(self.edge_attr_np.shape[1]),
            edge_index=_tensor(self.edge_index_np),
            edge_attr=_tensor(self.edge_attr_np),
        )

    def compute_kpis(self, trajectory: Optional[List[StepResult]] = None) -> Dict[str, Optional[float]]:
        reward_values = []
        if trajectory:
            for sr in trajectory:
                reward_values.extend(float(v) for v in sr.rewards.values())
        return {
            "cv_headway": None,
            "avg_wait_seconds": None,
            "bunching_rate": None,
            "on_time_rate": None,
            "intervention_rate": None,
            "energy_proxy": None,
            "scaffold_avg_reward": float(np.mean(reward_values)) if reward_values else None,
            "static_signal_context_loaded": 1.0,
            "performance_claim_allowed": 0.0,
            "causal_performance_claim_allowed": 0.0,
        }

    def close(self) -> None:
        self.closed = True

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def observation_space(self) -> Any:
        return {
            "actor_obs_dim": self.actor_obs_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "edge_attr_dim": int(self.edge_attr_np.shape[1]),
        }

    @property
    def action_space(self) -> Any:
        return {"type": "discrete", "n": self.action_dim, "meaning": {0: "noop", 1: "hold", 2: "serve_or_move"}}
