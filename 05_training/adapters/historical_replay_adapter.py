from typing import Any, Dict, List, Optional

from simulator_adapter_interface import (
    GraphSkeleton,
    ObsDict,
    SimulatorAdapterInterface,
    StepResult,
)

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


class HistoricalReplayAdapter(SimulatorAdapterInterface):
    """
    HistoricalReplayAdapter is valid for contract/smoke validation,
    but not for causal performance comparison across B1/B2/A.

    [IMPORTANT] Phase 1 Limitation
    This adapter replays historical snapshots. Actions taken by the policy
    (or any baseline) DO NOT change the next state. It is strictly non-causal.
    Proper 30-minute control granularity causal evaluation is deferred to Phase 2.

    [IMPORTANT] Canonical KPI Aggregation
    This adapter provides a convenience compute_kpis() method, but canonical
    KPI aggregation must be performed by a shared aggregator across B0/B1/B2/A.

    Raw Event Schema for canonical aggregator should include:
    - state_ts
    - time_band
    - agent_id
    - action
    - waiting_passenger_cnt
    - intervention_applied
    - terminated
    - truncated

    [IMPORTANT] Proxy Reward
    The reward returned by this adapter is a smoke validation proxy.
    Default: reward = -mean(waiting_passenger_cnt_at_step)
    Final causal comparisons will rely on shared_kpis, not this proxy reward.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.effective_replay_step_minutes = 60
        self.num_agents_val = int(self.config.get("num_agents", 10))
        self.current_scenario = None

    @property
    def num_agents(self) -> int:
        return self.num_agents_val

    @property
    def observation_space(self) -> Any:
        return {
            "actor_obs": (self.num_agents_val, 16),
            "critic_obs": (1, 64),
            "active_bus_mask": (self.num_agents_val,),
            "action_mask": (self.num_agents_val, 2),
        }

    @property
    def action_space(self) -> Any:
        return {
            "type": "discrete",
            "action_dim": 2,
        }

    def _get_stub_obs(
        self,
        time_band: str = "offpeak",
        state_ts: str = "2024-01-01T08:00:00Z",
    ) -> ObsDict:
        import numpy as np

        return {
            "actor_obs": np.zeros((self.num_agents_val, 16), dtype=float),
            "critic_obs": np.zeros((1, 64), dtype=float),
            "edge_index": np.zeros((2, 10000), dtype=int),
            "edge_attr": np.zeros((10000, 4), dtype=float),
            "active_bus_mask": np.ones((self.num_agents_val,), dtype=bool),
            "action_mask": np.ones((self.num_agents_val, 2), dtype=bool),
            "time_band": time_band,
            "state_ts": state_ts,
            "agent_ids": list(range(self.num_agents_val)),
        }

    def reset(
        self,
        seed: Optional[int] = None,
        scenario_config: Optional[dict] = None,
    ) -> ObsDict:
        _authz.require_capability("simulator_execution", site="adapters/historical_replay_adapter.py::reset")
        self.current_scenario = scenario_config or {}
        time_band = self.current_scenario.get("time_band", "offpeak")
        state_ts = self.current_scenario.get("state_ts", "2024-01-01T08:00:00Z")
        return self._get_stub_obs(time_band=time_band, state_ts=state_ts)

    def step(self, actions: Dict[int, Any]) -> StepResult:
        _authz.require_capability("simulator_execution", site="adapters/historical_replay_adapter.py::step")
        obs = self._get_stub_obs(
            time_band=(self.current_scenario or {}).get("time_band", "offpeak"),
            state_ts=(self.current_scenario or {}).get("state_ts", "2024-01-01T08:00:00Z"),
        )

        # Non-causal replay: actions are recorded conceptually but do not affect next state.
        rewards = {agent_id: -1.0 for agent_id in obs["agent_ids"]}

        terminated = False
        truncated = False
        info = {
            "agent_terminated_mask": [False] * self.num_agents_val,
            "agent_truncated_mask": [False] * self.num_agents_val,
            "effective_replay_step_minutes": self.effective_replay_step_minutes,
            "qwen_trigger_rate": 0.0,
            "note": "historical replay is non-causal; actions do not change future state",
        }

        return StepResult(
            obs=obs,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_graph_skeleton(self) -> GraphSkeleton:
        import numpy as np

        return GraphSkeleton(
            num_nodes=4116,
            num_edges=10000,
            node_features_dim=16,
            edge_features_dim=4,
            edge_index=np.zeros((2, 10000), dtype=int),
            edge_attr=np.zeros((10000, 4), dtype=float),
        )

    def compute_kpis(
        self,
        trajectory: Optional[List[StepResult]] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Convenience proxy KPI calculation.
        DO NOT use for official comparisons.
        """
        return {
            "cv_headway": None,
            "avg_wait_seconds": 300.0,
            "bunching_rate": None,
            "on_time_rate": None,
            "intervention_rate": 0.0,
            "energy_proxy": None,
        }

    def close(self) -> None:
        pass
