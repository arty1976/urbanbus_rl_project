import sys
import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:
    torch = Any


"""
[IMPORTANT] 2-Phase Evaluation Structure

- Phase 1: Contract/smoke validation via historical replay.
  HistoricalReplayAdapter operates in this phase only.
  Actions taken by the policy DO NOT change the next state (Not Causal).

- Phase 2: Causal policy evaluation via future simulator-backed adapter.
  This is where true baseline vs. MAPPO comparisons occur.
"""

SHARED_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
]


@dataclass
class GraphSkeleton:
    """
    Static graph skeleton for the transport graph.
    This should be directly consumable by the GATv2 encoder path.
    """
    num_nodes: int
    num_edges: int
    node_features_dim: int
    edge_features_dim: int
    edge_index: Any
    edge_attr: Any


try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class ObsDict(TypedDict):
    """
    CTDE (Centralized Training Decentralized Execution=중앙집중 학습 분산 실행)
    observation structure.
    """
    actor_obs: Any            # [num_agents, actor_dim]
    critic_obs: Any           # [num_agents, critic_dim] or [1, global_dim]
    edge_index: Any           # [2, num_edges]
    edge_attr: Any            # [num_edges, edge_dim]
    active_bus_mask: Any      # [num_agents] or [MAX_BUSES], bus-agent mask
    action_mask: Any          # [num_agents, action_dim]
    time_band: str
    state_ts: str
    agent_ids: List[int]


@dataclass
class StepResult:
    obs: ObsDict
    rewards: Dict[int, float]   # agent_id -> reward
    terminated: bool            # episode-level termination
    truncated: bool             # episode-level truncation
    info: Dict[str, Any]        # may include agent_terminated_mask / agent_truncated_mask


class SimulatorAdapterInterface(ABC):
    """
    Common interface for environment adapters.

    HistoricalReplayAdapter is valid for contract/smoke validation,
    but not for causal performance comparison across B1/B2/A.
    """

    @abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        scenario_config: Optional[dict] = None,
    ) -> ObsDict:
        pass

    @abstractmethod
    def step(self, actions: Dict[int, Any]) -> StepResult:
        pass

    @abstractmethod
    def get_graph_skeleton(self) -> GraphSkeleton:
        pass

    @abstractmethod
    def compute_kpis(
        self,
        trajectory: Optional[List[StepResult]] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Convenience KPI calculation inside the adapter.

        Canonical KPI aggregation for B0/B1/B2/A must be handled
        by a shared external aggregator, not by the adapter itself.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @property
    @abstractmethod
    def num_agents(self) -> int:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Any:
        pass

    @property
    @abstractmethod
    def action_space(self) -> Any:
        pass


def load_adapter_class(import_path: str) -> type:
    """
    Load an adapter class from a Python import path.

    Because the current directory is named '05_training', importing via a
    path like '05_training.adapters...' is awkward. Instead, we inject this
    file's parent directory into sys.path and support import paths like:

        adapters.historical_replay_adapter.HistoricalReplayAdapter
    """
    import os

    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    module_path, class_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
