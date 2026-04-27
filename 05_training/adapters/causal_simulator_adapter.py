from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from simulator_adapter_interface import (
    GraphSkeleton,
    ObsDict,
    SimulatorAdapterInterface,
    StepResult,
)

from rewards.mappo_reward_v1 import compute_total_reward


ACTION_HOLD = 0
ACTION_DISPATCH = 1
ACTION_SKIP = 2

ACTION_NAMES = {
    ACTION_HOLD: "hold",
    ACTION_DISPATCH: "dispatch",
    ACTION_SKIP: "skip",
}

SHARED_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
]


@dataclass
class ToyCausalState:
    step_index: int
    queues: np.ndarray
    bus_positions: np.ndarray
    last_arrival_seconds: np.ndarray
    demand_generated_total: int
    passenger_served_total: int
    queue_person_seconds: float
    wait_passenger_count: int
    energy_proxy_total: float
    intervention_count: int
    decision_step_count: int
    headway_samples: List[float]
    bunching_event_count: int
    ontime_event_count: int
    schedulable_arrival_count: int


class CausalSimulatorAdapter(SimulatorAdapterInterface):
    """
    Phase 2 toy/minimal causal simulator adapter.

    Scope:
    - toy Suseong-gu corridor
    - 8 stops
    - 5 to 10 bus agents
    - 30-minute control granularity

    Unlike HistoricalReplayAdapter, actions do change the next state.
    """

    adapter_version = "causal_simulator_adapter_v1_step77"
    source_mode = "causal_toy_suseong_v1"
    graph_scope = "toy_suseong_beomeo_manchon"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        self.condition_id = str(self.config.get("condition_id", "A"))
        self.seed = int(self.config.get("seed", 1))

        self.num_agents_val = int(self.config.get("num_agents", 8))
        if self.num_agents_val < 5 or self.num_agents_val > 10:
            raise ValueError("Step 77 toy simulator requires 5 <= num_agents <= 10")

        self.control_step_minutes = int(self.config.get("control_step_minutes", 30))
        if self.control_step_minutes != 30:
            raise ValueError("Step 77 contract requires control_step_minutes == 30")

        self.evaluation_horizon_minutes = int(self.config.get("evaluation_horizon_minutes", 30))
        self.max_steps = int(self.config.get("max_steps", 4))

        self.capacity_per_bus = int(self.config.get("capacity_per_bus", 20))
        self.target_headway_seconds = float(self.config.get("target_headway_seconds", 900.0))
        self.bunching_threshold_seconds = float(self.config.get("bunching_threshold_seconds", 360.0))
        self.ontime_tolerance_seconds = float(self.config.get("ontime_tolerance_seconds", 360.0))

        self.k_dist_kwh_per_m = float(self.config.get("k_dist_kwh_per_m", 0.0012))
        self.k_acc_kwh_per_event = float(self.config.get("k_acc_kwh_per_event", 0.1800))
        self.k_idle_kwh_per_sec = float(self.config.get("k_idle_kwh_per_sec", 0.0080))

        self.node_names = [
            "Suseong-gu Office",
            "Beomeo",
            "Beomeo Sageori",
            "Manchon",
            "Manchon Market",
            "Daeryun",
            "Children Hall",
            "Suseong Lake East",
        ]
        self.num_nodes_val = len(self.node_names)

        self.edge_index, self.edge_attr = self._build_toy_graph()

        self.rng: Optional[np.random.Generator] = None
        self.state: Optional[ToyCausalState] = None
        self.current_scenario: Dict[str, Any] = {}
        self.raw_events: List[Dict[str, Any]] = []

    @property
    def num_agents(self) -> int:
        return self.num_agents_val

    @property
    def observation_space(self) -> Any:
        return {
            "actor_obs": (self.num_agents_val, 16),
            "critic_obs": (1, 64),
            "active_bus_mask": (self.num_agents_val,),
            "action_mask": (self.num_agents_val, 3),
            "control_step_minutes": self.control_step_minutes,
        }

    @property
    def action_space(self) -> Any:
        return {
            "type": "discrete",
            "action_dim": 3,
            "actions": {
                "0": "hold",
                "1": "dispatch",
                "2": "skip",
            },
        }

    def _build_toy_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        src = np.arange(self.num_nodes_val, dtype=np.int64)
        dst = (src + 1) % self.num_nodes_val
        edge_index = np.stack([src, dst], axis=0)

        distance_m = np.array([650, 780, 520, 700, 600, 850, 900, 760], dtype=float)
        time_sec = distance_m / 1000.0 / 20.0 * 3600.0
        generalized_cost = time_sec + distance_m * 0.05
        long_edge = (distance_m >= 5000).astype(float)

        edge_attr = np.stack([distance_m, time_sec, generalized_cost, long_edge], axis=1)
        return edge_index.astype(np.int64), edge_attr.astype(float)

    def _parse_ts(self, value: str) -> datetime:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)

    def _state_ts(self) -> str:
        base_text = self.current_scenario.get("state_ts", "2024-01-01T08:00:00+00:00")
        base = self._parse_ts(base_text)

        if self.state is None:
            step_index = 0
        else:
            step_index = self.state.step_index

        ts = base + timedelta(minutes=self.control_step_minutes * step_index)
        return ts.isoformat().replace("+00:00", "Z")

    def _service_date(self) -> str:
        return self._parse_ts(self._state_ts()).date().isoformat()

    def _base_demand_rate(self, time_band: str) -> np.ndarray:
        tb = str(time_band).strip().lower()

        if tb == "peak":
            return np.array([8, 10, 7, 9, 6, 5, 6, 4], dtype=float)

        if tb == "night":
            return np.array([1, 2, 1, 2, 1, 1, 1, 1], dtype=float)

        return np.array([3, 4, 3, 4, 2, 2, 3, 2], dtype=float)

    def _initial_queues(self, time_band: str) -> np.ndarray:
        if self.rng is None:
            raise RuntimeError("rng is not initialized")

        base = self._base_demand_rate(time_band)
        return self.rng.poisson(lam=base * 1.5).astype(float)

    def _build_obs(self) -> ObsDict:
        if self.state is None:
            raise RuntimeError("state is not initialized")

        actor_obs = np.zeros((self.num_agents_val, 16), dtype=float)
        total_queue = float(self.state.queues.sum())
        mean_queue = float(self.state.queues.mean())

        for agent_id in range(self.num_agents_val):
            pos = int(self.state.bus_positions[agent_id])
            next_pos = (pos + 1) % self.num_nodes_val

            actor_obs[agent_id, 0] = pos / max(1, self.num_nodes_val - 1)
            actor_obs[agent_id, 1] = self.state.queues[pos]
            actor_obs[agent_id, 2] = self.state.queues[next_pos]
            actor_obs[agent_id, 3] = total_queue
            actor_obs[agent_id, 4] = mean_queue
            actor_obs[agent_id, 5] = self.state.step_index / max(1, self.max_steps)
            actor_obs[agent_id, 6] = self.control_step_minutes
            actor_obs[agent_id, 7] = self.capacity_per_bus

        critic_obs = np.zeros((1, 64), dtype=float)
        critic_obs[0, 0] = total_queue
        critic_obs[0, 1] = mean_queue
        critic_obs[0, 2] = self.state.passenger_served_total
        critic_obs[0, 3] = self.state.demand_generated_total
        critic_obs[0, 4] = self.state.energy_proxy_total
        critic_obs[0, 5] = self.state.intervention_count
        critic_obs[0, 6] = len(self.state.headway_samples)
        critic_obs[0, 7] = self.state.step_index

        return {
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
            "edge_index": self.edge_index.copy(),
            "edge_attr": self.edge_attr.copy(),
            "active_bus_mask": np.ones((self.num_agents_val,), dtype=bool),
            "action_mask": np.ones((self.num_agents_val, 3), dtype=bool),
            "time_band": str(self.current_scenario.get("time_band", "offpeak")),
            "state_ts": self._state_ts(),
            "agent_ids": list(range(self.num_agents_val)),
        }

    def reset(
        self,
        seed: Optional[int] = None,
        scenario_config: Optional[dict] = None,
    ) -> ObsDict:
        if seed is not None:
            self.seed = int(seed)

        self.rng = np.random.default_rng(self.seed)
        self.current_scenario = scenario_config or {}

        time_band = str(self.current_scenario.get("time_band", "offpeak")).strip().lower()
        queues = self._initial_queues(time_band)
        positions = np.arange(self.num_agents_val, dtype=np.int64) % self.num_nodes_val

        self.state = ToyCausalState(
            step_index=0,
            queues=queues,
            bus_positions=positions,
            last_arrival_seconds=np.full((self.num_nodes_val,), np.nan, dtype=float),
            demand_generated_total=int(queues.sum()),
            passenger_served_total=0,
            queue_person_seconds=0.0,
            wait_passenger_count=max(1, int(queues.sum())),
            energy_proxy_total=0.0,
            intervention_count=0,
            decision_step_count=0,
            headway_samples=[],
            bunching_event_count=0,
            ontime_event_count=0,
            schedulable_arrival_count=0,
        )

        self.raw_events = []
        return self._build_obs()

    def _parse_action(self, action: Any) -> int:
        if isinstance(action, dict):
            if "action_id" in action:
                return self._parse_action(action["action_id"])
            if "action" in action:
                return self._parse_action(action["action"])

        if isinstance(action, str):
            lowered = action.strip().lower()
            for action_id, name in ACTION_NAMES.items():
                if lowered == name:
                    return action_id
            try:
                return int(lowered)
            except ValueError:
                return ACTION_DISPATCH

        try:
            action_id = int(action)
        except Exception:
            return ACTION_DISPATCH

        if action_id not in ACTION_NAMES:
            return ACTION_DISPATCH

        return action_id

    def _segment_distance(self, src_node: int) -> float:
        return float(self.edge_attr[int(src_node) % self.num_nodes_val, 0])

    def _serve_at_node(self, node_idx: int) -> int:
        if self.state is None:
            raise RuntimeError("state is not initialized")

        served = int(min(float(self.capacity_per_bus), self.state.queues[node_idx]))
        self.state.queues[node_idx] -= served
        self.state.passenger_served_total += served
        return served

    def _record_arrival_headway(self, node_idx: int, current_seconds: float) -> None:
        if self.state is None:
            raise RuntimeError("state is not initialized")

        previous = self.state.last_arrival_seconds[node_idx]

        if np.isfinite(previous):
            headway = float(max(0.0, current_seconds - previous))
            self.state.headway_samples.append(headway)
            self.state.schedulable_arrival_count += 1

            if headway < self.bunching_threshold_seconds:
                self.state.bunching_event_count += 1

            lower = self.target_headway_seconds - self.ontime_tolerance_seconds
            upper = self.target_headway_seconds + self.ontime_tolerance_seconds
            if lower <= headway <= upper:
                self.state.ontime_event_count += 1

        self.state.last_arrival_seconds[node_idx] = current_seconds

    def step(self, actions: Dict[int, Any]) -> StepResult:
        if self.state is None:
            raise RuntimeError("reset() must be called before step()")
        if self.rng is None:
            raise RuntimeError("rng is not initialized")

        time_band = str(self.current_scenario.get("time_band", "offpeak")).strip().lower()

        demand = self.rng.poisson(lam=self._base_demand_rate(time_band)).astype(float)
        self.state.queues += demand
        self.state.demand_generated_total += int(demand.sum())

        event_state_ts = self._state_ts()
        current_seconds = float(self.state.step_index * self.control_step_minutes * 60)

        step_energy = 0.0
        step_served = 0

        for agent_id in range(self.num_agents_val):
            action_id = self._parse_action(actions.get(agent_id, ACTION_DISPATCH))
            action_name = ACTION_NAMES[action_id]

            position_before = int(self.state.bus_positions[agent_id])
            position_after = position_before
            distance_m = 0.0
            hold_seconds = 0.0
            acceleration_event_count = 0
            served = 0
            intervention_applied = action_id != ACTION_DISPATCH

            if action_id == ACTION_HOLD:
                position_after = position_before
                hold_seconds = float(self.control_step_minutes * 60)
                energy = self.k_idle_kwh_per_sec * hold_seconds
                self.state.intervention_count += 1

            elif action_id == ACTION_SKIP:
                skipped = (position_before + 1) % self.num_nodes_val
                position_after = (position_before + 2) % self.num_nodes_val
                distance_m = self._segment_distance(position_before) + self._segment_distance(skipped)
                served = self._serve_at_node(position_after)
                acceleration_event_count = 1
                energy = (
                    self.k_dist_kwh_per_m * distance_m * 1.10
                    + self.k_acc_kwh_per_event * 1.5
                )
                self._record_arrival_headway(position_after, current_seconds)
                self.state.intervention_count += 1

            else:
                position_after = (position_before + 1) % self.num_nodes_val
                distance_m = self._segment_distance(position_before)
                served = self._serve_at_node(position_after)
                acceleration_event_count = 1
                energy = (
                    self.k_dist_kwh_per_m * distance_m
                    + self.k_acc_kwh_per_event
                )
                self._record_arrival_headway(position_after, current_seconds)

            self.state.bus_positions[agent_id] = position_after
            self.state.energy_proxy_total += float(energy)
            self.state.decision_step_count += 1

            step_energy += float(energy)
            step_served += int(served)

            self.raw_events.append({
                "condition_id": self.condition_id,
                "seed": int(self.seed),
                "window_id": self.current_scenario.get("window_id", "toy_window_001"),
                "state_ts": event_state_ts,
                "time_band": time_band,
                "agent_id": int(agent_id),
                "action": int(action_id),
                "action_name": action_name,
                "position_before": int(position_before),
                "position_after": int(position_after),
                "waiting_passenger_cnt": float(self.state.queues.sum()),
                "passenger_served_count": int(served),
                "distance_m": float(distance_m),
                "hold_seconds": float(hold_seconds),
                "acceleration_event_count": int(acceleration_event_count),
                "energy_proxy_total": float(energy),
                "intervention_applied": bool(intervention_applied),
                "terminated": False,
                "truncated": False,
                "source_mode": self.source_mode,
                "causal_comparison_allowed": True,
            })

        queue_total = float(self.state.queues.sum())
        self.state.queue_person_seconds += queue_total * float(self.control_step_minutes * 60)
        self.state.wait_passenger_count += max(1, int(queue_total))
        self.state.step_index += 1

        terminated = False
        truncated = self.state.step_index >= self.max_steps

        avg_queue = float(self.state.queues.mean())
        bunching_penalty = float(self.state.bunching_event_count)

        rewards = {
            agent_id: float(-avg_queue - 0.1 * bunching_penalty)
            for agent_id in range(self.num_agents_val)
        }

        obs = self._build_obs()

        info = {
            "adapter_version": self.adapter_version,
            "source_mode": self.source_mode,
            "causal_comparison_allowed": True,
            "control_step_minutes": self.control_step_minutes,
            "evaluation_horizon_minutes": self.evaluation_horizon_minutes,
            "agent_terminated_mask": [False] * self.num_agents_val,
            "agent_truncated_mask": [bool(truncated)] * self.num_agents_val,
            "step_energy_proxy_total": float(step_energy),
            "step_passenger_served_count": int(step_served),
            "qwen_trigger_rate": 0.0,
            "note": "toy causal simulator: actions change queues, positions, headways, and KPI rollups",
        }

        reward_total, reward_info, _reward_metrics = self._compute_step_reward()
        rewards = {
            agent_id: reward_total
            for agent_id in range(self.num_agents_val)
        }
        info.update(reward_info)

        return StepResult(
            obs=obs,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_graph_skeleton(self) -> GraphSkeleton:
        return GraphSkeleton(
            num_nodes=self.num_nodes_val,
            num_edges=int(self.edge_index.shape[1]),
            node_features_dim=16,
            edge_features_dim=4,
            edge_index=self.edge_index.copy(),
            edge_attr=self.edge_attr.copy(),
        )

    def compute_kpis(
        self,
        trajectory: Optional[List[StepResult]] = None,
    ) -> Dict[str, Optional[float]]:
        if self.state is None:
            raise RuntimeError("reset() must be called before compute_kpis()")

        headways = np.array(self.state.headway_samples, dtype=float)

        if len(headways) >= 2 and float(headways.mean()) > 0:
            cv_headway = float(headways.std(ddof=1) / headways.mean())
        else:
            cv_headway = None

        avg_wait_seconds = float(
            self.state.queue_person_seconds / max(1, self.state.wait_passenger_count)
        )

        headway_event_count = max(1, int(len(self.state.headway_samples)))
        bunching_rate = float(self.state.bunching_event_count / headway_event_count)

        sched = max(1, int(self.state.schedulable_arrival_count))
        on_time_rate = float(self.state.ontime_event_count / sched)

        intervention_rate = float(
            self.state.intervention_count / max(1, self.state.decision_step_count)
        )

        return {
            "cv_headway": cv_headway,
            "avg_wait_seconds": avg_wait_seconds,
            "bunching_rate": bunching_rate,
            "on_time_rate": on_time_rate,
            "intervention_rate": intervention_rate,
            "energy_proxy": float(self.state.energy_proxy_total),
        }


    def _condition_active_bus_ratio(self) -> float:
        condition = str(self.condition_id).upper()
        if condition == "A90":
            return 0.9
        if condition == "A80":
            return 0.8
        if condition == "A70":
            return 0.7
        return 1.0

    def _build_reward_metrics(self) -> Dict[str, float]:
        if self.state is None:
            raise RuntimeError("reset() must be called before _build_reward_metrics()")

        kpis = self.compute_kpis()

        demand = float(max(0, int(self.state.demand_generated_total)))
        served = float(max(0, int(self.state.passenger_served_total)))

        service_rate = served / max(demand, 1.0)
        service_rate = float(min(1.0, max(0.0, service_rate)))

        avg_wait = kpis.get("avg_wait_seconds")
        avg_wait_seconds = float(avg_wait) if avg_wait is not None else 0.0
        avg_wait_seconds = float(max(0.0, avg_wait_seconds))

        queue_pressure = float(self.state.queues.sum()) / max(1.0, float(self.num_nodes_val))
        passenger_wait_p95_seconds = float(max(avg_wait_seconds * 1.65, avg_wait_seconds + queue_pressure))

        energy_proxy = float(max(0.0, float(kpis.get("energy_proxy") or self.state.energy_proxy_total)))
        energy_proxy_per_passenger = float(energy_proxy / max(served, 1.0))

        baseline_bus_count = float(self.config.get("baseline_bus_count", 8.0))
        if baseline_bus_count <= 0:
            baseline_bus_count = 8.0
        active_bus_count = baseline_bus_count * self._condition_active_bus_ratio()
        fleet_reduction_ratio = 1.0 - (active_bus_count / max(baseline_bus_count, 1.0))
        fleet_reduction_ratio = float(min(1.0, max(0.0, fleet_reduction_ratio)))

        bunching_rate = float(kpis.get("bunching_rate") or 0.0)
        on_time_rate = float(kpis.get("on_time_rate") or 0.0)
        intervention_rate = float(kpis.get("intervention_rate") or 0.0)

        return {
            "cv_headway": float(kpis["cv_headway"]) if kpis.get("cv_headway") is not None else 0.0,
            "avg_wait_seconds": avg_wait_seconds,
            "bunching_rate": float(min(1.0, max(0.0, bunching_rate))),
            "on_time_rate": float(min(1.0, max(0.0, on_time_rate))),
            "intervention_rate": float(min(1.0, max(0.0, intervention_rate))),
            "energy_proxy": energy_proxy,
            "passenger_demand_generated": demand,
            "passenger_served_count": served,
            "passenger_service_rate": service_rate,
            "passenger_wait_p95_seconds": passenger_wait_p95_seconds,
            "energy_proxy_per_passenger": energy_proxy_per_passenger,
            "fleet_reduction_ratio": fleet_reduction_ratio,
            "baseline_bus_count": float(baseline_bus_count),
            "active_bus_count": float(active_bus_count),
            "qwen_trigger_rate": 0.0,
        }

    def _compute_step_reward(self) -> tuple[float, Dict[str, Any], Dict[str, float]]:
        reward_metrics = self._build_reward_metrics()
        reward_result = compute_total_reward(reward_metrics)

        reward_total = float(reward_result["reward_total"])
        reward_info: Dict[str, Any] = {
            "reward_version": reward_result.get("reward_version", "mappo_reward_v1"),
            "reward_claim_boundary": reward_result.get(
                "reward_claim_boundary",
                "toy_causal_training_reward_contract_not_paper_performance_claim",
            ),
            "reward_total": reward_total,
            "reward_components": reward_result.get("reward_components", {}),
            "reward_debug": reward_result.get("reward_debug", {}),
            "reward_metrics": reward_metrics,
        }
        return reward_total, reward_info, reward_metrics

    def build_window_rollup_row(self) -> Dict[str, Any]:
        if self.state is None:
            raise RuntimeError("reset() must be called before build_window_rollup_row()")

        headways = np.array(self.state.headway_samples, dtype=float)

        if len(headways) > 0:
            headway_mean = float(headways.mean())
            headway_std = float(headways.std(ddof=1)) if len(headways) >= 2 else 0.0
        else:
            headway_mean = float(self.target_headway_seconds)
            headway_std = 0.0

        return {
            "condition_id": self.condition_id,
            "seed": int(self.seed),
            "window_id": self.current_scenario.get("window_id", "toy_window_001"),
            "state_ts": self._state_ts(),
            "service_date": self._service_date(),
            "time_band": str(self.current_scenario.get("time_band", "offpeak")).strip().lower(),
            "evaluation_horizon_minutes": int(self.evaluation_horizon_minutes),
            "qwen_trigger_rate": 0.0,
            "effective_replay_step_minutes": float(self.control_step_minutes),
            "headway_mean_seconds": float(max(1.0, headway_mean)),
            "headway_std_seconds": float(max(0.0, headway_std)),
            "headway_sample_count": int(len(headways)),
            "bunching_event_count": int(self.state.bunching_event_count),
            "headway_event_count": int(max(1, len(headways))),
            "wait_total_passenger_seconds": float(self.state.queue_person_seconds),
            "wait_passenger_count": int(max(1, self.state.wait_passenger_count)),
            "ontime_event_count": int(self.state.ontime_event_count),
            "schedulable_arrival_count": int(max(1, self.state.schedulable_arrival_count)),
            "intervention_count": int(self.state.intervention_count),
            "decision_step_count": int(max(1, self.state.decision_step_count)),
            "energy_proxy_total": float(self.state.energy_proxy_total),
            "source_mode": self.source_mode,
            "control_step_minutes": int(self.control_step_minutes),
            "causal_comparison_allowed": True,
            "simulator_adapter_version": self.adapter_version,
            "graph_scope": self.graph_scope,
            "num_agents": int(self.num_agents_val),
            "num_nodes": int(self.num_nodes_val),
            "passenger_demand_generated": int(self.state.demand_generated_total),
            "passenger_served_count": int(self.state.passenger_served_total),
        }

    def get_raw_events(self) -> List[Dict[str, Any]]:
        return list(self.raw_events)

    def close(self) -> None:
        self.rng = None
        self.state = None
        self.current_scenario = {}
        self.raw_events = []
