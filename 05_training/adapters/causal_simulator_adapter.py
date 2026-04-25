from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from simulator_adapter_interface import (
    GraphSkeleton,
    ObsDict,
    SimulatorAdapterInterface,
    StepResult,
)


class CausalSimulatorAdapter(SimulatorAdapterInterface):
    """
    Phase-2 causal simulator-backed adapter skeleton.

    This adapter is intentionally minimal at this step. It is NOT yet the full
    city-scale simulator. The purpose of this file is to establish the Phase-2
    interface boundary:

    - reset() creates an initial simulated state.
    - step(actions) advances internal simulated state.
    - actions can affect waiting passengers, intervention counters, and rewards.
    - official KPI aggregation must still be performed by canonical_kpi_aggregator.py.

    Important distinction:
    - HistoricalReplayAdapter is non-causal: actions do not change future state.
    - CausalSimulatorAdapter is causal: actions are allowed to change future state.
    """

    SHARED_KPIS = [
        "cv_headway",
        "avg_wait_seconds",
        "bunching_rate",
        "on_time_rate",
        "intervention_rate",
        "energy_proxy",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        self.condition_id = str(self.config.get("condition_id", "B0R"))
        self.source_mode = str(
            self.config.get(
                "source_mode",
                "causal_sim_current_ops_reconstructed_v1",
            )
        )

        self.num_agents_val = int(self.config.get("num_agents", 10))
        self.num_nodes_val = int(self.config.get("num_nodes", 4116))
        self.num_edges_val = int(self.config.get("num_edges", 5484))

        self.actor_dim = int(self.config.get("actor_dim", 32))
        self.critic_dim = int(self.config.get("critic_dim", 128))
        self.edge_features_dim = int(self.config.get("edge_features_dim", 4))

        self.evaluation_horizon_minutes = int(
            self.config.get("evaluation_horizon_minutes", 30)
        )
        self.sim_step_seconds = int(self.config.get("sim_step_seconds", 60))
        self.decision_interval_seconds = int(
            self.config.get("decision_interval_seconds", 300)
        )
        self.max_steps = max(
            1,
            int((self.evaluation_horizon_minutes * 60) / self.sim_step_seconds),
        )

        self.bus_capacity = int(self.config.get("bus_capacity", 70))
        self.default_target_headway_seconds = float(
            self.config.get("target_headway_seconds", 600.0)
        )

        self.allow_hold = bool(self.config.get("allow_hold", False))
        self.allow_skip = bool(self.config.get("allow_skip", False))
        self.allow_dispatch = bool(self.config.get("allow_dispatch", False))

        self.hold_buckets_seconds = list(
            self.config.get("hold_buckets_seconds", [0, 30, 60, 120])
        )
        self.dispatch_top_k = int(self.config.get("dispatch_top_k", 0))

        self.current_scenario: Dict[str, Any] = {}
        self.seed: Optional[int] = None
        self.rng = np.random.default_rng(0)

        self._closed = False

        self.feature_weight_profile_path = str(
            self.config.get("feature_weight_profile_path", "")
        )
        self.feature_weight_profile = self._load_feature_weight_profile(
            self.feature_weight_profile_path
        )

        self._reset_internal_state()

    @property
    def num_agents(self) -> int:
        return self.num_agents_val

    @property
    def observation_space(self) -> Any:
        return {
            "actor_obs": (self.num_agents_val, self.actor_dim),
            "critic_obs": (1, self.critic_dim),
            "active_bus_mask": (self.num_agents_val,),
            "action_mask": (self.num_agents_val, 4),
        }

    @property
    def action_space(self) -> Any:
        return {
            "type": "hybrid_discrete",
            "action_type_dim": 4,
            "action_types": {
                0: "proceed",
                1: "hold",
                2: "skip",
                3: "dispatch",
            },
            "hold_buckets_seconds": self.hold_buckets_seconds,
            "dispatch_top_k": self.dispatch_top_k,
        }


    def _default_feature_weight_profile(self) -> Dict[str, Any]:
        """
        Built-in fallback profile. The normal Phase-2 path should load
        05_training/configs/feature_weight_profile_v1.yaml.
        """
        return {
            "artifact_version": "feature_weight_profile_v1_builtin_default",
            "feature_weights": {
                "waiting_passenger_cnt": {"weight": 5},
                "boardings_recent": {"weight": 3},
                "alightings_recent": {"weight": 2},
                "is_peak": {"weight": 4},
                "hour_of_day": {"weight": 4},
                "day_of_week": {"weight": 4},
            },
            "normalization": {
                "caps": {
                    "waiting_passenger_cnt": 50,
                    "boardings_recent": 30,
                    "alightings_recent": 30,
                }
            },
            "pressure_score_v1": {
                "final_pressure_score": {
                    "time_context_max_boost_ratio": 0.2,
                }
            },
        }

    def _load_feature_weight_profile(self, path: str) -> Dict[str, Any]:
        """
        Load the adopted Phase-2 feature weight profile.

        If no path is supplied, use a built-in default with the same weights.
        If a path is supplied but missing/invalid, fail loudly because the
        experiment contract must be version-pinned.
        """
        if not path:
            return self._default_feature_weight_profile()

        profile_path = Path(path)
        if not profile_path.is_absolute():
            profile_path = Path.cwd() / profile_path

        if not profile_path.exists():
            raise FileNotFoundError(f"feature weight profile not found: {profile_path}")

        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(f"PyYAML is required to load feature weight profile: {exc}")

        with profile_path.open("r", encoding="utf-8-sig") as f:
            profile = yaml.safe_load(f)

        self._validate_feature_weight_profile(profile)
        return profile

    def _validate_feature_weight_profile(self, profile: Dict[str, Any]) -> None:
        required = {
            "waiting_passenger_cnt": 5,
            "is_peak": 4,
            "hour_of_day": 4,
            "day_of_week": 4,
            "boardings_recent": 3,
            "alightings_recent": 2,
        }

        weights = profile.get("feature_weights", {})
        for key, expected in required.items():
            if key not in weights:
                raise RuntimeError(f"feature weight missing: {key}")
            got = int(weights[key].get("weight"))
            if got != expected:
                raise RuntimeError(
                    f"feature weight mismatch for {key}: expected={expected}, got={got}"
                )

    def _feature_weight(self, name: str) -> float:
        return float(
            self.feature_weight_profile.get("feature_weights", {})
            .get(name, {})
            .get("weight", 0.0)
        )

    def _feature_cap(self, name: str, default: float) -> float:
        return float(
            self.feature_weight_profile.get("normalization", {})
            .get("caps", {})
            .get(name, default)
        )

    def _norm_log1p(self, value: float, cap: float) -> float:
        value = max(float(value), 0.0)
        cap = max(float(cap), 1.0)
        out = float(np.log1p(value) / np.log1p(cap))
        return float(np.clip(out, 0.0, 1.0))

    def pressure_score_v1(
        self,
        *,
        waiting_passenger_cnt: float,
        boardings_recent: float,
        alightings_recent: float,
        is_peak: float,
        hour_pattern_score: float,
        day_pattern_score: float,
    ) -> Dict[str, float]:
        """
        Adopted Phase-2 pressure score.

        demand_pressure =
          (5*waiting_norm + 3*boarding_norm + 2*alighting_norm) / 10

        time_context =
          (4*is_peak + 4*hour_pattern_score + 4*day_pattern_score) / 12

        pressure_score =
          100 * demand_pressure * (1 + 0.2 * time_context)
        """
        waiting_norm = self._norm_log1p(
            waiting_passenger_cnt,
            self._feature_cap("waiting_passenger_cnt", 50.0),
        )
        boarding_norm = self._norm_log1p(
            boardings_recent,
            self._feature_cap("boardings_recent", 30.0),
        )
        alighting_norm = self._norm_log1p(
            alightings_recent,
            self._feature_cap("alightings_recent", 30.0),
        )

        w_wait = self._feature_weight("waiting_passenger_cnt")
        w_board = self._feature_weight("boardings_recent")
        w_alight = self._feature_weight("alightings_recent")

        w_peak = self._feature_weight("is_peak")
        w_hour = self._feature_weight("hour_of_day")
        w_day = self._feature_weight("day_of_week")

        demand_weight_sum = max(w_wait + w_board + w_alight, 1.0)
        time_weight_sum = max(w_peak + w_hour + w_day, 1.0)

        demand_pressure = (
            w_wait * waiting_norm
            + w_board * boarding_norm
            + w_alight * alighting_norm
        ) / demand_weight_sum

        time_context = (
            w_peak * float(np.clip(is_peak, 0.0, 1.0))
            + w_hour * float(np.clip(hour_pattern_score, 0.0, 1.0))
            + w_day * float(np.clip(day_pattern_score, 0.0, 1.0))
        ) / time_weight_sum

        boost = float(
            self.feature_weight_profile.get("pressure_score_v1", {})
            .get("final_pressure_score", {})
            .get("time_context_max_boost_ratio", 0.2)
        )

        pressure_score = 100.0 * demand_pressure * (1.0 + boost * time_context)

        return {
            "waiting_norm": float(waiting_norm),
            "boarding_norm": float(boarding_norm),
            "alighting_norm": float(alighting_norm),
            "demand_pressure": float(demand_pressure),
            "time_context": float(time_context),
            "pressure_score": float(pressure_score),
        }

    def _scenario_time_context(self) -> Dict[str, float]:
        """
        Skeleton time-context estimator.

        Later this should be replaced with empirical hour/day pattern scores
        derived from full-year demand distributions.
        """
        tb = self._time_band()
        is_peak = 1.0 if tb == "peak" else 0.0

        if tb == "peak":
            hour_pattern_score = 0.9
        elif tb == "night":
            hour_pattern_score = 0.35
        else:
            hour_pattern_score = 0.55

        # Conservative default until explicit weekday/weekend pattern table exists.
        day_pattern_score = 0.70

        return {
            "is_peak": is_peak,
            "hour_pattern_score": float(hour_pattern_score),
            "day_pattern_score": float(day_pattern_score),
        }

    def _pressure_score_for_agent(self, idx: int) -> float:
        """
        Build a local pressure score from current skeleton state.

        This uses current waiting pressure plus simple recent boarding/alighting
        proxies. The full version should use true per-stop rolling features.
        """
        dyn = self._dynamics_profile() if hasattr(self, "_dynamics_profile") else {}
        waiting = float(self.waiting_by_agent_area[idx])

        boardings_recent = min(
            waiting,
            float(dyn.get("boarding_limit", 5.0)),
        )
        alightings_recent = max(float(self.onboard_by_agent[idx]) * 0.10, 0.0)

        ctx = self._scenario_time_context()

        return float(
            self.pressure_score_v1(
                waiting_passenger_cnt=waiting,
                boardings_recent=boardings_recent,
                alightings_recent=alightings_recent,
                is_peak=ctx["is_peak"],
                hour_pattern_score=ctx["hour_pattern_score"],
                day_pattern_score=ctx["day_pattern_score"],
            )["pressure_score"]
        )

    def _reset_internal_state(self) -> None:
        self.step_idx = 0
        self.sim_elapsed_seconds = 0

        self.agent_ids = list(range(self.num_agents_val))
        self.active_bus_mask = np.ones((self.num_agents_val,), dtype=bool)

        # Minimal state arrays. These are placeholders for Phase-2 v0.1 smoke.
        self.waiting_by_agent_area = np.full(
            (self.num_agents_val,),
            10.0,
            dtype=np.float32,
        )
        self.onboard_by_agent = np.zeros((self.num_agents_val,), dtype=np.float32)
        self.remaining_hold_seconds = np.zeros((self.num_agents_val,), dtype=np.float32)
        self.headway_samples: List[float] = []

        self.bunching_event_count = 0
        self.headway_event_count = 0
        self.ontime_event_count = 0
        self.schedulable_arrival_count = 0

        self.intervention_count = 0
        self.decision_step_count = 0
        self.invalid_action_count = 0

        self.wait_total_passenger_seconds = 0.0
        self.wait_passenger_count = 0
        self.energy_proxy_total = 0.0

        self.last_rewards: Dict[int, float] = {
            agent_id: 0.0 for agent_id in self.agent_ids
        }

    def _time_band(self) -> str:
        return str(self.current_scenario.get("time_band", "offpeak")).strip().lower()

    def _state_ts(self) -> str:
        return str(self.current_scenario.get("state_ts", "2024-01-01T08:00:00Z"))

    def _window_id(self) -> Any:
        return self.current_scenario.get("window_id", "smoke_window")

    def _target_headway_for_time_band(self) -> float:
        target = self.config.get("target_headway_by_time_band")
        if isinstance(target, dict):
            return float(target.get(self._time_band(), self.default_target_headway_seconds))
        return self.default_target_headway_seconds

    def _build_action_mask(self) -> np.ndarray:
        mask = np.zeros((self.num_agents_val, 4), dtype=bool)

        # proceed is always valid.
        mask[:, 0] = True

        # Other controls are enabled by config. B0R keeps them false.
        mask[:, 1] = self.allow_hold
        mask[:, 2] = self.allow_skip
        mask[:, 3] = self.allow_dispatch

        return mask

    def _build_obs(self) -> ObsDict:
        actor_obs = np.zeros((self.num_agents_val, self.actor_dim), dtype=np.float32)
        critic_obs = np.zeros((1, self.critic_dim), dtype=np.float32)

        target_headway = self._target_headway_for_time_band()

        for i, agent_id in enumerate(self.agent_ids):
            waiting = float(self.waiting_by_agent_area[i])
            onboard = float(self.onboard_by_agent[i])
            hold_remaining = float(self.remaining_hold_seconds[i])

            actor_obs[i, 0] = waiting / 100.0
            actor_obs[i, 1] = onboard / max(float(self.bus_capacity), 1.0)
            actor_obs[i, 2] = hold_remaining / 120.0
            actor_obs[i, 3] = target_headway / 1200.0
            actor_obs[i, 4] = float(self.step_idx) / max(float(self.max_steps), 1.0)
            actor_obs[i, 5] = self._pressure_score_for_agent(i) / 120.0

        pressure_scores = np.array(
            [self._pressure_score_for_agent(i) for i in range(self.num_agents_val)],
            dtype=np.float32,
        )

        critic_obs[0, 0] = float(np.sum(self.waiting_by_agent_area)) / 1000.0
        critic_obs[0, 1] = float(np.mean(self.waiting_by_agent_area)) / 100.0
        critic_obs[0, 2] = float(np.max(self.waiting_by_agent_area)) / 100.0
        critic_obs[0, 3] = float(np.sum(self.onboard_by_agent)) / max(
            float(self.num_agents_val * self.bus_capacity),
            1.0,
        )
        critic_obs[0, 4] = float(self.step_idx) / max(float(self.max_steps), 1.0)
        critic_obs[0, 5] = float(self.intervention_count)
        critic_obs[0, 6] = float(self.invalid_action_count)
        critic_obs[0, 7] = float(np.mean(pressure_scores)) / 120.0
        critic_obs[0, 8] = float(np.max(pressure_scores)) / 120.0

        edge_index = np.zeros((2, self.num_edges_val), dtype=np.int64)
        edge_attr = np.zeros((self.num_edges_val, self.edge_features_dim), dtype=np.float32)

        return {
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "active_bus_mask": self.active_bus_mask.copy(),
            "action_mask": self._build_action_mask(),
            "time_band": self._time_band(),
            "state_ts": self._state_ts(),
            "agent_ids": list(self.agent_ids),
        }

    def reset(
        self,
        seed: Optional[int] = None,
        scenario_config: Optional[dict] = None,
    ) -> ObsDict:
        self.seed = seed
        self.rng = np.random.default_rng(0 if seed is None else int(seed))
        self.current_scenario = scenario_config or {}

        self._reset_internal_state()

        # A deterministic, tiny variation by seed/window for smoke testing.
        base_wait = float(self.current_scenario.get("initial_waiting_passengers", 10.0))
        jitter = self.rng.integers(0, 5, size=(self.num_agents_val,)).astype(np.float32)
        self.waiting_by_agent_area = np.full(
            (self.num_agents_val,),
            base_wait,
            dtype=np.float32,
        ) + jitter

        return self._build_obs()

    def _parse_action_type(self, action: Any) -> int:
        if isinstance(action, dict):
            return int(action.get("action_type", 0))
        if isinstance(action, (list, tuple)) and len(action) > 0:
            return int(action[0])
        return int(action)

    def _parse_hold_seconds(self, action: Any) -> int:
        if isinstance(action, dict):
            if "hold_seconds" in action:
                return int(action.get("hold_seconds", 0))
            bucket = int(action.get("hold_bucket", 0))
        elif isinstance(action, (list, tuple)) and len(action) > 1:
            bucket = int(action[1])
        else:
            bucket = 0

        bucket = max(0, min(bucket, len(self.hold_buckets_seconds) - 1))
        return int(self.hold_buckets_seconds[bucket])

    def _is_action_valid(self, action_type: int) -> bool:
        if action_type == 0:
            return True
        if action_type == 1:
            return self.allow_hold
        if action_type == 2:
            return self.allow_skip
        if action_type == 3:
            return self.allow_dispatch
        return False


    def _dynamics_profile(self) -> Dict[str, float]:
        """
        Minimal Phase-2 v0.2 dynamics profile.

        B0R:
          reconstructed current fixed-route operation.
          It follows a stable target-headway service pattern.

        B1:
          no-op causal floor.
          It does not try to preserve the target headway, so headway noise,
          waiting pressure, and on-time degradation are larger.

        This is still a skeleton-stage causal model. It is intended to make
        B0R and B1 distinguishable before the full city-scale simulator is built.
        """
        cid = str(self.condition_id).upper()

        if cid == "B0R":
            return {
                "arrival_lambda": 1.2,
                "boarding_limit": 6.0,
                "headway_bias": 0.0,
                "headway_noise_std": 25.0,
                "base_energy_per_agent_step": 0.8,
            }

        if cid == "B1":
            return {
                "arrival_lambda": 2.2,
                "boarding_limit": 2.5,
                "headway_bias": 90.0,
                "headway_noise_std": 220.0,
                "base_energy_per_agent_step": 0.5,
            }

        if cid == "B2":
            return {
                "arrival_lambda": 1.7,
                "boarding_limit": 5.0,
                "headway_bias": 20.0,
                "headway_noise_std": 90.0,
                "base_energy_per_agent_step": 0.7,
            }

        if cid == "A":
            return {
                "arrival_lambda": 1.5,
                "boarding_limit": 5.5,
                "headway_bias": 10.0,
                "headway_noise_std": 70.0,
                "base_energy_per_agent_step": 0.65,
            }

        return {
            "arrival_lambda": 1.5,
            "boarding_limit": 5.0,
            "headway_bias": 0.0,
            "headway_noise_std": 60.0,
            "base_energy_per_agent_step": 0.7,
        }

    def step(self, actions: Dict[int, Any]) -> StepResult:
        if self._closed:
            raise RuntimeError("CausalSimulatorAdapter is closed")

        if actions is None:
            actions = {}

        rewards: Dict[int, float] = {}
        target_headway = self._target_headway_for_time_band()
        profile = self._dynamics_profile()

        # Count one decision opportunity per active agent per step.
        self.decision_step_count += int(np.sum(self.active_bus_mask))

        step_wait_penalty = 0.0
        step_intervention_penalty = 0.0
        step_invalid_penalty = 0.0
        step_energy_penalty = 0.0

        for i, agent_id in enumerate(self.agent_ids):
            action = actions.get(agent_id, 0)
            action_type = self._parse_action_type(action)

            if not self._is_action_valid(action_type):
                self.invalid_action_count += 1
                step_invalid_penalty += 1.0
                action_type = 0

            exogenous_arrivals = float(self.rng.poisson(profile["arrival_lambda"]))
            self.waiting_by_agent_area[i] += exogenous_arrivals

            boarded = 0.0
            headway = float(max(1.0, target_headway + profile["headway_bias"] + self.rng.normal(0.0, profile["headway_noise_std"])))

            if action_type == 0:
                # proceed: board a small number of passengers.
                available_capacity = max(float(self.bus_capacity) - float(self.onboard_by_agent[i]), 0.0)
                boarded = min(float(self.waiting_by_agent_area[i]), available_capacity, float(profile["boarding_limit"]))
                self.waiting_by_agent_area[i] -= boarded
                self.onboard_by_agent[i] += boarded

            elif action_type == 1:
                # hold: keep bus waiting, passenger queue continues to grow.
                hold_seconds = self._parse_hold_seconds(action)
                self.remaining_hold_seconds[i] += float(hold_seconds)
                self.intervention_count += 1
                step_intervention_penalty += 1.0
                step_energy_penalty += hold_seconds * 0.01
                headway += hold_seconds

            elif action_type == 2:
                # skip: no boarding at this local area.
                self.intervention_count += 1
                step_intervention_penalty += 1.0
                headway = max(1.0, headway - 30.0)

            elif action_type == 3:
                # dispatch: simplified effect, reduce local waiting pressure.
                self.intervention_count += 1
                step_intervention_penalty += 1.0
                reduced = min(float(self.waiting_by_agent_area[i]), 3.0)
                self.waiting_by_agent_area[i] -= reduced
                step_energy_penalty += 1.0

            self.remaining_hold_seconds[i] = max(
                0.0,
                float(self.remaining_hold_seconds[i]) - float(self.sim_step_seconds),
            )

            self.headway_samples.append(headway)
            self.headway_event_count += 1
            self.schedulable_arrival_count += 1

            fixed_bunching_threshold = float(
                self.config.get("bunching_threshold_seconds", 180.0)
            )
            bunching_threshold_ratio = float(
                self.config.get("bunching_threshold_ratio", 0.75)
            )
            effective_bunching_threshold = max(
                fixed_bunching_threshold,
                float(target_headway) * bunching_threshold_ratio,
            )

            if headway < effective_bunching_threshold:
                self.bunching_event_count += 1

            if abs(headway - target_headway) <= float(
                self.config.get("on_time_tolerance_seconds", 120.0)
            ):
                self.ontime_event_count += 1

            current_waiting = float(self.waiting_by_agent_area[i])
            self.wait_total_passenger_seconds += current_waiting * float(self.sim_step_seconds)
            # Denominator for avg_wait_seconds should represent served/new passengers,
            # not the same queued passengers counted repeatedly every simulator tick.
            self.wait_passenger_count += int(max(round(exogenous_arrivals + boarded), 1))
            step_wait_penalty += current_waiting

            self.energy_proxy_total += step_energy_penalty + float(profile["base_energy_per_agent_step"]) + float(profile["base_energy_per_agent_step"])

            reward = (
                -0.01 * current_waiting
                -0.05 * step_intervention_penalty
                -0.20 * step_invalid_penalty
                -0.01 * step_energy_penalty
            )
            rewards[agent_id] = float(reward)

        self.step_idx += 1
        self.sim_elapsed_seconds += self.sim_step_seconds

        terminated = False
        truncated = self.step_idx >= self.max_steps

        obs = self._build_obs()

        info = {
            "condition_id": self.condition_id,
            "source_mode": self.source_mode,
            "window_id": self._window_id(),
            "time_band": self._time_band(),
            "state_ts": self._state_ts(),
            "sim_step_idx": self.step_idx,
            "sim_elapsed_seconds": self.sim_elapsed_seconds,
            "evaluation_horizon_minutes": self.evaluation_horizon_minutes,
            "sim_step_seconds": self.sim_step_seconds,
            "decision_interval_seconds": self.decision_interval_seconds,
            "causal_comparison_allowed": True,
            "intervention_count": int(self.intervention_count),
            "decision_step_count": int(self.decision_step_count),
            "invalid_action_count": int(self.invalid_action_count),
            "agent_terminated_mask": [False] * self.num_agents_val,
            "agent_truncated_mask": [bool(truncated)] * self.num_agents_val,
        }

        self.last_rewards = rewards

        return StepResult(
            obs=obs,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def get_graph_skeleton(self) -> GraphSkeleton:
        edge_index = np.zeros((2, self.num_edges_val), dtype=np.int64)
        edge_attr = np.zeros((self.num_edges_val, self.edge_features_dim), dtype=np.float32)

        return GraphSkeleton(
            num_nodes=self.num_nodes_val,
            num_edges=self.num_edges_val,
            node_features_dim=self.actor_dim,
            edge_features_dim=self.edge_features_dim,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    def compute_kpis(
        self,
        trajectory: Optional[List[StepResult]] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Convenience KPI calculation only.

        Official comparison must use window_rollup.parquet and
        canonical_kpi_aggregator.py.
        """
        headways = np.array(self.headway_samples, dtype=np.float32)

        if headways.size >= 2 and float(np.mean(headways)) > 0:
            cv_headway = float(np.std(headways, ddof=1) / np.mean(headways))
        else:
            cv_headway = None

        if self.wait_passenger_count > 0:
            avg_wait_seconds = float(
                self.wait_total_passenger_seconds / max(float(self.wait_passenger_count), 1.0)
            )
        else:
            avg_wait_seconds = 0.0

        if self.headway_event_count > 0:
            bunching_rate = float(self.bunching_event_count / self.headway_event_count)
        else:
            bunching_rate = None

        if self.schedulable_arrival_count > 0:
            on_time_rate = float(self.ontime_event_count / self.schedulable_arrival_count)
        else:
            on_time_rate = None

        if self.decision_step_count > 0:
            intervention_rate = float(self.intervention_count / self.decision_step_count)
        else:
            intervention_rate = 0.0

        return {
            "cv_headway": cv_headway,
            "avg_wait_seconds": avg_wait_seconds,
            "bunching_rate": bunching_rate,
            "on_time_rate": on_time_rate,
            "intervention_rate": intervention_rate,
            "energy_proxy": float(self.energy_proxy_total),
        }

    def build_window_rollup_row(self, seed: int) -> Dict[str, Any]:
        """
        Build a single official_rollup-compatible row for later runner use.
        The runner will be responsible for writing window_rollup.parquet.
        """
        headways = np.array(self.headway_samples, dtype=np.float32)
        if headways.size:
            headway_mean = float(np.mean(headways))
            headway_std = float(np.std(headways, ddof=1)) if headways.size >= 2 else 0.0
        else:
            headway_mean = self._target_headway_for_time_band()
            headway_std = 0.0

        return {
            "condition_id": self.condition_id,
            "seed": int(seed),
            "window_id": self._window_id(),
            "state_ts": self._state_ts(),
            "service_date": str(self.current_scenario.get("service_date", "")),
            "time_band": self._time_band(),
            "evaluation_horizon_minutes": int(self.evaluation_horizon_minutes),
            "qwen_trigger_rate": 0.0,
            "effective_replay_step_minutes": float(self.sim_step_seconds) / 60.0,
            "headway_mean_seconds": headway_mean,
            "headway_std_seconds": headway_std,
            "headway_sample_count": int(len(self.headway_samples)),
            "bunching_event_count": int(self.bunching_event_count),
            "headway_event_count": int(self.headway_event_count),
            "wait_total_passenger_seconds": float(self.wait_total_passenger_seconds),
            "wait_passenger_count": int(self.wait_passenger_count),
            "ontime_event_count": int(self.ontime_event_count),
            "schedulable_arrival_count": int(self.schedulable_arrival_count),
            "intervention_count": int(self.intervention_count),
            "decision_step_count": int(self.decision_step_count),
            "energy_proxy_total": float(self.energy_proxy_total),
            "source_mode": self.source_mode,
        }

    def close(self) -> None:
        self._closed = True
