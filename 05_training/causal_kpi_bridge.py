#!/usr/bin/env python3
"""PV8 causal KPI measurement bridge (H4M-AE-R2 architecture A).

Implements the frozen bridge only: the authoritative PV8 state and graph feed
the frozen GATv2 encoder, the promoted target-conditioned actor consumes the
exact 131D observation, its legal action drives a causal transition that keeps
passenger, headway, vehicle and energy accounting, and that accounting alone
produces raw events and the window rollup.

Nothing here trains, updates or modifies a checkpoint, and no KPI value may be
derived from action counts, policy probabilities, condition names or constants.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


BRIDGE_ID = "A_PV8_ADAPTER_KPI_STATE_EXTENSION"
BRIDGE_CONTRACT_SHA256 = "73cde1eb4300719e"  # short prefix of the R2 selection contract; full value bound by the runner
ACTOR_OBS_DIM = 131
GATV2_HIDDEN = 128
TARGET_CONTEXT_DIM = 3
NODE_FEATURE_DIM = 12
ACTION_DIM = 3
ACTION_NAMES = {0: "HOLD_CURRENT_POSITION", 1: "SERVE_AND_MOVE_TO_NEXT_STOP", 2: "CONDITIONAL_SKIP_EMPTY_STOP"}
SOURCE_MODE = "causal_pv8_kpi_bridge_v1"

# -- measured passenger wait tail accounting (H4M-AE-R3.1) --------------------
WAIT_TAIL_REPAIR_ID = "R3_1_MEASURED_PASSENGER_WAIT_TAIL_ACCOUNTING"
ARRIVAL_ENTRY_SEMANTICS = "ONE_ENTRY_ONE_SIMULATED_PASSENGER_UNIT_WEIGHT"
CARRY_IN_CASE = "A_NO_INITIAL_WAITING_QUEUE"
P95_SEMANTICS = "LOWER_BOUND_UNDER_RIGHT_CENSORING"
P95_POPULATION = "SERVED_COMPLETED_PLUS_CENSORED_LOWER_BOUND"
PERCENTILE_Q = 95.0
PERCENTILE_METHOD = "linear"
PERCENTILE_LIBRARY = "numpy.percentile"
CENSOR_REFERENCE = "FIXED_EVALUATION_HORIZON_END"


def measured_wait_p95(completed: Sequence[float], censored: Sequence[float]) -> Dict[str, Any]:
    """Frozen unweighted empirical p95 over the canonical censored-inclusive population."""
    labelled = [(float(x), False) for x in completed] + [(float(x), True) for x in censored]
    population = [value for value, _ in labelled]
    payload = {
        "population_definition": P95_POPULATION,
        "semantics": P95_SEMANTICS,
        "censor_reference": CENSOR_REFERENCE,
        "percentile_q": PERCENTILE_Q,
        "percentile_method": PERCENTILE_METHOD,
        "percentile_library": PERCENTILE_LIBRARY,
        "weighting": "UNWEIGHTED_UNIT_MULTIPLICITY",
        "completed_count": len(completed),
        "censored_count": len(censored),
        "population_count": len(population),
        "censored_share": (len(censored) / len(population)) if population else None,
    }
    if not population:
        payload.update({"value_seconds": None, "status": "NOT_APPLICABLE"})
        return payload
    value = float(np.percentile(np.asarray(population, dtype=float), PERCENTILE_Q, method=PERCENTILE_METHOD))
    ordered = sorted(labelled, key=lambda item: item[0])
    rank = (len(ordered) - 1) * (PERCENTILE_Q / 100.0)
    low_index = int(math.floor(rank))
    high_index = int(math.ceil(rank))
    support = [
        {"value_seconds": ordered[low_index][0], "censored": ordered[low_index][1]},
        {"value_seconds": ordered[high_index][0], "censored": ordered[high_index][1]},
    ]
    payload.update(
        {
            "value_seconds": value,
            "status": "MEASURED",
            "median_seconds": float(np.median(np.asarray(population, dtype=float))),
            "max_seconds": float(max(population)),
            "p95_support": support,
            "p95_support_includes_censored_observation": bool(ordered[low_index][1] or ordered[high_index][1]),
            "p95_exact_support_is_censored": bool(ordered[low_index][1]) if low_index == high_index else None,
            "p95_interpolated_between_two_support_points": low_index != high_index,
        }
    )
    return payload


def pooled_wait_p95(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pool raw per-passenger ledgers across windows; never average window p95 values."""
    completed: List[float] = []
    censored: List[float] = []
    for payload in payloads:
        completed.extend(float(x) for x in payload.get("completed_wait_ledger", []))
        censored.extend(float(x) for x in payload.get("censored_wait_ledger", []))
    pooled = measured_wait_p95(completed, censored)
    pooled["aggregation"] = "POOLED_RAW_PASSENGER_LEDGER"
    pooled["window_count"] = len(payloads)
    return pooled
CAUSAL_COMPARISON_ALLOWED = True

# transition constants are physical/geometric parameters of the accounting model,
# never KPI values; KPIs are always computed from accumulated event quantities.
SEGMENT_DISTANCE_M = 420.0
CRUISE_SPEED_MPS = 7.5
DWELL_BASE_SECONDS = 8.0
DWELL_PER_PASSENGER_SECONDS = 2.5
HOLD_STEP_SECONDS = 30.0
BUNCHING_HEADWAY_SECONDS = 120.0
SCHEDULED_HEADWAY_SECONDS = 300.0
ONTIME_TOLERANCE_SECONDS = 120.0


class BridgeContractError(RuntimeError):
    """Fail-closed bridge contract violation."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# promoted policy bridge
# ---------------------------------------------------------------------------
@dataclass
class PolicyProvenance:
    checkpoint_path: str
    checkpoint_sha256: str
    seed: Optional[int]
    actor_input_dim: int
    gatv2_node_input_dim: int
    action_dim: int
    target_head_count: int
    policy_source_mode: str
    placeholder_used: bool = False
    mock_action_used: bool = False
    random_fallback_used: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class PromotedPolicyBridge:
    """Loads an actual promoted checkpoint and produces legal actions.

    Fail-closed on any dimensional mismatch; never pads, projects or truncates
    an observation, and never falls back to a placeholder, mock or random action.
    """

    def __init__(self, checkpoint_path: Path, dl1: Any, dl4: Any, sample_graph: Any, device: torch.device) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise BridgeContractError("CHECKPOINT_NOT_FOUND", str(path))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("gatv2_state_dict", "actor_state_dict", "critic_state_dict"):
            if key not in payload:
                raise BridgeContractError("CHECKPOINT_MISSING_STATE_DICT", key)
        actor_weight = payload["actor_state_dict"]["net.0.weight"]
        gatv2_weight = payload["gatv2_state_dict"]["conv1.lin_l.weight"]
        actor_input_dim = int(actor_weight.shape[1])
        node_input_dim = int(gatv2_weight.shape[1])
        if actor_input_dim != ACTOR_OBS_DIM:
            raise BridgeContractError("ACTOR_INPUT_DIM_MISMATCH", f"expected {ACTOR_OBS_DIM}, checkpoint has {actor_input_dim}")
        if node_input_dim != NODE_FEATURE_DIM:
            raise BridgeContractError("GATV2_NODE_INPUT_DIM_MISMATCH", f"expected {NODE_FEATURE_DIM}, checkpoint has {node_input_dim}")
        head_count = sum(1 for key in payload["actor_state_dict"] if key.startswith("target_heads.") and key.endswith(".weight"))
        if head_count != ACTION_DIM:
            raise BridgeContractError("TARGET_HEAD_COUNT_MISMATCH", f"expected {ACTION_DIM}, checkpoint has {head_count}")

        self.encoder = dl1.GATv2Encoder(sample_graph.x.size(1), GATV2_HIDDEN, sample_graph.edge_attr.size(1)).to(device)
        self.actor = dl1.MAPPOActor(GATV2_HIDDEN, ACTION_DIM, target_context_dim=TARGET_CONTEXT_DIM, target_head_specialization=True).to(device)
        self.critic = dl1.CentralizedCritic(GATV2_HIDDEN).to(device)
        self.encoder.load_state_dict(payload["gatv2_state_dict"], strict=True)
        self.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        self.critic.load_state_dict(payload["critic_state_dict"], strict=True)
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
        self.device = device
        self.dl1 = dl1
        self.dl4 = dl4
        self.provenance = PolicyProvenance(
            checkpoint_path=str(path),
            checkpoint_sha256=sha256_file(path),
            seed=payload.get("seed"),
            actor_input_dim=actor_input_dim,
            gatv2_node_input_dim=node_input_dim,
            action_dim=ACTION_DIM,
            target_head_count=head_count,
            policy_source_mode="actual_promoted_mappo_checkpoint",
        )
        self.last_observation_dim: Optional[int] = None

    @torch.no_grad()
    def act(self, data: Any, agent_indices: Sequence[int], mask_fn: Any) -> Dict[str, Any]:
        """Return legal actions for the given PV8 graph state."""
        device = self.device
        node_embeddings = self.encoder(data)
        index = torch.tensor(list(agent_indices), dtype=torch.long, device=device)
        agent_embeddings = node_embeddings[index]
        if int(agent_embeddings.size(1)) != GATV2_HIDDEN:
            raise BridgeContractError("AGENT_EMBEDDING_DIM_MISMATCH", str(int(agent_embeddings.size(1))))
        target_context = data.x[index][:, -TARGET_CONTEXT_DIM:]
        observation_dim = int(agent_embeddings.size(1) + target_context.size(1))
        if observation_dim != ACTOR_OBS_DIM:
            raise BridgeContractError("OBSERVATION_DIM_MISMATCH", f"built {observation_dim}, contract {ACTOR_OBS_DIM}")
        self.last_observation_dim = observation_dim
        logits = self.actor(agent_embeddings, target_context)
        targets = self.dl1.action_targets_from_y(data.y[index], ACTION_DIM)
        masked_logits, allowed = mask_fn(logits, targets)
        actions = torch.argmax(masked_logits, dim=-1)
        probs = torch.softmax(masked_logits, dim=-1)
        return {
            "actions": [int(value) for value in actions.detach().cpu().tolist()],
            "targets": [int(value) for value in targets.detach().cpu().tolist()],
            "legal_mask": [[bool(flag) for flag in row] for row in allowed.detach().cpu().tolist()],
            "probabilities": [[float(value) for value in row] for row in probs.detach().cpu().tolist()],
            "observation_dim": observation_dim,
            "observation_composition": "concat(gatv2_agent_embedding_128, r3_action_target_one_hot_3)",
        }


# ---------------------------------------------------------------------------
# causal transition state
# ---------------------------------------------------------------------------
@dataclass
class StopState:
    """One entry in arrival_schedule is one simulated passenger (unit weight)."""

    stop_id: int
    arrival_schedule: List[float] = field(default_factory=list)
    next_arrival_index: int = 0
    queue_arrival_ts: List[float] = field(default_factory=list)
    last_vehicle_arrival_ts: Optional[float] = None

    @property
    def waiting_count(self) -> int:
        return len(self.queue_arrival_ts)


@dataclass
class VehicleState:
    agent_id: int
    stop_index: int
    clock_seconds: float = 0.0
    distance_m: float = 0.0
    acceleration_event_count: int = 0
    hold_seconds: float = 0.0
    served_count: int = 0
    active: bool = True


@dataclass
class CausalAccounting:
    """Every canonical KPI input, accumulated from transitions only."""

    boarding_ledger: List[Tuple[float, float]] = field(default_factory=list)
    completed_wait_ledger: List[float] = field(default_factory=list)
    censored_wait_ledger: List[float] = field(default_factory=list)
    wait_total_passenger_seconds: float = 0.0
    wait_passenger_count: int = 0
    passenger_demand_generated: int = 0
    passenger_served_count: int = 0
    headway_samples: List[float] = field(default_factory=list)
    headway_event_count: int = 0
    bunching_event_count: int = 0
    ontime_event_count: int = 0
    schedulable_arrival_count: int = 0
    intervention_count: int = 0
    decision_step_count: int = 0
    distance_m: float = 0.0
    acceleration_event_count: int = 0
    hold_seconds: float = 0.0
    active_vehicle_ids: set = field(default_factory=set)


class PV8CausalKpiAdapter:
    """PV8 causal transition with the KPI accounting the scaffold lacked.

    The observation and graph come from the frozen PV8 pipeline; this class adds
    only the passenger, headway, vehicle and energy state needed to emit
    simulator-derived events.  It never returns an observation of its own, so the
    promoted 131D contract is untouched.
    """

    source_mode = SOURCE_MODE
    causal_comparison_allowed = CAUSAL_COMPARISON_ALLOWED

    def __init__(self, *, window: Mapping[str, Any], num_agents: int, seed: int, demand_fields: Mapping[str, Any], horizon_minutes: int = 30) -> None:
        if not demand_fields:
            raise BridgeContractError("DEMAND_FIELDS_REQUIRED", "authoritative demand fields must be supplied; demand may not be invented")
        self.window = dict(window)
        self.num_agents = int(num_agents)
        self.seed = int(seed)
        self.horizon_seconds = float(horizon_minutes) * 60.0
        self.demand_fields = dict(demand_fields)
        self.stop_count = max(self.num_agents, 8)
        self.state: Optional[Dict[str, Any]] = None
        self.events: List[Dict[str, Any]] = []
        self.accounting = CausalAccounting()
        self.reset()

    # -- demand is seeded and action independent -----------------------------
    def _arrival_schedule(self, stop_id: int) -> List[float]:
        intensity = float(self.demand_fields.get("historical_boarding_intensity", 0.0))
        score = float(self.demand_fields.get("historical_demand_score", 0.0))
        rate = max(0.0, intensity) + max(0.0, score) * 0.1
        digest = hashlib.sha256(f"{self.seed}|{self.window.get('window_id')}|{stop_id}".encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        expected = max(1, int(round(rate * self.horizon_seconds / 600.0)))
        offsets = np.sort(rng.uniform(0.0, self.horizon_seconds, size=expected))
        return [float(value) for value in offsets]

    def reset(self) -> Dict[str, Any]:
        stops = {}
        for stop_id in range(self.stop_count):
            schedule = self._arrival_schedule(stop_id)
            stops[stop_id] = StopState(stop_id=stop_id, arrival_schedule=schedule)
        vehicles = {
            agent_id: VehicleState(agent_id=agent_id, stop_index=agent_id % self.stop_count)
            for agent_id in range(self.num_agents)
        }
        self.state = {"stops": stops, "vehicles": vehicles}
        self.events = []
        self.accounting = CausalAccounting()
        self._wait_population = None
        self.accounting.passenger_demand_generated = sum(len(stop.arrival_schedule) for stop in stops.values())
        self.accounting.active_vehicle_ids = set(vehicles)
        return self.state_identity()

    def state_identity(self) -> Dict[str, Any]:
        vehicles = self.state["vehicles"]
        stops = self.state["stops"]
        payload = {
            "vehicles": [[v.agent_id, v.stop_index, round(v.clock_seconds, 6), round(v.distance_m, 6), v.acceleration_event_count, round(v.hold_seconds, 6), v.served_count] for v in vehicles.values()],
            "stops": [[s.stop_id, s.waiting_count, s.next_arrival_index, round(sum(s.queue_arrival_ts), 6)] for s in stops.values()],
        }
        return {
            "hash": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
            "vehicle_count": len(vehicles),
            "stop_count": len(stops),
            "served_total": self.accounting.passenger_served_count,
        }

    def _advance_arrivals(self, stop: StopState, to_ts: float) -> None:
        """Move every passenger whose authoritative arrival_ts has passed into the queue."""
        while stop.next_arrival_index < len(stop.arrival_schedule) and stop.arrival_schedule[stop.next_arrival_index] <= to_ts:
            stop.queue_arrival_ts.append(stop.arrival_schedule[stop.next_arrival_index])
            stop.next_arrival_index += 1

    def finalize_wait_population(self) -> Dict[str, Any]:
        """Close the passenger wait population at the fixed evaluation horizon end."""
        if self._wait_population is not None:
            return self._wait_population
        horizon_end = self.horizon_seconds
        completed: List[float] = []
        censored: List[float] = []
        post_horizon_boardings = 0
        for arrival_ts, board_ts in self.accounting.boarding_ledger:
            if board_ts <= horizon_end:
                completed.append(max(0.0, board_ts - arrival_ts))
            else:
                # Still waiting at the fixed horizon end: censored, never credited as served wait.
                censored.append(max(0.0, horizon_end - arrival_ts))
                post_horizon_boardings += 1
        for stop in self.state["stops"].values():
            pending = list(stop.queue_arrival_ts) + list(stop.arrival_schedule[stop.next_arrival_index:])
            for arrival_ts in pending:
                if arrival_ts <= horizon_end:
                    censored.append(max(0.0, horizon_end - arrival_ts))
        self.accounting.completed_wait_ledger = completed
        self.accounting.censored_wait_ledger = censored
        self.accounting.wait_total_passenger_seconds = float(sum(completed))
        self.accounting.wait_passenger_count = len(completed)
        canonical = measured_wait_p95(completed, censored)
        served_only = measured_wait_p95(completed, [])
        served_only.update({"canonical": False, "promotion_use_allowed": False, "performance_claim_allowed": False})
        self._wait_population = {
            "horizon_end_seconds": horizon_end,
            "arrival_entry_semantics": ARRIVAL_ENTRY_SEMANTICS,
            "carry_in_case": CARRY_IN_CASE,
            "completed_wait_ledger": completed,
            "censored_wait_ledger": censored,
            "canonical_p95": canonical,
            "served_only_p95_diagnostic": served_only,
            "boarding_event_count": len(self.accounting.boarding_ledger),
            "post_horizon_boarding_count": post_horizon_boardings,
            "boarded_within_horizon_count": len(completed),
            "unboarded_at_horizon_end_count": len(censored) - post_horizon_boardings,
            "demand_generated": int(self.accounting.passenger_demand_generated),
            "population_reconciles_demand": len(completed) + len(censored) == int(self.accounting.passenger_demand_generated),
        }
        return self._wait_population

    def step(self, agent_actions: Mapping[int, int], *, legal_mask: Mapping[int, Sequence[bool]], target_ids: Mapping[int, int], provenance: Mapping[str, Any]) -> Dict[str, Any]:
        """Apply one causal decision step; the action changes the next state."""
        if self.state is None:
            raise BridgeContractError("RESET_REQUIRED")
        pre_identity = self.state_identity()
        stops = self.state["stops"]
        vehicles = self.state["vehicles"]
        step_events: List[Dict[str, Any]] = []
        for agent_id, action in sorted(agent_actions.items()):
            action = int(action)
            mask = list(legal_mask.get(agent_id, [True] * ACTION_DIM))
            if not mask[action]:
                raise BridgeContractError("ILLEGAL_ACTION_APPLIED", f"agent={agent_id} action={action}")
            vehicle = vehicles[agent_id]
            stop = stops[vehicle.stop_index]
            start_ts = vehicle.clock_seconds
            self.accounting.decision_step_count += 1
            target_id = int(target_ids.get(agent_id, 0))
            if action != target_id:
                self.accounting.intervention_count += 1

            if stop.last_vehicle_arrival_ts is not None:
                headway = start_ts - stop.last_vehicle_arrival_ts
                if headway > 0:
                    self.accounting.headway_samples.append(headway)
                    self.accounting.headway_event_count += 1
                    if headway < BUNCHING_HEADWAY_SECONDS:
                        self.accounting.bunching_event_count += 1
                    self.accounting.schedulable_arrival_count += 1
                    if abs(headway - SCHEDULED_HEADWAY_SECONDS) <= ONTIME_TOLERANCE_SECONDS:
                        self.accounting.ontime_event_count += 1
            stop.last_vehicle_arrival_ts = start_ts

            served = 0
            if action == 0:  # HOLD: vehicle stays, queue keeps waiting
                elapsed = HOLD_STEP_SECONDS
                self._advance_arrivals(stop, start_ts + elapsed)
                vehicle.hold_seconds += elapsed
                self.accounting.hold_seconds += elapsed
            elif action == 1:  # SERVE: board the queue, then move
                self._advance_arrivals(stop, start_ts)
                board_ts = start_ts
                boarded = list(stop.queue_arrival_ts)
                served = len(boarded)
                for arrival_ts in boarded:
                    self.accounting.boarding_ledger.append((arrival_ts, board_ts))
                within = [max(0.0, b - a) for a, b in self.accounting.boarding_ledger if b <= self.horizon_seconds]
                self.accounting.wait_total_passenger_seconds = float(sum(within))
                self.accounting.wait_passenger_count = len(within)
                self.accounting.passenger_served_count += served
                stop.queue_arrival_ts.clear()
                dwell = DWELL_BASE_SECONDS + DWELL_PER_PASSENGER_SECONDS * served
                travel = SEGMENT_DISTANCE_M / CRUISE_SPEED_MPS
                elapsed = dwell + travel
                vehicle.distance_m += SEGMENT_DISTANCE_M
                vehicle.acceleration_event_count += 1
                vehicle.served_count += served
                self.accounting.distance_m += SEGMENT_DISTANCE_M
                self.accounting.acceleration_event_count += 1
                vehicle.stop_index = (vehicle.stop_index + 1) % self.stop_count
            else:  # CONDITIONAL_SKIP: move without boarding
                travel = SEGMENT_DISTANCE_M / CRUISE_SPEED_MPS
                elapsed = travel
                self._advance_arrivals(stop, start_ts + elapsed)
                vehicle.distance_m += SEGMENT_DISTANCE_M
                vehicle.acceleration_event_count += 1
                self.accounting.distance_m += SEGMENT_DISTANCE_M
                self.accounting.acceleration_event_count += 1
                vehicle.stop_index = (vehicle.stop_index + 1) % self.stop_count

            vehicle.clock_seconds = start_ts + elapsed
            step_events.append(
                {
                    "window_id": self.window.get("window_id"),
                    "event_ts": vehicle.clock_seconds,
                    "state_ts": self.window.get("start_iso"),
                    "condition_id": provenance.get("condition_id"),
                    "seed": self.seed,
                    "agent_id": agent_id,
                    "action_id": action,
                    "action_name": ACTION_NAMES[action],
                    "target_id": target_id,
                    "legal_action_mask": [bool(flag) for flag in mask],
                    "action_is_legal": True,
                    "simulator_pre_state_hash": pre_identity["hash"],
                    "passenger_served": served,
                    "stop_waiting_count_after": stop.waiting_count,
                    "wait_person_seconds_settled": self.accounting.wait_total_passenger_seconds,
                    "headway_seconds": self.accounting.headway_samples[-1] if self.accounting.headway_samples else None,
                    "distance_m": vehicle.distance_m,
                    "acceleration_event_count": vehicle.acceleration_event_count,
                    "hold_seconds": vehicle.hold_seconds,
                    "active_vehicle_count": len(self.accounting.active_vehicle_ids),
                    "passenger_ownership": "aggregate_stop_queue_ownership",
                    "passenger_identifier_available": False,
                    "policy_source_mode": provenance.get("policy_source_mode"),
                    "checkpoint_sha256": provenance.get("checkpoint_sha256"),
                    "adapter_source_mode": self.source_mode,
                    "causal_transition": True,
                    "causal_comparison_allowed": self.causal_comparison_allowed,
                    "terminated": False,
                    "truncated": False,
                }
            )
        post_identity = self.state_identity()
        for event in step_events:
            event["simulator_post_state_hash"] = post_identity["hash"]
        self.events.extend(step_events)
        return {"pre_state": pre_identity, "post_state": post_identity, "events": step_events}

    # -- rollup ---------------------------------------------------------------
    def window_rollup_row(self, *, condition_id: str, provenance: Mapping[str, Any], energy_model: Any, baseline_bus_count: int) -> Dict[str, Any]:
        """Build the rollup strictly from accumulated transition quantities."""
        if not self.events:
            raise BridgeContractError("NO_EVENTS_TO_ROLL_UP", "the rollup may only be built from simulator events")
        acc = self.accounting
        wait_population = self.finalize_wait_population()
        canonical_p95 = wait_population["canonical_p95"]
        energy = energy_model.compute_energy_proxy_from_components(
            distance_m=acc.distance_m,
            acceleration_event_count=acc.acceleration_event_count,
            hold_seconds=acc.hold_seconds,
            passenger_served_count=acc.passenger_served_count,
        )
        headways = np.array(acc.headway_samples, dtype=float)
        row = {
            "condition_id": str(condition_id).upper(),
            "seed": self.seed,
            "window_id": self.window.get("window_id"),
            "state_ts": self.window.get("start_iso"),
            "service_date": str(self.window.get("start_iso", ""))[:10],
            "time_band": self.window.get("time_band"),
            "evaluation_horizon_minutes": int(self.horizon_seconds // 60),
            "headway_mean_seconds": float(headways.mean()) if headways.size else 0.0,
            "headway_std_seconds": float(headways.std(ddof=1)) if headways.size > 1 else 0.0,
            "headway_sample_count": int(headways.size),
            "bunching_event_count": int(acc.bunching_event_count),
            "headway_event_count": int(acc.headway_event_count),
            "wait_total_passenger_seconds": float(acc.wait_total_passenger_seconds),
            "wait_passenger_count": int(acc.wait_passenger_count),
            "passenger_wait_p95_seconds": canonical_p95["value_seconds"],
            "passenger_wait_p95_status": canonical_p95["status"],
            "passenger_wait_p95_semantics": canonical_p95["semantics"],
            "passenger_wait_p95_population": canonical_p95["population_definition"],
            "passenger_wait_p95_percentile_method": canonical_p95["percentile_method"],
            "passenger_wait_p95_censor_reference": canonical_p95["censor_reference"],
            "wait_completed_passenger_count": int(canonical_p95["completed_count"]),
            "wait_censored_passenger_count": int(canonical_p95["censored_count"]),
            "wait_population_passenger_count": int(canonical_p95["population_count"]),
            "wait_censored_total_seconds": float(sum(wait_population["censored_wait_ledger"])),
            "wait_tail_measured": canonical_p95["status"] == "MEASURED",
            "wait_tail_fallback_used": False,
            "wait_tail_repair_id": WAIT_TAIL_REPAIR_ID,
            "wait_post_horizon_boarding_count": int(wait_population["post_horizon_boarding_count"]),
            "wait_censored_share": canonical_p95["censored_share"],
            "ontime_event_count": int(acc.ontime_event_count),
            "schedulable_arrival_count": int(acc.schedulable_arrival_count),
            "intervention_count": int(acc.intervention_count),
            "decision_step_count": int(acc.decision_step_count),
            "energy_proxy_total": float(energy["energy_kwh_equiv"] if "energy_kwh_equiv" in energy else energy.get("energy_proxy", 0.0)),
            "source_mode": self.source_mode,
            "passenger_demand_generated": int(acc.passenger_demand_generated),
            "passenger_served_count": int(acc.passenger_served_count),
            "distance_m": float(acc.distance_m),
            "acceleration_event_count": int(acc.acceleration_event_count),
            "hold_seconds": float(acc.hold_seconds),
            "active_bus_count": float(len(acc.active_vehicle_ids)),
            "baseline_bus_count": float(baseline_bus_count),
            "policy_source": provenance.get("policy_source_mode"),
            "checkpoint_path": provenance.get("checkpoint_path"),
            "checkpoint_loaded": True,
            "placeholder_fallback_used": False,
            "mock_action_used": False,
            "causal_transition": True,
            "kpi_provenance": "simulator_transition_accounting_only",
            "event_row_count": len(self.events),
        }
        forbidden = {"action_count", "nonzero_action_count", "policy_probability"}
        if forbidden & set(row):
            raise BridgeContractError("SYNTHETIC_KPI_FIELD_PRESENT", str(forbidden & set(row)))
        return row


KPI_FIELD_PROVENANCE = {
    "avg_wait_seconds": (
        "wait_total_passenger_seconds / wait_passenger_count over passengers boarded at SERVE settlement on or "
        "before the fixed evaluation horizon end; censored passengers never enter the mean"
    ),
    "cv_headway": "headway_std_seconds / headway_mean_seconds from vehicle arrival headway samples",
    "bunching_rate": "bunching_event_count / headway_event_count from headway samples",
    "on_time_rate": "ontime_event_count / schedulable_arrival_count from headway deviation",
    "intervention_rate": "intervention_count / decision_step_count from action vs target comparison",
    "energy_proxy": "frozen energy proxy over accumulated distance, acceleration events and hold seconds",
    "passenger_demand_generated": "seeded arrival schedule bound to the authoritative window demand fields",
    "passenger_served_count": "passengers boarded at SERVE transitions",
    "passenger_service_rate": "served / demand, both transition accounted",
    "passenger_wait_p95_seconds": (
        "measured unweighted numpy.percentile(q=95, method='linear') over served completed waits plus "
        "right-censored lower-bound waits of unserved passengers at the fixed evaluation horizon end; "
        "semantics LOWER_BOUND_UNDER_RIGHT_CENSORING"
    ),
    "energy_proxy_per_passenger": "energy proxy / served passengers",
    "fleet_reduction_ratio": "1 - active_bus_count / baseline_bus_count from vehicle accounting",
    "in_vehicle_time_seconds": "NOT_YET_MEASURABLE: no alighting event exists; no proxy emitted",
}
