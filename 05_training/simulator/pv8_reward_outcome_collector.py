from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite, sqrt
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from simulator.dynamics_replay_contract import canonical_hash


PV8_REWARD_SEMANTICS_VERSION = "PV8_REWARD_SEMANTICS_V2"
PV8_REWARD_V2_FREEZE_SHA256 = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
PV8_REWARD_V2_SETTLEMENT = "PV8_COMPONENT_SPECIFIC_EVENT_DRIVEN_SETTLEMENT_V1"


class RewardOutcomeCollectorError(ValueError):
    pass


class DuplicateDecisionError(RewardOutcomeCollectorError):
    pass


class DuplicateRewardEventError(RewardOutcomeCollectorError):
    pass


class RewardEventBoundaryError(RewardOutcomeCollectorError):
    pass


class RewardIdentityError(RewardOutcomeCollectorError):
    pass


class RewardOutcomeEventType(str, Enum):
    PASSENGER_GENERATED = "PASSENGER_GENERATED"
    PASSENGER_BOARDED = "PASSENGER_BOARDED"
    PASSENGER_ALIGHTED = "PASSENGER_ALIGHTED"
    REQUEST_COMPLETED = "REQUEST_COMPLETED"
    REQUEST_CANCELLED = "REQUEST_CANCELLED"
    VEHICLE_MOVEMENT = "VEHICLE_MOVEMENT"
    STOP_ARRIVAL = "STOP_ARRIVAL"
    FORCED_SAFETY_OVERRIDE = "FORCED_SAFETY_OVERRIDE"
    EXTERNAL_POLICY_INTERVENTION = "EXTERNAL_POLICY_INTERVENTION"


@dataclass(frozen=True)
class InitialRequestState:
    request_id: str
    passenger_id: str
    waiting_since_ts: int
    status: str = "WAITING"

    def __post_init__(self) -> None:
        if not self.request_id or not self.passenger_id:
            raise RewardIdentityError("initial request requires request_id and passenger_id")
        if int(self.waiting_since_ts) < 0:
            raise RewardEventBoundaryError("waiting_since_ts must be non-negative")
        if self.status not in {"WAITING", "ASSIGNED", "ONBOARD", "COMPLETED", "CANCELLED"}:
            raise RewardOutcomeCollectorError(f"unsupported initial request status: {self.status}")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "passenger_id": self.passenger_id,
            "waiting_since_ts": int(self.waiting_since_ts),
            "status": self.status,
        }


@dataclass(frozen=True)
class DecisionOutcomeBinding:
    decision_id: str
    decision_ts: int
    agent_id: int
    vehicle_token: str
    action_t: Optional[str]
    outcome_start_ts: int
    outcome_end_ts: int
    next_decision_ts: Optional[int]
    terminal_ts: Optional[int]
    active_bus: bool
    observation_hash: str
    k_mask_hash: str
    initial_requests: Tuple[InitialRequestState, ...] = ()
    episode_id: Optional[str] = None
    vehicle_slot_id: Optional[int] = None
    route_id: Optional[str] = None
    direction_id: Optional[str] = None
    occurrence_id: Optional[str] = None
    local_decision_ts: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.decision_id or not self.vehicle_token:
            raise RewardIdentityError("decision_id and vehicle_token are required")
        if int(self.outcome_start_ts) != int(self.decision_ts):
            raise RewardEventBoundaryError("outcome_start_ts must equal decision_ts")
        if int(self.outcome_end_ts) <= int(self.decision_ts):
            raise RewardEventBoundaryError("outcome_end_ts must be after decision_ts")
        boundary_count = int(self.next_decision_ts is not None) + int(self.terminal_ts is not None)
        if boundary_count != 1:
            raise RewardEventBoundaryError("exactly one next_decision_ts or terminal_ts is required")
        boundary = self.next_decision_ts if self.next_decision_ts is not None else self.terminal_ts
        if int(boundary) != int(self.outcome_end_ts):
            raise RewardEventBoundaryError("outcome_end_ts must match the selected boundary")
        if self.active_bus and not self.action_t:
            raise RewardIdentityError("active decision requires action_t")
        if not self.active_bus and self.action_t is not None:
            raise RewardIdentityError("inactive decision must not carry action_t")
        request_ids = [row.request_id for row in self.initial_requests]
        passenger_ids = [row.passenger_id for row in self.initial_requests]
        if len(request_ids) != len(set(request_ids)) or len(passenger_ids) != len(set(passenger_ids)):
            raise RewardIdentityError("initial request/passenger identities must be unique")

    @property
    def binding_hash(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision_ts": int(self.decision_ts),
            "agent_id": int(self.agent_id),
            "vehicle_token": self.vehicle_token,
            "action_t": self.action_t,
            "outcome_start_ts": int(self.outcome_start_ts),
            "outcome_end_ts": int(self.outcome_end_ts),
            "next_decision_ts": self.next_decision_ts,
            "terminal_ts": self.terminal_ts,
            "active_bus": bool(self.active_bus),
            "observation_hash": self.observation_hash,
            "k_mask_hash": self.k_mask_hash,
            "initial_requests": [row.to_payload() for row in self.initial_requests],
            "episode_id": self.episode_id,
            "vehicle_slot_id": self.vehicle_slot_id,
            "route_id": self.route_id,
            "direction_id": self.direction_id,
            "occurrence_id": self.occurrence_id,
            "local_decision_ts": self.local_decision_ts,
        }


@dataclass(frozen=True)
class RewardOutcomeEvent:
    event_id: str
    event_ts: int
    event_type: RewardOutcomeEventType
    agent_id: int
    vehicle_token: str
    request_id: Optional[str] = None
    passenger_id: Optional[str] = None
    route_id: Optional[str] = None
    direction_id: Optional[str] = None
    route_stop_occurrence_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id or not self.vehicle_token:
            raise RewardIdentityError("event_id and vehicle_token are required")
        if int(self.event_ts) < 0:
            raise RewardEventBoundaryError("event_ts must be non-negative")
        if self.event_type in {
            RewardOutcomeEventType.PASSENGER_GENERATED,
            RewardOutcomeEventType.PASSENGER_BOARDED,
            RewardOutcomeEventType.PASSENGER_ALIGHTED,
            RewardOutcomeEventType.REQUEST_COMPLETED,
            RewardOutcomeEventType.REQUEST_CANCELLED,
        } and (not self.request_id or not self.passenger_id):
            raise RewardIdentityError(f"{self.event_type.value} requires request_id and passenger_id")

    @property
    def event_hash(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_ts": int(self.event_ts),
            "event_type": self.event_type.value,
            "agent_id": int(self.agent_id),
            "vehicle_token": self.vehicle_token,
            "request_id": self.request_id,
            "passenger_id": self.passenger_id,
            "route_id": self.route_id,
            "direction_id": self.direction_id,
            "route_stop_occurrence_id": self.route_stop_occurrence_id,
            "metadata": dict(self.metadata),
        }


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise RewardOutcomeCollectorError("percentile requires at least one value")
    if not 0.0 <= float(quantile) <= 1.0:
        raise RewardOutcomeCollectorError("quantile must be in [0,1]")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(quantile)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class CausalOutcomeCollector:
    def __init__(self) -> None:
        self._bindings: Dict[str, DecisionOutcomeBinding] = {}
        self._events: Dict[str, List[RewardOutcomeEvent]] = {}
        self._event_ids: set[str] = set()
        self._finalized: Dict[str, Dict[str, Any]] = {}

    def register_decision(self, binding: DecisionOutcomeBinding) -> None:
        if binding.decision_id in self._bindings:
            raise DuplicateDecisionError(f"duplicate decision_id: {binding.decision_id}")
        self._bindings[binding.decision_id] = binding
        self._events[binding.decision_id] = []

    def record_event(self, decision_id: str, event: RewardOutcomeEvent) -> None:
        if decision_id not in self._bindings:
            raise RewardIdentityError(f"unknown decision_id: {decision_id}")
        if decision_id in self._finalized:
            raise RewardOutcomeCollectorError("cannot append event after finalization")
        binding = self._bindings[decision_id]
        if not binding.active_bus:
            raise RewardIdentityError("inactive decision cannot receive reward events")
        if event.event_id in self._event_ids:
            raise DuplicateRewardEventError(f"duplicate reward event: {event.event_id}")
        if event.agent_id != binding.agent_id or event.vehicle_token != binding.vehicle_token:
            raise RewardIdentityError("reward event agent/vehicle identity does not match decision binding")
        if int(event.event_ts) <= int(binding.outcome_start_ts):
            raise RewardEventBoundaryError("reward event must occur after action commitment")
        if int(event.event_ts) > int(binding.outcome_end_ts):
            raise RewardEventBoundaryError("reward event exceeds next-decision/terminal boundary")
        self._event_ids.add(event.event_id)
        self._events[decision_id].append(event)

    def finalize(self, decision_id: str) -> Dict[str, Any]:
        if decision_id in self._finalized:
            return dict(self._finalized[decision_id])
        if decision_id not in self._bindings:
            raise RewardIdentityError(f"unknown decision_id: {decision_id}")
        binding = self._bindings[decision_id]
        if not binding.active_bus:
            result = {
                "decision_id": decision_id,
                "binding_hash": binding.binding_hash,
                "agent_id": binding.agent_id,
                "vehicle_token": binding.vehicle_token,
                "active_bus": False,
                "learning_sample": False,
                "action_t": None,
                "event_count": 0,
                "reward_value_materialized": False,
                "reward_semantics_version": PV8_REWARD_SEMANTICS_VERSION,
                "reward_freeze_sha256": PV8_REWARD_V2_FREEZE_SHA256,
                "reward_v2_collector_role": "INACTIVE_SLOT_BYPASS",
                "p95_training_reward_enabled": False,
                "p95_training_normalization_active": False,
                "p95_local_transition_owner": None,
                "H240_active_reward_settlement": False,
                "H660_active_reward_settlement": False,
                "observation_hash_unchanged": True,
                "k_mask_hash_unchanged": True,
            }
            self._finalized[decision_id] = result
            return dict(result)

        events = sorted(self._events[decision_id], key=lambda row: (row.event_ts, row.event_type.value, row.event_id))
        cohort: Dict[str, Dict[str, Any]] = {
            row.request_id: {
                "request_id": row.request_id,
                "passenger_id": row.passenger_id,
                "waiting_since_ts": int(row.waiting_since_ts),
                "status_at_start": row.status,
            }
            for row in binding.initial_requests
        }
        generated_ids: set[str] = set()
        boarded_at: Dict[str, int] = {}
        completed_ids: set[str] = set()
        cancelled_ids: set[str] = set()
        boarding_ids: set[str] = set()
        alighting_ids: set[str] = set()
        distance_m = 0.0
        travel_seconds = 0.0
        dwell_seconds = 0.0
        energy_primitives: List[Dict[str, Any]] = []
        arrivals: List[Dict[str, Any]] = []
        forced_override_ids: set[str] = set()
        external_intervention_ids: set[str] = set()

        for event in events:
            if event.event_type == RewardOutcomeEventType.PASSENGER_GENERATED:
                if event.request_id in cohort:
                    raise RewardIdentityError(f"request identity reused: {event.request_id}")
                waiting_since = int(event.metadata.get("waiting_since_ts", event.event_ts))
                cohort[str(event.request_id)] = {
                    "request_id": str(event.request_id),
                    "passenger_id": str(event.passenger_id),
                    "waiting_since_ts": waiting_since,
                    "status_at_start": "GENERATED_IN_WINDOW",
                }
                generated_ids.add(str(event.request_id))
            elif event.event_type == RewardOutcomeEventType.PASSENGER_BOARDED:
                self._require_cohort_identity(cohort, event)
                boarded_at[str(event.request_id)] = int(event.event_ts)
                boarding_ids.add(str(event.request_id))
            elif event.event_type == RewardOutcomeEventType.PASSENGER_ALIGHTED:
                self._require_cohort_identity(cohort, event)
                alighting_ids.add(str(event.request_id))
                completed_ids.add(str(event.request_id))
            elif event.event_type == RewardOutcomeEventType.REQUEST_COMPLETED:
                self._require_cohort_identity(cohort, event)
                completed_ids.add(str(event.request_id))
            elif event.event_type == RewardOutcomeEventType.REQUEST_CANCELLED:
                self._require_cohort_identity(cohort, event)
                cancelled_ids.add(str(event.request_id))
            elif event.event_type == RewardOutcomeEventType.VEHICLE_MOVEMENT:
                distance_m += self._finite_nonnegative(event.metadata, "distance_m", default=0.0)
                travel_seconds += self._finite_nonnegative(event.metadata, "travel_seconds", default=0.0)
                dwell_seconds += self._finite_nonnegative(event.metadata, "dwell_seconds", default=0.0)
                energy_primitives.append({
                    "event_id": event.event_id,
                    "distance_m": self._finite_nonnegative(event.metadata, "distance_m", default=0.0),
                    "travel_seconds": self._finite_nonnegative(event.metadata, "travel_seconds", default=0.0),
                    "dwell_seconds": self._finite_nonnegative(event.metadata, "dwell_seconds", default=0.0),
                    "load_time_integral": self._optional_finite_nonnegative(event.metadata, "load_time_integral"),
                    "traction_energy_kwh": self._optional_finite_nonnegative(event.metadata, "traction_energy_kwh"),
                    "energy_model_version": event.metadata.get("energy_model_version"),
                })
            elif event.event_type == RewardOutcomeEventType.STOP_ARRIVAL:
                if not event.route_id or not event.direction_id or not event.route_stop_occurrence_id:
                    raise RewardIdentityError("STOP_ARRIVAL requires route/direction/occurrence identity")
                arrivals.append({
                    "event_id": event.event_id,
                    "event_ts": int(event.event_ts),
                    "agent_id": int(event.agent_id),
                    "vehicle_token": event.vehicle_token,
                    "route_id": event.route_id,
                    "direction_id": event.direction_id,
                    "route_stop_occurrence_id": event.route_stop_occurrence_id,
                })
            elif event.event_type == RewardOutcomeEventType.FORCED_SAFETY_OVERRIDE:
                forced_override_ids.add(event.event_id)
            elif event.event_type == RewardOutcomeEventType.EXTERNAL_POLICY_INTERVENTION:
                external_intervention_ids.add(event.event_id)

        wait_rows = []
        for request_id, row in sorted(cohort.items()):
            status_at_start = str(row["status_at_start"])
            if status_at_start in {"ONBOARD", "COMPLETED", "CANCELLED"}:
                continue
            waiting_since = int(row["waiting_since_ts"])
            wait_end = boarded_at.get(request_id, int(binding.outcome_end_ts))
            wait_seconds = float(wait_end - waiting_since)
            if wait_seconds < 0:
                raise RewardEventBoundaryError("wait duration is negative")
            wait_rows.append({
                "request_id": request_id,
                "passenger_id": row["passenger_id"],
                "waiting_since_ts": waiting_since,
                "wait_end_ts": wait_end,
                "wait_seconds": wait_seconds,
                "censored_at_outcome_end": request_id not in boarded_at,
            })
        waits = [float(row["wait_seconds"]) for row in wait_rows]
        eligible_ids = set(cohort)
        service_rate = float(len(completed_ids & eligible_ids) / len(eligible_ids)) if eligible_ids else None
        transition_identity = {
            "episode_id": binding.episode_id,
            "vehicle_slot_id": binding.vehicle_slot_id if binding.vehicle_slot_id is not None else binding.agent_id,
            "route_id": binding.route_id,
            "direction_id": binding.direction_id,
            "occurrence_id": binding.occurrence_id,
            "local_decision_ts": binding.local_decision_ts if binding.local_decision_ts is not None else binding.decision_ts,
            "action": binding.action_t,
        }
        resolved_wait_rows = [
            {
                "passenger_id": row["passenger_id"],
                "request_id": row["request_id"],
                "originating_transition_id": decision_id,
                "wait_ownership_key": f"{decision_id}:{row['request_id']}:{row['passenger_id']}",
                "request_ts": int(row["waiting_since_ts"]),
                "local_decision_ts": int(transition_identity["local_decision_ts"]),
                "first_eligible_service_ts": int(row["wait_end_ts"]),
                "actual_board_ts": int(row["wait_end_ts"]),
                "schedule_wait_seconds": float(row["wait_seconds"]),
                "alignment_excess_wait_seconds": 0.0,
                "total_wait_seconds": float(row["wait_seconds"]),
                "censored_at_outcome_end": bool(row["censored_at_outcome_end"]),
            }
            for row in wait_rows
            if not bool(row["censored_at_outcome_end"])
        ]
        result = {
            "decision_id": decision_id,
            "binding_hash": binding.binding_hash,
            "decision_ts": int(binding.decision_ts),
            "agent_id": int(binding.agent_id),
            "vehicle_token": binding.vehicle_token,
            "active_bus": True,
            "learning_sample": True,
            "action_t": binding.action_t,
            "outcome_start_ts": int(binding.outcome_start_ts),
            "outcome_end_ts": int(binding.outcome_end_ts),
            "next_decision_ts": binding.next_decision_ts,
            "terminal_ts": binding.terminal_ts,
            "event_count": len(events),
            "event_trace_hash": canonical_hash([event.to_payload() for event in events]),
            "passenger_generated_count": len(generated_ids),
            "eligible_request_count": len(eligible_ids),
            "passenger_served_count": len(completed_ids & eligible_ids),
            "passenger_cancelled_count": len(cancelled_ids & eligible_ids),
            "boarding_completion_count": len(boarding_ids),
            "alighting_completion_count": len(alighting_ids),
            "service_rate": service_rate,
            "wait_observation_count": len(waits),
            "censored_wait_count": sum(bool(row["censored_at_outcome_end"]) for row in wait_rows),
            "individual_waits": wait_rows,
            "avg_wait_seconds": float(sum(waits) / len(waits)) if waits else None,
            "p95_wait_seconds": float(percentile(waits, 0.95)) if waits else None,
            "vehicle_distance_m": float(distance_m),
            "vehicle_travel_seconds": float(travel_seconds),
            "vehicle_dwell_seconds": float(dwell_seconds),
            "energy_required_primitives": energy_primitives,
            "headway_required_primitives": arrivals,
            "forced_safety_override_count": len(forced_override_ids),
            "external_policy_intervention_count": len(external_intervention_ids),
            "intervention_count_candidate": len(forced_override_ids | external_intervention_ids),
            "ordinary_k_mask_restriction_counted": False,
            "reward_value_materialized": False,
            "reward_semantics_version": PV8_REWARD_SEMANTICS_VERSION,
            "reward_freeze_sha256": PV8_REWARD_V2_FREEZE_SHA256,
            "reward_v2_settlement": PV8_REWARD_V2_SETTLEMENT,
            "reward_v2_collector_role": "COMPONENT_EVENT_INPUT_ONLY",
            "transition_identity": transition_identity,
            "transition_id": decision_id,
            "reward_v2_affected_wait_rows": resolved_wait_rows,
            "duplicate_wait_ownership": 0,
            "duplicate_service_ownership": 0,
            "p95_semantics": "P95_EVALUATION_ONLY",
            "p95_training_reward_enabled": False,
            "p95_training_normalization_active": False,
            "p95_local_transition_owner": None,
            "H240_active_reward_settlement": False,
            "H660_active_reward_settlement": False,
            "missed_eligible_service_role": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
            "alignment_excess_wait_role": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
            "observation_hash_unchanged": True,
            "k_mask_hash_unchanged": True,
            "future_information_written_to_observation_or_mask": False,
        }
        self._finalized[decision_id] = result
        return dict(result)

    @staticmethod
    def _require_cohort_identity(cohort: Mapping[str, Mapping[str, Any]], event: RewardOutcomeEvent) -> None:
        row = cohort.get(str(event.request_id))
        if row is None or row.get("passenger_id") != event.passenger_id:
            raise RewardIdentityError("passenger/request event does not match the causal cohort")

    @staticmethod
    def _finite_nonnegative(metadata: Mapping[str, Any], key: str, *, default: float) -> float:
        value = float(metadata.get(key, default))
        if not isfinite(value) or value < 0:
            raise RewardOutcomeCollectorError(f"{key} must be finite and non-negative")
        return value

    @staticmethod
    def _optional_finite_nonnegative(metadata: Mapping[str, Any], key: str) -> Optional[float]:
        if metadata.get(key) is None:
            return None
        value = float(metadata[key])
        if not isfinite(value) or value < 0:
            raise RewardOutcomeCollectorError(f"{key} must be finite and non-negative")
        return value


def aggregate_team_outcomes(
    results: Sequence[Mapping[str, Any]],
    *,
    bunching_threshold_seconds: Optional[float] = None,
    reference_fleet_size: Optional[int] = None,
) -> Dict[str, Any]:
    active = [dict(row) for row in results if row.get("learning_sample")]
    waits = [
        float(wait["wait_seconds"])
        for row in active
        for wait in row.get("individual_waits", [])
    ]
    eligible = sum(int(row.get("eligible_request_count", 0)) for row in active)
    served = sum(int(row.get("passenger_served_count", 0)) for row in active)
    arrivals = [
        dict(arrival)
        for row in active
        for arrival in row.get("headway_required_primitives", [])
    ]
    grouped: Dict[Tuple[str, str, str], List[int]] = {}
    for row in arrivals:
        key = (str(row["route_id"]), str(row["direction_id"]), str(row["route_stop_occurrence_id"]))
        grouped.setdefault(key, []).append(int(row["event_ts"]))
    headways = []
    for timestamps in grouped.values():
        ordered = sorted(set(timestamps))
        headways.extend(float(right - left) for left, right in zip(ordered, ordered[1:]) if right > left)
    mean_headway = float(sum(headways) / len(headways)) if headways else None
    headway_cv = None
    if len(headways) >= 2 and mean_headway and mean_headway > 0:
        variance = sum((value - mean_headway) ** 2 for value in headways) / len(headways)
        headway_cv = float(sqrt(variance) / mean_headway)
    bunching_rate = None
    if bunching_threshold_seconds is not None and headways:
        threshold = float(bunching_threshold_seconds)
        if not isfinite(threshold) or threshold <= 0:
            raise RewardOutcomeCollectorError("bunching threshold must be positive and finite")
        bunching_rate = float(sum(value < threshold for value in headways) / len(headways))
    active_vehicle_tokens = sorted({str(row["vehicle_token"]) for row in active})
    fleet_reduction = None
    if reference_fleet_size is not None:
        if int(reference_fleet_size) <= 0:
            raise RewardOutcomeCollectorError("reference_fleet_size must be positive")
        fleet_reduction = float((int(reference_fleet_size) - len(active_vehicle_tokens)) / int(reference_fleet_size))
    energy_values = [
        primitive.get("traction_energy_kwh")
        for row in active
        for primitive in row.get("energy_required_primitives", [])
    ]
    energy_supported = bool(energy_values) and all(value is not None for value in energy_values)
    energy_total = float(sum(float(value) for value in energy_values)) if energy_supported else None
    return {
        "active_learning_sample_count": len(active),
        "active_vehicle_count": len(active_vehicle_tokens),
        "inactive_bypassed_count": sum(not row.get("learning_sample", False) for row in results),
        "eligible_request_count": eligible,
        "passenger_served_count": served,
        "service_rate": float(served / eligible) if eligible else None,
        "wait_observation_count": len(waits),
        "avg_wait_seconds": float(sum(waits) / len(waits)) if waits else None,
        "p95_wait_seconds": float(percentile(waits, 0.95)) if waits else None,
        "headway_interval_count": len(headways),
        "headway_values_seconds": headways,
        "mean_headway_seconds": mean_headway,
        "headway_cv": headway_cv,
        "bunching_threshold_seconds": bunching_threshold_seconds,
        "bunching_rate": bunching_rate,
        "vehicle_distance_m": float(sum(float(row.get("vehicle_distance_m", 0.0)) for row in active)),
        "vehicle_travel_seconds": float(sum(float(row.get("vehicle_travel_seconds", 0.0)) for row in active)),
        "vehicle_dwell_seconds": float(sum(float(row.get("vehicle_dwell_seconds", 0.0)) for row in active)),
        "energy_primitive_complete": energy_supported,
        "energy_total_kwh": energy_total,
        "energy_per_passenger_kwh": float(energy_total / served) if energy_total is not None and served > 0 else None,
        "reference_fleet_size": reference_fleet_size,
        "fleet_reduction": fleet_reduction,
        "intervention_count_candidate": sum(int(row.get("intervention_count_candidate", 0)) for row in active),
        "reward_value_materialized": False,
    }


def build_reward_v2_transition_input(
    finalized_outcome: Mapping[str, Any],
    *,
    pickup_obligation_count: int = 0,
    dropoff_obligation_count: int = 0,
    approved_static_mandatory_obligation_count: int = 0,
    completed_pickup_obligation_count: int = 0,
    completed_dropoff_obligation_count: int = 0,
    completed_static_mandatory_obligation_count: int = 0,
) -> Dict[str, Any]:
    identity = dict(finalized_outcome.get("transition_identity") or {})
    transition_id = str(finalized_outcome.get("transition_id") or finalized_outcome["decision_id"])
    return {
        "reward_mode": "PV8_REWARD_V2",
        "reward_semantics_version": PV8_REWARD_SEMANTICS_VERSION,
        "reward_freeze_sha256": PV8_REWARD_V2_FREEZE_SHA256,
        "transition_id": transition_id,
        "episode_id": identity.get("episode_id"),
        "vehicle_slot_id": identity.get("vehicle_slot_id", finalized_outcome.get("agent_id")),
        "route_id": identity.get("route_id") or "UNSPECIFIED_ROUTE",
        "direction_id": identity.get("direction_id") or "UNSPECIFIED_DIRECTION",
        "occurrence_id": identity.get("occurrence_id") or "UNSPECIFIED_OCCURRENCE",
        "local_decision_ts": identity.get("local_decision_ts", finalized_outcome.get("decision_ts")),
        "action": identity.get("action", finalized_outcome.get("action_t")),
        "pickup_obligation_count": int(pickup_obligation_count),
        "dropoff_obligation_count": int(dropoff_obligation_count),
        "approved_static_mandatory_obligation_count": int(approved_static_mandatory_obligation_count),
        "completed_pickup_obligation_count": int(completed_pickup_obligation_count),
        "completed_dropoff_obligation_count": int(completed_dropoff_obligation_count),
        "completed_static_mandatory_obligation_count": int(completed_static_mandatory_obligation_count),
        "affected_wait_rows": list(finalized_outcome.get("reward_v2_affected_wait_rows") or []),
        "forced_safety_override_count": int(finalized_outcome.get("forced_safety_override_count", 0)),
        "external_policy_intervention_count": int(finalized_outcome.get("external_policy_intervention_count", 0)),
        "ordinary_k_mask_restriction_counted": False,
        "p95_training_reward_enabled": False,
        "p95_training_normalization_active": False,
        "representative_reward_rematerialization": False,
    }


def map_k4_event_log(
    events: Iterable[Mapping[str, Any]],
    *,
    agent_id: int,
    vehicle_token: str,
    event_id_prefix: str,
) -> List[RewardOutcomeEvent]:
    mapped: List[RewardOutcomeEvent] = []
    event_rows = [dict(row) for row in events]
    waiting_since_by_passenger = {
        str(row["passenger_id"]): int(row["event_ts"])
        for row in event_rows
        if row.get("transition") == "passenger_waiting" and row.get("passenger_id") is not None
    }
    event_type_by_transition = {
        "request_created": RewardOutcomeEventType.PASSENGER_GENERATED,
        "passenger_boarded": RewardOutcomeEventType.PASSENGER_BOARDED,
        "passenger_alighted": RewardOutcomeEventType.PASSENGER_ALIGHTED,
        "request_completed": RewardOutcomeEventType.REQUEST_COMPLETED,
        "request_cancelled": RewardOutcomeEventType.REQUEST_CANCELLED,
    }
    for row in event_rows:
        transition = str(row.get("transition"))
        event_type = event_type_by_transition.get(transition)
        if event_type is None:
            continue
        event_agent = int(row.get("agent_id", agent_id))
        event_vehicle = str(row.get("vehicle_token", vehicle_token))
        passenger_id = str(row.get("passenger_id")) if row.get("passenger_id") is not None else None
        metadata: Dict[str, Any] = {
            "k4_transition": transition,
            "k4_event_sequence": int(row["event_sequence"]),
        }
        if transition == "request_created" and passenger_id in waiting_since_by_passenger:
            metadata["waiting_since_ts"] = waiting_since_by_passenger[passenger_id]
        mapped.append(RewardOutcomeEvent(
            event_id=f"{event_id_prefix}:{int(row['event_sequence']):04d}:{transition}",
            event_ts=int(row["event_ts"]),
            event_type=event_type,
            agent_id=event_agent,
            vehicle_token=event_vehicle,
            request_id=str(row.get("request_id")) if row.get("request_id") is not None else None,
            passenger_id=passenger_id,
            metadata=metadata,
        ))
    return mapped
