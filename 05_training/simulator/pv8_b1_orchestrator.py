from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    from simulator.k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime, RuntimeVersionBinding
    from simulator.k_mask_snapshot_lifecycle import (
        GlobalKMaskSnapshot,
        GlobalKMaskSnapshotLifecycle,
        KMaskAgentContext,
        clone_global_k_mask_snapshot,
        deserialize_global_k_mask_snapshot,
        serialize_global_k_mask_snapshot,
    )
    from simulator.k_safety_state import PassengerStatus, RequestStatus, ServiceObligationStateMachine
    from simulator.pv8_reward_outcome_collector import (
        CausalOutcomeCollector,
        DecisionOutcomeBinding,
        InitialRequestState,
        RewardOutcomeEvent,
        RewardOutcomeEventType,
        build_reward_v2_transition_input,
        map_k4_event_log,
    )
    from simulator.suseong_service_transition_engine import (
        ENGINE_ACTION_HOLD,
        ENGINE_ACTION_SERVE_MOVE,
        StopServiceResult,
        TransitionConfig,
        advance_vehicle_time_budget,
    )
except ImportError:  # Direct simulator-directory test execution.
    from k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime, RuntimeVersionBinding
    from k_mask_snapshot_lifecycle import (
        GlobalKMaskSnapshot,
        GlobalKMaskSnapshotLifecycle,
        KMaskAgentContext,
        clone_global_k_mask_snapshot,
        deserialize_global_k_mask_snapshot,
        serialize_global_k_mask_snapshot,
    )
    from k_safety_state import PassengerStatus, RequestStatus, ServiceObligationStateMachine
    from pv8_reward_outcome_collector import (
        CausalOutcomeCollector,
        DecisionOutcomeBinding,
        InitialRequestState,
        RewardOutcomeEvent,
        RewardOutcomeEventType,
        build_reward_v2_transition_input,
        map_k4_event_log,
    )
    from suseong_service_transition_engine import (
        ENGINE_ACTION_HOLD,
        ENGINE_ACTION_SERVE_MOVE,
        StopServiceResult,
        TransitionConfig,
        advance_vehicle_time_budget,
    )


RESEARCH_DEMAND_CLASS = "CONTRACT_FIXED_RESEARCH_DEMAND"
B1_ACTION_SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
B1_ACTION_HOLD = "HOLD_CURRENT_POSITION"
H4_SECONDS = 240
PV8_REWARD_SEMANTICS_VERSION = "PV8_REWARD_SEMANTICS_V2"
PV8_REWARD_V2_FREEZE_SHA256 = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
PV8_REWARD_V2_EVENT_ORDER = [
    "VEHICLE_ARRIVAL",
    "STATE_SNAPSHOT",
    "OBLIGATION_SNAPSHOT",
    "K_MASK_BUILD",
    "ACTION_SELECTION",
    "ACTION_VALIDATION",
    "BOARDING_ALIGHTING",
    "LOCAL_SERVICE_SETTLEMENT",
    "HOLD_IF_APPLICABLE",
    "DEPARTURE",
    "NEXT_LINK_TRAVEL",
]

try:
    from rewards.mappo_reward_v1 import compute_reward_v2
except ImportError:  # Direct rewards-directory test execution.
    from mappo_reward_v1 import compute_reward_v2


class PV8B1OrchestratorError(ValueError):
    pass


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PassengerRequestScheduleRow:
    passenger_id: str
    request_id: str
    request_ts: int
    origin_stop: str
    destination_stop: str
    route_id: str
    direction_id: str
    origin_stop_sequence: int
    destination_stop_sequence: int
    agent_id: int
    vehicle_token: str
    cancellation_ts: Optional[int] = None
    source_class: str = RESEARCH_DEMAND_CLASS
    research_generated: bool = True

    def __post_init__(self) -> None:
        required = {
            "passenger_id": self.passenger_id,
            "request_id": self.request_id,
            "origin_stop": self.origin_stop,
            "destination_stop": self.destination_stop,
            "route_id": self.route_id,
            "direction_id": self.direction_id,
            "vehicle_token": self.vehicle_token,
        }
        if any(not str(value).strip() for value in required.values()):
            raise PV8B1OrchestratorError("schedule identity and route fields must be non-empty")
        if int(self.request_ts) < 0:
            raise PV8B1OrchestratorError("request_ts must be non-negative")
        if self.origin_stop == self.destination_stop:
            raise PV8B1OrchestratorError("origin and destination stops must differ")
        if int(self.destination_stop_sequence) <= int(self.origin_stop_sequence):
            raise PV8B1OrchestratorError("destination must be downstream of origin")
        if self.cancellation_ts is not None and int(self.cancellation_ts) < int(self.request_ts):
            raise PV8B1OrchestratorError("cancellation cannot precede request creation")
        if self.source_class != RESEARCH_DEMAND_CLASS or self.research_generated is not True:
            raise PV8B1OrchestratorError("fixture schedules must retain the research-generated claim boundary")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "passenger_id": self.passenger_id,
            "request_id": self.request_id,
            "request_ts": int(self.request_ts),
            "origin_stop": self.origin_stop,
            "destination_stop": self.destination_stop,
            "route_id": self.route_id,
            "direction_id": self.direction_id,
            "origin_stop_sequence": int(self.origin_stop_sequence),
            "destination_stop_sequence": int(self.destination_stop_sequence),
            "agent_id": int(self.agent_id),
            "vehicle_token": self.vehicle_token,
            "cancellation_ts": int(self.cancellation_ts) if self.cancellation_ts is not None else None,
            "source_class": self.source_class,
            "research_generated": bool(self.research_generated),
        }


class TypedPV8B1Orchestrator:
    """Fixture-safe typed bridge across K4, K8/K9, service transitions, and H4 collection."""

    def __init__(
        self,
        *,
        runtime: FixedVehicleOccurrenceMaskRuntime,
        version_binding: RuntimeVersionBinding,
        rule_rows: Sequence[Mapping[str, Any]],
        fixed_vehicle_bindings: Mapping[int, str],
    ) -> None:
        bindings = {int(agent): str(token) for agent, token in fixed_vehicle_bindings.items()}
        if set(bindings) != set(range(8)) or len(set(bindings.values())) != 8:
            raise PV8B1OrchestratorError("exactly eight unique fixed physical-vehicle bindings are required")
        self.runtime = runtime
        self.version_binding = version_binding
        self.lifecycle = GlobalKMaskSnapshotLifecycle(runtime, version_binding)
        self.fixed_vehicle_bindings = bindings
        self.rules = [dict(row) for row in rule_rows]
        self.rules_by_occurrence = {str(row["route_stop_occurrence_id"]): row for row in self.rules}
        if len(self.rules_by_occurrence) != len(self.rules):
            raise PV8B1OrchestratorError("route_stop_occurrence_id must be unique")
        self.stop_ids = {str(row["stop_id"]) for row in self.rules}

    def new_state(self) -> ServiceObligationStateMachine:
        state = ServiceObligationStateMachine()
        for agent_id, token in sorted(self.fixed_vehicle_bindings.items()):
            state.register_vehicle(agent_id, token)
        for stop_id in sorted(self.stop_ids):
            state.register_stop(stop_id)
        return state

    def apply_schedule(
        self,
        state: ServiceObligationStateMachine,
        rows: Sequence[PassengerRequestScheduleRow],
    ) -> None:
        passenger_ids = [row.passenger_id for row in rows]
        request_ids = [row.request_id for row in rows]
        if len(passenger_ids) != len(set(passenger_ids)) or len(request_ids) != len(set(request_ids)):
            raise PV8B1OrchestratorError("schedule passenger_id and request_id values must be unique")
        for row in sorted(rows, key=lambda item: (item.request_ts, item.request_id)):
            if self.fixed_vehicle_bindings.get(int(row.agent_id)) != row.vehicle_token:
                raise PV8B1OrchestratorError("schedule assignment violates fixed physical-vehicle identity")
            if row.origin_stop not in self.stop_ids or row.destination_stop not in self.stop_ids:
                raise PV8B1OrchestratorError("schedule origin/destination is outside the frozen occurrence domain")
            state.schedule_transition(
                "passenger_waiting",
                row.request_ts,
                passenger_id=row.passenger_id,
                pickup_stop=row.origin_stop,
                dropoff_stop=row.destination_stop,
            )
            state.schedule_transition(
                "request_created",
                row.request_ts,
                request_id=row.request_id,
                passenger_id=row.passenger_id,
                service_leg_id=f"research-leg:{row.request_id}",
            )
            state.schedule_transition(
                "request_assigned",
                row.request_ts,
                request_id=row.request_id,
                agent_id=int(row.agent_id),
                vehicle_token=row.vehicle_token,
            )
            if row.cancellation_ts is not None:
                state.schedule_transition("request_cancelled", row.cancellation_ts, request_id=row.request_id)

    def capture_snapshot(
        self,
        *,
        state: ServiceObligationStateMachine,
        cycle_index: int,
        decision_ts: int,
        occurrence_by_agent: Mapping[int, Mapping[str, Any]],
        active_by_agent: Mapping[int, bool],
        identity_transition_by_agent: Optional[Mapping[int, Optional[str]]] = None,
    ) -> GlobalKMaskSnapshot:
        transitions = dict(identity_transition_by_agent or {})
        contexts: List[KMaskAgentContext] = []
        for agent_id in range(8):
            active = bool(active_by_agent.get(agent_id, False))
            rule = occurrence_by_agent.get(agent_id) if active else None
            if active and rule is None:
                raise PV8B1OrchestratorError(f"active agent {agent_id} lacks occurrence context")
            contexts.append(
                KMaskAgentContext(
                    agent_id=agent_id,
                    vehicle_token=self.fixed_vehicle_bindings[agent_id],
                    active_bus_mask=active,
                    decision_ts=int(decision_ts),
                    obligation_state_machine=state,
                    route_id=str(rule["route_id"]) if rule is not None else None,
                    direction_id=str(rule["direction_id"]) if rule is not None else None,
                    stop_sequence=int(rule["stop_sequence"]) if rule is not None else None,
                    stop_id=str(rule["stop_id"]) if rule is not None else None,
                    route_stop_occurrence_id=str(rule["route_stop_occurrence_id"]) if rule is not None else None,
                    identity_transition_type=transitions.get(agent_id),
                )
            )
        return self.lifecycle.capture_cycle(
            cycle_index=int(cycle_index),
            cycle_timestamp=str(int(decision_ts)),
            contexts=contexts,
        )

    @staticmethod
    def select_b1_actions(snapshot: GlobalKMaskSnapshot) -> Dict[int, Optional[str]]:
        actions: Dict[int, Optional[str]] = {}
        for row in snapshot.to_payload()["agent_snapshots"]:
            agent_id = int(row["agent_id"])
            if not bool(row["active_bus_mask"]):
                actions[agent_id] = None
                continue
            mask = [bool(value) for value in row["k_action_mask"]]
            if mask[1]:
                actions[agent_id] = B1_ACTION_SERVE
            elif mask[0]:
                actions[agent_id] = B1_ACTION_HOLD
            else:
                raise PV8B1OrchestratorError(f"active agent {agent_id} has no B1-valid action")
        return actions

    @staticmethod
    def _initial_requests(
        state: ServiceObligationStateMachine,
        *,
        vehicle_token: str,
    ) -> Tuple[InitialRequestState, ...]:
        rows: List[InitialRequestState] = []
        for request in sorted(state.requests.values(), key=lambda item: item.request_id):
            if request.assigned_vehicle != vehicle_token:
                continue
            passenger = state.passengers[request.passenger_id]
            if request.request_status in {RequestStatus.COMPLETED, RequestStatus.CANCELLED}:
                continue
            rows.append(
                InitialRequestState(
                    request_id=request.request_id,
                    passenger_id=request.passenger_id,
                    waiting_since_ts=int(passenger.created_ts),
                    status=request.request_status.value,
                )
            )
        return tuple(rows)

    def run_h4_fixture(
        self,
        *,
        fixture_id: str,
        state: ServiceObligationStateMachine,
        focal_agent_id: int,
        route_segment: Sequence[Mapping[str, Any]],
        decision_ts: int,
        materialize_reward_v2: bool = False,
    ) -> Dict[str, Any]:
        if len(route_segment) < 4:
            raise PV8B1OrchestratorError("fixture route segment requires at least four ordered occurrences")
        route_rows = [dict(row) for row in route_segment]
        route_id = str(route_rows[0]["route_id"])
        direction_id = str(route_rows[0]["direction_id"])
        if any(str(row["route_id"]) != route_id or str(row["direction_id"]) != direction_id for row in route_rows):
            raise PV8B1OrchestratorError("fixture route segment must stay within one route direction")

        agent_id = int(focal_agent_id)
        token = self.fixed_vehicle_bindings[agent_id]
        snapshot = self.capture_snapshot(
            state=state,
            cycle_index=int(decision_ts),
            decision_ts=int(decision_ts),
            occurrence_by_agent={agent_id: route_rows[0]},
            active_by_agent={agent_id: True},
        )
        actions = self.select_b1_actions(snapshot)
        action_name = actions[agent_id]
        if action_name not in {B1_ACTION_SERVE, B1_ACTION_HOLD}:
            raise PV8B1OrchestratorError("B1 selected an unauthorized action")

        serialized = serialize_global_k_mask_snapshot(snapshot)
        restored = deserialize_global_k_mask_snapshot(serialized)
        cloned = clone_global_k_mask_snapshot(restored)
        recomputed = self.lifecycle.recompute(restored)
        lifecycle_checks = {
            "serialize_restore_deterministic": restored.snapshot_hash == snapshot.snapshot_hash,
            "clone_deterministic": cloned.snapshot_hash == snapshot.snapshot_hash,
            "recompute_deterministic": recomputed.snapshot_hash == snapshot.snapshot_hash,
        }

        snapshot_rows = snapshot.to_payload()["agent_snapshots"]
        focal_snapshot = next(row for row in snapshot_rows if int(row["agent_id"]) == agent_id)
        collector = CausalOutcomeCollector()
        outcome_end = int(decision_ts) + H4_SECONDS
        for row in snapshot_rows:
            row_agent = int(row["agent_id"])
            active = bool(row["active_bus_mask"])
            collector.register_decision(
                DecisionOutcomeBinding(
                    decision_id=f"{fixture_id}:decision:{row_agent}",
                    decision_ts=int(decision_ts),
                    agent_id=row_agent,
                    vehicle_token=str(row["vehicle_token"]),
                    action_t=actions[row_agent] if active else None,
                    outcome_start_ts=int(decision_ts),
                    outcome_end_ts=outcome_end,
                    next_decision_ts=outcome_end,
                    terminal_ts=None,
                    active_bus=active,
                    observation_hash=canonical_hash({"agent_id": row_agent, "decision_ts": decision_ts}),
                    k_mask_hash=canonical_hash(row["k_action_mask"]),
                    initial_requests=self._initial_requests(state, vehicle_token=str(row["vehicle_token"])),
                    episode_id=fixture_id,
                    vehicle_slot_id=row_agent,
                    route_id=str(focal_snapshot.get("route_id") or route_id) if active else None,
                    direction_id=str(focal_snapshot.get("direction_id") or direction_id) if active else None,
                    occurrence_id=str(focal_snapshot.get("route_stop_occurrence_id") or route_rows[0]["route_stop_occurrence_id"]) if active else None,
                    local_decision_ts=int(decision_ts),
                )
            )

        vehicle = SimpleNamespace(
            agent_id=agent_id,
            route_key=(route_id, direction_id),
            route_id=route_id,
            direction_id=direction_id,
            position=0,
            target_position=None,
            current_edge_id=None,
            remaining_travel_seconds=0.0,
            remaining_dwell_seconds=0.0,
            consecutive_skip_count=0,
            vehicle_state="IN_SERVICE",
        )
        routes: MutableMapping[Tuple[str, str], List[Dict[str, Any]]] = {
            (route_id, direction_id): [
                {"stop_id": str(row["stop_id"]), "node_uid": f"STOP:{row['stop_id']}"}
                for row in route_rows
            ]
        }

        def stop_service(service_vehicle: Any, stop_row: Dict[str, Any]) -> StopServiceResult:
            event_ts = max(int(decision_ts) + 1, int(round(float(service_vehicle._service_event_time_seconds))))
            state.advance_to(event_ts)
            stop_id = str(stop_row["stop_id"])
            boarded = 0
            alighted = 0
            for request in sorted(list(state.requests.values()), key=lambda item: item.request_id):
                passenger = state.passengers[request.passenger_id]
                if (
                    request.request_status == RequestStatus.ASSIGNED
                    and request.assigned_vehicle == token
                    and request.pickup_stop == stop_id
                ):
                    state.passenger_boarded(
                        request_id=request.request_id,
                        passenger_id=passenger.passenger_id,
                        agent_id=agent_id,
                        vehicle_token=token,
                        stop_id=stop_id,
                        event_ts=event_ts,
                    )
                    boarded += 1
                if (
                    request.request_status == RequestStatus.BOARDED
                    and request.assigned_vehicle == token
                    and request.dropoff_stop == stop_id
                ):
                    state.passenger_alighted(
                        request_id=request.request_id,
                        passenger_id=passenger.passenger_id,
                        agent_id=agent_id,
                        vehicle_token=token,
                        stop_id=stop_id,
                        event_ts=event_ts,
                    )
                    state.request_completed(request_id=request.request_id, event_ts=event_ts)
                    alighted += 1
            return StopServiceResult(
                boardings=boarded,
                alightings=alighted,
                dwell_required=False,
                metadata={"typed_k4_service": True, "event_ts": event_ts},
            )

        engine_action = ENGINE_ACTION_SERVE_MOVE if action_name == B1_ACTION_SERVE else ENGINE_ACTION_HOLD
        trace = advance_vehicle_time_budget(
            vehicle=vehicle,
            routes=routes,
            delta_t_seconds=float(H4_SECONDS - 1),
            action=engine_action,
            stop_service=stop_service,
            config=TransitionConfig(edge_travel_seconds=30.0, dwell_seconds=0.0, allow_turnaround=False),
            run_id=fixture_id,
            service_day_id="RESEARCH_FIXTURE_ONLY",
            snapshot_id=int(decision_ts),
            snapshot_start_time_seconds=float(int(decision_ts) + 1),
            obligation_state_machine=state,
            decision_ts=int(decision_ts),
            vehicle_token=token,
        )
        state.advance_to(outcome_end)

        post_commit_events = [
            row for row in state.event_log
            if int(decision_ts) < int(row["event_ts"]) <= outcome_end
        ]
        mapped = map_k4_event_log(
            post_commit_events,
            agent_id=agent_id,
            vehicle_token=token,
            event_id_prefix=f"{fixture_id}:k4",
        )
        focal_decision_id = f"{fixture_id}:decision:{agent_id}"
        for event in mapped:
            collector.record_event(focal_decision_id, event)
        if trace.edges_traversed or trace.travel_seconds or trace.dwell_seconds:
            collector.record_event(
                focal_decision_id,
                RewardOutcomeEvent(
                    event_id=f"{fixture_id}:movement",
                    event_ts=int(decision_ts) + 1,
                    event_type=RewardOutcomeEventType.VEHICLE_MOVEMENT,
                    agent_id=agent_id,
                    vehicle_token=token,
                    metadata={
                        "distance_m": float(trace.edges_traversed * 100.0),
                        "travel_seconds": float(trace.travel_seconds),
                        "dwell_seconds": float(trace.dwell_seconds),
                    },
                ),
            )
        outcomes = {
            row_agent: collector.finalize(f"{fixture_id}:decision:{row_agent}")
            for row_agent in range(8)
        }
        focal_outcome = outcomes[agent_id]
        reward_v2_transition_input = None
        reward_v2_runtime_result = None
        reward_v2_runtime_error = None
        if materialize_reward_v2:
            reward_v2_transition_input = build_reward_v2_transition_input(
                focal_outcome,
                pickup_obligation_count=int(focal_outcome.get("boarding_completion_count", 0) > 0),
                dropoff_obligation_count=int(focal_outcome.get("alighting_completion_count", 0) > 0),
                approved_static_mandatory_obligation_count=0,
                completed_pickup_obligation_count=int(focal_outcome.get("boarding_completion_count", 0) > 0),
                completed_dropoff_obligation_count=int(focal_outcome.get("alighting_completion_count", 0) > 0),
                completed_static_mandatory_obligation_count=0,
            )
            try:
                reward_v2_runtime_result = compute_reward_v2(reward_v2_transition_input)
            except Exception as exc:
                reward_v2_runtime_error = {
                    "type": type(exc).__name__,
                    "code": getattr(exc, "code", None),
                    "message": str(exc),
                }
        decision_snapshot = focal_snapshot.get("decision_time_obligation_snapshot") or {}
        future_information_violation = bool(
            focal_outcome.get("future_information_written_to_observation_or_mask")
            or decision_snapshot.get("pickup_obligation")
            or decision_snapshot.get("dropoff_obligation")
            or decision_snapshot.get("onboard_destination_obligation")
        )
        return {
            "fixture_id": fixture_id,
            "snapshot_hash": snapshot.snapshot_hash,
            "snapshot_agent_count": len(snapshot_rows),
            "active_agent_count": sum(bool(row["active_bus_mask"]) for row in snapshot_rows),
            "inactive_agent_count": sum(not bool(row["active_bus_mask"]) for row in snapshot_rows),
            "inactive_learning_sample_count": sum(not bool(outcomes[index]["learning_sample"]) for index in range(8) if index != agent_id),
            "action_t": action_name,
            "conditional_skip_selected": False,
            "focal_action_mask": [bool(value) for value in focal_snapshot["k_action_mask"]],
            "active_all_false_mask_count": sum(
                bool(row["active_bus_mask"]) and not any(bool(value) for value in row["k_action_mask"])
                for row in snapshot_rows
            ),
            "occurrence_lookup_failure_count": sum(
                bool(row["active_bus_mask"]) and not bool(row["occurrence_lookup_integrity"])
                for row in snapshot_rows
            ),
            "identity_failure_count": sum(bool(row["identity_failure"]) for row in snapshot_rows),
            "future_information_violation": future_information_violation,
            "pending_future_transition_count": len(state.pending_transitions),
            "mapped_post_commit_event_count": len(mapped),
            "engine_event_count": len(trace.events),
            "engine_edges_traversed": int(trace.edges_traversed),
            "engine_boardings": int(trace.boardings),
            "engine_alightings": int(trace.alightings),
            "state_integrity": state.audit_integrity(),
            "lifecycle_checks": lifecycle_checks,
            "focal_outcome": focal_outcome,
            "reward_semantics_version": PV8_REWARD_SEMANTICS_VERSION,
            "reward_freeze_sha256": PV8_REWARD_V2_FREEZE_SHA256,
            "reward_v2_event_order": PV8_REWARD_V2_EVENT_ORDER,
            "reward_v2_transition_input": reward_v2_transition_input,
            "reward_v2_runtime_result": reward_v2_runtime_result,
            "reward_v2_runtime_error": reward_v2_runtime_error,
            "H240_active_reward_settlement": False,
            "H660_active_reward_settlement": False,
            "reward_value_materialized": False,
            "normalization_candidate_produced": False,
            "policy_execution_count": 0,
        }
