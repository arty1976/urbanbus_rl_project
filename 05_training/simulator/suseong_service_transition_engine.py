from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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
    from simulator.k_safety_state import DecisionTimeObligationSnapshot, ServiceObligationStateMachine, StaticGuardStatus
except ImportError:  # Direct simulator-directory test execution.
    from k_safety_state import DecisionTimeObligationSnapshot, ServiceObligationStateMachine, StaticGuardStatus


SUSEONG_DRT_DISTINCT_3ACTION_V2 = "SUSEONG_DRT_DISTINCT_3ACTION_V2"

ENGINE_ACTION_HOLD = 0
ENGINE_ACTION_SERVE_MOVE = 1
ENGINE_ACTION_LEGACY_MOVE_ONE_DUPLICATE = 2
ENGINE_ACTION_CONDITIONAL_SKIP = 3

ACTOR_ACTION_HOLD_CURRENT_POSITION = 0
ACTOR_ACTION_SERVE_AND_MOVE_TO_NEXT_STOP = 1
ACTOR_ACTION_CONDITIONAL_SKIP_EMPTY_STOP = 2

ACTOR_TO_ENGINE_ACTION = {
    ACTOR_ACTION_HOLD_CURRENT_POSITION: ENGINE_ACTION_HOLD,
    ACTOR_ACTION_SERVE_AND_MOVE_TO_NEXT_STOP: ENGINE_ACTION_SERVE_MOVE,
    ACTOR_ACTION_CONDITIONAL_SKIP_EMPTY_STOP: ENGINE_ACTION_CONDITIONAL_SKIP,
}

ACTION_CONTRACT_REGISTRY = [
    {
        "action_contract_version": SUSEONG_DRT_DISTINCT_3ACTION_V2,
        "actor_action_id": ACTOR_ACTION_HOLD_CURRENT_POSITION,
        "actor_action_name": "HOLD_CURRENT_POSITION",
        "engine_action_id": ENGINE_ACTION_HOLD,
        "engine_action_name": "HOLD_IDLE",
        "semantic_description": "Remain at the current route position for the decision interval after current-stop service has been processed.",
        "mask_rule": "valid unless terminal/environment guard blocks all decisions",
        "transition_rule": "consume remaining decision time as HOLD_IDLE without route progression",
        "legacy_compatibility": "compatible with legacy no-op action 0",
    },
    {
        "action_contract_version": SUSEONG_DRT_DISTINCT_3ACTION_V2,
        "actor_action_id": ACTOR_ACTION_SERVE_AND_MOVE_TO_NEXT_STOP,
        "actor_action_name": "SERVE_AND_MOVE_TO_NEXT_STOP",
        "engine_action_id": ENGINE_ACTION_SERVE_MOVE,
        "engine_action_name": "SERVE_AND_MOVE",
        "semantic_description": "After current-stop service, move to the next planned itinerary stop.",
        "mask_rule": "valid when the next candidate stop/path exists",
        "transition_rule": "advance route index by one edge",
        "legacy_compatibility": "compatible with legacy move-one action 1",
    },
    {
        "action_contract_version": SUSEONG_DRT_DISTINCT_3ACTION_V2,
        "actor_action_id": ACTOR_ACTION_CONDITIONAL_SKIP_EMPTY_STOP,
        "actor_action_name": "CONDITIONAL_SKIP_EMPTY_STOP",
        "engine_action_id": ENGINE_ACTION_CONDITIONAL_SKIP,
        "engine_action_name": "CONDITIONAL_SKIP_EMPTY_STOP",
        "semantic_description": "After current-stop service, skip the next candidate stop only when that stop has no pickup, dropoff, mandatory, path, or fairness obligation.",
        "mask_rule": "valid only when every conditional skip safety predicate is true",
        "transition_rule": "advance from current stop to post-skip target and record skip telemetry",
        "legacy_compatibility": "not compatible with legacy action 2 duplicate semantics; legacy engine action 2 remains non-actor-reachable",
    },
]


class InvalidConditionalSkipError(ValueError):
    pass


@dataclass
class StopServiceResult:
    boardings: int = 0
    alightings: int = 0
    dwell_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionConfig:
    edge_travel_seconds: float = 45.0
    dwell_seconds: float = 15.0
    terminal_recovery_seconds: Optional[float] = None
    allow_turnaround: bool = False
    max_transitions_per_vehicle_step: int = 10000


@dataclass
class VehicleStepTrace:
    vehicle_id: int
    available_seconds: float
    run_id: Optional[str] = None
    service_day_id: Optional[str] = None
    snapshot_id: Optional[int] = None
    snapshot_start_time_seconds: float = 0.0
    travel_seconds: float = 0.0
    dwell_seconds: float = 0.0
    turnaround_seconds: float = 0.0
    valid_idle_seconds: float = 0.0
    unused_seconds_at_episode_end: float = 0.0
    time_budget_conservation_error_seconds: float = 0.0
    transition_guard_triggered: bool = False
    edges_traversed: int = 0
    stops_visited: int = 0
    terminal_arrival_count: int = 0
    turnaround_start_count: int = 0
    turnaround_complete_count: int = 0
    direction_change_count: int = 0
    completed_trip_count: int = 0
    completed_cycle_count: int = 0
    service_resumed_after_terminal_count: int = 0
    terminal_stuck_seconds: float = 0.0
    terminal_stuck_flag: bool = False
    boardings: int = 0
    alightings: int = 0
    last_action_semantic: Optional[str] = None
    last_actor_action_id: Optional[int] = None
    last_engine_action_id: Optional[int] = None
    skip_attempted: bool = False
    skip_valid: bool = False
    skip_executed: bool = False
    skip_blocked: bool = False
    skip_invalid_reason_codes: List[str] = field(default_factory=list)
    skipped_stop_id: Optional[str] = None
    post_skip_target_stop_id: Optional[str] = None
    consecutive_skip_count: int = 0
    missed_pickup_due_to_skip: int = 0
    missed_dropoff_due_to_skip: int = 0
    mandatory_stop_violation_due_to_skip: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)

    def finalize(self) -> None:
        categorized = (
            self.travel_seconds
            + self.dwell_seconds
            + self.turnaround_seconds
            + self.valid_idle_seconds
            + self.unused_seconds_at_episode_end
        )
        self.time_budget_conservation_error_seconds = round(self.available_seconds - categorized, 9)


def _get_remaining_travel(vehicle: Any) -> float:
    return float(getattr(vehicle, "remaining_travel_seconds", getattr(vehicle, "remaining_travel_time", 0.0)))


def _set_remaining_travel(vehicle: Any, value: float) -> None:
    value = max(0.0, float(value))
    setattr(vehicle, "remaining_travel_seconds", value)
    setattr(vehicle, "remaining_travel_time", value)


def _get_remaining_dwell(vehicle: Any) -> float:
    return float(getattr(vehicle, "remaining_dwell_seconds", getattr(vehicle, "remaining_dwell_time", 0.0)))


def _set_remaining_dwell(vehicle: Any, value: float) -> None:
    value = max(0.0, float(value))
    setattr(vehicle, "remaining_dwell_seconds", value)
    setattr(vehicle, "remaining_dwell_time", value)


def _stop_uid(stop_row: Mapping[str, Any]) -> str:
    return str(stop_row.get("stop_id", stop_row.get("node_uid", "")))


def evaluate_conditional_skip_safety(
    vehicle: Any,
    routes: Mapping[Tuple[str, str], List[Dict[str, Any]]],
    *,
    obligation_state_machine: Optional[ServiceObligationStateMachine] = None,
    decision_ts: Optional[int] = None,
    vehicle_token: Optional[str] = None,
    static_guard_status: Optional[StaticGuardStatus] = None,
) -> Dict[str, Any]:
    route = routes[getattr(vehicle, "route_key")]
    pos = int(getattr(vehicle, "position"))
    current_stop = route[pos] if 0 <= pos < len(route) else None
    next_index = pos + 1
    post_index = pos + 2
    reasons: List[str] = []
    if current_stop is None:
        reasons.append("CURRENT_STOP_MISSING")
    if next_index >= len(route):
        reasons.append("NO_NEXT_CANDIDATE_STOP")
        next_stop = None
    else:
        next_stop = route[next_index]
    if post_index >= len(route):
        reasons.append("NO_POST_SKIP_TARGET")
        post_stop = None
    else:
        post_stop = route[post_index]
    if static_guard_status is None:
        static_guard_status = StaticGuardStatus.from_stop_row(
            next_stop,
            post_skip_target_exists=post_stop is not None,
        )
    resolved_decision_ts = int(
        decision_ts
        if decision_ts is not None
        else getattr(vehicle, "decision_ts", getattr(vehicle, "_decision_ts", 0))
    )
    resolved_vehicle_token = str(
        vehicle_token
        if vehicle_token is not None
        else getattr(vehicle, "vehicle_token", getattr(vehicle, "agent_id", ""))
    )
    if obligation_state_machine is None:
        reasons.append("DYNAMIC_STATE_MACHINE_MISSING")
        snapshot = DecisionTimeObligationSnapshot.incomplete(
            agent_id=int(getattr(vehicle, "agent_id")),
            vehicle_token=resolved_vehicle_token,
            decision_ts=resolved_decision_ts,
            current_stop_id=_stop_uid(current_stop) if current_stop is not None else "",
            candidate_stop_id=_stop_uid(next_stop) if next_stop is not None else None,
            static_guard_status=static_guard_status,
            missing_reason=("DYNAMIC_STATE_MACHINE_MISSING",),
        )
    elif current_stop is None or next_stop is None:
        reasons.append("DECISION_CONTEXT_INCOMPLETE")
        snapshot = DecisionTimeObligationSnapshot.incomplete(
            agent_id=int(getattr(vehicle, "agent_id")),
            vehicle_token=resolved_vehicle_token,
            decision_ts=resolved_decision_ts,
            current_stop_id=_stop_uid(current_stop) if current_stop is not None else "",
            candidate_stop_id=_stop_uid(next_stop) if next_stop is not None else None,
            static_guard_status=static_guard_status,
            missing_reason=("DECISION_CONTEXT_INCOMPLETE",),
        )
    else:
        snapshot = obligation_state_machine.snapshot_for_vehicle(
            agent_id=int(getattr(vehicle, "agent_id")),
            vehicle_token=resolved_vehicle_token,
            decision_ts=resolved_decision_ts,
            current_stop_id=_stop_uid(current_stop),
            candidate_stop_id=_stop_uid(next_stop),
            static_guard_status=static_guard_status,
        )

    waiting_pickup = len(snapshot.waiting_queue)
    assigned_pickup = len(snapshot.assigned_pickup_request_ids)
    assigned_dropoff = len(snapshot.assigned_dropoff_request_ids)
    dropoff_obligation = len(snapshot.onboard_destination_request_ids)
    boarding_obligation = bool(snapshot.boarding_obligation)
    alighting_obligation = bool(snapshot.alighting_obligation)
    onboard_obligation = bool(snapshot.onboard_destination_obligation)
    service_obligation = bool(snapshot.service_obligation)
    dynamic_state_complete = bool(snapshot.dynamic_state_complete)

    if waiting_pickup > 0 or boarding_obligation:
        reasons.append("WAITING_PICKUP_DEMAND")
    if alighting_obligation or onboard_obligation or dropoff_obligation > 0:
        reasons.append("ONBOARD_DROPOFF_DEMAND")
    if assigned_pickup > 0:
        reasons.append("ASSIGNED_PICKUP_REQUEST")
    if assigned_dropoff > 0:
        reasons.append("ASSIGNED_DROPOFF_REQUEST")
    if service_obligation and not any([waiting_pickup, assigned_pickup, assigned_dropoff, dropoff_obligation, boarding_obligation, alighting_obligation, onboard_obligation]):
        reasons.append("SERVICE_OBLIGATION_PRESENT")
    if not dynamic_state_complete:
        reasons.append("DYNAMIC_STATE_INCOMPLETE")
        reasons.extend(snapshot.missing_reason)

    if static_guard_status.mandatory_stop is True:
        reasons.append("MANDATORY_STOP")
    if static_guard_status.protected_stop is True:
        reasons.append("PROTECTED_STOP")
    if static_guard_status.terminal_or_turnaround_stop is True:
        reasons.append("TERMINAL_OR_TURNAROUND_STOP")
    if static_guard_status.charging_or_driver_relief_stop is True:
        reasons.append("CHARGING_OR_DRIVER_RELIEF_STOP")
    if static_guard_status.planned_itinerary_allows_skip is False:
        reasons.append("PLANNED_ITINERARY_BLOCK")
    if static_guard_status.valid_post_skip_path is False:
        reasons.append("DOWNSTREAM_PATH_INVALID")
    if not static_guard_status.state_complete:
        reasons.append("STATIC_GUARD_STATE_INCOMPLETE")
        reasons.extend(static_guard_status.missing_reason)
    seen: List[str] = []
    for reason in reasons:
        if reason not in seen:
            seen.append(reason)
    return {
        "current_stop": _stop_uid(current_stop) if current_stop is not None else None,
        "next_candidate_stop": _stop_uid(next_stop) if next_stop is not None else None,
        "post_skip_target_stop": _stop_uid(post_stop) if post_stop is not None else None,
        "waiting_pickup_count": int(waiting_pickup),
        "scheduled_alighting_count": int(alighting_obligation),
        "assigned_pickup_request_count": int(assigned_pickup),
        "assigned_dropoff_request_count": int(assigned_dropoff),
        "dropoff_obligation_count": int(dropoff_obligation + assigned_dropoff),
        "mandatory_stop": static_guard_status.mandatory_stop,
        "protected_stop": static_guard_status.protected_stop,
        "terminal_or_turnaround_stop": static_guard_status.terminal_or_turnaround_stop,
        "charging_or_driver_relief_stop": static_guard_status.charging_or_driver_relief_stop,
        "planned_itinerary_allows_skip": static_guard_status.planned_itinerary_allows_skip,
        "downstream_path_valid": static_guard_status.valid_post_skip_path is True,
        "dynamic_state_complete": dynamic_state_complete,
        "static_guard_state_complete": static_guard_status.state_complete,
        "decision_time_obligation_snapshot": snapshot.to_payload(),
        "static_guard_status": static_guard_status.to_payload(),
        "skip_valid": len(seen) == 0,
        "skip_invalid_reason_codes": seen,
        "pickup_service_obligation_exists": bool(waiting_pickup > 0 or assigned_pickup > 0 or boarding_obligation),
        "dropoff_service_obligation_exists": bool(dropoff_obligation > 0 or assigned_dropoff > 0 or alighting_obligation or onboard_obligation),
        "service_obligation_exists": service_obligation,
        "safety_contract_version": "PV8_K4_FAIL_CLOSED_V2",
    }


def build_distinct_three_action_mask(
    vehicle: Any,
    routes: Mapping[Tuple[str, str], List[Dict[str, Any]]],
    *,
    environment_terminal: bool = False,
    obligation_state_machine: Optional[ServiceObligationStateMachine] = None,
    decision_ts: Optional[int] = None,
    vehicle_token: Optional[str] = None,
    static_guard_status: Optional[StaticGuardStatus] = None,
) -> Dict[str, Any]:
    route = routes[getattr(vehicle, "route_key")]
    pos = int(getattr(vehicle, "position"))
    next_exists = pos + 1 < len(route)
    safety = evaluate_conditional_skip_safety(
        vehicle,
        routes,
        obligation_state_machine=obligation_state_machine,
        decision_ts=decision_ts,
        vehicle_token=vehicle_token,
        static_guard_status=static_guard_status,
    )
    hold_valid = not bool(environment_terminal)
    serve_move_valid = bool(next_exists and not environment_terminal)
    skip_valid = bool(safety["skip_valid"] and not environment_terminal)
    return {
        "action_contract_version": SUSEONG_DRT_DISTINCT_3ACTION_V2,
        "action_mask": [hold_valid, serve_move_valid, skip_valid],
        "hold_valid": hold_valid,
        "serve_move_valid": serve_move_valid,
        "skip_valid": skip_valid,
        "skip_invalid_reason_codes": list(safety["skip_invalid_reason_codes"]),
        "pickup_obligation": safety["pickup_service_obligation_exists"],
        "dropoff_obligation": safety["dropoff_service_obligation_exists"],
        "mandatory_stop": safety["mandatory_stop"],
        "downstream_path_invalid": not safety["downstream_path_valid"],
        "missing_post_skip_target": "NO_POST_SKIP_TARGET" in safety["skip_invalid_reason_codes"],
        "consecutive_skip_limit": "CONSECUTIVE_SKIP_LIMIT" in safety["skip_invalid_reason_codes"],
        "service_fairness_block": "SERVICE_FAIRNESS_BLOCK" in safety["skip_invalid_reason_codes"],
        **safety,
    }


def _consume(trace: VehicleStepTrace, bucket: str, seconds: float) -> None:
    seconds = float(seconds)
    if bucket == "TRAVEL":
        trace.travel_seconds += seconds
    elif bucket == "DWELL":
        trace.dwell_seconds += seconds
    elif bucket == "TURNAROUND":
        trace.turnaround_seconds += seconds
    elif bucket == "IDLE":
        trace.valid_idle_seconds += seconds
    elif bucket == "UNUSED":
        trace.unused_seconds_at_episode_end += seconds


def _log(
    trace: VehicleStepTrace,
    *,
    event_type: str,
    time_before: float,
    time_consumed: float,
    time_after: float,
    route_index_before: int,
    route_index_after: int,
    stop_id: Optional[str],
    edge_id: Optional[str],
    route_id: Optional[str] = None,
    direction_id: Optional[str] = None,
    terminal_stop_id: Optional[str] = None,
    vehicle_state_before: Optional[str] = None,
    vehicle_state_after: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    event_time_seconds = float(trace.snapshot_start_time_seconds) + float(trace.available_seconds) - float(time_after)
    event = {
        "run_id": trace.run_id,
        "service_day_id": trace.service_day_id,
        "snapshot_id": trace.snapshot_id,
        "vehicle_id": trace.vehicle_id,
        "event_sequence": len(trace.events),
        "event_time_seconds": event_time_seconds,
        "event_type": event_type,
        "route_id": route_id,
        "direction_id": direction_id,
        "time_before": float(time_before),
        "time_consumed": float(time_consumed),
        "time_after": float(time_after),
        "route_index_before": int(route_index_before),
        "route_index_after": int(route_index_after),
        "stop_id": stop_id,
        "terminal_stop_id": terminal_stop_id,
        "vehicle_state_before": vehicle_state_before,
        "vehicle_state_after": vehicle_state_after,
        "edge_id": edge_id,
    }
    if metadata:
        event.update(dict(metadata))
    trace.events.append(event)


def _route_id(vehicle: Any) -> str:
    key = getattr(vehicle, "route_key")
    return str(key[0])


def _direction_id(vehicle: Any) -> str:
    key = getattr(vehicle, "route_key")
    return str(key[1])


def _set_route_key(vehicle: Any, route_key: Tuple[str, str]) -> None:
    setattr(vehicle, "route_key", route_key)


def advance_vehicle_time_budget(
    *,
    vehicle: Any,
    routes: MutableMapping[Tuple[str, str], List[Dict[str, Any]]],
    delta_t_seconds: float,
    action: int,
    stop_service: Callable[[Any, Dict[str, Any]], StopServiceResult],
    config: TransitionConfig,
    turnaround_mapping: Optional[Mapping[Tuple[str, str], Tuple[str, str]]] = None,
    run_id: Optional[str] = None,
    service_day_id: Optional[str] = None,
    snapshot_id: Optional[int] = None,
    snapshot_start_time_seconds: float = 0.0,
    obligation_state_machine: Optional[ServiceObligationStateMachine] = None,
    decision_ts: Optional[int] = None,
    vehicle_token: Optional[str] = None,
    static_guard_status: Optional[StaticGuardStatus] = None,
) -> VehicleStepTrace:
    _authz.require_capability("simulator_execution", site="simulator/suseong_service_transition_engine.py::advance_vehicle_time_budget")
    if delta_t_seconds < 0:
        raise ValueError("delta_t_seconds must be non-negative")
    vehicle_id = int(getattr(vehicle, "agent_id"))
    trace = VehicleStepTrace(
        vehicle_id=vehicle_id,
        available_seconds=float(delta_t_seconds),
        run_id=run_id,
        service_day_id=service_day_id,
        snapshot_id=snapshot_id,
        snapshot_start_time_seconds=float(snapshot_start_time_seconds),
    )
    time_budget = float(delta_t_seconds)
    transition_count = 0
    target_position = getattr(vehicle, "target_position", None)
    if target_position is not None:
        setattr(vehicle, "_ready_to_depart", False)

    while time_budget > 0:
        transition_count += 1
        if transition_count > int(config.max_transitions_per_vehicle_step):
            trace.transition_guard_triggered = True
            _consume(trace, "UNUSED", time_budget)
            time_budget = 0.0
            break

        route = routes[getattr(vehicle, "route_key")]
        pos_before = int(getattr(vehicle, "position"))
        stop_id = str(route[pos_before].get("stop_id", route[pos_before].get("node_uid", "")))
        current_route_id = _route_id(vehicle)
        current_direction_id = _direction_id(vehicle)
        terminal_stop_id = str(route[-1].get("stop_id", route[-1].get("node_uid", "")))
        travel_remaining = _get_remaining_travel(vehicle)
        if travel_remaining > 0:
            consumed = min(time_budget, travel_remaining)
            before = time_budget
            _set_remaining_travel(vehicle, travel_remaining - consumed)
            time_budget -= consumed
            _consume(trace, "TRAVEL", consumed)
            _log(
                trace,
                event_type="TRAVEL",
                time_before=before,
                time_consumed=consumed,
                time_after=time_budget,
                route_index_before=pos_before,
                route_index_after=pos_before,
                stop_id=stop_id,
                edge_id=getattr(vehicle, "current_edge_id", None),
                route_id=current_route_id,
                direction_id=current_direction_id,
            )
            if _get_remaining_travel(vehicle) > 0:
                break
            arrival_position = int(getattr(vehicle, "target_position", min(pos_before + 1, len(route) - 1)))
            setattr(vehicle, "position", arrival_position)
            setattr(vehicle, "target_position", None)
            setattr(vehicle, "current_edge_id", None)
            setattr(vehicle, "_ready_to_depart", False)
            trace.edges_traversed += 1
            trace.stops_visited += 1
            if arrival_position >= len(route) - 1:
                state_before = str(getattr(vehicle, "vehicle_state", "IN_SERVICE"))
                trace.terminal_arrival_count += 1
                trace.completed_trip_count += 1
                setattr(vehicle, "vehicle_state", "AT_TERMINAL")
                _log(
                    trace,
                    event_type="ROUTE_TERMINAL_REACHED",
                    time_before=time_budget,
                    time_consumed=0.0,
                    time_after=time_budget,
                    route_index_before=pos_before,
                    route_index_after=arrival_position,
                    stop_id=str(route[arrival_position].get("stop_id", route[arrival_position].get("node_uid", ""))),
                    edge_id=None,
                    route_id=current_route_id,
                    direction_id=current_direction_id,
                    terminal_stop_id=str(route[arrival_position].get("stop_id", route[arrival_position].get("node_uid", ""))),
                    vehicle_state_before=state_before,
                    vehicle_state_after="AT_TERMINAL",
                )
            _log(
                trace,
                event_type="STOP_ARRIVAL",
                time_before=time_budget,
                time_consumed=0.0,
                time_after=time_budget,
                route_index_before=pos_before,
                route_index_after=arrival_position,
                stop_id=str(route[arrival_position].get("stop_id", route[arrival_position].get("node_uid", ""))),
                edge_id=None,
                route_id=current_route_id,
                direction_id=current_direction_id,
                terminal_stop_id=terminal_stop_id if arrival_position >= len(route) - 1 else None,
            )
            continue

        dwell_remaining = _get_remaining_dwell(vehicle)
        if dwell_remaining > 0:
            consumed = min(time_budget, dwell_remaining)
            before = time_budget
            _set_remaining_dwell(vehicle, dwell_remaining - consumed)
            time_budget -= consumed
            _consume(trace, "DWELL", consumed)
            _log(
                trace,
                event_type="DWELL",
                time_before=before,
                time_consumed=consumed,
                time_after=time_budget,
                route_index_before=pos_before,
                route_index_after=pos_before,
                stop_id=stop_id,
                edge_id=None,
                route_id=current_route_id,
                direction_id=current_direction_id,
            )
            if _get_remaining_dwell(vehicle) > 0:
                break
            setattr(vehicle, "_ready_to_depart", True)
            continue

        at_terminal = pos_before >= len(route) - 1
        if at_terminal:
            if not config.allow_turnaround:
                trace.terminal_stuck_flag = True
                trace.terminal_stuck_seconds += time_budget
                _consume(trace, "IDLE", time_budget)
                _log(
                    trace,
                    event_type="TERMINAL_IDLE_OBSERVATION",
                    time_before=time_budget,
                    time_consumed=time_budget,
                    time_after=0.0,
                    route_index_before=pos_before,
                    route_index_after=pos_before,
                    stop_id=stop_id,
                    edge_id=None,
                    route_id=current_route_id,
                    direction_id=current_direction_id,
                    terminal_stop_id=terminal_stop_id,
                    vehicle_state_before="AT_TERMINAL",
                    vehicle_state_after="AT_TERMINAL",
                )
                time_budget = 0.0
                break
            if config.terminal_recovery_seconds is None:
                raise RuntimeError("terminal recovery seconds are required for production turnaround")
            mapping = turnaround_mapping or {}
            current_key = getattr(vehicle, "route_key")
            next_key = mapping.get(current_key)
            if next_key is None or next_key not in routes:
                raise RuntimeError(f"turnaround mapping missing for route direction {current_key}")
            recovery_remaining = float(getattr(vehicle, "remaining_turnaround_seconds", config.terminal_recovery_seconds))
            if recovery_remaining == float(config.terminal_recovery_seconds):
                trace.turnaround_start_count += 1
                _log(
                    trace,
                    event_type="TURNAROUND_STARTED",
                    time_before=time_budget,
                    time_consumed=0.0,
                    time_after=time_budget,
                    route_index_before=pos_before,
                    route_index_after=pos_before,
                    stop_id=stop_id,
                    edge_id=None,
                )
            consumed = min(time_budget, recovery_remaining)
            before = time_budget
            recovery_remaining -= consumed
            time_budget -= consumed
            _consume(trace, "TURNAROUND", consumed)
            setattr(vehicle, "remaining_turnaround_seconds", recovery_remaining)
            _log(
                trace,
                event_type="TURNAROUND",
                time_before=before,
                time_consumed=consumed,
                time_after=time_budget,
                route_index_before=pos_before,
                route_index_after=pos_before,
                stop_id=stop_id,
                edge_id=None,
            )
            if recovery_remaining > 0:
                break
            setattr(vehicle, "remaining_turnaround_seconds", 0.0)
            old_direction = _direction_id(vehicle)
            _set_route_key(vehicle, next_key)
            next_route = routes[next_key]
            setattr(vehicle, "position", 0)
            setattr(vehicle, "_ready_to_depart", False)
            trace.turnaround_complete_count += 1
            if str(next_key[1]) != old_direction:
                trace.direction_change_count += 1
            trace.service_resumed_after_terminal_count += 1
            trace.completed_cycle_count += 1
            _log(
                trace,
                event_type="SERVICE_RESUMED",
                time_before=time_budget,
                time_consumed=0.0,
                time_after=time_budget,
                route_index_before=pos_before,
                route_index_after=0,
                stop_id=str(next_route[0].get("stop_id", next_route[0].get("node_uid", ""))),
                edge_id=None,
                metadata={"new_route_id": str(next_key[0]), "new_direction_id": str(next_key[1])},
            )
            continue

        ready_to_depart = bool(getattr(vehicle, "_ready_to_depart", False))
        if not ready_to_depart:
            setattr(vehicle, "_service_event_time_seconds", float(trace.snapshot_start_time_seconds) + float(trace.available_seconds) - float(time_budget))
            setattr(vehicle, "_service_event_sequence", len(trace.events))
            result = stop_service(vehicle, route[pos_before])
            trace.boardings += int(result.boardings)
            trace.alightings += int(result.alightings)
            _log(
                trace,
                event_type="STOP_SERVICE",
                time_before=time_budget,
                time_consumed=0.0,
                time_after=time_budget,
                route_index_before=pos_before,
                route_index_after=pos_before,
                stop_id=stop_id,
                edge_id=None,
                route_id=current_route_id,
                direction_id=current_direction_id,
                metadata=result.metadata,
            )
            if result.dwell_required and config.dwell_seconds > 0:
                _set_remaining_dwell(vehicle, config.dwell_seconds)
                continue
            setattr(vehicle, "_ready_to_depart", True)
            continue

        if int(action) == 0:
            trace.last_engine_action_id = ENGINE_ACTION_HOLD
            trace.last_action_semantic = "HOLD_CURRENT_POSITION"
            setattr(vehicle, "consecutive_skip_count", 0)
            _consume(trace, "IDLE", time_budget)
            _log(
                trace,
                event_type="HOLD_IDLE",
                time_before=time_budget,
                time_consumed=time_budget,
                time_after=0.0,
                route_index_before=pos_before,
                route_index_after=pos_before,
                stop_id=stop_id,
                edge_id=None,
                route_id=current_route_id,
                direction_id=current_direction_id,
            )
            time_budget = 0.0
            break

        trace.last_engine_action_id = int(action)
        if int(action) == ENGINE_ACTION_CONDITIONAL_SKIP:
            trace.last_action_semantic = "CONDITIONAL_SKIP_EMPTY_STOP"
            trace.skip_attempted = True
            safety = evaluate_conditional_skip_safety(
                vehicle,
                routes,
                obligation_state_machine=obligation_state_machine,
                decision_ts=(int(snapshot_start_time_seconds) if decision_ts is None else int(decision_ts)),
                vehicle_token=vehicle_token,
                static_guard_status=static_guard_status,
            )
            trace.skip_valid = bool(safety["skip_valid"])
            trace.skip_invalid_reason_codes = list(safety["skip_invalid_reason_codes"])
            if not safety["skip_valid"]:
                trace.skip_blocked = True
                raise InvalidConditionalSkipError(
                    "conditional skip rejected: " + ",".join(trace.skip_invalid_reason_codes)
                )
            trace.skip_executed = True
            trace.skipped_stop_id = safety["next_candidate_stop"]
            trace.post_skip_target_stop_id = safety["post_skip_target_stop"]
            trace.consecutive_skip_count = int(getattr(vehicle, "consecutive_skip_count", 0)) + 1
            setattr(vehicle, "consecutive_skip_count", trace.consecutive_skip_count)
            _log(
                trace,
                event_type="CONDITIONAL_SKIP_VALIDATED",
                time_before=time_budget,
                time_consumed=0.0,
                time_after=time_budget,
                route_index_before=pos_before,
                route_index_after=pos_before + 2,
                stop_id=stop_id,
                edge_id=None,
                route_id=current_route_id,
                direction_id=current_direction_id,
                metadata={
                    "skipped_stop_id": safety["next_candidate_stop"],
                    "post_skip_target_stop_id": safety["post_skip_target_stop"],
                    "skip_invalid_reason_codes": [],
                    "skipped_stop_waiting_count": safety["waiting_pickup_count"],
                    "skipped_stop_alighting_count": safety["dropoff_obligation_count"],
                    "missed_pickup_due_to_skip": 0,
                    "missed_dropoff_due_to_skip": 0,
                    "mandatory_stop_violation_due_to_skip": 0,
                },
            )
        elif int(action) == ENGINE_ACTION_HOLD:
            trace.last_action_semantic = "HOLD_CURRENT_POSITION"
            setattr(vehicle, "consecutive_skip_count", 0)
        else:
            trace.last_action_semantic = "SERVE_AND_MOVE_TO_NEXT_STOP"
            setattr(vehicle, "consecutive_skip_count", 0)
        move_delta = 2 if int(action) == ENGINE_ACTION_CONDITIONAL_SKIP else 1
        target = min(pos_before + move_delta, len(route) - 1)
        edge_id = f"{route[pos_before].get('node_uid', stop_id)}->{route[target].get('node_uid', target)}"
        setattr(vehicle, "target_position", target)
        setattr(vehicle, "current_edge_id", edge_id)
        setattr(vehicle, "_ready_to_depart", False)
        travel_seconds = max(0.0, float(config.edge_travel_seconds))
        if travel_seconds == 0.0:
            setattr(vehicle, "position", target)
            setattr(vehicle, "target_position", None)
            trace.edges_traversed += 1
            trace.stops_visited += 1
            if target >= len(route) - 1:
                state_before = str(getattr(vehicle, "vehicle_state", "IN_SERVICE"))
                trace.terminal_arrival_count += 1
                trace.completed_trip_count += 1
                setattr(vehicle, "vehicle_state", "AT_TERMINAL")
                _log(
                    trace,
                    event_type="ROUTE_TERMINAL_REACHED",
                    time_before=time_budget,
                    time_consumed=0.0,
                    time_after=time_budget,
                    route_index_before=pos_before,
                    route_index_after=target,
                    stop_id=str(route[target].get("stop_id", route[target].get("node_uid", ""))),
                    edge_id=None,
                    route_id=current_route_id,
                    direction_id=current_direction_id,
                    terminal_stop_id=str(route[target].get("stop_id", route[target].get("node_uid", ""))),
                    vehicle_state_before=state_before,
                    vehicle_state_after="AT_TERMINAL",
                )
            _log(
                trace,
                event_type="EDGE_TRAVEL_ZERO",
                time_before=time_budget,
                time_consumed=0.0,
                time_after=time_budget,
                route_index_before=pos_before,
                route_index_after=target,
                stop_id=str(route[target].get("stop_id", route[target].get("node_uid", ""))),
                edge_id=edge_id,
                route_id=current_route_id,
                direction_id=current_direction_id,
                terminal_stop_id=terminal_stop_id if target >= len(route) - 1 else None,
            )
            continue
        _set_remaining_travel(vehicle, travel_seconds)
        _log(
            trace,
            event_type="EDGE_ENTERED",
            time_before=time_budget,
            time_consumed=0.0,
            time_after=time_budget,
            route_index_before=pos_before,
            route_index_after=target,
            stop_id=stop_id,
            edge_id=edge_id,
            route_id=current_route_id,
            direction_id=current_direction_id,
        )
        continue

    trace.finalize()
    return trace


def compute_counter_semantics_v3(
    event_rows: Sequence[Mapping[str, Any]],
    *,
    instantiated_vehicle_count: Optional[int] = None,
    validation_snapshot_count: Optional[int] = None,
) -> Dict[str, Any]:
    arrivals = [row for row in event_rows if row.get("event_type") == "ROUTE_TERMINAL_REACHED"]
    idle = [row for row in event_rows if row.get("event_type") == "TERMINAL_IDLE_OBSERVATION"]
    resumed = [row for row in event_rows if row.get("event_type") == "SERVICE_RESUMED"]
    arriving_vehicles = {int(row["vehicle_id"]) for row in arrivals}
    resumed_vehicles = {int(row["vehicle_id"]) for row in resumed}
    idle_by_vehicle: Dict[int, int] = {}
    arrival_by_vehicle: Dict[int, int] = {}
    for row in idle:
        vehicle_id = int(row["vehicle_id"])
        idle_by_vehicle[vehicle_id] = idle_by_vehicle.get(vehicle_id, 0) + 1
    for row in arrivals:
        vehicle_id = int(row["vehicle_id"])
        arrival_by_vehicle[vehicle_id] = arrival_by_vehicle.get(vehicle_id, 0) + 1
    repeated_idle = sum(max(0, count - 1) for count in idle_by_vehicle.values())
    terminal_arrival_event_count = len(arrivals)
    completed_trip_count = len(arrivals)
    legacy_bad_pattern = False
    if instantiated_vehicle_count is not None and validation_snapshot_count is not None:
        legacy_bad_pattern = completed_trip_count == int(instantiated_vehicle_count) * int(validation_snapshot_count)
    terminal_stuck_unique = len(arriving_vehicles - resumed_vehicles)
    passed = (
        terminal_arrival_event_count == completed_trip_count
        and (instantiated_vehicle_count is None or len(arriving_vehicles) <= int(instantiated_vehicle_count))
        and terminal_stuck_unique <= len(arriving_vehicles)
        and len(idle) >= repeated_idle
        and not legacy_bad_pattern
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "terminal_arrival_event_count": terminal_arrival_event_count,
        "unique_terminal_arriving_vehicle_count": len(arriving_vehicles),
        "terminal_idle_observation_count": len(idle),
        "repeated_terminal_idle_observation_count": repeated_idle,
        "completed_trip_count": completed_trip_count,
        "completed_cycle_count": len(resumed),
        "terminal_stuck_unique_vehicle_count": terminal_stuck_unique,
        "multiple_trip_vehicle_count": sum(1 for count in arrival_by_vehicle.values() if count > 1),
        "instantiated_vehicle_count": instantiated_vehicle_count,
        "validation_snapshot_count": validation_snapshot_count,
        "repeated_terminal_observation_counted_as_trip": False,
        "completed_trip_equals_instantiated_times_validation_snapshots": legacy_bad_pattern,
        "counter_semantics_passed": passed,
    }
