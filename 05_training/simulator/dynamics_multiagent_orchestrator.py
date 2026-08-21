from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from simulator import suseong_service_transition_engine as engine
from simulator.dynamics_event_trace import DynamicsEvent, DynamicsEventType, adapt_stop_service_result, event_trace_hash
from simulator.dynamics_replay_contract import ReplayFrame, canonical_hash
from simulator.dynamics_state_snapshot import DynamicsStateSnapshot, hash_dynamics_state

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


ORCHESTRATION_SEMANTICS = "DETERMINISTIC_GLOBAL_STEP_WITH_CANONICAL_TIEBREAK"
DECISION_INTERVAL_SECONDS = 60
THIRTY_MINUTE_TOTAL_STEPS = 30


class UnsupportedDynamicsActionError(ValueError):
    pass


class MissingBranchExecutionContextError(ValueError):
    pass


class MissingEvaluationExecutionContextError(ValueError):
    pass


class MissingExecutionInstanceRegistryError(ValueError):
    pass


class DuplicateExecutionInstanceIdError(ValueError):
    pass


class AmbiguousServiceCompletionScopeError(ValueError):
    pass


class MissingServiceLegIdentityError(ValueError):
    pass


class DynamicsBranchAction(str, Enum):
    HOLD_CURRENT_POSITION = "HOLD_CURRENT_POSITION"
    SERVE_AND_MOVE_TO_NEXT_STOP = "SERVE_AND_MOVE_TO_NEXT_STOP"
    CONDITIONAL_SKIP_EMPTY_STOP = "CONDITIONAL_SKIP_EMPTY_STOP"


ACTION_TO_ENGINE = {
    DynamicsBranchAction.HOLD_CURRENT_POSITION: engine.ENGINE_ACTION_HOLD,
    DynamicsBranchAction.SERVE_AND_MOVE_TO_NEXT_STOP: engine.ENGINE_ACTION_SERVE_MOVE,
    DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP: engine.ENGINE_ACTION_CONDITIONAL_SKIP,
}


def adapt_branch_action(action: DynamicsBranchAction) -> int:
    if not isinstance(action, DynamicsBranchAction):
        raise UnsupportedDynamicsActionError(f"unsupported dynamics action: {action!r}")
    mapped = ACTION_TO_ENGINE[action]
    if mapped == engine.ENGINE_ACTION_LEGACY_MOVE_ONE_DUPLICATE:
        raise UnsupportedDynamicsActionError("legacy engine action 2 is not reachable through dynamics branch adapter")
    return int(mapped)


def reject_engine_action(action: int) -> int:
    if int(action) == engine.ENGINE_ACTION_LEGACY_MOVE_ONE_DUPLICATE:
        raise UnsupportedDynamicsActionError("legacy engine action 2 is not accepted by ER1 dynamics adapter")
    if int(action) not in {engine.ENGINE_ACTION_HOLD, engine.ENGINE_ACTION_SERVE_MOVE, engine.ENGINE_ACTION_CONDITIONAL_SKIP}:
        raise UnsupportedDynamicsActionError(f"unsupported engine action: {action}")
    return int(action)


def evaluate_action_mask(vehicle: Any, routes: Mapping[Any, Any], *, environment_terminal: bool = False) -> Dict[str, Any]:
    return engine.build_distinct_three_action_mask(vehicle, routes, environment_terminal=environment_terminal)


def evaluate_skip_safety(vehicle: Any, routes: Mapping[Any, Any]) -> Dict[str, Any]:
    return engine.evaluate_conditional_skip_safety(vehicle, routes)


def canonical_agent_order(agent_ids: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted(int(agent_id) for agent_id in agent_ids))


def action_vector_hash(action_by_agent: Mapping[int, DynamicsBranchAction]) -> str:
    payload = {str(agent_id): action.value for agent_id, action in sorted(action_by_agent.items(), key=lambda item: int(item[0]))}
    return canonical_hash(payload)


@dataclass(frozen=True)
class GlobalStepTrace:
    step_index: int
    elapsed_seconds_before: int
    elapsed_seconds_after: int
    state_hash_before: str
    state_hash_after: str
    action_vector_hash: str
    replay_frame_hash: str
    event_trace_hash: str
    canonical_agent_order: Tuple[int, ...]
    orchestration_semantics: str = ORCHESTRATION_SEMANTICS

    def to_payload(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "elapsed_seconds_before": self.elapsed_seconds_before,
            "elapsed_seconds_after": self.elapsed_seconds_after,
            "state_hash_before": self.state_hash_before,
            "state_hash_after": self.state_hash_after,
            "action_vector_hash": self.action_vector_hash,
            "replay_frame_hash": self.replay_frame_hash,
            "event_trace_hash": self.event_trace_hash,
            "canonical_agent_order": list(self.canonical_agent_order),
            "orchestration_semantics": self.orchestration_semantics,
        }


@dataclass(frozen=True)
class BranchExecutionContext:
    run_id: str
    fixture_id: str
    logical_branch_name: str
    execution_instance_id: str
    initial_state_hash: str
    replay_input_hash: str
    target_agent_id: int
    pulse_action: str
    caller_run_id: Optional[str] = None
    invocation_sequence: Optional[int] = None
    created_by: str = "caller"

    def logical_payload(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "fixture_id": self.fixture_id,
            "logical_branch_name": self.logical_branch_name,
            "initial_state_hash": self.initial_state_hash,
            "replay_input_hash": self.replay_input_hash,
            "target_agent_id": int(self.target_agent_id),
            "pulse_action": self.pulse_action,
        }

    @property
    def logical_branch_id(self) -> str:
        return canonical_hash(self.logical_payload())

    def to_payload(self) -> Dict[str, Any]:
        return {
            **self.logical_payload(),
            "execution_instance_id": self.execution_instance_id,
            "logical_branch_id": self.logical_branch_id,
            "caller_run_id": self.caller_run_id,
            "invocation_sequence": self.invocation_sequence,
            "created_by": self.created_by,
        }


class ServiceUnitScope(str, Enum):
    SERVICE_LEG = "SERVICE_LEG"
    REQUEST = "REQUEST"


@dataclass(frozen=True)
class CanonicalServiceUnitIdentity:
    scope: ServiceUnitScope
    request_id: str
    service_leg_id: Optional[str] = None

    @property
    def canonical_key(self) -> str:
        if self.scope == ServiceUnitScope.SERVICE_LEG:
            if not self.request_id or not self.service_leg_id:
                raise MissingServiceLegIdentityError("SERVICE_LEG identity requires request_id and service_leg_id")
            return f"service_leg:{self.request_id}:{self.service_leg_id}"
        if not self.request_id:
            raise ValueError("REQUEST identity requires request_id")
        return f"request:{self.request_id}"

    def to_payload(self) -> Dict[str, Any]:
        return {
            "scope": self.scope.value,
            "request_id": self.request_id,
            "service_leg_id": self.service_leg_id,
            "canonical_key": self.canonical_key,
        }


@dataclass
class ExecutionInstanceRegistry:
    records: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    registered_at_sequence: int = 0

    def register(self, branch_context: BranchExecutionContext, *, evaluation_run_id: Optional[str] = None) -> Dict[str, Any]:
        execution_instance_id = str(branch_context.execution_instance_id or "")
        if not execution_instance_id:
            raise MissingBranchExecutionContextError("execution_instance_id is required in BranchExecutionContext")
        if execution_instance_id in self.records:
            raise DuplicateExecutionInstanceIdError(f"duplicate execution_instance_id in execution scope: {execution_instance_id}")
        self.registered_at_sequence += 1
        record = {
            "execution_instance_id": execution_instance_id,
            "evaluation_run_id": evaluation_run_id or branch_context.caller_run_id or branch_context.run_id,
            "logical_branch_id": branch_context.logical_branch_id,
            "caller_run_id": branch_context.caller_run_id or branch_context.run_id,
            "invocation_sequence": branch_context.invocation_sequence,
            "created_by": branch_context.created_by,
            "registered_at_sequence": self.registered_at_sequence,
        }
        self.records[execution_instance_id] = record
        return dict(record)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "registered_execution_instance_count": len(self.records),
            "records": [self.records[key] for key in sorted(self.records)],
        }


@dataclass
class EvaluationExecutionContext:
    evaluation_run_id: str
    execution_instance_registry: Optional[ExecutionInstanceRegistry]
    invocation_sequence_counter: int = 0

    def validate(self) -> None:
        if not self.evaluation_run_id:
            raise MissingEvaluationExecutionContextError("evaluation_run_id is required")
        if self.execution_instance_registry is None:
            raise MissingExecutionInstanceRegistryError("EvaluationExecutionContext requires a shared ExecutionInstanceRegistry")

    def register(self, branch_context: BranchExecutionContext) -> Dict[str, Any]:
        self.validate()
        return self.execution_instance_registry.register(branch_context, evaluation_run_id=self.evaluation_run_id)

    def next_invocation_sequence(self) -> int:
        self.invocation_sequence_counter += 1
        return self.invocation_sequence_counter

    def to_payload(self) -> Dict[str, Any]:
        self.validate()
        return {
            "evaluation_run_id": self.evaluation_run_id,
            "invocation_sequence_counter": self.invocation_sequence_counter,
            "execution_instance_registry": self.execution_instance_registry.to_payload(),
        }


@dataclass(frozen=True)
class ConditionalSkipDecision:
    requested_action: str
    action_allowed: bool
    executed_action: str
    fallback_action: Optional[str]
    reason_code: str
    reason_details: Mapping[str, Any]
    action_mask_value: bool
    safety_predicate_source: str
    safety_predicate_source_sha256: str
    branch_id: str
    step_index: int
    agent_id: int
    vehicle_id: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "requested_action": self.requested_action,
            "action_allowed": bool(self.action_allowed),
            "executed_action": self.executed_action,
            "fallback_action": self.fallback_action,
            "reason_code": self.reason_code,
            "reason_details": dict(self.reason_details),
            "action_mask_value": bool(self.action_mask_value),
            "safety_predicate_source": self.safety_predicate_source,
            "safety_predicate_source_sha256": self.safety_predicate_source_sha256,
            "branch_id": self.branch_id,
            "step_index": int(self.step_index),
            "agent_id": int(self.agent_id),
            "vehicle_id": self.vehicle_id,
        }

    @property
    def decision_hash(self) -> str:
        return canonical_hash(self.to_payload())


@dataclass(frozen=True)
class RequestCandidate:
    request_id: str
    service_leg_id: Optional[str]
    passenger_id: Optional[str]
    request_timestamp_seconds: int
    candidate_service_start_seconds: Optional[float]
    agent_id: int
    vehicle_id: Optional[str]
    service_feasible: bool
    feasibility_reason: Optional[str]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "service_leg_id": self.service_leg_id,
            "passenger_id": self.passenger_id,
            "request_timestamp_seconds": int(self.request_timestamp_seconds),
            "candidate_service_start_seconds": self.candidate_service_start_seconds,
            "agent_id": int(self.agent_id),
            "vehicle_id": self.vehicle_id,
            "service_feasible": bool(self.service_feasible),
            "feasibility_reason": self.feasibility_reason,
        }


@dataclass(frozen=True)
class RequestWinner:
    request_id: str
    service_leg_id: Optional[str]
    passenger_id: Optional[str]
    agent_id: int
    vehicle_id: Optional[str]
    candidate_service_start_seconds: Optional[float]
    feasibility_reason: Optional[str]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "service_leg_id": self.service_leg_id,
            "passenger_id": self.passenger_id,
            "agent_id": int(self.agent_id),
            "vehicle_id": self.vehicle_id,
            "candidate_service_start_seconds": self.candidate_service_start_seconds,
            "feasibility_reason": self.feasibility_reason,
        }


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reason_code_from_engine(reason_codes: Sequence[str]) -> str:
    reason_map = {
        "WAITING_PICKUP_DEMAND": "WAITING_PASSENGER",
        "ASSIGNED_PICKUP_REQUEST": "ASSIGNED_PICKUP",
        "ONBOARD_DROPOFF_DEMAND": "ONBOARD_DROPOFF",
        "ASSIGNED_DROPOFF_REQUEST": "ASSIGNED_DROPOFF",
        "MANDATORY_STOP": "MANDATORY_STOP",
        "NO_NEXT_CANDIDATE_STOP": "INVALID_ROUTE_CONTINUATION",
        "NO_POST_SKIP_TARGET": "INVALID_ROUTE_CONTINUATION",
        "DOWNSTREAM_PATH_INVALID": "INVALID_ROUTE_CONTINUATION",
        "PLANNED_ITINERARY_BLOCK": "INVALID_ROUTE_CONTINUATION",
        "CONSECUTIVE_SKIP_LIMIT": "CONSECUTIVE_SKIP_LIMIT",
        "SERVICE_FAIRNESS_BLOCK": "SERVICE_FAIRNESS_LIMIT",
    }
    for reason in reason_codes:
        if reason in reason_map:
            return reason_map[reason]
    return "UNKNOWN_SAFETY_REASON"


def build_conditional_skip_decision(
    vehicle: Any,
    routes: Mapping[Any, Any],
    *,
    branch_id: str,
    step_index: int,
    agent_id: int,
    vehicle_id: Optional[str] = None,
    environment_terminal: bool = False,
) -> ConditionalSkipDecision:
    safety = evaluate_skip_safety(vehicle, routes)
    mask = evaluate_action_mask(vehicle, routes, environment_terminal=environment_terminal)
    safety_allows = bool(safety.get("skip_valid", False))
    mask_allows = bool(mask.get("skip_valid", False))
    reason_details = {
        "safety_result": dict(safety),
        "action_mask_result": dict(mask),
        "environment_terminal": bool(environment_terminal),
    }
    if safety_allows != mask_allows:
        action_allowed = False
        executed_action = DynamicsBranchAction.HOLD_CURRENT_POSITION.value
        fallback_action = DynamicsBranchAction.HOLD_CURRENT_POSITION.value
        reason_code = "ACTION_MASK_SAFETY_MISMATCH"
    elif safety_allows and mask_allows:
        action_allowed = True
        executed_action = DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP.value
        fallback_action = None
        reason_code = "ALLOWED"
    else:
        reason_codes = list(safety.get("skip_invalid_reason_codes", ())) or list(mask.get("skip_invalid_reason_codes", ()))
        action_allowed = False
        executed_action = DynamicsBranchAction.HOLD_CURRENT_POSITION.value
        fallback_action = DynamicsBranchAction.HOLD_CURRENT_POSITION.value
        reason_code = _reason_code_from_engine(reason_codes) if reason_codes else "ACTION_MASKED"
    source_path = Path(__file__).with_name("suseong_service_transition_engine.py")
    return ConditionalSkipDecision(
        requested_action=DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP.value,
        action_allowed=action_allowed,
        executed_action=executed_action,
        fallback_action=fallback_action,
        reason_code=reason_code,
        reason_details=reason_details,
        action_mask_value=mask_allows,
        safety_predicate_source="simulator.suseong_service_transition_engine.evaluate_conditional_skip_safety",
        safety_predicate_source_sha256=_source_sha256(source_path),
        branch_id=str(branch_id),
        step_index=int(step_index),
        agent_id=int(agent_id),
        vehicle_id=str(vehicle_id if vehicle_id is not None else getattr(vehicle, "agent_id", agent_id)),
    )


def invalid_skip_event_from_decision(
    decision: ConditionalSkipDecision,
    *,
    event_timestamp_seconds: int,
    route_id: Optional[str],
    stop_id: Optional[str],
    event_sequence: int,
    execution_instance_id: str,
    evaluation_run_id: str,
    source_trace_hash: Optional[str] = None,
) -> DynamicsEvent:
    event_id = (
        f"{decision.branch_id}-step{decision.step_index:02d}-agent{decision.agent_id}-"
        f"{DynamicsEventType.INVALID_SKIP.value.lower().replace('_', '-')}-{int(event_sequence):04d}"
    )
    runtime_record_key = canonical_hash({
        "evaluation_run_id": evaluation_run_id,
        "execution_instance_id": execution_instance_id,
        "event_id": event_id,
    })
    metadata = {
        "branch_id": decision.branch_id,
        "logical_branch_id": decision.branch_id,
        "evaluation_run_id": evaluation_run_id,
        "execution_instance_id": execution_instance_id,
        "event_sequence": int(event_sequence),
        "runtime_record_key": runtime_record_key,
        "requested_action": decision.requested_action,
        "executed_action": decision.executed_action,
        "fallback_action": decision.fallback_action,
        "reason_code": decision.reason_code,
        "reason_details": dict(decision.reason_details),
        "action_mask_value": bool(decision.action_mask_value),
        "source_decision_hash": decision.decision_hash,
        "invalid_skip_event_count_per_rejected_action_attempt": 1,
        "silent_substitution_allowed": False,
    }
    return DynamicsEvent(
        event_id=event_id,
        event_timestamp_seconds=int(event_timestamp_seconds),
        step_index=int(decision.step_index),
        event_type=DynamicsEventType.INVALID_SKIP,
        agent_id=int(decision.agent_id),
        vehicle_id=decision.vehicle_id,
        route_id=route_id,
        stop_id=stop_id,
        source_trace_hash=source_trace_hash or decision.decision_hash,
        metadata=metadata,
    )


def _normalize_request_candidate(agent_id: int, payload: Mapping[str, Any]) -> RequestCandidate:
    request_id = str(payload.get("request_id", ""))
    service_start = payload.get("candidate_service_start_seconds")
    return RequestCandidate(
        request_id=request_id,
        service_leg_id=str(payload["service_leg_id"]) if payload.get("service_leg_id") is not None else None,
        passenger_id=str(payload["passenger_id"]) if payload.get("passenger_id") is not None else None,
        request_timestamp_seconds=int(payload.get("request_timestamp_seconds", 0)),
        candidate_service_start_seconds=float(service_start) if service_start is not None else None,
        agent_id=int(payload.get("agent_id", agent_id)),
        vehicle_id=str(payload["vehicle_id"]) if payload.get("vehicle_id") is not None else None,
        service_feasible=bool(payload.get("service_feasible", False)),
        feasibility_reason=str(payload["feasibility_reason"]) if payload.get("feasibility_reason") is not None else None,
    )


def conflict_tiebreak_key(request: Mapping[str, Any], agent_id: int) -> Tuple[Any, ...]:
    service_start = request.get("candidate_service_start_seconds")
    return (
        0 if request.get("service_feasible") is True else 1,
        float(service_start) if service_start is not None else float("inf"),
        int(agent_id),
    )


def resolve_request_winner(*, request: Mapping[str, Any], candidates: Sequence[Tuple[int, Mapping[str, Any]]]) -> Optional[RequestWinner]:
    del request
    normalized = tuple(_normalize_request_candidate(agent_id, payload) for agent_id, payload in candidates)
    feasible_candidates = [
        candidate
        for candidate in normalized
        if candidate.service_feasible is True
    ]
    if not feasible_candidates:
        return None
    winner = min(
        feasible_candidates,
        key=lambda candidate: (
            candidate.candidate_service_start_seconds
            if candidate.candidate_service_start_seconds is not None
            else float("inf"),
            candidate.agent_id,
        ),
    )
    return RequestWinner(
        request_id=winner.request_id,
        service_leg_id=winner.service_leg_id,
        passenger_id=winner.passenger_id,
        agent_id=winner.agent_id,
        vehicle_id=winner.vehicle_id,
        candidate_service_start_seconds=winner.candidate_service_start_seconds,
        feasibility_reason=winner.feasibility_reason,
    )


def resolve_shared_request_conflict(candidates: Sequence[Tuple[int, Mapping[str, Any]]]) -> Optional[int]:
    if not candidates:
        return None
    request = dict(candidates[0][1])
    winner = resolve_request_winner(request=request, candidates=candidates)
    return None if winner is None else int(winner.agent_id)


def order_requests(requests: Iterable[Mapping[str, Any]]) -> Tuple[Mapping[str, Any], ...]:
    return tuple(sorted(
        requests,
        key=lambda request: (
            int(request.get("request_timestamp_seconds", 0)),
            str(request.get("request_id", "")),
        ),
    ))


def build_request_ownership_map(
    requests_by_id: Mapping[str, Mapping[str, Any]],
    candidates_by_request_id: Mapping[str, Sequence[Tuple[int, Mapping[str, Any]]]],
) -> Dict[str, Any]:
    ownership: Dict[str, Optional[int]] = {}
    winners: Dict[str, Optional[Dict[str, Any]]] = {}
    for request in order_requests(requests_by_id.values()):
        request_id = str(request.get("request_id", ""))
        winner = resolve_request_winner(
            request=request,
            candidates=candidates_by_request_id.get(request_id, ()),
        )
        ownership[request_id] = None if winner is None else int(winner.agent_id)
        winners[request_id] = None if winner is None else winner.to_payload()
    payload = {
        "request_ownership_by_request_id": ownership,
        "request_winners_by_request_id": winners,
        "request_ownership_frozen_before_mutation": True,
    }
    return {
        **payload,
        "request_ownership_map_hash": canonical_hash(payload),
    }


def _service_leg_identity(request_id: str, service_leg_id: str) -> CanonicalServiceUnitIdentity:
    if not request_id or not service_leg_id:
        raise MissingServiceLegIdentityError("leg-scoped evidence requires request_id and service_leg_id")
    return CanonicalServiceUnitIdentity(ServiceUnitScope.SERVICE_LEG, request_id=request_id, service_leg_id=service_leg_id)


def _request_identity(request_id: str) -> CanonicalServiceUnitIdentity:
    if not request_id:
        raise ValueError("request-scoped evidence requires request_id")
    return CanonicalServiceUnitIdentity(ServiceUnitScope.REQUEST, request_id=request_id, service_leg_id=None)


def _completion_scope_from_event(event: DynamicsEvent, request_id: str) -> Optional[ServiceUnitScope]:
    if event.event_type != DynamicsEventType.SERVICE_COMPLETED or not request_id:
        return None
    metadata = dict(event.metadata)
    scope = metadata.get("completion_scope")
    if scope is None:
        raise AmbiguousServiceCompletionScopeError("SERVICE_COMPLETED with request_id requires completion_scope")
    try:
        return ServiceUnitScope(str(scope))
    except ValueError as exc:
        raise AmbiguousServiceCompletionScopeError(f"unsupported completion_scope: {scope}") from exc


def check_request_service_invariants(
    events: Sequence[DynamicsEvent],
    onboard_assignments: Sequence[Mapping[str, Any]] = (),
    *,
    served_count: Optional[int] = None,
) -> Dict[str, Any]:
    board_events_by_request_id: Dict[str, int] = {}
    board_events_by_service_leg_id: Dict[str, int] = {}
    service_completed_by_request_id: Dict[str, int] = {}
    service_completed_by_service_leg_id: Dict[str, int] = {}
    onboard_assignments_by_request_id: Dict[str, int] = {}
    onboard_assignments_by_service_leg_id: Dict[str, int] = {}
    request_to_service_legs: Dict[str, set] = {}
    service_leg_to_request: Dict[str, str] = {}
    service_unit_evidence_channels: Dict[str, Dict[str, int]] = {}
    service_unit_identity_payloads: Dict[str, Dict[str, Any]] = {}
    board_vehicle_by_key: Dict[str, set] = {}
    assignment_vehicle_by_key: Dict[str, set] = {}
    passenger_request_leg_vehicle_assignments: Dict[Tuple[str, str, str], set] = {}

    def add_leg_evidence(request_id: str, service_leg_id: str, channel: str, vehicle_id: Optional[str] = None) -> str:
        identity = _service_leg_identity(request_id, service_leg_id)
        key = identity.canonical_key
        request_to_service_legs.setdefault(request_id, set()).add(service_leg_id)
        existing_request = service_leg_to_request.get(service_leg_id)
        if existing_request is not None and existing_request != request_id:
            raise ValueError(f"service_leg_id {service_leg_id} is linked to multiple request_id values")
        service_leg_to_request[service_leg_id] = request_id
        service_unit_identity_payloads[key] = identity.to_payload()
        service_unit_evidence_channels.setdefault(key, {})
        service_unit_evidence_channels[key][channel] = service_unit_evidence_channels[key].get(channel, 0) + 1
        if vehicle_id is not None:
            if channel == "BOARD_EVENT":
                board_vehicle_by_key.setdefault(key, set()).add(str(vehicle_id))
            if channel == "ONBOARD_ASSIGNMENT":
                assignment_vehicle_by_key.setdefault(key, set()).add(str(vehicle_id))
        return key

    def add_request_evidence(request_id: str, channel: str) -> str:
        identity = _request_identity(request_id)
        key = identity.canonical_key
        service_unit_identity_payloads[key] = identity.to_payload()
        service_unit_evidence_channels.setdefault(key, {})
        service_unit_evidence_channels[key][channel] = service_unit_evidence_channels[key].get(channel, 0) + 1
        return key

    for event in events:
        metadata = dict(event.metadata)
        request_id = event.request_id or str(metadata.get("request_id", "") or "")
        service_leg_id = str(metadata.get("service_leg_id", "") or "")
        if event.event_type == DynamicsEventType.PASSENGER_BOARD and request_id:
            board_events_by_request_id[request_id] = board_events_by_request_id.get(request_id, 0) + 1
            board_events_by_service_leg_id[service_leg_id] = board_events_by_service_leg_id.get(service_leg_id, 0) + 1
            add_leg_evidence(request_id, service_leg_id, "BOARD_EVENT", event.vehicle_id)
        if event.event_type == DynamicsEventType.SERVICE_COMPLETED and request_id:
            completion_scope = _completion_scope_from_event(event, request_id)
            if completion_scope == ServiceUnitScope.SERVICE_LEG:
                key = add_leg_evidence(request_id, service_leg_id, "LEG_COMPLETED_EVENT", event.vehicle_id)
                service_completed_by_service_leg_id[key] = service_completed_by_service_leg_id.get(key, 0) + 1
            elif completion_scope == ServiceUnitScope.REQUEST:
                if service_leg_id:
                    raise AmbiguousServiceCompletionScopeError("REQUEST completion must not include service_leg_id")
                add_request_evidence(request_id, "REQUEST_COMPLETED_EVENT")
                service_completed_by_request_id[request_id] = service_completed_by_request_id.get(request_id, 0) + 1
    for assignment in onboard_assignments:
        request_id = str(assignment.get("request_id", "") or "")
        passenger_id = str(assignment.get("passenger_id", "") or "")
        vehicle_id = str(assignment.get("vehicle_id", "") or "")
        service_leg_id = str(assignment.get("service_leg_id", "") or "")
        if request_id and passenger_id and vehicle_id and service_leg_id:
            passenger_request_leg_vehicle_assignments.setdefault((passenger_id, request_id, service_leg_id), set()).add(vehicle_id)
        if request_id:
            onboard_assignments_by_request_id[request_id] = onboard_assignments_by_request_id.get(request_id, 0) + 1
            onboard_assignments_by_service_leg_id[service_leg_id] = onboard_assignments_by_service_leg_id.get(service_leg_id, 0) + 1
            add_leg_evidence(request_id, service_leg_id, "ONBOARD_ASSIGNMENT", vehicle_id)
    duplicate_violations = []
    duplicate_service_leg_keys = set()
    duplicate_request_keys = set()
    for key, channels in sorted(service_unit_evidence_channels.items()):
        identity = service_unit_identity_payloads[key]
        if identity["scope"] == ServiceUnitScope.SERVICE_LEG.value:
            request_id = str(identity["request_id"])
            service_leg_id = str(identity["service_leg_id"])
            if channels.get("BOARD_EVENT", 0) > 1:
                duplicate_service_leg_keys.add(key)
                duplicate_violations.append({"invariant": "BOARD_EVENT_SERVICE_LEG_AT_MOST_ONCE", "service_unit_key": key, "request_id": request_id, "service_leg_id": service_leg_id, "count": channels["BOARD_EVENT"]})
            if channels.get("ONBOARD_ASSIGNMENT", 0) > 1:
                duplicate_service_leg_keys.add(key)
                duplicate_violations.append({"invariant": "ONBOARD_ASSIGNMENT_SERVICE_LEG_AT_MOST_ONCE", "service_unit_key": key, "request_id": request_id, "service_leg_id": service_leg_id, "count": channels["ONBOARD_ASSIGNMENT"]})
            if channels.get("LEG_COMPLETED_EVENT", 0) > 1:
                duplicate_service_leg_keys.add(key)
                duplicate_violations.append({"invariant": "SERVICE_LEG_COMPLETED_AT_MOST_ONCE", "service_unit_key": key, "request_id": request_id, "service_leg_id": service_leg_id, "count": channels["LEG_COMPLETED_EVENT"]})
        elif identity["scope"] == ServiceUnitScope.REQUEST.value:
            request_id = str(identity["request_id"])
            if channels.get("REQUEST_COMPLETED_EVENT", 0) > 1:
                duplicate_request_keys.add(key)
                duplicate_violations.append({"invariant": "REQUEST_COMPLETED_AT_MOST_ONCE", "service_unit_key": key, "request_id": request_id, "count": channels["REQUEST_COMPLETED_EVENT"]})
    for (passenger_id, request_id, service_leg_id), vehicles in sorted(passenger_request_leg_vehicle_assignments.items()):
        if len(vehicles) > 1:
            key = _service_leg_identity(request_id, service_leg_id).canonical_key
            duplicate_service_leg_keys.add(key)
            duplicate_violations.append({"invariant": "PASSENGER_REQUEST_LEG_ON_MULTIPLE_VEHICLES", "service_unit_key": key, "passenger_id": passenger_id, "request_id": request_id, "service_leg_id": service_leg_id, "vehicle_count": len(vehicles)})
    cross_source_violations = []
    for key in sorted(set(board_vehicle_by_key) | set(assignment_vehicle_by_key)):
        channels = service_unit_evidence_channels.get(key, {})
        identity = service_unit_identity_payloads[key]
        board_count = channels.get("BOARD_EVENT", 0)
        assignment_count = channels.get("ONBOARD_ASSIGNMENT", 0)
        board_vehicles = board_vehicle_by_key.get(key, set())
        assignment_vehicles = assignment_vehicle_by_key.get(key, set())
        if board_count == 1 and assignment_count == 0:
            cross_source_violations.append({"invariant": "BOARD_WITHOUT_ASSIGNMENT", "service_unit_key": key, "request_id": identity["request_id"], "service_leg_id": identity["service_leg_id"]})
        elif board_count == 0 and assignment_count == 1:
            cross_source_violations.append({"invariant": "ASSIGNMENT_WITHOUT_BOARD", "service_unit_key": key, "request_id": identity["request_id"], "service_leg_id": identity["service_leg_id"]})
        elif board_count == 1 and assignment_count == 1 and board_vehicles and assignment_vehicles and board_vehicles != assignment_vehicles:
            cross_source_violations.append({
                "invariant": "BOARD_ASSIGNMENT_VEHICLE_MISMATCH",
                "service_unit_key": key,
                "request_id": identity["request_id"],
                "service_leg_id": identity["service_leg_id"],
                "board_vehicle_ids": sorted(board_vehicles),
                "assignment_vehicle_ids": sorted(assignment_vehicles),
            })
    unique_completed = len({key for key, channels in service_unit_evidence_channels.items() if channels.get("LEG_COMPLETED_EVENT", 0) > 0})
    served_count_matches = served_count is None or int(served_count) == unique_completed
    if not served_count_matches:
        duplicate_violations.append({"invariant": "SERVED_COUNT_EQUALS_UNIQUE_COMPLETED_SERVICE_LEG_COUNT", "served_count": int(served_count), "unique_completed_service_leg_count": unique_completed})
    board_event_duplicate_count = sum(max(0, channels.get("BOARD_EVENT", 0) - 1) for channels in service_unit_evidence_channels.values())
    service_completed_duplicate_count = (
        sum(max(0, channels.get("LEG_COMPLETED_EVENT", 0) - 1) for channels in service_unit_evidence_channels.values())
        + sum(max(0, channels.get("REQUEST_COMPLETED_EVENT", 0) - 1) for channels in service_unit_evidence_channels.values())
    )
    onboard_assignment_duplicate_count = sum(max(0, channels.get("ONBOARD_ASSIGNMENT", 0) - 1) for channels in service_unit_evidence_channels.values())
    multiple_vehicle_assignment_count = sum(1 for value in passenger_request_leg_vehicle_assignments.values() if len(value) > 1)
    served_count_unique_mismatch_count = 0 if served_count_matches else 1
    unique_duplicate_service_key_count = len(duplicate_service_leg_keys) + len(duplicate_request_keys)
    duplicate_invariant_violation_count = len(duplicate_violations)
    actual_duplicate_service_count = unique_duplicate_service_key_count
    return {
        "invariant_name": "REQUEST_SERVED_AT_MOST_ONCE",
        "request_served_at_most_once": duplicate_invariant_violation_count == 0,
        "actual_duplicate_service_count": actual_duplicate_service_count,
        "actual_duplicate_service_count_semantics": "DEPRECATED_ALIAS_OF_UNIQUE_DUPLICATE_SERVICE_KEY_COUNT",
        "unique_duplicate_service_leg_key_count": len(duplicate_service_leg_keys),
        "unique_duplicate_request_key_count": len(duplicate_request_keys),
        "unique_duplicate_service_key_count": unique_duplicate_service_key_count,
        "duplicate_service_leg_keys": sorted(duplicate_service_leg_keys),
        "duplicate_request_keys": sorted(duplicate_request_keys),
        "duplicate_service_keys": sorted(set(duplicate_service_leg_keys) | set(duplicate_request_keys)),
        "duplicate_invariant_violation_count": duplicate_invariant_violation_count,
        "cross_source_correspondence_failure_count": len(cross_source_violations),
        "board_event_duplicate_count": board_event_duplicate_count,
        "service_completed_duplicate_count": service_completed_duplicate_count,
        "onboard_assignment_duplicate_count": onboard_assignment_duplicate_count,
        "multiple_vehicle_assignment_count": multiple_vehicle_assignment_count,
        "served_count_unique_mismatch_count": served_count_unique_mismatch_count,
        "violation_count": duplicate_invariant_violation_count,
        "violations": duplicate_violations,
        "cross_source_violations": cross_source_violations,
        "board_events_by_request_id": dict(board_events_by_request_id),
        "board_events_by_service_leg_id": dict(board_events_by_service_leg_id),
        "service_completed_by_request_id": dict(service_completed_by_request_id),
        "service_completed_by_service_leg_id": dict(service_completed_by_service_leg_id),
        "onboard_assignments_by_request_id": dict(onboard_assignments_by_request_id),
        "onboard_assignments_by_service_leg_id": dict(onboard_assignments_by_service_leg_id),
        "passenger_request_vehicle_assignments": {f"{key[0]}|{key[1]}|{key[2]}": sorted(value) for key, value in passenger_request_leg_vehicle_assignments.items()},
        "request_to_service_legs": {key: sorted(value) for key, value in request_to_service_legs.items()},
        "service_leg_to_request": dict(service_leg_to_request),
        "service_unit_evidence_channels": {key: dict(value) for key, value in sorted(service_unit_evidence_channels.items())},
        "service_unit_identities": dict(sorted(service_unit_identity_payloads.items())),
        "served_count": served_count,
        "unique_completed_request_count": unique_completed,
        "unique_completed_service_leg_count": unique_completed,
        "served_count_equals_unique_completed_request_count": served_count_matches,
    }


def _route_key(value: Any) -> Any:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, str) and "|" in value:
        left, right = value.split("|", 1)
        return left, right
    return value


def _canonical_route_key(value: Any) -> str:
    if isinstance(value, tuple) and len(value) == 2:
        return f"{value[0]}|{value[1]}"
    return str(value)


def _runtime_payload_to_engine_objects(payload: Mapping[str, Any]) -> Tuple[Dict[int, SimpleNamespace], Dict[Any, Any]]:
    vehicles = {}
    for raw_agent_id, vehicle_payload in dict(payload["vehicles"]).items():
        vehicle_data = dict(vehicle_payload)
        agent_id = int(vehicle_data.get("agent_id", raw_agent_id))
        if "route_key" in vehicle_data:
            vehicle_data["route_key"] = _route_key(vehicle_data["route_key"])
        vehicles[agent_id] = SimpleNamespace(**vehicle_data)
    routes = {_route_key(key): copy.deepcopy(value) for key, value in dict(payload["routes"]).items()}
    return vehicles, routes


def _engine_objects_to_runtime_payload(payload: Mapping[str, Any], vehicles: Mapping[int, SimpleNamespace], routes: Mapping[Any, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(payload))
    out["vehicles"] = {}
    for agent_id, vehicle in sorted(vehicles.items()):
        vehicle_data = copy.deepcopy(vars(vehicle))
        if "route_key" in vehicle_data:
            vehicle_data["route_key"] = _canonical_route_key(vehicle_data["route_key"])
        out["vehicles"][str(agent_id)] = vehicle_data
    out["routes"] = {_canonical_route_key(key): copy.deepcopy(value) for key, value in sorted(routes.items(), key=lambda item: _canonical_route_key(item[0]))}
    return out


def _vehicle_route_event_context(vehicle: Any, routes: Mapping[Any, Any]) -> Tuple[Optional[str], Optional[str]]:
    route_key = getattr(vehicle, "route_key", None)
    route_id = str(route_key[0]) if isinstance(route_key, tuple) and route_key else str(route_key) if route_key is not None else None
    stop_id = None
    try:
        route = routes[route_key]
        pos = int(getattr(vehicle, "position"))
        next_index = pos + 1
        if 0 <= next_index < len(route):
            stop_id = str(route[next_index].get("stop_id", route[next_index].get("node_uid", "")))
        elif 0 <= pos < len(route):
            stop_id = str(route[pos].get("stop_id", route[pos].get("node_uid", "")))
    except Exception:
        stop_id = None
    return route_id, stop_id


def advance_multiagent_global_step(
    *,
    state: DynamicsStateSnapshot,
    action_by_agent: Mapping[int, DynamicsBranchAction],
    replay_frame: ReplayFrame,
    delta_t_seconds: int,
    stop_service_provider: Callable[..., Any],
    config: Any,
    turnaround_mapping: Optional[Mapping[Any, Any]] = None,
    step_index: int = 0,
    branch_context: Optional[BranchExecutionContext] = None,
    evaluation_context: Optional[EvaluationExecutionContext] = None,
) -> Tuple[DynamicsStateSnapshot, GlobalStepTrace]:
    _authz.require_capability("simulator_execution", site="simulator/dynamics_multiagent_orchestrator.py::advance_multiagent_global_step")
    if branch_context is None:
        raise MissingBranchExecutionContextError("advance_multiagent_global_step requires explicit BranchExecutionContext")
    evaluation_run_id = (
        evaluation_context.evaluation_run_id
        if evaluation_context is not None
        else branch_context.caller_run_id or branch_context.run_id
    )
    if int(delta_t_seconds) != DECISION_INTERVAL_SECONDS:
        raise ValueError("delta_t_seconds must be 60 for ER1 global-step dynamics")
    if len(action_by_agent) != 8:
        raise ValueError("ER1 global-step orchestrator requires exactly 8 active agents")
    order = canonical_agent_order(action_by_agent.keys())
    frozen_actions = {agent_id: adapt_branch_action(action_by_agent[agent_id]) for agent_id in order}
    state_before = hash_dynamics_state(state)
    runtime_payload = copy.deepcopy(state.to_payload())
    vehicles, routes = _runtime_payload_to_engine_objects(runtime_payload)
    step_events = []
    branch_id = branch_context.logical_branch_id
    for agent_id in order:
        if agent_id not in vehicles:
            raise ValueError(f"missing runtime vehicle for active agent {agent_id}")
        requested_action = action_by_agent[agent_id]
        action_for_engine = frozen_actions[agent_id]
        conditional_skip_decision: Optional[ConditionalSkipDecision] = None
        if requested_action == DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP:
            conditional_skip_decision = build_conditional_skip_decision(
                vehicles[agent_id],
                routes,
                branch_id=branch_id,
                step_index=step_index,
                agent_id=agent_id,
                vehicle_id=str(getattr(vehicles[agent_id], "agent_id", agent_id)),
            )
            if not conditional_skip_decision.action_allowed:
                route_id, stop_id = _vehicle_route_event_context(vehicles[agent_id], routes)
                step_events.append(invalid_skip_event_from_decision(
                    conditional_skip_decision,
                    event_timestamp_seconds=int(step_index) * DECISION_INTERVAL_SECONDS,
                    route_id=route_id,
                    stop_id=stop_id,
                    event_sequence=0,
                    execution_instance_id=branch_context.execution_instance_id,
                    evaluation_run_id=evaluation_run_id,
                ))
                action_for_engine = engine.ENGINE_ACTION_HOLD

        def wrapped_stop_service(vehicle: Any, stop_row: Mapping[str, Any]) -> Any:
            result = stop_service_provider(vehicle=vehicle, stop_row=stop_row, replay_frame=replay_frame, agent_id=agent_id)
            _ = adapt_stop_service_result(result)
            return result

        trace = engine.advance_vehicle_time_budget(
            vehicle=vehicles[agent_id],
            routes=routes,
            delta_t_seconds=float(delta_t_seconds),
            action=action_for_engine,
            stop_service=wrapped_stop_service,
            config=config,
            turnaround_mapping=turnaround_mapping,
            snapshot_id=int(step_index),
            snapshot_start_time_seconds=int(step_index) * DECISION_INTERVAL_SECONDS,
        )
        source_trace_hash = canonical_hash({
            "agent_id": agent_id,
            "requested_action": requested_action.value,
            "engine_action": int(action_for_engine),
            "branch_context": branch_context.logical_payload(),
            "conditional_skip_decision": conditional_skip_decision.to_payload() if conditional_skip_decision is not None else None,
            "events": trace.events,
        })
        for event_index, event in enumerate(trace.events):
            event_type = event_type_from_engine_event(str(event.get("event_type", "")))
            event_metadata = dict(event)
            event_metadata.update({
                "logical_branch_id": branch_id,
                "evaluation_run_id": evaluation_run_id,
                "execution_instance_id": branch_context.execution_instance_id,
                "event_sequence": int(event_index),
                "runtime_record_key": canonical_hash({
                    "evaluation_run_id": evaluation_run_id,
                    "execution_instance_id": branch_context.execution_instance_id,
                    "event_id": f"{branch_id}-step{int(step_index):02d}-agent{agent_id}-{event_type.value.lower().replace('_', '-')}-{event_index:04d}",
                }),
            })
            step_events.append(DynamicsEvent(
                event_id=f"{branch_id}-step{int(step_index):02d}-agent{agent_id}-{event_type.value.lower().replace('_', '-')}-{event_index:04d}",
                event_timestamp_seconds=int(float(event.get("event_time_seconds", int(step_index) * DECISION_INTERVAL_SECONDS))),
                step_index=int(step_index),
                event_type=event_type,
                agent_id=agent_id,
                vehicle_id=str(event.get("vehicle_id", agent_id)),
                route_id=str(event.get("route_id", "")) if event.get("route_id") is not None else None,
                stop_id=str(event.get("stop_id", "")) if event.get("stop_id") is not None else None,
                source_trace_hash=source_trace_hash,
                metadata=event_metadata,
            ))
    state_after = DynamicsStateSnapshot(_engine_objects_to_runtime_payload(runtime_payload, vehicles, routes))
    trace = GlobalStepTrace(
        step_index=int(step_index),
        elapsed_seconds_before=int(step_index) * DECISION_INTERVAL_SECONDS,
        elapsed_seconds_after=(int(step_index) + 1) * DECISION_INTERVAL_SECONDS,
        state_hash_before=state_before,
        state_hash_after=hash_dynamics_state(state_after),
        action_vector_hash=action_vector_hash(action_by_agent),
        replay_frame_hash=replay_frame.frame_hash,
        event_trace_hash=event_trace_hash(step_events),
        canonical_agent_order=order,
    )
    return state_after, trace


def event_type_from_engine_event(engine_event_type: str) -> Any:
    from simulator.dynamics_event_trace import DynamicsEventType

    mapping = {
        "TRAVEL": DynamicsEventType.VEHICLE_MOVE,
        "EDGE_ENTERED": DynamicsEventType.VEHICLE_MOVE,
        "EDGE_TRAVEL_ZERO": DynamicsEventType.VEHICLE_MOVE,
        "DWELL": DynamicsEventType.VEHICLE_DWELL,
        "STOP_ARRIVAL": DynamicsEventType.STOP_ARRIVAL,
        "STOP_SERVICE": DynamicsEventType.SERVICE_COMPLETED,
        "CONDITIONAL_SKIP_VALIDATED": DynamicsEventType.CONDITIONAL_SKIP,
        "INVALID_SKIP": DynamicsEventType.INVALID_SKIP,
        "TERMINAL_IDLE_OBSERVATION": DynamicsEventType.SERVICE_COMPLETED,
        "ROUTE_TERMINAL_REACHED": DynamicsEventType.SERVICE_COMPLETED,
        "TURNAROUND": DynamicsEventType.ROUTE_TURNAROUND,
        "TURNAROUND_STARTED": DynamicsEventType.ROUTE_TURNAROUND,
        "SERVICE_RESUMED": DynamicsEventType.ROUTE_TURNAROUND,
    }
    return mapping.get(engine_event_type, DynamicsEventType.SERVICE_COMPLETED)


def build_single_pulse_action_vector(
    *,
    target_agent_id: int,
    target_action: DynamicsBranchAction,
    active_agent_ids: Sequence[int],
    step_index: int,
) -> Dict[int, DynamicsBranchAction]:
    order = canonical_agent_order(active_agent_ids)
    if len(order) != 8:
        raise ValueError("single-pulse action vector requires exactly 8 active agents")
    if int(target_agent_id) not in order:
        raise ValueError("target_agent_id must be active")
    if int(step_index) == 0:
        return {agent_id: (target_action if agent_id == int(target_agent_id) else DynamicsBranchAction.HOLD_CURRENT_POSITION) for agent_id in order}
    return {agent_id: DynamicsBranchAction.HOLD_CURRENT_POSITION for agent_id in order}


def run_thirty_minute_branch(
    *,
    initial_state: DynamicsStateSnapshot,
    target_agent_id: int,
    target_action: DynamicsBranchAction,
    active_agent_ids: Sequence[int],
    replay_frames: Sequence[ReplayFrame],
    stop_service_provider: Callable[..., Any],
    config: Any,
    evaluation_context: EvaluationExecutionContext,
    branch_context: BranchExecutionContext,
) -> Tuple[DynamicsStateSnapshot, Tuple[GlobalStepTrace, ...]]:
    if evaluation_context is None:
        raise MissingEvaluationExecutionContextError("run_thirty_minute_branch requires explicit EvaluationExecutionContext")
    evaluation_context.validate()
    if branch_context is None:
        raise MissingBranchExecutionContextError("run_thirty_minute_branch requires explicit BranchExecutionContext")
    if len(replay_frames) != THIRTY_MINUTE_TOTAL_STEPS:
        raise ValueError("run_thirty_minute_branch requires exactly 30 replay frames")
    state = initial_state
    traces = []
    replay_input_hash = canonical_hash({"frame_hashes": [frame.frame_hash for frame in replay_frames]})
    evaluation_context.register(branch_context)
    if branch_context.initial_state_hash != initial_state.state_hash:
        raise ValueError("branch_context.initial_state_hash does not match initial_state")
    if branch_context.replay_input_hash != replay_input_hash:
        raise ValueError("branch_context.replay_input_hash does not match replay_frames")
    if int(branch_context.target_agent_id) != int(target_agent_id):
        raise ValueError("branch_context.target_agent_id does not match target_agent_id")
    if branch_context.pulse_action != target_action.value:
        raise ValueError("branch_context.pulse_action does not match target_action")
    for step_index, replay_frame in enumerate(replay_frames):
        actions = build_single_pulse_action_vector(
            target_agent_id=target_agent_id,
            target_action=target_action,
            active_agent_ids=active_agent_ids,
            step_index=step_index,
        )
        state, trace = advance_multiagent_global_step(
            state=state,
            action_by_agent=actions,
            replay_frame=replay_frame,
            delta_t_seconds=DECISION_INTERVAL_SECONDS,
            stop_service_provider=stop_service_provider,
            config=config,
            step_index=step_index,
            branch_context=branch_context,
            evaluation_context=evaluation_context,
        )
        traces.append(trace)
    if len(traces) != THIRTY_MINUTE_TOTAL_STEPS:
        raise ValueError("executed_step_count must equal 30")
    if traces[-1].elapsed_seconds_after != 1800:
        raise ValueError("elapsed_seconds_total must equal 1800")
    return state, tuple(traces)
