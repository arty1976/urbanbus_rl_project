from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple


K_SAFETY_STATE_SCHEMA_VERSION = "SUSEONG_K_SAFETY_STATE_V2"


class KSafetyStateError(ValueError):
    pass


class KSafetyChronologyError(KSafetyStateError):
    pass


class KSafetyIdentityError(KSafetyStateError):
    pass


class KSafetyTransitionError(KSafetyStateError):
    pass


class PassengerStatus(str, Enum):
    WAITING = "WAITING"
    ASSIGNED = "ASSIGNED"
    ONBOARD = "ONBOARD"
    ALIGHTED = "ALIGHTED"
    SERVED = "SERVED"
    CANCELLED = "CANCELLED"


class RequestStatus(str, Enum):
    CREATED = "CREATED"
    ASSIGNED = "ASSIGNED"
    BOARDED = "BOARDED"
    ALIGHTED = "ALIGHTED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


def _require_id(name: str, value: Any) -> str:
    normalized = str(value or "")
    if not normalized:
        raise KSafetyIdentityError(f"{name} is required")
    return normalized


def _bool_or_none(row: Mapping[str, Any], key: str, missing: List[str]) -> Optional[bool]:
    if key not in row or row[key] is None:
        missing.append(f"{key.upper()}_MISSING")
        return None
    if not isinstance(row[key], bool):
        missing.append(f"{key.upper()}_NOT_BOOLEAN")
        return None
    return bool(row[key])


@dataclass(frozen=True)
class StaticGuardStatus:
    mandatory_stop: Optional[bool]
    protected_stop: Optional[bool]
    planned_itinerary_allows_skip: Optional[bool]
    terminal_or_turnaround_stop: Optional[bool]
    charging_or_driver_relief_stop: Optional[bool]
    valid_post_skip_path: Optional[bool]
    state_complete: bool
    missing_reason: Tuple[str, ...] = ()
    evidence_class: str = "CONTRACT_FIXED"

    @classmethod
    def from_stop_row(
        cls,
        stop_row: Optional[Mapping[str, Any]],
        *,
        post_skip_target_exists: bool,
    ) -> "StaticGuardStatus":
        if stop_row is None:
            return cls(None, None, None, None, None, False, False, ("NEXT_STOP_MISSING",), "NOT_AVAILABLE")
        missing: List[str] = []
        mandatory = _bool_or_none(stop_row, "mandatory_stop", missing)
        protected = _bool_or_none(stop_row, "protected_stop", missing)
        planned = _bool_or_none(stop_row, "planned_itinerary_allows_skip", missing)
        terminal = _bool_or_none(stop_row, "terminal_or_turnaround_stop", missing)
        charging = _bool_or_none(stop_row, "charging_or_driver_relief_stop", missing)
        downstream = _bool_or_none(stop_row, "downstream_path_valid", missing)
        graph_path = _bool_or_none(stop_row, "graph_edge_or_path_valid", missing)
        if not post_skip_target_exists:
            missing.append("POST_SKIP_TARGET_MISSING")
            path_valid: Optional[bool] = False
        elif downstream is None or graph_path is None:
            path_valid = None
        else:
            path_valid = bool(downstream and graph_path)
        reasons = tuple(dict.fromkeys(missing))
        return cls(
            mandatory,
            protected,
            planned,
            terminal,
            charging,
            path_valid,
            not reasons,
            reasons,
            "CONTRACT_FIXED" if not reasons else "NOT_AVAILABLE",
        )

    @property
    def positively_clear(self) -> bool:
        return bool(
            self.state_complete
            and self.mandatory_stop is False
            and self.protected_stop is False
            and self.planned_itinerary_allows_skip is True
            and self.terminal_or_turnaround_stop is False
            and self.charging_or_driver_relief_stop is False
            and self.valid_post_skip_path is True
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "mandatory_stop": self.mandatory_stop,
            "protected_stop": self.protected_stop,
            "planned_itinerary_allows_skip": self.planned_itinerary_allows_skip,
            "terminal_or_turnaround_stop": self.terminal_or_turnaround_stop,
            "charging_or_driver_relief_stop": self.charging_or_driver_relief_stop,
            "valid_post_skip_path": self.valid_post_skip_path,
            "state_complete": bool(self.state_complete),
            "missing_reason": list(self.missing_reason),
            "evidence_class": self.evidence_class,
            "positively_clear": self.positively_clear,
        }


@dataclass
class Passenger:
    passenger_id: str
    pickup_stop: str
    dropoff_stop: str
    passenger_status: PassengerStatus
    created_ts: int
    last_transition_ts: int
    assigned_vehicle: Optional[str] = None
    active_request_id: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "passenger_id": self.passenger_id,
            "pickup_stop": self.pickup_stop,
            "dropoff_stop": self.dropoff_stop,
            "assigned_vehicle": self.assigned_vehicle,
            "passenger_status": self.passenger_status.value,
            "active_request_id": self.active_request_id,
            "created_ts": int(self.created_ts),
            "last_transition_ts": int(self.last_transition_ts),
        }


@dataclass
class ServiceRequest:
    request_id: str
    passenger_id: str
    service_leg_id: str
    pickup_stop: str
    dropoff_stop: str
    request_status: RequestStatus
    created_ts: int
    last_transition_ts: int
    assigned_vehicle: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "passenger_id": self.passenger_id,
            "service_leg_id": self.service_leg_id,
            "pickup_stop": self.pickup_stop,
            "dropoff_stop": self.dropoff_stop,
            "assigned_vehicle": self.assigned_vehicle,
            "request_status": self.request_status.value,
            "created_ts": int(self.created_ts),
            "last_transition_ts": int(self.last_transition_ts),
        }


@dataclass
class StopQueue:
    stop_id: str
    waiting_queue: Set[str] = field(default_factory=set)
    assigned_pickup: Set[str] = field(default_factory=set)
    state_complete: bool = True
    missing_reason: Tuple[str, ...] = ()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "stop_id": self.stop_id,
            "waiting_queue": sorted(self.waiting_queue),
            "assigned_pickup": sorted(self.assigned_pickup),
            "state_complete": bool(self.state_complete),
            "missing_reason": list(self.missing_reason),
        }


@dataclass
class VehicleObligationLedger:
    agent_id: int
    vehicle_token: str
    assigned_pickup: Set[str] = field(default_factory=set)
    assigned_dropoff: Set[str] = field(default_factory=set)
    onboard_destination: Dict[str, str] = field(default_factory=dict)
    onboard_passenger_by_request: Dict[str, str] = field(default_factory=dict)
    state_complete: bool = True
    missing_reason: Tuple[str, ...] = ()

    def to_payload(self) -> Dict[str, Any]:
        return {
            "agent_id": int(self.agent_id),
            "vehicle_token": self.vehicle_token,
            "assigned_pickup": sorted(self.assigned_pickup),
            "assigned_dropoff": sorted(self.assigned_dropoff),
            "onboard_destination": dict(sorted(self.onboard_destination.items())),
            "onboard_passenger_by_request": dict(sorted(self.onboard_passenger_by_request.items())),
            "state_complete": bool(self.state_complete),
            "missing_reason": list(self.missing_reason),
        }


@dataclass(frozen=True)
class DecisionTimeObligationSnapshot:
    agent_id: int
    vehicle_token: str
    decision_ts: int
    current_stop_id: str
    candidate_stop_id: Optional[str]
    pickup_obligation: bool
    dropoff_obligation: bool
    boarding_obligation: bool
    alighting_obligation: bool
    onboard_destination_obligation: bool
    service_obligation: bool
    assigned_pickup: bool
    assigned_dropoff: bool
    waiting_queue: Tuple[str, ...]
    assigned_pickup_request_ids: Tuple[str, ...]
    assigned_dropoff_request_ids: Tuple[str, ...]
    onboard_destination_request_ids: Tuple[str, ...]
    static_guard_status: StaticGuardStatus
    dynamic_state_complete: bool
    state_complete: bool
    missing_reason: Tuple[str, ...]
    schema_version: str = K_SAFETY_STATE_SCHEMA_VERSION

    @classmethod
    def incomplete(
        cls,
        *,
        agent_id: int,
        vehicle_token: str,
        decision_ts: int,
        current_stop_id: str,
        candidate_stop_id: Optional[str],
        static_guard_status: StaticGuardStatus,
        missing_reason: Sequence[str],
    ) -> "DecisionTimeObligationSnapshot":
        reasons = tuple(dict.fromkeys(str(reason) for reason in missing_reason if str(reason)))
        return cls(
            int(agent_id),
            str(vehicle_token),
            int(decision_ts),
            str(current_stop_id),
            str(candidate_stop_id) if candidate_stop_id is not None else None,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            (),
            (),
            (),
            (),
            static_guard_status,
            False,
            False,
            reasons,
        )

    @property
    def skip_allowed(self) -> bool:
        return bool(
            self.dynamic_state_complete
            and self.state_complete
            and self.static_guard_status.positively_clear
            and not self.pickup_obligation
            and not self.dropoff_obligation
            and not self.boarding_obligation
            and not self.alighting_obligation
            and not self.onboard_destination_obligation
            and not self.service_obligation
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "agent_id": int(self.agent_id),
            "vehicle_token": self.vehicle_token,
            "decision_ts": int(self.decision_ts),
            "current_stop_id": self.current_stop_id,
            "candidate_stop_id": self.candidate_stop_id,
            "pickup_obligation": bool(self.pickup_obligation),
            "dropoff_obligation": bool(self.dropoff_obligation),
            "boarding_obligation": bool(self.boarding_obligation),
            "alighting_obligation": bool(self.alighting_obligation),
            "onboard_destination_obligation": bool(self.onboard_destination_obligation),
            "service_obligation": bool(self.service_obligation),
            "assigned_pickup": bool(self.assigned_pickup),
            "assigned_dropoff": bool(self.assigned_dropoff),
            "waiting_queue": list(self.waiting_queue),
            "assigned_pickup_request_ids": list(self.assigned_pickup_request_ids),
            "assigned_dropoff_request_ids": list(self.assigned_dropoff_request_ids),
            "onboard_destination_request_ids": list(self.onboard_destination_request_ids),
            "static_guard_status": self.static_guard_status.to_payload(),
            "dynamic_state_complete": bool(self.dynamic_state_complete),
            "state_complete": bool(self.state_complete),
            "missing_reason": list(self.missing_reason),
            "skip_allowed": self.skip_allowed,
        }


@dataclass(frozen=True)
class ScheduledTransition:
    event_ts: int
    sequence: int
    transition: str
    payload: Mapping[str, Any]


class ServiceObligationStateMachine:
    _SCHEDULABLE = {
        "passenger_waiting",
        "request_created",
        "request_assigned",
        "request_reassigned",
        "passenger_boarded",
        "passenger_alighted",
        "request_completed",
        "request_cancelled",
    }

    def __init__(self) -> None:
        self.passengers: Dict[str, Passenger] = {}
        self.requests: Dict[str, ServiceRequest] = {}
        self.stop_queues: Dict[str, StopQueue] = {}
        self.vehicle_ledgers: Dict[int, VehicleObligationLedger] = {}
        self.event_log: List[Dict[str, Any]] = []
        self.pending_transitions: List[ScheduledTransition] = []
        self.current_ts = 0
        self.state_complete = True
        self.missing_reason: List[str] = []
        self._event_sequence = 0
        self._schedule_sequence = 0

    def register_stop(self, stop_id: str) -> StopQueue:
        stop_key = _require_id("stop_id", stop_id)
        return self.stop_queues.setdefault(stop_key, StopQueue(stop_key))

    def register_vehicle(self, agent_id: int, vehicle_token: str) -> VehicleObligationLedger:
        agent = int(agent_id)
        token = _require_id("vehicle_token", vehicle_token)
        existing = self.vehicle_ledgers.get(agent)
        if existing is not None and existing.vehicle_token != token:
            raise KSafetyIdentityError(f"agent_id {agent} is already bound to vehicle_token {existing.vehicle_token}")
        if existing is None:
            existing = VehicleObligationLedger(agent, token)
            self.vehicle_ledgers[agent] = existing
        return existing

    def mark_state_incomplete(self, reason: str) -> None:
        reason_text = _require_id("missing_reason", reason)
        self.state_complete = False
        if reason_text not in self.missing_reason:
            self.missing_reason.append(reason_text)

    def _begin_transition(self, transition: str, event_ts: int) -> int:
        timestamp = int(event_ts)
        if timestamp < self.current_ts:
            raise KSafetyChronologyError(
                f"{transition} timestamp {timestamp} precedes current state timestamp {self.current_ts}"
            )
        self.current_ts = timestamp
        self._event_sequence += 1
        return self._event_sequence

    def _record(self, sequence: int, transition: str, event_ts: int, **identity: Any) -> None:
        self.event_log.append(
            {
                "event_sequence": int(sequence),
                "event_ts": int(event_ts),
                "transition": transition,
                **{str(key): value for key, value in identity.items()},
            }
        )

    def schedule_transition(self, transition: str, event_ts: int, **payload: Any) -> None:
        if transition not in self._SCHEDULABLE:
            raise KSafetyTransitionError(f"unsupported scheduled transition: {transition}")
        timestamp = int(event_ts)
        if timestamp < self.current_ts:
            raise KSafetyChronologyError("scheduled transition precedes current state timestamp")
        self._schedule_sequence += 1
        self.pending_transitions.append(ScheduledTransition(timestamp, self._schedule_sequence, transition, dict(payload)))

    def advance_to(self, decision_ts: int) -> None:
        timestamp = int(decision_ts)
        if timestamp < self.current_ts:
            raise KSafetyChronologyError(
                f"decision_ts {timestamp} precedes current state timestamp {self.current_ts}"
            )
        ordered = sorted(self.pending_transitions, key=lambda row: (row.event_ts, row.sequence))
        due = [row for row in ordered if row.event_ts <= timestamp]
        self.pending_transitions = [row for row in ordered if row.event_ts > timestamp]
        for row in due:
            method = getattr(self, row.transition)
            method(event_ts=row.event_ts, **dict(row.payload))
        self.current_ts = timestamp

    def passenger_waiting(
        self,
        *,
        passenger_id: str,
        pickup_stop: str,
        dropoff_stop: str,
        event_ts: int,
    ) -> Passenger:
        passenger_key = _require_id("passenger_id", passenger_id)
        pickup = _require_id("pickup_stop", pickup_stop)
        dropoff = _require_id("dropoff_stop", dropoff_stop)
        if passenger_key in self.passengers:
            raise KSafetyIdentityError(f"duplicate passenger_id: {passenger_key}")
        sequence = self._begin_transition("passenger_waiting", event_ts)
        passenger = Passenger(passenger_key, pickup, dropoff, PassengerStatus.WAITING, int(event_ts), int(event_ts))
        self.passengers[passenger_key] = passenger
        self.register_stop(pickup).waiting_queue.add(passenger_key)
        self.register_stop(dropoff)
        self._record(sequence, "passenger_waiting", event_ts, passenger_id=passenger_key, pickup_stop=pickup, dropoff_stop=dropoff)
        return passenger

    def request_created(
        self,
        *,
        request_id: str,
        passenger_id: str,
        event_ts: int,
        service_leg_id: Optional[str] = None,
    ) -> ServiceRequest:
        request_key = _require_id("request_id", request_id)
        passenger_key = _require_id("passenger_id", passenger_id)
        if request_key in self.requests:
            raise KSafetyIdentityError(f"duplicate request_id: {request_key}")
        passenger = self.passengers.get(passenger_key)
        if passenger is None or passenger.passenger_status != PassengerStatus.WAITING:
            raise KSafetyTransitionError("request_created requires a WAITING passenger")
        if passenger.active_request_id is not None:
            raise KSafetyTransitionError("passenger already has an active request")
        sequence = self._begin_transition("request_created", event_ts)
        leg = _require_id("service_leg_id", service_leg_id or f"leg:{request_key}")
        request = ServiceRequest(
            request_key,
            passenger_key,
            leg,
            passenger.pickup_stop,
            passenger.dropoff_stop,
            RequestStatus.CREATED,
            int(event_ts),
            int(event_ts),
        )
        self.requests[request_key] = request
        passenger.active_request_id = request_key
        passenger.last_transition_ts = int(event_ts)
        self._record(sequence, "request_created", event_ts, request_id=request_key, passenger_id=passenger_key, service_leg_id=leg)
        return request

    def request_assigned(
        self,
        *,
        request_id: str,
        agent_id: int,
        vehicle_token: str,
        event_ts: int,
        allow_reassignment: bool = False,
        transition_name: str = "request_assigned",
    ) -> ServiceRequest:
        request_key = _require_id("request_id", request_id)
        request = self.requests.get(request_key)
        if request is None:
            raise KSafetyIdentityError(f"unknown request_id: {request_key}")
        if request.request_status not in {RequestStatus.CREATED, RequestStatus.ASSIGNED}:
            raise KSafetyTransitionError("request assignment requires CREATED or ASSIGNED status")
        token = _require_id("vehicle_token", vehicle_token)
        existing_ledger = self.vehicle_ledgers.get(int(agent_id))
        if existing_ledger is not None and existing_ledger.vehicle_token != token:
            raise KSafetyIdentityError(
                f"agent_id {int(agent_id)} is already bound to vehicle_token {existing_ledger.vehicle_token}"
            )
        if request.request_status == RequestStatus.ASSIGNED and request.assigned_vehicle != token and not allow_reassignment:
            raise KSafetyTransitionError("vehicle change requires request_reassigned")
        sequence = self._begin_transition(transition_name, event_ts)
        ledger = self.register_vehicle(int(agent_id), token)
        if request.assigned_vehicle is not None and request.assigned_vehicle != token:
            for old_ledger in self.vehicle_ledgers.values():
                if old_ledger.vehicle_token == request.assigned_vehicle:
                    old_ledger.assigned_pickup.discard(request_key)
        request.assigned_vehicle = token
        request.request_status = RequestStatus.ASSIGNED
        request.last_transition_ts = int(event_ts)
        passenger = self.passengers[request.passenger_id]
        passenger.assigned_vehicle = token
        passenger.passenger_status = PassengerStatus.ASSIGNED
        passenger.last_transition_ts = int(event_ts)
        self.register_stop(request.pickup_stop).assigned_pickup.add(request_key)
        ledger.assigned_pickup.add(request_key)
        self._record(sequence, transition_name, event_ts, request_id=request_key, passenger_id=request.passenger_id, agent_id=int(agent_id), vehicle_token=token)
        return request

    def request_reassigned(
        self,
        *,
        request_id: str,
        agent_id: int,
        vehicle_token: str,
        event_ts: int,
    ) -> ServiceRequest:
        return self.request_assigned(
            request_id=request_id,
            agent_id=agent_id,
            vehicle_token=vehicle_token,
            event_ts=event_ts,
            allow_reassignment=True,
            transition_name="request_reassigned",
        )

    def passenger_boarded(
        self,
        *,
        request_id: str,
        passenger_id: str,
        agent_id: int,
        vehicle_token: str,
        stop_id: str,
        event_ts: int,
    ) -> ServiceRequest:
        request_key = _require_id("request_id", request_id)
        passenger_key = _require_id("passenger_id", passenger_id)
        token = _require_id("vehicle_token", vehicle_token)
        stop_key = _require_id("stop_id", stop_id)
        request = self.requests.get(request_key)
        passenger = self.passengers.get(passenger_key)
        ledger = self.vehicle_ledgers.get(int(agent_id))
        if request is None or passenger is None or ledger is None:
            raise KSafetyIdentityError("boarding identity is not registered")
        if request.passenger_id != passenger_key or request.request_status != RequestStatus.ASSIGNED:
            raise KSafetyTransitionError("boarding requires the matching ASSIGNED request")
        if request.assigned_vehicle != token or ledger.vehicle_token != token or request.pickup_stop != stop_key:
            raise KSafetyTransitionError("boarding vehicle or pickup stop does not match assignment")
        sequence = self._begin_transition("passenger_boarded", event_ts)
        queue = self.register_stop(stop_key)
        queue.waiting_queue.discard(passenger_key)
        queue.assigned_pickup.discard(request_key)
        ledger.assigned_pickup.discard(request_key)
        ledger.assigned_dropoff.add(request_key)
        ledger.onboard_destination[request_key] = request.dropoff_stop
        ledger.onboard_passenger_by_request[request_key] = passenger_key
        request.request_status = RequestStatus.BOARDED
        request.last_transition_ts = int(event_ts)
        passenger.passenger_status = PassengerStatus.ONBOARD
        passenger.last_transition_ts = int(event_ts)
        self._record(sequence, "passenger_boarded", event_ts, request_id=request_key, passenger_id=passenger_key, agent_id=int(agent_id), vehicle_token=token, stop_id=stop_key)
        return request

    def passenger_alighted(
        self,
        *,
        request_id: str,
        passenger_id: str,
        agent_id: int,
        vehicle_token: str,
        stop_id: str,
        event_ts: int,
    ) -> ServiceRequest:
        request_key = _require_id("request_id", request_id)
        passenger_key = _require_id("passenger_id", passenger_id)
        token = _require_id("vehicle_token", vehicle_token)
        stop_key = _require_id("stop_id", stop_id)
        request = self.requests.get(request_key)
        passenger = self.passengers.get(passenger_key)
        ledger = self.vehicle_ledgers.get(int(agent_id))
        if request is None or passenger is None or ledger is None:
            raise KSafetyIdentityError("alighting identity is not registered")
        if request.passenger_id != passenger_key or request.request_status != RequestStatus.BOARDED:
            raise KSafetyTransitionError("alighting requires the matching BOARDED request")
        if ledger.vehicle_token != token or request.assigned_vehicle != token or request.dropoff_stop != stop_key:
            raise KSafetyTransitionError("alighting vehicle or dropoff stop does not match onboard service leg")
        if ledger.onboard_passenger_by_request.get(request_key) != passenger_key:
            raise KSafetyTransitionError("onboard passenger identity chain is broken")
        sequence = self._begin_transition("passenger_alighted", event_ts)
        ledger.assigned_dropoff.discard(request_key)
        ledger.onboard_destination.pop(request_key, None)
        ledger.onboard_passenger_by_request.pop(request_key, None)
        request.request_status = RequestStatus.ALIGHTED
        request.last_transition_ts = int(event_ts)
        passenger.passenger_status = PassengerStatus.ALIGHTED
        passenger.last_transition_ts = int(event_ts)
        self._record(sequence, "passenger_alighted", event_ts, request_id=request_key, passenger_id=passenger_key, agent_id=int(agent_id), vehicle_token=token, stop_id=stop_key)
        return request

    def request_completed(self, *, request_id: str, event_ts: int) -> ServiceRequest:
        request_key = _require_id("request_id", request_id)
        request = self.requests.get(request_key)
        if request is None:
            raise KSafetyIdentityError(f"unknown request_id: {request_key}")
        if request.request_status != RequestStatus.ALIGHTED:
            raise KSafetyTransitionError("request_completed requires ALIGHTED status")
        sequence = self._begin_transition("request_completed", event_ts)
        passenger = self.passengers[request.passenger_id]
        request.request_status = RequestStatus.COMPLETED
        request.last_transition_ts = int(event_ts)
        passenger.passenger_status = PassengerStatus.SERVED
        passenger.active_request_id = None
        passenger.last_transition_ts = int(event_ts)
        self._record(sequence, "request_completed", event_ts, request_id=request_key, passenger_id=request.passenger_id)
        return request

    def request_cancelled(self, *, request_id: str, event_ts: int) -> ServiceRequest:
        request_key = _require_id("request_id", request_id)
        request = self.requests.get(request_key)
        if request is None:
            raise KSafetyIdentityError(f"unknown request_id: {request_key}")
        if request.request_status in {RequestStatus.BOARDED, RequestStatus.ALIGHTED, RequestStatus.COMPLETED}:
            raise KSafetyTransitionError("onboard, alighted, or completed request cannot be cancelled")
        sequence = self._begin_transition("request_cancelled", event_ts)
        passenger = self.passengers[request.passenger_id]
        queue = self.register_stop(request.pickup_stop)
        queue.waiting_queue.discard(passenger.passenger_id)
        queue.assigned_pickup.discard(request_key)
        for ledger in self.vehicle_ledgers.values():
            ledger.assigned_pickup.discard(request_key)
        request.request_status = RequestStatus.CANCELLED
        request.last_transition_ts = int(event_ts)
        passenger.passenger_status = PassengerStatus.CANCELLED
        passenger.active_request_id = None
        passenger.assigned_vehicle = None
        passenger.last_transition_ts = int(event_ts)
        self._record(sequence, "request_cancelled", event_ts, request_id=request_key, passenger_id=request.passenger_id)
        return request

    def audit_integrity(self) -> Dict[str, Any]:
        violations: List[str] = []
        service_leg_ids: Set[str] = set()
        for request_id, request in sorted(self.requests.items()):
            passenger = self.passengers.get(request.passenger_id)
            if passenger is None:
                violations.append(f"REQUEST_PASSENGER_MISSING:{request_id}")
            if request.service_leg_id in service_leg_ids:
                violations.append(f"DUPLICATE_SERVICE_LEG_ID:{request.service_leg_id}")
            service_leg_ids.add(request.service_leg_id)
            matching_ledgers = [
                ledger
                for ledger in self.vehicle_ledgers.values()
                if request_id in ledger.assigned_pickup
                or request_id in ledger.assigned_dropoff
                or request_id in ledger.onboard_destination
            ]
            if len(matching_ledgers) > 1:
                violations.append(f"REQUEST_ON_MULTIPLE_VEHICLES:{request_id}")
            if request.request_status == RequestStatus.ASSIGNED:
                pickup_queue = self.stop_queues.get(request.pickup_stop)
                if not matching_ledgers or pickup_queue is None or request_id not in pickup_queue.assigned_pickup:
                    violations.append(f"ASSIGNED_REQUEST_LEDGER_MISSING:{request_id}")
            if request.request_status == RequestStatus.BOARDED:
                if not matching_ledgers or request_id not in matching_ledgers[0].onboard_destination:
                    violations.append(f"BOARDED_REQUEST_LEDGER_MISSING:{request_id}")
            if request.request_status in {RequestStatus.ALIGHTED, RequestStatus.COMPLETED, RequestStatus.CANCELLED} and matching_ledgers:
                violations.append(f"CLOSED_REQUEST_LEDGER_REMAINS:{request_id}")
        for stop_id, queue in sorted(self.stop_queues.items()):
            for passenger_id in sorted(queue.waiting_queue):
                passenger = self.passengers.get(passenger_id)
                if passenger is None or passenger.pickup_stop != stop_id or passenger.passenger_status not in {PassengerStatus.WAITING, PassengerStatus.ASSIGNED}:
                    violations.append(f"WAITING_QUEUE_IDENTITY_MISMATCH:{stop_id}:{passenger_id}")
        return {
            "passed": not violations,
            "violation_count": len(violations),
            "violations": violations,
            "passenger_count": len(self.passengers),
            "request_count": len(self.requests),
            "stop_queue_count": len(self.stop_queues),
            "vehicle_ledger_count": len(self.vehicle_ledgers),
            "event_count": len(self.event_log),
            "pending_event_count": len(self.pending_transitions),
        }

    def snapshot_for_vehicle(
        self,
        *,
        agent_id: int,
        vehicle_token: str,
        decision_ts: int,
        current_stop_id: str,
        candidate_stop_id: Optional[str],
        static_guard_status: StaticGuardStatus,
    ) -> DecisionTimeObligationSnapshot:
        self.advance_to(int(decision_ts))
        agent = int(agent_id)
        token = _require_id("vehicle_token", vehicle_token)
        current_stop = _require_id("current_stop_id", current_stop_id)
        candidate = str(candidate_stop_id) if candidate_stop_id is not None else None
        missing: List[str] = list(self.missing_reason)
        ledger = self.vehicle_ledgers.get(agent)
        if ledger is None or ledger.vehicle_token != token:
            missing.append("VEHICLE_OBLIGATION_LEDGER_MISSING_OR_IDENTITY_MISMATCH")
        queue = self.stop_queues.get(candidate) if candidate is not None else None
        if queue is None:
            missing.append("CANDIDATE_STOP_QUEUE_MISSING")
        integrity = self.audit_integrity()
        if not integrity["passed"]:
            missing.append("STATE_INTEGRITY_AUDIT_FAILED")
        if ledger is not None and not ledger.state_complete:
            missing.extend(ledger.missing_reason or ("VEHICLE_LEDGER_INCOMPLETE",))
        if queue is not None and not queue.state_complete:
            missing.extend(queue.missing_reason or ("STOP_QUEUE_INCOMPLETE",))
        reasons = tuple(dict.fromkeys(missing))
        dynamic_complete = bool(self.state_complete and ledger is not None and queue is not None and not reasons)

        waiting_ids = tuple(sorted(queue.waiting_queue)) if queue is not None else ()
        assigned_pickup_ids: Tuple[str, ...] = ()
        assigned_dropoff_ids: Tuple[str, ...] = ()
        onboard_destination_ids: Tuple[str, ...] = ()
        if ledger is not None and candidate is not None:
            assigned_pickup_ids = tuple(
                sorted(
                    request_id
                    for request_id in ledger.assigned_pickup
                    if request_id in self.requests and self.requests[request_id].pickup_stop == candidate
                )
            )
            assigned_dropoff_ids = tuple(
                sorted(
                    request_id
                    for request_id in ledger.assigned_dropoff
                    if request_id in self.requests and self.requests[request_id].dropoff_stop == candidate
                )
            )
            onboard_destination_ids = tuple(
                sorted(request_id for request_id, stop_id in ledger.onboard_destination.items() if stop_id == candidate)
            )
        assigned_pickup = bool(assigned_pickup_ids)
        assigned_dropoff = bool(assigned_dropoff_ids)
        boarding = bool(waiting_ids or assigned_pickup_ids)
        onboard = bool(onboard_destination_ids)
        alighting = bool(assigned_dropoff_ids or onboard_destination_ids)
        pickup = bool(assigned_pickup or boarding)
        dropoff = bool(assigned_dropoff or alighting or onboard)
        service = bool(pickup or dropoff)
        return DecisionTimeObligationSnapshot(
            agent,
            token,
            int(decision_ts),
            current_stop,
            candidate,
            pickup,
            dropoff,
            boarding,
            alighting,
            onboard,
            service,
            assigned_pickup,
            assigned_dropoff,
            waiting_ids,
            assigned_pickup_ids,
            assigned_dropoff_ids,
            onboard_destination_ids,
            static_guard_status,
            dynamic_complete,
            bool(dynamic_complete and static_guard_status.state_complete),
            reasons,
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": K_SAFETY_STATE_SCHEMA_VERSION,
            "current_ts": int(self.current_ts),
            "state_complete": bool(self.state_complete),
            "missing_reason": list(self.missing_reason),
            "passengers": {key: row.to_payload() for key, row in sorted(self.passengers.items())},
            "requests": {key: row.to_payload() for key, row in sorted(self.requests.items())},
            "stop_queues": {key: row.to_payload() for key, row in sorted(self.stop_queues.items())},
            "vehicle_ledgers": {str(key): row.to_payload() for key, row in sorted(self.vehicle_ledgers.items())},
            "event_log": list(self.event_log),
            "pending_transitions": [
                {
                    "event_ts": row.event_ts,
                    "sequence": row.sequence,
                    "transition": row.transition,
                    "payload": dict(row.payload),
                }
                for row in sorted(self.pending_transitions, key=lambda item: (item.event_ts, item.sequence))
            ],
        }
