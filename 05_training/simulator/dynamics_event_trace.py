from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from simulator.dynamics_replay_contract import canonical_hash


class DynamicsEventError(ValueError):
    pass


class DynamicsEventType(str, Enum):
    VEHICLE_MOVE = "VEHICLE_MOVE"
    VEHICLE_DWELL = "VEHICLE_DWELL"
    STOP_ARRIVAL = "STOP_ARRIVAL"
    PASSENGER_ARRIVAL = "PASSENGER_ARRIVAL"
    PASSENGER_BOARD = "PASSENGER_BOARD"
    PASSENGER_ALIGHT = "PASSENGER_ALIGHT"
    PASSENGER_MISSED_PICKUP = "PASSENGER_MISSED_PICKUP"
    PASSENGER_MISSED_DROPOFF = "PASSENGER_MISSED_DROPOFF"
    CONDITIONAL_SKIP = "CONDITIONAL_SKIP"
    INVALID_SKIP = "INVALID_SKIP"
    MANDATORY_STOP_VIOLATION = "MANDATORY_STOP_VIOLATION"
    ROUTE_TURNAROUND = "ROUTE_TURNAROUND"
    SERVICE_COMPLETED = "SERVICE_COMPLETED"


@dataclass(frozen=True)
class PassengerWaitEvent:
    passenger_id: str
    arrival_timestamp_seconds: int
    service_timestamp_seconds: int
    wait_seconds: float
    route_id: str
    stop_id: str
    vehicle_id: str

    def __post_init__(self) -> None:
        if not self.passenger_id:
            raise DynamicsEventError("passenger_id is required for passenger wait event")
        if self.service_timestamp_seconds < self.arrival_timestamp_seconds:
            raise DynamicsEventError("service_timestamp_seconds must be >= arrival_timestamp_seconds")
        expected_wait = float(self.service_timestamp_seconds - self.arrival_timestamp_seconds)
        if abs(float(self.wait_seconds) - expected_wait) > 1e-9:
            raise DynamicsEventError("wait_seconds must equal service-arrival timestamp difference")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "passenger_id": self.passenger_id,
            "arrival_timestamp_seconds": int(self.arrival_timestamp_seconds),
            "service_timestamp_seconds": int(self.service_timestamp_seconds),
            "wait_seconds": float(self.wait_seconds),
            "route_id": self.route_id,
            "stop_id": self.stop_id,
            "vehicle_id": self.vehicle_id,
        }


@dataclass(frozen=True)
class DynamicsEvent:
    event_id: str
    event_timestamp_seconds: int
    step_index: int
    event_type: DynamicsEventType
    agent_id: Optional[int] = None
    vehicle_id: Optional[str] = None
    route_id: Optional[str] = None
    stop_id: Optional[str] = None
    passenger_id: Optional[str] = None
    request_id: Optional[str] = None
    source_trace_hash: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id:
            raise DynamicsEventError("event_id is required")
        if self.step_index < 0:
            raise DynamicsEventError("step_index must be non-negative")

    def to_payload(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_timestamp_seconds": int(self.event_timestamp_seconds),
            "step_index": int(self.step_index),
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "vehicle_id": self.vehicle_id,
            "route_id": self.route_id,
            "stop_id": self.stop_id,
            "passenger_id": self.passenger_id,
            "request_id": self.request_id,
            "source_trace_hash": self.source_trace_hash,
            "metadata": dict(self.metadata),
        }

    @property
    def event_hash(self) -> str:
        return canonical_hash(self.to_payload())


@dataclass(frozen=True)
class CanonicalStopServiceResult:
    served_passenger_count: int
    boarded_passenger_ids: Tuple[str, ...] = ()
    alighted_passenger_ids: Tuple[str, ...] = ()
    missed_pickup_ids: Tuple[str, ...] = ()
    missed_dropoff_ids: Tuple[str, ...] = ()
    passenger_wait_events: Tuple[PassengerWaitEvent, ...] = ()
    mandatory_stop_violation: bool = False
    invalid_skip_execution: bool = False
    passenger_level_wait_available: bool = False
    unavailable_reason: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        return {
            "served_passenger_count": int(self.served_passenger_count),
            "boarded_passenger_ids": list(self.boarded_passenger_ids),
            "alighted_passenger_ids": list(self.alighted_passenger_ids),
            "missed_pickup_ids": list(self.missed_pickup_ids),
            "missed_dropoff_ids": list(self.missed_dropoff_ids),
            "passenger_wait_events": [event.to_payload() for event in self.passenger_wait_events],
            "mandatory_stop_violation": bool(self.mandatory_stop_violation),
            "invalid_skip_execution": bool(self.invalid_skip_execution),
            "passenger_level_wait_available": bool(self.passenger_level_wait_available),
            "unavailable_reason": self.unavailable_reason,
        }


def adapt_stop_service_result(result: Any) -> CanonicalStopServiceResult:
    metadata = dict(getattr(result, "metadata", {}) or {})
    boarded = tuple(str(item) for item in metadata.get("boarded_passenger_ids", ()))
    alighted = tuple(str(item) for item in metadata.get("alighted_passenger_ids", ()))
    missed_pickup = tuple(str(item) for item in metadata.get("missed_pickup_ids", ()))
    missed_dropoff = tuple(str(item) for item in metadata.get("missed_dropoff_ids", ()))
    wait_payloads: Sequence[Mapping[str, Any]] = metadata.get("passenger_wait_events", ()) or ()
    wait_events = []
    for payload in wait_payloads:
        required = ["passenger_id", "arrival_timestamp_seconds", "service_timestamp_seconds", "route_id", "stop_id", "vehicle_id"]
        if any(key not in payload or payload[key] is None for key in required):
            continue
        arrival = int(payload["arrival_timestamp_seconds"])
        service = int(payload["service_timestamp_seconds"])
        wait_events.append(PassengerWaitEvent(
            passenger_id=str(payload["passenger_id"]),
            arrival_timestamp_seconds=arrival,
            service_timestamp_seconds=service,
            wait_seconds=float(service - arrival),
            route_id=str(payload["route_id"]),
            stop_id=str(payload["stop_id"]),
            vehicle_id=str(payload["vehicle_id"]),
        ))
    served_count = int(getattr(result, "boardings", 0) or 0) + int(getattr(result, "alightings", 0) or 0)
    wait_available = bool(wait_events)
    reason = None if wait_available else "PASSENGER_LEVEL_WAIT_DISTRIBUTION_UNAVAILABLE"
    return CanonicalStopServiceResult(
        served_passenger_count=served_count,
        boarded_passenger_ids=boarded,
        alighted_passenger_ids=alighted,
        missed_pickup_ids=missed_pickup,
        missed_dropoff_ids=missed_dropoff,
        passenger_wait_events=tuple(wait_events),
        mandatory_stop_violation=bool(metadata.get("mandatory_stop_violation", False)),
        invalid_skip_execution=bool(metadata.get("invalid_skip_execution", False)),
        passenger_level_wait_available=wait_available,
        unavailable_reason=reason,
    )


def event_trace_hash(events: Sequence[DynamicsEvent]) -> str:
    ordered = sorted(events, key=lambda event: (event.step_index, event.event_timestamp_seconds, event.event_type.value, event.event_id))
    return canonical_hash({"events": [_strip_runtime_identity(event.to_payload()) for event in ordered]})


def _strip_runtime_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_runtime_identity(item)
            for key, item in value.items()
            if str(key) not in {"evaluation_run_id", "execution_instance_id", "runtime_record_key"}
        }
    if isinstance(value, (list, tuple)):
        return [_strip_runtime_identity(item) for item in value]
    return value
