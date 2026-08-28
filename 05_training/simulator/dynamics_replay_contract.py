from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable


JSONScalar = Union[str, int, float, bool, None]
JSONValue = Union[JSONScalar, Mapping[str, "JSONValue"], Sequence["JSONValue"]]

DECISION_INTERVAL_SECONDS = 60
THIRTY_MINUTE_TOTAL_FRAMES = 30


class ReplayContractError(ValueError):
    pass


class ReplayEventType(str, Enum):
    PASSENGER_ARRIVAL = "PASSENGER_ARRIVAL"
    PICKUP_REQUEST = "PICKUP_REQUEST"
    PASSENGER_DESTINATION = "PASSENGER_DESTINATION"
    PASSENGER_CANCELLATION = "PASSENGER_CANCELLATION"
    TRAVEL_TIME_UPDATE = "TRAVEL_TIME_UPDATE"
    SIGNAL_STATE_UPDATE = "SIGNAL_STATE_UPDATE"
    ROAD_STATE_UPDATE = "ROAD_STATE_UPDATE"
    ROUTE_AVAILABILITY_UPDATE = "ROUTE_AVAILABILITY_UPDATE"
    SCHEDULE_UPDATE = "SCHEDULE_UPDATE"
    OPERATION_MODE_UPDATE = "OPERATION_MODE_UPDATE"


REQUIRED_FIELDS_BY_EVENT_TYPE: Dict[ReplayEventType, Tuple[str, ...]] = {
    ReplayEventType.PASSENGER_ARRIVAL: ("passenger_id", "stop_id", "event_timestamp_seconds"),
    ReplayEventType.PICKUP_REQUEST: ("request_id", "passenger_id", "stop_id"),
    ReplayEventType.PASSENGER_DESTINATION: ("passenger_id", "destination_stop_id"),
    ReplayEventType.PASSENGER_CANCELLATION: ("request_id", "passenger_id"),
    ReplayEventType.TRAVEL_TIME_UPDATE: ("event_timestamp_seconds",),
    ReplayEventType.SIGNAL_STATE_UPDATE: ("event_timestamp_seconds",),
    ReplayEventType.ROAD_STATE_UPDATE: ("event_timestamp_seconds",),
    ReplayEventType.ROUTE_AVAILABILITY_UPDATE: ("route_id",),
    ReplayEventType.SCHEDULE_UPDATE: ("route_id",),
    ReplayEventType.OPERATION_MODE_UPDATE: ("event_timestamp_seconds",),
}


def _clean_json(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _clean_json(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ReplayContractError("non-finite float is not allowed in replay canonical JSON")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise ReplayContractError(f"unsupported replay JSON value type: {type(value).__name__}")


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_clean_json(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReplayEvent:
    event_id: str
    event_timestamp_seconds: int
    event_type: ReplayEventType
    vehicle_id: Optional[str] = None
    agent_id: Optional[int] = None
    route_id: Optional[str] = None
    stop_id: Optional[str] = None
    passenger_id: Optional[str] = None
    request_id: Optional[str] = None
    destination_stop_id: Optional[str] = None
    payload: Mapping[str, JSONScalar] = field(default_factory=dict)
    source_path: Optional[str] = None
    source_row_hash: Optional[str] = None

    def __post_init__(self) -> None:
        validate_replay_event(self)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_timestamp_seconds": self.event_timestamp_seconds,
            "event_type": self.event_type.value,
            "vehicle_id": self.vehicle_id,
            "agent_id": self.agent_id,
            "route_id": self.route_id,
            "stop_id": self.stop_id,
            "passenger_id": self.passenger_id,
            "request_id": self.request_id,
            "destination_stop_id": self.destination_stop_id,
            "payload": dict(self.payload),
            "source_path": self.source_path,
            "source_row_hash": self.source_row_hash,
        }

    @property
    def event_hash(self) -> str:
        return canonical_hash(self.to_payload())


def validate_replay_event(event: ReplayEvent) -> None:
    if not event.event_id:
        raise ReplayContractError("event_id is required")
    if int(event.event_timestamp_seconds) < 0:
        raise ReplayContractError("event_timestamp_seconds must be non-negative")
    if not isinstance(event.event_type, ReplayEventType):
        raise ReplayContractError("event_type must be ReplayEventType")
    if event.event_type == ReplayEventType.TRAVEL_TIME_UPDATE:
        has_edge = "edge_id" in event.payload or "travel_time_seconds" in event.payload or "updated_travel_time_seconds" in event.payload
        if not event.route_id and not has_edge:
            raise ReplayContractError("TRAVEL_TIME_UPDATE requires route_id or edge/travel-time payload")
    missing = []
    for field_name in REQUIRED_FIELDS_BY_EVENT_TYPE.get(event.event_type, ()):
        if getattr(event, field_name, None) is None:
            missing.append(field_name)
    if missing:
        raise ReplayContractError(f"{event.event_type.value} missing required fields: {','.join(missing)}")
    _clean_json(dict(event.payload))


def event_sort_key(event: ReplayEvent) -> Tuple[Any, ...]:
    return (
        int(event.event_timestamp_seconds),
        event.event_type.value,
        event.route_id or "",
        event.stop_id or "",
        event.passenger_id or "",
        event.event_id,
    )


@dataclass(frozen=True)
class ReplayFrame:
    step_index: int
    frame_start_seconds: int
    frame_end_seconds: int
    events: Tuple[ReplayEvent, ...]

    def __post_init__(self) -> None:
        if self.frame_end_seconds <= self.frame_start_seconds:
            raise ReplayContractError("frame_end_seconds must be greater than frame_start_seconds")
        ordered = tuple(sorted(self.events, key=event_sort_key))
        object.__setattr__(self, "events", ordered)
        for event in ordered:
            if event.event_timestamp_seconds < self.frame_start_seconds or event.event_timestamp_seconds >= self.frame_end_seconds:
                raise ReplayContractError("ReplayEvent timestamp is outside the frame interval")

    @property
    def canonical_event_order(self) -> Tuple[str, ...]:
        return tuple(event.event_id for event in self.events)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "frame_start_seconds": self.frame_start_seconds,
            "frame_end_seconds": self.frame_end_seconds,
            "events": [event.to_payload() for event in self.events],
            "canonical_event_order": list(self.canonical_event_order),
        }

    @property
    def frame_hash(self) -> str:
        return canonical_hash(self.to_payload())


@dataclass(frozen=True)
class ReplayCursor:
    frame_index: int
    event_offset: int = 0

    def to_payload(self) -> Dict[str, int]:
        return {"frame_index": int(self.frame_index), "event_offset": int(self.event_offset)}

    @property
    def cursor_hash(self) -> str:
        return canonical_hash(self.to_payload())


@dataclass(frozen=True)
class ReplayInput:
    replay_start_timestamp: int
    replay_end_timestamp: int
    frames: Tuple[ReplayFrame, ...]
    source_manifest_hash: str
    decision_interval_seconds: int = DECISION_INTERVAL_SECONDS
    total_frames: int = THIRTY_MINUTE_TOTAL_FRAMES

    def __post_init__(self) -> None:
        validate_replay_input(self)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "replay_start_timestamp": self.replay_start_timestamp,
            "replay_end_timestamp": self.replay_end_timestamp,
            "decision_interval_seconds": self.decision_interval_seconds,
            "total_frames": self.total_frames,
            "frames": [frame.to_payload() for frame in self.frames],
            "event_stream_hash": self.event_stream_hash,
            "source_manifest_hash": self.source_manifest_hash,
        }

    @property
    def event_stream_hash(self) -> str:
        return canonical_hash({"frames": [frame.to_payload() for frame in self.frames]})


def validate_replay_input(replay_input: ReplayInput) -> None:
    if replay_input.decision_interval_seconds != DECISION_INTERVAL_SECONDS:
        raise ReplayContractError("decision_interval_seconds must be 60")
    if replay_input.total_frames != THIRTY_MINUTE_TOTAL_FRAMES:
        raise ReplayContractError("total_frames must be 30")
    if len(replay_input.frames) != replay_input.total_frames:
        raise ReplayContractError("ReplayInput frame count must equal total_frames")
    seen_event_ids = set()
    expected_start = int(replay_input.replay_start_timestamp)
    for expected_index, frame in enumerate(replay_input.frames):
        if frame.step_index != expected_index:
            raise ReplayContractError("ReplayFrame step_index sequence is not canonical")
        if frame.frame_start_seconds != expected_start:
            raise ReplayContractError("ReplayFrame gap or overlap detected")
        if frame.frame_end_seconds - frame.frame_start_seconds != replay_input.decision_interval_seconds:
            raise ReplayContractError("ReplayFrame duration must be 60 seconds")
        for event in frame.events:
            if event.event_id in seen_event_ids:
                raise ReplayContractError(f"duplicate ReplayEvent event_id: {event.event_id}")
            seen_event_ids.add(event.event_id)
        expected_start = frame.frame_end_seconds
    if expected_start != replay_input.replay_end_timestamp:
        raise ReplayContractError("ReplayInput end timestamp does not match frame sequence")


@dataclass(frozen=True)
class ExternalProviderState:
    provider_name: str
    provider_state_classification: str
    canonical_state: Mapping[str, JSONValue]

    def to_payload(self) -> Dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "provider_state_classification": self.provider_state_classification,
            "canonical_state": dict(self.canonical_state),
        }

    @property
    def provider_state_hash(self) -> str:
        return canonical_hash(self.to_payload())


@runtime_checkable
class ExternalProviderStateProtocol(Protocol):
    def export_provider_state(self) -> ExternalProviderState:
        ...

    def import_provider_state(self, state: ExternalProviderState) -> None:
        ...

    def clone_provider(self) -> "ExternalProviderStateProtocol":
        ...

    def provider_state_hash(self) -> str:
        ...
