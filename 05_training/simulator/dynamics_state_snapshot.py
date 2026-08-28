from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from simulator.dynamics_replay_contract import canonical_hash, canonical_json


SNAPSHOT_SCHEMA_VERSION = "SUSEONG_DYNAMICS_STATE_SNAPSHOT_V1"

REQUIRED_SNAPSHOT_FIELDS: Tuple[str, ...] = (
    "schema_version",
    "simulation_timestamp_seconds",
    "vehicles",
    "routes",
    "waiting_passengers",
    "assigned_pickups",
    "assigned_dropoffs",
    "onboard_passengers",
    "mandatory_stop_state",
    "action_mask_state",
    "schedule_state",
    "headway_state",
    "operation_mode",
    "shared_counters",
    "replay_cursor",
    "external_provider_states",
)


class DynamicsStateError(ValueError):
    pass


class MissingDynamicsStateFieldError(DynamicsStateError):
    pass


class UnknownDynamicsStateFieldError(DynamicsStateError):
    pass


@dataclass(frozen=True)
class DynamicsStateSnapshot:
    canonical_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        validate_snapshot_payload(self.canonical_payload)

    @property
    def schema_version(self) -> str:
        return str(self.canonical_payload["schema_version"])

    @property
    def canonical_json_sha256(self) -> str:
        return canonical_hash(dict(self.canonical_payload))

    @property
    def state_hash(self) -> str:
        return self.canonical_json_sha256

    def to_payload(self) -> Dict[str, Any]:
        return copy.deepcopy(dict(self.canonical_payload))


def validate_snapshot_payload(payload: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_SNAPSHOT_FIELDS if field not in payload]
    if missing:
        raise MissingDynamicsStateFieldError(f"missing required dynamics state fields: {','.join(missing)}")
    unknown = sorted(set(payload) - set(REQUIRED_SNAPSHOT_FIELDS))
    if unknown:
        raise UnknownDynamicsStateFieldError(f"unknown dynamics state fields: {','.join(unknown)}")
    if payload["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise DynamicsStateError(f"unsupported schema_version: {payload['schema_version']}")
    canonical_json(dict(payload))


def capture_dynamics_state(**fields: Any) -> DynamicsStateSnapshot:
    return DynamicsStateSnapshot(copy.deepcopy(dict(fields)))


def serialize_dynamics_state(snapshot: DynamicsStateSnapshot) -> str:
    return canonical_json(snapshot.to_payload())


def deserialize_dynamics_state(serialized: str) -> DynamicsStateSnapshot:
    import json

    payload = json.loads(serialized)
    if not isinstance(payload, Mapping):
        raise DynamicsStateError("serialized dynamics state must decode to a mapping")
    return DynamicsStateSnapshot(copy.deepcopy(dict(payload)))


def clone_dynamics_state(snapshot: DynamicsStateSnapshot) -> DynamicsStateSnapshot:
    return DynamicsStateSnapshot(copy.deepcopy(snapshot.to_payload()))


def restore_dynamics_state(snapshot: DynamicsStateSnapshot) -> Dict[str, Any]:
    validate_snapshot_payload(snapshot.canonical_payload)
    return copy.deepcopy(snapshot.to_payload())


def build_runtime_state_from_snapshot(snapshot: DynamicsStateSnapshot) -> Dict[str, Any]:
    return restore_dynamics_state(snapshot)


def reset_runtime_state_from_snapshot(snapshot: DynamicsStateSnapshot) -> Dict[str, Any]:
    restored = restore_dynamics_state(snapshot)
    restored_hash = hash_dynamics_state(DynamicsStateSnapshot(restored))
    if restored_hash != snapshot.state_hash:
        raise DynamicsStateError("reset_from_snapshot hash mismatch")
    return {
        "runtime_state": restored,
        "expected_state_hash": snapshot.state_hash,
        "restored_state_hash": restored_hash,
        "hash_match": True,
    }


def hash_dynamics_state(snapshot: DynamicsStateSnapshot) -> str:
    return snapshot.state_hash
