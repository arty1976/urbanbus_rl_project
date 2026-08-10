from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from simulator.dynamics_replay_contract import canonical_hash, canonical_json
from simulator.dynamics_state_snapshot import (
    SNAPSHOT_SCHEMA_VERSION,
    DynamicsStateSnapshot,
    capture_dynamics_state,
)
from simulator.k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime, RuntimeVersionBinding
from simulator.k_safety_state import (
    K_SAFETY_STATE_SCHEMA_VERSION,
    Passenger,
    PassengerStatus,
    RequestStatus,
    ScheduledTransition,
    ServiceObligationStateMachine,
    ServiceRequest,
    StopQueue,
    VehicleObligationLedger,
)


GLOBAL_K_MASK_SNAPSHOT_SCHEMA_VERSION = "SUSEONG_GLOBAL_K_MASK_SNAPSHOT_V1"


class KMaskSnapshotLifecycleError(ValueError):
    pass


class KMaskSnapshotVersionError(KMaskSnapshotLifecycleError):
    pass


class KMaskSnapshotInvariantError(KMaskSnapshotLifecycleError):
    pass


@dataclass(frozen=True)
class KMaskAgentContext:
    agent_id: int
    vehicle_token: str
    active_bus_mask: bool
    decision_ts: int
    obligation_state_machine: ServiceObligationStateMachine
    route_id: Optional[str] = None
    direction_id: Optional[str] = None
    stop_sequence: Optional[int] = None
    stop_id: Optional[str] = None
    route_stop_occurrence_id: Optional[str] = None
    identity_transition_type: Optional[str] = None


@dataclass(frozen=True)
class GlobalKMaskSnapshot:
    canonical_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        validate_global_k_mask_snapshot(self.canonical_payload)

    @property
    def snapshot_hash(self) -> str:
        return canonical_hash(dict(self.canonical_payload))

    def to_payload(self) -> Dict[str, Any]:
        return copy.deepcopy(dict(self.canonical_payload))


ROOT_FIELDS: Tuple[str, ...] = (
    "schema_version",
    "cycle_index",
    "cycle_timestamp",
    "agent_order",
    "agent_snapshots",
    "k_safety_state_version",
    "static_rulebook_version",
    "static_rulebook_sha256",
    "occurrence_master_sha256",
    "dynamic_state_contract_version",
    "mask_predicate_version",
    "experiment_version",
    "policy_execution_count",
)

AGENT_FIELDS: Tuple[str, ...] = (
    "agent_id",
    "vehicle_token",
    "active_bus_mask",
    "route_stop_occurrence_id",
    "route_id",
    "direction_id",
    "stop_sequence",
    "stop_id",
    "decision_ts",
    "k_action_mask",
    "hold_valid",
    "serve_move_valid",
    "conditional_skip_valid",
    "skip_invalid_reason_codes",
    "actor_sampling_bypassed",
    "occurrence_lookup_integrity",
    "identity_failure",
    "decision_time_obligation_snapshot",
    "service_obligation_state",
    "service_obligation_state_hash",
    "k_safety_state_version",
    "static_rulebook_version",
    "static_rulebook_sha256",
    "occurrence_master_sha256",
    "dynamic_state_contract_version",
    "mask_predicate_version",
    "experiment_version",
    "identity_transition_type",
    "policy_execution_count",
)


def _require_exact_fields(payload: Mapping[str, Any], required: Sequence[str], scope: str) -> None:
    missing = sorted(set(required) - set(payload))
    unknown = sorted(set(payload) - set(required))
    if missing:
        raise KMaskSnapshotInvariantError(f"{scope} missing fields: {','.join(missing)}")
    if unknown:
        raise KMaskSnapshotInvariantError(f"{scope} unknown fields: {','.join(unknown)}")


def validate_global_k_mask_snapshot(payload: Mapping[str, Any]) -> None:
    _require_exact_fields(payload, ROOT_FIELDS, "global K-mask snapshot")
    if payload["schema_version"] != GLOBAL_K_MASK_SNAPSHOT_SCHEMA_VERSION:
        raise KMaskSnapshotVersionError(f"unsupported global K-mask snapshot schema: {payload['schema_version']}")
    order = tuple(int(value) for value in payload["agent_order"])
    if order != tuple(range(8)):
        raise KMaskSnapshotInvariantError("global K-mask snapshot requires fixed agent order 0..7")
    rows = list(payload["agent_snapshots"])
    if len(rows) != 8:
        raise KMaskSnapshotInvariantError("global K-mask snapshot requires exactly eight agent rows")
    for expected_agent, raw in enumerate(rows):
        row = dict(raw)
        _require_exact_fields(row, AGENT_FIELDS, f"agent snapshot {expected_agent}")
        if int(row["agent_id"]) != expected_agent:
            raise KMaskSnapshotInvariantError("agent snapshot order or slot identity changed")
        mask = row["k_action_mask"]
        if not isinstance(mask, list) or len(mask) != 3 or any(not isinstance(value, bool) for value in mask):
            raise KMaskSnapshotInvariantError("K-action-mask must contain exactly three booleans")
        active = bool(row["active_bus_mask"])
        if active and not any(mask):
            raise KMaskSnapshotInvariantError("active agent cannot have an all-false K-action-mask")
        if not active and any(mask):
            raise KMaskSnapshotInvariantError("inactive agent cannot expose an action")
        if bool(row["actor_sampling_bypassed"]) is active:
            raise KMaskSnapshotInvariantError("actor sampling bypass must equal not active_bus_mask")
        if int(row["policy_execution_count"]) != 0:
            raise KMaskSnapshotInvariantError("snapshot lifecycle cannot execute a policy")
        for field in (
            "k_safety_state_version",
            "static_rulebook_version",
            "static_rulebook_sha256",
            "occurrence_master_sha256",
            "dynamic_state_contract_version",
            "mask_predicate_version",
            "experiment_version",
        ):
            if row[field] != payload[field]:
                raise KMaskSnapshotVersionError(f"agent snapshot {field} differs from global binding")
        if canonical_hash(dict(row["service_obligation_state"])) != row["service_obligation_state_hash"]:
            raise KMaskSnapshotInvariantError("service-obligation state hash mismatch")
    if int(payload["policy_execution_count"]) != 0:
        raise KMaskSnapshotInvariantError("global snapshot policy_execution_count must be zero")
    canonical_json(dict(payload))


def serialize_global_k_mask_snapshot(snapshot: GlobalKMaskSnapshot) -> str:
    return canonical_json(snapshot.to_payload())


def deserialize_global_k_mask_snapshot(serialized: str) -> GlobalKMaskSnapshot:
    payload = json.loads(serialized)
    if not isinstance(payload, Mapping):
        raise KMaskSnapshotLifecycleError("serialized global K-mask snapshot must decode to a mapping")
    return GlobalKMaskSnapshot(copy.deepcopy(dict(payload)))


def clone_global_k_mask_snapshot(snapshot: GlobalKMaskSnapshot) -> GlobalKMaskSnapshot:
    return GlobalKMaskSnapshot(copy.deepcopy(snapshot.to_payload()))


def bind_global_k_mask_to_dynamics_state(snapshot: GlobalKMaskSnapshot) -> DynamicsStateSnapshot:
    payload = snapshot.to_payload()
    agent_rows = payload["agent_snapshots"]
    vehicles = {
        str(row["agent_id"]): {
            "agent_id": int(row["agent_id"]),
            "vehicle_token": str(row["vehicle_token"]),
            "active_bus_mask": bool(row["active_bus_mask"]),
            "route_stop_occurrence_id": row["route_stop_occurrence_id"],
            "decision_ts": int(row["decision_ts"]),
        }
        for row in agent_rows
    }
    routes = {
        str(row["agent_id"]): {
            "route_id": row["route_id"],
            "direction_id": row["direction_id"],
            "stop_sequence": row["stop_sequence"],
            "stop_id": row["stop_id"],
            "route_stop_occurrence_id": row["route_stop_occurrence_id"],
        }
        for row in agent_rows
    }
    obligation_rows = {
        str(row["agent_id"]): copy.deepcopy(row["decision_time_obligation_snapshot"])
        for row in agent_rows
        if row["decision_time_obligation_snapshot"] is not None
    }
    return capture_dynamics_state(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        simulation_timestamp_seconds=max(int(row["decision_ts"]) for row in agent_rows),
        vehicles=vehicles,
        routes=routes,
        waiting_passengers={key: value.get("waiting_queue", []) for key, value in obligation_rows.items()},
        assigned_pickups={key: value.get("assigned_pickup_request_ids", []) for key, value in obligation_rows.items()},
        assigned_dropoffs={key: value.get("assigned_dropoff_request_ids", []) for key, value in obligation_rows.items()},
        onboard_passengers={key: value.get("onboard_destination_request_ids", []) for key, value in obligation_rows.items()},
        mandatory_stop_state={key: value.get("static_guard_status", {}) for key, value in obligation_rows.items()},
        action_mask_state={
            "global_k_mask_snapshot": payload,
            "global_k_mask_snapshot_hash": snapshot.snapshot_hash,
        },
        schedule_state={},
        headway_state={},
        operation_mode="PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATION_ONLY",
        shared_counters={"policy_execution_count": 0},
        replay_cursor={"cycle_index": int(payload["cycle_index"]), "event_offset": 0},
        external_provider_states={
            "k_safety_state_version": payload["k_safety_state_version"],
            "static_rulebook_version": payload["static_rulebook_version"],
            "static_rulebook_sha256": payload["static_rulebook_sha256"],
            "occurrence_master_sha256": payload["occurrence_master_sha256"],
            "dynamic_state_contract_version": payload["dynamic_state_contract_version"],
            "mask_predicate_version": payload["mask_predicate_version"],
            "experiment_version": payload["experiment_version"],
        },
    )


def extract_global_k_mask_from_dynamics_state(snapshot: DynamicsStateSnapshot) -> GlobalKMaskSnapshot:
    action_mask_state = dict(snapshot.to_payload()["action_mask_state"])
    payload = action_mask_state.get("global_k_mask_snapshot")
    expected_hash = action_mask_state.get("global_k_mask_snapshot_hash")
    if not isinstance(payload, Mapping):
        raise KMaskSnapshotInvariantError("dynamics action_mask_state does not contain a global K-mask snapshot")
    restored = GlobalKMaskSnapshot(copy.deepcopy(dict(payload)))
    if restored.snapshot_hash != expected_hash:
        raise KMaskSnapshotInvariantError("dynamics action_mask_state global K-mask hash mismatch")
    return restored


def restore_service_obligation_state(payload: Mapping[str, Any]) -> ServiceObligationStateMachine:
    if payload.get("schema_version") != K_SAFETY_STATE_SCHEMA_VERSION:
        raise KMaskSnapshotVersionError("service-obligation state schema mismatch")
    state = ServiceObligationStateMachine()
    state.current_ts = int(payload["current_ts"])
    state.state_complete = bool(payload["state_complete"])
    state.missing_reason = [str(value) for value in payload.get("missing_reason", ())]
    state.passengers = {
        str(key): Passenger(
            passenger_id=str(row["passenger_id"]),
            pickup_stop=str(row["pickup_stop"]),
            dropoff_stop=str(row["dropoff_stop"]),
            passenger_status=PassengerStatus(str(row["passenger_status"])),
            created_ts=int(row["created_ts"]),
            last_transition_ts=int(row["last_transition_ts"]),
            assigned_vehicle=str(row["assigned_vehicle"]) if row.get("assigned_vehicle") is not None else None,
            active_request_id=str(row["active_request_id"]) if row.get("active_request_id") is not None else None,
        )
        for key, row in dict(payload.get("passengers", {})).items()
    }
    state.requests = {
        str(key): ServiceRequest(
            request_id=str(row["request_id"]),
            passenger_id=str(row["passenger_id"]),
            service_leg_id=str(row["service_leg_id"]),
            pickup_stop=str(row["pickup_stop"]),
            dropoff_stop=str(row["dropoff_stop"]),
            request_status=RequestStatus(str(row["request_status"])),
            created_ts=int(row["created_ts"]),
            last_transition_ts=int(row["last_transition_ts"]),
            assigned_vehicle=str(row["assigned_vehicle"]) if row.get("assigned_vehicle") is not None else None,
        )
        for key, row in dict(payload.get("requests", {})).items()
    }
    state.stop_queues = {
        str(key): StopQueue(
            stop_id=str(row["stop_id"]),
            waiting_queue={str(value) for value in row.get("waiting_queue", ())},
            assigned_pickup={str(value) for value in row.get("assigned_pickup", ())},
            state_complete=bool(row["state_complete"]),
            missing_reason=tuple(str(value) for value in row.get("missing_reason", ())),
        )
        for key, row in dict(payload.get("stop_queues", {})).items()
    }
    state.vehicle_ledgers = {
        int(key): VehicleObligationLedger(
            agent_id=int(row["agent_id"]),
            vehicle_token=str(row["vehicle_token"]),
            assigned_pickup={str(value) for value in row.get("assigned_pickup", ())},
            assigned_dropoff={str(value) for value in row.get("assigned_dropoff", ())},
            onboard_destination={str(k): str(v) for k, v in dict(row.get("onboard_destination", {})).items()},
            onboard_passenger_by_request={str(k): str(v) for k, v in dict(row.get("onboard_passenger_by_request", {})).items()},
            state_complete=bool(row["state_complete"]),
            missing_reason=tuple(str(value) for value in row.get("missing_reason", ())),
        )
        for key, row in dict(payload.get("vehicle_ledgers", {})).items()
    }
    state.event_log = copy.deepcopy(list(payload.get("event_log", ())))
    state.pending_transitions = [
        ScheduledTransition(
            int(row["event_ts"]),
            int(row["sequence"]),
            str(row["transition"]),
            copy.deepcopy(dict(row["payload"])),
        )
        for row in payload.get("pending_transitions", ())
    ]
    state._event_sequence = max((int(row.get("event_sequence", 0)) for row in state.event_log), default=0)
    state._schedule_sequence = max((row.sequence for row in state.pending_transitions), default=0)
    restored_payload = state.to_payload()
    if canonical_hash(restored_payload) != canonical_hash(dict(payload)):
        raise KMaskSnapshotInvariantError("service-obligation state canonical round-trip mismatch")
    return state


class GlobalKMaskSnapshotLifecycle:
    def __init__(self, runtime: FixedVehicleOccurrenceMaskRuntime, version_binding: RuntimeVersionBinding) -> None:
        self.runtime = runtime
        self.version_binding = version_binding

    def _agent_snapshot(self, context: KMaskAgentContext) -> Dict[str, Any]:
        result = self.runtime.evaluate(
            agent_id=context.agent_id,
            vehicle_token=context.vehicle_token,
            active_bus_mask=context.active_bus_mask,
            route_id=context.route_id,
            direction_id=context.direction_id,
            stop_sequence=context.stop_sequence,
            stop_id=context.stop_id,
            route_stop_occurrence_id=context.route_stop_occurrence_id,
            obligation_state_machine=context.obligation_state_machine,
            decision_ts=context.decision_ts,
        )
        state_payload = context.obligation_state_machine.to_payload()
        binding = self.version_binding.to_payload()
        return {
            "agent_id": int(context.agent_id),
            "vehicle_token": str(context.vehicle_token),
            "active_bus_mask": bool(context.active_bus_mask),
            "route_stop_occurrence_id": result.get("route_stop_occurrence_id"),
            "route_id": str(context.route_id) if context.route_id is not None else None,
            "direction_id": str(context.direction_id) if context.direction_id is not None else None,
            "stop_sequence": int(context.stop_sequence) if context.stop_sequence is not None else None,
            "stop_id": str(context.stop_id) if context.stop_id is not None else None,
            "decision_ts": int(context.decision_ts),
            "k_action_mask": [bool(value) for value in result["action_mask"]],
            "hold_valid": bool(result["hold_valid"]),
            "serve_move_valid": bool(result["serve_move_valid"]),
            "conditional_skip_valid": bool(result["skip_valid"]),
            "skip_invalid_reason_codes": [str(value) for value in result["skip_invalid_reason_codes"]],
            "actor_sampling_bypassed": not bool(context.active_bus_mask),
            "occurrence_lookup_integrity": bool(result["occurrence_lookup_integrity"]),
            "identity_failure": bool(result["identity_failure"]),
            "decision_time_obligation_snapshot": copy.deepcopy(result.get("decision_time_obligation_snapshot")),
            "service_obligation_state": state_payload,
            "service_obligation_state_hash": canonical_hash(state_payload),
            **binding,
            "identity_transition_type": context.identity_transition_type,
            "policy_execution_count": int(result["policy_execution_count"]),
        }

    def capture_cycle(
        self,
        *,
        cycle_index: int,
        cycle_timestamp: Optional[str],
        contexts: Sequence[KMaskAgentContext],
    ) -> GlobalKMaskSnapshot:
        ordered = sorted(contexts, key=lambda row: int(row.agent_id))
        if tuple(int(row.agent_id) for row in ordered) != tuple(range(8)):
            raise KMaskSnapshotInvariantError("capture_cycle requires exactly one context for each fixed agent 0..7")
        binding = self.version_binding.to_payload()
        payload = {
            "schema_version": GLOBAL_K_MASK_SNAPSHOT_SCHEMA_VERSION,
            "cycle_index": int(cycle_index),
            "cycle_timestamp": cycle_timestamp,
            "agent_order": list(range(8)),
            "agent_snapshots": [self._agent_snapshot(context) for context in ordered],
            **binding,
            "policy_execution_count": 0,
        }
        return GlobalKMaskSnapshot(payload)

    def recompute(self, snapshot: GlobalKMaskSnapshot) -> GlobalKMaskSnapshot:
        payload = snapshot.to_payload()
        contexts = []
        for row in payload["agent_snapshots"]:
            contexts.append(
                KMaskAgentContext(
                    agent_id=int(row["agent_id"]),
                    vehicle_token=str(row["vehicle_token"]),
                    active_bus_mask=bool(row["active_bus_mask"]),
                    decision_ts=int(row["decision_ts"]),
                    obligation_state_machine=restore_service_obligation_state(row["service_obligation_state"]),
                    route_id=row["route_id"],
                    direction_id=row["direction_id"],
                    stop_sequence=row["stop_sequence"],
                    stop_id=row["stop_id"],
                    route_stop_occurrence_id=row["route_stop_occurrence_id"],
                    identity_transition_type=row["identity_transition_type"],
                )
            )
        return self.capture_cycle(
            cycle_index=int(payload["cycle_index"]),
            cycle_timestamp=payload["cycle_timestamp"],
            contexts=contexts,
        )
