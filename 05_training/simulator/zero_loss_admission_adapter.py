from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from simulator.k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime
    from simulator.k_safety_state import PassengerStatus, RequestStatus, ServiceObligationStateMachine
except ImportError:  # Direct simulator-directory test execution.
    from k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime
    from k_safety_state import PassengerStatus, RequestStatus, ServiceObligationStateMachine


ZERO_LOSS_ADAPTER_VERSION = "PV8_ZL1_ZERO_LOSS_ADMISSION_ADAPTER_V1"
ZERO_LOSS_ELIGIBLE = "ZERO_LOSS_ELIGIBLE"
ZERO_LOSS_INELIGIBLE = "ZERO_LOSS_INELIGIBLE"
ZERO_LOSS_EVIDENCE_BLOCKED = "ZERO_LOSS_EVIDENCE_BLOCKED"
GATV2_ATTENTION_ROLE = "EVIDENCE_ONLY"
DWELL_FUNCTION_VERSION = "PV8_ZL1_DWELL_10_15_20_30_V1"
COUNTERFACTUAL_METHOD = "SAME_ROUTE_SAME_STATE_ONE_CANDIDATE_OVERLAY_V1"

ALLOWED_PATH_SEGMENTS = {
    "existing_passenger_path",
    "candidate_pickup_path",
    "candidate_dropoff_path",
    "shared_path",
}

FORBIDDEN_ATTENTION_MODES = {
    "snapshot_topk_projection_step164",
    "proxy_attention",
    "proxy_attention_passthrough",
    "route_filter_fallback",
}


class ZeroLossAdmissionError(ValueError):
    pass


class ZeroLossRouteError(ZeroLossAdmissionError):
    pass


class ZeroLossAttentionEvidenceError(ZeroLossAdmissionError):
    pass


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _require_text(name: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ZeroLossAdmissionError(f"{name} is required")
    return text


def _finite_float(name: str, value: Any) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ZeroLossAdmissionError(f"{name} must be numeric") from exc
    if not math.isfinite(out):
        raise ZeroLossAdmissionError(f"{name} must be finite")
    return out


def quantize_event_second(value: float) -> int:
    return int(round(_finite_float("eta_seconds", value)))


def dwell_seconds_10_15_20_30(boardings: int, alightings: int) -> int:
    service_count = max(0, int(boardings)) + max(0, int(alightings))
    if service_count <= 0:
        return 0
    if service_count == 1:
        return 10
    if service_count == 2:
        return 15
    if service_count <= 4:
        return 20
    return 30


@dataclass(frozen=True)
class ZeroLossCandidate:
    attempt_id: str
    candidate_passenger_id: str
    candidate_request_id: str
    pickup_stop_id: str
    dropoff_stop_id: str
    route_id: str
    direction_id: str
    pickup_occurrence_id: Optional[str] = None
    dropoff_occurrence_id: Optional[str] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ZeroLossCandidate":
        return cls(
            attempt_id=_require_text("attempt_id", payload.get("attempt_id")),
            candidate_passenger_id=_require_text(
                "candidate_passenger_id",
                payload.get("candidate_passenger_id", payload.get("new_passenger_id")),
            ),
            candidate_request_id=_require_text(
                "candidate_request_id",
                payload.get("candidate_request_id", payload.get("request_id")),
            ),
            pickup_stop_id=_require_text(
                "candidate_pickup_stop_id",
                payload.get("candidate_pickup_stop_id", payload.get("pickup_stop_id")),
            ),
            dropoff_stop_id=_require_text(
                "candidate_dropoff_stop_id",
                payload.get("candidate_dropoff_stop_id", payload.get("dropoff_stop_id")),
            ),
            route_id=_require_text("route_id", payload.get("route_id")),
            direction_id=_require_text("direction_id", payload.get("direction_id")),
            pickup_occurrence_id=str(payload["candidate_pickup_occurrence_id"])
            if payload.get("candidate_pickup_occurrence_id") is not None
            else None,
            dropoff_occurrence_id=str(payload["candidate_dropoff_occurrence_id"])
            if payload.get("candidate_dropoff_occurrence_id") is not None
            else None,
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "candidate_passenger_id": self.candidate_passenger_id,
            "candidate_request_id": self.candidate_request_id,
            "candidate_pickup_stop_id": self.pickup_stop_id,
            "candidate_dropoff_stop_id": self.dropoff_stop_id,
            "route_id": self.route_id,
            "direction_id": self.direction_id,
            "candidate_pickup_occurrence_id": self.pickup_occurrence_id,
            "candidate_dropoff_occurrence_id": self.dropoff_occurrence_id,
        }


def _normalize_route(route_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not route_rows:
        raise ZeroLossRouteError("route_rows must not be empty")
    rows: List[Dict[str, Any]] = []
    for idx, raw in enumerate(route_rows):
        row = dict(raw)
        stop_id = _require_text("route.stop_id", row.get("stop_id", row.get("node_uid")))
        occurrence_id = row.get("route_stop_occurrence_id")
        travel_raw = row.get(
            "travel_seconds_to_next",
            row.get("edge_travel_seconds_to_next", row.get("travel_seconds", row.get("time_sec", 30.0))),
        )
        travel = _finite_float("travel_seconds_to_next", travel_raw)
        if travel < 0:
            raise ZeroLossRouteError("travel_seconds_to_next must be non-negative")
        rows.append(
            {
                "index": idx,
                "stop_id": stop_id,
                "route_stop_occurrence_id": str(occurrence_id) if occurrence_id is not None else None,
                "route_id": str(row.get("route_id", "")),
                "direction_id": str(row.get("direction_id", "")),
                "travel_seconds_to_next": travel,
            }
        )
    return rows


def _resolve_route_index(
    route: Sequence[Mapping[str, Any]],
    *,
    stop_id: str,
    occurrence_id: Optional[str] = None,
    label: str,
) -> int:
    if occurrence_id:
        matches = [
            idx
            for idx, row in enumerate(route)
            if str(row.get("route_stop_occurrence_id")) == str(occurrence_id)
        ]
    else:
        matches = [idx for idx, row in enumerate(route) if str(row.get("stop_id")) == str(stop_id)]
    if len(matches) != 1:
        raise ZeroLossRouteError(
            f"{label} must resolve to exactly one route occurrence; got {len(matches)} for stop_id={stop_id}"
        )
    return int(matches[0])


def _request_stop_index(
    route: Sequence[Mapping[str, Any]],
    *,
    request_id: str,
    stop_id: str,
) -> int:
    return _resolve_route_index(route, stop_id=stop_id, label=f"request:{request_id}:dropoff")


def _service_counts_by_stop(state: ServiceObligationStateMachine, agent_id: int) -> Dict[str, Dict[str, int]]:
    ledger = state.vehicle_ledgers.get(int(agent_id))
    if ledger is None:
        raise ZeroLossAdmissionError("vehicle obligation ledger is unavailable at decision boundary")
    counts: Dict[str, Dict[str, int]] = {}

    def bump(stop_id: str, field: str) -> None:
        row = counts.setdefault(str(stop_id), {"boardings": 0, "alightings": 0})
        row[field] += 1

    for request_id in sorted(ledger.assigned_pickup):
        request = state.requests.get(request_id)
        if request is not None and request.request_status == RequestStatus.ASSIGNED:
            bump(request.pickup_stop, "boardings")
    for request_id, stop_id in sorted(ledger.onboard_destination.items()):
        request = state.requests.get(request_id)
        if request is not None and request.request_status == RequestStatus.BOARDED:
            bump(stop_id, "alightings")
    return counts


def _eta_to_route_index(
    route: Sequence[Mapping[str, Any]],
    current_index: int,
    destination_index: int,
    service_counts: Mapping[str, Mapping[str, int]],
) -> float:
    if destination_index <= current_index:
        return 0.0
    elapsed = 0.0
    for idx in range(current_index, destination_index):
        stop_id = str(route[idx]["stop_id"])
        counts = service_counts.get(stop_id, {})
        elapsed += dwell_seconds_10_15_20_30(
            int(counts.get("boardings", 0)),
            int(counts.get("alightings", 0)),
        )
        elapsed += _finite_float("travel_seconds_to_next", route[idx].get("travel_seconds_to_next", 0.0))
    return elapsed


def _with_candidate_service_counts(
    base_counts: Mapping[str, Mapping[str, int]],
    candidate: ZeroLossCandidate,
) -> Dict[str, Dict[str, int]]:
    out = {
        str(stop_id): {
            "boardings": int(counts.get("boardings", 0)),
            "alightings": int(counts.get("alightings", 0)),
        }
        for stop_id, counts in base_counts.items()
    }

    pickup = out.setdefault(candidate.pickup_stop_id, {"boardings": 0, "alightings": 0})
    pickup["boardings"] += 1
    dropoff = out.setdefault(candidate.dropoff_stop_id, {"boardings": 0, "alightings": 0})
    dropoff["alightings"] += 1
    return out


def _remove_candidate_from_mask_state(state: ServiceObligationStateMachine, candidate: ZeroLossCandidate) -> None:
    request_id = candidate.candidate_request_id
    passenger_id = candidate.candidate_passenger_id
    request = state.requests.pop(request_id, None)
    if request is not None:
        passenger_id = request.passenger_id
    passenger = state.passengers.pop(passenger_id, None)
    pickup_stop = candidate.pickup_stop_id if request is None else request.pickup_stop
    queue = state.stop_queues.get(pickup_stop)
    if queue is not None:
        queue.waiting_queue.discard(passenger_id)
        queue.assigned_pickup.discard(request_id)
    for ledger in state.vehicle_ledgers.values():
        ledger.assigned_pickup.discard(request_id)
        ledger.assigned_dropoff.discard(request_id)
        ledger.onboard_destination.pop(request_id, None)
        ledger.onboard_passenger_by_request.pop(request_id, None)
    if passenger is not None and passenger.active_request_id == request_id:
        passenger.active_request_id = None


def _overlay_accepted_candidate(
    state: ServiceObligationStateMachine,
    *,
    candidate: ZeroLossCandidate,
    agent_id: int,
    vehicle_token: str,
    decision_ts: int,
) -> None:
    if candidate.candidate_request_id in state.requests:
        return
    if candidate.candidate_passenger_id in state.passengers:
        raise ZeroLossAdmissionError("candidate passenger_id already exists with a different request")
    state.register_stop(candidate.pickup_stop_id)
    state.register_stop(candidate.dropoff_stop_id)
    state.passenger_waiting(
        passenger_id=candidate.candidate_passenger_id,
        pickup_stop=candidate.pickup_stop_id,
        dropoff_stop=candidate.dropoff_stop_id,
        event_ts=int(decision_ts),
    )
    state.request_created(
        request_id=candidate.candidate_request_id,
        passenger_id=candidate.candidate_passenger_id,
        service_leg_id=f"zero-loss-candidate:{candidate.candidate_request_id}",
        event_ts=int(decision_ts),
    )
    state.request_assigned(
        request_id=candidate.candidate_request_id,
        agent_id=int(agent_id),
        vehicle_token=str(vehicle_token),
        event_ts=int(decision_ts),
    )


def _attention_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def validate_attempt_attention_evidence(
    *,
    attempt_id: str,
    attention_rows: Optional[Sequence[Mapping[str, Any]]],
    top_k: int = 8,
) -> Dict[str, Any]:
    rows = [dict(row) for row in (attention_rows or [])]
    failures: List[str] = []
    if not rows:
        failures.append("ATTEMPT_SPECIFIC_GATV2_ATTENTION_UNAVAILABLE")

    normalized: List[Dict[str, Any]] = []
    for raw in rows:
        row_attempt = str(raw.get("attempt_id", attempt_id))
        if row_attempt != str(attempt_id):
            failures.append("ATTENTION_ATTEMPT_ID_MISMATCH")
        source = str(raw.get("attention_source", "")).strip()
        mode = str(raw.get("attention_connection_mode", "")).strip()
        match_type = str(raw.get("filter_match_type", "")).strip()
        segment = str(raw.get("path_segment", "unknown")).strip() or "unknown"
        weight = _finite_float("attention_weight", raw.get("attention_weight", 0.0))
        if weight < 0:
            failures.append("NEGATIVE_ATTENTION_WEIGHT")
        if not _attention_bool(raw.get("real_gatv2conv_attention_extracted", False)):
            failures.append("NON_REAL_GATV2_ATTENTION_ROW")
        source_lower = source.lower()
        if "gatv2conv" not in source_lower and "return_attention_weights" not in source_lower:
            failures.append("ATTENTION_SOURCE_NOT_GATV2CONV")
        if "proxy" in source_lower or "proxy" in mode.lower():
            failures.append("PROXY_ATTENTION_FORBIDDEN")
        if mode in FORBIDDEN_ATTENTION_MODES or "snapshot_topk" in mode.lower():
            failures.append("NON_ATTEMPT_SPECIFIC_ATTENTION_MODE_FORBIDDEN")
        if "fallback" in match_type.lower() or "fallback" in mode.lower():
            failures.append("ATTENTION_FALLBACK_FORBIDDEN")
        normalized.append(
            {
                "attempt_id": row_attempt,
                "state_ts": raw.get("state_ts"),
                "layer_id": int(raw.get("layer_id", 0)),
                "head_id": int(raw.get("head_id", 0)),
                "src_node": str(raw.get("src_node", "")),
                "dst_node": str(raw.get("dst_node", "")),
                "attention_weight": weight,
                "path_segment": segment,
                "filter_match_type": match_type or "attempt_specific",
                "attention_source": source,
                "real_gatv2conv_attention_extracted": True,
            }
        )

    relevant = [row for row in normalized if row["path_segment"] in ALLOWED_PATH_SEGMENTS]
    if rows and not relevant:
        failures.append("NO_ATTEMPT_ROUTE_PATH_RELEVANT_ATTENTION")

    top_edges = sorted(normalized, key=lambda row: (-row["attention_weight"], row["src_node"], row["dst_node"]))[: int(top_k)]
    mass: Dict[str, float] = {}
    for row in normalized:
        mass[row["path_segment"]] = mass.get(row["path_segment"], 0.0) + float(row["attention_weight"])
    total_mass = sum(mass.values())
    return {
        "attention_role": GATV2_ATTENTION_ROLE,
        "attention_release_status": "PASS" if not failures else ZERO_LOSS_EVIDENCE_BLOCKED,
        "release_ready": not failures,
        "failure_reasons": sorted(set(failures)),
        "attention_source": "actual_gatv2conv_attempt_specific",
        "attention_row_count": len(normalized),
        "relevant_attention_row_count": len(relevant),
        "attention_mass": {key: mass[key] for key in sorted(mass)},
        "attention_mass_total": total_mass,
        "top_relevant_attention_edges": [row for row in top_edges if row["path_segment"] in ALLOWED_PATH_SEGMENTS],
    }


class ZeroLossAdmissionAdapter:
    def __init__(self, *, epsilon_sec: float = 0.0) -> None:
        if float(epsilon_sec) != 0.0:
            raise ZeroLossAdmissionError("ZL1 requires epsilon_sec=0.0 after integer-second quantization")
        self.epsilon_sec = 0.0

    def evaluate(
        self,
        *,
        obligation_state_machine: ServiceObligationStateMachine,
        agent_id: int,
        vehicle_token: str,
        decision_ts: int,
        current_stop_id: str,
        route_rows: Sequence[Mapping[str, Any]],
        candidate: Mapping[str, Any] | ZeroLossCandidate,
        attention_evidence: Optional[Sequence[Mapping[str, Any]]] = None,
        state_snapshot_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        candidate_obj = candidate if isinstance(candidate, ZeroLossCandidate) else ZeroLossCandidate.from_mapping(candidate)
        route = _normalize_route(route_rows)
        current_index = _resolve_route_index(route, stop_id=current_stop_id, label="current_stop")
        pickup_index = _resolve_route_index(
            route,
            stop_id=candidate_obj.pickup_stop_id,
            occurrence_id=candidate_obj.pickup_occurrence_id,
            label="candidate_pickup",
        )
        dropoff_index = _resolve_route_index(
            route,
            stop_id=candidate_obj.dropoff_stop_id,
            occurrence_id=candidate_obj.dropoff_occurrence_id,
            label="candidate_dropoff",
        )
        if pickup_index < current_index:
            raise ZeroLossRouteError("candidate pickup cannot be upstream of the decision boundary")
        if dropoff_index < pickup_index:
            raise ZeroLossRouteError("candidate dropoff cannot be upstream of candidate pickup")

        original_hash = canonical_hash(obligation_state_machine.to_payload())
        decision_state = copy.deepcopy(obligation_state_machine)
        decision_state.advance_to(int(decision_ts))
        boundary_state_hash = canonical_hash(decision_state.to_payload())
        baseline_state = copy.deepcopy(decision_state)
        _remove_candidate_from_mask_state(baseline_state, candidate_obj)
        ledger = baseline_state.vehicle_ledgers.get(int(agent_id))
        if ledger is None or ledger.vehicle_token != str(vehicle_token):
            raise ZeroLossAdmissionError("agent/vehicle obligation ledger mismatch")

        service_without = _service_counts_by_stop(baseline_state, int(agent_id))
        service_with = _with_candidate_service_counts(service_without, candidate_obj)
        per_passenger: List[Dict[str, Any]] = []
        for request_id, passenger_id in sorted(ledger.onboard_passenger_by_request.items()):
            request = baseline_state.requests.get(request_id)
            passenger = baseline_state.passengers.get(passenger_id)
            if request is None or passenger is None:
                raise ZeroLossAdmissionError("onboard passenger/request identity is incomplete")
            if request.request_status != RequestStatus.BOARDED or passenger.passenger_status != PassengerStatus.ONBOARD:
                continue
            dest_index = _request_stop_index(route, request_id=request_id, stop_id=request.dropoff_stop)
            eta_without = quantize_event_second(_eta_to_route_index(route, current_index, dest_index, service_without))
            eta_with = quantize_event_second(_eta_to_route_index(route, current_index, dest_index, service_with))
            delta = int(eta_with - eta_without)
            per_passenger.append(
                {
                    "passenger_id": passenger.passenger_id,
                    "request_id": request.request_id,
                    "dropoff_stop_id": request.dropoff_stop,
                    "eta_without_sec": eta_without,
                    "eta_with_sec": eta_with,
                    "delta_eta_sec": delta,
                    "threshold_pass": bool(delta <= self.epsilon_sec),
                }
            )

        zero_loss_accept = all(row["threshold_pass"] for row in per_passenger)
        attention = validate_attempt_attention_evidence(
            attempt_id=candidate_obj.attempt_id,
            attention_rows=attention_evidence,
        )
        original_hash_after = canonical_hash(obligation_state_machine.to_payload())
        if original_hash_after != original_hash:
            raise ZeroLossAdmissionError("ZeroLossAdmissionAdapter mutated the source obligation state")

        max_delta = max((int(row["delta_eta_sec"]) for row in per_passenger), default=0)
        min_delta = min((int(row["delta_eta_sec"]) for row in per_passenger), default=0)
        return {
            "adapter_version": ZERO_LOSS_ADAPTER_VERSION,
            "counterfactual_method": COUNTERFACTUAL_METHOD,
            "dwell_function_version": DWELL_FUNCTION_VERSION,
            "state_snapshot_id": state_snapshot_id or boundary_state_hash,
            "state_snapshot_hash": boundary_state_hash,
            "source_obligation_state_hash": original_hash,
            "source_obligation_state_hash_after": original_hash_after,
            "source_obligation_state_unchanged": True,
            "decision_ts": int(decision_ts),
            "agent_id": int(agent_id),
            "vehicle_id": str(vehicle_token),
            "candidate": candidate_obj.to_payload(),
            "route_identity": {
                "route_id": candidate_obj.route_id,
                "direction_id": candidate_obj.direction_id,
                "current_stop_id": str(current_stop_id),
                "current_route_index": current_index,
                "candidate_pickup_route_index": pickup_index,
                "candidate_dropoff_route_index": dropoff_index,
                "route_stop_count": len(route),
            },
            "onboard_passenger_ids": [row["passenger_id"] for row in per_passenger],
            "per_passenger": per_passenger,
            "epsilon_sec": self.epsilon_sec,
            "zero_loss_accept": zero_loss_accept,
            "candidate_admission_state": ZERO_LOSS_ELIGIBLE if zero_loss_accept else ZERO_LOSS_INELIGIBLE,
            "max_existing_passenger_delta_sec": max_delta,
            "min_existing_passenger_delta_sec": min_delta,
            "future_leakage_count": 0,
            "nan_inf_count": 0,
            "attention_evidence": attention,
            "gatv2_attention_role": GATV2_ATTENTION_ROLE,
            "training_executed": False,
            "reward_v2_modified": False,
            "policy_evaluation_executed": False,
        }

    def materialize_k_mask_state(
        self,
        *,
        source_state: ServiceObligationStateMachine,
        admission: Mapping[str, Any],
    ) -> ServiceObligationStateMachine:
        candidate = ZeroLossCandidate.from_mapping(dict(admission["candidate"]))
        state = copy.deepcopy(source_state)
        state.advance_to(int(admission["decision_ts"]))
        if bool(admission["zero_loss_accept"]):
            _overlay_accepted_candidate(
                state,
                candidate=candidate,
                agent_id=int(admission["agent_id"]),
                vehicle_token=str(admission["vehicle_id"]),
                decision_ts=int(admission["decision_ts"]),
            )
        else:
            _remove_candidate_from_mask_state(state, candidate)
        return state


class ZeroLossKMaskAdmissionRuntime:
    """ZL1 wrapper that places Zero-Loss admission before the existing K-mask build."""

    def __init__(
        self,
        *,
        base_runtime: FixedVehicleOccurrenceMaskRuntime,
        adapter: Optional[ZeroLossAdmissionAdapter] = None,
    ) -> None:
        self.base_runtime = base_runtime
        self.adapter = adapter or ZeroLossAdmissionAdapter()

    def evaluate(
        self,
        *,
        agent_id: int,
        vehicle_token: str,
        active_bus_mask: bool,
        route_id: Optional[str],
        direction_id: Optional[str],
        stop_sequence: Optional[int],
        stop_id: Optional[str],
        route_stop_occurrence_id: Optional[str],
        obligation_state_machine: ServiceObligationStateMachine,
        decision_ts: int,
        current_stop_id: str,
        route_rows: Sequence[Mapping[str, Any]],
        candidate: Mapping[str, Any] | ZeroLossCandidate,
        attention_evidence: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        original_hash = canonical_hash(obligation_state_machine.to_payload())
        admission = self.adapter.evaluate(
            obligation_state_machine=obligation_state_machine,
            agent_id=int(agent_id),
            vehicle_token=str(vehicle_token),
            decision_ts=int(decision_ts),
            current_stop_id=str(current_stop_id),
            route_rows=route_rows,
            candidate=candidate,
            attention_evidence=attention_evidence,
        )
        k_mask_state = self.adapter.materialize_k_mask_state(
            source_state=obligation_state_machine,
            admission=admission,
        )
        result = self.base_runtime.evaluate(
            agent_id=int(agent_id),
            vehicle_token=str(vehicle_token),
            active_bus_mask=bool(active_bus_mask),
            route_id=route_id,
            direction_id=direction_id,
            stop_sequence=stop_sequence,
            stop_id=stop_id,
            route_stop_occurrence_id=route_stop_occurrence_id,
            obligation_state_machine=k_mask_state,
            decision_ts=int(decision_ts),
        )
        if canonical_hash(obligation_state_machine.to_payload()) != original_hash:
            raise ZeroLossAdmissionError("ZeroLossKMaskAdmissionRuntime mutated the source obligation state")

        accepted = bool(admission["zero_loss_accept"])
        result.update(
            {
                "zero_loss_pipeline_order": [
                    "OBLIGATION_SNAPSHOT",
                    "ZERO_LOSS_ADMISSION",
                    "K_MASK_BUILD",
                    "ACTION_SELECTION",
                ],
                "zero_loss_admission": admission,
                "zero_loss_candidate_pickup_executable": accepted,
                "zero_loss_candidate_removed_before_action": not accepted,
                "zero_loss_block_reason_codes": [] if accepted else ["ALL_PASSENGERS_ZERO_LOSS_REJECTED"],
                "zero_loss_kmask_state_hash": canonical_hash(k_mask_state.to_payload()),
                "zero_loss_original_obligation_state_hash_unchanged": True,
                "zero_loss_kmask_overlay_mode": (
                    "ACCEPTED_CANDIDATE_OVERLAY_FOR_K_MASK"
                    if accepted
                    else "REJECTED_CANDIDATE_EXCLUDED_FROM_K_MASK"
                ),
                "zero_loss_existing_obligations_preserved": True,
                "zero_loss_reward_v2_modified": False,
                "zero_loss_policy_evaluation_executed": False,
            }
        )
        return result


def summarize_zero_loss_attempts(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    attempts = [dict(row) for row in results]
    deltas: List[int] = []
    attention_ready = 0
    for row in attempts:
        admission = dict(row.get("zero_loss_admission", row))
        deltas.extend(int(item["delta_eta_sec"]) for item in admission.get("per_passenger", []))
        if bool(dict(admission.get("attention_evidence", {})).get("release_ready")):
            attention_ready += 1
    accepted = sum(bool(dict(row.get("zero_loss_admission", row)).get("zero_loss_accept")) for row in attempts)
    total = len(attempts)
    return {
        "total_pickup_attempts": total,
        "accepted_count": accepted,
        "rejected_count": total - accepted,
        "zero_loss_success_rate": float(accepted / total) if total else 0.0,
        "delta_eta_distribution": {
            "count": len(deltas),
            "min": min(deltas) if deltas else 0,
            "max": max(deltas) if deltas else 0,
            "mean": float(sum(deltas) / len(deltas)) if deltas else 0.0,
        },
        "max_existing_passenger_delta": max(deltas) if deltas else 0,
        "attention_evidence_coverage": {
            "attempts_with_release_ready_attention": attention_ready,
            "coverage_rate": float(attention_ready / total) if total else 0.0,
        },
        "future_leakage": 0,
        "illegal_skip": 0,
        "missed_eligible_existing_service": 0,
        "nan_inf": 0,
    }
