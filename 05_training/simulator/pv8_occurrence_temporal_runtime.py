from __future__ import annotations

"""Occurrence-level bounded PV8 vehicle/passenger temporal runtime."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from simulator.k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime
from simulator.k_safety_state import PassengerStatus, RequestStatus, ServiceObligationStateMachine


DWELL_CONTRACT_VERSION = "PV8_RESEARCH_SERVE_DWELL_10_30S_V1"
REAL_NETWORK_MODE = "REAL_NETWORK_MODE"
RESEARCH_CONTRACT_MODE = "RESEARCH_CONTRACT_MODE"
ACTION_SERVE = "SERVE"
ACTION_PASS_THROUGH = "PASS_THROUGH"
ACTION_HOLD = "HOLD"


class OccurrenceTemporalRuntimeError(ValueError):
    pass


@dataclass(frozen=True)
class RouteLeg:
    route_id: str
    direction_id: str
    from_occurrence_id: str
    from_stop_id: str
    to_occurrence_id: str
    to_stop_id: str
    distance_m: float
    travel_time_sec: float
    distance_source: str
    travel_time_source: str
    source_relation: str
    source_row_id_or_key: str
    mapping_status: str = "OBSERVED_PROJECT_EDGE"

    def __post_init__(self) -> None:
        for name, value in (("distance_m", self.distance_m), ("travel_time_sec", self.travel_time_sec)):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise OccurrenceTemporalRuntimeError(
                    f"{name} is missing or non-numeric; no synthetic fallback is allowed"
                ) from exc
            if not math.isfinite(numeric) or numeric <= 0:
                raise OccurrenceTemporalRuntimeError(f"{name} must be finite and positive; no synthetic fallback is allowed")
        if self.mapping_status == "UNRESOLVED":
            raise OccurrenceTemporalRuntimeError("unresolved route leg cannot enter the repaired runtime")


@dataclass
class VehicleTemporalState:
    vehicle_slot_id: int
    agent_id: int
    vehicle_token: str
    route_id: str
    direction_id: str
    current_occurrence_id: str
    current_stop_id: str
    next_occurrence_id: Optional[str]
    next_stop_id: Optional[str]
    arrival_ts: float
    departure_ts: float
    travel_time_to_next_sec: float = 0.0
    current_dwell_seconds: float = 0.0
    hold_seconds: float = 0.0
    onboard_passenger_ids: List[str] = field(default_factory=list)
    active: bool = True

    def payload(self) -> Dict[str, Any]:
        return {
            "vehicle_slot_id": int(self.vehicle_slot_id),
            "agent_id": int(self.agent_id),
            "vehicle_token": self.vehicle_token,
            "route_id": self.route_id,
            "direction_id": self.direction_id,
            "current_occurrence_id": self.current_occurrence_id,
            "current_stop_id": self.current_stop_id,
            "next_occurrence_id": self.next_occurrence_id,
            "next_stop_id": self.next_stop_id,
            "arrival_ts": float(self.arrival_ts),
            "departure_ts": float(self.departure_ts),
            "travel_time_to_next_sec": float(self.travel_time_to_next_sec),
            "current_dwell_seconds": float(self.current_dwell_seconds),
            "hold_seconds": float(self.hold_seconds),
            "onboard_passenger_ids": list(self.onboard_passenger_ids),
            "active": bool(self.active),
        }


def serve_dwell_seconds(boarding_count: int, alighting_count: int) -> int:
    boarding = int(boarding_count)
    alighting = int(alighting_count)
    if boarding < 0 or alighting < 0:
        raise OccurrenceTemporalRuntimeError("passenger exchange counts cannot be negative")
    exchange = boarding + alighting
    if exchange <= 0:
        return 10
    if exchange <= 5:
        return 15
    if exchange <= 10:
        return 20
    return 30


class OccurrenceTemporalRuntime:
    """Deterministic bounded traversal with K4 obligations and K8 research masks."""

    def __init__(
        self,
        *,
        mask_runtime: FixedVehicleOccurrenceMaskRuntime,
        occurrences: Sequence[Mapping[str, Any]],
        legs: Sequence[RouteLeg],
        agent_id: int,
        vehicle_token: str,
    ) -> None:
        self.mask_runtime = mask_runtime
        self.occurrences = [dict(row) for row in occurrences]
        self.legs = list(legs)
        self.agent_id = int(agent_id)
        self.vehicle_token = str(vehicle_token)
        self._validate_route()

    def _validate_route(self) -> None:
        if len(self.occurrences) < 2 or len(self.legs) != len(self.occurrences) - 1:
            raise OccurrenceTemporalRuntimeError("route must contain N occurrences and N-1 bound legs")
        route_id = str(self.occurrences[0]["route_id"])
        direction_id = str(self.occurrences[0]["direction_id"])
        sequences = [int(row["stop_sequence"]) for row in self.occurrences]
        if sequences != list(range(sequences[0], sequences[0] + len(sequences))):
            raise OccurrenceTemporalRuntimeError("illegal occurrence jump or duplicate occurrence consumption")
        occurrence_ids = [str(row["route_stop_occurrence_id"]) for row in self.occurrences]
        if len(occurrence_ids) != len(set(occurrence_ids)):
            raise OccurrenceTemporalRuntimeError("duplicate occurrence consumption")
        for index, row in enumerate(self.occurrences):
            if str(row["route_id"]) != route_id or str(row["direction_id"]) != direction_id:
                raise OccurrenceTemporalRuntimeError("route-direction contamination")
            if index < len(self.legs):
                leg = self.legs[index]
                if (
                    leg.route_id != route_id
                    or leg.direction_id != direction_id
                    or leg.from_occurrence_id != occurrence_ids[index]
                    or leg.to_occurrence_id != occurrence_ids[index + 1]
                    or leg.from_stop_id != str(row["stop_id"])
                    or leg.to_stop_id != str(self.occurrences[index + 1]["stop_id"])
                ):
                    raise OccurrenceTemporalRuntimeError("route leg does not bind adjacent occurrence identities")

    @staticmethod
    def _onboard_ids(state: ServiceObligationStateMachine, agent_id: int) -> List[str]:
        ledger = state.vehicle_ledgers[int(agent_id)]
        return sorted(ledger.onboard_passenger_by_request.values())

    def _mask(self, state: ServiceObligationStateMachine, occurrence: Mapping[str, Any], decision_ts: int) -> Dict[str, Any]:
        return self.mask_runtime.evaluate(
            agent_id=self.agent_id,
            vehicle_token=self.vehicle_token,
            active_bus_mask=True,
            route_id=str(occurrence["route_id"]),
            direction_id=str(occurrence["direction_id"]),
            stop_sequence=int(occurrence["stop_sequence"]),
            stop_id=str(occurrence["stop_id"]),
            route_stop_occurrence_id=str(occurrence["route_stop_occurrence_id"]),
            obligation_state_machine=state,
            decision_ts=int(decision_ts),
        )

    @staticmethod
    def _passenger_transition_ts(
        state: ServiceObligationStateMachine,
        passenger_id: str,
        transition: str,
    ) -> Optional[int]:
        rows = [
            int(row["event_ts"])
            for row in state.event_log
            if row.get("passenger_id") == passenger_id and row.get("transition") == transition
        ]
        return rows[-1] if rows else None

    def _serve(self, state: ServiceObligationStateMachine, stop_id: str, event_ts: int) -> Dict[str, int]:
        state.advance_to(int(event_ts))
        alighted = 0
        boarded = 0
        for request in sorted(list(state.requests.values()), key=lambda row: row.request_id):
            if (
                request.request_status == RequestStatus.BOARDED
                and request.assigned_vehicle == self.vehicle_token
                and request.dropoff_stop == stop_id
            ):
                passenger = state.passengers[request.passenger_id]
                state.passenger_alighted(
                    request_id=request.request_id,
                    passenger_id=passenger.passenger_id,
                    agent_id=self.agent_id,
                    vehicle_token=self.vehicle_token,
                    stop_id=stop_id,
                    event_ts=int(event_ts),
                )
                state.request_completed(request_id=request.request_id, event_ts=int(event_ts))
                alighted += 1
        for request in sorted(list(state.requests.values()), key=lambda row: row.request_id):
            if (
                request.request_status == RequestStatus.ASSIGNED
                and request.assigned_vehicle == self.vehicle_token
                and request.pickup_stop == stop_id
            ):
                passenger = state.passengers[request.passenger_id]
                state.passenger_boarded(
                    request_id=request.request_id,
                    passenger_id=passenger.passenger_id,
                    agent_id=self.agent_id,
                    vehicle_token=self.vehicle_token,
                    stop_id=stop_id,
                    event_ts=int(event_ts),
                )
                boarded += 1
        return {"boarding_count": boarded, "alighting_count": alighted}

    def run(
        self,
        *,
        state: ServiceObligationStateMachine,
        start_arrival_ts: int,
        legality_mode: str,
        pass_through_occurrence_ids: Sequence[str] = (),
        hold_seconds_by_occurrence: Optional[Mapping[str, float]] = None,
        time_band: str = "fixture",
    ) -> Dict[str, List[Dict[str, Any]]]:
        if legality_mode not in {REAL_NETWORK_MODE, RESEARCH_CONTRACT_MODE}:
            raise OccurrenceTemporalRuntimeError("unknown legality mode")
        requested_passes = {str(value) for value in pass_through_occurrence_ids}
        holds = {str(key): float(value) for key, value in dict(hold_seconds_by_occurrence or {}).items()}
        if any(not math.isfinite(value) or value < 0 for value in holds.values()):
            raise OccurrenceTemporalRuntimeError("hold_seconds must be finite and non-negative")

        vehicle_rows: List[Dict[str, Any]] = []
        passenger_rows: List[Dict[str, Any]] = []
        dropoff_rows: List[Dict[str, Any]] = []
        temporal_rows: List[Dict[str, Any]] = []
        arrival = float(start_arrival_ts)
        prior_sequence: Optional[int] = None
        occurrence_by_stop = {
            str(row["stop_id"]): str(row["route_stop_occurrence_id"])
            for row in self.occurrences
        }

        for index, occurrence in enumerate(self.occurrences):
            occurrence_id = str(occurrence["route_stop_occurrence_id"])
            stop_id = str(occurrence["stop_id"])
            sequence = int(occurrence["stop_sequence"])
            if prior_sequence is not None and sequence != prior_sequence + 1:
                raise OccurrenceTemporalRuntimeError("illegal occurrence jump")
            prior_sequence = sequence
            if arrival < state.current_ts:
                raise OccurrenceTemporalRuntimeError("vehicle time regression")
            state.advance_to(int(round(arrival)))
            mask = self._mask(state, occurrence, int(round(arrival)))
            decision_snapshot = mask.get("decision_time_obligation_snapshot") or {}
            pickup = bool(decision_snapshot.get("pickup_obligation"))
            dropoff = bool(decision_snapshot.get("dropoff_obligation"))
            skip_eligible = bool(mask.get("skip_valid"))
            wants_pass = occurrence_id in requested_passes
            if wants_pass:
                if legality_mode != RESEARCH_CONTRACT_MODE:
                    raise OccurrenceTemporalRuntimeError("real-network empty-stop pass-through legality is unresolved")
                if not skip_eligible or pickup or dropoff:
                    raise OccurrenceTemporalRuntimeError("PASS_THROUGH blocked by K-safety obligation or static guard")
                action = ACTION_PASS_THROUGH
                exchange = {"boarding_count": 0, "alighting_count": 0}
                dwell = 0
            else:
                action = ACTION_SERVE
                exchange = self._serve(state, stop_id, int(round(arrival)))
                dwell = serve_dwell_seconds(exchange["boarding_count"], exchange["alighting_count"])
            hold = holds.get(occurrence_id, 0.0)
            if action == ACTION_PASS_THROUGH and hold:
                raise OccurrenceTemporalRuntimeError("PASS_THROUGH cannot hide a HOLD interval")
            departure = arrival + float(dwell) + hold
            if departure < arrival:
                raise OccurrenceTemporalRuntimeError("departure time regression")
            next_occurrence = self.occurrences[index + 1] if index + 1 < len(self.occurrences) else None
            leg = self.legs[index] if index < len(self.legs) else None
            onboard = self._onboard_ids(state, self.agent_id)
            vehicle = VehicleTemporalState(
                vehicle_slot_id=self.agent_id,
                agent_id=self.agent_id,
                vehicle_token=self.vehicle_token,
                route_id=str(occurrence["route_id"]),
                direction_id=str(occurrence["direction_id"]),
                current_occurrence_id=occurrence_id,
                current_stop_id=stop_id,
                next_occurrence_id=str(next_occurrence["route_stop_occurrence_id"]) if next_occurrence else None,
                next_stop_id=str(next_occurrence["stop_id"]) if next_occurrence else None,
                arrival_ts=arrival,
                departure_ts=departure,
                travel_time_to_next_sec=float(leg.travel_time_sec) if leg else 0.0,
                current_dwell_seconds=float(dwell),
                hold_seconds=hold,
                onboard_passenger_ids=onboard,
            )
            vehicle_row = vehicle.payload()
            vehicle_row.update(
                {
                    "occurrence_index": index,
                    "stop_sequence": sequence,
                    "stop_name": str(occurrence.get("stop_name") or ""),
                    "time_band": time_band,
                    "legality_mode": legality_mode,
                    "executed_action": action,
                    "boarding_count": exchange["boarding_count"],
                    "alighting_count": exchange["alighting_count"],
                    "passenger_exchange_count": exchange["boarding_count"] + exchange["alighting_count"],
                    "pickup_obligation_pre_action": pickup,
                    "dropoff_obligation_pre_action": dropoff,
                    "research_skip_eligible": skip_eligible,
                    "k_action_mask": list(mask["action_mask"]),
                    "skip_invalid_reason_codes": list(mask["skip_invalid_reason_codes"]),
                    "service_stop_count": int(action == ACTION_SERVE),
                    "pass_through_count": int(action == ACTION_PASS_THROUGH),
                    "stop_restart_event_candidate": int(action == ACTION_SERVE),
                }
            )
            vehicle_rows.append(vehicle_row)
            dropoff_rows.append(
                {
                    "occurrence_index": index,
                    "route_stop_occurrence_id": occurrence_id,
                    "stop_id": stop_id,
                    "arrival_ts": arrival,
                    "dropoff_obligation_pre_action": dropoff,
                    "assigned_dropoff_request_ids": list(decision_snapshot.get("assigned_dropoff_request_ids") or []),
                    "onboard_destination_request_ids": list(decision_snapshot.get("onboard_destination_request_ids") or []),
                    "executed_action": action,
                    "alighting_count": exchange["alighting_count"],
                    "dropoff_blocked_pass_through": bool(dropoff and not skip_eligible),
                }
            )
            for passenger in sorted(state.passengers.values(), key=lambda row: row.passenger_id):
                matching_requests = [
                    row for row in state.requests.values() if row.passenger_id == passenger.passenger_id
                ]
                request = matching_requests[-1] if matching_requests else None
                request_status = request.request_status.value if request is not None else passenger.passenger_status.value
                passenger_rows.append(
                    {
                        "occurrence_index": index,
                        "route_stop_occurrence_id": occurrence_id,
                        "stop_id": stop_id,
                        "event_ts": departure,
                        "passenger_id": passenger.passenger_id,
                        "request_id": request.request_id if request is not None else None,
                        "origin_stop_id": passenger.pickup_stop,
                        "origin_occurrence_id": occurrence_by_stop.get(passenger.pickup_stop),
                        "destination_stop_id": passenger.dropoff_stop,
                        "destination_occurrence_id": occurrence_by_stop.get(passenger.dropoff_stop),
                        "passenger_status": "COMPLETED" if request_status == RequestStatus.COMPLETED.value else request_status,
                        "k4_passenger_status": passenger.passenger_status.value,
                        "assigned_vehicle_slot": self.agent_id if passenger.assigned_vehicle == self.vehicle_token else None,
                        "request_ts": int(passenger.created_ts),
                        "board_ts": self._passenger_transition_ts(state, passenger.passenger_id, "passenger_boarded"),
                        "alight_ts": self._passenger_transition_ts(state, passenger.passenger_id, "passenger_alighted"),
                        "onboard": passenger.passenger_id in onboard,
                    }
                )
            if leg is not None:
                next_arrival = departure + float(leg.travel_time_sec)
                temporal_rows.append(
                    {
                        "leg_index": index,
                        "from_occurrence_id": leg.from_occurrence_id,
                        "to_occurrence_id": leg.to_occurrence_id,
                        "departure_ts": departure,
                        "route_distance_m": float(leg.distance_m),
                        "edge_travel_time_sec": float(leg.travel_time_sec),
                        "serve_dwell_seconds": float(dwell),
                        "hold_seconds": hold,
                        "arrival_ts_next": next_arrival,
                        "total_elapsed_seconds": float(dwell) + hold + float(leg.travel_time_sec),
                        "executed_action": action,
                        "distance_source": leg.distance_source,
                        "travel_time_source": leg.travel_time_source,
                    }
                )
                arrival = next_arrival

        integrity = state.audit_integrity()
        if not integrity["passed"]:
            raise OccurrenceTemporalRuntimeError(f"persistent service-state integrity failed: {integrity['violations']}")
        return {
            "vehicle_trace": vehicle_rows,
            "passenger_trace": passenger_rows,
            "dropoff_trace": dropoff_rows,
            "temporal_trace": temporal_rows,
            "state_integrity": [integrity],
        }
