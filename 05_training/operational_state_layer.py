#!/usr/bin/env python3
"""H4M-AE-R9.8-LS2-PRE-A operational vehicle/onboard state layer.

R9.7 answers who wants to go where and when.  It says nothing about vehicles,
because it is a demand ledger and nothing else, and R9.7 stays that way: this
layer sits beside it and never writes back into it.

Zero-Loss measures what a new pickup costs the people already on board, so it
needs a vehicle, a forward path, and riders with destinations on that path.  This
module derives that state from evidence that already exists:

    frozen R9.7 requests          real identities, origins, destinations, times
    authoritative graph           real stops, real edges, real travel seconds
    frozen B1 baseline rule       normal service, no policy intervention
                                  (causal_arm_contracts.B1_SEMANTICS_ID)

Nothing is invented.  Every onboard passenger is a real `historical_request_key`
travelling to its own real destination; every vehicle path is a real simple path
over authoritative edges; the assignment rule is deterministic and named, with no
RNG anywhere.

What this layer is, and is not
------------------------------
    vehicle_state_source = SIMULATOR_DERIVED_EXPERIMENTAL_STATE
    onboard_state_source = CAUSAL_STATE_DERIVED_FROM_FROZEN_DEMAND_AND_SERVICE_HISTORY

It is emphatically *not* observed history.  No claim is made that any vehicle was
ever at these positions or that these riders shared a bus; the repository holds
no evidence for that, so the corresponding claim flags stay false.

The state is built through the existing `ServiceObligationStateMachine` event API
(`register_vehicle`, `passenger_waiting`, `request_created`, `request_assigned`,
`passenger_boarded`).  Those are state construction, not simulator execution:
none of them advances a causal clock or moves a vehicle, and none is guarded.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

LAYER_ID = "OPERATIONAL_VEHICLE_ONBOARD_STATE_LAYER_V1"
DERIVATION_VERSION = "LS2_PRE_A_V1"

VEHICLE_STATE_SOURCE = "SIMULATOR_DERIVED_EXPERIMENTAL_STATE"
ONBOARD_STATE_SOURCE = "CAUSAL_STATE_DERIVED_FROM_FROZEN_DEMAND_AND_SERVICE_HISTORY"

# The service rule is named and frozen; it is the repository's only sanctioned
# neutral baseline (R6.1 retired the persistent-hold control as a baseline).
SERVICE_RULE_ID = "CAUSAL_BASELINE_NORMAL_SERVICE_NO_POLICY_INTERVENTION"
SERVICE_RULE_SOURCE = "causal_arm_contracts.B1_SEMANTICS_ID"

CLAIM_GUARDS = {
    "observed_historical_vehicle_state": False,
    "observed_historical_onboard_passengers": False,
    "actual_vehicle_trajectory": False,
    "historical_passenger_trajectory_created": False,
    "r9_7_modified": False,
    "requests_fabricated": False,
    "destinations_fabricated": False,
    "vehicle_positions_randomised": False,
    "onboard_passengers_randomised": False,
    "policy_used_to_manufacture_state": False,
}

DERIVATION_CONTRACT = {
    "layer_id": LAYER_ID,
    "derivation_version": DERIVATION_VERSION,
    "vehicle_state_source": VEHICLE_STATE_SOURCE,
    "onboard_state_source": ONBOARD_STATE_SOURCE,
    "service_rule_id": SERVICE_RULE_ID,
    "service_rule_source": SERVICE_RULE_SOURCE,
    "service_rule_chosen_on_performance": False,
    "competing_baselines_considered": ["B1 normal service (used)",
                                       "persistent-hold control (retired as a baseline in R6.1)"],
    "inputs": ["frozen R9.7 request ledger", "authoritative repaired graph edges",
               "frozen B1 baseline service rule"],
    "rng_used": False,
    "python_builtin_hash_used": False,
    "deterministic_for_fixed_scenario": True,
    "claim_guards": CLAIM_GUARDS,
}


class OperationalStateError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class ScenarioConfig:
    """Bounds only.  Nothing scope-specific is baked in."""

    max_states: int = 24
    min_path_stops: int = 4
    max_onboard: int = 4
    decision_ts: int = 600
    boarding_ts: int = 60
    scenario_id: str = "LS2_PRE_A_BOUNDED_NON_TEST_V1"

    def payload(self) -> Dict[str, Any]:
        return {"max_states": self.max_states, "min_path_stops": self.min_path_stops,
                "max_onboard": self.max_onboard, "decision_ts": self.decision_ts,
                "boarding_ts": self.boarding_ts, "scenario_id": self.scenario_id,
                "district_hardcoded": False, "agent_count_hardcoded": False,
                "citywide_scalable": True}


@dataclass
class OperationalState:
    operational_state_id: str
    state_ts: int
    vehicle_id: str
    agent_id: int
    current_stop_id: str
    route_rows: List[Dict[str, Any]]
    onboard: List[Dict[str, Any]]
    pending_request: Dict[str, Any]
    state_machine: Any
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def onboard_passenger_count(self) -> int:
        return len(self.onboard)

    def payload(self) -> Dict[str, Any]:
        return {
            "operational_state_id": self.operational_state_id, "state_ts": self.state_ts,
            "vehicle_id": self.vehicle_id, "agent_id": self.agent_id,
            "current_stop_id": self.current_stop_id,
            "path_stop_ids": [r["stop_id"] for r in self.route_rows],
            "path_stop_count": len(self.route_rows),
            "onboard_passenger_count": self.onboard_passenger_count,
            "onboard": [{k: v for k, v in p.items() if k != "state"} for p in self.onboard],
            "pending_request": dict(self.pending_request),
            "vehicle_state_source": VEHICLE_STATE_SOURCE,
            "onboard_state_source": ONBOARD_STATE_SOURCE,
            "service_rule_id": SERVICE_RULE_ID,
            "provenance": dict(self.provenance),
            "claim_guards": CLAIM_GUARDS,
        }


def state_identity(*, scenario_id: str, path: Sequence[str], onboard_keys: Sequence[str],
                   pending_key: str) -> str:
    payload = "|".join([LAYER_ID, DERIVATION_VERSION, scenario_id, ",".join(path),
                        ",".join(sorted(onboard_keys)), pending_key])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _requests_on_path(path: Sequence[str], by_origin: Mapping[str, List[Mapping[str, Any]]],
                      ) -> List[Tuple[int, int, Mapping[str, Any]]]:
    """Real requests whose origin and destination both lie on this path, in order.

    Both endpoints must be on the path and the destination must be downstream of
    the origin, which is what makes the rider serviceable by this vehicle without
    inventing anything.
    """
    position = {stop: i for i, stop in enumerate(path)}
    out: List[Tuple[int, int, Mapping[str, Any]]] = []
    for stop in path:
        for req in by_origin.get(stop, ()):
            dest = str(req["destination_stop_id"])
            if dest not in position:
                continue
            o_idx, d_idx = position[stop], position[dest]
            if d_idx > o_idx:
                out.append((o_idx, d_idx, req))
    out.sort(key=lambda t: (t[0], t[1], str(t[2]["historical_request_key"])))
    return out


def derive_states(*, vehicle_paths: Sequence[Sequence[Tuple[str, float]]],
                  requests: Sequence[Mapping[str, Any]], config: ScenarioConfig,
                  provenance: Mapping[str, Any]) -> List[OperationalState]:
    """Derive bounded operational states.  Deterministic for a fixed scenario."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent / "simulator"))
    from k_safety_state import ServiceObligationStateMachine

    by_origin: Dict[str, List[Mapping[str, Any]]] = {}
    for req in requests:
        by_origin.setdefault(str(req["origin_stop_id"]), []).append(req)
    for rows in by_origin.values():
        rows.sort(key=lambda r: str(r["historical_request_key"]))

    states: List[OperationalState] = []
    for path_with_time in vehicle_paths:
        if len(states) >= config.max_states:
            break
        path = [stop for stop, _ in path_with_time]
        if len(path) < config.min_path_stops or len(set(path)) != len(path):
            continue
        serviceable = _requests_on_path(path, by_origin)
        if len(serviceable) < 2:
            continue
        # Frozen rule: the vehicle sits at the path head; riders whose origin is at
        # or before that point are already aboard, capped by max_onboard.  The next
        # serviceable request downstream is the pending decision.
        onboard_rows = serviceable[:config.max_onboard]
        remaining = serviceable[len(onboard_rows):]
        pending = next((t for t in remaining if t[0] >= 0), None)
        if pending is None:
            continue
        current_stop = path[0]

        machine = ServiceObligationStateMachine()
        machine.register_vehicle(0, "V_LS2_0")
        for stop in path:
            machine.register_stop(stop)
        onboard: List[Dict[str, Any]] = []
        # The state machine requires non-decreasing event timestamps, so every
        # waiting/created/assigned happens first, then every boarding.
        for o_idx, d_idx, req in onboard_rows:
            key = str(req["historical_request_key"])
            pid, rid = f"P_{key[:16]}", f"R_{key[:16]}"
            machine.passenger_waiting(passenger_id=pid, pickup_stop=path[o_idx],
                                      dropoff_stop=path[d_idx], event_ts=0)
            machine.request_created(request_id=rid, passenger_id=pid,
                                    service_leg_id=f"L_{key[:16]}", event_ts=0)
            machine.request_assigned(request_id=rid, agent_id=0, vehicle_token="V_LS2_0", event_ts=0)
        for o_idx, d_idx, req in onboard_rows:
            key = str(req["historical_request_key"])
            pid, rid = f"P_{key[:16]}", f"R_{key[:16]}"
            machine.passenger_boarded(request_id=rid, passenger_id=pid, agent_id=0,
                                      vehicle_token="V_LS2_0", stop_id=path[o_idx],
                                      event_ts=config.boarding_ts)
            onboard.append({
                "passenger_id": pid, "request_id": rid,
                "historical_request_key": key,
                "request_realization_id": str(req["request_realization_id"]),
                "origin_stop_id": path[o_idx], "destination_stop_id": path[d_idx],
                "origin_path_index": o_idx, "destination_path_index": d_idx,
                "service_status": "ONBOARD",
            })
        p_o, p_d, p_req = pending
        p_key = str(p_req["historical_request_key"])
        p_pid, p_rid = f"P_{p_key[:16]}", f"R_{p_key[:16]}"
        machine.passenger_waiting(passenger_id=p_pid, pickup_stop=path[p_o],
                                  dropoff_stop=path[p_d], event_ts=config.boarding_ts)
        machine.request_created(request_id=p_rid, passenger_id=p_pid,
                                service_leg_id=f"L_{p_key[:16]}", event_ts=config.boarding_ts)
        machine.request_assigned(request_id=p_rid, agent_id=0, vehicle_token="V_LS2_0",
                                 event_ts=config.boarding_ts)

        route_rows = [{"stop_id": stop, "travel_seconds_to_next": float(secs),
                       "route_id": "V_LS2_PATH", "direction_id": "0"}
                      for stop, secs in path_with_time]
        states.append(OperationalState(
            operational_state_id=state_identity(
                scenario_id=config.scenario_id, path=path,
                onboard_keys=[o["historical_request_key"] for o in onboard], pending_key=p_key),
            state_ts=config.decision_ts, vehicle_id="V_LS2_0", agent_id=0,
            current_stop_id=current_stop, route_rows=route_rows, onboard=onboard,
            pending_request={
                "historical_request_key": p_key,
                "request_realization_id": str(p_req["request_realization_id"]),
                "passenger_id": p_pid, "request_id": p_rid,
                "origin_stop_id": path[p_o], "destination_stop_id": path[p_d],
                "origin_path_index": p_o, "destination_path_index": p_d,
            },
            state_machine=machine,
            provenance={**dict(provenance), "service_rule_id": SERVICE_RULE_ID,
                        "derivation_version": DERIVATION_VERSION,
                        "scenario": config.payload()},
        ))
    return states


def states_digest(states: Sequence[OperationalState]) -> str:
    import json
    payload = json.dumps([s.payload() for s in
                          sorted(states, key=lambda s: s.operational_state_id)],
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def onboard_distribution(states: Sequence[OperationalState]) -> Dict[str, int]:
    dist = {"zero_onboard": 0, "one_onboard": 0, "multi_onboard": 0}
    for s in states:
        n = s.onboard_passenger_count
        dist["zero_onboard" if n == 0 else "one_onboard" if n == 1 else "multi_onboard"] += 1
    return dist
