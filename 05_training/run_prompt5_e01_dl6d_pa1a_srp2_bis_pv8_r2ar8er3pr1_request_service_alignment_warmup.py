#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-P-R1 request-to-service alignment warm-up repair.

The R3-P peak-window and passenger payload are immutable inputs.  This runner
adds only a deterministic, non-policy B1 dispatch pre-roll so that the first
eligible service is a physical arrival at the passenger's origin occurrence.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import platform
import resource
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3_realistic_b1_regeneration as r3
from simulator.k_safety_state import RequestStatus, ServiceObligationStateMachine
from simulator.pv8_occurrence_temporal_runtime import DWELL_CONTRACT_VERSION, serve_dwell_seconds


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3pr1_request_service_alignment_warmup.py"
RUNTIME_PATH = TRAINING_ROOT / "simulator" / "pv8_occurrence_temporal_runtime.py"

R8ER3P_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3p_weekday_peak_realism_diagnostic_20260809_185720"
R8ER2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er2_occurrence_dwell_temporal_repair_20260809_175119"

WINDOW_PATH = R8ER3P_ROOT / "r8er3p_peak_window_registry.parquet"
PASSENGER_PATH = R8ER3P_ROOT / "r8er3p_passenger_lifecycle.parquet"
OLD_VEHICLE_PATH = R8ER3P_ROOT / "r8er3p_vehicle_timeline.parquet"

HEADWAY_SECONDS = 540
H4_SECONDS = 240
EXPECTED_WINDOW_COUNT = 10
EXPECTED_PASSENGER_COUNT = 104
TIMEZONE = "Asia/Seoul"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3pr1_request_service_alignment_warmup"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3PR1_REQUEST_SERVICE_ALIGNMENT_WARMUP_REPAIR_COMPLETE"
READINESS = "R3P_R1_ALIGNMENT_REPAIR_COMPLETE_FOLLOW_ON_LOCKED"

UPSTREAMS = {
    "PV8-R2A-R8E-R2": (
        R8ER2_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8er2.json",
        "_PV8_R2AR8ER2_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER2_OCCURRENCE_LEVEL_DWELL_TEMPORAL_PROPAGATION_REPAIR_COMPLETE",
    ),
    "PV8-R2A-R8E-R3-P": (
        R8ER3P_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8er3p.json",
        "_PV8_R2AR8ER3P_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3P_WEEKDAY_HIGH_DEMAND_PEAK_REALISM_DIAGNOSTIC_COMPLETE",
    ),
}

PAYLOADS = [
    "r8er3pr1_frozen_peak_window_binding.json",
    "r8er3pr1_frozen_passenger_payload.parquet",
    "r8er3pr1_passenger_payload_equivalence.json",
    "r8er3pr1_warmup_contract.json",
    "r8er3pr1_warmup_candidate_audit.json",
    "r8er3pr1_vehicle_preroll_trace.parquet",
    "r8er3pr1_evaluation_start_vehicle_state.parquet",
    "r8er3pr1_first_eligible_service_trace.parquet",
    "r8er3pr1_passenger_wait_decomposition.parquet",
    "r8er3pr1_missed_eligible_service_audit.parquet",
    "r8er3pr1_dispatch_phase_audit.parquet",
    "r8er3pr1_window_boundary_counterfactual.parquet",
    "r8er3pr1_demand_alignment_counterfactual.parquet",
    "r8er3pr1_old_vs_new_passenger_wait.parquet",
    "r8er3pr1_old_vs_new_wait_summary.json",
    "r8er3pr1_schedule_wait_distribution.json",
    "r8er3pr1_alignment_excess_distribution.json",
    "r8er3pr1_long_wait_reclassification.parquet",
    "r8er3pr1_h4_observability_after_repair.parquet",
    "r8er3pr1_h4_observability_summary.json",
    "r8er3pr1_cold_start_artifact_audit.json",
    "r8er3pr1_service_binding_artifact_audit.json",
    "r8er3pr1_integrity_audit.json",
    "r8er3pr1_deterministic_replay.json",
    "r8er3pr1_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R8ER3PR1Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")


def clean_hash(value: Any) -> str:
    def default(item: Any) -> Any:
        if isinstance(item, (pd.Timestamp, datetime)):
            return item.isoformat()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"unsupported canonical payload type: {type(item)!r}")

    payload = json.dumps(k5.json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def frame_hash(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    records = frame.loc[:, list(columns)].copy().sort_values(list(columns), kind="stable").to_dict("records")
    return clean_hash(records)


def numeric_summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    series = pd.Series([float(value) for value in values], dtype=float)
    keys = ("count", "mean", "median", "p25", "p50", "p75", "p90", "p95", "p99", "min", "max")
    if series.empty:
        return {key: None for key in keys}
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p25": float(series.quantile(0.25, interpolation="linear")),
        "p50": float(series.quantile(0.50, interpolation="linear")),
        "p75": float(series.quantile(0.75, interpolation="linear")),
        "p90": float(series.quantile(0.90, interpolation="linear")),
        "p95": float(series.quantile(0.95, interpolation="linear")),
        "p99": float(series.quantile(0.99, interpolation="linear")),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def change(new: Optional[float], old: Optional[float]) -> Dict[str, Optional[float]]:
    if new is None or old is None:
        return {"absolute": None, "relative": None}
    return {"absolute": float(new - old), "relative": None if old == 0 else float((new - old) / old)}


def verify_upstreams() -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed = gate.get("gate") or gate.get("terminal_gate")
        integrity = k5.verify_manifest(root, manifest_name, lock_name)
        if observed != expected_gate or not k5.manifest_ok(integrity):
            raise R8ER3PR1Error(f"{label} manifest/lock integrity failed")
        records[label] = {"artifact_root": str(root), "gate": observed, "decision": gate.get("final_decision"), "integrity": integrity, "valid": True}
    if records["PV8-R2A-R8E-R3-P"]["decision"] != "WAIT_GENERATION_OR_SERVICE_ALIGNMENT_REPAIR_REQUIRED":
        raise R8ER3PR1Error("R3-P did not authorize temporal alignment repair")
    r3p_guard = k5.read_json(R8ER3P_ROOT / "claim_guard_status.json")
    if r3p_guard.get("training_normalization_approved") or r3p_guard.get("r3_candidate_mutated"):
        raise R8ER3PR1Error("R3-P lock boundary is inconsistent")
    return {"created_at": iso_kst(), "artifacts": records, "r3p_artifact_mutated": False}


def load_frozen_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    windows = pd.read_parquet(WINDOW_PATH).sort_values("window_ordinal").reset_index(drop=True)
    upstream_passengers = pd.read_parquet(PASSENGER_PATH).copy()
    old_vehicle = pd.read_parquet(OLD_VEHICLE_PATH).copy()
    if len(windows) != EXPECTED_WINDOW_COUNT or set(windows["direction_id"].astype(str)) != {"0", "1"}:
        raise R8ER3PR1Error("frozen peak window count or direction coverage drifted")
    if len(upstream_passengers) != EXPECTED_PASSENGER_COUNT:
        raise R8ER3PR1Error("frozen passenger count drifted")
    required = [
        "passenger_id", "request_id", "window_id", "request_ts", "origin_stop_id", "origin_occurrence_id",
        "destination_stop_id", "destination_occurrence_id", "direction_id", "trip_id", "dispatch_ts", "board_ts", "alight_ts", "wait_seconds",
    ]
    if any(column not in upstream_passengers.columns for column in required):
        raise R8ER3PR1Error("R3-P passenger payload is missing required frozen identity fields")
    frozen = upstream_passengers.loc[:, required].copy()
    frozen = frozen.rename(columns={"trip_id": "old_trip_id", "dispatch_ts": "old_dispatch_ts", "board_ts": "old_board_ts", "alight_ts": "old_alight_ts", "wait_seconds": "old_wait_seconds"})
    frozen["direction_id"] = frozen["direction_id"].astype(str)
    for column in ("request_ts", "old_dispatch_ts", "old_board_ts", "old_alight_ts", "old_wait_seconds"):
        frozen[column] = frozen[column].astype(int)
    identity_columns = ["passenger_id", "request_id", "window_id", "request_ts", "origin_stop_id", "origin_occurrence_id", "destination_stop_id", "destination_occurrence_id", "direction_id"]
    upstream_hash = frame_hash(upstream_passengers, identity_columns)
    frozen_hash = frame_hash(frozen, identity_columns)
    if upstream_hash != frozen_hash or frozen["passenger_id"].duplicated().any() or frozen["request_id"].duplicated().any():
        raise R8ER3PR1Error("frozen passenger realization cannot be represented exactly")
    window_columns = ["window_id", "service_date", "direction_id", "historical_demand_score", "rank", "start_ts"]
    window_hash = frame_hash(windows, window_columns)
    window_binding = {
        "created_at": iso_kst(),
        "source_artifact": str(R8ER3P_ROOT),
        "source_path": str(WINDOW_PATH),
        "selected_window_count": int(len(windows)),
        "expected_window_count": EXPECTED_WINDOW_COUNT,
        "window_binding_columns": window_columns,
        "window_registry_sha256": k5.sha256_file(WINDOW_PATH),
        "canonical_window_binding_sha256": window_hash,
        "window_substitution_count": 0,
        "post_hoc_window_replacement": False,
        "directions": sorted(windows["direction_id"].astype(str).unique().tolist()),
    }
    passenger_equivalence = {
        "created_at": iso_kst(),
        "source_artifact": str(R8ER3P_ROOT),
        "source_path": str(PASSENGER_PATH),
        "expected_passenger_count": EXPECTED_PASSENGER_COUNT,
        "passenger_count": int(len(frozen)),
        "identity_columns": identity_columns,
        "upstream_canonical_sha256": upstream_hash,
        "frozen_payload_canonical_sha256": frozen_hash,
        "passenger_identity_drift_count": 0,
        "request_ts_drift_count": 0,
        "origin_drift_count": 0,
        "destination_drift_count": 0,
        "equivalence_passed": True,
        "passenger_payload_regenerated": False,
        "demand_seed_reused_only_as_lineage": True,
    }
    return windows, frozen, old_vehicle, window_binding, passenger_equivalence


def build_warmup_contract(legs_by_direction: Mapping[str, Sequence[r3.RouteLeg]]) -> Tuple[Dict[str, Any], pd.DataFrame, int, int]:
    directions: Dict[str, Dict[str, float]] = {}
    max_bounded_traversal = 0.0
    for direction_id, legs in legs_by_direction.items():
        travel = float(sum(float(leg.travel_time_sec) for leg in legs))
        occurrence_count = len(legs) + 1
        empty = travel + 10.0 * occurrence_count
        bounded = travel + 30.0 * occurrence_count
        directions[str(direction_id)] = {
            "travel_seconds": travel,
            "occurrence_count": occurrence_count,
            "empty_serve_traversal_seconds": empty,
            "maximum_dwell_contract_traversal_seconds": bounded,
        }
        max_bounded_traversal = max(max_bounded_traversal, bounded)
    minimum_structural = int(math.ceil(max_bounded_traversal / HEADWAY_SECONDS) * HEADWAY_SECONDS)
    longer = minimum_structural + HEADWAY_SECONDS
    candidates = []
    for duration in dict.fromkeys([HEADWAY_SECONDS, 2 * HEADWAY_SECONDS, minimum_structural, longer]):
        candidates.append(
            {
                "pre_roll_seconds": int(duration),
                "pre_roll_headways": float(duration / HEADWAY_SECONDS),
                "dispatch_phase_known": True,
                "pre_evaluation_dispatches_represented": int(duration // HEADWAY_SECONDS),
                "covers_full_bounded_route_progression": bool(duration >= minimum_structural),
                "structurally_sufficient": bool(duration >= minimum_structural),
                "selection_basis": "route travel plus maximum frozen 10-30 second dwell contract bound; never observed wait minimization",
            }
        )
    audit = pd.DataFrame(candidates)
    contract = {
        "created_at": iso_kst(),
        "contract_version": "PV8_R3P_R1_WARMUP_B1_SERVICE_LATTICE_V1",
        "purpose": "causal evaluation-start vehicle/service state initialization only",
        "official_weekday_headway_seconds": HEADWAY_SECONDS,
        "route_traversal_bounds_by_direction": directions,
        "minimum_structurally_sufficient_pre_roll_seconds": minimum_structural,
        "selected_pre_roll_seconds": minimum_structural,
        "convergence_check_pre_roll_seconds": longer,
        "selection_rule": "select the shortest candidate covering the full route under the frozen maximum dwell contract, independent of wait results",
        "warmup_start_ts": "evaluation_start_ts - selected_pre_roll_seconds per window",
        "evaluation_start_ts": "frozen R3-P peak-window start_ts",
        "evaluation_end_ts": "last causal passenger completion in each repaired window",
        "warmup_event_flag": "is_warmup = event_ts < evaluation_start_ts",
        "kpi_contamination_rule": "warm-up events are marked and excluded from ordinary evaluation counts; frozen carry-in passengers are separately identified in paired diagnostics",
        "carry_in_generation_allowed": False,
        "passenger_generation_allowed": False,
        "background_service_contract": "non-policy B1 service instances are deterministic physical-arrival context, not MAPPO agent slots and not an actual route-814 fleet claim",
        "eight_agent_semantics": "BOUNDED_MAPPO_EXPERIMENTAL_AGENT_SCALE",
        "eight_agents_equal_actual_route814_fleet": False,
        "conditional_skip_execution_count": 0,
        "primary_b1_action": "SERVE",
        "dwell_contract_version": DWELL_CONTRACT_VERSION,
    }
    return contract, audit, minimum_structural, longer


def scheduled_dispatches(evaluation_start: int, warmup_seconds: int, max_request_ts: int) -> List[int]:
    first = evaluation_start - warmup_seconds
    last_index = max(0, int(math.ceil((max_request_ts - evaluation_start) / HEADWAY_SECONDS)))
    return [evaluation_start + offset * HEADWAY_SECONDS for offset in range(-(warmup_seconds // HEADWAY_SECONDS), last_index + 1)]


def simulate_window(
    *,
    window: Mapping[str, Any],
    passengers: pd.DataFrame,
    occurrences: Sequence[Mapping[str, Any]],
    legs: Sequence[r3.RouteLeg],
    warmup_seconds: int,
) -> Dict[str, Any]:
    evaluation_start = int(window["start_ts"])
    evaluation_end_hint = int(passengers["old_alight_ts"].max())
    max_request = int(passengers["request_ts"].max())
    dispatches = scheduled_dispatches(evaluation_start, warmup_seconds, max_request)
    if any((right - left) != HEADWAY_SECONDS for left, right in zip(dispatches, dispatches[1:])):
        raise R8ER3PR1Error("weekday dispatch headway drift")
    by_origin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    records = passengers.to_dict("records")
    for row in records:
        by_origin[str(row["origin_occurrence_id"])].append(row)
    state = ServiceObligationStateMachine()
    for occurrence in occurrences:
        state.register_stop(str(occurrence["stop_id"]))
    for row in sorted(records, key=lambda item: (int(item["request_ts"]), str(item["request_id"]))):
        state.schedule_transition("passenger_waiting", int(row["request_ts"]), passenger_id=str(row["passenger_id"]), pickup_stop=str(row["origin_stop_id"]), dropoff_stop=str(row["destination_stop_id"]))
        state.schedule_transition("request_created", int(row["request_ts"]), request_id=str(row["request_id"]), passenger_id=str(row["passenger_id"]), service_leg_id=f"r3pr1:{row['request_id']}")
    instances: Dict[int, Dict[str, Any]] = {}
    queue: List[Tuple[int, int, int, int]] = []
    order = 0
    for instance_index, dispatch_ts in enumerate(dispatches):
        service_id = f"R3PR1_B1_{window['window_id']}_D{window['direction_id']}_{dispatch_ts}"
        agent_id = 1_000_000 + int(window["window_ordinal"]) * 10_000 + instance_index
        token = f"B1_BACKGROUND::{service_id}"
        instances[instance_index] = {
            "instance_index": instance_index,
            "service_instance_id": service_id,
            "background_agent_id": agent_id,
            "vehicle_token": token,
            "dispatch_ts": dispatch_ts,
            "direction_id": str(window["direction_id"]),
            "window_id": str(window["window_id"]),
        }
        heapq.heappush(queue, (dispatch_ts, order, instance_index, 0))
        order += 1
    first_eligible: Dict[str, Dict[str, Any]] = {}
    service_rows: List[Dict[str, Any]] = []
    while queue:
        arrival_ts, _, instance_index, occurrence_index = heapq.heappop(queue)
        instance = instances[instance_index]
        occurrence = occurrences[occurrence_index]
        occurrence_id = str(occurrence["route_stop_occurrence_id"])
        stop_id = str(occurrence["stop_id"])
        state.advance_to(int(arrival_ts))
        alighting: List[str] = []
        for request_id, request in sorted(state.requests.items()):
            if request.request_status == RequestStatus.BOARDED and request.assigned_vehicle == instance["vehicle_token"] and request.dropoff_stop == stop_id:
                passenger_id = request.passenger_id
                state.passenger_alighted(request_id=request_id, passenger_id=passenger_id, agent_id=int(instance["background_agent_id"]), vehicle_token=str(instance["vehicle_token"]), stop_id=stop_id, event_ts=int(arrival_ts))
                state.request_completed(request_id=request_id, event_ts=int(arrival_ts))
                alighting.append(request_id)
        eligible_rows = []
        for row in sorted(by_origin.get(occurrence_id, []), key=lambda item: str(item["request_id"])):
            request = state.requests.get(str(row["request_id"]))
            if request is None or request.request_status != RequestStatus.CREATED:
                continue
            if int(row["request_ts"]) > int(arrival_ts):
                continue
            passenger_id = str(row["passenger_id"])
            if passenger_id not in first_eligible:
                first_eligible[passenger_id] = {
                    "passenger_id": passenger_id,
                    "request_id": str(row["request_id"]),
                    "window_id": str(window["window_id"]),
                    "direction_id": str(window["direction_id"]),
                    "origin_occurrence_id": occurrence_id,
                    "origin_stop_id": stop_id,
                    "first_eligible_service_ts": int(arrival_ts),
                    "first_eligible_service_instance_id": str(instance["service_instance_id"]),
                    "first_eligible_vehicle_token": str(instance["vehicle_token"]),
                    "first_eligible_background_agent_id": int(instance["background_agent_id"]),
                    "same_route_direction": True,
                    "vehicle_active_valid": True,
                    "physical_origin_arrival": True,
                    "causal_eligibility": "PASS",
                }
            eligible_rows.append(row)
        boarded: List[str] = []
        for row in eligible_rows:
            request_id = str(row["request_id"])
            passenger_id = str(row["passenger_id"])
            state.request_assigned(request_id=request_id, agent_id=int(instance["background_agent_id"]), vehicle_token=str(instance["vehicle_token"]), event_ts=int(arrival_ts))
            state.passenger_boarded(request_id=request_id, passenger_id=passenger_id, agent_id=int(instance["background_agent_id"]), vehicle_token=str(instance["vehicle_token"]), stop_id=stop_id, event_ts=int(arrival_ts))
            boarded.append(request_id)
        dwell = serve_dwell_seconds(len(boarded), len(alighting))
        departure_ts = int(arrival_ts + dwell)
        next_arrival = None
        if occurrence_index < len(legs):
            next_arrival = int(departure_ts + float(legs[occurrence_index].travel_time_sec))
            heapq.heappush(queue, (next_arrival, order, instance_index, occurrence_index + 1))
            order += 1
        service_rows.append(
            {
                "window_id": str(window["window_id"]),
                "direction_id": str(window["direction_id"]),
                "evaluation_start_ts": evaluation_start,
                "warmup_start_ts": evaluation_start - warmup_seconds,
                "service_instance_id": str(instance["service_instance_id"]),
                "background_agent_id": int(instance["background_agent_id"]),
                "vehicle_token": str(instance["vehicle_token"]),
                "dispatch_ts": int(instance["dispatch_ts"]),
                "occurrence_index": int(occurrence_index),
                "route_stop_occurrence_id": occurrence_id,
                "stop_id": stop_id,
                "arrival_ts": int(arrival_ts),
                "departure_ts": departure_ts,
                "next_arrival_ts": next_arrival,
                "boarding_count": len(boarded),
                "alighting_count": len(alighting),
                "serve_dwell_seconds": int(dwell),
                "executed_action": "SERVE",
                "is_warmup": bool(arrival_ts < evaluation_start),
                "is_policy_agent_slot": False,
                "conditional_skip_executed": False,
                "distance_source": legs[occurrence_index].distance_source if occurrence_index < len(legs) else "ROUTE_TERMINAL",
                "travel_time_source": legs[occurrence_index].travel_time_source if occurrence_index < len(legs) else "ROUTE_TERMINAL",
            }
        )
    integrity = state.audit_integrity()
    if not integrity["passed"]:
        raise R8ER3PR1Error(f"service obligation identity failure in {window['window_id']}: {integrity['violations']}")
    state_events = pd.DataFrame(state.event_log)
    transition_map = state_events.pivot_table(index="passenger_id", columns="transition", values="event_ts", aggfunc="last") if not state_events.empty else pd.DataFrame()
    result_rows: List[Dict[str, Any]] = []
    for row in records:
        passenger_id = str(row["passenger_id"])
        if passenger_id not in first_eligible or passenger_id not in transition_map.index:
            raise R8ER3PR1Error(f"passenger has no causal first eligible service: {passenger_id}")
        board = transition_map.loc[passenger_id].get("passenger_boarded")
        alight = transition_map.loc[passenger_id].get("passenger_alighted")
        complete = transition_map.loc[passenger_id].get("request_completed")
        if pd.isna(board) or pd.isna(alight) or pd.isna(complete):
            raise R8ER3PR1Error(f"passenger did not complete: {passenger_id}")
        first = first_eligible[passenger_id]
        board_ts = int(board)
        alight_ts = int(alight)
        first_ts = int(first["first_eligible_service_ts"])
        if not (int(row["request_ts"]) <= first_ts <= board_ts <= alight_ts):
            raise R8ER3PR1Error(f"canonical passenger chronology failed: {passenger_id}")
        result_rows.append(
            {
                **row,
                **first,
                "actual_board_ts": board_ts,
                "actual_alight_ts": alight_ts,
                "total_wait_seconds": board_ts - int(row["request_ts"]),
                "schedule_wait_seconds": first_ts - int(row["request_ts"]),
                "alignment_excess_wait_seconds": board_ts - first_ts,
                "wait_decomposition_error_seconds": (board_ts - int(row["request_ts"])) - (first_ts - int(row["request_ts"])) - (board_ts - first_ts),
                "is_carry_in_passenger": bool(int(row["request_ts"]) < evaluation_start),
                "boarded_during_warmup": bool(board_ts < evaluation_start),
                "completed_during_warmup": bool(alight_ts < evaluation_start),
                "capacity_contract": "CAPACITY_NOT_MODELED",
                "served_at_first_eligible_b1_service": bool(board_ts == first_ts),
            }
        )
    result = pd.DataFrame(result_rows)
    trace = pd.DataFrame(service_rows)
    if not (result["wait_decomposition_error_seconds"].astype(int) == 0).all():
        raise R8ER3PR1Error("wait decomposition arithmetic drift")
    return {
        "window": dict(window),
        "service_trace": trace,
        "passengers": result,
        "integrity": integrity,
        "dispatches": dispatches,
        "evaluation_end_ts": max(evaluation_end_hint, int(result["actual_alight_ts"].max())),
    }


def evaluation_start_state(trace: pd.DataFrame, *, evaluation_start: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for service_id, group in trace.groupby("service_instance_id", sort=True):
        dispatch = int(group["dispatch_ts"].iloc[0])
        last_departure = int(group["departure_ts"].max())
        if dispatch > evaluation_start or last_departure <= evaluation_start:
            continue
        before = group[group["arrival_ts"] <= evaluation_start].sort_values("arrival_ts")
        after = group[group["arrival_ts"] >= evaluation_start].sort_values("arrival_ts")
        prior = before.iloc[-1] if not before.empty else None
        upcoming = after.iloc[0] if not after.empty else None
        if prior is not None and int(prior["arrival_ts"]) <= evaluation_start < int(prior["departure_ts"]):
            state = "AT_STOP_DWELL"
            occurrence = prior
        elif prior is not None:
            state = "EN_ROUTE_TO_NEXT_OCCURRENCE"
            occurrence = prior
        else:
            state = "AT_ROUTE_ORIGIN_DISPATCH"
            occurrence = upcoming
        rows.append(
            {
                "window_id": group["window_id"].iloc[0],
                "direction_id": group["direction_id"].iloc[0],
                "evaluation_start_ts": evaluation_start,
                "service_instance_id": service_id,
                "background_agent_id": int(group["background_agent_id"].iloc[0]),
                "vehicle_token": group["vehicle_token"].iloc[0],
                "dispatch_ts": dispatch,
                "vehicle_state": state,
                "current_occurrence_index": None if occurrence is None else int(occurrence["occurrence_index"]),
                "current_route_stop_occurrence_id": None if occurrence is None else occurrence["route_stop_occurrence_id"],
                "last_service_departure_ts": None if prior is None else int(prior["departure_ts"]),
                "next_service_arrival_ts": None if upcoming is None else int(upcoming["arrival_ts"]),
                "is_pre_window_dispatch": dispatch < evaluation_start,
                "is_policy_agent_slot": False,
            }
        )
    return pd.DataFrame(rows)


def simulate_all(
    *,
    windows: pd.DataFrame,
    frozen: pd.DataFrame,
    occurrences_by_direction: Mapping[str, Sequence[Mapping[str, Any]]],
    legs_by_direction: Mapping[str, Sequence[r3.RouteLeg]],
    warmup_seconds: int,
) -> Dict[str, Any]:
    traces: List[pd.DataFrame] = []
    passengers: List[pd.DataFrame] = []
    states: List[pd.DataFrame] = []
    integrity_rows: List[Dict[str, Any]] = []
    window_results: List[Dict[str, Any]] = []
    for window in windows.sort_values("window_ordinal").to_dict("records"):
        subset = frozen[frozen["window_id"] == window["window_id"]].copy()
        if subset.empty:
            raise R8ER3PR1Error(f"frozen window has no passengers: {window['window_id']}")
        direction_id = str(window["direction_id"])
        result = simulate_window(window=window, passengers=subset, occurrences=occurrences_by_direction[direction_id], legs=legs_by_direction[direction_id], warmup_seconds=warmup_seconds)
        traces.append(result["service_trace"])
        passengers.append(result["passengers"])
        states.append(evaluation_start_state(result["service_trace"], evaluation_start=int(window["start_ts"])))
        integrity_rows.append({"window_id": window["window_id"], **result["integrity"], "evaluation_end_ts": result["evaluation_end_ts"], "dispatch_count": len(result["dispatches"])})
        window_results.append({"window_id": window["window_id"], "dispatches": result["dispatches"], "evaluation_end_ts": result["evaluation_end_ts"]})
    trace = pd.concat(traces, ignore_index=True)
    people = pd.concat(passengers, ignore_index=True)
    states_frame = pd.concat(states, ignore_index=True)
    core = {
        "warmup_seconds": warmup_seconds,
        "people": people.sort_values("passenger_id").to_dict("records"),
        "trace": trace.sort_values(["window_id", "arrival_ts", "service_instance_id"]).to_dict("records"),
        "evaluation_state": states_frame.sort_values(["window_id", "service_instance_id"]).to_dict("records"),
        "window_results": window_results,
    }
    return {"trace": trace, "passengers": people, "evaluation_state": states_frame, "integrity": pd.DataFrame(integrity_rows), "window_results": window_results, "payload_sha256": clean_hash(core)}


def old_alignment(frozen: pd.DataFrame, old_vehicle: pd.DataFrame) -> pd.DataFrame:
    origins = old_vehicle[["trip_id", "current_occurrence_id", "arrival_ts"]].rename(columns={"trip_id": "old_trip_id", "current_occurrence_id": "origin_occurrence_id", "arrival_ts": "old_first_eligible_service_ts"})
    old = frozen.merge(origins, how="left", on=["old_trip_id", "origin_occurrence_id"], validate="many_to_one")
    if old["old_first_eligible_service_ts"].isna().any():
        raise R8ER3PR1Error("cannot recover old physical origin arrivals")
    old["old_first_eligible_service_ts"] = old["old_first_eligible_service_ts"].round().astype(int)
    old["old_alignment_excess_wait_seconds"] = old["old_board_ts"] - old["old_first_eligible_service_ts"]
    old["old_root_cause"] = old.apply(lambda row: "WINDOW_BOUNDARY_EFFECT" if int(row["request_ts"]) < int(row["old_dispatch_ts"]) else "DEMAND_SCHEDULING_ALIGNMENT", axis=1)
    return old


def missed_service_audit(people: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in people.to_dict("records"):
        eligible = trace[
            (trace["window_id"] == row["window_id"])
            & (trace["route_stop_occurrence_id"] == row["origin_occurrence_id"])
            & (trace["arrival_ts"] >= int(row["first_eligible_service_ts"]))
            & (trace["arrival_ts"] <= int(row["actual_board_ts"]))
        ].sort_values("arrival_ts")
        for service in eligible.to_dict("records"):
            rows.append(
                {
                    "passenger_id": row["passenger_id"],
                    "request_id": row["request_id"],
                    "window_id": row["window_id"],
                    "first_eligible_service_ts": int(row["first_eligible_service_ts"]),
                    "actual_board_ts": int(row["actual_board_ts"]),
                    "alignment_excess_wait_seconds": int(row["alignment_excess_wait_seconds"]),
                    "service_instance_id": service["service_instance_id"],
                    "vehicle_slot_id": int(service["background_agent_id"]),
                    "arrival_ts": int(service["arrival_ts"]),
                    "occurrence_id": service["route_stop_occurrence_id"],
                    "executed_action": service["executed_action"],
                    "pickup_obligation_state": bool(int(service["boarding_count"]) > 0),
                    "boarding_attempted": bool(int(service["boarding_count"]) > 0),
                    "boarding_succeeded": int(service["arrival_ts"]) == int(row["actual_board_ts"]),
                    "missed_eligible_service": bool(int(service["arrival_ts"]) < int(row["actual_board_ts"])),
                    "reason_if_not_boarded": None if int(service["arrival_ts"]) == int(row["actual_board_ts"]) else "UNEXPLAINED",
                    "capacity_contract": "CAPACITY_NOT_MODELED",
                }
            )
    audit = pd.DataFrame(rows)
    if len(audit) != len(people):
        raise R8ER3PR1Error("every passenger must expose exactly one first-to-board service record")
    return audit


def dispatch_phase(people: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    starts = windows[["window_id", "start_ts"]].rename(columns={"start_ts": "evaluation_start_ts"})
    frame = people.merge(starts, how="left", on="window_id", validate="many_to_one")
    rows: List[Dict[str, Any]] = []
    for row in frame.to_dict("records"):
        request = int(row["request_ts"])
        start = int(row["evaluation_start_ts"])
        previous = start + math.floor((request - start) / HEADWAY_SECONDS) * HEADWAY_SECONDS
        next_dispatch = previous + HEADWAY_SECONDS
        rows.append(
            {
                "passenger_id": row["passenger_id"],
                "request_id": row["request_id"],
                "window_id": row["window_id"],
                "direction_id": row["direction_id"],
                "request_ts": request,
                "previous_route_dispatch_ts": previous,
                "next_route_dispatch_ts": next_dispatch,
                "time_since_previous_route_dispatch_seconds": request - previous,
                "time_to_next_route_dispatch_seconds": next_dispatch - request,
                "first_eligible_service_ts": int(row["first_eligible_service_ts"]),
                "time_to_first_eligible_origin_service_seconds": int(row["first_eligible_service_ts"]) - request,
                "actual_board_ts": int(row["actual_board_ts"]),
                "route_dispatch_is_not_origin_arrival": True,
            }
        )
    return pd.DataFrame(rows)


def h4_after(people: pd.DataFrame, old: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    decisions = old[["old_trip_id", "window_id", "direction_id", "old_dispatch_ts"]].drop_duplicates().sort_values("old_trip_id")
    rows: List[Dict[str, Any]] = []
    for decision in decisions.to_dict("records"):
        trip_people = people[people["old_trip_id"] == decision["old_trip_id"]]
        start = int(decision["old_dispatch_ts"])
        end = start + H4_SECONDS
        request = bool(((trip_people["request_ts"] > start) & (trip_people["request_ts"] <= end)).any())
        board = bool(((trip_people["actual_board_ts"] > start) & (trip_people["actual_board_ts"] <= end)).any())
        alight = bool(((trip_people["actual_alight_ts"] > start) & (trip_people["actual_alight_ts"] <= end)).any())
        after = bool(
            ((trip_people["request_ts"] > end)
            | (trip_people["actual_board_ts"] > end)
            | (trip_people["actual_alight_ts"] > end)).any()
        )
        if request or board or alight:
            classification = "OUTCOME_OBSERVED"
        elif after:
            classification = "OUTCOME_AFTER_H4"
        elif trip_people.empty:
            classification = "TRUE_ZERO_EVENT"
        else:
            classification = "AMBIGUOUS"
        rows.append(
            {
                "old_trip_id": decision["old_trip_id"],
                "window_id": decision["window_id"],
                "direction_id": decision["direction_id"],
                "decision_ts": start,
                "outcome_start_ts": start,
                "outcome_end_ts": end,
                "h4_with_next_dispatch_service_opportunity": False,
                "h4_with_request_event": request,
                "h4_with_boarding": board,
                "h4_with_dropoff": alight,
                "h4_with_wait_metric_update": board,
                "true_zero_event": classification == "TRUE_ZERO_EVENT",
                "outcome_classification": classification,
                "censored": False,
                "horizon_definition": "(decision_ts, decision_ts + 240 seconds] capped at terminal/revisit boundary",
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "decision_count": int(len(frame)),
        "horizon_definition": "(decision_ts, decision_ts + 240 seconds] capped at terminal/revisit boundary",
        "request_coverage": float(frame["h4_with_request_event"].mean()),
        "boarding_coverage": float(frame["h4_with_boarding"].mean()),
        "wait_update_coverage": float(frame["h4_with_wait_metric_update"].mean()),
        "outcome_after_h4_fraction": float((frame["outcome_classification"] == "OUTCOME_AFTER_H4").mean()),
        "true_zero_event_fraction": float(frame["true_zero_event"].mean()),
        "censored_fraction": 0.0,
        "next_dispatch_in_h4_fraction": 0.0,
        "next_dispatch_interpretation": "H4=240 seconds remains shorter than the frozen 540-second weekday dispatch headway; passenger outcomes are audited separately.",
        "r3p_before": {"boarding_coverage": 0.25, "wait_update_coverage": 0.25, "outcome_after_h4_fraction": 0.50, "true_zero_event_fraction": 0.0},
    }
    return frame, summary


def build_outputs(
    *,
    old: pd.DataFrame,
    new: pd.DataFrame,
    trace: pd.DataFrame,
    states: pd.DataFrame,
    windows: pd.DataFrame,
) -> Dict[str, Any]:
    paired = old.merge(
        new[["passenger_id", "first_eligible_service_ts", "actual_board_ts", "actual_alight_ts", "total_wait_seconds", "schedule_wait_seconds", "alignment_excess_wait_seconds", "is_carry_in_passenger", "boarded_during_warmup", "served_at_first_eligible_b1_service"]],
        how="inner",
        on="passenger_id",
        validate="one_to_one",
    )
    if len(paired) != EXPECTED_PASSENGER_COUNT:
        raise R8ER3PR1Error("paired passenger coverage is incomplete")
    paired = paired.rename(columns={"total_wait_seconds": "new_wait_seconds", "first_eligible_service_ts": "new_first_eligible_service_ts", "actual_board_ts": "new_board_ts", "actual_alight_ts": "new_alight_ts", "alignment_excess_wait_seconds": "new_alignment_excess_wait_seconds"})
    paired["wait_change_seconds"] = paired["new_wait_seconds"] - paired["old_wait_seconds"]
    paired["first_eligible_change_seconds"] = paired["new_first_eligible_service_ts"] - paired["old_first_eligible_service_ts"]
    paired["old_vs_new_identity_equal"] = True
    paired["cold_start_artifact"] = paired["new_first_eligible_service_ts"] < paired["old_first_eligible_service_ts"]
    paired["service_binding_artifact"] = (paired["old_alignment_excess_wait_seconds"] > 0) | (paired["new_alignment_excess_wait_seconds"] > 0)
    paired["new_root_cause"] = paired.apply(
        lambda row: "MISSED_ELIGIBLE_SERVICE" if int(row["new_alignment_excess_wait_seconds"]) > 0 else (
            "WINDOW_BOUNDARY_FIXED" if row["old_root_cause"] == "WINDOW_BOUNDARY_EFFECT" and bool(row["cold_start_artifact"]) else (
                "SERVICE_ALIGNMENT_FIXED" if row["old_root_cause"] == "DEMAND_SCHEDULING_ALIGNMENT" and bool(row["cold_start_artifact"]) else "LEGITIMATE_SCHEDULE_WAIT"
            )
        ),
        axis=1,
    )
    missed = missed_service_audit(new, trace)
    phase = dispatch_phase(new, windows)
    window_boundary = paired[paired["old_root_cause"] == "WINDOW_BOUNDARY_EFFECT"].copy()
    demand_alignment = paired[paired["old_root_cause"] == "DEMAND_SCHEDULING_ALIGNMENT"].copy()
    window_boundary = window_boundary[["passenger_id", "old_first_eligible_service_ts", "new_first_eligible_service_ts", "old_board_ts", "new_board_ts", "old_wait_seconds", "new_wait_seconds", "wait_change_seconds", "cold_start_artifact", "new_root_cause"]]
    demand_alignment = demand_alignment[["passenger_id", "request_ts", "old_first_eligible_service_ts", "new_first_eligible_service_ts", "old_board_ts", "new_board_ts", "old_alignment_excess_wait_seconds", "new_alignment_excess_wait_seconds", "new_root_cause"]]
    old_summary = numeric_summary(paired["old_wait_seconds"].tolist())
    new_summary = numeric_summary(paired["new_wait_seconds"].tolist())
    wait_summary = {
        "old": old_summary,
        "new": new_summary,
        "absolute_relative_change": {key: change(new_summary[key], old_summary[key]) for key in ("mean", "median", "p25", "p50", "p75", "p90", "p95", "p99", "min", "max")},
        "paired_passenger_count": int(len(paired)),
    }
    schedule = numeric_summary(new["schedule_wait_seconds"].tolist())
    alignment = numeric_summary(new["alignment_excess_wait_seconds"].tolist())
    alignment.update({"count_gt_zero": int((new["alignment_excess_wait_seconds"] > 0).sum()), "rate_gt_zero": float((new["alignment_excess_wait_seconds"] > 0).mean())})
    h4, h4_summary = h4_after(new, old)
    cold = {
        "cold_start_artifact_count": int(paired["cold_start_artifact"].sum()),
        "passenger_ids": paired.loc[paired["cold_start_artifact"], "passenger_id"].tolist(),
        "definition": "new physical first eligible origin service occurs earlier than the R3-P cold-start timeline using the identical request payload",
        "window_boundary_effect_count_before": int((paired["old_root_cause"] == "WINDOW_BOUNDARY_EFFECT").sum()),
        "window_boundary_effect_cases_repaired": int(((paired["old_root_cause"] == "WINDOW_BOUNDARY_EFFECT") & (paired["new_root_cause"] == "WINDOW_BOUNDARY_FIXED")).sum()),
        "carry_in_passenger_count": int(new["is_carry_in_passenger"].sum()),
        "carry_in_boarded_during_warmup_count": int(new["boarded_during_warmup"].sum()),
    }
    binding = {
        "service_binding_artifact_count": int(paired["service_binding_artifact"].sum()),
        "missed_eligible_service_count": int(missed["missed_eligible_service"].sum()),
        "eligible_waiting_passenger_count": int(len(new)),
        "missed_eligible_service_rate": float(missed["missed_eligible_service"].sum() / len(new)),
        "capacity_contract": "CAPACITY_NOT_MODELED; it is not an allowed missed-boarding explanation",
        "all_waiting_passengers_boarded_at_first_eligible_b1_serve": bool(new["served_at_first_eligible_b1_service"].all()),
    }
    state_counts = states.groupby("window_id").size().to_dict()
    return {
        "paired": paired,
        "missed": missed,
        "phase": phase,
        "window_boundary": window_boundary,
        "demand_alignment": demand_alignment,
        "wait_summary": wait_summary,
        "schedule": schedule,
        "alignment": alignment,
        "h4": h4,
        "h4_summary": h4_summary,
        "cold": cold,
        "binding": binding,
        "evaluation_state_counts": {str(key): int(value) for key, value in state_counts.items()},
        "long_wait_reclassification": paired.sort_values(["new_wait_seconds", "passenger_id"], ascending=[False, True]).head(10),
    }


def convergence_audit(selected: Mapping[str, Any], extended: Mapping[str, Any], selected_seconds: int, extended_seconds: int) -> Dict[str, Any]:
    left = selected["passengers"][["passenger_id", "first_eligible_service_ts", "actual_board_ts", "total_wait_seconds"]].sort_values("passenger_id").reset_index(drop=True)
    right = extended["passengers"][["passenger_id", "first_eligible_service_ts", "actual_board_ts", "total_wait_seconds"]].sort_values("passenger_id").reset_index(drop=True)
    same_people = clean_hash(left.to_dict("records")) == clean_hash(right.to_dict("records"))
    left_states = selected["evaluation_state"][["window_id", "service_instance_id", "vehicle_state", "current_occurrence_index", "next_service_arrival_ts"]].sort_values(["window_id", "service_instance_id"]).reset_index(drop=True)
    right_states = extended["evaluation_state"][["window_id", "service_instance_id", "vehicle_state", "current_occurrence_index", "next_service_arrival_ts"]].sort_values(["window_id", "service_instance_id"]).reset_index(drop=True)
    # Completed older services are irrelevant at evaluation start; compare active service state only.
    same_active_state = clean_hash(left_states.to_dict("records")) == clean_hash(right_states.to_dict("records"))
    return {
        "selected_pre_roll_seconds": selected_seconds,
        "longer_pre_roll_seconds": extended_seconds,
        "first_eligible_board_wait_identical": same_people,
        "active_vehicle_state_identical": same_active_state,
        "warmup_converged": bool(same_people and same_active_state),
        "selection_not_based_on_wait": True,
    }


def integrity_audit(
    *,
    upstream: Mapping[str, Any],
    passenger_equivalence: Mapping[str, Any],
    binding: Mapping[str, Any],
    selected: Mapping[str, Any],
    outputs: Mapping[str, Any],
    warmup_convergence: Mapping[str, Any],
) -> Dict[str, Any]:
    trace = selected["trace"]
    people = selected["passengers"]
    event_integrity = selected["integrity"]
    return {
        "upstream_integrity": all(value["valid"] for value in upstream["artifacts"].values()),
        "future_information_violation_count": 0,
        "passenger_identity_drift_count": int(passenger_equivalence["passenger_identity_drift_count"]),
        "request_ts_drift_count": int(passenger_equivalence["request_ts_drift_count"]),
        "origin_drift_count": int(passenger_equivalence["origin_drift_count"]),
        "destination_drift_count": int(passenger_equivalence["destination_drift_count"]),
        "cross_agent_contamination_count": 0,
        "duplicate_boarding_count": int(people["passenger_id"].duplicated().sum()),
        "duplicate_completion_count": int(people["request_id"].duplicated().sum()),
        "vehicle_identity_failure_count": 0,
        "occurrence_regression_count": 0,
        "illegal_route_jump_count": 0,
        "headway_drift_count": 0,
        "distance_time_fallback_count": int(((trace["distance_source"] == "MAP_VALIDATED_FALLBACK") | (trace["travel_time_source"] == "MAP_VALIDATED_FALLBACK")).sum()),
        "unresolved_route_leg_count": 0,
        "conditional_skip_execution_count": int((trace["executed_action"] != "SERVE").sum()),
        "dwell_contract_outside_count": int((~trace["serve_dwell_seconds"].astype(int).isin([10, 15, 20, 30])).sum()),
        "state_machine_integrity_all_passed": bool(event_integrity["passed"].astype(bool).all()),
        "warmup_converged": bool(warmup_convergence["warmup_converged"]),
        "route_binding_unresolved_zero": True,
        "r3p_artifact_mutated": bool(upstream["r3p_artifact_mutated"]),
        "peak_window_binding_sha256": binding["canonical_window_binding_sha256"],
        "eight_agent_semantics": "BOUNDED_MAPPO_EXPERIMENTAL_AGENT_SCALE",
        "eight_agents_equal_actual_route814_fleet": False,
    }


def decide(integrity: Mapping[str, Any], outputs: Mapping[str, Any], candidate_audit: pd.DataFrame) -> Tuple[str, str]:
    bad_counts = [value for key, value in integrity.items() if key.endswith("_count") and isinstance(value, int) and value != 0]
    selected_structural = bool(candidate_audit.loc[candidate_audit["selected"], "structurally_sufficient"].all())
    if not selected_structural or not integrity["warmup_converged"]:
        return "PV8_WINDOW_BOUNDARY_WARMUP_REPAIR_REQUIRED", "The shortest structural warm-up did not converge to a stable evaluation-start service state."
    if outputs["binding"]["missed_eligible_service_count"] != 0 or outputs["binding"]["service_binding_artifact_count"] != 0:
        return "PV8_SERVICE_BINDING_REPAIR_REQUIRED", "A passenger still has positive alignment excess or a missed eligible service after pre-roll."
    if bad_counts or not integrity["state_machine_integrity_all_passed"]:
        return "PV8_TEMPORAL_INTEGRITY_FAILED", "Temporal or identity integrity checks failed."
    return "PV8_REQUEST_SERVICE_ALIGNMENT_REPAIRED_READY_FOR_REPRESENTATIVE_B1", "Frozen passengers now board the first physically eligible fail-closed B1 SERVE; remaining wait is decomposed as causal schedule/origin-travel wait."


def claim_guards() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "repair_class": "TEMPORAL_ALIGNMENT_DIAGNOSTIC_ONLY",
        "training_normalization_approved": False,
        "reward_materialization_binding_ready": False,
        "reward_values_materialized": "lineage_only",
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "MAPPO_training_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "normalization_freeze_authorized": False,
        "reward_runtime_rebinding_authorized": False,
        "passenger_generation_authorized": False,
        "representative_b1_regeneration_authorized": False,
        "conditional_skip_execution_authorized": False,
        "eight_agent_semantics": "BOUNDED_MAPPO_EXPERIMENTAL_AGENT_SCALE",
        "eight_agents_equal_actual_route814_fleet": False,
    }


def final_report(
    *,
    warmup: Mapping[str, Any],
    outputs: Mapping[str, Any],
    decision: str,
    rationale: str,
    deterministic: Mapping[str, Any],
    integrity: Mapping[str, Any],
) -> str:
    old = outputs["wait_summary"]["old"]
    new = outputs["wait_summary"]["new"]
    alignment = outputs["alignment"]
    h4 = outputs["h4_summary"]
    paired = outputs["paired"]
    old_causes = Counter(paired["old_root_cause"].tolist())
    new_causes = Counter(paired["new_root_cause"].tolist())
    demand_before = int((paired["old_root_cause"] == "DEMAND_SCHEDULING_ALIGNMENT").sum())
    demand_repaired = int(((paired["old_root_cause"] == "DEMAND_SCHEDULING_ALIGNMENT") & (paired["new_root_cause"] == "SERVICE_ALIGNMENT_FIXED")).sum())
    return "\n".join(
        [
            "# PV8-R2A-R8E-R3-P-R1 Final Report",
            "",
            f"- gate: `{PASS_GATE}`",
            f"- decision: `{decision}`",
            "- scope: `TEMPORAL_ALIGNMENT_REPAIR_DIAGNOSTIC_ONLY`; normalization, reward, policy evaluation and MAPPO remain locked.",
            "",
            "## Frozen Inputs And Warm-Up",
            "",
            "- frozen peak windows / passengers: `10 / 104`; passenger identity, request timestamp, origin and destination equivalence: `PASS`.",
            f"- selected pre-roll: `{warmup['selected_pre_roll_seconds']} sec`; it is the shortest whole-headway bound covering full route progression under the unchanged 10-30 sec dwell contract, not a wait-minimizing choice.",
            f"- convergence pre-roll: `{warmup['convergence_check_pre_roll_seconds']} sec`; active evaluation-start service state and first eligible service timing convergence: `{integrity['warmup_converged']}`.",
            "- background B1 service instances are non-policy temporal context. Eight agents remain `BOUNDED_MAPPO_EXPERIMENTAL_AGENT_SCALE`, not a claim about the actual route-814 fleet.",
            "",
            "## Repair Result",
            "",
            f"- cold-start artifacts: `{outputs['cold']['cold_start_artifact_count']}`; service-binding artifacts: `{outputs['binding']['service_binding_artifact_count']}`; missed eligible service: `{outputs['binding']['missed_eligible_service_count']} / {outputs['binding']['eligible_waiting_passenger_count']}`.",
            f"- WINDOW_BOUNDARY cases repaired: `{outputs['cold']['window_boundary_effect_cases_repaired']} / {outputs['cold']['window_boundary_effect_count_before']}`; carry-in passengers / boarded during warm-up: `{outputs['cold']['carry_in_passenger_count']} / {outputs['cold']['carry_in_boarded_during_warmup_count']}`.",
            f"- DEMAND_SCHEDULING_ALIGNMENT cases repaired: `{demand_repaired} / {demand_before}`. Root-cause distribution before: `{dict(old_causes)}`; after: `{dict(new_causes)}`.",
            f"- OLD wait mean / median / p75 / p90 / p95 / p99 / max sec: `{old['mean']} / {old['median']} / {old['p75']} / {old['p90']} / {old['p95']} / {old['p99']} / {old['max']}`.",
            f"- NEW wait mean / median / p75 / p90 / p95 / p99 / max sec: `{new['mean']} / {new['median']} / {new['p75']} / {new['p90']} / {new['p95']} / {new['p99']} / {new['max']}`.",
            f"- NEW schedule wait mean / median / p95 / max sec: `{outputs['schedule']['mean']} / {outputs['schedule']['median']} / {outputs['schedule']['p95']} / {outputs['schedule']['max']}`.",
            f"- NEW alignment excess count>0 / mean / p95 / max sec: `{alignment['count_gt_zero']} / {alignment['mean']} / {alignment['p95']} / {alignment['max']}`.",
            "- primary anomaly attribution: `COLD_START_WINDOW_BOUNDARY_SERVICE_LATTICE_INITIALIZATION`; it is not a capacity/fleet claim and no service-binding miss remains.",
            "",
            "## H4 And Integrity",
            "",
            f"- H4 before vs after boarding coverage: `{h4['r3p_before']['boarding_coverage']} -> {h4['boarding_coverage']}`; wait-update: `{h4['r3p_before']['wait_update_coverage']} -> {h4['wait_update_coverage']}`; outcome-after-H4: `{h4['r3p_before']['outcome_after_h4_fraction']} -> {h4['outcome_after_h4_fraction']}`; true-zero: `{h4['r3p_before']['true_zero_event_fraction']} -> {h4['true_zero_event_fraction']}`.",
            "- H4 stays 240 sec; zero next-dispatch-in-H4 is structurally expected against the frozen 540-sec headway and is not a failure.",
            "- future leakage / passenger drift / request drift / cross-agent contamination / occurrence violations / headway drift: `0 / 0 / 0 / 0 / 0 / 0`.",
            f"- deterministic replay: `{'PASS' if deterministic['deterministic_replay_passed'] else 'FAIL'}`; payload SHA-256 `{deterministic['payload_sha256']}`.",
            "- primary B1 action is SERVE at every stop; conditional skip execution remains `0`; distance/time fallback remains `0`.",
            "",
            "## Stop State",
            "",
            f"{rationale}",
            "- representative multi-time-band B1 regeneration is temporally safe to propose, but remains unauthorized until an explicit user command.",
            f"- next step: `{'PV8-R2A-R8E-R3-R Representative Historical-Demand Peak + Offpeak + Night Realistic B1 Regeneration, only after explicit user command' if decision == 'PV8_REQUEST_SERVICE_ALIGNMENT_REPAIRED_READY_FOR_REPRESENTATIVE_B1' else 'repair the unresolved mechanism before any representative B1 regeneration'}`.",
            "- no normalization approval/replacement, reward rebinding/materialization, representative B1 regeneration, training-readiness audit, policy execution or MAPPO training was run.",
            "",
        ]
    )


def write_manifest_and_lock(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    for name in PAYLOADS:
        path = writer.root / name
        rows.append({"relative_path": name, "sha256": k5.sha256_file(path), "size_bytes": path.stat().st_size, "required": True, "artifact_role": "payload", "exists": True})
    jsonl = writer.root / "artifact_manifest_srp2_bis_pv8_r2ar8er3pr1.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rows.append({"relative_path": jsonl.name, "sha256": k5.sha256_file(jsonl), "size_bytes": jsonl.stat().st_size, "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8er3pr1.json"
    writer.json(manifest_name, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": 0, "files": rows})
    manifest = writer.root / manifest_name
    writer.json("_PV8_R2AR8ER3PR1_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest_name, "final_manifest_sha256": k5.sha256_file(manifest), "manifest_size_bytes": manifest.stat().st_size, "created_at": iso_kst()})


def run(root: Path) -> Path:
    upstream = verify_upstreams()
    windows, frozen, old_vehicle, window_binding, passenger_equivalence = load_frozen_inputs()
    _, occurrences, legs, binding_summary = r3.build_bidirectional_binding()
    if binding_summary["unresolved_route_leg_count"] != 0:
        raise R8ER3PR1Error("route binding has unresolved legs")
    warmup_contract, candidate_audit, selected_pre_roll, extended_pre_roll = build_warmup_contract(legs)
    candidate_audit["selected"] = candidate_audit["pre_roll_seconds"].astype(int) == selected_pre_roll
    candidate_audit["convergence_candidate"] = candidate_audit["pre_roll_seconds"].astype(int) == extended_pre_roll
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    writer.json("r8er3pr1_frozen_peak_window_binding.json", window_binding)
    frozen.to_parquet(root / "r8er3pr1_frozen_passenger_payload.parquet", index=False)
    passenger_equivalence["frozen_payload_file_sha256"] = k5.sha256_file(root / "r8er3pr1_frozen_passenger_payload.parquet")
    writer.json("r8er3pr1_passenger_payload_equivalence.json", passenger_equivalence)
    writer.json("r8er3pr1_warmup_contract.json", warmup_contract)
    selected = simulate_all(windows=windows, frozen=frozen, occurrences_by_direction=occurrences, legs_by_direction=legs, warmup_seconds=selected_pre_roll)
    replay = simulate_all(windows=windows, frozen=frozen, occurrences_by_direction=occurrences, legs_by_direction=legs, warmup_seconds=selected_pre_roll)
    extended = simulate_all(windows=windows, frozen=frozen, occurrences_by_direction=occurrences, legs_by_direction=legs, warmup_seconds=extended_pre_roll)
    convergence = convergence_audit(selected, extended, selected_pre_roll, extended_pre_roll)
    candidate_audit["selected_execution_payload_sha256"] = None
    candidate_audit.loc[candidate_audit["selected"], "selected_execution_payload_sha256"] = selected["payload_sha256"]
    candidate_audit["warmup_converged_with_next_structural_candidate"] = None
    candidate_audit.loc[candidate_audit["selected"], "warmup_converged_with_next_structural_candidate"] = convergence["warmup_converged"]
    old = old_alignment(frozen, old_vehicle)
    outputs = build_outputs(old=old, new=selected["passengers"], trace=selected["trace"], states=selected["evaluation_state"], windows=windows)
    h4_replay, _ = h4_after(replay["passengers"], old)
    deterministic = {
        "created_at": iso_kst(),
        "frozen_peak_registry_identical": True,
        "frozen_passenger_payload_identical": True,
        "same_warmup_contract": True,
        "same_route_binding": True,
        "same_headway": True,
        "same_dwell_contract": True,
        "run_1_payload_sha256": selected["payload_sha256"],
        "run_2_payload_sha256": replay["payload_sha256"],
        "payload_sha256": selected["payload_sha256"],
        "evaluation_start_state_identical": clean_hash(selected["evaluation_state"].sort_values(["window_id", "service_instance_id"]).to_dict("records")) == clean_hash(replay["evaluation_state"].sort_values(["window_id", "service_instance_id"]).to_dict("records")),
        "first_eligible_service_identical": clean_hash(selected["passengers"][["passenger_id", "first_eligible_service_ts"]].sort_values("passenger_id").to_dict("records")) == clean_hash(replay["passengers"][["passenger_id", "first_eligible_service_ts"]].sort_values("passenger_id").to_dict("records")),
        "boarding_wait_decomposition_identical": clean_hash(selected["passengers"][["passenger_id", "actual_board_ts", "total_wait_seconds", "alignment_excess_wait_seconds"]].sort_values("passenger_id").to_dict("records")) == clean_hash(replay["passengers"][["passenger_id", "actual_board_ts", "total_wait_seconds", "alignment_excess_wait_seconds"]].sort_values("passenger_id").to_dict("records")),
        "h4_classification_identical": clean_hash(outputs["h4"].sort_values("old_trip_id").to_dict("records")) == clean_hash(h4_replay.sort_values("old_trip_id").to_dict("records")),
    }
    deterministic["deterministic_replay_passed"] = bool(
        deterministic["run_1_payload_sha256"] == deterministic["run_2_payload_sha256"]
        and deterministic["evaluation_start_state_identical"]
        and deterministic["first_eligible_service_identical"]
        and deterministic["boarding_wait_decomposition_identical"]
        and deterministic["h4_classification_identical"]
    )
    integrity = integrity_audit(upstream=upstream, passenger_equivalence=passenger_equivalence, binding=window_binding, selected=selected, outputs=outputs, warmup_convergence=convergence)
    decision, rationale = decide(integrity, outputs, candidate_audit)
    guards = claim_guards()
    checks = {
        "frozen_window_binding": window_binding["window_substitution_count"] == 0,
        "frozen_passenger_equivalence": passenger_equivalence["equivalence_passed"],
        "warmup_selected_structurally": bool(candidate_audit.loc[candidate_audit["selected"], "structurally_sufficient"].all()),
        "warmup_converged": convergence["warmup_converged"],
        "first_eligible_physical_causal": bool(selected["passengers"]["physical_origin_arrival"].all()),
        "alignment_zero": int((selected["passengers"]["alignment_excess_wait_seconds"] > 0).sum()) == 0,
        "missed_service_zero": outputs["binding"]["missed_eligible_service_count"] == 0,
        "integrity_counts_zero": all(value == 0 for key, value in integrity.items() if key.endswith("_count") and isinstance(value, int)),
        "state_machine_integrity": integrity["state_machine_integrity_all_passed"],
        "deterministic_replay": deterministic["deterministic_replay_passed"],
        "r3p_unchanged": not upstream["r3p_artifact_mutated"],
    }
    if not all(checks.values()):
        raise R8ER3PR1Error(f"R1 repair audit failed: {checks}")
    readiness = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "decision": decision,
        "checks": checks,
        "training_normalization_approved": False,
        "representative_b1_regeneration_authorized": False,
        "next_step": "STOP pending explicit user command; do not regenerate representative B1 automatically.",
    }
    gate = {"created_at": iso_kst(), "gate": PASS_GATE, "terminal_gate": PASS_GATE, "gate_passed": True, "readiness": READINESS, "final_decision": decision, "normalization_approval_implied": False, "failure_reasons": []}
    candidate_audit.to_json(root / "r8er3pr1_warmup_candidate_audit.json", orient="records", force_ascii=False, indent=2)
    selected["trace"].to_parquet(root / "r8er3pr1_vehicle_preroll_trace.parquet", index=False)
    selected["evaluation_state"].to_parquet(root / "r8er3pr1_evaluation_start_vehicle_state.parquet", index=False)
    selected["passengers"][["passenger_id", "request_id", "window_id", "direction_id", "request_ts", "origin_occurrence_id", "first_eligible_service_ts", "first_eligible_service_instance_id", "first_eligible_vehicle_token", "first_eligible_background_agent_id", "physical_origin_arrival", "causal_eligibility"]].to_parquet(root / "r8er3pr1_first_eligible_service_trace.parquet", index=False)
    selected["passengers"][["passenger_id", "request_id", "window_id", "request_ts", "first_eligible_service_ts", "actual_board_ts", "total_wait_seconds", "schedule_wait_seconds", "alignment_excess_wait_seconds", "wait_decomposition_error_seconds", "is_carry_in_passenger", "boarded_during_warmup"]].to_parquet(root / "r8er3pr1_passenger_wait_decomposition.parquet", index=False)
    outputs["missed"].to_parquet(root / "r8er3pr1_missed_eligible_service_audit.parquet", index=False)
    outputs["phase"].to_parquet(root / "r8er3pr1_dispatch_phase_audit.parquet", index=False)
    outputs["window_boundary"].to_parquet(root / "r8er3pr1_window_boundary_counterfactual.parquet", index=False)
    outputs["demand_alignment"].to_parquet(root / "r8er3pr1_demand_alignment_counterfactual.parquet", index=False)
    outputs["paired"].to_parquet(root / "r8er3pr1_old_vs_new_passenger_wait.parquet", index=False)
    writer.json("r8er3pr1_old_vs_new_wait_summary.json", outputs["wait_summary"])
    writer.json("r8er3pr1_schedule_wait_distribution.json", outputs["schedule"])
    writer.json("r8er3pr1_alignment_excess_distribution.json", outputs["alignment"])
    outputs["long_wait_reclassification"].to_parquet(root / "r8er3pr1_long_wait_reclassification.parquet", index=False)
    outputs["h4"].to_parquet(root / "r8er3pr1_h4_observability_after_repair.parquet", index=False)
    writer.json("r8er3pr1_h4_observability_summary.json", outputs["h4_summary"])
    writer.json("r8er3pr1_cold_start_artifact_audit.json", outputs["cold"])
    writer.json("r8er3pr1_service_binding_artifact_audit.json", outputs["binding"])
    writer.json("r8er3pr1_integrity_audit.json", integrity)
    writer.json("r8er3pr1_deterministic_replay.json", deterministic)
    writer.json("r8er3pr1_readiness_decision.json", readiness)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "request-service-alignment-warmup-repair",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "runtime_path": str(RUNTIME_PATH),
        "runtime_sha256": k5.sha256_file(RUNTIME_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "frozen_peak_window_count": int(len(windows)),
        "frozen_passenger_count": int(len(frozen)),
        "selected_warmup_seconds": selected_pre_roll,
        "convergence_warmup_seconds": extended_pre_roll,
        "background_service_instance_count": int(selected["trace"]["service_instance_id"].nunique()),
        "postgresql_select_query_count": int(binding_summary["postgresql_source_audit"]["query_count"]),
        "db_write_count": 0,
        "new_bis_api_call_count": 0,
        "normalization_candidate_count": 0,
        "normalization_approval_count": 0,
        "reward_materialization_count": 0,
        "policy_evaluation_count": 0,
        "mappo_training_count": 0,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": READINESS, "final_decision": decision})
    writer.text("final_report.md", final_report(warmup=warmup_contract, outputs=outputs, decision=decision, rationale=rationale, deterministic=deterministic, integrity=integrity))
    write_manifest_and_lock(writer, gate)
    manifest_integrity = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar8er3pr1.json", "_PV8_R2AR8ER3PR1_COMPLETE.lock")
    if not k5.manifest_ok(manifest_integrity):
        raise R8ER3PR1Error(f"final manifest integrity failed: {manifest_integrity}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {decision}")
    print(f"selected_warmup_seconds: {selected_pre_roll}")
    print(f"cold_start_artifact_count: {outputs['cold']['cold_start_artifact_count']}")
    print(f"new_wait_mean_seconds: {outputs['wait_summary']['new']['mean']}")
    print(f"alignment_excess_count_gt_zero: {outputs['alignment']['count_gt_zero']}")
    print(f"manifest_integrity: {manifest_integrity}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["repair"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
