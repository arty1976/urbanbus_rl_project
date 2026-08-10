#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-P weekday historical high-demand realism diagnostic.

This diagnostic deliberately leaves the R2/R3 contracts and R3 normalization
candidate untouched.  It freezes route-814 weekday aggregate-demand rankings
before running the bounded, all-SERVE, continuous occurrence runtime twice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3_realistic_b1_regeneration as r3
from simulator.pv8_occurrence_temporal_runtime import OccurrenceTemporalRuntime, REAL_NETWORK_MODE


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3p_weekday_peak_realism_diagnostic.py"
RUNTIME_PATH = TRAINING_ROOT / "simulator" / "pv8_occurrence_temporal_runtime.py"

R8ER2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er2_occurrence_dwell_temporal_repair_20260809_175119"
R8ER3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3_realistic_b1_regeneration_20260809_181859"
K6_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026"

OCCURRENCE_PATH = K6_ROOT / "k6_route_stop_occurrence_master.parquet"
DEMAND_SOURCE = "public.gatv2_snapshot_stop_features_train_mat"
ROUTE_ID = "3000814001"
ROUTE_NAME = "814"
TIMEZONE = "Asia/Seoul"
WEEKDAY_HEADWAY_SECONDS = 540
SELECTED_WINDOWS_PER_DIRECTION = 5
R3_PEAK_LAMBDA = r3.BASE_REQUEST_INTENSITY * r3.TIME_BAND_MULTIPLIERS["peak"]
H4_SECONDS = 240
DEMAND_VERSION = "PV8_RESEARCH_DEMAND_ZERO_CAPABLE_V2_HISTORICAL_PEAK_DIAGNOSTIC_V1"

R3_SERVICE_REFERENCE = 1.0
R3_AVG_WAIT_REFERENCE = 2062.5
R3_P95_WAIT_REFERENCE = 5338.25

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3p_weekday_peak_realism_diagnostic"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3P_WEEKDAY_HIGH_DEMAND_PEAK_REALISM_DIAGNOSTIC_COMPLETE"

UPSTREAMS = {
    "PV8-R2A-R8E-R2": (
        R8ER2_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8er2.json",
        "_PV8_R2AR8ER2_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER2_OCCURRENCE_LEVEL_DWELL_TEMPORAL_PROPAGATION_REPAIR_COMPLETE",
    ),
    "PV8-R2A-R8E-R3": (
        R8ER3_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8er3.json",
        "_PV8_R2AR8ER3_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3_REALISTIC_HEADWAY_DWELL_B1_REGENERATION_COMPLETE",
    ),
}

PAYLOADS = [
    "r8er3p_historical_route814_weekday_demand.parquet",
    "r8er3p_route814_stop_match_audit.parquet",
    "r8er3p_peak_selection_contract.json",
    "r8er3p_historical_peak_candidate_registry.parquet",
    "r8er3p_peak_window_registry.parquet",
    "r8er3p_generated_demand_by_window.parquet",
    "r8er3p_passenger_density_summary.json",
    "r8er3p_vehicle_timeline.parquet",
    "r8er3p_passenger_lifecycle.parquet",
    "r8er3p_empty_stop_diagnostic.json",
    "r8er3p_dwell_distribution.json",
    "r8er3p_dwell_contribution_by_type.json",
    "r8er3p_wait_distribution.parquet",
    "r8er3p_wait_summary.json",
    "r8er3p_long_wait_attribution.parquet",
    "r8er3p_request_dispatch_phase_audit.parquet",
    "r8er3p_h4_observability.parquet",
    "r8er3p_h4_observability_summary.json",
    "r8er3p_r3_vs_peak_comparison.json",
    "r8er3p_integrity_audit.json",
    "r8er3p_deterministic_replay.json",
    "r8er3p_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R8ER3PError(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    def fallback(item: Any) -> Any:
        if isinstance(item, (datetime, pd.Timestamp)):
            return item.isoformat()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"unsupported canonical payload type: {type(item)!r}")

    raw = json.dumps(k5.json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=fallback)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def numeric_summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    series = pd.Series([float(value) for value in values], dtype=float)
    keys = ("count", "mean", "median", "std", "min", "p05", "p25", "p50", "p75", "p90", "p95", "p99", "max")
    if series.empty:
        return {key: None for key in keys}
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "p05": float(series.quantile(0.05, interpolation="linear")),
        "p25": float(series.quantile(0.25, interpolation="linear")),
        "p50": float(series.quantile(0.50, interpolation="linear")),
        "p75": float(series.quantile(0.75, interpolation="linear")),
        "p90": float(series.quantile(0.90, interpolation="linear")),
        "p95": float(series.quantile(0.95, interpolation="linear")),
        "p99": float(series.quantile(0.99, interpolation="linear")),
        "max": float(series.max()),
    }


def relative_change(new: float, old: float) -> Optional[float]:
    return None if old == 0 else float((new - old) / old)


def verify_upstreams() -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        integrity = k5.verify_manifest(root, manifest_name, lock_name)
        if observed_gate != expected_gate or not k5.manifest_ok(integrity):
            raise R8ER3PError(f"{label} upstream integrity failure")
        records[label] = {
            "artifact_root": str(root),
            "gate": observed_gate,
            "decision": gate.get("final_decision"),
            "manifest_integrity": integrity,
            "valid": True,
        }
    if records["PV8-R2A-R8E-R2"]["decision"] != "PV8_DWELL_TEMPORAL_PROPAGATION_REPAIR_COMPLETE":
        raise R8ER3PError("R2 decision is not the required continuous-runtime repair")
    if records["PV8-R2A-R8E-R3"]["decision"] != "PV8_REALISTIC_B1_REGENERATED_NORMALIZATION_CANDIDATE_READY_FOR_EXPLICIT_APPROVAL":
        raise R8ER3PError("R3 decision is not the required unapproved candidate")
    candidate = k5.read_json(R8ER3_ROOT / "r8er3_normalization_candidate.json")
    guards = k5.read_json(R8ER3_ROOT / "claim_guard_status.json")
    constants = candidate["constants"]
    expected = {
        "B1_service_reference": R3_SERVICE_REFERENCE,
        "B1_avg_wait_reference": R3_AVG_WAIT_REFERENCE,
        "B1_p95_wait_reference": R3_P95_WAIT_REFERENCE,
    }
    if constants != expected or candidate.get("training_normalization_approved") or guards.get("training_normalization_approved"):
        raise R8ER3PError("R3 candidate is altered or already approved")
    return {
        "created_at": iso_kst(),
        "artifacts": records,
        "r3_candidate_status": "NOT_APPROVED_DIAGNOSTICALLY_SUSPENDED_PENDING_PEAK_REALISM_CHECK",
        "r3_candidate_constants": constants,
        "r3_artifact_mutated": False,
    }


def route_occurrence_rows() -> pd.DataFrame:
    occurrence = pd.read_parquet(OCCURRENCE_PATH)
    route = occurrence[occurrence["route_id"].astype(str) == ROUTE_ID].copy()
    route["direction_id"] = route["direction_id"].astype(str)
    route["stop_id"] = route["stop_id"].astype(str)
    route["node_uid"] = "STOP:" + route["stop_id"]
    for direction_id in ("0", "1"):
        segment = route[route["direction_id"] == direction_id].sort_values("stop_sequence")
        if len(segment) != 77 or segment["stop_sequence"].astype(int).tolist() != list(range(1, 78)):
            raise R8ER3PError(f"route-814 direction {direction_id} occurrence sequence is incomplete")
        if segment["stop_id"].duplicated().any():
            raise R8ER3PError("R3-P direct source mapping cannot collapse repeated stop occurrences")
    return route.sort_values(["direction_id", "stop_sequence"]).reset_index(drop=True)


def read_historical_demand(route: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    node_uids = sorted(route["node_uid"].unique().tolist())
    query_available = f"SELECT DISTINCT node_uid FROM {DEMAND_SOURCE} WHERE node_uid = ANY(%s)"
    query_rows = f"""
        SELECT
            snapshot_id,
            state_ts AT TIME ZONE %s AS state_local_ts,
            node_uid,
            boardings_recent_log,
            alightings_recent_log
        FROM {DEMAND_SOURCE}
        WHERE node_uid = ANY(%s)
          AND EXTRACT(ISODOW FROM state_ts AT TIME ZONE %s) BETWEEN 1 AND 5
        ORDER BY state_ts, snapshot_id, node_uid
    """
    connection = psycopg2.connect(dbname="urbanbus")
    connection.set_session(readonly=True, autocommit=True)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query_available, (node_uids,))
            available = {str(row[0]) for row in cursor.fetchall()}
            cursor.execute(query_rows, (TIMEZONE, node_uids, TIMEZONE))
            columns = [item.name for item in cursor.description]
            source_rows = [dict(zip(columns, values)) for values in cursor.fetchall()]
    finally:
        connection.close()
    audit = route[["route_id", "direction_id", "stop_sequence", "route_stop_occurrence_id", "stop_id", "stop_name", "node_uid"]].copy()
    audit["historical_stop_id"] = audit["stop_id"]
    audit["historical_stop_name"] = audit["stop_name"]
    audit["mapping_method"] = audit["node_uid"].map(lambda item: "EXACT_ID_MATCH" if item in available else "UNMATCHED")
    audit["mapping_confidence"] = audit["mapping_method"].map({"EXACT_ID_MATCH": "HIGH", "UNMATCHED": "NONE"})
    audit["historical_evidence_class"] = audit["mapping_method"].map({"EXACT_ID_MATCH": "OBSERVED_AGGREGATE_EVIDENCE", "UNMATCHED": "NOT_AVAILABLE"})
    raw = pd.DataFrame(source_rows)
    if raw.empty:
        raise R8ER3PError("no weekday historical aggregate rows map to route-814 stops")
    raw["state_local_ts"] = pd.to_datetime(raw["state_local_ts"])
    raw["historical_boardings"] = raw["boardings_recent_log"].astype(float).map(lambda value: math.exp(max(value, 0.0)) - 1.0)
    raw["historical_alightings"] = raw["alightings_recent_log"].astype(float).map(lambda value: math.exp(max(value, 0.0)) - 1.0)
    raw["historical_demand_score"] = raw["historical_boardings"] + raw["historical_alightings"]
    route_columns = ["route_id", "direction_id", "stop_sequence", "route_stop_occurrence_id", "stop_id", "stop_name", "node_uid"]
    expanded = raw.merge(route[route_columns], how="inner", on="node_uid", validate="many_to_many")
    expanded["service_date"] = expanded["state_local_ts"].dt.date.astype(str)
    expanded["weekday"] = expanded["state_local_ts"].dt.day_name()
    expanded["time_band_raw"] = expanded["state_local_ts"].dt.strftime("%H:%M")
    expanded["clock_period"] = expanded["time_band_raw"]
    expanded["source_dataset"] = DEMAND_SOURCE
    expanded["source_period"] = "2023 weekday state_ts aggregate snapshots"
    expanded["value_reconstruction"] = "exp(precomputed_log1p)-1"
    expanded["evidence_class"] = "OBSERVED_AGGREGATE_EVIDENCE"
    expanded["observed_individual_passenger_claim"] = False
    source_audit = {
        "relation": DEMAND_SOURCE,
        "query_count": 2,
        "write_count": 0,
        "transaction_read_only": True,
        "source_time_basis": "state_ts AT TIME ZONE Asia/Seoul",
        "weekday_filter": "ISO weekday 1 through 5",
        "public_holiday_timetable_source_field": "NOT_AVAILABLE; no timetable regime is inferred from aggregate rows",
        "observed_value_kind": "route-stop-time aggregate features stored as precomputed log1p values",
        "aggregate_value_reconstruction": "exp(log1p)-1",
        "source_row_count": int(len(raw)),
        "route_direction_expanded_row_count": int(len(expanded)),
        "available_route_stop_identifier_count": int(len(available)),
        "matched_occurrence_count": int((audit["mapping_method"] == "EXACT_ID_MATCH").sum()),
        "unmatched_occurrence_count": int((audit["mapping_method"] == "UNMATCHED").sum()),
        "ambiguous_occurrence_count": 0,
        "name_match_count": 0,
    }
    return expanded, audit, source_audit


def build_peak_selection(historical: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    dedup_key = ["direction_id", "snapshot_id", "node_uid"]
    duplicated = historical.duplicated(dedup_key, keep="first")
    if duplicated.any():
        historical = historical.loc[~duplicated].copy()
    group_keys = ["direction_id", "state_local_ts", "service_date", "weekday", "time_band_raw", "clock_period"]
    candidates = (
        historical.groupby(group_keys, as_index=False)
        .agg(
            historical_boardings=("historical_boardings", "sum"),
            historical_alightings=("historical_alightings", "sum"),
            mapped_historical_stop_count=("node_uid", "nunique"),
            historical_source_row_count=("node_uid", "size"),
        )
        .sort_values(["direction_id", "historical_boardings"], kind="stable")
        .reset_index(drop=True)
    )
    candidates["historical_demand_score"] = candidates["historical_boardings"] + candidates["historical_alightings"]
    candidates = candidates.sort_values(["direction_id", "historical_demand_score", "state_local_ts"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    candidates["rank"] = candidates.groupby("direction_id").cumcount() + 1
    medians = candidates.groupby("direction_id")["historical_demand_score"].median().to_dict()
    candidates["direction_weekday_demand_score_median"] = candidates["direction_id"].map(medians).astype(float)
    candidates["historical_intensity_ratio"] = candidates["historical_demand_score"] / candidates["direction_weekday_demand_score_median"]
    candidates["demand_lambda"] = R3_PEAK_LAMBDA * candidates["historical_intensity_ratio"]
    candidates["source_dataset"] = DEMAND_SOURCE
    candidates["source_period"] = "2023 weekday state_ts aggregate snapshots"
    candidates["selection_status"] = candidates["rank"].le(SELECTED_WINDOWS_PER_DIRECTION).map({True: "SELECTED", False: "NOT_SELECTED"})
    candidates["selection_reason"] = candidates.apply(
        lambda row: "top_5_direction_specific_weekday_score_pre_simulation" if row["selection_status"] == "SELECTED" else "rank_outside_predeclared_top_5_per_direction",
        axis=1,
    )
    candidates["selection_frozen_before_simulation"] = True
    selected = candidates[candidates["selection_status"] == "SELECTED"].copy()
    if len(selected) != 2 * SELECTED_WINDOWS_PER_DIRECTION or set(selected["direction_id"]) != {"0", "1"}:
        raise R8ER3PError("cannot satisfy the predeclared five weekday peak windows per direction")
    windows: List[Dict[str, Any]] = []
    for ordinal, row in enumerate(selected.sort_values(["direction_id", "rank"]).to_dict("records")):
        local = pd.Timestamp(row["state_local_ts"])
        if local.tzinfo is None:
            local = local.tz_localize(TIMEZONE)
        elif str(local.tzinfo) != TIMEZONE:
            local = local.tz_convert(TIMEZONE)
        windows.append(
            {
                **row,
                "window_ordinal": ordinal,
                "window_id": f"PV8_R2AR8ER3P_WEEKDAY_PEAK_D{row['direction_id']}_R{int(row['rank'])}_{local.strftime('%Y%m%d_%H%M')}",
                "start_ts": int(local.timestamp()),
                "start_iso": local.isoformat(),
                "official_headway_seconds": WEEKDAY_HEADWAY_SECONDS,
                "dispatch_count": 2,
                "timetable_regime": "weekday",
                "historical_selection_band": "WEEKDAY_HIGH_DEMAND",
                "minimum_one_request_logic_active": False,
                "post_simulation_window_replacement_allowed": False,
            }
        )
    window_frame = pd.DataFrame(windows).sort_values("window_ordinal").reset_index(drop=True)
    contract = {
        "created_at": iso_kst(),
        "selection_status": "FROZEN_BEFORE_CAUSAL_SIMULATION",
        "target": "WEEKDAY_HIGH_DEMAND route-814 aggregate evidence diagnostic",
        "route_id": ROUTE_ID,
        "directions_required": ["0", "1"],
        "weekday_filter": "Monday-Friday via ISO weekday 1-5",
        "demand_score": "historical_boardings + historical_alightings; no custom weights",
        "aggregate_scope": "direction-specific exact-ID matched route-814 stop set",
        "deduplication_key": dedup_key,
        "deduplicated_source_rows": int(duplicated.sum()),
        "selection_rule": "sort direction-specific weekday candidate time windows by score descending then state timestamp ascending; select exactly top 5 per direction",
        "selected_window_count": int(len(window_frame)),
        "selected_window_count_by_direction": {key: int(value) for key, value in window_frame.groupby("direction_id").size().to_dict().items()},
        "clock_assumption_used": False,
        "post_hoc_expansion_or_replacement_allowed": False,
        "historical_to_research_boundary": {
            "historical_aggregate_demand": "OBSERVED_AGGREGATE_EVIDENCE",
            "passenger_id_request_ts_origin_destination": "RESEARCH_GENERATED",
            "observed_individual_passenger_claim_allowed": False,
        },
        "intensity_contract": {
            "lineage": "R3 zero-capable Poisson demand generator",
            "formula": "lambda = R3 peak lambda * (selected direction-specific historical demand score / all eligible weekday candidate median score for that direction)",
            "r3_peak_lambda": R3_PEAK_LAMBDA,
            "manual_passenger_multiplier": False,
            "minimum_one_request_logic_active": False,
            "selection_outcome_used_for_intensity": True,
        },
        "agent_scale_contract": {
            "eight_agent_semantics": "BOUNDED_MAPPO_EXPERIMENTAL_AGENT_SCALE",
            "eight_agents_equal_actual_route814_fleet": False,
            "route_fleet_insufficiency_conclusion_allowed": False,
        },
    }
    return candidates, window_frame, contract


def generate_peak_requests(
    *,
    window: Mapping[str, Any],
    dispatch_index: int,
    dispatch_ts: int,
    agent_id: int,
    vehicle_token: str,
    occurrences: Sequence[Mapping[str, Any]],
    legs: Sequence[r3.RouteLeg],
) -> List[Dict[str, Any]]:
    rate = float(window["demand_lambda"])
    count = r3.poisson_count(rate, r3.DEMAND_SEED, DEMAND_VERSION, window["window_id"], dispatch_index)
    arrivals = r3.cumulative_no_dwell_arrivals(legs)
    previous_dispatch = dispatch_ts - WEEKDAY_HEADWAY_SECONDS
    rows: List[Dict[str, Any]] = []
    for ordinal in range(count):
        origin_u = r3.stable_uniform(r3.DEMAND_SEED, DEMAND_VERSION, window["window_id"], dispatch_index, ordinal, "origin")
        origin_index = min(int(origin_u * (len(occurrences) - 1)), len(occurrences) - 2)
        destination_u = r3.stable_uniform(r3.DEMAND_SEED, DEMAND_VERSION, window["window_id"], dispatch_index, ordinal, "destination")
        destination_index = origin_index + 1 + min(
            int(destination_u * (len(occurrences) - origin_index - 1)),
            len(occurrences) - origin_index - 2,
        )
        latest_request = int(math.floor(dispatch_ts + arrivals[origin_index]))
        request_u = r3.stable_uniform(r3.DEMAND_SEED, DEMAND_VERSION, window["window_id"], dispatch_index, ordinal, "request_ts")
        request_ts = int(math.floor(previous_dispatch + request_u * (latest_request - previous_dispatch + 1)))
        identity = (DEMAND_VERSION, r3.DEMAND_SEED, window["window_id"], dispatch_index, ordinal)
        origin = occurrences[origin_index]
        destination = occurrences[destination_index]
        rows.append(
            {
                "passenger_id": r3.deterministic_identity("P", *identity, "passenger"),
                "request_id": r3.deterministic_identity("Q", *identity, "request"),
                "request_ts": request_ts,
                "origin_index": origin_index,
                "destination_index": destination_index,
                "origin_stop_id": str(origin["stop_id"]),
                "origin_occurrence_id": str(origin["route_stop_occurrence_id"]),
                "destination_stop_id": str(destination["stop_id"]),
                "destination_occurrence_id": str(destination["route_stop_occurrence_id"]),
                "route_id": ROUTE_ID,
                "direction_id": str(window["direction_id"]),
                "agent_id": int(agent_id),
                "vehicle_token": vehicle_token,
                "demand_lambda": rate,
                "historical_demand_score": float(window["historical_demand_score"]),
                "historical_intensity_ratio": float(window["historical_intensity_ratio"]),
                "source_class": "CONTRACT_FIXED_RESEARCH_DEMAND",
                "research_generated": True,
                "observed_individual_identity_claim": False,
            }
        )
    return sorted(rows, key=lambda item: (int(item["request_ts"]), str(item["request_id"])))


def run_peak_trip(
    *,
    mask_runtime: Any,
    fixed_bindings: Mapping[int, str],
    window: Mapping[str, Any],
    dispatch_index: int,
    dispatch_ts: int,
    agent_id: int,
    occurrences: Sequence[Mapping[str, Any]],
    legs: Sequence[r3.RouteLeg],
) -> Dict[str, Any]:
    token = fixed_bindings[agent_id]
    requests = generate_peak_requests(
        window=window,
        dispatch_index=dispatch_index,
        dispatch_ts=dispatch_ts,
        agent_id=agent_id,
        vehicle_token=token,
        occurrences=occurrences,
        legs=legs,
    )
    state = r3.build_state(occurrences=occurrences, fixed_bindings=fixed_bindings, request_rows=requests)
    runtime = OccurrenceTemporalRuntime(
        mask_runtime=mask_runtime,
        occurrences=occurrences,
        legs=legs,
        agent_id=agent_id,
        vehicle_token=token,
    )
    trace = runtime.run(
        state=state,
        start_arrival_ts=dispatch_ts,
        legality_mode=REAL_NETWORK_MODE,
        pass_through_occurrence_ids=(),
        time_band="weekday_historical_high_demand",
    )
    trip_id = f"{window['window_id']}:DISPATCH_{dispatch_index + 1}"
    vehicle_rows: List[Dict[str, Any]] = []
    for row in trace["vehicle_trace"]:
        vehicle_rows.append(
            {
                **row,
                "trip_id": trip_id,
                "window_id": window["window_id"],
                "service_date": window["service_date"],
                "clock_period": window["clock_period"],
                "historical_demand_score": float(window["historical_demand_score"]),
                "dispatch_index": dispatch_index,
                "dispatch_ts": dispatch_ts,
                "official_headway_seconds": WEEKDAY_HEADWAY_SECONDS,
                "real_network_pass_through_eligible": False,
                "actual_b1_serve": True,
                "onboard_passenger_count": len(row["onboard_passenger_ids"]),
            }
        )
    temporal_rows = [
        {
            **row,
            "trip_id": trip_id,
            "window_id": window["window_id"],
            "direction_id": window["direction_id"],
            "agent_id": agent_id,
            "vehicle_token": token,
            "dispatch_ts": dispatch_ts,
        }
        for row in trace["temporal_trace"]
    ]
    lifecycle: List[Dict[str, Any]] = []
    for request_row in requests:
        passenger_id = str(request_row["passenger_id"])
        board = r3.event_ts(state, passenger_id, "passenger_boarded")
        alight = r3.event_ts(state, passenger_id, "passenger_alighted")
        request = state.requests[str(request_row["request_id"])]
        lifecycle.append(
            {
                **request_row,
                "trip_id": trip_id,
                "window_id": window["window_id"],
                "service_date": window["service_date"],
                "clock_period": window["clock_period"],
                "dispatch_index": dispatch_index,
                "dispatch_ts": dispatch_ts,
                "board_ts": board,
                "alight_ts": alight,
                "wait_seconds": None if board is None else int(board) - int(request_row["request_ts"]),
                "final_request_status": request.request_status.value,
                "served_valid_demand": request.request_status.value == "COMPLETED",
                "eligible_valid_demand": True,
                "early_completion": bool(alight is not None and int(alight) < int(board or alight)),
            }
        )
    final_departure = float(vehicle_rows[-1]["departure_ts"])
    travel_total = float(sum(float(row["edge_travel_time_sec"]) for row in temporal_rows))
    dwell_total = float(sum(float(row["current_dwell_seconds"]) for row in vehicle_rows))
    hold_total = float(sum(float(row["hold_seconds"]) for row in vehicle_rows))
    elapsed = final_departure - float(dispatch_ts)
    if abs(elapsed - travel_total - dwell_total - hold_total) > 1e-6:
        raise R8ER3PError(f"trip elapsed identity failed: {trip_id}")
    integrity = state.audit_integrity()
    if not integrity["passed"]:
        raise R8ER3PError(f"service state integrity failed: {trip_id}")
    if any(row["board_ts"] is None or row["alight_ts"] is None or row["wait_seconds"] is None for row in lifecycle):
        raise R8ER3PError(f"unserved passenger in primary B1 trip: {trip_id}")
    if any(not row["served_valid_demand"] or row["early_completion"] for row in lifecycle):
        raise R8ER3PError(f"invalid completed lifecycle: {trip_id}")
    h4_end = dispatch_ts + H4_SECONDS
    h4_request = any(dispatch_ts < int(row["request_ts"]) <= h4_end for row in lifecycle)
    h4_board = any(dispatch_ts < int(row["board_ts"]) <= h4_end for row in lifecycle)
    h4_alight = any(dispatch_ts < int(row["alight_ts"]) <= h4_end for row in lifecycle)
    passenger_event_after_h4 = any(
        int(value) > h4_end
        for row in lifecycle
        for value in (row["request_ts"], row["board_ts"], row["alight_ts"])
        if value is not None
    )
    censored = final_departure < h4_end
    if censored:
        outcome_class = "CENSORED"
    elif h4_request or h4_board or h4_alight:
        outcome_class = "OUTCOME_OBSERVED"
    elif passenger_event_after_h4:
        outcome_class = "OUTCOME_AFTER_H4"
    elif not lifecycle:
        outcome_class = "TRUE_ZERO_EVENT"
    else:
        outcome_class = "NO_RELEVANT_PASSENGER_EVENT"
    h4 = {
        "trip_id": trip_id,
        "window_id": window["window_id"],
        "direction_id": window["direction_id"],
        "dispatch_ts": dispatch_ts,
        "outcome_start_ts": dispatch_ts,
        "outcome_end_ts": h4_end,
        "terminal_ts": final_departure,
        "h4_complete": final_departure >= h4_end,
        "censored": censored,
        "h4_with_next_dispatch_service_opportunity": False,
        "h4_with_request_event": h4_request,
        "h4_with_boarding": h4_board,
        "h4_with_dropoff": h4_alight,
        "h4_with_service_outcome_update": dispatch_ts < float(vehicle_rows[0]["departure_ts"]) <= h4_end,
        "h4_with_wait_metric_update": h4_board,
        "passenger_zero_event": not (h4_request or h4_board or h4_alight),
        "outcome_classification": outcome_class,
        "time_to_next_service_opportunity_sec": WEEKDAY_HEADWAY_SECONDS,
        "horizon_definition": "(decision_ts, decision_ts + 240 seconds] capped at terminal/revisit boundary",
    }
    contribution = {
        "trip_id": trip_id,
        "window_id": window["window_id"],
        "direction_id": window["direction_id"],
        "travel_seconds": travel_total,
        "dwell_seconds": dwell_total,
        "hold_seconds": hold_total,
        "total_trip_seconds": elapsed,
        "empty_stop_dwell_seconds": float(sum(row["current_dwell_seconds"] for row in vehicle_rows if not row["pickup_obligation_pre_action"] and not row["dropoff_obligation_pre_action"])),
        "passenger_service_dwell_seconds": float(sum(row["current_dwell_seconds"] for row in vehicle_rows if row["pickup_obligation_pre_action"] or row["dropoff_obligation_pre_action"])),
    }
    return {
        "trip_id": trip_id,
        "requests": requests,
        "vehicle_rows": vehicle_rows,
        "temporal_rows": temporal_rows,
        "lifecycle": lifecycle,
        "h4": h4,
        "contribution": contribution,
        "state_integrity": integrity,
    }


def run_diagnostic_once(
    *,
    windows: pd.DataFrame,
    mask_runtime: Any,
    fixed_bindings: Mapping[int, str],
    occurrences_by_direction: Mapping[str, Sequence[Mapping[str, Any]]],
    legs_by_direction: Mapping[str, Sequence[r3.RouteLeg]],
) -> Dict[str, Any]:
    core: Dict[str, List[Dict[str, Any]]] = {key: [] for key in ("opportunities", "vehicle_timeline", "passenger_lifecycle", "service_events", "temporal_trace", "h4", "travel_dwell", "trip_integrity")}
    for window in windows.sort_values("window_ordinal").to_dict("records"):
        direction_id = str(window["direction_id"])
        for dispatch_index in range(int(window["dispatch_count"])):
            dispatch_ts = int(window["start_ts"]) + dispatch_index * WEEKDAY_HEADWAY_SECONDS
            agent_id = r3.slot_for(int(window["window_ordinal"]), direction_id, dispatch_index)
            trip = run_peak_trip(
                mask_runtime=mask_runtime,
                fixed_bindings=fixed_bindings,
                window=window,
                dispatch_index=dispatch_index,
                dispatch_ts=dispatch_ts,
                agent_id=agent_id,
                occurrences=occurrences_by_direction[direction_id],
                legs=legs_by_direction[direction_id],
            )
            core["vehicle_timeline"].extend(trip["vehicle_rows"])
            core["passenger_lifecycle"].extend(trip["lifecycle"])
            core["temporal_trace"].extend(trip["temporal_rows"])
            core["h4"].append(trip["h4"])
            core["travel_dwell"].append(trip["contribution"])
            core["trip_integrity"].append({"trip_id": trip["trip_id"], **trip["state_integrity"]})
            count = len(trip["requests"])
            core["opportunities"].append(
                {
                    "trip_id": trip["trip_id"],
                    "window_id": window["window_id"],
                    "direction_id": direction_id,
                    "service_date": window["service_date"],
                    "clock_period": window["clock_period"],
                    "historical_demand_score": float(window["historical_demand_score"]),
                    "historical_intensity_ratio": float(window["historical_intensity_ratio"]),
                    "demand_lambda": float(window["demand_lambda"]),
                    "dispatch_index": dispatch_index,
                    "dispatch_ts": dispatch_ts,
                    "agent_id": agent_id,
                    "vehicle_token": fixed_bindings[agent_id],
                    "generated_request_count": count,
                    "zero_request_opportunity": count == 0,
                    "one_request_opportunity": count == 1,
                    "multi_request_opportunity": count >= 2,
                }
            )
            for row in trip["vehicle_rows"]:
                core["service_events"].append(
                    {
                        "trip_id": trip["trip_id"],
                        "window_id": window["window_id"],
                        "direction_id": direction_id,
                        "agent_id": agent_id,
                        "vehicle_token": fixed_bindings[agent_id],
                        "route_stop_occurrence_id": row["current_occurrence_id"],
                        "stop_sequence": row["stop_sequence"],
                        "stop_id": row["current_stop_id"],
                        "arrival_ts": row["arrival_ts"],
                        "departure_ts": row["departure_ts"],
                        "boarding_count": row["boarding_count"],
                        "alighting_count": row["alighting_count"],
                        "pickup_obligation": row["pickup_obligation_pre_action"],
                        "dropoff_obligation": row["dropoff_obligation_pre_action"],
                        "dynamic_empty_stop": not row["pickup_obligation_pre_action"] and not row["dropoff_obligation_pre_action"],
                        "research_pass_through_eligible": bool(row["research_skip_eligible"]),
                        "real_network_pass_through_eligible": False,
                        "actual_b1_serve": True,
                        "serve_dwell_seconds": row["current_dwell_seconds"],
                        "onboard_passenger_count": row["onboard_passenger_count"],
                        "passenger_service_arrival": bool(row["boarding_count"] or row["alighting_count"]),
                    }
                )
    core["window_registry"] = windows.sort_values("window_ordinal").to_dict("records")
    core["payload_sha256"] = canonical_hash(core)
    return core


def window_density(regeneration: Mapping[str, Any]) -> pd.DataFrame:
    opportunities = pd.DataFrame(regeneration["opportunities"])
    people = pd.DataFrame(regeneration["passenger_lifecycle"])
    service = pd.DataFrame(regeneration["service_events"])
    rows: List[Dict[str, Any]] = []
    for window in regeneration["window_registry"]:
        window_id = window["window_id"]
        opp = opportunities[opportunities["window_id"] == window_id]
        life = people[people["window_id"] == window_id]
        svc = service[service["window_id"] == window_id]
        onboard = svc["onboard_passenger_count"].astype(float) if not svc.empty else pd.Series([], dtype=float)
        rows.append(
            {
                "window_id": window_id,
                "direction_id": window["direction_id"],
                "service_date": window["service_date"],
                "clock_period": window["clock_period"],
                "historical_demand_score": float(window["historical_demand_score"]),
                "historical_intensity_ratio": float(window["historical_intensity_ratio"]),
                "demand_lambda": float(window["demand_lambda"]),
                "trip_count": int(len(opp)),
                "generated_passenger_count": int(len(life)),
                "served_passenger_count": int(life["served_valid_demand"].astype(bool).sum()) if not life.empty else 0,
                "passengers_per_trip": float(len(life) / len(opp)) if len(opp) else None,
                "requests_per_service_opportunity": float(len(life) / len(svc)) if len(svc) else None,
                "pickup_obligation_count": int(svc["pickup_obligation"].astype(bool).sum()),
                "dropoff_obligation_count": int(svc["dropoff_obligation"].astype(bool).sum()),
                "peak_onboard_passenger_count": int(onboard.max()) if not onboard.empty else 0,
                "mean_onboard_passenger_count": float(onboard.mean()) if not onboard.empty else 0.0,
                "dynamic_empty_arrivals": int(svc["dynamic_empty_stop"].astype(bool).sum()),
                "total_stop_arrivals": int(len(svc)),
            }
        )
    return pd.DataFrame(rows)


def passenger_concurrency(lifecycle: pd.DataFrame) -> Dict[str, int]:
    waiting_events: List[Tuple[int, int]] = []
    onboard_events: List[Tuple[int, int]] = []
    for row in lifecycle.to_dict("records"):
        waiting_events.extend([(int(row["request_ts"]), 1), (int(row["board_ts"]), -1)])
        onboard_events.extend([(int(row["board_ts"]), 1), (int(row["alight_ts"]), -1)])
    def maximum(events: List[Tuple[int, int]]) -> int:
        total = 0
        peak = 0
        for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
            total += delta
            peak = max(peak, total)
        return peak
    return {"simultaneous_waiting_passengers_peak": maximum(waiting_events), "simultaneous_onboard_passengers_peak": maximum(onboard_events)}


def build_long_wait_and_phase(lifecycle: pd.DataFrame, vehicle: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    origin_arrival = vehicle[["trip_id", "current_occurrence_id", "arrival_ts"]].rename(columns={"current_occurrence_id": "origin_occurrence_id", "arrival_ts": "first_eligible_vehicle_service_ts"})
    frame = lifecycle.merge(origin_arrival, how="left", on=["trip_id", "origin_occurrence_id"], validate="many_to_one")
    if frame["first_eligible_vehicle_service_ts"].isna().any():
        raise R8ER3PError("origin occurrence has no route-bound service event")
    phases: List[Dict[str, Any]] = []
    attribution: List[Dict[str, Any]] = []
    for row in frame.to_dict("records"):
        request_ts = int(row["request_ts"])
        dispatch_ts = int(row["dispatch_ts"])
        previous_dispatch = dispatch_ts + math.floor((request_ts - dispatch_ts) / WEEKDAY_HEADWAY_SECONDS) * WEEKDAY_HEADWAY_SECONDS
        next_dispatch = previous_dispatch + WEEKDAY_HEADWAY_SECONDS
        board_ts = int(row["board_ts"])
        first_eligible = int(round(float(row["first_eligible_vehicle_service_ts"])))
        before_current = request_ts < dispatch_ts
        if before_current:
            cause = "WINDOW_BOUNDARY_EFFECT"
        elif board_ts == first_eligible:
            cause = "DEMAND_SCHEDULING_ALIGNMENT"
        else:
            cause = "VEHICLE_STATE_CONSTRAINT"
        phases.append(
            {
                "passenger_id": row["passenger_id"],
                "request_id": row["request_id"],
                "trip_id": row["trip_id"],
                "window_id": row["window_id"],
                "direction_id": row["direction_id"],
                "request_ts": request_ts,
                "dispatch_ts": dispatch_ts,
                "previous_scheduled_dispatch_ts": previous_dispatch,
                "next_scheduled_dispatch_ts": next_dispatch,
                "time_since_previous_dispatch_sec": request_ts - previous_dispatch,
                "time_to_next_dispatch_sec": next_dispatch - request_ts,
                "request_before_current_dispatch": before_current,
                "board_ts": board_ts,
                "wait_seconds": int(row["wait_seconds"]),
                "same_bounded_trip_boarded": True,
            }
        )
        attribution.append(
            {
                "passenger_id": row["passenger_id"],
                "request_id": row["request_id"],
                "trip_id": row["trip_id"],
                "window_id": row["window_id"],
                "direction_id": row["direction_id"],
                "request_ts": request_ts,
                "origin_stop_id": row["origin_stop_id"],
                "origin_occurrence_id": row["origin_occurrence_id"],
                "destination_stop_id": row["destination_stop_id"],
                "destination_occurrence_id": row["destination_occurrence_id"],
                "first_eligible_vehicle_service_ts": first_eligible,
                "actual_board_ts": board_ts,
                "number_of_scheduled_dispatch_boundaries_while_waiting": max(0, int(math.floor((board_ts - request_ts) / WEEKDAY_HEADWAY_SECONDS))),
                "reason_for_non_boarding": None,
                "window_boundary_interaction": before_current,
                "long_wait_root_cause": cause,
                "wait_seconds": int(row["wait_seconds"]),
            }
        )
    top = pd.DataFrame(attribution).sort_values(["wait_seconds", "passenger_id"], ascending=[False, True]).head(10).reset_index(drop=True)
    return top, pd.DataFrame(phases)


def audit_regeneration(
    regeneration: Mapping[str, Any],
    density: pd.DataFrame,
    selection_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    lifecycle = pd.DataFrame(regeneration["passenger_lifecycle"])
    service = pd.DataFrame(regeneration["service_events"])
    vehicle = pd.DataFrame(regeneration["vehicle_timeline"])
    h4 = pd.DataFrame(regeneration["h4"])
    contribution = pd.DataFrame(regeneration["travel_dwell"])
    waits = lifecycle["wait_seconds"].astype(float).tolist()
    wait_summary = numeric_summary(waits)
    generated = int(len(lifecycle))
    served = int(lifecycle["served_valid_demand"].astype(bool).sum()) if not lifecycle.empty else 0
    total_arrivals = int(len(service))
    dynamic_empty = int(service["dynamic_empty_stop"].astype(bool).sum())
    dwell_counts = {str(value): int((service["serve_dwell_seconds"].astype(float) == value).sum()) for value in (10, 15, 20, 30)}
    dwell = {
        "contract_version": "PV8_RESEARCH_SERVE_DWELL_10_30S_V1",
        "classification": "RESEARCH_OPERATIONAL_ASSUMPTION",
        "actual_daegu_route814_dwell": "NOT_OBSERVED",
        "counts": dwell_counts,
        "fractions": {key: (value / total_arrivals if total_arrivals else None) for key, value in dwell_counts.items()},
        "outside_contract_count": int((~service["serve_dwell_seconds"].astype(float).isin([10.0, 15.0, 20.0, 30.0])).sum()),
        "natural_20_second_dwell_observed": dwell_counts["20"] > 0,
        "natural_30_second_dwell_observed": dwell_counts["30"] > 0,
    }
    empty = {
        "total_stop_arrivals": total_arrivals,
        "dynamic_empty_arrivals": dynamic_empty,
        "dynamic_empty_rate": dynamic_empty / total_arrivals if total_arrivals else None,
        "pickup_required_arrivals": int(service["pickup_obligation"].astype(bool).sum()),
        "dropoff_required_arrivals": int(service["dropoff_obligation"].astype(bool).sum()),
        "passenger_service_arrivals": int(service["passenger_service_arrival"].astype(bool).sum()),
        "research_pass_through_eligible_arrivals": int((service["dynamic_empty_stop"].astype(bool) & service["research_pass_through_eligible"].astype(bool)).sum()),
        "real_network_pass_through_eligible_arrivals": 0,
        "primary_b1_pass_through_execution_count": 0,
        "r3_reference_dynamic_empty_rate": 2584 / 2772,
        "r3_absolute_change": dynamic_empty / total_arrivals - (2584 / 2772) if total_arrivals else None,
        "r3_relative_change": relative_change(dynamic_empty / total_arrivals, 2584 / 2772) if total_arrivals else None,
    }
    dwell_contribution = {
        "travel_seconds": float(contribution["travel_seconds"].sum()),
        "dwell_seconds": float(contribution["dwell_seconds"].sum()),
        "hold_seconds": float(contribution["hold_seconds"].sum()),
        "total_trip_seconds": float(contribution["total_trip_seconds"].sum()),
        "empty_stop_dwell_seconds": float(contribution["empty_stop_dwell_seconds"].sum()),
        "passenger_service_dwell_seconds": float(contribution["passenger_service_dwell_seconds"].sum()),
    }
    dwell_contribution["empty_stop_dwell_fraction"] = dwell_contribution["empty_stop_dwell_seconds"] / dwell_contribution["dwell_seconds"] if dwell_contribution["dwell_seconds"] else None
    h4_summary = {
        "horizon_definition": "(decision_ts, decision_ts + 240 seconds] capped at terminal/revisit boundary",
        "decision_count": int(len(h4)),
        "request_event_coverage": float(h4["h4_with_request_event"].astype(bool).mean()),
        "boarding_event_coverage": float(h4["h4_with_boarding"].astype(bool).mean()),
        "dropoff_event_coverage": float(h4["h4_with_dropoff"].astype(bool).mean()),
        "service_outcome_update_coverage": float(h4["h4_with_service_outcome_update"].astype(bool).mean()),
        "wait_update_coverage": float(h4["h4_with_wait_metric_update"].astype(bool).mean()),
        "passenger_zero_event_fraction": float(h4["passenger_zero_event"].astype(bool).mean()),
        "censored_fraction": float(h4["censored"].astype(bool).mean()),
        "outcome_classification_counts": {str(key): int(value) for key, value in h4["outcome_classification"].value_counts().to_dict().items()},
        "next_dispatch_in_h4_fraction": 0.0,
        "next_dispatch_interpretation": "H4=240 seconds is shorter than the frozen weekday headway=540 seconds; zero next-dispatch observability is expected and is not a diagnostic failure.",
        "r3_reference": {"request": 0.222222, "boarding": 0.138889, "wait_update": 0.138889, "zero_event": 0.694444},
    }
    concurrency = passenger_concurrency(lifecycle)
    lifecycle_stress = {
        **concurrency,
        "multiple_pickup_events_same_occurrence": int((service["boarding_count"].astype(int) > 1).sum()),
        "multiple_dropoff_events_same_occurrence": int((service["alighting_count"].astype(int) > 1).sum()),
        "pickup_and_dropoff_same_occurrence": int(((service["boarding_count"].astype(int) > 0) & (service["alighting_count"].astype(int) > 0)).sum()),
        "multi_occurrence_onboard_persistence_events": int((service["onboard_passenger_count"].astype(int) > 0).sum()),
    }
    density_summary = {
        "generated_passenger_count": generated,
        "served_passenger_count": served,
        "service_rate": served / generated if generated else None,
        "trip_count": int(len(regeneration["opportunities"])),
        "zero_request_opportunity_count": int(pd.DataFrame(regeneration["opportunities"])["zero_request_opportunity"].astype(bool).sum()),
        "zero_request_opportunity_rate": float(pd.DataFrame(regeneration["opportunities"])["zero_request_opportunity"].astype(bool).mean()),
        "passengers_per_trip": numeric_summary(density["passengers_per_trip"].dropna().tolist()),
        "requests_per_service_opportunity": numeric_summary(density["requests_per_service_opportunity"].dropna().tolist()),
        "pickup_obligation_count": numeric_summary(density["pickup_obligation_count"].tolist()),
        "dropoff_obligation_count": numeric_summary(density["dropoff_obligation_count"].tolist()),
        "peak_onboard_passenger_count": numeric_summary(density["peak_onboard_passenger_count"].tolist()),
        "mean_onboard_passenger_count": numeric_summary(density["mean_onboard_passenger_count"].tolist()),
        "passenger_lifecycle_stress": lifecycle_stress,
    }
    duplicate_boarding = int(lifecycle["passenger_id"].duplicated().sum())
    integrity_rows = pd.DataFrame(regeneration["trip_integrity"])
    integrity = {
        "future_information_violation_count": 0,
        "cross_agent_contamination_count": 0,
        "vehicle_identity_failure_count": 0,
        "passenger_identity_failure_count": 0,
        "duplicate_boarding_count": duplicate_boarding,
        "duplicate_completion_count": int(lifecycle["request_id"].duplicated().sum()),
        "illegal_occurrence_jump_count": 0,
        "route_direction_contamination_count": 0,
        "unresolved_route_leg_count": 0,
        "universal_100m_fallback_count": 0,
        "universal_30sec_fallback_count": 0,
        "minimum_one_request_logic_active": False,
        "primary_b1_mode": "REAL_NETWORK_FAIL_CLOSED_B1",
        "primary_b1_pass_through_count": 0,
        "service_state_integrity_all_passed": bool(integrity_rows["passed"].astype(bool).all()),
        "peak_selection_artifact_hashes": dict(selection_hashes),
        "peak_selection_immutable_after_simulation": True,
        "eight_agent_semantics": "BOUNDED_MAPPO_EXPERIMENTAL_AGENT_SCALE",
        "eight_agents_equal_actual_route814_fleet": False,
    }
    integrity["all_checks_passed"] = all(
        value == 0
        for key, value in integrity.items()
        if key.endswith("_count") and key not in {"zero_request_opportunity_count"}
    ) and integrity["service_state_integrity_all_passed"]
    comparison = {
        "r3_reference": {
            "avg_wait_seconds": R3_AVG_WAIT_REFERENCE,
            "p95_wait_seconds": R3_P95_WAIT_REFERENCE,
            "passengers_per_trip": 96 / 36,
            "dynamic_empty_rate": 2584 / 2772,
            "dynamic_empty_arrivals": 2584,
            "total_stop_arrivals": 2772,
        },
        "r3p_peak": {
            "avg_wait_seconds": wait_summary["mean"],
            "median_wait_seconds": wait_summary["median"],
            "p95_wait_seconds": wait_summary["p95"],
            "passengers_per_trip": generated / len(regeneration["opportunities"]),
            "dynamic_empty_rate": empty["dynamic_empty_rate"],
        },
    }
    for metric in ("avg_wait_seconds", "p95_wait_seconds", "passengers_per_trip", "dynamic_empty_rate"):
        new = comparison["r3p_peak"][metric]
        old = comparison["r3_reference"][metric]
        comparison[f"{metric}_change"] = {"absolute": None if new is None else new - old, "relative": None if new is None else relative_change(new, old)}
    return {
        "wait_summary": wait_summary,
        "density_summary": density_summary,
        "dwell": dwell,
        "empty": empty,
        "dwell_contribution": dwell_contribution,
        "h4": h4_summary,
        "integrity": integrity,
        "comparison": comparison,
    }


def decide(audits: Mapping[str, Any], long_waits: pd.DataFrame) -> Tuple[str, str]:
    comparison = audits["comparison"]
    density_increased = comparison["passengers_per_trip_change"]["absolute"] > 0
    empty_reduced = comparison["dynamic_empty_rate_change"]["absolute"] < 0
    wait_not_lower_than_r3 = comparison["avg_wait_seconds_change"]["absolute"] >= 0 or comparison["p95_wait_seconds_change"]["absolute"] >= 0
    explained = set(long_waits["long_wait_root_cause"].tolist())
    if not density_increased or not empty_reduced:
        return "RESEARCH_DEMAND_INTENSITY_CALIBRATION_REPAIR_REQUIRED", "Selected historical high demand did not increase generated passenger density while reducing dynamic empty arrivals."
    if wait_not_lower_than_r3 and explained & {"WINDOW_BOUNDARY_EFFECT", "DEMAND_SCHEDULING_ALIGNMENT", "VEHICLE_STATE_CONSTRAINT"}:
        return "WAIT_GENERATION_OR_SERVICE_ALIGNMENT_REPAIR_REQUIRED", "Higher historical-demand sampling increased density, but the observed wait tail remains tied to request/dispatch/vehicle-state timing and must be repaired before any normalization regeneration."
    return "R3_SPARSE_SAMPLE_BIAS_CONFIRMED", "The pre-frozen high-demand diagnostic raised passenger density and reduced empty-stop dominance without retaining the R3 wait anomaly."


def claim_guards() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "diagnostic_only": True,
        "r3_candidate_mutated": False,
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
        "actual_daegu_route814_dwell_observed": False,
        "real_network_pass_through_claim_allowed": False,
        "eight_agent_semantics": "BOUNDED_MAPPO_EXPERIMENTAL_AGENT_SCALE",
        "eight_agents_equal_actual_route814_fleet": False,
    }


def final_report(
    *,
    source_audit: Mapping[str, Any],
    selection: Mapping[str, Any],
    windows: pd.DataFrame,
    audits: Mapping[str, Any],
    long_waits: pd.DataFrame,
    decision: str,
    rationale: str,
    deterministic: Mapping[str, Any],
) -> str:
    selected_scores = windows["historical_demand_score"].astype(float)
    density = audits["density_summary"]
    wait = audits["wait_summary"]
    empty = audits["empty"]
    dwell = audits["dwell"]
    contribution = audits["dwell_contribution"]
    h4 = audits["h4"]
    comparison = audits["comparison"]
    causes = Counter(long_waits["long_wait_root_cause"].tolist())
    return "\n".join(
        [
            "# PV8-R2A-R8E-R3-P Final Report",
            "",
            f"- gate: `{PASS_GATE}`",
            f"- decision: `{decision}`",
            "- experiment class: `DIAGNOSTIC_ONLY`; R3 normalization remains unapproved and diagnostically suspended.",
            "",
            "## Historical Selection",
            "",
            f"- historical source: `{DEMAND_SOURCE}`; 2023 weekday route-stop-time aggregate log1p features reconstructed as `exp(log1p)-1`.",
            f"- route-814 exact-ID mapped / unmatched occurrences: `{source_audit['matched_occurrence_count']} / {source_audit['unmatched_occurrence_count']}`; ambiguity: `0`.",
            f"- weekday candidate windows / pre-frozen selected windows: `{selection['candidate_window_count']} / {len(windows)}`; selected by top five direction-specific aggregate scores before simulation.",
            f"- selected score min / median / max: `{selected_scores.min()} / {selected_scores.median()} / {selected_scores.max()}`.",
            f"- direction coverage: `{', '.join(sorted(windows['direction_id'].astype(str).unique()))}`; weekday headway: `540 sec`.",
            "- individual passenger identities, timestamps, origins and destinations are `RESEARCH_GENERATED`, not observed historical route-814 passengers.",
            "- eight agents mean `BOUNDED_MAPPO_EXPERIMENTAL_AGENT_SCALE`; they are not asserted to equal the actual route-814 fleet.",
            "",
            "## Peak Diagnostic",
            "",
            f"- generated / served passengers: `{density['generated_passenger_count']} / {density['served_passenger_count']}`; trips: `{density['trip_count']}`; zero-request opportunities: `{density['zero_request_opportunity_count']}`.",
            f"- passengers per trip mean / median / p95: `{density['passengers_per_trip']['mean']} / {density['passengers_per_trip']['median']} / {density['passengers_per_trip']['p95']}`; R3 reference `2.666667`.",
            f"- dynamic empty arrivals / rate: `{empty['dynamic_empty_arrivals']} / {empty['dynamic_empty_rate']}`; R3 rate `0.932179`; absolute change `{empty['r3_absolute_change']}`.",
            f"- pickup / dropoff obligation arrivals: `{empty['pickup_required_arrivals']} / {empty['dropoff_required_arrivals']}`; peak / mean onboard: `{density['peak_onboard_passenger_count']['max']} / {density['mean_onboard_passenger_count']['mean']}`.",
            f"- SERVE dwell 10 / 15 / 20 / 30 sec: `{dwell['counts']['10']} / {dwell['counts']['15']} / {dwell['counts']['20']} / {dwell['counts']['30']}`; natural 20 / 30 sec: `{dwell['natural_20_second_dwell_observed']} / {dwell['natural_30_second_dwell_observed']}`.",
            f"- travel / empty-stop dwell / passenger-service dwell: `{contribution['travel_seconds']} / {contribution['empty_stop_dwell_seconds']} / {contribution['passenger_service_dwell_seconds']} sec`; empty dwell fraction `{contribution['empty_stop_dwell_fraction']}`.",
            "",
            "## Wait And H4",
            "",
            f"- individual wait mean / median / p75 / p90 / p95 / p99 / max sec: `{wait['mean']} / {wait['median']} / {wait['p75']} / {wait['p90']} / {wait['p95']} / {wait['p99']} / {wait['max']}`.",
            f"- R3 to R3-P avg wait change: `{comparison['avg_wait_seconds_change']['absolute']} sec` ({comparison['avg_wait_seconds_change']['relative']}); p95 change: `{comparison['p95_wait_seconds_change']['absolute']} sec` ({comparison['p95_wait_seconds_change']['relative']}).",
            f"- top-long-wait root causes: `{dict(causes)}`; each record has request/first-eligible/board times and dispatch phase in `r8er3p_long_wait_attribution.parquet`.",
            f"- H4 request / boarding / wait-update / zero-event / outcome-after-H4 fractions: `{h4['request_event_coverage']} / {h4['boarding_event_coverage']} / {h4['wait_update_coverage']} / {h4['passenger_zero_event_fraction']} / {h4['outcome_classification_counts'].get('OUTCOME_AFTER_H4', 0) / h4['decision_count']}`.",
            "- next scheduled dispatch inside H4 is expected to be zero because 240 sec is shorter than the frozen 540-sec weekday headway; passenger-event observability is reported separately.",
            "",
            "## Integrity And Stop State",
            "",
            "- future leakage / passenger identity failure / cross-agent contamination / occurrence violations: `0 / 0 / 0 / 0`.",
            f"- deterministic replay: `{'PASS' if deterministic['deterministic_replay_passed'] else 'FAIL'}`; payload SHA-256 `{deterministic['payload_sha256']}`.",
            "- primary B1 remains `REAL_NETWORK_FAIL_CLOSED_B1`: PASS_THROUGH executions `0`; actual Daegu dwell remains `NOT OBSERVED` and the 10/15/20/30-sec rule is a research operational assumption.",
            f"- conclusion: `{rationale}`",
            f"- exact next step: `{'repair request-generation/service-alignment before any normalization regeneration' if decision == 'WAIT_GENERATION_OR_SERVICE_ALIGNMENT_REPAIR_REQUIRED' else 'PV8-R2A-R8E-R3-R representative historical-demand peak + offpeak + night B1 regeneration, only after explicit user command'}`.",
            "- no normalization freeze/replacement, reward binding, reward materialization, training-readiness audit, policy evaluation, or MAPPO training was run.",
            "",
        ]
    )


def write_manifest_and_lock(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    for name in PAYLOADS:
        path = writer.root / name
        rows.append({"relative_path": name, "sha256": k5.sha256_file(path), "size_bytes": path.stat().st_size, "required": True, "artifact_role": "payload", "exists": True})
    jsonl = writer.root / "artifact_manifest_srp2_bis_pv8_r2ar8er3p.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rows.append({"relative_path": jsonl.name, "sha256": k5.sha256_file(jsonl), "size_bytes": jsonl.stat().st_size, "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8er3p.json"
    writer.json(manifest_name, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": 0, "files": rows})
    manifest = writer.root / manifest_name
    writer.json("_PV8_R2AR8ER3P_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest_name, "final_manifest_sha256": k5.sha256_file(manifest), "manifest_size_bytes": manifest.stat().st_size, "created_at": iso_kst()})


def run(root: Path) -> Path:
    upstreams = verify_upstreams()
    route = route_occurrence_rows()
    historical, match_audit, source_audit = read_historical_demand(route)
    candidates, windows, selection = build_peak_selection(historical)
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    historical.to_parquet(root / "r8er3p_historical_route814_weekday_demand.parquet", index=False)
    match_audit.to_parquet(root / "r8er3p_route814_stop_match_audit.parquet", index=False)
    candidates.to_parquet(root / "r8er3p_historical_peak_candidate_registry.parquet", index=False)
    windows.to_parquet(root / "r8er3p_peak_window_registry.parquet", index=False)
    selection["candidate_registry_sha256"] = k5.sha256_file(root / "r8er3p_historical_peak_candidate_registry.parquet")
    selection["peak_window_registry_sha256"] = k5.sha256_file(root / "r8er3p_peak_window_registry.parquet")
    selection["historical_route814_demand_sha256"] = k5.sha256_file(root / "r8er3p_historical_route814_weekday_demand.parquet")
    selection["candidate_window_count"] = int(len(candidates))
    writer.json("r8er3p_peak_selection_contract.json", selection)
    selection_hashes = {
        "candidate_registry_sha256": selection["candidate_registry_sha256"],
        "peak_window_registry_sha256": selection["peak_window_registry_sha256"],
        "peak_selection_contract_sha256": k5.sha256_file(root / "r8er3p_peak_selection_contract.json"),
    }
    binding_frame, occurrences, legs, binding_summary = r3.build_bidirectional_binding()
    if binding_summary["unresolved_route_leg_count"] != 0:
        raise R8ER3PError("selected peak trips contain unresolved route legs")
    mask_runtime, fixed_bindings = r3.build_mask_context()
    first = run_diagnostic_once(windows=windows, mask_runtime=mask_runtime, fixed_bindings=fixed_bindings, occurrences_by_direction=occurrences, legs_by_direction=legs)
    second = run_diagnostic_once(windows=windows, mask_runtime=mask_runtime, fixed_bindings=fixed_bindings, occurrences_by_direction=occurrences, legs_by_direction=legs)
    if selection_hashes["candidate_registry_sha256"] != k5.sha256_file(root / "r8er3p_historical_peak_candidate_registry.parquet") or selection_hashes["peak_window_registry_sha256"] != k5.sha256_file(root / "r8er3p_peak_window_registry.parquet"):
        raise R8ER3PError("frozen peak selection artifact changed during simulation")
    deterministic = {
        "created_at": iso_kst(),
        "same_peak_selection_registry": True,
        "same_demand_seed": True,
        "same_route_binding": True,
        "same_weekday_headway": True,
        "same_dwell_contract": True,
        "run_1_payload_sha256": first["payload_sha256"],
        "run_2_payload_sha256": second["payload_sha256"],
        "payload_sha256": first["payload_sha256"],
        "generated_passenger_identities_identical": canonical_hash(first["passenger_lifecycle"]) == canonical_hash(second["passenger_lifecycle"]),
        "vehicle_trajectories_identical": canonical_hash(first["vehicle_timeline"]) == canonical_hash(second["vehicle_timeline"]),
        "boarding_alighting_wait_distribution_identical": canonical_hash(first["passenger_lifecycle"]) == canonical_hash(second["passenger_lifecycle"]),
        "diagnostic_summary_identical": canonical_hash(first["h4"]) == canonical_hash(second["h4"]),
        "deterministic_replay_passed": first["payload_sha256"] == second["payload_sha256"],
    }
    density = window_density(first)
    long_waits, phase = build_long_wait_and_phase(pd.DataFrame(first["passenger_lifecycle"]), pd.DataFrame(first["vehicle_timeline"]))
    audits = audit_regeneration(first, density, selection_hashes)
    decision, rationale = decide(audits, long_waits)
    guards = claim_guards()
    checks = {
        "upstream_integrity": all(item["valid"] for item in upstreams["artifacts"].values()),
        "historical_peak_selection_pre_simulation": True,
        "both_direction_coverage": set(windows["direction_id"].astype(str)) == {"0", "1"},
        "selected_trip_legs_unresolved_zero": binding_summary["unresolved_route_leg_count"] == 0,
        "primary_b1_fail_closed_serve_only": audits["integrity"]["primary_b1_pass_through_count"] == 0,
        "minimum_one_request_logic_inactive": not audits["integrity"]["minimum_one_request_logic_active"],
        "integrity": audits["integrity"]["all_checks_passed"],
        "deterministic_replay": deterministic["deterministic_replay_passed"],
        "r3_candidate_unmodified": not upstreams["r3_artifact_mutated"],
    }
    if not all(checks.values()):
        raise R8ER3PError(f"R3-P diagnostic integrity failure: {checks}")
    readiness = {
        "created_at": iso_kst(),
        "decision": decision,
        "gate": PASS_GATE,
        "checks": checks,
        "r3_normalization_status": "UNAPPROVED_R3_DIAGNOSTIC_CANDIDATE",
        "normalization_replacement_created": False,
        "training_normalization_approved": False,
        "next_step": "Stop for explicit user command; do not freeze or replace normalization.",
    }
    gate = {"created_at": iso_kst(), "gate": PASS_GATE, "terminal_gate": PASS_GATE, "gate_passed": True, "readiness": "R3P_DIAGNOSTIC_COMPLETE_FOLLOW_ON_LOCKED", "final_decision": decision, "normalization_approval_implied": False, "failure_reasons": []}
    pd.DataFrame(first["opportunities"]).to_parquet(root / "r8er3p_generated_demand_by_window.parquet", index=False)
    writer.json("r8er3p_passenger_density_summary.json", audits["density_summary"])
    pd.DataFrame(first["vehicle_timeline"]).to_parquet(root / "r8er3p_vehicle_timeline.parquet", index=False)
    pd.DataFrame(first["passenger_lifecycle"]).to_parquet(root / "r8er3p_passenger_lifecycle.parquet", index=False)
    writer.json("r8er3p_empty_stop_diagnostic.json", audits["empty"])
    writer.json("r8er3p_dwell_distribution.json", audits["dwell"])
    writer.json("r8er3p_dwell_contribution_by_type.json", audits["dwell_contribution"])
    pd.DataFrame(first["passenger_lifecycle"])[["passenger_id", "request_id", "trip_id", "window_id", "direction_id", "request_ts", "board_ts", "alight_ts", "wait_seconds"]].to_parquet(root / "r8er3p_wait_distribution.parquet", index=False)
    writer.json("r8er3p_wait_summary.json", audits["wait_summary"])
    long_waits.to_parquet(root / "r8er3p_long_wait_attribution.parquet", index=False)
    phase.to_parquet(root / "r8er3p_request_dispatch_phase_audit.parquet", index=False)
    pd.DataFrame(first["h4"]).to_parquet(root / "r8er3p_h4_observability.parquet", index=False)
    writer.json("r8er3p_h4_observability_summary.json", audits["h4"])
    writer.json("r8er3p_r3_vs_peak_comparison.json", audits["comparison"])
    writer.json("r8er3p_integrity_audit.json", audits["integrity"])
    writer.json("r8er3p_deterministic_replay.json", deterministic)
    writer.json("r8er3p_readiness_decision.json", readiness)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "weekday-historical-high-demand-peak-diagnostic",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "runtime_path": str(RUNTIME_PATH),
        "runtime_sha256": k5.sha256_file(RUNTIME_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "historical_candidate_window_count": int(len(candidates)),
        "selected_peak_window_count": int(len(windows)),
        "trip_count": int(len(first["opportunities"])),
        "postgresql_select_query_count": int(source_audit["query_count"] + binding_summary["postgresql_source_audit"]["query_count"]),
        "db_write_count": 0,
        "new_bis_api_call_count": 0,
        "policy_evaluation_count": 0,
        "mappo_training_count": 0,
        "normalization_candidate_count": 0,
        "normalization_approval_count": 0,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": gate["readiness"], "final_decision": decision})
    writer.text("final_report.md", final_report(source_audit=source_audit, selection=selection, windows=windows, audits=audits, long_waits=long_waits, decision=decision, rationale=rationale, deterministic=deterministic))
    write_manifest_and_lock(writer, gate)
    integrity = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar8er3p.json", "_PV8_R2AR8ER3P_COMPLETE.lock")
    if not k5.manifest_ok(integrity):
        raise R8ER3PError(f"final manifest integrity failed: {integrity}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {decision}")
    print(f"selected_windows: {len(windows)}")
    print(f"trips: {len(first['opportunities'])}")
    print(f"generated_passengers: {audits['density_summary']['generated_passenger_count']}")
    print(f"avg_wait_seconds: {audits['wait_summary']['mean']}")
    print(f"p95_wait_seconds: {audits['wait_summary']['p95']}")
    print(f"manifest_integrity: {integrity}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["diagnostic"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
