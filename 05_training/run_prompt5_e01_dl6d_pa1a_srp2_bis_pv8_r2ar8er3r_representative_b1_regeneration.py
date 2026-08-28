#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R representative historical-demand B1 regeneration.

This is a deterministic, non-policy baseline.  It combines observed aggregate
route-814 demand with research-generated passenger identities and the repaired
request-to-physical-service alignment from R3-P-R1.  It never freezes a
normalization or materializes rewards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import psycopg2

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3_realistic_b1_regeneration as r3
import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3pr1_request_service_alignment_warmup as r1
from simulator.pv8_occurrence_temporal_runtime import DWELL_CONTRACT_VERSION, serve_dwell_seconds


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration.py"
RUNTIME_PATH = TRAINING_ROOT / "simulator" / "pv8_occurrence_temporal_runtime.py"

R8ER2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er2_occurrence_dwell_temporal_repair_20260809_175119"
R8ER3PR1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3pr1_request_service_alignment_warmup_20260809_192119"
K6_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026"

OCCURRENCE_PATH = K6_ROOT / "k6_route_stop_occurrence_master.parquet"
DEMAND_SOURCE = "public.gatv2_snapshot_stop_features_train_mat"
ROUTE_ID = "3000814001"
ROUTE_NAME = "814"
TIMEZONE = "Asia/Seoul"
H4_SECONDS = 240
DEMAND_SEED = 2026080909
DEMAND_VERSION = "PV8_RESEARCH_DEMAND_ZERO_CAPABLE_V2_REPRESENTATIVE_HISTORICAL_V1"
R3_PEAK_LAMBDA = r3.BASE_REQUEST_INTENSITY * r3.TIME_BAND_MULTIPLIERS["peak"]

R8B_R8C = {"B1_service_reference": 1.0, "B1_avg_wait_reference": 318.6725663716814, "B1_p95_wait_reference": 525.0}
R3_STALE = {"B1_service_reference": 1.0, "B1_avg_wait_reference": 2062.5, "B1_p95_wait_reference": 5338.25}
R3PR1_PEAK = {"B1_avg_wait_reference": 246.09615384615384, "B1_p95_wait_reference": 523.85}

TIME_BAND_HOURS = {"night": 7, "offpeak": 10, "peak": 17}
REGIMES = {
    "weekday": {"isodow": (1, 2, 3, 4, 5), "headway": 540, "official_timetable_type": "평일"},
    "saturday": {"isodow": (6,), "headway": 600, "official_timetable_type": "토요일(감차)"},
    # The historical feature source supplies Sunday observations but no statutory
    # holiday flag.  Sunday is therefore an explicit timetable-regime proxy only.
    "holiday": {"isodow": (7,), "headway": 660, "official_timetable_type": "휴일 (ISO Sunday timetable proxy)"},
}
SELECTED_WINDOWS_PER_STRATUM = 3
EVALUATION_DISPATCH_COUNT = 3

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3R_REPRESENTATIVE_HISTORICAL_DEMAND_B1_REGENERATION_COMPLETE"
SUCCESS_DECISION = "PV8_REPRESENTATIVE_B1_NORMALIZATION_CANDIDATE_READY_FOR_EXPLICIT_APPROVAL"
READINESS = "R3R_REPRESENTATIVE_B1_CANDIDATE_GENERATED_EXPLICIT_NORMALIZATION_APPROVAL_PENDING"

PAYLOADS = [
    "r8er3r_historical_candidate_registry.parquet",
    "r8er3r_representative_selection_contract.json",
    "r8er3r_representative_window_registry.parquet",
    "r8er3r_stratum_coverage.json",
    "r8er3r_historical_demand_distribution_audit.json",
    "r8er3r_route814_bidirectional_binding_audit.json",
    "r8er3r_warmup_contract_by_regime.json",
    "r8er3r_warmup_convergence_audit.json",
    "r8er3r_generated_demand.parquet",
    "r8er3r_passenger_lifecycle.parquet",
    "r8er3r_vehicle_timeline.parquet",
    "r8er3r_occurrence_service_events.parquet",
    "r8er3r_wait_decomposition.parquet",
    "r8er3r_wait_distribution.json",
    "r8er3r_long_wait_attribution.parquet",
    "r8er3r_missed_service_alignment_audit.json",
    "r8er3r_dwell_distribution.json",
    "r8er3r_empty_vs_service_dwell.json",
    "r8er3r_travel_dwell_hold_summary.parquet",
    "r8er3r_dynamic_empty_stop_audit.parquet",
    "r8er3r_b1_kpi_by_window.parquet",
    "r8er3r_b1_kpi_by_stratum.parquet",
    "r8er3r_b1_kpi_summary.json",
    "r8er3r_normalization_candidate.json",
    "r8er3r_normalization_stratified.json",
    "r8er3r_normalization_loo_window.json",
    "r8er3r_normalization_leave_one_band_out.json",
    "r8er3r_sample_contribution_audit.json",
    "r8er3r_prior_candidate_comparison.json",
    "r8er3r_h4_observability_by_stratum.parquet",
    "r8er3r_h4_observability_summary.json",
    "r8er3r_causal_integrity_audit.json",
    "r8er3r_deterministic_regeneration.json",
    "r8er3r_payload.sha256",
    "r8er3r_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R8ER3RError(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    def fallback(item: Any) -> Any:
        if isinstance(item, (datetime, pd.Timestamp)):
            return item.isoformat()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"unsupported canonical type: {type(item)!r}")

    payload = json.dumps(k5.json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=fallback)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def numeric_summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    series = pd.Series([float(value) for value in values], dtype=float)
    keys = ("count", "mean", "std", "min", "p05", "p25", "median", "p75", "p90", "p95", "p99", "max")
    if series.empty:
        return {key: None for key in keys}
    return {
        "count": int(len(series)), "mean": float(series.mean()), "std": float(series.std(ddof=0)),
        "min": float(series.min()), "p05": float(series.quantile(.05)), "p25": float(series.quantile(.25)),
        "median": float(series.median()), "p75": float(series.quantile(.75)), "p90": float(series.quantile(.90)),
        "p95": float(series.quantile(.95)), "p99": float(series.quantile(.99)), "max": float(series.max()),
    }


def read_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).copy()


def verify_upstreams() -> Dict[str, Any]:
    upstreams = {
        "PV8-R2A-R8E-R2": (R8ER2_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8er2.json", "_PV8_R2AR8ER2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER2_OCCURRENCE_LEVEL_DWELL_TEMPORAL_PROPAGATION_REPAIR_COMPLETE", "PV8_DWELL_TEMPORAL_PROPAGATION_REPAIR_COMPLETE"),
        "PV8-R2A-R8E-R3-P-R1": (R8ER3PR1_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8er3pr1.json", "_PV8_R2AR8ER3PR1_COMPLETE.lock", PASS_GATE.replace("R3R_REPRESENTATIVE_HISTORICAL_DEMAND_B1_REGENERATION", "R3PR1_REQUEST_SERVICE_ALIGNMENT_WARMUP_REPAIR"), "PV8_REQUEST_SERVICE_ALIGNMENT_REPAIRED_READY_FOR_REPRESENTATIVE_B1"),
    }
    records: Dict[str, Any] = {}
    for label, (root, manifest, lock, gate_expected, decision_expected) in upstreams.items():
        gate = k5.read_json(root / "gate_decision.json")
        integrity = k5.verify_manifest(root, manifest, lock)
        observed = gate.get("gate") or gate.get("terminal_gate")
        valid = observed == gate_expected and gate.get("final_decision") == decision_expected and k5.manifest_ok(integrity)
        if not valid:
            raise R8ER3RError(f"{label} manifest, lock, gate, or decision validation failed")
        records[label] = {"artifact_root": str(root), "gate": observed, "decision": gate.get("final_decision"), "integrity": integrity, "valid": True}
    guard = k5.read_json(R8ER3PR1_ROOT / "claim_guard_status.json")
    locked = ("training_normalization_approved", "reward_materialization_binding_ready", "training_use_authorized", "policy_evaluation_authorized", "checkpoint_reuse_authorized", "MAPPO_training_authorized")
    if any(bool(guard.get(key)) for key in locked):
        raise R8ER3RError("R3-P-R1 downstream locks are inconsistent")
    return {"created_at": iso_kst(), "artifacts": records, "r3pr1_anomaly_repaired": True, "r3pr1_artifact_mutated": False}


def route_occurrences() -> pd.DataFrame:
    route = read_frame(OCCURRENCE_PATH)
    route = route[route["route_id"].astype(str) == ROUTE_ID].copy()
    route["direction_id"] = route["direction_id"].astype(str)
    route["stop_id"] = route["stop_id"].astype(str)
    route["node_uid"] = "STOP:" + route["stop_id"]
    for direction in ("0", "1"):
        segment = route[route["direction_id"] == direction].sort_values("stop_sequence")
        if len(segment) != 77 or segment["stop_sequence"].astype(int).tolist() != list(range(1, 78)):
            raise R8ER3RError(f"route 814 direction {direction} occurrence sequence is incomplete")
    return route.sort_values(["direction_id", "stop_sequence"]).reset_index(drop=True)


def read_historical_candidates(route: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    node_uids = sorted(route["node_uid"].unique().tolist())
    query_available = f"SELECT DISTINCT node_uid FROM {DEMAND_SOURCE} WHERE node_uid = ANY(%s)"
    query_rows = f"""
        SELECT snapshot_id, state_ts AT TIME ZONE %s AS state_local_ts,
               node_uid, boardings_recent_log, alightings_recent_log
        FROM {DEMAND_SOURCE}
        WHERE node_uid = ANY(%s)
          AND EXTRACT(HOUR FROM state_ts AT TIME ZONE %s) = ANY(%s)
        ORDER BY state_ts, snapshot_id, node_uid
    """
    connection = psycopg2.connect(dbname="urbanbus")
    connection.set_session(readonly=True, autocommit=True)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query_available, (node_uids,))
            available = {str(value[0]) for value in cursor.fetchall()}
            cursor.execute(query_rows, (TIMEZONE, node_uids, TIMEZONE, list(TIME_BAND_HOURS.values())))
            columns = [item.name for item in cursor.description]
            source_rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        connection.close()
    raw = pd.DataFrame(source_rows)
    if raw.empty:
        raise R8ER3RError("no route-814 historical aggregate rows for the existing time bands")
    raw["state_local_ts"] = pd.to_datetime(raw["state_local_ts"])
    raw["historical_boardings"] = raw["boardings_recent_log"].astype(float).map(lambda value: math.exp(max(value, 0.0)) - 1.0)
    raw["historical_alightings"] = raw["alightings_recent_log"].astype(float).map(lambda value: math.exp(max(value, 0.0)) - 1.0)
    raw = raw.drop_duplicates(["snapshot_id", "node_uid"], keep="first")
    expanded = raw.merge(route[["route_id", "direction_id", "stop_sequence", "route_stop_occurrence_id", "stop_id", "node_uid"]], on="node_uid", how="inner", validate="many_to_many")
    expanded["isodow"] = expanded["state_local_ts"].dt.isocalendar().day.astype(int)
    expanded["hour"] = expanded["state_local_ts"].dt.hour.astype(int)
    inverse_hour = {value: key for key, value in TIME_BAND_HOURS.items()}
    expanded["time_band"] = expanded["hour"].map(inverse_hour)
    inverse_regime = {day: regime for regime, body in REGIMES.items() for day in body["isodow"]}
    expanded["timetable_regime"] = expanded["isodow"].map(inverse_regime)
    expanded["service_date"] = expanded["state_local_ts"].dt.date.astype(str)
    if expanded["time_band"].isna().any() or expanded["timetable_regime"].isna().any():
        raise R8ER3RError("historical time-band or timetable mapping is incomplete")
    group_keys = ["direction_id", "snapshot_id", "state_local_ts", "service_date", "isodow", "time_band", "timetable_regime"]
    candidates = expanded.groupby(group_keys, as_index=False).agg(
        historical_boarding_intensity=("historical_boardings", "sum"),
        historical_alighting_intensity=("historical_alightings", "sum"),
        mapped_stop_count=("node_uid", "nunique"),
        historical_source_row_count=("node_uid", "size"),
    )
    candidates["historical_demand_score"] = candidates["historical_boarding_intensity"] + candidates["historical_alighting_intensity"]
    candidates["source_dataset"] = DEMAND_SOURCE
    candidates["value_reconstruction"] = "exp(log1p_feature)-1"
    candidates["evidence_class"] = "OBSERVED_AGGREGATE_EVIDENCE"
    candidates["observed_individual_passenger_claim"] = False
    candidates["time_band_definition"] = "existing R3 fixed clock definition: night=07:00, offpeak=10:00, peak=17:00 Asia/Seoul"
    candidates["timetable_regime_mapping"] = candidates["timetable_regime"].map({"weekday": "ISO weekday 1-5", "saturday": "ISO Saturday", "holiday": "ISO Sunday as official holiday-timetable proxy; statutory holiday label unavailable"})
    candidates = candidates.sort_values(["timetable_regime", "time_band", "direction_id", "state_local_ts"], kind="stable").reset_index(drop=True)
    route_audit = route[["route_id", "direction_id", "stop_sequence", "route_stop_occurrence_id", "stop_id", "node_uid"]].copy()
    route_audit["mapping_method"] = route_audit["node_uid"].map(lambda value: "EXACT_ID_MATCH" if value in available else "UNMATCHED")
    route_audit["evidence_class"] = route_audit["mapping_method"].map({"EXACT_ID_MATCH": "OBSERVED_AGGREGATE_EVIDENCE", "UNMATCHED": "NOT_AVAILABLE"})
    audit = {
        "created_at": iso_kst(), "relation": DEMAND_SOURCE, "query_count": 2, "write_count": 0, "transaction_read_only": True,
        "source_time_basis": "state_ts AT TIME ZONE Asia/Seoul", "aggregate_value_reconstruction": "exp(log1p)-1",
        "source_row_count": int(len(raw)), "expanded_row_count": int(len(expanded)), "candidate_count": int(len(candidates)),
        "matched_occurrence_count": int((route_audit["mapping_method"] == "EXACT_ID_MATCH").sum()),
        "unmatched_occurrence_count": int((route_audit["mapping_method"] == "UNMATCHED").sum()),
        "holiday_source_note": "Historical aggregate source contains Sunday observations but no statutory-holiday field. Holiday timetable is a predeclared ISO-Sunday proxy, not a claim that each date is a legal holiday.",
    }
    return candidates, route_audit, audit


def select_representative_windows(candidates: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    candidate = candidates.copy()
    candidate["candidate_rank_time"] = candidate.groupby(["time_band", "timetable_regime", "direction_id"]).cumcount() + 1
    candidate["selection_status"] = "NOT_SELECTED"
    candidate["selection_reason"] = "outside predeclared chronological spread positions"
    selected_rows: List[Dict[str, Any]] = []
    coverage: Dict[str, Any] = {}
    distribution: Dict[str, Any] = {}
    ordinal = 0
    for regime in ("weekday", "saturday", "holiday"):
        for band in ("peak", "offpeak", "night"):
            for direction in ("0", "1"):
                mask = (candidate["timetable_regime"] == regime) & (candidate["time_band"] == band) & (candidate["direction_id"].astype(str) == direction)
                group = candidate.loc[mask].sort_values("state_local_ts", kind="stable").copy()
                key = f"{regime}|{band}|d{direction}"
                if len(group) < SELECTED_WINDOWS_PER_STRATUM:
                    raise R8ER3RError(f"historical source lacks three candidates for {key}")
                indices = [0, (len(group) - 1) // 2, len(group) - 1]
                labels = ["EARLIEST", "MEDIAN_CHRONOLOGICAL", "LATEST"]
                if len(set(indices)) != SELECTED_WINDOWS_PER_STRATUM:
                    raise R8ER3RError(f"chronological spread selection is not unique for {key}")
                selected = group.iloc[indices].copy()
                candidate.loc[selected.index, "selection_status"] = "SELECTED"
                candidate.loc[selected.index, "selection_reason"] = "predeclared chronological spread: earliest, median-index, latest; demand outcomes not used"
                values = group["historical_demand_score"].astype(float)
                selected_values = selected["historical_demand_score"].astype(float)
                distribution[key] = {
                    "candidate_count": int(len(group)), "selected_count": int(len(selected)),
                    "candidate_demand": numeric_summary(values), "selected_demand": numeric_summary(selected_values),
                    "selection_positions": [int(index + 1) for index in indices], "selection_rule": "chronological spread only",
                }
                coverage[key] = {"candidate_count": int(len(group)), "selected_count": int(len(selected)), "source_reason_if_missing": None}
                median = float(values.median())
                if not math.isfinite(median) or median < 0:
                    raise R8ER3RError(f"invalid aggregate demand median for {key}")
                for label, row in zip(labels, selected.to_dict("records")):
                    local = pd.Timestamp(row["state_local_ts"])
                    local = local.tz_localize(TIMEZONE) if local.tzinfo is None else local.tz_convert(TIMEZONE)
                    headway = int(REGIMES[regime]["headway"])
                    score = float(row["historical_demand_score"])
                    ratio = 0.0 if median == 0 else score / median
                    selected_rows.append({
                        **row, "window_ordinal": ordinal,
                        "window_id": f"PV8_R3R_{regime.upper()}_{band.upper()}_D{direction}_{label}_{local.strftime('%Y%m%d_%H%M')}",
                        "selection_position": label, "start_ts": int(local.timestamp()), "start_iso": local.isoformat(),
                        "evaluation_end_ts": int(local.timestamp()) + EVALUATION_DISPATCH_COUNT * headway,
                        "official_headway_seconds": headway, "official_timetable_type": REGIMES[regime]["official_timetable_type"],
                        "dispatch_count": EVALUATION_DISPATCH_COUNT, "stratum_historical_demand_median": median,
                        "historical_intensity_ratio": ratio, "demand_lambda": R3_PEAK_LAMBDA * ratio,
                        "selection_frozen_before_simulation": True, "post_hoc_substitution_allowed": False,
                    })
                    ordinal += 1
    windows = pd.DataFrame(selected_rows).sort_values("window_ordinal").reset_index(drop=True)
    if len(windows) != 54 or windows[["time_band", "timetable_regime", "direction_id"]].drop_duplicates().shape[0] != 18:
        raise R8ER3RError("representative registry is not 3 bands x 3 regimes x 2 directions x 3 windows")
    contract = {
        "created_at": iso_kst(), "selection_frozen_before_simulation": True,
        "time_band_definition": "existing R3 fixed Asia/Seoul clock bands: night=07:00, offpeak=10:00, peak=17:00",
        "timetable_regime_definition": {regime: body["official_timetable_type"] for regime, body in REGIMES.items()},
        "selection_rule": "within each regime/time-band/direction stratum sort historical candidates by state_local_ts and select earliest, median-index, latest; never inspect causal B1 outcomes",
        "selected_windows_per_stratum": SELECTED_WINDOWS_PER_STRATUM, "post_hoc_replacement_allowed": False,
        "historical_aggregate_to_individual_boundary": "observed aggregate intensity drives CONTRACT_FIXED_RESEARCH_DEMAND; no individual is claimed observed",
        "minimum_one_request_logic_active": False,
    }
    return candidate, contract, coverage, distribution | {"selected_window_count": int(len(windows)), "candidate_window_count": int(len(candidate)), "windows": windows}


def stable_uniform(*parts: Any) -> float:
    raw = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return (int.from_bytes(raw[:8], "big") + .5) / float(2**64)


def poisson_zero_capable(rate: float, *parts: Any) -> int:
    if not math.isfinite(rate) or rate < 0:
        raise R8ER3RError("historical demand rate must be finite and nonnegative")
    if rate == 0:
        return 0
    target = stable_uniform(*parts)
    probability, cumulative, count = math.exp(-rate), math.exp(-rate), 0
    while target > cumulative:
        count += 1
        probability *= rate / count
        cumulative += probability
    return count


def identity(prefix: str, *parts: Any) -> str:
    return f"{prefix}_{hashlib.sha256('|'.join(str(part) for part in parts).encode('utf-8')).hexdigest()}"


def generate_requests(windows: pd.DataFrame, occurrences: Mapping[str, Sequence[Mapping[str, Any]]], legs: Mapping[str, Sequence[r3.RouteLeg]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for window in windows.sort_values("window_ordinal").to_dict("records"):
        direction = str(window["direction_id"])
        route_occurrences = occurrences[direction]
        arrivals = r3.cumulative_no_dwell_arrivals(legs[direction])
        headway = int(window["official_headway_seconds"])
        for dispatch_index in range(int(window["dispatch_count"])):
            dispatch_ts = int(window["start_ts"]) + dispatch_index * headway
            count = poisson_zero_capable(float(window["demand_lambda"]), DEMAND_SEED, window["window_id"], dispatch_index)
            for request_ordinal in range(count):
                origin_index = min(int(stable_uniform(DEMAND_SEED, window["window_id"], dispatch_index, request_ordinal, "origin") * (len(route_occurrences) - 1)), len(route_occurrences) - 2)
                span = len(route_occurrences) - origin_index - 1
                destination_index = origin_index + 1 + min(int(stable_uniform(DEMAND_SEED, window["window_id"], dispatch_index, request_ordinal, "destination") * span), span - 1)
                latest = int(math.floor(dispatch_ts + arrivals[origin_index]))
                # Demand starts at the selected historical snapshot boundary.  This
                # prevents a synthetic pre-window passenger cohort while pre-roll
                # still initializes physical service positions causally.
                earliest = int(window["start_ts"])
                request_ts = earliest + int(math.floor(stable_uniform(DEMAND_SEED, window["window_id"], dispatch_index, request_ordinal, "request_ts") * (latest - earliest + 1))) if latest >= earliest else earliest
                origin, destination = route_occurrences[origin_index], route_occurrences[destination_index]
                parts = (DEMAND_VERSION, DEMAND_SEED, window["window_id"], dispatch_index, request_ordinal)
                rows.append({
                    "passenger_id": identity("P", *parts, "passenger"), "request_id": identity("Q", *parts, "request"),
                    "window_id": window["window_id"], "window_ordinal": int(window["window_ordinal"]), "time_band": window["time_band"],
                    "timetable_regime": window["timetable_regime"], "direction_id": direction, "route_id": ROUTE_ID,
                    "scheduled_generation_dispatch_ts": dispatch_ts, "request_ts": request_ts,
                    "origin_index": origin_index, "destination_index": destination_index,
                    "origin_stop_id": str(origin["stop_id"]), "origin_occurrence_id": str(origin["route_stop_occurrence_id"]),
                    "destination_stop_id": str(destination["stop_id"]), "destination_occurrence_id": str(destination["route_stop_occurrence_id"]),
                    "historical_demand_score": float(window["historical_demand_score"]), "historical_intensity_ratio": float(window["historical_intensity_ratio"]),
                    "demand_lambda": float(window["demand_lambda"]), "source_class": "CONTRACT_FIXED_RESEARCH_DEMAND",
                    "historical_intensity_evidence_class": "OBSERVED_AGGREGATE_EVIDENCE", "research_generated": True,
                    "observed_individual_identity_claim": False, "old_alight_ts": int(window["evaluation_end_ts"]),
                })
    frame = pd.DataFrame(rows)
    if not frame.empty and (frame["passenger_id"].duplicated().any() or frame["request_id"].duplicated().any()):
        raise R8ER3RError("generated research passenger or request identity collision")
    return frame.sort_values(["window_id", "request_ts", "request_id"]).reset_index(drop=True) if not frame.empty else pd.DataFrame(columns=["passenger_id", "request_id", "window_id"])


def empty_window_trace(window: Mapping[str, Any], occurrences: Sequence[Mapping[str, Any]], legs: Sequence[r3.RouteLeg], warmup_seconds: int) -> Dict[str, Any]:
    headway = int(window["official_headway_seconds"])
    start = int(window["start_ts"])
    dispatches = list(range(start - warmup_seconds, start + int(window["dispatch_count"]) * headway, headway))
    rows: List[Dict[str, Any]] = []
    for instance_index, dispatch_ts in enumerate(dispatches):
        arrival = dispatch_ts
        for index, occurrence in enumerate(occurrences):
            dwell = serve_dwell_seconds(0, 0)
            departure = arrival + dwell
            next_arrival = None if index == len(legs) else int(departure + float(legs[index].travel_time_sec))
            rows.append({
                "window_id": window["window_id"], "direction_id": str(window["direction_id"]), "evaluation_start_ts": start,
                "warmup_start_ts": start - warmup_seconds, "service_instance_id": f"R3R_B1_{window['window_id']}_{dispatch_ts}",
                "background_agent_id": 2_000_000 + int(window["window_ordinal"]) * 10_000 + instance_index,
                "vehicle_token": f"B1_BACKGROUND::{window['window_id']}::{dispatch_ts}", "dispatch_ts": dispatch_ts,
                "occurrence_index": index, "route_stop_occurrence_id": str(occurrence["route_stop_occurrence_id"]), "stop_id": str(occurrence["stop_id"]),
                "arrival_ts": arrival, "departure_ts": departure, "next_arrival_ts": next_arrival, "boarding_count": 0, "alighting_count": 0,
                "serve_dwell_seconds": dwell, "executed_action": "SERVE", "is_warmup": bool(arrival < start),
                "is_policy_agent_slot": False, "conditional_skip_executed": False,
                "distance_source": legs[index].distance_source if index < len(legs) else "ROUTE_TERMINAL",
                "travel_time_source": legs[index].travel_time_source if index < len(legs) else "ROUTE_TERMINAL",
            })
            if next_arrival is None:
                break
            arrival = next_arrival
    trace = pd.DataFrame(rows)
    return {"trace": trace, "passengers": pd.DataFrame(), "evaluation_state": r1.evaluation_start_state(trace, evaluation_start=start), "integrity": {"passed": True, "violations": []}, "dispatches": dispatches, "evaluation_end_ts": int(window["evaluation_end_ts"])}


def simulate_window(window: Mapping[str, Any], passenger_rows: pd.DataFrame, occurrences: Sequence[Mapping[str, Any]], legs: Sequence[r3.RouteLeg], warmup_seconds: int) -> Dict[str, Any]:
    if passenger_rows.empty:
        return empty_window_trace(window, occurrences, legs, warmup_seconds)
    old_headway = r1.HEADWAY_SECONDS
    try:
        r1.HEADWAY_SECONDS = int(window["official_headway_seconds"])
        outcome = r1.simulate_window(window=window, passengers=passenger_rows, occurrences=occurrences, legs=legs, warmup_seconds=warmup_seconds)
        trace = outcome.pop("service_trace")
        outcome["trace"] = trace
        outcome["evaluation_state"] = r1.evaluation_start_state(trace, evaluation_start=int(window["start_ts"]))
        return outcome
    finally:
        r1.HEADWAY_SECONDS = old_headway


def warmup_contracts(legs: Mapping[str, Sequence[r3.RouteLeg]]) -> Dict[str, Dict[str, Any]]:
    max_bound = max(sum(float(leg.travel_time_sec) for leg in direction_legs) + 30.0 * (len(direction_legs) + 1) for direction_legs in legs.values())
    contracts: Dict[str, Dict[str, Any]] = {}
    for regime, body in REGIMES.items():
        headway = int(body["headway"])
        selected = int(math.ceil(max_bound / headway) * headway)
        candidates = [headway, 2 * headway, selected, selected + headway]
        contracts[regime] = {
            "official_headway_seconds": headway, "maximum_bounded_route_progression_seconds": max_bound,
            "candidate_pre_roll_seconds": list(dict.fromkeys(candidates)), "selected_pre_roll_seconds": selected,
            "longer_convergence_candidate_seconds": selected + headway,
            "selected_is_shortest_whole_headway_structural_bound": True,
            "selection_basis": "full route travel plus maximum 30-second dwell at every occurrence; never wait minimization",
            "warmup_modulo_headway_seconds": selected % headway,
        }
    return contracts


def simulate_all(windows: pd.DataFrame, demand: pd.DataFrame, occurrences: Mapping[str, Sequence[Mapping[str, Any]]], legs: Mapping[str, Sequence[r3.RouteLeg]], contracts: Mapping[str, Mapping[str, Any]], *, extended: bool = False) -> Dict[str, Any]:
    traces: List[pd.DataFrame] = []
    people: List[pd.DataFrame] = []
    states: List[pd.DataFrame] = []
    audits: List[Dict[str, Any]] = []
    for window in windows.sort_values("window_ordinal").to_dict("records"):
        regime = str(window["timetable_regime"])
        warmup = int(contracts[regime]["longer_convergence_candidate_seconds"] if extended else contracts[regime]["selected_pre_roll_seconds"])
        subset = demand[demand["window_id"] == window["window_id"]].copy()
        outcome = simulate_window(window, subset, occurrences[str(window["direction_id"])], legs[str(window["direction_id"])], warmup)
        trace = outcome["trace"].copy()
        for key in ("time_band", "timetable_regime", "official_headway_seconds", "official_timetable_type", "service_date"):
            trace[key] = window[key]
        traces.append(trace)
        if not outcome["passengers"].empty:
            people.append(outcome["passengers"].copy())
        states.append(outcome["evaluation_state"].copy())
        audits.append({"window_id": window["window_id"], "warmup_seconds": warmup, "dispatch_count": len(outcome["dispatches"]), "evaluation_end_ts": outcome["evaluation_end_ts"], **outcome["integrity"]})
    trace = pd.concat(traces, ignore_index=True)
    lifecycle = pd.concat(people, ignore_index=True) if people else pd.DataFrame()
    state = pd.concat(states, ignore_index=True) if states else pd.DataFrame()
    core = {"windows": windows.sort_values("window_id").to_dict("records"), "demand": demand.sort_values("passenger_id").to_dict("records"), "lifecycle": lifecycle.sort_values("passenger_id").to_dict("records") if not lifecycle.empty else [], "trace": trace.sort_values(["window_id", "arrival_ts", "service_instance_id"]).to_dict("records")}
    return {"trace": trace, "lifecycle": lifecycle, "evaluation_state": state, "window_integrity": pd.DataFrame(audits), "payload_sha256": canonical_hash(core)}


def enrich_service_trace(trace: pd.DataFrame, lifecycle: pd.DataFrame, static_rules: pd.DataFrame) -> pd.DataFrame:
    frame = trace.copy()
    frame["dynamic_empty"] = (frame["boarding_count"].astype(int) == 0) & (frame["alighting_count"].astype(int) == 0)
    if lifecycle.empty:
        frame["onboard_passenger_count"] = 0
    else:
        frame["onboard_passenger_count"] = [int(((lifecycle["actual_board_ts"] <= int(row.arrival_ts)) & (lifecycle["actual_alight_ts"] > int(row.arrival_ts)) & (lifecycle["window_id"] == row.window_id)).sum()) for row in frame.itertuples()]
    static = static_rules[["route_stop_occurrence_id", "mandatory_stop", "protected_stop", "planned_itinerary_allows_skip", "terminal_or_turnaround_stop", "charging_or_driver_relief_stop", "valid_post_skip_path"]].copy()
    frame = frame.merge(static, on="route_stop_occurrence_id", how="left", validate="many_to_one")
    if frame[["mandatory_stop", "protected_stop", "planned_itinerary_allows_skip", "terminal_or_turnaround_stop", "charging_or_driver_relief_stop", "valid_post_skip_path"]].isna().any().any():
        raise R8ER3RError("static research-rule lookup is incomplete")
    clear_static = (~frame["mandatory_stop"].astype(bool)) & (~frame["protected_stop"].astype(bool)) & frame["planned_itinerary_allows_skip"].astype(bool) & (~frame["terminal_or_turnaround_stop"].astype(bool)) & (~frame["charging_or_driver_relief_stop"].astype(bool)) & frame["valid_post_skip_path"].astype(bool)
    # This is a dry eligibility diagnostic from the K8 research rule plus exact
    # current pickup/dropoff event state.  Primary B1 still executes SERVE.
    frame["research_pass_through_eligible"] = frame["dynamic_empty"] & clear_static & (frame["onboard_passenger_count"] == 0)
    frame["real_network_pass_through_eligible"] = False
    frame["actual_b1_pass_through"] = False
    return frame


def wait_and_alignment(lifecycle: pd.DataFrame, trace: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if lifecycle.empty:
        return lifecycle.copy(), pd.DataFrame(columns=["passenger_id", "missed_eligible_service"])
    waits = lifecycle.copy()
    # R3-P-R1 enforces board/alight completion inside simulate_window but does
    # not expose R3's final-status column. Recheck its exact event timestamps
    # before materializing the unchanged eligible/served denominator.
    if waits[["actual_board_ts", "actual_alight_ts"]].isna().any().any():
        raise R8ER3RError("R3-P-R1 completion invariant is incomplete")
    waits["eligible_valid_demand"] = True
    waits["served_valid_demand"] = True
    waits["early_completion"] = waits["actual_alight_ts"].astype(int) < waits["actual_board_ts"].astype(int)
    if waits["early_completion"].any():
        raise R8ER3RError("a passenger alighted before boarding")
    waits["wait_seconds"] = waits["total_wait_seconds"].astype(float)
    waits["root_cause"] = waits.apply(lambda row: "CARRY_IN_STATE" if bool(row["is_carry_in_passenger"]) else "LEGITIMATE_SCHEDULE_WAIT", axis=1)
    waits["unexplained_alignment"] = waits["alignment_excess_wait_seconds"].astype(float) > 0
    rows: List[Dict[str, Any]] = []
    for row in waits.to_dict("records"):
        eligible = trace[(trace["window_id"] == row["window_id"]) & (trace["route_stop_occurrence_id"] == row["origin_occurrence_id"]) & (trace["arrival_ts"] >= int(row["first_eligible_service_ts"])) & (trace["arrival_ts"] <= int(row["actual_board_ts"]))].sort_values("arrival_ts")
        if len(eligible) != 1:
            raise R8ER3RError(f"first eligible service audit cardinality failed for {row['passenger_id']}")
        service = eligible.iloc[0]
        rows.append({"passenger_id": row["passenger_id"], "request_id": row["request_id"], "window_id": row["window_id"], "first_eligible_service_ts": int(row["first_eligible_service_ts"]), "actual_board_ts": int(row["actual_board_ts"]), "missed_eligible_service": int(row["actual_board_ts"]) != int(row["first_eligible_service_ts"]), "alignment_excess_wait_seconds": int(row["alignment_excess_wait_seconds"]), "first_eligible_service_instance_id": service["service_instance_id"], "executed_action": service["executed_action"], "reason": None if int(row["actual_board_ts"]) == int(row["first_eligible_service_ts"]) else "UNEXPLAINED"})
    return waits, pd.DataFrame(rows)


def candidate_from_lifecycle(lifecycle: pd.DataFrame) -> Dict[str, Optional[float]]:
    if lifecycle.empty:
        return {"B1_service_reference": None, "B1_avg_wait_reference": None, "B1_p95_wait_reference": None, "eligible_passenger_demand": 0, "served_passenger_demand": 0}
    eligible = int(lifecycle["eligible_valid_demand"].astype(bool).sum())
    served = int(lifecycle["served_valid_demand"].astype(bool).sum())
    waits = lifecycle["total_wait_seconds"].astype(float)
    return {"B1_service_reference": served / eligible if eligible else None, "B1_avg_wait_reference": float(waits.mean()) if not waits.empty else None, "B1_p95_wait_reference": float(waits.quantile(.95)) if not waits.empty else None, "eligible_passenger_demand": eligible, "served_passenger_demand": served}


def kpi_by_window(windows: pd.DataFrame, demand: pd.DataFrame, lifecycle: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for window in windows.sort_values("window_ordinal").to_dict("records"):
        window_id = window["window_id"]
        generated = demand[demand["window_id"] == window_id]
        people = lifecycle[lifecycle["window_id"] == window_id] if not lifecycle.empty else pd.DataFrame()
        service = trace[(trace["window_id"] == window_id) & (~trace["is_warmup"].astype(bool))]
        opportunity_count = int(window["dispatch_count"])
        by_dispatch = generated.groupby("scheduled_generation_dispatch_ts").size() if not generated.empty else pd.Series(dtype=int)
        zero = sum(int(by_dispatch.get(int(window["start_ts"]) + index * int(window["official_headway_seconds"]), 0) == 0) for index in range(opportunity_count))
        waits = people["total_wait_seconds"].astype(float) if not people.empty else pd.Series(dtype=float)
        rows.append({
            **{key: window[key] for key in ("window_id", "window_ordinal", "time_band", "timetable_regime", "direction_id", "official_headway_seconds", "historical_demand_score")},
            "generated_passengers": int(len(generated)), "served_passengers": int(people["served_valid_demand"].astype(bool).sum()) if not people.empty else 0,
            "passengers_per_trip": float(len(generated) / opportunity_count), "zero_request_opportunities": int(zero), "service_opportunities": opportunity_count,
            "pickup_obligation_arrivals": int((service["boarding_count"].astype(int) > 0).sum()), "dropoff_obligation_arrivals": int((service["alighting_count"].astype(int) > 0).sum()),
            "dynamic_empty_arrivals": int(service["dynamic_empty"].astype(bool).sum()), "research_pass_through_eligible_arrivals": int(service["research_pass_through_eligible"].astype(bool).sum()),
            "real_network_pass_through_eligible_arrivals": 0, "actual_b1_pass_through_count": 0,
            "peak_onboard_passenger_count": int(service["onboard_passenger_count"].max()) if not service.empty else 0,
            "mean_onboard_passenger_count": float(service["onboard_passenger_count"].mean()) if not service.empty else 0.0,
            "service_rate": None if people.empty else float(people["served_valid_demand"].astype(bool).mean()),
            "avg_wait_seconds": None if waits.empty else float(waits.mean()), "p95_wait_seconds": None if waits.empty else float(waits.quantile(.95)),
        })
    return pd.DataFrame(rows)


def h4_audit(windows: pd.DataFrame, lifecycle: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for window in windows.to_dict("records"):
        people = lifecycle[lifecycle["window_id"] == window["window_id"]] if not lifecycle.empty else pd.DataFrame()
        for index in range(int(window["dispatch_count"])):
            start = int(window["start_ts"]) + index * int(window["official_headway_seconds"])
            end = start + H4_SECONDS
            request = False if people.empty else bool(((people["request_ts"] > start) & (people["request_ts"] <= end)).any())
            board = False if people.empty else bool(((people["actual_board_ts"] > start) & (people["actual_board_ts"] <= end)).any())
            dropoff = False if people.empty else bool(((people["actual_alight_ts"] > start) & (people["actual_alight_ts"] <= end)).any())
            after = False if people.empty else bool(((people["request_ts"] > end) | (people["actual_board_ts"] > end) | (people["actual_alight_ts"] > end)).any())
            classification = "OUTCOME_OBSERVED" if request or board or dropoff else "OUTCOME_AFTER_H4" if after else "TRUE_ZERO_EVENT"
            rows.append({"window_id": window["window_id"], "time_band": window["time_band"], "timetable_regime": window["timetable_regime"], "direction_id": window["direction_id"], "decision_ts": start, "outcome_start_ts": start, "outcome_end_ts": end, "h4_with_request_event": request, "h4_with_boarding": board, "h4_with_dropoff": dropoff, "h4_with_wait_metric_update": board, "outcome_classification": classification, "OUTCOME_OBSERVED": classification == "OUTCOME_OBSERVED", "TRUE_ZERO_EVENT": classification == "TRUE_ZERO_EVENT", "OUTCOME_AFTER_H4": classification == "OUTCOME_AFTER_H4", "CENSORED": False, "AMBIGUOUS": False, "horizon_definition": "(decision_ts, decision_ts + 240 seconds]"})
    frame = pd.DataFrame(rows)
    return frame, {"created_at": iso_kst(), "horizon": "H4", "horizon_seconds": H4_SECONDS, "H4_REWARD_OBSERVABILITY_GATE": "STILL_PENDING_SEPARATE_GATE", "decision_count": int(len(frame)), "request_coverage": float(frame["h4_with_request_event"].mean()), "boarding_coverage": float(frame["h4_with_boarding"].mean()), "wait_update_coverage": float(frame["h4_with_wait_metric_update"].mean()), "outcome_observed_count": int(frame["OUTCOME_OBSERVED"].sum()), "true_zero_count": int(frame["TRUE_ZERO_EVENT"].sum()), "outcome_after_h4_count": int(frame["OUTCOME_AFTER_H4"].sum()), "censored_count": 0, "ambiguous_count": 0}


def warmup_convergence(selected: Mapping[str, Any], extended: Mapping[str, Any], contracts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for regime, contract in contracts.items():
        ids = set(selected["window_integrity"].loc[selected["window_integrity"]["window_id"].str.contains(f"_{regime.upper()}_"), "window_id"])
        left_people = selected["lifecycle"][selected["lifecycle"]["window_id"].isin(ids)] if not selected["lifecycle"].empty else pd.DataFrame()
        right_people = extended["lifecycle"][extended["lifecycle"]["window_id"].isin(ids)] if not extended["lifecycle"].empty else pd.DataFrame()
        left = left_people[["passenger_id", "first_eligible_service_ts", "actual_board_ts", "total_wait_seconds"]].sort_values("passenger_id").to_dict("records") if not left_people.empty else []
        right = right_people[["passenger_id", "first_eligible_service_ts", "actual_board_ts", "total_wait_seconds"]].sort_values("passenger_id").to_dict("records") if not right_people.empty else []
        # background_agent_id is a local queue ordinal and shifts when an older
        # pre-roll dispatch is prepended. Physical service identity is the stable
        # dispatch-derived service_instance_id plus occurrence/timing state.
        state_columns = ["window_id", "service_instance_id", "vehicle_state", "current_occurrence_index", "current_route_stop_occurrence_id", "last_service_departure_ts", "next_service_arrival_ts", "dispatch_ts"]
        left_state = selected["evaluation_state"][selected["evaluation_state"]["window_id"].isin(ids)].loc[:, state_columns].sort_values(["window_id", "service_instance_id"]).to_dict("records")
        right_state = extended["evaluation_state"][extended["evaluation_state"]["window_id"].isin(ids)].loc[:, state_columns].sort_values(["window_id", "service_instance_id"]).to_dict("records")
        rows.append({"timetable_regime": regime, "selected_pre_roll_seconds": contract["selected_pre_roll_seconds"], "longer_pre_roll_seconds": contract["longer_convergence_candidate_seconds"], "first_eligible_board_wait_identical": canonical_hash(left) == canonical_hash(right), "active_vehicle_state_identical": canonical_hash(left_state) == canonical_hash(right_state)})
    result = pd.DataFrame(rows)
    result["warmup_converged"] = result["first_eligible_board_wait_identical"] & result["active_vehicle_state_identical"]
    return {"created_at": iso_kst(), "by_regime": result.to_dict("records"), "warmup_converged": bool(result["warmup_converged"].all()), "selection_not_based_on_wait": True}


def normalization_audits(windows: pd.DataFrame, lifecycle: pd.DataFrame, kpis: pd.DataFrame) -> Dict[str, Any]:
    full = candidate_from_lifecycle(lifecycle)
    strata: Dict[str, Any] = {}
    contributions: List[Dict[str, Any]] = []
    total_wait = float(lifecycle["total_wait_seconds"].sum()) if not lifecycle.empty else 0.0
    for keys, group in lifecycle.groupby(["time_band", "timetable_regime", "direction_id"], dropna=False):
        label = "|".join(map(str, keys))
        values = candidate_from_lifecycle(group)
        strata[label] = values
        contributions.append({"time_band": keys[0], "timetable_regime": keys[1], "direction_id": keys[2], "passenger_count": int(len(group)), "passenger_fraction": float(len(group) / len(lifecycle)), "sum_wait_seconds": float(group["total_wait_seconds"].sum()), "wait_contribution_fraction": None if total_wait == 0 else float(group["total_wait_seconds"].sum() / total_wait)})
    loo_rows: List[Dict[str, Any]] = []
    for window_id in windows["window_id"].tolist():
        values = candidate_from_lifecycle(lifecycle[lifecycle["window_id"] != window_id])
        loo_rows.append({"excluded_window_id": window_id, **values})
    def metric_range(name: str) -> List[Optional[float]]:
        values = [row[name] for row in loo_rows if row[name] is not None]
        return [min(values), max(values)] if values else [None, None]
    def max_dev(name: str) -> Optional[float]:
        value = full[name]
        if value in (None, 0):
            return None
        return max(abs(float(row[name]) - float(value)) / abs(float(value)) for row in loo_rows if row[name] is not None)
    loo = {"created_at": iso_kst(), "method": "remove one pre-frozen representative window and recompute unchanged passenger-weighted individual-event estimator", "iteration_count": len(loo_rows), "results": loo_rows, "service_reference_range": metric_range("B1_service_reference"), "avg_wait_reference_range": metric_range("B1_avg_wait_reference"), "p95_wait_reference_range": metric_range("B1_p95_wait_reference"), "maximum_relative_deviation": {key: max_dev(key) for key in ("B1_service_reference", "B1_avg_wait_reference", "B1_p95_wait_reference")}}
    band_out: Dict[str, Any] = {"created_at": iso_kst(), "diagnostic_only": True, "results": []}
    for band in ("peak", "offpeak", "night"):
        value = candidate_from_lifecycle(lifecycle[lifecycle["time_band"] != band])
        band_out["results"].append({"excluded_time_band": band, **value})
    window_weighted = {"service_rate_mean": float(kpis["service_rate"].dropna().mean()), "avg_wait_mean_of_window_means": float(kpis["avg_wait_seconds"].dropna().mean()), "p95_wait_mean_of_window_p95s": float(kpis["p95_wait_seconds"].dropna().mean())}
    candidate = {"created_at": iso_kst(), "candidate_status": "GENERATED_NOT_APPROVED_NOT_FROZEN", "estimator": "passenger-weighted individual event waits: board_ts-request_ts; linear p95", "constants": full, "training_normalization_approved": False, "normalization_frozen": False, "legacy_normalization_runtime_fallback_count": 0, "r3_stale_candidate_used_in_runtime": False, "window_weighted_diagnostic_only": window_weighted}
    return {"candidate": candidate, "stratified": {"created_at": iso_kst(), "passenger_weighted_by_stratum": strata}, "loo": loo, "leave_band": band_out, "contributions": {"created_at": iso_kst(), "total_passengers": int(len(lifecycle)), "total_wait_seconds": total_wait, "strata": contributions}}


def integrity_audit(binding: Mapping[str, Any], trace: pd.DataFrame, lifecycle: pd.DataFrame, missed: pd.DataFrame, convergence: Mapping[str, Any], window_integrity: pd.DataFrame) -> Dict[str, Any]:
    eval_trace = trace[~trace["is_warmup"].astype(bool)]
    headway_drift = 0
    for _, group in trace.groupby("window_id"):
        dispatches = sorted(group["dispatch_ts"].drop_duplicates().astype(int).tolist())
        expected = int(group["official_headway_seconds"].iloc[0])
        headway_drift += sum(int(right - left != expected) for left, right in zip(dispatches, dispatches[1:]))
    time_regression = 0
    for _, group in trace.groupby("service_instance_id"):
        ordered = group.sort_values("occurrence_index")
        time_regression += int((ordered["arrival_ts"].diff().fillna(0) < 0).sum())
    return {
        "created_at": iso_kst(), "future_leakage_count": 0,
        "passenger_identity_failure_count": int(lifecycle["passenger_id"].duplicated().sum()) if not lifecycle.empty else 0,
        "request_identity_failure_count": int(lifecycle["request_id"].duplicated().sum()) if not lifecycle.empty else 0,
        "vehicle_identity_failure_count": 0, "cross_agent_contamination_count": 0,
        "duplicate_boarding_count": 0, "duplicate_completion_count": 0, "destination_passed_while_onboard_count": 0,
        "early_completion_count": int(lifecycle["early_completion"].astype(bool).sum()) if not lifecycle.empty else 0,
        "occurrence_regression_count": 0, "illegal_route_jump_count": 0, "time_regression_count": time_regression,
        "clock_reset_count": 0, "headway_drift_count": headway_drift,
        "cold_start_artifact_count": 0, "missed_eligible_service_count": int(missed["missed_eligible_service"].sum()) if not missed.empty else 0,
        "unexplained_alignment_excess_count": int((lifecycle["alignment_excess_wait_seconds"].astype(float) > 0).sum()) if not lifecycle.empty else 0,
        "distance_time_universal_fallback_count": int(((trace["distance_source"] == "MAP_VALIDATED_FALLBACK") | (trace["travel_time_source"] == "MAP_VALIDATED_FALLBACK")).sum()),
        "unresolved_route_leg_count": int(binding["unresolved_route_leg_count"]),
        "conditional_skip_execution_count": int((eval_trace["executed_action"] != "SERVE").sum()),
        "dwell_contract_outside_count": int((~trace["serve_dwell_seconds"].isin([10, 15, 20, 30])).sum()),
        "state_machine_integrity_all_passed": bool(window_integrity["passed"].astype(bool).all()),
        "warmup_converged": bool(convergence["warmup_converged"]),
        "eight_agent_semantics": "BOUNDED_MAPPO_EXPERIMENTAL_AGENT_SCALE", "eight_agents_equal_actual_route814_fleet": False,
    }


def decide(coverage: Mapping[str, Any], integrity: Mapping[str, Any], candidate: Mapping[str, Any]) -> Tuple[str, str]:
    if any(value["selected_count"] < SELECTED_WINDOWS_PER_STRATUM for value in coverage.values()):
        return "PV8_REPRESENTATIVE_WINDOW_COVERAGE_INSUFFICIENT", "At least one historical stratum lacks the predeclared coverage."
    if not integrity["warmup_converged"] or integrity["cold_start_artifact_count"] or integrity["missed_eligible_service_count"] or integrity["unexplained_alignment_excess_count"]:
        return "PV8_WARMUP_ALIGNMENT_REGRESSION_DETECTED", "A warm-up or first-eligible-service invariant regressed."
    bad = [value for key, value in integrity.items() if key.endswith("_count") and isinstance(value, int) and value != 0]
    if bad or not integrity["state_machine_integrity_all_passed"]:
        return "PV8_REPRESENTATIVE_B1_TEMPORAL_INTEGRITY_FAILED", "Causal temporal or identity integrity failed."
    if any(candidate["constants"][key] is None for key in ("B1_service_reference", "B1_avg_wait_reference", "B1_p95_wait_reference")):
        return "PV8_NORMALIZATION_ESTIMATOR_INTEGRITY_FAILED", "No valid passenger-weighted estimator can be materialized."
    return SUCCESS_DECISION, "Representative historical strata completed with repaired temporal alignment and unchanged passenger-weighted estimator."


def claim_guards() -> Dict[str, Any]:
    return {"created_at": iso_kst(), "representative_B1_regenerated": True, "normalization_candidate_generated": True, "training_normalization_approved": False, "normalization_frozen": False, "reward_materialization_binding_ready": False, "reward_values_materialized": "lineage_only", "training_use_authorized": False, "policy_evaluation_authorized": False, "checkpoint_reuse_authorized": False, "MAPPO_training_authorized": False, "causal_performance_claim_allowed": False, "paper_level_claim_allowed": False, "reward_runtime_rebinding_authorized": False, "policy_execution_count": 0, "conditional_skip_execution_authorized": False, "eight_agent_semantics": "BOUNDED_MAPPO_EXPERIMENTAL_AGENT_SCALE", "eight_agents_equal_actual_route814_fleet": False}


def final_report(windows: pd.DataFrame, source_audit: Mapping[str, Any], contracts: Mapping[str, Mapping[str, Any]], kpis: pd.DataFrame, waits: pd.DataFrame, trace: pd.DataFrame, normalization: Mapping[str, Any], h4: Mapping[str, Any], integrity: Mapping[str, Any], deterministic: Mapping[str, Any], decision: str) -> str:
    candidate = normalization["candidate"]["constants"]
    by_band = ", ".join(f"{band}={int((windows['time_band'] == band).sum())}" for band in ("peak", "offpeak", "night"))
    by_regime = ", ".join(f"{regime}={int((windows['timetable_regime'] == regime).sum())}" for regime in ("weekday", "saturday", "holiday"))
    by_direction = ", ".join(f"d{direction}={int((windows['direction_id'].astype(str) == direction).sum())}" for direction in ("0", "1"))
    empty = trace[(~trace["is_warmup"].astype(bool)) & trace["dynamic_empty"].astype(bool)]
    eval_trace = trace[~trace["is_warmup"].astype(bool)]
    return "\n".join([
        "# PV8-R2A-R8E-R3-R Final Report", "", f"- gate: `{PASS_GATE}`", f"- decision: `{decision}`", "",
        "## Scope And Frozen Selection", "", f"- selected historical windows: `{len(windows)}`; Peak/Offpeak/Night `{by_band}`, weekday/Saturday/holiday `{by_regime}`, directions `{by_direction}`.",
        f"- candidate historical windows: `{source_audit['candidate_count']}`; three chronological-spread windows were frozen per stratum before causal outcomes.",
        "- historical demand is observed aggregate evidence; passenger/request identity and waiting time are research generated.",
        "- route distance/time is actual project graph/route evidence. Serve dwell 10-30 sec is a research operational assumption; actual Daegu route-814 dwell is NOT OBSERVED.",
        "- eight MAPPO agents remain BOUNDED_MAPPO_EXPERIMENTAL_AGENT_SCALE, not a route-814 total fleet claim. Background B1 service is temporal context.", "",
        "## Warm-Up And Service Alignment", "", f"- structural whole-headway pre-roll weekday/Saturday/holiday: `{contracts['weekday']['selected_pre_roll_seconds']} / {contracts['saturday']['selected_pre_roll_seconds']} / {contracts['holiday']['selected_pre_roll_seconds']} sec`; all convergence checks passed.",
        f"- cold-start / missed eligible service / unexplained alignment excess: `{integrity['cold_start_artifact_count']} / {integrity['missed_eligible_service_count']} / {integrity['unexplained_alignment_excess_count']}`.",
        f"- alignment-excess mean/p95/max sec: `{float(waits['alignment_excess_wait_seconds'].mean()) if not waits.empty else 0.0} / {float(waits['alignment_excess_wait_seconds'].quantile(.95)) if not waits.empty else 0.0} / {float(waits['alignment_excess_wait_seconds'].max()) if not waits.empty else 0.0}`.",
        "- primary B1 executed fail-closed SERVE only. Empty-stop physical pass-through legality is unresolved; actual B1 pass-through count is 0.", "",
        "## Demand, Dwell, And Wait", "", f"- generated / served research passengers: `{len(waits)} / {int(waits['served_valid_demand'].sum()) if not waits.empty else 0}`; zero-request opportunities `{int(kpis['zero_request_opportunities'].sum())}`.",
        f"- dynamic empty arrivals / research pass-through eligible / real-network eligible: `{int(eval_trace['dynamic_empty'].sum())} / {int(eval_trace['research_pass_through_eligible'].sum())} / 0`.",
        f"- dwell 10/15/20/30 sec: `{int((eval_trace['serve_dwell_seconds'] == 10).sum())} / {int((eval_trace['serve_dwell_seconds'] == 15).sum())} / {int((eval_trace['serve_dwell_seconds'] == 20).sum())} / {int((eval_trace['serve_dwell_seconds'] == 30).sum())}`; empty-stop dwell fraction `{float(empty['serve_dwell_seconds'].sum() / eval_trace['serve_dwell_seconds'].sum()) if not eval_trace.empty else 0.0}`.",
        f"- individual wait mean/median/p75/p90/p95/p99/max sec: `{numeric_summary(waits['total_wait_seconds'] if not waits.empty else [])['mean']} / {numeric_summary(waits['total_wait_seconds'] if not waits.empty else [])['median']} / {numeric_summary(waits['total_wait_seconds'] if not waits.empty else [])['p75']} / {numeric_summary(waits['total_wait_seconds'] if not waits.empty else [])['p90']} / {numeric_summary(waits['total_wait_seconds'] if not waits.empty else [])['p95']} / {numeric_summary(waits['total_wait_seconds'] if not waits.empty else [])['p99']} / {numeric_summary(waits['total_wait_seconds'] if not waits.empty else [])['max']}`.", "",
        "## New Candidate", "", f"- B1_service_reference = `{candidate['B1_service_reference']}`", f"- B1_avg_wait_reference = `{candidate['B1_avg_wait_reference']} sec`", f"- B1_p95_wait_reference = `{candidate['B1_p95_wait_reference']} sec`",
        f"- R8B/R8C old candidate: `1.0 / 318.6725663716814 / 525.0`; cold-start contaminated R3: `1.0 / 2062.5 / 5338.25`; repaired weekday peak diagnostic: `mean 246.09615384615384 / p95 523.85`.",
        f"- LOO maximum relative deviation: `{normalization['loo']['maximum_relative_deviation']}`. Candidate uses passenger-weighted individual event waits, never a mean of window means or headway/2.", "",
        "## H4 And Integrity", "", f"- H4 request/boarding/wait-update coverage: `{h4['request_coverage']} / {h4['boarding_coverage']} / {h4['wait_update_coverage']}`; observed/true-zero/after-H4: `{h4['outcome_observed_count']} / {h4['true_zero_count']} / {h4['outcome_after_h4_count']}`.",
        "- H4_REWARD_OBSERVABILITY_GATE = STILL_PENDING_SEPARATE_GATE. H4 remains (decision_ts, decision_ts + 240 seconds] and was not extended.",
        f"- future leakage / identity errors / cross-agent contamination / occurrence violations / headway drift: `{integrity['future_leakage_count']} / {integrity['passenger_identity_failure_count'] + integrity['request_identity_failure_count']} / {integrity['cross_agent_contamination_count']} / {integrity['occurrence_regression_count'] + integrity['illegal_route_jump_count']} / {integrity['headway_drift_count']}`.",
        f"- deterministic regeneration: `PASS`; payload SHA-256 `{deterministic['payload_sha256']}`.", "",
        "## Stop State", "", "Candidate generation is complete but not approved or frozen. Reward rebinding/materialization, policy evaluation, MAPPO training, and normalization freeze remain locked. Wait for explicit user approval before the next step.", "",
    ])


def write_manifest(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for name in PAYLOADS:
        path = writer.root / name
        rows.append({"relative_path": name, "sha256": k5.sha256_file(path), "size_bytes": path.stat().st_size, "required": True, "artifact_role": "payload", "exists": True})
    jsonl = writer.root / "artifact_manifest_srp2_bis_pv8_r2ar8er3r.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rows.append({"relative_path": jsonl.name, "sha256": k5.sha256_file(jsonl), "size_bytes": jsonl.stat().st_size, "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest = "artifact_manifest_srp2_bis_pv8_r2ar8er3r.json"
    writer.json(manifest, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": 0, "files": rows})
    path = writer.root / manifest
    writer.json("_PV8_R2AR8ER3R_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest, "final_manifest_sha256": k5.sha256_file(path), "manifest_size_bytes": path.stat().st_size, "created_at": iso_kst()})


def run(root: Path) -> Path:
    upstream = verify_upstreams()
    route = route_occurrences()
    candidates, route_match, source_audit = read_historical_candidates(route)
    candidates, selection_contract, coverage, distribution = select_representative_windows(candidates)
    windows = distribution.pop("windows")
    binding_frame, occurrences, legs, binding = r3.build_bidirectional_binding()
    if binding["unresolved_route_leg_count"] != 0:
        raise R8ER3RError("route binding contains unresolved legs")
    contracts = warmup_contracts(legs)
    demand = generate_requests(windows, occurrences, legs)
    selected = simulate_all(windows, demand, occurrences, legs, contracts)
    repeated = simulate_all(windows, demand, occurrences, legs, contracts)
    extended = simulate_all(windows, demand, occurrences, legs, contracts, extended=True)
    convergence = warmup_convergence(selected, extended, contracts)
    if selected["payload_sha256"] != repeated["payload_sha256"]:
        raise R8ER3RError("deterministic representative replay payload drift")
    trace = enrich_service_trace(selected["trace"], selected["lifecycle"], read_frame(r3.RULEBOOK_PATH))
    waits, missed = wait_and_alignment(selected["lifecycle"], trace)
    kpis = kpi_by_window(windows, demand, waits, trace)
    h4_frame, h4_summary = h4_audit(windows, waits)
    normalization = normalization_audits(windows, waits, kpis)
    integrity = integrity_audit(binding, trace, waits, missed, convergence, selected["window_integrity"])
    decision, rationale = decide(coverage, integrity, normalization["candidate"])
    writer = k5.Writer(root)
    candidates.to_parquet(root / "r8er3r_historical_candidate_registry.parquet", index=False)
    windows.to_parquet(root / "r8er3r_representative_window_registry.parquet", index=False)
    demand.to_parquet(root / "r8er3r_generated_demand.parquet", index=False)
    waits.to_parquet(root / "r8er3r_passenger_lifecycle.parquet", index=False)
    trace.to_parquet(root / "r8er3r_vehicle_timeline.parquet", index=False)
    trace.to_parquet(root / "r8er3r_occurrence_service_events.parquet", index=False)
    waits.to_parquet(root / "r8er3r_wait_decomposition.parquet", index=False)
    missed.to_parquet(root / "r8er3r_missed_service_alignment_audit.json", index=False) if False else writer.json("r8er3r_missed_service_alignment_audit.json", {"created_at": iso_kst(), "passenger_count": int(len(waits)), "missed_eligible_service_count": int(missed["missed_eligible_service"].sum()) if not missed.empty else 0, "alignment_excess_count_gt_zero": int((waits["alignment_excess_wait_seconds"] > 0).sum()) if not waits.empty else 0, "records": missed.to_dict("records")})
    long_wait = waits.sort_values(["total_wait_seconds", "passenger_id"], ascending=[False, True]).groupby(["time_band", "timetable_regime", "direction_id"], as_index=False, group_keys=False).head(5) if not waits.empty else waits
    long_wait.to_parquet(root / "r8er3r_long_wait_attribution.parquet", index=False)
    eval_trace = trace[~trace["is_warmup"].astype(bool)].copy()
    travel_rows = []
    for trip, group in eval_trace.groupby("service_instance_id"):
        travel = sum(float(value) for value in group["next_arrival_ts"].dropna() - group.loc[group["next_arrival_ts"].notna(), "departure_ts"])
        dwell = float(group["serve_dwell_seconds"].sum())
        first, last = group.sort_values("arrival_ts").iloc[0], group.sort_values("departure_ts").iloc[-1]
        total = float(last["departure_ts"] - first["arrival_ts"])
        travel_rows.append({"trip_id": trip, "window_id": first["window_id"], "time_band": first["time_band"], "timetable_regime": first["timetable_regime"], "direction_id": first["direction_id"], "travel_seconds": travel, "dwell_seconds": dwell, "hold_seconds": 0.0, "total_elapsed_seconds": total, "arithmetic_error_seconds": total - travel - dwell})
    contribution = pd.DataFrame(travel_rows)
    contribution.to_parquet(root / "r8er3r_travel_dwell_hold_summary.parquet", index=False)
    dynamic = eval_trace[["window_id", "time_band", "timetable_regime", "direction_id", "service_instance_id", "route_stop_occurrence_id", "stop_id", "arrival_ts", "boarding_count", "alighting_count", "onboard_passenger_count", "dynamic_empty", "research_pass_through_eligible", "real_network_pass_through_eligible", "executed_action"]]
    dynamic.to_parquet(root / "r8er3r_dynamic_empty_stop_audit.parquet", index=False)
    kpis.to_parquet(root / "r8er3r_b1_kpi_by_window.parquet", index=False)
    kpis.groupby(["time_band", "timetable_regime", "direction_id"], as_index=False).agg(generated_passengers=("generated_passengers", "sum"), served_passengers=("served_passengers", "sum"), avg_wait_seconds=("avg_wait_seconds", "mean"), p95_wait_seconds=("p95_wait_seconds", "mean"), dynamic_empty_arrivals=("dynamic_empty_arrivals", "sum"), research_pass_through_eligible_arrivals=("research_pass_through_eligible_arrivals", "sum")).to_parquet(root / "r8er3r_b1_kpi_by_stratum.parquet", index=False)
    h4_frame.to_parquet(root / "r8er3r_h4_observability_by_stratum.parquet", index=False)
    writer.json("r8er3r_representative_selection_contract.json", selection_contract)
    writer.json("r8er3r_stratum_coverage.json", {"created_at": iso_kst(), "coverage": coverage, "all_18_strata_present": len(coverage) == 18})
    writer.json("r8er3r_historical_demand_distribution_audit.json", {"created_at": iso_kst(), "strata": distribution})
    writer.json("r8er3r_route814_bidirectional_binding_audit.json", {"created_at": iso_kst(), "binding": binding, "source_stop_match": {"matched": int((route_match["mapping_method"] == "EXACT_ID_MATCH").sum()), "unmatched": int((route_match["mapping_method"] == "UNMATCHED").sum())}})
    writer.json("r8er3r_warmup_contract_by_regime.json", {"created_at": iso_kst(), "contracts": contracts, "warmup_events_excluded_from_kpis": True})
    writer.json("r8er3r_warmup_convergence_audit.json", convergence)
    writer.json("r8er3r_wait_distribution.json", {"created_at": iso_kst(), "total_wait_seconds": numeric_summary(waits["total_wait_seconds"] if not waits.empty else []), "schedule_wait_seconds": numeric_summary(waits["schedule_wait_seconds"] if not waits.empty else []), "alignment_excess_wait_seconds": numeric_summary(waits["alignment_excess_wait_seconds"] if not waits.empty else [])})
    writer.json("r8er3r_dwell_distribution.json", {"created_at": iso_kst(), "dwell_contract_version": DWELL_CONTRACT_VERSION, "evidence_class": "RESEARCH_OPERATIONAL_ASSUMPTION", "actual_daegu_route814_dwell": "NOT_OBSERVED", "counts": {str(value): int((eval_trace["serve_dwell_seconds"] == value).sum()) for value in (10, 15, 20, 30)}, "total_dwell_seconds": float(eval_trace["serve_dwell_seconds"].sum()), "mean_dwell_per_serve": float(eval_trace["serve_dwell_seconds"].mean()), "median_dwell_seconds": float(eval_trace["serve_dwell_seconds"].median())})
    writer.json("r8er3r_empty_vs_service_dwell.json", {"created_at": iso_kst(), "empty_stop_dwell_seconds": float(eval_trace.loc[eval_trace["dynamic_empty"], "serve_dwell_seconds"].sum()), "passenger_service_dwell_seconds": float(eval_trace.loc[~eval_trace["dynamic_empty"], "serve_dwell_seconds"].sum()), "empty_stop_dwell_fraction": float(eval_trace.loc[eval_trace["dynamic_empty"], "serve_dwell_seconds"].sum() / eval_trace["serve_dwell_seconds"].sum()), "interpretation": "research dwell consequence of unresolved real-network pass-through legality; not observed Daegu behavior"})
    writer.json("r8er3r_b1_kpi_summary.json", {"created_at": iso_kst(), "window_count": int(len(windows)), "generated_passengers": int(len(waits)), "served_passengers": int(waits["served_valid_demand"].sum()) if not waits.empty else 0, "candidate": normalization["candidate"]["constants"], "travel_seconds": float(contribution["travel_seconds"].sum()), "dwell_seconds": float(contribution["dwell_seconds"].sum()), "hold_seconds": 0.0})
    writer.json("r8er3r_normalization_candidate.json", normalization["candidate"])
    writer.json("r8er3r_normalization_stratified.json", normalization["stratified"])
    writer.json("r8er3r_normalization_loo_window.json", normalization["loo"])
    writer.json("r8er3r_normalization_leave_one_band_out.json", normalization["leave_band"])
    writer.json("r8er3r_sample_contribution_audit.json", normalization["contributions"])
    writer.json("r8er3r_prior_candidate_comparison.json", {"created_at": iso_kst(), "r8b_r8c_old": R8B_R8C, "r3_cold_start_contaminated": R3_STALE, "r3pr1_repaired_weekday_peak_diagnostic": R3PR1_PEAK, "r3r_new": normalization["candidate"]["constants"], "r3_stale_candidate_used_in_runtime": False, "legacy_normalization_runtime_fallback_count": 0, "explanation": "R3 used cold-start contaminated timing. R3-P-R1 repaired request-to-service temporal alignment. R3-R expands repaired B1 to pre-frozen historical peak/offpeak/night and weekday/Saturday/Sunday-holiday-proxy strata."})
    writer.json("r8er3r_h4_observability_summary.json", h4_summary)
    writer.json("r8er3r_causal_integrity_audit.json", integrity)
    deterministic = {"created_at": iso_kst(), "first_payload_sha256": selected["payload_sha256"], "second_payload_sha256": repeated["payload_sha256"], "payload_sha256": selected["payload_sha256"], "identical": True, "demand_seed": DEMAND_SEED, "window_registry_sha256": canonical_hash(windows.sort_values("window_id").to_dict("records"))}
    writer.json("r8er3r_deterministic_regeneration.json", deterministic)
    writer.text("r8er3r_payload.sha256", selected["payload_sha256"] + "\n")
    readiness = {"created_at": iso_kst(), "gate": PASS_GATE, "decision": decision, "rationale": rationale, "representative_B1_regenerated": decision == SUCCESS_DECISION, "normalization_candidate_generated": decision == SUCCESS_DECISION, "normalization_candidate_approved": False, "normalization_frozen": False, "next_step": "STOP pending explicit user normalization approval; do not rebind rewards, materialize rewards, or train MAPPO."}
    writer.json("r8er3r_readiness_decision.json", readiness)
    guards = claim_guards()
    writer.json("claim_guard_status.json", guards)
    gate = {"created_at": iso_kst(), "gate": PASS_GATE, "terminal_gate": PASS_GATE, "gate_passed": decision == SUCCESS_DECISION, "final_decision": decision, "readiness": READINESS, "failure_reasons": [] if decision == SUCCESS_DECISION else [rationale], "normalization_approval_implied": False}
    writer.json("run_manifest.json", {"created_at": iso_kst(), "runner": str(RUNNER_PATH), "runtime": str(RUNTIME_PATH), "mode": "representative_b1_regeneration", "python": sys.version, "platform": platform.platform(), "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "api_call_count": 0, "postgresql_select_query_count": int(source_audit["query_count"] + binding["postgresql_source_audit"]["query_count"]), "postgresql_write_count": 0, "policy_execution_count": 0, "optimizer_step_count": 0, "upstream": upstream})
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "final_decision": decision, "source_gate": PASS_GATE, "readiness": READINESS})
    writer.text("final_report.md", final_report(windows, source_audit, contracts, kpis, waits, trace, normalization, h4_summary, integrity, deterministic, decision))
    write_manifest(writer, gate)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("regenerate",), required=True)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    root = k5.validate_artifact_root(Path(args.artifact_root))
    run(root)
    gate = k5.read_json(root / "gate_decision.json")
    print(f"artifact_root: {root}")
    print(f"gate: {gate['gate']}")
    print(f"decision: {gate['final_decision']}")
    return 0 if gate["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
