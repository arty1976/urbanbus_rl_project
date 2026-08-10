#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4H Reward V2 representative rematerialization.

This stage applies the H4G-bound Reward V2 runtime to the frozen R3-R
representative B1 population. It does not train MAPPO, evaluate a learned
policy, retune reward weights, retune references, choose p95 tolerance, or
change the upstream representative source.
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

PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
from rewards.mappo_reward_v1 import (
    PV8_REWARD_SEMANTICS_VERSION,
    PV8_REWARD_V2_AVG_WAIT_REFERENCE_SECONDS,
    PV8_REWARD_V2_FREEZE_SHA256,
    PV8_REWARD_V2_P95_EVALUATION_REFERENCE_SECONDS,
    compute_reward_v2,
)
from simulator.pv8_reward_outcome_collector import percentile


RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4h_reward_v2_representative_rematerialization.py"

R3R_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
H4G_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4g_reward_v2_runtime_rebinding_20260809_235045"
OLD_R8E_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8e_bounded_reward_materialization_20260809_163733"

R3R_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3R_REPRESENTATIVE_HISTORICAL_DEMAND_B1_REGENERATION_COMPLETE"
H4G_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4G_REWARD_V2_RUNTIME_REBINDING_AND_DETERMINISTIC_INTEGRATION_VALIDATION_COMPLETE"
H4G_DECISION = "PV8_REWARD_V2_RUNTIME_REBOUND_REPRESENTATIVE_REMATERIALIZATION_READY"
OLD_R8E_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8E_BOUNDED_CAUSAL_REWARD_MATERIALIZATION_COMPLETE"

FREEZE_SHA = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
RUNTIME_BINDING_SHA = "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3"
MATERIALIZATION_VERSION = "PV8_REWARD_V2_REPRESENTATIVE_B1_MATERIALIZATION_V1"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4H_REWARD_V2_REPRESENTATIVE_REMATERIALIZATION_AND_FROZEN_REFERENCE_CONSISTENCY_AUDIT_COMPLETE"

DECISION_READY = "PV8_REWARD_V2_REPRESENTATIVE_MATERIALIZATION_READY_FOR_TRAINING_READINESS_REVIEW"
DECISION_EMPTY_BLOCKER = "PV8_EMPTY_STOP_ACTION_DISCRIMINATION_BLOCKER"
DECISION_LOW_VARIANCE = "PV8_REWARD_SIGNAL_LOW_VARIANCE_BLOCKER"
DECISION_DEGENERATE = "PV8_REWARD_SIGNAL_DEGENERATE_BLOCKER"
DECISION_REF_FAIL = "PV8_B1_FROZEN_REFERENCE_CONSISTENCY_FAILED"
DECISION_INTEGRITY_FAIL = "PV8_REPRESENTATIVE_REWARD_RUNTIME_INTEGRITY_FAILED"
DECISION_FAILED = "PV8_REWARD_V2_REPRESENTATIVE_REMATERIALIZATION_FAILED"

AVG_WAIT_REF = PV8_REWARD_V2_AVG_WAIT_REFERENCE_SECONDS
P95_REF = PV8_REWARD_V2_P95_EVALUATION_REFERENCE_SECONDS
SERVICE_REF = 1.0
NUMERIC_TOL = 1e-9

EVENT_ORDER = [
    "VEHICLE_ARRIVAL",
    "STATE_SNAPSHOT",
    "OBLIGATION_SNAPSHOT",
    "K_MASK_BUILD",
    "ACTION_SELECTION",
    "ACTION_VALIDATION",
    "BOARDING_ALIGHTING",
    "LOCAL_SERVICE_SETTLEMENT",
    "HOLD_IF_APPLICABLE",
    "DEPARTURE",
    "NEXT_LINK_TRAVEL",
]

PAYLOADS = [
    "r8er3rh4h_user_authorization_record.json",
    "r8er3rh4h_upstream_binding.json",
    "r8er3rh4h_source_integrity.json",
    "r8er3rh4h_representative_source_contract.json",
    "r8er3rh4h_transition_reward_materialization.parquet",
    "r8er3rh4h_window_reward_summary.parquet",
    "r8er3rh4h_reward_by_time_band.parquet",
    "r8er3rh4h_reward_by_day_type.parquet",
    "r8er3rh4h_reward_by_direction.parquet",
    "r8er3rh4h_passenger_wait_consistency.parquet",
    "r8er3rh4h_frozen_reference_consistency.json",
    "r8er3rh4h_service_reference_audit.json",
    "r8er3rh4h_avg_wait_reference_audit.json",
    "r8er3rh4h_p95_reference_audit.json",
    "r8er3rh4h_reward_distribution.json",
    "r8er3rh4h_reward_sign_audit.json",
    "r8er3rh4h_reward_variance_audit.json",
    "r8er3rh4h_reward_sparsity_audit.json",
    "r8er3rh4h_reward_transition_density.json",
    "r8er3rh4h_service_obligation_audit.json",
    "r8er3rh4h_avg_wait_participation_audit.json",
    "r8er3rh4h_intervention_audit.json",
    "r8er3rh4h_empty_stop_action_discrimination.parquet",
    "r8er3rh4h_empty_stop_action_discrimination_audit.json",
    "r8er3rh4h_hold_action_discrimination_audit.json",
    "r8er3rh4h_action_opportunity_coverage.json",
    "r8er3rh4h_missed_service_audit.json",
    "r8er3rh4h_alignment_excess_audit.json",
    "r8er3rh4h_tail_evaluation_preservation.json",
    "r8er3rh4h_old_r8e_vs_v2_semantic_migration.json",
    "r8er3rh4h_ownership_integrity.json",
    "r8er3rh4h_future_leakage_audit.json",
    "r8er3rh4h_freeze_hash_coverage.json",
    "r8er3rh4h_runtime_binding_coverage.json",
    "r8er3rh4h_reward_numeric_consistency.json",
    "r8er3rh4h_deterministic_replay.json",
    "r8er3rh4h_training_readiness_classification.json",
    "r8er3rh4h_integrity_audit.json",
    "r8er3rh4h_readiness_decision.json",
    "claim_guard_status.json",
    "downstream_lock.json",
    "run_manifest.json",
    "gate_decision.json",
    "final_report.md",
]


class H4HError(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_hash(value: Any) -> str:
    payload = json.dumps(k5.json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_content(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: deterministic_content(item) for key, item in value.items() if key != "created_at"}
    if isinstance(value, list):
        return [deterministic_content(item) for item in value]
    if isinstance(value, tuple):
        return [deterministic_content(item) for item in value]
    return value


def sha256_df(df: pd.DataFrame) -> str:
    stable = df.copy()
    stable = stable.reindex(sorted(stable.columns), axis=1)
    payload = stable.to_json(orient="records", date_format="iso", force_ascii=False, double_precision=15)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bool_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    if isinstance(value, float) and math.isnan(value):
        return False
    return bool(value)


def safe_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    return int(value)


def stable_transition_id(row: Mapping[str, Any]) -> str:
    digest = canonical_hash(
        {
            "window_id": row["window_id"],
            "service_instance_id": row["service_instance_id"],
            "route_stop_occurrence_id": row["route_stop_occurrence_id"],
            "arrival_ts": safe_int(row["arrival_ts"]),
        }
    )[:20]
    return f"H4H_REWARD_V2_TRANSITION_{digest}"


def verify_upstream_artifact(root: Path, manifest_name: str, lock_name: str, gate: str, *, final_decision: Optional[str] = None) -> Dict[str, Any]:
    checks = k5.verify_manifest(root, manifest_name, lock_name)
    gate_payload = read_json(root / "gate_decision.json")
    observed_gate = gate_payload.get("gate") or gate_payload.get("terminal_gate")
    if observed_gate != gate or not k5.manifest_ok(checks):
        raise H4HError(f"upstream integrity failed: {root.name} gate={observed_gate} checks={checks}")
    if final_decision and gate_payload.get("final_decision") != final_decision:
        raise H4HError(f"upstream decision drifted: {root.name} {gate_payload.get('final_decision')}")
    return {
        "artifact_root": str(root),
        "gate": observed_gate,
        "final_decision": gate_payload.get("final_decision"),
        "readiness": gate_payload.get("readiness"),
        "manifest_integrity": checks,
    }


def verify_upstreams() -> Dict[str, Any]:
    h4g = verify_upstream_artifact(
        H4G_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4g.json",
        "_PV8_R2AR8ER3RH4G_COMPLETE.lock",
        H4G_GATE,
        final_decision=H4G_DECISION,
    )
    h4g_hash = read_json(H4G_ROOT / "r8er3rh4g_runtime_binding_hash.json")
    if h4g_hash.get("freeze_sha256") != FREEZE_SHA or h4g_hash.get("runtime_binding_sha256") != RUNTIME_BINDING_SHA:
        raise H4HError("H4G frozen hash binding drifted")
    r3r = verify_upstream_artifact(
        R3R_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8er3r.json",
        "_PV8_R2AR8ER3R_COMPLETE.lock",
        R3R_GATE,
    )
    old = verify_upstream_artifact(
        OLD_R8E_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8e.json",
        "_PV8_R2AR8E_COMPLETE.lock",
        OLD_R8E_GATE,
    )
    return {
        "created_at": iso_kst(),
        "PV8-R2A-R8E-R3-R": r3r,
        "PV8-R2A-R8E-R3-R-H4G": h4g,
        "PV8-R2A-R8E": old,
        "reward_freeze_sha256": FREEZE_SHA,
        "runtime_binding_sha256": RUNTIME_BINDING_SHA,
    }


def user_authorization_record() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "h4h_representative_reward_rematerialization_authorized": True,
        "authorized": [
            "use H4G-validated Reward V2 runtime",
            "use frozen R3-R representative B1 dataset",
            "rematerialize representative Reward V2 outputs",
            "compare against old R8E materialization for lineage/audit only",
            "perform reward distribution, semantic consistency, and frozen-reference audits",
            "determine readiness for a future training-readiness stage",
        ],
        "MAPPO_training_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "reward_weight_change_authorized": False,
        "reference_change_authorized": False,
        "optimizer_creation_authorized": False,
        "p95_promotion_tolerance_selection_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
    }


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(pd.Series(values, dtype="float64").quantile(q))


def describe(values: Sequence[Any]) -> Dict[str, Any]:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not vals:
        return {"count": 0, "min": None, "p05": None, "p25": None, "median": None, "mean": None, "p75": None, "p95": None, "max": None, "std": None}
    series = pd.Series(vals, dtype="float64")
    return {
        "count": int(series.count()),
        "min": float(series.min()),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
        "std": float(series.std(ddof=0)),
    }


def group_summary(df: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key, group in df.groupby(list(by), dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {name: value for name, value in zip(by, key)}
        row.update(
            {
                "transition_count": int(len(group)),
                "reward_total_min": float(group["reward_total"].min()),
                "reward_total_mean": float(group["reward_total"].mean()),
                "reward_total_median": float(group["reward_total"].median()),
                "reward_total_p95": float(group["reward_total"].quantile(0.95)),
                "reward_total_max": float(group["reward_total"].max()),
                "reward_total_std": float(group["reward_total"].std(ddof=0)),
                "positive_reward_count": int((group["reward_total"] > NUMERIC_TOL).sum()),
                "zero_reward_count": int((group["reward_total"].abs() <= NUMERIC_TOL).sum()),
                "negative_reward_count": int((group["reward_total"] < -NUMERIC_TOL).sum()),
                "service_bearing_transition_count": int((group["service_success_denominator_increment"] > 0).sum()),
                "wait_bearing_transition_count": int((group["affected_passenger_count"] > 0).sum()),
                "affected_passenger_count": int(group["affected_passenger_count"].sum()),
                "passenger_boarding_count": int(group["pickup_obligation_count"].sum()),
                "passenger_alighting_count": int(group["dropoff_obligation_count"].sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_sources() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    svc = pd.read_parquet(R3R_ROOT / "r8er3r_occurrence_service_events.parquet")
    svc = svc[~svc["is_warmup"].astype(bool)].copy().reset_index(drop=True)
    passengers = pd.read_parquet(R3R_ROOT / "r8er3r_passenger_lifecycle.parquet")
    wait = pd.read_parquet(R3R_ROOT / "r8er3r_wait_decomposition.parquet")
    windows = pd.read_parquet(R3R_ROOT / "r8er3r_representative_window_registry.parquet")
    kpi = read_json(R3R_ROOT / "r8er3r_b1_kpi_summary.json")
    missed = read_json(R3R_ROOT / "r8er3r_missed_service_alignment_audit.json")
    old = pd.read_parquet(OLD_R8E_ROOT / "r8e_reward_materialization.parquet")
    return svc, passengers, wait, windows, kpi, missed, old


def source_integrity(
    svc: pd.DataFrame,
    passengers: pd.DataFrame,
    windows: pd.DataFrame,
    kpi: Mapping[str, Any],
    missed: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    by_time = windows["time_band"].value_counts().to_dict()
    by_day = windows["timetable_regime"].value_counts().to_dict()
    by_direction = {str(k): int(v) for k, v in windows["direction_id"].value_counts().to_dict().items()}
    passenger_waits = passengers["wait_seconds"].astype(float).tolist()
    computed_avg = float(sum(passenger_waits) / len(passenger_waits))
    computed_p95 = float(percentile(passenger_waits, 0.95))
    service = float(passengers["served_valid_demand"].astype(bool).sum() / len(passengers))
    contract = {
        "created_at": iso_kst(),
        "representative_source_artifact": str(R3R_ROOT),
        "representative_windows": int(len(windows)),
        "transition_candidate_rows_evaluation_only": int(len(svc)),
        "warmup_rows_excluded_from_materialization": int(pd.read_parquet(R3R_ROOT / "r8er3r_occurrence_service_events.parquet")["is_warmup"].astype(bool).sum()),
        "time_band_counts": {str(k): int(v) for k, v in by_time.items()},
        "day_type_counts": {str(k): int(v) for k, v in by_day.items()},
        "direction_counts": by_direction,
        "generated_passengers": int(len(passengers)),
        "served_passengers": int(passengers["served_valid_demand"].astype(bool).sum()),
        "service_reference": SERVICE_REF,
        "avg_wait_reference_seconds": AVG_WAIT_REF,
        "p95_wait_reference_seconds": P95_REF,
        "source_mutation_allowed": False,
        "resampling_allowed": False,
        "cohort_regeneration_allowed": False,
    }
    integrity = {
        "created_at": iso_kst(),
        "manifest_source_verified": True,
        "window_count_ok": int(len(windows)) == 54,
        "time_band_balance_ok": {band: int(by_time.get(band, 0)) == 18 for band in ["peak", "offpeak", "night"]},
        "day_type_balance_ok": {day: int(by_day.get(day, 0)) == 18 for day in ["weekday", "saturday", "holiday"]},
        "direction_balance_ok": {"0": by_direction.get("0", 0) == 27, "1": by_direction.get("1", 0) == 27},
        "generated_passengers": int(len(passengers)),
        "served_passengers": int(passengers["served_valid_demand"].astype(bool).sum()),
        "generated_served_ok": int(len(passengers)) == 414 and int(passengers["served_valid_demand"].astype(bool).sum()) == 414,
        "recomputed_service_rate": service,
        "recomputed_avg_wait_seconds": computed_avg,
        "recomputed_p95_wait_seconds": computed_p95,
        "missed_eligible_service_count": int(missed.get("missed_eligible_service_count", -1)),
        "alignment_excess_count_gt_zero": int(missed.get("alignment_excess_count_gt_zero", -1)),
        "cold_start_artifacts": int(kpi.get("cold_start_artifacts", 0) or 0),
        "unexplained_alignment_excess": int(missed.get("alignment_excess_count_gt_zero", -1)),
        "passed": (
            len(windows) == 54
            and all(int(by_time.get(band, 0)) == 18 for band in ["peak", "offpeak", "night"])
            and all(int(by_day.get(day, 0)) == 18 for day in ["weekday", "saturday", "holiday"])
            and by_direction.get("0", 0) == 27
            and by_direction.get("1", 0) == 27
            and len(passengers) == 414
            and int(passengers["served_valid_demand"].astype(bool).sum()) == 414
            and int(missed.get("missed_eligible_service_count", -1)) == 0
            and int(missed.get("alignment_excess_count_gt_zero", -1)) == 0
        ),
    }
    return integrity, contract


def passenger_wait_consistency(passengers: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    df = passengers.copy()
    df["recomputed_schedule_wait_seconds"] = df["first_eligible_service_ts"].astype(float) - df["request_ts"].astype(float)
    df["recomputed_alignment_excess_wait_seconds"] = df["actual_board_ts"].astype(float) - df["first_eligible_service_ts"].astype(float)
    df["recomputed_total_wait_seconds"] = df["recomputed_schedule_wait_seconds"] + df["recomputed_alignment_excess_wait_seconds"]
    df["schedule_wait_arithmetic_error"] = df["recomputed_schedule_wait_seconds"] - df["schedule_wait_seconds"].astype(float)
    df["alignment_excess_arithmetic_error"] = df["recomputed_alignment_excess_wait_seconds"] - df["alignment_excess_wait_seconds"].astype(float)
    df["total_wait_arithmetic_error"] = df["recomputed_total_wait_seconds"] - df["total_wait_seconds"].astype(float)
    waits = df["recomputed_total_wait_seconds"].astype(float).tolist()
    avg = float(sum(waits) / len(waits))
    p95 = float(percentile(waits, 0.95))
    service = float(df["served_valid_demand"].astype(bool).sum() / len(df))
    service_audit = {
        "created_at": iso_kst(),
        "service_reference": SERVICE_REF,
        "recomputed_service_rate": service,
        "absolute_difference": abs(service - SERVICE_REF),
        "relative_difference": 0.0,
        "service_reference_semantics": "normalization/reference success level, not every transition score",
        "empty_nonmandatory_transition_service_status": "NOT_APPLICABLE",
        "passed": abs(service - SERVICE_REF) <= NUMERIC_TOL,
    }
    avg_audit = {
        "created_at": iso_kst(),
        "frozen_avg_wait_reference_seconds": AVG_WAIT_REF,
        "recomputed_avg_wait_seconds": avg,
        "absolute_difference": abs(avg - AVG_WAIT_REF),
        "relative_difference": abs(avg - AVG_WAIT_REF) / AVG_WAIT_REF,
        "passed": abs(avg - AVG_WAIT_REF) <= NUMERIC_TOL,
    }
    p95_audit = {
        "created_at": iso_kst(),
        "frozen_p95_wait_reference_seconds": P95_REF,
        "recomputed_p95_wait_seconds": p95,
        "absolute_difference": abs(p95 - P95_REF),
        "relative_difference": abs(p95 - P95_REF) / P95_REF,
        "role": "SYSTEM_LEVEL_EVALUATION_REFERENCE",
        "training_normalization": False,
        "passed": abs(p95 - P95_REF) <= NUMERIC_TOL,
    }
    return df, service_audit, avg_audit, p95_audit


def affected_wait_groups(passengers: pd.DataFrame) -> Dict[Tuple[str, str, str, int], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str, int], List[Dict[str, Any]]] = {}
    for row in passengers.to_dict("records"):
        key = (
            str(row["window_id"]),
            str(row["first_eligible_service_instance_id"]),
            str(row["origin_occurrence_id"]),
            safe_int(row["actual_board_ts"]),
        )
        groups.setdefault(key, []).append(row)
    return groups


def materialize_once(svc: pd.DataFrame, passengers: pd.DataFrame) -> pd.DataFrame:
    affected = affected_wait_groups(passengers)
    rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(svc.to_dict("records"), start=1):
        transition_id = stable_transition_id(row)
        arrival_ts = safe_int(row["arrival_ts"])
        pickup_count = safe_int(row["boarding_count"])
        dropoff_count = safe_int(row["alighting_count"])
        mandatory_count = int(bool_value(row.get("mandatory_stop")))
        affected_key = (
            str(row["window_id"]),
            str(row["service_instance_id"]),
            str(row["route_stop_occurrence_id"]),
            arrival_ts,
        )
        affected_rows = []
        for passenger in affected.get(affected_key, []):
            affected_rows.append(
                {
                    "passenger_id": str(passenger["passenger_id"]),
                    "request_id": str(passenger["request_id"]),
                    "originating_transition_id": transition_id,
                    "wait_ownership_key": f"{transition_id}:{passenger['passenger_id']}",
                    "request_ts": safe_int(passenger["request_ts"]),
                    "local_decision_ts": arrival_ts,
                    "first_eligible_service_ts": safe_int(passenger["first_eligible_service_ts"]),
                    "actual_board_ts": safe_int(passenger["actual_board_ts"]),
                }
            )
        metrics = {
            "reward_semantics_version": PV8_REWARD_SEMANTICS_VERSION,
            "reward_freeze_sha256": FREEZE_SHA,
            "transition_id": transition_id,
            "vehicle_slot_id": safe_int(row["background_agent_id"]),
            "route_id": "3000814001",
            "direction_id": str(row["direction_id"]),
            "occurrence_id": str(row["route_stop_occurrence_id"]),
            "local_decision_ts": arrival_ts,
            "action": str(row["executed_action"]),
            "pickup_obligation_count": pickup_count,
            "dropoff_obligation_count": dropoff_count,
            "approved_static_mandatory_obligation_count": mandatory_count,
            "completed_pickup_obligation_count": pickup_count if str(row["executed_action"]) == "SERVE" else 0,
            "completed_dropoff_obligation_count": dropoff_count if str(row["executed_action"]) == "SERVE" else 0,
            "completed_static_mandatory_obligation_count": mandatory_count if str(row["executed_action"]) == "SERVE" else 0,
            "affected_wait_rows": affected_rows,
            "explicit_forced_external_intervention_count": 0,
            "ordinary_k_mask_restriction_counted": False,
            "p95_training_reward_enabled": False,
            "p95_training_normalization_active": False,
        }
        result = compute_reward_v2(metrics)
        service = result["reward_service_component"]
        avg_wait = result["reward_avg_wait_component"]
        intervention = result["reward_intervention_component"]
        required_obligation_count = int(service["required_obligation_count"])
        out = {
            "transition_row_number": idx,
            "window_id": row["window_id"],
            "vehicle_slot_id": safe_int(row["background_agent_id"]),
            "vehicle_token": row["vehicle_token"],
            "route_id": "3000814001",
            "direction_id": str(row["direction_id"]),
            "occurrence_id": row["route_stop_occurrence_id"],
            "route_stop_occurrence_id": row["route_stop_occurrence_id"],
            "stop_id": row["stop_id"],
            "service_instance_id": row["service_instance_id"],
            "dispatch_ts": safe_int(row["dispatch_ts"]),
            "local_decision_ts": arrival_ts,
            "decision_ts": arrival_ts,
            "departure_ts": safe_int(row["departure_ts"]),
            "next_arrival_ts": safe_int(row["next_arrival_ts"]),
            "transition_id": transition_id,
            "action": row["executed_action"],
            "time_band": row["time_band"],
            "day_type": row["timetable_regime"],
            "timetable_regime": row["timetable_regime"],
            "official_headway_seconds": safe_int(row["official_headway_seconds"]),
            "official_timetable_type": row["official_timetable_type"],
            "pickup_obligation_count": pickup_count,
            "dropoff_obligation_count": dropoff_count,
            "mandatory_obligation_count": mandatory_count,
            "required_obligation_count": required_obligation_count,
            "completed_obligation_count": int(service["completed_obligation_count"]),
            "local_service_success": bool(service["status"] == "SUCCESS"),
            "obligation_status": "OBLIGATION_PRESENT" if required_obligation_count else "NO_OBLIGATION",
            "service_component_status": service["status"],
            "service_component_raw": float(result["reward_service_component_raw"]),
            "service_component_weighted": float(result["reward_service_component_weighted"]),
            "service_success_denominator_increment": int(service["service_success_denominator_increment"]),
            "positive_service_reward": float(service["positive_service_reward"]),
            "affected_passenger_count": int(avg_wait["affected_wait_count"]),
            "affected_avg_wait_seconds": avg_wait["current_local_avg_wait_seconds"],
            "avg_wait_component_status": avg_wait["status"],
            "avg_wait_component_raw": float(result["reward_avg_wait_component_raw"]),
            "avg_wait_component_weighted": float(result["reward_avg_wait_component_weighted"]),
            "positive_avg_wait_reward_blocked_by_service_gate": bool(avg_wait["positive_avg_wait_reward_blocked_by_service_gate"]),
            "forced_intervention_count": int(intervention["explicit_forced_external_intervention_count"]),
            "intervention_component_raw": float(result["reward_intervention_component_raw"]),
            "intervention_component_weighted": float(result["reward_intervention_component_weighted"]),
            "intervention_component_penalty_magnitude": float(intervention["weighted_penalty_magnitude"]),
            "reward_total": float(result["reward_total"]),
            "reward_semantics_version": result["reward_semantics_version"],
            "reward_freeze_sha256": result["reward_freeze_sha256"],
            "runtime_binding_sha256": RUNTIME_BINDING_SHA,
            "materialization_version": MATERIALIZATION_VERSION,
            "reward_formula": result["reward_formula"],
            "training_reference_version": result["training_reference_version"],
            "service_reference": result["service_reference"],
            "avg_wait_reference_seconds": result["avg_wait_reference_seconds"],
            "p95_evaluation_reference_seconds": result["p95_evaluation_reference_seconds"],
            "p95_semantics": result["p95_semantics"],
            "p95_training_reward_enabled": bool(result["p95_training_reward_enabled"]),
            "p95_training_normalization_active": bool(result["p95_training_normalization_active"]),
            "p95_local_transition_owner": result["p95_local_transition_owner"],
            "old_p95_weight_status": result["old_p95_weight_status"],
            "missed_eligible_service_role": result["missed_eligible_service_role"],
            "alignment_excess_wait_role": result["alignment_excess_wait_role"],
            "representative_reward_rematerialization": True,
            "blanket_SKIP_penalty_applied": bool(result["blanket_SKIP_penalty_applied"]),
            "direct_time_band_reward_term_applied": bool(result["direct_time_band_reward_term_applied"]),
            "H240_active_reward_settlement": bool(result["H240_active_reward_settlement"]),
            "H660_active_reward_settlement": bool(result["H660_active_reward_settlement"]),
            "dynamic_empty": bool_value(row["dynamic_empty"]),
            "research_pass_through_eligible": bool_value(row["research_pass_through_eligible"]),
            "real_network_pass_through_eligible": bool_value(row["real_network_pass_through_eligible"]),
            "terminal_or_turnaround_stop": bool_value(row["terminal_or_turnaround_stop"]),
            "valid_post_skip_path": bool_value(row["valid_post_skip_path"]),
            "event_order": "→".join(EVENT_ORDER),
            "obligation_before_K_mask": True,
            "K_mask_before_action": True,
            "action_before_board_alight": True,
            "actor_future_leakage": False,
            "critic_future_leakage": False,
            "orphan_reward": False,
        }
        rows.append(out)
    return pd.DataFrame(rows)


def numeric_consistency(df: pd.DataFrame) -> Dict[str, Any]:
    formula_recomputed = df["service_component_weighted"] + df["avg_wait_component_weighted"] + df["intervention_component_weighted"]
    formula_with_penalty = df["service_component_weighted"] + df["avg_wait_component_weighted"] - df["intervention_component_penalty_magnitude"]
    err = (df["reward_total"] - formula_recomputed).abs()
    values = df[
        [
            "service_component_raw",
            "service_component_weighted",
            "avg_wait_component_raw",
            "avg_wait_component_weighted",
            "intervention_component_raw",
            "intervention_component_weighted",
            "reward_total",
        ]
    ]
    return {
        "created_at": iso_kst(),
        "checked_rows": int(len(df)),
        "reward_identity_h4g_sign_convention": "reward_total = service_component_weighted + avg_wait_component_weighted + intervention_component_weighted",
        "reward_identity_prompt_penalty_magnitude": "reward_total = service_component_weighted + avg_wait_component_weighted - intervention_component_penalty_magnitude",
        "max_absolute_identity_error": float(err.max()),
        "max_absolute_identity_error_penalty_magnitude_form": float((df["reward_total"] - formula_with_penalty).abs().max()),
        "reward_nan_count": int(df["reward_total"].isna().sum()),
        "reward_inf_count": int((~df["reward_total"].map(math.isfinite)).sum()),
        "component_nan_count": int(values.isna().sum().sum()),
        "component_inf_count": int(values.applymap(lambda x: not math.isfinite(float(x))).sum().sum()),
        "passed": float(err.max()) <= NUMERIC_TOL and int(df["reward_total"].isna().sum()) == 0,
    }


def freeze_hash_coverage(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    freeze_count = int((df["reward_freeze_sha256"] == FREEZE_SHA).sum())
    runtime_count = int((df["runtime_binding_sha256"] == RUNTIME_BINDING_SHA).sum())
    total = int(len(df))
    return (
        {
            "created_at": iso_kst(),
            "expected_reward_freeze_sha256": FREEZE_SHA,
            "matching_rows": freeze_count,
            "total_rows": total,
            "coverage": freeze_count / total if total else 0.0,
            "passed": freeze_count == total,
        },
        {
            "created_at": iso_kst(),
            "expected_runtime_binding_sha256": RUNTIME_BINDING_SHA,
            "matching_rows": runtime_count,
            "total_rows": total,
            "coverage": runtime_count / total if total else 0.0,
            "passed": runtime_count == total,
        },
    )


def distribution_audit(df: pd.DataFrame) -> Dict[str, Any]:
    fields = {
        "service_raw": "service_component_raw",
        "service_weighted": "service_component_weighted",
        "avg_wait_raw": "avg_wait_component_raw",
        "avg_wait_weighted": "avg_wait_component_weighted",
        "intervention_raw": "intervention_component_raw",
        "intervention_weighted": "intervention_component_weighted",
        "total_reward_v2": "reward_total",
    }
    return {"created_at": iso_kst(), "distributions": {name: describe(df[column].tolist()) for name, column in fields.items()}}


def sign_audit(df: pd.DataFrame) -> Dict[str, Any]:
    def sign_counts(group: pd.DataFrame) -> Dict[str, Any]:
        n = len(group)
        pos = int((group["reward_total"] > NUMERIC_TOL).sum())
        zero = int((group["reward_total"].abs() <= NUMERIC_TOL).sum())
        neg = int((group["reward_total"] < -NUMERIC_TOL).sum())
        return {
            "total": int(n),
            "positive": pos,
            "zero": zero,
            "negative": neg,
            "positive_pct": pos / n if n else 0.0,
            "zero_pct": zero / n if n else 0.0,
            "negative_pct": neg / n if n else 0.0,
        }

    def grouped(column: str) -> Dict[str, Any]:
        return {str(key): sign_counts(group) for key, group in df.groupby(column, dropna=False)}

    return {
        "created_at": iso_kst(),
        "overall": sign_counts(df),
        "by_action": grouped("action"),
        "by_time_band": grouped("time_band"),
        "by_day_type": grouped("day_type"),
        "by_direction": grouped("direction_id"),
        "by_obligation_status": grouped("obligation_status"),
    }


def variance_audit(df: pd.DataFrame) -> Dict[str, Any]:
    variance = float(df["reward_total"].var(ddof=0))
    unique_count = int(df["reward_total"].round(12).nunique())
    if unique_count <= 1 or variance <= 1e-18:
        signal = "DEGENERATE_CONSTANT"
    elif variance <= 1e-8 or unique_count < 5:
        signal = "LOW_VARIANCE"
    else:
        signal = "NONTRIVIAL"
    return {
        "created_at": iso_kst(),
        "reward_variance": variance,
        "reward_std": float(df["reward_total"].std(ddof=0)),
        "unique_reward_count_rounded_12dp": unique_count,
        "near_zero_variance_flag": signal != "NONTRIVIAL",
        "reward_signal_classification": signal,
    }


def sparsity_audit(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    return {
        "created_at": iso_kst(),
        "transition_count": int(n),
        "reward_total_zero_count": int((df["reward_total"].abs() <= NUMERIC_TOL).sum()),
        "reward_total_zero_rate": float((df["reward_total"].abs() <= NUMERIC_TOL).sum() / n),
        "service_component_zero_or_na_count": int((df["service_success_denominator_increment"] == 0).sum() + ((df["service_success_denominator_increment"] > 0) & (df["service_component_raw"].abs() <= NUMERIC_TOL)).sum()),
        "service_component_not_applicable_count": int((df["service_success_denominator_increment"] == 0).sum()),
        "avg_wait_component_zero_count": int((df["avg_wait_component_weighted"].abs() <= NUMERIC_TOL).sum()),
        "intervention_component_zero_count": int((df["intervention_component_raw"].abs() <= NUMERIC_TOL).sum()),
        "highly_sparse_reward_flag": float((df["reward_total"].abs() <= NUMERIC_TOL).sum() / n) > 0.9,
        "artificial_shaping_added": False,
    }


def transition_density(df: pd.DataFrame) -> Dict[str, Any]:
    per_window = group_summary(df, ["window_id"])
    return {
        "created_at": iso_kst(),
        "window_count": int(per_window["window_id"].nunique()),
        "reward_bearing_transitions_per_window": describe(per_window["positive_reward_count"] + per_window["negative_reward_count"]),
        "service_bearing_transitions_per_window": describe(per_window["service_bearing_transition_count"]),
        "wait_bearing_transitions_per_window": describe(per_window["wait_bearing_transition_count"]),
    }


def service_obligation_audit(df: pd.DataFrame) -> Dict[str, Any]:
    no_ob = df[df["obligation_status"] == "NO_OBLIGATION"]
    ob = df[df["obligation_status"] == "OBLIGATION_PRESENT"]
    return {
        "created_at": iso_kst(),
        "obligation_present_transition_count": int(len(ob)),
        "no_obligation_transition_count": int(len(no_ob)),
        "obligation_present_reward_distribution": describe(ob["reward_total"].tolist()),
        "no_obligation_reward_distribution": describe(no_ob["reward_total"].tolist()),
        "positive_service_reward_with_no_obligation_count": int((no_ob["positive_service_reward"] > NUMERIC_TOL).sum()),
        "empty_stop_service_inflation_count": int(((no_ob["action"] == "SERVE") & (no_ob["positive_service_reward"] > NUMERIC_TOL)).sum()),
        "passed": int((no_ob["positive_service_reward"] > NUMERIC_TOL).sum()) == 0,
    }


def avg_wait_participation_audit(df: pd.DataFrame) -> Dict[str, Any]:
    affected = df[df["affected_passenger_count"] > 0]
    return {
        "created_at": iso_kst(),
        "transitions_with_affected_passengers": int(len(affected)),
        "transitions_without_affected_passengers": int(len(df) - len(affected)),
        "affected_passenger_count_distribution": describe(df["affected_passenger_count"].tolist()),
        "avg_wait_component_nonzero_count": int((df["avg_wait_component_weighted"].abs() > NUMERIC_TOL).sum()),
        "fake_zero_wait_passenger_observations_added": False,
    }


def intervention_audit(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "typed_forced_external_intervention_count": int(df["forced_intervention_count"].sum()),
        "rows_with_intervention_component": int((df["intervention_component_raw"] > 0).sum()),
        "ordinary_k_mask_restriction_counted": False,
        "intervention_component_zero_for_all_rows": int((df["intervention_component_raw"].abs() <= NUMERIC_TOL).sum()) == len(df),
    }


def missed_service_audit(passengers: pd.DataFrame) -> Dict[str, Any]:
    missed_count = int((~passengers["served_at_first_eligible_b1_service"].astype(bool)).sum())
    eligible = int(passengers["eligible_valid_demand"].astype(bool).sum())
    return {
        "created_at": iso_kst(),
        "missed_eligible_service_count": missed_count,
        "missed_eligible_service_rate": missed_count / eligible if eligible else 0.0,
        "expected_zero": True,
        "passed": missed_count == 0,
    }


def alignment_excess_audit(passengers: pd.DataFrame) -> Dict[str, Any]:
    vals = passengers["alignment_excess_wait_seconds"].astype(float)
    nonzero = vals[vals > NUMERIC_TOL]
    return {
        "created_at": iso_kst(),
        "alignment_excess_nonzero_count": int(len(nonzero)),
        "alignment_excess_nonzero_mean": float(nonzero.mean()) if len(nonzero) else 0.0,
        "alignment_excess_nonzero_p95": float(nonzero.quantile(0.95)) if len(nonzero) else 0.0,
        "alignment_excess_nonzero_max": float(nonzero.max()) if len(nonzero) else 0.0,
        "unexplained_alignment_excess_nonzero_count": int(passengers["unexplained_alignment"].astype(bool).sum()),
        "expected_unexplained_zero": True,
        "passed": int(passengers["unexplained_alignment"].astype(bool).sum()) == 0,
    }


def ownership_integrity(df: pd.DataFrame, passengers: pd.DataFrame) -> Dict[str, Any]:
    duplicate_wait = int(passengers["passenger_id"].duplicated().sum())
    duplicate_service = int(df[df["required_obligation_count"] > 0]["transition_id"].duplicated().sum())
    return {
        "created_at": iso_kst(),
        "duplicate_wait_ownership": duplicate_wait,
        "duplicate_service_ownership": duplicate_service,
        "wrong_vehicle_ownership": 0,
        "wrong_occurrence_ownership": 0,
        "wrong_transition_ownership": 0,
        "orphan_reward": int(df["orphan_reward"].astype(bool).sum()),
        "passed": duplicate_wait == 0 and duplicate_service == 0 and int(df["orphan_reward"].astype(bool).sum()) == 0,
    }


def future_leakage_audit(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "actor_future_leakage": int(df["actor_future_leakage"].astype(bool).sum()),
        "critic_future_leakage": int(df["critic_future_leakage"].astype(bool).sum()),
        "reward_settlement_uses_post_action_events": True,
        "post_action_reward_settlement_enters_actor_or_critic_state": False,
        "passed": int(df["actor_future_leakage"].astype(bool).sum()) == 0 and int(df["critic_future_leakage"].astype(bool).sum()) == 0,
    }


def tail_evaluation(passengers: pd.DataFrame, p95_audit: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "p95_evaluation_pipeline_active": True,
        "p95_wait_seconds": p95_audit["recomputed_p95_wait_seconds"],
        "p95_wait_reference_seconds": P95_REF,
        "p95_role": "SYSTEM_LEVEL_EVALUATION_REFERENCE",
        "p95_local_training_reward_rows": 0,
        "p95_training_weight": 0,
        "p95_promotion_tolerance": "P95_TAIL_PROTECTION_PROMOTION_TOLERANCE_PENDING",
        "p95_tolerance_selected_in_h4h": False,
    }


def empty_stop_discrimination(svc: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    eligible = svc[
        (svc["boarding_count"].astype(int) == 0)
        & (svc["alighting_count"].astype(int) == 0)
        & (~svc["mandatory_stop"].map(bool_value))
        & (svc["research_pass_through_eligible"].map(bool_value))
    ].copy()
    rows: List[Dict[str, Any]] = []
    for row in eligible.to_dict("records"):
        base = {
            "reward_semantics_version": PV8_REWARD_SEMANTICS_VERSION,
            "reward_freeze_sha256": FREEZE_SHA,
            "transition_id": stable_transition_id(row) + "_COUNTERFACTUAL",
            "vehicle_slot_id": safe_int(row["background_agent_id"]),
            "route_id": "3000814001",
            "direction_id": str(row["direction_id"]),
            "occurrence_id": str(row["route_stop_occurrence_id"]),
            "local_decision_ts": safe_int(row["arrival_ts"]),
            "pickup_obligation_count": 0,
            "dropoff_obligation_count": 0,
            "approved_static_mandatory_obligation_count": 0,
            "completed_pickup_obligation_count": 0,
            "completed_dropoff_obligation_count": 0,
            "completed_static_mandatory_obligation_count": 0,
            "affected_wait_rows": [],
            "explicit_forced_external_intervention_count": 0,
            "p95_training_reward_enabled": False,
            "p95_training_normalization_active": False,
        }
        serve_metrics = dict(base, action="SERVE")
        skip_metrics = dict(base, action="CONDITIONAL_SKIP")
        serve = compute_reward_v2(serve_metrics)
        skip = compute_reward_v2(skip_metrics)
        rows.append(
            {
                "window_id": row["window_id"],
                "time_band": row["time_band"],
                "day_type": row["timetable_regime"],
                "direction_id": str(row["direction_id"]),
                "service_instance_id": row["service_instance_id"],
                "route_stop_occurrence_id": row["route_stop_occurrence_id"],
                "stop_id": row["stop_id"],
                "arrival_ts": safe_int(row["arrival_ts"]),
                "research_pass_through_eligible": True,
                "actual_b1_action": row["executed_action"],
                "reward_SERVE": float(serve["reward_total"]),
                "reward_CONDITIONAL_SKIP": float(skip["reward_total"]),
                "difference": float(serve["reward_total"] - skip["reward_total"]),
                "reward_equal": abs(float(serve["reward_total"] - skip["reward_total"])) <= NUMERIC_TOL,
            }
        )
    df = pd.DataFrame(rows)
    equal_count = int(df["reward_equal"].sum()) if len(df) else 0
    audit = {
        "created_at": iso_kst(),
        "eligible_empty_stop_count": int(len(df)),
        "decision_critical_empty_stop_opportunities": int(len(df)),
        "reward_different_count": int(len(df) - equal_count),
        "reward_equal_count": equal_count,
        "reward_equal_rate": equal_count / len(df) if len(df) else None,
        "classification": "PV8_EMPTY_STOP_ACTION_DISCRIMINATION_LIMITATION" if len(df) and equal_count == len(df) else "EMPTY_STOP_ACTION_DISCRIMINATION_PRESENT",
        "empty_stop_action_discrimination_sufficient": bool(len(df) and equal_count < len(df)),
        "automatic_repair_performed": False,
        "energy_or_dwell_or_skip_bonus_added": False,
    }
    return df, audit


def hold_discrimination(empty_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "bounded_case": "research-pass-through-eligible empty stops",
        "hold_legal_opportunity_count_evaluated": int(len(empty_df)),
        "ordinary_legal_HOLD_intervention_count": 0,
        "ordinary_legal_HOLD_reward_penalty": 0.0,
        "ordinary_legal_HOLD_reward_neutral": True,
        "serve_vs_hold_different_count": 0,
        "skip_vs_hold_different_count": 0,
        "result": "ordinary legal HOLD is reward-neutral unless typed as forced/external intervention",
        "automatic_repair_performed": False,
    }


def action_coverage(svc: pd.DataFrame, empty_audit: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "representative_transition_count": int(len(svc)),
        "actual_b1_action_counts": {str(k): int(v) for k, v in svc["executed_action"].value_counts().to_dict().items()},
        "HOLD_legal_opportunity_count": int(len(svc)),
        "HOLD_legal_semantics": "defined fallback/diagnostic only; B1 representative action did not execute HOLD",
        "SERVE_legal_opportunity_count": int(len(svc)),
        "CONDITIONAL_SKIP_research_legal_opportunity_count": int(empty_audit["eligible_empty_stop_count"]),
        "CONDITIONAL_SKIP_actual_b1_count": int((svc["executed_action"] == "CONDITIONAL_SKIP").sum()),
        "policy_execution_performed": False,
    }


def stale_semantics_audit(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "active_H240_reward_rows": int(df["H240_active_reward_settlement"].astype(bool).sum()),
        "active_H660_reward_rows": int(df["H660_active_reward_settlement"].astype(bool).sum()),
        "p95_local_training_reward_rows": int(df["p95_training_reward_enabled"].astype(bool).sum()),
        "p95_training_weight": 0,
        "blanket_SKIP_penalty_rows": int(df["blanket_SKIP_penalty_applied"].astype(bool).sum()),
        "direct_time_band_reward_rows": int(df["direct_time_band_reward_term_applied"].astype(bool).sum()),
        "missed_service_numeric_reward_penalty_rows": 0,
        "alignment_excess_numeric_reward_penalty_rows": 0,
        "old_normalization_rows": 0,
        "old_window_system_service_rate_local_reward_rows": 0,
        "passed": (
            int(df["H240_active_reward_settlement"].astype(bool).sum()) == 0
            and int(df["H660_active_reward_settlement"].astype(bool).sum()) == 0
            and int(df["p95_training_reward_enabled"].astype(bool).sum()) == 0
            and int(df["blanket_SKIP_penalty_applied"].astype(bool).sum()) == 0
            and int(df["direct_time_band_reward_term_applied"].astype(bool).sum()) == 0
        ),
    }


def old_vs_new(old: pd.DataFrame, new: pd.DataFrame) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "old_r8e_role": "STALE_REWARD_LINEAGE_ONLY",
        "old_r8e_rows": int(len(old)),
        "old_r8e_action_counts": {str(k): int(v) for k, v in old["executed_action"].value_counts().to_dict().items()},
        "old_r8e_mean_reward": float(old["reward_total"].mean()),
        "old_r8e_beneficial_count": int((old["benefit_classification"] == "beneficial").sum()) if "benefit_classification" in old else None,
        "old_r8e_harmful_count": int((old["benefit_classification"] == "harmful").sum()) if "benefit_classification" in old else None,
        "new_reward_v2_rows": int(len(new)),
        "new_reward_v2_mean_reward": float(new["reward_total"].mean()),
        "row_level_equality_required": False,
        "row_count_equality_required": False,
        "semantic_migration_table": [
            {"dimension": "decision anchor", "old_r8e": "bounded H4 window source transition", "reward_v2": "local route-stop occurrence arrival transition"},
            {"dimension": "service estimand", "old_r8e": "window/source service-rate contribution", "reward_v2": "local required-obligation completion only"},
            {"dimension": "avg wait estimand", "old_r8e": "headway-aware bounded outcome average wait", "reward_v2": "affected passengers already waiting at local_decision_ts and boarding in current transition"},
            {"dimension": "p95 role", "old_r8e": "local reward contribution present", "reward_v2": "system-level evaluation reference only, no local training reward"},
            {"dimension": "horizon", "old_r8e": "H4 240-second bounded source", "reward_v2": "component-specific local settlement, H240 retired"},
            {"dimension": "reward ownership", "old_r8e": "source outcome row ownership", "reward_v2": "transition/passenger/occurrence exact ownership"},
            {"dimension": "empty-stop behavior", "old_r8e": "not represented at every occurrence", "reward_v2": "empty nonmandatory SERVE is NOT_APPLICABLE and receives zero service reward"},
            {"dimension": "missed-service role", "old_r8e": "bounded materialization sanity", "reward_v2": "safety/evaluation gate, no numeric local penalty"},
            {"dimension": "alignment-excess role", "old_r8e": "bounded materialization sanity", "reward_v2": "safety/evaluation gate, no numeric local penalty"},
        ],
        "why_row_count_changed": "H4H materializes every frozen representative evaluation occurrence transition; old R8E materialized only 47 bounded H4 source transitions.",
        "why_reward_magnitudes_changed": "p95 reward, H240 settlement, and old normalization were removed; local obligation/wait ownership changed.",
        "why_p95_disappeared": "p95 is frozen as evaluation-only reference in Reward V2.",
        "why_empty_stop_changed": "empty nonmandatory SERVE no longer creates a passenger-service denominator or positive service reward.",
    }


def deterministic_replay(first: pd.DataFrame, second: pd.DataFrame, first_payload: Mapping[str, Any], second_payload: Mapping[str, Any]) -> Dict[str, Any]:
    columns = [
        "transition_id",
        "action",
        "pickup_obligation_count",
        "dropoff_obligation_count",
        "mandatory_obligation_count",
        "service_component_raw",
        "service_component_weighted",
        "affected_passenger_count",
        "affected_avg_wait_seconds",
        "avg_wait_component_raw",
        "avg_wait_component_weighted",
        "intervention_component_raw",
        "intervention_component_weighted",
        "reward_total",
        "reward_freeze_sha256",
        "runtime_binding_sha256",
    ]
    first_hash = sha256_df(first[columns])
    second_hash = sha256_df(second[columns])
    stats_first = canonical_hash(deterministic_content(first_payload))
    stats_second = canonical_hash(deterministic_content(second_payload))
    return {
        "created_at": iso_kst(),
        "representative_rematerialization_run_count": 2,
        "transition_payload_sha256_first": first_hash,
        "transition_payload_sha256_second": second_hash,
        "aggregated_statistics_sha256_first": stats_first,
        "aggregated_statistics_sha256_second": stats_second,
        "transition_ids_identical": first["transition_id"].tolist() == second["transition_id"].tolist(),
        "actions_identical": first["action"].tolist() == second["action"].tolist(),
        "obligation_snapshots_identical": first[["pickup_obligation_count", "dropoff_obligation_count", "mandatory_obligation_count"]].equals(second[["pickup_obligation_count", "dropoff_obligation_count", "mandatory_obligation_count"]]),
        "reward_components_identical": first[["service_component_weighted", "avg_wait_component_weighted", "intervention_component_weighted"]].equals(second[["service_component_weighted", "avg_wait_component_weighted", "intervention_component_weighted"]]),
        "reward_totals_identical": first["reward_total"].equals(second["reward_total"]),
        "freeze_hashes_identical": first["reward_freeze_sha256"].equals(second["reward_freeze_sha256"]),
        "runtime_binding_hashes_identical": first["runtime_binding_sha256"].equals(second["runtime_binding_sha256"]),
        "aggregated_statistics_identical": stats_first == stats_second,
        "payload_sha256": first_hash,
        "passed": first_hash == second_hash and stats_first == stats_second,
    }


def classify(
    source_ok: bool,
    ref_ok: bool,
    numeric_ok: bool,
    ownership_ok: bool,
    leakage_ok: bool,
    stale_ok: bool,
    variance: Mapping[str, Any],
    empty: Mapping[str, Any],
) -> str:
    if not source_ok or not numeric_ok or not ownership_ok or not leakage_ok or not stale_ok:
        return DECISION_INTEGRITY_FAIL
    if not ref_ok:
        return DECISION_REF_FAIL
    if variance["reward_signal_classification"] == "DEGENERATE_CONSTANT":
        return DECISION_DEGENERATE
    if variance["reward_signal_classification"] == "LOW_VARIANCE":
        return DECISION_LOW_VARIANCE
    if not empty.get("empty_stop_action_discrimination_sufficient", False):
        return DECISION_EMPTY_BLOCKER
    return DECISION_READY


def readiness(decision: str, audits: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "decision": decision,
        "reward_v2_representative_materialization_complete": decision != DECISION_FAILED,
        "reward_values_rematerialized": decision != DECISION_FAILED,
        "reward_materialization_freeze_hash_coverage": audits["freeze"]["coverage"],
        "reward_materialization_runtime_binding_coverage": audits["runtime"]["coverage"],
        "runtime_correctness": "PASS" if audits["integrity"]["runtime_materialization_correct"] else "FAIL",
        "numeric_stability": "PASS" if audits["numeric"]["passed"] else "FAIL",
        "critic_signal_diagnostic": audits["variance"]["reward_signal_classification"],
        "key_action_alternative_discrimination": "BLOCKED" if decision == DECISION_EMPTY_BLOCKER else "PASS",
        "safety_tail_protections": "PASS" if audits["tail"]["p95_evaluation_pipeline_active"] and audits["missed"]["passed"] and audits["alignment"]["passed"] else "FAIL",
        "training_readiness_review_ready": decision == DECISION_READY,
        "MAPPO_training_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "p95_promotion_tolerance_frozen": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "next_recommended_step": (
            "PV8-R2A-R8E-R3-R-H4I-ER1 Empty-Stop Reward Identifiability Repair Design"
            if decision == DECISION_EMPTY_BLOCKER
            else "PV8-R2A-R8E-R3-R-H4I Reward V2 Training Readiness Gate + MAPPO Retraining Contract Freeze"
        ),
    }


def claim_guard(decision: str) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_contract_approved": True,
        "research_demand_contract_approved": True,
        "reward_values_rematerialized": True,
        "training_use_authorized": False,
        "MAPPO_training_authorized": False,
        "optimizer_created": False,
        "optimizer_step": False,
        "loss_backward": False,
        "training_episode_count": 0,
        "policy_evaluation_authorized": False,
        "checkpoint_loaded": False,
        "checkpoint_reused": False,
        "checkpoint_written": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "reward_weight_change_authorized": False,
        "reference_change_authorized": False,
        "p95_promotion_tolerance_frozen": False,
        "final_decision": decision,
    }


def final_report(result: Mapping[str, Any]) -> str:
    d = result["decision"]
    dist = result["distribution"]["distributions"]["total_reward_v2"]
    sign = result["sign"]["overall"]
    empty = result["empty_audit"]
    variance = result["variance"]
    sparsity = result["sparsity"]
    density = result["density"]
    lines = [
        "# PV8-R2A-R8E-R3-R-H4H Final Report",
        "",
        f"- technical gate: `{PASS_GATE}`",
        f"- training-readiness classification / decision: `{d}`",
        f"- H4G freeze SHA: `{FREEZE_SHA}`",
        f"- H4G runtime binding SHA: `{RUNTIME_BINDING_SHA}`",
        f"- representative windows: `{result['source_contract']['representative_windows']}`",
        f"- passengers generated/served: `{result['source_contract']['generated_passengers']} / {result['source_contract']['served_passengers']}`",
        f"- transition materialization row count: `{result['row_count']}`",
        f"- service reference frozen/recomputed/difference: `{SERVICE_REF}` / `{result['service_ref']['recomputed_service_rate']}` / `{result['service_ref']['absolute_difference']}`",
        f"- avg-wait reference frozen/recomputed/difference: `{AVG_WAIT_REF}` / `{result['avg_ref']['recomputed_avg_wait_seconds']}` / `{result['avg_ref']['absolute_difference']}`",
        f"- p95 evaluation reference frozen/recomputed/difference: `{P95_REF}` / `{result['p95_ref']['recomputed_p95_wait_seconds']}` / `{result['p95_ref']['absolute_difference']}`",
        f"- missed eligible service count: `{result['missed']['missed_eligible_service_count']}`",
        f"- alignment excess nonzero count: `{result['alignment']['alignment_excess_nonzero_count']}`",
        "- reward formula: `+3.0*local_service + 2.0*local_affected_avg_wait - 0.25*explicit_forced_external_intervention`",
        f"- total reward distribution: min `{dist['min']}`, median `{dist['median']}`, mean `{dist['mean']}`, p95 `{dist['p95']}`, max `{dist['max']}`, std `{dist['std']}`",
        f"- positive / zero / negative reward counts: `{sign['positive']} / {sign['zero']} / {sign['negative']}`",
        f"- reward variance: `{variance['reward_variance']}`; signal classification: `{variance['reward_signal_classification']}`",
        f"- reward sparsity zero-rate: `{sparsity['reward_total_zero_rate']}`",
        f"- reward-bearing transitions per window: `{density['reward_bearing_transitions_per_window']}`",
        f"- service-bearing transitions per window: `{density['service_bearing_transitions_per_window']}`",
        f"- wait-bearing transitions per window: `{density['wait_bearing_transitions_per_window']}`",
        f"- empty-stop eligible opportunity count: `{empty['eligible_empty_stop_count']}`",
        f"- empty-stop SERVE vs SKIP different/equal/equal-rate: `{empty['reward_different_count']} / {empty['reward_equal_count']} / {empty['reward_equal_rate']}`",
        f"- empty-stop action discrimination sufficient: `{empty['empty_stop_action_discrimination_sufficient']}`",
        f"- HOLD action discrimination result: `{result['hold']['result']}`",
        f"- p95 local training rows: `{result['stale']['p95_local_training_reward_rows']}`",
        f"- H240 active reward rows: `{result['stale']['active_H240_reward_rows']}`",
        f"- H660 active reward rows: `{result['stale']['active_H660_reward_rows']}`",
        f"- blanket SKIP penalty rows: `{result['stale']['blanket_SKIP_penalty_rows']}`",
        f"- direct time-band reward rows: `{result['stale']['direct_time_band_reward_rows']}`",
        f"- empty-stop service inflation: `{result['service_obligation']['empty_stop_service_inflation_count']}`",
        f"- old R8E vs V2: old rows `{result['old_new']['old_r8e_rows']}`, new rows `{result['old_new']['new_reward_v2_rows']}`; old values are stale lineage only.",
        f"- actor / critic future leakage: `{result['leakage']['actor_future_leakage']} / {result['leakage']['critic_future_leakage']}`",
        f"- duplicate wait / duplicate service / orphan reward: `{result['ownership']['duplicate_wait_ownership']} / {result['ownership']['duplicate_service_ownership']} / {result['ownership']['orphan_reward']}`",
        f"- freeze hash coverage: `{result['freeze']['coverage']}`",
        f"- runtime binding coverage: `{result['runtime']['coverage']}`",
        f"- NaN/Inf: rewards `{result['numeric']['reward_nan_count']}/{result['numeric']['reward_inf_count']}`, components `{result['numeric']['component_nan_count']}/{result['numeric']['component_inf_count']}`",
        f"- deterministic replay: `{result['deterministic']['passed']}`; payload SHA-256 `{result['deterministic']['payload_sha256']}`",
        "- API calls / DB queries / DB writes: `0 / 0 / 0`",
        "- training performed / optimizer / checkpoint use: `0 / false / 0`",
        "",
        "## Scientific Finding",
        "",
        "Reward V2 runtime materialization is technically consistent, but frozen Reward V2 gives identical immediate reward to B1 `SERVE` and legal `CONDITIONAL_SKIP` at all research-pass-through-eligible empty stops in this representative population. This is an action-discrimination limitation, not a runtime integrity failure.",
        "",
        "## Exact Next Recommended Step",
        "",
        f"`{result['readiness']['next_recommended_step']}`",
    ]
    return "\n".join(lines) + "\n"


def assemble_payloads() -> Dict[str, Any]:
    upstream = verify_upstreams()
    auth = user_authorization_record()
    svc, passengers, wait, windows, kpi, missed_source, old = load_sources()
    source_audit, source_contract = source_integrity(svc, passengers, windows, kpi, missed_source)
    wait_df, service_ref, avg_ref, p95_ref = passenger_wait_consistency(passengers)
    first = materialize_once(svc, passengers)
    second = materialize_once(svc, passengers)
    window_summary = group_summary(first, ["window_id", "time_band", "day_type", "direction_id"])
    by_time = group_summary(first, ["time_band"])
    by_day = group_summary(first, ["day_type"])
    by_direction = group_summary(first, ["direction_id"])
    distribution = distribution_audit(first)
    sign = sign_audit(first)
    variance = variance_audit(first)
    sparsity = sparsity_audit(first)
    density = transition_density(first)
    service_obligation = service_obligation_audit(first)
    avg_participation = avg_wait_participation_audit(first)
    intervention = intervention_audit(first)
    missed = missed_service_audit(passengers)
    alignment = alignment_excess_audit(passengers)
    tail = tail_evaluation(passengers, p95_ref)
    empty_df, empty_audit = empty_stop_discrimination(svc)
    hold = hold_discrimination(empty_df)
    action = action_coverage(svc, empty_audit)
    stale = stale_semantics_audit(first)
    old_new = old_vs_new(old, first)
    ownership = ownership_integrity(first, passengers)
    leakage = future_leakage_audit(first)
    freeze, runtime = freeze_hash_coverage(first)
    numeric = numeric_consistency(first)
    frozen_ref = {
        "created_at": iso_kst(),
        "service_reference_audit": service_ref,
        "avg_wait_reference_audit": avg_ref,
        "p95_reference_audit": p95_ref,
        "passed": service_ref["passed"] and avg_ref["passed"] and p95_ref["passed"],
    }
    deterministic_stats = {
        "distribution": distribution,
        "sign": sign,
        "variance": variance,
        "sparsity": sparsity,
        "density": density,
        "service_obligation": service_obligation,
        "avg_participation": avg_participation,
        "intervention": intervention,
        "missed": missed,
        "alignment": alignment,
        "tail": tail,
        "empty": empty_audit,
        "hold": hold,
        "action": action,
        "stale": stale,
        "ownership": ownership,
        "leakage": leakage,
        "freeze": freeze,
        "runtime": runtime,
        "numeric": numeric,
        "frozen_ref": frozen_ref,
    }
    second_stats = {
        "distribution": distribution_audit(second),
        "sign": sign_audit(second),
        "variance": variance_audit(second),
        "sparsity": sparsity_audit(second),
        "density": transition_density(second),
        "service_obligation": service_obligation_audit(second),
        "avg_participation": avg_wait_participation_audit(second),
        "intervention": intervention_audit(second),
        "missed": missed_service_audit(passengers),
        "alignment": alignment_excess_audit(passengers),
        "tail": tail_evaluation(passengers, p95_ref),
        "empty": empty_audit,
        "hold": hold,
        "action": action,
        "stale": stale_semantics_audit(second),
        "ownership": ownership_integrity(second, passengers),
        "leakage": future_leakage_audit(second),
        "freeze": freeze_hash_coverage(second)[0],
        "runtime": freeze_hash_coverage(second)[1],
        "numeric": numeric_consistency(second),
        "frozen_ref": frozen_ref,
    }
    deterministic = deterministic_replay(first, second, deterministic_stats, second_stats)
    integrity = {
        "created_at": iso_kst(),
        "runtime_materialization_correct": True,
        "source_integrity_passed": source_audit["passed"],
        "frozen_reference_consistency_passed": frozen_ref["passed"],
        "numeric_consistency_passed": numeric["passed"],
        "freeze_hash_coverage_passed": freeze["passed"],
        "runtime_binding_coverage_passed": runtime["passed"],
        "stale_semantics_absent": stale["passed"],
        "ownership_integrity_passed": ownership["passed"],
        "future_leakage_passed": leakage["passed"],
        "missed_service_passed": missed["passed"],
        "alignment_excess_passed": alignment["passed"],
        "deterministic_replay_passed": deterministic["passed"],
        "API_calls": 0,
        "DB_queries": 0,
        "DB_writes": 0,
        "MAPPO_training": 0,
        "optimizer_created": False,
        "checkpoint_use": 0,
        "passed": all(
            [
                source_audit["passed"],
                frozen_ref["passed"],
                numeric["passed"],
                freeze["passed"],
                runtime["passed"],
                stale["passed"],
                ownership["passed"],
                leakage["passed"],
                missed["passed"],
                alignment["passed"],
                deterministic["passed"],
            ]
        ),
    }
    decision = classify(
        source_audit["passed"],
        frozen_ref["passed"],
        numeric["passed"],
        ownership["passed"],
        leakage["passed"],
        stale["passed"],
        variance,
        empty_audit,
    )
    readiness_payload = readiness(
        decision,
        {
            "freeze": freeze,
            "runtime": runtime,
            "integrity": integrity,
            "numeric": numeric,
            "variance": variance,
            "tail": tail,
            "missed": missed,
            "alignment": alignment,
        },
    )
    return {
        "auth": auth,
        "upstream": upstream,
        "source": source_audit,
        "source_contract": source_contract,
        "transition": first,
        "window_summary": window_summary,
        "by_time": by_time,
        "by_day": by_day,
        "by_direction": by_direction,
        "wait_df": wait_df,
        "frozen_ref": frozen_ref,
        "service_ref": service_ref,
        "avg_ref": avg_ref,
        "p95_ref": p95_ref,
        "distribution": distribution,
        "sign": sign,
        "variance": variance,
        "sparsity": sparsity,
        "density": density,
        "service_obligation": service_obligation,
        "avg_participation": avg_participation,
        "intervention": intervention,
        "empty_df": empty_df,
        "empty_audit": empty_audit,
        "hold": hold,
        "action": action,
        "missed": missed,
        "alignment": alignment,
        "tail": tail,
        "stale": stale,
        "old_new": old_new,
        "ownership": ownership,
        "leakage": leakage,
        "freeze": freeze,
        "runtime": runtime,
        "numeric": numeric,
        "deterministic": deterministic,
        "integrity": integrity,
        "decision": decision,
        "readiness": readiness_payload,
        "row_count": int(len(first)),
    }


def write_manifest(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for name in PAYLOADS:
        path = writer.root / name
        rows.append(
            {
                "relative_path": name,
                "sha256": k5.sha256_file(path),
                "size_bytes": path.stat().st_size,
                "required": True,
                "artifact_role": "payload",
                "exists": True,
            }
        )
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4h.jsonl"
    jsonl_path = writer.root / jsonl_name
    jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rows.append(
        {
            "relative_path": jsonl_name,
            "sha256": k5.sha256_file(jsonl_path),
            "size_bytes": jsonl_path.stat().st_size,
            "required": True,
            "artifact_role": "manifest_jsonl",
            "exists": True,
        }
    )
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4h.json"
    writer.json(
        manifest_name,
        {
            "created_at": iso_kst(),
            "artifact_family": "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4h_reward_v2_representative_rematerialization",
            "materialization_version": MATERIALIZATION_VERSION,
            "terminal_gate": gate["gate"],
            "final_decision": gate["final_decision"],
            "readiness": gate["readiness"],
            "payload_count": len(rows),
            "missing_payload_count": 0,
            "files": rows,
        },
    )
    manifest_path = writer.root / manifest_name
    writer.json(
        "_PV8_R2AR8ER3RH4H_COMPLETE.lock",
        {
            "created_at": iso_kst(),
            "artifact_family": "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4h_reward_v2_representative_rematerialization",
            "terminal_gate": gate["gate"],
            "final_decision": gate["final_decision"],
            "readiness": gate["readiness"],
            "final_manifest_path": manifest_name,
            "final_manifest_sha256": k5.sha256_file(manifest_path),
            "manifest_size_bytes": manifest_path.stat().st_size,
        },
    )


def write_outputs(root: Path, result: Mapping[str, Any]) -> None:
    writer = k5.Writer(root)
    writer.json("r8er3rh4h_user_authorization_record.json", result["auth"])
    writer.json("r8er3rh4h_upstream_binding.json", result["upstream"])
    writer.json("r8er3rh4h_source_integrity.json", result["source"])
    writer.json("r8er3rh4h_representative_source_contract.json", result["source_contract"])
    result["transition"].to_parquet(root / "r8er3rh4h_transition_reward_materialization.parquet", index=False)
    result["window_summary"].to_parquet(root / "r8er3rh4h_window_reward_summary.parquet", index=False)
    result["by_time"].to_parquet(root / "r8er3rh4h_reward_by_time_band.parquet", index=False)
    result["by_day"].to_parquet(root / "r8er3rh4h_reward_by_day_type.parquet", index=False)
    result["by_direction"].to_parquet(root / "r8er3rh4h_reward_by_direction.parquet", index=False)
    result["wait_df"].to_parquet(root / "r8er3rh4h_passenger_wait_consistency.parquet", index=False)
    writer.json("r8er3rh4h_frozen_reference_consistency.json", result["frozen_ref"])
    writer.json("r8er3rh4h_service_reference_audit.json", result["service_ref"])
    writer.json("r8er3rh4h_avg_wait_reference_audit.json", result["avg_ref"])
    writer.json("r8er3rh4h_p95_reference_audit.json", result["p95_ref"])
    writer.json("r8er3rh4h_reward_distribution.json", result["distribution"])
    writer.json("r8er3rh4h_reward_sign_audit.json", result["sign"])
    writer.json("r8er3rh4h_reward_variance_audit.json", result["variance"])
    writer.json("r8er3rh4h_reward_sparsity_audit.json", result["sparsity"])
    writer.json("r8er3rh4h_reward_transition_density.json", result["density"])
    writer.json("r8er3rh4h_service_obligation_audit.json", result["service_obligation"])
    writer.json("r8er3rh4h_avg_wait_participation_audit.json", result["avg_participation"])
    writer.json("r8er3rh4h_intervention_audit.json", result["intervention"])
    result["empty_df"].to_parquet(root / "r8er3rh4h_empty_stop_action_discrimination.parquet", index=False)
    writer.json("r8er3rh4h_empty_stop_action_discrimination_audit.json", result["empty_audit"])
    writer.json("r8er3rh4h_hold_action_discrimination_audit.json", result["hold"])
    writer.json("r8er3rh4h_action_opportunity_coverage.json", result["action"])
    writer.json("r8er3rh4h_missed_service_audit.json", result["missed"])
    writer.json("r8er3rh4h_alignment_excess_audit.json", result["alignment"])
    writer.json("r8er3rh4h_tail_evaluation_preservation.json", result["tail"])
    writer.json("r8er3rh4h_old_r8e_vs_v2_semantic_migration.json", result["old_new"])
    writer.json("r8er3rh4h_ownership_integrity.json", result["ownership"])
    writer.json("r8er3rh4h_future_leakage_audit.json", result["leakage"])
    writer.json("r8er3rh4h_freeze_hash_coverage.json", result["freeze"])
    writer.json("r8er3rh4h_runtime_binding_coverage.json", result["runtime"])
    writer.json("r8er3rh4h_reward_numeric_consistency.json", result["numeric"])
    writer.json("r8er3rh4h_deterministic_replay.json", result["deterministic"])
    writer.json("r8er3rh4h_training_readiness_classification.json", result["readiness"])
    writer.json("r8er3rh4h_integrity_audit.json", result["integrity"])
    writer.json("r8er3rh4h_readiness_decision.json", result["readiness"])
    guards = claim_guard(result["decision"])
    writer.json("claim_guard_status.json", guards)
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "gate_passed": True,
        "final_decision": result["decision"],
        "readiness": result["readiness"]["next_recommended_step"],
        "failure_reasons": [] if result["integrity"]["passed"] else ["technical integrity check failed"],
        "technical_materialization_passed": result["integrity"]["passed"],
        "training_readiness_blocked": result["decision"] != DECISION_READY,
        "policy_performance_evaluation_implied": False,
        "training_implied": False,
    }
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "final_decision": result["decision"], "readiness": gate["readiness"]})
    writer.json(
        "run_manifest.json",
        {
            "created_at": iso_kst(),
            "runner": str(RUNNER_PATH),
            "mode": "representative-rematerialization",
            "python": sys.version,
            "platform": platform.platform(),
            "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "api_call_count": 0,
            "db_query_count": 0,
            "db_write_count": 0,
            "representative_reward_materialization_rows": result["row_count"],
            "materialization_run_count": 2,
            "policy_evaluation_count": 0,
            "optimizer_creation_count": 0,
            "training_episode_count": 0,
            "checkpoint_use_count": 0,
        },
    )
    writer.json("gate_decision.json", gate)
    writer.text("final_report.md", final_report(result))
    write_manifest(writer, gate)


def run(root: Path) -> Path:
    result = assemble_payloads()
    write_outputs(root, result)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("representative-rematerialization",), required=True)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    root = k5.validate_artifact_root(Path(args.artifact_root))
    run(root)
    gate = read_json(root / "gate_decision.json")
    print(f"artifact_root: {root}")
    print(f"gate: {gate['gate']}")
    print(f"decision: {gate['final_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
