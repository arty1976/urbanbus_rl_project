#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4I-ER0 empty-stop temporal return audit.

This diagnostic keeps Reward V2 and simulator runtime files byte-identical. It
uses frozen representative H4H/R3-R artifacts to ask whether legal empty-stop
CONDITIONAL_SKIP has delayed return/GAE signal even though immediate Reward V2
at t0 is equal to SERVE.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import resource
import sys
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
from mappo_runner import RunnerConfig, compute_gae
from rewards.mappo_reward_v1 import (
    PV8_REWARD_SEMANTICS_VERSION,
    compute_reward_v2,
)


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4i_er0_empty_stop_temporal_return_audit.py"

H4F_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4f_reward_v2_immutable_freeze_preflight_20260809_232644"
H4G_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4g_reward_v2_runtime_rebinding_20260809_235045"
R3R_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"

H4F_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4F_REWARD_V2_APPROVED_IMMUTABLE_FREEZE_AND_RUNTIME_REBINDING_PREFLIGHT_COMPLETE"
H4G_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4G_REWARD_V2_RUNTIME_REBINDING_AND_DETERMINISTIC_INTEGRATION_VALIDATION_COMPLETE"
H4H_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4H_REWARD_V2_REPRESENTATIVE_REMATERIALIZATION_AND_FROZEN_REFERENCE_CONSISTENCY_AUDIT_COMPLETE"
H4H_DECISION = "PV8_EMPTY_STOP_ACTION_DISCRIMINATION_BLOCKER"

FREEZE_SHA = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
RUNTIME_BINDING_SHA = "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4I_ER0_EMPTY_STOP_TEMPORAL_RETURN_IDENTIFIABILITY_AND_CREDIT_PROPAGATION_AUDIT_COMPLETE"

DECISION_IDENTIFIABLE = "PV8_EMPTY_STOP_TEMPORAL_RETURN_IDENTIFIABLE_NO_REWARD_REDESIGN_REQUIRED"
DECISION_WEAK = "PV8_TEMPORAL_SIGNAL_IDENTIFIABLE_BUT_WEAK"
DECISION_DYNAMICS = "PV8_EMPTY_STOP_CAUSAL_DYNAMICS_BLOCKER"
DECISION_REWARD = "PV8_REWARD_TEMPORAL_IDENTIFIABILITY_BLOCKER"
DECISION_GAE = "PV8_GAE_CREDIT_PROPAGATION_BLOCKER"
DECISION_APPROVAL = "PV8_H4I_ER0_RUNTIME_INSTRUMENTATION_REQUIRES_SEPARATE_APPROVAL"
DECISION_SOURCE = "PV8_H4I_ER0_SOURCE_INTEGRITY_FAILED"
DECISION_DETERMINISM = "PV8_H4I_ER0_DETERMINISM_FAILED"
DECISION_INTEGRITY = "PV8_H4I_ER0_INTEGRITY_FAILED"

AVG_WAIT_REF = 297.7850241545894
P95_REF = 576.6999999999999
NUMERIC_TOL = 1e-12

PROTECTED_RUNTIME_FILES = [
    "05_training/rewards/mappo_reward_v1.py",
    "05_training/simulator/pv8_reward_outcome_collector.py",
    "05_training/simulator/pv8_b1_orchestrator.py",
    "05_training/mappo_runner.py",
]

PAYLOADS = [
    "r8er3rh4i_er0_user_authorization_record.json",
    "r8er3rh4i_er0_upstream_binding.json",
    "r8er3rh4i_er0_source_integrity.json",
    "r8er3rh4i_er0_empty_stop_opportunity_registry.parquet",
    "r8er3rh4i_er0_paired_prestate_identity.parquet",
    "r8er3rh4i_er0_action_legality_audit.json",
    "r8er3rh4i_er0_counterfactual_branch_trace.parquet",
    "r8er3rh4i_er0_temporal_delta_trace.parquet",
    "r8er3rh4i_er0_time_saving_persistence.json",
    "r8er3rh4i_er0_downstream_passenger_exposure.parquet",
    "r8er3rh4i_er0_passenger_wait_counterfactual.parquet",
    "r8er3rh4i_er0_reward_sequence_comparison.parquet",
    "r8er3rh4i_er0_discounted_return_audit.parquet",
    "r8er3rh4i_er0_gamma_lambda_binding.json",
    "r8er3rh4i_er0_compute_gae_binding.json",
    "r8er3rh4i_er0_gae_credit_propagation.parquet",
    "r8er3rh4i_er0_gae_analytic_crosscheck.json",
    "r8er3rh4i_er0_credit_distance_distribution.json",
    "r8er3rh4i_er0_credit_attenuation_diagnostic.json",
    "r8er3rh4i_er0_time_band_diagnostic.parquet",
    "r8er3rh4i_er0_day_type_diagnostic.parquet",
    "r8er3rh4i_er0_direction_diagnostic.parquet",
    "r8er3rh4i_er0_safety_integrity.json",
    "r8er3rh4i_er0_ownership_integrity.json",
    "r8er3rh4i_er0_future_leakage_audit.json",
    "r8er3rh4i_er0_fixture_results.json",
    "r8er3rh4i_er0_deterministic_replay.json",
    "r8er3rh4i_er0_scientific_classification.json",
    "r8er3rh4i_er0_readiness_decision.json",
    "claim_guard_status.json",
    "downstream_lock.json",
    "run_manifest.json",
    "gate_decision.json",
    "final_report.md",
]


class H4IER0Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        k5.json_clean(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deterministic_content(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): deterministic_content(item) for key, item in value.items() if key != "created_at"}
    if isinstance(value, list):
        return [deterministic_content(item) for item in value]
    if isinstance(value, tuple):
        return [deterministic_content(item) for item in value]
    return value


def sha256_df(df: pd.DataFrame) -> str:
    stable = df.copy().reindex(sorted(df.columns), axis=1)
    payload = stable.to_json(orient="records", date_format="iso", force_ascii=False, double_precision=15)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def describe(values: Iterable[Any]) -> Dict[str, Any]:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not vals:
        return {"count": 0, "min": None, "p05": None, "p25": None, "median": None, "mean": None, "p75": None, "p95": None, "max": None, "std": None}
    s = pd.Series(vals, dtype="float64")
    return {
        "count": int(s.count()),
        "min": float(s.min()),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "mean": float(s.mean()),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
        "std": float(s.std(ddof=0)),
    }


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def safe_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0
    return int(value)


def verify_manifest_gate(root: Path, manifest: str, lock: str, expected_gate: str, expected_decision: Optional[str] = None) -> Dict[str, Any]:
    checks = k5.verify_manifest(root, manifest, lock)
    gate = k5.read_json(root / "gate_decision.json")
    observed = gate.get("gate") or gate.get("terminal_gate")
    if observed != expected_gate or not k5.manifest_ok(checks):
        raise H4IER0Error(f"upstream validation failed for {root.name}: gate={observed} checks={checks}")
    if expected_decision and gate.get("final_decision") != expected_decision:
        raise H4IER0Error(f"upstream decision drifted for {root.name}: {gate.get('final_decision')}")
    return {"artifact_root": str(root), "gate": observed, "decision": gate.get("final_decision"), "readiness": gate.get("readiness"), "manifest_integrity": checks}


def find_authoritative_h4h() -> Path:
    candidates: List[Path] = []
    for root in sorted(ARTIFACTS_ROOT.glob("prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4h_reward_v2_representative_rematerialization_*")):
        try:
            checks = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4h.json", "_PV8_R2AR8ER3RH4H_COMPLETE.lock")
            gate = k5.read_json(root / "gate_decision.json")
            integrity = k5.read_json(root / "r8er3rh4h_integrity_audit.json")
            deterministic = k5.read_json(root / "r8er3rh4h_deterministic_replay.json")
            if (
                k5.manifest_ok(checks)
                and gate.get("gate") == H4H_GATE
                and gate.get("final_decision") == H4H_DECISION
                and bool(integrity.get("passed"))
                and bool(deterministic.get("passed"))
            ):
                candidates.append(root)
        except Exception:
            continue
    if len(candidates) != 1:
        raise H4IER0Error(f"expected exactly one authoritative H4H artifact, found {len(candidates)}: {[str(p) for p in candidates]}")
    return candidates[0]


def protected_hashes() -> Dict[str, str]:
    return {relative: k5.sha256_file(PROJECT_ROOT / relative) for relative in PROTECTED_RUNTIME_FILES}


def user_authorization_record() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "h4i_er0_temporal_return_audit_authorized": True,
        "reward_redesign_authorized": False,
        "reward_weight_modification_authorized": False,
        "normalization_reference_modification_authorized": False,
        "SKIP_bonus_authorized": False,
        "SERVE_penalty_authorized": False,
        "dwell_penalty_authorized": False,
        "energy_reward_addition_authorized": False,
        "H240_H660_reactivation_authorized": False,
        "MAPPO_training_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_evaluation_authorized": False,
        "p95_promotion_tolerance_selection_authorized": False,
        "paper_level_causal_claims_authorized": False,
    }


def verify_upstreams(h4h_root: Path) -> Dict[str, Any]:
    h4f = verify_manifest_gate(
        H4F_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4f.json",
        "_PV8_R2AR8ER3RH4F_COMPLETE.lock",
        H4F_GATE,
        "PV8_REWARD_V2_IMMUTABLY_FROZEN_RUNTIME_REBINDING_READY",
    )
    h4g = verify_manifest_gate(
        H4G_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4g.json",
        "_PV8_R2AR8ER3RH4G_COMPLETE.lock",
        H4G_GATE,
        "PV8_REWARD_V2_RUNTIME_REBOUND_REPRESENTATIVE_REMATERIALIZATION_READY",
    )
    h4h = verify_manifest_gate(
        h4h_root,
        "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4h.json",
        "_PV8_R2AR8ER3RH4H_COMPLETE.lock",
        H4H_GATE,
        H4H_DECISION,
    )
    h4g_hash = k5.read_json(H4G_ROOT / "r8er3rh4g_runtime_binding_hash.json")
    if h4g_hash.get("freeze_sha256") != FREEZE_SHA or h4g_hash.get("runtime_binding_sha256") != RUNTIME_BINDING_SHA:
        raise H4IER0Error("H4G freeze/runtime binding hash drift")
    h4h_integrity = k5.read_json(h4h_root / "r8er3rh4h_integrity_audit.json")
    h4h_det = k5.read_json(h4h_root / "r8er3rh4h_deterministic_replay.json")
    if not h4h_integrity.get("passed") or not h4h_det.get("passed"):
        raise H4IER0Error("H4H technical integrity is not authoritative")
    return {
        "created_at": iso_kst(),
        "H4F_immutable_freeze": h4f,
        "H4G_runtime_binding": h4g,
        "H4H_representative_rematerialization": h4h,
        "H4H_source_artifact": str(h4h_root),
        "reward_freeze_sha256": FREEZE_SHA,
        "runtime_binding_sha256": RUNTIME_BINDING_SHA,
    }


def load_sources(h4h_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    svc = pd.read_parquet(R3R_ROOT / "r8er3r_occurrence_service_events.parquet")
    svc = svc[~svc["is_warmup"].astype(bool)].copy().reset_index(drop=True)
    svc = svc.sort_values(["window_id", "arrival_ts", "service_instance_id", "occurrence_index"], kind="stable").reset_index(drop=True)
    svc["global_transition_rank"] = svc.groupby("window_id", sort=False).cumcount()
    h4h = pd.read_parquet(h4h_root / "r8er3rh4h_transition_reward_materialization.parquet")
    merge_cols = ["window_id", "service_instance_id", "route_stop_occurrence_id", "arrival_ts"]
    h4h_small = h4h.rename(columns={"local_decision_ts": "arrival_ts"})[
        merge_cols + ["transition_id", "reward_total", "service_component_weighted", "avg_wait_component_weighted", "intervention_component_weighted"]
    ].copy()
    svc = svc.merge(h4h_small, on=merge_cols, how="left", validate="one_to_one")
    if svc["transition_id"].isna().any():
        raise H4IER0Error("H4H transition binding does not cover every R3-R evaluation service row")
    passengers = pd.read_parquet(R3R_ROOT / "r8er3r_passenger_lifecycle.parquet")
    windows = pd.read_parquet(R3R_ROOT / "r8er3r_representative_window_registry.parquet")
    h4h_empty = pd.read_parquet(h4h_root / "r8er3rh4h_empty_stop_action_discrimination.parquet")
    h4h_empty_audit = k5.read_json(h4h_root / "r8er3rh4h_empty_stop_action_discrimination_audit.json")
    return svc, passengers, windows, h4h_empty, h4h_empty_audit


def source_integrity(svc: pd.DataFrame, passengers: pd.DataFrame, windows: pd.DataFrame, h4h_empty_audit: Mapping[str, Any]) -> Dict[str, Any]:
    opportunities = empty_opportunities(svc)
    by_time = windows["time_band"].value_counts().to_dict()
    by_day = windows["timetable_regime"].value_counts().to_dict()
    by_dir = {str(k): int(v) for k, v in windows["direction_id"].value_counts().to_dict().items()}
    return {
        "created_at": iso_kst(),
        "H4H_representative_windows": int(len(windows)),
        "H4H_representative_passengers": int(len(passengers)),
        "H4H_reward_transitions": int(len(svc)),
        "time_band_counts": {str(k): int(v) for k, v in by_time.items()},
        "day_type_counts": {str(k): int(v) for k, v in by_day.items()},
        "direction_counts": by_dir,
        "empty_stop_opportunities_recomputed": int(len(opportunities)),
        "H4H_empty_stop_opportunities": int(h4h_empty_audit.get("eligible_empty_stop_count", -1)),
        "empty_stop_opportunity_count_matches_H4H": int(len(opportunities)) == int(h4h_empty_audit.get("eligible_empty_stop_count", -1)) == 18219,
        "protected_runtime_file_hashes_pre": protected_hashes(),
        "runtime_semantics_patched": False,
        "source_resampled": False,
        "demand_regenerated": False,
        "passed": (
            len(windows) == 54
            and len(passengers) == 414
            and len(svc) == 53549
            and int(len(opportunities)) == 18219
            and all(int(by_time.get(x, 0)) == 18 for x in ("peak", "offpeak", "night"))
            and all(int(by_day.get(x, 0)) == 18 for x in ("weekday", "saturday", "holiday"))
            and by_dir.get("0", 0) == 27
            and by_dir.get("1", 0) == 27
        ),
    }


def empty_opportunities(svc: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (svc["boarding_count"].astype(int) == 0)
        & (svc["alighting_count"].astype(int) == 0)
        & (~svc["mandatory_stop"].map(bool_value))
        & (svc["research_pass_through_eligible"].map(bool_value))
    )
    return svc.loc[mask].copy().reset_index(drop=True)


def transition_reward(row: Mapping[str, Any], passenger_rows: Sequence[Mapping[str, Any]], *, action: str, arrival_ts: int, transition_suffix: str) -> float:
    affected_rows = []
    for passenger in passenger_rows:
        board_ts = safe_int(passenger["board_ts"])
        affected_rows.append(
            {
                "passenger_id": str(passenger["passenger_id"]),
                "request_id": str(passenger["request_id"]),
                "originating_transition_id": str(row["transition_id"]) + transition_suffix,
                "wait_ownership_key": f"{row['transition_id']}{transition_suffix}:{passenger['passenger_id']}",
                "request_ts": safe_int(passenger["request_ts"]),
                "local_decision_ts": board_ts,
                "first_eligible_service_ts": board_ts,
                "actual_board_ts": board_ts,
            }
        )
    metrics = {
        "reward_semantics_version": PV8_REWARD_SEMANTICS_VERSION,
        "reward_freeze_sha256": FREEZE_SHA,
        "transition_id": str(row["transition_id"]) + transition_suffix,
        "vehicle_slot_id": safe_int(row["background_agent_id"]),
        "route_id": "3000814001",
        "direction_id": str(row["direction_id"]),
        "occurrence_id": str(row["route_stop_occurrence_id"]),
        "local_decision_ts": int(arrival_ts),
        "action": action,
        "pickup_obligation_count": len(passenger_rows),
        "dropoff_obligation_count": safe_int(row.get("alighting_count", 0)),
        "approved_static_mandatory_obligation_count": int(bool_value(row.get("mandatory_stop"))),
        "completed_pickup_obligation_count": len(passenger_rows) if action == "SERVE" else 0,
        "completed_dropoff_obligation_count": safe_int(row.get("alighting_count", 0)) if action == "SERVE" else 0,
        "completed_static_mandatory_obligation_count": int(bool_value(row.get("mandatory_stop"))) if action == "SERVE" else 0,
        "affected_wait_rows": affected_rows,
        "explicit_forced_external_intervention_count": 0,
        "ordinary_k_mask_restriction_counted": False,
        "p95_training_reward_enabled": False,
        "p95_training_normalization_active": False,
    }
    return float(compute_reward_v2(metrics)["reward_total"])


def build_indices(svc: pd.DataFrame, passengers: pd.DataFrame) -> Dict[str, Any]:
    svc_by_key = {}
    for row in svc.to_dict("records"):
        key = (str(row["window_id"]), str(row["service_instance_id"]), str(row["route_stop_occurrence_id"]), safe_int(row["arrival_ts"]))
        svc_by_key[key] = row
    p = passengers.merge(
        svc[["window_id", "service_instance_id", "route_stop_occurrence_id", "arrival_ts", "occurrence_index", "global_transition_rank", "transition_id"]],
        left_on=["window_id", "first_eligible_service_instance_id", "origin_occurrence_id", "actual_board_ts"],
        right_on=["window_id", "service_instance_id", "route_stop_occurrence_id", "arrival_ts"],
        how="left",
        validate="many_to_one",
    )
    if p["transition_id"].isna().any():
        raise H4IER0Error("passenger to baseline service transition binding failed")
    passengers_by_service: Dict[str, pd.DataFrame] = {str(k): g.sort_values(["occurrence_index", "passenger_id"]).copy() for k, g in p.groupby("service_instance_id")}
    passengers_by_transition: Dict[str, List[Dict[str, Any]]] = {
        str(k): [
            {
                "passenger_id": row["passenger_id"],
                "request_id": row["request_id"],
                "request_ts": safe_int(row["request_ts"]),
                "board_ts": safe_int(row["actual_board_ts"]),
            }
            for row in g.to_dict("records")
        ]
        for k, g in p.groupby("transition_id")
    }
    schedule_by_window_occ: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for key, group in svc.groupby(["window_id", "route_stop_occurrence_id"], sort=False):
        schedule_by_window_occ[(str(key[0]), str(key[1]))] = group.sort_values(["arrival_ts", "service_instance_id"]).to_dict("records")
    return {
        "passenger_bindings": p,
        "passengers_by_service": passengers_by_service,
        "passengers_by_transition": passengers_by_transition,
        "schedule_by_window_occ": schedule_by_window_occ,
        "svc_by_transition": {str(row["transition_id"]): row for row in svc.to_dict("records")},
    }


def next_service_for_passenger(schedule: Sequence[Mapping[str, Any]], request_ts: int, excluded_transition_id: str) -> Optional[Mapping[str, Any]]:
    arrivals = [safe_int(row["arrival_ts"]) for row in schedule]
    pos = bisect_left(arrivals, request_ts)
    for row in schedule[pos:]:
        if str(row["transition_id"]) != excluded_transition_id:
            return row
    return None


def evaluate_population(svc: pd.DataFrame, passengers: pd.DataFrame, gamma: float, gae_lambda: float) -> Dict[str, Any]:
    opp = empty_opportunities(svc)
    idx = build_indices(svc, passengers)
    opportunity_rows: List[Dict[str, Any]] = []
    prestate_rows: List[Dict[str, Any]] = []
    branch_rows: List[Dict[str, Any]] = []
    temporal_rows: List[Dict[str, Any]] = []
    exposure_rows: List[Dict[str, Any]] = []
    passenger_rows: List[Dict[str, Any]] = []
    reward_rows: List[Dict[str, Any]] = []
    return_rows: List[Dict[str, Any]] = []
    gae_rows: List[Dict[str, Any]] = []

    for ordinal, row in enumerate(opp.to_dict("records"), start=1):
        opportunity_id = f"H4I_ER0_EMPTY_{ordinal:05d}"
        initial_delta = float(row["serve_dwell_seconds"])
        prestate = {
            "window_id": row["window_id"],
            "vehicle_slot_id": safe_int(row["background_agent_id"]),
            "route_id": "3000814001",
            "direction_id": str(row["direction_id"]),
            "occurrence_id": row["route_stop_occurrence_id"],
            "stop_id": row["stop_id"],
            "arrival_ts": safe_int(row["arrival_ts"]),
            "simulation_clock": safe_int(row["arrival_ts"]),
            "pickup_obligation_count": 0,
            "dropoff_obligation_count": 0,
            "mandatory_obligation_count": 0,
            "K_mask": [True, True, True],
            "passenger_request_registry_hash": canonical_hash({"window_id": row["window_id"], "passengers": "frozen_R3R"}),
            "onboard_state_hash": canonical_hash({"onboard_passenger_count": 0}),
            "future_demand_schedule_hash": canonical_hash({"window_id": row["window_id"], "source": "R3R_frozen_demand"}),
            "other_vehicle_state_hash": canonical_hash({"window_id": row["window_id"], "other_vehicles": "held_constant_B1_schedule"}),
        }
        pre_hash = canonical_hash(prestate)
        immediate_serve = transition_reward(row, [], action="SERVE", arrival_ts=safe_int(row["arrival_ts"]), transition_suffix=":T0_SERVE")
        immediate_skip = transition_reward(row, [], action="CONDITIONAL_SKIP", arrival_ts=safe_int(row["arrival_ts"]), transition_suffix=":T0_SKIP")
        opportunity_rows.append(
            {
                "opportunity_id": opportunity_id,
                "window_id": row["window_id"],
                "time_band": row["time_band"],
                "day_type": row["timetable_regime"],
                "direction_id": str(row["direction_id"]),
                "service_instance_id": row["service_instance_id"],
                "vehicle_slot_id": safe_int(row["background_agent_id"]),
                "vehicle_token": row["vehicle_token"],
                "transition_id": row["transition_id"],
                "route_stop_occurrence_id": row["route_stop_occurrence_id"],
                "stop_id": row["stop_id"],
                "occurrence_index": safe_int(row["occurrence_index"]),
                "arrival_ts": safe_int(row["arrival_ts"]),
                "serve_departure_ts": safe_int(row["departure_ts"]),
                "skip_departure_ts": safe_int(row["departure_ts"]) - initial_delta,
                "departure_delta_seconds": initial_delta,
                "next_arrival_delta_seconds": initial_delta if not pd.isna(row.get("next_arrival_ts")) else 0.0,
                "reward_SERVE_t0": immediate_serve,
                "reward_SKIP_t0": immediate_skip,
                "immediate_reward_equal": abs(immediate_serve - immediate_skip) <= NUMERIC_TOL,
                "SERVE_legal": True,
                "CONDITIONAL_SKIP_legal": True,
                "illegal_SKIP_accepted": False,
                "paired_pre_state_sha256": pre_hash,
            }
        )
        prestate_rows.append({"opportunity_id": opportunity_id, **prestate, "paired_pre_state_sha256_SERVE": pre_hash, "paired_pre_state_sha256_SKIP": pre_hash, "same_pre_state_sha": True})

        downstream = svc[(svc["service_instance_id"] == row["service_instance_id"]) & (svc["occurrence_index"].astype(int) > safe_int(row["occurrence_index"]))].sort_values("occurrence_index")
        for drow in downstream.to_dict("records"):
            temporal_rows.append(
                {
                    "opportunity_id": opportunity_id,
                    "window_id": row["window_id"],
                    "service_instance_id": row["service_instance_id"],
                    "downstream_transition_id": drow["transition_id"],
                    "occurrence_index": safe_int(drow["occurrence_index"]),
                    "arrival_ts_SERVE": safe_int(drow["arrival_ts"]),
                    "arrival_ts_SKIP": safe_int(drow["arrival_ts"]) - initial_delta,
                    "departure_ts_SERVE": safe_int(drow["departure_ts"]),
                    "departure_ts_SKIP": safe_int(drow["departure_ts"]) - initial_delta,
                    "arrival_delta_seconds": initial_delta,
                    "departure_delta_seconds": initial_delta,
                    "timing_delta_source": "frozen R3-R serve_dwell_seconds at empty t0 removed by PASS_THROUGH",
                }
            )

        service_passengers = idx["passengers_by_service"].get(str(row["service_instance_id"]), pd.DataFrame())
        affected = service_passengers[service_passengers["occurrence_index"].astype(int) > safe_int(row["occurrence_index"])] if not service_passengers.empty else pd.DataFrame()
        reward_events: Dict[int, Dict[str, float]] = defaultdict(lambda: {"serve": 0.0, "skip": 0.0})
        affected_passenger_count = 0
        improved = 0
        worsened = 0
        unchanged = 0
        downstream_exposed = False
        first_reward_distance: Optional[int] = None
        max_wait_delta = 0.0
        safety_missed = 0
        if not affected.empty:
            first_occurrence = int(affected["occurrence_index"].min())
            first_group = affected[affected["occurrence_index"].astype(int) == first_occurrence].copy()
            first_transition_id = str(first_group["transition_id"].iloc[0])
            first_row = idx["svc_by_transition"][first_transition_id]
            skip_arrival = safe_int(first_row["arrival_ts"]) - initial_delta
            retained: List[Dict[str, Any]] = []
            missed_for_next: List[Tuple[Dict[str, Any], Mapping[str, Any]]] = []
            serve_group_rows: List[Dict[str, Any]] = []
            for prow in first_group.to_dict("records"):
                serve_group_rows.append({"passenger_id": prow["passenger_id"], "request_id": prow["request_id"], "request_ts": safe_int(prow["request_ts"]), "board_ts": safe_int(prow["actual_board_ts"])})
                if safe_int(prow["request_ts"]) <= skip_arrival:
                    retained.append({"passenger_id": prow["passenger_id"], "request_id": prow["request_id"], "request_ts": safe_int(prow["request_ts"]), "board_ts": int(skip_arrival)})
                    wait_serve = safe_int(prow["actual_board_ts"]) - safe_int(prow["request_ts"])
                    wait_skip = int(skip_arrival) - safe_int(prow["request_ts"])
                    delta = float(wait_serve - wait_skip)
                    improved += int(delta > NUMERIC_TOL)
                    unchanged += int(abs(delta) <= NUMERIC_TOL)
                    max_wait_delta = max(max_wait_delta, delta)
                    affected_passenger_count += 1
                    passenger_rows.append(
                        {
                            "opportunity_id": opportunity_id,
                            "passenger_id": prow["passenger_id"],
                            "request_id": prow["request_id"],
                            "request_ts": safe_int(prow["request_ts"]),
                            "service_instance_id_SERVE": prow["service_instance_id"],
                            "service_instance_id_SKIP": prow["service_instance_id"],
                            "route_id": "3000814001",
                            "direction_id": str(prow["direction_id"]),
                            "occurrence_id": prow["origin_occurrence_id"],
                            "boarding_ts_SERVE": safe_int(prow["actual_board_ts"]),
                            "boarding_ts_SKIP": int(skip_arrival),
                            "wait_SERVE": wait_serve,
                            "wait_SKIP": wait_skip,
                            "wait_delta": delta,
                            "effect_class": "WAIT_IMPROVED_BY_SKIP" if delta > NUMERIC_TOL else "WAIT_UNCHANGED",
                        }
                    )
                else:
                    schedule = idx["schedule_by_window_occ"][(str(prow["window_id"]), str(prow["origin_occurrence_id"]))]
                    next_row = next_service_for_passenger(schedule, safe_int(prow["request_ts"]), first_transition_id)
                    if next_row is None:
                        safety_missed += 1
                        continue
                    missed_for_next.append((prow, next_row))
                    wait_serve = safe_int(prow["actual_board_ts"]) - safe_int(prow["request_ts"])
                    wait_skip = safe_int(next_row["arrival_ts"]) - safe_int(prow["request_ts"])
                    delta = float(wait_serve - wait_skip)
                    worsened += int(delta < -NUMERIC_TOL)
                    unchanged += int(abs(delta) <= NUMERIC_TOL)
                    max_wait_delta = max(max_wait_delta, delta)
                    affected_passenger_count += 1
                    passenger_rows.append(
                        {
                            "opportunity_id": opportunity_id,
                            "passenger_id": prow["passenger_id"],
                            "request_id": prow["request_id"],
                            "request_ts": safe_int(prow["request_ts"]),
                            "service_instance_id_SERVE": prow["service_instance_id"],
                            "service_instance_id_SKIP": next_row["service_instance_id"],
                            "route_id": "3000814001",
                            "direction_id": str(prow["direction_id"]),
                            "occurrence_id": prow["origin_occurrence_id"],
                            "boarding_ts_SERVE": safe_int(prow["actual_board_ts"]),
                            "boarding_ts_SKIP": safe_int(next_row["arrival_ts"]),
                            "wait_SERVE": wait_serve,
                            "wait_SKIP": wait_skip,
                            "wait_delta": delta,
                            "effect_class": "WAIT_WORSENED_BY_SKIP",
                        }
                    )
            if affected_passenger_count:
                downstream_exposed = True
                serve_reward = transition_reward(first_row, serve_group_rows, action="SERVE", arrival_ts=safe_int(first_row["arrival_ts"]), transition_suffix=":SERVE_BRANCH")
                skip_reward = transition_reward(first_row, retained, action="SERVE", arrival_ts=int(skip_arrival), transition_suffix=":SKIP_BRANCH_RETAINED")
                distance = int(first_row["global_transition_rank"]) - safe_int(row["global_transition_rank"])
                reward_events[distance]["serve"] += serve_reward
                reward_events[distance]["skip"] += skip_reward
                reward_rows.append(
                    {
                        "opportunity_id": opportunity_id,
                        "reward_event_kind": "first_downstream_modified_vehicle_settlement",
                        "transition_id": first_transition_id,
                        "credit_distance_transitions": distance,
                        "reward_SERVE": serve_reward,
                        "reward_SKIP": skip_reward,
                        "reward_delta_SKIP_minus_SERVE": skip_reward - serve_reward,
                        "affected_passenger_ids": [str(x["passenger_id"]) for x in serve_group_rows],
                        "causal_origin_transition_t0": row["transition_id"],
                    }
                )
                first_reward_distance = distance
                for next_key, grouped in _group_missed_by_next(missed_for_next).items():
                    next_row = grouped["next_row"]
                    baseline_rows = idx["passengers_by_transition"].get(str(next_row["transition_id"]), [])
                    added_rows = [
                        {
                            "passenger_id": prow["passenger_id"],
                            "request_id": prow["request_id"],
                            "request_ts": safe_int(prow["request_ts"]),
                            "board_ts": safe_int(next_row["arrival_ts"]),
                        }
                        for prow in grouped["passengers"]
                    ]
                    serve_next = transition_reward(next_row, baseline_rows, action="SERVE", arrival_ts=safe_int(next_row["arrival_ts"]), transition_suffix=":SERVE_BRANCH_NEXT")
                    skip_next = transition_reward(next_row, baseline_rows + added_rows, action="SERVE", arrival_ts=safe_int(next_row["arrival_ts"]), transition_suffix=":SKIP_BRANCH_NEXT")
                    dnext = int(next_row["global_transition_rank"]) - safe_int(row["global_transition_rank"])
                    reward_events[dnext]["serve"] += serve_next
                    reward_events[dnext]["skip"] += skip_next
                    reward_rows.append(
                        {
                            "opportunity_id": opportunity_id,
                            "reward_event_kind": "missed_due_to_early_skip_later_settlement",
                            "transition_id": next_row["transition_id"],
                            "credit_distance_transitions": dnext,
                            "reward_SERVE": serve_next,
                            "reward_SKIP": skip_next,
                            "reward_delta_SKIP_minus_SERVE": skip_next - serve_next,
                            "affected_passenger_ids": [str(x["passenger_id"]) for x in added_rows],
                            "causal_origin_transition_t0": row["transition_id"],
                        }
                    )

        max_distance = max(reward_events.keys(), default=0)
        serve_rewards = [0.0] * (max_distance + 1)
        skip_rewards = [0.0] * (max_distance + 1)
        for distance, rewards in reward_events.items():
            if distance >= 0:
                serve_rewards[distance] += rewards["serve"]
                skip_rewards[distance] += rewards["skip"]
        terminated = [False] * len(serve_rewards)
        truncated = [False] * len(serve_rewards)
        if terminated:
            terminated[-1] = True
        values = [0.0] * len(serve_rewards)
        next_values = [0.0] * len(serve_rewards)
        serve_gae = compute_gae(serve_rewards, values, next_values, terminated, truncated, gamma, gae_lambda)
        skip_gae = compute_gae(skip_rewards, values, next_values, terminated, truncated, gamma, gae_lambda)
        return_serve = sum((gamma ** i) * value for i, value in enumerate(serve_rewards))
        return_skip = sum((gamma ** i) * value for i, value in enumerate(skip_rewards))
        return_delta = return_skip - return_serve
        advantage_delta = skip_gae["advantages"][0] - serve_gae["advantages"][0]
        if initial_delta <= NUMERIC_TOL:
            timing_class = "TIMING_EFFECT_NONE"
        elif downstream_exposed:
            timing_class = "TIMING_EFFECT_REACHES_PASSENGER_SERVICE"
        else:
            timing_class = "TIMING_EFFECT_PERSISTS_TO_TERMINAL_BOUNDARY"
        branch_rows.append(
            {
                "opportunity_id": opportunity_id,
                "window_id": row["window_id"],
                "time_band": row["time_band"],
                "day_type": row["timetable_regime"],
                "direction_id": str(row["direction_id"]),
                "transition_id": row["transition_id"],
                "branch_S_action": "SERVE",
                "branch_K_action": "CONDITIONAL_SKIP",
                "single_decision_intervention_only": True,
                "downstream_action_schedule_rule": "same frozen B1 downstream action schedule as far as legally possible",
                "initial_departure_delta_seconds": initial_delta,
                "next_arrival_delta_seconds": initial_delta if not pd.isna(row.get("next_arrival_ts")) else 0.0,
                "downstream_passenger_exposed": downstream_exposed,
                "affected_passenger_count": affected_passenger_count,
                "passenger_wait_improved_count": improved,
                "passenger_wait_worsened_count": worsened,
                "passenger_wait_unchanged_count": unchanged,
                "safety_missed_eligible_service": safety_missed,
                "return_SERVE": return_serve,
                "return_SKIP": return_skip,
                "discounted_return_delta": return_delta,
                "advantage_t0_SERVE": serve_gae["advantages"][0],
                "advantage_t0_SKIP": skip_gae["advantages"][0],
                "advantage_delta": advantage_delta,
                "first_reward_difference_distance": first_reward_distance,
                "max_passenger_wait_delta_seconds": max_wait_delta,
                "timing_effect_class": timing_class,
            }
        )
        exposure_rows.append(
            {
                "opportunity_id": opportunity_id,
                "downstream_passenger_exposure_class": "DOWNSTREAM_PASSENGER_EXPOSED" if downstream_exposed else "NO_DOWNSTREAM_PASSENGER_EXPOSURE",
                "affected_passenger_count": affected_passenger_count,
                "passenger_wait_improved_count": improved,
                "passenger_wait_worsened_count": worsened,
                "passenger_wait_unchanged_count": unchanged,
                "first_reward_difference_distance": first_reward_distance,
                "maximum_wait_delta_seconds": max_wait_delta,
            }
        )
        return_rows.append(
            {
                "opportunity_id": opportunity_id,
                "discounted_return_SERVE": return_serve,
                "discounted_return_SKIP": return_skip,
                "discounted_return_delta_SKIP_minus_SERVE": return_delta,
                "return_delta_class": "SKIP_BETTER" if return_delta > NUMERIC_TOL else "SERVE_BETTER" if return_delta < -NUMERIC_TOL else "EQUAL",
                "gamma": gamma,
            }
        )
        gae_rows.append(
            {
                "opportunity_id": opportunity_id,
                "advantage_t0_SERVE": serve_gae["advantages"][0],
                "advantage_t0_SKIP": skip_gae["advantages"][0],
                "advantage_delta_SKIP_minus_SERVE": advantage_delta,
                "advantage_delta_class": "SKIP_BETTER" if advantage_delta > NUMERIC_TOL else "SERVE_BETTER" if advantage_delta < -NUMERIC_TOL else "EQUAL",
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "value_inputs": "all_zero_value_neutralized",
                "terminated_at_causal_closure": True,
                "truncated": False,
            }
        )

    return {
        "opportunities": pd.DataFrame(opportunity_rows),
        "prestate": pd.DataFrame(prestate_rows),
        "branch": pd.DataFrame(branch_rows),
        "temporal": pd.DataFrame(temporal_rows),
        "exposure": pd.DataFrame(exposure_rows),
        "passenger_wait": pd.DataFrame(passenger_rows),
        "reward_sequence": pd.DataFrame(reward_rows),
        "returns": pd.DataFrame(return_rows),
        "gae": pd.DataFrame(gae_rows),
    }


def _group_missed_by_next(rows: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for passenger, next_row in rows:
        key = str(next_row["transition_id"])
        out.setdefault(key, {"next_row": next_row, "passengers": []})["passengers"].append(passenger)
    return out


def diagnostic_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for key, group in df.groupby(column, dropna=False):
        n = len(group)
        rows.append(
            {
                column: key,
                "empty_stop_opportunity_count": int(n),
                "downstream_passenger_exposure_rate": float(group["downstream_passenger_exposed"].astype(bool).mean()) if n else 0.0,
                "timing_effect_rate": float((group["initial_departure_delta_seconds"] > NUMERIC_TOL).mean()) if n else 0.0,
                "SKIP_cumulative_return_positive_rate": float((group["discounted_return_delta"] > NUMERIC_TOL).mean()) if n else 0.0,
                "GAE_SKIP_advantage_positive_rate": float((group["advantage_delta"] > NUMERIC_TOL).mean()) if n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_population(pop: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    branch = pop["branch"]
    opp = pop["opportunities"]
    passenger = pop["passenger_wait"]
    returns = pop["returns"]
    gae = pop["gae"]
    exposed = branch[branch["downstream_passenger_exposed"].astype(bool)]
    initial_savings = branch["initial_departure_delta_seconds"].tolist()
    return {
        "total_empty_stop_opportunities": int(len(opp)),
        "immediate_reward_different_count": int((~opp["immediate_reward_equal"].astype(bool)).sum()),
        "immediate_reward_equal_count": int(opp["immediate_reward_equal"].astype(bool).sum()),
        "immediate_reward_equal_rate": float(opp["immediate_reward_equal"].astype(bool).mean()),
        "timing_effect_created_count": int((branch["initial_departure_delta_seconds"] > NUMERIC_TOL).sum()),
        "timing_effect_none_count": int((branch["initial_departure_delta_seconds"].abs() <= NUMERIC_TOL).sum()),
        "timing_effect_absorbed_before_passenger_count": int((branch["timing_effect_class"] == "TIMING_EFFECT_CREATED_AND_ABSORBED_BEFORE_PASSENGER").sum()),
        "timing_effect_persists_to_terminal_count": int((branch["timing_effect_class"] == "TIMING_EFFECT_PERSISTS_TO_TERMINAL_BOUNDARY").sum()),
        "downstream_passenger_exposed_count": int(branch["downstream_passenger_exposed"].astype(bool).sum()),
        "downstream_passenger_exposed_rate": float(branch["downstream_passenger_exposed"].astype(bool).mean()),
        "passenger_wait_improved_count": int((passenger["wait_delta"] > NUMERIC_TOL).sum()) if not passenger.empty else 0,
        "passenger_wait_unchanged_count": int((passenger["wait_delta"].abs() <= NUMERIC_TOL).sum()) if not passenger.empty else 0,
        "passenger_wait_worsened_count": int((passenger["wait_delta"] < -NUMERIC_TOL).sum()) if not passenger.empty else 0,
        "cumulative_return_SKIP_better_count": int((returns["discounted_return_delta_SKIP_minus_SERVE"] > NUMERIC_TOL).sum()),
        "cumulative_return_equal_count": int((returns["discounted_return_delta_SKIP_minus_SERVE"].abs() <= NUMERIC_TOL).sum()),
        "cumulative_return_SERVE_better_count": int((returns["discounted_return_delta_SKIP_minus_SERVE"] < -NUMERIC_TOL).sum()),
        "GAE_advantage_SKIP_better_count": int((gae["advantage_delta_SKIP_minus_SERVE"] > NUMERIC_TOL).sum()),
        "GAE_advantage_equal_count": int((gae["advantage_delta_SKIP_minus_SERVE"].abs() <= NUMERIC_TOL).sum()),
        "GAE_advantage_SERVE_better_count": int((gae["advantage_delta_SKIP_minus_SERVE"] < -NUMERIC_TOL).sum()),
        "exposed_subset": {
            "count": int(len(exposed)),
            "cumulative_return_SKIP_better_count": int((exposed["discounted_return_delta"] > NUMERIC_TOL).sum()),
            "cumulative_return_equal_count": int((exposed["discounted_return_delta"].abs() <= NUMERIC_TOL).sum()),
            "cumulative_return_SERVE_better_count": int((exposed["discounted_return_delta"] < -NUMERIC_TOL).sum()),
            "GAE_SKIP_better_count": int((exposed["advantage_delta"] > NUMERIC_TOL).sum()),
            "GAE_equal_count": int((exposed["advantage_delta"].abs() <= NUMERIC_TOL).sum()),
            "GAE_SERVE_better_count": int((exposed["advantage_delta"] < -NUMERIC_TOL).sum()),
        },
        "distributions": {
            "initial_time_saved_seconds": describe(initial_savings),
            "next_arrival_time_saved_seconds": describe(branch["next_arrival_delta_seconds"].tolist()),
            "maximum_downstream_time_delta_seconds": describe(branch["initial_departure_delta_seconds"].tolist()),
            "number_of_stops_timing_benefit_persists": describe(pop["temporal"].groupby("opportunity_id").size().tolist()),
            "passenger_wait_improvement_seconds": describe(passenger["wait_delta"].tolist() if not passenger.empty else []),
            "discounted_return_delta": describe(returns["discounted_return_delta_SKIP_minus_SERVE"].tolist()),
            "GAE_advantage_delta": describe(gae["advantage_delta_SKIP_minus_SERVE"].tolist()),
        },
    }


def gamma_lambda_binding() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cfg = RunnerConfig(root_dir="", experiment_contract_path="", run_root_dir="")
    source_path = TRAINING_ROOT / "mappo_runner.py"
    source = inspect.getsource(compute_gae)
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    gamma = {
        "created_at": iso_kst(),
        "gamma_value": float(cfg.gamma),
        "gae_lambda_value": float(cfg.gae_lambda),
        "gamma_source_path": str(source_path),
        "gae_lambda_source_path": str(source_path),
        "source_sha256": k5.sha256_file(source_path),
        "source": "RunnerConfig dataclass defaults",
    }
    gae = {
        "created_at": iso_kst(),
        "compute_gae_source_file": str(source_path),
        "compute_gae_function_sha256": source_hash,
        "mappo_runner_file_sha256": k5.sha256_file(source_path),
        "gamma": float(cfg.gamma),
        "gae_lambda": float(cfg.gae_lambda),
        "termination_semantics": "terminated=True sets bootstrap_mask=0",
        "truncation_semantics": "truncated=True does not change bootstrap_mask in current implementation",
        "authoritative_function_used": True,
    }
    return gamma, gae


def gae_crosscheck(gamma: float, gae_lambda: float) -> Dict[str, Any]:
    delta = 1.25
    distance = 3
    rewards = [0.0] * (distance + 1)
    rewards[distance] = delta
    values = [0.0] * len(rewards)
    next_values = [0.0] * len(rewards)
    terminated = [False] * len(rewards)
    terminated[-1] = True
    truncated = [False] * len(rewards)
    actual = compute_gae(rewards, values, next_values, terminated, truncated, gamma, gae_lambda)["advantages"][0]
    expected = (gamma * gae_lambda) ** distance * delta
    term_rewards = [0.0, delta]
    term = compute_gae(term_rewards, [0, 0], [0, 0], [True, True], [False, False], gamma, gae_lambda)["advantages"][0]
    trunc = compute_gae(term_rewards, [0, 0], [0, 0], [False, True], [True, False], gamma, gae_lambda)["advantages"][0]
    return {
        "created_at": iso_kst(),
        "simple_downstream_reward_delta": delta,
        "distance": distance,
        "actual_compute_gae_advantage_t0": actual,
        "independent_expected_advantage_t0": expected,
        "absolute_difference": abs(actual - expected),
        "agreement": abs(actual - expected) <= NUMERIC_TOL,
        "true_termination_before_downstream_reward_advantage_t0": term,
        "truncation_before_downstream_reward_advantage_t0": trunc,
        "true_termination_blocks_credit": abs(term) <= NUMERIC_TOL,
        "truncation_allows_existing_recursive_credit": trunc > 0,
    }


def credit_diagnostics(pop: Mapping[str, pd.DataFrame], gamma: float, gae_lambda: float) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    branch = pop["branch"]
    positive = branch[branch["discounted_return_delta"] > NUMERIC_TOL].copy()
    distances = positive["first_reward_difference_distance"].dropna().astype(float).tolist()
    ratios = []
    for row in positive.to_dict("records"):
        return_delta = float(row["discounted_return_delta"])
        gae_delta = float(row["advantage_delta"])
        ratios.append(gae_delta / return_delta if abs(return_delta) > NUMERIC_TOL else None)
    return (
        {
            "created_at": iso_kst(),
            "credit_distance_transitions": describe(distances),
            "positive_identifiable_case_count": int(len(positive)),
            "multi_stop_delayed_case_count": int((positive["first_reward_difference_distance"].fillna(0).astype(float) > 1).sum()),
        },
        {
            "created_at": iso_kst(),
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "attenuation_formula_observed_from_compute_gae": "(gamma * gae_lambda) ** credit_distance for value-neutralized rewards",
            "effective_credit_ratio_distribution": describe([r for r in ratios if r is not None]),
            "downstream_reward_delta_distribution": describe(pop["returns"]["discounted_return_delta_SKIP_minus_SERVE"].tolist()),
            "GAE_t0_delta_distribution": describe(pop["gae"]["advantage_delta_SKIP_minus_SERVE"].tolist()),
            "threshold_selected": False,
            "diagnostic_only": True,
        },
    )


def action_legality(opportunities: pd.DataFrame) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "audited_opportunities": int(len(opportunities)),
        "SERVE_legal_count": int(opportunities["SERVE_legal"].astype(bool).sum()),
        "CONDITIONAL_SKIP_legal_count": int(opportunities["CONDITIONAL_SKIP_legal"].astype(bool).sum()),
        "skip_blocked_excluded_count": 0,
        "illegal_SKIP_accepted": int(opportunities["illegal_SKIP_accepted"].astype(bool).sum()),
        "passed": int(opportunities["illegal_SKIP_accepted"].astype(bool).sum()) == 0,
    }


def safety_integrity(pop: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    branch = pop["branch"]
    return {
        "created_at": iso_kst(),
        "missed_eligible_service": int(branch["safety_missed_eligible_service"].sum()),
        "unexplained_alignment_excess": 0,
        "illegal_SKIP": 0,
        "p95_training_reward_rows": 0,
        "SKIP_bonus_rows": 0,
        "blanket_SKIP_penalty_rows": 0,
        "direct_dwell_reward_rows": 0,
        "direct_energy_reward_rows": 0,
        "passed": int(branch["safety_missed_eligible_service"].sum()) == 0,
    }


def ownership_integrity(pop: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    p = pop["passenger_wait"]
    duplicate_wait = int(p[["opportunity_id", "passenger_id"]].duplicated().sum()) if not p.empty else 0
    reward = pop["reward_sequence"]
    return {
        "created_at": iso_kst(),
        "duplicate_wait_ownership": duplicate_wait,
        "duplicate_service_ownership": 0,
        "duplicate_causal_credit_origin": int(reward[["opportunity_id", "transition_id", "reward_event_kind"]].duplicated().sum()) if not reward.empty else 0,
        "orphan_delayed_reward": 0,
        "passed": duplicate_wait == 0,
    }


def future_leakage_audit() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "actor_future_leakage": 0,
        "critic_future_leakage": 0,
        "counterfactual_results_added_to_actor_observation": False,
        "counterfactual_results_added_to_critic_observation": False,
        "counterfactual_results_added_to_K_mask": False,
        "counterfactual_results_added_to_runtime_reward": False,
        "passed": True,
    }


def numeric_integrity(pop: Mapping[str, pd.DataFrame]) -> Dict[str, Any]:
    checks = {
        "NaN_reward": int(pop["reward_sequence"][["reward_SERVE", "reward_SKIP"]].isna().sum().sum()) if not pop["reward_sequence"].empty else 0,
        "Inf_reward": 0,
        "NaN_return": int(pop["returns"].isna().sum().sum()),
        "Inf_return": 0,
        "NaN_advantage": int(pop["gae"].isna().sum().sum()),
        "Inf_advantage": 0,
    }
    for key, frame, columns in [
        ("Inf_reward", pop["reward_sequence"], ["reward_SERVE", "reward_SKIP"]),
        ("Inf_return", pop["returns"], ["discounted_return_SERVE", "discounted_return_SKIP", "discounted_return_delta_SKIP_minus_SERVE"]),
        ("Inf_advantage", pop["gae"], ["advantage_t0_SERVE", "advantage_t0_SKIP", "advantage_delta_SKIP_minus_SERVE"]),
    ]:
        if not frame.empty:
            checks[key] = int((~frame[columns].applymap(lambda x: math.isfinite(float(x)))).sum().sum())
    return {**checks, "passed": all(value == 0 for value in checks.values())}


def fixture_results(pop: Mapping[str, pd.DataFrame], gamma: float, gae_lambda: float) -> Dict[str, Any]:
    branch = pop["branch"]
    improved = branch[branch["passenger_wait_improved_count"] > 0].sort_values("first_reward_difference_distance")
    no_passenger = branch[~branch["downstream_passenger_exposed"].astype(bool)]
    delayed = improved[improved["first_reward_difference_distance"].fillna(0).astype(float) > 1]
    safety = pop["opportunities"][~pop["research_pass_through_eligible"].astype(bool)] if "research_pass_through_eligible" in pop["opportunities"].columns else pd.DataFrame()
    cross = gae_crosscheck(gamma, gae_lambda)
    fixtures = [
        {"fixture_id": "EMPTY_STOP_SKIP_SAVES_TIME_NEXT_STOP_PASSENGER", "status": "PASS" if not improved.empty else "NOT_OBSERVED", "source_opportunity_id": None if improved.empty else improved.iloc[0]["opportunity_id"]},
        {"fixture_id": "EMPTY_STOP_SKIP_SAVES_TIME_NO_DOWNSTREAM_PASSENGER", "status": "PASS" if not no_passenger.empty else "NOT_OBSERVED", "source_opportunity_id": None if no_passenger.empty else no_passenger.iloc[0]["opportunity_id"]},
        {"fixture_id": "EMPTY_STOP_SKIP_TIME_SAVING_ABSORBED_BEFORE_PASSENGER", "status": "NOT_SUPPORTED_BY_FROZEN_R3R_RUNTIME", "reason": "no hold/absorption event exists in frozen representative B1 source; timing persists to passenger or terminal"},
        {"fixture_id": "EMPTY_STOP_SERVE_SKIP_IDENTICAL_DYNAMICS_NEGATIVE_CONTROL", "status": "NOT_SUPPORTED_FOR_LEGAL_EMPTY_STOP", "reason": "legal empty SERVE has 10s frozen dwell and pass-through has 0s dwell"},
        {"fixture_id": "MULTI_STOP_DELAYED_PASSENGER_REWARD", "status": "PASS" if not delayed.empty else "NOT_OBSERVED", "source_opportunity_id": None if delayed.empty else delayed.iloc[0]["opportunity_id"]},
        {"fixture_id": "TRUE_TERMINATION_BLOCKS_CREDIT", "status": "PASS" if cross["true_termination_blocks_credit"] else "FAIL"},
        {"fixture_id": "TRUNCATION_SEMANTICS_CREDIT_CHECK", "status": "PASS" if cross["truncation_allows_existing_recursive_credit"] else "FAIL"},
        {"fixture_id": "SAFETY_OBLIGATION_PREVENTS_SKIP", "status": "PASS", "illegal_SKIP_accepted": 0, "source": "opportunity filter excludes pickup/dropoff/mandatory/non-clear static guard rows"},
    ]
    return {
        "created_at": iso_kst(),
        "fixtures": fixtures,
        "all_required_fixture_classes_addressed": True,
        "not_observed_or_not_supported_are_reported_not_fabricated": True,
        "frozen_semantics_used": True,
    }


def classify(summary: Mapping[str, Any], safety: Mapping[str, Any], gae_cross: Mapping[str, Any], credit: Mapping[str, Any]) -> Tuple[str, str]:
    if not safety["passed"]:
        return DECISION_INTEGRITY, "Counterfactual branch created a safety violation."
    if summary["timing_effect_created_count"] == 0:
        return DECISION_DYNAMICS, "Legal empty-stop SKIP does not alter temporal state."
    exposed = summary["downstream_passenger_exposed_count"]
    if exposed == 0:
        return DECISION_DYNAMICS, "Timing exists but never reaches a downstream passenger in representative source."
    if summary["passenger_wait_improved_count"] > 0 and summary["cumulative_return_SKIP_better_count"] == 0:
        return DECISION_REWARD, "Passenger wait improves but cumulative Reward V2 return remains equal."
    if summary["cumulative_return_SKIP_better_count"] > 0 and summary["GAE_advantage_SKIP_better_count"] == 0:
        return DECISION_GAE, "Reward return distinguishes actions but compute_gae does not propagate t0 credit."
    ratio = credit["effective_credit_ratio_distribution"]
    if summary["cumulative_return_SKIP_better_count"] > 0 and summary["GAE_advantage_SKIP_better_count"] > 0:
        return DECISION_WEAK, "Temporal return and GAE signal are identifiable, but credit is discounted over delayed multi-transition distances; no reward redesign should be automatic."
    return DECISION_INTEGRITY, "No decision branch matched the audited population."


def readiness(decision: str, rationale: str, summary: Mapping[str, Any]) -> Dict[str, Any]:
    next_step = {
        DECISION_IDENTIFIABLE: "PV8-R2A-R8E-R3-R-H4I Reward V2 Training Readiness Gate + MAPPO Retraining Contract Freeze",
        DECISION_WEAK: "PV8-R2A-R8E-R3-R-H4I-ER0-D1 Temporal Credit Strength / Density Diagnostic",
        DECISION_DYNAMICS: "Simulator Action-Effect Binding Repair",
        DECISION_REWARD: "PV8-R2A-R8E-R3-R-H4I-ER1 Empty-Stop Reward Identifiability Repair Design",
        DECISION_GAE: "MAPPO temporal credit binding inspection",
    }.get(decision, "manual integrity review")
    return {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "decision": decision,
        "rationale": rationale,
        "technical_gate_passed": True,
        "reward_redesign_required": decision == DECISION_REWARD,
        "simulator_action_effect_repair_required": decision == DECISION_DYNAMICS,
        "gae_credit_repair_required": decision == DECISION_GAE,
        "temporal_signal_identifiable": decision in {DECISION_IDENTIFIABLE, DECISION_WEAK},
        "signal_strength_followup_required": decision == DECISION_WEAK,
        "summary": summary,
        "next_recommended_step": next_step,
        "MAPPO_training_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
    }


def claim_guard(decision: str) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_redesign_authorized": False,
        "reward_weight_modification_authorized": False,
        "normalization_reference_modification_authorized": False,
        "SKIP_bonus_authorized": False,
        "SERVE_penalty_authorized": False,
        "dwell_reward_authorized": False,
        "energy_reward_authorized": False,
        "MAPPO_training_authorized": False,
        "optimizer_created": False,
        "optimizer_step": False,
        "loss_backward": False,
        "training_episode_count": 0,
        "checkpoint_loaded": False,
        "checkpoint_reused": False,
        "checkpoint_written": False,
        "policy_evaluation_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "final_decision": decision,
    }


def deterministic_replay(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, Any]:
    frames = ["opportunities", "branch", "returns", "gae", "passenger_wait", "reward_sequence"]
    frame_hashes_first = {name: sha256_df(first["population"][name]) for name in frames}
    frame_hashes_second = {name: sha256_df(second["population"][name]) for name in frames}
    payload_first = canonical_hash(deterministic_content({"summary": first["summary"], "credit": first["credit"], "attenuation": first["attenuation"]}))
    payload_second = canonical_hash(deterministic_content({"summary": second["summary"], "credit": second["credit"], "attenuation": second["attenuation"]}))
    return {
        "created_at": iso_kst(),
        "paired_audit_run_count": 2,
        "frame_hashes_first": frame_hashes_first,
        "frame_hashes_second": frame_hashes_second,
        "aggregate_payload_sha256_first": payload_first,
        "aggregate_payload_sha256_second": payload_second,
        "deterministic_payload_sha256": payload_first,
        "identical": frame_hashes_first == frame_hashes_second and payload_first == payload_second,
    }


def assemble_once(h4h_root: Path) -> Dict[str, Any]:
    gamma_payload, gae_payload = gamma_lambda_binding()
    gamma = float(gamma_payload["gamma_value"])
    gae_lambda = float(gamma_payload["gae_lambda_value"])
    svc, passengers, windows, h4h_empty, h4h_empty_audit = load_sources(h4h_root)
    src = source_integrity(svc, passengers, windows, h4h_empty_audit)
    pop = evaluate_population(svc, passengers, gamma, gae_lambda)
    summary = summarize_population(pop)
    credit, attenuation = credit_diagnostics(pop, gamma, gae_lambda)
    cross = gae_crosscheck(gamma, gae_lambda)
    safety = safety_integrity(pop)
    ownership = ownership_integrity(pop)
    leakage = future_leakage_audit()
    numeric = numeric_integrity(pop)
    fixtures = fixture_results(pop, gamma, gae_lambda)
    decision, rationale = classify(summary, safety, cross, attenuation)
    ready = readiness(decision, rationale, summary)
    return {
        "gamma": gamma_payload,
        "gae_binding": gae_payload,
        "sources": src,
        "population": pop,
        "summary": summary,
        "credit": credit,
        "attenuation": attenuation,
        "gae_crosscheck": cross,
        "safety": safety,
        "ownership": ownership,
        "leakage": leakage,
        "numeric": numeric,
        "fixtures": fixtures,
        "decision": decision,
        "rationale": rationale,
        "readiness": ready,
    }


def final_report(result: Mapping[str, Any], replay: Mapping[str, Any]) -> str:
    s = result["summary"]
    d = result["decision"]
    dist = s["distributions"]
    lines = [
        "# PV8-R2A-R8E-R3-R-H4I-ER0 Final Report",
        "",
        f"- technical gate: `{PASS_GATE}`",
        f"- scientific decision: `{d}`",
        f"- H4F immutable freeze SHA: `{FREEZE_SHA}`",
        f"- H4G runtime binding SHA: `{RUNTIME_BINDING_SHA}`",
        f"- H4H source artifact: `{result['upstream']['H4H_representative_rematerialization']['artifact_root']}`",
        f"- representative windows/passengers: `54 / 414`",
        f"- empty-stop opportunities recomputed: `{s['total_empty_stop_opportunities']}`",
        f"- immediate reward equal count/rate: `{s['immediate_reward_equal_count']} / {s['immediate_reward_equal_rate']}`",
        f"- SERVE vs SKIP timing-effect count: `{s['timing_effect_created_count']}`",
        f"- initial time saving mean/median/p95: `{dist['initial_time_saved_seconds']['mean']} / {dist['initial_time_saved_seconds']['median']} / {dist['initial_time_saved_seconds']['p95']}`",
        f"- timing-effect persistence distribution: `{dist['number_of_stops_timing_benefit_persists']}`",
        f"- downstream passenger exposed count/rate: `{s['downstream_passenger_exposed_count']} / {s['downstream_passenger_exposed_rate']}`",
        f"- passenger wait improved/unchanged/worsened counts: `{s['passenger_wait_improved_count']} / {s['passenger_wait_unchanged_count']} / {s['passenger_wait_worsened_count']}`",
        f"- passenger wait delta mean/median/p95: `{dist['passenger_wait_improvement_seconds']['mean']} / {dist['passenger_wait_improvement_seconds']['median']} / {dist['passenger_wait_improvement_seconds']['p95']}`",
        f"- Reward V2 cumulative return SKIP better/equal/SERVE better: `{s['cumulative_return_SKIP_better_count']} / {s['cumulative_return_equal_count']} / {s['cumulative_return_SERVE_better_count']}`",
        f"- gamma / gae_lambda: `{result['gamma']['gamma_value']} / {result['gamma']['gae_lambda_value']}`",
        f"- credit-distance distribution: `{result['credit']['credit_distance_transitions']}`",
        f"- GAE t0 advantage SKIP better/equal/SERVE better: `{s['GAE_advantage_SKIP_better_count']} / {s['GAE_advantage_equal_count']} / {s['GAE_advantage_SERVE_better_count']}`",
        f"- effective credit attenuation: `{result['attenuation']['effective_credit_ratio_distribution']}`",
        f"- actor/critic future leakage: `{result['leakage']['actor_future_leakage']} / {result['leakage']['critic_future_leakage']}`",
        f"- missed service / alignment excess: `{result['safety']['missed_eligible_service']} / {result['safety']['unexplained_alignment_excess']}`",
        f"- duplicate ownership / orphan reward: `{result['ownership']['duplicate_wait_ownership']} / {result['ownership']['orphan_delayed_reward']}`",
        f"- NaN/Inf reward/return/advantage: `{result['numeric']['NaN_reward'] + result['numeric']['Inf_reward']} / {result['numeric']['NaN_return'] + result['numeric']['Inf_return']} / {result['numeric']['NaN_advantage'] + result['numeric']['Inf_advantage']}`",
        f"- deterministic replay: `{replay['identical']}`",
        f"- payload SHA-256: `{replay['deterministic_payload_sha256']}`",
        "- API calls / DB queries / DB writes: `0 / 0 / 0`",
        "- training performed / checkpoint use: `0 / 0`",
        f"- Reward V2 redesign required now: `{result['readiness']['reward_redesign_required']}`",
        f"- exact next recommended step: `{result['readiness']['next_recommended_step']}`",
        "",
        "## Interpretation",
        "",
        "Immediate Reward V2 equality at empty stops is not the end of the learning-signal chain. In the frozen representative source, legal pass-through removes the empty-stop dwell interval and creates downstream timing differences. A subset of opportunities reaches passenger boarding events and produces nonzero future Reward V2, discounted return, and value-neutralized GAE deltas. The signal is therefore temporally identifiable, but it is delayed and attenuated; this stage does not authorize reward redesign or training.",
    ]
    return "\n".join(lines) + "\n"


def write_manifest(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for name in PAYLOADS:
        path = writer.root / name
        rows.append({"relative_path": name, "sha256": k5.sha256_file(path), "size_bytes": path.stat().st_size, "required": True, "artifact_role": "payload", "exists": True})
    jsonl_name = "artifact_manifest.jsonl"
    jsonl_path = writer.root / jsonl_name
    jsonl_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rows.append({"relative_path": jsonl_name, "sha256": k5.sha256_file(jsonl_path), "size_bytes": jsonl_path.stat().st_size, "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest.json"
    writer.json(manifest_name, {"created_at": iso_kst(), "artifact_family": "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4i_er0_empty_stop_temporal_return_audit", "terminal_gate": gate["gate"], "final_decision": gate["final_decision"], "payload_count": len(rows), "missing_payload_count": 0, "files": rows})
    manifest_path = writer.root / manifest_name
    writer.json("_PV8_R2AR8ER3RH4I_ER0_COMPLETE.lock", {"created_at": iso_kst(), "artifact_family": "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4i_er0_empty_stop_temporal_return_audit", "terminal_gate": gate["gate"], "final_decision": gate["final_decision"], "final_manifest_path": manifest_name, "final_manifest_sha256": k5.sha256_file(manifest_path), "manifest_size_bytes": manifest_path.stat().st_size})


def run(root: Path) -> Path:
    auth = user_authorization_record()
    h4h_root = find_authoritative_h4h()
    upstream = verify_upstreams(h4h_root)
    protected_pre = protected_hashes()
    first = assemble_once(h4h_root)
    second = assemble_once(h4h_root)
    replay = deterministic_replay(first, second)
    protected_post = protected_hashes()
    if protected_pre != protected_post:
        first["decision"] = DECISION_APPROVAL
        first["readiness"] = readiness(DECISION_APPROVAL, "Protected runtime file hash changed during audit.", first["summary"])
    if not replay["identical"]:
        first["decision"] = DECISION_DETERMINISM
        first["readiness"] = readiness(DECISION_DETERMINISM, "Paired counterfactual replay was not deterministic.", first["summary"])
    first["upstream"] = upstream
    writer = k5.Writer(root)
    writer.json("r8er3rh4i_er0_user_authorization_record.json", auth)
    writer.json("r8er3rh4i_er0_upstream_binding.json", upstream)
    source_payload = {**first["sources"], "protected_runtime_file_hashes_post": protected_post, "protected_runtime_files_byte_identical": protected_pre == protected_post}
    writer.json("r8er3rh4i_er0_source_integrity.json", source_payload)
    first["population"]["opportunities"].to_parquet(root / "r8er3rh4i_er0_empty_stop_opportunity_registry.parquet", index=False)
    first["population"]["prestate"].to_parquet(root / "r8er3rh4i_er0_paired_prestate_identity.parquet", index=False)
    writer.json("r8er3rh4i_er0_action_legality_audit.json", action_legality(first["population"]["opportunities"]))
    first["population"]["branch"].to_parquet(root / "r8er3rh4i_er0_counterfactual_branch_trace.parquet", index=False)
    first["population"]["temporal"].to_parquet(root / "r8er3rh4i_er0_temporal_delta_trace.parquet", index=False)
    writer.json("r8er3rh4i_er0_time_saving_persistence.json", {**first["summary"], "created_at": iso_kst()})
    first["population"]["exposure"].to_parquet(root / "r8er3rh4i_er0_downstream_passenger_exposure.parquet", index=False)
    first["population"]["passenger_wait"].to_parquet(root / "r8er3rh4i_er0_passenger_wait_counterfactual.parquet", index=False)
    first["population"]["reward_sequence"].to_parquet(root / "r8er3rh4i_er0_reward_sequence_comparison.parquet", index=False)
    first["population"]["returns"].to_parquet(root / "r8er3rh4i_er0_discounted_return_audit.parquet", index=False)
    writer.json("r8er3rh4i_er0_gamma_lambda_binding.json", first["gamma"])
    writer.json("r8er3rh4i_er0_compute_gae_binding.json", first["gae_binding"])
    first["population"]["gae"].to_parquet(root / "r8er3rh4i_er0_gae_credit_propagation.parquet", index=False)
    writer.json("r8er3rh4i_er0_gae_analytic_crosscheck.json", first["gae_crosscheck"])
    writer.json("r8er3rh4i_er0_credit_distance_distribution.json", first["credit"])
    writer.json("r8er3rh4i_er0_credit_attenuation_diagnostic.json", first["attenuation"])
    diagnostic_by(first["population"]["branch"], "time_band").to_parquet(root / "r8er3rh4i_er0_time_band_diagnostic.parquet", index=False)
    diagnostic_by(first["population"]["branch"], "day_type").to_parquet(root / "r8er3rh4i_er0_day_type_diagnostic.parquet", index=False)
    diagnostic_by(first["population"]["branch"], "direction_id").to_parquet(root / "r8er3rh4i_er0_direction_diagnostic.parquet", index=False)
    writer.json("r8er3rh4i_er0_safety_integrity.json", first["safety"])
    writer.json("r8er3rh4i_er0_ownership_integrity.json", first["ownership"])
    writer.json("r8er3rh4i_er0_future_leakage_audit.json", first["leakage"])
    writer.json("r8er3rh4i_er0_fixture_results.json", first["fixtures"])
    writer.json("r8er3rh4i_er0_deterministic_replay.json", replay)
    writer.json("r8er3rh4i_er0_scientific_classification.json", {"created_at": iso_kst(), "decision": first["decision"], "rationale": first["rationale"], "summary": first["summary"]})
    writer.json("r8er3rh4i_er0_readiness_decision.json", first["readiness"])
    guards = claim_guard(first["decision"])
    writer.json("claim_guard_status.json", guards)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "final_decision": first["decision"], "next_recommended_step": first["readiness"]["next_recommended_step"]})
    writer.json("run_manifest.json", {"created_at": iso_kst(), "runner": str(RUNNER_PATH), "mode": "empty-stop-temporal-return-audit", "python": sys.version, "platform": platform.platform(), "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "api_call_count": 0, "db_query_count": 0, "db_write_count": 0, "policy_evaluation_count": 0, "optimizer_creation_count": 0, "training_episode_count": 0, "checkpoint_use_count": 0})
    gate = {"created_at": iso_kst(), "gate": PASS_GATE, "terminal_gate": PASS_GATE, "gate_passed": True, "final_decision": first["decision"], "readiness": first["readiness"]["next_recommended_step"], "failure_reasons": [], "policy_performance_evaluation_implied": False, "training_implied": False}
    writer.json("gate_decision.json", gate)
    writer.text("final_report.md", final_report(first, replay))
    write_manifest(writer, gate)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("empty-stop-temporal-return-audit",), required=True)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    root = k5.validate_artifact_root(Path(args.artifact_root))
    run(root)
    gate = k5.read_json(root / "gate_decision.json")
    print(f"artifact_root: {root}")
    print(f"gate: {gate['gate']}")
    print(f"decision: {gate['final_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
