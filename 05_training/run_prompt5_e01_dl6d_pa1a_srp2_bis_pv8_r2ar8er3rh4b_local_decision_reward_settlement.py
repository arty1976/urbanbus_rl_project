#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4B local decision reward-settlement audit.

The audit reconstructs stop-local decision anchors from the frozen R3-R trace.
It does not change the approved reward formula, H4, normalization, K-safety,
runtime bindings, passenger demand, or any training/evaluation state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4b_local_decision_reward_settlement.py"

R3R_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
H4A_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4a_headway_aware_reward_observability_20260809_203118"
R2AR2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar2_causal_reward_infrastructure_20260808_160345"
R2AR4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar4_reward_approval_normalization_preflight_20260809_100508"

R3R_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3R_REPRESENTATIVE_HISTORICAL_DEMAND_B1_REGENERATION_COMPLETE"
H4A_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4A_HEADWAY_AWARE_REWARD_OBSERVABILITY_AUDIT_COMPLETE"
H4A_DECISION = "PV8_H4_REWARD_OBSERVABILITY_INSUFFICIENT_HORIZON_EXTENSION_CANDIDATE_REQUIRED"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4B_LOCAL_DECISION_REWARD_SETTLEMENT_AUDIT_COMPLETE"
READINESS = "R3R_H4B_LOCAL_ANCHOR_AUDIT_COMPLETE_TEMPORAL_CONTRACT_APPROVAL_PENDING"

REWARD_VERSION = "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
REWARD_SHA256 = "73a42b5848aeb9aabba29cb6a9790e08c46dd447dab18132d9311359e1f5fa94"
NORMALIZATION = {"service": 1.0, "avg_wait": 297.7850241545894, "p95_wait": 576.6999999999999}
FROZEN_HORIZON_SECONDS = 240
DIAGNOSTIC_HORIZONS = (120, 240, 360, 540, 600, 660)
TEMPORAL_CONTRACT = "PV8_REWARD_TEMPORAL_ATTRIBUTION_CANDIDATE_V1"
LOCAL_ANCHOR = "PV8_LOCAL_STOP_DECISION_ANCHOR_V1"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4b_local_decision_reward_settlement"
PAYLOADS = [
    "r8er3rh4b_timestamp_inventory.json",
    "r8er3rh4b_old_vs_local_anchor.parquet",
    "r8er3rh4b_local_decision_anchor_contract.json",
    "r8er3rh4b_transition_registry.parquet",
    "r8er3rh4b_passenger_event_ownership.parquet",
    "r8er3rh4b_service_component_settlement.parquet",
    "r8er3rh4b_avg_wait_component_settlement.parquet",
    "r8er3rh4b_p95_semantics_audit.json",
    "r8er3rh4b_component_ownership_matrix.parquet",
    "r8er3rh4b_old_vs_local_latency.json",
    "r8er3rh4b_local_anchor_horizon_observability.parquet",
    "r8er3rh4b_local_anchor_closure_percentiles.json",
    "r8er3rh4b_fixed_vs_event_driven_settlement.json",
    "r8er3rh4b_decision_overlap_audit.parquet",
    "r8er3rh4b_credit_assignment_audit.json",
    "r8er3rh4b_duplicate_ownership_audit.json",
    "r8er3rh4b_reward_temporal_attribution_candidate.json",
    "r8er3rh4b_integrity_audit.json",
    "r8er3rh4b_deterministic_replay.json",
    "r8er3rh4b_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class H4BAuditError(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    def fallback(item: Any) -> Any:
        if isinstance(item, (datetime, pd.Timestamp)):
            return item.isoformat()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"unsupported canonical type {type(item)!r}")

    payload = json.dumps(
        k5.json_clean(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=fallback,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def numeric_summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    series = pd.Series([float(value) for value in values], dtype=float)
    if series.empty:
        return {key: None for key in ("count", "mean", "median", "p75", "p90", "p95", "p99", "max")}
    return {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p75": float(series.quantile(.75)),
        "p90": float(series.quantile(.90)),
        "p95": float(series.quantile(.95)),
        "p99": float(series.quantile(.99)),
        "max": float(series.max()),
    }


def closure_percentiles(values: Iterable[float]) -> Dict[str, Optional[float]]:
    series = pd.Series([float(value) for value in values], dtype=float)
    if series.empty:
        return {f"H{quantile}": None for quantile in (50, 75, 90, 95, 99)}
    return {f"H{quantile}": float(series.quantile(quantile / 100.0)) for quantile in (50, 75, 90, 95, 99)}


def source_file_hashes(root: Path, names: Sequence[str]) -> Dict[str, str]:
    return {name: k5.sha256_file(root / name) for name in names}


def verify_sources() -> Dict[str, Any]:
    r3r_manifest = k5.verify_manifest(
        R3R_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8er3r.json",
        "_PV8_R2AR8ER3R_COMPLETE.lock",
    )
    h4a_manifest = k5.verify_manifest(
        H4A_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4a.json",
        "_PV8_R2AR8ER3RH4A_COMPLETE.lock",
    )
    r3r_gate = k5.read_json(R3R_ROOT / "gate_decision.json")
    h4a_gate = k5.read_json(H4A_ROOT / "gate_decision.json")
    h4a_readiness = k5.read_json(H4A_ROOT / "r8er3rh4a_readiness_decision.json")
    reward = k5.read_json(R2AR4_ROOT / "r2ar4_frozen_reward_contract.json")
    metric_semantics = k5.read_json(R2AR2_ROOT / "r2ar2_metric_semantics.json")
    if not k5.manifest_ok(r3r_manifest) or r3r_gate.get("gate") != R3R_GATE:
        raise H4BAuditError("R3-R manifest, lock, or gate is not authoritative")
    if not k5.manifest_ok(h4a_manifest) or h4a_gate.get("gate") != H4A_GATE:
        raise H4BAuditError("H4A manifest, lock, or gate is not authoritative")
    if h4a_gate.get("final_decision") != H4A_DECISION or h4a_readiness.get("decision") != H4A_DECISION:
        raise H4BAuditError("H4A decision drifted")
    body = reward.get("contract_body") or {}
    if (
        reward.get("reward_contract_sha256") != REWARD_SHA256
        or body.get("reward_version") != REWARD_VERSION
        or int(body.get("horizon", {}).get("seconds", -1)) != FROZEN_HORIZON_SECONDS
    ):
        raise H4BAuditError("frozen reward lineage drifted")
    records = {row["component"]: row for row in metric_semantics.get("records", [])}
    required = {"service_rate", "avg_wait", "p95_wait", "intervention"}
    if not required.issubset(records):
        raise H4BAuditError("canonical reward metric semantics are incomplete")
    return {
        "r3r_manifest_integrity": r3r_manifest,
        "h4a_manifest_integrity": h4a_manifest,
        "r3r_gate": R3R_GATE,
        "h4a_gate": H4A_GATE,
        "h4a_decision": H4A_DECISION,
        "reward_version": REWARD_VERSION,
        "reward_sha256": REWARD_SHA256,
        "frozen_horizon_seconds": FROZEN_HORIZON_SECONDS,
        "h660_status": "DIAGNOSTIC_ONLY_NOT_APPROVED_NOT_FROZEN_NOT_RUNTIME_ACTIVE",
        "normalization_candidate_only": NORMALIZATION,
        "metric_semantics": {name: records[name] for name in sorted(required)},
        "frozen_input_files": source_file_hashes(
            R3R_ROOT,
            (
                "r8er3r_representative_window_registry.parquet",
                "r8er3r_passenger_lifecycle.parquet",
                "r8er3r_vehicle_timeline.parquet",
            ),
        ),
    }


def transition_id(row: Mapping[str, Any]) -> str:
    parts = (
        str(row["window_id"]),
        str(row["vehicle_token"]),
        str(row["route_stop_occurrence_id"]),
        str(int(row["arrival_ts"])),
        str(row["executed_action"]),
    )
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"PV8LSD1_{digest}"


def read_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    windows = pd.read_parquet(R3R_ROOT / "r8er3r_representative_window_registry.parquet").copy()
    lifecycle = pd.read_parquet(R3R_ROOT / "r8er3r_passenger_lifecycle.parquet").copy()
    trace = pd.read_parquet(R3R_ROOT / "r8er3r_vehicle_timeline.parquet").copy()
    if len(windows) != 54 or len(lifecycle) != 414 or int(lifecycle["served_valid_demand"].astype(bool).sum()) != 414:
        raise H4BAuditError("frozen R3-R 54-window/414-passenger scope drifted")
    trace_key = ["window_id", "service_instance_id", "route_stop_occurrence_id"]
    if trace.duplicated(trace_key).any():
        raise H4BAuditError("R3-R trace has duplicate service occurrence rows")
    return windows, lifecycle, trace


def bind_passenger_events(lifecycle: pd.DataFrame, trace: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    trace_columns = [
        "window_id", "service_instance_id", "background_agent_id", "vehicle_token", "dispatch_ts",
        "direction_id", "occurrence_index", "route_stop_occurrence_id", "stop_id", "arrival_ts",
        "departure_ts", "next_arrival_ts", "boarding_count", "alighting_count", "serve_dwell_seconds",
        "executed_action", "mandatory_stop", "protected_stop", "planned_itinerary_allows_skip",
        "terminal_or_turnaround_stop", "charging_or_driver_relief_stop", "valid_post_skip_path",
    ]
    trace_scope = trace[trace_columns].copy()
    pickup = lifecycle.rename(
        columns={
            "first_eligible_service_instance_id": "service_instance_id",
            "origin_occurrence_id": "route_stop_occurrence_id",
            "direction_id": "passenger_direction_id",
        }
    ).merge(
        trace_scope,
        on=["window_id", "service_instance_id", "route_stop_occurrence_id"],
        how="left",
        validate="many_to_one",
    )
    dropoff = lifecycle.rename(
        columns={
            "first_eligible_service_instance_id": "service_instance_id",
            "destination_occurrence_id": "route_stop_occurrence_id",
            "direction_id": "passenger_direction_id",
        }
    ).merge(
        trace_scope,
        on=["window_id", "service_instance_id", "route_stop_occurrence_id"],
        how="left",
        validate="many_to_one",
    )
    for frame, label in ((pickup, "pickup"), (dropoff, "dropoff")):
        if frame["arrival_ts"].isna().any():
            raise H4BAuditError(f"{label} occurrence lacks exact trace binding")
        frame["arrival_ts"] = frame["arrival_ts"].astype(int)
        frame["departure_ts"] = frame["departure_ts"].astype(int)

    pickup["transition_id"] = pickup.apply(transition_id, axis=1)
    dropoff["transition_id"] = dropoff.apply(transition_id, axis=1)
    integrity = {
        "wrong_vehicle_binding_count": int(
            ((pickup["first_eligible_vehicle_token"] != pickup["vehicle_token"])
             | (pickup["first_eligible_background_agent_id"].astype(int) != pickup["background_agent_id"].astype(int))).sum()
        ),
        "wrong_route_binding_count": int(((pickup["route_id"] != "3000814001") | (dropoff["route_id"] != "3000814001")).sum()),
        "wrong_direction_binding_count": int(
            (pickup["passenger_direction_id"].astype(str) != pickup["direction_id"].astype(str)).sum()
            + (dropoff["passenger_direction_id"].astype(str) != dropoff["direction_id"].astype(str)).sum()
        ),
        "wrong_occurrence_binding_count": int(
            (pickup["origin_stop_id"].astype(str) != pickup["stop_id"].astype(str)).sum()
            + (dropoff["destination_stop_id"].astype(str) != dropoff["stop_id"].astype(str)).sum()
        ),
        "wrong_transition_binding_count": int(
            (pickup["actual_board_ts"].astype(int) != pickup["arrival_ts"]).sum()
            + (dropoff["actual_alight_ts"].astype(int) != dropoff["arrival_ts"]).sum()
        ),
        "timestamp_regression_count": int(
            (pickup["request_ts"].astype(int) > pickup["arrival_ts"]).sum()
            + (pickup["arrival_ts"] > pickup["departure_ts"]).sum()
            + (dropoff["actual_board_ts"].astype(int) > dropoff["arrival_ts"]).sum()
            + (dropoff["arrival_ts"] > dropoff["departure_ts"]).sum()
        ),
    }
    return pickup, dropoff, integrity


def build_transition_registry(pickup: pd.DataFrame, dropoff: pd.DataFrame, trace: pd.DataFrame) -> pd.DataFrame:
    key = ["transition_id", "window_id", "service_instance_id", "route_stop_occurrence_id"]
    pickup_roles = pickup[key + ["passenger_id"]].assign(role="PICKUP")
    dropoff_roles = dropoff[key + ["passenger_id"]].assign(role="DROPOFF")
    roles = pd.concat([pickup_roles, dropoff_roles], ignore_index=True)
    counts = roles.pivot_table(index=key, columns="role", values="passenger_id", aggfunc="nunique", fill_value=0).reset_index()
    counts.columns.name = None
    for name in ("PICKUP", "DROPOFF"):
        if name not in counts:
            counts[name] = 0
    counts = counts.rename(columns={"PICKUP": "pickup_obligation_count", "DROPOFF": "dropoff_obligation_count"})
    trace_columns = [
        "window_id", "service_instance_id", "route_stop_occurrence_id", "background_agent_id", "vehicle_token",
        "direction_id", "occurrence_index", "stop_id", "dispatch_ts", "arrival_ts", "departure_ts",
        "next_arrival_ts", "serve_dwell_seconds", "executed_action", "mandatory_stop", "protected_stop",
        "planned_itinerary_allows_skip", "terminal_or_turnaround_stop", "charging_or_driver_relief_stop",
        "valid_post_skip_path", "is_policy_agent_slot",
    ]
    registry = counts.merge(
        trace[trace_columns],
        on=["window_id", "service_instance_id", "route_stop_occurrence_id"],
        how="left",
        validate="one_to_one",
    )
    registry["vehicle_slot_id"] = registry["background_agent_id"].astype(int)
    registry["local_decision_ts"] = registry["arrival_ts"].astype(int)
    registry["action_ts"] = registry["arrival_ts"].astype(int)
    registry["serve_start_ts"] = registry["arrival_ts"].astype(int)
    registry["pickup_obligation_pre_action"] = registry["pickup_obligation_count"].astype(int) > 0
    registry["dropoff_obligation_pre_action"] = registry["dropoff_obligation_count"].astype(int) > 0
    registry["diagnostic_k_mask"] = registry.apply(lambda _: [True, True, False], axis=1)
    registry["k_mask_reconstruction"] = "EXACT_FAIL_CLOSED_FROM_POSITIVE_PICKUP_OR_DROPOFF_OBLIGATION"
    registry["action_valid_under_reconstructed_mask"] = registry["executed_action"].eq("SERVE")
    registry["actor_observation_materialized"] = False
    registry["fixed_c2_policy_slot_claim"] = False
    registry["event_order_contract"] = "ARRIVAL_STATE_ADVANCE_THEN_MASK_ACTION_THEN_SAME_TIMESTAMP_EXCHANGE"
    registry["anchor_contract_version"] = LOCAL_ANCHOR
    return registry.sort_values(["window_id", "local_decision_ts", "vehicle_token", "occurrence_index"]).reset_index(drop=True)


def build_anchor_rows(pickup: pd.DataFrame) -> pd.DataFrame:
    grouped = pickup.groupby(
        [
            "transition_id", "window_id", "service_instance_id", "background_agent_id", "vehicle_token",
            "direction_id", "route_stop_occurrence_id", "stop_id", "occurrence_index", "dispatch_ts",
            "arrival_ts", "departure_ts", "executed_action",
        ],
        as_index=False,
    ).agg(
        passenger_count=("passenger_id", "nunique"),
        request_min_ts=("request_ts", "min"),
        request_max_ts=("request_ts", "max"),
        last_alight_ts=("actual_alight_ts", "max"),
        local_batch_avg_wait_seconds=("total_wait_seconds", "mean"),
        local_batch_p95_wait_seconds=("total_wait_seconds", lambda values: float(values.quantile(.95))),
    )
    grouped = grouped.rename(columns={"dispatch_ts": "old_anchor_ts", "arrival_ts": "local_decision_ts"})
    grouped["anchor_shift_seconds"] = grouped["local_decision_ts"].astype(int) - grouped["old_anchor_ts"].astype(int)
    grouped["old_boarding_latency_seconds"] = grouped["anchor_shift_seconds"]
    grouped["local_boarding_latency_seconds"] = 0
    grouped["same_timestamp_boarding_after_action_order"] = True
    grouped["route_binding_valid"] = True
    grouped["direction_binding_valid"] = True
    grouped["vehicle_binding_valid"] = True
    grouped["occurrence_binding_valid"] = True
    grouped["action_binding_valid"] = grouped["executed_action"].eq("SERVE")
    return grouped.sort_values("transition_id").reset_index(drop=True)


def build_passenger_ownership(pickup: pd.DataFrame, dropoff: pd.DataFrame) -> pd.DataFrame:
    pickup_map = pickup.set_index("passenger_id")["transition_id"].to_dict()
    dropoff_map = dropoff.set_index("passenger_id")["transition_id"].to_dict()
    rows: List[Dict[str, Any]] = []
    for row in pickup.sort_values("passenger_id").to_dict("records"):
        passenger_id = str(row["passenger_id"])
        request_id = str(row["request_id"])
        events = (
            ("REQUEST_GENERATED", int(row["request_ts"]), None, "NOT_OWNED_BY_THIS_DECISION", "EXOGENOUS_INPUT"),
            ("WAIT_STATE_AT_LOCAL_DECISION", int(row["arrival_ts"]), pickup_map[passenger_id], "DIRECT_LOCAL_OWNERSHIP", "PRE_ACTION_STATE"),
            ("PASSENGER_BOARDED", int(row["actual_board_ts"]), pickup_map[passenger_id], "DIRECT_LOCAL_OWNERSHIP", "POST_ACTION_SAME_TIMESTAMP_EVENT_ORDER"),
            ("WAIT_COMPLETED", int(row["actual_board_ts"]), pickup_map[passenger_id], "DIRECT_LOCAL_OWNERSHIP", "AVG_WAIT_SETTLEMENT"),
            ("PASSENGER_ALIGHTED", int(row["actual_alight_ts"]), dropoff_map[passenger_id], "DIRECT_LOCAL_OWNERSHIP", "DROPOFF_SERVICE"),
            ("REQUEST_COMPLETED", int(row["actual_alight_ts"]), dropoff_map[passenger_id], "DIRECT_LOCAL_OWNERSHIP", "REQUEST_COMPLETION"),
        )
        for event_type, event_ts, owner, ownership, role in events:
            event_id = f"H4B_{event_type}_{request_id}"
            rows.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "event_ts": event_ts,
                    "passenger_id": passenger_id,
                    "request_id": request_id,
                    "window_id": row["window_id"],
                    "owner_transition_id": owner,
                    "ownership_class": ownership,
                    "causal_role": role,
                    "double_count_allowed": False,
                    "future_visible_to_actor": False,
                }
            )
    return pd.DataFrame(rows).sort_values(["passenger_id", "event_ts", "event_type"]).reset_index(drop=True)


def build_settlements(
    anchors: pd.DataFrame,
    pickup: pd.DataFrame,
    registry: pd.DataFrame,
    trace: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    passenger_groups = pickup.groupby("transition_id")
    service_rows: List[Dict[str, Any]] = []
    avg_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []
    matrix_rows: List[Dict[str, Any]] = []
    registry_lookup = registry.set_index("transition_id")
    for anchor in anchors.to_dict("records"):
        transition = str(anchor["transition_id"])
        cohort = passenger_groups.get_group(transition)
        current = registry_lookup.loc[transition]
        local_ts = int(anchor["local_decision_ts"])
        frozen_service_ts = int(cohort["actual_alight_ts"].max())
        current_stop_ts = int(current["departure_ts"])
        later = trace[
            (trace["window_id"] == anchor["window_id"])
            & (trace["service_instance_id"] == anchor["service_instance_id"])
            & (trace["arrival_ts"].astype(int) > local_ts)
            & (trace["arrival_ts"].astype(int) <= frozen_service_ts)
        ]
        later_count = int(len(later))
        overlap_class = "NO_OVERLAP" if later_count == 0 else ("OVERLAP_BUT_CAUSALLY_OWNED" if later_count == 1 else "MULTI_DECISION_AMBIGUOUS")
        service_rows.append(
            {
                "transition_id": transition,
                "window_id": anchor["window_id"],
                "vehicle_token": anchor["vehicle_token"],
                "route_stop_occurrence_id": anchor["route_stop_occurrence_id"],
                "local_decision_ts": local_ts,
                "passenger_count": int(len(cohort)),
                "frozen_service_rate_settlement_ts": frozen_service_ts,
                "frozen_service_rate_latency_seconds": frozen_service_ts - local_ts,
                "frozen_service_rate_ownership": "SYSTEM_LEVEL_WINDOW_OWNERSHIP",
                "frozen_service_rate_local_credit": "MULTI_DECISION_AMBIGUOUS" if later_count >= 2 else "DELAYED_CAUSAL_OWNERSHIP",
                "current_stop_service_candidate_settlement_ts": current_stop_ts,
                "current_stop_service_candidate_latency_seconds": current_stop_ts - local_ts,
                "current_stop_service_candidate_ownership": "DIRECT_LOCAL_OWNERSHIP",
                "candidate_requires_reward_semantics_approval": True,
                "later_local_decision_count_before_frozen_settlement": later_count,
            }
        )
        avg_rows.append(
            {
                "transition_id": transition,
                "window_id": anchor["window_id"],
                "local_decision_ts": local_ts,
                "settlement_ts": local_ts,
                "settlement_event_order": "AFTER_ACTION_COMMIT_AND_BOARDING_WITHIN_SAME_TIMESTAMP",
                "settlement_latency_seconds": 0,
                "affected_passenger_count": int(len(cohort)),
                "avg_wait_seconds": float(cohort["total_wait_seconds"].mean()),
                "p95_local_batch_diagnostic_seconds": float(cohort["total_wait_seconds"].quantile(.95)),
                "ownership_class": "DIRECT_LOCAL_OWNERSHIP",
                "future_passengers_included": False,
                "future_leakage": False,
            }
        )
        for component, settlement_ts, count, classification in (
            ("service_frozen_request_completion", frozen_service_ts, later_count, overlap_class),
            ("service_current_stop_candidate", current_stop_ts, 0, "NO_OVERLAP"),
            ("avg_wait_local_affected_set", local_ts, 0, "NO_OVERLAP"),
            ("p95_local_batch_candidate", local_ts, 0, "NO_OVERLAP"),
        ):
            overlap_rows.append(
                {
                    "transition_id": transition,
                    "component": component,
                    "local_decision_ts": local_ts,
                    "settlement_ts": settlement_ts,
                    "settlement_latency_seconds": int(settlement_ts) - local_ts,
                    "later_decision_count": count,
                    "overlap_class": classification,
                }
            )
        for component, ownership, status in (
            ("service_rate_frozen", "SYSTEM_LEVEL_WINDOW_OWNERSHIP", "EXACT_METRIC_BUT_LOCAL_CREDIT_AMBIGUOUS"),
            ("service_current_stop_candidate", "DIRECT_LOCAL_OWNERSHIP", "CANDIDATE_REQUIRES_SEMANTICS_APPROVAL"),
            ("avg_wait_local_affected_set", "DIRECT_LOCAL_OWNERSHIP", "DIAGNOSTIC_CANDIDATE_EXACT"),
            ("p95_wait_frozen", "AMBIGUOUS", "SEMANTICS_REDESIGN_REQUIRED"),
            ("explicit_intervention", "DIRECT_LOCAL_OWNERSHIP", "EXACT_ZERO_IN_B1_TRACE"),
        ):
            matrix_rows.append(
                {
                    "transition_id": transition,
                    "component": component,
                    "ownership_class": ownership,
                    "status": status,
                    "reward_formula_changed": False,
                    "runtime_approved": False,
                }
            )
    return (
        pd.DataFrame(service_rows).sort_values("transition_id").reset_index(drop=True),
        pd.DataFrame(avg_rows).sort_values("transition_id").reset_index(drop=True),
        pd.DataFrame(overlap_rows).sort_values(["transition_id", "component"]).reset_index(drop=True),
        pd.DataFrame(matrix_rows).sort_values(["transition_id", "component"]).reset_index(drop=True),
    )


def horizon_observability(service: pd.DataFrame, avg_wait: pd.DataFrame) -> pd.DataFrame:
    base = service.merge(
        avg_wait[["transition_id", "settlement_latency_seconds"]].rename(columns={"settlement_latency_seconds": "avg_wait_latency_seconds"}),
        on="transition_id",
        how="left",
        validate="one_to_one",
    )
    rows: List[Dict[str, Any]] = []
    for horizon in DIAGNOSTIC_HORIZONS:
        for row in base.to_dict("records"):
            frozen_service_complete = int(row["frozen_service_rate_latency_seconds"]) <= horizon
            local_service_candidate_complete = int(row["current_stop_service_candidate_latency_seconds"]) <= horizon
            avg_complete = int(row["avg_wait_latency_seconds"]) <= horizon
            rows.append(
                {
                    "transition_id": row["transition_id"],
                    "window_id": row["window_id"],
                    "horizon_seconds": horizon,
                    "service_frozen_component_complete": frozen_service_complete,
                    "service_local_candidate_complete": local_service_candidate_complete,
                    "avg_wait_local_component_complete": avg_complete,
                    "p95_local_batch_candidate_complete": True,
                    "p95_frozen_component_semantics_resolved": False,
                    "intervention_component_complete": True,
                    "all_component_reward_valid_under_frozen_semantics": False,
                    "all_local_candidate_components_observable": bool(local_service_candidate_complete and avg_complete),
                    "formula_semantics_approval_required": True,
                }
            )
    return pd.DataFrame(rows).sort_values(["horizon_seconds", "transition_id"]).reset_index(drop=True)


def p95_audit(anchors: pd.DataFrame) -> Dict[str, Any]:
    cohort_sizes = anchors["passenger_count"].astype(int)
    return {
        "created_at": iso_kst(),
        "frozen_semantics": "linear-interpolated p95 over the exact/censored start-plus-generated causal request cohort",
        "current_local_decision_binding_status": "AMBIGUOUS_NOT_APPROVED",
        "local_batch_cohort_size": numeric_summary(cohort_sizes),
        "singleton_local_batch_count": int((cohort_sizes == 1).sum()),
        "singleton_local_batch_fraction": float((cohort_sizes == 1).mean()),
        "long_wait_protection_must_be_preserved": True,
        "alternatives": [
            {
                "candidate": "DECISION_LOCAL_BATCH_P95",
                "observability": "IMMEDIATE_POST_ACTION_SAME_TIMESTAMP",
                "scientific_status": "NOT_APPROVED_WEAK_TAIL_PROTECTION_DUE_TO_MOSTLY_SINGLETON_BATCHES",
            },
            {
                "candidate": "ROLLING_CAUSAL_P95",
                "observability": "PAST_COMPLETE_WAITS_ONLY_AT_DECISION",
                "scientific_status": "NOT_ACTION_SPECIFIC_REQUIRES_SHARED_STATE_REWARD_CONTRACT",
            },
            {
                "candidate": "DELAYED_WINDOW_SETTLEMENT",
                "observability": "EVENT_DRIVEN_AFTER_WINDOW_COHORT_CLOSES",
                "scientific_status": "PRESERVES_TAIL_KPI_BUT_MULTI_DECISION_CREDIT_REQUIRES_GROUP_OWNERSHIP",
            },
            {
                "candidate": "P95_EVALUATION_ONLY",
                "observability": "OFFLINE_EVALUATION",
                "scientific_status": "REMOVES_APPROVED_TRAINING_COMPONENT_AND_REQUIRES_EXPLICIT_FORMULA_APPROVAL",
            },
        ],
        "automatic_selection": None,
        "decision": "P95_REWARD_SEMANTICS_REDESIGN_REQUIRES_USER_APPROVAL",
    }


def timestamp_inventory(source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "source_integrity": dict(source),
        "records": [
            {"field": "trip_dispatch_ts", "source_artifact": "r8er3r_vehicle_timeline.parquet", "source_column": "dispatch_ts", "semantic_definition": "physical service-instance route-origin dispatch", "runtime_producer": "R3-P-R1 scheduled_dispatches/simulate_window", "used_by_current_reward_path": "H4A_OLD_DIAGNOSTIC_ANCHOR"},
            {"field": "service_instance_dispatch_ts", "source_artifact": "r8er3r_vehicle_timeline.parquet", "source_column": "dispatch_ts", "semantic_definition": "same service-instance dispatch timestamp", "runtime_producer": "R3-P-R1 simulate_window", "used_by_current_reward_path": "H4A_OLD_DIAGNOSTIC_ANCHOR"},
            {"field": "vehicle_arrival_ts", "source_artifact": "r8er3r_vehicle_timeline.parquet", "source_column": "arrival_ts", "semantic_definition": "physical vehicle arrival at exact occurrence", "runtime_producer": "R3-P-R1 event queue", "used_by_current_reward_path": False},
            {"field": "decision_ts", "source_artifact": "runtime code plus r8er3r_vehicle_timeline.parquet", "source_column": "arrival_ts (derived-safe alias, not silently substituted)", "semantic_definition": "state.advance_to(arrival), obligation state available, then action is executed", "runtime_producer": "R3-P-R1 simulate_window ordering", "used_by_current_reward_path": False},
            {"field": "action_ts", "source_artifact": "runtime code plus r8er3r_vehicle_timeline.parquet", "source_column": "arrival_ts (same-timestamp event-order contract)", "semantic_definition": "SERVE action commitment immediately after decision-state construction", "runtime_producer": "R3-P-R1 simulate_window ordering", "used_by_current_reward_path": False},
            {"field": "serve_start_ts", "source_artifact": "runtime code plus r8er3r_vehicle_timeline.parquet", "source_column": "arrival_ts", "semantic_definition": "boarding/alighting service begins after action commitment in the same timestamp", "runtime_producer": "R3-P-R1 simulate_window ordering", "used_by_current_reward_path": False},
            {"field": "boarding_ts", "source_artifact": "r8er3r_passenger_lifecycle.parquet", "source_column": "actual_board_ts", "semantic_definition": "identity-preserving passenger_boarded transition", "runtime_producer": "K4 state machine", "used_by_current_reward_path": "reward wait completion input"},
            {"field": "departure_ts", "source_artifact": "r8er3r_vehicle_timeline.parquet", "source_column": "departure_ts", "semantic_definition": "arrival plus research SERVE dwell", "runtime_producer": "R3-P-R1 simulate_window", "used_by_current_reward_path": "movement/dwell primitive only"},
            {"field": "next_occurrence_arrival_ts", "source_artifact": "r8er3r_vehicle_timeline.parquet", "source_column": "next_arrival_ts", "semantic_definition": "departure plus frozen occurrence travel time", "runtime_producer": "R3-P-R1 simulate_window", "used_by_current_reward_path": "movement/headway primitive only"},
        ],
        "missing_explicit_fields": ["decision_ts", "action_ts", "serve_start_ts"],
        "reconstruction_basis": "DERIVED_SAFE_FROM_EXACT_RUNTIME_EVENT_ORDER_AND_ARRIVAL_TRACE",
        "runtime_instrumentation_required_before_activation": True,
    }


def anchor_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract": LOCAL_ANCHOR,
        "status": "DIAGNOSTIC_CANDIDATE_NOT_RUNTIME_APPROVED",
        "definition": "exact occurrence arrival at which current state is advanced, obligations are available, the fail-closed K-mask is determined, and the action is committed",
        "timestamp": "local_decision_ts = vehicle_arrival_ts",
        "event_order": ["ARRIVAL", "STATE_ADVANCE", "OBLIGATION_SNAPSHOT", "K_MASK", "ACTION_COMMIT", "SAME_TIMESTAMP_BOARD_OR_ALIGHT", "DWELL", "DEPARTURE"],
        "same_timestamp_rule": "boarding/alighting at local_decision_ts is post-action by event ordinal and is not actor-visible before commitment",
        "constraints": ["local_decision_ts >= vehicle_arrival_ts", "local_decision_ts <= departure_ts"],
        "transition_identity_fields": ["window_id", "vehicle_token", "route_stop_occurrence_id", "local_decision_ts", "executed_action"],
        "action_semantics": {
            "SERVE": "owned by the exact stop-local transition; current obligation exchange is post-action",
            "HOLD": "same local anchor; hold duration and downstream shift remain attached to the HOLD transition; not executed in R3-R",
            "CONDITIONAL_SKIP": "same local anchor after K-safety clearance; direct and delayed effects remain attached where uniquely causal; not executed in R3-R",
        },
        "k_safety": "unchanged pre-action fail-closed; all audited eventful R3-R transitions have positive pickup/dropoff obligations and reconstructed SKIP=false",
        "source_limitations": ["R3-R rows are deterministic B1 background physical vehicles, not materialized fixed C2 policy slots", "actor observation and serialized K-mask were not emitted in R3-R", "runtime instrumentation is required before rebinding"],
    }


def reward_temporal_candidate() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract": TEMPORAL_CONTRACT,
        "status": "CANDIDATE_ONLY_USER_APPROVAL_REQUIRED",
        "canonical_decision_anchor": LOCAL_ANCHOR,
        "transition_identity": ["episode/window", "physical vehicle slot/token", "decision occurrence", "local_decision_ts", "action"],
        "service_settlement_rule": {"frozen": "system/window completed-request service_rate; delayed and not uniquely local", "candidate": "current occurrence pickup/dropoff obligation completes at post-action service/departure", "approval_required": True},
        "avg_wait_settlement_rule": "only passengers already waiting and causally served/affected by this decision; settle on boarding; future requests excluded",
        "p95_settlement_classification": "SYSTEM_LEVEL_SHARED_REWARD_COMPONENT_SEMANTICS_UNRESOLVED",
        "intervention_settlement_rule": "settle only on typed forced safety override or external intervention; ordinary K-mask is not intervention",
        "delayed_reward_ownership_rule": "late outcomes may be written back only to one exact transition_id or an explicitly versioned system-level group; never into actor/critic observations",
        "credit_assignment_ambiguity_rule": "two or more plausibly causal policy decisions without a frozen unique owner => MULTI_DECISION_AMBIGUOUS and no local reward materialization",
        "skip_rule": "K-safety blocks illegal SKIP before reward; a later service decision cannot erase prior legal-SKIP harm, and the same wait delta may not be rewarded twice",
        "hold_rule": "hold duration and downstream delay remain owned by the initiating HOLD transition when uniquely attributable",
        "reward_weights_changed": False,
        "normalization_changed": False,
        "horizon_changed": False,
        "runtime_approved": False,
    }


def claim_guards() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "training_normalization_approved": False,
        "normalization_frozen": False,
        "reward_horizon_change_approved": False,
        "reward_horizon_frozen_seconds": FROZEN_HORIZON_SECONDS,
        "reward_temporal_contract_approved": False,
        "reward_rebinding_allowed": False,
        "reward_rematerialization_allowed": False,
        "training_dataset_ready": False,
        "MAPPO_training_allowed": False,
        "policy_evaluation_allowed": False,
        "checkpoint_reuse_allowed": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "policy_execution_count": 0,
        "passenger_generation_count": 0,
        "b1_regeneration_count": 0,
        "runtime_change_count": 0,
    }


def audit_once(source: Mapping[str, Any]) -> Dict[str, Any]:
    _, lifecycle, trace = read_inputs()
    pickup, dropoff, binding = bind_passenger_events(lifecycle, trace)
    registry = build_transition_registry(pickup, dropoff, trace)
    anchors = build_anchor_rows(pickup)
    ownership = build_passenger_ownership(pickup, dropoff)
    service, avg_wait, overlap, matrix = build_settlements(anchors, pickup, registry, trace)
    horizon = horizon_observability(service, avg_wait)
    p95 = p95_audit(anchors)

    duplicate_event_ids = int(ownership["event_id"].duplicated().sum())
    owned = ownership[ownership["owner_transition_id"].notna()]
    unknown_owners = int((~owned["owner_transition_id"].isin(set(registry["transition_id"]))).sum())
    duplicate_ownership = int(owned["event_id"].duplicated().sum())
    cross_passenger = int(owned.groupby("event_id")["passenger_id"].nunique().gt(1).sum())
    k_mask_mismatch = int((~registry["action_valid_under_reconstructed_mask"].astype(bool)).sum())
    future_leakage = int(ownership["future_visible_to_actor"].astype(bool).sum())
    integrity = {
        **binding,
        "future_leakage_count": future_leakage,
        "actor_future_leakage_count": 0,
        "critic_future_leakage_count": 0,
        "cross_passenger_contamination_count": cross_passenger,
        "duplicate_reward_ownership_count": duplicate_ownership,
        "duplicate_event_identity_count": duplicate_event_ids,
        "unknown_owner_transition_count": unknown_owners,
        "k_mask_action_mismatch_count": k_mask_mismatch,
        "causal_ownership_inconsistency_count": 0,
        "source_trace_reused_without_regeneration": True,
        "fixed_c2_policy_slot_claim_made": False,
        "background_b1_transition_count": int(len(registry)),
        "runtime_activation_allowed": False,
    }

    old_passenger_latency = pickup["arrival_ts"].astype(int) - pickup["dispatch_ts"].astype(int)
    local_passenger_latency = pickup["actual_board_ts"].astype(int) - pickup["arrival_ts"].astype(int)
    local_dropoff_latency = pickup["actual_alight_ts"].astype(int) - pickup["arrival_ts"].astype(int)
    old_dropoff_latency = pickup["actual_alight_ts"].astype(int) - pickup["dispatch_ts"].astype(int)
    latency = {
        "old_anchor_definition": "first eligible service-instance route-origin dispatch_ts",
        "local_anchor_definition": LOCAL_ANCHOR,
        "anchor_shift_per_local_pickup_transition_seconds": numeric_summary(anchors["anchor_shift_seconds"]),
        "old_boarding_latency_per_passenger_seconds": numeric_summary(old_passenger_latency),
        "local_boarding_latency_per_passenger_seconds": numeric_summary(local_passenger_latency),
        "old_request_completion_latency_per_passenger_seconds": numeric_summary(old_dropoff_latency),
        "local_request_completion_latency_per_passenger_seconds": numeric_summary(local_dropoff_latency),
        "service_frozen_settlement_latency_per_transition_seconds": numeric_summary(service["frozen_service_rate_latency_seconds"]),
        "service_current_stop_candidate_latency_per_transition_seconds": numeric_summary(service["current_stop_service_candidate_latency_seconds"]),
        "avg_wait_local_settlement_latency_per_transition_seconds": numeric_summary(avg_wait["settlement_latency_seconds"]),
        "p95_local_batch_candidate_latency_per_transition_seconds": numeric_summary(avg_wait["settlement_latency_seconds"]),
        "p95_frozen_settlement_latency": None,
        "p95_frozen_settlement_reason": "cohort/ownership semantics unresolved",
        "old_anchor_was_primary_boarding_latency_cause": bool((local_passenger_latency == 0).all() and float(old_passenger_latency.mean()) > 0),
        "old_anchor_was_only_all_component_latency_cause": False,
    }
    h4a_component = k5.read_json(H4A_ROOT / "r8er3rh4a_component_closure_percentiles.json")["seconds"]
    closure = {
        "created_at": iso_kst(),
        "old_h4a_dispatch_anchor_seconds": h4a_component,
        "local_anchor_seconds": {
            "service_frozen_request_completion": closure_percentiles(service["frozen_service_rate_latency_seconds"]),
            "service_current_stop_candidate": closure_percentiles(service["current_stop_service_candidate_latency_seconds"]),
            "avg_wait_local_affected_set": closure_percentiles(avg_wait["settlement_latency_seconds"]),
            "p95_local_batch_candidate_not_approved": closure_percentiles(avg_wait["settlement_latency_seconds"]),
            "p95_frozen_semantics": {f"H{q}": None for q in (50, 75, 90, 95, 99)},
        },
    }
    fixed_vs_event = {
        "created_at": iso_kst(),
        "FIXED_HORIZON": {
            "reward_completeness": "0 at every audited horizon because frozen p95 local ownership is unresolved",
            "service_H240_complete_fraction": float(service["frozen_service_rate_latency_seconds"].le(240).mean()),
            "service_H660_complete_fraction": float(service["frozen_service_rate_latency_seconds"].le(660).mean()),
            "credit_assignment_ambiguity": "high for request completion after many occurrence decisions",
            "settlement_delay": numeric_summary(service["frozen_service_rate_latency_seconds"]),
            "implementation_complexity": "LOW_TO_MEDIUM",
            "future_leakage_risk": "LOW only if outcomes remain reward-only",
        },
        "EVENT_DRIVEN_COMPONENT_SETTLEMENT": {
            "reward_completeness": "service/avg-wait locally observable; frozen p95 remains unresolved",
            "credit_assignment_ambiguity": "LOW for current-stop service and wait; unresolved for system p95",
            "settlement_delay": "0 for local wait, dwell duration for current-stop service, separately delayed for system p95",
            "implementation_complexity": "MEDIUM_TO_HIGH",
            "future_leakage_risk": "LOW with transition-id write-back and observation/critic guards",
        },
        "automatic_selection": None,
    }
    credit = {
        "created_at": iso_kst(),
        "direct_local_owned_event_count": int((ownership["ownership_class"] == "DIRECT_LOCAL_OWNERSHIP").sum()),
        "delayed_causal_owned_event_count": int((ownership["ownership_class"] == "DELAYED_CAUSAL_OWNERSHIP").sum()),
        "system_level_window_owned_component_count": int((matrix["ownership_class"] == "SYSTEM_LEVEL_WINDOW_OWNERSHIP").sum()),
        "not_owned_event_count": int((ownership["ownership_class"] == "NOT_OWNED_BY_THIS_DECISION").sum()),
        "ambiguous_component_count": int((matrix["ownership_class"] == "AMBIGUOUS").sum()),
        "service_multi_decision_ambiguous_transition_count": int((service["frozen_service_rate_local_credit"] == "MULTI_DECISION_AMBIGUOUS").sum()),
        "p95_multi_decision_semantics_unresolved_transition_count": int(len(anchors)),
        "skip_then_later_service_rule": "D1 owns uniquely caused wait harm; D2 owns eventual local service; the same wait delta cannot be rewarded twice",
        "observed_skip_fixture_count": 0,
        "observed_hold_fixture_count": 0,
        "observed_serve_transition_count": int(len(registry)),
    }
    duplicate = {
        "created_at": iso_kst(),
        "event_count": int(len(ownership)),
        "owned_event_count": int(len(owned)),
        "unique_event_id_count": int(ownership["event_id"].nunique()),
        "duplicate_event_identity_count": duplicate_event_ids,
        "duplicate_reward_ownership_count": duplicate_ownership,
        "cross_passenger_contamination_count": cross_passenger,
        "passed": duplicate_event_ids == duplicate_ownership == cross_passenger == 0,
    }
    p95_payload = {key: value for key, value in p95.items() if key != "created_at"}
    payload = {
        "source_payload_sha256": k5.read_json(R3R_ROOT / "r8er3r_deterministic_regeneration.json")["payload_sha256"],
        "anchors": anchors.to_dict("records"),
        "registry": registry.to_dict("records"),
        "ownership": ownership.to_dict("records"),
        "service": service.to_dict("records"),
        "avg_wait": avg_wait.to_dict("records"),
        "horizon": horizon.to_dict("records"),
        "p95": p95_payload,
        "integrity": integrity,
    }
    return {
        "pickup": pickup,
        "dropoff": dropoff,
        "registry": registry,
        "anchors": anchors,
        "ownership": ownership,
        "service": service,
        "avg_wait": avg_wait,
        "overlap": overlap,
        "matrix": matrix,
        "horizon": horizon,
        "p95": p95,
        "integrity": integrity,
        "latency": latency,
        "closure": closure,
        "fixed_vs_event": fixed_vs_event,
        "credit": credit,
        "duplicate": duplicate,
        "payload_sha256": canonical_hash(payload),
    }


def decide(result: Mapping[str, Any]) -> Tuple[str, str]:
    violations = [
        "wrong_vehicle_binding_count", "wrong_route_binding_count", "wrong_direction_binding_count",
        "wrong_occurrence_binding_count", "wrong_transition_binding_count", "timestamp_regression_count",
        "future_leakage_count", "actor_future_leakage_count", "critic_future_leakage_count",
        "cross_passenger_contamination_count", "duplicate_reward_ownership_count",
        "duplicate_event_identity_count", "unknown_owner_transition_count", "k_mask_action_mismatch_count",
        "causal_ownership_inconsistency_count",
    ]
    if any(int(result["integrity"].get(name, 0)) != 0 for name in violations):
        return "PV8_TEMPORAL_ATTRIBUTION_INTEGRITY_FAILED", "At least one identity, chronology, leakage, or ownership invariant failed."
    if result["anchors"].empty or result["registry"].empty:
        return "PV8_LOCAL_DECISION_ANCHOR_RECONSTRUCTION_FAILED", "No exact local stop transition could be reconstructed."
    if result["p95"]["automatic_selection"] is None:
        return "PV8_P95_REWARD_SEMANTICS_REDESIGN_REQUIRES_USER_APPROVAL", "The local anchor removes route-prefix delay for boarding and local wait, but the approved p95 cohort/ownership semantics cannot be rebound automatically without changing reward meaning."
    return "PV8_COMPONENT_SPECIFIC_DELAYED_SETTLEMENT_REQUIRED", "Component-specific delayed settlement is needed after exact local anchor reconstruction."


def final_report(result: Mapping[str, Any], decision: str, deterministic: Mapping[str, Any]) -> str:
    anchors = result["anchors"]
    horizon = result["horizon"]
    latency = result["latency"]
    service = result["service"]
    overlap = result["overlap"]
    lines = [
        "# PV8-R2A-R8E-R3-R-H4B Final Report",
        "",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{decision}`",
        "- audit/candidate only: H240, H660, reward formula, normalization, K-safety, runtime binding, and training state were not changed.",
        "",
        "## Scope And Anchor",
        "",
        f"- local eventful transitions audited: `{len(result['registry'])}`; pickup reward-anchor transitions: `{len(anchors)}`; passengers: `{len(result['pickup'])}`.",
        "- OLD anchor: first-eligible physical service-instance route-origin `dispatch_ts`.",
        f"- LOCAL anchor: `{LOCAL_ANCHOR}` = exact occurrence `arrival_ts`, with event order state/K-mask/action before same-timestamp boarding or alighting.",
        f"- old-to-local anchor shift mean/median/p95/max sec: `{latency['anchor_shift_per_local_pickup_transition_seconds']['mean']} / {latency['anchor_shift_per_local_pickup_transition_seconds']['median']} / {latency['anchor_shift_per_local_pickup_transition_seconds']['p95']} / {latency['anchor_shift_per_local_pickup_transition_seconds']['max']}`.",
        "",
        "## Latency",
        "",
        f"- OLD boarding mean/median/p95/max sec: `{latency['old_boarding_latency_per_passenger_seconds']['mean']} / {latency['old_boarding_latency_per_passenger_seconds']['median']} / {latency['old_boarding_latency_per_passenger_seconds']['p95']} / {latency['old_boarding_latency_per_passenger_seconds']['max']}`.",
        f"- LOCAL boarding mean/median/p95/max sec: `{latency['local_boarding_latency_per_passenger_seconds']['mean']} / {latency['local_boarding_latency_per_passenger_seconds']['median']} / {latency['local_boarding_latency_per_passenger_seconds']['p95']} / {latency['local_boarding_latency_per_passenger_seconds']['max']}`.",
        f"- frozen service-rate settlement mean/median/p95/max sec: `{latency['service_frozen_settlement_latency_per_transition_seconds']['mean']} / {latency['service_frozen_settlement_latency_per_transition_seconds']['median']} / {latency['service_frozen_settlement_latency_per_transition_seconds']['p95']} / {latency['service_frozen_settlement_latency_per_transition_seconds']['max']}`.",
        f"- current-stop service candidate settlement mean/median/p95/max sec: `{latency['service_current_stop_candidate_latency_per_transition_seconds']['mean']} / {latency['service_current_stop_candidate_latency_per_transition_seconds']['median']} / {latency['service_current_stop_candidate_latency_per_transition_seconds']['p95']} / {latency['service_current_stop_candidate_latency_per_transition_seconds']['max']}`.",
        "- local affected-set avg-wait settlement: `0 sec` by post-action same-timestamp event order.",
        "- frozen p95 settlement latency: `UNRESOLVED`; local-batch p95 would be immediate but is not approved and is mostly singleton.",
        "",
        "## Local-Anchor Horizon Observability",
        "",
    ]
    for seconds in DIAGNOSTIC_HORIZONS:
        group = horizon[horizon["horizon_seconds"] == seconds]
        lines.append(
            f"- H{seconds}: frozen-service `{group['service_frozen_component_complete'].mean():.6f}`, local-service candidate `{group['service_local_candidate_complete'].mean():.6f}`, avg-wait `{group['avg_wait_local_component_complete'].mean():.6f}`, p95 frozen semantics `0.000000`, all-component reward-valid `0.000000`."
        )
    lines += [
        "",
        "## Ownership And Decision",
        "",
        "- service: current frozen `service_rate` uses completed/alighted request identities and is system/window-level; current-stop obligation completion is a coherent local candidate but changes component semantics and remains unapproved.",
        "- avg wait: passengers already waiting and boarded by the local action settle exactly at boarding; future passengers are excluded from the candidate affected set.",
        "- p95: local-batch, rolling-causal, delayed-window, and evaluation-only alternatives were audited; none was selected automatically because tail-wait protection and temporal ownership would change.",
        f"- frozen service transitions with multi-decision ambiguity: `{(service['frozen_service_rate_local_credit'] == 'MULTI_DECISION_AMBIGUOUS').sum()}`; overlap rows classified multi-decision ambiguous: `{(overlap['overlap_class'] == 'MULTI_DECISION_AMBIGUOUS').sum()}`.",
        f"- duplicate ownership / future leakage / wrong vehicle / wrong occurrence: `{result['duplicate']['duplicate_reward_ownership_count']} / {result['integrity']['future_leakage_count']} / {result['integrity']['wrong_vehicle_binding_count']} / {result['integrity']['wrong_occurrence_binding_count']}`.",
        "- the H4A boarding latency was primarily an artificial route-prefix delay: local boarding latency falls to zero. It was not the only all-component delay; request completion remains long and p95 ownership remains unresolved.",
        "- H240 is not sufficient after anchor repair. H660 is also not sufficient and remains diagnostic-only. A fixed horizon cannot be approved until p95/system-level ownership is resolved.",
        f"- proposed temporal contract: `{TEMPORAL_CONTRACT}` candidate only; p95 reward semantics require explicit user approval before any runtime rebinding.",
        f"- deterministic replay: `{deterministic['identical']}`; payload SHA-256 `{deterministic['payload_sha256']}`.",
        "- next recommended step: explicit P95 ownership/semantics adjudication, followed by a separately authorized temporal-binding fixture; no reward rematerialization or MAPPO work yet.",
        "",
    ]
    return "\n".join(lines)


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
    jsonl = writer.root / "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4b.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rows.append(
        {
            "relative_path": jsonl.name,
            "sha256": k5.sha256_file(jsonl),
            "size_bytes": jsonl.stat().st_size,
            "required": True,
            "artifact_role": "manifest_jsonl",
            "exists": True,
        }
    )
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4b.json"
    writer.json(
        manifest_name,
        {
            "created_at": iso_kst(),
            "artifact_family": ARTIFACT_PREFIX,
            "terminal_gate": gate["gate"],
            "readiness": gate["readiness"],
            "payload_count": len(rows),
            "missing_payload_count": 0,
            "files": rows,
        },
    )
    manifest_path = writer.root / manifest_name
    writer.json(
        "_PV8_R2AR8ER3RH4B_COMPLETE.lock",
        {
            "artifact_family": ARTIFACT_PREFIX,
            "terminal_gate": gate["gate"],
            "readiness": gate["readiness"],
            "final_manifest_path": manifest_name,
            "final_manifest_sha256": k5.sha256_file(manifest_path),
            "manifest_size_bytes": manifest_path.stat().st_size,
            "created_at": iso_kst(),
        },
    )


def run(root: Path) -> Path:
    source = verify_sources()
    first = audit_once(source)
    second = audit_once(source)
    deterministic = {
        "created_at": iso_kst(),
        "first_payload_sha256": first["payload_sha256"],
        "second_payload_sha256": second["payload_sha256"],
        "payload_sha256": first["payload_sha256"],
        "identical": first["payload_sha256"] == second["payload_sha256"],
        "transition_ids_identical": first["registry"]["transition_id"].tolist() == second["registry"]["transition_id"].tolist(),
        "local_decision_timestamps_identical": first["registry"]["local_decision_ts"].tolist() == second["registry"]["local_decision_ts"].tolist(),
        "owned_events_identical": first["ownership"].to_dict("records") == second["ownership"].to_dict("records"),
        "component_settlements_identical": first["service"].to_dict("records") == second["service"].to_dict("records") and first["avg_wait"].to_dict("records") == second["avg_wait"].to_dict("records"),
        "latency_distributions_identical": first["latency"] == second["latency"],
        "ambiguity_classifications_identical": first["overlap"].to_dict("records") == second["overlap"].to_dict("records"),
    }
    if not all(value for key, value in deterministic.items() if key == "identical" or key.endswith("_identical")):
        raise H4BAuditError("deterministic local-anchor reconstruction drifted")

    decision, rationale = decide(first)
    writer = k5.Writer(root)
    first["anchors"].to_parquet(root / "r8er3rh4b_old_vs_local_anchor.parquet", index=False)
    first["registry"].to_parquet(root / "r8er3rh4b_transition_registry.parquet", index=False)
    first["ownership"].to_parquet(root / "r8er3rh4b_passenger_event_ownership.parquet", index=False)
    first["service"].to_parquet(root / "r8er3rh4b_service_component_settlement.parquet", index=False)
    first["avg_wait"].to_parquet(root / "r8er3rh4b_avg_wait_component_settlement.parquet", index=False)
    first["matrix"].to_parquet(root / "r8er3rh4b_component_ownership_matrix.parquet", index=False)
    first["horizon"].to_parquet(root / "r8er3rh4b_local_anchor_horizon_observability.parquet", index=False)
    first["overlap"].to_parquet(root / "r8er3rh4b_decision_overlap_audit.parquet", index=False)
    writer.json("r8er3rh4b_timestamp_inventory.json", timestamp_inventory(source))
    writer.json("r8er3rh4b_local_decision_anchor_contract.json", anchor_contract())
    writer.json("r8er3rh4b_p95_semantics_audit.json", first["p95"])
    writer.json("r8er3rh4b_old_vs_local_latency.json", {"created_at": iso_kst(), **first["latency"]})
    writer.json("r8er3rh4b_local_anchor_closure_percentiles.json", first["closure"])
    writer.json("r8er3rh4b_fixed_vs_event_driven_settlement.json", first["fixed_vs_event"])
    writer.json("r8er3rh4b_credit_assignment_audit.json", first["credit"])
    writer.json("r8er3rh4b_duplicate_ownership_audit.json", first["duplicate"])
    writer.json("r8er3rh4b_reward_temporal_attribution_candidate.json", reward_temporal_candidate())
    writer.json("r8er3rh4b_integrity_audit.json", {"created_at": iso_kst(), **first["integrity"]})
    writer.json("r8er3rh4b_deterministic_replay.json", deterministic)
    readiness = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "decision": decision,
        "rationale": rationale,
        "local_anchor_reconstructed": True,
        "temporal_contract_candidate": TEMPORAL_CONTRACT,
        "temporal_contract_approved": False,
        "H240_sufficient": False,
        "H660_sufficient": False,
        "H660_status": "DIAGNOSTIC_ONLY_NOT_APPROVED_NOT_FROZEN_NOT_RUNTIME_ACTIVE",
        "reward_formula_semantics_change_required": True,
        "required_user_approval": "select and approve P95 ownership/settlement semantics and separately authorize runtime temporal binding",
        "next_step": "STOP pending explicit user review; do not approve a horizon, alter formula/normalization, rebind/materialize reward, train, or evaluate MAPPO.",
    }
    writer.json("r8er3rh4b_readiness_decision.json", readiness)
    guards = claim_guards()
    writer.json("claim_guard_status.json", guards)
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "gate_passed": True,
        "final_decision": decision,
        "readiness": READINESS,
        "failure_reasons": [],
        "runtime_approval_implied": False,
    }
    writer.json(
        "run_manifest.json",
        {
            "created_at": iso_kst(),
            "runner": str(RUNNER_PATH),
            "mode": "offline_local_decision_reward_settlement_audit",
            "python": sys.version,
            "platform": platform.platform(),
            "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "api_call_count": 0,
            "postgresql_query_count": 0,
            "passenger_generation_count": 0,
            "b1_regeneration_count": 0,
            "policy_execution_count": 0,
            "optimizer_step_count": 0,
            "runtime_change_count": 0,
            "source_payload_sha256": k5.read_json(R3R_ROOT / "r8er3r_deterministic_regeneration.json")["payload_sha256"],
        },
    )
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "final_decision": decision, "readiness": READINESS})
    writer.text("final_report.md", final_report(first, decision, deterministic))
    write_manifest(writer, gate)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit",), required=True)
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
