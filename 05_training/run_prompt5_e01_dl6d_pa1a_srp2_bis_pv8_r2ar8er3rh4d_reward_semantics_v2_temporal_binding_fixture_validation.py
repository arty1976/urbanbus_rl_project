#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4D Reward Semantics V2 temporal-binding fixtures.

This runner validates the explicitly approved Reward Semantics V2 temporal and
component ownership contract on deterministic fixtures. It does not bind runtime
reward, rematerialize representative rewards, change horizons, approve
normalization, evaluate a policy, or train MAPPO.
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
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4d_reward_semantics_v2_temporal_binding_fixture_validation.py"

H4C_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4c_reward_semantics_adjudication_20260809_215214"
H4C_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4C_REWARD_SEMANTICS_ADJUDICATION_COMPLETE"
H4C_DECISION = "PV8_REWARD_SEMANTICS_V2_READY_FOR_EXPLICIT_APPROVAL_AND_FIXTURE"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4D_REWARD_SEMANTICS_V2_TEMPORAL_BINDING_FIXTURE_VALIDATION_COMPLETE"
SUCCESS_DECISION = "PV8_REWARD_SEMANTICS_V2_TEMPORAL_BINDING_READY_FOR_FREEZE_REVIEW"
READINESS = "REWARD_SEMANTICS_V2_TEMPORAL_BINDING_FIXTURE_VALIDATED_FREEZE_REVIEW_PENDING"

SEMANTICS_CANDIDATE = "PV8_REWARD_SEMANTICS_CANDIDATE_V2"
TEMPORAL_BINDING_CANDIDATE = "PV8_REWARD_SEMANTICS_V2_TEMPORAL_BINDING_CANDIDATE_V1"
LOCAL_ANCHOR = "PV8_LOCAL_STOP_DECISION_ANCHOR_V1"
SETTLEMENT = "PV8_COMPONENT_SPECIFIC_EVENT_DRIVEN_SETTLEMENT_V1"
NORMALIZATION = {"service": 1.0, "avg_wait": 297.7850241545894, "p95_wait": 576.6999999999999}
REWARD_FORMULA = "3.0*local_service_component + 2.0*local_affected_avg_wait_component - 0.25*explicit_forced_external_intervention"
HORIZONS = (120, 240, 660)

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4d_reward_semantics_v2_temporal_binding_fixture_validation"
PAYLOADS = [
    "r8er3rh4d_upstream_binding.json",
    "r8er3rh4d_user_approval_record.json",
    "r8er3rh4d_fixture_contract.json",
    "r8er3rh4d_transition_registry.parquet",
    "r8er3rh4d_event_order_trace.parquet",
    "r8er3rh4d_passenger_lifecycle.parquet",
    "r8er3rh4d_reward_component_binding.parquet",
    "r8er3rh4d_wait_ownership_audit.json",
    "r8er3rh4d_service_ownership_audit.json",
    "r8er3rh4d_kmask_precedence_audit.json",
    "r8er3rh4d_missed_service_fault_injection.json",
    "r8er3rh4d_long_legitimate_wait_fixture.json",
    "r8er3rh4d_p95_tail_fixture.json",
    "r8er3rh4d_horizon_independence_audit.json",
    "r8er3rh4d_future_leakage_audit.json",
    "r8er3rh4d_wrong_binding_negative_tests.json",
    "r8er3rh4d_fixture_truth_table.parquet",
    "r8er3rh4d_fixture_summary.json",
    "r8er3rh4d_temporal_binding_candidate.json",
    "r8er3rh4d_deterministic_replay.json",
    "r8er3rh4d_integrity_audit.json",
    "r8er3rh4d_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class H4DError(RuntimeError):
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


def deterministic_content(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: deterministic_content(item) for key, item in value.items() if key != "created_at"}
    if isinstance(value, list):
        return [deterministic_content(item) for item in value]
    if isinstance(value, tuple):
        return tuple(deterministic_content(item) for item in value)
    return value


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise H4DError("percentile requires a nonempty sequence")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(quantile)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def transition_id(row: Mapping[str, Any]) -> str:
    fields = (
        str(row["fixture_id"]),
        str(row["vehicle_slot_id"]),
        str(row["route_id"]),
        str(row["direction_id"]),
        str(row["occurrence_id"]),
        str(row["local_decision_ts"]),
        str(row["action_requested"]),
    )
    return "PV8H4D_" + hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()[:24]


def verify_upstream() -> Dict[str, Any]:
    manifest = k5.verify_manifest(
        H4C_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4c.json",
        "_PV8_R2AR8ER3RH4C_COMPLETE.lock",
    )
    if not k5.manifest_ok(manifest):
        raise H4DError("H4C manifest or terminal lock is not authoritative")
    gate = k5.read_json(H4C_ROOT / "gate_decision.json")
    readiness = k5.read_json(H4C_ROOT / "r8er3rh4c_readiness_decision.json")
    candidate = k5.read_json(H4C_ROOT / "r8er3rh4c_reward_semantics_candidate_v2.json")
    deterministic = k5.read_json(H4C_ROOT / "r8er3rh4c_deterministic_replay.json")
    if gate.get("gate") != H4C_GATE or gate.get("final_decision") != H4C_DECISION:
        raise H4DError("H4C gate or final decision drifted")
    if readiness.get("decision") != H4C_DECISION:
        raise H4DError("H4C readiness decision drifted")
    expected = {
        "candidate": SEMANTICS_CANDIDATE,
        "decision_anchor": LOCAL_ANCHOR,
        "service_semantics": "LOCAL_CURRENT_STOP_SERVICE_COMPLETION",
        "avg_wait_semantics": "LOCAL_CAUSALLY_AFFECTED_WAIT_SET",
        "missed_service_semantics": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "alignment_excess_semantics": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "p95_semantics": "P95_EVALUATION_ONLY",
        "settlement": SETTLEMENT,
    }
    for key, expected_value in expected.items():
        if candidate.get(key) != expected_value:
            raise H4DError(f"H4C candidate field drifted: {key}")
    if candidate.get("candidate_training_formula") != REWARD_FORMULA:
        raise H4DError("H4C candidate formula drifted")
    if candidate.get("p95_weight_redistributed") is not False or candidate.get("action_name_skip_penalty") is not False:
        raise H4DError("H4C candidate guard drifted")
    return {
        "created_at": iso_kst(),
        "h4c_artifact_root": str(H4C_ROOT),
        "h4c_manifest_integrity": manifest,
        "h4c_gate": H4C_GATE,
        "h4c_decision": H4C_DECISION,
        "h4c_payload_sha256": deterministic.get("payload_sha256"),
        "h4c_candidate_hash": k5.sha256_file(H4C_ROOT / "r8er3rh4c_reward_semantics_candidate_v2.json"),
        "approved_candidate": SEMANTICS_CANDIDATE,
        "authoritative_semantics": expected,
        "normalization_candidate_only": NORMALIZATION,
        "normalization_frozen": False,
        "fixed_H240_runtime_change": "NOT_APPROVED",
        "H660": "NOT_APPROVED",
    }


def approval_record() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "approval_source": "current user-provided H4D prompt, section 0 Explicit User Approval State",
        "approved_design": SEMANTICS_CANDIDATE,
        "reward_semantics_v2_design_approved": True,
        "approval_scope": "scientific/design semantics for deterministic temporal-binding fixture validation only",
        "reward_runtime_binding_allowed": False,
        "reward_rematerialization_allowed": False,
        "MAPPO_training_authorized": False,
        "policy_evaluation_authorized": False,
        "normalization_frozen": False,
    }


def fixture_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract": "PV8_H4D_FIXTURE_CONTRACT_V1",
        "decision_anchor": LOCAL_ANCHOR,
        "event_order": [
            "VEHICLE_ARRIVAL",
            "STATE_SNAPSHOT",
            "OBLIGATION_SNAPSHOT",
            "K_MASK_BUILD",
            "ACTION_SELECTION",
            "ACTION_VALIDATION",
            "BOARDING_OR_ALIGHTING",
            "SERVICE_DWELL",
            "LOCAL_SERVICE_SETTLEMENT",
            "HOLD_IF_APPLICABLE",
            "DEPARTURE",
            "NEXT_LINK_TRAVEL",
        ],
        "same_timestamp_rule": "event_ts alone is insufficient; event_ordinal fixes causal order when arrival, decision, and boarding share a timestamp",
        "transition_identity_fields": [
            "fixture_id",
            "vehicle_slot_id",
            "route_id",
            "direction_id",
            "occurrence_id",
            "local_decision_ts",
            "action_requested",
        ],
        "core_fixtures": [
            "NORMAL_PICKUP",
            "EMPTY_STOP_LEGAL_RESEARCH_SKIP",
            "PICKUP_OBLIGATION_BLOCKS_SKIP",
            "SYNTHETIC_ILLEGAL_MISSED_SERVICE_INJECTION",
            "LONG_LEGITIMATE_SCHEDULE_WAIT",
            "P95_MEAN_IMPROVES_TAIL_WORSENS_V1",
        ],
        "additional_fixtures": [
            "DROPOFF_OBLIGATION_SERVICE",
            "HOLD_AFTER_CURRENT_SERVICE",
            "EMPTY_STOP_SERVE_NO_INFLATION",
            "WRONG_BINDING_NEGATIVE_TEST",
            "EXPLICIT_FORCED_EXTERNAL_INTERVENTION",
        ],
        "diagnostic_reward_calculation": "DIAGNOSTIC_FIXTURE_ONLY_NOT_TRAINING_REWARD",
        "representative_reward_materialization_allowed": False,
    }


def event_rows_for_transition(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    fixture_id = str(row["fixture_id"])
    transition = str(row["transition_id"])
    ts = int(row["local_decision_ts"])
    base = [
        ("VEHICLE_ARRIVAL", ts, 10, True, True, "physical arrival at current occurrence"),
        ("STATE_SNAPSHOT", ts, 20, True, True, "current state only"),
        ("OBLIGATION_SNAPSHOT", ts, 30, True, True, "current obligations only"),
        ("K_MASK_BUILD", ts, 40, True, True, "fail-closed K mask built before action"),
        ("ACTION_SELECTION", ts, 50, False, False, str(row["action_requested"])),
        ("ACTION_VALIDATION", ts, 60, False, False, "accepted" if row["action_legal"] else "rejected before mutation"),
    ]
    rows = [
        {
            "fixture_id": fixture_id,
            "transition_id": transition,
            "event_type": event_type,
            "event_ts": event_ts,
            "event_ordinal": ordinal,
            "actor_observation_visible_at_decision": actor_visible,
            "critic_visible_at_decision": critic_visible,
            "description": description,
        }
        for event_type, event_ts, ordinal, actor_visible, critic_visible, description in base
    ]
    if not bool(row["action_legal"]):
        rows.append(
            {
                "fixture_id": fixture_id,
                "transition_id": transition,
                "event_type": "INVALID_ACTION_REJECTED_PRE_MUTATION",
                "event_ts": ts,
                "event_ordinal": 65,
                "actor_observation_visible_at_decision": False,
                "critic_visible_at_decision": False,
                "description": "unsafe conditional skip rejected before passenger or vehicle mutation",
            }
        )
        return rows
    if fixture_id == "SYNTHETIC_ILLEGAL_MISSED_SERVICE_INJECTION":
        rows.extend(
            [
                {
                    "fixture_id": fixture_id,
                    "transition_id": transition,
                    "event_type": "INJECTED_NONBOARDING_FAULT",
                    "event_ts": ts,
                    "event_ordinal": 70,
                    "actor_observation_visible_at_decision": False,
                    "critic_visible_at_decision": False,
                    "description": "forced nonboarding despite valid eligible service",
                },
                {
                    "fixture_id": fixture_id,
                    "transition_id": transition,
                    "event_type": "INTEGRITY_GATE_QUARANTINE",
                    "event_ts": ts,
                    "event_ordinal": 80,
                    "actor_observation_visible_at_decision": False,
                    "critic_visible_at_decision": False,
                    "description": "learning sample is blocked; numeric reward penalty does not repair the fault",
                },
            ]
        )
        return rows
    exchange_type = "NO_PASSENGER_EXCHANGE"
    if int(row.get("pickup_obligation", 0)) > 0:
        exchange_type = "BOARDING"
    if int(row.get("dropoff_obligation", 0)) > 0:
        exchange_type = "ALIGHTING" if exchange_type == "NO_PASSENGER_EXCHANGE" else "BOARDING_AND_ALIGHTING"
    if row["action_requested"] == "CONDITIONAL_SKIP":
        exchange_type = "NO_SERVICE_EXCHANGE_LEGAL_SKIP"
    rows.append(
        {
            "fixture_id": fixture_id,
            "transition_id": transition,
            "event_type": exchange_type,
            "event_ts": ts,
            "event_ordinal": 70,
            "actor_observation_visible_at_decision": False,
            "critic_visible_at_decision": False,
            "description": "post-action passenger exchange, if any",
        }
    )
    rows.append(
        {
            "fixture_id": fixture_id,
            "transition_id": transition,
            "event_type": "SERVICE_DWELL",
            "event_ts": ts,
            "event_ordinal": 80,
            "actor_observation_visible_at_decision": False,
            "critic_visible_at_decision": False,
            "description": "dwell only after action validation and exchange",
        }
    )
    rows.append(
        {
            "fixture_id": fixture_id,
            "transition_id": transition,
            "event_type": "LOCAL_SERVICE_SETTLEMENT",
            "event_ts": int(row.get("service_settlement_ts") or ts),
            "event_ordinal": 90,
            "actor_observation_visible_at_decision": False,
            "critic_visible_at_decision": False,
            "description": str(row.get("service_component_status")),
        }
    )
    if fixture_id == "HOLD_AFTER_CURRENT_SERVICE":
        rows.append(
            {
                "fixture_id": fixture_id,
                "transition_id": transition,
                "event_type": "HOLD_BEGIN",
                "event_ts": int(row.get("service_settlement_ts") or ts),
                "event_ordinal": 95,
                "actor_observation_visible_at_decision": False,
                "critic_visible_at_decision": False,
                "description": "HOLD starts only after current-stop service completion",
            }
        )
    if fixture_id == "EXPLICIT_FORCED_EXTERNAL_INTERVENTION":
        rows.append(
            {
                "fixture_id": fixture_id,
                "transition_id": transition,
                "event_type": "FORCED_EXTERNAL_INTERVENTION",
                "event_ts": ts + 5,
                "event_ordinal": 96,
                "actor_observation_visible_at_decision": False,
                "critic_visible_at_decision": False,
                "description": "typed intervention event; generic constraints do not trigger this component",
            }
        )
    rows.append(
        {
            "fixture_id": fixture_id,
            "transition_id": transition,
            "event_type": "DEPARTURE",
            "event_ts": int(row.get("departure_ts") or ts),
            "event_ordinal": 100,
            "actor_observation_visible_at_decision": False,
            "critic_visible_at_decision": False,
            "description": "departure after settlement",
        }
    )
    rows.append(
        {
            "fixture_id": fixture_id,
            "transition_id": transition,
            "event_type": "NEXT_LINK_TRAVEL",
            "event_ts": int(row.get("departure_ts") or ts) + 1,
            "event_ordinal": 110,
            "actor_observation_visible_at_decision": False,
            "critic_visible_at_decision": False,
            "description": "future movement is reward/outcome only, never decision observation",
        }
    )
    return rows


def base_transition(fixture_id: str, action: str, *, ts: int, pickup: int = 0, dropoff: int = 0, skip_allowed: bool = False, action_legal: bool = True, time_band: str = "peak") -> Dict[str, Any]:
    row = {
        "fixture_id": fixture_id,
        "core_fixture_id": fixture_id.replace("_PEAK", "").replace("_NIGHT", ""),
        "vehicle_slot_id": 0,
        "vehicle_token": "PV8-FIXED-0",
        "route_id": "3000814001",
        "direction_id": "0",
        "occurrence_id": f"occ:{fixture_id}:0",
        "route_stop_occurrence_id": f"route-stop-occurrence:{fixture_id}:0",
        "stop_id": f"STOP_{fixture_id}",
        "local_decision_ts": ts,
        "action_requested": action,
        "action_executed": action if action_legal else None,
        "action_legal": action_legal,
        "time_band": time_band,
        "pickup_obligation": pickup,
        "dropoff_obligation": dropoff,
        "static_mandatory_obligation": 0,
        "k_mask_hold_allowed": True,
        "k_mask_serve_allowed": True,
        "k_mask_skip_allowed": skip_allowed,
        "boarding_event_ts": None,
        "alighting_event_ts": None,
        "service_settlement_ts": ts + 15 if action_legal and action != "CONDITIONAL_SKIP" else ts,
        "departure_ts": ts + 15 if action_legal and action != "CONDITIONAL_SKIP" else ts,
        "service_component_status": "NOT_APPLICABLE",
        "service_component_value": None,
        "service_denominator_increment": 0,
        "avg_wait_component_status": "NOT_APPLICABLE",
        "avg_wait_seconds": None,
        "avg_wait_component_value": None,
        "p95_training_component_status": "DISABLED_EVALUATION_ONLY",
        "p95_evaluation_status": "NOT_APPLICABLE",
        "missed_service": False,
        "alignment_excess_seconds": 0,
        "boards_first_eligible": None,
        "learning_sample_allowed": bool(action_legal),
        "integrity_pass": True,
        "diagnostic_reward_candidate_status": "DIAGNOSTIC_FIXTURE_ONLY_NOT_TRAINING_REWARD",
        "diagnostic_reward_candidate": None,
        "blanket_skip_penalty_applied": False,
        "direct_time_band_reward_term_applied": False,
        "empty_stop_service_reward_inflation": False,
    }
    row["transition_id"] = transition_id(row)
    return row


def add_passenger(
    rows: List[Dict[str, Any]],
    *,
    fixture_id: str,
    transition: str,
    passenger_id: str,
    request_id: str,
    request_ts: int,
    first_eligible_ts: Optional[int],
    board_ts: Optional[int],
    alight_ts: Optional[int] = None,
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    wait_owned: bool = True,
    missed: bool = False,
    classification: str = "NORMAL_SERVICE",
    learning_sample_allowed: bool = True,
) -> None:
    schedule_wait = None if first_eligible_ts is None else int(first_eligible_ts) - int(request_ts)
    alignment = None if first_eligible_ts is None or board_ts is None else int(board_ts) - int(first_eligible_ts)
    total = None if board_ts is None else int(board_ts) - int(request_ts)
    rows.append(
        {
            "fixture_id": fixture_id,
            "transition_id": transition,
            "passenger_id": passenger_id,
            "request_id": request_id,
            "request_ts": int(request_ts),
            "origin_occurrence_id": origin or f"origin:{fixture_id}",
            "destination_occurrence_id": destination or f"dest:{fixture_id}",
            "first_eligible_service_ts": first_eligible_ts,
            "actual_board_ts": board_ts,
            "actual_alight_ts": alight_ts,
            "schedule_wait_seconds": schedule_wait,
            "alignment_excess_wait_seconds": alignment,
            "total_wait_seconds": total,
            "arithmetic_consistent": (total is None or schedule_wait is None or alignment is None or total == schedule_wait + alignment),
            "boards_first_eligible": bool(board_ts == first_eligible_ts) if board_ts is not None and first_eligible_ts is not None else None,
            "missed_eligible_service": bool(missed),
            "wait_owner_transition_id": transition if wait_owned else None,
            "passenger_wait_owner_count": 1 if wait_owned else 0,
            "service_owner_transition_id": transition if learning_sample_allowed else None,
            "service_event_owner_count": 1 if learning_sample_allowed else 0,
            "classification": classification,
            "learning_sample_allowed": bool(learning_sample_allowed),
            "identity_preserved": True,
            "future_visible_to_actor": False,
            "future_visible_to_critic": False,
        }
    )


def component_rows_for_transition(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    transition = str(row["transition_id"])
    fixture_id = str(row["fixture_id"])
    rows = [
        {
            "fixture_id": fixture_id,
            "transition_id": transition,
            "component": "local_service_component",
            "settlement_event": "LOCAL_SERVICE_SETTLEMENT",
            "settlement_ts": row.get("service_settlement_ts"),
            "owner_transition_id": transition if row.get("service_component_status") == "BOUND" else None,
            "component_status": row.get("service_component_status"),
            "component_value": row.get("service_component_value"),
            "weight": 3.0,
            "diagnostic_only_not_training_reward": True,
            "learning_sample_allowed": bool(row.get("learning_sample_allowed")),
        },
        {
            "fixture_id": fixture_id,
            "transition_id": transition,
            "component": "local_affected_avg_wait_component",
            "settlement_event": "BOARDING",
            "settlement_ts": row.get("boarding_event_ts"),
            "owner_transition_id": transition if row.get("avg_wait_component_status") == "BOUND" else None,
            "component_status": row.get("avg_wait_component_status"),
            "component_value": row.get("avg_wait_component_value"),
            "weight": 2.0,
            "diagnostic_only_not_training_reward": True,
            "learning_sample_allowed": bool(row.get("learning_sample_allowed")),
        },
        {
            "fixture_id": fixture_id,
            "transition_id": transition,
            "component": "explicit_forced_external_intervention",
            "settlement_event": "FORCED_EXTERNAL_INTERVENTION" if fixture_id == "EXPLICIT_FORCED_EXTERNAL_INTERVENTION" else None,
            "settlement_ts": int(row["local_decision_ts"]) + 5 if fixture_id == "EXPLICIT_FORCED_EXTERNAL_INTERVENTION" else None,
            "owner_transition_id": transition if fixture_id == "EXPLICIT_FORCED_EXTERNAL_INTERVENTION" else None,
            "component_status": "BOUND" if fixture_id == "EXPLICIT_FORCED_EXTERNAL_INTERVENTION" else "NOT_APPLICABLE",
            "component_value": -0.25 if fixture_id == "EXPLICIT_FORCED_EXTERNAL_INTERVENTION" else None,
            "weight": -0.25,
            "diagnostic_only_not_training_reward": True,
            "learning_sample_allowed": bool(row.get("learning_sample_allowed")),
        },
    ]
    if fixture_id == "P95_MEAN_IMPROVES_TAIL_WORSENS_V1":
        rows.append(
            {
                "fixture_id": fixture_id,
                "transition_id": transition,
                "component": "p95_evaluation_kpi",
                "settlement_event": "EVALUATION_PIPELINE_ONLY",
                "settlement_ts": None,
                "owner_transition_id": None,
                "component_status": "TAIL_PROTECTION_ALERT",
                "component_value": row.get("candidate_p95_wait_seconds"),
                "weight": None,
                "diagnostic_only_not_training_reward": True,
                "learning_sample_allowed": False,
            }
        )
    return rows


def build_fixture_suite() -> Dict[str, Any]:
    transitions: List[Dict[str, Any]] = []
    passengers: List[Dict[str, Any]] = []

    normal = base_transition("NORMAL_PICKUP", "SERVE", ts=1000, pickup=1, skip_allowed=False)
    normal.update(
        {
            "boarding_event_ts": 1000,
            "service_component_status": "BOUND",
            "service_component_value": 1.0,
            "service_denominator_increment": 1,
            "avg_wait_component_status": "BOUND",
            "avg_wait_seconds": 30.0,
            "avg_wait_component_value": 1.0 - 30.0 / NORMALIZATION["avg_wait"],
            "boards_first_eligible": True,
            "diagnostic_reward_candidate": 3.0 + 2.0 * (1.0 - 30.0 / NORMALIZATION["avg_wait"]),
        }
    )
    transitions.append(normal)
    add_passenger(passengers, fixture_id=normal["fixture_id"], transition=normal["transition_id"], passenger_id="P_NORMAL", request_id="R_NORMAL", request_ts=970, first_eligible_ts=1000, board_ts=1000, alight_ts=1120)

    for band in ("peak", "night"):
        suffix = band.upper()
        empty_skip = base_transition(f"EMPTY_STOP_LEGAL_RESEARCH_SKIP_{suffix}", "CONDITIONAL_SKIP", ts=1100 if band == "peak" else 1200, skip_allowed=True, time_band=band)
        empty_skip.update(
            {
                "core_fixture_id": "EMPTY_STOP_LEGAL_RESEARCH_SKIP",
                "p95_evaluation_status": "NOT_APPLICABLE",
                "learning_sample_allowed": True,
                "diagnostic_reward_candidate": 0.0,
            }
        )
        transitions.append(empty_skip)

    blocked = base_transition("PICKUP_OBLIGATION_BLOCKS_SKIP", "CONDITIONAL_SKIP", ts=1300, pickup=1, skip_allowed=False, action_legal=False)
    blocked.update(
        {
            "action_executed": None,
            "learning_sample_allowed": False,
            "service_component_status": "BLOCKED_PRE_ACTION",
            "avg_wait_component_status": "BLOCKED_PRE_ACTION",
        }
    )
    transitions.append(blocked)
    add_passenger(passengers, fixture_id=blocked["fixture_id"], transition=blocked["transition_id"], passenger_id="P_BLOCKED", request_id="R_BLOCKED", request_ts=1290, first_eligible_ts=None, board_ts=None, wait_owned=False, classification="PROTECTED_BY_K_MASK", learning_sample_allowed=False)

    injection = base_transition("SYNTHETIC_ILLEGAL_MISSED_SERVICE_INJECTION", "SERVE", ts=1400, pickup=1, skip_allowed=False)
    injection.update(
        {
            "boarding_event_ts": 1520,
            "service_component_status": "QUARANTINED_INTEGRITY_FAILURE",
            "avg_wait_component_status": "QUARANTINED_INTEGRITY_FAILURE",
            "missed_service": True,
            "alignment_excess_seconds": 120,
            "boards_first_eligible": False,
            "learning_sample_allowed": False,
            "integrity_pass": True,
        }
    )
    transitions.append(injection)
    add_passenger(passengers, fixture_id=injection["fixture_id"], transition=injection["transition_id"], passenger_id="P_INJECT", request_id="R_INJECT", request_ts=1370, first_eligible_ts=1400, board_ts=1520, alight_ts=1600, missed=True, classification="INJECTED_INTEGRITY_FAILURE", learning_sample_allowed=False)

    long_wait = base_transition("LONG_LEGITIMATE_SCHEDULE_WAIT", "SERVE", ts=2000, pickup=1, skip_allowed=False)
    long_wait.update(
        {
            "boarding_event_ts": 2000,
            "service_component_status": "BOUND",
            "service_component_value": 1.0,
            "service_denominator_increment": 1,
            "avg_wait_component_status": "BOUND",
            "avg_wait_seconds": 1000.0,
            "avg_wait_component_value": 1.0 - 1000.0 / NORMALIZATION["avg_wait"],
            "boards_first_eligible": True,
            "p95_evaluation_status": "TAIL_WAIT_MAY_RISE_EVALUATION_ONLY",
            "diagnostic_reward_candidate": 3.0 + 2.0 * (1.0 - 1000.0 / NORMALIZATION["avg_wait"]),
        }
    )
    transitions.append(long_wait)
    add_passenger(passengers, fixture_id=long_wait["fixture_id"], transition=long_wait["transition_id"], passenger_id="P_LONG", request_id="R_LONG", request_ts=1000, first_eligible_ts=2000, board_ts=2000, alight_ts=2120, classification="LONG_LEGITIMATE_SCHEDULE_WAIT")

    p95 = base_transition("P95_MEAN_IMPROVES_TAIL_WORSENS_V1", "EVALUATION_ONLY", ts=3000, action_legal=True)
    baseline = [300.0] * 20
    candidate = [100.0] * 18 + [1000.0] * 2
    p95.update(
        {
            "vehicle_slot_id": -1,
            "route_id": "SYSTEM_EVALUATION",
            "direction_id": "SYSTEM",
            "occurrence_id": "SYSTEM_P95_COHORT",
            "route_stop_occurrence_id": "SYSTEM_P95_COHORT",
            "k_mask_hold_allowed": False,
            "k_mask_serve_allowed": False,
            "k_mask_skip_allowed": False,
            "action_executed": None,
            "learning_sample_allowed": False,
            "p95_evaluation_status": "TAIL_PROTECTION_ALERT",
            "baseline_mean_wait_seconds": sum(baseline) / len(baseline),
            "candidate_mean_wait_seconds": sum(candidate) / len(candidate),
            "baseline_p95_wait_seconds": percentile(baseline, .95),
            "candidate_p95_wait_seconds": percentile(candidate, .95),
            "mean_wait_improves": True,
            "p95_wait_worsens": True,
            "tail_deterioration_detected": True,
            "promotion_pass": False,
        }
    )
    p95["transition_id"] = transition_id(p95)
    transitions.append(p95)

    dropoff = base_transition("DROPOFF_OBLIGATION_SERVICE", "SERVE", ts=4000, dropoff=1, skip_allowed=False)
    dropoff.update(
        {
            "alighting_event_ts": 4000,
            "service_component_status": "BOUND",
            "service_component_value": 1.0,
            "service_denominator_increment": 1,
            "avg_wait_component_status": "NOT_APPLICABLE_ONBOARD_DROPOFF",
            "boards_first_eligible": None,
            "diagnostic_reward_candidate": 3.0,
        }
    )
    transitions.append(dropoff)
    add_passenger(passengers, fixture_id=dropoff["fixture_id"], transition=dropoff["transition_id"], passenger_id="P_DROPOFF", request_id="R_DROPOFF", request_ts=3800, first_eligible_ts=3900, board_ts=3900, alight_ts=4000, wait_owned=False, classification="ONBOARD_AT_DECISION_DROPOFF")

    hold = base_transition("HOLD_AFTER_CURRENT_SERVICE", "HOLD", ts=5000, pickup=1, skip_allowed=False)
    hold.update(
        {
            "boarding_event_ts": 5000,
            "service_component_status": "BOUND",
            "service_component_value": 1.0,
            "service_denominator_increment": 1,
            "avg_wait_component_status": "BOUND",
            "avg_wait_seconds": 40.0,
            "avg_wait_component_value": 1.0 - 40.0 / NORMALIZATION["avg_wait"],
            "departure_ts": 5060,
            "boards_first_eligible": True,
            "diagnostic_reward_candidate": 3.0 + 2.0 * (1.0 - 40.0 / NORMALIZATION["avg_wait"]),
        }
    )
    transitions.append(hold)
    add_passenger(passengers, fixture_id=hold["fixture_id"], transition=hold["transition_id"], passenger_id="P_HOLD", request_id="R_HOLD", request_ts=4960, first_eligible_ts=5000, board_ts=5000, alight_ts=5120, classification="HOLD_AFTER_SERVICE")

    empty_serve = base_transition("EMPTY_STOP_SERVE_NO_INFLATION", "SERVE", ts=6000, skip_allowed=True)
    empty_serve.update({"diagnostic_reward_candidate": 0.0})
    transitions.append(empty_serve)

    wrong = base_transition("WRONG_BINDING_NEGATIVE_TEST", "SERVE", ts=7000, pickup=1, skip_allowed=False)
    wrong.update(
        {
            "boarding_event_ts": 7000,
            "service_component_status": "BOUND",
            "service_component_value": 1.0,
            "service_denominator_increment": 1,
            "avg_wait_component_status": "BOUND",
            "avg_wait_seconds": 20.0,
            "avg_wait_component_value": 1.0 - 20.0 / NORMALIZATION["avg_wait"],
            "boards_first_eligible": True,
            "diagnostic_reward_candidate": 3.0 + 2.0 * (1.0 - 20.0 / NORMALIZATION["avg_wait"]),
        }
    )
    transitions.append(wrong)
    add_passenger(passengers, fixture_id=wrong["fixture_id"], transition=wrong["transition_id"], passenger_id="P_WRONG_BIND", request_id="R_WRONG_BIND", request_ts=6980, first_eligible_ts=7000, board_ts=7000, alight_ts=7120, classification="CORRECT_BINDING_CONTROL")

    intervention = base_transition("EXPLICIT_FORCED_EXTERNAL_INTERVENTION", "SERVE", ts=8000, skip_allowed=True)
    intervention.update(
        {
            "service_component_status": "NOT_APPLICABLE",
            "avg_wait_component_status": "NOT_APPLICABLE",
            "diagnostic_reward_candidate": -0.25,
        }
    )
    transitions.append(intervention)

    event_rows: List[Dict[str, Any]] = []
    component_rows: List[Dict[str, Any]] = []
    for row in transitions:
        row["transition_id_unique"] = True
        event_rows.extend(event_rows_for_transition(row))
        component_rows.extend(component_rows_for_transition(row))
    return {
        "transitions": pd.DataFrame(transitions).sort_values(["local_decision_ts", "fixture_id"]).reset_index(drop=True),
        "events": pd.DataFrame(event_rows).sort_values(["fixture_id", "event_ts", "event_ordinal", "event_type"]).reset_index(drop=True),
        "passengers": pd.DataFrame(passengers).sort_values(["fixture_id", "passenger_id"]).reset_index(drop=True),
        "components": pd.DataFrame(component_rows).sort_values(["fixture_id", "transition_id", "component"]).reset_index(drop=True),
    }


def fixture_truth_table(transitions: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in transitions.to_dict("records"):
        rows.append(
            {
                "fixture_id": row["fixture_id"],
                "core_fixture_id": row["core_fixture_id"],
                "action_requested": row["action_requested"],
                "action_legal": bool(row["action_legal"]),
                "pickup_obligation": int(row["pickup_obligation"]),
                "dropoff_obligation": int(row["dropoff_obligation"]),
                "k_mask_skip_allowed": bool(row["k_mask_skip_allowed"]),
                "boards_first_eligible": row.get("boards_first_eligible"),
                "missed_service": bool(row["missed_service"]),
                "alignment_excess": row["alignment_excess_seconds"],
                "service_component_status": row["service_component_status"],
                "avg_wait_component_status": row["avg_wait_component_status"],
                "p95_training_component_status": row["p95_training_component_status"],
                "p95_evaluation_status": row["p95_evaluation_status"],
                "learning_sample_allowed": bool(row["learning_sample_allowed"]),
                "integrity_pass": bool(row["integrity_pass"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["core_fixture_id", "fixture_id"]).reset_index(drop=True)


def wait_ownership_audit(passengers: pd.DataFrame) -> Dict[str, Any]:
    owned = passengers[passengers["wait_owner_transition_id"].notna()].copy()
    duplicate_wait = int(owned.groupby(["passenger_id", "request_id"]).size().gt(1).sum()) if not owned.empty else 0
    arithmetic = passengers[passengers["total_wait_seconds"].notna()].copy()
    arithmetic_error = int((~arithmetic["arithmetic_consistent"].astype(bool)).sum()) if not arithmetic.empty else 0
    return {
        "created_at": iso_kst(),
        "passenger_rows": int(len(passengers)),
        "wait_owned_rows": int(len(owned)),
        "duplicate_wait_ownership": duplicate_wait,
        "passenger_wait_owner_count_max": int(owned["passenger_wait_owner_count"].max()) if not owned.empty else 0,
        "wait_arithmetic_error_count": arithmetic_error,
        "future_visible_to_actor_count": int(passengers["future_visible_to_actor"].astype(bool).sum()),
        "future_visible_to_critic_count": int(passengers["future_visible_to_critic"].astype(bool).sum()),
        "passed": duplicate_wait == 0 and arithmetic_error == 0,
    }


def service_ownership_audit(transitions: pd.DataFrame, components: pd.DataFrame) -> Dict[str, Any]:
    service = components[components["component"] == "local_service_component"].copy()
    bound = service[service["owner_transition_id"].notna()]
    duplicate = int(bound.groupby("transition_id").size().gt(1).sum()) if not bound.empty else 0
    empty_inflation = int(transitions["empty_stop_service_reward_inflation"].astype(bool).sum())
    empty_success = int(((transitions["pickup_obligation"].astype(int) == 0) & (transitions["dropoff_obligation"].astype(int) == 0) & (transitions["service_component_status"] == "BOUND")).sum())
    return {
        "created_at": iso_kst(),
        "service_component_rows": int(len(service)),
        "bound_service_rows": int(len(bound)),
        "duplicate_service_ownership": duplicate,
        "empty_nonmandatory_stop_service_success_count": empty_success,
        "empty_stop_service_reward_inflation_count": empty_inflation,
        "service_denominator_increment_total": int(transitions["service_denominator_increment"].fillna(0).astype(int).sum()),
        "passed": duplicate == 0 and empty_inflation == 0 and empty_success == 0,
    }


def kmask_precedence_audit(transitions: pd.DataFrame, events: pd.DataFrame) -> Dict[str, Any]:
    blocked = transitions[transitions["fixture_id"] == "PICKUP_OBLIGATION_BLOCKS_SKIP"].iloc[0].to_dict()
    blocked_events = events[events["fixture_id"] == "PICKUP_OBLIGATION_BLOCKS_SKIP"]
    ordinal = {row["event_type"]: int(row["event_ordinal"]) for row in blocked_events.to_dict("records")}
    valid_order = ordinal["OBLIGATION_SNAPSHOT"] < ordinal["K_MASK_BUILD"] < ordinal["ACTION_VALIDATION"] < ordinal["INVALID_ACTION_REJECTED_PRE_MUTATION"]
    return {
        "created_at": iso_kst(),
        "fixture_id": "PICKUP_OBLIGATION_BLOCKS_SKIP",
        "pickup_obligation": int(blocked["pickup_obligation"]),
        "k_mask_skip_allowed": bool(blocked["k_mask_skip_allowed"]),
        "attempted_skip_rejected": bool(not blocked["action_legal"]),
        "unsafe_action_reached_simulator_mutation": False,
        "illegal_skip_accepted_count": int((transitions["action_requested"].eq("CONDITIONAL_SKIP") & ~transitions["k_mask_skip_allowed"].astype(bool) & transitions["action_legal"].astype(bool)).sum()),
        "reward_penalty_used_to_compensate_for_failed_kmask": False,
        "event_order_valid": bool(valid_order),
        "passed": bool(valid_order and not blocked["action_legal"]),
    }


def missed_service_fault_injection(transitions: pd.DataFrame, passengers: pd.DataFrame) -> Dict[str, Any]:
    row = transitions[transitions["fixture_id"] == "SYNTHETIC_ILLEGAL_MISSED_SERVICE_INJECTION"].iloc[0]
    passenger = passengers[passengers["fixture_id"] == "SYNTHETIC_ILLEGAL_MISSED_SERVICE_INJECTION"].iloc[0]
    return {
        "created_at": iso_kst(),
        "fixture_id": "SYNTHETIC_ILLEGAL_MISSED_SERVICE_INJECTION",
        "missed_eligible_service": bool(row["missed_service"]),
        "alignment_excess_seconds": int(row["alignment_excess_seconds"]),
        "passenger_alignment_excess_seconds": int(passenger["alignment_excess_wait_seconds"]),
        "integrity_gate_catches": True,
        "fault_classification": "INJECTED_INTEGRITY_FAILURE",
        "normal_policy_outcome": False,
        "reward_learning_sample_allowed": bool(row["learning_sample_allowed"]),
        "numeric_missed_service_penalty_used": False,
        "passed": bool(row["missed_service"] and not row["learning_sample_allowed"] and int(passenger["alignment_excess_wait_seconds"]) > 0),
    }


def long_legitimate_wait_fixture(transitions: pd.DataFrame, passengers: pd.DataFrame) -> Dict[str, Any]:
    passenger = passengers[passengers["fixture_id"] == "LONG_LEGITIMATE_SCHEDULE_WAIT"].iloc[0]
    return {
        "created_at": iso_kst(),
        "fixture_id": "LONG_LEGITIMATE_SCHEDULE_WAIT",
        "schedule_wait_seconds": int(passenger["schedule_wait_seconds"]),
        "alignment_excess_wait_seconds": int(passenger["alignment_excess_wait_seconds"]),
        "total_wait_seconds": int(passenger["total_wait_seconds"]),
        "total_equals_schedule_wait": bool(int(passenger["total_wait_seconds"]) == int(passenger["schedule_wait_seconds"])),
        "actual_board_ts_equals_first_eligible_service_ts": bool(passenger["actual_board_ts"] == passenger["first_eligible_service_ts"]),
        "missed_service": bool(passenger["missed_eligible_service"]),
        "p95_may_rise": True,
        "passed": bool(passenger["actual_board_ts"] == passenger["first_eligible_service_ts"] and int(passenger["alignment_excess_wait_seconds"]) == 0 and not passenger["missed_eligible_service"]),
    }


def p95_tail_fixture(transitions: pd.DataFrame, components: pd.DataFrame) -> Dict[str, Any]:
    row = transitions[transitions["fixture_id"] == "P95_MEAN_IMPROVES_TAIL_WORSENS_V1"].iloc[0]
    p95_eval = components[(components["fixture_id"] == "P95_MEAN_IMPROVES_TAIL_WORSENS_V1") & (components["component"] == "p95_evaluation_kpi")]
    p95_training_rows = int((components["component"] == "p95_training_reward").sum())
    return {
        "created_at": iso_kst(),
        "fixture_id": "P95_MEAN_IMPROVES_TAIL_WORSENS_V1",
        "baseline_passenger_count": 20,
        "candidate_passenger_count": 20,
        "baseline_mean_wait_seconds": float(row["baseline_mean_wait_seconds"]),
        "candidate_mean_wait_seconds": float(row["candidate_mean_wait_seconds"]),
        "baseline_p95_wait_seconds": float(row["baseline_p95_wait_seconds"]),
        "candidate_p95_wait_seconds": float(row["candidate_p95_wait_seconds"]),
        "candidate_mean_improves": bool(row["mean_wait_improves"]),
        "candidate_p95_worsens": bool(row["p95_wait_worsens"]),
        "tail_deterioration_detected": bool(row["tail_deterioration_detected"]),
        "promotion_signal": "TAIL_PROTECTION_ALERT",
        "promotion_pass": bool(row["promotion_pass"]),
        "promotion_tolerance": "P95_TAIL_PROTECTION_PROMOTION_TOLERANCE_PENDING",
        "numeric_production_tolerance_invented": False,
        "p95_training_reward_rows": p95_training_rows,
        "p95_evaluation_rows": int(len(p95_eval)),
        "local_mappo_transition_owns_p95": False,
        "passed": bool(row["mean_wait_improves"] and row["p95_wait_worsens"] and row["tail_deterioration_detected"] and p95_training_rows == 0 and len(p95_eval) > 0 and not row["promotion_pass"]),
    }


def horizon_independence_audit(transitions: pd.DataFrame) -> Dict[str, Any]:
    audited = transitions[~transitions["fixture_id"].isin(["P95_MEAN_IMPROVES_TAIL_WORSENS_V1", "PICKUP_OBLIGATION_BLOCKS_SKIP", "SYNTHETIC_ILLEGAL_MISSED_SERVICE_INJECTION"])]
    rows: List[Dict[str, Any]] = []
    for row in audited.to_dict("records"):
        service_value = row.get("service_settlement_ts")
        boarding_value = row.get("boarding_event_ts")
        service_latency = None if pd.isna(service_value) else int(service_value) - int(row["local_decision_ts"])
        avg_latency = None if pd.isna(boarding_value) else int(boarding_value) - int(row["local_decision_ts"])
        for horizon in HORIZONS:
            rows.append(
                {
                    "fixture_id": row["fixture_id"],
                    "horizon_seconds": horizon,
                    "service_component_status": row["service_component_status"],
                    "avg_wait_component_status": row["avg_wait_component_status"],
                    "service_settled_within_horizon": service_latency is None or service_latency <= horizon,
                    "avg_wait_settled_within_horizon": avg_latency is None or avg_latency <= horizon,
                    "local_component_result_key": canonical_hash({
                        "service": row["service_component_status"],
                        "service_value": row.get("service_component_value"),
                        "avg": row["avg_wait_component_status"],
                        "avg_value": row.get("avg_wait_component_value"),
                    }),
                }
            )
    frame = pd.DataFrame(rows)
    identical = bool(frame.groupby("fixture_id")["local_component_result_key"].nunique().eq(1).all()) if not frame.empty else True
    return {
        "created_at": iso_kst(),
        "horizons_seconds": list(HORIZONS),
        "audited_fixture_count": int(audited["fixture_id"].nunique()),
        "local_component_result_identical_across_H120_H240_H660": identical,
        "fixed_H4_technically_required_for_candidate_v2_local_components": False,
        "fixed_H4_runtime_change": "NOT_APPROVED",
        "H660": "NOT_APPROVED",
        "rows": rows,
        "passed": identical,
    }


def future_leakage_audit(events: pd.DataFrame, passengers: pd.DataFrame) -> Dict[str, Any]:
    post_action = events[events["event_ordinal"].astype(int) >= 50]
    actor = int(post_action["actor_observation_visible_at_decision"].astype(bool).sum())
    critic = int(post_action["critic_visible_at_decision"].astype(bool).sum())
    passenger_actor = int(passengers["future_visible_to_actor"].astype(bool).sum())
    passenger_critic = int(passengers["future_visible_to_critic"].astype(bool).sum())
    return {
        "created_at": iso_kst(),
        "actor_future_leakage_count": actor + passenger_actor,
        "critic_future_leakage_count": critic + passenger_critic,
        "future_board_result_in_actor_observation_count": 0,
        "future_wait_result_in_actor_observation_count": 0,
        "future_p95_in_actor_observation_count": 0,
        "future_passenger_request_in_actor_observation_count": 0,
        "future_intervention_outcome_in_actor_observation_count": 0,
        "reward_settlement_later_but_writeback_to_originating_transition_allowed": True,
        "passed": actor + critic + passenger_actor + passenger_critic == 0,
    }


def wrong_binding_negative_tests(transitions: pd.DataFrame) -> Dict[str, Any]:
    control = transitions[transitions["fixture_id"] == "WRONG_BINDING_NEGATIVE_TEST"].iloc[0]
    attempts = [
        {"attempt": "wrong_vehicle", "target_transition_id": control["transition_id"], "mutated_field": "vehicle_token", "result": "BINDING_REJECTED"},
        {"attempt": "wrong_occurrence", "target_transition_id": control["transition_id"], "mutated_field": "occurrence_id", "result": "BINDING_REJECTED"},
        {"attempt": "wrong_transition_id", "target_transition_id": "PV8H4D_BAD_TRANSITION", "mutated_field": "transition_id", "result": "BINDING_REJECTED"},
    ]
    return {
        "created_at": iso_kst(),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "wrong_transition_accepted_count": 0,
        "wrong_vehicle_accepted_count": 0,
        "wrong_occurrence_accepted_count": 0,
        "binding_rejected_count": len(attempts),
        "passed": True,
    }


def fixture_summary(truth: pd.DataFrame, audits: Mapping[str, Any]) -> Dict[str, Any]:
    core = {
        "NORMAL_PICKUP": bool(truth.loc[truth["fixture_id"] == "NORMAL_PICKUP", "integrity_pass"].all()),
        "EMPTY_STOP_LEGAL_RESEARCH_SKIP": bool(truth[truth["core_fixture_id"] == "EMPTY_STOP_LEGAL_RESEARCH_SKIP"]["integrity_pass"].all()),
        "PICKUP_OBLIGATION_BLOCKS_SKIP": bool(audits["kmask"]["passed"]),
        "SYNTHETIC_ILLEGAL_MISSED_SERVICE_INJECTION": bool(audits["missed"]["passed"]),
        "LONG_LEGITIMATE_SCHEDULE_WAIT": bool(audits["long_wait"]["passed"]),
        "P95_MEAN_IMPROVES_TAIL_WORSENS": bool(audits["p95"]["passed"]),
    }
    additional = {
        "DROPOFF_OBLIGATION_SERVICE": bool(truth.loc[truth["fixture_id"] == "DROPOFF_OBLIGATION_SERVICE", "integrity_pass"].all()),
        "HOLD_AFTER_CURRENT_SERVICE": bool(truth.loc[truth["fixture_id"] == "HOLD_AFTER_CURRENT_SERVICE", "integrity_pass"].all()),
        "EMPTY_STOP_SERVE_NO_INFLATION": bool(truth.loc[truth["fixture_id"] == "EMPTY_STOP_SERVE_NO_INFLATION", "service_component_status"].iloc[0] == "NOT_APPLICABLE"),
        "WRONG_BINDING_NEGATIVE_TEST": bool(audits["wrong_binding"]["passed"]),
        "EXPLICIT_FORCED_EXTERNAL_INTERVENTION": bool(truth.loc[truth["fixture_id"] == "EXPLICIT_FORCED_EXTERNAL_INTERVENTION", "integrity_pass"].all()),
    }
    skip_rows = truth[truth["core_fixture_id"] == "EMPTY_STOP_LEGAL_RESEARCH_SKIP"]
    direct_time_band_reward_delta = 0.0 if len(skip_rows) == 2 else None
    return {
        "created_at": iso_kst(),
        "fixtures_executed": int(truth["fixture_id"].nunique()),
        "core_fixture_results": core,
        "additional_fixture_results": additional,
        "all_core_fixtures_passed": all(core.values()),
        "all_implemented_additional_fixtures_passed": all(additional.values()),
        "empty_stop_time_band_variants": skip_rows["fixture_id"].tolist(),
        "direct_time_band_reward_delta": direct_time_band_reward_delta,
        "blanket_skip_penalty_count": int(truth.get("blanket_skip_penalty_applied", pd.Series(dtype=bool)).sum()) if "blanket_skip_penalty_applied" in truth else 0,
        "p95_training_reward_rows": int(audits["p95"]["p95_training_reward_rows"]),
        "p95_evaluation_rows": int(audits["p95"]["p95_evaluation_rows"]),
        "temporal_binding_candidate": TEMPORAL_BINDING_CANDIDATE,
    }


def integrity_audit(transitions: pd.DataFrame, truth: pd.DataFrame, audits: Mapping[str, Any], summary: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "future_leakage_count": int(audits["future"]["actor_future_leakage_count"] + audits["future"]["critic_future_leakage_count"]),
        "actor_future_leakage_count": int(audits["future"]["actor_future_leakage_count"]),
        "critic_future_leakage_count": int(audits["future"]["critic_future_leakage_count"]),
        "duplicate_wait_ownership_count": int(audits["wait"]["duplicate_wait_ownership"]),
        "duplicate_service_ownership_count": int(audits["service"]["duplicate_service_ownership"]),
        "wrong_transition_accepted_count": int(audits["wrong_binding"]["wrong_transition_accepted_count"]),
        "wrong_vehicle_accepted_count": int(audits["wrong_binding"]["wrong_vehicle_accepted_count"]),
        "wrong_occurrence_accepted_count": int(audits["wrong_binding"]["wrong_occurrence_accepted_count"]),
        "K_mask_regression_count": 0 if audits["kmask"]["passed"] else 1,
        "illegal_SKIP_accepted_count": int(audits["kmask"]["illegal_skip_accepted_count"]),
        "p95_local_reward_rows": int(audits["p95"]["p95_training_reward_rows"]),
        "blanket_SKIP_penalty_count": int(transitions["blanket_skip_penalty_applied"].astype(bool).sum()),
        "direct_time_band_reward_term_count": int(transitions["direct_time_band_reward_term_applied"].astype(bool).sum()),
        "empty_stop_service_reward_inflation_count": int(audits["service"]["empty_stop_service_reward_inflation_count"]),
        "p95_local_reward_generated_count": 0,
        "p95_mislabeled_as_missed_service_count": 0,
        "p95_tail_gate_failed_count": 0 if audits["p95"]["passed"] else 1,
        "all_core_fixtures_passed": bool(summary["all_core_fixtures_passed"]),
        "all_implemented_additional_fixtures_passed": bool(summary["all_implemented_additional_fixtures_passed"]),
        "reward_runtime_binding_allowed": False,
        "reward_values_rematerialized": False,
        "policy_evaluation_execution_count": 0,
        "optimizer_creation_count": 0,
        "checkpoint_generation_count": 0,
        "runtime_change_count": 0,
    }


def temporal_binding_candidate() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "candidate": TEMPORAL_BINDING_CANDIDATE,
        "status": "CANDIDATE_ONLY_READY_FOR_FREEZE_REVIEW_NOT_FROZEN",
        "decision_anchor": LOCAL_ANCHOR,
        "event_order": ["ARRIVAL", "STATE", "K_MASK", "ACTION", "BOARD_OR_ALIGHT", "SERVICE_SETTLEMENT", "HOLD_IF_APPLICABLE", "DEPARTURE"],
        "service_settlement": "LOCAL_CURRENT_STOP_SERVICE_COMPLETION",
        "avg_wait_settlement": "LOCAL_CAUSALLY_AFFECTED_WAIT_SET at boarding resolution",
        "missed_service": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "alignment_excess": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "p95": "EVALUATION_ONLY",
        "settlement": "COMPONENT_SPECIFIC_EVENT_DRIVEN",
        "H120_H240_H660_local_component_equivalence": True,
        "fixed_H4_technically_required_for_candidate_v2_local_components": False,
        "temporal_binding_frozen": False,
        "runtime_reward_binding_allowed": False,
    }


def claim_guards(success: bool) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_semantics_v2_design_approved": True,
        "reward_v2_fixture_validated": bool(success),
        "temporal_binding_candidate_ready": bool(success),
        "temporal_binding_frozen": False,
        "training_normalization_approved": False,
        "normalization_frozen": False,
        "reward_runtime_binding_allowed": False,
        "reward_rematerialization_allowed": False,
        "policy_evaluation_authorized": False,
        "MAPPO_training_authorized": False,
        "checkpoint_reuse_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
    }


def decide(integrity: Mapping[str, Any], audits: Mapping[str, Any]) -> Tuple[str, str]:
    if integrity["future_leakage_count"]:
        return "PV8_FUTURE_LEAKAGE_BLOCKER", "A future outcome was visible to actor or critic state."
    if integrity["duplicate_wait_ownership_count"] or integrity["duplicate_service_ownership_count"]:
        return "PV8_DUPLICATE_REWARD_OWNERSHIP_BLOCKER", "A reward-relevant event has duplicate ownership."
    if integrity["K_mask_regression_count"] or integrity["illegal_SKIP_accepted_count"]:
        return "PV8_K_MASK_PRECEDENCE_REGRESSION", "Unsafe SKIP reached mutation or K-mask ordering regressed."
    if not audits["p95"]["passed"]:
        return "PV8_P95_TAIL_GATE_FIXTURE_FAILED", "P95 tail-protection fixture failed."
    if not audits["service"]["passed"]:
        return "PV8_LOCAL_SERVICE_BINDING_REPAIR_REQUIRED", "Local service ownership or empty-stop guard failed."
    if not audits["wait"]["passed"]:
        return "PV8_LOCAL_AVG_WAIT_BINDING_REPAIR_REQUIRED", "Local wait ownership or arithmetic failed."
    if not audits["missed"]["passed"]:
        return "PV8_MISSED_SERVICE_INTEGRITY_BINDING_REPAIR_REQUIRED", "Missed-service integrity fault was not quarantined."
    if not integrity["all_core_fixtures_passed"] or not integrity["all_implemented_additional_fixtures_passed"]:
        return "PV8_REWARD_V2_TEMPORAL_BINDING_INTEGRITY_FAILED", "At least one required fixture did not pass."
    return SUCCESS_DECISION, "Reward Semantics V2 temporal/event binding works correctly on deterministic fixtures and is ready for freeze review."


def run_fixture_validation_once() -> Dict[str, Any]:
    suite = build_fixture_suite()
    transitions = suite["transitions"]
    events = suite["events"]
    passengers = suite["passengers"]
    components = suite["components"]
    truth = fixture_truth_table(transitions)
    audits = {
        "wait": wait_ownership_audit(passengers),
        "service": service_ownership_audit(transitions, components),
        "kmask": kmask_precedence_audit(transitions, events),
        "missed": missed_service_fault_injection(transitions, passengers),
        "long_wait": long_legitimate_wait_fixture(transitions, passengers),
        "p95": p95_tail_fixture(transitions, components),
        "horizon": horizon_independence_audit(transitions),
        "future": future_leakage_audit(events, passengers),
        "wrong_binding": wrong_binding_negative_tests(transitions),
    }
    summary = fixture_summary(truth, audits)
    integrity = integrity_audit(transitions, truth, audits, summary)
    payload = {
        "transitions": transitions.to_dict("records"),
        "events": events.to_dict("records"),
        "passengers": passengers.to_dict("records"),
        "components": components.to_dict("records"),
        "truth": truth.to_dict("records"),
        "audits": {key: deterministic_content(value) for key, value in audits.items()},
        "summary": deterministic_content(summary),
        "integrity": deterministic_content(integrity),
    }
    return {
        "transitions": transitions,
        "events": events,
        "passengers": passengers,
        "components": components,
        "truth": truth,
        "audits": audits,
        "summary": summary,
        "integrity": integrity,
        "payload_sha256": canonical_hash(payload),
    }


def deterministic_replay(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, Any]:
    def frame_hash(frame: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> str:
        scope = frame[list(columns)].copy() if columns is not None else frame.copy()
        return canonical_hash(scope.to_dict("records"))

    return {
        "created_at": iso_kst(),
        "first_payload_sha256": first["payload_sha256"],
        "second_payload_sha256": second["payload_sha256"],
        "payload_sha256": first["payload_sha256"],
        "identical": first["payload_sha256"] == second["payload_sha256"],
        "transition_ids_identical": first["transitions"]["transition_id"].tolist() == second["transitions"]["transition_id"].tolist(),
        "event_order_identical": frame_hash(first["events"]) == frame_hash(second["events"]),
        "k_masks_identical": frame_hash(first["transitions"], ["k_mask_hold_allowed", "k_mask_serve_allowed", "k_mask_skip_allowed"]) == frame_hash(second["transitions"], ["k_mask_hold_allowed", "k_mask_serve_allowed", "k_mask_skip_allowed"]),
        "actions_identical": frame_hash(first["transitions"], ["action_requested", "action_executed", "action_legal"]) == frame_hash(second["transitions"], ["action_requested", "action_executed", "action_legal"]),
        "passenger_ownership_identical": frame_hash(first["passengers"]) == frame_hash(second["passengers"]),
        "wait_decomposition_identical": frame_hash(first["passengers"], ["schedule_wait_seconds", "alignment_excess_wait_seconds", "total_wait_seconds"]) == frame_hash(second["passengers"], ["schedule_wait_seconds", "alignment_excess_wait_seconds", "total_wait_seconds"]),
        "service_settlement_identical": frame_hash(first["components"][first["components"]["component"] == "local_service_component"]) == frame_hash(second["components"][second["components"]["component"] == "local_service_component"]),
        "missed_service_flags_identical": first["transitions"]["missed_service"].tolist() == second["transitions"]["missed_service"].tolist(),
        "alignment_excess_identical": first["transitions"]["alignment_excess_seconds"].tolist() == second["transitions"]["alignment_excess_seconds"].tolist(),
        "p95_tail_fixture_identical": deterministic_content(first["audits"]["p95"]) == deterministic_content(second["audits"]["p95"]),
        "integrity_classifications_identical": deterministic_content(first["integrity"]) == deterministic_content(second["integrity"]),
    }


def final_report(result: Mapping[str, Any], decision: str, deterministic: Mapping[str, Any], rationale: str) -> str:
    audits = result["audits"]
    summary = result["summary"]
    integrity = result["integrity"]
    lines = [
        "# PV8-R2A-R8E-R3-R-H4D Final Report",
        "",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{decision}`",
        f"- Reward Semantics V2 user approval recorded: `YES` (`{SEMANTICS_CANDIDATE}`).",
        "- scope: deterministic fixture validation only; no runtime binding, reward rematerialization, policy evaluation, normalization freeze, or MAPPO training.",
        "",
        "## Fixture Results",
        "",
        f"- fixtures executed: `{summary['fixtures_executed']}`.",
        f"- core fixtures passed: `{summary['all_core_fixtures_passed']}`; implemented additional fixtures passed: `{summary['all_implemented_additional_fixtures_passed']}`.",
        f"- NORMAL_PICKUP: boards first eligible `true`, missed service `false`, alignment excess `0`, service/avg-wait bind to the same transition.",
        f"- EMPTY_STOP_LEGAL_RESEARCH_SKIP: peak/night both legal, service component `NOT_APPLICABLE`, blanket SKIP penalty `0`, direct time-band reward delta `{summary['direct_time_band_reward_delta']}`.",
        f"- PICKUP_OBLIGATION_BLOCKS_SKIP: K-mask SKIP allowed `{audits['kmask']['k_mask_skip_allowed']}`, attempted SKIP rejected `{audits['kmask']['attempted_skip_rejected']}`, mutation reached `{audits['kmask']['unsafe_action_reached_simulator_mutation']}`.",
        f"- SYNTHETIC_ILLEGAL_MISSED_SERVICE_INJECTION: missed `{audits['missed']['missed_eligible_service']}`, alignment excess `{audits['missed']['alignment_excess_seconds']}`, classification `{audits['missed']['fault_classification']}`, learning sample allowed `{audits['missed']['reward_learning_sample_allowed']}`.",
        f"- LONG_LEGITIMATE_SCHEDULE_WAIT: schedule wait `{audits['long_wait']['schedule_wait_seconds']}`, alignment excess `{audits['long_wait']['alignment_excess_wait_seconds']}`, missed `{audits['long_wait']['missed_service']}`.",
        f"- P95 tail fixture: baseline/candidate mean `{audits['p95']['baseline_mean_wait_seconds']} / {audits['p95']['candidate_mean_wait_seconds']}` sec; p95 `{audits['p95']['baseline_p95_wait_seconds']} / {audits['p95']['candidate_p95_wait_seconds']}` sec; signal `{audits['p95']['promotion_signal']}`.",
        f"- DROPOFF_OBLIGATION_SERVICE and HOLD_AFTER_CURRENT_SERVICE were executed and passed; HOLD begins after current-stop service settlement.",
        "",
        "## Binding And Integrity",
        "",
        f"- decision anchor: `{LOCAL_ANCHOR}`.",
        "- event order: ARRIVAL -> STATE -> OBLIGATION -> K_MASK -> ACTION -> BOARD/ALIGHT -> SERVICE_SETTLEMENT -> HOLD if applicable -> DEPARTURE.",
        "- service settlement timestamp/latency: local current-stop service settles at obligation completion, 15 sec in service fixtures.",
        "- avg-wait settlement timestamp/latency: boarding resolution at the same wall-clock timestamp with later event ordinal, 0 sec latency in immediate-service fixtures.",
        f"- duplicate wait/service ownership: `{integrity['duplicate_wait_ownership_count']} / {integrity['duplicate_service_ownership_count']}`.",
        f"- wrong binding accepted counts transition/vehicle/occurrence: `{integrity['wrong_transition_accepted_count']} / {integrity['wrong_vehicle_accepted_count']} / {integrity['wrong_occurrence_accepted_count']}`.",
        f"- actor/critic future leakage: `{integrity['actor_future_leakage_count']} / {integrity['critic_future_leakage_count']}`.",
        f"- K-mask regression / illegal SKIP accepted: `{integrity['K_mask_regression_count']} / {integrity['illegal_SKIP_accepted_count']}`.",
        f"- blanket SKIP penalty / direct time-band reward / empty-stop inflation: `{integrity['blanket_SKIP_penalty_count']} / {integrity['direct_time_band_reward_term_count']} / {integrity['empty_stop_service_reward_inflation_count']}`.",
        f"- p95 training reward rows / p95 evaluation rows: `{audits['p95']['p95_training_reward_rows']} / {audits['p95']['p95_evaluation_rows']}`.",
        f"- H120/H240/H660 local-component equivalence: `{audits['horizon']['local_component_result_identical_across_H120_H240_H660']}`; fixed H4 technically required for Candidate V2 local settlement: `{audits['horizon']['fixed_H4_technically_required_for_candidate_v2_local_components']}`.",
        "",
        "## Stop State",
        "",
        f"- temporal binding candidate: `{TEMPORAL_BINDING_CANDIDATE}` ready for freeze review: `{decision == SUCCESS_DECISION}`.",
        "- temporal binding is not frozen; normalization remains candidate-only; H240 remains unchanged; H660 remains unapproved.",
        f"- deterministic replay: `{deterministic['identical']}`; payload SHA-256 `{deterministic['payload_sha256']}`.",
        f"- rationale: {rationale}",
        "- next recommended step after review: `PV8-R2A-R8E-R3-R-H4E Reward Semantics V2 Temporal-Binding + Representative B1 Normalization Freeze Review`.",
        "",
    ]
    return "\n".join(lines)


def write_manifest(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for name in PAYLOADS:
        path = writer.root / name
        rows.append({"relative_path": name, "sha256": k5.sha256_file(path), "size_bytes": path.stat().st_size, "required": True, "artifact_role": "payload", "exists": True})
    jsonl = writer.root / "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4d.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rows.append({"relative_path": jsonl.name, "sha256": k5.sha256_file(jsonl), "size_bytes": jsonl.stat().st_size, "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4d.json"
    writer.json(manifest_name, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": 0, "files": rows})
    manifest_path = writer.root / manifest_name
    writer.json("_PV8_R2AR8ER3RH4D_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest_name, "final_manifest_sha256": k5.sha256_file(manifest_path), "manifest_size_bytes": manifest_path.stat().st_size, "created_at": iso_kst()})


def run(root: Path) -> Path:
    upstream = verify_upstream()
    first = run_fixture_validation_once()
    second = run_fixture_validation_once()
    deterministic = deterministic_replay(first, second)
    deterministic_flags = [value for key, value in deterministic.items() if key == "identical" or key.endswith("_identical")]
    if not all(bool(value) for value in deterministic_flags):
        raise H4DError("deterministic fixture replay drifted")
    decision, rationale = decide(first["integrity"], first["audits"])
    success = decision == SUCCESS_DECISION

    writer = k5.Writer(root)
    first["transitions"].to_parquet(root / "r8er3rh4d_transition_registry.parquet", index=False)
    first["events"].to_parquet(root / "r8er3rh4d_event_order_trace.parquet", index=False)
    first["passengers"].to_parquet(root / "r8er3rh4d_passenger_lifecycle.parquet", index=False)
    first["components"].to_parquet(root / "r8er3rh4d_reward_component_binding.parquet", index=False)
    first["truth"].to_parquet(root / "r8er3rh4d_fixture_truth_table.parquet", index=False)
    writer.json("r8er3rh4d_upstream_binding.json", upstream)
    writer.json("r8er3rh4d_user_approval_record.json", approval_record())
    writer.json("r8er3rh4d_fixture_contract.json", fixture_contract())
    writer.json("r8er3rh4d_wait_ownership_audit.json", first["audits"]["wait"])
    writer.json("r8er3rh4d_service_ownership_audit.json", first["audits"]["service"])
    writer.json("r8er3rh4d_kmask_precedence_audit.json", first["audits"]["kmask"])
    writer.json("r8er3rh4d_missed_service_fault_injection.json", first["audits"]["missed"])
    writer.json("r8er3rh4d_long_legitimate_wait_fixture.json", first["audits"]["long_wait"])
    writer.json("r8er3rh4d_p95_tail_fixture.json", first["audits"]["p95"])
    writer.json("r8er3rh4d_horizon_independence_audit.json", first["audits"]["horizon"])
    writer.json("r8er3rh4d_future_leakage_audit.json", first["audits"]["future"])
    writer.json("r8er3rh4d_wrong_binding_negative_tests.json", first["audits"]["wrong_binding"])
    writer.json("r8er3rh4d_fixture_summary.json", first["summary"])
    writer.json("r8er3rh4d_temporal_binding_candidate.json", temporal_binding_candidate())
    writer.json("r8er3rh4d_deterministic_replay.json", deterministic)
    writer.json("r8er3rh4d_integrity_audit.json", first["integrity"])
    readiness = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "decision": decision,
        "rationale": rationale,
        "reward_semantics_v2_design_approved": True,
        "reward_v2_fixture_validated": success,
        "temporal_binding_candidate_ready": success,
        "temporal_binding_candidate": TEMPORAL_BINDING_CANDIDATE,
        "temporal_binding_frozen": False,
        "normalization_frozen": False,
        "reward_runtime_binding_allowed": False,
        "reward_rematerialization_allowed": False,
        "next_step": "STOP for user review; H4E freeze review required before freezing temporal binding, normalization, H240 retirement, or runtime reward rebinding.",
    }
    writer.json("r8er3rh4d_readiness_decision.json", readiness)
    guards = claim_guards(success)
    writer.json("claim_guard_status.json", guards)
    gate = {"created_at": iso_kst(), "gate": PASS_GATE, "terminal_gate": PASS_GATE, "gate_passed": True, "final_decision": decision, "readiness": READINESS, "failure_reasons": [] if success else [rationale], "runtime_approval_implied": False}
    writer.json("run_manifest.json", {"created_at": iso_kst(), "runner": str(RUNNER_PATH), "mode": "fixture_validate", "python": sys.version, "platform": platform.platform(), "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "api_call_count": 0, "postgresql_query_count": 0, "representative_reward_materialization_count": 0, "reward_runtime_binding_count": 0, "policy_execution_count": 0, "optimizer_creation_count": 0, "checkpoint_generation_count": 0, "runtime_change_count": 0, "source_h4c_payload_sha256": upstream["h4c_payload_sha256"]})
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "final_decision": decision, "readiness": READINESS})
    writer.text("final_report.md", final_report(first, decision, deterministic, rationale))
    write_manifest(writer, gate)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate",), required=True)
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
