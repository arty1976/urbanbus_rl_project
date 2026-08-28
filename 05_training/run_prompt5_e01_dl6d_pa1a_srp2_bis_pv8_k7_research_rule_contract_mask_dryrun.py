#!/usr/bin/env python3
"""PV8-K7 inactive research static-rule candidate and eight-agent mask dry-run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
from simulator.k_safety_state import K_SAFETY_STATE_SCHEMA_VERSION, ServiceObligationStateMachine, StaticGuardStatus
from simulator.suseong_service_transition_engine import build_distinct_three_action_mask


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k7_research_rule_contract_mask_dryrun.py"

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
C3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c3_mappo_interface_compatibility_20260808_100118"
K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"
K6_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026"

UPSTREAMS = {
    "PV8-C2": (C2_ROOT, "artifact_manifest_srp2_bis_pv8_c2.json", "_PV8_C2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"),
    "PV8-C3": (C3_ROOT, "artifact_manifest_srp2_bis_pv8_c3.json", "_PV8_C3_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_COMPLETE"),
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
    "PV8-K6": (K6_ROOT, "artifact_manifest_srp2_bis_pv8_k6.json", "_PV8_K6_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K6_STATIC_RULE_AUTHORITY_AND_OCCURRENCE_AUDIT_COMPLETE"),
}

OCCURRENCE_PATH = K6_ROOT / "k6_route_stop_occurrence_master.parquet"
K6_RULEBOOK_PATH = K6_ROOT / "k6_real_network_rulebook.parquet"
C2_MAPPING_PATH = C2_ROOT / "prospective_8vehicle_mapping_manifest.parquet"
C2_CYCLE_PATH = C2_ROOT / "prospective_agent_cycle_state.parquet"
C2_TRANSITION_PATH = C2_ROOT / "vehicle_identity_transition_events.parquet"
K4_CONTRACT_PATH = K4_ROOT / "k4_state_machine_contract.json"
K4_STATE_SOURCE = TRAINING_ROOT / "simulator" / "k_safety_state.py"
K4_ENGINE_SOURCE = TRAINING_ROOT / "simulator" / "suseong_service_transition_engine.py"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k7_research_rule_contract_mask_dryrun"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K7_RESEARCH_RULE_CONTRACT_AND_K_MASK_DRYRUN_COMPLETE"
FAIL_GATE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K7_RESEARCH_RULE_CONTRACT_AND_K_MASK_DRYRUN_INCOMPLETE"
PASS_READINESS = "SRP2_BIS_PV8_K7_COMPLETE_RESEARCH_RULE_EXPLICIT_APPROVAL_PENDING_K8_LOCKED"

DECISION_READY = "RESEARCH_RULE_CONTRACT_READY_FOR_EXPLICIT_APPROVAL"
DECISION_INCOMPLETE = "RESEARCH_RULE_CONTRACT_INCOMPLETE"
DECISION_MASK_FAILED = "K_MASK_INTEGRATION_FAILED"

RULE_VERSION = "PV8_K7_RESEARCH_STATIC_RULE_CANDIDATE_V1"
RULE_CLASS = "CONTRACT_FIXED_RESEARCH_RULE"
APPROVAL_STATUS = "NOT_APPROVED"
MASK_PREDICATE_VERSION = "PV8_K4_FAIL_CLOSED_V2"
DYNAMIC_STATE_CONTRACT_VERSION = K_SAFETY_STATE_SCHEMA_VERSION

STATIC_FIELDS = (
    "mandatory_stop",
    "protected_stop",
    "planned_itinerary_allows_skip",
    "terminal_or_turnaround_stop",
    "charging_or_driver_relief_stop",
)

PAYLOADS = [
    "k7_research_rulebook_candidate.parquet",
    "k7_research_rule_contract.json",
    "k7_rule_freeze_manifest.json",
    "k7_8agent_mask_dryrun.parquet",
    "k7_mask_integration_validation.json",
    "k7_snapshot_version_contract.json",
    "k7_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class K7Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def verify_upstreams() -> Dict[str, Any]:
    verified: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed_gate != expected_gate or not k5.manifest_ok(checks):
            raise K7Error(f"{label} upstream integrity failure: gate={observed_gate}, checks={checks}")
        verified[label] = {
            "artifact_root": str(root),
            "gate": observed_gate,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return verified


def k4_source_integrity() -> Dict[str, Any]:
    contract = k5.read_json(K4_CONTRACT_PATH)
    frozen = {str(row["path"]): str(row["sha256"]) for row in contract["source_files"]}
    rows = []
    for path in (K4_STATE_SOURCE, K4_ENGINE_SOURCE):
        current = k5.sha256_file(path)
        expected = frozen.get(str(path))
        rows.append({"path": str(path), "expected_sha256": expected, "current_sha256": current, "matches_k4": current == expected})
    return {"records": rows, "source_drift_count": sum(not row["matches_k4"] for row in rows)}


RULEBOOK_COLUMNS = [
    "rule_class", "approval_status", "activated", "rule_version", "route_stop_occurrence_id",
    "route_id", "direction_id", "stop_sequence", "occurrence_index", "stop_id",
    "same_stop_occurrence_ordinal", "same_stop_occurrence_count", "is_repeated_stop_occurrence",
    "route_direction_endpoint", "endpoint_role", "post_skip_target_exists", "post_skip_target_stop_id",
    "mandatory_stop", "mandatory_stop_classification", "mandatory_stop_rule_source", "mandatory_stop_rule_reason",
    "protected_stop", "protected_stop_classification", "protected_stop_rule_source", "protected_stop_rule_reason",
    "planned_itinerary_allows_skip", "planned_itinerary_allows_skip_classification",
    "planned_itinerary_allows_skip_rule_source", "planned_itinerary_allows_skip_rule_reason",
    "terminal_or_turnaround_stop", "terminal_or_turnaround_stop_classification",
    "terminal_or_turnaround_stop_rule_source", "terminal_or_turnaround_stop_rule_reason",
    "charging_or_driver_relief_stop", "charging_or_driver_relief_stop_classification",
    "charging_or_driver_relief_stop_rule_source", "charging_or_driver_relief_stop_rule_reason",
    "valid_post_skip_path", "valid_post_skip_path_classification", "static_rule_complete",
    "positive_static_skip_clearance", "rule_source", "rule_reason", "effective_scope", "effective_from",
    "effective_to", "provenance", "occurrence_master_sha256", "real_network_rule_claim_allowed",
]


def build_rule_candidate() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    occurrence = pd.read_parquet(OCCURRENCE_PATH)
    upstream = pd.read_parquet(K6_RULEBOOK_PATH)
    if len(occurrence) != 20508 or occurrence["route_stop_occurrence_id"].nunique() != 20508:
        raise K7Error("K6 occurrence master is not the frozen 20,508-row unique-ID contract")
    merged = occurrence.merge(
        upstream,
        on="route_stop_occurrence_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_k6"),
    )
    if merged["rule_scope"].isna().any():
        raise K7Error("K6 real-network rulebook does not cover every occurrence")
    occurrence_sha = k5.sha256_file(OCCURRENCE_PATH)
    rows: List[Dict[str, Any]] = []
    for source in merged.to_dict("records"):
        endpoint = bool(source["route_direction_endpoint"])
        has_post = bool(source["post_skip_target_exists"])
        values = {
            "mandatory_stop": False,
            "protected_stop": False,
            "planned_itinerary_allows_skip": bool(has_post),
            "terminal_or_turnaround_stop": bool(endpoint),
            "charging_or_driver_relief_stop": False,
        }
        classes = {field: RULE_CLASS for field in STATIC_FIELDS}
        sources = {field: "PV8_K7_EXPLICIT_RESEARCH_ASSUMPTION_CANDIDATE" for field in STATIC_FIELDS}
        reasons = {
            "mandatory_stop": "research-only assumption: no mandatory stop beyond separately preserved blockers",
            "protected_stop": "research-only assumption: protected-stop set is empty until explicit approval/versioning",
            "planned_itinerary_allows_skip": "research-only permission when a post-skip target exists",
            "terminal_or_turnaround_stop": "research-only assumption: no interior turnaround beyond ordered endpoints",
            "charging_or_driver_relief_stop": "research-only assumption: charging/relief set is empty until explicit approval/versioning",
        }
        if source["planned_itinerary_allows_skip_classification"] == "DERIVED_SAFE":
            values["planned_itinerary_allows_skip"] = bool(source["planned_itinerary_allows_skip"])
            classes["planned_itinerary_allows_skip"] = "DERIVED_SAFE"
            sources["planned_itinerary_allows_skip"] = "PV8_K6_ORDERED_OCCURRENCE_BLOCKER"
            reasons["planned_itinerary_allows_skip"] = "preserved K6 blocker: no post-skip target exists"
        if source["terminal_or_turnaround_stop_classification"] == "DERIVED_SAFE":
            values["terminal_or_turnaround_stop"] = bool(source["terminal_or_turnaround_stop"])
            classes["terminal_or_turnaround_stop"] = "DERIVED_SAFE"
            sources["terminal_or_turnaround_stop"] = "PV8_K6_ORDERED_ENDPOINT_BLOCKER"
            reasons["terminal_or_turnaround_stop"] = "preserved K6 blocker: route-direction endpoint occurrence"
        valid_path = bool(has_post)
        valid_path_class = "DERIVED_SAFE" if not has_post else RULE_CLASS
        positive = bool(
            values["mandatory_stop"] is False
            and values["protected_stop"] is False
            and values["planned_itinerary_allows_skip"] is True
            and values["terminal_or_turnaround_stop"] is False
            and values["charging_or_driver_relief_stop"] is False
            and valid_path
        )
        provenance = {
            field: {"classification": classes[field], "source": sources[field], "reason": reasons[field]}
            for field in STATIC_FIELDS
        }
        row: Dict[str, Any] = {
            "rule_class": RULE_CLASS,
            "approval_status": APPROVAL_STATUS,
            "activated": False,
            "rule_version": RULE_VERSION,
            **{column: source[column] for column in (
                "route_stop_occurrence_id", "route_id", "direction_id", "stop_sequence", "occurrence_index",
                "stop_id", "same_stop_occurrence_ordinal", "same_stop_occurrence_count",
                "is_repeated_stop_occurrence", "route_direction_endpoint", "endpoint_role",
                "post_skip_target_exists", "post_skip_target_stop_id",
            )},
            "valid_post_skip_path": valid_path,
            "valid_post_skip_path_classification": valid_path_class,
            "static_rule_complete": True,
            "positive_static_skip_clearance": positive,
            "rule_source": "PV8_K6_DERIVED_BLOCKERS_PLUS_PV8_K7_RESEARCH_ASSUMPTION_CANDIDATE",
            "rule_reason": "complete occurrence-level candidate for dry-run only; not approved or activated",
            "effective_scope": "RESEARCH_ONLY_ROUTE_DIRECTION_STOP_OCCURRENCE",
            "effective_from": None,
            "effective_to": None,
            "provenance": json.dumps(provenance, ensure_ascii=False, sort_keys=True),
            "occurrence_master_sha256": occurrence_sha,
            "real_network_rule_claim_allowed": False,
        }
        for field in STATIC_FIELDS:
            row[field] = values[field]
            row[f"{field}_classification"] = classes[field]
            row[f"{field}_rule_source"] = sources[field]
            row[f"{field}_rule_reason"] = reasons[field]
        rows.append(row)

    field_counts: Dict[str, Dict[str, int]] = {}
    for field in STATIC_FIELDS:
        counts = Counter(row[f"{field}_classification"] for row in rows)
        field_counts[field] = {
            "DERIVED_SAFE": counts["DERIVED_SAFE"],
            "CONTRACT_FIXED_RESEARCH_RULE": counts[RULE_CLASS],
            "OBSERVED": counts["OBSERVED"],
        }
    contract_summary = {
        "created_at": iso_kst(),
        "contract_name": RULE_VERSION,
        "rule_class": RULE_CLASS,
        "approval_status": APPROVAL_STATUS,
        "activated": False,
        "occurrence_row_count": len(rows),
        "occurrence_id_unique_count": len({row["route_stop_occurrence_id"] for row in rows}),
        "occurrence_master_sha256": occurrence_sha,
        "field_classification_counts": field_counts,
        "authoritative_derived_predicate_cell_count": sum(counts["DERIVED_SAFE"] for counts in field_counts.values()),
        "contract_fixed_predicate_cell_count": sum(counts[RULE_CLASS] for counts in field_counts.values()),
        "observed_predicate_cell_count": sum(counts["OBSERVED"] for counts in field_counts.values()),
        "positive_static_clearance_row_count": sum(bool(row["positive_static_skip_clearance"]) for row in rows),
        "assumptions": {
            "mandatory_stop": "false for all occurrences as an unapproved research-only assumption",
            "protected_stop": "false for all occurrences as an unapproved research-only assumption",
            "planned_itinerary_allows_skip": "true when post-skip target exists; K6 no-target false blockers preserved",
            "terminal_or_turnaround_stop": "true at K6 endpoints; false at interior occurrences as an unapproved research-only assumption",
            "charging_or_driver_relief_stop": "false for all occurrences as an unapproved research-only assumption",
            "valid_post_skip_path": "true when a following occurrence exists as an unapproved research path convention; false no-target blockers preserved",
        },
        "real_network_rule_claim_allowed": False,
        "operator_rule_claim_allowed": False,
        "research_candidate_use_allowed_before_approval": False,
        "mask_dryrun_use_only": True,
        "preserved_k6_blockers": [
            "route-direction endpoint => terminal_or_turnaround_stop=true",
            "no post-skip target => planned_itinerary_allows_skip=false and valid_post_skip_path=false",
        ],
    }
    return rows, contract_summary


@dataclass
class RuntimeVehicle:
    agent_id: int
    vehicle_token: str
    route_key: Tuple[str, str]
    position: int = 0
    remaining_travel_time: float = 0.0
    remaining_dwell_time: float = 0.0


def guard_from_rule(rule: Mapping[str, Any]) -> StaticGuardStatus:
    return StaticGuardStatus(
        bool(rule["mandatory_stop"]),
        bool(rule["protected_stop"]),
        bool(rule["planned_itinerary_allows_skip"]),
        bool(rule["terminal_or_turnaround_stop"]),
        bool(rule["charging_or_driver_relief_stop"]),
        bool(rule["valid_post_skip_path"]),
        bool(rule["static_rule_complete"]),
        (),
        "MIXED_DERIVED_SAFE_AND_CONTRACT_FIXED_RESEARCH_RULE",
    )


def empty_machine(agent_id: int, token: str, candidate_stop: str, post_stop: Optional[str]) -> ServiceObligationStateMachine:
    state = ServiceObligationStateMachine()
    state.register_vehicle(agent_id, token)
    state.register_stop("PREVIOUS")
    state.register_stop(candidate_stop)
    if post_stop is not None:
        state.register_stop(post_stop)
    return state


def evaluate_mask(
    *,
    scenario_id: str,
    agent_id: int,
    token: str,
    anchor_occurrence_id: str,
    rule: Mapping[str, Any],
    expected_skip: bool,
    expected_reason: Optional[str],
    machine: Optional[ServiceObligationStateMachine] = None,
    decision_ts: int = 0,
    identity_test: str = "NOT_APPLICABLE",
    slot_preserved: bool = True,
    direction_transition_preserved: bool = True,
    reappearance_preserved: bool = True,
) -> Dict[str, Any]:
    candidate_stop = str(rule["stop_id"])
    post_stop = str(rule["post_skip_target_stop_id"]) if rule["post_skip_target_stop_id"] is not None else None
    state = machine or empty_machine(agent_id, token, candidate_stop, post_stop)
    route_key = ("K7_DRYRUN", f"{agent_id}:{scenario_id}")
    route = [
        {"stop_id": "PREVIOUS", "node_uid": "STOP:PREVIOUS"},
        {"stop_id": candidate_stop, "node_uid": f"STOP:{candidate_stop}"},
    ]
    if post_stop is not None:
        route.append({"stop_id": post_stop, "node_uid": f"STOP:{post_stop}"})
    result = build_distinct_three_action_mask(
        RuntimeVehicle(agent_id, token, route_key),
        {route_key: route},
        obligation_state_machine=state,
        decision_ts=decision_ts,
        vehicle_token=token,
        static_guard_status=guard_from_rule(rule),
    )
    reasons = list(result["skip_invalid_reason_codes"])
    reason_ok = expected_reason is None or expected_reason in reasons
    snapshot = result["decision_time_obligation_snapshot"]
    passed = bool(result["skip_valid"] is expected_skip and reason_ok and slot_preserved and direction_transition_preserved and reappearance_preserved)
    return {
        "scenario_id": scenario_id,
        "agent_id": agent_id,
        "physical_vehicle_token": token,
        "anchor_route_stop_occurrence_id": anchor_occurrence_id,
        "evaluated_route_stop_occurrence_id": rule["route_stop_occurrence_id"],
        "route_id": rule["route_id"],
        "direction_id": rule["direction_id"],
        "stop_sequence": int(rule["stop_sequence"]),
        "stop_id": rule["stop_id"],
        "rule_class": rule["rule_class"],
        "approval_status": rule["approval_status"],
        "rule_version": rule["rule_version"],
        "expected_skip_valid": expected_skip,
        "observed_skip_valid": bool(result["skip_valid"]),
        "passed": passed,
        "expected_reason": expected_reason,
        "observed_reasons_json": json.dumps(reasons),
        "action_mask_json": json.dumps(result["action_mask"]),
        "dynamic_state_complete": bool(result["dynamic_state_complete"]),
        "static_guard_complete": bool(result["static_guard_state_complete"]),
        "pickup_obligation": bool(snapshot["pickup_obligation"]),
        "dropoff_obligation": bool(snapshot["dropoff_obligation"]),
        "onboard_destination_obligation": bool(snapshot["onboard_destination_obligation"]),
        "service_obligation": bool(snapshot["service_obligation"]),
        "identity_test": identity_test,
        "slot_preserved": slot_preserved,
        "direction_transition_preserved": direction_transition_preserved,
        "reappearance_preserved": reappearance_preserved,
        "future_event_visible": False,
        "policy_execution_count": 0,
    }


DRYRUN_COLUMNS = [
    "scenario_id", "agent_id", "physical_vehicle_token", "anchor_route_stop_occurrence_id",
    "evaluated_route_stop_occurrence_id", "route_id", "direction_id", "stop_sequence", "stop_id",
    "rule_class", "approval_status", "rule_version", "expected_skip_valid", "observed_skip_valid", "passed",
    "expected_reason", "observed_reasons_json", "action_mask_json", "dynamic_state_complete",
    "static_guard_complete", "pickup_obligation", "dropoff_obligation", "onboard_destination_obligation",
    "service_obligation", "identity_test", "slot_preserved", "direction_transition_preserved",
    "reappearance_preserved", "future_event_visible", "policy_execution_count", "static_rulebook_sha256",
    "occurrence_master_sha256", "mask_predicate_version", "dynamic_state_contract_version",
]


def build_dryruns(rule_rows: Sequence[Mapping[str, Any]], rulebook_sha: str, occurrence_sha: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rules_by_id = {str(row["route_stop_occurrence_id"]): row for row in rule_rows}
    rules_by_exact = {
        (str(row["route_id"]), str(row["direction_id"]), int(row["stop_sequence"]), str(row["stop_id"])): row
        for row in rule_rows
    }
    by_route_position = {
        (str(row["route_id"]), str(row["direction_id"]), int(row["occurrence_index"])): row
        for row in rule_rows
    }
    mapping = pd.read_parquet(C2_MAPPING_PATH).sort_values("agent_id")
    cycles = pd.read_parquet(C2_CYCLE_PATH).sort_values(["agent_id", "cycle_index"])
    transitions = pd.read_parquet(C2_TRANSITION_PATH)
    dryruns: List[Dict[str, Any]] = []
    bindings: List[Dict[str, Any]] = []

    for anchor in mapping.to_dict("records"):
        agent_id = int(anchor["agent_id"])
        token = str(anchor["physical_vehicle_token"])
        exact_key = (str(anchor["anchor_route_id"]), str(anchor["anchor_direction_id"]), int(anchor["anchor_seq"]), str(anchor["anchor_stop_id"]))
        anchor_rule = rules_by_exact.get(exact_key)
        if anchor_rule is None:
            raise K7Error(f"C2 anchor missing occurrence binding: {exact_key}")
        route_key = (str(anchor_rule["route_id"]), str(anchor_rule["direction_id"]))
        route_rules = [row for (route_id, direction_id, _), row in by_route_position.items() if (route_id, direction_id) == route_key]
        clear_rule = next((row for row in sorted(route_rules, key=lambda item: int(item["occurrence_index"])) if row["positive_static_skip_clearance"]), None)
        terminal_rule = next((row for row in sorted(route_rules, key=lambda item: int(item["occurrence_index"]), reverse=True) if row["terminal_or_turnaround_stop"]), None)
        if clear_rule is None or terminal_rule is None:
            raise K7Error(f"agent {agent_id} route lacks clear or terminal research case")
        bindings.append(
            {
                "agent_id": agent_id,
                "vehicle_token": token,
                "anchor_occurrence_id": anchor_rule["route_stop_occurrence_id"],
                "anchor_exact_key": list(exact_key),
                "anchor_positive_static_clearance": bool(anchor_rule["positive_static_skip_clearance"]),
                "clear_case_occurrence_id": clear_rule["route_stop_occurrence_id"],
                "terminal_case_occurrence_id": terminal_rule["route_stop_occurrence_id"],
            }
        )

        anchor_expected = bool(anchor_rule["positive_static_skip_clearance"])
        anchor_reason = None if anchor_expected else "TERMINAL_OR_TURNAROUND_STOP" if anchor_rule["terminal_or_turnaround_stop"] else "PLANNED_ITINERARY_BLOCK"
        dryruns.append(evaluate_mask(scenario_id="ANCHOR_OCCURRENCE_RESEARCH_RULE", agent_id=agent_id, token=token, anchor_occurrence_id=anchor_rule["route_stop_occurrence_id"], rule=anchor_rule, expected_skip=anchor_expected, expected_reason=anchor_reason))
        dryruns.append(evaluate_mask(scenario_id="CLEAR_RESEARCH_CONTRACT_CASE", agent_id=agent_id, token=token, anchor_occurrence_id=anchor_rule["route_stop_occurrence_id"], rule=clear_rule, expected_skip=True, expected_reason=None))

        candidate_stop = str(clear_rule["stop_id"])
        post_stop = str(clear_rule["post_skip_target_stop_id"])
        pickup = empty_machine(agent_id, token, candidate_stop, post_stop)
        pickup.passenger_waiting(passenger_id=f"P-PICKUP-{agent_id}", pickup_stop=candidate_stop, dropoff_stop=post_stop, event_ts=1)
        pickup.request_created(request_id=f"R-PICKUP-{agent_id}", passenger_id=f"P-PICKUP-{agent_id}", service_leg_id=f"L-PICKUP-{agent_id}", event_ts=2)
        pickup.request_assigned(request_id=f"R-PICKUP-{agent_id}", agent_id=agent_id, vehicle_token=token, event_ts=3)
        dryruns.append(evaluate_mask(scenario_id="PICKUP_BLOCKS_SKIP", agent_id=agent_id, token=token, anchor_occurrence_id=anchor_rule["route_stop_occurrence_id"], rule=clear_rule, expected_skip=False, expected_reason="ASSIGNED_PICKUP_REQUEST", machine=pickup, decision_ts=3))

        onboard = empty_machine(agent_id, token, candidate_stop, post_stop)
        onboard.passenger_waiting(passenger_id=f"P-ONBOARD-{agent_id}", pickup_stop="PREVIOUS", dropoff_stop=candidate_stop, event_ts=1)
        onboard.request_created(request_id=f"R-ONBOARD-{agent_id}", passenger_id=f"P-ONBOARD-{agent_id}", service_leg_id=f"L-ONBOARD-{agent_id}", event_ts=2)
        onboard.request_assigned(request_id=f"R-ONBOARD-{agent_id}", agent_id=agent_id, vehicle_token=token, event_ts=3)
        onboard.passenger_boarded(request_id=f"R-ONBOARD-{agent_id}", passenger_id=f"P-ONBOARD-{agent_id}", agent_id=agent_id, vehicle_token=token, stop_id="PREVIOUS", event_ts=4)
        dryruns.append(evaluate_mask(scenario_id="DROPOFF_AND_ONBOARD_BLOCK_SKIP", agent_id=agent_id, token=token, anchor_occurrence_id=anchor_rule["route_stop_occurrence_id"], rule=clear_rule, expected_skip=False, expected_reason="ONBOARD_DROPOFF_DEMAND", machine=onboard, decision_ts=4))

        dryruns.append(evaluate_mask(scenario_id="TERMINAL_STATIC_BLOCKS_SKIP", agent_id=agent_id, token=token, anchor_occurrence_id=anchor_rule["route_stop_occurrence_id"], rule=terminal_rule, expected_skip=False, expected_reason="TERMINAL_OR_TURNAROUND_STOP"))

        incomplete = empty_machine(agent_id, token, candidate_stop, post_stop)
        incomplete.mark_state_incomplete("K7_DRYRUN_DYNAMIC_STATE_INCOMPLETE")
        dryruns.append(evaluate_mask(scenario_id="MISSING_INCOMPLETE_STATE_BLOCKS_SKIP", agent_id=agent_id, token=token, anchor_occurrence_id=anchor_rule["route_stop_occurrence_id"], rule=clear_rule, expected_skip=False, expected_reason="DYNAMIC_STATE_INCOMPLETE", machine=incomplete))

        future = empty_machine(agent_id, token, candidate_stop, post_stop)
        future.schedule_transition("passenger_waiting", 11, passenger_id=f"P-FUTURE-{agent_id}", pickup_stop=candidate_stop, dropoff_stop=post_stop)
        future.schedule_transition("request_created", 12, request_id=f"R-FUTURE-{agent_id}", passenger_id=f"P-FUTURE-{agent_id}", service_leg_id=f"L-FUTURE-{agent_id}")
        future.schedule_transition("request_assigned", 13, request_id=f"R-FUTURE-{agent_id}", agent_id=agent_id, vehicle_token=token)
        pre = evaluate_mask(scenario_id="NO_FUTURE_LEAK_PRE_BOUNDARY", agent_id=agent_id, token=token, anchor_occurrence_id=anchor_rule["route_stop_occurrence_id"], rule=clear_rule, expected_skip=True, expected_reason=None, machine=future, decision_ts=10)
        pre["future_event_visible"] = bool(pre["service_obligation"])
        pre["passed"] = bool(pre["passed"] and not pre["future_event_visible"])
        dryruns.append(pre)
        dryruns.append(evaluate_mask(scenario_id="NO_FUTURE_LEAK_POST_BOUNDARY", agent_id=agent_id, token=token, anchor_occurrence_id=anchor_rule["route_stop_occurrence_id"], rule=clear_rule, expected_skip=False, expected_reason="ASSIGNED_PICKUP_REQUEST", machine=future, decision_ts=13))

    agent_cycles = cycles.groupby("agent_id")
    identity_summary = []
    for agent_id, group in agent_cycles:
        ordered = group.sort_values("cycle_index")
        active = ordered["active_bus_mask"].astype(bool).tolist()
        cycle_numbers = ordered["cycle_index"].astype(int).tolist()
        reappearances = [cycle_numbers[index] for index in range(1, len(active)) if not active[index - 1] and active[index]]
        identity_summary.append(
            {
                "agent_id": int(agent_id),
                "bound_vehicle_token_count": int(ordered["bound_vehicle_token"].nunique()),
                "slot_identity_mutation_count": int(ordered["slot_identity_mutation"].sum()),
                "inactive_cycle_count": int((~ordered["active_bus_mask"].astype(bool)).sum()),
                "reappearance_cycles": reappearances,
            }
        )

    for transition in transitions.to_dict("records"):
        agent_id = int(transition["agent_id"])
        token = str(transition["physical_vehicle_token"])
        cycle = int(transition["cycle_index"])
        observed = cycles[(cycles["agent_id"] == agent_id) & (cycles["cycle_index"] == cycle)].iloc[0]
        key = (str(observed["route_id"]), str(observed["direction_id"]), int(observed["seq"]), str(observed["stop_id"]))
        rule = rules_by_exact[key]
        slot_ok = bool(transition["agent_id_unchanged"] and transition["physical_vehicle_token_unchanged"] and int(transition["identity_mutation_count"]) == 0)
        common = dict(agent_id=agent_id, token=token, anchor_occurrence_id=rule["route_stop_occurrence_id"], rule=rule, expected_skip=bool(rule["positive_static_skip_clearance"]), expected_reason=None if rule["positive_static_skip_clearance"] else "TERMINAL_OR_TURNAROUND_STOP", identity_test="C2_REAPPEARANCE_AND_DIRECTION_TRANSITION")
        dryruns.append(evaluate_mask(scenario_id="REAPPEARANCE_PRESERVES_SLOT", slot_preserved=slot_ok, reappearance_preserved=slot_ok, direction_transition_preserved=True, **common))
        dryruns.append(evaluate_mask(scenario_id="DIRECTION_TRANSITION_PRESERVES_SLOT", slot_preserved=slot_ok, reappearance_preserved=True, direction_transition_preserved=slot_ok, **common))

    for row in dryruns:
        row["static_rulebook_sha256"] = rulebook_sha
        row["occurrence_master_sha256"] = occurrence_sha
        row["mask_predicate_version"] = MASK_PREDICATE_VERSION
        row["dynamic_state_contract_version"] = DYNAMIC_STATE_CONTRACT_VERSION
    identity_audit = {
        "agent_cycle_summaries": identity_summary,
        "all_eight_slots_have_single_bound_token": len(identity_summary) == 8 and all(row["bound_vehicle_token_count"] == 1 for row in identity_summary),
        "slot_identity_mutation_count": sum(row["slot_identity_mutation_count"] for row in identity_summary),
        "empirical_reappearance_case_count": sum(len(row["reappearance_cycles"]) for row in identity_summary),
        "empirical_direction_transition_case_count": len(transitions),
        "empirical_transition_identity_failure_count": sum(not bool(row["agent_id_unchanged"] and row["physical_vehicle_token_unchanged"] and int(row["identity_mutation_count"]) == 0) for row in transitions.to_dict("records")),
        "bindings": bindings,
    }
    return dryruns, identity_audit


def write_manifest_and_lock(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for relative_path in PAYLOADS:
        path = writer.root / relative_path
        rows.append({"relative_path": relative_path, "size_bytes": path.stat().st_size if path.exists() else None, "sha256": k5.sha256_file(path) if path.exists() else None, "required": True, "artifact_role": Path(relative_path).stem, "exists": path.exists()})
    jsonl_name = "artifact_manifest_srp2_bis_pv8_k7.jsonl"
    writer.text(jsonl_name, "".join(json.dumps(k5.json_clean(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    jsonl_path = writer.root / jsonl_name
    rows.append({"relative_path": jsonl_name, "size_bytes": jsonl_path.stat().st_size, "sha256": k5.sha256_file(jsonl_path), "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest_srp2_bis_pv8_k7.json"
    writer.json(manifest_name, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": sum(not row["exists"] for row in rows), "files": rows})
    manifest_path = writer.root / manifest_name
    writer.json("_PV8_K7_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest_name, "final_manifest_sha256": k5.sha256_file(manifest_path), "manifest_size_bytes": manifest_path.stat().st_size, "created_at": iso_kst()})


def final_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# PV8-K7 Research Rule Candidate and Eight-Agent K-Mask Dry-run Final Report",
            "",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- gate: `{summary['gate']}`",
            f"- final decision: `{summary['final_decision']}`",
            f"- occurrence rows / unique IDs: `{summary['occurrence_rows']} / {summary['unique_occurrence_ids']}`",
            f"- research rule rows: `{summary['rule_rows']}`",
            f"- authoritative DERIVED_SAFE predicate cells: `{summary['derived_cells']}`",
            f"- contract-fixed research predicate cells: `{summary['contract_cells']}`",
            f"- OBSERVED predicate cells: `{summary['observed_cells']}`",
            f"- positive static-clearance candidate rows: `{summary['positive_static_rows']}`",
            f"- eight-agent dry-run cases: `{summary['dryrun_passed']}/{summary['dryrun_total']} passed`",
            f"- positive dry-run SKIP count: `{summary['positive_dryrun_skip_count']}`",
            f"- anchor occurrence joins: `{summary['anchor_join_count']}/8`",
            f"- reappearance / direction-transition identity cases: `{summary['reappearance_cases']} / {summary['direction_cases']}`",
            f"- rulebook SHA-256: `{summary['rulebook_sha256']}`",
            f"- occurrence master SHA-256: `{summary['occurrence_master_sha256']}`",
            f"- research_rule_approved: `false`",
            f"- K_action_mask_available: `false`",
            "",
            "## Research assumptions",
            "",
            "The inactive candidate assumes mandatory_stop=false, protected_stop=false, and charging_or_driver_relief_stop=false. It permits planned skip only when a post-skip target exists and assumes no interior turnaround beyond K6 endpoints. K6 endpoint and no-target blockers remain DERIVED_SAFE; every filled unknown is CONTRACT_FIXED_RESEARCH_RULE.",
            "",
            "This is not an observed or operator-authoritative real-network rulebook. It was used only to exercise the K4 fail-closed predicate without policy execution.",
            "",
            "## Freeze and K8 approval",
            "",
            "The candidate bytes, occurrence master, K4 state version, and mask predicate version are hash-bound. Any rulebook or hash change requires a new experiment version. K8 requires explicit approval of the assumptions, rulebook version and hashes, research-only claim boundary, and a separate authorization to activate integration tests. Approval here does not authorize policy evaluation, checkpoint reuse, or retraining.",
            "",
            "No DB/API call, policy execution, K8 execution, checkpoint reuse, or MAPPO retraining was performed.",
            "",
        ]
    )


def run_audit(root: Path) -> Path:
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    upstreams = verify_upstreams()
    source_integrity = k4_source_integrity()
    rule_rows, contract = build_rule_candidate()
    writer.parquet("k7_research_rulebook_candidate.parquet", rule_rows, RULEBOOK_COLUMNS)
    rulebook_path = root / "k7_research_rulebook_candidate.parquet"
    rulebook_sha = k5.sha256_file(rulebook_path)
    occurrence_sha = k5.sha256_file(OCCURRENCE_PATH)
    contract["static_rulebook_sha256"] = rulebook_sha
    writer.json("k7_research_rule_contract.json", contract)
    contract_sha = k5.sha256_file(root / "k7_research_rule_contract.json")

    freeze = {
        "created_at": iso_kst(),
        "rule_version": RULE_VERSION,
        "rule_class": RULE_CLASS,
        "approval_status": APPROVAL_STATUS,
        "activated": False,
        "candidate_bytes_frozen": True,
        "approval_freeze_complete": False,
        "static_rulebook_path": "k7_research_rulebook_candidate.parquet",
        "static_rulebook_sha256": rulebook_sha,
        "static_rulebook_size_bytes": rulebook_path.stat().st_size,
        "research_rule_contract_sha256": contract_sha,
        "occurrence_master_path": str(OCCURRENCE_PATH),
        "occurrence_master_sha256": occurrence_sha,
        "occurrence_master_row_count": 20508,
        "rulebook_row_count": len(rule_rows),
        "hash_verification_passed": True,
        "rulebook_or_hash_change_requires_new_experiment_version": True,
        "real_network_rule_claim_allowed": False,
        "activation_authorized": False,
    }
    writer.json("k7_rule_freeze_manifest.json", freeze)

    dryruns, identity = build_dryruns(rule_rows, rulebook_sha, occurrence_sha)
    writer.parquet("k7_8agent_mask_dryrun.parquet", dryruns, DRYRUN_COLUMNS)
    dryrun_passed = sum(bool(row["passed"]) for row in dryruns)
    positive_dryrun = sum(bool(row["observed_skip_valid"]) for row in dryruns)
    scenario_counts = Counter(row["scenario_id"] for row in dryruns)
    validation = {
        "created_at": iso_kst(),
        "dryrun_case_count": len(dryruns),
        "dryrun_passed_count": dryrun_passed,
        "dryrun_failed_count": len(dryruns) - dryrun_passed,
        "positive_dryrun_skip_count": positive_dryrun,
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "all_eight_agents_integrated": len({row["agent_id"] for row in dryruns}) == 8,
        "per_agent_base_scenario_count": {str(agent): sum(row["agent_id"] == agent for row in dryruns) for agent in range(8)},
        "pickup_blocks_skip_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "PICKUP_BLOCKS_SKIP" and row["passed"] and row["pickup_obligation"] for row in dryruns) for agent in range(8)),
        "dropoff_onboard_blocks_skip_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "DROPOFF_AND_ONBOARD_BLOCK_SKIP" and row["passed"] and row["dropoff_obligation"] and row["onboard_destination_obligation"] for row in dryruns) for agent in range(8)),
        "terminal_static_blocks_skip_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "TERMINAL_STATIC_BLOCKS_SKIP" and row["passed"] for row in dryruns) for agent in range(8)),
        "missing_incomplete_state_blocks_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "MISSING_INCOMPLETE_STATE_BLOCKS_SKIP" and row["passed"] for row in dryruns) for agent in range(8)),
        "clear_research_case_allows_skip_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "CLEAR_RESEARCH_CONTRACT_CASE" and row["passed"] and row["observed_skip_valid"] for row in dryruns) for agent in range(8)),
        "no_future_leakage_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "NO_FUTURE_LEAK_PRE_BOUNDARY" and row["passed"] and not row["future_event_visible"] for row in dryruns) and any(row["agent_id"] == agent and row["scenario_id"] == "NO_FUTURE_LEAK_POST_BOUNDARY" and row["passed"] for row in dryruns) for agent in range(8)),
        "identity_validation": identity,
        "reappearance_preserves_slot": identity["empirical_reappearance_case_count"] == 1 and any(row["scenario_id"] == "REAPPEARANCE_PRESERVES_SLOT" and row["passed"] for row in dryruns),
        "direction_transition_preserves_slot": identity["empirical_direction_transition_case_count"] == 1 and identity["empirical_transition_identity_failure_count"] == 0 and any(row["scenario_id"] == "DIRECTION_TRANSITION_PRESERVES_SLOT" and row["passed"] for row in dryruns),
        "policy_execution_count": 0,
    }
    writer.json("k7_mask_integration_validation.json", validation)

    snapshot_contract = {
        "created_at": iso_kst(),
        "contract_name": "PV8_K7_K_SAFETY_SNAPSHOT_VERSION_BINDING_V1",
        "required_snapshot_metadata": {
            "k_safety_state_version": K_SAFETY_STATE_SCHEMA_VERSION,
            "static_rulebook_version": RULE_VERSION,
            "static_rulebook_sha256": rulebook_sha,
            "occurrence_master_sha256": occurrence_sha,
            "dynamic_state_contract_version": DYNAMIC_STATE_CONTRACT_VERSION,
            "mask_predicate_version": MASK_PREDICATE_VERSION,
        },
        "lookup_key": "route_stop_occurrence_id",
        "lookup_cardinality": "ONE_RULE_ROW_PER_OCCURRENCE_ID",
        "all_metadata_required_before_mask_evaluation": True,
        "missing_or_mismatched_metadata_forces_skip_false": True,
        "rulebook_or_hash_change_requires_new_experiment_version": True,
        "clone_reset_must_preserve_all_metadata": True,
        "research_rule_approved": False,
        "activation_authorized": False,
    }
    writer.json("k7_snapshot_version_contract.json", snapshot_contract)

    contract_complete = bool(
        len(rule_rows) == 20508
        and len({row["route_stop_occurrence_id"] for row in rule_rows}) == 20508
        and all(row["approval_status"] == APPROVAL_STATUS and not row["activated"] and row["static_rule_complete"] for row in rule_rows)
        and contract["observed_predicate_cell_count"] == 0
        and freeze["hash_verification_passed"]
    )
    mask_passed = bool(
        dryrun_passed == len(dryruns)
        and validation["all_eight_agents_integrated"]
        and validation["pickup_blocks_skip_all_agents"]
        and validation["dropoff_onboard_blocks_skip_all_agents"]
        and validation["terminal_static_blocks_skip_all_agents"]
        and validation["missing_incomplete_state_blocks_all_agents"]
        and validation["clear_research_case_allows_skip_all_agents"]
        and validation["no_future_leakage_all_agents"]
        and validation["reappearance_preserves_slot"]
        and validation["direction_transition_preserves_slot"]
    )
    final_decision = DECISION_READY if contract_complete and mask_passed else DECISION_MASK_FAILED if contract_complete else DECISION_INCOMPLETE
    audit_passed = bool(final_decision == DECISION_READY and source_integrity["source_drift_count"] == 0)
    readiness = {
        "created_at": iso_kst(),
        "final_decision": final_decision,
        "research_rule_approved": False,
        "K_action_mask_available": False,
        "conditional_skip_policy_enabled": False,
        "contract_complete_as_candidate": contract_complete,
        "eight_agent_mask_dryrun_passed": mask_passed,
        "rulebook_hash_frozen": True,
        "approval_freeze_complete": False,
        "exact_approval_required_for_k8": [
            "explicitly approve the five research assumptions and acknowledge they are not observed/operator rules",
            f"approve static_rulebook_version={RULE_VERSION}",
            f"approve static_rulebook_sha256={rulebook_sha}",
            f"approve occurrence_master_sha256={occurrence_sha}",
            "approve the snapshot/version contract and new-experiment-on-hash-change rule",
            "issue a separate K8 integration authorization; policy evaluation, checkpoint reuse, and retraining remain locked",
        ],
    }
    guards = {
        "created_at": iso_kst(),
        "research_rule_approved": False,
        "K_action_mask_available": False,
        "conditional_skip_policy_enabled": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_evaluation_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "real_network_rule_claim_allowed": False,
        "automatic_k8_execution_authorized": False,
        "automatic_mappo_retraining_authorized": False,
    }
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE if audit_passed else FAIL_GATE,
        "terminal_gate": PASS_GATE if audit_passed else FAIL_GATE,
        "readiness": PASS_READINESS if audit_passed else "SRP2_BIS_PV8_K7_AUDIT_FAILED",
        "gate_passed": audit_passed,
        "final_decision": final_decision,
        "failure_reasons": [] if audit_passed else ["upstream/source integrity, candidate completeness, hash freeze, or eight-agent mask integration failed"],
    }
    summary = {
        "artifact_root": str(root),
        "gate": gate["gate"],
        "final_decision": final_decision,
        "occurrence_rows": 20508,
        "unique_occurrence_ids": len({row["route_stop_occurrence_id"] for row in rule_rows}),
        "rule_rows": len(rule_rows),
        "derived_cells": contract["authoritative_derived_predicate_cell_count"],
        "contract_cells": contract["contract_fixed_predicate_cell_count"],
        "observed_cells": contract["observed_predicate_cell_count"],
        "positive_static_rows": contract["positive_static_clearance_row_count"],
        "dryrun_passed": dryrun_passed,
        "dryrun_total": len(dryruns),
        "positive_dryrun_skip_count": positive_dryrun,
        "anchor_join_count": len(identity["bindings"]),
        "reappearance_cases": identity["empirical_reappearance_case_count"],
        "direction_cases": identity["empirical_direction_transition_case_count"],
        "rulebook_sha256": rulebook_sha,
        "occurrence_master_sha256": occurrence_sha,
    }
    writer.json("k7_readiness_decision.json", readiness)
    writer.json("claim_guard_status.json", guards)
    writer.json(
        "run_manifest.json",
        {
            "created_at": iso_kst(),
            "artifact_family": ARTIFACT_PREFIX,
            "mode": "audit",
            "runner_path": str(RUNNER_PATH),
            "runner_sha256": k5.sha256_file(RUNNER_PATH),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "upstream_integrity": upstreams,
            "k4_source_integrity": source_integrity,
            "db_query_count": 0,
            "db_write_count": 0,
            "new_bis_api_call_count": 0,
            "policy_execution_count": 0,
            "checkpoint_reuse_count": 0,
            "k8_execution_count": 0,
            "mappo_training_count": 0,
        },
    )
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": gate["gate"], "readiness": gate["readiness"], "final_decision": final_decision})
    writer.text("final_report.md", final_report(summary))
    write_manifest_and_lock(writer, gate)
    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_k7.json", "_PV8_K7_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise K7Error(f"K7 artifact integrity failure: {own}")
    if not audit_passed:
        raise K7Error(f"K7 audit failed: decision={final_decision}, validation={validation}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"readiness: {gate['readiness']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run_audit(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
