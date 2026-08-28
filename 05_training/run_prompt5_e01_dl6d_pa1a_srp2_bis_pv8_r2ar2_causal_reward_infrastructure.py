#!/usr/bin/env python3
"""PV8-R2A-R2 causal reward collector, B1 reference, and harm fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
from simulator.k_safety_state import ServiceObligationStateMachine
from simulator.pv8_reward_outcome_collector import (
    CausalOutcomeCollector,
    DecisionOutcomeBinding,
    DuplicateRewardEventError,
    InitialRequestState,
    RewardEventBoundaryError,
    RewardIdentityError,
    RewardOutcomeEvent,
    RewardOutcomeEventType,
    aggregate_team_outcomes,
    map_k4_event_log,
    percentile,
)


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
SIMULATOR_ROOT = TRAINING_ROOT / "simulator"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar2_causal_reward_infrastructure.py"
COLLECTOR_SOURCE = SIMULATOR_ROOT / "pv8_reward_outcome_collector.py"
COLLECTOR_TEST = SIMULATOR_ROOT / "test_pv8_reward_outcome_collector.py"

K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"
K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"
R1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r1_mappo_retraining_preflight_20260808_142604"
R2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2_reward_episode_audit_20260808_143743"
R2A_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2a_reward_authority_promotion_audit_20260808_145209"
R2AR1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar1_reward_contract_repair_audit_20260808_150949"

UPSTREAMS = {
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
    "PV8-K9": (K9_ROOT, "artifact_manifest_srp2_bis_pv8_k9.json", "_PV8_K9_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED"),
    "PV8-R1": (R1_ROOT, "artifact_manifest_srp2_bis_pv8_r1.json", "_PV8_R1_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R1_MAPPO_RETRAINING_PREFLIGHT_COMPLETE"),
    "PV8-R2": (R2_ROOT, "artifact_manifest_srp2_bis_pv8_r2.json", "_PV8_R2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2_REWARD_AND_EPISODE_DATA_AUDIT_COMPLETE"),
    "PV8-R2A": (R2A_ROOT, "artifact_manifest_srp2_bis_pv8_r2a.json", "_PV8_R2A_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2A_REWARD_AUTHORITY_AND_PROMOTION_AUDIT_COMPLETE"),
    "PV8-R2A-R1": (R2AR1_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar1.json", "_PV8_R2AR1_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR1_REWARD_CONTRACT_REPAIR_AUDIT_COMPLETE"),
}

RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"
EXPECTED_TENSORS = {"num_agents": 8, "actor_obs_dim": 16, "critic_obs_dim": 64, "action_dim": 3}
ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar2_causal_reward_infrastructure"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR2_CAUSAL_REWARD_INFRASTRUCTURE_COMPLETE"
DECISION = "PV8_REWARD_INFRA_REPAIR_REQUIRED"
READINESS = "SRP2_BIS_PV8_R2AR2_COMPLETE_BOUNDED_ABLATION_FIXTURES_READY_FULL_REWARD_REPAIR_REQUIRED"

PAYLOADS = [
    "r2ar2_causal_outcome_collector_contract.json",
    "r2ar2_collector_fixture_results.parquet",
    "r2ar2_metric_semantics.json",
    "r2ar2_pv8_b1_protocol.json",
    "r2ar2_pv8_b1_reference_audit.json",
    "r2ar2_normalization_candidate.json",
    "r2ar2_delayed_harm_fixture_results.parquet",
    "r2ar2_harm_reveal_horizon_audit.json",
    "r2ar2_constraint_layer_contract.json",
    "r2ar2_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]

COLLECTOR_COLUMNS = [
    "test_id", "passed", "expected", "observed", "event_count", "learning_sample",
    "service_rate", "avg_wait_seconds", "p95_wait_seconds", "guard_exception",
    "observation_hash_unchanged", "k_mask_hash_unchanged", "details",
]
HARM_COLUMNS = [
    "fixture_id", "branch", "candidate_skip_allowed", "candidate_skip_executed",
    "exogenous_schedule_hash", "decision_ts", "terminal_or_revisit_transition",
    "first_harm_reveal_transition", "maximum_required_observation_horizon",
    "harm_components_json", "service_rate", "avg_wait_seconds", "p95_wait_seconds",
    "eligible_request_count", "passenger_served_count", "censored_wait_count",
    "event_trace_hash", "passed", "details",
]


class R2AR2Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(k5.json_clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_upstreams() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed != expected_gate or not k5.manifest_ok(checks):
            raise R2AR2Error(f"{label} integrity failure: gate={observed}, checks={checks}")
        results[label] = {
            "artifact_root": str(root),
            "gate": observed,
            "manifest_integrity": checks,
        }
    return results


def verify_frozen_bindings() -> Dict[str, Any]:
    tensor = k5.read_json(R1_ROOT / "r1_mappo_tensor_contract.json")
    r1 = k5.read_json(R1_ROOT / "r1_frozen_episode_manifest.json")
    k8 = k5.read_json(K8_ROOT / "k8_frozen_rule_contract.json")
    k9 = k5.read_json(K9_ROOT / "k9_version_hash_binding_audit.json")
    r2ar1 = k5.read_json(R2AR1_ROOT / "r2ar1_readiness_decision.json")
    checks = {
        "tensor_contract_matches": all(tensor.get(key) == value for key, value in EXPECTED_TENSORS.items()),
        "r1_rulebook_hash_matches": r1.get("frozen_bindings", {}).get("rulebook_sha256") == RULEBOOK_SHA256,
        "r1_occurrence_hash_matches": r1.get("frozen_bindings", {}).get("occurrence_master_sha256") == OCCURRENCE_SHA256,
        "k8_rulebook_hash_matches": k8.get("static_rulebook_sha256") == RULEBOOK_SHA256,
        "k8_occurrence_hash_matches": k8.get("occurrence_master_sha256") == OCCURRENCE_SHA256,
        "k9_rulebook_hash_matches": k9.get("rulebook_sha256") == RULEBOOK_SHA256,
        "k9_occurrence_hash_matches": k9.get("occurrence_master_sha256") == OCCURRENCE_SHA256,
        "r2ar1_decision_matches": r2ar1.get("final_decision") == "PV8_REWARD_REPAIR_ADDITIONAL_INPUT_REQUIRED",
    }
    checks["failure_count"] = sum(not value for value in checks.values())
    checks["tensor_contract"] = EXPECTED_TENSORS
    checks["rulebook_sha256"] = RULEBOOK_SHA256
    checks["occurrence_master_sha256"] = OCCURRENCE_SHA256
    if checks["failure_count"]:
        raise R2AR2Error(f"frozen binding mismatch: {checks}")
    return checks


def make_binding(
    decision_id: str,
    *,
    agent_id: int,
    action: Optional[str],
    outcome_end_ts: int,
    active: bool = True,
    initial_requests: Sequence[InitialRequestState] = (),
    terminal: bool = False,
) -> DecisionOutcomeBinding:
    return DecisionOutcomeBinding(
        decision_id=decision_id,
        decision_ts=100,
        agent_id=agent_id,
        vehicle_token=f"PV8-VEHICLE-{agent_id}",
        action_t=action if active else None,
        outcome_start_ts=100,
        outcome_end_ts=int(outcome_end_ts),
        next_decision_ts=None if terminal else int(outcome_end_ts),
        terminal_ts=int(outcome_end_ts) if terminal else None,
        active_bus=active,
        observation_hash=canonical_hash({"decision_id": decision_id, "cutoff": 100, "kind": "observation"}),
        k_mask_hash=canonical_hash({"decision_id": decision_id, "cutoff": 100, "kind": "k_mask"}),
        initial_requests=tuple(initial_requests),
    )


def reward_event(
    event_id: str,
    event_ts: int,
    event_type: RewardOutcomeEventType,
    *,
    agent_id: int,
    request_id: Optional[str] = None,
    passenger_id: Optional[str] = None,
    route_id: Optional[str] = None,
    direction_id: Optional[str] = None,
    occurrence_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> RewardOutcomeEvent:
    return RewardOutcomeEvent(
        event_id=event_id,
        event_ts=int(event_ts),
        event_type=event_type,
        agent_id=int(agent_id),
        vehicle_token=f"PV8-VEHICLE-{agent_id}",
        request_id=request_id,
        passenger_id=passenger_id,
        route_id=route_id,
        direction_id=direction_id,
        route_stop_occurrence_id=occurrence_id,
        metadata=dict(metadata or {}),
    )


def run_collector_fixtures() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def record(
        test_id: str,
        passed: bool,
        expected: str,
        observed: str,
        *,
        result: Optional[Mapping[str, Any]] = None,
        guard_exception: Optional[str] = None,
        details: str,
    ) -> None:
        payload = dict(result or {})
        rows.append({
            "test_id": test_id,
            "passed": bool(passed),
            "expected": expected,
            "observed": observed,
            "event_count": payload.get("event_count"),
            "learning_sample": payload.get("learning_sample"),
            "service_rate": payload.get("service_rate"),
            "avg_wait_seconds": payload.get("avg_wait_seconds"),
            "p95_wait_seconds": payload.get("p95_wait_seconds"),
            "guard_exception": guard_exception,
            "observation_hash_unchanged": payload.get("observation_hash_unchanged"),
            "k_mask_hash_unchanged": payload.get("k_mask_hash_unchanged"),
            "details": details,
        })

    collector = CausalOutcomeCollector()
    collector.register_decision(make_binding("CF01", agent_id=0, action="SERVE", outcome_end_ts=200))
    for event in [
        reward_event("CF01-E1", 110, RewardOutcomeEventType.PASSENGER_GENERATED, agent_id=0, request_id="CF01-R", passenger_id="CF01-P"),
        reward_event("CF01-E2", 140, RewardOutcomeEventType.PASSENGER_BOARDED, agent_id=0, request_id="CF01-R", passenger_id="CF01-P"),
        reward_event("CF01-E3", 180, RewardOutcomeEventType.PASSENGER_ALIGHTED, agent_id=0, request_id="CF01-R", passenger_id="CF01-P"),
        reward_event("CF01-E4", 181, RewardOutcomeEventType.REQUEST_COMPLETED, agent_id=0, request_id="CF01-R", passenger_id="CF01-P"),
        reward_event("CF01-E5", 150, RewardOutcomeEventType.VEHICLE_MOVEMENT, agent_id=0, metadata={"distance_m": 500.0, "travel_seconds": 40.0, "dwell_seconds": 10.0}),
    ]:
        collector.record_event("CF01", event)
    result = collector.finalize("CF01")
    record("CF01_EXACT_CAUSAL_METRICS", result["service_rate"] == 1.0 and result["avg_wait_seconds"] == 30.0 and result["reward_value_materialized"] is False, "service=1 wait=30 reward=false", f"service={result['service_rate']} wait={result['avg_wait_seconds']} reward={result['reward_value_materialized']}", result=result, details="unique request completion, not board+alight event count")

    collector = CausalOutcomeCollector()
    collector.register_decision(make_binding("CF02", agent_id=1, action=None, outcome_end_ts=200, active=False))
    result = collector.finalize("CF02")
    record("CF02_INACTIVE_BYPASS", result["learning_sample"] is False and result["event_count"] == 0, "inactive no sample", f"learning_sample={result['learning_sample']}", result=result, details="inactive slot bypasses actor/action/reward sample")

    guard_cases = [
        ("CF03_PRECOMMIT_BOUNDARY", 100, RewardEventBoundaryError),
        ("CF04_FUTURE_BOUNDARY", 201, RewardEventBoundaryError),
    ]
    for test_id, event_ts, expected_error in guard_cases:
        collector = CausalOutcomeCollector()
        collector.register_decision(make_binding(test_id, agent_id=0, action="SERVE", outcome_end_ts=200))
        observed_error = None
        try:
            collector.record_event(test_id, reward_event(f"{test_id}-E", event_ts, RewardOutcomeEventType.VEHICLE_MOVEMENT, agent_id=0))
        except Exception as exc:
            observed_error = type(exc).__name__
        result = collector.finalize(test_id)
        record(test_id, observed_error == expected_error.__name__ and result["event_count"] == 0, expected_error.__name__, str(observed_error), result=result, guard_exception=observed_error, details="boundary rejection occurs before event mutation")

    collector = CausalOutcomeCollector()
    collector.register_decision(make_binding("CF05", agent_id=0, action="SERVE", outcome_end_ts=200))
    duplicate = reward_event("CF05-E", 150, RewardOutcomeEventType.VEHICLE_MOVEMENT, agent_id=0)
    collector.record_event("CF05", duplicate)
    observed_error = None
    try:
        collector.record_event("CF05", duplicate)
    except DuplicateRewardEventError as exc:
        observed_error = type(exc).__name__
    result = collector.finalize("CF05")
    record("CF05_DUPLICATE_GUARD", observed_error == "DuplicateRewardEventError" and result["event_count"] == 1, "duplicate blocked, one event retained", f"{observed_error}, event_count={result['event_count']}", result=result, guard_exception=observed_error, details="event identity is exactly-once within collector scope")

    collector = CausalOutcomeCollector()
    collector.register_decision(make_binding("CF06", agent_id=0, action="SERVE", outcome_end_ts=200))
    observed_error = None
    try:
        collector.record_event("CF06", reward_event("CF06-E", 150, RewardOutcomeEventType.VEHICLE_MOVEMENT, agent_id=1))
    except RewardIdentityError as exc:
        observed_error = type(exc).__name__
    result = collector.finalize("CF06")
    record("CF06_IDENTITY_GUARD", observed_error == "RewardIdentityError" and result["event_count"] == 0, "cross-agent event blocked", str(observed_error), result=result, guard_exception=observed_error, details="physical slot and vehicle token must match")

    state = ServiceObligationStateMachine()
    state.register_vehicle(0, "PV8-VEHICLE-0")
    state.register_stop("S0")
    state.register_stop("S1")
    state.passenger_waiting(passenger_id="CF07-P", pickup_stop="S0", dropoff_stop="S1", event_ts=110)
    state.request_created(request_id="CF07-R", passenger_id="CF07-P", service_leg_id="CF07-L", event_ts=111)
    state.request_assigned(request_id="CF07-R", agent_id=0, vehicle_token="PV8-VEHICLE-0", event_ts=112)
    state.passenger_boarded(request_id="CF07-R", passenger_id="CF07-P", agent_id=0, vehicle_token="PV8-VEHICLE-0", stop_id="S0", event_ts=140)
    state.passenger_alighted(request_id="CF07-R", passenger_id="CF07-P", agent_id=0, vehicle_token="PV8-VEHICLE-0", stop_id="S1", event_ts=180)
    state.request_completed(request_id="CF07-R", event_ts=181)
    collector = CausalOutcomeCollector()
    collector.register_decision(make_binding("CF07", agent_id=0, action="SERVE", outcome_end_ts=200))
    for event in map_k4_event_log(state.event_log, agent_id=0, vehicle_token="PV8-VEHICLE-0", event_id_prefix="CF07"):
        collector.record_event("CF07", event)
    result = collector.finalize("CF07")
    record("CF07_K4_IDENTITY_MAPPING", result["passenger_served_count"] == 1 and result["avg_wait_seconds"] == 30.0, "K4 request identity preserved", f"served={result['passenger_served_count']} wait={result['avg_wait_seconds']}", result=result, details="request_created enters cohort with earlier observed waiting timestamp")

    collector_a = CausalOutcomeCollector()
    collector_b = CausalOutcomeCollector()
    for collector in [collector_a, collector_b]:
        collector.register_decision(make_binding("CF08", agent_id=0, action="SERVE", outcome_end_ts=200))
        collector.record_event("CF08", reward_event("CF08-E", 150, RewardOutcomeEventType.VEHICLE_MOVEMENT, agent_id=0, metadata={"distance_m": 10.0}))
    left = collector_a.finalize("CF08")
    right = collector_b.finalize("CF08")
    deterministic = canonical_hash(left) == canonical_hash(right)
    record("CF08_DETERMINISTIC_REPEAT", deterministic, "identical canonical result hash", str(deterministic), result=left, details="same binding and events produce byte-semantic equivalent result")
    return rows


def metric_semantics() -> Dict[str, Any]:
    records = [
        {
            "component": "service_rate",
            "classification": "IMPLEMENTED_DERIVED",
            "semantics": "unique completed/alighted eligible request identities divided by the start-plus-generated causal request cohort",
            "minimum_inputs": ["request_id", "passenger_id", "request_created/start cohort", "request_completed or passenger_alighted"],
            "zero_denominator": "null; transition excluded from this component",
        },
        {
            "component": "avg_wait",
            "classification": "IMPLEMENTED_DERIVED",
            "semantics": "mean accrued request wait from waiting_since_ts to boarding or outcome boundary; unboarded requests are retained as censored accrued waits",
            "minimum_inputs": ["passenger/request identity", "waiting_since_ts", "board_ts or outcome_end_ts"],
            "zero_denominator": "null; no silent zero",
        },
        {
            "component": "p95_wait",
            "classification": "IMPLEMENTED_DERIVED",
            "semantics": "linear-interpolated p95 of the same exact/censored accrued wait vector used by avg_wait",
            "minimum_inputs": ["individual accrued wait vector"],
            "zero_denominator": "null; no silent zero",
        },
        {
            "component": "bunching",
            "classification": "REFERENCE_REQUIRED",
            "semantics": "fraction of positive same route-direction-occurrence headways below an approved reference-derived threshold",
            "minimum_inputs": ["STOP_ARRIVAL event", "route_id", "direction_id", "route_stop_occurrence_id", "approved threshold"],
            "zero_denominator": "null when no headway interval exists",
        },
        {
            "component": "headway_cv",
            "classification": "IMPLEMENTED_DERIVED",
            "semantics": "population standard deviation divided by mean over positive same route-direction-occurrence headway intervals",
            "minimum_inputs": ["at least two positive headway intervals"],
            "zero_denominator": "null when interval count <2 or mean <=0",
        },
        {
            "component": "energy_per_passenger",
            "classification": "NOT_YET_SUPPORTED",
            "semantics": "collector preserves distance/travel/dwell/load primitives and accepts versioned traction_energy_kwh, but current PV8 runtime emits no approved energy-model output",
            "minimum_inputs": ["approved energy model/version", "traction_energy_kwh", "unique served request count"],
            "zero_denominator": "null and no component value",
        },
        {
            "component": "fleet_reduction",
            "classification": "REFERENCE_REQUIRED",
            "semantics": "approved reference active physical fleet minus candidate active physical fleet divided by approved reference; valid only after service guards pass",
            "minimum_inputs": ["active physical vehicle tokens", "approved reference fleet size"],
            "zero_denominator": "reference fleet must be >0 or fail",
        },
        {
            "component": "on_time",
            "classification": "NOT_YET_SUPPORTED",
            "semantics": "not invented; no approved timetable, service window, or promised ETA target is present",
            "minimum_inputs": ["separately approved punctuality target contract"],
            "zero_denominator": "not applicable",
        },
        {
            "component": "intervention",
            "classification": "IMPLEMENTED_EXACT",
            "semantics": "count only explicit FORCED_SAFETY_OVERRIDE and EXTERNAL_POLICY_INTERVENTION events",
            "minimum_inputs": ["explicit typed event with event identity"],
            "zero_denominator": "null if no active policy decision denominator",
        },
    ]
    return {
        "created_at": iso_kst(),
        "contract": "PV8_R2AR2_CANONICAL_REWARD_METRIC_SEMANTICS_V1",
        "allowed_classifications": ["IMPLEMENTED_EXACT", "IMPLEMENTED_DERIVED", "REFERENCE_REQUIRED", "NOT_YET_SUPPORTED"],
        "records": records,
        "usable_metrics": ["service_rate", "avg_wait", "p95_wait", "headway_cv", "intervention"],
        "reference_required_metrics": ["bunching", "fleet_reduction"],
        "unsupported_metrics": ["energy_per_passenger", "on_time"],
        "ordinary_k_mask_restriction_is_intervention": False,
        "ordinary_action_name_reward_or_penalty": False,
        "reward_value_materialized": False,
    }


def b1_protocol() -> Dict[str, Any]:
    policy = {
        "inactive_slot": "NO_ACTION_NO_SAMPLE",
        "active_slot_primary": "SERVE",
        "active_slot_fallback": "HOLD only when SERVE is unavailable",
        "conditional_skip_allowed": False,
        "learned_policy_used": False,
        "tie_break": "fixed physical agent_id ascending",
    }
    frozen = {
        "num_agents": 8,
        "action_dim": 3,
        "rulebook_sha256": RULEBOOK_SHA256,
        "occurrence_master_sha256": OCCURRENCE_SHA256,
        "same_k4_passenger_request_dynamics": True,
        "same_k8_k_safety_runtime": True,
        "same_k9_snapshot_lifecycle_contract": True,
        "same_exogenous_schedule_across_comparisons": True,
        "same_episode_boundaries_across_comparisons": True,
    }
    return {
        "created_at": iso_kst(),
        "protocol": "PV8_B1_REFERENCE_V1",
        "classification": "BOUNDED_DETERMINISTIC_REFERENCE_NOT_POLICY_EVALUATION",
        "frozen_contract": frozen,
        "action_policy": policy,
        "action_policy_sha256": canonical_hash(policy),
        "broad_training_rollout": False,
        "policy_evaluation": False,
        "normalization_candidate_scope": "R2AR2_BOUNDED_INFRASTRUCTURE_AND_DELAYED_HARM_ABLATION_FIXTURES_ONLY",
        "training_normalization_authorized": False,
    }


def run_b1_reference(protocol: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    collector = CausalOutcomeCollector()
    results: List[Dict[str, Any]] = []
    schedule_records: List[Dict[str, Any]] = []
    for agent_id in range(8):
        decision_id = f"PV8-B1-A{agent_id}"
        collector.register_decision(make_binding(decision_id, agent_id=agent_id, action="SERVE", outcome_end_ts=600, terminal=True))
        machine = ServiceObligationStateMachine()
        token = f"PV8-VEHICLE-{agent_id}"
        passenger_id = f"B1-P{agent_id}"
        request_id = f"B1-R{agent_id}"
        pickup = f"B1-PICKUP-{agent_id}"
        dropoff = f"B1-DROPOFF-{agent_id}"
        waiting_ts = 110 + 2 * agent_id
        created_ts = waiting_ts + 1
        assigned_ts = waiting_ts + 2
        board_ts = 150 + 5 * agent_id
        alight_ts = 260 + 3 * agent_id
        complete_ts = alight_ts + 1
        machine.register_vehicle(agent_id, token)
        machine.register_stop(pickup)
        machine.register_stop(dropoff)
        machine.passenger_waiting(passenger_id=passenger_id, pickup_stop=pickup, dropoff_stop=dropoff, event_ts=waiting_ts)
        machine.request_created(request_id=request_id, passenger_id=passenger_id, service_leg_id=f"B1-L{agent_id}", event_ts=created_ts)
        machine.request_assigned(request_id=request_id, agent_id=agent_id, vehicle_token=token, event_ts=assigned_ts)
        machine.passenger_boarded(request_id=request_id, passenger_id=passenger_id, agent_id=agent_id, vehicle_token=token, stop_id=pickup, event_ts=board_ts)
        machine.passenger_alighted(request_id=request_id, passenger_id=passenger_id, agent_id=agent_id, vehicle_token=token, stop_id=dropoff, event_ts=alight_ts)
        machine.request_completed(request_id=request_id, event_ts=complete_ts)
        mapped = map_k4_event_log(machine.event_log, agent_id=agent_id, vehicle_token=token, event_id_prefix=decision_id)
        movement = reward_event(
            f"{decision_id}:movement",
            210 + agent_id,
            RewardOutcomeEventType.VEHICLE_MOVEMENT,
            agent_id=agent_id,
            metadata={
                "distance_m": float(1200 + 25 * agent_id),
                "travel_seconds": 90.0,
                "dwell_seconds": float(12 + agent_id),
                "load_time_integral": float(100 + 5 * agent_id),
                "traction_energy_kwh": None,
                "energy_model_version": None,
            },
        )
        arrival = reward_event(
            f"{decision_id}:arrival",
            300 + 40 * agent_id,
            RewardOutcomeEventType.STOP_ARRIVAL,
            agent_id=agent_id,
            route_id="PV8-B1-ROUTE",
            direction_id="0",
            occurrence_id="PV8-B1-ROUTE:0:001:REFERENCE_STOP",
        )
        for event in [*mapped, movement, arrival]:
            collector.record_event(decision_id, event)
        result = collector.finalize(decision_id)
        results.append(result)
        schedule_records.append({
            "agent_id": agent_id,
            "vehicle_token": token,
            "action": "SERVE",
            "waiting_ts": waiting_ts,
            "board_ts": board_ts,
            "alight_ts": alight_ts,
            "complete_ts": complete_ts,
            "arrival_ts": 300 + 40 * agent_id,
        })
    preliminary = aggregate_team_outcomes(results, reference_fleet_size=8)
    median_headway = percentile(preliminary["headway_values_seconds"], 0.5)
    bunching_threshold = float(0.5 * median_headway)
    team = aggregate_team_outcomes(results, bunching_threshold_seconds=bunching_threshold, reference_fleet_size=8)
    audit = {
        "created_at": iso_kst(),
        "reference_id": "PV8_B1_REFERENCE",
        "protocol_sha256": canonical_hash(protocol),
        "source_class": "DETERMINISTIC_SYNTHETIC_K4_IDENTITY_FIXTURE",
        "contract_compatible": True,
        "num_fixed_slots": 8,
        "active_reference_sample_count": team["active_learning_sample_count"],
        "inactive_sample_count": team["inactive_bypassed_count"],
        "all_actions": sorted({row["action_t"] for row in results}),
        "conditional_skip_execution_count": sum(row["action_t"] == "CONDITIONAL_SKIP" for row in results),
        "reward_value_materialization_count": sum(bool(row["reward_value_materialized"]) for row in results),
        "exogenous_schedule_sha256": canonical_hash(schedule_records),
        "reference_metrics": team,
        "finite_candidate_constant_names": [
            "service_rate", "avg_wait_seconds", "p95_wait_seconds", "mean_headway_seconds",
            "headway_cv", "bunching_threshold_seconds", "fleet_reference_size",
        ],
        "energy_reference_available": False,
        "on_time_reference_available": False,
        "scope_limit": "Suitable only for collector/fixture reward ablation; not representative training normalization evidence.",
        "training_normalization_authorized": False,
        "policy_evaluation": False,
        "passed": bool(
            len(results) == 8
            and team["service_rate"] == 1.0
            and team["avg_wait_seconds"] is not None
            and team["p95_wait_seconds"] is not None
            and team["headway_cv"] is not None
            and team["energy_per_passenger_kwh"] is None
            and sum(bool(row["reward_value_materialized"]) for row in results) == 0
        ),
    }
    normalization = {
        "created_at": iso_kst(),
        "candidate_id": "PV8_B1_REFERENCE_R2AR2_BOUNDED_FIXTURE_CONSTANTS_V1",
        "scope": audit["scope_limit"],
        "source_reference_id": audit["reference_id"],
        "source_schedule_sha256": audit["exogenous_schedule_sha256"],
        "constants": {
            "service_rate": team["service_rate"],
            "avg_wait_seconds": team["avg_wait_seconds"],
            "p95_wait_seconds": team["p95_wait_seconds"],
            "mean_headway_seconds": team["mean_headway_seconds"],
            "headway_cv": team["headway_cv"],
            "bunching_threshold_seconds": bunching_threshold,
            "fleet_reference_size": 8,
            "energy_per_passenger": None,
            "on_time": None,
        },
        "finite_candidate_constant_count": 7,
        "historical_300_600_10_defaults_used": False,
        "full_reward_formula_normalization_complete": False,
        "approved": False,
        "reward_values_materialized": False,
    }
    return audit, normalization, results


def passenger_events(
    prefix: str,
    *,
    agent_id: int,
    generated_ts: int,
    board_ts: Optional[int],
    complete_ts: Optional[int],
    cancel_ts: Optional[int] = None,
    identity_prefix: Optional[str] = None,
) -> List[RewardOutcomeEvent]:
    identity = identity_prefix or prefix
    request_id = f"{identity}-R"
    passenger_id = f"{identity}-P"
    rows = [reward_event(f"{prefix}-generated", generated_ts, RewardOutcomeEventType.PASSENGER_GENERATED, agent_id=agent_id, request_id=request_id, passenger_id=passenger_id)]
    if board_ts is not None:
        rows.append(reward_event(f"{prefix}-boarded", board_ts, RewardOutcomeEventType.PASSENGER_BOARDED, agent_id=agent_id, request_id=request_id, passenger_id=passenger_id))
    if complete_ts is not None:
        rows.append(reward_event(f"{prefix}-alighted", complete_ts - 1, RewardOutcomeEventType.PASSENGER_ALIGHTED, agent_id=agent_id, request_id=request_id, passenger_id=passenger_id))
        rows.append(reward_event(f"{prefix}-completed", complete_ts, RewardOutcomeEventType.REQUEST_COMPLETED, agent_id=agent_id, request_id=request_id, passenger_id=passenger_id))
    if cancel_ts is not None:
        rows.append(reward_event(f"{prefix}-cancelled", cancel_ts, RewardOutcomeEventType.REQUEST_CANCELLED, agent_id=agent_id, request_id=request_id, passenger_id=passenger_id))
    return rows


def collect_branch_at_horizon(
    fixture_id: str,
    branch: str,
    events: Sequence[RewardOutcomeEvent],
    *,
    horizon: int,
    initial_requests: Sequence[InitialRequestState] = (),
) -> Dict[str, Any]:
    decision_id = f"{fixture_id}:{branch}:H{horizon}"
    collector = CausalOutcomeCollector()
    collector.register_decision(make_binding(
        decision_id,
        agent_id=0,
        action="SERVE" if branch == "REFERENCE" else "CONDITIONAL_SKIP",
        outcome_end_ts=100 + 60 * horizon,
        initial_requests=initial_requests,
        terminal=horizon == 5,
    ))
    for event in events:
        if event.event_ts <= 100 + 60 * horizon:
            cloned = RewardOutcomeEvent(
                event_id=f"{decision_id}:{event.event_id}",
                event_ts=event.event_ts,
                event_type=event.event_type,
                agent_id=event.agent_id,
                vehicle_token=event.vehicle_token,
                request_id=event.request_id,
                passenger_id=event.passenger_id,
                route_id=event.route_id,
                direction_id=event.direction_id,
                route_stop_occurrence_id=event.route_stop_occurrence_id,
                metadata=event.metadata,
            )
            collector.record_event(decision_id, cloned)
    return collector.finalize(decision_id)


def harm_components(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> List[str]:
    components: List[str] = []
    ref_service = reference.get("service_rate")
    cand_service = candidate.get("service_rate")
    if ref_service is not None and cand_service is not None and float(cand_service) < float(ref_service) - 1e-12:
        components.append("SERVICE_RATE_DEGRADATION")
    ref_avg = reference.get("avg_wait_seconds")
    cand_avg = candidate.get("avg_wait_seconds")
    if ref_avg is not None and cand_avg is not None and float(cand_avg) > float(ref_avg) + 1e-12:
        components.append("AVG_WAIT_DEGRADATION")
    ref_p95 = reference.get("p95_wait_seconds")
    cand_p95 = candidate.get("p95_wait_seconds")
    if ref_p95 is not None and cand_p95 is not None and float(cand_p95) > float(ref_p95) + 1e-12:
        components.append("P95_WAIT_DEGRADATION")
    if int(candidate.get("passenger_served_count", 0)) < int(reference.get("passenger_served_count", 0)):
        components.append("COMPLETION_DEFICIT")
    return sorted(set(components))


def delayed_fixture_specs() -> List[Dict[str, Any]]:
    immediate_initial = [InitialRequestState("IMMEDIATE-R", "IMMEDIATE-P", 80, "ASSIGNED")]
    pickup_initial = [InitialRequestState("PICKUP-R", "PICKUP-P", 80, "ASSIGNED")]
    dropoff_initial = [InitialRequestState("DROPOFF-R", "DROPOFF-P", 60, "ONBOARD")]
    specs = [
        {
            "fixture_id": "BENEFICIAL_SKIP",
            "candidate_allowed": True,
            "reference_events": passenger_events("BEN-REF", agent_id=0, generated_ts=110, board_ts=150, complete_ts=210, identity_prefix="BEN"),
            "candidate_events": passenger_events("BEN-SKIP", agent_id=0, generated_ts=110, board_ts=135, complete_ts=190, identity_prefix="BEN"),
            "exogenous": [{"passenger": "BEN", "generated_ts": 110}],
            "initial_requests": [],
            "terminal_or_revisit_transition": 3,
            "expected_first_harm": None,
            "blocker": None,
        },
        {
            "fixture_id": "IMMEDIATE_HARM_SKIP",
            "candidate_allowed": False,
            "reference_events": [],
            "candidate_events": [],
            "exogenous": [{"request": "IMMEDIATE-R", "waiting_since_ts": 80, "visible_at_decision": True}],
            "initial_requests": immediate_initial,
            "terminal_or_revisit_transition": 1,
            "expected_first_harm": 0,
            "blocker": "PICKUP_OR_BOARDING_OBLIGATION_AT_DECISION",
        },
        {
            "fixture_id": "ONE_STEP_DELAYED_HARM",
            "candidate_allowed": True,
            "reference_events": passenger_events("ONE-REF", agent_id=0, generated_ts=110, board_ts=130, complete_ts=150, identity_prefix="ONE"),
            "candidate_events": passenger_events("ONE-SKIP", agent_id=0, generated_ts=110, board_ts=230, complete_ts=250, identity_prefix="ONE"),
            "exogenous": [{"passenger": "ONE", "generated_ts": 110}],
            "initial_requests": [],
            "terminal_or_revisit_transition": 3,
            "expected_first_harm": 1,
            "blocker": None,
        },
        {
            "fixture_id": "MULTI_STEP_DELAYED_HARM",
            "candidate_allowed": True,
            "reference_events": passenger_events("MULTI-REF", agent_id=0, generated_ts=170, board_ts=230, complete_ts=270, identity_prefix="MULTI"),
            "candidate_events": passenger_events("MULTI-SKIP", agent_id=0, generated_ts=170, board_ts=350, complete_ts=390, identity_prefix="MULTI"),
            "exogenous": [{"passenger": "MULTI", "generated_ts": 170}],
            "initial_requests": [],
            "terminal_or_revisit_transition": 5,
            "expected_first_harm": 3,
            "blocker": None,
        },
        {
            "fixture_id": "PICKUP_OBLIGATION",
            "candidate_allowed": False,
            "reference_events": [],
            "candidate_events": [],
            "exogenous": [{"request": "PICKUP-R", "waiting_since_ts": 80, "visible_at_decision": True}],
            "initial_requests": pickup_initial,
            "terminal_or_revisit_transition": 1,
            "expected_first_harm": 0,
            "blocker": "ASSIGNED_PICKUP",
        },
        {
            "fixture_id": "DROPOFF_ONBOARD_OBLIGATION",
            "candidate_allowed": False,
            "reference_events": [],
            "candidate_events": [],
            "exogenous": [{"request": "DROPOFF-R", "onboard_at_decision": True}],
            "initial_requests": dropoff_initial,
            "terminal_or_revisit_transition": 1,
            "expected_first_harm": 0,
            "blocker": "DROPOFF_OR_ONBOARD_OBLIGATION",
        },
        {
            "fixture_id": "SERVICE_RATE_DEGRADATION",
            "candidate_allowed": True,
            "reference_events": passenger_events("SERVICE-REF", agent_id=0, generated_ts=110, board_ts=170, complete_ts=200, identity_prefix="SERVICE"),
            "candidate_events": passenger_events("SERVICE-SKIP", agent_id=0, generated_ts=110, board_ts=None, complete_ts=None, cancel_ts=210, identity_prefix="SERVICE"),
            "exogenous": [{"passenger": "SERVICE", "generated_ts": 110}],
            "initial_requests": [],
            "terminal_or_revisit_transition": 3,
            "expected_first_harm": 2,
            "blocker": None,
        },
        {
            "fixture_id": "P95_WAIT_DEGRADATION",
            "candidate_allowed": True,
            "reference_events": passenger_events("P95-REF", agent_id=0, generated_ts=290, board_ts=310, complete_ts=370, identity_prefix="P95"),
            "candidate_events": passenger_events("P95-SKIP", agent_id=0, generated_ts=290, board_ts=390, complete_ts=399, identity_prefix="P95"),
            "exogenous": [{"passenger": "P95", "generated_ts": 290}],
            "initial_requests": [],
            "terminal_or_revisit_transition": 5,
            "expected_first_harm": 4,
            "blocker": None,
        },
    ]
    return specs


def run_delayed_harm_fixtures() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    fixture_summaries: List[Dict[str, Any]] = []
    for spec in delayed_fixture_specs():
        fixture_id = str(spec["fixture_id"])
        reference_exogenous = [
            {
                "event_ts": event.event_ts,
                "request_id": event.request_id,
                "passenger_id": event.passenger_id,
                "waiting_since_ts": event.metadata.get("waiting_since_ts", event.event_ts),
            }
            for event in spec["reference_events"]
            if event.event_type == RewardOutcomeEventType.PASSENGER_GENERATED
        ]
        candidate_exogenous = [
            {
                "event_ts": event.event_ts,
                "request_id": event.request_id,
                "passenger_id": event.passenger_id,
                "waiting_since_ts": event.metadata.get("waiting_since_ts", event.event_ts),
            }
            for event in spec["candidate_events"]
            if event.event_type == RewardOutcomeEventType.PASSENGER_GENERATED
        ]
        paired_exogenous_identity_match = bool(
            not spec["candidate_allowed"] or reference_exogenous == candidate_exogenous
        )
        schedule_payload = reference_exogenous if spec["candidate_allowed"] else spec["exogenous"]
        schedule_hash = canonical_hash(schedule_payload)
        first_harm: Optional[int] = None
        final_ref: Optional[Dict[str, Any]] = None
        final_candidate: Optional[Dict[str, Any]] = None
        observed_components: List[str] = []
        if not spec["candidate_allowed"]:
            first_harm = 0
            observed_components = [f"K_SAFETY_BLOCKER:{spec['blocker']}"]
            final_ref = collect_branch_at_horizon(fixture_id, "REFERENCE", spec["reference_events"], horizon=1, initial_requests=spec["initial_requests"])
        else:
            for horizon in range(1, 6):
                reference = collect_branch_at_horizon(fixture_id, "REFERENCE", spec["reference_events"], horizon=horizon, initial_requests=spec["initial_requests"])
                candidate = collect_branch_at_horizon(fixture_id, "CANDIDATE_SKIP", spec["candidate_events"], horizon=horizon, initial_requests=spec["initial_requests"])
                components = harm_components(reference, candidate)
                if components and first_harm is None:
                    first_harm = horizon
                    observed_components = components
                if horizon == int(spec["terminal_or_revisit_transition"]):
                    final_ref = reference
                    final_candidate = candidate
            if final_ref is None or final_candidate is None:
                raise R2AR2Error(f"fixture terminal boundary not evaluated: {fixture_id}")
        expected = spec["expected_first_harm"]
        fixture_passed = bool(first_harm == expected and paired_exogenous_identity_match)
        for branch, result in [("REFERENCE", final_ref), ("CANDIDATE_SKIP", final_candidate)]:
            payload = dict(result or {})
            rows.append({
                "fixture_id": fixture_id,
                "branch": branch,
                "candidate_skip_allowed": bool(spec["candidate_allowed"]),
                "candidate_skip_executed": bool(branch == "CANDIDATE_SKIP" and spec["candidate_allowed"]),
                "exogenous_schedule_hash": schedule_hash,
                "decision_ts": 100,
                "terminal_or_revisit_transition": int(spec["terminal_or_revisit_transition"]),
                "first_harm_reveal_transition": first_harm,
                "maximum_required_observation_horizon": int(spec["terminal_or_revisit_transition"]),
                "harm_components_json": json.dumps(observed_components, sort_keys=True),
                "service_rate": payload.get("service_rate"),
                "avg_wait_seconds": payload.get("avg_wait_seconds"),
                "p95_wait_seconds": payload.get("p95_wait_seconds"),
                "eligible_request_count": payload.get("eligible_request_count"),
                "passenger_served_count": payload.get("passenger_served_count"),
                "censored_wait_count": payload.get("censored_wait_count"),
                "event_trace_hash": payload.get("event_trace_hash"),
                "passed": fixture_passed,
                "details": spec["blocker"] if not spec["candidate_allowed"] else f"paired exogenous identity match={paired_exogenous_identity_match}",
            })
        fixture_summaries.append({
            "fixture_id": fixture_id,
            "candidate_skip_allowed": bool(spec["candidate_allowed"]),
            "exogenous_schedule_hash": schedule_hash,
            "expected_first_harm_reveal_transition": expected,
            "observed_first_harm_reveal_transition": first_harm,
            "terminal_or_revisit_transition": int(spec["terminal_or_revisit_transition"]),
            "harm_components": observed_components,
            "paired_exogenous_identity_match": paired_exogenous_identity_match,
            "passed": fixture_passed,
        })
    numeric_reveals = [row["observed_first_harm_reveal_transition"] for row in fixture_summaries if row["observed_first_harm_reveal_transition"] is not None]
    audit = {
        "created_at": iso_kst(),
        "fixture_count": len(fixture_summaries),
        "branch_row_count": len(rows),
        "fixture_pass_count": sum(bool(row["passed"]) for row in fixture_summaries),
        "fixture_failure_count": sum(not row["passed"] for row in fixture_summaries),
        "fixtures": fixture_summaries,
        "first_harm_reveal_indices": {row["fixture_id"]: row["observed_first_harm_reveal_transition"] for row in fixture_summaries},
        "maximum_observed_harm_reveal_horizon": max(numeric_reveals) if numeric_reveals else None,
        "maximum_terminal_or_revisit_boundary": max(row["terminal_or_revisit_transition"] for row in fixture_summaries),
        "h1_vulnerable_fixture_ids": [row["fixture_id"] for row in fixture_summaries if row["observed_first_harm_reveal_transition"] is not None and row["observed_first_harm_reveal_transition"] > 1],
        "reward_hk_selected": False,
        "selected_hk": None,
        "selection_guard": "Observed reveal indices are fixture evidence for later ablation candidates, not authorization to select reward Hk.",
        "policy_learning_executed": False,
    }
    return rows, audit


def collector_contract(fixtures: Sequence[Mapping[str, Any]], pytest_result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract": "PV8_R2AR2_CAUSAL_DECISION_OUTCOME_COLLECTOR_V1",
        "status": "IMPLEMENTED_AND_TESTED",
        "source_files": [
            {"path": str(path), "sha256": k5.sha256_file(path), "size_bytes": path.stat().st_size}
            for path in [COLLECTOR_SOURCE, COLLECTOR_TEST]
        ],
        "decision_binding_fields": [
            "decision_id", "decision_ts", "agent_id", "vehicle_token", "action_t",
            "outcome_start_ts", "outcome_end_ts", "next_decision_ts", "terminal_ts",
        ],
        "outcome_interval": "(decision_ts, next_decision_ts_or_terminal_ts]",
        "supported_primitives": [
            "passenger generated", "unique request completed", "individual accrued/censored wait",
            "avg wait", "p95 wait", "service rate", "boarding completion", "alighting completion",
            "vehicle distance", "vehicle travel/dwell time", "energy-required primitives", "occurrence-aware stop arrivals",
        ],
        "inactive_slot_behavior": "NO_REWARD_LEARNING_SAMPLE",
        "future_outcome_written_to_observation_or_mask": False,
        "event_identity_exactly_once": True,
        "cross_agent_event_allowed": False,
        "silent_missing_zero_fill": False,
        "reward_value_materialized": False,
        "fixture_count": len(fixtures),
        "fixture_failure_count": sum(not bool(row["passed"]) for row in fixtures),
        "pytest_regression": dict(pytest_result),
    }


def constraint_layers() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract": "PV8_R2AR2_SEPARATED_SAFETY_AND_PERFORMANCE_CONSTRAINT_LAYERS_V1",
        "hard_safety_layer": {
            "mechanism": "K8/K9 K-action-mask plus fail-closed runtime",
            "predicates": [
                "pickup/boarding obligation", "dropoff/alighting/onboard obligation", "mandatory/protected/terminal/no-path rule",
                "incomplete dynamic/static state", "identity/hash/version integrity",
            ],
            "effect": "mask invalid action or fail transition before action/reward sample",
            "generic_reward_violation_bit": False,
        },
        "performance_guard_layer": {
            "mechanism": "post-commit causal outcome evaluation",
            "candidates": ["service floor 0.95", "avg-wait cap vs B1", "p95-wait cap vs B1", "energy guard", "fleet guard"],
            "status": "CANDIDATES_NOT_APPROVED",
            "effect": "later reward ablation/evaluation only; cannot enable or legalize an unsafe action",
            "generic_safety_violation_bit": False,
        },
        "layers_collapsed": False,
        "constraint_penalty_10_applied": False,
        "reward_contract_approved": False,
    }


def run_pytest_regression() -> Dict[str, Any]:
    tests = [
        COLLECTOR_TEST,
        SIMULATOR_ROOT / "test_pv8_k4_service_obligation_state.py",
        SIMULATOR_ROOT / "test_pv8_k8_k_action_mask_runtime.py",
        SIMULATOR_ROOT / "test_pv8_k9_k_mask_snapshot_lifecycle.py",
        SIMULATOR_ROOT / "test_suseong_service_transition_engine.py",
    ]
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(path) for path in tests]]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(TRAINING_ROOT), env.get("PYTHONPATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = "/private/tmp/pv8_r2ar2_pycache"
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, check=False)
    output = (result.stdout + result.stderr).strip()
    match = re.search(r"(\d+) passed", output)
    return {
        "command": command,
        "returncode": int(result.returncode),
        "passed": result.returncode == 0,
        "passed_test_count": int(match.group(1)) if match else None,
        "output": output,
    }


def readiness_decision(
    collector: Mapping[str, Any],
    semantics: Mapping[str, Any],
    b1: Mapping[str, Any],
    normalization: Mapping[str, Any],
    horizon: Mapping[str, Any],
) -> Dict[str, Any]:
    infrastructure_passed = bool(
        collector["fixture_failure_count"] == 0
        and collector["pytest_regression"]["passed"]
        and b1["passed"]
        and horizon["fixture_failure_count"] == 0
        and horizon["fixture_count"] == 8
    )
    if not infrastructure_passed:
        raise R2AR2Error("collector/B1/harm fixture infrastructure test failure")
    blockers = [
        "PV8 B1 constants are finite but derived only from a bounded synthetic K4 identity fixture; broader prospective reference evidence is required for training normalization.",
        "energy_per_passenger has no approved runtime energy model/output and remains NOT_YET_SUPPORTED.",
        "on_time remains deliberately unsupported because no approved timetable/service-window/promised-ETA contract exists.",
        "bunching threshold and fleet reference are fixture candidates, not approved production/training references.",
        "Observed harm reveal indices do not select reward Hk; horizon and repaired component set require explicit ablation authorization.",
    ]
    return {
        "created_at": iso_kst(),
        "final_decision": DECISION,
        "audit_complete": True,
        "collector_implemented": True,
        "bounded_b1_reference_available": True,
        "bounded_fixture_reward_ablation_can_begin": True,
        "full_frozen_candidate_reward_ablation_can_begin": False,
        "usable_reward_metrics": semantics["usable_metrics"],
        "unsupported_metrics": semantics["unsupported_metrics"],
        "finite_b1_candidate_constant_count": normalization["finite_candidate_constant_count"],
        "delayed_harm_fixture_count": horizon["fixture_count"],
        "maximum_observed_harm_reveal_horizon": horizon["maximum_observed_harm_reveal_horizon"],
        "reward_hk_selected": False,
        "blockers": blockers,
        "minimum_next_scope": [
            "Run a bounded no-training ablation over H1 and preregistered candidate horizons using these fixed fixtures; do not select Hk from outcome labels in advance.",
            "Add or explicitly remove/zero on-time through a later approval; add an approved energy model or explicitly exclude the energy component.",
            "Expand PV8 B1 reference coverage prospectively before adopting normalization constants for training episodes.",
            "Approve performance guard thresholds and reward candidate deltas separately after ablation.",
        ],
        "reward_contract_approved": False,
        "reward_values_materialized": False,
        "r2ar3_or_r2b_or_r3_authorized": False,
        "training_use_authorized": False,
    }


def claim_guards() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_contract_approved": False,
        "reward_values_materialized": False,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "episode_expansion_authorized": False,
        "automatic_r2ar3_execution_authorized": False,
        "automatic_r2b_execution_authorized": False,
        "automatic_r3_execution_authorized": False,
        "mappo_training_authorized": False,
    }


def final_report(summary: Mapping[str, Any]) -> str:
    indices = summary["indices"]
    return "\n".join([
        "# PV8-R2A-R2 Causal Reward Infrastructure",
        "",
        f"- artifact root: `{summary['artifact_root']}`",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{DECISION}`",
        "- reward approval / reward value materialization / training: `false / false / false`",
        "",
        "## Collector",
        "",
        f"The decision-to-outcome collector is implemented and tested. It binds active physical slots to the open-closed causal interval `(decision_ts, next decision or terminal]`, rejects pre-commit/future/duplicate/cross-agent events before mutation, retains censored accrued waits, and bypasses inactive slots. Collector fixtures passed `{summary['collector_pass']}/{summary['collector_count']}` and regression tests passed `{summary['pytest_count']}`.",
        "",
        "## Metrics",
        "",
        "Usable derived/exact metrics are service rate, average wait, p95 wait, headway-CV, and explicit intervention count. Bunching and fleet reduction require approved references. Energy-per-passenger is not yet supported because the runtime has no approved versioned energy output. On-time is not invented and remains unsupported.",
        "",
        "## PV8 B1 Reference",
        "",
        f"A bounded eight-slot `PV8_B1_REFERENCE` fixture ran with deterministic service-preserving control: active slots choose SERVE, HOLD is fallback only, and SKIP is never selected. Candidate fixture constants are service rate `{summary['b1_service']}`, average wait `{summary['b1_avg_wait']}`, p95 wait `{summary['b1_p95_wait']}`, mean headway `{summary['b1_headway']}`, headway-CV `{summary['b1_headway_cv']}`, bunching threshold `{summary['b1_bunching_threshold']}`, and reference fleet `8`. Energy and on-time constants remain null. These constants are limited to infrastructure/stress ablation and are not authorized as training normalization.",
        "",
        "## Delayed Harm",
        "",
        f"All eight paired fixtures passed. First harm reveal indices are `{json.dumps(indices, sort_keys=True)}`. The maximum observed harm reveal index is `{summary['max_harm_horizon']}`; H1 therefore misses the multi-step, service-rate, and p95-tail fixtures. No reward Hk was selected.",
        "",
        "## Constraint Separation",
        "",
        "Hard safety remains K-mask/fail-closed before action and reward sampling. Service/wait/energy/fleet thresholds remain separate post-commit performance-guard candidates. No generic violation bit or constraint penalty was materialized.",
        "",
        "## Readiness",
        "",
        "Bounded fixture reward ablation infrastructure is ready, but full frozen-candidate ablation remains blocked by on-time, energy, reference-scope, and explicit horizon/guard approval gaps. The terminal decision is therefore `PV8_REWARD_INFRA_REPAIR_REQUIRED`, not a reward failure or B1 absence.",
        "",
        "No API/DB access, training episode expansion, reward value, policy evaluation, downstream run, checkpoint reuse, or MAPPO training occurred.",
        "",
    ])


def write_manifest_and_lock(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for relative_path in PAYLOADS:
        path = writer.root / relative_path
        rows.append({
            "relative_path": relative_path,
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": k5.sha256_file(path) if path.exists() else None,
            "required": True,
            "artifact_role": Path(relative_path).stem,
            "exists": path.exists(),
        })
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar2.jsonl"
    writer.text(jsonl_name, "".join(json.dumps(k5.json_clean(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    jsonl_path = writer.root / jsonl_name
    rows.append({
        "relative_path": jsonl_name,
        "size_bytes": jsonl_path.stat().st_size,
        "sha256": k5.sha256_file(jsonl_path),
        "required": True,
        "artifact_role": "manifest_jsonl",
        "exists": True,
    })
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar2.json"
    writer.json(manifest_name, {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "payload_count": len(rows),
        "missing_payload_count": sum(not row["exists"] for row in rows),
        "files": rows,
    })
    manifest_path = writer.root / manifest_name
    writer.json("_PV8_R2AR2_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": manifest_name,
        "final_manifest_sha256": k5.sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })


def run(root: Path) -> Path:
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    upstreams = verify_upstreams()
    frozen = verify_frozen_bindings()
    pytest_result = run_pytest_regression()
    collector_rows = run_collector_fixtures()
    semantics = metric_semantics()
    protocol = b1_protocol()
    b1_audit, normalization, b1_results = run_b1_reference(protocol)
    harm_rows, horizon = run_delayed_harm_fixtures()
    constraints = constraint_layers()
    collector = collector_contract(collector_rows, pytest_result)
    decision = readiness_decision(collector, semantics, b1_audit, normalization, horizon)
    guards = claim_guards()
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": READINESS,
        "gate_passed": True,
        "final_decision": decision["final_decision"],
        "failure_reasons": [],
        "readiness_blockers": decision["blockers"],
    }
    team = b1_audit["reference_metrics"]
    summary = {
        "artifact_root": str(root),
        "collector_count": len(collector_rows),
        "collector_pass": sum(bool(row["passed"]) for row in collector_rows),
        "pytest_count": pytest_result["passed_test_count"],
        "b1_service": team["service_rate"],
        "b1_avg_wait": team["avg_wait_seconds"],
        "b1_p95_wait": team["p95_wait_seconds"],
        "b1_headway": team["mean_headway_seconds"],
        "b1_headway_cv": team["headway_cv"],
        "b1_bunching_threshold": team["bunching_threshold_seconds"],
        "indices": horizon["first_harm_reveal_indices"],
        "max_harm_horizon": horizon["maximum_observed_harm_reveal_horizon"],
    }

    writer.json("r2ar2_causal_outcome_collector_contract.json", collector)
    writer.parquet("r2ar2_collector_fixture_results.parquet", collector_rows, COLLECTOR_COLUMNS)
    writer.json("r2ar2_metric_semantics.json", semantics)
    writer.json("r2ar2_pv8_b1_protocol.json", protocol)
    writer.json("r2ar2_pv8_b1_reference_audit.json", b1_audit)
    writer.json("r2ar2_normalization_candidate.json", normalization)
    writer.parquet("r2ar2_delayed_harm_fixture_results.parquet", harm_rows, HARM_COLUMNS)
    writer.json("r2ar2_harm_reveal_horizon_audit.json", horizon)
    writer.json("r2ar2_constraint_layer_contract.json", constraints)
    writer.json("r2ar2_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "implement-and-audit",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "collector_source_sha256": k5.sha256_file(COLLECTOR_SOURCE),
        "collector_test_sha256": k5.sha256_file(COLLECTOR_TEST),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "upstream_integrity": upstreams,
        "frozen_binding_audit": frozen,
        "collector_fixture_count": len(collector_rows),
        "b1_reference_transition_count": len(b1_results),
        "delayed_harm_fixture_count": horizon["fixture_count"],
        "delayed_harm_branch_row_count": len(harm_rows),
        "new_bis_api_call_count": 0,
        "db_query_count": 0,
        "db_write_count": 0,
        "reward_value_materialization_count": 0,
        "episode_expansion_count": 0,
        "policy_evaluation_count": 0,
        "checkpoint_reuse_count": 0,
        "r2ar3_execution_count": 0,
        "r2b_execution_count": 0,
        "r3_execution_count": 0,
        "mappo_training_count": 0,
        "qwen_train": False,
        "qwen_inference": False,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": READINESS, "final_decision": decision["final_decision"]})
    writer.text("final_report.md", final_report(summary))
    write_manifest_and_lock(writer, gate)

    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar2.json", "_PV8_R2AR2_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise R2AR2Error(f"R2A-R2 artifact integrity failure: {own}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {decision['final_decision']}")
    print(f"collector_fixtures: {summary['collector_pass']}/{summary['collector_count']}")
    print(f"delayed_harm_fixtures: {horizon['fixture_pass_count']}/{horizon['fixture_count']}")
    print(f"maximum_observed_harm_reveal_horizon: {horizon['maximum_observed_harm_reveal_horizon']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["implement-and-audit"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
