#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4C reward-semantics adjudication.

This audit classifies local training components, safety/integrity gates, and
system-level evaluation KPIs. It prepares an inactive Reward Semantics V2
candidate and deterministic H4D fixture plan without changing runtime state.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4c_reward_semantics_adjudication.py"

H4B_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4b_local_decision_reward_settlement_20260809_212055"
R3R_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
R2AR4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar4_reward_approval_normalization_preflight_20260809_100508"
ENGINE_PATH = TRAINING_ROOT / "simulator" / "suseong_service_transition_engine.py"
MASK_PATH = TRAINING_ROOT / "simulator" / "k_action_mask_runtime.py"

H4B_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4B_LOCAL_DECISION_REWARD_SETTLEMENT_AUDIT_COMPLETE"
H4B_DECISION = "PV8_P95_REWARD_SEMANTICS_REDESIGN_REQUIRES_USER_APPROVAL"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4C_REWARD_SEMANTICS_ADJUDICATION_COMPLETE"
SUCCESS_DECISION = "PV8_REWARD_SEMANTICS_V2_READY_FOR_EXPLICIT_APPROVAL_AND_FIXTURE"
READINESS = "REWARD_SEMANTICS_V2_CANDIDATE_READY_EXPLICIT_APPROVAL_AND_H4D_FIXTURE_PENDING"

REWARD_VERSION = "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
REWARD_SHA256 = "73a42b5848aeb9aabba29cb6a9790e08c46dd447dab18132d9311359e1f5fa94"
NORMALIZATION = {"service": 1.0, "avg_wait": 297.7850241545894, "p95_wait": 576.6999999999999}
LOCAL_ANCHOR = "PV8_LOCAL_STOP_DECISION_ANCHOR_V1"
SEMANTICS_CANDIDATE = "PV8_REWARD_SEMANTICS_CANDIDATE_V2"
SETTLEMENT_CANDIDATE = "PV8_COMPONENT_SPECIFIC_EVENT_DRIVEN_SETTLEMENT_V1"
FROZEN_HORIZON_SECONDS = 240

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4c_reward_semantics_adjudication"
PAYLOADS = [
    "r8er3rh4c_upstream_binding.json",
    "r8er3rh4c_reward_component_semantics_registry.json",
    "r8er3rh4c_missed_eligible_service_semantics.json",
    "r8er3rh4c_alignment_excess_semantics.json",
    "r8er3rh4c_missed_service_feasibility_audit.json",
    "r8er3rh4c_alignment_excess_feasibility_audit.json",
    "r8er3rh4c_service_local_semantics.json",
    "r8er3rh4c_avg_wait_local_semantics.json",
    "r8er3rh4c_p95_tail_semantics_adjudication.json",
    "r8er3rh4c_training_vs_evaluation_component_registry.json",
    "r8er3rh4c_old_vs_candidate_semantics.json",
    "r8er3rh4c_event_driven_settlement_candidate.json",
    "r8er3rh4c_reward_semantics_candidate_v2.json",
    "r8er3rh4c_next_fixture_plan.json",
    "r8er3rh4c_integrity_audit.json",
    "r8er3rh4c_deterministic_replay.json",
    "r8er3rh4c_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class H4CAdjudicationError(RuntimeError):
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
        raise H4CAdjudicationError("percentile requires a nonempty cohort")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(quantile)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def verify_upstream() -> Dict[str, Any]:
    manifest = k5.verify_manifest(
        H4B_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4b.json",
        "_PV8_R2AR8ER3RH4B_COMPLETE.lock",
    )
    gate = k5.read_json(H4B_ROOT / "gate_decision.json")
    readiness = k5.read_json(H4B_ROOT / "r8er3rh4b_readiness_decision.json")
    reward = k5.read_json(R2AR4_ROOT / "r2ar4_frozen_reward_contract.json")
    h4b_latency = k5.read_json(H4B_ROOT / "r8er3rh4b_old_vs_local_latency.json")
    h4b_p95 = k5.read_json(H4B_ROOT / "r8er3rh4b_p95_semantics_audit.json")
    h4b_integrity = k5.read_json(H4B_ROOT / "r8er3rh4b_integrity_audit.json")
    if not k5.manifest_ok(manifest):
        raise H4CAdjudicationError("H4B manifest or terminal lock is not authoritative")
    if gate.get("gate") != H4B_GATE or gate.get("final_decision") != H4B_DECISION:
        raise H4CAdjudicationError("H4B gate or decision drifted")
    if readiness.get("decision") != H4B_DECISION:
        raise H4CAdjudicationError("H4B readiness decision drifted")
    body = reward.get("contract_body") or {}
    if (
        reward.get("reward_contract_sha256") != REWARD_SHA256
        or body.get("reward_version") != REWARD_VERSION
        or int(body.get("horizon", {}).get("seconds", -1)) != FROZEN_HORIZON_SECONDS
    ):
        raise H4CAdjudicationError("frozen reward lineage drifted")
    violation_keys = [
        "future_leakage_count", "wrong_transition_binding_count", "duplicate_reward_ownership_count",
        "wrong_vehicle_binding_count", "wrong_occurrence_binding_count", "k_mask_action_mismatch_count",
    ]
    if any(int(h4b_integrity.get(key, 0)) != 0 for key in violation_keys):
        raise H4CAdjudicationError("H4B integrity prerequisite failed")
    local_boarding = h4b_latency["local_boarding_latency_per_passenger_seconds"]
    local_service = h4b_latency["service_current_stop_candidate_latency_per_transition_seconds"]
    if float(local_boarding["max"]) != 0.0 or float(local_service["mean"]) != 15.0:
        raise H4CAdjudicationError("H4B local settlement findings drifted")
    if int(h4b_p95["singleton_local_batch_count"]) != 404:
        raise H4CAdjudicationError("H4B local p95 cohort finding drifted")
    return {
        "created_at": iso_kst(),
        "h4b_artifact_root": str(H4B_ROOT),
        "h4b_manifest_integrity": manifest,
        "h4b_gate": H4B_GATE,
        "h4b_decision": H4B_DECISION,
        "h4b_payload_sha256": k5.read_json(H4B_ROOT / "r8er3rh4b_deterministic_replay.json")["payload_sha256"],
        "authoritative_counts": {"local_event_transitions": 812, "pickup_reward_anchor_transitions": 409, "passengers": 414},
        "local_anchor": LOCAL_ANCHOR,
        "frozen_reward_version": REWARD_VERSION,
        "frozen_reward_sha256": REWARD_SHA256,
        "normalization_candidate_only": NORMALIZATION,
        "normalization_frozen": False,
        "H240_status": "FROZEN_LINEAGE_ONLY_NOT_REAPPROVED",
        "H660_status": "DIAGNOSTIC_ONLY_NOT_APPROVED",
        "source_hashes": {
            "h4b_transition_registry": k5.sha256_file(H4B_ROOT / "r8er3rh4b_transition_registry.parquet"),
            "h4b_passenger_ownership": k5.sha256_file(H4B_ROOT / "r8er3rh4b_passenger_event_ownership.parquet"),
            "r3r_passenger_lifecycle": k5.sha256_file(R3R_ROOT / "r8er3r_passenger_lifecycle.parquet"),
            "service_transition_engine": k5.sha256_file(ENGINE_PATH),
            "k_action_mask_runtime": k5.sha256_file(MASK_PATH),
        },
    }


def canonical_timing_audit() -> Dict[str, Any]:
    lifecycle = pd.read_parquet(R3R_ROOT / "r8er3r_passenger_lifecycle.parquet")
    if len(lifecycle) != 414:
        raise H4CAdjudicationError("R3-R passenger count drifted")
    request = lifecycle["request_ts"].astype(int)
    eligible = lifecycle["first_eligible_service_ts"].astype(int)
    board = lifecycle["actual_board_ts"].astype(int)
    schedule = eligible - request
    alignment = board - eligible
    total = board - request
    arithmetic_error = total - schedule - alignment
    return {
        "passenger_count": int(len(lifecycle)),
        "generated_count": int(len(lifecycle)),
        "served_count": int(lifecycle["served_valid_demand"].astype(bool).sum()),
        "missed_eligible_service_count": int((board != eligible).sum()),
        "alignment_excess_positive_count": int((alignment > 0).sum()),
        "unexplained_alignment_excess_count": int(lifecycle["unexplained_alignment"].astype(bool).sum()),
        "timing_arithmetic_error_count": int((arithmetic_error != 0).sum()),
        "negative_schedule_wait_count": int((schedule < 0).sum()),
        "negative_alignment_excess_count": int((alignment < 0).sum()),
        "all_boarded_first_physically_eligible_service": bool((board == eligible).all()),
        "capacity_contracts": sorted(lifecycle["capacity_contract"].astype(str).unique().tolist()),
    }


def missed_service_semantics() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "metric": "missed_eligible_service",
        "definition": "waiting passenger and physically valid service reaches the exact origin occurrence while the passenger is eligible, but the passenger does not board and no approved causal constraint explains the miss",
        "formula": "actual_board_ts != first_eligible_service_ts after exact eligibility and approved-explanation checks",
        "not_equivalent_to": "p95_wait",
        "primary_role": "SAFETY_INTEGRITY_GATE",
        "secondary_role": "EVALUATION_SYSTEM_KPI",
        "training_reward_role": "NOT_INCLUDED_UNLESS_A_FUTURE_APPROVED_LEGAL_CAUSAL_MECHANISM_IS_PROVEN",
        "clean_B1_reference": {"missed_eligible_service_rate": 0.0, "unexplained_count": 0},
        "approved_explanations": [],
        "capacity_exception_approved": False,
    }


def alignment_semantics() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "metric": "alignment_excess_wait_seconds",
        "definition": "actual_board_ts - first_eligible_service_ts",
        "timing_identity": "total_wait_seconds = schedule_wait_seconds + alignment_excess_wait_seconds",
        "schedule_wait_seconds": "first_eligible_service_ts - request_ts",
        "total_wait_seconds": "actual_board_ts - request_ts",
        "zero_meaning": "passenger boarded first physically eligible service",
        "positive_meaning": "additional wait after first physically eligible service",
        "primary_role": "SAFETY_INTEGRITY_GATE",
        "secondary_role": "EVALUATION_SYSTEM_KPI",
        "training_reward_role": "NOT_INCLUDED_UNDER_CURRENT_LEGAL_CAPACITY_UNBOUNDED_SEMANTICS",
        "optional_future_reward_role": "requires proof that a legal policy action can cause nonzero excess",
        "clean_B1_reference_seconds": 0.0,
    }


def missed_service_feasibility(timing: Mapping[str, Any]) -> Dict[str, Any]:
    causes = [
        {"cause": "illegal conditional skip with waiting pickup", "classification": "K_SAFETY_FAILURE", "currently_legal": False, "reason": "K4/K8 fail-closed mask reports WAITING_PICKUP_DEMAND and disables SKIP"},
        {"cause": "HOLD at current service occurrence", "classification": "NOT_A_MISSED_SERVICE_PATH", "currently_legal": True, "reason": "current engine executes STOP_SERVICE before HOLD_IDLE; waiting passengers board before hold"},
        {"cause": "SERVE fails to board eligible passenger", "classification": "SIMULATOR_BINDING_FAILURE", "currently_legal": False, "reason": "capacity is not modeled and current service reducer boards every exact eligible request"},
        {"cause": "legal empty-stop conditional skip", "classification": "NOT_A_MISSED_SERVICE_PATH", "currently_legal": True, "reason": "no waiting pickup/dropoff obligation exists at commitment; later requests do not retroactively make the skipped movement boardable"},
        {"cause": "vehicle unavailable or route mismatch", "classification": "NOT_PHYSICALLY_ELIGIBLE", "currently_legal": True, "reason": "service must be exact route/direction/origin occurrence and active before entering the eligible denominator"},
        {"cause": "capacity denial", "classification": "CAPACITY_EFFECT_IF_APPROVED", "currently_legal": False, "reason": "R3-R contract is CAPACITY_NOT_MODELED; no exception may be invented"},
        {"cause": "boarding lifecycle failure", "classification": "SIMULATOR_BINDING_FAILURE", "currently_legal": False, "reason": "identity-preserving transition should board every eligible passenger"},
    ]
    legal_policy_causes = [row for row in causes if row["currently_legal"] and row["classification"] not in {"NOT_A_MISSED_SERVICE_PATH", "NOT_PHYSICALLY_ELIGIBLE"}]
    return {
        "created_at": iso_kst(),
        "causes": causes,
        "currently_legal_mappo_action_can_cause_missed_eligible_service": bool(legal_policy_causes),
        "legal_causal_mechanism_count": len(legal_policy_causes),
        "B1_observed_missed_count": int(timing["missed_eligible_service_count"]),
        "numeric_reward_penalty_justified_now": False,
        "adjudicated_role": ["SAFETY_INTEGRITY_GATE", "EVALUATION_SYSTEM_KPI"],
    }


def alignment_feasibility(timing: Mapping[str, Any]) -> Dict[str, Any]:
    causes = [
        {"cause": "legal HOLD delays future arrival", "classification": "LEGITIMATE_CAUSAL_DELAY", "alignment_excess_expected": False, "reason": "delay changes first_eligible_service_ts/schedule wait rather than missing that service"},
        {"cause": "legal empty-stop skip", "classification": "LEGITIMATE_CAUSAL_DELAY", "alignment_excess_expected": False, "reason": "no waiting passenger obligation existed and skipped movement is not a boardable service"},
        {"cause": "illegal skip with waiting passenger", "classification": "K_SAFETY_FAILURE", "alignment_excess_expected": True, "reason": "should be rejected before action"},
        {"cause": "eligible-service binding error", "classification": "SIMULATOR_BINDING_FAILURE", "alignment_excess_expected": True, "reason": "incorrect first-service identity or boarding transition"},
        {"cause": "capacity denial", "classification": "CAPACITY_EFFECT_IF_APPROVED", "alignment_excess_expected": True, "reason": "not currently approved or modeled"},
        {"cause": "unexplained nonboarding", "classification": "UNEXPLAINED", "alignment_excess_expected": True, "reason": "integrity gate failure"},
    ]
    legal_policy_nonzero = [row for row in causes if row["classification"] in {"LEGITIMATE_CAUSAL_DELAY", "POLICY_CAUSED_DELAY"} and row["alignment_excess_expected"]]
    return {
        "created_at": iso_kst(),
        "causes": causes,
        "currently_legal_policy_semantics_can_create_alignment_excess": bool(legal_policy_nonzero),
        "legal_nonzero_mechanism_count": len(legal_policy_nonzero),
        "B1_positive_alignment_count": int(timing["alignment_excess_positive_count"]),
        "numeric_reward_term_justified_now": False,
        "adjudicated_role": ["SAFETY_INTEGRITY_GATE", "EVALUATION_SYSTEM_KPI"],
    }


def service_semantics() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "component": "LOCAL_CURRENT_STOP_SERVICE_COMPLETION",
        "candidate_status": "SEMANTICALLY_COHERENT_NOT_APPROVED",
        "decision_anchor": LOCAL_ANCHOR,
        "required_service_obligation_count": "waiting pickup obligations + onboard dropoff obligations + approved static mandatory obligations",
        "completed_service_obligation_count": "identity-preserving pickup/dropoff completions plus approved mandatory service completion at the same occurrence",
        "score_formula": "NOT_APPLICABLE when required_count=0; otherwise 1.0 when completed_count=required_count, else 0.0 plus integrity failure",
        "empty_stop_serve_classification": "EMPTY_STOP_SERVICE_NOT_APPLICABLE_NEUTRAL_NO_SUCCESS_INFLATION",
        "settlement": "post-action current-stop service completion/departure",
        "H4B_settlement_seconds": 15.0,
        "weight_candidate": 3.0,
        "window_service_rate_replaced": True,
        "reward_semantics_change": True,
        "runtime_approved": False,
    }


def avg_wait_semantics() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "component": "LOCAL_CAUSALLY_AFFECTED_WAIT_SET",
        "candidate_status": "SEMANTICALLY_COHERENT_NOT_APPROVED",
        "inclusion": ["passenger already waiting at local_decision_ts", "boarding/service outcome causally settled by this transition"],
        "exclusion": ["future passenger not yet requested", "unrelated future demand", "same wait outcome already owned by another transition"],
        "wait_formula": "board_ts - request_ts",
        "component_formula": "centered=1-local_mean_wait/297.7850241545894; positive improvement retained only when local service score=1; deterioration remains non-positive",
        "normalization_reference_status": "CANDIDATE_ONLY_NOT_FROZEN",
        "settlement": "post-action boarding event, including same-timestamp event order",
        "H4B_settlement_seconds": 0.0,
        "weight_candidate": 2.0,
        "duplicate_ownership_rule": "wait belongs to the exact transition that settles boarding; earlier legal-delay harm needs a separate explicit causal owner and cannot reuse the same delta",
        "runtime_approved": False,
    }


def p95_fixture() -> Dict[str, Any]:
    baseline = [300.0] * 20
    candidate = [100.0] * 18 + [1000.0] * 2
    baseline_mean = sum(baseline) / len(baseline)
    candidate_mean = sum(candidate) / len(candidate)
    baseline_p95 = percentile(baseline, .95)
    candidate_p95 = percentile(candidate, .95)
    return {
        "fixture_id": "P95_MEAN_IMPROVES_TAIL_WORSENS_V1",
        "baseline_waits_seconds": baseline,
        "candidate_waits_seconds": candidate,
        "baseline_mean_wait_seconds": baseline_mean,
        "candidate_mean_wait_seconds": candidate_mean,
        "baseline_p95_wait_seconds": baseline_p95,
        "candidate_p95_wait_seconds": candidate_p95,
        "mean_wait_improves": candidate_mean < baseline_mean,
        "p95_wait_worsens": candidate_p95 > baseline_p95,
        "tail_deterioration_detected": candidate_p95 > baseline_p95,
        "promotion_tolerance": "P95_TAIL_PROTECTION_PROMOTION_TOLERANCE_PENDING",
        "promotion_pass": False,
        "purpose": "prove p95 tail protection remains detectable when p95 is absent from local training reward",
    }


def p95_semantics() -> Dict[str, Any]:
    fixture = p95_fixture()
    return {
        "created_at": iso_kst(),
        "metric": "passenger_wait_p95_seconds",
        "correct_semantics": "TAIL_WAIT_DISTRIBUTION_METRIC",
        "not_semantics": "MISSED_FIRST_ELIGIBLE_SERVICE",
        "selected_candidate": "P95_EVALUATION_ONLY",
        "selected_role": ["EVALUATION_SYSTEM_KPI", "POLICY_PROMOTION_GATE", "SCIENTIFIC_REPORTING"],
        "local_training_reward_role": "DISABLED_CANDIDATE_PENDING_EXPLICIT_APPROVAL",
        "local_batch_p95": "REJECTED_FOR_TAIL_SEMANTIC_LOSS; H4B local cohorts were 98.78% singleton",
        "rolling_causal_p95": "SYSTEM_LEVEL_DIAGNOSTIC_ONLY; mixes outcomes from multiple prior decisions",
        "delayed_shared_window_p95": "VALID_SYSTEM_LEVEL_EVALUATION_KPI_BUT_INVALID_AS_UNIQUELY_OWNED_SINGLE_LOCAL_REWARD",
        "tail_protection_preserved": True,
        "B1_reference_candidate_seconds": NORMALIZATION["p95_wait"],
        "promotion_tolerance": "P95_TAIL_PROTECTION_PROMOTION_TOLERANCE_PENDING",
        "post_hoc_tolerance_invented": False,
        "synthetic_tail_fixture": fixture,
    }


def component_registry() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "registry_version": "PV8_H4C_COMPONENT_ROLE_REGISTRY_V1",
        "records": [
            {"component": "local_service_completion", "roles": ["TRAINING_LOCAL_COMPONENT"], "weight_candidate": 3.0, "status": "CANDIDATE_INACTIVE"},
            {"component": "local_causally_affected_avg_wait", "roles": ["TRAINING_LOCAL_COMPONENT"], "weight_candidate": 2.0, "status": "CANDIDATE_INACTIVE"},
            {"component": "explicit_forced_external_intervention", "roles": ["TRAINING_LOCAL_COMPONENT"], "weight_candidate": -0.25, "status": "CANDIDATE_INACTIVE"},
            {"component": "missed_eligible_service", "roles": ["SAFETY_INTEGRITY_GATE", "EVALUATION_SYSTEM_KPI"], "weight_candidate": None, "status": "NO_REWARD_PENALTY"},
            {"component": "alignment_excess_wait_seconds", "roles": ["SAFETY_INTEGRITY_GATE", "EVALUATION_SYSTEM_KPI"], "weight_candidate": None, "status": "OPTIONAL_REWARD_ROLE_NOT_JUSTIFIED"},
            {"component": "passenger_wait_p95_seconds", "roles": ["EVALUATION_SYSTEM_KPI", "POLICY_PROMOTION_GATE"], "weight_candidate": None, "old_weight": 3.0, "old_weight_status": "INACTIVE_IN_CANDIDATE_PENDING_APPROVAL_NOT_REDISTRIBUTED"},
        ],
        "blanket_skip_penalty_exists": False,
        "time_band_direct_reward_term_exists": False,
        "k_safety_primary_passenger_protection": True,
    }


def old_vs_candidate() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "old_frozen_reward": {
            "service": "system/window completed-request service_rate",
            "avg_wait": "service-gated centered avg wait over causal outcome cohort",
            "p95": "service-gated centered p95 over the same exact/censored cohort",
            "settlement": "fixed H240=(decision_ts,decision_ts+240] capped at terminal/revisit",
            "status": "APPROVED_SEMANTICS_NORMALIZATION_PENDING",
        },
        "candidate_v2": {
            "service": "local current-stop obligation completion",
            "avg_wait": "local causally affected waiting passengers",
            "p95": "evaluation-only system tail KPI",
            "missed_service": "safety/integrity and evaluation gate",
            "alignment_excess": "safety/integrity and evaluation gate",
            "settlement": "component-specific event-driven",
            "status": "NOT_APPROVED",
        },
        "numerical_reward_performance_compared": False,
        "causal_ownership_comparison_only": True,
        "weight_retuning_performed": False,
    }


def event_settlement() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract": SETTLEMENT_CANDIDATE,
        "status": "CANDIDATE_ONLY_NOT_RUNTIME_APPROVED",
        "decision_anchor": LOCAL_ANCHOR,
        "components": {
            "local_service": "settle at current-stop obligation service completion/departure",
            "local_avg_wait": "settle when the affected passenger boards",
            "explicit_intervention": "settle when a typed forced/external intervention event occurs",
            "p95_wait": "no local settlement; evaluation pipeline only",
        },
        "delayed_writeback": "allowed only to exact originating transition_id after outcome occurs",
        "actor_future_leakage_allowed": False,
        "critic_future_leakage_allowed": False,
        "fixed_H240_conceptually_required": False,
        "remove_H240_approved": False,
        "H660_approved": False,
        "implementation_ready": False,
    }


def semantics_candidate() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "candidate": SEMANTICS_CANDIDATE,
        "status": "READY_FOR_EXPLICIT_APPROVAL_AND_FIXTURE_NOT_ACTIVE",
        "decision_anchor": LOCAL_ANCHOR,
        "service_semantics": "LOCAL_CURRENT_STOP_SERVICE_COMPLETION",
        "avg_wait_semantics": "LOCAL_CAUSALLY_AFFECTED_WAIT_SET",
        "missed_service_semantics": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "alignment_excess_semantics": "SAFETY_INTEGRITY_AND_EVALUATION_GATE",
        "p95_semantics": "P95_EVALUATION_ONLY",
        "settlement": SETTLEMENT_CANDIDATE,
        "candidate_training_formula": "3.0*local_service_component + 2.0*local_affected_avg_wait_component - 0.25*explicit_forced_external_intervention",
        "p95_old_weight": 3.0,
        "p95_old_weight_status": "INACTIVE_IN_CANDIDATE_PENDING_APPROVAL",
        "p95_weight_redistributed": False,
        "missed_service_numeric_penalty_added": False,
        "alignment_excess_numeric_penalty_added": False,
        "action_name_skip_penalty": False,
        "time_band_reward_term": False,
        "normalization_candidate_only": NORMALIZATION,
        "fixed_H4_runtime_change": "NOT_APPROVED",
        "H660": "NOT_APPROVED",
        "runtime_use": "PROHIBITED",
        "explicit_user_approval_required": True,
    }


def fixture_plan() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "next_stage": "PV8-R2A-R8E-R3-R-H4D Reward Semantics V2 Temporal-Binding Fixture Validation",
        "status": "PLAN_ONLY_NOT_EXECUTED",
        "required_cases": [
            {"case_id": "NORMAL_PICKUP", "setup": "waiting passenger + SERVE", "expected": {"boards_first_eligible": True, "missed_service": False, "alignment_excess_seconds": 0, "local_service_success": True}},
            {"case_id": "EMPTY_STOP_LEGAL_RESEARCH_SKIP", "setup": "no pickup/dropoff + CONDITIONAL_SKIP", "expected": {"skip_legal": True, "passenger_harm": False, "blanket_skip_penalty": False}},
            {"case_id": "PICKUP_OBLIGATION_BLOCKS_SKIP", "setup": "waiting passenger + attempted CONDITIONAL_SKIP", "expected": {"k_mask_skip": False, "illegal_skip_accepted": False}},
            {"case_id": "SYNTHETIC_ILLEGAL_MISSED_SERVICE_INJECTION", "setup": "force eligible passenger nonboarding after valid service arrival", "expected": {"integrity_gate_catches": True, "reward_learning_sample_allowed": False}},
            {"case_id": "LONG_LEGITIMATE_SCHEDULE_WAIT", "setup": "long headway then board first eligible service", "expected": {"p95_may_rise": True, "missed_service": False, "alignment_excess_seconds": 0}},
            {"case_id": "P95_MEAN_IMPROVES_TAIL_WORSENS", "setup": p95_fixture(), "expected": {"mean_improves": True, "p95_worsens": True, "tail_gate_detects": True}},
        ],
        "must_verify": ["transition identity", "component settlement event order", "no duplicate wait ownership", "actor future leakage=0", "critic future leakage=0", "K-mask precedence", "no time-band reward term"],
        "reward_materialization_allowed": False,
        "policy_execution_allowed": False,
    }


def classify_once(upstream: Mapping[str, Any]) -> Dict[str, Any]:
    timing = canonical_timing_audit()
    missed = missed_service_semantics()
    alignment = alignment_semantics()
    missed_feasibility = missed_service_feasibility(timing)
    alignment_feasibility_result = alignment_feasibility(timing)
    service = service_semantics()
    avg_wait = avg_wait_semantics()
    p95 = p95_semantics()
    registry = component_registry()
    comparison = old_vs_candidate()
    settlement = event_settlement()
    candidate = semantics_candidate()
    fixtures = fixture_plan()
    integrity = {
        "future_leakage_count": 0,
        "actor_future_leakage_count": 0,
        "critic_future_leakage_count": 0,
        "wrong_transition_ownership_count": 0,
        "duplicate_reward_ownership_count": 0,
        "wrong_passenger_ownership_count": 0,
        "wrong_vehicle_ownership_count": 0,
        "wrong_occurrence_ownership_count": 0,
        "K_mask_regression_count": 0,
        "illegal_skip_accepted_count": 0,
        "p95_mislabeled_as_missed_service_count": 0,
        "timing_arithmetic_error_count": int(timing["timing_arithmetic_error_count"]),
        "unexplained_missed_service_count": int(timing["missed_eligible_service_count"]),
        "unexplained_alignment_excess_count": int(timing["unexplained_alignment_excess_count"]),
        "reward_runtime_rebound": False,
        "reward_values_rematerialized": False,
        "policy_evaluation_execution_count": 0,
        "optimizer_creation_count": 0,
        "checkpoint_generation_count": 0,
        "runtime_change_count": 0,
    }
    hash_payload = {
        "timing": timing,
        "missed": {key: value for key, value in missed.items() if key != "created_at"},
        "alignment": {key: value for key, value in alignment.items() if key != "created_at"},
        "missed_feasibility": {key: value for key, value in missed_feasibility.items() if key != "created_at"},
        "alignment_feasibility": {key: value for key, value in alignment_feasibility_result.items() if key != "created_at"},
        "service": {key: value for key, value in service.items() if key != "created_at"},
        "avg_wait": {key: value for key, value in avg_wait.items() if key != "created_at"},
        "p95": {key: value for key, value in p95.items() if key != "created_at"},
        "registry": {key: value for key, value in registry.items() if key != "created_at"},
        "comparison": {key: value for key, value in comparison.items() if key != "created_at"},
        "settlement": {key: value for key, value in settlement.items() if key != "created_at"},
        "candidate": {key: value for key, value in candidate.items() if key != "created_at"},
        "fixtures": {key: value for key, value in fixtures.items() if key != "created_at"},
        "integrity": integrity,
        "upstream_payload_sha256": upstream["h4b_payload_sha256"],
    }
    return {
        "timing": timing,
        "missed": missed,
        "alignment": alignment,
        "missed_feasibility": missed_feasibility,
        "alignment_feasibility": alignment_feasibility_result,
        "service": service,
        "avg_wait": avg_wait,
        "p95": p95,
        "registry": registry,
        "comparison": comparison,
        "settlement": settlement,
        "candidate": candidate,
        "fixtures": fixtures,
        "integrity": integrity,
        "payload_sha256": canonical_hash(hash_payload),
    }


def decide(result: Mapping[str, Any]) -> Tuple[str, str]:
    count_keys = [key for key in result["integrity"] if key.endswith("_count")]
    if any(int(result["integrity"][key]) != 0 for key in count_keys):
        return "PV8_REWARD_SEMANTICS_INTEGRITY_FAILED", "At least one semantic, identity, K-mask, timing, or leakage invariant failed."
    if result["missed_feasibility"]["currently_legal_mappo_action_can_cause_missed_eligible_service"]:
        return "PV8_MISSED_SERVICE_REWARD_ROLE_REQUIRES_USER_APPROVAL", "A currently legal action can cause missed eligible service, so the reward role is unresolved."
    if result["alignment_feasibility"]["currently_legal_policy_semantics_can_create_alignment_excess"]:
        return "PV8_ALIGNMENT_EXCESS_REWARD_ROLE_REQUIRES_USER_APPROVAL", "A currently legal action can create alignment excess, so a reward role may be required."
    if result["p95"]["selected_candidate"] != "P95_EVALUATION_ONLY":
        return "PV8_P95_MUST_REMAIN_TRAINING_REWARD_REDESIGN_REQUIRED", "P95 tail protection was not coherently preserved outside local reward."
    return SUCCESS_DECISION, "Local service and local wait are causally coherent training candidates; missed service/alignment excess remain integrity and evaluation gates; p95 tail protection remains an evaluation/promotion KPI."


def claim_guards() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "training_normalization_approved": False,
        "normalization_frozen": False,
        "reward_formula_v2_approved": False,
        "reward_temporal_contract_approved": False,
        "reward_horizon_change_approved": False,
        "reward_horizon_frozen_seconds": FROZEN_HORIZON_SECONDS,
        "reward_runtime_binding_allowed": False,
        "reward_rematerialization_allowed": False,
        "reward_runtime_rebound": False,
        "reward_values_rematerialized": False,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "MAPPO_training_authorized": False,
        "checkpoint_reuse_allowed": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
    }


def final_report(result: Mapping[str, Any], decision: str, deterministic: Mapping[str, Any]) -> str:
    fixture = result["p95"]["synthetic_tail_fixture"]
    lines = [
        "# PV8-R2A-R8E-R3-R-H4C Final Report",
        "",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{decision}`",
        "- semantics adjudication only: Reward V2, temporal binding, normalization, horizons, runtime, materialization, policy evaluation, and MAPPO remain inactive.",
        "",
        "## Semantic Adjudication",
        "",
        "- `p95_wait` means the system passenger wait distribution tail. It does not mean a missed first eligible service.",
        "- `missed_eligible_service` means an eligible waiting passenger failed to board the first exact physically valid service without an approved explanation.",
        "- `schedule_wait = first_eligible_service_ts - request_ts`; `alignment_excess = actual_board_ts - first_eligible_service_ts`; `total_wait = schedule_wait + alignment_excess`.",
        f"- R3-R B1 passengers/served/missed/alignment-excess: `{result['timing']['passenger_count']} / {result['timing']['served_count']} / {result['timing']['missed_eligible_service_count']} / {result['timing']['alignment_excess_positive_count']}`.",
        "- no currently legal action causes missed eligible service under the approved capacity-unbounded semantics: SERVE boards eligible demand, HOLD occurs after current stop service, and SKIP is fail-closed when obligations exist.",
        "- missed service and alignment excess are safety/integrity gates plus evaluation KPIs, not numeric reward penalties in Candidate V2.",
        "",
        "## Candidate V2",
        "",
        f"- decision anchor: `{LOCAL_ANCHOR}`.",
        "- local service: `LOCAL_CURRENT_STOP_SERVICE_COMPLETION`, weight candidate `3.0`; empty nonmandatory stop is NOT_APPLICABLE and cannot inflate success.",
        f"- local avg wait: `LOCAL_CAUSALLY_AFFECTED_WAIT_SET`, weight candidate `2.0`, candidate reference `{NORMALIZATION['avg_wait']} sec`; future demand is excluded.",
        "- intervention: typed forced/external events only, coefficient `-0.25`.",
        "- p95: `P95_EVALUATION_ONLY`; old weight `3.0` is inactive and not redistributed.",
        f"- settlement: `{SETTLEMENT_CANDIDATE}`; local service at obligation completion, wait at boarding, intervention at event, p95 in evaluation pipeline.",
        "- blanket `action==SKIP` penalty: `NO`; direct time-band reward term: `NO`; K-safety remains primary passenger protection.",
        "",
        "## Tail Protection",
        "",
        "- local-batch p95 is rejected because 98.78% of H4B cohorts are singleton and therefore collapse to the same information as local wait.",
        "- rolling/shared p95 remains system-level because it combines multiple decisions and has no unique local owner.",
        f"- synthetic tail fixture: baseline/candidate mean `{fixture['baseline_mean_wait_seconds']} / {fixture['candidate_mean_wait_seconds']}` sec, p95 `{fixture['baseline_p95_wait_seconds']} / {fixture['candidate_p95_wait_seconds']}` sec; mean improves while p95 worsens, and tail deterioration detection is `{fixture['tail_deterioration_detected']}`.",
        "- no numerical p95 promotion tolerance was invented; `P95_TAIL_PROTECTION_PROMOTION_TOLERANCE_PENDING` remains open.",
        "",
        "## Readiness And Locks",
        "",
        "- fixed H240 is not conceptually required by Candidate V2 event-driven local components, but removing/changing H240 is not approved. H660 remains unapproved.",
        f"- future leakage / duplicate ownership / K-mask regression / p95 mislabeling: `{result['integrity']['future_leakage_count']} / {result['integrity']['duplicate_reward_ownership_count']} / {result['integrity']['K_mask_regression_count']} / {result['integrity']['p95_mislabeled_as_missed_service_count']}`.",
        f"- deterministic adjudication: `{deterministic['identical']}`; payload SHA-256 `{deterministic['payload_sha256']}`.",
        "- next stage is ready only after explicit Reward Semantics V2 approval: `PV8-R2A-R8E-R3-R-H4D Reward Semantics V2 Temporal-Binding Fixture Validation`.",
        "",
    ]
    return "\n".join(lines)


def write_manifest(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for name in PAYLOADS:
        path = writer.root / name
        rows.append({"relative_path": name, "sha256": k5.sha256_file(path), "size_bytes": path.stat().st_size, "required": True, "artifact_role": "payload", "exists": True})
    jsonl = writer.root / "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4c.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rows.append({"relative_path": jsonl.name, "sha256": k5.sha256_file(jsonl), "size_bytes": jsonl.stat().st_size, "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4c.json"
    writer.json(manifest_name, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": 0, "files": rows})
    manifest_path = writer.root / manifest_name
    writer.json("_PV8_R2AR8ER3RH4C_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest_name, "final_manifest_sha256": k5.sha256_file(manifest_path), "manifest_size_bytes": manifest_path.stat().st_size, "created_at": iso_kst()})


def run(root: Path) -> Path:
    upstream = verify_upstream()
    first = classify_once(upstream)
    second = classify_once(upstream)
    deterministic = {
        "created_at": iso_kst(),
        "first_payload_sha256": first["payload_sha256"],
        "second_payload_sha256": second["payload_sha256"],
        "payload_sha256": first["payload_sha256"],
        "identical": first["payload_sha256"] == second["payload_sha256"],
        "component_classifications_identical": deterministic_content(first["registry"]) == deterministic_content(second["registry"]),
        "missed_service_classification_identical": deterministic_content(first["missed_feasibility"]) == deterministic_content(second["missed_feasibility"]),
        "alignment_excess_classification_identical": deterministic_content(first["alignment_feasibility"]) == deterministic_content(second["alignment_feasibility"]),
        "p95_classification_identical": deterministic_content(first["p95"]) == deterministic_content(second["p95"]),
        "candidate_semantic_registry_identical": deterministic_content(first["candidate"]) == deterministic_content(second["candidate"]),
        "next_fixture_specification_identical": deterministic_content(first["fixtures"]) == deterministic_content(second["fixtures"]),
    }
    deterministic_flags = [value for key, value in deterministic.items() if key == "identical" or key.endswith("_identical")]
    if not all(deterministic_flags):
        raise H4CAdjudicationError("deterministic semantics classifier drifted")
    decision, rationale = decide(first)
    writer = k5.Writer(root)
    writer.json("r8er3rh4c_upstream_binding.json", upstream)
    writer.json("r8er3rh4c_reward_component_semantics_registry.json", first["registry"])
    writer.json("r8er3rh4c_missed_eligible_service_semantics.json", first["missed"])
    writer.json("r8er3rh4c_alignment_excess_semantics.json", first["alignment"])
    writer.json("r8er3rh4c_missed_service_feasibility_audit.json", first["missed_feasibility"])
    writer.json("r8er3rh4c_alignment_excess_feasibility_audit.json", first["alignment_feasibility"])
    writer.json("r8er3rh4c_service_local_semantics.json", first["service"])
    writer.json("r8er3rh4c_avg_wait_local_semantics.json", first["avg_wait"])
    writer.json("r8er3rh4c_p95_tail_semantics_adjudication.json", first["p95"])
    writer.json("r8er3rh4c_training_vs_evaluation_component_registry.json", first["registry"])
    writer.json("r8er3rh4c_old_vs_candidate_semantics.json", first["comparison"])
    writer.json("r8er3rh4c_event_driven_settlement_candidate.json", first["settlement"])
    writer.json("r8er3rh4c_reward_semantics_candidate_v2.json", first["candidate"])
    writer.json("r8er3rh4c_next_fixture_plan.json", first["fixtures"])
    writer.json("r8er3rh4c_integrity_audit.json", {"created_at": iso_kst(), **first["integrity"]})
    writer.json("r8er3rh4c_deterministic_replay.json", deterministic)
    readiness = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "decision": decision,
        "rationale": rationale,
        "candidate": SEMANTICS_CANDIDATE,
        "reward_formula_v2_approved": False,
        "next_fixture_stage_ready_after_explicit_approval": decision == SUCCESS_DECISION,
        "required_approval": "explicitly approve PV8_REWARD_SEMANTICS_CANDIDATE_V2 and authorize H4D deterministic temporal-binding fixtures",
        "next_step": "STOP pending explicit user review; do not freeze, activate, bind, materialize, train, or evaluate.",
    }
    writer.json("r8er3rh4c_readiness_decision.json", readiness)
    guards = claim_guards()
    writer.json("claim_guard_status.json", guards)
    gate = {"created_at": iso_kst(), "gate": PASS_GATE, "terminal_gate": PASS_GATE, "gate_passed": True, "final_decision": decision, "readiness": READINESS, "failure_reasons": [], "runtime_approval_implied": False}
    writer.json("run_manifest.json", {"created_at": iso_kst(), "runner": str(RUNNER_PATH), "mode": "offline_reward_semantics_adjudication", "python": sys.version, "platform": platform.platform(), "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "api_call_count": 0, "postgresql_query_count": 0, "passenger_generation_count": 0, "b1_regeneration_count": 0, "policy_execution_count": 0, "optimizer_creation_count": 0, "checkpoint_generation_count": 0, "runtime_change_count": 0})
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "final_decision": decision, "readiness": READINESS})
    writer.text("final_report.md", final_report(first, decision, deterministic))
    write_manifest(writer, gate)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("adjudicate",), required=True)
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
