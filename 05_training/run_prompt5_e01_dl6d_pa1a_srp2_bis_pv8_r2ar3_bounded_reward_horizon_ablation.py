#!/usr/bin/env python3
"""PV8-R2A-R3 bounded reward/horizon ablation and service-alignment gate."""

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
import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar2_causal_reward_infrastructure as r2ar2


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar3_bounded_reward_horizon_ablation.py"
R2AR2_RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar2_causal_reward_infrastructure.py"
COLLECTOR_SOURCE = TRAINING_ROOT / "simulator" / "pv8_reward_outcome_collector.py"

K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"
K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"
R2A_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2a_reward_authority_promotion_audit_20260808_145209"
R2AR1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar1_reward_contract_repair_audit_20260808_150949"
R2AR2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar2_causal_reward_infrastructure_20260808_160345"

UPSTREAMS = {
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
    "PV8-K9": (K9_ROOT, "artifact_manifest_srp2_bis_pv8_k9.json", "_PV8_K9_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED"),
    "PV8-R2A": (R2A_ROOT, "artifact_manifest_srp2_bis_pv8_r2a.json", "_PV8_R2A_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2A_REWARD_AUTHORITY_AND_PROMOTION_AUDIT_COMPLETE"),
    "PV8-R2A-R1": (R2AR1_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar1.json", "_PV8_R2AR1_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR1_REWARD_CONTRACT_REPAIR_AUDIT_COMPLETE"),
    "PV8-R2A-R2": (R2AR2_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar2.json", "_PV8_R2AR2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR2_CAUSAL_REWARD_INFRASTRUCTURE_COMPLETE"),
}

HISTORICAL_WEIGHTS = {
    "service": 3.0,
    "avg_wait": 2.0,
    "p95_wait": 3.0,
    "on_time": 1.0,
    "bunching": 1.5,
    "headway_cv": 1.0,
    "energy_per_passenger": 1.0,
    "fleet_reduction": 0.5,
    "intervention": 0.25,
    "constraint": 10.0,
}
TOLERANCE = 1e-9
ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar3_bounded_reward_horizon_ablation"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR3_BOUNDED_REWARD_HORIZON_ABLATION_COMPLETE"
DECISION = "PV8_REWARD_CANDIDATE_READY_FOR_EXPLICIT_APPROVAL"
READINESS = "SRP2_BIS_PV8_R2AR3_COMPLETE_CANDIDATE_AND_HCOVER_READY_FOR_EXPLICIT_APPROVAL"

PAYLOADS = [
    "r2ar3_candidate_registry.json",
    "r2ar3_horizon_contract.json",
    "r2ar3_fixture_reward_results.parquet",
    "r2ar3_paired_reward_margins.parquet",
    "r2ar3_component_contributions.parquet",
    "r2ar3_horizon_sensitivity.json",
    "r2ar3_service_alignment_audit.json",
    "r2ar3_candidate_selection_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]

RESULT_COLUMNS = [
    "candidate_id", "candidate_executable", "horizon_id", "requested_horizon", "effective_horizon",
    "fixture_id", "branch", "fixture_harmful", "candidate_skip_allowed", "evaluation_status",
    "reward_total", "service_rate", "avg_wait_seconds", "p95_wait_seconds", "intervention_rate",
    "event_trace_hash", "exogenous_schedule_hash", "b1_scope", "reward_contract_approved",
]
MARGIN_COLUMNS = [
    "candidate_id", "candidate_executable", "horizon_id", "requested_horizon", "effective_horizon",
    "fixture_id", "fixture_harmful", "candidate_skip_allowed", "comparison_status",
    "reference_reward", "candidate_skip_reward", "skip_minus_reference_margin", "tolerance",
    "harm_reveal_transition", "horizon_covers_reveal", "primary_safety_criterion_passed",
    "beneficial_preservation_passed", "harmful_skip_positive_absolute_reward", "exogenous_schedule_hash",
]
CONTRIBUTION_COLUMNS = [
    "candidate_id", "horizon_id", "fixture_id", "branch", "component", "raw_value",
    "reference_value", "normalized_value", "weight", "weighted_contribution", "normalization_mode",
]


class R2AR3Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(k5.json_clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_upstreams() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed != expected_gate or not k5.manifest_ok(checks):
            raise R2AR3Error(f"{label} integrity failure: gate={observed}, checks={checks}")
        out[label] = {"artifact_root": str(root), "gate": observed, "manifest_integrity": checks}
    r2_run = k5.read_json(R2AR2_ROOT / "run_manifest.json")
    source_checks = {
        "r2ar2_runner_hash_matches": k5.sha256_file(R2AR2_RUNNER_PATH) == r2_run.get("runner_sha256"),
        "collector_hash_matches": k5.sha256_file(COLLECTOR_SOURCE) == r2_run.get("collector_source_sha256"),
    }
    if not all(source_checks.values()):
        raise R2AR3Error(f"R2A-R2 executable source drift: {source_checks}")
    out["executable_source_binding"] = source_checks
    return out


def candidate_registry() -> Dict[str, Any]:
    common_delta = {
        "on_time": "excluded: no approved timetable/service-window/promised-ETA input",
        "energy_per_passenger": "excluded: no approved runtime energy model/output",
        "constraint": "excluded: K-safety is pre-action and no generic constraint penalty is allowed",
    }
    candidates = [
        {
            "candidate_id": "A_HISTORICAL_FULL_REFERENCE_V1",
            "family": "HISTORICAL_FULL_REFERENCE",
            "status": "LINEAGE_ONLY_NON_EXECUTABLE",
            "executable": False,
            "components": list(HISTORICAL_WEIGHTS),
            "weights": HISTORICAL_WEIGHTS,
            "normalization_mode": "HISTORICAL_DRAFT_UNLOCKED",
            "non_executable_reasons": ["on_time and energy unsupported", "headway fixture primitives absent", "constraint semantics unapproved"],
            "candidate_delta": {},
        },
        {
            "candidate_id": "B_PV8_CAUSAL_CORE_V1",
            "family": "PV8_CAUSAL_CORE",
            "status": "REGISTERED_NON_EXECUTABLE_ON_FROZEN_FIXTURES",
            "executable": False,
            "components": ["service", "avg_wait", "p95_wait", "headway_cv", "intervention"],
            "weights": {key: HISTORICAL_WEIGHTS[key] for key in ["service", "avg_wait", "p95_wait", "headway_cv", "intervention"]},
            "normalization_mode": "STEP113_BASELINE_EXCESS",
            "non_executable_reasons": ["frozen delayed-harm fixtures contain no same-occurrence headway series", "B1 headway_cv reference is zero and no epsilon is approved"],
            "candidate_delta": {**common_delta, "bunching": "excluded", "fleet_reduction": "excluded"},
        },
        {
            "candidate_id": "C_PV8_CAUSAL_CORE_PLUS_PROVISIONAL_REFERENCES_V1",
            "family": "PV8_CAUSAL_CORE_PLUS_PROVISIONAL_REFERENCES",
            "status": "REGISTERED_NON_EXECUTABLE_ON_FROZEN_FIXTURES",
            "executable": False,
            "components": ["service", "avg_wait", "p95_wait", "headway_cv", "intervention", "bunching", "fleet_reduction"],
            "weights": {key: HISTORICAL_WEIGHTS[key] for key in ["service", "avg_wait", "p95_wait", "headway_cv", "intervention", "bunching", "fleet_reduction"]},
            "normalization_mode": "STEP113_PLUS_R2AR2_PROVISIONAL_REFERENCES",
            "non_executable_reasons": ["headway/bunching branch outcomes absent", "bunching and fleet references are ablation-only and branch-invariant"],
            "candidate_delta": common_delta,
        },
        {
            "candidate_id": "D_PV8_SERVICE_WAIT_CORE_EXCESS_V1",
            "family": "PV8_EXECUTABLE_SERVICE_WAIT_CORE",
            "status": "EXECUTABLE_ABLATION_ONLY",
            "executable": True,
            "components": ["service", "avg_wait", "p95_wait", "intervention"],
            "weights": {key: HISTORICAL_WEIGHTS[key] for key in ["service", "avg_wait", "p95_wait", "intervention"]},
            "normalization_mode": "STEP113_BASELINE_EXCESS",
            "non_executable_reasons": [],
            "candidate_delta": {**common_delta, "bunching": "excluded", "headway_cv": "excluded: fixture primitive absent", "fleet_reduction": "excluded"},
        },
        {
            "candidate_id": "E_PV8_SERVICE_WAIT_CORE_CENTERED_V1",
            "family": "PV8_EXECUTABLE_SERVICE_WAIT_CORE",
            "status": "EXECUTABLE_NORMALIZATION_SENSITIVITY_ONLY",
            "executable": True,
            "components": ["service", "avg_wait", "p95_wait", "intervention"],
            "weights": {key: HISTORICAL_WEIGHTS[key] for key in ["service", "avg_wait", "p95_wait", "intervention"]},
            "normalization_mode": "B1_CENTERED_SIGNED_DELTA",
            "non_executable_reasons": [],
            "candidate_delta": {**common_delta, "bunching": "excluded", "headway_cv": "excluded: fixture primitive absent", "fleet_reduction": "excluded"},
        },
        {
            "candidate_id": "F_PV8_SERVICE_GATED_CENTERED_CORE_V1",
            "family": "PV8_EXECUTABLE_SERVICE_ALIGNED_CORE",
            "status": "EXECUTABLE_SELECTION_ELIGIBLE",
            "executable": True,
            "components": ["service", "avg_wait", "p95_wait", "intervention"],
            "weights": {key: HISTORICAL_WEIGHTS[key] for key in ["service", "avg_wait", "p95_wait", "intervention"]},
            "normalization_mode": "B1_CENTERED_SIGNED_DELTA_WITH_SERVICE_GATED_IMPROVEMENTS",
            "non_executable_reasons": [],
            "candidate_delta": {**common_delta, "bunching": "excluded", "headway_cv": "excluded: fixture primitive absent", "fleet_reduction": "excluded"},
        },
    ]
    return {
        "created_at": iso_kst(),
        "registry_version": "PV8_R2AR3_CANDIDATE_REGISTRY_V1",
        "historical_lineage_candidate": "R1_BALANCED_DEFAULT",
        "historical_weights_unchanged": HISTORICAL_WEIGHTS,
        "candidate_count": len(candidates),
        "executable_candidate_count": sum(bool(row["executable"]) for row in candidates),
        "candidates": candidates,
        "weight_optimization_or_search_performed": False,
        "unsupported_component_silently_zeroed": False,
        "action_name_bonus_or_penalty": False,
        "generic_constraint_penalty": False,
        "k_safety_layer": "PRE_ACTION_HARD_CONSTRAINT_NOT_A_REWARD_COMPONENT",
    }


def horizon_contract() -> Dict[str, Any]:
    upstream = k5.read_json(R2AR2_ROOT / "r2ar2_harm_reveal_horizon_audit.json")
    indices = dict(upstream["first_harm_reveal_indices"])
    positive = sorted({int(value) for value in indices.values() if value is not None and int(value) > 0})
    safety_zero = sorted(key for key, value in indices.items() if value == 0)
    h_cover = max(positive)
    if h_cover != int(upstream["maximum_observed_harm_reveal_horizon"]) or h_cover != 4:
        raise R2AR3Error("mechanical H_cover does not match frozen R2A-R2 evidence")
    candidates = sorted({1, *positive, h_cover})
    return {
        "created_at": iso_kst(),
        "contract": "PV8_R2AR3_HORIZON_CONTRACT_V1",
        "collector_index_semantics": "transition i has outcome interval (decision_ts + 60*(i-1), decision_ts + 60*i] within the branch; events at the right boundary are included",
        "requested_horizon_semantics": "Hi aggregates causal outcomes in (decision_ts, decision_ts + 60*i], capped at the fixture terminal/revisit boundary",
        "left_boundary_inclusive": False,
        "right_boundary_inclusive": True,
        "fixture_first_harm_reveal_indices": indices,
        "pre_action_safety_boundary_fixture_ids": safety_zero,
        "distinct_positive_fixture_reveal_boundaries": positive,
        "horizon_candidates": [f"H{value}" for value in candidates],
        "horizon_candidate_values": candidates,
        "H_cover_derivation": "max(distinct positive first_harm_reveal_transition values from frozen R2A-R2 fixtures)",
        "H_cover": f"H{h_cover}",
        "H_cover_value": h_cover,
        "H_cover_guessed_or_manually_selected": False,
        "reward_horizon_approved": False,
    }


def _clip(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def score_candidate(
    candidate: Mapping[str, Any],
    metrics: Mapping[str, Any],
    references: Mapping[str, Any],
) -> Tuple[Optional[float], List[Dict[str, Any]], str]:
    if not candidate["executable"]:
        return None, [], "CANDIDATE_NON_EXECUTABLE"
    service = metrics.get("service_rate")
    avg_wait = metrics.get("avg_wait_seconds")
    p95_wait = metrics.get("p95_wait_seconds")
    if service is None or avg_wait is None or p95_wait is None:
        return None, [], "NOT_EVALUABLE_NO_ELIGIBLE_CAUSAL_COHORT"
    ref_service = float(references["service_rate"])
    ref_avg = float(references["avg_wait_seconds"])
    ref_p95 = float(references["p95_wait_seconds"])
    if min(ref_service, ref_avg, ref_p95) <= 0:
        raise R2AR3Error("ablation reference denominator must be positive")
    service = float(service)
    avg_wait = float(avg_wait)
    p95_wait = float(p95_wait)
    intervention_rate = _clip(float(metrics.get("intervention_count_candidate", 0)), 0.0, 1.0)
    mode = str(candidate["normalization_mode"])
    service_norm = _clip(service, 0.0, 1.0)
    if mode == "STEP113_BASELINE_EXCESS":
        avg_norm = max(0.0, avg_wait / ref_avg - 1.0)
        p95_norm = max(0.0, p95_wait / ref_p95 - 1.0)
        avg_contribution = -HISTORICAL_WEIGHTS["avg_wait"] * avg_norm
        p95_contribution = -HISTORICAL_WEIGHTS["p95_wait"] * p95_norm
    elif mode == "B1_CENTERED_SIGNED_DELTA":
        avg_norm = avg_wait / ref_avg - 1.0
        p95_norm = p95_wait / ref_p95 - 1.0
        avg_contribution = -HISTORICAL_WEIGHTS["avg_wait"] * avg_norm
        p95_contribution = -HISTORICAL_WEIGHTS["p95_wait"] * p95_norm
    elif mode == "B1_CENTERED_SIGNED_DELTA_WITH_SERVICE_GATED_IMPROVEMENTS":
        avg_delta = 1.0 - avg_wait / ref_avg
        p95_delta = 1.0 - p95_wait / ref_p95
        service_clear = service + TOLERANCE >= ref_service
        avg_norm = avg_delta if avg_delta <= 0.0 or service_clear else 0.0
        p95_norm = p95_delta if p95_delta <= 0.0 or service_clear else 0.0
        avg_contribution = HISTORICAL_WEIGHTS["avg_wait"] * avg_norm
        p95_contribution = HISTORICAL_WEIGHTS["p95_wait"] * p95_norm
    else:
        raise R2AR3Error(f"unsupported executable normalization mode: {mode}")
    service_contribution = HISTORICAL_WEIGHTS["service"] * service_norm
    intervention_contribution = -HISTORICAL_WEIGHTS["intervention"] * intervention_rate
    contributions = [
        ("service", service, ref_service, service_norm, HISTORICAL_WEIGHTS["service"], service_contribution),
        ("avg_wait", avg_wait, ref_avg, avg_norm, HISTORICAL_WEIGHTS["avg_wait"], avg_contribution),
        ("p95_wait", p95_wait, ref_p95, p95_norm, HISTORICAL_WEIGHTS["p95_wait"], p95_contribution),
        ("intervention", intervention_rate, 0.0, intervention_rate, HISTORICAL_WEIGHTS["intervention"], intervention_contribution),
    ]
    rows = [
        {
            "component": component,
            "raw_value": raw_value,
            "reference_value": reference_value,
            "normalized_value": normalized_value,
            "weight": weight,
            "weighted_contribution": contribution,
            "normalization_mode": mode,
        }
        for component, raw_value, reference_value, normalized_value, weight, contribution in contributions
    ]
    total = float(sum(row["weighted_contribution"] for row in rows))
    return total, rows, "EXECUTED_ABLATION_SCORE"


def fixture_exogenous_hash(spec: Mapping[str, Any]) -> str:
    reference_events = [
        {
            "event_ts": event.event_ts,
            "request_id": event.request_id,
            "passenger_id": event.passenger_id,
            "waiting_since_ts": event.metadata.get("waiting_since_ts", event.event_ts),
        }
        for event in spec["reference_events"]
        if event.event_type.value == "PASSENGER_GENERATED"
    ]
    candidate_events = [
        {
            "event_ts": event.event_ts,
            "request_id": event.request_id,
            "passenger_id": event.passenger_id,
            "waiting_since_ts": event.metadata.get("waiting_since_ts", event.event_ts),
        }
        for event in spec["candidate_events"]
        if event.event_type.value == "PASSENGER_GENERATED"
    ]
    if spec["candidate_allowed"] and reference_events != candidate_events:
        raise R2AR3Error(f"paired exogenous identity drift: {spec['fixture_id']}")
    payload = reference_events if spec["candidate_allowed"] else spec["exogenous"]
    return canonical_hash(payload)


def run_ablation(
    registry: Mapping[str, Any],
    horizons: Mapping[str, Any],
    references: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    specs = r2ar2.delayed_fixture_specs()
    reveal_map = horizons["fixture_first_harm_reveal_indices"]
    result_rows: List[Dict[str, Any]] = []
    margin_rows: List[Dict[str, Any]] = []
    contribution_rows: List[Dict[str, Any]] = []
    for candidate in registry["candidates"]:
        for requested_horizon in horizons["horizon_candidate_values"]:
            horizon_id = f"H{requested_horizon}"
            for spec in specs:
                fixture_id = str(spec["fixture_id"])
                fixture_harmful = fixture_id != "BENEFICIAL_SKIP"
                effective_horizon = min(int(requested_horizon), int(spec["terminal_or_revisit_transition"]))
                exogenous_hash = fixture_exogenous_hash(spec)
                branch_scores: Dict[str, Optional[float]] = {}
                branch_status: Dict[str, str] = {}
                if not spec["candidate_allowed"]:
                    for branch in ["REFERENCE", "CANDIDATE_SKIP"]:
                        result_rows.append({
                            "candidate_id": candidate["candidate_id"],
                            "candidate_executable": candidate["executable"],
                            "horizon_id": horizon_id,
                            "requested_horizon": requested_horizon,
                            "effective_horizon": effective_horizon,
                            "fixture_id": fixture_id,
                            "branch": branch,
                            "fixture_harmful": fixture_harmful,
                            "candidate_skip_allowed": False,
                            "evaluation_status": "PREACTION_K_SAFETY_BLOCKED",
                            "reward_total": None,
                            "service_rate": None,
                            "avg_wait_seconds": None,
                            "p95_wait_seconds": None,
                            "intervention_rate": None,
                            "event_trace_hash": None,
                            "exogenous_schedule_hash": exogenous_hash,
                            "b1_scope": "ABLATION_REFERENCE_ONLY",
                            "reward_contract_approved": False,
                        })
                    margin_rows.append({
                        "candidate_id": candidate["candidate_id"],
                        "candidate_executable": candidate["executable"],
                        "horizon_id": horizon_id,
                        "requested_horizon": requested_horizon,
                        "effective_horizon": effective_horizon,
                        "fixture_id": fixture_id,
                        "fixture_harmful": True,
                        "candidate_skip_allowed": False,
                        "comparison_status": "PREACTION_K_SAFETY_BLOCKED_PASS",
                        "reference_reward": None,
                        "candidate_skip_reward": None,
                        "skip_minus_reference_margin": None,
                        "tolerance": TOLERANCE,
                        "harm_reveal_transition": reveal_map[fixture_id],
                        "horizon_covers_reveal": True,
                        "primary_safety_criterion_passed": True,
                        "beneficial_preservation_passed": None,
                        "harmful_skip_positive_absolute_reward": False,
                        "exogenous_schedule_hash": exogenous_hash,
                    })
                    continue
                for branch, event_key in [("REFERENCE", "reference_events"), ("CANDIDATE_SKIP", "candidate_events")]:
                    metrics = r2ar2.collect_branch_at_horizon(
                        fixture_id,
                        branch,
                        spec[event_key],
                        horizon=effective_horizon,
                        initial_requests=spec["initial_requests"],
                    )
                    score, contributions, status = score_candidate(candidate, metrics, references)
                    branch_scores[branch] = score
                    branch_status[branch] = status
                    result_rows.append({
                        "candidate_id": candidate["candidate_id"],
                        "candidate_executable": candidate["executable"],
                        "horizon_id": horizon_id,
                        "requested_horizon": requested_horizon,
                        "effective_horizon": effective_horizon,
                        "fixture_id": fixture_id,
                        "branch": branch,
                        "fixture_harmful": fixture_harmful,
                        "candidate_skip_allowed": True,
                        "evaluation_status": status,
                        "reward_total": score,
                        "service_rate": metrics.get("service_rate"),
                        "avg_wait_seconds": metrics.get("avg_wait_seconds"),
                        "p95_wait_seconds": metrics.get("p95_wait_seconds"),
                        "intervention_rate": float(metrics.get("intervention_count_candidate", 0)),
                        "event_trace_hash": metrics.get("event_trace_hash"),
                        "exogenous_schedule_hash": exogenous_hash,
                        "b1_scope": "ABLATION_REFERENCE_ONLY",
                        "reward_contract_approved": False,
                    })
                    for component in contributions:
                        contribution_rows.append({
                            "candidate_id": candidate["candidate_id"],
                            "horizon_id": horizon_id,
                            "fixture_id": fixture_id,
                            "branch": branch,
                            **component,
                        })
                reference_reward = branch_scores.get("REFERENCE")
                skip_reward = branch_scores.get("CANDIDATE_SKIP")
                margin = None if reference_reward is None or skip_reward is None else float(skip_reward - reference_reward)
                reveal = reveal_map[fixture_id]
                horizon_covers = bool(reveal is None or int(reveal) <= requested_horizon)
                status = "EXECUTED" if margin is not None else "NOT_EVALUABLE_CAUSAL_COHORT_ABSENT"
                primary_pass = bool(margin is not None and (not fixture_harmful or margin <= TOLERANCE))
                beneficial_pass = bool(margin is not None and margin >= -TOLERANCE) if not fixture_harmful else None
                margin_rows.append({
                    "candidate_id": candidate["candidate_id"],
                    "candidate_executable": candidate["executable"],
                    "horizon_id": horizon_id,
                    "requested_horizon": requested_horizon,
                    "effective_horizon": effective_horizon,
                    "fixture_id": fixture_id,
                    "fixture_harmful": fixture_harmful,
                    "candidate_skip_allowed": True,
                    "comparison_status": status if candidate["executable"] else "CANDIDATE_NON_EXECUTABLE",
                    "reference_reward": reference_reward,
                    "candidate_skip_reward": skip_reward,
                    "skip_minus_reference_margin": margin,
                    "tolerance": TOLERANCE,
                    "harm_reveal_transition": reveal,
                    "horizon_covers_reveal": horizon_covers,
                    "primary_safety_criterion_passed": primary_pass,
                    "beneficial_preservation_passed": beneficial_pass,
                    "harmful_skip_positive_absolute_reward": bool(fixture_harmful and skip_reward is not None and skip_reward > TOLERANCE),
                    "exogenous_schedule_hash": exogenous_hash,
                })
    return result_rows, margin_rows, contribution_rows


def service_alignment_audit(
    registry: Mapping[str, Any],
    horizons: Mapping[str, Any],
    margins: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    h_cover = int(horizons["H_cover_value"])
    candidate_rows = []
    for candidate in registry["candidates"]:
        cid = candidate["candidate_id"]
        scoped = [row for row in margins if row["candidate_id"] == cid and row["requested_horizon"] == h_cover]
        harmful = [row for row in scoped if row["fixture_harmful"]]
        beneficial = next(row for row in scoped if not row["fixture_harmful"])
        harmful_numeric = [float(row["skip_minus_reference_margin"]) for row in harmful if row["skip_minus_reference_margin"] is not None]
        blocked_count = sum(row["comparison_status"] == "PREACTION_K_SAFETY_BLOCKED_PASS" for row in harmful)
        harmful_pass = bool(candidate["executable"] and all(bool(row["primary_safety_criterion_passed"]) for row in harmful))
        beneficial_pass = bool(beneficial["beneficial_preservation_passed"])
        reference_rewards = [
            float(row["reward_total"])
            for row in results
            if row["candidate_id"] == cid
            and row["requested_horizon"] == h_cover
            and row["branch"] == "REFERENCE"
            and row["reward_total"] is not None
        ]
        every_reference_strongly_negative = bool(reference_rewards and all(value < -TOLERANCE for value in reference_rewards))
        candidate_rows.append({
            "candidate_id": cid,
            "executable": bool(candidate["executable"]),
            "h_cover": h_cover,
            "harmful_fixture_count": len(harmful),
            "preaction_blocked_harmful_count": blocked_count,
            "harmful_numeric_margin_count": len(harmful_numeric),
            "largest_harmful_margin": max(harmful_numeric) if harmful_numeric else None,
            "smallest_harmful_margin": min(harmful_numeric) if harmful_numeric else None,
            "harmful_positive_advantage_count": sum(value > TOLERANCE for value in harmful_numeric),
            "harmful_positive_absolute_skip_reward_count": sum(bool(row["harmful_skip_positive_absolute_reward"]) for row in harmful),
            "all_horizon_harmful_positive_absolute_skip_reward_count": sum(
                bool(row["harmful_skip_positive_absolute_reward"])
                for row in margins
                if row["candidate_id"] == cid and row["fixture_harmful"]
            ),
            "beneficial_margin": beneficial["skip_minus_reference_margin"],
            "beneficial_preserved": beneficial_pass,
            "every_reference_action_strongly_negative": every_reference_strongly_negative,
            "service_alignment_passed": bool(harmful_pass and beneficial_pass and not every_reference_strongly_negative),
        })
    executable = [row for row in candidate_rows if row["executable"]]
    ranked = sorted(
        executable,
        key=lambda row: (
            not row["service_alignment_passed"],
            row["harmful_positive_advantage_count"],
            row["harmful_positive_absolute_skip_reward_count"],
            -(float(row["beneficial_margin"]) if row["beneficial_margin"] is not None else float("-inf")),
            row["candidate_id"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    selected = ranked[0] if ranked and ranked[0]["service_alignment_passed"] else None
    return {
        "created_at": iso_kst(),
        "contract": "PV8_R2AR3_SERVICE_ALIGNMENT_AUDIT_V1",
        "primary_safety_criterion": "harmful skip_reward - reference_reward <= 1e-9; pre-action K-blocked SKIP passes without a reward sample",
        "beneficial_preservation_criterion": "BENEFICIAL_SKIP margin >= -1e-9 and reference paths are not all strongly negative",
        "candidate_h_cover_results": candidate_rows,
        "candidate_ranking": ranked,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_candidate_alignment_passed": bool(selected),
        "harmful_skip_positive_advantage_for_selected": bool(selected and selected["harmful_positive_advantage_count"] > 0),
        "harmful_skip_positive_absolute_reward_for_selected": bool(selected and selected["harmful_positive_absolute_skip_reward_count"] > 0),
        "harmful_skip_positive_absolute_reward_in_any_tested_horizon_for_selected": bool(
            selected and selected["all_horizon_harmful_positive_absolute_skip_reward_count"] > 0
        ),
        "reward_contract_approved": False,
    }


def horizon_sensitivity(
    selected_candidate_id: str,
    horizons: Mapping[str, Any],
    margins: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = []
    reveal_map = horizons["fixture_first_harm_reveal_indices"]
    positive_required = {fixture: int(value) for fixture, value in reveal_map.items() if value is not None and int(value) > 0}
    for horizon in horizons["horizon_candidate_values"]:
        scoped = [row for row in margins if row["candidate_id"] == selected_candidate_id and row["requested_horizon"] == horizon]
        harmful = [row for row in scoped if row["fixture_harmful"]]
        beneficial = next(row for row in scoped if not row["fixture_harmful"])
        numeric = [float(row["skip_minus_reference_margin"]) for row in harmful if row["skip_minus_reference_margin"] is not None]
        uncovered = sorted(fixture for fixture, reveal in positive_required.items() if reveal > horizon)
        all_observed_safe = all(
            row["comparison_status"] == "PREACTION_K_SAFETY_BLOCKED_PASS"
            or (row["skip_minus_reference_margin"] is not None and row["skip_minus_reference_margin"] <= TOLERANCE)
            for row in harmful
            if row["fixture_id"] not in uncovered
        )
        rows.append({
            "horizon_id": f"H{horizon}",
            "horizon_value": horizon,
            "uncovered_harm_fixture_ids": uncovered,
            "harm_coverage_complete": not uncovered,
            "largest_observed_harmful_margin": max(numeric) if numeric else None,
            "beneficial_margin": beneficial["skip_minus_reference_margin"],
            "observed_safety_passed": all_observed_safe,
            "full_alignment_passed": bool(not uncovered and all_observed_safe and beneficial["beneficial_preservation_passed"]),
        })
    normalization = [
        {
            "candidate_id": row["candidate_id"],
            "rank": row.get("rank"),
            "largest_harmful_margin": row["largest_harmful_margin"],
            "harmful_positive_absolute_skip_reward_count": row["harmful_positive_absolute_skip_reward_count"],
            "beneficial_margin": row["beneficial_margin"],
            "service_alignment_passed": row["service_alignment_passed"],
        }
        for row in alignment["candidate_ranking"]
    ]
    return {
        "created_at": iso_kst(),
        "selected_candidate_id": selected_candidate_id,
        "horizon_rows": rows,
        "H1_failed": not next(row for row in rows if row["horizon_value"] == 1)["full_alignment_passed"],
        "H_cover_passed": next(row for row in rows if row["horizon_value"] == horizons["H_cover_value"])["full_alignment_passed"],
        "normalization_sensitivity_at_H_cover": normalization,
        "weights_optimized_or_searched": False,
        "training_normalization_approved": False,
    }


def selection_decision(
    registry: Mapping[str, Any],
    horizons: Mapping[str, Any],
    alignment: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> Dict[str, Any]:
    selected_id = alignment["selected_candidate_id"]
    selected = next(row for row in registry["candidates"] if row["candidate_id"] == selected_id)
    passed = bool(
        selected_id == "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
        and alignment["selected_candidate_alignment_passed"]
        and not alignment["harmful_skip_positive_advantage_for_selected"]
        and not alignment["harmful_skip_positive_absolute_reward_for_selected"]
        and not alignment["harmful_skip_positive_absolute_reward_in_any_tested_horizon_for_selected"]
        and sensitivity["H1_failed"]
        and sensitivity["H_cover_passed"]
        and horizons["H_cover_value"] == 4
    )
    if not passed:
        raise R2AR3Error("no candidate satisfies frozen service-alignment selection gate")
    return {
        "created_at": iso_kst(),
        "final_decision": DECISION,
        "selected_candidate_id": selected_id,
        "selected_candidate_formula": "3*clip(service_rate,0,1) + 2*gated(1-avg_wait/B1_avg) + 3*gated(1-p95_wait/B1_p95) - 0.25*explicit_intervention_rate",
        "selected_candidate_components": selected["components"],
        "selected_candidate_weights": selected["weights"],
        "selected_normalization_mode": selected["normalization_mode"],
        "selected_horizon_candidate": horizons["H_cover"],
        "selected_horizon_value": horizons["H_cover_value"],
        "selection_scope": "EXPLICIT_APPROVAL_CANDIDATE_ONLY",
        "pv8_b1_constants_scope": "ABLATION_REFERENCE_ONLY",
        "excluded_components": selected["candidate_delta"],
        "exact_approval_required": [
            "approve F_PV8_SERVICE_GATED_CENTERED_CORE_V1 component set and unchanged positive-magnitude weights",
            "approve service-gated B1-centered normalization semantics and the use of H_cover=H4",
            "approve exclusion of on-time, energy, bunching, headway-CV, fleet, and generic constraint penalty from this candidate",
            "approve whether bounded B1 constants may be regenerated on a broader prospective reference before any training reward materialization",
            "approve performance guards separately; K-safety remains a non-reward pre-action hard constraint",
        ],
        "reward_contract_approved": False,
        "training_reward_values_materialized": False,
        "training_normalization_approved": False,
        "downstream_authorized": False,
    }


def claim_guards() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_contract_approved": False,
        "reward_values_materialized": False,
        "training_normalization_approved": False,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "automatic_r2ar4_execution_authorized": False,
        "automatic_r2b_execution_authorized": False,
        "automatic_r3_execution_authorized": False,
        "mappo_training_authorized": False,
    }


def run_pytest_regression() -> Dict[str, Any]:
    tests = [
        TRAINING_ROOT / "simulator" / "test_pv8_reward_outcome_collector.py",
        TRAINING_ROOT / "simulator" / "test_pv8_k4_service_obligation_state.py",
        TRAINING_ROOT / "simulator" / "test_pv8_k8_k_action_mask_runtime.py",
        TRAINING_ROOT / "simulator" / "test_pv8_k9_k_mask_snapshot_lifecycle.py",
    ]
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(path) for path in tests]]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(TRAINING_ROOT), env.get("PYTHONPATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = "/private/tmp/pv8_r2ar3_pycache"
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


def final_report(summary: Mapping[str, Any]) -> str:
    horizon_lines = [
        f"- H{row['horizon_value']}: coverage_complete={str(row['harm_coverage_complete']).lower()}, largest_harmful_margin={row['largest_observed_harmful_margin']}, beneficial_margin={row['beneficial_margin']}, pass={str(row['full_alignment_passed']).lower()}"
        for row in summary["horizon_rows"]
    ]
    return "\n".join([
        "# PV8-R2A-R3 Bounded Reward/Horizon Ablation",
        "",
        f"- artifact root: `{summary['artifact_root']}`",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{DECISION}`",
        f"- selected candidate: `{summary['selected_candidate']}`",
        "- reward approval / training materialization / training: `false / false / false`",
        "",
        "## Horizon Contract",
        "",
        "`H_cover = H4`, derived mechanically as the maximum positive first-harm-reveal transition in the frozen R2A-R2 fixtures. Hi includes events in `(decision_ts, decision_ts + 60*i]`; the left boundary is excluded, the right boundary is included, and evaluation is capped at each fixture terminal/revisit boundary.",
        "",
        "## Candidate Executability",
        "",
        "A is lineage-only because on-time/energy/constraint inputs are unsupported. B is non-executable because frozen fixtures have no branch headway series and B1 headway-CV is zero without an approved epsilon. C is additionally blocked by provisional bunching/fleet references. D, E, and F execute using only service rate, accrued average/p95 wait, and explicit forced/external intervention events.",
        "",
        "The selected F formula is `3*service + 2*gated(1-avg/B1_avg) + 3*gated(1-p95/B1_p95) - 0.25*explicit_intervention`. Positive wait improvements contribute only when B1 service is preserved; wait deterioration always contributes negatively.",
        "",
        "## Horizon Sensitivity",
        "",
        *horizon_lines,
        "",
        "Selected-candidate harmful fixture margins (`SKIP - reference`):",
        "",
        *summary["harmful_margin_lines"],
        "",
        f"H1 failed because it does not cover fixtures whose first harm appears at transitions 2, 3, and 4. At H4, the largest harmful SKIP advantage is `{summary['largest_harmful_margin']}` and the beneficial-SKIP margin is `{summary['beneficial_margin']}`. Across all tested horizons, F's harmful positive absolute SKIP reward count is `{summary['all_horizon_positive_absolute_count']}`; its H4 harmful positive advantage count is `0`.",
        "",
        "## Exclusions and Normalization",
        "",
        "On-time and energy are intentionally excluded because they remain unsupported. Bunching, headway-CV, and fleet are excluded because the frozen paired fixtures cannot causally vary or measure them. The generic constraint penalty is excluded; K-safety remains a pre-action hard constraint. R2A-R2 B1 constants remain `ABLATION_REFERENCE_ONLY` and are not promoted to training normalization.",
        "",
        "## Approval Boundary",
        "",
        "F with H4 is ready only for explicit reward-contract approval. Approval must cover the reduced component set, service-gated centered normalization, H4, exclusions, and the requirement to regenerate broader PV8 B1 references before training materialization.",
        "",
        "No API/DB access, policy evaluation, reward approval, training normalization approval, checkpoint reuse, downstream run, episode expansion, or MAPPO training occurred.",
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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar3.jsonl"
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
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar3.json"
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
    writer.json("_PV8_R2AR3_COMPLETE.lock", {
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
    pytest_result = run_pytest_regression()
    if not pytest_result["passed"]:
        raise R2AR3Error(f"collector/K-safety regression failed: {pytest_result}")
    registry = candidate_registry()
    horizons = horizon_contract()
    normalization = k5.read_json(R2AR2_ROOT / "r2ar2_normalization_candidate.json")
    if normalization.get("approved") or normalization.get("scope") is None:
        raise R2AR3Error("R2A-R2 B1 scope/approval guard mismatch")
    references = normalization["constants"]
    results, margins, contributions = run_ablation(registry, horizons, references)
    alignment = service_alignment_audit(registry, horizons, margins, results)
    if not alignment["selected_candidate_id"]:
        raise R2AR3Error("no aligned candidate")
    sensitivity = horizon_sensitivity(alignment["selected_candidate_id"], horizons, margins, alignment)
    selection = selection_decision(registry, horizons, alignment, sensitivity)
    guards = claim_guards()
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": READINESS,
        "gate_passed": True,
        "final_decision": selection["final_decision"],
        "selected_candidate_id": selection["selected_candidate_id"],
        "selected_horizon": selection["selected_horizon_candidate"],
        "failure_reasons": [],
    }
    selected_alignment = next(row for row in alignment["candidate_h_cover_results"] if row["candidate_id"] == selection["selected_candidate_id"])
    harmful_margin_lines = []
    for horizon in horizons["horizon_candidate_values"]:
        scoped = [
            row for row in margins
            if row["candidate_id"] == selection["selected_candidate_id"]
            and row["requested_horizon"] == horizon
            and row["fixture_harmful"]
        ]
        values = []
        for row in sorted(scoped, key=lambda value: value["fixture_id"]):
            if row["comparison_status"] == "PREACTION_K_SAFETY_BLOCKED_PASS":
                rendered = "K_BLOCKED"
            elif row["skip_minus_reference_margin"] is None:
                rendered = "NO_CAUSAL_COHORT"
            else:
                rendered = str(row["skip_minus_reference_margin"])
            values.append(f"{row['fixture_id']}={rendered}")
        harmful_margin_lines.append(f"- H{horizon}: " + ", ".join(values))
    summary = {
        "artifact_root": str(root),
        "selected_candidate": selection["selected_candidate_id"],
        "horizon_rows": sensitivity["horizon_rows"],
        "largest_harmful_margin": selected_alignment["largest_harmful_margin"],
        "beneficial_margin": selected_alignment["beneficial_margin"],
        "harmful_margin_lines": harmful_margin_lines,
        "all_horizon_positive_absolute_count": sum(
            bool(row["harmful_skip_positive_absolute_reward"])
            for row in margins
            if row["candidate_id"] == selection["selected_candidate_id"] and row["fixture_harmful"]
        ),
    }

    writer.json("r2ar3_candidate_registry.json", registry)
    writer.json("r2ar3_horizon_contract.json", horizons)
    writer.parquet("r2ar3_fixture_reward_results.parquet", results, RESULT_COLUMNS)
    writer.parquet("r2ar3_paired_reward_margins.parquet", margins, MARGIN_COLUMNS)
    writer.parquet("r2ar3_component_contributions.parquet", contributions, CONTRIBUTION_COLUMNS)
    writer.json("r2ar3_horizon_sensitivity.json", sensitivity)
    writer.json("r2ar3_service_alignment_audit.json", alignment)
    writer.json("r2ar3_candidate_selection_decision.json", selection)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "bounded-ablation",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "upstream_integrity": upstreams,
        "r2ar2_normalization_candidate_sha256": k5.sha256_file(R2AR2_ROOT / "r2ar2_normalization_candidate.json"),
        "candidate_count": registry["candidate_count"],
        "executable_candidate_count": registry["executable_candidate_count"],
        "horizon_count": len(horizons["horizon_candidate_values"]),
        "fixture_count": len(r2ar2.delayed_fixture_specs()),
        "fixture_reward_row_count": len(results),
        "paired_margin_row_count": len(margins),
        "component_contribution_row_count": len(contributions),
        "ablation_reward_score_count": sum(row["reward_total"] is not None for row in results),
        "training_reward_value_materialization_count": 0,
        "new_bis_api_call_count": 0,
        "db_query_count": 0,
        "db_write_count": 0,
        "policy_evaluation_count": 0,
        "checkpoint_reuse_count": 0,
        "episode_expansion_count": 0,
        "r2ar4_execution_count": 0,
        "r2b_execution_count": 0,
        "r3_execution_count": 0,
        "mappo_training_count": 0,
        "qwen_train": False,
        "qwen_inference": False,
        "pytest_regression": pytest_result,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": READINESS, "final_decision": selection["final_decision"]})
    writer.text("final_report.md", final_report(summary))
    write_manifest_and_lock(writer, gate)

    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar3.json", "_PV8_R2AR3_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise R2AR3Error(f"R2A-R3 artifact integrity failure: {own}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {selection['final_decision']}")
    print(f"selected_candidate: {selection['selected_candidate_id']}")
    print(f"H_cover: {selection['selected_horizon_candidate']}")
    print(f"largest_harmful_margin: {selected_alignment['largest_harmful_margin']}")
    print(f"beneficial_margin: {selected_alignment['beneficial_margin']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bounded-ablation"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
