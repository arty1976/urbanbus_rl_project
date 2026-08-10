#!/usr/bin/env python3
"""PV8-R2A-R1 reward contract repair and delayed-harm readiness audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping
from zoneinfo import ZoneInfo

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar1_reward_contract_repair_audit.py"

K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"
K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"
R1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r1_mappo_retraining_preflight_20260808_142604"
R2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2_reward_episode_audit_20260808_143743"
R2A_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2a_reward_authority_promotion_audit_20260808_145209"
LEGACY_B1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2a_threshold_baseline_feasibility_20260720_000000"

UPSTREAMS = {
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
    "PV8-K9": (K9_ROOT, "artifact_manifest_srp2_bis_pv8_k9.json", "_PV8_K9_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED"),
    "PV8-R1": (R1_ROOT, "artifact_manifest_srp2_bis_pv8_r1.json", "_PV8_R1_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R1_MAPPO_RETRAINING_PREFLIGHT_COMPLETE"),
    "PV8-R2": (R2_ROOT, "artifact_manifest_srp2_bis_pv8_r2.json", "_PV8_R2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2_REWARD_AND_EPISODE_DATA_AUDIT_COMPLETE"),
    "PV8-R2A": (R2A_ROOT, "artifact_manifest_srp2_bis_pv8_r2a.json", "_PV8_R2A_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2A_REWARD_AUTHORITY_AND_PROMOTION_AUDIT_COMPLETE"),
}

EXPECTED_RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
EXPECTED_OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"
EXPECTED_TENSORS = {"num_agents": 8, "actor_obs_dim": 16, "critic_obs_dim": 64, "action_dim": 3}
EXPECTED_WEIGHTS = {
    "w_service": 3.0,
    "w_avg_wait": 2.0,
    "w_long_wait": 3.0,
    "w_ontime": 1.0,
    "w_bunching": 1.5,
    "w_headway_cv": 1.0,
    "w_energy_per_passenger": 1.0,
    "w_fleet_reduction": 0.5,
    "w_intervention": 0.25,
    "w_constraint": 10.0,
}

ALLOWED_INPUT_CLASSES = [
    "AVAILABLE_EXACT",
    "DERIVABLE_CAUSAL",
    "BASELINE_CONSTANT_REQUIRED",
    "SEMANTICS_REPAIR_REQUIRED",
    "NOT_AVAILABLE",
]
ALLOWED_CONSTRAINT_CLASSES = ["SUPPORTED", "REPAIR_REQUIRED", "REMOVE_FROM_CANDIDATE"]
ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar1_reward_contract_repair_audit"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR1_REWARD_CONTRACT_REPAIR_AUDIT_COMPLETE"
DECISION = "PV8_REWARD_REPAIR_ADDITIONAL_INPUT_REQUIRED"
READINESS = "SRP2_BIS_PV8_R2AR1_COMPLETE_ADDITIONAL_REWARD_INPUT_REQUIRED"

PAYLOADS = [
    "r2ar1_reward_input_closure.json",
    "r2ar1_b1_normalization_audit.json",
    "r2ar1_on_time_semantics.json",
    "r2ar1_intervention_semantics.json",
    "r2ar1_constraint_repair_audit.json",
    "r2ar1_delayed_harm_test_contract.json",
    "r2ar1_reward_candidate_delta.json",
    "r2ar1_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R2AR1Error(RuntimeError):
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
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed_gate != expected_gate or not k5.manifest_ok(checks):
            raise R2AR1Error(f"{label} integrity failure: gate={observed_gate}, checks={checks}")
        results[label] = {
            "artifact_root": str(root),
            "observed_gate": observed_gate,
            "expected_gate": expected_gate,
            "manifest_integrity": checks,
        }
    return results


def verify_frozen_contract() -> Dict[str, Any]:
    tensor = k5.read_json(R1_ROOT / "r1_mappo_tensor_contract.json")
    episode = k5.read_json(R1_ROOT / "r1_frozen_episode_manifest.json")
    k8 = k5.read_json(K8_ROOT / "k8_frozen_rule_contract.json")
    k9 = k5.read_json(K9_ROOT / "k9_version_hash_binding_audit.json")
    r2 = k5.read_json(R2_ROOT / "r2_training_data_readiness.json")
    r2a = k5.read_json(R2A_ROOT / "r2a_reward_promotion_decision.json")
    weights = k5.read_json(R2A_ROOT / "r2a_reward_weight_contract.json")
    observed_weights = weights.get("exact_positive_magnitude_weights")
    checks = {
        "tensor_contract": {key: tensor.get(key) for key in EXPECTED_TENSORS},
        "tensor_matches": all(tensor.get(key) == value for key, value in EXPECTED_TENSORS.items()),
        "r1_rulebook_hash_matches": episode.get("frozen_bindings", {}).get("rulebook_sha256") == EXPECTED_RULEBOOK_SHA256,
        "r1_occurrence_hash_matches": episode.get("frozen_bindings", {}).get("occurrence_master_sha256") == EXPECTED_OCCURRENCE_SHA256,
        "k8_rulebook_hash_matches": k8.get("static_rulebook_sha256") == EXPECTED_RULEBOOK_SHA256,
        "k8_occurrence_hash_matches": k8.get("occurrence_master_sha256") == EXPECTED_OCCURRENCE_SHA256,
        "k9_rulebook_hash_matches": k9.get("rulebook_sha256") == EXPECTED_RULEBOOK_SHA256,
        "k9_occurrence_hash_matches": k9.get("occurrence_master_sha256") == EXPECTED_OCCURRENCE_SHA256,
        "r2_reward_materialization_blocked": r2.get("final_decision") == "PV8_REWARD_MATERIALIZATION_BLOCKED",
        "r2a_repair_required": r2a.get("final_decision") == "PV8_REWARD_CONTRACT_REPAIR_REQUIRED",
        "r1_balanced_default_unchanged": observed_weights == EXPECTED_WEIGHTS,
    }
    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    checks["failure_count"] = sum(not value for value in boolean_checks)
    checks["frozen_weights"] = observed_weights
    if checks["failure_count"]:
        raise R2AR1Error(f"frozen contract mismatch: {checks}")
    return checks


def reward_input_closure() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = [
        {
            "component": "service_rate",
            "weight_key": "w_service",
            "weight": 3.0,
            "classification": "DERIVABLE_CAUSAL",
            "evidence": "K4 preserves request/passenger identity and request completion/cancellation chronology.",
            "derivation_contract": "completed eligible service requests divided by all eligible requests created within the frozen causal cohort, with terminal censoring reported separately",
            "remaining_requirement": "Implement and test a PV8 transition collector; do not use board+alight event counts as a double-counted served numerator.",
        },
        {
            "component": "avg_wait",
            "weight_key": "w_avg_wait",
            "weight": 2.0,
            "classification": "DERIVABLE_CAUSAL",
            "evidence": "K4 request/passenger timestamps support request-to-board wait and censored waiting duration.",
            "derivation_contract": "mean wait from request/waiting timestamp to board timestamp; unboarded requests remain right-censored and may not be dropped",
            "remaining_requirement": "Implement collector and obtain a PV8-compatible B1 normalization constant.",
        },
        {
            "component": "p95_wait",
            "weight_key": "w_long_wait",
            "weight": 3.0,
            "classification": "DERIVABLE_CAUSAL",
            "evidence": "K4 passenger-level identity and timestamps can retain the wait distribution.",
            "derivation_contract": "pre-registered percentile estimator over exact and explicitly censored passenger waits",
            "remaining_requirement": "Freeze censoring/percentile semantics and obtain a PV8-compatible B1 normalization constant.",
        },
        {
            "component": "on_time_rate",
            "weight_key": "w_ontime",
            "weight": 1.0,
            "classification": "NOT_AVAILABLE",
            "evidence": "No approved PV8 route timetable, service-window, or promised pickup/dropoff ETA is bound to K4 requests.",
            "derivation_contract": None,
            "remaining_requirement": "Provide one authoritative punctuality target and bind it prospectively to request/stop events, or separately approve candidate removal/zero weight.",
        },
        {
            "component": "bunching",
            "weight_key": "w_bunching",
            "weight": 1.5,
            "classification": "SEMANTICS_REPAIR_REQUIRED",
            "evidence": "The existing horizon aggregator explicitly requires an external canonical headway aggregator.",
            "derivation_contract": None,
            "remaining_requirement": "Define route-direction-stop occurrence arrival events, comparison cohort, threshold, and inactive/reappearance handling.",
        },
        {
            "component": "headway_cv",
            "weight_key": "w_headway_cv",
            "weight": 1.0,
            "classification": "SEMANTICS_REPAIR_REQUIRED",
            "evidence": "K9 has decision cycles but no approved canonical same-route headway series.",
            "derivation_contract": None,
            "remaining_requirement": "Implement a causal occurrence-aware headway series and freeze zero-mean/low-count behavior.",
        },
        {
            "component": "energy_per_passenger",
            "weight_key": "w_energy_per_passenger",
            "weight": 1.0,
            "classification": "SEMANTICS_REPAIR_REQUIRED",
            "evidence": "The current horizon aggregator labels energy FORMULA_NOT_SELECTED; historical 10.0 is only an unapproved config default.",
            "derivation_contract": None,
            "remaining_requirement": "Approve an energy model/proxy, exact served-passenger denominator, zero-denominator behavior, and compatible baseline.",
        },
        {
            "component": "fleet_reduction",
            "weight_key": "w_fleet_reduction",
            "weight": 0.5,
            "classification": "BASELINE_CONSTANT_REQUIRED",
            "evidence": "Fixed physical slots and active_bus_mask are exact, but reduction is defined only relative to an approved reference fleet/service plan.",
            "derivation_contract": "approved reference active fleet minus candidate active fleet, divided by approved reference active fleet, only after service guards pass",
            "remaining_requirement": "Freeze the PV8 B1 reference fleet and effective time scope.",
        },
        {
            "component": "intervention",
            "weight_key": "w_intervention",
            "weight": 0.25,
            "classification": "SEMANTICS_REPAIR_REQUIRED",
            "evidence": "K-mask restrictions, policy actions, safety overrides, reassignments, and simulator recovery are not one causal event class.",
            "derivation_contract": None,
            "remaining_requirement": "Implement the explicit taxonomy in r2ar1_intervention_semantics.json and approve which event classes carry cost.",
        },
        {
            "component": "constraint_violation",
            "weight_key": "w_constraint",
            "weight": 10.0,
            "classification": "SEMANTICS_REPAIR_REQUIRED",
            "evidence": "Candidate thresholds exist, but PV8 B1 constants and hard-fail-versus-penalty behavior are not approved.",
            "derivation_contract": None,
            "remaining_requirement": "Approve each threshold and mutually exclusive hard-fail/penalty semantics before producing a value.",
        },
    ]
    invalid = [row for row in rows if row["classification"] not in ALLOWED_INPUT_CLASSES]
    if invalid or len(rows) != 10:
        raise R2AR1Error(f"invalid reward input classification: {invalid}")
    return {
        "created_at": iso_kst(),
        "contract": "PV8_R2AR1_REWARD_INPUT_CLOSURE_V1",
        "allowed_classifications": ALLOWED_INPUT_CLASSES,
        "records": rows,
        "classification_counts": {
            value: sum(row["classification"] == value for row in rows)
            for value in ALLOWED_INPUT_CLASSES
        },
        "usable_after_collector_implementation": ["service_rate", "avg_wait", "p95_wait"],
        "currently_materializable_under_full_frozen_formula": False,
        "silent_zero_fill_allowed": False,
        "reward_values_materialized": False,
    }


def b1_normalization_audit() -> Dict[str, Any]:
    legacy_contract_path = LEGACY_B1_ROOT / "hard_constraint_contract_v2_resolved.json"
    legacy_registry_path = LEGACY_B1_ROOT / "baseline_registry_reference.json"
    legacy_gate_path = LEGACY_B1_ROOT / "prompt5_e01_r2a_gate.json"
    legacy_contract = k5.read_json(legacy_contract_path)
    legacy_registry = k5.read_json(legacy_registry_path)
    legacy_gate = k5.read_json(legacy_gate_path)
    legacy_avg = legacy_contract.get("B1_noop_repaired_avg_wait_seconds_reference")
    legacy_p95 = legacy_contract.get("B1_noop_repaired_p95_wait_seconds_reference")
    legacy_noop = legacy_registry.get("action_contract", {}).get("B1_noop", {})
    legacy_skip = legacy_registry.get("action_contract", {}).get("B2_rulebased_calibrated", {})
    rows = [
        {
            "component": "avg_wait",
            "reference_value": legacy_avg,
            "unit": "seconds",
            "source": str(legacy_contract_path),
            "scope": "legacy repaired B1 no-op feasibility rollout",
            "effective_period": "artifact created 2026-07-20; underlying service period not promoted to PV8",
            "derivation": "legacy no-op baseline mean wait; downstream cap formula min(reference*1.10, 900)",
            "compatibility": "INCOMPATIBLE_WITH_PV8",
            "adopted_for_pv8": False,
        },
        {
            "component": "p95_wait",
            "reference_value": legacy_p95,
            "unit": "seconds",
            "source": str(legacy_contract_path),
            "scope": "legacy repaired B1 no-op feasibility rollout",
            "effective_period": "artifact created 2026-07-20; underlying service period not promoted to PV8",
            "derivation": "legacy no-op baseline p95 wait; downstream cap formula min(reference*1.15, 900)",
            "compatibility": "INCOMPATIBLE_WITH_PV8",
            "adopted_for_pv8": False,
        },
        {
            "component": "energy_per_passenger",
            "reference_value": None,
            "unit": None,
            "source": None,
            "scope": None,
            "effective_period": None,
            "derivation": None,
            "compatibility": "NO_COMPATIBLE_CONSTANT_FOUND",
            "adopted_for_pv8": False,
        },
        {
            "component": "fleet_reduction",
            "reference_value": None,
            "unit": "physical active vehicles",
            "source": None,
            "scope": None,
            "effective_period": None,
            "derivation": None,
            "compatibility": "NO_PV8_REFERENCE_FLEET_FROZEN",
            "adopted_for_pv8": False,
        },
    ]
    reasons = [
        f"Legacy B1 uses discrete_runtime_action={legacy_noop.get('discrete_runtime_action')} while PV8 actions are HOLD/SERVE/CONDITIONAL_SKIP with action_dim=3.",
        f"Legacy B2 names SKIP_STOP action id {legacy_skip.get('skip_stop_action_id')}, which is not the frozen PV8 action contract.",
        "Legacy feasibility gate is BLOCKED_SIMULATOR_OR_FLEET_INFEASIBLE and uses a separately preregistered absolute 900-second cap.",
        "No evidence binds legacy episodes, fixed physical slots, K4 obligations, K8 rulebook, or K9 lifecycle to the PV8 prospective dataset.",
    ]
    return {
        "created_at": iso_kst(),
        "contract": "PV8_R2AR1_B1_NORMALIZATION_AUDIT_V1",
        "b1_identity_authority": "Step113 identity only; numeric values unlocked",
        "legacy_numeric_evidence_found": True,
        "legacy_source_integrity": {
            "contract_sha256": k5.sha256_file(legacy_contract_path),
            "registry_sha256": k5.sha256_file(legacy_registry_path),
            "gate_sha256": k5.sha256_file(legacy_gate_path),
            "legacy_gate_status": legacy_gate.get("status"),
        },
        "records": rows,
        "legacy_values_rejected_for_pv8_reasons": reasons,
        "historical_defaults_300_600_10_used": False,
        "pv8_compatible_numeric_constant_count": 0,
        "normalization_ready": False,
        "minimum_repair": "Run an approved prospective PV8 B1 no-op baseline with the frozen 8-slot, K4/K8/K9, action, episode, and causal outcome contracts; then freeze constants before ablation.",
    }


def on_time_semantics() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract": "PV8_R2AR1_ON_TIME_SEMANTICS_AUDIT_V1",
        "component": "on_time_rate",
        "frozen_weight": 1.0,
        "classification": "NOT_AVAILABLE",
        "fixed_route_candidate": {
            "definition": "arrival/departure within an approved route-direction-stop timetable or service window",
            "required_inputs": ["scheduled_stop_ts_or_window", "actual_causal_arrival_or_departure_ts", "route_stop_occurrence_id"],
            "available_now": False,
        },
        "request_service_candidate": {
            "definition": "pickup/dropoff within a promised ETA or service window frozen at or before request assignment",
            "required_inputs": ["promise_created_ts", "promised_pickup_or_dropoff_window", "actual_board_or_alight_ts", "request_id"],
            "available_now": False,
        },
        "fixed_route_and_drt_semantics_may_be_mixed": False,
        "selected_semantics": None,
        "silent_weight_change": False,
        "candidate_change_requiring_later_approval": "Remove the component or set w_ontime=0 only if no prospective authoritative punctuality target is supplied.",
        "reward_use_ready": False,
    }


def intervention_semantics() -> Dict[str, Any]:
    records = [
        {
            "event_class": "K_SAFETY_ACTION_MASK_RESTRICTION",
            "counts_as_intervention": False,
            "reason": "Availability restriction is a safety precondition, not a policy burden or executed action.",
        },
        {
            "event_class": "ORDINARY_VALID_POLICY_ACTION",
            "counts_as_intervention": False,
            "reason": "HOLD, SERVE, and legal CONDITIONAL_SKIP are not penalized by action name.",
        },
        {
            "event_class": "FORCED_SAFETY_OVERRIDE",
            "counts_as_intervention": "CANDIDATE_TRUE_REQUIRES_APPROVAL",
            "reason": "Count only when an invalid proposal is replaced by a different explicit fallback and both intended/executed actions are logged.",
        },
        {
            "event_class": "REQUEST_REASSIGNMENT",
            "counts_as_intervention": False,
            "reason": "Keep as a separate service operation unless a later causal-cost contract explicitly promotes it.",
        },
        {
            "event_class": "SIMULATOR_RECOVERY",
            "counts_as_intervention": False,
            "reason": "Recovery/turnaround is simulator health telemetry, not policy intervention; it may trigger a hard validity failure.",
        },
        {
            "event_class": "EXTERNAL_POLICY_INTERVENTION",
            "counts_as_intervention": "CANDIDATE_TRUE_REQUIRES_APPROVAL",
            "reason": "Must be an explicit, identity-preserving dispatch/control event beyond ordinary valid actions.",
        },
    ]
    return {
        "created_at": iso_kst(),
        "contract": "PV8_R2AR1_INTERVENTION_SEMANTICS_V1",
        "component": "intervention",
        "frozen_weight": 0.25,
        "classification": "SEMANTICS_REPAIR_REQUIRED",
        "ordinary_k_mask_restriction_penalized": False,
        "action_equals_skip_penalized": False,
        "action_not_equal_hold_penalized": False,
        "records": records,
        "proposed_numerator": "approved forced safety overrides plus approved external policy intervention events",
        "proposed_denominator": "active policy decisions with actor sampling attempted",
        "zero_denominator_policy": "NOT_APPLICABLE_TO_TRANSITION; report null and exclude transition rather than zero-fill",
        "collector_implemented": False,
        "semantics_approved": False,
        "reward_use_ready": False,
    }


def constraint_repair_audit() -> Dict[str, Any]:
    records = [
        {
            "candidate": "service floor = 0.95",
            "classification": "REPAIR_REQUIRED",
            "reason": "Historical candidate exists, but no PV8 causal stress/feasibility evidence validates 0.95.",
            "minimum_repair": "Evaluate prospectively under frozen PV8 B1 and stress scenarios before approval.",
        },
        {
            "candidate": "avg_wait cap = 1.10 x approved reference",
            "classification": "REPAIR_REQUIRED",
            "reason": "No compatible PV8 B1 average-wait reference exists.",
            "minimum_repair": "Freeze PV8 B1 average-wait constant and test cap behavior without the legacy 900-second truncation unless separately approved.",
        },
        {
            "candidate": "p95_wait cap = 1.15 x approved reference",
            "classification": "REPAIR_REQUIRED",
            "reason": "No compatible PV8 B1 p95-wait reference or approved censoring estimator exists.",
            "minimum_repair": "Freeze percentile/censoring semantics and PV8 B1 p95 reference.",
        },
        {
            "candidate": "bunching/on-time guards",
            "classification": "REPAIR_REQUIRED",
            "reason": "Neither metric has an approved PV8 canonical aggregator/target.",
            "minimum_repair": "Resolve metric semantics and thresholds, or explicitly remove affected frozen components in a later approval step.",
        },
        {
            "candidate": "energy guard",
            "classification": "REPAIR_REQUIRED",
            "reason": "The service-first ordering is sound, but energy formula, reference, and threshold are absent.",
            "minimum_repair": "Approve proxy/formula and require all service/wait guards to pass before energy contributes.",
        },
        {
            "candidate": "fleet guard",
            "classification": "REPAIR_REQUIRED",
            "reason": "The service-first ordering is sound, but no PV8 reference fleet and eligibility rule are frozen.",
            "minimum_repair": "Freeze reference fleet and make fleet bonus zero whenever a service/wait guard fails.",
        },
        {
            "candidate": "constraint penalty = 10.0",
            "classification": "REPAIR_REQUIRED",
            "reason": "Frozen magnitude is unchanged, but hard failure versus numeric penalty is not resolved and must not be double counted.",
            "minimum_repair": "Partition invalid transitions (hard fail/no reward) from valid threshold violations (candidate penalty) and approve the partition.",
        },
        {
            "candidate": "K-safety invalid-action prevention",
            "classification": "SUPPORTED",
            "reason": "K8/K9 mask invalid CONDITIONAL_SKIP before sampling; this remains outside reward shaping.",
            "minimum_repair": None,
        },
        {
            "candidate": "critical input missing or NaN/Inf",
            "classification": "SUPPORTED",
            "reason": "Fail closed and produce no reward value; silent zero-fill is forbidden.",
            "minimum_repair": None,
        },
    ]
    invalid = [row for row in records if row["classification"] not in ALLOWED_CONSTRAINT_CLASSES]
    if invalid:
        raise R2AR1Error(f"invalid constraint classification: {invalid}")
    return {
        "created_at": iso_kst(),
        "contract": "PV8_R2AR1_CONSTRAINT_REPAIR_AUDIT_V1",
        "allowed_classifications": ALLOWED_CONSTRAINT_CLASSES,
        "records": records,
        "classification_counts": {
            value: sum(row["classification"] == value for row in records)
            for value in ALLOWED_CONSTRAINT_CLASSES
        },
        "candidate_structurally_invalid": False,
        "hard_constraints_approved": False,
        "constraint_values_materialized": False,
    }


def delayed_harm_contract() -> Dict[str, Any]:
    scenarios = [
        {
            "scenario_id": "S01_BENEFICIAL_SKIP",
            "kind": "beneficial SKIP",
            "decision_state": "all K4 obligations clear, static rules clear, valid post-skip path",
            "causal_schedule": "no request at skipped occurrence before the comparison vehicle can return; downstream service remains protected",
            "expected_mask": "SKIP_AVAILABLE",
            "required_reward_assertion": "SKIP reward may be positive only through measured service outcomes, never action identity",
        },
        {
            "scenario_id": "S02_IMMEDIATE_HARMFUL_SKIP",
            "kind": "immediate harmful SKIP",
            "decision_state": "pickup/boarding obligation already exists at decision_ts",
            "causal_schedule": "obligation is visible before action",
            "expected_mask": "SKIP_BLOCKED",
            "required_reward_assertion": "scenario validates K-safety; an illegal SKIP must not execute or create a reward sample",
        },
        {
            "scenario_id": "S03_ONE_STEP_DELAYED_HARM",
            "kind": "1-step delayed harm",
            "decision_state": "legal SKIP at decision_ts",
            "causal_schedule": "request arrives after decision_ts and harm is first measurable at the next global transition",
            "expected_mask": "SKIP_AVAILABLE_AT_DECISION",
            "required_reward_assertion": "H1 must make net SKIP advantage non-positive once service/wait harm is measured",
        },
        {
            "scenario_id": "S04_MULTI_STEP_DELAYED_HARM",
            "kind": "multi-step delayed harm",
            "decision_state": "legal SKIP at decision_ts",
            "causal_schedule": "request arrives after decision_ts; missed/revisit delay is first measurable after more than one transition",
            "expected_mask": "SKIP_AVAILABLE_AT_DECISION",
            "required_reward_assertion": "H1 is insufficient; bounded Hk must include the first measurable harm and downstream completion/censoring effect",
        },
        {
            "scenario_id": "S05_PICKUP_OBLIGATION",
            "kind": "pickup obligation",
            "decision_state": "assigned pickup at current occurrence",
            "causal_schedule": "assignment timestamp <= decision_ts",
            "expected_mask": "SKIP_BLOCKED",
            "required_reward_assertion": "no illegal action reward sample",
        },
        {
            "scenario_id": "S06_DROPOFF_ONBOARD_OBLIGATION",
            "kind": "dropoff/onboard obligation",
            "decision_state": "onboard destination or assigned dropoff at current occurrence",
            "causal_schedule": "board/assignment timestamp <= decision_ts",
            "expected_mask": "SKIP_BLOCKED",
            "required_reward_assertion": "no illegal action reward sample",
        },
        {
            "scenario_id": "S07_SERVICE_RATE_DEGRADATION",
            "kind": "service-rate degradation",
            "decision_state": "legal action at decision_ts",
            "causal_schedule": "paired SKIP branch leaves one eligible request incomplete/cancelled relative to comparator",
            "expected_mask": "SKIP_AVAILABLE_AT_DECISION",
            "required_reward_assertion": "service floor/penalty must prevent positive harmful SKIP reward within selected horizon",
        },
        {
            "scenario_id": "S08_P95_WAIT_DEGRADATION",
            "kind": "p95-wait degradation",
            "decision_state": "legal action at decision_ts",
            "causal_schedule": "tail passenger wait crosses the approved p95 cap only after multiple transitions",
            "expected_mask": "SKIP_AVAILABLE_AT_DECISION",
            "required_reward_assertion": "selected horizon must expose tail harm and prevent positive net harmful SKIP reward",
        },
    ]
    return {
        "created_at": iso_kst(),
        "contract": "PV8_R2AR1_DELAYED_HARM_TEST_CONTRACT_V1",
        "status": "DESIGNED_NOT_EXECUTABLE_UNTIL_INPUT_CLOSURE",
        "paired_branch_contract": {
            "same_initial_snapshot": True,
            "same_exogenous_event_schedule": True,
            "treatment": "legal CONDITIONAL_SKIP",
            "comparator": "SERVE or HOLD selected by frozen scenario contract",
            "only_committed_action_differs_at_decision_ts": True,
            "future_events_visible_to_observation_or_mask": False,
            "future_events_may_determine_reward": True,
        },
        "candidate_horizons": {
            "H1": "(decision_ts, next global transition]",
            "Hk": "bounded causal window ending when every affected request is terminal/served/cancelled and the affected vehicle has rejoined the comparator path, capped by episode terminal",
        },
        "scenarios": scenarios,
        "h1_vulnerability": "STRUCTURALLY_VULNERABLE_TO_MULTI_STEP_DELAYED_HARM",
        "minimum_horizon_selection_rule": "After the scenario event schedules and a route-revisit upper bound are frozen, set k to the maximum first-harm-reveal transition index across all frozen stress scenarios; retain longer terminal/censoring accounting when needed.",
        "numeric_k_selected": None,
        "numeric_k_not_selected_reason": "PV8 has no causal action-outcome rollout, reward collector, frozen exogenous stress schedules, or approved route-revisit bound from which to derive k.",
        "arbitrary_k_allowed": False,
        "minimum_horizon_status": "NOT_IDENTIFIED_ADDITIONAL_CAUSAL_STRESS_INPUT_REQUIRED",
        "harmful_skip_positive_net_reward_test_executed": False,
        "policy_learning_executed": False,
    }


def reward_candidate_delta() -> Dict[str, Any]:
    proposed = [
        {
            "component": "on_time_rate",
            "candidate_change": "remove component or set w_ontime=0 if no authoritative target is supplied",
            "applied": False,
            "requires_explicit_approval": True,
        },
        {
            "component": "intervention",
            "candidate_change": "retain only explicitly logged forced override/external intervention classes, or set weight to zero until implemented",
            "applied": False,
            "requires_explicit_approval": True,
        },
        {
            "component": "bunching/headway_cv/energy",
            "candidate_change": "implement approved causal/proxy semantics before use; removal or zero weight is a separate approval",
            "applied": False,
            "requires_explicit_approval": True,
        },
    ]
    return {
        "created_at": iso_kst(),
        "contract": "PV8_R2AR1_REWARD_CANDIDATE_DELTA_V1",
        "candidate_id": "R1_BALANCED_DEFAULT",
        "frozen_weights_before": EXPECTED_WEIGHTS,
        "frozen_weights_after": EXPECTED_WEIGHTS,
        "weight_change_count": 0,
        "formula_change_count": 0,
        "direct_action_shaping_added": False,
        "proposed_later_changes_not_applied": proposed,
        "reward_contract_approved": False,
    }


def readiness_decision(
    inputs: Mapping[str, Any],
    normalization: Mapping[str, Any],
    on_time: Mapping[str, Any],
    intervention: Mapping[str, Any],
    constraints: Mapping[str, Any],
    delayed: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers = [
        "No PV8-compatible B1 numeric normalization constants are available; legacy B1 values use incompatible action/simulator contracts.",
        "on_time_rate has no approved timetable, service-window, or promised-ETA target and remains NOT_AVAILABLE at frozen weight 1.0.",
        "intervention event semantics are specified as a candidate taxonomy but are not approved or implemented at frozen weight 0.25.",
        "Bunching/headway and energy require canonical causal/proxy semantics and collectors.",
        "Service/wait constraints and the 10.0 violation penalty are not approved for PV8.",
        "H1 is structurally vulnerable to multi-step delayed harm; numeric Hk cannot be derived before prospective stress schedules and route-revisit bounds exist.",
    ]
    return {
        "created_at": iso_kst(),
        "final_decision": DECISION,
        "audit_complete": True,
        "candidate_structurally_invalid": False,
        "repair_ready_for_ablation": False,
        "additional_input_required": True,
        "reward_components_usable_after_collector": inputs["usable_after_collector_implementation"],
        "pv8_compatible_normalization_constant_count": normalization["pv8_compatible_numeric_constant_count"],
        "on_time_resolution": on_time["classification"],
        "intervention_resolution": intervention["classification"],
        "hard_constraint_repair_required_count": constraints["classification_counts"]["REPAIR_REQUIRED"],
        "minimum_horizon_status": delayed["minimum_horizon_status"],
        "blockers": blockers,
        "minimum_next_scope": [
            "Implement a no-training causal reward outcome collector for service rate and censored wait metrics.",
            "Generate a prospective frozen PV8 B1 no-op reference under the exact K4/K8/K9 and 8-slot contracts.",
            "Supply/approve on-time targets or explicitly approve its removal/zero weight; approve intervention taxonomy and collector.",
            "Freeze deterministic paired delayed-harm schedules and route-revisit bound, then run H1 versus derived Hk reward ablation without policy learning.",
            "Approve component semantics, normalization constants, constraints, horizon, and any weight delta only after that evidence.",
        ],
        "r2ar2_or_r2b_or_r3_authorized": False,
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
        "automatic_r2ar2_execution_authorized": False,
        "automatic_r2b_execution_authorized": False,
        "automatic_r3_execution_authorized": False,
        "mappo_training_authorized": False,
    }


def final_report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# PV8-R2A-R1 Reward Contract Repair and Delayed-Harm Readiness",
        "",
        f"- artifact root: `{summary['artifact_root']}`",
        f"- gate: `{summary['gate']}`",
        f"- decision: `{summary['decision']}`",
        "- frozen candidate: `R1_BALANCED_DEFAULT`, unchanged",
        "- reward approval / value materialization / training: `false / false / false`",
        "",
        "## Reward Inputs",
        "",
        "Service rate, average wait, and p95 wait are causally derivable from the K4 identity/timestamp lifecycle after an explicit collector is implemented. They are not materialized here. On-time is unavailable. Bunching, headway-CV, energy, intervention, and constraint semantics still require repair; fleet reduction requires an approved PV8 reference constant.",
        "",
        "## B1 Normalization",
        "",
        "A legacy B1 artifact contains average wait 3812.8040558761663 seconds and p95 wait 39600.0 seconds. It used discrete no-op action 1, SKIP_STOP action 3, a different simulator/fleet feasibility contract, and a separate 900-second absolute cap. These values are recorded for lineage but rejected as PV8 constants. Historical 300/600/10 defaults were not used. Supported PV8 numeric constants: 0.",
        "",
        "## On-Time and Intervention",
        "",
        "No timetable, service-window, or promised-ETA target is bound to PV8 request/occurrence events, so on-time remains NOT_AVAILABLE. Ordinary K-mask restriction and ordinary valid HOLD/SERVE/CONDITIONAL_SKIP actions are not intervention penalties. Only explicit forced override or external policy intervention classes are candidate costs, pending approval and implementation.",
        "",
        "## Constraints",
        "",
        "K-safety prevention and missing/invalid-input fail-closed behavior are supported. The 0.95 service floor, 1.10 average-wait cap, 1.15 p95-wait cap, bunching/on-time guards, energy/fleet guards, and constraint penalty behavior all require PV8 evidence or approval.",
        "",
        "## Delayed Harm",
        "",
        "H1 is structurally unable to observe a harm that first appears after the next global transition. The deterministic paired stress contract freezes beneficial, immediate, one-step, multi-step, obligation, service-rate, and p95-wait cases. Numeric Hk is deliberately not guessed: it must be derived from frozen exogenous schedules and the maximum first-harm-reveal transition index, with a pre-registered route-revisit/terminal bound.",
        "",
        "## Decision",
        "",
        "The candidate is not structurally invalid, but it is not ready for ablation. A PV8 B1 run, causal outcome collector, punctuality/intervention decisions, canonical headway/energy semantics, constraint approval, and an executable delayed-harm fixture set are required before explicit reward approval.",
        "",
        "No API/DB access, reward value, episode expansion, policy execution/evaluation, checkpoint reuse, downstream run, or MAPPO training occurred.",
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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar1.jsonl"
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
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar1.json"
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
    writer.json("_PV8_R2AR1_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": manifest_name,
        "final_manifest_sha256": k5.sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })


def run_audit(root: Path) -> Path:
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    upstreams = verify_upstreams()
    frozen = verify_frozen_contract()
    inputs = reward_input_closure()
    normalization = b1_normalization_audit()
    on_time = on_time_semantics()
    intervention = intervention_semantics()
    constraints = constraint_repair_audit()
    delayed = delayed_harm_contract()
    delta = reward_candidate_delta()
    decision = readiness_decision(inputs, normalization, on_time, intervention, constraints, delayed)
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
    summary = {"artifact_root": str(root), "gate": PASS_GATE, "decision": decision["final_decision"]}

    writer.json("r2ar1_reward_input_closure.json", inputs)
    writer.json("r2ar1_b1_normalization_audit.json", normalization)
    writer.json("r2ar1_on_time_semantics.json", on_time)
    writer.json("r2ar1_intervention_semantics.json", intervention)
    writer.json("r2ar1_constraint_repair_audit.json", constraints)
    writer.json("r2ar1_delayed_harm_test_contract.json", delayed)
    writer.json("r2ar1_reward_candidate_delta.json", delta)
    writer.json("r2ar1_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
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
        "frozen_contract_audit": frozen,
        "audit_payload_sha256": canonical_hash({
            "inputs": inputs,
            "normalization": normalization,
            "on_time": on_time,
            "intervention": intervention,
            "constraints": constraints,
            "delayed": delayed,
            "delta": delta,
            "decision": decision,
        }),
        "new_bis_api_call_count": 0,
        "db_query_count": 0,
        "db_write_count": 0,
        "reward_value_materialization_count": 0,
        "episode_expansion_count": 0,
        "policy_action_execution_count": 0,
        "policy_evaluation_count": 0,
        "checkpoint_reuse_count": 0,
        "r2ar2_execution_count": 0,
        "r2b_execution_count": 0,
        "r3_execution_count": 0,
        "mappo_training_count": 0,
        "qwen_train": False,
        "qwen_inference": False,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {
        **guards,
        "source_gate": PASS_GATE,
        "readiness": READINESS,
        "final_decision": decision["final_decision"],
    })
    writer.text("final_report.md", final_report(summary))
    write_manifest_and_lock(writer, gate)

    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar1.json", "_PV8_R2AR1_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise R2AR1Error(f"R2A-R1 artifact integrity failure: {own}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {decision['final_decision']}")
    print(f"pv8_compatible_normalization_constant_count: {normalization['pv8_compatible_numeric_constant_count']}")
    print(f"numeric_hk_selected: {delayed['numeric_k_selected']}")
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
