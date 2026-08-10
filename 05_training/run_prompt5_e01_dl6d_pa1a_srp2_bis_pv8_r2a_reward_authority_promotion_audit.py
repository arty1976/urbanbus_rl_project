#!/usr/bin/env python3
"""PV8-R2A reward authority recovery and trainable-contract promotion audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping
from zoneinfo import ZoneInfo

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
REWARDS_ROOT = TRAINING_ROOT / "rewards"
ADAPTERS_ROOT = TRAINING_ROOT / "adapters"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2a_reward_authority_promotion_audit.py"

K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"
K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"
R1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r1_mappo_retraining_preflight_20260808_142604"
R2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2_reward_episode_audit_20260808_143743"

UPSTREAMS = {
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
    "PV8-K9": (K9_ROOT, "artifact_manifest_srp2_bis_pv8_k9.json", "_PV8_K9_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED"),
    "PV8-R1": (R1_ROOT, "artifact_manifest_srp2_bis_pv8_r1.json", "_PV8_R1_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R1_MAPPO_RETRAINING_PREFLIGHT_COMPLETE"),
    "PV8-R2": (R2_ROOT, "artifact_manifest_srp2_bis_pv8_r2.json", "_PV8_R2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2_REWARD_AND_EPISODE_DATA_AUDIT_COMPLETE"),
}

EXPECTED_RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
EXPECTED_OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"
EXPECTED_TENSORS = {"num_agents": 8, "actor_obs_dim": 16, "critic_obs_dim": 64, "action_dim": 3}

STEP108 = ADAPTERS_ROOT / "queue_demand_reward_wiring_step108.py"
STEP109 = ADAPTERS_ROOT / "reward_policy_interface_step109.py"
STEP110 = ADAPTERS_ROOT / "final_reward_design_review_gate_step110.py"
STEP111 = REWARDS_ROOT / "final_reward_spec_step111.json"
STEP112 = REWARDS_ROOT / "reward_candidate_protocol_step112.json"
STEP113 = REWARDS_ROOT / "reward_normalization_baseline_lock_step113.json"
STEP114 = REWARDS_ROOT / "hard_constraint_review_step114.json"
STEP115 = REWARDS_ROOT / "reward_ablation_matrix_step115.json"
STEP116 = REWARDS_ROOT / "trainable_reward_promotion_gate_step116.json"
STEP127 = REWARDS_ROOT / "trainable_reward_promotion_decision_package_step127.json"
STEP128 = REWARDS_ROOT / "reward_pipeline_status_actual_ablation_wait_step128.json"
STEP129 = REWARDS_ROOT / "actual_reward_ablation_runbook_step129.json"
STEP130 = REWARDS_ROOT / "reward_pipeline_project_log_update_step130.json"
STEP138 = REWARDS_ROOT / "reward_execution_gate_project_log_update_step138.json"
STEP141 = REWARDS_ROOT / "baseline_reference_manifest_generator_step141.py"
HISTORICAL_REWARD_CODE = REWARDS_ROOT / "mappo_reward_v1.py"
HISTORICAL_REWARD_CONFIG = REWARDS_ROOT / "reward_config_v1.yaml"

HISTORICAL_SOURCES = [
    (108, STEP108, "SCAFFOLD"),
    (109, STEP109, "SCAFFOLD_INTERFACE"),
    (110, STEP110, "REVIEW_GUARD"),
    (111, STEP111, "DRAFT_NOT_TRAINABLE"),
    (112, STEP112, "CANDIDATE_PROTOCOL_NOT_SELECTED"),
    (113, STEP113, "BASELINE_IDENTITY_LOCK_NUMERIC_VALUES_UNLOCKED"),
    (114, STEP114, "CONSTRAINT_REVIEW_NOT_FINAL"),
    (115, STEP115, "ABLATION_MATRIX_NOT_SELECTED"),
    (116, STEP116, "PROMOTION_GATE_NOT_PROMOTED"),
    (127, STEP127, "DECISION_PACKAGE_NOT_PROMOTED"),
    (128, STEP128, "ACTUAL_ABLATION_WAIT"),
    (129, STEP129, "ACTUAL_ABLATION_RUNBOOK_NOT_EXECUTED"),
    (130, STEP130, "PROJECT_LOG_NOT_PROMOTED"),
    (138, STEP138, "EXECUTION_GATE_NOT_PROMOTED"),
    (141, STEP141, "NONCAUSAL_BASELINE_REFERENCE_ONLY"),
    (None, HISTORICAL_REWARD_CODE, "HISTORICAL_IMPLEMENTATION_NOT_PV8_PROMOTED"),
    (None, HISTORICAL_REWARD_CONFIG, "HISTORICAL_CONFIG_NOT_PV8_PROMOTED"),
]

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2a_reward_authority_promotion_audit"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2A_REWARD_AUTHORITY_AND_PROMOTION_AUDIT_COMPLETE"
READINESS = "SRP2_BIS_PV8_R2A_COMPLETE_REWARD_CONTRACT_REPAIR_REQUIRED_APPROVAL_LOCKED"
DECISION_REPAIR = "PV8_REWARD_CONTRACT_REPAIR_REQUIRED"

PAYLOADS = [
    "r2a_reward_source_authority_audit.json",
    "r2a_historical_reward_lineage.json",
    "r2a_pv8_reward_formula_candidate.json",
    "r2a_reward_weight_contract.json",
    "r2a_reward_normalization_contract.json",
    "r2a_hard_constraint_contract.json",
    "r2a_decision_outcome_binding_contract.json",
    "r2a_reward_input_availability.json",
    "r2a_reward_hacking_alignment_audit.json",
    "r2a_reward_promotion_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]

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


class R2AError(RuntimeError):
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
            raise R2AError(f"{label} integrity failure: gate={observed}, checks={checks}")
        out[label] = {"artifact_root": str(root), "gate": observed, "manifest_integrity": checks}
    return out


def verify_frozen_bindings() -> Dict[str, Any]:
    tensor = k5.read_json(R1_ROOT / "r1_mappo_tensor_contract.json")
    episode = k5.read_json(R1_ROOT / "r1_frozen_episode_manifest.json")
    k8 = k5.read_json(K8_ROOT / "k8_frozen_rule_contract.json")
    k9 = k5.read_json(K9_ROOT / "k9_version_hash_binding_audit.json")
    r2 = k5.read_json(R2_ROOT / "r2_training_data_readiness.json")
    checks = {
        "tensor_values": {key: tensor.get(key) for key in EXPECTED_TENSORS},
        "tensor_matches": all(tensor.get(key) == value for key, value in EXPECTED_TENSORS.items()),
        "r1_rulebook_matches": episode.get("frozen_bindings", {}).get("rulebook_sha256") == EXPECTED_RULEBOOK_SHA256,
        "r1_occurrence_matches": episode.get("frozen_bindings", {}).get("occurrence_master_sha256") == EXPECTED_OCCURRENCE_SHA256,
        "k8_rulebook_matches": k8.get("static_rulebook_sha256") == EXPECTED_RULEBOOK_SHA256,
        "k8_occurrence_matches": k8.get("occurrence_master_sha256") == EXPECTED_OCCURRENCE_SHA256,
        "k9_rulebook_matches": k9.get("rulebook_sha256") == EXPECTED_RULEBOOK_SHA256,
        "k9_occurrence_matches": k9.get("occurrence_master_sha256") == EXPECTED_OCCURRENCE_SHA256,
        "r2_decision_matches": r2.get("final_decision") == "PV8_REWARD_MATERIALIZATION_BLOCKED",
        "r2_reward_values_locked": r2.get("reward_materialization_ready") is False,
    }
    checks["failure_count"] = sum(
        not value for key, value in checks.items()
        if key.endswith("_matches") or key == "r2_reward_values_locked"
    )
    if checks["failure_count"]:
        raise R2AError(f"frozen contract mismatch: {checks}")
    return checks


def tracked_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "05_training/rewards", "05_training/adapters"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def source_lineage() -> Dict[str, Any]:
    tracked = tracked_files()
    records: List[Dict[str, Any]] = []
    for step, path, authority_class in HISTORICAL_SOURCES:
        relative = str(path.relative_to(PROJECT_ROOT))
        if not path.exists() or relative not in tracked:
            raise R2AError(f"required tracked reward source missing or untracked: {relative}")
        records.append({
            "step": step,
            "path": str(path),
            "relative_path": relative,
            "git_tracked": True,
            "sha256": k5.sha256_file(path),
            "size_bytes": path.stat().st_size,
            "authority_class": authority_class,
            "trainable_reward_promoted": False,
        })
    return {
        "created_at": iso_kst(),
        "tracked_historical_reward_source_count": len(records),
        "records": records,
        "lineage_summary": "Tracked reward design lineage exists from scaffold through promotion gates, but every later gate preserves NOT_PROMOTED.",
    }


def authority_audit(lineage: Mapping[str, Any]) -> Dict[str, Any]:
    step111 = k5.read_json(STEP111)
    step112 = k5.read_json(STEP112)
    step113 = k5.read_json(STEP113)
    step114 = k5.read_json(STEP114)
    step115 = k5.read_json(STEP115)
    step116 = k5.read_json(STEP116)
    step127 = k5.read_json(STEP127)
    step128 = k5.read_json(STEP128)
    mandatory = {
        "step111_reward_formula_finalized_false": step111["claim_guards"]["reward_formula_finalized"] is False,
        "step111_reward_weights_locked_false": step111["claim_guards"]["reward_weights_locked"] is False,
        "step111_normalization_locked_false": step111["claim_guards"]["normalization_locked"] is False,
        "step111_hard_constraints_locked_false": step111["claim_guards"]["hard_constraints_locked"] is False,
        "step111_train_allowed_false": step111["claim_guards"]["train_with_this_reward_allowed"] is False,
        "step112_candidate_selected_false": step112["claim_guards"]["candidate_selected"] is False,
        "step113_numeric_baseline_locked_false": step113["baseline_reference_lock"]["numeric_baseline_values_locked"] is False,
        "step114_constraints_finalized_false": step114["claim_guards"]["hard_constraints_finalized"] is False,
        "step115_weights_locked_false": step115["claim_guards"]["reward_weights_locked"] is False,
        "step116_not_promoted": step116["trainable_reward_promoted"] is False and step116["selected_candidate_id"] is None,
        "step127_not_promoted": step127["decision"]["trainable_reward_promoted"] is False and step127["decision"]["selected_candidate_id"] is None,
        "step128_not_promoted": step128["current_artifact_class"]["promotion_decision"] == "not_promoted",
    }
    if not all(mandatory.values()):
        raise R2AError(f"historical guard mismatch: {mandatory}")
    return {
        "created_at": iso_kst(),
        "historical_reward_authority_found": True,
        "authoritative_historical_reward_artifacts_found": [row["relative_path"] for row in lineage["records"]],
        "mandatory_historical_guards": mandatory,
        "prior_trainable_reward_actually_promoted": False,
        "later_authoritative_promotion_found": False,
        "selected_historical_candidate_id": None,
        "actual_ablation_results_found": False,
        "source_authority_sufficient_for_candidate_recovery": True,
        "source_authority_sufficient_for_trainable_promotion": False,
        "reason": "The tracked lineage is authoritative for recovering design intent and exact candidate values, but explicitly denies selection and promotion.",
    }


def formula_candidate() -> Dict[str, Any]:
    step111 = k5.read_json(STEP111)
    formula = step111["reward_formula_draft"]
    weights = step111["draft_weight_candidates_not_locked"]
    if weights != EXPECTED_WEIGHTS:
        raise R2AError(f"Step111 balanced weights changed: {weights}")
    if "action" in formula["formula_text"].lower() or "skip" in formula["formula_text"].lower():
        raise R2AError("historical formula unexpectedly contains direct action/skip reward")
    return {
        "created_at": iso_kst(),
        "candidate_id": "PV8_R2A_R1_BALANCED_DEFAULT_RECOVERY_CANDIDATE",
        "historical_source_candidate": "Step111 draft / Step115 R1_BALANCED_DEFAULT",
        "status": "RECOVERED_CANDIDATE_NOT_SELECTED_NOT_APPROVED",
        "weight_sign_convention": step111["weight_sign_convention"],
        "formula_text": formula["formula_text"],
        "component_order": [
            "+ service_rate", "- avg_wait", "- p95_wait", "+ on_time", "- bunching",
            "- headway_cv", "- energy_per_passenger", "+ fleet_reduction",
            "- intervention", "- constraint_violation",
        ],
        "exact_weights": weights,
        "reward_scope_candidate": "GLOBAL_TEAM_OUTCOME_REWARD_SHARED_BY_ACTIVE_PHYSICAL_SLOTS",
        "reward_scope_approved": False,
        "pv8_compatibility": {
            "num_agents": 8,
            "action_dim": 3,
            "action_names": ["HOLD", "SERVE", "CONDITIONAL_SKIP"],
            "physical_fixed_slots": True,
            "inactive_sampling_bypass": True,
            "k_safety_mask_applied_before_action": True,
            "k4_obligation_state_preserved": True,
            "formula_depends_on_agent_count": False,
            "formula_depends_on_action_id": False,
            "direct_skip_bonus_or_penalty": False,
        },
        "causal_outcome_only": True,
        "missing_input_policy": "FAIL_CLOSED_NO_ZERO_FILL",
        "reward_values_materialized": False,
    }


def weight_contract(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    step115 = k5.read_json(STEP115)
    r1 = next(row for row in step115["candidate_weight_sets"] if row["candidate_id"] == "R1_BALANCED_DEFAULT")
    checks = {key: r1["weights"].get(key) == value for key, value in EXPECTED_WEIGHTS.items()}
    if not all(checks.values()):
        raise R2AError(f"Step111 and Step115 R1 weights disagree: {checks}")
    return {
        "created_at": iso_kst(),
        "contract_name": "PV8_R2A_BALANCED_WEIGHT_RECOVERY_CANDIDATE_V1",
        "candidate_id": candidate["candidate_id"],
        "exact_positive_magnitude_weights": EXPECTED_WEIGHTS,
        "step111_step115_exact_match": checks,
        "weights_changed_by_r2a": False,
        "weights_selected": False,
        "weights_locked": False,
        "weight_approval_required": True,
        "alternative_historical_candidates_retained": [
            row["candidate_id"] for row in step115["candidate_weight_sets"]
            if row["candidate_id"] != "R1_BALANCED_DEFAULT"
        ],
        "selection_claim_allowed": False,
    }


def normalization_contract() -> Dict[str, Any]:
    step113 = k5.read_json(STEP113)
    return {
        "created_at": iso_kst(),
        "contract_name": "PV8_R2A_NORMALIZATION_RECOVERY_CANDIDATE_V1",
        "status": "REPAIR_REQUIRED_NUMERIC_BASELINES_NOT_LOCKED",
        "baseline_identity": "B1_noop",
        "baseline_identity_locked_for_candidate_evaluation": True,
        "numeric_baseline_values_locked": False,
        "normalization_terms_recovered_unchanged": step113["normalization_terms"],
        "baseline_roles": {
            "B0R": "historical reporting comparison only; noncausal and forbidden as PV8 decision-time normalization input",
            "B1": "no-op normalization anchor identity; must be regenerated causally under the frozen PV8 simulator, episodes, seeds, and contract",
            "B2": "rule-based reporting comparator only; forbidden as primary training normalization anchor",
        },
        "historical_default_constants": {
            "avg_wait_seconds": 300.0,
            "passenger_wait_p95_seconds": 600.0,
            "energy_proxy_per_passenger": 10.0,
            "authority": "HISTORICAL_CONFIG_DEFAULT_NOT_APPROVED_PV8_BASELINE",
            "used_by_r2a": False,
        },
        "zero_denominator_policy": "FAIL_CLOSED; no max(served,1), no silent epsilon substitution without a separately approved numeric epsilon contract",
        "reward_total_clipping": {
            "historical_implementation": "NONE",
            "pv8_candidate": "NONE_PROPOSED_NOT_APPROVED",
            "component_bounding_only": ["service_rate", "on_time_rate", "bunching_rate", "fleet_reduction_ratio"],
        },
        "normalization_approved": False,
    }


def hard_constraint_contract() -> Dict[str, Any]:
    step114 = k5.read_json(STEP114)
    return {
        "created_at": iso_kst(),
        "contract_name": "PV8_R2A_HARD_CONSTRAINT_RECOVERY_CANDIDATE_V1",
        "status": "REPAIR_REQUIRED_THRESHOLDS_AND_ACTIONS_NOT_LOCKED",
        "recovered_constraints_unchanged": step114["constraints"],
        "candidate_threshold_summary": {
            "service_rate_floor": 0.95,
            "avg_wait_regression_cap_vs_B1": 1.10,
            "p95_wait_regression_cap_vs_B1": 1.15,
            "qwen_trigger_rate": 0.0,
        },
        "bunching_guard": "REQUIRED_BUT_NUMERIC_THRESHOLD_NOT_DEFINED",
        "on_time_guard": "REQUIRED_BUT_NUMERIC_THRESHOLD_NOT_DEFINED",
        "energy_guard": "energy bonus/penalty may be counted only after service and wait guards pass; numeric baseline missing",
        "fleet_bonus_semantics": "secondary global outcome bonus only after service/wait guards pass; never an action-ID bonus",
        "constraint_penalty_weight": 10.0,
        "constraint_violation_action": "MUST_BE_EXPLICITLY_CHOSEN_BETWEEN_PENALTY_AND_HARD_FAIL",
        "k_safety_illegal_action_handling": "MASK_BEFORE_SAMPLING; never represented as a reward shaping term",
        "missing_critical_input": "HARD_FAIL_TRANSITION_NO_REWARD_VALUE",
        "hard_constraints_approved": False,
    }


def decision_outcome_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract_name": "PV8_R2A_DECISION_TO_CAUSAL_OUTCOME_BINDING_CANDIDATE_V1",
        "status": "DEFINED_FOR_REPAIR_NOT_IMPLEMENTED_NOT_APPROVED",
        "decision_ts": "global orchestrator timestamp captured after all due events are reduced and before observation/K-mask/action sampling",
        "action_t": "executed action after K-mask validation for each active physical slot; null for inactive slots; intended or invalid action is not reward input",
        "outcome_start_ts": "open boundary immediately after the committed action vector at decision_ts",
        "outcome_end_ts": "next global simulator decision boundary for the same episode, or the terminal timestamp if the episode terminates first",
        "reward_measurement_horizon": "(decision_ts, outcome_end_ts] exactly one causal global control transition",
        "next_state_binding": {
            "key": ["episode_id", "cycle_index_plus_one", "agent_id", "vehicle_token"],
            "global_binding": ["source_global_snapshot_hash", "next_global_snapshot_hash"],
            "physical_slot_identity_must_match": True,
            "reappearance_and_direction_transition_preserve_slot": True,
        },
        "terminal_handling": {
            "measure_until_terminal_timestamp": True,
            "bootstrap_value": 0.0,
            "fabricated_next_state_allowed": False,
            "censored_waiting_passengers_must_remain_in_service_and_wait_denominators": True,
        },
        "no_future_leakage": {
            "observation_and_k_mask_cutoff": "events with event_ts <= decision_ts only",
            "future_outcome_allowed_for_reward": True,
            "future_outcome_allowed_in_observation_or_mask": False,
            "pending_event_peek_allowed": False,
        },
        "reward_scope_candidate": "one global team outcome reward shared only among active slots at decision_ts",
        "direct_action_identity_reward_allowed": False,
        "direct_skip_bonus_or_penalty_allowed": False,
        "collector_implemented": False,
        "binding_approved": False,
    }


def input_availability() -> Dict[str, Any]:
    rows = [
        ("decision_ts", "SIMULATOR_EXACT", "K9 global snapshot lifecycle"),
        ("executed_action_t", "SIMULATOR_EXACT", "future runtime transition record after K-mask validation"),
        ("K4_passenger_request_identity_and_timestamps", "SIMULATOR_EXACT", "K4 ServiceRequest/Passenger lifecycle"),
        ("K4_service_obligation_state", "SIMULATOR_EXACT", "K4 decision-time obligation snapshot; safety state only, not a reward bonus"),
        ("passenger_service_rate", "DERIVED_CAUSAL", "derive completed/generated request counts over the bound causal horizon"),
        ("avg_wait_seconds", "DERIVED_CAUSAL", "derive from request/wait/board timestamps with censored passengers retained"),
        ("passenger_wait_p95_seconds", "DERIVED_CAUSAL", "derive weighted percentile from exact lifecycle ledger including censored waits"),
        ("on_time_rate", "NOT_AVAILABLE", "no approved PV8 schedule/timetable-to-event binding"),
        ("bunching_rate", "PROXY_GUARDED", "requires approved simulator arrival/headway event semantics"),
        ("cv_headway", "PROXY_GUARDED", "requires approved simulator arrival/headway event semantics; BIS ETA is not actual headway"),
        ("energy_proxy_per_passenger", "PROXY_GUARDED", "energy is a model proxy and denominator must be exact nonzero served demand"),
        ("fleet_reduction_ratio", "BASELINE_NORMALIZED", "active physical slots versus a causal PV8 B1 no-op baseline"),
        ("intervention_rate", "NOT_AVAILABLE", "must be a causal control burden, not action!=HOLD or action==SKIP"),
        ("constraint_violation", "DERIVED_CAUSAL", "derive after inputs, B1 constants, thresholds, and violation semantics are approved"),
    ]
    classes = ["SIMULATOR_EXACT", "DERIVED_CAUSAL", "BASELINE_NORMALIZED", "PROXY_GUARDED", "NOT_AVAILABLE"]
    return {
        "created_at": iso_kst(),
        "allowed_classifications": classes,
        "records": [{"field": field, "classification": classification, "source_or_reason": reason} for field, classification, reason in rows],
        "classification_counts": {classification: sum(row[1] == classification for row in rows) for classification in classes},
        "critical_not_available_fields": ["on_time_rate", "intervention_rate"],
        "currently_reward_materializable": False,
        "missing_input_zero_fill_allowed": False,
    }


def hacking_audit() -> Dict[str, Any]:
    risks = [
        ("DIRECT_SKIP_SHAPING", "CLOSED_BY_CANDIDATE", "No action identity appears in the formula; SKIP may affect reward only through causal outcome."),
        ("SERVICE_SUPPRESSION_FOR_ENERGY", "OPEN_UNTIL_CONSTRAINT_APPROVAL", "Service floor and wait caps exist only as unlocked candidates."),
        ("ZERO_SERVED_DENOMINATOR", "OPEN_REPAIR_REQUIRED", "Historical max(served,1) fallback is forbidden for PV8; transition must fail closed."),
        ("FLEET_BONUS_OVERRIDES_SERVICE", "OPEN_UNTIL_CONSTRAINT_APPROVAL", "Fleet bonus must be disabled unless all service/wait guards pass."),
        ("INTERVENTION_ACTION_ID_PENALTY", "OPEN_REPAIR_REQUIRED", "Historical intervention semantics are not PV8 causal burden semantics."),
        ("SHORT_HORIZON_DELAYED_SERVICE_HARM", "OPEN_REPAIR_REQUIRED", "One-step reward may miss delayed wait/headway harm; multi-step stress test is required."),
        ("GLOBAL_REWARD_CREDIT_DILUTION", "OPEN_REVIEW_REQUIRED", "Shared team reward is aligned but may weaken per-agent credit assignment."),
        ("TERMINAL_CENSORING", "CONTRACT_GUARDED_NOT_IMPLEMENTED", "Waiting passengers must remain censored in wait/service denominators at terminal."),
        ("PROXY_OVERCLAIM", "CONTRACT_GUARDED", "Headway, bunching, energy, and wait provenance must retain proxy/derived labels."),
        ("FUTURE_LEAKAGE", "CONTRACT_GUARDED_NOT_IMPLEMENTED", "Future outcomes are reward-only and forbidden in the pre-decision observation/K-mask."),
    ]
    return {
        "created_at": iso_kst(),
        "service_alignment_status": "REPAIR_REQUIRED_BEFORE_APPROVAL",
        "direct_skip_bonus_or_penalty": False,
        "records": [{"risk": risk, "status": status, "assessment": assessment} for risk, status, assessment in risks],
        "open_repair_count": sum("OPEN" in status for _, status, _ in risks),
        "reward_hacking_stress_test_executed": False,
        "policy_or_training_executed": False,
    }


def promotion_decision(authority: Mapping[str, Any], inputs: Mapping[str, Any], hacking: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "final_decision": DECISION_REPAIR,
        "audit_complete": True,
        "historical_source_authority_recovered": bool(authority["historical_reward_authority_found"]),
        "prior_trainable_reward_promoted": False,
        "candidate_formula_recovered": True,
        "candidate_exact_weights_recovered": True,
        "candidate_selected": False,
        "reward_contract_approved": False,
        "reward_values_materialized": False,
        "repair_blockers": [
            "No historical reward candidate was selected or promoted; actual R0-R5 ablation evidence is absent.",
            "B1 no-op baseline identity is known, but causal PV8 numeric normalization constants are not locked.",
            "Hard-constraint thresholds/actions remain candidates; bunching and on-time guards are not numerically defined.",
            "on_time_rate and causal intervention burden are NOT_AVAILABLE for current PV8 transitions.",
            "The decision-to-outcome collector and next-state binding contract are defined but not implemented or tested.",
            "Reward-hacking, delayed-harm, terminal-censoring, and shared-credit stress tests have not run.",
        ],
        "exact_approval_required_before_reward_materialization": [
            "Explicitly approve PV8_R2A_R1_BALANCED_DEFAULT_RECOVERY_CANDIDATE and its unchanged ten positive-magnitude weights.",
            "Approve global shared active-slot reward scope and one-global-transition decision-to-outcome horizon.",
            "Generate and freeze causal PV8 B1 normalization constants; preserve B0R/B2 as reporting comparators only.",
            "Approve service/wait/energy/fleet/constraint thresholds and hard-fail versus penalty behavior.",
            "Resolve or remove only through a separately authorized contract the unavailable on-time and intervention inputs; no silent weight change is allowed.",
            "Implement and validate no-future outcome collection, terminal censoring, missing-input hard fail, and causal service alignment before any reward value is produced.",
        ],
        "critical_not_available_fields": inputs["critical_not_available_fields"],
        "open_alignment_repair_count": hacking["open_repair_count"],
        "r2b_or_r3_authorized": False,
        "training_use_authorized": False,
    }


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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2a.jsonl"
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
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2a.json"
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
    writer.json("_PV8_R2A_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": manifest_name,
        "final_manifest_sha256": k5.sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })


def final_report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# PV8-R2A Reward Authority Recovery and Promotion Audit",
        "",
        f"- artifact root: `{summary['artifact_root']}`",
        f"- gate: `{summary['gate']}`",
        f"- decision: `{summary['decision']}`",
        f"- tracked historical reward sources: `{summary['source_count']}`",
        "- prior trainable reward promoted: `false`",
        "- reward contract approved / reward values materialized: `false / false`",
        "",
        "## Recovered Candidate",
        "",
        "The candidate is the unchanged Step 111 / Step 115 `R1_BALANCED_DEFAULT` service-quality-first formula. Positive-magnitude weights are service 3.0, avg-wait 2.0, p95-wait 3.0, on-time 1.0, bunching 1.5, headway-CV 1.0, energy-per-passenger 1.0, fleet 0.5, intervention 0.25, and constraint 10.0. It is recovered for audit only; no historical record selected or promoted it.",
        "",
        "SKIP receives neither a bonus nor a penalty because it is SKIP. The candidate reward depends only on causal service outcomes after the K-safe action vector is committed. Inactive slots receive no learning sample.",
        "",
        "## Normalization and Constraints",
        "",
        "B1 no-op is historically locked only as the normalization reference identity. Its numeric PV8 causal constants are absent; historical defaults 300/600/10 were not adopted. B0R remains a noncausal reporting baseline and B2 remains a rule-based comparator. The 0.95 service floor, 1.10 average-wait cap, 1.15 p95-wait cap, energy/fleet guards, and penalty behavior remain unapproved candidates.",
        "",
        "## Decision-to-Outcome Binding",
        "",
        "The candidate horizon is exactly one global causal transition: observation and K-mask use events at or before decision_ts, the committed action vector starts the open outcome interval, and reward is measured through the next global decision boundary or terminal timestamp. Future outcome may enter reward but never the observation or mask. This collector is defined here but not implemented.",
        "",
        "## Blocking Repair",
        "",
        "Current PV8 inputs cannot provide approved on-time rate or causal intervention burden, B1 numeric constants are missing, hard constraints are unlocked, and no actual R0-R5 ablation or delayed-harm stress test exists. Therefore the candidate is not ready for explicit approval or reward materialization.",
        "",
        "No reward value, episode expansion, policy execution/evaluation, checkpoint reuse, API/DB access, R2B/R3 run, or MAPPO training occurred.",
        "",
    ])


def run_audit(root: Path) -> Path:
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    upstreams = verify_upstreams()
    frozen = verify_frozen_bindings()
    lineage = source_lineage()
    authority = authority_audit(lineage)
    candidate = formula_candidate()
    weights = weight_contract(candidate)
    normalization = normalization_contract()
    constraints = hard_constraint_contract()
    binding = decision_outcome_contract()
    inputs = input_availability()
    hacking = hacking_audit()
    promotion = promotion_decision(authority, inputs, hacking)
    guards = {
        "created_at": iso_kst(),
        "reward_contract_approved": False,
        "reward_values_materialized": False,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "automatic_r2b_execution_authorized": False,
        "automatic_r3_execution_authorized": False,
        "episode_expansion_authorized": False,
    }
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": READINESS,
        "gate_passed": True,
        "final_decision": promotion["final_decision"],
        "failure_reasons": [],
        "readiness_blockers": promotion["repair_blockers"],
    }
    summary = {
        "artifact_root": str(root),
        "gate": gate["gate"],
        "decision": promotion["final_decision"],
        "source_count": lineage["tracked_historical_reward_source_count"],
    }

    writer.json("r2a_reward_source_authority_audit.json", authority)
    writer.json("r2a_historical_reward_lineage.json", lineage)
    writer.json("r2a_pv8_reward_formula_candidate.json", candidate)
    writer.json("r2a_reward_weight_contract.json", weights)
    writer.json("r2a_reward_normalization_contract.json", normalization)
    writer.json("r2a_hard_constraint_contract.json", constraints)
    writer.json("r2a_decision_outcome_binding_contract.json", binding)
    writer.json("r2a_reward_input_availability.json", inputs)
    writer.json("r2a_reward_hacking_alignment_audit.json", hacking)
    writer.json("r2a_reward_promotion_decision.json", promotion)
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
        "frozen_binding_audit": frozen,
        "historical_source_bundle_sha256": canonical_hash(lineage["records"]),
        "new_bis_api_call_count": 0,
        "db_query_count": 0,
        "db_write_count": 0,
        "reward_value_materialization_count": 0,
        "episode_expansion_count": 0,
        "policy_action_execution_count": 0,
        "policy_evaluation_count": 0,
        "checkpoint_reuse_count": 0,
        "r2b_execution_count": 0,
        "r3_execution_count": 0,
        "mappo_training_count": 0,
        "qwen_train": False,
        "qwen_inference": False,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {
        **guards,
        "source_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_decision": promotion["final_decision"],
    })
    writer.text("final_report.md", final_report(summary))
    write_manifest_and_lock(writer, gate)

    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2a.json", "_PV8_R2A_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise R2AError(f"R2A artifact integrity failure: {own}")
    print(f"artifact_root: {root}")
    print(f"gate: {gate['gate']}")
    print(f"decision: {promotion['final_decision']}")
    print(f"tracked_historical_reward_sources: {lineage['tracked_historical_reward_source_count']}")
    print("prior_trainable_reward_promoted: false")
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
