#!/usr/bin/env python3
"""PV8-R2A-R4 explicit reward approval and normalization preflight."""

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
from typing import Any, Dict, Mapping
from zoneinfo import ZoneInfo

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar4_reward_approval_normalization_preflight.py"

K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"
R1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r1_mappo_retraining_preflight_20260808_142604"
R2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2_reward_episode_audit_20260808_143743"
R2AR2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar2_causal_reward_infrastructure_20260808_160345"
R2AR3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar3_bounded_reward_horizon_ablation_20260809_005642"

UPSTREAMS = {
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
    "PV8-K9": (K9_ROOT, "artifact_manifest_srp2_bis_pv8_k9.json", "_PV8_K9_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED"),
    "PV8-R2A-R2": (R2AR2_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar2.json", "_PV8_R2AR2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR2_CAUSAL_REWARD_INFRASTRUCTURE_COMPLETE"),
    "PV8-R2A-R3": (R2AR3_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar3.json", "_PV8_R2AR3_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR3_BOUNDED_REWARD_HORIZON_ABLATION_COMPLETE"),
}

SUPPORTING_EVIDENCE = {
    "PV8-R1": (R1_ROOT, "artifact_manifest_srp2_bis_pv8_r1.json", "_PV8_R1_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R1_MAPPO_RETRAINING_PREFLIGHT_COMPLETE"),
    "PV8-R2": (R2_ROOT, "artifact_manifest_srp2_bis_pv8_r2.json", "_PV8_R2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2_REWARD_AND_EPISODE_DATA_AUDIT_COMPLETE"),
}

RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"
REWARD_VERSION = "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
APPROVAL_TOKEN = "APPROVE_F_PV8_SERVICE_GATED_CENTERED_CORE_V1_H4_SERVICE_GATING_EXCLUSIONS"
ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar4_reward_approval_normalization_preflight"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR4_REWARD_APPROVAL_AND_NORMALIZATION_PREFLIGHT_COMPLETE"
DECISION = "PV8_REWARD_APPROVED_B1_EXPANSION_REQUIRED"
READINESS = "SRP2_BIS_PV8_R2AR4_REWARD_APPROVED_B1_TRAINING_REFERENCE_EXPANSION_REQUIRED"

EXCLUSIONS = [
    "on_time",
    "energy_per_passenger",
    "bunching",
    "headway_cv",
    "fleet_reduction",
    "generic_constraint_penalty",
]

PAYLOADS = [
    "r2ar4_explicit_reward_approval_record.json",
    "r2ar4_frozen_reward_contract.json",
    "r2ar4_reward_contract_hash.json",
    "r2ar4_b1_training_reference_protocol.json",
    "r2ar4_b1_reference_coverage.json",
    "r2ar4_normalization_preflight.json",
    "r2ar4_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R2AR4Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(k5.json_clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_artifact_set(definitions: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in definitions.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed != expected_gate or not k5.manifest_ok(checks):
            raise R2AR4Error(f"{label} integrity failure: gate={observed}, checks={checks}")
        out[label] = {
            "artifact_root": str(root),
            "gate": observed,
            "manifest_integrity": checks,
        }
    return out


def verify_upstream_contract() -> Dict[str, Any]:
    upstream = verify_artifact_set(UPSTREAMS)
    supporting = verify_artifact_set(SUPPORTING_EVIDENCE)
    selection = k5.read_json(R2AR3_ROOT / "r2ar3_candidate_selection_decision.json")
    horizon = k5.read_json(R2AR3_ROOT / "r2ar3_horizon_contract.json")
    bounded = k5.read_json(R2AR2_ROOT / "r2ar2_normalization_candidate.json")
    k8_binding = k5.read_json(K8_ROOT / "k8_snapshot_version_binding.json")
    k9_binding = k5.read_json(K9_ROOT / "k9_version_hash_binding_audit.json")
    checks = {
        "selected_candidate_matches": selection.get("selected_candidate_id") == REWARD_VERSION,
        "selected_horizon_matches": selection.get("selected_horizon_candidate") == "H4" and selection.get("selected_horizon_value") == 4,
        "selected_weights_match": selection.get("selected_candidate_weights") == {"avg_wait": 2.0, "intervention": 0.25, "p95_wait": 3.0, "service": 3.0},
        "bounded_constants_not_approved": bounded.get("approved") is False,
        "bounded_scope_is_limited": "not representative training normalization evidence" in str(bounded.get("scope", "")),
        "k8_rulebook_hash_matches": k8_binding.get("static_rulebook_sha256") == RULEBOOK_SHA256,
        "k8_occurrence_hash_matches": k8_binding.get("occurrence_master_sha256") == OCCURRENCE_SHA256,
        "k9_rulebook_hash_matches": k9_binding.get("rulebook_sha256") == RULEBOOK_SHA256,
        "k9_occurrence_hash_matches": k9_binding.get("occurrence_master_sha256") == OCCURRENCE_SHA256,
        "k9_binding_failure_count_zero": k9_binding.get("binding_failure_count") == 0,
    }
    checks["failure_count"] = sum(not value for value in checks.values())
    if checks["failure_count"]:
        raise R2AR4Error(f"frozen reward or K-safety binding mismatch: {checks}")
    return {"authoritative_upstreams": upstream, "supporting_coverage_evidence": supporting, "contract_checks": checks}


def approval_record(token: str) -> Dict[str, Any]:
    if token != APPROVAL_TOKEN:
        raise R2AR4Error("STOP_PENDING_EXPLICIT_PV8_REWARD_CONTRACT_APPROVAL")
    approved_scope = {
        "reward_version": REWARD_VERSION,
        "formula": "3.0*service + 2.0*service_gated_centered_avg_wait + 3.0*service_gated_centered_p95_wait - 0.25*explicit_forced_external_intervention",
        "horizon": "H4",
        "horizon_interval": "(decision_ts, decision_ts + 240 seconds] capped at terminal/revisit boundary",
        "service_gating": "positive wait improvement is rewarded only when B1 service is preserved; wait deterioration always remains negative",
        "excluded_components": EXCLUSIONS,
        "k_safety_layer": "PRE_ACTION_HARD_CONSTRAINT",
        "bounded_b1_constants_scope": "ABLATION_REFERENCE_ONLY_NOT_TRAINING_NORMALIZATION",
    }
    return {
        "created_at": iso_kst(),
        "approval_status": "EXPLICITLY_APPROVED_BY_USER",
        "approval_source": "current Codex task user message dated 2026-08-09 Asia/Seoul",
        "approval_token": token,
        "approval_token_sha256": hashlib.sha256(token.encode("ascii")).hexdigest(),
        "approved_scope": approved_scope,
        "approval_scope_sha256": canonical_hash(approved_scope),
        "approval_inferred_from_upstream_pass": False,
        "reward_contract_approved": True,
        "training_normalization_approved": False,
    }


def frozen_reward_contract(approval: Mapping[str, Any]) -> Dict[str, Any]:
    contract_body = {
        "reward_version": REWARD_VERSION,
        "status": "APPROVED_REWARD_SEMANTICS_NORMALIZATION_VALUES_PENDING",
        "formula": {
            "expression": "3.0*service + 2.0*service_gated_centered_avg_wait + 3.0*service_gated_centered_p95_wait - 0.25*explicit_forced_external_intervention",
            "positive_magnitude_weights": {"service": 3.0, "avg_wait": 2.0, "p95_wait": 3.0, "intervention": 0.25},
            "service": "clip(service_rate, 0.0, 1.0)",
            "centered_wait_delta": "1.0 - observed_wait / approved_training_B1_wait",
            "service_gated_centered_wait": "retain non-positive deterioration; retain positive improvement only when observed service_rate >= approved_training_B1_service_rate",
            "explicit_forced_external_intervention": "count/rate of typed forced safety override or external policy intervention events only",
            "ordinary_k_mask_restriction_is_intervention": False,
            "action_name_bonus_or_penalty": False,
        },
        "horizon": {
            "id": "H4",
            "seconds": 240,
            "interval": "(decision_ts, decision_ts + 240 seconds]",
            "left_boundary_inclusive": False,
            "right_boundary_inclusive": True,
            "cap": "terminal_or_revisit_boundary",
            "source": "R2A-R3 mechanical H_cover=max positive first-harm reveal transition",
        },
        "excluded_components": EXCLUSIONS,
        "k_safety": {
            "role": "PRE_ACTION_HARD_CONSTRAINT_NOT_REWARD_COMPONENT",
            "static_rulebook_sha256": RULEBOOK_SHA256,
            "occurrence_master_sha256": OCCURRENCE_SHA256,
            "unknown_or_incomplete_skip_legality": "CONDITIONAL_SKIP_FALSE",
        },
        "normalization_dependency": {
            "required": True,
            "training_reference_protocol": "PV8_B1_TRAINING_REFERENCE_PROTOCOL_V1",
            "approved_training_constants": None,
            "bounded_r2ar2_constants_scope": "ABLATION_REFERENCE_ONLY",
            "reward_values_may_be_materialized_before_normalization_approval": False,
        },
    }
    return {
        "created_at": iso_kst(),
        "contract_body": contract_body,
        "reward_contract_sha256": canonical_hash(contract_body),
        "approval_scope_sha256": approval["approval_scope_sha256"],
        "reward_contract_approved": True,
        "training_normalization_approved": False,
        "reward_values_materialized": False,
    }


def reward_hash_record(contract: Mapping[str, Any]) -> Dict[str, Any]:
    body = contract["contract_body"]
    observed = canonical_hash(body)
    if observed != contract["reward_contract_sha256"]:
        raise R2AR4Error("reward contract canonical hash mismatch")
    return {
        "created_at": iso_kst(),
        "reward_version": REWARD_VERSION,
        "hash_algorithm": "SHA-256",
        "canonicalization": "UTF-8 JSON, sorted keys, compact separators, finite JSON values",
        "hashed_payload": "r2ar4_frozen_reward_contract.json#/contract_body",
        "reward_contract_sha256": observed,
        "hash_verified": True,
    }


def b1_training_reference_protocol(contract_hash: str) -> Dict[str, Any]:
    action_policy = {
        "learned_policy_used": False,
        "conditional_skip_selected": False,
        "active_slot_primary": "SERVE where valid",
        "active_slot_fallback": "HOLD only when SERVE is unavailable",
        "inactive_slot": "NO_ACTION_NO_REWARD_SAMPLE",
        "deterministic_tie_break": "fixed physical agent_id ascending",
    }
    protocol_body = {
        "protocol_version": "PV8_B1_TRAINING_REFERENCE_PROTOCOL_V1",
        "classification": "PROSPECTIVE_DETERMINISTIC_CONTROL_REFERENCE_NOT_POLICY_EVALUATION",
        "reward_contract_sha256": contract_hash,
        "num_agents": 8,
        "action_dim": 3,
        "rulebook_sha256": RULEBOOK_SHA256,
        "occurrence_master_sha256": OCCURRENCE_SHA256,
        "required_equal_contracts": [
            "physical 8-agent slots",
            "K8/K9 safety and snapshot lifecycle",
            "passenger/request dynamics",
            "causal decision-to-outcome collector",
            "exogenous schedule contract",
            "episode boundaries",
        ],
        "action_policy": action_policy,
        "reference_fit_partition": "chronological training partition only",
        "validation_and_test_use": "evaluation only; never fit normalization constants",
        "required_coverage": [
            "multiple prospective episodes and windows beyond the bounded R2A-R2 fixture",
            "all 8 fixed physical agents",
            "active and inactive lifecycle states",
            "multiple dates/time bands and exogenous schedules",
            "reward-valid causal H4 outcomes",
        ],
        "required_metrics": ["service_rate", "avg_wait_seconds", "p95_wait_seconds"],
        "required_audits": [
            "per-window finite and strictly positive wait denominators",
            "service-rate support and service-preserving control compliance",
            "window/date/agent coverage",
            "distribution stability across reference windows",
            "leave-one-window-out normalization sensitivity",
            "identity/hash/version drift and future-leakage checks",
        ],
        "promotion_rule": "constants remain unapproved unless broader prospective coverage is complete and stability/sensitivity audits are reviewed explicitly",
        "api_or_db_collection_authorized": False,
        "policy_evaluation": False,
        "mappo_training": False,
    }
    return {
        "created_at": iso_kst(),
        "protocol_body": protocol_body,
        "protocol_sha256": canonical_hash(protocol_body),
        "training_normalization_approved": False,
    }


def b1_reference_coverage() -> Dict[str, Any]:
    bounded = k5.read_json(R2AR2_ROOT / "r2ar2_pv8_b1_reference_audit.json")
    normalization = k5.read_json(R2AR2_ROOT / "r2ar2_normalization_candidate.json")
    r1 = k5.read_json(R1_ROOT / "r1_episode_coverage.json")
    r2 = k5.read_json(R2_ROOT / "r2_dataset_coverage.json")
    k9 = k5.read_json(K9_ROOT / "k9_action_availability_summary.json")
    constants = normalization["constants"]
    expected = {"service_rate": 1.0, "avg_wait_seconds": 50.5, "p95_wait_seconds": 59.95}
    bounded_matches = all(constants.get(key) == value for key, value in expected.items())
    coverage = {
        "created_at": iso_kst(),
        "bounded_r2ar2_reference": {
            "source_class": bounded["source_class"],
            "episode_or_fixture_count": 1,
            "window_count": 1,
            "active_reference_sample_count": bounded["active_reference_sample_count"],
            "agent_count": bounded["num_fixed_slots"],
            "constants": expected,
            "constants_match_prompt": bounded_matches,
            "scope": "ABLATION_REFERENCE_ONLY",
            "training_normalization_eligible": False,
        },
        "available_prospective_snapshot_evidence": {
            "episode_count": r1["episode_count"],
            "cycle_count": r1["cycle_count"],
            "agent_snapshot_count": r1["agent_snapshot_count"],
            "active_snapshot_count": r1["active_agent_snapshot_count"],
            "inactive_snapshot_count": r1["inactive_agent_snapshot_count"],
            "all_agents_represented": len(r1["per_agent"]) == 8,
            "k9_active_count": k9["active_count"],
            "k9_inactive_count": k9["inactive_count"],
            "reward_valid_transition_count": r2["reward_valid_transition_count"],
            "date_coverage_utc": r2["date_coverage_utc"],
            "time_band_utc": r2["time_band_utc"],
            "causal_reward_outcomes_bound": False,
            "b1_control_executed": False,
            "training_normalization_eligible": False,
            "ineligibility_reason": "snapshot lifecycle coverage has no reward-valid decision-to-H4 outcomes and is not a B1 control rollout",
        },
        "broader_training_reference": {
            "episode_count": 0,
            "window_count": 0,
            "reward_valid_transition_count": 0,
            "time_coverage": [],
            "agent_coverage": [],
            "service_rate_distribution": None,
            "avg_wait_distribution": None,
            "p95_wait_distribution": None,
            "distribution_stability": "NOT_EVALUABLE_NO_BROADER_REFERENCE",
            "window_sensitivity": "NOT_EVALUABLE_NO_BROADER_REFERENCE",
            "finite_nonzero_denominators_verified": False,
            "generated_in_r2ar4": False,
        },
        "bounded_reference_reused_as_training_normalization": False,
        "synthetic_or_missing_evidence_fabricated": False,
        "coverage_sufficient_for_training_normalization": False,
    }
    if not bounded_matches or r2["reward_valid_transition_count"] != 0:
        raise R2AR4Error("unexpected B1 or prospective reward coverage state")
    return coverage


def normalization_preflight(coverage: Mapping[str, Any], contract_hash: str) -> Dict[str, Any]:
    checks = {
        "reward_contract_frozen": True,
        "reward_contract_hash_bound": bool(contract_hash),
        "bounded_constants_finite_nonzero": True,
        "bounded_constants_remain_ablation_only": True,
        "broader_than_bounded_reference_available": False,
        "multiple_reference_windows_available": False,
        "multiple_reference_dates_or_time_bands_available": False,
        "all_agent_reward_valid_coverage_available": False,
        "distribution_stability_evaluable": False,
        "window_sensitivity_evaluable": False,
        "training_partition_only_fit_possible_now": False,
        "future_leakage_detected": False,
    }
    return {
        "created_at": iso_kst(),
        "preflight_status": "NOT_READY_B1_EXPANSION_REQUIRED",
        "reward_contract_sha256": contract_hash,
        "checks": checks,
        "bounded_constants": coverage["bounded_r2ar2_reference"]["constants"],
        "bounded_constants_status": "ABLATION_REFERENCE_ONLY_NOT_PROMOTED",
        "approved_training_normalization_constants": None,
        "normalization_semantics_changed": False,
        "training_normalization_approved": False,
        "reward_values_materialized": False,
        "blockers": [
            "No broader prospective PV8-B1 control reference has been executed.",
            "The only prospective lifecycle episode has zero reward-valid causal transitions.",
            "Cross-window distribution stability and normalization sensitivity cannot be evaluated.",
            "A chronological training partition does not yet exist for leakage-free normalization fitting.",
        ],
        "minimum_next_scope": [
            "Execute the frozen PV8_B1_TRAINING_REFERENCE_PROTOCOL_V1 over multiple prospective episodes/windows with the R2A-R2 causal collector and H4 outcomes.",
            "Fit candidate service/avg-wait/p95-wait references on the chronological training partition only.",
            "Audit finite positive denominators, all-agent/date/time coverage, distribution stability, and leave-one-window-out sensitivity.",
            "Submit generated constants for a separate explicit training-normalization approval before reward materialization.",
        ],
    }


def readiness_decision(preflight: Mapping[str, Any], contract_hash: str) -> Dict[str, Any]:
    if preflight["preflight_status"] != "NOT_READY_B1_EXPANSION_REQUIRED":
        raise R2AR4Error("unexpected normalization preflight status")
    return {
        "created_at": iso_kst(),
        "final_decision": DECISION,
        "audit_complete": True,
        "explicit_reward_approval_valid": True,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": contract_hash,
        "reward_contract_approved": True,
        "training_normalization_ready": False,
        "training_normalization_approved": False,
        "reward_values_materialized": False,
        "remaining_blocker": "broader prospective PV8-B1 reference expansion and separate normalization approval",
        "minimum_next_stage": "PV8-R2B B1 reference expansion/materialization preflight only; no MAPPO training",
        "r2b_or_episode_expansion_authorized": False,
        "training_use_authorized": False,
    }


def claim_guards() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_contract_approved": True,
        "training_normalization_approved": False,
        "reward_values_materialized": False,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "conditional_skip_policy_execution_authorized": False,
        "episode_expansion_authorized": False,
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
    env["PYTHONPYCACHEPREFIX"] = "/private/tmp/pv8_r2ar4_pycache"
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
    return "\n".join([
        "# PV8-R2A-R4 Reward Approval and Normalization Preflight",
        "",
        f"- artifact root: `{summary['artifact_root']}`",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{DECISION}`",
        f"- reward version: `{REWARD_VERSION}`",
        f"- reward contract SHA-256: `{summary['contract_hash']}`",
        "",
        "## Approval",
        "",
        "The user explicitly approved the F candidate, H4 semantics, service gating, and every listed exclusion. Reward-contract approval is therefore true. Training-normalization approval remains false and was not inferred from reward approval.",
        "",
        "## Frozen Contract",
        "",
        "The formula is `3.0*service + 2.0*service_gated_centered_avg_wait + 3.0*service_gated_centered_p95_wait - 0.25*explicit_forced_external_intervention`. H4 includes causal events in `(decision_ts, decision_ts + 240 seconds]`, capped at terminal/revisit. Positive wait improvement is gated by preservation of the approved B1 service level; wait deterioration is always negative.",
        "",
        "Excluded components are on-time, energy-per-passenger, bunching, headway-CV, fleet reduction, and the generic constraint penalty. K-safety remains a pre-action hard constraint.",
        "",
        "## B1 Coverage",
        "",
        "The R2A-R2 constants remain bounded ablation references only: service `1.0`, average wait `50.5`, and p95 wait `59.95`. The bounded fixture has one window and eight active samples. It was not promoted.",
        "",
        f"The available prospective lifecycle evidence has `{summary['episode_count']}` episode, `{summary['cycle_count']}` cycles, `{summary['active_count']}` active snapshots, and `{summary['inactive_count']}` inactive snapshots, but reward-valid causal transitions are `{summary['reward_valid_count']}`. It is not a broader B1 control rollout and cannot support training normalization.",
        "",
        "## Normalization Readiness",
        "",
        "No broader B1 distribution exists, so cross-window stability, leave-one-window-out sensitivity, and training-partition-only fitting cannot yet be evaluated. The decision is `PV8_REWARD_APPROVED_B1_EXPANSION_REQUIRED`. The approved reward semantics and hash are frozen, while training constants remain null.",
        "",
        "The minimum next repair is a bounded prospective B1 reference expansion using the frozen eight-agent K8/K9 contract, deterministic SERVE/HOLD control, R2A-R2 causal collector, and H4 outcomes. Generated constants require a separate explicit normalization approval.",
        "",
        "No API/DB access, reward value materialization, episode expansion, policy evaluation, checkpoint reuse, downstream run, or MAPPO training occurred.",
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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar4.jsonl"
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
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar4.json"
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
    writer.json("_PV8_R2AR4_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": manifest_name,
        "final_manifest_sha256": k5.sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })


def run(root: Path, approval_token: str) -> Path:
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    upstream = verify_upstream_contract()
    approval = approval_record(approval_token)
    regression = run_pytest_regression()
    if not regression["passed"]:
        raise R2AR4Error(f"reward/K-safety regression failed: {regression}")
    contract = frozen_reward_contract(approval)
    hash_record = reward_hash_record(contract)
    protocol = b1_training_reference_protocol(hash_record["reward_contract_sha256"])
    coverage = b1_reference_coverage()
    preflight = normalization_preflight(coverage, hash_record["reward_contract_sha256"])
    decision = readiness_decision(preflight, hash_record["reward_contract_sha256"])
    guards = claim_guards()
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": READINESS,
        "gate_passed": True,
        "final_decision": decision["final_decision"],
        "reward_contract_approved": True,
        "training_normalization_approved": False,
        "failure_reasons": [],
        "readiness_blockers": preflight["blockers"],
    }
    prospective = coverage["available_prospective_snapshot_evidence"]
    summary = {
        "artifact_root": str(root),
        "contract_hash": hash_record["reward_contract_sha256"],
        "episode_count": prospective["episode_count"],
        "cycle_count": prospective["cycle_count"],
        "active_count": prospective["active_snapshot_count"],
        "inactive_count": prospective["inactive_snapshot_count"],
        "reward_valid_count": prospective["reward_valid_transition_count"],
    }

    writer.json("r2ar4_explicit_reward_approval_record.json", approval)
    writer.json("r2ar4_frozen_reward_contract.json", contract)
    writer.json("r2ar4_reward_contract_hash.json", hash_record)
    writer.json("r2ar4_b1_training_reference_protocol.json", protocol)
    writer.json("r2ar4_b1_reference_coverage.json", coverage)
    writer.json("r2ar4_normalization_preflight.json", preflight)
    writer.json("r2ar4_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "explicit-approval-normalization-preflight",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "upstream_integrity": upstream,
        "reward_contract_sha256": hash_record["reward_contract_sha256"],
        "pytest_regression": regression,
        "bounded_b1_reference_sample_count": coverage["bounded_r2ar2_reference"]["active_reference_sample_count"],
        "broader_b1_reference_episode_count": coverage["broader_training_reference"]["episode_count"],
        "broader_b1_reference_window_count": coverage["broader_training_reference"]["window_count"],
        "new_bis_api_call_count": 0,
        "db_query_count": 0,
        "db_write_count": 0,
        "reward_value_materialization_count": 0,
        "episode_expansion_count": 0,
        "policy_evaluation_count": 0,
        "checkpoint_reuse_count": 0,
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

    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar4.json", "_PV8_R2AR4_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise R2AR4Error(f"R2A-R4 artifact integrity failure: {own}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {decision['final_decision']}")
    print(f"reward_contract_sha256: {hash_record['reward_contract_sha256']}")
    print("reward_contract_approved: true")
    print("training_normalization_approved: false")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["approve-and-preflight"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--approval-token", required=True)
    args = parser.parse_args()
    run(args.artifact_root, args.approval_token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
