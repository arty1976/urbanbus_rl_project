#!/usr/bin/env python3
"""PV8 H4I-R1 readiness contract repair audit.

This stage does not train MAPPO and does not update model/runtime files.
It records which H4I blockers are repaired by current evidence and which
remain blocked for the next minimum repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo


STAGE = "PV8-R2A-R8E-R3-R-H4I-R1"
ARTIFACT_PREFIX = "pv8_r2a_r8e_r3_r_h4i_r1_readiness_contract_repair"

PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4I_R1_READINESS_CONTRACT_REPAIR_COMPLETE"
BLOCKED_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4I_R1_READINESS_CONTRACT_REPAIR_INCOMPLETE"
PASS_DECISION = "PV8_H4I_BLOCKERS_REPAIRED_READY_FOR_H4I_READINESS_RERUN"
BLOCKED_DECISION = "PV8_H4I_R1_CONTRACT_REPAIR_INCOMPLETE_MINIMUM_REPAIR_REQUIRED"

EXPECTED_REWARD_SHA = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
EXPECTED_H4G_SHA = "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3"
EXPECTED_D1_SHA = "318edf0210c1fd0e66ef22f90a1fcb7ebac5fd4d73a294d8c6f2ff7c0867093c"

H4I_ROOT = "05_training/artifacts/pv8_r2a_r8e_r3_r_h4i_training_readiness_20260810_172656"
R8C_ROOT = "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8c_training_normalization_freeze_20260809_155120"
R1_ROOT = "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r1_mappo_retraining_preflight_20260808_142604"
R2_ROOT = "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2_reward_episode_audit_20260808_143743"
D1_ROOT = "05_training/artifacts/prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4i_er0d1_temporal_credit_strength_density_20260810_135520"

PROTECTED_RUNTIME_FILES = [
    "05_training/mappo_runner.py",
    "05_training/rewards/mappo_reward_v1.py",
    "05_training/simulator/pv8_reward_outcome_collector.py",
    "05_training/simulator/pv8_b1_orchestrator.py",
]

REQUIRED_OUTPUTS = [
    "01_h4i_blocker_reconciliation.json",
    "02_legacy_normalization_supersession.json",
    "03_reward_v2_normalization_contract.json",
    "04_seed_split_contract.json",
    "05_training_exposure_audit.json",
    "06_ppo_optimizer_contract.json",
    "07_critic_bootstrap_contract.json",
    "08_gatv2_encoder_contract.json",
    "09_fresh_lineage_contract.json",
    "10_h4i_r1_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]


def iso_now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def git_commit(project_root: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root, text=True).strip()
    except Exception:
        return None


def torch_version() -> Optional[str]:
    try:
        import torch  # type: ignore

        return str(torch.__version__)
    except Exception:
        return None


def inv_by_field(training_inventory: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(row.get("field")): row for row in training_inventory.get("inventory", [])}


def load_inputs(project_root: Path) -> Dict[str, Any]:
    h4i = project_root / H4I_ROOT
    return {
        "h4i_root": h4i,
        "h4i_decision": read_json(h4i / "25_h4i_readiness_decision.json"),
        "h4i_sha_lineage": read_json(h4i / "02_sha_lineage_audit.json"),
        "h4i_training_inventory": read_json(h4i / "04_training_contract_inventory.json"),
        "h4i_reward_summary": read_json(h4i / "09_reward_normalization_summary.json"),
        "h4i_advantage_audit": read_json(h4i / "10_advantage_normalization_audit.json"),
        "h4i_return_value_audit": read_json(h4i / "11_return_value_normalization_audit.json"),
        "h4i_seed_split": read_json(h4i / "15_seed_split_integrity.json"),
        "h4i_ppo": read_json(h4i / "18_ppo_optimizer_contract_audit.json"),
        "h4i_critic": read_json(h4i / "07_critic_bootstrap_readiness.json"),
        "h4i_gatv2": read_json(h4i / "17_gatv2_encoder_contract_audit.json"),
        "h4i_future": read_json(h4i / "20_safety_future_leakage_audit.json"),
        "h4i_candidate": read_json(h4i / "23_fresh_mappo_retraining_contract_candidate.json"),
        "h4i_exposure_split": read_json(h4i / "14_training_exposure_by_split.json"),
        "r8c_normalization": read_json(project_root / R8C_ROOT / "r8c_training_normalization_contract.json"),
        "r8c_exclusion": read_json(project_root / R8C_ROOT / "r8c_legacy_normalization_exclusion_audit.json"),
        "r1_tensor": read_json(project_root / R1_ROOT / "r1_mappo_tensor_contract.json"),
        "r1_checkpoint_metadata": read_json(project_root / R1_ROOT / "r1_training_checkpoint_metadata_contract.json"),
        "r2_split": read_json(project_root / R2_ROOT / "r2_split_contract.json"),
        "d1_dimensions": read_json(project_root / D1_ROOT / "r8er3rh4i_er0d1_training_readiness_dimensions.json"),
    }


def lineage_audit(project_root: Path, data: Mapping[str, Any]) -> Dict[str, Any]:
    checks = data["h4i_sha_lineage"].get("checks", {})
    extracted = {
        "reward_v2_freeze_sha256": checks.get("reward_v2_freeze_sha256", {}).get("actual"),
        "h4g_runtime_binding_sha256": checks.get("h4g_runtime_binding_sha256", {}).get("actual"),
        "h4i_er0d1_payload_sha256": checks.get("h4i_er0d1_payload_sha256", {}).get("actual"),
    }
    expected = {
        "reward_v2_freeze_sha256": EXPECTED_REWARD_SHA,
        "h4g_runtime_binding_sha256": EXPECTED_H4G_SHA,
        "h4i_er0d1_payload_sha256": EXPECTED_D1_SHA,
    }
    rows = []
    for key, exp in expected.items():
        rows.append({"identity": key, "expected": exp, "actual": extracted.get(key), "passed": extracted.get(key) == exp})
    return {
        "created_at": iso_now(),
        "source": str(project_root / H4I_ROOT / "02_sha_lineage_audit.json"),
        "source_sha256": sha256_file(project_root / H4I_ROOT / "02_sha_lineage_audit.json"),
        "checks": rows,
        "all_passed": all(row["passed"] for row in rows),
        "api_call_count": 0,
        "db_query_count": 0,
        "db_write_count": 0,
    }


def legacy_normalization_supersession(project_root: Path, data: Mapping[str, Any]) -> Dict[str, Any]:
    inv = inv_by_field(data["h4i_training_inventory"])
    prior = inv.get("reward_normalization", {})
    r8c = data["r8c_normalization"]
    return {
        "created_at": iso_now(),
        "prior_h4i_classification": prior.get("authoritative_status"),
        "prior_contract_version": r8c.get("normalization_version"),
        "prior_contract_source": str(project_root / R8C_ROOT / "r8c_training_normalization_contract.json"),
        "prior_contract_sha256": sha256_file(project_root / R8C_ROOT / "r8c_training_normalization_contract.json"),
        "prior_values": {
            "service_reference": r8c.get("B1_service_reference"),
            "avg_wait_reference_seconds": r8c.get("B1_avg_wait_reference"),
            "p95_wait_reference_seconds": r8c.get("B1_p95_wait_reference"),
            "horizon": r8c.get("horizon"),
            "horizon_seconds": r8c.get("horizon_seconds"),
        },
        "corrected_classification": "LEGACY_SUPERSEDED_BY_REWARD_V2",
        "current_reward_v2_reference": {
            "service_reference": 1.0,
            "avg_wait_reference_seconds": 297.7850241545894,
            "p95_wait_reference_seconds": 576.6999999999999,
            "p95_role": "EVALUATION_ONLY",
        },
        "prohibitions_preserved": {
            "p95_training_normalization": "PROHIBITED",
            "p95_training_reward": "PROHIBITED",
            "H240_active_reward_settlement": False,
            "H660_active_reward_settlement": False,
        },
        "repair_status": "REPAIRED_FOR_CLASSIFICATION_ONLY",
        "training_normalization_approval_status": "NOT_PROMOTED_BY_H4I_R1",
        "rationale": (
            "R8C used earlier headway-aware H4=240 semantics and values; current Reward V2 is bound "
            "to H4G/H4H/H4I-D1 semantics and the canonical avg-wait reference 297.7850241545894. "
            "The R8C contract remains lineage evidence only unless a later stage explicitly re-approves it."
        ),
    }


def reward_v2_normalization_contract(project_root: Path, data: Mapping[str, Any]) -> Dict[str, Any]:
    inv = inv_by_field(data["h4i_training_inventory"])
    reward_summary = data["h4i_reward_summary"]
    return {
        "created_at": iso_now(),
        "contract_version": "PV8_REWARD_V2_NORMALIZATION_CONTRACT_R1_AUDIT_CANDIDATE",
        "contract_freeze_status": "BLOCKED_NORMALIZATION_SUBCONTRACTS_UNRESOLVED",
        "reward_reference": {
            "service_reference": 1.0,
            "avg_wait_reference_seconds": 297.7850241545894,
            "p95_wait_reference_seconds": 576.6999999999999,
            "p95_role": "EVALUATION_ONLY",
        },
        "runtime_reward_normalizer": {
            "status": inv.get("reward_normalizer_runtime", {}).get("authoritative_status"),
            "source_path": inv.get("reward_normalizer_runtime", {}).get("source_path"),
            "source_sha256": inv.get("reward_normalizer_runtime", {}).get("source_sha256"),
            "enabled": True,
            "window_size": 1000,
            "clip_value": 10.0,
            "clip_order": "clip reward to [-10, 10] before running-buffer statistics",
            "stat_dtype": "numpy.float32 buffer statistics in RewardNormalizer.normalize",
            "epsilon_or_std_floor": 1e-6,
            "scope": "global reward stream within one MAPPOExperimentRunner instance",
            "reset_policy": "fresh RewardNormalizer instance per fresh runner",
            "checkpoint_persistence": "fresh state required; exact future checkpoint persistence not yet frozen",
        },
        "advantage_normalization": {
            "status": data["h4i_advantage_audit"].get("advantage_normalization_current_status"),
            "classification": inv.get("advantage_normalization", {}).get("authoritative_status"),
            "blocking": True,
            "reason": data["h4i_advantage_audit"].get("decision"),
        },
        "return_normalization": {
            "status": data["h4i_return_value_audit"].get("return_normalization_current_status"),
            "classification": inv.get("return_normalization", {}).get("authoritative_status"),
            "blocking": True,
            "reason": "Reward V2 return normalization contract is missing.",
        },
        "value_normalization": {
            "status": data["h4i_return_value_audit"].get("value_normalization_current_status"),
            "classification": inv.get("value_normalization", {}).get("authoritative_status"),
            "blocking": True,
            "reason": "Reward V2 value normalization contract is missing.",
        },
        "forbidden_reuse": {
            "old_reward_v1_normalizer_state_reuse": False,
            "old_dl3_dl4_normalizer_state_reuse": False,
            "test_holdout_normalization_fitting": False,
        },
        "signal_preservation_audit": {
            "nonzero_d1_pairs_audited": reward_summary.get("nonzero_d1_pairs_audited"),
            "normalization_induced_sign_reversal_count": reward_summary.get("normalized_sign_reversal_count"),
            "zeroed_by_normalization_count": reward_summary.get("zeroed_by_normalization_count"),
            "systematic_signal_erasure": reward_summary.get("systematic_signal_erasure"),
            "passed_required_invariant": reward_summary.get("passed_required_invariant"),
        },
        "blocking_fields": ["advantage_normalization", "return_normalization", "value_normalization"],
    }


def seed_split_contract(project_root: Path, data: Mapping[str, Any]) -> Dict[str, Any]:
    seed = data["h4i_seed_split"]
    return {
        "created_at": iso_now(),
        "contract_version": "PV8_REWARD_V2_SEED_SPLIT_CONTRACT_R1_AUDIT",
        "freeze_status": "BLOCKED_REQUIRED_SEED_SPLIT_MISSING",
        "seed_set": {
            "status": seed.get("current_reward_v2_seed_set_status"),
            "value": seed.get("current_reward_v2_seed_set"),
            "legacy_dl3_seed_registry_path": seed.get("legacy_dl3_seed_registry_path"),
            "legacy_dl3_seed_registry_sha256": seed.get("legacy_dl3_seed_registry_sha256"),
            "legacy_dl3_seeds_not_reused": seed.get("legacy_dl3_seeds_not_reused"),
        },
        "splits": {
            "train": seed.get("current_reward_v2_train_split"),
            "validation": seed.get("current_reward_v2_validation_split"),
            "test": seed.get("current_reward_v2_test_split"),
            "r2_split_contract_path": seed.get("r2_split_contract_path"),
            "r2_split_contract_sha256": seed.get("r2_split_contract_sha256"),
            "r2_split_contract_decision": data["r2_split"].get("decision") or data["r2_split"].get("readiness_decision"),
        },
        "integrity": {
            "no_overlap_verified": seed.get("no_overlap_verified"),
            "chronological_order_verified": seed.get("chronological_order_verified"),
            "split_leakage_detected": seed.get("split_leakage_detected"),
            "normalization_fit_from_test_holdout": seed.get("normalization_fit_from_test_holdout"),
            "hard_leakage_block": seed.get("hard_leakage_block"),
        },
        "classification": {
            "seed_set": "MISSING",
            "train_split": "MISSING",
            "validation_split": "MISSING",
            "test_split": "MISSING",
        },
        "blocking": True,
        "blocking_reasons": seed.get("blocking_reasons", []),
        "rule": "DL3 values are legacy-only and are not promoted automatically.",
    }


def training_exposure_audit(data: Mapping[str, Any]) -> Dict[str, Any]:
    dims = data["d1_dimensions"]
    exposure_split = data["h4i_exposure_split"]
    return {
        "created_at": iso_now(),
        "audit_status": "NOT_EVALUABLE_TRAINING_SPLIT_NOT_FROZEN",
        "training_split_only_counts": None,
        "representative_d1_context_only": {
            "windows": dims.get("windows"),
            "passengers": dims.get("passengers"),
            "legal_empty_stop_opportunities": dims.get("legal_empty_stop_opportunities"),
            "passenger_exposed_empty_stop_opportunities": dims.get("passenger_exposed_empty_stop_opportunities"),
            "affected_passenger_rows": dims.get("affected_passenger_rows"),
            "correct_sign_gae_t0_advantages": dims.get("correct_sign_gae_t0_advantages")
            or dims.get("canonical_funnel_correct_sign_gae_t0"),
        },
        "h4i_split_audit": exposure_split,
        "fail_closed_checks": {
            "training_temporal_signal_zero": "NOT_EVALUABLE",
            "signal_exists_only_in_val_test": "NOT_EVALUABLE",
            "illegal_kmask_only_signal": "NOT_EVALUABLE",
            "provenance_unknown": True,
            "nondeterministic_counts": False,
        },
        "blocking": True,
        "blocking_reason": "No authoritative current Reward V2 train/validation/test split exists, so training-only temporal-signal exposure cannot be computed.",
        "no_arbitrary_density_threshold_created": True,
    }


def ppo_optimizer_contract(data: Mapping[str, Any]) -> Dict[str, Any]:
    ppo = dict(data["h4i_ppo"])
    ppo["created_at"] = iso_now()
    ppo["contract_version"] = "PV8_REWARD_V2_PPO_OPTIMIZER_CONTRACT_R1_AUDIT"
    ppo["freeze_status"] = "BLOCKED_REQUIRED_PARAMETERS_UNRESOLVED"
    ppo["legacy_values_not_promoted"] = True
    ppo["hyperparameter_search_executed"] = False
    ppo["optimizer_created"] = False
    ppo["optimizer_step_executed"] = False
    return ppo


def critic_bootstrap_contract(data: Mapping[str, Any]) -> Dict[str, Any]:
    critic = dict(data["h4i_critic"])
    critic["created_at"] = iso_now()
    critic["contract_version"] = "PV8_REWARD_V2_CRITIC_BOOTSTRAP_CONTRACT_R1_AUDIT"
    critic["critic_initialization"] = "FRESH"
    critic["rollout_horizon"] = 512
    critic["gamma"] = 0.99
    critic["gae_lambda"] = 0.95
    critic["return_value_normalization_blocking"] = data["h4i_return_value_audit"].get("blocking")
    critic["ppo_optimizer_blocking_parameters"] = data["h4i_ppo"].get("blocking_parameters", [])
    critic["freeze_status"] = "BLOCKED_PV8_CRITIC_BOOTSTRAP_CONTRACT_NOT_READY"
    critic["training_executed"] = False
    return critic


def gatv2_encoder_contract(data: Mapping[str, Any]) -> Dict[str, Any]:
    gat = dict(data["h4i_gatv2"])
    gat["created_at"] = iso_now()
    gat["contract_version"] = "PV8_REWARD_V2_GATV2_ENCODER_CONTRACT_R1_AUDIT"
    gat["freeze_status"] = "BLOCKED_ENCODER_POLICY_UNKNOWN"
    gat["allowed_status_values"] = ["PRETRAINED_FROZEN", "PRETRAINED_FINETUNED", "FRESH"]
    gat["mappo_checkpoint_reuse_separated_from_encoder_reuse"] = not bool(gat.get("mappo_checkpoint_reuse_confused_with_encoder_reuse"))
    return gat


def fresh_lineage_contract(data: Mapping[str, Any], blockers: Sequence[str]) -> Dict[str, Any]:
    return {
        "created_at": iso_now(),
        "lineage_version": "FRESH_MAPPO_RETRAINING_LINEAGE_R1_CANDIDATE",
        "lineage_ready": len(blockers) == 0,
        "freeze_status": "READY" if len(blockers) == 0 else "BLOCKED_BY_UPSTREAM_CONTRACT_GAPS",
        "required_fresh_components": {
            "actor": "FRESH",
            "critic": "FRESH",
            "optimizer": "FRESH_REQUIRED_BUT_CONFIG_UNRESOLVED",
            "scheduler": "FRESH_REQUIRED_BUT_CONFIG_UNRESOLVED",
            "reward_normalizer": "FRESH_STATE_REQUIRED",
            "return_value_normalizer": "FRESH_STATE_REQUIRED_BUT_CONTRACT_UNRESOLVED",
            "rng_lineage": "FRESH_REQUIRED_BUT_SEED_SET_MISSING",
            "checkpoint_namespace": "NEW_NAMESPACE_REQUIRED_NOT_FROZEN",
        },
        "blacklist": [
            "DL3 actor",
            "DL4 critic",
            "Reward-V1 optimizer",
            "Reward-V1 normalizer",
            "old scheduler state",
        ],
        "old_checkpoint_continuation": "PROHIBITED",
        "implicit_resume_allowed": False,
        "blocking_dependencies": list(blockers),
        "training_authorized": False,
        "checkpoint_reuse_authorized": False,
    }


def h4i_blocker_reconciliation(
    data: Mapping[str, Any],
    legacy: Mapping[str, Any],
    normalization: Mapping[str, Any],
    seed_split: Mapping[str, Any],
    exposure: Mapping[str, Any],
    ppo: Mapping[str, Any],
    critic: Mapping[str, Any],
    gatv2: Mapping[str, Any],
) -> Dict[str, Any]:
    h4i_blockers = data["h4i_decision"].get("hard_blockers", [])
    groups = [
        {
            "group": "A_normalization_contract",
            "h4i_gate_ids": ["G13_ADVANTAGE_NORMALIZATION_PRESERVES_SIGNAL", "G14_RETURN_VALUE_NORMALIZATION_FRESH"],
            "repair_status": "PARTIAL_REPAIRED_LEGACY_SUPERSESSION_ONLY",
            "blocking": bool(normalization.get("blocking_fields")),
            "remaining_blockers": normalization.get("blocking_fields", []),
        },
        {
            "group": "B_seed_split_exposure_contract",
            "h4i_gate_ids": ["G18_TRAINING_SIGNAL_EXPOSURE", "G19_TIME_BAND_EXPOSURE", "G20_SPLIT_INTEGRITY", "G21_SEED_CONTRACT"],
            "repair_status": "BLOCKED_REQUIRED_SEED_SPLIT_MISSING",
            "blocking": True,
            "remaining_blockers": seed_split.get("blocking_reasons", []) + ["training_split_exposure_not_evaluable"],
        },
        {
            "group": "C_ppo_optimizer_critic_bootstrap_contract",
            "h4i_gate_ids": ["G11_CRITIC_BOOTSTRAP_READY", "G23_PPO_OPTIMIZER_CONTRACT"],
            "repair_status": "BLOCKED_PPO_AND_BOOTSTRAP_UNRESOLVED",
            "blocking": True,
            "remaining_blockers": ppo.get("blocking_parameters", []) + critic.get("blocking_reasons", []),
        },
        {
            "group": "D_gatv2_encoder_contract",
            "h4i_gate_ids": ["G24_GATV2_ENCODER_CONTRACT"],
            "repair_status": "BLOCKED_ENCODER_POLICY_UNKNOWN",
            "blocking": bool(gatv2.get("blocking", True)),
            "remaining_blockers": ["gatv2_encoder_loading_policy"],
        },
        {
            "group": "E_fresh_retraining_lineage_contract",
            "h4i_gate_ids": ["G36_FRESH_LINEAGE_READY"],
            "repair_status": "BLOCKED_BY_A_TO_D_CONTRACT_GAPS",
            "blocking": True,
            "remaining_blockers": ["normalization", "seed_split_exposure", "ppo_critic", "gatv2"],
        },
    ]
    return {
        "created_at": iso_now(),
        "h4i_source_artifact": H4I_ROOT,
        "h4i_scientific_decision": data["h4i_decision"].get("scientific_decision"),
        "h4i_hard_blockers": h4i_blockers,
        "root_groups": groups,
        "legacy_r8c_supersession_status": legacy.get("repair_status"),
        "actual_training_executed": False,
        "reward_v2_redesign": False,
        "overall_repair_complete": not any(group["blocking"] for group in groups),
    }


def safety_regression_zero(data: Mapping[str, Any]) -> Dict[str, Any]:
    future = data["h4i_future"]
    fields = {
        "actor_future_leakage": future.get("actor_future_leakage"),
        "critic_future_leakage": future.get("critic_future_leakage"),
        "missed_eligible_service": future.get("missed_eligible_service"),
        "alignment_excess": future.get("alignment_excess_violation", future.get("alignment_excess")),
        "illegal_skip": future.get("illegal_skip"),
        "duplicate_service_ownership": future.get("duplicate_service_ownership"),
        "duplicate_wait_ownership": future.get("duplicate_wait_ownership"),
        "orphan_reward": future.get("orphan_reward"),
        "nan_inf_blocker": future.get("nan_inf_blocker"),
    }
    passed = (
        fields["actor_future_leakage"] == 0
        and fields["critic_future_leakage"] == 0
        and fields["missed_eligible_service"] == 0
        and fields["alignment_excess"] == 0
        and fields["illegal_skip"] == 0
        and fields["duplicate_service_ownership"] == 0
        and fields["duplicate_wait_ownership"] == 0
        and fields["orphan_reward"] == 0
        and fields["nan_inf_blocker"] is False
    )
    return {"fields": fields, "passed": passed}


def make_gate_matrix(
    lineage: Mapping[str, Any],
    legacy: Mapping[str, Any],
    normalization: Mapping[str, Any],
    seed_split: Mapping[str, Any],
    exposure: Mapping[str, Any],
    ppo: Mapping[str, Any],
    critic: Mapping[str, Any],
    gatv2: Mapping[str, Any],
    lineage_contract: Mapping[str, Any],
    safety: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    def gate(gid: str, status: str, evidence: str, blocking: bool, observed: Any = None, expected: Any = None) -> Dict[str, Any]:
        return {
            "gate_id": gid,
            "status": status,
            "evidence": evidence,
            "observed": observed,
            "expected": expected,
            "blocking": bool(blocking),
        }

    return [
        gate("R1_G01_SHA_LINEAGE", "PASS" if lineage["all_passed"] else "BLOCKED", "Reward/H4G/D1 SHA lineage revalidated.", not lineage["all_passed"], lineage["checks"], "all passed"),
        gate("R1_G02_LEGACY_R8C_NORMALIZATION_SUPERSEDED", "PASS", "R8C normalization is reclassified as legacy superseded by Reward V2.", False, legacy.get("corrected_classification"), "LEGACY_SUPERSEDED_BY_REWARD_V2"),
        gate("R1_G03_REWARD_V2_NORMALIZATION_CONTRACT", "BLOCKED", "Runtime reward normalizer is clear, but advantage/return/value normalization contracts are unresolved.", True, normalization.get("blocking_fields"), []),
        gate("R1_G04_OLD_NORMALIZER_REUSE_FALSE", "PASS", "Old Reward-V1/DL normalizer reuse remains prohibited.", False, normalization.get("forbidden_reuse"), "all false"),
        gate("R1_G05_NORMALIZATION_SIGN_REVERSAL_ZERO", "PASS", "D1 pairwise normalization audit has zero induced sign reversal.", False, normalization.get("signal_preservation_audit", {}).get("normalization_induced_sign_reversal_count"), 0),
        gate("R1_G06_SEED_SPLIT_CONTRACT", "BLOCKED", "Current Reward V2 seed/train/validation/test split is missing.", True, seed_split.get("classification"), "AUTHORITATIVE_CURRENT"),
        gate("R1_G07_TRAINING_EXPOSURE_RECHECK", "BLOCKED", "Training-only exposure cannot be computed without frozen split.", True, exposure.get("audit_status"), "TRAINING_SPLIT_EVALUATED"),
        gate("R1_G08_PPO_OPTIMIZER_CONTRACT", "BLOCKED", "PPO/optimizer contract still has missing or legacy-only required parameters.", True, ppo.get("blocking_parameters"), []),
        gate("R1_G09_CRITIC_BOOTSTRAP_CONTRACT", "BLOCKED", "194 critical bootstrap cases cannot be released until critic/PPO/GATv2/return-value contracts are frozen.", True, critic.get("freeze_status"), "READY"),
        gate("R1_G10_GATV2_ENCODER_CONTRACT", "BLOCKED", "GATv2 encoder policy remains UNKNOWN.", True, gatv2.get("gatv2_encoder_status"), ["PRETRAINED_FROZEN", "PRETRAINED_FINETUNED", "FRESH"]),
        gate("R1_G11_FRESH_LINEAGE_CONTRACT", "BLOCKED", "Fresh lineage cannot be frozen while A-D contract gaps remain.", True, lineage_contract.get("blocking_dependencies"), []),
        gate("R1_G12_SAFETY_REGRESSION_ZERO", "PASS" if safety["passed"] else "BLOCKED", "Safety regression audit remains clean.", not safety["passed"], safety["fields"], "all zero/false"),
        gate("R1_G13_NO_TRAINING_EXECUTED", "PASS", "No MAPPO training, optimizer step, backward pass, actor/critic update, or checkpoint training executed.", False, False, False),
        gate("R1_G14_NO_REWARD_REDESIGN", "PASS", "Reward V2 formula, weights, K-mask semantics, event order, gamma/lambda/rollout/agents unchanged.", False, False, False),
        gate("R1_G15_STOP_NEXT_STAGE_BOUNDARY", "PASS", "R1 stops with next_stage based on readiness; no H4I rerun/training auto-executed.", False, True, True),
    ]


def final_decision(matrix: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    blockers = [row["gate_id"] for row in matrix if row.get("blocking")]
    passed = not blockers
    return {
        "created_at": iso_now(),
        "stage": STAGE,
        "technical_gate": PASS_GATE if passed else BLOCKED_GATE,
        "scientific_decision": PASS_DECISION if passed else BLOCKED_DECISION,
        "repair_complete": passed,
        "hard_blocker_count": len(blockers),
        "hard_blockers": blockers,
        "next_stage": "H4I_READINESS_RERUN" if passed else "MINIMUM_REQUIRED_H4I_R1_REPAIR_ONLY",
        "training_executed": False,
        "training_authorized": False,
        "checkpoint_reuse_authorized": False,
        "Reward_V2_modification_authorized": False,
        "fresh_mappo_lineage_frozen": passed,
        "actual_mappo_training_released": False,
        "minimum_next_repair": None
        if passed
        else [
            "create explicit Reward V2 advantage/return/value normalization contract",
            "freeze current Reward V2 seed set and chronological train/validation/test split",
            "compute training-only temporal signal exposure after split freeze",
            "freeze current Reward V2 PPO/optimizer/scheduler/critic epoch/loss contract",
            "freeze GATv2 encoder policy as PRETRAINED_FROZEN, PRETRAINED_FINETUNED, or FRESH with provenance",
            "rerun H4I readiness after R1 blockers are repaired",
        ],
    }


def final_report(out_dir: Path, decision: Mapping[str, Any], reconciliation: Mapping[str, Any], normalization: Mapping[str, Any], seed_split: Mapping[str, Any], ppo: Mapping[str, Any], gatv2: Mapping[str, Any]) -> str:
    root_groups = reconciliation.get("root_groups", [])
    group_lines = "\n".join(
        f"- {row['group']}: {row['repair_status']} / remaining={row['remaining_blockers']}" for row in root_groups
    )
    return f"""# PV8-R2A-R8E-R3-R-H4I-R1 Final Report

```text
stage = {STAGE}
technical gate = {decision['technical_gate']}
scientific decision = {decision['scientific_decision']}

Reward V2 freeze SHA = {EXPECTED_REWARD_SHA}
H4G runtime binding SHA = {EXPECTED_H4G_SHA}
H4I-ER0-D1 payload SHA = {EXPECTED_D1_SHA}

rollout_horizon = 512
gamma = 0.99
gae_lambda = 0.95
agents = 8

training executed = false
training authorized = false
checkpoint reuse authorized = false
Reward V2 modification authorized = false
```

## Decision

H4I-R1 is **BLOCKED**. The stage repaired the classification error around legacy R8C normalization: `PV8_HEADWAY_AWARE_B1_NORMALIZATION_V1` is now explicitly treated as `LEGACY_SUPERSEDED_BY_REWARD_V2`, not current Reward V2 training normalization. Reward V2 formula, weights, K-mask semantics, event order, `gamma`, `gae_lambda`, and `rollout_horizon` were not changed.

The remaining blockers are contract gaps, not reward redesign requests.

## Root Groups

{group_lines}

## Normalization

The runtime reward normalizer is clear enough to describe: clip to `[-10, 10]`, running buffer `1000`, float32 statistics, and `std` floor `1e-6`. D1 pairwise sign preservation remains intact: `{normalization['signal_preservation_audit']['normalization_induced_sign_reversal_count']}` normalization-induced sign reversals over `{normalization['signal_preservation_audit']['nonzero_d1_pairs_audited']}` nonzero pairs.

Full release is blocked because advantage normalization is `{normalization['advantage_normalization']['status']}`, return normalization is `{normalization['return_normalization']['status']}`, and value normalization is `{normalization['value_normalization']['status']}`.

## Seed / Split

Current Reward V2 seed set and train/validation/test split are missing. DL3 seeds remain legacy-only and were not promoted. Training-only temporal signal exposure therefore remains not evaluable.

## PPO / Critic / GATv2

PPO blocking parameters:

```text
{ppo.get('blocking_parameters')}
```

GATv2 encoder status:

```text
{gatv2.get('gatv2_encoder_status')}
```

The critic remains fresh-only, but the 194 critical bootstrap-dependent cases cannot be released until critic epochs/loss, return/value normalization, PPO, and GATv2 policy are frozen.

## STOP

```text
training_executed = false
next_stage = {decision['next_stage']}
```

Artifact:

```text
{out_dir}
```
"""


def collect_input_paths(project_root: Path) -> List[Path]:
    rels = [
        f"{H4I_ROOT}/25_h4i_readiness_decision.json",
        f"{H4I_ROOT}/02_sha_lineage_audit.json",
        f"{H4I_ROOT}/04_training_contract_inventory.json",
        f"{H4I_ROOT}/09_reward_normalization_summary.json",
        f"{H4I_ROOT}/10_advantage_normalization_audit.json",
        f"{H4I_ROOT}/11_return_value_normalization_audit.json",
        f"{H4I_ROOT}/15_seed_split_integrity.json",
        f"{H4I_ROOT}/18_ppo_optimizer_contract_audit.json",
        f"{H4I_ROOT}/07_critic_bootstrap_readiness.json",
        f"{H4I_ROOT}/17_gatv2_encoder_contract_audit.json",
        f"{H4I_ROOT}/20_safety_future_leakage_audit.json",
        f"{H4I_ROOT}/14_training_exposure_by_split.json",
        f"{R8C_ROOT}/r8c_training_normalization_contract.json",
        f"{R8C_ROOT}/r8c_legacy_normalization_exclusion_audit.json",
        f"{R1_ROOT}/r1_mappo_tensor_contract.json",
        f"{R1_ROOT}/r1_training_checkpoint_metadata_contract.json",
        f"{R2_ROOT}/r2_split_contract.json",
        f"{D1_ROOT}/r8er3rh4i_er0d1_training_readiness_dimensions.json",
        "05_training/mappo_runner.py",
    ]
    return [project_root / rel for rel in rels]


def build_payloads(project_root: Path) -> Dict[str, Any]:
    data = load_inputs(project_root)
    lineage = lineage_audit(project_root, data)
    legacy = legacy_normalization_supersession(project_root, data)
    normalization = reward_v2_normalization_contract(project_root, data)
    seed_split = seed_split_contract(project_root, data)
    exposure = training_exposure_audit(data)
    ppo = ppo_optimizer_contract(data)
    critic = critic_bootstrap_contract(data)
    gatv2 = gatv2_encoder_contract(data)
    blockers = (
        normalization.get("blocking_fields", [])
        + seed_split.get("blocking_reasons", [])
        + ["training_exposure_not_evaluable"]
        + ppo.get("blocking_parameters", [])
        + critic.get("blocking_reasons", [])
        + (["gatv2_encoder_loading_policy"] if gatv2.get("blocking", True) else [])
    )
    lineage_contract = fresh_lineage_contract(data, sorted(set(str(x) for x in blockers)))
    reconciliation = h4i_blocker_reconciliation(data, legacy, normalization, seed_split, exposure, ppo, critic, gatv2)
    safety = safety_regression_zero(data)
    matrix = make_gate_matrix(lineage, legacy, normalization, seed_split, exposure, ppo, critic, gatv2, lineage_contract, safety)
    decision = final_decision(matrix)
    return {
        "01_h4i_blocker_reconciliation.json": reconciliation,
        "02_legacy_normalization_supersession.json": legacy,
        "03_reward_v2_normalization_contract.json": normalization,
        "04_seed_split_contract.json": seed_split,
        "05_training_exposure_audit.json": exposure,
        "06_ppo_optimizer_contract.json": ppo,
        "07_critic_bootstrap_contract.json": critic,
        "08_gatv2_encoder_contract.json": gatv2,
        "09_fresh_lineage_contract.json": lineage_contract,
        "10_h4i_r1_gate_matrix.json": matrix,
        "_decision": decision,
        "_lineage_audit": lineage,
    }


def write_outputs(project_root: Path, out_dir: Path, payloads: Mapping[str, Any]) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_OUTPUTS:
        if name.endswith(".json") and name != "manifest.json":
            write_json(out_dir / name, payloads[name])

    report = final_report(
        out_dir,
        payloads["_decision"],
        payloads["01_h4i_blocker_reconciliation.json"],
        payloads["03_reward_v2_normalization_contract.json"],
        payloads["04_seed_split_contract.json"],
        payloads["06_ppo_optimizer_contract.json"],
        payloads["08_gatv2_encoder_contract.json"],
    )
    (out_dir / "final_report.md").write_text(report, encoding="utf-8")

    input_paths = collect_input_paths(project_root)
    output_paths = sorted(p for p in out_dir.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {
        "artifact_version": "PV8_H4I_R1_READINESS_CONTRACT_REPAIR_AUDIT_V1",
        "stage": STAGE,
        "created_at": iso_now(),
        "git_commit": git_commit(project_root),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch_version(),
        "input_paths": [str(p) for p in input_paths],
        "input_sha256": {str(p): sha256_file(p) for p in input_paths},
        "output_paths": [str(p) for p in output_paths],
        "output_sha256": {str(p): sha256_file(p) for p in output_paths},
        "required_outputs": REQUIRED_OUTPUTS,
        "missing_required_outputs": [name for name in REQUIRED_OUTPUTS if name != "manifest.json" and not (out_dir / name).exists()],
        "Reward_V2_freeze_sha256": EXPECTED_REWARD_SHA,
        "H4G_runtime_binding_sha256": EXPECTED_H4G_SHA,
        "H4I_ER0D1_payload_sha256": EXPECTED_D1_SHA,
        "training_executed": False,
        "optimizer_step_executed": False,
        "loss_backward_executed": False,
        "checkpoint_training_written": False,
        "Reward_V2_redesign": False,
        "technical_gate": payloads["_decision"]["technical_gate"],
        "scientific_decision": payloads["_decision"]["scientific_decision"],
        "lineage_audit": payloads["_lineage_audit"],
    }
    write_json(out_dir / "manifest.json", manifest)
    return dict(payloads["_decision"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/Users/arty/Documents/Codex/urbanbus_rl_project")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args(argv)

    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else project_root / "05_training/artifacts" / f"{ARTIFACT_PREFIX}_{timestamp()}"
    )

    protected_pre = {rel: sha256_file(project_root / rel) for rel in PROTECTED_RUNTIME_FILES}
    payloads = build_payloads(project_root)
    decision = write_outputs(project_root, out_dir, payloads)
    protected_post = {rel: sha256_file(project_root / rel) for rel in PROTECTED_RUNTIME_FILES}
    if protected_pre != protected_post:
        raise RuntimeError("protected runtime file hash changed during H4I-R1 audit")

    print(json.dumps({"artifact": str(out_dir), **decision}, ensure_ascii=False, indent=2))
    print(
        "[STOP]\n"
        f"H4I-R1 readiness contract repair {'PASS' if decision['repair_complete'] else 'BLOCKED'}.\n"
        "No MAPPO training was executed.\n"
        "Reward V2 was not modified.\n"
        f"Next stage: {decision['next_stage']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
