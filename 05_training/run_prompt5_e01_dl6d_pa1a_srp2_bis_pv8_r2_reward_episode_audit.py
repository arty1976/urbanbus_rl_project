#!/usr/bin/env python3
"""PV8-R2 reward materialization and prospective episode split audit.

This runner is intentionally fail-closed.  It audits whether a separately
approved PV8 reward contract exists before producing any reward value.  It
never repurposes historical reward scaffolds, fills missing reward inputs with
zero, assigns a split from a single episode, or executes a policy/training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2_reward_episode_audit.py"

K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"
R1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r1_mappo_retraining_preflight_20260808_142604"

UPSTREAMS = {
    "PV8-K8": (
        K8_ROOT,
        "artifact_manifest_srp2_bis_pv8_k8.json",
        "_PV8_K8_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE",
    ),
    "PV8-K9": (
        K9_ROOT,
        "artifact_manifest_srp2_bis_pv8_k9.json",
        "_PV8_K9_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED",
    ),
    "PV8-R1": (
        R1_ROOT,
        "artifact_manifest_srp2_bis_pv8_r1.json",
        "_PV8_R1_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R1_MAPPO_RETRAINING_PREFLIGHT_COMPLETE",
    ),
}

EXPECTED_RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
EXPECTED_OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"
EXPECTED_TENSORS = {"num_agents": 8, "actor_obs_dim": 16, "critic_obs_dim": 64, "action_dim": 3}
REWARD_VERSION = "PV8_REWARD_V1_PRETRAINING_CONTRACT"

R1_EPISODE_PATH = R1_ROOT / "r1_episode_coverage.parquet"
R1_MANIFEST_PATH = R1_ROOT / "r1_frozen_episode_manifest.json"
R1_TENSOR_PATH = R1_ROOT / "r1_mappo_tensor_contract.json"
R1_CHECKPOINT_CONTRACT_PATH = R1_ROOT / "r1_training_checkpoint_metadata_contract.json"
R1_READINESS_PATH = R1_ROOT / "r1_retraining_readiness_decision.json"
K8_CONTRACT_PATH = K8_ROOT / "k8_frozen_rule_contract.json"
K9_BINDING_PATH = K9_ROOT / "k9_version_hash_binding_audit.json"
K6_OCCURRENCE_PATH = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026" / "k6_route_stop_occurrence_master.parquet"

FINAL_REWARD_SPEC_PATH = TRAINING_ROOT / "rewards" / "final_reward_spec_step111.json"
STEP110_PATH = TRAINING_ROOT / "adapters" / "final_reward_design_review_gate_step110.py"
HISTORICAL_REWARD_PATH = TRAINING_ROOT / "rewards" / "mappo_reward_v1.py"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2_reward_episode_audit"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2_REWARD_AND_EPISODE_DATA_AUDIT_COMPLETE"
FAIL_GATE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2_REWARD_AND_EPISODE_DATA_AUDIT_FAILED"
READINESS = "SRP2_BIS_PV8_R2_AUDIT_COMPLETE_REWARD_AND_MULTI_EPISODE_DATA_BLOCKED"
DECISION_READY = "PV8_TRAINING_DATA_READY"
DECISION_EXPANSION = "PV8_EPISODE_EXPANSION_REQUIRED"
DECISION_REWARD_BLOCKED = "PV8_REWARD_MATERIALIZATION_BLOCKED"

PAYLOADS = [
    "r2_reward_materialization.parquet",
    "r2_reward_audit.json",
    "r2_episode_manifest.parquet",
    "r2_episode_manifest.json",
    "r2_split_contract.json",
    "r2_train_manifest.json",
    "r2_validation_manifest.json",
    "r2_test_manifest.json",
    "r2_dataset_coverage.json",
    "r2_training_data_readiness.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]

REWARD_COLUMNS = [
    "episode_id", "cycle_index", "cycle_timestamp", "agent_id", "vehicle_token",
    "route_stop_occurrence_id", "active_bus_mask", "reward_total", "reward_components_json",
    "reward_version", "reward_input_completeness", "reward_value_materialized",
    "learning_sample_eligible", "materialization_status", "critical_missing_fields_json",
    "known_decision_input_fields_json", "next_state_binding_status", "no_future_input_source",
]

EPISODE_COLUMNS = [
    "episode_id", "chronological_episode_index", "decision_ts_start", "decision_ts_end",
    "date_start", "date_end", "agent_snapshot_count", "active_snapshot_count",
    "inactive_snapshot_count", "active_transition_candidate_count", "reward_valid_transition_count",
    "source_episode_manifest_sha256", "episode_feature_hash", "next_state_binding_status",
    "learning_eligible", "split_assignment", "source_scope", "c1_retrospective_vehicle_selection_used",
    "no_future_leakage", "static_rulebook_sha256", "occurrence_master_sha256",
]


class R2Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(k5.json_clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_upstreams() -> Dict[str, Any]:
    verified: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed_gate != expected_gate or not k5.manifest_ok(checks):
            raise R2Error(f"{label} integrity failure: gate={observed_gate}, checks={checks}")
        verified[label] = {
            "artifact_root": str(root),
            "gate": observed_gate,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return verified


def verify_frozen_contract() -> Dict[str, Any]:
    tensor = k5.read_json(R1_TENSOR_PATH)
    checkpoint = k5.read_json(R1_CHECKPOINT_CONTRACT_PATH)
    r1_manifest = k5.read_json(R1_MANIFEST_PATH)
    k8_contract = k5.read_json(K8_CONTRACT_PATH)
    k9_binding = k5.read_json(K9_BINDING_PATH)
    rulebook_path = Path(k8_contract["static_rulebook_source_path"])
    required_metadata = checkpoint.get("required_metadata", {})
    actual = {
        "num_agents": tensor.get("num_agents"),
        "actor_obs_dim": tensor.get("actor_obs_dim"),
        "critic_obs_dim": tensor.get("critic_obs_dim"),
        "action_dim": tensor.get("action_dim"),
    }
    tensor_matches = {key: actual[key] == value for key, value in EXPECTED_TENSORS.items()}
    checks = {
        "expected_tensors": EXPECTED_TENSORS,
        "observed_tensors": actual,
        "tensor_matches": tensor_matches,
        "r1_reward_version_matches": tensor.get("reward_version") == REWARD_VERSION,
        "checkpoint_reward_version_matches": required_metadata.get("reward_version") == REWARD_VERSION,
        "checkpoint_agent_action_metadata_matches": all(
            required_metadata.get(key) == EXPECTED_TENSORS[key] for key in ("num_agents", "action_dim")
        ),
        "r1_manifest_rulebook_hash_matches": r1_manifest.get("frozen_bindings", {}).get("rulebook_sha256") == EXPECTED_RULEBOOK_SHA256,
        "r1_manifest_occurrence_hash_matches": r1_manifest.get("frozen_bindings", {}).get("occurrence_master_sha256") == EXPECTED_OCCURRENCE_SHA256,
        "k8_rulebook_hash_matches": k8_contract.get("static_rulebook_sha256") == EXPECTED_RULEBOOK_SHA256,
        "k8_occurrence_hash_matches": k8_contract.get("occurrence_master_sha256") == EXPECTED_OCCURRENCE_SHA256,
        "k9_rulebook_hash_matches": k9_binding.get("rulebook_sha256") == EXPECTED_RULEBOOK_SHA256,
        "k9_occurrence_hash_matches": k9_binding.get("occurrence_master_sha256") == EXPECTED_OCCURRENCE_SHA256,
        "actual_rulebook_sha256": k5.sha256_file(rulebook_path),
        "actual_occurrence_master_sha256": k5.sha256_file(K6_OCCURRENCE_PATH),
    }
    checks["actual_rulebook_hash_matches"] = checks["actual_rulebook_sha256"] == EXPECTED_RULEBOOK_SHA256
    checks["actual_occurrence_hash_matches"] = checks["actual_occurrence_master_sha256"] == EXPECTED_OCCURRENCE_SHA256
    boolean_checks = [
        *tensor_matches.values(),
        checks["r1_reward_version_matches"],
        checks["checkpoint_reward_version_matches"],
        checks["checkpoint_agent_action_metadata_matches"],
        checks["r1_manifest_rulebook_hash_matches"],
        checks["r1_manifest_occurrence_hash_matches"],
        checks["k8_rulebook_hash_matches"],
        checks["k8_occurrence_hash_matches"],
        checks["k9_rulebook_hash_matches"],
        checks["k9_occurrence_hash_matches"],
        checks["actual_rulebook_hash_matches"],
        checks["actual_occurrence_hash_matches"],
    ]
    checks["failure_count"] = sum(not value for value in boolean_checks)
    if checks["failure_count"]:
        raise R2Error(f"frozen PV8 contract mismatch: {checks}")
    return checks


def reward_contract_audit(episode_rows: pd.DataFrame) -> Dict[str, Any]:
    final_spec = k5.read_json(FINAL_REWARD_SPEC_PATH)
    step110_source = STEP110_PATH.read_text(encoding="utf-8-sig")
    historical_source = HISTORICAL_REWARD_PATH.read_text(encoding="utf-8-sig")
    readiness = k5.read_json(R1_READINESS_PATH)
    serialized = [json.loads(value) for value in episode_rows["reward_required_fields_json"].tolist()]
    materialized_flags = [bool(value.get("reward_value_materialized")) for value in serialized]
    reward_versions = sorted({str(value.get("reward_version")) for value in serialized})
    candidate_sources = [
        {
            "path": str(R1_CHECKPOINT_CONTRACT_PATH),
            "classification": "PV8_IDENTIFIER_ONLY_NOT_MATERIALIZED",
            "reward_version": REWARD_VERSION,
            "approved_for_pv8_materialization": False,
            "evidence": "R1 required metadata names a pretraining contract; R1 readiness records reward_values_materialized=false.",
        },
        {
            "path": str(FINAL_REWARD_SPEC_PATH),
            "classification": "HISTORICAL_DRAFT_NOT_TRAINABLE",
            "approved_for_pv8_materialization": False,
            "evidence": {
                "spec_status": final_spec.get("spec_status"),
                "reward_formula_finalized": final_spec.get("claim_guards", {}).get("reward_formula_finalized"),
                "reward_weights_locked": final_spec.get("claim_guards", {}).get("reward_weights_locked"),
                "train_with_this_reward_allowed": final_spec.get("claim_guards", {}).get("train_with_this_reward_allowed"),
            },
        },
        {
            "path": str(STEP110_PATH),
            "classification": "HISTORICAL_REWARD_REVIEW_GUARD_LOCKED",
            "approved_for_pv8_materialization": False,
            "evidence_tokens_present": {
                "reward_weights_locked_false": '"reward_weights_locked": False' in step110_source,
                "reward_formula_finalized_false": '"reward_formula_finalized": False' in step110_source,
                "train_reward_promotion_approved_false": '"train_reward_promotion_approved": False' in step110_source,
            },
        },
        {
            "path": str(HISTORICAL_REWARD_PATH),
            "classification": "HISTORICAL_DEFAULT_ZERO_FILL_INCOMPATIBLE",
            "approved_for_pv8_materialization": False,
            "evidence": "Historical helper defines a default-zero metric conversion; PV8-R2 forbids silent zero filling for a missing critical reward input.",
            "default_zero_converter_present": "default: float = 0.0" in historical_source,
        },
    ]
    approved = any(bool(item["approved_for_pv8_materialization"]) for item in candidate_sources)
    if approved:
        raise R2Error("unexpected approved PV8 reward contract candidate; update the explicit R2 audit before materialization")
    return {
        "created_at": iso_kst(),
        "approved_pv8_reward_contract_found": False,
        "approved_pv8_reward_contract_path": None,
        "approved_reward_formula_or_weight_values_used": False,
        "reward_version_identifier": REWARD_VERSION,
        "source_reward_versions": reward_versions,
        "r1_reward_value_materialized_count": sum(materialized_flags),
        "r1_reward_value_unmaterialized_count": len(materialized_flags) - sum(materialized_flags),
        "r1_readiness_reward_values_materialized": readiness.get("reward_values_materialized"),
        "candidate_sources": candidate_sources,
        "fail_closed_reason": "No separately approved PV8 reward formula, weights, normalization, hard constraints, or reward-input contract is available.",
        "historical_reward_reuse": False,
        "historical_reward_weights_used": False,
        "silent_zero_fill_used": False,
    }


def build_blocked_reward_rows(episode_rows: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    active = episode_rows[episode_rows["active_bus_mask"].astype(bool)].copy()
    last_cycle = int(episode_rows["cycle_index"].max())
    for source in active.sort_values(["cycle_index", "agent_id"], kind="mergesort").itertuples(index=False):
        required = json.loads(source.reward_required_fields_json)
        has_successor_cycle = int(source.cycle_index) < last_cycle
        missing = [
            "approved_pv8_reward_contract",
            "locked_reward_formula_and_weights",
            "approved_reward_normalization_and_hard_constraints",
            "executed_action_outcome",
            "reward_metric_outcome_fields",
        ]
        if has_successor_cycle:
            missing.append("materialized_successor_state_binding")
            next_state_status = "SUCCESSOR_CYCLE_EXISTS_BUT_BINDING_NOT_MATERIALIZED"
        else:
            missing.append("terminal_or_successor_state_binding")
            next_state_status = "EPISODE_FINAL_CYCLE_NO_SUCCESSOR_CYCLE"
        known_fields = sorted(key for key, value in required.items() if value is not None)
        rows.append(
            {
                "episode_id": str(source.episode_id),
                "cycle_index": int(source.cycle_index),
                "cycle_timestamp": str(source.cycle_timestamp),
                "agent_id": int(source.agent_id),
                "vehicle_token": str(source.vehicle_token),
                "route_stop_occurrence_id": str(source.route_stop_occurrence_id),
                "active_bus_mask": True,
                "reward_total": None,
                "reward_components_json": None,
                "reward_version": REWARD_VERSION,
                "reward_input_completeness": "INCOMPLETE_FAIL_CLOSED",
                "reward_value_materialized": False,
                "learning_sample_eligible": False,
                "materialization_status": "BLOCKED_NO_APPROVED_PV8_REWARD_CONTRACT",
                "critical_missing_fields_json": json.dumps(missing, ensure_ascii=False, sort_keys=True),
                "known_decision_input_fields_json": json.dumps(known_fields, ensure_ascii=False, sort_keys=True),
                "next_state_binding_status": next_state_status,
                "no_future_input_source": bool(required.get("no_future_feature_source")),
            }
        )
    summary = {
        "active_snapshot_candidate_count": int(len(active)),
        "active_nonterminal_transition_candidate_count": int((active["cycle_index"] < last_cycle).sum()),
        "active_terminal_cycle_candidate_count": int((active["cycle_index"] == last_cycle).sum()),
        "inactive_excluded_from_actor_action_reward_learning_count": int((~episode_rows["active_bus_mask"].astype(bool)).sum()),
        "reward_valid_transition_count": 0,
        "reward_value_materialized_count": 0,
        "reward_value_withheld_count": int(len(rows)),
        "reward_total_zero_fill_count": 0,
        "reward_component_zero_fill_count": 0,
        "reward_nan_count": 0,
        "reward_inf_count": 0,
        "all_candidate_rows_fail_closed": all(row["reward_input_completeness"] == "INCOMPLETE_FAIL_CLOSED" for row in rows),
    }
    return rows, summary


def build_episode_manifest(episode_rows: pd.DataFrame, reward_summary: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source = k5.read_json(R1_MANIFEST_PATH)
    episode_ids = sorted(str(value) for value in episode_rows["episode_id"].unique())
    if episode_ids != ["PV8_PROSPECTIVE_EPISODE_0001"]:
        raise R2Error(f"R1 source must contain exactly its one frozen prospective episode, got {episode_ids}")
    timestamp_values = episode_rows.loc[
        episode_rows["active_bus_mask"].astype(bool) & episode_rows["cycle_timestamp"].notna(), "cycle_timestamp"
    ]
    timestamps = pd.to_datetime(timestamp_values, utc=True, errors="raise", format="ISO8601")
    row = {
        "episode_id": episode_ids[0],
        "chronological_episode_index": 1,
        "decision_ts_start": timestamps.min().isoformat(),
        "decision_ts_end": timestamps.max().isoformat(),
        "date_start": timestamps.min().date().isoformat(),
        "date_end": timestamps.max().date().isoformat(),
        "agent_snapshot_count": int(len(episode_rows)),
        "active_snapshot_count": int(episode_rows["active_bus_mask"].astype(bool).sum()),
        "inactive_snapshot_count": int((~episode_rows["active_bus_mask"].astype(bool)).sum()),
        "active_transition_candidate_count": int(reward_summary["active_nonterminal_transition_candidate_count"]),
        "reward_valid_transition_count": 0,
        "source_episode_manifest_sha256": k5.sha256_file(R1_MANIFEST_PATH),
        "episode_feature_hash": source["episode_feature_hash"],
        "next_state_binding_status": "NOT_MATERIALIZED_PENDING_APPROVED_REWARD_CONTRACT",
        "learning_eligible": False,
        "split_assignment": None,
        "source_scope": "PV8_R1_FROZEN_PROSPECTIVE_EPISODE_ONLY",
        "c1_retrospective_vehicle_selection_used": False,
        "no_future_leakage": bool(source["no_future_leakage"]),
        "static_rulebook_sha256": EXPECTED_RULEBOOK_SHA256,
        "occurrence_master_sha256": EXPECTED_OCCURRENCE_SHA256,
    }
    details = {
        "created_at": iso_kst(),
        "available_prospective_episode_count": 1,
        "available_episode_ids": episode_ids,
        "prospective_multi_episode_expansion_built": False,
        "episode_expansion_status": "BLOCKED_ONE_FROZEN_PROSPECTIVE_EPISODE_ONLY",
        "synthetic_episode_generation_count": 0,
        "retrospective_c1_vehicle_selection_reused": False,
        "next_state_binding": {
            "required_for_learning": True,
            "source_has_explicit_materialized_binding": False,
            "reason": "R1 freezes decision-time snapshots; R2 may not create learning transitions before the approved reward contract exists.",
        },
        "episode_rows": [row],
    }
    return [row], details


def build_split_contract(episode_manifest_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    source_count = len(episode_manifest_rows)
    feasible = source_count >= 3 and all(bool(row["learning_eligible"]) for row in episode_manifest_rows)
    reason = (
        "At least three chronologically distinct, reward-valid prospective episodes are required for non-empty train/validation/test partitions."
        if source_count < 3
        else "One or more prospective episodes has no reward-valid learning transition."
    )
    contract = {
        "created_at": iso_kst(),
        "contract_name": "PV8_R2_CHRONOLOGICAL_EPISODE_SPLIT_V1",
        "split_unit": "episode",
        "randomization_allowed": False,
        "episode_order_key": "decision_ts_start ascending, then episode_id ascending",
        "split_order": ["train", "validation", "test"],
        "allocation_rule_when_eligible": {
            "train_fraction": 0.70,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "minimum_nonempty_partitions": 3,
            "rounding": "allocate validation/test at least one episode each, then assign all remaining earliest episodes to train",
        },
        "mandatory_invariants": {
            "no_episode_overlap": True,
            "no_snapshot_overlap": True,
            "no_future_to_past_leakage": True,
            "strict_chronological_train_before_validation_before_test": True,
        },
        "source_episode_count": source_count,
        "eligible_episode_count": sum(bool(row["learning_eligible"]) for row in episode_manifest_rows),
        "split_feasible": feasible,
        "split_assignment_count": 0,
        "blocked_reason": reason if not feasible else None,
        "no_episode_overlap_check": True,
        "no_snapshot_overlap_check": True,
        "future_to_past_leakage_count": 0,
        "chronological_order_check": "NOT_APPLICABLE_NO_ASSIGNMENTS" if not feasible else "PENDING_ASSIGNMENT",
    }
    common = {
        "created_at": iso_kst(),
        "split_contract": "PV8_R2_CHRONOLOGICAL_EPISODE_SPLIT_V1",
        "episode_ids": [],
        "episode_count": 0,
        "snapshot_count": 0,
        "learning_transition_count": 0,
        "assignment_status": "UNASSIGNED_BLOCKED",
        "blocked_reason": reason,
    }
    manifests = tuple({**common, "partition": partition} for partition in ("train", "validation", "test"))
    return contract, manifests[0], manifests[1], manifests[2]


def coverage_audit(episode_rows: pd.DataFrame, reward_summary: Mapping[str, Any], split_contract: Mapping[str, Any]) -> Dict[str, Any]:
    active = episode_rows[episode_rows["active_bus_mask"].astype(bool)].copy()
    masks = active["k_action_mask_json"].map(json.loads)
    timestamp_values = episode_rows.loc[
        episode_rows["active_bus_mask"].astype(bool) & episode_rows["cycle_timestamp"].notna(), "cycle_timestamp"
    ]
    timestamps = pd.to_datetime(timestamp_values, utc=True, errors="raise", format="ISO8601")
    per_agent = []
    for agent_id, group in episode_rows.groupby("agent_id", sort=True):
        active_group = group[group["active_bus_mask"].astype(bool)]
        action_masks = active_group["k_action_mask_json"].map(json.loads).tolist()
        per_agent.append(
            {
                "agent_id": int(agent_id),
                "snapshot_count": int(len(group)),
                "active_snapshot_count": int(len(active_group)),
                "inactive_snapshot_count": int(len(group) - len(active_group)),
                "hold_available_count": sum(bool(mask[0]) for mask in action_masks),
                "serve_available_count": sum(bool(mask[1]) for mask in action_masks),
                "conditional_skip_available_count": sum(bool(mask[2]) for mask in action_masks),
                "reward_valid_transition_count": 0,
            }
        )
    return {
        "created_at": iso_kst(),
        "episode_count": 1,
        "cycle_count": int(episode_rows["cycle_index"].nunique()),
        "agent_snapshot_count": int(len(episode_rows)),
        "active_snapshot_count": int(len(active)),
        "inactive_snapshot_count": int(len(episode_rows) - len(active)),
        "reward_valid_transition_count": int(reward_summary["reward_valid_transition_count"]),
        "reward_value_materialized_count": int(reward_summary["reward_value_materialized_count"]),
        "per_action_availability": {
            "HOLD": sum(bool(mask[0]) for mask in masks),
            "SERVE": sum(bool(mask[1]) for mask in masks),
            "CONDITIONAL_SKIP": sum(bool(mask[2]) for mask in masks),
        },
        "active_all_false_mask_count": sum(not any(mask) for mask in masks),
        "per_agent_coverage": per_agent,
        "time_band_utc": {"start": timestamps.min().isoformat(), "end": timestamps.max().isoformat()},
        "date_coverage_utc": sorted({value.date().isoformat() for value in timestamps}),
        "split_sizes": {"train": 0, "validation": 0, "test": 0},
        "split_feasible": bool(split_contract["split_feasible"]),
        "no_future_leakage": True,
        "c1_retrospective_vehicle_selection_used": False,
        "nan_inf_audit": {"reward_nan_count": 0, "reward_inf_count": 0, "reason": "No reward values were produced."},
    }


def write_manifest_and_lock(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for relative_path in PAYLOADS:
        path = writer.root / relative_path
        rows.append(
            {
                "relative_path": relative_path,
                "size_bytes": path.stat().st_size if path.exists() else None,
                "sha256": k5.sha256_file(path) if path.exists() else None,
                "required": True,
                "artifact_role": Path(relative_path).stem,
                "exists": path.exists(),
            }
        )
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2.jsonl"
    writer.text(jsonl_name, "".join(json.dumps(k5.json_clean(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    jsonl_path = writer.root / jsonl_name
    rows.append(
        {
            "relative_path": jsonl_name,
            "size_bytes": jsonl_path.stat().st_size,
            "sha256": k5.sha256_file(jsonl_path),
            "required": True,
            "artifact_role": "manifest_jsonl",
            "exists": True,
        }
    )
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2.json"
    writer.json(
        manifest_name,
        {
            "created_at": iso_kst(),
            "artifact_family": ARTIFACT_PREFIX,
            "terminal_gate": gate["gate"],
            "readiness": gate["readiness"],
            "payload_count": len(rows),
            "missing_payload_count": sum(not row["exists"] for row in rows),
            "files": rows,
        },
    )
    manifest_path = writer.root / manifest_name
    writer.json(
        "_PV8_R2_COMPLETE.lock",
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


def final_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# PV8-R2 Reward Materialization and Episode Data Audit",
            "",
            f"- artifact root: `{summary['artifact_root']}`",
            f"- gate: `{summary['gate']}`",
            f"- decision: `{summary['decision']}`",
            f"- frozen tensor contract: `{summary['num_agents']} agents / actor {summary['actor_dim']} / critic {summary['critic_dim']} / {summary['action_dim']} actions`",
            f"- available prospective episodes: `{summary['episodes']}`",
            f"- active / inactive snapshot candidates: `{summary['active']} / {summary['inactive']}`",
            f"- active nonterminal transition candidates / reward-valid transitions: `{summary['transition_candidates']} / {summary['reward_valid']}`",
            f"- train / validation / test assignments: `0 / 0 / 0`",
            "",
            "## Decision",
            "",
            "The audit completed deterministically, but reward materialization is blocked. R1 records only the PV8 pretraining reward identifier and explicitly records `reward_values_materialized=false`. No separately approved PV8 reward formula, weights, normalization, hard constraints, or decision-to-outcome input contract was found.",
            "",
            "The 214 active candidate rows were written with `reward_total=null`, `reward_components_json=null`, and `INCOMPLETE_FAIL_CLOSED`; no value was zero-filled. The 26 inactive snapshots remain excluded from actor/action/reward learning samples.",
            "",
            "## Episode and Split Status",
            "",
            "Only one frozen prospective 30-cycle episode is available. R2 did not fabricate additional episodes, reuse C1 retrospective vehicle selection, or assign the sole episode to a split. The frozen chronological split contract requires at least three reward-valid chronological episodes, one for each non-empty partition.",
            "",
            "## Minimum Repair Before First PV8 MAPPO Training",
            "",
            "1. Approve and freeze a PV8-specific reward formula, weights, normalization, hard constraints, and required decision-to-outcome reward inputs.",
            "2. Materialize explicit no-future next-state/outcome bindings and finite reward values for active transitions under that frozen contract.",
            "3. Expand the prospective collection to at least three chronologically ordered, non-overlapping reward-valid episodes, then apply the frozen episode-level split contract.",
            "",
            "No MAPPO training, policy execution, checkpoint reuse, policy evaluation, API call, DB query, or DB write occurred in R2.",
            "",
        ]
    )


def run_audit(root: Path) -> Path:
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    upstreams = verify_upstreams()
    frozen = verify_frozen_contract()
    episode_rows = pd.read_parquet(R1_EPISODE_PATH).sort_values(["cycle_index", "agent_id"], kind="mergesort").reset_index(drop=True)
    if len(episode_rows) != 240 or int(episode_rows["agent_id"].nunique()) != 8:
        raise R2Error("R1 frozen prospective episode source must contain exactly 240 rows across 8 fixed agents")
    if not bool(episode_rows["active_bus_mask"].astype(bool).any()):
        raise R2Error("R1 prospective episode has no active agent snapshots")

    reward_evidence = reward_contract_audit(episode_rows)
    reward_rows, reward_summary = build_blocked_reward_rows(episode_rows)
    episode_manifest_rows, episode_details = build_episode_manifest(episode_rows, reward_summary)
    split_contract, train_manifest, validation_manifest, test_manifest = build_split_contract(episode_manifest_rows)
    coverage = coverage_audit(episode_rows, reward_summary, split_contract)

    decision = DECISION_REWARD_BLOCKED
    readiness = {
        "created_at": iso_kst(),
        "final_decision": decision,
        "audit_complete": True,
        "reward_materialization_ready": False,
        "prospective_episode_expansion_ready": False,
        "chronological_train_validation_test_split_ready": False,
        "training_data_ready": False,
        "blocked_reason_priority": "NO_APPROVED_PV8_REWARD_CONTRACT",
        "blockers": [
            "No separately approved PV8 reward formula, weights, normalization, hard constraints, or reward-input contract.",
            "No policy action outcome and no materialized no-future next-state binding for reward transitions.",
            "Only one prospective episode is available; a non-empty chronological train/validation/test split requires at least three reward-valid episodes.",
        ],
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
    }
    guards = {
        "created_at": iso_kst(),
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "conditional_skip_policy_execution_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "automatic_r3_execution_authorized": False,
        "automatic_mappo_training_authorized": False,
        "historical_reward_contract_reuse_authorized": False,
    }
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": READINESS,
        "gate_passed": True,
        "final_decision": decision,
        "failure_reasons": [],
        "readiness_blockers": readiness["blockers"],
    }
    summary = {
        "artifact_root": str(root),
        "gate": gate["gate"],
        "decision": decision,
        "num_agents": EXPECTED_TENSORS["num_agents"],
        "actor_dim": EXPECTED_TENSORS["actor_obs_dim"],
        "critic_dim": EXPECTED_TENSORS["critic_obs_dim"],
        "action_dim": EXPECTED_TENSORS["action_dim"],
        "episodes": coverage["episode_count"],
        "active": coverage["active_snapshot_count"],
        "inactive": coverage["inactive_snapshot_count"],
        "transition_candidates": reward_summary["active_nonterminal_transition_candidate_count"],
        "reward_valid": reward_summary["reward_valid_transition_count"],
    }

    writer.parquet("r2_reward_materialization.parquet", reward_rows, REWARD_COLUMNS)
    writer.json(
        "r2_reward_audit.json",
        {
            "created_at": iso_kst(),
            "reward_contract_evidence": reward_evidence,
            "materialization_summary": reward_summary,
            "component_ranges": {"status": "NOT_COMPUTED_NO_APPROVED_PV8_REWARD_CONTRACT"},
            "nonfinite_audit": {"nan_count": 0, "inf_count": 0, "reason": "No reward value or component was produced."},
            "fail_closed": True,
        },
    )
    writer.parquet("r2_episode_manifest.parquet", episode_manifest_rows, EPISODE_COLUMNS)
    writer.json("r2_episode_manifest.json", episode_details)
    writer.json("r2_split_contract.json", split_contract)
    writer.json("r2_train_manifest.json", train_manifest)
    writer.json("r2_validation_manifest.json", validation_manifest)
    writer.json("r2_test_manifest.json", test_manifest)
    writer.json("r2_dataset_coverage.json", coverage)
    writer.json("r2_training_data_readiness.json", readiness)
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
            "frozen_contract_audit": frozen,
            "new_bis_api_call_count": 0,
            "db_query_count": 0,
            "db_write_count": 0,
            "policy_action_execution_count": 0,
            "policy_evaluation_count": 0,
            "checkpoint_reuse_count": 0,
            "mappo_training_count": 0,
            "r3_execution_count": 0,
            "qwen_train": False,
            "qwen_inference": False,
            "c1_retrospective_vehicle_selection_used": False,
        },
    )
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": gate["gate"], "readiness": gate["readiness"], "final_decision": decision})
    writer.text("final_report.md", final_report(summary))
    write_manifest_and_lock(writer, gate)

    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2.json", "_PV8_R2_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise R2Error(f"R2 artifact integrity failure: {own}")
    print(f"artifact_root: {root}")
    print(f"gate: {gate['gate']}")
    print(f"decision: {decision}")
    print(f"active_reward_candidates: {reward_summary['active_snapshot_candidate_count']}")
    print(f"reward_valid_transitions: {reward_summary['reward_valid_transition_count']}")
    print(f"available_episodes: {coverage['episode_count']}")
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
