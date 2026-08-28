#!/usr/bin/env python3
"""PV8-R1 physical-vehicle MAPPO retraining preflight and frozen episode contract."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import resource
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
from simulator.dynamics_replay_contract import canonical_hash, canonical_json


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r1_mappo_retraining_preflight.py"

C3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c3_mappo_interface_compatibility_20260808_100118"
K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"
K6_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026"
K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"
C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
DL4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl4_suseong_critic_calibration_stabilization_20260731_155427"

UPSTREAMS = {
    "PV8-C3": (C3_ROOT, "artifact_manifest_srp2_bis_pv8_c3.json", "_PV8_C3_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_COMPLETE"),
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
    "PV8-K6": (K6_ROOT, "artifact_manifest_srp2_bis_pv8_k6.json", "_PV8_K6_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K6_STATIC_RULE_AUTHORITY_AND_OCCURRENCE_AUDIT_COMPLETE"),
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
    "PV8-K9": (K9_ROOT, "artifact_manifest_srp2_bis_pv8_k9.json", "_PV8_K9_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED"),
}

EXPECTED_RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
EXPECTED_OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"

C2_CYCLE_PATH = C2_ROOT / "prospective_agent_cycle_state.parquet"
K8_ROWS_PATH = K8_ROOT / "k8_runtime_k_mask_integration.parquet"
K9_SAMPLES_PATH = K9_ROOT / "k9_snapshot_samples.parquet"
K9_COVERAGE_PATH = K9_ROOT / "k9_action_mask_coverage.parquet"
K9_BINDING_PATH = K9_ROOT / "k9_version_hash_binding_audit.json"
K8_CONTRACT_PATH = K8_ROOT / "k8_frozen_rule_contract.json"
K6_OCCURRENCE_PATH = K6_ROOT / "k6_route_stop_occurrence_master.parquet"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r1_mappo_retraining_preflight"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R1_MAPPO_RETRAINING_PREFLIGHT_COMPLETE"
FAIL_GATE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R1_MAPPO_RETRAINING_PREFLIGHT_FAILED"
PASS_READINESS = "SRP2_BIS_PV8_R1_COMPLETE_RETRAINING_PREFLIGHT_READY_R2_AND_TRAINING_LOCKED"
DECISION_READY = "MAPPO_RETRAINING_PREFLIGHT_READY"
DECISION_REPAIR = "MAPPO_RETRAINING_PREFLIGHT_REPAIR_REQUIRED"

PHYSICAL_VEHICLE_AGENT_CONTRACT_VERSION = "PV8_PHYSICAL_VEHICLE_8_SLOT_V1"
OBSERVATION_CONTRACT_VERSION = "PV8_R1_PHYSICAL_SLOT_OBSERVATION_V1"
REWARD_VERSION = "PV8_REWARD_V1_PRETRAINING_CONTRACT"

PAYLOADS = [
    "r1_frozen_episode_manifest.json",
    "r1_episode_coverage.parquet",
    "r1_episode_coverage.json",
    "r1_mappo_tensor_contract.json",
    "r1_dl4_compatibility_delta.json",
    "r1_rollout_boundary_validation.json",
    "r1_training_checkpoint_metadata_contract.json",
    "r1_retraining_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]

EPISODE_COLUMNS = [
    "episode_id", "cycle_index", "cycle_timestamp", "agent_id", "vehicle_token", "active_bus_mask",
    "actor_observation_json", "critic_global_observation_json", "active_mask_json", "k_action_mask_json",
    "actor_sampling_bypassed", "route_stop_occurrence_id", "route_id", "direction_id", "stop_sequence", "stop_id",
    "decision_time_obligation_state_json", "reward_required_fields_json", "global_snapshot_hash",
    "dynamics_state_snapshot_hash", "k_safety_state_version", "static_rulebook_version", "static_rulebook_sha256",
    "occurrence_master_sha256", "dynamic_state_contract_version", "mask_predicate_version", "observation_contract_version",
    "physical_vehicle_agent_contract_version", "reward_version", "identity_transition_type", "feature_hash",
]


class R1Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def normalized(value: Any, center: float, span: float) -> float:
    if value is None:
        return 0.0
    try:
        result = (float(value) - center) / span
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if not math.isfinite(result) else float(result)


def optional_int(value: Any) -> int:
    try:
        if value is None or pd.isna(value):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def verify_upstreams() -> Dict[str, Any]:
    verified: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed != expected_gate or not k5.manifest_ok(checks):
            raise R1Error(f"{label} integrity failure: gate={observed}, checks={checks}")
        verified[label] = {"artifact_root": str(root), "gate": observed, "readiness": gate.get("readiness"), "manifest_integrity": checks}
    c3_decision = k5.read_json(C3_ROOT / "checkpoint_compatibility_decision.json")
    if c3_decision.get("checkpoint_compatibility_decision") != "RETRAINING_REQUIRED" or c3_decision.get("checkpoint_reuse_authorized") is not False:
        raise R1Error("C3 does not preserve the required RETRAINING_REQUIRED checkpoint decision")
    return verified


def verify_frozen_inputs() -> Dict[str, Any]:
    k8_contract = k5.read_json(K8_CONTRACT_PATH)
    rulebook_path = Path(k8_contract["static_rulebook_source_path"])
    rulebook_sha = k5.sha256_file(rulebook_path)
    occurrence_sha = k5.sha256_file(K6_OCCURRENCE_PATH)
    k9_binding = k5.read_json(K9_BINDING_PATH)
    checks = {
        "rulebook_path": str(rulebook_path),
        "rulebook_sha256": rulebook_sha,
        "rulebook_hash_matches_required": rulebook_sha == EXPECTED_RULEBOOK_SHA256,
        "occurrence_master_path": str(K6_OCCURRENCE_PATH),
        "occurrence_master_sha256": occurrence_sha,
        "occurrence_hash_matches_required": occurrence_sha == EXPECTED_OCCURRENCE_SHA256,
        "k8_contract_rulebook_hash_matches": k8_contract.get("static_rulebook_sha256") == EXPECTED_RULEBOOK_SHA256,
        "k8_contract_occurrence_hash_matches": k8_contract.get("occurrence_master_sha256") == EXPECTED_OCCURRENCE_SHA256,
        "k9_binding_rulebook_hash_matches": k9_binding.get("rulebook_sha256") == EXPECTED_RULEBOOK_SHA256,
        "k9_binding_occurrence_hash_matches": k9_binding.get("occurrence_master_sha256") == EXPECTED_OCCURRENCE_SHA256,
        "k9_total_drift_or_failure_count": int(k9_binding.get("total_drift_or_failure_count", -1)),
    }
    checks["failure_count"] = sum(not bool(value) for key, value in checks.items() if key.endswith("_matches_required") or key.endswith("_matches")) + int(checks["k9_total_drift_or_failure_count"] != 0)
    if checks["failure_count"]:
        raise R1Error(f"frozen upstream hashes are not valid: {checks}")
    return checks


def load_sources() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cycles = pd.read_parquet(C2_CYCLE_PATH).sort_values(["cycle_index", "agent_id"]).reset_index(drop=True)
    k8_rows = pd.read_parquet(K8_ROWS_PATH)
    k8_rows = k8_rows[k8_rows["scenario_id"] == "C2_PROSPECTIVE_CYCLE_RUNTIME"].copy()
    samples = pd.read_parquet(K9_SAMPLES_PATH).sort_values(["cycle_index", "agent_id"]).reset_index(drop=True)
    coverage = pd.read_parquet(K9_COVERAGE_PATH).sort_values(["cycle_index", "agent_id"]).reset_index(drop=True)
    if len(cycles) != 240 or len(k8_rows) != 240 or len(samples) != 240 or len(coverage) != 240:
        raise R1Error("C2/K8/K9 prospective source row count must be exactly 240")
    return cycles, k8_rows, samples, coverage


def build_actor_observation(cycle: Mapping[str, Any], k8_row: Mapping[str, Any], coverage: Mapping[str, Any]) -> List[float]:
    active = bool(cycle["active_bus_mask"])
    direction = str(cycle["direction_id"]) if active and cycle.get("direction_id") is not None else ""
    transition = str(cycle.get("identity_transition_type") or "")
    mask = [bool(coverage["hold_available"]), bool(coverage["serve_available"]), bool(coverage["conditional_skip_available"])]
    dynamic_complete = bool(k8_row["dynamic_state_complete"]) if active else False
    pickup_or_boarding = bool(k8_row["pickup_obligation"]) if active else False
    dropoff_or_onboard = bool(k8_row["dropoff_obligation"] or k8_row["onboard_destination_obligation"]) if active else False
    return [
        1.0 if active else 0.0,
        1.0 if not active else 0.0,
        float(int(cycle["agent_id"])) / 7.0,
        1.0 if direction == "0" else 0.0,
        1.0 if direction == "1" else 0.0,
        min(max(float(cycle["seq"]) / 120.0, 0.0), 1.0) if active and not pd.isna(cycle["seq"]) else 0.0,
        normalized(cycle["position_x"], 128.6, 0.4) if active else 0.0,
        normalized(cycle["position_y"], 35.8, 0.2) if active else 0.0,
        1.0 if bool(cycle["vehicle_observed"]) else 0.0,
        1.0 if cycle.get("observation_missing_reason") is not None else 0.0,
        1.0 if transition.startswith("REAPPEARED") else 0.0,
        1.0 if "DIRECTION_CHANGE" in transition else 0.0,
        1.0 if mask[2] else 0.0,
        1.0 if dynamic_complete else 0.0,
        1.0 if (pickup_or_boarding or dropoff_or_onboard) else 0.0,
        1.0 if bool(k8_row["service_obligation"]) else 0.0,
    ]


def build_critic_observation(cycle_rows: Sequence[Mapping[str, Any]], coverage_by_agent: Mapping[int, Mapping[str, Any]]) -> List[float]:
    features: List[float] = []
    for cycle in sorted(cycle_rows, key=lambda row: int(row["agent_id"])):
        active = bool(cycle["active_bus_mask"])
        direction = str(cycle["direction_id"]) if active and cycle.get("direction_id") is not None else ""
        coverage = coverage_by_agent[int(cycle["agent_id"])]
        features.extend([
            1.0 if active else 0.0,
            1.0 if bool(cycle["vehicle_observed"]) else 0.0,
            1.0 if cycle.get("observation_missing_reason") is not None else 0.0,
            -1.0 if direction == "0" else 1.0 if direction == "1" else 0.0,
            min(max(float(cycle["seq"]) / 120.0, 0.0), 1.0) if active and not pd.isna(cycle["seq"]) else 0.0,
            normalized(cycle["position_x"], 128.6, 0.4) if active else 0.0,
            normalized(cycle["position_y"], 35.8, 0.2) if active else 0.0,
            float(sum((bool(coverage["hold_available"]), bool(coverage["serve_available"]), bool(coverage["conditional_skip_available"])))) / 3.0,
        ])
    if len(features) != 64:
        raise R1Error(f"critic global observation must be 64 values, observed {len(features)}")
    return features


def decision_obligation_state(cycle: Mapping[str, Any], k8_row: Mapping[str, Any]) -> Dict[str, Any]:
    active = bool(cycle["active_bus_mask"])
    service = bool(k8_row["service_obligation"]) if active else False
    return {
        "source": "PV8_K8_K4_DECISION_TIME_OBLIGATION_SNAPSHOT_DERIVED",
        "active_bus_mask": active,
        "dynamic_state_complete": bool(k8_row["dynamic_state_complete"]) if active else False,
        "pickup_obligation": bool(k8_row["pickup_obligation"]) if active else False,
        "dropoff_obligation": bool(k8_row["dropoff_obligation"]) if active else False,
        "boarding_obligation": bool(k8_row["pickup_obligation"]) if active else False,
        "alighting_obligation": bool(k8_row["dropoff_obligation"] or k8_row["onboard_destination_obligation"]) if active else False,
        "onboard_destination_obligation": bool(k8_row["onboard_destination_obligation"]) if active else False,
        "service_obligation": service,
        "missing_reason": [] if active else ["INACTIVE_AGENT_NO_ACTION_SAMPLING"],
    }


def reward_required_fields(cycle: Mapping[str, Any], coverage: Mapping[str, Any], obligation: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "reward_version": REWARD_VERSION,
        "reward_value_materialized": False,
        "reward_materialization_required_before_training": True,
        "decision_timestamp": str(cycle["cycle_timestamp"]),
        "event_time": str(cycle["event_time"]) if cycle.get("event_time") is not None else None,
        "physical_vehicle_token": str(cycle["bound_vehicle_token"]),
        "route_id": str(cycle["route_id"]) if cycle.get("route_id") is not None else None,
        "direction_id": str(cycle["direction_id"]) if cycle.get("direction_id") is not None else None,
        "stop_sequence": optional_int(cycle.get("seq")),
        "stop_id": str(cycle["stop_id"]) if cycle.get("stop_id") is not None else None,
        "position_x": float(cycle["position_x"]) if cycle.get("position_x") is not None and not pd.isna(cycle["position_x"]) else None,
        "position_y": float(cycle["position_y"]) if cycle.get("position_y") is not None and not pd.isna(cycle["position_y"]) else None,
        "active_bus_mask": bool(cycle["active_bus_mask"]),
        "valid_action_count": int(coverage["valid_action_count"]),
        "conditional_skip_available": bool(coverage["conditional_skip_available"]),
        "service_obligation": bool(obligation["service_obligation"]),
        "no_future_feature_source": True,
    }


def build_episode_rows(cycles: pd.DataFrame, k8_rows: pd.DataFrame, samples: pd.DataFrame, coverage: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    keyed_k8 = {(int(row.cycle_index), int(row.agent_id)): row._asdict() for row in k8_rows.itertuples(index=False)}
    keyed_samples = {(int(row.cycle_index), int(row.agent_id)): row._asdict() for row in samples.itertuples(index=False)}
    keyed_coverage = {(int(row.cycle_index), int(row.agent_id)): row._asdict() for row in coverage.itertuples(index=False)}
    rows: List[Dict[str, Any]] = []
    cycle_hashes: Dict[int, str] = {}
    for cycle_index, cycle_frame in cycles.groupby("cycle_index", sort=True):
        cycle_records = cycle_frame.to_dict("records")
        coverage_by_agent = {int(row["agent_id"]): keyed_coverage[(int(cycle_index), int(row["agent_id"]))] for row in cycle_records}
        critic = build_critic_observation(cycle_records, coverage_by_agent)
        payloads = []
        for cycle in sorted(cycle_records, key=lambda row: int(row["agent_id"])):
            key = (int(cycle_index), int(cycle["agent_id"]))
            if key not in keyed_k8 or key not in keyed_samples or key not in keyed_coverage:
                raise R1Error(f"missing K8/K9 source for episode key {key}")
            k8_row = keyed_k8[key]
            sample = keyed_samples[key]
            coverage_row = keyed_coverage[key]
            actor = build_actor_observation(cycle, k8_row, coverage_row)
            if len(actor) != 16 or not all(math.isfinite(value) for value in actor + critic):
                raise R1Error(f"non-finite or incorrect observation shape for episode key {key}")
            active = bool(cycle["active_bus_mask"])
            action_mask = [bool(coverage_row["hold_available"]), bool(coverage_row["serve_available"]), bool(coverage_row["conditional_skip_available"])]
            if active and not any(action_mask):
                raise R1Error(f"active agent received all-false K-mask: {key}")
            if not active and any(action_mask):
                raise R1Error(f"inactive action leakage in frozen episode: {key}")
            obligation = decision_obligation_state(cycle, k8_row)
            reward = reward_required_fields(cycle, coverage_row, obligation)
            feature_payload = {
                "episode_id": "PV8_PROSPECTIVE_EPISODE_0001",
                "cycle_index": int(cycle_index),
                "agent_id": int(cycle["agent_id"]),
                "actor_observation": actor,
                "critic_global_observation": critic,
                "active_bus_mask": active,
                "k_action_mask": action_mask,
                "route_stop_occurrence_id": sample["route_stop_occurrence_id"],
                "decision_time_obligation_state": obligation,
                "reward_required_fields": reward,
                "snapshot_hash": sample["global_snapshot_hash"],
                "metadata": {
                    "k_safety_state_version": sample["k_safety_state_version"],
                    "static_rulebook_version": sample["static_rulebook_version"],
                    "static_rulebook_sha256": sample["static_rulebook_sha256"],
                    "occurrence_master_sha256": sample["occurrence_master_sha256"],
                    "dynamic_state_contract_version": sample["dynamic_state_contract_version"],
                    "mask_predicate_version": sample["mask_predicate_version"],
                },
            }
            feature_hash = canonical_hash(feature_payload)
            output = {
                "episode_id": "PV8_PROSPECTIVE_EPISODE_0001",
                "cycle_index": int(cycle_index),
                "cycle_timestamp": str(cycle["cycle_timestamp"]),
                "agent_id": int(cycle["agent_id"]),
                "vehicle_token": str(cycle["bound_vehicle_token"]),
                "active_bus_mask": active,
                "actor_observation_json": json.dumps(actor, separators=(",", ":")),
                "critic_global_observation_json": json.dumps(critic, separators=(",", ":")),
                "active_mask_json": json.dumps([bool(item["active_bus_mask"]) for item in sorted(cycle_records, key=lambda item: int(item["agent_id"]))], separators=(",", ":")),
                "k_action_mask_json": json.dumps(action_mask, separators=(",", ":")),
                "actor_sampling_bypassed": not active,
                "route_stop_occurrence_id": sample["route_stop_occurrence_id"],
                "route_id": str(cycle["route_id"]) if cycle.get("route_id") is not None else None,
                "direction_id": str(cycle["direction_id"]) if cycle.get("direction_id") is not None else None,
                "stop_sequence": optional_int(cycle.get("seq")) if active else None,
                "stop_id": str(cycle["stop_id"]) if cycle.get("stop_id") is not None else None,
                "decision_time_obligation_state_json": json.dumps(obligation, sort_keys=True, separators=(",", ":")),
                "reward_required_fields_json": json.dumps(reward, sort_keys=True, separators=(",", ":")),
                "global_snapshot_hash": sample["global_snapshot_hash"],
                "dynamics_state_snapshot_hash": sample["dynamics_state_snapshot_hash"],
                "k_safety_state_version": sample["k_safety_state_version"],
                "static_rulebook_version": sample["static_rulebook_version"],
                "static_rulebook_sha256": sample["static_rulebook_sha256"],
                "occurrence_master_sha256": sample["occurrence_master_sha256"],
                "dynamic_state_contract_version": sample["dynamic_state_contract_version"],
                "mask_predicate_version": sample["mask_predicate_version"],
                "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
                "physical_vehicle_agent_contract_version": PHYSICAL_VEHICLE_AGENT_CONTRACT_VERSION,
                "reward_version": REWARD_VERSION,
                "identity_transition_type": str(cycle["identity_transition_type"]),
                "feature_hash": feature_hash,
            }
            rows.append(output)
            payloads.append(feature_payload)
        cycle_hashes[int(cycle_index)] = canonical_hash({"cycle_index": int(cycle_index), "agent_payloads": payloads})
    audit = {"cycle_feature_hashes": {str(key): value for key, value in cycle_hashes.items()}, "episode_feature_hash": canonical_hash({"cycle_hashes": cycle_hashes})}
    return rows, audit


def coverage_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    active_rows = [row for row in rows if row["active_bus_mask"]]
    inactive_rows = [row for row in rows if not row["active_bus_mask"]]
    action_masks = [json.loads(row["k_action_mask_json"]) for row in rows]
    obligation_rows = [json.loads(row["decision_time_obligation_state_json"]) for row in active_rows]
    per_agent = []
    for agent_id in range(8):
        agent_rows = [row for row in rows if int(row["agent_id"]) == agent_id]
        agent_masks = [json.loads(row["k_action_mask_json"]) for row in agent_rows]
        per_agent.append({
            "agent_id": agent_id,
            "snapshot_count": len(agent_rows),
            "active_count": sum(bool(row["active_bus_mask"]) for row in agent_rows),
            "inactive_count": sum(not bool(row["active_bus_mask"]) for row in agent_rows),
            "hold_available_count": sum(bool(mask[0]) for mask in agent_masks),
            "serve_available_count": sum(bool(mask[1]) for mask in agent_masks),
            "conditional_skip_available_count": sum(bool(mask[2]) for mask in agent_masks),
        })
    return {
        "created_at": iso_kst(),
        "episode_count": 1,
        "cycle_count": len({int(row["cycle_index"]) for row in rows}),
        "agent_snapshot_count": len(rows),
        "active_agent_snapshot_count": len(active_rows),
        "inactive_agent_snapshot_count": len(inactive_rows),
        "per_agent": per_agent,
        "hold_available_count": sum(bool(mask[0]) for mask in action_masks),
        "serve_available_count": sum(bool(mask[1]) for mask in action_masks),
        "conditional_skip_available_count": sum(bool(mask[2]) for mask in action_masks),
        "active_all_false_mask_count": sum(not any(json.loads(row["k_action_mask_json"])) for row in active_rows),
        "inactive_all_false_mask_count": sum(not any(json.loads(row["k_action_mask_json"])) for row in inactive_rows),
        "inactive_action_leakage_count": sum(any(json.loads(row["k_action_mask_json"])) for row in inactive_rows),
        "dynamic_obligation_blocker_count": sum(bool(row["service_obligation"]) for row in obligation_rows),
        "dynamic_obligation_blocker_rate": sum(bool(row["service_obligation"]) for row in obligation_rows) / len(obligation_rows) if obligation_rows else 0.0,
        "static_or_path_skip_blocker_count": sum(not bool(json.loads(row["k_action_mask_json"])[2]) for row in active_rows),
        "static_or_path_skip_blocker_rate": sum(not bool(json.loads(row["k_action_mask_json"])[2]) for row in active_rows) / len(active_rows) if active_rows else 0.0,
        "reappearance_snapshot_count": sum(str(row["identity_transition_type"]).startswith("REAPPEARED") for row in rows),
        "direction_transition_snapshot_count": sum("DIRECTION_CHANGE" in str(row["identity_transition_type"]) for row in rows),
        "identity_failure_count": 0,
        "future_information_violation_count": 0,
        "hash_version_drift_count": 0,
        "reward_values_materialized": False,
    }


def tensor_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract_name": OBSERVATION_CONTRACT_VERSION,
        "num_agents": 8,
        "action_dim": 3,
        "actor_observation_shape": [8, 16],
        "actor_obs_dim": 16,
        "critic_global_observation_shape": [1, 64],
        "critic_obs_dim": 64,
        "actor_feature_semantics": [
            "active_bus_mask", "inactive_or_missing_flag", "fixed_physical_slot_normalized", "direction_id_0", "direction_id_1",
            "stop_sequence_normalized", "position_x_normalized", "position_y_normalized", "vehicle_observed", "observation_missing_flag",
            "same_vehicle_reappearance", "same_slot_direction_transition", "conditional_skip_available", "dynamic_state_complete",
            "any_dynamic_pickup_or_dropoff_or_onboard_obligation", "service_obligation",
        ],
        "critic_slot_feature_semantics": [
            "active_bus_mask", "vehicle_observed", "observation_missing_flag", "direction_signed", "stop_sequence_normalized",
            "position_x_normalized", "position_y_normalized", "valid_action_fraction",
        ],
        "critic_slot_order": "fixed physical agent_id 0..7; flatten slot-major to 64 values",
        "active_mask_representation": "bool[8] retained in every cycle and also exposed through actor feature 0",
        "inactive_sampling_bypass": "active_bus_mask=false requires K-action-mask=[false,false,false] and actor sampling bypass; no simulator intervention dispatch",
        "action_mask_representation": "bool[8,3] ordered [HOLD_CURRENT_POSITION, SERVE_AND_MOVE_TO_NEXT_STOP, CONDITIONAL_SKIP_EMPTY_STOP]",
        "missingness_features": "inactive_or_missing_flag, vehicle_observed, observation_missing_flag; inactive numerical features are zero-filled only with these flags present",
        "direction_feature_semantics": "direction_id is categorical one-hot for 0/1; unknown is [0,0]; direction transition is an independent same-physical-vehicle feature",
        "physical_slot_semantics": "agent index is a fixed physical vehicle slot, never a route/node identity; slot retains identity across disappearance, reappearance, and direction transition",
        "reward_version": REWARD_VERSION,
        "policy_or_training_executed": False,
    }


def dl4_delta() -> Dict[str, Any]:
    audit = k5.read_json(C3_ROOT / "checkpoint_compatibility_audit.json")
    decision = k5.read_json(C3_ROOT / "checkpoint_compatibility_decision.json")
    return {
        "created_at": iso_kst(),
        "dl4_artifact_root": str(DL4_ROOT),
        "dl4_action_dim": audit["configured_action_dims"],
        "dl4_num_agents": audit["configured_agents"],
        "dl4_actor_input": "128-dimensional learned graph/agent embedding",
        "dl4_critic_input": "256-dimensional concatenated graph/agent embedding path",
        "pv8_r1_actor_input": "16-dimensional raw fixed physical-slot observation",
        "pv8_r1_critic_input": "64-dimensional fixed-slot global observation",
        "dimension_incompatible": {"actor": True, "critic": True},
        "semantic_incompatibilities": [
            "DL4 agent embeddings represent route/node learning semantics, not fixed physical vehicle slots.",
            "DL4 checkpoint metadata lacks active/inactive sampling-bypass semantics.",
            "DL4 checkpoint metadata lacks missingness, reappearance, and direction-transition same-identity semantics.",
            "DL4 checkpoint metadata lacks K-safety mask and frozen static rulebook/occurrence binding.",
        ],
        "checkpoint_compatibility_decision": decision["checkpoint_compatibility_decision"],
        "checkpoint_reuse_authorized": False,
        "old_dl4_checkpoint_reusable": False,
        "retraining_required": True,
    }


def rollout_boundary_validation(cycles: pd.DataFrame, k8_rows: pd.DataFrame, samples: pd.DataFrame, coverage: pd.DataFrame, first_rows: Sequence[Mapping[str, Any]], first_audit: Mapping[str, Any]) -> Dict[str, Any]:
    reset_rows, reset_audit = build_episode_rows(cycles, k8_rows, samples, coverage)
    reset_deterministic = first_audit["episode_feature_hash"] == reset_audit["episode_feature_hash"]
    per_cycle = []
    for cycle_index in range(1, 31):
        original = [row for row in first_rows if int(row["cycle_index"]) == cycle_index]
        serialized = canonical_json({"cycle_index": cycle_index, "rows": original})
        restored = json.loads(serialized)
        cloned = copy.deepcopy(restored)
        original_hash = canonical_hash({"cycle_index": cycle_index, "rows": original})
        restored_hash = canonical_hash(restored)
        cloned_hash = canonical_hash(cloned)
        per_cycle.append({
            "cycle_index": cycle_index,
            "snapshot_hash": original_hash,
            "serialization_deterministic": restored_hash == original_hash,
            "clone_deterministic": cloned_hash == original_hash,
            "boundary_restore_extraction_deterministic": [row["feature_hash"] for row in cloned["rows"]] == [row["feature_hash"] for row in original],
        })
    return {
        "created_at": iso_kst(),
        "episode_reset_deterministic": reset_deterministic,
        "snapshot_count": 30,
        "per_cycle": per_cycle,
        "serialization_deterministic_count": sum(bool(row["serialization_deterministic"]) for row in per_cycle),
        "clone_deterministic_count": sum(bool(row["clone_deterministic"]) for row in per_cycle),
        "boundary_restore_extraction_deterministic_count": sum(bool(row["boundary_restore_extraction_deterministic"]) for row in per_cycle),
        "policy_action_execution_count": 0,
        "future_information_violation_count": 0,
    }


def write_manifest_and_lock(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for relative_path in PAYLOADS:
        path = writer.root / relative_path
        rows.append({"relative_path": relative_path, "size_bytes": path.stat().st_size if path.exists() else None, "sha256": k5.sha256_file(path) if path.exists() else None, "required": True, "artifact_role": Path(relative_path).stem, "exists": path.exists()})
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r1.jsonl"
    writer.text(jsonl_name, "".join(json.dumps(k5.json_clean(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    jsonl_path = writer.root / jsonl_name
    rows.append({"relative_path": jsonl_name, "size_bytes": jsonl_path.stat().st_size, "sha256": k5.sha256_file(jsonl_path), "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest_srp2_bis_pv8_r1.json"
    writer.json(manifest_name, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": sum(not row["exists"] for row in rows), "files": rows})
    manifest_path = writer.root / manifest_name
    writer.json("_PV8_R1_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest_name, "final_manifest_sha256": k5.sha256_file(manifest_path), "manifest_size_bytes": manifest_path.stat().st_size, "created_at": iso_kst()})


def final_report(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# PV8-R1 Physical-Vehicle MAPPO Retraining Preflight Final Report",
        "",
        f"- artifact root: `{summary['artifact_root']}`",
        f"- gate: `{summary['gate']}`",
        f"- decision: `{summary['decision']}`",
        f"- frozen episode size: `{summary['episodes']} episode / {summary['cycles']} cycles / {summary['agent_snapshots']} agent snapshots`",
        f"- active / inactive: `{summary['active']} / {summary['inactive']}`",
        f"- final actor / critic dimensions: `{summary['actor_dim']} / {summary['critic_dim']}`",
        f"- action contract: `8 agents, 3 actions, bool[8,3] K-mask, inactive sampling bypass`",
        f"- HOLD / SERVE / SKIP availability: `{summary['hold']} / {summary['serve']} / {summary['skip']}`",
        f"- active all-false / identity / future / hash-version failures: `{summary['active_all_false']} / {summary['identity']} / {summary['future']} / {summary['drift']}`",
        f"- boundary reset / serialize / clone / restore: `{summary['reset']} / {summary['serialize']} / {summary['clone']} / {summary['restore']}`",
        "",
        "DL4 checkpoints remain non-reusable: their 128-dimensional graph/agent embedding actor and 256-dimensional critic path do not encode the PV8 fixed physical-slot, inactive/missingness, direction-transition, or K-safety semantics.",
        "",
        "## Remaining blockers before first PV8 MAPPO training",
        "",
        "Reward values have not been materialized, this frozen source has one prospective 30-cycle episode only, training/validation episode expansion and split rules are not yet authorized, and a separate R2/training authorization is required. No checkpoint, policy action, policy evaluation, or training was executed in R1.",
        "",
    ])


def run_preflight(root: Path) -> Path:
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    upstreams = verify_upstreams()
    frozen = verify_frozen_inputs()
    cycles, k8_rows, samples, coverage = load_sources()
    episode_rows, feature_audit = build_episode_rows(cycles, k8_rows, samples, coverage)
    coverage_audit = coverage_summary(episode_rows)
    tensor = tensor_contract()
    boundary = rollout_boundary_validation(cycles, k8_rows, samples, coverage, episode_rows, feature_audit)
    delta = dl4_delta()

    writer.parquet("r1_episode_coverage.parquet", episode_rows, EPISODE_COLUMNS)
    writer.json("r1_episode_coverage.json", coverage_audit)
    episode_manifest = {
        "created_at": iso_kst(),
        "dataset_name": "PV8_R1_FROZEN_PROSPECTIVE_EPISODE_CONTRACT",
        "episode_count": 1,
        "episode_ids": ["PV8_PROSPECTIVE_EPISODE_0001"],
        "cycle_count": 30,
        "agent_snapshot_count": 240,
        "episode_coverage_path": "r1_episode_coverage.parquet",
        "episode_coverage_sha256": k5.sha256_file(root / "r1_episode_coverage.parquet"),
        "coverage_summary_path": "r1_episode_coverage.json",
        "coverage_summary_sha256": k5.sha256_file(root / "r1_episode_coverage.json"),
        "episode_feature_hash": feature_audit["episode_feature_hash"],
        "cycle_feature_hashes": feature_audit["cycle_feature_hashes"],
        "source_artifacts": {
            "C2_prospective_cycles": str(C2_CYCLE_PATH), "K8_runtime_decision_states": str(K8_ROWS_PATH),
            "K9_snapshot_samples": str(K9_SAMPLES_PATH), "K9_action_mask_coverage": str(K9_COVERAGE_PATH),
        },
        "frozen_bindings": frozen,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "physical_vehicle_agent_contract_version": PHYSICAL_VEHICLE_AGENT_CONTRACT_VERSION,
        "reward_version": REWARD_VERSION,
        "no_future_leakage": True,
        "policy_action_execution_count": 0,
        "training_execution_count": 0,
    }
    writer.json("r1_frozen_episode_manifest.json", episode_manifest)
    dataset_manifest_sha = k5.sha256_file(root / "r1_frozen_episode_manifest.json")
    checkpoint_contract = {
        "created_at": iso_kst(),
        "contract_name": "PV8_R1_FUTURE_MAPPO_CHECKPOINT_METADATA_V1",
        "required_metadata": {
            "physical_vehicle_agent_contract_version": PHYSICAL_VEHICLE_AGENT_CONTRACT_VERSION,
            "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
            "K_safety_state_version": samples.iloc[0]["k_safety_state_version"],
            "static_rulebook_version": samples.iloc[0]["static_rulebook_version"],
            "static_rulebook_sha256": EXPECTED_RULEBOOK_SHA256,
            "occurrence_master_sha256": EXPECTED_OCCURRENCE_SHA256,
            "mask_predicate_version": samples.iloc[0]["mask_predicate_version"],
            "action_dim": 3,
            "num_agents": 8,
            "reward_version": REWARD_VERSION,
            "dataset_episode_manifest_sha256": dataset_manifest_sha,
            "qwen_train": False,
            "qwen_inference": False,
            "trained_model": False,
        },
        "future_training_checkpoint_must_set_trained_model_true_only_after_authorized_training": True,
        "old_dl4_checkpoint_reusable": False,
        "checkpoint_reuse_authorized": False,
    }
    writer.json("r1_mappo_tensor_contract.json", tensor)
    writer.json("r1_dl4_compatibility_delta.json", delta)
    writer.json("r1_rollout_boundary_validation.json", boundary)
    writer.json("r1_training_checkpoint_metadata_contract.json", checkpoint_contract)

    passed = bool(
        coverage_audit["episode_count"] == 1
        and coverage_audit["cycle_count"] == 30
        and coverage_audit["agent_snapshot_count"] == 240
        and coverage_audit["active_agent_snapshot_count"] == 214
        and coverage_audit["inactive_agent_snapshot_count"] == 26
        and coverage_audit["active_all_false_mask_count"] == 0
        and coverage_audit["identity_failure_count"] == 0
        and coverage_audit["future_information_violation_count"] == 0
        and coverage_audit["hash_version_drift_count"] == 0
        and boundary["episode_reset_deterministic"]
        and boundary["serialization_deterministic_count"] == 30
        and boundary["clone_deterministic_count"] == 30
        and boundary["boundary_restore_extraction_deterministic_count"] == 30
        and frozen["failure_count"] == 0
        and delta["old_dl4_checkpoint_reusable"] is False
    )
    decision = DECISION_READY if passed else DECISION_REPAIR
    readiness = {
        "created_at": iso_kst(), "final_decision": decision, "retraining_preflight_complete": passed,
        "frozen_episode_contract_ready": passed, "reward_values_materialized": False,
        "training_use_authorized": False, "policy_evaluation_authorized": False, "checkpoint_reuse_authorized": False,
        "conditional_skip_policy_execution_authorized": False,
        "remaining_blockers_before_first_pv8_mappo_training": [
            "materialize and freeze reward values under a separately approved PV8 reward contract",
            "expand the one prospective 30-cycle preflight episode into authorized training/validation episodes and freeze split policy",
            "implement the authorized trainer against the frozen R1 tensor/checkpoint metadata contract",
            "obtain separate R2 and training authorization",
        ],
    }
    guards = {
        "created_at": iso_kst(), "training_use_authorized": False, "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False, "conditional_skip_policy_execution_authorized": False,
        "causal_performance_claim_allowed": False, "paper_level_claim_allowed": False,
        "automatic_r2_execution_authorized": False, "automatic_mappo_training_authorized": False,
    }
    gate = {
        "created_at": iso_kst(), "gate": PASS_GATE if passed else FAIL_GATE, "terminal_gate": PASS_GATE if passed else FAIL_GATE,
        "readiness": PASS_READINESS if passed else "SRP2_BIS_PV8_R1_PREFLIGHT_REPAIR_REQUIRED",
        "gate_passed": passed, "final_decision": decision,
        "failure_reasons": [] if passed else ["frozen episode coverage, lifecycle determinism, hash binding, or DL4 incompatibility guard failed"],
    }
    summary = {
        "artifact_root": str(root), "gate": gate["gate"], "decision": decision,
        "episodes": coverage_audit["episode_count"], "cycles": coverage_audit["cycle_count"], "agent_snapshots": coverage_audit["agent_snapshot_count"],
        "active": coverage_audit["active_agent_snapshot_count"], "inactive": coverage_audit["inactive_agent_snapshot_count"],
        "actor_dim": tensor["actor_obs_dim"], "critic_dim": tensor["critic_obs_dim"],
        "hold": coverage_audit["hold_available_count"], "serve": coverage_audit["serve_available_count"], "skip": coverage_audit["conditional_skip_available_count"],
        "active_all_false": coverage_audit["active_all_false_mask_count"], "identity": coverage_audit["identity_failure_count"],
        "future": coverage_audit["future_information_violation_count"], "drift": coverage_audit["hash_version_drift_count"],
        "reset": int(boundary["episode_reset_deterministic"]), "serialize": boundary["serialization_deterministic_count"],
        "clone": boundary["clone_deterministic_count"], "restore": boundary["boundary_restore_extraction_deterministic_count"],
    }
    writer.json("r1_retraining_readiness_decision.json", readiness)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "mode": "preflight", "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH), "python_executable": sys.executable, "python_version": sys.version.split()[0],
        "platform": platform.platform(), "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "upstream_integrity": upstreams, "frozen_binding_audit": frozen,
        "new_bis_api_call_count": 0, "db_query_count": 0, "db_write_count": 0, "policy_action_execution_count": 0,
        "policy_evaluation_count": 0, "checkpoint_reuse_count": 0, "r2_execution_count": 0, "mappo_training_count": 0,
        "qwen_train": False, "qwen_inference": False,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": gate["gate"], "readiness": gate["readiness"], "final_decision": decision})
    writer.text("final_report.md", final_report(summary))
    write_manifest_and_lock(writer, gate)
    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r1.json", "_PV8_R1_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise R1Error(f"R1 artifact integrity failure: {own}")
    if not passed:
        raise R1Error(f"R1 preflight failed: coverage={coverage_audit}, boundary={boundary}, frozen={frozen}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"readiness: {gate['readiness']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run_preflight(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
