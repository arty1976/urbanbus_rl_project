#!/usr/bin/env python3
"""PV8-K9 global orchestrator/snapshot lifecycle K-mask integration validation."""

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
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
from simulator.k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime, RuntimeVersionBinding
from simulator.k_mask_snapshot_lifecycle import (
    GLOBAL_K_MASK_SNAPSHOT_SCHEMA_VERSION,
    GlobalKMaskSnapshotLifecycle,
    KMaskAgentContext,
    bind_global_k_mask_to_dynamics_state,
    clone_global_k_mask_snapshot,
    deserialize_global_k_mask_snapshot,
    extract_global_k_mask_from_dynamics_state,
    restore_service_obligation_state,
    serialize_global_k_mask_snapshot,
)
from simulator.k_safety_state import ServiceObligationStateMachine
from simulator.dynamics_state_snapshot import (
    clone_dynamics_state,
    deserialize_dynamics_state,
    reset_runtime_state_from_snapshot,
    serialize_dynamics_state,
)


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation.py"
LIFECYCLE_SOURCE = TRAINING_ROOT / "simulator" / "k_mask_snapshot_lifecycle.py"
LIFECYCLE_TEST_SOURCE = TRAINING_ROOT / "simulator" / "test_pv8_k9_k_mask_snapshot_lifecycle.py"
K8_RUNTIME_SOURCE = TRAINING_ROOT / "simulator" / "k_action_mask_runtime.py"
K4_STATE_SOURCE = TRAINING_ROOT / "simulator" / "k_safety_state.py"
K4_ENGINE_SOURCE = TRAINING_ROOT / "simulator" / "suseong_service_transition_engine.py"

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
C3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c3_mappo_interface_compatibility_20260808_100118"
K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"
K6_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026"
K7_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k7_research_rule_contract_mask_dryrun_20260808_130432"
K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"

UPSTREAMS = {
    "PV8-C2": (C2_ROOT, "artifact_manifest_srp2_bis_pv8_c2.json", "_PV8_C2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"),
    "PV8-C3": (C3_ROOT, "artifact_manifest_srp2_bis_pv8_c3.json", "_PV8_C3_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_COMPLETE"),
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
    "PV8-K6": (K6_ROOT, "artifact_manifest_srp2_bis_pv8_k6.json", "_PV8_K6_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K6_STATIC_RULE_AUTHORITY_AND_OCCURRENCE_AUDIT_COMPLETE"),
    "PV8-K7": (K7_ROOT, "artifact_manifest_srp2_bis_pv8_k7.json", "_PV8_K7_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K7_RESEARCH_RULE_CONTRACT_AND_K_MASK_DRYRUN_COMPLETE"),
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
}

RULEBOOK_PATH = K7_ROOT / "k7_research_rulebook_candidate.parquet"
OCCURRENCE_PATH = K6_ROOT / "k6_route_stop_occurrence_master.parquet"
C2_MAPPING_PATH = C2_ROOT / "prospective_8vehicle_mapping_manifest.parquet"
C2_CYCLE_PATH = C2_ROOT / "prospective_agent_cycle_state.parquet"
C2_TRANSITION_PATH = C2_ROOT / "vehicle_identity_transition_events.parquet"
K8_APPROVAL_PATH = K8_ROOT / "k8_explicit_rule_approval_record.json"
K8_BINDING_PATH = K8_ROOT / "k8_snapshot_version_binding.json"

EXPECTED_RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
EXPECTED_OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED"
FAIL_GATE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATION_FAILED"
PASS_READINESS = "SRP2_BIS_PV8_K9_COMPLETE_GLOBAL_KMASK_LIFECYCLE_READY_K10_AND_RETRAINING_LOCKED"
DECISION_READY = "K_MASK_GLOBAL_LIFECYCLE_READY"
DECISION_FAILED = "K_MASK_GLOBAL_INTEGRATION_REPAIR_REQUIRED"

PAYLOADS = [
    "k9_global_snapshot_contract.json",
    "k9_snapshot_samples.parquet",
    "k9_clone_reset_replay_validation.json",
    "k9_action_mask_coverage.parquet",
    "k9_action_availability_summary.json",
    "k9_version_hash_binding_audit.json",
    "k9_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]

SAMPLE_COLUMNS = [
    "cycle_index", "cycle_timestamp", "global_snapshot_hash", "serialized_snapshot_sha256",
    "dynamics_state_snapshot_hash", "dynamics_serialized_sha256",
    "agent_id", "vehicle_token", "active_bus_mask", "route_stop_occurrence_id", "decision_ts",
    "k_action_mask_json", "valid_action_count", "actor_sampling_bypassed", "identity_transition_type",
    "occurrence_lookup_integrity", "identity_failure", "service_obligation_state_hash",
    "decision_obligation_snapshot_hash", "k_safety_state_version", "static_rulebook_version",
    "static_rulebook_sha256", "occurrence_master_sha256", "dynamic_state_contract_version",
    "mask_predicate_version", "experiment_version", "policy_execution_count",
]

COVERAGE_COLUMNS = [
    "cycle_index", "agent_id", "active_bus_mask", "hold_available", "serve_available",
    "conditional_skip_available", "valid_action_count", "all_false_mask", "actor_sampling_bypassed",
    "route_stop_occurrence_id", "identity_transition_type",
]


class K9Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def verify_upstreams() -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed_gate != expected_gate or not k5.manifest_ok(checks):
            raise K9Error(f"{label} upstream integrity failure: gate={observed_gate}, checks={checks}")
        records[label] = {
            "artifact_root": str(root),
            "gate": observed_gate,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return records


def verify_frozen_bindings() -> Tuple[Dict[str, Any], RuntimeVersionBinding, Dict[str, Any]]:
    approval = k5.read_json(K8_APPROVAL_PATH)
    binding = k5.read_json(K8_BINDING_PATH)
    rulebook_sha = k5.sha256_file(RULEBOOK_PATH)
    occurrence_sha = k5.sha256_file(OCCURRENCE_PATH)
    k8_runtime_sha = k5.sha256_file(K8_RUNTIME_SOURCE)
    mapping_sha = k5.sha256_file(C2_MAPPING_PATH)
    approval_sha = k5.sha256_file(K8_APPROVAL_PATH)
    checks = {
        "rulebook_sha256": rulebook_sha,
        "rulebook_hash_matches_required": rulebook_sha == EXPECTED_RULEBOOK_SHA256,
        "occurrence_master_sha256": occurrence_sha,
        "occurrence_hash_matches_required": occurrence_sha == EXPECTED_OCCURRENCE_SHA256,
        "k8_binding_rulebook_hash_matches": binding.get("static_rulebook_sha256") == EXPECTED_RULEBOOK_SHA256,
        "k8_binding_occurrence_hash_matches": binding.get("occurrence_master_sha256") == EXPECTED_OCCURRENCE_SHA256,
        "k8_runtime_source_sha256": k8_runtime_sha,
        "k8_runtime_source_matches_binding": k8_runtime_sha == binding.get("runtime_source_sha256"),
        "c2_fixed_vehicle_mapping_sha256": mapping_sha,
        "c2_fixed_vehicle_mapping_matches_binding": mapping_sha == binding.get("fixed_vehicle_mapping_sha256"),
        "k8_approval_record_sha256": approval_sha,
        "k8_approval_record_matches_binding": approval_sha == binding.get("research_rule_approval_record_sha256"),
        "research_rule_approved": approval.get("research_rule_approved") is True and binding.get("research_rule_approved") is True,
        "K_action_mask_available": binding.get("K_action_mask_available") is True,
        "real_network_rule_claim_allowed": approval.get("real_network_rule_claim_allowed") is True,
    }
    checks["binding_failure_count"] = sum(
        not bool(value)
        for key, value in checks.items()
        if key.endswith("_matches_required") or key.endswith("_matches") or key in {"research_rule_approved", "K_action_mask_available"}
    ) + int(bool(checks["real_network_rule_claim_allowed"]))
    if checks["binding_failure_count"]:
        raise K9Error(f"K8 frozen binding failure: {checks}")
    version = RuntimeVersionBinding(
        k_safety_state_version=str(binding["k_safety_state_version"]),
        static_rulebook_version=str(binding["static_rulebook_version"]),
        static_rulebook_sha256=str(binding["static_rulebook_sha256"]),
        occurrence_master_sha256=str(binding["occurrence_master_sha256"]),
        dynamic_state_contract_version=str(binding["dynamic_state_contract_version"]),
        mask_predicate_version=str(binding["mask_predicate_version"]),
        experiment_version=str(binding["experiment_version"]),
    )
    return checks, version, approval


def verify_k4_source_hashes() -> Dict[str, Any]:
    contract = k5.read_json(K4_ROOT / "k4_state_machine_contract.json")
    expected = {str(row["path"]): str(row["sha256"]) for row in contract["source_files"]}
    records = []
    for path in (K4_STATE_SOURCE, K4_ENGINE_SOURCE):
        current = k5.sha256_file(path)
        records.append({"path": str(path), "expected_sha256": expected.get(str(path)), "current_sha256": current, "matches_k4": current == expected.get(str(path))})
    return {"records": records, "source_drift_count": sum(not row["matches_k4"] for row in records)}


def fixed_bindings() -> Tuple[Dict[int, str], pd.DataFrame]:
    frame = pd.read_parquet(C2_MAPPING_PATH).sort_values("agent_id").reset_index(drop=True)
    if len(frame) != 8 or frame["agent_id"].nunique() != 8 or frame["physical_vehicle_token"].nunique() != 8:
        raise K9Error("C2 fixed mapping must contain exactly eight unique agent slots and vehicle tokens")
    return {int(row.agent_id): str(row.physical_vehicle_token) for row in frame.itertuples()}, frame


def machine_for_rule(agent_id: int, token: str, rule: Optional[Mapping[str, Any]]) -> ServiceObligationStateMachine:
    state = ServiceObligationStateMachine()
    state.register_vehicle(agent_id, token)
    state.register_stop("__K8_PREVIOUS__")
    if rule is not None:
        state.register_stop(str(rule["stop_id"]))
        post = rule.get("post_skip_target_stop_id")
        if post is not None and str(post).lower() != "nan":
            state.register_stop(str(post))
    return state


def build_contexts(cycle_rows: Sequence[Mapping[str, Any]], by_exact: Mapping[Tuple[str, str, int, str], Mapping[str, Any]]) -> List[KMaskAgentContext]:
    contexts = []
    for cycle in sorted(cycle_rows, key=lambda row: int(row["agent_id"])):
        agent_id = int(cycle["agent_id"])
        token = str(cycle["bound_vehicle_token"])
        active = bool(cycle["active_bus_mask"])
        rule = None
        if active:
            exact = (str(cycle["route_id"]), str(cycle["direction_id"]), int(cycle["seq"]), str(cycle["stop_id"]))
            rule = by_exact.get(exact)
            if rule is None:
                raise K9Error(f"active C2 occurrence missing from K7 rulebook: {exact}")
        contexts.append(
            KMaskAgentContext(
                agent_id=agent_id,
                vehicle_token=token,
                active_bus_mask=active,
                decision_ts=int(cycle["cycle_index"]),
                obligation_state_machine=machine_for_rule(agent_id, token, rule),
                route_id=str(rule["route_id"]) if rule is not None else None,
                direction_id=str(rule["direction_id"]) if rule is not None else None,
                stop_sequence=int(rule["stop_sequence"]) if rule is not None else None,
                stop_id=str(rule["stop_id"]) if rule is not None else None,
                route_stop_occurrence_id=str(rule["route_stop_occurrence_id"]) if rule is not None else None,
                identity_transition_type=str(cycle["identity_transition_type"]) if cycle.get("identity_transition_type") is not None else None,
            )
        )
    return contexts


def runtime_adapter(version: RuntimeVersionBinding, approval: Mapping[str, Any]) -> Tuple[FixedVehicleOccurrenceMaskRuntime, List[Dict[str, Any]], Dict[int, str]]:
    frame = pd.read_parquet(RULEBOOK_PATH)
    rules = frame.to_dict("records")
    bindings, _ = fixed_bindings()
    runtime = FixedVehicleOccurrenceMaskRuntime(
        rule_rows=rules,
        fixed_vehicle_bindings=bindings,
        approval_record=approval,
        version_binding=version,
        actual_rulebook_sha256=k5.sha256_file(RULEBOOK_PATH),
        actual_occurrence_master_sha256=k5.sha256_file(OCCURRENCE_PATH),
    )
    return runtime, rules, bindings


def run_lifecycle(
    lifecycle: GlobalKMaskSnapshotLifecycle,
    rules: Sequence[Mapping[str, Any]],
    fixed_vehicle_tokens: Mapping[int, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    by_exact = {
        (str(row["route_id"]), str(row["direction_id"]), int(row["stop_sequence"]), str(row["stop_id"])): row
        for row in rules
    }
    cycles = pd.read_parquet(C2_CYCLE_PATH).sort_values(["cycle_index", "agent_id"])
    transitions = pd.read_parquet(C2_TRANSITION_PATH)
    cycle_validation = []
    sample_rows: List[Dict[str, Any]] = []
    coverage_rows: List[Dict[str, Any]] = []
    for cycle_index, cycle_frame in cycles.groupby("cycle_index", sort=True):
        cycle_rows = cycle_frame.to_dict("records")
        timestamp = str(cycle_rows[0]["cycle_timestamp"])
        baseline = lifecycle.capture_cycle(
            cycle_index=int(cycle_index),
            cycle_timestamp=timestamp,
            contexts=build_contexts(cycle_rows, by_exact),
        )
        reset_repeat = lifecycle.capture_cycle(
            cycle_index=int(cycle_index),
            cycle_timestamp=timestamp,
            contexts=build_contexts(cycle_rows, by_exact),
        )
        serialized = serialize_global_k_mask_snapshot(baseline)
        restored = deserialize_global_k_mask_snapshot(serialized)
        cloned = clone_global_k_mask_snapshot(restored)
        replayed = lifecycle.recompute(restored)
        clone_recomputed = lifecycle.recompute(cloned)
        serialized_sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        dynamics_snapshot = bind_global_k_mask_to_dynamics_state(baseline)
        dynamics_serialized = serialize_dynamics_state(dynamics_snapshot)
        dynamics_restored = deserialize_dynamics_state(dynamics_serialized)
        dynamics_cloned = clone_dynamics_state(dynamics_restored)
        dynamics_reset = reset_runtime_state_from_snapshot(dynamics_cloned)
        dynamics_extracted = extract_global_k_mask_from_dynamics_state(dynamics_cloned)
        dynamics_recomputed = lifecycle.recompute(dynamics_extracted)
        dynamics_serialized_sha = hashlib.sha256(dynamics_serialized.encode("utf-8")).hexdigest()
        checks = {
            "cycle_index": int(cycle_index),
            "global_snapshot_hash": baseline.snapshot_hash,
            "reset_deterministic": reset_repeat.snapshot_hash == baseline.snapshot_hash,
            "serialization_roundtrip_deterministic": restored.snapshot_hash == baseline.snapshot_hash,
            "clone_deterministic": cloned.snapshot_hash == baseline.snapshot_hash,
            "restore_replay_deterministic": replayed.snapshot_hash == baseline.snapshot_hash,
            "mask_recomputation_deterministic": clone_recomputed.snapshot_hash == baseline.snapshot_hash,
            "dynamics_serialization_deterministic": dynamics_restored.state_hash == dynamics_snapshot.state_hash,
            "dynamics_clone_deterministic": dynamics_cloned.state_hash == dynamics_snapshot.state_hash,
            "dynamics_reset_deterministic": bool(dynamics_reset["hash_match"]),
            "dynamics_extract_recompute_deterministic": dynamics_recomputed.snapshot_hash == baseline.snapshot_hash,
        }
        checks["passed"] = all(value for key, value in checks.items() if key.endswith("deterministic"))
        cycle_validation.append(checks)
        for row in baseline.to_payload()["agent_snapshots"]:
            action_mask = [bool(value) for value in row["k_action_mask"]]
            obligation_snapshot = row.get("decision_time_obligation_snapshot")
            obligation_hash = hashlib.sha256(json.dumps(k5.json_clean(obligation_snapshot), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest() if obligation_snapshot is not None else None
            sample_rows.append(
                {
                    "cycle_index": int(cycle_index),
                    "cycle_timestamp": timestamp,
                    "global_snapshot_hash": baseline.snapshot_hash,
                    "serialized_snapshot_sha256": serialized_sha,
                    "dynamics_state_snapshot_hash": dynamics_snapshot.state_hash,
                    "dynamics_serialized_sha256": dynamics_serialized_sha,
                    "agent_id": int(row["agent_id"]),
                    "vehicle_token": str(row["vehicle_token"]),
                    "active_bus_mask": bool(row["active_bus_mask"]),
                    "route_stop_occurrence_id": row["route_stop_occurrence_id"],
                    "decision_ts": int(row["decision_ts"]),
                    "k_action_mask_json": json.dumps(action_mask),
                    "valid_action_count": sum(action_mask),
                    "actor_sampling_bypassed": bool(row["actor_sampling_bypassed"]),
                    "identity_transition_type": row["identity_transition_type"],
                    "occurrence_lookup_integrity": bool(row["occurrence_lookup_integrity"]),
                    "identity_failure": bool(row["identity_failure"]),
                    "service_obligation_state_hash": row["service_obligation_state_hash"],
                    "decision_obligation_snapshot_hash": obligation_hash,
                    "k_safety_state_version": row["k_safety_state_version"],
                    "static_rulebook_version": row["static_rulebook_version"],
                    "static_rulebook_sha256": row["static_rulebook_sha256"],
                    "occurrence_master_sha256": row["occurrence_master_sha256"],
                    "dynamic_state_contract_version": row["dynamic_state_contract_version"],
                    "mask_predicate_version": row["mask_predicate_version"],
                    "experiment_version": row["experiment_version"],
                    "policy_execution_count": int(row["policy_execution_count"]),
                }
            )
            coverage_rows.append(
                {
                    "cycle_index": int(cycle_index),
                    "agent_id": int(row["agent_id"]),
                    "active_bus_mask": bool(row["active_bus_mask"]),
                    "hold_available": action_mask[0],
                    "serve_available": action_mask[1],
                    "conditional_skip_available": action_mask[2],
                    "valid_action_count": sum(action_mask),
                    "all_false_mask": not any(action_mask),
                    "actor_sampling_bypassed": bool(row["actor_sampling_bypassed"]),
                    "route_stop_occurrence_id": row["route_stop_occurrence_id"],
                    "identity_transition_type": row["identity_transition_type"],
                }
            )

    first_cycle = cycles[cycles["cycle_index"] == cycles["cycle_index"].min()].to_dict("records")
    future_contexts = build_contexts(first_cycle, by_exact)
    target = next(context for context in future_contexts if context.active_bus_mask)
    rule = by_exact[(str(target.route_id), str(target.direction_id), int(target.stop_sequence), str(target.stop_id))]
    post_stop = str(rule["post_skip_target_stop_id"])
    target.obligation_state_machine.schedule_transition("passenger_waiting", 101, passenger_id="K9_FUTURE_P", pickup_stop=str(target.stop_id), dropoff_stop=post_stop)
    target.obligation_state_machine.schedule_transition("request_created", 102, request_id="K9_FUTURE_R", passenger_id="K9_FUTURE_P", service_leg_id="K9_FUTURE_L")
    target.obligation_state_machine.schedule_transition("request_assigned", 103, request_id="K9_FUTURE_R", agent_id=target.agent_id, vehicle_token=target.vehicle_token)
    future_contexts = [
        KMaskAgentContext(**{**context.__dict__, "decision_ts": 100})
        for context in future_contexts
    ]
    pre_boundary = lifecycle.capture_cycle(cycle_index=100, cycle_timestamp="K9_SYNTHETIC_PRE_BOUNDARY", contexts=future_contexts)
    pre_recomputed = lifecycle.recompute(pre_boundary)
    pre_target = pre_boundary.to_payload()["agent_snapshots"][target.agent_id]
    restored_contexts = []
    for row in pre_boundary.to_payload()["agent_snapshots"]:
        restored_contexts.append(
            KMaskAgentContext(
                agent_id=int(row["agent_id"]), vehicle_token=str(row["vehicle_token"]), active_bus_mask=bool(row["active_bus_mask"]),
                decision_ts=103, obligation_state_machine=restore_service_obligation_state(row["service_obligation_state"]),
                route_id=row["route_id"], direction_id=row["direction_id"], stop_sequence=row["stop_sequence"], stop_id=row["stop_id"],
                route_stop_occurrence_id=row["route_stop_occurrence_id"], identity_transition_type=row["identity_transition_type"],
            )
        )
    post_boundary = lifecycle.capture_cycle(cycle_index=103, cycle_timestamp="K9_SYNTHETIC_POST_BOUNDARY", contexts=restored_contexts)
    post_target = post_boundary.to_payload()["agent_snapshots"][target.agent_id]
    future_test = {
        "target_agent_id": target.agent_id,
        "pre_boundary_skip_available": bool(pre_target["conditional_skip_valid"]),
        "pre_boundary_service_obligation": bool(pre_target["decision_time_obligation_snapshot"]["service_obligation"]),
        "pre_boundary_recompute_deterministic": pre_recomputed.snapshot_hash == pre_boundary.snapshot_hash,
        "post_boundary_skip_available": bool(post_target["conditional_skip_valid"]),
        "post_boundary_service_obligation": bool(post_target["decision_time_obligation_snapshot"]["service_obligation"]),
    }
    future_test["passed"] = bool(
        future_test["pre_boundary_skip_available"]
        and not future_test["pre_boundary_service_obligation"]
        and future_test["pre_boundary_recompute_deterministic"]
        and not future_test["post_boundary_skip_available"]
        and future_test["post_boundary_service_obligation"]
    )

    token_failures = sum(str(row["vehicle_token"]) != fixed_vehicle_tokens[int(row["agent_id"])] for row in sample_rows)
    occurrence_failures = sum(bool(row["active_bus_mask"]) and not bool(row["occurrence_lookup_integrity"]) for row in sample_rows)
    identity_failures = sum(bool(row["identity_failure"]) for row in sample_rows)
    slot_mutations = int(cycles["slot_identity_mutation"].astype(bool).sum()) + token_failures
    reappearance_count = sum(str(row["identity_transition_type"]) == "REAPPEARED_SAME_IDENTITY_WITH_DIRECTION_CHANGE" for row in sample_rows)
    direction_transition_count = len(transitions)
    audit = {
        "global_snapshot_count": len(cycle_validation),
        "agent_snapshot_count": len(sample_rows),
        "active_agent_snapshot_count": sum(bool(row["active_bus_mask"]) for row in sample_rows),
        "inactive_agent_snapshot_count": sum(not bool(row["active_bus_mask"]) for row in sample_rows),
        "fixed_agent_count": len({int(row["agent_id"]) for row in sample_rows}),
        "reset_deterministic_count": sum(bool(row["reset_deterministic"]) for row in cycle_validation),
        "clone_deterministic_count": sum(bool(row["clone_deterministic"]) for row in cycle_validation),
        "restore_replay_deterministic_count": sum(bool(row["restore_replay_deterministic"]) for row in cycle_validation),
        "mask_recomputation_deterministic_count": sum(bool(row["mask_recomputation_deterministic"]) for row in cycle_validation),
        "serialization_roundtrip_deterministic_count": sum(bool(row["serialization_roundtrip_deterministic"]) for row in cycle_validation),
        "dynamics_serialization_deterministic_count": sum(bool(row["dynamics_serialization_deterministic"]) for row in cycle_validation),
        "dynamics_clone_deterministic_count": sum(bool(row["dynamics_clone_deterministic"]) for row in cycle_validation),
        "dynamics_reset_deterministic_count": sum(bool(row["dynamics_reset_deterministic"]) for row in cycle_validation),
        "dynamics_extract_recompute_deterministic_count": sum(bool(row["dynamics_extract_recompute_deterministic"]) for row in cycle_validation),
        "cycle_validation_failure_count": sum(not bool(row["passed"]) for row in cycle_validation),
        "reappearance_identity_case_count": reappearance_count,
        "direction_transition_identity_case_count": direction_transition_count,
        "slot_mutation_count": slot_mutations,
        "identity_failure_count": identity_failures,
        "occurrence_lookup_failure_count": occurrence_failures,
        "future_information_violation_count": 0 if future_test["passed"] else 1,
        "future_boundary_test": future_test,
        "per_cycle": cycle_validation,
    }
    return sample_rows, coverage_rows, audit


def availability_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    active = [row for row in rows if row["active_bus_mask"]]
    inactive = [row for row in rows if not row["active_bus_mask"]]
    per_agent = []
    for agent_id in range(8):
        agent_rows = [row for row in rows if int(row["agent_id"]) == agent_id]
        per_agent.append(
            {
                "agent_id": agent_id,
                "row_count": len(agent_rows),
                "active_count": sum(bool(row["active_bus_mask"]) for row in agent_rows),
                "inactive_count": sum(not bool(row["active_bus_mask"]) for row in agent_rows),
                "hold_available_count": sum(bool(row["hold_available"]) for row in agent_rows),
                "serve_available_count": sum(bool(row["serve_available"]) for row in agent_rows),
                "conditional_skip_available_count": sum(bool(row["conditional_skip_available"]) for row in agent_rows),
                "valid_action_count_distribution": dict(sorted(Counter(int(row["valid_action_count"]) for row in agent_rows).items())),
            }
        )
    return {
        "created_at": iso_kst(),
        "row_count": len(rows),
        "active_count": len(active),
        "inactive_count": len(inactive),
        "hold_available_count": sum(bool(row["hold_available"]) for row in rows),
        "serve_available_count": sum(bool(row["serve_available"]) for row in rows),
        "conditional_skip_available_count": sum(bool(row["conditional_skip_available"]) for row in rows),
        "active_all_false_mask_count": sum(bool(row["all_false_mask"]) for row in active),
        "inactive_all_false_mask_count": sum(bool(row["all_false_mask"]) for row in inactive),
        "inactive_action_leakage_count": sum(any((row["hold_available"], row["serve_available"], row["conditional_skip_available"])) for row in inactive),
        "inactive_actor_sampling_bypass_count": sum(bool(row["actor_sampling_bypassed"]) for row in inactive),
        "valid_action_count_distribution_active": dict(sorted(Counter(int(row["valid_action_count"]) for row in active).items())),
        "valid_action_count_distribution_inactive": dict(sorted(Counter(int(row["valid_action_count"]) for row in inactive).items())),
        "per_agent": per_agent,
    }


def run_pytest() -> Dict[str, Any]:
    tests = [
        "05_training/simulator/test_pv8_k9_k_mask_snapshot_lifecycle.py",
        "05_training/simulator/test_pv8_k8_k_action_mask_runtime.py",
        "05_training/simulator/test_pv8_k4_service_obligation_state.py",
        "05_training/simulator/test_dl6c_conditional_skip_safety.py",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TRAINING_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests], cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, check=False)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    match = re.search(r"(\d+) passed", output)
    return {
        "command": [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        "exit_code": completed.returncode,
        "passed_test_count": int(match.group(1)) if completed.returncode == 0 and match else 0,
        "summary": output[-1000:],
    }


def write_manifest_and_lock(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for relative_path in PAYLOADS:
        path = writer.root / relative_path
        rows.append({"relative_path": relative_path, "size_bytes": path.stat().st_size if path.exists() else None, "sha256": k5.sha256_file(path) if path.exists() else None, "required": True, "artifact_role": Path(relative_path).stem, "exists": path.exists()})
    jsonl_name = "artifact_manifest_srp2_bis_pv8_k9.jsonl"
    writer.text(jsonl_name, "".join(json.dumps(k5.json_clean(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    jsonl_path = writer.root / jsonl_name
    rows.append({"relative_path": jsonl_name, "size_bytes": jsonl_path.stat().st_size, "sha256": k5.sha256_file(jsonl_path), "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest_srp2_bis_pv8_k9.json"
    writer.json(manifest_name, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": sum(not row["exists"] for row in rows), "files": rows})
    manifest_path = writer.root / manifest_name
    writer.json("_PV8_K9_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest_name, "final_manifest_sha256": k5.sha256_file(manifest_path), "manifest_size_bytes": manifest_path.stat().st_size, "created_at": iso_kst()})


def final_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# PV8-K9 Global K-Mask Snapshot Lifecycle Validation Final Report",
            "",
            f"- artifact root: `{summary['artifact_root']}`",
            f"- gate: `{summary['gate']}`",
            f"- final decision: `{summary['decision']}`",
            f"- global snapshot count: `{summary['global_snapshots']}`",
            f"- agent snapshot count: `{summary['agent_snapshots']}`",
            f"- active / inactive coverage: `{summary['active']} / {summary['inactive']}`",
            f"- reset / clone / restore-replay / recompute deterministic: `{summary['reset']} / {summary['clone']} / {summary['restore']} / {summary['recompute']}` of `{summary['global_snapshots']}`",
            f"- existing DynamicsStateSnapshot serialize / clone / reset / extract-recompute deterministic: `{summary['dynamics_serialize']} / {summary['dynamics_clone']} / {summary['dynamics_reset']} / {summary['dynamics_recompute']}` of `{summary['global_snapshots']}`",
            f"- active / inactive all-false masks: `{summary['active_all_false']} / {summary['inactive_all_false']}`",
            f"- HOLD / SERVE / CONDITIONAL_SKIP availability: `{summary['hold']} / {summary['serve']} / {summary['skip']}`",
            f"- identity / occurrence / hash-version failures: `{summary['identity']} / {summary['occurrence']} / {summary['binding_failures']}`",
            f"- future-information violations: `{summary['future']}`",
            f"- regression tests: `{summary['pytest']} passed`",
            "",
            "The lifecycle uses the K8 approved research-rule contract. These masks are not observed Daegu operating rules and do not authorize a real-network rule claim.",
            "",
            "Every global cycle contains the fixed agent order 0..7. The serialized agent snapshot binds physical identity, active state, exact route-stop occurrence, decision-time obligation state, the three-action mask, and all K8 version/hash metadata. Inactive rows carry an all-false mask and an explicit actor-sampling bypass; active rows always retain at least one valid action.",
            "",
            "## Remaining locks",
            "",
            "Conditional-SKIP policy execution, training use, checkpoint reuse, policy evaluation, causal and paper claims, K10, and MAPPO retraining remain unauthorized.",
            "",
            "## Minimum MAPPO retraining-preflight scope",
            "",
            "Freeze a prospective episode dataset with the K9 snapshot/version binding, audit mask availability and obligation-block rates over full episodes, verify orchestrator clone/reset behavior at rollout boundaries, define active-mask sampling safeguards, and obtain separate authorization for checkpoint compatibility, policy evaluation, and retraining preflight.",
            "",
        ]
    )


def run_validation(root: Path) -> Path:
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    upstreams = verify_upstreams()
    binding_audit, version, approval = verify_frozen_bindings()
    k4_sources = verify_k4_source_hashes()
    if k4_sources["source_drift_count"]:
        raise K9Error(f"K4 source drift: {k4_sources}")
    runtime, rules, fixed_tokens = runtime_adapter(version, approval)
    lifecycle = GlobalKMaskSnapshotLifecycle(runtime, version)
    sample_rows, coverage_rows, lifecycle_audit = run_lifecycle(lifecycle, rules, fixed_tokens)
    availability = availability_summary(coverage_rows)
    pytest_result = run_pytest()

    binding_audit.update(
        {
            "created_at": iso_kst(),
            "k4_source_integrity": k4_sources,
            "lifecycle_source_path": str(LIFECYCLE_SOURCE),
            "lifecycle_source_sha256": k5.sha256_file(LIFECYCLE_SOURCE),
            "lifecycle_test_source_sha256": k5.sha256_file(LIFECYCLE_TEST_SOURCE),
            "snapshot_schema_version": GLOBAL_K_MASK_SNAPSHOT_SCHEMA_VERSION,
            "snapshot_metadata_drift_count": sum(
                row["static_rulebook_sha256"] != EXPECTED_RULEBOOK_SHA256
                or row["occurrence_master_sha256"] != EXPECTED_OCCURRENCE_SHA256
                or row["k_safety_state_version"] != version.k_safety_state_version
                or row["static_rulebook_version"] != version.static_rulebook_version
                or row["dynamic_state_contract_version"] != version.dynamic_state_contract_version
                or row["mask_predicate_version"] != version.mask_predicate_version
                for row in sample_rows
            ),
        }
    )
    binding_audit["total_drift_or_failure_count"] = int(binding_audit["binding_failure_count"] + binding_audit["snapshot_metadata_drift_count"] + k4_sources["source_drift_count"])

    writer.json(
        "k9_global_snapshot_contract.json",
        {
            "created_at": iso_kst(),
            "contract_name": "PV8_K9_GLOBAL_K_MASK_SNAPSHOT_LIFECYCLE_V1",
            "snapshot_schema_version": GLOBAL_K_MASK_SNAPSHOT_SCHEMA_VERSION,
            "lifecycle_order": ["reset", "state_event_reduction", "occurrence_lookup", "DecisionTimeObligationSnapshot", "K_action_mask", "snapshot_serialization", "clone", "restore_replay", "deterministic_recomputation"],
            "global_agent_order": list(range(8)),
            "global_snapshot_count_expected": 30,
            "agent_snapshot_count_expected": 240,
            "required_agent_metadata": ["agent_id", "vehicle_token", "active_bus_mask", "route_stop_occurrence_id", "decision_ts", "k_action_mask", "k_safety_state_version", "static_rulebook_version", "static_rulebook_sha256", "occurrence_master_sha256", "dynamic_state_contract_version", "mask_predicate_version"],
            "service_obligation_state_serialized": True,
            "existing_dynamics_state_snapshot_bridge": True,
            "existing_dynamics_action_mask_state_binding": "action_mask_state.global_k_mask_snapshot",
            "inactive_actor_sampling_bypass_required": True,
            "active_all_false_mask_forbidden": True,
            "policy_execution_authorized": False,
            "research_rule_claim_boundary": "CONTRACT_FIXED_RESEARCH_RULE; not observed Daegu operational rules",
        },
    )
    writer.parquet("k9_snapshot_samples.parquet", sample_rows, SAMPLE_COLUMNS)
    writer.json("k9_clone_reset_replay_validation.json", {**lifecycle_audit, "pytest_regression": pytest_result})
    writer.parquet("k9_action_mask_coverage.parquet", coverage_rows, COVERAGE_COLUMNS)
    writer.json("k9_action_availability_summary.json", availability)
    writer.json("k9_version_hash_binding_audit.json", binding_audit)

    passed = bool(
        lifecycle_audit["global_snapshot_count"] == 30
        and lifecycle_audit["agent_snapshot_count"] == 240
        and lifecycle_audit["active_agent_snapshot_count"] == 214
        and lifecycle_audit["inactive_agent_snapshot_count"] == 26
        and lifecycle_audit["fixed_agent_count"] == 8
        and lifecycle_audit["reset_deterministic_count"] == 30
        and lifecycle_audit["clone_deterministic_count"] == 30
        and lifecycle_audit["restore_replay_deterministic_count"] == 30
        and lifecycle_audit["mask_recomputation_deterministic_count"] == 30
        and lifecycle_audit["serialization_roundtrip_deterministic_count"] == 30
        and lifecycle_audit["dynamics_serialization_deterministic_count"] == 30
        and lifecycle_audit["dynamics_clone_deterministic_count"] == 30
        and lifecycle_audit["dynamics_reset_deterministic_count"] == 30
        and lifecycle_audit["dynamics_extract_recompute_deterministic_count"] == 30
        and lifecycle_audit["cycle_validation_failure_count"] == 0
        and lifecycle_audit["reappearance_identity_case_count"] == 1
        and lifecycle_audit["direction_transition_identity_case_count"] == 1
        and lifecycle_audit["slot_mutation_count"] == 0
        and lifecycle_audit["identity_failure_count"] == 0
        and lifecycle_audit["occurrence_lookup_failure_count"] == 0
        and lifecycle_audit["future_information_violation_count"] == 0
        and availability["active_all_false_mask_count"] == 0
        and availability["inactive_all_false_mask_count"] == 26
        and availability["inactive_action_leakage_count"] == 0
        and availability["inactive_actor_sampling_bypass_count"] == 26
        and binding_audit["total_drift_or_failure_count"] == 0
        and pytest_result["exit_code"] == 0
    )
    decision = DECISION_READY if passed else DECISION_FAILED
    readiness = {
        "created_at": iso_kst(),
        "final_decision": decision,
        "global_lifecycle_validation_passed": passed,
        "K_action_mask_available": passed,
        "conditional_skip_policy_enabled": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_evaluation_authorized": False,
        "minimum_mappo_retraining_preflight_scope": [
            "freeze full prospective episode snapshots with the K9 version/hash binding",
            "audit per-episode action availability, obligation blockers, and active sampling safeguards",
            "validate rollout-boundary reset/clone/restore behavior in the global orchestrator",
            "freeze candidate retraining dataset and configuration hashes",
            "obtain separate checkpoint-compatibility, policy-evaluation, and retraining-preflight authorization",
        ],
    }
    guards = {
        "created_at": iso_kst(),
        "research_rule_approved": True,
        "K_action_mask_available": passed,
        "conditional_skip_policy_enabled": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_evaluation_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "real_network_rule_claim_allowed": False,
        "automatic_k10_execution_authorized": False,
        "automatic_mappo_retraining_authorized": False,
    }
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE if passed else FAIL_GATE,
        "terminal_gate": PASS_GATE if passed else FAIL_GATE,
        "readiness": PASS_READINESS if passed else "SRP2_BIS_PV8_K9_GLOBAL_INTEGRATION_REPAIR_REQUIRED",
        "gate_passed": passed,
        "final_decision": decision,
        "failure_reasons": [] if passed else ["global K-mask snapshot lifecycle invariant or regression failed"],
    }
    summary = {
        "artifact_root": str(root), "gate": gate["gate"], "decision": decision,
        "global_snapshots": lifecycle_audit["global_snapshot_count"], "agent_snapshots": lifecycle_audit["agent_snapshot_count"],
        "active": lifecycle_audit["active_agent_snapshot_count"], "inactive": lifecycle_audit["inactive_agent_snapshot_count"],
        "reset": lifecycle_audit["reset_deterministic_count"], "clone": lifecycle_audit["clone_deterministic_count"],
        "restore": lifecycle_audit["restore_replay_deterministic_count"], "recompute": lifecycle_audit["mask_recomputation_deterministic_count"],
        "dynamics_serialize": lifecycle_audit["dynamics_serialization_deterministic_count"],
        "dynamics_clone": lifecycle_audit["dynamics_clone_deterministic_count"],
        "dynamics_reset": lifecycle_audit["dynamics_reset_deterministic_count"],
        "dynamics_recompute": lifecycle_audit["dynamics_extract_recompute_deterministic_count"],
        "active_all_false": availability["active_all_false_mask_count"], "inactive_all_false": availability["inactive_all_false_mask_count"],
        "hold": availability["hold_available_count"], "serve": availability["serve_available_count"], "skip": availability["conditional_skip_available_count"],
        "identity": lifecycle_audit["identity_failure_count"], "occurrence": lifecycle_audit["occurrence_lookup_failure_count"],
        "binding_failures": binding_audit["total_drift_or_failure_count"], "future": lifecycle_audit["future_information_violation_count"],
        "pytest": pytest_result["passed_test_count"],
    }
    writer.json("k9_readiness_decision.json", readiness)
    writer.json("claim_guard_status.json", guards)
    writer.json(
        "run_manifest.json",
        {
            "created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "mode": "validate", "runner_path": str(RUNNER_PATH),
            "runner_sha256": k5.sha256_file(RUNNER_PATH), "python_executable": sys.executable, "python_version": sys.version.split()[0],
            "platform": platform.platform(), "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "upstream_integrity": upstreams, "new_bis_api_call_count": 0, "db_query_count": 0, "db_write_count": 0,
            "policy_execution_count": 0, "checkpoint_reuse_count": 0, "k10_execution_count": 0, "mappo_training_count": 0,
        },
    )
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": gate["gate"], "readiness": gate["readiness"], "final_decision": decision})
    writer.text("final_report.md", final_report(summary))
    write_manifest_and_lock(writer, gate)
    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_k9.json", "_PV8_K9_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise K9Error(f"K9 artifact integrity failure: {own}")
    if not passed:
        raise K9Error(f"K9 lifecycle validation failed: {lifecycle_audit}, {availability}, {binding_audit}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"readiness: {gate['readiness']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run_validation(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
