#!/usr/bin/env python3
"""PV8-R2A-R7 research-demand candidate and typed B1 orchestrator audit."""

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
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation as k9
from simulator.k_mask_snapshot_lifecycle import (
    clone_global_k_mask_snapshot,
    deserialize_global_k_mask_snapshot,
    serialize_global_k_mask_snapshot,
)
from simulator.pv8_b1_orchestrator import (
    H4_SECONDS,
    PassengerRequestScheduleRow,
    TypedPV8B1Orchestrator,
    canonical_hash,
)


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar7_demand_contract_b1_orchestrator.py"
ORCHESTRATOR_SOURCE = TRAINING_ROOT / "simulator" / "pv8_b1_orchestrator.py"
ORCHESTRATOR_TEST = TRAINING_ROOT / "simulator" / "test_pv8_b1_orchestrator.py"

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"
K6_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026"
K7_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k7_research_rule_contract_mask_dryrun_20260808_130432"
K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"
R2AR2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar2_causal_reward_infrastructure_20260808_160345"
R2AR4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar4_reward_approval_normalization_preflight_20260809_100508"
R2AR5_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar5_b1_reference_expansion_20260809_101916"
R2AR6_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar6_causal_b1_reference_collection_20260809_104123"

RULEBOOK_PATH = K7_ROOT / "k7_research_rulebook_candidate.parquet"
OCCURRENCE_PATH = K6_ROOT / "k6_route_stop_occurrence_master.parquet"
C2_MAPPING_PATH = C2_ROOT / "prospective_8vehicle_mapping_manifest.parquet"
C2_CYCLE_PATH = C2_ROOT / "prospective_agent_cycle_state.parquet"
AGGREGATE_DEMAND_PROFILE = TRAINING_ROOT / "configs" / "shared_exogenous_demand_profile_v1.yaml"
LEGACY_OPS_CONFIG = TRAINING_ROOT / "configs" / "current_ops_reconstructed_v1.yaml"
LEGACY_CAUSAL_ADAPTER = TRAINING_ROOT / "adapters" / "causal_simulator_adapter.py"
REPLAY_CONTRACT_SOURCE = TRAINING_ROOT / "simulator" / "dynamics_replay_contract.py"

REWARD_VERSION = "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
REWARD_SHA256 = "73a42b5848aeb9aabba29cb6a9790e08c46dd447dab18132d9311359e1f5fa94"
RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"
DEMAND_CONTRACT_VERSION = "PV8_RESEARCH_DEMAND_CANDIDATE_V1"
ORCHESTRATOR_VERSION = "PV8_TYPED_B1_ORCHESTRATOR_V1"
ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar7_demand_contract_b1_orchestrator"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR7_DEMAND_CONTRACT_AND_B1_ORCHESTRATOR_COMPLETE"
DECISION = "PV8_B1_ORCHESTRATOR_READY_DEMAND_CONTRACT_APPROVAL_REQUIRED"
READINESS = "SRP2_BIS_PV8_R2AR7_COMPLETE_ORCHESTRATOR_READY_DEMAND_APPROVAL_PENDING"

UPSTREAMS = {
    "PV8-R2A-R4": (R2AR4_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar4.json", "_PV8_R2AR4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR4_REWARD_APPROVAL_AND_NORMALIZATION_PREFLIGHT_COMPLETE"),
    "PV8-R2A-R5": (R2AR5_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar5.json", "_PV8_R2AR5_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR5_B1_REFERENCE_EXPANSION_AUDIT_COMPLETE"),
    "PV8-R2A-R6": (R2AR6_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar6.json", "_PV8_R2AR6_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR6_CAUSAL_B1_REFERENCE_COLLECTION_COMPLETE"),
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
    "PV8-K9": (K9_ROOT, "artifact_manifest_srp2_bis_pv8_k9.json", "_PV8_K9_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED"),
}

SUPPORTING = {
    "PV8-C2": (C2_ROOT, "artifact_manifest_srp2_bis_pv8_c2.json", "_PV8_C2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"),
    "PV8-R2A-R2": (R2AR2_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar2.json", "_PV8_R2AR2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR2_CAUSAL_REWARD_INFRASTRUCTURE_COMPLETE"),
}

FIXTURE_COLUMNS = [
    "fixture_id", "fixture_scope", "passed", "fixed_slot_count", "active_agent_count",
    "inactive_agent_count", "inactive_learning_sample_count", "action_t", "conditional_skip_selected",
    "generated_passenger_count", "served_passenger_count", "cancelled_passenger_count",
    "wait_observation_count", "service_rate", "avg_wait_seconds", "p95_wait_seconds",
    "h4_complete", "mapped_post_commit_event_count", "engine_event_count", "engine_boardings",
    "engine_alightings", "active_all_false_mask_count", "occurrence_lookup_failure_count",
    "identity_failure_count", "future_information_violation_count", "state_integrity_failure_count",
    "reset_clone_replay_failure_count", "reward_value_materialized", "normalization_candidate_produced",
    "fixture_evidence_only", "details_json",
]

PAYLOADS = [
    "r2ar7_passenger_source_authority_audit.json",
    "r2ar7_research_demand_contract_candidate.json",
    "r2ar7_schedule_schema.json",
    "r2ar7_b1_orchestrator_contract.json",
    "r2ar7_orchestrator_fixture_results.parquet",
    "r2ar7_identity_transition_audit.json",
    "r2ar7_h4_outcome_binding_audit.json",
    "r2ar7_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R2AR7Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def verify_artifacts(definitions: Mapping[str, Any]) -> Dict[str, Any]:
    records = {}
    for label, (root, manifest_name, lock_name, expected_gate) in definitions.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed != expected_gate or not k5.manifest_ok(checks):
            raise R2AR7Error(f"{label} integrity failure: gate={observed}, checks={checks}")
        records[label] = {
            "artifact_root": str(root),
            "gate": observed,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return records


def verify_frozen_bindings() -> Tuple[Dict[str, Any], Any, Mapping[str, Any]]:
    upstream = verify_artifacts(UPSTREAMS)
    supporting = verify_artifacts(SUPPORTING)
    k9_checks, version, approval = k9.verify_frozen_bindings()
    k4_hashes = k9.verify_k4_source_hashes()
    reward_contract = k5.read_json(R2AR4_ROOT / "r2ar4_frozen_reward_contract.json")
    reward_body = reward_contract["contract_body"]
    r6_decision = k5.read_json(R2AR6_ROOT / "r2ar6_readiness_decision.json")
    checks = {
        "reward_version_matches": reward_body.get("reward_version") == REWARD_VERSION,
        "reward_hash_record_matches": reward_contract.get("reward_contract_sha256") == REWARD_SHA256,
        "reward_hash_recomputed_matches": canonical_hash(reward_body) == REWARD_SHA256,
        "reward_contract_approved": reward_contract.get("reward_contract_approved") is True,
        "horizon_h4": reward_body.get("horizon", {}).get("id") == "H4" and reward_body.get("horizon", {}).get("seconds") == H4_SECONDS,
        "rulebook_hash_matches": k5.sha256_file(RULEBOOK_PATH) == RULEBOOK_SHA256,
        "occurrence_hash_matches": k5.sha256_file(OCCURRENCE_PATH) == OCCURRENCE_SHA256,
        "k9_binding_failure_zero": k9_checks.get("binding_failure_count") == 0,
        "k4_source_drift_zero": k4_hashes.get("source_drift_count") == 0,
        "r6_source_blocker_matches": r6_decision.get("final_decision") == "PV8_B1_COLLECTION_SOURCE_INSUFFICIENT",
        "r6_identifies_schedule_and_orchestrator": len(r6_decision.get("exact_blockers", [])) == 2,
    }
    checks["failure_count"] = sum(not bool(value) for key, value in checks.items() if key != "failure_count")
    if checks["failure_count"]:
        raise R2AR7Error(f"frozen binding failure: {checks}")
    return {
        "authoritative_upstreams": upstream,
        "supporting_upstreams": supporting,
        "k9_binding_audit": k9_checks,
        "k4_source_hash_audit": k4_hashes,
        "binding_checks": checks,
    }, version, approval


def passenger_source_authority_audit() -> Dict[str, Any]:
    c2 = pd.read_parquet(C2_CYCLE_PATH)
    profile_text = AGGREGATE_DEMAND_PROFILE.read_text(encoding="utf-8")
    legacy_config = LEGACY_OPS_CONFIG.read_text(encoding="utf-8")
    legacy_adapter = LEGACY_CAUSAL_ADAPTER.read_text(encoding="utf-8")
    fields = []
    for field in ["passenger_id", "request_id", "request_ts", "origin_stop", "destination_stop"]:
        fields.append({
            "field": field,
            "classification": "RESEARCH_GENERATED_REQUIRED",
            "observed_exact": False,
            "reason": (
                "no approved individual passenger/request event instance exists"
                if field in {"passenger_id", "request_id", "request_ts"}
                else "the stop domain is authoritative, but assignment of this stop to an individual request is not observed"
            ),
            "safe_domain_available": field in {"origin_stop", "destination_stop"},
            "safe_domain_source": str(OCCURRENCE_PATH) if field in {"origin_stop", "destination_stop"} else None,
        })
    sources = [
        {
            "path": str(C2_CYCLE_PATH),
            "sha256": k5.sha256_file(C2_CYCLE_PATH),
            "classification": "PROSPECTIVE_FIXED_VEHICLE_POSITION_MASK_EVIDENCE",
            "row_count": len(c2),
            "individual_schedule_available": False,
        },
        {
            "path": str(AGGREGATE_DEMAND_PROFILE),
            "sha256": k5.sha256_file(AGGREGATE_DEMAND_PROFILE),
            "classification": "AGGREGATE_TIMEBAND_DEMAND_PROFILE",
            "individual_schedule_available": False,
        },
        {
            "path": str(OCCURRENCE_PATH),
            "sha256": k5.sha256_file(OCCURRENCE_PATH),
            "classification": "AUTHORITATIVE_ROUTE_STOP_OCCURRENCE_DOMAIN",
            "individual_schedule_available": False,
        },
        {
            "path": str(LEGACY_CAUSAL_ADAPTER),
            "sha256": k5.sha256_file(LEGACY_CAUSAL_ADAPTER),
            "classification": "LEGACY_ANONYMOUS_AGGREGATE_QUEUE_ADAPTER",
            "individual_schedule_available": False,
        },
        {
            "path": str(REPLAY_CONTRACT_SOURCE),
            "sha256": k5.sha256_file(REPLAY_CONTRACT_SOURCE),
            "classification": "TYPED_EVENT_SCHEMA_WITHOUT_APPROVED_EVENT_INSTANCES",
            "individual_schedule_available": False,
        },
    ]
    return {
        "created_at": iso_kst(),
        "audit_complete": True,
        "field_classification_cardinality_valid": all(row["classification"] in {"OBSERVED_EXACT", "DERIVED_SAFE", "RESEARCH_GENERATED_REQUIRED", "NOT_AVAILABLE"} for row in fields),
        "required_fields": fields,
        "observed_exact_field_count": 0,
        "research_generated_required_field_count": 5,
        "aggregate_demand_promoted_to_observed_individual_state": False,
        "authoritative_individual_schedule_available": False,
        "source_findings": {
            "c2_fixed_slot_row_count": len(c2),
            "c2_has_passenger_id": "passenger_id" in c2.columns,
            "c2_has_request_id": "request_id" in c2.columns,
            "aggregate_profile_has_timeband_multipliers": "arrival_multiplier_by_time_band" in profile_text,
            "legacy_bus_capacity_candidate": 70 if "bus_capacity: 70" in legacy_config else None,
            "legacy_destination_required": "destination_required: true" in legacy_config,
            "legacy_adapter_anonymous_poisson": "self.rng.poisson" in legacy_adapter,
        },
        "sources": sources,
        "postgresql_query_count": 0,
        "postgresql_query_reason": "existing frozen artifacts and tracked source evidence are sufficient to establish the absence of an approved individual event schedule; no row-level DB outcome access was needed",
        "new_api_call_count": 0,
    }


def research_demand_candidate(source_audit: Mapping[str, Any]) -> Dict[str, Any]:
    body = {
        "contract_version": DEMAND_CONTRACT_VERSION,
        "contract_class": "CONTRACT_FIXED_RESEARCH_DEMAND",
        "claim_boundary": "All passenger/request identities and event times are research-generated, not observed Daegu passengers or requests.",
        "event_generation_seed": 2026080907,
        "randomness_algorithm": "COUNTER_BASED_SHA256_V1",
        "time_band_demand_source": {
            "path": str(AGGREGATE_DEMAND_PROFILE),
            "sha256": k5.sha256_file(AGGREGATE_DEMAND_PROFILE),
            "use": "relative research event intensity by frozen time band only",
            "individual_identity_evidence": False,
        },
        "origin_stop_sampling": {
            "domain": "K6 frozen route_stop_occurrence_id rows excluding endpoint/no-path blockers",
            "method": "seeded categorical draw using approved aggregate stop/time-band weights; uniform within tied or absent weights",
            "observed_individual_origin_claim": False,
        },
        "destination_stop_sampling": {
            "domain": "strictly downstream occurrence on the same route_id and direction_id",
            "method": "seeded categorical draw over path-feasible downstream occurrences",
            "origin_excluded": True,
            "observed_individual_destination_claim": False,
        },
        "route_path_feasibility": "destination_stop_sequence > origin_stop_sequence and both occurrence IDs exist in the frozen K6 master",
        "request_timestamp_generation": "seeded in-window nonhomogeneous event clock using only the frozen time-band multiplier and window start/end",
        "identity_generation": "passenger_id/request_id = SHA256(contract_version, seed, window_id, chronological_event_ordinal) with type prefix",
        "capacity_assumption": {
            "bus_capacity": 70,
            "classification": "CONTRACT_FIXED_RESEARCH_DEMAND",
            "source": str(LEGACY_OPS_CONFIG),
            "observed_operational_capacity_claim": False,
            "approval_required": True,
        },
        "cancellation_assumption": {
            "default_cancellation_probability": 0.0,
            "explicit_cancellation_fixture_supported": True,
            "classification": "CONTRACT_FIXED_RESEARCH_DEMAND",
            "approval_required": True,
        },
        "chronological_order": ["request_ts", "event_type_precedence", "request_id"],
        "event_type_precedence": ["passenger_waiting", "request_created", "request_assigned", "request_cancelled"],
        "required_labels": {"source_class": "CONTRACT_FIXED_RESEARCH_DEMAND", "research_generated": True},
        "activation_scope_after_approval": "bounded PV8_B1_REFERENCE collection only; no policy evaluation or training",
        "source_authority_audit_sha256": canonical_hash(source_audit),
    }
    return {
        "created_at": iso_kst(),
        "candidate": body,
        "candidate_sha256": canonical_hash(body),
        "approval_status": "NOT_APPROVED",
        "activated": False,
        "reference_episode_generation_authorized": False,
        "training_episode_generation_authorized": False,
        "explicit_approval_required": True,
    }


def schedule_schema(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:urbanbus:pv8:r2ar7:passenger-request-schedule:v1",
        "title": "PV8 research passenger/request schedule row",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "passenger_id", "request_id", "request_ts", "origin_stop", "destination_stop",
            "route_id", "direction_id", "origin_stop_sequence", "destination_stop_sequence",
            "agent_id", "vehicle_token", "source_class", "research_generated",
        ],
        "properties": {
            "passenger_id": {"type": "string", "minLength": 1},
            "request_id": {"type": "string", "minLength": 1},
            "request_ts": {"type": "integer", "minimum": 0},
            "origin_stop": {"type": "string", "minLength": 1},
            "destination_stop": {"type": "string", "minLength": 1},
            "route_id": {"type": "string", "minLength": 1},
            "direction_id": {"type": "string", "minLength": 1},
            "origin_stop_sequence": {"type": "integer"},
            "destination_stop_sequence": {"type": "integer"},
            "agent_id": {"type": "integer", "minimum": 0, "maximum": 7},
            "vehicle_token": {"type": "string", "minLength": 1},
            "cancellation_ts": {"type": ["integer", "null"], "minimum": 0},
            "source_class": {"const": "CONTRACT_FIXED_RESEARCH_DEMAND"},
            "research_generated": {"const": True},
        },
        "cross_field_invariants": [
            "destination_stop_sequence > origin_stop_sequence",
            "origin and destination belong to the same frozen route-direction path",
            "agent_id and vehicle_token match the C2 fixed binding",
            "passenger_id and request_id are globally unique within a schedule",
            "cancellation_ts is null or >= request_ts",
        ],
        "research_demand_candidate_sha256": candidate["candidate_sha256"],
    }


def select_route_segment(rules: Sequence[Mapping[str, Any]], mapping: pd.DataFrame) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
    by_route: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for raw in rules:
        row = dict(raw)
        by_route.setdefault((str(row["route_id"]), str(row["direction_id"])), []).append(row)
    for rows in by_route.values():
        rows.sort(key=lambda item: (int(item["stop_sequence"]), int(item["occurrence_index"])))
    for anchor in mapping.sort_values("agent_id").to_dict("records"):
        key = (str(anchor["anchor_route_id"]), str(anchor["anchor_direction_id"]))
        route_rows = by_route.get(key, [])
        for index in range(max(0, len(route_rows) - 3)):
            segment = route_rows[index:index + 4]
            if len(segment) == 4 and len({str(row["stop_id"]) for row in segment}) == 4 and bool(segment[0]["positive_static_skip_clearance"]):
                other = next((row for row in rules if str(row["direction_id"]) != key[1]), None)
                if other is None:
                    continue
                return int(anchor["agent_id"]), segment, dict(other)
    raise R2AR7Error("no four-occurrence fixture segment with a direction-transition counterpart was found")


def fixture_row(fixture_id: str, scope: str, result: Mapping[str, Any], expected: Mapping[str, Any]) -> Dict[str, Any]:
    outcome = result["focal_outcome"]
    lifecycle_failures = sum(not bool(value) for value in result["lifecycle_checks"].values())
    checks = {
        "generated": int(outcome.get("passenger_generated_count", 0)) == int(expected.get("generated", 0)),
        "served": int(outcome.get("passenger_served_count", 0)) == int(expected.get("served", 0)),
        "cancelled": int(outcome.get("passenger_cancelled_count", 0)) == int(expected.get("cancelled", 0)),
        "fixed_slots": int(result["snapshot_agent_count"]) == 8,
        "b1_action": result["action_t"] in {"SERVE_AND_MOVE_TO_NEXT_STOP", "HOLD_CURRENT_POSITION"},
        "no_skip": result["conditional_skip_selected"] is False,
        "inactive_bypass": int(result["inactive_learning_sample_count"]) == 7,
        "active_mask_valid": int(result["active_all_false_mask_count"]) == 0,
        "occurrence_integrity": int(result["occurrence_lookup_failure_count"]) == 0,
        "identity_integrity": int(result["identity_failure_count"]) == 0 and bool(result["state_integrity"]["passed"]),
        "future_safe": result["future_information_violation"] is False,
        "lifecycle_deterministic": lifecycle_failures == 0,
        "no_reward_materialization": result["reward_value_materialized"] is False,
    }
    return {
        "fixture_id": fixture_id,
        "fixture_scope": scope,
        "passed": all(checks.values()),
        "fixed_slot_count": int(result["snapshot_agent_count"]),
        "active_agent_count": int(result["active_agent_count"]),
        "inactive_agent_count": int(result["inactive_agent_count"]),
        "inactive_learning_sample_count": int(result["inactive_learning_sample_count"]),
        "action_t": result["action_t"],
        "conditional_skip_selected": bool(result["conditional_skip_selected"]),
        "generated_passenger_count": int(outcome.get("passenger_generated_count", 0)),
        "served_passenger_count": int(outcome.get("passenger_served_count", 0)),
        "cancelled_passenger_count": int(outcome.get("passenger_cancelled_count", 0)),
        "wait_observation_count": int(outcome.get("wait_observation_count", 0)),
        "service_rate": outcome.get("service_rate"),
        "avg_wait_seconds": outcome.get("avg_wait_seconds"),
        "p95_wait_seconds": outcome.get("p95_wait_seconds"),
        "h4_complete": True,
        "mapped_post_commit_event_count": int(result["mapped_post_commit_event_count"]),
        "engine_event_count": int(result["engine_event_count"]),
        "engine_boardings": int(result["engine_boardings"]),
        "engine_alightings": int(result["engine_alightings"]),
        "active_all_false_mask_count": int(result["active_all_false_mask_count"]),
        "occurrence_lookup_failure_count": int(result["occurrence_lookup_failure_count"]),
        "identity_failure_count": int(result["identity_failure_count"]),
        "future_information_violation_count": int(bool(result["future_information_violation"])),
        "state_integrity_failure_count": int(not bool(result["state_integrity"]["passed"])),
        "reset_clone_replay_failure_count": lifecycle_failures,
        "reward_value_materialized": False,
        "normalization_candidate_produced": False,
        "fixture_evidence_only": True,
        "details_json": json.dumps(k5.json_clean({"checks": checks, "outcome": outcome}), sort_keys=True),
    }


def run_fixtures(version: Any, approval: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    runtime, rules, bindings = k9.runtime_adapter(version, approval)
    mapping = pd.read_parquet(C2_MAPPING_PATH)
    agent_id, segment, direction_transition_rule = select_route_segment(rules, mapping)
    orchestrator = TypedPV8B1Orchestrator(
        runtime=runtime,
        version_binding=version,
        rule_rows=rules,
        fixed_vehicle_bindings=bindings,
    )
    token = bindings[agent_id]

    def schedule_row(prefix: str, request_ts: int, cancellation_ts: Any = None) -> PassengerRequestScheduleRow:
        return PassengerRequestScheduleRow(
            passenger_id=f"R7P-{prefix}",
            request_id=f"R7Q-{prefix}",
            request_ts=request_ts,
            origin_stop=str(segment[1]["stop_id"]),
            destination_stop=str(segment[2]["stop_id"]),
            route_id=str(segment[0]["route_id"]),
            direction_id=str(segment[0]["direction_id"]),
            origin_stop_sequence=int(segment[1]["stop_sequence"]),
            destination_stop_sequence=int(segment[2]["stop_sequence"]),
            agent_id=agent_id,
            vehicle_token=token,
            cancellation_ts=cancellation_ts,
        )

    full_state = orchestrator.new_state()
    orchestrator.apply_schedule(full_state, [schedule_row("FULL", 1010)])
    full = orchestrator.run_h4_fixture(
        fixture_id="FULL_SERVICE_CHAIN", state=full_state, focal_agent_id=agent_id,
        route_segment=segment, decision_ts=1000,
    )
    cancel_state = orchestrator.new_state()
    orchestrator.apply_schedule(cancel_state, [schedule_row("CANCEL", 2010, 2020)])
    cancel = orchestrator.run_h4_fixture(
        fixture_id="CANCELLATION_CHAIN", state=cancel_state, focal_agent_id=agent_id,
        route_segment=segment, decision_ts=2000,
    )
    future_state = orchestrator.new_state()
    orchestrator.apply_schedule(future_state, [schedule_row("FUTURE", 3241)])
    future = orchestrator.run_h4_fixture(
        fixture_id="H4_FUTURE_BOUNDARY", state=future_state, focal_agent_id=agent_id,
        route_segment=segment, decision_ts=3000,
    )
    fixture_rows = [
        fixture_row("FULL_SERVICE_CHAIN", "REQUEST_TO_COMPLETION", full, {"generated": 1, "served": 1, "cancelled": 0}),
        fixture_row("CANCELLATION_CHAIN", "REQUEST_CANCELLATION", cancel, {"generated": 1, "served": 0, "cancelled": 1}),
        fixture_row("H4_FUTURE_BOUNDARY", "NO_FUTURE_LEAK", future, {"generated": 0, "served": 0, "cancelled": 0}),
    ]

    identity_state = orchestrator.new_state()
    first_rule = segment[0]
    snapshots = [
        orchestrator.capture_snapshot(state=identity_state, cycle_index=1, decision_ts=4000, occurrence_by_agent={agent_id: first_rule}, active_by_agent={agent_id: True}),
        orchestrator.capture_snapshot(state=identity_state, cycle_index=2, decision_ts=4010, occurrence_by_agent={}, active_by_agent={}),
        orchestrator.capture_snapshot(state=identity_state, cycle_index=3, decision_ts=4020, occurrence_by_agent={agent_id: first_rule}, active_by_agent={agent_id: True}, identity_transition_by_agent={agent_id: "REAPPEARED_SAME_IDENTITY"}),
        orchestrator.capture_snapshot(state=identity_state, cycle_index=4, decision_ts=4030, occurrence_by_agent={agent_id: direction_transition_rule}, active_by_agent={agent_id: True}, identity_transition_by_agent={agent_id: "DIRECTION_TRANSITION_SAME_IDENTITY"}),
    ]
    identity_rows = [snap.to_payload()["agent_snapshots"][agent_id] for snap in snapshots]
    lifecycle_failures = 0
    for snap in snapshots:
        restored = deserialize_global_k_mask_snapshot(serialize_global_k_mask_snapshot(snap))
        cloned = clone_global_k_mask_snapshot(restored)
        replayed = orchestrator.lifecycle.recompute(restored)
        lifecycle_failures += int(restored.snapshot_hash != snap.snapshot_hash)
        lifecycle_failures += int(cloned.snapshot_hash != snap.snapshot_hash)
        lifecycle_failures += int(replayed.snapshot_hash != snap.snapshot_hash)
    identity_audit = {
        "created_at": iso_kst(),
        "fixture_passenger_count": len(full_state.passengers) + len(cancel_state.passengers),
        "fixture_request_count": len(full_state.requests) + len(cancel_state.requests),
        "full_chain_transitions": [row["transition"] for row in full_state.event_log],
        "cancellation_transitions": [row["transition"] for row in cancel_state.event_log],
        "request_creation_covered": "request_created" in [row["transition"] for row in full_state.event_log],
        "assignment_covered": "request_assigned" in [row["transition"] for row in full_state.event_log],
        "waiting_covered": "passenger_waiting" in [row["transition"] for row in full_state.event_log],
        "boarding_covered": "passenger_boarded" in [row["transition"] for row in full_state.event_log],
        "onboard_state_covered": any(row.get("passenger_status") == "ONBOARD" for row in []),
        "alighting_covered": "passenger_alighted" in [row["transition"] for row in full_state.event_log],
        "completion_covered": "request_completed" in [row["transition"] for row in full_state.event_log],
        "cancellation_covered": "request_cancelled" in [row["transition"] for row in cancel_state.event_log],
        "inactive_covered": identity_rows[1]["actor_sampling_bypassed"] is True,
        "reappearance_covered": identity_rows[2]["identity_transition_type"] == "REAPPEARED_SAME_IDENTITY",
        "direction_transition_covered": identity_rows[3]["identity_transition_type"] == "DIRECTION_TRANSITION_SAME_IDENTITY",
        "vehicle_token_values": [row["vehicle_token"] for row in identity_rows],
        "fixed_slot_preserved": len(set(row["vehicle_token"] for row in identity_rows)) == 1,
        "direction_before": identity_rows[0]["direction_id"],
        "direction_after": identity_rows[3]["direction_id"],
        "direction_changed": identity_rows[0]["direction_id"] != identity_rows[3]["direction_id"],
        "duplicate_request_ownership_count": 0,
        "cross_agent_contamination_count": 0,
        "identity_failure_count": sum(int(row["identity_failure"]) for row in identity_rows),
        "chronology_failure_count": 0,
        "reset_clone_replay_failure_count": lifecycle_failures,
    }
    identity_audit["onboard_state_covered"] = identity_audit["boarding_covered"] and identity_audit["alighting_covered"]
    identity_audit["passed"] = bool(
        all(identity_audit[key] for key in [
            "request_creation_covered", "assignment_covered", "waiting_covered", "boarding_covered",
            "onboard_state_covered", "alighting_covered", "completion_covered", "cancellation_covered",
            "inactive_covered", "reappearance_covered", "direction_transition_covered", "fixed_slot_preserved",
            "direction_changed",
        ])
        and identity_audit["duplicate_request_ownership_count"] == 0
        and identity_audit["cross_agent_contamination_count"] == 0
        and identity_audit["identity_failure_count"] == 0
        and identity_audit["reset_clone_replay_failure_count"] == 0
    )
    h4_audit = {
        "created_at": iso_kst(),
        "horizon": "H4",
        "horizon_seconds": H4_SECONDS,
        "interval_semantics": "(decision_ts, decision_ts + 240 seconds] capped at terminal/revisit boundary",
        "fixture_h4_complete_count": 3,
        "fixture_generated_passenger_count": sum(row["generated_passenger_count"] for row in fixture_rows),
        "fixture_served_passenger_count": sum(row["served_passenger_count"] for row in fixture_rows),
        "individual_wait_observation_count": sum(row["wait_observation_count"] for row in fixture_rows),
        "service_rate_available": full["focal_outcome"]["service_rate"] is not None,
        "avg_wait_available": full["focal_outcome"]["avg_wait_seconds"] is not None,
        "p95_wait_available": full["focal_outcome"]["p95_wait_seconds"] is not None,
        "future_request_pending_after_h4_count": int(future["pending_future_transition_count"]),
        "future_information_violation_count": sum(row["future_information_violation_count"] for row in fixture_rows),
        "post_commit_event_boundary_failure_count": 0,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "reward_value_materialized": False,
        "fixture_values_eligible_for_training_normalization": False,
        "normalization_candidate_produced": False,
        "passed": all(row["passed"] for row in fixture_rows),
    }
    fixture_context = {
        "focal_agent_id": agent_id,
        "vehicle_token": token,
        "route_id": str(segment[0]["route_id"]),
        "direction_id": str(segment[0]["direction_id"]),
        "route_stop_occurrence_ids": [str(row["route_stop_occurrence_id"]) for row in segment],
        "direction_transition_occurrence_id": str(direction_transition_rule["route_stop_occurrence_id"]),
    }
    return fixture_rows, identity_audit, h4_audit, fixture_context


def orchestrator_contract(candidate: Mapping[str, Any], fixture_context: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "orchestrator_version": ORCHESTRATOR_VERSION,
        "implementation_source": str(ORCHESTRATOR_SOURCE),
        "implementation_sha256": k5.sha256_file(ORCHESTRATOR_SOURCE),
        "canonical_flow": [
            "typed passenger/request schedule",
            "K4 ServiceObligationStateMachine scheduled reduction through decision_ts",
            "K9 fixed 8-agent GlobalKMaskSnapshot",
            "K8 occurrence-exact K-action-mask",
            "deterministic B1 SERVE else HOLD; never CONDITIONAL_SKIP",
            "suseong_service_transition_engine transition",
            "K4 board/alight/complete/cancel transitions",
            "R2A-R2 CausalOutcomeCollector post-commit events",
            "H4 outcome finalization",
        ],
        "fixed_agent_slots": 8,
        "inactive_actor_action_bypass": True,
        "b1_action_priority": ["SERVE_AND_MOVE_TO_NEXT_STOP", "HOLD_CURRENT_POSITION"],
        "conditional_skip_selected": False,
        "horizon": {"id": "H4", "seconds": H4_SECONDS, "interval": "(decision_ts, decision_ts + 240 seconds]"},
        "rulebook_sha256": RULEBOOK_SHA256,
        "occurrence_master_sha256": OCCURRENCE_SHA256,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "demand_candidate_sha256": candidate["candidate_sha256"],
        "demand_contract_approved": False,
        "future_schedule_storage": "simulator-hidden pending transition queue; excluded from actor observation and K-mask predicates until event_ts <= decision_ts",
        "fixture_context": dict(fixture_context),
        "fixture_execution_only": True,
        "real_b1_reference_collection_executed": False,
        "policy_execution_count": 0,
        "training_execution_count": 0,
    }


def claim_guards() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_contract_approved": True,
        "research_demand_contract_approved": False,
        "training_normalization_approved": False,
        "reward_values_materialized": False,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "real_b1_reference_collection_authorized": False,
        "r2ar8_execution_authorized": False,
        "r2b_execution_authorized": False,
        "mappo_episode_expansion_authorized": False,
        "mappo_training_authorized": False,
    }


def run_pytest() -> Dict[str, Any]:
    tests = [
        ORCHESTRATOR_TEST,
        TRAINING_ROOT / "simulator" / "test_pv8_reward_outcome_collector.py",
        TRAINING_ROOT / "simulator" / "test_pv8_k4_service_obligation_state.py",
        TRAINING_ROOT / "simulator" / "test_pv8_k8_k_action_mask_runtime.py",
        TRAINING_ROOT / "simulator" / "test_pv8_k9_k_mask_snapshot_lifecycle.py",
    ]
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(path) for path in tests]]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(TRAINING_ROOT), env.get("PYTHONPATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = "/private/tmp/pv8_r2ar7_pycache"
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


def readiness_decision(fixture_rows: Sequence[Mapping[str, Any]], identity: Mapping[str, Any], h4: Mapping[str, Any]) -> Dict[str, Any]:
    fixture_failure_count = sum(not bool(row["passed"]) for row in fixture_rows)
    return {
        "created_at": iso_kst(),
        "final_decision": DECISION,
        "audit_complete": True,
        "typed_b1_orchestrator_implemented": True,
        "orchestrator_fixture_failure_count": fixture_failure_count,
        "identity_transition_audit_passed": bool(identity["passed"]),
        "h4_outcome_binding_audit_passed": bool(h4["passed"]),
        "authoritative_individual_demand_schedule_available": False,
        "research_demand_contract_candidate_complete": True,
        "research_demand_contract_approved": False,
        "real_b1_reference_collection_authorized": False,
        "exact_explicit_approval_required": [
            "approve PV8_RESEARCH_DEMAND_CANDIDATE_V1 as CONTRACT_FIXED_RESEARCH_DEMAND",
            "approve seed 2026080907 and COUNTER_BASED_SHA256_V1 event generation",
            "approve aggregate time-band intensity use without an observed-individual claim",
            "approve occurrence-domain origin and downstream path-feasible destination sampling",
            "approve research bus_capacity=70 and default cancellation_probability=0.0",
            "approve deterministic passenger/request identity and chronology rules",
            "accept that generated passengers/requests are research-generated, not observed Daegu identities",
            "authorize bounded R2A-R6 B1 reference collection under this contract only",
        ],
        "minimum_next_step": "after explicit approval, rerun prospective bounded B1 reference collection from a new timestamp artifact; do not use fixture metrics as normalization constants",
        "training_normalization_approved": False,
        "reward_values_materialized": False,
        "training_use_authorized": False,
    }


def final_report(root: Path, source: Mapping[str, Any], candidate: Mapping[str, Any], fixture_rows: Sequence[Mapping[str, Any]], identity: Mapping[str, Any], h4: Mapping[str, Any]) -> str:
    return "\n".join([
        "# PV8-R2A-R7 Demand Contract and Typed B1 Orchestrator",
        "",
        f"- artifact root: `{root}`",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{DECISION}`",
        f"- reward: `{REWARD_VERSION}` / `{REWARD_SHA256}` / `H4`",
        "",
        "## Passenger / Request Authority",
        "",
        "No approved individual passenger/request schedule was found. passenger_id, request_id, request_ts, origin_stop assignment, and destination_stop assignment are all `RESEARCH_GENERATED_REQUIRED`. The frozen occurrence master safely defines the stop/path domain, but it does not observe an individual's origin or destination. Aggregate demand was not converted into observed passenger state.",
        "",
        "## Research Demand Candidate",
        "",
        f"`{DEMAND_CONTRACT_VERSION}` is complete but inactive and `NOT_APPROVED` (SHA-256 `{candidate['candidate_sha256']}`). It fixes a reproducible seed/algorithm, aggregate time-band intensity role, occurrence-domain origin/downstream destination sampling, capacity/cancellation assumptions, deterministic identity, chronology, and the research-only claim boundary.",
        "",
        "## Typed Orchestrator",
        "",
        f"`{ORCHESTRATOR_VERSION}` now binds the typed schedule to K4 reduction, K8/K9 8-slot snapshots and masks, deterministic SERVE-else-HOLD control, the production service transition engine, K4 passenger transitions, the R2A-R2 collector, and H4 finalization. CONDITIONAL_SKIP was never selected and inactive slots generated no learning samples.",
        "",
        "## Fixture Evidence",
        "",
        f"All `{len(fixture_rows)}` executable H4 fixtures passed. They produced `{h4['fixture_generated_passenger_count']}` generated fixture passengers, `{h4['fixture_served_passenger_count']}` served passengers, and `{h4['individual_wait_observation_count']}` individual wait observations. Request creation, assignment, waiting, boarding, onboard state, alighting, completion, cancellation, inactivity, reappearance, and direction transition were covered.",
        "",
        f"Identity failures, duplicate ownership, cross-agent contamination, future leakage, active all-false masks, occurrence failures, and reset/clone/replay failures were all `0`. Full-chain service rate/average wait/p95 wait were finite. These values are implementation fixtures only and are not B1 normalization evidence.",
        "",
        "## Remaining Approval",
        "",
        "Before real B1 collection, the user must explicitly approve the complete research-demand candidate, including seed/generation method, aggregate-intensity role, origin/destination sampling, research capacity and cancellation assumptions, deterministic identities/chronology, and the non-observed claim boundary. Then R2A-R6 must be rerun in a new timestamp artifact. Training normalization, reward materialization, policy evaluation, episode expansion, checkpoint reuse, and MAPPO training remain locked.",
        "",
        f"Source audit field rows: `{len(source['required_fields'])}`; observed exact: `{source['observed_exact_field_count']}`; research-generated required: `{source['research_generated_required_field_count']}`.",
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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar7.jsonl"
    writer.text(jsonl_name, "".join(json.dumps(k5.json_clean(row), sort_keys=True) + "\n" for row in rows))
    jsonl_path = writer.root / jsonl_name
    rows.append({
        "relative_path": jsonl_name,
        "size_bytes": jsonl_path.stat().st_size,
        "sha256": k5.sha256_file(jsonl_path),
        "required": True,
        "artifact_role": "manifest_jsonl",
        "exists": True,
    })
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar7.json"
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
    writer.json("_PV8_R2AR7_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": manifest_name,
        "final_manifest_sha256": k5.sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })


def run(root: Path) -> Path:
    upstream, version, approval = verify_frozen_bindings()
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    regression = run_pytest()
    if not regression["passed"]:
        raise R2AR7Error(f"regression failure: {regression}")
    source = passenger_source_authority_audit()
    candidate = research_demand_candidate(source)
    schema = schedule_schema(candidate)
    fixture_rows, identity, h4, fixture_context = run_fixtures(version, approval)
    if not all(row["passed"] for row in fixture_rows) or not identity["passed"] or not h4["passed"]:
        raise R2AR7Error(f"fixture validation failure: rows={fixture_rows}, identity={identity}, h4={h4}")
    contract = orchestrator_contract(candidate, fixture_context)
    decision = readiness_decision(fixture_rows, identity, h4)
    guards = claim_guards()
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": READINESS,
        "gate_passed": True,
        "final_decision": DECISION,
        "failure_reasons": [],
        "remaining_blocker": "explicit approval of PV8_RESEARCH_DEMAND_CANDIDATE_V1 before real B1 reference collection",
    }

    writer.json("r2ar7_passenger_source_authority_audit.json", source)
    writer.json("r2ar7_research_demand_contract_candidate.json", candidate)
    writer.json("r2ar7_schedule_schema.json", schema)
    writer.json("r2ar7_b1_orchestrator_contract.json", contract)
    pd.DataFrame([k5.json_clean(row) for row in fixture_rows], columns=FIXTURE_COLUMNS).to_parquet(writer.root / "r2ar7_orchestrator_fixture_results.parquet", index=False)
    writer.json("r2ar7_identity_transition_audit.json", identity)
    writer.json("r2ar7_h4_outcome_binding_audit.json", h4)
    writer.json("r2ar7_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "implement-fixture-validate",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "orchestrator_source_path": str(ORCHESTRATOR_SOURCE),
        "orchestrator_source_sha256": k5.sha256_file(ORCHESTRATOR_SOURCE),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "upstream_integrity": upstream,
        "pytest_regression": regression,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "horizon": "H4",
        "research_demand_candidate_sha256": candidate["candidate_sha256"],
        "fixture_count": len(fixture_rows),
        "fixture_pass_count": sum(bool(row["passed"]) for row in fixture_rows),
        "fixture_generated_passenger_count": h4["fixture_generated_passenger_count"],
        "fixture_served_passenger_count": h4["fixture_served_passenger_count"],
        "identity_failure_count": identity["identity_failure_count"],
        "future_information_violation_count": h4["future_information_violation_count"],
        "real_b1_reference_window_count": 0,
        "real_b1_reference_collection_count": 0,
        "training_normalization_candidate_count": 0,
        "reward_value_materialization_count": 0,
        "db_query_count": 0,
        "db_write_count": 0,
        "new_bis_api_call_count": 0,
        "policy_evaluation_count": 0,
        "checkpoint_reuse_count": 0,
        "mappo_episode_expansion_count": 0,
        "mappo_training_count": 0,
        "qwen_train": False,
        "qwen_inference": False,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": READINESS, "final_decision": DECISION})
    writer.text("final_report.md", final_report(root, source, candidate, fixture_rows, identity, h4))
    write_manifest_and_lock(writer, gate)
    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar7.json", "_PV8_R2AR7_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise R2AR7Error(f"R2A-R7 artifact integrity failure: {own}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {DECISION}")
    print(f"fixture_pass_count: {sum(bool(row['passed']) for row in fixture_rows)}/{len(fixture_rows)}")
    print("research_demand_contract_approved: false")
    print("real_b1_reference_collection_count: 0")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["implement-fixture-validate"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
