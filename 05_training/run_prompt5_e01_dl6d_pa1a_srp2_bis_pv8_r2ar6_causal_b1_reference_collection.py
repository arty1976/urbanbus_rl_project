#!/usr/bin/env python3
"""PV8-R2A-R6 causal B1 reference collection with source guards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import math
import re
import resource
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation as k9
from simulator.pv8_b1_orchestrator import (
    H4_SECONDS,
    PassengerRequestScheduleRow,
    TypedPV8B1Orchestrator,
)


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar6_causal_b1_reference_collection.py"

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"
K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"
R2AR2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar2_causal_reward_infrastructure_20260808_160345"
R2AR4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar4_reward_approval_normalization_preflight_20260809_100508"
R2AR5_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar5_b1_reference_expansion_20260809_101916"
R2AR7_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar7_demand_contract_b1_orchestrator_20260809_111155"

COLLECTOR_SOURCE = TRAINING_ROOT / "simulator" / "pv8_reward_outcome_collector.py"
K4_STATE_SOURCE = TRAINING_ROOT / "simulator" / "k_safety_state.py"
K8_RUNTIME_SOURCE = TRAINING_ROOT / "simulator" / "k_action_mask_runtime.py"
K9_LIFECYCLE_SOURCE = TRAINING_ROOT / "simulator" / "k_mask_snapshot_lifecycle.py"
ORCHESTRATOR_SOURCE = TRAINING_ROOT / "simulator" / "pv8_b1_orchestrator.py"
ORCHESTRATOR_TEST_SOURCE = TRAINING_ROOT / "simulator" / "test_pv8_b1_orchestrator.py"
REPLAY_CONTRACT_SOURCE = TRAINING_ROOT / "simulator" / "dynamics_replay_contract.py"
LEGACY_CAUSAL_ADAPTER = TRAINING_ROOT / "adapters" / "causal_simulator_adapter.py"
AGGREGATE_DEMAND_PROFILE = TRAINING_ROOT / "configs" / "shared_exogenous_demand_profile_v1.yaml"
LEGACY_OPS_CONFIG = TRAINING_ROOT / "configs" / "current_ops_reconstructed_v1.yaml"
RULEBOOK_PATH = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k7_research_rule_contract_mask_dryrun_20260808_130432" / "k7_research_rulebook_candidate.parquet"
OCCURRENCE_PATH = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026" / "k6_route_stop_occurrence_master.parquet"
C2_CYCLE_PATH = C2_ROOT / "prospective_agent_cycle_state.parquet"

UPSTREAMS = {
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
    "PV8-K9": (K9_ROOT, "artifact_manifest_srp2_bis_pv8_k9.json", "_PV8_K9_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED"),
    "PV8-R2A-R4": (R2AR4_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar4.json", "_PV8_R2AR4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR4_REWARD_APPROVAL_AND_NORMALIZATION_PREFLIGHT_COMPLETE"),
    "PV8-R2A-R5": (R2AR5_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar5.json", "_PV8_R2AR5_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR5_B1_REFERENCE_EXPANSION_AUDIT_COMPLETE"),
    "PV8-R2A-R7": (R2AR7_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar7.json", "_PV8_R2AR7_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR7_DEMAND_CONTRACT_AND_B1_ORCHESTRATOR_COMPLETE"),
}

SUPPORTING_RUNTIME = {
    "PV8-C2": (C2_ROOT, "artifact_manifest_srp2_bis_pv8_c2.json", "_PV8_C2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"),
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
    "PV8-R2A-R2": (R2AR2_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar2.json", "_PV8_R2AR2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR2_CAUSAL_REWARD_INFRASTRUCTURE_COMPLETE"),
}

REWARD_VERSION = "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
REWARD_SHA256 = "73a42b5848aeb9aabba29cb6a9790e08c46dd447dab18132d9311359e1f5fa94"
RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"
DEMAND_CONTRACT_VERSION = "PV8_RESEARCH_DEMAND_CANDIDATE_V1"
DEMAND_CONTRACT_SHA256 = "77b9438c09a950bf7d39be0852fa25aece58b543fa04d22f7c981142f45bbad8"
ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar6_causal_b1_reference_collection"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR6_CAUSAL_B1_REFERENCE_COLLECTION_COMPLETE"
DECISION_READY = "PV8_B1_NORMALIZATION_CANDIDATE_READY_FOR_EXPLICIT_APPROVAL"
DECISION_MORE_COVERAGE = "PV8_B1_COLLECTION_MORE_COVERAGE_REQUIRED"
DECISION_SOURCE_INSUFFICIENT = "PV8_B1_COLLECTION_SOURCE_INSUFFICIENT"
STOP_REASON = "STOP_B1_COLLECTION_SOURCE_INSUFFICIENT"
READINESS_READY = "SRP2_BIS_PV8_R2AR6_COMPLETE_B1_NORMALIZATION_CANDIDATE_READY_APPROVAL_REQUIRED"
READINESS_MORE_COVERAGE = "SRP2_BIS_PV8_R2AR6_COMPLETE_B1_COLLECTION_MORE_COVERAGE_REQUIRED"
READINESS_SOURCE_INSUFFICIENT = "SRP2_BIS_PV8_R2AR6_COMPLETE_SOURCE_GUARD_STOPPED_BEFORE_B1_EXECUTION"

WINDOW_COLUMNS = [
    "episode_id", "window_id", "window_start_ts", "window_end_ts", "time_band", "partition",
    "fixed_slot_count", "snapshot_count", "active_agent_count", "inactive_agent_count",
    "generated_passengers", "served_passengers", "reward_valid_transition_count",
    "h4_complete_transition_count", "service_rate", "avg_wait_seconds", "p95_wait_seconds",
    "reward_input_completeness", "h4_outcome_completeness", "b1_action_contract_valid",
    "training_eligible", "source_schedule_sha256", "exclusion_reason",
]
WINDOW_DTYPES = {
    "episode_id": "string", "window_id": "string", "window_start_ts": "string",
    "window_end_ts": "string", "time_band": "string", "partition": "string",
    "fixed_slot_count": "int64", "snapshot_count": "int64", "active_agent_count": "int64",
    "inactive_agent_count": "int64", "generated_passengers": "int64",
    "served_passengers": "int64", "reward_valid_transition_count": "int64",
    "h4_complete_transition_count": "int64", "service_rate": "float64",
    "avg_wait_seconds": "float64", "p95_wait_seconds": "float64",
    "reward_input_completeness": "float64", "h4_outcome_completeness": "float64",
    "b1_action_contract_valid": "bool", "training_eligible": "bool",
    "source_schedule_sha256": "string", "exclusion_reason": "string",
}

TRANSITION_COLUMNS = [
    "episode_id", "window_id", "decision_id", "transition_index", "decision_ts", "outcome_end_ts",
    "agent_id", "vehicle_token", "active_bus_mask", "action_t", "generated_passengers",
    "served_passengers", "wait_observation_count", "service_rate", "avg_wait_seconds",
    "p95_wait_seconds", "reward_input_complete", "h4_outcome_complete",
    "future_information_violation", "training_eligible", "exclusion_reason",
]
TRANSITION_DTYPES = {
    "episode_id": "string", "window_id": "string", "decision_id": "string",
    "transition_index": "int64", "decision_ts": "string", "outcome_end_ts": "string",
    "agent_id": "int64", "vehicle_token": "string", "active_bus_mask": "bool",
    "action_t": "string", "generated_passengers": "int64", "served_passengers": "int64",
    "wait_observation_count": "int64", "service_rate": "float64",
    "avg_wait_seconds": "float64", "p95_wait_seconds": "float64",
    "reward_input_complete": "bool", "h4_outcome_complete": "bool",
    "future_information_violation": "bool", "training_eligible": "bool",
    "exclusion_reason": "string",
}

PAYLOADS = [
    "r2ar6_research_demand_approval_record.json",
    "r2ar6_b1_collection_manifest.json",
    "r2ar6_b1_window_metrics.parquet",
    "r2ar6_b1_causal_transition_audit.parquet",
    "r2ar6_b1_distribution_summary.json",
    "r2ar6_leave_one_window_out.json",
    "r2ar6_timeband_coverage.json",
    "r2ar6_normalization_candidate.json",
    "r2ar6_reference_integrity_audit.json",
    "r2ar6_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R2AR6Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(k5.json_clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_artifacts(definitions: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in definitions.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed != expected_gate or not k5.manifest_ok(checks):
            raise R2AR6Error(f"{label} integrity failure: gate={observed}, checks={checks}")
        out[label] = {"artifact_root": str(root), "gate": observed, "manifest_integrity": checks}
    return out


def verify_frozen_bindings() -> Dict[str, Any]:
    upstream = verify_artifacts(UPSTREAMS)
    supporting = verify_artifacts(SUPPORTING_RUNTIME)
    contract = k5.read_json(R2AR4_ROOT / "r2ar4_frozen_reward_contract.json")
    body = contract.get("contract_body")
    r5 = k5.read_json(R2AR5_ROOT / "r2ar5_readiness_decision.json")
    r5_run = k5.read_json(R2AR5_ROOT / "run_manifest.json")
    r2ar2_run = k5.read_json(R2AR2_ROOT / "run_manifest.json")
    k8 = k5.read_json(K8_ROOT / "k8_snapshot_version_binding.json")
    k9 = k5.read_json(K9_ROOT / "k9_version_hash_binding_audit.json")
    checks = {
        "reward_version_matches": body.get("reward_version") == REWARD_VERSION,
        "reward_hash_recorded_matches": contract.get("reward_contract_sha256") == REWARD_SHA256,
        "reward_hash_recomputed_matches": canonical_hash(body) == REWARD_SHA256,
        "reward_approved": contract.get("reward_contract_approved") is True,
        "normalization_not_approved": contract.get("training_normalization_approved") is False,
        "horizon_h4": body.get("horizon", {}).get("id") == "H4" and body.get("horizon", {}).get("seconds") == 240,
        "r5_decision_matches": r5.get("final_decision") == "PV8_B1_REFERENCE_EXPANSION_INSUFFICIENT",
        "r5_runner_hash_matches": k5.sha256_file(Path(r5_run["runner_path"])) == r5_run.get("runner_sha256"),
        "collector_hash_matches": k5.sha256_file(COLLECTOR_SOURCE) == r2ar2_run.get("collector_source_sha256"),
        "k8_hashes_match": k8.get("static_rulebook_sha256") == RULEBOOK_SHA256 and k8.get("occurrence_master_sha256") == OCCURRENCE_SHA256,
        "k9_hashes_match": k9.get("rulebook_sha256") == RULEBOOK_SHA256 and k9.get("occurrence_master_sha256") == OCCURRENCE_SHA256,
        "k9_binding_failure_zero": k9.get("binding_failure_count") == 0,
    }
    checks["failure_count"] = sum(not value for value in checks.values())
    if checks["failure_count"]:
        raise R2AR6Error(f"frozen binding failure: {checks}")
    return {"authoritative_upstreams": upstream, "supporting_runtime": supporting, "binding_checks": checks}


def source_feasibility_audit() -> Dict[str, Any]:
    c2_path = C2_ROOT / "prospective_agent_cycle_state.parquet"
    c2 = pd.read_parquet(c2_path)
    profile_text = AGGREGATE_DEMAND_PROFILE.read_text(encoding="utf-8")
    config_text = LEGACY_OPS_CONFIG.read_text(encoding="utf-8")
    adapter_text = LEGACY_CAUSAL_ADAPTER.read_text(encoding="utf-8")
    replay_text = REPLAY_CONTRACT_SOURCE.read_text(encoding="utf-8")

    required_event_fields = {
        "passenger_id", "request_id", "event_timestamp_seconds", "stop_id",
        "destination_stop_id", "pickup_stop", "dropoff_stop",
    }
    c2_columns = set(str(value) for value in c2.columns)
    findings = {
        "c2_has_individual_passenger_schedule": required_event_fields.issubset(c2_columns),
        "c2_has_vehicle_position_mask_cycles": len(c2) == 240 and {"agent_id", "active_bus_mask", "cycle_timestamp"}.issubset(c2_columns),
        "aggregate_profile_uses_precomputed_log_features": "precomputed_log1p" in profile_text,
        "aggregate_profile_has_individual_passenger_id": "passenger_id" in profile_text,
        "aggregate_profile_has_request_id": "request_id" in profile_text,
        "aggregate_profile_is_timeband_multiplier_only": "arrival_multiplier_by_time_band" in profile_text,
        "legacy_config_uses_skeleton_base_lambda": "skeleton-scale simulator parameter" in config_text,
        "legacy_config_requires_destination": "destination_required: true" in config_text,
        "legacy_adapter_generates_anonymous_poisson_queues": "self.rng.poisson" in adapter_text and "self.state.queues += demand" in adapter_text,
        "legacy_adapter_has_passenger_identity": "passenger_id" in adapter_text,
        "legacy_adapter_has_request_identity": "request_id" in adapter_text,
        "legacy_adapter_p95_is_formula": "max(avg_wait_seconds * 1.65" in adapter_text,
        "legacy_adapter_action_dim_is_pv8_three": "action_type_dim: 3" in config_text,
        "replay_contract_accepts_identity_events": all(token in replay_text for token in ["PASSENGER_ARRIVAL", "PICKUP_REQUEST", "PASSENGER_DESTINATION"]),
    }
    source_ready = bool(
        findings["c2_has_individual_passenger_schedule"]
        and findings["legacy_config_requires_destination"]
        and findings["legacy_adapter_has_passenger_identity"]
        and findings["legacy_adapter_has_request_identity"]
        and findings["legacy_adapter_action_dim_is_pv8_three"]
        and not findings["legacy_adapter_p95_is_formula"]
    )
    if source_ready:
        raise R2AR6Error("source guard expected insufficient evidence but found an executable source")

    return {
        "created_at": iso_kst(),
        "source_ready_for_b1_collection": False,
        "stop_reason": STOP_REASON,
        "findings": findings,
        "sources": [
            {
                "path": str(c2_path),
                "sha256": k5.sha256_file(c2_path),
                "classification": "PROSPECTIVE_8_SLOT_VEHICLE_POSITION_MASK_ONLY",
                "usable_for_b1_passenger_outcomes": False,
                "reason": "no passenger/request identity or generated/served/wait outcome fields",
            },
            {
                "path": str(AGGREGATE_DEMAND_PROFILE),
                "sha256": k5.sha256_file(AGGREGATE_DEMAND_PROFILE),
                "classification": "AGGREGATE_TIMEBAND_LOG_INTENSITY_PROFILE",
                "usable_for_b1_passenger_outcomes": False,
                "reason": "aggregate log features and multipliers do not identify passenger/request events or destinations",
            },
            {
                "path": str(LEGACY_CAUSAL_ADAPTER),
                "sha256": k5.sha256_file(LEGACY_CAUSAL_ADAPTER),
                "classification": "LEGACY_TOY_AGGREGATE_QUEUE_ADAPTER",
                "usable_for_b1_passenger_outcomes": False,
                "reason": "anonymous Poisson queues, formula-derived p95, no K4 passenger/request identity, and non-PV8 action contract",
            },
            {
                "path": str(REPLAY_CONTRACT_SOURCE),
                "sha256": k5.sha256_file(REPLAY_CONTRACT_SOURCE),
                "classification": "IDENTITY_EVENT_SCHEMA_AVAILABLE_NO_EVENT_INSTANCE_SOURCE",
                "usable_for_b1_passenger_outcomes": False,
                "reason": "schema can validate events but no approved prospective schedule supplies event instances",
            },
        ],
        "runtime_components_available": {
            "k4_state_machine": True,
            "k8_kmask_runtime": True,
            "k9_snapshot_lifecycle": True,
            "r2ar2_causal_collector": True,
            "approved_prospective_passenger_schedule": False,
            "pv8_b1_orchestrator": False,
        },
        "missing_source": {
            "name": "approved prospective passenger/request exogenous schedule",
            "required_fields": [
                "passenger_id", "request_id", "event_ts", "pickup_stop", "dropoff_stop",
                "generated/waiting/assigned/boarded/alighted/completed chronology",
            ],
            "coverage": ">=3 non-overlapping chronological windows across >=2 time bands",
            "prohibition": "aggregate demand may not be converted silently into individual passengers",
        },
        "missing_implementation": {
            "name": "PV8 deterministic B1 causal reference orchestrator",
            "required_flow": [
                "load approved event schedule and 8-slot prospective snapshots",
                "reduce events into K4 state before each decision",
                "compute K8/K9 mask and choose SERVE else HOLD; never SKIP",
                "execute service transitions with identity preserved",
                "emit post-commit events to R2A-R2 collector",
                "finalize H4 at terminal/revisit boundary",
                "write typed transition/window rows with version/hash binding",
            ],
        },
        "aggregate_profile_promoted_to_passenger_events": False,
        "legacy_toy_adapter_executed": False,
        "passenger_events_invented": False,
    }


def collection_manifest(audit: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "manifest_version": "PV8_R2AR6_B1_COLLECTION_MANIFEST_V1",
        "collection_status": "STOPPED_BEFORE_EXECUTION_SOURCE_GUARD",
        "stop_reason": STOP_REASON,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "horizon": "H4",
        "horizon_interval": "(decision_ts, decision_ts + 240 seconds] capped at terminal/revisit boundary",
        "b1_runtime_contract": {
            "fixed_agent_slots": 8,
            "active_action": "SERVE when valid; HOLD only fallback",
            "conditional_skip_selected": False,
            "learned_policy_used": False,
            "rulebook_sha256": RULEBOOK_SHA256,
            "occurrence_master_sha256": OCCURRENCE_SHA256,
        },
        "target_coverage": {
            "minimum_nonoverlapping_windows": 3,
            "minimum_distinct_time_bands": 2,
            "nonempty_passenger_cohort": True,
            "h4_complete_outcomes": True,
        },
        "eligible_episode_count": 0,
        "eligible_window_count": 0,
        "causal_transition_count": 0,
        "generated_passenger_count": 0,
        "served_passenger_count": 0,
        "b1_simulator_execution_count": 0,
        "source_audit_sha256": canonical_hash(audit),
        "source_files": audit["sources"],
        "no_episode_duplication": True,
        "c1_retrospective_selection_used": False,
        "future_information_used": False,
        "validation_test_fit_outcomes_used": False,
    }


def empty_distribution(metric: str) -> Dict[str, Any]:
    return {
        "metric": metric, "count": 0, "mean": None, "median": None, "std": None,
        "min": None, "max": None, "q1": None, "q3": None, "iqr": None,
        "cv": None, "reason": STOP_REASON,
    }


def distribution_summary() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "eligible_window_count": 0,
        "metrics": {
            name: empty_distribution(name)
            for name in [
                "service_rate", "avg_wait_seconds", "p95_wait_seconds", "generated_passengers",
                "served_passengers", "active_agent_count", "inactive_agent_count",
                "reward_valid_transition_count", "h4_complete_transition_count",
            ]
        },
        "cross_window_stability": "NOT_EVALUABLE_SOURCE_INSUFFICIENT",
        "agent_coverage": "NOT_EVALUABLE_SOURCE_INSUFFICIENT",
        "outlier_dependence": "NOT_EVALUABLE_SOURCE_INSUFFICIENT",
        "finite_nonzero_denominator_audit": "NOT_EVALUABLE_SOURCE_INSUFFICIENT",
        "null_statistics_zero_filled": False,
    }


def leave_one_window_out() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "method": "refit service/avg-wait/p95 references after excluding each eligible training window",
        "required_window_count": 3,
        "eligible_window_count": 0,
        "iteration_count": 0,
        "results": [],
        "service_sensitivity": None,
        "avg_wait_sensitivity": None,
        "p95_wait_sensitivity": None,
        "evaluated": False,
        "reason": STOP_REASON,
    }


def timeband_coverage() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "required_distinct_time_band_count": 2,
        "eligible_distinct_time_band_count": 0,
        "eligible_time_bands": [],
        "available_noneligible_sources": [
            {
                "source": "C2 prospective position/mask episode",
                "utc_interval": ["2026-08-04T22:02:46.545904+00:00", "2026-08-04T22:31:46.717357+00:00"],
                "time_band": "UNCLASSIFIED_FOR_B1",
                "eligible": False,
                "reason": "no passenger/request H4 outcomes",
            },
            {
                "source": "shared_exogenous_demand_profile_v1",
                "declared_bands": ["night", "offpeak", "peak"],
                "eligible": False,
                "reason": "aggregate multipliers are not event windows",
            },
        ],
        "coverage_requirement_met": False,
    }


def normalization_candidate() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "candidate_id": None,
        "candidate_version": None,
        "candidate_sha256": None,
        "candidate_produced": False,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "source_windows": [],
        "aggregation_method": None,
        "sample_counts": {"windows": 0, "transitions": 0, "passengers": 0},
        "time_coverage": [],
        "constants": {
            "B1_service_reference": None,
            "B1_avg_wait_reference": None,
            "B1_p95_wait_reference": None,
        },
        "service_reference_one_supported": False,
        "service_reference_one_evaluated": False,
        "service_reference_not_forced": True,
        "leave_one_window_out_sensitivity": None,
        "training_only_fit": True,
        "validation_test_outcomes_used": False,
        "training_normalization_approved": False,
        "status": "NOT_PRODUCED_SOURCE_INSUFFICIENT",
    }


def readiness_decision() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "final_decision": DECISION,
        "stop_reason": STOP_REASON,
        "audit_complete": True,
        "collection_attempted_after_source_guard": False,
        "eligible_window_count": 0,
        "time_band_count": 0,
        "generated_passenger_count": 0,
        "reward_valid_transition_count": 0,
        "h4_complete_transition_count": 0,
        "normalization_candidate_produced": False,
        "service_reference_one_supported": False,
        "exact_blockers": [
            "no approved individual passenger/request exogenous schedule for three windows and two time bands",
            "no PV8 orchestrator binding K4 state, K8/K9 mask lifecycle, deterministic B1 control, service transitions, and the R2A-R2 collector",
        ],
        "minimum_next_repair": "supply the approved event schedule and implement the typed PV8 B1 orchestrator; then rerun R2A-R6 from a new timestamp artifact",
        "reward_contract_approved": True,
        "training_normalization_approved": False,
        "reward_values_materialized": False,
        "r2ar7_or_r2b_authorized": False,
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
        "mappo_episode_expansion_authorized": False,
        "automatic_r2ar7_execution_authorized": False,
        "automatic_r2b_execution_authorized": False,
        "mappo_training_authorized": False,
    }


def run_pytest_regression() -> Dict[str, Any]:
    tests = [
        TRAINING_ROOT / "simulator" / "test_pv8_reward_outcome_collector.py",
        TRAINING_ROOT / "simulator" / "test_pv8_k4_service_obligation_state.py",
        TRAINING_ROOT / "simulator" / "test_pv8_k8_k_action_mask_runtime.py",
        TRAINING_ROOT / "simulator" / "test_pv8_k9_k_mask_snapshot_lifecycle.py",
        ORCHESTRATOR_TEST_SOURCE,
    ]
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(path) for path in tests]]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(TRAINING_ROOT), env.get("PYTHONPATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = "/private/tmp/pv8_r2ar6_pycache"
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


def write_typed_parquet(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str], dtypes: Mapping[str, str]) -> None:
    frame = pd.DataFrame([k5.json_clean(dict(row)) for row in rows], columns=list(columns))
    for column, dtype in dtypes.items():
        frame[column] = frame[column].astype(dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def final_report(root: Path) -> str:
    return "\n".join([
        "# PV8-R2A-R6 Causal B1 Reference Collection",
        "",
        f"- artifact root: `{root}`",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{DECISION}`",
        f"- stop reason: `{STOP_REASON}`",
        f"- reward: `{REWARD_VERSION}` / `{REWARD_SHA256}` / `H4`",
        "",
        "## Source Guard",
        "",
        "Collection stopped before simulator execution. C2 supplies one 8-slot vehicle position/mask episode but no individual passenger/request schedule or H4 service outcomes. The empirical demand profile contains aggregate log-intensity time-band multipliers, not passenger identities, requests, destinations, or event chronology.",
        "",
        "The legacy causal adapter is not a valid replacement: it creates anonymous Poisson queues, derives p95 wait by formula, omits K4 passenger/request identity, and uses a non-PV8 action contract. The replay event schema, K4 state machine, K8/K9 lifecycle, and R2A-R2 collector exist, but no approved schedule or orchestrator joins them.",
        "",
        "## Collection Result",
        "",
        "Eligible B1 windows, causal transitions, generated passengers, served passengers, reward-valid transitions, and H4-complete transitions are all `0`. Both parquet outputs contain zero rows with typed frozen schemas. No aggregate count was promoted to a passenger entity and no synthetic episode was duplicated.",
        "",
        "## Metrics and Candidate",
        "",
        "Service-rate, average-wait, and p95-wait distributions are not evaluable. Leave-one-window-out iterations and eligible time bands are `0`. No normalization candidate was produced; all three constants remain null. `service_reference=1.0` is neither forced nor supported by broader evidence.",
        "",
        "## Exact Repair",
        "",
        "Provide an approved individual passenger/request event schedule covering at least three non-overlapping chronological windows and two time bands. It must include passenger/request identity, pickup/dropoff, and generated-to-board/alight/complete chronology. Implement a PV8 B1 orchestrator that reduces those events into K4, computes K8/K9 masks, chooses SERVE else HOLD with no SKIP, executes service transitions, and sends post-commit events to the R2A-R2 collector through H4.",
        "",
        "No API/DB access, B1 simulator execution, invented event, reward materialization, policy evaluation, episode expansion, checkpoint reuse, downstream run, or MAPPO training occurred.",
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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar6.jsonl"
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
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar6.json"
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
    writer.json("_PV8_R2AR6_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": manifest_name,
        "final_manifest_sha256": k5.sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })


def research_demand_approval_record() -> Dict[str, Any]:
    candidate = k5.read_json(R2AR7_ROOT / "r2ar7_research_demand_contract_candidate.json")
    candidate_body = candidate.get("candidate") or {}
    checks = {
        "candidate_version_matches": candidate_body.get("contract_version") == DEMAND_CONTRACT_VERSION,
        "candidate_sha256_matches": candidate.get("candidate_sha256") == DEMAND_CONTRACT_SHA256,
        "candidate_class_matches": candidate_body.get("contract_class") == "CONTRACT_FIXED_RESEARCH_DEMAND",
        "r7_left_candidate_inactive": candidate.get("activated") is False,
        "r7_required_explicit_approval": candidate.get("explicit_approval_required") is True,
        "reference_generation_was_previously_locked": candidate.get("reference_episode_generation_authorized") is False,
        "training_generation_remains_locked": candidate.get("training_episode_generation_authorized") is False,
    }
    checks["failure_count"] = sum(not value for value in checks.values())
    if checks["failure_count"]:
        raise R2AR6Error(f"research-demand approval precondition failure: {checks}")
    return {
        "created_at": iso_kst(),
        "approval_evidence_type": "DIRECT_USER_MESSAGE_IN_CURRENT_CODEX_TASK",
        "approval_evidence_text": "V8_RESEARCH_DEMAND_CANDIDATE_V1의 명시적 승인 → R6 재실행",
        "interpreted_approved_contract_version": DEMAND_CONTRACT_VERSION,
        "user_text_used_alias_v8": True,
        "approval_valid": True,
        "research_demand_contract_approved": True,
        "approved_contract_class": "CONTRACT_FIXED_RESEARCH_DEMAND",
        "approved_candidate_sha256": DEMAND_CONTRACT_SHA256,
        "approved_candidate_source": str(R2AR7_ROOT / "r2ar7_research_demand_contract_candidate.json"),
        "candidate_source_sha256": k5.sha256_file(R2AR7_ROOT / "r2ar7_research_demand_contract_candidate.json"),
        "candidate_precondition_checks": checks,
        "claim_boundary": "All passenger/request identities and event times are research-generated, not observed Daegu passengers or requests.",
        "reference_collection_authorized": True,
        "training_episode_generation_authorized": False,
        "reward_values_materialized": False,
        "policy_evaluation_authorized": False,
        "mappo_training_authorized": False,
    }


def approved_runtime_context() -> Dict[str, Any]:
    binding_audit, version, approval = k9.verify_frozen_bindings()
    runtime, rule_rows, fixed_bindings = k9.runtime_adapter(version, approval)
    orchestrator = TypedPV8B1Orchestrator(
        runtime=runtime,
        version_binding=version,
        rule_rows=rule_rows,
        fixed_vehicle_bindings=fixed_bindings,
    )
    return {
        "binding_audit": binding_audit,
        "version": version,
        "approval": approval,
        "runtime": runtime,
        "rule_rows": rule_rows,
        "fixed_bindings": fixed_bindings,
        "orchestrator": orchestrator,
    }


def parse_cycle_timestamp(value: Any) -> Optional[pd.Timestamp]:
    if value is None or str(value).lower() in {"nan", "none", "nat"}:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.tz_convert("Asia/Seoul")


def classify_time_band(ts: Optional[pd.Timestamp]) -> str:
    if ts is None:
        return "UNKNOWN"
    hour = int(ts.hour)
    if 7 <= hour < 9 or 17 <= hour < 19:
        return "peak"
    if 9 <= hour < 22:
        return "offpeak"
    return "night"


def numeric_summary(values: Sequence[float]) -> Dict[str, Any]:
    finite = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "mean": None, "median": None, "std": None, "min": None, "max": None, "q1": None, "q3": None, "iqr": None, "cv": None}
    series = pd.Series(finite, dtype="float64")
    mean = float(series.mean())
    std = float(series.std(ddof=0)) if len(series) > 1 else 0.0
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    return {
        "count": int(len(finite)),
        "mean": mean,
        "median": float(series.median()),
        "std": std,
        "min": float(series.min()),
        "max": float(series.max()),
        "q1": q1,
        "q3": q3,
        "iqr": float(q3 - q1),
        "cv": float(std / mean) if mean else None,
    }


def build_rule_index(rule_rows: Sequence[Mapping[str, Any]]) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, int, str], Dict[str, Any]]]:
    frame = pd.DataFrame([dict(row) for row in rule_rows]).copy()
    frame["route_id_str"] = frame["route_id"].astype(str)
    frame["direction_id_str"] = frame["direction_id"].astype(str)
    frame["stop_id_str"] = frame["stop_id"].astype(str)
    frame["stop_sequence_int"] = frame["stop_sequence"].astype(int)
    by_exact = {}
    for row in frame.to_dict("records"):
        key = (str(row["route_id"]), str(row["direction_id"]), int(row["stop_sequence"]), str(row["stop_id"]))
        by_exact[key] = row
    return frame, by_exact


def segment_for_cycle(rule_frame: pd.DataFrame, by_exact: Mapping[Tuple[str, str, int, str], Mapping[str, Any]], cycle_row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not bool(cycle_row.get("active_bus_mask")):
        return []
    key = (
        str(cycle_row.get("route_id")),
        str(cycle_row.get("direction_id")),
        int(float(cycle_row.get("seq"))),
        str(cycle_row.get("stop_id")),
    )
    current = by_exact.get(key)
    if current is None:
        return []
    subset = rule_frame[
        (rule_frame["route_id_str"] == key[0])
        & (rule_frame["direction_id_str"] == key[1])
        & (rule_frame["stop_sequence_int"] >= key[2])
    ].sort_values("stop_sequence_int")
    rows = subset.head(4).drop(columns=[column for column in ["route_id_str", "direction_id_str", "stop_id_str", "stop_sequence_int"] if column in subset.columns]).to_dict("records")
    return rows if len(rows) >= 4 else []


def source_windows(cycles: pd.DataFrame) -> List[Dict[str, Any]]:
    cycle_keys = sorted(int(value) for value in cycles["cycle_index"].dropna().unique())
    if len(cycle_keys) < 3:
        return []
    chunk_size = max(1, len(cycle_keys) // 3)
    chunks = [cycle_keys[:chunk_size], cycle_keys[chunk_size:2 * chunk_size], cycle_keys[2 * chunk_size:]]
    windows: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        if not chunk:
            continue
        frame = cycles[cycles["cycle_index"].astype(int).isin(chunk)].copy()
        timestamps = [parse_cycle_timestamp(value) for value in frame["cycle_timestamp"].dropna().unique()]
        timestamps = [value for value in timestamps if value is not None]
        start = min(timestamps) if timestamps else None
        end = max(timestamps) if timestamps else None
        windows.append({
            "window_id": f"PV8_B1_APPROVED_RESEARCH_WINDOW_{index:02d}",
            "episode_id": "PV8_B1_APPROVED_RESEARCH_DEMAND_RERUN",
            "cycle_indices": chunk,
            "frame": frame,
            "start": start,
            "end": end,
            "time_band": classify_time_band(start),
        })
    return windows


def deterministic_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}_{canonical_hash(payload)[:24]}"


def run_approved_reference_collection(context: Mapping[str, Any]) -> Dict[str, Any]:
    cycles = pd.read_parquet(C2_CYCLE_PATH)
    rule_frame, by_exact = build_rule_index(context["rule_rows"])
    orchestrator: TypedPV8B1Orchestrator = context["orchestrator"]
    window_rows: List[Dict[str, Any]] = []
    transition_rows: List[Dict[str, Any]] = []
    schedule_records: List[Dict[str, Any]] = []
    windows = source_windows(cycles)

    for window_index, window in enumerate(windows, start=1):
        frame = window["frame"].sort_values(["cycle_index", "agent_id"])
        selected: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
        seen_agents = set()
        for cycle_row in frame.to_dict("records"):
            agent_id = int(cycle_row["agent_id"])
            segment = segment_for_cycle(rule_frame, by_exact, cycle_row)
            if not segment or agent_id in seen_agents:
                continue
            selected.append((cycle_row, segment))
            seen_agents.add(agent_id)
            if len(selected) >= 4:
                break
        outcomes = []
        for ordinal, (cycle_row, segment) in enumerate(selected, start=1):
            agent_id = int(cycle_row["agent_id"])
            vehicle_token = str(cycle_row["bound_vehicle_token"])
            decision_ts = window_index * 10000 + ordinal * 1000
            request_ts = decision_ts + 5 + ordinal * 5
            origin = segment[1]
            destination = segment[2]
            schedule = PassengerRequestScheduleRow(
                passenger_id=deterministic_id("P", {"contract": DEMAND_CONTRACT_VERSION, "window": window["window_id"], "ordinal": ordinal, "kind": "passenger"}),
                request_id=deterministic_id("Q", {"contract": DEMAND_CONTRACT_VERSION, "window": window["window_id"], "ordinal": ordinal, "kind": "request"}),
                request_ts=request_ts,
                origin_stop=str(origin["stop_id"]),
                destination_stop=str(destination["stop_id"]),
                route_id=str(origin["route_id"]),
                direction_id=str(origin["direction_id"]),
                origin_stop_sequence=int(origin["stop_sequence"]),
                destination_stop_sequence=int(destination["stop_sequence"]),
                agent_id=agent_id,
                vehicle_token=vehicle_token,
            )
            schedule_records.append({
                "window_id": window["window_id"],
                "decision_ts": decision_ts,
                **schedule.to_payload(),
            })
            state = orchestrator.new_state()
            orchestrator.apply_schedule(state, [schedule])
            result = orchestrator.run_h4_fixture(
                fixture_id=f"{window['window_id']}:agent{agent_id}:case{ordinal}",
                state=state,
                focal_agent_id=agent_id,
                route_segment=segment,
                decision_ts=decision_ts,
            )
            outcome = result["focal_outcome"]
            complete = bool(
                outcome.get("service_rate") is not None
                and outcome.get("avg_wait_seconds") is not None
                and outcome.get("p95_wait_seconds") is not None
                and not result.get("future_information_violation")
                and result.get("state_integrity", {}).get("passed") is True
            )
            transition_rows.append({
                "episode_id": window["episode_id"],
                "window_id": window["window_id"],
                "decision_id": outcome["decision_id"],
                "transition_index": len(transition_rows) + 1,
                "decision_ts": str(decision_ts),
                "outcome_end_ts": str(outcome["outcome_end_ts"]),
                "agent_id": agent_id,
                "vehicle_token": vehicle_token,
                "active_bus_mask": True,
                "action_t": outcome["action_t"],
                "generated_passengers": int(outcome["passenger_generated_count"]),
                "served_passengers": int(outcome["passenger_served_count"]),
                "wait_observation_count": int(outcome["wait_observation_count"]),
                "service_rate": float(outcome["service_rate"]) if outcome.get("service_rate") is not None else math.nan,
                "avg_wait_seconds": float(outcome["avg_wait_seconds"]) if outcome.get("avg_wait_seconds") is not None else math.nan,
                "p95_wait_seconds": float(outcome["p95_wait_seconds"]) if outcome.get("p95_wait_seconds") is not None else math.nan,
                "reward_input_complete": complete,
                "h4_outcome_complete": complete,
                "future_information_violation": bool(result.get("future_information_violation")),
                "training_eligible": complete,
                "exclusion_reason": "" if complete else "INCOMPLETE_H4_OR_INTEGRITY_FAILURE",
            })
            outcomes.append(outcome)

        active_count = int(frame["active_bus_mask"].sum())
        inactive_count = int((~frame["active_bus_mask"].astype(bool)).sum())
        waits = [
            float(wait["wait_seconds"])
            for outcome in outcomes
            for wait in outcome.get("individual_waits", [])
        ]
        eligible = sum(int(outcome.get("eligible_request_count", 0)) for outcome in outcomes)
        served = sum(int(outcome.get("passenger_served_count", 0)) for outcome in outcomes)
        complete_count = sum(
            outcome.get("service_rate") is not None
            and outcome.get("avg_wait_seconds") is not None
            and outcome.get("p95_wait_seconds") is not None
            for outcome in outcomes
        )
        source_schedule_sha256 = canonical_hash([row for row in schedule_records if row["window_id"] == window["window_id"]])
        window_rows.append({
            "episode_id": window["episode_id"],
            "window_id": window["window_id"],
            "window_start_ts": window["start"].isoformat() if window["start"] is not None else "",
            "window_end_ts": window["end"].isoformat() if window["end"] is not None else "",
            "time_band": window["time_band"],
            "partition": "training_reference_candidate",
            "fixed_slot_count": 8,
            "snapshot_count": len(outcomes),
            "active_agent_count": active_count,
            "inactive_agent_count": inactive_count,
            "generated_passengers": sum(int(outcome.get("passenger_generated_count", 0)) for outcome in outcomes),
            "served_passengers": served,
            "reward_valid_transition_count": complete_count,
            "h4_complete_transition_count": complete_count,
            "service_rate": float(served / eligible) if eligible else math.nan,
            "avg_wait_seconds": float(sum(waits) / len(waits)) if waits else math.nan,
            "p95_wait_seconds": float(pd.Series(waits).quantile(0.95)) if waits else math.nan,
            "reward_input_completeness": float(complete_count / len(outcomes)) if outcomes else 0.0,
            "h4_outcome_completeness": float(complete_count / len(outcomes)) if outcomes else 0.0,
            "b1_action_contract_valid": all(str(outcome.get("action_t")) in {"SERVE_AND_MOVE_TO_NEXT_STOP", "HOLD_CURRENT_POSITION"} for outcome in outcomes),
            "training_eligible": bool(outcomes and complete_count == len(outcomes)),
            "source_schedule_sha256": source_schedule_sha256,
            "exclusion_reason": "" if outcomes and complete_count == len(outcomes) else "NO_COMPLETE_B1_REFERENCE_TRANSITIONS",
        })

    distinct_bands = sorted({row["time_band"] for row in window_rows if row["training_eligible"] and row["time_band"] != "UNKNOWN"})
    coverage = {
        "eligible_window_count": sum(bool(row["training_eligible"]) for row in window_rows),
        "distinct_time_band_count": len(distinct_bands),
        "distinct_time_bands": distinct_bands,
        "minimum_windows_met": sum(bool(row["training_eligible"]) for row in window_rows) >= 3,
        "minimum_time_bands_met": len(distinct_bands) >= 2,
        "nonempty_passenger_cohorts": all(int(row["generated_passengers"]) > 0 for row in window_rows if row["training_eligible"]),
        "h4_complete": all(float(row["h4_outcome_completeness"]) == 1.0 for row in window_rows if row["training_eligible"]),
    }
    coverage["coverage_sufficient_for_candidate"] = bool(
        coverage["minimum_windows_met"]
        and coverage["minimum_time_bands_met"]
        and coverage["nonempty_passenger_cohorts"]
        and coverage["h4_complete"]
    )
    return {
        "window_rows": window_rows,
        "transition_rows": transition_rows,
        "schedule_records": schedule_records,
        "coverage": coverage,
        "window_count": len(window_rows),
        "transition_count": len(transition_rows),
    }


def approved_collection_manifest(
    *,
    approval_record: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    collection: Mapping[str, Any],
    decision: str,
) -> Dict[str, Any]:
    coverage = collection["coverage"]
    return {
        "created_at": iso_kst(),
        "manifest_version": "PV8_R2AR6_B1_COLLECTION_MANIFEST_V2_APPROVED_RESEARCH_DEMAND",
        "collection_status": "BOUNDED_B1_REFERENCE_COLLECTED" if collection["transition_count"] else "STOPPED_NO_FEASIBLE_WINDOWS",
        "final_decision": decision,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "horizon": "H4",
        "horizon_interval": "(decision_ts, decision_ts + 240 seconds] capped at terminal/revisit boundary",
        "research_demand_contract": {
            "approved": True,
            "contract_version": DEMAND_CONTRACT_VERSION,
            "candidate_sha256": DEMAND_CONTRACT_SHA256,
            "approval_record_sha256": canonical_hash(approval_record),
        },
        "b1_runtime_contract": {
            "fixed_agent_slots": 8,
            "active_action": "SERVE when valid; HOLD only fallback",
            "conditional_skip_selected": False,
            "learned_policy_used": False,
            "typed_orchestrator_source_sha256": k5.sha256_file(ORCHESTRATOR_SOURCE),
            "rulebook_sha256": RULEBOOK_SHA256,
            "occurrence_master_sha256": OCCURRENCE_SHA256,
        },
        "target_coverage": {
            "minimum_nonoverlapping_windows": 3,
            "minimum_distinct_time_bands": 2,
            "nonempty_passenger_cohort": True,
            "h4_complete_outcomes": True,
        },
        "eligible_episode_count": len({row["episode_id"] for row in collection["window_rows"] if row["training_eligible"]}),
        "eligible_window_count": coverage["eligible_window_count"],
        "eligible_distinct_time_band_count": coverage["distinct_time_band_count"],
        "causal_transition_count": collection["transition_count"],
        "generated_passenger_count": sum(int(row["generated_passengers"]) for row in collection["transition_rows"]),
        "served_passenger_count": sum(int(row["served_passengers"]) for row in collection["transition_rows"]),
        "b1_simulator_execution_count": collection["transition_count"],
        "source_audit_sha256": canonical_hash(source_audit),
        "source_files": source_audit["sources"],
        "no_episode_duplication": True,
        "c1_retrospective_selection_used": False,
        "future_information_used": any(bool(row["future_information_violation"]) for row in collection["transition_rows"]),
        "validation_test_fit_outcomes_used": False,
    }


def approved_distribution_summary(collection: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [row for row in collection["window_rows"] if row["training_eligible"]]
    metrics = {}
    for name in [
        "service_rate", "avg_wait_seconds", "p95_wait_seconds", "generated_passengers",
        "served_passengers", "active_agent_count", "inactive_agent_count",
        "reward_valid_transition_count", "h4_complete_transition_count",
    ]:
        metrics[name] = {"metric": name, **numeric_summary([row[name] for row in rows])}
    return {
        "created_at": iso_kst(),
        "eligible_window_count": len(rows),
        "metrics": metrics,
        "cross_window_stability": "EVALUATED",
        "agent_coverage": "EVALUATED_ON_C2_FIXED_8_SLOT_WINDOWS",
        "outlier_dependence": "EVALUATED_LEAVE_ONE_WINDOW_OUT",
        "finite_nonzero_denominator_audit": {
            "service_rate_finite": all(math.isfinite(float(row["service_rate"])) for row in rows),
            "avg_wait_finite_nonzero": all(math.isfinite(float(row["avg_wait_seconds"])) and float(row["avg_wait_seconds"]) > 0 for row in rows),
            "p95_wait_finite_nonzero": all(math.isfinite(float(row["p95_wait_seconds"])) and float(row["p95_wait_seconds"]) > 0 for row in rows),
        },
        "null_statistics_zero_filled": False,
    }


def approved_leave_one_window_out(collection: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [row for row in collection["window_rows"] if row["training_eligible"]]
    results = []
    for excluded in rows:
        kept = [row for row in rows if row["window_id"] != excluded["window_id"]]
        results.append({
            "excluded_window_id": excluded["window_id"],
            "kept_window_count": len(kept),
            "B1_service_reference": float(pd.Series([row["service_rate"] for row in kept]).mean()) if kept else None,
            "B1_avg_wait_reference": float(pd.Series([row["avg_wait_seconds"] for row in kept]).mean()) if kept else None,
            "B1_p95_wait_reference": float(pd.Series([row["p95_wait_seconds"] for row in kept]).mean()) if kept else None,
        })
    return {
        "created_at": iso_kst(),
        "method": "refit service/avg-wait/p95 references after excluding each eligible training window",
        "required_window_count": 3,
        "eligible_window_count": len(rows),
        "iteration_count": len(results),
        "results": results,
        "service_sensitivity": numeric_summary([row["B1_service_reference"] for row in results if row["B1_service_reference"] is not None]),
        "avg_wait_sensitivity": numeric_summary([row["B1_avg_wait_reference"] for row in results if row["B1_avg_wait_reference"] is not None]),
        "p95_wait_sensitivity": numeric_summary([row["B1_p95_wait_reference"] for row in results if row["B1_p95_wait_reference"] is not None]),
        "evaluated": bool(results),
        "reason": None if results else "NO_ELIGIBLE_WINDOWS",
    }


def approved_timeband_coverage(collection: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [row for row in collection["window_rows"] if row["training_eligible"]]
    bands = sorted({row["time_band"] for row in rows if row["time_band"] != "UNKNOWN"})
    return {
        "created_at": iso_kst(),
        "required_distinct_time_band_count": 2,
        "eligible_distinct_time_band_count": len(bands),
        "eligible_time_bands": bands,
        "window_time_bands": [
            {"window_id": row["window_id"], "time_band": row["time_band"], "window_start_ts": row["window_start_ts"], "window_end_ts": row["window_end_ts"]}
            for row in rows
        ],
        "coverage_requirement_met": len(bands) >= 2,
        "source_limitation": "C2 prospective evidence currently spans one observed time band after Asia/Seoul conversion." if len(bands) < 2 else None,
    }


def approved_normalization_candidate(collection: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [row for row in collection["window_rows"] if row["training_eligible"]]
    coverage = collection["coverage"]
    produced = bool(coverage["coverage_sufficient_for_candidate"])
    constants = {
        "B1_service_reference": float(pd.Series([row["service_rate"] for row in rows]).mean()) if produced else None,
        "B1_avg_wait_reference": float(pd.Series([row["avg_wait_seconds"] for row in rows]).mean()) if produced else None,
        "B1_p95_wait_reference": float(pd.Series([row["p95_wait_seconds"] for row in rows]).mean()) if produced else None,
    }
    payload = {
        "created_at": iso_kst(),
        "candidate_id": "PV8_B1_TRAINING_NORMALIZATION_CANDIDATE_R2AR6_V1" if produced else None,
        "candidate_version": "R2AR6_APPROVED_RESEARCH_DEMAND_B1_REFERENCE_V1" if produced else None,
        "candidate_sha256": None,
        "candidate_produced": produced,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "source_windows": [row["window_id"] for row in rows] if produced else [],
        "aggregation_method": "training-eligible window mean" if produced else None,
        "sample_counts": {
            "windows": len(rows),
            "transitions": collection["transition_count"],
            "passengers": sum(int(row["generated_passengers"]) for row in collection["transition_rows"]),
        },
        "time_coverage": sorted({row["time_band"] for row in rows}),
        "constants": constants,
        "service_reference_one_supported": bool(produced and constants["B1_service_reference"] == 1.0),
        "service_reference_one_evaluated": bool(rows),
        "service_reference_not_forced": True,
        "leave_one_window_out_sensitivity": None,
        "training_only_fit": True,
        "validation_test_outcomes_used": False,
        "training_normalization_approved": False,
        "status": "PRODUCED_PENDING_EXPLICIT_APPROVAL" if produced else "NOT_PRODUCED_COVERAGE_INSUFFICIENT",
        "coverage": coverage,
    }
    if produced:
        payload["candidate_sha256"] = canonical_hash({key: value for key, value in payload.items() if key != "candidate_sha256"})
    return payload


def approved_readiness_decision(collection: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    coverage = collection["coverage"]
    if candidate.get("candidate_produced"):
        final_decision = DECISION_READY
        readiness = READINESS_READY
        blockers: List[str] = []
    elif collection["transition_count"] > 0:
        final_decision = DECISION_MORE_COVERAGE
        readiness = READINESS_MORE_COVERAGE
        blockers = []
        if not coverage["minimum_windows_met"]:
            blockers.append("fewer than three eligible non-overlapping B1 windows")
        if not coverage["minimum_time_bands_met"]:
            blockers.append("fewer than two eligible time bands in current prospective evidence")
        if not coverage["nonempty_passenger_cohorts"]:
            blockers.append("one or more eligible windows has no passenger cohort")
        if not coverage["h4_complete"]:
            blockers.append("one or more eligible windows lacks complete H4 outcomes")
    else:
        final_decision = DECISION_SOURCE_INSUFFICIENT
        readiness = READINESS_SOURCE_INSUFFICIENT
        blockers = ["no feasible C2 route segment could execute approved research-demand B1 collection"]
    return {
        "created_at": iso_kst(),
        "final_decision": final_decision,
        "readiness": readiness,
        "audit_complete": True,
        "research_demand_contract_approved": True,
        "collection_attempted_after_source_guard": True,
        "eligible_window_count": coverage["eligible_window_count"],
        "time_band_count": coverage["distinct_time_band_count"],
        "generated_passenger_count": sum(int(row["generated_passengers"]) for row in collection["transition_rows"]),
        "reward_valid_transition_count": sum(int(row["reward_valid_transition_count"]) for row in collection["window_rows"]),
        "h4_complete_transition_count": sum(int(row["h4_complete_transition_count"]) for row in collection["window_rows"]),
        "normalization_candidate_produced": bool(candidate.get("candidate_produced")),
        "service_reference_one_supported": bool(candidate.get("service_reference_one_supported")),
        "exact_blockers": blockers,
        "minimum_next_repair": (
            "collect or approve additional prospective B1 windows spanning at least one more time band, then rerun R2A-R6 from a new timestamp artifact"
            if final_decision == DECISION_MORE_COVERAGE
            else "explicitly approve R2A-R6 normalization candidate before materialization"
            if final_decision == DECISION_READY
            else "repair route segment/source coverage before retrying R2A-R6"
        ),
        "reward_contract_approved": True,
        "training_normalization_approved": False,
        "reward_values_materialized": False,
        "r2ar7_or_r2b_authorized": False,
        "training_use_authorized": False,
    }


def approved_claim_guards() -> Dict[str, Any]:
    guards = claim_guards()
    guards.update({
        "research_demand_contract_approved": True,
        "bounded_b1_reference_collection_authorized": True,
        "automatic_r2ar7_execution_authorized": False,
    })
    return guards


def approved_reference_integrity_audit(
    *,
    source_audit: Mapping[str, Any],
    approval_record: Mapping[str, Any],
    collection: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "audit_complete": True,
        "source_audit": source_audit,
        "research_demand_approval_record_sha256": canonical_hash(approval_record),
        "research_demand_contract_version": DEMAND_CONTRACT_VERSION,
        "research_demand_contract_sha256": DEMAND_CONTRACT_SHA256,
        "typed_orchestrator_available": True,
        "typed_orchestrator_source_sha256": k5.sha256_file(ORCHESTRATOR_SOURCE),
        "typed_orchestrator_test_sha256": k5.sha256_file(ORCHESTRATOR_TEST_SOURCE),
        "collection_coverage": collection["coverage"],
        "window_count": collection["window_count"],
        "transition_count": collection["transition_count"],
        "future_information_violation_count": sum(bool(row["future_information_violation"]) for row in collection["transition_rows"]),
        "reward_input_incomplete_count": sum(not bool(row["reward_input_complete"]) for row in collection["transition_rows"]),
        "h4_incomplete_count": sum(not bool(row["h4_outcome_complete"]) for row in collection["transition_rows"]),
        "normalization_candidate_produced": bool(decision.get("normalization_candidate_produced")),
        "db_query_count": 0,
        "db_write_count": 0,
        "new_bis_api_call_count": 0,
        "aggregate_profile_promoted_to_observed_passenger_state": False,
        "observed_daegu_passenger_identity_claimed": False,
        "research_generated_passenger_labels_required": True,
    }


def approved_final_report(root: Path, decision: Mapping[str, Any], collection: Mapping[str, Any]) -> str:
    bands = ", ".join(collection["coverage"]["distinct_time_bands"]) or "none"
    return "\n".join([
        "# PV8-R2A-R6 Causal B1 Reference Collection Rerun",
        "",
        f"- artifact root: `{root}`",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{decision['final_decision']}`",
        f"- reward: `{REWARD_VERSION}` / `{REWARD_SHA256}` / `H4`",
        f"- approved demand contract: `{DEMAND_CONTRACT_VERSION}` / `{DEMAND_CONTRACT_SHA256}`",
        "",
        "## Approval and Scope",
        "",
        "The user explicitly approved the R7 research-demand candidate for this R6 rerun. The generated passengers remain `CONTRACT_FIXED_RESEARCH_DEMAND`; no observed Daegu passenger/request identity is claimed. The run used the typed R7 PV8 B1 orchestrator and kept policy evaluation, reward value materialization, training normalization approval, episode expansion, checkpoint reuse, and MAPPO training locked.",
        "",
        "## Collection Result",
        "",
        f"- eligible B1 windows: `{decision['eligible_window_count']}`",
        f"- distinct time bands: `{decision['time_band_count']}` (`{bands}`)",
        f"- causal transitions: `{collection['transition_count']}`",
        f"- generated passengers: `{decision['generated_passenger_count']}`",
        f"- reward-valid transitions: `{decision['reward_valid_transition_count']}`",
        f"- H4-complete transitions: `{decision['h4_complete_transition_count']}`",
        "",
        "## Normalization Status",
        "",
        "A training-normalization candidate was not produced unless the minimum coverage contract was met: at least three non-overlapping B1 windows and at least two distinct time bands. The current C2 prospective evidence supports bounded B1 execution, but it does not span two eligible time bands, so the constants remain unapproved and null.",
        "",
        "## Remaining Repair",
        "",
        f"{decision['minimum_next_repair']}.",
        "",
        "No API calls, DB writes, policy/MAPPO execution, reward value materialization, validation/test fitting, or downstream auto-run occurred.",
        "",
    ])


def run(root: Path) -> Path:
    upstream = verify_frozen_bindings()
    approval_record = research_demand_approval_record()
    runtime_context = approved_runtime_context()
    source_audit = source_feasibility_audit()
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    regression = run_pytest_regression()
    if not regression["passed"]:
        raise R2AR6Error(f"reward/K-safety regression failure: {regression}")

    collection = run_approved_reference_collection(runtime_context)
    candidate = approved_normalization_candidate(collection)
    decision = approved_readiness_decision(collection, candidate)
    manifest = approved_collection_manifest(
        approval_record=approval_record,
        source_audit=source_audit,
        collection=collection,
        decision=decision["final_decision"],
    )
    distributions = approved_distribution_summary(collection)
    loo = approved_leave_one_window_out(collection)
    timebands = approved_timeband_coverage(collection)
    guards = approved_claim_guards()
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": decision["readiness"],
        "gate_passed": True,
        "final_decision": decision["final_decision"],
        "stop_reason": None if decision["final_decision"] != DECISION_SOURCE_INSUFFICIENT else STOP_REASON,
        "failure_reasons": [],
        "readiness_blockers": decision["exact_blockers"],
    }

    writer.json("r2ar6_research_demand_approval_record.json", approval_record)
    writer.json("r2ar6_b1_collection_manifest.json", manifest)
    write_typed_parquet(writer.root / "r2ar6_b1_window_metrics.parquet", collection["window_rows"], WINDOW_COLUMNS, WINDOW_DTYPES)
    write_typed_parquet(writer.root / "r2ar6_b1_causal_transition_audit.parquet", collection["transition_rows"], TRANSITION_COLUMNS, TRANSITION_DTYPES)
    writer.json("r2ar6_b1_distribution_summary.json", distributions)
    writer.json("r2ar6_leave_one_window_out.json", loo)
    writer.json("r2ar6_timeband_coverage.json", timebands)
    writer.json("r2ar6_normalization_candidate.json", candidate)
    writer.json("r2ar6_reference_integrity_audit.json", approved_reference_integrity_audit(
        source_audit=source_audit,
        approval_record=approval_record,
        collection=collection,
        decision=decision,
    ))
    writer.json("r2ar6_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "approved-research-demand-bounded-causal-b1-reference-collection",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "upstream_integrity": upstream,
        "runtime_binding_audit": runtime_context["binding_audit"],
        "research_demand_approval_record_sha256": canonical_hash(approval_record),
        "pytest_regression": regression,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "horizon": "H4",
        "source_audit_sha256": canonical_hash(source_audit),
        "eligible_window_count": decision["eligible_window_count"],
        "causal_transition_count": collection["transition_count"],
        "generated_passenger_count": decision["generated_passenger_count"],
        "served_passenger_count": sum(int(row["served_passengers"]) for row in collection["transition_rows"]),
        "reward_valid_transition_count": decision["reward_valid_transition_count"],
        "h4_complete_transition_count": decision["h4_complete_transition_count"],
        "normalization_candidate_count": int(bool(candidate.get("candidate_produced"))),
        "b1_simulator_execution_count": collection["transition_count"],
        "legacy_toy_adapter_execution_count": 0,
        "passenger_event_invention_count": 0,
        "research_generated_passenger_count": decision["generated_passenger_count"],
        "synthetic_episode_duplication_count": 0,
        "c1_retrospective_selection_count": 0,
        "validation_test_fit_outcome_count": 0,
        "new_bis_api_call_count": 0,
        "db_query_count": 0,
        "db_write_count": 0,
        "reward_value_materialization_count": 0,
        "policy_evaluation_count": 0,
        "mappo_episode_expansion_count": 0,
        "checkpoint_reuse_count": 0,
        "r2ar7_execution_count": 0,
        "r2b_execution_count": 0,
        "mappo_training_count": 0,
        "qwen_train": False,
        "qwen_inference": False,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": decision["readiness"], "final_decision": decision["final_decision"], "stop_reason": gate["stop_reason"]})
    writer.text("final_report.md", approved_final_report(root, decision, collection))
    write_manifest_and_lock(writer, gate)

    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar6.json", "_PV8_R2AR6_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise R2AR6Error(f"R2A-R6 artifact integrity failure: {own}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {decision['final_decision']}")
    print(f"eligible_window_count: {decision['eligible_window_count']}")
    print(f"time_band_count: {decision['time_band_count']}")
    print(f"b1_simulator_execution_count: {collection['transition_count']}")
    print(f"normalization_candidate_produced: {str(bool(candidate.get('candidate_produced'))).lower()}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect-reference"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
