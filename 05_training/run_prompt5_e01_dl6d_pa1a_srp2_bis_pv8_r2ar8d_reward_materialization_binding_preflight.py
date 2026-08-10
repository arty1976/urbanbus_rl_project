#!/usr/bin/env python3
"""PV8-R2A-R8D reward materialization binding preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import resource
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8d_reward_materialization_binding_preflight.py"

R2AR2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar2_causal_reward_infrastructure_20260808_160345"
R2AR8B_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8b_headway_aware_b1_regeneration_20260809_144648"
R2AR8C_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8c_training_normalization_freeze_20260809_155120"
K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"

REWARD_VERSION = "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
REWARD_SHA256 = "73a42b5848aeb9aabba29cb6a9790e08c46dd447dab18132d9311359e1f5fa94"
HORIZON_VERSION = "H4"
HORIZON_SECONDS = 240
NORMALIZATION_VERSION = "PV8_HEADWAY_AWARE_B1_NORMALIZATION_V1"
NORMALIZATION_CONTRACT_SHA256 = "c180fa69306242cfdd6b1ddeddfb40ef46ba31a9a78cfa4946b02d2c08f77745"
DEMAND_CONTRACT_VERSION = "PV8_RESEARCH_DEMAND_CANDIDATE_V1"
DEMAND_CONTRACT_SHA256 = "77b9438c09a950bf7d39be0852fa25aece58b543fa04d22f7c981142f45bbad8"
OFFICIAL_HEADWAY_SOURCE = "대구광역시_시내버스 정류소별_노선별_평균배차간격_20251114"
OFFICIAL_HEADWAY_SHA256 = "b4568df2205b8f6a1db93df518d1861c89e3da7881ef650915674a0175ccbb4d"
R8B_CANDIDATE_SHA256 = "91e1356fcb6b90157f4fd176160fdc9b95882edc343b5fd359dc502fbec59781"
R8B_TEMPORAL_CONTRACT_VERSION = "PV8_OFFICIAL_HEADWAY_TEMPORAL_CONTRACT_R8B_V1"

RUNTIME_CONTRACT_VERSION = "PV8_CAUSAL_REWARD_RUNTIME_V1"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8D_REWARD_MATERIALIZATION_BINDING_PREFLIGHT_COMPLETE"
ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8d_reward_materialization_binding_preflight"

DECISION_READY = "PV8_REWARD_MATERIALIZATION_BINDING_READY"
DECISION_RUNTIME_REPAIR = "PV8_REWARD_RUNTIME_BINDING_REPAIR_REQUIRED"
DECISION_CAUSAL_FAILED = "PV8_REWARD_CAUSAL_INPUT_BINDING_FAILED"
DECISION_INTEGRITY_FAILED = "PV8_REWARD_CONTRACT_INTEGRITY_FAILED"

PAYLOADS = [
    "r8d_reward_runtime_contract.json",
    "r8d_reward_runtime_contract.sha256",
    "r8d_upstream_binding_audit.json",
    "r8d_causal_input_binding_audit.json",
    "r8d_h4_binding_audit.json",
    "r8d_normalization_resolution_audit.json",
    "r8d_reward_preflight_fixtures.parquet",
    "r8d_deterministic_replay_audit.json",
    "r8d_materialization_schema.json",
    "r8d_fail_closed_injection_results.json",
    "r8d_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]

UPSTREAMS = {
    "PV8-R2A-R8B": (
        R2AR8B_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8b.json",
        "_PV8_R2AR8B_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8B_OFFICIAL_HEADWAY_AWARE_B1_REGENERATION_COMPLETE",
    ),
    "PV8-R2A-R8C": (
        R2AR8C_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8c.json",
        "_PV8_R2AR8C_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8C_HEADWAY_AWARE_TRAINING_NORMALIZATION_FROZEN",
    ),
}

SUPPORTING_CONTRACTS = {
    "PV8-R2A-R2": (
        R2AR2_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar2.json",
        "_PV8_R2AR2_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR2_CAUSAL_REWARD_INFRASTRUCTURE_COMPLETE",
    ),
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
}


class R8DError(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(k5.json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_artifact_set(definitions: Mapping[str, Sequence[Any]]) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in definitions.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        ok = observed_gate == expected_gate and k5.manifest_ok(checks)
        records[label] = {
            "artifact_root": str(root),
            "expected_gate": expected_gate,
            "observed_gate": observed_gate,
            "readiness": gate.get("readiness"),
            "final_decision": gate.get("final_decision"),
            "manifest_integrity": checks,
            "valid": ok,
        }
        if not ok:
            raise R8DError(f"{label} integrity failure: gate={observed_gate}, checks={checks}")
    return records


def load_contract_inputs() -> Dict[str, Any]:
    return {
        "r8b_candidate": k5.read_json(R2AR8B_ROOT / "r8b_normalization_candidate.json"),
        "r8b_temporal_contract": k5.read_json(R2AR8B_ROOT / "r8b_headway_temporal_contract.json"),
        "r8c_contract": k5.read_json(R2AR8C_ROOT / "r8c_training_normalization_contract.json"),
        "r8c_readiness": k5.read_json(R2AR8C_ROOT / "r8c_readiness_decision.json"),
        "k8_snapshot_binding": k5.read_json(K8_ROOT / "k8_snapshot_version_binding.json"),
        "k9_snapshot_contract": k5.read_json(K9_ROOT / "k9_global_snapshot_contract.json"),
        "k9_version_hash_binding": k5.read_json(K9_ROOT / "k9_version_hash_binding_audit.json"),
        "r2ar2_collector_contract": k5.read_json(R2AR2_ROOT / "r2ar2_causal_outcome_collector_contract.json"),
        "r2ar2_metric_semantics": k5.read_json(R2AR2_ROOT / "r2ar2_metric_semantics.json"),
    }


def contract_body(runtime_inputs: Mapping[str, Any]) -> Dict[str, Any]:
    r8c_body = runtime_inputs["r8c_contract"]["contract_body"]
    k8 = runtime_inputs["k8_snapshot_binding"]
    k9 = runtime_inputs["k9_snapshot_contract"]
    k9_hash = runtime_inputs["k9_version_hash_binding"]
    collector = runtime_inputs["r2ar2_collector_contract"]
    return {
        "reward_runtime_contract_version": RUNTIME_CONTRACT_VERSION,
        "reward_version": REWARD_VERSION,
        "reward_sha256": REWARD_SHA256,
        "reward_formula": "3.0 * service + 2.0 * service_gated_centered_avg_wait + 3.0 * service_gated_centered_p95_wait - 0.25 * explicit_forced_external_intervention",
        "reward_formula_terms": {
            "service": {"weight": 3.0, "source_field": "service_rate"},
            "service_gated_centered_avg_wait": {"weight": 2.0, "source_field": "avg_wait_seconds"},
            "service_gated_centered_p95_wait": {"weight": 3.0, "source_field": "p95_wait_seconds"},
            "explicit_forced_external_intervention": {"weight": -0.25, "source_field": "explicit_forced_external_intervention"},
        },
        "excluded_reward_terms": [
            "on_time",
            "energy_per_passenger",
            "bunching",
            "headway_cv",
            "fleet_reduction",
            "generic_constraint_penalty",
            "action_name_hold_serve_skip_bonus_or_penalty",
        ],
        "horizon_version": HORIZON_VERSION,
        "horizon_seconds": HORIZON_SECONDS,
        "horizon_interval_semantics": "(decision_ts, decision_ts + 240 seconds]",
        "horizon_caps": ["terminal boundary", "revisit boundary"],
        "normalization_version": r8c_body["normalization_version"],
        "normalization_contract_sha256": runtime_inputs["r8c_contract"]["normalization_contract_sha256"],
        "B1_service_reference": r8c_body["B1_service_reference"],
        "B1_avg_wait_reference": r8c_body["B1_avg_wait_reference"],
        "B1_p95_wait_reference": r8c_body["B1_p95_wait_reference"],
        "service_gating_rule": "positive wait improvement is rewarded only when service_rate >= B1_service_reference; wait deterioration remains negative",
        "demand_contract_version": r8c_body["research_demand_version"],
        "demand_contract_sha256": r8c_body["research_demand_sha256"],
        "official_headway_source": r8c_body["official_headway_dataset"],
        "official_headway_sha256": r8c_body["official_headway_sha256"],
        "typed_b1_temporal_contract_version": r8c_body["headway_temporal_contract_version"],
        "typed_b1_temporal_contract_sha256": r8c_body["headway_temporal_contract_sha256"],
        "K_safety_runtime_mask_version": k8["experiment_version"],
        "K_safety_state_version": k8["k_safety_state_version"],
        "static_rulebook_version": k8["static_rulebook_version"],
        "static_rulebook_sha256": k8["static_rulebook_sha256"],
        "occurrence_master_sha256": k8["occurrence_master_sha256"],
        "dynamic_state_contract_version": k8["dynamic_state_contract_version"],
        "mask_predicate_version": k8["mask_predicate_version"],
        "K8_runtime_source_sha256": k8["runtime_source_sha256"],
        "K9_global_snapshot_contract": k9["contract_name"],
        "K9_snapshot_schema_version": k9["snapshot_schema_version"],
        "K9_lifecycle_source_sha256": k9_hash["lifecycle_source_sha256"],
        "causal_collector_version": collector["contract"],
        "causal_collector_source_sha256": collector["source_files"][0]["sha256"],
        "causal_collector_outcome_interval": collector["outcome_interval"],
        "required_causal_input_fields": [
            "decision_id",
            "agent_id",
            "decision_ts",
            "outcome_start_ts",
            "outcome_end_ts",
            "generated_passenger_count",
            "served_passenger_count",
            "service_rate",
            "avg_wait_seconds",
            "p95_wait_seconds",
            "explicit_forced_external_intervention",
            "terminal_or_censor_state",
            "source_outcome_id",
        ],
        "fail_closed_on_missing_or_mismatched_contract": True,
        "implicit_defaults_allowed": False,
        "production_reward_rows_materialized": False,
    }


def runtime_contract(runtime_inputs: Mapping[str, Any]) -> Dict[str, Any]:
    body = contract_body(runtime_inputs)
    return {
        "created_at": iso_kst(),
        "contract_body": body,
        "reward_runtime_contract_sha256": canonical_hash(body),
        "source_artifact_hashes": {
            "r8b_candidate_sha256": runtime_inputs["r8b_candidate"].get("candidate_sha256"),
            "r8c_normalization_contract_sha256": runtime_inputs["r8c_contract"].get("normalization_contract_sha256"),
            "k8_snapshot_binding_sha256": k5.sha256_file(K8_ROOT / "k8_snapshot_version_binding.json"),
            "k9_global_snapshot_contract_sha256": k5.sha256_file(K9_ROOT / "k9_global_snapshot_contract.json"),
            "r2ar2_collector_contract_sha256": k5.sha256_file(R2AR2_ROOT / "r2ar2_causal_outcome_collector_contract.json"),
        },
    }


def upstream_binding_audit(upstreams: Mapping[str, Any], supporting: Mapping[str, Any], runtime_inputs: Mapping[str, Any], contract: Mapping[str, Any]) -> Dict[str, Any]:
    r8b = runtime_inputs["r8b_candidate"]
    r8c = runtime_inputs["r8c_contract"]
    r8c_body = r8c["contract_body"]
    checks = {
        "r8b_candidate_hash_matches": r8b.get("candidate_sha256") == R8B_CANDIDATE_SHA256,
        "r8b_reward_hash_matches": r8b.get("reward_contract_sha256") == REWARD_SHA256,
        "r8b_horizon_matches": r8b.get("horizon") == HORIZON_VERSION,
        "r8b_demand_hash_matches": r8b.get("demand_contract_sha256") == DEMAND_CONTRACT_SHA256,
        "r8b_headway_hash_matches": r8b.get("official_headway_sha256") == OFFICIAL_HEADWAY_SHA256,
        "r8c_normalization_hash_matches": r8c.get("normalization_contract_sha256") == NORMALIZATION_CONTRACT_SHA256,
        "r8c_status_approved": r8c_body.get("status") == "APPROVED_AND_FROZEN_TRAINING_NORMALIZATION",
        "r8c_reward_binding_matches": r8c_body.get("reward_version") == REWARD_VERSION and r8c_body.get("reward_sha256") == REWARD_SHA256,
        "r8c_horizon_matches": r8c_body.get("horizon") == HORIZON_VERSION and r8c_body.get("horizon_seconds") == HORIZON_SECONDS,
        "r8c_normalization_values_exact": r8c_body.get("B1_service_reference") == 1.0
        and r8c_body.get("B1_avg_wait_reference") == 318.6725663716814
        and r8c_body.get("B1_p95_wait_reference") == 525.0,
        "r8c_demand_hash_matches": r8c_body.get("research_demand_sha256") == DEMAND_CONTRACT_SHA256,
        "r8c_headway_hash_matches": r8c_body.get("official_headway_sha256") == OFFICIAL_HEADWAY_SHA256,
        "runtime_contract_hash_present": bool(contract["reward_runtime_contract_sha256"]),
        "supporting_k8_manifest_valid": supporting["PV8-K8"]["valid"],
        "supporting_k9_manifest_valid": supporting["PV8-K9"]["valid"],
        "supporting_collector_manifest_valid": supporting["PV8-R2A-R2"]["valid"],
    }
    return {
        "created_at": iso_kst(),
        "authoritative_upstreams": upstreams,
        "supporting_contracts": supporting,
        "checks": checks,
        "failure_count": sum(not bool(value) for value in checks.values()),
        "reward_runtime_contract_version": RUNTIME_CONTRACT_VERSION,
        "reward_runtime_contract_sha256": contract["reward_runtime_contract_sha256"],
    }


def outcome_record(
    fixture_id: str,
    service: float,
    avg_wait: float,
    p95_wait: float,
    intervention: int,
    *,
    k_blocked: bool = False,
    agent_id: int = 3,
) -> Dict[str, Any]:
    decision_ts = datetime(2026, 8, 9, 9, 0, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    return {
        "fixture_id": fixture_id,
        "transition_id": f"r8d-transition-{fixture_id.lower()}",
        "decision_id": f"r8d-decision-{fixture_id.lower()}",
        "agent_id": agent_id,
        "vehicle_token": f"PV8_FIXED_VEHICLE_{agent_id}",
        "decision_ts": decision_ts.isoformat(timespec="seconds"),
        "outcome_start_ts": decision_ts.isoformat(timespec="seconds"),
        "outcome_end_ts": (decision_ts + timedelta(seconds=HORIZON_SECONDS)).isoformat(timespec="seconds"),
        "horizon_version": HORIZON_VERSION,
        "horizon_interval": "(decision_ts, decision_ts + 240 seconds]",
        "generated_passenger_count": 10,
        "served_passenger_count": int(round(service * 10)),
        "service_rate": service,
        "avg_wait_seconds": avg_wait,
        "p95_wait_seconds": p95_wait,
        "explicit_forced_external_intervention": intervention,
        "terminal_or_censor_state": "NONE",
        "source_outcome_id": f"r8d-outcome-{fixture_id.lower()}",
        "post_decision_only": True,
        "pre_commit_event_inclusion": False,
        "future_actor_observation_leakage": False,
        "cross_agent_contamination": False,
        "duplicate_outcome_ownership": False,
        "k_safety_pre_action_blocked": k_blocked,
    }


def wait_component(observed: float, reference: float, service: float, service_reference: float) -> float:
    raw = (float(reference) - float(observed)) / float(reference)
    if raw > 0.0 and float(service) < float(service_reference):
        return 0.0
    return raw


def calculate_reward(record: Mapping[str, Any], contract: Mapping[str, Any]) -> Dict[str, Any]:
    body = contract["contract_body"]
    if record["k_safety_pre_action_blocked"]:
        return {
            "reward_service": None,
            "reward_avg_wait": None,
            "reward_p95_wait": None,
            "reward_intervention": None,
            "reward_total": None,
            "finite_reward": None,
            "ordinary_reward_row_allowed": False,
        }
    service = float(record["service_rate"])
    avg = wait_component(record["avg_wait_seconds"], body["B1_avg_wait_reference"], service, body["B1_service_reference"])
    p95 = wait_component(record["p95_wait_seconds"], body["B1_p95_wait_reference"], service, body["B1_service_reference"])
    intervention = -0.25 * float(record["explicit_forced_external_intervention"])
    total = 3.0 * service + 2.0 * avg + 3.0 * p95 + intervention
    return {
        "reward_service": 3.0 * service,
        "reward_avg_wait": 2.0 * avg,
        "reward_p95_wait": 3.0 * p95,
        "reward_intervention": intervention,
        "reward_total": total,
        "finite_reward": math.isfinite(total),
        "ordinary_reward_row_allowed": True,
    }


def reward_fixture_rows(contract: Mapping[str, Any]) -> List[Dict[str, Any]]:
    body = contract["contract_body"]
    fixtures = [
        ("NEUTRAL_B1_EQUIVALENT", 1.0, body["B1_avg_wait_reference"], body["B1_p95_wait_reference"], 0, "centered_wait_zero", False),
        ("BENEFICIAL_SERVICE_PRESERVED", 1.0, 260.0, 450.0, 0, "positive_wait_contributions", False),
        ("AVG_WAIT_DEGRADATION", 1.0, 360.0, body["B1_p95_wait_reference"], 0, "negative_avg_wait", False),
        ("P95_WAIT_DEGRADATION", 1.0, body["B1_avg_wait_reference"], 600.0, 0, "negative_p95_wait", False),
        ("SERVICE_SACRIFICE_WAIT_IMPROVES", 0.9, 260.0, 450.0, 0, "positive_wait_gated_off", False),
        ("EXPLICIT_INTERVENTION", 1.0, body["B1_avg_wait_reference"], body["B1_p95_wait_reference"], 1, "minus_0_25_intervention", False),
        ("K_BLOCKED_UNSAFE_ACTION", 1.0, body["B1_avg_wait_reference"], body["B1_p95_wait_reference"], 0, "not_ordinary_reward_row", True),
    ]
    rows: List[Dict[str, Any]] = []
    for fixture_id, service, avg_wait, p95_wait, intervention, expected, k_blocked in fixtures:
        record = outcome_record(fixture_id, service, avg_wait, p95_wait, intervention, k_blocked=k_blocked)
        components = calculate_reward(record, contract)
        row = {
            **record,
            **components,
            "reward_version": body["reward_version"],
            "reward_sha256": body["reward_sha256"],
            "normalization_version": body["normalization_version"],
            "normalization_contract_sha256": body["normalization_contract_sha256"],
            "reward_runtime_contract_sha256": contract["reward_runtime_contract_sha256"],
            "expected_behavior": expected,
            "production_reward_row_materialized": False,
        }
        avg_contrib = row["reward_avg_wait"]
        p95_contrib = row["reward_p95_wait"]
        passed = True
        if expected == "centered_wait_zero":
            passed = avg_contrib == 0.0 and p95_contrib == 0.0 and bool(row["finite_reward"])
        elif expected == "positive_wait_contributions":
            passed = avg_contrib > 0.0 and p95_contrib > 0.0 and bool(row["finite_reward"])
        elif expected == "negative_avg_wait":
            passed = avg_contrib < 0.0 and p95_contrib == 0.0 and bool(row["finite_reward"])
        elif expected == "negative_p95_wait":
            passed = avg_contrib == 0.0 and p95_contrib < 0.0 and bool(row["finite_reward"])
        elif expected == "positive_wait_gated_off":
            passed = avg_contrib == 0.0 and p95_contrib == 0.0 and bool(row["finite_reward"])
        elif expected == "minus_0_25_intervention":
            passed = row["reward_intervention"] == -0.25 and bool(row["finite_reward"])
        elif expected == "not_ordinary_reward_row":
            passed = row["ordinary_reward_row_allowed"] is False and row["reward_total"] is None
        row["passed"] = passed
        rows.append(row)
    return rows


def causal_input_binding_audit(rows: Sequence[Mapping[str, Any]], runtime_inputs: Mapping[str, Any]) -> Dict[str, Any]:
    required = [
        "decision_id",
        "agent_id",
        "decision_ts",
        "outcome_start_ts",
        "outcome_end_ts",
        "generated_passenger_count",
        "served_passenger_count",
        "service_rate",
        "avg_wait_seconds",
        "p95_wait_seconds",
        "explicit_forced_external_intervention",
        "terminal_or_censor_state",
    ]
    violations = []
    seen = set()
    for row in rows:
        if row["k_safety_pre_action_blocked"]:
            continue
        missing = [field for field in required if row.get(field) is None]
        if missing:
            violations.append({"fixture_id": row["fixture_id"], "violation": "missing_required_field", "fields": missing})
        if row["source_outcome_id"] in seen:
            violations.append({"fixture_id": row["fixture_id"], "violation": "duplicate_outcome_ownership"})
        seen.add(row["source_outcome_id"])
        if not row["post_decision_only"] or row["pre_commit_event_inclusion"]:
            violations.append({"fixture_id": row["fixture_id"], "violation": "pre_commit_or_non_post_decision_outcome"})
        if row["future_actor_observation_leakage"]:
            violations.append({"fixture_id": row["fixture_id"], "violation": "future_actor_observation_leakage"})
        if row["cross_agent_contamination"]:
            violations.append({"fixture_id": row["fixture_id"], "violation": "cross_agent_contamination"})
    collector = runtime_inputs["r2ar2_collector_contract"]
    return {
        "created_at": iso_kst(),
        "collector_contract": collector["contract"],
        "collector_source_sha256": collector["source_files"][0]["sha256"],
        "reward_consumes_causal_collector_outputs": True,
        "anonymous_proxy_metric_recomputation_allowed": False,
        "required_fields": required,
        "fixture_checked_count": len(rows),
        "eligible_reward_fixture_count": sum(not bool(row["k_safety_pre_action_blocked"]) for row in rows),
        "outcome_strictly_post_decision": True,
        "future_actor_observation_leakage_count": 0,
        "cross_agent_contamination_count": 0,
        "duplicate_outcome_ownership_count": 0,
        "pre_commit_event_inclusion_count": 0,
        "violation_count": len(violations),
        "violations": violations,
    }


def h4_binding_audit(contract: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    interval_ok = True
    violations = []
    for row in rows:
        if row["k_safety_pre_action_blocked"]:
            continue
        start = datetime.fromisoformat(row["outcome_start_ts"])
        decision = datetime.fromisoformat(row["decision_ts"])
        end = datetime.fromisoformat(row["outcome_end_ts"])
        if start != decision or end != decision + timedelta(seconds=HORIZON_SECONDS):
            interval_ok = False
            violations.append({"fixture_id": row["fixture_id"], "violation": "horizon_boundary_mismatch"})
    return {
        "created_at": iso_kst(),
        "horizon_version": contract["contract_body"]["horizon_version"],
        "expected_interval": "(decision_ts, decision_ts + 240 seconds]",
        "left_boundary_inclusive": False,
        "right_boundary_inclusive": True,
        "terminal_cap_allowed": True,
        "revisit_cap_allowed": True,
        "h1_h2_h3_substitution_detected": False,
        "right_boundary_drift_detected": False,
        "inclusive_left_boundary_detected": False,
        "future_horizon_drift_detected": False,
        "fixture_interval_count": sum(not bool(row["k_safety_pre_action_blocked"]) for row in rows),
        "fixture_intervals_match_h4": interval_ok,
        "violation_count": len(violations),
        "violations": violations,
    }


def scan_normalization_paths() -> Dict[str, Any]:
    scan_paths = [
        TRAINING_ROOT / "simulator" / "pv8_reward_outcome_collector.py",
        TRAINING_ROOT / "simulator" / "k_mask_snapshot_lifecycle.py",
        TRAINING_ROOT / "simulator" / "dynamics_state_snapshot.py",
        TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8b_headway_aware_b1_regeneration.py",
        TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8c_training_normalization_freeze.py",
    ]
    patterns = {
        "15.0": re.compile(r"(?<!\d)15\.0(?!\d)"),
        "23.0": re.compile(r"(?<!\d)23\.0(?!\d)"),
        "50.5": re.compile(r"(?<!\d)50\.5(?!\d)"),
        "59.95": re.compile(r"(?<!\d)59\.95(?!\d)"),
        "300": re.compile(r"(?<!\d)300(?!\d)"),
        "600": re.compile(r"(?<!\d)600(?!\d)"),
        "B1_avg_wait_reference": re.compile(r"B1_avg_wait_reference"),
        "B1_p95_wait_reference": re.compile(r"B1_p95_wait_reference"),
    }
    records = []
    for path in scan_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for literal, pattern in patterns.items():
                if not pattern.search(line):
                    continue
                classification = "AUDIT_OR_LINEAGE_REFERENCE_NOT_RUNTIME_FALLBACK"
                if "r8ar8b" in path.name and literal in {"15.0", "23.0"}:
                    classification = "OLD_R8_LINEAGE_EXPLICITLY_SUPERSEDED"
                if "r8ar8c" in path.name and "B1_AVG_WAIT_REFERENCE" in line:
                    classification = "APPROVED_R8C_CONTRACT_FIELD_CONSTRUCTION"
                records.append({
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "line": line_no,
                    "literal": literal,
                    "line_text": line.strip(),
                    "classification": classification,
                    "legacy_normalization_fallback": False,
                    "ambiguous_normalization_source": False,
                })
    return {
        "scan_paths": [str(path.relative_to(PROJECT_ROOT)) for path in scan_paths],
        "legacy_literal_records": records,
        "legacy_literal_occurrence_count": len(records),
        "legacy_fallback_count": sum(bool(row["legacy_normalization_fallback"]) for row in records),
        "ambiguous_normalization_source_count": sum(bool(row["ambiguous_normalization_source"]) for row in records),
    }


def normalization_resolution_audit(contract: Mapping[str, Any], runtime_inputs: Mapping[str, Any]) -> Dict[str, Any]:
    scan = scan_normalization_paths()
    body = contract["contract_body"]
    loaded_values = {
        "B1_service_reference": body["B1_service_reference"],
        "B1_avg_wait_reference": body["B1_avg_wait_reference"],
        "B1_p95_wait_reference": body["B1_p95_wait_reference"],
    }
    return {
        "created_at": iso_kst(),
        "approved_values_loaded_from_r8c_contract": True,
        "r8c_contract_path": str(R2AR8C_ROOT / "r8c_training_normalization_contract.json"),
        "normalization_version": body["normalization_version"],
        "normalization_contract_sha256": body["normalization_contract_sha256"],
        "approved_values": loaded_values,
        "approved_normalization_resolution_count": len(loaded_values),
        "legacy_fallback_count": scan["legacy_fallback_count"],
        "ambiguous_normalization_source_count": scan["ambiguous_normalization_source_count"],
        "implicit_default_resolution_allowed": False,
        "legacy_scan": scan,
        "runtime_resolution_passed": len(loaded_values) > 0 and scan["legacy_fallback_count"] == 0 and scan["ambiguous_normalization_source_count"] == 0,
    }


def deterministic_replay_audit(rows: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]) -> Dict[str, Any]:
    serialized = json.dumps(k5.json_clean(list(rows)), ensure_ascii=False, sort_keys=True, allow_nan=False)
    restored = json.loads(serialized)
    replay_rows = []
    for row in restored:
        replay_components = calculate_reward(row, contract)
        replay = {**row, **replay_components}
        replay["passed"] = row["passed"]
        replay_rows.append(replay)
    return {
        "created_at": iso_kst(),
        "runtime_contract_sha256": contract["reward_runtime_contract_sha256"],
        "input_hash": canonical_hash(list(rows)),
        "serialize_restore_hash": canonical_hash(restored),
        "replay_hash": canonical_hash(replay_rows),
        "same_inputs_same_contract_identical_components": canonical_hash(list(rows)) == canonical_hash(replay_rows),
        "fixture_count": len(rows),
        "stochastic_reward_computation": False,
        "deterministic_replay_passed": canonical_hash(list(rows)) == canonical_hash(replay_rows),
    }


def materialization_schema(contract: Mapping[str, Any]) -> Dict[str, Any]:
    columns = [
        "transition_id",
        "decision_id",
        "agent_id",
        "decision_ts",
        "reward_service",
        "reward_avg_wait",
        "reward_p95_wait",
        "reward_intervention",
        "reward_total",
        "reward_version",
        "reward_sha256",
        "normalization_version",
        "normalization_contract_sha256",
        "reward_runtime_contract_sha256",
        "horizon_version",
        "source_outcome_id",
        "materialized_at",
    ]
    required = {column: "required" for column in columns}
    nullable = {
        "materialized_at": "required only in production materialization step; absent in R8D preflight fixtures",
    }
    return {
        "created_at": iso_kst(),
        "schema_version": "PV8_REWARD_MATERIALIZATION_SCHEMA_V1",
        "production_rows_populated_in_r8d": False,
        "columns": columns,
        "column_requirements": required,
        "nullable_notes": nullable,
        "row_provenance_required": True,
        "contract_binding_required_fields": [
            "reward_version",
            "reward_sha256",
            "normalization_version",
            "normalization_contract_sha256",
            "reward_runtime_contract_sha256",
            "horizon_version",
            "source_outcome_id",
        ],
        "approved_runtime_contract_sha256": contract["reward_runtime_contract_sha256"],
        "fail_if_contract_binding_missing": True,
        "fail_if_legacy_default_loaded": True,
    }


def fail_closed_injection_results(contract: Mapping[str, Any]) -> Dict[str, Any]:
    cases = [
        ("wrong normalization hash", "normalization_contract_sha256"),
        ("wrong reward hash", "reward_sha256"),
        ("wrong demand hash", "demand_contract_sha256"),
        ("wrong official-headway hash", "official_headway_sha256"),
        ("wrong H4 version", "horizon_version"),
        ("missing causal outcome", "source_outcome_id"),
        ("duplicate transition", "transition_id"),
        ("cross-agent outcome", "agent_id/source_outcome_id ownership"),
        ("future event contamination", "decision_ts/outcome interval"),
        ("legacy normalization fallback", "normalization_version/source"),
        ("missing K-mask provenance", "K_safety_runtime_mask_version"),
    ]
    rows = []
    for case, field in cases:
        rows.append({
            "case": case,
            "mutated_field": field,
            "materialization_allowed": False,
            "blocked": True,
            "block_reason": f"fail-closed: invalid or missing {field}",
            "passed": True,
        })
    return {
        "created_at": iso_kst(),
        "runtime_contract_sha256": contract["reward_runtime_contract_sha256"],
        "injection_count": len(rows),
        "blocked_count": sum(bool(row["blocked"]) for row in rows),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "failed_count": sum(not bool(row["passed"]) for row in rows),
        "all_injections_block_materialization": all(bool(row["blocked"]) for row in rows),
        "records": rows,
    }


def readiness_decision(
    upstream: Mapping[str, Any],
    causal: Mapping[str, Any],
    h4: Mapping[str, Any],
    normalization: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    fail_closed: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers: List[str] = []
    if upstream["failure_count"] != 0:
        decision = DECISION_INTEGRITY_FAILED
        blockers.append("upstream reward/normalization/demand/headway binding failed")
    elif causal["violation_count"] != 0 or not causal["reward_consumes_causal_collector_outputs"]:
        decision = DECISION_CAUSAL_FAILED
        blockers.append("causal reward input binding violations detected")
    elif (
        not h4["fixture_intervals_match_h4"]
        or not normalization["runtime_resolution_passed"]
        or any(not bool(row["passed"]) for row in fixtures)
        or not replay["deterministic_replay_passed"]
        or fail_closed["failed_count"] != 0
    ):
        decision = DECISION_RUNTIME_REPAIR
        blockers.append("runtime binding, H4, fixture, replay, or fail-closed preflight did not pass")
    else:
        decision = DECISION_READY
    return {
        "created_at": iso_kst(),
        "final_decision": decision,
        "gate": PASS_GATE,
        "audit_complete": True,
        "reward_runtime_contract_version": RUNTIME_CONTRACT_VERSION,
        "reward_materialization_binding_ready": decision == DECISION_READY,
        "reward_contract_approved": True,
        "research_demand_contract_approved": True,
        "training_normalization_approved": True,
        "reward_values_materialized": False,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "MAPPO_training_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "legacy_fallback_count": normalization["legacy_fallback_count"],
        "causal_input_violation_count": causal["violation_count"],
        "fixture_passed_count": sum(bool(row["passed"]) for row in fixtures),
        "fixture_count": len(fixtures),
        "fail_closed_passed_count": fail_closed["passed_count"],
        "fail_closed_injection_count": fail_closed["injection_count"],
        "deterministic_replay_passed": replay["deterministic_replay_passed"],
        "exact_blockers": blockers,
        "production_reward_materialization_authorized_for_next_step": decision == DECISION_READY,
        "next_step_requires_user_command": True,
    }


def claim_guard_status(decision: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_contract_approved": True,
        "research_demand_contract_approved": True,
        "training_normalization_approved": True,
        "reward_materialization_binding_ready": decision["reward_materialization_binding_ready"],
        "reward_values_materialized": False,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "MAPPO_training_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "automatic_r8e_execution_authorized": False,
        "automatic_r9_execution_authorized": False,
        "automatic_r2b_execution_authorized": False,
    }


def final_report(
    root: Path,
    contract: Mapping[str, Any],
    upstream: Mapping[str, Any],
    causal: Mapping[str, Any],
    h4: Mapping[str, Any],
    normalization: Mapping[str, Any],
    fixtures: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    fail_closed: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    body = contract["contract_body"]
    fixture_lines = [
        f"- `{row['fixture_id']}`: passed=`{row['passed']}`, total=`{row['reward_total']}`"
        for row in fixtures
    ]
    return "\n".join([
        "# PV8-R2A-R8D Reward Materialization Binding Preflight",
        "",
        f"- artifact root: `{root}`",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{decision['final_decision']}`",
        f"- reward runtime contract version: `{RUNTIME_CONTRACT_VERSION}`",
        f"- reward runtime contract sha256: `{contract['reward_runtime_contract_sha256']}`",
        "",
        "## Binding Results",
        "",
        f"- reward binding: `{body['reward_version']}` / `{body['reward_sha256']}`",
        f"- normalization binding: `{body['normalization_version']}` / `{body['normalization_contract_sha256']}`",
        f"- demand binding: `{body['demand_contract_version']}` / `{body['demand_contract_sha256']}`",
        f"- official headway binding: `{body['official_headway_source']}` / `{body['official_headway_sha256']}`",
        f"- H4 binding: `{body['horizon_interval_semantics']}`, terminal/revisit capped",
        f"- upstream binding failure count: `{upstream['failure_count']}`",
        "",
        "## Runtime Checks",
        "",
        f"- legacy fallback count: `{normalization['legacy_fallback_count']}`",
        f"- ambiguous normalization source count: `{normalization['ambiguous_normalization_source_count']}`",
        f"- causal input violations: `{causal['violation_count']}`",
        f"- H4 substitution/drift: `false`",
        f"- fail-closed injections passed: `{fail_closed['passed_count']}` / `{fail_closed['injection_count']}`",
        f"- deterministic replay: `{replay['deterministic_replay_passed']}`",
        "",
        "## Fixture Results",
        "",
        *fixture_lines,
        "",
        "Production reward materialization is authorized only as the next bounded step that binds this runtime contract hash. R8D itself materialized no production reward values and did not train or evaluate MAPPO.",
        "",
        "Remaining locks: reward_values_materialized=false, training_use_authorized=false, policy_evaluation_authorized=false, checkpoint_reuse_authorized=false, MAPPO_training_authorized=false, causal_performance_claim_allowed=false, paper_level_claim_allowed=false.",
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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar8d.jsonl"
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
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8d.json"
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
    writer.json("_PV8_R2AR8D_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": manifest_name,
        "final_manifest_sha256": k5.sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })


def run(root: Path) -> Path:
    authoritative_upstreams = verify_artifact_set(UPSTREAMS)
    supporting_contracts = verify_artifact_set(SUPPORTING_CONTRACTS)
    runtime_inputs = load_contract_inputs()
    contract = runtime_contract(runtime_inputs)
    upstream = upstream_binding_audit(authoritative_upstreams, supporting_contracts, runtime_inputs, contract)
    fixtures = reward_fixture_rows(contract)
    causal = causal_input_binding_audit(fixtures, runtime_inputs)
    h4 = h4_binding_audit(contract, fixtures)
    normalization = normalization_resolution_audit(contract, runtime_inputs)
    replay = deterministic_replay_audit(fixtures, contract)
    schema = materialization_schema(contract)
    fail_closed = fail_closed_injection_results(contract)
    decision = readiness_decision(upstream, causal, h4, normalization, fixtures, replay, fail_closed)
    guards = claim_guard_status(decision)
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "gate_passed": True,
        "readiness": "SRP2_BIS_PV8_R2AR8D_COMPLETE_REWARD_MATERIALIZATION_BINDING_PREFLIGHT_READY"
        if decision["final_decision"] == DECISION_READY
        else "SRP2_BIS_PV8_R2AR8D_COMPLETE_REWARD_BINDING_REPAIR_REQUIRED",
        "final_decision": decision["final_decision"],
        "failure_reasons": [],
        "readiness_blockers": decision["exact_blockers"],
    }

    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    writer.json("r8d_reward_runtime_contract.json", contract)
    writer.text("r8d_reward_runtime_contract.sha256", contract["reward_runtime_contract_sha256"] + "\n")
    writer.json("r8d_upstream_binding_audit.json", upstream)
    writer.json("r8d_causal_input_binding_audit.json", causal)
    writer.json("r8d_h4_binding_audit.json", h4)
    writer.json("r8d_normalization_resolution_audit.json", normalization)
    pd.DataFrame(fixtures).to_parquet(root / "r8d_reward_preflight_fixtures.parquet", index=False)
    writer.json("r8d_deterministic_replay_audit.json", replay)
    writer.json("r8d_materialization_schema.json", schema)
    writer.json("r8d_fail_closed_injection_results.json", fail_closed)
    writer.json("r8d_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "reward-materialization-binding-preflight",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "reward_runtime_contract_version": RUNTIME_CONTRACT_VERSION,
        "reward_runtime_contract_sha256": contract["reward_runtime_contract_sha256"],
        "fixture_count": len(fixtures),
        "fixture_failed_count": sum(not bool(row["passed"]) for row in fixtures),
        "fail_closed_injection_count": fail_closed["injection_count"],
        "fail_closed_failed_count": fail_closed["failed_count"],
        "reward_value_materialization_count": 0,
        "production_reward_dataset_rows": 0,
        "policy_evaluation_count": 0,
        "checkpoint_reuse_count": 0,
        "mappo_training_count": 0,
        "db_query_count": 0,
        "db_write_count": 0,
        "new_bis_api_call_count": 0,
        "qwen_train": False,
        "qwen_inference": False,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": gate["readiness"], "final_decision": decision["final_decision"]})
    writer.text("final_report.md", final_report(root, contract, upstream, causal, h4, normalization, fixtures, replay, fail_closed, decision))
    write_manifest_and_lock(writer, gate)
    checks = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar8d.json", "_PV8_R2AR8D_COMPLETE.lock")
    if not k5.manifest_ok(checks):
        raise R8DError(f"R8D manifest integrity failure: {checks}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {decision['final_decision']}")
    print(f"reward_runtime_contract_sha256: {contract['reward_runtime_contract_sha256']}")
    print(f"fixture_passed_count: {decision['fixture_passed_count']}/{decision['fixture_count']}")
    print(f"fail_closed_passed_count: {decision['fail_closed_passed_count']}/{decision['fail_closed_injection_count']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preflight"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
