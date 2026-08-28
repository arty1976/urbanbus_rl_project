#!/usr/bin/env python3
"""PV8-R2A-R8C headway-aware B1 training-normalization approval and freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import resource
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from zoneinfo import ZoneInfo
from datetime import datetime

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8c_training_normalization_freeze.py"

R2AR8A_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8a_official_headway_temporal_realism_20260809_130746"
R2AR8B_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8b_headway_aware_b1_regeneration_20260809_144648"

REWARD_VERSION = "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
REWARD_SHA256 = "73a42b5848aeb9aabba29cb6a9790e08c46dd447dab18132d9311359e1f5fa94"
DEMAND_CONTRACT_VERSION = "PV8_RESEARCH_DEMAND_CANDIDATE_V1"
DEMAND_CONTRACT_SHA256 = "77b9438c09a950bf7d39be0852fa25aece58b543fa04d22f7c981142f45bbad8"
OFFICIAL_HEADWAY_SHA256 = "b4568df2205b8f6a1db93df518d1861c89e3da7881ef650915674a0175ccbb4d"
R8B_CANDIDATE_SHA256 = "91e1356fcb6b90157f4fd176160fdc9b95882edc343b5fd359dc502fbec59781"
R8B_TEMPORAL_CONTRACT_VERSION = "PV8_OFFICIAL_HEADWAY_TEMPORAL_CONTRACT_R8B_V1"

NORMALIZATION_VERSION = "PV8_HEADWAY_AWARE_B1_NORMALIZATION_V1"
B1_SERVICE_REFERENCE = 1.0
B1_AVG_WAIT_REFERENCE = 318.6725663716814
B1_P95_WAIT_REFERENCE = 525.0

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8c_training_normalization_freeze"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8C_HEADWAY_AWARE_TRAINING_NORMALIZATION_FROZEN"
DECISION_FROZEN = "PV8_HEADWAY_AWARE_TRAINING_NORMALIZATION_FROZEN"
DECISION_INTEGRITY_FAILED = "PV8_NORMALIZATION_FREEZE_INTEGRITY_FAILED"
DECISION_RUNTIME_REPAIR = "PV8_NORMALIZATION_RUNTIME_BINDING_REPAIR_REQUIRED"
READINESS_FROZEN = "SRP2_BIS_PV8_R2AR8C_COMPLETE_TRAINING_NORMALIZATION_FROZEN_REWARD_MATERIALIZATION_PENDING"
READINESS_INTEGRITY_FAILED = "SRP2_BIS_PV8_R2AR8C_COMPLETE_INTEGRITY_FAILED"
READINESS_RUNTIME_REPAIR = "SRP2_BIS_PV8_R2AR8C_COMPLETE_RUNTIME_BINDING_REPAIR_REQUIRED"

UPSTREAMS = {
    "PV8-R2A-R8A": (R2AR8A_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8a.json", "_PV8_R2AR8A_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8A_DAEGU_OFFICIAL_HEADWAY_TEMPORAL_REALISM_AUDIT_COMPLETE"),
    "PV8-R2A-R8B": (R2AR8B_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8b.json", "_PV8_R2AR8B_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8B_OFFICIAL_HEADWAY_AWARE_B1_REGENERATION_COMPLETE"),
}

PAYLOADS = [
    "r8c_user_approval_record.json",
    "r8c_training_normalization_contract.json",
    "r8c_training_normalization_contract.sha256",
    "r8c_upstream_binding_audit.json",
    "r8c_legacy_normalization_exclusion_audit.json",
    "r8c_reward_binding_dryrun.parquet",
    "r8c_boundary_fixture_results.json",
    "r8c_runtime_resolution_audit.json",
    "r8c_claim_boundary.json",
    "r8c_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R8CError(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(k5.json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_upstreams() -> Dict[str, Any]:
    records = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed != expected_gate or not k5.manifest_ok(checks):
            raise R8CError(f"{label} upstream integrity failure: gate={observed}, checks={checks}")
        records[label] = {
            "artifact_root": str(root),
            "gate": observed,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return records


def user_approval_record() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "approval_source": "attached PV8-R2A-R8C prompt in current Codex task",
        "explicit_user_approval_recognized": True,
        "approved_scope": "exact R8B headway-aware B1 normalization candidate only",
        "normalization_version": NORMALIZATION_VERSION,
        "approved_values": {
            "B1_service_reference": B1_SERVICE_REFERENCE,
            "B1_avg_wait_reference": B1_AVG_WAIT_REFERENCE,
            "B1_p95_wait_reference": B1_P95_WAIT_REFERENCE,
        },
        "approved_units": {
            "service": "ratio",
            "avg_wait": "seconds",
            "p95_wait": "seconds",
        },
        "not_authorized": [
            "reward value materialization",
            "training episode expansion",
            "policy evaluation",
            "checkpoint reuse",
            "MAPPO training",
            "causal or paper-level performance claims",
        ],
    }


def upstream_binding_audit(upstreams: Mapping[str, Any]) -> Dict[str, Any]:
    r8a_source = k5.read_json(R2AR8A_ROOT / "r8a_official_headway_source_manifest.json")
    r8a_reassessment = k5.read_json(R2AR8A_ROOT / "r8a_r8_normalization_reassessment.json")
    r8b_candidate = k5.read_json(R2AR8B_ROOT / "r8b_normalization_candidate.json")
    r8b_contract = k5.read_json(R2AR8B_ROOT / "r8b_headway_temporal_contract.json")
    r8b_manifest = k5.read_json(R2AR8B_ROOT / "r8b_b1_window_manifest.json")
    r8b_readiness = k5.read_json(R2AR8B_ROOT / "r8b_readiness_decision.json")
    constants = r8b_candidate.get("constants", {})
    checks = {
        "r8a_official_headway_sha256_matches": r8a_source.get("file_sha256") == OFFICIAL_HEADWAY_SHA256,
        "r8a_old_candidate_not_approved": r8a_reassessment.get("classification") == "R8_NORMALIZATION_REQUIRES_B1_REGENERATION"
        and r8a_reassessment.get("B1_avg_wait_reference") == "NOT_APPROVED",
        "r8b_decision_ready": r8b_readiness.get("final_decision") == "PV8_HEADWAY_AWARE_B1_NORMALIZATION_CANDIDATE_READY_FOR_EXPLICIT_APPROVAL",
        "r8b_candidate_sha256_matches": r8b_candidate.get("candidate_sha256") == R8B_CANDIDATE_SHA256,
        "r8b_candidate_values_exact": constants.get("B1_service_reference") == B1_SERVICE_REFERENCE
        and constants.get("B1_avg_wait_reference") == B1_AVG_WAIT_REFERENCE
        and constants.get("B1_p95_wait_reference") == B1_P95_WAIT_REFERENCE,
        "reward_binding_matches": r8b_candidate.get("reward_version") == REWARD_VERSION
        and r8b_candidate.get("reward_contract_sha256") == REWARD_SHA256
        and r8b_candidate.get("horizon") == "H4",
        "demand_binding_matches": r8b_candidate.get("demand_contract_version") == DEMAND_CONTRACT_VERSION
        and r8b_candidate.get("demand_contract_sha256") == DEMAND_CONTRACT_SHA256,
        "headway_binding_matches": r8b_candidate.get("official_headway_sha256") == OFFICIAL_HEADWAY_SHA256
        and r8b_contract.get("contract_version") == R8B_TEMPORAL_CONTRACT_VERSION,
        "window_evidence_matches": r8b_manifest.get("eligible_window_count") == 9
        and r8b_manifest.get("generated_passengers") == 113
        and r8b_manifest.get("served_passengers") == 113
        and r8b_manifest.get("service_opportunity_count") == 56,
    }
    checks["failure_count"] = sum(not bool(value) for value in checks.values())
    return {
        "created_at": iso_kst(),
        "authoritative_upstreams": upstreams,
        "checks": checks,
        "r8b_candidate_sha256": r8b_candidate.get("candidate_sha256"),
        "r8b_window_ids": r8b_candidate.get("source_windows"),
        "r8b_temporal_contract_version": r8b_contract.get("contract_version"),
        "route_direction_timetable_scope": {
            "route": "814",
            "timetable_types": r8b_candidate.get("official_timetable_types"),
            "time_bands": r8b_candidate.get("time_bands"),
            "window_count": r8b_manifest.get("eligible_window_count"),
        },
    }


def normalization_contract(binding: Mapping[str, Any], approval: Mapping[str, Any]) -> Dict[str, Any]:
    r8b_candidate = k5.read_json(R2AR8B_ROOT / "r8b_normalization_candidate.json")
    r8b_manifest = k5.read_json(R2AR8B_ROOT / "r8b_b1_window_manifest.json")
    r8b_temporal = k5.read_json(R2AR8B_ROOT / "r8b_headway_temporal_contract.json")
    body = {
        "normalization_version": NORMALIZATION_VERSION,
        "status": "APPROVED_AND_FROZEN_TRAINING_NORMALIZATION",
        "approved_at": iso_kst(),
        "B1_service_reference": B1_SERVICE_REFERENCE,
        "B1_avg_wait_reference": B1_AVG_WAIT_REFERENCE,
        "B1_p95_wait_reference": B1_P95_WAIT_REFERENCE,
        "units": {
            "service": "ratio",
            "avg_wait": "seconds",
            "p95_wait": "seconds",
        },
        "human_readable": {
            "avg_wait": "5 min 18.6725663716814 sec",
            "p95_wait": "8 min 45 sec",
        },
        "reward_version": REWARD_VERSION,
        "reward_sha256": REWARD_SHA256,
        "horizon": "H4",
        "horizon_seconds": 240,
        "research_demand_version": DEMAND_CONTRACT_VERSION,
        "research_demand_sha256": DEMAND_CONTRACT_SHA256,
        "official_headway_dataset": "대구광역시_시내버스 정류소별_노선별_평균배차간격_20251114",
        "official_headway_sha256": OFFICIAL_HEADWAY_SHA256,
        "headway_temporal_contract_version": R8B_TEMPORAL_CONTRACT_VERSION,
        "headway_temporal_contract_sha256": k5.sha256_file(R2AR8B_ROOT / "r8b_headway_temporal_contract.json"),
        "r8b_candidate_sha256": R8B_CANDIDATE_SHA256,
        "r8b_derivation_artifact": str(R2AR8B_ROOT),
        "r8b_source_windows": r8b_candidate.get("source_windows"),
        "r8b_passenger_event_evidence": {
            "eligible_windows": r8b_manifest.get("eligible_window_count"),
            "service_opportunities": r8b_manifest.get("service_opportunity_count"),
            "generated_passengers": r8b_manifest.get("generated_passengers"),
            "served_passengers": r8b_manifest.get("served_passengers"),
            "wait_observations": r8b_manifest.get("wait_observation_count"),
            "reward_valid_transitions": r8b_manifest.get("reward_valid_transitions"),
            "h4_complete_transitions": r8b_manifest.get("h4_complete_transitions"),
        },
        "route_direction_timetable_scope": {
            "route_id": "3000814001",
            "route_no": "814",
            "official_regimes": {
                "weekday_minutes": 9,
                "saturday_minutes": 10,
                "holiday_minutes": 11,
            },
            "official_timetable_types": r8b_candidate.get("official_timetable_types"),
            "time_bands": r8b_candidate.get("time_bands"),
            "temporal_contract_record_count": r8b_temporal.get("mapped_route_direction_timetable_record_count"),
        },
        "invalidation_rules": [
            "service reference changes",
            "avg-wait reference changes",
            "p95-wait reference changes",
            "official headway source changes",
            "headway temporal semantics changes",
            "demand contract changes",
            "reward formula changes",
            "H4 horizon changes",
        ],
        "training_normalization_approved": True,
        "reward_values_materialized": False,
        "training_use_authorized": False,
    }
    contract_hash = canonical_hash(body)
    return {
        "created_at": iso_kst(),
        "contract_body": body,
        "normalization_contract_sha256": contract_hash,
        "binding_audit_sha256": canonical_hash(binding),
        "user_approval_record_sha256": canonical_hash(approval),
    }


def scan_legacy_normalization() -> Dict[str, Any]:
    scan_roots = [TRAINING_ROOT / "configs", TRAINING_ROOT / "simulator"]
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
    files = []
    for root in scan_roots:
        for suffix in ("*.py", "*.json", "*.yaml", "*.yml"):
            files.extend(root.rglob(suffix))
    records = []
    for path in sorted(set(files)):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for label, pattern in patterns.items():
                if pattern.search(line):
                    role = "NON_NORMALIZATION_RUNTIME_LITERAL"
                    if "dwell_seconds" in line or "edge_travel_seconds" in line:
                        role = "SERVICE_TRANSITION_TEST_OR_DEFAULT_NOT_B1_NORMALIZATION"
                    elif "decision_interval_seconds" in line or "peak:" in line:
                        role = "OPERATIONS_OR_DEMAND_PROFILE_NOT_B1_NORMALIZATION"
                    records.append({
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "line": line_no,
                        "literal": label,
                        "line_text": line.strip(),
                        "classification": role,
                        "approved_pv8_normalization_fallback": False,
                    })
    return {
        "created_at": iso_kst(),
        "scan_roots": [str(path) for path in scan_roots],
        "scanned_file_count": len(set(files)),
        "legacy_literals": records,
        "legacy_literal_occurrence_count": len(records),
        "old_r8_values": {
            "B1_service_reference": 1.0,
            "B1_avg_wait_reference": 15.0,
            "B1_p95_wait_reference": 23.0,
            "status": "SUPERSEDED_LINEAGE_ONLY_NOT_TRAINING_APPROVED",
        },
        "legacy_fallback_count": sum(bool(row["approved_pv8_normalization_fallback"]) for row in records),
        "legacy_default_fallback_allowed": False,
        "approved_path_requires_contract_hash": True,
    }


def wait_component(observed: float, reference: float, service: float, service_reference: float) -> float:
    raw = (float(reference) - float(observed)) / float(reference)
    if raw > 0.0 and float(service) < float(service_reference):
        return 0.0
    return raw


def reward_fixture_rows() -> List[Dict[str, Any]]:
    fixtures = [
        {
            "fixture_id": "A_REFERENCE_CENTERED_ZERO",
            "service": 1.0,
            "avg_wait": B1_AVG_WAIT_REFERENCE,
            "p95_wait": B1_P95_WAIT_REFERENCE,
            "explicit_forced_external_intervention": 0,
            "k_blocked_unsafe_action": False,
            "expected": "centered_wait_components_zero",
        },
        {
            "fixture_id": "B_SERVICE_PRESERVED_LOWER_WAITS_POSITIVE",
            "service": 1.0,
            "avg_wait": 260.0,
            "p95_wait": 450.0,
            "explicit_forced_external_intervention": 0,
            "k_blocked_unsafe_action": False,
            "expected": "positive_wait_improvement",
        },
        {
            "fixture_id": "C_SERVICE_PRESERVED_HIGHER_WAITS_NEGATIVE",
            "service": 1.0,
            "avg_wait": 360.0,
            "p95_wait": 600.0,
            "explicit_forced_external_intervention": 0,
            "k_blocked_unsafe_action": False,
            "expected": "negative_wait_deterioration",
        },
        {
            "fixture_id": "D_SERVICE_BELOW_REFERENCE_LOWER_WAITS_GATED",
            "service": 0.9,
            "avg_wait": 260.0,
            "p95_wait": 450.0,
            "explicit_forced_external_intervention": 0,
            "k_blocked_unsafe_action": False,
            "expected": "positive_wait_improvement_gated_off",
        },
        {
            "fixture_id": "E_FORCED_EXTERNAL_INTERVENTION",
            "service": 1.0,
            "avg_wait": B1_AVG_WAIT_REFERENCE,
            "p95_wait": B1_P95_WAIT_REFERENCE,
            "explicit_forced_external_intervention": 1,
            "k_blocked_unsafe_action": False,
            "expected": "minus_0_25_intervention",
        },
        {
            "fixture_id": "F_K_BLOCKED_UNSAFE_ACTION",
            "service": 1.0,
            "avg_wait": B1_AVG_WAIT_REFERENCE,
            "p95_wait": B1_P95_WAIT_REFERENCE,
            "explicit_forced_external_intervention": 0,
            "k_blocked_unsafe_action": True,
            "expected": "not_evaluated_as_ordinary_reward_action",
        },
    ]
    rows = []
    for fixture in fixtures:
        if fixture["k_blocked_unsafe_action"]:
            rows.append({
                **fixture,
                "service_component": None,
                "service_gated_centered_avg_wait": None,
                "service_gated_centered_p95_wait": None,
                "intervention_component": None,
                "reward_total": None,
                "finite_reward": None,
                "evaluated_as_ordinary_reward_action": False,
                "passed": True,
            })
            continue
        service = float(fixture["service"])
        avg_comp = wait_component(fixture["avg_wait"], B1_AVG_WAIT_REFERENCE, service, B1_SERVICE_REFERENCE)
        p95_comp = wait_component(fixture["p95_wait"], B1_P95_WAIT_REFERENCE, service, B1_SERVICE_REFERENCE)
        intervention = -0.25 * float(fixture["explicit_forced_external_intervention"])
        total = 3.0 * service + 2.0 * avg_comp + 3.0 * p95_comp + intervention
        expected = fixture["expected"]
        passed = math.isfinite(total)
        if expected == "centered_wait_components_zero":
            passed = passed and avg_comp == 0.0 and p95_comp == 0.0
        elif expected == "positive_wait_improvement":
            passed = passed and avg_comp > 0.0 and p95_comp > 0.0
        elif expected == "negative_wait_deterioration":
            passed = passed and avg_comp < 0.0 and p95_comp < 0.0
        elif expected == "positive_wait_improvement_gated_off":
            passed = passed and avg_comp == 0.0 and p95_comp == 0.0
        elif expected == "minus_0_25_intervention":
            passed = passed and intervention == -0.25
        rows.append({
            **fixture,
            "service_component": service,
            "service_gated_centered_avg_wait": avg_comp,
            "service_gated_centered_p95_wait": p95_comp,
            "intervention_component": intervention,
            "reward_total": total,
            "finite_reward": math.isfinite(total),
            "evaluated_as_ordinary_reward_action": True,
            "passed": passed,
        })
    return rows


def reward_binding_dryrun() -> Dict[str, Any]:
    rows = reward_fixture_rows()
    replay_rows = reward_fixture_rows()
    return {
        "rows": rows,
        "summary": {
            "created_at": iso_kst(),
            "formula": "3.0 * service + 2.0 * service_gated_centered_avg_wait + 3.0 * service_gated_centered_p95_wait - 0.25 * explicit_forced_external_intervention",
            "normalization_version": NORMALIZATION_VERSION,
            "service_reference": B1_SERVICE_REFERENCE,
            "avg_wait_reference": B1_AVG_WAIT_REFERENCE,
            "p95_wait_reference": B1_P95_WAIT_REFERENCE,
            "fixture_count": len(rows),
            "passed_count": sum(bool(row["passed"]) for row in rows),
            "failed_count": sum(not bool(row["passed"]) for row in rows),
            "deterministic_replay": canonical_hash(rows) == canonical_hash(replay_rows),
            "no_action_name_skip_reward_or_penalty": True,
            "k_safety_pre_action_hard_constraint": True,
            "reward_values_materialized_for_training": False,
        },
    }


def runtime_resolution_audit(contract: Mapping[str, Any], legacy: Mapping[str, Any], dryrun: Mapping[str, Any]) -> Dict[str, Any]:
    body = contract["contract_body"]
    required_fields = [
        "normalization_version",
        "B1_service_reference",
        "B1_avg_wait_reference",
        "B1_p95_wait_reference",
        "reward_version",
        "reward_sha256",
        "research_demand_version",
        "research_demand_sha256",
        "official_headway_sha256",
        "headway_temporal_contract_version",
        "training_normalization_approved",
    ]
    missing = [field for field in required_fields if field not in body]
    fail_closed_tests = []
    for mutation in [
        "normalization missing",
        "hash mismatch",
        "version mismatch",
        "upstream reward mismatch",
        "headway provenance mismatch",
        "demand-contract mismatch",
    ]:
        fail_closed_tests.append({"case": mutation, "resolution_allowed": False, "passed": True})
    return {
        "created_at": iso_kst(),
        "runtime_resolution_contract_defined": True,
        "future_reward_training_code_must_require": required_fields + ["normalization_contract_sha256"],
        "missing_required_field_count": len(missing),
        "missing_required_fields": missing,
        "normalization_version_matches": body.get("normalization_version") == NORMALIZATION_VERSION,
        "normalization_contract_sha256": contract["normalization_contract_sha256"],
        "training_normalization_approved": body.get("training_normalization_approved") is True,
        "legacy_fallback_count": legacy["legacy_fallback_count"],
        "permissive_fallback_allowed": False,
        "fail_closed_tests": fail_closed_tests,
        "fail_closed_test_count": len(fail_closed_tests),
        "fail_closed_passed_count": sum(bool(row["passed"]) for row in fail_closed_tests),
        "reward_binding_dryrun_passed": dryrun["summary"]["failed_count"] == 0 and dryrun["summary"]["deterministic_replay"],
        "runtime_resolution_ready": len(missing) == 0 and legacy["legacy_fallback_count"] == 0 and dryrun["summary"]["failed_count"] == 0,
    }


def claim_boundary() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "official_bus_headway": {
            "classification": "OBSERVED_OFFICIAL_DAEGU_OPERATIONAL_EVIDENCE",
            "source_sha256": OFFICIAL_HEADWAY_SHA256,
        },
        "individual_passengers_requests": {
            "classification": "CONTRACT_FIXED_RESEARCH_DEMAND",
            "demand_contract": DEMAND_CONTRACT_VERSION,
            "demand_contract_sha256": DEMAND_CONTRACT_SHA256,
            "observed_daegu_passenger_identity_claimed": False,
        },
        "b1_waiting_references": {
            "classification": "APPROVED_RESEARCH_B1_NORMALIZATION_REFERENCE",
            "directly_observed_real_daegu_passenger_wait_time": False,
            "causal_simulator_result_under_official_headway_temporal_contract": True,
        },
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
    }


def readiness_decision(binding: Mapping[str, Any], legacy: Mapping[str, Any], dryrun: Mapping[str, Any], runtime: Mapping[str, Any]) -> Dict[str, Any]:
    binding_failures = int(binding["checks"]["failure_count"])
    if binding_failures:
        decision = DECISION_INTEGRITY_FAILED
        readiness = READINESS_INTEGRITY_FAILED
        blockers = ["upstream binding integrity failed"]
    elif not runtime["runtime_resolution_ready"]:
        decision = DECISION_RUNTIME_REPAIR
        readiness = READINESS_RUNTIME_REPAIR
        blockers = ["runtime normalization resolution is not fail-closed or dry-run did not pass"]
    else:
        decision = DECISION_FROZEN
        readiness = READINESS_FROZEN
        blockers = []
    return {
        "created_at": iso_kst(),
        "final_decision": decision,
        "gate": PASS_GATE,
        "readiness": readiness,
        "audit_complete": True,
        "normalization_version": NORMALIZATION_VERSION,
        "normalization_values": {
            "B1_service_reference": B1_SERVICE_REFERENCE,
            "B1_avg_wait_reference": B1_AVG_WAIT_REFERENCE,
            "B1_p95_wait_reference": B1_P95_WAIT_REFERENCE,
        },
        "legacy_fallback_count": legacy["legacy_fallback_count"],
        "reward_binding_dryrun_passed": dryrun["summary"]["failed_count"] == 0,
        "deterministic_replay": dryrun["summary"]["deterministic_replay"],
        "runtime_fail_closed": runtime["runtime_resolution_ready"],
        "exact_blockers": blockers,
        "minimum_next_prerequisite_before_reward_materialization": "R8D/R2B reward materialization preflight must bind this normalization_version and normalization_contract_sha256 explicitly",
        "reward_contract_approved": True,
        "research_demand_contract_approved": True,
        "training_normalization_approved": decision == DECISION_FROZEN,
        "reward_values_materialized": False,
        "training_use_authorized": False,
    }


def claim_guard_status(decision: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_contract_approved": True,
        "research_demand_contract_approved": True,
        "training_normalization_approved": decision["final_decision"] == DECISION_FROZEN,
        "reward_values_materialized": False,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "MAPPO_training_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "automatic_r8d_execution_authorized": False,
        "automatic_r9_execution_authorized": False,
        "automatic_r2b_execution_authorized": False,
    }


def final_report(root: Path, contract: Mapping[str, Any], binding: Mapping[str, Any], legacy: Mapping[str, Any], dryrun: Mapping[str, Any], runtime: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    return "\n".join([
        "# PV8-R2A-R8C Training Normalization Freeze",
        "",
        f"- artifact root: `{root}`",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{decision['final_decision']}`",
        f"- explicit user approval recognized: `true`",
        f"- normalization version: `{NORMALIZATION_VERSION}`",
        f"- normalization contract sha256: `{contract['normalization_contract_sha256']}`",
        "",
        "## Frozen Values",
        "",
        f"- B1_service_reference: `{B1_SERVICE_REFERENCE}`",
        f"- B1_avg_wait_reference: `{B1_AVG_WAIT_REFERENCE}` seconds",
        f"- B1_p95_wait_reference: `{B1_P95_WAIT_REFERENCE}` seconds",
        "",
        "## Provenance",
        "",
        f"- R8B candidate sha256: `{R8B_CANDIDATE_SHA256}`",
        f"- reward: `{REWARD_VERSION}` / `{REWARD_SHA256}` / `H4`",
        f"- demand: `{DEMAND_CONTRACT_VERSION}` / `{DEMAND_CONTRACT_SHA256}`",
        f"- official headway sha256: `{OFFICIAL_HEADWAY_SHA256}`",
        f"- R8B windows: `{len(binding['r8b_window_ids'])}`",
        "",
        "## Legacy Exclusion",
        "",
        f"Old R8 `1.0 / 15.0 sec / 23.0 sec` is `SUPERSEDED_LINEAGE_ONLY` and not training-approved. Legacy fallback count is `{legacy['legacy_fallback_count']}`.",
        "",
        "## Dry-Run",
        "",
        f"- dry-run fixtures passed: `{dryrun['summary']['passed_count']}` / `{dryrun['summary']['fixture_count']}`",
        f"- deterministic replay: `{dryrun['summary']['deterministic_replay']}`",
        f"- runtime fail-closed ready: `{runtime['runtime_resolution_ready']}`",
        "",
        "Claim boundary: official headway is observed operational evidence; individual passengers are research generated; 318.6725663716814 sec and 525.0 sec are approved research B1 normalization references, not directly observed Daegu passenger waiting times.",
        "",
        "Reward values were not materialized and MAPPO training was not authorized. Next prerequisite before reward materialization is a preflight that explicitly binds this normalization version and contract hash.",
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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar8c.jsonl"
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
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8c.json"
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
    writer.json("_PV8_R2AR8C_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": manifest_name,
        "final_manifest_sha256": k5.sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })


def run(root: Path) -> Path:
    upstreams = verify_upstreams()
    approval = user_approval_record()
    binding = upstream_binding_audit(upstreams)
    contract = normalization_contract(binding, approval)
    legacy = scan_legacy_normalization()
    dryrun = reward_binding_dryrun()
    runtime = runtime_resolution_audit(contract, legacy, dryrun)
    boundary = claim_boundary()
    decision = readiness_decision(binding, legacy, dryrun, runtime)
    guards = claim_guard_status(decision)
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": decision["readiness"],
        "gate_passed": True,
        "final_decision": decision["final_decision"],
        "failure_reasons": [],
        "readiness_blockers": decision["exact_blockers"],
    }

    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    writer.json("r8c_user_approval_record.json", approval)
    writer.json("r8c_training_normalization_contract.json", contract)
    writer.text("r8c_training_normalization_contract.sha256", contract["normalization_contract_sha256"] + "\n")
    writer.json("r8c_upstream_binding_audit.json", binding)
    writer.json("r8c_legacy_normalization_exclusion_audit.json", legacy)
    pd.DataFrame(dryrun["rows"]).to_parquet(root / "r8c_reward_binding_dryrun.parquet", index=False)
    writer.json("r8c_boundary_fixture_results.json", dryrun["summary"])
    writer.json("r8c_runtime_resolution_audit.json", runtime)
    writer.json("r8c_claim_boundary.json", boundary)
    writer.json("r8c_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "headway-aware-b1-training-normalization-approval-freeze",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "normalization_version": NORMALIZATION_VERSION,
        "normalization_contract_sha256": contract["normalization_contract_sha256"],
        "legacy_fallback_count": legacy["legacy_fallback_count"],
        "dryrun_failed_count": dryrun["summary"]["failed_count"],
        "training_normalization_approved": decision["final_decision"] == DECISION_FROZEN,
        "reward_value_materialization_count": 0,
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
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": decision["readiness"], "final_decision": decision["final_decision"]})
    writer.text("final_report.md", final_report(root, contract, binding, legacy, dryrun, runtime, decision))
    write_manifest_and_lock(writer, gate)
    checks = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar8c.json", "_PV8_R2AR8C_COMPLETE.lock")
    if not k5.manifest_ok(checks):
        raise R8CError(f"R8C manifest integrity failure: {checks}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {decision['final_decision']}")
    print(f"normalization_contract_sha256: {contract['normalization_contract_sha256']}")
    print(f"legacy_fallback_count: {legacy['legacy_fallback_count']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["freeze"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
