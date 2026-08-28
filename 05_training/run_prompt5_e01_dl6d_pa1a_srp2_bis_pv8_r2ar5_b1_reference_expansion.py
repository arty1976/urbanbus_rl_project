#!/usr/bin/env python3
"""PV8-R2A-R5 prospective B1 reference expansion audit."""

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
from typing import Any, Dict, List, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar5_b1_reference_expansion.py"
COLLECTOR_SOURCE = TRAINING_ROOT / "simulator" / "pv8_reward_outcome_collector.py"

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"
R1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r1_mappo_retraining_preflight_20260808_142604"
R2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2_reward_episode_audit_20260808_143743"
R2AR2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar2_causal_reward_infrastructure_20260808_160345"
R2AR3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar3_bounded_reward_horizon_ablation_20260809_005642"
R2AR4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar4_reward_approval_normalization_preflight_20260809_100508"

UPSTREAMS = {
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
    "PV8-K9": (K9_ROOT, "artifact_manifest_srp2_bis_pv8_k9.json", "_PV8_K9_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED"),
    "PV8-R2A-R2": (R2AR2_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar2.json", "_PV8_R2AR2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR2_CAUSAL_REWARD_INFRASTRUCTURE_COMPLETE"),
    "PV8-R2A-R3": (R2AR3_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar3.json", "_PV8_R2AR3_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR3_BOUNDED_REWARD_HORIZON_ABLATION_COMPLETE"),
    "PV8-R2A-R4": (R2AR4_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar4.json", "_PV8_R2AR4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR4_REWARD_APPROVAL_AND_NORMALIZATION_PREFLIGHT_COMPLETE"),
}

SUPPORTING_SOURCES = {
    "PV8-C2": (C2_ROOT, "artifact_manifest_srp2_bis_pv8_c2.json", "_PV8_C2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"),
    "PV8-R1": (R1_ROOT, "artifact_manifest_srp2_bis_pv8_r1.json", "_PV8_R1_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R1_MAPPO_RETRAINING_PREFLIGHT_COMPLETE"),
    "PV8-R2": (R2_ROOT, "artifact_manifest_srp2_bis_pv8_r2.json", "_PV8_R2_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2_REWARD_AND_EPISODE_DATA_AUDIT_COMPLETE"),
}

REWARD_VERSION = "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
REWARD_SHA256 = "73a42b5848aeb9aabba29cb6a9790e08c46dd447dab18132d9311359e1f5fa94"
RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"
ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar5_b1_reference_expansion"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR5_B1_REFERENCE_EXPANSION_AUDIT_COMPLETE"
DECISION = "PV8_B1_REFERENCE_EXPANSION_INSUFFICIENT"
READINESS = "SRP2_BIS_PV8_R2AR5_COMPLETE_NO_TRAINING_ELIGIBLE_B1_H4_WINDOWS_ADDITIONAL_COLLECTION_REQUIRED"

WINDOW_COLUMNS = [
    "episode_id", "window_id", "window_start_ts", "window_end_ts", "partition",
    "service_rate", "avg_wait_seconds", "p95_wait_seconds", "active_count",
    "inactive_count", "eligible_request_count", "served_request_count",
    "reward_input_completeness", "h4_outcome_completeness", "all_8_slots_present",
    "b1_action_contract_valid", "training_eligible", "exclusion_reason",
]

WINDOW_DTYPES = {
    "episode_id": "string",
    "window_id": "string",
    "window_start_ts": "string",
    "window_end_ts": "string",
    "partition": "string",
    "service_rate": "float64",
    "avg_wait_seconds": "float64",
    "p95_wait_seconds": "float64",
    "active_count": "int64",
    "inactive_count": "int64",
    "eligible_request_count": "int64",
    "served_request_count": "int64",
    "reward_input_completeness": "float64",
    "h4_outcome_completeness": "float64",
    "all_8_slots_present": "bool",
    "b1_action_contract_valid": "bool",
    "training_eligible": "bool",
    "exclusion_reason": "string",
}

PAYLOADS = [
    "r2ar5_b1_episode_manifest.json",
    "r2ar5_b1_window_metrics.parquet",
    "r2ar5_b1_distribution_summary.json",
    "r2ar5_leave_one_window_out.json",
    "r2ar5_normalization_candidate.json",
    "r2ar5_reference_scope_audit.json",
    "r2ar5_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R2AR5Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(k5.json_clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_artifacts(definitions: Mapping[str, Any]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in definitions.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed != expected_gate or not k5.manifest_ok(checks):
            raise R2AR5Error(f"{label} integrity failure: gate={observed}, checks={checks}")
        results[label] = {"artifact_root": str(root), "gate": observed, "manifest_integrity": checks}
    return results


def verify_frozen_contract() -> Dict[str, Any]:
    upstream = verify_artifacts(UPSTREAMS)
    supporting = verify_artifacts(SUPPORTING_SOURCES)
    contract = k5.read_json(R2AR4_ROOT / "r2ar4_frozen_reward_contract.json")
    contract_body = contract.get("contract_body")
    r3 = k5.read_json(R2AR3_ROOT / "r2ar3_candidate_selection_decision.json")
    r4 = k5.read_json(R2AR4_ROOT / "r2ar4_readiness_decision.json")
    k8 = k5.read_json(K8_ROOT / "k8_snapshot_version_binding.json")
    k9 = k5.read_json(K9_ROOT / "k9_version_hash_binding_audit.json")
    r2ar2_run = k5.read_json(R2AR2_ROOT / "run_manifest.json")
    checks = {
        "reward_version_matches": contract_body.get("reward_version") == REWARD_VERSION,
        "reward_hash_recorded_matches": contract.get("reward_contract_sha256") == REWARD_SHA256,
        "reward_hash_recomputed_matches": canonical_hash(contract_body) == REWARD_SHA256,
        "reward_contract_approved": contract.get("reward_contract_approved") is True,
        "training_normalization_not_approved": contract.get("training_normalization_approved") is False,
        "horizon_h4_matches": contract_body.get("horizon", {}).get("id") == "H4" and contract_body.get("horizon", {}).get("seconds") == 240,
        "r3_selection_matches": r3.get("selected_candidate_id") == REWARD_VERSION and r3.get("selected_horizon_candidate") == "H4",
        "r4_decision_matches": r4.get("final_decision") == "PV8_REWARD_APPROVED_B1_EXPANSION_REQUIRED",
        "k8_hashes_match": k8.get("static_rulebook_sha256") == RULEBOOK_SHA256 and k8.get("occurrence_master_sha256") == OCCURRENCE_SHA256,
        "k9_hashes_match": k9.get("rulebook_sha256") == RULEBOOK_SHA256 and k9.get("occurrence_master_sha256") == OCCURRENCE_SHA256,
        "collector_source_matches_r2ar2": k5.sha256_file(COLLECTOR_SOURCE) == r2ar2_run.get("collector_source_sha256"),
    }
    checks["failure_count"] = sum(not value for value in checks.values())
    if checks["failure_count"]:
        raise R2AR5Error(f"frozen contract integrity failure: {checks}")
    return {"authoritative_upstreams": upstream, "supporting_sources": supporting, "frozen_contract_checks": checks}


def source_inventory() -> Dict[str, Any]:
    c2_path = C2_ROOT / "prospective_agent_cycle_state.parquet"
    c2 = pd.read_parquet(c2_path)
    r1_path = R1_ROOT / "r1_episode_coverage.parquet"
    r1_rows = pd.read_parquet(r1_path)
    r1_summary = k5.read_json(R1_ROOT / "r1_episode_coverage.json")
    r2 = k5.read_json(R2_ROOT / "r2_dataset_coverage.json")
    bounded = k5.read_json(R2AR2_ROOT / "r2ar2_pv8_b1_reference_audit.json")

    required_outcome_fields = {
        "passenger_generated", "passenger_served", "individual_waiting_time",
        "service_rate", "avg_wait_seconds", "p95_wait_seconds",
    }
    c2_columns = set(str(value) for value in c2.columns)
    reward_payloads = [json.loads(value) for value in r1_rows["reward_required_fields_json"].tolist()]
    materialized_count = sum(bool(row.get("reward_value_materialized")) for row in reward_payloads)
    missing_outcome_fields = sorted(required_outcome_fields - c2_columns)

    if len(c2) != 240 or r2.get("reward_valid_transition_count") != 0 or materialized_count != 0:
        raise R2AR5Error("prospective source inventory differs from frozen R1/R2 evidence")

    return {
        "created_at": iso_kst(),
        "prospective_position_mask_episode": {
            "source_path": str(c2_path),
            "source_sha256": k5.sha256_file(c2_path),
            "episode_ids": ["PV8_PROSPECTIVE_EPISODE_0001"],
            "episode_count": r1_summary["episode_count"],
            "cycle_count": r1_summary["cycle_count"],
            "snapshot_count": r1_summary["agent_snapshot_count"],
            "active_count": r1_summary["active_agent_snapshot_count"],
            "inactive_count": r1_summary["inactive_agent_snapshot_count"],
            "chronological_start_utc": r2["time_band_utc"]["start"],
            "chronological_end_utc": r2["time_band_utc"]["end"],
            "date_coverage_utc": r2["date_coverage_utc"],
            "fixed_slot_count": len(r1_summary["per_agent"]),
            "reward_value_materialized_count": materialized_count,
            "reward_valid_transition_count": r2["reward_valid_transition_count"],
            "passenger_request_outcome_columns_present": False,
            "missing_required_outcome_fields": missing_outcome_fields,
            "b1_control_executed": False,
            "h4_outcomes_available": False,
            "training_eligible": False,
            "exclusion_reason": "position/mask snapshots do not contain passenger/request causal outcomes and were not generated under B1 SERVE/HOLD control",
        },
        "bounded_r2ar2_fixture": {
            "source_class": bounded["source_class"],
            "active_sample_count": bounded["active_reference_sample_count"],
            "window_count": 1,
            "service_rate": bounded["reference_metrics"]["service_rate"],
            "avg_wait_seconds": bounded["reference_metrics"]["avg_wait_seconds"],
            "p95_wait_seconds": bounded["reference_metrics"]["p95_wait_seconds"],
            "scope": "ABLATION_REFERENCE_ONLY",
            "prospective_training_reference": False,
            "training_eligible": False,
            "exclusion_reason": "deterministic synthetic K4 identity fixture is not broader prospective training-normalization evidence",
        },
        "historical_pre_pv8_b1_sources": {
            "searched": True,
            "usable_count": 0,
            "classification": "PRE_PV8_CONTRACT_INCOMPATIBLE_NOT_USED",
            "reason": "historical B1 artifacts predate physical-slot, K8/K9, H4 collector, and approved F reward bindings",
        },
        "training_eligible_episode_count": 0,
        "training_eligible_window_count": 0,
        "window_rows_fabricated": 0,
        "c1_retrospective_selection_used": False,
        "validation_or_test_outcomes_used": False,
        "future_information_used": False,
    }


def episode_manifest(inventory: Mapping[str, Any]) -> Dict[str, Any]:
    source = inventory["prospective_position_mask_episode"]
    bounded = inventory["bounded_r2ar2_fixture"]
    return {
        "created_at": iso_kst(),
        "manifest_version": "PV8_R2AR5_B1_EPISODE_MANIFEST_V1",
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "horizon": "H4",
        "horizon_interval": "(decision_ts, decision_ts + 240 seconds] capped at terminal/revisit boundary",
        "b1_control": {
            "active_primary": "SERVE_WHEN_VALID",
            "active_fallback": "HOLD_ONLY_WHEN_SERVE_UNAVAILABLE",
            "conditional_skip_selected": False,
            "learned_policy_used": False,
        },
        "source_candidates": [
            {"source_id": "PV8_PROSPECTIVE_EPISODE_0001", **source},
            {"source_id": "R2AR2_BOUNDED_IDENTITY_FIXTURE", **bounded},
        ],
        "available_prospective_episode_count": source["episode_count"],
        "training_eligible_episode_count": 0,
        "training_eligible_window_count": 0,
        "window_metrics_row_count": 0,
        "chronological_training_fit_windows": [],
        "normalization_fit_partition": "TRAIN_ONLY",
        "validation_test_fit_window_count": 0,
        "episode_or_window_fabrication": False,
    }


def empty_distribution(metric: str) -> Dict[str, Any]:
    return {
        "metric": metric,
        "count": 0,
        "mean": None,
        "median": None,
        "std": None,
        "min": None,
        "max": None,
        "q1": None,
        "q3": None,
        "iqr": None,
        "cv": None,
        "cv_mean_zero_or_absent": True,
        "reason": "no training-eligible prospective B1 H4 window",
    }


def distribution_summary() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "training_eligible_window_count": 0,
        "metrics": {
            name: empty_distribution(name)
            for name in ["service_rate", "avg_wait_seconds", "p95_wait_seconds", "active_count", "inactive_count"]
        },
        "cross_window_stability": "NOT_EVALUABLE",
        "agent_balance": "NOT_EVALUABLE_NO_TRAINING_ELIGIBLE_WINDOWS",
        "time_band_coverage": [],
        "outlier_dependence": "NOT_EVALUABLE",
        "finite_nonzero_denominator_check": "NOT_EVALUABLE",
        "synthetic_statistics_generated": False,
    }


def leave_one_window_out() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "method": "for each eligible training window, refit service/avg-wait/p95 constants on all other eligible training windows",
        "minimum_mechanical_window_requirement": 3,
        "minimum_requirement_derivation": "three windows ensure every leave-one-window-out fit retains at least two chronological windows",
        "eligible_window_count": 0,
        "iteration_count": 0,
        "results": [],
        "service_reference_sensitivity": None,
        "avg_wait_reference_sensitivity": None,
        "p95_wait_reference_sensitivity": None,
        "evaluated": False,
        "reason": "no eligible window; position-only cycles cannot be converted to passenger outcome windows",
    }


def normalization_candidate() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "candidate_id": None,
        "candidate_produced": False,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "source_partition": "TRAIN_ONLY",
        "source_window_count": 0,
        "constants": {
            "service_preservation_reference": None,
            "avg_wait_seconds": None,
            "p95_wait_seconds": None,
        },
        "service_reference_fixed_to_one": False,
        "bounded_r2ar2_constants_used_for_fit": False,
        "bounded_r2ar2_constants_scope": "ABLATION_REFERENCE_ONLY",
        "validation_test_outcomes_used_for_fit": False,
        "finite_nonzero_denominators_verified": False,
        "training_normalization_approved": False,
        "approval_required_after_future_candidate_generation": True,
        "status": "NOT_PRODUCED_INSUFFICIENT_PROSPECTIVE_B1_WINDOWS",
    }


def reference_scope_audit(inventory: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "audit_complete": True,
        "sources_examined": [
            {
                "source": "C2/R1 prospective physical-slot episode",
                "classification": "PROSPECTIVE_POSITION_AND_MASK_ONLY",
                "training_normalization_use": False,
            },
            {
                "source": "R2 reward episode audit",
                "classification": "ZERO_REWARD_VALID_TRANSITIONS",
                "training_normalization_use": False,
            },
            {
                "source": "R2A-R2 bounded B1 fixture",
                "classification": "SYNTHETIC_ABLATION_REFERENCE_ONLY",
                "training_normalization_use": False,
            },
            {
                "source": "historical pre-PV8 B1 artifacts",
                "classification": "CONTRACT_INCOMPATIBLE",
                "training_normalization_use": False,
            },
        ],
        "eligible_episode_count": inventory["training_eligible_episode_count"],
        "eligible_window_count": inventory["training_eligible_window_count"],
        "position_only_episode_partitioned_into_fake_outcome_windows": False,
        "c1_retrospective_selection_used": False,
        "future_information_used": False,
        "validation_test_outcomes_used": False,
        "exact_additional_collection_requirement": {
            "minimum_nonoverlapping_training_windows": 3,
            "window_count_derivation": "leave-one-window-out refits must retain at least two chronological windows",
            "minimum_distinct_time_bands": 2,
            "fixed_slots_per_snapshot": 8,
            "required_action_contract": "SERVE when valid; HOLD only fallback; CONDITIONAL_SKIP never selected",
            "required_outcomes": [
                "passenger/request identity and generated timestamp",
                "served/completed status",
                "individual waiting time or exact generated-to-board timestamps",
                "H4-complete causal event interval for every active sampled decision",
                "active/inactive slot counts and complete reward inputs",
            ],
            "per_window_requirements": [
                "non-empty eligible passenger/request cohort",
                "finite service rate",
                "finite strictly positive avg-wait and p95-wait denominators",
                "100% H4 outcome completeness for included active decisions",
                "no identity/hash/version/future-leakage failure",
            ],
            "fit_partition": "chronological training windows only",
            "validation_test_use": "never fit constants",
            "approved_source_requirement": "prospective simulator B1 control execution with approved passenger/request event schedule; no synthetic duplication of the bounded fixture",
        },
        "new_bis_api_required_by_this_audit": False,
        "additional_collection_executed": False,
    }


def readiness_decision() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "final_decision": DECISION,
        "audit_complete": True,
        "reference_integrity_failure": False,
        "available_prospective_episode_count": 1,
        "training_eligible_episode_count": 0,
        "training_eligible_window_count": 0,
        "normalization_candidate_produced": False,
        "broader_evidence_sufficient": False,
        "reward_contract_approved": True,
        "training_normalization_approved": False,
        "reward_values_materialized": False,
        "exact_blocker": "no prospective B1 SERVE/HOLD execution with passenger/request outcomes and complete H4 causal windows",
        "minimum_next_stage": "collect the exact R2A-R5 scope-audit requirements, then rerun R2A-R5 in a new artifact before any normalization approval",
        "r2ar6_or_r2b_authorized": False,
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
        "automatic_r2ar6_execution_authorized": False,
        "automatic_r2b_execution_authorized": False,
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
    env["PYTHONPYCACHEPREFIX"] = "/private/tmp/pv8_r2ar5_pycache"
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


def final_report(root: Path) -> str:
    return "\n".join([
        "# PV8-R2A-R5 Prospective B1 Reference Expansion Audit",
        "",
        f"- artifact root: `{root}`",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{DECISION}`",
        f"- reward: `{REWARD_VERSION}` / `{REWARD_SHA256}` / `H4`",
        "",
        "## Available Evidence",
        "",
        "The current prospective evidence contains one episode, 30 cycles, 240 fixed-slot snapshots, 214 active rows, and 26 inactive rows from one UTC date/time band. It contains vehicle position, identity, occurrence, and K-mask state, but no passenger-generated/served events, individual waits, or H4 causal reward outcomes. Reward-valid transitions are zero.",
        "",
        "The R2A-R2 bounded B1 fixture has one window and eight synthetic active samples with service `1.0`, average wait `50.5`, and p95 wait `59.95`. It remains `ABLATION_REFERENCE_ONLY` and was not included in training fitting. Historical pre-PV8 B1 artifacts were also excluded as contract-incompatible.",
        "",
        "## Window Metrics and Distributions",
        "",
        "Training-eligible B1 episode/window count is `0 / 0`; the window metrics parquet therefore has zero rows with a frozen schema. Service-rate, average-wait, p95-wait, active/inactive distributions, cross-window stability, agent balance, outlier dependence, and finite-denominator statistics are not evaluable. No null statistic was converted into zero.",
        "",
        "## Normalization Candidate",
        "",
        "No candidate constants were produced. Service preservation was not fixed to `1.0`; service, average-wait, and p95-wait references remain null. Leave-one-window-out sensitivity has zero iterations and is not evaluable.",
        "",
        "## Exact Additional Collection",
        "",
        "A rerun requires at least three non-overlapping chronological training windows so every leave-one-window-out fit retains two windows, at least two time bands, eight fixed slots per snapshot, non-empty eligible passenger cohorts, complete generated/served/wait events, and H4-complete causal outcomes. The control must select SERVE when valid, HOLD only as fallback, and never CONDITIONAL_SKIP. Only chronological training windows may fit constants; validation/test outcomes remain excluded.",
        "",
        "The audit itself passed deterministically, but broader evidence is insufficient. No API/DB access, synthetic episode duplication, C1 selection reuse, reward materialization, policy evaluation, episode expansion, checkpoint reuse, downstream run, or MAPPO training occurred.",
        "",
    ])


def write_window_metrics(writer: k5.Writer, rows: Sequence[Mapping[str, Any]]) -> None:
    frame = pd.DataFrame([k5.json_clean(dict(row)) for row in rows], columns=WINDOW_COLUMNS)
    for column, dtype in WINDOW_DTYPES.items():
        frame[column] = frame[column].astype(dtype)
    path = writer.root / "r2ar5_b1_window_metrics.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar5.jsonl"
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
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar5.json"
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
    writer.json("_PV8_R2AR5_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": manifest_name,
        "final_manifest_sha256": k5.sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })


def run(root: Path) -> Path:
    upstream = verify_frozen_contract()
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    regression = run_pytest_regression()
    if not regression["passed"]:
        raise R2AR5Error(f"reward/K-safety regression failure: {regression}")
    inventory = source_inventory()
    episodes = episode_manifest(inventory)
    window_rows: List[Dict[str, Any]] = []
    distributions = distribution_summary()
    loo = leave_one_window_out()
    candidate = normalization_candidate()
    scope = reference_scope_audit(inventory)
    decision = readiness_decision()
    guards = claim_guards()
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": READINESS,
        "gate_passed": True,
        "final_decision": DECISION,
        "failure_reasons": [],
        "readiness_blockers": [decision["exact_blocker"]],
    }

    writer.json("r2ar5_b1_episode_manifest.json", episodes)
    write_window_metrics(writer, window_rows)
    writer.json("r2ar5_b1_distribution_summary.json", distributions)
    writer.json("r2ar5_leave_one_window_out.json", loo)
    writer.json("r2ar5_normalization_candidate.json", candidate)
    writer.json("r2ar5_reference_scope_audit.json", scope)
    writer.json("r2ar5_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "prospective-b1-reference-expansion-audit",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "collector_source_path": str(COLLECTOR_SOURCE),
        "collector_source_sha256": k5.sha256_file(COLLECTOR_SOURCE),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "upstream_integrity": upstream,
        "pytest_regression": regression,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "horizon": "H4",
        "available_prospective_episode_count": episodes["available_prospective_episode_count"],
        "training_eligible_episode_count": 0,
        "training_eligible_window_count": 0,
        "window_metrics_row_count": 0,
        "normalization_candidate_count": 0,
        "new_bis_api_call_count": 0,
        "db_query_count": 0,
        "db_write_count": 0,
        "reward_value_materialization_count": 0,
        "synthetic_episode_generation_count": 0,
        "c1_retrospective_selection_count": 0,
        "validation_test_fit_outcome_count": 0,
        "policy_evaluation_count": 0,
        "mappo_episode_expansion_count": 0,
        "checkpoint_reuse_count": 0,
        "r2ar6_execution_count": 0,
        "r2b_execution_count": 0,
        "mappo_training_count": 0,
        "qwen_train": False,
        "qwen_inference": False,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": READINESS, "final_decision": DECISION})
    writer.text("final_report.md", final_report(root))
    write_manifest_and_lock(writer, gate)

    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar5.json", "_PV8_R2AR5_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise R2AR5Error(f"R2A-R5 artifact integrity failure: {own}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {DECISION}")
    print("training_eligible_b1_windows: 0")
    print("normalization_candidate_produced: false")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["expand-and-audit"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
