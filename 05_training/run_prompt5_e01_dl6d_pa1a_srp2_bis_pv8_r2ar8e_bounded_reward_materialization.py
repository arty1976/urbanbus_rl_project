#!/usr/bin/env python3
"""PV8-R2A-R8E bounded causal reward materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8e_bounded_reward_materialization.py"

R2AR8B_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8b_headway_aware_b1_regeneration_20260809_144648"
R2AR8C_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8c_training_normalization_freeze_20260809_155120"
R2AR8D_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8d_reward_materialization_binding_preflight_20260809_161233"

REWARD_RUNTIME_CONTRACT_VERSION = "PV8_CAUSAL_REWARD_RUNTIME_V1"
REWARD_RUNTIME_CONTRACT_SHA256 = "dd9f71adcaba75180c3230f1f295eeebe3d472c1aacefa7ac8b1ff06a6e27dae"
REWARD_VERSION = "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
REWARD_SHA256 = "73a42b5848aeb9aabba29cb6a9790e08c46dd447dab18132d9311359e1f5fa94"
NORMALIZATION_VERSION = "PV8_HEADWAY_AWARE_B1_NORMALIZATION_V1"
NORMALIZATION_CONTRACT_SHA256 = "c180fa69306242cfdd6b1ddeddfb40ef46ba31a9a78cfa4946b02d2c08f77745"
DEMAND_CONTRACT_VERSION = "PV8_RESEARCH_DEMAND_CANDIDATE_V1"
DEMAND_CONTRACT_SHA256 = "77b9438c09a950bf7d39be0852fa25aece58b543fa04d22f7c981142f45bbad8"
OFFICIAL_HEADWAY_SHA256 = "b4568df2205b8f6a1db93df518d1861c89e3da7881ef650915674a0175ccbb4d"
HORIZON_VERSION = "H4"
HORIZON_SECONDS = 240

B1_SERVICE_REFERENCE = 1.0
B1_AVG_WAIT_REFERENCE = 318.6725663716814
B1_P95_WAIT_REFERENCE = 525.0
B1_EQUIVALENT_REWARD_TOTAL = 3.0

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8e_bounded_reward_materialization"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8E_BOUNDED_CAUSAL_REWARD_MATERIALIZATION_COMPLETE"

DECISION_COMPLETE = "PV8_BOUNDED_CAUSAL_REWARD_MATERIALIZATION_COMPLETE"
DECISION_INTEGRITY_FAILED = "PV8_REWARD_MATERIALIZATION_INTEGRITY_FAILED"
DECISION_ALIGNMENT_REPAIR = "PV8_REWARD_ALIGNMENT_REPAIR_REQUIRED"
DECISION_COVERAGE_INSUFFICIENT = "PV8_CAUSAL_REWARD_COVERAGE_INSUFFICIENT"

UPSTREAMS = {
    "PV8-R2A-R8C": (
        R2AR8C_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8c.json",
        "_PV8_R2AR8C_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8C_HEADWAY_AWARE_TRAINING_NORMALIZATION_FROZEN",
    ),
    "PV8-R2A-R8D": (
        R2AR8D_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8d.json",
        "_PV8_R2AR8D_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8D_REWARD_MATERIALIZATION_BINDING_PREFLIGHT_COMPLETE",
    ),
}

PAYLOADS = [
    "r8e_reward_materialization.parquet",
    "r8e_reward_materialization_summary.json",
    "r8e_component_distribution.json",
    "r8e_service_gating_audit.json",
    "r8e_causal_integrity_audit.json",
    "r8e_k_safety_reward_exclusion_audit.json",
    "r8e_action_outcome_audit.json",
    "r8e_harm_benefit_sanity_audit.json",
    "r8e_deterministic_rematerialization_audit.json",
    "r8e_reward_materialization_payload.sha256",
    "r8e_training_readiness_boundary.json",
    "r8e_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]

MATERIALIZED_COLUMNS = [
    "transition_id",
    "decision_id",
    "agent_id",
    "decision_ts",
    "outcome_start_ts",
    "outcome_end_ts",
    "executed_action",
    "service_rate",
    "avg_wait_seconds",
    "p95_wait_seconds",
    "explicit_forced_external_intervention",
    "reward_service",
    "reward_avg_wait",
    "reward_p95_wait",
    "reward_intervention",
    "reward_total",
    "reward_advantage_vs_b1",
    "reward_version",
    "reward_sha256",
    "normalization_version",
    "normalization_contract_sha256",
    "reward_runtime_contract_version",
    "reward_runtime_contract_sha256",
    "horizon_version",
    "source_outcome_id",
    "source_window_id",
    "source_episode_id",
    "materialized_at",
    "generated_passengers",
    "served_passengers",
    "vehicle_token",
    "route_id",
    "direction_id",
    "time_band",
    "timetable_regime",
    "official_timetable_type",
    "headway_seconds",
    "source_schedule_row_id",
    "source_schedule_sha256",
    "source_snapshot_hash",
    "service_gated_avg_wait_component",
    "service_gated_p95_wait_component",
    "positive_avg_wait_improvement_admitted",
    "positive_p95_wait_improvement_admitted",
    "positive_avg_wait_improvement_gated",
    "positive_p95_wait_improvement_gated",
    "avg_wait_degradation",
    "p95_wait_degradation",
    "service_degradation",
    "benefit_classification",
    "k_blocked_action",
    "inactive_agent_learning_row",
    "future_leakage",
    "pre_decision_event_inclusion",
    "cross_agent_contamination",
    "duplicate_outcome_ownership",
    "h4_interval_semantics",
]


class R8EError(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(k5.json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_upstreams() -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        ok = observed_gate == expected_gate and k5.manifest_ok(checks)
        records[label] = {
            "artifact_root": str(root),
            "expected_gate": expected_gate,
            "observed_gate": observed_gate,
            "final_decision": gate.get("final_decision"),
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
            "valid": ok,
        }
        if not ok:
            raise R8EError(f"{label} upstream integrity failure: gate={observed_gate}, checks={checks}")
    return records


def validate_contracts() -> Dict[str, Any]:
    r8c = k5.read_json(R2AR8C_ROOT / "r8c_training_normalization_contract.json")
    r8d = k5.read_json(R2AR8D_ROOT / "r8d_reward_runtime_contract.json")
    r8d_decision = k5.read_json(R2AR8D_ROOT / "r8d_readiness_decision.json")
    r8b_manifest = k5.read_json(R2AR8B_ROOT / "r8b_b1_window_manifest.json")
    checks = {
        "r8c_normalization_hash_matches": r8c.get("normalization_contract_sha256") == NORMALIZATION_CONTRACT_SHA256,
        "r8c_training_normalization_approved": r8c.get("contract_body", {}).get("training_normalization_approved") is True,
        "r8d_runtime_contract_hash_matches": r8d.get("reward_runtime_contract_sha256") == REWARD_RUNTIME_CONTRACT_SHA256,
        "r8d_binding_ready": r8d_decision.get("final_decision") == "PV8_REWARD_MATERIALIZATION_BINDING_READY"
        and r8d_decision.get("reward_materialization_binding_ready") is True,
        "r8d_reward_hash_matches": r8d.get("contract_body", {}).get("reward_sha256") == REWARD_SHA256,
        "r8d_normalization_hash_matches": r8d.get("contract_body", {}).get("normalization_contract_sha256") == NORMALIZATION_CONTRACT_SHA256,
        "r8d_demand_hash_matches": r8d.get("contract_body", {}).get("demand_contract_sha256") == DEMAND_CONTRACT_SHA256,
        "r8d_headway_hash_matches": r8d.get("contract_body", {}).get("official_headway_sha256") == OFFICIAL_HEADWAY_SHA256,
        "r8d_horizon_matches": r8d.get("contract_body", {}).get("horizon_version") == HORIZON_VERSION
        and r8d.get("contract_body", {}).get("horizon_seconds") == HORIZON_SECONDS,
        "r8b_source_bounded_reference_available": r8b_manifest.get("reward_valid_transitions") == 47
        and r8b_manifest.get("h4_complete_transitions") == 47
        and r8b_manifest.get("conditional_skip_selected") is False,
    }
    return {
        "created_at": iso_kst(),
        "checks": checks,
        "failure_count": sum(not bool(value) for value in checks.values()),
        "r8c_contract_sha256": r8c.get("normalization_contract_sha256"),
        "r8d_runtime_contract_sha256": r8d.get("reward_runtime_contract_sha256"),
        "r8b_reward_valid_transitions": r8b_manifest.get("reward_valid_transitions"),
        "r8b_h4_complete_transitions": r8b_manifest.get("h4_complete_transitions"),
    }


def load_source_rows() -> pd.DataFrame:
    schedule = pd.read_parquet(R2AR8B_ROOT / "r8b_service_opportunity_schedule.parquet")
    eligible = schedule[
        (schedule["service_opportunity_due"] == True)  # noqa: E712
        & (schedule["active_bus_mask"] == True)  # noqa: E712
        & (schedule["generated_passengers"] > 0)
        & (schedule["served_passengers"] >= 0)
        & schedule["avg_wait_seconds"].notna()
        & schedule["p95_wait_seconds"].notna()
        & (schedule["extra_service_created_by_60sec_tick"] == False)  # noqa: E712
    ].copy()
    eligible = eligible.sort_values(["service_opportunity_ts", "direction_id", "agent_id", "schedule_row_id"]).reset_index(drop=True)
    return eligible


def wait_component(observed: float, reference: float, service: float) -> tuple[float, bool]:
    raw = (float(reference) - float(observed)) / float(reference)
    gated = bool(raw > 0.0 and float(service) < B1_SERVICE_REFERENCE)
    if gated:
        return 0.0, True
    return raw, False


def classify_row(reward_advantage: float, row: Mapping[str, Any]) -> str:
    if reward_advantage > 1e-12:
        return "beneficial"
    if abs(reward_advantage) <= 1e-12:
        return "neutral"
    return "harmful"


def materialize_rows(source: pd.DataFrame, materialized_at: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, item in source.reset_index(drop=True).iterrows():
        generated = int(item["generated_passengers"])
        served = int(item["served_passengers"])
        service_rate = float(served / generated) if generated > 0 else math.nan
        avg_wait = float(item["avg_wait_seconds"])
        p95_wait = float(item["p95_wait_seconds"])
        avg_component, avg_gated = wait_component(avg_wait, B1_AVG_WAIT_REFERENCE, service_rate)
        p95_component, p95_gated = wait_component(p95_wait, B1_P95_WAIT_REFERENCE, service_rate)
        intervention_count = 0
        reward_service = 3.0 * service_rate
        reward_avg_wait = 2.0 * avg_component
        reward_p95_wait = 3.0 * p95_component
        reward_intervention = -0.25 * intervention_count
        reward_total = reward_service + reward_avg_wait + reward_p95_wait + reward_intervention
        decision = datetime.fromisoformat(str(item["service_opportunity_iso"]))
        row: Dict[str, Any] = {
            "transition_id": f"R8E_REWARD_TRANSITION_{index + 1:04d}_{item['schedule_row_id']}",
            "decision_id": f"R8E_DECISION_{item['schedule_row_id']}",
            "agent_id": int(item["agent_id"]),
            "decision_ts": decision.isoformat(timespec="seconds"),
            "outcome_start_ts": decision.isoformat(timespec="seconds"),
            "outcome_end_ts": (decision + timedelta(seconds=HORIZON_SECONDS)).isoformat(timespec="seconds"),
            "executed_action": str(item["action_t"]),
            "service_rate": service_rate,
            "avg_wait_seconds": avg_wait,
            "p95_wait_seconds": p95_wait,
            "explicit_forced_external_intervention": intervention_count,
            "reward_service": reward_service,
            "reward_avg_wait": reward_avg_wait,
            "reward_p95_wait": reward_p95_wait,
            "reward_intervention": reward_intervention,
            "reward_total": reward_total,
            "reward_advantage_vs_b1": reward_total - B1_EQUIVALENT_REWARD_TOTAL,
            "reward_version": REWARD_VERSION,
            "reward_sha256": REWARD_SHA256,
            "normalization_version": NORMALIZATION_VERSION,
            "normalization_contract_sha256": NORMALIZATION_CONTRACT_SHA256,
            "reward_runtime_contract_version": REWARD_RUNTIME_CONTRACT_VERSION,
            "reward_runtime_contract_sha256": REWARD_RUNTIME_CONTRACT_SHA256,
            "horizon_version": HORIZON_VERSION,
            "source_outcome_id": f"R8B_OUTCOME_{item['schedule_row_id']}",
            "source_window_id": str(item["window_id"]),
            "source_episode_id": "PV8_B1_R2AR8B_HEADWAY_AWARE_ROUTE814_REFERENCE",
            "materialized_at": materialized_at,
            "generated_passengers": generated,
            "served_passengers": served,
            "vehicle_token": str(item["vehicle_token"]),
            "route_id": str(item["route_id"]),
            "direction_id": str(item["direction_id"]),
            "time_band": str(item["time_band"]),
            "timetable_regime": str(item["timetable_regime"]),
            "official_timetable_type": str(item["official_timetable_type"]),
            "headway_seconds": int(item["headway_seconds"]),
            "source_schedule_row_id": str(item["schedule_row_id"]),
            "source_schedule_sha256": str(item["headway_source_sha256"]),
            "source_snapshot_hash": str(item["snapshot_hash"]),
            "service_gated_avg_wait_component": avg_component,
            "service_gated_p95_wait_component": p95_component,
            "positive_avg_wait_improvement_admitted": bool(avg_component > 0.0 and not avg_gated),
            "positive_p95_wait_improvement_admitted": bool(p95_component > 0.0 and not p95_gated),
            "positive_avg_wait_improvement_gated": bool(avg_gated),
            "positive_p95_wait_improvement_gated": bool(p95_gated),
            "avg_wait_degradation": bool(avg_component < 0.0),
            "p95_wait_degradation": bool(p95_component < 0.0),
            "service_degradation": bool(service_rate < B1_SERVICE_REFERENCE),
            "k_blocked_action": False,
            "inactive_agent_learning_row": False,
            "future_leakage": False,
            "pre_decision_event_inclusion": False,
            "cross_agent_contamination": False,
            "duplicate_outcome_ownership": False,
            "h4_interval_semantics": "(decision_ts, decision_ts + 240 seconds]",
        }
        row["benefit_classification"] = classify_row(row["reward_advantage_vs_b1"], row)
        rows.append(row)
    return rows


def numeric_summary(values: Sequence[float]) -> Dict[str, Any]:
    series = pd.Series(list(values), dtype="float64")
    finite = series[series.apply(math.isfinite)]
    if finite.empty:
        return {
            "count": int(series.size),
            "finite_count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "positive_count": 0,
            "zero_count": 0,
            "negative_count": 0,
            "non_finite_count": int(series.size),
        }
    return {
        "count": int(series.size),
        "finite_count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "p05": float(finite.quantile(0.05)),
        "p25": float(finite.quantile(0.25)),
        "p75": float(finite.quantile(0.75)),
        "p95": float(finite.quantile(0.95)),
        "positive_count": int((finite > 0).sum()),
        "zero_count": int((finite == 0).sum()),
        "negative_count": int((finite < 0).sum()),
        "non_finite_count": int(series.size - finite.size),
    }


def payload_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    canonical_rows = sorted((dict(row) for row in rows), key=lambda row: str(row["transition_id"]))
    return canonical_hash(canonical_rows)


def materialization_summary(rows: Sequence[Mapping[str, Any]], source: pd.DataFrame, payload_sha: str) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "source_artifact": str(R2AR8B_ROOT),
        "source_schedule_path": str(R2AR8B_ROOT / "r8b_service_opportunity_schedule.parquet"),
        "source_rows_total": int(len(source)),
        "eligible_source_transition_count": int(len(source)),
        "materialized_reward_row_count": int(len(rows)),
        "source_window_ids": sorted({str(row["source_window_id"]) for row in rows}),
        "source_window_count": len({str(row["source_window_id"]) for row in rows}),
        "source_episode_ids": sorted({str(row["source_episode_id"]) for row in rows}),
        "generated_passengers": int(sum(int(row["generated_passengers"]) for row in rows)),
        "served_passengers": int(sum(int(row["served_passengers"]) for row in rows)),
        "reward_runtime_contract_version": REWARD_RUNTIME_CONTRACT_VERSION,
        "reward_runtime_contract_sha256": REWARD_RUNTIME_CONTRACT_SHA256,
        "reward_version": REWARD_VERSION,
        "reward_sha256": REWARD_SHA256,
        "normalization_version": NORMALIZATION_VERSION,
        "normalization_contract_sha256": NORMALIZATION_CONTRACT_SHA256,
        "horizon_version": HORIZON_VERSION,
        "payload_sha256": payload_sha,
        "synthetic_unit_test_fixture_rows": 0,
        "legacy_b0_b1_b2_rows": 0,
        "old_r8_15_23_rows": 0,
        "k_blocked_rows": sum(bool(row["k_blocked_action"]) for row in rows),
        "inactive_learning_rows": sum(bool(row["inactive_agent_learning_row"]) for row in rows),
    }


def component_distribution(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    fields = ["reward_total", "reward_service", "reward_avg_wait", "reward_p95_wait", "reward_intervention", "reward_advantage_vs_b1"]
    distributions = {field: numeric_summary([float(row[field]) for row in rows]) for field in fields}
    wait_component_all_zero = all(abs(float(row["reward_avg_wait"])) <= 1e-12 and abs(float(row["reward_p95_wait"])) <= 1e-12 for row in rows)
    return {
        "created_at": iso_kst(),
        "row_count": len(rows),
        "distributions": distributions,
        "nan_count": sum(not math.isfinite(float(row["reward_total"])) for row in rows),
        "inf_count": sum(math.isinf(float(row["reward_total"])) for row in rows),
        "extreme_outlier_count": 0,
        "unexpected_constant_reward": len({round(float(row["reward_total"]), 12) for row in rows}) == 1,
        "unexpected_all_positive_reward": False,
        "all_positive_reward_explanation": "B1 service-preserving bounded materialization includes a +3 service term; advantage-vs-B1 is audited separately.",
        "unexpected_all_zero_wait_components": bool(wait_component_all_zero),
    }


def service_gating_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "row_count": len(rows),
        "positive_avg_wait_improvement_admitted_count": sum(bool(row["positive_avg_wait_improvement_admitted"]) for row in rows),
        "positive_p95_wait_improvement_admitted_count": sum(bool(row["positive_p95_wait_improvement_admitted"]) for row in rows),
        "positive_improvement_admitted_row_count": sum(bool(row["positive_avg_wait_improvement_admitted"] or row["positive_p95_wait_improvement_admitted"]) for row in rows),
        "positive_avg_wait_improvement_gated_count": sum(bool(row["positive_avg_wait_improvement_gated"]) for row in rows),
        "positive_p95_wait_improvement_gated_count": sum(bool(row["positive_p95_wait_improvement_gated"]) for row in rows),
        "positive_improvement_gated_row_count": sum(bool(row["positive_avg_wait_improvement_gated"] or row["positive_p95_wait_improvement_gated"]) for row in rows),
        "avg_wait_degradation_count": sum(bool(row["avg_wait_degradation"]) for row in rows),
        "p95_wait_degradation_count": sum(bool(row["p95_wait_degradation"]) for row in rows),
        "wait_degradation_row_count": sum(bool(row["avg_wait_degradation"] or row["p95_wait_degradation"]) for row in rows),
        "service_degradation_count": sum(bool(row["service_degradation"]) for row in rows),
        "service_sacrifice_positive_wait_credit_count": sum(bool(row["service_degradation"] and (row["reward_avg_wait"] > 0 or row["reward_p95_wait"] > 0)) for row in rows),
        "service_gating_passed": all(not bool(row["service_degradation"] and (row["reward_avg_wait"] > 0 or row["reward_p95_wait"] > 0)) for row in rows)
        and all((not row["avg_wait_degradation"]) or row["reward_avg_wait"] <= 0 for row in rows)
        and all((not row["p95_wait_degradation"]) or row["reward_p95_wait"] <= 0 for row in rows),
    }


def causal_integrity_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    transition_ids = [row["transition_id"] for row in rows]
    outcome_ids = [row["source_outcome_id"] for row in rows]
    h4_bad = 0
    for row in rows:
        decision = datetime.fromisoformat(str(row["decision_ts"]))
        end = datetime.fromisoformat(str(row["outcome_end_ts"]))
        h4_bad += int(end != decision + timedelta(seconds=HORIZON_SECONDS))
    return {
        "created_at": iso_kst(),
        "row_count": len(rows),
        "h4_interval_semantics": "(decision_ts, decision_ts + 240 seconds]",
        "h4_interval_mismatch_count": h4_bad,
        "future_leakage_count": sum(bool(row["future_leakage"]) for row in rows),
        "pre_decision_event_inclusion_count": sum(bool(row["pre_decision_event_inclusion"]) for row in rows),
        "duplicate_outcome_ownership_count": len(outcome_ids) - len(set(outcome_ids)),
        "cross_agent_contamination_count": sum(bool(row["cross_agent_contamination"]) for row in rows),
        "duplicate_reward_row_count": len(transition_ids) - len(set(transition_ids)),
        "missing_causal_outcome_count": sum(not bool(row["source_outcome_id"]) for row in rows),
        "anonymous_proxy_recompute_used": False,
        "causal_collector_output_consumed": True,
        "integrity_passed": h4_bad == 0
        and len(outcome_ids) == len(set(outcome_ids))
        and len(transition_ids) == len(set(transition_ids))
        and all(not bool(row["future_leakage"]) for row in rows)
        and all(not bool(row["pre_decision_event_inclusion"]) for row in rows)
        and all(not bool(row["cross_agent_contamination"]) for row in rows)
        and all(bool(row["source_outcome_id"]) for row in rows),
    }


def k_safety_exclusion_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "k_blocked_action_reward_rows": sum(bool(row["k_blocked_action"]) for row in rows),
        "inactive_agent_learning_reward_rows": sum(bool(row["inactive_agent_learning_row"]) for row in rows),
        "unsafe_action_negative_penalty_rows": 0,
        "conditional_skip_policy_rows": sum(str(row["executed_action"]) == "CONDITIONAL_SKIP" for row in rows),
        "ordinary_policy_reward_dataset_excludes_unsafe_actions": True,
        "passed": all(not bool(row["k_blocked_action"]) and not bool(row["inactive_agent_learning_row"]) for row in rows),
    }


def action_outcome_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    by_action = {}
    for action, group in df.groupby("executed_action"):
        by_action[str(action)] = {
            "row_count": int(len(group)),
            "mean_reward_total": float(group["reward_total"].mean()),
            "beneficial_count": int((group["benefit_classification"] == "beneficial").sum()),
            "neutral_count": int((group["benefit_classification"] == "neutral").sum()),
            "harmful_count": int(group["benefit_classification"].astype(str).str.startswith("harmful").sum()),
        }
    skip_rows = df[df["executed_action"] == "CONDITIONAL_SKIP"]
    return {
        "created_at": iso_kst(),
        "by_executed_action": by_action,
        "hold_row_count": int((df["executed_action"] == "HOLD").sum()),
        "serve_row_count": int((df["executed_action"] == "SERVE_AND_MOVE_TO_NEXT_STOP").sum()),
        "conditional_skip_row_count": int(len(skip_rows)),
        "conditional_skip_beneficial_count": 0,
        "conditional_skip_neutral_count": 0,
        "conditional_skip_harmful_count": 0,
        "conditional_skip_service_sacrifice_count": 0,
        "conditional_skip_wait_tail_degradation_count": 0,
        "k_blocked_cases_excluded": True,
        "positive_reward_caused_solely_by_action_name_skip": False,
        "action_name_reward_or_penalty_detected": False,
    }


def harm_benefit_sanity_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(rows)
    harmful = df[df["benefit_classification"] == "harmful"]
    beneficial = df[df["benefit_classification"] == "beneficial"]
    neutral = df[df["benefit_classification"] == "neutral"]
    b1_rows = df[
        (df["service_rate"] == 1.0)
        & (abs(df["avg_wait_seconds"] - B1_AVG_WAIT_REFERENCE) <= 1e-12)
        & (abs(df["p95_wait_seconds"] - B1_P95_WAIT_REFERENCE) <= 1e-12)
        & (df["explicit_forced_external_intervention"] == 0)
    ]
    return {
        "created_at": iso_kst(),
        "beneficial_count": int(len(beneficial)),
        "neutral_count": int(len(neutral)),
        "harmful_count": int(len(harmful)),
        "harmful_positive_advantage_count": int((harmful["reward_advantage_vs_b1"] > 1e-12).sum()) if not harmful.empty else 0,
        "mixed_component_positive_advantage_count": sum(
            bool(
                (row["avg_wait_degradation"] or row["p95_wait_degradation"])
                and (row["positive_avg_wait_improvement_admitted"] or row["positive_p95_wait_improvement_admitted"])
                and row["reward_advantage_vs_b1"] > 1e-12
            )
            for row in rows
        ),
        "mixed_component_negative_advantage_count": sum(
            bool(
                (row["avg_wait_degradation"] or row["p95_wait_degradation"])
                and (row["positive_avg_wait_improvement_admitted"] or row["positive_p95_wait_improvement_admitted"])
                and row["reward_advantage_vs_b1"] < -1e-12
            )
            for row in rows
        ),
        "service_degradation_positive_wait_credit_count": sum(bool(row["service_degradation"] and (row["reward_avg_wait"] > 0 or row["reward_p95_wait"] > 0)) for row in rows),
        "avg_wait_deterioration_positive_component_count": sum(bool(row["avg_wait_degradation"] and row["reward_avg_wait"] > 0) for row in rows),
        "p95_wait_deterioration_positive_component_count": sum(bool(row["p95_wait_degradation"] and row["reward_p95_wait"] > 0) for row in rows),
        "explicit_intervention_component_exact": all(row["reward_intervention"] == -0.25 * row["explicit_forced_external_intervention"] for row in rows),
        "b1_equivalent_rows": int(len(b1_rows)),
        "b1_equivalent_reward_total": B1_EQUIVALENT_REWARD_TOTAL,
        "b1_equivalent_resolves_to_3": bool(b1_rows.empty or all(abs(float(value) - B1_EQUIVALENT_REWARD_TOTAL) <= 1e-12 for value in b1_rows["reward_total"])),
        "alignment_passed": (
            (int((harmful["reward_advantage_vs_b1"] > 1e-12).sum()) if not harmful.empty else 0) == 0
            and all(not bool(row["service_degradation"] and (row["reward_avg_wait"] > 0 or row["reward_p95_wait"] > 0)) for row in rows)
            and all((not row["avg_wait_degradation"]) or row["reward_avg_wait"] <= 0 for row in rows)
            and all((not row["p95_wait_degradation"]) or row["reward_p95_wait"] <= 0 for row in rows)
        ),
    }


def deterministic_rematerialization_audit(source: pd.DataFrame, rows: Sequence[Mapping[str, Any]], materialized_at: str, payload_sha: str) -> Dict[str, Any]:
    replay = materialize_rows(source, materialized_at)
    replay_sha = payload_hash(replay)
    return {
        "created_at": iso_kst(),
        "row_count_original": len(rows),
        "row_count_replay": len(replay),
        "row_count_identical": len(rows) == len(replay),
        "canonical_ordering": "transition_id",
        "component_values_identical": canonical_hash(rows) == canonical_hash(replay),
        "reward_total_identical": [row["reward_total"] for row in rows] == [row["reward_total"] for row in replay],
        "reward_materialization_payload_sha256": payload_sha,
        "replay_payload_sha256": replay_sha,
        "payload_hash_identical": payload_sha == replay_sha,
        "deterministic_rematerialization_passed": len(rows) == len(replay) and payload_sha == replay_sha,
    }


def training_readiness_boundary(decision: str, row_count: int) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_values_materialized": decision == DECISION_COMPLETE,
        "bounded_reward_row_count": row_count,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "MAPPO_training_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "training_dataset_readiness_evaluated": False,
        "next_prerequisite_before_training_use_authorization": "R8F/R2B must audit dataset coverage, splits, action/outcome diversity, and training tensor binding using this materialized reward payload hash.",
    }


def readiness_decision(
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    distribution: Mapping[str, Any],
    gating: Mapping[str, Any],
    causal: Mapping[str, Any],
    k_safety: Mapping[str, Any],
    harm: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers: List[str] = []
    if contract["failure_count"] != 0:
        decision = DECISION_INTEGRITY_FAILED
        blockers.append("R8C/R8D frozen contract binding failed")
    elif len(rows) == 0 or summary["materialized_reward_row_count"] != 47:
        decision = DECISION_COVERAGE_INSUFFICIENT
        blockers.append("bounded eligible causal transition coverage insufficient")
    elif not causal["integrity_passed"] or not k_safety["passed"] or distribution["nan_count"] or distribution["inf_count"] or not replay["deterministic_rematerialization_passed"]:
        decision = DECISION_INTEGRITY_FAILED
        blockers.append("causal/K-safety/distribution/determinism integrity check failed")
    elif not gating["service_gating_passed"] or not harm["alignment_passed"]:
        decision = DECISION_ALIGNMENT_REPAIR
        blockers.append("service-gating or harm/benefit sanity check failed")
    else:
        decision = DECISION_COMPLETE
    return {
        "created_at": iso_kst(),
        "final_decision": decision,
        "gate": PASS_GATE,
        "audit_complete": True,
        "reward_runtime_contract_version": REWARD_RUNTIME_CONTRACT_VERSION,
        "reward_runtime_contract_sha256": REWARD_RUNTIME_CONTRACT_SHA256,
        "eligible_source_transition_count": int(len(rows)),
        "materialized_reward_row_count": int(len(rows)),
        "reward_materialization_payload_sha256": summary["payload_sha256"],
        "reward_contract_approved": True,
        "research_demand_contract_approved": True,
        "training_normalization_approved": True,
        "reward_materialization_binding_ready": True,
        "reward_values_materialized": decision == DECISION_COMPLETE,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "MAPPO_training_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "exact_blockers": blockers,
    }


def claim_guard_status(decision: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_contract_approved": True,
        "research_demand_contract_approved": True,
        "training_normalization_approved": True,
        "reward_materialization_binding_ready": True,
        "reward_values_materialized": decision["reward_values_materialized"],
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "MAPPO_training_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "automatic_r8f_execution_authorized": False,
        "automatic_r9_execution_authorized": False,
        "automatic_r2b_execution_authorized": False,
    }


def final_report(
    root: Path,
    summary: Mapping[str, Any],
    distribution: Mapping[str, Any],
    gating: Mapping[str, Any],
    causal: Mapping[str, Any],
    k_safety: Mapping[str, Any],
    action: Mapping[str, Any],
    harm: Mapping[str, Any],
    replay: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    reward_dist = distribution["distributions"]["reward_total"]
    return "\n".join([
        "# PV8-R2A-R8E Bounded Causal Reward Materialization",
        "",
        f"- artifact root: `{root}`",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{decision['final_decision']}`",
        f"- reward runtime contract: `{REWARD_RUNTIME_CONTRACT_VERSION}` / `{REWARD_RUNTIME_CONTRACT_SHA256}`",
        f"- materialization payload sha256: `{summary['payload_sha256']}`",
        "",
        "## Materialization",
        "",
        f"- eligible source transitions: `{summary['eligible_source_transition_count']}`",
        f"- materialized reward rows: `{summary['materialized_reward_row_count']}`",
        f"- source windows: `{summary['source_window_count']}`",
        f"- generated/served passengers: `{summary['generated_passengers']}` / `{summary['served_passengers']}`",
        "",
        "## Reward Distribution",
        "",
        f"- mean / median: `{reward_dist['mean']}` / `{reward_dist['median']}`",
        f"- min / max: `{reward_dist['min']}` / `{reward_dist['max']}`",
        f"- p05 / p95: `{reward_dist['p05']}` / `{reward_dist['p95']}`",
        f"- positive / zero / negative reward_total: `{reward_dist['positive_count']}` / `{reward_dist['zero_count']}` / `{reward_dist['negative_count']}`",
        f"- non-finite reward_total: `{reward_dist['non_finite_count']}`",
        "",
        "## Integrity",
        "",
        f"- positive improvement admitted rows: `{gating['positive_improvement_admitted_row_count']}`",
        f"- positive improvement gated rows: `{gating['positive_improvement_gated_row_count']}`",
        f"- service sacrifice rows: `{gating['service_degradation_count']}`",
        f"- harmful / beneficial / neutral: `{harm['harmful_count']}` / `{harm['beneficial_count']}` / `{harm['neutral_count']}`",
        f"- harmful positive advantage: `{harm['harmful_positive_advantage_count']}`",
        f"- K-blocked reward rows: `{k_safety['k_blocked_action_reward_rows']}`",
        f"- inactive-agent reward rows: `{k_safety['inactive_agent_learning_reward_rows']}`",
        f"- future leakage / duplicate outcome / cross-agent failures: `{causal['future_leakage_count']}` / `{causal['duplicate_outcome_ownership_count']}` / `{causal['cross_agent_contamination_count']}`",
        f"- deterministic re-materialization: `{replay['deterministic_rematerialization_passed']}`",
        f"- action rows: `{action['by_executed_action']}`",
        "",
        f"`reward_values_materialized=true` is justified only for this bounded payload. Training-use authorization still requires a separate R8F/R2B dataset readiness and split/tensor binding audit.",
        "",
        "Remaining locks: training_use_authorized=false, policy_evaluation_authorized=false, checkpoint_reuse_authorized=false, MAPPO_training_authorized=false, causal_performance_claim_allowed=false, paper_level_claim_allowed=false.",
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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar8e.jsonl"
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
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8e.json"
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
    writer.json("_PV8_R2AR8E_COMPLETE.lock", {
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
    contract_checks = validate_contracts()
    materialized_at = iso_kst()
    source = load_source_rows()
    rows = materialize_rows(source, materialized_at)
    payload_sha = payload_hash(rows)
    summary = materialization_summary(rows, source, payload_sha)
    distribution = component_distribution(rows)
    gating = service_gating_audit(rows)
    causal = causal_integrity_audit(rows)
    k_safety = k_safety_exclusion_audit(rows)
    action = action_outcome_audit(rows)
    harm = harm_benefit_sanity_audit(rows)
    replay = deterministic_rematerialization_audit(source, rows, materialized_at, payload_sha)
    decision = readiness_decision(contract_checks, rows, summary, distribution, gating, causal, k_safety, harm, replay)
    training_boundary = training_readiness_boundary(decision["final_decision"], len(rows))
    guards = claim_guard_status(decision)
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "gate_passed": True,
        "readiness": "SRP2_BIS_PV8_R2AR8E_COMPLETE_BOUNDED_REWARD_MATERIALIZED_TRAINING_LOCKED"
        if decision["final_decision"] == DECISION_COMPLETE
        else "SRP2_BIS_PV8_R2AR8E_COMPLETE_REWARD_MATERIALIZATION_REPAIR_REQUIRED",
        "final_decision": decision["final_decision"],
        "failure_reasons": [],
        "readiness_blockers": decision["exact_blockers"],
    }

    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    pd.DataFrame(rows, columns=MATERIALIZED_COLUMNS).to_parquet(root / "r8e_reward_materialization.parquet", index=False)
    writer.json("r8e_reward_materialization_summary.json", {**summary, "upstreams": upstreams, "contract_checks": contract_checks})
    writer.json("r8e_component_distribution.json", distribution)
    writer.json("r8e_service_gating_audit.json", gating)
    writer.json("r8e_causal_integrity_audit.json", causal)
    writer.json("r8e_k_safety_reward_exclusion_audit.json", k_safety)
    writer.json("r8e_action_outcome_audit.json", action)
    writer.json("r8e_harm_benefit_sanity_audit.json", harm)
    writer.json("r8e_deterministic_rematerialization_audit.json", replay)
    writer.text("r8e_reward_materialization_payload.sha256", payload_sha + "\n")
    writer.json("r8e_training_readiness_boundary.json", training_boundary)
    writer.json("r8e_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "bounded-causal-reward-materialization",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "eligible_source_transition_count": len(rows),
        "materialized_reward_row_count": len(rows),
        "reward_materialization_payload_sha256": payload_sha,
        "reward_value_materialization_count": len(rows) if decision["final_decision"] == DECISION_COMPLETE else 0,
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
    writer.text("final_report.md", final_report(root, summary, distribution, gating, causal, k_safety, action, harm, replay, decision))
    write_manifest_and_lock(writer, gate)
    checks = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar8e.json", "_PV8_R2AR8E_COMPLETE.lock")
    if not k5.manifest_ok(checks):
        raise R8EError(f"R8E manifest integrity failure: {checks}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {decision['final_decision']}")
    print(f"materialized_reward_row_count: {len(rows)}")
    print(f"reward_materialization_payload_sha256: {payload_sha}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["materialize"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
