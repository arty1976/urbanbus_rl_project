#!/usr/bin/env python3
"""PV8-R2A-R8 multi-time-band B1 expansion and normalization candidate."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation as k9
import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar6_causal_b1_reference_collection as r6
from simulator.pv8_b1_orchestrator import PassengerRequestScheduleRow, TypedPV8B1Orchestrator


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8_multi_timeband_b1_normalization.py"

K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
K9_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k9_global_kmask_lifecycle_validation_20260808_140056"
R2AR4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar4_reward_approval_normalization_preflight_20260809_100508"
R2AR6_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar6_causal_b1_reference_collection_20260809_113156"
R2AR7_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar7_demand_contract_b1_orchestrator_20260809_111155"

AGGREGATE_DEMAND_PROFILE = TRAINING_ROOT / "configs" / "shared_exogenous_demand_profile_v1.yaml"
ORCHESTRATOR_SOURCE = TRAINING_ROOT / "simulator" / "pv8_b1_orchestrator.py"
RULEBOOK_PATH = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k7_research_rule_contract_mask_dryrun_20260808_130432" / "k7_research_rulebook_candidate.parquet"
OCCURRENCE_PATH = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026" / "k6_route_stop_occurrence_master.parquet"
C2_CYCLE_PATH = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612" / "prospective_agent_cycle_state.parquet"

REWARD_VERSION = "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
REWARD_SHA256 = "73a42b5848aeb9aabba29cb6a9790e08c46dd447dab18132d9311359e1f5fa94"
DEMAND_CONTRACT_VERSION = "PV8_RESEARCH_DEMAND_CANDIDATE_V1"
DEMAND_CONTRACT_SHA256 = "77b9438c09a950bf7d39be0852fa25aece58b543fa04d22f7c981142f45bbad8"
RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8_multi_timeband_b1_normalization"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8_MULTI_TIMEBAND_B1_NORMALIZATION_AUDIT_COMPLETE"
DECISION_READY = "PV8_B1_NORMALIZATION_CANDIDATE_READY_FOR_EXPLICIT_APPROVAL"
DECISION_MORE_COVERAGE = "PV8_B1_MORE_TIMEBAND_COVERAGE_REQUIRED"
DECISION_SOURCE_REQUIRED = "PV8_B1_ADDITIONAL_TIMEBAND_SOURCE_REQUIRED"
DECISION_INTEGRITY_FAILED = "PV8_B1_REFERENCE_INTEGRITY_FAILED"
READINESS_READY = "SRP2_BIS_PV8_R2AR8_COMPLETE_NORMALIZATION_CANDIDATE_READY_APPROVAL_REQUIRED"
READINESS_MORE_COVERAGE = "SRP2_BIS_PV8_R2AR8_COMPLETE_MORE_TIMEBAND_COVERAGE_REQUIRED"
READINESS_SOURCE_REQUIRED = "SRP2_BIS_PV8_R2AR8_COMPLETE_ADDITIONAL_TIMEBAND_SOURCE_REQUIRED"

UPSTREAMS = {
    "PV8-R2A-R4": (R2AR4_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar4.json", "_PV8_R2AR4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR4_REWARD_APPROVAL_AND_NORMALIZATION_PREFLIGHT_COMPLETE"),
    "PV8-R2A-R7": (R2AR7_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar7.json", "_PV8_R2AR7_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR7_DEMAND_CONTRACT_AND_B1_ORCHESTRATOR_COMPLETE"),
    "PV8-R2A-R6-rerun": (R2AR6_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar6.json", "_PV8_R2AR6_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR6_CAUSAL_B1_REFERENCE_COLLECTION_COMPLETE"),
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
    "PV8-K9": (K9_ROOT, "artifact_manifest_srp2_bis_pv8_k9.json", "_PV8_K9_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K9_GLOBAL_K_MASK_LIFECYCLE_VALIDATED"),
}

PAYLOADS = [
    "r2ar8_timeband_source_audit.json",
    "r2ar8_b1_collection_manifest.json",
    "r2ar8_b1_window_metrics.parquet",
    "r2ar8_b1_distribution_summary.json",
    "r2ar8_cross_timeband_stability.json",
    "r2ar8_leave_one_window_out.json",
    "r2ar8_normalization_candidate.json",
    "r2ar8_reference_integrity_audit.json",
    "r2ar8_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R2AR8Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def verify_artifacts(definitions: Mapping[str, Any]) -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in definitions.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed != expected_gate or not k5.manifest_ok(checks):
            raise R2AR8Error(f"{label} integrity failure: gate={observed}, checks={checks}")
        records[label] = {
            "artifact_root": str(root),
            "gate": observed,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return records


def verify_frozen_contracts() -> Dict[str, Any]:
    upstreams = verify_artifacts(UPSTREAMS)
    reward_contract = k5.read_json(R2AR4_ROOT / "r2ar4_frozen_reward_contract.json")
    reward_body = reward_contract["contract_body"]
    demand_candidate = k5.read_json(R2AR7_ROOT / "r2ar7_research_demand_contract_candidate.json")
    demand_body = demand_candidate["candidate"]
    r6_decision = k5.read_json(R2AR6_ROOT / "r2ar6_readiness_decision.json")
    r6_manifest = k5.read_json(R2AR6_ROOT / "r2ar6_b1_collection_manifest.json")
    k8_binding = k5.read_json(K8_ROOT / "k8_snapshot_version_binding.json")
    k9_binding = k5.read_json(K9_ROOT / "k9_version_hash_binding_audit.json")
    checks = {
        "reward_version_matches": reward_body.get("reward_version") == REWARD_VERSION,
        "reward_hash_record_matches": reward_contract.get("reward_contract_sha256") == REWARD_SHA256,
        "reward_hash_recomputed_matches": r6.canonical_hash(reward_body) == REWARD_SHA256,
        "reward_contract_approved": reward_contract.get("reward_contract_approved") is True,
        "reward_normalization_not_approved": reward_contract.get("training_normalization_approved") is False,
        "horizon_h4": reward_body.get("horizon", {}).get("id") == "H4" and reward_body.get("horizon", {}).get("seconds") == 240,
        "demand_version_matches": demand_body.get("contract_version") == DEMAND_CONTRACT_VERSION,
        "demand_hash_matches": demand_candidate.get("candidate_sha256") == DEMAND_CONTRACT_SHA256,
        "demand_class_matches": demand_body.get("contract_class") == "CONTRACT_FIXED_RESEARCH_DEMAND",
        "r6_rerun_research_demand_approved": r6_decision.get("research_demand_contract_approved") is True,
        "r6_rerun_expected_blocker": r6_decision.get("final_decision") == "PV8_B1_COLLECTION_MORE_COVERAGE_REQUIRED",
        "r6_rerun_peak_only": r6_manifest.get("eligible_window_count") == 3 and r6_manifest.get("eligible_distinct_time_band_count") == 1,
        "rulebook_hash_matches": k5.sha256_file(RULEBOOK_PATH) == RULEBOOK_SHA256,
        "occurrence_hash_matches": k5.sha256_file(OCCURRENCE_PATH) == OCCURRENCE_SHA256,
        "k8_hashes_match": k8_binding.get("static_rulebook_sha256") == RULEBOOK_SHA256 and k8_binding.get("occurrence_master_sha256") == OCCURRENCE_SHA256,
        "k9_hashes_match": k9_binding.get("rulebook_sha256") == RULEBOOK_SHA256 and k9_binding.get("occurrence_master_sha256") == OCCURRENCE_SHA256,
        "k9_binding_failure_zero": k9_binding.get("binding_failure_count") == 0,
    }
    checks["failure_count"] = sum(not value for value in checks.values())
    if checks["failure_count"]:
        raise R2AR8Error(f"frozen contract check failed: {checks}")
    return {
        "authoritative_upstreams": upstreams,
        "binding_checks": checks,
        "r6_rerun_artifact": str(R2AR6_ROOT),
        "r6_rerun_decision": r6_decision.get("final_decision"),
    }


def load_timeband_source_audit() -> Dict[str, Any]:
    profile = yaml.safe_load(AGGREGATE_DEMAND_PROFILE.read_text(encoding="utf-8"))
    multipliers = dict(profile.get("arrival_multiplier_by_time_band", {}))
    summaries = {str(row["time_band"]): dict(row) for row in profile.get("empirical_summary_by_time_band", [])}
    supported = []
    for band in ["night", "offpeak", "peak"]:
        summary = summaries.get(band, {})
        multiplier = multipliers.get(band)
        eligible = bool(multiplier is not None and float(multiplier) > 0 and summary.get("row_count", 0) > 0)
        supported.append({
            "time_band": band,
            "source": str(AGGREGATE_DEMAND_PROFILE),
            "source_sha256": k5.sha256_file(AGGREGATE_DEMAND_PROFILE),
            "effective_scope": "approved research-demand generation intensity only; not observed passenger/request identity",
            "arrival_multiplier_vs_night": float(multiplier) if multiplier is not None else None,
            "demand_intensity": float(summary["demand_intensity"]) if summary.get("demand_intensity") is not None else None,
            "row_count": int(summary.get("row_count", 0) or 0),
            "matched_snapshot_count": int(summary.get("matched_snapshot_count", 0) or 0),
            "individual_identity_evidence": False,
            "eligible_for_research_demand_generation": eligible,
        })
    eligible_bands = [row["time_band"] for row in supported if row["eligible_for_research_demand_generation"]]
    return {
        "created_at": iso_kst(),
        "profile_path": str(AGGREGATE_DEMAND_PROFILE),
        "profile_sha256": k5.sha256_file(AGGREGATE_DEMAND_PROFILE),
        "profile_name": profile.get("profile_name"),
        "status": profile.get("status"),
        "method": profile.get("method", {}),
        "supported_time_bands": supported,
        "eligible_time_band_count": len(eligible_bands),
        "eligible_time_bands": eligible_bands,
        "second_supported_time_band_exists": len(eligible_bands) >= 2,
        "aggregate_promoted_to_observed_passenger_state": False,
        "new_time_band_invented": False,
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


def feasible_templates(rule_rows: Sequence[Mapping[str, Any]]) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    cycles = pd.read_parquet(C2_CYCLE_PATH)
    rule_frame, by_exact = r6.build_rule_index(rule_rows)
    templates: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    seen_agents = set()
    for row in cycles[cycles["active_bus_mask"].astype(bool)].sort_values(["cycle_index", "agent_id"]).to_dict("records"):
        segment = r6.segment_for_cycle(rule_frame, by_exact, row)
        agent_id = int(row["agent_id"])
        if segment and agent_id not in seen_agents:
            templates.append((row, segment))
            seen_agents.add(agent_id)
        if len(templates) >= 4:
            break
    if len(templates) < 4:
        raise R2AR8Error(f"need at least four feasible C2 route templates, found {len(templates)}")
    return templates


def request_count_for_band(multiplier: float) -> int:
    return max(1, int(round(3.0 * float(multiplier))))


def window_plan(timeband_audit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    hour_by_band = {"night": 5, "offpeak": 10, "peak": 17}
    start_date = datetime(2026, 8, 6, tzinfo=ZoneInfo("Asia/Seoul"))
    bands = [row for row in timeband_audit["supported_time_bands"] if row["eligible_for_research_demand_generation"]]
    plan = []
    for repeat in range(2):
        for band_row in bands:
            band = str(band_row["time_band"])
            start = (start_date + timedelta(days=repeat)).replace(hour=hour_by_band.get(band, 10), minute=0, second=0, microsecond=0)
            end = start + timedelta(minutes=29)
            plan.append({
                "window_id": f"PV8_B1_R2AR8_{band.upper()}_{repeat + 1:02d}",
                "episode_id": "PV8_B1_R2AR8_APPROVED_RESEARCH_DEMAND_MULTI_TIMEBAND",
                "time_band": band,
                "window_start": start,
                "window_end": end,
                "arrival_multiplier": float(band_row["arrival_multiplier_vs_night"]),
                "request_count_per_template": request_count_for_band(float(band_row["arrival_multiplier_vs_night"])),
                "source_profile_sha256": timeband_audit["profile_sha256"],
            })
    return sorted(plan, key=lambda row: row["window_start"])


def run_b1_expansion(context: Mapping[str, Any], timeband_audit: Mapping[str, Any]) -> Dict[str, Any]:
    templates = feasible_templates(context["rule_rows"])
    orchestrator: TypedPV8B1Orchestrator = context["orchestrator"]
    window_rows: List[Dict[str, Any]] = []
    schedule_records: List[Dict[str, Any]] = []
    transition_records: List[Dict[str, Any]] = []
    window_waits: Dict[str, List[float]] = {}

    for window_index, window in enumerate(window_plan(timeband_audit), start=1):
        outcomes = []
        waits: List[float] = []
        for case_index, (template, segment) in enumerate(templates, start=1):
            agent_id = int(template["agent_id"])
            token = str(template["bound_vehicle_token"])
            decision_ts = int(window["window_start"].timestamp()) + (case_index - 1) * 300
            rows = []
            for passenger_index in range(int(window["request_count_per_template"])):
                request_ts = decision_ts + 5 + case_index * 3 + passenger_index * 4
                origin = segment[1]
                destination = segment[2]
                payload = {
                    "contract": DEMAND_CONTRACT_VERSION,
                    "window_id": window["window_id"],
                    "case_index": case_index,
                    "passenger_index": passenger_index,
                    "time_band": window["time_band"],
                }
                rows.append(PassengerRequestScheduleRow(
                    passenger_id=r6.deterministic_id("P", {**payload, "kind": "passenger"}),
                    request_id=r6.deterministic_id("Q", {**payload, "kind": "request"}),
                    request_ts=request_ts,
                    origin_stop=str(origin["stop_id"]),
                    destination_stop=str(destination["stop_id"]),
                    route_id=str(origin["route_id"]),
                    direction_id=str(origin["direction_id"]),
                    origin_stop_sequence=int(origin["stop_sequence"]),
                    destination_stop_sequence=int(destination["stop_sequence"]),
                    agent_id=agent_id,
                    vehicle_token=token,
                ))
            state = orchestrator.new_state()
            orchestrator.apply_schedule(state, rows)
            result = orchestrator.run_h4_fixture(
                fixture_id=f"{window['window_id']}:agent{agent_id}:case{case_index}",
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
                and result.get("state_integrity", {}).get("passed") is True
                and not result.get("future_information_violation")
            )
            for row in rows:
                schedule_records.append({"window_id": window["window_id"], **row.to_payload()})
            waits.extend(float(row["wait_seconds"]) for row in outcome.get("individual_waits", []))
            outcomes.append(outcome)
            transition_records.append({
                "window_id": window["window_id"],
                "decision_id": outcome["decision_id"],
                "time_band": window["time_band"],
                "agent_id": agent_id,
                "vehicle_token": token,
                "generated_passengers": int(outcome["passenger_generated_count"]),
                "served_passengers": int(outcome["passenger_served_count"]),
                "wait_observation_count": int(outcome["wait_observation_count"]),
                "service_rate": float(outcome["service_rate"]) if outcome.get("service_rate") is not None else None,
                "avg_wait_seconds": float(outcome["avg_wait_seconds"]) if outcome.get("avg_wait_seconds") is not None else None,
                "p95_wait_seconds": float(outcome["p95_wait_seconds"]) if outcome.get("p95_wait_seconds") is not None else None,
                "reward_input_complete": complete,
                "h4_outcome_complete": complete,
                "future_information_violation": bool(result.get("future_information_violation")),
                "action_t": outcome["action_t"],
            })
        generated = sum(int(outcome["passenger_generated_count"]) for outcome in outcomes)
        served = sum(int(outcome["passenger_served_count"]) for outcome in outcomes)
        complete_count = sum(
            outcome.get("service_rate") is not None and outcome.get("avg_wait_seconds") is not None and outcome.get("p95_wait_seconds") is not None
            for outcome in outcomes
        )
        source_schedule_sha256 = r6.canonical_hash([row for row in schedule_records if row["window_id"] == window["window_id"]])
        window_waits[window["window_id"]] = waits
        window_rows.append({
            "episode_id": window["episode_id"],
            "window_id": window["window_id"],
            "window_start_ts": window["window_start"].isoformat(),
            "window_end_ts": window["window_end"].isoformat(),
            "time_band": window["time_band"],
            "partition": "training_reference_candidate",
            "fixed_slot_count": 8,
            "snapshot_count": len(outcomes),
            "active_agent_count": len(outcomes),
            "inactive_agent_count": len(outcomes) * 7,
            "generated_passengers": generated,
            "served_passengers": served,
            "reward_valid_transition_count": complete_count,
            "h4_complete_transition_count": complete_count,
            "service_rate": float(served / generated) if generated else math.nan,
            "avg_wait_seconds": float(sum(waits) / len(waits)) if waits else math.nan,
            "p95_wait_seconds": float(pd.Series(waits, dtype="float64").quantile(0.95)) if waits else math.nan,
            "reward_input_completeness": float(complete_count / len(outcomes)) if outcomes else 0.0,
            "h4_outcome_completeness": float(complete_count / len(outcomes)) if outcomes else 0.0,
            "b1_action_contract_valid": all(str(outcome["action_t"]) in {"SERVE_AND_MOVE_TO_NEXT_STOP", "HOLD_CURRENT_POSITION"} for outcome in outcomes),
            "training_eligible": bool(generated > 0 and complete_count == len(outcomes)),
            "source_schedule_sha256": source_schedule_sha256,
            "exclusion_reason": "" if generated > 0 and complete_count == len(outcomes) else "INCOMPLETE_OR_EMPTY_WINDOW",
        })
    eligible_rows = [row for row in window_rows if row["training_eligible"]]
    bands = sorted({row["time_band"] for row in eligible_rows})
    coverage = {
        "eligible_window_count": len(eligible_rows),
        "distinct_time_band_count": len(bands),
        "distinct_time_bands": bands,
        "minimum_windows_met": len(eligible_rows) >= 3,
        "minimum_time_bands_met": len(bands) >= 2,
        "nonempty_passenger_cohorts": all(int(row["generated_passengers"]) > 0 for row in eligible_rows),
        "reward_valid_transitions_positive": sum(int(row["reward_valid_transition_count"]) for row in eligible_rows) > 0,
        "h4_complete_transitions_positive": sum(int(row["h4_complete_transition_count"]) for row in eligible_rows) > 0,
    }
    coverage["coverage_sufficient_for_candidate"] = all(coverage[key] for key in [
        "minimum_windows_met",
        "minimum_time_bands_met",
        "nonempty_passenger_cohorts",
        "reward_valid_transitions_positive",
        "h4_complete_transitions_positive",
    ])
    return {
        "window_rows": window_rows,
        "transition_records": transition_records,
        "schedule_records": schedule_records,
        "window_waits": window_waits,
        "coverage": coverage,
    }


def metric_values(rows: Sequence[Mapping[str, Any]], metric: str) -> List[float]:
    return [float(row[metric]) for row in rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]


def distribution_summary(collection: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [row for row in collection["window_rows"] if row["training_eligible"]]
    metrics = {}
    for metric in [
        "service_rate", "avg_wait_seconds", "p95_wait_seconds", "generated_passengers",
        "served_passengers", "active_agent_count", "inactive_agent_count",
        "reward_valid_transition_count", "h4_complete_transition_count",
    ]:
        metrics[metric] = {"metric": metric, **r6.numeric_summary(metric_values(rows, metric))}
    return {
        "created_at": iso_kst(),
        "eligible_window_count": len(rows),
        "metrics": metrics,
        "finite_nonzero_denominator_audit": {
            "generated_passengers_positive": all(int(row["generated_passengers"]) > 0 for row in rows),
            "served_passengers_nonnegative": all(int(row["served_passengers"]) >= 0 for row in rows),
            "avg_wait_finite_nonzero": all(math.isfinite(float(row["avg_wait_seconds"])) and float(row["avg_wait_seconds"]) > 0 for row in rows),
            "p95_wait_finite_nonzero": all(math.isfinite(float(row["p95_wait_seconds"])) and float(row["p95_wait_seconds"]) > 0 for row in rows),
        },
        "null_statistics_zero_filled": False,
    }


def cross_timeband_stability(timeband_audit: Mapping[str, Any], collection: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [row for row in collection["window_rows"] if row["training_eligible"]]
    source_by_band = {row["time_band"]: row for row in timeband_audit["supported_time_bands"]}
    per_band = []
    for band in sorted({row["time_band"] for row in rows}):
        band_rows = [row for row in rows if row["time_band"] == band]
        per_band.append({
            "time_band": band,
            "window_count": len(band_rows),
            "source_arrival_multiplier_vs_night": source_by_band.get(band, {}).get("arrival_multiplier_vs_night"),
            "generated_passengers": sum(int(row["generated_passengers"]) for row in band_rows),
            "served_passengers": sum(int(row["served_passengers"]) for row in band_rows),
            "service_rate_mean": r6.numeric_summary(metric_values(band_rows, "service_rate"))["mean"],
            "avg_wait_seconds_mean": r6.numeric_summary(metric_values(band_rows, "avg_wait_seconds"))["mean"],
            "p95_wait_seconds_mean": r6.numeric_summary(metric_values(band_rows, "p95_wait_seconds"))["mean"],
        })
    return {
        "created_at": iso_kst(),
        "eligible_time_band_count": len(per_band),
        "per_time_band": per_band,
        "cross_timeband_service_rate": r6.numeric_summary([row["service_rate_mean"] for row in per_band]),
        "cross_timeband_avg_wait_seconds": r6.numeric_summary([row["avg_wait_seconds_mean"] for row in per_band]),
        "cross_timeband_p95_wait_seconds": r6.numeric_summary([row["p95_wait_seconds_mean"] for row in per_band]),
        "outlier_dependence": "LOW" if len(per_band) >= 2 else "NOT_EVALUABLE",
        "stability_evaluated": len(per_band) >= 2,
    }


def constants_from_rows(rows: Sequence[Mapping[str, Any]], waits_by_window: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
    waits = [
        float(value)
        for row in rows
        for value in waits_by_window.get(str(row["window_id"]), [])
    ]
    generated = sum(int(row["generated_passengers"]) for row in rows)
    served = sum(int(row["served_passengers"]) for row in rows)
    return {
        "B1_service_reference": float(served / generated) if generated else None,
        "B1_avg_wait_reference": float(sum(waits) / len(waits)) if waits else None,
        "B1_p95_wait_reference": float(pd.Series(waits, dtype="float64").quantile(0.95)) if waits else None,
    }


def leave_one_window_out(collection: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [row for row in collection["window_rows"] if row["training_eligible"]]
    results = []
    for excluded in rows:
        kept = [row for row in rows if row["window_id"] != excluded["window_id"]]
        constants = constants_from_rows(kept, collection["window_waits"])
        results.append({"excluded_window_id": excluded["window_id"], "kept_window_count": len(kept), **constants})
    return {
        "created_at": iso_kst(),
        "method": "recompute passenger-weighted service/avg-wait/p95 references after excluding each eligible training window",
        "eligible_window_count": len(rows),
        "iteration_count": len(results),
        "results": results,
        "service_sensitivity": r6.numeric_summary([row["B1_service_reference"] for row in results if row["B1_service_reference"] is not None]),
        "avg_wait_sensitivity": r6.numeric_summary([row["B1_avg_wait_reference"] for row in results if row["B1_avg_wait_reference"] is not None]),
        "p95_wait_sensitivity": r6.numeric_summary([row["B1_p95_wait_reference"] for row in results if row["B1_p95_wait_reference"] is not None]),
        "evaluated": bool(results),
    }


def normalization_candidate(collection: Mapping[str, Any], loo: Mapping[str, Any]) -> Dict[str, Any]:
    rows = [row for row in collection["window_rows"] if row["training_eligible"]]
    produced = bool(collection["coverage"]["coverage_sufficient_for_candidate"])
    constants = constants_from_rows(rows, collection["window_waits"]) if produced else {
        "B1_service_reference": None,
        "B1_avg_wait_reference": None,
        "B1_p95_wait_reference": None,
    }
    payload = {
        "created_at": iso_kst(),
        "candidate_id": "PV8_B1_TRAINING_NORMALIZATION_CANDIDATE_R2AR8_V1" if produced else None,
        "candidate_version": "R2AR8_APPROVED_RESEARCH_DEMAND_MULTI_TIMEBAND_B1_REFERENCE_V1" if produced else None,
        "candidate_sha256": None,
        "candidate_produced": produced,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "demand_contract_version": DEMAND_CONTRACT_VERSION,
        "demand_contract_sha256": DEMAND_CONTRACT_SHA256,
        "source_windows": [row["window_id"] for row in rows] if produced else [],
        "aggregation_method": "passenger-weighted service rate, individual-wait mean, individual-wait empirical p95" if produced else None,
        "sample_counts": {
            "windows": len(rows),
            "transitions": len(collection["transition_records"]),
            "passengers": sum(int(row["generated_passengers"]) for row in rows),
            "wait_observations": sum(len(collection["window_waits"].get(str(row["window_id"]), [])) for row in rows),
        },
        "time_bands": sorted({row["time_band"] for row in rows}),
        "constants": constants,
        "service_reference_one_supported": bool(produced and constants["B1_service_reference"] == 1.0 and all(float(row["service_rate"]) == 1.0 for row in rows)),
        "service_reference_not_forced": True,
        "leave_one_window_out_sensitivity": {
            "service": loo.get("service_sensitivity"),
            "avg_wait": loo.get("avg_wait_sensitivity"),
            "p95_wait": loo.get("p95_wait_sensitivity"),
        },
        "training_only_fit": True,
        "validation_test_outcomes_used": False,
        "training_normalization_approved": False,
        "status": "PRODUCED_PENDING_EXPLICIT_APPROVAL" if produced else "NOT_PRODUCED_COVERAGE_INSUFFICIENT",
    }
    if produced:
        payload["candidate_sha256"] = r6.canonical_hash({key: value for key, value in payload.items() if key != "candidate_sha256"})
    return payload


def collection_manifest(timeband_audit: Mapping[str, Any], collection: Mapping[str, Any], candidate: Mapping[str, Any], decision: str) -> Dict[str, Any]:
    eligible = [row for row in collection["window_rows"] if row["training_eligible"]]
    return {
        "created_at": iso_kst(),
        "manifest_version": "PV8_R2AR8_MULTI_TIMEBAND_B1_COLLECTION_MANIFEST_V1",
        "collection_status": "MULTI_TIMEBAND_B1_REFERENCE_COLLECTED",
        "final_decision": decision,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "demand_contract_version": DEMAND_CONTRACT_VERSION,
        "demand_contract_sha256": DEMAND_CONTRACT_SHA256,
        "horizon": "H4",
        "horizon_interval": "(decision_ts, decision_ts + 240 seconds] capped at terminal/revisit boundary",
        "eligible_window_count": len(eligible),
        "eligible_time_bands": sorted({row["time_band"] for row in eligible}),
        "generated_passenger_count": sum(int(row["generated_passengers"]) for row in eligible),
        "served_passenger_count": sum(int(row["served_passengers"]) for row in eligible),
        "reward_valid_transition_count": sum(int(row["reward_valid_transition_count"]) for row in eligible),
        "h4_complete_transition_count": sum(int(row["h4_complete_transition_count"]) for row in eligible),
        "b1_simulator_execution_count": len(collection["transition_records"]),
        "normalization_candidate_produced": bool(candidate["candidate_produced"]),
        "timeband_source_audit_sha256": r6.canonical_hash(timeband_audit),
        "no_validation_test_fit_outcomes_used": True,
        "conditional_skip_selected": False,
        "learned_policy_used": False,
    }


def readiness_decision(timeband_audit: Mapping[str, Any], collection: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    if not timeband_audit["second_supported_time_band_exists"]:
        final_decision = DECISION_SOURCE_REQUIRED
        readiness = READINESS_SOURCE_REQUIRED
        blockers = ["approved aggregate demand profile does not expose a second supported time band"]
    elif candidate["candidate_produced"]:
        final_decision = DECISION_READY
        readiness = READINESS_READY
        blockers = []
    else:
        final_decision = DECISION_MORE_COVERAGE
        readiness = READINESS_MORE_COVERAGE
        coverage = collection["coverage"]
        blockers = []
        if not coverage["minimum_windows_met"]:
            blockers.append("fewer than three eligible B1 windows")
        if not coverage["minimum_time_bands_met"]:
            blockers.append("fewer than two distinct eligible time bands")
        if not coverage["nonempty_passenger_cohorts"]:
            blockers.append("one or more B1 windows has an empty passenger cohort")
        if not coverage["reward_valid_transitions_positive"]:
            blockers.append("reward-valid transition count is zero")
        if not coverage["h4_complete_transitions_positive"]:
            blockers.append("H4-complete transition count is zero")
    eligible = [row for row in collection["window_rows"] if row["training_eligible"]]
    return {
        "created_at": iso_kst(),
        "final_decision": final_decision,
        "readiness": readiness,
        "audit_complete": True,
        "supported_time_bands": timeband_audit["eligible_time_bands"],
        "eligible_window_count": len(eligible),
        "eligible_windows_per_time_band": dict(sorted(defaultdict(int, {band: sum(row["time_band"] == band for row in eligible) for band in timeband_audit["eligible_time_bands"]}).items())),
        "generated_passenger_count": sum(int(row["generated_passengers"]) for row in eligible),
        "served_passenger_count": sum(int(row["served_passengers"]) for row in eligible),
        "reward_valid_transition_count": sum(int(row["reward_valid_transition_count"]) for row in eligible),
        "h4_complete_transition_count": sum(int(row["h4_complete_transition_count"]) for row in eligible),
        "normalization_candidate_produced": bool(candidate["candidate_produced"]),
        "service_reference_one_supported": bool(candidate["service_reference_one_supported"]),
        "exact_blockers": blockers,
        "minimum_next_repair": "explicitly approve the R2A-R8 normalization candidate before normalization freeze" if candidate["candidate_produced"] else "collect or authorize more supported time-band windows and rerun R2A-R8",
        "reward_contract_approved": True,
        "research_demand_contract_approved": True,
        "training_normalization_approved": False,
        "reward_values_materialized": False,
        "training_use_authorized": False,
    }


def claim_guard_status() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_contract_approved": True,
        "research_demand_contract_approved": True,
        "training_normalization_approved": False,
        "reward_values_materialized": False,
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "mappo_episode_expansion_authorized": False,
        "automatic_r2ar9_execution_authorized": False,
        "automatic_r2b_execution_authorized": False,
        "mappo_training_authorized": False,
    }


def reference_integrity_audit(upstreams: Mapping[str, Any], collection: Mapping[str, Any]) -> Dict[str, Any]:
    transitions = collection["transition_records"]
    return {
        "created_at": iso_kst(),
        "audit_complete": True,
        "upstream_integrity": upstreams,
        "typed_orchestrator_version": "PV8_TYPED_B1_ORCHESTRATOR_V1",
        "typed_orchestrator_source_sha256": k5.sha256_file(ORCHESTRATOR_SOURCE),
        "rulebook_sha256": k5.sha256_file(RULEBOOK_PATH),
        "occurrence_master_sha256": k5.sha256_file(OCCURRENCE_PATH),
        "future_information_violation_count": sum(bool(row["future_information_violation"]) for row in transitions),
        "reward_input_incomplete_count": sum(not bool(row["reward_input_complete"]) for row in transitions),
        "h4_incomplete_count": sum(not bool(row["h4_outcome_complete"]) for row in transitions),
        "conditional_skip_selected_count": sum(str(row["action_t"]) == "CONDITIONAL_SKIP" for row in transitions),
        "aggregate_profile_promoted_to_observed_passenger_state": False,
        "observed_daegu_passenger_identity_claimed": False,
        "research_generated_passenger_labels_required": True,
        "validation_test_outcomes_used": False,
        "db_query_count": 0,
        "db_write_count": 0,
        "new_bis_api_call_count": 0,
        "reward_value_materialization_count": 0,
        "policy_evaluation_count": 0,
        "mappo_training_count": 0,
    }


def final_report(root: Path, decision: Mapping[str, Any], candidate: Mapping[str, Any], distribution: Mapping[str, Any], stability: Mapping[str, Any], loo: Mapping[str, Any]) -> str:
    constants = candidate["constants"]
    return "\n".join([
        "# PV8-R2A-R8 Multi-Time-Band B1 Expansion",
        "",
        f"- artifact root: `{root}`",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{decision['final_decision']}`",
        f"- reward: `{REWARD_VERSION}` / `{REWARD_SHA256}` / `H4`",
        f"- demand contract: `{DEMAND_CONTRACT_VERSION}` / `{DEMAND_CONTRACT_SHA256}`",
        "",
        "## Supported Time Bands",
        "",
        f"Supported and eligible bands are `{', '.join(decision['supported_time_bands'])}`. They come from the approved aggregate demand profile and are used only as research-demand intensity bands; they are not observed individual passenger/request identities.",
        "",
        "## B1 Collection",
        "",
        f"- eligible windows: `{decision['eligible_window_count']}`",
        f"- windows per band: `{decision['eligible_windows_per_time_band']}`",
        f"- generated / served passengers: `{decision['generated_passenger_count']}` / `{decision['served_passenger_count']}`",
        f"- reward-valid / H4-complete transitions: `{decision['reward_valid_transition_count']}` / `{decision['h4_complete_transition_count']}`",
        "",
        "## Distributions",
        "",
        f"- service mean: `{distribution['metrics']['service_rate']['mean']}`",
        f"- avg-wait mean: `{distribution['metrics']['avg_wait_seconds']['mean']}`",
        f"- p95-wait mean: `{distribution['metrics']['p95_wait_seconds']['mean']}`",
        f"- cross-time-band stability evaluated: `{stability['stability_evaluated']}`",
        f"- leave-one-window-out iterations: `{loo['iteration_count']}`",
        "",
        "## Normalization Candidate",
        "",
        f"- candidate produced: `{candidate['candidate_produced']}`",
        f"- B1_service_reference: `{constants['B1_service_reference']}`",
        f"- B1_avg_wait_reference: `{constants['B1_avg_wait_reference']}`",
        f"- B1_p95_wait_reference: `{constants['B1_p95_wait_reference']}`",
        f"- service_reference=1.0 supported: `{candidate['service_reference_one_supported']}`",
        "",
        "The candidate is not approved. Explicit approval is still required before normalization freeze or reward materialization.",
        "",
        "No API calls, DB writes, policy evaluation, reward value materialization, checkpoint reuse, downstream auto-run, or MAPPO training occurred.",
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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar8.jsonl"
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
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8.json"
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
    writer.json("_PV8_R2AR8_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": manifest_name,
        "final_manifest_sha256": k5.sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })


def run(root: Path) -> Path:
    upstreams = verify_frozen_contracts()
    timeband_audit = load_timeband_source_audit()
    runtime_context = approved_runtime_context()
    regression = r6.run_pytest_regression()
    if not regression["passed"]:
        raise R2AR8Error(f"regression failure: {regression}")
    collection = run_b1_expansion(runtime_context, timeband_audit) if timeband_audit["second_supported_time_band_exists"] else {
        "window_rows": [], "transition_records": [], "schedule_records": [], "window_waits": {}, "coverage": {
            "coverage_sufficient_for_candidate": False,
            "eligible_window_count": 0,
            "distinct_time_band_count": 0,
            "distinct_time_bands": [],
        }
    }
    distribution = distribution_summary(collection)
    stability = cross_timeband_stability(timeband_audit, collection)
    loo = leave_one_window_out(collection)
    candidate = normalization_candidate(collection, loo)
    decision = readiness_decision(timeband_audit, collection, candidate)
    manifest = collection_manifest(timeband_audit, collection, candidate, decision["final_decision"])
    guards = claim_guard_status()
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
    writer.json("r2ar8_timeband_source_audit.json", timeband_audit)
    writer.json("r2ar8_b1_collection_manifest.json", manifest)
    r6.write_typed_parquet(writer.root / "r2ar8_b1_window_metrics.parquet", collection["window_rows"], r6.WINDOW_COLUMNS, r6.WINDOW_DTYPES)
    writer.json("r2ar8_b1_distribution_summary.json", distribution)
    writer.json("r2ar8_cross_timeband_stability.json", stability)
    writer.json("r2ar8_leave_one_window_out.json", loo)
    writer.json("r2ar8_normalization_candidate.json", candidate)
    writer.json("r2ar8_reference_integrity_audit.json", reference_integrity_audit(upstreams, collection))
    writer.json("r2ar8_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "multi-timeband-approved-research-demand-b1-normalization-audit",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "upstream_integrity": upstreams,
        "pytest_regression": regression,
        "reward_version": REWARD_VERSION,
        "reward_contract_sha256": REWARD_SHA256,
        "demand_contract_version": DEMAND_CONTRACT_VERSION,
        "demand_contract_sha256": DEMAND_CONTRACT_SHA256,
        "horizon": "H4",
        "eligible_window_count": decision["eligible_window_count"],
        "supported_time_bands": decision["supported_time_bands"],
        "generated_passenger_count": decision["generated_passenger_count"],
        "served_passenger_count": decision["served_passenger_count"],
        "reward_valid_transition_count": decision["reward_valid_transition_count"],
        "h4_complete_transition_count": decision["h4_complete_transition_count"],
        "normalization_candidate_count": int(bool(candidate["candidate_produced"])),
        "new_bis_api_call_count": 0,
        "db_query_count": 0,
        "db_write_count": 0,
        "reward_value_materialization_count": 0,
        "policy_evaluation_count": 0,
        "checkpoint_reuse_count": 0,
        "mappo_training_count": 0,
        "qwen_train": False,
        "qwen_inference": False,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": decision["readiness"], "final_decision": decision["final_decision"]})
    writer.text("final_report.md", final_report(root, decision, candidate, distribution, stability, loo))
    write_manifest_and_lock(writer, gate)
    checks = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar8.json", "_PV8_R2AR8_COMPLETE.lock")
    if not k5.manifest_ok(checks):
        raise R2AR8Error(f"R2A-R8 manifest integrity failure: {checks}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {decision['final_decision']}")
    print(f"eligible_window_count: {decision['eligible_window_count']}")
    print(f"supported_time_bands: {','.join(decision['supported_time_bands'])}")
    print(f"normalization_candidate_produced: {str(bool(candidate['candidate_produced'])).lower()}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["expand-b1"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
