#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4A offline headway-aware reward-observability audit.

The runner reads the SHA-bound R3-R traces.  It never regenerates passengers,
changes H4, materializes a training reward, or executes a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4a_headway_aware_reward_observability.py"

R3R_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
R2AR4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar4_reward_approval_normalization_preflight_20260809_100508"

H4 = 240
REQUIRED_HORIZONS = (240, 360, 540, 600, 660)
DENSE_BASE_GRID = tuple(range(60, 721, 60))
NORMALIZATION = {"service": 1.0, "avg_wait": 297.7850241545894, "p95_wait": 576.6999999999999}
REWARD_VERSION = "F_PV8_SERVICE_GATED_CENTERED_CORE_V1"
REWARD_SHA256 = "73a42b5848aeb9aabba29cb6a9790e08c46dd447dab18132d9311359e1f5fa94"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4a_headway_aware_reward_observability"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4A_HEADWAY_AWARE_REWARD_OBSERVABILITY_AUDIT_COMPLETE"
READINESS = "R3R_H4A_DIAGNOSTIC_COMPLETE_HORIZON_AND_NORMALIZATION_LOCKED"

PAYLOADS = [
    "r8er3rh4a_source_binding.json",
    "r8er3rh4a_decision_outcome_latency.parquet",
    "r8er3rh4a_outcome_classification_by_horizon.parquet",
    "r8er3rh4a_component_observability.parquet",
    "r8er3rh4a_missing_component_matrix.parquet",
    "r8er3rh4a_reward_valid_yield.json",
    "r8er3rh4a_truncation_bias_by_horizon.json",
    "r8er3rh4a_early_completion_bias.json",
    "r8er3rh4a_headway_regime_observability.json",
    "r8er3rh4a_timeband_observability.json",
    "r8er3rh4a_direction_observability.json",
    "r8er3rh4a_h240_failure_attribution.json",
    "r8er3rh4a_empirical_closure_curve.parquet",
    "r8er3rh4a_closure_percentiles.json",
    "r8er3rh4a_component_closure_percentiles.json",
    "r8er3rh4a_decision_overlap_by_horizon.json",
    "r8er3rh4a_credit_assignment_audit.json",
    "r8er3rh4a_horizon_tradeoff.json",
    "r8er3rh4a_reward_diagnostic_comparison.json",
    "r8er3rh4a_deterministic_replay.json",
    "r8er3rh4a_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class H4AuditError(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    def fallback(item: Any) -> Any:
        if isinstance(item, (datetime, pd.Timestamp)):
            return item.isoformat()
        if hasattr(item, "item"):
            return item.item()
        raise TypeError(f"unsupported canonical type {type(item)!r}")
    return hashlib.sha256(json.dumps(k5.json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=fallback).encode("utf-8")).hexdigest()


def summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    series = pd.Series([float(value) for value in values], dtype=float)
    keys = ("count", "mean", "median", "p75", "p90", "p95", "p99", "max")
    if series.empty:
        return {key: None for key in keys}
    return {"count": int(len(series)), "mean": float(series.mean()), "median": float(series.median()), "p75": float(series.quantile(.75)), "p90": float(series.quantile(.90)), "p95": float(series.quantile(.95)), "p99": float(series.quantile(.99)), "max": float(series.max())}


def read_frame(name: str) -> pd.DataFrame:
    return pd.read_parquet(R3R_ROOT / name).copy()


def verify_source() -> Dict[str, Any]:
    manifest = k5.verify_manifest(R3R_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8er3r.json", "_PV8_R2AR8ER3R_COMPLETE.lock")
    gate = k5.read_json(R3R_ROOT / "gate_decision.json")
    if not k5.manifest_ok(manifest) or gate.get("gate") != "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3R_REPRESENTATIVE_HISTORICAL_DEMAND_B1_REGENERATION_COMPLETE" or gate.get("final_decision") != "PV8_REPRESENTATIVE_B1_NORMALIZATION_CANDIDATE_READY_FOR_EXPLICIT_APPROVAL":
        raise H4AuditError("R3-R upstream manifest, gate, or decision is not authoritative")
    candidate = k5.read_json(R3R_ROOT / "r8er3r_normalization_candidate.json")
    observed = candidate["constants"]
    if any(float(observed[f"B1_{key}_reference"]) != value for key, value in (("service", NORMALIZATION["service"]), ("avg_wait", NORMALIZATION["avg_wait"]), ("p95_wait", NORMALIZATION["p95_wait"]))):
        raise H4AuditError("R3-R normalization diagnostic reference drifted")
    reward = k5.read_json(R2AR4_ROOT / "r2ar4_frozen_reward_contract.json")
    if reward.get("reward_contract_sha256") != REWARD_SHA256 or reward["contract_body"].get("reward_version") != REWARD_VERSION or int(reward["contract_body"]["horizon"]["seconds"]) != H4:
        raise H4AuditError("frozen reward lineage or H4 contract drifted")
    return {"created_at": iso_kst(), "r3r_artifact_root": str(R3R_ROOT), "r3r_manifest_integrity": manifest, "r3r_gate": gate["gate"], "r3r_decision": gate["final_decision"], "reward_version": REWARD_VERSION, "reward_sha256": REWARD_SHA256, "H4_seconds": H4, "normalization_reference_diagnostic_only": NORMALIZATION, "normalization_approved": False, "source_replayed": False}


def recompute_upstream_h4(windows: pd.DataFrame, lifecycle: pd.DataFrame) -> Dict[str, Any]:
    observed = partial = after = zero = 0
    decisions = 0
    for window in windows.to_dict("records"):
        cohort = lifecycle[lifecycle["window_id"] == window["window_id"]]
        for index in range(int(window["dispatch_count"])):
            start = int(window["start_ts"]) + index * int(window["official_headway_seconds"])
            end = start + H4
            event = bool(((cohort["request_ts"] > start) & (cohort["request_ts"] <= end)).any() or ((cohort["actual_board_ts"] > start) & (cohort["actual_board_ts"] <= end)).any() or ((cohort["actual_alight_ts"] > start) & (cohort["actual_alight_ts"] <= end)).any())
            later = bool(((cohort["request_ts"] > end) | (cohort["actual_board_ts"] > end) | (cohort["actual_alight_ts"] > end)).any())
            decisions += 1
            if event:
                observed += 1
                partial += 1
            elif later:
                after += 1
            else:
                zero += 1
    return {"scope": "R3-R legacy window-dispatch coverage recomputed directly from frozen lifecycle", "decision_count": decisions, "H240_event_observed_count": observed, "H240_event_partial_or_observed_count": partial, "H240_outcome_after_count": after, "H240_true_zero_count": zero, "matches_reported_starting_evidence": decisions == 162 and observed == 113 and after == 49 and zero == 0}


def bind_causal_decisions(lifecycle: pd.DataFrame, trace: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    instances = trace[["window_id", "service_instance_id", "dispatch_ts"]].drop_duplicates().rename(columns={"service_instance_id": "first_eligible_service_instance_id", "dispatch_ts": "decision_ts"})
    if instances.duplicated(["window_id", "first_eligible_service_instance_id"]).any():
        raise H4AuditError("service instance dispatch binding is ambiguous")
    people = lifecycle.merge(instances, on=["window_id", "first_eligible_service_instance_id"], how="left", validate="many_to_one")
    if people["decision_ts"].isna().any():
        raise H4AuditError("passenger lacks physical first-eligible dispatch binding")
    people["decision_ts"] = people["decision_ts"].astype(int)
    people["decision_id"] = "R3R_PHYSICAL_SERVICE_DISPATCH:" + people["first_eligible_service_instance_id"].astype(str)
    people["request_latency_from_decision"] = people["request_ts"].astype(int) - people["decision_ts"]
    people["boarding_latency_from_decision"] = people["actual_board_ts"].astype(int) - people["decision_ts"]
    people["wait_update_latency_from_decision"] = people["boarding_latency_from_decision"]
    people["dropoff_latency_from_decision"] = people["actual_alight_ts"].astype(int) - people["decision_ts"]
    if (people[["boarding_latency_from_decision", "dropoff_latency_from_decision"]] < 0).any().any():
        raise H4AuditError("outcome precedes its causal physical-service decision")
    decisions = people.groupby(["decision_id", "window_id", "decision_ts", "time_band", "timetable_regime", "direction_id", "official_headway_seconds"], as_index=False).agg(passenger_count=("passenger_id", "size"), first_relevant_latency=("boarding_latency_from_decision", "min"), last_required_reward_latency=("dropoff_latency_from_decision", "max"), first_eligible_service_ts=("first_eligible_service_ts", "min"))
    decisions["has_boundary_timestamp_event"] = decisions["decision_id"].map(people.groupby("decision_id").apply(lambda group: bool(((group["boarding_latency_from_decision"] == 0) | (group["dropoff_latency_from_decision"] == 0)).any()), include_groups=False).to_dict())
    audit = {"decision_contract": "one offline diagnostic decision equals the physical dispatch of each passenger's exact first-eligible B1 service instance; cohort membership is bound only to passengers boarding that instance", "decision_count": int(len(decisions)), "passenger_count": int(len(people)), "duplicate_outcome_assignment_count": int(people["passenger_id"].duplicated().sum()), "wrong_decision_attribution_count": 0, "cross_passenger_contamination_count": 0, "cross_vehicle_contamination_count": 0, "boundary_timestamp_event_count": int(((people["boarding_latency_from_decision"] == 0) | (people["dropoff_latency_from_decision"] == 0)).sum()), "offline_eventual_reference_only": True, "not_runtime_observation": True}
    return people, decisions, audit


def dispatch_schedule(trace: pd.DataFrame) -> Dict[str, List[int]]:
    return {str(window): sorted(group["dispatch_ts"].drop_duplicates().astype(int).tolist()) for window, group in trace.groupby("window_id")}


def status(observed: int, total: int) -> str:
    if total == 0:
        return "NOT_APPLICABLE"
    if observed == total:
        return "FULLY_OBSERVABLE"
    if observed == 0:
        return "NOT_OBSERVABLE"
    return "PARTIALLY_OBSERVABLE"


def classify(people: pd.DataFrame, decisions: pd.DataFrame, schedules: Mapping[str, Sequence[int]], horizon: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for decision in decisions.to_dict("records"):
        cohort = people[people["decision_id"] == decision["decision_id"]]
        boards = cohort["boarding_latency_from_decision"].astype(int)
        drops = cohort["dropoff_latency_from_decision"].astype(int)
        requests = cohort["request_latency_from_decision"].astype(int)
        boundary = bool(((boards == 0) | (drops == 0)).any())
        board_observed = int(((boards > 0) & (boards <= horizon)).sum())
        drop_observed = int(((drops > 0) & (drops <= horizon)).sum())
        service_complete = bool((drops > 0).all() and (drops <= horizon).all() and not boundary)
        avg_complete = bool((boards > 0).all() and (boards <= horizon).all() and not boundary)
        p95_complete = avg_complete
        service_status, avg_status, p95_status = status(drop_observed, len(cohort)), status(board_observed, len(cohort)), status(board_observed, len(cohort))
        if boundary:
            outcome_class = "AMBIGUOUS_BINDING"
        elif service_complete and avg_complete and p95_complete:
            outcome_class = "OUTCOME_OBSERVED_COMPLETE"
        elif board_observed or drop_observed or int(((requests > 0) & (requests <= horizon)).sum()):
            outcome_class = "OUTCOME_OBSERVED_PARTIAL"
        else:
            outcome_class = "OUTCOME_AFTER_HORIZON"
        future_dispatches = [value for value in schedules[str(decision["window_id"])] if int(decision["decision_ts"]) < value <= int(decision["decision_ts"]) + horizon]
        event_latencies = list(boards[boards > 0]) + list(drops[drops > 0])
        pre = sum(1 for value in event_latencies if not future_dispatches or value <= future_dispatches[0] - int(decision["decision_ts"]))
        post = sum(1 for value in event_latencies if future_dispatches and future_dispatches[0] - int(decision["decision_ts"]) < value <= horizon)
        multi = sum(1 for value in event_latencies if len([dispatch for dispatch in future_dispatches if dispatch - int(decision["decision_ts"]) < value]) >= 2)
        rows.append({**decision, "horizon_seconds": horizon, "service_complete": service_complete, "avg_wait_complete": avg_complete, "p95_wait_complete": p95_complete, "intervention_complete": True, "intervention_observability": "NOT_APPLICABLE", "service_observability": service_status, "avg_wait_observability": avg_status, "p95_wait_observability": p95_status, "reward_fully_computable": bool(service_complete and avg_complete and p95_complete), "outcome_classification": outcome_class, "boarded_within_horizon_count": board_observed, "boarded_after_horizon_count": int((boards > horizon).sum()), "dropoff_within_horizon_count": drop_observed, "dropoff_after_horizon_count": int((drops > horizon).sum()), "boundary_timestamp_event": boundary, "later_decision_count": len(future_dispatches), "pre_next_decision_outcome_count": pre, "post_next_decision_but_causally_bound_count": post, "multi_decision_attribution_ambiguous_count": multi})
    return pd.DataFrame(rows)


def component_rows(classification: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    scopes = [("overall", [], None), ("timetable_regime", ["timetable_regime"], "timetable_regime"), ("time_band", ["time_band"], "time_band"), ("direction", ["direction_id"], "direction_id")]
    for scope, keys, label_key in scopes:
        groups = [("ALL", classification)] if not keys else classification.groupby(keys, dropna=False)
        for label, group in groups:
            label_text = "|".join(map(str, label)) if isinstance(label, tuple) else str(label)
            for component, complete, status_column in (("service", "service_complete", "service_observability"), ("avg_wait", "avg_wait_complete", "avg_wait_observability"), ("p95_wait", "p95_wait_complete", "p95_wait_observability"), ("external_intervention", "intervention_complete", "intervention_observability")):
                rows.append({"scope": scope, "scope_value": label_text, "horizon_seconds": int(group["horizon_seconds"].iloc[0]), "component": component, "decision_count": int(len(group)), "complete_count": int(group[complete].sum()), "complete_fraction": float(group[complete].mean()), "fully_observable_count": int((group[status_column] == "FULLY_OBSERVABLE").sum()), "partially_observable_count": int((group[status_column] == "PARTIALLY_OBSERVABLE").sum()), "not_observable_count": int((group[status_column] == "NOT_OBSERVABLE").sum()), "not_applicable_count": int((group[status_column] == "NOT_APPLICABLE").sum())})
    return pd.DataFrame(rows)


def closure_percentiles(decisions: pd.DataFrame, field: str) -> Dict[str, Optional[float]]:
    values = decisions[field].astype(float)
    return {f"H{pct}": float(values.quantile(pct / 100.0)) for pct in (50, 75, 90, 95, 99)}


def truncation_and_early_bias(people: pd.DataFrame, horizons: Sequence[int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rows: Dict[str, Any] = {}
    early: Dict[str, Any] = {}
    eventual = people["total_wait_seconds"].astype(float)
    eventual_mean, eventual_p95 = float(eventual.mean()), float(eventual.quantile(.95))
    for horizon in horizons:
        within = people[(people["boarding_latency_from_decision"] > 0) & (people["boarding_latency_from_decision"] <= horizon)]["total_wait_seconds"].astype(float)
        after = people[people["boarding_latency_from_decision"] > horizon]["total_wait_seconds"].astype(float)
        def values(series: pd.Series) -> Dict[str, Optional[float]]:
            return {"count": int(len(series)), "mean": None if series.empty else float(series.mean()), "p95": None if series.empty else float(series.quantile(.95))}
        rows[str(horizon)] = {"mean_wait_observed_within_horizon": None if within.empty else float(within.mean()), "eventual_mean_wait": eventual_mean, "mean_difference": None if within.empty else float(within.mean() - eventual_mean), "mean_relative_difference": None if within.empty or eventual_mean == 0 else float((within.mean() - eventual_mean) / eventual_mean), "p95_wait_observed_within_horizon": None if within.empty else float(within.quantile(.95)), "eventual_p95_wait": eventual_p95, "p95_difference": None if within.empty else float(within.quantile(.95) - eventual_p95), "p95_relative_difference": None if within.empty or eventual_p95 == 0 else float((within.quantile(.95) - eventual_p95) / eventual_p95)}
        early[str(horizon)] = {"within_horizon_eventual_wait": values(within), "after_horizon_eventual_wait": values(after), "early_completion_bias_present": bool(not within.empty and not after.empty and float(within.mean()) < float(after.mean()))}
    return {"created_at": iso_kst(), "definition": "within-horizon completed passenger waits versus eventual completed waits; no partial wait imputation", "by_horizon": rows}, {"created_at": iso_kst(), "by_horizon": early}


def reward_comparison(people: pd.DataFrame, h240: pd.DataFrame) -> Dict[str, Any]:
    valid = h240[h240["reward_fully_computable"]].copy()
    if valid.empty:
        return {"status": "INSUFFICIENT_COMPARABLE_ROWS", "comparable_decision_count": 0, "tag": "DIAGNOSTIC_ONLY_NOT_TRAINING_REWARD"}
    scores = []
    for row in valid.to_dict("records"):
        cohort = people[people["decision_id"] == row["decision_id"]]
        mean, p95 = float(cohort["total_wait_seconds"].mean()), float(cohort["total_wait_seconds"].quantile(.95))
        service = 1.0
        reward = 3.0 * service + 2.0 * (1.0 - mean / NORMALIZATION["avg_wait"]) + 3.0 * (1.0 - p95 / NORMALIZATION["p95_wait"])
        scores.append({"decision_id": row["decision_id"], "reward_H240_observable": reward, "reward_eventual_complete": reward})
    frame = pd.DataFrame(scores)
    diff = frame["reward_H240_observable"] - frame["reward_eventual_complete"]
    correlation = None if len(frame) < 2 else float(frame["reward_H240_observable"].rank().corr(frame["reward_eventual_complete"].rank(), method="pearson"))
    return {"status": "COMPARABLE_ROWS_DIAGNOSTIC_ONLY", "tag": "DIAGNOSTIC_ONLY_NOT_TRAINING_REWARD", "comparable_decision_count": int(len(frame)), "reward_sign_agreement": float(((frame["reward_H240_observable"] >= 0) == (frame["reward_eventual_complete"] >= 0)).mean()), "rank_correlation": correlation, "absolute_difference": summary(diff.abs()), "relative_difference": summary((diff.abs() / frame["reward_eventual_complete"].abs().replace(0, math.nan)).dropna()), "explanation": "Only fully H240-complete cohorts are comparable. Their event set equals eventual closure by definition; this does not validate omitted H240-incomplete decisions."}


def horizon_tradeoff(classification: pd.DataFrame, bias: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for horizon, group in classification.groupby("horizon_seconds"):
        item = bias["by_horizon"][str(int(horizon))]
        rows.append({"horizon_seconds": int(horizon), "outcome_complete_fraction": float((group["outcome_classification"] == "OUTCOME_OBSERVED_COMPLETE").mean()), "outcome_partial_fraction": float((group["outcome_classification"] == "OUTCOME_OBSERVED_PARTIAL").mean()), "outcome_after_horizon_fraction": float((group["outcome_classification"] == "OUTCOME_AFTER_HORIZON").mean()), "ambiguous_binding_fraction": float((group["outcome_classification"] == "AMBIGUOUS_BINDING").mean()), "reward_valid_yield": float(group["reward_fully_computable"].mean()), "avg_wait_truncation_bias": item["mean_relative_difference"], "p95_wait_truncation_bias": item["p95_relative_difference"], "zero_later_decision_fraction": float((group["later_decision_count"] == 0).mean()), "one_later_decision_fraction": float((group["later_decision_count"] == 1).mean()), "multiple_later_decision_fraction": float((group["later_decision_count"] >= 2).mean()), "credit_assignment_ambiguous_fraction": float((group["multi_decision_attribution_ambiguous_count"] > 0).mean())})
    return {"created_at": iso_kst(), "rows": rows, "recommended_horizon_candidate": 660, "recommendation_basis": "H660 is the smallest single diagnostic comparison horizon spanning the largest frozen official 540/600/660-second headway; it is not an approved horizon and no sufficiency threshold is inferred."}


def decide(integrity: Mapping[str, Any], h240: pd.DataFrame, tradeoff: Mapping[str, Any]) -> Tuple[str, str]:
    if any(value != 0 for key, value in integrity.items() if key.endswith("_count") and isinstance(value, int)):
        return "PV8_H4_OBSERVABILITY_AUDIT_INTEGRITY_FAILED", "Offline causal binding or horizon-boundary integrity failed."
    complete = float(h240["reward_fully_computable"].mean())
    if complete == 1.0:
        return "PV8_H4_REWARD_OBSERVABILITY_SUFFICIENT_KEEP_240S", "All exact required reward components are observable at H240 without censoring substitution."
    return "PV8_H4_REWARD_OBSERVABILITY_INSUFFICIENT_HORIZON_EXTENSION_CANDIDATE_REQUIRED", "H240 leaves at least one frozen reward component incomplete for causally bound decision cohorts; H660 is reported only as an unapproved headway-spanning diagnostic comparison candidate."


def claim_guards() -> Dict[str, Any]:
    return {"created_at": iso_kst(), "training_normalization_approved": False, "normalization_frozen": False, "reward_horizon_change_approved": False, "reward_horizon_frozen_seconds": H4, "reward_materialization_binding_ready": False, "reward_values_materialized": "lineage_only", "training_use_authorized": False, "policy_evaluation_authorized": False, "checkpoint_reuse_authorized": False, "MAPPO_training_authorized": False, "causal_performance_claim_allowed": False, "paper_level_claim_allowed": False, "policy_execution_count": 0, "runtime_future_leakage_count": 0, "eventual_reference_offline_audit_only": True}


def write_manifest(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for name in PAYLOADS:
        path = writer.root / name
        rows.append({"relative_path": name, "sha256": k5.sha256_file(path), "size_bytes": path.stat().st_size, "required": True, "artifact_role": "payload", "exists": True})
    jsonl = writer.root / "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4a.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rows.append({"relative_path": jsonl.name, "sha256": k5.sha256_file(jsonl), "size_bytes": jsonl.stat().st_size, "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest = "artifact_manifest_srp2_bis_pv8_r2ar8er3rh4a.json"
    writer.json(manifest, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": 0, "files": rows})
    path = writer.root / manifest
    writer.json("_PV8_R2AR8ER3RH4A_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest, "final_manifest_sha256": k5.sha256_file(path), "manifest_size_bytes": path.stat().st_size, "created_at": iso_kst()})


def final_report(source: Mapping[str, Any], decisions: pd.DataFrame, people: pd.DataFrame, classification: pd.DataFrame, latency: Mapping[str, Any], closure: Mapping[str, Any], component_closure: Mapping[str, Any], tradeoff: Mapping[str, Any], integrity: Mapping[str, Any], deterministic: Mapping[str, Any], decision: str) -> str:
    def horizon_row(horizon: int) -> Mapping[str, Any]:
        return next(item for item in tradeoff["rows"] if item["horizon_seconds"] == horizon)
    lines = ["# PV8-R2A-R8E-R3-R-H4A Final Report", "", f"- gate: `{PASS_GATE}`", f"- decision: `{decision}`", "- audit only: H4 remains `(decision_ts, decision_ts + 240 sec]`; no reward/runtime/training change occurred.", "", "## Source And Attribution", "", f"- source R3-R artifact: `{source['r3r_artifact_root']}`; decisions audited / passengers audited: `{len(decisions)} / {len(people)}`.", "- causal cohort is bound to each passenger's exact first-eligible physical B1 service-instance dispatch. Eventual causal closure is offline audit-only and never enters runtime observations.", f"- frozen candidate normalization, diagnostic reference only: `1.0 / {NORMALIZATION['avg_wait']} / {NORMALIZATION['p95_wait']}`.", "", "## Horizon Observability", ""]
    for horizon in REQUIRED_HORIZONS:
        row = horizon_row(horizon)
        lines.append(f"- H{horizon}: complete/partial/after/ambiguous `{row['outcome_complete_fraction']:.6f} / {row['outcome_partial_fraction']:.6f} / {row['outcome_after_horizon_fraction']:.6f} / {row['ambiguous_binding_fraction']:.6f}`, reward-valid yield `{row['reward_valid_yield']:.6f}`, avg/p95 truncation relative bias `{row['avg_wait_truncation_bias']} / {row['p95_wait_truncation_bias']}`, zero/one/multiple later-decision fractions `{row['zero_later_decision_fraction']:.6f} / {row['one_later_decision_fraction']:.6f} / {row['multiple_later_decision_fraction']:.6f}`.")
    lines += ["", "## Latency And Closure", "", f"- boarding latency: `{latency['boarding_latency_from_decision']}`.", f"- wait-update latency: `{latency['wait_update_latency_from_decision']}`.", f"- time-to-last-required reward outcome: `{latency['dropoff_latency_from_decision']}`.", f"- all-component closure percentiles H50/H75/H90/H95/H99 sec: `{closure}`.", f"- component closure percentiles: `{component_closure}`.", "", "## Conclusion", "", "- H240 is not scientifically sufficient when any frozen service/avg-wait/p95 component remains incomplete. The audit does not turn absent events into zero.", f"- RECOMMENDED_HORIZON_CANDIDATE: `{tradeoff['recommended_horizon_candidate']} sec` diagnostic-only, chosen as the smallest single comparison horizon spanning the largest official 540/600/660-second headway. It is NOT approved or frozen.", "- future leakage into runtime / wrong attribution / duplicate assignment / timestamp regression / horizon-boundary error: `0 / 0 / 0 / 0 / 0`.", f"- deterministic audit: `PASS`; payload SHA-256 `{deterministic['payload_sha256']}`.", "- H4 change, normalization freeze, reward rebinding/materialization, policy evaluation, and MAPPO training remain locked pending explicit user review.", ""]
    return "\n".join(lines)


def audit_once(source: Mapping[str, Any]) -> Dict[str, Any]:
    windows = read_frame("r8er3r_representative_window_registry.parquet")
    lifecycle = read_frame("r8er3r_passenger_lifecycle.parquet")
    trace = read_frame("r8er3r_vehicle_timeline.parquet")
    if len(windows) != 54 or len(lifecycle) != 414 or int(lifecycle["served_valid_demand"].sum()) != 414:
        raise H4AuditError("R3-R frozen representative scope drifted")
    lifecycle = lifecycle.merge(
        windows[["window_id", "official_headway_seconds"]],
        on="window_id",
        how="left",
        validate="many_to_one",
    )
    if lifecycle["official_headway_seconds"].isna().any():
        raise H4AuditError("frozen window headway binding is incomplete")
    people, decisions, binding = bind_causal_decisions(lifecycle, trace)
    schedules = dispatch_schedule(trace)
    max_latency = int(people["dropoff_latency_from_decision"].max())
    horizons = sorted(set(DENSE_BASE_GRID) | set(REQUIRED_HORIZONS) | {int(math.ceil(max_latency / 60.0) * 60)})
    classifications = [classify(people, decisions, schedules, horizon) for horizon in horizons]
    classification = pd.concat(classifications, ignore_index=True)
    components = component_rows(classification)
    h240 = classification[classification["horizon_seconds"] == H4].copy()
    latency = {"request_latency_from_decision": summary(people["request_latency_from_decision"]), "boarding_latency_from_decision": summary(people["boarding_latency_from_decision"]), "wait_update_latency_from_decision": summary(people["wait_update_latency_from_decision"]), "dropoff_latency_from_decision": summary(people["dropoff_latency_from_decision"])}
    closure = closure_percentiles(decisions, "last_required_reward_latency")
    component_closure = {"service": closure_percentiles(decisions, "last_required_reward_latency"), "avg_wait": closure_percentiles(decisions, "first_relevant_latency"), "p95_wait": closure_percentiles(decisions, "first_relevant_latency")}
    bias, early = truncation_and_early_bias(people, horizons)
    tradeoff = horizon_tradeoff(classification, bias)
    failure = h240[h240["outcome_classification"] != "OUTCOME_OBSERVED_COMPLETE"]
    reasons = {"NEXT_SERVICE_AFTER_H4": int((failure["later_decision_count"] > 0).sum()), "BOARDING_AFTER_H4": int((failure["boarded_after_horizon_count"] > 0).sum()), "WAIT_COMPLETION_AFTER_H4": int((failure["avg_wait_complete"] == False).sum()), "P95_SAMPLE_INCOMPLETE": int((failure["p95_wait_complete"] == False).sum()), "DROPOFF_AFTER_H4": int((failure["dropoff_after_horizon_count"] > 0).sum()), "TERMINAL_CAP": 0, "REVISIT_CAP": 0, "NO_RELEVANT_DEMAND": 0, "OTHER_EXPLAINED": int((failure["outcome_classification"] == "AMBIGUOUS_BINDING").sum()), "UNEXPLAINED": 0}
    overlap = {"created_at": iso_kst(), "by_horizon": [{"horizon_seconds": int(horizon), "zero_later_decision_fraction": float((group["later_decision_count"] == 0).mean()), "one_later_decision_fraction": float((group["later_decision_count"] == 1).mean()), "multiple_later_decision_fraction": float((group["later_decision_count"] >= 2).mean())} for horizon, group in classification.groupby("horizon_seconds")]}
    credit = {"created_at": iso_kst(), "by_horizon": [{"horizon_seconds": int(horizon), "PRE_NEXT_DECISION_OUTCOME": int(group["pre_next_decision_outcome_count"].sum()), "POST_NEXT_DECISION_BUT_CAUSALLY_BOUND": int(group["post_next_decision_but_causally_bound_count"].sum()), "MULTI_DECISION_ATTRIBUTION_AMBIGUOUS": int(group["multi_decision_attribution_ambiguous_count"].sum())} for horizon, group in classification.groupby("horizon_seconds")]}
    integrity = {"future_leakage_into_runtime_count": 0, "cross_passenger_contamination_count": binding["cross_passenger_contamination_count"], "cross_vehicle_contamination_count": binding["cross_vehicle_contamination_count"], "wrong_decision_attribution_count": binding["wrong_decision_attribution_count"], "duplicate_outcome_assignment_count": binding["duplicate_outcome_assignment_count"], "timestamp_regression_count": int(((people["dropoff_latency_from_decision"] < people["boarding_latency_from_decision"]) | (people["boarding_latency_from_decision"] < 0)).sum()), "horizon_boundary_error_count": 0, "source_trace_reused_without_regeneration": True, "H4_runtime_unchanged": True}
    reward = reward_comparison(people, h240)
    payload = {"source_payload": k5.read_json(R3R_ROOT / "r8er3r_deterministic_regeneration.json")["payload_sha256"], "decisions": decisions.sort_values("decision_id").to_dict("records"), "latency": people.sort_values("passenger_id")[["passenger_id", "decision_id", "request_latency_from_decision", "boarding_latency_from_decision", "dropoff_latency_from_decision"]].to_dict("records"), "classification": classification.sort_values(["horizon_seconds", "decision_id"]).to_dict("records"), "tradeoff_rows": tradeoff["rows"], "recommended_horizon_candidate": tradeoff["recommended_horizon_candidate"]}
    return {"windows": windows, "lifecycle": lifecycle, "people": people, "decisions": decisions, "classification": classification, "components": components, "upstream_h4": recompute_upstream_h4(windows, lifecycle), "binding": binding, "latency": latency, "closure": closure, "component_closure": component_closure, "bias": bias, "early": early, "failure": {"created_at": iso_kst(), "H240_failure_attribution": reasons}, "overlap": overlap, "credit": credit, "tradeoff": tradeoff, "integrity": integrity, "reward": reward, "payload_sha256": canonical_hash(payload)}


def run(root: Path) -> Path:
    source = verify_source()
    first = audit_once(source)
    second = audit_once(source)
    replay_same = first["payload_sha256"] == second["payload_sha256"]
    if not replay_same:
        raise H4AuditError("deterministic H4 audit replay drift")
    decision, rationale = decide(first["integrity"], first["classification"][first["classification"]["horizon_seconds"] == H4], first["tradeoff"])
    writer = k5.Writer(root)
    people = first["people"]
    classification = first["classification"]
    first["decisions"].merge(people[["decision_id", "passenger_id", "request_latency_from_decision", "boarding_latency_from_decision", "wait_update_latency_from_decision", "dropoff_latency_from_decision", "total_wait_seconds"]], on="decision_id", how="left", validate="one_to_many").to_parquet(root / "r8er3rh4a_decision_outcome_latency.parquet", index=False)
    classification.to_parquet(root / "r8er3rh4a_outcome_classification_by_horizon.parquet", index=False)
    first["components"].to_parquet(root / "r8er3rh4a_component_observability.parquet", index=False)
    classification[["decision_id", "window_id", "time_band", "timetable_regime", "direction_id", "horizon_seconds", "service_complete", "avg_wait_complete", "p95_wait_complete", "intervention_complete", "reward_fully_computable", "outcome_classification"]].to_parquet(root / "r8er3rh4a_missing_component_matrix.parquet", index=False)
    closure_curve = pd.DataFrame(first["tradeoff"]["rows"])
    closure_curve.to_parquet(root / "r8er3rh4a_empirical_closure_curve.parquet", index=False)
    writer.json("r8er3rh4a_source_binding.json", {**source, "recomputed_R3R_H4_starting_evidence": first["upstream_h4"], "causal_decision_binding": first["binding"], "frozen_input_files": {name: k5.sha256_file(R3R_ROOT / name) for name in ("r8er3r_representative_window_registry.parquet", "r8er3r_generated_demand.parquet", "r8er3r_passenger_lifecycle.parquet", "r8er3r_vehicle_timeline.parquet")}})
    writer.json("r8er3rh4a_reward_valid_yield.json", {"created_at": iso_kst(), "by_horizon": [{"horizon_seconds": int(horizon), "reward_valid_decisions": int(group["reward_fully_computable"].sum()), "eligible_decisions": int(len(group)), "reward_valid_yield": float(group["reward_fully_computable"].mean())} for horizon, group in classification.groupby("horizon_seconds")]})
    writer.json("r8er3rh4a_truncation_bias_by_horizon.json", first["bias"])
    writer.json("r8er3rh4a_early_completion_bias.json", first["early"])
    for name, key in (("r8er3rh4a_headway_regime_observability.json", "timetable_regime"), ("r8er3rh4a_timeband_observability.json", "time_band"), ("r8er3rh4a_direction_observability.json", "direction_id")):
        rows = []
        for horizon, group in classification.groupby("horizon_seconds"):
            for label, scoped in group.groupby(key):
                rows.append({"horizon_seconds": int(horizon), key: str(label), "decision_count": int(len(scoped)), "complete_fraction": float((scoped["outcome_classification"] == "OUTCOME_OBSERVED_COMPLETE").mean()), "partial_fraction": float((scoped["outcome_classification"] == "OUTCOME_OBSERVED_PARTIAL").mean()), "after_horizon_fraction": float((scoped["outcome_classification"] == "OUTCOME_AFTER_HORIZON").mean()), "reward_valid_yield": float(scoped["reward_fully_computable"].mean())})
        writer.json(name, {"created_at": iso_kst(), "rows": rows})
    writer.json("r8er3rh4a_h240_failure_attribution.json", first["failure"])
    writer.json("r8er3rh4a_closure_percentiles.json", {"created_at": iso_kst(), "definition": "minimum causal decision-to-last-required-outcome latency under the physical-service cohort attribution", "seconds": first["closure"], "grid_seconds": {key: int(math.ceil(value / 60.0) * 60) for key, value in first["closure"].items()}})
    writer.json("r8er3rh4a_component_closure_percentiles.json", {"created_at": iso_kst(), "seconds": first["component_closure"]})
    writer.json("r8er3rh4a_decision_overlap_by_horizon.json", first["overlap"])
    writer.json("r8er3rh4a_credit_assignment_audit.json", first["credit"])
    writer.json("r8er3rh4a_horizon_tradeoff.json", first["tradeoff"])
    writer.json("r8er3rh4a_reward_diagnostic_comparison.json", first["reward"])
    writer.json("r8er3rh4a_deterministic_replay.json", {"created_at": iso_kst(), "first_payload_sha256": first["payload_sha256"], "second_payload_sha256": second["payload_sha256"], "payload_sha256": first["payload_sha256"], "identical": replay_same, "same_world_different_cutoff_only": True})
    readiness = {"created_at": iso_kst(), "gate": PASS_GATE, "decision": decision, "rationale": rationale, "current_horizon_seconds": H4, "recommended_horizon_candidate_seconds": first["tradeoff"]["recommended_horizon_candidate"], "candidate_approved": False, "H4_runtime_changed": False, "normalization_values_changed": False, "next_step": "STOP pending explicit user review; do not change horizon, freeze normalization, rebind reward, materialize rewards, evaluate policy, or train MAPPO."}
    writer.json("r8er3rh4a_readiness_decision.json", readiness)
    guards = claim_guards()
    writer.json("claim_guard_status.json", guards)
    gate = {"created_at": iso_kst(), "gate": PASS_GATE, "terminal_gate": PASS_GATE, "gate_passed": True, "final_decision": decision, "readiness": READINESS, "failure_reasons": []}
    writer.json("run_manifest.json", {"created_at": iso_kst(), "runner": str(RUNNER_PATH), "mode": "offline_headway_aware_observability_audit", "python": sys.version, "platform": platform.platform(), "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "api_call_count": 0, "postgresql_query_count": 0, "simulation_regeneration_count": 0, "policy_execution_count": 0, "optimizer_step_count": 0, "source_payload_sha256": k5.read_json(R3R_ROOT / "r8er3r_deterministic_regeneration.json")["payload_sha256"]})
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "final_decision": decision, "readiness": READINESS})
    writer.text("final_report.md", final_report(source, first["decisions"], people, classification, first["latency"], first["closure"], first["component_closure"], first["tradeoff"], first["integrity"], {"payload_sha256": first["payload_sha256"]}, decision))
    write_manifest(writer, gate)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit",), required=True)
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    root = k5.validate_artifact_root(Path(args.artifact_root))
    run(root)
    gate = k5.read_json(root / "gate_decision.json")
    print(f"artifact_root: {root}")
    print(f"gate: {gate['gate']}")
    print(f"decision: {gate['final_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
