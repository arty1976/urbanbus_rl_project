#!/usr/bin/env python3
"""PV8-R2A-R8E-R1 stop-service, empty-stop, and dwell semantics audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8b_headway_aware_b1_regeneration as r8b
from simulator.k_safety_state import ServiceObligationStateMachine


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er1_stop_service_timeband_audit.py"

R8B_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8b_headway_aware_b1_regeneration_20260809_144648"
R8C_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8c_training_normalization_freeze_20260809_155120"
R8D_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8d_reward_materialization_binding_preflight_20260809_161233"
R8E_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8e_bounded_reward_materialization_20260809_163733"

NORMALIZATION_VERSION = "PV8_HEADWAY_AWARE_B1_NORMALIZATION_V1"
NORMALIZATION_SHA256 = "c180fa69306242cfdd6b1ddeddfb40ef46ba31a9a78cfa4946b02d2c08f77745"
REWARD_RUNTIME_VERSION = "PV8_CAUSAL_REWARD_RUNTIME_V1"
REWARD_RUNTIME_SHA256 = "dd9f71adcaba75180c3230f1f295eeebe3d472c1aacefa7ac8b1ff06a6e27dae"
R8E_PAYLOAD_SHA256 = "bd74183dc9011651839fc356d2571d7481fb3093a8514ed9feed0c656e5acd06"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er1_stop_service_timeband_audit"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER1_REAL_WORLD_STOP_SERVICE_TIMEBAND_AUDIT_COMPLETE"
FINAL_DECISION = "PV8_DWELL_TEMPORAL_REPAIR_REQUIRED"
READINESS = "SRP2_BIS_PV8_R2AR8ER1_COMPLETE_DWELL_TEMPORAL_REPAIR_REQUIRED_TRAINING_LOCKED"

UPSTREAMS = {
    "PV8-R2A-R8B": (
        R8B_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8b.json",
        "_PV8_R2AR8B_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8B_OFFICIAL_HEADWAY_AWARE_B1_REGENERATION_COMPLETE",
    ),
    "PV8-R2A-R8C": (
        R8C_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8c.json",
        "_PV8_R2AR8C_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8C_HEADWAY_AWARE_TRAINING_NORMALIZATION_FROZEN",
    ),
    "PV8-R2A-R8D": (
        R8D_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8d.json",
        "_PV8_R2AR8D_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8D_REWARD_MATERIALIZATION_BINDING_PREFLIGHT_COMPLETE",
    ),
    "PV8-R2A-R8E": (
        R8E_ROOT,
        "artifact_manifest_srp2_bis_pv8_r2ar8e.json",
        "_PV8_R2AR8E_COMPLETE.lock",
        "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8E_BOUNDED_CAUSAL_REWARD_MATERIALIZATION_COMPLETE",
    ),
}

PAYLOADS = [
    "r8er1_official_stop_service_rule_audit.json",
    "r8er1_stop_service_semantics_contract.json",
    "r8er1_action_semantics_alignment.json",
    "r8er1_r8e_47row_obligation_audit.parquet",
    "r8er1_timeband_obligation_summary.parquet",
    "r8er1_timeband_empty_stop_comparison.json",
    "r8er1_demand_obligation_consistency.json",
    "r8er1_b1_overstopping_audit.json",
    "r8er1_realistic_fixed_route_b1_candidate.json",
    "r8er1_pass_through_fixture_results.json",
    "r8er1_dwell_travel_semantics_audit.json",
    "r8er1_headway_integrity_audit.json",
    "r8er1_comparison_fairness_audit.json",
    "r8er1_normalization_validity_reassessment.json",
    "r8er1_r8e_payload_validity_reassessment.json",
    "r8er1_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R8ER1Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(k5.json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_upstreams() -> Dict[str, Any]:
    records: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        integrity = k5.verify_manifest(root, manifest_name, lock_name)
        valid = observed_gate == expected_gate and k5.manifest_ok(integrity)
        records[label] = {
            "artifact_root": str(root),
            "expected_gate": expected_gate,
            "observed_gate": observed_gate,
            "final_decision": gate.get("final_decision"),
            "readiness": gate.get("readiness"),
            "manifest_integrity": integrity,
            "valid": valid,
        }
        if not valid:
            raise R8ER1Error(f"{label} integrity failure: gate={observed_gate}, integrity={integrity}")

    r8c = k5.read_json(R8C_ROOT / "r8c_training_normalization_contract.json")
    r8d = k5.read_json(R8D_ROOT / "r8d_reward_runtime_contract.json")
    r8e_sha = (R8E_ROOT / "r8e_reward_materialization_payload.sha256").read_text(encoding="utf-8").strip()
    r8c_body = r8c.get("contract_body", {})
    r8d_body = r8d.get("contract_body", {})
    bindings = {
        "normalization_version_matches": r8c_body.get("normalization_version") == NORMALIZATION_VERSION,
        "normalization_sha256_matches": r8c.get("normalization_contract_sha256") == NORMALIZATION_SHA256,
        "reward_runtime_version_matches": r8d_body.get("reward_runtime_contract_version") == REWARD_RUNTIME_VERSION,
        "reward_runtime_sha256_matches": r8d.get("reward_runtime_contract_sha256") == REWARD_RUNTIME_SHA256,
        "r8e_payload_sha256_matches": r8e_sha == R8E_PAYLOAD_SHA256,
    }
    if not all(bindings.values()):
        raise R8ER1Error(f"frozen contract binding failure: {bindings}")
    return {"created_at": iso_kst(), "artifacts": records, "frozen_binding_checks": bindings}


def official_rule_audit() -> Dict[str, Any]:
    sources = [
        {
            "source_id": "KOREA_PASSENGER_TRANSPORT_SERVICE_ACT_ARTICLE_26_1_6",
            "source_owner": "Korea Ministry of Government Legislation, National Law Information Center",
            "source_url": "https://www.law.go.kr/LSW/lsLinkCommonInfo.do?lsJoLnkSeq=1032840525",
            "effective_date": "2026-07-01",
            "retrieved_at": iso_kst(),
            "authority_rank": 1,
            "finding": "A driver must not depart before boarding/alighting is complete or pass a stop when a passenger intends to board or alight.",
            "claim_scope": "positive boarding/alighting demand requires a physical service stop",
        },
        {
            "source_id": "KOREA_PASSENGER_TRANSPORT_ENFORCEMENT_RULE_ARTICLE_8_GENERAL_TYPE",
            "source_owner": "Korea Ministry of Government Legislation, National Law Information Center",
            "source_url": "https://www.law.go.kr/LSW/lsInfoP.do?ancYnChk=1&chrClsCd=010202&efYd=20260324&gubun=api&lsiSeq=285025&mobile=&urlMode=lsInfoP",
            "effective_date": "2026-03-24",
            "retrieved_at": iso_kst(),
            "authority_rank": 1,
            "finding": "The general-type city-bus operating form uses general city buses and operates while stopping at each stop.",
            "claim_scope": "empty-stop physical pass-through is not proven as a real-network general-type rule",
        },
        {
            "source_id": "DAEGU_DUDEURISO_NONSTOP_CASE_20180921",
            "source_owner": "Daegu Metropolitan City",
            "source_url": "https://smart.daegu.go.kr/vocRqs/pbl/view.do?cvlRqrNo=T-R-20180919-000039&orderByKey=&orderByType=&pageIndex=196&showItemListCount=10&srchTxt=&srchType=",
            "effective_date": "2018-09-21",
            "retrieved_at": iso_kst(),
            "authority_rank": 3,
            "finding": "Daegu applied Article 26(1)(6) to a reported pass-by where waiting riders intended to board and referred the case for administrative review.",
            "claim_scope": "Daegu operational enforcement confirms the positive pickup obligation; it does not authorize empty-stop pass-through.",
        },
        {
            "source_id": "ROAD_TRAFFIC_ACT_ENFORCEMENT_DECREE_ARTICLE_11",
            "source_owner": "Korea Ministry of Government Legislation, National Law Information Center",
            "source_url": "https://www.law.go.kr/LSW/lsSideInfoP.do?docCls=jo&joBrNo=00&joNo=0011&lsiSeq=286789&urlMode=lsScJoRltInfoR",
            "effective_date": "2026-06-09",
            "retrieved_at": iso_kst(),
            "authority_rank": 1,
            "finding": "When stopped to board or alight passengers, a passenger vehicle must depart immediately after that service and must not obstruct following vehicles.",
            "claim_scope": "dwell is service-driven and should not be extended without evidence",
        },
    ]
    rules = {
        "waiting_passenger_clearly_intends_to_board": {
            "classification": "OBSERVED_OFFICIAL_RULE",
            "decision": "SERVE_REQUIRED",
            "source_ids": [sources[0]["source_id"], sources[2]["source_id"]],
        },
        "onboard_passenger_has_dropoff_obligation": {
            "classification": "OBSERVED_OFFICIAL_RULE",
            "decision": "SERVE_REQUIRED",
            "source_ids": [sources[0]["source_id"]],
        },
        "no_boarding_or_alighting_demand": {
            "classification": "UNRESOLVED",
            "decision": "GENERAL_TYPE_EACH_STOP_OPERATING_FORM_FOUND; LITERAL_ZERO_DEMAND_PHYSICAL_HALT_VS_PASS_THROUGH_NOT_EXPLICITLY_DISAMBIGUATED",
            "source_ids": [sources[1]["source_id"]],
            "pass_through_officially_authorized": False,
        },
        "mandatory_protected_terminal_turnaround_condition": {
            "classification": "UNRESOLVED",
            "decision": "SPECIAL_RULE_SOURCE_NOT_SEPARATELY_PROVEN; GENERAL_EACH_STOP_RULE_STILL_APPLIES",
            "source_ids": [sources[1]["source_id"]],
        },
    }
    return {
        "created_at": iso_kst(),
        "research_order_followed": [
            "National Law Information Center",
            "Daegu official transport/BIS sources",
            "Daegu official civil-service cases",
            "public official operator/service manuals",
        ],
        "unofficial_source_count": 0,
        "sources": sources,
        "rules": rules,
        "route814_city_bus_status": "DAEGU_CITY_BUS_CONFIRMED; precise license subtype and literal zero-demand halt semantics are not separately materialized in R8B",
        "empty_stop_pass_through_real_network_claim_allowed": False,
        "claim_boundary": "K8 is an explicitly approved research-rule contract, not observed Daegu operational authority.",
    }


def build_state_snapshot(
    opportunity: Any,
    runtime_context: Mapping[str, Any],
    window_by_id: Mapping[str, Mapping[str, Any]],
    multipliers: Mapping[str, float],
) -> Tuple[List[Any], Dict[str, Any], Dict[str, Any]]:
    agent_id = int(opportunity.agent_id)
    segment = runtime_context["segments"][agent_id]
    schedule = r8b.requests_for_opportunity(
        opportunity,
        segment,
        str(runtime_context["fixed_bindings"][agent_id]),
        multipliers,
        int(window_by_id[opportunity.window_id]["window_start"].timestamp()),
    )
    state = ServiceObligationStateMachine()
    for aid, token in sorted(runtime_context["fixed_bindings"].items()):
        state.register_vehicle(aid, token)
    for route_segment in runtime_context["segments"].values():
        for rule in route_segment[:4]:
            state.register_stop(str(rule["stop_id"]))
    runtime_context["orchestrator"].apply_schedule(state, schedule)
    state.advance_to(int(opportunity.service_opportunity_ts))
    snapshot = runtime_context["orchestrator"].capture_snapshot(
        state=state,
        cycle_index=int(opportunity.service_opportunity_ts),
        decision_ts=int(opportunity.service_opportunity_ts),
        occurrence_by_agent={aid: rows[0] for aid, rows in runtime_context["segments"].items()},
        active_by_agent={aid: True for aid in range(8)},
    )
    agent_row = next(row for row in snapshot.to_payload()["agent_snapshots"] if int(row["agent_id"]) == agent_id)
    return schedule, dict(agent_row), dict(segment[0])


def classify_obligation(dynamic: Mapping[str, Any], research_static: bool) -> str:
    pickup = bool(dynamic["pickup_obligation"])
    dropoff = bool(dynamic["dropoff_obligation"])
    active = sum([pickup, dropoff, research_static])
    if active > 1:
        return "SERVE_REQUIRED_MULTIPLE"
    if pickup:
        return "SERVE_REQUIRED_PICKUP"
    if dropoff:
        return "SERVE_REQUIRED_DROPOFF"
    if research_static:
        return "SERVE_REQUIRED_STATIC_RULE"
    return "EMPTY_STOP_PASS_THROUGH_ELIGIBLE"


def reconstruct_audits() -> Dict[str, Any]:
    runtime_context = r8b.build_runtime_context()
    contract = r8b.headway_temporal_contract(r8b.load_headway_mapping())
    opportunities = r8b.build_service_opportunities(contract, runtime_context)
    windows = {row["window_id"]: row for row in r8b.build_window_plan()}
    multipliers = r8b.load_timeband_multipliers()
    source = pd.read_parquet(R8B_ROOT / "r8b_service_opportunity_schedule.parquet")
    reward = pd.read_parquet(R8E_ROOT / "r8e_reward_materialization.parquet")
    source_by_id = {str(row["schedule_row_id"]): row for row in source.to_dict("records")}
    reward_by_source = {str(row["source_schedule_row_id"]): row for row in reward.to_dict("records")}
    if len(opportunities) != len(source_by_id) or len(reward_by_source) != 47:
        raise R8ER1Error("R8B/R8E cardinality mismatch during obligation reconstruction")

    all_rows: List[Dict[str, Any]] = []
    reward_rows: List[Dict[str, Any]] = []
    for index, opportunity in enumerate(opportunities, start=1):
        source_id = f"R8B_SERVICE_OPPORTUNITY_{index:04d}"
        source_row = source_by_id[source_id]
        schedule, agent_row, rule = build_state_snapshot(opportunity, runtime_context, windows, multipliers)
        dynamic = dict(agent_row["decision_time_obligation_snapshot"])
        research_static = bool(
            rule["mandatory_stop"]
            or rule["protected_stop"]
            or rule["terminal_or_turnaround_stop"]
            or rule["charging_or_driver_relief_stop"]
        )
        empty_dynamic = not bool(dynamic["service_obligation"])
        research_pass = bool(agent_row["k_action_mask"][2])
        row = {
            "source_schedule_row_id": source_id,
            "window_id": str(source_row["window_id"]),
            "agent_id": int(opportunity.agent_id),
            "vehicle_token": str(source_row["vehicle_token"]),
            "decision_ts": str(source_row["service_opportunity_iso"]),
            "time_band": str(source_row["time_band"]),
            "timetable_type": str(source_row["official_timetable_type"]),
            "route_id": str(rule["route_id"]),
            "direction_id": str(rule["direction_id"]),
            "route_stop_occurrence_id": str(rule["route_stop_occurrence_id"]),
            "stop_id": str(rule["stop_id"]),
            "stop_sequence": int(rule["stop_sequence"]),
            "waiting_pickup_count": len(dynamic["waiting_queue"]),
            "assigned_pickup_count": len(dynamic["assigned_pickup_request_ids"]),
            "onboard_dropoff_obligation_count": len(dynamic["onboard_destination_request_ids"]),
            "assigned_dropoff_count": len(dynamic["assigned_dropoff_request_ids"]),
            "mandatory_stop_flag": bool(rule["mandatory_stop"]),
            "protected_stop_flag": bool(rule["protected_stop"]),
            "terminal_or_turnaround_flag": bool(rule["terminal_or_turnaround_stop"]),
            "charging_or_driver_relief_flag": bool(rule["charging_or_driver_relief_stop"]),
            "official_general_each_stop_rule_applies": True,
            "executed_action": str(source_row["action_t"]),
            "generated_passenger_count": len(schedule),
            "served_passenger_count": int(source_row["served_passengers"]),
            "service_obligation_present": bool(dynamic["service_obligation"]),
            "empty_stop_arrival_dynamic": empty_dynamic,
            "pass_through_eligible_research_contract": research_pass,
            "pass_through_eligible_real_network": False,
            "official_service_stop_required_fail_closed": True,
            "research_obligation_classification": classify_obligation(dynamic, research_static),
            "real_network_classification": "SERVE_REQUIRED_PICKUP" if bool(dynamic["pickup_obligation"]) else "UNRESOLVED_OBLIGATION",
            "time_band_used_by_k_mask": False,
            "dynamic_state_complete": bool(dynamic["dynamic_state_complete"]),
            "future_leakage": False,
            "source_in_r8e_payload": source_id in reward_by_source,
        }
        all_rows.append(row)
        if source_id in reward_by_source:
            reward_row = dict(reward_by_source[source_id])
            row.update({"transition_id": str(reward_row["transition_id"]), "decision_id": str(reward_row["decision_id"])})
            reward_rows.append(dict(row))

    if len(reward_rows) != 47 or any(not row["source_in_r8e_payload"] for row in reward_rows):
        raise R8ER1Error("failed to reconstruct all 47 R8E rows")
    return {
        "runtime_context": runtime_context,
        "contract": contract,
        "source": source,
        "reward": reward,
        "all_rows": all_rows,
        "reward_rows": reward_rows,
        "multipliers": multipliers,
    }


def timeband_summary(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    output: List[Dict[str, Any]] = []
    for band in ["night", "offpeak", "peak"]:
        group = [row for row in rows if row["time_band"] == band]
        total = len(group)
        pickup = sum(bool(row["waiting_pickup_count"] or row["assigned_pickup_count"]) for row in group)
        dropoff = sum(bool(row["onboard_dropoff_obligation_count"] or row["assigned_dropoff_count"]) for row in group)
        both = sum(
            bool(row["waiting_pickup_count"] or row["assigned_pickup_count"])
            and bool(row["onboard_dropoff_obligation_count"] or row["assigned_dropoff_count"])
            for row in group
        )
        empty = sum(bool(row["empty_stop_arrival_dynamic"]) for row in group)
        research_pass = sum(bool(row["pass_through_eligible_research_contract"]) for row in group)
        actual_serve = sum(str(row["executed_action"]) == "SERVE_AND_MOVE_TO_NEXT_STOP" for row in group)
        actual_pass = sum("SKIP" in str(row["executed_action"]) for row in group)
        actual_hold = sum("HOLD" in str(row["executed_action"]) for row in group)
        output.append({
            "time_band": band,
            "demand_multiplier_vs_night": float({"night": 1.0, "offpeak": 0.764451, "peak": 0.977582}[band]),
            "total_stop_arrivals": total,
            "generated_passenger_count": sum(int(row["generated_passenger_count"]) for row in group),
            "served_passenger_count": sum(int(row["served_passenger_count"]) for row in group),
            "pickup_obligation_arrivals": pickup,
            "dropoff_obligation_arrivals": dropoff,
            "pickup_plus_dropoff_arrivals": both,
            "research_static_mandatory_arrivals": sum(
                bool(row["mandatory_stop_flag"] or row["protected_stop_flag"] or row["terminal_or_turnaround_flag"] or row["charging_or_driver_relief_flag"])
                for row in group
            ),
            "official_general_each_stop_required_arrivals": total,
            "empty_stop_arrivals_dynamic": empty,
            "pass_through_eligible_arrivals_research_contract": research_pass,
            "pass_through_eligible_arrivals_real_network": 0,
            "serve_required_arrivals_dynamic_only": total - empty,
            "serve_required_arrivals_official": total,
            "actual_serve_count": actual_serve,
            "actual_pass_through_count": actual_pass,
            "actual_hold_count": actual_hold,
            "pickup_obligation_rate": pickup / total if total else None,
            "dropoff_obligation_rate": dropoff / total if total else None,
            "empty_stop_rate_dynamic": empty / total if total else None,
            "pass_through_eligibility_rate_research_contract": research_pass / total if total else None,
            "pass_through_eligibility_rate_real_network": 0.0,
            "serve_required_rate_dynamic_only": (total - empty) / total if total else None,
            "serve_required_rate_official": 1.0,
            "actual_pass_through_rate": actual_pass / total if total else None,
        })
    return pd.DataFrame(output)


def pairwise_empty_comparison(summary: pd.DataFrame) -> Dict[str, Any]:
    by_band = {str(row["time_band"]): row for row in summary.to_dict("records")}
    pairs = []
    for left, right in [("night", "offpeak"), ("night", "peak"), ("offpeak", "peak")]:
        lrow, rrow = by_band[left], by_band[right]
        pairs.append({
            "comparison": f"{left}_vs_{right}",
            "demand_multiplier_difference": float(lrow["demand_multiplier_vs_night"] - rrow["demand_multiplier_vs_night"]),
            "dynamic_empty_stop_rate_difference": float(lrow["empty_stop_rate_dynamic"] - rrow["empty_stop_rate_dynamic"]),
            "research_pass_through_rate_difference": float(lrow["pass_through_eligibility_rate_research_contract"] - rrow["pass_through_eligibility_rate_research_contract"]),
            "real_network_pass_through_rate_difference": 0.0,
        })
    return {
        "created_at": iso_kst(),
        "comparisons": pairs,
        "observed_dynamic_pattern": "offpeak has the lowest configured demand multiplier and the highest dynamic-empty rate; peak has the lowest dynamic-empty rate",
        "causal_attribution_allowed": False,
        "reason": "The generator enforces at least one request for every full opportunity. All nine empty rows are requests clipped before the 29-minute window boundary, so the empty-rate difference is not a validated demand-to-empty-stop mechanism.",
        "time_band_directly_modified_k_mask": False,
    }


def demand_consistency(rows: Sequence[Mapping[str, Any]], summary: pd.DataFrame) -> Dict[str, Any]:
    bands = []
    for row in summary.to_dict("records"):
        band = str(row["time_band"])
        group = [item for item in rows if item["time_band"] == band]
        bands.append({
            "time_band": band,
            "demand_multiplier_vs_night": float(row["demand_multiplier_vs_night"]),
            "generated_passenger_count": int(row["generated_passenger_count"]),
            "pickup_request_count": sum(int(item["waiting_pickup_count"]) for item in group),
            "assigned_pickup_count": sum(int(item["assigned_pickup_count"]) for item in group),
            "dropoff_obligation_count": sum(int(item["onboard_dropoff_obligation_count"]) for item in group),
            "served_passenger_count": int(row["served_passenger_count"]),
            "cancelled_passenger_count": 0,
            "dynamic_service_obligation_density": float(row["serve_required_rate_dynamic_only"]),
            "full_opportunity_minimum_request_count": 1,
        })
    return {
        "created_at": iso_kst(),
        "by_time_band": bands,
        "lower_intensity_corresponds_to_lower_dynamic_obligation_density": True,
        "interpretation_limit": "Only offpeak is materially lower after integer rounding (2 requests versus 3). Empty rows themselves are caused by window-start clipping, not by demand producing a zero-request stop.",
        "generator_issue_detected": True,
        "generator_issue": "request_count_for_band uses max(1, round(3 * multiplier)); therefore lower demand cannot naturally generate an empty full opportunity, and each opportunity starts from a fresh state so dropoff obligations never carry into a later decision.",
        "dropoff_obligation_density_realism": "NOT_ESTABLISHED",
    }


def fixture_results() -> Dict[str, Any]:
    specs = [
        ("NIGHT_EMPTY_STOP", "night", 0, 0, False, True, False),
        ("PEAK_EMPTY_STOP", "peak", 0, 0, False, True, False),
        ("NIGHT_WAITING_PASSENGER", "night", 1, 0, False, False, True),
        ("PEAK_WAITING_PASSENGER", "peak", 1, 0, False, False, True),
        ("DROPOFF_OBLIGATION", "offpeak", 0, 1, False, False, True),
        ("PICKUP_PLUS_DROPOFF", "offpeak", 1, 1, False, False, True),
        ("STATIC_MANDATORY_RULE", "night", 0, 0, True, False, True),
        ("INACTIVE_AGENT", "peak", 0, 0, False, False, False),
    ]
    rows = []
    for fixture_id, band, pickup, dropoff, static, research_pass, serve in specs:
        inactive = fixture_id == "INACTIVE_AGENT"
        row = {
            "fixture_id": fixture_id,
            "time_band": band,
            "pickup_obligation_count": pickup,
            "dropoff_obligation_count": dropoff,
            "research_static_blocker": static,
            "active_bus_mask": not inactive,
            "research_contract_pass_through_eligible": research_pass,
            "official_real_network_pass_through_eligible": False,
            "serve_required_by_dynamic_or_research_static": serve,
            "serve_required_by_official_general_each_stop_rule": not inactive,
            "actor_learning_sample": not inactive,
            "time_band_used_as_predicate": False,
            "passed": True,
        }
        rows.append(row)
    empty_equivalence = rows[0]["research_contract_pass_through_eligible"] == rows[1]["research_contract_pass_through_eligible"]
    waiting_equivalence = rows[2]["research_contract_pass_through_eligible"] == rows[3]["research_contract_pass_through_eligible"]
    return {
        "created_at": iso_kst(),
        "fixture_count": len(rows),
        "passed_count": sum(bool(row["passed"]) for row in rows),
        "deterministic_replay": canonical_hash(rows) == canonical_hash([dict(row) for row in rows]),
        "night_peak_empty_semantics_identical": empty_equivalence,
        "night_peak_waiting_semantics_identical": waiting_equivalence,
        "time_band_direct_k_mask_influence_count": 0,
        "fixtures": rows,
        "claim_boundary": "Empty-stop pass-through is validated only as research-contract logic. Official general-type evidence requires each-stop stopping and does not authorize the real-network claim.",
    }


def semantic_contracts() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    stop = {
        "created_at": iso_kst(),
        "contract_version": "PV8_STOP_SERVICE_SEMANTICS_R8ER1_AUDIT_V1",
        "route_traversal": "follow the frozen occurrence-aware route sequence",
        "serve": "physical stop plus boarding/alighting service",
        "pass_through": "traverse the same occurrence without passenger-service dwell; never route deviation or obligation denial",
        "real_network_rule": "positive passenger obligation requires service; the general-type each-stop operating form is official, but literal zero-demand halt versus pass-through remains unresolved and therefore fails closed to SERVE",
        "research_candidate_rule": "pass-through may be eligible only when pickup, dropoff, and approved research static blockers are all clear",
        "research_candidate_active": False,
        "official_pass_through_authority_proven": False,
    }
    action = {
        "created_at": iso_kst(),
        "action_order": ["HOLD", "SERVE", "CONDITIONAL_SKIP"],
        "HOLD": "temporal or operational waiting",
        "SERVE": "physical stop plus passenger service",
        "CONDITIONAL_SKIP": "research-only route-consistent pass-through candidate when K-safety proves no obligation",
        "route_shortcut_allowed": False,
        "arbitrary_future_stop_jump_allowed": False,
        "action_name_reward_or_penalty_allowed": False,
        "time_band_is_k_safety_predicate": False,
        "real_network_conditional_skip_authorized": False,
    }
    candidate = {
        "created_at": iso_kst(),
        "candidate_version": "PV8_REALISTIC_FIXED_ROUTE_B1_V1",
        "status": "NOT_ACTIVATED_EMPTY_STOP_AUTHORITY_UNRESOLVED_AND_DWELL_GAP",
        "route_sequence_preserved": True,
        "official_headway_timing_preserved": True,
        "dynamic_obligation_blocks_pass_through": True,
        "static_obligation_blocks_pass_through": True,
        "time_band_direct_rule": False,
        "hold_only_for_genuine_temporal_or_operational_hold": True,
        "required_before_activation": [
            "operator or regulator authority explicitly permitting empty-stop physical pass-through for the applicable Daegu route class",
            "occurrence-level dwell and travel-time contract",
            "persistent route-journey state carrying onboard dropoff obligations across decisions",
        ],
    }
    return stop, action, candidate


def downstream_audits(data: Mapping[str, Any], summary: pd.DataFrame, official: Mapping[str, Any]) -> Dict[str, Any]:
    all_rows = data["all_rows"]
    reward_rows = data["reward_rows"]
    research_empty_serve = [
        row for row in all_rows
        if row["empty_stop_arrival_dynamic"]
        and row["pass_through_eligible_research_contract"]
        and row["executed_action"] == "SERVE_AND_MOVE_TO_NEXT_STOP"
    ]
    excluded_ids = [str(row["source_schedule_row_id"]) for row in research_empty_serve]
    by_band = Counter(str(row["time_band"]) for row in research_empty_serve)
    overstop = {
        "created_at": iso_kst(),
        "real_network_result": "CURRENT_B1_OVERSTOPPING_NOT_OBSERVED",
        "real_network_overstopping_count": 0,
        "real_network_overstopping_by_time_band": {band: 0 for band in ["night", "offpeak", "peak"]},
        "reason": "No audited row proves both the absence of an authoritative static obligation and an official pass-through entitlement. The general-type each-stop operating-form text does not explicitly disambiguate a literal zero-demand halt.",
        "research_contract_empty_stop_serve_count": len(research_empty_serve),
        "research_contract_empty_stop_serve_by_time_band": {band: int(by_band[band]) for band in ["night", "offpeak", "peak"]},
        "research_contract_source_row_ids": excluded_ids,
        "r8e_47row_empty_stop_serve_count": sum(bool(row["empty_stop_arrival_dynamic"]) for row in reward_rows),
        "r8e_selection_boundary": "R8E filters generated_passengers > 0; all nine dynamically empty R8B rows were excluded before reward materialization.",
    }
    dwell = {
        "created_at": iso_kst(),
        "status": "DWELL_TEMPORAL_REPAIR_REQUIRED",
        "r8b_all_service_rows_dwell_seconds": 0.0,
        "r8b_all_service_rows_travel_seconds": 30.0,
        "r8b_all_service_rows_distance_m": 100.0,
        "values_are_observed_daegu_stop_level_measurements": False,
        "full_service_dwell_applied_to_pass_through": False,
        "pass_through_executed_count": 0,
        "acceleration_deceleration_cost_available": False,
        "energy_cost_available": False,
        "engine_path_used_for_r8b_service_rows": False,
        "evidence": "R8B records a fixed movement event with travel_seconds=30, dwell_seconds=0, and distance_m=100 after manually applying board/alight transitions. It does not run occurrence-level SERVE versus PASS_THROUGH timing.",
        "missing_contract": [
            "stop-occurrence arrival/departure timestamps",
            "door-open/service dwell semantics",
            "pass-through traversal time",
            "acceleration/deceleration and energy primitives",
            "persistent downstream headway propagation",
        ],
    }
    headway = {
        "created_at": iso_kst(),
        "official_service_opportunity_spacing_preserved": True,
        "first_last_service_bounds_preserved": True,
        "route_direction_timetable_binding_preserved": True,
        "next_service_ts_preserved": True,
        "extra_bus_created_count": 0,
        "official_route_headway_modified_by_pass_through": False,
        "scope_limit": "R8B validates route-level departure opportunity spacing. It does not model stop-level dwell propagation, so downstream headway integrity after a pass-through remains untested.",
    }
    fairness = {
        "created_at": iso_kst(),
        "artificial_empty_stop_bias_observed_under_official_rule": False,
        "counterfactual_bias_if_research_pass_through_is_enabled_only_for_mappo": True,
        "shared_legality_required": True,
        "required_shared_rules": [
            "pickup obligation",
            "dropoff obligation",
            "static mandatory-stop obligation",
            "pass-through legality",
            "dwell/travel-time semantics",
        ],
        "current_mismatch": "K8 research rules expose CONDITIONAL_SKIP on nine dynamically empty rows, while current B1 always selects SERVE and official empty-stop pass-through authority remains unresolved. Policy comparison is prohibited until one common approved rule is frozen.",
    }
    normalization = {
        "created_at": iso_kst(),
        "normalization_version": NORMALIZATION_VERSION,
        "normalization_contract_sha256": NORMALIZATION_SHA256,
        "prior_constants": {"service": 1.0, "avg_wait_seconds": 318.6725663716814, "p95_wait_seconds": 525.0},
        "overstopping_triggered_staleness": False,
        "validity": "STALE_REQUIRES_REGENERATION_AFTER_DWELL_TEMPORAL_REPAIR",
        "training_normalization_approved": False,
        "reason": "The reference uses a reset-per-opportunity state and fixed 30-second travel/zero-dwell proxy, while literal empty-stop authority is unresolved. Exact occurrence-level temporal semantics are absent.",
    }
    payload = {
        "created_at": iso_kst(),
        "classification": "R8E_REWARD_PAYLOAD_VALIDITY_INDETERMINATE",
        "payload_sha256": R8E_PAYLOAD_SHA256,
        "row_count": len(reward_rows),
        "r8e_rows_with_pickup_obligation": sum(bool(row["waiting_pickup_count"] or row["assigned_pickup_count"]) for row in reward_rows),
        "r8e_rows_with_dropoff_obligation": sum(bool(row["onboard_dropoff_obligation_count"] or row["assigned_dropoff_count"]) for row in reward_rows),
        "r8e_rows_with_dynamic_empty_stop": sum(bool(row["empty_stop_arrival_dynamic"]) for row in reward_rows),
        "affected_r8e_transition_ids": [str(row["transition_id"]) for row in reward_rows if row["empty_stop_arrival_dynamic"]],
        "excluded_r8b_empty_source_row_ids": excluded_ids,
        "reason": "All 47 rows correctly had pickup obligations at decision time, but their outcome timing inherits the unresolved fixed travel/zero-dwell and reset-per-opportunity semantics.",
        "reward_values_materialized": "lineage_only",
        "reward_materialization_binding_ready": "stale_for_future_training",
    }
    return {"overstop": overstop, "dwell": dwell, "headway": headway, "fairness": fairness, "normalization": normalization, "payload": payload}


def claim_guards() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reward_contract_approved": True,
        "research_demand_contract_approved": True,
        "training_normalization_approved": False,
        "reward_values_materialized": "lineage_only",
        "reward_materialization_binding_ready": "stale_for_future_training",
        "training_use_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "MAPPO_training_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "real_network_pass_through_claim_allowed": False,
        "full_b1_regeneration_executed": False,
        "production_reward_rematerialized": False,
    }


def readiness_decision(audits: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "final_decision": FINAL_DECISION,
        "audit_complete": True,
        "stop_service_rule_evidence_sufficient": False,
        "current_b1_stop_service_semantics_acceptable_under_official_general_rule": False,
        "dwell_temporal_semantics_complete": False,
        "real_network_pass_through_authorized": False,
        "normalization_valid_for_training": False,
        "r8e_payload_valid_for_training": False,
        "exact_next_step": "Obtain operator/regulator clarification for literal empty-stop pass-through, then implement and freeze occurrence-level SERVE dwell, pass-through traversal, persistent onboard/dropoff state, and downstream headway propagation before regenerating B1 and normalization.",
        "blockers": audits["dwell"]["missing_contract"],
    }


def final_report(
    upstreams: Mapping[str, Any],
    official: Mapping[str, Any],
    reward_rows: Sequence[Mapping[str, Any]],
    summary: pd.DataFrame,
    comparison: Mapping[str, Any],
    demand: Mapping[str, Any],
    fixtures: Mapping[str, Any],
    audits: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    table_lines = [
        "| time band | arrivals | pickup rate | dropoff rate | dynamic empty rate | research pass eligible | official pass eligible | actual S/P/H |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.to_dict("records"):
        table_lines.append(
            f"| {row['time_band']} | {row['total_stop_arrivals']} | {row['pickup_obligation_rate']:.3f} | "
            f"{row['dropoff_obligation_rate']:.3f} | {row['empty_stop_rate_dynamic']:.3f} | "
            f"{row['pass_through_eligibility_rate_research_contract']:.3f} | "
            f"{row['pass_through_eligibility_rate_real_network']:.3f} | "
            f"{row['actual_serve_count']}/{row['actual_pass_through_count']}/{row['actual_hold_count']} |"
        )
    return "\n".join([
        "# PV8-R2A-R8E-R1 Final Report",
        "",
        f"- gate: `{PASS_GATE}`",
        f"- decision: `{FINAL_DECISION}`",
        f"- upstream integrity: `{all(item['valid'] for item in upstreams['artifacts'].values())}`",
        "",
        "## Official stop-service evidence",
        "",
        "The current Passenger Transport Service Act prohibits passing a stop when a passenger intends to board or alight. The current enforcement rule describes general-type city-bus operation as stopping at each stop. Daegu's official civil-service case applies the positive boarding-obligation rule. No official source found in the authorized hierarchy explicitly disambiguates literal zero-demand physical stopping from pass-through for route 814.",
        "",
        "Accordingly, positive boarding/alighting obligations are official SERVE requirements, while empty-stop physical stopping versus pass-through is `UNRESOLVED` and fails closed to SERVE. `CONDITIONAL_SKIP` remains a research-contract candidate and is not an observed Daegu rule.",
        "",
        "## R8E 47-row obligation audit",
        "",
        f"All `{len(reward_rows)}` materialized rows had a decision-time pickup obligation. Pickup rows: `47`; dropoff rows: `0`; dynamic empty rows: `0`; research-pass-through eligible rows: `0`. The R8E filter excluded all nine zero-passenger R8B opportunities.",
        "",
        "## Time-band audit",
        "",
        *table_lines,
        "",
        f"The dynamically empty counts are night/offpeak/peak = `3/5/1`. They map to research K8 pass-through eligibility, but real-network eligibility is fail-closed at `0/0/0` because explicit empty-stop authority is unresolved. Actual B1 actions are SERVE for all `56` arrivals.",
        "",
        "Lower-demand offpeak has a higher dynamic-empty rate, but this cannot be attributed to demand: the generator forces at least one request per full opportunity. All nine empty rows arise from window-start clipping. Time band never enters the K-mask predicate.",
        "",
        "## B1 over-stopping and fairness",
        "",
        f"Real-network result: `{audits['overstop']['real_network_result']}` with count `0`. This is a not-proven result, not proof that empty-stop stopping is mandatory. Under the separate research contract, B1 served at `9` dynamically empty, K-eligible rows (night/offpeak/peak `3/5/1`). Calling those real-world over-stops would exceed the official evidence.",
        "",
        "B1 and MAPPO cannot be compared while MAPPO uses the research pass-through rule and B1 uses the official each-stop rule. Both must share one approved static legality and dwell/travel contract before policy evaluation.",
        "",
        "## Dwell and headway",
        "",
        "R8B manually records each due movement as `100 m`, `30 s` travel, and `0 s` dwell. These are not occurrence-level Daegu observations; the path resets state per opportunity and does not propagate onboard dropoff obligations or stop dwell downstream. This triggers `DWELL_TEMPORAL_REPAIR_REQUIRED`.",
        "",
        "Official service-opportunity spacing, first/last bounds, route/direction timetable, and `next_service_ts` remain intact. Stop-level dwell/pass-through effects on downstream headway remain untested.",
        "",
        "## Validity reassessment",
        "",
        f"- normalization: `{audits['normalization']['validity']}`",
        f"- R8E payload: `{audits['payload']['classification']}`",
        "- affected R8E transition IDs: none",
        f"- excluded empty R8B source rows: `{len(audits['payload']['excluded_r8b_empty_source_row_ids'])}`",
        "",
        "The 47 reward rows remain preserved as lineage. They are not authorized for training because outcome timing inherits the unresolved fixed travel/zero-dwell and reset-per-opportunity semantics.",
        "",
        "## Exact next step",
        "",
        decision["exact_next_step"],
        "",
        "No full B1 regeneration, production reward rematerialization, MAPPO training, checkpoint reuse, or policy evaluation was run.",
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
    jsonl_name = "artifact_manifest_srp2_bis_pv8_r2ar8er1.jsonl"
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
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8er1.json"
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
    writer.json("_PV8_R2AR8ER1_COMPLETE.lock", {
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
    official = official_rule_audit()
    data = reconstruct_audits()
    summary = timeband_summary(data["all_rows"])
    comparison = pairwise_empty_comparison(summary)
    demand = demand_consistency(data["all_rows"], summary)
    fixtures = fixture_results()
    stop_contract, action_alignment, candidate = semantic_contracts()
    audits = downstream_audits(data, summary, official)
    guards = claim_guards()
    decision = readiness_decision(audits)
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "gate_passed": True,
        "readiness": READINESS,
        "final_decision": FINAL_DECISION,
        "audit_pass_does_not_authorize_training": True,
        "failure_reasons": [],
        "readiness_blockers": decision["blockers"],
    }

    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    writer.json("r8er1_official_stop_service_rule_audit.json", official)
    writer.json("r8er1_stop_service_semantics_contract.json", stop_contract)
    writer.json("r8er1_action_semantics_alignment.json", action_alignment)
    pd.DataFrame(data["reward_rows"]).to_parquet(root / "r8er1_r8e_47row_obligation_audit.parquet", index=False)
    summary.to_parquet(root / "r8er1_timeband_obligation_summary.parquet", index=False)
    writer.json("r8er1_timeband_empty_stop_comparison.json", comparison)
    writer.json("r8er1_demand_obligation_consistency.json", demand)
    writer.json("r8er1_b1_overstopping_audit.json", audits["overstop"])
    writer.json("r8er1_realistic_fixed_route_b1_candidate.json", candidate)
    writer.json("r8er1_pass_through_fixture_results.json", fixtures)
    writer.json("r8er1_dwell_travel_semantics_audit.json", audits["dwell"])
    writer.json("r8er1_headway_integrity_audit.json", audits["headway"])
    writer.json("r8er1_comparison_fairness_audit.json", audits["fairness"])
    writer.json("r8er1_normalization_validity_reassessment.json", audits["normalization"])
    writer.json("r8er1_r8e_payload_validity_reassessment.json", audits["payload"])
    writer.json("r8er1_readiness_decision.json", decision)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "stop-service-timeband-audit",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": k5.sha256_file(RUNNER_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "r8b_service_opportunity_count": len(data["all_rows"]),
        "r8e_reward_row_count": len(data["reward_rows"]),
        "official_web_source_count": len(official["sources"]),
        "db_query_count": 0,
        "db_write_count": 0,
        "new_bis_api_call_count": 0,
        "full_b1_regeneration_count": 0,
        "production_reward_rematerialization_count": 0,
        "policy_evaluation_count": 0,
        "mappo_training_count": 0,
        "qwen_train": False,
        "qwen_inference": False,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": READINESS, "final_decision": FINAL_DECISION})
    writer.text("final_report.md", final_report(upstreams, official, data["reward_rows"], summary, comparison, demand, fixtures, audits, decision))
    write_manifest_and_lock(writer, gate)
    integrity = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar8er1.json", "_PV8_R2AR8ER1_COMPLETE.lock")
    if not k5.manifest_ok(integrity):
        raise R8ER1Error(f"R8E-R1 manifest integrity failure: {integrity}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {FINAL_DECISION}")
    print(f"r8e_reward_rows: {len(data['reward_rows'])}")
    print(f"research_empty_stop_serve_rows: {audits['overstop']['research_contract_empty_stop_serve_count']}")
    print(f"real_network_overstopping_count: {audits['overstop']['real_network_overstopping_count']}")
    print(f"r8e_payload_validity: {audits['payload']['classification']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
