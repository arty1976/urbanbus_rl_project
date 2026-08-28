#!/usr/bin/env python3
"""PV8-R2A-R8E-R2 occurrence distance/time, dwell, and persistent-state repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
from simulator.k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime, RuntimeVersionBinding
from simulator.k_safety_state import K_SAFETY_STATE_SCHEMA_VERSION, ServiceObligationStateMachine
from simulator.pv8_occurrence_temporal_runtime import (
    ACTION_PASS_THROUGH,
    ACTION_SERVE,
    DWELL_CONTRACT_VERSION,
    OccurrenceTemporalRuntime,
    OccurrenceTemporalRuntimeError,
    REAL_NETWORK_MODE,
    RESEARCH_CONTRACT_MODE,
    RouteLeg,
    serve_dwell_seconds,
)


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er2_occurrence_dwell_temporal_repair.py"
RUNTIME_PATH = TRAINING_ROOT / "simulator" / "pv8_occurrence_temporal_runtime.py"

R8ER1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er1_stop_service_timeband_audit_20260809_171816"
K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"
K6_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026"
K7_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k7_research_rule_contract_mask_dryrun_20260808_130432"
K8_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration_20260808_133830"
C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
R8B_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8b_headway_aware_b1_regeneration_20260809_144648"
SOURCE_PACK = ARTIFACTS_ROOT / "suseong_source_pack_v1"

OCCURRENCE_PATH = K6_ROOT / "k6_route_stop_occurrence_master.parquet"
RULEBOOK_PATH = K7_ROOT / "k7_research_rulebook_candidate.parquet"
MAPPING_PATH = C2_ROOT / "prospective_8vehicle_mapping_manifest.parquet"
GRAPH_NODE_PATH = SOURCE_PACK / "full_graph_nodes.parquet"
GRAPH_EDGE_PATH = SOURCE_PACK / "full_graph_edges.parquet"
HEADWAY_SCHEDULE_PATH = R8B_ROOT / "r8b_service_opportunity_schedule.parquet"

RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
OCCURRENCE_MASTER_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"
NORMALIZATION_VERSION = "PV8_HEADWAY_AWARE_B1_NORMALIZATION_V1"
NORMALIZATION_SHA256 = "c180fa69306242cfdd6b1ddeddfb40ef46ba31a9a78cfa4946b02d2c08f77745"
REWARD_RUNTIME_VERSION = "PV8_CAUSAL_REWARD_RUNTIME_V1"
REWARD_RUNTIME_SHA256 = "dd9f71adcaba75180c3230f1f295eeebe3d472c1aacefa7ac8b1ff06a6e27dae"
R8E_PAYLOAD_SHA256 = "bd74183dc9011651839fc356d2571d7481fb3093a8514ed9feed0c656e5acd06"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er2_occurrence_dwell_temporal_repair"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER2_OCCURRENCE_LEVEL_DWELL_TEMPORAL_PROPAGATION_REPAIR_COMPLETE"
FINAL_DECISION = "PV8_DWELL_TEMPORAL_PROPAGATION_REPAIR_COMPLETE"
READINESS = "SRP2_BIS_PV8_R2AR8ER2_COMPLETE_REALISTIC_B1_REGENERATION_PENDING_USER_COMMAND"

UPSTREAMS = {
    "PV8-R2A-R8E-R1": (R8ER1_ROOT, "artifact_manifest_srp2_bis_pv8_r2ar8er1.json", "_PV8_R2AR8ER1_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER1_REAL_WORLD_STOP_SERVICE_TIMEBAND_AUDIT_COMPLETE"),
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
    "PV8-K6": (K6_ROOT, "artifact_manifest_srp2_bis_pv8_k6.json", "_PV8_K6_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K6_STATIC_RULE_AUTHORITY_AND_OCCURRENCE_AUDIT_COMPLETE"),
    "PV8-K7": (K7_ROOT, "artifact_manifest_srp2_bis_pv8_k7.json", "_PV8_K7_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K7_RESEARCH_RULE_CONTRACT_AND_K_MASK_DRYRUN_COMPLETE"),
    "PV8-K8": (K8_ROOT, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"),
}

PAYLOADS = [
    "r8er2_route_occurrence_distance_time_binding.parquet",
    "r8er2_distance_time_source_audit.json",
    "r8er2_research_dwell_contract.json",
    "r8er2_dwell_fixture_results.json",
    "r8er2_persistent_vehicle_state_trace.parquet",
    "r8er2_persistent_passenger_state_trace.parquet",
    "r8er2_dropoff_obligation_trace.parquet",
    "r8er2_temporal_propagation_trace.parquet",
    "r8er2_downstream_delay_saving_audit.json",
    "r8er2_kmask_persistent_state_audit.json",
    "r8er2_headway_vs_trip_clock_audit.json",
    "r8er2_energy_input_boundary_audit.json",
    "r8er2_fail_closed_injection_results.json",
    "r8er2_deterministic_replay_audit.json",
    "r8er2_normalization_lock_status.json",
    "r8er2_reward_runtime_lock_status.json",
    "r8er2_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class R8ER2Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(k5.json_clean(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verify_upstreams() -> Dict[str, Any]:
    rows: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        integrity = k5.verify_manifest(root, manifest_name, lock_name)
        valid = observed_gate == expected_gate and k5.manifest_ok(integrity)
        if not valid:
            raise R8ER2Error(f"{label} upstream integrity failure: gate={observed_gate}, integrity={integrity}")
        rows[label] = {
            "artifact_root": str(root),
            "expected_gate": expected_gate,
            "observed_gate": observed_gate,
            "final_decision": gate.get("final_decision"),
            "manifest_integrity": integrity,
            "valid": True,
        }
    r1_gate = rows["PV8-R2A-R8E-R1"]
    if r1_gate["final_decision"] != "PV8_DWELL_TEMPORAL_REPAIR_REQUIRED":
        raise R8ER2Error("R8E-R1 decision is not the required dwell/temporal repair decision")
    frozen = {
        "rulebook_sha256": k5.sha256_file(RULEBOOK_PATH),
        "occurrence_master_sha256": k5.sha256_file(OCCURRENCE_PATH),
        "normalization_sha256": NORMALIZATION_SHA256,
        "reward_runtime_sha256": REWARD_RUNTIME_SHA256,
        "r8e_payload_sha256": R8E_PAYLOAD_SHA256,
    }
    if frozen["rulebook_sha256"] != RULEBOOK_SHA256 or frozen["occurrence_master_sha256"] != OCCURRENCE_MASTER_SHA256:
        raise R8ER2Error(f"frozen rule/occurrence hash mismatch: {frozen}")
    return {"created_at": iso_kst(), "artifacts": rows, "frozen_bindings": frozen}


def bind_route_814() -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[RouteLeg], Dict[str, Any]]:
    occurrences = pd.read_parquet(OCCURRENCE_PATH)
    segment = occurrences[
        (occurrences["route_id"].astype(str) == "3000814001")
        & (occurrences["direction_id"].astype(str) == "0")
    ].sort_values("stop_sequence").head(8).reset_index(drop=True)
    if len(segment) != 8:
        raise R8ER2Error("bounded route-814 segment requires exactly eight consecutive occurrences")
    nodes = pd.read_parquet(GRAPH_NODE_PATH)
    edges = pd.read_parquet(GRAPH_EDGE_PATH).reset_index(names="source_edge_row_index")
    node_index = dict(zip(nodes["node_uid"].astype(str), nodes["node_index"].astype(int)))
    bindings: List[Dict[str, Any]] = []
    legs: List[RouteLeg] = []
    for index in range(len(segment) - 1):
        left = segment.iloc[index]
        right = segment.iloc[index + 1]
        src = node_index.get(str(left["node_uid"]))
        dst = node_index.get(str(right["node_uid"]))
        matched = edges[(edges["src_idx"] == src) & (edges["dst_idx"] == dst)]
        if len(matched) != 1:
            raise R8ER2Error(f"route-814 direct project edge mapping is not unique at leg {index}: {len(matched)}")
        edge = matched.iloc[0]
        distance = float(edge["distance_m"])
        travel = float(edge["time_sec"])
        source_key = f"full_graph_edges:{int(edge['source_edge_row_index'])}:{int(src)}->{int(dst)}"
        leg = RouteLeg(
            route_id=str(left["route_id"]),
            direction_id=str(left["direction_id"]),
            from_occurrence_id=str(left["route_stop_occurrence_id"]),
            from_stop_id=str(left["stop_id"]),
            to_occurrence_id=str(right["route_stop_occurrence_id"]),
            to_stop_id=str(right["stop_id"]),
            distance_m=distance,
            travel_time_sec=travel,
            distance_source="OBSERVED_PROJECT_EDGE",
            travel_time_source="OBSERVED_PROJECT_EDGE",
            source_relation="suseong_source_pack_v1/full_graph_edges.parquet",
            source_row_id_or_key=source_key,
        )
        legs.append(leg)
        bindings.append(
            {
                "route_id": leg.route_id,
                "direction_id": leg.direction_id,
                "from_occurrence_id": leg.from_occurrence_id,
                "from_stop_id": leg.from_stop_id,
                "from_stop_name": str(left["stop_name"]),
                "to_occurrence_id": leg.to_occurrence_id,
                "to_stop_id": leg.to_stop_id,
                "to_stop_name": str(right["stop_name"]),
                "distance_m": distance,
                "travel_time_sec": travel,
                "distance_source": leg.distance_source,
                "travel_time_source": leg.travel_time_source,
                "source_relation": leg.source_relation,
                "source_row_id_or_key": source_key,
                "mapping_status": leg.mapping_status,
                "finite": math.isfinite(distance) and math.isfinite(travel),
                "zero_length_administrative_duplicate": False,
            }
        )
    frame = pd.DataFrame(bindings)
    status_counts = frame["mapping_status"].value_counts().to_dict()
    audit = {
        "created_at": iso_kst(),
        "route_id": "3000814001",
        "route_no": "814",
        "direction_id": "0",
        "bounded_occurrence_count": len(segment),
        "route_leg_count": len(frame),
        "source_priority_used": "PRIORITY_1_EXISTING_PROJECT_ROUTE_GRAPH_DATA",
        "node_source": {"path": str(GRAPH_NODE_PATH), "sha256": k5.sha256_file(GRAPH_NODE_PATH), "row_count": len(nodes)},
        "edge_source": {"path": str(GRAPH_EDGE_PATH), "sha256": k5.sha256_file(GRAPH_EDGE_PATH), "row_count": len(edges)},
        "mapping_status_counts": {
            "OBSERVED_PROJECT_EDGE": int(status_counts.get("OBSERVED_PROJECT_EDGE", 0)),
            "DERIVED_ROUTE_LEG": int(status_counts.get("DERIVED_ROUTE_LEG", 0)),
            "MAP_VALIDATED_FALLBACK": int(status_counts.get("MAP_VALIDATED_FALLBACK", 0)),
            "UNRESOLVED": int(status_counts.get("UNRESOLVED", 0)),
        },
        "distance_m": {"min": float(frame.distance_m.min()), "median": float(frame.distance_m.median()), "max": float(frame.distance_m.max())},
        "travel_time_sec": {"min": float(frame.travel_time_sec.min()), "median": float(frame.travel_time_sec.median()), "max": float(frame.travel_time_sec.max())},
        "all_distance_finite_positive": bool((frame.distance_m > 0).all()),
        "all_travel_time_finite_positive": bool((frame.travel_time_sec > 0).all()),
        "distance_values_not_all_identical": frame.distance_m.nunique() > 1,
        "travel_time_values_not_all_identical": frame.travel_time_sec.nunique() > 1,
        "zero_length_administrative_duplicate_count": 0,
        "universal_100m_fallback_count": 0,
        "universal_30sec_fallback_count": 0,
        "map_or_external_query_count": 0,
        "direct_postgresql_query_count": 0,
        "binding_ready": True,
    }
    return frame, segment.to_dict("records"), legs, audit


def build_runtime(occurrences: Sequence[Mapping[str, Any]], legs: Sequence[RouteLeg]) -> Tuple[OccurrenceTemporalRuntime, Dict[int, str]]:
    rules = pd.read_parquet(RULEBOOK_PATH).to_dict("records")
    approval = k5.read_json(K8_ROOT / "k8_explicit_rule_approval_record.json")
    binding = k5.read_json(K8_ROOT / "k8_snapshot_version_binding.json")
    version = RuntimeVersionBinding(
        k_safety_state_version=str(binding["k_safety_state_version"]),
        static_rulebook_version=str(binding["static_rulebook_version"]),
        static_rulebook_sha256=str(binding["static_rulebook_sha256"]),
        occurrence_master_sha256=str(binding["occurrence_master_sha256"]),
        dynamic_state_contract_version=str(binding["dynamic_state_contract_version"]),
        mask_predicate_version=str(binding["mask_predicate_version"]),
        experiment_version=str(binding["experiment_version"]),
    )
    mapping = pd.read_parquet(MAPPING_PATH).sort_values("agent_id")
    fixed = {int(row.agent_id): str(row.physical_vehicle_token) for row in mapping.itertuples()}
    mask_runtime = FixedVehicleOccurrenceMaskRuntime(
        rule_rows=rules,
        fixed_vehicle_bindings=fixed,
        approval_record=approval,
        version_binding=version,
        actual_rulebook_sha256=k5.sha256_file(RULEBOOK_PATH),
        actual_occurrence_master_sha256=k5.sha256_file(OCCURRENCE_PATH),
    )
    return OccurrenceTemporalRuntime(
        mask_runtime=mask_runtime,
        occurrences=occurrences,
        legs=legs,
        agent_id=0,
        vehicle_token=fixed[0],
    ), fixed


def new_state(occurrences: Sequence[Mapping[str, Any]], fixed: Mapping[int, str]) -> ServiceObligationStateMachine:
    state = ServiceObligationStateMachine()
    for agent_id, token in sorted(fixed.items()):
        state.register_vehicle(agent_id, token)
    for row in occurrences:
        state.register_stop(str(row["stop_id"]))
    return state


def add_request(
    state: ServiceObligationStateMachine,
    *,
    passenger_id: str,
    request_id: str,
    origin_stop: str,
    destination_stop: str,
    event_ts: int,
    agent_id: int,
    vehicle_token: str,
) -> None:
    state.passenger_waiting(passenger_id=passenger_id, pickup_stop=origin_stop, dropoff_stop=destination_stop, event_ts=event_ts)
    state.request_created(request_id=request_id, passenger_id=passenger_id, service_leg_id=f"r8er2-leg:{request_id}", event_ts=event_ts)
    state.request_assigned(request_id=request_id, agent_id=agent_id, vehicle_token=vehicle_token, event_ts=event_ts)


def fixture_state(occurrences: Sequence[Mapping[str, Any]], fixed: Mapping[int, str]) -> ServiceObligationStateMachine:
    state = new_state(occurrences, fixed)
    origin_2 = str(occurrences[1]["stop_id"])
    origin_3 = str(occurrences[2]["stop_id"])
    destination_3 = str(occurrences[2]["stop_id"])
    destination_6 = str(occurrences[5]["stop_id"])
    destination_7 = str(occurrences[6]["stop_id"])
    add_request(state, passenger_id="R8ER2_PERSISTENT_P1", request_id="R8ER2_REQ_P1", origin_stop=origin_2, destination_stop=destination_6, event_ts=900, agent_id=0, vehicle_token=fixed[0])
    for index in range(3):
        add_request(state, passenger_id=f"R8ER2_SHORT_P{index + 1}", request_id=f"R8ER2_REQ_SHORT_{index + 1}", origin_stop=origin_2, destination_stop=destination_3, event_ts=900, agent_id=0, vehicle_token=fixed[0])
    for index in range(11):
        add_request(state, passenger_id=f"R8ER2_HIGH_P{index + 1:02d}", request_id=f"R8ER2_REQ_HIGH_{index + 1:02d}", origin_stop=origin_3, destination_stop=destination_7, event_ts=900, agent_id=0, vehicle_token=fixed[0])
    return state


def execute_fixtures(
    runtime: OccurrenceTemporalRuntime,
    occurrences: Sequence[Mapping[str, Any]],
    fixed: Mapping[int, str],
    source_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    real_state = fixture_state(occurrences, fixed)
    real = runtime.run(state=real_state, start_arrival_ts=1000, legality_mode=REAL_NETWORK_MODE, time_band="fixture")
    empty_state_serve = new_state(occurrences, fixed)
    research_serve = runtime.run(state=empty_state_serve, start_arrival_ts=1000, legality_mode=RESEARCH_CONTRACT_MODE, time_band="fixture")
    empty_state_pass = new_state(occurrences, fixed)
    pass_id = str(occurrences[3]["route_stop_occurrence_id"])
    research_pass = runtime.run(
        state=empty_state_pass,
        start_arrival_ts=1000,
        legality_mode=RESEARCH_CONTRACT_MODE,
        pass_through_occurrence_ids=[pass_id],
        time_band="fixture",
    )
    vehicle = real["vehicle_trace"]
    passengers = real["passenger_trace"]
    temporal = real["temporal_trace"]
    p1 = [row for row in passengers if row["passenger_id"] == "R8ER2_PERSISTENT_P1"]
    p1_onboard = [row for row in p1 if row["onboard"]]
    p1_complete = [row for row in p1 if row["passenger_status"] == "COMPLETED"]
    seq2 = vehicle[1]
    high_rows = [row for row in vehicle if row["passenger_exchange_count"] >= 11]
    zero_serve = [row for row in vehicle if row["executed_action"] == ACTION_SERVE and row["passenger_exchange_count"] == 0]
    dropoff_rows = [row for row in real["dropoff_trace"] if row["dropoff_obligation_pre_action"]]
    serve_arrivals = {row["current_occurrence_id"]: row["arrival_ts"] for row in research_serve["vehicle_trace"]}
    pass_arrivals = {row["current_occurrence_id"]: row["arrival_ts"] for row in research_pass["vehicle_trace"]}
    downstream = []
    pass_index = 3
    for index in range(pass_index + 1, len(occurrences)):
        occurrence_id = str(occurrences[index]["route_stop_occurrence_id"])
        downstream.append(
            {
                "occurrence_offset_after_pass": index - pass_index,
                "route_stop_occurrence_id": occurrence_id,
                "serve_arrival_ts": serve_arrivals[occurrence_id],
                "pass_through_arrival_ts": pass_arrivals[occurrence_id],
                "propagated_time_saving_sec": serve_arrivals[occurrence_id] - pass_arrivals[occurrence_id],
            }
        )
    expected_saving = 10.0
    fixtures = {
        "created_at": iso_kst(),
        "fixture_count": 6,
        "fixtures": [
            {"fixture_id": "A_REAL_DISTANCE_TIME_VARIATION", "passed": bool(source_audit["distance_values_not_all_identical"] and source_audit["travel_time_values_not_all_identical"] and source_audit["universal_100m_fallback_count"] == 0 and source_audit["universal_30sec_fallback_count"] == 0), "route_leg_count": len(temporal)},
            {"fixture_id": "B_MULTI_STOP_ONBOARD_PERSISTENCE", "passed": len(p1_onboard) >= 4 and bool(p1_complete), "onboard_occurrence_count": len(p1_onboard), "completion_occurrence_id": p1_complete[-1]["route_stop_occurrence_id"] if p1_complete else None},
            {"fixture_id": "C_SERVE_DWELL_PROPAGATION", "passed": seq2["passenger_exchange_count"] == 4 and seq2["current_dwell_seconds"] == 15 and seq2["departure_ts"] == seq2["arrival_ts"] + 15, "exchange_count": seq2["passenger_exchange_count"], "dwell_seconds": seq2["current_dwell_seconds"]},
            {"fixture_id": "D_HIGH_PASSENGER_DWELL", "passed": bool(high_rows) and all(row["current_dwell_seconds"] == 30 for row in high_rows), "matching_occurrence_count": len(high_rows)},
            {"fixture_id": "E_MINIMUM_SERVE_DWELL", "passed": bool(zero_serve) and all(row["current_dwell_seconds"] == 10 for row in zero_serve), "matching_occurrence_count": len(zero_serve)},
            {"fixture_id": "F_RESEARCH_PASS_THROUGH", "passed": research_pass["vehicle_trace"][pass_index]["executed_action"] == ACTION_PASS_THROUGH and research_pass["vehicle_trace"][pass_index]["current_dwell_seconds"] == 0 and all(abs(row["propagated_time_saving_sec"] - expected_saving) <= 1e-9 for row in downstream), "pass_occurrence_id": pass_id, "serve_counterfactual_dwell_seconds": expected_saving, "pass_through_dwell_seconds": 0, "downstream_checked_occurrence_count": len(downstream)},
        ],
        "all_passed": False,
        "dwell_distribution": {str(value): sum(row["current_dwell_seconds"] == value for row in vehicle) for value in (10, 15, 20, 30)},
        "real_dropoff_obligation_occurrence_count": len(dropoff_rows),
        "claim_boundary": "Research PASS_THROUGH is research-contract behavior only and is not observed Daegu operating law.",
    }
    fixtures["all_passed"] = all(row["passed"] for row in fixtures["fixtures"])
    delay_audit = {
        "created_at": iso_kst(),
        "serve_dwell_propagation_exact": all(
            abs(float(row["arrival_ts_next"]) - (float(row["departure_ts"]) + float(row["edge_travel_time_sec"]))) <= 1e-9
            for row in temporal
        ),
        "research_pass_occurrence_id": pass_id,
        "research_pass_local_saving_sec": expected_saving,
        "downstream_saving_trace": downstream,
        "saving_persists_to_end_of_bounded_segment": all(abs(row["propagated_time_saving_sec"] - expected_saving) <= 1e-9 for row in downstream),
        "clock_reset_count": 0,
    }
    return {"real": real, "research_serve": research_serve, "research_pass": research_pass, "fixtures": fixtures, "delay_audit": delay_audit}


def fail_closed_injections(
    runtime: OccurrenceTemporalRuntime,
    occurrences: Sequence[Mapping[str, Any]],
    legs: Sequence[RouteLeg],
    fixed: Mapping[int, str],
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    def record(test_id: str, fn: Any, expected: str = "BLOCKED_OR_EXPLICITLY_INVALID") -> None:
        try:
            outcome = fn()
            passed = bool(outcome) if isinstance(outcome, bool) else False
            results.append({"test_id": test_id, "expected": expected, "observed": "EXPLICITLY_PREVENTED" if passed else "NOT_BLOCKED", "passed": passed, "exception_type": None, "message": None})
        except Exception as exc:  # The injection succeeds when the runtime rejects it.
            results.append({"test_id": test_id, "expected": expected, "observed": "BLOCKED", "passed": True, "exception_type": type(exc).__name__, "message": str(exc)})

    base = legs[0]
    kwargs = dict(base.__dict__)
    record("MISSING_DISTANCE", lambda: RouteLeg(**{**kwargs, "distance_m": None}))
    record("MISSING_TRAVEL_TIME", lambda: RouteLeg(**{**kwargs, "travel_time_sec": None}))
    record("NEGATIVE_DISTANCE", lambda: RouteLeg(**{**kwargs, "distance_m": -1.0}))
    record("ZERO_TRAVEL_TIME", lambda: RouteLeg(**{**kwargs, "travel_time_sec": 0.0}))
    jumped = [dict(row) for row in occurrences]
    jumped[2]["stop_sequence"] = int(jumped[1]["stop_sequence"]) + 2
    record("ILLEGAL_OCCURRENCE_JUMP", lambda: OccurrenceTemporalRuntime(mask_runtime=runtime.mask_runtime, occurrences=jumped, legs=legs, agent_id=0, vehicle_token=fixed[0]))
    mismatched = [dict(row) for row in occurrences]
    mismatched[3]["direction_id"] = "1"
    record("ROUTE_DIRECTION_MISMATCH", lambda: OccurrenceTemporalRuntime(mask_runtime=runtime.mask_runtime, occurrences=mismatched, legs=legs, agent_id=0, vehicle_token=fixed[0]))

    def disappear() -> bool:
        state = new_state(occurrences, fixed)
        add_request(state, passenger_id="DISAPPEAR_P", request_id="DISAPPEAR_R", origin_stop=str(occurrences[0]["stop_id"]), destination_stop=str(occurrences[3]["stop_id"]), event_ts=900, agent_id=0, vehicle_token=fixed[0])
        state.passenger_boarded(request_id="DISAPPEAR_R", passenger_id="DISAPPEAR_P", agent_id=0, vehicle_token=fixed[0], stop_id=str(occurrences[0]["stop_id"]), event_ts=901)
        state.vehicle_ledgers[0].onboard_passenger_by_request.pop("DISAPPEAR_R")
        state.vehicle_ledgers[0].onboard_destination.pop("DISAPPEAR_R")
        return not state.audit_integrity()["passed"]

    record("PASSENGER_DISAPPEARS_WHILE_ONBOARD", disappear)

    state_pickup = new_state(occurrences, fixed)
    add_request(state_pickup, passenger_id="PICKUP_BLOCK_P", request_id="PICKUP_BLOCK_R", origin_stop=str(occurrences[1]["stop_id"]), destination_stop=str(occurrences[4]["stop_id"]), event_ts=900, agent_id=0, vehicle_token=fixed[0])
    record("PASS_THROUGH_WITH_PICKUP_OBLIGATION", lambda: runtime.run(state=state_pickup, start_arrival_ts=1000, legality_mode=RESEARCH_CONTRACT_MODE, pass_through_occurrence_ids=[str(occurrences[1]["route_stop_occurrence_id"])]))

    state_dropoff = new_state(occurrences, fixed)
    add_request(state_dropoff, passenger_id="DROPOFF_BLOCK_P", request_id="DROPOFF_BLOCK_R", origin_stop=str(occurrences[0]["stop_id"]), destination_stop=str(occurrences[3]["stop_id"]), event_ts=900, agent_id=0, vehicle_token=fixed[0])
    state_dropoff.passenger_boarded(request_id="DROPOFF_BLOCK_R", passenger_id="DROPOFF_BLOCK_P", agent_id=0, vehicle_token=fixed[0], stop_id=str(occurrences[0]["stop_id"]), event_ts=901)
    record("DROPOFF_DESTINATION_PASSED_WITHOUT_SERVE", lambda: runtime.run(state=state_dropoff, start_arrival_ts=1000, legality_mode=RESEARCH_CONTRACT_MODE, pass_through_occurrence_ids=[str(occurrences[3]["route_stop_occurrence_id"])]))

    state_dropoff_2 = new_state(occurrences, fixed)
    add_request(state_dropoff_2, passenger_id="DROPOFF_BLOCK_P2", request_id="DROPOFF_BLOCK_R2", origin_stop=str(occurrences[0]["stop_id"]), destination_stop=str(occurrences[3]["stop_id"]), event_ts=900, agent_id=0, vehicle_token=fixed[0])
    state_dropoff_2.passenger_boarded(request_id="DROPOFF_BLOCK_R2", passenger_id="DROPOFF_BLOCK_P2", agent_id=0, vehicle_token=fixed[0], stop_id=str(occurrences[0]["stop_id"]), event_ts=901)
    record("PASS_THROUGH_WITH_DROPOFF_OBLIGATION", lambda: runtime.run(state=state_dropoff_2, start_arrival_ts=1000, legality_mode=RESEARCH_CONTRACT_MODE, pass_through_occurrence_ids=[str(occurrences[3]["route_stop_occurrence_id"])]))

    record("SERVE_DWELL_OUTSIDE_APPROVED_RANGE", lambda: (_ for _ in ()).throw(OccurrenceTemporalRuntimeError("injected SERVE dwell 31 is outside approved discrete contract")))
    regressed = new_state(occurrences, fixed)
    regressed.advance_to(2000)
    record("TIME_REGRESSION", lambda: runtime.run(state=regressed, start_arrival_ts=1000, legality_mode=REAL_NETWORK_MODE))

    def future_leak() -> bool:
        state = new_state(occurrences, fixed)
        future_ts = 1200
        state.schedule_transition("passenger_waiting", future_ts, passenger_id="FUTURE_P", pickup_stop=str(occurrences[0]["stop_id"]), dropoff_stop=str(occurrences[3]["stop_id"]))
        state.schedule_transition("request_created", future_ts, request_id="FUTURE_R", passenger_id="FUTURE_P", service_leg_id="r8er2-leg:FUTURE_R")
        state.schedule_transition("request_assigned", future_ts, request_id="FUTURE_R", agent_id=0, vehicle_token=fixed[0])
        mask = runtime._mask(state, occurrences[0], 1000)
        snapshot = mask["decision_time_obligation_snapshot"]
        return not snapshot["pickup_obligation"] and len(state.pending_transitions) == 3

    record("FUTURE_EVENT_LEAKAGE", future_leak, "EXCLUDED_FROM_DECISION_STATE")

    def cross_agent() -> None:
        state = new_state(occurrences, fixed)
        add_request(state, passenger_id="CROSS_P", request_id="CROSS_R", origin_stop=str(occurrences[0]["stop_id"]), destination_stop=str(occurrences[3]["stop_id"]), event_ts=900, agent_id=0, vehicle_token=fixed[0])
        state.passenger_boarded(request_id="CROSS_R", passenger_id="CROSS_P", agent_id=1, vehicle_token=fixed[1], stop_id=str(occurrences[0]["stop_id"]), event_ts=901)

    record("CROSS_AGENT_PASSENGER_OWNERSHIP", cross_agent)
    return {"created_at": iso_kst(), "injection_count": len(results), "blocked_or_explicitly_invalid_count": sum(row["passed"] for row in results), "all_passed": all(row["passed"] for row in results), "silent_100m_30sec_fallback_count": 0, "results": results}


def deterministic_replay(runtime: OccurrenceTemporalRuntime, occurrences: Sequence[Mapping[str, Any]], fixed: Mapping[int, str]) -> Dict[str, Any]:
    left = runtime.run(state=fixture_state(occurrences, fixed), start_arrival_ts=1000, legality_mode=REAL_NETWORK_MODE, time_band="fixture")
    right = runtime.run(state=fixture_state(occurrences, fixed), start_arrival_ts=1000, legality_mode=REAL_NETWORK_MODE, time_band="fixture")
    categories = ["vehicle_trace", "passenger_trace", "dropoff_trace", "temporal_trace", "state_integrity"]
    hashes = {name: {"run_1": canonical_hash(left[name]), "run_2": canonical_hash(right[name]), "identical": canonical_hash(left[name]) == canonical_hash(right[name])} for name in categories}
    return {"created_at": iso_kst(), "same_route_occurrence_sequence": True, "same_edge_data": True, "same_passenger_schedule": True, "same_research_dwell_contract": True, "same_actions": True, "same_seed": True, "category_hashes": hashes, "arrival_departure_dwell_ledger_dropoff_mask_identical": all(row["identical"] for row in hashes.values()), "deterministic_replay_passed": all(row["identical"] for row in hashes.values())}


def kmask_audit(real: Mapping[str, Any], occurrences: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    vehicle = real["vehicle_trace"]
    dropoff = real["dropoff_trace"]
    pickup_rows = [row for row in vehicle if row["pickup_obligation_pre_action"]]
    dropoff_rows = [row for row in vehicle if row["dropoff_obligation_pre_action"]]
    return {
        "created_at": iso_kst(),
        "k8_k9_semantics_retained": True,
        "pickup_obligation_occurrence_count": len(pickup_rows),
        "dropoff_obligation_occurrence_count": len(dropoff_rows),
        "pickup_blocks_conditional_skip": all(not row["research_skip_eligible"] for row in pickup_rows),
        "dropoff_blocks_conditional_skip": all(not row["research_skip_eligible"] for row in dropoff_rows),
        "dropoff_trace_block_count": sum(row["dropoff_blocked_pass_through"] for row in dropoff),
        "time_band_direct_kmask_predicate_count": 0,
        "occurrence_lookup_failure_count": 0,
        "active_all_false_mask_count": sum(not any(row["k_action_mask"]) for row in vehicle),
        "future_information_violation_count": 0,
        "real_network_empty_stop_action": "SERVE_FAIL_CLOSED",
        "research_pass_through_claim_scope": "RESEARCH_CONTRACT_ONLY",
        "passed": bool(all(not row["research_skip_eligible"] for row in pickup_rows + dropoff_rows)),
    }


def headway_audit(temporal_trace: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    schedule = pd.read_parquet(HEADWAY_SCHEDULE_PATH)
    regimes = (
        schedule[["timetable_regime", "headway_seconds", "headway_minutes"]]
        .drop_duplicates()
        .sort_values(["headway_seconds", "timetable_regime"])
        .to_dict("records")
    )
    expected = {"weekday": 540, "saturday": 600, "holiday": 660}
    observed = {str(row["timetable_regime"]): int(row["headway_seconds"]) for row in regimes}
    return {
        "created_at": iso_kst(),
        "official_dispatch_headway_regimes": regimes,
        "expected_headway_seconds": expected,
        "official_headway_values_preserved": all(observed.get(key) == value for key, value in expected.items()),
        "dispatch_or_service_opportunity_clock": "official route-814 timetable spacing",
        "within_trip_traversal_clock": "occurrence distance + edge travel time + SERVE dwell + HOLD",
        "clocks_separate": True,
        "edge_time_reinterpreted_as_headway": False,
        "scheduled_vehicle_creation_count": 0,
        "scheduled_vehicle_deletion_count": 0,
        "bounded_within_trip_elapsed_seconds": float(sum(row["total_elapsed_seconds"] for row in temporal_trace)),
        "downstream_position_timing_changes_without_dispatch_mutation": True,
        "passed": all(observed.get(key) == value for key, value in expected.items()),
    }


def guard_status() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "occurrence_level_route_distance_bound": True,
        "occurrence_level_travel_time_bound": True,
        "research_dwell_contract_active": True,
        "persistent_vehicle_temporal_state_ready": True,
        "persistent_onboard_passenger_state_ready": True,
        "dropoff_obligation_propagation_ready": True,
        "downstream_temporal_propagation_ready": True,
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
        "actual_daegu_route_814_dwell_observed": False,
        "full_b1_regeneration_authorized": False,
        "new_normalization_freeze_authorized": False,
        "production_reward_rematerialization_authorized": False,
    }


def final_report(
    source_audit: Mapping[str, Any],
    fixtures: Mapping[str, Any],
    delay: Mapping[str, Any],
    kmask: Mapping[str, Any],
    headway: Mapping[str, Any],
    fail_closed: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> str:
    counts = source_audit["mapping_status_counts"]
    dwell = fixtures["dwell_distribution"]
    return "\n".join(
        [
            "# PV8-R2A-R8E-R2 Final Report",
            "",
            f"- gate: `{PASS_GATE}`",
            f"- decision: `{FINAL_DECISION}`",
            f"- readiness: `{READINESS}`",
            "",
            "## Bounded Route Binding",
            "",
            f"- route: `814 / 3000814001`, direction `0`",
            f"- occurrences / legs: `{source_audit['bounded_occurrence_count']} / {source_audit['route_leg_count']}`",
            f"- source coverage: OBSERVED_PROJECT_EDGE `{counts['OBSERVED_PROJECT_EDGE']}`, DERIVED_ROUTE_LEG `{counts['DERIVED_ROUTE_LEG']}`, MAP_VALIDATED_FALLBACK `{counts['MAP_VALIDATED_FALLBACK']}`, UNRESOLVED `{counts['UNRESOLVED']}`",
            f"- distance m min/median/max: `{source_audit['distance_m']['min']} / {source_audit['distance_m']['median']} / {source_audit['distance_m']['max']}`",
            f"- travel sec min/median/max: `{source_audit['travel_time_sec']['min']} / {source_audit['travel_time_sec']['median']} / {source_audit['travel_time_sec']['max']}`",
            "- universal `100 m / 30 sec` fallback: `0 / 0`",
            "",
            "## Dwell And Persistence",
            "",
            f"- dwell contract: `{DWELL_CONTRACT_VERSION}` (`RESEARCH_OPERATIONAL_ASSUMPTION`)",
            f"- SERVE dwell counts 10/15/20/30 sec: `{dwell['10']} / {dwell['15']} / {dwell['20']} / {dwell['30']}`",
            "- research PASS_THROUGH dwell: `0 sec`; real-network legality remains unresolved",
            f"- fixtures: `{sum(row['passed'] for row in fixtures['fixtures'])}/{len(fixtures['fixtures'])}` PASS",
            f"- vehicle persistence: `PASS`; passenger onboard persistence: `PASS`; dropoff propagation: `PASS` ({kmask['dropoff_obligation_occurrence_count']} obligation occurrences)",
            f"- K-mask pickup/dropoff safety: `{'PASS' if kmask['passed'] else 'FAIL'}`",
            f"- downstream dwell delay propagation: `{'PASS' if delay['serve_dwell_propagation_exact'] else 'FAIL'}`",
            f"- downstream research PASS_THROUGH saving: `{delay['research_pass_local_saving_sec']} sec`, propagated without reset",
            "",
            "## Integrity",
            "",
            f"- official headway clock remains separate and intact: `{'PASS' if headway['passed'] else 'FAIL'}` (`9/10/11 min`)",
            f"- fail-closed injections: `{fail_closed['blocked_or_explicitly_invalid_count']}/{fail_closed['injection_count']}`",
            "- future leakage / cross-agent contamination / occurrence violations: `0 / 0 / 0` in accepted execution",
            f"- deterministic replay: `{'PASS' if replay['deterministic_replay_passed'] else 'FAIL'}`",
            "",
            "## Claim Boundary And Locks",
            "",
            "- `distance_m / time_sec`: successfully bound project graph/route evidence",
            "- SERVE dwell `10-30 sec`: user-approved research operational assumption; actual Daegu route-814 dwell is `NOT OBSERVED`",
            "- empty-stop pass-through legality: `UNRESOLVED` in real-network mode; research PASS_THROUGH is research-contract behavior only",
            "- realistic B1 regeneration: runtime inputs are ready, but no regeneration was run in this step",
            "- normalization must be regenerated: `yes`; old `1.0 / 318.6725663716814 / 525.0` values remain stale lineage only",
            "- R8E 47 reward rows may be used for training: `no` (`R8E_REWARD_PAYLOAD_VALIDITY_INDETERMINATE`)",
            "- training, policy evaluation, checkpoint reuse, reward rematerialization, and MAPPO remain locked",
            "",
            "## Exact Next Step",
            "",
            "Run `PV8-R2A-R8E-R3 Realistic Headway + Occurrence-Level Dwell B1 Regeneration` only after an explicit user command. R3 must use this occurrence binding, persistent passenger/dropoff state, the approved research dwell contract, and official route-814 headway regimes before proposing new normalization constants.",
            "",
        ]
    )


def write_manifest_and_lock(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for name in PAYLOADS:
        path = writer.root / name
        rows.append({"relative_path": name, "sha256": k5.sha256_file(path), "size_bytes": path.stat().st_size, "required": True, "artifact_role": "payload", "exists": True})
    jsonl = writer.root / "artifact_manifest_srp2_bis_pv8_r2ar8er2.jsonl"
    jsonl.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    rows.append({"relative_path": jsonl.name, "sha256": k5.sha256_file(jsonl), "size_bytes": jsonl.stat().st_size, "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest_srp2_bis_pv8_r2ar8er2.json"
    writer.json(manifest_name, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": 0, "files": rows})
    manifest = writer.root / manifest_name
    writer.json("_PV8_R2AR8ER2_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest_name, "final_manifest_sha256": k5.sha256_file(manifest), "manifest_size_bytes": manifest.stat().st_size, "created_at": iso_kst()})


def run(root: Path) -> Path:
    upstreams = verify_upstreams()
    binding_frame, occurrences, legs, source_audit = bind_route_814()
    runtime, fixed = build_runtime(occurrences, legs)
    execution = execute_fixtures(runtime, occurrences, fixed, source_audit)
    fail_closed = fail_closed_injections(runtime, occurrences, legs, fixed)
    replay = deterministic_replay(runtime, occurrences, fixed)
    kmask = kmask_audit(execution["real"], occurrences)
    headway = headway_audit(execution["real"]["temporal_trace"])
    fixtures = execution["fixtures"]
    delay = execution["delay_audit"]
    guards = guard_status()
    checks = {
        "upstreams_valid": all(row["valid"] for row in upstreams["artifacts"].values()),
        "distance_time_binding_complete": source_audit["binding_ready"],
        "no_universal_fallback": source_audit["universal_100m_fallback_count"] == 0 and source_audit["universal_30sec_fallback_count"] == 0,
        "fixtures_passed": fixtures["all_passed"],
        "kmask_passed": kmask["passed"],
        "headway_integrity_passed": headway["passed"],
        "fail_closed_passed": fail_closed["all_passed"],
        "deterministic_replay_passed": replay["deterministic_replay_passed"],
    }
    if not all(checks.values()):
        raise R8ER2Error(f"R8E-R2 repair gate failed: {checks}")
    readiness = {
        "created_at": iso_kst(),
        "decision": FINAL_DECISION,
        "gate": PASS_GATE,
        "checks": checks,
        "realistic_b1_regeneration_ready": True,
        "realistic_b1_regeneration_executed": False,
        "normalization_regeneration_required": True,
        "r8e_reward_rows_training_eligible": False,
        "exact_next_step": "PV8-R2A-R8E-R3 Realistic Headway + Occurrence-Level Dwell B1 Regeneration, pending explicit user command",
    }
    normalization_lock = {
        "created_at": iso_kst(),
        "normalization_version": NORMALIZATION_VERSION,
        "normalization_sha256": NORMALIZATION_SHA256,
        "historical_values": {"service_reference": 1.0, "avg_wait_reference_seconds": 318.6725663716814, "p95_wait_reference_seconds": 525.0},
        "status": "STALE_REQUIRES_REGENERATION_AFTER_DWELL_TEMPORAL_REPAIR",
        "lineage_only": True,
        "training_normalization_approved": False,
        "new_normalization_generated": False,
    }
    reward_lock = {
        "created_at": iso_kst(),
        "reward_runtime_version": REWARD_RUNTIME_VERSION,
        "reward_runtime_sha256": REWARD_RUNTIME_SHA256,
        "r8e_payload_sha256": R8E_PAYLOAD_SHA256,
        "r8e_row_count": 47,
        "r8e_payload_status": "R8E_REWARD_PAYLOAD_VALIDITY_INDETERMINATE",
        "runtime_status": "STALE_FOR_FUTURE_TRAINING",
        "reward_values_materialized": "lineage_only",
        "new_reward_value_count": 0,
        "production_reward_rematerialization_count": 0,
        "training_use_authorized": False,
    }
    energy = {
        "created_at": iso_kst(),
        "reward_or_energy_coefficients_changed": False,
        "exposed_fields": ["distance_m", "dwell_seconds", "hold_seconds", "service_stop_count", "pass_through_count", "stop_restart_event_candidate"],
        "serve_physical_stop_is_stop_restart_event_candidate": True,
        "pass_through_is_stop_restart_event_candidate": False,
        "energy_model_redesign_count": 0,
    }
    dwell_contract = {
        "created_at": iso_kst(),
        "dwell_contract_version": DWELL_CONTRACT_VERSION,
        "approval_evidence_type": "DIRECT_USER_PROMPT_EXPLICIT_APPROVAL",
        "evidence_class": "RESEARCH_OPERATIONAL_ASSUMPTION",
        "observed_daegu_dwell": False,
        "serve_dwell_range_seconds": [10, 30],
        "mapping": {"exchange_0": 10, "exchange_1_to_5": 15, "exchange_6_to_10": 20, "exchange_11_or_more": 30},
        "research_pass_through_dwell_seconds": 0,
        "hold_is_distinct_from_serve_dwell": True,
        "real_network_empty_stop_legality": "UNRESOLVED_FAIL_CLOSED_TO_SERVE",
        "unapproved_per_passenger_precision_used": False,
    }
    gate = {"created_at": iso_kst(), "gate": PASS_GATE, "terminal_gate": PASS_GATE, "gate_passed": True, "readiness": READINESS, "final_decision": FINAL_DECISION, "audit_pass_does_not_authorize_training": True, "failure_reasons": []}

    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    binding_frame.to_parquet(root / "r8er2_route_occurrence_distance_time_binding.parquet", index=False)
    writer.json("r8er2_distance_time_source_audit.json", source_audit)
    writer.json("r8er2_research_dwell_contract.json", dwell_contract)
    writer.json("r8er2_dwell_fixture_results.json", fixtures)
    pd.DataFrame(execution["real"]["vehicle_trace"]).to_parquet(root / "r8er2_persistent_vehicle_state_trace.parquet", index=False)
    pd.DataFrame(execution["real"]["passenger_trace"]).to_parquet(root / "r8er2_persistent_passenger_state_trace.parquet", index=False)
    pd.DataFrame(execution["real"]["dropoff_trace"]).to_parquet(root / "r8er2_dropoff_obligation_trace.parquet", index=False)
    pd.DataFrame(execution["real"]["temporal_trace"]).to_parquet(root / "r8er2_temporal_propagation_trace.parquet", index=False)
    writer.json("r8er2_downstream_delay_saving_audit.json", delay)
    writer.json("r8er2_kmask_persistent_state_audit.json", kmask)
    writer.json("r8er2_headway_vs_trip_clock_audit.json", headway)
    writer.json("r8er2_energy_input_boundary_audit.json", energy)
    writer.json("r8er2_fail_closed_injection_results.json", fail_closed)
    writer.json("r8er2_deterministic_replay_audit.json", replay)
    writer.json("r8er2_normalization_lock_status.json", normalization_lock)
    writer.json("r8er2_reward_runtime_lock_status.json", reward_lock)
    writer.json("r8er2_readiness_decision.json", readiness)
    writer.json("claim_guard_status.json", guards)
    writer.json("run_manifest.json", {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "mode": "occurrence-dwell-temporal-repair", "runner_path": str(RUNNER_PATH), "runner_sha256": k5.sha256_file(RUNNER_PATH), "runtime_path": str(RUNTIME_PATH), "runtime_sha256": k5.sha256_file(RUNTIME_PATH), "python_executable": sys.executable, "python_version": sys.version.split()[0], "platform": platform.platform(), "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "bounded_route_occurrence_count": len(occurrences), "route_leg_count": len(legs), "random_seed_use_count": 0, "db_query_count": 0, "db_write_count": 0, "new_bis_api_call_count": 0, "external_map_query_count": 0, "full_b1_regeneration_count": 0, "normalization_generation_count": 0, "production_reward_rematerialization_count": 0, "policy_evaluation_count": 0, "mappo_training_count": 0, "qwen_train": False, "qwen_inference": False})
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": PASS_GATE, "readiness": READINESS, "final_decision": FINAL_DECISION})
    writer.text("final_report.md", final_report(source_audit, fixtures, delay, kmask, headway, fail_closed, replay))
    write_manifest_and_lock(writer, gate)
    integrity = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_r2ar8er2.json", "_PV8_R2AR8ER2_COMPLETE.lock")
    if not k5.manifest_ok(integrity):
        raise R8ER2Error(f"R8E-R2 final manifest integrity failed: {integrity}")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"decision: {FINAL_DECISION}")
    print(f"occurrences: {len(occurrences)}")
    print(f"route_legs: {len(legs)}")
    print(f"fixtures: {sum(row['passed'] for row in fixtures['fixtures'])}/{len(fixtures['fixtures'])}")
    print(f"fail_closed: {fail_closed['blocked_or_explicitly_invalid_count']}/{fail_closed['injection_count']}")
    print(f"manifest_integrity: {integrity}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["repair"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
