#!/usr/bin/env python3
"""PV8-K8 explicitly approved research-rule runtime K-mask integration gate."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

import run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k5_static_rulebook_readiness as k5
from simulator.k_action_mask_runtime import FixedVehicleOccurrenceMaskRuntime, RuntimeVersionBinding
from simulator.k_safety_state import K_SAFETY_STATE_SCHEMA_VERSION, ServiceObligationStateMachine


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration.py"
RUNTIME_SOURCE = TRAINING_ROOT / "simulator" / "k_action_mask_runtime.py"
RUNTIME_TEST_SOURCE = TRAINING_ROOT / "simulator" / "test_pv8_k8_k_action_mask_runtime.py"
K4_STATE_SOURCE = TRAINING_ROOT / "simulator" / "k_safety_state.py"
K4_ENGINE_SOURCE = TRAINING_ROOT / "simulator" / "suseong_service_transition_engine.py"

K4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k4_dynamic_service_obligation_state_20260808_120008"
K6_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026"
K7_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k7_research_rule_contract_mask_dryrun_20260808_130432"
C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"

UPSTREAMS = {
    "PV8-K4": (K4_ROOT, "artifact_manifest_srp2_bis_pv8_k4.json", "_PV8_K4_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K4_DYNAMIC_SERVICE_OBLIGATION_STATE_IMPLEMENTED"),
    "PV8-K6": (K6_ROOT, "artifact_manifest_srp2_bis_pv8_k6.json", "_PV8_K6_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K6_STATIC_RULE_AUTHORITY_AND_OCCURRENCE_AUDIT_COMPLETE"),
    "PV8-K7": (K7_ROOT, "artifact_manifest_srp2_bis_pv8_k7.json", "_PV8_K7_COMPLETE.lock", "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K7_RESEARCH_RULE_CONTRACT_AND_K_MASK_DRYRUN_COMPLETE"),
}

RULEBOOK_PATH = K7_ROOT / "k7_research_rulebook_candidate.parquet"
OCCURRENCE_PATH = K6_ROOT / "k6_route_stop_occurrence_master.parquet"
K7_RULE_CONTRACT_PATH = K7_ROOT / "k7_research_rule_contract.json"
C2_MAPPING_PATH = C2_ROOT / "prospective_8vehicle_mapping_manifest.parquet"
C2_CYCLE_PATH = C2_ROOT / "prospective_agent_cycle_state.parquet"
C2_TRANSITION_PATH = C2_ROOT / "vehicle_identity_transition_events.parquet"

EXPECTED_RULEBOOK_SHA256 = "f0b655ab4871a6faae4d7a519438435ebe80cb95e537c3e1e16134d31fdf3ff2"
EXPECTED_OCCURRENCE_SHA256 = "45e8ae3ff61a6a8e89de36281b288ea4d6c077b857f708d8cd4b7850954928cd"
RULE_VERSION = "PV8_K7_RESEARCH_STATIC_RULE_CANDIDATE_V1"
EXPERIMENT_VERSION = "PV8_K8_APPROVED_RESEARCH_KMASK_RUNTIME_V1"
MASK_PREDICATE_VERSION = "PV8_K4_FAIL_CLOSED_V2"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k8_approved_research_kmask_integration"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_COMPLETE"
FAIL_GATE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K8_APPROVED_RESEARCH_K_ACTION_MASK_INTEGRATION_FAILED"
PASS_READINESS = "SRP2_BIS_PV8_K8_COMPLETE_APPROVED_RESEARCH_KMASK_RUNTIME_READY_K9_LOCKED"

DECISION_READY = "K_ACTION_MASK_RUNTIME_READY_RESEARCH_RULES"
DECISION_PENDING = "PENDING_EXPLICIT_APPROVAL"
DECISION_FAILED = "K_ACTION_MASK_RUNTIME_INTEGRATION_FAILED"

PAYLOADS = [
    "k8_explicit_rule_approval_record.json",
    "k8_frozen_rule_contract.json",
    "k8_runtime_k_mask_integration.parquet",
    "k8_runtime_mask_validation.json",
    "k8_snapshot_version_binding.json",
    "k8_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class K8Error(RuntimeError):
    pass


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def verify_upstreams() -> Dict[str, Any]:
    verified: Dict[str, Any] = {}
    for label, (root, manifest_name, lock_name, expected_gate) in UPSTREAMS.items():
        gate = k5.read_json(root / "gate_decision.json")
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        checks = k5.verify_manifest(root, manifest_name, lock_name)
        if observed_gate != expected_gate or not k5.manifest_ok(checks):
            raise K8Error(f"{label} upstream integrity failure: gate={observed_gate}, checks={checks}")
        verified[label] = {
            "artifact_root": str(root),
            "gate": observed_gate,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return verified


def approval_record() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "approval_evidence_type": "DIRECT_USER_MESSAGE_IN_CURRENT_CODEX_TASK",
        "approval_evidence_text": "승인 계속진행",
        "approval_context": "direct response to the K8 stop message listing all six assumptions, research-only claim boundary, frozen hashes, and runtime-mask-only scope",
        "approval_actor": "USER",
        "operator_approval_claimed": False,
        "user_research_contract_approval": True,
        "approval_valid": True,
        "research_rule_approved": True,
        "approved_assumptions": {
            "mandatory_stop": False,
            "protected_stop": False,
            "charging_or_driver_relief_stop": False,
            "valid_post_skip_target_implies_planned_itinerary_allows_skip": True,
            "interior_occurrence_implies_terminal_or_turnaround_stop": False,
            "k6_derived_safe_endpoint_and_no_target_blockers_override_research_assumptions": True,
        },
        "claim_boundary": "These are CONTRACT_FIXED_RESEARCH_RULE values, not observed Daegu operational rules.",
        "claim_boundary_approved": True,
        "static_rulebook_version": RULE_VERSION,
        "static_rulebook_sha256": EXPECTED_RULEBOOK_SHA256,
        "occurrence_master_sha256": EXPECTED_OCCURRENCE_SHA256,
        "approved_scope": "K8_RUNTIME_K_ACTION_MASK_INTEGRATION_ONLY",
        "real_network_rule_claim_allowed": False,
        "policy_execution_authorized": False,
        "policy_evaluation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "training_use_authorized": False,
        "mappo_retraining_authorized": False,
    }


def current_k4_hashes() -> Dict[str, Any]:
    contract = k5.read_json(K4_ROOT / "k4_state_machine_contract.json")
    expected = {str(row["path"]): str(row["sha256"]) for row in contract["source_files"]}
    rows = []
    for path in (K4_STATE_SOURCE, K4_ENGINE_SOURCE):
        current = k5.sha256_file(path)
        rows.append({"path": str(path), "expected_sha256": expected.get(str(path)), "current_sha256": current, "matches_k4": current == expected.get(str(path))})
    return {"records": rows, "source_drift_count": sum(not row["matches_k4"] for row in rows)}


def fixed_vehicle_bindings() -> Tuple[Dict[int, str], pd.DataFrame]:
    mapping = pd.read_parquet(C2_MAPPING_PATH).sort_values("agent_id").reset_index(drop=True)
    if len(mapping) != 8 or mapping["agent_id"].nunique() != 8 or mapping["physical_vehicle_token"].nunique() != 8:
        raise K8Error("C2 fixed physical-vehicle mapping must contain exactly eight unique slots and tokens")
    return {int(row.agent_id): str(row.physical_vehicle_token) for row in mapping.itertuples()}, mapping


def machine_for_rule(agent_id: int, token: str, rule: Mapping[str, Any]) -> ServiceObligationStateMachine:
    state = ServiceObligationStateMachine()
    state.register_vehicle(agent_id, token)
    state.register_stop("__K8_PREVIOUS__")
    state.register_stop(str(rule["stop_id"]))
    post = rule.get("post_skip_target_stop_id")
    if post is not None and str(post).lower() != "nan":
        state.register_stop(str(post))
    return state


RUNTIME_COLUMNS = [
    "scenario_id", "scenario_scope", "cycle_index", "cycle_timestamp", "identity_transition_type",
    "agent_id", "physical_vehicle_token", "active_bus_mask", "route_stop_occurrence_id", "route_id",
    "direction_id", "stop_sequence", "stop_id", "occurrence_lookup_integrity", "rule_version",
    "experiment_version", "static_rulebook_sha256", "occurrence_master_sha256", "expected_skip_valid",
    "observed_skip_valid", "passed", "expected_reason", "observed_reasons_json", "action_mask_json",
    "hold_valid", "serve_move_valid", "dynamic_state_complete", "static_guard_complete", "pickup_obligation",
    "dropoff_obligation", "onboard_destination_obligation", "service_obligation", "slot_mutation",
    "inactive_action_leakage", "identity_failure", "future_information_violation", "rule_hash_drift",
    "reappearance_case", "direction_transition_case", "policy_execution_count",
]


def runtime_row(
    *,
    runtime: FixedVehicleOccurrenceMaskRuntime,
    scenario_id: str,
    scenario_scope: str,
    agent_id: int,
    token: str,
    active: bool,
    rule: Optional[Mapping[str, Any]],
    state: Optional[ServiceObligationStateMachine],
    decision_ts: int,
    expected_skip: bool,
    expected_reason: Optional[str],
    cycle_index: Optional[int] = None,
    cycle_timestamp: Optional[str] = None,
    identity_transition_type: Optional[str] = None,
    slot_mutation: bool = False,
    reappearance_case: bool = False,
    direction_transition_case: bool = False,
) -> Dict[str, Any]:
    result = runtime.evaluate(
        agent_id=agent_id,
        vehicle_token=token,
        active_bus_mask=active,
        route_id=str(rule["route_id"]) if active and rule is not None else None,
        direction_id=str(rule["direction_id"]) if active and rule is not None else None,
        stop_sequence=int(rule["stop_sequence"]) if active and rule is not None else None,
        stop_id=str(rule["stop_id"]) if active and rule is not None else None,
        route_stop_occurrence_id=str(rule["route_stop_occurrence_id"]) if active and rule is not None else None,
        obligation_state_machine=state,
        decision_ts=decision_ts,
    )
    reasons = list(result["skip_invalid_reason_codes"])
    reason_ok = expected_reason is None or expected_reason in reasons
    inactive_leakage = bool(not active and any(bool(value) for value in result["action_mask"]))
    future_violation = bool(scenario_id == "NO_FUTURE_LEAK_PRE_BOUNDARY" and result.get("service_obligation_exists"))
    rule_hash_drift = bool(result["version_binding"]["static_rulebook_sha256"] != EXPECTED_RULEBOOK_SHA256 or result["version_binding"]["occurrence_master_sha256"] != EXPECTED_OCCURRENCE_SHA256)
    passed = bool(
        result["skip_valid"] is expected_skip
        and reason_ok
        and not slot_mutation
        and not inactive_leakage
        and not result["identity_failure"]
        and not future_violation
        and not rule_hash_drift
        and result["policy_execution_count"] == 0
    )
    snapshot = result.get("decision_time_obligation_snapshot") or {}
    return {
        "scenario_id": scenario_id,
        "scenario_scope": scenario_scope,
        "cycle_index": cycle_index,
        "cycle_timestamp": cycle_timestamp,
        "identity_transition_type": identity_transition_type,
        "agent_id": agent_id,
        "physical_vehicle_token": token,
        "active_bus_mask": active,
        "route_stop_occurrence_id": result.get("route_stop_occurrence_id"),
        "route_id": str(rule["route_id"]) if rule is not None else None,
        "direction_id": str(rule["direction_id"]) if rule is not None else None,
        "stop_sequence": int(rule["stop_sequence"]) if rule is not None else None,
        "stop_id": str(rule["stop_id"]) if rule is not None else None,
        "occurrence_lookup_integrity": bool(result["occurrence_lookup_integrity"]),
        "rule_version": result.get("rule_version", RULE_VERSION),
        "experiment_version": EXPERIMENT_VERSION,
        "static_rulebook_sha256": result["version_binding"]["static_rulebook_sha256"],
        "occurrence_master_sha256": result["version_binding"]["occurrence_master_sha256"],
        "expected_skip_valid": expected_skip,
        "observed_skip_valid": bool(result["skip_valid"]),
        "passed": passed,
        "expected_reason": expected_reason,
        "observed_reasons_json": json.dumps(reasons),
        "action_mask_json": json.dumps(result["action_mask"]),
        "hold_valid": bool(result["hold_valid"]),
        "serve_move_valid": bool(result["serve_move_valid"]),
        "dynamic_state_complete": result.get("dynamic_state_complete"),
        "static_guard_complete": result.get("static_guard_state_complete"),
        "pickup_obligation": bool(snapshot.get("pickup_obligation", False)),
        "dropoff_obligation": bool(snapshot.get("dropoff_obligation", False)),
        "onboard_destination_obligation": bool(snapshot.get("onboard_destination_obligation", False)),
        "service_obligation": bool(snapshot.get("service_obligation", False)),
        "slot_mutation": slot_mutation,
        "inactive_action_leakage": inactive_leakage,
        "identity_failure": bool(result["identity_failure"]),
        "future_information_violation": future_violation,
        "rule_hash_drift": rule_hash_drift,
        "reappearance_case": reappearance_case,
        "direction_transition_case": direction_transition_case,
        "policy_execution_count": int(result["policy_execution_count"]),
    }


def build_runtime_rows(runtime: FixedVehicleOccurrenceMaskRuntime, rule_rows: Sequence[Mapping[str, Any]], mapping: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_exact = {
        (str(row["route_id"]), str(row["direction_id"]), int(row["stop_sequence"]), str(row["stop_id"])): row
        for row in rule_rows
    }
    by_route = {}
    for row in rule_rows:
        by_route.setdefault((str(row["route_id"]), str(row["direction_id"])), []).append(row)
    for rows in by_route.values():
        rows.sort(key=lambda item: int(item["occurrence_index"]))

    rows: List[Dict[str, Any]] = []
    cycles = pd.read_parquet(C2_CYCLE_PATH).sort_values(["cycle_index", "agent_id"])
    transition_events = pd.read_parquet(C2_TRANSITION_PATH)
    transition_keys = {(int(row.agent_id), int(row.cycle_index)): row for row in transition_events.itertuples()}

    for cycle in cycles.to_dict("records"):
        agent_id = int(cycle["agent_id"])
        token = str(cycle["bound_vehicle_token"])
        active = bool(cycle["active_bus_mask"])
        rule = None
        if active:
            exact = (str(cycle["route_id"]), str(cycle["direction_id"]), int(cycle["seq"]), str(cycle["stop_id"]))
            rule = by_exact.get(exact)
            if rule is None:
                raise K8Error(f"active C2 cycle occurrence lookup missing: {exact}")
        state = machine_for_rule(agent_id, token, rule) if active and rule is not None else None
        expected = bool(rule["positive_static_skip_clearance"]) if rule is not None else False
        expected_reason = None
        if not active:
            expected_reason = "INACTIVE_AGENT"
        elif not expected:
            expected_reason = "TERMINAL_OR_TURNAROUND_STOP" if rule["terminal_or_turnaround_stop"] else "PLANNED_ITINERARY_BLOCK"
        transition = transition_keys.get((agent_id, int(cycle["cycle_index"])))
        reappearance = str(cycle.get("identity_transition_type")) == "REAPPEARED_SAME_IDENTITY_WITH_DIRECTION_CHANGE"
        direction_change = transition is not None
        transition_identity_ok = True if transition is None else bool(transition.agent_id_unchanged and transition.physical_vehicle_token_unchanged and int(transition.identity_mutation_count) == 0)
        rows.append(
            runtime_row(
                runtime=runtime,
                scenario_id="C2_PROSPECTIVE_CYCLE_RUNTIME",
                scenario_scope="C2_PROSPECTIVE_FORWARD_CYCLE",
                agent_id=agent_id,
                token=token,
                active=active,
                rule=rule,
                state=state,
                decision_ts=int(cycle["cycle_index"]),
                expected_skip=expected,
                expected_reason=expected_reason,
                cycle_index=int(cycle["cycle_index"]),
                cycle_timestamp=cycle.get("cycle_timestamp"),
                identity_transition_type=cycle.get("identity_transition_type"),
                slot_mutation=bool(cycle["slot_identity_mutation"]) or not transition_identity_ok,
                reappearance_case=reappearance,
                direction_transition_case=direction_change,
            )
        )

    for anchor in mapping.to_dict("records"):
        agent_id = int(anchor["agent_id"])
        token = str(anchor["physical_vehicle_token"])
        route_key = (str(anchor["anchor_route_id"]), str(anchor["anchor_direction_id"]))
        route_rules = by_route[route_key]
        clear_rule = next(row for row in route_rules if row["positive_static_skip_clearance"])
        terminal_rule = next(row for row in reversed(route_rules) if row["terminal_or_turnaround_stop"])
        no_target_rule = next(row for row in reversed(route_rules) if not row["post_skip_target_exists"])
        scenario_specs = [
            ("POSITIVE_SKIP_APPROVED_RESEARCH_RULE", clear_rule, True, None),
            ("TERMINAL_BLOCKER", terminal_rule, False, "TERMINAL_OR_TURNAROUND_STOP"),
            ("NO_TARGET_BLOCKER", no_target_rule, False, "NO_POST_SKIP_TARGET"),
        ]
        for scenario_id, rule, expected, reason in scenario_specs:
            rows.append(runtime_row(runtime=runtime, scenario_id=scenario_id, scenario_scope="K8_REQUIRED_SCENARIO", agent_id=agent_id, token=token, active=True, rule=rule, state=machine_for_rule(agent_id, token, rule), decision_ts=0, expected_skip=expected, expected_reason=reason))

        candidate_stop = str(clear_rule["stop_id"])
        post_stop = str(clear_rule["post_skip_target_stop_id"])
        pickup = machine_for_rule(agent_id, token, clear_rule)
        pickup.passenger_waiting(passenger_id=f"P8-PICKUP-{agent_id}", pickup_stop=candidate_stop, dropoff_stop=post_stop, event_ts=1)
        pickup.request_created(request_id=f"R8-PICKUP-{agent_id}", passenger_id=f"P8-PICKUP-{agent_id}", service_leg_id=f"L8-PICKUP-{agent_id}", event_ts=2)
        pickup.request_assigned(request_id=f"R8-PICKUP-{agent_id}", agent_id=agent_id, vehicle_token=token, event_ts=3)
        rows.append(runtime_row(runtime=runtime, scenario_id="PICKUP_BLOCKER", scenario_scope="K8_REQUIRED_SCENARIO", agent_id=agent_id, token=token, active=True, rule=clear_rule, state=pickup, decision_ts=3, expected_skip=False, expected_reason="ASSIGNED_PICKUP_REQUEST"))

        onboard = machine_for_rule(agent_id, token, clear_rule)
        onboard.passenger_waiting(passenger_id=f"P8-ONBOARD-{agent_id}", pickup_stop="__K8_PREVIOUS__", dropoff_stop=candidate_stop, event_ts=1)
        onboard.request_created(request_id=f"R8-ONBOARD-{agent_id}", passenger_id=f"P8-ONBOARD-{agent_id}", service_leg_id=f"L8-ONBOARD-{agent_id}", event_ts=2)
        onboard.request_assigned(request_id=f"R8-ONBOARD-{agent_id}", agent_id=agent_id, vehicle_token=token, event_ts=3)
        onboard.passenger_boarded(request_id=f"R8-ONBOARD-{agent_id}", passenger_id=f"P8-ONBOARD-{agent_id}", agent_id=agent_id, vehicle_token=token, stop_id="__K8_PREVIOUS__", event_ts=4)
        rows.append(runtime_row(runtime=runtime, scenario_id="DROPOFF_ONBOARD_BLOCKER", scenario_scope="K8_REQUIRED_SCENARIO", agent_id=agent_id, token=token, active=True, rule=clear_rule, state=onboard, decision_ts=4, expected_skip=False, expected_reason="ONBOARD_DROPOFF_DEMAND"))

        incomplete = machine_for_rule(agent_id, token, clear_rule)
        incomplete.mark_state_incomplete("K8_RUNTIME_DYNAMIC_STATE_INCOMPLETE")
        rows.append(runtime_row(runtime=runtime, scenario_id="INCOMPLETE_STATE_BLOCKER", scenario_scope="K8_REQUIRED_SCENARIO", agent_id=agent_id, token=token, active=True, rule=clear_rule, state=incomplete, decision_ts=0, expected_skip=False, expected_reason="DYNAMIC_STATE_INCOMPLETE"))

        future = machine_for_rule(agent_id, token, clear_rule)
        future.schedule_transition("passenger_waiting", 11, passenger_id=f"P8-FUTURE-{agent_id}", pickup_stop=candidate_stop, dropoff_stop=post_stop)
        future.schedule_transition("request_created", 12, request_id=f"R8-FUTURE-{agent_id}", passenger_id=f"P8-FUTURE-{agent_id}", service_leg_id=f"L8-FUTURE-{agent_id}")
        future.schedule_transition("request_assigned", 13, request_id=f"R8-FUTURE-{agent_id}", agent_id=agent_id, vehicle_token=token)
        rows.append(runtime_row(runtime=runtime, scenario_id="NO_FUTURE_LEAK_PRE_BOUNDARY", scenario_scope="K8_REQUIRED_SCENARIO", agent_id=agent_id, token=token, active=True, rule=clear_rule, state=future, decision_ts=10, expected_skip=True, expected_reason=None))
        rows.append(runtime_row(runtime=runtime, scenario_id="NO_FUTURE_LEAK_POST_BOUNDARY", scenario_scope="K8_REQUIRED_SCENARIO", agent_id=agent_id, token=token, active=True, rule=clear_rule, state=future, decision_ts=13, expected_skip=False, expected_reason="ASSIGNED_PICKUP_REQUEST"))

    audit = {
        "c2_cycle_row_count": len(cycles),
        "c2_active_cycle_count": int(cycles["active_bus_mask"].astype(bool).sum()),
        "c2_inactive_cycle_count": int((~cycles["active_bus_mask"].astype(bool)).sum()),
        "c2_empirical_reappearance_case_count": sum(row["reappearance_case"] for row in rows if row["scenario_id"] == "C2_PROSPECTIVE_CYCLE_RUNTIME"),
        "c2_empirical_direction_transition_case_count": sum(row["direction_transition_case"] for row in rows if row["scenario_id"] == "C2_PROSPECTIVE_CYCLE_RUNTIME"),
    }
    return rows, audit


def run_pytest() -> Dict[str, Any]:
    tests = [
        "05_training/simulator/test_pv8_k8_k_action_mask_runtime.py",
        "05_training/simulator/test_pv8_k4_service_obligation_state.py",
        "05_training/simulator/test_dl6c_conditional_skip_safety.py",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(TRAINING_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (completed.stdout + "\n" + completed.stderr).strip()
    passed_match = re.search(r"(\d+) passed", output)
    return {
        "command": [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
        "exit_code": completed.returncode,
        "passed_test_count": int(passed_match.group(1)) if completed.returncode == 0 and passed_match else 0,
        "summary": output[-1000:],
    }


def write_manifest_and_lock(writer: k5.Writer, gate: Mapping[str, Any]) -> None:
    rows = []
    for relative_path in PAYLOADS:
        path = writer.root / relative_path
        rows.append({"relative_path": relative_path, "size_bytes": path.stat().st_size if path.exists() else None, "sha256": k5.sha256_file(path) if path.exists() else None, "required": True, "artifact_role": Path(relative_path).stem, "exists": path.exists()})
    jsonl_name = "artifact_manifest_srp2_bis_pv8_k8.jsonl"
    writer.text(jsonl_name, "".join(json.dumps(k5.json_clean(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    jsonl_path = writer.root / jsonl_name
    rows.append({"relative_path": jsonl_name, "size_bytes": jsonl_path.stat().st_size, "sha256": k5.sha256_file(jsonl_path), "required": True, "artifact_role": "manifest_jsonl", "exists": True})
    manifest_name = "artifact_manifest_srp2_bis_pv8_k8.json"
    writer.json(manifest_name, {"created_at": iso_kst(), "artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "payload_count": len(rows), "missing_payload_count": sum(not row["exists"] for row in rows), "files": rows})
    manifest_path = writer.root / manifest_name
    writer.json("_PV8_K8_COMPLETE.lock", {"artifact_family": ARTIFACT_PREFIX, "terminal_gate": gate["gate"], "readiness": gate["readiness"], "final_manifest_path": manifest_name, "final_manifest_sha256": k5.sha256_file(manifest_path), "manifest_size_bytes": manifest_path.stat().st_size, "created_at": iso_kst()})


def final_report(summary: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# PV8-K8 Approved Research K-Action-Mask Runtime Integration Final Report",
            "",
            f"- artifact_root: `{summary['artifact_root']}`",
            f"- gate: `{summary['gate']}`",
            f"- final decision: `{summary['final_decision']}`",
            f"- approval evidence: `{summary['approval_evidence']}`",
            f"- rule version: `{RULE_VERSION}`",
            f"- experiment version: `{EXPERIMENT_VERSION}`",
            f"- static rulebook SHA-256: `{EXPECTED_RULEBOOK_SHA256}`",
            f"- occurrence master SHA-256: `{EXPECTED_OCCURRENCE_SHA256}`",
            f"- runtime mask rows: `{summary['runtime_rows']}`",
            f"- runtime rows passed: `{summary['runtime_passed']}/{summary['runtime_rows']}`",
            f"- C2 active / inactive cycles: `{summary['active_cycles']} / {summary['inactive_cycles']}`",
            f"- positive SKIP count: `{summary['positive_skip_count']}`",
            f"- occurrence lookup failures: `{summary['lookup_failures']}`",
            f"- identity failures / slot mutations: `{summary['identity_failures']} / {summary['slot_mutations']}`",
            f"- inactive action leakage / future violations: `{summary['inactive_leakage']} / {summary['future_violations']}`",
            f"- rule/hash drift: `{summary['rule_hash_drift']}`",
            f"- relevant pytest: `{summary['pytest_passed']} passed`",
            f"- research_rule_approved: `true`",
            f"- K_action_mask_available: `true`",
            "",
            "The direct user response '승인 계속진행' approved the previously enumerated six assumptions, research-only claim boundary, exact hashes, and K8 runtime-mask-only scope. It does not claim operator approval or observed Daegu operating rules.",
            "",
            "The runtime adapter binds an exact route_stop_occurrence_id, the approved research static row, K4 decision-time obligations, and frozen version metadata. Inactive agents receive an all-false mask. Any identity, occurrence, state, path, or hash problem fails closed for SKIP.",
            "",
            "## Remaining locks",
            "",
            "Conditional-SKIP policy execution, training, checkpoint reuse, policy evaluation, causal claims, paper claims, K9, and MAPPO retraining remain unauthorized.",
            "",
            "## Minimum K9 / retraining-preflight scope",
            "",
            "Bind the approved runtime adapter and version metadata into the global orchestrator/replay snapshot lifecycle; verify clone/reset determinism and action-mask coverage over frozen prospective episodes; audit all-false active masks and action availability distributions; then request separate policy-evaluation and retraining-preflight authorization. K8 does not provide either authorization.",
            "",
        ]
    )


def run_integration(root: Path) -> Path:
    root = k5.validate_artifact_root(root)
    writer = k5.Writer(root)
    upstreams = verify_upstreams()
    rulebook_sha = k5.sha256_file(RULEBOOK_PATH)
    occurrence_sha = k5.sha256_file(OCCURRENCE_PATH)
    if rulebook_sha != EXPECTED_RULEBOOK_SHA256 or occurrence_sha != EXPECTED_OCCURRENCE_SHA256:
        raise K8Error(f"frozen hash mismatch: rulebook={rulebook_sha}, occurrence={occurrence_sha}")

    approval = approval_record()
    writer.json("k8_explicit_rule_approval_record.json", approval)
    approval_sha = k5.sha256_file(root / "k8_explicit_rule_approval_record.json")
    k4_hashes = current_k4_hashes()
    if k4_hashes["source_drift_count"]:
        raise K8Error(f"K4 source drift: {k4_hashes}")

    rule_frame = pd.read_parquet(RULEBOOK_PATH)
    rule_rows = rule_frame.to_dict("records")
    bindings, mapping = fixed_vehicle_bindings()
    version_binding = RuntimeVersionBinding(
        k_safety_state_version=K_SAFETY_STATE_SCHEMA_VERSION,
        static_rulebook_version=RULE_VERSION,
        static_rulebook_sha256=rulebook_sha,
        occurrence_master_sha256=occurrence_sha,
        dynamic_state_contract_version=K_SAFETY_STATE_SCHEMA_VERSION,
        mask_predicate_version=MASK_PREDICATE_VERSION,
        experiment_version=EXPERIMENT_VERSION,
    )
    runtime = FixedVehicleOccurrenceMaskRuntime(
        rule_rows=rule_rows,
        fixed_vehicle_bindings=bindings,
        approval_record=approval,
        version_binding=version_binding,
        actual_rulebook_sha256=rulebook_sha,
        actual_occurrence_master_sha256=occurrence_sha,
    )

    frozen_contract = {
        "created_at": iso_kst(),
        "contract_name": EXPERIMENT_VERSION,
        "research_rule_approved": True,
        "approval_record_path": "k8_explicit_rule_approval_record.json",
        "approval_record_sha256": approval_sha,
        "static_rulebook_source_path": str(RULEBOOK_PATH),
        "static_rulebook_version": RULE_VERSION,
        "static_rulebook_sha256": rulebook_sha,
        "static_rulebook_row_count": len(rule_rows),
        "occurrence_master_path": str(OCCURRENCE_PATH),
        "occurrence_master_sha256": occurrence_sha,
        "runtime_source_path": str(RUNTIME_SOURCE),
        "runtime_source_sha256": k5.sha256_file(RUNTIME_SOURCE),
        "runtime_test_source_sha256": k5.sha256_file(RUNTIME_TEST_SOURCE),
        "fixed_vehicle_mapping_sha256": k5.sha256_file(C2_MAPPING_PATH),
        "approved_claim_boundary": approval["claim_boundary"],
        "approved_scope": approval["approved_scope"],
        "real_network_rule_claim_allowed": False,
        "policy_execution_authorized": False,
        "rule_or_hash_change_requires_new_experiment_version": True,
    }
    writer.json("k8_frozen_rule_contract.json", frozen_contract)

    runtime_rows, cycle_audit = build_runtime_rows(runtime, rule_rows, mapping)
    writer.parquet("k8_runtime_k_mask_integration.parquet", runtime_rows, RUNTIME_COLUMNS)
    pytest_result = run_pytest()
    scenario_counts = Counter(row["scenario_id"] for row in runtime_rows)
    validation = {
        "created_at": iso_kst(),
        "runtime_row_count": len(runtime_rows),
        "runtime_passed_count": sum(bool(row["passed"]) for row in runtime_rows),
        "runtime_failed_count": sum(not bool(row["passed"]) for row in runtime_rows),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "fixed_agent_count": len({row["agent_id"] for row in runtime_rows}),
        "cycle_audit": cycle_audit,
        "positive_skip_count": sum(bool(row["observed_skip_valid"]) for row in runtime_rows),
        "occurrence_lookup_failure_count": sum(not bool(row["occurrence_lookup_integrity"]) for row in runtime_rows if row["active_bus_mask"]),
        "slot_mutation_count": sum(bool(row["slot_mutation"]) for row in runtime_rows),
        "inactive_action_leakage_count": sum(bool(row["inactive_action_leakage"]) for row in runtime_rows),
        "identity_failure_count": sum(bool(row["identity_failure"]) for row in runtime_rows),
        "future_information_violation_count": sum(bool(row["future_information_violation"]) for row in runtime_rows),
        "rule_hash_drift_count": sum(bool(row["rule_hash_drift"]) for row in runtime_rows),
        "reappearance_case_count": sum(bool(row["reappearance_case"]) for row in runtime_rows),
        "direction_transition_case_count": sum(bool(row["direction_transition_case"]) for row in runtime_rows),
        "all_eight_agents_integrated": len({row["agent_id"] for row in runtime_rows}) == 8,
        "pickup_blocker_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "PICKUP_BLOCKER" and row["passed"] and row["pickup_obligation"] for row in runtime_rows) for agent in range(8)),
        "dropoff_onboard_blocker_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "DROPOFF_ONBOARD_BLOCKER" and row["passed"] and row["dropoff_obligation"] and row["onboard_destination_obligation"] for row in runtime_rows) for agent in range(8)),
        "terminal_blocker_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "TERMINAL_BLOCKER" and row["passed"] for row in runtime_rows) for agent in range(8)),
        "no_target_blocker_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "NO_TARGET_BLOCKER" and row["passed"] for row in runtime_rows) for agent in range(8)),
        "incomplete_state_blocker_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "INCOMPLETE_STATE_BLOCKER" and row["passed"] for row in runtime_rows) for agent in range(8)),
        "positive_skip_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "POSITIVE_SKIP_APPROVED_RESEARCH_RULE" and row["passed"] and row["observed_skip_valid"] for row in runtime_rows) for agent in range(8)),
        "no_future_boundary_all_agents": all(any(row["agent_id"] == agent and row["scenario_id"] == "NO_FUTURE_LEAK_PRE_BOUNDARY" and row["passed"] for row in runtime_rows) and any(row["agent_id"] == agent and row["scenario_id"] == "NO_FUTURE_LEAK_POST_BOUNDARY" and row["passed"] for row in runtime_rows) for agent in range(8)),
        "pytest_regression": pytest_result,
        "policy_execution_count": 0,
    }
    writer.json("k8_runtime_mask_validation.json", validation)

    snapshot_binding = {
        "created_at": iso_kst(),
        "binding_name": "PV8_K8_APPROVED_RESEARCH_KMASK_SNAPSHOT_BINDING_V1",
        **version_binding.to_payload(),
        "research_rule_approval_record_sha256": approval_sha,
        "runtime_source_sha256": k5.sha256_file(RUNTIME_SOURCE),
        "fixed_vehicle_mapping_sha256": k5.sha256_file(C2_MAPPING_PATH),
        "route_stop_occurrence_id_required": True,
        "all_metadata_required_before_runtime_mask": True,
        "identity_occurrence_state_or_hash_problem_forces_skip_false": True,
        "inactive_agent_action_mask": [False, False, False],
        "rule_or_hash_change_requires_new_experiment_version": True,
        "clone_reset_must_preserve_binding": True,
        "research_rule_approved": True,
        "K_action_mask_available": True,
        "conditional_skip_policy_enabled": False,
    }
    writer.json("k8_snapshot_version_binding.json", snapshot_binding)

    integration_passed = bool(
        validation["runtime_failed_count"] == 0
        and validation["fixed_agent_count"] == 8
        and validation["cycle_audit"]["c2_cycle_row_count"] == 240
        and validation["cycle_audit"]["c2_inactive_cycle_count"] == 26
        and validation["occurrence_lookup_failure_count"] == 0
        and validation["slot_mutation_count"] == 0
        and validation["inactive_action_leakage_count"] == 0
        and validation["identity_failure_count"] == 0
        and validation["future_information_violation_count"] == 0
        and validation["rule_hash_drift_count"] == 0
        and validation["reappearance_case_count"] == 1
        and validation["direction_transition_case_count"] == 1
        and validation["pickup_blocker_all_agents"]
        and validation["dropoff_onboard_blocker_all_agents"]
        and validation["terminal_blocker_all_agents"]
        and validation["no_target_blocker_all_agents"]
        and validation["incomplete_state_blocker_all_agents"]
        and validation["positive_skip_all_agents"]
        and validation["no_future_boundary_all_agents"]
        and pytest_result["exit_code"] == 0
    )
    final_decision = DECISION_READY if approval["approval_valid"] and integration_passed else DECISION_PENDING if not approval["approval_valid"] else DECISION_FAILED
    audit_passed = final_decision == DECISION_READY
    readiness = {
        "created_at": iso_kst(),
        "final_decision": final_decision,
        "research_rule_approved": audit_passed,
        "K_action_mask_available": audit_passed,
        "conditional_skip_policy_enabled": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_evaluation_authorized": False,
        "runtime_integration_passed": integration_passed,
        "minimum_k9_retraining_preflight_scope": [
            "bind the approved runtime adapter and snapshot binding into the global orchestrator and replay clone/reset lifecycle",
            "run frozen prospective episode-level mask coverage and determinism checks without policy action selection",
            "audit active all-false masks, HOLD/SERVE/SKIP availability distributions, and obligation-block frequencies",
            "freeze the K8 experiment version and all source/config hashes for any candidate retraining dataset",
            "request separate policy-evaluation and MAPPO retraining-preflight authorization",
        ],
    }
    guards = {
        "created_at": iso_kst(),
        "research_rule_approved": audit_passed,
        "K_action_mask_available": audit_passed,
        "conditional_skip_policy_enabled": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_evaluation_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "real_network_rule_claim_allowed": False,
        "automatic_k9_execution_authorized": False,
        "automatic_mappo_retraining_authorized": False,
    }
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE if audit_passed else FAIL_GATE,
        "terminal_gate": PASS_GATE if audit_passed else FAIL_GATE,
        "readiness": PASS_READINESS if audit_passed else "SRP2_BIS_PV8_K8_INTEGRATION_FAILED",
        "gate_passed": audit_passed,
        "final_decision": final_decision,
        "failure_reasons": [] if audit_passed else ["approval, frozen hash, upstream/source integrity, runtime mask, identity, no-future, or regression condition failed"],
    }
    summary = {
        "artifact_root": str(root),
        "gate": gate["gate"],
        "final_decision": final_decision,
        "approval_evidence": approval["approval_evidence_text"],
        "runtime_rows": len(runtime_rows),
        "runtime_passed": validation["runtime_passed_count"],
        "active_cycles": cycle_audit["c2_active_cycle_count"],
        "inactive_cycles": cycle_audit["c2_inactive_cycle_count"],
        "positive_skip_count": validation["positive_skip_count"],
        "lookup_failures": validation["occurrence_lookup_failure_count"],
        "identity_failures": validation["identity_failure_count"],
        "slot_mutations": validation["slot_mutation_count"],
        "inactive_leakage": validation["inactive_action_leakage_count"],
        "future_violations": validation["future_information_violation_count"],
        "rule_hash_drift": validation["rule_hash_drift_count"],
        "pytest_passed": pytest_result["passed_test_count"],
    }
    writer.json("k8_readiness_decision.json", readiness)
    writer.json("claim_guard_status.json", guards)
    writer.json(
        "run_manifest.json",
        {
            "created_at": iso_kst(),
            "artifact_family": ARTIFACT_PREFIX,
            "mode": "integrate",
            "runner_path": str(RUNNER_PATH),
            "runner_sha256": k5.sha256_file(RUNNER_PATH),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "upstream_integrity": upstreams,
            "k4_source_integrity": k4_hashes,
            "runtime_source_sha256": k5.sha256_file(RUNTIME_SOURCE),
            "db_query_count": 0,
            "db_write_count": 0,
            "new_bis_api_call_count": 0,
            "policy_execution_count": 0,
            "checkpoint_reuse_count": 0,
            "k9_execution_count": 0,
            "mappo_training_count": 0,
        },
    )
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {**guards, "source_gate": gate["gate"], "readiness": gate["readiness"], "final_decision": final_decision})
    writer.text("final_report.md", final_report(summary))
    write_manifest_and_lock(writer, gate)
    own = k5.verify_manifest(root, "artifact_manifest_srp2_bis_pv8_k8.json", "_PV8_K8_COMPLETE.lock")
    if not k5.manifest_ok(own):
        raise K8Error(f"K8 artifact integrity failure: {own}")
    if not audit_passed:
        raise K8Error(f"K8 integration failed: decision={final_decision}, validation={validation}")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"research_rule_approved: {guards['research_rule_approved']}")
    print(f"K_action_mask_available: {guards['K_action_mask_available']}")
    print(f"readiness: {gate['readiness']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["integrate"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run_integration(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
