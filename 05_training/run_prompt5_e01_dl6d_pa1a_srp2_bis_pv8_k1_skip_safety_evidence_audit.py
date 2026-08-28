#!/usr/bin/env python3
"""Prompt 5-E01-DL-6D-PA1A-SRP2-BIS-PV8-K1.

Passenger / service-obligation / skip-safety evidence gap closure.

This runner performs a deterministic read-only audit of existing project
evidence. It does not call BIS, query or write PostgreSQL directly, execute a
simulator step, evaluate policy performance, train, mutate checkpoints, or
promote aggregate demand into individual passenger state.
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
from typing import Any, Dict, List, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_REL = "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k1_skip_safety_evidence_audit.py"
RUNNER_PATH = TRAINING_ROOT / RUNNER_REL

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
C2_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"
C2_MANIFEST = "artifact_manifest_srp2_bis_pv8_c2.json"
C2_LOCK = "_PV8_C2_COMPLETE.lock"

C3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c3_mappo_interface_compatibility_20260808_100118"
C3_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_COMPLETE"
C3_MANIFEST = "artifact_manifest_srp2_bis_pv8_c3.json"
C3_LOCK = "_PV8_C3_COMPLETE.lock"

SRP1_R4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp1_r4_postgresql_first_reconciliation_20260803_224422"
SF0_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_sf0_historical_state_feasibility_audit_20260803_185838"
C1_CAPTURE_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_c1_limited_pilot_capture_20260805_070246"
C0_PREFLIGHT_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_c0_capture_contract_preflight_20260804_184415"
DEMAND_ROOT = ARTIFACTS_ROOT / "prompt5_e01_r2b_demand_fleet_capacity_calibration_20260720_000000"
SERVICE_GRAPH_ROOT = ARTIFACTS_ROOT / "suseong_service_graph_v1"
SOURCE_PACK_ROOT = ARTIFACTS_ROOT / "suseong_source_pack_v1"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k1_skip_safety_evidence_audit"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K1_SKIP_SAFETY_EVIDENCE_AUDIT_COMPLETE"
READINESS = "SRP2_BIS_PV8_K1_COMPLETE_SKIP_SAFETY_EVIDENCE_AUDIT_BLOCKED_CRITICAL_EVIDENCE_GAPS_PENDING_REPAIR"

FINAL_DECISIONS = {
    "K_SAFETY_READY",
    "K_SAFETY_READY_WITH_CONSERVATIVE_MASK",
    "K_SAFETY_BLOCKED_CRITICAL_EVIDENCE_GAPS",
}
FIELD_CLASSES = {"OBSERVED", "DERIVED_SAFE", "PROXY_GUARDED", "NOT_AVAILABLE"}

PAYLOADS = [
    "k_safety_field_inventory.json",
    "k_safety_source_lineage.json",
    "service_obligation_gap_audit.json",
    "skip_legality_evidence_audit.json",
    "conservative_action_mask_design.json",
    "k_safety_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class K1Error(RuntimeError):
    pass


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return value


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root

    def text(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def verify_manifest(root: Path, manifest_name: str, lock_name: str) -> Dict[str, Any]:
    manifest_path = root / manifest_name
    lock_path = root / lock_name
    checks: Dict[str, Any] = {
        "manifest_present": manifest_path.exists(),
        "lock_present": lock_path.exists(),
        "manifest_hash_ok": False,
        "manifest_size_ok": False,
        "manifest_entry_count": None,
        "missing_files": None,
        "required_missing_files": None,
        "sha256_mismatches": None,
        "size_mismatches": None,
        "terminal_lock_binding": False,
    }
    if not manifest_path.exists() or not lock_path.exists():
        return checks
    manifest = read_json(manifest_path)
    lock = read_json(lock_path)
    missing = required_missing = sha_bad = size_bad = 0
    for row in manifest.get("files", []):
        path = root / str(row.get("relative_path"))
        if not path.exists():
            missing += 1
            if row.get("required", True):
                required_missing += 1
            continue
        if row.get("sha256") and sha256_file(path) != row["sha256"]:
            sha_bad += 1
        if row.get("size_bytes") is not None and path.stat().st_size != row["size_bytes"]:
            size_bad += 1
    manifest_sha = sha256_file(manifest_path)
    checks.update({
        "manifest_hash_ok": manifest_sha == lock.get("manifest_sha256") or manifest_sha == lock.get("final_manifest_sha256"),
        "manifest_size_ok": manifest_path.stat().st_size == lock.get("manifest_size_bytes", manifest_path.stat().st_size),
        "manifest_entry_count": len(manifest.get("files", [])),
        "missing_files": missing,
        "required_missing_files": required_missing,
        "sha256_mismatches": sha_bad,
        "size_mismatches": size_bad,
        "terminal_lock_binding": lock.get("manifest_relative_path") == manifest_name or lock.get("final_manifest_path") == manifest_name,
    })
    return checks


def manifest_ok(checks: Mapping[str, Any]) -> bool:
    return (
        bool(checks.get("manifest_present"))
        and bool(checks.get("lock_present"))
        and bool(checks.get("manifest_hash_ok"))
        and bool(checks.get("manifest_size_ok"))
        and checks.get("missing_files") == 0
        and checks.get("required_missing_files") == 0
        and checks.get("sha256_mismatches") == 0
        and checks.get("size_mismatches") == 0
        and bool(checks.get("terminal_lock_binding"))
    )


def verify_upstream() -> Dict[str, Any]:
    c2_gate = read_json(C2_ROOT / "gate_decision.json")
    c3_gate = read_json(C3_ROOT / "gate_decision.json")
    c2_checks = verify_manifest(C2_ROOT, C2_MANIFEST, C2_LOCK)
    c3_checks = verify_manifest(C3_ROOT, C3_MANIFEST, C3_LOCK)
    if c2_gate.get("gate") != C2_GATE or not manifest_ok(c2_checks):
        raise K1Error("UPSTREAM_C2_INTEGRITY_FAILURE")
    if c3_gate.get("gate") != C3_GATE or not manifest_ok(c3_checks):
        raise K1Error("UPSTREAM_C3_INTEGRITY_FAILURE")
    return {
        "c2": {"artifact_root": str(C2_ROOT), "gate": c2_gate.get("gate"), "readiness": c2_gate.get("readiness"), "manifest_integrity": c2_checks},
        "c3": {"artifact_root": str(C3_ROOT), "gate": c3_gate.get("gate"), "readiness": c3_gate.get("readiness"), "manifest_integrity": c3_checks},
    }


def file_record(role: str, path: Path) -> Dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def source_lineage(upstream: Mapping[str, Any]) -> Dict[str, Any]:
    sources = [
        file_record("PV8-C2 prospective mapping manifest", C2_ROOT / "prospective_8vehicle_mapping_manifest.parquet"),
        file_record("PV8-C2 agent-cycle state", C2_ROOT / "prospective_agent_cycle_state.parquet"),
        file_record("PV8-C3 action mask path audit", C3_ROOT / "action_mask_path_audit.json"),
        file_record("SF0 route-stop safety field matrix", SF0_ROOT / "route_stop_safety_field_matrix.json"),
        file_record("SRP1-R4 aggregate demand use contract", SRP1_R4_ROOT / "aggregate_demand_use_contract.json"),
        file_record("SRP1-R4 dynamics state DB field mapping", SRP1_R4_ROOT / "dynamics_state_db_field_mapping.json"),
        file_record("SRP1-R4 dynamics state DB field mapping rows", SRP1_R4_ROOT / "dynamics_state_db_field_mapping.jsonl"),
        file_record("SRP1-R4 PostgreSQL column registry", SRP1_R4_ROOT / "postgresql_column_registry.jsonl"),
        file_record("SRP1-R4 existing BIS artifact reconciliation", SRP1_R4_ROOT / "existing_bis_artifact_reconciliation.json"),
        file_record("C0 getPos02 normalized schema", C0_PREFLIGHT_ROOT / "getpos02_normalized_schema.json"),
        file_record("C0 getRealtime02 normalized schema", C0_PREFLIGHT_ROOT / "getrealtime02_normalized_schema.json"),
        file_record("C1 getPos02 normalized capture", C1_CAPTURE_ROOT / "getpos02_normalized.parquet"),
        file_record("C1 getRealtime02 normalized capture", C1_CAPTURE_ROOT / "getrealtime02_normalized.parquet"),
        file_record("C1 route-stop crossmatch audit", C1_CAPTURE_ROOT / "route_stop_crossmatch_audit.json"),
        file_record("route stop sequences", SOURCE_PACK_ROOT / "route_stop_sequences.parquet"),
        file_record("Suseong service route sequences", SERVICE_GRAPH_ROOT / "service_route_sequences.csv"),
        file_record("demand alignment by stop", DEMAND_ROOT / "demand_alignment_by_stop.parquet"),
        file_record("tensor DB data requirements", TRAINING_ROOT / "adapters/tensor_db_data_requirements_v2.json"),
        file_record("tensor DB availability audit", TRAINING_ROOT / "adapters/tensor_db_data_availability_audit.md"),
        file_record("conditional skip safety engine", TRAINING_ROOT / "simulator/suseong_service_transition_engine.py"),
        file_record("multiagent dynamics orchestrator", TRAINING_ROOT / "simulator/dynamics_multiagent_orchestrator.py"),
        file_record("DL6C conditional skip safety tests", TRAINING_ROOT / "simulator/test_dl6c_conditional_skip_safety.py"),
    ]
    getpos_cols = list(pd.read_parquet(C1_CAPTURE_ROOT / "getpos02_normalized.parquet").columns)
    realtime_cols = list(pd.read_parquet(C1_CAPTURE_ROOT / "getrealtime02_normalized.parquet").columns)
    demand_cols = list(pd.read_parquet(DEMAND_ROOT / "demand_alignment_by_stop.parquet").columns)
    route_seq_cols = list(pd.read_parquet(SOURCE_PACK_ROOT / "route_stop_sequences.parquet").columns)
    crossmatch = read_json(C1_CAPTURE_ROOT / "route_stop_crossmatch_audit.json")
    dynamics_mapping = read_json(SRP1_R4_ROOT / "dynamics_state_db_field_mapping.json")
    aggregate_contract = read_json(SRP1_R4_ROOT / "aggregate_demand_use_contract.json")
    return {
        "created_at": iso_kst(),
        "upstream": upstream,
        "sources": sources,
        "data_source_scope": {
            "new_bis_api_collection_count": 0,
            "direct_postgresql_write_count": 0,
            "direct_postgresql_query_count": 0,
            "postgresql_evidence_mode": "existing read-only SRP1-R4 reconciliation artifacts",
            "simulator_execution_count": 0,
        },
        "observed_columns": {
            "getpos02_normalized": getpos_cols,
            "getrealtime02_normalized": realtime_cols,
            "demand_alignment_by_stop": demand_cols,
            "route_stop_sequences": route_seq_cols,
        },
        "route_stop_crossmatch": crossmatch,
        "dynamics_state_db_field_mapping_summary": dynamics_mapping,
        "aggregate_demand_use_contract": aggregate_contract,
    }


def field_record(field: str, classification: str, evidence: Sequence[str], safe_use: str, blocking_reason: str = "") -> Dict[str, Any]:
    if classification not in FIELD_CLASSES:
        raise ValueError(f"invalid field class {classification}")
    return {
        "field": field,
        "classification": classification,
        "evidence": list(evidence),
        "safe_use": safe_use,
        "blocking_reason": blocking_reason,
        "aggregate_to_individual_promotion_allowed": False,
    }


def build_field_inventory() -> Dict[str, Any]:
    records = [
        field_record("vehicle_identity", "OBSERVED", ["PV8-C2 fixed physical_vehicle_token from getPos02 tokenized provider vehicle id"], "bind fixed 8 physical vehicle slots"),
        field_record("current_vehicle_position", "OBSERVED", ["C1/C2 getPos02 current_stop_id, route_sequence, x/y, provider event time"], "identify current stop/sequence for active agents"),
        field_record("route_id", "OBSERVED", ["C1/C2 getPos02 route_id; route_stop_sequences route_id"], "route-specific topology lookup"),
        field_record("direction_id", "OBSERVED", ["C1/C2 getPos02 direction_id; direction transition preserved in C2"], "current direction feature and topology lookup"),
        field_record("ordered_stop_sequence", "DERIVED_SAFE", ["getBs02/full route_stop_sequences 20508 rows; C1 route-stop crossmatch 677/677"], "derive next and post-skip candidate stops from static topology"),
        field_record("graph_edge_or_path_valid", "DERIVED_SAFE", ["service route sequence and graph topology artifacts"], "guard SERVE path existence and post-skip target existence"),
        field_record("waiting_passenger_demand", "PROXY_GUARDED", ["graph_state_timeslice.waiting_passenger_cnt and demand_alignment_by_stop are aggregate/proxy context"], "may guard by blocking skip when positive/unknown; cannot prove empty stop", "node/hour aggregate is not an anchor-time passenger queue"),
        field_record("boarding_obligation", "PROXY_GUARDED", ["boardings_recent / historical demand by stop exist only as aggregate demand"], "may be used as conservative risk signal only", "no request-level pickup entity or exact waiting passenger state"),
        field_record("alighting_obligation", "PROXY_GUARDED", ["alightings_recent exists as aggregate demand"], "may be used as conservative risk signal only", "no vehicle-specific onboard destination or scheduled alighting state"),
        field_record("onboard_destination", "NOT_AVAILABLE", ["SRP1-R4 mapping: onboard_passengers unavailable; tensor requirements: vehicle_load/onboard_count missing"], "must block skip clearance", "no onboard passenger destination entity"),
        field_record("assigned_pickup", "NOT_AVAILABLE", ["SRP1-R4 mapping: assigned_pickups unavailable"], "must block skip clearance", "no request-level pickup assignment source"),
        field_record("assigned_dropoff", "NOT_AVAILABLE", ["SRP1-R4 mapping: assigned_dropoffs unavailable"], "must block skip clearance", "no request-level dropoff assignment source"),
        field_record("mandatory_stop", "NOT_AVAILABLE", ["SF0: mandatory_stop unavailable; SRP1-R4: mandatory_stop_state unavailable"], "must block skip clearance", "no mandatory stop flag/rulebook source"),
        field_record("protected_stop", "NOT_AVAILABLE", ["SF0: protected_stop unavailable; no protected stop rulebook source"], "must block skip clearance", "no protected stop flag/rulebook source"),
        field_record("terminal_or_turnaround_stop", "NOT_AVAILABLE", ["SF0: terminal_or_turnaround_stop unavailable for safety flags"], "must block skip clearance except static terminal can be separately reviewed", "no authoritative terminal/turnaround operational flag in K safety source"),
        field_record("planned_itinerary_allows_skip", "NOT_AVAILABLE", ["SF0: planned_itinerary_allows_skip unavailable"], "must block skip clearance", "no operation rulebook or itinerary skip permission table"),
        field_record("downstream_path_valid", "NOT_AVAILABLE", ["SF0: downstream_path_valid unavailable"], "must block skip clearance", "no dynamic operational path-valid flag"),
        field_record("max_consecutive_skip_constraint", "NOT_AVAILABLE", ["SF0: max_consecutive_skip_constraint_satisfied unavailable"], "must block skip clearance", "no running skip counter/source for production policy"),
        field_record("service_fairness_constraint", "NOT_AVAILABLE", ["SF0: service_fairness_constraint_satisfied unavailable"], "must block skip clearance", "no service fairness rule state"),
        field_record("service_obligation", "NOT_AVAILABLE", ["pickup/dropoff/onboard/mandatory/protected components are missing or proxy-only"], "cannot clear CONDITIONAL_SKIP", "critical service obligation state is incomplete"),
        field_record("skip_legality", "NOT_AVAILABLE", ["tensor requirements classify skip_stop_action_legality as assumed/missing external rulebook"], "uncertain legality -> mask skip out", "no legal/operational skip permission source"),
        field_record("actual_dwell", "NOT_AVAILABLE", ["tensor requirements: observed dwell requires arrival/departure events"], "not usable for K skip clearance", "arrival/departure stop events absent"),
        field_record("estimated_dwell", "PROXY_GUARDED", ["boarding/alighting aggregate can support formula dwell only as proxy"], "debug/scenario shaping only; not a service-obligation clearance", "estimated dwell cannot prove no passenger service obligation"),
        field_record("arrival_evidence", "PROXY_GUARDED", ["getRealtime02 ETA and getPos02 current stop/sequence exist"], "arrival context only, not exact stop-arrival event stream", "no authoritative arrival event log"),
        field_record("departure_evidence", "NOT_AVAILABLE", ["tensor requirements: departure_time missing for observed dwell"], "not usable for dwell/service clearance", "no departure event log"),
    ]
    counts: Dict[str, int] = {}
    for row in records:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    critical_missing = [
        row["field"]
        for row in records
        if row["field"] in {
            "onboard_destination",
            "assigned_pickup",
            "assigned_dropoff",
            "mandatory_stop",
            "protected_stop",
            "service_obligation",
            "skip_legality",
        }
        and row["classification"] == "NOT_AVAILABLE"
    ]
    return {
        "created_at": iso_kst(),
        "classification_classes": sorted(FIELD_CLASSES),
        "record_count": len(records),
        "classification_counts": counts,
        "critical_missing_fields": critical_missing,
        "records": records,
    }


def service_gap_audit(inventory: Mapping[str, Any]) -> Dict[str, Any]:
    by_field = {row["field"]: row for row in inventory["records"]}
    tests = {
        "can_serve_be_identified_safely": {
            "answer": "PARTIAL",
            "reason": "SERVE/MOVE route feasibility can be derived from current position and static topology, but actual board/alight counts and dwell are not observed.",
        },
        "can_onboard_alighting_obligations_be_known": {
            "answer": "NO",
            "blocking_fields": ["onboard_destination", "assigned_dropoff", "alighting_obligation"],
        },
        "can_pickup_obligations_be_known": {
            "answer": "NO_EXACT_ONLY_PROXY_GUARD",
            "blocking_fields": ["waiting_passenger_demand", "boarding_obligation", "assigned_pickup"],
        },
        "can_service_obligation_be_cleared": {
            "answer": "NO",
            "blocking_fields": ["service_obligation"],
        },
    }
    missing = [
        field for field in [
            "onboard_destination",
            "assigned_pickup",
            "assigned_dropoff",
            "mandatory_stop",
            "protected_stop",
            "planned_itinerary_allows_skip",
            "max_consecutive_skip_constraint",
            "service_fairness_constraint",
        ]
        if by_field[field]["classification"] == "NOT_AVAILABLE"
    ]
    return {
        "created_at": iso_kst(),
        "aggregate_demand_promoted_to_individual_state": False,
        "service_obligation_clearance_available": False,
        "tests": tests,
        "critical_missing_evidence": missing,
        "minimum_next_repair": [
            "Add an authoritative stop-safety/rulebook table for mandatory_stop, protected_stop, planned_itinerary_allows_skip, consecutive skip, and service fairness.",
            "Add request/passenger-event or APC/AVL-derived state for waiting passenger queue, assigned pickup/dropoff, onboard destination, and scheduled alighting at the decision timestamp.",
            "Add arrival/departure stop event evidence if dwell or service completion is to be observed rather than proxied.",
        ],
    }


def skip_legality_audit(inventory: Mapping[str, Any]) -> Dict[str, Any]:
    by_field = {row["field"]: row for row in inventory["records"]}
    engine_required = [
        "waiting_passenger_demand",
        "alighting_obligation",
        "assigned_pickup",
        "assigned_dropoff",
        "mandatory_stop",
        "protected_stop",
        "planned_itinerary_allows_skip",
        "downstream_path_valid",
        "graph_edge_or_path_valid",
        "max_consecutive_skip_constraint",
        "service_fairness_constraint",
        "onboard_destination",
    ]
    usable_for_positive_clearance = []
    not_usable = []
    for field in engine_required:
        cls = by_field[field]["classification"]
        if cls in {"OBSERVED", "DERIVED_SAFE"}:
            usable_for_positive_clearance.append(field)
        else:
            not_usable.append(field)
    can_declare_skip_legal = len(not_usable) == 0
    return {
        "created_at": iso_kst(),
        "can_skip_be_declared_legal_safely": can_declare_skip_legal,
        "safety_rule": "uncertain skip legality -> SKIP masked out",
        "engine_predicate_source": "05_training/simulator/suseong_service_transition_engine.py:evaluate_conditional_skip_safety",
        "engine_predicate_source_sha256": sha256_file(TRAINING_ROOT / "simulator/suseong_service_transition_engine.py"),
        "engine_required_fields": engine_required,
        "positive_clearance_fields_available": usable_for_positive_clearance,
        "positive_clearance_fields_missing_or_proxy_only": not_usable,
        "mandatory_protected_stops_identifiable": False,
        "skip_legal_positive_case_count": 0,
        "skip_masked_out_due_to_uncertainty": True,
    }


def conservative_mask_design() -> Dict[str, Any]:
    state = pd.read_parquet(C2_ROOT / "prospective_agent_cycle_state.parquet")
    active_counts = state.groupby("cycle_index")["active_bus_mask"].sum().astype(int)
    return {
        "created_at": iso_kst(),
        "mask_name": "PV8_K1_CONSERVATIVE_FAIL_CLOSED_MASK_V1",
        "constructible_without_future_leakage": True,
        "ready_for_conditional_skip_enablement": False,
        "action_order": ["HOLD", "SERVE", "CONDITIONAL_SKIP"],
        "active_agent_rule": {
            "HOLD": True,
            "SERVE": "true iff active_bus_mask=true and current route has a next candidate stop/path",
            "CONDITIONAL_SKIP": False,
        },
        "inactive_agent_rule": {
            "all_actions": False,
            "simulator_intervention_dispatch": "drop inactive agent IDs before dispatch",
        },
        "skip_enablement_rule_future": "Enable only when every passenger/service/rulebook/path predicate is OBSERVED or DERIVED_SAFE at the decision timestamp.",
        "current_k1_skip_action_mask": "always false because positive skip legality cannot be cleared from existing evidence",
        "c2_cycle_count": int(state["cycle_index"].nunique()),
        "c2_active_agent_count_distribution": {str(int(k)): int(v) for k, v in active_counts.value_counts().sort_index().to_dict().items()},
        "no_future_inputs": ["C2 current cycle state", "static route topology", "existing read-only evidence classifications"],
        "future_or_forbidden_inputs": ["future passenger outcomes", "future survival/support", "aggregate demand promoted to passenger entity"],
    }


def readiness_decision(service_gap: Mapping[str, Any], skip_audit: Mapping[str, Any], mask: Mapping[str, Any]) -> Dict[str, Any]:
    final_decision = "K_SAFETY_BLOCKED_CRITICAL_EVIDENCE_GAPS"
    assert final_decision in FINAL_DECISIONS
    return {
        "created_at": iso_kst(),
        "final_decision": final_decision,
        "audit_complete": True,
        "deterministic": True,
        "can_serve_be_identified_safely": service_gap["tests"]["can_serve_be_identified_safely"]["answer"],
        "can_skip_be_declared_legal_safely": skip_audit["can_skip_be_declared_legal_safely"],
        "can_mandatory_protected_stops_be_identified": skip_audit["mandatory_protected_stops_identifiable"],
        "can_onboard_alighting_obligations_be_known": service_gap["tests"]["can_onboard_alighting_obligations_be_known"]["answer"],
        "can_pickup_obligations_be_known": service_gap["tests"]["can_pickup_obligations_be_known"]["answer"],
        "conservative_k_action_mask_constructible_without_future_leakage": mask["constructible_without_future_leakage"],
        "why_not_ready_with_conservative_mask": (
            "A fail-closed mask can safely disable CONDITIONAL_SKIP, but it cannot distinguish "
            "legal skip opportunities or close K-safety evidence gaps. The K layer remains blocked "
            "until positive skip clearance predicates are sourced."
        ),
        "critical_missing_evidence": service_gap["critical_missing_evidence"],
        "minimum_next_repair": service_gap["minimum_next_repair"],
    }


def claim_guard() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "K_safety_layer_complete": False,
        "K_action_mask_available_for_positive_skip": False,
        "safe_skip_decision_ready": False,
        "conservative_skip_disabled_mask_available": True,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_performance_evaluation": False,
        "policy_performance_evaluation_authorized": False,
        "simulator_performance_evaluation_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "new_bis_api_collection_authorized": False,
        "db_write_authorized": False,
    }


def write_manifest_and_lock(writer: Writer, gate: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for rel in PAYLOADS:
        path = writer.root / rel
        rows.append({
            "relative_path": rel,
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
            "required": True,
            "artifact_role": Path(rel).stem,
            "exists": path.exists(),
        })
    writer.text("artifact_manifest_srp2_bis_pv8_k1.jsonl", "".join(json.dumps(json_clean(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    rows.append({
        "relative_path": "artifact_manifest_srp2_bis_pv8_k1.jsonl",
        "size_bytes": (writer.root / "artifact_manifest_srp2_bis_pv8_k1.jsonl").stat().st_size,
        "sha256": sha256_file(writer.root / "artifact_manifest_srp2_bis_pv8_k1.jsonl"),
        "required": True,
        "artifact_role": "manifest_jsonl",
        "exists": True,
    })
    manifest = {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "payload_count": len(rows),
        "missing_payload_count": sum(1 for row in rows if not row["exists"]),
        "files": rows,
    }
    writer.json("artifact_manifest_srp2_bis_pv8_k1.json", manifest)
    manifest_path = writer.root / "artifact_manifest_srp2_bis_pv8_k1.json"
    writer.json("_PV8_K1_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": "artifact_manifest_srp2_bis_pv8_k1.json",
        "final_manifest_sha256": sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })
    return manifest


def final_report_text(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# PV8-K1 Skip-Safety Evidence Audit Final Report",
        "",
        f"- artifact_root: `{summary['artifact_root']}`",
        f"- final_decision: `{summary['final_decision']}`",
        f"- can_serve_be_identified_safely: `{summary['can_serve_be_identified_safely']}`",
        f"- can_skip_be_declared_legal_safely: `{summary['can_skip_be_declared_legal_safely']}`",
        f"- can_mandatory_protected_stops_be_identified: `{summary['can_mandatory_protected_stops_be_identified']}`",
        f"- can_onboard_alighting_obligations_be_known: `{summary['can_onboard_alighting_obligations_be_known']}`",
        f"- can_pickup_obligations_be_known: `{summary['can_pickup_obligations_be_known']}`",
        f"- conservative_k_action_mask_constructible_without_future_leakage: `{summary['conservative_k_action_mask_constructible_without_future_leakage']}`",
        f"- critical_missing_evidence: `{summary['critical_missing_evidence']}`",
        f"- minimum_next_repair: `{summary['minimum_next_repair']}`",
        f"- training_use_authorized: `{summary['training_use_authorized']}`",
        f"- checkpoint_reuse_authorized: `{summary['checkpoint_reuse_authorized']}`",
        f"- policy_performance_evaluation: `{summary['policy_performance_evaluation']}`",
        f"- causal_performance_claim_allowed: `{summary['causal_performance_claim_allowed']}`",
        f"- paper_level_claim_allowed: `{summary['paper_level_claim_allowed']}`",
        f"- gate: `{summary['gate']}`",
        f"- readiness: `{summary['readiness']}`",
        "",
        "Exact missing evidence: request/passenger-level pickup assignments, dropoff assignments, onboard destination or scheduled alighting state, mandatory/protected stop flags, skip permission/rulebook fields, consecutive-skip/fairness state, and observed arrival/departure events for dwell.",
        "",
        "Minimum next repair: add authoritative stop-safety rulebook fields and decision-time passenger/service-obligation state. Aggregate boarding/alighting/waiting demand may remain a guarded proxy, but it must not be promoted into individual passenger state.",
        "",
        "K2, retraining, checkpoint reuse, simulator policy evaluation, and performance claims were not run.",
        "",
    ])


def run_validate(root: Path) -> Path:
    root = validate_artifact_root(root)
    writer = Writer(root)
    upstream = verify_upstream()
    lineage = source_lineage(upstream)
    inventory = build_field_inventory()
    service_gap = service_gap_audit(inventory)
    skip_audit = skip_legality_audit(inventory)
    mask = conservative_mask_design()
    readiness = readiness_decision(service_gap, skip_audit, mask)
    guard = claim_guard()
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": READINESS,
        "gate_passed": True,
        "final_decision": readiness["final_decision"],
        "failure_reasons": [],
    }
    summary = {
        "artifact_root": str(root),
        "final_decision": readiness["final_decision"],
        "can_serve_be_identified_safely": readiness["can_serve_be_identified_safely"],
        "can_skip_be_declared_legal_safely": readiness["can_skip_be_declared_legal_safely"],
        "can_mandatory_protected_stops_be_identified": readiness["can_mandatory_protected_stops_be_identified"],
        "can_onboard_alighting_obligations_be_known": readiness["can_onboard_alighting_obligations_be_known"],
        "can_pickup_obligations_be_known": readiness["can_pickup_obligations_be_known"],
        "conservative_k_action_mask_constructible_without_future_leakage": readiness["conservative_k_action_mask_constructible_without_future_leakage"],
        "critical_missing_evidence": readiness["critical_missing_evidence"],
        "minimum_next_repair": readiness["minimum_next_repair"],
        "training_use_authorized": guard["training_use_authorized"],
        "checkpoint_reuse_authorized": guard["checkpoint_reuse_authorized"],
        "policy_performance_evaluation": guard["policy_performance_evaluation"],
        "causal_performance_claim_allowed": guard["causal_performance_claim_allowed"],
        "paper_level_claim_allowed": guard["paper_level_claim_allowed"],
        "gate": gate["gate"],
        "readiness": gate["readiness"],
    }

    writer.json("k_safety_source_lineage.json", lineage)
    writer.json("k_safety_field_inventory.json", inventory)
    writer.json("service_obligation_gap_audit.json", service_gap)
    writer.json("skip_legality_evidence_audit.json", skip_audit)
    writer.json("conservative_action_mask_design.json", mask)
    writer.json("k_safety_readiness_decision.json", readiness)
    writer.json("claim_guard_status.json", guard)
    writer.json("run_manifest.json", {
        "created_at": iso_kst(),
        "artifact_family": ARTIFACT_PREFIX,
        "mode": "validate",
        "runner_path": str(RUNNER_PATH),
        "runner_sha256": sha256_file(RUNNER_PATH),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "new_bis_api_collection_count": 0,
        "api_call_count": 0,
        "direct_postgresql_query_count": 0,
        "db_write_count": 0,
        "simulator_execution_count": 0,
        "policy_performance_evaluation_count": 0,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {
        "created_at": iso_kst(),
        "source_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_decision": readiness["final_decision"],
        "K_safety_layer_complete": False,
        "K_action_mask_available_for_positive_skip": False,
        "safe_skip_decision_ready": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_performance_evaluation": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "automatic_k2_execution_authorized": False,
        "automatic_retraining_authorized": False,
    })
    writer.text("final_report.md", final_report_text(summary))
    write_manifest_and_lock(writer, gate)
    own_checks = verify_manifest(root, "artifact_manifest_srp2_bis_pv8_k1.json", "_PV8_K1_COMPLETE.lock")
    if not manifest_ok(own_checks):
        raise K1Error(f"ARTIFACT_INTEGRITY_FAILURE: {own_checks}")

    for key in [
        "artifact_root",
        "final_decision",
        "can_serve_be_identified_safely",
        "can_skip_be_declared_legal_safely",
        "can_mandatory_protected_stops_be_identified",
        "can_onboard_alighting_obligations_be_known",
        "can_pickup_obligations_be_known",
        "conservative_k_action_mask_constructible_without_future_leakage",
        "critical_missing_evidence",
        "minimum_next_repair",
        "training_use_authorized",
        "checkpoint_reuse_authorized",
        "policy_performance_evaluation",
        "causal_performance_claim_allowed",
        "paper_level_claim_allowed",
        "gate",
        "readiness",
    ]:
        print(f"{key}: {summary[key]}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate"], required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    run_validate(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
