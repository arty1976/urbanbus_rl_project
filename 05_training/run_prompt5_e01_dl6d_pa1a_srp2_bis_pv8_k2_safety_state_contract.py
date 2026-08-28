#!/usr/bin/env python3
"""Prompt 5-E01-DL-6D-PA1A-SRP2-BIS-PV8-K2.

Stop safety rulebook and decision-time service obligation state contract.

This runner defines the static and dynamic contracts needed to repair the K1
gaps without fabricating unavailable passenger state. It reads only existing
artifacts and source files, writes a new K2 artifact, and keeps skip enablement,
training, checkpoint reuse, policy evaluation, simulator evaluation, DB writes,
and new BIS collection locked.
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
RUNNER_REL = "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k2_safety_state_contract.py"
RUNNER_PATH = TRAINING_ROOT / RUNNER_REL

C2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c2_prospective_mapping_validation_20260808_084612"
C3_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c3_mappo_interface_compatibility_20260808_100118"
K1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k1_skip_safety_evidence_audit_20260808_101303"

C2_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C2_PROSPECTIVE_NO_FUTURE_8VEHICLE_MAPPING_VALIDATED"
C3_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C3_MAPPO_INTERFACE_COMPATIBILITY_AUDIT_COMPLETE"
K1_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K1_SKIP_SAFETY_EVIDENCE_AUDIT_COMPLETE"

UPSTREAMS = {
    "PV8-C2": (C2_ROOT, "artifact_manifest_srp2_bis_pv8_c2.json", "_PV8_C2_COMPLETE.lock", C2_GATE),
    "PV8-C3": (C3_ROOT, "artifact_manifest_srp2_bis_pv8_c3.json", "_PV8_C3_COMPLETE.lock", C3_GATE),
    "PV8-K1": (K1_ROOT, "artifact_manifest_srp2_bis_pv8_k1.json", "_PV8_K1_COMPLETE.lock", K1_GATE),
}

SOURCE_PACK_ROOT = ARTIFACTS_ROOT / "suseong_source_pack_v1"
SERVICE_GRAPH_ROOT = ARTIFACTS_ROOT / "suseong_service_graph_v1"
C1_CAPTURE_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_c1_limited_pilot_capture_20260805_070246"
SRP1_R4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp1_r4_postgresql_first_reconciliation_20260803_224422"

ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k2_safety_state_contract"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_K2_SAFETY_STATE_CONTRACT_COMPLETE"
READINESS = "SRP2_BIS_PV8_K2_COMPLETE_SAFETY_STATE_CONTRACT_PARTIAL_REPAIR_FURTHER_DATA_REQUIRED"
FINAL_DECISION = "K_SAFETY_PARTIAL_REPAIR_FURTHER_DATA_REQUIRED"
FIELD_CLASSES = {"OBSERVED", "DERIVED_SAFE", "PROXY_GUARDED", "CONTRACT_FIXED", "NOT_AVAILABLE"}

PAYLOADS = [
    "stop_safety_rulebook_contract.json",
    "stop_safety_rulebook_candidate.parquet",
    "decision_time_service_obligation_contract.json",
    "decision_time_field_source_matrix.json",
    "skip_safety_predicate_contract.json",
    "k2_remaining_evidence_gaps.json",
    "k2_readiness_decision.json",
    "claim_guard_status.json",
    "run_manifest.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.md",
]


class K2Error(RuntimeError):
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

    def parquet(self, rel: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([json_clean(dict(row)) for row in rows], columns=list(columns)).to_parquet(path, index=False)


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


def verify_upstreams() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, (root, manifest, lock, expected_gate) in UPSTREAMS.items():
        gate = read_json(root / "gate_decision.json")
        checks = verify_manifest(root, manifest, lock)
        observed_gate = gate.get("gate") or gate.get("terminal_gate")
        if observed_gate != expected_gate or not manifest_ok(checks):
            raise K2Error(f"{name} integrity failure")
        out[name] = {
            "artifact_root": str(root),
            "gate": observed_gate,
            "readiness": gate.get("readiness"),
            "manifest_integrity": checks,
        }
    return out


def field(name: str, classification: str, source: str, required_for_positive_skip: bool, missing_reason: str = "", safe_use: str = "") -> Dict[str, Any]:
    if classification not in FIELD_CLASSES:
        raise ValueError(f"bad classification: {classification}")
    return {
        "field": name,
        "classification": classification,
        "source": source,
        "required_for_positive_skip": bool(required_for_positive_skip),
        "missing_reason": missing_reason,
        "safe_use": safe_use,
        "missing_is_zero_allowed": False,
    }


def stop_rulebook_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract_name": "PV8_K2_STATIC_STOP_SAFETY_RULEBOOK_CONTRACT_V1",
        "scope": "stop_id-level static safety and operational skip rulebook",
        "primary_key": ["stop_id", "effective_scope"],
        "required_fields": [
            field("stop_id", "OBSERVED", "getBs02 route_stop_sequences / service_nodes", True, safe_use="join static rulebook to route stop sequence"),
            field("mandatory_stop", "NOT_AVAILABLE", "authoritative stop-safety rulebook required", True, "no mandatory stop source currently available"),
            field("protected_stop", "NOT_AVAILABLE", "authoritative stop-safety rulebook required", True, "no protected stop source currently available"),
            field("rule_source", "CONTRACT_FIXED", "K2 contract", True, safe_use="source provenance required for every rule row"),
            field("rule_reason", "CONTRACT_FIXED", "K2 contract", True, safe_use="human-readable reason required for non-null rule values"),
            field("effective_scope", "CONTRACT_FIXED", "K2 contract", True, safe_use="route/direction/stop/all-network scope required"),
            field("evidence_class", "CONTRACT_FIXED", "K2 contract", True, safe_use="OBSERVED/DERIVED_SAFE/PROXY_GUARDED/CONTRACT_FIXED/NOT_AVAILABLE"),
            field("planned_itinerary_allows_skip", "NOT_AVAILABLE", "operation rulebook required", True, "no skip permission table currently available"),
            field("terminal_or_turnaround_stop", "NOT_AVAILABLE", "operation rulebook or terminal source required", True, "no K-safety terminal flag currently available"),
            field("charging_or_driver_relief_stop", "NOT_AVAILABLE", "operation rulebook required", True, "no charging/relief stop source currently available"),
        ],
        "row_validity_rules": [
            "mandatory_stop and protected_stop must be explicit booleans before any positive skip clearance.",
            "NOT_AVAILABLE or PROXY_GUARDED critical rulebook fields force CONDITIONAL_SKIP=false.",
            "rule_source must name the upstream table/file/rulebook; free-text defaulting is forbidden.",
        ],
    }


def rulebook_candidate_rows() -> List[Dict[str, Any]]:
    seq = pd.read_parquet(SOURCE_PACK_ROOT / "route_stop_sequences.parquet")
    cols = ["stop_id", "node_uid", "stop_name", "route_id", "direction_id", "stop_order", "source_classification"]
    available = [c for c in cols if c in seq.columns]
    dedup = seq[available].drop_duplicates().sort_values(["route_id", "direction_id", "stop_order", "stop_id"]).reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for row in dedup.itertuples(index=False):
        item = row._asdict()
        rows.append({
            **item,
            "mandatory_stop": None,
            "protected_stop": None,
            "rule_source": None,
            "rule_reason": "static rulebook source unavailable; value intentionally not filled",
            "effective_scope": "ROUTE_DIRECTION_STOP",
            "evidence_class": "NOT_AVAILABLE",
            "positive_skip_clearance_allowed": False,
        })
    return rows


def decision_time_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "contract_name": "PV8_K2_DECISION_TIME_SERVICE_OBLIGATION_STATE_CONTRACT_V1",
        "scope": "per fixed physical-vehicle agent and decision timestamp",
        "primary_key": ["agent_id", "vehicle_token", "decision_ts"],
        "required_fields": [
            field("agent_id", "OBSERVED", "PV8-C2 fixed agent slots", True, safe_use="stable slot id"),
            field("vehicle_token", "OBSERVED", "PV8-C2 fixed physical vehicle token", True, safe_use="stable physical-vehicle binding"),
            field("decision_ts", "OBSERVED", "PV8-C2 cycle timestamp", True, safe_use="decision-time no-future boundary"),
            field("current_stop_id", "OBSERVED", "getPos02 current_stop_id", True, safe_use="current stop context"),
            field("onboard_destination_obligation", "NOT_AVAILABLE", "APC/AVL/request passenger state required", True, "no onboard destination entity exists"),
            field("assigned_pickup", "NOT_AVAILABLE", "request assignment state required", True, "no request-level pickup assignment source"),
            field("assigned_dropoff", "NOT_AVAILABLE", "request assignment state required", True, "no request-level dropoff assignment source"),
            field("boarding_obligation", "PROXY_GUARDED", "aggregate waiting/boarding demand only", True, "aggregate demand cannot prove exact obligation absence"),
            field("alighting_obligation", "PROXY_GUARDED", "aggregate alighting demand only", True, "aggregate alighting cannot prove onboard destination absence"),
            field("service_obligation", "NOT_AVAILABLE", "logical OR over exact pickup/dropoff/onboard/rulebook obligations", True, "critical components unavailable or proxy-only"),
            field("evidence_class", "CONTRACT_FIXED", "K2 contract", True, safe_use="classification of row-level evidence"),
            field("missing_reason", "CONTRACT_FIXED", "K2 contract", True, safe_use="required whenever evidence_class is NOT_AVAILABLE or PROXY_GUARDED"),
        ],
        "row_validity_rules": [
            "missing passenger/service data must not be interpreted as zero obligation.",
            "service_obligation=false is valid only when all critical components are OBSERVED or DERIVED_SAFE and equal false/zero at decision_ts.",
            "PROXY_GUARDED demand can only block or prioritize review; it cannot clear skip legality.",
        ],
    }


def field_source_matrix(stop_contract: Mapping[str, Any], dynamic_contract: Mapping[str, Any], upstreams: Mapping[str, Any]) -> Dict[str, Any]:
    k1_inventory = read_json(K1_ROOT / "k_safety_field_inventory.json")
    matrix_rows = []
    for section, contract in [("static_stop_rulebook", stop_contract), ("decision_time_service_obligation", dynamic_contract)]:
        for row in contract["required_fields"]:
            matrix_rows.append({
                "section": section,
                **row,
                "positive_skip_clearance_contribution": row["classification"] in {"OBSERVED", "DERIVED_SAFE"} and row["required_for_positive_skip"],
            })
    class_counts: Dict[str, int] = {}
    for row in matrix_rows:
        class_counts[row["classification"]] = class_counts.get(row["classification"], 0) + 1
    return {
        "created_at": iso_kst(),
        "upstream_integrity": upstreams,
        "k1_final_decision": read_json(K1_ROOT / "k_safety_readiness_decision.json")["final_decision"],
        "k1_classification_counts": k1_inventory["classification_counts"],
        "records": matrix_rows,
        "classification_counts": class_counts,
        "aggregate_to_individual_promotion_allowed": False,
        "new_api_collection_count": 0,
        "db_write_count": 0,
    }


def skip_predicate_contract() -> Dict[str, Any]:
    required = [
        "active_bus_mask",
        "next_stop_exists",
        "post_skip_target_exists",
        "graph_edge_or_path_valid",
        "mandatory_stop == false",
        "protected_stop == false",
        "planned_itinerary_allows_skip == true",
        "terminal_or_turnaround_stop == false",
        "charging_or_driver_relief_stop == false",
        "waiting_passenger_demand == 0 with exact evidence",
        "assigned_pickup == 0 with exact evidence",
        "assigned_dropoff == 0 with exact evidence",
        "onboard_destination_obligation == 0 with exact evidence",
        "boarding_obligation == 0 with exact evidence",
        "alighting_obligation == 0 with exact evidence",
        "max_consecutive_skip_constraint_satisfied == true",
        "service_fairness_constraint_satisfied == true",
    ]
    return {
        "created_at": iso_kst(),
        "contract_name": "PV8_K2_FAIL_CLOSED_SKIP_SAFETY_PREDICATE_CONTRACT_V1",
        "action_order": ["HOLD", "SERVE", "CONDITIONAL_SKIP"],
        "conservative_rule": "CONDITIONAL_SKIP=true only if ALL required predicates are positively proven.",
        "unknown_missing_proxy_rule": "unknown / missing / proxy-only critical predicate -> CONDITIONAL_SKIP=false",
        "missing_passenger_data_zero_interpretation_allowed": False,
        "future_leakage_forbidden": True,
        "required_predicates": required,
        "machine_readable_expression": {
            "CONDITIONAL_SKIP": {
                "all": required,
                "all_evidence_class_in": ["OBSERVED", "DERIVED_SAFE"],
                "if_any_evidence_class_in": ["PROXY_GUARDED", "NOT_AVAILABLE"],
                "then": False,
            },
            "SERVE": {
                "all": ["active_bus_mask", "next_stop_exists", "graph_edge_or_path_valid"],
                "passenger_obligation_clearance_required": False,
            },
            "HOLD": {
                "valid_when": "active_bus_mask and not terminal environment guard",
            },
        },
        "conditional_skip_enabled": False,
        "positive_legal_skip_examples_available_now": False,
        "positive_legal_skip_examples_eventually_constructible_without_future_leakage": True,
        "eventual_positive_example_conditions": [
            "all static rulebook predicates are timestamp-valid before decision_ts",
            "decision-time passenger/service obligations are observed or safely derived at decision_ts",
            "no future passenger outcomes, future vehicle survival, or post-decision trajectory evidence is used",
        ],
    }


def remaining_gaps(matrix: Mapping[str, Any]) -> Dict[str, Any]:
    rows = matrix["records"]
    unavailable = [r["field"] for r in rows if r["classification"] == "NOT_AVAILABLE"]
    proxy = [r["field"] for r in rows if r["classification"] == "PROXY_GUARDED"]
    recoverable = [r["field"] for r in rows if r["classification"] in {"OBSERVED", "DERIVED_SAFE", "CONTRACT_FIXED"}]
    return {
        "created_at": iso_kst(),
        "fields_recoverable_now": recoverable,
        "fields_still_unavailable": unavailable,
        "fields_proxy_guarded_only": proxy,
        "minimum_data_required_for_k3": [
            "authoritative stop-safety/rulebook table with mandatory_stop, protected_stop, planned_itinerary_allows_skip, terminal/turnaround, charging/driver-relief, consecutive-skip and fairness predicates",
            "decision-time passenger/request state with onboard_destination_obligation, assigned_pickup, assigned_dropoff, exact waiting/boarding and alighting obligations",
            "source provenance and timestamp/effective-scope columns for every predicate row",
        ],
        "minimum_implementation_required_for_k3": [
            "materialize rulebook join against PV8 fixed route stop sequence",
            "materialize decision-time service-obligation rows keyed by agent_id, vehicle_token, decision_ts",
            "wire fail-closed predicate into action mask without enabling skip when critical predicates are missing/proxy-only",
            "add tests proving missing passenger data is not treated as zero obligation",
        ],
    }


def readiness_decision(gaps: Mapping[str, Any], predicate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "final_decision": FINAL_DECISION,
        "contract_a_static_rulebook_defined": True,
        "contract_b_decision_time_obligation_state_defined": True,
        "conditional_skip_enabled": False,
        "K_action_mask_available": False,
        "positive_legal_skip_examples_available_now": False,
        "positive_legal_skip_examples_eventually_constructible_without_future_leakage": predicate["positive_legal_skip_examples_eventually_constructible_without_future_leakage"],
        "fields_recoverable_now": gaps["fields_recoverable_now"],
        "fields_still_unavailable": gaps["fields_still_unavailable"],
        "fields_proxy_guarded_only": gaps["fields_proxy_guarded_only"],
        "minimum_data_required_for_k3": gaps["minimum_data_required_for_k3"],
        "minimum_implementation_required_for_k3": gaps["minimum_implementation_required_for_k3"],
    }


def claim_guard() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "K_action_mask_available": False,
        "conditional_skip_enabled": False,
        "K_safety_layer_complete": False,
        "safe_skip_decision_ready": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_evaluation_authorized": False,
        "policy_performance_evaluation": False,
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
    writer.text("artifact_manifest_srp2_bis_pv8_k2.jsonl", "".join(json.dumps(json_clean(r), ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    rows.append({
        "relative_path": "artifact_manifest_srp2_bis_pv8_k2.jsonl",
        "size_bytes": (writer.root / "artifact_manifest_srp2_bis_pv8_k2.jsonl").stat().st_size,
        "sha256": sha256_file(writer.root / "artifact_manifest_srp2_bis_pv8_k2.jsonl"),
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
    writer.json("artifact_manifest_srp2_bis_pv8_k2.json", manifest)
    manifest_path = writer.root / "artifact_manifest_srp2_bis_pv8_k2.json"
    writer.json("_PV8_K2_COMPLETE.lock", {
        "artifact_family": ARTIFACT_PREFIX,
        "terminal_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_manifest_path": "artifact_manifest_srp2_bis_pv8_k2.json",
        "final_manifest_sha256": sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
        "created_at": iso_kst(),
    })
    return manifest


def final_report_text(summary: Mapping[str, Any]) -> str:
    return "\n".join([
        "# PV8-K2 Safety State Contract Final Report",
        "",
        f"- artifact_root: `{summary['artifact_root']}`",
        f"- final_decision: `{summary['final_decision']}`",
        f"- fields actually recoverable now: `{summary['fields_recoverable_now']}`",
        f"- fields still unavailable: `{summary['fields_still_unavailable']}`",
        f"- proxy-guarded fields: `{summary['fields_proxy_guarded_only']}`",
        f"- minimum data required for K3: `{summary['minimum_data_required_for_k3']}`",
        f"- minimum implementation required for K3: `{summary['minimum_implementation_required_for_k3']}`",
        f"- positive legal-SKIP examples eventually constructible without future leakage: `{summary['positive_legal_skip_examples_eventually_constructible_without_future_leakage']}`",
        f"- positive legal-SKIP examples available now: `{summary['positive_legal_skip_examples_available_now']}`",
        f"- K_action_mask_available: `{summary['K_action_mask_available']}`",
        f"- conditional_skip_enabled: `{summary['conditional_skip_enabled']}`",
        f"- training_use_authorized: `{summary['training_use_authorized']}`",
        f"- checkpoint_reuse_authorized: `{summary['checkpoint_reuse_authorized']}`",
        f"- policy_evaluation_authorized: `{summary['policy_evaluation_authorized']}`",
        f"- causal_performance_claim_allowed: `{summary['causal_performance_claim_allowed']}`",
        f"- gate: `{summary['gate']}`",
        f"- readiness: `{summary['readiness']}`",
        "",
        "The static and dynamic contracts are now defined, but unavailable and proxy-only critical predicates still force `CONDITIONAL_SKIP=false`. Missing passenger data is never interpreted as zero obligation.",
        "",
        "K3 and retraining were not run.",
        "",
    ])


def run_validate(root: Path) -> Path:
    root = validate_artifact_root(root)
    writer = Writer(root)
    upstreams = verify_upstreams()
    stop_contract = stop_rulebook_contract()
    rows = rulebook_candidate_rows()
    dynamic_contract = decision_time_contract()
    matrix = field_source_matrix(stop_contract, dynamic_contract, upstreams)
    predicate = skip_predicate_contract()
    gaps = remaining_gaps(matrix)
    decision = readiness_decision(gaps, predicate)
    guard = claim_guard()
    gate = {
        "created_at": iso_kst(),
        "gate": PASS_GATE,
        "terminal_gate": PASS_GATE,
        "readiness": READINESS,
        "gate_passed": True,
        "final_decision": decision["final_decision"],
        "failure_reasons": [],
    }
    summary = {
        "artifact_root": str(root),
        "final_decision": decision["final_decision"],
        "fields_recoverable_now": decision["fields_recoverable_now"],
        "fields_still_unavailable": decision["fields_still_unavailable"],
        "fields_proxy_guarded_only": decision["fields_proxy_guarded_only"],
        "minimum_data_required_for_k3": decision["minimum_data_required_for_k3"],
        "minimum_implementation_required_for_k3": decision["minimum_implementation_required_for_k3"],
        "positive_legal_skip_examples_eventually_constructible_without_future_leakage": decision["positive_legal_skip_examples_eventually_constructible_without_future_leakage"],
        "positive_legal_skip_examples_available_now": decision["positive_legal_skip_examples_available_now"],
        "K_action_mask_available": guard["K_action_mask_available"],
        "conditional_skip_enabled": guard["conditional_skip_enabled"],
        "training_use_authorized": guard["training_use_authorized"],
        "checkpoint_reuse_authorized": guard["checkpoint_reuse_authorized"],
        "policy_evaluation_authorized": guard["policy_evaluation_authorized"],
        "causal_performance_claim_allowed": guard["causal_performance_claim_allowed"],
        "gate": gate["gate"],
        "readiness": gate["readiness"],
    }

    writer.json("stop_safety_rulebook_contract.json", stop_contract)
    writer.parquet("stop_safety_rulebook_candidate.parquet", rows, [
        "stop_id",
        "node_uid",
        "stop_name",
        "route_id",
        "direction_id",
        "stop_order",
        "source_classification",
        "mandatory_stop",
        "protected_stop",
        "rule_source",
        "rule_reason",
        "effective_scope",
        "evidence_class",
        "positive_skip_clearance_allowed",
    ])
    writer.json("decision_time_service_obligation_contract.json", dynamic_contract)
    writer.json("decision_time_field_source_matrix.json", matrix)
    writer.json("skip_safety_predicate_contract.json", predicate)
    writer.json("k2_remaining_evidence_gaps.json", gaps)
    writer.json("k2_readiness_decision.json", decision)
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
        "policy_evaluation_count": 0,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "rulebook_candidate_row_count": len(rows),
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {
        "created_at": iso_kst(),
        "source_gate": gate["gate"],
        "readiness": gate["readiness"],
        "final_decision": decision["final_decision"],
        "K_action_mask_available": False,
        "conditional_skip_enabled": False,
        "training_use_authorized": False,
        "checkpoint_reuse_authorized": False,
        "policy_evaluation_authorized": False,
        "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "automatic_k3_execution_authorized": False,
        "automatic_retraining_authorized": False,
    })
    writer.text("final_report.md", final_report_text(summary))
    write_manifest_and_lock(writer, gate)
    own_checks = verify_manifest(root, "artifact_manifest_srp2_bis_pv8_k2.json", "_PV8_K2_COMPLETE.lock")
    if not manifest_ok(own_checks):
        raise K2Error(f"ARTIFACT_INTEGRITY_FAILURE: {own_checks}")

    for key in [
        "artifact_root",
        "final_decision",
        "fields_recoverable_now",
        "fields_still_unavailable",
        "fields_proxy_guarded_only",
        "positive_legal_skip_examples_eventually_constructible_without_future_leakage",
        "positive_legal_skip_examples_available_now",
        "K_action_mask_available",
        "conditional_skip_enabled",
        "training_use_authorized",
        "checkpoint_reuse_authorized",
        "policy_evaluation_authorized",
        "causal_performance_claim_allowed",
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
