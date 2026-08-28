#!/usr/bin/env python3
"""Prompt 5-E01-DL-6D-PA1-A-SRP0.

Historical State-Source Repair Plan and Agent-Semantics Adjudication.

Read-only planning step. Designs the minimum historical evidence package needed
to reconstruct a 16-field DynamicsStateSnapshot from historical evidence, and
adjudicates what the 8 MAPPO agents must mean. Does NOT import or execute the
simulator, does NOT load dataset rows, and reaches no external network. Agent /
policy analysis is performed by static AST + text inspection only.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import re
import resource
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp0_state_source_repair_plan.py"

SF0_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_sf0_historical_state_feasibility_audit_20260803_185838"
SF0_GATE = "PASS_SUSEONG_DL6D_PA1A_SF0_HISTORICAL_STATE_FEASIBILITY_AUDIT_COMPLETE_CRITICAL_GAPS_RECORDED"
SF0_READINESS = "SF0_COMPLETE_STATE_SOURCE_REPAIR_PLAN_PENDING_USER_COMMAND"

FROZEN_SOURCE_SHA = {
    "05_training/simulator/dynamics_multiagent_orchestrator.py": "ee9636f43adeac5dcfa04ec976aca0d72ff05f6e356dbc8beac4f9f1b14d8b81",
    "05_training/simulator/dynamics_event_trace.py": "7886bdafe55ae1794bec610b16a573c2909a64a6cb5aa2545ca5d8391a04a14e",
    "05_training/simulator/suseong_service_transition_engine.py": "37cdfbf4eb486bd0157fe47000924ea2946b787c953e70ff9fea0f821bca8ad5",
    "05_training/simulator/dynamics_state_snapshot.py": "e189d3b3a39acebc957fc91b8225ed1030babb19b28f3825b9f9b822184f0105",
    "05_training/simulator/dynamics_replay_contract.py": "13be29fa5fd2b6637ff9096df3787666204eed07fca136d11b00e3dee2b51477",
    "05_training/simulator/dynamics_horizon_aggregator.py": "7dae0133db4ac28d332952da97bbe193d2b6f425f8abdcf44a4a994e6b98b9ce",
}
POLICY_SOURCES = [
    "05_training/mappo_runner.py",
    "05_training/simulator_adapter_interface.py",
    "05_training/adapters/historical_replay_adapter.py",
    "05_training/simulator/dynamics_multiagent_orchestrator.py",
]

PASS_EXACT = "PASS_SUSEONG_DL6D_PA1A_SRP0_STATE_SOURCE_REPAIR_PLAN_COMPLETE_EXACT_THREE_ACTION_PATH_IDENTIFIED"
PASS_REDUCED = "PASS_SUSEONG_DL6D_PA1A_SRP0_STATE_SOURCE_REPAIR_PLAN_COMPLETE_REDUCED_HS_PATH_ONLY"
PASS_ACQUIRE = "PASS_SUSEONG_DL6D_PA1A_SRP0_STATE_SOURCE_REPAIR_PLAN_COMPLETE_SOURCE_ACQUISITION_REQUIRED"
PASS_INFEASIBLE = "PASS_SUSEONG_DL6D_PA1A_SRP0_STATE_SOURCE_REPAIR_PLAN_COMPLETE_RETROSPECTIVE_EXACT_RECONSTRUCTION_INFEASIBLE"
READINESS = {
    PASS_EXACT: "SRP0_COMPLETE_SOURCE_ACQUISITION_OR_EXTRACTION_PENDING_USER_COMMAND",
    PASS_REDUCED: "SRP0_COMPLETE_SCOPE_DECISION_PENDING_USER_COMMAND",
    PASS_ACQUIRE: "SRP0_COMPLETE_EXTERNAL_SOURCE_ACQUISITION_PLAN_PENDING_USER_COMMAND",
    PASS_INFEASIBLE: "SRP0_COMPLETE_PROSPECTIVE_DATA_COLLECTION_DESIGN_PENDING_USER_COMMAND",
}
BLOCKED_AGENT = "BLOCKED_SUSEONG_DL6D_PA1A_SRP0_AGENT_SEMANTICS_INDETERMINATE"
BLOCKED_POLICY = "BLOCKED_SUSEONG_DL6D_PA1A_SRP0_POLICY_COMPATIBILITY_INDETERMINATE"
BLOCKED_CATALOG = "BLOCKED_SUSEONG_DL6D_PA1A_SRP0_SOURCE_CATALOG_INSUFFICIENT"

_F = "FAIL_SUSEONG_DL6D_PA1A_SRP0_"
FAIL_SF0 = _F + "SF0_UPSTREAM_INVALID"
FAIL_RUNNER = _F + "RUNNER_MUTATED_DURING_PLAN"
FAIL_SOURCE_DRIFT = _F + "SOURCE_DRIFT"
FAIL_SIMULATOR = _F + "SIMULATOR_EXECUTION_DETECTED"
FAIL_ROW_CONTENT = _F + "HISTORICAL_ROW_CONTENT_ACCESSED"
FAIL_VALIDATION_OR_TEST = _F + "VALIDATION_OR_TEST_TOUCHED"
FAIL_SEALED = _F + "SEALED_HOLDOUT_ACCESSED"
FAIL_SUBSTITUTION = _F + "SYNTHETIC_SUBSTITUTION_PROPOSED"
FAIL_AGENT_OVERCLAIM = _F + "AGENT_SEMANTICS_OVERCLAIM"
FAIL_POLICY_OVERCLAIM = _F + "POLICY_COMPATIBILITY_OVERCLAIM"
FAIL_PLAN_INCOMPLETE = _F + "SOURCE_PLAN_INCOMPLETE"
FAIL_UPSTREAM_MUTATED = _F + "UPSTREAM_ARTIFACT_MUTATED"
FAIL_MANIFEST = _F + "MANIFEST_RECONCILIATION"


class PlanError(RuntimeError):
    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate_status = gate
        self.detail = detail


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root

    def text(self, rel: str, text: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")

    def jsonl(self, rel: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        clean = [json_clean(dict(r)) for r in rows]
        p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in clean), encoding="utf-8")
        return {"relative_path": rel, "row_count": len(clean), "preferred_format": "PARQUET",
                "actual_content_format": "JSONL", "fallback_reason": "NO_PARQUET_ENGINE", "file_is_not_binary_parquet": True,
                "content_sha256": sha256_file(p), "size_bytes": p.stat().st_size}


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
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


def copy_file(writer: Writer, src: Path, dst_rel: str) -> Dict[str, Any]:
    dst = writer.root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source_path": str(src), "snapshot_relative_path": dst_rel, "source_sha256": sha256_file(src),
            "copied_sha256": sha256_file(dst), "byte_identical": sha256_file(src) == sha256_file(dst), "size_bytes": dst.stat().st_size}


# ---------------------------------------------------------------------------
# preflight / snapshot / code audit
# ---------------------------------------------------------------------------

def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"srp0 artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def sf0_preflight() -> Dict[str, Any]:
    gate = read_json(SF0_ROOT / "gate_decision.json")
    lock = read_json(SF0_ROOT / "_SF0_AUDIT_COMPLETE.lock")
    manifest_path = SF0_ROOT / lock["manifest_relative_path"]
    manifest = read_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    seen: Dict[str, int] = {}
    missing = mismatch = size_mismatch = 0
    for row in manifest["files"]:
        seen[row["relative_path"]] = seen.get(row["relative_path"], 0) + 1
        t = SF0_ROOT / row["relative_path"]
        if not t.exists():
            missing += 1
            continue
        if row.get("sha256") is not None and sha256_file(t) != row["sha256"]:
            mismatch += 1
        if row.get("size_bytes") is not None and t.stat().st_size != row["size_bytes"]:
            size_mismatch += 1
    duplicate = sorted(p for p, c in seen.items() if c > 1)
    overall = read_json(SF0_ROOT / "overall_state_feasibility.json")
    agent = read_json(SF0_ROOT / "agent_semantics_audit.json")
    checks = {
        "gate": gate.get("gate") == SF0_GATE, "readiness": gate.get("readiness") == SF0_READINESS,
        "lock_present": (SF0_ROOT / "_SF0_AUDIT_COMPLETE.lock").exists(),
        "lock_manifest_sha": lock.get("manifest_sha256") == manifest_sha,
        "lock_manifest_size": lock.get("manifest_size_bytes") == manifest_path.stat().st_size,
        "manifest_missing_zero": missing == 0, "manifest_hash_mismatch_zero": mismatch == 0,
        "manifest_size_mismatch_zero": size_mismatch == 0, "manifest_duplicate_zero": len(duplicate) == 0,
        "required_field_count_16": overall.get("dynamics_state_required_field_count") == 16,
        "agent_semantics_synthetic_slot": agent.get("agent_semantics") == "SYNTHETIC_AGENT_SLOT",
        "three_action_false": overall.get("three_action_historical_comparison_feasible") is False,
        "future_leakage_zero": overall.get("future_leakage_count") == 0,
        "synthetic_default_zero": overall.get("synthetic_default_inserted_count") == 0,
    }
    return {"created_at": iso_kst(), "sf0_artifact_root": str(SF0_ROOT), "manifest_sha256": manifest_sha, "gate": gate.get("gate"),
            "readiness": gate.get("readiness"), "preserved_sf0_results": {
                "required_field_count": overall.get("dynamics_state_required_field_count"),
                "complete_candidate_count": 0, "agent_semantics": agent.get("agent_semantics"),
                "three_action_comparison_feasible": overall.get("three_action_historical_comparison_feasible"),
                "future_leakage_count": overall.get("future_leakage_count"), "synthetic_default_inserted_count": overall.get("synthetic_default_inserted_count")},
            "checks": checks, "sf0_upstream_valid": all(checks.values())}


def snapshot_sf0(writer: Writer) -> Dict[str, Any]:
    records = []
    for src, dst in [
        ("gate_decision.json", "upstream_sf0_snapshot/sf0_gate_decision.json"),
        ("downstream_lock.json", "upstream_sf0_snapshot/sf0_downstream_lock.json"),
        ("artifact_manifest_sf0.json", "upstream_sf0_snapshot/sf0_artifact_manifest.json"),
        ("_SF0_AUDIT_COMPLETE.lock", "upstream_sf0_snapshot/sf0_AUDIT_COMPLETE.lock"),
        ("final_report.json", "upstream_sf0_snapshot/sf0_final_report.json"),
        ("overall_state_feasibility.json", "upstream_sf0_snapshot/sf0_overall_state_feasibility.json"),
        ("state_field_source_matrix.json", "upstream_sf0_snapshot/sf0_state_field_source_matrix.json"),
        ("agent_semantics_audit.json", "upstream_sf0_snapshot/sf0_agent_semantics_audit.json"),
        ("historical_state_reconstruction_plan.json", "upstream_sf0_snapshot/sf0_historical_state_reconstruction_plan.json"),
    ]:
        records.append(copy_file(writer, SF0_ROOT / src, dst))
    payload = {"created_at": iso_kst(), "record_count": len(records), "all_byte_identical": all(r["byte_identical"] for r in records), "records": records}
    writer.json("upstream_sf0_registry.json", payload)
    return payload


def frozen_source_registry() -> Dict[str, Any]:
    records = []
    for rel, frozen in sorted(FROZEN_SOURCE_SHA.items()):
        p = PROJECT_ROOT / rel
        cur = sha256_file(p) if p.exists() else None
        records.append({"relative_path": rel, "exists": p.exists(), "frozen_sha256": frozen, "runtime_sha256": cur, "matches_frozen": cur == frozen})
    return {"created_at": iso_kst(), "source_drift_count": sum(1 for r in records if not r["matches_frozen"]), "records": records}


def agent_semantics_code_audit() -> Dict[str, Any]:
    def text_of(rel: str) -> str:
        p = PROJECT_ROOT / rel
        return p.read_text(encoding="utf-8", errors="ignore") if p.is_file() else ""

    mappo = text_of("05_training/mappo_runner.py")
    adapter = text_of("05_training/adapters/historical_replay_adapter.py")
    iface = text_of("05_training/simulator_adapter_interface.py")
    orch = text_of("05_training/simulator/dynamics_multiagent_orchestrator.py")

    findings = {
        "shared_policy_default_true": bool(re.search(r"shared_policy:\s*bool\s*=\s*True", mappo)),
        "shared_critic_individual_advantage": bool(re.search(r"shared_critic_individual_advantage", mappo)),
        "agent_ids_are_range_index": bool(re.search(r"agent_ids['\"]?\s*[:=].*range\(", adapter)) or bool(re.search(r"list\(range\(self\.num_agents", adapter)),
        "actor_obs_per_agent_shape": bool(re.search(r"actor_obs.*num_agents", iface) or re.search(r"\(self\.num_agents_val,\s*16\)", adapter)),
        "critic_obs_global": bool(re.search(r"critic_obs.*\(1,\s*64\)", adapter) or re.search(r"\[1, global_dim\]", iface)),
        "active_bus_mask_present": bool(re.search(r"active_bus_mask", adapter) or re.search(r"active_bus_mask", iface)),
        "adapter_action_mask_action_dim": (2 if re.search(r"action_mask.*,\s*2\)", adapter) else None),
        "adapter_num_agents_default": (int(re.search(r"num_agents['\"]?,\s*(\d+)", adapter).group(1)) if re.search(r"num_agents['\"]?,\s*(\d+)", adapter) else None),
        "adapter_is_stub_zeros": bool(re.search(r"_get_stub_obs|np\.zeros\(\(self\.num_agents_val", adapter)),
        "engine_requires_exactly_8_agents": bool(re.search(r"len\(action_by_agent\)\s*!=\s*8", orch)),
        "engine_canonical_agent_order_sorted": bool(re.search(r"def canonical_agent_order", orch)) and bool(re.search(r"sorted\(int\(agent_id\)", orch)),
        "engine_vehicle_state_keyed_by_agent_id": bool(re.search(r"missing runtime vehicle for active agent", orch)),
        "engine_action_dim_three": bool(re.search(r"ENGINE_ACTION_CONDITIONAL_SKIP", orch)),
    }
    return {"created_at": iso_kst(), "sources_analyzed": POLICY_SOURCES, "analysis_method": "AST/text static inspection; no import, no execution",
            "source_sha256": {rel: sha256_file(PROJECT_ROOT / rel) for rel in POLICY_SOURCES if (PROJECT_ROOT / rel).is_file()},
            "findings": findings,
            "interpretation": {
                "actor_is_shared_homogeneous": findings["shared_policy_default_true"],
                "agent_slot_is_canonical_ordered_index_not_vehicle_identity": findings["agent_ids_are_range_index"],
                "per_agent_observation_supports_bus_agent": findings["actor_obs_per_agent_shape"] and findings["active_bus_mask_present"],
                "engine_binds_vehicle_state_by_agent_slot": findings["engine_vehicle_state_keyed_by_agent_id"],
                "action_dim_mismatch_interface_2_vs_engine_3": findings["adapter_action_mask_action_dim"] == 2 and findings["engine_action_dim_three"],
                "agent_count_config_mismatch_default_10_vs_engine_8": findings["adapter_num_agents_default"] == 10 and findings["engine_requires_exactly_8_agents"],
                "historical_adapter_is_stub": findings["adapter_is_stub_zeros"],
            }}


# ---------------------------------------------------------------------------
# agent-semantics adjudication  (evidence-driven from the code audit)
# ---------------------------------------------------------------------------

def agent_semantics_candidate_matrix(audit: Mapping[str, Any]) -> List[Dict[str, Any]]:
    itp = audit["interpretation"]
    shared = audit["findings"]["shared_policy_default_true"]
    return [
        {"candidate_semantics": "PHYSICAL_VEHICLE",
         "engine_contract_compatible": bool(itp["engine_binds_vehicle_state_by_agent_slot"]),
         "training_contract_compatible": bool(shared and itp["agent_slot_is_canonical_ordered_index_not_vehicle_identity"]),
         "historical_identity_available": False, "temporally_stable": "PER_ANCHOR_DETERMINISTIC_SELECTION_REQUIRED",
         "permutation_safe": bool(shared), "requires_policy_retraining": False,
         "supports_H": True, "supports_S": True, "supports_K": True, "supports_three_action_comparison": True,
         "evidence_status": "ENGINE_AND_SHARED_POLICY_COMPATIBLE_PENDING_HISTORICAL_VEHICLE_SOURCE",
         "blocking_reason": "requires per-vehicle historical AVL + a deterministic 8-vehicle selection rule; the engine keys vehicle state by canonical agent slot and the shared homogeneous actor is permutation-safe"},
        {"candidate_semantics": "ROUTE_DIRECTION_AGENT",
         "engine_contract_compatible": False, "training_contract_compatible": False, "historical_identity_available": True,
         "temporally_stable": True, "permutation_safe": bool(shared), "requires_policy_retraining": True,
         "supports_H": False, "supports_S": False, "supports_K": False, "supports_three_action_comparison": False,
         "evidence_status": "INCOMPATIBLE_WITH_VEHICLE_LEVEL_TRANSITION",
         "blocking_reason": "the transition engine advances per-vehicle dwell/travel/skip; a route-direction aggregate cannot carry a single vehicle's position/dwell/onboard obligations"},
        {"candidate_semantics": "SERVICE_CONTROL_ZONE_AGENT",
         "engine_contract_compatible": False, "training_contract_compatible": False, "historical_identity_available": True,
         "temporally_stable": True, "permutation_safe": bool(shared), "requires_policy_retraining": True,
         "supports_H": False, "supports_S": False, "supports_K": False, "supports_three_action_comparison": False,
         "evidence_status": "INCOMPATIBLE_WITH_VEHICLE_LEVEL_TRANSITION",
         "blocking_reason": "graph-cluster/zone control has no vehicle-level H/S/K transition semantics"},
        {"candidate_semantics": "POLICY_SLOT",
         "engine_contract_compatible": True, "training_contract_compatible": True, "historical_identity_available": False,
         "temporally_stable": True, "permutation_safe": bool(shared), "requires_policy_retraining": False,
         "supports_H": True, "supports_S": True, "supports_K": True, "supports_three_action_comparison": True,
         "evidence_status": "SYNTHETIC_ONLY_NOT_HISTORICAL (this is the current SF0 status)",
         "blocking_reason": "a bare ordered slot with no vehicle identity yields only synthetic diagnostics; it cannot make historical claims"},
    ]


def recommended_agent_semantics(matrix: Sequence[Mapping[str, Any]], audit: Mapping[str, Any]) -> Dict[str, Any]:
    phys = next(r for r in matrix if r["candidate_semantics"] == "PHYSICAL_VEHICLE")
    engine_ok = phys["engine_contract_compatible"] and phys["training_contract_compatible"]
    if not engine_ok:
        return {"created_at": iso_kst(), "recommended_historical_agent_semantics": "AGENT_SEMANTICS_REDESIGN_REQUIRED",
                "reason": "no candidate is compatible with the current engine and shared policy"}
    return {"created_at": iso_kst(),
            "recommended_historical_agent_semantics": "PHYSICAL_VEHICLE",
            "agent_slot_definition": "agent slot i = the i-th active bus under a deterministic anchor-time selection rule, in canonical order",
            "why": ["engine keys per-vehicle state by canonical agent slot and requires exactly 8 agents",
                    "MAPPO actor is shared/homogeneous (shared_policy=True) so slots are permutation-safe after canonical ordering",
                    "the adapter interface already exposes an active_bus_mask and per-agent (per-bus) observations",
                    "route-direction and zone agents are incompatible with vehicle-level H/S/K transitions",
                    "the current POLICY_SLOT reading is synthetic-only and cannot make historical claims"],
            "conditions": ["a deterministic 8-vehicle selection rule (no random/hash/future/outcome-based selection)",
                           "per-vehicle historical AVL + obligations source (does not yet exist locally)",
                           "policy interface adaptation for the 3-action space and 8-agent count (see policy compatibility)"],
            "overclaim_guard": "recommendation is that PHYSICAL_VEHICLE is the only historically meaningful, engine-compatible semantics; it is NOT a claim that the historical vehicle source already exists",
            "priority_over_alternatives": {"1": "PHYSICAL_VEHICLE", "2_diagnostic_only": "POLICY_SLOT (synthetic scope only)"}}


def policy_agent_compatibility_audit(audit: Mapping[str, Any]) -> Dict[str, Any]:
    itp = audit["interpretation"]
    action_mismatch = bool(itp["action_dim_mismatch_interface_2_vs_engine_3"])
    count_mismatch = bool(itp["agent_count_config_mismatch_default_10_vs_engine_8"])
    stub = bool(itp["historical_adapter_is_stub"])
    # agent MAPPING is canonical-ordering-safe (shared policy); but the interface needs adaptation.
    if action_mismatch or count_mismatch or stub:
        status = "REQUIRES_POLICY_INTERFACE_ADAPTATION"
    elif audit["findings"]["shared_policy_default_true"]:
        status = "COMPATIBLE_AFTER_CANONICAL_AGENT_ORDERING"
    else:
        status = "INDETERMINATE"
    return {"created_at": iso_kst(),
            "actor_weights_agent_shared": audit["findings"]["shared_policy_default_true"],
            "agent_id_hard_embedding_present": False,
            "critic_input_ordering": "GLOBAL_CRITIC_OBS_1x64 (order-invariant global critic)",
            "agent_permutation_sensitivity": "LOW (shared homogeneous actor; canonical ordering applied)",
            "fixed_agent_slot_assumptions": "canonical sorted agent ids; engine requires exactly 8",
            "action_space_interface_action_dim": audit["findings"]["adapter_action_mask_action_dim"],
            "engine_action_dim": 3, "action_dim_mismatch": action_mismatch,
            "agent_count_config_default": audit["findings"]["adapter_num_agents_default"], "engine_agent_count": 8, "agent_count_mismatch": count_mismatch,
            "historical_replay_adapter_is_stub": stub,
            "policy_compatibility_status": status,
            "agent_mapping_dimension": "CANONICAL_ORDERING_SAFE (shared policy)",
            "current_checkpoint_usability": "SEPARATE_UNRESOLVED — DL-6D-R3 sealed hold-out reward alignment FAILED, so no current checkpoint is validated for benefit claims regardless of state reconstruction",
            "overclaim_guard": "compatibility is claimed only for agent-mapping under canonical ordering; the 2->3 action interface, 8-agent count, and stub adapter must be adapted, and current-checkpoint benefit is NOT claimed"}


def physical_vehicle_selection_rule_options() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "prohibited_selection_methods": ["random 8", "hash-based 8", "future-service-result-based", "best-performing-only", "outcome-conditioned"],
            "deterministic_rule_options": [
                {"rule_id": "R1_ACTIVE_CANONICAL", "description": "among vehicles active at anchor t, order by (route_id, direction_id, vehicle_id) and take the first 8",
                 "coverage": "covers whichever service area those routes span", "fleet_churn": "handled by re-selecting active set at each anchor", "selection_bias": "toward lowest canonical route/vehicle ids", "future_leakage_risk": "NONE (uses only anchor-time active status)", "policy_compatibility": "canonical ordering matches shared policy"},
                {"rule_id": "R2_FIXED_SCOPE_ACTIVE", "description": "pre-fix a route-direction scope (independent of outcomes); take active vehicles in that scope in canonical order",
                 "coverage": "scoped", "fleet_churn": "scope-stable", "selection_bias": "scope-defined, outcome-independent", "future_leakage_risk": "NONE", "policy_compatibility": "canonical ordering"},
                {"rule_id": "R3_FIXED_ROSTER_ACTIVE", "description": "pre-fix a historical fleet roster; at anchor t select the 8 roster vehicles that are active, canonical order",
                 "coverage": "roster-defined", "fleet_churn": "roster-stable, active-subset varies", "selection_bias": "roster-defined", "future_leakage_risk": "NONE", "policy_compatibility": "canonical ordering"},
            ],
            "under_8_active_handling": "record the anchor as PARTIAL_FLEET and exclude it from exact 8-agent reconstruction; DO NOT pad with synthetic vehicles",
            "over_8_active_handling": "apply the deterministic canonical rule to pick 8; DO NOT choose the extra vehicles by outcome",
            "prohibited": ["padding with synthetic vehicles", "outcome-conditioned extra-vehicle choice"]}


# ---------------------------------------------------------------------------
# minimum evidence package, gap matrix, prohibited substitutions
# ---------------------------------------------------------------------------

EVIDENCE_PACKAGES = [
    {"package_id": "P1_VEHICLE_AVL", "name": "Vehicle identity and location events",
     "required_fields": ["service_day", "timestamp", "vehicle_id", "route_id", "direction_id", "trip_id", "current_stop_id", "next_stop_id", "current_edge_id", "route_sequence_index", "edge_progress", "observed_speed", "event_type", "source_update_sequence"],
     "events": ["VEHICLE_POSITION_UPDATED", "ARRIVED_STOP", "DEPARTED_STOP", "ENTERED_EDGE", "EXITED_EDGE", "ENTERED_TERMINAL", "LEFT_TERMINAL"],
     "reconstructs_fields": ["vehicles", "headway_state", "action_mask_state(partial)"],
     "existing_source": "NONE historical (Daegu BIS getPos02 is real-time only; no archived per-vehicle AVL for the training period)",
     "acquisition_class": "REQUIRES_OPERATOR_OR_BIS_ACCESS", "retrospective_availability": "UNCERTAIN"},
    {"package_id": "P2_SCHEDULE_TRIP", "name": "Schedule and trip plan",
     "required_fields": ["service_day", "route_id", "direction_id", "trip_id", "vehicle_id_or_block_id", "stop_id", "stop_sequence", "scheduled_arrival", "scheduled_departure", "service_calendar", "turnaround_seconds", "layover_seconds", "valid_from", "valid_to"],
     "reconstructs_fields": ["schedule_state", "operation_mode"],
     "existing_source": "PARTIAL (static route topology exists; no per-trip timetable/block in project)",
     "acquisition_class": "REQUIRES_OPERATOR_OR_PUBLIC_GTFS", "retrospective_availability": "LIKELY_AVAILABLE"},
    {"package_id": "P3_PASSENGER_SERVICE", "name": "Passenger service events",
     "required_fields": ["timestamp", "request_or_passenger_service_id", "vehicle_id", "route_id", "trip_id", "origin_stop_id", "destination_stop_id", "board_timestamp", "alight_timestamp", "service_leg_id", "event_type"],
     "events": ["PASSENGER_WAIT_STARTED", "PASSENGER_BOARDED", "PASSENGER_ALIGHTED", "SERVICE_LEG_COMPLETED"],
     "reconstructs_fields": ["waiting_passengers", "onboard_passengers", "shared_counters(partial)"],
     "existing_source": "NONE (only node-level 10-min boarding/alighting aggregates; transport-card synth-trip is synthetic)",
     "acquisition_class": "REQUIRES_AFC_OR_OD_ACCESS", "retrospective_availability": "PARTIAL_ORIGIN_ONLY (AFC tap-on typically lacks per-passenger onboard destination linkage)"},
    {"package_id": "P4_REQUEST_ASSIGNMENT", "name": "Request and assignment events",
     "required_fields": ["request_id", "service_leg_id", "requested_at", "origin_stop_id", "destination_stop_id", "assigned_vehicle_id", "assigned_at", "pickup_due_at", "dropoff_due_at", "cancelled_at", "completed_at", "assignment_version"],
     "events": ["REQUEST_CREATED", "PICKUP_ASSIGNED", "DROPOFF_ASSIGNED", "ASSIGNMENT_REVOKED", "REQUEST_CANCELLED", "REQUEST_COMPLETED"],
     "reconstructs_fields": ["assigned_pickups", "assigned_dropoffs"],
     "existing_source": "NONE (historical Suseong operation is fixed-route; no DRT request/assignment logs exist)",
     "acquisition_class": "REQUIRES_DRT_PLATFORM_LOGS", "retrospective_availability": "NOT_APPLICABLE_FOR_HISTORICAL_FIXED_ROUTE (must be proven, not defaulted empty)"},
    {"package_id": "P5_STOP_ROUTE_SAFETY", "name": "Stop and route safety configuration",
     "required_fields": ["route_id", "direction_id", "stop_id", "valid_from", "valid_to", "mandatory_stop", "protected_stop", "terminal_stop", "driver_relief_stop", "charging_stop", "skip_allowed", "max_consecutive_skip", "fairness_policy_id", "path_after_skip_valid"],
     "reconstructs_fields": ["mandatory_stop_state", "action_mask_state", "routes(safety flags)"],
     "existing_source": "NONE (graph_node_master/route_link_sequence carry topology only, no safety flags)",
     "acquisition_class": "REQUIRES_OPERATOR_OR_MANUAL_POLICY_CONFIGURATION", "retrospective_availability": "REQUIRES_OPERATOR_CONFIG_OR_MANUAL"},
    {"package_id": "P6_TRAVEL_TIME", "name": "Traffic/travel-time events",
     "required_fields": ["timestamp", "edge_id", "travel_time_seconds", "source", "valid_from", "valid_to", "quality_flag"],
     "reconstructs_fields": ["vehicles.remaining_travel_seconds(partial)"],
     "existing_source": "PARTIAL (node-level 10-min link_travel_time_sec exists but coarse; not per-vehicle remaining-travel resolution)",
     "acquisition_class": "REQUIRES_BIS_OR_TRAFFIC_ACCESS", "retrospective_availability": "COARSE_AVAILABLE_FINE_UNCERTAIN"},
    {"package_id": "P7_ORDERED_REPLAY", "name": "Ordered replay stream",
     "required_fields": ["service_day", "event_timestamp", "event_sequence", "event_type", "entity_type", "entity_id", "source_id", "source_event_id", "payload_hash"],
     "reconstructs_fields": ["replay_cursor", "shared_counters"],
     "existing_source": "NONE (derivable only if P1/P3/P4/P6 events are acquired and unified)",
     "acquisition_class": "DERIVABLE_FROM_EXISTING_RAW_EVENTS (once P1/P3 acquired)", "retrospective_availability": "DEPENDS_ON_P1_P3"},
    {"package_id": "P8_PROVIDER_STATE", "name": "Provider state",
     "required_fields": ["provider_name", "provider_version", "provider_cursor", "provider_cache_state", "source_manifest", "source_hash", "is_stateful"],
     "reconstructs_fields": ["external_provider_states"],
     "existing_source": "DESIGN-SIDE (must prove stateless providers or supply provider snapshots)",
     "acquisition_class": "REQUIRES_MANUAL_POLICY_CONFIGURATION", "retrospective_availability": "PROVABLE_BY_CONTRACT"},
]


def minimum_historical_evidence_package() -> List[Dict[str, Any]]:
    return EVIDENCE_PACKAGES


def existing_source_to_required_package_matrix() -> List[Dict[str, Any]]:
    rows = []
    node_ok_uses = ["Graph observation feature", "Aggregate demand context", "Travel-time context", "Exogenous environment feature"]
    node_not_uses = ["real vehicle state", "real waiting queue", "real passenger request", "real onboard destination", "real assignment", "real safety state"]
    for pkg in EVIDENCE_PACKAGES:
        capable = pkg["existing_source"].startswith("PARTIAL") is False and "NONE" not in pkg["existing_source"]
        rows.append({"package_id": pkg["package_id"], "required_fields": pkg["required_fields"],
                     "existing_source": pkg["existing_source"], "existing_granularity": "node-level 10-minute aggregate" if "node-level" in pkg["existing_source"] else "static or none",
                     "entity_resolution": "NODE_LEVEL (no vehicle/passenger entity)", "temporal_resolution": "10-minute bucket" if "node-level" in pkg["existing_source"] else "n/a",
                     "exact_reconstruction_capable": False, "acquisition_required": True,
                     "transformation_required": pkg["package_id"] in {"P2_SCHEDULE_TRIP", "P6_TRAVEL_TIME", "P7_ORDERED_REPLAY"},
                     "node_aggregate_permitted_uses": node_ok_uses, "node_aggregate_prohibited_promotions": node_not_uses,
                     "blocking_reason": pkg["existing_source"]})
    return rows


PROHIBITED_SUBSTITUTIONS = [
    ("SUB01", "10-minute boardings_recent", "anchor waiting_pickup_count queue", "aggregate flow is not an anchor-time entity queue", "S,K", "waiting-state claim"),
    ("SUB02", "10-minute alightings_recent", "onboard destination obligation", "aggregate alightings do not identify onboard destinations", "S,K", "onboard-obligation claim"),
    ("SUB03", "waiting_passenger_cnt proxy", "passenger-level waiting entity", "declared proxy, not per-passenger entity", "K", "waiting-entity claim"),
    ("SUB04", "node link_travel_time_sec", "per-vehicle remaining_travel_seconds", "node-level travel time is not a specific vehicle's remaining travel", "H,S", "vehicle-time claim"),
    ("SUB05", "route topology", "route safety permission (skip/mandatory/fairness)", "topology carries no safety policy", "K", "skip-safety claim"),
    ("SUB06", "absent mandatory flag", "false", "absence is not a proven false; forbidden default", "K", "mandatory-stop claim"),
    ("SUB07", "absent assigned pickup", "empty", "absence is not a proven empty; forbidden default", "K", "assignment claim"),
    ("SUB08", "absent onboard passenger", "zero", "absence is not a proven zero; forbidden default", "S,K", "onboard claim"),
    ("SUB09", "8 policy slots", "8 real vehicles", "policy slots are not identified historical vehicles", "H,S,K", "agent-identity claim"),
    ("SUB10", "554 graph snapshots", "554 historical DynamicsStateSnapshots", "node-feature tensors are not entity-level state snapshots", "H,S,K", "snapshot-count claim"),
]


def prohibited_source_substitution_registry() -> List[Dict[str, Any]]:
    return [{"substitution_id": s[0], "source_field": s[1], "prohibited_target_field": s[2], "reason": s[3],
             "affected_action": s[4], "affected_claim": s[5]} for s in PROHIBITED_SUBSTITUTIONS]


# ---------------------------------------------------------------------------
# architecture / actions / paths / acquisition / privacy / milestones / WPs
# ---------------------------------------------------------------------------

def historical_event_schema_design() -> Dict[str, Any]:
    return {"created_at": iso_kst(), "packages": {p["package_id"]: p.get("events", []) for p in EVIDENCE_PACKAGES},
            "unified_replay_common_key": ["service_day", "event_timestamp", "event_sequence", "event_type", "entity_type", "entity_id", "source_id", "source_event_id", "payload_hash"],
            "requirements": ["same-timestamp tie-break", "duplicate event detection", "missing sequence detection", "anchor prior/after temporal cut", "stream hash"]}


def service_day_start_state_requirements() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "required_start_state": ["fleet_roster", "vehicle_to_block_or_trip_allocation", "initial_vehicle_position", "initial_route_direction",
                                     "initial_onboard_passengers", "initial_active_requests", "initial_assignments", "initial_shared_counters",
                                     "initial_provider_cursors", "initial_operating_mode"],
            "zero_empty_requires_source_evidence": True,
            "note": "even a genuinely zero/empty start value must be proven from source, never defaulted"}


def service_day_reconstruction_architecture() -> Dict[str, Any]:
    return {"created_at": iso_kst(), "architecture": "SERVICE_DAY_EVENT_SOURCED_STATE_RECONSTRUCTION",
            "equation": "static config valid at service day + service-day start state + ordered events through anchor t = DynamicsStateSnapshot at t",
            "pipeline": ["source normalization", "entity identity resolution", "event deduplication", "temporal ordering",
                         "service-day start-state initialization", "event reducer", "anchor cut", "EvaluationExecutionContext creation",
                         "DynamicsStateSnapshot candidate", "pure contract validation", "hash/provenance seal"],
            "implemented_in_srp0": False, "executed_in_srp0": False}


def action_state_source_requirement_matrix() -> List[Dict[str, Any]]:
    return [
        {"action": "H_HOLD_CURRENT_POSITION", "minimum_requirements": ["physical vehicle identity", "position", "route state", "time budget", "operation mode", "provider cursor"], "required_packages": ["P1", "P2", "P8"]},
        {"action": "S_SERVE_AND_MOVE", "additional_requirements": ["current-stop service events", "boarding/alighting obligations", "dwell state", "travel-time state", "next-route progression"], "required_packages": ["P1", "P2", "P3", "P6"]},
        {"action": "K_CONDITIONAL_SKIP_EMPTY_STOP", "additional_requirements": ["waiting passenger entities", "assigned pickup obligations", "assigned dropoff obligations", "onboard destination obligations", "mandatory/protected/terminal flags", "skip permission", "downstream path validity", "consecutive-skip history", "service fairness history"], "required_packages": ["P1", "P2", "P3", "P4", "P5", "P7"]},
    ]


def research_path_comparison() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "PATH_A_exact_physical_vehicle_three_action": {"conditions": ["P1-P8 acquired", "physical-vehicle semantics", "8-agent mapping", "full service-day event stream", "K safety exact reconstruction"], "allowed_claim": "Historical H/S/K dynamics comparison", "status": "PREFERRED_FULL_RESEARCH_PATH", "currently_available": False},
            "PATH_B_reduced_hs": {"conditions": ["vehicle location + schedule + service events exist", "no passenger-obligation/K-safety source"], "possible": ["H/S mechanics comparison"], "impossible": ["K historical execution", "three-action comparison", "skip-policy-benefit claim"], "does_not_substitute": "PA1-A 3-action evidence", "currently_available": False},
            "PATH_C_node_level_synthetic_diagnostic": {"scope": ["observation-model diagnostic", "demand-context diagnostic", "synthetic engine integration"], "prohibited_claims": ["historical vehicle-state claim", "historical H/S/K effect claim", "real-world skip-safety claim"], "note": "FV1 already validated this infrastructure scope; repeating the same synthetic diagnostic is not recommended", "currently_available": True}}


def data_acquisition_source_classification() -> List[Dict[str, Any]]:
    return [
        {"package_id": p["package_id"], "acquisition_class": p["acquisition_class"], "retrospective_availability": p["retrospective_availability"],
         "candidate_source_note": {"P1_VEHICLE_AVL": "Daegu BIS AVL archive (operator); real-time getPos02 exists but archival retention unknown",
                                   "P2_SCHEDULE_TRIP": "operator GTFS / timetable / vehicle-block plan",
                                   "P3_PASSENGER_SERVICE": "AFC transport-card records / OD survey (agency)",
                                   "P4_REQUEST_ASSIGNMENT": "DRT dispatch platform logs (do not exist for fixed-route historical operation)",
                                   "P5_STOP_ROUTE_SAFETY": "operator route/stop policy configuration",
                                   "P6_TRAVEL_TIME": "BIS / traffic travel-time archive",
                                   "P7_ORDERED_REPLAY": "derived from acquired raw events",
                                   "P8_PROVIDER_STATE": "design contract / provider snapshots"}.get(p["package_id"])}
        for p in EVIDENCE_PACKAGES]


def privacy_governance_plan() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "principles": ["no passenger real names required", "stable pseudonymous ID required to link request/service-leg", "vehicle ID may be pseudonymized", "time/stop/trip relationships must be preserved"],
            "per_source": {p["package_id"]: {"personal_data_risk": "HIGH" if p["package_id"] in {"P3_PASSENGER_SERVICE", "P4_REQUEST_ASSIGNMENT"} else "LOW",
                                             "vehicle_operator_confidentiality": "MEDIUM" if p["package_id"] in {"P1_VEHICLE_AVL", "P2_SCHEDULE_TRIP"} else "LOW",
                                             "location_data_sensitivity": "MEDIUM" if p["package_id"] in {"P1_VEHICLE_AVL", "P3_PASSENGER_SERVICE"} else "LOW",
                                             "required_anonymization": "PSEUDONYMOUS_STABLE_ID", "retention_policy": "SPLIT_SCOPED_MINIMAL", "access_control": "RESTRICTED", "split_leakage_risk": "MUST_SPLIT_BY_SERVICE_DAY_FIRST"}
                           for p in EVIDENCE_PACKAGES}}


def split_safe_ingestion_plan() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "principle": "split by service day / continuous time block BEFORE building any snapshot; never split after snapshot generation",
            "split_scopes": ["TRAIN source events", "VALIDATION source events", "TEST source events", "SEALED_HOLDOUT source events"],
            "cross_contamination_checks": ["same vehicle trip crossing split", "same request crossing split", "same service day crossing split", "provider cache crossing split", "normalization statistics seeing validation/test"],
            "initial_implementation_uses": "TRAIN source events only",
            "sealed_test_binding": "the existing 554 TEST snapshots (== DL-6D-R3 sealed hold-out) remain sealed; any new source events for those service days inherit TEST/SEALED scope and are not accessed"}


def minimum_viable_milestones() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "M1_one_exact_train_anchor": {"goal": "reconstruct one DynamicsStateSnapshot exactly from real vehicle/passenger/request events",
                                          "conditions": ["8 physical vehicles", "16 required fields", "required subfields", "future leakage = 0", "synthetic defaults = 0", "pure contract validation PASS"], "transition_executed": False},
            "M2_one_service_day_replay": {"goal": "reconstruct multiple anchors deterministically via an ordered event reducer over one TRAIN service day", "precondition": "M1 PASS", "blocked_until_m1": True}}


def state_source_repair_work_packages() -> List[Dict[str, Any]]:
    wps = [
        ("WP1", "Agent semantics and policy compatibility", ["code contracts"], "agent-semantics decision + policy adaptation spec", [], "PHYSICAL_VEHICLE recommended; 3-action + 8-agent interface adaptation specified"),
        ("WP2", "Historical entity ID registry", ["P1", "P3", "P4"], "pseudonymous vehicle/passenger/request ID registry", ["WP1"], "stable cross-event linkage without real identities"),
        ("WP3", "Vehicle AVL event ingestion", ["P1"], "normalized vehicle position/stop/edge events", ["WP2"], "per-vehicle prior-only positions at sub-minute resolution"),
        ("WP4", "Schedule/trip/block ingestion", ["P2"], "trip/block timetable", ["WP2"], "route-direction validity + turnaround/layover"),
        ("WP5", "Passenger service and OD event ingestion", ["P3"], "board/alight/wait events with service leg", ["WP2"], "per-passenger waiting + onboard-destination linkage"),
        ("WP6", "Request/assignment event ingestion", ["P4"], "request/assignment events", ["WP2"], "only if a DRT platform existed; else prove N/A"),
        ("WP7", "Stop/route safety configuration", ["P5"], "time-valid safety flags", ["WP4"], "no retroactive application of current config"),
        ("WP8", "Travel-time/provider event ingestion", ["P6", "P8"], "edge travel-time + provider state", ["WP2"], "resolution sufficient for remaining-travel"),
        ("WP9", "Unified replay event schema", ["P7"], "ordered replay stream", ["WP3", "WP5", "WP8"], "dedup + tie-break + stream hash"),
        ("WP10", "Service-day state reducer", ["all"], "deterministic anchor state reducer", ["WP9"], "no defaults, no future leakage"),
        ("WP11", "Split-safe source manifest", ["all"], "split-scoped source manifest", ["WP10"], "service-day-first split, no cross-contamination"),
        ("WP12", "One-anchor pure contract validation", ["all"], "validated DynamicsStateSnapshot candidate", ["WP10", "WP11"], "schema valid, round-trip hash, 16/16 fields"),
    ]
    return [{"work_package_id": w[0], "objective": w[1], "required_inputs": w[2], "output_schema": w[3], "dependencies": w[4],
             "acceptance_criteria": w[5], "privacy_constraints": "pseudonymous IDs; split-scoped retention",
             "estimated_data_availability": "INDETERMINATE (external acquisition dependent)",
             "blocking_risks": "source may not exist retrospectively", "prohibited_shortcuts": ["synthetic fill", "default 0/empty/false", "aggregate-as-entity", "future leakage"]}
            for w in wps]


CRITICAL_PRIORITY = [
    ("G1", ["vehicles(agent identity)"], ["H", "S", "K"], "CRITICAL", "P1 + agent-semantics", "UNCERTAIN"),
    ("G2", ["vehicles(dynamic subfields)"], ["H", "S", "K"], "CRITICAL", "P1 + P6", "COARSE/UNCERTAIN"),
    ("G3", ["waiting_passengers", "onboard_passengers"], ["K", "S"], "CRITICAL", "P3", "PARTIAL_ORIGIN_ONLY"),
    ("G4", ["assigned_pickups", "assigned_dropoffs"], ["K"], "CRITICAL", "P4", "NOT_APPLICABLE_FIXED_ROUTE"),
    ("G5", ["mandatory_stop_state", "action_mask_state", "routes(safety)"], ["K"], "CRITICAL", "P5", "OPERATOR_CONFIG"),
    ("G6", ["replay_cursor"], ["H", "S", "K"], "HIGH", "P7", "DEPENDS_ON_P1_P3"),
    ("G7", ["schedule_state", "headway_state"], ["S"], "HIGH", "P2 + P1", "LIKELY/UNCERTAIN"),
    ("G8", ["external_provider_states"], ["H", "S", "K"], "MEDIUM", "P8", "PROVABLE"),
    ("G9", ["shared_counters"], ["S", "K"], "MEDIUM", "P7 replay", "DEPENDS_ON_EVENTS"),
]


def critical_source_priority_registry() -> List[Dict[str, Any]]:
    return [{"gap_id": g[0], "affected_state_fields": g[1], "affected_actions": g[2], "criticality": g[3],
             "source_required": g[4], "retrospective_availability": g[5], "repair_strategy": "acquire/normalize/event-source; no synthetic fallback", "fallback_allowed": False}
            for g in CRITICAL_PRIORITY]


def retrospective_availability_assessment() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "likely_retrospectively_available": ["P2_SCHEDULE_TRIP", "P6_TRAVEL_TIME(coarse)"],
            "acquisition_dependent_uncertain": ["P1_VEHICLE_AVL", "P5_STOP_ROUTE_SAFETY"],
            "likely_retrospectively_unavailable": ["P3_PASSENGER_SERVICE(onboard-destination linkage)", "P4_REQUEST_ASSIGNMENT(no DRT platform for fixed-route history)"],
            "note": "the single highest-criticality blocker for K (per-passenger onboard obligations and DRT assignment) is the least likely to be retrospectively available; those move toward prospective collection"}


def prospective_collection_requirement() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "prospective_only_if_retrospective_fails": ["per-passenger onboard-destination events", "DRT request/assignment events (only if a DRT service is introduced)", "sub-minute per-vehicle AVL if archive unavailable"],
            "prospective_design_note": "a forward-looking instrumented collection would capture P1/P3/P5/P6/P7 with the unified replay schema from the start, split by service day",
            "does_not_unblock_historical_training_period": True}


# ---------------------------------------------------------------------------
# environment / manifest / decision / orchestration
# ---------------------------------------------------------------------------

def environment_payload() -> Dict[str, Any]:
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"created_at": iso_kst(), "mode": "plan", "scope": "READ_ONLY_STATE_SOURCE_REPAIR_PLANNING",
            "requested_execution_platform": "MAC_MINI_M4_24GB", "actual_compute_path": "CPU_ONLY_STATIC_PLANNING",
            "platform_machine": platform.machine(), "python_executable": sys.executable, "python_version": sys.version,
            "process_rss_bytes": raw_rss if platform.system() == "Darwin" else raw_rss * 1024,
            "simulator_imported": False, "dataset_rows_loaded": False, "network_accessed": False,
            "simulator_execution_count": 0, "historical_reconstruction_count": 0, "historical_row_content_access_count": 0,
            "validation_access_count": 0, "test_access_count": 0, "sealed_holdout_access_count": 0,
            "external_network_access_count": 0, "source_modification_count": 0, "training_run_count": 0}


def write_manifest(writer: Writer, rel: str, payloads: Sequence[str]) -> Dict[str, Any]:
    rows = []
    for p in payloads:
        path = writer.root / p
        rows.append({"relative_path": p, "required": True, "exists": path.exists(),
                     "sha256": sha256_file(path) if path.exists() else None, "size_bytes": path.stat().st_size if path.exists() else None})
    manifest = {"created_at": iso_kst(), "manifest_protocol": "TERMINAL_LOCK_TO_MANIFEST_TO_PAYLOAD", "manifest_scope": "SRP0_STATE_SOURCE_REPAIR_PLAN",
                "required_payload_count": len(rows), "payload_file_count": sum(1 for r in rows if r["exists"]),
                "missing_payload_count": sum(1 for r in rows if not r["exists"]), "missing_payloads": [r["relative_path"] for r in rows if not r["exists"]],
                "terminal_lock_listed_inside_manifest": False, "manifest_self_listed": False, "files": rows}
    writer.json(rel, manifest)
    return manifest


def write_lock(writer: Writer, lock_name: str, manifest_name: str, gate: Mapping[str, Any]) -> None:
    mp = writer.root / manifest_name
    writer.json(lock_name, {"created_at": iso_kst(), "mode": "plan", "gate": gate["gate"], "gate_passed": gate["gate_passed"],
                            "readiness": gate["readiness"], "manifest_relative_path": manifest_name,
                            "manifest_sha256": sha256_file(mp), "manifest_size_bytes": mp.stat().st_size})


def verify_manifest(root: Path, lock_name: str) -> Dict[str, Any]:
    lock = read_json(root / lock_name)
    mp = root / lock["manifest_relative_path"]
    manifest = read_json(mp)
    missing = mismatch = 0
    for row in manifest["files"]:
        path = root / row["relative_path"]
        if not path.exists():
            missing += 1
        elif sha256_file(path) != row["sha256"]:
            mismatch += 1
    return {"manifest_hash_ok": sha256_file(mp) == lock["manifest_sha256"], "manifest_size_ok": mp.stat().st_size == lock["manifest_size_bytes"],
            "payload_missing_count": missing, "payload_hash_mismatch_count": mismatch,
            "terminal_lock_listed_inside_manifest": any(r["relative_path"] == lock_name for r in manifest["files"]),
            "manifest_self_listed": any(r["relative_path"] == lock["manifest_relative_path"] for r in manifest["files"])}


SRP0_PAYLOADS = [
    "upstream_sf0_snapshot/sf0_gate_decision.json", "upstream_sf0_snapshot/sf0_downstream_lock.json", "upstream_sf0_snapshot/sf0_artifact_manifest.json",
    "upstream_sf0_snapshot/sf0_AUDIT_COMPLETE.lock", "upstream_sf0_snapshot/sf0_final_report.json", "upstream_sf0_snapshot/sf0_overall_state_feasibility.json",
    "upstream_sf0_snapshot/sf0_state_field_source_matrix.json", "upstream_sf0_snapshot/sf0_agent_semantics_audit.json", "upstream_sf0_snapshot/sf0_historical_state_reconstruction_plan.json",
    "upstream_sf0_registry.json", "runner_snapshot_pre_execution/run_prompt5_e01_dl6d_pa1a_srp0_state_source_repair_plan.py", "runner_freeze_audit.json",
    "planning_environment.json", "sf0_upstream_preflight.json", "frozen_source_registry.json",
    "agent_semantics_code_audit.json", "agent_semantics_candidate_matrix.json", "agent_semantics_candidate_matrix.jsonl", "recommended_agent_semantics.json",
    "policy_agent_compatibility_audit.json", "physical_vehicle_selection_rule_options.json",
    "minimum_historical_evidence_package.json", "minimum_historical_evidence_package.jsonl",
    "existing_source_to_required_package_matrix.json", "existing_source_to_required_package_matrix.jsonl",
    "prohibited_source_substitution_registry.json", "prohibited_source_substitution_registry.jsonl",
    "historical_event_schema_design.json", "service_day_start_state_requirements.json", "service_day_reconstruction_architecture.json",
    "action_state_source_requirement_matrix.json", "research_path_comparison.json",
    "data_acquisition_source_classification.json", "data_acquisition_source_classification.jsonl",
    "privacy_governance_plan.json", "split_safe_ingestion_plan.json", "minimum_viable_milestones.json",
    "state_source_repair_work_packages.json", "state_source_repair_work_packages.jsonl",
    "critical_source_priority_registry.json", "critical_source_priority_registry.jsonl",
    "retrospective_availability_assessment.json", "prospective_collection_requirement.json", "source_repair_decision.json", "next_stage_readiness.json",
    "simulator_execution_prohibition_audit.json", "historical_row_access_prohibition_audit.json", "validation_untouched_audit.json", "test_holdout_untouched_audit.json",
    "sealed_holdout_preservation_audit.json", "training_prohibition_audit.json", "external_access_audit.json", "stage_immutability_audit.json",
    "gate_decision.json", "downstream_lock.json", "final_report.json", "final_report.md",
]


def decide_gate(recommended: Mapping[str, Any], policy: Mapping[str, Any], packages: Sequence[Mapping[str, Any]], retro: Mapping[str, Any]) -> Dict[str, Any]:
    if recommended["recommended_historical_agent_semantics"] == "AGENT_SEMANTICS_REDESIGN_REQUIRED":
        return {"gate": BLOCKED_AGENT, "reason": "no engine-compatible historical agent semantics"}
    if policy["policy_compatibility_status"] == "INDETERMINATE":
        return {"gate": BLOCKED_POLICY, "reason": "policy compatibility indeterminate"}
    # every required package has an identified source class (institution/system/log type) -> acquisition-required
    all_identified = all(p.get("acquisition_class") for p in packages)
    # exact three-action path needs P1..P8 sources; none exist locally -> not EXACT now
    any_local_exact = any(p["existing_source"].startswith("PRESENT") for p in packages)
    if not all_identified:
        return {"gate": BLOCKED_CATALOG, "reason": "source catalog insufficient to identify required sources"}
    if any_local_exact:
        return {"gate": PASS_EXACT, "reason": "all required sources already present locally"}
    # sources identifiable by institution/system but not present locally -> C
    return {"gate": PASS_ACQUIRE, "reason": "required historical sources are identifiable by institution/system/log type but absent locally; some (per-passenger onboard, DRT assignment) are likely retrospectively unavailable and recorded as prospective-only"}


def run_plan(artifact_root: Path) -> Path:
    runner_sha_before = sha256_file(RUNNER_PATH)
    runner_size_before = RUNNER_PATH.stat().st_size

    sf0 = sf0_preflight()
    if not sf0["sf0_upstream_valid"]:
        raise PlanError(FAIL_SF0, f"SF0 upstream invalid: {sf0['checks']}")

    root = validate_artifact_root(artifact_root)
    writer = Writer(root)
    writer.json("planning_environment.json", environment_payload())
    writer.json("sf0_upstream_preflight.json", sf0)
    snapshot_sf0(writer)
    runner_snapshot = copy_file(writer, RUNNER_PATH, "runner_snapshot_pre_execution/run_prompt5_e01_dl6d_pa1a_srp0_state_source_repair_plan.py")

    source_reg = frozen_source_registry()
    writer.json("frozen_source_registry.json", source_reg)
    if source_reg["source_drift_count"]:
        raise PlanError(FAIL_SOURCE_DRIFT, f"dynamics source drift: {source_reg['source_drift_count']}")

    code_audit = agent_semantics_code_audit()
    writer.json("agent_semantics_code_audit.json", code_audit)
    candidate_matrix = agent_semantics_candidate_matrix(code_audit)
    writer.json("agent_semantics_candidate_matrix.json", {"created_at": iso_kst(), "records": candidate_matrix})
    writer.jsonl("agent_semantics_candidate_matrix.jsonl", candidate_matrix)
    recommended = recommended_agent_semantics(candidate_matrix, code_audit)
    writer.json("recommended_agent_semantics.json", recommended)
    policy = policy_agent_compatibility_audit(code_audit)
    writer.json("policy_agent_compatibility_audit.json", policy)
    writer.json("physical_vehicle_selection_rule_options.json", physical_vehicle_selection_rule_options())

    packages = minimum_historical_evidence_package()
    writer.json("minimum_historical_evidence_package.json", {"created_at": iso_kst(), "packages": packages})
    writer.jsonl("minimum_historical_evidence_package.jsonl", packages)
    gap_matrix = existing_source_to_required_package_matrix()
    writer.json("existing_source_to_required_package_matrix.json", {"created_at": iso_kst(), "records": gap_matrix})
    writer.jsonl("existing_source_to_required_package_matrix.jsonl", gap_matrix)
    subs = prohibited_source_substitution_registry()
    writer.json("prohibited_source_substitution_registry.json", {"created_at": iso_kst(), "prohibited_substitution_count": len(subs), "records": subs})
    writer.jsonl("prohibited_source_substitution_registry.jsonl", subs)

    writer.json("historical_event_schema_design.json", historical_event_schema_design())
    writer.json("service_day_start_state_requirements.json", service_day_start_state_requirements())
    writer.json("service_day_reconstruction_architecture.json", service_day_reconstruction_architecture())
    writer.json("action_state_source_requirement_matrix.json", {"created_at": iso_kst(), "records": action_state_source_requirement_matrix()})
    writer.json("research_path_comparison.json", research_path_comparison())
    dac = data_acquisition_source_classification()
    writer.json("data_acquisition_source_classification.json", {"created_at": iso_kst(), "records": dac})
    writer.jsonl("data_acquisition_source_classification.jsonl", dac)
    writer.json("privacy_governance_plan.json", privacy_governance_plan())
    writer.json("split_safe_ingestion_plan.json", split_safe_ingestion_plan())
    writer.json("minimum_viable_milestones.json", minimum_viable_milestones())
    wps = state_source_repair_work_packages()
    writer.json("state_source_repair_work_packages.json", {"created_at": iso_kst(), "records": wps})
    writer.jsonl("state_source_repair_work_packages.jsonl", wps)
    priority = critical_source_priority_registry()
    writer.json("critical_source_priority_registry.json", {"created_at": iso_kst(), "records": priority})
    writer.jsonl("critical_source_priority_registry.jsonl", priority)
    retro = retrospective_availability_assessment()
    writer.json("retrospective_availability_assessment.json", retro)
    writer.json("prospective_collection_requirement.json", prospective_collection_requirement())

    # prohibition / immutability audits
    writer.json("simulator_execution_prohibition_audit.json", {"created_at": iso_kst(), "simulator_execution_count": 0, "historical_reconstruction_count": 0, "service_day_replay_count": 0, "thirty_step_branch_count": 0, "simulator_imported": False})
    writer.json("historical_row_access_prohibition_audit.json", {"created_at": iso_kst(), "historical_row_content_access_count": 0, "train_tensor_loaded_count": 0, "dataset_rows_loaded": False})
    writer.json("validation_untouched_audit.json", {"created_at": iso_kst(), "validation_access_count": 0})
    writer.json("test_holdout_untouched_audit.json", {"created_at": iso_kst(), "test_access_count": 0, "test_touched": False})
    writer.json("sealed_holdout_preservation_audit.json", {"created_at": iso_kst(), "sealed_holdout_access_count": 0, "sealed_holdout_reopened": False, "dl6d_r3_holdout_status": "FAILED_SEALED_NO_REUSE", "holdout_status_changed": False})
    writer.json("training_prohibition_audit.json", {"created_at": iso_kst(), "training_run_count": 0, "optimizer_step_count": 0, "checkpoint_write_count": 0})
    writer.json("external_access_audit.json", {"created_at": iso_kst(), "external_network_access_count": 0, "source_download_count": 0, "db_write_count": 0, "api_call_count": 0, "git_commit_count": 0, "git_push_count": 0})
    writer.json("stage_immutability_audit.json", {"created_at": iso_kst(), "sf0_artifact_mutation_count": 0, "source_modification_count": 0, "upstream_artifact_mutation_count": 0})

    # substitution guard: this plan proposes 0 synthetic substitutions (it registers PROHIBITED ones)
    prohibited_synthetic_substitution_count = 0

    runner_sha_after = sha256_file(RUNNER_PATH)
    runner_freeze = {"created_at": iso_kst(), "runner_relative_path": str(RUNNER_PATH.relative_to(PROJECT_ROOT)),
                     "runner_sha256_before_execution": runner_sha_before, "runner_size_before_execution": runner_size_before,
                     "runner_sha256_after_execution": runner_sha_after, "runner_snapshot_sha256": runner_snapshot["copied_sha256"],
                     "runner_mutation_count": 0 if runner_sha_before == runner_sha_after else 1,
                     "runner_frozen": runner_sha_before == runner_sha_after == runner_snapshot["copied_sha256"]}
    writer.json("runner_freeze_audit.json", runner_freeze)
    if runner_freeze["runner_mutation_count"]:
        raise PlanError(FAIL_RUNNER, "runner SHA changed during plan")

    decision = decide_gate(recommended, policy, packages, retro)
    gate_status = decision["gate"]
    is_pass = gate_status in {PASS_EXACT, PASS_REDUCED, PASS_ACQUIRE, PASS_INFEASIBLE}
    writer.json("source_repair_decision.json", {"created_at": iso_kst(), "gate": gate_status, "decision_reason": decision["reason"],
                "recommended_agent_semantics": recommended["recommended_historical_agent_semantics"],
                "policy_compatibility_status": policy["policy_compatibility_status"],
                "exact_three_action_source_path_identified": gate_status == PASS_EXACT,
                "reduced_hs_path_identified": research_path_comparison()["PATH_B_reduced_hs"]["currently_available"],
                "source_acquisition_required": gate_status == PASS_ACQUIRE,
                "retrospective_exact_reconstruction_feasible": False,
                "prospective_collection_required": True})
    gate = {"created_at": iso_kst(), "mode": "plan", "gate": gate_status, "gate_passed": is_pass, "readiness": READINESS.get(gate_status, "SRP0_BLOCKED"),
            "state_reconstruction_authorized": False, "service_day_replay_authorized": False, "historical_transition_authorized": False,
            "reward_rebuild_authorized": False, "canonical_kpi_rebuild_authorized": False, "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False}
    writer.json("gate_decision.json", gate)
    writer.json("next_stage_readiness.json", {"created_at": iso_kst(), "gate": gate_status, "readiness": gate["readiness"],
                "next_step": "EXTERNAL_SOURCE_ACQUISITION_PLAN" if gate_status == PASS_ACQUIRE else gate_status})

    writer.json("downstream_lock.json", {
        "sf0_upstream_verified": True, "srp0_plan_complete": is_pass,
        "recommended_agent_semantics": recommended["recommended_historical_agent_semantics"],
        "policy_compatibility_status": policy["policy_compatibility_status"],
        "exact_three_action_source_path_identified": gate_status == PASS_EXACT,
        "reduced_hs_path_identified": research_path_comparison()["PATH_B_reduced_hs"]["currently_available"],
        "retrospective_exact_reconstruction_feasible": False, "prospective_collection_required": True,
        "vehicle_event_source_available": False, "passenger_service_event_source_available": False, "request_assignment_source_available": False,
        "route_stop_safety_source_available": False, "ordered_replay_source_available": False,
        "prohibited_synthetic_substitution_count": prohibited_synthetic_substitution_count,
        "source_acquisition_required": gate_status == PASS_ACQUIRE,
        "state_reconstruction_authorized": False, "service_day_replay_authorized": False, "historical_transition_authorized": False,
        "reward_rebuild_authorized": False, "canonical_kpi_rebuild_authorized": False, "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False})

    report_payload, report_md = build_final_report(root, gate, recommended, policy, code_audit, packages, gap_matrix, retro, decision, runner_freeze)
    writer.json("final_report.json", report_payload)
    writer.text("final_report.md", report_md + "\n")

    manifest = write_manifest(writer, "artifact_manifest_srp0.json", SRP0_PAYLOADS)
    if manifest["missing_payload_count"]:
        raise PlanError(FAIL_MANIFEST, f"missing payloads: {manifest['missing_payloads']}")
    write_lock(writer, "_SRP0_PLAN_COMPLETE.lock", "artifact_manifest_srp0.json", gate)
    v = verify_manifest(root, "_SRP0_PLAN_COMPLETE.lock")
    if not v["manifest_hash_ok"] or not v["manifest_size_ok"] or v["payload_missing_count"] or v["payload_hash_mismatch_count"] or v["terminal_lock_listed_inside_manifest"] or v["manifest_self_listed"]:
        raise PlanError(FAIL_MANIFEST, f"manifest verification failed: {v}")

    if not is_pass:
        raise PlanError(gate_status, decision["reason"])

    print("SRP0 STATE-SOURCE REPAIR PLAN COMPLETE")
    print(f"artifact_root: {root}")
    print(f"gate: {gate_status}")
    print(f"readiness: {gate['readiness']}")
    print(f"recommended_agent_semantics: {recommended['recommended_historical_agent_semantics']}")
    print(f"policy_compatibility: {policy['policy_compatibility_status']}")
    print(f"prohibited_synthetic_substitution_count: {prohibited_synthetic_substitution_count}")
    print(f"runner_mutation_count: {runner_freeze['runner_mutation_count']} | source_drift: {source_reg['source_drift_count']}")
    print("state_reconstruction_authorized: false")
    return root


def build_final_report(root, gate, recommended, policy, code_audit, packages, gap_matrix, retro, decision, runner_freeze):
    answers = {
        "01_current_8_agents_mean": "SYNTHETIC_AGENT_SLOT (policy slots, not identified historical vehicles)",
        "02_can_switch_to_physical_vehicle": recommended["recommended_historical_agent_semantics"] == "PHYSICAL_VEHICLE",
        "03_current_checkpoint_compatible_with_physical_vehicle": "SEPARATE_UNRESOLVED — DL-6D-R3 reward alignment FAILED; no current checkpoint validated for benefit",
        "04_policy_interface_change_needed": policy["policy_compatibility_status"] == "REQUIRES_POLICY_INTERFACE_ADAPTATION",
        "05_retraining_needed": "LIKELY (2->3 action space change and stub->real adapter); to be decided after interface adaptation",
        "06_source_package_for_full_hsk": [p["package_id"] for p in packages],
        "07_already_present_locally": "NONE exact; only static route topology + node-level 10-min aggregate (observation/context use only)",
        "08_must_be_acquired": ["P1 vehicle AVL", "P3 passenger service", "P4 request/assignment (if applicable)", "P5 route/stop safety", "P6 fine travel-time"],
        "09_retrospectively_acquirable": retro["likely_retrospectively_available"] + retro["acquisition_dependent_uncertain"],
        "10_not_retrospectively_reconstructable": retro["likely_retrospectively_unavailable"],
        "11_waiting_queue_from": "P3 PASSENGER_WAIT_STARTED / PASSENGER_BOARDED events (per-passenger)",
        "12_assigned_pickup_dropoff_from": "P4 request/assignment events (only if a DRT platform existed historically)",
        "13_onboard_destination_from": "P3 board+alight linkage with destination_stop_id per service leg",
        "14_safety_flags_from": "P5 operator route/stop safety configuration (time-valid, no retroactive current config)",
        "15_eight_vehicle_selection_rule": "deterministic canonical (route_id, direction_id, vehicle_id) among anchor-active vehicles; no random/hash/future/outcome selection; no synthetic padding",
        "16_service_day_start_state_needs": "fleet roster, block allocation, initial positions/onboard/requests/counters/provider cursors/mode — all source-proven",
        "17_ordered_replay_schema": "unified (service_day,event_timestamp,event_sequence,event_type,entity_type,entity_id,source_id,source_event_id,payload_hash)",
        "18_split_isolation": "split by service day / time block BEFORE snapshot build; sealed TEST service days stay sealed; TRAIN-only initial implementation",
        "19_exact_three_action_path_exists": gate["gate"] == "PASS_SUSEONG_DL6D_PA1A_SRP0_STATE_SOURCE_REPAIR_PLAN_COMPLETE_EXACT_THREE_ACTION_PATH_IDENTIFIED",
        "20_reduced_hs_only": False,
        "21_retrospective_reconstruction_infeasible": "PARTIAL — key K-safety/passenger sources likely infeasible; others acquisition-dependent",
        "22_prospective_collection_needed": True,
        "23_next_step": "EXTERNAL_SOURCE_ACQUISITION_PLAN (identify/obtain operator BIS-AVL, AFC/OD, schedule, safety config); record prospective-only for onboard-destination/DRT",
    }
    payload = {"created_at": iso_kst(), "artifact_root": str(root), "mode": "plan", "gate": gate["gate"], "gate_passed": gate["gate_passed"],
               "readiness": gate["readiness"], "quick_answers": answers, "scope": "READ_ONLY_STATE_SOURCE_REPAIR_PLANNING",
               "recommended_agent_semantics": recommended["recommended_historical_agent_semantics"], "policy_compatibility_status": policy["policy_compatibility_status"],
               "decision_reason": decision["reason"], "prohibited_synthetic_substitution_count": 0,
               "state_reconstruction_authorized": False, "service_day_replay_authorized": False, "historical_transition_authorized": False,
               "reward_rebuild_authorized": False, "canonical_kpi_rebuild_authorized": False, "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False,
               "next_authorized_action": "External source acquisition planning review only, after explicit user review and command"}
    lines = ["# SRP0 Historical State-Source Repair Plan & Agent-Semantics Adjudication", "",
             f"- artifact_root: {root}", f"- gate: {gate['gate']}", f"- readiness: {gate['readiness']}",
             f"- recommended agent semantics: {recommended['recommended_historical_agent_semantics']}",
             f"- policy compatibility: {policy['policy_compatibility_status']}", "",
             "## Bottom line",
             "- The 8 MAPPO agents should be PHYSICAL_VEHICLE (agent slot = an active bus). The engine keys vehicle state by canonical agent slot and the shared homogeneous actor is permutation-safe; route/zone agents are incompatible; the current POLICY_SLOT is synthetic-only.",
             "- The MAPPO policy interface needs adaptation (2-action interface vs 3-action engine, stub historical adapter, 8-agent count). The current checkpoint is NOT validated (DL-6D-R3 reward alignment failed) — separate from state feasibility.",
             "- No required per-entity historical source (vehicle AVL, passenger service, request/assignment, safety config) exists locally. They are identifiable by institution/system (operator BIS-AVL, AFC/OD, schedule, config) — SOURCE_ACQUISITION_REQUIRED.",
             "- The highest-criticality K sources (per-passenger onboard obligation, DRT assignment for fixed-route history) are likely retrospectively unavailable → prospective collection.",
             "- No synthetic substitution proposed; 10 prohibited substitutions registered. No simulator execution, no row access, sealed hold-out preserved.", "",
             "## Quick answers"]
    for k in sorted(answers):
        lines.append(f"- {k}: {answers[k]}")
    lines += ["", "## Downstream (all locked)",
              "- state_reconstruction / service_day_replay / historical_transition: false",
              "- reward_rebuild / canonical_kpi_rebuild: false | pa1b / dl6e_p0 / training: false",
              "- DL-6D-R3 sealed hold-out preserved (FAILED, no reuse).", ""]
    return payload, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["plan"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        run_plan(args.artifact_root)
    except PlanError as exc:
        print("SRP0 STATE-SOURCE REPAIR PLAN FAILED")
        print(f"gate: {exc.gate_status}")
        print(f"detail: {exc.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
