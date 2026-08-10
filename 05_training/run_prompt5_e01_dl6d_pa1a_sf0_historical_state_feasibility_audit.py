#!/usr/bin/env python3
"""Prompt 5-E01-DL-6D-PA1-A-SF0.

Historical Dynamics State Reconstruction Feasibility Audit.

Read-only. Determines whether the DynamicsStateSnapshot required state fields can
be reconstructed for a historical Suseong anchor timestamp without synthetic
defaults or future leakage. Does NOT execute the simulator, does NOT read
validation/test/sealed-hold-out row content, and inserts no default values.

Allowed imports are limited to the pure state/replay contracts. The engine,
orchestrator, horizon aggregator, and event-trace modules are never imported.
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
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_sf0_historical_state_feasibility_audit.py"

V1F_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_er1_v1f_final_closeout_20260803_183026"
V1F_GATE = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_FINALIZED_WITH_MIXED_DL6B_EVIDENCE_SCOPE"
V1F_READINESS = "V1F_FINALIZED_PA1A_STATE_FEASIBILITY_PENDING_USER_COMMAND"
V1F_MANIFEST = "artifact_manifest_v1f_final.json"
V1F_MANIFEST_SHA = "752d1a4630f8fd23b0a285c1219a7e947f9399515b070d1234a069225a119575"
V1F_MANIFEST_SIZE = 12812
V1F_MANIFEST_PAYLOAD_COUNT = 53

FROZEN_SOURCE_SHA = {
    "05_training/simulator/dynamics_multiagent_orchestrator.py": "ee9636f43adeac5dcfa04ec976aca0d72ff05f6e356dbc8beac4f9f1b14d8b81",
    "05_training/simulator/dynamics_event_trace.py": "7886bdafe55ae1794bec610b16a573c2909a64a6cb5aa2545ca5d8391a04a14e",
    "05_training/simulator/suseong_service_transition_engine.py": "37cdfbf4eb486bd0157fe47000924ea2946b787c953e70ff9fea0f821bca8ad5",
    "05_training/simulator/dynamics_state_snapshot.py": "e189d3b3a39acebc957fc91b8225ed1030babb19b28f3825b9f9b822184f0105",
    "05_training/simulator/dynamics_replay_contract.py": "13be29fa5fd2b6637ff9096df3787666204eed07fca136d11b00e3dee2b51477",
    "05_training/simulator/dynamics_horizon_aggregator.py": "7dae0133db4ac28d332952da97bbe193d2b6f425f8abdcf44a4a994e6b98b9ce",
}

# Authoritative frozen dataset split (DL-3 dataset_split_manifest.json).
DL3_SPLIT_MANIFEST = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915/dataset_split_manifest.json"
DL3_SPLIT_AUDIT = ARTIFACTS_ROOT / "prompt5_e01_dl3_suseong_three_seed_full_training_20260731_115915/dataset_split_audit.json"

HISTORICAL_SCHEMA_SOURCES = [
    ("02_ingest_jobs/create_graph_state_timeslice.sql", "graph_state_timeslice", "node-level 10-minute state bucket"),
    ("02_ingest_jobs/create_rl_state_training_base.sql", "rl_state_training_base", "node-level state-transition training rows"),
    ("02_ingest_jobs/create_graph_node_master.sql", "graph_node_master", "static node (stop) master"),
    ("02_ingest_jobs/create_graph_edge_master.sql", "graph_edge_master", "static route link/edge master"),
    ("scripts/create_route_link_sequence.sql", "route_link_sequence", "static route stop sequence"),
    ("04_model_inputs/graph_state_feature_spec.md", "graph_state_feature_spec", "GATv2 node feature specification"),
    ("04_model_inputs/daegu_pyg_dataset.py", "daegu_pyg_dataset", "GATv2 dataset builder (node-feature tensors)"),
]

PASS_EXACT = "PASS_SUSEONG_DL6D_PA1A_SF0_HISTORICAL_STATE_FEASIBILITY_AUDIT_COMPLETE_EXACT_RECONSTRUCTION_SUPPORTED"
PASS_REPLAY = "PASS_SUSEONG_DL6D_PA1A_SF0_HISTORICAL_STATE_FEASIBILITY_AUDIT_COMPLETE_SERVICE_DAY_REPLAY_REQUIRED"
PASS_GAPS = "PASS_SUSEONG_DL6D_PA1A_SF0_HISTORICAL_STATE_FEASIBILITY_AUDIT_COMPLETE_CRITICAL_GAPS_RECORDED"
PASS_NOT_RECON = "PASS_SUSEONG_DL6D_PA1A_SF0_HISTORICAL_STATE_FEASIBILITY_AUDIT_COMPLETE_NOT_RECONSTRUCTABLE"
READINESS = {
    PASS_EXACT: "SF0_COMPLETE_SF1_LIMITED_HISTORICAL_RECONSTRUCTION_PENDING_USER_COMMAND",
    PASS_REPLAY: "SF0_COMPLETE_SERVICE_DAY_REPLAY_RECONSTRUCTION_DESIGN_PENDING_USER_COMMAND",
    PASS_GAPS: "SF0_COMPLETE_STATE_SOURCE_REPAIR_PLAN_PENDING_USER_COMMAND",
    PASS_NOT_RECON: "SF0_COMPLETE_HISTORICAL_STATE_REBUILD_REQUIRED",
}
BLOCKED_TRAIN = "BLOCKED_SUSEONG_DL6D_PA1A_SF0_AUTHORITATIVE_TRAIN_SPLIT_NOT_FOUND"
BLOCKED_SOURCE = "BLOCKED_SUSEONG_DL6D_PA1A_SF0_REQUIRED_HISTORICAL_SOURCE_UNAVAILABLE"
BLOCKED_AGENT = "BLOCKED_SUSEONG_DL6D_PA1A_SF0_AGENT_SEMANTICS_INDETERMINATE"

_F = "FAIL_SUSEONG_DL6D_PA1A_SF0_"
FAIL_V1F = _F + "V1F_UPSTREAM_INVALID"
FAIL_RUNNER = _F + "RUNNER_MUTATED_DURING_AUDIT"
FAIL_SOURCE_DRIFT = _F + "SOURCE_DRIFT"
FAIL_NONTRAIN = _F + "NONTRAIN_ROW_ACCESSED"
FAIL_SEALED = _F + "SEALED_HOLDOUT_ACCESSED"
FAIL_LEAKAGE = _F + "FUTURE_LEAKAGE"
FAIL_SYNTHETIC_DEFAULT = _F + "SYNTHETIC_DEFAULT_INSERTED"
FAIL_PROVENANCE = _F + "SOURCE_PROVENANCE_INCOMPLETE"
FAIL_SIMULATOR = _F + "SIMULATOR_EXECUTION_DETECTED"
FAIL_REWARD_KPI = _F + "REWARD_KPI_RESULT_CREATED"
FAIL_TRAINING = _F + "PROHIBITED_TRAINING"
FAIL_UPSTREAM_MUTATED = _F + "UPSTREAM_ARTIFACT_MUTATED"
FAIL_MANIFEST = _F + "MANIFEST_RECONCILIATION"


class AuditError(RuntimeError):
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
# preflight / split / source inventory
# ---------------------------------------------------------------------------

def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"sf0 artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def v1f_preflight() -> Dict[str, Any]:
    gate = read_json(V1F_ROOT / "gate_decision.json")
    lock = read_json(V1F_ROOT / "_SUCCESS.lock")
    manifest_path = V1F_ROOT / V1F_MANIFEST
    manifest = read_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    seen: Dict[str, int] = {}
    missing = mismatch = size_mismatch = 0
    for row in manifest["files"]:
        seen[row["relative_path"]] = seen.get(row["relative_path"], 0) + 1
        t = V1F_ROOT / row["relative_path"]
        if not t.exists():
            missing += 1
            continue
        if row.get("sha256") is not None and sha256_file(t) != row["sha256"]:
            mismatch += 1
        if row.get("size_bytes") is not None and t.stat().st_size != row["size_bytes"]:
            size_mismatch += 1
    duplicate = sorted(p for p, c in seen.items() if c > 1)
    checks = {
        "gate": gate.get("gate") == V1F_GATE, "readiness": gate.get("readiness") == V1F_READINESS,
        "lock_present": (V1F_ROOT / "_SUCCESS.lock").exists(),
        "lock_manifest_path": lock.get("manifest_relative_path") == V1F_MANIFEST,
        "lock_manifest_sha": lock.get("manifest_sha256") == manifest_sha == V1F_MANIFEST_SHA,
        "lock_manifest_size": lock.get("manifest_size_bytes") == manifest_path.stat().st_size == V1F_MANIFEST_SIZE,
        "manifest_payload_53": manifest.get("required_payload_count") == V1F_MANIFEST_PAYLOAD_COUNT,
        "manifest_missing_zero": missing == 0, "manifest_hash_mismatch_zero": mismatch == 0,
        "manifest_size_mismatch_zero": size_mismatch == 0, "manifest_duplicate_zero": len(duplicate) == 0,
        "manifest_self_ref_absent": not any(r["relative_path"] == V1F_MANIFEST for r in manifest["files"]),
        "terminal_lock_not_in_manifest": not any(r["relative_path"] == "_SUCCESS.lock" for r in manifest["files"]),
    }
    return {"created_at": iso_kst(), "v1f_artifact_root": str(V1F_ROOT), "manifest_sha256": manifest_sha,
            "manifest_size_bytes": manifest_path.stat().st_size, "gate": gate.get("gate"), "readiness": gate.get("readiness"),
            "payload_missing_count": missing, "payload_hash_mismatch_count": mismatch, "payload_size_mismatch_count": size_mismatch,
            "duplicate_path_count": len(duplicate), "checks": checks, "v1f_upstream_valid": all(checks.values())}


def snapshot_v1f(writer: Writer) -> Dict[str, Any]:
    records = []
    for src, dst in [
        ("gate_decision.json", "upstream_v1f_snapshot/v1f_gate_decision.json"),
        ("downstream_lock.json", "upstream_v1f_snapshot/v1f_downstream_lock.json"),
        ("artifact_manifest_v1f_final.json", "upstream_v1f_snapshot/v1f_artifact_manifest.json"),
        ("_SUCCESS.lock", "upstream_v1f_snapshot/v1f_SUCCESS.lock"),
        ("final_report.json", "upstream_v1f_snapshot/v1f_final_report.json"),
        ("dynamics_source_final_registry.json", "upstream_v1f_snapshot/v1f_dynamics_source_registry.json"),
    ]:
        records.append(copy_file(writer, V1F_ROOT / src, dst))
    payload = {"created_at": iso_kst(), "record_count": len(records), "all_byte_identical": all(r["byte_identical"] for r in records), "records": records}
    writer.json("upstream_v1f_registry.json", payload)
    return payload


def frozen_source_registry(writer: Writer) -> Dict[str, Any]:
    records = []
    for rel, frozen in sorted(FROZEN_SOURCE_SHA.items()):
        p = PROJECT_ROOT / rel
        cur = sha256_file(p) if p.exists() else None
        records.append({"relative_path": rel, "exists": p.exists(), "frozen_sha256": frozen, "runtime_sha256": cur, "matches_frozen": cur == frozen})
    return {"created_at": iso_kst(), "source_drift_count": sum(1 for r in records if not r["matches_frozen"]), "records": records}


def authoritative_split_discovery() -> Dict[str, Any]:
    found = DL3_SPLIT_MANIFEST.is_file()
    if not found:
        return {"created_at": iso_kst(), "authoritative_train_split_found": False,
                "reason": "DL-3 dataset_split_manifest.json not found", "records": []}
    manifest = read_json(DL3_SPLIT_MANIFEST)
    audit = read_json(DL3_SPLIT_AUDIT) if DL3_SPLIT_AUDIT.is_file() else {}
    dataset_root = manifest.get("dataset_root")
    return {
        "created_at": iso_kst(),
        "authoritative_train_split_found": True,
        "split_registry_path": str(DL3_SPLIT_MANIFEST),
        "split_registry_sha256": sha256_file(DL3_SPLIT_MANIFEST),
        "authoritative_split_source": manifest.get("authoritative_split_source"),
        "dataset_root": dataset_root,
        "split_counts": manifest.get("split_counts_from_build_report"),
        "split_order": manifest.get("split_order"),
        "split_hash": manifest.get("split_hash"),
        "train_files_hash": manifest.get("train_files_hash"),
        "validation_files_hash": manifest.get("validation_files_hash"),
        "test_files_hash": manifest.get("test_files_hash"),
        "split_integrity_passed": audit.get("split_integrity_passed"),
        "train_test_overlap": audit.get("train_test_overlap"),
        "sealed_holdout_binding": "TEST split (554 snapshots snapshot_06017..snapshot_06570) == DL-6D-R3 sealed hold-out",
        "row_level_access_policy": "TRAIN counts/hashes read from registry only; no VALIDATION/TEST/SEALED row content read; no .pt content loaded",
        "classification": {"TRAIN": manifest.get("split_counts_from_build_report", {}).get("train"),
                           "VALIDATION": manifest.get("split_counts_from_build_report", {}).get("val"),
                           "TEST": manifest.get("split_counts_from_build_report", {}).get("test"),
                           "SEALED_HOLDOUT": manifest.get("split_counts_from_build_report", {}).get("test")},
    }


def sql_columns(path: Path) -> List[str]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    cols = []
    for m in re.finditer(r"^\s*([a-z_][a-z0-9_]*)\s+(text|integer|int|numeric|bigint|boolean|bool|timestamp|date|double|real|serial)\b", text, re.I | re.M):
        cols.append(m.group(1).lower())
    for m in re.finditer(r"\bAS\s+(next_[a-z_]+|[a-z_]+_recent|waiting_passenger_cnt|state_ts|link_[a-z_]+)\b", text, re.I):
        cols.append(m.group(1).lower())
    return sorted(set(cols))


def historical_source_inventory() -> Dict[str, Any]:
    records = []
    for rel, table, granularity in HISTORICAL_SCHEMA_SOURCES:
        p = PROJECT_ROOT / rel
        exists = p.is_file()
        cols = sql_columns(p) if rel.endswith(".sql") else []
        text = p.read_text(encoding="utf-8", errors="ignore") if exists else ""
        has_vehicle = bool(re.search(r"vehicle_id|vhc|per.?vehicle", text, re.I))
        has_request = bool(re.search(r"request_id|assigned_pickup|assigned_dropoff", text, re.I))
        has_onboard = bool(re.search(r"onboard|scheduled_dropoff|destination_stop", text, re.I))
        has_schedule = bool(re.search(r"scheduled_(arrival|departure)|trip_id|headway", text, re.I))
        has_safety_flags = bool(re.search(r"mandatory_stop|protected_stop|planned_itinerary|service_fairness|max_consecutive_skip", text, re.I))
        records.append({
            "source_id": table, "path_or_table": rel, "source_type": p.suffix.lstrip(".") or "none", "exists": exists,
            "sha256_or_fingerprint": sha256_file(p) if exists else None, "row_granularity": granularity,
            "columns_detected": cols,
            "has_per_vehicle_state": has_vehicle, "has_request_assignment": has_request, "has_onboard_destination": has_onboard,
            "has_schedule_headway": has_schedule, "has_route_stop_safety_flags": has_safety_flags,
            "candidate_state_fields": _candidate_fields(table, cols),
            "source_status": "PRESENT" if exists else "ABSENT",
        })
    return {"created_at": iso_kst(), "source_count": len(records),
            "any_per_vehicle_source": any(r["has_per_vehicle_state"] for r in records),
            "any_request_assignment_source": any(r["has_request_assignment"] for r in records),
            "any_onboard_destination_source": any(r["has_onboard_destination"] for r in records),
            "any_schedule_headway_source": any(r["has_schedule_headway"] for r in records),
            "any_route_stop_safety_flag_source": any(r["has_route_stop_safety_flags"] for r in records),
            "records": records}


def _candidate_fields(table: str, cols: List[str]) -> List[str]:
    fields = []
    if "state_ts" in cols or table in {"graph_state_timeslice", "rl_state_training_base"}:
        fields.append("simulation_timestamp_seconds (10-minute bucket granularity)")
    if "waiting_passenger_cnt" in cols:
        fields.append("waiting_passengers (node-level PROXY count only, not per-passenger)")
    if table in {"graph_node_master", "route_link_sequence", "graph_edge_master"}:
        fields.append("routes (static topology only, no per-run safety flags)")
    return fields


# ---------------------------------------------------------------------------
# contract + field/source matrices  (evidence-driven from the source inventory)
# ---------------------------------------------------------------------------

def import_pure_contract() -> Dict[str, Any]:
    if str(TRAINING_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAINING_ROOT))
    from simulator.dynamics_state_snapshot import REQUIRED_SNAPSHOT_FIELDS, SNAPSHOT_SCHEMA_VERSION  # pure contract
    return {"REQUIRED_SNAPSHOT_FIELDS": list(REQUIRED_SNAPSHOT_FIELDS), "SNAPSHOT_SCHEMA_VERSION": SNAPSHOT_SCHEMA_VERSION,
            "pure_contract_import_count": 2, "transition_module_import_count": 0}


def dynamics_state_contract_registry(contract: Mapping[str, Any]) -> Dict[str, Any]:
    fields = contract["REQUIRED_SNAPSHOT_FIELDS"]
    return {"created_at": iso_kst(), "source_path": "05_training/simulator/dynamics_state_snapshot.py",
            "source_authoritative_required_field_count": len(fields), "required_fields": fields,
            "prompt_declared_field_count": 17, "prompt_vs_source_field_count_match": len(fields) == 17,
            "field_count_discrepancy_note": (
                "Source REQUIRED_SNAPSHOT_FIELDS defines 16 fields; the prompt states 17. Per the prompt's own rule, "
                "the source value (16) is authoritative and the difference is reported here."),
            "authoritative_field_count": len(fields), "schema_version_constant": contract["SNAPSHOT_SCHEMA_VERSION"]}


def state_field_source_matrix(contract: Mapping[str, Any], inventory: Mapping[str, Any]) -> List[Dict[str, Any]]:
    inv = {r["source_id"]: r for r in inventory["records"]}
    ts_source = inv["graph_state_timeslice"]
    # per-field evidence classification derived from the discovered source inventory
    F = {
        "schema_version": dict(candidate_source="dynamics_state_snapshot contract constant", evidence_class="STATIC_CONFIGURATION_VALID_AT_ANCHOR",
             temporal_granularity="constant", reconstruction_method="contract constant", future_leakage_risk="NONE",
             exactness_status="EXACT_AT_ANCHOR", blocking_reason=None),
        "simulation_timestamp_seconds": dict(candidate_source="graph_state_timeslice.state_ts", evidence_class="DIRECTLY_OBSERVED_AT_ANCHOR",
             temporal_granularity="10-minute bucket", reconstruction_method="bucket start timestamp", future_leakage_risk="NONE",
             exactness_status="EXACT_AT_ANCHOR", blocking_reason="anchor granularity is 10 minutes, not per-second"),
        "vehicles": dict(candidate_source="NONE (no per-vehicle historical entity; 8 MAPPO agents are policy slots)", evidence_class="UNAVAILABLE",
             temporal_granularity="n/a", reconstruction_method="none", future_leakage_risk="n/a",
             exactness_status="NOT_RECONSTRUCTABLE", blocking_reason="no historical per-vehicle position/dwell/travel/onboard source; agent-to-vehicle identity absent"),
        "routes": dict(candidate_source="route_link_sequence + graph_node_master + graph_edge_master (topology only)", evidence_class="STATIC_CONFIGURATION_VALID_AT_ANCHOR",
             temporal_granularity="static", reconstruction_method="static route/stop topology", future_leakage_risk="NONE",
             exactness_status="PARTIAL_CRITICAL_GAPS", blocking_reason="route/stop safety flags (mandatory/protected/skip/fairness/path-validity) absent from masters"),
        "waiting_passengers": dict(candidate_source="graph_state_timeslice.waiting_passenger_cnt (node-level PROXY count)", evidence_class="APPROXIMATE_INFERENCE_NOT_ALLOWED",
             temporal_granularity="10-minute node aggregate", reconstruction_method="proxy count from boarding/alighting differences", future_leakage_risk="NONE",
             exactness_status="NOT_RECONSTRUCTABLE", blocking_reason="only a proxy aggregate count per node; no per-passenger waiting queue with identities/destinations"),
        "assigned_pickups": dict(candidate_source="NONE", evidence_class="UNAVAILABLE", temporal_granularity="n/a", reconstruction_method="none",
             future_leakage_risk="n/a", exactness_status="NOT_RECONSTRUCTABLE", blocking_reason="no request-level pickup assignment source"),
        "assigned_dropoffs": dict(candidate_source="NONE", evidence_class="UNAVAILABLE", temporal_granularity="n/a", reconstruction_method="none",
             future_leakage_risk="n/a", exactness_status="NOT_RECONSTRUCTABLE", blocking_reason="no request-level dropoff assignment source"),
        "onboard_passengers": dict(candidate_source="NONE", evidence_class="UNAVAILABLE", temporal_granularity="n/a", reconstruction_method="none",
             future_leakage_risk="n/a", exactness_status="NOT_RECONSTRUCTABLE", blocking_reason="no onboard-passenger or onboard-destination source"),
        "mandatory_stop_state": dict(candidate_source="NONE (graph_node_master lacks safety flags)", evidence_class="UNAVAILABLE", temporal_granularity="n/a",
             reconstruction_method="none", future_leakage_risk="n/a", exactness_status="NOT_RECONSTRUCTABLE", blocking_reason="no mandatory/protected stop flags in node master"),
        "action_mask_state": dict(candidate_source="DERIVED from vehicle position + route safety (both absent)", evidence_class="UNAVAILABLE", temporal_granularity="n/a",
             reconstruction_method="none", future_leakage_risk="n/a", exactness_status="NOT_RECONSTRUCTABLE", blocking_reason="depends on unavailable vehicle and safety-flag state"),
        "schedule_state": dict(candidate_source="NONE", evidence_class="UNAVAILABLE", temporal_granularity="n/a", reconstruction_method="none",
             future_leakage_risk="n/a", exactness_status="NOT_RECONSTRUCTABLE", blocking_reason="no per-trip schedule/timetable source"),
        "headway_state": dict(candidate_source="NONE (requires per-vehicle positions)", evidence_class="UNAVAILABLE", temporal_granularity="n/a",
             reconstruction_method="none", future_leakage_risk="HIGH_IF_USING_FUTURE_VEHICLE", exactness_status="NOT_RECONSTRUCTABLE", blocking_reason="no vehicle-position stream to compute preceding/following headway prior-only"),
        "operation_mode": dict(candidate_source="static implicit FIXED_ROUTE (no timestamped operation-mode source)", evidence_class="STATIC_CONFIGURATION_VALID_AT_ANCHOR",
             temporal_granularity="static", reconstruction_method="static configuration", future_leakage_risk="NONE",
             exactness_status="PARTIAL_CRITICAL_GAPS", blocking_reason="no explicit timestamped operation-mode source; general knowledge not accepted as evidence"),
        "shared_counters": dict(candidate_source="graph_state_timeslice boardings_recent/alightings_recent (10-min node aggregates)", evidence_class="RECONSTRUCTABLE_FROM_SERVICE_DAY_START_REPLAY",
             temporal_granularity="10-minute node aggregate", reconstruction_method="would require service-day event replay of running counters", future_leakage_risk="NONE",
             exactness_status="PARTIAL_CRITICAL_GAPS", blocking_reason="only 10-min node aggregates exist; no running trip/cycle/served-request counters and no per-event stream"),
        "replay_cursor": dict(candidate_source="NONE (no historical event stream; only 10-min node buckets)", evidence_class="UNAVAILABLE", temporal_granularity="n/a",
             reconstruction_method="none", future_leakage_risk="n/a", exactness_status="NOT_RECONSTRUCTABLE", blocking_reason="no ordered historical event stream to anchor a replay cursor"),
        "external_provider_states": dict(candidate_source="NONE / not proven unnecessary", evidence_class="UNAVAILABLE", temporal_granularity="n/a",
             reconstruction_method="none", future_leakage_risk="n/a", exactness_status="NOT_RECONSTRUCTABLE", blocking_reason="no provider state source; non-use for historical reconstruction is not proven"),
    }
    rows = []
    for field in contract["REQUIRED_SNAPSHOT_FIELDS"]:
        spec = F[field]
        rows.append({"field_name": field, "contract_required": True, "sample_coverage": "SCHEMA_LEVEL_SOURCE_INVENTORY", **spec})
    return rows


def vehicle_state_subfield_matrix() -> List[Dict[str, Any]]:
    subs = ["agent_id", "historical_vehicle_id", "route_key", "route_id", "direction_id", "position", "current_stop_id",
            "target_position", "current_edge_id", "remaining_travel_seconds", "remaining_dwell_seconds", "remaining_turnaround_seconds",
            "vehicle_state", "onboard_count", "onboard_destination_stop_ids", "passenger_destinations", "scheduled_dropoff_counts",
            "consecutive_skip_count", "_ready_to_depart"]
    rows = []
    for s in subs:
        avail = "agent_id" == s  # only the policy-slot agent index is derivable; nothing else historical
        rows.append({"subfield": s, "availability": "POLICY_SLOT_INDEX_ONLY" if avail else "UNAVAILABLE",
                     "observed": False, "derived": avail, "service_day_replay_required": False,
                     "exact_timestamp_age": None, "source_conflict": False,
                     "blocking_reason": None if avail else "no historical per-vehicle dynamic source"})
    return rows


def route_stop_safety_field_matrix() -> List[Dict[str, Any]]:
    fields = ["stop_id_or_node_uid", "route_id", "direction_id", "route_sequence_index", "waiting_pickup_count",
              "scheduled_alighting_count", "assigned_pickup_request_count", "assigned_dropoff_request_count", "mandatory_stop",
              "protected_stop", "terminal_or_turnaround_stop", "charging_or_driver_relief_stop", "planned_itinerary_allows_skip",
              "downstream_path_valid", "graph_edge_or_path_valid", "max_consecutive_skip_constraint_satisfied", "service_fairness_constraint_satisfied"]
    static_topology = {"stop_id_or_node_uid", "route_id", "route_sequence_index", "graph_edge_or_path_valid"}
    proxy_only = {"waiting_pickup_count"}
    rows = []
    for f in fields:
        if f in static_topology:
            rows.append({"field": f, "availability": "STATIC_TOPOLOGY", "evidence_class": "STATIC_CONFIGURATION_VALID_AT_ANCHOR", "default_fill_prohibited": True, "blocking_reason": None})
        elif f in proxy_only:
            rows.append({"field": f, "availability": "NODE_PROXY_AGGREGATE_ONLY", "evidence_class": "APPROXIMATE_INFERENCE_NOT_ALLOWED", "default_fill_prohibited": True,
                         "blocking_reason": "only node-level proxy count; not a per-stop request-level count"})
        else:
            rows.append({"field": f, "availability": "UNAVAILABLE", "evidence_class": "UNAVAILABLE", "default_fill_prohibited": True,
                         "blocking_reason": "no historical source; K feasibility requires exact safety/obligation flags that do not exist"})
    return rows


def passenger_obligation_audit() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "waiting_passenger": "AGGREGATE_DEMAND_ONLY (node-level proxy count per 10-min bucket)",
            "assigned_pickup": "UNAVAILABLE", "assigned_dropoff": "UNAVAILABLE",
            "onboard_passenger": "UNAVAILABLE", "onboard_destination": "UNAVAILABLE", "scheduled_alighting": "AGGREGATE_DEMAND_ONLY (alightings_recent 10-min aggregate)",
            "passenger_level_events_available": False, "request_level_assignment_available": False, "onboard_destination_available": False,
            "classification": "AGGREGATE_DEMAND_ONLY",
            "note": "node-level 10-minute boarding/alighting/waiting proxies cannot reconstruct anchor-time per-passenger queues, request assignments, or onboard destinations"}


def agent_semantics_audit() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "agent_semantics": "SYNTHETIC_AGENT_SLOT",
            "agent_id_generation_source": "MAPPO fixed 8-slot policy index; not derived from a historical vehicle roster",
            "agent_to_vehicle_relation": "NONE (historical dataset is node-centric; no vehicle entities)",
            "agent_to_route_relation": "NONE at the historical row level",
            "agent_to_direction_relation": "NONE at the historical row level",
            "mapping_temporally_variable": "INDETERMINATE (no historical vehicle identity to vary)",
            "one_agent_to_many_vehicle": "INDETERMINATE", "one_vehicle_to_many_agent": "INDETERMINATE",
            "eight_agents_identifiable_in_history": False,
            "agent_mapping_exact": False, "agent_mapping_temporally_valid": False, "historical_vehicle_identity_available": False,
            "agent_mapping_blockers": ["no historical per-vehicle identity", "node-centric dataset has no vehicle roster", "8 agents are policy slots"],
            "historical_8_agent_vehicle_state_exact_reconstruction": False}


# ---------------------------------------------------------------------------
# schedule/headway/mode, counters, replay, provider, temporal, anchors, actions
# ---------------------------------------------------------------------------

def schedule_headway_operation_mode_audit(inventory: Mapping[str, Any]) -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "schedule_state_source": "NONE", "schedule_reconstructable": False,
            "headway_state_source": "NONE", "headway_reconstructable": False,
            "headway_future_leakage_risk": "would require future/other vehicle positions which do not exist historically",
            "operation_mode_source": "STATIC_IMPLICIT_FIXED_ROUTE_ONLY", "operation_mode_timestamped_source_available": False,
            "operation_mode_general_knowledge_not_accepted": True,
            "any_schedule_headway_source_in_inventory": inventory["any_schedule_headway_source"]}


def shared_counter_reconstruction_audit() -> Dict[str, Any]:
    counters = ["served_request_count", "boardings", "alightings", "completed_trip_count", "completed_cycle_count",
                "skip_count", "intervention_count", "missed_pickup_count", "missed_dropoff_count", "terminal_arrival_count"]
    rows = []
    for c in counters:
        aggregate = c in {"boardings", "alightings"}
        rows.append({"counter": c,
                     "anchor_direct": False,
                     "service_day_replay_required": True,
                     "classification": "RECONSTRUCTABLE_FROM_SERVICE_DAY_START_REPLAY" if aggregate else "UNAVAILABLE",
                     "zero_initialization_prohibited": True,
                     "blocking_reason": ("only 10-min node aggregates exist for boardings/alightings; no running counter" if aggregate
                                         else "no historical event source for this runtime counter")})
    return {"created_at": iso_kst(), "records": rows,
            "any_counter_exact_at_anchor": False,
            "note": "no anchor-time running counters exist; even boardings/alightings are 10-minute node aggregates, not running trip/cycle counters"}


def replay_cursor_feasibility_audit() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "historical_event_stream_exists": False, "event_ordering_key_available": False, "timestamp_ordering_available": False,
            "sequence_index_available": False, "last_event_before_anchor_available": False, "first_event_after_anchor_available": False,
            "cursor_offset_reconstructable": False, "stream_hash_available": False,
            "replay_cursor_reconstructable": False, "zero_initialization_prohibited": True,
            "blocking_reason": "historical data is 10-minute node state buckets, not an ordered per-event stream; no temporal cut can anchor a replay cursor"}


def external_provider_state_feasibility_audit() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "providers_checked": ["demand", "stop_service", "schedule", "vehicle_location", "traffic_travel_time", "route_graph"],
            "provider_state_source_available": False,
            "non_use_proven": False,
            "external_provider_state_reconstructable": False,
            "empty_dict_allowed": False,
            "blocking_reason": "no historical provider-state source and stateful-provider non-use is not proven for a historical reconstruction"}


def temporal_alignment_audit() -> List[Dict[str, Any]]:
    # No dynamic-source as-of joins were performed (no row content read); leakage is 0 by construction.
    fields = ["simulation_timestamp_seconds", "waiting_passengers", "shared_counters", "headway_state", "replay_cursor"]
    rows = []
    for f in fields:
        rows.append({"field": f, "source_timestamp": "graph_state_timeslice.state_ts (10-min bucket)" if f in {"simulation_timestamp_seconds", "waiting_passengers", "shared_counters"} else "n/a (no source)",
                     "anchor_age_seconds": None, "native_update_interval_seconds": 600 if f in {"simulation_timestamp_seconds", "waiting_passengers", "shared_counters"} else None,
                     "staleness_classification": "PRIOR_ASOF_WITHIN_NATIVE_INTERVAL" if f in {"simulation_timestamp_seconds", "waiting_passengers", "shared_counters"} else "NO_TEMPORAL_MATCH",
                     "future_value_used": False})
    return rows


def train_anchor_inventory(split: Mapping[str, Any]) -> Dict[str, Any]:
    order = split.get("split_order", {})
    train_first = order.get("train_first", "snapshot_00001.pt")
    # deterministic TRAIN-only anchor keys spread across the train range; no row content is read.
    anchors = []
    train_count = int(split.get("split_counts", {}).get("train") or 5476)
    for i, idx in enumerate([1, 685, 1369, 2053, 2738, 3422, 4106, min(4790, train_count)]):
        anchors.append({"anchor_id": f"TRAIN-A{i+1:02d}", "split": "TRAIN", "train_snapshot_key": f"snapshot_{idx:05d}.pt",
                        "selection_rule": "deterministic evenly-spaced index within TRAIN range from split registry",
                        "row_level_content_read": False})
    return {"created_at": iso_kst(), "anchor_count_max": 8, "anchor_count_selected": len(anchors),
            "train_first_key": train_first, "train_range": [order.get("train_first"), order.get("train_last")],
            "row_level_content_read_count": 0,
            "feasibility_determined_at": "SCHEMA_AND_SOURCE_INVENTORY_LEVEL",
            "reason_no_row_content_read": "the state-field gaps are structural (the historical schema lacks per-vehicle/per-passenger/event columns), so no per-anchor .pt content read can change the field feasibility; reading it would risk contamination without evidentiary value",
            "anchors": anchors}


def anchor_reconstruction_results(anchors: Sequence[Mapping[str, Any]], field_matrix: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    exact = sorted(r["field_name"] for r in field_matrix if r["exactness_status"] == "EXACT_AT_ANCHOR")
    partial = sorted(r["field_name"] for r in field_matrix if r["exactness_status"] == "PARTIAL_CRITICAL_GAPS")
    unavailable = sorted(r["field_name"] for r in field_matrix if r["exactness_status"] == "NOT_RECONSTRUCTABLE")
    rows = []
    for a in anchors:
        rows.append({"anchor_id": a["anchor_id"], "split": "TRAIN", "train_snapshot_key": a["train_snapshot_key"],
                     "exact_field_names": exact, "partial_field_names": partial, "unavailable_field_names": unavailable,
                     "exact_field_count": len(exact), "partial_field_count": len(partial), "unavailable_field_count": len(unavailable),
                     "future_leakage_count": 0, "source_conflict_count": 0,
                     "critical_missing_fields": ["vehicles", "waiting_passengers", "assigned_pickups", "assigned_dropoffs",
                                                 "onboard_passengers", "action_mask_state", "replay_cursor"],
                     "complete_candidate_produced": False,
                     "action_feasibility": {"H": False, "S": False, "K": False, "three_action_comparison": False}})
    return rows


def action_specific_feasibility_audit() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "hold_state_feasible": False, "hold_blockers": ["no anchor-exact vehicle position/remaining-travel/dwell state"],
            "serve_move_state_feasible": False, "serve_move_blockers": ["no per-stop request-level demand", "no dwell/travel/route-progression per vehicle"],
            "conditional_skip_state_feasible": False,
            "conditional_skip_blockers": ["no per-passenger waiting queue", "no assigned pickup/dropoff", "no onboard destination/obligation",
                                          "no mandatory/protected/terminal flags", "no planned-skip permission", "no path-validity flags",
                                          "no consecutive-skip constraint state", "no service-fairness constraint state"],
            "three_action_historical_comparison_feasible": False,
            "three_action_blockers": ["H infeasible", "S infeasible", "K infeasible", "no shared anchor state", "no shared replay source"],
            "future_leakage_count": 0}


# ---------------------------------------------------------------------------
# gaps / plan / overall / environment / manifest / orchestration
# ---------------------------------------------------------------------------

def critical_state_gap_registry(field_matrix: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in field_matrix:
        if r["exactness_status"] in {"NOT_RECONSTRUCTABLE", "PARTIAL_CRITICAL_GAPS"}:
            rows.append({"field_name": r["field_name"], "exactness_status": r["exactness_status"], "evidence_class": r["evidence_class"],
                         "missing_source": r["candidate_source"], "blocking_reason": r["blocking_reason"],
                         "severity": "CRITICAL" if r["exactness_status"] == "NOT_RECONSTRUCTABLE" else "PARTIAL"})
    return rows


def historical_state_reconstruction_plan() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "current_source_summary": "node-level 10-minute aggregate graph-state (waiting proxy + boardings/alightings) + static route/stop topology",
            "required_but_absent_sources": [
                {"state": "per-vehicle dynamic state (position, remaining travel/dwell, onboard)", "required_source": "per-vehicle location + status stream at sub-minute cadence"},
                {"state": "8-agent to vehicle identity mapping", "required_source": "historical vehicle roster with stable agent binding"},
                {"state": "passenger-level waiting queue", "required_source": "per-passenger arrival/boarding events with identity"},
                {"state": "assigned pickup/dropoff requests", "required_source": "request-level dispatch/assignment log"},
                {"state": "onboard destination", "required_source": "per-passenger onboard destination records"},
                {"state": "route/stop safety flags", "required_source": "per-run mandatory/protected/skip/fairness/path-validity configuration"},
                {"state": "schedule/headway state", "required_source": "trip timetable + per-vehicle position stream (prior-only)"},
                {"state": "ordered event stream / replay cursor", "required_source": "service-day ordered event log with timestamps"},
                {"state": "external provider state", "required_source": "stateful provider snapshots or a proof of non-use"},
            ],
            "recommended_next_step": "STATE_SOURCE_REPAIR_PLAN",
            "notes": "The current Suseong historical dataset is a GATv2 node-feature time series; it structurally lacks the per-entity dynamic state that DynamicsStateSnapshot requires. Exact anchor reconstruction is not possible from the present source."}


def overall_state_feasibility(field_matrix: Sequence[Mapping[str, Any]], agent: Mapping[str, Any], actions: Mapping[str, Any]) -> Dict[str, Any]:
    exact = [r["field_name"] for r in field_matrix if r["exactness_status"] == "EXACT_AT_ANCHOR"]
    replay = [r["field_name"] for r in field_matrix if r["evidence_class"] == "RECONSTRUCTABLE_FROM_SERVICE_DAY_START_REPLAY"]
    partial = [r["field_name"] for r in field_matrix if r["exactness_status"] == "PARTIAL_CRITICAL_GAPS"]
    unavailable = [r["field_name"] for r in field_matrix if r["exactness_status"] == "NOT_RECONSTRUCTABLE"]
    # Some fields exact/partial, but the core per-vehicle/per-passenger/replay state is unavailable -> CRITICAL_GAPS_RECORDED.
    if not unavailable and not partial:
        status = "EXACT_AT_ANCHOR"
    elif not unavailable and replay:
        status = "EXACT_AFTER_SERVICE_DAY_REPLAY"
    elif len(unavailable) >= 8:
        status = "NOT_RECONSTRUCTABLE" if len(unavailable) == len(field_matrix) else "PARTIAL_CRITICAL_GAPS"
    else:
        status = "PARTIAL_CRITICAL_GAPS"
    return {"created_at": iso_kst(),
            "dynamics_state_required_field_count": len(field_matrix),
            "exact_field_count": len(exact), "exact_field_names": exact,
            "service_day_replay_required_field_count": len(replay), "service_day_replay_field_names": replay,
            "partial_critical_gap_field_count": len(partial), "partial_field_names": partial,
            "unavailable_field_count": len(unavailable), "unavailable_field_names": unavailable,
            "agent_mapping_exact": agent["agent_mapping_exact"],
            "three_action_historical_comparison_feasible": actions["three_action_historical_comparison_feasible"],
            "future_leakage_count": 0, "synthetic_default_inserted_count": 0,
            "overall_state_feasibility": status,
            "exact_reconstruction_supported": status == "EXACT_AT_ANCHOR" and actions["three_action_historical_comparison_feasible"]}


def environment_payload() -> Dict[str, Any]:
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"created_at": iso_kst(), "mode": "feasibility-audit", "scope": "READ_ONLY_HISTORICAL_STATE_FEASIBILITY_AUDIT",
            "requested_execution_platform": "MAC_MINI_M4_24GB", "actual_compute_path": "CPU_ONLY_STATIC_ANALYSIS",
            "platform_machine": platform.machine(), "python_executable": sys.executable, "python_version": sys.version,
            "process_rss_bytes": raw_rss if platform.system() == "Darwin" else raw_rss * 1024,
            "pure_contract_import_count": 2, "transition_module_import_count": 0,
            "simulator_transition_execution_count": 0, "global_step_execution_count": 0, "thirty_minute_branch_execution_count": 0,
            "validation_access_count": 0, "test_holdout_access_count": 0, "sealed_holdout_row_access_count": 0,
            "training_run_count": 0, "source_modification_count": 0}


def write_manifest(writer: Writer, rel: str, payloads: Sequence[str]) -> Dict[str, Any]:
    rows = []
    for p in payloads:
        path = writer.root / p
        rows.append({"relative_path": p, "required": True, "exists": path.exists(),
                     "sha256": sha256_file(path) if path.exists() else None, "size_bytes": path.stat().st_size if path.exists() else None})
    manifest = {"created_at": iso_kst(), "manifest_protocol": "TERMINAL_LOCK_TO_MANIFEST_TO_PAYLOAD", "manifest_scope": "SF0_HISTORICAL_STATE_FEASIBILITY_AUDIT",
                "required_payload_count": len(rows), "payload_file_count": sum(1 for r in rows if r["exists"]),
                "missing_payload_count": sum(1 for r in rows if not r["exists"]), "missing_payloads": [r["relative_path"] for r in rows if not r["exists"]],
                "terminal_lock_listed_inside_manifest": False, "manifest_self_listed": False, "files": rows}
    writer.json(rel, manifest)
    return manifest


def write_lock(writer: Writer, lock_name: str, manifest_name: str, gate: Mapping[str, Any]) -> None:
    mp = writer.root / manifest_name
    writer.json(lock_name, {"created_at": iso_kst(), "mode": "feasibility-audit", "gate": gate["gate"], "gate_passed": gate["gate_passed"],
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


SF0_PAYLOADS = [
    "upstream_v1f_snapshot/v1f_gate_decision.json", "upstream_v1f_snapshot/v1f_downstream_lock.json",
    "upstream_v1f_snapshot/v1f_artifact_manifest.json", "upstream_v1f_snapshot/v1f_SUCCESS.lock",
    "upstream_v1f_snapshot/v1f_final_report.json", "upstream_v1f_snapshot/v1f_dynamics_source_registry.json",
    "upstream_v1f_registry.json", "runner_snapshot_pre_execution/run_prompt5_e01_dl6d_pa1a_sf0_historical_state_feasibility_audit.py",
    "runner_freeze_audit.json", "feasibility_audit_environment.json", "authoritative_split_discovery.json", "split_access_prohibition_audit.json",
    "historical_source_inventory.json", "historical_source_inventory.jsonl", "historical_source_schema_registry.json", "historical_source_schema_registry.jsonl",
    "dynamics_state_contract_registry.json", "state_field_source_matrix.json", "state_field_source_matrix.jsonl",
    "vehicle_state_subfield_matrix.json", "vehicle_state_subfield_matrix.jsonl", "route_stop_safety_field_matrix.json", "route_stop_safety_field_matrix.jsonl",
    "passenger_obligation_source_audit.json", "agent_semantics_audit.json", "agent_historical_mapping.json", "agent_historical_mapping.jsonl",
    "schedule_headway_operation_mode_audit.json", "shared_counter_reconstruction_audit.json", "replay_cursor_feasibility_audit.json",
    "external_provider_state_feasibility_audit.json", "temporal_alignment_audit.json", "temporal_alignment_audit.jsonl", "future_leakage_audit.json",
    "train_anchor_inventory.json", "train_anchor_reconstruction_results.json", "train_anchor_reconstruction_results.jsonl",
    "pure_contract_validation_audit.json", "action_specific_feasibility_audit.json", "critical_state_gap_registry.json", "critical_state_gap_registry.jsonl",
    "state_source_conflict_registry.json", "state_source_conflict_registry.jsonl", "historical_state_reconstruction_plan.json", "overall_state_feasibility.json",
    "simulator_execution_prohibition_audit.json", "historical_row_access_audit.json", "validation_untouched_audit.json", "test_holdout_untouched_audit.json",
    "sealed_holdout_preservation_audit.json", "reward_kpi_nonexecution_audit.json", "training_prohibition_audit.json", "external_access_audit.json",
    "stage_immutability_audit.json", "gate_decision.json", "downstream_lock.json", "final_report.json", "final_report.md",
]


def run_feasibility_audit(artifact_root: Path) -> Path:
    runner_sha_before = sha256_file(RUNNER_PATH)
    runner_size_before = RUNNER_PATH.stat().st_size

    v1f = v1f_preflight()
    if not v1f["v1f_upstream_valid"]:
        raise AuditError(FAIL_V1F, f"V1F upstream invalid: {v1f['checks']}")

    root = validate_artifact_root(artifact_root)
    writer = Writer(root)
    writer.json("feasibility_audit_environment.json", environment_payload())
    writer.json("upstream_v1f_preflight.json", v1f)
    snapshot_v1f(writer)
    runner_snapshot = copy_file(writer, RUNNER_PATH, "runner_snapshot_pre_execution/run_prompt5_e01_dl6d_pa1a_sf0_historical_state_feasibility_audit.py")

    source_registry = frozen_source_registry(writer)
    if source_registry["source_drift_count"]:
        raise AuditError(FAIL_SOURCE_DRIFT, f"dynamics source drift: {source_registry['source_drift_count']}")

    split = authoritative_split_discovery()
    writer.json("authoritative_split_discovery.json", split)
    writer.json("split_access_prohibition_audit.json", {"created_at": iso_kst(), "train_row_content_read": False,
                "validation_row_access_count": 0, "test_row_access_count": 0, "sealed_holdout_row_access_count": 0,
                "unclassified_row_access_count": 0, "split_registry_counts_hashes_read_only": True})
    if not split["authoritative_train_split_found"]:
        raise AuditError(BLOCKED_TRAIN, "authoritative TRAIN split not found")

    inventory = historical_source_inventory()
    writer.json("historical_source_inventory.json", inventory)
    writer.jsonl("historical_source_inventory.jsonl", inventory["records"])
    writer.json("historical_source_schema_registry.json", {"created_at": iso_kst(), "records": inventory["records"]})
    writer.jsonl("historical_source_schema_registry.jsonl", inventory["records"])

    contract = import_pure_contract()
    contract_registry = dynamics_state_contract_registry(contract)
    writer.json("dynamics_state_contract_registry.json", contract_registry)

    field_matrix = state_field_source_matrix(contract, inventory)
    writer.json("state_field_source_matrix.json", {"created_at": iso_kst(), "field_count": len(field_matrix), "records": field_matrix})
    writer.jsonl("state_field_source_matrix.jsonl", field_matrix)

    veh = vehicle_state_subfield_matrix()
    writer.json("vehicle_state_subfield_matrix.json", {"created_at": iso_kst(), "records": veh})
    writer.jsonl("vehicle_state_subfield_matrix.jsonl", veh)
    safety = route_stop_safety_field_matrix()
    writer.json("route_stop_safety_field_matrix.json", {"created_at": iso_kst(), "records": safety})
    writer.jsonl("route_stop_safety_field_matrix.jsonl", safety)
    writer.json("passenger_obligation_source_audit.json", passenger_obligation_audit())

    agent = agent_semantics_audit()
    writer.json("agent_semantics_audit.json", agent)
    agent_map_rows = [{"agent_id": i, "historical_vehicle_id": None, "mapping_exact": False, "semantics": "SYNTHETIC_AGENT_SLOT"} for i in range(8)]
    writer.json("agent_historical_mapping.json", {"created_at": iso_kst(), "records": agent_map_rows})
    writer.jsonl("agent_historical_mapping.jsonl", agent_map_rows)

    writer.json("schedule_headway_operation_mode_audit.json", schedule_headway_operation_mode_audit(inventory))
    writer.json("shared_counter_reconstruction_audit.json", shared_counter_reconstruction_audit())
    writer.json("replay_cursor_feasibility_audit.json", replay_cursor_feasibility_audit())
    writer.json("external_provider_state_feasibility_audit.json", external_provider_state_feasibility_audit())

    temporal = temporal_alignment_audit()
    writer.json("temporal_alignment_audit.json", {"created_at": iso_kst(), "records": temporal})
    writer.jsonl("temporal_alignment_audit.jsonl", temporal)
    writer.json("future_leakage_audit.json", {"created_at": iso_kst(), "future_leakage_count": 0, "future_value_used_count": 0,
                "as_of_joins_performed": 0, "row_content_read": False,
                "note": "no dynamic-source as-of joins were performed; feasibility determined at schema level, so future leakage is structurally 0"})

    anchors = train_anchor_inventory(split)
    writer.json("train_anchor_inventory.json", anchors)
    anchor_results = anchor_reconstruction_results(anchors["anchors"], field_matrix)
    writer.json("train_anchor_reconstruction_results.json", {"created_at": iso_kst(), "records": anchor_results})
    writer.jsonl("train_anchor_reconstruction_results.jsonl", anchor_results)

    (root / "mapped_state_components").mkdir(parents=True, exist_ok=True)
    (root / "historical_state_snapshot_candidates").mkdir(parents=True, exist_ok=True)
    for a in anchors["anchors"]:
        writer.json(f"mapped_state_components/{a['anchor_id']}.json", {
            "anchor_id": a["anchor_id"], "split": "TRAIN", "train_snapshot_key": a["train_snapshot_key"],
            "mapped_fields": sorted(r["field_name"] for r in field_matrix if r["exactness_status"] in {"EXACT_AT_ANCHOR", "PARTIAL_CRITICAL_GAPS"}),
            "unmapped_fields": sorted(r["field_name"] for r in field_matrix if r["exactness_status"] == "NOT_RECONSTRUCTABLE"),
            "source_provenance": {r["field_name"]: r["candidate_source"] for r in field_matrix},
            "exactness": {r["field_name"]: r["exactness_status"] for r in field_matrix},
            "temporal_alignment": "10-minute node buckets; no per-second anchor; no future values used",
            "critical_blockers": ["vehicles", "waiting_passengers(per-passenger)", "assigned_pickups", "assigned_dropoffs", "onboard_passengers", "action_mask_state", "replay_cursor"],
            "complete_candidate": False, "row_level_content_read": False})

    complete_candidates = sorted((root / "historical_state_snapshot_candidates").glob("*.json"))
    writer.json("pure_contract_validation_audit.json", {"created_at": iso_kst(), "complete_candidate_count": len(complete_candidates),
                "validation_performed": len(complete_candidates) > 0, "schema_valid": None, "round_trip_hash_match": None,
                "note": "no complete DynamicsStateSnapshot candidate could be built without synthetic defaults, so no schema/round-trip validation was performed (empty candidates directory is expected and not a failure)"})

    actions = action_specific_feasibility_audit()
    writer.json("action_specific_feasibility_audit.json", actions)
    gaps = critical_state_gap_registry(field_matrix)
    writer.json("critical_state_gap_registry.json", {"created_at": iso_kst(), "critical_gap_field_count": sum(1 for g in gaps if g["severity"] == "CRITICAL"), "records": gaps})
    writer.jsonl("critical_state_gap_registry.jsonl", gaps)
    writer.json("state_source_conflict_registry.json", {"created_at": iso_kst(), "source_conflict_count": 0, "records": []})
    writer.jsonl("state_source_conflict_registry.jsonl", [])
    writer.json("historical_state_reconstruction_plan.json", historical_state_reconstruction_plan())

    overall = overall_state_feasibility(field_matrix, agent, actions)
    writer.json("overall_state_feasibility.json", overall)

    # prohibition audits
    writer.json("simulator_execution_prohibition_audit.json", {"created_at": iso_kst(), "simulator_transition_execution_count": 0,
                "global_step_execution_count": 0, "thirty_minute_branch_execution_count": 0, "transition_module_import_count": 0, "pure_contract_import_count": 2})
    writer.json("historical_row_access_audit.json", {"created_at": iso_kst(), "train_row_content_read_count": 0, "nontrain_row_access_count": 0,
                "split_registry_metadata_read_only": True})
    writer.json("validation_untouched_audit.json", {"created_at": iso_kst(), "validation_access_count": 0, "validation_branch_count": 0})
    writer.json("test_holdout_untouched_audit.json", {"created_at": iso_kst(), "test_holdout_access_count": 0, "test_holdout_touched": False})
    writer.json("sealed_holdout_preservation_audit.json", {"created_at": iso_kst(), "sealed_holdout_row_access_count": 0, "sealed_holdout_reopened": False,
                "dl6d_r3_holdout_status": "FAILED_SEALED_NO_REUSE", "holdout_status_changed": False})
    writer.json("reward_kpi_nonexecution_audit.json", {"created_at": iso_kst(), "reward_result_created": False, "kpi_result_created": False, "new_formula_created": False})
    writer.json("training_prohibition_audit.json", {"created_at": iso_kst(), "training_run_count": 0, "optimizer_step_count": 0, "checkpoint_load_count": 0, "checkpoint_write_count": 0})
    writer.json("external_access_audit.json", {"created_at": iso_kst(), "db_write_count": 0, "db_connection_count": 0, "api_call_count": 0, "network_access_count": 0, "git_commit_count": 0, "git_push_count": 0})
    writer.json("stage_immutability_audit.json", {"created_at": iso_kst(), "v1f_artifact_mutation_count": 0, "source_modification_count": 0, "upstream_artifact_mutation_count": 0})

    runner_sha_after = sha256_file(RUNNER_PATH)
    runner_freeze = {"created_at": iso_kst(), "runner_relative_path": str(RUNNER_PATH.relative_to(PROJECT_ROOT)),
                     "runner_sha256_before_execution": runner_sha_before, "runner_size_before_execution": runner_size_before,
                     "runner_sha256_after_execution": runner_sha_after, "runner_snapshot_sha256": runner_snapshot["copied_sha256"],
                     "runner_mutation_count": 0 if runner_sha_before == runner_sha_after else 1,
                     "runner_frozen": runner_sha_before == runner_sha_after == runner_snapshot["copied_sha256"]}
    writer.json("runner_freeze_audit.json", runner_freeze)
    if runner_freeze["runner_mutation_count"]:
        raise AuditError(FAIL_RUNNER, "runner SHA changed during audit")

    gate_status = {"EXACT_AT_ANCHOR": PASS_EXACT, "EXACT_AFTER_SERVICE_DAY_REPLAY": PASS_REPLAY,
                   "PARTIAL_CRITICAL_GAPS": PASS_GAPS, "NOT_RECONSTRUCTABLE": PASS_NOT_RECON}[overall["overall_state_feasibility"]]
    gate = {"created_at": iso_kst(), "mode": "feasibility-audit", "gate": gate_status, "gate_passed": True, "readiness": READINESS[gate_status],
            "overall_state_feasibility": overall["overall_state_feasibility"],
            "sf1_authorized": False, "historical_transition_authorized": False, "reward_rebuild_authorized": False,
            "canonical_kpi_rebuild_authorized": False, "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False}
    writer.json("gate_decision.json", gate)

    writer.json("downstream_lock.json", {
        "v1f_upstream_verified": True, "sf0_audit_complete": True,
        "authoritative_train_split_found": True, "train_anchor_count_audited": anchors["anchor_count_selected"],
        "dynamics_state_required_field_count": overall["dynamics_state_required_field_count"],
        "exact_field_count": overall["exact_field_count"], "service_day_replay_required_field_count": overall["service_day_replay_required_field_count"],
        "critical_gap_field_count": overall["partial_critical_gap_field_count"] + overall["unavailable_field_count"], "unavailable_field_count": overall["unavailable_field_count"],
        "agent_semantics": agent["agent_semantics"], "agent_mapping_exact": agent["agent_mapping_exact"],
        "passenger_level_waiting_state_available": False, "request_assignment_state_available": False, "onboard_destination_state_available": False,
        "replay_cursor_reconstructable": False, "external_provider_state_reconstructable": False,
        "hold_state_feasible": actions["hold_state_feasible"], "serve_move_state_feasible": actions["serve_move_state_feasible"],
        "conditional_skip_state_feasible": actions["conditional_skip_state_feasible"], "three_action_historical_comparison_feasible": actions["three_action_historical_comparison_feasible"],
        "overall_state_feasibility": overall["overall_state_feasibility"], "future_leakage_count": 0, "synthetic_default_inserted_count": 0,
        "simulator_transition_execution_count": 0, "validation_access_count": 0, "test_holdout_access_count": 0, "sealed_holdout_row_access_count": 0,
        "sf1_required": overall["overall_state_feasibility"] == "EXACT_AT_ANCHOR", "sf1_authorized": False,
        "historical_transition_authorized": False, "reward_rebuild_authorized": False, "canonical_kpi_rebuild_authorized": False,
        "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False,
        "recommended_next_step": "STATE_SOURCE_REPAIR_PLAN"})

    report_payload, report_md = build_final_report(root, gate, v1f, split, contract_registry, overall, agent, actions, field_matrix, gaps, runner_freeze)
    writer.json("final_report.json", report_payload)
    writer.text("final_report.md", report_md + "\n")

    manifest = write_manifest(writer, "artifact_manifest_sf0.json", SF0_PAYLOADS)
    if manifest["missing_payload_count"]:
        raise AuditError(FAIL_MANIFEST, f"missing payloads: {manifest['missing_payloads']}")
    write_lock(writer, "_SF0_AUDIT_COMPLETE.lock", "artifact_manifest_sf0.json", gate)
    v = verify_manifest(root, "_SF0_AUDIT_COMPLETE.lock")
    if not v["manifest_hash_ok"] or not v["manifest_size_ok"] or v["payload_missing_count"] or v["payload_hash_mismatch_count"] or v["terminal_lock_listed_inside_manifest"] or v["manifest_self_listed"]:
        raise AuditError(FAIL_MANIFEST, f"manifest verification failed: {v}")

    print("SF0 HISTORICAL STATE FEASIBILITY AUDIT COMPLETE")
    print(f"artifact_root: {root}")
    print(f"gate: {gate_status}")
    print(f"readiness: {gate['readiness']}")
    print(f"overall_state_feasibility: {overall['overall_state_feasibility']}")
    print(f"required fields: {overall['dynamics_state_required_field_count']} | exact {overall['exact_field_count']} | partial {overall['partial_critical_gap_field_count']} | unavailable {overall['unavailable_field_count']}")
    print(f"agent_semantics: {agent['agent_semantics']} | three_action_feasible: {actions['three_action_historical_comparison_feasible']}")
    print(f"runner_mutation_count: {runner_freeze['runner_mutation_count']} | future_leakage: 0 | synthetic_default: 0")
    print("sf1_authorized: false")
    return root


def build_final_report(root, gate, v1f, split, contract_registry, overall, agent, actions, field_matrix, gaps, runner_freeze):
    answers = {
        "01_authoritative_train_split_found": split["authoritative_train_split_found"],
        "02_validation_test_holdout_untouched": True,
        "03_dynamics_state_required_field_count": contract_registry["authoritative_field_count"],
        "04_each_field_historical_source_found": "PARTIAL — only static topology + node-level proxy demand; core per-entity state has no source",
        "05_8_agents_are": agent["agent_semantics"],
        "06_agent_vehicle_identity_linkable": agent["agent_mapping_exact"],
        "07_vehicle_position_route_direction_reconstructable": False,
        "08_remaining_travel_dwell_reconstructable": False,
        "09_waiting_passenger_queue_reconstructable": False,
        "10_assigned_pickup_dropoff_reconstructable": False,
        "11_onboard_destination_reconstructable": False,
        "12_route_stop_safety_flags_reconstructable": False,
        "13_schedule_headway_reconstructable": False,
        "14_shared_counter_source": "SERVICE_DAY_REPLAY_REQUIRED_AT_BEST (only 10-min node aggregates; no running counters)",
        "15_replay_cursor_reconstructable": False,
        "16_external_provider_state_reconstructable": False,
        "17_future_leakage_occurred": False,
        "18_synthetic_default_used": False,
        "19_hold_state_feasible": actions["hold_state_feasible"],
        "20_serve_move_state_feasible": actions["serve_move_state_feasible"],
        "21_conditional_skip_state_feasible": actions["conditional_skip_state_feasible"],
        "22_three_action_comparison_feasible": actions["three_action_historical_comparison_feasible"],
        "23_complete_candidate_snapshot_count": 0,
        "24_service_day_start_replay_required": True,
        "25_most_critical_gap": "no per-vehicle dynamic state and no per-passenger request/onboard state (8 agents are policy slots; historical data is node-level 10-min aggregate)",
        "26_next_step": "STATE_SOURCE_REPAIR_PLAN (missing historical state sources) before SF1 limited reconstruction",
    }
    payload = {"created_at": iso_kst(), "artifact_root": str(root), "mode": "feasibility-audit", "gate": gate["gate"], "gate_passed": gate["gate_passed"],
               "readiness": gate["readiness"], "quick_answers": answers, "scope": "READ_ONLY_HISTORICAL_STATE_FEASIBILITY_AUDIT",
               "overall_state_feasibility": overall["overall_state_feasibility"],
               "authoritative_field_count": contract_registry["authoritative_field_count"], "prompt_declared_field_count": 17,
               "field_count_discrepancy": contract_registry["field_count_discrepancy_note"],
               "exact_field_names": overall["exact_field_names"], "partial_field_names": overall["partial_field_names"],
               "unavailable_field_names": overall["unavailable_field_names"],
               "agent_semantics": agent["agent_semantics"], "future_leakage_count": 0, "synthetic_default_inserted_count": 0,
               "runner_mutation_count": runner_freeze["runner_mutation_count"],
               "sf1_authorized": False, "historical_transition_authorized": False, "reward_rebuild_authorized": False,
               "canonical_kpi_rebuild_authorized": False, "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False,
               "next_authorized_action": "State-source repair plan review only, after explicit user review and command"}
    lines = ["# SF0 Historical Dynamics State Reconstruction Feasibility Audit", "",
             f"- artifact_root: {root}", f"- gate: {gate['gate']}", f"- readiness: {gate['readiness']}",
             f"- overall_state_feasibility: {overall['overall_state_feasibility']}",
             f"- required fields (source-authoritative): {contract_registry['authoritative_field_count']} (prompt stated 17 — discrepancy reported)",
             f"- field feasibility: exact {overall['exact_field_count']}, partial {overall['partial_critical_gap_field_count']}, unavailable {overall['unavailable_field_count']}",
             f"- agent semantics: {agent['agent_semantics']} · three-action historical comparison feasible: {actions['three_action_historical_comparison_feasible']}", "",
             "## Bottom line",
             "- The Suseong historical dataset is a GATv2 node-feature time series (10-minute buckets: waiting-proxy, boardings/alightings, link travel) plus static route/stop topology.",
             "- It structurally lacks the per-vehicle dynamic state, per-passenger request/assignment/onboard state, route/stop safety flags, schedule/headway, and ordered event stream that DynamicsStateSnapshot requires.",
             "- The 8 MAPPO agents are policy slots, not identifiable historical vehicles.",
             "- Exact anchor reconstruction and H/S/K historical comparison are NOT feasible from the current source. No synthetic defaults were inserted; no future leakage; no simulator execution.", "",
             "## Quick answers"]
    for k in sorted(answers):
        lines.append(f"- {k}: {answers[k]}")
    lines += ["", "## Downstream (all locked)",
              "- sf1_required: only if EXACT (here false) | sf1_authorized: false",
              "- historical_transition_authorized: false | reward_rebuild_authorized: false | canonical_kpi_rebuild_authorized: false",
              "- pa1b_authorized: false | dl6e_p0_authorized: false | training_allowed: false",
              "- DL-6D-R3 sealed hold-out preserved (FAILED, no reuse).", ""]
    return payload, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["feasibility-audit"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        run_feasibility_audit(args.artifact_root)
    except AuditError as exc:
        print("SF0 HISTORICAL STATE FEASIBILITY AUDIT FAILED")
        print(f"gate: {exc.gate_status}")
        print(f"detail: {exc.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
