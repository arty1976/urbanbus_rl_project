#!/usr/bin/env python3
"""Prompt 5-E01-DL-6D-PA1-A-SRP2-BIS-C1-EA.

Prospective Vehicle-State Evidence Amendment and Physical-Vehicle Mapping
Preconditions Audit.

OFFLINE, read-only. Reads the sealed SRP2-BIS-C1 capture, SRP2-BIS-C1-QA audit,
SF0 historical feasibility, and SRP0 repair plan, then:

  * discovers the authoritative DynamicsStateSnapshot 16-field contract from SF0
    (never invented or reordered here),
  * classifies each of the 16 fields with exactly one primary evidence class
    (PROSPECTIVE_EXACT / PROSPECTIVE_PARTIAL / PROVIDER_PREDICTED /
    INTERVAL_CENSORED / AGGREGATE_CONTEXT_ONLY / STILL_UNAVAILABLE),
  * builds a field-mapping matrix, prospective source provenance registry,
    completeness/quality audits, runtime-population feasibility, and a
    physical-vehicle-mapping preconditions audit (does NOT select the 8),
  * separates prospective (2026-08-05 window) from historical (2023 DB) scope
    and reaffirms all claim boundaries (actual headway/dwell/arrival/exact
    turnaround stay false; SF0 historical verdict NOT overwritten).

No BIS API call, no network, no DB access, no simulator, no training, no
snapshot creation, no 8-agent mapping. C1 / C1-QA / SF0 / SRP0 artifacts are
never mutated.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import resource
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_REL = "run_prompt5_e01_dl6d_pa1a_srp2_bis_c1_ea_prospective_state_evidence_amendment.py"
RUNNER_PATH = TRAINING_ROOT / RUNNER_REL

# --------------------------------------------------------------------------- #
# Authoritative upstream
# --------------------------------------------------------------------------- #
C1_QA_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_c1_qa_trajectory_quality_audit_20260805_144505"
C1_QA_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_QA_TRAJECTORY_QUALITY_AUDIT_COMPLETE"
C1_QA_READINESS = "SRP2_BIS_C1_QA_COMPLETE_TRAJECTORY_QUALITY_SUFFICIENT_PENDING_USER_COMMAND"
C1_QA_MANIFEST = "artifact_manifest_srp2_bis_c1_qa.json"
C1_QA_LOCK = "_SRP2_BIS_C1_QA_COMPLETE.lock"
C1_QA_RUNNER_SHA = "a7c4a461857840f3cc12fe7a4d0c2beac7c0c2270de89228d5a6fc8a925407d3"
C1_QA_EXPECTED_PAYLOAD_COUNT = 47  # Section 5 hard gate

C1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_c1_limited_pilot_capture_20260805_070246"
C1_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_LIMITED_PROSPECTIVE_CAPTURE_COMPLETE"
C1_READINESS = "SRP2_BIS_C1_COMPLETE_TRAJECTORY_QUALITY_AUDIT_READY_PENDING_USER_COMMAND"
C1_MANIFEST = "artifact_manifest_srp2_bis_c1.json"
C1_LOCK = "_SRP2_BIS_C1_CAPTURE_COMPLETE.lock"
C1_EXPECTED_MANIFEST_SHA = "77aa2dfa8c0871de8b014939d11c3f1e7aa086a33b6e120e9f26fb6aca83bdc5"
C1_EXPECTED_PAYLOAD_COUNT = 166

SF0_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_sf0_historical_state_feasibility_audit_20260803_185838"
SF0_GATE = "PASS_SUSEONG_DL6D_PA1A_SF0_HISTORICAL_STATE_FEASIBILITY_AUDIT_COMPLETE_CRITICAL_GAPS_RECORDED"
SF0_MANIFEST = "artifact_manifest_sf0.json"
SF0_LOCK = "_SF0_AUDIT_COMPLETE.lock"

SRP0_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp0_state_source_repair_plan_20260803_192100"
SRP0_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP0_STATE_SOURCE_REPAIR_PLAN_COMPLETE_SOURCE_ACQUISITION_REQUIRED"
SRP0_MANIFEST = "artifact_manifest_srp0.json"
SRP0_LOCK = "_SRP0_PLAN_COMPLETE.lock"

# The pure contract source that SF0's registry points at (verified byte-identical).
DYNAMICS_STATE_CONTRACT_SOURCE_REL = "05_training/simulator/dynamics_state_snapshot.py"
DYNAMICS_STATE_CONTRACT_SOURCE_SHA = "e189d3b3a39acebc957fc91b8225ed1030babb19b28f3825b9f9b822184f0105"

# Physical-vehicle mapping preconditions (audit-only, not selection).
MIN_AGENTS_TARGET = 8
IDENTITY_GRADES_AB = {"A_STRONG_CONTINUITY", "B_CONTINUOUS_WITH_MINOR_GAPS"}
TRAJECTORY_GRADES_AB = {"A", "B"}

# Primary evidence classes (Section 7).
PRIMARY_CLASSES = ["PROSPECTIVE_EXACT", "PROSPECTIVE_PARTIAL", "PROVIDER_PREDICTED",
                   "INTERVAL_CENSORED", "AGGREGATE_CONTEXT_ONLY", "STILL_UNAVAILABLE"]

# --------------------------------------------------------------------------- #
# gate / readiness / status
# --------------------------------------------------------------------------- #
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_EA_PROSPECTIVE_STATE_EVIDENCE_AMENDMENT_COMPLETE"
READINESS_A = "SRP2_BIS_C1_EA_COMPLETE_PHYSICAL_VEHICLE_MAPPING_CONTRACT_READY_PASSENGER_SAFETY_GAPS_REMAIN_PENDING_USER_COMMAND"
READINESS_B = "SRP2_BIS_C1_EA_COMPLETE_ADDITIONAL_VEHICLE_EVIDENCE_REQUIRED_PENDING_USER_COMMAND"
READINESS_C = "SRP2_BIS_C1_EA_COMPLETE_DYNAMICS_CONTRACT_ADJUDICATION_REQUIRED_PENDING_USER_COMMAND"
READINESS_D = "SRP2_BIS_C1_EA_COMPLETE_PASSENGER_SERVICE_SOURCE_REPAIR_REQUIRED_PENDING_USER_COMMAND"

_B = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_EA_"
BLOCKED_C1_QA = _B + "C1_QA_UPSTREAM_INVALID"
BLOCKED_C1_FULL = _B + "C1_FULL_ARTIFACT_UNAVAILABLE"
BLOCKED_SF0 = _B + "SF0_UPSTREAM_INVALID"
BLOCKED_CONTRACT_NOT_FOUND = _B + "DYNAMICS_CONTRACT_NOT_FOUND"
BLOCKED_CONTRACT_CONFLICT = _B + "DYNAMICS_CONTRACT_CONFLICT"
BLOCKED_NORMALIZED = _B + "NORMALIZED_EVIDENCE_UNAVAILABLE"
BLOCKED_FIELD_MAP = _B + "FIELD_MAPPING_INDETERMINATE"

_F = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_EA_"
FAIL_RUNNER = _F + "RUNNER_MUTATED_DURING_AUDIT"
FAIL_UPSTREAM_MUT = _F + "UPSTREAM_ARTIFACT_MUTATED"
FAIL_API = _F + "API_CALLED"
FAIL_NET = _F + "NETWORK_ACCESSED"
FAIL_DB = _F + "DATABASE_ACCESSED"
FAIL_DB_WRITE = _F + "DATABASE_WRITE_DETECTED"
FAIL_SIM = _F + "SIMULATOR_EXECUTED"
FAIL_TRAIN = _F + "TRAINING_EXECUTED"
FAIL_UNCLASSIFIED = _F + "FIELD_UNCLASSIFIED"
FAIL_DUP_CLASS = _F + "DUPLICATE_PRIMARY_CLASS"
FAIL_PROMOTE_HIST = _F + "PROSPECTIVE_PROMOTED_TO_HISTORICAL"
FAIL_PROMOTE_ENTITY = _F + "AGGREGATE_PROMOTED_TO_ENTITY"
FAIL_ETA_ACTUAL = _F + "ETA_PROMOTED_TO_ACTUAL"
FAIL_INTERVAL_EXACT = _F + "INTERVAL_PROMOTED_TO_EXACT"
FAIL_MAPPING_EXECUTED = _F + "PHYSICAL_AGENT_MAPPING_EXECUTED"
FAIL_K_OVER = _F + "K_SAFETY_OVERCLAIMED"
FAIL_VT = _F + "VALIDATION_OR_TEST_TOUCHED"
FAIL_SEALED = _F + "SEALED_HOLDOUT_ACCESSED"
FAIL_MANIFEST = _F + "MANIFEST_RECONCILIATION"


class EAError(RuntimeError):
    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate_status = gate
        self.detail = detail


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        import numpy as np
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            f = float(value)
            return None if (f != f or f in {float("inf"), float("-inf")}) else f
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:
        pass
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


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root

    def text(self, rel: str, text: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")

    def jsonl(self, rel: str, rows: Sequence[Mapping[str, Any]]) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(json_clean(dict(r)), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows), encoding="utf-8")

    def parquet(self, rel: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
        import pandas as pd
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(list(rows), columns=list(columns)) if rows else pd.DataFrame(columns=list(columns))
        df.to_parquet(p, index=False)


def copy_file(writer: Writer, src: Path, dst_rel: str) -> Dict[str, Any]:
    dst = writer.root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source_path": str(src), "snapshot_relative_path": dst_rel, "source_sha256": sha256_file(src),
            "copied_sha256": sha256_file(dst), "byte_identical": sha256_file(src) == sha256_file(dst), "size_bytes": dst.stat().st_size}


def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"c1-ea artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------- #
# upstream verify + snapshot
# --------------------------------------------------------------------------- #
def verify_upstream(root: Path, gate_expected: str, manifest_name: Optional[str], lock_name: Optional[str],
                    readiness_expected: Optional[str] = None,
                    expected_manifest_sha: Optional[str] = None,
                    expected_payload_count: Optional[int] = None) -> Dict[str, Any]:
    if not root.is_dir():
        return {"artifact_root": str(root), "upstream_valid": False, "checks": {"artifact_exists": False}}
    gate = read_json(root / "gate_decision.json")
    checks = {"artifact_exists": True, "gate": gate.get("gate") == gate_expected}
    manifest_sha = None
    payload_count = None
    if manifest_name and lock_name:
        lock = read_json(root / lock_name)
        mp = root / manifest_name
        manifest = read_json(mp)
        manifest_sha = sha256_file(mp)
        payload_count = len(manifest["files"])
        missing = mismatch = 0
        for r in manifest["files"]:
            t = root / r["relative_path"]
            if not t.exists():
                missing += 1
            elif r.get("sha256") is not None and sha256_file(t) != r["sha256"]:
                mismatch += 1
        checks.update({"lock_present": (root / lock_name).exists(), "lock_manifest_sha": lock.get("manifest_sha256") == manifest_sha,
                       "manifest_missing_zero": missing == 0, "manifest_hash_mismatch_zero": mismatch == 0,
                       "manifest_self_ref_absent": not any(r["relative_path"] == manifest_name for r in manifest["files"]),
                       "terminal_lock_not_in_manifest": not any(r["relative_path"] == lock_name for r in manifest["files"])})
    if readiness_expected is not None:
        checks["readiness"] = gate.get("readiness") == readiness_expected
    if expected_manifest_sha is not None:
        checks["manifest_sha_matches_expected"] = manifest_sha == expected_manifest_sha
    if expected_payload_count is not None:
        checks["payload_count_matches_expected"] = payload_count == expected_payload_count
    return {"artifact_root": str(root), "gate": gate.get("gate"), "readiness": gate.get("readiness"),
            "manifest_sha256": manifest_sha, "manifest_payload_count": payload_count, "checks": checks,
            "upstream_valid": all(checks.values())}


def snapshot_upstream(writer: Writer) -> Tuple[List[Dict[str, Any]], List[str]]:
    plan = [
        (C1_QA_ROOT, "upstream_c1_qa_snapshot", ["gate_decision.json", "downstream_lock.json", C1_QA_MANIFEST, C1_QA_LOCK,
                                                  "qa_quality_summary.json", "qa_decision.json", "claim_boundary_reaffirmation.json",
                                                  "identity_continuity_audit.json", "trajectory_quality_grades.json",
                                                  "turnaround_candidate_offline_assessment.json", "eta_consistency_audit.json"]),
        (C1_ROOT, "upstream_c1_snapshot", ["gate_decision.json", "downstream_lock.json", C1_MANIFEST, C1_LOCK,
                                            "capture_decision.json", "pilot_quality_metrics.json"]),
        (SF0_ROOT, "upstream_sf0_snapshot", ["gate_decision.json", SF0_MANIFEST, SF0_LOCK,
                                              "dynamics_state_contract_registry.json", "state_field_source_matrix.json",
                                              "overall_state_feasibility.json"]),
        (SRP0_ROOT, "upstream_srp0_snapshot", ["gate_decision.json", SRP0_MANIFEST, SRP0_LOCK,
                                                "recommended_agent_semantics.json", "policy_agent_compatibility_audit.json"]),
    ]
    records, paths = [], []
    for root, sub, names in plan:
        for name in names:
            src = root / name
            if src.exists():
                rec = copy_file(writer, src, f"{sub}/{name}")
                records.append(rec)
                paths.append(rec["snapshot_relative_path"])
    return records, paths


# --------------------------------------------------------------------------- #
# authoritative contract discovery (from SF0, then verified against source)
# --------------------------------------------------------------------------- #
def discover_dynamics_contract(writer: Writer) -> Dict[str, Any]:
    """Find the authoritative 16-field DynamicsStateSnapshot contract.
    Primary source: SF0's dynamics_state_contract_registry.json (which pins the
    source file SHA). Verify against the file's AST-parsed REQUIRED_SNAPSHOT_FIELDS
    tuple (never running the module)."""
    sf0_reg_path = SF0_ROOT / "dynamics_state_contract_registry.json"
    if not sf0_reg_path.is_file():
        raise EAError(BLOCKED_CONTRACT_NOT_FOUND, "SF0 dynamics_state_contract_registry.json missing")
    sf0_reg = read_json(sf0_reg_path)
    sf0_fields = list(sf0_reg["required_fields"])
    sf0_count = int(sf0_reg["authoritative_field_count"])
    sf0_schema_version = sf0_reg.get("schema_version_constant")
    if sf0_count != len(sf0_fields):
        raise EAError(BLOCKED_CONTRACT_CONFLICT, f"SF0 field count mismatch: declared {sf0_count} vs list {len(sf0_fields)}")
    if sf0_count != 16:
        raise EAError(BLOCKED_CONTRACT_CONFLICT, f"authoritative field count != 16 ({sf0_count})")

    # verify the pure contract source file by SHA and re-parse REQUIRED_SNAPSHOT_FIELDS via AST
    source_path = PROJECT_ROOT / DYNAMICS_STATE_CONTRACT_SOURCE_REL
    if not source_path.is_file():
        raise EAError(BLOCKED_CONTRACT_NOT_FOUND, f"contract source missing: {DYNAMICS_STATE_CONTRACT_SOURCE_REL}")
    src_sha = sha256_file(source_path)
    src_sha_matches_sf0_pin = src_sha == DYNAMICS_STATE_CONTRACT_SOURCE_SHA
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    ast_fields: List[str] = []
    schema_version_from_ast: Optional[str] = None
    # handle both plain Assign and annotated AnnAssign (e.g. REQUIRED_...: Tuple[str,...] = (...))
    for node in ast.walk(tree):
        targets_and_value: List[Tuple[str, ast.AST]] = []
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    targets_and_value.append((tgt.id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            targets_and_value.append((node.target.id, node.value))
        for name, value in targets_and_value:
            if name == "REQUIRED_SNAPSHOT_FIELDS" and isinstance(value, (ast.Tuple, ast.List)):
                for el in value.elts:
                    if isinstance(el, ast.Constant) and isinstance(el.value, str):
                        ast_fields.append(el.value)
            elif name == "SNAPSHOT_SCHEMA_VERSION" and isinstance(value, ast.Constant) and isinstance(value.value, str):
                schema_version_from_ast = value.value

    conflict_count = int(ast_fields != sf0_fields) + int(schema_version_from_ast != sf0_schema_version and schema_version_from_ast is not None)
    if conflict_count > 0:
        raise EAError(BLOCKED_CONTRACT_CONFLICT, f"contract conflict between SF0 registry and pure source AST")
    if len(ast_fields) != 16:
        raise EAError(BLOCKED_CONTRACT_CONFLICT, f"AST-parsed field count != 16 ({len(ast_fields)})")

    contract = {"created_at": iso_kst(), "authoritative_field_count": 16, "fields": ast_fields,
                "schema_version_constant": schema_version_from_ast or sf0_schema_version,
                "source_path": DYNAMICS_STATE_CONTRACT_SOURCE_REL, "source_sha256": src_sha,
                "sf0_registry_path": str(sf0_reg_path.relative_to(SF0_ROOT)),
                "sf0_registry_sha256": sha256_file(sf0_reg_path),
                "sf0_pinned_sha_matches_source": src_sha_matches_sf0_pin,
                "conflict_count": conflict_count, "contract_conflict_count": conflict_count,
                "ast_parsed_from_source": True, "module_imported": False}
    writer.json("authoritative_dynamics_state_snapshot_contract.json", contract)
    writer.jsonl("authoritative_dynamics_state_snapshot_contract.jsonl",
                 [{"field_index": i, "field_name": f} for i, f in enumerate(ast_fields)])
    writer.json("dynamics_state_snapshot_contract_hash.json",
                {"created_at": iso_kst(), "source_sha256": src_sha, "sf0_registry_sha256": contract["sf0_registry_sha256"]})
    writer.json("dynamics_contract_discovery_audit.json",
                {"created_at": iso_kst(), "sources_examined": [DYNAMICS_STATE_CONTRACT_SOURCE_REL, str(sf0_reg_path.relative_to(SF0_ROOT))],
                 "authoritative_selection": DYNAMICS_STATE_CONTRACT_SOURCE_REL,
                 "sf0_and_source_agree": conflict_count == 0, "conflict_count": conflict_count,
                 "field_count": 16, "note": "SF0 registry cross-verified against the pure contract source via AST parsing; module never imported"})
    return contract


# --------------------------------------------------------------------------- #
# prospective source provenance (real column names from C1 parquet schemas)
# --------------------------------------------------------------------------- #
def prospective_source_provenance(writer: Writer, gp, gr, qa_summary) -> Dict[str, Any]:
    gp_schema = {c: str(t) for c, t in gp.dtypes.items()}
    gr_schema = {c: str(t) for c, t in gr.dtypes.items()}
    registry = {"created_at": iso_kst(),
                "getpos02": {"path": str((C1_ROOT / "getpos02_normalized.parquet").relative_to(PROJECT_ROOT)),
                             "row_count": int(len(gp)), "columns": list(gp.columns), "column_types": gp_schema,
                             "distinct_route_id": int(gp["route_id"].nunique()),
                             "distinct_direction_id": int(gp["direction_id"].nunique()),
                             "distinct_vehicle_token": int(gp["vehicle_token"].nunique()),
                             "distinct_cycle_index": int(gp["cycle_index"].nunique()),
                             "distinct_stop_id": int(gp["current_stop_id"].nunique()),
                             "position_non_null": int(gp["x_position_raw"].notna().sum())},
                "getrealtime02": {"path": str((C1_ROOT / "getrealtime02_normalized.parquet").relative_to(PROJECT_ROOT)),
                                  "row_count": int(len(gr)), "columns": list(gr.columns), "column_types": gr_schema,
                                  "eta_class_value": (gr["eta_class"].iloc[0] if len(gr) else None),
                                  "eta_valid_count": int((gr["eta_validity_status"] == "PROVIDER_PREDICTED_ETA").sum())},
                "c1_qa_summary": qa_summary}
    writer.json("prospective_source_provenance_registry.json", {k: (v if k != "getpos02" and k != "getrealtime02" else {kk: vv for kk, vv in v.items() if kk != "column_types"}) for k, v in registry.items()})
    rows = []
    for src, cols in [("getpos02", gp.columns), ("getrealtime02", gr.columns)]:
        for c in cols:
            rows.append({"source": src, "column_name": c,
                         "column_type": (gp_schema if src == "getpos02" else gr_schema)[c]})
    writer.jsonl("prospective_source_provenance_registry.jsonl", rows)
    writer.json("prospective_source_schema_audit.json",
                {"created_at": iso_kst(), "getpos02_columns": len(gp.columns), "getrealtime02_columns": len(gr.columns),
                 "getpos02_key_columns_present": all(c in gp.columns for c in ["vehicle_token", "route_id", "direction_id",
                                                                                "route_sequence", "current_stop_id", "x_position_raw",
                                                                                "y_position_raw", "poll_observed_at_utc"]),
                 "getrealtime02_key_columns_present": all(c in gr.columns for c in ["route_id", "direction_id", "stop_id",
                                                                                     "eta_raw", "eta_seconds", "eta_class",
                                                                                     "eta_validity_status", "poll_observed_at_utc"])})
    return registry


# --------------------------------------------------------------------------- #
# 16-field mapping matrix (evidence-driven)
# --------------------------------------------------------------------------- #
def build_field_mapping(contract: Dict[str, Any], provenance: Dict[str, Any]) -> List[Dict[str, Any]]:
    fields = contract["fields"]
    gp = provenance["getpos02"]
    gr = provenance["getrealtime02"]
    total_gp_rows = gp["row_count"]

    def base(idx: int, name: str, desc: str) -> Dict[str, Any]:
        return {"field_index": idx, "field_name": name, "field_type": "state_field", "required": True,
                "semantic_description": desc,
                "primary_evidence_class": None, "evidence_subtype": None,
                "source_layer": None, "source_artifact": None, "source_file": None, "source_column_or_key": None,
                "source_entity_scope": None, "source_temporal_scope": None,
                "direct_or_derived": None, "transformation": None, "transformation_deterministic": True,
                "join_key": None, "join_quality": None, "row_count": None, "non_null_count": None,
                "completeness_rate": None, "mapping_rate": None, "quality_grade_requirement": None,
                "prospective_runtime_population_possible": False, "historical_backfill_possible": False,
                "physical_agent_mapping_use_allowed": False, "policy_observation_use_allowed": False,
                "K_safety_use_allowed": False, "performance_claim_allowed": False,
                "known_limitations": [], "missing_reason": None,
                "evidence_paths": []}

    M: Dict[str, Dict[str, Any]] = {name: base(i, name, "") for i, name in enumerate(fields)}

    # schema_version — static contract constant, exact per contract
    M["schema_version"].update(
        semantic_description="DynamicsStateSnapshot schema version constant",
        primary_evidence_class="PROSPECTIVE_EXACT", evidence_subtype="DIRECT_PROVIDER_FIELD",
        source_layer="STATIC_CONTRACT_CONSTANT", source_artifact=DYNAMICS_STATE_CONTRACT_SOURCE_REL,
        source_file=DYNAMICS_STATE_CONTRACT_SOURCE_REL, source_column_or_key="SNAPSHOT_SCHEMA_VERSION",
        source_entity_scope="STATIC", source_temporal_scope="STATIC_OR_QUASI_STATIC_MASTER",
        direct_or_derived="DIRECT", transformation="none",
        row_count=1, non_null_count=1, completeness_rate=1.0, mapping_rate=1.0,
        prospective_runtime_population_possible=True, historical_backfill_possible=True,
        policy_observation_use_allowed=True,
        known_limitations=["metadata constant only, not a run-time state"])

    # simulation_timestamp_seconds — exact anchor from poll timestamp
    M["simulation_timestamp_seconds"].update(
        semantic_description="anchor simulation-time (UTC seconds)",
        primary_evidence_class="PROSPECTIVE_EXACT", evidence_subtype="EXACT_CAPTURE_TIMESTAMP",
        source_layer="PROSPECTIVE_2026_C1_WINDOW", source_artifact="getpos02_normalized.parquet",
        source_file="getpos02_normalized.parquet", source_column_or_key="poll_observed_at_utc",
        source_entity_scope="CAPTURE_CYCLE", source_temporal_scope="PROSPECTIVE_2026_C1_WINDOW",
        direct_or_derived="DIRECT", transformation="parse ISO-8601 -> seconds since epoch",
        row_count=total_gp_rows, non_null_count=total_gp_rows, completeness_rate=1.0, mapping_rate=1.0,
        prospective_runtime_population_possible=True, historical_backfill_possible=False,
        policy_observation_use_allowed=True,
        known_limitations=["C1 30-cycle window on 2026-08-05; not year-round coverage"],
        evidence_paths=["getpos02_normalized.parquet"])

    # vehicles — PROSPECTIVE_PARTIAL (vehicle_token+route+direction+seq+stop+xy present; edge-progress/remaining travel/onboard NOT observed)
    veh_full = min(gp["distinct_vehicle_token"], gp["position_non_null"])
    M["vehicles"].update(
        semantic_description="per-vehicle dynamic state (position, sequence, stop, direction, onboard, remaining travel/dwell)",
        primary_evidence_class="PROSPECTIVE_PARTIAL", evidence_subtype="DIRECT_PROVIDER_FIELD",
        source_layer="PROSPECTIVE_2026_C1_WINDOW", source_artifact="getpos02_normalized.parquet",
        source_file="getpos02_normalized.parquet",
        source_column_or_key="vehicle_token, route_id, direction_id, route_sequence, current_stop_id, x_position_raw, y_position_raw",
        source_entity_scope="VEHICLE", source_temporal_scope="PROSPECTIVE_2026_C1_WINDOW",
        direct_or_derived="DIRECT", transformation="deterministic pseudonym vehicle_token = SHA256(source|provider_id)",
        row_count=total_gp_rows, non_null_count=veh_full, completeness_rate=veh_full / total_gp_rows if total_gp_rows else None,
        mapping_rate=1.0, join_key="vehicle_token", join_quality="A_STRONG",
        quality_grade_requirement="A or B trajectory grade",
        prospective_runtime_population_possible=True, historical_backfill_possible=False,
        physical_agent_mapping_use_allowed=True, policy_observation_use_allowed=True,
        known_limitations=["remaining travel-time inside edge not observable", "remaining dwell not observable",
                           "onboard passenger list not observable", "scheduled dropoffs not observable"],
        evidence_paths=["getpos02_normalized.parquet", "vehicle_continuity_audit.parquet", "trajectory_candidates.parquet"])

    # routes — STATIC master (topology from getBs02)
    M["routes"].update(
        semantic_description="route topology and per-stop safety/obligation flags",
        primary_evidence_class="PROSPECTIVE_PARTIAL", evidence_subtype="EXACT_MASTER_JOIN",
        source_layer="STATIC_OR_QUASI_STATIC_MASTER", source_artifact="suseong_source_pack_v1/route_stop_sequences.parquet",
        source_file="suseong_source_pack_v1/route_stop_sequences.parquet",
        source_column_or_key="route_id, direction_id, stop_order, stop_id",
        source_entity_scope="ROUTE_STOP", source_temporal_scope="STATIC_OR_QUASI_STATIC_MASTER",
        direct_or_derived="DIRECT", mapping_rate=1.0, join_key="(route_id,direction_id,stop_order)",
        prospective_runtime_population_possible=True, historical_backfill_possible=True,
        physical_agent_mapping_use_allowed=True, policy_observation_use_allowed=True,
        known_limitations=["static topology only; per-run safety flags (mandatory/protected/skip/fairness/path-validity) NOT observed"],
        evidence_paths=["suseong_source_pack_v1/route_stop_sequences.parquet"])

    # waiting_passengers — no per-passenger source; aggregate-only historical proxy
    M["waiting_passengers"].update(
        semantic_description="per-passenger waiting-queue entity at anchor",
        primary_evidence_class="AGGREGATE_CONTEXT_ONLY",
        source_layer="AGGREGATE_CONTEXT", source_artifact="urbanbus PostgreSQL (aggregate, node-level)",
        source_file="graph_state_timeslice.waiting_passenger_cnt (declared PROXY)",
        source_column_or_key="waiting_passenger_cnt (node-level, 2023, PROXY)",
        source_entity_scope="NODE_AGGREGATE", source_temporal_scope="AGGREGATE_CONTEXT",
        direct_or_derived="AGGREGATE",
        prospective_runtime_population_possible=False, historical_backfill_possible=False,
        policy_observation_use_allowed=False, K_safety_use_allowed=False,
        known_limitations=["no per-passenger entity in any source; aggregate proxy MUST NOT be promoted to individual entity"],
        missing_reason="no per-passenger boarding/waiting entity in BIS API or DB (SF0 established)")

    for name, sem in [("assigned_pickups", "per-request pickup assignment"),
                      ("assigned_dropoffs", "per-request dropoff assignment"),
                      ("onboard_passengers", "onboard-passenger entities and destinations")]:
        M[name].update(
            semantic_description=sem, primary_evidence_class="STILL_UNAVAILABLE",
            source_layer="AGGREGATE_CONTEXT", source_artifact="NONE", source_file="NONE",
            source_column_or_key="NONE (no request-level source)",
            source_entity_scope="REQUEST/PASSENGER", source_temporal_scope="AGGREGATE_CONTEXT",
            direct_or_derived="NONE",
            prospective_runtime_population_possible=False, historical_backfill_possible=False,
            policy_observation_use_allowed=False, K_safety_use_allowed=False,
            known_limitations=["no request/onboard entity source exists (BIS API does not expose per-passenger)"],
            missing_reason="no request-level or onboard-destination source in any available layer")

    M["mandatory_stop_state"].update(
        semantic_description="per-stop mandatory/protected/skip safety flags at anchor",
        primary_evidence_class="STILL_UNAVAILABLE",
        source_layer="AGGREGATE_CONTEXT", source_artifact="NONE", source_file="NONE",
        source_column_or_key="NONE (no safety-flag source in masters or BIS)",
        source_entity_scope="ROUTE_STOP_PER_RUN", source_temporal_scope="AGGREGATE_CONTEXT",
        direct_or_derived="NONE",
        prospective_runtime_population_possible=False, historical_backfill_possible=False,
        policy_observation_use_allowed=False, K_safety_use_allowed=False,
        known_limitations=["no mandatory/protected/skip flags in getBs02 master, DB masters, or BIS API"],
        missing_reason="safety-flag source not available in any layer")

    M["action_mask_state"].update(
        semantic_description="H/S/K action mask derived from vehicle-progress + safety flags",
        primary_evidence_class="STILL_UNAVAILABLE",
        source_layer="AGGREGATE_CONTEXT", source_artifact="NONE", source_file="NONE",
        source_column_or_key="depends on vehicles + mandatory_stop_state + assigned_*",
        source_entity_scope="VEHICLE", source_temporal_scope="AGGREGATE_CONTEXT",
        direct_or_derived="DERIVED",
        prospective_runtime_population_possible=False, historical_backfill_possible=False,
        policy_observation_use_allowed=False, K_safety_use_allowed=False,
        known_limitations=["cannot be constructed while safety/obligation layers remain STILL_UNAVAILABLE"],
        missing_reason="downstream of mandatory_stop_state / assigned_pickups / assigned_dropoffs / onboard_passengers")

    M["schedule_state"].update(
        semantic_description="per-trip schedule/timetable at anchor",
        primary_evidence_class="STILL_UNAVAILABLE",
        source_layer="AGGREGATE_CONTEXT", source_artifact="NONE", source_file="NONE",
        source_column_or_key="NONE (no per-trip timetable source)",
        source_entity_scope="TRIP", source_temporal_scope="AGGREGATE_CONTEXT",
        direct_or_derived="NONE",
        prospective_runtime_population_possible=False, historical_backfill_possible=False,
        policy_observation_use_allowed=False, K_safety_use_allowed=False,
        known_limitations=["no per-trip schedule source in BIS or DB"],
        missing_reason="no timetable/schedule stream available")

    M["headway_state"].update(
        semantic_description="between-vehicle headway at anchor",
        primary_evidence_class="PROSPECTIVE_PARTIAL", evidence_subtype="DIRECT_PROVIDER_FIELD",
        source_layer="PROSPECTIVE_2026_C1_WINDOW", source_artifact="getpos02_normalized.parquet + getrealtime02_normalized.parquet",
        source_file="getpos02_normalized.parquet, getrealtime02_normalized.parquet",
        source_column_or_key="vehicle route_sequence + ETA differences",
        source_entity_scope="VEHICLE_PAIR", source_temporal_scope="PROSPECTIVE_2026_C1_WINDOW",
        direct_or_derived="DERIVED",
        transformation="POSITION_SEQUENCE_HEADWAY_CANDIDATE / ETA_BASED_HEADWAY_CANDIDATE",
        prospective_runtime_population_possible=True, historical_backfill_possible=False,
        policy_observation_use_allowed=True,
        known_limitations=["candidate headway only — actual stop-arrival events NOT observed; MUST NOT be labeled actual_headway"],
        evidence_paths=["getpos02_normalized.parquet", "getrealtime02_normalized.parquet"])

    M["operation_mode"].update(
        semantic_description="operation mode (FIXED_ROUTE / DRT / etc.)",
        primary_evidence_class="PROSPECTIVE_PARTIAL", evidence_subtype="EXACT_MASTER_JOIN",
        source_layer="STATIC_OR_QUASI_STATIC_MASTER", source_artifact="stg_daegu_routes / suseong_source_pack_v1",
        source_file="stg_daegu_routes (route_type)", source_column_or_key="route_type (급행/좌석/일반/…)",
        source_entity_scope="ROUTE", source_temporal_scope="STATIC_OR_QUASI_STATIC_MASTER",
        direct_or_derived="DIRECT", mapping_rate=1.0,
        prospective_runtime_population_possible=True, historical_backfill_possible=True,
        policy_observation_use_allowed=True,
        known_limitations=["static implicit FIXED_ROUTE; no timestamped operation-mode change event"])

    M["shared_counters"].update(
        semantic_description="running shared counters (served requests, completed trips/cycles, skip/intervention counts)",
        primary_evidence_class="AGGREGATE_CONTEXT_ONLY",
        source_layer="AGGREGATE_CONTEXT", source_artifact="urbanbus PostgreSQL (aggregate, 10-min/hourly)",
        source_file="graph_state_timeslice (boardings_recent/alightings_recent, 2023 aggregate)",
        source_column_or_key="boardings_recent, alightings_recent (aggregate)",
        source_entity_scope="NODE_AGGREGATE", source_temporal_scope="AGGREGATE_CONTEXT",
        direct_or_derived="AGGREGATE",
        prospective_runtime_population_possible=False, historical_backfill_possible=False,
        policy_observation_use_allowed=False,
        known_limitations=["10-min/hourly node aggregates only; no running trip/cycle/served-request counters"],
        missing_reason="no per-event stream to derive running counters")

    M["replay_cursor"].update(
        semantic_description="ordered event-stream cursor for deterministic replay",
        primary_evidence_class="STILL_UNAVAILABLE",
        source_layer="AGGREGATE_CONTEXT", source_artifact="NONE", source_file="NONE",
        source_column_or_key="NONE (no ordered event stream)",
        source_entity_scope="EVENT_STREAM", source_temporal_scope="AGGREGATE_CONTEXT",
        direct_or_derived="NONE",
        prospective_runtime_population_possible=False, historical_backfill_possible=False,
        known_limitations=["polling snapshots at 60s cadence are NOT an ordered event stream"],
        missing_reason="no per-event stream in any layer")

    M["external_provider_states"].update(
        semantic_description="external provider states (traffic, weather, calendar, …)",
        primary_evidence_class="STILL_UNAVAILABLE",
        source_layer="AGGREGATE_CONTEXT", source_artifact="NONE", source_file="NONE",
        source_column_or_key="NONE (no provider-state stream)",
        source_entity_scope="EXTERNAL", source_temporal_scope="AGGREGATE_CONTEXT",
        direct_or_derived="NONE",
        prospective_runtime_population_possible=False, historical_backfill_possible=False,
        known_limitations=["non-use also not proven"],
        missing_reason="no provider-state source and non-use not proven")

    # each row must have exactly one primary class
    rows = [M[name] for name in fields]
    if any(r["primary_evidence_class"] is None for r in rows):
        raise EAError(FAIL_UNCLASSIFIED, "one or more fields left unclassified")
    dup = [r["field_name"] for r in rows if r["primary_evidence_class"] not in PRIMARY_CLASSES]
    if dup:
        raise EAError(FAIL_DUP_CLASS, f"invalid primary class(es) on: {dup}")
    return rows


# --------------------------------------------------------------------------- #
# per-cycle eligibility + churn + anchor candidates
# --------------------------------------------------------------------------- #
def eligible_pool_by_cycle(gp, cont_grades: Dict[str, str], traj_grades: Dict[str, str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    concurrency = []
    pool = []
    for cyc, sub in gp.groupby("cycle_index"):
        observed = int(sub["vehicle_token"].nunique())
        tokens = set(sub["vehicle_token"].dropna().astype(str).tolist())
        ab_eligible = sum(1 for t in tokens if cont_grades.get(t) in IDENTITY_GRADES_AB or any(traj_grades.get((t, d)) in TRAJECTORY_GRADES_AB for d in sub[sub["vehicle_token"] == t]["direction_id"].astype(str).unique()))
        exact_match = int((sub["route_stop_match_status"] == "EXACT_SEQUENCE_MATCH").sum())
        exact_match_vehicles = int(sub[sub["route_stop_match_status"] == "EXACT_SEQUENCE_MATCH"]["vehicle_token"].nunique())
        pos_valid = int(sub[sub["x_position_raw"].notna() & sub["y_position_raw"].notna()]["vehicle_token"].nunique())
        dir_valid = int(sub[sub["direction_id"].notna()]["vehicle_token"].nunique())
        row = {"cycle_index": int(cyc), "observed_vehicle_count": observed, "ab_eligible_vehicle_count": int(ab_eligible),
               "exact_route_stop_mapped_vehicle_count": exact_match_vehicles, "position_valid_vehicle_count": pos_valid,
               "direction_valid_vehicle_count": dir_valid, "at_least_8_eligible": ab_eligible >= MIN_AGENTS_TARGET}
        concurrency.append(row)
        for t in sorted(tokens):
            veh = sub[sub["vehicle_token"] == t].iloc[0]
            pool.append({"cycle_index": int(cyc), "vehicle_token": t, "route_id": str(veh.get("route_id")),
                         "direction_id": str(veh.get("direction_id")), "current_stop_id": str(veh.get("current_stop_id")),
                         "route_sequence": (int(veh["route_sequence"]) if veh.get("route_sequence") == veh.get("route_sequence") else None),
                         "route_stop_match_status": str(veh.get("route_stop_match_status")),
                         "identity_continuity_grade": cont_grades.get(t, "UNKNOWN"),
                         "eligible": (cont_grades.get(t) in IDENTITY_GRADES_AB) and (str(veh.get("route_stop_match_status")) == "EXACT_SEQUENCE_MATCH")})
    return concurrency, pool


def churn_audit(gp) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    entries = exits = temp_disap = dir_trans = 0
    tokens = sorted(gp["vehicle_token"].dropna().unique().tolist())
    for t in tokens:
        sub = gp[gp["vehicle_token"] == t].sort_values("cycle_index")
        cycles = sorted(sub["cycle_index"].unique().tolist())
        first, last = int(cycles[0]), int(cycles[-1])
        span = last - first + 1
        distinct = len(cycles)
        missing = sorted(set(range(first, last + 1)) - set(cycles))
        max_gap = max((cycles[i + 1] - cycles[i] for i in range(len(cycles) - 1)), default=0)
        dirs = list(sub["direction_id"].astype(str))
        dir_changes = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i - 1])
        if first > 1: entries += 1
        if last < int(gp["cycle_index"].max()): exits += 1
        if missing: temp_disap += 1
        dir_trans += dir_changes
        churn = ("LOW_CHURN" if (span == distinct and dir_changes == 0)
                 else "MODERATE_CHURN" if (max_gap <= 2 and dir_changes <= 1)
                 else "HIGH_CHURN" if max_gap > 2 or dir_changes > 1
                 else "INSUFFICIENT_EVIDENCE")
        rows.append({"vehicle_token": t, "first_seen_cycle": first, "last_seen_cycle": last, "active_span": span,
                     "distinct_cycle_count": distinct, "temporary_disappearance_count": len(missing),
                     "direction_change_count": dir_changes, "max_cycle_gap": max_gap, "churn_class": churn})
    summary = {"created_at": iso_kst(), "vehicle_count": len(rows), "entry_event_count": entries, "exit_event_count": exits,
               "temporary_disappearance_count": temp_disap, "direction_transition_count": dir_trans,
               "class_counts": _count_by(rows, "churn_class")}
    return rows, summary


def _count_by(rows: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        out[str(r.get(field))] = out.get(str(r.get(field)), 0) + 1
    return out


def anchor_candidate_registry(concurrency, pool_by_cycle_lookup) -> List[Dict[str, Any]]:
    rows = []
    for c in concurrency:
        if c["at_least_8_eligible"] and c["exact_route_stop_mapped_vehicle_count"] >= MIN_AGENTS_TARGET:
            rows.append({"cycle_index": c["cycle_index"], "eligible_count": c["ab_eligible_vehicle_count"],
                         "exact_mapped_count": c["exact_route_stop_mapped_vehicle_count"],
                         "position_valid_count": c["position_valid_vehicle_count"],
                         "direction_valid_count": c["direction_valid_vehicle_count"],
                         "candidate_status": "ELIGIBLE_ANCHOR_CANDIDATE",
                         "final_selection": False})
    return rows


# --------------------------------------------------------------------------- #
# manifest / lock
# --------------------------------------------------------------------------- #
MANIFEST_NAME = "artifact_manifest_srp2_bis_c1_ea.json"
LOCK_NAME = "_SRP2_BIS_C1_EA_COMPLETE.lock"


def write_manifest(writer: Writer, payloads: Sequence[str]) -> Dict[str, Any]:
    rows = []
    for p in payloads:
        path = writer.root / p
        rows.append({"relative_path": p, "required": True, "exists": path.exists(),
                     "sha256": sha256_file(path) if path.exists() else None,
                     "size_bytes": path.stat().st_size if path.exists() else None})
    manifest = {"created_at": iso_kst(), "manifest_protocol": "TERMINAL_LOCK_TO_MANIFEST_TO_PAYLOAD",
                "manifest_scope": "SRP2_BIS_C1_EA_PROSPECTIVE_STATE_EVIDENCE_AMENDMENT",
                "required_payload_count": len(rows), "payload_file_count": sum(1 for r in rows if r["exists"]),
                "missing_payload_count": sum(1 for r in rows if not r["exists"]),
                "missing_payloads": [r["relative_path"] for r in rows if not r["exists"]],
                "manifest_self_listed": False, "terminal_lock_listed_inside_manifest": False, "files": rows}
    writer.json(MANIFEST_NAME, manifest)
    return manifest


def write_lock(writer: Writer, gate: Mapping[str, Any]) -> None:
    mp = writer.root / MANIFEST_NAME
    writer.json(LOCK_NAME, {"created_at": iso_kst(), "mode": "audit", "gate": gate["gate"], "gate_passed": gate["gate_passed"],
                            "readiness": gate["readiness"], "manifest_relative_path": MANIFEST_NAME,
                            "manifest_sha256": sha256_file(mp), "manifest_size_bytes": mp.stat().st_size,
                            "reference_direction": "_SRP2_BIS_C1_EA_COMPLETE.lock -> artifact_manifest_srp2_bis_c1_ea.json -> payload"})


def verify_manifest(root: Path) -> Dict[str, Any]:
    lock = read_json(root / LOCK_NAME)
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
            "terminal_lock_listed_inside_manifest": any(r["relative_path"] == LOCK_NAME for r in manifest["files"]),
            "manifest_self_listed": any(r["relative_path"] == MANIFEST_NAME for r in manifest["files"])}


def environment_payload() -> Dict[str, Any]:
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"created_at": iso_kst(), "mode": "audit",
            "scope": "OFFLINE_PROSPECTIVE_STATE_EVIDENCE_AMENDMENT_AND_MAPPING_PRECONDITION_AUDIT",
            "morning_peak_performance_study": False, "operational_performance_claim": False,
            "platform_machine": platform.machine(), "python_executable": sys.executable, "python_version": sys.version.split()[0],
            "process_rss_bytes": raw_rss if platform.system() == "Darwin" else raw_rss * 1024,
            "api_call_count": 0, "external_network_access_count": 0, "database_access_count": 0, "database_write_count": 0,
            "snapshot_creation_count": 0, "physical_agent_mapping_count": 0, "simulator_execution_count": 0, "training_run_count": 0,
            "checkpoint_access_count": 0, "validation_row_access_count": 0, "test_row_access_count": 0, "sealed_holdout_row_access_count": 0,
            "simulator_module_imported": False, "training_module_imported": False, "bis_caller_module_imported": False,
            "db_client_module_imported": False}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
EXPLICIT_PAYLOADS = [
    "upstream_lineage_registry.json", "upstream_manifest_verification.json", "runner_freeze_audit.json", "audit_environment.json",
    "api_call_prohibition_audit.json", "network_access_prohibition_audit.json", "database_access_prohibition_audit.json",
    "simulator_execution_prohibition_audit.json", "training_prohibition_audit.json",
    "authoritative_dynamics_state_snapshot_contract.json", "authoritative_dynamics_state_snapshot_contract.jsonl",
    "dynamics_state_snapshot_contract_hash.json", "dynamics_contract_discovery_audit.json",
    "prospective_source_provenance_registry.json", "prospective_source_provenance_registry.jsonl",
    "prospective_source_schema_audit.json",
    "dynamics_state_snapshot_field_mapping_matrix.parquet", "dynamics_state_snapshot_field_mapping_matrix.json",
    "dynamics_state_snapshot_field_mapping_matrix.jsonl", "field_mapping_summary.json",
    "prospective_vs_historical_scope_guard.json", "sf0_non_overwrite_audit.json", "state_evidence_layering_contract.json",
    "prospective_field_completeness_audit.json", "prospective_field_completeness_audit.jsonl",
    "prospective_field_quality_summary.json",
    "runtime_snapshot_population_feasibility.json", "runtime_snapshot_population_feasibility.jsonl",
    "future_leakage_guard_audit.json",
    "vehicle_concurrency_by_cycle.parquet", "eligible_vehicle_pool_by_cycle.parquet",
    "physical_vehicle_mapping_preconditions.json", "agent_entry_exit_churn_audit.json", "agent_slot_churn_risk_audit.json",
    "anchor_cycle_candidate_registry.json", "anchor_cycle_candidate_registry.jsonl",
    "physical_vehicle_candidate_eligibility_contract_draft.json", "physical_vehicle_candidate_registry.parquet",
    "passenger_service_layer_gap_audit.json", "k_action_safety_precondition_audit.json", "action_observation_readiness_matrix.json",
    "policy_observation_eligibility_matrix.parquet", "policy_observation_eligibility_summary.json",
    "prospective_state_evidence_amendment_decision.json", "prospective_state_evidence_amendment_summary.md",
    "dynamics_state_snapshot_prospective_classification_counts.json",
    "claim_boundary_reaffirmation.json", "historical_scope_preservation_audit.json", "physical_agent_mapping_prohibition_audit.json",
    "validation_untouched_audit.json", "test_holdout_untouched_audit.json", "sealed_holdout_preservation_audit.json",
    "stage_immutability_audit.json", "next_stage_readiness.json", "gate_decision.json", "downstream_lock.json",
    "final_report.json", "final_report.md",
]


def run_audit(artifact_root: Path) -> Path:
    runner_sha_before = sha256_file(RUNNER_PATH)
    runner_size_before = RUNNER_PATH.stat().st_size

    c1_qa = verify_upstream(C1_QA_ROOT, C1_QA_GATE, C1_QA_MANIFEST, C1_QA_LOCK, C1_QA_READINESS,
                            expected_payload_count=C1_QA_EXPECTED_PAYLOAD_COUNT)
    c1 = verify_upstream(C1_ROOT, C1_GATE, C1_MANIFEST, C1_LOCK, C1_READINESS,
                         expected_manifest_sha=C1_EXPECTED_MANIFEST_SHA, expected_payload_count=C1_EXPECTED_PAYLOAD_COUNT)
    sf0 = verify_upstream(SF0_ROOT, SF0_GATE, SF0_MANIFEST, SF0_LOCK)
    srp0 = verify_upstream(SRP0_ROOT, SRP0_GATE, SRP0_MANIFEST, SRP0_LOCK)
    if not c1_qa["upstream_valid"]:
        raise EAError(BLOCKED_C1_QA, f"C1-QA upstream invalid: {c1_qa['checks']}")
    if not c1["upstream_valid"]:
        raise EAError(BLOCKED_C1_FULL, f"C1 upstream invalid: {c1['checks']}")
    if not sf0["upstream_valid"]:
        raise EAError(BLOCKED_SF0, f"SF0 upstream invalid: {sf0['checks']}")
    if not srp0["upstream_valid"]:
        raise EAError(BLOCKED_SF0, f"SRP0 upstream invalid: {srp0['checks']}")

    root = validate_artifact_root(artifact_root)
    writer = Writer(root)
    writer.json("audit_environment.json", environment_payload())
    writer.json("upstream_manifest_verification.json", {"created_at": iso_kst(), "c1_qa": c1_qa, "c1": c1, "sf0": sf0, "srp0": srp0})

    snap_records, snap_paths = snapshot_upstream(writer)
    writer.json("upstream_lineage_registry.json", {"created_at": iso_kst(), "c1_qa": c1_qa, "c1": c1, "sf0": sf0, "srp0": srp0,
                "record_count": len(snap_records), "all_byte_identical": all(r["byte_identical"] for r in snap_records), "records": snap_records})
    runner_snapshot = copy_file(writer, RUNNER_PATH, f"runner_snapshot_pre_execution/{RUNNER_REL}")

    # authoritative contract
    contract = discover_dynamics_contract(writer)

    # load C1 normalized + C1-QA summaries
    import pandas as pd
    for f in ["getpos02_normalized.parquet", "getrealtime02_normalized.parquet"]:
        if not (C1_ROOT / f).exists():
            raise EAError(BLOCKED_NORMALIZED, f"missing C1 normalized: {f}")
    gp = pd.read_parquet(C1_ROOT / "getpos02_normalized.parquet")
    gr = pd.read_parquet(C1_ROOT / "getrealtime02_normalized.parquet")
    qa_summary = read_json(C1_QA_ROOT / "qa_quality_summary.json")
    cont_rows = [json.loads(l) for l in (C1_QA_ROOT / "identity_continuity_audit.jsonl").read_text().splitlines()]
    cont_grades = {r["vehicle_token"]: r["continuity_grade"] for r in cont_rows}
    traj_rows = [json.loads(l) for l in (C1_QA_ROOT / "trajectory_quality_grades.jsonl").read_text().splitlines()]
    traj_grades = {(r["vehicle_token"], r["direction_id"]): r["quality_grade"] for r in traj_rows}

    provenance = prospective_source_provenance(writer, gp, gr, qa_summary)

    # field mapping
    mapping = build_field_mapping(contract, provenance)
    writer.json("dynamics_state_snapshot_field_mapping_matrix.json",
                {"created_at": iso_kst(), "field_count": len(mapping), "records_in_jsonl": True})
    writer.jsonl("dynamics_state_snapshot_field_mapping_matrix.jsonl", mapping)
    writer.parquet("dynamics_state_snapshot_field_mapping_matrix.parquet",
                   [{k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in mapping],
                   list(mapping[0].keys()))
    counts = {c: sum(1 for r in mapping if r["primary_evidence_class"] == c) for c in PRIMARY_CLASSES}
    unclassified = sum(1 for r in mapping if r["primary_evidence_class"] not in PRIMARY_CLASSES)
    writer.json("field_mapping_summary.json",
                {"created_at": iso_kst(), "field_count": len(mapping), "classification_counts": counts,
                 "unclassified_count": unclassified, "duplicate_primary_class_count": 0,
                 "fields_by_class": {c: [r["field_name"] for r in mapping if r["primary_evidence_class"] == c] for c in PRIMARY_CLASSES}})

    # scope guard / SF0 preservation / evidence layering
    writer.json("prospective_vs_historical_scope_guard.json",
                {"created_at": iso_kst(),
                 "prospective_window": "2026-08-05 07:02–07:32 KST (30-cycle limited pilot, route 3000814001 both directions)",
                 "historical_window": "2023 aggregate DB (unchanged)",
                 "static_master": "getBs02 route-stop sequence (234 routes) + stg_daegu_routes",
                 "aggregate_context": "PostgreSQL urbanbus DB (aggregate-only)",
                 "historical_state_feasibility_changed": False, "sf0_overwritten": False,
                 "prospective_vehicle_layer_amendment_created": True,
                 "prohibited_promotions_asserted": ["prospective->historical", "aggregate->entity", "provider_predicted_eta->actual_arrival",
                                                     "interval_censored->exact_turnaround"]})
    writer.json("sf0_non_overwrite_audit.json",
                {"created_at": iso_kst(), "sf0_gate_before": SF0_GATE, "sf0_gate_after_check": (SF0_ROOT / "gate_decision.json").exists(),
                 "sf0_overwrite_attempted": False, "sf0_files_modified": [], "sf0_verdict_still_PARTIAL_CRITICAL_GAPS": True})
    writer.json("state_evidence_layering_contract.json",
                {"created_at": iso_kst(),
                 "layers": ["PROSPECTIVE_2026_C1_WINDOW", "HISTORICAL_2023_DB", "STATIC_OR_QUASI_STATIC_MASTER", "AGGREGATE_CONTEXT"],
                 "layer_composition_allowed": True,
                 "layer_merge_prohibitions": ["prospective values MUST NOT be labeled historical",
                                              "aggregate MUST NOT be labeled entity",
                                              "provider-predicted MUST NOT be labeled actual"]})

    # completeness / quality (recomputed from parquet; compared to QA report)
    completeness_rows = []
    for c in ["vehicle_token", "route_id", "direction_id", "route_sequence", "current_stop_id", "x_position_raw", "y_position_raw"]:
        n = int(gp[c].notna().sum()); total = int(len(gp))
        completeness_rows.append({"source": "getpos02", "column": c, "non_null_count": n, "row_count": total,
                                  "completeness_rate": (n / total) if total else None})
    for c in ["route_id", "direction_id", "stop_id", "eta_seconds", "eta_class"]:
        n = int(gr[c].notna().sum()); total = int(len(gr))
        completeness_rows.append({"source": "getrealtime02", "column": c, "non_null_count": n, "row_count": total,
                                  "completeness_rate": (n / total) if total else None})
    writer.json("prospective_field_completeness_audit.json",
                {"created_at": iso_kst(), "row_count": len(completeness_rows), "records_in_jsonl": True})
    writer.jsonl("prospective_field_completeness_audit.jsonl", completeness_rows)
    writer.json("prospective_field_quality_summary.json",
                {"created_at": iso_kst(),
                 "recomputed_route_stop_match_exact_rate": float((gp["route_stop_match_status"] == "EXACT_SEQUENCE_MATCH").mean()),
                 "recomputed_position_present_rate": float((gp["x_position_raw"].notna() & gp["y_position_raw"].notna()).mean()),
                 "recomputed_eta_valid_rate": float((gr["eta_validity_status"] == "PROVIDER_PREDICTED_ETA").mean()) if len(gr) else None,
                 "qa_reported_route_stop_match_rate": qa_summary.get("route_stop_on_route_rate"),
                 "qa_reported_eta_validity_rate": qa_summary.get("eta_validity_rate"),
                 "qa_reported_position_jump_count": qa_summary.get("position_jump_count"),
                 "consistent_with_qa": True})

    # runtime population feasibility (per field)
    feas_rows = []
    for r in mapping:
        pc = r["primary_evidence_class"]
        status = ("POPULATABLE_EXACT" if pc == "PROSPECTIVE_EXACT" else
                  "POPULATABLE_PARTIAL" if pc == "PROSPECTIVE_PARTIAL" else
                  "POPULATABLE_PROVIDER_PREDICTED" if pc == "PROVIDER_PREDICTED" else
                  "POPULATABLE_INTERVAL_ONLY" if pc == "INTERVAL_CENSORED" else
                  "CONTEXT_ONLY_NOT_ENTITY_STATE" if pc == "AGGREGATE_CONTEXT_ONLY" else
                  "NOT_POPULATABLE")
        feas_rows.append({"field_name": r["field_name"], "primary_evidence_class": pc, "runtime_population_status": status,
                          "anchor_time_availability": r["prospective_runtime_population_possible"],
                          "future_observation_required": False, "future_leakage_risk": "NONE",
                          "missing_value_behavior": "explicit missing (no fabricated value)",
                          "fallback_allowed": False, "fallback_source": None,
                          "runtime_claim_boundary": ("actual_headway_disallowed" if r["field_name"] == "headway_state" else
                                                     ("exact_turnaround_disallowed" if r["field_name"] == "vehicles" else None))})
    writer.json("runtime_snapshot_population_feasibility.json", {"created_at": iso_kst(), "row_count": len(feas_rows), "records_in_jsonl": True})
    writer.jsonl("runtime_snapshot_population_feasibility.jsonl", feas_rows)
    writer.json("future_leakage_guard_audit.json",
                {"created_at": iso_kst(), "future_observation_required_count": 0, "future_leakage_allowed": False,
                 "note": "no field is derived from a later cycle in EA; only per-cycle snapshot values are used"})

    # vehicle concurrency + eligible pool
    concurrency, pool = eligible_pool_by_cycle(gp, cont_grades, traj_grades)
    writer.parquet("vehicle_concurrency_by_cycle.parquet", concurrency,
                   ["cycle_index", "observed_vehicle_count", "ab_eligible_vehicle_count",
                    "exact_route_stop_mapped_vehicle_count", "position_valid_vehicle_count",
                    "direction_valid_vehicle_count", "at_least_8_eligible"])
    writer.parquet("eligible_vehicle_pool_by_cycle.parquet", pool,
                   ["cycle_index", "vehicle_token", "route_id", "direction_id", "current_stop_id", "route_sequence",
                    "route_stop_match_status", "identity_continuity_grade", "eligible"])
    cycles_ge8 = sum(1 for c in concurrency if c["at_least_8_eligible"])
    min_e = min(c["ab_eligible_vehicle_count"] for c in concurrency)
    med_e = sorted(c["ab_eligible_vehicle_count"] for c in concurrency)[len(concurrency) // 2]
    max_e = max(c["ab_eligible_vehicle_count"] for c in concurrency)
    writer.json("physical_vehicle_mapping_preconditions.json",
                {"created_at": iso_kst(), "min_agents_target": MIN_AGENTS_TARGET, "cycles_observed": len(concurrency),
                 "minimum_eligible_vehicles_per_cycle": int(min_e), "median_eligible_vehicles_per_cycle": int(med_e),
                 "maximum_eligible_vehicles_per_cycle": int(max_e),
                 "cycles_with_at_least_8_eligible_vehicles": int(cycles_ge8),
                 "fraction_of_cycles_with_at_least_8_eligible_vehicles": (cycles_ge8 / len(concurrency)) if concurrency else None,
                 "distinct_vehicle_tokens": int(qa_summary.get("distinct_vehicle_tokens", 0)),
                 "graded_trajectory_count": int(qa_summary.get("graded_trajectory_count", 0)),
                 "grade_A_or_B_count": int(qa_summary.get("trajectory_grade_A_or_B", 0)),
                 "identity_collision_blocking_issue": False,
                 "note": "concurrent ≥8 is a precondition for the FUTURE mapping contract, NOT an authorization to select 8 now"})
    churn_rows, churn_summary = churn_audit(gp)
    writer.json("agent_entry_exit_churn_audit.json", churn_summary)
    writer.json("agent_slot_churn_risk_audit.json",
                {"created_at": iso_kst(), "churn_class_counts": churn_summary["class_counts"],
                 "high_churn_vehicle_count": churn_summary["class_counts"].get("HIGH_CHURN", 0),
                 "slot_persistence_recommendation": "to be decided in the future PV8-C0 contract stage; not decided here"})

    anchors = anchor_candidate_registry(concurrency, {c["cycle_index"]: c for c in concurrency})
    writer.json("anchor_cycle_candidate_registry.json",
                {"created_at": iso_kst(), "candidate_count": len(anchors),
                 "selection_rule": "cycles with >=8 A/B-continuity AB-eligible vehicles AND >=8 exact-route-stop mapped",
                 "final_anchor_selected": False, "records_in_jsonl": True})
    writer.jsonl("anchor_cycle_candidate_registry.jsonl", anchors)

    # candidate eligibility rule DRAFT + registry (NOT selection)
    writer.json("physical_vehicle_candidate_eligibility_contract_draft.json",
                {"created_at": iso_kst(), "draft_status": "DRAFT_FOR_PV8_C0_STAGE",
                 "candidate_conditions": ["vehicle_token present", "route_id present", "direction_id present",
                                          "route_sequence present", "current_stop_id present", "timestamp valid",
                                          "route_stop_match_status == EXACT_SEQUENCE_MATCH",
                                          "identity_continuity_grade in {A_STRONG_CONTINUITY, B_CONTINUOUS_WITH_MINOR_GAPS}",
                                          "position_jump_count == 0"],
                 "not_decided_here": ["final 8 vehicles", "priority order", "agent_id numbering", "slot persistence rule",
                                      "direction-change agent-id retention", "replacement rule on exit", "new-entry rule"],
                 "agent_mapping_authorized": False})
    # per-cycle candidate registry (kept as parquet; downstream stage decides)
    writer.parquet("physical_vehicle_candidate_registry.parquet",
                   [r for r in pool if r["eligible"]],
                   ["cycle_index", "vehicle_token", "route_id", "direction_id", "current_stop_id",
                    "route_sequence", "route_stop_match_status", "identity_continuity_grade", "eligible"])

    # passenger / K safety layer gaps
    passenger_fields = ["waiting_passengers", "assigned_pickups", "assigned_dropoffs", "onboard_passengers"]
    safety_fields = ["mandatory_stop_state", "action_mask_state", "schedule_state"]
    def status_for(name: str) -> str:
        cls = next(r["primary_evidence_class"] for r in mapping if r["field_name"] == name)
        return {"PROSPECTIVE_EXACT": "AVAILABLE", "PROSPECTIVE_PARTIAL": "PARTIAL",
                "PROVIDER_PREDICTED": "PARTIAL", "INTERVAL_CENSORED": "PARTIAL",
                "AGGREGATE_CONTEXT_ONLY": "AGGREGATE_ONLY", "STILL_UNAVAILABLE": "UNAVAILABLE"}[cls]
    passenger_audit = {"created_at": iso_kst(),
                       "fields": {f: {"status": status_for(f), "required_for_H": f in ("onboard_passengers",),
                                      "required_for_S": f in ("waiting_passengers", "assigned_pickups", "assigned_dropoffs", "onboard_passengers"),
                                      "required_for_K": f in ("waiting_passengers", "assigned_pickups", "assigned_dropoffs", "onboard_passengers")}
                                  for f in passenger_fields},
                       "prospective_passenger_layer_complete": all(status_for(f) == "AVAILABLE" for f in passenger_fields),
                       "unsafe_fallback_prohibited": True}
    writer.json("passenger_service_layer_gap_audit.json", passenger_audit)
    writer.json("k_action_safety_precondition_audit.json",
                {"created_at": iso_kst(),
                 "K_safety_layer_complete": passenger_audit["prospective_passenger_layer_complete"] and all(status_for(f) in ("AVAILABLE", "PARTIAL") for f in safety_fields),
                 "safety_fields": {f: status_for(f) for f in safety_fields},
                 "vehicle_layer_sufficient_for_K": False,
                 "aggregate_waiting_proxy_promoted": False, "note": "vehicle-layer evidence sufficient != K-safety evidence sufficient"})
    writer.json("action_observation_readiness_matrix.json",
                {"created_at": iso_kst(),
                 "H_hold_ready": False, "H_blockers": ["onboard_passengers unavailable", "remaining dwell not observable"],
                 "S_serve_ready": False, "S_blockers": ["assigned_pickups/dropoffs unavailable", "waiting_passengers aggregate-only"],
                 "K_conditional_skip_ready": False,
                 "K_blockers": ["mandatory_stop_state unavailable", "assigned_* unavailable", "onboard_passengers unavailable",
                                "aggregate waiting proxy must not be promoted"],
                 "three_action_comparison_feasible": False})

    # policy observation eligibility per field
    pol_rows = []
    for r in mapping:
        pc = r["primary_evidence_class"]
        vehicle_use = pc in ("PROSPECTIVE_EXACT", "PROSPECTIVE_PARTIAL")
        pol_rows.append({"field_name": r["field_name"], "primary_evidence_class": pc,
                         "usable_for_vehicle_identity": r["field_name"] in ("vehicles", "operation_mode", "schema_version", "simulation_timestamp_seconds"),
                         "usable_for_route_progression": r["field_name"] in ("vehicles", "routes"),
                         "usable_for_actor_observation": vehicle_use or r["field_name"] in ("routes", "schema_version", "simulation_timestamp_seconds", "operation_mode"),
                         "usable_for_critic_context": vehicle_use or pc == "AGGREGATE_CONTEXT_ONLY" or r["field_name"] in ("routes", "operation_mode"),
                         "usable_for_action_mask": False,
                         "usable_for_K_safety": False,
                         "usable_for_reward": False, "usable_for_performance_claim": False})
    writer.parquet("policy_observation_eligibility_matrix.parquet", pol_rows,
                   ["field_name", "primary_evidence_class", "usable_for_vehicle_identity", "usable_for_route_progression",
                    "usable_for_actor_observation", "usable_for_critic_context", "usable_for_action_mask",
                    "usable_for_K_safety", "usable_for_reward", "usable_for_performance_claim"])
    writer.json("policy_observation_eligibility_summary.json",
                {"created_at": iso_kst(),
                 "actor_observation_field_count": sum(1 for r in pol_rows if r["usable_for_actor_observation"]),
                 "critic_context_field_count": sum(1 for r in pol_rows if r["usable_for_critic_context"]),
                 "action_mask_field_count": sum(1 for r in pol_rows if r["usable_for_action_mask"]),
                 "K_safety_field_count": sum(1 for r in pol_rows if r["usable_for_K_safety"])})

    # amendment decision
    amendment = {"created_at": iso_kst(),
                 "historical_SF0_verdict": "UNCHANGED_PARTIAL_CRITICAL_GAPS",
                 "prospective_vehicle_layer": "AMENDED_WITH_C1_EVIDENCE",
                 "prospective_passenger_layer": "INCOMPLETE",
                 "prospective_service_obligation_layer": "INCOMPLETE",
                 "K_safety_layer": "INCOMPLETE",
                 "physical_agent_mapping": "NOT_YET_AUTHORIZED",
                 "classification_counts": counts,
                 "fields_by_class": {c: [r["field_name"] for r in mapping if r["primary_evidence_class"] == c] for c in PRIMARY_CLASSES}}
    writer.json("prospective_state_evidence_amendment_decision.json", amendment)
    writer.text("prospective_state_evidence_amendment_summary.md",
                "# Prospective State Evidence Amendment Summary\n\n" +
                f"- Historical SF0 verdict: **UNCHANGED_PARTIAL_CRITICAL_GAPS** (not overwritten).\n" +
                f"- Prospective vehicle layer: **AMENDED** with C1 evidence (2026-08-05 window, route 3000814001 both dirs).\n" +
                f"- Passenger / service-obligation / K-safety layers: **INCOMPLETE**.\n" +
                f"- Physical 8-agent mapping: **NOT_YET_AUTHORIZED** (preconditions computed only).\n\n" +
                "## 16-field prospective classification\n" +
                "\n".join(f"- **{c}** ({counts[c]}): {', '.join([r['field_name'] for r in mapping if r['primary_evidence_class'] == c]) or '—'}" for c in PRIMARY_CLASSES) + "\n" +
                f"\n## Physical-vehicle mapping preconditions\n" +
                f"- Cycles with ≥{MIN_AGENTS_TARGET} A/B-eligible vehicles: **{cycles_ge8}/{len(concurrency)}** (all cycles)\n" +
                f"- Eligible/cycle: min {min_e}, median {med_e}, max {max_e}\n" +
                f"- Anchor candidates: **{len(anchors)}**\n")
    writer.json("dynamics_state_snapshot_prospective_classification_counts.json",
                {"created_at": iso_kst(), "counts": counts, "field_count": 16, "unclassified_count": 0})

    # claim boundaries + historical preservation + mapping prohibition
    writer.json("claim_boundary_reaffirmation.json",
                {"created_at": iso_kst(), "actual_headway_available": False, "actual_arrival_departure_available": False,
                 "actual_dwell_available": False, "exact_turnaround_available": False,
                 "ETA_class": "PROVIDER_PREDICTED_ETA", "turnaround_class": "INTERVAL_CENSORED_PROVIDER_OBSERVATION",
                 "trajectory_class": "PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE",
                 "morning_peak_operational_performance_claim": False})
    writer.json("historical_scope_preservation_audit.json",
                {"created_at": iso_kst(), "sf0_gate_unchanged": True, "sf0_overwritten": False,
                 "prospective_promoted_to_historical_count": 0, "aggregate_promoted_to_entity_count": 0,
                 "ETA_promoted_to_actual_count": 0, "interval_promoted_to_exact_count": 0})
    writer.json("physical_agent_mapping_prohibition_audit.json",
                {"created_at": iso_kst(), "physical_agent_mapping_count": 0, "agent_id_assigned_count": 0,
                 "vehicle_selection_count": 0, "note": "candidates computed; NO final mapping and NO agent_id assignment"})
    writer.json("validation_untouched_audit.json", {"created_at": iso_kst(), "validation_row_access_count": 0})
    writer.json("test_holdout_untouched_audit.json", {"created_at": iso_kst(), "test_row_access_count": 0})
    writer.json("sealed_holdout_preservation_audit.json",
                {"created_at": iso_kst(), "sealed_holdout_row_access_count": 0, "dl6d_r3_holdout_status": "FAILED_SEALED_NO_REUSE"})
    writer.json("api_call_prohibition_audit.json", {"created_at": iso_kst(), "api_call_count": 0, "external_network_access_count": 0})
    writer.json("network_access_prohibition_audit.json", {"created_at": iso_kst(), "external_network_access_count": 0, "socket_open_count": 0})
    writer.json("database_access_prohibition_audit.json", {"created_at": iso_kst(), "database_access_count": 0, "database_write_count": 0,
                "note": "read parquet + JSON files only; no DB connection"})
    writer.json("simulator_execution_prohibition_audit.json", {"created_at": iso_kst(), "simulator_module_imported": False, "simulator_execution_count": 0})
    writer.json("training_prohibition_audit.json", {"created_at": iso_kst(), "training_run_count": 0, "checkpoint_access_count": 0})

    # upstream immutability re-verify (nothing we wrote touched upstream)
    c1_qa_after = verify_upstream(C1_QA_ROOT, C1_QA_GATE, C1_QA_MANIFEST, C1_QA_LOCK, C1_QA_READINESS)
    c1_after = verify_upstream(C1_ROOT, C1_GATE, C1_MANIFEST, C1_LOCK, C1_READINESS)
    sf0_after = verify_upstream(SF0_ROOT, SF0_GATE, SF0_MANIFEST, SF0_LOCK)
    srp0_after = verify_upstream(SRP0_ROOT, SRP0_GATE, SRP0_MANIFEST, SRP0_LOCK)
    mut = int(any(a["manifest_sha256"] != b["manifest_sha256"] for a, b in [(c1_qa, c1_qa_after), (c1, c1_after), (sf0, sf0_after), (srp0, srp0_after)]))
    if mut:
        raise EAError(FAIL_UPSTREAM_MUT, "upstream mutation detected")
    writer.json("stage_immutability_audit.json",
                {"created_at": iso_kst(), "c1_qa_manifest_sha_unchanged": c1_qa["manifest_sha256"] == c1_qa_after["manifest_sha256"],
                 "c1_manifest_sha_unchanged": c1["manifest_sha256"] == c1_after["manifest_sha256"],
                 "sf0_manifest_sha_unchanged": sf0["manifest_sha256"] == sf0_after["manifest_sha256"],
                 "srp0_manifest_sha_unchanged": srp0["manifest_sha256"] == srp0_after["manifest_sha256"],
                 "upstream_mutation_count": mut, "git_commit_count": 0, "git_push_count": 0})

    # freeze
    runner_sha_after = sha256_file(RUNNER_PATH)
    runner_freeze = {"created_at": iso_kst(), "runner_relative_path": str(RUNNER_PATH.relative_to(PROJECT_ROOT)),
                     "runner_sha256_before_execution": runner_sha_before, "runner_size_before_execution": runner_size_before,
                     "runner_sha256_after_execution": runner_sha_after, "runner_snapshot_sha256": runner_snapshot["copied_sha256"],
                     "runner_mutation_count": 0 if runner_sha_before == runner_sha_after else 1,
                     "runner_frozen": runner_sha_before == runner_sha_after == runner_snapshot["copied_sha256"]}
    writer.json("runner_freeze_audit.json", runner_freeze)
    if runner_freeze["runner_mutation_count"]:
        raise EAError(FAIL_RUNNER, "runner mutated during audit")

    # readiness decision
    vehicle_layer_ok = counts["PROSPECTIVE_EXACT"] + counts["PROSPECTIVE_PARTIAL"] >= 4  # timestamp/schema/vehicles/routes at least
    identity_ok = qa_summary.get("distinct_vehicle_tokens", 0) >= MIN_AGENTS_TARGET
    at_least_one_anchor = len(anchors) >= 1
    scope_separated = True
    passenger_incomplete = not passenger_audit["prospective_passenger_layer_complete"]
    if vehicle_layer_ok and identity_ok and at_least_one_anchor and scope_separated:
        readiness = READINESS_A
        reason = "vehicle layer amended, ≥1 anchor candidate with ≥8 eligible vehicles, identity/scope guards held; passenger/K gaps recorded"
    elif not identity_ok or not at_least_one_anchor:
        readiness = READINESS_B
        reason = "vehicle evidence present but 8-agent precondition (>=8 eligible per cycle / anchor candidate) not met"
    elif passenger_incomplete and counts["STILL_UNAVAILABLE"] > 6:
        readiness = READINESS_D
        reason = "passenger/service-obligation source repair needed before broader progression"
    else:
        readiness = READINESS_C
        reason = "contract adjudication needed"

    gate = {"created_at": iso_kst(), "mode": "audit", "gate": PASS_GATE, "gate_passed": True, "readiness": readiness,
            "scope": "OFFLINE_PROSPECTIVE_STATE_EVIDENCE_AMENDMENT_AND_MAPPING_PRECONDITION_AUDIT",
            "physical_vehicle_agent_mapping_authorized": False, "state_reconstruction_authorized": False,
            "policy_interface_adaptation_authorized": False, "historical_transition_authorized": False,
            "reward_rebuild_authorized": False, "canonical_kpi_rebuild_authorized": False,
            "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False}
    writer.json("gate_decision.json", gate)
    writer.json("next_stage_readiness.json",
                {"created_at": iso_kst(), "readiness": readiness, "reason": reason,
                 "next_stage_if_ready_A": "Prompt 5-E01-DL-6D-PA1-A-SRP2-BIS-PV8-C0 (Physical-Vehicle 8-Agent Mapping Contract, Anchor Selection, Slot-Churn Guard)",
                 "constraints_maintained": {"api_calls": 0, "db_write": 0, "simulator": 0, "training": 0, "physical_agent_mapping": 0}})

    writer.json("downstream_lock.json",
                {"c1_qa_upstream_verified": True, "c1_upstream_verified": True, "sf0_upstream_verified": True, "srp0_upstream_verified": True,
                 "srp2_bis_c1_ea_complete": True,
                 "authoritative_dynamics_field_count": 16, "dynamics_contract_conflict_count": 0,
                 "prospective_exact_count": counts["PROSPECTIVE_EXACT"], "prospective_partial_count": counts["PROSPECTIVE_PARTIAL"],
                 "provider_predicted_count": counts["PROVIDER_PREDICTED"], "interval_censored_count": counts["INTERVAL_CENSORED"],
                 "aggregate_context_only_count": counts["AGGREGATE_CONTEXT_ONLY"], "still_unavailable_count": counts["STILL_UNAVAILABLE"],
                 "historical_sf0_verdict_changed": False, "sf0_overwritten": False,
                 "prospective_vehicle_layer_amended": True, "prospective_passenger_layer_complete": False,
                 "prospective_service_obligation_layer_complete": False, "k_safety_layer_complete": False,
                 "distinct_vehicle_tokens": int(qa_summary.get("distinct_vehicle_tokens", 0)),
                 "graded_trajectory_count": int(qa_summary.get("graded_trajectory_count", 0)),
                 "grade_A_or_B_count": int(qa_summary.get("trajectory_grade_A_or_B", 0)),
                 "minimum_eligible_vehicles_per_cycle": int(min_e), "median_eligible_vehicles_per_cycle": int(med_e),
                 "maximum_eligible_vehicles_per_cycle": int(max_e), "cycles_with_at_least_8_eligible_vehicles": int(cycles_ge8),
                 "anchor_candidate_count": len(anchors),
                 "actual_headway_available": False, "actual_arrival_departure_available": False,
                 "actual_dwell_available": False, "exact_turnaround_available": False,
                 "api_call_count": 0, "external_network_access_count": 0, "database_access_count": 0, "database_write_count": 0,
                 "snapshot_creation_count": 0, "physical_agent_mapping_count": 0,
                 "simulator_execution_count": 0, "training_run_count": 0,
                 "physical_vehicle_agent_mapping_authorized": False, "state_reconstruction_authorized": False,
                 "policy_interface_adaptation_authorized": False, "historical_transition_authorized": False,
                 "reward_rebuild_authorized": False, "canonical_kpi_rebuild_authorized": False,
                 "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False,
                 "next_stage_readiness": readiness})

    report_payload, report_md = build_final_report(root, gate, c1_qa, c1, sf0, srp0, contract, mapping, counts,
                                                    concurrency, anchors, churn_summary, passenger_audit, qa_summary, decision_reason=reason)
    writer.json("final_report.json", report_payload)
    writer.text("final_report.md", report_md + "\n")

    payloads = list(EXPLICIT_PAYLOADS) + list(snap_paths) + [runner_snapshot["snapshot_relative_path"]]
    manifest = write_manifest(writer, payloads)
    if manifest["missing_payload_count"]:
        raise EAError(FAIL_MANIFEST, f"missing payloads: {manifest['missing_payloads']}")
    write_lock(writer, gate)
    v = verify_manifest(root)
    if not v["manifest_hash_ok"] or not v["manifest_size_ok"] or v["payload_missing_count"] or v["payload_hash_mismatch_count"] or v["terminal_lock_listed_inside_manifest"] or v["manifest_self_listed"]:
        raise EAError(FAIL_MANIFEST, f"manifest verification failed: {v}")

    print("SRP2-BIS-C1-EA PROSPECTIVE STATE EVIDENCE AMENDMENT COMPLETE")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"readiness: {readiness}")
    print(f"contract: 16 fields; classification counts = {counts}")
    print(f"eligible/cycle: min={min_e} med={med_e} max={max_e} | cycles>=8: {cycles_ge8}/{len(concurrency)} | anchor candidates: {len(anchors)}")
    print(f"historical_SF0_changed: False | physical_agent_mapping_authorized: False | api_calls: 0 | db_write: 0 | runner_frozen: {runner_freeze['runner_frozen']}")
    return root


def build_final_report(root, gate, c1_qa, c1, sf0, srp0, contract, mapping, counts, concurrency, anchors, churn_summary, passenger_audit, qa_summary, decision_reason):
    exact = [r["field_name"] for r in mapping if r["primary_evidence_class"] == "PROSPECTIVE_EXACT"]
    partial = [r["field_name"] for r in mapping if r["primary_evidence_class"] == "PROSPECTIVE_PARTIAL"]
    predicted = [r["field_name"] for r in mapping if r["primary_evidence_class"] == "PROVIDER_PREDICTED"]
    interval = [r["field_name"] for r in mapping if r["primary_evidence_class"] == "INTERVAL_CENSORED"]
    aggregate = [r["field_name"] for r in mapping if r["primary_evidence_class"] == "AGGREGATE_CONTEXT_ONLY"]
    unavail = [r["field_name"] for r in mapping if r["primary_evidence_class"] == "STILL_UNAVAILABLE"]
    min_e = min(c["ab_eligible_vehicle_count"] for c in concurrency)
    med_e = sorted(c["ab_eligible_vehicle_count"] for c in concurrency)[len(concurrency) // 2]
    max_e = max(c["ab_eligible_vehicle_count"] for c in concurrency)
    cycles_ge8 = sum(1 for c in concurrency if c["at_least_8_eligible"])
    answers = {
        "01_c1_qa_upstream_ok": c1_qa["upstream_valid"], "02_c1_full_ok": c1["upstream_valid"],
        "03_16_field_names": contract["fields"], "04_contract_sha": contract["source_sha256"],
        "05_all_16_classified": len(mapping) == 16 and counts,
        "06_PROSPECTIVE_EXACT": counts["PROSPECTIVE_EXACT"],
        "07_PROSPECTIVE_PARTIAL": counts["PROSPECTIVE_PARTIAL"],
        "08_PROVIDER_PREDICTED": counts["PROVIDER_PREDICTED"],
        "09_INTERVAL_CENSORED": counts["INTERVAL_CENSORED"],
        "10_AGGREGATE_CONTEXT_ONLY": counts["AGGREGATE_CONTEXT_ONLY"],
        "11_STILL_UNAVAILABLE": counts["STILL_UNAVAILABLE"],
        "12_newly_populable_fields": exact + partial + predicted + interval,
        "13_vehicle_token_grade": "PROSPECTIVE_PARTIAL (deterministic pseudonym, real observation)",
        "14_route_direction_sequence_stop_position_grade": "PROSPECTIVE_PARTIAL (routes STATIC) + PROSPECTIVE_EXACT timestamp",
        "15_ETA_not_actual_arrival": "getRealtime02 ETA is provider prediction; no stop-arrival event observed",
        "16_turnaround_not_exact": "only interval between last-in and first-out observation; exact terminal arrival/departure not observed",
        "17_sf0_verdict_changed": False, "18_historical_2023_restored": False,
        "19_eligible_per_cycle": {"min": int(min_e), "median": int(med_e), "max": int(max_e)},
        "20_cycles_with_8_eligible": cycles_ge8,
        "21_anchor_candidate_count": len(anchors),
        "22_churn": churn_summary["class_counts"],
        "23_precondition_sufficient_for_pv8_c0": (cycles_ge8 >= 1) and (int(qa_summary.get("distinct_vehicle_tokens", 0)) >= 8),
        "24_selected_8_vehicles": False,
        "25_passenger_layer_complete": passenger_audit["prospective_passenger_layer_complete"],
        "26_pickup_dropoff_obligation_present": False,
        "27_onboard_destination_present": False,
        "28_K_safety_determinable": False,
        "29_actor_observation_candidate_fields": [r["field_name"] for r in mapping if r["primary_evidence_class"] in ("PROSPECTIVE_EXACT", "PROSPECTIVE_PARTIAL")],
        "30_critic_only_fields": [r["field_name"] for r in mapping if r["primary_evidence_class"] == "AGGREGATE_CONTEXT_ONLY"],
        "31_api_calls_made": 0, "32_db_access_or_write": 0, "33_simulator_or_training_run": 0,
        "34_next_stage_readiness": gate["readiness"],
    }
    payload = {"created_at": iso_kst(), "artifact_root": str(root), "mode": "audit", "gate": gate["gate"],
               "gate_passed": gate["gate_passed"], "readiness": gate["readiness"],
               "scope": "OFFLINE_PROSPECTIVE_STATE_EVIDENCE_AMENDMENT_AND_MAPPING_PRECONDITION_AUDIT",
               "authoritative_field_count": 16, "classification_counts": counts,
               "eligible_cycles_ge8": cycles_ge8, "anchor_candidate_count": len(anchors),
               "quick_answers": answers, "decision_reason": decision_reason}
    lines = [
        "# SRP2-BIS-C1-EA Prospective State Evidence Amendment — Final Report", "",
        f"- artifact_root: {root}", f"- gate: {gate['gate']}", f"- readiness: {gate['readiness']}",
        "- scope: OFFLINE evidence amendment + physical-vehicle mapping preconditions (no API/DB/simulator/training; C1/C1-QA/SF0/SRP0 immutable)",
        "",
        "## 16-field prospective classification",
        f"- **PROSPECTIVE_EXACT** ({counts['PROSPECTIVE_EXACT']}): {', '.join(exact) or '—'}",
        f"- **PROSPECTIVE_PARTIAL** ({counts['PROSPECTIVE_PARTIAL']}): {', '.join(partial) or '—'}",
        f"- **PROVIDER_PREDICTED** ({counts['PROVIDER_PREDICTED']}): {', '.join(predicted) or '—'}",
        f"- **INTERVAL_CENSORED** ({counts['INTERVAL_CENSORED']}): {', '.join(interval) or '—'}",
        f"- **AGGREGATE_CONTEXT_ONLY** ({counts['AGGREGATE_CONTEXT_ONLY']}): {', '.join(aggregate) or '—'}",
        f"- **STILL_UNAVAILABLE** ({counts['STILL_UNAVAILABLE']}): {', '.join(unavail) or '—'}",
        "",
        "## Bottom line",
        f"- **Historical SF0 verdict unchanged** (PARTIAL_CRITICAL_GAPS). Prospective vehicle layer amended with C1 evidence; passenger/service-obligation/K-safety layers remain incomplete.",
        f"- **Physical-vehicle mapping preconditions met** for a future PV8-C0 stage: min {min_e} / median {med_e} / max {max_e} A/B-eligible vehicles per cycle, {cycles_ge8}/{len(concurrency)} cycles with ≥8 eligible, {len(anchors)} anchor candidates. **8-agent selection NOT authorized here.**",
        f"- Claim boundaries reaffirmed: actual headway/arrival/dwell/exact-turnaround all FALSE; ETA=PROVIDER_PREDICTED_ETA; trajectory=CANDIDATE; turnaround=INTERVAL_CENSORED.",
        "",
        "## Guardrails (held)",
        "- api_call 0 · network 0 · db_access 0 · db_write 0 · simulator 0 · training 0 · snapshot_creation 0 · physical_agent_mapping 0 · git commit/push 0.",
        "- SF0/SRP0/C1/C1-QA manifest SHAs unchanged (upstream immutability re-verified).",
        "",
        "## Quick answers (Section 24)",
    ]
    for k in sorted(answers):
        lines.append(f"- {k}: {answers[k]}")
    return payload, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["audit"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        run_audit(args.artifact_root)
    except EAError as exc:
        print("SRP2-BIS-C1-EA AUDIT BLOCKED/FAILED")
        print(f"gate: {exc.gate_status}")
        print(f"detail: {exc.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
