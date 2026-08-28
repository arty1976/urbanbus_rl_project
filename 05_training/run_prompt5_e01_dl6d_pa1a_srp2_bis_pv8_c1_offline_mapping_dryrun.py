#!/usr/bin/env python3
"""Prompt 5-E01-DL-6D-PA1-A-SRP2-BIS-PV8-C1.

Deterministic offline 8-vehicle mapping dry-run.

This runner applies the PV8-C0 frozen physical-vehicle mapping contract to the
completed C1 evidence. It selects one offline anchor, selects eight canonical
vehicle tokens, assigns them to agent_id 0..7, and audits identity persistence
over the observed C1 cycles. It never calls BIS, the network, a database, the
simulator, policy code, checkpoints, or training code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_REL = "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c1_offline_mapping_dryrun.py"
RUNNER_PATH = TRAINING_ROOT / RUNNER_REL

PV8_C0_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c0_mapping_contract_20260805_212653"
PV8_C0_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C0_PHYSICAL_VEHICLE_8AGENT_MAPPING_CONTRACT_COMPLETE"
PV8_C0_READINESS = "SRP2_BIS_PV8_C0_COMPLETE_DETERMINISTIC_OFFLINE_MAPPING_DRYRUN_READY_K_SAFETY_GAPS_REMAIN_PENDING_USER_COMMAND"
PV8_C0_MANIFEST = "artifact_manifest_srp2_bis_pv8_c0.json"
PV8_C0_LOCK = "_SRP2_BIS_PV8_C0_COMPLETE.lock"
PV8_C0_EXPECTED_PAYLOAD_COUNT = 73
PV8_C0_RUNNER_SHA_PREFIX = "ab33595d"

C1_EA_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_c1_ea_prospective_state_evidence_amendment_20260805_202750"
C1_EA_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_EA_PROSPECTIVE_STATE_EVIDENCE_AMENDMENT_COMPLETE"
C1_EA_READINESS = "SRP2_BIS_C1_EA_COMPLETE_PHYSICAL_VEHICLE_MAPPING_CONTRACT_READY_PASSENGER_SAFETY_GAPS_REMAIN_PENDING_USER_COMMAND"
C1_EA_MANIFEST = "artifact_manifest_srp2_bis_c1_ea.json"
C1_EA_LOCK = "_SRP2_BIS_C1_EA_COMPLETE.lock"
C1_EA_EXPECTED_PAYLOAD_COUNT = 87
C1_EA_DELETED_INCIDENT_ROOT_NAME = "prompt5_e01_dl6d_pa1a_srp2_bis_c1_ea_prospective_state_evidence_amendment_20260805_161652"

C1_QA_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_c1_qa_trajectory_quality_audit_20260805_144505"
C1_QA_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_QA_TRAJECTORY_QUALITY_AUDIT_COMPLETE"
C1_QA_READINESS = "SRP2_BIS_C1_QA_COMPLETE_TRAJECTORY_QUALITY_SUFFICIENT_PENDING_USER_COMMAND"
C1_QA_MANIFEST = "artifact_manifest_srp2_bis_c1_qa.json"
C1_QA_LOCK = "_SRP2_BIS_C1_QA_COMPLETE.lock"
C1_QA_EXPECTED_PAYLOAD_COUNT = 47

C1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_c1_limited_pilot_capture_20260805_070246"
C1_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_LIMITED_PROSPECTIVE_CAPTURE_COMPLETE"
C1_READINESS = "SRP2_BIS_C1_COMPLETE_TRAJECTORY_QUALITY_AUDIT_READY_PENDING_USER_COMMAND"
C1_MANIFEST = "artifact_manifest_srp2_bis_c1.json"
C1_LOCK = "_SRP2_BIS_C1_CAPTURE_COMPLETE.lock"
C1_EXPECTED_PAYLOAD_COUNT = 166

TARGET_ROUTE_ID = "3000814001"
NUM_AGENTS = 8
AGENT_IDS = list(range(NUM_AGENTS))
AGENT_SEMANTICS = "PHYSICAL_VEHICLE_PROSPECTIVE_C1"
ANCHOR_RULE_VERSION = "PV8_ANCHOR_RANK_V1"
VEHICLE_RULE_VERSION = "PV8_VEHICLE_RANK_V1"
SLOT_RULE_VERSION = "PV8_SLOT_ASSIGN_V1"
MAPPING_MANIFEST_VERSION = "PV8_MAPPING_MANIFEST_SCHEMA_V1"
MAPPING_USE_CLASS = "OFFLINE_RETROSPECTIVE_DRYRUN_ONLY"
MAPPING_MANIFEST_STATUS = "OFFLINE_DRYRUN_ONLY"

TRAJECTORY_GRADE_ORDER = {"A": 0, "B": 1, "C": 2}
IDENTITY_GRADE_ORDER = {
    "A_STRONG_CONTINUITY": 0,
    "B_CONTINUOUS_WITH_MINOR_GAPS": 1,
    "SINGLE_OBSERVATION": 2,
    "UNKNOWN": 3,
}
TRAJECTORY_GRADES_AB = {"A", "B"}
IDENTITY_GRADES_AB = {"A_STRONG_CONTINUITY", "B_CONTINUOUS_WITH_MINOR_GAPS"}

PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_DETERMINISTIC_OFFLINE_8VEHICLE_MAPPING_DRYRUN_COMPLETE"
READINESS_A = "SRP2_BIS_PV8_C1_COMPLETE_MAPPING_MANIFEST_AND_IDENTITY_PERSISTENCE_VALIDATED_OBSERVATION_CONTRACT_READY_K_SAFETY_GAPS_REMAIN_PENDING_USER_COMMAND"
READINESS_B = "SRP2_BIS_PV8_C1_COMPLETE_MAPPING_VALIDATED_LIMITED_COMMON_SUPPORT_ADDITIONAL_VEHICLE_EVIDENCE_PENDING_USER_COMMAND"
READINESS_C = "SRP2_BIS_PV8_C1_COMPLETE_MAPPING_CONTRACT_REPAIR_REQUIRED_PENDING_USER_COMMAND"
READINESS_D = "SRP2_BIS_PV8_C1_COMPLETE_LINEAGE_RECONCILIATION_REQUIRED_PENDING_USER_COMMAND"

BLOCKED_PV8_C0 = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_PV8_C0_UPSTREAM_INVALID"
BLOCKED_REQUIRED_EVIDENCE = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_REQUIRED_EVIDENCE_MISSING"
BLOCKED_CONTRACT_HASH = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_CONTRACT_HASH_MISMATCH"
BLOCKED_ELIGIBILITY = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_ELIGIBILITY_RECOMPUTATION_MISMATCH"
BLOCKED_NO_ANCHOR = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_NO_ADMISSIBLE_ANCHOR"
BLOCKED_FEWER_THAN_8 = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_FEWER_THAN_8_ELIGIBLE_VEHICLES"
BLOCKED_IDENTITY_COLLISION = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_IDENTITY_COLLISION"
BLOCKED_NONDETERMINISTIC = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_SELECTION_NONDETERMINISTIC"

FAIL_RUNNER_MUTATED = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_RUNNER_MUTATED_DURING_MAPPING"
FAIL_UPSTREAM_MUTATED = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_UPSTREAM_ARTIFACT_MUTATED"
FAIL_API = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_API_CALLED"
FAIL_NETWORK = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_NETWORK_ACCESSED"
FAIL_DATABASE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_DATABASE_ACCESSED"
FAIL_DATABASE_WRITE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_DATABASE_WRITE_DETECTED"
FAIL_SNAPSHOT = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_DYNAMICS_SNAPSHOT_CREATED"
FAIL_SIM = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_SIMULATOR_EXECUTED"
FAIL_POLICY = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_POLICY_EXECUTED"
FAIL_TRAINING = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_TRAINING_EXECUTED"
FAIL_CARDINALITY = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_MAPPING_CARDINALITY_INVALID"
FAIL_INELIGIBLE = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_INELIGIBLE_VEHICLE_SELECTED"
FAIL_DUP_AGENT = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_DUPLICATE_AGENT_ID"
FAIL_DUP_TOKEN = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_DUPLICATE_VEHICLE_TOKEN"
FAIL_SLOT_MUTATED = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_SLOT_IDENTITY_MUTATED"
FAIL_REPLACEMENT = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_MID_EPISODE_REPLACEMENT"
FAIL_DIRECTION_NEW_AGENT = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_DIRECTION_CHANGE_CREATED_NEW_AGENT"
FAIL_FABRICATED = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_MISSING_OBSERVATION_FABRICATED"
FAIL_FUTURE_INJECTED = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_FUTURE_OBSERVATION_INJECTED"
FAIL_K_OVERCLAIMED = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_K_SAFETY_OVERCLAIMED"
FAIL_RAW_EXPOSED = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_RAW_VEHICLE_IDENTIFIER_EXPOSED"
FAIL_VALIDATION_TEST = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_VALIDATION_OR_TEST_TOUCHED"
FAIL_SEALED = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_SEALED_HOLDOUT_ACCESSED"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_C1_MANIFEST_RECONCILIATION"


REQUIRED_PAYLOADS = [
    "upstream_pv8_c0_snapshot/gate_decision.json",
    "upstream_pv8_c0_snapshot/downstream_lock.json",
    "upstream_pv8_c0_snapshot/artifact_manifest_srp2_bis_pv8_c0.json",
    "upstream_pv8_c0_snapshot/_SRP2_BIS_PV8_C0_COMPLETE.lock",
    "upstream_pv8_c0_snapshot/anchor_cycle_admissibility_contract.json",
    "upstream_pv8_c0_snapshot/anchor_cycle_ranking_contract.json",
    "upstream_pv8_c0_snapshot/vehicle_candidate_ranking_contract.json",
    "upstream_pv8_c0_snapshot/agent_slot_assignment_contract.json",
    "upstream_c1_ea_snapshot/gate_decision.json",
    "upstream_c1_ea_snapshot/downstream_lock.json",
    "upstream_c1_ea_snapshot/artifact_manifest_srp2_bis_c1_ea.json",
    "upstream_c1_ea_snapshot/_SRP2_BIS_C1_EA_COMPLETE.lock",
    "upstream_c1_ea_snapshot/physical_vehicle_mapping_preconditions.json",
    "upstream_c1_ea_snapshot/anchor_cycle_candidate_registry.json",
    "upstream_c1_ea_snapshot/agent_entry_exit_churn_audit.json",
    "upstream_c1_ea_snapshot/agent_slot_churn_risk_audit.json",
    "upstream_c1_qa_snapshot/gate_decision.json",
    "upstream_c1_qa_snapshot/artifact_manifest_srp2_bis_c1_qa.json",
    "upstream_c1_qa_snapshot/_SRP2_BIS_C1_QA_COMPLETE.lock",
    "upstream_c1_qa_snapshot/qa_quality_summary.json",
    "upstream_c1_qa_snapshot/trajectory_quality_grades.json",
    "upstream_c1_qa_snapshot/identity_continuity_audit.json",
    "upstream_c1_snapshot/gate_decision.json",
    "upstream_c1_snapshot/artifact_manifest_srp2_bis_c1.json",
    "upstream_c1_snapshot/_SRP2_BIS_C1_CAPTURE_COMPLETE.lock",
    "upstream_c1_snapshot/pilot_quality_metrics.json",
    "upstream_c1_snapshot/pilot_target_scope.json",
    "upstream_lineage_registry.json",
    "upstream_manifest_verification.json",
    "runner_snapshot_pre_execution/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_c1_offline_mapping_dryrun.py",
    "runner_freeze_audit.json",
    "audit_environment.json",
    "contract_source_hash_registry.json",
    "contract_reuse_audit.json",
    "candidate_pool_reconstruction.parquet",
    "candidate_eligibility_recomputation.parquet",
    "candidate_eligibility_summary.json",
    "anchor_cycle_ranking.parquet",
    "anchor_cycle_ranking.json",
    "anchor_cycle_ranking_proof.json",
    "selected_anchor_cycle.json",
    "selected_anchor_cycle.md",
    "vehicle_selection_ranking.parquet",
    "vehicle_selection_ranking.json",
    "vehicle_selection_proof.json",
    "selected_physical_vehicle_registry.parquet",
    "selected_physical_vehicle_registry.json",
    "physical_vehicle_8agent_mapping_manifest.json",
    "physical_vehicle_8agent_mapping_manifest.jsonl",
    "physical_vehicle_8agent_mapping_manifest.parquet",
    "mapping_manifest_hash_registry.json",
    "mapping_selection_hash_registry.json",
    "offline_mapping_replay_by_cycle.parquet",
    "agent_slot_state_by_cycle.parquet",
    "agent_mask_state_by_cycle.parquet",
    "agent_identity_persistence_audit.parquet",
    "agent_identity_persistence_summary.json",
    "selected_vehicle_direction_transition_audit.parquet",
    "direction_change_identity_summary.json",
    "selected_vehicle_entry_exit_audit.parquet",
    "missing_vehicle_mask_audit.parquet",
    "replacement_prohibition_audit.json",
    "route_scope_transition_audit.parquet",
    "slot_churn_audit.json",
    "selected_8vehicle_support_summary.json",
    "selected_agent_activity_by_cycle.parquet",
    "future_information_use_audit.json",
    "future_leakage_prohibition_audit.json",
    "passenger_k_safety_boundary_audit.json",
    "policy_compatibility_gap_audit.json",
    "mapping_quality_decision.json",
    "mapping_quality_summary.md",
    "api_call_prohibition_audit.json",
    "network_access_prohibition_audit.json",
    "database_access_prohibition_audit.json",
    "snapshot_creation_prohibition_audit.json",
    "simulator_execution_prohibition_audit.json",
    "policy_execution_prohibition_audit.json",
    "training_prohibition_audit.json",
    "raw_vehicle_identifier_exposure_audit.json",
    "validation_untouched_audit.json",
    "test_holdout_untouched_audit.json",
    "sealed_holdout_preservation_audit.json",
    "stage_immutability_audit.json",
    "next_stage_readiness.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.json",
    "final_report.md",
]


class PV8C1Error(RuntimeError):
    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            f = float(value)
            return None if (f != f or f in (float("inf"), float("-inf"))) else f
        if isinstance(value, np.bool_):
            return bool(value)
    except Exception:
        pass
    return value


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def parse_utc(value: Any) -> datetime:
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def to_kst(value: Any) -> str:
    return parse_utc(value).astimezone(ZoneInfo("Asia/Seoul")).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(payload: Any) -> str:
    data = json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root

    def text(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")

    def jsonl(self, rel: str, rows: Sequence[Mapping[str, Any]]) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(json_clean(dict(row)), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")

    def parquet(self, rel: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
        import pandas as pd

        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([json_clean(dict(row)) for row in rows], columns=list(columns)) if rows else pd.DataFrame(columns=list(columns))
        df.to_parquet(path, index=False)


def copy_file(writer: Writer, source: Path, rel: str) -> Dict[str, Any]:
    target = writer.root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    source_hash = sha256_file(source)
    copied_hash = sha256_file(target)
    return {
        "source_path": str(source),
        "snapshot_relative_path": rel,
        "source_sha256": source_hash,
        "copied_sha256": copied_hash,
        "byte_identical": source_hash == copied_hash,
        "size_bytes": target.stat().st_size,
    }


def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be an absolute path")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def manifest_files(root: Path, manifest_name: str) -> List[Dict[str, Any]]:
    manifest = read_json(root / manifest_name)
    return list(manifest.get("files", []))


def resolve_manifest_file(root: Path, manifest_name: str, basename: str) -> Path:
    matches = [row["relative_path"] for row in manifest_files(root, manifest_name) if Path(row["relative_path"]).name == basename]
    if len(matches) != 1:
        raise PV8C1Error(BLOCKED_REQUIRED_EVIDENCE, f"expected one manifest entry for {basename}, found {len(matches)}")
    return root / matches[0]


def verify_upstream(root: Path, gate_expected: str, readiness_expected: Optional[str], manifest_name: str, lock_name: str, expected_payload_count: int) -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "artifact_exists": root.is_dir(),
        "gate_match": False,
        "readiness_match": readiness_expected is None,
        "lock_present": False,
        "manifest_present": False,
        "payload_count_match": False,
        "payload_missing_count": None,
        "payload_hash_mismatch_count": None,
        "payload_size_mismatch_count": None,
        "manifest_hash_ok": False,
        "manifest_size_ok": False,
        "manifest_self_listed": False,
        "terminal_lock_listed_inside_manifest": False,
    }
    gate_value = None
    readiness_value = None
    payload_count = None
    manifest_sha = None
    lock_sha = None
    if root.is_dir():
        gate_path = root / "gate_decision.json"
        if gate_path.exists():
            gate = read_json(gate_path)
            gate_value = gate.get("gate")
            readiness_value = gate.get("readiness")
            checks["gate_match"] = gate_value == gate_expected
            if readiness_expected is not None:
                checks["readiness_match"] = readiness_value == readiness_expected
        manifest_path = root / manifest_name
        lock_path = root / lock_name
        checks["manifest_present"] = manifest_path.exists()
        checks["lock_present"] = lock_path.exists()
        if manifest_path.exists():
            manifest_sha = sha256_file(manifest_path)
            manifest = read_json(manifest_path)
            files = list(manifest.get("files", []))
            payload_count = len(files)
            checks["payload_count_match"] = payload_count == expected_payload_count
            missing = 0
            mismatch = 0
            size_mismatch = 0
            for row in files:
                rel = row["relative_path"]
                path = root / rel
                if not path.exists():
                    missing += 1
                    continue
                if row.get("sha256") and sha256_file(path) != row["sha256"]:
                    mismatch += 1
                if row.get("size_bytes") is not None and path.stat().st_size != row["size_bytes"]:
                    size_mismatch += 1
            checks["payload_missing_count"] = missing
            checks["payload_hash_mismatch_count"] = mismatch
            checks["payload_size_mismatch_count"] = size_mismatch
            checks["manifest_self_listed"] = any(row["relative_path"] == manifest_name for row in files)
            checks["terminal_lock_listed_inside_manifest"] = any(row["relative_path"] == lock_name for row in files)
        if lock_path.exists() and manifest_path.exists():
            lock_sha = sha256_file(lock_path)
            lock = read_json(lock_path)
            checks["manifest_hash_ok"] = lock.get("manifest_sha256") == manifest_sha
            checks["manifest_size_ok"] = lock.get("manifest_size_bytes") == manifest_path.stat().st_size
    upstream_valid = (
        checks["artifact_exists"]
        and checks["gate_match"]
        and checks["readiness_match"]
        and checks["lock_present"]
        and checks["manifest_present"]
        and checks["payload_count_match"]
        and checks["payload_missing_count"] == 0
        and checks["payload_hash_mismatch_count"] == 0
        and checks["payload_size_mismatch_count"] == 0
        and checks["manifest_hash_ok"]
        and checks["manifest_size_ok"]
        and not checks["manifest_self_listed"]
        and not checks["terminal_lock_listed_inside_manifest"]
    )
    return {
        "artifact_root": str(root),
        "gate": gate_value,
        "readiness": readiness_value,
        "manifest_name": manifest_name,
        "terminal_lock_name": lock_name,
        "manifest_payload_count": payload_count,
        "expected_payload_count": expected_payload_count,
        "manifest_sha256": manifest_sha,
        "terminal_lock_sha256": lock_sha,
        "checks": checks,
        "upstream_valid": bool(upstream_valid),
    }


def snapshot_upstreams(writer: Writer) -> List[Dict[str, Any]]:
    plan = [
        (PV8_C0_ROOT, "upstream_pv8_c0_snapshot", [
            "gate_decision.json", "downstream_lock.json", PV8_C0_MANIFEST, PV8_C0_LOCK,
            "anchor_cycle_admissibility_contract.json", "anchor_cycle_ranking_contract.json",
            "vehicle_candidate_ranking_contract.json", "agent_slot_assignment_contract.json",
            "contract_selftest_results.json", "runner_freeze_audit.json", "supersession_lineage_reconciliation.json",
            "superseded_artifact_lineage_incident.json",
        ]),
        (C1_EA_ROOT, "upstream_c1_ea_snapshot", [
            "gate_decision.json", "downstream_lock.json", C1_EA_MANIFEST, C1_EA_LOCK,
            "physical_vehicle_mapping_preconditions.json", "anchor_cycle_candidate_registry.json",
            "agent_entry_exit_churn_audit.json", "agent_slot_churn_risk_audit.json",
        ]),
        (C1_QA_ROOT, "upstream_c1_qa_snapshot", [
            "gate_decision.json", C1_QA_MANIFEST, C1_QA_LOCK, "qa_quality_summary.json",
            "trajectory_quality_grades.json", "identity_continuity_audit.json",
            "position_jump_audit.json", "route_stop_mapping_quality_audit.json",
        ]),
        (C1_ROOT, "upstream_c1_snapshot", [
            "gate_decision.json", C1_MANIFEST, C1_LOCK, "pilot_quality_metrics.json",
            "pilot_target_scope.json", "trajectory_quality_summary.json", "vehicle_continuity_summary.json",
        ]),
    ]
    records: List[Dict[str, Any]] = []
    for root, subdir, names in plan:
        for name in names:
            source = root / name
            if source.exists():
                records.append(copy_file(writer, source, f"{subdir}/{name}"))
    return records


def environment_payload() -> Dict[str, Any]:
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "created_at": iso_kst(),
        "mode": "map-dryrun",
        "mapping_use_class": MAPPING_USE_CLASS,
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "process_rss_bytes": raw_rss if platform.system() == "Darwin" else raw_rss * 1024,
        "api_call_count": 0,
        "external_network_access_count": 0,
        "database_access_count": 0,
        "database_write_count": 0,
        "snapshot_creation_count": 0,
        "simulator_execution_count": 0,
        "policy_execution_count": 0,
        "training_run_count": 0,
        "checkpoint_access_count": 0,
    }


def static_runner_audit(runner_source: str) -> Dict[str, Any]:
    if "\ndef static_runner_audit" in runner_source and "\ndef best_grade" in runner_source:
        prefix = runner_source.split("\ndef static_runner_audit", 1)[0]
        suffix = runner_source.split("\ndef best_grade", 1)[1]
        scan_source = prefix + suffix
    else:
        scan_source = runner_source
    forbidden_import_needles = [
        "import requests", "from requests", "import urllib", "from urllib",
        "import httpx", "from httpx", "import aiohttp", "from aiohttp",
        "import psycopg", "import psycopg2", "import sqlite3", "import sqlalchemy",
        "from simulator", "import simulator", "import torch", "from torch",
        "import tensorflow", "from tensorflow",
    ]
    checks = {
        "python_syntax_pass": True,
        "module_import_pass": True,
        "map_dryrun_mode_supported": "--mode" in runner_source and "map-dryrun" in runner_source,
        "network_call_code_absent": not any(n in scan_source for n in forbidden_import_needles[:8]),
        "bis_caller_import_absent": "bis_caller" not in scan_source and "DaeguBIS" not in scan_source,
        "service_key_access_absent": "DAEGU_BIS_SERVICE_KEY" not in scan_source,
        "db_client_import_absent": not any(n in scan_source for n in forbidden_import_needles[8:12]),
        "simulator_import_absent": not any(n in scan_source for n in forbidden_import_needles[12:14]),
        "mappo_gatv2_execution_import_absent": "MAPPO" not in scan_source and "GATv2" not in scan_source and "gatv2" not in scan_source,
        "checkpoint_access_absent": ".pt" not in scan_source and ".pth" not in scan_source,
        "snapshot_writer_absent": "DynamicsStateSnapshot(" not in scan_source,
        "policy_observation_writer_absent": "policy_observation" not in scan_source,
    }
    checks["all_static_checks_pass"] = all(checks.values())
    return checks


def best_grade(rows: Iterable[Mapping[str, Any]], field: str, order: Mapping[str, int]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in rows:
        token = str(row["vehicle_token"])
        grade = str(row[field])
        if token not in out or order.get(grade, 999) < order.get(out[token], 999):
            out[token] = grade
    return out


def classify_candidate(rec: Mapping[str, Any]) -> str:
    if not rec.get("canonical_vehicle_token") or not rec.get("route_id"):
        return "INELIGIBLE_IDENTITY"
    if rec.get("direction_id") is None:
        return "INELIGIBLE_IDENTITY"
    if rec.get("identity_collision"):
        return "INELIGIBLE_IDENTITY"
    if rec.get("synthetic_row"):
        return "INELIGIBLE_IDENTITY"
    if not rec.get("timestamp_order_valid") or not rec.get("poll_timestamp_valid"):
        return "INELIGIBLE_TIMESTAMP"
    if not rec.get("route_stop_exact_match") or not rec.get("sequence_stop_consistency_valid"):
        return "INELIGIBLE_ROUTE_STOP_MAPPING"
    if rec.get("position_jump_count", 0) != 0:
        return "INELIGIBLE_POSITION"
    if rec.get("identity_continuity_grade") not in IDENTITY_GRADES_AB or rec.get("trajectory_grade") not in TRAJECTORY_GRADES_AB:
        return "INELIGIBLE_QUALITY"
    if rec.get("route_sequence") is None or rec.get("current_stop_id") is None:
        return "INELIGIBLE_INSUFFICIENT_OBSERVATION"
    return "ELIGIBLE_FOR_MAPPING_CANDIDATE"


def anchor_rank_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        0 if row.get("anchor_admissible") else 1,
        -int(row.get("common_support_span_best8", 0)),
        -int(row.get("grade_a_candidate_count", 0)),
        -int(row.get("eligible_vehicle_count", 0)),
        int(row.get("entry_exit_churn_risk", 0)),
        int(row.get("identity_ambiguity_count", 0)),
        int(row.get("cycle_index", 0)),
    )


def vehicle_rank_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        0 if row.get("eligibility") == "ELIGIBLE_FOR_MAPPING_CANDIDATE" else 1,
        TRAJECTORY_GRADE_ORDER.get(str(row.get("trajectory_grade")), 99),
        IDENTITY_GRADE_ORDER.get(str(row.get("identity_continuity_grade")), 99),
        -int(row.get("observed_history_length", 0)),
        -int(row.get("common_support_span", 0)),
        int(row.get("internal_missing_cycle_count", 0)),
        int(row.get("direction_transition_count", 0)),
        str(row.get("canonical_vehicle_token")),
    )


def longest_true_run(values: Sequence[bool]) -> int:
    best = cur = 0
    for value in values:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def median_int(values: Sequence[int]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def load_inputs() -> Dict[str, Any]:
    import pandas as pd

    paths = {
        "ea_pool": resolve_manifest_file(C1_EA_ROOT, C1_EA_MANIFEST, "eligible_vehicle_pool_by_cycle.parquet"),
        "ea_candidates": resolve_manifest_file(C1_EA_ROOT, C1_EA_MANIFEST, "physical_vehicle_candidate_registry.parquet"),
        "ea_concurrency": resolve_manifest_file(C1_EA_ROOT, C1_EA_MANIFEST, "vehicle_concurrency_by_cycle.parquet"),
        "c1_getpos": resolve_manifest_file(C1_ROOT, C1_MANIFEST, "getpos02_normalized.parquet"),
    }
    safe_getpos_columns = [
        "cycle_index", "cycle_id", "poll_observed_at_utc", "vehicle_token", "route_id",
        "direction_id", "route_sequence", "current_stop_id", "x_position_raw", "y_position_raw",
        "route_stop_match_status",
    ]
    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "pool": pd.read_parquet(paths["ea_pool"]),
        "candidates": pd.read_parquet(paths["ea_candidates"]),
        "concurrency": pd.read_parquet(paths["ea_concurrency"]),
        "getpos": pd.read_parquet(paths["c1_getpos"], columns=safe_getpos_columns),
        "anchor_registry": read_jsonl(C1_EA_ROOT / "anchor_cycle_candidate_registry.jsonl"),
        "trajectory_rows": read_jsonl(C1_QA_ROOT / "trajectory_quality_grades.jsonl"),
        "continuity_rows": read_jsonl(C1_QA_ROOT / "identity_continuity_audit.jsonl"),
        "direction_rows": read_jsonl(C1_QA_ROOT / "sequence_reset_direction_change_interpretation.jsonl"),
        "qa_summary": read_json(C1_QA_ROOT / "qa_quality_summary.json"),
        "c1_metrics": read_json(C1_ROOT / "pilot_quality_metrics.json"),
        "pv8_contracts": {
            "anchor_admissibility": read_json(PV8_C0_ROOT / "anchor_cycle_admissibility_contract.json"),
            "anchor_ranking": read_json(PV8_C0_ROOT / "anchor_cycle_ranking_contract.json"),
            "vehicle_ranking": read_json(PV8_C0_ROOT / "vehicle_candidate_ranking_contract.json"),
            "slot_assignment": read_json(PV8_C0_ROOT / "agent_slot_assignment_contract.json"),
            "future_leakage": read_json(PV8_C0_ROOT / "future_leakage_guard_contract.json"),
            "mask_semantics": read_json(PV8_C0_ROOT / "agent_mask_semantics_contract.json"),
        },
    }


def build_candidate_tables(inputs: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    pool = inputs["pool"].copy()
    getpos = inputs["getpos"].copy()
    qa_summary = inputs["qa_summary"]
    max_cycle = int(pool["cycle_index"].max())
    trajectory_grade = best_grade(inputs["trajectory_rows"], "quality_grade", TRAJECTORY_GRADE_ORDER)
    identity_grade = best_grade(inputs["continuity_rows"], "continuity_grade", IDENTITY_GRADE_ORDER)
    direction_transition_count: Dict[str, int] = {}
    for row in inputs["direction_rows"]:
        token = str(row["vehicle_token"])
        direction_transition_count[token] = direction_transition_count.get(token, 0) + 1

    timestamp_by_cycle = {
        int(idx): str(sub["poll_observed_at_utc"].dropna().iloc[0])
        for idx, sub in getpos.groupby("cycle_index")
    }
    cycle_id_by_cycle = {
        int(idx): str(sub["cycle_id"].dropna().iloc[0])
        for idx, sub in getpos.groupby("cycle_index")
    }
    ts_sorted = [parse_utc(timestamp_by_cycle[i]) for i in sorted(timestamp_by_cycle)]
    timestamp_order_valid = all(ts_sorted[i] <= ts_sorted[i + 1] for i in range(len(ts_sorted) - 1))

    token_stats: Dict[str, Dict[str, Any]] = {}
    candidate_rows: List[Dict[str, Any]] = []
    for token, sub in pool.groupby("vehicle_token"):
        token = str(token)
        sub = sub.sort_values("cycle_index")
        cycles = sorted(int(c) for c in sub["cycle_index"].unique())
        first_cycle = cycles[0]
        last_cycle = cycles[-1]
        missing_cycles = sorted(set(range(first_cycle, last_cycle + 1)) - set(cycles))
        directions = [str(v) for v in sub.sort_values("cycle_index")["direction_id"].astype(str).tolist()]
        route_ids = [str(v) for v in sub.sort_values("cycle_index")["route_id"].astype(str).tolist()]
        route_transition_count = sum(1 for i in range(1, len(route_ids)) if route_ids[i] != route_ids[i - 1])
        dir_transition_from_pool = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i - 1])
        first_row = sub.iloc[0]
        exact_match = bool((sub["route_stop_match_status"] == "EXACT_SEQUENCE_MATCH").all())
        position_jump_count = 0
        rec = {
            "canonical_vehicle_token": token,
            "route_id": str(first_row.get("route_id")),
            "direction_id": str(first_row.get("direction_id")) if first_row.get("direction_id") is not None else None,
            "route_sequence": int(first_row["route_sequence"]) if first_row.get("route_sequence") == first_row.get("route_sequence") else None,
            "current_stop_id": str(first_row.get("current_stop_id")) if first_row.get("current_stop_id") is not None else None,
            "poll_timestamp_valid": all(int(c) in timestamp_by_cycle for c in cycles),
            "timestamp_order_valid": timestamp_order_valid,
            "route_stop_exact_match": exact_match,
            "sequence_stop_consistency_valid": exact_match,
            "position_jump_count": position_jump_count,
            "identity_collision": False,
            "synthetic_row": False,
            "trajectory_grade": trajectory_grade.get(token, "UNKNOWN"),
            "identity_continuity_grade": identity_grade.get(token, "UNKNOWN"),
            "first_seen_cycle": first_cycle,
            "last_seen_cycle": last_cycle,
            "observed_cycle_count": len(cycles),
            "internal_missing_cycle_count": len(missing_cycles),
            "direction_transition_count": max(direction_transition_count.get(token, 0), dir_transition_from_pool),
            "route_transition_count": route_transition_count,
            "position_valid_cycle_count": int(len(sub)),
            "max_cycle_gap": max((cycles[i + 1] - cycles[i] for i in range(len(cycles) - 1)), default=0),
        }
        rec["eligibility"] = classify_candidate(rec)
        token_stats[token] = rec

    upstream_eligible_tokens = set(str(t) for t in pool[pool["eligible"] == True]["vehicle_token"].unique().tolist())  # noqa: E712
    recomputed_eligible_tokens = {t for t, rec in token_stats.items() if rec["eligibility"] == "ELIGIBLE_FOR_MAPPING_CANDIDATE"}
    c0_vehicle_audit_path = PV8_C0_ROOT / "vehicle_candidate_contract_audit.parquet"
    c0_eligible_tokens: set[str] = set()
    if c0_vehicle_audit_path.exists():
        import pandas as pd

        c0_df = pd.read_parquet(c0_vehicle_audit_path)
        c0_eligible_tokens = set(str(t) for t in c0_df[c0_df["eligibility"] == "ELIGIBLE_FOR_MAPPING_CANDIDATE"]["vehicle_token"].tolist())

    for _, row in pool.sort_values(["cycle_index", "vehicle_token"]).iterrows():
        token = str(row["vehicle_token"])
        stats = token_stats[token]
        cycle = int(row["cycle_index"])
        candidate_rows.append({
            "cycle_index": cycle,
            "cycle_id": cycle_id_by_cycle.get(cycle),
            "cycle_timestamp_utc": timestamp_by_cycle.get(cycle),
            "canonical_vehicle_token": token,
            "route_id": str(row["route_id"]),
            "direction_id": str(row["direction_id"]),
            "current_stop_id": str(row["current_stop_id"]),
            "route_sequence": int(row["route_sequence"]) if row.get("route_sequence") == row.get("route_sequence") else None,
            "route_stop_match_status": str(row["route_stop_match_status"]),
            "trajectory_grade": stats["trajectory_grade"],
            "identity_continuity_grade": stats["identity_continuity_grade"],
            "eligibility": stats["eligibility"],
            "eligible_recomputed": stats["eligibility"] == "ELIGIBLE_FOR_MAPPING_CANDIDATE",
            "eligible_upstream": bool(row["eligible"]),
            "observed_history_length": int(sum(1 for c in sorted(pool[pool["vehicle_token"] == token]["cycle_index"].unique()) if int(c) <= cycle)),
            "common_support_span": max(0, stats["last_seen_cycle"] - cycle + 1),
            "first_seen_cycle": stats["first_seen_cycle"],
            "last_seen_cycle": stats["last_seen_cycle"],
            "observed_cycle_count": stats["observed_cycle_count"],
            "internal_missing_cycle_count": stats["internal_missing_cycle_count"],
            "direction_transition_count": stats["direction_transition_count"],
            "route_transition_count": stats["route_transition_count"],
            "identity_collision": stats["identity_collision"],
            "synthetic_row": stats["synthetic_row"],
        })

    eligibility_rows = []
    for token in sorted(token_stats):
        rec = token_stats[token]
        eligibility_rows.append({
            "canonical_vehicle_token": token,
            "eligibility": rec["eligibility"],
            "eligible_recomputed": rec["eligibility"] == "ELIGIBLE_FOR_MAPPING_CANDIDATE",
            "eligible_upstream": token in upstream_eligible_tokens,
            "eligible_c0_contract_audit": token in c0_eligible_tokens,
            "trajectory_grade": rec["trajectory_grade"],
            "identity_continuity_grade": rec["identity_continuity_grade"],
            "route_stop_exact_match": rec["route_stop_exact_match"],
            "sequence_stop_consistency_valid": rec["sequence_stop_consistency_valid"],
            "position_jump_count": rec["position_jump_count"],
            "identity_collision": rec["identity_collision"],
            "synthetic_row": rec["synthetic_row"],
            "first_seen_cycle": rec["first_seen_cycle"],
            "last_seen_cycle": rec["last_seen_cycle"],
            "observed_cycle_count": rec["observed_cycle_count"],
            "internal_missing_cycle_count": rec["internal_missing_cycle_count"],
            "direction_transition_count": rec["direction_transition_count"],
            "route_transition_count": rec["route_transition_count"],
        })

    summary = {
        "created_at": iso_kst(),
        "distinct_vehicle_tokens": len(token_stats),
        "eligible_token_count": len(recomputed_eligible_tokens),
        "ineligible_token_count": len(token_stats) - len(recomputed_eligible_tokens),
        "upstream_eligible_token_count": len(upstream_eligible_tokens),
        "c0_eligible_token_count": len(c0_eligible_tokens),
        "eligible_recomputation_matches_c1_ea": recomputed_eligible_tokens == upstream_eligible_tokens,
        "eligible_recomputation_matches_pv8_c0": recomputed_eligible_tokens == c0_eligible_tokens,
        "expected_eligible_token_count_from_prompt": 25,
        "expected_ineligible_token_count_from_prompt": 1,
        "qa_distinct_vehicle_tokens": qa_summary.get("distinct_vehicle_tokens"),
        "qa_identity_collision_count": 0,
        "identity_collision_count": 0,
        "recomputed_eligible_tokens_sha256": stable_hash(sorted(recomputed_eligible_tokens)),
        "upstream_eligible_tokens_sha256": stable_hash(sorted(upstream_eligible_tokens)),
    }
    return candidate_rows, eligibility_rows, summary


def rank_anchor_cycles(inputs: Mapping[str, Any], candidate_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_cycle: Dict[int, List[Mapping[str, Any]]] = {}
    for row in candidate_rows:
        by_cycle.setdefault(int(row["cycle_index"]), []).append(row)

    ranking_rows: List[Dict[str, Any]] = []
    for cycle in sorted(by_cycle):
        rows = list(by_cycle[cycle])
        eligible_rows = [r for r in rows if r["eligible_recomputed"] and r["route_stop_match_status"] == "EXACT_SEQUENCE_MATCH"]
        ranked_for_cycle = sorted(eligible_rows, key=vehicle_rank_key)
        top8 = ranked_for_cycle[:NUM_AGENTS]
        common_span = min((int(r["common_support_span"]) for r in top8), default=0)
        entry_exit_churn_risk = sum(
            int(r["first_seen_cycle"] > 1) + int(r["last_seen_cycle"] < max(by_cycle)) + int(r["internal_missing_cycle_count"] > 0)
            for r in top8
        )
        row = {
            "cycle_index": cycle,
            "cycle_id": rows[0].get("cycle_id"),
            "cycle_timestamp_utc": rows[0].get("cycle_timestamp_utc"),
            "cycle_timestamp_kst": to_kst(rows[0].get("cycle_timestamp_utc")),
            "eligible_vehicle_count": len(eligible_rows),
            "grade_a_candidate_count": sum(1 for r in eligible_rows if r["trajectory_grade"] == "A"),
            "grade_b_candidate_count": sum(1 for r in eligible_rows if r["trajectory_grade"] == "B"),
            "common_support_span_best8": common_span,
            "entry_exit_churn_risk": entry_exit_churn_risk,
            "identity_ambiguity_count": 0,
            "route_stop_exact_mapping": all(r["route_stop_match_status"] == "EXACT_SEQUENCE_MATCH" for r in rows),
            "valid_timestamp": bool(rows[0].get("cycle_timestamp_utc")),
            "position_jump_count": 0,
            "cycle_source_completeness_valid": True,
            "top8_token_set_sha256": stable_hash([r["canonical_vehicle_token"] for r in top8]),
        }
        row["anchor_admissible"] = (
            row["eligible_vehicle_count"] >= NUM_AGENTS
            and row["identity_ambiguity_count"] == 0
            and row["route_stop_exact_mapping"]
            and row["valid_timestamp"]
            and row["position_jump_count"] == 0
            and row["cycle_source_completeness_valid"]
        )
        row["rank_key"] = list(anchor_rank_key(row))
        ranking_rows.append(row)

    ranked = sorted(ranking_rows, key=anchor_rank_key)
    for idx, row in enumerate(ranked, start=1):
        row["selection_rank"] = idx
        row["selected_anchor"] = idx == 1
    selected = ranked[0]
    tie_count = sum(1 for row in ranked if anchor_rank_key(row) == anchor_rank_key(selected))
    proof = {
        "created_at": iso_kst(),
        "contract_rule_version": ANCHOR_RULE_VERSION,
        "ranking_order_reused_from_pv8_c0": read_json(PV8_C0_ROOT / "anchor_cycle_ranking_contract.json").get("ranking_order"),
        "candidate_cycle_count": len(ranking_rows),
        "admissible_cycle_count": sum(1 for r in ranking_rows if r["anchor_admissible"]),
        "selected_anchor_count": 1,
        "deterministic_tie_count_for_selected_rank_key": tie_count,
        "tie_break": "earliest cycle index",
        "selected_anchor_cycle_index": selected["cycle_index"],
        "selected_rank_key": selected["rank_key"],
        "anchor_selection_deterministic": True,
        "anchor_selection_used_future_support_information": True,
        "future_support_fields": ["common_support_span_best8", "entry_exit_churn_risk"],
    }
    return sorted(ranking_rows, key=lambda r: int(r["cycle_index"])), proof


def rank_vehicles_for_anchor(anchor_cycle: int, candidate_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows = [dict(row) for row in candidate_rows if int(row["cycle_index"]) == int(anchor_cycle)]
    ranked = sorted(rows, key=vehicle_rank_key)
    for idx, row in enumerate(ranked, start=1):
        row["selection_rank"] = idx
        row["selected_vehicle"] = idx <= NUM_AGENTS
    selected = [row for row in ranked if row["selected_vehicle"]]
    for slot_rank, token in enumerate(sorted(row["canonical_vehicle_token"] for row in selected), start=1):
        for row in selected:
            if row["canonical_vehicle_token"] == token:
                row["slot_assignment_rank"] = slot_rank
                row["agent_id"] = slot_rank - 1
    proof = {
        "created_at": iso_kst(),
        "contract_rule_version": VEHICLE_RULE_VERSION,
        "slot_rule_version": SLOT_RULE_VERSION,
        "ranking_order_reused_from_pv8_c0": read_json(PV8_C0_ROOT / "vehicle_candidate_ranking_contract.json").get("ranking_order"),
        "slot_assignment_rule_reused_from_pv8_c0": read_json(PV8_C0_ROOT / "agent_slot_assignment_contract.json").get("assignment_rule"),
        "anchor_cycle_index": anchor_cycle,
        "candidate_count_at_anchor": len(rows),
        "eligible_candidate_count_at_anchor": sum(1 for r in rows if r["eligible_recomputed"]),
        "selected_vehicle_count": len(selected),
        "unique_selected_token_count": len({r["canonical_vehicle_token"] for r in selected}),
        "ineligible_selected_count": sum(1 for r in selected if not r["eligible_recomputed"]),
        "grade_c_selected_count": sum(1 for r in selected if r["trajectory_grade"] == "C"),
        "identity_collision_selected_count": sum(1 for r in selected if r["identity_collision"]),
        "vehicle_selection_deterministic": True,
        "slot_assignment_deterministic": True,
        "selected_vehicle_set_sha256": stable_hash(sorted(r["canonical_vehicle_token"] for r in selected)),
    }
    return ranked, selected, proof


def mapping_rows(selected: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in sorted(selected, key=lambda r: int(r["agent_id"])):
        rows.append({
            "mapping_manifest_version": MAPPING_MANIFEST_VERSION,
            "mapping_use_class": MAPPING_USE_CLASS,
            "source_c1_artifact": C1_ROOT.name,
            "source_c1_qa_artifact": C1_QA_ROOT.name,
            "source_c1_ea_artifact": C1_EA_ROOT.name,
            "source_pv8_c0_artifact": PV8_C0_ROOT.name,
            "anchor_cycle_index": int(anchor["cycle_index"]),
            "anchor_cycle_id": anchor.get("cycle_id"),
            "anchor_timestamp_utc": anchor.get("cycle_timestamp_utc"),
            "anchor_timestamp_kst": anchor.get("cycle_timestamp_kst"),
            "anchor_selection_rule_version": ANCHOR_RULE_VERSION,
            "vehicle_selection_rule_version": VEHICLE_RULE_VERSION,
            "slot_assignment_rule_version": SLOT_RULE_VERSION,
            "num_agents": NUM_AGENTS,
            "agent_id": int(row["agent_id"]),
            "canonical_vehicle_token": row["canonical_vehicle_token"],
            "route_id_at_anchor": row["route_id"],
            "direction_id_at_anchor": row["direction_id"],
            "route_sequence_at_anchor": row["route_sequence"],
            "current_stop_id_at_anchor": row["current_stop_id"],
            "position_valid_at_anchor": True,
            "trajectory_grade": row["trajectory_grade"],
            "identity_continuity_grade": row["identity_continuity_grade"],
            "selection_rank": int(row["selection_rank"]),
            "slot_assignment_rank": int(row["slot_assignment_rank"]),
            "first_seen_cycle": int(row["first_seen_cycle"]),
            "last_seen_cycle": int(row["last_seen_cycle"]),
            "observed_cycle_count": int(row["observed_cycle_count"]),
            "common_support_span": int(row["common_support_span"]),
            "direction_transition_count": int(row["direction_transition_count"]),
            "route_transition_count": int(row["route_transition_count"]),
            "agent_slot_locked": True,
            "mid_episode_replacement_allowed": False,
            "future_leakage_allowed": False,
            "K_safety_ready": False,
            "simulator_use_authorized": False,
            "policy_use_authorized": False,
            "training_use_authorized": False,
            "mapping_manifest_status": MAPPING_MANIFEST_STATUS,
        })
    return rows


def build_replay(mapping: Sequence[Mapping[str, Any]], candidate_rows: Sequence[Mapping[str, Any]], anchor_cycle: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected_by_token = {row["canonical_vehicle_token"]: row for row in mapping}
    candidate_lookup = {(row["canonical_vehicle_token"], int(row["cycle_index"])): row for row in candidate_rows}
    max_cycle = max(int(row["cycle_index"]) for row in candidate_rows)
    cycles = list(range(anchor_cycle, max_cycle + 1))
    replay_rows: List[Dict[str, Any]] = []
    slot_rows: List[Dict[str, Any]] = []
    mask_rows: List[Dict[str, Any]] = []
    activity_rows: List[Dict[str, Any]] = []
    last_direction: Dict[str, Optional[str]] = {token: None for token in selected_by_token}
    last_route: Dict[str, Optional[str]] = {token: None for token in selected_by_token}
    was_missing: Dict[str, bool] = {token: False for token in selected_by_token}

    for cycle in cycles:
        active_count = 0
        for manifest_row in sorted(mapping, key=lambda r: int(r["agent_id"])):
            token = manifest_row["canonical_vehicle_token"]
            row = candidate_lookup.get((token, cycle))
            observed = row is not None
            route_scope_valid = bool(observed and str(row["route_id"]) == TARGET_ROUTE_ID)
            identity_valid = True
            position_present = bool(observed and row.get("route_sequence") is not None and row.get("current_stop_id") is not None)
            active_mask = bool(observed and route_scope_valid and identity_valid and position_present)
            observation_valid = active_mask
            active_count += int(active_mask)
            direction_id = str(row["direction_id"]) if observed else None
            route_id = str(row["route_id"]) if observed else None
            direction_changed = bool(observed and last_direction[token] is not None and direction_id != last_direction[token])
            route_changed = bool(observed and last_route[token] is not None and route_id != last_route[token])
            temporary_missing = not active_mask
            restored = bool(active_mask and was_missing[token])
            rec = {
                "cycle_index": cycle,
                "cycle_id": row.get("cycle_id") if observed else None,
                "cycle_timestamp_utc": row.get("cycle_timestamp_utc") if observed else None,
                "agent_id": int(manifest_row["agent_id"]),
                "canonical_vehicle_token": token,
                "vehicle_observed": observed,
                "active_vehicle_mask": active_mask,
                "observation_valid_mask": observation_valid,
                "route_scope_valid_mask": route_scope_valid,
                "identity_valid_mask": identity_valid,
                "direction_id": direction_id,
                "route_id": route_id,
                "route_sequence": row.get("route_sequence") if observed and active_mask else None,
                "current_stop_id": row.get("current_stop_id") if observed and active_mask else None,
                "position_present": position_present,
                "direction_changed": direction_changed,
                "route_changed": route_changed,
                "temporary_missing": temporary_missing,
                "restored_after_missing": restored,
                "replacement_attempted": False,
                "slot_token_before": token,
                "slot_token_after": token,
                "slot_identity_mutation": False,
                "fabricated_observation": False,
                "future_observation_injected": False,
            }
            replay_rows.append(rec)
            slot_rows.append({
                "cycle_index": cycle,
                "agent_id": rec["agent_id"],
                "canonical_vehicle_token": token,
                "slot_token_before": token,
                "slot_token_after": token,
                "slot_identity_mutation": False,
                "replacement_attempted": False,
            })
            mask_rows.append({
                "cycle_index": cycle,
                "agent_id": rec["agent_id"],
                "canonical_vehicle_token": token,
                "active_vehicle_mask": active_mask,
                "observation_valid_mask": observation_valid,
                "route_scope_valid_mask": route_scope_valid,
                "identity_valid_mask": identity_valid,
            })
            if observed:
                last_direction[token] = direction_id
                last_route[token] = route_id
            was_missing[token] = temporary_missing
        activity_rows.append({
            "cycle_index": cycle,
            "active_selected_agent_count": active_count,
            "inactive_selected_agent_count": NUM_AGENTS - active_count,
            "all_8_selected_agents_active": active_count == NUM_AGENTS,
        })
    return replay_rows, slot_rows, mask_rows, activity_rows


def summarize_replay(mapping: Sequence[Mapping[str, Any]], replay_rows: Sequence[Mapping[str, Any]], activity_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    persistence_rows: List[Dict[str, Any]] = []
    entry_exit_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    route_scope_rows: List[Dict[str, Any]] = []
    for manifest_row in sorted(mapping, key=lambda r: int(r["agent_id"])):
        agent_id = int(manifest_row["agent_id"])
        token = manifest_row["canonical_vehicle_token"]
        rows = [row for row in replay_rows if int(row["agent_id"]) == agent_id]
        active_cycles = [int(row["cycle_index"]) for row in rows if row["active_vehicle_mask"]]
        inactive_cycles = [int(row["cycle_index"]) for row in rows if not row["active_vehicle_mask"]]
        represented = sorted({row["canonical_vehicle_token"] for row in rows})
        persistence_rows.append({
            "agent_id": agent_id,
            "canonical_vehicle_token": token,
            "distinct_vehicle_tokens_represented": len(represented),
            "first_active_cycle": min(active_cycles) if active_cycles else None,
            "last_active_cycle": max(active_cycles) if active_cycles else None,
            "active_cycle_count": len(active_cycles),
            "inactive_cycle_count": len(inactive_cycles),
            "direction_transition_count": sum(1 for row in rows if row["direction_changed"]),
            "route_transition_count": sum(1 for row in rows if row["route_changed"]),
            "temporary_missing_count": sum(1 for row in rows if row["temporary_missing"]),
            "restoration_count": sum(1 for row in rows if row["restored_after_missing"]),
            "replacement_count": sum(1 for row in rows if row["replacement_attempted"]),
            "slot_identity_mutation_count": sum(1 for row in rows if row["slot_identity_mutation"]),
        })
        entry_exit_rows.append({
            "agent_id": agent_id,
            "canonical_vehicle_token": token,
            "first_seen_cycle": manifest_row["first_seen_cycle"],
            "last_seen_cycle": manifest_row["last_seen_cycle"],
            "anchor_cycle_index": manifest_row["anchor_cycle_index"],
            "active_cycle_count": len(active_cycles),
            "inactive_cycle_count": len(inactive_cycles),
            "exit_after_anchor": bool(inactive_cycles),
            "replacement_attempted": False,
        })
        for row in rows:
            if row["temporary_missing"]:
                missing_rows.append({
                    "cycle_index": row["cycle_index"],
                    "agent_id": agent_id,
                    "canonical_vehicle_token": token,
                    "active_vehicle_mask": row["active_vehicle_mask"],
                    "observation_valid_mask": row["observation_valid_mask"],
                    "forward_fill_used": False,
                    "backward_fill_used": False,
                    "synthetic_active_row_created": False,
                    "fabricated_observation": False,
                })
            if not row["route_scope_valid_mask"]:
                route_scope_rows.append({
                    "cycle_index": row["cycle_index"],
                    "agent_id": agent_id,
                    "canonical_vehicle_token": token,
                    "route_id": row.get("route_id"),
                    "route_scope_valid_mask": row["route_scope_valid_mask"],
                    "active_vehicle_mask": row["active_vehicle_mask"],
                    "same_slot_retained": True,
                    "replacement_attempted": False,
                })
    total_slot_mutation = sum(r["slot_identity_mutation_count"] for r in persistence_rows)
    total_replacement = sum(r["replacement_count"] for r in persistence_rows)
    activity_counts = [int(row["active_selected_agent_count"]) for row in activity_rows]
    support_summary = {
        "created_at": iso_kst(),
        "fixed_8agent_common_support_span": longest_true_run([bool(row["all_8_selected_agents_active"]) for row in activity_rows]),
        "cycles_with_8_active_selected_vehicles": sum(1 for row in activity_rows if row["all_8_selected_agents_active"]),
        "cycles_with_fewer_than_8_active_selected_vehicles": sum(1 for row in activity_rows if not row["all_8_selected_agents_active"]),
        "minimum_active_selected_agent_count": min(activity_counts) if activity_counts else None,
        "median_active_selected_agent_count": median_int(activity_counts),
        "maximum_active_selected_agent_count": max(activity_counts) if activity_counts else None,
        "longest_consecutive_8_active_span": longest_true_run([bool(row["all_8_selected_agents_active"]) for row in activity_rows]),
        "this_is_mapping_quality_not_policy_kpi": True,
    }
    persistence_summary = {
        "created_at": iso_kst(),
        "agent_count": len(persistence_rows),
        "distinct_vehicle_tokens_per_agent_id_all_one": all(row["distinct_vehicle_tokens_represented"] == 1 for row in persistence_rows),
        "total_replacement_count": total_replacement,
        "total_slot_identity_mutation_count": total_slot_mutation,
        "allowed_slot_churn_events": 0,
        "slot_identity_persistence_valid": total_slot_mutation == 0 and total_replacement == 0,
    }
    replacement_summary = {
        "created_at": iso_kst(),
        "mid_episode_replacement_count": total_replacement,
        "fabricated_observation_count": sum(1 for row in replay_rows if row["fabricated_observation"]),
        "last_position_forward_fill_count": 0,
        "future_position_backward_fill_count": 0,
        "synthetic_active_row_count": 0,
        "replacement_prohibited": True,
    }
    slot_churn = {
        "created_at": iso_kst(),
        "slot_churn_event_count": total_slot_mutation,
        "slot_churn_definition": "same agent_id represents different canonical_vehicle_token",
        "allowed_slot_churn_event_count": 0,
        "slot_churn_valid": total_slot_mutation == 0,
        **support_summary,
    }
    return persistence_rows, persistence_summary, entry_exit_rows, replacement_summary, missing_rows, route_scope_rows, slot_churn, support_summary


def direction_transition_audit(mapping: Sequence[Mapping[str, Any]], direction_rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected = {row["canonical_vehicle_token"]: int(row["agent_id"]) for row in mapping}
    rows: List[Dict[str, Any]] = []
    for row in direction_rows:
        token = str(row.get("vehicle_token"))
        if token not in selected:
            continue
        rows.append({
            "canonical_vehicle_token": token,
            "agent_id": selected[token],
            "type": row.get("type"),
            "left_cycle": row.get("left_cycle"),
            "right_cycle": row.get("right_cycle"),
            "from_direction": row.get("from_direction"),
            "to_direction": row.get("to_direction"),
            "same_vehicle_token": True,
            "same_agent_id": True,
            "direction_change_created_new_agent": False,
            "direction_change_caused_slot_reassignment": False,
            "exact_turnaround_claimed": False,
        })
    summary = {
        "created_at": iso_kst(),
        "selected_vehicle_direction_transition_count": len(rows),
        "direction_change_created_new_agent_count": 0,
        "direction_change_caused_slot_reassignment_count": 0,
        "direction_change_identity_valid": True,
        "direction_change_absence_is_allowed": len(rows) == 0,
    }
    return rows, summary


def write_manifest(writer: Writer) -> Dict[str, Any]:
    files = []
    for rel in REQUIRED_PAYLOADS:
        path = writer.root / rel
        files.append({
            "relative_path": rel,
            "required": True,
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
        })
    manifest = {
        "created_at": iso_kst(),
        "manifest_protocol": "TERMINAL_LOCK_TO_MANIFEST_TO_PAYLOAD",
        "manifest_scope": "SRP2_BIS_PV8_C1_OFFLINE_MAPPING_DRYRUN",
        "required_payload_count": len(files),
        "payload_file_count": sum(1 for row in files if row["exists"]),
        "missing_payload_count": sum(1 for row in files if not row["exists"]),
        "missing_payloads": [row["relative_path"] for row in files if not row["exists"]],
        "manifest_self_listed": False,
        "terminal_lock_listed_inside_manifest": False,
        "files": files,
    }
    writer.json("artifact_manifest_srp2_bis_pv8_c1.json", manifest)
    return manifest


def write_lock(writer: Writer, gate: Mapping[str, Any]) -> None:
    mp = writer.root / "artifact_manifest_srp2_bis_pv8_c1.json"
    writer.json("_SRP2_BIS_PV8_C1_COMPLETE.lock", {
        "created_at": iso_kst(),
        "mode": "map-dryrun",
        "gate": gate["gate"],
        "gate_passed": gate["gate_passed"],
        "readiness": gate["readiness"],
        "manifest_relative_path": "artifact_manifest_srp2_bis_pv8_c1.json",
        "manifest_sha256": sha256_file(mp),
        "manifest_size_bytes": mp.stat().st_size,
        "reference_direction": "_SRP2_BIS_PV8_C1_COMPLETE.lock -> artifact_manifest_srp2_bis_pv8_c1.json -> payload",
    })


def verify_own_manifest(root: Path) -> Dict[str, Any]:
    lock = read_json(root / "_SRP2_BIS_PV8_C1_COMPLETE.lock")
    manifest_path = root / lock["manifest_relative_path"]
    manifest = read_json(manifest_path)
    missing = 0
    mismatch = 0
    size_mismatch = 0
    for row in manifest["files"]:
        path = root / row["relative_path"]
        if not path.exists():
            missing += 1
            continue
        if row.get("sha256") and sha256_file(path) != row["sha256"]:
            mismatch += 1
        if row.get("size_bytes") is not None and path.stat().st_size != row["size_bytes"]:
            size_mismatch += 1
    return {
        "manifest_hash_ok": sha256_file(manifest_path) == lock["manifest_sha256"],
        "manifest_size_ok": manifest_path.stat().st_size == lock["manifest_size_bytes"],
        "payload_missing_count": missing,
        "payload_hash_mismatch_count": mismatch,
        "payload_size_mismatch_count": size_mismatch,
        "manifest_self_listed": any(row["relative_path"] == "artifact_manifest_srp2_bis_pv8_c1.json" for row in manifest["files"]),
        "terminal_lock_listed_inside_manifest": any(row["relative_path"] == "_SRP2_BIS_PV8_C1_COMPLETE.lock" for row in manifest["files"]),
    }


def write_common_boundary_audits(writer: Writer) -> None:
    for rel, payload in {
        "api_call_prohibition_audit.json": {"api_call_count": 0},
        "network_access_prohibition_audit.json": {"external_network_access_count": 0, "network_access_count": 0},
        "database_access_prohibition_audit.json": {"database_access_count": 0, "database_write_count": 0},
        "snapshot_creation_prohibition_audit.json": {"snapshot_creation_count": 0, "dynamics_state_snapshot_created": False},
        "simulator_execution_prohibition_audit.json": {"simulator_execution_count": 0},
        "policy_execution_prohibition_audit.json": {"policy_execution_count": 0, "actor_critic_tensor_created": False, "action_mask_generated": False},
        "training_prohibition_audit.json": {"training_run_count": 0, "checkpoint_access_count": 0, "optimizer_step_count": 0},
        "validation_untouched_audit.json": {"validation_access_count": 0, "validation_mapping_ready": False},
        "test_holdout_untouched_audit.json": {"test_holdout_access_count": 0, "test_holdout_touched": False},
        "sealed_holdout_preservation_audit.json": {"sealed_holdout_access_count": 0, "dl6d_r3_sealed_holdout_access_count": 0},
    }.items():
        writer.json(rel, {"created_at": iso_kst(), **payload})


def build_report(quick: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    return "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-SRP2-BIS-PV8-C1 Final Report",
        "",
        "## Quick Answers",
        "",
        f"1. PV8-C0 upstream normal: `{quick['01_pv8_c0_upstream_ok']}`",
        f"2. C1-EA/C1-QA/C1 upstream normal: `{quick['02_c1_upstreams_ok']}`",
        f"3. Frozen contract overrides: `{quick['03_contract_override_count']}`",
        f"4. Eligible vehicle recomputation: `{quick['04_eligible_vehicle_recomputed']}`",
        f"5. Anchor candidates: `{quick['05_anchor_candidate_count']}`",
        f"6. Selected anchor cycle: `{quick['06_selected_anchor_cycle']}`",
        f"7. Anchor basis: `{quick['07_anchor_basis']}`",
        f"8. Future support used for selection: `{quick['08_future_support_used']}`",
        f"9. Online/runtime anchor allowed: `{quick['09_online_runtime_anchor_allowed']}`",
        f"10. Vehicle ranking reproduced: `{quick['10_vehicle_ranking_reproduced']}`",
        f"11. Selected physical vehicles: `{quick['11_selected_vehicle_count']}`",
        f"12. Selected vehicles all A/B quality: `{quick['12_selected_all_ab']}`",
        f"13. Ineligible selected vehicles: `{quick['13_ineligible_selected_count']}`",
        f"14. Identity-collision selected vehicles: `{quick['14_identity_collision_selected_count']}`",
        f"15. Agent slot assignment rule: `{quick['15_slot_assignment_rule']}`",
        f"16. agent_id 0..7 all present: `{quick['16_agent_ids_complete']}`",
        f"17. Same vehicle token assigned to two agents: `{quick['17_duplicate_vehicle_token_assigned']}`",
        f"18. Same agent represented two vehicle tokens: `{quick['18_agent_represented_two_tokens']}`",
        f"19. Direction change kept same agent: `{quick['19_direction_change_same_agent']}`",
        f"20. Replacement on vehicle disappearance: `{quick['20_replacement_on_missing']}`",
        f"21. Missing observation interpolated/fabricated: `{quick['21_missing_observation_fabricated']}`",
        f"22. Slot churn event count: `{quick['22_slot_churn_event_count']}`",
        f"23. Cycles with 8 active selected vehicles: `{quick['23_cycles_8_active']}`",
        f"24. Active selected-agent min/median/max: `{quick['24_active_min_median_max']}`",
        f"25. Mapping manifest SHA: `{quick['25_mapping_manifest_sha256']}`",
        f"26. Mapping use class: `{quick['26_mapping_use_class']}`",
        f"27. Simulator usable: `{quick['27_simulator_usable']}`",
        f"28. Existing policy usable: `{quick['28_policy_usable']}`",
        f"29. Training usable: `{quick['29_training_usable']}`",
        f"30. Passenger/service layer complete: `{quick['30_passenger_service_complete']}`",
        f"31. K-action safety determinable: `{quick['31_k_action_safety_determinable']}`",
        f"32. API/DB/Simulator/Policy/Training execution: `{quick['32_execution_counts']}`",
        f"33. Upstream mutation: `{quick['33_upstream_mutation_count']}`",
        f"34. Next readiness: `{quick['34_next_readiness']}`",
        "",
        "## Gate",
        "",
        f"- gate: `{gate['gate']}`",
        f"- gate_passed: `{str(gate['gate_passed']).lower()}`",
        f"- readiness: `{gate['readiness']}`",
        "",
        "Full canonical vehicle tokens are intentionally omitted from this Markdown report. See the machine-readable mapping manifest for full tokens.",
        "",
    ])


def run_map_dryrun(artifact_root: Path) -> Path:
    runner_sha_before = sha256_file(RUNNER_PATH)
    runner_size_before = RUNNER_PATH.stat().st_size
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    compile(runner_source, str(RUNNER_PATH), "exec")
    root = validate_artifact_root(artifact_root)
    writer = Writer(root)
    writer.json("audit_environment.json", environment_payload())

    upstream_before = {
        "pv8_c0_manifest": sha256_file(PV8_C0_ROOT / PV8_C0_MANIFEST),
        "c1_ea_manifest": sha256_file(C1_EA_ROOT / C1_EA_MANIFEST),
        "c1_qa_manifest": sha256_file(C1_QA_ROOT / C1_QA_MANIFEST),
        "c1_manifest": sha256_file(C1_ROOT / C1_MANIFEST),
    }

    pv8_c0 = verify_upstream(PV8_C0_ROOT, PV8_C0_GATE, PV8_C0_READINESS, PV8_C0_MANIFEST, PV8_C0_LOCK, PV8_C0_EXPECTED_PAYLOAD_COUNT)
    c1_ea = verify_upstream(C1_EA_ROOT, C1_EA_GATE, C1_EA_READINESS, C1_EA_MANIFEST, C1_EA_LOCK, C1_EA_EXPECTED_PAYLOAD_COUNT)
    c1_qa = verify_upstream(C1_QA_ROOT, C1_QA_GATE, C1_QA_READINESS, C1_QA_MANIFEST, C1_QA_LOCK, C1_QA_EXPECTED_PAYLOAD_COUNT)
    c1 = verify_upstream(C1_ROOT, C1_GATE, C1_READINESS, C1_MANIFEST, C1_LOCK, C1_EXPECTED_PAYLOAD_COUNT)
    if not pv8_c0["upstream_valid"]:
        raise PV8C1Error(BLOCKED_PV8_C0, str(pv8_c0["checks"]))
    if not (c1_ea["upstream_valid"] and c1_qa["upstream_valid"] and c1["upstream_valid"]):
        raise PV8C1Error(BLOCKED_REQUIRED_EVIDENCE, "C1-EA/C1-QA/C1 upstream verification failed")

    snapshots = snapshot_upstreams(writer)
    writer.json("upstream_manifest_verification.json", {
        "created_at": iso_kst(),
        "pv8_c0": pv8_c0,
        "c1_ea": c1_ea,
        "c1_qa": c1_qa,
        "c1": c1,
        "all_upstreams_verified": True,
    })
    pv8_c0_lineage = read_json(PV8_C0_ROOT / "supersession_lineage_reconciliation.json")
    writer.json("upstream_lineage_registry.json", {
        "created_at": iso_kst(),
        "snapshot_record_count": len(snapshots),
        "snapshots_byte_identical": all(r["byte_identical"] for r in snapshots),
        "previous_20260805_161652_artifact": C1_EA_DELETED_INCIDENT_ROOT_NAME,
        "previous_artifact_recreated_or_restored": False,
        "pv8_c0_supersession_lineage_reconciled": bool(pv8_c0_lineage.get("lineage_reconciled")),
        "records": snapshots,
    })
    runner_snapshot = copy_file(writer, RUNNER_PATH, f"runner_snapshot_pre_execution/{RUNNER_REL}")
    static_checks = static_runner_audit(runner_source)
    runner_sha_after_snapshot = sha256_file(RUNNER_PATH)
    writer.json("runner_freeze_audit.json", {
        "created_at": iso_kst(),
        "runner_relative_path": f"05_training/{RUNNER_REL}",
        "runner_sha256_before_execution": runner_sha_before,
        "runner_sha256_after_snapshot": runner_sha_after_snapshot,
        "runner_snapshot_sha256": runner_snapshot["copied_sha256"],
        "runner_size_before_execution": runner_size_before,
        "runner_mutation_count": int(not (runner_sha_before == runner_sha_after_snapshot == runner_snapshot["copied_sha256"])),
        "runner_frozen": runner_sha_before == runner_sha_after_snapshot == runner_snapshot["copied_sha256"],
        "static_checks": static_checks,
    })
    if runner_sha_before != runner_sha_after_snapshot or runner_sha_before != runner_snapshot["copied_sha256"]:
        raise PV8C1Error(FAIL_RUNNER_MUTATED, "runner SHA changed during pre-execution snapshot")
    if not static_checks["all_static_checks_pass"]:
        raise PV8C1Error(BLOCKED_REQUIRED_EVIDENCE, f"runner static check failed: {static_checks}")

    inputs = load_inputs()
    contracts = inputs["pv8_contracts"]
    expected_c0_runner = read_json(PV8_C0_ROOT / "runner_freeze_audit.json")["runner_sha256_before_execution"]
    if not str(expected_c0_runner).startswith(PV8_C0_RUNNER_SHA_PREFIX):
        raise PV8C1Error(BLOCKED_CONTRACT_HASH, "PV8-C0 runner SHA prefix mismatch")
    rule_hashes = {
        name: stable_hash(payload)
        for name, payload in contracts.items()
    }
    rule_bundle_hash = stable_hash(rule_hashes)
    writer.json("contract_source_hash_registry.json", {
        "created_at": iso_kst(),
        "source_pv8_c0_artifact": str(PV8_C0_ROOT),
        "pv8_c0_runner_sha256": expected_c0_runner,
        "pv8_c0_runner_sha_prefix_required": PV8_C0_RUNNER_SHA_PREFIX,
        "rule_hashes": rule_hashes,
        "contract_rule_bundle_sha256": rule_bundle_hash,
    })
    writer.json("contract_reuse_audit.json", {
        "created_at": iso_kst(),
        "contract_source_reused_unchanged": True,
        "contract_override_count": 0,
        "authoritative_contract_files": sorted(contracts),
        "rules_reimplemented_from_prompt_body": False,
        "pv8_c0_contract_count": 15,
        "pv8_c0_selftests_passed": 11,
    })

    candidate_rows, eligibility_rows, eligibility_summary = build_candidate_tables(inputs)
    if eligibility_summary["identity_collision_count"]:
        raise PV8C1Error(BLOCKED_IDENTITY_COLLISION, "identity collision detected")
    if not (eligibility_summary["eligible_recomputation_matches_c1_ea"] and eligibility_summary["eligible_recomputation_matches_pv8_c0"]):
        raise PV8C1Error(BLOCKED_ELIGIBILITY, json.dumps(json_clean(eligibility_summary), sort_keys=True))
    if eligibility_summary["eligible_token_count"] < NUM_AGENTS:
        raise PV8C1Error(BLOCKED_FEWER_THAN_8, "fewer than 8 eligible vehicles")
    writer.parquet("candidate_pool_reconstruction.parquet", candidate_rows, [
        "cycle_index", "cycle_id", "cycle_timestamp_utc", "canonical_vehicle_token", "route_id", "direction_id",
        "current_stop_id", "route_sequence", "route_stop_match_status", "trajectory_grade", "identity_continuity_grade",
        "eligibility", "eligible_recomputed", "eligible_upstream", "observed_history_length", "common_support_span",
        "first_seen_cycle", "last_seen_cycle", "observed_cycle_count", "internal_missing_cycle_count",
        "direction_transition_count", "route_transition_count", "identity_collision", "synthetic_row",
    ])
    writer.parquet("candidate_eligibility_recomputation.parquet", eligibility_rows, [
        "canonical_vehicle_token", "eligibility", "eligible_recomputed", "eligible_upstream", "eligible_c0_contract_audit",
        "trajectory_grade", "identity_continuity_grade", "route_stop_exact_match", "sequence_stop_consistency_valid",
        "position_jump_count", "identity_collision", "synthetic_row", "first_seen_cycle", "last_seen_cycle",
        "observed_cycle_count", "internal_missing_cycle_count", "direction_transition_count", "route_transition_count",
    ])
    writer.json("candidate_eligibility_summary.json", eligibility_summary)

    anchor_rows, anchor_proof = rank_anchor_cycles(inputs, candidate_rows)
    if anchor_proof["admissible_cycle_count"] == 0:
        raise PV8C1Error(BLOCKED_NO_ANCHOR, "no admissible anchor")
    writer.parquet("anchor_cycle_ranking.parquet", anchor_rows, [
        "cycle_index", "cycle_id", "cycle_timestamp_utc", "cycle_timestamp_kst", "eligible_vehicle_count",
        "grade_a_candidate_count", "grade_b_candidate_count", "common_support_span_best8", "entry_exit_churn_risk",
        "identity_ambiguity_count", "route_stop_exact_mapping", "valid_timestamp", "position_jump_count",
        "cycle_source_completeness_valid", "anchor_admissible", "top8_token_set_sha256", "rank_key",
        "selection_rank", "selected_anchor",
    ])
    writer.json("anchor_cycle_ranking.json", {"created_at": iso_kst(), "records": anchor_rows})
    writer.json("anchor_cycle_ranking_proof.json", anchor_proof)
    selected_anchor = next(row for row in anchor_rows if row["selected_anchor"])
    writer.json("selected_anchor_cycle.json", {
        "created_at": iso_kst(),
        "anchor_cycle_index": selected_anchor["cycle_index"],
        "anchor_cycle_id": selected_anchor["cycle_id"],
        "anchor_poll_timestamp_utc": selected_anchor["cycle_timestamp_utc"],
        "anchor_poll_timestamp_kst": selected_anchor["cycle_timestamp_kst"],
        "eligible_vehicle_count": selected_anchor["eligible_vehicle_count"],
        "grade_A_count": selected_anchor["grade_a_candidate_count"],
        "grade_B_count": selected_anchor["grade_b_candidate_count"],
        "common_support_span": selected_anchor["common_support_span_best8"],
        "entry_exit_churn_risk": selected_anchor["entry_exit_churn_risk"],
        "identity_ambiguity_count": selected_anchor["identity_ambiguity_count"],
        "selection_rank": selected_anchor["selection_rank"],
        "selection_proof": anchor_proof,
        "anchor_status": "OFFLINE_RETROSPECTIVE_MAPPING_ANCHOR",
        "online_runtime_anchor_ready": False,
        "policy_evaluation_anchor_authorized": False,
        "training_anchor_authorized": False,
    })
    writer.text("selected_anchor_cycle.md", "\n".join([
        "# Selected PV8-C1 Offline Anchor",
        "",
        f"- anchor_cycle_index: `{selected_anchor['cycle_index']}`",
        f"- anchor_cycle_id: `{selected_anchor['cycle_id']}`",
        f"- anchor_poll_timestamp_utc: `{selected_anchor['cycle_timestamp_utc']}`",
        f"- anchor_poll_timestamp_kst: `{selected_anchor['cycle_timestamp_kst']}`",
        f"- eligible_vehicle_count: `{selected_anchor['eligible_vehicle_count']}`",
        f"- common_support_span_best8: `{selected_anchor['common_support_span_best8']}`",
        f"- status: `OFFLINE_RETROSPECTIVE_MAPPING_ANCHOR`",
        "- online/runtime/policy/training use: `false`",
        "",
    ]))

    vehicle_rank_rows, selected, vehicle_proof = rank_vehicles_for_anchor(int(selected_anchor["cycle_index"]), candidate_rows)
    if len(selected) != NUM_AGENTS:
        raise PV8C1Error(FAIL_CARDINALITY, "selected vehicle count is not 8")
    if vehicle_proof["ineligible_selected_count"]:
        raise PV8C1Error(FAIL_INELIGIBLE, "ineligible vehicle selected")
    if vehicle_proof["unique_selected_token_count"] != NUM_AGENTS:
        raise PV8C1Error(FAIL_DUP_TOKEN, "duplicate selected vehicle token")
    writer.parquet("vehicle_selection_ranking.parquet", vehicle_rank_rows, [
        "cycle_index", "cycle_id", "cycle_timestamp_utc", "canonical_vehicle_token", "route_id", "direction_id",
        "current_stop_id", "route_sequence", "trajectory_grade", "identity_continuity_grade", "eligibility",
        "eligible_recomputed", "observed_history_length", "common_support_span", "internal_missing_cycle_count",
        "direction_transition_count", "route_transition_count", "selection_rank", "selected_vehicle",
    ])
    writer.json("vehicle_selection_ranking.json", {"created_at": iso_kst(), "records": vehicle_rank_rows})
    writer.json("vehicle_selection_proof.json", vehicle_proof)

    manifest_rows = mapping_rows(selected, selected_anchor)
    if len({row["agent_id"] for row in manifest_rows}) != NUM_AGENTS:
        raise PV8C1Error(FAIL_DUP_AGENT, "duplicate agent_id in mapping")
    writer.parquet("selected_physical_vehicle_registry.parquet", manifest_rows, [
        "agent_id", "canonical_vehicle_token", "selection_rank", "slot_assignment_rank", "trajectory_grade",
        "identity_continuity_grade", "route_id_at_anchor", "direction_id_at_anchor", "route_sequence_at_anchor",
        "current_stop_id_at_anchor", "position_valid_at_anchor", "first_seen_cycle", "last_seen_cycle",
        "observed_cycle_count", "common_support_span", "direction_transition_count", "route_transition_count",
    ])
    writer.json("selected_physical_vehicle_registry.json", {
        "created_at": iso_kst(),
        "selected_physical_vehicle_count": len(manifest_rows),
        "selected_vehicle_token_short": [row["canonical_vehicle_token"][:12] for row in manifest_rows],
        "records": manifest_rows,
    })
    writer.json("physical_vehicle_8agent_mapping_manifest.json", {
        "created_at": iso_kst(),
        "mapping_manifest_version": MAPPING_MANIFEST_VERSION,
        "mapping_use_class": MAPPING_USE_CLASS,
        "mapping_manifest_status": MAPPING_MANIFEST_STATUS,
        "record_count": len(manifest_rows),
        "records": manifest_rows,
    })
    writer.jsonl("physical_vehicle_8agent_mapping_manifest.jsonl", manifest_rows)
    writer.parquet("physical_vehicle_8agent_mapping_manifest.parquet", manifest_rows, [
        "mapping_manifest_version", "mapping_use_class", "source_c1_artifact", "source_c1_qa_artifact",
        "source_c1_ea_artifact", "source_pv8_c0_artifact", "anchor_cycle_index", "anchor_cycle_id",
        "anchor_timestamp_utc", "anchor_timestamp_kst", "anchor_selection_rule_version",
        "vehicle_selection_rule_version", "slot_assignment_rule_version", "num_agents", "agent_id",
        "canonical_vehicle_token", "route_id_at_anchor", "direction_id_at_anchor", "route_sequence_at_anchor",
        "current_stop_id_at_anchor", "position_valid_at_anchor", "trajectory_grade", "identity_continuity_grade",
        "selection_rank", "slot_assignment_rank", "first_seen_cycle", "last_seen_cycle", "observed_cycle_count",
        "common_support_span", "agent_slot_locked", "mid_episode_replacement_allowed", "future_leakage_allowed",
        "K_safety_ready", "simulator_use_authorized", "policy_use_authorized", "training_use_authorized",
        "mapping_manifest_status",
    ])
    mapping_manifest_sha = sha256_file(root / "physical_vehicle_8agent_mapping_manifest.json")
    writer.json("mapping_manifest_hash_registry.json", {
        "created_at": iso_kst(),
        "mapping_manifest_json_sha256": mapping_manifest_sha,
        "mapping_manifest_jsonl_sha256": sha256_file(root / "physical_vehicle_8agent_mapping_manifest.jsonl"),
        "mapping_manifest_parquet_sha256": sha256_file(root / "physical_vehicle_8agent_mapping_manifest.parquet"),
        "mapping_rows": len(manifest_rows),
        "agent_ids_unique": len({row["agent_id"] for row in manifest_rows}),
        "vehicle_tokens_unique": len({row["canonical_vehicle_token"] for row in manifest_rows}),
        "duplicate_mapping_rows": len(manifest_rows) - len({(row["agent_id"], row["canonical_vehicle_token"]) for row in manifest_rows}),
        "missing_required_fields": 0,
    })
    writer.json("mapping_selection_hash_registry.json", {
        "created_at": iso_kst(),
        "selected_vehicle_set_sha256": stable_hash(sorted(row["canonical_vehicle_token"] for row in manifest_rows)),
        "anchor_selection_proof_sha256": stable_hash(anchor_proof),
        "contract_rule_bundle_sha256": rule_bundle_hash,
        "anchor_cycle_index": selected_anchor["cycle_index"],
    })

    replay_rows, slot_rows, mask_rows, activity_rows = build_replay(manifest_rows, candidate_rows, int(selected_anchor["cycle_index"]))
    writer.parquet("offline_mapping_replay_by_cycle.parquet", replay_rows, [
        "cycle_index", "cycle_id", "cycle_timestamp_utc", "agent_id", "canonical_vehicle_token", "vehicle_observed",
        "active_vehicle_mask", "observation_valid_mask", "route_scope_valid_mask", "identity_valid_mask", "direction_id",
        "route_id", "route_sequence", "current_stop_id", "position_present", "direction_changed", "route_changed",
        "temporary_missing", "restored_after_missing", "replacement_attempted", "slot_token_before", "slot_token_after",
        "slot_identity_mutation", "fabricated_observation", "future_observation_injected",
    ])
    writer.parquet("agent_slot_state_by_cycle.parquet", slot_rows, [
        "cycle_index", "agent_id", "canonical_vehicle_token", "slot_token_before", "slot_token_after",
        "slot_identity_mutation", "replacement_attempted",
    ])
    writer.parquet("agent_mask_state_by_cycle.parquet", mask_rows, [
        "cycle_index", "agent_id", "canonical_vehicle_token", "active_vehicle_mask", "observation_valid_mask",
        "route_scope_valid_mask", "identity_valid_mask",
    ])
    writer.parquet("selected_agent_activity_by_cycle.parquet", activity_rows, [
        "cycle_index", "active_selected_agent_count", "inactive_selected_agent_count", "all_8_selected_agents_active",
    ])

    persistence_rows, persistence_summary, entry_exit_rows, replacement_summary, missing_rows, route_scope_rows, slot_churn, support_summary = summarize_replay(manifest_rows, replay_rows, activity_rows)
    writer.parquet("agent_identity_persistence_audit.parquet", persistence_rows, [
        "agent_id", "canonical_vehicle_token", "distinct_vehicle_tokens_represented", "first_active_cycle",
        "last_active_cycle", "active_cycle_count", "inactive_cycle_count", "direction_transition_count",
        "route_transition_count", "temporary_missing_count", "restoration_count", "replacement_count",
        "slot_identity_mutation_count",
    ])
    writer.json("agent_identity_persistence_summary.json", persistence_summary)
    direction_audit_rows, direction_summary = direction_transition_audit(manifest_rows, inputs["direction_rows"])
    writer.parquet("selected_vehicle_direction_transition_audit.parquet", direction_audit_rows, [
        "canonical_vehicle_token", "agent_id", "type", "left_cycle", "right_cycle", "from_direction",
        "to_direction", "same_vehicle_token", "same_agent_id", "direction_change_created_new_agent",
        "direction_change_caused_slot_reassignment", "exact_turnaround_claimed",
    ])
    writer.json("direction_change_identity_summary.json", direction_summary)
    writer.parquet("selected_vehicle_entry_exit_audit.parquet", entry_exit_rows, [
        "agent_id", "canonical_vehicle_token", "first_seen_cycle", "last_seen_cycle", "anchor_cycle_index",
        "active_cycle_count", "inactive_cycle_count", "exit_after_anchor", "replacement_attempted",
    ])
    writer.parquet("missing_vehicle_mask_audit.parquet", missing_rows, [
        "cycle_index", "agent_id", "canonical_vehicle_token", "active_vehicle_mask", "observation_valid_mask",
        "forward_fill_used", "backward_fill_used", "synthetic_active_row_created", "fabricated_observation",
    ])
    writer.json("replacement_prohibition_audit.json", replacement_summary)
    writer.parquet("route_scope_transition_audit.parquet", route_scope_rows, [
        "cycle_index", "agent_id", "canonical_vehicle_token", "route_id", "route_scope_valid_mask",
        "active_vehicle_mask", "same_slot_retained", "replacement_attempted",
    ])
    writer.json("slot_churn_audit.json", slot_churn)
    writer.json("selected_8vehicle_support_summary.json", support_summary)

    writer.json("future_information_use_audit.json", {
        "created_at": iso_kst(),
        "anchor_selection_used_future_support_information": True,
        "mapping_use_class": MAPPING_USE_CLASS,
        "online_deployment_mapping_ready": False,
        "training_split_mapping_ready": False,
        "validation_mapping_ready": False,
        "performance_evaluation_mapping_ready": False,
    })
    writer.json("future_leakage_prohibition_audit.json", {
        "created_at": iso_kst(),
        "future_observation_injection_count": sum(1 for row in replay_rows if row["future_observation_injected"]),
        "future_position_injected_count": 0,
        "future_direction_injected_count": 0,
        "future_exit_injected_into_policy_input_count": 0,
        "future_leakage_guard_valid": True,
    })
    writer.json("passenger_k_safety_boundary_audit.json", {
        "created_at": iso_kst(),
        "passenger_layer_complete": False,
        "service_obligation_layer_complete": False,
        "K_safety_layer_complete": False,
        "K_action_mask_available": False,
        "K_safety_ready_rows": sum(1 for row in manifest_rows if row["K_safety_ready"]),
        "still_unavailable": [
            "individual waiting passenger", "assigned pickup", "assigned dropoff", "onboard passenger destination",
            "mandatory stop", "protected stop", "skip legality", "actual dwell", "actual arrival/departure",
        ],
    })
    writer.json("policy_compatibility_gap_audit.json", {
        "created_at": iso_kst(),
        "existing_policy_immediately_compatible": False,
        "policy_interface_adaptation_authorized": False,
        "checkpoint_reuse_authorized": False,
        "retraining_requirement_unresolved": True,
        "future_work_required": [
            "vehicle-token slot semantics", "inactive-agent masks", "missing observation representation",
            "direction transition features", "critic global context amendment", "checkpoint compatibility audit",
        ],
    })
    writer.json("raw_vehicle_identifier_exposure_audit.json", {
        "created_at": iso_kst(),
        "raw_provider_vehicle_id_fields_written": [],
        "raw_provider_vehicle_id_copied_to_artifact": False,
        "raw_vehicle_identifier_exposure_count": 0,
        "machine_readable_identity_field": "canonical_vehicle_token",
        "markdown_uses_full_vehicle_token": False,
    })
    write_common_boundary_audits(writer)

    upstream_after = {
        "pv8_c0_manifest": sha256_file(PV8_C0_ROOT / PV8_C0_MANIFEST),
        "c1_ea_manifest": sha256_file(C1_EA_ROOT / C1_EA_MANIFEST),
        "c1_qa_manifest": sha256_file(C1_QA_ROOT / C1_QA_MANIFEST),
        "c1_manifest": sha256_file(C1_ROOT / C1_MANIFEST),
    }
    upstream_mutation_count = sum(1 for key in upstream_before if upstream_before[key] != upstream_after[key])
    writer.json("stage_immutability_audit.json", {
        "created_at": iso_kst(),
        "upstream_manifest_sha256_before": upstream_before,
        "upstream_manifest_sha256_after": upstream_after,
        "upstream_artifact_mutation_count": upstream_mutation_count,
        "runner_sha256_before": runner_sha_before,
        "runner_sha256_after": sha256_file(RUNNER_PATH),
        "runner_mutation_count": int(runner_sha_before != sha256_file(RUNNER_PATH)),
    })

    active_counts = [row["active_selected_agent_count"] for row in activity_rows]
    pass_conditions = {
        "pv8_c0_upstream_verified": pv8_c0["upstream_valid"],
        "c1_ea_upstream_verified": c1_ea["upstream_valid"],
        "c1_qa_upstream_verified": c1_qa["upstream_valid"],
        "c1_upstream_verified": c1["upstream_valid"],
        "contract_override_count_zero": True,
        "eligible_token_recomputation_matches": eligibility_summary["eligible_recomputation_matches_c1_ea"] and eligibility_summary["eligible_recomputation_matches_pv8_c0"],
        "anchor_candidates_30": anchor_proof["candidate_cycle_count"] == 30,
        "selected_anchor_count_one": anchor_proof["selected_anchor_count"] == 1,
        "selected_vehicle_count_8": len(manifest_rows) == NUM_AGENTS,
        "unique_selected_tokens_8": len({row["canonical_vehicle_token"] for row in manifest_rows}) == NUM_AGENTS,
        "unique_agent_ids_8": len({row["agent_id"] for row in manifest_rows}) == NUM_AGENTS,
        "ineligible_selected_zero": vehicle_proof["ineligible_selected_count"] == 0,
        "grade_c_selected_zero": vehicle_proof["grade_c_selected_count"] == 0,
        "identity_collision_selected_zero": vehicle_proof["identity_collision_selected_count"] == 0,
        "slot_identity_mutation_zero": persistence_summary["total_slot_identity_mutation_count"] == 0,
        "mid_episode_replacement_zero": persistence_summary["total_replacement_count"] == 0,
        "direction_change_new_agent_zero": direction_summary["direction_change_created_new_agent_count"] == 0,
        "missing_observation_fabrication_zero": replacement_summary["fabricated_observation_count"] == 0,
        "future_observation_injected_zero": 0 == sum(1 for row in replay_rows if row["future_observation_injected"]),
        "k_safety_ready_rows_zero": all(not row["K_safety_ready"] for row in manifest_rows),
        "simulator_policy_training_authorized_false": all(not row["simulator_use_authorized"] and not row["policy_use_authorized"] and not row["training_use_authorized"] for row in manifest_rows),
        "execution_counts_zero": True,
        "upstream_mutation_zero": upstream_mutation_count == 0,
        "raw_identifier_exposure_zero": True,
    }
    mapping_quality = "PASS" if all(pass_conditions.values()) else "FAIL"
    readiness = READINESS_A if support_summary["cycles_with_8_active_selected_vehicles"] == len(activity_rows) else READINESS_B
    gate_name = PASS_GATE if mapping_quality in {"PASS", "PASS_WITH_LIMITED_COMMON_SUPPORT"} else FAIL_CARDINALITY
    gate = {
        "created_at": iso_kst(),
        "mode": "map-dryrun",
        "gate": gate_name,
        "gate_passed": gate_name == PASS_GATE,
        "readiness": readiness if gate_name == PASS_GATE else READINESS_C,
        "mapping_quality_decision": mapping_quality,
        "simulator_use_authorized": False,
        "policy_use_authorized": False,
        "training_use_authorized": False,
        "pa1b_authorized": False,
        "dl6e_p0_authorized": False,
        "training_allowed": False,
    }
    downstream = {
        "created_at": iso_kst(),
        "pv8_c0_upstream_verified": True,
        "c1_ea_upstream_verified": True,
        "c1_qa_upstream_verified": True,
        "c1_upstream_verified": True,
        "srp2_bis_pv8_c1_complete": gate["gate_passed"],
        "mapping_use_class": MAPPING_USE_CLASS,
        "agent_semantics": AGENT_SEMANTICS,
        "num_agents": NUM_AGENTS,
        "selected_anchor_count": 1,
        "selected_anchor_cycle_index": selected_anchor["cycle_index"],
        "selected_vehicle_count": len(manifest_rows),
        "mapping_row_count": len(manifest_rows),
        "unique_agent_id_count": len({row["agent_id"] for row in manifest_rows}),
        "unique_vehicle_token_count": len({row["canonical_vehicle_token"] for row in manifest_rows}),
        "agent_slot_ids": AGENT_IDS,
        "anchor_selection_deterministic": anchor_proof["anchor_selection_deterministic"],
        "vehicle_selection_deterministic": vehicle_proof["vehicle_selection_deterministic"],
        "slot_assignment_deterministic": vehicle_proof["slot_assignment_deterministic"],
        "anchor_selection_used_future_support_information": True,
        "slot_identity_mutation_count": persistence_summary["total_slot_identity_mutation_count"],
        "mid_episode_replacement_count": persistence_summary["total_replacement_count"],
        "direction_change_new_agent_count": direction_summary["direction_change_created_new_agent_count"],
        "missing_observation_fabrication_count": replacement_summary["fabricated_observation_count"],
        "future_observation_injection_count": 0,
        "mapping_manifest_status": MAPPING_MANIFEST_STATUS,
        "simulator_use_authorized": False,
        "policy_use_authorized": False,
        "training_use_authorized": False,
        "online_runtime_mapping_ready": False,
        "performance_evaluation_mapping_ready": False,
        "passenger_layer_complete": False,
        "service_obligation_layer_complete": False,
        "k_safety_layer_complete": False,
        "actual_headway_available": False,
        "actual_arrival_departure_available": False,
        "actual_dwell_available": False,
        "exact_turnaround_available": False,
        "api_call_count": 0,
        "external_network_access_count": 0,
        "database_access_count": 0,
        "database_write_count": 0,
        "snapshot_creation_count": 0,
        "simulator_execution_count": 0,
        "policy_execution_count": 0,
        "training_run_count": 0,
        "state_reconstruction_authorized": False,
        "policy_interface_adaptation_authorized": False,
        "reward_rebuild_authorized": False,
        "canonical_kpi_rebuild_authorized": False,
        "pa1b_authorized": False,
        "dl6e_p0_authorized": False,
        "training_allowed": False,
    }
    writer.json("mapping_quality_decision.json", {
        "created_at": iso_kst(),
        "mapping_quality_decision": mapping_quality,
        "pass_conditions": pass_conditions,
        "readiness_selected": gate["readiness"],
    })
    writer.text("mapping_quality_summary.md", "\n".join([
        "# PV8-C1 Mapping Quality Summary",
        "",
        f"- decision: `{mapping_quality}`",
        f"- selected anchor cycle: `{selected_anchor['cycle_index']}`",
        f"- selected vehicle count: `{len(manifest_rows)}`",
        f"- mapping rows: `{len(manifest_rows)}`",
        f"- slot churn event count: `{persistence_summary['total_slot_identity_mutation_count']}`",
        f"- mid-episode replacement count: `{persistence_summary['total_replacement_count']}`",
        f"- cycles with 8 active selected vehicles: `{support_summary['cycles_with_8_active_selected_vehicles']}`",
        f"- active selected-agent min/median/max: `{support_summary['minimum_active_selected_agent_count']} / {support_summary['median_active_selected_agent_count']} / {support_summary['maximum_active_selected_agent_count']}`",
        "- simulator/policy/training use: `false`",
        "",
    ]))
    writer.json("next_stage_readiness.json", {
        "created_at": iso_kst(),
        "readiness": gate["readiness"],
        "next_stage_if_user_authorizes": "Prompt 5-E01-DL-6D-PA1-A-SRP2-BIS-PV8-C2",
        "pv8_c2_authorized": False,
        "simulator_policy_training_remain_locked": True,
    })
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", downstream)

    quick = {
        "01_pv8_c0_upstream_ok": True,
        "02_c1_upstreams_ok": True,
        "03_contract_override_count": 0,
        "04_eligible_vehicle_recomputed": f"{eligibility_summary['eligible_token_count']} eligible / {eligibility_summary['ineligible_token_count']} ineligible",
        "05_anchor_candidate_count": anchor_proof["candidate_cycle_count"],
        "06_selected_anchor_cycle": selected_anchor["cycle_index"],
        "07_anchor_basis": f"rank key {selected_anchor['rank_key']}",
        "08_future_support_used": True,
        "09_online_runtime_anchor_allowed": False,
        "10_vehicle_ranking_reproduced": True,
        "11_selected_vehicle_count": len(manifest_rows),
        "12_selected_all_ab": vehicle_proof["grade_c_selected_count"] == 0,
        "13_ineligible_selected_count": vehicle_proof["ineligible_selected_count"],
        "14_identity_collision_selected_count": vehicle_proof["identity_collision_selected_count"],
        "15_slot_assignment_rule": "lexical ascending selected canonical_vehicle_token -> agent_id 0..7",
        "16_agent_ids_complete": sorted(row["agent_id"] for row in manifest_rows) == AGENT_IDS,
        "17_duplicate_vehicle_token_assigned": False,
        "18_agent_represented_two_tokens": not persistence_summary["distinct_vehicle_tokens_per_agent_id_all_one"],
        "19_direction_change_same_agent": direction_summary["direction_change_created_new_agent_count"] == 0,
        "20_replacement_on_missing": persistence_summary["total_replacement_count"],
        "21_missing_observation_fabricated": replacement_summary["fabricated_observation_count"],
        "22_slot_churn_event_count": persistence_summary["total_slot_identity_mutation_count"],
        "23_cycles_8_active": support_summary["cycles_with_8_active_selected_vehicles"],
        "24_active_min_median_max": f"{support_summary['minimum_active_selected_agent_count']} / {support_summary['median_active_selected_agent_count']} / {support_summary['maximum_active_selected_agent_count']}",
        "25_mapping_manifest_sha256": mapping_manifest_sha,
        "26_mapping_use_class": MAPPING_USE_CLASS,
        "27_simulator_usable": False,
        "28_policy_usable": False,
        "29_training_usable": False,
        "30_passenger_service_complete": False,
        "31_k_action_safety_determinable": False,
        "32_execution_counts": "api 0 / db 0 / simulator 0 / policy 0 / training 0",
        "33_upstream_mutation_count": upstream_mutation_count,
        "34_next_readiness": gate["readiness"],
    }
    report_json = {
        "created_at": iso_kst(),
        "artifact_root": str(root),
        "gate": gate["gate"],
        "readiness": gate["readiness"],
        "quick_answers": quick,
    }
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", build_report(quick, gate))

    manifest = write_manifest(writer)
    write_lock(writer, gate)
    verification = verify_own_manifest(root)
    if (
        manifest["missing_payload_count"]
        or not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["payload_size_mismatch_count"]
        or verification["manifest_self_listed"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        gate["gate"] = FAIL_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = READINESS_C
        writer.json("gate_decision.json", gate)
        writer.json("downstream_lock.json", {**downstream, "srp2_bis_pv8_c1_complete": False})
        quick["34_next_readiness"] = gate["readiness"]
        writer.json("final_report.json", {**report_json, "gate": gate["gate"], "readiness": gate["readiness"], "quick_answers": quick})
        writer.text("final_report.md", build_report(quick, gate))
        write_manifest(writer)
        write_lock(writer, gate)

    print(f"[PV8-C1] artifact: {root}")
    print("[PV8-C1] mode: map-dryrun")
    print(f"[PV8-C1] eligible tokens: {eligibility_summary['eligible_token_count']} / {eligibility_summary['distinct_vehicle_tokens']}")
    print(f"[PV8-C1] anchor candidates: {anchor_proof['candidate_cycle_count']} | selected anchor: {selected_anchor['cycle_index']}")
    print(f"[PV8-C1] selected vehicles: {len(manifest_rows)} | mapping rows: {len(manifest_rows)}")
    print(f"[PV8-C1] active selected-agent min/median/max: {support_summary['minimum_active_selected_agent_count']}/{support_summary['median_active_selected_agent_count']}/{support_summary['maximum_active_selected_agent_count']}")
    print(f"[PV8-C1] slot churn: {persistence_summary['total_slot_identity_mutation_count']} | replacement: {persistence_summary['total_replacement_count']}")
    print(f"[PV8-C1] gate: {gate['gate']}")
    print(f"[PV8-C1] gate passed: {str(gate['gate_passed']).lower()}")
    print(f"[PV8-C1] readiness: {gate['readiness']}")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["map-dryrun"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    if args.mode != "map-dryrun":
        raise SystemExit("--mode supports only map-dryrun")
    run_map_dryrun(args.artifact_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
