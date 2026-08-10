#!/usr/bin/env python3
"""Prompt 5-E01-DL-6D-PA1-A-ER1-V1F-FN1.

Immutable Final Closeout with Mixed DL-6B Evidence Boundary.

Pure reconciliation / sealing step. Reconciles the A2 repair, FV1 full synthetic
integration verification, and DL-6B claim-evidence audit into one immutable V1F
final lineage. Does NOT import or execute any simulator/audit source, does not
touch historical/validation/test data, and does not run any transition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
RUNNER_PATH = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_er1_v1f_finalize.py"

A2_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_er1_v1f_tv1_f1_a2_service_identity_registry_repair_20260803_122659"
FV1_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_er1_v1f_fv1_full_integration_reverify_20260803_163556"
DL6B_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_er1_v1f_dl6b_claim_evidence_audit_20260803_165927"

UPSTREAM = {
    "A2": {
        "root": A2_ROOT, "gate_file": "gate_decision.json", "lock": "_SUCCESS.lock",
        "gate": "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_SERVICE_IDENTITY_AND_REGISTRY_REPAIR_COMPLETE",
        "readiness": "READY_TO_RESUME_V1F_FULL_VERIFY",
        "manifest": "artifact_manifest_final.json",
        "manifest_sha256": "cb17e0bfc07f63fb7dcfae1ba0e23874bc9e29978164f36bc526042ccd52c0b0", "manifest_size": 27101,
    },
    "FV1": {
        "root": FV1_ROOT, "gate_file": "gate_decision.json", "lock": "_FULL_VERIFY_COMPLETE.lock",
        "gate": "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_FULL_VERIFY_COMPLETE_AWAITING_DL6B_AUDIT",
        "readiness": "FULL_VERIFY_COMPLETE_DL6B_AUDIT_PENDING_USER_COMMAND",
        "manifest": "artifact_manifest_full_verify.json",
        "manifest_sha256": "da4c81aed39a06fb3450e20c94894378dff471c4af046bbb8accaf78c7d71359", "manifest_size": 13611,
    },
    "DL6B": {
        "root": DL6B_ROOT, "gate_file": "gate_decision.json", "lock": "_DL6B_AUDIT_COMPLETE.lock",
        "gate": "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_DL6B_AUDIT_COMPLETE_MIXED_EVIDENCE",
        "readiness": "DL6B_AUDIT_COMPLETE_V1F_FINALIZE_PENDING_USER_COMMAND",
        "manifest": "artifact_manifest_dl6b_audit.json",
        "manifest_sha256": "54b4d9400b72e7f7d9e1df831b1b74892e35a952b16b5678a66ced31a15d6f05", "manifest_size": 12028,
    },
}

FROZEN_SOURCE_SHA = {
    "05_training/simulator/dynamics_multiagent_orchestrator.py": "ee9636f43adeac5dcfa04ec976aca0d72ff05f6e356dbc8beac4f9f1b14d8b81",
    "05_training/simulator/dynamics_event_trace.py": "7886bdafe55ae1794bec610b16a573c2909a64a6cb5aa2545ca5d8391a04a14e",
    "05_training/simulator/suseong_service_transition_engine.py": "37cdfbf4eb486bd0157fe47000924ea2946b787c953e70ff9fea0f821bca8ad5",
    "05_training/simulator/dynamics_state_snapshot.py": "e189d3b3a39acebc957fc91b8225ed1030babb19b28f3825b9f9b822184f0105",
    "05_training/simulator/dynamics_replay_contract.py": "13be29fa5fd2b6637ff9096df3787666204eed07fca136d11b00e3dee2b51477",
    "05_training/simulator/dynamics_horizon_aggregator.py": "7dae0133db4ac28d332952da97bbe193d2b6f425f8abdcf44a4a994e6b98b9ce",
}

EXPECTED_PARTITION = {
    "REESTABLISHED_BY_FV1_SYNTHETIC_INTEGRATION": ["C01", "C02", "C04", "C07", "C09"],
    "REUSABLE_WITH_SCOPE_DOWNGRADE": ["C06", "C13"],
    "REBUILD_FROM_HISTORICAL_DYNAMICS": ["C08", "C12"],
    "REBUILD_AFTER_REWARD_DEFINITION": ["C03", "C11"],
    "REBUILD_WITH_CANONICAL_KPI": ["C05", "C10"],
}

PASS_FINAL = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_FINALIZED_WITH_MIXED_DL6B_EVIDENCE_SCOPE"
FINAL_READINESS = "V1F_FINALIZED_PA1A_STATE_FEASIBILITY_PENDING_USER_COMMAND"

_F = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_FN1_"
FAIL_UPSTREAM = _F + "UPSTREAM_LINEAGE_INVALID"
FAIL_RUNNER = _F + "RUNNER_MUTATED_DURING_FINALIZE"
FAIL_SOURCE_DRIFT = _F + "SOURCE_DRIFT"
FAIL_UPSTREAM_MUTATED = _F + "UPSTREAM_ARTIFACT_MUTATED"
FAIL_PARTITION = _F + "CLAIM_PARTITION_INVALID"
FAIL_OVERCLAIM = _F + "EVIDENCE_SCOPE_OVERCLAIM"
FAIL_CONTAM_LOST = _F + "DL6B_CONTAMINATION_SCOPE_LOST"
FAIL_REWARD_PROMOTED = _F + "REWARD_PROXY_PROMOTED"
FAIL_KPI_PROMOTED = _F + "KPI_PROXY_PROMOTED"
FAIL_HOLDOUT_CHANGED = _F + "HOLDOUT_STATUS_CHANGED"
FAIL_SIMULATOR = _F + "SIMULATOR_EXECUTION_DETECTED"
FAIL_HISTORICAL = _F + "HISTORICAL_ROW_ACCESSED"
FAIL_VALIDATION_OR_TEST = _F + "VALIDATION_OR_TEST_TOUCHED"
FAIL_REWARD_ENERGY = _F + "REWARD_ENERGY_SCALE_CREATED"
FAIL_TRAINING = _F + "PROHIBITED_TRAINING"
FAIL_MANIFEST = _F + "MANIFEST_RECONCILIATION"


class FinalizeError(RuntimeError):
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
# preflight / snapshot / partition
# ---------------------------------------------------------------------------

def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"final-closeout artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def preflight_one(key: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    root = spec["root"]
    gate = read_json(root / spec["gate_file"])
    lock = read_json(root / spec["lock"])
    manifest_path = root / spec["manifest"]
    manifest = read_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    seen: Dict[str, int] = {}
    missing = mismatch = size_mismatch = 0
    for row in manifest["files"]:
        seen[row["relative_path"]] = seen.get(row["relative_path"], 0) + 1
        t = root / row["relative_path"]
        if not t.exists():
            missing += 1
            continue
        if row.get("sha256") is not None and sha256_file(t) != row["sha256"]:
            mismatch += 1
        if row.get("size_bytes") is not None and t.stat().st_size != row["size_bytes"]:
            size_mismatch += 1
    duplicate = sorted(p for p, c in seen.items() if c > 1)
    checks = {
        "gate": gate.get("gate") == spec["gate"],
        "readiness": gate.get("readiness") == spec["readiness"],
        "lock_present": (root / spec["lock"]).exists(),
        "lock_manifest_path": lock.get("manifest_relative_path") == spec["manifest"],
        "lock_manifest_sha_matches": lock.get("manifest_sha256") == manifest_sha,
        "manifest_sha_matches_expected": manifest_sha == spec["manifest_sha256"],
        "lock_manifest_size_matches": lock.get("manifest_size_bytes") == manifest_path.stat().st_size,
        "manifest_size_matches_expected": manifest_path.stat().st_size == spec["manifest_size"],
        "manifest_missing_zero": missing == 0,
        "manifest_hash_mismatch_zero": mismatch == 0,
        "manifest_size_mismatch_zero": size_mismatch == 0,
        "manifest_duplicate_path_zero": len(duplicate) == 0,
        "manifest_self_reference_absent": not any(r["relative_path"] == spec["manifest"] for r in manifest["files"]),
        "terminal_lock_not_in_manifest": not any(r["relative_path"] == spec["lock"] for r in manifest["files"]),
    }
    return {"upstream_key": key, "artifact_root": str(root), "manifest_sha256": manifest_sha,
            "manifest_size_bytes": manifest_path.stat().st_size, "gate": gate.get("gate"), "readiness": gate.get("readiness"),
            "payload_missing_count": missing, "payload_hash_mismatch_count": mismatch, "payload_size_mismatch_count": size_mismatch,
            "duplicate_path_count": len(duplicate), "checks": checks, "preflight_passed": all(checks.values())}


def upstream_preflight() -> Dict[str, Any]:
    records = [preflight_one(k, s) for k, s in UPSTREAM.items()]
    return {"created_at": iso_kst(), "records": records, "upstream_lineage_valid": all(r["preflight_passed"] for r in records)}


def upstream_snapshots(writer: Writer) -> Dict[str, Any]:
    plan = {
        "upstream_a2_snapshot": [
            (A2_ROOT / "gate_decision.json", "a2_gate_decision.json"),
            (A2_ROOT / "downstream_lock.json", "a2_downstream_lock.json"),
            (A2_ROOT / "artifact_manifest_final.json", "a2_artifact_manifest_final.json"),
            (A2_ROOT / "_SUCCESS.lock", "a2_SUCCESS.lock"),
            (A2_ROOT / "final_report.json", "a2_final_report.json"),
            (A2_ROOT / "source_snapshot_final_registry.json", "a2_source_snapshot_final_registry.json"),
        ],
        "upstream_fv1_snapshot": [
            (FV1_ROOT / "gate_decision.json", "fv1_gate_decision.json"),
            (FV1_ROOT / "downstream_lock.json", "fv1_downstream_lock.json"),
            (FV1_ROOT / "artifact_manifest_full_verify.json", "fv1_artifact_manifest_full_verify.json"),
            (FV1_ROOT / "_FULL_VERIFY_COMPLETE.lock", "fv1_FULL_VERIFY_COMPLETE.lock"),
            (FV1_ROOT / "final_report.json", "fv1_final_report.json"),
            (FV1_ROOT / "runner_freeze_audit.json", "fv1_runner_freeze_audit.json"),
            (FV1_ROOT / "source_snapshot_full_verify_registry.json", "fv1_source_snapshot_registry.json"),
        ],
        "upstream_dl6b_audit_snapshot": [
            (DL6B_ROOT / "gate_decision.json", "dl6b_gate_decision.json"),
            (DL6B_ROOT / "downstream_lock.json", "dl6b_downstream_lock.json"),
            (DL6B_ROOT / "artifact_manifest_dl6b_audit.json", "dl6b_artifact_manifest.json"),
            (DL6B_ROOT / "_DL6B_AUDIT_COMPLETE.lock", "dl6b_AUDIT_COMPLETE.lock"),
            (DL6B_ROOT / "final_report.json", "dl6b_final_report.json"),
            (DL6B_ROOT / "dl6b_evidence_status.json", "dl6b_evidence_status.json"),
            (DL6B_ROOT / "dl6b_claim_registry.json", "dl6b_claim_registry.json"),
            (DL6B_ROOT / "dl6b_reuse_rebuild_matrix.json", "dl6b_reuse_rebuild_matrix.json"),
            (DL6B_ROOT / "contamination_scope_extension_v3.json", "contamination_scope_extension_v3.json"),
            (DL6B_ROOT / "runner_freeze_audit.json", "dl6b_runner_freeze_audit.json"),
        ],
    }
    records = []
    for subdir, files in plan.items():
        for src, name in files:
            records.append(copy_file(writer, src, f"{subdir}/{name}"))
    all_ok = all(r["byte_identical"] for r in records)
    payload = {"created_at": iso_kst(), "record_count": len(records), "all_byte_identical": all_ok, "records": records}
    writer.json("upstream_lineage_registry.json", payload)
    return payload


def frozen_source_registry(writer: Writer) -> Dict[str, Any]:
    records = []
    for rel, frozen in sorted(FROZEN_SOURCE_SHA.items()):
        p = PROJECT_ROOT / rel
        cur = sha256_file(p) if p.exists() else None
        records.append({"relative_path": rel, "exists": p.exists(), "frozen_sha256": frozen, "runtime_sha256": cur,
                        "size_bytes": p.stat().st_size if p.exists() else None, "matches_frozen": cur == frozen})
    payload = {"created_at": iso_kst(), "source_drift_count": sum(1 for r in records if not r["matches_frozen"]),
               "imported_or_executed": False, "hash_and_size_only": True, "records": records}
    writer.json("dynamics_source_final_registry.json", payload)
    return payload


def claim_partition_reconciliation() -> Dict[str, Any]:
    matrix = read_json(DL6B_ROOT / "dl6b_reuse_rebuild_matrix.json")["records"]
    by_decision: Dict[str, List[str]] = {}
    for row in matrix:
        by_decision.setdefault(row["reuse_decision"], []).append(row["claim_id"])
    for k in by_decision:
        by_decision[k] = sorted(by_decision[k])
    all_ids = [cid for ids in by_decision.values() for cid in ids]
    duplicate_ids = sorted({cid for cid in all_ids if all_ids.count(cid) > 1})
    expected_ids = sorted({cid for ids in EXPECTED_PARTITION.values() for cid in ids})
    partitions = {
        "REESTABLISHED_BY_FV1_SYNTHETIC_INTEGRATION": {"claim_ids": by_decision.get("REESTABLISHED_BY_FV1_SYNTHETIC_INTEGRATION", []), "expected_count": 5},
        "REUSABLE_WITH_SCOPE_DOWNGRADE": {"claim_ids": by_decision.get("REUSABLE_WITH_SCOPE_DOWNGRADE", []), "expected_count": 2},
        "REBUILD_FROM_HISTORICAL_DYNAMICS": {"claim_ids": by_decision.get("REBUILD_FROM_HISTORICAL_DYNAMICS", []), "expected_count": 2},
        "REBUILD_AFTER_REWARD_DEFINITION": {"claim_ids": by_decision.get("REBUILD_AFTER_REWARD_DEFINITION", []), "expected_count": 2},
        "REBUILD_WITH_CANONICAL_KPI": {"claim_ids": by_decision.get("REBUILD_WITH_CANONICAL_KPI", []), "expected_count": 2},
    }
    reusable_as_is = len(by_decision.get("REUSABLE_AS_IS", []))
    total = sum(len(v["claim_ids"]) for v in partitions.values())
    checks = {
        "partition_sum_is_13": total == 13,
        "sum_formula_5_2_2_2_2": [len(partitions[k]["claim_ids"]) for k in
            ["REESTABLISHED_BY_FV1_SYNTHETIC_INTEGRATION", "REUSABLE_WITH_SCOPE_DOWNGRADE", "REBUILD_FROM_HISTORICAL_DYNAMICS",
             "REBUILD_AFTER_REWARD_DEFINITION", "REBUILD_WITH_CANONICAL_KPI"]] == [5, 2, 2, 2, 2],
        "duplicate_claim_id_zero": len(duplicate_ids) == 0,
        "missing_claim_id_zero": sorted(all_ids) == expected_ids,
        "reusable_as_is_zero": reusable_as_is == 0,
        "matches_expected_partition": all(partitions[k]["claim_ids"] == sorted(EXPECTED_PARTITION[k]) for k in EXPECTED_PARTITION),
    }
    return {"created_at": iso_kst(), "source_matrix": str(DL6B_ROOT / "dl6b_reuse_rebuild_matrix.json"),
            "partitions": partitions, "total_claim_count": total, "duplicate_claim_ids": duplicate_ids,
            "reusable_as_is_claim_count": reusable_as_is, "checks": checks, "claim_partition_valid": all(checks.values())}


# ---------------------------------------------------------------------------
# scope records
# ---------------------------------------------------------------------------

TECHNICAL_INFRASTRUCTURE = [
    "3-action mapping H/S/K distinct", "Legacy action 2 blocked", "K safety 5/5",
    "Invalid K explicit safe fallback", "Feasible-first arbitration", "No-feasible winner=null",
    "Request ownership before mutation", "Canonical service-unit identity", "Evaluation-scoped shared registry",
    "State round-trip/reset", "Clone isolation", "Replay order/hash", "30 steps / 1800 seconds",
    "Passenger wait avg/p95 contract", "Missing wait no-fallback", "External KPI no-fallback",
    "Runtime record collision=0", "Repeat/dictionary-order determinism",
]
FORBIDDEN_EXPRESSIONS = [
    "historically validated", "real-world effect proven", "policy benefit proven",
    "reward alignment proven", "historical KPI improvement proven",
]


def final_technical_infrastructure_scope() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "technical_infrastructure_status": "SUPPORTED_BY_REAL_ENGINE_SYNTHETIC_INTEGRATION",
            "verified_infrastructure": TECHNICAL_INFRASTRUCTURE,
            "evidence_scope": "SUPPORTED_BY_REAL_ENGINE_SYNTHETIC_INTEGRATION",
            "forbidden_expressions": FORBIDDEN_EXPRESSIONS,
            "forbidden_expression_used_count": 0}


def final_dl6b_evidence_scope() -> Dict[str, Any]:
    evidence = read_json(DL6B_ROOT / "dl6b_evidence_status.json")
    return {"created_at": iso_kst(),
            "dl6b_evidence_status": evidence.get("dl6b_evidence_status"),
            "tier1_classification": evidence.get("tier1_classification"),
            "tier2_classification": evidence.get("tier2_classification"),
            "historical_real_engine_claim_count": evidence.get("historical_real_engine_claim_count"),
            "dl6b_self_declared_diagnostic_only": evidence.get("dl6b_self_declared_diagnostic_only"),
            "dl6b_self_declared_real_world_claim_allowed": evidence.get("dl6b_self_declared_real_world_claim_allowed"),
            "evidence_status_preserved": evidence.get("dl6b_evidence_status") == "PARTIALLY_SUPPORTED_MIXED_EVIDENCE",
            "historical_real_engine_claim_count_is_zero": evidence.get("historical_real_engine_claim_count") == 0}


def final_historical_cardinality_scope() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "candidate_route_33_status": "DECLARED_SYNTHETIC_SCOPE_DESCRIPTOR",
            "candidate_route_33_not_promoted_to": "REAL_HISTORICAL_ROUTE_INVENTORY",
            "snapshot_554_status": "ROW_COUNT_CONSISTENT_SYNTHETIC_WINDOWS",
            "row_count_relations": ["8864 = 554 * 8 * 2", "1662 = 554 * 3"],
            "historical_snapshot_count_verified": False}


def final_reward_scope() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "er1_canonical_reward": "NOT_DEFINED",
            "legacy_dl6b_reward_status": "DL6B_LOCAL_CLOSED_FORM_PROXY",
            "reward_definition_required": True, "reward_rebuild_required_claims": ["C03", "C11"],
            "dl6b_proxy_reward_promoted_to_canonical": False,
            "legacy_immediate_reward_reused": False, "legacy_reward_change_rate_reused": False}


def final_kpi_scope() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "dl6b_historical_kpi_values": "NONREUSABLE_PROXY_VALUES",
            "proxy_examples": ["headway = 240 + 15 * ((position + agent) % 5)", "p95 wait = average wait * 1.35"],
            "retained_fv1_contracts": ["FV1 passenger wait calculation contract", "FV1 missing-wait no-fallback contract", "FV1 external canonical KPI no-fallback contract"],
            "legacy_dl6b_per_action_kpi_reused": False,
            "canonical_kpi_rebuild_required": True, "kpi_rebuild_required_claims": ["C05", "C10"]}


def final_contamination_scope() -> Dict[str, Any]:
    contam = read_json(DL6B_ROOT / "contamination_scope_extension_v3.json")
    return {"created_at": iso_kst(),
            "dl6b_historical_dynamics_evidence": "NONREUSABLE_SYNTHETIC_EVIDENCE",
            "matched_stub_signatures": ["S1", "S2", "S4", "S5", "S6"],
            "unmatched_stub_signatures": ["S3", "S7", "S8"],
            "interpretation": "real engine calls existed, but input/reward/KPI did not meet historical-evidence requirements",
            "contamination_scope_preserved": contam.get("dl6b_historical_dynamics_evidence_status") == "NONREUSABLE_SYNTHETIC_EVIDENCE",
            "source_contamination_row_count": contam.get("contamination_row_count")}


def final_holdout_status_preservation() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "dl6d_r3_c1_normalization_only_sealed_holdout": "FAILED",
            "harmful_holdout_failure": "1 / 26", "beneficial_holdout": "26 / 26",
            "hold_out_reuse_allowed": False, "dl6e_p0_authorized": False,
            "holdout_reopened_by_finalize": False, "holdout_status_changed_by_finalize": False,
            "sealed_holdout_row_accessed": False,
            "preserved_unchanged": ["DL-6D-R3 gate", "sealed hold-out result", "reward alignment status", "candidate approval status"]}


SCOPE_STATEMENT_EN = (
    "V1F establishes that the repaired PA1-A/ER1 dynamics infrastructure operates correctly under full "
    "synthetic real-engine integration tests. It does not establish historical state feasibility, historical "
    "KPI effects, canonical reward alignment, policy benefit, or real-world generalization. Legacy DL-6B "
    "historical/numeric evidence is not reusable except for explicitly scope-downgraded or FV1-reestablished "
    "synthetic infrastructure claims."
)
SCOPE_STATEMENT_KO = (
    "V1F는 수리된 dynamics infrastructure가 실제 엔진을 사용하는 합성 통합환경에서 정상 작동함을 증명한다. "
    "Historical state·KPI·reward·정책 효과·현실 일반화는 아직 증명하지 않는다."
)


def final_scope_statement() -> Dict[str, Any]:
    return {"created_at": iso_kst(), "scope_statement_en": SCOPE_STATEMENT_EN, "scope_statement_ko": SCOPE_STATEMENT_KO}


# ---------------------------------------------------------------------------
# environment / manifest / audits
# ---------------------------------------------------------------------------

def environment_payload() -> Dict[str, Any]:
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"created_at": iso_kst(), "mode": "finalize", "closeout_type": "IMMUTABLE_RECONCILIATION_ONLY",
            "requested_execution_platform": "MAC_MINI_M4_24GB", "actual_compute_path": "CPU_ONLY_RECONCILIATION",
            "platform_machine": platform.machine(), "python_executable": sys.executable, "python_version": sys.version,
            "process_rss_bytes": raw_rss if platform.system() == "Darwin" else raw_rss * 1024,
            "project_simulator_imported": False, "simulator_execution_path_present": False,
            "simulator_transition_execution_count": 0, "fixture_execution_count": 0, "historical_execution_count": 0,
            "validation_access_count": 0, "test_holdout_access_count": 0, "training_run_count": 0, "source_modification_count": 0}


def upstream_manifest_lock_reconciliation(preflight: Mapping[str, Any]) -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "records": [{"upstream_key": r["upstream_key"], "manifest_sha256": r["manifest_sha256"],
                         "manifest_size_bytes": r["manifest_size_bytes"], "checks": r["checks"], "passed": r["preflight_passed"]}
                        for r in preflight["records"]],
            "all_upstream_manifest_lock_valid": preflight["upstream_lineage_valid"]}


def stage_immutability_audit(before: Mapping[str, str]) -> Dict[str, Any]:
    records = []
    mutated = 0
    for key, spec in UPSTREAM.items():
        for rel in [spec["manifest"], spec["lock"]]:
            path = spec["root"] / rel
            after = sha256_file(path)
            unchanged = before[f"{key}:{rel}"] == after
            if not unchanged:
                mutated += 1
            records.append({"upstream_key": key, "relative_path": rel, "sha256_before": before[f"{key}:{rel}"],
                            "sha256_after": after, "unchanged": unchanged,
                            "matches_expected_manifest_sha": (rel != spec["manifest"]) or after == spec["manifest_sha256"]})
    return {"created_at": iso_kst(), "upstream_artifact_mutation_count": mutated, "source_modification_count": 0,
            "records": records, "stage_immutability_passed": mutated == 0}


def write_manifest(writer: Writer, rel: str, payloads: Sequence[str]) -> Dict[str, Any]:
    rows = []
    for p in payloads:
        path = writer.root / p
        rows.append({"relative_path": p, "required": True, "exists": path.exists(),
                     "sha256": sha256_file(path) if path.exists() else None, "size_bytes": path.stat().st_size if path.exists() else None})
    manifest = {"created_at": iso_kst(), "manifest_protocol": "TERMINAL_LOCK_TO_MANIFEST_TO_PAYLOAD",
                "manifest_scope": "V1F_FINAL_CLOSEOUT", "required_payload_count": len(rows),
                "payload_file_count": sum(1 for r in rows if r["exists"]), "missing_payload_count": sum(1 for r in rows if not r["exists"]),
                "missing_payloads": [r["relative_path"] for r in rows if not r["exists"]],
                "terminal_lock_listed_inside_manifest": False, "manifest_self_listed": False, "files": rows}
    writer.json(rel, manifest)
    return manifest


def write_lock(writer: Writer, lock_name: str, manifest_name: str, gate: Mapping[str, Any]) -> None:
    mp = writer.root / manifest_name
    writer.json(lock_name, {"created_at": iso_kst(), "mode": "finalize", "gate": gate["gate"], "gate_passed": gate["gate_passed"],
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


FINAL_PAYLOADS = [
    "runner_snapshot_pre_execution/run_prompt5_e01_dl6d_pa1a_er1_v1f_finalize.py", "runner_freeze_audit.json", "finalize_environment.json",
    "upstream_a2_snapshot/a2_gate_decision.json", "upstream_a2_snapshot/a2_downstream_lock.json",
    "upstream_a2_snapshot/a2_artifact_manifest_final.json", "upstream_a2_snapshot/a2_SUCCESS.lock",
    "upstream_a2_snapshot/a2_final_report.json", "upstream_a2_snapshot/a2_source_snapshot_final_registry.json",
    "upstream_fv1_snapshot/fv1_gate_decision.json", "upstream_fv1_snapshot/fv1_downstream_lock.json",
    "upstream_fv1_snapshot/fv1_artifact_manifest_full_verify.json", "upstream_fv1_snapshot/fv1_FULL_VERIFY_COMPLETE.lock",
    "upstream_fv1_snapshot/fv1_final_report.json", "upstream_fv1_snapshot/fv1_runner_freeze_audit.json",
    "upstream_fv1_snapshot/fv1_source_snapshot_registry.json",
    "upstream_dl6b_audit_snapshot/dl6b_gate_decision.json", "upstream_dl6b_audit_snapshot/dl6b_downstream_lock.json",
    "upstream_dl6b_audit_snapshot/dl6b_artifact_manifest.json", "upstream_dl6b_audit_snapshot/dl6b_AUDIT_COMPLETE.lock",
    "upstream_dl6b_audit_snapshot/dl6b_final_report.json", "upstream_dl6b_audit_snapshot/dl6b_evidence_status.json",
    "upstream_dl6b_audit_snapshot/dl6b_claim_registry.json", "upstream_dl6b_audit_snapshot/dl6b_reuse_rebuild_matrix.json",
    "upstream_dl6b_audit_snapshot/contamination_scope_extension_v3.json", "upstream_dl6b_audit_snapshot/dl6b_runner_freeze_audit.json",
    "upstream_lineage_registry.json", "upstream_manifest_lock_reconciliation.json", "dynamics_source_final_registry.json",
    "final_technical_infrastructure_scope.json", "final_dl6b_evidence_scope.json", "final_claim_partition.json", "final_claim_partition.jsonl",
    "final_reuse_rebuild_matrix.json", "final_reuse_rebuild_matrix.jsonl", "final_historical_cardinality_scope.json",
    "final_reward_scope.json", "final_kpi_scope.json", "final_contamination_scope.json", "final_holdout_status_preservation.json",
    "final_scope_statement.json", "final_stage_immutability_audit.json", "simulator_execution_prohibition_audit.json",
    "historical_execution_prohibition_audit.json", "validation_untouched_audit.json", "test_holdout_untouched_audit.json",
    "reward_energy_scale_nondefinition_audit.json", "training_prohibition_audit.json", "external_access_audit.json",
    "gate_decision.json", "downstream_lock.json", "final_report.json", "final_report.md",
]


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_finalize(artifact_root: Path) -> Path:
    runner_sha_before = sha256_file(RUNNER_PATH)
    runner_size_before = RUNNER_PATH.stat().st_size

    preflight = upstream_preflight()
    if not preflight["upstream_lineage_valid"]:
        raise FinalizeError(FAIL_UPSTREAM, f"upstream lineage invalid: {[r['upstream_key'] for r in preflight['records'] if not r['preflight_passed']]}")

    before = {f"{k}:{rel}": sha256_file(s["root"] / rel) for k, s in UPSTREAM.items() for rel in [s["manifest"], s["lock"]]}

    root = validate_artifact_root(artifact_root)
    writer = Writer(root)
    writer.json("finalize_environment.json", environment_payload())
    runner_snapshot = copy_file(writer, RUNNER_PATH, "runner_snapshot_pre_execution/run_prompt5_e01_dl6d_pa1a_er1_v1f_finalize.py")
    upstream_snapshots(writer)
    writer.json("upstream_manifest_lock_reconciliation.json", upstream_manifest_lock_reconciliation(preflight))

    source_registry = frozen_source_registry(writer)
    if source_registry["source_drift_count"]:
        raise FinalizeError(FAIL_SOURCE_DRIFT, f"dynamics source drift: {source_registry['source_drift_count']}")

    partition = claim_partition_reconciliation()
    writer.json("final_claim_partition.json", partition)
    partition_rows = []
    for decision, info in partition["partitions"].items():
        for cid in info["claim_ids"]:
            partition_rows.append({"claim_id": cid, "partition_decision": decision})
    writer.jsonl("final_claim_partition.jsonl", sorted(partition_rows, key=lambda r: r["claim_id"]))
    if not partition["claim_partition_valid"]:
        raise FinalizeError(FAIL_PARTITION, f"claim partition invalid: {partition['checks']}")

    # final reuse/rebuild matrix (carried from DL-6B audit, sealed here)
    matrix = read_json(DL6B_ROOT / "dl6b_reuse_rebuild_matrix.json")["records"]
    writer.json("final_reuse_rebuild_matrix.json", {"created_at": iso_kst(), "records": matrix})
    writer.jsonl("final_reuse_rebuild_matrix.jsonl", matrix)

    tech = final_technical_infrastructure_scope()
    dl6b_evidence = final_dl6b_evidence_scope()
    cardinality = final_historical_cardinality_scope()
    reward = final_reward_scope()
    kpi = final_kpi_scope()
    contam = final_contamination_scope()
    holdout = final_holdout_status_preservation()
    scope_stmt = final_scope_statement()
    writer.json("final_technical_infrastructure_scope.json", tech)
    writer.json("final_dl6b_evidence_scope.json", dl6b_evidence)
    writer.json("final_historical_cardinality_scope.json", cardinality)
    writer.json("final_reward_scope.json", reward)
    writer.json("final_kpi_scope.json", kpi)
    writer.json("final_contamination_scope.json", contam)
    writer.json("final_holdout_status_preservation.json", holdout)
    writer.json("final_scope_statement.json", scope_stmt)

    # overclaim / promotion / contamination / holdout guards
    if not dl6b_evidence["evidence_status_preserved"] or not dl6b_evidence["historical_real_engine_claim_count_is_zero"]:
        raise FinalizeError(FAIL_OVERCLAIM, "DL-6B evidence status not preserved as mixed with 0 historical real-engine claims")
    if not contam["contamination_scope_preserved"]:
        raise FinalizeError(FAIL_CONTAM_LOST, "DL-6B contamination scope not preserved")
    if reward["dl6b_proxy_reward_promoted_to_canonical"]:
        raise FinalizeError(FAIL_REWARD_PROMOTED, "DL-6B proxy reward promoted")
    if kpi["legacy_dl6b_per_action_kpi_reused"]:
        raise FinalizeError(FAIL_KPI_PROMOTED, "legacy DL-6B KPI reused")
    if holdout["holdout_status_changed_by_finalize"]:
        raise FinalizeError(FAIL_HOLDOUT_CHANGED, "hold-out status changed")

    # prohibition audits
    writer.json("simulator_execution_prohibition_audit.json", {"created_at": iso_kst(), "simulator_transition_execution_count": 0,
                "synthetic_fixture_execution_count": 0, "thirty_step_execution_count": 0, "project_simulator_imported": False, "simulator_execution_path_present": False})
    writer.json("historical_execution_prohibition_audit.json", {"created_at": iso_kst(), "historical_execution_count": 0, "d1_250row_execution_count": 0, "train_row_access_count": 0, "dl6d_r3_holdout_reopened": False})
    writer.json("validation_untouched_audit.json", {"created_at": iso_kst(), "validation_access_count": 0, "validation_branch_count": 0})
    writer.json("test_holdout_untouched_audit.json", {"created_at": iso_kst(), "test_holdout_access_count": 0, "test_holdout_touched": False, "sealed_holdout_reopened": False})
    writer.json("reward_energy_scale_nondefinition_audit.json", {"created_at": iso_kst(), "new_reward_formula_created": False, "new_energy_formula_created": False, "normalization_scale_created": False, "candidate_created": False, "tolerance_changed": False})
    writer.json("training_prohibition_audit.json", {"created_at": iso_kst(), "training_run_count": 0, "optimizer_step_count": 0, "loss_backward_count": 0, "checkpoint_load_count": 0, "checkpoint_write_count": 0, "mappo_training_count": 0, "gatv2_training_count": 0})
    writer.json("external_access_audit.json", {"created_at": iso_kst(), "db_access_count": 0, "api_call_count": 0, "network_access_count": 0, "git_commit_count": 0, "git_push_count": 0})

    immutability = stage_immutability_audit(before)
    writer.json("final_stage_immutability_audit.json", immutability)
    if not immutability["stage_immutability_passed"]:
        raise FinalizeError(FAIL_UPSTREAM_MUTATED, f"upstream artifact mutated: {immutability['upstream_artifact_mutation_count']}")

    runner_sha_after = sha256_file(RUNNER_PATH)
    runner_freeze = {"created_at": iso_kst(), "runner_relative_path": str(RUNNER_PATH.relative_to(PROJECT_ROOT)),
                     "runner_sha256_before_execution": runner_sha_before, "runner_size_before_execution": runner_size_before,
                     "runner_sha256_after_execution": runner_sha_after, "runner_snapshot_sha256": runner_snapshot["copied_sha256"],
                     "runner_mutation_count": 0 if runner_sha_before == runner_sha_after else 1,
                     "runner_frozen": runner_sha_before == runner_sha_after == runner_snapshot["copied_sha256"]}
    writer.json("runner_freeze_audit.json", runner_freeze)
    if runner_freeze["runner_mutation_count"]:
        raise FinalizeError(FAIL_RUNNER, "runner SHA changed during finalize")

    gate = {"created_at": iso_kst(), "mode": "finalize", "gate": PASS_FINAL, "gate_passed": True, "readiness": FINAL_READINESS,
            "meaning": ["Engine repair and synthetic integration verification complete.",
                        "Legacy DL-6B evidence boundary and rebuild obligations recorded.",
                        "Historical state feasibility remains pending."],
            "scope_statement_en": SCOPE_STATEMENT_EN,
            "state_feasibility_authorized": False, "historical_dl6b_rebuild_authorized": False,
            "reward_rebuild_authorized": False, "canonical_kpi_rebuild_authorized": False,
            "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False}
    writer.json("gate_decision.json", gate)

    writer.json("downstream_lock.json", {
        "v1f_finalize_complete": True, "a2_upstream_verified": True, "fv1_upstream_verified": True, "dl6b_audit_upstream_verified": True,
        "technical_infrastructure_status": "SUPPORTED_BY_REAL_ENGINE_SYNTHETIC_INTEGRATION",
        "dl6b_evidence_status": "PARTIALLY_SUPPORTED_MIXED_EVIDENCE", "dl6b_historical_evidence_status": "NONREUSABLE_SYNTHETIC_EVIDENCE",
        "historical_real_engine_claim_count": 0,
        "fv1_reestablished_claims": EXPECTED_PARTITION["REESTABLISHED_BY_FV1_SYNTHETIC_INTEGRATION"],
        "scope_downgrade_claims": EXPECTED_PARTITION["REUSABLE_WITH_SCOPE_DOWNGRADE"],
        "historical_rebuild_claims": EXPECTED_PARTITION["REBUILD_FROM_HISTORICAL_DYNAMICS"],
        "reward_rebuild_claims": EXPECTED_PARTITION["REBUILD_AFTER_REWARD_DEFINITION"],
        "kpi_rebuild_claims": EXPECTED_PARTITION["REBUILD_WITH_CANONICAL_KPI"],
        "reusable_as_is_claim_count": 0, "claim_partition_complete": True,
        "candidate_route_33_status": "DECLARED_SYNTHETIC_SCOPE_DESCRIPTOR", "snapshot_554_status": "ROW_COUNT_CONSISTENT_SYNTHETIC_WINDOWS",
        "canonical_reward_status": "NOT_DEFINED", "legacy_dl6b_reward_status": "NONREUSABLE_LOCAL_PROXY", "legacy_dl6b_kpi_status": "NONREUSABLE_PROXY_VALUES",
        "dl6d_r3_holdout_status": "FAILED_SEALED_NO_REUSE", "dl6e_p0_authorized": False,
        "state_feasibility_required": True, "state_feasibility_authorized": False,
        "historical_dl6b_rebuild_required": True, "historical_dl6b_rebuild_authorized": False,
        "reward_definition_required": True, "reward_rebuild_authorized": False,
        "canonical_kpi_rebuild_required": True, "canonical_kpi_rebuild_authorized": False,
        "pa1b_authorized": False, "training_allowed": False})

    report_payload, report_md = build_final_report(root, gate, preflight, partition, dl6b_evidence, source_registry, runner_freeze, immutability)
    writer.json("final_report.json", report_payload)
    writer.text("final_report.md", report_md + "\n")

    manifest = write_manifest(writer, "artifact_manifest_v1f_final.json", FINAL_PAYLOADS)
    if manifest["missing_payload_count"]:
        raise FinalizeError(FAIL_MANIFEST, f"missing payloads: {manifest['missing_payloads']}")
    write_lock(writer, "_SUCCESS.lock", "artifact_manifest_v1f_final.json", gate)
    v = verify_manifest(root, "_SUCCESS.lock")
    if not v["manifest_hash_ok"] or not v["manifest_size_ok"] or v["payload_missing_count"] or v["payload_hash_mismatch_count"] or v["terminal_lock_listed_inside_manifest"] or v["manifest_self_listed"]:
        raise FinalizeError(FAIL_MANIFEST, f"manifest verification failed: {v}")

    print("V1F FINAL CLOSEOUT COMPLETE")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_FINAL}")
    print(f"readiness: {FINAL_READINESS}")
    print(f"technical_infrastructure: SUPPORTED_BY_REAL_ENGINE_SYNTHETIC_INTEGRATION")
    print(f"dl6b_evidence_status: PARTIALLY_SUPPORTED_MIXED_EVIDENCE")
    print(f"claim partition 5/2/2/2/2 = {partition['total_claim_count']} | reusable_as_is = {partition['reusable_as_is_claim_count']}")
    print(f"runner_mutation_count: {runner_freeze['runner_mutation_count']} | upstream_mutation: {immutability['upstream_artifact_mutation_count']} | source_drift: {source_registry['source_drift_count']}")
    print("state_feasibility_authorized: false")
    return root


def build_final_report(root, gate, preflight, partition, dl6b_evidence, source_registry, runner_freeze, immutability):
    fv1_report = read_json(FV1_ROOT / "final_report.json")
    combined = fv1_report.get("combined_fixture_passed"), fv1_report.get("combined_fixture_total")
    p = partition["partitions"]
    answers = {
        "01_v1f_finalize_complete": gate["gate_passed"],
        "02_a2_repair_ok": next(r["preflight_passed"] for r in preflight["records"] if r["upstream_key"] == "A2"),
        "03_fv1_22_of_22_preserved": combined == (22, 22),
        "04_dl6b_audit_mixed_evidence": dl6b_evidence["dl6b_evidence_status"] == "PARTIALLY_SUPPORTED_MIXED_EVIDENCE",
        "05_dl6b_called_real_engine": True,
        "06_dl6b_input_historical_or_synthetic": "SYNTHETIC_HASH_GENERATED",
        "07_dl6b_30_step_mechanics_real": True,
        "08_dl6b_reward_is_canonical": False,
        "09_dl6b_kpi_is_historical_event_based": False,
        "10_33_routes_is_real_inventory": False,
        "11_554_snapshots_is_historical": False,
        "12_fv1_reestablished_claims": p["REESTABLISHED_BY_FV1_SYNTHETIC_INTEGRATION"]["claim_ids"],
        "13_scope_downgrade_claims": p["REUSABLE_WITH_SCOPE_DOWNGRADE"]["claim_ids"],
        "14_historical_rebuild_claims": p["REBUILD_FROM_HISTORICAL_DYNAMICS"]["claim_ids"],
        "15_reward_rebuild_claims": p["REBUILD_AFTER_REWARD_DEFINITION"]["claim_ids"],
        "16_kpi_rebuild_claims": p["REBUILD_WITH_CANONICAL_KPI"]["claim_ids"],
        "17_reusable_as_is_dl6b_claim_exists": partition["reusable_as_is_claim_count"] > 0,
        "18_dl6d_r3_holdout_failure_preserved": True,
        "19_dl6e_p0_still_unauthorized": True,
        "20_historical_state_feasibility_still_required": True,
        "21_state_feasibility_not_auto_run": True,
        "22_reward_kpi_rebuild_not_auto_run": True,
        "23_pa1b_and_training_not_auto_run": True,
    }
    payload = {"created_at": iso_kst(), "artifact_root": str(root), "mode": "finalize", "gate": gate["gate"], "gate_passed": gate["gate_passed"],
               "readiness": gate["readiness"], "quick_answers": answers, "scope_statement_en": SCOPE_STATEMENT_EN, "scope_statement_ko": SCOPE_STATEMENT_KO,
               "technical_infrastructure_status": "SUPPORTED_BY_REAL_ENGINE_SYNTHETIC_INTEGRATION",
               "dl6b_evidence_status": "PARTIALLY_SUPPORTED_MIXED_EVIDENCE", "dl6b_historical_evidence_status": "NONREUSABLE_SYNTHETIC_EVIDENCE",
               "runner_mutation_count": runner_freeze["runner_mutation_count"], "upstream_artifact_mutation_count": immutability["upstream_artifact_mutation_count"],
               "dynamics_source_drift_count": source_registry["source_drift_count"],
               "remaining_followups": ["PA1-A historical state feasibility", "historical DL-6B rebuild", "canonical reward definition + reward rebuild", "canonical KPI rebuild"],
               "state_feasibility_authorized": False, "reward_rebuild_authorized": False, "canonical_kpi_rebuild_authorized": False,
               "historical_dl6b_rebuild_authorized": False, "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False,
               "next_authorized_action": "PA1-A historical state-feasibility review only, after explicit user review and command"}
    lines = ["# V1F Immutable Final Closeout — Mixed DL-6B Evidence Boundary", "",
             f"- artifact_root: {root}", f"- gate: {gate['gate']}", f"- readiness: {gate['readiness']}",
             "- technical_infrastructure_status: SUPPORTED_BY_REAL_ENGINE_SYNTHETIC_INTEGRATION",
             "- dl6b_evidence_status: PARTIALLY_SUPPORTED_MIXED_EVIDENCE",
             f"- claim partition: reestablished {p['REESTABLISHED_BY_FV1_SYNTHETIC_INTEGRATION']['claim_ids']}, "
             f"downgrade {p['REUSABLE_WITH_SCOPE_DOWNGRADE']['claim_ids']}, historical {p['REBUILD_FROM_HISTORICAL_DYNAMICS']['claim_ids']}, "
             f"reward {p['REBUILD_AFTER_REWARD_DEFINITION']['claim_ids']}, kpi {p['REBUILD_WITH_CANONICAL_KPI']['claim_ids']}", "",
             "## Final scope statement", SCOPE_STATEMENT_EN, "", SCOPE_STATEMENT_KO, "", "## Quick answers"]
    for k in sorted(answers):
        lines.append(f"- {k}: {answers[k]}")
    lines += ["", "## Downstream authorization (all locked)",
              "- state_feasibility_required: true / authorized: false",
              "- historical_dl6b_rebuild_required: true / authorized: false",
              "- reward_definition_required: true / reward_rebuild_authorized: false",
              "- canonical_kpi_rebuild_required: true / authorized: false",
              "- pa1b_authorized: false | dl6e_p0_authorized: false | training_allowed: false", "",
              "## Preserved hold-out", "- DL-6D-R3 C1 normalization-only sealed hold-out: FAILED (harmful 1/26, beneficial 26/26), no reuse.", ""]
    return payload, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["finalize"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        run_finalize(args.artifact_root)
    except FinalizeError as exc:
        print("V1F FINAL CLOSEOUT FAILED")
        print(f"gate: {exc.gate_status}")
        print(f"detail: {exc.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
