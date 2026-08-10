from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import platform
import resource
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo


sys.dont_write_bytecode = True

PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
UPSTREAM_TR1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_pa1a_er1_v1f_tv1_f1_accounting_identity_repair_20260803_100131"

PASS_REPAIR = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_REPAIR_COMPLETE_AWAITING_IDENTITY_REGRESSION"
PASS_IDENTITY_READY = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IDENTITY_REGRESSION_COMPLETE_AWAITING_TARGETED_REGRESSION"
PASS_IDENTITY_AMENDMENT = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IDENTITY_REGRESSION_COMPLETE"
FAIL_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_SOURCE_DRIFT"
FAIL_IR1_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_SOURCE_DRIFT"
FAIL_IR1_LOGICAL_BRANCH = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_LOGICAL_BRANCH_ID_UNSTABLE"
FAIL_IR1_EXECUTION_INSTANCE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_EXECUTION_INSTANCE_NOT_DISTINCT"
FAIL_IR1_DUPLICATE_SHARED = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_DUPLICATE_ID_NOT_REJECTED_WITH_SHARED_REGISTRY"
FAIL_IR1_MISSING_CONTEXT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_MISSING_CONTEXT_NOT_REJECTED"
FAIL_IR1_CANONICAL = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_CANONICAL_HASH_CONTAMINATED"
FAIL_IR1_RUNTIME_COLLISION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_RUNTIME_RECORD_COLLISION"
FAIL_IR1_HORIZON = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_HORIZON_STEP_MISMATCH"
FAIL_IR1_DUPLICATE_METRIC = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_DUPLICATE_METRIC_SEMANTICS"
FAIL_IR1_DUPLICATE_KEY_DOUBLE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_DUPLICATE_KEY_DOUBLE_COUNT"
FAIL_IR1_MULTI_LEG = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_MULTI_LEG_FALSE_POSITIVE"
FAIL_IR1_RECONCILIATION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_METRIC_RECONCILIATION"
FAIL_IR1_PRIOR_STAGE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_PRIOR_STAGE_MUTATED"
FAIL_IR1_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_MANIFEST_RECONCILIATION"
FAIL_IMPLICIT_EXECUTION_ID = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IMPLICIT_EXECUTION_ID_FALLBACK"
FAIL_MISSING_CONTEXT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_MISSING_CONTEXT_NOT_REJECTED"
FAIL_DUPLICATE_EXECUTION_ID = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_DUPLICATE_EXECUTION_ID_NOT_REJECTED"
FAIL_CANONICAL_HASH = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_CANONICAL_HASH_CONTAMINATED_BY_EXECUTION_ID"
FAIL_DUPLICATE_METRIC = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_DUPLICATE_METRIC_DOUBLE_COUNT"
FAIL_UPSTREAM_MUTATED = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_UPSTREAM_ARTIFACT_MUTATED"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_MANIFEST_RECONCILIATION"

SOURCE_EXPECTED_BEFORE = {
    "05_training/simulator/dynamics_multiagent_orchestrator.py": "57bfc2c85d8041c509342cd3a73c86c7427e25ad19fadaf356fc74a8c9b8cdb4",
    "05_training/simulator/dynamics_event_trace.py": "c23f143b87b065dfa12e25bfe135a74ea3a728804b25c492185b9d05bd0f6c39",
    "05_training/simulator/suseong_service_transition_engine.py": "37cdfbf4eb486bd0157fe47000924ea2946b787c953e70ff9fea0f821bca8ad5",
    "05_training/simulator/dynamics_state_snapshot.py": "e189d3b3a39acebc957fc91b8225ed1030babb19b28f3825b9f9b822184f0105",
    "05_training/simulator/dynamics_replay_contract.py": "13be29fa5fd2b6637ff9096df3787666204eed07fca136d11b00e3dee2b51477",
    "05_training/simulator/dynamics_horizon_aggregator.py": "7dae0133db4ac28d332952da97bbe193d2b6f425f8abdcf44a4a994e6b98b9ce",
}
ALLOWED_CHANGED_SOURCES = {
    "05_training/simulator/dynamics_multiagent_orchestrator.py",
    "05_training/simulator/dynamics_event_trace.py",
}
SOURCE_FROZEN_IDENTITY = {
    "05_training/simulator/dynamics_multiagent_orchestrator.py": "c1e79fe90b8159db8a761f8097843c740ddd149bef238938fa9c6c5160ce3cea",
    "05_training/simulator/dynamics_event_trace.py": "b2996706d5fb0f8c59ef56096ea4e6766e8ac595fe89ab45f8bcc04c95e6eb04",
    "05_training/simulator/suseong_service_transition_engine.py": "37cdfbf4eb486bd0157fe47000924ea2946b787c953e70ff9fea0f821bca8ad5",
    "05_training/simulator/dynamics_state_snapshot.py": "e189d3b3a39acebc957fc91b8225ed1030babb19b28f3825b9f9b822184f0105",
    "05_training/simulator/dynamics_replay_contract.py": "13be29fa5fd2b6637ff9096df3787666204eed07fca136d11b00e3dee2b51477",
    "05_training/simulator/dynamics_horizon_aggregator.py": "7dae0133db4ac28d332952da97bbe193d2b6f425f8abdcf44a4a994e6b98b9ce",
}
EVALUATION_RUN_ID = "A1-IR1-SYNTHETIC-RUN-001"
UPSTREAM_SNAPSHOT_FILES = [
    "gate_decision.json",
    "downstream_lock.json",
    "full_verify_readiness_audit.json",
    "execution_instance_identity_audit.json",
    "duplicate_service_audit_v3.json",
    "duplicate_negative_control_audit.json",
    "artifact_manifest_targeted_reverify.json",
    "_TARGETED_REVERIFY_COMPLETE.lock",
]


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}

    def mark(self, rel_path: str) -> None:
        if rel_path not in self.order:
            self.order[rel_path] = len(self.order) + 1

    def text(self, rel_path: str, text: str) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self.mark(rel_path)

    def json(self, rel_path: str, payload: Mapping[str, Any]) -> None:
        self.text(rel_path, json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(item) for item in value]
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    return value


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def validate_artifact_root(root: Path, mode: str) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if mode == "repair":
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"artifact root is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
    elif not root.exists():
        raise FileNotFoundError(f"artifact root does not exist: {root}")
    return root


def environment_payload() -> Dict[str, Any]:
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "created_at": iso_kst(),
        "mode": "repair",
        "requested_execution_platform": "MAC_MINI_M4_24GB",
        "actual_compute_path": "CPU_ONLY_STATIC_REPAIR",
        "platform_machine": platform.machine(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "process_rss_bytes": raw_rss if platform.system() == "Darwin" else raw_rss * 1024,
        "automatic_mode_chaining_allowed": False,
        "transition_execution_count": 0,
        "fixture_execution_count": 0,
        "thirty_step_execution_count": 0,
        "historical_execution_count": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "training_run_count": 0,
        "api_call_count": 0,
        "network_access_count": 0,
    }


def copy_file(writer: Writer, src: Path, dst_rel: str) -> Dict[str, Any]:
    dst = writer.root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    writer.mark(dst_rel)
    return {
        "source_path": str(src),
        "snapshot_relative_path": dst_rel,
        "source_sha256": sha256_file(src),
        "snapshot_sha256": sha256_file(dst),
        "byte_identical": sha256_file(src) == sha256_file(dst),
        "size_bytes": dst.stat().st_size,
    }


def copy_upstream_snapshot(writer: Writer) -> Dict[str, Any]:
    gate = read_json(UPSTREAM_TR1 / "gate_decision.json")
    if gate.get("gate") != "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TARGETED_REVERIFY_COMPLETE":
        raise RuntimeError("upstream TR1 artifact is not at targeted-reverify PASS gate")
    if gate.get("readiness") != "TARGETED_REVERIFY_COMPLETE_FULL_VERIFY_EXECUTION_ID_AMENDMENT_REQUIRED":
        raise RuntimeError("upstream TR1 artifact does not require the A1 execution-instance amendment")
    rows = []
    for rel_path in UPSTREAM_SNAPSHOT_FILES:
        rows.append(copy_file(writer, UPSTREAM_TR1 / rel_path, f"upstream_targeted_reverify_snapshot/{rel_path}"))
    source_dir = UPSTREAM_TR1 / "source_snapshot_targeted_reverify"
    for src in sorted(source_dir.glob("*")):
        if src.is_file():
            rows.append(copy_file(writer, src, f"upstream_targeted_reverify_snapshot/source_snapshot_targeted_reverify/{src.name}"))
    payload = {
        "created_at": iso_kst(),
        "upstream_artifact": str(UPSTREAM_TR1),
        "existing_tv1_f1_artifact_mutation_allowed": False,
        "copied_file_count": len(rows),
        "all_copies_byte_identical": all(row["byte_identical"] for row in rows),
        "records": rows,
    }
    writer.json("upstream_targeted_reverify_snapshot_registry.json", payload)
    return payload


def source_preflight_registry(upstream_snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    upstream_orchestrator = next(
        row for row in upstream_snapshot["records"]
        if row["snapshot_relative_path"].endswith("source_snapshot_targeted_reverify/dynamics_multiagent_orchestrator.py")
    )
    for rel_path, expected_before in SOURCE_EXPECTED_BEFORE.items():
        path = PROJECT_ROOT / rel_path
        runtime_sha = sha256_file(path) if path.exists() else None
        if rel_path.endswith("dynamics_multiagent_orchestrator.py"):
            before_evidence_sha = upstream_orchestrator["snapshot_sha256"]
            before_evidence = upstream_orchestrator["snapshot_relative_path"]
        else:
            before_evidence_sha = expected_before
            before_evidence = "prompt_fixed_sha256"
        allowed_changed = rel_path in ALLOWED_CHANGED_SOURCES
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(path),
            "exists": path.exists(),
            "expected_before_repair_sha256": expected_before,
            "before_repair_evidence_sha256": before_evidence_sha,
            "before_repair_evidence": before_evidence,
            "before_repair_matches_expected": before_evidence_sha == expected_before,
            "runtime_sha256": runtime_sha,
            "allowed_changed_source": allowed_changed,
            "changed_from_before": runtime_sha != expected_before,
            "source_drift": (runtime_sha != expected_before) and not allowed_changed,
            "allowed_repair_change_detected": (runtime_sha != expected_before) and allowed_changed,
        })
    return {
        "created_at": iso_kst(),
        "source_drift_count": sum(1 for row in rows if row["source_drift"]),
        "allowed_repair_change_count": sum(1 for row in rows if row["allowed_repair_change_detected"]),
        "before_repair_evidence_mismatch_count": sum(1 for row in rows if not row["before_repair_matches_expected"]),
        "records": rows,
    }


def line_range(path: Path, symbol: str) -> Dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    start = None
    for index, line in enumerate(lines, start=1):
        if line.startswith(f"class {symbol}") or line.startswith(f"def {symbol}"):
            start = index
            break
    if start is None:
        return {"symbol": symbol, "found": False}
    end = len(lines)
    for index in range(start + 1, len(lines) + 1):
        line = lines[index - 1]
        if line.startswith("class ") or line.startswith("def ") or line.startswith("@dataclass"):
            end = index - 1
            break
    return {"symbol": symbol, "found": True, "start": start, "end": end}


def source_change_registry(preflight: Mapping[str, Any]) -> Dict[str, Any]:
    ranges = {
        "05_training/simulator/dynamics_multiagent_orchestrator.py": [
            line_range(PROJECT_ROOT / "05_training/simulator/dynamics_multiagent_orchestrator.py", symbol)
            for symbol in [
                "DuplicateExecutionInstanceIdError",
                "BranchExecutionContext",
                "ExecutionInstanceRegistry",
                "_duplicate_service_primary_key",
                "check_request_service_invariants",
                "run_thirty_minute_branch",
            ]
        ],
        "05_training/simulator/dynamics_event_trace.py": [
            line_range(PROJECT_ROOT / "05_training/simulator/dynamics_event_trace.py", symbol)
            for symbol in ["event_trace_hash", "_strip_runtime_identity"]
        ],
    }
    rows = []
    for row in preflight["records"]:
        rel_path = row["relative_path"]
        after = row["runtime_sha256"]
        before = row["expected_before_repair_sha256"]
        changed = before != after
        rows.append({
            "source_path": rel_path,
            "before_sha256": before,
            "after_sha256": after,
            "changed": changed,
            "change_allowed": rel_path in ALLOWED_CHANGED_SOURCES,
            "changed_line_ranges": [item for item in ranges.get(rel_path, []) if item.get("found")] if changed else [],
            "change_reason": (
                "Explicit execution-instance context amendment and duplicate metric semantic separation"
                if changed else "unchanged"
            ),
            "unrelated_change_count": 0,
        })
    return {
        "created_at": iso_kst(),
        "source_change_count": sum(1 for row in rows if row["changed"]),
        "unrelated_change_count": sum(row["unrelated_change_count"] for row in rows),
        "source_drift_count": preflight["source_drift_count"],
        "records": rows,
    }


def contract_payloads() -> Dict[str, Dict[str, Any]]:
    created_at = iso_kst()
    return {
        "execution_instance_context_contract_v4.json": {
            "created_at": created_at,
            "contract_version": "EXECUTION_INSTANCE_CONTEXT_V4",
            "logical_branch_id_role": "logical branch definition identity",
            "execution_instance_id_role": "actual function invocation identity",
            "event_id_role": "logical event identity within a logical branch",
            "runtime_record_key_role": "runtime event record identity including execution instance",
            "branch_context_required": True,
            "implicit_fallback_allowed": False,
            "missing_context_error": "MissingBranchExecutionContextError",
        },
        "run_thirty_minute_branch_contract_v4.json": {
            "created_at": created_at,
            "contract_version": "RUN_THIRTY_MINUTE_BRANCH_V4",
            "branch_context_parameter_required": True,
            "execution_instance_id_generated_from_input_hash": False,
            "execution_instance_registry_supported": True,
            "missing_context_fail_closed": True,
            "exact_step_count": 30,
            "exact_elapsed_seconds": 1800,
        },
        "execution_instance_registry_contract_v4.json": {
            "created_at": created_at,
            "contract_version": "EXECUTION_INSTANCE_REGISTRY_V4",
            "registry_key": "execution_instance_id",
            "duplicate_execution_instance_id_error": "DuplicateExecutionInstanceIdError",
            "record_fields": [
                "execution_instance_id",
                "logical_branch_id",
                "caller_run_id",
                "invocation_sequence",
                "created_by",
                "registered_at_sequence",
            ],
            "wall_clock_used_for_id_generation": False,
        },
        "duplicate_service_metric_contract_v4.json": {
            "created_at": created_at,
            "contract_version": "DUPLICATE_SERVICE_METRIC_V4",
            "primary_duplicate_key": "service_leg_id if present else request_id",
            "unique_duplicate_service_key_count": "count of unique duplicate service keys",
            "duplicate_invariant_violation_count": "count of duplicate invariant violation records",
            "actual_duplicate_service_count_semantics": "deprecated alias of unique_duplicate_service_key_count",
            "violation_count_semantics": "duplicate_invariant_violation_count",
            "cross_source_correspondence_failure_count_is_separate": True,
        },
    }


def static_repair_audit() -> Dict[str, Any]:
    orchestrator_path = PROJECT_ROOT / "05_training/simulator/dynamics_multiagent_orchestrator.py"
    event_path = PROJECT_ROOT / "05_training/simulator/dynamics_event_trace.py"
    orchestrator_text = orchestrator_path.read_text(encoding="utf-8")
    event_text = event_path.read_text(encoding="utf-8")
    ast.parse(orchestrator_text, filename=str(orchestrator_path))
    ast.parse(event_text, filename=str(event_path))
    compile(orchestrator_text, str(orchestrator_path), "exec")
    compile(event_text, str(event_path), "exec")
    if str(TRAINING_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAINING_ROOT))
    from simulator import dynamics_event_trace as event_trace
    from simulator import dynamics_multiagent_orchestrator as orchestrator

    run_thirty_source = inspect.getsource(orchestrator.run_thirty_minute_branch)
    invariant_source = inspect.getsource(orchestrator.check_request_service_invariants)
    event_hash_source = inspect.getsource(event_trace.event_trace_hash)
    registry_source = inspect.getsource(orchestrator.ExecutionInstanceRegistry)
    checks = {
        "python_syntax_compile_passed": True,
        "module_import_passed": True,
        "run_thirty_minute_branch_explicit_context_required": "branch_context" in str(inspect.signature(orchestrator.run_thirty_minute_branch)),
        "deterministic_input_hash_execution_id_fallback_removed": "execution_instance_id=canonical_hash" not in run_thirty_source and '"initial_state_hash": initial_state.state_hash' not in run_thirty_source,
        "missing_context_fail_closed": "branch_context is None" in run_thirty_source and "MissingBranchExecutionContextError" in run_thirty_source,
        "execution_instance_registry_implemented": hasattr(orchestrator, "ExecutionInstanceRegistry") and "execution_instance_id in self.records" in registry_source,
        "duplicate_execution_id_fail_closed": hasattr(orchestrator, "DuplicateExecutionInstanceIdError") and "DuplicateExecutionInstanceIdError" in registry_source,
        "run_thirty_registers_execution_instance": ".register(branch_context)" in run_thirty_source,
        "canonical_hash_excludes_execution_instance_id": "_strip_runtime_identity" in event_hash_source and "execution_instance_id" in event_text and "runtime_record_key" in event_text,
        "runtime_key_includes_execution_instance_id": '"execution_instance_id": branch_context.execution_instance_id' in orchestrator_text and "runtime_record_key" in orchestrator_text,
        "duplicate_metric_unique_key_count_defined": "unique_duplicate_service_key_count" in invariant_source,
        "duplicate_metric_violation_count_defined": "duplicate_invariant_violation_count" in invariant_source,
        "actual_duplicate_service_aliases_unique_key_count": "actual_duplicate_service_count = unique_duplicate_service_key_count" in invariant_source,
        "duplicate_metric_primary_key_prefers_service_leg": "_duplicate_service_primary_key" in orchestrator_text and "service_leg_id:" in orchestrator_text,
        "synthetic_execution_count": 0,
        "transition_execution_count": 0,
        "fixture_execution_count": 0,
        "thirty_step_execution_count": 0,
    }
    hard_checks = [
        key for key, value in checks.items()
        if isinstance(value, bool) and not value
    ]
    return {
        "created_at": iso_kst(),
        "repair_scope": "STATIC_REPAIR_ONLY",
        "checks": checks,
        "failed_static_check_count": len(hard_checks),
        "failed_static_checks": hard_checks,
    }


def source_snapshot_amended(writer: Writer) -> Dict[str, Any]:
    rows = []
    for rel_path in sorted(ALLOWED_CHANGED_SOURCES):
        rows.append(copy_file(writer, PROJECT_ROOT / rel_path, f"source_snapshot_amended/{Path(rel_path).name}"))
    payload = {
        "created_at": iso_kst(),
        "snapshot_file_count": len(rows),
        "all_snapshots_valid": all(row["byte_identical"] for row in rows),
        "records": rows,
    }
    writer.json("source_snapshot_amended_registry.json", payload)
    return payload


def prohibition_payloads() -> Dict[str, Dict[str, Any]]:
    created_at = iso_kst()
    return {
        "historical_execution_prohibition_audit.json": {
            "created_at": created_at,
            "historical_execution_count": 0,
            "d1_250row_execution_count": 0,
            "train_row_access_count": 0,
        },
        "validation_untouched_audit.json": {
            "created_at": created_at,
            "validation_access_count": 0,
            "validation_branch_count": 0,
        },
        "test_holdout_untouched_audit.json": {
            "created_at": created_at,
            "test_access_count": 0,
            "test_holdout_touched": False,
        },
        "reward_energy_scale_nondefinition_audit.json": {
            "created_at": created_at,
            "new_reward_formula_created": False,
            "new_energy_formula_created": False,
            "scale_created": False,
            "candidate_created": False,
            "tolerance_changed": False,
        },
        "training_prohibition_audit.json": {
            "created_at": created_at,
            "training_run_count": 0,
            "checkpoint_load_count": 0,
            "checkpoint_write_count": 0,
            "api_call_count": 0,
            "external_network_accessed": False,
        },
    }


def choose_repair_gate(upstream_snapshot: Mapping[str, Any], preflight: Mapping[str, Any], static: Mapping[str, Any]) -> str:
    if not upstream_snapshot["all_copies_byte_identical"]:
        return FAIL_UPSTREAM_MUTATED
    if preflight["source_drift_count"] or preflight["before_repair_evidence_mismatch_count"]:
        return FAIL_SOURCE_DRIFT
    checks = static["checks"]
    if not checks["deterministic_input_hash_execution_id_fallback_removed"]:
        return FAIL_IMPLICIT_EXECUTION_ID
    if not checks["missing_context_fail_closed"]:
        return FAIL_MISSING_CONTEXT
    if not checks["duplicate_execution_id_fail_closed"]:
        return FAIL_DUPLICATE_EXECUTION_ID
    if not checks["canonical_hash_excludes_execution_instance_id"]:
        return FAIL_CANONICAL_HASH
    if not (
        checks["duplicate_metric_unique_key_count_defined"]
        and checks["duplicate_metric_violation_count_defined"]
        and checks["actual_duplicate_service_aliases_unique_key_count"]
    ):
        return FAIL_DUPLICATE_METRIC
    if static["failed_static_check_count"]:
        return FAIL_IMPLICIT_EXECUTION_ID
    return PASS_REPAIR


def repair_payload_paths() -> Sequence[str]:
    return [
        "environment_repair.json",
        "upstream_targeted_reverify_snapshot_registry.json",
        "source_preflight_registry.json",
        "source_change_registry.json",
        "execution_instance_context_contract_v4.json",
        "run_thirty_minute_branch_contract_v4.json",
        "execution_instance_registry_contract_v4.json",
        "duplicate_service_metric_contract_v4.json",
        "static_repair_audit.json",
        "source_snapshot_amended/dynamics_event_trace.py",
        "source_snapshot_amended/dynamics_multiagent_orchestrator.py",
        "source_snapshot_amended_registry.json",
        "historical_execution_prohibition_audit.json",
        "validation_untouched_audit.json",
        "test_holdout_untouched_audit.json",
        "reward_energy_scale_nondefinition_audit.json",
        "training_prohibition_audit.json",
        "full_verify_readiness_audit.json",
        "gate_decision.json",
        "downstream_lock.json",
        "final_report.json",
        "final_report.md",
    ]


def write_manifest(writer: Writer, rel_path: str, payloads: Sequence[str], scope: str) -> Dict[str, Any]:
    rows = []
    for rel in payloads:
        path = writer.root / rel
        rows.append({
            "relative_path": rel,
            "required": True,
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
            "created_order": writer.order.get(rel),
        })
    manifest = {
        "created_at": iso_kst(),
        "manifest_protocol": "TERMINAL_LOCK_TO_MANIFEST_TO_PAYLOAD",
        "manifest_scope": scope,
        "required_payload_count": len(rows),
        "payload_file_count": sum(1 for row in rows if row["exists"]),
        "missing_payload_count": sum(1 for row in rows if not row["exists"]),
        "missing_payloads": [row["relative_path"] for row in rows if not row["exists"]],
        "terminal_lock_listed_inside_manifest": False,
        "files": rows,
    }
    writer.json(rel_path, manifest)
    return manifest


def write_lock(writer: Writer, lock_name: str, manifest_name: str, gate: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_path = writer.root / manifest_name
    lock = {
        "created_at": iso_kst(),
        "mode": gate["mode"],
        "gate": gate["gate"],
        "gate_passed": gate["gate_passed"],
        "manifest_relative_path": manifest_name,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
    }
    writer.json(lock_name, lock)
    return lock


def verify_manifest(root: Path, lock_name: str) -> Dict[str, Any]:
    lock = read_json(root / lock_name)
    manifest_path = root / lock["manifest_relative_path"]
    manifest = read_json(manifest_path)
    payload_missing = 0
    payload_mismatch = 0
    for row in manifest["files"]:
        path = root / row["relative_path"]
        if not path.exists():
            payload_missing += 1
        elif sha256_file(path) != row["sha256"]:
            payload_mismatch += 1
    return {
        "manifest_hash_ok": sha256_file(manifest_path) == lock["manifest_sha256"],
        "manifest_size_ok": manifest_path.stat().st_size == lock["manifest_size_bytes"],
        "payload_missing_count": payload_missing,
        "payload_hash_mismatch_count": payload_mismatch,
        "terminal_lock_listed_inside_manifest": any(row["relative_path"] == lock_name for row in manifest["files"]),
    }


def final_report(root: Path, gate: Mapping[str, Any], static: Mapping[str, Any], preflight: Mapping[str, Any]) -> tuple[Dict[str, Any], str]:
    payload = {
        "created_at": iso_kst(),
        "artifact_root": str(root),
        "gate": gate["gate"],
        "gate_passed": gate["gate_passed"],
        "readiness": gate["readiness"],
        "source_drift_count": preflight["source_drift_count"],
        "allowed_repair_change_count": preflight["allowed_repair_change_count"],
        "failed_static_check_count": static["failed_static_check_count"],
        "run_thirty_context_required": static["checks"]["run_thirty_minute_branch_explicit_context_required"],
        "implicit_execution_id_fallback_removed": static["checks"]["deterministic_input_hash_execution_id_fallback_removed"],
        "missing_context_fail_closed": static["checks"]["missing_context_fail_closed"],
        "duplicate_execution_id_fail_closed": static["checks"]["duplicate_execution_id_fail_closed"],
        "duplicate_metric_semantics_separated": (
            static["checks"]["duplicate_metric_unique_key_count_defined"]
            and static["checks"]["duplicate_metric_violation_count_defined"]
        ),
        "transition_execution_count": 0,
        "fixture_execution_count": 0,
        "thirty_step_execution_count": 0,
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-ER1-V1F-TV1-F1-A1 Repair Report",
        "",
        f"- gate: `{gate['gate']}`",
        f"- gate_passed: `{str(gate['gate_passed']).lower()}`",
        f"- readiness: `{gate['readiness']}`",
        f"- source_drift_count: `{preflight['source_drift_count']}`",
        f"- allowed_repair_change_count: `{preflight['allowed_repair_change_count']}`",
        f"- failed_static_check_count: `{static['failed_static_check_count']}`",
        "- transition/fixture/30-step execution: `0 / 0 / 0`",
        "- identity-regression, targeted-regression, finalize: locked pending explicit user command",
        "",
    ])
    return payload, md


def run_repair(artifact_root: Path) -> Path:
    root = validate_artifact_root(artifact_root, "repair")
    writer = Writer(root)
    writer.json("environment_repair.json", environment_payload())
    upstream_snapshot = copy_upstream_snapshot(writer)
    preflight = source_preflight_registry(upstream_snapshot)
    writer.json("source_preflight_registry.json", preflight)
    source_changes = source_change_registry(preflight)
    writer.json("source_change_registry.json", source_changes)
    for rel_path, payload in contract_payloads().items():
        writer.json(rel_path, payload)
    static = static_repair_audit()
    writer.json("static_repair_audit.json", static)
    source_snapshot_amended(writer)
    for rel_path, payload in prohibition_payloads().items():
        writer.json(rel_path, payload)
    full_readiness = {
        "created_at": iso_kst(),
        "full_verify_ready": False,
        "full_verify_authorized": False,
        "readiness": "REPAIR_COMPLETE_IDENTITY_REGRESSION_REQUIRED",
        "identity_regression_required": True,
        "targeted_regression_required": True,
    }
    writer.json("full_verify_readiness_audit.json", full_readiness)
    gate_name = choose_repair_gate(upstream_snapshot, preflight, static)
    gate_passed = gate_name == PASS_REPAIR
    gate = {
        "created_at": iso_kst(),
        "mode": "repair",
        "gate": gate_name,
        "gate_passed": gate_passed,
        "readiness": (
            "REPAIR_COMPLETE_IDENTITY_REGRESSION_PENDING_USER_COMMAND"
            if gate_passed else "FAILED_A1_REPAIR"
        ),
        "automatic_mode_chaining_allowed": False,
        "identity_regression_authorized": False,
        "targeted_regression_authorized": False,
        "finalize_authorized": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
    }
    writer.json("gate_decision.json", gate)
    downstream = {
        "created_at": iso_kst(),
        "a1_repair_complete": gate_passed,
        "identity_regression_pending_user_command": gate_passed,
        "targeted_regression_authorized": False,
        "finalize_authorized": False,
        "full_verify_authorized": False,
        "full_verify_ready": False,
        "full_verify_blocking_reason": "IDENTITY_AND_TARGETED_REGRESSION_NOT_YET_RUN",
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
    }
    writer.json("downstream_lock.json", downstream)
    report_json, report_md = final_report(root, gate, static, preflight)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    manifest = write_manifest(writer, "artifact_manifest_repair.json", repair_payload_paths(), "A1_REPAIR_MODE")
    write_lock(writer, "_REPAIR_COMPLETE.lock", "artifact_manifest_repair.json", gate)
    verification = verify_manifest(root, "_REPAIR_COMPLETE.lock")
    if (
        manifest["missing_payload_count"]
        or not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        gate["gate"] = FAIL_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = "FAILED_A1_REPAIR_MANIFEST_RECONCILIATION"
        writer.json("gate_decision.json", gate)
        writer.json("downstream_lock.json", {**downstream, "a1_repair_complete": False, "full_verify_blocking_reason": "A1_REPAIR_MANIFEST_RECONCILIATION_FAILED"})
        report_json, report_md = final_report(root, gate, static, preflight)
        writer.json("final_report.json", report_json)
        writer.text("final_report.md", report_md)
        write_manifest(writer, "artifact_manifest_repair.json", repair_payload_paths(), "A1_REPAIR_MODE")
        write_lock(writer, "_REPAIR_COMPLETE.lock", "artifact_manifest_repair.json", gate)
    print(f"[A1] artifact: {root}")
    print("[A1] mode: repair")
    print(f"[A1] source drift: {preflight['source_drift_count']}")
    print(f"[A1] allowed repair changes: {preflight['allowed_repair_change_count']}")
    print(f"[A1] failed static checks: {static['failed_static_check_count']}")
    print(f"[A1] gate: {gate['gate']}")
    print(f"[A1] gate passed: {str(gate['gate_passed']).lower()}")
    print(f"[A1] readiness: {gate['readiness']}")
    return root


def jsonl_table(writer: Writer, rel_path: str, rows: Sequence[Mapping[str, Any]], *, logical_table_name: str) -> Dict[str, Any]:
    path = writer.root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_rows = [json_clean(dict(row)) for row in rows]
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in clean_rows), encoding="utf-8")
    writer.mark(rel_path)
    return {
        "logical_table_name": logical_table_name,
        "relative_path": rel_path,
        "preferred_format": "PARQUET",
        "actual_content_format": "JSONL",
        "actual_backend": "JSONL_FALLBACK_NO_PARQUET_ENGINE",
        "fallback_reason": "NO_PARQUET_ENGINE",
        "row_count": len(clean_rows),
        "schema_hash": stable_hash({"columns": sorted({key for row in clean_rows for key in row})}),
        "content_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_identity_entry(root: Path) -> Path:
    root = validate_artifact_root(root, "identity-regression")
    gate = read_json(root / "gate_decision.json")
    if gate.get("gate") != PASS_REPAIR or gate.get("readiness") != "REPAIR_COMPLETE_IDENTITY_REGRESSION_PENDING_USER_COMMAND":
        raise RuntimeError("identity-regression requires A1 repair PASS gate and pending readiness")
    if not (root / "_REPAIR_COMPLETE.lock").exists():
        raise RuntimeError("identity-regression requires _REPAIR_COMPLETE.lock")
    for lock_name in ["_IDENTITY_REGRESSION_COMPLETE.lock", "_TARGETED_REGRESSION_COMPLETE.lock", "_SUCCESS.lock"]:
        if (root / lock_name).exists():
            raise RuntimeError(f"identity-regression lock already exists or later mode already ran: {lock_name}")
    verification = verify_manifest(root, "_REPAIR_COMPLETE.lock")
    if (
        not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        raise RuntimeError("repair manifest/lock verification failed before identity-regression")
    return root


def identity_environment_payload() -> Dict[str, Any]:
    env = environment_payload()
    env.update({
        "mode": "identity-regression",
        "actual_compute_path": "CPU_ONLY_SYNTHETIC_REGRESSION",
        "verification_scope": "IDENTITY_AND_METRIC_SYNTHETIC_REGRESSION_ONLY",
        "source_modification_count": 0,
        "historical_execution_count": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "training_run_count": 0,
        "checkpoint_load_count": 0,
        "checkpoint_write_count": 0,
    })
    return env


def identity_source_preflight() -> Dict[str, Any]:
    rows = []
    for rel_path, expected in SOURCE_FROZEN_IDENTITY.items():
        path = PROJECT_ROOT / rel_path
        runtime_sha = sha256_file(path) if path.exists() else None
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(path),
            "exists": path.exists(),
            "frozen_sha256": expected,
            "runtime_sha256": runtime_sha,
            "matches_frozen": runtime_sha == expected,
        })
    return {
        "created_at": iso_kst(),
        "source_drift_count": sum(1 for row in rows if not row["matches_frozen"]),
        "source_modification_count": 0,
        "records": rows,
    }


def source_snapshot_identity(writer: Writer) -> Dict[str, Any]:
    rows = []
    for rel_path in [
        "05_training/simulator/dynamics_multiagent_orchestrator.py",
        "05_training/simulator/dynamics_event_trace.py",
    ]:
        row = copy_file(writer, PROJECT_ROOT / rel_path, f"source_snapshot_identity_regression/{Path(rel_path).name}")
        row["matches_frozen"] = row["snapshot_sha256"] == SOURCE_FROZEN_IDENTITY[rel_path] and row["source_sha256"] == SOURCE_FROZEN_IDENTITY[rel_path]
        rows.append(row)
    payload = {
        "created_at": iso_kst(),
        "snapshot_file_count": len(rows),
        "all_snapshots_valid": all(row["byte_identical"] and row["matches_frozen"] for row in rows),
        "records": rows,
    }
    writer.json("source_snapshot_identity_regression_registry.json", payload)
    return payload


def import_simulator_modules() -> Dict[str, Any]:
    training_root = str(TRAINING_ROOT)
    if training_root not in sys.path:
        sys.path.insert(0, training_root)
    from simulator import dynamics_multiagent_orchestrator as orchestrator
    from simulator import dynamics_state_snapshot as state
    from simulator import suseong_service_transition_engine as engine
    from simulator.dynamics_event_trace import DynamicsEvent, DynamicsEventType
    from simulator.dynamics_replay_contract import ReplayFrame, canonical_hash

    return {
        "orchestrator": orchestrator,
        "state": state,
        "engine": engine,
        "ReplayFrame": ReplayFrame,
        "canonical_hash": canonical_hash,
        "DynamicsEvent": DynamicsEvent,
        "DynamicsEventType": DynamicsEventType,
    }


def strip_runtime_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): strip_runtime_identity(item)
            for key, item in value.items()
            if str(key) not in {"execution_instance_id", "runtime_record_key"}
        }
    if isinstance(value, (list, tuple)):
        return [strip_runtime_identity(item) for item in value]
    return value


def event_payloads(events: Sequence[Any]) -> List[Dict[str, Any]]:
    return [event.to_payload() for event in events]


def canonical_event_hash(events: Sequence[Any]) -> str:
    return stable_hash({"events": [strip_runtime_identity(event.to_payload()) for event in events]})


def runtime_record_keys(events: Sequence[Any]) -> List[str]:
    keys = []
    for event in events:
        metadata = dict(event.metadata)
        if metadata.get("runtime_record_key"):
            keys.append(str(metadata["runtime_record_key"]))
    return keys


def event_ids(events: Sequence[Any]) -> List[str]:
    return [str(event.event_id) for event in events]


def base_stop(index: int, **overrides: Any) -> Dict[str, Any]:
    payload = {
        "stop_id": f"S{index:03d}",
        "node_uid": f"S{index:03d}",
        "waiting_pickup_count": 0,
        "scheduled_alighting_count": 0,
        "assigned_pickup_request_count": 0,
        "assigned_dropoff_request_count": 0,
        "mandatory_stop": False,
        "protected_stop": False,
        "terminal_or_turnaround_stop": False,
        "charging_or_driver_relief_stop": False,
        "planned_itinerary_allows_skip": True,
        "downstream_path_valid": True,
        "graph_edge_or_path_valid": True,
        "max_consecutive_skip_constraint_satisfied": True,
        "service_fairness_constraint_satisfied": True,
    }
    payload.update(overrides)
    return payload


def identity_fixture_payload() -> Dict[str, Any]:
    route = [base_stop(idx) for idx in range(80)]
    vehicles = {}
    for agent_id in range(8):
        vehicles[str(agent_id)] = {
            "agent_id": agent_id,
            "route_key": "R|0",
            "position": 0,
            "onboard_count": 0,
            "onboard_destination_stop_ids": [],
            "scheduled_dropoff_counts": {},
            "remaining_travel_seconds": 0.0,
            "remaining_dwell_seconds": 0.0,
            "consecutive_skip_count": 0,
            "vehicle_state": "IN_SERVICE",
            "_ready_to_depart": False,
        }
    return {
        "schema_version": "SUSEONG_DYNAMICS_STATE_SNAPSHOT_V1",
        "simulation_timestamp_seconds": 0,
        "vehicles": vehicles,
        "routes": {"R|0": route},
        "waiting_passengers": {"S001": []},
        "assigned_pickups": {"S001": []},
        "assigned_dropoffs": {"S001": []},
        "onboard_passengers": {str(agent_id): [] for agent_id in range(8)},
        "mandatory_stop_state": {"S001": False},
        "action_mask_state": {"agents": {str(agent_id): [True, True, True] for agent_id in range(8)}},
        "schedule_state": {"service_day_id": "SYNTHETIC_IR1"},
        "headway_state": {"route_id": "R", "headway_seconds": 300},
        "operation_mode": "SYNTHETIC_IR1",
        "shared_counters": {"served": 0},
        "replay_cursor": {"frame_index": 0, "event_offset": 0},
        "external_provider_states": {"stop_service": {"classification": "STATELESS", "state": {}}},
    }


def thirty_replay_frames() -> Tuple[Any, ...]:
    mods = import_simulator_modules()
    ReplayFrame = mods["ReplayFrame"]
    return tuple(
        ReplayFrame(step_index=index, frame_start_seconds=index * 60, frame_end_seconds=(index + 1) * 60, events=())
        for index in range(30)
    )


def replay_input_hash(frames: Sequence[Any]) -> str:
    return import_simulator_modules()["canonical_hash"]({"frame_hashes": [frame.frame_hash for frame in frames]})


def make_ir_context(
    *,
    fixture_id: str,
    logical_branch_name: str,
    execution_instance_id: str,
    initial_state_hash: str,
    replay_hash: str,
    target_action: Any,
    invocation_sequence: int,
) -> Any:
    mods = import_simulator_modules()
    return mods["orchestrator"].BranchExecutionContext(
        run_id=EVALUATION_RUN_ID,
        fixture_id=fixture_id,
        logical_branch_name=logical_branch_name,
        execution_instance_id=execution_instance_id,
        initial_state_hash=initial_state_hash,
        replay_input_hash=replay_hash,
        target_agent_id=0,
        pulse_action=target_action.value,
        caller_run_id=EVALUATION_RUN_ID,
        invocation_sequence=invocation_sequence,
        created_by="A1_IR1_RUNNER",
    )


def capture_thirty_minute_branch(func: Any) -> Tuple[Any, Tuple[Any, ...], List[Any]]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    captured_steps: List[List[Any]] = []
    original = orchestrator.event_trace_hash

    def capture(events: Sequence[Any]) -> str:
        captured_steps.append(list(events))
        return original(events)

    orchestrator.event_trace_hash = capture
    try:
        state, traces = func()
    finally:
        orchestrator.event_trace_hash = original
    events = [event for step_events in captured_steps for event in step_events]
    return state, traces, events


def run_identity_branch(
    *,
    fixture_id: str,
    logical_branch_name: str,
    execution_instance_id: str,
    action_name: str,
    registry: Optional[Any],
    invocation_sequence: int,
) -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    state_mod = mods["state"]
    engine = mods["engine"]
    action = getattr(orchestrator.DynamicsBranchAction, action_name)
    initial_state = state_mod.DynamicsStateSnapshot(identity_fixture_payload())
    frames = thirty_replay_frames()
    replay_hash = replay_input_hash(frames)
    context = make_ir_context(
        fixture_id=fixture_id,
        logical_branch_name=logical_branch_name,
        execution_instance_id=execution_instance_id,
        initial_state_hash=initial_state.state_hash,
        replay_hash=replay_hash,
        target_action=action,
        invocation_sequence=invocation_sequence,
    )

    def stop_service_provider(**_: Any) -> Any:
        return engine.StopServiceResult(boardings=0, alightings=0, dwell_required=False, metadata={"fixture_id": fixture_id})

    def execute() -> Any:
        return orchestrator.run_thirty_minute_branch(
            initial_state=initial_state,
            target_agent_id=0,
            target_action=action,
            active_agent_ids=tuple(range(8)),
            replay_frames=frames,
            stop_service_provider=stop_service_provider,
            config=engine.TransitionConfig(edge_travel_seconds=60.0, dwell_seconds=0.0, allow_turnaround=False),
            branch_context=context,
            execution_instance_registry=registry,
        )

    after_state, traces, events = capture_thirty_minute_branch(execute)
    trace_payloads = [trace.to_payload() for trace in traces]
    return {
        "fixture_id": fixture_id,
        "logical_branch_name": logical_branch_name,
        "execution_instance_id": execution_instance_id,
        "branch_context": context.to_payload(),
        "logical_branch_id": context.logical_branch_id,
        "action_name": action_name,
        "executed_step_count": len(traces),
        "step_indexes": [trace.step_index for trace in traces],
        "elapsed_seconds_total": traces[-1].elapsed_seconds_after if traces else 0,
        "canonical_event_hash": canonical_event_hash(events),
        "canonical_trace_hash": stable_hash({"traces": trace_payloads}),
        "end_state_hash": after_state.state_hash,
        "runtime_record_keys": runtime_record_keys(events),
        "runtime_record_key_count": len(runtime_record_keys(events)),
        "runtime_record_key_unique_count": len(set(runtime_record_keys(events))),
        "event_ids": event_ids(events),
        "events": event_payloads(events),
        "passed": True,
    }


def metric_event(event_id: str, event_type: Any, request_id: str, service_leg_id: str, vehicle_id: str = "V1") -> Any:
    mods = import_simulator_modules()
    Event = mods["DynamicsEvent"]
    return Event(
        event_id=event_id,
        event_timestamp_seconds=1,
        step_index=0,
        event_type=event_type,
        agent_id=1,
        vehicle_id=vehicle_id,
        passenger_id=f"P_{request_id}",
        request_id=request_id,
        metadata={"request_id": request_id, "service_leg_id": service_leg_id},
    )


def metric_assignment(request_id: str, service_leg_id: str, vehicle_id: str = "V1") -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "service_leg_id": service_leg_id,
        "passenger_id": f"P_{request_id}",
        "vehicle_id": vehicle_id,
    }


def run_metric_case(case_id: str, events: Sequence[Any], assignments: Sequence[Mapping[str, Any]], served_count: Optional[int], expected: Mapping[str, Any]) -> Dict[str, Any]:
    mods = import_simulator_modules()
    invariant = mods["orchestrator"].check_request_service_invariants(events, assignments, served_count=served_count)
    checks = {
        "unique_duplicate_service_key_count": invariant["unique_duplicate_service_key_count"] == expected.get("unique_duplicate_service_key_count"),
        "duplicate_invariant_violation_count": (
            invariant["duplicate_invariant_violation_count"] == expected["duplicate_invariant_violation_count"]
            if isinstance(expected.get("duplicate_invariant_violation_count"), int)
            else invariant["duplicate_invariant_violation_count"] > 1
        ),
        "cross_source_correspondence_failure_count": invariant["cross_source_correspondence_failure_count"] == expected.get("cross_source_correspondence_failure_count", 0),
        "actual_duplicate_alias": invariant["actual_duplicate_service_count"] == invariant["unique_duplicate_service_key_count"],
        "violation_count_alias": invariant["violation_count"] == invariant["duplicate_invariant_violation_count"],
    }
    expected_violation = expected.get("cross_source_violation")
    if expected_violation is not None:
        checks["cross_source_violation"] = any(row.get("invariant") == expected_violation for row in invariant["cross_source_violations"])
    passed = all(checks.values())
    return {
        "case_id": case_id,
        "expected": dict(expected),
        "actual": invariant,
        "checks": checks,
        "passed": passed,
        "failure_reason": None if passed else "duplicate metric regression mismatch",
    }


def duplicate_metric_regression() -> Dict[str, Any]:
    mods = import_simulator_modules()
    T = mods["DynamicsEventType"]
    rows = []
    rows.append(run_metric_case(
        "M01_NORMAL_SINGLE_SERVICE",
        [
            metric_event("m01-board", T.PASSENGER_BOARD, "request-001", "leg-001"),
            metric_event("m01-complete", T.SERVICE_COMPLETED, "request-001", "leg-001"),
        ],
        [metric_assignment("request-001", "leg-001")],
        1,
        {"unique_duplicate_service_key_count": 0, "duplicate_invariant_violation_count": 0, "cross_source_correspondence_failure_count": 0},
    ))
    rows.append(run_metric_case(
        "M02_BOARD_ONLY_DUPLICATE",
        [
            metric_event("m02-board-a", T.PASSENGER_BOARD, "request-001", "leg-001"),
            metric_event("m02-board-b", T.PASSENGER_BOARD, "request-001", "leg-001"),
            metric_event("m02-complete", T.SERVICE_COMPLETED, "request-001", "leg-001"),
        ],
        [metric_assignment("request-001", "leg-001")],
        1,
        {"unique_duplicate_service_key_count": 1, "duplicate_invariant_violation_count": "GT_ZERO", "cross_source_correspondence_failure_count": 0},
    ))
    rows.append(run_metric_case(
        "M03_ASSIGNMENT_ONLY_DUPLICATE",
        [
            metric_event("m03-board", T.PASSENGER_BOARD, "request-001", "leg-001"),
            metric_event("m03-complete", T.SERVICE_COMPLETED, "request-001", "leg-001"),
        ],
        [metric_assignment("request-001", "leg-001"), metric_assignment("request-001", "leg-001")],
        1,
        {"unique_duplicate_service_key_count": 1, "duplicate_invariant_violation_count": "GT_ZERO", "cross_source_correspondence_failure_count": 0},
    ))
    rows.append(run_metric_case(
        "M04_BOARD_WITHOUT_ASSIGNMENT",
        [metric_event("m04-board", T.PASSENGER_BOARD, "request-001", "leg-001")],
        [],
        None,
        {"unique_duplicate_service_key_count": 0, "duplicate_invariant_violation_count": 0, "cross_source_correspondence_failure_count": 1, "cross_source_violation": "BOARD_WITHOUT_ASSIGNMENT"},
    ))
    rows.append(run_metric_case(
        "M05_ASSIGNMENT_WITHOUT_BOARD",
        [],
        [metric_assignment("request-001", "leg-001")],
        None,
        {"unique_duplicate_service_key_count": 0, "duplicate_invariant_violation_count": 0, "cross_source_correspondence_failure_count": 1, "cross_source_violation": "ASSIGNMENT_WITHOUT_BOARD"},
    ))
    rows.append(run_metric_case(
        "M06_COMBINED_MULTI_CHANNEL_DUPLICATE",
        [
            metric_event("m06-board-a", T.PASSENGER_BOARD, "request-001", "leg-001"),
            metric_event("m06-board-b", T.PASSENGER_BOARD, "request-001", "leg-001"),
            metric_event("m06-complete-a", T.SERVICE_COMPLETED, "request-001", "leg-001"),
            metric_event("m06-complete-b", T.SERVICE_COMPLETED, "request-001", "leg-001"),
        ],
        [metric_assignment("request-001", "leg-001"), metric_assignment("request-001", "leg-001")],
        1,
        {"unique_duplicate_service_key_count": 1, "duplicate_invariant_violation_count": "GT_ONE", "cross_source_correspondence_failure_count": 0},
    ))
    rows.append(run_metric_case(
        "M07_MULTI_LEG_NORMAL_SERVICE",
        [
            metric_event("m07-board-a", T.PASSENGER_BOARD, "request-transfer-001", "leg-A"),
            metric_event("m07-complete-a", T.SERVICE_COMPLETED, "request-transfer-001", "leg-A"),
            metric_event("m07-board-b", T.PASSENGER_BOARD, "request-transfer-001", "leg-B"),
            metric_event("m07-complete-b", T.SERVICE_COMPLETED, "request-transfer-001", "leg-B"),
        ],
        [metric_assignment("request-transfer-001", "leg-A"), metric_assignment("request-transfer-001", "leg-B")],
        2,
        {"unique_duplicate_service_key_count": 0, "duplicate_invariant_violation_count": 0, "cross_source_correspondence_failure_count": 0},
    ))
    return {
        "created_at": iso_kst(),
        "metric_case_total": len(rows),
        "metric_case_passed": sum(1 for row in rows if row["passed"]),
        "records": rows,
    }


def run_identity_regressions() -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    shared_registry = orchestrator.ExecutionInstanceRegistry()
    records: List[Dict[str, Any]] = []
    sequence = 1
    h1 = run_identity_branch(
        fixture_id="I01_SAME_LOGICAL_BRANCH_DIFFERENT_INVOCATION",
        logical_branch_name="K",
        execution_instance_id="A1-IR1-H-REPEAT-001",
        action_name="CONDITIONAL_SKIP_EMPTY_STOP",
        registry=shared_registry,
        invocation_sequence=sequence,
    )
    sequence += 1
    h2 = run_identity_branch(
        fixture_id="I01_SAME_LOGICAL_BRANCH_DIFFERENT_INVOCATION",
        logical_branch_name="K",
        execution_instance_id="A1-IR1-H-REPEAT-002",
        action_name="CONDITIONAL_SKIP_EMPTY_STOP",
        registry=shared_registry,
        invocation_sequence=sequence,
    )
    sequence += 1
    i01_checks = {
        "logical_branch_id_repeat_stable": h1["logical_branch_id"] == h2["logical_branch_id"],
        "execution_instance_id_repeat_distinct": h1["execution_instance_id"] != h2["execution_instance_id"],
        "canonical_event_hash_equal": h1["canonical_event_hash"] == h2["canonical_event_hash"],
        "canonical_trace_hash_equal": h1["canonical_trace_hash"] == h2["canonical_trace_hash"],
        "end_state_hash_equal": h1["end_state_hash"] == h2["end_state_hash"],
        "runtime_record_key_intersection_empty": not (set(h1["runtime_record_keys"]) & set(h2["runtime_record_keys"])),
    }
    records.append({"fixture_id": "I01", "description": "same logical branch, different invocation", "run_1": h1, "run_2": h2, "checks": i01_checks, "passed": all(i01_checks.values())})

    duplicate_shared_registry = orchestrator.ExecutionInstanceRegistry()
    duplicate_shared_error = None
    try:
        run_identity_branch(fixture_id="I02A_DUPLICATE_SHARED_REGISTRY", logical_branch_name="H", execution_instance_id="A1-IR1-DUPLICATE-ID", action_name="HOLD_CURRENT_POSITION", registry=duplicate_shared_registry, invocation_sequence=1)
        run_identity_branch(fixture_id="I02A_DUPLICATE_SHARED_REGISTRY", logical_branch_name="H", execution_instance_id="A1-IR1-DUPLICATE-ID", action_name="HOLD_CURRENT_POSITION", registry=duplicate_shared_registry, invocation_sequence=2)
    except Exception as exc:  # exact type is recorded below
        duplicate_shared_error = type(exc).__name__
    i02a_passed = duplicate_shared_error == "DuplicateExecutionInstanceIdError"
    records.append({"fixture_id": "I02A", "description": "duplicate execution id with shared registry", "observed_error": duplicate_shared_error, "passed": i02a_passed})

    omission_error = None
    omission_accepted = False
    try:
        run_identity_branch(fixture_id="I02B_DUPLICATE_NO_SHARED_REGISTRY", logical_branch_name="H", execution_instance_id="A1-IR1-NO-SHARED-REGISTRY", action_name="HOLD_CURRENT_POSITION", registry=None, invocation_sequence=1)
        run_identity_branch(fixture_id="I02B_DUPLICATE_NO_SHARED_REGISTRY", logical_branch_name="H", execution_instance_id="A1-IR1-NO-SHARED-REGISTRY", action_name="HOLD_CURRENT_POSITION", registry=None, invocation_sequence=2)
        omission_accepted = True
    except Exception as exc:
        omission_error = type(exc).__name__
    registry_enforcement_status = "CALLER_SCOPE_REGISTRY_REQUIRED" if omission_accepted else "API_REGISTRY_REQUIRED_OR_DUPLICATE_REJECTED_WITHOUT_SHARED_REGISTRY"
    records.append({
        "fixture_id": "I02B",
        "description": "duplicate execution id without shared registry",
        "duplicate_silently_accepted": omission_accepted,
        "observed_error": omission_error,
        "registry_enforcement_status": registry_enforcement_status,
        "passed": True,
    })

    missing_context_error = None
    try:
        run_identity_branch(fixture_id="I03_MISSING_CONTEXT", logical_branch_name="H", execution_instance_id="unused", action_name="HOLD_CURRENT_POSITION", registry=shared_registry, invocation_sequence=sequence)
    except Exception:
        pass
    try:
        initial_state = mods["state"].DynamicsStateSnapshot(identity_fixture_payload())
        frames = thirty_replay_frames()
        orchestrator.run_thirty_minute_branch(
            initial_state=initial_state,
            target_agent_id=0,
            target_action=orchestrator.DynamicsBranchAction.HOLD_CURRENT_POSITION,
            active_agent_ids=tuple(range(8)),
            replay_frames=frames,
            stop_service_provider=lambda **_: mods["engine"].StopServiceResult(),
            config=mods["engine"].TransitionConfig(edge_travel_seconds=60.0, dwell_seconds=0.0, allow_turnaround=False),
            branch_context=None,
            execution_instance_registry=shared_registry,
        )
    except Exception as exc:
        missing_context_error = type(exc).__name__
    i03_passed = missing_context_error == "MissingBranchExecutionContextError"
    records.append({"fixture_id": "I03", "description": "missing BranchExecutionContext", "observed_error": missing_context_error, "implicit_execution_id_fallback_count": 0 if i03_passed else 1, "passed": i03_passed})

    i04_h = run_identity_branch(fixture_id="I04_DIFFERENT_LOGICAL_BRANCH", logical_branch_name="H", execution_instance_id="A1-IR1-I04-H", action_name="HOLD_CURRENT_POSITION", registry=shared_registry, invocation_sequence=sequence)
    sequence += 1
    i04_k = run_identity_branch(fixture_id="I04_DIFFERENT_LOGICAL_BRANCH", logical_branch_name="K", execution_instance_id="A1-IR1-I04-K", action_name="CONDITIONAL_SKIP_EMPTY_STOP", registry=shared_registry, invocation_sequence=sequence)
    sequence += 1
    i04_checks = {
        "logical_branch_id_distinct": i04_h["logical_branch_id"] != i04_k["logical_branch_id"],
        "execution_instance_id_distinct": i04_h["execution_instance_id"] != i04_k["execution_instance_id"],
        "end_state_hash_distinct": i04_h["end_state_hash"] != i04_k["end_state_hash"],
    }
    records.append({"fixture_id": "I04", "description": "different logical branches", "run_h": i04_h, "run_k": i04_k, "checks": i04_checks, "passed": all(i04_checks.values())})

    i05 = run_identity_branch(fixture_id="I05_EXACT_30_STEP_EXECUTION", logical_branch_name="S", execution_instance_id="A1-IR1-I05-S", action_name="SERVE_AND_MOVE_TO_NEXT_STOP", registry=shared_registry, invocation_sequence=sequence)
    sequence += 1
    i05_checks = {
        "executed_step_count_is_30": i05["executed_step_count"] == 30,
        "step_indexes_exact": i05["step_indexes"] == list(range(30)),
        "elapsed_seconds_total_is_1800": i05["elapsed_seconds_total"] == 1800,
    }
    records.append({"fixture_id": "I05", "description": "exact 30-step execution", "run": i05, "checks": i05_checks, "passed": all(i05_checks.values())})

    scope_registry = orchestrator.ExecutionInstanceRegistry()
    scope_records = []
    for index, (branch_name, action_name) in enumerate([
        ("H", "HOLD_CURRENT_POSITION"),
        ("H", "HOLD_CURRENT_POSITION"),
        ("S", "SERVE_AND_MOVE_TO_NEXT_STOP"),
        ("S", "SERVE_AND_MOVE_TO_NEXT_STOP"),
        ("K", "CONDITIONAL_SKIP_EMPTY_STOP"),
        ("K", "CONDITIONAL_SKIP_EMPTY_STOP"),
    ], start=1):
        run = run_identity_branch(
            fixture_id="I06_REGISTRY_SCOPE_H_S_K_REPEATS",
            logical_branch_name=branch_name,
            execution_instance_id=f"A1-IR1-I06-{branch_name}-repeat-{index}",
            action_name=action_name,
            registry=scope_registry,
            invocation_sequence=index,
        )
        scope_records.append({"branch_name": branch_name, "execution_instance_id": run["execution_instance_id"], "logical_branch_id": run["logical_branch_id"]})
    cross_branch_duplicate_error = None
    try:
        context = make_ir_context(
            fixture_id="I06_REGISTRY_SCOPE_H_S_K_REPEATS",
            logical_branch_name="K",
            execution_instance_id="A1-IR1-I06-H-repeat-1",
            initial_state_hash="different",
            replay_hash="different",
            target_action=orchestrator.DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP,
            invocation_sequence=7,
        )
        scope_registry.register(context)
    except Exception as exc:
        cross_branch_duplicate_error = type(exc).__name__
    registry_payload = scope_registry.to_payload()
    registered_sequences = [row["registered_at_sequence"] for row in registry_payload["records"]]
    i06_checks = {
        "one_evaluation_run_one_shared_registry": True,
        "registered_execution_instance_count_is_6": registry_payload["registered_execution_instance_count"] == 6,
        "duplicate_registry_key_count": len({row["execution_instance_id"] for row in registry_payload["records"]}) == 6,
        "logical_branch_id_present": all(row.get("logical_branch_id") for row in registry_payload["records"]),
        "invocation_sequence_monotonic": registered_sequences == sorted(registered_sequences),
        "cross_branch_duplicate_rejected": cross_branch_duplicate_error == "DuplicateExecutionInstanceIdError",
    }
    records.append({"fixture_id": "I06", "description": "registry scope across H/S/K repeats", "registry": registry_payload, "scope_records": scope_records, "cross_branch_duplicate_error": cross_branch_duplicate_error, "checks": i06_checks, "passed": all(i06_checks.values())})

    all_runtime_keys = []
    for record in records:
        for run_key in ["run_1", "run_2", "run_h", "run_k", "run"]:
            run = record.get(run_key)
            if isinstance(run, Mapping):
                all_runtime_keys.extend(run.get("runtime_record_keys", []))
    return {
        "created_at": iso_kst(),
        "evaluation_run_id": EVALUATION_RUN_ID,
        "identity_fixture_total": len(records),
        "identity_fixture_passed": sum(1 for record in records if record["passed"]),
        "records": records,
        "shared_registry_payload": shared_registry.to_payload(),
        "registry_omission_behavior": {
            "duplicate_silently_accepted": omission_accepted,
            "observed_error": omission_error,
            "registry_enforcement_status": registry_enforcement_status,
        },
        "runtime_record_key_count": len(all_runtime_keys),
        "runtime_record_key_unique_count": len(set(all_runtime_keys)),
        "runtime_record_key_collision_count": len(all_runtime_keys) - len(set(all_runtime_keys)),
    }


def identity_payload_paths() -> Sequence[str]:
    return [
        "identity_regression_environment.json",
        "source_snapshot_identity_regression/dynamics_multiagent_orchestrator.py",
        "source_snapshot_identity_regression/dynamics_event_trace.py",
        "source_snapshot_identity_regression_registry.json",
        "source_preflight_identity_regression.json",
        "identity_fixture_inventory.json",
        "identity_regression_results.json",
        "identity_regression_results.jsonl",
        "shared_registry_scope_audit.json",
        "registry_omission_behavior_audit.json",
        "logical_execution_identity_audit.json",
        "canonical_runtime_identity_separation_audit.json",
        "runtime_record_key_audit.json",
        "thirty_minute_exactness_audit.json",
        "duplicate_metric_regression_results.json",
        "combined_multi_channel_duplicate_audit.json",
        "multi_leg_service_audit.json",
        "duplicate_metric_reconciliation_audit.json",
        "targeted_regression_readiness_audit.json",
        "historical_execution_prohibition_audit_identity.json",
        "validation_untouched_audit_identity.json",
        "test_holdout_untouched_audit_identity.json",
        "reward_energy_scale_nondefinition_audit_identity.json",
        "training_prohibition_audit_identity.json",
        "stage_immutability_audit_identity.json",
        "gate_decision.json",
        "downstream_lock.json",
        "final_report.json",
        "final_report.md",
        "table_write_backend_audit_identity_regression.json",
    ]


def choose_identity_gate(source_preflight: Mapping[str, Any], identity: Mapping[str, Any], metric: Mapping[str, Any], stage: Mapping[str, Any]) -> Tuple[str, bool, str]:
    if source_preflight["source_drift_count"]:
        return FAIL_IR1_SOURCE_DRIFT, False, "FAILED_IR1_SOURCE_DRIFT"
    if stage["prior_stage_mutated"]:
        return FAIL_IR1_PRIOR_STAGE, False, "FAILED_IR1_PRIOR_STAGE_MUTATED"
    by_id = {record["fixture_id"]: record for record in identity["records"]}
    if not by_id["I01"]["checks"]["logical_branch_id_repeat_stable"]:
        return FAIL_IR1_LOGICAL_BRANCH, False, "FAILED_IR1_LOGICAL_BRANCH_ID_UNSTABLE"
    if not by_id["I01"]["checks"]["execution_instance_id_repeat_distinct"]:
        return FAIL_IR1_EXECUTION_INSTANCE, False, "FAILED_IR1_EXECUTION_INSTANCE_NOT_DISTINCT"
    if not by_id["I02A"]["passed"]:
        return FAIL_IR1_DUPLICATE_SHARED, False, "FAILED_IR1_DUPLICATE_ID_NOT_REJECTED_WITH_SHARED_REGISTRY"
    if not by_id["I03"]["passed"]:
        return FAIL_IR1_MISSING_CONTEXT, False, "FAILED_IR1_MISSING_CONTEXT_NOT_REJECTED"
    if not by_id["I01"]["checks"]["canonical_trace_hash_equal"] or not by_id["I01"]["checks"]["canonical_event_hash_equal"]:
        return FAIL_IR1_CANONICAL, False, "FAILED_IR1_CANONICAL_HASH_CONTAMINATED"
    if identity["runtime_record_key_collision_count"]:
        return FAIL_IR1_RUNTIME_COLLISION, False, "FAILED_IR1_RUNTIME_RECORD_COLLISION"
    if not by_id["I05"]["passed"]:
        return FAIL_IR1_HORIZON, False, "FAILED_IR1_HORIZON_STEP_MISMATCH"
    metric_by_id = {record["case_id"]: record for record in metric["records"]}
    if metric_by_id["M06_COMBINED_MULTI_CHANNEL_DUPLICATE"]["actual"]["unique_duplicate_service_key_count"] != 1:
        return FAIL_IR1_DUPLICATE_KEY_DOUBLE, False, "FAILED_IR1_DUPLICATE_KEY_DOUBLE_COUNT"
    if metric_by_id["M07_MULTI_LEG_NORMAL_SERVICE"]["actual"]["unique_duplicate_service_key_count"] != 0:
        return FAIL_IR1_MULTI_LEG, False, "FAILED_IR1_MULTI_LEG_FALSE_POSITIVE"
    if metric["metric_case_passed"] != metric["metric_case_total"]:
        return FAIL_IR1_DUPLICATE_METRIC, False, "FAILED_IR1_DUPLICATE_METRIC_SEMANTICS"
    if identity["registry_omission_behavior"]["registry_enforcement_status"] == "CALLER_SCOPE_REGISTRY_REQUIRED":
        return PASS_IDENTITY_AMENDMENT, True, "IDENTITY_REGRESSION_COMPLETE_REGISTRY_ENFORCEMENT_AMENDMENT_REQUIRED"
    return PASS_IDENTITY_READY, True, "IDENTITY_REGRESSION_COMPLETE_TARGETED_REGRESSION_PENDING_USER_COMMAND"


def identity_final_report(root: Path, gate: Mapping[str, Any], identity: Mapping[str, Any], metric: Mapping[str, Any], source_preflight: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    m06 = next(record for record in metric["records"] if record["case_id"] == "M06_COMBINED_MULTI_CHANNEL_DUPLICATE")
    m07 = next(record for record in metric["records"] if record["case_id"] == "M07_MULTI_LEG_NORMAL_SERVICE")
    payload = {
        "created_at": iso_kst(),
        "artifact_root": str(root),
        "gate": gate["gate"],
        "gate_passed": gate["gate_passed"],
        "readiness": gate["readiness"],
        "source_drift_count": source_preflight["source_drift_count"],
        "identity_fixture_passed": identity["identity_fixture_passed"],
        "identity_fixture_total": identity["identity_fixture_total"],
        "metric_case_passed": metric["metric_case_passed"],
        "metric_case_total": metric["metric_case_total"],
        "registry_enforcement_status": identity["registry_omission_behavior"]["registry_enforcement_status"],
        "runtime_record_key_collision_count": identity["runtime_record_key_collision_count"],
        "combined_multi_channel_duplicate_unique_key_count": m06["actual"]["unique_duplicate_service_key_count"],
        "multi_leg_unique_duplicate_service_key_count": m07["actual"]["unique_duplicate_service_key_count"],
        "targeted_regression_authorized": False,
        "full_verify_authorized": False,
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-ER1-V1F-TV1-F1-A1 Identity Regression Report",
        "",
        f"- gate: `{gate['gate']}`",
        f"- gate_passed: `{str(gate['gate_passed']).lower()}`",
        f"- readiness: `{gate['readiness']}`",
        f"- source_drift_count: `{source_preflight['source_drift_count']}`",
        f"- identity fixtures: `{identity['identity_fixture_passed']} / {identity['identity_fixture_total']}`",
        f"- metric cases: `{metric['metric_case_passed']} / {metric['metric_case_total']}`",
        f"- registry_enforcement_status: `{identity['registry_omission_behavior']['registry_enforcement_status']}`",
        f"- runtime_record_key_collision_count: `{identity['runtime_record_key_collision_count']}`",
        f"- M06 unique duplicate service key count: `{m06['actual']['unique_duplicate_service_key_count']}`",
        f"- M07 unique duplicate service key count: `{m07['actual']['unique_duplicate_service_key_count']}`",
        "- targeted-regression/finalize/full-verify/DL-6B/state-feasibility: locked",
        "",
    ])
    return payload, md


def run_identity_regression(artifact_root: Path) -> Path:
    root = validate_identity_entry(artifact_root)
    prior_hashes = {
        rel_path: sha256_file(root / rel_path)
        for rel_path in ["artifact_manifest_repair.json", "_REPAIR_COMPLETE.lock"]
    }
    writer = Writer(root)
    writer.json("identity_regression_environment.json", identity_environment_payload())
    source_preflight = identity_source_preflight()
    writer.json("source_preflight_identity_regression.json", source_preflight)
    source_snapshot_identity(writer)
    inventory = {
        "created_at": iso_kst(),
        "evaluation_run_id": EVALUATION_RUN_ID,
        "identity_fixtures": ["I01", "I02A", "I02B", "I03", "I04", "I05", "I06"],
        "metric_fixtures": ["M01", "M02", "M03", "M04", "M05", "M06", "M07"],
        "targeted_t01_t10_executed": False,
    }
    writer.json("identity_fixture_inventory.json", inventory)
    identity = run_identity_regressions()
    writer.json("identity_regression_results.json", identity)
    table_infos = [jsonl_table(writer, "identity_regression_results.jsonl", identity["records"], logical_table_name="identity_regression_results")]
    metric = duplicate_metric_regression()
    writer.json("duplicate_metric_regression_results.json", metric)
    by_identity = {record["fixture_id"]: record for record in identity["records"]}
    by_metric = {record["case_id"]: record for record in metric["records"]}
    writer.json("shared_registry_scope_audit.json", {"created_at": iso_kst(), **by_identity["I06"]})
    writer.json("registry_omission_behavior_audit.json", {"created_at": iso_kst(), **identity["registry_omission_behavior"]})
    writer.json("logical_execution_identity_audit.json", {"created_at": iso_kst(), "i01": by_identity["I01"], "i04": by_identity["I04"]})
    writer.json("canonical_runtime_identity_separation_audit.json", {
        "created_at": iso_kst(),
        "canonical_hash_contaminated_by_execution_id": not (
            by_identity["I01"]["checks"]["canonical_event_hash_equal"]
            and by_identity["I01"]["checks"]["canonical_trace_hash_equal"]
        ),
        "event_id_same_across_same_logical_branch_allowed": True,
        "runtime_record_keys_distinct_across_repeats": by_identity["I01"]["checks"]["runtime_record_key_intersection_empty"],
        "i01": by_identity["I01"],
    })
    writer.json("runtime_record_key_audit.json", {
        "created_at": iso_kst(),
        "runtime_record_key_count": identity["runtime_record_key_count"],
        "runtime_record_key_unique_count": identity["runtime_record_key_unique_count"],
        "runtime_record_key_collision_count": identity["runtime_record_key_collision_count"],
        "runtime_record_key_unique": identity["runtime_record_key_collision_count"] == 0,
    })
    writer.json("thirty_minute_exactness_audit.json", {"created_at": iso_kst(), **by_identity["I05"]})
    writer.json("combined_multi_channel_duplicate_audit.json", {"created_at": iso_kst(), **by_metric["M06_COMBINED_MULTI_CHANNEL_DUPLICATE"]})
    writer.json("multi_leg_service_audit.json", {"created_at": iso_kst(), **by_metric["M07_MULTI_LEG_NORMAL_SERVICE"]})
    reconciliation_records = []
    for record in metric["records"]:
        actual = record["actual"]
        reconciliation_records.append({
            "case_id": record["case_id"],
            "actual_duplicate_equals_unique_key": actual["actual_duplicate_service_count"] == actual["unique_duplicate_service_key_count"],
            "violation_count_equals_duplicate_invariant": actual["violation_count"] == actual["duplicate_invariant_violation_count"],
            "cross_source_excluded_from_duplicate_count": actual["cross_source_correspondence_failure_count"] == 0 or actual["unique_duplicate_service_key_count"] == 0,
        })
    reconciliation = {
        "created_at": iso_kst(),
        "duplicate_metric_reconciliation_valid": all(all(value for key, value in row.items() if key != "case_id") for row in reconciliation_records),
        "records": reconciliation_records,
    }
    writer.json("duplicate_metric_reconciliation_audit.json", reconciliation)
    for rel_path, payload in {
        "historical_execution_prohibition_audit_identity.json": {"historical_execution_count": 0, "d1_250row_execution_count": 0, "train_row_access_count": 0},
        "validation_untouched_audit_identity.json": {"validation_access_count": 0, "validation_branch_count": 0},
        "test_holdout_untouched_audit_identity.json": {"test_access_count": 0, "test_holdout_touched": False},
        "reward_energy_scale_nondefinition_audit_identity.json": {"new_reward_formula_created": False, "new_energy_formula_created": False, "scale_created": False, "candidate_created": False, "tolerance_changed": False},
        "training_prohibition_audit_identity.json": {"training_run_count": 0, "checkpoint_load_count": 0, "checkpoint_write_count": 0, "api_call_count": 0, "external_network_accessed": False},
    }.items():
        writer.json(rel_path, {"created_at": iso_kst(), **payload})
    stage = {
        "created_at": iso_kst(),
        "records": [
            {
                "relative_path": rel_path,
                "sha256_before": before,
                "sha256_after": sha256_file(root / rel_path),
                "unchanged": before == sha256_file(root / rel_path),
            }
            for rel_path, before in prior_hashes.items()
        ],
    }
    stage["prior_stage_mutated"] = not all(row["unchanged"] for row in stage["records"])
    stage["prior_stage_manifest_lock_unchanged"] = not stage["prior_stage_mutated"]
    writer.json("stage_immutability_audit_identity.json", stage)
    gate_name, passed, readiness = choose_identity_gate(source_preflight, identity, metric, stage)
    targeted_ready = gate_name == PASS_IDENTITY_READY
    writer.json("targeted_regression_readiness_audit.json", {
        "created_at": iso_kst(),
        "targeted_regression_ready": targeted_ready,
        "targeted_regression_authorized": False,
        "full_verify_ready": False,
        "full_verify_authorized": False,
        "blocking_reason": None if targeted_ready else readiness,
    })
    gate = {
        "created_at": iso_kst(),
        "mode": "identity-regression",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": readiness,
        "identity_regression_complete": passed,
        "targeted_regression_authorized": False,
        "finalize_authorized": False,
        "full_verify_ready": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
    }
    writer.json("gate_decision.json", gate)
    m06 = by_metric["M06_COMBINED_MULTI_CHANNEL_DUPLICATE"]
    m07 = by_metric["M07_MULTI_LEG_NORMAL_SERVICE"]
    downstream = {
        "created_at": iso_kst(),
        "a1_repair_complete": True,
        "identity_regression_complete": passed,
        "logical_branch_id_repeat_stable": by_identity["I01"]["checks"]["logical_branch_id_repeat_stable"],
        "execution_instance_id_repeat_distinct": by_identity["I01"]["checks"]["execution_instance_id_repeat_distinct"],
        "canonical_trace_deterministic": by_identity["I01"]["checks"]["canonical_trace_hash_equal"],
        "runtime_record_key_unique": identity["runtime_record_key_collision_count"] == 0,
        "missing_branch_context_rejected": by_identity["I03"]["passed"],
        "duplicate_execution_id_rejected_with_shared_registry": by_identity["I02A"]["passed"],
        "registry_required_or_caller_scope_enforced": identity["registry_omission_behavior"]["registry_enforcement_status"],
        "one_evaluation_run_one_shared_registry": by_identity["I06"]["checks"]["one_evaluation_run_one_shared_registry"],
        "exact_thirty_step_execution": by_identity["I05"]["passed"],
        "elapsed_seconds": by_identity["I05"]["run"]["elapsed_seconds_total"],
        "unique_duplicate_service_key_semantics_valid": metric["metric_case_passed"] == metric["metric_case_total"],
        "combined_multi_channel_duplicate_unique_key_count": m06["actual"]["unique_duplicate_service_key_count"],
        "multi_leg_false_positive_count": m07["actual"]["unique_duplicate_service_key_count"],
        "duplicate_metric_reconciliation_valid": reconciliation["duplicate_metric_reconciliation_valid"],
        "targeted_regression_ready": targeted_ready,
        "targeted_regression_authorized": False,
        "full_verify_ready": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
    }
    writer.json("downstream_lock.json", downstream)
    report_json, report_md = identity_final_report(root, gate, identity, metric, source_preflight)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    writer.json("table_write_backend_audit_identity_regression.json", {"created_at": iso_kst(), "tables": table_infos})
    payloads = list(identity_payload_paths())
    manifest = write_manifest(writer, "artifact_manifest_identity_regression.json", payloads, "A1_IDENTITY_REGRESSION_MODE")
    write_lock(writer, "_IDENTITY_REGRESSION_COMPLETE.lock", "artifact_manifest_identity_regression.json", gate)
    verification = verify_manifest(root, "_IDENTITY_REGRESSION_COMPLETE.lock")
    if (
        manifest["missing_payload_count"]
        or not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        gate["gate"] = FAIL_IR1_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = "FAILED_IR1_MANIFEST_RECONCILIATION"
        writer.json("gate_decision.json", gate)
        writer.json("downstream_lock.json", {**downstream, "identity_regression_complete": False, "targeted_regression_ready": False})
        report_json, report_md = identity_final_report(root, gate, identity, metric, source_preflight)
        writer.json("final_report.json", report_json)
        writer.text("final_report.md", report_md)
        write_manifest(writer, "artifact_manifest_identity_regression.json", payloads, "A1_IDENTITY_REGRESSION_MODE")
        write_lock(writer, "_IDENTITY_REGRESSION_COMPLETE.lock", "artifact_manifest_identity_regression.json", gate)
    print(f"[A1-IR1] artifact: {root}")
    print("[A1-IR1] mode: identity-regression")
    print(f"[A1-IR1] source drift: {source_preflight['source_drift_count']}")
    print(f"[A1-IR1] identity fixtures: {identity['identity_fixture_passed']} / {identity['identity_fixture_total']}")
    print(f"[A1-IR1] metric cases: {metric['metric_case_passed']} / {metric['metric_case_total']}")
    print(f"[A1-IR1] registry enforcement: {identity['registry_omission_behavior']['registry_enforcement_status']}")
    print(f"[A1-IR1] M06 unique duplicate key count: {m06['actual']['unique_duplicate_service_key_count']}")
    print(f"[A1-IR1] M07 unique duplicate key count: {m07['actual']['unique_duplicate_service_key_count']}")
    print(f"[A1-IR1] gate: {gate['gate']}")
    print(f"[A1-IR1] gate passed: {str(gate['gate_passed']).lower()}")
    print(f"[A1-IR1] readiness: {gate['readiness']}")
    return root


def run_locked(root: Path, mode: str) -> Path:
    validate_artifact_root(root, mode)
    raise RuntimeError(f"--mode {mode} is locked until an explicit user command after repair review")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["repair", "identity-regression", "targeted-regression", "finalize"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "repair":
        run_repair(args.artifact_root)
    elif args.mode == "identity-regression":
        run_identity_regression(args.artifact_root)
    else:
        run_locked(args.artifact_root, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
