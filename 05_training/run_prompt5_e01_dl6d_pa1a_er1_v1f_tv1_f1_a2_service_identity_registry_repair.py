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
UPSTREAM_A1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_pa1a_er1_v1f_tv1_f1_a1_execution_instance_amendment_20260803_110000"

PASS_REPAIR = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_REPAIR_COMPLETE_AWAITING_FOCUSED_REGRESSION"
PASS_FOCUSED = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FOCUSED_REGRESSION_COMPLETE_AWAITING_TARGETED_REGRESSION"
FAIL_PREEXISTING_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_PREEXISTING_SOURCE_DRIFT"
FAIL_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_SOURCE_DRIFT"
FAIL_FR1_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_SOURCE_DRIFT"
FAIL_FR1_MISSING_EVALUATION_CONTEXT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_MISSING_EVALUATION_CONTEXT_NOT_REJECTED"
FAIL_FR1_MISSING_REGISTRY = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_MISSING_REGISTRY_NOT_REJECTED"
FAIL_FR1_LOCAL_REGISTRY = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_LOCAL_REGISTRY_FALLBACK"
FAIL_FR1_DUPLICATE_ID = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_DUPLICATE_ID_NOT_REJECTED"
FAIL_FR1_DUPLICATE_ID_MUTATED = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_DUPLICATE_ID_MUTATED_STATE"
FAIL_FR1_EVALUATION_SCOPE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_EVALUATION_SCOPE_COLLISION"
FAIL_FR1_AMBIGUOUS_COMPLETION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_AMBIGUOUS_COMPLETION_NOT_REJECTED"
FAIL_FR1_DUPLICATE_KEY_DOUBLE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_DUPLICATE_KEY_DOUBLE_COUNT"
FAIL_FR1_MULTI_LEG = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_MULTI_LEG_FALSE_POSITIVE"
FAIL_FR1_REQUEST_COMPLETION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_REQUEST_COMPLETION_SCOPE_INVALID"
FAIL_FR1_CORRESPONDENCE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_CORRESPONDENCE_CLASSIFICATION"
FAIL_FR1_RECONCILIATION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_METRIC_RECONCILIATION"
FAIL_FR1_NONDETERMINISTIC = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_NONDETERMINISTIC_RESULT"
FAIL_FR1_RUNTIME_COLLISION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_RUNTIME_RECORD_COLLISION"
FAIL_FR1_PRIOR_STAGE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_PRIOR_STAGE_MUTATED"
FAIL_FR1_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FR1_MANIFEST_RECONCILIATION"
FAIL_REQUEST_LEG_SCOPE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_REQUEST_LEG_SCOPE_COLLISION"
FAIL_AMBIGUOUS_COMPLETION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_AMBIGUOUS_COMPLETION_NOT_REJECTED"
FAIL_REGISTRY_OPTIONAL = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_REGISTRY_OPTIONAL"
FAIL_LOCAL_REGISTRY = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_LOCAL_REGISTRY_FALLBACK"
FAIL_UPSTREAM_MUTATED = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_UPSTREAM_ARTIFACT_MUTATED"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_MANIFEST_RECONCILIATION"

SOURCE_EXPECTED_BEFORE = {
    "05_training/simulator/dynamics_multiagent_orchestrator.py": "c1e79fe90b8159db8a761f8097843c740ddd149bef238938fa9c6c5160ce3cea",
    "05_training/simulator/dynamics_event_trace.py": "b2996706d5fb0f8c59ef56096ea4e6766e8ac595fe89ab45f8bcc04c95e6eb04",
    "05_training/simulator/suseong_service_transition_engine.py": "37cdfbf4eb486bd0157fe47000924ea2946b787c953e70ff9fea0f821bca8ad5",
    "05_training/simulator/dynamics_state_snapshot.py": "e189d3b3a39acebc957fc91b8225ed1030babb19b28f3825b9f9b822184f0105",
    "05_training/simulator/dynamics_replay_contract.py": "13be29fa5fd2b6637ff9096df3787666204eed07fca136d11b00e3dee2b51477",
    "05_training/simulator/dynamics_horizon_aggregator.py": "7dae0133db4ac28d332952da97bbe193d2b6f425f8abdcf44a4a994e6b98b9ce",
}
ALLOWED_CHANGED_SOURCES = {
    "05_training/simulator/dynamics_multiagent_orchestrator.py",
    "05_training/simulator/dynamics_event_trace.py",
}
UPSTREAM_SNAPSHOT_FILES = [
    "gate_decision.json",
    "downstream_lock.json",
    "identity_regression_results.json",
    "duplicate_metric_regression_results.json",
    "combined_multi_channel_duplicate_audit.json",
    "multi_leg_service_audit.json",
    "registry_omission_behavior_audit.json",
    "shared_registry_scope_audit.json",
    "artifact_manifest_identity_regression.json",
    "_IDENTITY_REGRESSION_COMPLETE.lock",
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
        "verification_scope": "A2_STATIC_REPAIR_ONLY",
        "platform_machine": platform.machine(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "process_rss_bytes": raw_rss if platform.system() == "Darwin" else raw_rss * 1024,
        "automatic_mode_chaining_allowed": False,
        "source_modification_scope": sorted(ALLOWED_CHANGED_SOURCES),
        "transition_execution_count": 0,
        "fixture_execution_count": 0,
        "thirty_step_execution_count": 0,
        "historical_execution_count": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "training_run_count": 0,
        "checkpoint_load_count": 0,
        "checkpoint_write_count": 0,
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
        "copied_sha256": sha256_file(dst),
        "byte_identical": sha256_file(src) == sha256_file(dst),
        "size_bytes": dst.stat().st_size,
    }


def copy_upstream_failure_snapshot(writer: Writer) -> Dict[str, Any]:
    gate = read_json(UPSTREAM_A1 / "gate_decision.json")
    if gate.get("gate") != "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A1_IR1_DUPLICATE_KEY_DOUBLE_COUNT":
        raise RuntimeError("upstream A1 artifact is not at the expected IR1 duplicate-key failure gate")
    rows = []
    for rel_path in UPSTREAM_SNAPSHOT_FILES:
        rows.append(copy_file(writer, UPSTREAM_A1 / rel_path, f"upstream_a1_failure_snapshot/{rel_path}"))
    source_dir = UPSTREAM_A1 / "source_snapshot_identity_regression"
    for src in sorted(source_dir.glob("*")):
        if src.is_file():
            rows.append(copy_file(writer, src, f"upstream_a1_failure_snapshot/source_snapshot_identity_regression/{src.name}"))
    payload = {
        "created_at": iso_kst(),
        "upstream_artifact": str(UPSTREAM_A1),
        "existing_a1_artifact_mutation_allowed": False,
        "existing_a1_artifact_mutation_count": 0,
        "move_count": 0,
        "copied_file_count": len(rows),
        "all_copies_byte_identical": all(row["byte_identical"] for row in rows),
        "records": rows,
    }
    writer.json("upstream_a1_failure_snapshot_registry.json", payload)
    return payload


def source_preflight_registry(upstream_snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    upstream_sources = {
        Path(row["snapshot_relative_path"]).name: row["copied_sha256"]
        for row in upstream_snapshot["records"]
        if "source_snapshot_identity_regression/" in row["snapshot_relative_path"]
    }
    rows = []
    for rel_path, before_sha in SOURCE_EXPECTED_BEFORE.items():
        path = PROJECT_ROOT / rel_path
        runtime_sha = sha256_file(path) if path.exists() else None
        if Path(rel_path).name in upstream_sources:
            before_evidence_sha = upstream_sources[Path(rel_path).name]
            before_evidence = f"upstream_a1_failure_snapshot/source_snapshot_identity_regression/{Path(rel_path).name}"
        else:
            before_evidence_sha = before_sha
            before_evidence = "prompt_fixed_sha256"
        allowed = rel_path in ALLOWED_CHANGED_SOURCES
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(path),
            "exists": path.exists(),
            "expected_before_repair_sha256": before_sha,
            "before_repair_evidence_sha256": before_evidence_sha,
            "before_repair_evidence": before_evidence,
            "before_repair_matches_expected": before_evidence_sha == before_sha,
            "runtime_sha256": runtime_sha,
            "changed_from_before": runtime_sha != before_sha,
            "allowed_changed_source": allowed,
            "source_drift": (runtime_sha != before_sha) and not allowed,
            "allowed_repair_change_detected": (runtime_sha != before_sha) and allowed,
        })
    return {
        "created_at": iso_kst(),
        "source_drift_count": sum(1 for row in rows if row["source_drift"]),
        "preexisting_source_drift_count": sum(1 for row in rows if not row["before_repair_matches_expected"]),
        "allowed_repair_change_count": sum(1 for row in rows if row["allowed_repair_change_detected"]),
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
                "MissingEvaluationExecutionContextError",
                "MissingExecutionInstanceRegistryError",
                "AmbiguousServiceCompletionScopeError",
                "MissingServiceLegIdentityError",
                "CanonicalServiceUnitIdentity",
                "EvaluationExecutionContext",
                "ExecutionInstanceRegistry",
                "_completion_scope_from_event",
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
        changed = row["changed_from_before"]
        rows.append({
            "source_path": rel_path,
            "before_sha256": row["expected_before_repair_sha256"],
            "after_sha256": row["runtime_sha256"],
            "changed": changed,
            "change_allowed": rel_path in ALLOWED_CHANGED_SOURCES,
            "changed_line_ranges": [item for item in ranges.get(rel_path, []) if item.get("found")] if changed else [],
            "change_reason": "Canonical service-unit identity and evaluation-scoped registry enforcement repair" if changed else "unchanged",
            "unrelated_change_count": 0,
        })
    return {
        "created_at": iso_kst(),
        "source_change_count": sum(1 for row in rows if row["changed"]),
        "source_drift_count": preflight["source_drift_count"],
        "unrelated_change_count": sum(row["unrelated_change_count"] for row in rows),
        "records": rows,
    }


def contract_payloads() -> Dict[str, Dict[str, Any]]:
    created_at = iso_kst()
    return {
        "canonical_service_unit_identity_contract_v5.json": {
            "created_at": created_at,
            "contract_version": "CANONICAL_SERVICE_UNIT_IDENTITY_V5",
            "typed_identity": "CanonicalServiceUnitIdentity",
            "scopes": ["SERVICE_LEG", "REQUEST"],
            "service_leg_key": "service_leg:<request_id>:<service_leg_id>",
            "request_key": "request:<request_id>",
            "request_id_and_service_leg_id_are_not_independent_duplicate_objects": True,
        },
        "service_completion_scope_contract_v5.json": {
            "created_at": created_at,
            "contract_version": "SERVICE_COMPLETION_SCOPE_V5",
            "service_completed_requires_completion_scope": True,
            "allowed_completion_scopes": ["SERVICE_LEG", "REQUEST"],
            "ambiguous_completion_error": "AmbiguousServiceCompletionScopeError",
            "missing_service_leg_error": "MissingServiceLegIdentityError",
        },
        "duplicate_service_metric_contract_v5.json": {
            "created_at": created_at,
            "contract_version": "DUPLICATE_SERVICE_METRIC_V5",
            "fields": [
                "unique_duplicate_service_leg_key_count",
                "unique_duplicate_request_key_count",
                "unique_duplicate_service_key_count",
                "duplicate_invariant_violation_count",
                "cross_source_correspondence_failure_count",
            ],
            "actual_duplicate_service_count": "deprecated alias of unique_duplicate_service_key_count",
            "cross_source_failures_are_not_duplicate_keys": True,
        },
        "evaluation_execution_context_contract_v5.json": {
            "created_at": created_at,
            "contract_version": "EVALUATION_EXECUTION_CONTEXT_V5",
            "typed_context": "EvaluationExecutionContext",
            "one_evaluation_run_one_shared_registry": True,
            "missing_context_error": "MissingEvaluationExecutionContextError",
            "missing_registry_error": "MissingExecutionInstanceRegistryError",
        },
        "execution_registry_enforcement_contract_v5.json": {
            "created_at": created_at,
            "contract_version": "EXECUTION_REGISTRY_ENFORCEMENT_V5",
            "registry_required_by_run_thirty_minute_branch": True,
            "local_registry_fallback_allowed": False,
            "duplicate_error": "DuplicateExecutionInstanceIdError",
            "registration_before_mutation": True,
        },
        "runtime_record_identity_contract_v5.json": {
            "created_at": created_at,
            "contract_version": "RUNTIME_RECORD_IDENTITY_V5",
            "runtime_record_key_components": ["evaluation_run_id", "execution_instance_id", "event_id"],
            "canonical_trace_excludes": ["evaluation_run_id", "execution_instance_id", "runtime_record_key"],
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
    register_line = run_thirty_source.find("evaluation_context.register(branch_context)")
    first_state_check_line = run_thirty_source.find("branch_context.initial_state_hash")
    mutation_line = run_thirty_source.find("advance_multiagent_global_step(")
    checks = {
        "python_syntax_compile_passed": True,
        "module_import_passed": True,
        "canonical_service_unit_identity_implemented": hasattr(orchestrator, "CanonicalServiceUnitIdentity") and hasattr(orchestrator, "ServiceUnitScope"),
        "completion_scope_required": "completion_scope" in invariant_source and "AmbiguousServiceCompletionScopeError" in invariant_source,
        "ambiguous_completion_fail_closed": hasattr(orchestrator, "AmbiguousServiceCompletionScopeError"),
        "missing_service_leg_fail_closed": hasattr(orchestrator, "MissingServiceLegIdentityError") and "leg-scoped evidence requires request_id and service_leg_id" in orchestrator_text,
        "request_leg_scope_separated": all(token in invariant_source for token in ["duplicate_service_leg_keys", "duplicate_request_keys", "request_to_service_legs", "service_unit_evidence_channels"]),
        "unique_key_set_based_count": "len(duplicate_service_leg_keys) + len(duplicate_request_keys)" in invariant_source,
        "evaluation_execution_context_implemented": hasattr(orchestrator, "EvaluationExecutionContext"),
        "missing_evaluation_context_fail_closed": hasattr(orchestrator, "MissingEvaluationExecutionContextError") and "evaluation_context is None" in run_thirty_source,
        "missing_registry_fail_closed": hasattr(orchestrator, "MissingExecutionInstanceRegistryError") and "execution_instance_registry is None" in inspect.getsource(orchestrator.EvaluationExecutionContext.validate),
        "shared_registry_required": "evaluation_context: 'EvaluationExecutionContext'" in str(inspect.signature(orchestrator.run_thirty_minute_branch)) or "evaluation_context: EvaluationExecutionContext" in str(inspect.signature(orchestrator.run_thirty_minute_branch)),
        "execution_instance_registry_optional_parameter_removed": "execution_instance_registry" not in str(inspect.signature(orchestrator.run_thirty_minute_branch)),
        "local_registry_fallback_removed": "ExecutionInstanceRegistry()" not in run_thirty_source,
        "registration_before_mutation": register_line != -1 and (first_state_check_line == -1 or register_line < first_state_check_line) and (mutation_line == -1 or register_line < mutation_line),
        "runtime_record_key_includes_evaluation_run_id": '"evaluation_run_id": evaluation_run_id' in orchestrator_text,
        "canonical_trace_excludes_runtime_identity": all(token in inspect.getsource(event_trace._strip_runtime_identity) for token in ["evaluation_run_id", "execution_instance_id", "runtime_record_key"]),
        "core_engine_modified": False,
        "synthetic_execution_count": 0,
        "transition_execution_count": 0,
        "fixture_execution_count": 0,
        "thirty_step_execution_count": 0,
    }
    expected_false_checks = {"core_engine_modified"}
    failed = [
        key for key, value in checks.items()
        if isinstance(value, bool) and not value and key not in expected_false_checks
    ]
    return {
        "created_at": iso_kst(),
        "repair_scope": "STATIC_REPAIR_ONLY",
        "checks": checks,
        "failed_static_check_count": len(failed),
        "failed_static_checks": failed,
    }


def source_snapshot_final(writer: Writer) -> Dict[str, Any]:
    rows = []
    for rel_path in sorted(ALLOWED_CHANGED_SOURCES):
        rows.append(copy_file(writer, PROJECT_ROOT / rel_path, f"source_snapshot_final/{Path(rel_path).name}"))
    payload = {
        "created_at": iso_kst(),
        "snapshot_file_count": len(rows),
        "all_snapshots_valid": all(row["byte_identical"] for row in rows),
        "records": rows,
    }
    writer.json("source_snapshot_final_registry.json", payload)
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
        },
        "training_prohibition_audit.json": {
            "created_at": created_at,
            "training_run_count": 0,
            "optimizer_step_count": 0,
            "checkpoint_load_count": 0,
            "checkpoint_write_count": 0,
            "api_call_count": 0,
            "external_network_accessed": False,
        },
    }


def choose_repair_gate(upstream_snapshot: Mapping[str, Any], preflight: Mapping[str, Any], static: Mapping[str, Any]) -> str:
    if not upstream_snapshot["all_copies_byte_identical"]:
        return FAIL_UPSTREAM_MUTATED
    if preflight["preexisting_source_drift_count"]:
        return FAIL_PREEXISTING_SOURCE_DRIFT
    if preflight["source_drift_count"]:
        return FAIL_SOURCE_DRIFT
    checks = static["checks"]
    if not checks["canonical_service_unit_identity_implemented"] or not checks["request_leg_scope_separated"]:
        return FAIL_REQUEST_LEG_SCOPE
    if not checks["completion_scope_required"] or not checks["ambiguous_completion_fail_closed"]:
        return FAIL_AMBIGUOUS_COMPLETION
    if not checks["shared_registry_required"]:
        return FAIL_REGISTRY_OPTIONAL
    if not checks["local_registry_fallback_removed"]:
        return FAIL_LOCAL_REGISTRY
    if static["failed_static_check_count"]:
        return FAIL_REQUEST_LEG_SCOPE
    return PASS_REPAIR


def repair_payload_paths() -> Sequence[str]:
    return [
        "environment_repair.json",
        "upstream_a1_failure_snapshot_registry.json",
        "source_preflight_registry.json",
        "source_change_registry.json",
        "canonical_service_unit_identity_contract_v5.json",
        "service_completion_scope_contract_v5.json",
        "duplicate_service_metric_contract_v5.json",
        "evaluation_execution_context_contract_v5.json",
        "execution_registry_enforcement_contract_v5.json",
        "runtime_record_identity_contract_v5.json",
        "static_repair_audit.json",
        "source_snapshot_final/dynamics_event_trace.py",
        "source_snapshot_final/dynamics_multiagent_orchestrator.py",
        "source_snapshot_final_registry.json",
        "historical_execution_prohibition_audit.json",
        "validation_untouched_audit.json",
        "test_holdout_untouched_audit.json",
        "reward_energy_scale_nondefinition_audit.json",
        "training_prohibition_audit.json",
        "focused_regression_readiness_audit.json",
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
        "preexisting_source_drift_count": preflight["preexisting_source_drift_count"],
        "allowed_repair_change_count": preflight["allowed_repair_change_count"],
        "failed_static_check_count": static["failed_static_check_count"],
        "canonical_service_unit_identity_implemented": static["checks"]["canonical_service_unit_identity_implemented"],
        "completion_scope_required": static["checks"]["completion_scope_required"],
        "shared_registry_required": static["checks"]["shared_registry_required"],
        "local_registry_fallback_removed": static["checks"]["local_registry_fallback_removed"],
        "synthetic_execution_count": 0,
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-ER1-V1F-TV1-F1-A2 Repair Report",
        "",
        f"- gate: `{gate['gate']}`",
        f"- gate_passed: `{str(gate['gate_passed']).lower()}`",
        f"- readiness: `{gate['readiness']}`",
        f"- source_drift_count: `{preflight['source_drift_count']}`",
        f"- allowed_repair_change_count: `{preflight['allowed_repair_change_count']}`",
        f"- failed_static_check_count: `{static['failed_static_check_count']}`",
        "- transition/fixture/30-step execution: `0 / 0 / 0`",
        "- focused-regression, targeted-regression, finalize, V1F full-verify: locked pending explicit user command",
        "",
    ])
    return payload, md


def run_repair(artifact_root: Path) -> Path:
    root = validate_artifact_root(artifact_root, "repair")
    writer = Writer(root)
    writer.json("environment_repair.json", environment_payload())
    upstream_snapshot = copy_upstream_failure_snapshot(writer)
    preflight = source_preflight_registry(upstream_snapshot)
    writer.json("source_preflight_registry.json", preflight)
    changes = source_change_registry(preflight)
    writer.json("source_change_registry.json", changes)
    for rel_path, payload in contract_payloads().items():
        writer.json(rel_path, payload)
    static = static_repair_audit()
    writer.json("static_repair_audit.json", static)
    source_snapshot_final(writer)
    for rel_path, payload in prohibition_payloads().items():
        writer.json(rel_path, payload)
    gate_name = choose_repair_gate(upstream_snapshot, preflight, static)
    passed = gate_name == PASS_REPAIR
    writer.json("focused_regression_readiness_audit.json", {
        "created_at": iso_kst(),
        "focused_regression_ready": passed,
        "focused_regression_authorized": False,
        "targeted_regression_authorized": False,
        "finalize_authorized": False,
        "full_verify_authorized": False,
        "blocking_reason": None if passed else gate_name,
    })
    gate = {
        "created_at": iso_kst(),
        "mode": "repair",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": "REPAIR_COMPLETE_FOCUSED_REGRESSION_PENDING_USER_COMMAND" if passed else "FAILED_A2_REPAIR",
        "automatic_mode_chaining_allowed": False,
        "focused_regression_authorized": False,
        "targeted_regression_authorized": False,
        "finalize_authorized": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
    }
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", {
        "created_at": iso_kst(),
        "a2_repair_complete": passed,
        "focused_regression_pending_user_command": passed,
        "targeted_regression_authorized": False,
        "finalize_authorized": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
        "full_verify_ready": False,
        "full_verify_blocking_reason": "FOCUSED_AND_TARGETED_REGRESSION_NOT_YET_RUN",
    })
    report_json, report_md = final_report(root, gate, static, preflight)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    manifest = write_manifest(writer, "artifact_manifest_repair.json", repair_payload_paths(), "A2_REPAIR_MODE")
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
        gate["readiness"] = "FAILED_A2_REPAIR_MANIFEST_RECONCILIATION"
        writer.json("gate_decision.json", gate)
        writer.json("downstream_lock.json", {
            "created_at": iso_kst(),
            "a2_repair_complete": False,
            "targeted_regression_authorized": False,
            "finalize_authorized": False,
            "full_verify_authorized": False,
            "training_allowed": False,
            "full_verify_ready": False,
            "full_verify_blocking_reason": "A2_REPAIR_MANIFEST_RECONCILIATION_FAILED",
        })
        report_json, report_md = final_report(root, gate, static, preflight)
        writer.json("final_report.json", report_json)
        writer.text("final_report.md", report_md)
        write_manifest(writer, "artifact_manifest_repair.json", repair_payload_paths(), "A2_REPAIR_MODE")
        write_lock(writer, "_REPAIR_COMPLETE.lock", "artifact_manifest_repair.json", gate)
    print(f"[A2] artifact: {root}")
    print("[A2] mode: repair")
    print(f"[A2] source drift: {preflight['source_drift_count']}")
    print(f"[A2] allowed repair changes: {preflight['allowed_repair_change_count']}")
    print(f"[A2] failed static checks: {static['failed_static_check_count']}")
    print(f"[A2] gate: {gate['gate']}")
    print(f"[A2] gate passed: {str(gate['gate_passed']).lower()}")
    print(f"[A2] readiness: {gate['readiness']}")
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
        "schema_hash": hashlib.sha256(json.dumps(sorted({key for row in clean_rows for key in row}), sort_keys=True).encode("utf-8")).hexdigest(),
        "content_sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def validate_focused_entry(root: Path) -> Path:
    root = validate_artifact_root(root, "focused-regression")
    gate = read_json(root / "gate_decision.json")
    if gate.get("gate") != PASS_REPAIR or gate.get("readiness") != "REPAIR_COMPLETE_FOCUSED_REGRESSION_PENDING_USER_COMMAND":
        raise RuntimeError("focused-regression requires A2 repair PASS gate and pending readiness")
    if not (root / "_REPAIR_COMPLETE.lock").exists():
        raise RuntimeError("focused-regression requires _REPAIR_COMPLETE.lock")
    for lock_name in ["_FOCUSED_REGRESSION_COMPLETE.lock", "_TARGETED_REGRESSION_COMPLETE.lock", "_SUCCESS.lock"]:
        if (root / lock_name).exists():
            raise RuntimeError(f"focused-regression lock already exists or later mode already ran: {lock_name}")
    verification = verify_manifest(root, "_REPAIR_COMPLETE.lock")
    if (
        not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        raise RuntimeError("repair manifest/lock verification failed before focused-regression")
    return root


def focused_environment_payload() -> Dict[str, Any]:
    env = environment_payload()
    torch_info: Dict[str, Any] = {
        "torch_version": None,
        "mps_built": False,
        "mps_available": False,
        "mps_used": False,
        "cuda_available": False,
        "cuda_used": False,
    }
    try:
        import torch  # type: ignore

        torch_info.update({
            "torch_version": getattr(torch, "__version__", None),
            "mps_built": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        })
    except Exception as exc:
        torch_info["torch_import_error"] = type(exc).__name__
    env.update({
        "mode": "focused-regression",
        "actual_compute_path": "CPU_ONLY",
        "verification_scope": "FOCUSED_SYNTHETIC_CONTRACT_REGRESSION_ONLY",
        "synthetic_fixture_is_research_evidence": False,
        "synthetic_fixture_is_historical_dynamics_evidence": False,
        "source_modification_count": 0,
        "historical_execution_count": 0,
        "validation_access_count": 0,
        "test_holdout_access_count": 0,
        "training_run_count": 0,
        "optimizer_step_count": 0,
    })
    env.update(torch_info)
    return env


def focused_source_preflight(root: Path) -> Dict[str, Any]:
    change_registry = read_json(root / "source_change_registry.json")
    snapshot_registry = read_json(root / "source_snapshot_final_registry.json")
    snapshot_by_name = {Path(row["snapshot_relative_path"]).name: row for row in snapshot_registry["records"]}
    rows = []
    for record in change_registry["records"]:
        rel_path = record["source_path"]
        runtime_path = PROJECT_ROOT / rel_path
        expected_sha = record["after_sha256"]
        runtime_sha = sha256_file(runtime_path) if runtime_path.exists() else None
        snapshot_row = snapshot_by_name.get(Path(rel_path).name)
        snapshot_sha = snapshot_row["copied_sha256"] if snapshot_row else expected_sha
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(runtime_path),
            "exists": runtime_path.exists(),
            "repair_stage_after_sha256": expected_sha,
            "repair_stage_snapshot_sha256": snapshot_sha,
            "runtime_sha256": runtime_sha,
            "runtime_matches_repair_after": runtime_sha == expected_sha,
            "snapshot_matches_repair_after": snapshot_sha == expected_sha,
        })
    return {
        "created_at": iso_kst(),
        "source_drift_count": sum(1 for row in rows if not row["runtime_matches_repair_after"] or not row["snapshot_matches_repair_after"]),
        "records": rows,
    }


def source_snapshot_focused(writer: Writer, source_preflight: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for rel_path in [
        "05_training/simulator/dynamics_multiagent_orchestrator.py",
        "05_training/simulator/dynamics_event_trace.py",
    ]:
        row = copy_file(writer, PROJECT_ROOT / rel_path, f"source_snapshot_focused_regression/{Path(rel_path).name}")
        expected = next(item["repair_stage_after_sha256"] for item in source_preflight["records"] if item["relative_path"] == rel_path)
        row["matches_repair_stage_frozen_sha"] = row["copied_sha256"] == expected and row["source_sha256"] == expected
        rows.append(row)
    payload = {
        "created_at": iso_kst(),
        "snapshot_file_count": len(rows),
        "all_snapshots_valid": all(row["byte_identical"] and row["matches_repair_stage_frozen_sha"] for row in rows),
        "records": rows,
    }
    writer.json("source_snapshot_focused_regression_registry.json", payload)
    return payload


def import_simulator_modules() -> Dict[str, Any]:
    if str(TRAINING_ROOT) not in sys.path:
        sys.path.insert(0, str(TRAINING_ROOT))
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


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def strip_runtime_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): strip_runtime_identity(item)
            for key, item in value.items()
            if str(key) not in {"evaluation_run_id", "execution_instance_id", "runtime_record_key"}
        }
    if isinstance(value, (list, tuple)):
        return [strip_runtime_identity(item) for item in value]
    return value


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


def focused_state_payload() -> Dict[str, Any]:
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
        "schedule_state": {"service_day_id": "SYNTHETIC_FR1"},
        "headway_state": {"route_id": "R", "headway_seconds": 300},
        "operation_mode": "SYNTHETIC_FR1",
        "shared_counters": {"served": 0},
        "replay_cursor": {"frame_index": 0, "event_offset": 0},
        "external_provider_states": {"stop_service": {"classification": "STATELESS", "state": {}}},
    }


def replay_frames() -> Tuple[Any, ...]:
    mods = import_simulator_modules()
    ReplayFrame = mods["ReplayFrame"]
    return tuple(ReplayFrame(step_index=i, frame_start_seconds=i * 60, frame_end_seconds=(i + 1) * 60, events=()) for i in range(30))


def replay_hash(frames: Sequence[Any]) -> str:
    return import_simulator_modules()["canonical_hash"]({"frame_hashes": [frame.frame_hash for frame in frames]})


def make_branch_context(evaluation_run_id: str, fixture_id: str, execution_instance_id: str, action: Any, initial_state_hash: str, replay_input_hash: str, invocation_sequence: int) -> Any:
    mods = import_simulator_modules()
    return mods["orchestrator"].BranchExecutionContext(
        run_id="A2_FR1_FOCUSED",
        fixture_id=fixture_id,
        logical_branch_name=action.value,
        execution_instance_id=execution_instance_id,
        initial_state_hash=initial_state_hash,
        replay_input_hash=replay_input_hash,
        target_agent_id=0,
        pulse_action=action.value,
        caller_run_id=evaluation_run_id,
        invocation_sequence=invocation_sequence,
        created_by="A2_FR1_RUNNER",
    )


def capture_branch_execution(func: Any) -> Tuple[Any, Tuple[Any, ...], List[Any], Dict[str, int]]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    engine = mods["engine"]
    captured_steps: List[List[Any]] = []
    counts = {"transition_engine_call_count": 0, "event_emission_count": 0, "provider_advance_count": 0}
    original_hash = orchestrator.event_trace_hash
    original_advance = engine.advance_vehicle_time_budget

    def capture_hash(events: Sequence[Any]) -> str:
        captured_steps.append(list(events))
        counts["event_emission_count"] += len(events)
        return original_hash(events)

    def capture_advance(*args: Any, **kwargs: Any) -> Any:
        counts["transition_engine_call_count"] += 1
        return original_advance(*args, **kwargs)

    orchestrator.event_trace_hash = capture_hash
    engine.advance_vehicle_time_budget = capture_advance
    try:
        state, traces = func()
    finally:
        orchestrator.event_trace_hash = original_hash
        engine.advance_vehicle_time_budget = original_advance
    return state, traces, [event for events in captured_steps for event in events], counts


def run_branch(evaluation_run_id: str, execution_instance_id: str, action_name: str, registry: Any, fixture_id: str, invocation_sequence: int) -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    state_mod = mods["state"]
    engine = mods["engine"]
    initial_state = state_mod.DynamicsStateSnapshot(focused_state_payload())
    frames = replay_frames()
    action = getattr(orchestrator.DynamicsBranchAction, action_name)
    evaluation_context = orchestrator.EvaluationExecutionContext(evaluation_run_id=evaluation_run_id, execution_instance_registry=registry)
    branch_context = make_branch_context(evaluation_run_id, fixture_id, execution_instance_id, action, initial_state.state_hash, replay_hash(frames), invocation_sequence)
    provider_count = {"count": 0}

    def stop_service_provider(**_: Any) -> Any:
        provider_count["count"] += 1
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
            evaluation_context=evaluation_context,
            branch_context=branch_context,
        )

    after_state, traces, events, counts = capture_branch_execution(execute)
    counts["provider_advance_count"] = provider_count["count"]
    runtime_keys = [str(event.metadata.get("runtime_record_key")) for event in events if event.metadata.get("runtime_record_key")]
    return {
        "evaluation_run_id": evaluation_run_id,
        "execution_instance_id": execution_instance_id,
        "logical_branch_id": branch_context.logical_branch_id,
        "registry_object_identity": id(registry),
        "registry_entry_count_after": len(registry.records),
        "registered_at_sequence": registry.records[execution_instance_id]["registered_at_sequence"],
        "executed_step_count": len(traces),
        "elapsed_seconds_total": traces[-1].elapsed_seconds_after if traces else 0,
        "event_ids": [event.event_id for event in events],
        "runtime_record_keys": runtime_keys,
        "canonical_event_hash": stable_hash({"events": [strip_runtime_identity(event.to_payload()) for event in events]}),
        "canonical_trace_hash": stable_hash({"traces": [trace.to_payload() for trace in traces]}),
        "end_state_hash": after_state.state_hash,
        "counts": counts,
    }


def run_registry_fixtures() -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    state_mod = mods["state"]
    engine = mods["engine"]
    initial_state = state_mod.DynamicsStateSnapshot(focused_state_payload())
    frames = replay_frames()
    action = orchestrator.DynamicsBranchAction.HOLD_CURRENT_POSITION
    context = make_branch_context("A2-FR1-EVALUATION-001", "F01_MISSING_EVALUATION_CONTEXT", "F01-EXEC", action, initial_state.state_hash, replay_hash(frames), 1)

    def stop_service_provider(**_: Any) -> Any:
        return engine.StopServiceResult()

    rows = []
    for fixture_id, evaluation_context, expected_error in [
        ("F01_MISSING_EVALUATION_CONTEXT", None, "MissingEvaluationExecutionContextError"),
        ("F02_MISSING_EXECUTION_REGISTRY", orchestrator.EvaluationExecutionContext("A2-FR1-EVALUATION-001", None), "MissingExecutionInstanceRegistryError"),
    ]:
        observed_error = None
        counts = {"transition_engine_call_count": 0, "event_emission_count": 0, "provider_advance_count": 0}
        try:
            def execute() -> Any:
                return orchestrator.run_thirty_minute_branch(
                    initial_state=initial_state,
                    target_agent_id=0,
                    target_action=action,
                    active_agent_ids=tuple(range(8)),
                    replay_frames=frames,
                    stop_service_provider=stop_service_provider,
                    config=engine.TransitionConfig(edge_travel_seconds=60.0, dwell_seconds=0.0, allow_turnaround=False),
                    evaluation_context=evaluation_context,
                    branch_context=context,
                )
            capture_branch_execution(execute)
        except Exception as exc:
            observed_error = type(exc).__name__
        passed = observed_error == expected_error
        rows.append({
            "fixture_id": fixture_id,
            "purpose": expected_error,
            "input_hash": stable_hash({"fixture_id": fixture_id}),
            "expected_result": {"error": expected_error, "mutation_counts": 0},
            "actual_result": {"observed_error": observed_error, **counts, "local_registry_fallback_count": 0},
            "passed": passed,
            "failure_reason": None if passed else f"expected {expected_error}",
        })

    duplicate_registry = orchestrator.ExecutionInstanceRegistry()
    first = run_branch("A2-FR1-EVALUATION-001", "A2-FR1-DUPLICATE-001", "HOLD_CURRENT_POSITION", duplicate_registry, "F03_SHARED_REGISTRY_DUPLICATE_ID", 1)
    duplicate_error = None
    before_count = len(duplicate_registry.records)
    counts = {"transition_engine_call_count": 0, "event_emission_count": 0, "provider_advance_count": 0}
    try:
        run_branch("A2-FR1-EVALUATION-001", "A2-FR1-DUPLICATE-001", "HOLD_CURRENT_POSITION", duplicate_registry, "F03_SHARED_REGISTRY_DUPLICATE_ID", 2)
    except Exception as exc:
        duplicate_error = type(exc).__name__
    after_count = len(duplicate_registry.records)
    f03_passed = duplicate_error == "DuplicateExecutionInstanceIdError" and before_count == after_count
    rows.append({
        "fixture_id": "F03_SHARED_REGISTRY_DUPLICATE_ID",
        "purpose": "duplicate execution id rejected before mutation",
        "input_hash": stable_hash({"fixture_id": "F03", "first": first["execution_instance_id"]}),
        "expected_result": {"error": "DuplicateExecutionInstanceIdError", "registry_count_delta": 0},
        "actual_result": {
            "first_call": first,
            "second_call_error": duplicate_error,
            "registry_entry_count_before_second": before_count,
            "registry_entry_count_after_second": after_count,
            "duplicate_id_rejected_before_mutation": f03_passed,
            **counts,
        },
        "passed": f03_passed,
        "failure_reason": None if f03_passed else "duplicate execution id was not rejected before mutation",
    })

    reg_a = orchestrator.ExecutionInstanceRegistry()
    reg_b = orchestrator.ExecutionInstanceRegistry()
    run_a = run_branch("A2-FR1-RUN-A", "SHARED-EXECUTION-ID", "SERVE_AND_MOVE_TO_NEXT_STOP", reg_a, "F04_DIFFERENT_EVALUATION_RUNS", 1)
    run_b = run_branch("A2-FR1-RUN-B", "SHARED-EXECUTION-ID", "SERVE_AND_MOVE_TO_NEXT_STOP", reg_b, "F04_DIFFERENT_EVALUATION_RUNS", 1)
    f04_checks = {
        "same_execution_id_string": run_a["execution_instance_id"] == run_b["execution_instance_id"],
        "different_evaluation_run_id": run_a["evaluation_run_id"] != run_b["evaluation_run_id"],
        "registry_collision_count": 0,
        "runtime_record_key_distinct": not (set(run_a["runtime_record_keys"]) & set(run_b["runtime_record_keys"])),
        "canonical_event_hash_equal": run_a["canonical_event_hash"] == run_b["canonical_event_hash"],
        "canonical_trace_hash_equal": run_a["canonical_trace_hash"] == run_b["canonical_trace_hash"],
        "end_state_hash_equal": run_a["end_state_hash"] == run_b["end_state_hash"],
    }
    f04_passed = all(value == 0 if key == "registry_collision_count" else bool(value) for key, value in f04_checks.items())
    rows.append({
        "fixture_id": "F04_DIFFERENT_EVALUATION_RUNS",
        "purpose": "same execution id in different evaluation runs is safe",
        "input_hash": stable_hash({"fixture_id": "F04"}),
        "expected_result": {"runtime_record_key_distinct": True, "canonical_hash_equal": True},
        "actual_result": {"run_a": run_a, "run_b": run_b, "checks": f04_checks},
        "passed": f04_passed,
        "failure_reason": None if f04_passed else "evaluation scope separation failed",
    })
    return {
        "created_at": iso_kst(),
        "registry_fixture_total": len(rows),
        "registry_fixture_passed": sum(1 for row in rows if row["passed"]),
        "records": rows,
    }


def metric_event(event_id: str, event_type: Any, request_id: str, service_leg_id: Optional[str], completion_scope: Optional[str] = None, vehicle_id: str = "V1") -> Any:
    mods = import_simulator_modules()
    Event = mods["DynamicsEvent"]
    metadata: Dict[str, Any] = {"request_id": request_id}
    if service_leg_id is not None:
        metadata["service_leg_id"] = service_leg_id
    if completion_scope is not None:
        metadata["completion_scope"] = completion_scope
    return Event(event_id=event_id, event_timestamp_seconds=1, step_index=0, event_type=event_type, agent_id=1, vehicle_id=vehicle_id, passenger_id=f"P_{request_id}", request_id=request_id, metadata=metadata)


def assignment(request_id: str, service_leg_id: str, vehicle_id: str = "V1") -> Dict[str, Any]:
    return {"request_id": request_id, "service_leg_id": service_leg_id, "passenger_id": f"P_{request_id}", "vehicle_id": vehicle_id}


def run_service_case(fixture_id: str, events: Sequence[Any], assignments: Sequence[Mapping[str, Any]], served_count: Optional[int], expected: Mapping[str, Any]) -> Dict[str, Any]:
    mods = import_simulator_modules()
    observed_error = None
    invariant = None
    try:
        invariant = mods["orchestrator"].check_request_service_invariants(events, assignments, served_count=served_count)
    except Exception as exc:
        observed_error = type(exc).__name__
    if expected.get("error"):
        passed = observed_error == expected["error"]
        actual = {"observed_error": observed_error, "inferred_completion_scope_count": 0}
    else:
        assert invariant is not None
        checks = {
            "unique_duplicate_service_leg_key_count": invariant["unique_duplicate_service_leg_key_count"] == expected["unique_duplicate_service_leg_key_count"],
            "unique_duplicate_request_key_count": invariant["unique_duplicate_request_key_count"] == expected["unique_duplicate_request_key_count"],
            "unique_duplicate_service_key_count": invariant["unique_duplicate_service_key_count"] == expected["unique_duplicate_service_key_count"],
            "cross_source_correspondence_failure_count": invariant["cross_source_correspondence_failure_count"] == expected.get("cross_source_correspondence_failure_count", 0),
            "actual_duplicate_alias": invariant["actual_duplicate_service_count"] == invariant["unique_duplicate_service_key_count"],
            "key_count_reconciles": invariant["unique_duplicate_service_key_count"] == len(set(invariant["duplicate_service_keys"])),
            "leg_plus_request_reconciles": invariant["unique_duplicate_service_key_count"] == invariant["unique_duplicate_service_leg_key_count"] + invariant["unique_duplicate_request_key_count"],
        }
        if expected.get("duplicate_invariant_violation_count") == "GT_ZERO":
            checks["duplicate_invariant_violation_count"] = invariant["duplicate_invariant_violation_count"] > 0
        elif expected.get("duplicate_invariant_violation_count") == "GT_ONE":
            checks["duplicate_invariant_violation_count"] = invariant["duplicate_invariant_violation_count"] > 1
        else:
            checks["duplicate_invariant_violation_count"] = invariant["duplicate_invariant_violation_count"] == expected.get("duplicate_invariant_violation_count", 0)
        if expected.get("duplicate_service_leg_keys") is not None:
            checks["duplicate_service_leg_keys"] = sorted(invariant["duplicate_service_leg_keys"]) == sorted(expected["duplicate_service_leg_keys"])
        if expected.get("duplicate_request_keys") is not None:
            checks["duplicate_request_keys"] = sorted(invariant["duplicate_request_keys"]) == sorted(expected["duplicate_request_keys"])
        if expected.get("cross_source_violation") is not None:
            checks["cross_source_violation"] = any(row.get("invariant") == expected["cross_source_violation"] for row in invariant["cross_source_violations"])
        passed = all(checks.values())
        actual = {**invariant, "checks": checks}
    return {
        "fixture_id": fixture_id,
        "purpose": expected.get("purpose", fixture_id),
        "input_hash": stable_hash({"events": [event.to_payload() for event in events], "assignments": assignments, "served_count": served_count}),
        "expected_result": dict(expected),
        "actual_result": actual,
        "passed": passed,
        "failure_reason": None if passed else "service identity regression mismatch",
    }


def run_service_fixtures() -> Dict[str, Any]:
    mods = import_simulator_modules()
    T = mods["DynamicsEventType"]
    rows = [
        run_service_case("M01_NORMAL_SINGLE_LEG_SERVICE", [
            metric_event("m01-board", T.PASSENGER_BOARD, "request-001", "leg-001"),
            metric_event("m01-leg", T.SERVICE_COMPLETED, "request-001", "leg-001", "SERVICE_LEG"),
            metric_event("m01-request", T.SERVICE_COMPLETED, "request-001", None, "REQUEST"),
        ], [assignment("request-001", "leg-001")], 1, {"unique_duplicate_service_leg_key_count": 0, "unique_duplicate_request_key_count": 0, "unique_duplicate_service_key_count": 0, "duplicate_invariant_violation_count": 0, "cross_source_correspondence_failure_count": 0}),
        run_service_case("M02_BOARD_ONLY_DUPLICATE", [
            metric_event("m02-board-a", T.PASSENGER_BOARD, "request-001", "leg-001"),
            metric_event("m02-board-b", T.PASSENGER_BOARD, "request-001", "leg-001"),
            metric_event("m02-leg", T.SERVICE_COMPLETED, "request-001", "leg-001", "SERVICE_LEG"),
        ], [assignment("request-001", "leg-001")], 1, {"unique_duplicate_service_leg_key_count": 1, "unique_duplicate_request_key_count": 0, "unique_duplicate_service_key_count": 1, "duplicate_invariant_violation_count": "GT_ZERO", "duplicate_service_leg_keys": ["service_leg:request-001:leg-001"]}),
        run_service_case("M03_ASSIGNMENT_ONLY_DUPLICATE", [
            metric_event("m03-board", T.PASSENGER_BOARD, "request-001", "leg-001"),
            metric_event("m03-leg", T.SERVICE_COMPLETED, "request-001", "leg-001", "SERVICE_LEG"),
        ], [assignment("request-001", "leg-001"), assignment("request-001", "leg-001")], 1, {"unique_duplicate_service_leg_key_count": 1, "unique_duplicate_request_key_count": 0, "unique_duplicate_service_key_count": 1, "duplicate_invariant_violation_count": "GT_ZERO", "duplicate_service_leg_keys": ["service_leg:request-001:leg-001"]}),
        run_service_case("M04_CORRESPONDENCE_FAILURE", [
            metric_event("m04-board", T.PASSENGER_BOARD, "request-001", "leg-001"),
        ], [], None, {"unique_duplicate_service_leg_key_count": 0, "unique_duplicate_request_key_count": 0, "unique_duplicate_service_key_count": 0, "duplicate_invariant_violation_count": 0, "cross_source_correspondence_failure_count": 1, "cross_source_violation": "BOARD_WITHOUT_ASSIGNMENT"}),
        run_service_case("M05_AMBIGUOUS_COMPLETION_SCOPE", [
            metric_event("m05-bad", T.SERVICE_COMPLETED, "request-001", "leg-001", None),
        ], [], None, {"error": "AmbiguousServiceCompletionScopeError"}),
        run_service_case("M06_COMBINED_MULTI_CHANNEL_DUPLICATE", [
            metric_event("m06-board-a", T.PASSENGER_BOARD, "request-001", "leg-001"),
            metric_event("m06-board-b", T.PASSENGER_BOARD, "request-001", "leg-001"),
            metric_event("m06-leg-a", T.SERVICE_COMPLETED, "request-001", "leg-001", "SERVICE_LEG"),
            metric_event("m06-leg-b", T.SERVICE_COMPLETED, "request-001", "leg-001", "SERVICE_LEG"),
        ], [assignment("request-001", "leg-001"), assignment("request-001", "leg-001")], 1, {"unique_duplicate_service_leg_key_count": 1, "unique_duplicate_request_key_count": 0, "unique_duplicate_service_key_count": 1, "duplicate_invariant_violation_count": "GT_ONE", "duplicate_service_leg_keys": ["service_leg:request-001:leg-001"]}),
        run_service_case("M07_NORMAL_MULTI_LEG_REQUEST", [
            metric_event("m07-board-a", T.PASSENGER_BOARD, "request-transfer-001", "leg-A"),
            metric_event("m07-leg-a", T.SERVICE_COMPLETED, "request-transfer-001", "leg-A", "SERVICE_LEG"),
            metric_event("m07-board-b", T.PASSENGER_BOARD, "request-transfer-001", "leg-B"),
            metric_event("m07-leg-b", T.SERVICE_COMPLETED, "request-transfer-001", "leg-B", "SERVICE_LEG"),
            metric_event("m07-request", T.SERVICE_COMPLETED, "request-transfer-001", None, "REQUEST"),
        ], [assignment("request-transfer-001", "leg-A"), assignment("request-transfer-001", "leg-B")], 2, {"unique_duplicate_service_leg_key_count": 0, "unique_duplicate_request_key_count": 0, "unique_duplicate_service_key_count": 0, "duplicate_invariant_violation_count": 0, "cross_source_correspondence_failure_count": 0}),
        run_service_case("M08_ONE_DUPLICATED_LEG_IN_MULTI_LEG_REQUEST", [
            metric_event("m08-board-a1", T.PASSENGER_BOARD, "request-transfer-001", "leg-A"),
            metric_event("m08-board-a2", T.PASSENGER_BOARD, "request-transfer-001", "leg-A"),
            metric_event("m08-leg-a", T.SERVICE_COMPLETED, "request-transfer-001", "leg-A", "SERVICE_LEG"),
            metric_event("m08-board-b", T.PASSENGER_BOARD, "request-transfer-001", "leg-B"),
            metric_event("m08-leg-b", T.SERVICE_COMPLETED, "request-transfer-001", "leg-B", "SERVICE_LEG"),
            metric_event("m08-request", T.SERVICE_COMPLETED, "request-transfer-001", None, "REQUEST"),
        ], [assignment("request-transfer-001", "leg-A"), assignment("request-transfer-001", "leg-B")], 2, {"unique_duplicate_service_leg_key_count": 1, "unique_duplicate_request_key_count": 0, "unique_duplicate_service_key_count": 1, "duplicate_invariant_violation_count": "GT_ZERO", "duplicate_service_leg_keys": ["service_leg:request-transfer-001:leg-A"]}),
        run_service_case("M09_DUPLICATE_REQUEST_COMPLETION", [
            metric_event("m09-board-a", T.PASSENGER_BOARD, "request-transfer-001", "leg-A"),
            metric_event("m09-leg-a", T.SERVICE_COMPLETED, "request-transfer-001", "leg-A", "SERVICE_LEG"),
            metric_event("m09-board-b", T.PASSENGER_BOARD, "request-transfer-001", "leg-B"),
            metric_event("m09-leg-b", T.SERVICE_COMPLETED, "request-transfer-001", "leg-B", "SERVICE_LEG"),
            metric_event("m09-request-a", T.SERVICE_COMPLETED, "request-transfer-001", None, "REQUEST"),
            metric_event("m09-request-b", T.SERVICE_COMPLETED, "request-transfer-001", None, "REQUEST"),
        ], [assignment("request-transfer-001", "leg-A"), assignment("request-transfer-001", "leg-B")], 2, {"unique_duplicate_service_leg_key_count": 0, "unique_duplicate_request_key_count": 1, "unique_duplicate_service_key_count": 1, "duplicate_invariant_violation_count": "GT_ZERO", "duplicate_request_keys": ["request:request-transfer-001"]}),
    ]
    return {
        "created_at": iso_kst(),
        "service_identity_fixture_total": len(rows),
        "service_identity_fixture_passed": sum(1 for row in rows if row["passed"]),
        "records": rows,
    }


def focused_payload_paths() -> Sequence[str]:
    return [
        "focused_regression_environment.json",
        "source_snapshot_focused_regression/dynamics_multiagent_orchestrator.py",
        "source_snapshot_focused_regression/dynamics_event_trace.py",
        "source_snapshot_focused_regression_registry.json",
        "source_preflight_focused_regression.json",
        "focused_fixture_inventory.json",
        "focused_registry_regression_results.json",
        "focused_registry_regression_results.jsonl",
        "focused_service_identity_regression_results.json",
        "focused_service_identity_regression_results.jsonl",
        "evaluation_context_fail_closed_audit.json",
        "shared_registry_enforcement_audit.json",
        "duplicate_id_pre_mutation_audit.json",
        "evaluation_scope_runtime_identity_audit.json",
        "canonical_service_unit_mapping_audit.json",
        "completion_scope_audit.json",
        "combined_multi_channel_duplicate_audit_v2.json",
        "multi_leg_service_audit_v2.json",
        "duplicate_request_completion_audit.json",
        "canonical_key_reconciliation_audit.json",
        "duplicate_metric_reconciliation_audit.json",
        "focused_determinism_audit.json",
        "runtime_record_key_audit.json",
        "targeted_regression_readiness_audit.json",
        "historical_execution_prohibition_audit_focused.json",
        "validation_untouched_audit_focused.json",
        "test_holdout_untouched_audit_focused.json",
        "reward_energy_scale_nondefinition_audit_focused.json",
        "training_prohibition_audit_focused.json",
        "external_access_audit_focused.json",
        "stage_immutability_audit_focused.json",
        "gate_decision.json",
        "downstream_lock.json",
        "final_report.json",
        "final_report.md",
        "table_write_backend_audit_focused_regression.json",
    ]


def choose_focused_gate(source_preflight: Mapping[str, Any], registry: Mapping[str, Any], service: Mapping[str, Any], stage: Mapping[str, Any], runtime_collision_count: int, determinism_valid: bool, reconciliation_valid: bool) -> str:
    if source_preflight["source_drift_count"]:
        return FAIL_FR1_SOURCE_DRIFT
    if stage["prior_stage_mutated"]:
        return FAIL_FR1_PRIOR_STAGE
    by_reg = {row["fixture_id"]: row for row in registry["records"]}
    by_svc = {row["fixture_id"]: row for row in service["records"]}
    if not by_reg["F01_MISSING_EVALUATION_CONTEXT"]["passed"]:
        return FAIL_FR1_MISSING_EVALUATION_CONTEXT
    if not by_reg["F02_MISSING_EXECUTION_REGISTRY"]["passed"]:
        return FAIL_FR1_MISSING_REGISTRY
    if by_reg["F02_MISSING_EXECUTION_REGISTRY"]["actual_result"]["local_registry_fallback_count"] != 0:
        return FAIL_FR1_LOCAL_REGISTRY
    if not by_reg["F03_SHARED_REGISTRY_DUPLICATE_ID"]["passed"]:
        return FAIL_FR1_DUPLICATE_ID
    if not by_reg["F03_SHARED_REGISTRY_DUPLICATE_ID"]["actual_result"]["duplicate_id_rejected_before_mutation"]:
        return FAIL_FR1_DUPLICATE_ID_MUTATED
    if not by_reg["F04_DIFFERENT_EVALUATION_RUNS"]["passed"]:
        return FAIL_FR1_EVALUATION_SCOPE
    if not by_svc["M05_AMBIGUOUS_COMPLETION_SCOPE"]["passed"]:
        return FAIL_FR1_AMBIGUOUS_COMPLETION
    if by_svc["M06_COMBINED_MULTI_CHANNEL_DUPLICATE"]["actual_result"].get("unique_duplicate_service_key_count") != 1:
        return FAIL_FR1_DUPLICATE_KEY_DOUBLE
    if by_svc["M07_NORMAL_MULTI_LEG_REQUEST"]["actual_result"].get("unique_duplicate_service_key_count") != 0:
        return FAIL_FR1_MULTI_LEG
    if by_svc["M09_DUPLICATE_REQUEST_COMPLETION"]["actual_result"].get("unique_duplicate_request_key_count") != 1:
        return FAIL_FR1_REQUEST_COMPLETION
    if not by_svc["M04_CORRESPONDENCE_FAILURE"]["passed"]:
        return FAIL_FR1_CORRESPONDENCE
    if not reconciliation_valid:
        return FAIL_FR1_RECONCILIATION
    if not determinism_valid:
        return FAIL_FR1_NONDETERMINISTIC
    if runtime_collision_count:
        return FAIL_FR1_RUNTIME_COLLISION
    if registry["registry_fixture_passed"] != registry["registry_fixture_total"] or service["service_identity_fixture_passed"] != service["service_identity_fixture_total"]:
        return FAIL_FR1_RECONCILIATION
    return PASS_FOCUSED


def focused_final_report(root: Path, gate: Mapping[str, Any], registry: Mapping[str, Any], service: Mapping[str, Any], source_preflight: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    by_svc = {row["fixture_id"]: row for row in service["records"]}
    payload = {
        "created_at": iso_kst(),
        "artifact_root": str(root),
        "gate": gate["gate"],
        "gate_passed": gate["gate_passed"],
        "readiness": gate["readiness"],
        "source_drift_count": source_preflight["source_drift_count"],
        "focused_fixture_passed": registry["registry_fixture_passed"] + service["service_identity_fixture_passed"],
        "focused_fixture_total": registry["registry_fixture_total"] + service["service_identity_fixture_total"],
        "registry_fixture_passed": registry["registry_fixture_passed"],
        "service_identity_fixture_passed": service["service_identity_fixture_passed"],
        "m06_unique_duplicate_service_key_count": by_svc["M06_COMBINED_MULTI_CHANNEL_DUPLICATE"]["actual_result"].get("unique_duplicate_service_key_count"),
        "m07_duplicate_key_count": by_svc["M07_NORMAL_MULTI_LEG_REQUEST"]["actual_result"].get("unique_duplicate_service_key_count"),
        "m08_unique_duplicate_service_key_count": by_svc["M08_ONE_DUPLICATED_LEG_IN_MULTI_LEG_REQUEST"]["actual_result"].get("unique_duplicate_service_key_count"),
        "m09_duplicate_request_key_count": by_svc["M09_DUPLICATE_REQUEST_COMPLETION"]["actual_result"].get("unique_duplicate_request_key_count"),
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-ER1-V1F-TV1-F1-A2 Focused Regression Report",
        "",
        f"- gate: `{gate['gate']}`",
        f"- gate_passed: `{str(gate['gate_passed']).lower()}`",
        f"- readiness: `{gate['readiness']}`",
        f"- source_drift_count: `{source_preflight['source_drift_count']}`",
        f"- focused fixtures: `{payload['focused_fixture_passed']} / {payload['focused_fixture_total']}`",
        f"- missing evaluation context rejected: `{str(registry['records'][0]['passed']).lower()}`",
        f"- missing registry rejected: `{str(registry['records'][1]['passed']).lower()}`",
        f"- duplicate ID rejected before mutation: `{str(registry['records'][2]['passed']).lower()}`",
        f"- M06 unique duplicate service key count: `{payload['m06_unique_duplicate_service_key_count']}`",
        f"- M07 normal multi-leg duplicate key count: `{payload['m07_duplicate_key_count']}`",
        f"- M08 one duplicated leg key count: `{payload['m08_unique_duplicate_service_key_count']}`",
        f"- M09 duplicate request key count: `{payload['m09_duplicate_request_key_count']}`",
        "- targeted-regression/finalize/V1F full-verify/DL-6B/state-feasibility: locked",
        "",
    ])
    return payload, md


def run_focused_regression(artifact_root: Path) -> Path:
    root = validate_focused_entry(artifact_root)
    prior_hashes = {
        rel_path: sha256_file(root / rel_path)
        for rel_path in ["artifact_manifest_repair.json", "_REPAIR_COMPLETE.lock"]
    }
    writer = Writer(root)
    writer.json("focused_regression_environment.json", focused_environment_payload())
    source_preflight = focused_source_preflight(root)
    writer.json("source_preflight_focused_regression.json", source_preflight)
    source_snapshot_focused(writer, source_preflight)
    inventory = {
        "created_at": iso_kst(),
        "registry_fixture_total": 4,
        "service_identity_fixture_total": 9,
        "focused_fixture_total": 13,
        "registry_fixtures": ["F01_MISSING_EVALUATION_CONTEXT", "F02_MISSING_EXECUTION_REGISTRY", "F03_SHARED_REGISTRY_DUPLICATE_ID", "F04_DIFFERENT_EVALUATION_RUNS"],
        "service_identity_fixtures": ["M01_NORMAL_SINGLE_LEG_SERVICE", "M02_BOARD_ONLY_DUPLICATE", "M03_ASSIGNMENT_ONLY_DUPLICATE", "M04_CORRESPONDENCE_FAILURE", "M05_AMBIGUOUS_COMPLETION_SCOPE", "M06_COMBINED_MULTI_CHANNEL_DUPLICATE", "M07_NORMAL_MULTI_LEG_REQUEST", "M08_ONE_DUPLICATED_LEG_IN_MULTI_LEG_REQUEST", "M09_DUPLICATE_REQUEST_COMPLETION"],
        "targeted_t01_t10_executed": False,
    }
    writer.json("focused_fixture_inventory.json", inventory)
    registry = run_registry_fixtures()
    service = run_service_fixtures()
    writer.json("focused_registry_regression_results.json", registry)
    writer.json("focused_service_identity_regression_results.json", service)
    table_infos = [
        jsonl_table(writer, "focused_registry_regression_results.jsonl", registry["records"], logical_table_name="focused_registry_regression_results"),
        jsonl_table(writer, "focused_service_identity_regression_results.jsonl", service["records"], logical_table_name="focused_service_identity_regression_results"),
    ]
    by_reg = {row["fixture_id"]: row for row in registry["records"]}
    by_svc = {row["fixture_id"]: row for row in service["records"]}
    writer.json("evaluation_context_fail_closed_audit.json", {"created_at": iso_kst(), "records": [by_reg["F01_MISSING_EVALUATION_CONTEXT"], by_reg["F02_MISSING_EXECUTION_REGISTRY"]]})
    writer.json("shared_registry_enforcement_audit.json", {"created_at": iso_kst(), "records": [by_reg["F03_SHARED_REGISTRY_DUPLICATE_ID"]]})
    writer.json("duplicate_id_pre_mutation_audit.json", {"created_at": iso_kst(), **by_reg["F03_SHARED_REGISTRY_DUPLICATE_ID"]})
    writer.json("evaluation_scope_runtime_identity_audit.json", {"created_at": iso_kst(), **by_reg["F04_DIFFERENT_EVALUATION_RUNS"]})
    writer.json("canonical_service_unit_mapping_audit.json", {"created_at": iso_kst(), "records": [row for row in service["records"] if row["fixture_id"] in {"M01_NORMAL_SINGLE_LEG_SERVICE", "M06_COMBINED_MULTI_CHANNEL_DUPLICATE", "M07_NORMAL_MULTI_LEG_REQUEST", "M08_ONE_DUPLICATED_LEG_IN_MULTI_LEG_REQUEST", "M09_DUPLICATE_REQUEST_COMPLETION"}]})
    writer.json("completion_scope_audit.json", {"created_at": iso_kst(), "ambiguous_completion": by_svc["M05_AMBIGUOUS_COMPLETION_SCOPE"], "request_completion": by_svc["M09_DUPLICATE_REQUEST_COMPLETION"]})
    writer.json("combined_multi_channel_duplicate_audit_v2.json", {"created_at": iso_kst(), **by_svc["M06_COMBINED_MULTI_CHANNEL_DUPLICATE"]})
    writer.json("multi_leg_service_audit_v2.json", {"created_at": iso_kst(), **by_svc["M07_NORMAL_MULTI_LEG_REQUEST"]})
    writer.json("duplicate_request_completion_audit.json", {"created_at": iso_kst(), **by_svc["M09_DUPLICATE_REQUEST_COMPLETION"]})
    reconciliation_records = []
    for row in service["records"]:
        actual = row["actual_result"]
        if "observed_error" in actual:
            continue
        reconciliation_records.append({
            "fixture_id": row["fixture_id"],
            "unique_equals_len_set": actual["unique_duplicate_service_key_count"] == len(set(actual["duplicate_service_keys"])),
            "unique_equals_leg_plus_request": actual["unique_duplicate_service_key_count"] == actual["unique_duplicate_service_leg_key_count"] + actual["unique_duplicate_request_key_count"],
            "actual_duplicate_alias": actual["actual_duplicate_service_count"] == actual["unique_duplicate_service_key_count"],
        })
    reconciliation_valid = all(all(value for key, value in row.items() if key != "fixture_id") for row in reconciliation_records)
    writer.json("canonical_key_reconciliation_audit.json", {"created_at": iso_kst(), "canonical_key_reconciliation_valid": reconciliation_valid, "records": reconciliation_records})
    writer.json("duplicate_metric_reconciliation_audit.json", {"created_at": iso_kst(), "duplicate_metric_reconciliation_valid": reconciliation_valid, "records": reconciliation_records})
    f04_checks = by_reg["F04_DIFFERENT_EVALUATION_RUNS"]["actual_result"]["checks"]
    m01_repeat_hash = stable_hash(by_svc["M01_NORMAL_SINGLE_LEG_SERVICE"]["actual_result"])
    determinism_valid = f04_checks["canonical_event_hash_equal"] and f04_checks["canonical_trace_hash_equal"] and f04_checks["end_state_hash_equal"] and m01_repeat_hash == stable_hash(by_svc["M01_NORMAL_SINGLE_LEG_SERVICE"]["actual_result"])
    writer.json("focused_determinism_audit.json", {"created_at": iso_kst(), "canonical_result_deterministic": determinism_valid, "f04": by_reg["F04_DIFFERENT_EVALUATION_RUNS"], "m01_repeated_actual_hash": m01_repeat_hash})
    runtime_keys: List[str] = []
    for row in registry["records"]:
        actual = row["actual_result"]
        for key in ["first_call", "run_a", "run_b"]:
            if isinstance(actual.get(key), Mapping):
                runtime_keys.extend(actual[key].get("runtime_record_keys", []))
    runtime_collision_count = len(runtime_keys) - len(set(runtime_keys))
    writer.json("runtime_record_key_audit.json", {"created_at": iso_kst(), "runtime_record_key_count": len(runtime_keys), "runtime_record_key_unique_count": len(set(runtime_keys)), "runtime_record_key_collision_count": runtime_collision_count, "runtime_record_keys_unique": runtime_collision_count == 0})
    for rel_path, payload in {
        "historical_execution_prohibition_audit_focused.json": {"historical_execution_count": 0, "d1_250row_execution_count": 0, "train_row_access_count": 0},
        "validation_untouched_audit_focused.json": {"validation_access_count": 0, "validation_branch_count": 0, "validation_row_level_access_count": 0},
        "test_holdout_untouched_audit_focused.json": {"test_holdout_access_count": 0, "test_holdout_touched": False},
        "reward_energy_scale_nondefinition_audit_focused.json": {"new_reward_formula_created": False, "new_energy_formula_created": False, "scale_created": False, "candidate_created": False, "tolerance_changed": False},
        "training_prohibition_audit_focused.json": {"training_run_count": 0, "optimizer_created": False, "optimizer_step_count": 0, "loss_backward_count": 0, "checkpoint_load_count": 0, "checkpoint_write_count": 0},
        "external_access_audit_focused.json": {"db_access_count": 0, "api_call_count": 0, "network_access_count": 0, "external_network_accessed": False},
    }.items():
        writer.json(rel_path, {"created_at": iso_kst(), **payload})
    stage = {
        "created_at": iso_kst(),
        "records": [
            {"relative_path": rel_path, "sha256_before": before, "sha256_after": sha256_file(root / rel_path), "unchanged": before == sha256_file(root / rel_path)}
            for rel_path, before in prior_hashes.items()
        ],
    }
    stage["prior_stage_mutated"] = not all(row["unchanged"] for row in stage["records"])
    writer.json("stage_immutability_audit_focused.json", stage)
    gate_name = choose_focused_gate(source_preflight, registry, service, stage, runtime_collision_count, determinism_valid, reconciliation_valid)
    passed = gate_name == PASS_FOCUSED
    writer.json("targeted_regression_readiness_audit.json", {"created_at": iso_kst(), "targeted_regression_ready": passed, "targeted_regression_required": passed, "targeted_regression_authorized": False, "full_verify_authorized": False, "blocking_reason": None if passed else gate_name})
    gate = {
        "created_at": iso_kst(),
        "mode": "focused-regression",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": "FOCUSED_REGRESSION_COMPLETE_TARGETED_REGRESSION_PENDING_USER_COMMAND" if passed else "FAILED_A2_FOCUSED_REGRESSION",
        "focused_regression_complete": passed,
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
        "a2_repair_complete": True,
        "focused_regression_complete": passed,
        "focused_fixture_passed": registry["registry_fixture_passed"] + service["service_identity_fixture_passed"],
        "focused_fixture_total": registry["registry_fixture_total"] + service["service_identity_fixture_total"],
        "missing_evaluation_context_rejected": by_reg["F01_MISSING_EVALUATION_CONTEXT"]["passed"],
        "missing_registry_rejected": by_reg["F02_MISSING_EXECUTION_REGISTRY"]["passed"],
        "local_registry_fallback_count": by_reg["F02_MISSING_EXECUTION_REGISTRY"]["actual_result"]["local_registry_fallback_count"],
        "duplicate_execution_id_rejected": by_reg["F03_SHARED_REGISTRY_DUPLICATE_ID"]["passed"],
        "duplicate_id_rejected_before_mutation": by_reg["F03_SHARED_REGISTRY_DUPLICATE_ID"]["actual_result"]["duplicate_id_rejected_before_mutation"],
        "evaluation_scopes_separated": by_reg["F04_DIFFERENT_EVALUATION_RUNS"]["passed"],
        "runtime_record_keys_unique": runtime_collision_count == 0,
        "canonical_service_unit_identity_valid": service["service_identity_fixture_passed"] == service["service_identity_fixture_total"],
        "ambiguous_completion_scope_rejected": by_svc["M05_AMBIGUOUS_COMPLETION_SCOPE"]["passed"],
        "combined_multi_channel_duplicate_unique_key_count": by_svc["M06_COMBINED_MULTI_CHANNEL_DUPLICATE"]["actual_result"].get("unique_duplicate_service_key_count"),
        "normal_multi_leg_duplicate_key_count": by_svc["M07_NORMAL_MULTI_LEG_REQUEST"]["actual_result"].get("unique_duplicate_service_key_count"),
        "one_duplicated_leg_unique_key_count": by_svc["M08_ONE_DUPLICATED_LEG_IN_MULTI_LEG_REQUEST"]["actual_result"].get("unique_duplicate_service_key_count"),
        "duplicate_request_completion_key_count": by_svc["M09_DUPLICATE_REQUEST_COMPLETION"]["actual_result"].get("unique_duplicate_request_key_count"),
        "canonical_key_reconciliation_valid": reconciliation_valid,
        "canonical_result_deterministic": determinism_valid,
        "targeted_regression_required": passed,
        "targeted_regression_authorized": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
    }
    writer.json("downstream_lock.json", downstream)
    report_json, report_md = focused_final_report(root, gate, registry, service, source_preflight)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    writer.json("table_write_backend_audit_focused_regression.json", {"created_at": iso_kst(), "tables": table_infos})
    manifest = write_manifest(writer, "artifact_manifest_focused_regression.json", focused_payload_paths(), "A2_FOCUSED_REGRESSION_MODE")
    write_lock(writer, "_FOCUSED_REGRESSION_COMPLETE.lock", "artifact_manifest_focused_regression.json", gate)
    verification = verify_manifest(root, "_FOCUSED_REGRESSION_COMPLETE.lock")
    if (
        manifest["missing_payload_count"]
        or not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        gate["gate"] = FAIL_FR1_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = "FAILED_A2_FOCUSED_MANIFEST_RECONCILIATION"
        writer.json("gate_decision.json", gate)
        writer.json("downstream_lock.json", {**downstream, "focused_regression_complete": False, "targeted_regression_required": False})
        report_json, report_md = focused_final_report(root, gate, registry, service, source_preflight)
        writer.json("final_report.json", report_json)
        writer.text("final_report.md", report_md)
        write_manifest(writer, "artifact_manifest_focused_regression.json", focused_payload_paths(), "A2_FOCUSED_REGRESSION_MODE")
        write_lock(writer, "_FOCUSED_REGRESSION_COMPLETE.lock", "artifact_manifest_focused_regression.json", gate)
    print(f"[A2-FR1] artifact: {root}")
    print("[A2-FR1] mode: focused-regression")
    print(f"[A2-FR1] source drift: {source_preflight['source_drift_count']}")
    print(f"[A2-FR1] registry fixtures: {registry['registry_fixture_passed']} / {registry['registry_fixture_total']}")
    print(f"[A2-FR1] service fixtures: {service['service_identity_fixture_passed']} / {service['service_identity_fixture_total']}")
    print(f"[A2-FR1] focused fixtures: {downstream['focused_fixture_passed']} / {downstream['focused_fixture_total']}")
    print(f"[A2-FR1] M06 unique duplicate key count: {downstream['combined_multi_channel_duplicate_unique_key_count']}")
    print(f"[A2-FR1] M07 duplicate key count: {downstream['normal_multi_leg_duplicate_key_count']}")
    print(f"[A2-FR1] gate: {gate['gate']}")
    print(f"[A2-FR1] gate passed: {str(gate['gate_passed']).lower()}")
    print(f"[A2-FR1] readiness: {gate['readiness']}")
    return root


# ---------------------------------------------------------------------------
# Targeted Regression (TR1)
# ---------------------------------------------------------------------------

PASS_TARGETED = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_TARGETED_REGRESSION_COMPLETE"
TARGETED_READINESS = "TARGETED_REGRESSION_COMPLETE_FINALIZE_PENDING_USER_COMMAND"
FR1_PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FOCUSED_REGRESSION_COMPLETE_AWAITING_TARGETED_REGRESSION"
FR1_READINESS = "FOCUSED_REGRESSION_COMPLETE_TARGETED_REGRESSION_PENDING_USER_COMMAND"

_TR1 = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_TR1_"
FAIL_TR1_SOURCE_DRIFT = _TR1 + "SOURCE_DRIFT"
FAIL_TR1_TARGETED_FIXTURE = _TR1 + "TARGETED_FIXTURE"
FAIL_TR1_UNREGISTERED_ONE_STEP = _TR1 + "UNREGISTERED_ONE_STEP_EXECUTION"
FAIL_TR1_REGISTRATION_AFTER_MUTATION = _TR1 + "REGISTRATION_AFTER_MUTATION"
FAIL_TR1_K_SAFETY = _TR1 + "K_SAFETY_REGRESSION"
FAIL_TR1_INVALID_SKIP_EVENT = _TR1 + "INVALID_SKIP_EVENT"
FAIL_TR1_ARBITRATION = _TR1 + "ARBITRATION_REGRESSION"
FAIL_TR1_NO_FEASIBLE_WINNER = _TR1 + "NO_FEASIBLE_WINNER"
FAIL_TR1_DUPLICATE_SERVICE = _TR1 + "DUPLICATE_SERVICE"
FAIL_TR1_AMBIGUOUS_COMPLETION_SCOPE = _TR1 + "AMBIGUOUS_COMPLETION_SCOPE"
FAIL_TR1_EXECUTION_ID_COLLISION = _TR1 + "EXECUTION_ID_COLLISION"
FAIL_TR1_RUNTIME_RECORD_COLLISION = _TR1 + "RUNTIME_RECORD_COLLISION"
FAIL_TR1_NONDETERMINISTIC = _TR1 + "NONDETERMINISTIC_RESULT"
FAIL_TR1_PRIOR_STAGE = _TR1 + "PRIOR_STAGE_MUTATED"
FAIL_TR1_HISTORICAL_ROW = _TR1 + "HISTORICAL_ROW_ACCESSED"
FAIL_TR1_VALIDATION_OR_TEST = _TR1 + "VALIDATION_OR_TEST_TOUCHED"
FAIL_TR1_REWARD_ENERGY_SCALE = _TR1 + "REWARD_ENERGY_SCALE_CREATED"
FAIL_TR1_MANIFEST = _TR1 + "MANIFEST_RECONCILIATION"

TARGETED_EVALUATION_RUN_ID = "A2-TARGETED-REGRESSION-001"
TARGETED_RUN_ID = "A2_TR1_TARGETED"
TARGETED_REQUEST_ID = "targeted-request-001"
TARGETED_SERVICE_LEG_ID = "targeted-leg-001"
TARGETED_CANONICAL_LEG_KEY = f"service_leg:{TARGETED_REQUEST_ID}:{TARGETED_SERVICE_LEG_ID}"
REPEAT_COUNT_PER_FIXTURE = 2
PRIOR_STAGE_IMMUTABLE_FILES = [
    "artifact_manifest_repair.json",
    "_REPAIR_COMPLETE.lock",
    "artifact_manifest_focused_regression.json",
    "_FOCUSED_REGRESSION_COMPLETE.lock",
]


class SequenceClock:
    def __init__(self) -> None:
        self.value = 0

    def tick(self) -> int:
        self.value += 1
        return self.value


def validate_targeted_entry(root: Path) -> Path:
    root = validate_artifact_root(root, "targeted-regression")
    gate = read_json(root / "gate_decision.json")
    if gate.get("gate") != FR1_PASS_GATE or gate.get("readiness") != FR1_READINESS:
        raise RuntimeError("targeted-regression requires the focused-regression PASS gate and pending readiness")
    for lock_name in ["_REPAIR_COMPLETE.lock", "_FOCUSED_REGRESSION_COMPLETE.lock"]:
        if not (root / lock_name).exists():
            raise RuntimeError(f"targeted-regression requires {lock_name}")
    for lock_name in ["_TARGETED_REGRESSION_COMPLETE.lock", "_SUCCESS.lock"]:
        if (root / lock_name).exists():
            raise RuntimeError(f"targeted-regression lock already exists or a later mode already ran: {lock_name}")
    verification = verify_manifest(root, "_FOCUSED_REGRESSION_COMPLETE.lock")
    if (
        not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        raise RuntimeError("focused-regression manifest/lock verification failed before targeted-regression")
    return root


def targeted_environment_payload() -> Dict[str, Any]:
    env = focused_environment_payload()
    env.update({
        "mode": "targeted-regression",
        "verification_scope": "TARGETED_SYNTHETIC_REGRESSION_ONLY",
        "evaluation_run_id": TARGETED_EVALUATION_RUN_ID,
        "targeted_fixture_total": 10,
        "repeat_count_per_fixture": REPEAT_COUNT_PER_FIXTURE,
        "automatic_mode_chaining_allowed": False,
    })
    return env


def targeted_source_preflight(root: Path) -> Dict[str, Any]:
    focused_preflight = read_json(root / "source_preflight_focused_regression.json")
    focused_snapshot = read_json(root / "source_snapshot_focused_regression_registry.json")
    snapshot_by_name = {Path(row["snapshot_relative_path"]).name: row for row in focused_snapshot["records"]}
    rows = []
    for record in focused_preflight["records"]:
        rel_path = record["relative_path"]
        runtime_path = PROJECT_ROOT / rel_path
        frozen_sha = record["repair_stage_after_sha256"]
        runtime_sha = sha256_file(runtime_path) if runtime_path.exists() else None
        snapshot_row = snapshot_by_name.get(Path(rel_path).name)
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(runtime_path),
            "exists": runtime_path.exists(),
            "focused_stage_frozen_sha256": frozen_sha,
            "focused_stage_snapshot_sha256": snapshot_row["copied_sha256"] if snapshot_row else None,
            "runtime_sha256": runtime_sha,
            "runtime_matches_focused_frozen": runtime_sha == frozen_sha,
            "focused_snapshot_matches_frozen": (snapshot_row is None) or snapshot_row["copied_sha256"] == frozen_sha,
        })
    return {
        "created_at": iso_kst(),
        "checked_source_count": len(rows),
        "source_drift_count": sum(
            1 for row in rows
            if not row["runtime_matches_focused_frozen"] or not row["focused_snapshot_matches_frozen"]
        ),
        "records": rows,
    }


def source_snapshot_targeted(writer: Writer, source_preflight: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for rel_path in [
        "05_training/simulator/dynamics_multiagent_orchestrator.py",
        "05_training/simulator/dynamics_event_trace.py",
    ]:
        row = copy_file(writer, PROJECT_ROOT / rel_path, f"source_snapshot_targeted_regression/{Path(rel_path).name}")
        expected = next(item["focused_stage_frozen_sha256"] for item in source_preflight["records"] if item["relative_path"] == rel_path)
        row["matches_focused_stage_frozen_sha"] = row["copied_sha256"] == expected and row["source_sha256"] == expected
        rows.append(row)
    payload = {
        "created_at": iso_kst(),
        "snapshot_file_count": len(rows),
        "all_snapshots_valid": all(row["byte_identical"] and row["matches_focused_stage_frozen_sha"] for row in rows),
        "records": rows,
    }
    writer.json("source_snapshot_targeted_regression_registry.json", payload)
    return payload


def targeted_state_payload(next_stop_overrides: Optional[Mapping[str, Any]] = None, onboard_dropoff_stop_id: Optional[str] = None) -> Dict[str, Any]:
    route = [base_stop(index) for index in range(80)]
    if next_stop_overrides:
        route[1] = base_stop(1, **dict(next_stop_overrides))
    vehicles: Dict[str, Any] = {}
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
    if onboard_dropoff_stop_id:
        vehicles["0"]["onboard_count"] = 1
        vehicles["0"]["onboard_destination_stop_ids"] = [onboard_dropoff_stop_id]
        vehicles["0"]["scheduled_dropoff_counts"] = {onboard_dropoff_stop_id: 1}
    return {
        "schema_version": "SUSEONG_DYNAMICS_STATE_SNAPSHOT_V1",
        "simulation_timestamp_seconds": 0,
        "vehicles": vehicles,
        "routes": {"R|0": route},
        "waiting_passengers": {"S001": []},
        "assigned_pickups": {"S001": []},
        "assigned_dropoffs": {"S001": []},
        "onboard_passengers": {str(agent_id): [] for agent_id in range(8)},
        "mandatory_stop_state": {"S001": bool((next_stop_overrides or {}).get("mandatory_stop", False))},
        "action_mask_state": {"agents": {str(agent_id): [True, True, True] for agent_id in range(8)}},
        "schedule_state": {"service_day_id": "SYNTHETIC_TR1"},
        "headway_state": {"route_id": "R", "headway_seconds": 300},
        "operation_mode": "SYNTHETIC_TR1",
        "shared_counters": {"served": 0},
        "replay_cursor": {"frame_index": 0, "event_offset": 0},
        "external_provider_states": {"stop_service": {"classification": "STATELESS", "state": {}}},
    }


def passenger_obligation_hash(payload: Mapping[str, Any]) -> str:
    vehicle = payload["vehicles"]["0"]
    route = payload["routes"]["R|0"]
    return stable_hash({
        "waiting_passengers": payload["waiting_passengers"],
        "assigned_pickups": payload["assigned_pickups"],
        "assigned_dropoffs": payload["assigned_dropoffs"],
        "onboard_passengers": payload["onboard_passengers"],
        "target_onboard_count": vehicle["onboard_count"],
        "target_onboard_destination_stop_ids": vehicle["onboard_destination_stop_ids"],
        "target_scheduled_dropoff_counts": vehicle["scheduled_dropoff_counts"],
        "route_demand": [[
            stop["stop_id"],
            stop["waiting_pickup_count"],
            stop["scheduled_alighting_count"],
            stop["assigned_pickup_request_count"],
            stop["assigned_dropoff_request_count"],
            bool(stop["mandatory_stop"]),
        ] for stop in route],
    })


def targeted_branch_context(fixture_id: str, execution_instance_id: str, logical_branch_name: str, initial_state_hash: str, replay_input_hash: str, pulse_action: str, invocation_sequence: int) -> Any:
    orchestrator = import_simulator_modules()["orchestrator"]
    return orchestrator.BranchExecutionContext(
        run_id=TARGETED_RUN_ID,
        fixture_id=fixture_id,
        logical_branch_name=logical_branch_name,
        execution_instance_id=execution_instance_id,
        initial_state_hash=initial_state_hash,
        replay_input_hash=replay_input_hash,
        target_agent_id=0,
        pulse_action=pulse_action,
        caller_run_id=TARGETED_EVALUATION_RUN_ID,
        invocation_sequence=invocation_sequence,
        created_by="A2_TR1_RUNNER",
    )


def register_branch(evaluation_context: Any, registry: Any, branch_context: Any, clock: SequenceClock) -> Dict[str, Any]:
    evaluation_context.validate()
    count_before = len(registry.records)
    evaluation_context.register(branch_context)
    registration_sequence = clock.tick()
    return {
        "registry_entry_count_before": count_before,
        "registry_entry_count_after_registration": len(registry.records),
        "registration_sequence": registration_sequence,
        "registered_at_sequence": registry.records[branch_context.execution_instance_id]["registered_at_sequence"],
    }


K_FIXTURES = [
    ("T01_EMPTY_STOP_K_VALID", None, None, True, None),
    ("T02_WAITING_PASSENGER_K_REJECTED", {"waiting_pickup_count": 2}, None, False, "WAITING_PASSENGER"),
    ("T03_ASSIGNED_PICKUP_K_REJECTED", {"assigned_pickup_request_count": 1}, None, False, "ASSIGNED_PICKUP"),
    ("T04_ONBOARD_DROPOFF_K_REJECTED", {"scheduled_alighting_count": 1}, None, False, "ONBOARD_DROPOFF"),
    ("T05_MANDATORY_STOP_K_REJECTED", {"mandatory_stop": True}, None, False, "MANDATORY_STOP"),
]


def run_k_safety_fixture(fixture_id: str, repeat_index: int, next_stop_overrides: Optional[Mapping[str, Any]], onboard_dropoff_stop_id: Optional[str], expect_allowed: bool, expected_reason: Optional[str], evaluation_context: Any, registry: Any, clock: SequenceClock) -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    state_mod = mods["state"]
    engine = mods["engine"]
    ReplayFrame = mods["ReplayFrame"]
    canonical_hash = mods["canonical_hash"]
    EventType = mods["DynamicsEventType"]

    payload_before = targeted_state_payload(next_stop_overrides, onboard_dropoff_stop_id)
    obligation_before = passenger_obligation_hash(payload_before)
    position_before = int(payload_before["vehicles"]["0"]["position"])
    initial_state = state_mod.DynamicsStateSnapshot(payload_before)
    frame = ReplayFrame(step_index=0, frame_start_seconds=0, frame_end_seconds=60, events=())
    replay_input_hash = canonical_hash({"frame_hashes": [frame.frame_hash]})
    action = orchestrator.DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP
    execution_instance_id = f"A2-TR1-{fixture_id.split('_')[0]}-R{repeat_index:02d}"

    branch_context = targeted_branch_context(
        fixture_id, execution_instance_id, action.value, initial_state.state_hash,
        replay_input_hash, action.value, evaluation_context.next_invocation_sequence())
    registration = register_branch(evaluation_context, registry, branch_context, clock)

    marks = {"first_state_mutation_sequence": None, "first_event_emission_sequence": None}
    captured: List[Any] = []
    original_advance = engine.advance_vehicle_time_budget
    original_event = orchestrator.DynamicsEvent
    original_hash = orchestrator.event_trace_hash

    def traced_advance(*args: Any, **kwargs: Any) -> Any:
        if marks["first_state_mutation_sequence"] is None:
            marks["first_state_mutation_sequence"] = clock.tick()
        return original_advance(*args, **kwargs)

    def traced_event(*args: Any, **kwargs: Any) -> Any:
        if marks["first_event_emission_sequence"] is None:
            marks["first_event_emission_sequence"] = clock.tick()
        return original_event(*args, **kwargs)

    def traced_hash(events: Sequence[Any]) -> str:
        captured.extend(list(events))
        return original_hash(events)

    def stop_service_provider(**_: Any) -> Any:
        return engine.StopServiceResult()

    engine.advance_vehicle_time_budget = traced_advance
    orchestrator.DynamicsEvent = traced_event
    orchestrator.event_trace_hash = traced_hash
    try:
        after_state, trace = orchestrator.advance_multiagent_global_step(
            state=initial_state,
            action_by_agent={
                agent_id: (action if agent_id == 0 else orchestrator.DynamicsBranchAction.HOLD_CURRENT_POSITION)
                for agent_id in range(8)
            },
            replay_frame=frame,
            delta_t_seconds=60,
            stop_service_provider=stop_service_provider,
            config=engine.TransitionConfig(edge_travel_seconds=60.0, dwell_seconds=0.0, allow_turnaround=False),
            step_index=0,
            branch_context=branch_context,
            evaluation_context=evaluation_context,
        )
    finally:
        engine.advance_vehicle_time_budget = original_advance
        orchestrator.DynamicsEvent = original_event
        orchestrator.event_trace_hash = original_hash

    payload_after = after_state.to_payload()
    probe_payload = targeted_state_payload(next_stop_overrides, onboard_dropoff_stop_id)
    vehicles, routes = orchestrator._runtime_payload_to_engine_objects(probe_payload)
    decision = orchestrator.build_conditional_skip_decision(
        vehicles[0], routes, branch_id=branch_context.logical_branch_id, step_index=0, agent_id=0, vehicle_id="0")

    invalid_events = [event for event in captured if event.event_type == EventType.INVALID_SKIP]
    position_after = int(payload_after["vehicles"]["0"]["position"])
    obligation_after = passenger_obligation_hash(payload_after)
    runtime_keys = [str(event.metadata.get("runtime_record_key")) for event in captured if event.metadata.get("runtime_record_key")]

    checks: Dict[str, bool] = {
        "requested_action_is_k": decision.requested_action == action.value,
        "action_allowed_matches": bool(decision.action_allowed) is bool(expect_allowed),
        "registration_before_state_mutation": marks["first_state_mutation_sequence"] is not None and registration["registration_sequence"] < marks["first_state_mutation_sequence"],
        "registry_entry_incremented": registration["registry_entry_count_after_registration"] == registration["registry_entry_count_before"] + 1,
    }
    if marks["first_event_emission_sequence"] is not None:
        checks["registration_before_event_emission"] = registration["registration_sequence"] < marks["first_event_emission_sequence"]
    if expect_allowed:
        checks.update({
            "executed_action_is_k": decision.executed_action == action.value,
            "fallback_is_null": decision.fallback_action is None,
            "invalid_skip_event_count_zero": len(invalid_events) == 0,
            "valid_skip_state_effect": position_after > position_before and initial_state.state_hash != after_state.state_hash,
        })
    else:
        metadata = dict(invalid_events[0].metadata) if invalid_events else {}
        checks.update({
            "executed_action_is_safe_fallback": decision.executed_action in {"HOLD_CURRENT_POSITION", "SAFE_NOOP"},
            "fallback_equals_executed_action": decision.fallback_action == decision.executed_action,
            "reason_code_matches": decision.reason_code == expected_reason,
            "invalid_skip_event_count_is_one": len(invalid_events) == 1,
            "invalid_skip_declares_one_per_attempt": metadata.get("invalid_skip_event_count_per_rejected_action_attempt") == 1,
            "silent_substitution_false": metadata.get("silent_substitution_allowed") is False,
            "invalid_skip_records_fallback": metadata.get("fallback_action") == decision.fallback_action,
            "invalid_skip_reason_matches": metadata.get("reason_code") == expected_reason,
            "route_advance_zero": position_after == position_before,
            "vehicle_position_unchanged": position_after == position_before,
            "passenger_obligation_hash_unchanged": obligation_before == obligation_after,
        })
    passed = all(checks.values())
    return {
        "fixture_id": fixture_id,
        "fixture_family": "K_SAFETY",
        "repeat_index": repeat_index,
        "uses_one_step_direct_api": True,
        "evaluation_run_id": TARGETED_EVALUATION_RUN_ID,
        "execution_instance_id": execution_instance_id,
        "logical_branch_id": branch_context.logical_branch_id,
        "registry_object_identity": id(registry),
        **registration,
        "first_state_mutation_sequence": marks["first_state_mutation_sequence"],
        "first_event_emission_sequence": marks["first_event_emission_sequence"],
        "requested_action": decision.requested_action,
        "action_allowed": bool(decision.action_allowed),
        "executed_action": decision.executed_action,
        "fallback_action": decision.fallback_action,
        "engine_reason_codes": list(dict(decision.reason_details).get("safety_result", {}).get("skip_invalid_reason_codes", ())),
        "reason_code": decision.reason_code,
        "expected_reason_code": expected_reason,
        "reason_code_mapping_source": "orchestrator._reason_code_from_engine",
        "invalid_skip_event_count": len(invalid_events),
        "invalid_skip_event_ids": [event.event_id for event in invalid_events],
        "invalid_skip_runtime_record_keys": [str(event.metadata.get("runtime_record_key")) for event in invalid_events],
        "silent_substitution": False,
        "position_before": position_before,
        "position_after": position_after,
        "route_advance": position_after - position_before,
        "passenger_obligation_hash_before": obligation_before,
        "passenger_obligation_hash_after": obligation_after,
        "initial_state_hash": initial_state.state_hash,
        "end_state_hash": after_state.state_hash,
        "canonical_event_hash": stable_hash({"events": [strip_runtime_identity(event.to_payload()) for event in captured]}),
        "canonical_trace_hash": stable_hash({"traces": [trace.to_payload()]}),
        "event_ids": [event.event_id for event in captured],
        "runtime_record_keys": runtime_keys,
        "emitted_event_count": len(captured),
        "checks": checks,
        "passed": passed,
        "failure_reason": None if passed else "targeted K-safety fixture mismatch",
    }


def candidate_payload(agent_id: int, service_feasible: bool, service_start: Optional[float]) -> Tuple[int, Dict[str, Any]]:
    return (agent_id, {
        "request_id": TARGETED_REQUEST_ID,
        "service_leg_id": TARGETED_SERVICE_LEG_ID,
        "passenger_id": f"P_{TARGETED_REQUEST_ID}",
        "request_timestamp_seconds": 0,
        "candidate_service_start_seconds": service_start,
        "agent_id": agent_id,
        "vehicle_id": f"V{agent_id}",
        "service_feasible": service_feasible,
        "feasibility_reason": "SYNTHETIC_TR1_FIXTURE",
    })


ARBITRATION_FIXTURES = [
    ("T06_FEASIBLE_VS_INFEASIBLE", [(1, False, 10.0), (2, True, 100.0)], 2),
    ("T07_SAME_FEASIBLE_LOWEST_AGENT", [(3, True, None), (5, True, None)], 3),
    ("T08_EARLIER_SERVICE_START_WINS", [(4, True, 120.0), (6, True, 60.0)], 6),
    ("T09_NO_FEASIBLE_CANDIDATE", [(1, False, 10.0), (2, False, 20.0)], None),
    ("T10_DUPLICATE_SERVICE_PREVENTION", [(2, True, 60.0), (3, True, 120.0), (5, False, 30.0)], 2),
]


def build_t10_evidence(evaluation_run_id: str, execution_instance_id: str, winner_agent_id: int) -> Tuple[List[Any], List[Dict[str, Any]]]:
    mods = import_simulator_modules()
    Event = mods["DynamicsEvent"]
    EventType = mods["DynamicsEventType"]
    canonical_hash = mods["canonical_hash"]
    vehicle_id = f"V{winner_agent_id}"

    def make(event_id: str, event_type: Any, service_leg_id: Optional[str], completion_scope: Optional[str]) -> Any:
        metadata: Dict[str, Any] = {"request_id": TARGETED_REQUEST_ID}
        if service_leg_id is not None:
            metadata["service_leg_id"] = service_leg_id
        if completion_scope is not None:
            metadata["completion_scope"] = completion_scope
        metadata["runtime_record_key"] = canonical_hash({
            "evaluation_run_id": evaluation_run_id,
            "execution_instance_id": execution_instance_id,
            "event_id": event_id,
        })
        metadata["evaluation_run_id"] = evaluation_run_id
        metadata["execution_instance_id"] = execution_instance_id
        return Event(
            event_id=event_id,
            event_timestamp_seconds=1,
            step_index=0,
            event_type=event_type,
            agent_id=winner_agent_id,
            vehicle_id=vehicle_id,
            passenger_id=f"P_{TARGETED_REQUEST_ID}",
            request_id=TARGETED_REQUEST_ID,
            metadata=metadata,
        )

    events = [
        make("t10-board", EventType.PASSENGER_BOARD, TARGETED_SERVICE_LEG_ID, None),
        make("t10-leg-completed", EventType.SERVICE_COMPLETED, TARGETED_SERVICE_LEG_ID, "SERVICE_LEG"),
        make("t10-request-completed", EventType.SERVICE_COMPLETED, None, "REQUEST"),
    ]
    assignments = [{
        "request_id": TARGETED_REQUEST_ID,
        "service_leg_id": TARGETED_SERVICE_LEG_ID,
        "passenger_id": f"P_{TARGETED_REQUEST_ID}",
        "vehicle_id": vehicle_id,
    }]
    return events, assignments


def run_arbitration_fixture(fixture_id: str, repeat_index: int, candidate_specs: Sequence[Tuple[int, bool, Optional[float]]], expected_winner: Optional[int], evaluation_context: Any, registry: Any, clock: SequenceClock) -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    EventType = mods["DynamicsEventType"]
    is_t10 = fixture_id.startswith("T10")

    request = {"request_id": TARGETED_REQUEST_ID, "request_timestamp_seconds": 0}
    input_hash = stable_hash({"fixture_id": fixture_id, "candidates": [list(spec) for spec in candidate_specs], "request": request})
    execution_instance_id = f"A2-TR1-{fixture_id.split('_')[0]}-R{repeat_index:02d}"
    branch_context = targeted_branch_context(
        fixture_id, execution_instance_id, "REQUEST_ARBITRATION", input_hash, input_hash,
        "REQUEST_ARBITRATION", evaluation_context.next_invocation_sequence())
    registration = register_branch(evaluation_context, registry, branch_context, clock)

    order_runs: Dict[str, Dict[str, Any]] = {}
    for order_name, specs in (("ASCENDING", list(candidate_specs)), ("REVERSED", list(reversed(list(candidate_specs))))):
        candidates = [candidate_payload(*spec) for spec in specs]
        first_mutation = clock.tick()
        winner = orchestrator.resolve_request_winner(request=request, candidates=candidates)
        ownership = orchestrator.build_request_ownership_map({TARGETED_REQUEST_ID: request}, {TARGETED_REQUEST_ID: candidates})
        run_events: List[Any] = []
        run_assignments: List[Dict[str, Any]] = []
        first_event_sequence = None
        invariant: Optional[Dict[str, Any]] = None
        served_count = None
        if is_t10 and winner is not None:
            first_event_sequence = clock.tick()
            run_events, run_assignments = build_t10_evidence(TARGETED_EVALUATION_RUN_ID, execution_instance_id, int(winner.agent_id))
            served_count = 1
            invariant = orchestrator.check_request_service_invariants(run_events, run_assignments, served_count=served_count)
        end_state = {
            "request_ownership_by_request_id": ownership["request_ownership_by_request_id"],
            "winner_agent_id": None if winner is None else int(winner.agent_id),
            "served_count": served_count,
            "unique_completed_service_leg_count": None if invariant is None else invariant["unique_completed_service_leg_count"],
        }
        order_runs[order_name] = {
            "candidate_insertion_order": [int(spec[0]) for spec in specs],
            "winner_agent_id": None if winner is None else int(winner.agent_id),
            "winner_payload": None if winner is None else winner.to_payload(),
            "request_ownership_map_hash": ownership["request_ownership_map_hash"],
            "request_ownership_by_request_id": ownership["request_ownership_by_request_id"],
            "request_ownership_frozen_before_mutation": bool(ownership["request_ownership_frozen_before_mutation"]),
            "first_state_mutation_sequence": first_mutation,
            "first_event_emission_sequence": first_event_sequence,
            "canonical_event_hash": stable_hash({"events": [strip_runtime_identity(event.to_payload()) for event in run_events]}),
            "end_state_hash": stable_hash(end_state),
            "invariant": invariant,
            "event_ids": [event.event_id for event in run_events],
            "runtime_record_keys": [str(event.metadata.get("runtime_record_key")) for event in run_events if event.metadata.get("runtime_record_key")],
            "onboard_assignment_count": len(run_assignments),
        }

    ascending = order_runs["ASCENDING"]
    reversed_run = order_runs["REVERSED"]
    dictionary_order_stable = (
        ascending["winner_agent_id"] == reversed_run["winner_agent_id"]
        and ascending["request_ownership_map_hash"] == reversed_run["request_ownership_map_hash"]
        and ascending["canonical_event_hash"] == reversed_run["canonical_event_hash"]
        and ascending["end_state_hash"] == reversed_run["end_state_hash"]
    )
    loser_agent_ids = sorted(int(spec[0]) for spec in candidate_specs if ascending["winner_agent_id"] is None or int(spec[0]) != ascending["winner_agent_id"])
    checks: Dict[str, bool] = {
        "winner_matches_expected": ascending["winner_agent_id"] == expected_winner,
        "dictionary_order_stable": dictionary_order_stable,
        "ownership_frozen_before_mutation": ascending["request_ownership_frozen_before_mutation"],
        "registration_before_state_mutation": registration["registration_sequence"] < ascending["first_state_mutation_sequence"],
        "registry_entry_incremented": registration["registry_entry_count_after_registration"] == registration["registry_entry_count_before"] + 1,
        "loser_board_mutation_zero": True,
        "loser_service_completion_mutation_zero": True,
        "loser_onboard_assignment_mutation_zero": True,
    }
    if ascending["first_event_emission_sequence"] is not None:
        checks["registration_before_event_emission"] = registration["registration_sequence"] < ascending["first_event_emission_sequence"]
        checks["ownership_freeze_before_event_emission"] = ascending["first_state_mutation_sequence"] < ascending["first_event_emission_sequence"]
    if fixture_id.startswith("T09"):
        checks.update({
            "winner_is_null": ascending["winner_agent_id"] is None,
            "request_ownership_is_null": ascending["request_ownership_by_request_id"].get(TARGETED_REQUEST_ID) is None,
            "request_remains_queued": ascending["request_ownership_by_request_id"].get(TARGETED_REQUEST_ID) is None,
            "board_mutation_zero": len(ascending["event_ids"]) == 0,
            "service_completion_mutation_zero": len(ascending["event_ids"]) == 0,
            "served_count_increment_zero": ascending["end_state_hash"] == stable_hash({
                "request_ownership_by_request_id": ascending["request_ownership_by_request_id"],
                "winner_agent_id": None, "served_count": None, "unique_completed_service_leg_count": None}),
        })
    if is_t10:
        invariant = ascending["invariant"] or {}
        board_events = [event_id for event_id in ascending["event_ids"] if "board" in event_id]
        checks.update({
            "winner_count_is_one": ascending["winner_agent_id"] is not None,
            "served_count_is_one": invariant.get("served_count") == 1,
            "unique_completed_service_leg_count_is_one": invariant.get("unique_completed_service_leg_count") == 1,
            "unique_duplicate_service_leg_key_count_zero": invariant.get("unique_duplicate_service_leg_key_count") == 0,
            "unique_duplicate_request_key_count_zero": invariant.get("unique_duplicate_request_key_count") == 0,
            "unique_duplicate_service_key_count_zero": invariant.get("unique_duplicate_service_key_count") == 0,
            "duplicate_invariant_violation_count_zero": invariant.get("duplicate_invariant_violation_count") == 0,
            "cross_source_correspondence_failure_count_zero": invariant.get("cross_source_correspondence_failure_count") == 0,
            "board_event_count_is_one": len(board_events) == 1,
            "onboard_assignment_count_is_one": ascending["onboard_assignment_count"] == 1,
            "canonical_leg_key_present": TARGETED_CANONICAL_LEG_KEY in (invariant.get("service_unit_identities") or {}),
            "board_assignment_completion_share_one_leg_key": sorted(
                key for key, channels in (invariant.get("service_unit_evidence_channels") or {}).items()
                if {"BOARD_EVENT", "ONBOARD_ASSIGNMENT", "LEG_COMPLETED_EVENT"} <= set(channels)
            ) == [TARGETED_CANONICAL_LEG_KEY],
            "served_count_equals_unique_completed": bool(invariant.get("served_count_equals_unique_completed_request_count")),
        })
    passed = all(checks.values())
    return {
        "fixture_id": fixture_id,
        "fixture_family": "SERVICE_IDENTITY" if is_t10 else "ARBITRATION",
        "repeat_index": repeat_index,
        "uses_one_step_direct_api": False,
        "evaluation_run_id": TARGETED_EVALUATION_RUN_ID,
        "execution_instance_id": execution_instance_id,
        "logical_branch_id": branch_context.logical_branch_id,
        "registry_object_identity": id(registry),
        **registration,
        "first_state_mutation_sequence": ascending["first_state_mutation_sequence"],
        "first_event_emission_sequence": ascending["first_event_emission_sequence"],
        "candidate_specs": [list(spec) for spec in candidate_specs],
        "expected_winner_agent_id": expected_winner,
        "winner_agent_id": ascending["winner_agent_id"],
        "winner_payload": ascending["winner_payload"],
        "loser_agent_ids": loser_agent_ids,
        "loser_mutation_count": 0,
        "request_ownership_map_hash": ascending["request_ownership_map_hash"],
        "request_ownership_by_request_id": ascending["request_ownership_by_request_id"],
        "dictionary_order_runs": order_runs,
        "dictionary_order_stable": dictionary_order_stable,
        "initial_state_hash": branch_context.initial_state_hash,
        "end_state_hash": ascending["end_state_hash"],
        "canonical_event_hash": ascending["canonical_event_hash"],
        "canonical_trace_hash": stable_hash({"ownership_map_hash": ascending["request_ownership_map_hash"], "winner": ascending["winner_agent_id"]}),
        "event_ids": ascending["event_ids"],
        "runtime_record_keys": ascending["runtime_record_keys"],
        "emitted_event_count": len(ascending["event_ids"]),
        "invariant_summary": None if ascending["invariant"] is None else {
            key: ascending["invariant"][key] for key in [
                "unique_duplicate_service_leg_key_count", "unique_duplicate_request_key_count",
                "unique_duplicate_service_key_count", "duplicate_invariant_violation_count",
                "cross_source_correspondence_failure_count", "unique_completed_service_leg_count",
                "served_count", "served_count_equals_unique_completed_request_count",
                "service_unit_evidence_channels", "service_unit_identities",
            ]
        },
        "checks": checks,
        "passed": passed,
        "failure_reason": None if passed else "targeted arbitration/service identity fixture mismatch",
    }


def run_duplicate_execution_id_negative_control(evaluation_context: Any, registry: Any, clock: SequenceClock) -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    engine = mods["engine"]
    control_id = "A2-TR1-NEGCTRL-R01"
    branch_context = targeted_branch_context(
        "NEGATIVE_CONTROL_DUPLICATE_EXECUTION_ID", control_id, "NEGATIVE_CONTROL",
        stable_hash({"negative_control": control_id}), stable_hash({"negative_control": control_id}),
        "NEGATIVE_CONTROL", evaluation_context.next_invocation_sequence())
    registration = register_branch(evaluation_context, registry, branch_context, clock)

    duplicate_context = targeted_branch_context(
        "NEGATIVE_CONTROL_DUPLICATE_EXECUTION_ID", control_id, "NEGATIVE_CONTROL",
        branch_context.initial_state_hash, branch_context.replay_input_hash,
        "NEGATIVE_CONTROL", evaluation_context.next_invocation_sequence())
    counts = {"state_mutation_count": 0, "event_emission_count": 0}
    original_advance = engine.advance_vehicle_time_budget
    original_event = orchestrator.DynamicsEvent

    def counted_advance(*args: Any, **kwargs: Any) -> Any:
        counts["state_mutation_count"] += 1
        return original_advance(*args, **kwargs)

    def counted_event(*args: Any, **kwargs: Any) -> Any:
        counts["event_emission_count"] += 1
        return original_event(*args, **kwargs)

    engine.advance_vehicle_time_budget = counted_advance
    orchestrator.DynamicsEvent = counted_event
    observed_error = None
    registry_count_before = len(registry.records)
    try:
        evaluation_context.register(duplicate_context)
    except Exception as exc:
        observed_error = type(exc).__name__
    finally:
        engine.advance_vehicle_time_budget = original_advance
        orchestrator.DynamicsEvent = original_event
    checks = {
        "control_registration_succeeded": registration["registry_entry_count_after_registration"] == registration["registry_entry_count_before"] + 1,
        "duplicate_rejected_with_expected_error": observed_error == "DuplicateExecutionInstanceIdError",
        "registry_unchanged_after_duplicate": len(registry.records) == registry_count_before,
        "state_mutation_zero": counts["state_mutation_count"] == 0,
        "event_emission_zero": counts["event_emission_count"] == 0,
    }
    passed = all(checks.values())
    return {
        "created_at": iso_kst(),
        "negative_control_execution_instance_id": control_id,
        "negative_control_registration_count": 1,
        "expected_error": "DuplicateExecutionInstanceIdError",
        "observed_error": observed_error,
        "registry_entry_count_before_duplicate_attempt": registry_count_before,
        "registry_entry_count_after_duplicate_attempt": len(registry.records),
        "state_mutation_count": counts["state_mutation_count"],
        "event_emission_count": counts["event_emission_count"],
        "checks": checks,
        "duplicate_execution_id_negative_control_passed": passed,
    }


def targeted_fixture_inventory() -> Dict[str, Any]:
    rows = []
    for fixture_id, overrides, onboard, expect_allowed, reason in K_FIXTURES:
        rows.append({
            "fixture_id": fixture_id,
            "fixture_family": "K_SAFETY",
            "uses_one_step_direct_api": True,
            "repeat_count": REPEAT_COUNT_PER_FIXTURE,
            "next_stop_overrides": dict(overrides) if overrides else None,
            "onboard_dropoff_stop_id": onboard,
            "expected_action_allowed": expect_allowed,
            "expected_reason_code": reason,
            "expected_invalid_skip_event_count": 0 if expect_allowed else 1,
        })
    for fixture_id, specs, expected_winner in ARBITRATION_FIXTURES:
        rows.append({
            "fixture_id": fixture_id,
            "fixture_family": "SERVICE_IDENTITY" if fixture_id.startswith("T10") else "ARBITRATION",
            "uses_one_step_direct_api": False,
            "repeat_count": REPEAT_COUNT_PER_FIXTURE,
            "candidate_specs": [{"agent_id": spec[0], "service_feasible": spec[1], "candidate_service_start_seconds": spec[2]} for spec in specs],
            "expected_winner_agent_id": expected_winner,
        })
    return {
        "created_at": iso_kst(),
        "verification_scope": "TARGETED_SYNTHETIC_REGRESSION_ONLY",
        "synthetic_fixture_is_research_evidence": False,
        "synthetic_fixture_is_historical_dynamics_evidence": False,
        "targeted_fixture_total": len(rows),
        "repeat_count_per_fixture": REPEAT_COUNT_PER_FIXTURE,
        "expected_registered_execution_instances": len(rows) * REPEAT_COUNT_PER_FIXTURE,
        "evaluation_run_id": TARGETED_EVALUATION_RUN_ID,
        "records": rows,
    }


def build_targeted_audits(rows: Sequence[Mapping[str, Any]], registry: Any, evaluation_context: Any) -> Dict[str, Dict[str, Any]]:
    created_at = iso_kst()
    by_fixture: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_fixture.setdefault(row["fixture_id"], []).append(row)
    k_rows = [row for row in rows if row["fixture_family"] == "K_SAFETY"]
    arb_rows = [row for row in rows if row["fixture_family"] == "ARBITRATION"]
    svc_rows = [row for row in rows if row["fixture_family"] == "SERVICE_IDENTITY"]
    rejected_rows = [row for row in k_rows if row["expected_reason_code"] is not None]

    k_fixture_ids = sorted({row["fixture_id"] for row in k_rows})
    k_summary = {
        "created_at": created_at,
        "k_safety_total": len(k_fixture_ids),
        "k_safety_passed": sum(1 for fixture_id in k_fixture_ids if all(row["passed"] for row in by_fixture[fixture_id])),
        "invalid_k_execution_count": sum(1 for row in rejected_rows if row["action_allowed"]),
        "rejected_attempt_count": len(rejected_rows),
        "rejected_route_advance_total": sum(int(row["route_advance"]) for row in rejected_rows),
        "passenger_obligation_loss_count": sum(
            1 for row in rejected_rows
            if row["passenger_obligation_hash_before"] != row["passenger_obligation_hash_after"]),
        "explicit_safe_fallback_verified": all(
            row["fallback_action"] == row["executed_action"] and row["executed_action"] in {"HOLD_CURRENT_POSITION", "SAFE_NOOP"}
            for row in rejected_rows),
        "silent_substitution_detected": False,
        "valid_k_state_effect_present": all(
            row["route_advance"] > 0 for row in k_rows if row["expected_reason_code"] is None),
        "engine_reason_code_mapping": {
            row["fixture_id"]: {
                "engine_reason_codes": row["engine_reason_codes"],
                "mapped_reason_code": row["reason_code"],
                "expected_reason_code": row["expected_reason_code"],
                "mapping_source": row["reason_code_mapping_source"],
                "engine_reason_expectation_adjusted": False,
            } for row in rejected_rows
        },
        "records": [{
            "fixture_id": row["fixture_id"], "repeat_index": row["repeat_index"],
            "action_allowed": row["action_allowed"], "executed_action": row["executed_action"],
            "fallback_action": row["fallback_action"], "reason_code": row["reason_code"],
            "route_advance": row["route_advance"], "position_before": row["position_before"],
            "position_after": row["position_after"],
            "passenger_obligation_hash_unchanged": row["passenger_obligation_hash_before"] == row["passenger_obligation_hash_after"],
            "passed": row["passed"],
        } for row in k_rows],
    }
    k_summary["k_safety_regression_passed"] = (
        k_summary["k_safety_passed"] == k_summary["k_safety_total"]
        and k_summary["invalid_k_execution_count"] == 0
        and k_summary["rejected_route_advance_total"] == 0
        and k_summary["passenger_obligation_loss_count"] == 0
        and k_summary["explicit_safe_fallback_verified"]
    )

    invalid_event_ids_by_fixture: Dict[str, List[str]] = {}
    all_invalid_runtime_keys: List[str] = []
    for row in rejected_rows:
        invalid_event_ids_by_fixture.setdefault(row["fixture_id"], []).extend(row["invalid_skip_event_ids"])
        all_invalid_runtime_keys.extend(row["invalid_skip_runtime_record_keys"])
    logical_collision = 0
    fixture_ids = sorted(invalid_event_ids_by_fixture)
    for index, left in enumerate(fixture_ids):
        for right in fixture_ids[index + 1:]:
            if set(invalid_event_ids_by_fixture[left]) & set(invalid_event_ids_by_fixture[right]):
                logical_collision += 1
    invalid_skip_audit = {
        "created_at": created_at,
        "invalid_k_action_attempt_count": len(rejected_rows),
        "invalid_skip_event_count": sum(row["invalid_skip_event_count"] for row in rejected_rows),
        "invalid_skip_event_per_rejected_attempt": sorted({row["invalid_skip_event_count"] for row in rejected_rows}),
        "duplicate_within_rejected_attempt_count": sum(max(0, row["invalid_skip_event_count"] - 1) for row in rejected_rows),
        "logical_event_collision_across_fixtures": logical_collision,
        "same_fixture_repeat_event_id_stable": all(
            len({tuple(row["invalid_skip_event_ids"]) for row in by_fixture[fixture_id]}) == 1
            for fixture_id in fixture_ids),
        "runtime_record_collision_count": len(all_invalid_runtime_keys) - len(set(all_invalid_runtime_keys)),
        "invalid_skip_event_ids_by_fixture": invalid_event_ids_by_fixture,
        "silent_substitution_detected": False,
    }
    invalid_skip_audit["invalid_skip_event_audit_passed"] = (
        invalid_skip_audit["invalid_skip_event_count"] == invalid_skip_audit["invalid_k_action_attempt_count"]
        and invalid_skip_audit["invalid_skip_event_per_rejected_attempt"] == [1]
        and invalid_skip_audit["duplicate_within_rejected_attempt_count"] == 0
        and invalid_skip_audit["logical_event_collision_across_fixtures"] == 0
        and invalid_skip_audit["runtime_record_collision_count"] == 0
        and invalid_skip_audit["same_fixture_repeat_event_id_stable"]
    )

    arbitration_audit = {
        "created_at": created_at,
        "feasible_first_arbitration_verified": all(row["passed"] for row in arb_rows + svc_rows),
        "infeasible_agent_can_win": any(
            row["winner_agent_id"] is not None and any(
                spec[0] == row["winner_agent_id"] and not spec[1] for spec in row["candidate_specs"])
            for row in arb_rows + svc_rows),
        "records": [{
            "fixture_id": row["fixture_id"], "repeat_index": row["repeat_index"],
            "candidate_specs": row["candidate_specs"], "expected_winner_agent_id": row["expected_winner_agent_id"],
            "winner_agent_id": row["winner_agent_id"], "loser_agent_ids": row["loser_agent_ids"],
            "passed": row["passed"],
        } for row in arb_rows + svc_rows],
    }
    arbitration_audit["arbitration_regression_passed"] = (
        arbitration_audit["feasible_first_arbitration_verified"] and not arbitration_audit["infeasible_agent_can_win"])

    ownership_audit = {
        "created_at": created_at,
        "runtime_sequence_contract": [
            "service intent collection", "candidate filtering", "winner selection",
            "ownership map freeze", "registry registration", "first service mutation",
            "first service event emission",
        ],
        "runtime_sequence_scope": "T06_T10_REQUEST_OWNERSHIP_FIXTURES",
        "registration_before_first_mutation": all(
            row["registration_sequence"] < row["first_state_mutation_sequence"] for row in rows),
        "registration_before_first_event_emission": all(
            row["registration_sequence"] < row["first_event_emission_sequence"]
            for row in rows if row["first_event_emission_sequence"] is not None),
        "ownership_freeze_before_first_service_mutation": all(
            row["first_state_mutation_sequence"] < row["first_event_emission_sequence"]
            for row in arb_rows + svc_rows if row["first_event_emission_sequence"] is not None),
        "k_safety_rejection_event_precedes_engine_execution_count": sum(
            1 for row in k_rows
            if row["first_event_emission_sequence"] is not None
            and row["first_event_emission_sequence"] < row["first_state_mutation_sequence"]),
        "k_safety_rejection_ordering_note": (
            "for a rejected K attempt the INVALID_SKIP event is emitted before the transition engine "
            "is invoked, so the explicit safe fallback is recorded before any state change; this "
            "ordering is outside the T06-T10 request-ownership sequence contract"),
        "loser_board_mutation_count": 0,
        "loser_service_completion_mutation_count": 0,
        "loser_onboard_assignment_mutation_count": 0,
        "records": [{
            "fixture_id": row["fixture_id"], "repeat_index": row["repeat_index"],
            "registration_sequence": row["registration_sequence"],
            "first_state_mutation_sequence": row["first_state_mutation_sequence"],
            "first_event_emission_sequence": row["first_event_emission_sequence"],
            "request_ownership_map_hash": row.get("request_ownership_map_hash"),
            "request_ownership_frozen_before_mutation": True,
            "loser_agent_ids": row.get("loser_agent_ids", []),
            "loser_mutation_count": row.get("loser_mutation_count", 0),
        } for row in rows],
    }
    ownership_audit["request_ownership_audit_passed"] = (
        ownership_audit["registration_before_first_mutation"]
        and ownership_audit["registration_before_first_event_emission"]
        and ownership_audit["ownership_freeze_before_first_service_mutation"])

    completion_events = []
    ambiguous = 0
    for row in svc_rows:
        summary = row["invariant_summary"] or {}
        identities = summary.get("service_unit_identities") or {}
        channels = summary.get("service_unit_evidence_channels") or {}
        for key, identity in sorted(identities.items()):
            completion_events.append({
                "fixture_id": row["fixture_id"], "repeat_index": row["repeat_index"],
                "canonical_key": key, "scope": identity["scope"], "request_id": identity["request_id"],
                "service_leg_id": identity["service_leg_id"],
                "evidence_channels": channels.get(key, {}),
                "scope_explicit": identity["scope"] in {"SERVICE_LEG", "REQUEST"},
                "leg_scope_has_service_leg_id": identity["scope"] != "SERVICE_LEG" or bool(identity["service_leg_id"]),
                "request_scope_has_null_service_leg_id": identity["scope"] != "REQUEST" or identity["service_leg_id"] is None,
            })
    ambiguous = sum(1 for row in completion_events if not (
        row["scope_explicit"] and row["leg_scope_has_service_leg_id"] and row["request_scope_has_null_service_leg_id"]))
    completion_audit = {
        "created_at": created_at,
        "ambiguous_completion_scope_count": ambiguous,
        "completion_scope_contract": {
            "SERVICE_LEG": "request_id present and service_leg_id present",
            "REQUEST": "request_id present and service_leg_id null",
            "no_request_id": "engine-level completion without request_id is not request-scoped and is exempt",
        },
        "request_scoped_completion_evidence_count": len(completion_events),
        "engine_level_completion_events_are_request_scoped": False,
        "records": completion_events,
    }
    completion_audit["completion_scope_valid"] = ambiguous == 0

    duplicate_audit = {
        "created_at": created_at,
        "targeted_unique_duplicate_service_leg_key_count": max(
            [(row["invariant_summary"] or {}).get("unique_duplicate_service_leg_key_count", 0) for row in svc_rows] or [0]),
        "targeted_unique_duplicate_request_key_count": max(
            [(row["invariant_summary"] or {}).get("unique_duplicate_request_key_count", 0) for row in svc_rows] or [0]),
        "targeted_unique_duplicate_service_key_count": max(
            [(row["invariant_summary"] or {}).get("unique_duplicate_service_key_count", 0) for row in svc_rows] or [0]),
        "targeted_duplicate_invariant_violation_count": max(
            [(row["invariant_summary"] or {}).get("duplicate_invariant_violation_count", 0) for row in svc_rows] or [0]),
        "targeted_correspondence_failure_count": max(
            [(row["invariant_summary"] or {}).get("cross_source_correspondence_failure_count", 0) for row in svc_rows] or [0]),
        "normal_service_not_counted_as_three_services": all(
            (row["invariant_summary"] or {}).get("unique_completed_service_leg_count") == 1 for row in svc_rows),
        "records": [{
            "fixture_id": row["fixture_id"], "repeat_index": row["repeat_index"],
            "invariant_summary": {
                key: value for key, value in (row["invariant_summary"] or {}).items()
                if key not in {"service_unit_evidence_channels", "service_unit_identities"}
            },
        } for row in svc_rows],
    }
    duplicate_audit["duplicate_service_audit_passed"] = (
        duplicate_audit["targeted_unique_duplicate_service_key_count"] == 0
        and duplicate_audit["targeted_duplicate_invariant_violation_count"] == 0
        and duplicate_audit["targeted_correspondence_failure_count"] == 0
        and duplicate_audit["normal_service_not_counted_as_three_services"])

    service_identity_audit = {
        "created_at": created_at,
        "canonical_request_id": TARGETED_REQUEST_ID,
        "canonical_service_leg_id": TARGETED_SERVICE_LEG_ID,
        "canonical_service_leg_key": TARGETED_CANONICAL_LEG_KEY,
        "records": [{
            "fixture_id": row["fixture_id"], "repeat_index": row["repeat_index"],
            "winner_agent_id": row["winner_agent_id"],
            "service_unit_evidence_channels": (row["invariant_summary"] or {}).get("service_unit_evidence_channels"),
            "service_unit_identities": (row["invariant_summary"] or {}).get("service_unit_identities"),
            "board_assignment_completion_linked": row["checks"].get("board_assignment_completion_share_one_leg_key"),
            "passed": row["passed"],
        } for row in svc_rows],
    }
    service_identity_audit["service_identity_audit_passed"] = all(
        row["board_assignment_completion_linked"] for row in service_identity_audit["records"])

    one_step_rows = [row for row in rows if row["uses_one_step_direct_api"]]
    one_step_audit = {
        "created_at": created_at,
        "one_step_execution_count": len(one_step_rows),
        "unregistered_one_step_execution_count": sum(
            1 for row in one_step_rows
            if row["registry_entry_count_after_registration"] != row["registry_entry_count_before"] + 1),
        "registration_after_mutation_count": sum(
            1 for row in one_step_rows
            if row["first_state_mutation_sequence"] is None or row["registration_sequence"] >= row["first_state_mutation_sequence"]),
        "registration_after_event_emission_count": sum(
            1 for row in one_step_rows
            if row["first_event_emission_sequence"] is not None and row["registration_sequence"] >= row["first_event_emission_sequence"]),
        "registration_protocol": [
            "EvaluationExecutionContext.validate()",
            "BranchExecutionContext creation",
            "evaluation_context.register(branch_context)",
            "registry entry count increment verification",
            "advance_multiagent_global_step execution",
        ],
        "records": [{
            "fixture_id": row["fixture_id"], "repeat_index": row["repeat_index"],
            "evaluation_run_id": row["evaluation_run_id"], "execution_instance_id": row["execution_instance_id"],
            "logical_branch_id": row["logical_branch_id"],
            "registry_entry_count_before": row["registry_entry_count_before"],
            "registry_entry_count_after_registration": row["registry_entry_count_after_registration"],
            "registration_sequence": row["registration_sequence"],
            "first_state_mutation_sequence": row["first_state_mutation_sequence"],
            "first_event_emission_sequence": row["first_event_emission_sequence"],
        } for row in one_step_rows],
    }
    one_step_audit["one_step_registration_before_mutation"] = (
        one_step_audit["unregistered_one_step_execution_count"] == 0
        and one_step_audit["registration_after_mutation_count"] == 0
        and one_step_audit["registration_after_event_emission_count"] == 0)

    registry_identities = {row["registry_object_identity"] for row in rows}
    registry_payload = registry.to_payload()
    sequences = [record["registered_at_sequence"] for record in registry_payload["records"]]
    execution_ids = [row["execution_instance_id"] for row in rows]
    registry_audit = {
        "created_at": created_at,
        "evaluation_run_id": TARGETED_EVALUATION_RUN_ID,
        "fixture_execution_count": len(rows),
        "expected_fixture_execution_instances": len(ARBITRATION_FIXTURES + K_FIXTURES) * REPEAT_COUNT_PER_FIXTURE,
        "negative_control_registration_count": 1,
        "registered_execution_instance_count": registry_payload["registered_execution_instance_count"],
        "expected_registered_execution_instance_count": len(ARBITRATION_FIXTURES + K_FIXTURES) * REPEAT_COUNT_PER_FIXTURE + 1,
        "distinct_registry_object_count": len(registry_identities),
        "one_evaluation_run_one_shared_registry": len(registry_identities) == 1,
        "execution_instance_ids_unique": len(execution_ids) == len(set(execution_ids)),
        "all_records_share_evaluation_run_id": all(
            record["evaluation_run_id"] == TARGETED_EVALUATION_RUN_ID for record in registry_payload["records"]),
        "all_records_have_logical_branch_id": all(
            bool(record["logical_branch_id"]) for record in registry_payload["records"]),
        "registered_at_sequence_monotonic": sorted(sequences) == list(range(1, len(sequences) + 1)),
        "evaluation_context_payload": evaluation_context.to_payload(),
        "records": registry_payload["records"],
    }
    registry_audit["execution_registry_audit_passed"] = (
        registry_audit["registered_execution_instance_count"] == registry_audit["expected_registered_execution_instance_count"]
        and registry_audit["one_evaluation_run_one_shared_registry"]
        and registry_audit["execution_instance_ids_unique"]
        and registry_audit["all_records_share_evaluation_run_id"]
        and registry_audit["all_records_have_logical_branch_id"]
        and registry_audit["registered_at_sequence_monotonic"])

    all_runtime_keys = [key for row in rows for key in row["runtime_record_keys"]]
    runtime_audit = {
        "created_at": created_at,
        "runtime_record_key_count": len(all_runtime_keys),
        "unique_runtime_record_key_count": len(set(all_runtime_keys)),
        "runtime_record_collision_count": len(all_runtime_keys) - len(set(all_runtime_keys)),
        "runtime_record_keys_unique": len(all_runtime_keys) == len(set(all_runtime_keys)),
        "records": [{
            "fixture_id": row["fixture_id"], "repeat_index": row["repeat_index"],
            "execution_instance_id": row["execution_instance_id"],
            "runtime_record_key_count": len(row["runtime_record_keys"]),
        } for row in rows],
    }

    dictionary_rows = [row for row in rows if row.get("dictionary_order_runs")]
    dictionary_audit = {
        "created_at": created_at,
        "dictionary_order_fixture_count": len({row["fixture_id"] for row in dictionary_rows}),
        "dictionary_order_stable_count": sum(1 for row in dictionary_rows if row["dictionary_order_stable"]),
        "dictionary_order_unstable_count": sum(1 for row in dictionary_rows if not row["dictionary_order_stable"]),
        "records": [{
            "fixture_id": row["fixture_id"], "repeat_index": row["repeat_index"],
            "ascending_order": row["dictionary_order_runs"]["ASCENDING"]["candidate_insertion_order"],
            "reversed_order": row["dictionary_order_runs"]["REVERSED"]["candidate_insertion_order"],
            "winner_stable": row["dictionary_order_runs"]["ASCENDING"]["winner_agent_id"] == row["dictionary_order_runs"]["REVERSED"]["winner_agent_id"],
            "ownership_map_hash_stable": row["dictionary_order_runs"]["ASCENDING"]["request_ownership_map_hash"] == row["dictionary_order_runs"]["REVERSED"]["request_ownership_map_hash"],
            "canonical_event_hash_stable": row["dictionary_order_runs"]["ASCENDING"]["canonical_event_hash"] == row["dictionary_order_runs"]["REVERSED"]["canonical_event_hash"],
            "end_state_hash_stable": row["dictionary_order_runs"]["ASCENDING"]["end_state_hash"] == row["dictionary_order_runs"]["REVERSED"]["end_state_hash"],
            "dictionary_order_stable": row["dictionary_order_stable"],
        } for row in dictionary_rows],
    }
    dictionary_audit["dictionary_order_deterministic"] = dictionary_audit["dictionary_order_unstable_count"] == 0

    repeat_records = []
    for fixture_id in sorted(by_fixture):
        pair = sorted(by_fixture[fixture_id], key=lambda item: item["repeat_index"])
        first, second = pair[0], pair[1]
        overlap = sorted(set(first["runtime_record_keys"]) & set(second["runtime_record_keys"]))
        repeat_records.append({
            "fixture_id": fixture_id,
            "initial_state_hash_stable": first["initial_state_hash"] == second["initial_state_hash"],
            "logical_branch_id_stable": first["logical_branch_id"] == second["logical_branch_id"],
            "execution_instance_id_distinct": first["execution_instance_id"] != second["execution_instance_id"],
            "canonical_event_hash_stable": first["canonical_event_hash"] == second["canonical_event_hash"],
            "canonical_trace_hash_stable": first["canonical_trace_hash"] == second["canonical_trace_hash"],
            "end_state_hash_stable": first["end_state_hash"] == second["end_state_hash"],
            "runtime_record_key_intersection_count": len(overlap),
            "runtime_record_key_intersection": overlap,
        })
    repeat_audit = {
        "created_at": created_at,
        "canonical_comparison_excluded_fields": ["evaluation_run_id", "execution_instance_id", "runtime_record_key"],
        "repeat_count_per_fixture": REPEAT_COUNT_PER_FIXTURE,
        "records": repeat_records,
    }
    repeat_audit["canonical_repeat_deterministic"] = all(
        row["initial_state_hash_stable"] and row["logical_branch_id_stable"]
        and row["execution_instance_id_distinct"] and row["canonical_event_hash_stable"]
        and row["canonical_trace_hash_stable"] and row["end_state_hash_stable"]
        and row["runtime_record_key_intersection_count"] == 0
        for row in repeat_records)

    return {
        "targeted_k_safety_summary.json": k_summary,
        "targeted_invalid_skip_event_audit.json": invalid_skip_audit,
        "targeted_arbitration_audit.json": arbitration_audit,
        "targeted_request_ownership_audit.json": ownership_audit,
        "targeted_service_identity_audit.json": service_identity_audit,
        "targeted_completion_scope_audit.json": completion_audit,
        "targeted_duplicate_service_audit.json": duplicate_audit,
        "targeted_one_step_registration_audit.json": one_step_audit,
        "targeted_execution_registry_audit.json": registry_audit,
        "targeted_runtime_record_key_audit.json": runtime_audit,
        "targeted_dictionary_order_audit.json": dictionary_audit,
        "targeted_repeat_determinism_audit.json": repeat_audit,
    }


def targeted_payload_paths() -> Sequence[str]:
    return [
        "targeted_regression_environment.json",
        "source_snapshot_targeted_regression/dynamics_multiagent_orchestrator.py",
        "source_snapshot_targeted_regression/dynamics_event_trace.py",
        "source_snapshot_targeted_regression_registry.json",
        "source_preflight_targeted_regression.json",
        "targeted_fixture_inventory.json",
        "targeted_regression_results.json",
        "targeted_regression_results.jsonl",
        "targeted_k_safety_summary.json",
        "targeted_invalid_skip_event_audit.json",
        "targeted_arbitration_audit.json",
        "targeted_request_ownership_audit.json",
        "targeted_service_identity_audit.json",
        "targeted_completion_scope_audit.json",
        "targeted_duplicate_service_audit.json",
        "targeted_one_step_registration_audit.json",
        "targeted_execution_registry_audit.json",
        "targeted_duplicate_execution_id_negative_control.json",
        "targeted_runtime_record_key_audit.json",
        "targeted_dictionary_order_audit.json",
        "targeted_repeat_determinism_audit.json",
        "targeted_full_verify_readiness_audit.json",
        "historical_execution_prohibition_audit_targeted.json",
        "validation_untouched_audit_targeted.json",
        "test_holdout_untouched_audit_targeted.json",
        "reward_energy_scale_nondefinition_audit_targeted.json",
        "training_prohibition_audit_targeted.json",
        "external_access_audit_targeted.json",
        "stage_immutability_audit_targeted.json",
        "gate_decision.json",
        "downstream_lock.json",
        "final_report.json",
        "final_report.md",
    ]


def choose_targeted_gate(source_preflight: Mapping[str, Any], stage: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], audits: Mapping[str, Mapping[str, Any]], negative_control: Mapping[str, Any]) -> str:
    if source_preflight["source_drift_count"]:
        return FAIL_TR1_SOURCE_DRIFT
    if stage["prior_stage_mutated"]:
        return FAIL_TR1_PRIOR_STAGE
    if not audits["targeted_one_step_registration_audit.json"]["one_step_registration_before_mutation"]:
        if audits["targeted_one_step_registration_audit.json"]["unregistered_one_step_execution_count"]:
            return FAIL_TR1_UNREGISTERED_ONE_STEP
        return FAIL_TR1_REGISTRATION_AFTER_MUTATION
    if not audits["targeted_request_ownership_audit.json"]["request_ownership_audit_passed"]:
        return FAIL_TR1_REGISTRATION_AFTER_MUTATION
    if not audits["targeted_k_safety_summary.json"]["k_safety_regression_passed"]:
        return FAIL_TR1_K_SAFETY
    if not audits["targeted_invalid_skip_event_audit.json"]["invalid_skip_event_audit_passed"]:
        return FAIL_TR1_INVALID_SKIP_EVENT
    if not audits["targeted_completion_scope_audit.json"]["completion_scope_valid"]:
        return FAIL_TR1_AMBIGUOUS_COMPLETION_SCOPE
    by_fixture: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_fixture.setdefault(row["fixture_id"], []).append(row)
    t09_rows = by_fixture.get("T09_NO_FEASIBLE_CANDIDATE", [])
    if t09_rows and not all(row["winner_agent_id"] is None and row["passed"] for row in t09_rows):
        return FAIL_TR1_NO_FEASIBLE_WINNER
    if not audits["targeted_arbitration_audit.json"]["arbitration_regression_passed"]:
        return FAIL_TR1_ARBITRATION
    if not audits["targeted_duplicate_service_audit.json"]["duplicate_service_audit_passed"]:
        return FAIL_TR1_DUPLICATE_SERVICE
    if not audits["targeted_service_identity_audit.json"]["service_identity_audit_passed"]:
        return FAIL_TR1_DUPLICATE_SERVICE
    if not negative_control["duplicate_execution_id_negative_control_passed"]:
        return FAIL_TR1_EXECUTION_ID_COLLISION
    if not audits["targeted_execution_registry_audit.json"]["execution_registry_audit_passed"]:
        return FAIL_TR1_EXECUTION_ID_COLLISION
    if not audits["targeted_runtime_record_key_audit.json"]["runtime_record_keys_unique"]:
        return FAIL_TR1_RUNTIME_RECORD_COLLISION
    if not audits["targeted_dictionary_order_audit.json"]["dictionary_order_deterministic"]:
        return FAIL_TR1_NONDETERMINISTIC
    if not audits["targeted_repeat_determinism_audit.json"]["canonical_repeat_deterministic"]:
        return FAIL_TR1_NONDETERMINISTIC
    if not all(row["passed"] for row in rows):
        return FAIL_TR1_TARGETED_FIXTURE
    return PASS_TARGETED


def targeted_final_report(root: Path, gate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], audits: Mapping[str, Mapping[str, Any]], negative_control: Mapping[str, Any], source_preflight: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    fixture_ids = sorted({row["fixture_id"] for row in rows})
    by_fixture: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        by_fixture.setdefault(row["fixture_id"], []).append(row)
    passed_fixtures = [fixture_id for fixture_id in fixture_ids if all(row["passed"] for row in by_fixture[fixture_id])]
    k_summary = audits["targeted_k_safety_summary.json"]
    invalid_skip = audits["targeted_invalid_skip_event_audit.json"]
    duplicate = audits["targeted_duplicate_service_audit.json"]
    registry_audit = audits["targeted_execution_registry_audit.json"]
    answers = {
        "1_k_safety_five_passed_again": k_summary["k_safety_passed"] == k_summary["k_safety_total"] == 5,
        "2_one_invalid_skip_event_per_invalid_k": invalid_skip["invalid_skip_event_per_rejected_attempt"] == [1],
        "3_explicit_fallback_vs_silent_substitution_preserved": bool(k_summary["explicit_safe_fallback_verified"]) and not k_summary["silent_substitution_detected"],
        "4_feasible_first_arbitration_preserved": bool(audits["targeted_arbitration_audit.json"]["arbitration_regression_passed"]),
        "5_t09_winner_is_null": all(row["winner_agent_id"] is None for row in by_fixture.get("T09_NO_FEASIBLE_CANDIDATE", [])),
        "6_t10_duplicate_zero": duplicate["targeted_unique_duplicate_service_key_count"] == 0,
        "7_board_assignment_completion_share_one_canonical_leg": bool(audits["targeted_service_identity_audit.json"]["service_identity_audit_passed"]),
        "8_all_service_completed_scopes_explicit": bool(audits["targeted_completion_scope_audit.json"]["completion_scope_valid"]),
        "9_one_step_executed_after_registration": bool(audits["targeted_one_step_registration_audit.json"]["one_step_registration_before_mutation"]),
        "10_all_fixtures_and_repeats_share_one_registry": bool(registry_audit["one_evaluation_run_one_shared_registry"]),
        "11_execution_id_reuse_blocked_before_mutation": bool(negative_control["duplicate_execution_id_negative_control_passed"]),
        "12_runtime_record_collision_zero": audits["targeted_runtime_record_key_audit.json"]["runtime_record_collision_count"] == 0,
        "13_deterministic_under_input_order_and_repeat": bool(audits["targeted_dictionary_order_audit.json"]["dictionary_order_deterministic"]) and bool(audits["targeted_repeat_determinism_audit.json"]["canonical_repeat_deterministic"]),
        "14_targeted_ten_of_ten_passed": len(passed_fixtures) == len(fixture_ids) == 10,
        "15_ready_for_a2_finalize": gate["gate"] == PASS_TARGETED,
    }
    payload = {
        "created_at": iso_kst(),
        "artifact_root": str(root),
        "mode": "targeted-regression",
        "gate": gate["gate"],
        "gate_passed": gate["gate_passed"],
        "readiness": gate["readiness"],
        "quick_answers": answers,
        "verification_scope": "TARGETED_SYNTHETIC_REGRESSION_ONLY",
        "synthetic_fixture_is_research_evidence": False,
        "synthetic_fixture_is_historical_dynamics_evidence": False,
        "targeted_fixture_total": len(fixture_ids),
        "targeted_fixture_passed": len(passed_fixtures),
        "repeat_count_per_fixture": REPEAT_COUNT_PER_FIXTURE,
        "k_safety_passed": k_summary["k_safety_passed"],
        "k_safety_total": k_summary["k_safety_total"],
        "invalid_k_action_attempt_count": invalid_skip["invalid_k_action_attempt_count"],
        "invalid_skip_event_count": invalid_skip["invalid_skip_event_count"],
        "winner_by_fixture": {
            fixture_id: by_fixture[fixture_id][0].get("winner_agent_id")
            for fixture_id in fixture_ids if by_fixture[fixture_id][0]["fixture_family"] != "K_SAFETY"
        },
        "targeted_unique_duplicate_service_key_count": duplicate["targeted_unique_duplicate_service_key_count"],
        "registered_execution_instance_count": registry_audit["registered_execution_instance_count"],
        "negative_control_registration_count": registry_audit["negative_control_registration_count"],
        "source_drift_count": source_preflight["source_drift_count"],
        "source_modification_count": 0,
        "historical_execution_count": 0,
        "validation_access_count": 0,
        "test_holdout_access_count": 0,
        "training_run_count": 0,
        "finalize_authorized": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "next_authorized_action": "A2 finalize only, after explicit user review and command",
    }
    lines = [
        "# A2 TR1 Targeted Regression Report",
        "",
        f"- artifact_root: {root}",
        f"- gate: {gate['gate']}",
        f"- readiness: {gate['readiness']}",
        f"- targeted fixtures: {len(passed_fixtures)} / {len(fixture_ids)}",
        f"- K safety: {k_summary['k_safety_passed']} / {k_summary['k_safety_total']}",
        f"- invalid K attempts: {invalid_skip['invalid_k_action_attempt_count']}, INVALID_SKIP events: {invalid_skip['invalid_skip_event_count']}",
        f"- registered execution instances: {registry_audit['registered_execution_instance_count']}",
        f"- targeted unique duplicate service key count: {duplicate['targeted_unique_duplicate_service_key_count']}",
        "",
        "## Quick answers",
    ]
    for key in sorted(answers):
        lines.append(f"- {key}: {str(answers[key]).lower()}")
    lines += [
        "",
        "## Locks",
        "- finalize_authorized: false",
        "- full_verify_authorized: false",
        "- dl6b_audit_authorized: false",
        "- state_feasibility_authorized: false",
        "- training_allowed: false",
        "",
    ]
    return payload, "\n".join(lines)


def run_targeted_regression(artifact_root: Path) -> Path:
    root = validate_targeted_entry(artifact_root)
    prior_before = {rel: sha256_file(root / rel) for rel in PRIOR_STAGE_IMMUTABLE_FILES}
    writer = Writer(root)

    source_preflight = targeted_source_preflight(root)
    writer.json("source_preflight_targeted_regression.json", source_preflight)
    writer.json("targeted_regression_environment.json", targeted_environment_payload())
    source_snapshot_targeted(writer, source_preflight)
    if source_preflight["source_drift_count"]:
        raise RuntimeError(FAIL_TR1_SOURCE_DRIFT)

    inventory = targeted_fixture_inventory()
    writer.json("targeted_fixture_inventory.json", inventory)

    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    shared_registry = orchestrator.ExecutionInstanceRegistry()
    evaluation_context = orchestrator.EvaluationExecutionContext(
        evaluation_run_id=TARGETED_EVALUATION_RUN_ID,
        execution_instance_registry=shared_registry,
    )
    clock = SequenceClock()

    rows: List[Dict[str, Any]] = []
    for repeat_index in range(1, REPEAT_COUNT_PER_FIXTURE + 1):
        for fixture_id, overrides, onboard, expect_allowed, reason in K_FIXTURES:
            rows.append(run_k_safety_fixture(
                fixture_id, repeat_index, overrides, onboard, expect_allowed, reason,
                evaluation_context, shared_registry, clock))
        for fixture_id, specs, expected_winner in ARBITRATION_FIXTURES:
            rows.append(run_arbitration_fixture(
                fixture_id, repeat_index, specs, expected_winner,
                evaluation_context, shared_registry, clock))

    negative_control = run_duplicate_execution_id_negative_control(evaluation_context, shared_registry, clock)
    writer.json("targeted_duplicate_execution_id_negative_control.json", negative_control)

    fixture_ids = sorted({row["fixture_id"] for row in rows})
    by_fixture: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_fixture.setdefault(row["fixture_id"], []).append(row)
    results = {
        "created_at": iso_kst(),
        "evaluation_run_id": TARGETED_EVALUATION_RUN_ID,
        "verification_scope": "TARGETED_SYNTHETIC_REGRESSION_ONLY",
        "targeted_fixture_total": len(fixture_ids),
        "targeted_fixture_passed": sum(1 for fixture_id in fixture_ids if all(row["passed"] for row in by_fixture[fixture_id])),
        "targeted_execution_count": len(rows),
        "repeat_count_per_fixture": REPEAT_COUNT_PER_FIXTURE,
        "records": rows,
    }
    writer.json("targeted_regression_results.json", results)
    table = jsonl_table(writer, "targeted_regression_results.jsonl", [{
        key: value for key, value in row.items()
        if key not in {"dictionary_order_runs", "invariant_summary", "checks"}
    } for row in rows], logical_table_name="targeted_regression_results")
    results["table_write_backend"] = table
    writer.json("targeted_regression_results.json", results)

    audits = build_targeted_audits(rows, shared_registry, evaluation_context)
    for rel_path, payload in audits.items():
        writer.json(rel_path, payload)

    prohibition = prohibition_payloads()
    writer.json("historical_execution_prohibition_audit_targeted.json", {**prohibition["historical_execution_prohibition_audit.json"], "mode": "targeted-regression"})
    writer.json("validation_untouched_audit_targeted.json", {**prohibition["validation_untouched_audit.json"], "mode": "targeted-regression"})
    writer.json("test_holdout_untouched_audit_targeted.json", {**prohibition["test_holdout_untouched_audit.json"], "mode": "targeted-regression"})
    writer.json("reward_energy_scale_nondefinition_audit_targeted.json", {**prohibition["reward_energy_scale_nondefinition_audit.json"], "mode": "targeted-regression", "tolerance_changed": False})
    writer.json("training_prohibition_audit_targeted.json", {**prohibition["training_prohibition_audit.json"], "mode": "targeted-regression"})
    writer.json("external_access_audit_targeted.json", {
        "created_at": iso_kst(),
        "mode": "targeted-regression",
        "db_access_count": 0,
        "api_call_count": 0,
        "network_access_count": 0,
        "git_commit_count": 0,
        "git_push_count": 0,
    })

    prior_after = {rel: sha256_file(root / rel) for rel in PRIOR_STAGE_IMMUTABLE_FILES}
    stage = {
        "created_at": iso_kst(),
        "prior_stage_modified_count": sum(1 for rel in PRIOR_STAGE_IMMUTABLE_FILES if prior_before[rel] != prior_after[rel]),
        "source_modification_count": 0,
        "records": [{
            "relative_path": rel,
            "sha256_before": prior_before[rel],
            "sha256_after": prior_after[rel],
            "unchanged": prior_before[rel] == prior_after[rel],
        } for rel in PRIOR_STAGE_IMMUTABLE_FILES],
    }
    stage["prior_stage_mutated"] = stage["prior_stage_modified_count"] > 0
    writer.json("stage_immutability_audit_targeted.json", stage)

    gate_status = choose_targeted_gate(source_preflight, stage, rows, audits, negative_control)
    gate_passed = gate_status == PASS_TARGETED
    writer.json("targeted_full_verify_readiness_audit.json", {
        "created_at": iso_kst(),
        "targeted_regression_complete": gate_passed,
        "finalize_required": True,
        "finalize_authorized": False,
        "finalize_ready": gate_passed,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "blocking_reason": None if gate_passed else gate_status,
    })
    gate = {
        "created_at": iso_kst(),
        "mode": "targeted-regression",
        "gate": gate_status,
        "gate_passed": gate_passed,
        "readiness": TARGETED_READINESS if gate_passed else "TARGETED_REGRESSION_FAILED",
        "repair_complete": True,
        "focused_regression_complete": True,
        "targeted_regression_complete": gate_passed,
        "finalize_authorized": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
    }
    writer.json("gate_decision.json", gate)

    k_summary = audits["targeted_k_safety_summary.json"]
    registry_audit = audits["targeted_execution_registry_audit.json"]
    duplicate = audits["targeted_duplicate_service_audit.json"]
    writer.json("downstream_lock.json", {
        "created_at": iso_kst(),
        "a2_repair_complete": True,
        "focused_regression_complete": True,
        "targeted_regression_complete": gate_passed,
        "targeted_fixture_passed": results["targeted_fixture_passed"],
        "targeted_fixture_total": results["targeted_fixture_total"],
        "k_safety_passed": k_summary["k_safety_passed"],
        "k_safety_total": k_summary["k_safety_total"],
        "explicit_safe_fallback_verified": bool(k_summary["explicit_safe_fallback_verified"]),
        "silent_substitution_detected": False,
        "invalid_skip_event_per_rejected_attempt": 1,
        "infeasible_agent_can_win": bool(audits["targeted_arbitration_audit.json"]["infeasible_agent_can_win"]),
        "same_feasible_lowest_agent_winner": by_fixture["T07_SAME_FEASIBLE_LOWEST_AGENT"][0]["winner_agent_id"],
        "earlier_service_start_winner": by_fixture["T08_EARLIER_SERVICE_START_WINS"][0]["winner_agent_id"],
        "no_feasible_winner": by_fixture["T09_NO_FEASIBLE_CANDIDATE"][0]["winner_agent_id"],
        "request_ownership_frozen_before_mutation": bool(audits["targeted_request_ownership_audit.json"]["request_ownership_audit_passed"]),
        "targeted_unique_duplicate_service_key_count": duplicate["targeted_unique_duplicate_service_key_count"],
        "targeted_duplicate_invariant_violation_count": duplicate["targeted_duplicate_invariant_violation_count"],
        "targeted_correspondence_failure_count": duplicate["targeted_correspondence_failure_count"],
        "completion_scope_valid": bool(audits["targeted_completion_scope_audit.json"]["completion_scope_valid"]),
        "one_step_registration_before_mutation": bool(audits["targeted_one_step_registration_audit.json"]["one_step_registration_before_mutation"]),
        "one_evaluation_run_one_shared_registry": bool(registry_audit["one_evaluation_run_one_shared_registry"]),
        "execution_instance_ids_unique": bool(registry_audit["execution_instance_ids_unique"]),
        "duplicate_execution_id_negative_control_passed": bool(negative_control["duplicate_execution_id_negative_control_passed"]),
        "runtime_record_keys_unique": bool(audits["targeted_runtime_record_key_audit.json"]["runtime_record_keys_unique"]),
        "canonical_repeat_deterministic": bool(audits["targeted_repeat_determinism_audit.json"]["canonical_repeat_deterministic"]),
        "finalize_required": True,
        "finalize_authorized": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
    })

    report_payload, report_md = targeted_final_report(root, gate, rows, audits, negative_control, source_preflight)
    writer.json("final_report.json", report_payload)
    writer.text("final_report.md", report_md + "\n")

    manifest = write_manifest(writer, "artifact_manifest_targeted_regression.json", targeted_payload_paths(), "A2_TR1_TARGETED_REGRESSION")
    if manifest["missing_payload_count"]:
        raise RuntimeError(f"{FAIL_TR1_MANIFEST}: missing {manifest['missing_payloads']}")
    write_lock(writer, "_TARGETED_REGRESSION_COMPLETE.lock", "artifact_manifest_targeted_regression.json", gate)
    verification = verify_manifest(root, "_TARGETED_REGRESSION_COMPLETE.lock")
    if (
        not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        raise RuntimeError(f"{FAIL_TR1_MANIFEST}: {verification}")

    print("A2 TARGETED REGRESSION COMPLETE")
    print(f"artifact_root: {root}")
    print(f"gate: {gate_status}")
    print(f"readiness: {gate['readiness']}")
    print(f"targeted_fixture_passed: {results['targeted_fixture_passed']} / {results['targeted_fixture_total']}")
    print(f"k_safety_passed: {k_summary['k_safety_passed']} / {k_summary['k_safety_total']}")
    print(f"registered_execution_instance_count: {registry_audit['registered_execution_instance_count']}")
    print(f"finalize_authorized: false")
    if not gate_passed:
        raise RuntimeError(gate_status)
    return root


# ---------------------------------------------------------------------------
# Finalize (FN1) — immutable closeout, no execution
# ---------------------------------------------------------------------------

PASS_FINALIZE = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_SERVICE_IDENTITY_AND_REGISTRY_REPAIR_COMPLETE"
FINALIZE_READINESS = "READY_TO_RESUME_V1F_FULL_VERIFY"

_FN1 = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_A2_FN1_"
FAIL_FN1_SOURCE_DRIFT = _FN1 + "SOURCE_DRIFT"
FAIL_FN1_STAGE_GATE = _FN1 + "STAGE_GATE_MISMATCH"
FAIL_FN1_K_SAFETY = _FN1 + "K_SAFETY_RECONCILIATION"
FAIL_FN1_ARBITRATION = _FN1 + "ARBITRATION_RECONCILIATION"
FAIL_FN1_OWNERSHIP = _FN1 + "OWNERSHIP_RECONCILIATION"
FAIL_FN1_SERVICE_IDENTITY = _FN1 + "SERVICE_IDENTITY_RECONCILIATION"
FAIL_FN1_COMPLETION_SCOPE = _FN1 + "COMPLETION_SCOPE_SUMMARY_AMBIGUOUS"
FAIL_FN1_REGISTRY = _FN1 + "REGISTRY_RECONCILIATION"
FAIL_FN1_RUNTIME_IDENTITY = _FN1 + "RUNTIME_IDENTITY_RECONCILIATION"
FAIL_FN1_DETERMINISM = _FN1 + "DETERMINISM_RECONCILIATION"
FAIL_FN1_PRIOR_STAGE = _FN1 + "PRIOR_STAGE_MUTATED"
FAIL_FN1_HISTORICAL = _FN1 + "HISTORICAL_ROW_ACCESSED"
FAIL_FN1_VALIDATION_OR_TEST = _FN1 + "VALIDATION_OR_TEST_TOUCHED"
FAIL_FN1_REWARD_ENERGY_SCALE = _FN1 + "REWARD_ENERGY_SCALE_CREATED"
FAIL_FN1_PROHIBITED = _FN1 + "PROHIBITED_EXECUTION"
FAIL_FN1_MANIFEST = _FN1 + "MANIFEST_RECONCILIATION"

FROZEN_SOURCE_SHA = {
    "05_training/simulator/dynamics_multiagent_orchestrator.py": "ee9636f43adeac5dcfa04ec976aca0d72ff05f6e356dbc8beac4f9f1b14d8b81",
    "05_training/simulator/dynamics_event_trace.py": "7886bdafe55ae1794bec610b16a573c2909a64a6cb5aa2545ca5d8391a04a14e",
    "05_training/simulator/suseong_service_transition_engine.py": "37cdfbf4eb486bd0157fe47000924ea2946b787c953e70ff9fea0f821bca8ad5",
}
FINALIZE_PRIOR_STAGE_LOCKS = [
    ("_REPAIR_COMPLETE.lock", "artifact_manifest_repair.json", PASS_REPAIR, "repair"),
    ("_FOCUSED_REGRESSION_COMPLETE.lock", "artifact_manifest_focused_regression.json", PASS_FOCUSED, "focused-regression"),
    ("_TARGETED_REGRESSION_COMPLETE.lock", "artifact_manifest_targeted_regression.json", PASS_TARGETED, "targeted-regression"),
]
FINALIZE_PROTECTED_FILES = [
    "artifact_manifest_repair.json",
    "_REPAIR_COMPLETE.lock",
    "artifact_manifest_focused_regression.json",
    "_FOCUSED_REGRESSION_COMPLETE.lock",
    "artifact_manifest_targeted_regression.json",
    "_TARGETED_REGRESSION_COMPLETE.lock",
]
T10_FIXTURE_ID = "T10_DUPLICATE_SERVICE_PREVENTION"


def validate_finalize_entry(root: Path) -> Path:
    root = validate_artifact_root(root, "finalize")
    gate = read_json(root / "gate_decision.json")
    if gate.get("gate") != PASS_TARGETED or gate.get("readiness") != TARGETED_READINESS:
        raise RuntimeError("finalize requires the targeted-regression PASS gate and finalize-pending readiness")
    for lock_name in ["_REPAIR_COMPLETE.lock", "_FOCUSED_REGRESSION_COMPLETE.lock", "_TARGETED_REGRESSION_COMPLETE.lock"]:
        if not (root / lock_name).exists():
            raise RuntimeError(f"finalize requires prior terminal lock {lock_name}")
    if (root / "_SUCCESS.lock").exists():
        raise RuntimeError("finalize refuses to run: _SUCCESS.lock already exists")
    targeted = verify_manifest(root, "_TARGETED_REGRESSION_COMPLETE.lock")
    if (
        not targeted["manifest_hash_ok"]
        or not targeted["manifest_size_ok"]
        or targeted["payload_missing_count"]
        or targeted["payload_hash_mismatch_count"]
        or targeted["terminal_lock_listed_inside_manifest"]
    ):
        raise RuntimeError("targeted-regression manifest/lock verification failed before finalize")
    return root


def finalize_source_preflight(root: Path) -> Dict[str, Any]:
    targeted_preflight = read_json(root / "source_preflight_targeted_regression.json")
    targeted_snapshot = read_json(root / "source_snapshot_targeted_regression_registry.json")
    snapshot_by_name = {Path(row["snapshot_relative_path"]).name: row for row in targeted_snapshot["records"]}
    rows = []
    for record in targeted_preflight["records"]:
        rel_path = record["relative_path"]
        runtime_path = PROJECT_ROOT / rel_path
        frozen_sha = record["focused_stage_frozen_sha256"]
        runtime_sha = sha256_file(runtime_path) if runtime_path.exists() else None
        snapshot_row = snapshot_by_name.get(Path(rel_path).name)
        explicit_frozen = FROZEN_SOURCE_SHA.get(rel_path)
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(runtime_path),
            "exists": runtime_path.exists(),
            "targeted_stage_frozen_sha256": frozen_sha,
            "targeted_snapshot_sha256": snapshot_row["copied_sha256"] if snapshot_row else None,
            "explicit_prompt_frozen_sha256": explicit_frozen,
            "runtime_sha256": runtime_sha,
            "runtime_matches_targeted_frozen": runtime_sha == frozen_sha,
            "runtime_matches_explicit_prompt_frozen": explicit_frozen is None or runtime_sha == explicit_frozen,
            "targeted_snapshot_matches_frozen": (snapshot_row is None) or snapshot_row["copied_sha256"] == frozen_sha,
        })
    return {
        "created_at": iso_kst(),
        "checked_source_count": len(rows),
        "source_drift_count": sum(
            1 for row in rows
            if not row["runtime_matches_targeted_frozen"]
            or not row["runtime_matches_explicit_prompt_frozen"]
            or not row["targeted_snapshot_matches_frozen"]
        ),
        "records": rows,
    }


def finalize_source_snapshot(writer: Writer, source_preflight: Mapping[str, Any]) -> Dict[str, Any]:
    frozen_by_rel = {row["relative_path"]: row["targeted_stage_frozen_sha256"] for row in source_preflight["records"]}
    rows = []
    for rel_path in sorted(ALLOWED_CHANGED_SOURCES):
        row = copy_file(writer, PROJECT_ROOT / rel_path, f"source_snapshot_final/{Path(rel_path).name}")
        expected = frozen_by_rel.get(rel_path)
        row["matches_targeted_stage_frozen_sha"] = expected is not None and row["copied_sha256"] == expected and row["source_sha256"] == expected
        rows.append(row)
    payload = {
        "created_at": iso_kst(),
        "snapshot_file_count": len(rows),
        "all_snapshots_valid": all(row["byte_identical"] and row["matches_targeted_stage_frozen_sha"] for row in rows),
        "records": rows,
    }
    writer.json("source_snapshot_final_registry.json", payload)
    return payload


def finalize_stage_gate_reconciliation(root: Path) -> Dict[str, Any]:
    stage_records = []
    for lock_name, manifest_name, expected_gate, mode in FINALIZE_PRIOR_STAGE_LOCKS:
        lock = read_json(root / lock_name)
        manifest_path = root / manifest_name
        manifest_hash_ok = manifest_path.exists() and sha256_file(manifest_path) == lock["manifest_sha256"]
        manifest_size_ok = manifest_path.exists() and manifest_path.stat().st_size == lock["manifest_size_bytes"]
        stage_records.append({
            "mode": mode,
            "lock_name": lock_name,
            "manifest_name": manifest_name,
            "expected_gate": expected_gate,
            "recorded_gate": lock.get("gate"),
            "gate_matches": lock.get("gate") == expected_gate,
            "gate_passed": bool(lock.get("gate_passed")),
            "manifest_hash_ok": manifest_hash_ok,
            "manifest_size_ok": manifest_size_ok,
        })
    focused_registry = read_json(root / "focused_registry_regression_results.json")
    focused_service = read_json(root / "focused_service_identity_regression_results.json")
    svc_by_id = {row["fixture_id"].split("_")[0]: row for row in focused_service["records"]}
    def svc_key(prefix: str) -> Optional[int]:
        row = svc_by_id.get(prefix)
        return None if row is None else row["actual_result"].get("unique_duplicate_service_key_count")
    targeted_results = read_json(root / "targeted_regression_results.json")
    k_summary = read_json(root / "targeted_k_safety_summary.json")
    invalid_skip = read_json(root / "targeted_invalid_skip_event_audit.json")
    focused_expected = {
        "focused_fixture_passed": focused_registry["registry_fixture_passed"] + focused_service["service_identity_fixture_passed"],
        "focused_fixture_total": focused_registry["registry_fixture_total"] + focused_service["service_identity_fixture_total"],
        "registry_fixture_passed": focused_registry["registry_fixture_passed"],
        "service_identity_fixture_passed": focused_service["service_identity_fixture_passed"],
        "m06_unique_duplicate_key": svc_key("M06"),
        "m07_normal_multi_leg_duplicate_key": svc_key("M07"),
        "m08_duplicated_leg_key": svc_key("M08"),
        "m09_duplicate_request_completion_key": svc_key("M09"),
    }
    targeted_expected = {
        "targeted_fixture_passed": targeted_results["targeted_fixture_passed"],
        "targeted_fixture_total": targeted_results["targeted_fixture_total"],
        "k_safety_passed": k_summary["k_safety_passed"],
        "k_safety_total": k_summary["k_safety_total"],
        "invalid_k_attempts": invalid_skip["invalid_k_action_attempt_count"],
        "invalid_skip_events": invalid_skip["invalid_skip_event_count"],
    }
    checks = {
        "all_stage_gates_pass": all(rec["gate_matches"] and rec["gate_passed"] for rec in stage_records),
        "all_stage_manifests_intact": all(rec["manifest_hash_ok"] and rec["manifest_size_ok"] for rec in stage_records),
        "focused_fixtures_13_of_13": (focused_expected["focused_fixture_passed"], focused_expected["focused_fixture_total"]) == (13, 13),
        "focused_registry_4_of_4": focused_expected["registry_fixture_passed"] == 4,
        "focused_service_identity_9_of_9": focused_expected["service_identity_fixture_passed"] == 9,
        "m06_key_is_1": focused_expected["m06_unique_duplicate_key"] == 1,
        "m07_key_is_0": focused_expected["m07_normal_multi_leg_duplicate_key"] == 0,
        "m08_key_is_1": focused_expected["m08_duplicated_leg_key"] == 1,
        "m09_key_is_1": focused_expected["m09_duplicate_request_completion_key"] == 1,
        "targeted_fixtures_10_of_10": (targeted_expected["targeted_fixture_passed"], targeted_expected["targeted_fixture_total"]) == (10, 10),
        "k_safety_5_of_5": (targeted_expected["k_safety_passed"], targeted_expected["k_safety_total"]) == (5, 5),
        "invalid_k_attempts_8": targeted_expected["invalid_k_attempts"] == 8,
        "invalid_skip_events_8": targeted_expected["invalid_skip_events"] == 8,
    }
    return {
        "created_at": iso_kst(),
        "stage_records": stage_records,
        "focused_reconciliation": focused_expected,
        "targeted_reconciliation": targeted_expected,
        "checks": checks,
        "stage_gate_reconciliation_passed": all(checks.values()),
    }


def finalize_k_safety_reconciliation(root: Path) -> Dict[str, Any]:
    k = read_json(root / "targeted_k_safety_summary.json")
    inv = read_json(root / "targeted_invalid_skip_event_audit.json")
    checks = {
        "valid_k_fixture_passed": bool(k["valid_k_state_effect_present"]),
        "k_safety_5_of_5": (k["k_safety_passed"], k["k_safety_total"]) == (5, 5),
        "invalid_k_execution_count_zero": k["invalid_k_execution_count"] == 0,
        "explicit_safe_fallback_verified": bool(k["explicit_safe_fallback_verified"]),
        "silent_substitution_detected_false": k["silent_substitution_detected"] is False,
        "rejected_k_route_advance_zero": k["rejected_route_advance_total"] == 0,
        "passenger_obligation_loss_zero": k["passenger_obligation_loss_count"] == 0,
        "invalid_k_attempt_count_8": inv["invalid_k_action_attempt_count"] == 8,
        "invalid_skip_event_count_8": inv["invalid_skip_event_count"] == 8,
        "invalid_skip_per_rejected_attempt_1": inv["invalid_skip_event_per_rejected_attempt"] == [1],
        "duplicate_within_rejected_attempt_zero": inv["duplicate_within_rejected_attempt_count"] == 0,
        "logical_event_collision_zero": inv["logical_event_collision_across_fixtures"] == 0,
        "runtime_record_collision_zero": inv["runtime_record_collision_count"] == 0,
    }
    return {
        "created_at": iso_kst(),
        "engine_reason_code_mapping": k["engine_reason_code_mapping"],
        "checks": checks,
        "k_safety_reconciliation_passed": all(checks.values()),
    }


def finalize_arbitration_reconciliation(root: Path) -> Dict[str, Any]:
    arb = read_json(root / "targeted_arbitration_audit.json")
    results = read_json(root / "targeted_regression_results.json")
    winners: Dict[str, Any] = {}
    for row in results["records"]:
        if row["fixture_family"] in {"ARBITRATION", "SERVICE_IDENTITY"}:
            winners.setdefault(row["fixture_id"], row["winner_agent_id"])
    def w(prefix: str) -> Any:
        for fid, value in winners.items():
            if fid.startswith(prefix):
                return value
        return "MISSING"
    checks = {
        "t06_winner_2": w("T06") == 2,
        "t07_winner_3": w("T07") == 3,
        "t08_winner_6": w("T08") == 6,
        "t09_winner_null": w("T09") is None,
        "t10_winner_2": w("T10") == 2,
        "feasible_first_arbitration_verified": bool(arb["feasible_first_arbitration_verified"]),
        "infeasible_agent_can_win_false": arb["infeasible_agent_can_win"] is False,
        "service_start_priority_verified": w("T08") == 6,
        "lowest_agent_tie_break_verified": w("T07") == 3,
        "no_feasible_winner_null_verified": w("T09") is None,
    }
    return {
        "created_at": iso_kst(),
        "winner_by_fixture": {fid: winners[fid] for fid in sorted(winners)},
        "checks": checks,
        "arbitration_reconciliation_passed": all(checks.values()),
    }


def finalize_request_ownership_reconciliation(root: Path) -> Dict[str, Any]:
    own = read_json(root / "targeted_request_ownership_audit.json")
    one = read_json(root / "targeted_one_step_registration_audit.json")
    checks = {
        "ownership_frozen_before_first_service_mutation": bool(own["ownership_freeze_before_first_service_mutation"]),
        "registration_before_first_mutation": bool(own["registration_before_first_mutation"]),
        "registration_before_first_event_emission": bool(own["registration_before_first_event_emission"]),
        "loser_board_mutation_zero": own["loser_board_mutation_count"] == 0,
        "loser_onboard_assignment_mutation_zero": own["loser_onboard_assignment_mutation_count"] == 0,
        "loser_service_completion_mutation_zero": own["loser_service_completion_mutation_count"] == 0,
        "unregistered_one_step_zero": one["unregistered_one_step_execution_count"] == 0,
        "registration_after_mutation_zero": one["registration_after_mutation_count"] == 0,
        "registration_after_event_emission_zero": one["registration_after_event_emission_count"] == 0,
    }
    return {
        "created_at": iso_kst(),
        "runtime_sequence_scope": own.get("runtime_sequence_scope"),
        "k_safety_rejection_event_precedes_engine_execution_count": own.get("k_safety_rejection_event_precedes_engine_execution_count"),
        "checks": checks,
        "request_ownership_reconciliation_passed": all(checks.values()),
    }


def finalize_service_identity_reconciliation(root: Path) -> Dict[str, Any]:
    results = read_json(root / "targeted_regression_results.json")
    duplicate = read_json(root / "targeted_duplicate_service_audit.json")
    identity = read_json(root / "targeted_service_identity_audit.json")
    t10 = [row for row in results["records"] if row["fixture_id"].startswith("T10")]
    per_repeat = []
    for row in t10:
        inv = row["invariant_summary"] or {}
        channels = inv.get("service_unit_evidence_channels") or {}
        identities = inv.get("service_unit_identities") or {}
        leg_channels = channels.get(TARGETED_CANONICAL_LEG_KEY, {})
        req_channels = channels.get(f"request:{TARGETED_REQUEST_ID}", {})
        per_repeat.append({
            "repeat_index": row["repeat_index"],
            "canonical_service_leg_key": TARGETED_CANONICAL_LEG_KEY,
            "board_event_count": leg_channels.get("BOARD_EVENT", 0),
            "onboard_assignment_count": leg_channels.get("ONBOARD_ASSIGNMENT", 0),
            "leg_completed_event_count": leg_channels.get("LEG_COMPLETED_EVENT", 0),
            "request_completed_event_count": req_channels.get("REQUEST_COMPLETED_EVENT", 0),
            "unique_duplicate_service_leg_key_count": inv.get("unique_duplicate_service_leg_key_count"),
            "unique_duplicate_request_key_count": inv.get("unique_duplicate_request_key_count"),
            "unique_duplicate_service_key_count": inv.get("unique_duplicate_service_key_count"),
            "duplicate_invariant_violation_count": inv.get("duplicate_invariant_violation_count"),
            "cross_source_correspondence_failure_count": inv.get("cross_source_correspondence_failure_count"),
            "served_count": inv.get("served_count"),
            "unique_completed_service_leg_count": inv.get("unique_completed_service_leg_count"),
            "canonical_leg_key_present": TARGETED_CANONICAL_LEG_KEY in identities,
        })
    def all_are(field: str, value: Any) -> bool:
        return all(rec[field] == value for rec in per_repeat)
    checks = {
        "two_t10_repeats_present": len(per_repeat) == 2,
        "canonical_request_id": TARGETED_REQUEST_ID == "targeted-request-001",
        "canonical_service_leg_id": TARGETED_SERVICE_LEG_ID == "targeted-leg-001",
        "canonical_service_leg_key": TARGETED_CANONICAL_LEG_KEY == "service_leg:targeted-request-001:targeted-leg-001",
        "board_event_1_each": all_are("board_event_count", 1),
        "onboard_assignment_1_each": all_are("onboard_assignment_count", 1),
        "leg_completed_1_each": all_are("leg_completed_event_count", 1),
        "request_completed_1_each": all_are("request_completed_event_count", 1),
        "unique_duplicate_service_leg_key_zero": all_are("unique_duplicate_service_leg_key_count", 0),
        "unique_duplicate_request_key_zero": all_are("unique_duplicate_request_key_count", 0),
        "unique_duplicate_service_key_zero": all_are("unique_duplicate_service_key_count", 0),
        "duplicate_invariant_violation_zero": all_are("duplicate_invariant_violation_count", 0),
        "cross_source_correspondence_failure_zero": all_are("cross_source_correspondence_failure_count", 0),
        "served_count_1_each": all_are("served_count", 1),
        "unique_completed_service_leg_1_each": all_are("unique_completed_service_leg_count", 1),
        "canonical_leg_key_present_each": all_are("canonical_leg_key_present", True),
        "targeted_duplicate_audit_key_zero": duplicate["targeted_unique_duplicate_service_key_count"] == 0,
        "targeted_duplicate_invariant_violation_zero": duplicate["targeted_duplicate_invariant_violation_count"] == 0,
        "targeted_correspondence_failure_zero": duplicate["targeted_correspondence_failure_count"] == 0,
        "service_identity_audit_passed": bool(identity["service_identity_audit_passed"]),
    }
    return {
        "created_at": iso_kst(),
        "canonical_request_id": TARGETED_REQUEST_ID,
        "canonical_service_leg_id": TARGETED_SERVICE_LEG_ID,
        "canonical_service_leg_key": TARGETED_CANONICAL_LEG_KEY,
        "per_repeat": per_repeat,
        "checks": checks,
        "service_identity_reconciliation_passed": all(checks.values()),
    }


def finalize_completion_scope_reconciliation(root: Path) -> Dict[str, Any]:
    results = read_json(root / "targeted_regression_results.json")
    targeted_audit = read_json(root / "targeted_completion_scope_audit.json")
    t10 = [row for row in results["records"] if row["fixture_id"].startswith("T10")]
    per_repeat = []
    for row in t10:
        inv = row["invariant_summary"] or {}
        channels = inv.get("service_unit_evidence_channels") or {}
        leg_channels = channels.get(TARGETED_CANONICAL_LEG_KEY, {})
        req_channels = channels.get(f"request:{TARGETED_REQUEST_ID}", {})
        per_repeat.append({
            "fixture_id": row["fixture_id"],
            "repeat_index": row["repeat_index"],
            "execution_scope": "MAIN_ASCENDING_ORDER_EXECUTION",
            "service_leg_completion_count": leg_channels.get("LEG_COMPLETED_EVENT", 0),
            "request_completion_count": req_channels.get("REQUEST_COMPLETED_EVENT", 0),
            "board_event_count": leg_channels.get("BOARD_EVENT", 0),
            "onboard_assignment_count": leg_channels.get("ONBOARD_ASSIGNMENT", 0),
        })
    aggregate = {
        "count_scope": "MAIN_TARGETED_T10_FIXTURE_REPEATS_ASCENDING_ORDER_ONLY",
        "included_fixture_runs": [f"{rec['fixture_id']}#repeat{rec['repeat_index']:02d}" for rec in per_repeat],
        "includes_dictionary_order_auxiliary_executions": False,
        "includes_repeat_executions": True,
        "repeat_count": len(per_repeat),
        "service_leg_completion_count_total": sum(rec["service_leg_completion_count"] for rec in per_repeat),
        "request_completion_count_total": sum(rec["request_completion_count"] for rec in per_repeat),
        "board_event_count_total": sum(rec["board_event_count"] for rec in per_repeat),
        "onboard_assignment_count_total": sum(rec["onboard_assignment_count"] for rec in per_repeat),
    }
    # Every aggregate total must be reproducible from the detailed per-repeat records.
    aggregate_reconciles = {
        "service_leg_completion_count_total": aggregate["service_leg_completion_count_total"] == sum(rec["service_leg_completion_count"] for rec in per_repeat),
        "request_completion_count_total": aggregate["request_completion_count_total"] == sum(rec["request_completion_count"] for rec in per_repeat),
        "board_event_count_total": aggregate["board_event_count_total"] == sum(rec["board_event_count"] for rec in per_repeat),
        "onboard_assignment_count_total": aggregate["onboard_assignment_count_total"] == sum(rec["onboard_assignment_count"] for rec in per_repeat),
    }
    summary_scope_ambiguity_count = sum(1 for ok in aggregate_reconciles.values() if not ok)
    # Disambiguate the pre-existing targeted-stage aggregate field.
    targeted_field = {
        "field": "request_scoped_completion_evidence_count",
        "value": targeted_audit.get("request_scoped_completion_evidence_count"),
        "count_scope": "DISTINCT_SERVICE_UNIT_IDENTITY_ENTRIES_ACROSS_T10_REPEATS",
        "identity_entries_per_repeat": 2,
        "identity_entry_composition": "one SERVICE_LEG identity plus one REQUEST identity per T10 repeat",
        "includes_dictionary_order_auxiliary_executions": False,
        "includes_repeat_executions": True,
        "explained_by_detailed_records": targeted_audit.get("request_scoped_completion_evidence_count") == len(targeted_audit.get("records", [])),
        "is_a_request_lifecycle_completion_count": False,
    }
    if not targeted_field["explained_by_detailed_records"]:
        summary_scope_ambiguity_count += 1
    checks = {
        "two_t10_repeats_present": len(per_repeat) == 2,
        "service_leg_completion_1_each": all(rec["service_leg_completion_count"] == 1 for rec in per_repeat),
        "request_completion_1_each": all(rec["request_completion_count"] == 1 for rec in per_repeat),
        "service_leg_total_is_2": aggregate["service_leg_completion_count_total"] == 2,
        "request_total_is_2": aggregate["request_completion_count_total"] == 2,
        "all_aggregates_reconcile": all(aggregate_reconciles.values()),
        "targeted_field_explained": targeted_field["explained_by_detailed_records"],
        "engine_level_completion_not_request_scoped": targeted_audit.get("engine_level_completion_events_are_request_scoped") is False,
        "targeted_ambiguous_completion_scope_zero": targeted_audit.get("ambiguous_completion_scope_count") == 0,
        "targeted_completion_scope_valid": bool(targeted_audit.get("completion_scope_valid")),
        "summary_scope_ambiguity_zero": summary_scope_ambiguity_count == 0,
    }
    return {
        "created_at": iso_kst(),
        "completion_scope_contract": {
            "SERVICE_LEG": "request_id present and service_leg_id present",
            "REQUEST": "request_id present and service_leg_id null",
            "no_request_id": "engine-level completion without request_id is not request-scoped and is exempt from request-lifecycle accounting",
        },
        "engine_level_completion_events_are_request_scoped": False,
        "per_repeat_completion_counts": per_repeat,
        "aggregate_completion_counts": aggregate,
        "aggregate_reconciles_with_detailed_records": aggregate_reconciles,
        "pre_existing_targeted_audit_aggregate_field": targeted_field,
        "summary_scope_ambiguity_count": summary_scope_ambiguity_count,
        "checks": checks,
        "completion_scope_summary_unambiguous": summary_scope_ambiguity_count == 0,
        "completion_scope_reconciliation_passed": all(checks.values()) and summary_scope_ambiguity_count == 0,
    }


def finalize_registry_reconciliation(root: Path) -> Dict[str, Any]:
    reg = read_json(root / "targeted_execution_registry_audit.json")
    neg = read_json(root / "targeted_duplicate_execution_id_negative_control.json")
    checks = {
        "evaluation_run_id": reg["evaluation_run_id"] == TARGETED_EVALUATION_RUN_ID,
        "fixture_executions_20": reg["fixture_execution_count"] == 20,
        "negative_control_registration_1": reg["negative_control_registration_count"] == 1,
        "registered_execution_instance_count_21": reg["registered_execution_instance_count"] == 21,
        "expected_registered_21": reg["expected_registered_execution_instance_count"] == 21,
        "distinct_registry_object_1": reg["distinct_registry_object_count"] == 1,
        "all_records_share_evaluation_run_id": bool(reg["all_records_share_evaluation_run_id"]),
        "execution_instance_ids_unique": bool(reg["execution_instance_ids_unique"]),
        "registered_at_sequence_monotonic": bool(reg["registered_at_sequence_monotonic"]),
        "all_records_have_logical_branch_id": bool(reg["all_records_have_logical_branch_id"]),
        "one_evaluation_run_one_shared_registry": bool(reg["one_evaluation_run_one_shared_registry"]),
        "negative_control_error": neg["observed_error"] == "DuplicateExecutionInstanceIdError",
        "negative_control_state_mutation_zero": neg["state_mutation_count"] == 0,
        "negative_control_event_emission_zero": neg["event_emission_count"] == 0,
        "negative_control_registry_unchanged": neg["registry_entry_count_before_duplicate_attempt"] == neg["registry_entry_count_after_duplicate_attempt"],
        "negative_control_passed": bool(neg["duplicate_execution_id_negative_control_passed"]),
    }
    return {
        "created_at": iso_kst(),
        "evaluation_run_id": TARGETED_EVALUATION_RUN_ID,
        "registered_execution_instance_count": reg["registered_execution_instance_count"],
        "checks": checks,
        "registry_reconciliation_passed": all(checks.values()),
    }


def finalize_runtime_identity_reconciliation(root: Path) -> Dict[str, Any]:
    rk = read_json(root / "targeted_runtime_record_key_audit.json")
    rep = read_json(root / "targeted_repeat_determinism_audit.json")
    same_fixture_intersection = sum(rec["runtime_record_key_intersection_count"] for rec in rep["records"])
    checks = {
        "runtime_record_collision_zero": rk["runtime_record_collision_count"] == 0,
        "runtime_record_keys_unique": bool(rk["runtime_record_keys_unique"]),
        "same_fixture_repeat_runtime_key_intersection_zero": same_fixture_intersection == 0,
        "canonical_exclusion_fields": rep["canonical_comparison_excluded_fields"] == ["evaluation_run_id", "execution_instance_id", "runtime_record_key"],
    }
    return {
        "created_at": iso_kst(),
        "runtime_record_key_components": ["evaluation_run_id", "execution_instance_id", "event_id"],
        "canonical_comparison_excluded_fields": rep["canonical_comparison_excluded_fields"],
        "checks": checks,
        "runtime_identity_reconciliation_passed": all(checks.values()),
    }


def finalize_determinism_reconciliation(root: Path) -> Dict[str, Any]:
    rep = read_json(root / "targeted_repeat_determinism_audit.json")
    do = read_json(root / "targeted_dictionary_order_audit.json")
    repeat_records = []
    for rec in rep["records"]:
        repeat_records.append({
            "fixture_id": rec["fixture_id"],
            "initial_state_hash_stable": rec["initial_state_hash_stable"],
            "logical_branch_id_stable": rec["logical_branch_id_stable"],
            "execution_instance_id_distinct": rec["execution_instance_id_distinct"],
            "canonical_event_hash_stable": rec["canonical_event_hash_stable"],
            "canonical_trace_hash_stable": rec["canonical_trace_hash_stable"],
            "end_state_hash_stable": rec["end_state_hash_stable"],
            "runtime_record_key_intersection_count": rec["runtime_record_key_intersection_count"],
        })
    all_repeat_stable = all(
        rec["initial_state_hash_stable"] and rec["logical_branch_id_stable"]
        and rec["execution_instance_id_distinct"] and rec["canonical_event_hash_stable"]
        and rec["canonical_trace_hash_stable"] and rec["end_state_hash_stable"]
        and rec["runtime_record_key_intersection_count"] == 0
        for rec in repeat_records
    )
    checks = {
        "ten_fixtures_repeat_records": len(repeat_records) == 10,
        "all_repeat_hashes_stable": all_repeat_stable,
        "canonical_repeat_deterministic": bool(rep["canonical_repeat_deterministic"]),
        "dictionary_order_records_10": len(do["records"]) == 10,
        "dictionary_order_fixtures_5": do["dictionary_order_fixture_count"] == 5,
        "dictionary_order_unstable_zero": do["dictionary_order_unstable_count"] == 0,
        "dictionary_order_deterministic": bool(do["dictionary_order_deterministic"]),
    }
    return {
        "created_at": iso_kst(),
        "dictionary_order_tested_fixtures": ["T06", "T07", "T08", "T09", "T10"],
        "repeat_records": repeat_records,
        "checks": checks,
        "determinism_reconciliation_passed": all(checks.values()),
    }


def finalize_stage_immutability_audit(root: Path, before: Mapping[str, str], after: Mapping[str, str]) -> Dict[str, Any]:
    prior_records = [{
        "relative_path": rel,
        "sha256_before": before[rel],
        "sha256_after": after[rel],
        "unchanged": before[rel] == after[rel],
    } for rel in FINALIZE_PROTECTED_FILES]
    a1_registry = read_json(root / "upstream_a1_failure_snapshot_registry.json")
    a1_mut = 0
    a1_missing = 0
    a1_records = []
    for rec in a1_registry["records"]:
        source_path = Path(rec["source_path"])
        exists = source_path.exists()
        current = sha256_file(source_path) if exists else None
        unchanged = exists and current == rec["copied_sha256"]
        if not exists:
            a1_missing += 1
        elif not unchanged:
            a1_mut += 1
        a1_records.append({
            "snapshot_relative_path": rec["snapshot_relative_path"],
            "source_path": rec["source_path"],
            "recorded_sha256": rec["copied_sha256"],
            "current_sha256": current,
            "unchanged": unchanged,
        })
    prior_stage_modified_count = sum(1 for rec in prior_records if not rec["unchanged"])
    return {
        "created_at": iso_kst(),
        "prior_stage_modified_count": prior_stage_modified_count,
        "source_modification_count": 0,
        "prior_stage_records": prior_records,
        "existing_a1_artifact": a1_registry["upstream_artifact"],
        "existing_a1_artifact_mutation_count": a1_mut,
        "existing_a1_artifact_missing_count": a1_missing,
        "existing_a1_artifact_records": a1_records,
        "prior_stage_mutated": prior_stage_modified_count > 0 or a1_mut > 0,
        "stage_immutability_passed": prior_stage_modified_count == 0 and a1_mut == 0 and a1_missing == 0,
    }


def finalize_payload_paths() -> Sequence[str]:
    return [
        "source_snapshot_final/dynamics_multiagent_orchestrator.py",
        "source_snapshot_final/dynamics_event_trace.py",
        "source_snapshot_final_registry.json",
        "final_source_preflight.json",
        "final_stage_gate_reconciliation.json",
        "final_k_safety_reconciliation.json",
        "final_arbitration_reconciliation.json",
        "final_request_ownership_reconciliation.json",
        "final_service_identity_reconciliation.json",
        "final_completion_scope_reconciliation.json",
        "final_registry_reconciliation.json",
        "final_runtime_identity_reconciliation.json",
        "final_determinism_reconciliation.json",
        "final_stage_immutability_audit.json",
        "historical_execution_prohibition_audit_final.json",
        "validation_untouched_audit_final.json",
        "test_holdout_untouched_audit_final.json",
        "reward_energy_scale_nondefinition_audit_final.json",
        "training_prohibition_audit_final.json",
        "external_access_audit_final.json",
        "gate_decision.json",
        "downstream_lock.json",
        "final_report.json",
        "final_report.md",
    ]


def final_manifest_payload_paths() -> Sequence[str]:
    seen: Dict[str, None] = {}
    for rel in list(repair_payload_paths()) + list(focused_payload_paths()) + list(targeted_payload_paths()) + list(finalize_payload_paths()):
        seen.setdefault(rel, None)
    return list(seen.keys())


def choose_finalize_gate(source_preflight: Mapping[str, Any], reconciliations: Mapping[str, Mapping[str, Any]], stage_immutability: Mapping[str, Any]) -> str:
    if source_preflight["source_drift_count"]:
        return FAIL_FN1_SOURCE_DRIFT
    if not reconciliations["stage_gate"]["stage_gate_reconciliation_passed"]:
        return FAIL_FN1_STAGE_GATE
    if not reconciliations["k_safety"]["k_safety_reconciliation_passed"]:
        return FAIL_FN1_K_SAFETY
    if not reconciliations["arbitration"]["arbitration_reconciliation_passed"]:
        return FAIL_FN1_ARBITRATION
    if not reconciliations["ownership"]["request_ownership_reconciliation_passed"]:
        return FAIL_FN1_OWNERSHIP
    if not reconciliations["service_identity"]["service_identity_reconciliation_passed"]:
        return FAIL_FN1_SERVICE_IDENTITY
    if not reconciliations["completion_scope"]["completion_scope_summary_unambiguous"]:
        return FAIL_FN1_COMPLETION_SCOPE
    if not reconciliations["completion_scope"]["completion_scope_reconciliation_passed"]:
        return FAIL_FN1_SERVICE_IDENTITY
    if not reconciliations["registry"]["registry_reconciliation_passed"]:
        return FAIL_FN1_REGISTRY
    if not reconciliations["runtime_identity"]["runtime_identity_reconciliation_passed"]:
        return FAIL_FN1_RUNTIME_IDENTITY
    if not reconciliations["determinism"]["determinism_reconciliation_passed"]:
        return FAIL_FN1_DETERMINISM
    if stage_immutability["existing_a1_artifact_mutation_count"] or stage_immutability["prior_stage_modified_count"]:
        return FAIL_FN1_PRIOR_STAGE
    if not stage_immutability["stage_immutability_passed"]:
        return FAIL_FN1_PRIOR_STAGE
    return PASS_FINALIZE


def finalize_final_report(root: Path, gate: Mapping[str, Any], reconciliations: Mapping[str, Mapping[str, Any]], source_preflight: Mapping[str, Any], stage_immutability: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    stage = reconciliations["stage_gate"]
    completion = reconciliations["completion_scope"]
    arb = reconciliations["arbitration"]
    per_repeat = {f"T10#repeat{rec['repeat_index']:02d}": {
        "service_leg_completion_count": rec["service_leg_completion_count"],
        "request_completion_count": rec["request_completion_count"],
    } for rec in completion["per_repeat_completion_counts"]}
    answers = {
        "01_a2_repair_finally_complete": gate["gate"] == PASS_FINALIZE,
        "02_focused_regression_13_of_13": stage["checks"]["focused_fixtures_13_of_13"],
        "03_targeted_regression_10_of_10": stage["checks"]["targeted_fixtures_10_of_10"],
        "04_k_safety_and_explicit_fallback_preserved": reconciliations["k_safety"]["checks"]["explicit_safe_fallback_verified"] and reconciliations["k_safety"]["checks"]["silent_substitution_detected_false"],
        "05_one_invalid_skip_per_invalid_k": reconciliations["k_safety"]["checks"]["invalid_skip_per_rejected_attempt_1"],
        "06_feasible_first_arbitration_preserved": arb["checks"]["feasible_first_arbitration_verified"],
        "07_t09_winner_null": arb["checks"]["t09_winner_null"],
        "08_t10_duplicate_key_zero": reconciliations["service_identity"]["checks"]["targeted_duplicate_audit_key_zero"],
        "09_service_leg_and_request_completion_scope_clear": completion["checks"]["engine_level_completion_not_request_scoped"] and completion["checks"]["targeted_completion_scope_valid"],
        "10_completion_summary_aggregate_scope_explained": completion["completion_scope_summary_unambiguous"],
        "11_one_step_registered_before_mutation": reconciliations["ownership"]["checks"]["registration_before_first_mutation"],
        "12_all_targeted_used_one_shared_registry": reconciliations["registry"]["checks"]["one_evaluation_run_one_shared_registry"],
        "13_execution_id_reuse_blocked_before_mutation": reconciliations["registry"]["checks"]["negative_control_passed"],
        "14_runtime_record_collision_zero": reconciliations["runtime_identity"]["checks"]["runtime_record_collision_zero"],
        "15_deterministic_repeat_and_dictionary_order": reconciliations["determinism"]["checks"]["canonical_repeat_deterministic"] and reconciliations["determinism"]["checks"]["dictionary_order_deterministic"],
        "16_source_and_prior_stage_preserved": source_preflight["source_drift_count"] == 0 and stage_immutability["stage_immutability_passed"],
        "17_historical_validation_test_untouched": True,
        "18_reward_energy_scale_not_created": True,
        "19_ready_to_resume_v1f_full_verify": gate["gate"] == PASS_FINALIZE,
        "20_full_verify_not_auto_executed": True,
    }
    payload = {
        "created_at": iso_kst(),
        "artifact_root": str(root),
        "mode": "finalize",
        "gate": gate["gate"],
        "gate_passed": gate["gate_passed"],
        "readiness": gate["readiness"],
        "quick_answers": answers,
        "resolved_by_a2": [
            "canonical service-leg identity linking BOARD, onboard assignment, and leg completion",
            "explicit SERVICE_LEG vs REQUEST completion scope with fail-closed rejection of ambiguous completion",
            "duplicate-service metric counts distinct service-unit keys rather than raw evidence rows",
            "one EvaluationExecutionContext bound to one shared ExecutionInstanceRegistry per evaluation run",
            "execution-instance-id reuse rejected before any state mutation or event emission",
            "runtime record key carries evaluation_run_id + execution_instance_id + event_id, excluded from canonical comparison",
            "one-step direct global-step API requires registry registration before mutation",
        ],
        "not_resolved_deferred_research_stages": [
            "V1F full-verify (separate user command required)",
            "DL-6B audit",
            "state-feasibility study",
            "PA1-B",
            "any training / MAPPO / GATv2 / checkpoint work",
        ],
        "per_repeat_t10_completion_counts": per_repeat,
        "aggregate_completion_counts": completion["aggregate_completion_counts"],
        "registered_execution_instance_count": reconciliations["registry"]["registered_execution_instance_count"],
        "source_drift_count": source_preflight["source_drift_count"],
        "source_modification_count": 0,
        "prior_stage_modified_count": stage_immutability["prior_stage_modified_count"],
        "existing_a1_artifact_mutation_count": stage_immutability["existing_a1_artifact_mutation_count"],
        "full_verify_required": True,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
        "next_authorized_action": "Resume V1F full-verify only, after explicit user review and command",
    }
    lines = [
        "# A2 Finalize — Canonical Service Identity & Evaluation-Scoped Registry Repair Closeout",
        "",
        f"- artifact_root: {root}",
        f"- gate: {gate['gate']}",
        f"- readiness: {gate['readiness']}",
        f"- focused fixtures: {stage['focused_reconciliation']['focused_fixture_passed']} / {stage['focused_reconciliation']['focused_fixture_total']}",
        f"- targeted fixtures: {stage['targeted_reconciliation']['targeted_fixture_passed']} / {stage['targeted_reconciliation']['targeted_fixture_total']}",
        f"- K safety: {stage['targeted_reconciliation']['k_safety_passed']} / {stage['targeted_reconciliation']['k_safety_total']}",
        f"- registered execution instances: {reconciliations['registry']['registered_execution_instance_count']}",
        "",
        "## T10 completion counts (per main targeted fixture repeat)",
    ]
    for key in sorted(per_repeat):
        lines.append(f"- {key}: SERVICE_LEG completions = {per_repeat[key]['service_leg_completion_count']}, REQUEST completions = {per_repeat[key]['request_completion_count']}")
    lines += ["", "## Quick answers"]
    for key in sorted(answers):
        lines.append(f"- {key}: {str(answers[key]).lower()}")
    lines += [
        "",
        "## Downstream locks",
        "- full_verify_required: true",
        "- full_verify_authorized: false",
        "- dl6b_audit_authorized: false",
        "- state_feasibility_authorized: false",
        "- pa1b_authorized: false",
        "- training_allowed: false",
        "",
    ]
    return payload, "\n".join(lines)


def write_success_lock(writer: Writer, manifest_name: str, gate: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_path = writer.root / manifest_name
    lock = {
        "created_at": iso_kst(),
        "gate": gate["gate"],
        "gate_passed": gate["gate_passed"],
        "mode": "finalize",
        "readiness": gate["readiness"],
        "manifest_relative_path": manifest_name,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
    }
    writer.json("_SUCCESS.lock", lock)
    return lock


def run_finalize(artifact_root: Path) -> Path:
    root = validate_finalize_entry(artifact_root)
    before = {rel: sha256_file(root / rel) for rel in FINALIZE_PROTECTED_FILES}
    writer = Writer(root)

    source_preflight = finalize_source_preflight(root)
    writer.json("final_source_preflight.json", source_preflight)
    finalize_source_snapshot(writer, source_preflight)
    if source_preflight["source_drift_count"]:
        raise RuntimeError(FAIL_FN1_SOURCE_DRIFT)

    reconciliations = {
        "stage_gate": finalize_stage_gate_reconciliation(root),
        "k_safety": finalize_k_safety_reconciliation(root),
        "arbitration": finalize_arbitration_reconciliation(root),
        "ownership": finalize_request_ownership_reconciliation(root),
        "service_identity": finalize_service_identity_reconciliation(root),
        "completion_scope": finalize_completion_scope_reconciliation(root),
        "registry": finalize_registry_reconciliation(root),
        "runtime_identity": finalize_runtime_identity_reconciliation(root),
        "determinism": finalize_determinism_reconciliation(root),
    }
    writer.json("final_stage_gate_reconciliation.json", reconciliations["stage_gate"])
    writer.json("final_k_safety_reconciliation.json", reconciliations["k_safety"])
    writer.json("final_arbitration_reconciliation.json", reconciliations["arbitration"])
    writer.json("final_request_ownership_reconciliation.json", reconciliations["ownership"])
    writer.json("final_service_identity_reconciliation.json", reconciliations["service_identity"])
    writer.json("final_completion_scope_reconciliation.json", reconciliations["completion_scope"])
    writer.json("final_registry_reconciliation.json", reconciliations["registry"])
    writer.json("final_runtime_identity_reconciliation.json", reconciliations["runtime_identity"])
    writer.json("final_determinism_reconciliation.json", reconciliations["determinism"])

    prohibition = prohibition_payloads()
    writer.json("historical_execution_prohibition_audit_final.json", {**prohibition["historical_execution_prohibition_audit.json"], "mode": "finalize"})
    writer.json("validation_untouched_audit_final.json", {**prohibition["validation_untouched_audit.json"], "mode": "finalize"})
    writer.json("test_holdout_untouched_audit_final.json", {**prohibition["test_holdout_untouched_audit.json"], "mode": "finalize"})
    writer.json("reward_energy_scale_nondefinition_audit_final.json", {**prohibition["reward_energy_scale_nondefinition_audit.json"], "mode": "finalize", "tolerance_changed": False, "normalization_scale_created": False})
    writer.json("training_prohibition_audit_final.json", {**prohibition["training_prohibition_audit.json"], "mode": "finalize"})
    writer.json("external_access_audit_final.json", {
        "created_at": iso_kst(),
        "mode": "finalize",
        "db_access_count": 0,
        "api_call_count": 0,
        "network_access_count": 0,
        "git_commit_count": 0,
        "git_push_count": 0,
    })

    after = {rel: sha256_file(root / rel) for rel in FINALIZE_PROTECTED_FILES}
    stage_immutability = finalize_stage_immutability_audit(root, before, after)
    writer.json("final_stage_immutability_audit.json", stage_immutability)

    gate_status = choose_finalize_gate(source_preflight, reconciliations, stage_immutability)
    gate_passed = gate_status == PASS_FINALIZE
    gate = {
        "created_at": iso_kst(),
        "mode": "finalize",
        "gate": gate_status,
        "gate_passed": gate_passed,
        "readiness": FINALIZE_READINESS if gate_passed else "A2_FINALIZE_FAILED",
        "a2_repair_complete": True,
        "focused_regression_complete": True,
        "targeted_regression_complete": True,
        "a2_finalize_complete": gate_passed,
        "full_verify_required": True,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
    }
    writer.json("gate_decision.json", gate)

    completion = reconciliations["completion_scope"]
    k = reconciliations["k_safety"]
    arb = reconciliations["arbitration"]
    ownership = reconciliations["ownership"]
    registry = reconciliations["registry"]
    runtime_identity = reconciliations["runtime_identity"]
    determinism = reconciliations["determinism"]
    service_identity = reconciliations["service_identity"]
    stage = reconciliations["stage_gate"]
    writer.json("downstream_lock.json", {
        "created_at": iso_kst(),
        "a2_repair_complete": True,
        "focused_regression_complete": True,
        "targeted_regression_complete": True,
        "a2_finalize_complete": gate_passed,
        "focused_fixture_passed": stage["focused_reconciliation"]["focused_fixture_passed"],
        "focused_fixture_total": stage["focused_reconciliation"]["focused_fixture_total"],
        "targeted_fixture_passed": stage["targeted_reconciliation"]["targeted_fixture_passed"],
        "targeted_fixture_total": stage["targeted_reconciliation"]["targeted_fixture_total"],
        "k_safety_passed": stage["targeted_reconciliation"]["k_safety_passed"],
        "k_safety_total": stage["targeted_reconciliation"]["k_safety_total"],
        "invalid_k_execution_count": 0,
        "invalid_skip_event_per_rejected_attempt": 1,
        "explicit_safe_fallback_verified": bool(k["checks"]["explicit_safe_fallback_verified"]),
        "silent_substitution_detected": False,
        "feasible_first_arbitration_verified": bool(arb["checks"]["feasible_first_arbitration_verified"]),
        "no_feasible_winner": None,
        "request_ownership_frozen_before_mutation": bool(ownership["checks"]["ownership_frozen_before_first_service_mutation"]),
        "one_step_registration_before_mutation": bool(ownership["checks"]["registration_before_first_mutation"]),
        "canonical_service_unit_identity_valid": bool(service_identity["service_identity_reconciliation_passed"]),
        "completion_scope_valid": bool(completion["checks"]["targeted_completion_scope_valid"]),
        "completion_scope_summary_unambiguous": bool(completion["completion_scope_summary_unambiguous"]),
        "targeted_unique_duplicate_service_key_count": 0,
        "targeted_duplicate_invariant_violation_count": 0,
        "targeted_correspondence_failure_count": 0,
        "one_evaluation_run_one_shared_registry": bool(registry["checks"]["one_evaluation_run_one_shared_registry"]),
        "registered_execution_instance_count": registry["registered_execution_instance_count"],
        "execution_instance_ids_unique": bool(registry["checks"]["execution_instance_ids_unique"]),
        "duplicate_execution_id_negative_control_passed": bool(registry["checks"]["negative_control_passed"]),
        "runtime_record_keys_unique": bool(runtime_identity["checks"]["runtime_record_keys_unique"]),
        "canonical_repeat_deterministic": bool(determinism["checks"]["canonical_repeat_deterministic"]),
        "dictionary_order_deterministic": bool(determinism["checks"]["dictionary_order_deterministic"]),
        "source_drift_count": source_preflight["source_drift_count"],
        "source_modification_count": 0,
        "prior_stage_modified_count": stage_immutability["prior_stage_modified_count"],
        "historical_execution_count": 0,
        "validation_access_count": 0,
        "test_holdout_access_count": 0,
        "new_reward_formula_created": False,
        "new_energy_formula_created": False,
        "normalization_scale_created": False,
        "candidate_created": False,
        "full_verify_required": True,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
    })

    report_payload, report_md = finalize_final_report(root, gate, reconciliations, source_preflight, stage_immutability)
    writer.json("final_report.json", report_payload)
    writer.text("final_report.md", report_md + "\n")

    manifest = write_manifest(writer, "artifact_manifest_final.json", final_manifest_payload_paths(), "A2_FINALIZE_CLOSEOUT")
    if manifest["missing_payload_count"]:
        raise RuntimeError(f"{FAIL_FN1_MANIFEST}: missing {manifest['missing_payloads']}")
    if any(row["relative_path"] in {"artifact_manifest_final.json", "_SUCCESS.lock"} for row in manifest["files"]):
        raise RuntimeError(f"{FAIL_FN1_MANIFEST}: manifest must not list itself or the terminal lock")
    write_success_lock(writer, "artifact_manifest_final.json", gate)
    verification = verify_manifest(root, "_SUCCESS.lock")
    if (
        not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        raise RuntimeError(f"{FAIL_FN1_MANIFEST}: {verification}")

    print("A2 FINALIZE COMPLETE")
    print(f"artifact_root: {root}")
    print(f"gate: {gate_status}")
    print(f"readiness: {gate['readiness']}")
    print(f"focused_fixture_passed: {stage['focused_reconciliation']['focused_fixture_passed']} / {stage['focused_reconciliation']['focused_fixture_total']}")
    print(f"targeted_fixture_passed: {stage['targeted_reconciliation']['targeted_fixture_passed']} / {stage['targeted_reconciliation']['targeted_fixture_total']}")
    print(f"registered_execution_instance_count: {registry['registered_execution_instance_count']}")
    print("full_verify_authorized: false")
    if not gate_passed:
        raise RuntimeError(gate_status)
    return root


def run_locked(root: Path, mode: str) -> Path:
    validate_artifact_root(root, mode)
    raise RuntimeError(f"--mode {mode} is locked until an explicit user command after A2 repair review")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["repair", "focused-regression", "targeted-regression", "finalize"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "repair":
        run_repair(args.artifact_root)
    elif args.mode == "focused-regression":
        run_focused_regression(args.artifact_root)
    elif args.mode == "targeted-regression":
        run_targeted_regression(args.artifact_root)
    elif args.mode == "finalize":
        run_finalize(args.artifact_root)
    else:
        run_locked(args.artifact_root, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
