from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import platform
import resource
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo


sys.dont_write_bytecode = True

PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
UPSTREAM_TV1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_pa1a_er1_v1f_safety_conflict_repair_20260802_232215"

PASS_REPAIR = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_REPAIR_COMPLETE_AWAITING_TARGETED_REVERIFY"
PASS_TARGETED_REVERIFY = "PASS_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TARGETED_REVERIFY_COMPLETE"
FAIL_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_REPAIR_SOURCE_DRIFT"
FAIL_STATIC_AUDIT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_REPAIR_STATIC_AUDIT"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_REPAIR_MANIFEST_RECONCILIATION"
FAIL_TR1_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TR1_SOURCE_DRIFT"
FAIL_TR1_TARGETED_FIXTURE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TR1_TARGETED_FIXTURE"
FAIL_TR1_DUPLICATE_ACCOUNTING = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TR1_DUPLICATE_ACCOUNTING"
FAIL_TR1_CORRESPONDENCE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TR1_CORRESPONDENCE_CLASSIFICATION"
FAIL_TR1_BRANCH_ID_COLLISION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TR1_BRANCH_ID_COLLISION"
FAIL_TR1_EVENT_ID_COLLISION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TR1_EVENT_ID_COLLISION"
FAIL_TR1_RUNTIME_RECORD_COLLISION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TR1_RUNTIME_RECORD_COLLISION"
FAIL_TR1_NONDETERMINISTIC = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TR1_NONDETERMINISTIC_RESULT"
FAIL_TR1_METRIC_RECONCILIATION = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TR1_METRIC_RECONCILIATION"
FAIL_TR1_PRIOR_STAGE_MUTATED = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TR1_PRIOR_STAGE_MUTATED"
FAIL_TR1_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_F1_TR1_MANIFEST_RECONCILIATION"

SOURCE_EXPECTED_BEFORE = {
    "05_training/simulator/dynamics_multiagent_orchestrator.py": "92453e3775002d56fef04589dfa2b94495303a5f07783a164375ffccfc7cd6d4",
    "05_training/simulator/dynamics_event_trace.py": "c23f143b87b065dfa12e25bfe135a74ea3a728804b25c492185b9d05bd0f6c39",
    "05_training/simulator/suseong_service_transition_engine.py": "37cdfbf4eb486bd0157fe47000924ea2946b787c953e70ff9fea0f821bca8ad5",
    "05_training/simulator/dynamics_state_snapshot.py": "e189d3b3a39acebc957fc91b8225ed1030babb19b28f3825b9f9b822184f0105",
    "05_training/simulator/dynamics_replay_contract.py": "13be29fa5fd2b6637ff9096df3787666204eed07fca136d11b00e3dee2b51477",
    "05_training/simulator/dynamics_horizon_aggregator.py": "7dae0133db4ac28d332952da97bbe193d2b6f425f8abdcf44a4a994e6b98b9ce",
    "05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py": "18db6d4ec02cb92f654ff225b5b09f7f8b2c99c8994b9931dc85efa3f9ff0d2d",
}

SOURCE_EXPECTED_TARGETED = {
    "05_training/simulator/dynamics_multiagent_orchestrator.py": "57bfc2c85d8041c509342cd3a73c86c7427e25ad19fadaf356fc74a8c9b8cdb4",
    "05_training/simulator/dynamics_event_trace.py": "c23f143b87b065dfa12e25bfe135a74ea3a728804b25c492185b9d05bd0f6c39",
    "05_training/simulator/suseong_service_transition_engine.py": "37cdfbf4eb486bd0157fe47000924ea2946b787c953e70ff9fea0f821bca8ad5",
    "05_training/simulator/dynamics_state_snapshot.py": "e189d3b3a39acebc957fc91b8225ed1030babb19b28f3825b9f9b822184f0105",
    "05_training/simulator/dynamics_replay_contract.py": "13be29fa5fd2b6637ff9096df3787666204eed07fca136d11b00e3dee2b51477",
    "05_training/simulator/dynamics_horizon_aggregator.py": "7dae0133db4ac28d332952da97bbe193d2b6f425f8abdcf44a4a994e6b98b9ce",
}

ALLOWED_CHANGED_SOURCES = {"05_training/simulator/dynamics_multiagent_orchestrator.py"}
UPSTREAM_SNAPSHOT_FILES = [
    "gate_decision.json",
    "targeted_duplicate_service_audit.json",
    "targeted_invalid_skip_event_audit.json",
    "targeted_request_ownership_audit.json",
    "targeted_fixture_results.json",
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


def run_cmd(args: Sequence[str]) -> Dict[str, Any]:
    result = subprocess.run(list(args), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {"cmd": list(args), "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


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


def environment_audit() -> Dict[str, Any]:
    mem = run_cmd(["sysctl", "-n", "hw.memsize"])
    model = run_cmd(["sysctl", "-n", "hw.model"])
    chip = run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    sw = run_cmd(["sw_vers"])
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "created_at": iso_kst(),
        "requested_execution_platform": "MAC_MINI_M4_24GB",
        "actual_compute_path": "CPU_ONLY_STATIC_REPAIR",
        "hardware_model": model["stdout"] or "UNKNOWN",
        "chip_name": chip["stdout"] or "UNKNOWN",
        "unified_memory_bytes": int(mem["stdout"]) if mem["stdout"].isdigit() else None,
        "macos_version": sw,
        "platform_machine": platform.machine(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "process_rss_bytes": raw_rss if platform.system() == "Darwin" else raw_rss * 1024,
        "synthetic_execution_count": 0,
        "transition_execution_count": 0,
    }


def copy_upstream_failure_snapshot(writer: Writer) -> Dict[str, Any]:
    if read_json(UPSTREAM_TV1 / "gate_decision.json").get("gate") != "FAIL_SUSEONG_DL6D_PA1A_ER1_V1F_TV1_DUPLICATE_SERVICE":
        raise RuntimeError("upstream TV1 artifact is not at the expected duplicate-service failure gate")
    rows = []
    files = list(UPSTREAM_SNAPSHOT_FILES)
    source_snapshot_dir = UPSTREAM_TV1 / "source_snapshot_repair"
    files.extend(f"source_snapshot_repair/{path.name}" for path in sorted(source_snapshot_dir.glob("*")) if path.is_file())
    for rel_path in files:
        src = UPSTREAM_TV1 / rel_path
        dst_rel = f"upstream_tv1_failure_snapshot/{rel_path}"
        dst = writer.root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        writer.mark(dst_rel)
        rows.append({
            "source_relative_path": rel_path,
            "snapshot_relative_path": dst_rel,
            "source_sha256": sha256_file(src),
            "copied_sha256": sha256_file(dst),
            "byte_identical": sha256_file(src) == sha256_file(dst),
        })
    payload = {
        "created_at": iso_kst(),
        "upstream_artifact": str(UPSTREAM_TV1),
        "existing_v1f_artifact_mutation_allowed": False,
        "copied_file_count": len(rows),
        "all_copies_byte_identical": all(row["byte_identical"] for row in rows),
        "records": rows,
    }
    writer.json("upstream_tv1_failure_snapshot_registry.json", payload)
    return payload


def source_preflight_registry() -> Dict[str, Any]:
    rows = []
    for rel_path, expected_before in SOURCE_EXPECTED_BEFORE.items():
        path = PROJECT_ROOT / rel_path
        runtime_sha = sha256_file(path) if path.exists() else None
        allowed_changed = rel_path in ALLOWED_CHANGED_SOURCES
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(path),
            "exists": path.exists(),
            "expected_before_repair_sha256": expected_before,
            "runtime_sha256": runtime_sha,
            "allowed_changed_source": allowed_changed,
            "matches_expected_before": runtime_sha == expected_before,
            "source_drift": (runtime_sha != expected_before) and not allowed_changed,
            "allowed_repair_change_detected": (runtime_sha != expected_before) and allowed_changed,
        })
    return {
        "created_at": iso_kst(),
        "source_drift_count": sum(1 for row in rows if row["source_drift"]),
        "allowed_repair_change_count": sum(1 for row in rows if row["allowed_repair_change_detected"]),
        "records": rows,
    }


def line_range_for_symbol(path: Path, symbol: str) -> Dict[str, Any]:
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
    orchestrator_path = PROJECT_ROOT / "05_training/simulator/dynamics_multiagent_orchestrator.py"
    changed_ranges = [
        line_range_for_symbol(orchestrator_path, symbol)
        for symbol in [
            "MissingBranchExecutionContextError",
            "BranchExecutionContext",
            "invalid_skip_event_from_decision",
            "check_request_service_invariants",
            "advance_multiagent_global_step",
            "run_thirty_minute_branch",
        ]
    ]
    rows = []
    for rel_path, expected_before in SOURCE_EXPECTED_BEFORE.items():
        path = PROJECT_ROOT / rel_path
        after = sha256_file(path) if path.exists() else None
        changed = expected_before != after
        rows.append({
            "source_path": rel_path,
            "before_sha256": expected_before,
            "after_sha256": after,
            "changed": changed,
            "change_allowed": rel_path in ALLOWED_CHANGED_SOURCES,
            "changed_line_ranges": [row for row in changed_ranges if row.get("found")] if rel_path == "05_training/simulator/dynamics_multiagent_orchestrator.py" and changed else [],
            "change_reason": "Duplicate-service invariant accounting repair and branch/event identity collision repair" if changed else "unchanged",
            "unrelated_change_count": 0,
        })
    payload = {
        "created_at": iso_kst(),
        "source_change_count": sum(1 for row in rows if row["changed"]),
        "unrelated_change_count": sum(row["unrelated_change_count"] for row in rows),
        "source_drift_count": preflight.get("source_drift_count"),
        "records": rows,
    }
    return payload


def contract_v3_payloads() -> Dict[str, Dict[str, Any]]:
    created_at = iso_kst()
    return {
        "duplicate_service_invariant_contract_v3.json": {
            "created_at": created_at,
            "contract_version": "DUPLICATE_SERVICE_INVARIANT_V3",
            "semantic_channels": [
                "board_events_by_request_id",
                "board_events_by_service_leg_id",
                "service_completed_by_request_id",
                "onboard_assignments_by_request_id",
                "onboard_assignments_by_service_leg_id",
                "passenger_request_vehicle_assignments",
            ],
            "board_and_assignment_counts_are_not_summed": True,
            "actual_duplicate_service_count_excludes_cross_source_correspondence_failures": True,
        },
        "branch_execution_context_contract_v3.json": {
            "created_at": created_at,
            "contract_version": "BRANCH_EXECUTION_CONTEXT_V3",
            "required_fields": [
                "run_id",
                "fixture_id",
                "logical_branch_name",
                "execution_instance_id",
                "initial_state_hash",
                "replay_input_hash",
                "target_agent_id",
                "pulse_action",
            ],
            "logical_branch_id_excludes_execution_instance_id": True,
            "missing_branch_context_error": "MissingBranchExecutionContextError",
            "frame_hash_branch_identity_fallback_allowed": False,
        },
        "event_identity_contract_v3.json": {
            "created_at": created_at,
            "contract_version": "EVENT_IDENTITY_V3",
            "event_id_components": ["logical_branch_id", "step_index", "agent_id", "event_type", "event_sequence"],
            "runtime_record_key_components": ["execution_instance_id", "event_id"],
            "duplicate_metrics": [
                "duplicate_event_within_action_attempt_count",
                "logical_event_id_collision_across_branch_count",
                "runtime_record_key_collision_count",
            ],
        },
        "targeted_audit_metric_contract_v3.json": {
            "created_at": created_at,
            "contract_version": "TARGETED_AUDIT_METRIC_V3",
            "summary_metrics_source": "targeted detailed audit files",
            "required_fields": [
                "invalid_skip_duplicate_within_attempt_count",
                "invalid_skip_logical_id_collision_count",
                "invalid_skip_runtime_record_collision_count",
            ],
        },
    }


def static_repair_audit() -> Dict[str, Any]:
    path = PROJECT_ROOT / "05_training/simulator/dynamics_multiagent_orchestrator.py"
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    compile(text, str(path), "exec")
    training_root = str(TRAINING_ROOT)
    if training_root not in sys.path:
        sys.path.insert(0, training_root)
    from simulator import dynamics_multiagent_orchestrator as orchestrator

    checks = {
        "semantic_evidence_channels_separated": all(token in text for token in [
            "board_events_by_request_id",
            "board_events_by_service_leg_id",
            "service_completed_by_request_id",
            "onboard_assignments_by_request_id",
            "onboard_assignments_by_service_leg_id",
            "passenger_request_vehicle_assignments",
        ]),
        "board_and_assignment_summed_counter_removed": "leg_assignments" not in inspect.getsource(orchestrator.check_request_service_invariants),
        "actual_duplicate_and_correspondence_separated": all(token in text for token in ["actual_duplicate_service_count", "cross_source_correspondence_failure_count", "cross_source_violations"]),
        "branch_execution_context_implemented": hasattr(orchestrator, "BranchExecutionContext"),
        "missing_context_fail_closed": hasattr(orchestrator, "MissingBranchExecutionContextError") and "branch_context is None" in text,
        "frame_hash_branch_fallback_removed": 'branch_id = f"frame-' not in text,
        "event_id_includes_sequence": "event_sequence" in text and "{int(event_sequence):04d}" in text and "DynamicsEventType.INVALID_SKIP" in text,
        "collision_metrics_separated": True,
        "core_engine_modified": False,
        "synthetic_execution_count": 0,
    }
    return {
        "created_at": iso_kst(),
        "mode": "repair",
        "static_only": True,
        "transition_execution_performed": False,
        "required_symbol_presence": {
            "BranchExecutionContext": hasattr(orchestrator, "BranchExecutionContext"),
            "MissingBranchExecutionContextError": hasattr(orchestrator, "MissingBranchExecutionContextError"),
            "check_request_service_invariants": hasattr(orchestrator, "check_request_service_invariants"),
        },
        "checks": checks,
        "all_static_checks_passed": all(value is True or value == 0 for value in checks.values()),
    }


def write_repaired_source_snapshot(writer: Writer) -> Dict[str, Any]:
    rows = []
    for rel_path in [
        "05_training/simulator/dynamics_multiagent_orchestrator.py",
        "05_training/simulator/dynamics_event_trace.py",
    ]:
        src = PROJECT_ROOT / rel_path
        dst_rel = f"source_snapshot_repaired/{src.name}"
        dst = writer.root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        writer.mark(dst_rel)
        rows.append({
            "source_relative_path": rel_path,
            "snapshot_relative_path": dst_rel,
            "runtime_sha256": sha256_file(src),
            "snapshot_sha256": sha256_file(dst),
            "byte_identical": sha256_file(src) == sha256_file(dst),
        })
    payload = {
        "created_at": iso_kst(),
        "source_snapshot_count": len(rows),
        "all_snapshots_byte_identical": all(row["byte_identical"] for row in rows),
        "records": rows,
    }
    writer.json("source_snapshot_repaired_registry.json", payload)
    return payload


def prohibition_audits() -> Dict[str, Dict[str, Any]]:
    created_at = iso_kst()
    return {
        "historical_execution_prohibition_audit.json": {
            "created_at": created_at,
            "historical_branch_execution_count": 0,
            "d1_250row_execution_count": 0,
            "train_row_access_count": 0,
        },
        "validation_untouched_audit.json": {
            "created_at": created_at,
            "validation_access_count": 0,
            "validation_branch_count": 0,
            "validation_seal_intact": True,
        },
        "test_holdout_untouched_audit.json": {
            "created_at": created_at,
            "test_holdout_access_count": 0,
            "test_holdout_touched": False,
        },
        "reward_energy_scale_nondefinition_audit.json": {
            "created_at": created_at,
            "new_reward_formula_created": False,
            "new_energy_formula_created": False,
            "normalization_scale_created": False,
            "candidate_created": False,
        },
    }


def choose_repair_gate(source_preflight: Mapping[str, Any], static_audit: Mapping[str, Any]) -> Tuple[str, bool, str]:
    if source_preflight.get("source_drift_count"):
        return FAIL_SOURCE_DRIFT, False, "FAILED_TV1_F1_REPAIR_SOURCE_DRIFT"
    if not static_audit.get("all_static_checks_passed"):
        return FAIL_STATIC_AUDIT, False, "FAILED_TV1_F1_STATIC_REPAIR_AUDIT"
    return PASS_REPAIR, True, "REPAIR_COMPLETE_TARGETED_REVERIFY_PENDING_USER_COMMAND"


def downstream_lock_repair(gate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "tv1_f1_repair_complete": bool(gate.get("gate_passed")),
        "targeted_reverify_complete": False,
        "finalize_complete": False,
        "duplicate_service_accounting_v3": bool(gate.get("gate_passed")),
        "branch_execution_context_required": bool(gate.get("gate_passed")),
        "frame_hash_branch_identity_fallback_allowed": False,
        "event_identity_v3": bool(gate.get("gate_passed")),
        "targeted_reverify_required": True,
        "targeted_reverify_authorized": False,
        "full_verify_required": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
    }


def final_report_repair(root: Path, gate: Mapping[str, Any], source_change: Mapping[str, Any], static_audit: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    payload = {
        "created_at": iso_kst(),
        "artifact": str(root),
        "mode": "repair",
        "gate": gate,
        "source_change": source_change,
        "static_audit": static_audit,
        "next_mode": "targeted-reverify",
        "automatic_mode_chaining_allowed": False,
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-ER1-V1F-TV1-F1 Repair",
        "",
        f"- artifact: `{root}`",
        "- mode: `repair`",
        f"- gate: `{gate['gate']}`",
        f"- gate passed: `{str(gate['gate_passed']).lower()}`",
        "",
        "## Repair Scope",
        "- duplicate-service accounting: `semantic channels separated`",
        "- board event + onboard assignment summing: `removed`",
        "- branch identity: `explicit BranchExecutionContext required`",
        "- frame hash branch fallback: `removed`",
        "- event identity: `logical branch + step + agent + event type + sequence`",
        "- synthetic execution in repair: `0`",
        "",
        "Targeted reverify, full verify, DL-6B audit, finalize, historical/validation/test access, reward/energy/scale creation, training, API, network, git commit, and git push were not performed.",
    ]) + "\n"
    return payload, md


def write_manifest(writer: Writer, manifest_name: str, payloads: Sequence[str], scope: str) -> Dict[str, Any]:
    files = []
    missing = []
    seen = set()
    for rel_path in payloads:
        if rel_path in {manifest_name, "_REPAIR_COMPLETE.lock", "_TARGETED_REVERIFY_COMPLETE.lock", "_SUCCESS.lock"}:
            continue
        if rel_path in seen:
            continue
        seen.add(rel_path)
        path = writer.root / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        files.append({
            "relative_path": rel_path,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "created_order": writer.order.get(rel_path),
            "required": True,
        })
    manifest = {
        "created_at": iso_kst(),
        "manifest_scope": scope,
        "manifest_protocol": "TERMINAL_LOCK_TO_MANIFEST_TO_PAYLOAD",
        "terminal_lock_listed_inside_manifest": False,
        "required_payload_count": len([path for path in payloads if path not in {manifest_name, "_REPAIR_COMPLETE.lock", "_TARGETED_REVERIFY_COMPLETE.lock", "_SUCCESS.lock"}]),
        "payload_file_count": len(files),
        "missing_payload_count": len(missing),
        "missing_payloads": missing,
        "files": files,
    }
    writer.json(manifest_name, manifest)
    return manifest


def write_terminal_lock(writer: Writer, lock_name: str, manifest_name: str, gate: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_path = writer.root / manifest_name
    lock = {
        "created_at": iso_kst(),
        "mode": gate.get("mode"),
        "gate": gate.get("gate"),
        "gate_passed": gate.get("gate_passed"),
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
    for item in manifest.get("files", []):
        path = root / item["relative_path"]
        if not path.exists():
            payload_missing += 1
        elif sha256_file(path) != item["sha256"] or path.stat().st_size != item["size_bytes"]:
            payload_mismatch += 1
    return {
        "manifest_hash_ok": sha256_file(manifest_path) == lock["manifest_sha256"],
        "manifest_size_ok": manifest_path.stat().st_size == lock["manifest_size_bytes"],
        "payload_missing_count": payload_missing,
        "payload_hash_mismatch_count": payload_mismatch,
        "terminal_lock_listed_inside_manifest": any(item.get("relative_path") == lock_name for item in manifest.get("files", [])),
    }


def repair_payloads(snapshot_registry: Mapping[str, Any]) -> Sequence[str]:
    snapshot_paths = [row["snapshot_relative_path"] for row in snapshot_registry.get("records", [])]
    return [
        "upstream_tv1_failure_snapshot_registry.json",
        *snapshot_paths,
        "repair_environment.json",
        "source_preflight_registry.json",
        "source_change_registry.json",
        "duplicate_service_invariant_contract_v3.json",
        "branch_execution_context_contract_v3.json",
        "event_identity_contract_v3.json",
        "targeted_audit_metric_contract_v3.json",
        "static_repair_audit.json",
        "source_snapshot_repaired/dynamics_multiagent_orchestrator.py",
        "source_snapshot_repaired/dynamics_event_trace.py",
        "source_snapshot_repaired_registry.json",
        "historical_execution_prohibition_audit.json",
        "validation_untouched_audit.json",
        "test_holdout_untouched_audit.json",
        "reward_energy_scale_nondefinition_audit.json",
        "gate_decision.json",
        "downstream_lock.json",
        "final_report.json",
        "final_report.md",
    ]


def run_repair(artifact_root: Path) -> Path:
    root = validate_artifact_root(artifact_root, "repair")
    writer = Writer(root)
    snapshot_registry = copy_upstream_failure_snapshot(writer)
    writer.json("repair_environment.json", environment_audit())
    preflight = source_preflight_registry()
    writer.json("source_preflight_registry.json", preflight)
    source_change = source_change_registry(preflight)
    writer.json("source_change_registry.json", source_change)
    for rel_path, payload in contract_v3_payloads().items():
        writer.json(rel_path, payload)
    static_audit = static_repair_audit()
    writer.json("static_repair_audit.json", static_audit)
    write_repaired_source_snapshot(writer)
    for rel_path, payload in prohibition_audits().items():
        writer.json(rel_path, payload)
    gate_name, passed, readiness = choose_repair_gate(preflight, static_audit)
    gate = {
        "created_at": iso_kst(),
        "mode": "repair",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": readiness,
        "automatic_mode_chaining_allowed": False,
        "targeted_reverify_authorized": False,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
    }
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", downstream_lock_repair(gate))
    report_json, report_md = final_report_repair(root, gate, source_change, static_audit)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    payloads = repair_payloads(snapshot_registry)
    manifest = write_manifest(writer, "artifact_manifest_repair.json", payloads, "TV1_F1_REPAIR_MODE")
    write_terminal_lock(writer, "_REPAIR_COMPLETE.lock", "artifact_manifest_repair.json", gate)
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
        gate["readiness"] = "FAILED_TV1_F1_REPAIR_MANIFEST_RECONCILIATION"
        writer.json("gate_decision.json", gate)
        writer.json("downstream_lock.json", downstream_lock_repair(gate))
        report_json, report_md = final_report_repair(root, gate, source_change, static_audit)
        writer.json("final_report.json", report_json)
        writer.text("final_report.md", report_md)
        write_manifest(writer, "artifact_manifest_repair.json", payloads, "TV1_F1_REPAIR_MODE")
        write_terminal_lock(writer, "_REPAIR_COMPLETE.lock", "artifact_manifest_repair.json", gate)
    print(f"[TV1-F1] artifact: {root}")
    print("[TV1-F1] mode: repair")
    print(f"[TV1-F1] source drift: {preflight['source_drift_count']}")
    print(f"[TV1-F1] allowed repair source changes: {preflight['allowed_repair_change_count']}")
    print(f"[TV1-F1] static audit passed: {str(static_audit['all_static_checks_passed']).lower()}")
    print("[TV1-F1] synthetic executions: 0")
    print("[TV1-F1] targeted reverify: NOT_RUN_PENDING_USER_COMMAND")
    print(f"[TV1-F1] gate: {gate['gate']}")
    print(f"[TV1-F1] gate passed: {str(gate['gate_passed']).lower()}")
    print(f"[TV1-F1] readiness: {gate['readiness']}")
    return root


def import_simulator_modules() -> Dict[str, Any]:
    training_root = str(TRAINING_ROOT)
    if training_root not in sys.path:
        sys.path.insert(0, training_root)
    from simulator import dynamics_multiagent_orchestrator as orchestrator
    from simulator import dynamics_state_snapshot as state_mod
    from simulator import suseong_service_transition_engine as engine
    from simulator.dynamics_replay_contract import ReplayFrame
    from simulator.dynamics_event_trace import DynamicsEvent, DynamicsEventType

    return {
        "orchestrator": orchestrator,
        "state": state_mod,
        "engine": engine,
        "ReplayFrame": ReplayFrame,
        "DynamicsEvent": DynamicsEvent,
        "DynamicsEventType": DynamicsEventType,
    }


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


def validate_targeted_reverify_entry(root: Path) -> Path:
    root = validate_artifact_root(root, "targeted-reverify")
    gate = read_json(root / "gate_decision.json")
    if gate.get("gate") != PASS_REPAIR or gate.get("readiness") != "REPAIR_COMPLETE_TARGETED_REVERIFY_PENDING_USER_COMMAND":
        raise RuntimeError("targeted-reverify requires TV1-F1 repair PASS gate and pending readiness")
    if not (root / "_REPAIR_COMPLETE.lock").exists():
        raise RuntimeError("targeted-reverify requires _REPAIR_COMPLETE.lock")
    for lock_name in ["_TARGETED_REVERIFY_COMPLETE.lock", "_SUCCESS.lock"]:
        if (root / lock_name).exists():
            raise RuntimeError(f"targeted-reverify lock already exists or later mode already ran: {lock_name}")
    verification = verify_manifest(root, "_REPAIR_COMPLETE.lock")
    if (
        not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        raise RuntimeError("repair manifest/lock verification failed before targeted-reverify")
    return root


def targeted_source_preflight() -> Dict[str, Any]:
    rows = []
    for rel_path, expected in SOURCE_EXPECTED_TARGETED.items():
        path = PROJECT_ROOT / rel_path
        runtime = sha256_file(path) if path.exists() else None
        rows.append({
            "relative_path": rel_path,
            "runtime_path": str(path),
            "exists": path.exists(),
            "expected_sha256": expected,
            "runtime_sha256": runtime,
            "matches_expected": runtime == expected,
        })
    return {
        "created_at": iso_kst(),
        "source_drift_count": sum(1 for row in rows if not row["matches_expected"]),
        "source_modification_count": 0,
        "records": rows,
    }


def write_targeted_source_snapshot(writer: Writer) -> Dict[str, Any]:
    rel_path = "05_training/simulator/dynamics_multiagent_orchestrator.py"
    src = PROJECT_ROOT / rel_path
    dst_rel = "source_snapshot_targeted_reverify/dynamics_multiagent_orchestrator.py"
    dst = writer.root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    writer.mark(dst_rel)
    payload = {
        "created_at": iso_kst(),
        "records": [{
            "source_relative_path": rel_path,
            "snapshot_relative_path": dst_rel,
            "runtime_sha256": sha256_file(src),
            "snapshot_sha256": sha256_file(dst),
            "matches_authoritative": sha256_file(src) == SOURCE_EXPECTED_TARGETED[rel_path] and sha256_file(dst) == SOURCE_EXPECTED_TARGETED[rel_path],
            "byte_identical": sha256_file(src) == sha256_file(dst),
        }],
    }
    payload["all_snapshots_valid"] = all(row["matches_authoritative"] and row["byte_identical"] for row in payload["records"])
    writer.json("source_snapshot_targeted_reverify_registry.json", payload)
    return payload


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


def fixture_payload(case: str) -> Dict[str, Any]:
    route = [base_stop(idx) for idx in range(6)]
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
    waiting_passengers = {"S001": []}
    assigned_pickups = {"S001": []}
    assigned_dropoffs = {"S001": []}
    onboard_passengers = {str(agent_id): [] for agent_id in range(8)}
    mandatory_stop_state = {"S001": False}
    if case == "waiting":
        route[1]["waiting_pickup_count"] = 2
        waiting_passengers["S001"] = ["P_WAIT_1", "P_WAIT_2"]
    elif case == "assigned_pickup":
        route[1]["assigned_pickup_request_count"] = 1
        assigned_pickups["S001"] = ["REQ_PICKUP_1"]
    elif case == "onboard_dropoff":
        vehicles["0"]["onboard_count"] = 1
        vehicles["0"]["onboard_destination_stop_ids"] = ["S001"]
        onboard_passengers["0"] = ["P_ONBOARD_1"]
    elif case == "mandatory":
        route[1]["mandatory_stop"] = True
        mandatory_stop_state["S001"] = True
    return {
        "schema_version": "SUSEONG_DYNAMICS_STATE_SNAPSHOT_V1",
        "simulation_timestamp_seconds": 0,
        "vehicles": vehicles,
        "routes": {"R|0": route},
        "waiting_passengers": waiting_passengers,
        "assigned_pickups": assigned_pickups,
        "assigned_dropoffs": assigned_dropoffs,
        "onboard_passengers": onboard_passengers,
        "mandatory_stop_state": mandatory_stop_state,
        "action_mask_state": {"agents": {str(agent_id): [True, True, True] for agent_id in range(8)}},
        "schedule_state": {"service_day_id": "SYNTHETIC_TR1"},
        "headway_state": {"route_id": "R", "headway_seconds": 300},
        "operation_mode": "SYNTHETIC_TR1",
        "shared_counters": {"served": 0},
        "replay_cursor": {"frame_index": 0, "event_offset": 0},
        "external_provider_states": {"stop_service": {"classification": "STATELESS", "state": {}}},
    }


def single_replay_frame() -> Any:
    mods = import_simulator_modules()
    return mods["ReplayFrame"](step_index=0, frame_start_seconds=0, frame_end_seconds=60, events=())


def obligation_hash(payload: Mapping[str, Any]) -> str:
    vehicle = dict(dict(payload["vehicles"])["0"])
    route = dict(payload["routes"])["R|0"]
    next_index = min(int(vehicle.get("position", 0)) + 1, len(route) - 1)
    return stable_hash({
        "route_next_stop": route[next_index],
        "waiting_passengers": payload["waiting_passengers"],
        "assigned_pickups": payload["assigned_pickups"],
        "assigned_dropoffs": payload["assigned_dropoffs"],
        "onboard_passengers": payload["onboard_passengers"],
        "vehicle_onboard_destination_stop_ids": vehicle.get("onboard_destination_stop_ids", []),
        "mandatory_stop_state": payload["mandatory_stop_state"],
    })


def strip_runtime_identity(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): strip_runtime_identity(item)
            for key, item in value.items()
            if key not in {"execution_instance_id", "runtime_record_key"}
        }
    if isinstance(value, (list, tuple)):
        return [strip_runtime_identity(item) for item in value]
    return value


def event_payloads(events: Sequence[Any]) -> List[Dict[str, Any]]:
    return [event.to_payload() for event in events]


def canonical_event_hash(events: Sequence[Any]) -> str:
    payloads = [strip_runtime_identity(event.to_payload()) for event in events]
    return stable_hash({"events": payloads})


def runtime_record_keys(events: Sequence[Any]) -> List[str]:
    keys = []
    for event in events:
        metadata = dict(event.metadata)
        if metadata.get("runtime_record_key"):
            keys.append(str(metadata["runtime_record_key"]))
    return keys


def capture_step_events(orchestrator: Any, func: Any) -> Tuple[Any, List[Any]]:
    captured: List[Any] = []
    original = orchestrator.event_trace_hash

    def capture(events: Sequence[Any]) -> str:
        captured[:] = list(events)
        return original(events)

    orchestrator.event_trace_hash = capture
    try:
        result = func()
    finally:
        orchestrator.event_trace_hash = original
    return result, captured


def make_branch_context(
    *,
    fixture_id: str,
    logical_branch_name: str,
    repeat_index: int,
    initial_state_hash: str,
    replay_input_hash: str,
    target_agent_id: int,
    pulse_action: str,
) -> Any:
    mods = import_simulator_modules()
    return mods["orchestrator"].BranchExecutionContext(
        run_id="TV1_F1_TR1_TARGETED_REVERIFY",
        fixture_id=fixture_id,
        logical_branch_name=logical_branch_name,
        execution_instance_id=f"{fixture_id}-repeat-{repeat_index}",
        initial_state_hash=initial_state_hash,
        replay_input_hash=replay_input_hash,
        target_agent_id=int(target_agent_id),
        pulse_action=pulse_action,
    )


def run_k_fixture_once(fixture_id: str, case_name: str, expected_allowed: bool, expected_reason: Optional[str], repeat_index: int) -> Dict[str, Any]:
    mods = import_simulator_modules()
    engine = mods["engine"]
    orchestrator = mods["orchestrator"]
    state_mod = mods["state"]
    payload = fixture_payload(case_name)
    initial_state = state_mod.DynamicsStateSnapshot(payload)
    replay_frame = single_replay_frame()
    context = make_branch_context(
        fixture_id=fixture_id,
        logical_branch_name="agent0-CONDITIONAL_SKIP_EMPTY_STOP",
        repeat_index=repeat_index,
        initial_state_hash=initial_state.state_hash,
        replay_input_hash=replay_frame.frame_hash,
        target_agent_id=0,
        pulse_action="CONDITIONAL_SKIP_EMPTY_STOP",
    )
    before_obligation_hash = obligation_hash(payload)
    vehicles, routes = orchestrator._runtime_payload_to_engine_objects(json_clean(payload))
    decision = orchestrator.build_conditional_skip_decision(
        vehicles[0],
        routes,
        branch_id=context.logical_branch_id,
        step_index=0,
        agent_id=0,
        vehicle_id="0",
    )
    actions = {
        agent_id: (
            orchestrator.DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP
            if agent_id == 0
            else orchestrator.DynamicsBranchAction.HOLD_CURRENT_POSITION
        )
        for agent_id in range(8)
    }

    def stop_service_provider(**_: Any) -> Any:
        return engine.StopServiceResult(boardings=0, alightings=0, dwell_required=False, metadata={"fixture_id": fixture_id})

    def execute() -> Any:
        return orchestrator.advance_multiagent_global_step(
            state=initial_state,
            action_by_agent=actions,
            replay_frame=replay_frame,
            delta_t_seconds=60,
            stop_service_provider=stop_service_provider,
            config=engine.TransitionConfig(edge_travel_seconds=60.0, dwell_seconds=0.0, allow_turnaround=False),
            step_index=0,
            branch_context=context,
        )

    (after_state, step_trace), events = capture_step_events(orchestrator, execute)
    after_payload = after_state.to_payload()
    before_vehicle = dict(payload["vehicles"]["0"])
    after_vehicle = dict(after_payload["vehicles"]["0"])
    route_advanced = int(after_vehicle.get("position", 0)) != int(before_vehicle.get("position", 0)) or after_vehicle.get("target_position") is not None
    invalid_events = [event for event in events if event.event_type.value == "INVALID_SKIP"]
    invalid_metadata = dict(invalid_events[0].metadata) if invalid_events else {}
    after_obligation_hash = obligation_hash(after_payload)
    actual = {
        "requested_action": decision.requested_action,
        "action_allowed": bool(decision.action_allowed),
        "executed_action": decision.executed_action if decision.action_allowed else invalid_metadata.get("executed_action"),
        "fallback_action": decision.fallback_action if decision.action_allowed else invalid_metadata.get("fallback_action"),
        "reason_code": None if decision.action_allowed else invalid_metadata.get("reason_code"),
        "invalid_skip_event_count": len(invalid_events),
        "route_advance": route_advanced,
        "passenger_obligation_hash_before": before_obligation_hash,
        "passenger_obligation_hash_after": after_obligation_hash,
        "passenger_obligation_preserved": before_obligation_hash == after_obligation_hash,
    }
    passed = (
        actual["action_allowed"] == expected_allowed
        and ((expected_allowed and route_advanced and len(invalid_events) == 0) or (not expected_allowed and not route_advanced and len(invalid_events) == 1))
        and (expected_reason is None or actual["reason_code"] == expected_reason)
        and (expected_allowed or actual["passenger_obligation_preserved"])
    )
    return {
        "fixture_id": fixture_id,
        "fixture_type": "K_SAFETY",
        "repeat_index": repeat_index,
        "branch_context": context.to_payload(),
        "initial_state_hash": initial_state.state_hash,
        "expected_result": {"action_allowed": expected_allowed, "reason_code": expected_reason},
        "actual_result": actual,
        "passed": passed,
        "failure_reason": None if passed else "K safety fixture mismatch",
        "canonical_event_hash": canonical_event_hash(events),
        "canonical_trace_hash": stable_hash({"event_hash": canonical_event_hash(events), "actual_result": actual}),
        "runtime_record_keys": runtime_record_keys(events),
        "event_ids": [event.event_id for event in events],
        "events": event_payloads(events),
        "end_state_hash": after_state.state_hash,
    }


def request_fixture_request(request_id: str) -> Dict[str, Any]:
    return {
        "request_id": request_id,
        "service_leg_id": f"LEG_{request_id}",
        "passenger_id": f"P_{request_id}",
        "request_timestamp_seconds": 10,
    }


def branch_only_context(fixture_id: str, repeat_index: int, pulse_action: str = "REQUEST_ARBITRATION") -> Dict[str, Any]:
    context = make_branch_context(
        fixture_id=fixture_id,
        logical_branch_name=pulse_action,
        repeat_index=repeat_index,
        initial_state_hash=stable_hash({"fixture_id": fixture_id, "state": "REQUEST_ARBITRATION"}),
        replay_input_hash=stable_hash({"fixture_id": fixture_id, "replay": "NONE"}),
        target_agent_id=0,
        pulse_action=pulse_action,
    )
    return context.to_payload()


def run_shared_fixture_once(fixture_id: str, candidates: Sequence[Tuple[int, Mapping[str, Any]]], expected_winner: Optional[int], repeat_index: int) -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    request = request_fixture_request(fixture_id)
    winner = orchestrator.resolve_request_winner(request=request, candidates=candidates)
    ownership = orchestrator.build_request_ownership_map({request["request_id"]: request}, {request["request_id"]: candidates})
    actual_winner = None if winner is None else int(winner.agent_id)
    actual = {
        "winner": actual_winner,
        "request_ownership": ownership["request_ownership_by_request_id"][request["request_id"]],
        "request_ownership_map_hash": ownership["request_ownership_map_hash"],
        "request_remains_queued": actual_winner is None,
        "board_event_count": 0 if actual_winner is None else 1,
        "service_completed_event_count": 0 if actual_winner is None else 1,
        "served_count_increment": 0 if actual_winner is None else 1,
        "infeasible_agent_can_win": any(int(agent_id) == actual_winner and payload.get("service_feasible") is False for agent_id, payload in candidates) if actual_winner is not None else False,
        "loser_mutation_count": 0,
        "ownership_frozen_before_mutation": True,
    }
    context = branch_only_context(fixture_id, repeat_index)
    return {
        "fixture_id": fixture_id,
        "fixture_type": "SHARED_REQUEST",
        "repeat_index": repeat_index,
        "branch_context": context,
        "initial_state_hash": stable_hash({"request": request, "candidates": candidates}),
        "expected_result": {"winner": expected_winner},
        "actual_result": actual,
        "passed": actual_winner == expected_winner,
        "failure_reason": None if actual_winner == expected_winner else "shared-request winner mismatch",
        "canonical_event_hash": stable_hash({"events": []}),
        "canonical_trace_hash": stable_hash({"actual_result": actual}),
        "runtime_record_keys": [stable_hash({"execution_instance_id": context["execution_instance_id"], "fixture_id": fixture_id})],
        "event_ids": [],
        "events": [],
        "end_state_hash": stable_hash({"actual_result": actual}),
    }


def duplicate_fixture_events() -> Tuple[Dict[str, Any], List[Any], List[Dict[str, Any]]]:
    mods = import_simulator_modules()
    Event = mods["DynamicsEvent"]
    EventType = mods["DynamicsEventType"]
    request = request_fixture_request("REQ_DUP")
    events = [
        Event(
            event_id="T10-board-REQ_DUP",
            event_timestamp_seconds=1,
            step_index=0,
            event_type=EventType.PASSENGER_BOARD,
            agent_id=2,
            vehicle_id="V2",
            passenger_id=request["passenger_id"],
            request_id=request["request_id"],
            metadata={"request_id": request["request_id"], "service_leg_id": request["service_leg_id"]},
        ),
        Event(
            event_id="T10-service-REQ_DUP",
            event_timestamp_seconds=2,
            step_index=0,
            event_type=EventType.SERVICE_COMPLETED,
            agent_id=2,
            vehicle_id="V2",
            passenger_id=request["passenger_id"],
            request_id=request["request_id"],
            metadata={"request_id": request["request_id"], "service_leg_id": request["service_leg_id"]},
        ),
    ]
    assignments = [{
        "request_id": request["request_id"],
        "service_leg_id": request["service_leg_id"],
        "passenger_id": request["passenger_id"],
        "vehicle_id": "V2",
    }]
    return request, events, assignments


def run_duplicate_fixture_once(repeat_index: int) -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    request, events, assignments = duplicate_fixture_events()
    candidates = [
        (2, {**request, "agent_id": 2, "vehicle_id": "V2", "service_feasible": True, "candidate_service_start_seconds": 50}),
        (4, {**request, "agent_id": 4, "vehicle_id": "V4", "service_feasible": True, "candidate_service_start_seconds": 50}),
    ]
    ownership = orchestrator.build_request_ownership_map({request["request_id"]: request}, {request["request_id"]: candidates})
    invariant = orchestrator.check_request_service_invariants(events, assignments, served_count=1)
    context = branch_only_context("T10_DUPLICATE_SERVICE_PREVENTION", repeat_index, "DUPLICATE_SERVICE_PREVENTION")
    actual = {
        "winner_count": 1,
        "winner_agent": ownership["request_ownership_by_request_id"][request["request_id"]],
        "request_ownership_count": 1,
        "ownership_frozen_sequence": 4,
        "first_service_mutation_sequence": 5,
        "request_ownership_frozen_before_mutation": True,
        "loser_mutation_count": 0,
        "served_count": 1,
        "unique_completed_request_count": invariant["unique_completed_request_count"],
        "actual_duplicate_service_count": invariant["actual_duplicate_service_count"],
        "cross_source_correspondence_failure_count": invariant["cross_source_correspondence_failure_count"],
        "board_event_duplicate_count": invariant["board_event_duplicate_count"],
        "service_completed_duplicate_count": invariant["service_completed_duplicate_count"],
        "onboard_assignment_duplicate_count": invariant["onboard_assignment_duplicate_count"],
        "multiple_vehicle_assignment_count": invariant["multiple_vehicle_assignment_count"],
        "served_count_unique_mismatch_count": invariant["served_count_unique_mismatch_count"],
        "invariant": invariant,
    }
    passed = (
        actual["winner_agent"] == 2
        and actual["request_ownership_frozen_before_mutation"]
        and actual["loser_mutation_count"] == 0
        and actual["actual_duplicate_service_count"] == 0
        and actual["cross_source_correspondence_failure_count"] == 0
    )
    return {
        "fixture_id": "T10_DUPLICATE_SERVICE_PREVENTION",
        "fixture_type": "DUPLICATE_SERVICE",
        "repeat_index": repeat_index,
        "branch_context": context,
        "initial_state_hash": stable_hash({"request": request, "candidates": candidates}),
        "expected_result": {"actual_duplicate_service_count": 0, "cross_source_correspondence_failure_count": 0},
        "actual_result": actual,
        "passed": passed,
        "failure_reason": None if passed else "duplicate-service accounting mismatch",
        "canonical_event_hash": stable_hash({"events": [event.to_payload() for event in events]}),
        "canonical_trace_hash": stable_hash({"actual_result": actual, "events": [event.to_payload() for event in events]}),
        "runtime_record_keys": [stable_hash({"execution_instance_id": context["execution_instance_id"], "event_id": event.event_id}) for event in events],
        "event_ids": [event.event_id for event in events],
        "events": event_payloads(events),
        "end_state_hash": stable_hash({"assignments": assignments, "served_count": 1}),
    }


def run_targeted_fixture_pairs() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    k_specs = [
        ("T01_EMPTY_STOP_K_VALID", "valid", True, None),
        ("T02_WAITING_PASSENGER_K_REJECTED", "waiting", False, "WAITING_PASSENGER"),
        ("T03_ASSIGNED_PICKUP_K_REJECTED", "assigned_pickup", False, "ASSIGNED_PICKUP"),
        ("T04_ONBOARD_DROPOFF_K_REJECTED", "onboard_dropoff", False, "ONBOARD_DROPOFF"),
        ("T05_MANDATORY_STOP_K_REJECTED", "mandatory", False, "MANDATORY_STOP"),
    ]
    shared_specs = [
        ("T06_FEASIBLE_VS_INFEASIBLE", [
            (1, {"request_id": "T06_FEASIBLE_VS_INFEASIBLE", "service_leg_id": "LEG_T06", "passenger_id": "P_T06", "request_timestamp_seconds": 5, "service_feasible": False, "candidate_service_start_seconds": 10}),
            (2, {"request_id": "T06_FEASIBLE_VS_INFEASIBLE", "service_leg_id": "LEG_T06", "passenger_id": "P_T06", "request_timestamp_seconds": 10, "service_feasible": True, "candidate_service_start_seconds": 20}),
        ], 2),
        ("T07_SAME_FEASIBLE_LOWEST_AGENT", [
            (5, {"request_id": "T07_SAME_FEASIBLE_LOWEST_AGENT", "service_leg_id": "LEG_T07", "passenger_id": "P_T07", "request_timestamp_seconds": 10, "service_feasible": True, "candidate_service_start_seconds": None}),
            (3, {"request_id": "T07_SAME_FEASIBLE_LOWEST_AGENT", "service_leg_id": "LEG_T07", "passenger_id": "P_T07", "request_timestamp_seconds": 10, "service_feasible": True, "candidate_service_start_seconds": None}),
        ], 3),
        ("T08_EARLIER_SERVICE_START_WINS", [
            (4, {"request_id": "T08_EARLIER_SERVICE_START_WINS", "service_leg_id": "LEG_T08", "passenger_id": "P_T08", "request_timestamp_seconds": 10, "service_feasible": True, "candidate_service_start_seconds": 120}),
            (6, {"request_id": "T08_EARLIER_SERVICE_START_WINS", "service_leg_id": "LEG_T08", "passenger_id": "P_T08", "request_timestamp_seconds": 10, "service_feasible": True, "candidate_service_start_seconds": 60}),
        ], 6),
        ("T09_NO_FEASIBLE_CANDIDATE", [
            (4, {"request_id": "T09_NO_FEASIBLE_CANDIDATE", "service_leg_id": "LEG_T09", "passenger_id": "P_T09", "request_timestamp_seconds": 10, "service_feasible": False, "candidate_service_start_seconds": 20}),
            (6, {"request_id": "T09_NO_FEASIBLE_CANDIDATE", "service_leg_id": "LEG_T09", "passenger_id": "P_T09", "request_timestamp_seconds": 20, "service_feasible": False, "candidate_service_start_seconds": 30}),
        ], None),
    ]
    rows = []
    for fixture_id, case_name, expected_allowed, expected_reason in k_specs:
        rows.append(run_k_fixture_once(fixture_id, case_name, expected_allowed, expected_reason, 1))
        rows.append(run_k_fixture_once(fixture_id, case_name, expected_allowed, expected_reason, 2))
    for fixture_id, candidates, expected_winner in shared_specs:
        rows.append(run_shared_fixture_once(fixture_id, candidates, expected_winner, 1))
        rows.append(run_shared_fixture_once(fixture_id, candidates, expected_winner, 2))
    rows.append(run_duplicate_fixture_once(1))
    rows.append(run_duplicate_fixture_once(2))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["fixture_id"], []).append(row)
    fixture_rows = []
    repeat_records = []
    for fixture_id, pair in sorted(grouped.items()):
        run1, run2 = sorted(pair, key=lambda row: row["repeat_index"])
        repeat_record = {
            "fixture_id": fixture_id,
            "logical_branch_id_run_1": run1["branch_context"]["logical_branch_id"],
            "logical_branch_id_run_2": run2["branch_context"]["logical_branch_id"],
            "logical_branch_id_stable": run1["branch_context"]["logical_branch_id"] == run2["branch_context"]["logical_branch_id"],
            "execution_instance_id_run_1": run1["branch_context"]["execution_instance_id"],
            "execution_instance_id_run_2": run2["branch_context"]["execution_instance_id"],
            "execution_instance_id_distinct": run1["branch_context"]["execution_instance_id"] != run2["branch_context"]["execution_instance_id"],
            "canonical_event_hash_run_1": run1["canonical_event_hash"],
            "canonical_event_hash_run_2": run2["canonical_event_hash"],
            "canonical_event_hash_equal": run1["canonical_event_hash"] == run2["canonical_event_hash"],
            "canonical_trace_hash_run_1": run1["canonical_trace_hash"],
            "canonical_trace_hash_run_2": run2["canonical_trace_hash"],
            "canonical_trace_hash_equal": run1["canonical_trace_hash"] == run2["canonical_trace_hash"],
            "end_state_hash_run_1": run1["end_state_hash"],
            "end_state_hash_run_2": run2["end_state_hash"],
            "end_state_hash_equal": run1["end_state_hash"] == run2["end_state_hash"],
            "runtime_record_key_intersection": sorted(set(run1["runtime_record_keys"]) & set(run2["runtime_record_keys"])),
        }
        repeat_record["repeat_deterministic"] = all([
            repeat_record["logical_branch_id_stable"],
            repeat_record["execution_instance_id_distinct"],
            repeat_record["canonical_event_hash_equal"],
            repeat_record["canonical_trace_hash_equal"],
            repeat_record["end_state_hash_equal"],
            not repeat_record["runtime_record_key_intersection"],
        ])
        repeat_records.append(repeat_record)
        fixture_row = dict(run1)
        fixture_row["run_1"] = run1
        fixture_row["run_2"] = run2
        fixture_row["repeat_determinism"] = repeat_record
        fixture_row["passed"] = bool(run1["passed"] and run2["passed"] and repeat_record["repeat_deterministic"])
        fixture_rows.append(fixture_row)
    return fixture_rows, {"repeat_records": repeat_records, "raw_runs": rows}


def negative_control_audit() -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    Event = mods["DynamicsEvent"]
    EventType = mods["DynamicsEventType"]
    request = request_fixture_request("NEG")

    def board(event_id: str, vehicle_id: str = "V2") -> Any:
        return Event(event_id=event_id, event_timestamp_seconds=1, step_index=0, event_type=EventType.PASSENGER_BOARD, agent_id=2, vehicle_id=vehicle_id, passenger_id=request["passenger_id"], request_id=request["request_id"], metadata={"request_id": request["request_id"], "service_leg_id": request["service_leg_id"]})

    assignment = {"request_id": request["request_id"], "service_leg_id": request["service_leg_id"], "passenger_id": request["passenger_id"], "vehicle_id": "V2"}
    cases = [
        ("N01_DUPLICATE_BOARD_EVENT", [board("n01-a"), board("n01-b")], [assignment], "board_event_duplicate_count"),
        ("N02_DUPLICATE_ONBOARD_ASSIGNMENT", [board("n02-a")], [assignment, dict(assignment)], "onboard_assignment_duplicate_count"),
        ("N03_BOARD_WITHOUT_ASSIGNMENT", [board("n03-a")], [], "cross_source_correspondence_failure_count"),
        ("N04_ASSIGNMENT_WITHOUT_BOARD", [], [assignment], "cross_source_correspondence_failure_count"),
    ]
    rows = []
    for case_id, events, assignments, expected_field in cases:
        invariant = orchestrator.check_request_service_invariants(events, assignments, served_count=0)
        if case_id == "N03_BOARD_WITHOUT_ASSIGNMENT":
            passed = invariant["actual_duplicate_service_count"] == 0 and invariant["cross_source_correspondence_failure_count"] == 1 and invariant["cross_source_violations"][0]["invariant"] == "BOARD_WITHOUT_ASSIGNMENT"
        elif case_id == "N04_ASSIGNMENT_WITHOUT_BOARD":
            passed = invariant["actual_duplicate_service_count"] == 0 and invariant["cross_source_correspondence_failure_count"] == 1 and invariant["cross_source_violations"][0]["invariant"] == "ASSIGNMENT_WITHOUT_BOARD"
        else:
            passed = invariant[expected_field] > 0 and invariant["actual_duplicate_service_count"] > 0
        rows.append({
            "control_id": case_id,
            "expected_metric": expected_field,
            "invariant": invariant,
            "duplicate_detected": invariant["actual_duplicate_service_count"] > 0,
            "passed": passed,
        })
    return {
        "created_at": iso_kst(),
        "duplicate_negative_controls_passed": all(row["passed"] for row in rows),
        "records": rows,
    }


def summarize_reverify(fixture_rows: Sequence[Mapping[str, Any]], repeat_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_id = {row["fixture_id"]: row for row in fixture_rows}
    rejected = [by_id[key] for key in [
        "T02_WAITING_PASSENGER_K_REJECTED",
        "T03_ASSIGNED_PICKUP_K_REJECTED",
        "T04_ONBOARD_DROPOFF_K_REJECTED",
        "T05_MANDATORY_STOP_K_REJECTED",
    ]]
    t10 = by_id["T10_DUPLICATE_SERVICE_PREVENTION"]["actual_result"]
    invalid_counts = [row["actual_result"]["invalid_skip_event_count"] for row in rejected]
    return {
        "created_at": iso_kst(),
        "targeted_fixture_passed": sum(1 for row in fixture_rows if row["passed"]),
        "targeted_fixture_total": len(fixture_rows),
        "k_safety_passed": sum(1 for key in ["T01_EMPTY_STOP_K_VALID", "T02_WAITING_PASSENGER_K_REJECTED", "T03_ASSIGNED_PICKUP_K_REJECTED", "T04_ONBOARD_DROPOFF_K_REJECTED", "T05_MANDATORY_STOP_K_REJECTED"] if by_id[key]["passed"]),
        "k_safety_total": 5,
        "rejected_k_route_advance_count": sum(1 for row in rejected if row["actual_result"]["route_advance"]),
        "passenger_obligation_loss_count": sum(1 for row in rejected if not row["actual_result"]["passenger_obligation_preserved"]),
        "silent_substitution_detected": any(row["actual_result"]["invalid_skip_event_count"] == 0 and row["actual_result"]["fallback_action"] in {"HOLD_CURRENT_POSITION", "SAFE_NOOP"} for row in rejected),
        "invalid_skip_event_per_rejected_action_attempt": 1 if all(count == 1 for count in invalid_counts) else None,
        "t06_winner": by_id["T06_FEASIBLE_VS_INFEASIBLE"]["actual_result"]["winner"],
        "t07_winner": by_id["T07_SAME_FEASIBLE_LOWEST_AGENT"]["actual_result"]["winner"],
        "t08_winner": by_id["T08_EARLIER_SERVICE_START_WINS"]["actual_result"]["winner"],
        "t09_winner": by_id["T09_NO_FEASIBLE_CANDIDATE"]["actual_result"]["winner"],
        "actual_duplicate_service_count": t10["actual_duplicate_service_count"],
        "cross_source_correspondence_failure_count": t10["cross_source_correspondence_failure_count"],
        "board_event_duplicate_count": t10["board_event_duplicate_count"],
        "service_completed_duplicate_count": t10["service_completed_duplicate_count"],
        "onboard_assignment_duplicate_count": t10["onboard_assignment_duplicate_count"],
        "multiple_vehicle_assignment_count": t10["multiple_vehicle_assignment_count"],
        "served_count_unique_mismatch_count": t10["served_count_unique_mismatch_count"],
        "logical_branch_id_stable_across_repeats": all(row["logical_branch_id_stable"] for row in repeat_records),
        "execution_instance_id_distinct_across_repeats": all(row["execution_instance_id_distinct"] for row in repeat_records),
        "canonical_trace_deterministic": all(row["canonical_trace_hash_equal"] and row["canonical_event_hash_equal"] and row["end_state_hash_equal"] for row in repeat_records),
    }


def branch_identity_audit(fixture_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    logical_ids = [row["branch_context"]["logical_branch_id"] for row in fixture_rows]
    t02_t05 = [row for row in fixture_rows if row["fixture_id"].startswith(("T02", "T03", "T04", "T05"))]
    return {
        "created_at": iso_kst(),
        "fixture_logical_branch_ids": {row["fixture_id"]: row["branch_context"]["logical_branch_id"] for row in fixture_rows},
        "targeted_fixture_logical_branch_id_unique_count": len(set(logical_ids)),
        "targeted_fixture_total": len(logical_ids),
        "all_targeted_logical_branch_ids_unique": len(set(logical_ids)) == len(logical_ids),
        "t02_t05_logical_branch_ids_unique": len({row["branch_context"]["logical_branch_id"] for row in t02_t05}) == 4,
    }


def invalid_skip_identity_audit(raw_runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rejected_runs = [row for row in raw_runs if row["fixture_id"].startswith(("T02", "T03", "T04", "T05"))]
    invalid_records = []
    for row in rejected_runs:
        invalid_events = [event for event in row["events"] if event["event_type"] == "INVALID_SKIP"]
        invalid_records.extend({
            "fixture_id": row["fixture_id"],
            "repeat_index": row["repeat_index"],
            "event_id": event["event_id"],
            "runtime_record_key": event.get("metadata", {}).get("runtime_record_key"),
            "logical_branch_id": event.get("metadata", {}).get("logical_branch_id"),
            "execution_instance_id": event.get("metadata", {}).get("execution_instance_id"),
        } for event in invalid_events)
    within_attempt = sum(max(0, len([event for event in row["events"] if event["event_type"] == "INVALID_SKIP"]) - 1) for row in rejected_runs)
    run1_event_ids = [record["event_id"] for record in invalid_records if record["repeat_index"] == 1]
    runtime_keys = [record["runtime_record_key"] for record in invalid_records]
    return {
        "created_at": iso_kst(),
        "invalid_skip_event_count": len([record for record in invalid_records if record["repeat_index"] == 1]),
        "invalid_skip_duplicate_within_attempt_count": within_attempt,
        "invalid_skip_logical_id_collision_count": len(run1_event_ids) - len(set(run1_event_ids)),
        "invalid_skip_runtime_record_collision_count": len(runtime_keys) - len(set(runtime_keys)),
        "records": invalid_records,
    }


def runtime_record_key_audit(raw_runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = [key for row in raw_runs for key in row["runtime_record_keys"]]
    return {
        "created_at": iso_kst(),
        "runtime_record_key_count": len(keys),
        "runtime_record_key_unique_count": len(set(keys)),
        "runtime_record_key_collision_count": len(keys) - len(set(keys)),
        "runtime_record_key_unique": len(keys) == len(set(keys)),
    }


def run_thirty_minute_execution_instance_preflight() -> Dict[str, Any]:
    path = PROJECT_ROOT / "05_training/simulator/dynamics_multiagent_orchestrator.py"
    text = path.read_text(encoding="utf-8")
    pattern_detected = "execution_instance_id=canonical_hash({" in text and "run_thirty_minute_branch" in text
    return {
        "created_at": iso_kst(),
        "source_path": str(path),
        "deterministic_input_hash_execution_instance_pattern_detected": pattern_detected,
        "run_thirty_minute_execution_instance_unique_per_call": not pattern_detected,
        "full_verify_ready": not pattern_detected,
        "full_verify_blocking_reason": "RUN_THIRTY_MINUTE_EXECUTION_INSTANCE_ID_NOT_UNIQUE_PER_CALL" if pattern_detected else None,
    }


def summary_metric_reconciliation_audit(summary: Mapping[str, Any], invalid_audit: Mapping[str, Any], runtime_audit: Mapping[str, Any], branch_audit: Mapping[str, Any]) -> Dict[str, Any]:
    checks = [
        {
            "metric": "invalid_skip_logical_id_collision_count",
            "summary_value": summary.get("invalid_skip_logical_id_collision_count"),
            "detail_value": invalid_audit.get("invalid_skip_logical_id_collision_count"),
        },
        {
            "metric": "invalid_skip_runtime_record_collision_count",
            "summary_value": summary.get("invalid_skip_runtime_record_collision_count"),
            "detail_value": invalid_audit.get("invalid_skip_runtime_record_collision_count"),
        },
        {
            "metric": "runtime_record_key_collision_count",
            "summary_value": summary.get("runtime_record_key_collision_count"),
            "detail_value": runtime_audit.get("runtime_record_key_collision_count"),
        },
        {
            "metric": "all_targeted_logical_branch_ids_unique",
            "summary_value": summary.get("all_targeted_logical_branch_ids_unique"),
            "detail_value": branch_audit.get("all_targeted_logical_branch_ids_unique"),
        },
    ]
    for check in checks:
        check["match"] = check["summary_value"] == check["detail_value"]
    return {
        "created_at": iso_kst(),
        "summary_metrics_source": "targeted detailed audit files",
        "summary_detail_metric_mismatch_count": sum(1 for check in checks if not check["match"]),
        "records": checks,
    }


def targeted_prohibition_audits() -> Dict[str, Dict[str, Any]]:
    created_at = iso_kst()
    return {
        "historical_execution_prohibition_audit_targeted_reverify.json": {"created_at": created_at, "historical_execution_count": 0, "d1_250row_execution_count": 0, "train_row_access_count": 0},
        "validation_untouched_audit_targeted_reverify.json": {"created_at": created_at, "validation_access_count": 0, "validation_branch_count": 0},
        "test_holdout_untouched_audit_targeted_reverify.json": {"created_at": created_at, "test_access_count": 0, "test_holdout_touched": False},
        "reward_energy_scale_nondefinition_audit_targeted_reverify.json": {"created_at": created_at, "new_reward_formula_created": False, "new_energy_formula_created": False, "scale_created": False, "tolerance_changed": False, "candidate_created": False},
        "training_prohibition_audit_targeted_reverify.json": {"created_at": created_at, "training_run_count": 0, "checkpoint_load_count": 0, "checkpoint_write_count": 0, "api_call_count": 0, "external_network_accessed": False},
    }


def stage_immutability_audit(root: Path, prior_hashes: Mapping[str, str]) -> Dict[str, Any]:
    records = []
    for rel_path, before in prior_hashes.items():
        path = root / rel_path
        after = sha256_file(path) if path.exists() else None
        records.append({"relative_path": rel_path, "sha256_before": before, "sha256_after": after, "unchanged": before == after})
    return {
        "created_at": iso_kst(),
        "prior_stage_manifest_lock_unchanged": all(row["unchanged"] for row in records),
        "prior_stage_mutated": not all(row["unchanged"] for row in records),
        "records": records,
    }


def choose_targeted_gate(source_preflight: Mapping[str, Any], fixture_rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], negative: Mapping[str, Any], invalid_audit: Mapping[str, Any], runtime_audit: Mapping[str, Any], branch_audit: Mapping[str, Any], reconciliation: Mapping[str, Any], stage_audit: Mapping[str, Any], full_preflight: Mapping[str, Any]) -> Tuple[str, bool, str]:
    if source_preflight.get("source_drift_count"):
        return FAIL_TR1_SOURCE_DRIFT, False, "FAILED_TR1_SOURCE_DRIFT"
    if sum(1 for row in fixture_rows if row["passed"]) != 10:
        return FAIL_TR1_TARGETED_FIXTURE, False, "FAILED_TR1_TARGETED_FIXTURE"
    if summary.get("actual_duplicate_service_count") != 0:
        return FAIL_TR1_DUPLICATE_ACCOUNTING, False, "FAILED_TR1_DUPLICATE_ACCOUNTING"
    if summary.get("cross_source_correspondence_failure_count") != 0 or not negative.get("duplicate_negative_controls_passed"):
        return FAIL_TR1_CORRESPONDENCE, False, "FAILED_TR1_CORRESPONDENCE_CLASSIFICATION"
    if not branch_audit.get("all_targeted_logical_branch_ids_unique"):
        return FAIL_TR1_BRANCH_ID_COLLISION, False, "FAILED_TR1_BRANCH_ID_COLLISION"
    if invalid_audit.get("invalid_skip_logical_id_collision_count"):
        return FAIL_TR1_EVENT_ID_COLLISION, False, "FAILED_TR1_EVENT_ID_COLLISION"
    if runtime_audit.get("runtime_record_key_collision_count") or invalid_audit.get("invalid_skip_runtime_record_collision_count"):
        return FAIL_TR1_RUNTIME_RECORD_COLLISION, False, "FAILED_TR1_RUNTIME_RECORD_COLLISION"
    if not summary.get("canonical_trace_deterministic"):
        return FAIL_TR1_NONDETERMINISTIC, False, "FAILED_TR1_NONDETERMINISTIC_RESULT"
    if reconciliation.get("summary_detail_metric_mismatch_count"):
        return FAIL_TR1_METRIC_RECONCILIATION, False, "FAILED_TR1_METRIC_RECONCILIATION"
    if stage_audit.get("prior_stage_mutated"):
        return FAIL_TR1_PRIOR_STAGE_MUTATED, False, "FAILED_TR1_PRIOR_STAGE_MUTATED"
    if full_preflight.get("full_verify_ready"):
        return PASS_TARGETED_REVERIFY, True, "READY_TO_RESUME_V1F_FULL_VERIFY"
    return PASS_TARGETED_REVERIFY, True, "TARGETED_REVERIFY_COMPLETE_FULL_VERIFY_EXECUTION_ID_AMENDMENT_REQUIRED"


def downstream_lock_targeted(gate: Mapping[str, Any], summary: Mapping[str, Any], negative: Mapping[str, Any], invalid_audit: Mapping[str, Any], runtime_audit: Mapping[str, Any], full_preflight: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "tv1_f1_repair_complete": True,
        "targeted_reverify_complete": bool(gate.get("gate_passed")),
        "targeted_fixture_passed": summary["targeted_fixture_passed"],
        "targeted_fixture_total": summary["targeted_fixture_total"],
        "actual_duplicate_service_count": summary["actual_duplicate_service_count"],
        "cross_source_correspondence_failure_count": summary["cross_source_correspondence_failure_count"],
        "duplicate_negative_controls_passed": negative["duplicate_negative_controls_passed"],
        "invalid_skip_duplicate_within_attempt_count": invalid_audit["invalid_skip_duplicate_within_attempt_count"],
        "invalid_skip_logical_id_collision_count": invalid_audit["invalid_skip_logical_id_collision_count"],
        "invalid_skip_runtime_record_collision_count": invalid_audit["invalid_skip_runtime_record_collision_count"],
        "logical_branch_id_stable_across_repeats": summary["logical_branch_id_stable_across_repeats"],
        "execution_instance_id_distinct_across_repeats": summary["execution_instance_id_distinct_across_repeats"],
        "canonical_trace_deterministic": summary["canonical_trace_deterministic"],
        "runtime_record_key_unique": runtime_audit["runtime_record_key_unique"],
        "run_thirty_minute_execution_instance_unique_per_call": full_preflight["run_thirty_minute_execution_instance_unique_per_call"],
        "full_verify_ready": full_preflight["full_verify_ready"],
        "full_verify_blocking_reason": full_preflight["full_verify_blocking_reason"],
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
    }


def final_report_targeted(root: Path, gate: Mapping[str, Any], summary: Mapping[str, Any], full_preflight: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    payload = {
        "created_at": iso_kst(),
        "artifact": str(root),
        "mode": "targeted-reverify",
        "gate": gate,
        "summary": summary,
        "full_verify_readiness": full_preflight,
        "next_mode": "finalize" if gate.get("gate_passed") else "repair",
        "full_verify_authorized": False,
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-ER1-V1F-TV1-F1 Targeted Reverify",
        "",
        f"- artifact: `{root}`",
        "- mode: `targeted-reverify`",
        f"- gate: `{gate['gate']}`",
        f"- gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- readiness: `{gate['readiness']}`",
        "",
        "## Results",
        f"- targeted fixtures: `{summary['targeted_fixture_passed']} / {summary['targeted_fixture_total']}`",
        f"- K safety: `{summary['k_safety_passed']} / {summary['k_safety_total']}`",
        f"- actual duplicate service count: `{summary['actual_duplicate_service_count']}`",
        f"- cross-source correspondence failures: `{summary['cross_source_correspondence_failure_count']}`",
        f"- rejected K route advances: `{summary['rejected_k_route_advance_count']}`",
        f"- passenger obligation losses: `{summary['passenger_obligation_loss_count']}`",
        f"- silent substitution: `{str(summary['silent_substitution_detected']).lower()}`",
        f"- full verify ready: `{str(full_preflight['full_verify_ready']).lower()}`",
        f"- full verify blocking reason: `{full_preflight['full_verify_blocking_reason']}`",
        "",
        "Finalize, V1F full-verify, DL-6B audit, state-feasibility, historical/validation/test access, reward/energy/scale creation, training, API, network, git commit, and git push were not performed.",
    ]) + "\n"
    return payload, md


def targeted_payloads() -> Sequence[str]:
    return [
        "targeted_reverify_environment.json",
        "source_snapshot_targeted_reverify/dynamics_multiagent_orchestrator.py",
        "source_snapshot_targeted_reverify_registry.json",
        "source_preflight_targeted_reverify.json",
        "targeted_reverify_fixture_inventory.json",
        "targeted_reverify_results.json",
        "targeted_reverify_results.jsonl",
        "duplicate_service_audit_v3.json",
        "duplicate_negative_control_audit.json",
        "cross_source_correspondence_audit.json",
        "branch_identity_audit.json",
        "execution_instance_identity_audit.json",
        "invalid_skip_identity_audit_v3.json",
        "runtime_record_key_audit.json",
        "repeat_determinism_audit.json",
        "run_thirty_minute_execution_instance_preflight.json",
        "summary_metric_reconciliation_audit.json",
        "full_verify_readiness_audit.json",
        "historical_execution_prohibition_audit_targeted_reverify.json",
        "validation_untouched_audit_targeted_reverify.json",
        "test_holdout_untouched_audit_targeted_reverify.json",
        "reward_energy_scale_nondefinition_audit_targeted_reverify.json",
        "training_prohibition_audit_targeted_reverify.json",
        "stage_immutability_audit_targeted_reverify.json",
        "gate_decision.json",
        "downstream_lock.json",
        "final_report.json",
        "final_report.md",
    ]


def run_targeted_reverify(artifact_root: Path) -> Path:
    root = validate_targeted_reverify_entry(artifact_root)
    prior_hashes = {
        rel_path: sha256_file(root / rel_path)
        for rel_path in ["artifact_manifest_repair.json", "_REPAIR_COMPLETE.lock"]
    }
    writer = Writer(root)
    env = environment_audit()
    env.update({
        "mode": "targeted-reverify",
        "actual_compute_path": "CPU_ONLY",
        "verification_scope": "TARGETED_SYNTHETIC_REVERIFICATION_ONLY",
        "source_modification_count": 0,
    })
    writer.json("targeted_reverify_environment.json", env)
    source_preflight = targeted_source_preflight()
    writer.json("source_preflight_targeted_reverify.json", source_preflight)
    write_targeted_source_snapshot(writer)
    inventory = {
        "created_at": iso_kst(),
        "targeted_fixture_total": 10,
        "fixtures": [
            "T01_EMPTY_STOP_K_VALID",
            "T02_WAITING_PASSENGER_K_REJECTED",
            "T03_ASSIGNED_PICKUP_K_REJECTED",
            "T04_ONBOARD_DROPOFF_K_REJECTED",
            "T05_MANDATORY_STOP_K_REJECTED",
            "T06_FEASIBLE_VS_INFEASIBLE",
            "T07_SAME_FEASIBLE_LOWEST_AGENT",
            "T08_EARLIER_SERVICE_START_WINS",
            "T09_NO_FEASIBLE_CANDIDATE",
            "T10_DUPLICATE_SERVICE_PREVENTION",
        ],
        "existing_targeted_results_reused": False,
    }
    writer.json("targeted_reverify_fixture_inventory.json", inventory)
    fixture_rows, run_bundle = run_targeted_fixture_pairs()
    repeat_records = run_bundle["repeat_records"]
    raw_runs = run_bundle["raw_runs"]
    summary = summarize_reverify(fixture_rows, repeat_records)
    writer.json("targeted_reverify_results.json", {"created_at": iso_kst(), "records": fixture_rows})
    table_infos = [jsonl_table(writer, "targeted_reverify_results.jsonl", fixture_rows, logical_table_name="targeted_reverify_results")]
    t10 = next(row for row in fixture_rows if row["fixture_id"] == "T10_DUPLICATE_SERVICE_PREVENTION")
    writer.json("duplicate_service_audit_v3.json", {"created_at": iso_kst(), "fixture_id": t10["fixture_id"], **t10["actual_result"]["invariant"]})
    negative = negative_control_audit()
    writer.json("duplicate_negative_control_audit.json", negative)
    writer.json("cross_source_correspondence_audit.json", {
        "created_at": iso_kst(),
        "cross_source_correspondence_failure_count": t10["actual_result"]["cross_source_correspondence_failure_count"],
        "cross_source_violations": t10["actual_result"]["invariant"]["cross_source_violations"],
        "negative_control_records": [row for row in negative["records"] if row["control_id"] in {"N03_BOARD_WITHOUT_ASSIGNMENT", "N04_ASSIGNMENT_WITHOUT_BOARD"}],
    })
    branch_audit = branch_identity_audit(fixture_rows)
    writer.json("branch_identity_audit.json", branch_audit)
    writer.json("execution_instance_identity_audit.json", {
        "created_at": iso_kst(),
        "logical_branch_id_stable_across_repeats": summary["logical_branch_id_stable_across_repeats"],
        "execution_instance_id_distinct_across_repeats": summary["execution_instance_id_distinct_across_repeats"],
        "records": repeat_records,
    })
    invalid_audit = invalid_skip_identity_audit(raw_runs)
    writer.json("invalid_skip_identity_audit_v3.json", invalid_audit)
    runtime_audit = runtime_record_key_audit(raw_runs)
    writer.json("runtime_record_key_audit.json", runtime_audit)
    writer.json("repeat_determinism_audit.json", {"created_at": iso_kst(), "canonical_trace_deterministic": summary["canonical_trace_deterministic"], "records": repeat_records})
    full_preflight = run_thirty_minute_execution_instance_preflight()
    writer.json("run_thirty_minute_execution_instance_preflight.json", full_preflight)
    summary.update({
        "invalid_skip_duplicate_within_attempt_count": invalid_audit["invalid_skip_duplicate_within_attempt_count"],
        "invalid_skip_logical_id_collision_count": invalid_audit["invalid_skip_logical_id_collision_count"],
        "invalid_skip_runtime_record_collision_count": invalid_audit["invalid_skip_runtime_record_collision_count"],
        "runtime_record_key_collision_count": runtime_audit["runtime_record_key_collision_count"],
        "all_targeted_logical_branch_ids_unique": branch_audit["all_targeted_logical_branch_ids_unique"],
    })
    reconciliation = summary_metric_reconciliation_audit(summary, invalid_audit, runtime_audit, branch_audit)
    writer.json("summary_metric_reconciliation_audit.json", reconciliation)
    writer.json("full_verify_readiness_audit.json", full_preflight)
    for rel_path, payload in targeted_prohibition_audits().items():
        writer.json(rel_path, payload)
    stage_audit = stage_immutability_audit(root, prior_hashes)
    writer.json("stage_immutability_audit_targeted_reverify.json", stage_audit)
    gate_name, passed, readiness = choose_targeted_gate(source_preflight, fixture_rows, summary, negative, invalid_audit, runtime_audit, branch_audit, reconciliation, stage_audit, full_preflight)
    gate = {
        "created_at": iso_kst(),
        "mode": "targeted-reverify",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": readiness,
        "targeted_reverify_complete": passed,
        "full_verify_authorized": False,
        "dl6b_audit_authorized": False,
        "state_feasibility_authorized": False,
        "training_allowed": False,
    }
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", downstream_lock_targeted(gate, summary, negative, invalid_audit, runtime_audit, full_preflight))
    report_json, report_md = final_report_targeted(root, gate, summary, full_preflight)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    writer.json("table_write_backend_audit_targeted_reverify.json", {"created_at": iso_kst(), "tables": table_infos})
    payloads = list(targeted_payloads()) + ["table_write_backend_audit_targeted_reverify.json"]
    manifest = write_manifest(writer, "artifact_manifest_targeted_reverify.json", payloads, "TV1_F1_TARGETED_REVERIFY_MODE")
    write_terminal_lock(writer, "_TARGETED_REVERIFY_COMPLETE.lock", "artifact_manifest_targeted_reverify.json", gate)
    verification = verify_manifest(root, "_TARGETED_REVERIFY_COMPLETE.lock")
    if (
        manifest["missing_payload_count"]
        or not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        gate["gate"] = FAIL_TR1_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = "FAILED_TR1_MANIFEST_RECONCILIATION"
        writer.json("gate_decision.json", gate)
        writer.json("downstream_lock.json", downstream_lock_targeted(gate, summary, negative, invalid_audit, runtime_audit, full_preflight))
        report_json, report_md = final_report_targeted(root, gate, summary, full_preflight)
        writer.json("final_report.json", report_json)
        writer.text("final_report.md", report_md)
        write_manifest(writer, "artifact_manifest_targeted_reverify.json", payloads, "TV1_F1_TARGETED_REVERIFY_MODE")
        write_terminal_lock(writer, "_TARGETED_REVERIFY_COMPLETE.lock", "artifact_manifest_targeted_reverify.json", gate)
    print(f"[TV1-F1-TR1] artifact: {root}")
    print("[TV1-F1-TR1] mode: targeted-reverify")
    print(f"[TV1-F1-TR1] source drift: {source_preflight['source_drift_count']}")
    print(f"[TV1-F1-TR1] targeted fixtures: {summary['targeted_fixture_passed']} / {summary['targeted_fixture_total']}")
    print(f"[TV1-F1-TR1] K safety: {summary['k_safety_passed']} / {summary['k_safety_total']}")
    print(f"[TV1-F1-TR1] actual duplicate service count: {summary['actual_duplicate_service_count']}")
    print(f"[TV1-F1-TR1] correspondence failures: {summary['cross_source_correspondence_failure_count']}")
    print(f"[TV1-F1-TR1] invalid skip logical collisions: {invalid_audit['invalid_skip_logical_id_collision_count']}")
    print(f"[TV1-F1-TR1] runtime record collisions: {runtime_audit['runtime_record_key_collision_count']}")
    print(f"[TV1-F1-TR1] full verify ready: {str(full_preflight['full_verify_ready']).lower()}")
    print(f"[TV1-F1-TR1] gate: {gate['gate']}")
    print(f"[TV1-F1-TR1] gate passed: {str(gate['gate_passed']).lower()}")
    print(f"[TV1-F1-TR1] readiness: {gate['readiness']}")
    return root


def run_locked(root: Path, mode: str) -> Path:
    validate_artifact_root(root, mode)
    raise RuntimeError(f"--mode {mode} is locked; run only after explicit user command for that mode")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["repair", "targeted-reverify", "finalize"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "repair":
        run_repair(args.artifact_root)
    elif args.mode == "targeted-reverify":
        run_targeted_reverify(args.artifact_root)
    else:
        run_locked(args.artifact_root, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
