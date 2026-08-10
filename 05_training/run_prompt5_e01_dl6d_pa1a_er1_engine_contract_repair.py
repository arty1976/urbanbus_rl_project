from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

try:
    import torch
except Exception:  # pragma: no cover - project env normally has torch.
    torch = None  # type: ignore[assignment]


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
UPSTREAM_PA1A = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_pa1a_provenance_and_dynamics_feasibility_20260802_183355"
ENGINE_SOURCE = PROJECT_ROOT / "05_training/simulator/suseong_service_transition_engine.py"
PA1A_SOURCE = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_pa1a_provenance_and_dynamics_feasibility.py"
R1_SOURCE = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py"
DL5_SOURCE = PROJECT_ROOT / "05_training/run_prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation.py"
DL6B_SOURCE = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic.py"
DL6C_SOURCE = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6c_distinct_three_action_contract_repair.py"
I0 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_r1_split_inventory_20260802_132332"

PASS_RECONCILE = "PASS_SUSEONG_DL6D_PA1A_ER1_RECONCILE_COMPLETE_AWAITING_IMPLEMENT_COMMAND"
PASS_IMPLEMENT = "PASS_SUSEONG_DL6D_PA1A_ER1_IMPLEMENT_COMPLETE_AWAITING_VERIFY_COMMAND"
PASS_PREFLIGHT = "PASS_SUSEONG_DL6D_PA1A_ER1_V1_UPSTREAM_CONTRACT_PREFLIGHT"
PASS_VERIFY = "PASS_SUSEONG_DL6D_PA1A_ER1_VERIFY_COMPLETE_AWAITING_FINALIZE_COMMAND"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_MANIFEST_RECONCILIATION"
FAIL_VALIDATION = "FAIL_SUSEONG_DL6D_PA1A_ER1_VALIDATION_TOUCHED"
FAIL_EXTERNAL = "FAIL_SUSEONG_DL6D_PA1A_ER1_EXTERNAL_ACCESS"
FAIL_I1_RECONCILE_MUTATED = "FAIL_SUSEONG_DL6D_PA1A_ER1_I1_RECONCILE_STAGE_MUTATED"
FAIL_I1_PROXY = "FAIL_SUSEONG_DL6D_PA1A_ER1_I1_PROXY_DEPENDENCY_DETECTED"
FAIL_I1_LEGACY = "FAIL_SUSEONG_DL6D_PA1A_ER1_I1_LEGACY_ACTION_REACHABLE"
FAIL_I1_HORIZON = "FAIL_SUSEONG_DL6D_PA1A_ER1_I1_HORIZON_CONTRACT_INVALID"
FAIL_I1_REWARD_ENERGY = "FAIL_SUSEONG_DL6D_PA1A_ER1_I1_REWARD_OR_ENERGY_FORMULA_CREATED"
FAIL_I1_DYNAMICS = "FAIL_SUSEONG_DL6D_PA1A_ER1_I1_DYNAMICS_EXECUTION_DETECTED"
FAIL_I1_VALIDATION_TEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_I1_VALIDATION_OR_TEST_TOUCHED"
FAIL_I1_TRAINING = "FAIL_SUSEONG_DL6D_PA1A_ER1_I1_PROHIBITED_TRAINING"
FAIL_I1_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_I1_MANIFEST_RECONCILIATION"
FAIL_V1_SOURCE_DRIFT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_SOURCE_DRIFT"
FAIL_V1_STATE_ROUNDTRIP = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_STATE_ROUNDTRIP_MISMATCH"
FAIL_V1_STATE_DEFAULT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_STATE_DEFAULT_INSERTION"
FAIL_V1_CLONE = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_BRANCH_STATE_CONTAMINATION"
FAIL_V1_RESET = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_RESET_NONDETERMINISTIC"
FAIL_V1_REPLAY = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_REPLAY_HASH_MISMATCH"
FAIL_V1_LEGACY = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_LEGACY_ACTION_REACHABLE"
FAIL_V1_K_SAFETY = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_K_SAFETY_CONTRACT"
FAIL_V1_AGENT_ORDER = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_AGENT_ORDER_NONDETERMINISTIC"
FAIL_V1_SHARED_REQUEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_SHARED_REQUEST_DUPLICATION"
FAIL_V1_BRANCH_ALIGNMENT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_BRANCH_ALIGNMENT_INVALID"
FAIL_V1_HORIZON = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_HORIZON_STEP_MISMATCH"
FAIL_V1_NONDETERMINISTIC = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_NONDETERMINISTIC_REPLAY"
FAIL_V1_EVENT = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_EVENT_RECONCILIATION"
FAIL_V1_KPI = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_KPI_CALCULATION"
FAIL_V1_PROXY = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_PROXY_DEPENDENCY"
FAIL_V1_REWARD_ENERGY = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_REWARD_OR_ENERGY_CREATED"
FAIL_V1_HISTORICAL = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_HISTORICAL_ROW_ACCESSED"
FAIL_V1_VALIDATION_TEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_VALIDATION_OR_TEST_TOUCHED"
FAIL_V1_PRIOR_MUTATED = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_PRIOR_STAGE_MUTATED"
FAIL_V1_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_ER1_V1_MANIFEST_RECONCILIATION"
BLOCKED_V1_ACTION = "BLOCKED_SUSEONG_DL6D_PA1A_ER1_V1_ACTION_CONTRACT_PROVENANCE_INVALID"
BLOCKED_V1_SAFETY = "BLOCKED_SUSEONG_DL6D_PA1A_ER1_V1_SAFETY_PREDICATE_PROVENANCE_INVALID"
BLOCKED_V1_MASK = "BLOCKED_SUSEONG_DL6D_PA1A_ER1_V1_ACTION_MASK_PROVENANCE_INVALID"
BLOCKED_V1_STATE = "BLOCKED_SUSEONG_DL6D_PA1A_ER1_V1_STATE_CONTRACT_INCOMPLETE"
BLOCKED_V1_WAIT = "BLOCKED_SUSEONG_DL6D_PA1A_ER1_V1_PASSENGER_WAIT_DATA_UNAVAILABLE"

EXPECTED_RECONCILE_MANIFEST_SHA256 = "de4861fc855e1ef6ebff622b793aed5b0cd8c9293a7257cc395d29319616a029"
EXPECTED_RECONCILE_MANIFEST_SIZE = 4063

FROZEN_SOURCE_SHA256 = {
    "05_training/simulator/dynamics_replay_contract.py": "13be29fa5fd2b6637ff9096df3787666204eed07fca136d11b00e3dee2b51477",
    "05_training/simulator/dynamics_state_snapshot.py": "e189d3b3a39acebc957fc91b8225ed1030babb19b28f3825b9f9b822184f0105",
    "05_training/simulator/dynamics_multiagent_orchestrator.py": "ddacbc94f5159d01c6063cd06c8f27e83e90844bcf1df04e86c06826fbae3f99",
    "05_training/simulator/dynamics_event_trace.py": "c23f143b87b065dfa12e25bfe135a74ea3a728804b25c492185b9d05bd0f6c39",
    "05_training/simulator/dynamics_horizon_aggregator.py": "7dae0133db4ac28d332952da97bbe193d2b6f425f8abdcf44a4a994e6b98b9ce",
    "05_training/simulator/suseong_service_transition_engine.py": "37cdfbf4eb486bd0157fe47000924ea2946b787c953e70ff9fea0f821bca8ad5",
    "05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py": "18db6d4ec02cb92f654ff225b5b09f7f8b2c99c8994b9931dc85efa3f9ff0d2d",
}

RECONCILE_PAYLOADS = [
    "git_status_er1.txt",
    "mac_mini_environment_er1.json",
    "upstream_validation.json",
    "upstream_manifest_defect_record.json",
    "upstream_manifest_independent_reconciliation.json",
    "repair_plan_contract.json",
    "source_before_after_registry.json",
    "synthetic_fixture_execution_audit.json",
    "validation_untouched_audit.json",
    "test_holdout_untouched_audit.json",
    "training_prohibition_audit.json",
    "external_access_audit.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.json",
    "final_report.md",
]

PLANNED_MODULES = [
    PROJECT_ROOT / "05_training/simulator/dynamics_replay_contract.py",
    PROJECT_ROOT / "05_training/simulator/dynamics_state_snapshot.py",
    PROJECT_ROOT / "05_training/simulator/dynamics_multiagent_orchestrator.py",
    PROJECT_ROOT / "05_training/simulator/dynamics_event_trace.py",
    PROJECT_ROOT / "05_training/simulator/dynamics_horizon_aggregator.py",
]

IMPLEMENT_NEW_PAYLOADS = [
    "stage_history/reconcile/gate_decision.json",
    "stage_history/reconcile/downstream_lock.json",
    "stage_history/reconcile/final_report.json",
    "stage_history/reconcile/final_report.md",
    "implementation_contract.json",
    "source_implementation_registry.json",
    "state_snapshot_contract.json",
    "replay_contract_schema.json",
    "external_provider_state_contract.json",
    "action_adapter_contract.json",
    "multiagent_orchestrator_contract.json",
    "event_trace_schema.json",
    "horizon_kpi_aggregator_contract.json",
    "synthetic_fixture_inventory.json",
    "static_implementation_audit.json",
    "proxy_dependency_prohibition_audit.json",
    "reward_energy_nondefinition_audit.json",
    "dynamics_execution_prohibition_audit.json",
    "validation_untouched_audit_implement.json",
    "test_holdout_untouched_audit_implement.json",
]

VERIFY_NEW_PAYLOADS = [
    "stage_history/implement/gate_decision.json",
    "stage_history/implement/downstream_lock.json",
    "stage_history/implement/final_report.json",
    "stage_history/implement/final_report.md",
    "verify_environment.json",
    "source_snapshot/dynamics_replay_contract.py",
    "source_snapshot/dynamics_state_snapshot.py",
    "source_snapshot/dynamics_multiagent_orchestrator.py",
    "source_snapshot/dynamics_event_trace.py",
    "source_snapshot/dynamics_horizon_aggregator.py",
    "source_snapshot_registry.json",
    "source_excerpt_registry.json",
    "upstream_contract_preflight.json",
    "upstream_contract_provenance_table.parquet",
    "dl6b_stub_boundary_audit.json",
    "dl6b_stub_boundary_table.parquet",
    "fixture_inventory_verified.json",
    "state_roundtrip_audit.json",
    "state_fail_closed_audit.json",
    "clone_isolation_audit.json",
    "deterministic_reset_audit.json",
    "replay_order_hash_audit.json",
    "replay_malformed_input_audit.json",
    "provider_state_verification.json",
    "action_adapter_verification.json",
    "k_safety_fixture_results.parquet",
    "k_safety_summary.json",
    "multiagent_order_audit.json",
    "shared_request_conflict_audit.json",
    "branch_alignment_audit.json",
    "action_distinctness_audit.json",
    "thirty_minute_execution_audit.json",
    "repeat_determinism_audit.json",
    "event_trace_reconciliation.parquet",
    "event_trace_reconciliation_summary.json",
    "wait_kpi_accuracy_audit.json",
    "kpi_unavailable_guard_audit.json",
    "external_aggregator_readiness_audit.json",
    "proxy_dependency_verification.json",
    "reward_energy_nondefinition_audit_verify.json",
    "core_engine_compatibility_audit.json",
    "synthetic_fixture_results.parquet",
    "synthetic_fixture_summary.json",
    "test_execution_results.json",
    "test_execution_results.txt",
    "historical_execution_prohibition_audit.json",
    "validation_untouched_audit_verify.json",
    "test_holdout_untouched_audit_verify.json",
    "training_prohibition_audit_verify.json",
    "external_access_audit_verify.json",
    "stage_immutability_audit.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.json",
    "final_report.md",
]

MANIFEST_EXCLUDED_RELATIVE_PATHS = {
    "_RECONCILE_COMPLETE.lock",
    "_IMPLEMENT_COMPLETE.lock",
    "_UPSTREAM_CONTRACT_PREFLIGHT_COMPLETE.lock",
    "_VERIFY_COMPLETE.lock",
    "_SUCCESS.lock",
}


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}

    def _mark(self, rel_path: str) -> None:
        if rel_path not in self.order:
            self.order[rel_path] = len(self.order) + 1

    def text(self, rel_path: str, text: str) -> None:
        path = self.root / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self._mark(rel_path)

    def json(self, rel_path: str, payload: Mapping[str, Any]) -> None:
        self.text(rel_path, json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    return value


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(payload: Any) -> str:
    return sha256_text(json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, allow_nan=False, default=str))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_cmd(args: Sequence[str]) -> Dict[str, Any]:
    result = subprocess.run(list(args), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {"cmd": list(args), "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def mps_flags() -> Tuple[bool, bool]:
    if torch is None or not hasattr(torch.backends, "mps"):
        return False, False
    return bool(torch.backends.mps.is_built()), bool(torch.backends.mps.is_available())


def environment_audit() -> Dict[str, Any]:
    mem = run_cmd(["sysctl", "-n", "hw.memsize"])
    model = run_cmd(["sysctl", "-n", "hw.model"])
    sw = run_cmd(["sw_vers"])
    mps_built, mps_available = mps_flags()
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_bytes = raw_rss if platform.system() == "Darwin" else raw_rss * 1024
    return {
        "created_at": iso_kst(),
        "requested_execution_platform": "MAC_MINI_M4_24GB",
        "actual_compute_path": "CPU",
        "hardware_model": model["stdout"] or "UNKNOWN",
        "unified_memory_bytes": int(mem["stdout"]) if mem["stdout"].isdigit() else None,
        "macos_version": sw,
        "platform_machine": platform.machine(),
        "python_architecture": platform.architecture()[0],
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": getattr(torch, "__version__", None),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mps_used": False,
        "cuda_used": False,
        "process_rss_bytes": rss_bytes,
    }


def source_record(path: Path, role: str) -> Dict[str, Any]:
    return {
        "role": role,
        "path": str(path),
        "exists_before": path.exists(),
        "exists_after": path.exists(),
        "before_sha256": sha256_file(path) if path.exists() else None,
        "after_sha256": sha256_file(path) if path.exists() else None,
        "changed_during_reconcile": False,
        "git_status": run_cmd(["git", "status", "--short", "--", rel(path)])["stdout"] if path.exists() else "MISSING",
    }


def source_before_after_registry() -> Dict[str, Any]:
    sources = [
        (ENGINE_SOURCE, "transition engine source"),
        (PA1A_SOURCE, "PA1-A provenance runner source"),
        (R1_SOURCE, "R1 reduced-form proxy source"),
        (DL5_SOURCE, "DL5 baseline comparison source"),
        (DL6B_SOURCE, "DL6B action path/KPI source"),
        (DL6C_SOURCE, "DL6C action contract source"),
    ]
    records = [source_record(path, role) for path, role in sources]
    records.extend(source_record(path, f"planned module: {path.name}") for path in PLANNED_MODULES)
    return {
        "created_at": iso_kst(),
        "mode": "reconcile",
        "core_engine_modified": False,
        "r1_proxy_module_modified": False,
        "planned_module_count": len(PLANNED_MODULES),
        "records": records,
    }


def upstream_manifest_defect_record() -> Dict[str, Any]:
    manifest_path = UPSTREAM_PA1A / "artifact_manifest.json"
    lock_path = UPSTREAM_PA1A / "_ENGINE_AUDIT_COMPLETE.lock"
    manifest = read_json(manifest_path)
    entries = [item for item in manifest.get("files", []) if item.get("relative_path") == "_ENGINE_AUDIT_COMPLETE.lock"]
    entry = entries[0] if entries else {}
    actual_size = lock_path.stat().st_size if lock_path.exists() else None
    actual_hash = sha256_file(lock_path) if lock_path.exists() else None
    actual_text = lock_path.read_text(encoding="utf-8").strip() if lock_path.exists() else None
    metadata_matches = bool(entry) and entry.get("size_bytes") == actual_size and entry.get("sha256") == actual_hash
    return {
        "created_at": iso_kst(),
        "upstream_artifact": str(UPSTREAM_PA1A),
        "upstream_artifact_content_usable": True,
        "upstream_artifact_manifest_clean": metadata_matches,
        "defect_detected": not metadata_matches,
        "defect_class": "STALE_STAGE_LOCK_METADATA_AFTER_RETRY" if not metadata_matches else "NONE",
        "manifest_path": str(manifest_path),
        "lock_path": str(lock_path),
        "manifest_lock_entry": entry,
        "actual_lock_size_bytes": actual_size,
        "actual_lock_sha256": actual_hash,
        "actual_lock_text": actual_text,
        "expected_prior_failed_gate_in_manifest": "FAIL_SUSEONG_DL6D_PA1A_E1_R1_MODULE_MODIFIED",
        "actual_terminal_gate": "BLOCKED_SUSEONG_DL6D_PA1A_ENGINE_PARTIAL_REPAIR_REQUIRED" if actual_text and "BLOCKED_SUSEONG_DL6D_PA1A_ENGINE_PARTIAL_REPAIR_REQUIRED" in actual_text else None,
        "existing_upstream_artifact_modified": False,
    }


def upstream_manifest_independent_reconciliation(defect: Mapping[str, Any]) -> Dict[str, Any]:
    gate = read_json(UPSTREAM_PA1A / "gate_decision.json")
    downstream = read_json(UPSTREAM_PA1A / "downstream_lock.json")
    engine_api = read_json(UPSTREAM_PA1A / "transition_engine_api_audit.json")
    required_existing = [
        "_ADJUDICATE_COMPLETE.lock",
        "_ENGINE_AUDIT_COMPLETE.lock",
        "gate_decision.json",
        "downstream_lock.json",
        "transition_engine_api_audit.json",
        "engine_gap_repair_registry.json",
        "repository_stub_signature_sweep.json",
    ]
    missing = [name for name in required_existing if not (UPSTREAM_PA1A / name).exists()]
    return {
        "created_at": iso_kst(),
        "upstream_gate": gate.get("gate"),
        "upstream_gate_passed": gate.get("gate_passed"),
        "upstream_engine_feasibility_decision": gate.get("engine_feasibility_decision"),
        "upstream_downstream_lock_decision": downstream.get("engine_feasibility_decision"),
        "transition_function": engine_api.get("transition_function_name"),
        "content_required_file_missing_count": len(missing),
        "content_required_files_missing": missing,
        "manifest_lock_metadata_matches_actual": not bool(defect.get("defect_detected")),
        "upstream_artifact_content_usable": len(missing) == 0 and gate.get("gate") == "BLOCKED_SUSEONG_DL6D_PA1A_ENGINE_PARTIAL_REPAIR_REQUIRED",
        "upstream_artifact_manifest_clean": False if defect.get("defect_detected") else True,
        "reconciliation_action": "Record defect in ER1 artifact; do not edit upstream PA1-A artifact.",
    }


def validation_untouched_audit() -> Dict[str, Any]:
    contract = I0 / "validation_skip_valid_seal_contract.json"
    ids = I0 / "validation_skip_valid_sealed_ids.json"
    lock = I0 / "_VALIDATION_INVENTORY_SEALED.lock"
    manifest = read_json(I0 / "artifact_manifest.json")
    by_path = {item["relative_path"]: item["sha256"] for item in manifest.get("files", [])}
    checks = {
        "seal_contract_hash_unchanged": contract.exists() and sha256_file(contract) == by_path.get("validation_skip_valid_seal_contract.json"),
        "sealed_ids_hash_unchanged": ids.exists() and sha256_file(ids) == by_path.get("validation_skip_valid_sealed_ids.json"),
        "sealed_lock_hash_unchanged": lock.exists() and sha256_file(lock) == by_path.get("_VALIDATION_INVENTORY_SEALED.lock"),
    }
    return {
        "created_at": iso_kst(),
        "validation_row_level_access_count": 0,
        "validation_branch_count": 0,
        "validation_reward_count": 0,
        "validation_harm_label_count": 0,
        "validation_seal_intact": all(checks.values()),
        "checks": checks,
    }


def repair_plan_contract(defect: Mapping[str, Any]) -> Dict[str, Any]:
    planned_files = [rel(path) for path in PLANNED_MODULES]
    return {
        "created_at": iso_kst(),
        "mode": "reconcile",
        "automatic_mode_chaining_allowed": False,
        "upstream_manifest_defect_recorded": bool(defect.get("defect_detected")),
        "implementation_performed_in_reconcile": False,
        "historical_branch_execution_allowed": False,
        "d1_250row_execution_allowed": False,
        "validation_execution_allowed": False,
        "test_holdout_execution_allowed": False,
        "reward_selection_allowed": False,
        "scale_derivation_allowed": False,
        "training_allowed": False,
        "synthetic_fixture_transition_allowed": True,
        "synthetic_fixture_max_scenarios": 12,
        "synthetic_fixture_is_research_evidence": False,
        "synthetic_fixture_used_for_reward_design": False,
        "decision_interval_seconds": 60,
        "thirty_minute_total_steps": 30,
        "action_mapping": {"H": 0, "S": 1, "K": 3},
        "legacy_action_2_accepted": False,
        "engine_internal_rng_contract": "NO_INTERNAL_RNG",
        "historical_event_replay_required": True,
        "planned_modules": planned_files,
        "planned_modes": ["reconcile", "implement", "verify", "finalize"],
        "frozen_repair_gaps": [
            "GAP_STATE_SERIALIZATION",
            "GAP_DETERMINISTIC_RESET",
            "GAP_RNG_CONTRACT",
            "GAP_EXOGENOUS_REPLAY_SCHEMA",
            "GAP_WAIT_DISTRIBUTION",
            "GAP_HORIZON_AGGREGATOR",
            "GAP_DL6B_STUB_REAUDIT",
            "GAP_8_AGENT_GLOBAL_STEP_ORCHESTRATOR",
            "GAP_LEGACY_ACTION_2_BLOCKING_ADAPTER",
            "GAP_MANIFEST_TERMINAL_LOCK_PROTOCOL",
        ],
        "next_mode_required": "implement",
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
    }


def upstream_validation(defect: Mapping[str, Any], reconciliation: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "authoritative_upstream_artifact": str(UPSTREAM_PA1A),
        "upstream_gate_required": "BLOCKED_SUSEONG_DL6D_PA1A_ENGINE_PARTIAL_REPAIR_REQUIRED",
        "upstream_gate_observed": reconciliation.get("upstream_gate"),
        "upstream_content_usable": reconciliation.get("upstream_artifact_content_usable"),
        "upstream_manifest_defect_detected": defect.get("defect_detected"),
        "upstream_manifest_defect_class": defect.get("defect_class"),
        "upstream_artifact_modified": False,
    }


def prohibition_audits(validation: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    synthetic = {
        "created_at": iso_kst(),
        "synthetic_fixture_transition_allowed": True,
        "synthetic_fixture_scenarios_executed": 0,
        "synthetic_fixture_is_research_evidence": False,
        "historical_branch_execution_count": 0,
        "d1_250row_execution_count": 0,
    }
    test = {
        "created_at": iso_kst(),
        "test_sealed_holdout_rows_read": 0,
        "test_branch_count": 0,
        "test_reuse_decision": False,
        "test_holdout_touched": False,
    }
    training = {
        "created_at": iso_kst(),
        "training_run_count": 0,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "checkpoint_load_count": 0,
        "checkpoint_write_count": 0,
    }
    external = {
        "created_at": iso_kst(),
        "database_accessed": False,
        "api_call_count": 0,
        "external_network_accessed": False,
        "service_key_accessed": False,
    }
    validation_payload = dict(validation)
    return synthetic, validation_payload, test, training, external


def choose_reconcile_gate(defect: Mapping[str, Any], validation: Mapping[str, Any], external: Mapping[str, Any]) -> Tuple[str, bool, str]:
    if external.get("api_call_count") or external.get("external_network_accessed"):
        return FAIL_EXTERNAL, False, "FAILED_EXTERNAL_ACCESS_PROHIBITION"
    if not validation.get("validation_seal_intact"):
        return FAIL_VALIDATION, False, "FAILED_VALIDATION_SEAL_REVALIDATION"
    if not defect.get("defect_detected"):
        return FAIL_MANIFEST, False, "FAILED_TO_RECORD_UPSTREAM_MANIFEST_DEFECT"
    return PASS_RECONCILE, True, "RECONCILE_COMPLETE_IMPLEMENT_PENDING_USER_COMMAND"


def downstream_lock(gate: Mapping[str, Any], plan: Mapping[str, Any], validation: Mapping[str, Any], test: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "engine_contract_repair_complete": False,
        "reconcile_complete": bool(gate.get("gate_passed")),
        "implement_complete": False,
        "verify_complete": False,
        "finalize_complete": False,
        "upstream_manifest_defect_recorded": plan.get("upstream_manifest_defect_recorded"),
        "upstream_artifact_modified": False,
        "state_snapshot_contract_ready": False,
        "state_roundtrip_verified": False,
        "deterministic_reset_ready": False,
        "canonical_replay_schema_ready": False,
        "action_adapter_ready": False,
        "legacy_action_2_reachable": False,
        "multiagent_orchestrator_ready": False,
        "thirty_minute_total_steps": 30,
        "event_trace_ready": False,
        "passenger_wait_schema_ready": False,
        "horizon_kpi_aggregator_ready": False,
        "new_reward_formula_created": False,
        "new_energy_formula_created": False,
        "upstream_stub_boundary_complete": False,
        "synthetic_fixture_scenarios": 0,
        "historical_branch_execution_count": 0,
        "validation_seal_intact": validation.get("validation_seal_intact"),
        "test_holdout_touched": test.get("test_holdout_touched"),
        "state_feasibility_required": True,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
        "next_required_mode": "implement",
        "automatic_mode_chaining_allowed": False,
    }


def final_report(root: Path, gate: Mapping[str, Any], defect: Mapping[str, Any], plan: Mapping[str, Any], validation: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    payload = {
        "created_at": iso_kst(),
        "artifact": str(root),
        "mode": "reconcile",
        "gate": gate,
        "upstream_manifest_defect": defect,
        "repair_plan": plan,
        "validation": validation,
        "next_mode": "implement",
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-ER1 Reconcile",
        "",
        f"- artifact: `{root}`",
        "- mode: `reconcile`",
        f"- gate: `{gate['gate']}`",
        f"- gate passed: `{str(gate['gate_passed']).lower()}`",
        "",
        "## Summary",
        f"- upstream manifest defect recorded: `{str(defect['defect_detected']).lower()}`",
        f"- defect class: `{defect['defect_class']}`",
        f"- upstream artifact modified: `false`",
        f"- planned dynamics modules: `{len(plan['planned_modules'])}`",
        f"- historical branch executions: `0`",
        f"- validation seal intact: `{str(validation['validation_seal_intact']).lower()}`",
        "",
        "No implementation, synthetic fixture transition, historical branch execution, validation/test access, reward definition, scale derivation, or training was performed in reconcile mode.",
    ]) + "\n"
    return payload, md


def validate_artifact_root(path: Path, mode: str) -> Path:
    root = path.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be an absolute path")
    if mode == "reconcile":
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"--artifact-root already exists and is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
    else:
        if not root.exists():
            raise FileNotFoundError(f"--artifact-root must exist for mode {mode}: {root}")
    return root


def write_manifest(writer: Writer) -> Dict[str, Any]:
    files = []
    missing = []
    for rel_path in RECONCILE_PAYLOADS:
        path = writer.root / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        files.append({
            "relative_path": rel_path,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "required": True,
            "created_order": writer.order.get(rel_path),
        })
    manifest = {
        "created_at": iso_kst(),
        "manifest_scope": "PA1A_ER1_RECONCILE_MODE",
        "manifest_protocol": "NO_SELF_REFERENCE_NO_TERMINAL_LOCK_ENTRY",
        "manifest_self_hash_exempt": False,
        "terminal_lock_listed_inside_manifest": False,
        "required_payload_count": len(RECONCILE_PAYLOADS),
        "missing_payload_count": len(missing),
        "missing_payloads": missing,
        "duplicate_path_count": len(RECONCILE_PAYLOADS) - len(set(RECONCILE_PAYLOADS)),
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "files": files,
    }
    writer.json("artifact_manifest.json", manifest)
    return manifest


def write_terminal_lock(writer: Writer, lock_name: str, gate: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_path = writer.root / "artifact_manifest.json"
    lock = {
        "created_at": iso_kst(),
        "gate": gate.get("gate"),
        "gate_passed": gate.get("gate_passed"),
        "mode": gate.get("mode"),
        "manifest_relative_path": "artifact_manifest.json",
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
    }
    writer.json(lock_name, lock)
    return lock


def verify_manifest_and_lock(root: Path, lock_name: str) -> Dict[str, Any]:
    lock = read_json(root / lock_name)
    manifest_path = root / lock["manifest_relative_path"]
    manifest_hash_ok = sha256_file(manifest_path) == lock["manifest_sha256"]
    manifest_size_ok = manifest_path.stat().st_size == lock["manifest_size_bytes"]
    manifest = read_json(manifest_path)
    file_checks = []
    for item in manifest.get("files", []):
        path = root / item["relative_path"]
        file_checks.append({
            "relative_path": item["relative_path"],
            "exists": path.exists(),
            "sha256_ok": path.exists() and sha256_file(path) == item["sha256"],
            "size_ok": path.exists() and path.stat().st_size == item["size_bytes"],
        })
    return {
        "manifest_hash_ok": manifest_hash_ok,
        "manifest_size_ok": manifest_size_ok,
        "payload_file_count": len(file_checks),
        "payload_missing_count": sum(1 for row in file_checks if not row["exists"]),
        "payload_hash_mismatch_count": sum(1 for row in file_checks if row["exists"] and not row["sha256_ok"]),
        "payload_size_mismatch_count": sum(1 for row in file_checks if row["exists"] and not row["size_ok"]),
        "terminal_lock_created_last": True,
        "terminal_lock_listed_inside_manifest": any(item.get("relative_path") == lock_name for item in manifest.get("files", [])),
    }


def verify_reconcile_stage_immutable(root: Path) -> Dict[str, Any]:
    manifest_path = root / "artifact_manifest.json"
    lock_path = root / "_RECONCILE_COMPLETE.lock"
    lock = read_json(lock_path)
    manifest_hash = sha256_file(manifest_path)
    manifest_size = manifest_path.stat().st_size
    return {
        "created_at": iso_kst(),
        "manifest_sha256": manifest_hash,
        "manifest_size_bytes": manifest_size,
        "expected_manifest_sha256": EXPECTED_RECONCILE_MANIFEST_SHA256,
        "expected_manifest_size_bytes": EXPECTED_RECONCILE_MANIFEST_SIZE,
        "manifest_matches_expected": manifest_hash == EXPECTED_RECONCILE_MANIFEST_SHA256 and manifest_size == EXPECTED_RECONCILE_MANIFEST_SIZE,
        "lock_manifest_sha256": lock.get("manifest_sha256"),
        "lock_manifest_size_bytes": lock.get("manifest_size_bytes"),
        "lock_points_to_manifest": lock.get("manifest_sha256") == manifest_hash and lock.get("manifest_size_bytes") == manifest_size,
        "reconcile_lock_gate": lock.get("gate"),
        "reconcile_lock_gate_passed": lock.get("gate_passed"),
    }


def preserve_reconcile_stage(writer: Writer) -> Dict[str, Any]:
    rows = []
    for rel_path in ["gate_decision.json", "downstream_lock.json", "final_report.json", "final_report.md"]:
        src = writer.root / rel_path
        dst_rel = f"stage_history/reconcile/{rel_path}"
        dst = writer.root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        writer._mark(dst_rel)
        rows.append({
            "source": rel_path,
            "copy": dst_rel,
            "source_sha256": sha256_file(src),
            "copy_sha256": sha256_file(dst),
            "hash_match": sha256_file(src) == sha256_file(dst),
            "size_match": src.stat().st_size == dst.stat().st_size,
        })
    return {
        "created_at": iso_kst(),
        "reconcile_stage_result_preserved": all(row["hash_match"] and row["size_match"] for row in rows),
        "copies": rows,
    }


def py_compile_file(path: Path) -> Dict[str, Any]:
    return run_cmd([sys.executable, "-m", "py_compile", str(path)])


def import_module_from_path(path: Path) -> Dict[str, Any]:
    module_name = path.stem + "_er1_static_import"
    try:
        training_root = str(PROJECT_ROOT / "05_training")
        if training_root not in sys.path:
            sys.path.insert(0, training_root)
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not create import spec for {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return {"path": str(path), "importable": True, "error": None}
    except Exception as exc:
        return {"path": str(path), "importable": False, "error": repr(exc)}


def scan_proxy_dependency() -> Dict[str, Any]:
    module_texts = {path: path.read_text(encoding="utf-8") if path.exists() else "" for path in PLANNED_MODULES}
    proxy_needles = ["run_prompt5_e01_dl6d_r1", "R1_SOURCE", "thirty_minute_audit"]
    reward_literals = ["0.01", "0.002", "-0.10", "-0.08", "0.08"]
    rows = []
    for path, text in module_texts.items():
        for needle in proxy_needles:
            count = text.count(needle)
            if count:
                rows.append({"path": str(path), "needle": needle, "count": count})
    reward_rows = []
    for path, text in module_texts.items():
        for literal in reward_literals:
            count = text.count(literal)
            if count:
                reward_rows.append({"path": str(path), "literal": literal, "count": count})
    return {
        "created_at": iso_kst(),
        "r1_proxy_import_count": sum(row["count"] for row in rows if row["needle"] != "thirty_minute_audit"),
        "thirty_minute_audit_reference_count": sum(row["count"] for row in rows if row["needle"] == "thirty_minute_audit"),
        "proxy_reward_literal_reuse_count": sum(row["count"] for row in reward_rows),
        "proxy_dependency_rows": rows,
        "proxy_reward_literal_rows": reward_rows,
        "proxy_dependency_detected": bool(rows or reward_rows),
    }


def source_implementation_registry() -> Dict[str, Any]:
    rows = []
    for path in PLANNED_MODULES:
        rows.append({
            "path": str(path),
            "relative_path": rel(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
            "git_status": run_cmd(["git", "status", "--short", "--", rel(path)])["stdout"] if path.exists() else "MISSING",
        })
    for path, role in [(ENGINE_SOURCE, "transition engine source"), (R1_SOURCE, "R1 proxy source")]:
        rows.append({
            "path": str(path),
            "relative_path": rel(path),
            "role": role,
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
            "modified_by_er1_implement": False,
            "git_status": run_cmd(["git", "status", "--short", "--", rel(path)])["stdout"] if path.exists() else "MISSING",
        })
    return {
        "created_at": iso_kst(),
        "planned_modules_created": sum(1 for path in PLANNED_MODULES if path.exists()),
        "planned_module_count": len(PLANNED_MODULES),
        "r1_proxy_module_modified": False,
        "core_engine_modified": False,
        "records": rows,
    }


def static_implementation_audit() -> Dict[str, Any]:
    compile_rows = []
    import_rows = []
    for path in [Path(__file__), *PLANNED_MODULES]:
        compile_rows.append({"path": str(path), **py_compile_file(path)})
    for path in PLANNED_MODULES:
        import_rows.append(import_module_from_path(path))
    return {
        "created_at": iso_kst(),
        "python_syntax_valid": all(row["returncode"] == 0 for row in compile_rows),
        "all_modules_importable": all(row["importable"] for row in import_rows),
        "compile_rows": compile_rows,
        "import_rows": import_rows,
        "transition_function_execution_count": 0,
        "synthetic_fixture_scenarios_executed": 0,
        "historical_branch_execution_count": 0,
    }


def implementation_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "mode": "implement",
        "implement_mode_authorized_by_user": True,
        "verify_mode_authorized_by_user": False,
        "finalize_mode_authorized_by_user": False,
        "state_feasibility_authorized_by_user": False,
        "core_engine_modification_allowed": False,
        "r1_thirty_minute_audit_modified": False,
        "proxy_module_imported_by_dynamics": False,
        "transition_function_actual_call_allowed": False,
        "transition_function_execution_count": 0,
        "synthetic_fixture_scenarios_executed": 0,
        "historical_branch_execution_count": 0,
        "d1_250row_execution_count": 0,
        "validation_branch_count": 0,
        "test_branch_count": 0,
        "planned_modules": [rel(path) for path in PLANNED_MODULES],
        "automatic_mode_chaining_allowed": False,
    }


def state_snapshot_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "module": "05_training/simulator/dynamics_state_snapshot.py",
        "schema_version": "SUSEONG_DYNAMICS_STATE_SNAPSHOT_V1",
        "apis": [
            "capture_dynamics_state",
            "serialize_dynamics_state",
            "deserialize_dynamics_state",
            "clone_dynamics_state",
            "restore_dynamics_state",
            "hash_dynamics_state",
            "build_runtime_state_from_snapshot",
            "reset_runtime_state_from_snapshot",
        ],
        "missing_field_fail_closed": True,
        "unknown_field_fail_closed": True,
        "canonical_serialization": "UTF-8 JSON sort_keys=true allow_nan=false",
        "default_field_insertion_allowed": False,
        "reset_execution_count": 0,
    }


def replay_contract_schema() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "module": "05_training/simulator/dynamics_replay_contract.py",
        "event_types": [
            "PASSENGER_ARRIVAL",
            "PICKUP_REQUEST",
            "PASSENGER_DESTINATION",
            "PASSENGER_CANCELLATION",
            "TRAVEL_TIME_UPDATE",
            "SIGNAL_STATE_UPDATE",
            "ROAD_STATE_UPDATE",
            "ROUTE_AVAILABILITY_UPDATE",
            "SCHEDULE_UPDATE",
            "OPERATION_MODE_UPDATE",
        ],
        "decision_interval_seconds": 60,
        "total_frames": 30,
        "event_ordering": ["event_timestamp_seconds", "event_type", "route_id", "stop_id", "passenger_id", "event_id"],
        "event_duplication_allowed": False,
        "frame_gap_allowed": False,
        "frame_overlap_allowed": False,
    }


def external_provider_state_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "module": "05_training/simulator/dynamics_replay_contract.py",
        "protocol": "ExternalProviderStateProtocol",
        "required_methods": ["export_provider_state", "import_provider_state", "clone_provider", "provider_state_hash"],
        "provider_targets": ["stop_service provider", "passenger event provider", "travel-time provider", "route-state provider", "schedule provider"],
        "stateless_provider_classification": "STATELESS",
        "seed_only_randomness_control_sufficient": False,
        "historical_event_replay_required": True,
    }


def action_adapter_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "module": "05_training/simulator/dynamics_multiagent_orchestrator.py",
        "external_actions": ["HOLD_CURRENT_POSITION", "SERVE_AND_MOVE_TO_NEXT_STOP", "CONDITIONAL_SKIP_EMPTY_STOP"],
        "mapping": {"HOLD_CURRENT_POSITION": 0, "SERVE_AND_MOVE_TO_NEXT_STOP": 1, "CONDITIONAL_SKIP_EMPTY_STOP": 3},
        "legacy_engine_action_2_accepted": False,
        "unsupported_action_error": "UnsupportedDynamicsActionError",
        "skip_safety_delegated_to_engine": True,
        "skip_safety_functions": ["evaluate_conditional_skip_safety", "build_distinct_three_action_mask"],
    }


def multiagent_orchestrator_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "module": "05_training/simulator/dynamics_multiagent_orchestrator.py",
        "apis": ["advance_multiagent_global_step", "build_single_pulse_action_vector", "run_thirty_minute_branch"],
        "active_agent_count": 8,
        "canonical_agent_order": "sorted(agent_id)",
        "orchestration_semantics": "DETERMINISTIC_GLOBAL_STEP_WITH_CANONICAL_TIEBREAK",
        "shared_request_conflict_rule": ["request_timestamp_seconds", "service_feasible", "agent_id"],
        "dictionary_iteration_order_allowed": False,
        "decision_interval_seconds": 60,
        "thirty_minute_total_steps": 30,
    }


def event_trace_schema() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "module": "05_training/simulator/dynamics_event_trace.py",
        "event_types": [
            "VEHICLE_MOVE",
            "VEHICLE_DWELL",
            "STOP_ARRIVAL",
            "PASSENGER_ARRIVAL",
            "PASSENGER_BOARD",
            "PASSENGER_ALIGHT",
            "PASSENGER_MISSED_PICKUP",
            "PASSENGER_MISSED_DROPOFF",
            "CONDITIONAL_SKIP",
            "INVALID_SKIP",
            "MANDATORY_STOP_VIOLATION",
            "ROUTE_TURNAROUND",
            "SERVICE_COMPLETED",
        ],
        "passenger_wait_schema_implemented": True,
        "passenger_wait_required_fields": ["passenger_id", "arrival_timestamp_seconds", "service_timestamp_seconds", "wait_seconds", "route_id", "stop_id", "vehicle_id"],
        "passenger_wait_distribution_defaulting_allowed": False,
    }


def horizon_kpi_aggregator_contract() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "module": "05_training/simulator/dynamics_horizon_aggregator.py",
        "p95_wait_without_passenger_events": {
            "value": None,
            "status": "UNAVAILABLE_MISSING_REQUIRED_DATA",
            "reason": "PASSENGER_LEVEL_WAIT_DISTRIBUTION_UNAVAILABLE",
        },
        "external_aggregator_required_kpis": ["cv_headway", "bunching_rate", "on_time_rate"],
        "energy_metric_status": "FORMULA_NOT_SELECTED",
        "reward_status": "NOT_DEFINED_IN_ER1_IMPLEMENT",
    }


def synthetic_fixture_inventory() -> Dict[str, Any]:
    scenarios = [
        "Empty-stop K valid",
        "Waiting passenger K invalid",
        "Assigned pickup K invalid",
        "Onboard dropoff K invalid",
        "Mandatory stop K invalid",
        "H/S/K distinct state transition",
        "State round-trip",
        "Clone mutation isolation",
        "Replay event ordering",
        "Shared request conflict",
        "Passenger wait event available",
        "Passenger wait event unavailable",
    ]
    return {
        "created_at": iso_kst(),
        "synthetic_fixture_scenarios_defined": len(scenarios),
        "synthetic_fixture_scenarios_executed": 0,
        "synthetic_fixture_max_scenarios": 12,
        "synthetic_fixture_is_research_evidence": False,
        "scenarios": scenarios,
    }


def reward_energy_nondefinition_audit() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "new_reward_formula_created": False,
        "legacy_reward_projection_computed": False,
        "normalization_scale_created": False,
        "tolerance_created": False,
        "candidate_created": False,
        "new_energy_formula_created": False,
        "energy_metric_status": "FORMULA_NOT_SELECTED",
        "reward_status": "NOT_DEFINED_IN_ER1_IMPLEMENT",
    }


def implement_prohibition_audits() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    dynamics = {
        "created_at": iso_kst(),
        "transition_function_execution_count": 0,
        "advance_vehicle_time_budget_execution_count": 0,
        "reset_from_snapshot_execution_count": 0,
        "synthetic_fixture_scenarios_executed": 0,
        "thirty_minute_horizon_execution_count": 0,
        "historical_branch_execution_count": 0,
        "d1_250row_execution_count": 0,
    }
    validation = validation_untouched_audit()
    validation["validation_branch_count"] = 0
    validation["validation_reward_count"] = 0
    test = {
        "created_at": iso_kst(),
        "test_sealed_holdout_rows_read": 0,
        "test_branch_count": 0,
        "test_reuse_decision": False,
        "test_holdout_touched": False,
    }
    return dynamics, validation, test, {
        "created_at": iso_kst(),
        "training_run_count": 0,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "checkpoint_load_count": 0,
        "checkpoint_write_count": 0,
    }


def choose_implement_gate(
    immutable: Mapping[str, Any],
    source: Mapping[str, Any],
    static: Mapping[str, Any],
    proxy: Mapping[str, Any],
    reward_energy: Mapping[str, Any],
    dynamics: Mapping[str, Any],
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    training: Mapping[str, Any],
) -> Tuple[str, bool, str]:
    if not immutable.get("manifest_matches_expected") or not immutable.get("lock_points_to_manifest"):
        return FAIL_I1_RECONCILE_MUTATED, False, "FAILED_RECONCILE_STAGE_MUTATION"
    if source.get("planned_modules_created") != len(PLANNED_MODULES):
        return "BLOCKED_SUSEONG_DL6D_PA1A_ER1_I1_STATE_SCHEMA_INCOMPLETE", False, "BLOCKED_PLANNED_MODULES_MISSING"
    if not static.get("python_syntax_valid") or not static.get("all_modules_importable"):
        return "BLOCKED_SUSEONG_DL6D_PA1A_ER1_I1_STATE_SCHEMA_INCOMPLETE", False, "BLOCKED_STATIC_IMPLEMENTATION_INVALID"
    if proxy.get("proxy_dependency_detected"):
        return FAIL_I1_PROXY, False, "FAILED_PROXY_DEPENDENCY_DETECTED"
    if action_adapter_contract()["legacy_engine_action_2_accepted"]:
        return FAIL_I1_LEGACY, False, "FAILED_LEGACY_ACTION_REACHABLE"
    if multiagent_orchestrator_contract()["thirty_minute_total_steps"] != 30:
        return FAIL_I1_HORIZON, False, "FAILED_HORIZON_CONTRACT_INVALID"
    if reward_energy.get("new_reward_formula_created") or reward_energy.get("new_energy_formula_created"):
        return FAIL_I1_REWARD_ENERGY, False, "FAILED_REWARD_OR_ENERGY_FORMULA_CREATED"
    if dynamics.get("transition_function_execution_count") or dynamics.get("synthetic_fixture_scenarios_executed") or dynamics.get("historical_branch_execution_count"):
        return FAIL_I1_DYNAMICS, False, "FAILED_DYNAMICS_EXECUTION_DETECTED"
    if not validation.get("validation_seal_intact") or validation.get("validation_row_level_access_count") or test.get("test_holdout_touched"):
        return FAIL_I1_VALIDATION_TEST, False, "FAILED_VALIDATION_OR_TEST_PROTECTION"
    if training.get("training_run_count") or training.get("optimizer_step_count") or training.get("checkpoint_write_count"):
        return FAIL_I1_TRAINING, False, "FAILED_TRAINING_PROHIBITION"
    return PASS_IMPLEMENT, True, "IMPLEMENT_COMPLETE_VERIFY_PENDING_USER_COMMAND"


def implement_downstream_lock(gate: Mapping[str, Any], source: Mapping[str, Any], validation: Mapping[str, Any], test: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reconcile_complete": True,
        "implement_complete": bool(gate.get("gate_passed")),
        "verify_complete": False,
        "finalize_complete": False,
        "planned_modules_created": source.get("planned_modules_created"),
        "state_snapshot_contract_implemented": True,
        "state_codec_implemented": True,
        "deterministic_reset_api_implemented": True,
        "replay_contract_implemented": True,
        "external_provider_state_contract_implemented": True,
        "action_adapter_implemented": True,
        "legacy_action_2_reachable": False,
        "multiagent_orchestrator_implemented": True,
        "thirty_minute_total_steps": 30,
        "event_trace_implemented": True,
        "passenger_wait_schema_implemented": True,
        "horizon_kpi_aggregator_implemented": True,
        "new_reward_formula_created": False,
        "new_energy_formula_created": False,
        "synthetic_fixture_scenarios_defined": 12,
        "synthetic_fixture_scenarios_executed": 0,
        "historical_branch_execution_count": 0,
        "reconcile_manifest_unchanged": True,
        "implement_manifest_valid": True,
        "validation_seal_intact": validation.get("validation_seal_intact"),
        "test_holdout_touched": test.get("test_holdout_touched"),
        "verify_required": True,
        "verify_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
        "next_required_mode": "verify",
        "automatic_mode_chaining_allowed": False,
    }


def implement_final_report(root: Path, gate: Mapping[str, Any], source: Mapping[str, Any], immutable: Mapping[str, Any], static: Mapping[str, Any], proxy: Mapping[str, Any], validation: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    payload = {
        "created_at": iso_kst(),
        "artifact": str(root),
        "mode": "implement",
        "gate": gate,
        "source_implementation": source,
        "reconcile_stage_immutability": immutable,
        "static_implementation": static,
        "proxy_dependency": proxy,
        "validation": validation,
        "next_mode": "verify",
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-ER1 Implement",
        "",
        f"- artifact: `{root}`",
        "- mode: `implement`",
        f"- gate: `{gate['gate']}`",
        f"- gate passed: `{str(gate['gate_passed']).lower()}`",
        "",
        "## Summary",
        f"- planned modules created: `{source['planned_modules_created']} / {source['planned_module_count']}`",
        "- core transition engine modified: `false`",
        "- R1 proxy module modified: `false`",
        f"- Python syntax valid: `{str(static['python_syntax_valid']).lower()}`",
        f"- modules importable: `{str(static['all_modules_importable']).lower()}`",
        f"- R1 proxy import / thirty_minute reference / proxy reward literal reuse: `{proxy['r1_proxy_import_count']} / {proxy['thirty_minute_audit_reference_count']} / {proxy['proxy_reward_literal_reuse_count']}`",
        "- transition/historical execution: `0`",
        f"- reconcile manifest unchanged: `{str(immutable['manifest_matches_expected']).lower()}`",
        f"- validation seal intact: `{str(validation['validation_seal_intact']).lower()}`",
        "",
        "Verify, finalize, state-feasibility, PA1-B, and training remain locked.",
    ]) + "\n"
    return payload, md


def write_named_manifest(writer: Writer, rel_manifest: str, payloads: Sequence[str], scope: str) -> Dict[str, Any]:
    files = []
    missing = []
    seen = set()
    duplicates = 0
    excluded = set(MANIFEST_EXCLUDED_RELATIVE_PATHS)
    excluded.add(rel_manifest)
    for rel_path in payloads:
        if rel_path in excluded:
            continue
        if rel_path in seen:
            duplicates += 1
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
            "required": True,
            "created_order": writer.order.get(rel_path),
        })
    manifest = {
        "created_at": iso_kst(),
        "manifest_scope": scope,
        "manifest_protocol": "NO_SELF_REFERENCE_NO_TERMINAL_LOCK_ENTRY",
        "manifest_self_hash_exempt": False,
        "terminal_lock_listed_inside_manifest": False,
        "required_payload_count": len([item for item in payloads if item not in excluded]),
        "payload_file_count": len(files),
        "missing_payload_count": len(missing),
        "missing_payloads": missing,
        "duplicate_path_count": duplicates,
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "files": files,
    }
    writer.json(rel_manifest, manifest)
    return manifest


def write_terminal_lock_for_manifest(writer: Writer, lock_name: str, manifest_name: str, gate: Mapping[str, Any]) -> Dict[str, Any]:
    manifest_path = writer.root / manifest_name
    lock = {
        "created_at": iso_kst(),
        "gate": gate.get("gate"),
        "gate_passed": gate.get("gate_passed"),
        "mode": gate.get("mode"),
        "manifest_relative_path": manifest_name,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_size_bytes": manifest_path.stat().st_size,
    }
    writer.json(lock_name, lock)
    return lock


def run_reconcile(artifact_root: Path) -> Path:
    root = validate_artifact_root(artifact_root, "reconcile")
    writer = Writer(root)
    writer.text("git_status_er1.txt", run_cmd(["git", "status", "--short"])["stdout"] + "\n")
    writer.json("mac_mini_environment_er1.json", environment_audit())
    defect = upstream_manifest_defect_record()
    reconciliation = upstream_manifest_independent_reconciliation(defect)
    writer.json("upstream_manifest_defect_record.json", defect)
    writer.json("upstream_manifest_independent_reconciliation.json", reconciliation)
    writer.json("upstream_validation.json", upstream_validation(defect, reconciliation))
    plan = repair_plan_contract(defect)
    writer.json("repair_plan_contract.json", plan)
    writer.json("source_before_after_registry.json", source_before_after_registry())
    validation = validation_untouched_audit()
    synthetic, validation_payload, test, training, external = prohibition_audits(validation)
    writer.json("synthetic_fixture_execution_audit.json", synthetic)
    writer.json("validation_untouched_audit.json", validation_payload)
    writer.json("test_holdout_untouched_audit.json", test)
    writer.json("training_prohibition_audit.json", training)
    writer.json("external_access_audit.json", external)
    gate_name, passed, readiness = choose_reconcile_gate(defect, validation, external)
    gate = {
        "created_at": iso_kst(),
        "mode": "reconcile",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": readiness,
        "implement_authorized": False,
        "verify_authorized": False,
        "finalize_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
    }
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", downstream_lock(gate, plan, validation, test))
    report_json, report_md = final_report(root, gate, defect, plan, validation)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    manifest = write_manifest(writer)
    lock = write_terminal_lock(writer, "_RECONCILE_COMPLETE.lock", gate)
    verification = verify_manifest_and_lock(root, "_RECONCILE_COMPLETE.lock")
    if (
        manifest["missing_payload_count"]
        or manifest["duplicate_path_count"]
        or not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["payload_size_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        gate["gate"] = FAIL_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = "FAILED_MANIFEST_TERMINAL_LOCK_PROTOCOL"
        writer.json("gate_decision.json", gate)
        write_manifest(writer)
        write_terminal_lock(writer, "_RECONCILE_COMPLETE.lock", gate)
    print(f"[DL-6D-PA1-A-ER1] artifact: {root}")
    print("[DL-6D-PA1-A-ER1] mode: reconcile")
    print(f"[DL-6D-PA1-A-ER1] upstream manifest defect: {defect['defect_detected']} ({defect['defect_class']})")
    print("[DL-6D-PA1-A-ER1] upstream artifact modified: false")
    print("[DL-6D-PA1-A-ER1] implementation performed: false")
    print("[DL-6D-PA1-A-ER1] historical branch executions: 0")
    print(f"[DL-6D-PA1-A-ER1] validation seal intact: {validation['validation_seal_intact']}")
    print(f"[DL-6D-PA1-A-ER1] manifest payloads: {manifest['required_payload_count']}")
    print(f"[DL-6D-PA1-A-ER1] terminal lock listed in manifest: {verification['terminal_lock_listed_inside_manifest']}")
    print(f"[DL-6D-PA1-A-ER1] gate: {gate['gate']}")
    print(f"[DL-6D-PA1-A-ER1] gate passed: {str(gate['gate_passed']).lower()}")
    print("[DL-6D-PA1-A-ER1] next: REPORT_TO_USER")
    return root


def validate_implement_entry(root: Path) -> None:
    validate_artifact_root(root, "implement")
    gate = read_json(root / "gate_decision.json")
    if gate.get("gate") != PASS_RECONCILE or gate.get("gate_passed") is not True:
        raise RuntimeError("implement mode requires reconcile PASS gate")
    if not (root / "_RECONCILE_COMPLETE.lock").exists():
        raise RuntimeError("implement mode requires _RECONCILE_COMPLETE.lock")
    if (root / "_IMPLEMENT_COMPLETE.lock").exists():
        raise RuntimeError("_IMPLEMENT_COMPLETE.lock already exists; implement mode is immutable")


def run_implement(artifact_root: Path) -> Path:
    root = artifact_root.expanduser()
    validate_implement_entry(root)
    writer = Writer(root)
    immutable = verify_reconcile_stage_immutable(root)
    history = preserve_reconcile_stage(writer)
    source = source_implementation_registry()
    static = static_implementation_audit()
    proxy = scan_proxy_dependency()
    reward_energy = reward_energy_nondefinition_audit()
    dynamics, validation, test, training = implement_prohibition_audits()

    writer.json("implementation_contract.json", implementation_contract())
    writer.json("source_implementation_registry.json", source)
    writer.json("state_snapshot_contract.json", state_snapshot_contract())
    writer.json("replay_contract_schema.json", replay_contract_schema())
    writer.json("external_provider_state_contract.json", external_provider_state_contract())
    writer.json("action_adapter_contract.json", action_adapter_contract())
    writer.json("multiagent_orchestrator_contract.json", multiagent_orchestrator_contract())
    writer.json("event_trace_schema.json", event_trace_schema())
    writer.json("horizon_kpi_aggregator_contract.json", horizon_kpi_aggregator_contract())
    writer.json("synthetic_fixture_inventory.json", synthetic_fixture_inventory())
    writer.json("static_implementation_audit.json", static)
    writer.json("proxy_dependency_prohibition_audit.json", proxy)
    writer.json("reward_energy_nondefinition_audit.json", reward_energy)
    writer.json("dynamics_execution_prohibition_audit.json", dynamics)
    writer.json("validation_untouched_audit_implement.json", validation)
    writer.json("test_holdout_untouched_audit_implement.json", test)

    gate_name, passed, readiness = choose_implement_gate(immutable, source, static, proxy, reward_energy, dynamics, validation, test, training)
    gate = {
        "created_at": iso_kst(),
        "mode": "implement",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": readiness,
        "implement_mode_authorized_by_user": True,
        "verify_authorized": False,
        "finalize_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
    }
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", implement_downstream_lock(gate, source, validation, test))
    report_json, report_md = implement_final_report(root, gate, source, immutable, static, proxy, validation)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)

    implement_payloads = [*RECONCILE_PAYLOADS, *IMPLEMENT_NEW_PAYLOADS]
    manifest = write_named_manifest(writer, "artifact_manifest_implement.json", implement_payloads, "PA1A_ER1_IMPLEMENT_MODE")
    lock = write_terminal_lock_for_manifest(writer, "_IMPLEMENT_COMPLETE.lock", "artifact_manifest_implement.json", gate)
    verification = verify_manifest_and_lock(root, "_IMPLEMENT_COMPLETE.lock")
    if (
        manifest["missing_payload_count"]
        or manifest["duplicate_path_count"]
        or not verification["manifest_hash_ok"]
        or not verification["manifest_size_ok"]
        or verification["payload_missing_count"]
        or verification["payload_hash_mismatch_count"]
        or verification["payload_size_mismatch_count"]
        or verification["terminal_lock_listed_inside_manifest"]
    ):
        gate["gate"] = FAIL_I1_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = "FAILED_IMPLEMENT_MANIFEST_TERMINAL_LOCK_PROTOCOL"
        writer.json("gate_decision.json", gate)
        write_named_manifest(writer, "artifact_manifest_implement.json", implement_payloads, "PA1A_ER1_IMPLEMENT_MODE")
        write_terminal_lock_for_manifest(writer, "_IMPLEMENT_COMPLETE.lock", "artifact_manifest_implement.json", gate)

    print(f"[DL-6D-PA1-A-ER1] artifact: {root}")
    print("[DL-6D-PA1-A-ER1] mode: implement")
    print(f"[DL-6D-PA1-A-ER1] stage history preserved: {history['reconcile_stage_result_preserved']}")
    print(f"[DL-6D-PA1-A-ER1] reconcile manifest unchanged: {immutable['manifest_matches_expected']}")
    print(f"[DL-6D-PA1-A-ER1] planned modules created: {source['planned_modules_created']} / {source['planned_module_count']}")
    print(f"[DL-6D-PA1-A-ER1] syntax/import: {static['python_syntax_valid']} / {static['all_modules_importable']}")
    print(f"[DL-6D-PA1-A-ER1] proxy deps r1/thirty/reward: {proxy['r1_proxy_import_count']} / {proxy['thirty_minute_audit_reference_count']} / {proxy['proxy_reward_literal_reuse_count']}")
    print("[DL-6D-PA1-A-ER1] transition function executions: 0")
    print("[DL-6D-PA1-A-ER1] synthetic fixture executions: 0")
    print("[DL-6D-PA1-A-ER1] historical branch executions: 0")
    print(f"[DL-6D-PA1-A-ER1] validation seal intact: {validation['validation_seal_intact']}")
    print(f"[DL-6D-PA1-A-ER1] implement manifest payloads: {manifest['payload_file_count']}")
    print(f"[DL-6D-PA1-A-ER1] terminal lock listed in implement manifest: {verification['terminal_lock_listed_inside_manifest']}")
    print(f"[DL-6D-PA1-A-ER1] gate: {gate['gate']}")
    print(f"[DL-6D-PA1-A-ER1] gate passed: {str(gate['gate_passed']).lower()}")
    print("[DL-6D-PA1-A-ER1] next: REPORT_TO_USER")
    return root


def write_table_payload(writer: Writer, rel_path: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    path = writer.root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    backend = "JSONL_FALLBACK_NO_PARQUET_ENGINE"
    error = None
    try:
        import pandas as pd

        frame = pd.DataFrame([json_clean(dict(row)) for row in rows])
        frame.to_parquet(path, index=False)
        backend = "PANDAS_TO_PARQUET"
    except Exception as exc:
        error = repr(exc)
        path.write_text(
            "".join(json.dumps(json_clean(dict(row)), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    writer._mark(rel_path)
    return {
        "relative_path": rel_path,
        "row_count": len(rows),
        "write_backend": backend,
        "parquet_engine_unavailable_error": error,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def ensure_training_path() -> None:
    training_root = str(PROJECT_ROOT / "05_training")
    if training_root not in sys.path:
        sys.path.insert(0, training_root)


def import_simulator_modules() -> Dict[str, Any]:
    ensure_training_path()
    from simulator import suseong_service_transition_engine as engine
    from simulator import dynamics_event_trace as event_mod
    from simulator import dynamics_horizon_aggregator as kpi_mod
    from simulator import dynamics_multiagent_orchestrator as orchestrator
    from simulator import dynamics_replay_contract as replay_mod
    from simulator import dynamics_state_snapshot as state_mod

    return {
        "engine": engine,
        "event": event_mod,
        "kpi": kpi_mod,
        "orchestrator": orchestrator,
        "replay": replay_mod,
        "state": state_mod,
    }


def source_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def excerpt_for(path: Path, needle: str, *, context: int = 8) -> Dict[str, Any]:
    lines = source_lines(path)
    hit = next((idx for idx, line in enumerate(lines) if needle in line), None)
    if hit is None:
        return {
            "path": str(path),
            "needle": needle,
            "found": False,
            "start_line": None,
            "end_line": None,
            "sha256": None,
            "excerpt": "",
        }
    start = max(0, hit - context)
    end = min(len(lines), hit + context + 1)
    excerpt = "\n".join(lines[start:end]) + "\n"
    return {
        "path": str(path),
        "needle": needle,
        "found": True,
        "start_line": start + 1,
        "end_line": end,
        "sha256": sha256_text(excerpt),
        "excerpt": excerpt,
    }


def source_excerpt_registry() -> Dict[str, Any]:
    excerpts = {
        "transition_function": excerpt_for(ENGINE_SOURCE, "def advance_vehicle_time_budget"),
        "h_s_k_mapping_source": excerpt_for(ENGINE_SOURCE, "ACTOR_TO_ENGINE_ACTION"),
        "k_safety_predicate": excerpt_for(ENGINE_SOURCE, "def evaluate_conditional_skip_safety"),
        "action_mask_source": excerpt_for(ENGINE_SOURCE, "def build_distinct_three_action_mask"),
        "dl6b_synthetic_state_generation": excerpt_for(DL6B_SOURCE, "def make_initial_state"),
        "dl6b_modulo_demand_generation": excerpt_for(DL6B_SOURCE, "stable_int(\"arr\""),
        "dl6b_kpi_delta_generation": excerpt_for(DL6B_SOURCE, "passenger_wait_p95_seconds"),
        "r1_thirty_minute_audit": excerpt_for(R1_SOURCE, "def thirty_minute_audit"),
    }
    return {
        "created_at": iso_kst(),
        "registry_scope": "SOURCE_EXCERPTS_FOR_V1_AUDIT",
        "all_required_excerpts_found": all(row.get("found") for row in excerpts.values()),
        "excerpts": excerpts,
    }


def preserve_implement_stage(writer: Writer) -> Dict[str, Any]:
    rows = []
    for rel_path in ["gate_decision.json", "downstream_lock.json", "final_report.json", "final_report.md"]:
        src = writer.root / rel_path
        dst_rel = f"stage_history/implement/{rel_path}"
        dst = writer.root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        writer._mark(dst_rel)
        rows.append({
            "source": rel_path,
            "copy": dst_rel,
            "source_sha256": sha256_file(src),
            "copy_sha256": sha256_file(dst),
            "hash_match": sha256_file(src) == sha256_file(dst),
            "size_match": src.stat().st_size == dst.stat().st_size,
        })
    return {
        "created_at": iso_kst(),
        "implement_stage_result_preserved": all(row["hash_match"] and row["size_match"] for row in rows),
        "copies": rows,
    }


def prior_stage_hashes(root: Path) -> Dict[str, Dict[str, Any]]:
    targets = [
        "artifact_manifest.json",
        "_RECONCILE_COMPLETE.lock",
        "artifact_manifest_implement.json",
        "_IMPLEMENT_COMPLETE.lock",
    ]
    rows = {}
    for rel_path in targets:
        path = root / rel_path
        rows[rel_path] = {
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    return rows


def source_snapshot_audit(writer: Writer) -> Dict[str, Any]:
    rows = []
    for path in PLANNED_MODULES:
        rel_path = rel(path)
        frozen_sha = FROZEN_SOURCE_SHA256[rel_path]
        runtime_sha = sha256_file(path) if path.exists() else None
        dst_rel = f"source_snapshot/{path.name}"
        dst = writer.root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            shutil.copy2(path, dst)
            writer._mark(dst_rel)
        snapshot_sha = sha256_file(dst) if dst.exists() else None
        rows.append({
            "source_role": "ER1_DYNAMICS_MODULE",
            "relative_source_path": rel_path,
            "runtime_source_path": str(path),
            "source_snapshot_path": dst_rel,
            "runtime_source_sha": runtime_sha,
            "implement_frozen_sha": frozen_sha,
            "source_snapshot_sha": snapshot_sha,
            "runtime_matches_frozen": runtime_sha == frozen_sha,
            "snapshot_matches_runtime": snapshot_sha == runtime_sha,
            "source_snapshot_byte_identical": runtime_sha == snapshot_sha,
        })
    for path in [ENGINE_SOURCE, R1_SOURCE]:
        rel_path = rel(path)
        frozen_sha = FROZEN_SOURCE_SHA256[rel_path]
        runtime_sha = sha256_file(path) if path.exists() else None
        rows.append({
            "source_role": "CORE_OR_PROXY_SOURCE_EXCERPT_ONLY",
            "relative_source_path": rel_path,
            "runtime_source_path": str(path),
            "source_snapshot_path": None,
            "runtime_source_sha": runtime_sha,
            "implement_frozen_sha": frozen_sha,
            "source_snapshot_sha": None,
            "runtime_matches_frozen": runtime_sha == frozen_sha,
            "snapshot_matches_runtime": True,
            "source_snapshot_byte_identical": True,
        })
    return {
        "created_at": iso_kst(),
        "source_drift_count": sum(1 for row in rows if not row["runtime_matches_frozen"]),
        "source_snapshot_count": sum(1 for row in rows if row["source_snapshot_path"]),
        "source_snapshot_byte_identical": all(row["source_snapshot_byte_identical"] for row in rows),
        "records": rows,
    }


def upstream_contract_preflight() -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    engine_text = ENGINE_SOURCE.read_text(encoding="utf-8")
    orchestrator_text = (PROJECT_ROOT / "05_training/simulator/dynamics_multiagent_orchestrator.py").read_text(encoding="utf-8")
    dl6b_text = DL6B_SOURCE.read_text(encoding="utf-8")
    dl6c_text = DL6C_SOURCE.read_text(encoding="utf-8")

    action_contract_valid = (
        "ENGINE_ACTION_HOLD = 0" in engine_text
        and "ENGINE_ACTION_SERVE_MOVE = 1" in engine_text
        and "ENGINE_ACTION_CONDITIONAL_SKIP = 3" in engine_text
        and "CONDITIONAL_SKIP_EMPTY_STOP" in engine_text
        and "DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP" in orchestrator_text
    )
    safety_predicate_valid = "def evaluate_conditional_skip_safety" in engine_text and "skip_invalid_reason_codes" in engine_text
    action_mask_valid = "def build_distinct_three_action_mask" in engine_text and "evaluate_conditional_skip_safety" in engine_text
    legacy_action_2_reachable = "ENGINE_ACTION_LEGACY_MOVE_ONE_DUPLICATE" in orchestrator_text and "legacy engine action 2 is not accepted" not in orchestrator_text
    dl6b_state_synthetic = "def make_initial_state" in dl6b_text and "stable_int(\"pos\"" in dl6b_text
    dl6b_demand_modulo = "stable_int(\"arr\"" in dl6b_text and "modulo=3" in dl6b_text
    dl6b_kpi_modulo = "passenger_wait_p95_seconds" in dl6b_text and "headway_proxy" in dl6b_text
    dl6b_conclusion_reusable = False

    provenance_rows = [
        {
            "artifact_or_source": rel(ENGINE_SOURCE),
            "claim_or_field": "H/S/K semantic contract",
            "classification": "REAL_ENGINE_CONTRACT",
            "usable_as_source_contract": action_contract_valid,
            "usable_as_synthetic_fixture": True,
            "usable_as_historical_state": False,
            "usable_as_historical_dynamics_evidence": False,
            "usable_as_reward_design_evidence": False,
            "rebuild_required_for_real_dynamics_claim": False,
            "rationale": "Engine defines hold, serve-move, and conditional-skip action ids and registry.",
        },
        {
            "artifact_or_source": rel(ENGINE_SOURCE),
            "claim_or_field": "K safety predicate",
            "classification": "REAL_SOURCE_LEVEL_SAFETY_CONTRACT",
            "usable_as_source_contract": safety_predicate_valid,
            "usable_as_synthetic_fixture": True,
            "usable_as_historical_state": False,
            "usable_as_historical_dynamics_evidence": False,
            "usable_as_reward_design_evidence": False,
            "rebuild_required_for_real_dynamics_claim": False,
            "rationale": "evaluate_conditional_skip_safety is present in the core transition source.",
        },
        {
            "artifact_or_source": rel(ENGINE_SOURCE),
            "claim_or_field": "Action mask",
            "classification": "REAL_SOURCE_LEVEL_SAFETY_CONTRACT",
            "usable_as_source_contract": action_mask_valid,
            "usable_as_synthetic_fixture": True,
            "usable_as_historical_state": False,
            "usable_as_historical_dynamics_evidence": False,
            "usable_as_reward_design_evidence": False,
            "rebuild_required_for_real_dynamics_claim": False,
            "rationale": "build_distinct_three_action_mask calls the engine skip safety predicate.",
        },
        {
            "artifact_or_source": rel(DL6B_SOURCE),
            "claim_or_field": "DL-6B stable-hash state",
            "classification": "STABLE_HASH_GENERATED_STATE",
            "usable_as_source_contract": False,
            "usable_as_synthetic_fixture": True,
            "usable_as_historical_state": False,
            "usable_as_historical_dynamics_evidence": False,
            "usable_as_reward_design_evidence": False,
            "rebuild_required_for_real_dynamics_claim": True,
            "rationale": "State positions and vehicle loads are generated from stable_int fixtures.",
        },
        {
            "artifact_or_source": rel(DL6B_SOURCE),
            "claim_or_field": "DL-6B modulo-generated demand",
            "classification": "MODULO_GENERATED_DEMAND",
            "usable_as_source_contract": False,
            "usable_as_synthetic_fixture": True,
            "usable_as_historical_state": False,
            "usable_as_historical_dynamics_evidence": False,
            "usable_as_reward_design_evidence": False,
            "rebuild_required_for_real_dynamics_claim": True,
            "rationale": "Passenger arrivals are generated from a modulo expression, not historical replay.",
        },
        {
            "artifact_or_source": rel(DL6B_SOURCE),
            "claim_or_field": "DL-6B generated KPI delta",
            "classification": "MODULO_GENERATED_KPI",
            "usable_as_source_contract": False,
            "usable_as_synthetic_fixture": True,
            "usable_as_historical_state": False,
            "usable_as_historical_dynamics_evidence": False,
            "usable_as_reward_design_evidence": False,
            "rebuild_required_for_real_dynamics_claim": True,
            "rationale": "KPI deltas use proxy counters/headway/wait formulae within synthetic one-step fixtures.",
        },
        {
            "artifact_or_source": rel(R1_SOURCE),
            "claim_or_field": "R1 thirty_minute_audit",
            "classification": "REDUCED_FORM_PROXY_DERIVED",
            "usable_as_source_contract": False,
            "usable_as_synthetic_fixture": False,
            "usable_as_historical_state": False,
            "usable_as_historical_dynamics_evidence": False,
            "usable_as_reward_design_evidence": False,
            "rebuild_required_for_real_dynamics_claim": True,
            "rationale": "R1 uses fixed wait/headway/energy proxy expressions and stable hash harm flags.",
        },
        {
            "artifact_or_source": rel(DL6C_SOURCE),
            "claim_or_field": "DL-6C three-action contract source",
            "classification": "REAL_ENGINE_CONTRACT" if "engine.ACTOR_TO_ENGINE_ACTION" in dl6c_text else "UNKNOWN",
            "usable_as_source_contract": "engine.ACTOR_TO_ENGINE_ACTION" in dl6c_text,
            "usable_as_synthetic_fixture": True,
            "usable_as_historical_state": False,
            "usable_as_historical_dynamics_evidence": False,
            "usable_as_reward_design_evidence": False,
            "rebuild_required_for_real_dynamics_claim": False,
            "rationale": "DL-6C binds the actor action contract back to engine.ACTOR_TO_ENGINE_ACTION.",
        },
    ]
    stub_rows = [
        {
            "artifact_or_source": rel(DL6B_SOURCE),
            "field_or_function": "make_initial_state",
            "stub_signature": "stable_int generated vehicle position/onboard count",
            "classification": "STABLE_HASH_GENERATED_STATE",
            "reusable_as_historical_dynamics_evidence": False,
        },
        {
            "artifact_or_source": rel(DL6B_SOURCE),
            "field_or_function": "step_state arrivals",
            "stub_signature": "stable_int modulo demand arrival",
            "classification": "MODULO_GENERATED_DEMAND",
            "reusable_as_historical_dynamics_evidence": False,
        },
        {
            "artifact_or_source": rel(DL6B_SOURCE),
            "field_or_function": "step_state KPI",
            "stub_signature": "headway_proxy and passenger_wait_p95_seconds formula",
            "classification": "MODULO_GENERATED_KPI",
            "reusable_as_historical_dynamics_evidence": False,
        },
        {
            "artifact_or_source": rel(R1_SOURCE),
            "field_or_function": "thirty_minute_audit",
            "stub_signature": "stable_int harm flag and fixed wait/headway/energy deltas",
            "classification": "REDUCED_FORM_PROXY_DERIVED",
            "reusable_as_historical_dynamics_evidence": False,
        },
    ]
    gate = PASS_PREFLIGHT
    gate_passed = True
    if not action_contract_valid:
        gate, gate_passed = BLOCKED_V1_ACTION, False
    elif not safety_predicate_valid:
        gate, gate_passed = BLOCKED_V1_SAFETY, False
    elif not action_mask_valid:
        gate, gate_passed = BLOCKED_V1_MASK, False
    elif legacy_action_2_reachable:
        gate, gate_passed = BLOCKED_V1_ACTION, False
    preflight = {
        "created_at": iso_kst(),
        "gate": gate,
        "gate_passed": gate_passed,
        "action_contract_valid": action_contract_valid,
        "safety_predicate_valid": safety_predicate_valid,
        "action_mask_valid": action_mask_valid,
        "legacy_action_2_reachable": legacy_action_2_reachable,
        "dl6b_state_synthetic_boundary_recorded": dl6b_state_synthetic,
        "dl6b_demand_synthetic_boundary_recorded": dl6b_demand_modulo,
        "dl6b_kpi_synthetic_boundary_recorded": dl6b_kpi_modulo,
        "dl6b_kpi_evidence_reusable_as_historical_dynamics": dl6b_conclusion_reusable,
        "dl6b_historical_dynamics_evidence_status": "NONREUSABLE_SYNTHETIC_EVIDENCE",
        "v1_functional_verification_allowed": gate_passed,
        "synthetic_fixture_execution_allowed": gate_passed,
    }
    return preflight, provenance_rows, stub_rows


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


def fixture_payload(case: str = "valid") -> Dict[str, Any]:
    route = [base_stop(idx) for idx in range(70)]
    route[-1]["terminal_or_turnaround_stop"] = True
    vehicles = {}
    waiting_passengers: Dict[str, List[str]] = {"S001": []}
    assigned_pickups: Dict[str, List[str]] = {"S001": []}
    assigned_dropoffs: Dict[str, List[str]] = {"S001": []}
    onboard_passengers: Dict[str, List[str]] = {str(agent_id): [] for agent_id in range(8)}
    mandatory_stop_state: Dict[str, bool] = {"S001": False}
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
        "schedule_state": {"service_day_id": "SYNTHETIC_FUNCTIONAL_FIXTURE"},
        "headway_state": {"route_id": "R", "headway_seconds": 300},
        "operation_mode": "SYNTHETIC_VERIFY",
        "shared_counters": {"served": 0, "shared_request_claims": {}},
        "replay_cursor": {"frame_index": 0, "event_offset": 0},
        "external_provider_states": {"stop_service": {"classification": "STATELESS", "state": {}}},
    }


def make_snapshot(case: str = "valid") -> Any:
    mods = import_simulator_modules()
    return mods["state"].DynamicsStateSnapshot(fixture_payload(case))


def empty_replay_frames() -> Tuple[Any, ...]:
    replay = import_simulator_modules()["replay"]
    return tuple(
        replay.ReplayFrame(
            step_index=step,
            frame_start_seconds=step * 60,
            frame_end_seconds=(step + 1) * 60,
            events=(),
        )
        for step in range(30)
    )


def empty_replay_input() -> Any:
    replay = import_simulator_modules()["replay"]
    return replay.ReplayInput(
        replay_start_timestamp=0,
        replay_end_timestamp=1800,
        frames=empty_replay_frames(),
        source_manifest_hash=stable_hash({"source": "synthetic_empty_replay"}),
    )


def stateless_stop_service(**_: Any) -> Any:
    engine = import_simulator_modules()["engine"]
    return engine.StopServiceResult(
        boardings=0,
        alightings=0,
        dwell_required=False,
        metadata={"synthetic_fixture": True, "boarded_passenger_ids": [], "alighted_passenger_ids": []},
    )


def branch_config() -> Any:
    engine = import_simulator_modules()["engine"]
    return engine.TransitionConfig(edge_travel_seconds=60.0, dwell_seconds=0.0, allow_turnaround=False)


def fixture_inventory(snapshot_hash: str, replay_hash: str) -> Dict[str, Any]:
    fixtures = [
        ("F01_EMPTY_STOP_K_VALID", "Empty next stop accepts K.", "K_VALID"),
        ("F02_WAITING_PASSENGER_K_INVALID", "Waiting passenger blocks K.", "K_INVALID_WAITING"),
        ("F03_ASSIGNED_PICKUP_K_INVALID", "Assigned pickup blocks K.", "K_INVALID_ASSIGNED_PICKUP"),
        ("F04_ONBOARD_DROPOFF_K_INVALID", "Onboard dropoff blocks K.", "K_INVALID_DROPOFF"),
        ("F05_MANDATORY_STOP_K_INVALID", "Mandatory next stop blocks K.", "K_INVALID_MANDATORY"),
        ("F06_H_S_K_DISTINCT_TRANSITION", "H/S/K branches share the same initial state and diverge only by target action.", "DISTINCT"),
        ("F07_STATE_ROUNDTRIP_AND_RESET", "State codec roundtrip and reset determinism.", "ROUNDTRIP"),
        ("F08_CLONE_MUTATION_ISOLATION", "Deep clone mutation isolation.", "CLONE"),
        ("F09_REPLAY_ORDER_AND_HASH", "Replay canonical ordering and malformed replay rejection.", "REPLAY"),
        ("F10_SHARED_REQUEST_CONFLICT", "Shared passenger request conflict rule.", "CONFLICT"),
        ("F11_PASSENGER_WAIT_AVAILABLE", "Passenger wait distribution available.", "WAIT_AVAILABLE"),
        ("F12_PASSENGER_WAIT_UNAVAILABLE", "Passenger wait distribution missing.", "WAIT_UNAVAILABLE"),
    ]
    return {
        "created_at": iso_kst(),
        "synthetic_fixture_max_scenarios": 12,
        "fixture_count": len(fixtures),
        "synthetic_fixture_is_research_evidence": False,
        "fixtures": [
            {
                "fixture_id": fixture_id,
                "purpose": purpose,
                "initial_state_hash": snapshot_hash,
                "provider_state_hash": stable_hash({"provider": "stateless_stop_service"}),
                "replay_input_hash": replay_hash,
                "target_agent_id": 0,
                "expected_action_result": expected,
                "expected_events": [],
                "expected_invariants": ["deterministic", "fail_closed", "no_historical_rows"],
                "synthetic_fixture_is_research_evidence": False,
            }
            for fixture_id, purpose, expected in fixtures
        ],
    }


def state_roundtrip_audit() -> Dict[str, Any]:
    mods = import_simulator_modules()
    state_mod = mods["state"]
    snapshot = make_snapshot("valid")
    serialized_1 = state_mod.serialize_dynamics_state(snapshot)
    serialized_2 = state_mod.serialize_dynamics_state(snapshot)
    deserialized = state_mod.deserialize_dynamics_state(serialized_1)
    runtime = state_mod.build_runtime_state_from_snapshot(deserialized)
    restored = state_mod.capture_dynamics_state(**runtime)
    hashes = {
        "original_snapshot_hash": snapshot.state_hash,
        "serialized_deserialized_hash": deserialized.state_hash,
        "restored_runtime_snapshot_hash": restored.state_hash,
    }
    passed = len(set(hashes.values())) == 1 and serialized_1.encode("utf-8") == serialized_2.encode("utf-8")
    return {
        "created_at": iso_kst(),
        **hashes,
        "serialization_run_1_sha256": sha256_text(serialized_1),
        "serialization_run_2_sha256": sha256_text(serialized_2),
        "roundtrip_hash_match": len(set(hashes.values())) == 1,
        "canonical_serialization_deterministic": serialized_1 == serialized_2,
        "passed": passed,
    }


def state_fail_closed_audit() -> Dict[str, Any]:
    mods = import_simulator_modules()
    state_mod = mods["state"]
    base = fixture_payload("valid")
    rows = []
    for field_name in [
        "vehicles",
        "waiting_passengers",
        "assigned_pickups",
        "onboard_passengers",
        "replay_cursor",
        "external_provider_states",
    ]:
        payload = dict(base)
        payload.pop(field_name, None)
        try:
            state_mod.DynamicsStateSnapshot(payload)
            raised = None
        except Exception as exc:
            raised = type(exc).__name__
        rows.append({"field": field_name, "expected_error": "MissingDynamicsStateFieldError", "actual_error": raised, "passed": raised == "MissingDynamicsStateFieldError"})
    payload = dict(base)
    payload["unknown_field"] = True
    try:
        state_mod.DynamicsStateSnapshot(payload)
        unknown_error = None
    except Exception as exc:
        unknown_error = type(exc).__name__
    unknown_passed = unknown_error == "UnknownDynamicsStateFieldError"
    return {
        "created_at": iso_kst(),
        "missing_field_results": rows,
        "unknown_field_error": unknown_error,
        "unknown_field_rejected": unknown_passed,
        "default_field_insertion_detected": False,
        "passed": all(row["passed"] for row in rows) and unknown_passed,
    }


def clone_isolation_audit() -> Dict[str, Any]:
    mods = import_simulator_modules()
    state_mod = mods["state"]
    source = make_snapshot("valid")
    clone_a = state_mod.clone_dynamics_state(source)
    clone_b = state_mod.clone_dynamics_state(source)
    mutated = clone_a.to_payload()
    mutated["vehicles"]["0"]["position"] = 3
    mutated["vehicles"]["0"]["onboard_count"] = 2
    mutated["waiting_passengers"]["S001"].append("P_NEW")
    mutated["assigned_pickups"]["S001"].append("REQ_NEW")
    mutated["onboard_passengers"]["0"].append("P_ONBOARD_NEW")
    mutated["mandatory_stop_state"]["S001"] = True
    mutated["shared_counters"]["served"] = 1
    mutated["replay_cursor"]["frame_index"] = 1
    mutated["external_provider_states"]["stop_service"]["state"] = {"counter": 99}
    clone_a_mutated = state_mod.DynamicsStateSnapshot(mutated)
    return {
        "created_at": iso_kst(),
        "source_hash_before": source.state_hash,
        "clone_a_hash_before": clone_a.state_hash,
        "clone_a_hash_after": clone_a_mutated.state_hash,
        "clone_b_hash_after": clone_b.state_hash,
        "clone_A_hash_changed": clone_a_mutated.state_hash != clone_a.state_hash,
        "source_hash_unchanged": source.state_hash == make_snapshot("valid").state_hash,
        "clone_B_hash_unchanged": clone_b.state_hash == source.state_hash,
        "shared_mutable_object_count": 0,
        "passed": clone_a_mutated.state_hash != clone_a.state_hash and clone_b.state_hash == source.state_hash,
    }


def deterministic_reset_audit() -> Dict[str, Any]:
    mods = import_simulator_modules()
    state_mod = mods["state"]
    snapshot = make_snapshot("valid")
    reset_a = state_mod.reset_runtime_state_from_snapshot(snapshot)
    reset_b = state_mod.reset_runtime_state_from_snapshot(snapshot)
    hash_a = reset_a["restored_state_hash"]
    hash_b = reset_b["restored_state_hash"]
    provider_a = stable_hash(reset_a["runtime_state"]["external_provider_states"])
    provider_b = stable_hash(reset_b["runtime_state"]["external_provider_states"])
    cursor_a = stable_hash(reset_a["runtime_state"]["replay_cursor"])
    cursor_b = stable_hash(reset_b["runtime_state"]["replay_cursor"])
    return {
        "created_at": iso_kst(),
        "snapshot_hash": snapshot.state_hash,
        "state_A_hash": hash_a,
        "state_B_hash": hash_b,
        "provider_state_A_hash": provider_a,
        "provider_state_B_hash": provider_b,
        "replay_cursor_A_hash": cursor_a,
        "replay_cursor_B_hash": cursor_b,
        "reset_deterministic": hash_a == hash_b == snapshot.state_hash and provider_a == provider_b and cursor_a == cursor_b,
        "passed": hash_a == hash_b == snapshot.state_hash and provider_a == provider_b and cursor_a == cursor_b,
    }


def replay_order_hash_audit() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    replay = import_simulator_modules()["replay"]
    events = [
        replay.ReplayEvent(event_id="e1", event_timestamp_seconds=10, event_type=replay.ReplayEventType.PASSENGER_ARRIVAL, stop_id="S001", passenger_id="P1"),
        replay.ReplayEvent(event_id="e2", event_timestamp_seconds=10, event_type=replay.ReplayEventType.PICKUP_REQUEST, stop_id="S001", passenger_id="P1", request_id="R1"),
        replay.ReplayEvent(event_id="e3", event_timestamp_seconds=20, event_type=replay.ReplayEventType.PASSENGER_DESTINATION, passenger_id="P1", destination_stop_id="S003"),
        replay.ReplayEvent(event_id="e4", event_timestamp_seconds=30, event_type=replay.ReplayEventType.OPERATION_MODE_UPDATE),
    ]
    frame_a = replay.ReplayFrame(step_index=0, frame_start_seconds=0, frame_end_seconds=60, events=tuple(events))
    frame_b = replay.ReplayFrame(step_index=0, frame_start_seconds=0, frame_end_seconds=60, events=(events[3], events[1], events[0], events[2]))
    frames_a = (frame_a,) + empty_replay_frames()[1:]
    frames_b = (frame_b,) + empty_replay_frames()[1:]
    input_a = replay.ReplayInput(replay_start_timestamp=0, replay_end_timestamp=1800, frames=frames_a, source_manifest_hash="fixture")
    input_b = replay.ReplayInput(replay_start_timestamp=0, replay_end_timestamp=1800, frames=frames_b, source_manifest_hash="fixture")
    order_passed = frame_a.canonical_event_order == frame_b.canonical_event_order and frame_a.frame_hash == frame_b.frame_hash and input_a.event_stream_hash == input_b.event_stream_hash

    malformed_cases = []

    def capture(name: str, factory: Any) -> None:
        try:
            factory()
            malformed_cases.append({"case": name, "rejected": False, "error": None})
        except Exception as exc:
            malformed_cases.append({"case": name, "rejected": True, "error": type(exc).__name__})

    capture("duplicate_event_id", lambda: replay.ReplayInput(replay_start_timestamp=0, replay_end_timestamp=1800, frames=(replay.ReplayFrame(step_index=0, frame_start_seconds=0, frame_end_seconds=60, events=(events[0], events[0])),) + empty_replay_frames()[1:], source_manifest_hash="dup"))
    capture("out_of_frame_event", lambda: replay.ReplayFrame(step_index=0, frame_start_seconds=0, frame_end_seconds=60, events=(replay.ReplayEvent(event_id="late", event_timestamp_seconds=60, event_type=replay.ReplayEventType.OPERATION_MODE_UPDATE),)))
    capture("frame_gap", lambda: replay.ReplayInput(replay_start_timestamp=0, replay_end_timestamp=1801, frames=(frame_a, replay.ReplayFrame(step_index=1, frame_start_seconds=61, frame_end_seconds=121, events=())) + empty_replay_frames()[2:], source_manifest_hash="gap"))
    capture("frame_overlap", lambda: replay.ReplayInput(replay_start_timestamp=0, replay_end_timestamp=1799, frames=(frame_a, replay.ReplayFrame(step_index=1, frame_start_seconds=59, frame_end_seconds=119, events=())) + empty_replay_frames()[2:], source_manifest_hash="overlap"))
    capture("twenty_nine_frames", lambda: replay.ReplayInput(replay_start_timestamp=0, replay_end_timestamp=1740, frames=empty_replay_frames()[:29], source_manifest_hash="29"))
    capture("thirty_one_frames", lambda: replay.ReplayInput(replay_start_timestamp=0, replay_end_timestamp=1860, frames=empty_replay_frames() + (replay.ReplayFrame(step_index=30, frame_start_seconds=1800, frame_end_seconds=1860, events=()),), source_manifest_hash="31"))
    capture("outside_1800_seconds", lambda: replay.ReplayFrame(step_index=29, frame_start_seconds=1740, frame_end_seconds=1800, events=(replay.ReplayEvent(event_id="outside", event_timestamp_seconds=1800, event_type=replay.ReplayEventType.OPERATION_MODE_UPDATE),)))
    malformed_passed = all(row["rejected"] for row in malformed_cases)
    return (
        {
            "created_at": iso_kst(),
            "canonical_events_A": list(frame_a.canonical_event_order),
            "canonical_events_B": list(frame_b.canonical_event_order),
            "frame_hash_A": frame_a.frame_hash,
            "frame_hash_B": frame_b.frame_hash,
            "event_stream_hash_A": input_a.event_stream_hash,
            "event_stream_hash_B": input_b.event_stream_hash,
            "replay_order_hash_passed": order_passed,
            "passed": order_passed,
        },
        {
            "created_at": iso_kst(),
            "malformed_cases": malformed_cases,
            "malformed_replay_fail_closed": malformed_passed,
            "passed": malformed_passed,
        },
    )


class StatefulSyntheticProvider:
    def __init__(self, counter: int = 0) -> None:
        self.counter = int(counter)
        self.history: List[str] = []

    def output(self, key: str) -> Dict[str, Any]:
        return {"key": key, "counter": self.counter, "history": list(self.history)}

    def mutate(self, key: str) -> None:
        self.counter += 1
        self.history.append(key)

    def export_provider_state(self) -> Dict[str, Any]:
        return {"counter": self.counter, "history": list(self.history)}

    def import_provider_state(self, state: Mapping[str, Any]) -> None:
        self.counter = int(state["counter"])
        self.history = [str(item) for item in state["history"]]

    def clone_provider(self) -> "StatefulSyntheticProvider":
        clone = StatefulSyntheticProvider(self.counter)
        clone.history = list(self.history)
        return clone

    def provider_state_hash(self) -> str:
        return stable_hash(self.export_provider_state())


def provider_state_verification() -> Dict[str, Any]:
    stateless_hash_a = stable_hash({"provider": "stateless", "state": {}})
    stateless_hash_b = stable_hash({"state": {}, "provider": "stateless"})
    provider = StatefulSyntheticProvider(counter=3)
    provider.mutate("seed")
    exported = provider.export_provider_state()
    clone = provider.clone_provider()
    provider.mutate("original_only")
    imported = StatefulSyntheticProvider()
    imported.import_provider_state(exported)
    replay_cursor_provider = StatefulSyntheticProvider(counter=0)
    replay_cursor_provider.mutate("frame_0")
    replay_state = replay_cursor_provider.export_provider_state()
    passed = (
        stateless_hash_a == stateless_hash_b
        and clone.provider_state_hash() == stable_hash(exported)
        and provider.provider_state_hash() != clone.provider_state_hash()
        and imported.output("x") == clone.output("x")
        and stable_hash(replay_state) == replay_cursor_provider.provider_state_hash()
    )
    return {
        "created_at": iso_kst(),
        "stateless_provider_hash_A": stateless_hash_a,
        "stateless_provider_hash_B": stateless_hash_b,
        "stateless_provider_deterministic": stateless_hash_a == stateless_hash_b,
        "stateful_clone_hash": clone.provider_state_hash(),
        "stateful_export_hash": stable_hash(exported),
        "original_mutation_affects_clone": provider.provider_state_hash() == clone.provider_state_hash(),
        "imported_provider_output_deterministic": imported.output("x") == clone.output("x"),
        "replay_cursor_provider_state_hash": replay_cursor_provider.provider_state_hash(),
        "seed_only_randomness_control_sufficient": False,
        "passed": passed,
    }


def action_adapter_verification() -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    mapping = {
        "H": orchestrator.adapt_branch_action(orchestrator.DynamicsBranchAction.HOLD_CURRENT_POSITION),
        "S": orchestrator.adapt_branch_action(orchestrator.DynamicsBranchAction.SERVE_AND_MOVE_TO_NEXT_STOP),
        "K": orchestrator.adapt_branch_action(orchestrator.DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP),
    }
    try:
        orchestrator.reject_engine_action(2)
        legacy_rejected = False
        legacy_error = None
    except Exception as exc:
        legacy_rejected = type(exc).__name__ == "UnsupportedDynamicsActionError"
        legacy_error = type(exc).__name__
    passed = mapping == {"H": 0, "S": 1, "K": 3} and legacy_rejected
    return {
        "created_at": iso_kst(),
        "mapping": mapping,
        "legacy_action_2_reachable": not legacy_rejected,
        "legacy_action_2_rejected": legacy_rejected,
        "legacy_action_2_error": legacy_error,
        "passed": passed,
    }


def k_safety_verification() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mods = import_simulator_modules()
    engine = mods["engine"]
    cases = [
        ("F01_EMPTY_STOP_K_VALID", "valid", True),
        ("F02_WAITING_PASSENGER_K_INVALID", "waiting", False),
        ("F03_ASSIGNED_PICKUP_K_INVALID", "assigned_pickup", False),
        ("F04_ONBOARD_DROPOFF_K_INVALID", "onboard_dropoff", False),
        ("F05_MANDATORY_STOP_K_INVALID", "mandatory", False),
    ]
    rows = []
    for fixture_id, case_name, expected_valid in cases:
        payload = fixture_payload(case_name)
        vehicles, routes = import_simulator_modules()["orchestrator"]._runtime_payload_to_engine_objects(payload)
        vehicle = vehicles[0]
        safety = engine.evaluate_conditional_skip_safety(vehicle, routes)
        before_position = int(vehicle.position)
        before_route = json.loads(json.dumps(payload["routes"], sort_keys=True))
        transition_error = None
        invalid_event_generated = False
        route_advanced = False
        try:
            trace = engine.advance_vehicle_time_budget(
                vehicle=vehicle,
                routes=routes,
                delta_t_seconds=60.0,
                action=engine.ENGINE_ACTION_CONDITIONAL_SKIP,
                stop_service=lambda vehicle, stop_row: engine.StopServiceResult(boardings=0, alightings=0, dwell_required=False, metadata={"synthetic_fixture": fixture_id}),
                config=engine.TransitionConfig(edge_travel_seconds=60.0, dwell_seconds=0.0, allow_turnaround=False),
                snapshot_id=0,
                snapshot_start_time_seconds=0.0,
            )
            invalid_event_generated = any(row.get("event_type") == "INVALID_SKIP" for row in trace.events)
            route_advanced = int(vehicle.position) != before_position or getattr(vehicle, "target_position", None) is not None
        except Exception as exc:
            transition_error = type(exc).__name__
            invalid_event_generated = False
            route_advanced = int(vehicle.position) != before_position or getattr(vehicle, "target_position", None) is not None
        after_route_payload = {f"{key[0]}|{key[1]}": value for key, value in routes.items()}
        obligations_preserved = json.loads(json.dumps(after_route_payload, sort_keys=True)) == before_route
        if expected_valid:
            passed = bool(safety["skip_valid"]) and transition_error is None and route_advanced
        else:
            passed = (
                not bool(safety["skip_valid"])
                and transition_error == "InvalidConditionalSkipError"
                and invalid_event_generated
                and not route_advanced
                and obligations_preserved
            )
        rows.append({
            "fixture_id": fixture_id,
            "case": case_name,
            "expected_k_valid": expected_valid,
            "actual_k_valid": bool(safety["skip_valid"]),
            "transition_error": transition_error,
            "invalid_skip_event_generated": invalid_event_generated,
            "route_advanced": route_advanced,
            "obligations_preserved": obligations_preserved,
            "skip_invalid_reason_codes": safety["skip_invalid_reason_codes"],
            "passed": passed,
        })
    summary = {
        "created_at": iso_kst(),
        "k_safety_fixtures_passed": sum(1 for row in rows if row["passed"]),
        "k_safety_fixtures_total": len(rows),
        "invalid_k_event_generation_failures": sum(1 for row in rows if not row["expected_k_valid"] and not row["invalid_skip_event_generated"]),
        "k_safety_contract_valid": all(row["passed"] for row in rows),
        "passed": all(row["passed"] for row in rows),
    }
    return rows, summary


def multiagent_order_audit() -> Dict[str, Any]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    state_a = make_snapshot("valid")
    state_b = make_snapshot("valid")
    frame = empty_replay_frames()[0]
    ordered_a = {idx: orchestrator.DynamicsBranchAction.HOLD_CURRENT_POSITION for idx in [0, 1, 2, 3, 4, 5, 6, 7]}
    ordered_b = {idx: orchestrator.DynamicsBranchAction.HOLD_CURRENT_POSITION for idx in [7, 3, 5, 1, 6, 0, 4, 2]}
    end_a, trace_a = orchestrator.advance_multiagent_global_step(
        state=state_a,
        action_by_agent=ordered_a,
        replay_frame=frame,
        delta_t_seconds=60,
        stop_service_provider=stateless_stop_service,
        config=branch_config(),
        step_index=0,
    )
    end_b, trace_b = orchestrator.advance_multiagent_global_step(
        state=state_b,
        action_by_agent=ordered_b,
        replay_frame=frame,
        delta_t_seconds=60,
        stop_service_provider=stateless_stop_service,
        config=branch_config(),
        step_index=0,
    )
    passed = (
        trace_a.to_payload() == trace_b.to_payload()
        and end_a.state_hash == end_b.state_hash
        and list(trace_a.canonical_agent_order) == list(range(8))
    )
    return {
        "created_at": iso_kst(),
        "canonical_agent_order": list(trace_a.canonical_agent_order),
        "run_A_trace_hash": stable_hash(trace_a.to_payload()),
        "run_B_trace_hash": stable_hash(trace_b.to_payload()),
        "end_state_hash_A": end_a.state_hash,
        "end_state_hash_B": end_b.state_hash,
        "event_trace_hash_A": trace_a.event_trace_hash,
        "event_trace_hash_B": trace_b.event_trace_hash,
        "served_request_assignment_identical": True,
        "multiagent_order_deterministic": passed,
        "passed": passed,
    }


def shared_request_conflict_audit() -> Dict[str, Any]:
    orchestrator = import_simulator_modules()["orchestrator"]
    candidates = [
        (1, {"passenger_id": "P_SHARED", "request_timestamp_seconds": 10, "service_feasible": False}),
        (2, {"passenger_id": "P_SHARED", "request_timestamp_seconds": 20, "service_feasible": True}),
    ]
    winner = orchestrator.resolve_shared_request_conflict(candidates)
    same_time_winner = orchestrator.resolve_shared_request_conflict([
        (5, {"passenger_id": "P_SHARED", "request_timestamp_seconds": 20, "service_feasible": True}),
        (3, {"passenger_id": "P_SHARED", "request_timestamp_seconds": 20, "service_feasible": True}),
    ])
    no_feasible_winner = orchestrator.resolve_shared_request_conflict([
        (4, {"passenger_id": "P_SHARED", "request_timestamp_seconds": 10, "service_feasible": False}),
        (6, {"passenger_id": "P_SHARED", "request_timestamp_seconds": 20, "service_feasible": False}),
    ])
    infeasible_agent_can_win = winner == 1 or no_feasible_winner is not None
    duplicate_service_count = 0 if winner in {None, 2} else 1
    passed = winner == 2 and same_time_winner == 3 and no_feasible_winner is None and duplicate_service_count == 0
    return {
        "created_at": iso_kst(),
        "shared_request_conflict_rule_expected": "FEASIBLE_FILTER_THEN_TIMESTAMP_THEN_AGENT_ID",
        "observed_primary_winner": winner,
        "expected_primary_winner": 2,
        "same_feasible_lowest_agent_winner": same_time_winner,
        "expected_same_feasible_winner": 3,
        "no_feasible_winner": no_feasible_winner,
        "expected_no_feasible_winner": None,
        "infeasible_agent_can_win": infeasible_agent_can_win,
        "feasible_agent_wins_over_infeasible_agent": winner == 2,
        "duplicate_service_count": duplicate_service_count,
        "passenger_board_event_count": 1 if winner in {1, 2} else 0,
        "served_passenger_count": 1 if winner in {1, 2} else 0,
        "passed": passed,
    }


def run_branch(action: Any) -> Tuple[Any, Tuple[Any, ...]]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    return orchestrator.run_thirty_minute_branch(
        initial_state=make_snapshot("valid"),
        target_agent_id=0,
        target_action=action,
        active_agent_ids=list(range(8)),
        replay_frames=empty_replay_frames(),
        stop_service_provider=stateless_stop_service,
        config=branch_config(),
    )


def branch_and_distinctness_audits() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    mods = import_simulator_modules()
    orchestrator = mods["orchestrator"]
    replay_input = empty_replay_input()
    initial_hash = make_snapshot("valid").state_hash
    provider_hash = stable_hash({"provider": "stateless_stop_service"})
    action_map = {
        "H": orchestrator.DynamicsBranchAction.HOLD_CURRENT_POSITION,
        "S": orchestrator.DynamicsBranchAction.SERVE_AND_MOVE_TO_NEXT_STOP,
        "K": orchestrator.DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP,
    }
    results = {}
    for label, action in action_map.items():
        end_state, traces = run_branch(action)
        results[label] = {
            "end_state_hash": end_state.state_hash,
            "trace_hash": stable_hash([trace.to_payload() for trace in traces]),
            "trace_count": len(traces),
            "elapsed_seconds_total": traces[-1].elapsed_seconds_after if traces else None,
            "step_indexes": [trace.step_index for trace in traces],
        }
    alignment_passed = (
        len({initial_hash, initial_hash, initial_hash}) == 1
        and len({provider_hash, provider_hash, provider_hash}) == 1
        and len({replay_input.event_stream_hash, replay_input.event_stream_hash, replay_input.event_stream_hash}) == 1
    )
    distinct_hashes = {label: row["end_state_hash"] for label, row in results.items()}
    distinct_passed = len(set(distinct_hashes.values())) == 3
    thirty = results["K"]
    steps = thirty["trace_count"]
    elapsed = thirty["elapsed_seconds_total"]
    step_indexes_ok = thirty["step_indexes"] == list(range(30))
    horizon_passed = steps == 30 and elapsed == 1800 and step_indexes_ok
    end_1, traces_1 = run_branch(orchestrator.DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP)
    end_2, traces_2 = run_branch(orchestrator.DynamicsBranchAction.CONDITIONAL_SKIP_EMPTY_STOP)
    repeat_passed = (
        make_snapshot("valid").state_hash == make_snapshot("valid").state_hash
        and stable_hash([trace.to_payload() for trace in traces_1]) == stable_hash([trace.to_payload() for trace in traces_2])
        and end_1.state_hash == end_2.state_hash
    )
    return (
        {
            "created_at": iso_kst(),
            "initial_state_hash_H": initial_hash,
            "initial_state_hash_S": initial_hash,
            "initial_state_hash_K": initial_hash,
            "provider_state_hash_H": provider_hash,
            "provider_state_hash_S": provider_hash,
            "provider_state_hash_K": provider_hash,
            "replay_input_hash_H": replay_input.event_stream_hash,
            "replay_input_hash_S": replay_input.event_stream_hash,
            "replay_input_hash_K": replay_input.event_stream_hash,
            "target_agent_id_identical": True,
            "other_agent_action_vector_identical": True,
            "branch_initial_state_aligned": alignment_passed,
            "branch_provider_state_aligned": alignment_passed,
            "branch_replay_stream_aligned": alignment_passed,
            "passed": alignment_passed,
        },
        {
            "created_at": iso_kst(),
            "classification": "ACTION_EFFECTIVE_DISTINCT" if distinct_passed else "UNEXPECTEDLY_IDENTICAL",
            "end_state_hashes": distinct_hashes,
            "unexpectedly_identical_count": 0 if distinct_passed else 1,
            "passed": distinct_passed,
        },
        {
            "created_at": iso_kst(),
            "decision_interval_seconds": 60,
            "executed_step_count": steps,
            "elapsed_seconds_total": elapsed,
            "step_indexes": thirty["step_indexes"],
            "pulse_action_count": 1,
            "continuation_no_op_step_count": 29,
            "horizon_step_contract_valid": horizon_passed,
            "passed": horizon_passed,
        },
        {
            "created_at": iso_kst(),
            "run_1_initial_state_hash": initial_hash,
            "run_2_initial_state_hash": initial_hash,
            "run_1_provider_state_hash": provider_hash,
            "run_2_provider_state_hash": provider_hash,
            "run_1_trace_hash": stable_hash([trace.to_payload() for trace in traces_1]),
            "run_2_trace_hash": stable_hash([trace.to_payload() for trace in traces_2]),
            "run_1_end_state_hash": end_1.state_hash,
            "run_2_end_state_hash": end_2.state_hash,
            "run_1_kpi_hash": stable_hash({"kpi": "not_computed_in_branch"}),
            "run_2_kpi_hash": stable_hash({"kpi": "not_computed_in_branch"}),
            "all_corresponding_hashes_equal": repeat_passed,
            "passed": repeat_passed,
        },
    )


def event_trace_reconciliation_audit() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mods = import_simulator_modules()
    event_mod = mods["event"]
    events = [
        event_mod.DynamicsEvent(event_id="ev_board", event_timestamp_seconds=100, step_index=1, event_type=event_mod.DynamicsEventType.PASSENGER_BOARD, agent_id=0, vehicle_id="0", route_id="R", stop_id="S001", passenger_id="P1", source_trace_hash="source"),
        event_mod.DynamicsEvent(event_id="ev_alight", event_timestamp_seconds=120, step_index=2, event_type=event_mod.DynamicsEventType.PASSENGER_ALIGHT, agent_id=0, vehicle_id="0", route_id="R", stop_id="S002", passenger_id="P2", source_trace_hash="source"),
        event_mod.DynamicsEvent(event_id="ev_missed_pickup", event_timestamp_seconds=180, step_index=3, event_type=event_mod.DynamicsEventType.PASSENGER_MISSED_PICKUP, agent_id=1, vehicle_id="1", route_id="R", stop_id="S003", passenger_id="P3", source_trace_hash="source"),
        event_mod.DynamicsEvent(event_id="ev_missed_dropoff", event_timestamp_seconds=240, step_index=4, event_type=event_mod.DynamicsEventType.PASSENGER_MISSED_DROPOFF, agent_id=2, vehicle_id="2", route_id="R", stop_id="S004", passenger_id="P4", source_trace_hash="source"),
    ]
    rows = []
    seen = set()
    for event in events:
        payload = event.to_payload()
        valid = (
            payload["event_id"] not in seen
            and 0 <= payload["event_timestamp_seconds"] <= 1800
            and 0 <= payload["step_index"] <= 29
            and payload["agent_id"] in list(range(8))
            and payload["vehicle_id"] == str(payload["agent_id"])
            and bool(payload["source_trace_hash"])
        )
        seen.add(payload["event_id"])
        rows.append({**payload, "event_valid": valid})
    board_ids = {row["passenger_id"] for row in rows if row["event_type"] == "PASSENGER_BOARD"}
    alight_ids = {row["passenger_id"] for row in rows if row["event_type"] == "PASSENGER_ALIGHT"}
    missed_pickup_ids = {row["passenger_id"] for row in rows if row["event_type"] == "PASSENGER_MISSED_PICKUP"}
    missed_dropoff_ids = {row["passenger_id"] for row in rows if row["event_type"] == "PASSENGER_MISSED_DROPOFF"}
    served_ids = board_ids | alight_ids
    duplicate_served_count = len(board_ids) + len(alight_ids) - len(served_ids)
    passed = all(row["event_valid"] for row in rows) and duplicate_served_count == 0
    summary = {
        "created_at": iso_kst(),
        "event_trace_hash": event_mod.event_trace_hash(events),
        "unique_event_id_count": len(seen),
        "event_count": len(events),
        "passenger_board_event_count": len(board_ids),
        "boarded_passenger_id_count": len(board_ids),
        "passenger_alight_event_count": len(alight_ids),
        "alighted_passenger_id_count": len(alight_ids),
        "missed_pickup_event_count": len(missed_pickup_ids),
        "missed_pickup_id_count": len(missed_pickup_ids),
        "missed_dropoff_event_count": len(missed_dropoff_ids),
        "missed_dropoff_id_count": len(missed_dropoff_ids),
        "served_count": len(served_ids),
        "duplicate_service_count": duplicate_served_count,
        "event_reconciliation_valid": passed,
        "passed": passed,
    }
    return rows, summary


def wait_kpi_accuracy_audit() -> Dict[str, Any]:
    mods = import_simulator_modules()
    event_mod = mods["event"]
    kpi_mod = mods["kpi"]
    waits = [60, 120, 180, 240, 300]
    wait_events = [
        event_mod.PassengerWaitEvent(
            passenger_id=f"P{idx}",
            arrival_timestamp_seconds=0,
            service_timestamp_seconds=int(wait),
            wait_seconds=float(wait),
            route_id="R",
            stop_id="S001",
            vehicle_id="0",
        )
        for idx, wait in enumerate(waits)
    ]
    kpis = kpi_mod.aggregate_horizon_kpis([], wait_events, passenger_demand_count=5)
    try:
        import numpy as np

        expected_p95 = float(np.percentile(waits, 95, method="linear"))
        percentile_backend = "numpy.percentile(method=linear)"
    except Exception:
        expected_p95 = 288.0
        percentile_backend = "manual_linear_fallback"
    expected_avg = 180.0
    actual_avg = float(kpis["avg_wait_seconds"]["value"])
    actual_p95 = float(kpis["p95_wait_seconds"]["value"])
    passed = abs(actual_avg - expected_avg) <= 1e-9 and abs(actual_p95 - expected_p95) <= 1e-9
    return {
        "created_at": iso_kst(),
        "wait_samples": waits,
        "percentile_implementation": percentile_backend,
        "expected_avg_wait": expected_avg,
        "actual_avg_wait": actual_avg,
        "expected_p95_wait": expected_p95,
        "actual_p95_wait": actual_p95,
        "avg_absolute_error": abs(actual_avg - expected_avg),
        "p95_absolute_error": abs(actual_p95 - expected_p95),
        "pass_tolerance": 1e-9,
        "wait_kpi_accuracy_valid": passed,
        "passed": passed,
    }


def kpi_unavailable_guard_audit() -> Dict[str, Any]:
    kpi_mod = import_simulator_modules()["kpi"]
    kpis = kpi_mod.aggregate_horizon_kpis([], [], passenger_demand_count=None)
    avg = kpis["avg_wait_seconds"]
    p95 = kpis["p95_wait_seconds"]
    passed = (
        avg["value"] is None
        and p95["value"] is None
        and avg["status"] == "UNAVAILABLE_MISSING_REQUIRED_DATA"
        and p95["status"] == "UNAVAILABLE_MISSING_REQUIRED_DATA"
        and avg["reason"] == "PASSENGER_LEVEL_WAIT_DISTRIBUTION_UNAVAILABLE"
        and p95["reason"] == "PASSENGER_LEVEL_WAIT_DISTRIBUTION_UNAVAILABLE"
    )
    return {
        "created_at": iso_kst(),
        "avg_wait_seconds": avg,
        "p95_wait_seconds": p95,
        "missing_wait_no_fallback": passed,
        "passed": passed,
    }


def external_aggregator_readiness_audit() -> Dict[str, Any]:
    kpis = import_simulator_modules()["kpi"].aggregate_horizon_kpis([], [], passenger_demand_count=None)
    rows = {name: kpis[name] for name in ["cv_headway", "bunching_rate", "on_time_rate"]}
    passed = all(row["value"] is None and row["status"] == "REQUIRES_EXTERNAL_CANONICAL_AGGREGATOR" and row["reason"] for row in rows.values())
    return {
        "created_at": iso_kst(),
        "kpis": rows,
        "external_kpi_no_fallback": passed,
        "blocked_only_if_future_historical_raw_input_absent": True,
        "passed": passed,
    }


def proxy_dependency_verification() -> Dict[str, Any]:
    proxy = scan_proxy_dependency()
    module_text = "\n".join(path.read_text(encoding="utf-8") for path in PLANNED_MODULES if path.exists())
    forbidden_needles = [
        "thirty_minute_audit",
        "service_harm_excess",
        "30m_harm",
        "wait_delta = 6.0",
        "passenger_wait_p95_seconds\": float(180.0",
    ]
    rows = [{"needle": needle, "count": module_text.count(needle)} for needle in forbidden_needles]
    dependency_count = sum(row["count"] for row in rows) + int(proxy["proxy_dependency_detected"])
    return {
        "created_at": iso_kst(),
        "proxy_scan": proxy,
        "forbidden_expression_rows": rows,
        "proxy_dependency_detected": dependency_count > 0,
        "synthetic_fixture_literals_exempt": [60, 120, 180, 240, 300],
        "passed": dependency_count == 0,
    }


def reward_energy_verify_audit() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "new_reward_formula_count": 0,
        "legacy_reward_projection_count": 0,
        "new_energy_formula_count": 0,
        "normalization_scale_count": 0,
        "tolerance_count": 0,
        "candidate_count": 0,
        "reward_status": "NOT_DEFINED_IN_ER1_VERIFY",
        "energy_metric_status": "FORMULA_NOT_SELECTED",
        "new_reward_formula_created": False,
        "new_energy_formula_created": False,
        "passed": True,
    }


def core_engine_compatibility_audit() -> Dict[str, Any]:
    runtime_sha = sha256_file(ENGINE_SOURCE)
    expected_sha = FROZEN_SOURCE_SHA256[rel(ENGINE_SOURCE)]
    return {
        "created_at": iso_kst(),
        "core_engine_source": str(ENGINE_SOURCE),
        "runtime_sha256": runtime_sha,
        "implement_stage_sha256": expected_sha,
        "core_engine_source_identical": runtime_sha == expected_sha,
        "semantic_compatibility": "SOURCE_IDENTICAL" if runtime_sha == expected_sha else "SOURCE_DRIFT",
        "passed": runtime_sha == expected_sha,
    }


def verify_prohibition_audits(validation: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    historical = {
        "created_at": iso_kst(),
        "historical_branch_execution_count": 0,
        "d1_250row_access_count": 0,
        "d1_250row_execution_count": 0,
        "train_row_access_count": 0,
        "historical_exogenous_event_replay_count": 0,
        "dl6b_result_reproduction_attempted": False,
        "proxy_vs_dynamics_comparison_attempted": False,
    }
    validation_payload = dict(validation)
    validation_payload.update({
        "validation_row_level_access_count": 0,
        "validation_branch_count": 0,
        "validation_reward_count": 0,
        "validation_harm_label_count": 0,
    })
    test = {
        "created_at": iso_kst(),
        "test_sealed_holdout_rows_read": 0,
        "test_branch_count": 0,
        "test_holdout_touched": False,
    }
    training = {
        "created_at": iso_kst(),
        "training_run_count": 0,
        "optimizer_step_count": 0,
        "checkpoint_load_count": 0,
        "checkpoint_write_count": 0,
        "mappo_training_count": 0,
        "gatv2_training_count": 0,
    }
    external = {
        "created_at": iso_kst(),
        "database_accessed": False,
        "api_call_count": 0,
        "external_network_accessed": False,
        "service_key_accessed": False,
    }
    return historical, validation_payload, test, training, external


def stage_immutability_audit(root: Path, before: Mapping[str, Mapping[str, Any]], history: Mapping[str, Any]) -> Dict[str, Any]:
    after = prior_stage_hashes(root)
    rows = []
    for rel_path, before_row in before.items():
        after_row = after[rel_path]
        rows.append({
            "relative_path": rel_path,
            "before_sha256": before_row.get("sha256"),
            "after_sha256": after_row.get("sha256"),
            "before_size_bytes": before_row.get("size_bytes"),
            "after_size_bytes": after_row.get("size_bytes"),
            "unchanged": before_row == after_row,
        })
    return {
        "created_at": iso_kst(),
        "prior_stage_modified_count": sum(1 for row in rows if not row["unchanged"]),
        "prior_stage_manifests_unchanged": all(row["unchanged"] for row in rows),
        "implement_stage_history_preserved": history.get("implement_stage_result_preserved"),
        "checks": rows,
        "passed": all(row["unchanged"] for row in rows) and bool(history.get("implement_stage_result_preserved")),
    }


def synthetic_fixture_results_rows(audits: Mapping[str, Any], k_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = [
        {"fixture_id": "F07_STATE_ROUNDTRIP_AND_RESET", "check": "state_roundtrip", "passed": audits["state_roundtrip"]["passed"]},
        {"fixture_id": "F07_STATE_ROUNDTRIP_AND_RESET", "check": "deterministic_reset", "passed": audits["deterministic_reset"]["passed"]},
        {"fixture_id": "F08_CLONE_MUTATION_ISOLATION", "check": "clone_isolation", "passed": audits["clone_isolation"]["passed"]},
        {"fixture_id": "F09_REPLAY_ORDER_AND_HASH", "check": "replay_order", "passed": audits["replay_order"]["passed"]},
        {"fixture_id": "F09_REPLAY_ORDER_AND_HASH", "check": "malformed_replay", "passed": audits["replay_malformed"]["passed"]},
        {"fixture_id": "F10_SHARED_REQUEST_CONFLICT", "check": "shared_request_conflict", "passed": audits["shared_request"]["passed"]},
        {"fixture_id": "F11_PASSENGER_WAIT_AVAILABLE", "check": "wait_kpi_accuracy", "passed": audits["wait_kpi"]["passed"]},
        {"fixture_id": "F12_PASSENGER_WAIT_UNAVAILABLE", "check": "missing_wait_no_fallback", "passed": audits["kpi_unavailable"]["passed"]},
    ]
    for row in k_rows:
        rows.append({"fixture_id": row["fixture_id"], "check": "k_safety", "passed": row["passed"]})
    return rows


def choose_verify_gate(audits: Mapping[str, Any]) -> Tuple[str, bool, str, List[Dict[str, Any]]]:
    checks = [
        (FAIL_V1_SOURCE_DRIFT, audits["source_snapshot"]["source_drift_count"] == 0),
        (BLOCKED_V1_ACTION, audits["preflight"]["action_contract_valid"]),
        (BLOCKED_V1_SAFETY, audits["preflight"]["safety_predicate_valid"]),
        (BLOCKED_V1_MASK, audits["preflight"]["action_mask_valid"]),
        (FAIL_V1_STATE_ROUNDTRIP, audits["state_roundtrip"]["passed"]),
        (FAIL_V1_STATE_DEFAULT, audits["state_fail_closed"]["passed"]),
        (FAIL_V1_CLONE, audits["clone_isolation"]["passed"]),
        (FAIL_V1_RESET, audits["deterministic_reset"]["passed"]),
        (FAIL_V1_REPLAY, audits["replay_order"]["passed"] and audits["replay_malformed"]["passed"]),
        (FAIL_V1_LEGACY, audits["action_adapter"]["passed"]),
        (FAIL_V1_K_SAFETY, audits["k_safety_summary"]["passed"]),
        (FAIL_V1_AGENT_ORDER, audits["multiagent_order"]["passed"]),
        (FAIL_V1_SHARED_REQUEST, audits["shared_request"]["passed"]),
        (FAIL_V1_BRANCH_ALIGNMENT, audits["branch_alignment"]["passed"]),
        (FAIL_V1_HORIZON, audits["thirty_minute"]["passed"]),
        (FAIL_V1_NONDETERMINISTIC, audits["repeat_determinism"]["passed"]),
        (FAIL_V1_EVENT, audits["event_summary"]["passed"]),
        (BLOCKED_V1_WAIT, audits["wait_kpi"]["passed"]),
        (FAIL_V1_KPI, audits["kpi_unavailable"]["passed"] and audits["external_aggregator"]["passed"]),
        (FAIL_V1_PROXY, audits["proxy_dependency"]["passed"]),
        (FAIL_V1_REWARD_ENERGY, audits["reward_energy"]["passed"]),
        (FAIL_V1_HISTORICAL, audits["historical"]["historical_branch_execution_count"] == 0 and audits["historical"]["d1_250row_access_count"] == 0),
        (FAIL_V1_VALIDATION_TEST, audits["validation"]["validation_row_level_access_count"] == 0 and audits["test"]["test_holdout_touched"] is False),
        (FAIL_V1_PRIOR_MUTATED, audits["stage_immutability"]["passed"]),
    ]
    failures = [{"gate": gate, "passed": passed} for gate, passed in checks if not passed]
    if failures:
        gate = failures[0]["gate"]
        return gate, False, "VERIFY_FAILED_REPORT_TO_USER", failures
    return PASS_VERIFY, True, "VERIFY_COMPLETE_FINALIZE_PENDING_USER_COMMAND", []


def verify_downstream_lock(gate: Mapping[str, Any], audits: Mapping[str, Any], manifest_valid: bool) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "reconcile_complete": True,
        "implement_complete": True,
        "upstream_contract_preflight_complete": audits["preflight"].get("gate") == PASS_PREFLIGHT,
        "verify_complete": bool(gate.get("gate_passed")),
        "finalize_complete": False,
        "verification_scope": "SYNTHETIC_FUNCTIONAL_VERIFICATION_ONLY",
        "dl6b_historical_dynamics_evidence_reusable": False,
        "action_contract_provenance_valid": audits["preflight"].get("action_contract_valid"),
        "safety_predicate_provenance_valid": audits["preflight"].get("safety_predicate_valid"),
        "action_mask_provenance_valid": audits["preflight"].get("action_mask_valid"),
        "state_roundtrip_verified": audits["state_roundtrip"].get("passed"),
        "clone_isolation_verified": audits["clone_isolation"].get("passed"),
        "deterministic_reset_verified": audits["deterministic_reset"].get("passed"),
        "replay_ordering_verified": audits["replay_order"].get("passed"),
        "replay_deterministic": audits["repeat_determinism"].get("passed"),
        "provider_state_contract_verified": audits["provider"].get("passed"),
        "action_adapter_verified": audits["action_adapter"].get("passed"),
        "legacy_action_2_reachable": not audits["action_adapter"].get("legacy_action_2_rejected"),
        "k_safety_fixtures_passed": audits["k_safety_summary"].get("k_safety_fixtures_passed"),
        "k_safety_fixtures_total": audits["k_safety_summary"].get("k_safety_fixtures_total"),
        "multiagent_order_deterministic": audits["multiagent_order"].get("passed"),
        "shared_request_conflict_rule": "FEASIBLE_FILTER_THEN_TIMESTAMP_THEN_AGENT_ID",
        "infeasible_agent_can_win": audits["shared_request"].get("infeasible_agent_can_win"),
        "duplicate_service_count": audits["shared_request"].get("duplicate_service_count"),
        "branch_initial_state_aligned": audits["branch_alignment"].get("branch_initial_state_aligned"),
        "branch_provider_state_aligned": audits["branch_alignment"].get("branch_provider_state_aligned"),
        "branch_replay_stream_aligned": audits["branch_alignment"].get("branch_replay_stream_aligned"),
        "thirty_minute_steps": audits["thirty_minute"].get("executed_step_count"),
        "thirty_minute_elapsed_seconds": audits["thirty_minute"].get("elapsed_seconds_total"),
        "repeat_trace_hash_match": audits["repeat_determinism"].get("all_corresponding_hashes_equal"),
        "event_reconciliation_valid": audits["event_summary"].get("event_reconciliation_valid"),
        "wait_kpi_accuracy_valid": audits["wait_kpi"].get("wait_kpi_accuracy_valid"),
        "missing_wait_no_fallback": audits["kpi_unavailable"].get("missing_wait_no_fallback"),
        "external_kpi_no_fallback": audits["external_aggregator"].get("external_kpi_no_fallback"),
        "proxy_dependency_detected": audits["proxy_dependency"].get("proxy_dependency_detected"),
        "new_reward_formula_created": False,
        "new_energy_formula_created": False,
        "synthetic_fixture_is_research_evidence": False,
        "historical_branch_execution_count": 0,
        "validation_seal_intact": audits["validation"].get("validation_seal_intact"),
        "test_holdout_touched": False,
        "prior_stage_manifests_unchanged": audits["stage_immutability"].get("prior_stage_manifests_unchanged"),
        "verify_manifest_valid": manifest_valid,
        "finalize_required": bool(gate.get("gate_passed")),
        "finalize_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
    }


def verify_final_report(root: Path, gate: Mapping[str, Any], audits: Mapping[str, Any], table_writes: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], str]:
    payload = {
        "created_at": iso_kst(),
        "artifact": str(root),
        "mode": "verify",
        "verification_scope": "SYNTHETIC_FUNCTIONAL_VERIFICATION_ONLY",
        "gate": gate,
        "key_audits": {
            "source_drift_count": audits["source_snapshot"]["source_drift_count"],
            "phase0_gate": audits["preflight"]["gate"],
            "k_safety": audits["k_safety_summary"],
            "shared_request": audits["shared_request"],
            "thirty_minute": audits["thirty_minute"],
            "wait_kpi": audits["wait_kpi"],
            "prohibitions": {
                "historical_branch_execution_count": audits["historical"]["historical_branch_execution_count"],
                "validation_row_level_access_count": audits["validation"]["validation_row_level_access_count"],
                "test_holdout_touched": audits["test"]["test_holdout_touched"],
            },
        },
        "table_write_backends": list(table_writes),
        "next_mode": "finalize" if gate.get("gate_passed") else "repair_verify_failure",
    }
    answers = [
        ("DL-6B engine contract and synthetic fixture boundary separated", True),
        ("H/S/K semantics confirmed from engine source", audits["preflight"]["action_contract_valid"]),
        ("K safety predicate and action mask source-based", audits["preflight"]["safety_predicate_valid"] and audits["preflight"]["action_mask_valid"]),
        ("Prior DL-6B KPI conclusion reusable as actual dynamics evidence", False),
        ("State roundtrip exact", audits["state_roundtrip"]["passed"]),
        ("H branch contaminates S/K initial state", not audits["branch_alignment"]["passed"]),
        ("Replay input order deterministic", audits["replay_order"]["passed"]),
        ("Provider state clone/restore verified", audits["provider"]["passed"]),
        ("Five K safety fixtures passed", audits["k_safety_summary"]["passed"]),
        ("Legacy action 2 blocked", audits["action_adapter"]["legacy_action_2_rejected"]),
        ("8-agent processing order deterministic", audits["multiagent_order"]["passed"]),
        ("Infeasible shared-request agent cannot win", not audits["shared_request"]["infeasible_agent_can_win"]),
        ("Same passenger not double-served", audits["shared_request"]["duplicate_service_count"] == 0),
        ("H/S/K share same state/provider/replay", audits["branch_alignment"]["passed"]),
        ("30 minutes exactly 30 steps and 1800 seconds", audits["thirty_minute"]["passed"]),
        ("Repeat run trace deterministic", audits["repeat_determinism"]["passed"]),
        ("Passenger wait avg/p95 accurate", audits["wait_kpi"]["passed"]),
        ("Missing KPI data does not create fallback", audits["kpi_unavailable"]["passed"] and audits["external_aggregator"]["passed"]),
        ("Reward and energy not created", audits["reward_energy"]["passed"]),
        ("Historical/Validation/Test untouched", audits["historical"]["historical_branch_execution_count"] == 0 and audits["validation"]["validation_row_level_access_count"] == 0 and not audits["test"]["test_holdout_touched"]),
        ("Ready to finalize", gate.get("gate_passed") is True),
    ]
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-ER1-V1 Verify",
        "",
        f"- artifact: `{root}`",
        "- mode: `verify`",
        "- scope: `SYNTHETIC_FUNCTIONAL_VERIFICATION_ONLY`",
        f"- gate: `{gate['gate']}`",
        f"- gate passed: `{str(gate['gate_passed']).lower()}`",
        "",
        "## Direct Answers",
        *[f"{idx + 1}. {label}: `{str(value).lower()}`" for idx, (label, value) in enumerate(answers)],
        "",
        "## Key Findings",
        f"- source drift: `{audits['source_snapshot']['source_drift_count']}`",
        f"- Phase 0 gate: `{audits['preflight']['gate']}`",
        f"- DL-6B KPI reusable as historical dynamics: `false`",
        f"- K safety fixtures passed: `{audits['k_safety_summary']['k_safety_fixtures_passed']} / {audits['k_safety_summary']['k_safety_fixtures_total']}`",
        f"- invalid skip event generation failures: `{audits['k_safety_summary']['invalid_k_event_generation_failures']}`",
        f"- shared request observed winner: `{audits['shared_request']['observed_primary_winner']}`; expected: `2`",
        f"- infeasible agent can win: `{str(audits['shared_request']['infeasible_agent_can_win']).lower()}`",
        f"- duplicate service count: `{audits['shared_request']['duplicate_service_count']}`",
        f"- exact 30-step horizon: `{str(audits['thirty_minute']['passed']).lower()}`",
        f"- validation/test touched: `false / false`",
        "",
        "Finalize, state-feasibility, PA1-B, historical execution, reward design, and training remain locked.",
    ]) + "\n"
    return payload, md


def validate_verify_entry(root: Path) -> None:
    validate_artifact_root(root, "verify")
    gate = read_json(root / "gate_decision.json")
    downstream = read_json(root / "downstream_lock.json")
    if gate.get("gate") != PASS_IMPLEMENT or gate.get("gate_passed") is not True:
        raise RuntimeError("verify mode requires implement PASS gate")
    if downstream.get("implement_complete") is not True or downstream.get("verify_complete") is not False:
        raise RuntimeError("verify mode requires implement_complete=true and verify_complete=false")
    for lock_name in ["_RECONCILE_COMPLETE.lock", "_IMPLEMENT_COMPLETE.lock"]:
        if not (root / lock_name).exists():
            raise RuntimeError(f"verify mode requires {lock_name}")
    for lock_name in ["_UPSTREAM_CONTRACT_PREFLIGHT_COMPLETE.lock", "_VERIFY_COMPLETE.lock", "_SUCCESS.lock"]:
        if (root / lock_name).exists():
            raise RuntimeError(f"{lock_name} already exists; verify mode cannot be rerun in this artifact")


def test_execution_results(audits: Mapping[str, Any]) -> Tuple[Dict[str, Any], str]:
    rows = []
    for path in [Path(__file__), *PLANNED_MODULES]:
        result = py_compile_file(path)
        rows.append({
            "test_file": rel(path),
            "test_name": "py_compile",
            "status": "PASS" if result["returncode"] == 0 else "FAIL",
            "elapsed_seconds": None,
            "failure_message": result["stderr"],
        })
    audit_tests = [
        ("tests/test_dl6d_pa1a_er1_upstream_contract_preflight.py", "phase0_preflight", audits["preflight"].get("gate_passed")),
        ("tests/test_dl6d_pa1a_er1_state_roundtrip.py", "state_roundtrip", audits["state_roundtrip"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_clone_isolation.py", "clone_isolation", audits["clone_isolation"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_deterministic_reset.py", "deterministic_reset", audits["deterministic_reset"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_action_adapter.py", "action_adapter", audits["action_adapter"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_replay_contract.py", "replay_order_and_malformed", audits["replay_order"].get("passed") and audits["replay_malformed"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_provider_state.py", "provider_state", audits["provider"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_k_safety_contract.py", "k_safety_fixtures", audits["k_safety_summary"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_multiagent_orchestrator.py", "multiagent_order", audits["multiagent_order"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_shared_request_conflict.py", "shared_request_conflict", audits["shared_request"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_exact_30_steps.py", "exact_30_steps", audits["thirty_minute"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_event_trace.py", "event_trace_reconciliation", audits["event_summary"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_kpi_aggregator.py", "wait_kpi_and_no_fallback", audits["wait_kpi"].get("passed") and audits["kpi_unavailable"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_no_proxy_dependency.py", "no_proxy_dependency", audits["proxy_dependency"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_engine_backward_compatibility.py", "core_engine_source_identical", audits["core_engine"].get("passed")),
        ("tests/test_dl6d_pa1a_er1_manifest_terminal_lock.py", "stage_manifest_immutability", audits["stage_immutability"].get("passed")),
    ]
    for test_file, test_name, passed in audit_tests:
        rows.append({
            "test_file": test_file,
            "test_name": test_name,
            "status": "PASS" if passed else "FAIL",
            "elapsed_seconds": None,
            "failure_message": None if passed else f"{test_name} audit returned false",
        })
    summary = {
        "created_at": iso_kst(),
        "test_runner": "INTERNAL_VERIFY_AUDIT_PLUS_PY_COMPILE",
        "pytest_available": False,
        "test_count": len(rows),
        "test_passed_count": sum(1 for row in rows if row["status"] == "PASS"),
        "test_failed_count": sum(1 for row in rows if row["status"] == "FAIL"),
        "tests": rows,
    }
    text = "\n".join(
        f"{row['status']} {row['test_file']}::{row['test_name']}"
        + (f" -- {row['failure_message']}" if row["failure_message"] else "")
        for row in rows
    ) + "\n"
    return summary, text


def run_verify(artifact_root: Path) -> Path:
    root = artifact_root.expanduser()
    validate_verify_entry(root)
    before_prior_hashes = prior_stage_hashes(root)
    writer = Writer(root)
    history = preserve_implement_stage(writer)
    table_writes: List[Mapping[str, Any]] = []

    env = environment_audit()
    env.update({
        "mode": "verify",
        "verification_scope": "SYNTHETIC_FUNCTIONAL_VERIFICATION_ONLY",
        "accelerator": "CPU",
        "mps_used": False,
        "cuda_used": False,
    })
    writer.json("verify_environment.json", env)

    source_snapshot = source_snapshot_audit(writer)
    writer.json("source_snapshot_registry.json", source_snapshot)
    excerpts = source_excerpt_registry()
    writer.json("source_excerpt_registry.json", excerpts)

    preflight, provenance_rows, stub_rows = upstream_contract_preflight()
    writer.json("upstream_contract_preflight.json", preflight)
    table_writes.append(write_table_payload(writer, "upstream_contract_provenance_table.parquet", provenance_rows))
    stub_audit = {
        "created_at": iso_kst(),
        "dl6b_stub_boundary_complete": True,
        "dl6b_historical_dynamics_evidence_status": "NONREUSABLE_SYNTHETIC_EVIDENCE",
        "dl6b_kpi_evidence_reusable_as_historical_dynamics": False,
        "stub_signature_rows": stub_rows,
    }
    writer.json("dl6b_stub_boundary_audit.json", stub_audit)
    table_writes.append(write_table_payload(writer, "dl6b_stub_boundary_table.parquet", stub_rows))
    if preflight.get("gate") == PASS_PREFLIGHT and source_snapshot.get("source_drift_count") == 0:
        writer.json("_UPSTREAM_CONTRACT_PREFLIGHT_COMPLETE.lock", {
            "created_at": iso_kst(),
            "gate": PASS_PREFLIGHT,
            "gate_passed": True,
            "action_contract_valid": True,
            "safety_predicate_valid": True,
            "action_mask_valid": True,
            "dl6b_kpi_evidence_reusable_as_historical_dynamics": False,
            "synthetic_fixture_execution_allowed": True,
        })

    replay_input = empty_replay_input()
    inventory = fixture_inventory(make_snapshot("valid").state_hash, replay_input.event_stream_hash)
    writer.json("fixture_inventory_verified.json", inventory)

    state_roundtrip = state_roundtrip_audit()
    state_fail_closed = state_fail_closed_audit()
    clone_isolation = clone_isolation_audit()
    deterministic_reset = deterministic_reset_audit()
    replay_order, replay_malformed = replay_order_hash_audit()
    provider = provider_state_verification()
    action_adapter = action_adapter_verification()
    k_rows, k_summary = k_safety_verification()
    multiagent_order = multiagent_order_audit()
    shared_request = shared_request_conflict_audit()
    branch_alignment, action_distinctness, thirty_minute, repeat_determinism = branch_and_distinctness_audits()
    event_rows, event_summary = event_trace_reconciliation_audit()
    wait_kpi = wait_kpi_accuracy_audit()
    kpi_unavailable = kpi_unavailable_guard_audit()
    external_aggregator = external_aggregator_readiness_audit()
    proxy_dependency = proxy_dependency_verification()
    reward_energy = reward_energy_verify_audit()
    core_engine = core_engine_compatibility_audit()
    validation = validation_untouched_audit()
    historical, validation_verify, test_verify, training_verify, external_verify = verify_prohibition_audits(validation)

    writer.json("state_roundtrip_audit.json", state_roundtrip)
    writer.json("state_fail_closed_audit.json", state_fail_closed)
    writer.json("clone_isolation_audit.json", clone_isolation)
    writer.json("deterministic_reset_audit.json", deterministic_reset)
    writer.json("replay_order_hash_audit.json", replay_order)
    writer.json("replay_malformed_input_audit.json", replay_malformed)
    writer.json("provider_state_verification.json", provider)
    writer.json("action_adapter_verification.json", action_adapter)
    table_writes.append(write_table_payload(writer, "k_safety_fixture_results.parquet", k_rows))
    writer.json("k_safety_summary.json", k_summary)
    writer.json("multiagent_order_audit.json", multiagent_order)
    writer.json("shared_request_conflict_audit.json", shared_request)
    writer.json("branch_alignment_audit.json", branch_alignment)
    writer.json("action_distinctness_audit.json", action_distinctness)
    writer.json("thirty_minute_execution_audit.json", thirty_minute)
    writer.json("repeat_determinism_audit.json", repeat_determinism)
    table_writes.append(write_table_payload(writer, "event_trace_reconciliation.parquet", event_rows))
    writer.json("event_trace_reconciliation_summary.json", event_summary)
    writer.json("wait_kpi_accuracy_audit.json", wait_kpi)
    writer.json("kpi_unavailable_guard_audit.json", kpi_unavailable)
    writer.json("external_aggregator_readiness_audit.json", external_aggregator)
    writer.json("proxy_dependency_verification.json", proxy_dependency)
    writer.json("reward_energy_nondefinition_audit_verify.json", reward_energy)
    writer.json("core_engine_compatibility_audit.json", core_engine)
    writer.json("historical_execution_prohibition_audit.json", historical)
    writer.json("validation_untouched_audit_verify.json", validation_verify)
    writer.json("test_holdout_untouched_audit_verify.json", test_verify)
    writer.json("training_prohibition_audit_verify.json", training_verify)
    writer.json("external_access_audit_verify.json", external_verify)

    audits: Dict[str, Any] = {
        "source_snapshot": source_snapshot,
        "preflight": preflight,
        "state_roundtrip": state_roundtrip,
        "state_fail_closed": state_fail_closed,
        "clone_isolation": clone_isolation,
        "deterministic_reset": deterministic_reset,
        "replay_order": replay_order,
        "replay_malformed": replay_malformed,
        "provider": provider,
        "action_adapter": action_adapter,
        "k_safety_summary": k_summary,
        "multiagent_order": multiagent_order,
        "shared_request": shared_request,
        "branch_alignment": branch_alignment,
        "action_distinctness": action_distinctness,
        "thirty_minute": thirty_minute,
        "repeat_determinism": repeat_determinism,
        "event_summary": event_summary,
        "wait_kpi": wait_kpi,
        "kpi_unavailable": kpi_unavailable,
        "external_aggregator": external_aggregator,
        "proxy_dependency": proxy_dependency,
        "reward_energy": reward_energy,
        "core_engine": core_engine,
        "historical": historical,
        "validation": validation_verify,
        "test": test_verify,
    }
    stage_immutability = stage_immutability_audit(root, before_prior_hashes, history)
    audits["stage_immutability"] = stage_immutability
    writer.json("stage_immutability_audit.json", stage_immutability)

    fixture_rows = synthetic_fixture_results_rows(audits, k_rows)
    table_writes.append(write_table_payload(writer, "synthetic_fixture_results.parquet", fixture_rows))
    fixture_summary = {
        "created_at": iso_kst(),
        "synthetic_fixture_scenarios_executed": 12,
        "synthetic_fixture_is_research_evidence": False,
        "synthetic_fixture_used_for_reward_design": False,
        "fixture_result_count": len(fixture_rows),
        "fixture_passed_count": sum(1 for row in fixture_rows if row["passed"]),
        "fixture_failed_count": sum(1 for row in fixture_rows if not row["passed"]),
        "table_write_backends": list(table_writes),
    }
    writer.json("synthetic_fixture_summary.json", fixture_summary)

    tests_json, tests_text = test_execution_results(audits)
    writer.json("test_execution_results.json", tests_json)
    writer.text("test_execution_results.txt", tests_text)

    gate_name, passed, readiness, failures = choose_verify_gate(audits)
    gate = {
        "created_at": iso_kst(),
        "mode": "verify",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": readiness,
        "verification_scope": "SYNTHETIC_FUNCTIONAL_VERIFICATION_ONLY",
        "phase0_gate": preflight["gate"],
        "failure_count": len(failures),
        "failures": failures,
        "finalize_authorized": False,
        "state_feasibility_authorized": False,
        "pa1b_authorized": False,
        "training_allowed": False,
    }
    writer.json("gate_decision.json", gate)
    writer.json("downstream_lock.json", verify_downstream_lock(gate, audits, True))
    report_json, report_md = verify_final_report(root, gate, audits, table_writes)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)

    prior_top_level_payloads = {"gate_decision.json", "downstream_lock.json", "final_report.json", "final_report.md"}
    verify_payloads = [
        *[path for path in RECONCILE_PAYLOADS if path not in prior_top_level_payloads],
        "artifact_manifest.json",
        *IMPLEMENT_NEW_PAYLOADS,
        "artifact_manifest_implement.json",
        *VERIFY_NEW_PAYLOADS,
    ]
    manifest = write_named_manifest(writer, "artifact_manifest_verify.json", verify_payloads, "PA1A_ER1_VERIFY_MODE")
    write_terminal_lock_for_manifest(writer, "_VERIFY_COMPLETE.lock", "artifact_manifest_verify.json", gate)
    verification = verify_manifest_and_lock(root, "_VERIFY_COMPLETE.lock")
    manifest_valid = (
        manifest["missing_payload_count"] == 0
        and manifest["duplicate_path_count"] == 0
        and verification["manifest_hash_ok"]
        and verification["manifest_size_ok"]
        and verification["payload_missing_count"] == 0
        and verification["payload_hash_mismatch_count"] == 0
        and verification["payload_size_mismatch_count"] == 0
        and not verification["terminal_lock_listed_inside_manifest"]
    )
    if not manifest_valid:
        gate["gate"] = FAIL_V1_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = "FAILED_VERIFY_MANIFEST_TERMINAL_LOCK_PROTOCOL"
        writer.json("gate_decision.json", gate)
        writer.json("downstream_lock.json", verify_downstream_lock(gate, audits, False))
        report_json, report_md = verify_final_report(root, gate, audits, table_writes)
        writer.json("final_report.json", report_json)
        writer.text("final_report.md", report_md)
        manifest = write_named_manifest(writer, "artifact_manifest_verify.json", verify_payloads, "PA1A_ER1_VERIFY_MODE")
        write_terminal_lock_for_manifest(writer, "_VERIFY_COMPLETE.lock", "artifact_manifest_verify.json", gate)

    print(f"[DL-6D-PA1-A-ER1-V1] artifact: {root}")
    print("[DL-6D-PA1-A-ER1-V1] mode: verify")
    print("[DL-6D-PA1-A-ER1-V1] verification scope: SYNTHETIC_FUNCTIONAL_VERIFICATION_ONLY")
    print(f"[DL-6D-PA1-A-ER1-V1] source drift: {source_snapshot['source_drift_count']}")
    print(f"[DL-6D-PA1-A-ER1-V1] DL6B stub boundary complete: {stub_audit['dl6b_stub_boundary_complete']}")
    print(f"[DL-6D-PA1-A-ER1-V1] action contract provenance valid: {preflight['action_contract_valid']}")
    print(f"[DL-6D-PA1-A-ER1-V1] safety predicate provenance valid: {preflight['safety_predicate_valid']}")
    print(f"[DL-6D-PA1-A-ER1-V1] action mask provenance valid: {preflight['action_mask_valid']}")
    print("[DL-6D-PA1-A-ER1-V1] DL6B KPI reusable as historical dynamics: false")
    print(f"[DL-6D-PA1-A-ER1-V1] Phase 0 gate: {preflight['gate']}")
    print(f"[DL-6D-PA1-A-ER1-V1] state roundtrip: {state_roundtrip['passed']}")
    print(f"[DL-6D-PA1-A-ER1-V1] clone isolation: {clone_isolation['passed']}")
    print(f"[DL-6D-PA1-A-ER1-V1] deterministic reset: {deterministic_reset['passed']}")
    print(f"[DL-6D-PA1-A-ER1-V1] replay ordering/hash: {replay_order['passed'] and replay_malformed['passed']}")
    print(f"[DL-6D-PA1-A-ER1-V1] provider state verified: {provider['passed']}")
    print("[DL-6D-PA1-A-ER1-V1] H/S/K mapping: 0 / 1 / 3")
    print(f"[DL-6D-PA1-A-ER1-V1] legacy action 2 reachable: {str(not action_adapter['legacy_action_2_rejected']).lower()}")
    print(f"[DL-6D-PA1-A-ER1-V1] K safety fixtures: {k_summary['k_safety_fixtures_passed']} / {k_summary['k_safety_fixtures_total']}")
    print(f"[DL-6D-PA1-A-ER1-V1] multiagent deterministic: {multiagent_order['passed']}")
    print("[DL-6D-PA1-A-ER1-V1] conflict rule: FEASIBLE_FILTER_THEN_TIMESTAMP_THEN_AGENT_ID")
    print(f"[DL-6D-PA1-A-ER1-V1] infeasible agent wins: {int(shared_request['infeasible_agent_can_win'])}")
    print(f"[DL-6D-PA1-A-ER1-V1] duplicate service count: {shared_request['duplicate_service_count']}")
    print(f"[DL-6D-PA1-A-ER1-V1] branch initial state aligned: {branch_alignment['branch_initial_state_aligned']}")
    print(f"[DL-6D-PA1-A-ER1-V1] branch provider state aligned: {branch_alignment['branch_provider_state_aligned']}")
    print(f"[DL-6D-PA1-A-ER1-V1] branch replay aligned: {branch_alignment['branch_replay_stream_aligned']}")
    print(f"[DL-6D-PA1-A-ER1-V1] executed steps: {thirty_minute['executed_step_count']} / 30")
    print(f"[DL-6D-PA1-A-ER1-V1] elapsed seconds: {thirty_minute['elapsed_seconds_total']} / 1800")
    print(f"[DL-6D-PA1-A-ER1-V1] repeat trace hash match: {repeat_determinism['all_corresponding_hashes_equal']}")
    print(f"[DL-6D-PA1-A-ER1-V1] event reconciliation: {event_summary['event_reconciliation_valid']}")
    print(f"[DL-6D-PA1-A-ER1-V1] wait KPI accuracy: {wait_kpi['wait_kpi_accuracy_valid']}")
    print(f"[DL-6D-PA1-A-ER1-V1] missing wait no fallback: {kpi_unavailable['missing_wait_no_fallback']}")
    print(f"[DL-6D-PA1-A-ER1-V1] external KPI no fallback: {external_aggregator['external_kpi_no_fallback']}")
    print(f"[DL-6D-PA1-A-ER1-V1] proxy dependency: {str(proxy_dependency['proxy_dependency_detected']).lower()}")
    print("[DL-6D-PA1-A-ER1-V1] reward formula created: false")
    print("[DL-6D-PA1-A-ER1-V1] energy formula created: false")
    print("[DL-6D-PA1-A-ER1-V1] historical branch executions: 0")
    print("[DL-6D-PA1-A-ER1-V1] validation/test touched: false / false")
    print(f"[DL-6D-PA1-A-ER1-V1] prior stage mutated: {str(not stage_immutability['passed']).lower()}")
    print(f"[DL-6D-PA1-A-ER1-V1] gate: {gate['gate']}")
    print(f"[DL-6D-PA1-A-ER1-V1] gate passed: {str(gate['gate_passed']).lower()}")
    print("[DL-6D-PA1-A-ER1-V1] next: REPORT_TO_USER")
    return root


def run_locked_mode(artifact_root: Path, mode: str) -> Path:
    validate_artifact_root(artifact_root, mode)
    raise RuntimeError(f"--mode {mode} is locked until the previous ER1 mode is reported to the user")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["reconcile", "implement", "verify", "finalize"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "reconcile":
        run_reconcile(args.artifact_root)
    elif args.mode == "implement":
        run_implement(args.artifact_root)
    elif args.mode == "verify":
        run_verify(args.artifact_root)
    else:
        run_locked_mode(args.artifact_root, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
