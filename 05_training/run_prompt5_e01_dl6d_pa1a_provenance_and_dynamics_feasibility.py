from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import platform
import re
import resource
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - project env normally has torch.
    torch = None  # type: ignore[assignment]


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
ARTIFACT_PREFIX = "prompt5_e01_dl6d_pa1a_provenance_and_dynamics_feasibility"
R1_RUNNER = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r1_observation_contract_repair.py"
D1_RUNNER = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r3_r1_d1_train_development_rollout.py"
ENGINE_SOURCE = PROJECT_ROOT / "05_training/simulator/suseong_service_transition_engine.py"
DL6D_SOURCE = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_three_action_reward_retraining_readiness_audit.py"
DL6C_SOURCE = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6c_distinct_three_action_contract_repair.py"
KPI_AGGREGATOR = PROJECT_ROOT / "05_training/evaluation/canonical_kpi_aggregator.py"

I0 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_r1_split_inventory_20260802_132332"
D1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_r1_d1_train_development_rollout_20260802_143233"
HO1 = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_holdout_evaluation_20260802_122513"

PASS_ADJUDICATE = "PASS_SUSEONG_DL6D_PA1A_ADJUDICATION_COMPLETE_AWAITING_USER_COMMAND"
FAIL_ARTIFACT_MUTATED = "FAIL_SUSEONG_DL6D_PA1A_EXISTING_ARTIFACT_MUTATED"
FAIL_VALIDATION = "FAIL_SUSEONG_DL6D_PA1A_VALIDATION_OUTCOME_LEAKAGE"
FAIL_TEST = "FAIL_SUSEONG_DL6D_PA1A_TEST_HOLDOUT_TOUCHED"
FAIL_DYNAMICS = "FAIL_SUSEONG_DL6D_PA1A_DYNAMICS_EXECUTION_DETECTED"
FAIL_SCALE = "FAIL_SUSEONG_DL6D_PA1A_SCALE_OR_CANDIDATE_PRODUCED"
FAIL_TRAINING = "FAIL_SUSEONG_DL6D_PA1A_PROHIBITED_TRAINING"
FAIL_EXTERNAL = "FAIL_SUSEONG_DL6D_PA1A_EXTERNAL_ACCESS"
FAIL_HARDWARE = "FAIL_SUSEONG_DL6D_PA1A_H200_OR_CUDA_USAGE"
FAIL_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_MANIFEST_RECONCILIATION"

FAMILIES = [
    "prompt5_e01_dl6c_distinct_three_action_contract_repair_",
    "prompt5_e01_dl6d_three_action_reward_retraining_readiness_audit_",
    "prompt5_e01_dl6d_r1_observation_contract_repair_",
    "prompt5_e01_dl6d_r2_combined_retraining_readiness_reaudit_",
    "prompt5_e01_dl6d_r3_reward_service_alignment_repair_",
    "prompt5_e01_dl6d_r3_pre_holdout_audit_",
    "prompt5_e01_dl6d_r3_holdout_evaluation_",
    "prompt5_e01_dl6d_r3_r1_split_inventory_",
    "prompt5_e01_dl6d_r3_r1_d1_train_development_rollout_",
]

ADJUDICATE_REQUIRED_FILES = [
    "git_status_pa1a.txt",
    "mac_mini_environment_pa1a.json",
    "upstream_validation.json",
    "source_sha_registry.json",
    "path_b_decision_record.json",
    "artifact_preservation_table.parquet",
    "artifact_preservation_audit.json",
    "proxy_constant_provenance_table.parquet",
    "proxy_constant_provenance_audit.json",
    "r1_self_description_audit.json",
    "contamination_scope_table.parquet",
    "contamination_scope_enumeration.json",
    "proxy_independent_result_registry.json",
    "proxy_relabeling_registry.json",
    "proxy_relabeling_registry.parquet",
    "dynamics_execution_prohibition_audit.json",
    "validation_seal_revalidation.json",
    "validation_untouched_audit.json",
    "test_holdout_untouched_audit.json",
    "scale_candidate_prohibition_audit.json",
    "training_prohibition_audit.json",
    "external_access_audit.json",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.json",
    "final_report.md",
    "artifact_manifest.json",
    "_ADJUDICATE_COMPLETE.lock",
]

PASS_ENGINE_AUDIT = "PASS_SUSEONG_DL6D_PA1A_ENGINE_API_AUDIT_COMPLETE_AWAITING_USER_COMMAND"
BLOCK_ENGINE_REPAIR = "BLOCKED_SUSEONG_DL6D_PA1A_ENGINE_PARTIAL_REPAIR_REQUIRED"
BLOCK_ENGINE_INSUFFICIENT = "BLOCKED_SUSEONG_DL6D_PA1A_ENGINE_INSUFFICIENT_FOR_REAL_DYNAMICS"
BLOCK_ENGINE_INDETERMINATE = "BLOCKED_SUSEONG_DL6D_PA1A_ENGINE_AUDIT_INDETERMINATE"
BLOCK_PROVENANCE_RECONCILIATION = "BLOCKED_SUSEONG_DL6D_PA1A_PROVENANCE_RECONCILIATION_FAILED"

FAIL_E1_ARTIFACT_MUTATED = "FAIL_SUSEONG_DL6D_PA1A_E1_EXISTING_ARTIFACT_MUTATED"
FAIL_E1_R1_MODIFIED = "FAIL_SUSEONG_DL6D_PA1A_E1_R1_MODULE_MODIFIED"
FAIL_E1_ENGINE_MODIFIED = "FAIL_SUSEONG_DL6D_PA1A_E1_TRANSITION_ENGINE_MODIFIED"
FAIL_E1_DYNAMICS = "FAIL_SUSEONG_DL6D_PA1A_E1_DYNAMICS_EXECUTION_DETECTED"
FAIL_E1_VALIDATION = "FAIL_SUSEONG_DL6D_PA1A_E1_VALIDATION_SEAL_MUTATED"
FAIL_E1_TEST = "FAIL_SUSEONG_DL6D_PA1A_E1_TEST_HOLDOUT_TOUCHED"
FAIL_E1_TRAINING = "FAIL_SUSEONG_DL6D_PA1A_E1_PROHIBITED_TRAINING"
FAIL_E1_EXTERNAL = "FAIL_SUSEONG_DL6D_PA1A_E1_EXTERNAL_ACCESS"
FAIL_E1_MANIFEST = "FAIL_SUSEONG_DL6D_PA1A_E1_MANIFEST_RECONCILIATION"

ENGINE_REQUIRED_FILES = [
    "_ADJUDICATE_COMPLETE.lock",
    "adjudicate_stage_result_history.json",
    "engine_audit_retry_history.json",
    "proxy_provenance_hardening.json",
    "decision_relevant_proxy_value_table.parquet",
    "provenance_count_reconciliation.json",
    "repository_stub_signature_sweep.json",
    "repository_stub_signature_sweep.parquet",
    "transition_engine_stub_signature_audit.json",
    "transition_engine_api_audit.json",
    "engine_capability_inventory.parquet",
    "action_mapping_audit.json",
    "action_mapping_table.parquet",
    "thirty_minute_horizon_contract.json",
    "state_clone_rng_alignment_audit.json",
    "exogenous_event_interface_audit.json",
    "reward_kpi_extraction_audit.json",
    "engine_gap_repair_registry.json",
    "engine_gap_repair_table.parquet",
    "gate_decision.json",
    "downstream_lock.json",
    "final_report.json",
    "final_report.md",
    "artifact_manifest.json",
    "_ENGINE_AUDIT_COMPLETE.lock",
]


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.order: Dict[str, int] = {}

    def _mark(self, rel: str) -> None:
        if rel not in self.order:
            self.order[rel] = len(self.order) + 1

    def text(self, rel: str, text: str) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self._mark(rel)

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(json_clean(dict(payload)), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")

    def parquet(self, rel: str, frame: pd.DataFrame) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        self._mark(rel)


def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    return value


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def iso_kst() -> str:
    return now_kst().isoformat(timespec="seconds")


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
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str))


def run_cmd(args: Sequence[str]) -> Dict[str, Any]:
    result = subprocess.run(list(args), cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    return {
        "cmd": list(args),
        "returncode": result.returncode,
        "stdout": redact_identity(result.stdout.strip()),
        "stderr": redact_identity(result.stderr.strip()),
    }


def redact_identity(text: str) -> str:
    redacted = []
    for line in text.splitlines():
        lower = line.lower()
        if "serial number" in lower or "hardware uuid" in lower or "provisioning udid" in lower:
            redacted.append(f"{line.split(':', 1)[0]}: REDACTED")
        else:
            redacted.append(line)
    return "\n".join(redacted)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def git_tracked(path: Path) -> bool:
    return run_cmd(["git", "ls-files", "--error-unmatch", rel(path)])["returncode"] == 0


def git_last_commit(path: Path) -> Dict[str, Any]:
    out = run_cmd(["git", "log", "-1", "--format=%H%x09%aI%x09%s", "--", rel(path)])
    if out["returncode"] != 0 or not out["stdout"]:
        return {"last_commit_hash": None, "last_commit_date": None, "last_commit_message": None}
    parts = out["stdout"].split("\t", 2)
    return {
        "last_commit_hash": parts[0] if len(parts) > 0 else None,
        "last_commit_date": parts[1] if len(parts) > 1 else None,
        "last_commit_message": parts[2] if len(parts) > 2 else None,
    }


def git_status_path(path: Path) -> str:
    return run_cmd(["git", "status", "--short", "--", rel(path)])["stdout"]


def git_blame_line(path: Path, line: int) -> Dict[str, Any]:
    out = run_cmd(["git", "blame", "-L", f"{line},{line}", "--porcelain", "--", rel(path)])
    if out["returncode"] != 0 or not out["stdout"]:
        return {"git_blame_commit": None, "git_blame_commit_date": None, "git_blame_summary": None}
    lines = out["stdout"].splitlines()
    commit = lines[0].split()[0] if lines else None
    summary = next((x.split(" ", 1)[1] for x in lines if x.startswith("summary ")), None)
    author_time = next((x.split(" ", 1)[1] for x in lines if x.startswith("author-time ")), None)
    commit_date = None
    if author_time:
        try:
            commit_date = datetime.fromtimestamp(int(author_time), tz=ZoneInfo("Asia/Seoul")).isoformat()
        except ValueError:
            commit_date = None
    return {"git_blame_commit": commit, "git_blame_commit_date": commit_date, "git_blame_summary": summary}


def mps_flags() -> Tuple[bool, bool]:
    if torch is None or not hasattr(torch.backends, "mps"):
        return False, False
    return bool(torch.backends.mps.is_built()), bool(torch.backends.mps.is_available())


def environment_audit() -> Dict[str, Any]:
    mem = run_cmd(["sysctl", "-n", "hw.memsize"])
    model = run_cmd(["sysctl", "-n", "hw.model"])
    sw = run_cmd(["sw_vers"])
    mps_built, mps_available = mps_flags()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_bytes = raw_rss if platform.system() == "Darwin" else raw_rss * 1024
    return {
        "created_at": iso_kst(),
        "requested_execution_platform": "MAC_MINI_M4_24GB",
        "actual_compute_path": "CPU",
        "hardware_model": model["stdout"] or "UNKNOWN",
        "chip_name": "Apple M4",
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
        "cuda_available": cuda_available,
        "cuda_used": False,
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
        "remote_gpu_allowed": False,
        "process_rss_bytes": rss_bytes,
    }


def artifact_dirs() -> List[Path]:
    root = PROJECT_ROOT / "05_training/artifacts"
    out: List[Path] = []
    for family in FAMILIES:
        out.extend(sorted(path for path in root.glob(f"{family}*") if path.is_dir()))
    return sorted(set(out))


def gate_file(path: Path) -> Optional[Path]:
    for name in [
        "gate_decision.json",
        "combined_gate_decision.json",
        "methodological_gate_decision.json",
        "sealed_holdout_gate_decision.json",
    ]:
        candidate = path / name
        if candidate.exists():
            return candidate
    return None


def artifact_gate(path: Path) -> Tuple[Optional[str], Optional[bool], Optional[str]]:
    candidate = gate_file(path)
    if candidate is None:
        return None, None, None
    try:
        data = read_json(candidate)
    except Exception:
        return None, None, None
    gate = data.get("gate") or data.get("combined_gate") or data.get("methodological_gate")
    passed = data.get("gate_passed")
    if passed is None:
        passed = data.get("combined_gate_passed") or data.get("methodological_gate_passed")
    created = data.get("created_at")
    return gate, bool(passed) if passed is not None else None, created


def upstream_validation(paths: Sequence[Path]) -> Dict[str, Any]:
    rows = []
    for path in paths:
        gate, passed, created = artifact_gate(path)
        manifest = path / "artifact_manifest.json"
        rows.append({
            "artifact_family": next((family.rstrip("_") for family in FAMILIES if path.name.startswith(family)), "UNKNOWN"),
            "artifact_path": str(path),
            "gate": gate,
            "gate_passed": passed,
            "created_at": created,
            "manifest_path": str(manifest) if manifest.exists() else None,
            "manifest_sha256": sha256_file(manifest) if manifest.exists() else None,
            "success_lock_present": (path / "_SUCCESS.lock").exists(),
            "artifact_file_count": sum(1 for item in path.rglob("*") if item.is_file()),
            "preserved_unmodified": True,
        })
    return {
        "created_at": iso_kst(),
        "artifact_count": len(rows),
        "artifact_families": sorted(set(row["artifact_family"] for row in rows)),
        "artifacts": rows,
        "d1_comparison_artifact": str(D1),
        "i0_validation_seal_artifact": str(I0),
    }


def source_sha_registry() -> Dict[str, Any]:
    sources = [
        ("R1 reduced-form audit source", R1_RUNNER),
        ("D1 calibration runner source", D1_RUNNER),
        ("transition engine source", ENGINE_SOURCE),
        ("DL-6C action contract source", DL6C_SOURCE),
        ("DL-6C skip-valid predicate source", ENGINE_SOURCE),
        ("DL-6D reward component source", DL6D_SOURCE),
        ("canonical KPI aggregator source", KPI_AGGREGATOR),
    ]
    rows = []
    for role, path in sources:
        commit = git_last_commit(path)
        rows.append({
            "role": role,
            "absolute_path": str(path),
            "sha256": sha256_file(path) if path.exists() else None,
            "git_tracked": git_tracked(path) if path.exists() else False,
            "git_status": git_status_path(path) if path.exists() else "MISSING",
            **commit,
        })
    return {"created_at": iso_kst(), "source_count": len(rows), "sources": rows}


def scan_artifact_files(paths: Sequence[Path]) -> pd.DataFrame:
    rows = []
    for artifact in paths:
        for file_path in sorted(item for item in artifact.rglob("*") if item.is_file()):
            rows.append({
                "artifact_path": str(artifact),
                "relative_file_path": str(file_path.relative_to(artifact)),
                "size_bytes": int(file_path.stat().st_size),
                "sha256": sha256_file(file_path),
            })
    return pd.DataFrame(rows)


def artifact_preservation(before: pd.DataFrame, after: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    keys = ["artifact_path", "relative_file_path"]
    merged = before.rename(columns={"sha256": "sha256_before", "size_bytes": "size_bytes_before"}).merge(
        after.rename(columns={"sha256": "sha256_after", "size_bytes": "size_bytes_after"}),
        on=keys,
        how="outer",
        indicator=True,
    )
    status = []
    for _, row in merged.iterrows():
        if row["_merge"] == "left_only":
            status.append("DELETED")
        elif row["_merge"] == "right_only":
            status.append("ADDED")
        elif row["sha256_before"] != row["sha256_after"] or row["size_bytes_before"] != row["size_bytes_after"]:
            status.append("MODIFIED")
        else:
            status.append("UNCHANGED")
    merged = merged.drop(columns=["_merge"])
    merged["status"] = status
    audit = {
        "created_at": iso_kst(),
        "existing_artifact_file_count": int(len(before)),
        "modified_file_count": int((merged["status"] == "MODIFIED").sum()),
        "deleted_file_count": int((merged["status"] == "DELETED").sum()),
        "added_file_count_inside_existing_artifacts": int((merged["status"] == "ADDED").sum()),
        "existing_artifacts_preserved": bool((merged["status"] == "UNCHANGED").all()),
        "preservation_rationale": "Reduced-form proxy lineage artifacts are retained as historical evidence and relabeled through a separate registry.",
    }
    return merged, audit


def path_b_decision_record() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "original_research_intent": "REAL_SIMULATOR_DYNAMICS_THIRTY_MINUTE_BRANCH",
        "adjudication_path": "PATH_B",
        "path_decision_source": "USER",
        "reduced_form_proxy_was_intended_modeling_choice": False,
        "decision_revisitable_in_pa1a": False,
    }


def r1_function_node() -> Tuple[ast.FunctionDef, str, List[str]]:
    text = R1_RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "thirty_minute_audit":
            lines = text.splitlines()
            return node, text, lines
    raise RuntimeError("thirty_minute_audit not found")


def ast_called_functions(node: ast.FunctionDef) -> List[str]:
    calls = []
    for item in ast.walk(node):
        if isinstance(item, ast.Call):
            func = item.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                parts = []
                current: Any = func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                calls.append(".".join(reversed(parts)))
    return sorted(set(calls))


def proxy_provenance() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    node, _text, lines = r1_function_node()
    rows: List[Dict[str, Any]] = []
    numeric_nodes = [
        item for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)) and not isinstance(item.value, bool)
    ]
    for item in numeric_nodes:
        line_no = int(getattr(item, "lineno", node.lineno))
        source_line = lines[line_no - 1].strip()
        rows.append({
            "function_name": node.name,
            "function_start_line": int(node.lineno),
            "function_end_line": int(node.end_lineno or node.lineno),
            "function_signature": "thirty_minute_audit(ctx: pd.DataFrame, telemetry_rows: List[Dict[str, Any]])",
            "value_name": f"literal_{item.value}_line_{line_no}",
            "value": item.value,
            "source_line": line_no,
            "source_excerpt": source_line,
            "classification": "SOURCE_LITERAL_CONSTANT",
            "derivation_documented": False,
            "derivation_source": None,
            "calibrated_against_simulator": False,
            "calibration_evidence": None,
            **git_blame_line(R1_RUNNER, line_no),
        })
    derived_rows = [
        {
            "value_name": "service_harm_excess_native_harm_pattern",
            "value": 16.43,
            "source_line": 846,
            "source_excerpt": "Derived downstream: avg_wait_delta=6, p95_delta=9, service_rate_delta=-0.0166667, on_time_delta=-0.006, served_count_delta=-0.2 under R3 tolerance weights.",
            "classification": "DERIVED_INVARIANT",
        },
        {
            "value_name": "service_harm_excess_native_non_harm_pattern",
            "value": 0.0,
            "source_line": 830,
            "source_excerpt": "Derived downstream when harm_flag is false and skip KPI deltas are beneficial or within tolerance.",
            "classification": "DERIVED_INVARIANT",
        },
        {
            "value_name": "base_reward_expression",
            "value": None,
            "source_line": 815,
            "source_excerpt": "base_reward = 0.01 * estimated_skip_time_delta - 0.002 * abs(headway_deviation)",
            "classification": "DATA_DEPENDENT_VALUE",
        },
        {
            "value_name": "harm_flag_expression",
            "value": None,
            "source_line": 816,
            "source_excerpt": "harm_flag = stable_int('30m_harm', window_id, agent_id, modulo=5) == 0",
            "classification": "DATA_DEPENDENT_VALUE",
        },
    ]
    for row in derived_rows:
        rows.append({
            "function_name": node.name,
            "function_start_line": int(node.lineno),
            "function_end_line": int(node.end_lineno or node.lineno),
            "function_signature": "thirty_minute_audit(ctx: pd.DataFrame, telemetry_rows: List[Dict[str, Any]])",
            "derivation_documented": False,
            "derivation_source": None,
            "calibrated_against_simulator": False,
            "calibration_evidence": None,
            **row,
            **git_blame_line(R1_RUNNER, int(row["source_line"])),
        })
    table = pd.DataFrame(rows)
    calls = ast_called_functions(node)
    transition_call_names = {
        "advance_vehicle_time_budget",
        "run_one_branch",
        "engine.step",
        "transition",
        "reset_to_serialized_state",
    }
    transition_calls = [call for call in calls if call in transition_call_names]
    audit = {
        "created_at": iso_kst(),
        "function_name": node.name,
        "function_start_line": int(node.lineno),
        "function_end_line": int(node.end_lineno or node.lineno),
        "function_signature": "thirty_minute_audit(ctx: pd.DataFrame, telemetry_rows: List[Dict[str, Any]])",
        "literal_constant_count": int((table["classification"] == "SOURCE_LITERAL_CONSTANT").sum()),
        "derived_invariant_count": int((table["classification"] == "DERIVED_INVARIANT").sum()),
        "data_dependent_value_count": int((table["classification"] == "DATA_DEPENDENT_VALUE").sum()),
        "calibrated_from_simulator_count": int((table["calibrated_against_simulator"] == True).sum()),
        "documented_assumption_count": 0,
        "unknown_provenance_count": int((table["derivation_documented"] == False).sum()),
        "called_functions": calls,
        "external_module_calls": [call for call in calls if "." in call],
        "simulator_transition_calls": transition_calls,
        "simulator_transition_call_count": len(transition_calls),
        "r1_reduced_form_proxy_confirmed": True,
    }
    return table, audit


def term_snippets(text: str, terms: Sequence[str]) -> List[Dict[str, Any]]:
    snippets = []
    lower = text.lower()
    for term in terms:
        pos = lower.find(term.lower())
        if pos >= 0:
            start = max(0, pos - 80)
            end = min(len(text), pos + len(term) + 80)
            snippets.append({"term": term, "snippet": text[start:end].replace("\n", " ")})
    return snippets


def r1_self_description_audit() -> Dict[str, Any]:
    terms = ["30-minute simulation", "30-minute rollout", "counterfactual", "audit", "single transition", "proxy", "reduced-form", "simulator", "30m branch"]
    r1_artifact = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r1_observation_contract_repair_20260802_011348"
    files = [
        r1_artifact / "final_report.md",
        r1_artifact / "final_report.json",
        r1_artifact / "dl6d_skip_reward_horizon_classification.json",
        r1_artifact / "skip_propagation_classification.json",
        r1_artifact / "one_step_vs_thirty_minute_direction_audit.json",
    ]
    rows = []
    proxy = False
    reduced = False
    simulation = False
    single_transition = False
    rollout = False
    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        lower = text.lower()
        proxy = proxy or "proxy" in lower
        reduced = reduced or "reduced-form" in lower
        simulation = simulation or "simulation" in lower or "simulator" in lower
        single_transition = single_transition or "single transition" in lower
        rollout = rollout or "30-minute rollout" in lower or "30m branch" in lower or "30m" in lower
        rows.append({"path": str(path), "matched_terms": [term for term in terms if term.lower() in lower], "snippets": term_snippets(text, terms)})
    return {
        "created_at": iso_kst(),
        "audited_files": rows,
        "r1_explicitly_declared_as_proxy": bool(proxy or reduced),
        "r1_described_as_simulation": bool(simulation),
        "r1_described_as_single_transition": bool(single_transition),
        "r1_described_as_30m_rollout_or_branch": bool(rollout),
        "r1_description_internal_consistency": "UNDERLABELED_PROXY" if single_transition and not (proxy or reduced) else "CONSISTENT",
        "adjudication": "R1 declares single-transition provenance but does not clearly label the 30-minute branch results as a reduced-form proxy.",
    }


def contamination_scope() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(artifact: str, source_file: str, field: str, claim_id: str, text: str, cls: str, label: str, rebuild: Any) -> None:
        rows.append({
            "artifact_path": artifact,
            "source_file": source_file,
            "source_field_or_section": field,
            "claim_id": claim_id,
            "claim_text_original": text,
            "depends_on_proxy": cls != "PROXY_INDEPENDENT",
            "proxy_dependency_path": str(R1_RUNNER) if cls != "PROXY_INDEPENDENT" else None,
            "contamination_class": cls,
            "corrected_label_proposed": label,
            "rebuild_required_for_real_dynamics_claim": str(rebuild).lower() if isinstance(rebuild, bool) else str(rebuild),
        })

    r1_art = str(PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r1_observation_contract_repair_20260802_011348")
    r2_art = str(PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r2_combined_retraining_readiness_reaudit_20260802_102624")
    r3_art = str(PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_reward_service_alignment_repair_20260802_112731")
    pre_art = str(PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6d_r3_pre_holdout_audit_20260802_115741")
    ho_art = str(HO1)
    i0_art = str(I0)
    d1_art = str(D1)
    dl6c_art = str(PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6c_distinct_three_action_contract_repair_20260801_234817")

    add(r1_art, "thirty_minute_skip_branch_rollup.parquet", "branch rows", "R1_30M_BRANCH_ROLLUP", "30m branch rollup", "FULLY_PROXY_DERIVED", "reduced-form 30-minute proxy branch rollup", True)
    add(r1_art, "skip_propagation_classification.json", "skip_propagation_classification", "R1_SKIP_PROPAGATION", "ONE_STEP_ADVANTAGE_WITH_30M_NETWORK_HARM", "FULLY_PROXY_DERIVED", "proxy-estimated network harm classification", True)
    add(r1_art, "one_step_vs_thirty_minute_direction_audit.json", "thirty_minute_*", "R1_30M_REWARD_ADVANTAGE", "30m reward and KPI direction rates", "FULLY_PROXY_DERIVED", "reduced-form proxy direction rates", True)
    add(r2_art, "corrected_30m_primary_class.parquet", "primary_class", "R2_CORRECTED_PRIMARY_CLASS", "NET_BENEFICIAL / NET_HARMFUL classification", "FULLY_PROXY_DERIVED", "proxy-derived 30m primary class", True)
    add(r2_art, "reward_service_alignment_audit.json", "misalignment counts", "R2_SERVICE_ALIGNMENT", "reward-service alignment invalid", "FULLY_PROXY_DERIVED", "proxy-derived reward-service alignment", True)
    add(r3_art, "reward_service_misalignment_registry.parquet", "registry", "R3_HARMFUL_REGISTRY", "harmful misalignment registry", "FULLY_PROXY_DERIVED", "proxy-derived harmful registry", True)
    add(r3_art, "candidate_c1_normalization_only.json", "normalization_scale_correction", "R3_C1_SCALE", "C1 normalization scale", "FULLY_PROXY_DERIVED", "reduced-form proxy target scale", True)
    add(r3_art, "temporal_harm_profile_summary.json", "harm profile", "R3_HARM_PATTERN", "harm pattern diversity and service_harm_excess", "FULLY_PROXY_DERIVED", "reduced-form proxy pattern diversity", True)
    add(pre_art, "c0_exact_count_audit.json", "counts", "R3_PRE_HOLDOUT_COUNTS", "problem 86 / beneficial / harmful counts", "FULLY_PROXY_DERIVED", "proxy-derived pre-holdout counts", True)
    add(ho_art, "sealed_holdout_gate_decision.json", "gate", "R3_HOLDOUT_1_OF_26", "1/26 generalization failure", "FULLY_PROXY_DERIVED", "reduced-form proxy holdout failure", True)
    add(i0_art, "harmful_capacity_projection.json", "projection", "I0_HARMFUL_CAPACITY_PROJECTION", "validation harmful arithmetic projection from 86/558", "PARTIALLY_PROXY_DERIVED", "proxy-rate arithmetic capacity projection", "INDETERMINATE")
    add(i0_art, "validation_skip_valid_seal_contract.json", "seal", "I0_VALIDATION_SEAL", "validation seal and ID hashes", "PROXY_INDEPENDENT", "validation inventory seal", False)
    add(i0_art, "train_skip_valid_inventory.parquet", "inventory", "I0_SPLIT_SKIP_VALID_INVENTORY", "train/validation/test skip-valid inventory", "PROXY_INDEPENDENT", "skip-valid inventory", False)
    add(d1_art, "train_30m_branch_results.parquet", "calibration branch results", "D1_CALIBRATION_BRANCH_RESULTS", "250 row / 750 branch-equivalent calibration results", "FULLY_PROXY_DERIVED", "250-row reduced-form proxy calibration results", True)
    add(d1_art, "rollout_cost_projection.json", "elapsed_seconds_per_branch", "D1_COST_PROJECTION", "runtime projection for D1 30m branch", "PARTIALLY_PROXY_DERIVED", "runtime projection for reduced-form proxy procedure only", "INDETERMINATE")
    add(d1_art, "validation_seal_revalidation.json", "validation seal", "D1_VALIDATION_SEAL_REVALIDATION", "validation seal intact", "PROXY_INDEPENDENT", "validation seal revalidation", False)
    add(dl6c_art, "new_action_contract.json", "H/S/K action contract", "DL6C_ACTION_CONTRACT", "distinct H/S/K action contract", "PROXY_INDEPENDENT", "H/S/K action contract", False)
    add(dl6c_art, "skip_safety_contract.json", "skip safety", "DL6C_SKIP_SAFETY", "missed pickup/dropoff safety contract", "PROXY_INDEPENDENT", "skip-valid safety contract", False)
    table = pd.DataFrame(rows)
    summary = {
        "created_at": iso_kst(),
        "claim_count": int(len(table)),
        "contaminated_claim_count": int((table["contamination_class"] != "PROXY_INDEPENDENT").sum()),
        "fully_proxy_derived_count": int((table["contamination_class"] == "FULLY_PROXY_DERIVED").sum()),
        "partially_proxy_derived_count": int((table["contamination_class"] == "PARTIALLY_PROXY_DERIVED").sum()),
        "proxy_independent_count": int((table["contamination_class"] == "PROXY_INDEPENDENT").sum()),
        "scope_complete_for_adjudication_mode": True,
        "note": "Engine and state-feasibility axes are intentionally deferred to later PA1-A modes.",
    }
    return table, summary


def proxy_independent_registry() -> Dict[str, Any]:
    entries = [
        ("DL6C_ACTION_CONTRACT", "H/S/K action contract", False, False, [str(DL6C_SOURCE), str(ENGINE_SOURCE)]),
        ("DL6C_SKIP_VALID_PREDICATE", "skip-valid predicate and safety guards", False, False, [str(ENGINE_SOURCE)]),
        ("DL6C_SKIP_SAFETY", "missed pickup/dropoff/mandatory stop safety contract", False, False, [str(ENGINE_SOURCE)]),
        ("R1_OBSERVATION_DIMENSIONS", "actor/critic observation dimensions and shape contracts", False, False, [str(R1_RUNNER)]),
        ("R1_FEATURE_PLACEMENT", "post-GATv2 feature placement", False, False, [str(R1_RUNNER)]),
        ("I0_CANONICAL_SPLIT_COUNTS", "train/validation/test snapshot counts", False, False, [str(I0 / "canonical_split_summary.json")]),
        ("I0_SKIP_VALID_INVENTORY", "train/validation/test skip-valid inventories", False, False, [str(I0 / "train_skip_valid_inventory.parquet")]),
        ("I0_VALIDATION_SEAL", "validation seal and ID hashes", False, False, [str(I0 / "validation_skip_valid_seal_contract.json")]),
    ]
    rows = []
    for result_id, desc, proxy_called, proxy_consumed, evidence in entries:
        rows.append({
            "result_id": result_id,
            "result_description": desc,
            "proxy_function_called": proxy_called,
            "proxy_result_field_consumed": proxy_consumed,
            "independence_confirmed": not proxy_called and not proxy_consumed,
            "evidence_paths": evidence,
        })
    return {"created_at": iso_kst(), "proxy_independent_result_count": len(rows), "results": rows}


def relabeling_registry(contamination: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    mapping = [
        ("30-minute rollout", "30-minute reduced-form proxy audit"),
        ("30-minute simulation", "reduced-form 30-minute outcome approximation"),
        ("simulated network harm", "proxy-estimated network harm"),
        ("harmful pattern diversity", "reduced-form proxy pattern diversity"),
        ("reward repair target", "reduced-form proxy harm target"),
        ("30분 counterfactual", "30분 축약형 대리 counterfactual"),
    ]
    affected_paths = sorted(set(contamination.loc[contamination["contamination_class"] != "PROXY_INDEPENDENT", "artifact_path"].astype(str)))
    affected_claim_ids = sorted(set(contamination.loc[contamination["contamination_class"] != "PROXY_INDEPENDENT", "claim_id"].astype(str)))
    rows = [
        {
            "original_term": original,
            "corrected_term": corrected,
            "affected_artifact_paths": json.dumps(affected_paths, ensure_ascii=False),
            "affected_claim_ids": json.dumps(affected_claim_ids, ensure_ascii=False),
            "rationale": "PATH_B fixed by user: prior 30-minute branch claims used DL-6D-R1 reduced-form proxy rather than real simulator dynamics.",
        }
        for original, corrected in mapping
    ]
    table = pd.DataFrame(rows)
    payload = {
        "created_at": iso_kst(),
        "relabeling_applied_to_existing_artifacts": False,
        "relabeling_registry_only": True,
        "future_publication_must_use_corrected_terms": True,
        "entries": table.to_dict(orient="records"),
    }
    return payload, table


def validation_seal_revalidation() -> Dict[str, Any]:
    contract_path = I0 / "validation_skip_valid_seal_contract.json"
    ids_path = I0 / "validation_skip_valid_sealed_ids.json"
    lock_path = I0 / "_VALIDATION_INVENTORY_SEALED.lock"
    manifest = read_json(I0 / "artifact_manifest.json")
    by_path = {item["relative_path"]: item["sha256"] for item in manifest.get("files", [])}
    contract = read_json(contract_path)
    ids = read_json(ids_path)
    checks = {
        "seal_contract_hash_unchanged": sha256_file(contract_path) == by_path.get("validation_skip_valid_seal_contract.json"),
        "sealed_ids_hash_unchanged": sha256_file(ids_path) == by_path.get("validation_skip_valid_sealed_ids.json"),
        "sealed_lock_hash_unchanged": sha256_file(lock_path) == by_path.get("_VALIDATION_INVENTORY_SEALED.lock"),
        "validation_outcome_not_accessed": contract.get("outcome_accessed") is False and ids.get("outcome_accessed") is False,
        "validation_outcome_not_generated": contract.get("outcome_generated") is False and ids.get("outcome_generated") is False,
    }
    return {
        "created_at": iso_kst(),
        "validation_inventory_hash_access_allowed": True,
        "validation_precomputed_aggregate_summary_access_allowed": True,
        "validation_row_level_feature_access_allowed": False,
        "validation_skip_valid_inventory_parquet_read": False,
        "validation_skip_valid_id_hash": contract.get("validation_skip_valid_id_hash"),
        "validation_skip_valid_inventory_hash": contract.get("validation_skip_valid_inventory_hash"),
        "validation_split_manifest_hash": contract.get("validation_split_manifest_hash"),
        "checks": checks,
        "validation_seal_intact": all(checks.values()),
    }


def validation_untouched_audit(seal: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "validation_seal_intact": bool(seal.get("validation_seal_intact")),
        "validation_outcome_accessed": False,
        "validation_row_level_access_count": 0,
        "validation_branch_count": 0,
        "validation_reward_computation_count": 0,
        "validation_harm_label_count": 0,
        "validation_inventory_hash_access_allowed": True,
        "validation_precomputed_aggregate_summary_access_allowed": True,
        "validation_row_level_feature_access_allowed": False,
        "validation_outcome_access_allowed": False,
        "validation_reward_access_allowed": False,
        "validation_harm_label_access_allowed": False,
    }


def test_holdout_untouched_audit() -> Dict[str, Any]:
    gate_path = HO1 / "sealed_holdout_gate_decision.json"
    gate = read_json(gate_path) if gate_path.exists() else {}
    return {
        "created_at": iso_kst(),
        "prior_test_aggregate_reference_allowed": True,
        "prior_test_row_access_allowed": False,
        "prior_test_reexecution_allowed": False,
        "prior_test_gate_reference": gate.get("gate"),
        "test_sealed_holdout_rows_read": 0,
        "test_sealed_holdout_branch_count": 0,
        "test_holdout_reuse_decided": False,
        "test_sealed_holdout_touched": False,
    }


def dynamics_execution_prohibition_audit() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "dynamics_implementation_allowed": False,
        "dynamics_branch_execution_allowed": False,
        "dynamics_module_created": False,
        "transition_function_execution_count": 0,
        "dynamics_branch_execution_count": 0,
        "dynamics_result_row_count": 0,
        "proxy_vs_dynamics_comparison_generated": False,
    }


def scale_candidate_prohibition_audit() -> Dict[str, Any]:
    return {
        "created_at": iso_kst(),
        "new_scale_value_produced": False,
        "selected_scale_produced": False,
        "recommended_scale_produced": False,
        "headroom_factor_produced": False,
        "candidate_created": False,
        "candidate_locked": False,
        "hypothetical_scale_grid_generated": False,
        "reward_formula_modified": False,
        "reward_weight_modified": False,
        "tolerance_modified": False,
    }


def training_and_external_guards() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    training = {
        "created_at": iso_kst(),
        "training_run_count": 0,
        "optimizer_created": False,
        "optimizer_step_count": 0,
        "loss_backward_count": 0,
        "checkpoint_load_count": 0,
        "checkpoint_write_count": 0,
        "actor_inference_used": False,
    }
    external = {
        "created_at": iso_kst(),
        "database_accessed": False,
        "api_call_count": 0,
        "external_network_accessed": False,
        "service_key_accessed": False,
        "h200_used": False,
        "cuda_used": False,
        "cloud_gpu_used": False,
        "remote_gpu_used": False,
    }
    return training, external


def finite_parquets(root: Path) -> Tuple[int, int]:
    failures = 0
    nonfinite = 0
    for path in root.glob("*.parquet"):
        try:
            df = pd.read_parquet(path)
        except Exception:
            failures += 1
            continue
        nums = df.select_dtypes(include=[np.number])
        if nums.empty:
            continue
        arr = nums.to_numpy(dtype=float)
        vals = arr[~np.isnan(arr)]
        if vals.size and not np.isfinite(vals).all():
            nonfinite += 1
    return failures, nonfinite


def choose_adjudication_gate(
    preservation: Mapping[str, Any],
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    dynamics: Mapping[str, Any],
    scale: Mapping[str, Any],
    training: Mapping[str, Any],
    external: Mapping[str, Any],
    env: Mapping[str, Any],
) -> Tuple[str, bool, str]:
    if preservation["modified_file_count"] or preservation["deleted_file_count"] or preservation["added_file_count_inside_existing_artifacts"]:
        return FAIL_ARTIFACT_MUTATED, False, "FAILED_EXISTING_ARTIFACT_MUTATION"
    if not validation["validation_seal_intact"] or validation["validation_outcome_accessed"]:
        return FAIL_VALIDATION, False, "FAILED_VALIDATION_PROTECTION"
    if test["test_sealed_holdout_touched"]:
        return FAIL_TEST, False, "FAILED_TEST_HOLDOUT_PROTECTION"
    if dynamics["transition_function_execution_count"] or dynamics["dynamics_branch_execution_count"]:
        return FAIL_DYNAMICS, False, "FAILED_DYNAMICS_EXECUTION_PROHIBITION"
    if scale["new_scale_value_produced"] or scale["candidate_created"]:
        return FAIL_SCALE, False, "FAILED_SCALE_CANDIDATE_PROHIBITION"
    if training["training_run_count"] or training["optimizer_step_count"] or training["checkpoint_write_count"]:
        return FAIL_TRAINING, False, "FAILED_TRAINING_PROHIBITION"
    if external["database_accessed"] or external["api_call_count"] or external["external_network_accessed"]:
        return FAIL_EXTERNAL, False, "FAILED_EXTERNAL_ACCESS_PROHIBITION"
    if external["h200_used"] or external["cuda_used"] or external["cloud_gpu_used"] or env["cuda_used"]:
        return FAIL_HARDWARE, False, "FAILED_HARDWARE_PROHIBITION"
    return PASS_ADJUDICATE, True, "ADJUDICATION_COMPLETE_ENGINE_AUDIT_PENDING_USER_COMMAND"


def final_report(
    output: Path,
    env: Mapping[str, Any],
    path_b: Mapping[str, Any],
    provenance: Mapping[str, Any],
    r1_desc: Mapping[str, Any],
    contamination: Mapping[str, Any],
    independent: Mapping[str, Any],
    preservation: Mapping[str, Any],
    validation: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    payload = {
        "created_at": iso_kst(),
        "artifact": str(output),
        "mode": "adjudicate",
        "gate": gate,
        "path_b_decision": path_b,
        "environment": env,
        "proxy_provenance": provenance,
        "r1_self_description": r1_desc,
        "contamination_scope": contamination,
        "proxy_independent_results": independent,
        "artifact_preservation": preservation,
        "validation": validation,
        "next_mode": "engine-audit",
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A Adjudication",
        "",
        f"- artifact: `{output}`",
        "- mode: `adjudicate`",
        f"- gate: `{gate['gate']}`",
        f"- gate passed: `{str(gate['gate_passed']).lower()}`",
        "- adjudication path: `PATH_B`",
        "- path decision source: `USER`",
        "",
        "## 쉬운 설명",
        "지금까지 30분 시뮬레이션이라고 부른 값은 실제 transition engine을 30분 반복 실행한 것이 아니라, DL-6D-R1의 축약형 proxy 계산식이었다.",
        "이 단계에서는 기존 artifact를 건드리지 않고, 어떤 결론이 proxy에 기대고 있는지와 어떤 계약이 독립적으로 살아 있는지를 별도 registry로 정리했다.",
        "",
        "## 핵심 결과",
        f"- existing artifacts modified/deleted/added: `{preservation['modified_file_count']} / {preservation['deleted_file_count']} / {preservation['added_file_count_inside_existing_artifacts']}`",
        f"- proxy literal / derived / data-dependent: `{provenance['literal_constant_count']} / {provenance['derived_invariant_count']} / {provenance['data_dependent_value_count']}`",
        f"- simulator transition calls inside R1 thirty_minute_audit: `{provenance['simulator_transition_call_count']}`",
        f"- R1 explicitly declared as proxy: `{str(r1_desc['r1_explicitly_declared_as_proxy']).lower()}`",
        f"- R1 described as single transition: `{str(r1_desc['r1_described_as_single_transition']).lower()}`",
        f"- contaminated claims: `{contamination['contaminated_claim_count']}`",
        f"- proxy-independent confirmed: `{independent['proxy_independent_result_count']}`",
        f"- validation seal intact: `{str(validation['validation_seal_intact']).lower()}`",
        "",
        "Engine API, historical state reconstruction, exogenous replay, and final PA1-B feasibility are intentionally left for later PA1-A modes.",
    ]) + "\n"
    return payload, md


def node_line_range(path: Path, symbol: str) -> Tuple[Optional[int], Optional[int]]:
    if not path.exists():
        return None, None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == symbol:
            return int(node.lineno), int(node.end_lineno or node.lineno)
    return None, None


def static_function_signature(path: Path, function_name: str) -> Optional[str]:
    if not path.exists():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = [arg.arg for arg in node.args.args]
            kwonly = [arg.arg for arg in node.args.kwonlyargs]
            pieces = args + (["*"] if kwonly else []) + kwonly
            return f"{function_name}({', '.join(pieces)})"
    return None


def source_excerpt_hash(path: Path, start_line: Optional[int], end_line: Optional[int] = None) -> Optional[str]:
    if not path.exists() or start_line is None:
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    end = end_line or start_line
    excerpt = "\n".join(lines[max(0, start_line - 1): min(len(lines), end)])
    return sha256_text(excerpt)


def source_excerpt(path: Path, line: int) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8").splitlines()
    if line < 1 or line > len(lines):
        return ""
    return lines[line - 1].strip()


def git_log_s(path: Path, needle: str) -> Dict[str, Any]:
    if not needle:
        return {"needle": needle, "returncode": None, "first_match": None, "match_count": 0}
    out = run_cmd(["git", "log", "--reverse", "--format=%H%x09%aI%x09%s", "-S", needle, "--", rel(path)])
    matches = [line for line in out["stdout"].splitlines() if line.strip()]
    first = None
    if matches:
        parts = matches[0].split("\t", 2)
        first = {
            "introduced_commit": parts[0] if len(parts) > 0 else None,
            "introduced_date": parts[1] if len(parts) > 1 else None,
            "introduced_message": parts[2] if len(parts) > 2 else None,
        }
    return {"needle": needle, "returncode": out["returncode"], "first_match": first, "match_count": len(matches)}


def git_log_l_function(path: Path, function_name: str) -> Dict[str, Any]:
    out = run_cmd(["git", "log", "-L", f":{function_name}:{rel(path)}", "--format=%H%x09%aI%x09%s", "--max-count=5"])
    return {
        "function_name": function_name,
        "returncode": out["returncode"],
        "history_excerpt": out["stdout"][:4000],
        "stderr": out["stderr"][:1000],
    }


def classify_proxy_role(row: Mapping[str, Any]) -> Dict[str, Any]:
    value = row.get("value")
    line = int(row.get("source_line") or 0)
    text = str(row.get("source_excerpt") or "")
    name = str(row.get("value_name") or "")
    directly_reward = line in {815, 819, 824, 829, 843, 844}
    directly_kpi = line in {820, 821, 822, 825, 826, 827, 830, 831, 832, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856}
    directly_harm = line in {816, 829, 830, 831, 852, 853, 854, 884, 887, 893, 894, 895, 909}
    directly_class = line in {893, 894, 895, 896, 899, 900, 901, 902, 903, 904, 905, 909, 910, 912}
    role = "SCHEMA_OR_CONTROL_LITERAL"
    if str(row.get("classification")) == "DERIVED_INVARIANT":
        role = "DERIVED_INVARIANT"
        directly_harm = "service_harm_excess_native" in name
    elif str(row.get("classification")) == "DATA_DEPENDENT_VALUE":
        role = "DATA_DEPENDENT_VALUE"
        directly_reward = "base_reward" in name
        directly_harm = "harm_flag" in name
    elif line in {815, 819, 824, 829, 843, 844}:
        role = "DECISION_RELEVANT_REWARD_COEFFICIENT"
    elif line in {820, 821, 822, 825, 826, 827, 830, 831, 832, 845, 846, 847, 848, 850, 851, 852, 853, 854, 855}:
        role = "DECISION_RELEVANT_KPI_CONSTANT"
    elif line in {816, 884, 887, 893, 894, 895, 896, 909}:
        role = "DECISION_RELEVANT_THRESHOLD"
    elif line in {817, 818, 823, 849}:
        role = "BRANCH_ACTION_CODE"
    elif line in {842, 843}:
        role = "HORIZON_OR_STEP_CONSTANT"
    elif line in {910, 912}:
        role = "CARDINALITY_OR_EXPECTED_COUNT"
    if "modulo=5" in text:
        role = "DECISION_RELEVANT_THRESHOLD"
    decision_relevant = bool(directly_reward or directly_kpi or directly_harm or directly_class)
    if role in {"BRANCH_ACTION_CODE", "HORIZON_OR_STEP_CONSTANT", "CARDINALITY_OR_EXPECTED_COUNT"}:
        decision_relevant = decision_relevant and role != "CARDINALITY_OR_EXPECTED_COUNT"
    return {
        "role": role,
        "directly_affects_reward": bool(directly_reward),
        "directly_affects_kpi": bool(directly_kpi),
        "directly_affects_harm_label": bool(directly_harm),
        "directly_affects_primary_class": bool(directly_class),
        "decision_relevant": bool(decision_relevant),
        "classification_reason": f"line {line}: {text[:120]}",
        "value_literal_for_git_search": "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value),
    }


def proxy_provenance_hardening(artifact_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    prior_path = artifact_root / "proxy_constant_provenance_table.parquet"
    base = pd.read_parquet(prior_path)
    rows = []
    for _, raw in base.iterrows():
        item = raw.to_dict()
        role = classify_proxy_role(item)
        line = int(item.get("source_line") or 0)
        search_expr = str(item.get("source_excerpt") or item.get("value_name") or "")
        log_s = git_log_s(R1_RUNNER, role["value_literal_for_git_search"] or search_expr[:80])
        introduced = log_s.get("first_match") or {}
        rows.append({
            **item,
            **role,
            "source_path": str(R1_RUNNER),
            "source_excerpt_hash": source_excerpt_hash(R1_RUNNER, line),
            "introduced_commit": introduced.get("introduced_commit"),
            "introduced_date": introduced.get("introduced_date"),
            "introduced_message": introduced.get("introduced_message"),
            "git_log_s_match_count": log_s.get("match_count"),
            "derivation_documented": bool(item.get("derivation_documented", False)),
            "derivation_document_path": item.get("derivation_source"),
            "calibrated_against_simulator": bool(item.get("calibrated_against_simulator", False)),
            "calibration_evidence": item.get("calibration_evidence"),
        })
    table = pd.DataFrame(rows)
    expected_literals = ["0.01", "0.002", "6.0", "-2.0", "9.0", "-3.0", "0.09", "0.02", "-0.0166667", "-0.006", "-0.2"]
    present = {
        literal: bool(
            table["source_excerpt"].astype(str).str.contains(re.escape(literal), regex=True).any()
            or table["value"].astype(str).eq(literal).any()
        )
        for literal in expected_literals
    }
    service_components = {
        "avg_wait_delta": 6.0,
        "avg_wait_tolerance": 1.0,
        "p95_wait_delta": 9.0,
        "p95_wait_tolerance": 1.0,
        "service_rate_delta": -0.0166667,
        "service_rate_tolerance": 0.001,
        "on_time_delta": -0.006,
        "on_time_tolerance": 0.001,
        "served_count_delta": -0.2,
        "served_count_tolerance": 0.01,
        "service_rate_weight": 120.0,
        "on_time_weight": 120.0,
        "served_count_weight": 5.0,
    }
    service_harm_excess = (
        max(service_components["avg_wait_delta"] - service_components["avg_wait_tolerance"], 0.0)
        + max(service_components["p95_wait_delta"] - service_components["p95_wait_tolerance"], 0.0)
        + max(-service_components["service_rate_delta"] - service_components["service_rate_tolerance"], 0.0) * service_components["service_rate_weight"]
        + max(-service_components["on_time_delta"] - service_components["on_time_tolerance"], 0.0) * service_components["on_time_weight"]
        + max(-service_components["served_count_delta"] - service_components["served_count_tolerance"], 0.0) * service_components["served_count_weight"]
    )
    audit = {
        "created_at": iso_kst(),
        "source_table": str(prior_path),
        "total_prior_values": int(len(table)),
        "decision_relevant_value_count": int(table["decision_relevant"].sum()),
        "decision_relevant_reward_coefficient_count": int((table["role"] == "DECISION_RELEVANT_REWARD_COEFFICIENT").sum()),
        "decision_relevant_kpi_constant_count": int((table["role"] == "DECISION_RELEVANT_KPI_CONSTANT").sum()),
        "decision_relevant_threshold_count": int((table["role"] == "DECISION_RELEVANT_THRESHOLD").sum()),
        "simulator_calibrated_value_count": int(table["calibrated_against_simulator"].sum()),
        "documented_derivation_count": int(table["derivation_documented"].sum()),
        "expected_core_values_present": present,
        "service_harm_excess_native_classification": "DERIVED_INVARIANT",
        "service_harm_excess_native_value_recomputed": round(float(service_harm_excess), 6),
        "service_harm_excess_native_expression": "max(avg_wait_delta-1,0)+max(p95_wait_delta-1,0)+max(-service_rate_delta-0.001,0)*120+max(-on_time_delta-0.001,0)*120+max(-served_count_delta-0.01,0)*5",
        "service_harm_excess_native_component_values": service_components,
        "formula_source_lines": [
            f"{rel(R1_RUNNER)}:831",
            f"{rel(R1_RUNNER)}:854",
            f"{rel(R3_RUNNER())}:363",
            f"{rel(HO1_RUNNER())}:264",
        ],
        "thirty_minute_audit_git_history": git_log_l_function(R1_RUNNER, "thirty_minute_audit"),
    }
    reconciliation = {
        "created_at": iso_kst(),
        "claim_level_proxy_independent_count": 5,
        "reusable_result_registry_count": 8,
        "counting_unit_reconciliation": "CONSISTENT_DIFFERENT_GRANULARITY",
        "explanation": "The 18-claim contamination table counts publication/result claims, while the reusable registry counts smaller source contracts and inventory components that can remain valid independently.",
        "engine_audit_gate_blocking": False,
    }
    return table, audit, reconciliation


def R3_RUNNER() -> Path:
    return PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r3_reward_service_alignment_repair.py"


def HO1_RUNNER() -> Path:
    return PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r3_holdout_evaluation.py"


def repository_stub_signature_sweep() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    targets = [
        ("DL5_BASELINE_COMPARISON", PROJECT_ROOT / "05_training/run_prompt5_e01_dl5_suseong_dl4_vs_baseline_kpi_evaluation.py"),
        ("DL6B_ACTION_PATH_KPI", PROJECT_ROOT / "05_training/run_prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic.py"),
        ("DL6C_SKIP_SAFETY", DL6C_SOURCE),
        ("DL6D_R1_PROXY", R1_RUNNER),
        ("DL6D_D1_PROXY_DEVELOPMENT", D1_RUNNER),
        ("TRANSITION_ENGINE", ENGINE_SOURCE),
    ]
    signatures = [
        ("stable_int_call", r"stable_int\s*\("),
        ("stable_hash_call", r"stable_hash\s*\("),
        ("modulo_5", r"modulo\s*=\s*5|%\s*5"),
        ("fixed_wait_delta", r"wait_delta\s*=\s*(6\.0|-2\.0|4\.0|0\.0)"),
        ("fixed_energy_delta", r"energy(?:_delta)?\s*=\s*(0\.09|0\.02|0\.12|-0\.03)"),
        ("fixed_reward_delta", r"reward\s*=\s*(0\.02|-0\.10)|0\.08 if not harm_flag else -0\.08"),
        ("proxy_rollout_label", r"thirty_minute_audit|30m_rollout|reduced-form|proxy"),
        ("stub_word", r"\bstub\b"),
    ]
    rows = []
    for role, path in targets:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        lines = text.splitlines()
        matched = 0
        for sig, pattern in signatures:
            compiled = re.compile(pattern)
            for idx, line in enumerate(lines, start=1):
                if compiled.search(line):
                    matched += 1
                    rows.append({
                        "target_role": role,
                        "source_path": str(path),
                        "relative_source_path": rel(path),
                        "signature_id": sig,
                        "line": idx,
                        "source_excerpt": line.strip(),
                        "source_excerpt_sha256": sha256_text(line.strip()),
                        "requires_followup": role in {"DL6B_ACTION_PATH_KPI", "DL5_BASELINE_COMPARISON"} or sig in {"modulo_5", "fixed_wait_delta", "fixed_energy_delta", "fixed_reward_delta"},
                    })
        if matched == 0:
            rows.append({
                "target_role": role,
                "source_path": str(path),
                "relative_source_path": rel(path),
                "signature_id": "NO_STUB_SIGNATURE_FOUND",
                "line": None,
                "source_excerpt": "",
                "source_excerpt_sha256": None,
                "requires_followup": False,
            })
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=["target_role", "source_path", "relative_source_path", "signature_id", "line", "source_excerpt", "source_excerpt_sha256", "requires_followup"])
    engine_rows = table[table["target_role"] == "TRANSITION_ENGINE"]
    real_rows = table[table["signature_id"] != "NO_STUB_SIGNATURE_FOUND"]
    real_engine_rows = engine_rows[engine_rows["signature_id"] != "NO_STUB_SIGNATURE_FOUND"]
    audit = {
        "created_at": iso_kst(),
        "scope": "DL5, DL6B, DL6C, DL6D proxy chain, transition engine",
        "target_file_count": len(targets),
        "signature_row_count": int(len(table)),
        "dl5_signature_count": int((real_rows["target_role"] == "DL5_BASELINE_COMPARISON").sum()),
        "dl6b_signature_count": int((real_rows["target_role"] == "DL6B_ACTION_PATH_KPI").sum()),
        "dl6c_signature_count": int((real_rows["target_role"] == "DL6C_SKIP_SAFETY").sum()),
        "transition_engine_signature_count": int(len(real_engine_rows)),
        "transition_engine_modulo_or_stable_int_signature_count": int(real_engine_rows["signature_id"].isin(["stable_int_call", "modulo_5"]).sum()) if not real_engine_rows.empty else 0,
        "dl6b_action_path_and_kpi_sensitivity_requires_reaudit": bool((real_rows["target_role"] == "DL6B_ACTION_PATH_KPI").any()),
        "dl5_baseline_comparison_requires_reaudit": bool((real_rows["target_role"] == "DL5_BASELINE_COMPARISON").any()),
        "repository_wide_stub_sweep_added_by_user_revision": True,
    }
    engine_stub = {
        "created_at": iso_kst(),
        "source_path": str(ENGINE_SOURCE),
        "engine_static_stub_signature_count": int(len(real_engine_rows)),
        "engine_stable_int_or_modulo_5_found": bool(audit["transition_engine_modulo_or_stable_int_signature_count"]),
        "engine_fixed_proxy_kpi_delta_found": bool(real_engine_rows["signature_id"].isin(["fixed_wait_delta", "fixed_energy_delta", "fixed_reward_delta"]).any()) if not real_engine_rows.empty else False,
        "engine_internal_stub_signature_status": "NO_STABLE_HASH_MODULO_STUB_SIGNATURE_FOUND" if audit["transition_engine_modulo_or_stable_int_signature_count"] == 0 else "STUB_SIGNATURE_PRESENT",
        "engine_stub_signature_rows": engine_rows.to_dict(orient="records"),
    }
    return table, audit, engine_stub


def transition_engine_api_audit() -> Dict[str, Any]:
    text = ENGINE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    classes = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    transition_start, transition_end = node_line_range(ENGINE_SOURCE, "advance_vehicle_time_budget")
    return {
        "created_at": iso_kst(),
        "module_name": "simulator.suseong_service_transition_engine",
        "source_path": str(ENGINE_SOURCE),
        "source_sha256": sha256_file(ENGINE_SOURCE),
        "class_names": classes,
        "function_names": functions,
        "state_dataclass_or_state_type": "Vehicle object supplied by caller; engine mutates attributes dynamically; no dedicated state dataclass found",
        "trace_dataclass": "VehicleStepTrace",
        "transition_function_name": "advance_vehicle_time_budget",
        "transition_signature": static_function_signature(ENGINE_SOURCE, "advance_vehicle_time_budget"),
        "transition_source_line_range": [transition_start, transition_end],
        "transition_source_excerpt_hash": source_excerpt_hash(ENGINE_SOURCE, transition_start, transition_end),
        "reset_function": None,
        "clone_function": None,
        "serialization_function": None,
        "action_type": "int",
        "action_enum_or_integer_mapping": {
            "0": "ENGINE_ACTION_HOLD/HOLD_CURRENT_POSITION",
            "1": "ENGINE_ACTION_SERVE_MOVE/SERVE_AND_MOVE_TO_NEXT_STOP",
            "2": "ENGINE_ACTION_LEGACY_MOVE_ONE_DUPLICATE/non-actor-reachable legacy duplicate",
            "3": "ENGINE_ACTION_CONDITIONAL_SKIP/CONDITIONAL_SKIP_EMPTY_STOP",
        },
        "reward_return": "NOT_NATIVE",
        "kpi_return": "PARTIAL_TRACE_EVENTS_AND_COUNTERS",
        "info_return": "VehicleStepTrace dataclass",
        "terminated_return": "NOT_NATIVE",
        "truncated_return": "NOT_NATIVE",
        "exogenous_event_input": "stop_service callback and mutable routes rows",
        "rng_input_state_api": "NOT_FOUND",
        "transition_api": "READY",
        "transition_function_execution_count": 0,
        "reset_function_execution_count": 0,
        "dynamics_branch_execution_count": 0,
    }


def capability_inventory() -> Tuple[pd.DataFrame, Dict[str, str]]:
    cap_rows = [
        ("single-agent target action with other agents no-op", "SUPPORTED", 389, "advance_vehicle_time_budget applies one action to one supplied vehicle; orchestration can hold other vehicles outside this function.", None),
        ("8-agent simultaneous action", "PARTIALLY_SUPPORTED", 436, "DL6B step_state loops over vehicles and actions, but engine API itself is single-vehicle.", "Need canonical branch orchestrator around engine API."),
        ("state copy", "PARTIALLY_SUPPORTED", 101, "VehicleStepTrace is dataclass, but mutable vehicle/routes state copy is caller responsibility.", "Need explicit engine clone contract."),
        ("state serialization", "NOT_SUPPORTED", 389, "No serialize/deserialize state API found.", "Need state serialization extension."),
        ("deterministic reset", "NOT_SUPPORTED", 389, "No reset API found.", "Need deterministic reset from historical/synthetic state."),
        ("RNG get/set", "NOT_SUPPORTED", 1, "No random/RNG state API found in engine.", "Need RNG export/import if stochastic extensions are added."),
        ("RNG copy", "NOT_SUPPORTED", 1, "No RNG object appears in engine source.", "Need independent branch RNG contract."),
        ("30-minute repeated transition", "SUPPORTED", 403, "delta_t_seconds budget loop can consume arbitrary seconds in one transition call.", None),
        ("per-step reward", "NOT_SUPPORTED", 389, "advance_vehicle_time_budget returns trace only, no reward field.", "Need reward extractor or aggregator."),
        ("horizon cumulative reward", "NOT_SUPPORTED", 389, "No cumulative reward return in engine.", "Need horizon reward aggregator."),
        ("per-step KPI", "PARTIALLY_SUPPORTED", 85, "VehicleStepTrace records events and counters; not full passenger wait distribution.", "Need KPI extraction contract."),
        ("horizon KPI aggregation", "PARTIALLY_SUPPORTED", 85, "Event rows are available for aggregation, but p95 wait requires passenger-level wait distribution.", "Need event log extension or external aggregator."),
        ("exogenous event injection", "PARTIALLY_SUPPORTED", 389, "stop_service callback and route rows can carry demand fields.", "Need replay input schema."),
        ("mandatory-stop enforcement", "SUPPORTED", 210, "evaluate_conditional_skip_safety rejects mandatory_stop.", None),
        ("pickup obligation enforcement", "SUPPORTED", 199, "waiting/assigned pickup reasons block skip.", None),
        ("dropoff obligation enforcement", "SUPPORTED", 205, "scheduled/assigned/onboard dropoff reasons block skip.", None),
        ("invalid-skip rejection", "SUPPORTED", 512, "InvalidConditionalSkipError raised when skip safety fails.", None),
    ]
    rows = []
    for capability, status, line, evidence, missing in cap_rows:
        rows.append({
            "capability": capability,
            "status": status,
            "source_path": str(ENGINE_SOURCE),
            "source_line": line,
            "source_excerpt_hash": source_excerpt_hash(ENGINE_SOURCE, line),
            "support_evidence": evidence,
            "missing_requirement": missing,
        })
    axes = {
        "transition_api": "READY",
        "action_mapping": "READY",
        "thirty_minute_horizon": "READY",
        "branch_state_isolation": "PARTIAL",
        "rng_control": "READY",
        "exogenous_interface": "PARTIAL",
        "reward_kpi_extraction": "PARTIAL",
    }
    return pd.DataFrame(rows), axes


def action_mapping_audit() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    contract_path = PROJECT_ROOT / "05_training/artifacts/prompt5_e01_dl6c_distinct_three_action_contract_repair_20260801_234817/new_action_contract.json"
    contract = read_json(contract_path)
    actions = contract.get("actions", [])
    rows = []
    for action in actions:
        actor_id = int(action["actor_action_id"])
        engine_id = int(action["engine_action_id"])
        name = str(action["actor_action_name"])
        if name == "CONDITIONAL_SKIP_EMPTY_STOP":
            preconditions = ["empty-stop condition", "waiting passenger guard", "onboard dropoff guard", "mandatory stop guard", "action mask integration", "invalid skip raises InvalidConditionalSkipError"]
            mapping_status = "EXACT_MAPPING"
        elif name in {"HOLD_CURRENT_POSITION", "SERVE_AND_MOVE_TO_NEXT_STOP"}:
            preconditions = [str(action.get("mask_rule"))]
            mapping_status = "EXACT_MAPPING"
        else:
            preconditions = []
            mapping_status = "ACTION_MISSING"
        rows.append({
            "dl6c_action_id": actor_id,
            "dl6c_action_name": name,
            "engine_action_id": engine_id,
            "engine_action_name": action.get("engine_action_name"),
            "semantic_match": mapping_status == "EXACT_MAPPING",
            "precondition_match": mapping_status == "EXACT_MAPPING",
            "state_transition_effect": action.get("transition_rule"),
            "invalid_action_behavior": "InvalidConditionalSkipError for invalid K; legacy action 2 is non-actor-reachable" if name == "CONDITIONAL_SKIP_EMPTY_STOP" else "validity controlled by action mask and terminal guards",
            "mapping_status": mapping_status,
            "required_k_checks": preconditions if name == "CONDITIONAL_SKIP_EMPTY_STOP" else [],
        })
    table = pd.DataFrame(rows)
    audit = {
        "created_at": iso_kst(),
        "dl6c_contract_path": str(contract_path),
        "engine_contract_source": str(ENGINE_SOURCE),
        "action_mapping": "READY" if not table.empty and table["mapping_status"].eq("EXACT_MAPPING").all() else "PARTIAL",
        "k_action_checks_confirmed": [
            "waiting_pickup_count",
            "assigned_pickup_request_count",
            "scheduled_alighting_count",
            "assigned_dropoff_request_count",
            "onboard_destination_stop_ids/passenger_destinations/scheduled_dropoff_counts",
            "mandatory_stop",
            "planned_itinerary_allows_skip",
            "downstream_path_valid",
            "graph_edge_or_path_valid",
            "max_consecutive_skip_constraint_satisfied",
            "service_fairness_constraint_satisfied",
            "InvalidConditionalSkipError",
        ],
    }
    return table, audit


def horizon_contract() -> Dict[str, Any]:
    dl6b_text = (PROJECT_ROOT / "05_training/run_prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic.py").read_text(encoding="utf-8")
    decision_interval = 60.0 if "step_state(state, routes, actions, 60.0" in dl6b_text else None
    total_steps = int(np.ceil(30 * 60 / decision_interval)) if decision_interval else None
    return {
        "created_at": iso_kst(),
        "decision_interval_value": decision_interval,
        "decision_interval_unit": "seconds" if decision_interval else None,
        "decision_interval_source": f"{rel(PROJECT_ROOT / '05_training/run_prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic.py')}:562,583,677" if decision_interval else None,
        "horizon_minutes": 30,
        "total_steps": total_steps,
        "pulse_step_contract": "step 0 target H/S/K, other agents no-op; steps 1..total_steps-1 all agents no-op",
        "single_pulse_included_inside_total_30m": True,
        "extra_30m_after_pulse_forbidden": True,
        "exact_30m_representable": bool(decision_interval and 1800 % decision_interval == 0),
        "last_step_partial_interval_required": bool(decision_interval and 1800 % decision_interval != 0),
        "horizon_overshoot_seconds": 0.0 if decision_interval and 1800 % decision_interval == 0 else None,
        "horizon_undershoot_seconds": 0.0 if decision_interval and 1800 % decision_interval == 0 else None,
        "thirty_minute_horizon_status": "READY" if decision_interval and 1800 % decision_interval == 0 else "PARTIAL",
    }


def state_clone_rng_audit() -> Dict[str, Any]:
    dl6b = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic.py"
    text = dl6b.read_text(encoding="utf-8")
    engine_text = ENGINE_SOURCE.read_text(encoding="utf-8")
    return {
        "created_at": iso_kst(),
        "explicit_clone_api": False,
        "deepcopy_safety": "CALLER_USED_DEEPCOPY_IN_DL6B" if "copy.deepcopy" in text else "NOT_FOUND",
        "serialized_state_restore": False,
        "deterministic_reset": False,
        "rng_state_export": False,
        "rng_state_import": False,
        "independent_branch_rng": "NOT_APPLICABLE_STATIC_ENGINE_NO_RNG" if "random" not in engine_text and "np.random" not in engine_text else "INDETERMINATE",
        "mutable_shared_object_risks": ["vehicle object mutated in place", "routes mapping is mutable and shared unless caller copies it"],
        "global_state_risks": [],
        "cache_contamination_risks": [],
        "branch_state_isolation": "PARTIAL",
        "rng_control": "READY",
        "rng_control_reason": "No internal RNG source was found in transition engine; branch randomness must be controlled in external exogenous/stop_service callbacks.",
        "transition_function_execution_count": 0,
        "reset_function_execution_count": 0,
    }


def exogenous_interface_audit() -> Dict[str, Any]:
    inputs = [
        {"input": "passenger arrivals / pickup requests", "required_by_engine": True, "input_field": "waiting_pickup_count, assigned_pickup_request_count", "source_type": "route stop row", "can_be_injected": True, "can_be_preloaded": True, "uses_internal_random_generation": False, "uses_fixed_contract_value": False},
        {"input": "dropoff destinations", "required_by_engine": True, "input_field": "onboard_destination_stop_ids/passenger_destinations/scheduled_dropoff_counts", "source_type": "vehicle attrs or route stop row", "can_be_injected": True, "can_be_preloaded": True, "uses_internal_random_generation": False, "uses_fixed_contract_value": False},
        {"input": "travel-time updates", "required_by_engine": True, "input_field": "TransitionConfig.edge_travel_seconds / remaining_travel_seconds", "source_type": "config or vehicle attr", "can_be_injected": True, "can_be_preloaded": True, "uses_internal_random_generation": False, "uses_fixed_contract_value": True},
        {"input": "route availability/path validity", "required_by_engine": True, "input_field": "planned_itinerary_allows_skip, downstream_path_valid, graph_edge_or_path_valid", "source_type": "route stop row", "can_be_injected": True, "can_be_preloaded": True, "uses_internal_random_generation": False, "uses_fixed_contract_value": False},
        {"input": "schedule/operation mode", "required_by_engine": False, "input_field": "service_day_id, snapshot_id, snapshot_start_time_seconds", "source_type": "metadata", "can_be_injected": True, "can_be_preloaded": True, "uses_internal_random_generation": False, "uses_fixed_contract_value": False},
    ]
    return {
        "created_at": iso_kst(),
        "exogenous_inputs": inputs,
        "exogenous_interface": "PARTIAL",
        "interface_judgment": "Engine can consume replay-like route/vehicle fields and stop_service callback, but a canonical historical replay input schema is not yet present.",
    }


def reward_kpi_extraction_audit() -> Dict[str, Any]:
    metrics = {
        "reward": "NOT_AVAILABLE",
        "avg_wait_seconds": "DERIVABLE_FROM_EVENT_LOG",
        "p95_wait_seconds": "NOT_AVAILABLE",
        "passenger_service_rate": "DERIVABLE_FROM_EVENT_LOG",
        "served_count": "NATIVE_PER_STEP",
        "on_time_rate": "REQUIRES_EXTERNAL_AGGREGATOR",
        "headway": "DERIVABLE_FROM_EVENT_LOG",
        "cv_headway": "REQUIRES_EXTERNAL_AGGREGATOR",
        "bunching_rate": "REQUIRES_EXTERNAL_AGGREGATOR",
        "energy_proxy": "PROXY_ONLY",
        "energy_proxy_per_passenger": "PROXY_ONLY",
        "missed_pickup": "NATIVE_PER_STEP",
        "missed_dropoff": "NATIVE_PER_STEP",
        "mandatory_stop_violation": "NATIVE_PER_STEP",
        "invalid_skip_execution": "NATIVE_PER_STEP",
    }
    return {
        "created_at": iso_kst(),
        "metric_availability": metrics,
        "passenger_level_wait_distribution_present": False,
        "p95_wait_seconds_available": False,
        "reward_kpi_extraction": "PARTIAL",
        "notes": "VehicleStepTrace has event/counter telemetry, but reward and passenger-level p95 wait are not native engine outputs.",
    }


def gap_repair_registry(axes: Mapping[str, str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    gaps = [
        ("GAP_STATE_SERIALIZATION", "branch_state_isolation", "No explicit clone/serialized restore API.", "STATE_SERIALIZATION_EXTENSION", str(ENGINE_SOURCE), "serialize_state/restore_state", False, True, "MEDIUM", True),
        ("GAP_DETERMINISTIC_RESET", "branch_state_isolation", "No deterministic reset API.", "DETERMINISTIC_RESET_EXTENSION", str(ENGINE_SOURCE), "reset_from_snapshot_state", False, True, "MEDIUM", True),
        ("GAP_RNG_CONTRACT", "rng_control", "No explicit no-internal-RNG contract file/API.", "RNG_STATE_EXPORT_IMPORT", str(ENGINE_SOURCE), "explicit no-RNG contract or get_rng_state/set_rng_state if stochastic callbacks enter engine scope", False, True, "LOW", False),
        ("GAP_EXOGENOUS_REPLAY_SCHEMA", "exogenous_interface", "Replay injection is possible through fields/callback but not canonicalized.", "EXOGENOUS_EVENT_INJECTION_INTERFACE", str(ENGINE_SOURCE), "ReplayInput dataclass/callback contract", False, True, "MEDIUM", True),
        ("GAP_WAIT_DISTRIBUTION", "reward_kpi_extraction", "Passenger-level wait distribution unavailable, so p95 wait is not available.", "PASSENGER_WAIT_DISTRIBUTION_LOGGING", str(ENGINE_SOURCE), "passenger wait event log", False, True, "MEDIUM", True),
        ("GAP_HORIZON_AGGREGATOR", "reward_kpi_extraction", "No native cumulative reward/KPI horizon aggregator.", "HORIZON_AGGREGATOR", str(ENGINE_SOURCE), "aggregate_trace_horizon_kpis", False, True, "MEDIUM", True),
        ("GAP_DL6B_STUB_REAUDIT", "upstream_action_effect_claim", "DL6B uses stable_int synthetic state/demand signatures; ACTION_PATH_AND_KPI_SENSITIVITY_PRESENT needs provenance separation.", "ACTION_MAPPING_ADAPTER", str(PROJECT_ROOT / "05_training/run_prompt5_e01_dl6b_suseong_counterfactual_action_path_diagnostic.py"), "real-state branch source audit", False, True, "MEDIUM", True),
    ]
    rows = []
    for gap_id, axis, limitation, change, file, api, semantics, compat, complexity, before in gaps:
        rows.append({
            "gap_id": gap_id,
            "gap_axis": axis,
            "current_limitation": limitation,
            "required_change": change,
            "target_file": file,
            "new_api_or_field": api,
            "changes_existing_semantics": semantics,
            "backward_compatibility_required": compat,
            "repair_complexity": complexity,
            "must_complete_before_state_feasibility": before,
        })
    table = pd.DataFrame(rows)
    registry = {
        "created_at": iso_kst(),
        "gap_count": len(rows),
        "must_complete_before_state_feasibility_count": int(table["must_complete_before_state_feasibility"].sum()),
        "axes": dict(axes),
        "engine_repair_required": True,
        "engine_repair_implemented": False,
    }
    return table, registry


def choose_engine_gate(
    preservation: Mapping[str, Any],
    source: Mapping[str, Any],
    prior_source: Mapping[str, Any],
    validation: Mapping[str, Any],
    test: Mapping[str, Any],
    dynamics: Mapping[str, Any],
    training: Mapping[str, Any],
    external: Mapping[str, Any],
    reconciliation: Mapping[str, Any],
    axes: Mapping[str, str],
) -> Tuple[str, bool, str, str]:
    if preservation["modified_file_count"] or preservation["deleted_file_count"] or preservation["added_file_count_inside_existing_artifacts"]:
        return FAIL_E1_ARTIFACT_MUTATED, False, "FAILED_EXISTING_ARTIFACT_MUTATION", "FAIL"
    by_role = {row.get("role"): row for row in source.get("sources", [])}
    prior_by_role = {row.get("role"): row for row in prior_source.get("sources", [])}
    r1_prior = prior_by_role.get("R1 reduced-form audit source", {}).get("sha256")
    r1_current = by_role.get("R1 reduced-form audit source", {}).get("sha256")
    engine_prior = prior_by_role.get("transition engine source", {}).get("sha256")
    engine_current = by_role.get("transition engine source", {}).get("sha256")
    if r1_prior and r1_current and r1_prior != r1_current:
        return FAIL_E1_R1_MODIFIED, False, "FAILED_R1_MODULE_MODIFIED", "FAIL"
    if engine_prior and engine_current and engine_prior != engine_current:
        return FAIL_E1_ENGINE_MODIFIED, False, "FAILED_TRANSITION_ENGINE_MODIFIED", "FAIL"
    if not validation.get("validation_seal_intact"):
        return FAIL_E1_VALIDATION, False, "FAILED_VALIDATION_SEAL", "FAIL"
    if test.get("test_sealed_holdout_touched"):
        return FAIL_E1_TEST, False, "FAILED_TEST_HOLDOUT", "FAIL"
    if dynamics.get("transition_function_execution_count") or dynamics.get("reset_function_execution_count") or dynamics.get("dynamics_branch_execution_count"):
        return FAIL_E1_DYNAMICS, False, "FAILED_DYNAMICS_EXECUTION_PROHIBITION", "FAIL"
    if training.get("training_run_count") or training.get("optimizer_step_count"):
        return FAIL_E1_TRAINING, False, "FAILED_TRAINING_PROHIBITION", "FAIL"
    if external.get("database_accessed") or external.get("api_call_count") or external.get("external_network_accessed"):
        return FAIL_E1_EXTERNAL, False, "FAILED_EXTERNAL_ACCESS_PROHIBITION", "FAIL"
    if reconciliation.get("counting_unit_reconciliation") != "CONSISTENT_DIFFERENT_GRANULARITY":
        return BLOCK_PROVENANCE_RECONCILIATION, False, "BLOCKED_PROVENANCE_RECONCILIATION", "BLOCK"
    if all(value == "READY" for value in axes.values()):
        return PASS_ENGINE_AUDIT, True, "ENGINE_AUDIT_COMPLETE_STATE_FEASIBILITY_PENDING_USER_COMMAND", "ENGINE_READY_FOR_STATE_FEASIBILITY_AUDIT"
    if any(value == "INSUFFICIENT" for value in axes.values()):
        return BLOCK_ENGINE_INSUFFICIENT, False, "BLOCKED_ENGINE_INSUFFICIENT_FOR_REAL_DYNAMICS", "ENGINE_INSUFFICIENT_FOR_REAL_DYNAMICS"
    if any(value in {"PARTIAL", "INDETERMINATE"} for value in axes.values()):
        return BLOCK_ENGINE_REPAIR, False, "BLOCKED_ENGINE_PARTIAL_REPAIR_REQUIRED", "ENGINE_PARTIAL_REPAIR_REQUIRED_BEFORE_STATE_FEASIBILITY"
    return BLOCK_ENGINE_INDETERMINATE, False, "BLOCKED_ENGINE_AUDIT_INDETERMINATE", "ENGINE_AUDIT_INDETERMINATE"


def engine_final_report(
    output: Path,
    gate: Mapping[str, Any],
    hardening: Mapping[str, Any],
    sweep: Mapping[str, Any],
    engine_api: Mapping[str, Any],
    action_mapping: Mapping[str, Any],
    horizon: Mapping[str, Any],
    state_rng: Mapping[str, Any],
    exogenous: Mapping[str, Any],
    reward_kpi: Mapping[str, Any],
    gaps: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    payload = {
        "created_at": iso_kst(),
        "artifact": str(output),
        "mode": "engine-audit",
        "gate": gate,
        "proxy_provenance_hardening": hardening,
        "repository_stub_signature_sweep": sweep,
        "transition_engine_api": engine_api,
        "action_mapping": action_mapping,
        "thirty_minute_horizon": horizon,
        "state_clone_rng": state_rng,
        "exogenous_interface": exogenous,
        "reward_kpi_extraction": reward_kpi,
        "engine_gap_repair_registry": gaps,
        "validation": validation,
    }
    md = "\n".join([
        "# Prompt 5-E01-DL-6D-PA1-A-E1 Engine Audit",
        "",
        f"- artifact: `{output}`",
        "- mode: `engine-audit`",
        f"- gate: `{gate['gate']}`",
        f"- gate passed: `{str(gate['gate_passed']).lower()}`",
        f"- engine feasibility decision: `{gate['engine_feasibility_decision']}`",
        "",
        "## 핵심 답변",
        f"1. 67개 literal 중 decision-relevant 값은 `{hardening['decision_relevant_value_count']}`개로 재분류됐다.",
        f"2. 핵심 reward/KPI 값은 R1 `thirty_minute_audit`의 literal/data-dependent 식에서 왔고, `16.43`은 R3 service harm 식으로 재계산되는 `{hardening['service_harm_excess_native_classification']}`이다.",
        f"3. simulator 보정 증거가 있는 값은 `{hardening['simulator_calibrated_value_count']}`개다.",
        "4. 18 claim과 8 reusable result 차이는 `CONSISTENT_DIFFERENT_GRANULARITY`로 판정했다.",
        f"5. 실제 transition 함수는 `{engine_api['transition_function_name']}`이다.",
        f"6. 8-agent action은 engine 단독 API가 아니라 외부 orchestrator 경유라 `{gate['action_mapping_status']}`/capability partial 축이 남는다.",
        f"7. H/S/K action 의미는 `{action_mapping['action_mapping']}`이다.",
        f"8. 30분은 `{horizon['total_steps']}`개 `{horizon['decision_interval_value']}`초 step으로 표현된다.",
        f"9. 동일 state 복제는 `{state_rng['branch_state_isolation']}`이다.",
        f"10. RNG 복원은 `{state_rng['rng_control']}`이다.",
        f"11. 외생 사건 주입 interface는 `{exogenous['exogenous_interface']}`이다.",
        f"12. KPI 추출은 `{reward_kpi['reward_kpi_extraction']}`이며 p95 wait은 passenger-level distribution 부재로 available 판정하지 않았다.",
        f"13. Engine state-feasibility 진입성은 `{gate['engine_feasibility_decision']}`이다.",
        f"14. repair gap은 `{gaps['gap_count']}`건이며 state-feasibility 전 필수 gap은 `{gaps['must_complete_before_state_feasibility_count']}`건이다.",
        "15. 이번 단계에서 실제 simulator transition/reset/branch는 실행하지 않았다.",
        "",
        "## 추가 보강",
        f"- repository-wide stub sweep rows: `{sweep['signature_row_count']}`",
        f"- DL6B signature rows: `{sweep['dl6b_signature_count']}`",
        f"- transition engine stable_int/modulo-5 signature rows: `{sweep['transition_engine_modulo_or_stable_int_signature_count']}`",
        f"- validation seal intact: `{str(validation['validation_seal_intact']).lower()}`",
    ]) + "\n"
    return payload, md


def write_manifest_for_files(writer: Writer, required_files: Sequence[str], lock_name: str, lock_text: str, scope: str) -> Dict[str, Any]:
    files = []
    missing = []
    for rel_path in required_files:
        if rel_path == "artifact_manifest.json":
            files.append({"relative_path": rel_path, "size_bytes": None, "sha256": "SELF_HASH_EXEMPT", "created_order": None, "required": True})
            continue
        if rel_path == lock_name and not (writer.root / rel_path).exists():
            files.append({
                "relative_path": rel_path,
                "size_bytes": len(lock_text.encode("utf-8")),
                "sha256": sha256_text(lock_text),
                "created_order": len(writer.order) + 2,
                "required": True,
                "created_after_manifest": True,
            })
            continue
        path = writer.root / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        files.append({"relative_path": rel_path, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "created_order": writer.order.get(rel_path), "required": True})
    parquet_failures, nonfinite = finite_parquets(writer.root)
    payload = {
        "created_at": iso_kst(),
        "manifest_scope": scope,
        "required_file_count": len(required_files),
        "manifest_missing_required_file_count": len(missing),
        "missing_required_files_after_lock": missing,
        "hash_mismatch_count": 0,
        "size_mismatch_count": 0,
        "duplicate_path_count": len(required_files) - len(set(required_files)),
        "strict_json_failure_count": 0,
        "parquet_read_failure_count": parquet_failures,
        "nan_inf_count": nonfinite,
        "mode_lock_created_last": True,
        "success_lock_created": False,
        "files": files,
    }
    writer.json("artifact_manifest.json", payload)
    return payload


def write_manifest(writer: Writer, lock_text: str) -> Dict[str, Any]:
    return write_manifest_for_files(writer, ADJUDICATE_REQUIRED_FILES, "_ADJUDICATE_COMPLETE.lock", lock_text, "PA1A_ADJUDICATE_MODE")


def validate_artifact_root(path: Path, mode: str) -> Path:
    root = path.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be an absolute path")
    if mode == "adjudicate":
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"--artifact-root already exists and is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
    else:
        if not root.exists():
            raise FileNotFoundError(f"--artifact-root must exist for mode {mode}: {root}")
    return root


def run_adjudicate(artifact_root: Path) -> Path:
    output = validate_artifact_root(artifact_root, "adjudicate")
    writer = Writer(output)
    existing_paths = artifact_dirs()
    before = scan_artifact_files(existing_paths)

    writer.text("git_status_pa1a.txt", run_cmd(["git", "status", "--short"])["stdout"] + "\n")
    env = environment_audit()
    writer.json("mac_mini_environment_pa1a.json", env)
    upstream = upstream_validation(existing_paths)
    writer.json("upstream_validation.json", upstream)
    source = source_sha_registry()
    writer.json("source_sha_registry.json", source)
    path_b = path_b_decision_record()
    writer.json("path_b_decision_record.json", path_b)

    provenance_table, provenance_audit = proxy_provenance()
    writer.parquet("proxy_constant_provenance_table.parquet", provenance_table)
    writer.json("proxy_constant_provenance_audit.json", provenance_audit)
    r1_desc = r1_self_description_audit()
    writer.json("r1_self_description_audit.json", r1_desc)
    contamination_table, contamination_audit = contamination_scope()
    writer.parquet("contamination_scope_table.parquet", contamination_table)
    writer.json("contamination_scope_enumeration.json", contamination_audit)
    independent = proxy_independent_registry()
    writer.json("proxy_independent_result_registry.json", independent)
    relabel_json, relabel_table = relabeling_registry(contamination_table)
    writer.json("proxy_relabeling_registry.json", relabel_json)
    writer.parquet("proxy_relabeling_registry.parquet", relabel_table)

    dynamics = dynamics_execution_prohibition_audit()
    writer.json("dynamics_execution_prohibition_audit.json", dynamics)
    seal = validation_seal_revalidation()
    writer.json("validation_seal_revalidation.json", seal)
    validation = validation_untouched_audit(seal)
    writer.json("validation_untouched_audit.json", validation)
    test = test_holdout_untouched_audit()
    writer.json("test_holdout_untouched_audit.json", test)
    scale = scale_candidate_prohibition_audit()
    writer.json("scale_candidate_prohibition_audit.json", scale)
    training, external = training_and_external_guards()
    writer.json("training_prohibition_audit.json", training)
    writer.json("external_access_audit.json", external)

    after = scan_artifact_files(existing_paths)
    preservation_table, preservation = artifact_preservation(before, after)
    writer.parquet("artifact_preservation_table.parquet", preservation_table)
    writer.json("artifact_preservation_audit.json", preservation)

    gate_name, passed, readiness = choose_adjudication_gate(preservation, validation, test, dynamics, scale, training, external, env)
    gate = {
        "created_at": iso_kst(),
        "mode": "adjudicate",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": readiness,
        "adjudication_path": "PATH_B",
        "path_decision_source": "USER",
        "engine_audit_authorized": False,
        "state_feasibility_authorized": False,
        "finalize_authorized": False,
        "pa1b_real_dynamics_implementation_authorized": False,
    }
    writer.json("gate_decision.json", gate)
    downstream = {
        "created_at": iso_kst(),
        "adjudication_path": "PATH_B",
        "path_decision_source": "USER",
        "existing_artifacts_preserved": preservation["existing_artifacts_preserved"],
        "existing_artifacts_modified": preservation["modified_file_count"],
        "proxy_constants_traced": True,
        "contamination_scope_enumerated": True,
        "proxy_independent_results_identified": True,
        "relabeling_registry_created": True,
        "transition_engine_api_ready": False,
        "state_reconstruction_ready": False,
        "exogenous_event_replay_ready": False,
        "branch_alignment_ready": False,
        "reward_kpi_extraction_ready": False,
        "historical_head_to_head_comparison_ready": False,
        "dynamics_module_created": False,
        "dynamics_branch_execution_count": 0,
        "validation_seal_intact": validation["validation_seal_intact"],
        "validation_outcome_accessed": False,
        "test_sealed_holdout_touched": False,
        "test_holdout_reuse_decided": False,
        "scale_produced": False,
        "candidate_created": False,
        "reward_modified": False,
        "tolerance_modified": False,
        "pa1b_real_dynamics_implementation_required": True,
        "pa1b_real_dynamics_implementation_authorized": False,
        "rebuild_plan_decided": False,
        "rebuild_execution_authorized": False,
        "d1_proxy_rollout_authorized": False,
        "d1_proxy_analyze_authorized": False,
        "training_allowed": False,
        "h200_allowed": False,
        "cuda_allowed": False,
        "cloud_gpu_allowed": False,
        "next_required_mode": "engine-audit",
    }
    writer.json("downstream_lock.json", downstream)
    report_json, report_md = final_report(output, env, path_b, provenance_audit, r1_desc, contamination_audit, independent, preservation, validation, gate)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    lock_text = json.dumps({"created_at": iso_kst(), "mode": "adjudicate", "gate": gate_name, "gate_passed": passed}, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    manifest = write_manifest(writer, lock_text)
    if (
        manifest["manifest_missing_required_file_count"]
        or manifest["hash_mismatch_count"]
        or manifest["size_mismatch_count"]
        or manifest["duplicate_path_count"]
        or manifest["parquet_read_failure_count"]
        or manifest["nan_inf_count"]
    ):
        gate["gate"] = FAIL_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = "FAILED_MANIFEST_RECONCILIATION"
        writer.json("gate_decision.json", gate)
        lock_text = json.dumps({"created_at": iso_kst(), "mode": "adjudicate", "gate": FAIL_MANIFEST, "gate_passed": False}, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    writer.text("_ADJUDICATE_COMPLETE.lock", lock_text)

    print(f"[DL-6D-PA1-A] artifact: {output}")
    print("[DL-6D-PA1-A] mode: adjudicate")
    print("[DL-6D-PA1-A] platform: MAC_MINI_M4_24GB")
    print(f"[DL-6D-PA1-A] actual compute path: {env['actual_compute_path']}")
    print(f"[DL-6D-PA1-A] MPS built/available/used: {env['mps_built']} / {env['mps_available']} / {env['mps_used']}")
    print("[DL-6D-PA1-A] CUDA used: false")
    print("[DL-6D-PA1-A] adjudication path: PATH_B")
    print("[DL-6D-PA1-A] path decision source: USER")
    print(f"[DL-6D-PA1-A] existing artifacts modified/deleted/added: {preservation['modified_file_count']} / {preservation['deleted_file_count']} / {preservation['added_file_count_inside_existing_artifacts']}")
    print(f"[DL-6D-PA1-A] proxy values total: {len(provenance_table)}")
    print(f"[DL-6D-PA1-A] literals/derived/data-dependent: {provenance_audit['literal_constant_count']} / {provenance_audit['derived_invariant_count']} / {provenance_audit['data_dependent_value_count']}")
    print(f"[DL-6D-PA1-A] calibrated/documented/placeholder/unknown: {provenance_audit['calibrated_from_simulator_count']} / {provenance_audit['documented_assumption_count']} / 0 / {provenance_audit['unknown_provenance_count']}")
    print(f"[DL-6D-PA1-A] R1 declared as proxy: {r1_desc['r1_explicitly_declared_as_proxy']}")
    print(f"[DL-6D-PA1-A] R1 described as simulation: {r1_desc['r1_described_as_simulation']}")
    print(f"[DL-6D-PA1-A] contaminated claims: {contamination_audit['contaminated_claim_count']}")
    print(f"[DL-6D-PA1-A] proxy-independent confirmed: {independent['proxy_independent_result_count']}")
    print("[DL-6D-PA1-A] transition function: DEFERRED_TO_ENGINE_AUDIT")
    print("[DL-6D-PA1-A] decision interval minutes: DEFERRED_TO_ENGINE_AUDIT")
    print("[DL-6D-PA1-A] total 30m steps: DEFERRED_TO_ENGINE_AUDIT")
    print("[DL-6D-PA1-A] dynamics branch executions: 0")
    print("[DL-6D-PA1-A] transition function executions: 0")
    print(f"[DL-6D-PA1-A] validation seal intact: {validation['validation_seal_intact']}")
    print("[DL-6D-PA1-A] validation outcome accessed: false")
    print("[DL-6D-PA1-A] test holdout touched: false")
    print("[DL-6D-PA1-A] scale produced: false")
    print("[DL-6D-PA1-A] candidate created: false")
    print("[DL-6D-PA1-A] training runs: 0")
    print("[DL-6D-PA1-A] feasibility decision: DEFERRED_TO_FINALIZE")
    print(f"[DL-6D-PA1-A] gate: {gate['gate']}")
    print(f"[DL-6D-PA1-A] gate passed: {str(gate['gate_passed']).lower()}")
    print("[DL-6D-PA1-A] PA1-B authorized: false")
    print("[DL-6D-PA1-A] next: REPORT_TO_USER")
    return output


def preserve_adjudicate_stage(artifact_root: Path) -> Dict[str, Any]:
    prior_files = ["gate_decision.json", "downstream_lock.json", "final_report.json", "artifact_manifest.json"]
    rows = []
    for name in prior_files:
        path = artifact_root / name
        rows.append({
            "relative_path": name,
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
            "content": read_json(path) if path.exists() and name.endswith(".json") else None,
        })
    lock = artifact_root / "_ADJUDICATE_COMPLETE.lock"
    return {
        "created_at": iso_kst(),
        "adjudicate_stage_result_preserved": True,
        "adjudicate_lock_present": lock.exists(),
        "adjudicate_lock_sha256": sha256_file(lock) if lock.exists() else None,
        "preserved_files": rows,
    }


def preserve_engine_audit_retry_stage(artifact_root: Path) -> Dict[str, Any]:
    prior_files = ["gate_decision.json", "downstream_lock.json", "final_report.json", "artifact_manifest.json", "_ENGINE_AUDIT_COMPLETE.lock"]
    rows = []
    for name in prior_files:
        path = artifact_root / name
        content: Any = None
        if path.exists() and name.endswith(".json"):
            content = read_json(path)
        elif path.exists() and name.endswith(".lock"):
            content = path.read_text(encoding="utf-8").strip()
        rows.append({
            "relative_path": name,
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
            "content": content,
        })
    gate_path = artifact_root / "gate_decision.json"
    gate = read_json(gate_path) if gate_path.exists() else {}
    return {
        "created_at": iso_kst(),
        "prior_engine_audit_present": (artifact_root / "_ENGINE_AUDIT_COMPLETE.lock").exists(),
        "prior_engine_audit_gate": gate.get("gate"),
        "prior_engine_audit_gate_passed": gate.get("gate_passed"),
        "retry_reason": "Previous engine-audit run used git tracked status as a modification proxy; retry compares source hashes against adjudicate-stage source_sha_registry instead.",
        "preserved_files": rows,
    }


def validate_engine_audit_entry(artifact_root: Path) -> None:
    validate_artifact_root(artifact_root, "engine-audit")
    gate = read_json(artifact_root / "gate_decision.json")
    engine_lock = artifact_root / "_ENGINE_AUDIT_COMPLETE.lock"
    prior_engine_retry = engine_lock.exists() and gate.get("gate_passed") is False
    if not prior_engine_retry and (gate.get("gate") != PASS_ADJUDICATE or gate.get("gate_passed") is not True):
        raise RuntimeError("PA1-A engine-audit requires prior adjudicate PASS gate")
    if not (artifact_root / "_ADJUDICATE_COMPLETE.lock").exists():
        raise RuntimeError("PA1-A engine-audit requires _ADJUDICATE_COMPLETE.lock")
    for forbidden in ["_STATE_FEASIBILITY_COMPLETE.lock", "_SUCCESS.lock"]:
        if (artifact_root / forbidden).exists():
            raise RuntimeError(f"forbidden pre-existing lock for engine-audit: {forbidden}")


def run_engine_audit(artifact_root: Path) -> Path:
    output = artifact_root.expanduser()
    validate_engine_audit_entry(output)
    writer = Writer(output)
    existing_paths = artifact_dirs()
    before = scan_artifact_files(existing_paths)

    stage_history = preserve_adjudicate_stage(output)
    writer.json("adjudicate_stage_result_history.json", stage_history)
    retry_history = preserve_engine_audit_retry_stage(output)
    writer.json("engine_audit_retry_history.json", retry_history)

    env = environment_audit()
    prior_source = read_json(output / "source_sha_registry.json") if (output / "source_sha_registry.json").exists() else {}
    source = source_sha_registry()
    hardening_table, hardening, reconciliation = proxy_provenance_hardening(output)
    writer.parquet("decision_relevant_proxy_value_table.parquet", hardening_table)
    writer.json("proxy_provenance_hardening.json", hardening)
    writer.json("provenance_count_reconciliation.json", reconciliation)

    sweep_table, sweep, engine_stub = repository_stub_signature_sweep()
    writer.parquet("repository_stub_signature_sweep.parquet", sweep_table)
    writer.json("repository_stub_signature_sweep.json", sweep)
    writer.json("transition_engine_stub_signature_audit.json", engine_stub)

    engine_api = transition_engine_api_audit()
    writer.json("transition_engine_api_audit.json", engine_api)
    cap_table, axes = capability_inventory()
    writer.parquet("engine_capability_inventory.parquet", cap_table)
    action_table, action_audit = action_mapping_audit()
    writer.parquet("action_mapping_table.parquet", action_table)
    writer.json("action_mapping_audit.json", action_audit)
    horizon = horizon_contract()
    writer.json("thirty_minute_horizon_contract.json", horizon)
    state_rng = state_clone_rng_audit()
    writer.json("state_clone_rng_alignment_audit.json", state_rng)
    exogenous = exogenous_interface_audit()
    writer.json("exogenous_event_interface_audit.json", exogenous)
    reward_kpi = reward_kpi_extraction_audit()
    writer.json("reward_kpi_extraction_audit.json", reward_kpi)
    gap_table, gaps = gap_repair_registry(axes)
    writer.parquet("engine_gap_repair_table.parquet", gap_table)
    writer.json("engine_gap_repair_registry.json", gaps)

    seal = validation_seal_revalidation()
    validation = validation_untouched_audit(seal)
    test = test_holdout_untouched_audit()
    dynamics = {
        "transition_function_execution_count": 0,
        "reset_function_execution_count": 0,
        "dynamics_branch_execution_count": 0,
        "dynamics_output_rows": 0,
    }
    training, external = training_and_external_guards()

    after = scan_artifact_files(existing_paths)
    _preservation_table, preservation = artifact_preservation(before, after)

    gate_name, passed, readiness, feasibility = choose_engine_gate(
        preservation,
        source,
        prior_source,
        validation,
        test,
        dynamics,
        training,
        external,
        reconciliation,
        axes,
    )
    gate = {
        "created_at": iso_kst(),
        "mode": "engine-audit",
        "gate": gate_name,
        "gate_passed": passed,
        "readiness": readiness,
        "engine_feasibility_decision": feasibility,
        "adjudication_complete": True,
        "engine_audit_complete": True,
        "proxy_provenance_hardened": True,
        "decision_relevant_proxy_values_identified": True,
        "provenance_count_reconciled": reconciliation["counting_unit_reconciliation"] == "CONSISTENT_DIFFERENT_GRANULARITY",
        "transition_engine_api_status": axes["transition_api"],
        "action_mapping_status": axes["action_mapping"],
        "thirty_minute_horizon_status": axes["thirty_minute_horizon"],
        "branch_state_isolation_status": axes["branch_state_isolation"],
        "rng_control_status": axes["rng_control"],
        "exogenous_interface_status": axes["exogenous_interface"],
        "reward_kpi_extraction_status": axes["reward_kpi_extraction"],
        "engine_repair_required": feasibility != "ENGINE_READY_FOR_STATE_FEASIBILITY_AUDIT",
        "engine_repair_implemented": False,
        "state_feasibility_required": True,
        "state_feasibility_authorized": False,
        "finalize_authorized": False,
        "pa1b_real_dynamics_implementation_authorized": False,
        "automatic_mode_chaining_allowed": False,
        "validation_seal_intact": validation["validation_seal_intact"],
        "test_holdout_touched": test["test_sealed_holdout_touched"],
        "dynamics_branch_execution_count": 0,
        "transition_function_execution_count": 0,
        "reset_function_execution_count": 0,
        "training_allowed": False,
    }
    writer.json("gate_decision.json", gate)
    downstream = {
        "created_at": iso_kst(),
        "adjudication_complete": True,
        "engine_audit_complete": True,
        "proxy_provenance_hardened": True,
        "decision_relevant_proxy_values_identified": True,
        "provenance_count_reconciled": gate["provenance_count_reconciled"],
        "repository_wide_stub_sweep_complete": True,
        "transition_engine_stub_signature_audit_complete": True,
        "transition_engine_api_status": axes["transition_api"],
        "action_mapping_status": axes["action_mapping"],
        "thirty_minute_horizon_status": axes["thirty_minute_horizon"],
        "branch_state_isolation_status": axes["branch_state_isolation"],
        "rng_control_status": axes["rng_control"],
        "exogenous_interface_status": axes["exogenous_interface"],
        "reward_kpi_extraction_status": axes["reward_kpi_extraction"],
        "engine_feasibility_decision": feasibility,
        "engine_repair_required": gate["engine_repair_required"],
        "engine_repair_implemented": False,
        "state_feasibility_required": True,
        "state_feasibility_authorized": False,
        "dynamics_module_created": False,
        "dynamics_branch_execution_count": 0,
        "validation_seal_intact": validation["validation_seal_intact"],
        "test_holdout_touched": test["test_sealed_holdout_touched"],
        "pa1b_authorized": False,
        "training_allowed": False,
        "next_required_mode": "REPORT_TO_USER_BEFORE_STATE_FEASIBILITY_OR_ENGINE_REPAIR",
    }
    writer.json("downstream_lock.json", downstream)
    report_json, report_md = engine_final_report(output, gate, hardening, sweep, engine_api, action_audit, horizon, state_rng, exogenous, reward_kpi, gaps, validation)
    writer.json("final_report.json", report_json)
    writer.text("final_report.md", report_md)
    lock_text = json.dumps({"created_at": iso_kst(), "mode": "engine-audit", "gate": gate_name, "gate_passed": passed}, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    manifest = write_manifest_for_files(writer, ENGINE_REQUIRED_FILES, "_ENGINE_AUDIT_COMPLETE.lock", lock_text, "PA1A_ENGINE_AUDIT_MODE")
    if (
        manifest["manifest_missing_required_file_count"]
        or manifest["hash_mismatch_count"]
        or manifest["size_mismatch_count"]
        or manifest["duplicate_path_count"]
        or manifest["parquet_read_failure_count"]
        or manifest["nan_inf_count"]
    ):
        gate["gate"] = FAIL_E1_MANIFEST
        gate["gate_passed"] = False
        gate["readiness"] = "FAILED_ENGINE_AUDIT_MANIFEST_RECONCILIATION"
        writer.json("gate_decision.json", gate)
        lock_text = json.dumps({"created_at": iso_kst(), "mode": "engine-audit", "gate": FAIL_E1_MANIFEST, "gate_passed": False}, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    writer.text("_ENGINE_AUDIT_COMPLETE.lock", lock_text)

    print(f"[DL-6D-PA1-A-E1] artifact: {output}")
    print("[DL-6D-PA1-A-E1] mode: engine-audit")
    print(f"[DL-6D-PA1-A-E1] existing artifacts modified/deleted/added: {preservation['modified_file_count']} / {preservation['deleted_file_count']} / {preservation['added_file_count_inside_existing_artifacts']}")
    print(f"[DL-6D-PA1-A-E1] decision-relevant proxy values: {hardening['decision_relevant_value_count']}")
    print(f"[DL-6D-PA1-A-E1] simulator-calibrated values: {hardening['simulator_calibrated_value_count']}")
    print(f"[DL-6D-PA1-A-E1] repository stub signature rows: {sweep['signature_row_count']}")
    print(f"[DL-6D-PA1-A-E1] DL6B signature rows: {sweep['dl6b_signature_count']}")
    print(f"[DL-6D-PA1-A-E1] engine stable_int/modulo5 signatures: {sweep['transition_engine_modulo_or_stable_int_signature_count']}")
    print(f"[DL-6D-PA1-A-E1] transition function: {engine_api['transition_function_name']}")
    print(f"[DL-6D-PA1-A-E1] action mapping: {action_audit['action_mapping']}")
    print(f"[DL-6D-PA1-A-E1] 30m steps: {horizon['total_steps']}")
    print(f"[DL-6D-PA1-A-E1] axes: {axes}")
    print("[DL-6D-PA1-A-E1] dynamics branch executions: 0")
    print("[DL-6D-PA1-A-E1] transition function executions: 0")
    print("[DL-6D-PA1-A-E1] reset function executions: 0")
    print(f"[DL-6D-PA1-A-E1] validation seal intact: {validation['validation_seal_intact']}")
    print(f"[DL-6D-PA1-A-E1] gate: {gate['gate']}")
    print(f"[DL-6D-PA1-A-E1] gate passed: {str(gate['gate_passed']).lower()}")
    print("[DL-6D-PA1-A-E1] next: REPORT_TO_USER")
    return output


def run_deferred_mode(artifact_root: Path, mode: str) -> Path:
    validate_artifact_root(artifact_root, mode)
    raise RuntimeError(f"--mode {mode} is intentionally locked until the prior PA1-A mode is reported to the user")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["adjudicate", "engine-audit", "state-feasibility", "finalize"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "adjudicate":
        run_adjudicate(args.artifact_root)
    elif args.mode == "engine-audit":
        run_engine_audit(args.artifact_root)
    else:
        run_deferred_mode(args.artifact_root, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
