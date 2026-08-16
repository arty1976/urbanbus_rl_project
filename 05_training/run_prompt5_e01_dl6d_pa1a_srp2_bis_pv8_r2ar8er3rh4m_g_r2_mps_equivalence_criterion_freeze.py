#!/usr/bin/env python3
"""H4M-G-R2 MPS numerical equivalence criterion selection and freeze.

This stage freezes the MPS-specific numerical equivalence criterion for
instrumentation OFF↔ON preflight checks. It does not repair instrumentation,
rerun H4M-H, run full training, force deterministic algorithms, change dtype,
adopt CPU fallback, open TEST6/validation, or push to GitHub.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-G-R2"
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_G_R2_MPS_NUMERICAL_EQUIVALENCE_CRITERION_SELECTION_AND_FREEZE_COMPLETE"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_G_R2_MPS_NUMERICAL_EQUIVALENCE_CRITERION_SELECTION_AND_FREEZE_FAILED"
PASS_DECISION = "PV8_MPS_NUMERICAL_EQUIVALENCE_CRITERION_FROZEN_READY_FOR_H4M_H_PREFLIGHT_RERUN"
NEXT_GATE = "H4M-H-RERUN_FRESH_INSTRUMENTED_CREDIT_DIAGNOSTIC_RETRAINING"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name

R1_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_g_r1_mps_numerical_reproducibility_audit_20260815_144716+0900"
H4M_H_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_h_fresh_instrumented_credit_diagnostic_retraining_20260815_132209+0900"
H4M_G_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_g_instrumentation_equivalence_validation_20260815_122304+0900"
H4M_F_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_f_credit_trace_instrumentation_selection_freeze_20260815_111506+0900"
H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"

R1_SCRIPT = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_r1_mps_numerical_reproducibility_audit.py"

EXPECTED = {
    "r1_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_G_R1_MPS_NUMERICAL_REPRODUCIBILITY_AND_INSTRUMENTATION_EQUIVALENCE_AUDIT_COMPLETE",
    "r1_decision": "MPS_VARIABILITY_WITH_INSTRUMENTATION_WITHIN_ENVELOPE",
    "r1_exact_next_gate": "H4M-G-R2_MPS_EQUIVALENCE_CRITERION_SELECTION_AND_FREEZE",
    "h4m_f_contract_sha256": "9b95d0dc46459eedd2cf51e85789407be247e9db1e23e3a52a5bbae9c6216ffe",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha256": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
}

ORDERED_PIPELINE = [
    "sampled_actions",
    "raw_rewards",
    "value_t",
    "next_value_t",
    "td_delta",
    "raw_gae",
    "normalized_advantage",
    "return_target",
    "old_log_prob",
    "ppo_ratio_surrogate",
    "parameter_update",
    "final_checkpoint",
]

FLOAT_METRICS = [
    {
        "name": "value_t",
        "source": ("value_t",),
        "gate_dimensions": ["max_abs_diff"],
        "diagnostic_dimensions": ["mean_abs_diff", "max_rel_diff"],
        "exact_vs_numerical_requirement": "numerical_mps_envelope_sha_mismatch_allowed_finite_required",
        "rationale": "First MPS divergence appears at critic value_t even without instrumentation.",
    },
    {
        "name": "next_value_t",
        "source": ("next_value_t",),
        "gate_dimensions": ["max_abs_diff"],
        "diagnostic_dimensions": ["mean_abs_diff", "max_rel_diff"],
        "exact_vs_numerical_requirement": "numerical_mps_envelope_sha_mismatch_allowed_finite_required",
        "rationale": "next_value_t is the one-step bootstrapped critic value paired with value_t.",
    },
    {
        "name": "td_delta",
        "source": ("td_delta",),
        "gate_dimensions": ["max_abs_diff"],
        "diagnostic_dimensions": ["mean_abs_diff", "max_rel_diff"],
        "exact_vs_numerical_requirement": "numerical_mps_envelope_sha_mismatch_allowed_finite_required",
        "rationale": "TD delta is downstream of value_t/next_value_t and reward.",
    },
    {
        "name": "raw_gae",
        "source": ("raw_gae",),
        "gate_dimensions": ["max_abs_diff"],
        "diagnostic_dimensions": ["mean_abs_diff", "max_rel_diff"],
        "exact_vs_numerical_requirement": "numerical_mps_envelope_sha_mismatch_allowed_finite_required",
        "rationale": "Raw GAE accumulates TD deltas over the controlled horizon.",
    },
    {
        "name": "normalized_advantage",
        "source": ("normalized_advantage",),
        "gate_dimensions": ["max_abs_diff"],
        "diagnostic_dimensions": ["mean_abs_diff", "max_rel_diff"],
        "exact_vs_numerical_requirement": "numerical_mps_envelope_sha_mismatch_allowed_finite_required",
        "rationale": "Normalized advantage is the actor-credit input after frozen normalization.",
    },
    {
        "name": "return_target",
        "source": ("return_target",),
        "gate_dimensions": ["max_abs_diff"],
        "diagnostic_dimensions": ["mean_abs_diff", "max_rel_diff"],
        "exact_vs_numerical_requirement": "numerical_mps_envelope_sha_mismatch_allowed_finite_required",
        "rationale": "Return target is the critic target derived from value/GAE.",
    },
    {
        "name": "old_log_prob",
        "source": ("old_log_prob",),
        "gate_dimensions": ["max_abs_diff"],
        "diagnostic_dimensions": ["mean_abs_diff", "max_rel_diff"],
        "exact_vs_numerical_requirement": "numerical_mps_envelope_sha_mismatch_allowed_finite_required",
        "rationale": "old_log_prob is part of the frozen PPO sample input fingerprint.",
    },
    {
        "name": "ppo_ratio_surrogate",
        "source": ("ppo_inputs",),
        "gate_dimensions": ["max_abs_diff"],
        "diagnostic_dimensions": ["mean_abs_diff", "max_rel_diff"],
        "exact_vs_numerical_requirement": "numerical_mps_envelope_sha_mismatch_allowed_finite_required",
        "rationale": "R1 ppo_inputs fingerprint contains ratio, unclipped/clipped/effective surrogate, entropy, and related log-prob fields.",
    },
    {
        "name": "actor_parameter_update",
        "source": ("parameter_update", "actor"),
        "gate_dimensions": ["max_abs_diff"],
        "diagnostic_dimensions": ["mean_abs_diff", "max_rel_diff"],
        "relative_diff_policy": "record_only_near_zero_denominator_parameter_diagnostic",
        "exact_vs_numerical_requirement": "numerical_mps_envelope_sha_mismatch_allowed_finite_required",
        "rationale": "Actor update is sensitive to tiny upstream value/advantage drift; max_rel is recorded but not gated because near-zero parameters make it unstable.",
    },
    {
        "name": "critic_parameter_update",
        "source": ("parameter_update", "critic"),
        "gate_dimensions": ["max_abs_diff"],
        "diagnostic_dimensions": ["mean_abs_diff", "max_rel_diff"],
        "relative_diff_policy": "record_only_near_zero_denominator_parameter_diagnostic",
        "exact_vs_numerical_requirement": "numerical_mps_envelope_sha_mismatch_allowed_finite_required",
        "rationale": "Critic update is directly downstream of value/return-target drift; max_rel is recorded but not gated because near-zero parameters make it unstable.",
    },
    {
        "name": "gatv2_parameter_update",
        "source": ("parameter_update", "gatv2"),
        "gate_dimensions": ["max_abs_diff"],
        "diagnostic_dimensions": ["mean_abs_diff", "max_rel_diff"],
        "relative_diff_policy": "record_only_near_zero_denominator_parameter_diagnostic",
        "exact_vs_numerical_requirement": "numerical_mps_envelope_sha_mismatch_allowed_finite_required",
        "rationale": "Shared encoder update accumulates actor/critic gradient sensitivity; max_rel is recorded but not gated because near-zero parameters make it unstable.",
    },
]

EXACT_INVARIANTS = [
    "sampled_actions",
    "legal_masks",
    "reward_v2_sequence",
    "sample_uid_sequence",
    "targets",
    "rng_lineage",
    "shadow_contamination",
]

ADDITIONAL_REPEAT_BATCHES = 2
MIN_COMPARISON_ROWS_PER_CLASS = 3

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_r1_variability_evidence.json",
    "03_additional_repeats_if_required.json",
    "04_metric_level_equivalence_criteria.json",
    "05_exact_invariant_contract.json",
    "06_mps_equivalence_decision_rule.json",
    "07_h4mh_preflight_release_contract.json",
    "08_h4m_g_r2_gate_matrix.json",
    "mps_numerical_equivalence_contract.json",
    "final_report.md",
    "manifest.json",
]


def kst_now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n"


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=jsonable)


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def git_bool(args: Sequence[str]) -> bool:
    return git_run(args, check=False).returncode == 0


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    source = SOURCE_REL.as_posix()
    latest = git_run(["log", "-1", "--format=%H", "--", source]).stdout.strip()
    head_files = [line.strip() for line in git_run(["show", "--name-only", "--pretty=format:", "HEAD"]).stdout.splitlines() if line.strip()]
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": git_run(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip(),
        "git_commit": head,
        "h4m_g_r2_source_git_commit": head,
        "h4m_g_r2_source_path": source,
        "h4m_g_r2_source_sha256": sha256_file(PROJECT_ROOT / source),
        "h4m_g_r2_source_present_in_head": git_bool(["cat-file", "-e", f"HEAD:{source}"]),
        "h4m_g_r2_source_latest_commit": latest,
        "h4m_g_r2_source_no_uncommitted_diff_vs_head": git_bool(["diff", "--quiet", "--", source]),
        "h4m_g_r2_source_no_staged_diff_vs_head": git_bool(["diff", "--cached", "--quiet", "--", source]),
        "head_commit_files": head_files,
        "local_source_only_commit_created_before_audit": latest == head and head_files == [source],
        "post_commit_provenance_gate_passed": latest == head
        and head_files == [source]
        and git_bool(["diff", "--quiet", "--", source])
        and git_bool(["diff", "--cached", "--quiet", "--", source]),
        "status_short": git_run(["status", "--short"]).stdout,
        "github_push_performed": False,
    }


def r1_comparison_records() -> Dict[str, Dict[str, Any]]:
    return {
        "R1_OFF_OFF": {"comparison_class": "OFF_OFF", "details": read_json(R1_ROOT / "04_off_off_reproducibility.json")["details"]},
        "R1_ON_ON": {"comparison_class": "ON_ON", "details": read_json(R1_ROOT / "05_on_on_reproducibility.json")["details"]},
        "R1_OFF_ON": {"comparison_class": "OFF_ON", "details": read_json(R1_ROOT / "06_off_on_variability_comparison.json")["details"]},
    }


def metric_from_comparison(comparison: Mapping[str, Any], metric_spec: Mapping[str, Any]) -> Mapping[str, Any]:
    node: Any = comparison["stage_metrics"]
    for key in metric_spec["source"]:
        node = node[key]
    return node


def metric_pick(metric: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sha_match": bool(metric.get("sha_match")),
        "max_abs_diff": float(metric.get("max_abs_diff", 0.0) or 0.0),
        "mean_abs_diff": float(metric.get("mean_abs_diff", 0.0) or 0.0),
        "max_rel_diff": float(metric.get("max_rel_diff", 0.0) or 0.0),
        "differing_element_count": int(metric.get("differing_element_count", 0) or 0),
        "finite": bool(metric.get("finite", True)),
    }


def comparison_stage_pick(comparison: Mapping[str, Any], stage: str) -> Dict[str, Any]:
    return metric_pick(comparison["stage_metrics"][stage])


def r1_variability_evidence(created_at: str) -> Dict[str, Any]:
    r1_gate = read_json(R1_ROOT / "10_gate_matrix.json")
    r1_binding = read_json(R1_ROOT / "01_authoritative_binding.json")
    r1_runtime = read_json(R1_ROOT / "02_mps_runtime_environment.json")
    r1_critic = read_json(R1_ROOT / "03_critic_forward_reproducibility.json")
    r1_root_cause = read_json(R1_ROOT / "09_root_cause_classification.json")
    r1_rng_shadow = read_json(R1_ROOT / "08_rng_shadow_integrity.json")
    comparisons = r1_comparison_records()
    return {
        "stage": STAGE,
        "created_at": created_at,
        "r1_artifact_root": str(R1_ROOT),
        "r1_gate": r1_gate.get("gate"),
        "r1_decision": r1_gate.get("decision"),
        "r1_exact_next_gate": r1_gate.get("exact_next_gate"),
        "r1_source_commit": r1_binding.get("source_provenance", {}).get("h4m_g_r1_source_git_commit"),
        "r1_runtime_device": {
            "torch_version": r1_runtime.get("torch_version"),
            "mps_built": r1_runtime.get("mps_built"),
            "mps_available": r1_runtime.get("mps_available"),
            "mps_device_test": r1_runtime.get("mps_device_test"),
            "deterministic_algorithms_enabled": r1_runtime.get("deterministic_algorithms_enabled"),
            "deterministic_mode_forced_by_audit": r1_runtime.get("deterministic_mode_forced_by_audit"),
            "cpu_fallback_adopted": r1_runtime.get("cpu_fallback_adopted"),
            "dtype_change_applied": r1_runtime.get("dtype_change_applied"),
            "sw_vers": r1_runtime.get("sw_vers"),
        },
        "critic_forward_reproducibility": {
            "same_process_all_value_t_sha_match": r1_critic.get("same_process", {}).get("all_value_t_sha_match"),
            "same_process_all_next_value_t_sha_match": r1_critic.get("same_process", {}).get("all_next_value_t_sha_match"),
            "fresh_process_value_t_sha_match": r1_critic.get("fresh_process", {}).get("value_t_diff", {}).get("sha_match"),
            "fresh_process_next_value_t_sha_match": r1_critic.get("fresh_process", {}).get("next_value_t_diff", {}).get("sha_match"),
            "finite": r1_critic.get("finite"),
        },
        "root_cause": r1_root_cause,
        "rng_shadow_integrity": {
            "rng_initial_state_reproducible": r1_rng_shadow.get("rng_initial_state_reproducible"),
            "shadow_contamination": r1_rng_shadow.get("shadow_contamination"),
            "integrity_passed": r1_rng_shadow.get("integrity_passed"),
        },
        "comparison_summaries": {
            name: {
                "comparison_class": row["comparison_class"],
                "first_divergence": row["details"].get("first_divergence"),
                "sampled_actions": comparison_stage_pick(row["details"], "sampled_actions"),
                "raw_rewards": comparison_stage_pick(row["details"], "raw_rewards"),
                "value_t": metric_pick(metric_from_comparison(row["details"], FLOAT_METRICS[0])),
                "actor_parameter_update": metric_pick(metric_from_comparison(row["details"], FLOAT_METRICS[8])),
                "critic_parameter_update": metric_pick(metric_from_comparison(row["details"], FLOAT_METRICS[9])),
                "gatv2_parameter_update": metric_pick(metric_from_comparison(row["details"], FLOAT_METRICS[10])),
            }
            for name, row in comparisons.items()
        },
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any], r1_evidence: Mapping[str, Any]) -> Dict[str, Any]:
    r1_gate = read_json(R1_ROOT / "10_gate_matrix.json")
    r1_binding = read_json(R1_ROOT / "01_authoritative_binding.json")
    r1_runtime = read_json(R1_ROOT / "02_mps_runtime_environment.json")
    h4mf_contract_path = H4M_F_ROOT / "13_h4m_f_credit_trace_instrumentation_contract.json"
    h4mb_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    upstream = read_json(h4mf_contract_path).get("authoritative_upstream_SHA_bindings", {})
    checks = {
        "r1_gate_match": r1_gate.get("gate") == EXPECTED["r1_gate"],
        "r1_decision_match": r1_gate.get("decision") == EXPECTED["r1_decision"],
        "r1_next_gate_match": r1_gate.get("exact_next_gate") == EXPECTED["r1_exact_next_gate"],
        "r1_source_commit_bound": bool(r1_binding.get("source_provenance", {}).get("h4m_g_r1_source_git_commit")),
        "r1_runtime_mps_available": r1_runtime.get("mps_available") is True,
        "r1_runtime_device_exact_bound": str(r1_runtime.get("mps_device_test", {}).get("tensor_device", "")).startswith("mps"),
        "r1_no_deterministic_forced": r1_runtime.get("deterministic_mode_forced_by_audit") is False,
        "r1_no_cpu_fallback": r1_runtime.get("cpu_fallback_adopted") is False,
        "r1_no_dtype_change": r1_runtime.get("dtype_change_applied") is False,
        "h4m_f_contract_sha_match": sha256_file(h4mf_contract_path) == EXPECTED["h4m_f_contract_sha256"],
        "h4m_b_schedule_sha_match": h4mb_schedule.get("extended_training_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "reward_v2_sha_match": upstream.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "h4g_runtime_sha_match": upstream.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha256"],
        "r3_split_sha_match": upstream.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "zero_loss_adapter_sha_match": upstream.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "h4m_g_r2_source_commit_frozen": provenance.get("post_commit_provenance_gate_passed") is True,
        "r1_raw_evidence_directly_bound": bool(r1_evidence.get("comparison_summaries")),
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {
            "h4m_g_r1": str(R1_ROOT),
            "h4m_h": str(H4M_H_ROOT),
            "h4m_g": str(H4M_G_ROOT),
            "h4m_f": str(H4M_F_ROOT),
            "h4m_b": str(H4M_B_ROOT),
        },
        "source_provenance": dict(provenance),
        "r1_bindings": {
            "gate": r1_gate.get("gate"),
            "decision": r1_gate.get("decision"),
            "exact_next_gate": r1_gate.get("exact_next_gate"),
            "source_commit": r1_binding.get("source_provenance", {}).get("h4m_g_r1_source_git_commit"),
            "source_sha256": r1_binding.get("source_provenance", {}).get("h4m_g_r1_source_sha256"),
            "runtime_device": r1_evidence.get("r1_runtime_device"),
        },
        "instrumentation_contract_sha256": sha256_file(h4mf_contract_path),
        "authoritative_upstream_sha_bindings": {
            "reward_v2_sha256": upstream.get("reward_v2_sha256"),
            "h4g_runtime_sha256": upstream.get("h4g_runtime_sha256"),
            "r3_split_sha256": upstream.get("r3_split_sha256"),
            "zero_loss_adapter_sha256": upstream.get("zero_loss_adapter_sha256"),
            "h4m_b_schedule_sha256": h4mb_schedule.get("extended_training_schedule_sha256"),
        },
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "validation_executed": False,
        "test6_status": "SEALED_NOT_OPENED",
        "github_push_performed": False,
    }


def initial_evidence_gaps(comparisons: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows = collect_metric_rows(comparisons)
    gaps: List[Dict[str, Any]] = []
    grouped = group_metric_rows(rows)
    for spec in FLOAT_METRICS:
        name = spec["name"]
        within = grouped[name]["within"]
        offon = grouped[name]["off_on"]
        natural = max_or_zero(row["max_abs_diff"] for row in within)
        candidate = max_or_zero(row["max_abs_diff"] for row in offon)
        if len(within) < MIN_COMPARISON_ROWS_PER_CLASS * 2:
            gaps.append({
                "metric": name,
                "reason": "R1 has only one OFF↔OFF and one ON↔ON row; metric-level envelope is too thin for freeze.",
                "within_mode_rows": len(within),
                "off_on_rows": len(offon),
            })
        if candidate > natural:
            gaps.append({
                "metric": name,
                "reason": "R1 OFF↔ON max_abs exceeds the single-pair within-mode max, requiring minimal repeats before freezing.",
                "r1_within_mode_max_abs": natural,
                "r1_off_on_max_abs": candidate,
            })
    return gaps


def run_additional_repeats(created_at: str, artifact_root: Path, required: bool, gaps: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not required:
        return {
            "stage": STAGE,
            "created_at": created_at,
            "required": False,
            "performed": False,
            "reason": "R1 evidence sufficient; no additional repeats needed.",
            "comparison_records": {},
            "trial_summaries": {},
        }
    if not torch.backends.mps.is_available():
        return {
            "stage": STAGE,
            "created_at": created_at,
            "required": True,
            "performed": False,
            "blocked_reason": "MPS_UNAVAILABLE_FOR_REQUIRED_MINIMAL_REPEATS",
            "initial_evidence_gaps": list(gaps),
            "comparison_records": {},
            "trial_summaries": {},
        }
    r1 = import_module(R1_SCRIPT, f"h4m_g_r2_r1_{time.time_ns()}")
    h4mg, h4mh = r1.load_h4_modules()
    device = torch.device("mps")
    comparison_records: Dict[str, Dict[str, Any]] = {}
    trial_summaries: Dict[str, Dict[str, Any]] = {}
    for repeat_index in range(1, ADDITIONAL_REPEAT_BATCHES + 1):
        prefix = f"R2_REP{repeat_index}"
        trials = {
            f"{prefix}_OFF_A": r1.run_trial(f"{prefix}_OFF_A", "OFF", h4mg, h4mh, created_at, artifact_root, device),
            f"{prefix}_OFF_B": r1.run_trial(f"{prefix}_OFF_B", "OFF", h4mg, h4mh, created_at, artifact_root, device),
            f"{prefix}_ON_A": r1.run_trial(f"{prefix}_ON_A", "ON", h4mg, h4mh, created_at, artifact_root, device),
            f"{prefix}_ON_B": r1.run_trial(f"{prefix}_ON_B", "ON", h4mg, h4mh, created_at, artifact_root, device),
        }
        for label, trial in trials.items():
            trial_summaries[label] = summarize_trial(r1, trial)
        comparisons = {
            f"{prefix}_OFF_OFF": ("OFF_OFF", r1.compare_trials(trials[f"{prefix}_OFF_A"], trials[f"{prefix}_OFF_B"], f"{prefix}_OFF↔OFF")),
            f"{prefix}_ON_ON": ("ON_ON", r1.compare_trials(trials[f"{prefix}_ON_A"], trials[f"{prefix}_ON_B"], f"{prefix}_ON↔ON")),
            f"{prefix}_OFF_ON": ("OFF_ON", r1.compare_trials(trials[f"{prefix}_OFF_A"], trials[f"{prefix}_ON_A"], f"{prefix}_OFF↔ON")),
        }
        for name, (comparison_class, details) in comparisons.items():
            comparison_records[name] = {"comparison_class": comparison_class, "details": details}
    return {
        "stage": STAGE,
        "created_at": created_at,
        "required": True,
        "performed": True,
        "repeat_batches": ADDITIONAL_REPEAT_BATCHES,
        "controlled_scope": {
            "seed": 1,
            "outer_cycle": 1,
            "device": "mps",
            "same_initialization_rng_data_order": True,
            "full_training_executed": False,
        },
        "reason": "R1 value-level root-cause classification was sufficient, but metric-level parameter/update criteria require at least three empirical rows per comparison class; two additional batches are the minimal extension from R1's one row.",
        "initial_evidence_gaps": list(gaps),
        "comparison_records": {
            name: summarize_comparison(row["comparison_class"], row["details"])
            for name, row in comparison_records.items()
        },
        "trial_summaries": trial_summaries,
        "_comparison_records_for_internal_use": comparison_records,
    }


def summarize_trial(r1_module: Any, trial: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "label": trial.get("label"),
        "mode": trial.get("mode"),
        "initial_rng": trial.get("initial_rng"),
        "post_rollout_pre_update_rng": trial.get("post_rollout_pre_update_rng"),
        "post_update_pre_shadow_rng": trial.get("post_update_pre_shadow_rng"),
        "post_shadow_rng": trial.get("post_shadow_rng"),
        "initial_model_hashes": trial.get("initial_model_hashes"),
        "final_model_hashes": trial.get("final_model_hashes"),
        "final_optimizer_hashes": trial.get("final_optimizer_hashes"),
        "parameter_delta_sha256": canonical_sha(trial.get("parameter_delta")),
        "hashes": trial.get("hashes"),
        "ppo_fingerprint_row_count": trial.get("ppo_fingerprint_row_count"),
        "finite_loss_passed": trial.get("finite_loss_passed"),
        "shadow_summary": r1_module.shadow_summary(trial),
    }


def summarize_comparison(comparison_class: str, comparison: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "comparison_class": comparison_class,
        "left": comparison.get("left"),
        "right": comparison.get("right"),
        "first_divergence": comparison.get("first_divergence"),
        "rng": comparison.get("rng"),
        "shadow": comparison.get("shadow"),
        "sampled_actions": comparison_stage_pick(comparison, "sampled_actions"),
        "raw_rewards": comparison_stage_pick(comparison, "raw_rewards"),
        "float_metrics": {
            spec["name"]: metric_pick(metric_from_comparison(comparison, spec))
            for spec in FLOAT_METRICS
        },
    }


def collect_metric_rows(comparisons: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for comparison_id, row in comparisons.items():
        comparison_class = row["comparison_class"]
        details = row["details"]
        for spec in FLOAT_METRICS:
            metric = metric_pick(metric_from_comparison(details, spec))
            rows.append({
                "comparison_id": comparison_id,
                "comparison_class": comparison_class,
                "metric": spec["name"],
                **metric,
            })
    return rows


def group_metric_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, List[Mapping[str, Any]]]]:
    grouped: Dict[str, Dict[str, List[Mapping[str, Any]]]] = {
        spec["name"]: {"within": [], "off_on": []}
        for spec in FLOAT_METRICS
    }
    for row in rows:
        target = "off_on" if row["comparison_class"] == "OFF_ON" else "within"
        grouped[row["metric"]][target].append(row)
    return grouped


def max_or_zero(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return max(vals) if vals else 0.0


def build_metric_level_criteria(comparisons: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    rows = collect_metric_rows(comparisons)
    grouped = group_metric_rows(rows)
    criteria: Dict[str, Any] = {}
    all_metrics_within = True
    evidence_sufficient = True
    excess_metrics: List[str] = []
    insufficient_metrics: List[str] = []
    for spec in FLOAT_METRICS:
        name = spec["name"]
        within_rows = grouped[name]["within"]
        off_on_rows = grouped[name]["off_on"]
        gate_dimensions = list(spec["gate_dimensions"])
        reference = {
            "max_abs_diff": max_or_zero([row["max_abs_diff"] for row in within_rows]),
            "mean_abs_diff": max_or_zero([row["mean_abs_diff"] for row in within_rows]),
            "max_rel_diff": max_or_zero([row["max_rel_diff"] for row in within_rows]),
            "differing_element_count_max": max_or_zero([row["differing_element_count"] for row in within_rows]),
            "all_within_rows_finite": all(bool(row["finite"]) for row in within_rows),
            "within_mode_row_count": len(within_rows),
            "off_off_row_count": sum(1 for row in within_rows if row["comparison_class"] == "OFF_OFF"),
            "on_on_row_count": sum(1 for row in within_rows if row["comparison_class"] == "ON_ON"),
        }
        off_on_observed = {
            "max_abs_diff": max_or_zero([row["max_abs_diff"] for row in off_on_rows]),
            "mean_abs_diff": max_or_zero([row["mean_abs_diff"] for row in off_on_rows]),
            "max_rel_diff": max_or_zero([row["max_rel_diff"] for row in off_on_rows]),
            "differing_element_count_max": max_or_zero([row["differing_element_count"] for row in off_on_rows]),
            "all_off_on_rows_finite": all(bool(row["finite"]) for row in off_on_rows),
            "off_on_row_count": len(off_on_rows),
        }
        sufficient = (
            reference["off_off_row_count"] >= MIN_COMPARISON_ROWS_PER_CLASS
            and reference["on_on_row_count"] >= MIN_COMPARISON_ROWS_PER_CLASS
            and off_on_observed["off_on_row_count"] >= MIN_COMPARISON_ROWS_PER_CLASS
            and reference["all_within_rows_finite"]
            and off_on_observed["all_off_on_rows_finite"]
        )
        if not sufficient:
            evidence_sufficient = False
            insufficient_metrics.append(name)
        dimension_results = {
            dim: {
                "reference_within_mode_envelope": reference[dim],
                "off_on_observed_max": off_on_observed[dim],
                "within_reference": off_on_observed[dim] <= reference[dim],
            }
            for dim in gate_dimensions
        }
        metric_within = sufficient and all(row["within_reference"] for row in dimension_results.values())
        if not metric_within:
            all_metrics_within = False
            if sufficient:
                excess_metrics.append(name)
        criteria[name] = {
            "criterion_type": "empirical_mps_within_mode_envelope_no_multiplier",
            "reference_within_mode_envelope": reference,
            "threshold_or_comparison_rule": {
                "rule": "MPS-equivalent iff candidate OFF↔ON metric is finite and every hard-gated dimension is <= the frozen within-mode empirical envelope from OFF↔OFF and ON↔ON rows.",
                "gated_dimensions": gate_dimensions,
                "thresholds": {dim: reference[dim] for dim in gate_dimensions},
                "diagnostic_dimensions": list(spec.get("diagnostic_dimensions", [])),
                "diagnostic_reference_within_mode_envelope": {
                    dim: reference[dim]
                    for dim in spec.get("diagnostic_dimensions", [])
                },
                "safety_multiplier": None,
                "arbitrary_tolerance_added": False,
                "non_gated_diagnostic_policy": spec.get(
                    "relative_diff_policy",
                    "mean_abs_diff_and_max_rel_diff_are_frozen_report_only_diagnostics",
                ),
            },
            "evidence_rows": {
                "within_mode": within_rows,
                "off_on": off_on_rows,
            },
            "run_count": {
                "within_mode_rows": len(within_rows),
                "off_on_rows": len(off_on_rows),
            },
            "exact_vs_numerical_requirement": spec["exact_vs_numerical_requirement"],
            "rationale": spec["rationale"],
            "r2_observed_off_on": off_on_observed,
            "r2_observed_off_on_within_reference": metric_within,
            "dimension_results": dimension_results,
        }
    return {
        "stage": STAGE,
        "criterion_family": "MPS empirical within-mode numerical equivalence",
        "minimum_rows_per_comparison_class": MIN_COMPARISON_ROWS_PER_CLASS,
        "float_metric_criteria": criteria,
        "all_metric_criteria_evidence_sufficient": evidence_sufficient,
        "all_observed_off_on_float_metrics_within_reference": all_metrics_within,
        "insufficient_metrics": insufficient_metrics,
        "instrumentation_specific_excess_metrics": excess_metrics,
        "rows_sha256": canonical_sha(rows),
        "arbitrary_tolerance_added": False,
    }


def exact_invariant_contract(comparisons: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    h4mh_preflight = read_json(H4M_H_ROOT / "02_mps_off_on_preflight.json")
    r1_rng_shadow = read_json(R1_ROOT / "08_rng_shadow_integrity.json")
    sample_equivalence = h4mh_preflight.get("sample_equivalence", {})
    invariant_checks = {
        "sampled_actions_exact_match": all(comparison_stage_pick(row["details"], "sampled_actions")["sha_match"] for row in comparisons.values()),
        "reward_v2_sequence_exact_match": all(comparison_stage_pick(row["details"], "raw_rewards")["sha_match"] for row in comparisons.values()),
        "legal_masks_exact_match": sample_equivalence.get("legal_masks_match") is True,
        "sample_uid_sequence_exact_match": sample_equivalence.get("sample_uid_sequence_match") is True,
        "targets_exact_match": sample_equivalence.get("targets_match") is True,
        "rng_lineage_exact_match": all(
            bool(value)
            for row in comparisons.values()
            for value in row["details"].get("rng", {}).values()
        ),
        "shadow_contamination_zero": r1_rng_shadow.get("shadow_contamination") is False
        and all(
            not bool((row["details"].get("shadow", {}).get(side, {}) or {}).get("shadow_data_entered_training", False))
            for row in comparisons.values()
            for side in ["left_shadow", "right_shadow"]
        ),
    }
    contract = {
        "stage": STAGE,
        "exact_invariants": {
            "sampled_actions": {
                "requirement": "exact SHA/value match",
                "rationale": "Policy sample path must remain unchanged by instrumentation.",
            },
            "legal_masks": {
                "requirement": "exact match",
                "rationale": "K-mask/legal action feasibility is locked and cannot be numerically relaxed.",
            },
            "reward_v2_sequence": {
                "requirement": "exact SHA/value match",
                "rationale": "Reward V2 is authoritative and instrumentation must not change rewards.",
            },
            "sample_uid_sequence": {
                "requirement": "exact match",
                "rationale": "OFF and ON must evaluate the same samples in the same order.",
            },
            "targets": {
                "requirement": "exact match",
                "rationale": "Supervised/controlled target lineage must remain unchanged.",
            },
            "rng_lineage": {
                "requirement": "all recorded RNG snapshots equal at comparable checkpoints; RNG drift = 0",
                "rationale": "Instrumentation must not consume or perturb training RNG streams.",
            },
            "shadow_contamination": {
                "requirement": "zero shadow data in training, PPO, critic, normalizers, or RNG",
                "rationale": "Counterfactual shadow trace is side-channel only.",
            },
        },
        "evidence": {
            "h4m_h_preflight_sample_equivalence": sample_equivalence,
            "h4m_h_rng_noninterference_passed": h4mh_preflight.get("rng_noninterference", {}).get("rng_noninterference_passed"),
            "h4m_h_shadow_training_separation_passed": h4mh_preflight.get("shadow_separation", {}).get("shadow_training_separation_passed"),
            "r1_rng_shadow_integrity": {
                "rng_initial_state_reproducible": r1_rng_shadow.get("rng_initial_state_reproducible"),
                "shadow_contamination": r1_rng_shadow.get("shadow_contamination"),
                "integrity_passed": r1_rng_shadow.get("integrity_passed"),
            },
        },
        "checks": invariant_checks,
        "exact_invariant_contract_passed": all(invariant_checks.values()),
    }
    return contract


def mps_equivalence_decision_rule(metric_criteria: Mapping[str, Any], exact_contract: Mapping[str, Any]) -> Dict[str, Any]:
    exact_ok = exact_contract.get("exact_invariant_contract_passed") is True
    sufficient = metric_criteria.get("all_metric_criteria_evidence_sufficient") is True
    float_within = metric_criteria.get("all_observed_off_on_float_metrics_within_reference") is True
    if exact_ok and sufficient and float_within:
        r2_application = "MPS_EQUIVALENT"
    elif not sufficient:
        r2_application = "INDETERMINATE"
    else:
        r2_application = "MPS_NOT_EQUIVALENT"
    return {
        "stage": STAGE,
        "scientific_equivalence_rule": {
            "MPS_EQUIVALENT": [
                "all exact invariants pass",
                "all float metrics are finite",
                "OFF↔ON max_abs_diff for every float metric is within the frozen MPS within-mode empirical envelope",
                "mean_abs_diff and max_rel_diff are recorded as frozen diagnostics without adding a multiplier or tolerance",
                "first divergence is not earlier than value_t",
                "no instrumentation-specific excess variability",
                "RNG drift = 0",
                "shadow contamination = 0",
            ],
            "MPS_NOT_EQUIVALENT": [
                "any exact invariant fails",
                "any hard-gated float metric exceeds its frozen within-mode envelope with sufficient evidence",
                "any non-finite value appears",
                "new earlier divergence appears before value_t",
                "RNG drift or shadow contamination appears",
            ],
            "INDETERMINATE": [
                "within-mode evidence rows are insufficient after minimal repeats",
                "evidence cannot distinguish natural MPS variability from instrumentation-induced excess without inventing tolerance",
            ],
        },
        "bit_exact_equivalence_rule": {
            "BIT_EXACT_EQUIVALENT": "all SHA comparisons match, including float tensors, PPO fingerprints, parameter updates, and checkpoint hash",
            "SCIENTIFIC_MPS_EQUIVALENT": "exact invariants match and float differences are inside frozen MPS envelope even when SHA differs",
            "rationale": "H4M-G-R1 showed natural MPS SHA drift at value_t; bit-exact equivalence is therefore stricter than scientific MPS equivalence.",
        },
        "r2_observed_application": r2_application,
        "metric_evidence_sufficient": sufficient,
        "exact_invariants_passed": exact_ok,
        "observed_float_metrics_within_reference": float_within,
        "instrumentation_specific_excess_metrics": metric_criteria.get("instrumentation_specific_excess_metrics", []),
        "arbitrary_tolerance_added": False,
    }


def h4mh_preflight_release_contract(metric_criteria: Mapping[str, Any], exact_contract: Mapping[str, Any]) -> Dict[str, Any]:
    thresholds = {
        name: data["threshold_or_comparison_rule"]
        for name, data in metric_criteria.get("float_metric_criteria", {}).items()
    }
    return {
        "stage": STAGE,
        "release_scope": "H4M-H-RERUN PRE-1 only; does not authorize full training success or TEST6.",
        "h4m_h_rerun_auto_execution_authorized": False,
        "pre_1_pass_conditions": {
            "exact_invariants": exact_contract.get("exact_invariants"),
            "float_metrics": thresholds,
            "earliest_allowed_numerical_divergence": "value_t",
            "no_new_earlier_divergence": "sampled_actions, legal masks, reward sequence, sample UID/order, targets, and RNG lineage must remain exact before value_t.",
            "no_instrumentation_specific_excess_variability": "Every hard-gated OFF↔ON float metric must be <= the frozen within-mode envelope; mean_abs/max_rel diagnostics must be recorded but are not tolerance gates.",
            "rng_drift": 0,
            "shadow_contamination": 0,
        },
        "pre_1_block_conditions": [
            "authoritative SHA/source/runtime mismatch",
            "exact invariant failure",
            "new divergence before value_t",
            "any gated float metric exceeds frozen MPS envelope",
            "non-finite float output",
            "RNG drift",
            "shadow contamination",
            "TEST6/validation access",
        ],
        "bit_exact_equivalence_required": False,
        "scientific_mps_equivalence_required": True,
        "stop_flags": {
            "h4m_h_executed_by_r2": False,
            "model_reward_ppo_modified": False,
            "environment_expanded": False,
            "test6_opened": False,
            "github_push_performed": False,
        },
    }


def gate_matrix(
    binding: Mapping[str, Any],
    additional: Mapping[str, Any],
    metric_criteria: Mapping[str, Any],
    exact_contract: Mapping[str, Any],
    decision_rule: Mapping[str, Any],
    contract_sha256: Optional[str],
) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding_match": binding.get("authoritative_binding_passed") is True,
        "minimal_repeats_performed_if_required": (not additional.get("required")) or additional.get("performed") is True,
        "within_mode_envelope_sufficient": metric_criteria.get("all_metric_criteria_evidence_sufficient") is True,
        "arbitrary_tolerance_absent": metric_criteria.get("arbitrary_tolerance_added") is False
        and decision_rule.get("arbitrary_tolerance_added") is False,
        "instrumentation_specific_excess_absent": metric_criteria.get("all_observed_off_on_float_metrics_within_reference") is True,
        "exact_invariant_contract_passed": exact_contract.get("exact_invariant_contract_passed") is True,
        "rng_shadow_integrity_passed": exact_contract.get("checks", {}).get("rng_lineage_exact_match") is True
        and exact_contract.get("checks", {}).get("shadow_contamination_zero") is True,
        "r2_observed_application_mps_equivalent": decision_rule.get("r2_observed_application") == "MPS_EQUIVALENT",
        "contract_sha256_created": bool(contract_sha256),
        "test6_not_opened": True,
        "github_push_false": True,
    }
    pass_ready = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if pass_ready else BLOCK_GATE,
        "decision": PASS_DECISION if pass_ready else "H4M_G_R2_MPS_EQUIVALENCE_CRITERION_FREEZE_BLOCKED",
        "exact_next_gate": NEXT_GATE if pass_ready else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "mps_numerical_equivalence_contract_sha256": contract_sha256,
        "criteria": criteria,
        "failing_criteria": [name for name, passed in criteria.items() if not passed],
        "block_reason": None if pass_ready else "H4M_G_R2_GATE_CRITERIA_FAILED",
        "final_flags": {
            "h4m_h_rerun_executed": False,
            "full_training_executed": False,
            "repair_executed": False,
            "deterministic_algorithm_forced": False,
            "dtype_changed": False,
            "cpu_fallback_adopted": False,
            "reward_zero_loss_kmask_modified": False,
            "model_observation_modified": False,
            "ppo_gae_normalization_modified": False,
            "hyperparameters_modified": False,
            "environment_data_rng_semantics_modified": False,
            "validation_executed": False,
            "TEST6_opened": False,
            "github_push_performed": False,
        },
    }


def build_contract(
    created_at: str,
    binding: Mapping[str, Any],
    r1_evidence: Mapping[str, Any],
    additional: Mapping[str, Any],
    metric_criteria: Mapping[str, Any],
    exact_contract: Mapping[str, Any],
    decision_rule: Mapping[str, Any],
    release_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    public_additional = {k: v for k, v in additional.items() if not k.startswith("_")}
    return {
        "stage": STAGE,
        "created_at": created_at,
        "contract_name": "PV8_R2A_R8E_R3_R_H4M_G_R2_MPS_NUMERICAL_EQUIVALENCE_CONTRACT",
        "contract_version": 1,
        "authoritative_binding": binding,
        "r1_variability_evidence": r1_evidence,
        "additional_repeats_if_required": public_additional,
        "metric_level_equivalence_criteria": metric_criteria,
        "exact_invariant_contract": exact_contract,
        "mps_equivalence_decision_rule": decision_rule,
        "h4mh_preflight_release_contract": release_contract,
        "hard_locks": {
            "reward_v2_zero_loss_kmask_changed": False,
            "actor_critic_gatv2_changed": False,
            "observation_changed": False,
            "ppo_gae_normalization_changed": False,
            "hyperparameters_changed": False,
            "environment_data_rng_semantics_changed": False,
            "deterministic_algorithm_forced": False,
            "dtype_changed": False,
            "cpu_fallback_adopted": False,
            "test6_opened": False,
            "github_push_performed": False,
        },
    }


def final_report(
    gate: Mapping[str, Any],
    binding: Mapping[str, Any],
    additional: Mapping[str, Any],
    metric_criteria: Mapping[str, Any],
    exact_contract: Mapping[str, Any],
    decision_rule: Mapping[str, Any],
    contract_sha256: str,
) -> str:
    metric_lines = []
    for name, criterion in metric_criteria.get("float_metric_criteria", {}).items():
        thresholds = criterion["threshold_or_comparison_rule"]["thresholds"]
        status = "within" if criterion["r2_observed_off_on_within_reference"] else "exceeds"
        metric_lines.append(f"- {name}: {status}; thresholds={json.dumps(thresholds, sort_keys=True)}")
    return f"""# H4M-G-R2 MPS Numerical Equivalence Criterion Selection & Freeze

gate = {gate["gate"]}
decision = {gate["decision"]}
source_commit = {binding["source_provenance"]["h4m_g_r2_source_git_commit"]}
contract_sha256 = {contract_sha256}
exact_next_gate = {gate["exact_next_gate"]}

## Additional repeats

required = {additional.get("required")}
performed = {additional.get("performed")}
repeat_batches = {additional.get("repeat_batches", 0)}

## R2 observed application

{decision_rule["r2_observed_application"]}

## Frozen float criteria

{chr(10).join(metric_lines)}

## Exact invariants

{json.dumps(exact_contract["checks"], ensure_ascii=False, indent=2)}

STOP: no H4M-H execution, no model/reward/PPO modification, no environment expansion, no TEST6, no GitHub push.
"""


def make_manifest(artifact_root: Path, gate: Mapping[str, Any]) -> Dict[str, Any]:
    output_files: Dict[str, str] = {}
    for path in artifact_root.rglob("*"):
        if path.is_file() and path.name != "manifest.json":
            output_files[str(path.relative_to(artifact_root))] = str(path)
    return {
        "stage": STAGE,
        "created_at": kst_now(),
        "artifact_root": str(artifact_root),
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "mps_numerical_equivalence_contract_sha256": gate.get("mps_numerical_equivalence_contract_sha256"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_files": output_files,
        "output_sha256": {name: sha256_file(Path(path)) for name, path in output_files.items()},
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256",
        "github_push_performed": False,
        "TEST6_opened": False,
    }


def block_payloads(
    created_at: str,
    artifact_root: Path,
    binding: Mapping[str, Any],
    r1_evidence: Mapping[str, Any],
    additional: Mapping[str, Any],
    reason: str,
) -> None:
    metric_criteria = {
        "stage": STAGE,
        "all_metric_criteria_evidence_sufficient": False,
        "all_observed_off_on_float_metrics_within_reference": False,
        "arbitrary_tolerance_added": False,
        "block_reason": reason,
    }
    exact_contract = {"stage": STAGE, "checks": {}, "exact_invariant_contract_passed": False, "block_reason": reason}
    decision_rule = {
        "stage": STAGE,
        "r2_observed_application": "INDETERMINATE",
        "arbitrary_tolerance_added": False,
        "block_reason": reason,
    }
    release_contract = {"stage": STAGE, "h4m_h_rerun_auto_execution_authorized": False, "block_reason": reason}
    contract = build_contract(created_at, binding, r1_evidence, additional, metric_criteria, exact_contract, decision_rule, release_contract)
    write_json(artifact_root / "mps_numerical_equivalence_contract.json", contract)
    contract_sha = sha256_file(artifact_root / "mps_numerical_equivalence_contract.json")
    gate = gate_matrix(binding, additional, metric_criteria, exact_contract, decision_rule, contract_sha)
    payloads = {
        "01_authoritative_binding.json": binding,
        "02_r1_variability_evidence.json": r1_evidence,
        "03_additional_repeats_if_required.json": {k: v for k, v in additional.items() if not k.startswith("_")},
        "04_metric_level_equivalence_criteria.json": metric_criteria,
        "05_exact_invariant_contract.json": exact_contract,
        "06_mps_equivalence_decision_rule.json": decision_rule,
        "07_h4mh_preflight_release_contract.json": release_contract,
        "08_h4m_g_r2_gate_matrix.json": gate,
    }
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(
        f"# H4M-G-R2 BLOCK\n\nreason = {reason}\ngate = {gate['gate']}\nSTOP\n",
        encoding="utf-8",
    )
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-G-R2] artifact root: {artifact_root}")
    print(f"[H4M-G-R2] gate: {gate['gate']}")
    print(f"[H4M-G-R2] block_reason: {reason}")


def main() -> None:
    created_at = kst_now()
    stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_g_r2_mps_equivalence_criterion_freeze_{stamp}"
    artifact_root.mkdir(parents=True, exist_ok=False)
    provenance = source_provenance(created_at)
    r1_evidence = r1_variability_evidence(created_at)
    binding = authoritative_binding(created_at, provenance, r1_evidence)
    r1_comparisons = r1_comparison_records()
    gaps = initial_evidence_gaps(r1_comparisons)
    repeats_required = bool(gaps)
    if not binding["authoritative_binding_passed"]:
        additional = {
            "stage": STAGE,
            "created_at": created_at,
            "required": repeats_required,
            "performed": False,
            "initial_evidence_gaps": gaps,
            "comparison_records": {},
            "trial_summaries": {},
        }
        block_payloads(created_at, artifact_root, binding, r1_evidence, additional, "AUTHORITATIVE_BINDING_MISMATCH")
        return
    additional = run_additional_repeats(created_at, artifact_root, repeats_required, gaps)
    if additional.get("required") and not additional.get("performed"):
        block_payloads(created_at, artifact_root, binding, r1_evidence, additional, str(additional.get("blocked_reason", "ADDITIONAL_REPEATS_REQUIRED_BUT_NOT_PERFORMED")))
        return
    all_comparisons = dict(r1_comparisons)
    all_comparisons.update(additional.get("_comparison_records_for_internal_use", {}))
    metric_criteria = build_metric_level_criteria(all_comparisons)
    exact_contract = exact_invariant_contract(all_comparisons)
    decision_rule = mps_equivalence_decision_rule(metric_criteria, exact_contract)
    release_contract = h4mh_preflight_release_contract(metric_criteria, exact_contract)
    contract = build_contract(created_at, binding, r1_evidence, additional, metric_criteria, exact_contract, decision_rule, release_contract)
    write_json(artifact_root / "mps_numerical_equivalence_contract.json", contract)
    contract_sha = sha256_file(artifact_root / "mps_numerical_equivalence_contract.json")
    gate = gate_matrix(binding, additional, metric_criteria, exact_contract, decision_rule, contract_sha)
    payloads = {
        "01_authoritative_binding.json": binding,
        "02_r1_variability_evidence.json": r1_evidence,
        "03_additional_repeats_if_required.json": {k: v for k, v in additional.items() if not k.startswith("_")},
        "04_metric_level_equivalence_criteria.json": metric_criteria,
        "05_exact_invariant_contract.json": exact_contract,
        "06_mps_equivalence_decision_rule.json": decision_rule,
        "07_h4mh_preflight_release_contract.json": release_contract,
        "08_h4m_g_r2_gate_matrix.json": gate,
    }
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(
        final_report(gate, binding, additional, metric_criteria, exact_contract, decision_rule, contract_sha),
        encoding="utf-8",
    )
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate))
    print(f"[H4M-G-R2] artifact root: {artifact_root}")
    print(f"[H4M-G-R2] gate: {gate['gate']}")
    print(f"[H4M-G-R2] decision: {gate['decision']}")
    print(f"[H4M-G-R2] r2_observed_application: {decision_rule['r2_observed_application']}")
    print(f"[H4M-G-R2] additional_repeats_required: {additional.get('required')}")
    print(f"[H4M-G-R2] additional_repeats_performed: {additional.get('performed')}")
    print(f"[H4M-G-R2] contract_sha256: {contract_sha}")
    print(f"[H4M-G-R2] next_gate: {gate['exact_next_gate']}")
    print("[H4M-G-R2] STOP no H4M-H rerun no TEST6 no GitHub push")


if __name__ == "__main__":
    main()
