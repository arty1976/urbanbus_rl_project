#!/usr/bin/env python3
"""H4M-X window episode-boundary credit-horizon repair implementation + validation.

Implements the frozen H4M-W W1 repair only and validates it with synthetic
fixtures.  This program does not train, does not construct or step an optimizer,
does not open TEST6, does not redesign reward or models, does not fall back to
W2/W3/W4, does not tune hyperparameters, does not mutate the database, and does
not push to GitHub.  Previously published artifacts are read only.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import py_compile
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence
from zoneinfo import ZoneInfo


STAGE = "PV8-R2A-R8E-R3-R-H4M-X"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_X_"
    "WINDOW_EPISODE_BOUNDARY_CREDIT_HORIZON_REPAIR_IMPLEMENTATION_AND_NO_TRAINING_EQUIVALENCE_VALIDATION_COMPLETE"
)
BLOCK_GATE_PREFIX = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_X"
NEXT_GATE = (
    "H4M-Y_FRESH_WINDOW_EPISODE_BOUNDARY_REPAIRED_THREE_SEED_RETRAINING_WITH_DURABLE_EVIDENCE"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"

SOURCE_REL = Path("05_training") / Path(__file__).name
H4MG_REL = Path("05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py")
TEST_REL = Path("05_training/test_h4m_x_window_episode_boundary.py")
DL1_REL = Path("05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py")
REWARD_REL = Path("05_training/rewards/mappo_reward_v1.py")
DURABLE_REL = Path("05_training/durable_training_evidence.py")
OBS_REL = Path("05_training/observation_target_context_repair.py")
INTENDED_CHANGED_FILES = [H4MG_REL.as_posix(), TEST_REL.as_posix(), SOURCE_REL.as_posix()]
IMMUTABLE_SOURCES = {
    "td_gae_actor_critic_dl1": DL1_REL.as_posix(),
    "reward_v2": REWARD_REL.as_posix(),
    "observation_contract": OBS_REL.as_posix(),
    "durable_evidence_instrumentation": DURABLE_REL.as_posix(),
    "h4m_k_training_loop": "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_k_fresh_target_context_repaired_three_seed_retraining.py",
    "h4m_q_execution_lineage": "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining.py",
}

H4MW_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_w_window_episode_boundary_credit_repair_selection_freeze_20260818_131002+09:00"
H4MV_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_v_repaired_retraining_outcome_review_next_decision_20260818_010904+09:00"

EXPECTED = {
    "source_before_commit": "1d9457e368883e2cc48dd7760f020380e270700e",
    "h4m_w_gate": (
        "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_W_"
        "ROLLOUT_WINDOW_EPISODE_BOUNDARY_AND_CREDIT_HORIZON_REPAIR_SELECTION_AND_FREEZE_COMPLETE"
    ),
    "w1_repair_id": "W1_WINDOW_EPISODE_BOUNDARY_MASKING",
    "w1_repair_contract_sha256": "d1bb5b4c68de19746ffde42fed57bc55b6a0b328c0d31acad1743139e416cef3",
    "root_cause": "V2_TARGET_RETURN_CONSTRUCTION",
    "earliest_divergence": "rollout_trajectory_assembly_and_episode_boundary",
    "boundary_schema": "independent_window_causal_horizon_termination_v1",
    "actor_repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
    "critic_repair_contract_sha256": "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
}

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "repair_binding.json",
    "boundary_validation.json",
    "credit_horizon_validation.json",
    "context_sign_validation.json",
    "equivalence_validation.json",
    "checkpoint_compatibility.json",
    "changed_files.json",
    "gate_matrix.json",
    "test_results.json",
    "test_results.stdout.txt",
]
MUTABLE_STATE_FILES: List[str] = []


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n", encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_run(args: Sequence[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def py_compile_audit() -> Dict[str, Any]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="h4mx_pycompile_") as tmp:
        for rel in INTENDED_CHANGED_FILES:
            source = PROJECT_ROOT / rel
            try:
                py_compile.compile(str(source), cfile=str(Path(tmp) / (source.name + ".pyc")), doraise=True)
                rows.append({"source_rel": rel, "passed": True, "error": None})
            except Exception as exc:
                rows.append({"source_rel": rel, "passed": False, "error": repr(exc)})
    cached = git_run(["diff", "--cached", "--check"], check=False)
    return {
        "rows": rows,
        "py_compile_passed": all(row["passed"] for row in rows),
        "git_diff_cached_check_passed": cached.returncode == 0,
    }


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    parent = git_run(["rev-parse", "HEAD^"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    status = git_run(["status", "--short"]).stdout.strip()
    latest = {rel: git_run(["log", "-1", "--format=%H", "--", rel]).stdout.strip() for rel in INTENDED_CHANGED_FILES}
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_commit_before_work": parent,
        "source_commit_after_pass": head,
        "head_commit_files": head_files,
        "intended_changed_files": INTENDED_CHANGED_FILES,
        "head_commit_intended_source_test_only": sorted(head_files) == sorted(INTENDED_CHANGED_FILES),
        "source_parent_matches_h4m_w": parent == EXPECTED["source_before_commit"],
        "all_intended_files_latest_at_head": all(value == head for value in latest.values()),
        "clean_worktree": status == "",
        "status_short": status,
        "github_push_performed": False,
    }


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def repair_binding(created_at: str, provenance: Mapping[str, Any], compile_audit: Mapping[str, Any]) -> Dict[str, Any]:
    w_gate = read_json(H4MW_ROOT / "gate_matrix.json")
    w_freeze = read_json(H4MW_ROOT / "repair_freeze_contract.json")
    w_selection = read_json(H4MW_ROOT / "selected_repair.json")
    v_review = read_json(H4MV_ROOT / "root_cause_review.json")
    h4mg = import_module("h4mx_runner_h4mg", PROJECT_ROOT / H4MG_REL)
    contract = w_freeze.get("contract", {})
    checks = {
        "source_only_commit": provenance.get("head_commit_intended_source_test_only") is True
        and provenance.get("all_intended_files_latest_at_head") is True
        and provenance.get("clean_worktree") is True,
        "source_parent_matches_h4m_w": provenance.get("source_parent_matches_h4m_w") is True,
        "py_compile_passed": compile_audit.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_audit.get("git_diff_cached_check_passed") is True,
        "h4m_w_gate_match": w_gate.get("gate") == EXPECTED["h4m_w_gate"],
        "w1_selected_by_h4m_w": w_selection.get("selected_repair") == EXPECTED["w1_repair_id"],
        "w1_contract_sha_match": w_freeze.get("contract_sha256") == EXPECTED["w1_repair_contract_sha256"],
        "w1_contract_sha_bound_in_source": getattr(h4mg, "W1_REPAIR_CONTRACT_SHA256", None) == EXPECTED["w1_repair_contract_sha256"],
        "w1_repair_id_bound_in_source": getattr(h4mg, "W1_REPAIR_ID", None) == EXPECTED["w1_repair_id"],
        "boundary_schema_bound_in_source": getattr(h4mg, "WINDOW_EPISODE_BOUNDARY_SCHEMA", None) == EXPECTED["boundary_schema"],
        "boundary_builder_present": callable(getattr(h4mg, "window_episode_boundary_mask", None)),
        "root_cause_match": v_review.get("unique_root_cause") == EXPECTED["root_cause"],
        "contract_forbids_compute_gae_edit": contract.get("compute_gae_source_unchanged") is True
        and contract.get("td_gae_equation_change") is False,
        "s3_binding_schema_retained": getattr(h4mg, "CRITIC_VALUE_TARGET_BINDING_SCHEMA", None) == "rollout_pre_update_return_normalizer_state_v1",
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "checks": checks,
        "repair_binding_passed": all(checks.values()),
        "source_provenance": provenance,
        "w1_repair_contract": contract,
        "w1_repair_contract_sha256": EXPECTED["w1_repair_contract_sha256"],
        "frozen_sha_bindings": {
            "actor_repair_contract_sha256": EXPECTED["actor_repair_contract_sha256"],
            "critic_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"],
            "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
        },
        "implemented_repairs": [EXPECTED["w1_repair_id"]],
        "not_implemented": ["W2_WINDOW_SCOPED_ADVANTAGE_NORMALIZATION", "W3_BANDIT_ALIGNED_RETURN_TARGET", "W4_CONTEXT_BASELINE"],
        "hard_lock_attestation": {
            "training_executed": False,
            "training_count": 0,
            "optimizer_step_count": 0,
            "test6_access_count": 0,
            "reward_or_model_redesign": False,
            "hyperparameter_tuning": False,
            "w2_w3_w4_fallback": False,
            "database_or_data_mutation": False,
            "old_artifacts_mutated": False,
            "github_push_performed": False,
        },
    }


def run_tests(artifact_root: Path) -> Dict[str, Any]:
    output = artifact_root / "test_results.json"
    command = [sys.executable, str(PROJECT_ROOT / TEST_REL), "--json-output", str(output)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    (artifact_root / "test_results.stdout.txt").write_text(
        "command:\n" + " ".join(command) + "\n\nstdout:\n" + completed.stdout + "\nstderr:\n" + completed.stderr,
        encoding="utf-8",
    )
    payload = read_json(output) if output.exists() else {"passed": False, "reason": "TEST_RESULT_JSON_NOT_WRITTEN"}
    if not output.exists():
        write_json(output, payload)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-1500:],
        "stderr_tail": completed.stderr[-1500:],
        "payload": payload,
        "passed": completed.returncode == 0 and payload.get("passed") is True,
    }


def section(tests: Mapping[str, Any], name: str) -> Dict[str, Any]:
    return tests.get("payload", {}).get("sections", {}).get(name, {})


def boundary_validation(tests: Mapping[str, Any]) -> Dict[str, Any]:
    binding = section(tests, "repair_binding")
    boundary = section(tests, "boundary_mask")
    checks = {
        "w1_repair_sha_binding_exact": binding.get("checks", {}).get("repair_contract_sha_bound") is True,
        "every_independent_window_boundary_terminates": boundary.get("checks", {}).get("one_step_per_window_terminates_every_step") is True,
        "boundary_bootstrap_mask_zero": boundary.get("checks", {}).get("boundary_bootstrap_mask_is_zero") is True,
        "boundary_derived_from_window_plan": boundary.get("checks", {}).get("boundary_derived_from_window_plan_not_assumed") is True,
        "interior_step_keeps_bootstrap": boundary.get("checks", {}).get("interior_step_keeps_bootstrap") is True,
        "w2_w3_w4_not_implemented": binding.get("checks", {}).get("w2_w3_w4_not_implemented") is True,
    }
    return {
        "stage": STAGE,
        "implementation_site": f"{H4MG_REL.as_posix()}::collect_controlled_rollout",
        "boundary_semantics": (
            "causal-horizon termination: the world continues, but the counterfactual evidence proves no part of the "
            "future is attributable to the sampled action, so the decision's credit horizon ends at its window"
        ),
        "repair_binding_section": binding,
        "boundary_mask_section": boundary,
        "checks": checks,
        "passed": all(checks.values()),
    }


def credit_horizon_validation(tests: Mapping[str, Any]) -> Dict[str, Any]:
    credit = section(tests, "credit_horizon")
    scale = section(tests, "target_scale_consistency")
    checks = {
        "next_window_value_influence_zero": credit.get("checks", {}).get("next_window_value_influence_zero") is True,
        "cross_window_gae_recursion_zero": credit.get("cross_window_gae_recursion_count") == 0
        and credit.get("checks", {}).get("later_window_reward_influence_zero") is True,
        "advantage_is_own_window_residual": credit.get("checks", {}).get("advantage_is_own_window_residual") is True,
        "same_window_return_ownership": credit.get("checks", {}).get("return_target_is_own_window_reward") is True,
        "future_leakage_zero": credit.get("checks", {}).get("terminal_bootstrap_leak_count_zero") is True,
        "no_false_truncation_claim": credit.get("checks", {}).get("no_truncated_bootstrap_claimed") is True,
        "unrepaired_path_demonstrably_leaked": credit.get("checks", {}).get("unrepaired_path_did_leak_later_rewards") is True,
        "one_decision_target_scale_consistent": scale.get("passed") is True,
        "stale_multi_window_critic_not_used_as_evidence": scale.get("checks", {}).get("advantage_action_contrast_unaffected_by_critic_scale") is True,
    }
    return {
        "stage": STAGE,
        "credit_horizon_section": credit,
        "target_scale_section": scale,
        "residual_risk_addressed": (
            "the repaired critic target is the window's own reward, so a critic trained under it converges to the reward "
            "scale; a stale multi-window critic only shifts every advantage in the window by a constant and cannot change "
            "the action contrast, so it is neither evidence for nor against the repair"
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def context_sign_validation(tests: Mapping[str, Any]) -> Dict[str, Any]:
    signs = section(tests, "context_sign_preservation")
    checks = {
        "hold_better_sign_preserved": signs.get("checks", {}).get("hold_better_sign_preserved") is True,
        "serve_better_sign_preserved": signs.get("checks", {}).get("serve_better_sign_preserved") is True,
        "normalized_signs_preserved": signs.get("checks", {}).get("hold_better_normalized_sign_preserved") is True
        and signs.get("checks", {}).get("serve_better_normalized_sign_preserved") is True,
        "contrast_equals_causal_reward_difference": signs.get("checks", {}).get("contrast_equals_causal_reward_difference") is True,
        "same_window_reward_ownership": signs.get("checks", {}).get("same_window_reward_ownership_preserved") is True,
        "naive_r_minus_v_counterexample_not_reused": signs.get("discredited_baseline_reused") is False,
    }
    return {
        "stage": STAGE,
        "context_sign_section": signs,
        "checks": checks,
        "interpretation_rule": "sign safety is a credit-direction property; it is not a claim about HOLD ratio or performance",
        "passed": all(checks.values()),
    }


def equivalence_validation(tests: Mapping[str, Any], changed: Mapping[str, Any]) -> Dict[str, Any]:
    immutable = section(tests, "immutable_surfaces")
    payload = tests.get("payload", {})
    checks = {
        "compute_gae_source_diff_zero": immutable.get("checks", {}).get("compute_gae_source_diff_zero") is True
        and immutable.get("compute_gae_sha256_head") == immutable.get("compute_gae_sha256_worktree"),
        "reward_v2_output_diff_zero": immutable.get("checks", {}).get("reward_v2_output_diff_zero") is True,
        "actor_logits_masks_diff_zero": immutable.get("checks", {}).get("actor_and_mask_diff_zero") is True,
        "critic_diff_zero": immutable.get("checks", {}).get("critic_diff_zero") is True,
        "s3_binding_retained": immutable.get("checks", {}).get("s3_binding_schema_unchanged") is True
        and immutable.get("checks", {}).get("s3_binding_functions_present") is True,
        "immutable_sources_unchanged": changed.get("immutable_sources_unchanged") is True,
        "nan_inf_zero": payload.get("nan_inf_count") == 0,
        "future_leakage_zero": payload.get("future_leakage_count") == 0,
        "test6_access_zero": payload.get("test6_access_count") == 0,
        "training_zero": payload.get("training_count") == 0 and payload.get("training_executed") is False,
        "optimizer_step_zero": payload.get("optimizer_step_count") == 0,
        "w2_w3_w4_fallback_not_used": payload.get("w2_w3_w4_fallback_used") is False,
    }
    return {"stage": STAGE, "immutable_section": immutable, "immutable_source_diff": changed.get("immutable_source_diff"), "checks": checks, "passed": all(checks.values())}


def checkpoint_compatibility(tests: Mapping[str, Any]) -> Dict[str, Any]:
    immutable = section(tests, "immutable_surfaces")
    scale = section(tests, "target_scale_consistency")
    checks = {
        "actor_and_critic_state_dict_strict_reload": immutable.get("checks", {}).get("checkpoint_schema_compatible") is True,
        "s3_binding_accepts_repaired_target": scale.get("checks", {}).get("s3_binding_accepts_new_target") is True,
        "no_parameter_schema_change": immutable.get("checks", {}).get("actor_and_mask_diff_zero") is True
        and immutable.get("checks", {}).get("critic_diff_zero") is True,
    }
    return {
        "stage": STAGE,
        "checks": checks,
        "diffs": immutable.get("diffs"),
        "old_checkpoint_promotion": False,
        "old_multi_window_critic_used_as_performance_evidence": False,
        "note": "checkpoint schema is unchanged, so the repair needs no migration; a fresh retraining is still required before any performance claim",
        "passed": all(checks.values()),
    }


def changed_files_audit(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for rel in INTENDED_CHANGED_FILES:
        path = PROJECT_ROOT / rel
        existed = git_run(["cat-file", "-e", f"HEAD^:{rel}"], check=False).returncode == 0
        rows.append(
            {
                "path": rel,
                "change_type": "modified" if existed else "added",
                "sha256": sha256_file(path),
                "line_count": len(path.read_text(encoding="utf-8-sig").splitlines()),
            }
        )
    immutable = {
        label: {
            "path": rel,
            "unchanged_in_head_commit": git_run(["diff", "--quiet", "HEAD^", "HEAD", "--", rel], check=False).returncode == 0,
            "sha256": sha256_file(PROJECT_ROOT / rel),
        }
        for label, rel in IMMUTABLE_SOURCES.items()
    }
    diff_stat = git_run(["diff", "--numstat", "HEAD^", "HEAD", "--", H4MG_REL.as_posix()]).stdout.strip()
    return {
        "stage": STAGE,
        "source_before_commit": EXPECTED["source_before_commit"],
        "source_only_pass_commit": provenance.get("source_commit_after_pass"),
        "changed_files": rows,
        "only_intended_files_changed": provenance.get("head_commit_intended_source_test_only") is True,
        "artifact_or_log_committed": any("artifacts/" in row["path"] for row in rows),
        "implementation_diff_numstat": diff_stat,
        "immutable_source_diff": immutable,
        "immutable_sources_unchanged": all(entry["unchanged_in_head_commit"] for entry in immutable.values()),
        "github_push_performed": False,
    }


def gate_matrix(binding, boundary, credit, signs, equivalence, checkpoints, changed, tests) -> Dict[str, Any]:
    criteria = {
        "repair_binding": binding.get("repair_binding_passed") is True,
        "source_only_commit": changed.get("only_intended_files_changed") is True and changed.get("artifact_or_log_committed") is False,
        "boundary_validated": boundary.get("passed") is True,
        "credit_horizon_validated": credit.get("passed") is True,
        "both_context_sign_preserved": signs.get("passed") is True,
        "equivalence_validated": equivalence.get("passed") is True,
        "checkpoint_compatibility": checkpoints.get("passed") is True,
        "fixture_tests_passed": tests.get("passed") is True,
        "immutable_sources_unchanged": changed.get("immutable_sources_unchanged") is True,
        "training_zero": binding["hard_lock_attestation"]["training_count"] == 0,
        "optimizer_step_zero": binding["hard_lock_attestation"]["optimizer_step_count"] == 0,
        "test6_zero": binding["hard_lock_attestation"]["test6_access_count"] == 0,
        "w2_w3_w4_fallback_false": binding["hard_lock_attestation"]["w2_w3_w4_fallback"] is False,
        "github_push_false": changed.get("github_push_performed") is False,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else f"{BLOCK_GATE_PREFIX}_VALIDATION_FAILED",
        "decision": "W1_WINDOW_EPISODE_BOUNDARY_MASKING_IMPLEMENTED_AND_EQUIVALENCE_VALIDATED" if passed else "H4M_X_BLOCKED",
        "w1_repair_contract_sha256": EXPECTED["w1_repair_contract_sha256"],
        "exact_next_gate": NEXT_GATE if passed else f"STOP_{BLOCK_GATE_PREFIX}",
        "next_gate_type": "fresh three-seed retraining",
        "next_gate_auto_execution": False,
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "training_executed": False,
            "training_count": 0,
            "optimizer_step_count": 0,
            "TEST6_opened": False,
            "w2_w3_w4_fallback": False,
            "github_push_performed": False,
        },
    }


def make_manifest(root: Path, gate: Mapping[str, Any], provenance: Mapping[str, Any]) -> Dict[str, Any]:
    files = {
        path.relative_to(root).as_posix(): str(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    immutable_files = {name: path for name, path in files.items() if name not in MUTABLE_STATE_FILES}
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "source_commit_before_work": EXPECTED["source_before_commit"],
        "source_only_pass_commit": provenance.get("source_commit_after_pass"),
        "changed_files": INTENDED_CHANGED_FILES,
        "w1_repair_contract_sha256": EXPECTED["w1_repair_contract_sha256"],
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts_present": all((root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "immutable_evidence_sha256": {name: sha256_file(Path(path)) for name, path in immutable_files.items()},
        "mutable_lifecycle_state_files": MUTABLE_STATE_FILES,
        "hash_scope_policy": (
            "immutable_evidence_sha256 covers files that never change after they are written; mutable lifecycle state is "
            "declared separately so a normal post-write transition can never be mistaken for tampering"
        ),
        "append_only_artifact": True,
        "training_count": 0,
        "optimizer_step_count": 0,
        "TEST6_opened": False,
        "test6_access_count": 0,
        "github_push_performed": False,
    }


def final_report(binding, boundary, credit, signs, equivalence, checkpoints, changed, tests, gate) -> str:
    return f"""# H4M-X Window Episode-Boundary Credit-Horizon Repair Implementation and No-Training Equivalence Validation

gate = {gate['gate']}
decision = {gate['decision']}
w1_repair_id = {EXPECTED['w1_repair_id']}
w1_repair_contract_sha256 = {EXPECTED['w1_repair_contract_sha256']}
source_before_commit = {EXPECTED['source_before_commit']}
source_only_pass_commit = {binding['source_provenance']['source_commit_after_pass']}
training_count = 0
optimizer_step_count = 0
TEST6_access_count = 0
github_push_performed = false
exact_next_gate = {gate['exact_next_gate']} (fresh three-seed retraining; not executed automatically)

## Implemented change

Only the rollout trajectory assembly changed. `compute_gae` already implemented the terminated bootstrap
mask and the GAE carry reset, so the repair supplies the flag the counterfactual evidence proves and stops
offering the next window's value to the credit path.

```json
{json.dumps({'implementation_site': boundary['implementation_site'], 'boundary_semantics': boundary['boundary_semantics'], 'diff_numstat': changed['implementation_diff_numstat'], 'changed_files': [row['path'] for row in changed['changed_files']]}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Boundary and credit horizon

```json
{json.dumps({'boundary': boundary['checks'], 'credit_horizon': credit['checks']}, ensure_ascii=False, indent=2, default=jsonable)}
```

Residual risk: {credit['residual_risk_addressed']}

## Both-context sign preservation

```json
{json.dumps(signs['context_sign_section'].get('by_context'), ensure_ascii=False, indent=2, default=jsonable)}
```

The discredited naive r - V counterexample was not reused: the baseline is the value a critic trained under
the repaired one-decision target converges to.

## Equivalence and checkpoint compatibility

```json
{json.dumps({'equivalence': equivalence['checks'], 'checkpoint': checkpoints['checks']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Tests

```json
{json.dumps({'command': tests.get('command'), 'returncode': tests.get('returncode'), 'passed': tests.get('passed')}, ensure_ascii=False, indent=2, default=jsonable)}
```

STOP: no training, no optimizer step, no TEST6 access, no reward or model redesign, no W2/W3/W4 fallback, no
hyperparameter tuning, no data mutation, and no GitHub push. The old multi-window critic and the existing
checkpoints are not used as performance evidence; a fresh three-seed retraining is required before any
behavioural claim.
"""


def write_block(root: Path, binding: Mapping[str, Any], reason: str, provenance: Mapping[str, Any]) -> None:
    gate = {
        "stage": STAGE,
        "gate": f"{BLOCK_GATE_PREFIX}_{reason}",
        "decision": "H4M_X_BLOCKED",
        "exact_next_gate": f"STOP_{BLOCK_GATE_PREFIX}",
        "block_reason": reason,
    }
    empty = {"stage": STAGE, "not_executed_or_incomplete": reason}
    for name in REQUIRED_ARTIFACTS:
        if name in {"manifest.json", "gate_matrix.json"} or (root / name).exists():
            continue
        if name == "final_report.md":
            (root / name).write_text(f"# H4M-X\n\ngate = {gate['gate']}\nblock_reason = {reason}\nSTOP.\n", encoding="utf-8")
        elif name == "test_results.stdout.txt":
            (root / name).write_text(f"not executed: {reason}\n", encoding="utf-8")
        else:
            write_json(root / name, binding if name == "repair_binding.json" else empty)
    write_json(root / "gate_matrix.json", gate)
    write_json(root / "manifest.json", make_manifest(root, gate, provenance))
    print(f"[H4M-X] artifact root: {root}\n[H4M-X] gate: {gate['gate']}\n[H4M-X] block_reason: {reason}")


def main() -> None:
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_x_window_boundary_repair_implementation_equivalence_validation_{stamp}"
    root.mkdir(parents=True, exist_ok=True)

    provenance = source_provenance(created_at)
    compile_audit = py_compile_audit()
    binding = repair_binding(created_at, provenance, compile_audit)
    write_json(root / "repair_binding.json", binding)
    if not binding["repair_binding_passed"]:
        write_block(root, binding, "REPAIR_BINDING_MISMATCH", provenance)
        return

    tests = run_tests(root)
    changed = changed_files_audit(provenance)
    boundary = boundary_validation(tests)
    credit = credit_horizon_validation(tests)
    signs = context_sign_validation(tests)
    equivalence = equivalence_validation(tests, changed)
    checkpoints = checkpoint_compatibility(tests)
    gate = gate_matrix(binding, boundary, credit, signs, equivalence, checkpoints, changed, tests)

    for name, payload in {
        "boundary_validation.json": boundary,
        "credit_horizon_validation.json": credit,
        "context_sign_validation.json": signs,
        "equivalence_validation.json": equivalence,
        "checkpoint_compatibility.json": checkpoints,
        "changed_files.json": changed,
        "gate_matrix.json": gate,
    }.items():
        write_json(root / name, payload)
    (root / "final_report.md").write_text(
        final_report(binding, boundary, credit, signs, equivalence, checkpoints, changed, tests, gate), encoding="utf-8"
    )
    write_json(root / "manifest.json", make_manifest(root, gate, provenance))

    print(f"[H4M-X] artifact root: {root}")
    print(f"[H4M-X] gate: {gate['gate']}")
    print(f"[H4M-X] w1_repair_contract_sha256: {EXPECTED['w1_repair_contract_sha256']}")
    print(f"[H4M-X] source_only_pass_commit: {provenance['source_commit_after_pass']}")
    print(f"[H4M-X] failing criteria: {gate['failing_criteria']}")
    print(f"[H4M-X] exact next gate: {gate['exact_next_gate']} (not executed)")


if __name__ == "__main__":
    main()
