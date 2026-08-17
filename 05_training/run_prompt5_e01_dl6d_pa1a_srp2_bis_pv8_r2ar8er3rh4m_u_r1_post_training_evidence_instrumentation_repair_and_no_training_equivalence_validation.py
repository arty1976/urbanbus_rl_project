#!/usr/bin/env python3
"""H4M-U-R1 post-training evidence instrumentation repair and validation.

This stage repairs the training-evidence recording device and the reporting
path only.  It runs fixture/mock validations; it does not train, build an
optimizer, step an optimizer, retrain, promote a checkpoint, open TEST6,
mutate the database, or push to GitHub.  The permanently incomplete H4M-U
per-cycle critic loss and parameter deltas are not reconstructed.
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


STAGE = "PV8-R2A-R8E-R3-R-H4M-U-R1"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_U_R1_"
    "POST_TRAINING_EVIDENCE_INSTRUMENTATION_REPAIR_AND_NO_TRAINING_EQUIVALENCE_VALIDATION_COMPLETE"
)
BLOCK_GATE_PREFIX = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_U_R1"
NEXT_GATE = (
    "H4M-U-R2_FRESH_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_AND_"
    "CRITIC_VALUE_TARGET_REPAIRED_THREE_SEED_RETRAINING_WITH_DURABLE_EVIDENCE"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"

SOURCE_REL = Path("05_training") / Path(__file__).name
DURABLE_REL = Path("05_training/durable_training_evidence.py")
H4MU_REL = Path(
    "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_u_fresh_actor_head_and_critic_target_repaired_three_seed_retraining.py"
)
TEST_REL = Path("05_training/test_h4m_u_r1_durable_training_evidence.py")
INTENDED_CHANGED_FILES = [
    DURABLE_REL.as_posix(),
    H4MU_REL.as_posix(),
    TEST_REL.as_posix(),
    SOURCE_REL.as_posix(),
]

IMMUTABLE_SOURCES = {
    "reward_v2": "05_training/rewards/mappo_reward_v1.py",
    "dl1_actor_critic_gae": "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "dl4_return_normalizer": "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py",
    "observation_contract": "05_training/observation_target_context_repair.py",
    "h4m_g_instrumented_execution": "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py",
    "h4m_h_trace_helpers": "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_h_fresh_instrumented_credit_diagnostic_retraining.py",
    "h4m_k_training_loop": "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_k_fresh_target_context_repaired_three_seed_retraining.py",
    "h4m_q_execution_lineage": "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining.py",
    "h4m_t_critic_target_repair": "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_t_ppo_credit_advantage_critic_value_target_repair_implementation_equivalence_validation.py",
    "h4m_u_recovery_audit": "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_u_post_training_evidence_recovery_audit.py",
}

H4MU_RAW_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_u_fresh_target_conditioned_actor_head_specialized_"
    "and_critic_value_target_repaired_three_seed_retraining_20260817_193849+09:00"
)
H4MU_RECOVERY_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_u_post_training_evidence_recovery_audit_20260817_195409+09:00"
H4MT_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_t_ppo_credit_advantage_critic_value_target_repair_implementation_equivalence_validation_20260817_190257+09:00"
)
H4MQ_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining_20260817_153440+0900"

EXPECTED = {
    "source_before_commit": "483913d316ee926a7f5c293d16ae461185adc561",
    "h4m_u_raw_training_commit": "5584d59cbf64f6bad7ba63c4770523fb04bc619c",
    "h4m_u_raw_training_blob": "2abdfad1b00d0afa7bd2228001bb7eecc5017f1d",
    "h4m_t_source_commit": "04d38ad3ba596645c66e7772cd0612b66dd29e00",
    "h4m_u_recovery_gate": "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_U_POST_TRAINING_EVIDENCE_NOT_RECOVERABLE",
    "h4m_u_raw_gate": "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_U_EXECUTION_EXCEPTION_AttributeError",
    "h4m_t_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_T_PPO_CREDIT_ADVANTAGE_CRITIC_VALUE_TARGET_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE",
    "actor_repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
    "critic_repair_contract_sha256": "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f",
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "seeds": [1, 2, 3],
    "outer_training_count": 11,
}

H4MU_EVIDENCE_STATUS = ["DIAGNOSTIC_ONLY", "NON_PROMOTABLE", "INCOMPLETE_EVIDENCE"]

REQUIRED_ARTIFACTS = [
    "final_report.md",
    "manifest.json",
    "evidence_schema.json",
    "instrumentation_validation.json",
    "report_recovery_validation.json",
    "equivalence_validation.json",
    "changed_files.json",
    "test_results.json",
    "test_results.stdout.txt",
    "repair_binding.json",
    "gate_matrix.json",
]


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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n", encoding="utf-8"
    )


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


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def py_compile_audit() -> Dict[str, Any]:
    rows = []
    with tempfile.TemporaryDirectory(prefix="h4mur1_pycompile_") as tmp:
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
        "git_diff_cached_check_stdout": cached.stdout.strip(),
    }


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    parent = git_run(["rev-parse", "HEAD^"]).stdout.strip()
    head_files = [line for line in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    status_short = git_run(["status", "--short"]).stdout.strip()
    latest = {rel: git_run(["log", "-1", "--format=%H", "--", rel]).stdout.strip() for rel in INTENDED_CHANGED_FILES}
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_commit_before_work": parent,
        "source_commit_after_pass": head,
        "head_commit_files": head_files,
        "intended_changed_files": INTENDED_CHANGED_FILES,
        "head_commit_intended_source_test_only": sorted(head_files) == sorted(INTENDED_CHANGED_FILES),
        "source_parent_matches_h4m_u_recovery": parent == EXPECTED["source_before_commit"],
        "all_intended_files_latest_at_head": all(value == head for value in latest.values()),
        "status_short": status_short,
        "clean_worktree": status_short == "",
        "github_push_performed": False,
    }


def historical_artifact_integrity() -> Dict[str, Any]:
    """Prove the historical H4M-U artifacts were not modified by this stage."""
    rows = []
    for label, root in (("h4m_u_raw_training", H4MU_RAW_ROOT), ("h4m_u_recovery_audit", H4MU_RECOVERY_ROOT)):
        manifest = read_json(root / "manifest.json")
        recorded = manifest.get("output_sha256", {})
        mismatched = []
        missing = []
        for name, expected_sha in recorded.items():
            path = root / name
            if not path.exists():
                missing.append(name)
            elif sha256_file(path) != expected_sha:
                mismatched.append(name)
        rows.append(
            {
                "artifact": label,
                "artifact_root": str(root),
                "gate": manifest.get("gate"),
                "recorded_file_count": len(recorded),
                "missing_files": missing,
                "mismatched_files": mismatched,
                "unmodified": not missing and not mismatched,
            }
        )
    return {
        "rows": rows,
        "historical_artifacts_unmodified": all(row["unmodified"] for row in rows),
        "policy": "historical failure artifacts are append-only evidence; R1 proves the past fact from the source-before blob instead of editing them",
    }


def authoritative_binding(created_at: str, provenance: Mapping[str, Any], compile_audit: Mapping[str, Any], historical: Mapping[str, Any]) -> Dict[str, Any]:
    raw_gate = read_json(H4MU_RAW_ROOT / "gate_matrix.json")
    raw_binding = read_json(H4MU_RAW_ROOT / "repair_binding.json")
    recovery_gate = read_json(H4MU_RECOVERY_ROOT / "gate_matrix.json")
    recovery_evidence = read_json(H4MU_RECOVERY_ROOT / "evidence_recoverability.json")
    recovery_traces = read_json(H4MU_RECOVERY_ROOT / "raw_execution_completeness.json")
    recovery_checkpoints = read_json(H4MU_RECOVERY_ROOT / "checkpoint_integrity.json")
    t_gate = read_json(H4MT_ROOT / "gate_matrix.json")
    q_sha = read_json(H4MQ_ROOT / "repair_binding.json").get("sha_bindings", {})
    unrecoverable = recovery_evidence.get("unrecoverable_required_metrics", [])
    checks = {
        "source_parent_matches_h4m_u_recovery": provenance.get("source_parent_matches_h4m_u_recovery") is True,
        "source_only_local_commit": provenance.get("head_commit_intended_source_test_only") is True
        and provenance.get("all_intended_files_latest_at_head") is True
        and provenance.get("clean_worktree") is True,
        "py_compile_passed": compile_audit.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_audit.get("git_diff_cached_check_passed") is True,
        "h4m_u_raw_gate_bound": raw_gate.get("gate") == EXPECTED["h4m_u_raw_gate"],
        "h4m_u_raw_training_commit_bound": raw_binding.get("source_provenance", {}).get("source_commit_before_run")
        == EXPECTED["h4m_u_raw_training_commit"],
        "h4m_u_recovery_gate_bound": recovery_gate.get("gate") == EXPECTED["h4m_u_recovery_gate"],
        "h4m_u_missing_metrics_bound": sorted(unrecoverable)
        == sorted(["actual_critic_loss", "cycle_actor_parameter_delta", "cycle_critic_parameter_delta"]),
        "h4m_u_raw_execution_integrity_bound": recovery_traces.get("integrity_passed") is True,
        "h4m_u_checkpoint_integrity_bound": recovery_checkpoints.get("checkpoint_integrity_passed") is True,
        "h4m_t_gate_bound": t_gate.get("gate") == EXPECTED["h4m_t_gate"],
        "critic_repair_sha_bound": t_gate.get("repair_contract_sha256") == EXPECTED["critic_repair_contract_sha256"],
        "actor_repair_sha_bound": q_sha.get("h4m_p_repair_contract_sha256") == EXPECTED["actor_repair_contract_sha256"],
        "reward_v2_sha_bound": q_sha.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "split_sha_bound": q_sha.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "schedule_sha_bound": q_sha.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "zero_loss_sha_bound": q_sha.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "historical_artifacts_unmodified": historical.get("historical_artifacts_unmodified") is True,
    }
    return {
        "stage": STAGE,
        "created_at": created_at,
        "source_provenance": provenance,
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "repair_scope": "training evidence instrumentation and reporting path only",
        "repair_contracts": {
            "actor": EXPECTED["actor_repair_contract_sha256"],
            "critic": EXPECTED["critic_repair_contract_sha256"],
        },
        "frozen_sha_bindings": {
            "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
            "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        },
        "h4m_u_prior_evidence": {
            "artifact_root": str(H4MU_RAW_ROOT),
            "status": H4MU_EVIDENCE_STATUS,
            "permanently_unrecoverable_metrics": unrecoverable,
            "reconstruction_attempted": False,
            "estimation_attempted": False,
            "reused_for_promotion": False,
            "reused_as_r1_input": False,
        },
        "historical_artifact_integrity": historical,
        "hard_lock_attestation": {
            "reward_v2_modified": False,
            "s3_critic_value_target_repair_modified": False,
            "target_conditioned_actor_head_repair_modified": False,
            "td_gae_equations_modified": False,
            "advantage_normalization_modified": False,
            "actor_logits_modified": False,
            "action_or_k_mask_modified": False,
            "observation_contract_modified": False,
            "simulator_or_zero_loss_modified": False,
            "split_seed_schedule_or_budget_modified": False,
            "mappo_training_executed": False,
            "optimizer_created_or_stepped": False,
            "fresh_retraining_executed": False,
            "checkpoint_promoted": False,
            "test6_opened_or_used": False,
            "database_or_data_mutated": False,
            "github_push_performed": False,
        },
    }


def run_tests(artifact_root: Path) -> Dict[str, Any]:
    output_path = artifact_root / "test_results.json"
    command = [sys.executable, str(PROJECT_ROOT / TEST_REL), "--json-output", str(output_path)]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    (artifact_root / "test_results.stdout.txt").write_text(
        "command:\n" + " ".join(command) + "\n\nstdout:\n" + completed.stdout + "\nstderr:\n" + completed.stderr,
        encoding="utf-8",
    )
    if output_path.exists():
        payload = read_json(output_path)
    else:
        payload = {"passed": False, "reason": "TEST_RESULT_JSON_NOT_WRITTEN"}
        write_json(output_path, payload)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "payload": payload,
        "passed": completed.returncode == 0 and payload.get("passed") is True,
    }


def section(tests: Mapping[str, Any], name: str) -> Dict[str, Any]:
    return tests.get("payload", {}).get("sections", {}).get(name, {})


def instrumentation_validation(tests: Mapping[str, Any]) -> Dict[str, Any]:
    reporting = section(tests, "reporting_call_binding")
    schema = section(tests, "evidence_schema")
    critic_loss = section(tests, "critic_loss_provenance")
    delta = section(tests, "parameter_delta_provenance")
    wiring = section(tests, "runner_instrumentation_wiring")
    binder = section(tests, "runner_binder_integration")
    checks = {
        "source_level_reporting_typo_fixed": reporting.get("passed") is True,
        "historical_typo_proven_from_source_before_blob": reporting.get("historical_typo_present_in_source_before_blob") is True
        and reporting.get("source_before_blob") == EXPECTED["h4m_u_raw_training_blob"],
        "current_source_binds_owning_module": reporting.get("current_source_binds_owning_module") is True,
        "seed_cycle_schema_complete": schema.get("passed") is True,
        "record_hash_covers_run_identity": schema.get("record_hash_covers_identity") is True,
        "critic_loss_from_actual_update_result": critic_loss.get("passed") is True
        and critic_loss.get("values_match_update_result_exactly") is True
        and critic_loss.get("estimated") is False,
        "parameter_delta_from_real_pre_post_parameters": delta.get("passed") is True
        and delta.get("matches_analytic") is True
        and delta.get("pre_post_hashes_differ") is True,
        "runner_persists_before_advancing_cycle": wiring.get("passed") is True
        and wiring.get("markers", {}).get("flush_bound_after_trace_write") is True
        and wiring.get("markers", {}).get("flush_failure_not_swallowed") is True,
        "runner_binder_persists_real_cycle_evidence": binder.get("passed") is True
        and binder.get("actor_delta_matches_real_parameters") is True
        and binder.get("critic_delta_matches_real_parameters") is True,
        "trace_without_evidence_fail_closed": binder.get("flush_without_staged_update_fail_closed", {}).get("raised") is True,
    }
    return {
        "stage": STAGE,
        "repair_summary": {
            "1_reporting_exception": "the post-training conditional-parquet call is bound to its owning H4M-K module",
            "2_immediate_persistence": "each seed x cycle record is appended, flushed, fsynced and read-back verified before the loop advances",
            "3_critic_loss": "persisted verbatim from the real update result; missing or non-finite values fail closed instead of being estimated",
            "4_actor_parameter_delta": "computed from the real pre-cycle and post-cycle Actor parameters",
            "5_critic_parameter_delta": "computed from the real pre-cycle and post-cycle Critic parameters",
            "6_credit_diagnostics": "V(s)-return, raw GAE, normalized advantage, action counts and probabilities persisted per cycle",
            "7_durability": "append-only JSONL for records, atomic replace for index/state sidecars",
            "8_flush_ordering": "evidence flush happens after the cycle trace write and before the next cycle",
            "9_report_source": "final_report is rendered from reloaded persisted evidence, not volatile in-memory metrics",
            "10_recovery": "--regenerate-report rebuilds the report from persisted evidence with zero training",
        },
        "evidence_stop_policy": "an evidence persistence failure raises out of the training loop so trace and evidence never sit on different cycles",
        "reporting_call_binding": reporting,
        "evidence_schema_validation": schema,
        "critic_loss_provenance": critic_loss,
        "parameter_delta_provenance": delta,
        "runner_instrumentation_wiring": wiring,
        "runner_binder_integration": binder,
        "checks": checks,
        "passed": all(checks.values()),
    }


def report_recovery_validation(tests: Mapping[str, Any]) -> Dict[str, Any]:
    survival = section(tests, "reporter_failure_survival")
    subprocess_recovery = section(tests, "report_regeneration_without_torch")
    fail_closed = section(tests, "fail_closed_integrity")
    partial = fail_closed.get("partial_record", {})
    duplicate = fail_closed.get("duplicate_and_conflict", {})
    identity = fail_closed.get("identity_and_completeness", {})
    checks = {
        "evidence_survives_forced_reporter_failure": survival.get("passed") is True
        and survival.get("evidence_bytes_unchanged_by_failure") is True,
        "report_regenerated_from_persisted_evidence": survival.get("recovered") is True
        and survival.get("report_source") == "PERSISTED_DURABLE_EVIDENCE_ONLY",
        "regeneration_training_count_zero": survival.get("regeneration_training_invocations") == 0,
        "regeneration_optimizer_step_count_zero": survival.get("regeneration_optimizer_steps") == 0
        and survival.get("regeneration_backward_calls") == 0,
        "regeneration_needs_no_torch_or_device": subprocess_recovery.get("passed") is True
        and subprocess_recovery.get("subprocess_result", {}).get("torch_in_sys_modules") is False,
        "partial_artifact_detected_fail_closed": partial.get("detected") is True
        and partial.get("regeneration_refuses_pass") is True,
        "duplicate_cycle_evidence_rejected": duplicate.get("writer_rejects_duplicate", {}).get("raised") is True,
        "conflicting_cycle_evidence_rejected": duplicate.get("writer_rejects_conflict", {}).get("raised") is True
        and duplicate.get("loader_integrity_passed") is False,
        "foreign_run_identity_rejected": identity.get("foreign_identity_rejected", {}).get("raised") is True,
        "incomplete_grid_blocks_report": identity.get("incomplete_grid_blocks_report") is True,
    }
    return {
        "stage": STAGE,
        "reporter_failure_survival": survival,
        "torch_free_regeneration": subprocess_recovery,
        "fail_closed_integrity": fail_closed,
        "recovery_command": f"{sys.executable} {H4MU_REL.as_posix()} --regenerate-report <ARTIFACT_ROOT>",
        "training_invocations": 0,
        "optimizer_step_count": 0,
        "checks": checks,
        "passed": all(checks.values()),
    }


def max_abs_diff(numerical: Mapping[str, Any], key: str) -> float:
    """Reported diff for ``key``; a missing diff is treated as a failure."""
    value = ((numerical.get("diffs") or {}).get(key) or {}).get("max_abs_diff")
    return float("inf") if value is None else float(value)


def equivalence_validation(tests: Mapping[str, Any], changed: Mapping[str, Any]) -> Dict[str, Any]:
    numerical = section(tests, "numerical_equivalence")
    leakage = section(tests, "leakage_and_isolation")
    payload = tests.get("payload", {})
    checks = {
        "actor_logits_and_probabilities_unchanged": numerical.get("passed") is True,
        "critic_values_unchanged": max_abs_diff(numerical, "critic_values") <= 1.0e-7,
        "td_gae_and_advantage_normalization_unchanged": max_abs_diff(numerical, "gae_raw_advantages") <= 1.0e-7
        and max_abs_diff(numerical, "gae_normalized_advantages") <= 1.0e-7
        and max_abs_diff(numerical, "gae_matches_canonical_recursion") <= 1.0e-7,
        "reward_v2_unchanged": numerical.get("reward_total_unchanged") is True
        and numerical.get("reward_freeze_sha256") == EXPECTED["reward_v2_sha256"],
        "k_mask_unchanged": numerical.get("k_mask_unchanged") is True,
        "immutable_sources_unchanged": changed.get("immutable_sources_unchanged") is True,
        "nan_inf_zero": payload.get("nan_inf_count") == 0,
        "future_leakage_zero": leakage.get("future_leakage_count") == 0 and leakage.get("cross_cycle_trace_binding_count") == 0,
        "test6_access_zero": payload.get("test6_access_count") == 0
        and all(count == 0 for count in (leakage.get("test6_path_reference_counts") or {"x": 1}).values()),
        "training_zero": payload.get("training_executed") is False and payload.get("training_invocations") == 0,
        "optimizer_step_zero": payload.get("optimizer_step_count") == 0,
    }
    return {
        "stage": STAGE,
        "numerical_equivalence": numerical,
        "leakage_and_isolation": leakage,
        "immutable_contract_diff": changed.get("immutable_source_diff"),
        "checks": checks,
        "passed": all(checks.values()),
    }


def changed_files_audit(provenance: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for rel in INTENDED_CHANGED_FILES:
        path = PROJECT_ROOT / rel
        status = "modified" if git_run(["cat-file", "-e", f"HEAD^:{rel}"], check=False).returncode == 0 else "added"
        rows.append(
            {
                "path": rel,
                "change_type": status,
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
    return {
        "stage": STAGE,
        "source_before_commit": EXPECTED["source_before_commit"],
        "source_only_pass_commit": provenance.get("source_commit_after_pass"),
        "changed_files": rows,
        "changed_file_count": len(rows),
        "only_intended_files_changed": provenance.get("head_commit_intended_source_test_only") is True,
        "artifact_or_log_committed": any("artifacts/" in row["path"] for row in rows),
        "immutable_source_diff": immutable,
        "immutable_sources_unchanged": all(entry["unchanged_in_head_commit"] for entry in immutable.values()),
        "github_push_performed": False,
        "passed": provenance.get("head_commit_intended_source_test_only") is True
        and all(entry["unchanged_in_head_commit"] for entry in immutable.values()),
    }


def gate_matrix(
    binding: Mapping[str, Any],
    instrumentation: Mapping[str, Any],
    recovery: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    changed: Mapping[str, Any],
    tests: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = tests.get("payload", {})
    criteria = {
        "authoritative_binding_passed": binding.get("authoritative_binding_passed") is True,
        "source_only_commit": changed.get("only_intended_files_changed") is True
        and changed.get("artifact_or_log_committed") is False,
        "immutable_contracts_unchanged": changed.get("immutable_sources_unchanged") is True,
        "instrumentation_repair_validated": instrumentation.get("passed") is True,
        "report_recovery_validated": recovery.get("passed") is True,
        "no_training_equivalence_validated": equivalence.get("passed") is True,
        "focused_tests_passed": tests.get("passed") is True,
        "training_count_zero": payload.get("training_invocations") == 0 and payload.get("training_executed") is False,
        "optimizer_step_count_zero": payload.get("optimizer_step_count") == 0,
        "nan_inf_zero": payload.get("nan_inf_count") == 0,
        "future_leakage_zero": payload.get("future_leakage_count") == 0,
        "test6_access_zero": payload.get("test6_access_count") == 0,
        "h4m_u_missing_evidence_not_reconstructed": binding.get("h4m_u_prior_evidence", {}).get("reconstruction_attempted") is False
        and binding.get("h4m_u_prior_evidence", {}).get("estimation_attempted") is False,
        "h4m_u_artifacts_remain_non_promotable": binding.get("h4m_u_prior_evidence", {}).get("status") == H4MU_EVIDENCE_STATUS
        and binding.get("h4m_u_prior_evidence", {}).get("reused_for_promotion") is False,
        "historical_artifacts_unmodified": binding.get("historical_artifact_integrity", {}).get("historical_artifacts_unmodified") is True,
        "github_push_false": changed.get("github_push_performed") is False,
    }
    passed = all(criteria.values())
    return {
        "stage": STAGE,
        "gate": PASS_GATE if passed else f"{BLOCK_GATE_PREFIX}_VALIDATION_FAILED",
        "decision": "POST_TRAINING_EVIDENCE_INSTRUMENTATION_REPAIRED_AND_NO_TRAINING_EQUIVALENCE_VALIDATED"
        if passed
        else "H4M_U_R1_BLOCKED",
        "exact_next_gate": NEXT_GATE if passed else f"STOP_{BLOCK_GATE_PREFIX}",
        "next_gate_auto_execution": False,
        "criteria": criteria,
        "failing_criteria": [key for key, value in criteria.items() if not value],
        "final_flags": {
            "training_executed": False,
            "training_count": 0,
            "optimizer_step_count": 0,
            "TEST6_opened": False,
            "test6_access_count": 0,
            "checkpoint_promoted": False,
            "h4m_u_evidence_reconstructed": False,
            "github_push_performed": False,
        },
    }


def make_manifest(artifact_root: Path, gate: Mapping[str, Any], provenance: Mapping[str, Any]) -> Dict[str, Any]:
    files = {
        path.relative_to(artifact_root).as_posix(): str(path)
        for path in artifact_root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    return {
        "stage": STAGE,
        "artifact_root": str(artifact_root),
        "source_commit_before_work": provenance.get("source_commit_before_work"),
        "source_only_pass_commit": provenance.get("source_commit_after_pass"),
        "changed_files": INTENDED_CHANGED_FILES,
        "gate": gate.get("gate"),
        "decision": gate.get("decision"),
        "exact_next_gate": gate.get("exact_next_gate"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "manifest_self_hash_policy": "manifest.json excluded to avoid self-reference",
        "append_only_artifact": True,
        "training_count": 0,
        "optimizer_step_count": 0,
        "TEST6_opened": False,
        "test6_access_count": 0,
        "github_push_performed": False,
    }


def final_report(
    binding: Mapping[str, Any],
    instrumentation: Mapping[str, Any],
    recovery: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    changed: Mapping[str, Any],
    tests: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> str:
    provenance = binding["source_provenance"]
    return f"""# H4M-U-R1 Post-Training Evidence Instrumentation Repair and No-Training Equivalence Validation

gate = {gate['gate']}
decision = {gate['decision']}
source_before_commit = {EXPECTED['source_before_commit']}
source_only_pass_commit = {provenance['source_commit_after_pass']}
actor_repair_contract_sha256 = {EXPECTED['actor_repair_contract_sha256']}
critic_repair_contract_sha256 = {EXPECTED['critic_repair_contract_sha256']}
training_count = 0
optimizer_step_count = 0
TEST6_access_count = 0
github_push_performed = false
exact_next_gate = {gate['exact_next_gate']} (not executed automatically)

## What was repaired

The H4M-U failure was an evidence-instrumentation failure, not a training failure. The 33 seed x cycle
rollouts completed; the post-training reporter then raised `AttributeError` because
`write_conditional_parquet` was called on the H4M-Q module instead of its owning H4M-K module, and the
per-cycle critic loss and parameter deltas only ever existed in the update result held in memory.

R1 repairs the recording device and the reporting path only:

```json
{json.dumps(instrumentation['repair_summary'], ensure_ascii=False, indent=2)}
```

Evidence stop policy: {instrumentation['evidence_stop_policy']}

## Historical fact vs current state

The historical typo is proven from the source-before blob, not from the current worktree; the current
source is proven repaired. Historical H4M-U artifacts were not edited.

```json
{json.dumps({
    'historical_call_owners': instrumentation['reporting_call_binding'].get('historical_call_owners'),
    'source_before_commit': instrumentation['reporting_call_binding'].get('source_before_commit'),
    'source_before_blob': instrumentation['reporting_call_binding'].get('source_before_blob'),
    'current_call_owners': instrumentation['reporting_call_binding'].get('current_call_owners'),
    'historical_artifacts_unmodified': binding['historical_artifact_integrity']['historical_artifacts_unmodified'],
}, ensure_ascii=False, indent=2)}
```

## Validation

```json
{json.dumps({
    'instrumentation': instrumentation['checks'],
    'report_recovery': recovery['checks'],
    'no_training_equivalence': equivalence['checks'],
    'tests': {'command': tests.get('command'), 'returncode': tests.get('returncode'), 'passed': tests.get('passed')},
}, ensure_ascii=False, indent=2, default=jsonable)}
```

## Changed files

```json
{json.dumps({'changed_files': changed['changed_files'], 'immutable_sources_unchanged': changed['immutable_sources_unchanged']}, ensure_ascii=False, indent=2, default=jsonable)}
```

## H4M-U prior evidence

```json
{json.dumps(binding['h4m_u_prior_evidence'], ensure_ascii=False, indent=2, default=jsonable)}
```

The permanently incomplete H4M-U per-cycle critic loss and parameter deltas were not estimated,
reconstructed, or synthesized, and the existing H4M-U checkpoints and traces stay
DIAGNOSTIC_ONLY / NON_PROMOTABLE / INCOMPLETE_EVIDENCE.

STOP: no MAPPO training, no optimizer step, no fresh retraining, no checkpoint promotion, no TEST6
access, no reward/model/hyperparameter tuning, no DB or data mutation, and no GitHub push occurred.
H4M-U-R2 is not executed automatically.
"""


def write_block(artifact_root: Path, binding: Mapping[str, Any], reason: str, provenance: Mapping[str, Any]) -> None:
    gate = {
        "stage": STAGE,
        "gate": f"{BLOCK_GATE_PREFIX}_{reason}",
        "decision": "H4M_U_R1_BLOCKED",
        "exact_next_gate": f"STOP_{BLOCK_GATE_PREFIX}",
        "block_reason": reason,
    }
    empty = {"stage": STAGE, "not_executed_or_incomplete": reason}
    for name in REQUIRED_ARTIFACTS:
        if name in {"manifest.json", "gate_matrix.json"}:
            continue
        if name == "final_report.md":
            (artifact_root / name).write_text(
                f"# H4M-U-R1\n\ngate = {gate['gate']}\nblock_reason = {reason}\n\nSTOP. No training, optimizer step, TEST6 access, or GitHub push occurred.\n",
                encoding="utf-8",
            )
        elif name == "test_results.stdout.txt":
            if not (artifact_root / name).exists():
                (artifact_root / name).write_text(f"not executed: {reason}\n", encoding="utf-8")
        elif not (artifact_root / name).exists():
            write_json(artifact_root / name, binding if name == "repair_binding.json" else empty)
    write_json(artifact_root / "gate_matrix.json", gate)
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate, provenance))
    print(f"[H4M-U-R1] artifact root: {artifact_root}\n[H4M-U-R1] gate: {gate['gate']}\n[H4M-U-R1] block_reason: {reason}")


def main() -> None:
    created_at = kst_now().isoformat()
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    artifact_root = ARTIFACTS_ROOT / (
        "pv8_r2a_r8e_r3_r_h4m_u_r1_post_training_evidence_instrumentation_repair_and_"
        f"no_training_equivalence_validation_{stamp}"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    provenance = source_provenance(created_at)
    compile_audit = py_compile_audit()
    historical = historical_artifact_integrity()
    binding = authoritative_binding(created_at, provenance, compile_audit, historical)
    write_json(artifact_root / "repair_binding.json", binding)
    if not binding["authoritative_binding_passed"]:
        write_block(artifact_root, binding, "AUTHORITATIVE_BINDING_MISMATCH", provenance)
        return

    dte = import_module("h4mur1_runner_dte", PROJECT_ROOT / DURABLE_REL)
    write_json(artifact_root / "evidence_schema.json", dte.evidence_schema())

    tests = run_tests(artifact_root)
    changed = changed_files_audit(provenance)
    instrumentation = instrumentation_validation(tests)
    recovery = report_recovery_validation(tests)
    equivalence = equivalence_validation(tests, changed)
    gate = gate_matrix(binding, instrumentation, recovery, equivalence, changed, tests)

    write_json(artifact_root / "instrumentation_validation.json", instrumentation)
    write_json(artifact_root / "report_recovery_validation.json", recovery)
    write_json(artifact_root / "equivalence_validation.json", equivalence)
    write_json(artifact_root / "changed_files.json", changed)
    write_json(artifact_root / "gate_matrix.json", gate)
    (artifact_root / "final_report.md").write_text(
        final_report(binding, instrumentation, recovery, equivalence, changed, tests, gate), encoding="utf-8"
    )
    write_json(artifact_root / "manifest.json", make_manifest(artifact_root, gate, provenance))

    print(f"[H4M-U-R1] artifact root: {artifact_root}")
    print(f"[H4M-U-R1] gate: {gate['gate']}")
    print(f"[H4M-U-R1] source_before_commit: {EXPECTED['source_before_commit']}")
    print(f"[H4M-U-R1] source_only_pass_commit: {provenance['source_commit_after_pass']}")
    print(f"[H4M-U-R1] training_count=0 optimizer_step_count=0 test6_access_count=0 github_push=False")
    print(f"[H4M-U-R1] exact_next_gate: {gate['exact_next_gate']} (not executed automatically)")
    if not gate["criteria"]["focused_tests_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
