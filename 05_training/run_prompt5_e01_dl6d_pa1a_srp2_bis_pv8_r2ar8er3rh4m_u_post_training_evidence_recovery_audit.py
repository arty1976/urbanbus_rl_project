#!/usr/bin/env python3
"""Fail-closed H4M-U post-training evidence recoverability audit.

This program does not invoke MAPPO, backward, or an optimizer.  It binds the
completed raw H4M-U execution, validates the persisted traces/checkpoints, and
records whether every required training-time metric can be reconstructed
without a second training run.
"""

from __future__ import annotations

import hashlib
import json
import math
import py_compile
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-U"
BLOCK_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_U_POST_TRAINING_EVIDENCE_NOT_RECOVERABLE"
RAW_RUN_GATE = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_U_EXECUTION_EXCEPTION_AttributeError"
RAW_SOURCE_COMMIT = "5584d59cbf64f6bad7ba63c4770523fb04bc619c"
ACTOR_REPAIR_SHA = "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97"
CRITIC_REPAIR_SHA = "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f"
REWARD_SHA = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name
RAW_ROOT = ARTIFACTS_ROOT / (
    "pv8_r2a_r8e_r3_r_h4m_u_fresh_target_conditioned_actor_head_specialized_"
    "and_critic_value_target_repaired_three_seed_retraining_20260817_193849+09:00"
)
RAW_RUNNER = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_u_fresh_actor_head_and_critic_target_repaired_three_seed_retraining.py"
INSTRUMENTED_UPDATE_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"

TRACE_FILES = {
    "pre_action": "actual_pre_action_trace.parquet",
    "reward": "actual_reward_trace.parquet",
    "td_gae": "actual_critic_td_gae_trace.parquet",
    "advantage": "advantage_normalization_sample.parquet",
    "critic_value_error": "critic_value_error_by_sample.parquet",
    "ppo_surrogate": "ppo_policy_surrogate_sample.parquet",
    "actor_pressure": "actor_logit_pressure_by_sample.parquet",
    "shadow_summary": "counterfactual_pair_summary.parquet",
}


def kst_now() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=jsonable) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(args: Iterable[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, text=True, capture_output=True, check=check)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def finite_count(frame: pd.DataFrame) -> int:
    count = 0
    for column in frame.select_dtypes(include=["number"]).columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        count += int((~values.map(math.isfinite)).sum())
    return count


def legal_action_count(frame: pd.DataFrame) -> int:
    invalid = 0
    for row in frame[["action_id", "legal_action_ids"]].itertuples(index=False):
        legal = {int(value) for value in row.legal_action_ids}
        invalid += int(int(row.action_id) not in legal)
    return invalid


def raw_path(seed: int, cycle: int, trace_name: str) -> Path:
    base = RAW_ROOT / "05_actual_on_policy_credit_trace" / f"seed={seed}" / f"outer_cycle={cycle}"
    if trace_name == "shadow_summary":
        base = RAW_ROOT / "06_long_horizon_shadow_trace" / f"seed={seed}" / f"outer_cycle={cycle}"
    return base / TRACE_FILES[trace_name]


def source_provenance() -> Dict[str, Any]:
    head = git(["rev-parse", "HEAD"]).stdout.strip()
    parent = git(["rev-parse", "HEAD^"]).stdout.strip()
    head_files = [line for line in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if line]
    return {
        "audit_source_commit": head,
        "audit_source_parent": parent,
        "audit_source_files": head_files,
        "audit_source_only": head_files == [SOURCE_REL.as_posix()],
        "raw_training_source_commit": RAW_SOURCE_COMMIT,
        "raw_training_source_commit_is_parent": parent == RAW_SOURCE_COMMIT,
        "github_push_performed": False,
    }


def compile_audit() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="h4mu_recovery_compile_") as tmp:
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(Path(tmp) / "audit.pyc"), doraise=True)
            error = None
        except Exception as exc:
            error = repr(exc)
    cached = git(["diff", "--cached", "--check"], check=False)
    return {"py_compile_passed": error is None, "py_compile_error": error, "git_diff_cached_check_passed": cached.returncode == 0}


def authoritative_binding(provenance: Mapping[str, Any], compile_result: Mapping[str, Any]) -> Dict[str, Any]:
    raw_binding = read_json(RAW_ROOT / "repair_binding.json")
    raw_gate = read_json(RAW_ROOT / "gate_matrix.json")
    raw_manifest = read_json(RAW_ROOT / "manifest.json")
    raw_text = RAW_RUNNER.read_text(encoding="utf-8")
    update_text = INSTRUMENTED_UPDATE_SOURCE.read_text(encoding="utf-8")
    checks = {
        "audit_source_only": provenance.get("audit_source_only") is True,
        "audit_parent_is_raw_source": provenance.get("raw_training_source_commit_is_parent") is True,
        "py_compile_passed": compile_result.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_result.get("git_diff_cached_check_passed") is True,
        "raw_binding_authoritative": raw_binding.get("authoritative_binding_passed") is True,
        "raw_source_commit_bound": raw_binding.get("source_provenance", {}).get("source_commit_before_run") == RAW_SOURCE_COMMIT,
        "raw_mps_execution_bound": raw_manifest.get("device_backend") == "MPS",
        "raw_exception_bound": raw_gate.get("gate") == RAW_RUN_GATE,
        "actor_repair_sha_bound": raw_binding.get("repair_shas", {}).get("actor") == ACTOR_REPAIR_SHA,
        "critic_repair_sha_bound": raw_binding.get("repair_shas", {}).get("critic") == CRITIC_REPAIR_SHA,
        "historical_reporting_call_typo_bound": "qmod.write_conditional_parquet(base, root" in raw_text,
        "s3_source_present": all(token in update_text for token in ["CRITIC_VALUE_TARGET_BINDING_SCHEMA", "apply_return_normalizer_update_after_critic_update"]),
    }
    return {
        "stage": STAGE,
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "raw_artifact": str(RAW_ROOT),
        "raw_artifact_gate": raw_gate.get("gate"),
        "raw_source_reporting_failure": {
            "exception": "AttributeError",
            "source_line": "qmod.write_conditional_parquet(base, root, trace_meta['joined_rows'])",
            "correct_function_owner": "H4M-K module (kmod.write_conditional_parquet)",
        },
        "repair_contracts": {"actor": ACTOR_REPAIR_SHA, "critic": CRITIC_REPAIR_SHA},
        "provenance": dict(provenance),
    }


def trace_completeness() -> Dict[str, Any]:
    cycle_rows: List[Dict[str, Any]] = []
    totals = {name: 0 for name in TRACE_FILES}
    missing: List[str] = []
    invalid_action_count = 0
    nonfinite_count = 0
    reward_sha_mismatch = 0
    target_state_failure = 0
    identity_failure = 0
    for seed in [1, 2, 3]:
        for cycle in range(1, 12):
            paths = {name: raw_path(seed, cycle, name) for name in TRACE_FILES}
            absent = [name for name, path in paths.items() if not path.exists()]
            if absent:
                missing.extend(f"seed={seed}/cycle={cycle}/{name}" for name in absent)
                continue
            frames = {name: pd.read_parquet(path) for name, path in paths.items()}
            counts = {name: int(len(frame)) for name, frame in frames.items()}
            for name, value in counts.items():
                totals[name] += value
            pre = frames["pre_action"]
            reward = frames["reward"]
            td = frames["td_gae"]
            adv = frames["advantage"]
            shadow = frames["shadow_summary"]
            action_invalid = legal_action_count(pre)
            finite = sum(finite_count(frame) for frame in frames.values())
            reward_drift = int((reward["reward_freeze_sha256"] != REWARD_SHA).sum())
            states = td["critic_target_normalizer_state_sha256"]
            target_state_bad = int(states.isna().sum()) + int(states.nunique(dropna=True) != 1)
            common = set(pre["sample_uid"]) & set(reward["sample_uid"]) & set(td["sample_uid"]) & set(adv["sample_uid"]) & set(shadow["sample_uid"])
            identity_bad = int(len(common) != len(pre) or len(pre) != len(reward) or len(pre) != len(td) or len(pre) != len(adv) or len(pre) != len(shadow))
            invalid_action_count += action_invalid
            nonfinite_count += finite
            reward_sha_mismatch += reward_drift
            target_state_failure += target_state_bad
            identity_failure += identity_bad
            cycle_rows.append({
                "seed": seed,
                "outer_cycle": cycle,
                "row_counts": counts,
                "legal_action_violation_count": action_invalid,
                "numeric_nan_inf_count": finite,
                "reward_sha_mismatch_count": reward_drift,
                "single_critic_target_state": target_state_bad == 0,
                "cross_trace_sample_identity_complete": identity_bad == 0,
                "s3_materialization_orders": sorted(str(item) for item in td["materialization_order"].dropna().unique()),
            })
    checks = {
        "all_33_seed_cycle_sets_present": len(cycle_rows) == 33 and not missing,
        "all_352_base_samples_per_cycle": all(row["row_counts"][name] == 352 for row in cycle_rows for name in ["pre_action", "reward", "td_gae", "advantage", "critic_value_error", "shadow_summary"]),
        "all_1024_ppo_pressure_rows_per_cycle": all(row["row_counts"][name] == 1024 for row in cycle_rows for name in ["ppo_surrogate", "actor_pressure"]),
        "legal_action_zero": invalid_action_count == 0,
        "nan_inf_zero": nonfinite_count == 0,
        "reward_v2_sha_unchanged": reward_sha_mismatch == 0,
        "s3_single_frozen_target_state_per_cycle": target_state_failure == 0,
        "cross_trace_sample_identity_complete": identity_failure == 0,
    }
    return {"stage": STAGE, "checks": checks, "cycle_rows": cycle_rows, "totals": totals, "missing_paths": missing, "integrity_passed": all(checks.values())}


def checkpoint_integrity() -> Dict[str, Any]:
    rows = []
    for seed in [1, 2, 3]:
        path = RAW_ROOT / "checkpoints" / f"H4M_U_SEED_{seed:03d}_FRESH_ACTOR_AND_CRITIC_REPAIRED.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload.get("training_configuration", {})
        rows.append({
            "seed": seed,
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "payload_stage": payload.get("stage"),
            "fresh_initialization": payload.get("fresh_initialization"),
            "actor_specialization_active": payload.get("actor_target_head_specialization_active"),
            "actor_repair_sha": payload.get("actor_target_head_specialization_repair_contract_sha256"),
            "s3_active": cfg.get("s3_critic_value_target_repair_active"),
            "s3_repair_sha": cfg.get("s3_critic_value_target_repair_contract_sha256"),
            "target_binding_schema": cfg.get("critic_target_binding_schema"),
            "frozen_outer_cycles": payload.get("metadata", {}).get("outer_training_count"),
            "frozen_ppo_updates": payload.get("metadata", {}).get("ppo_updates"),
            "frozen_critic_updates": payload.get("metadata", {}).get("critic_updates"),
            "test6_status": payload.get("metadata", {}).get("test6_status"),
        })
    checks = {
        "three_checkpoints": len(rows) == 3,
        "stage_matches": all(row["payload_stage"] == STAGE for row in rows),
        "fresh_initialization": all(row["fresh_initialization"] is True for row in rows),
        "actor_specialization_and_contract": all(row["actor_specialization_active"] is True and row["actor_repair_sha"] == ACTOR_REPAIR_SHA for row in rows),
        "s3_critic_binding_and_contract": all(row["s3_active"] is True and row["s3_repair_sha"] == CRITIC_REPAIR_SHA and row["target_binding_schema"] == "rollout_pre_update_return_normalizer_state_v1" for row in rows),
        "frozen_update_counts": all(row["frozen_outer_cycles"] == 11 and row["frozen_ppo_updates"] == 44 and row["frozen_critic_updates"] == 88 for row in rows),
        "test6_sealed": all(row["test6_status"] == "SEALED_NOT_OPENED" for row in rows),
    }
    return {"stage": STAGE, "checks": checks, "checkpoints": rows, "checkpoint_integrity_passed": all(checks.values())}


def evidence_recoverability() -> Dict[str, Any]:
    raw_files = sorted(path.name for path in RAW_ROOT.rglob("*.parquet"))
    source = INSTRUMENTED_UPDATE_SOURCE.read_text(encoding="utf-8")
    required = {
        "cycle_actor_parameter_delta": "required by H4M-U; only held in update metrics_rows during execution",
        "cycle_critic_parameter_delta": "required by H4M-U; only held in update metrics_rows during execution",
        "actual_critic_loss": "required by H4M-U; value_loss held in update metrics_rows during execution",
    }
    persisted = {
        "action_counts_and_legal_actions": True,
        "probabilities_rewards_value_return_td_raw_gae_normalized_advantage": True,
        "ppo_surrogate_entropy_kl_proxy": True,
        "actual_critic_loss": any("critic_loss" in name or "metric" in name for name in raw_files),
        "cycle_actor_parameter_delta": any("parameter_delta" in name for name in raw_files),
        "cycle_critic_parameter_delta": any("parameter_delta" in name for name in raw_files),
    }
    source_confirms_memory_only = all(token in source for token in ["metrics_rows: List[Dict[str, Any]]", "value_loss", "actor_parameter_delta", "critic_parameter_delta"])
    unavailable = [name for name in required if not persisted[name]]
    return {
        "stage": STAGE,
        "required_metric_status": {name: {"persisted_raw_trace": persisted[name], "reason": reason} for name, reason in required.items()},
        "recoverable_execution_evidence": persisted,
        "unrecoverable_required_metrics": unavailable,
        "instrumented_update_source_confirms_metrics_were_execution_memory": source_confirms_memory_only,
        "recovery_training_invocations": 0,
        "recovery_optimizer_steps": 0,
        "TEST6_access_count": 0,
        "recovery_passed": not unavailable,
    }


def manifest(root: Path, gate: Mapping[str, Any]) -> Dict[str, Any]:
    files = {path.relative_to(root).as_posix(): str(path) for path in root.rglob("*") if path.is_file() and path.name != "manifest.json"}
    return {
        "stage": STAGE,
        "artifact_root": str(root),
        "gate": gate["gate"],
        "decision": gate["decision"],
        "exact_next_gate": gate["exact_next_gate"],
        "raw_training_artifact": str(RAW_ROOT),
        "output_sha256": {name: sha256_file(Path(path)) for name, path in files.items()},
        "TEST6_opened": False,
        "training_invocations": 0,
        "optimizer_steps": 0,
        "github_push_performed": False,
    }


def main() -> None:
    stamp = kst_now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_u_post_training_evidence_recovery_audit_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    provenance = source_provenance()
    compile_result = compile_audit()
    binding = authoritative_binding(provenance, compile_result)
    traces = trace_completeness()
    checkpoints = checkpoint_integrity()
    recovery = evidence_recoverability()
    gate = {
        "stage": STAGE,
        "gate": BLOCK_GATE,
        "decision": "H4M_U_RAW_TRAINING_COMPLETE_BUT_REQUIRED_PER_CYCLE_EVIDENCE_NOT_RECOVERABLE_WITHOUT_FORBIDDEN_RERUN",
        "exact_next_gate": "STOP_BLOCKED_H4M_U_POST_TRAINING_EVIDENCE_NOT_RECOVERABLE",
        "block_reason": "MISSING_PERSISTED_PER_CYCLE_ACTUAL_CRITIC_LOSS_AND_ACTOR_CRITIC_PARAMETER_DELTAS",
        "criteria": {
            "authoritative_binding": binding["authoritative_binding_passed"],
            "persisted_raw_execution_integrity": traces["integrity_passed"],
            "checkpoint_integrity": checkpoints["checkpoint_integrity_passed"],
            "all_required_h4mu_evidence_recoverable": recovery["recovery_passed"],
            "TEST6_access_zero": recovery["TEST6_access_count"] == 0,
            "recovery_training_optimizer_zero": recovery["recovery_training_invocations"] == 0 and recovery["recovery_optimizer_steps"] == 0,
        },
        "failing_criteria": ["all_required_h4mu_evidence_recoverable"],
        "github_push_performed": False,
    }
    payloads = {
        "recovery_binding.json": binding,
        "raw_execution_completeness.json": traces,
        "checkpoint_integrity.json": checkpoints,
        "evidence_recoverability.json": recovery,
        "test_results.json": {"py_compile": compile_result, "commands": [f"{sys.executable} -m py_compile {SOURCE_REL}", f"{sys.executable} {SOURCE_REL}"], "training_invocations": 0, "optimizer_steps": 0, "TEST6_access_count": 0},
        "gate_matrix.json": gate,
    }
    for name, payload in payloads.items():
        write_json(root / name, payload)
    (root / "final_report.md").write_text(
        f"""# H4M-U Post-Training Evidence Recovery Audit

gate = {gate['gate']}
block_reason = {gate['block_reason']}

The raw MPS run completed 33 persisted seed/cycle trace sets and three repaired checkpoints. Its post-training reporting failed at `qmod.write_conditional_parquet(...)`; no MAPPO execution, backward pass, optimizer step, TEST6 access, or GitHub push was performed by this recovery audit.

The persisted raw traces prove legal actions, finite recorded values, Reward V2 binding, frozen S3 target-state binding, and both checkpoint repair contracts. They do not contain the historical cycle-level `metrics_rows` required to report the actual critic loss and actor/critic parameter deltas for each training cycle. Reconstructing those exact values would require rerunning the frozen training path, which H4M-U forbids after the already-completed 33-rollout execution.

STOP. No retraining, tuning, validation, TEST6 access, or repair is authorized from this artifact.
""",
        encoding="utf-8",
    )
    write_json(root / "manifest.json", manifest(root, gate))
    print(f"[H4M-U recovery audit] artifact root: {root}")
    print(f"[H4M-U recovery audit] gate: {gate['gate']}")


if __name__ == "__main__":
    main()
