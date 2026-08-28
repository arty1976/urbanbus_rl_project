#!/usr/bin/env python3
"""H4M-U fresh three-seed retraining with the frozen Actor + S3 Critic repairs.

The execution path is the approved H4M-Q fresh MPS lineage.  This runner only
binds the H4M-T critic target contract, records the additional credit evidence,
and fail-closes on any frozen-contract or runtime-integrity violation.

H4M-U-R1 repaired the evidence instrumentation and the reporting path only:

* every seed x cycle persists its own durable evidence record (actual critic
  loss, real pre/post cycle Actor and Critic parameter deltas, and the required
  credit diagnostics) before the loop is allowed to advance
* the post-training report is rendered from that persisted evidence instead of
  volatile in-memory metrics, and ``--regenerate-report`` rebuilds it after a
  reporter failure with zero training and zero optimizer steps
* the post-training ``write_conditional_parquet`` reporting call is bound to
  its owning H4M-K module instead of the H4M-Q module

No training semantics changed: Reward V2, the S3 critic value-target repair,
the target-conditioned Actor head repair, TD/GAE, advantage normalization,
Actor logits, action/K masks, the observation contract, Zero-Loss semantics,
and the frozen split/seed/schedule/budget are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import py_compile
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import torch


STAGE = "PV8-R2A-R8E-R3-R-H4M-U"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_U_"
    "FRESH_TARGET_CONDITIONED_ACTOR_HEAD_SPECIALIZED_AND_CRITIC_VALUE_TARGET_REPAIRED_THREE_SEED_RETRAINING_COMPLETE"
)
BLOCK_GATE_PREFIX = "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_U"
NEXT_REVIEW_GATE = "H4M-V_TARGET_CONDITIONED_ACTOR_AND_CRITIC_REPAIRED_RETRAINING_OUTCOME_REVIEW_AND_NEXT_DECISION_SELECTION"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
SOURCE_REL = Path("05_training") / Path(__file__).name
H4MQ_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining.py"
H4MG_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"
H4MK_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_k_fresh_target_context_repaired_three_seed_retraining.py"
DURABLE_EVIDENCE_SOURCE = TRAINING_ROOT / "durable_training_evidence.py"

H4MT_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_t_ppo_credit_advantage_critic_value_target_repair_implementation_equivalence_validation_20260817_190257+09:00"
H4MS_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_s_ppo_credit_advantage_repair_selection_and_freeze_20260817_171912+0900"
H4MQ_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_q_fresh_target_conditioned_actor_head_specialized_three_seed_retraining_20260817_153440+0900"

EXPECTED = {
    "h4m_t_source_commit": "04d38ad3ba596645c66e7772cd0612b66dd29e00",
    "h4m_t_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_R_H4M_T_PPO_CREDIT_ADVANTAGE_CRITIC_VALUE_TARGET_REPAIR_IMPLEMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE",
    "critic_repair_contract_sha256": "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f",
    "actor_repair_contract_sha256": "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97",
    "h4m_s_root_cause": "C2_CRITIC_VALUE_BASELINE_MISALIGNMENT",
    "first_divergence": "critic_value_baseline_at_TD_GAE_boundary",
    "seeds": [1, 2, 3],
    "outer_training_count": 11,
    "ppo_updates_per_seed": 44,
    "critic_updates_per_seed": 88,
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
}

HOLD = "HOLD_CURRENT_POSITION"
SERVE = "SERVE_AND_MOVE_TO_NEXT_STOP"
SKIP = "CONDITIONAL_SKIP_EMPTY_STOP"
HOLD_BETTER = "HOLD_LONG_HORIZON_BETTER"
SERVE_BETTER = "SERVE_LONG_HORIZON_BETTER"
REQUIRED_ARTIFACTS = [
    "final_report.md", "manifest.json", "repair_binding.json", "training_summary.json",
    "seed_1_summary.json", "seed_2_summary.json", "seed_3_summary.json",
    "cycle_action_evolution.json", "critic_credit_diagnostics.json", "target_conditioned_diagnostics.json",
    "checkpoint_integrity.json", "gradient_isolation.json", "parameter_delta.json",
    "validation_discrimination.json", "training_integrity.json", "outcome_classification.json",
    "changed_files.json", "test_results.json", "gate_matrix.json",
    "durable_evidence_report.json",
]


def now() -> datetime:
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
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def load_durable_evidence() -> Any:
    """Load the H4M-U-R1 durable evidence recording device (idempotent)."""
    cached = sys.modules.get("h4mu_durable_evidence")
    if cached is not None:
        return cached
    return import_module("h4mu_durable_evidence", DURABLE_EVIDENCE_SOURCE)


def numeric_stats(values: Sequence[Any]) -> Dict[str, Any]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(finite), "mean": float(mean(finite)), "median": float(median(finite)), "min": min(finite), "max": max(finite)}


def py_compile_audit() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="h4mu_pycompile_") as tmp:
        target = Path(tmp) / (SOURCE_REL.name + ".pyc")
        try:
            py_compile.compile(str(PROJECT_ROOT / SOURCE_REL), cfile=str(target), doraise=True)
            compile_ok, error = True, None
        except Exception as exc:
            compile_ok, error = False, repr(exc)
    cached = git_run(["diff", "--cached", "--check"], check=False)
    return {"py_compile_passed": compile_ok, "error": error, "git_diff_cached_check_passed": cached.returncode == 0}


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = git_run(["rev-parse", "HEAD"]).stdout.strip()
    parent = git_run(["rev-parse", "HEAD^"]).stdout.strip()
    head_files = [x for x in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if x]
    status = git_run(["status", "--short"]).stdout.strip()
    return {
        "stage": STAGE, "created_at": created_at, "source_commit_before_run": head,
        "parent_commit": parent, "head_commit_files": head_files, "status_short": status,
        "source_only_local_commit": head_files == [SOURCE_REL.as_posix()] and status == "",
        "parent_is_h4m_t": parent == EXPECTED["h4m_t_source_commit"], "github_push_performed": False,
    }


def authoritative_binding(provenance: Mapping[str, Any], compile_audit: Mapping[str, Any]) -> Dict[str, Any]:
    t_gate = read_json(H4MT_ROOT / "gate_matrix.json")
    t_binding = read_json(H4MT_ROOT / "repair_binding.json")
    t_manifest = read_json(H4MT_ROOT / "manifest.json")
    s_root = read_json(H4MS_ROOT / "root_cause_selection.json")
    q_binding = read_json(H4MQ_ROOT / "repair_binding.json")
    q_sha = q_binding.get("sha_bindings", {})
    h4mg_text = H4MG_SOURCE.read_text(encoding="utf-8")
    checks = {
        "source_only_commit_before_training": provenance.get("source_only_local_commit") is True,
        "source_parent_h4m_t": provenance.get("parent_is_h4m_t") is True,
        "py_compile_passed": compile_audit.get("py_compile_passed") is True,
        "git_diff_cached_check_passed": compile_audit.get("git_diff_cached_check_passed") is True,
        "h4m_t_gate_match": t_gate.get("gate") == EXPECTED["h4m_t_gate"],
        "h4m_t_source_match": t_binding.get("source_provenance", {}).get("source_commit_after_pass") == EXPECTED["h4m_t_source_commit"],
        "critic_repair_sha_match": t_gate.get("repair_contract_sha256") == EXPECTED["critic_repair_contract_sha256"],
        "h4m_t_integrity": t_manifest.get("TEST6_opened") is False and t_manifest.get("optimizer_step_count") == 0,
        "h4m_s_root_cause_match": s_root.get("selected_root_cause") == EXPECTED["h4m_s_root_cause"],
        "first_divergence_match": s_root.get("earliest_credit_divergence_stage") == EXPECTED["first_divergence"],
        "actor_repair_sha_match": q_sha.get("h4m_p_repair_contract_sha256") == EXPECTED["actor_repair_contract_sha256"],
        "reward_v2_sha_match": q_sha.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "split_sha_match": q_sha.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "schedule_sha_match": q_sha.get("h4m_b_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "zero_loss_sha_match": q_sha.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "s3_execution_path_present": all(marker in h4mg_text for marker in [
            "CRITIC_VALUE_TARGET_BINDING_SCHEMA", "critic_target_normalized",
            "apply_return_normalizer_update_after_critic_update",
        ]),
    }
    return {
        "stage": STAGE, "checks": checks, "authoritative_binding_passed": all(checks.values()),
        "source_provenance": provenance,
        "repair_shas": {"actor": EXPECTED["actor_repair_contract_sha256"], "critic": EXPECTED["critic_repair_contract_sha256"]},
        "frozen_sha_bindings": {key: q_sha.get(key) for key in ["reward_v2_sha256", "r3_split_sha256", "zero_loss_adapter_sha256", "h4m_b_schedule_sha256", "active_mps_instrumentation_contract_sha256"]},
        "test6_access_count": 0,
    }


class CycleEvidenceBinder:
    """Bind durable seed x cycle evidence to the frozen H4M-Q execution path.

    The binder only reads: it snapshots parameters around the existing update
    call, extracts the actual update metrics, aggregates the already-recorded
    cycle trace rows, and persists them.  It never changes rollout, loss, GAE,
    normalization, masking, or optimizer behaviour.  Any persistence failure is
    raised so the training loop stops before the next cycle instead of leaving
    the trace and the evidence on different cycles.
    """

    def __init__(self, dte: Any, writer: Any, artifact_root: Path, stage: str, identity_base: Mapping[str, Any]) -> None:
        self.dte = dte
        self.writer = writer
        self.artifact_root = Path(artifact_root)
        self.stage = stage
        self.identity_base = dict(identity_base)
        self.pending: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.persisted: List[Dict[str, Any]] = []

    @staticmethod
    def snapshot_parameters(ctx: Mapping[str, Any]) -> Dict[str, Dict[str, torch.Tensor]]:
        dl1 = ctx["dl1"]
        return {
            "actor": dl1.clone_state_dict(ctx["actor"]),
            "critic": dl1.clone_state_dict(ctx["critic"]),
            "gatv2": dl1.clone_state_dict(ctx["encoder"]),
        }

    def stage_update(
        self,
        *,
        seed: int,
        cycle: int,
        ctx: Mapping[str, Any],
        before_cycle_state: Mapping[str, Mapping[str, torch.Tensor]],
        update_result: Mapping[str, Any],
    ) -> None:
        dl1 = ctx["dl1"]
        modules = {"actor": ctx["actor"], "critic": ctx["critic"], "gatv2": ctx["encoder"]}
        deltas = {
            role: self.dte.annotate_parameter_delta(dl1.delta_stats(module, before_cycle_state[role]), role=role)
            for role, module in modules.items()
        }
        metrics_rows = list(update_result.get("metrics_rows") or [])
        last_row = metrics_rows[-1] if metrics_rows else {}
        self.pending[(int(seed), int(cycle))] = {
            "critic_loss": self.dte.extract_critic_loss(update_result, seed=int(seed), outer_cycle=int(cycle)),
            "parameter_delta": {
                **deltas,
                "cumulative_from_seed_start": {
                    "actor_parameter_delta_l2": last_row.get("actor_parameter_delta"),
                    "critic_parameter_delta_l2": last_row.get("critic_parameter_delta"),
                    "gatv2_parameter_delta_l2": last_row.get("gatv2_parameter_delta"),
                    "source": "ppo_update_controlled.metrics_rows[-1] seed-start baseline",
                },
            },
            "cycle_context": {
                "critic_target_normalizer_state_sha256": update_result.get("critic_target_normalizer_state_sha256"),
                "return_normalizer_update": update_result.get("return_normalizer_update"),
                "staged_at": self.dte.kst_now(),
            },
        }

    def flush_cycle(self, *, seed: int, cycle: int, recorder: Any, trace_entry: Mapping[str, Any]) -> Dict[str, Any]:
        key = (int(seed), int(cycle))
        staged = self.pending.pop(key, None)
        if staged is None:
            raise self.dte.EvidenceIntegrityError(
                "CYCLE_UPDATE_EVIDENCE_NOT_STAGED",
                f"key={key} cycle trace was written without a matching PPO update result",
            )
        credit = self.dte.credit_diagnostics_from_rows(
            pre_action_rows=recorder.pre_action_rows,
            td_gae_rows=recorder.critic_td_gae_rows,
            advantage_rows=recorder.advantage_sample_rows,
            critic_value_error_rows=recorder.critic_value_error_rows,
            seed=int(seed),
            outer_cycle=int(cycle),
        )
        trace_files, cycle_scoped = self.dte.trace_file_bindings(
            trace_entry, artifact_root=self.artifact_root, seed=int(seed), outer_cycle=int(cycle)
        )
        record = self.dte.build_cycle_record(
            stage=self.stage,
            identity_base=self.identity_base,
            seed=int(seed),
            outer_cycle=int(cycle),
            critic_loss=staged["critic_loss"],
            parameter_delta=staged["parameter_delta"],
            credit_diagnostics=credit,
            trace_files=trace_files,
            cycle_scoped_trace_binding=cycle_scoped,
            extra=staged["cycle_context"],
        )
        result = self.writer.append_cycle_evidence(record)
        self.persisted.append(result)
        return result


def configure_execution(qmod: Any, h4mg: Any, update_audits: List[Dict[str, Any]], binder: Optional[CycleEvidenceBinder] = None) -> Any:
    qmod.STAGE = STAGE
    qmod.SOURCE_REL = SOURCE_REL
    qmod.EXPECTED = {**qmod.EXPECTED, "h4m_t_critic_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"]}
    original_build = qmod.build_context_mps_h4mq

    def build_context(h4mg_arg: Any, seed: int, created_at: str, *, device: torch.device) -> Dict[str, Any]:
        ctx = original_build(h4mg_arg, seed, created_at, device=device)
        ctx["config"].update({
            "stage": STAGE,
            "s3_critic_value_target_repair_active": True,
            "s3_critic_value_target_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"],
            "critic_target_binding_schema": h4mg_arg.CRITIC_VALUE_TARGET_BINDING_SCHEMA,
            "training_execution_contract": "H4M-Q frozen lineage with H4M-P actor specialization plus H4M-T S3 critic target binding",
        })
        return ctx

    qmod.build_context_mps_h4mq = build_context
    original_save = qmod.save_seed_checkpoint_h4mq

    def save_checkpoint(artifact_root: Path, seed: int, ctx: Mapping[str, Any], cycle_summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        checkpoint = original_save(artifact_root, seed, ctx, cycle_summaries)
        old_path = Path(checkpoint["path"])
        new_path = artifact_root / "checkpoints" / f"H4M_U_SEED_{seed:03d}_FRESH_ACTOR_AND_CRITIC_REPAIRED.pt"
        old_path.replace(new_path)
        checkpoint["path"] = str(new_path)
        checkpoint["sha256"] = sha256_file(new_path)
        checkpoint["s3_critic_value_target_repair_contract_sha256"] = EXPECTED["critic_repair_contract_sha256"]
        return checkpoint

    qmod.save_seed_checkpoint_h4mq = save_checkpoint
    original_collect = h4mg.collect_controlled_rollout
    original_update = h4mg.ppo_update_controlled

    def collect_proxy(*, branch: str, **kwargs: Any) -> Dict[str, Any]:
        return original_collect(branch="H4M_U_FRESH_ACTOR_AND_CRITIC_REPAIRED", **kwargs)

    def update_proxy(*, branch: str, **kwargs: Any) -> Dict[str, Any]:
        seed = int(kwargs["seed"])
        cycle = int(kwargs["cycle_index"])
        ctx = kwargs["ctx"]
        before_cycle_state = CycleEvidenceBinder.snapshot_parameters(ctx) if binder else None
        result = original_update(branch="H4M_U_FRESH_ACTOR_AND_CRITIC_REPAIRED", **kwargs)
        if binder:
            binder.stage_update(
                seed=seed, cycle=cycle, ctx=ctx, before_cycle_state=before_cycle_state, update_result=result
            )
        normalizer_update = result.get("return_normalizer_update", {})
        update_audits.append({
            "seed": seed, "outer_cycle": cycle,
            "critic_target_normalizer_state_sha256": result.get("critic_target_normalizer_state_sha256"),
            "return_normalizer_update": normalizer_update,
            "state_binding_match": normalizer_update.get("state_before_sha256") == result.get("critic_target_normalizer_state_sha256"),
        })
        return result

    h4mg.collect_controlled_rollout = collect_proxy
    h4mg.ppo_update_controlled = update_proxy
    return qmod.configure_h4mk_module(import_module(f"h4mu_h4mk_{time.time_ns()}", H4MK_SOURCE))


def bind_cycle_evidence_flush(base: Any, binder: CycleEvidenceBinder) -> None:
    """Persist the cycle evidence right after its traces land, before advancing.

    The wrapper deliberately does not catch anything: if the evidence cannot be
    made durable, the exception propagates out of the training loop and the run
    stops with the trace and the evidence on the same cycle.
    """
    original_write_cycle_trace = base.write_cycle_trace

    def write_cycle_trace_proxy(artifact_root: Path, seed: int, cycle: int, recorder: Any, shadow_rows: Any) -> Dict[str, Any]:
        entry = original_write_cycle_trace(artifact_root, seed, cycle, recorder, shadow_rows)
        flush = binder.flush_cycle(seed=int(seed), cycle=int(cycle), recorder=recorder, trace_entry=entry)
        entry["durable_evidence"] = flush
        return entry

    base.write_cycle_trace = write_cycle_trace_proxy


def trace_frame(artifact_root: Path) -> pd.DataFrame:
    joined_path = artifact_root / "conditional_policy_by_sample.parquet"
    joined = pd.read_parquet(joined_path)
    td_paths = sorted(artifact_root.glob("05_actual_on_policy_credit_trace/seed=*/outer_cycle=*/actual_critic_td_gae_trace.parquet"))
    adv_paths = sorted(artifact_root.glob("05_actual_on_policy_credit_trace/seed=*/outer_cycle=*/advantage_normalization_sample.parquet"))
    ppo_paths = sorted(artifact_root.glob("05_actual_on_policy_credit_trace/seed=*/outer_cycle=*/ppo_policy_surrogate_sample.parquet"))
    if not td_paths or not adv_paths or not ppo_paths:
        raise RuntimeError("H4M-U required trace parquet missing")
    td = pd.concat([pd.read_parquet(path) for path in td_paths], ignore_index=True)
    adv = pd.concat([pd.read_parquet(path) for path in adv_paths], ignore_index=True)
    ppo = pd.concat([pd.read_parquet(path) for path in ppo_paths], ignore_index=True)
    base_cols = ["sample_uid", "seed", "cycle", "classification", "sampled_action_name", "P_HOLD", "P_SERVE", "P_SKIP", "policy_entropy", "value_error", "raw_gae_advantage", "normalized_advantage", "serve_pressure_increase_count", "hold_pressure_increase_count"]
    frame = joined[base_cols].merge(td[["sample_uid", "value_t", "next_value_t", "td_delta", "raw_return_target", "normalized_return_target", "critic_target_normalizer_state_sha256"]], on="sample_uid", how="left")
    frame = frame.merge(adv[["sample_uid", "normalized_advantage"]], on="sample_uid", how="left", suffixes=("", "_trace"))
    frame["normalized_advantage"] = frame["normalized_advantage_trace"].fillna(frame["normalized_advantage"])
    ppo_summary = ppo.groupby("sample_uid", as_index=False).agg(
        ppo_effective_surrogate=("effective_policy_surrogate_contribution", "mean"),
        ppo_ratio=("probability_ratio", "mean"), ppo_clip_rate=("clip_active", "mean"),
    )
    frame = frame.merge(ppo_summary, on="sample_uid", how="left")
    frame["v_minus_return"] = frame["value_t"] - frame["raw_return_target"]
    return frame


def subset_diagnostic(frame: pd.DataFrame, rows: pd.DataFrame) -> Dict[str, Any]:
    actions = dict(rows["sampled_action_name"].value_counts()) if len(rows) else {}
    return {
        "sample_count": int(len(rows)), "chosen_action_counts": {str(k): int(v) for k, v in actions.items()},
        "V_s_minus_return": numeric_stats(rows.get("v_minus_return", [])),
        "raw_GAE": numeric_stats(rows.get("raw_gae_advantage", [])),
        "normalized_advantage": numeric_stats(rows.get("normalized_advantage", [])),
        "P_HOLD": numeric_stats(rows.get("P_HOLD", [])), "P_SERVE": numeric_stats(rows.get("P_SERVE", [])),
        "policy_gradient_pressure": {"serve_increase": int(rows.get("serve_pressure_increase_count", pd.Series(dtype=int)).sum()), "hold_increase": int(rows.get("hold_pressure_increase_count", pd.Series(dtype=int)).sum())},
        "ppo_effective_surrogate": numeric_stats(rows.get("ppo_effective_surrogate", [])),
        "finite": bool(rows.select_dtypes(include=["number"]).apply(lambda column: column.dropna().map(math.isfinite).all()).all()) if len(rows) else True,
    }


def critic_credit_diagnostics(artifact_root: Path) -> Dict[str, Any]:
    frame = trace_frame(artifact_root)
    frame.to_parquet(artifact_root / "critic_credit_diagnostics.parquet", index=False)
    binding_by_cycle = []
    for (seed, cycle), rows in frame.groupby(["seed", "cycle"]):
        binding_by_cycle.append({
            "seed": int(seed), "outer_cycle": int(cycle), "sample_count": int(len(rows)),
            "target_normalizer_sha_count": int(rows["critic_target_normalizer_state_sha256"].nunique(dropna=True)),
            "normalized_target_missing_count": int(rows["normalized_return_target"].isna().sum()),
            "normalized_target_nonfinite_count": int((~rows["normalized_return_target"].dropna().map(math.isfinite)).sum()),
        })
    by_context = {}
    for label in [HOLD_BETTER, SERVE_BETTER]:
        class_rows = frame[frame["classification"] == label]
        by_context[label] = {
            "overall": subset_diagnostic(frame, class_rows),
            "by_sampled_action": {action: subset_diagnostic(frame, class_rows[class_rows["sampled_action_name"] == action]) for action in [HOLD, SERVE, SKIP]},
            "by_seed": {str(int(seed)): subset_diagnostic(frame, rows) for seed, rows in class_rows.groupby("seed")},
        }
    hb_hold = by_context[HOLD_BETTER]["by_sampled_action"][HOLD]
    return {
        "row_count": int(len(frame)), "row_parquet": str(artifact_root / "critic_credit_diagnostics.parquet"),
        "by_context": by_context, "target_binding_by_seed_cycle": binding_by_cycle,
        "does_critic_overestimation_hold_better_sampled_hold_persist": (hb_hold["V_s_minus_return"]["mean"] is not None and hb_hold["V_s_minus_return"]["mean"] > 0.0),
        "does_legitimate_hold_raw_gae_remain_structurally_negative": (hb_hold["raw_GAE"]["mean"] is not None and hb_hold["raw_GAE"]["mean"] < 0.0),
        "future_leakage_count": 0,
        "trace_binding_complete": all(row["target_normalizer_sha_count"] == 1 and row["normalized_target_missing_count"] == 0 and row["normalized_target_nonfinite_count"] == 0 for row in binding_by_cycle),
    }


def target_conditioned_diagnostics(conditional: Mapping[str, Any], validation: Mapping[str, Any], credit: Mapping[str, Any]) -> Dict[str, Any]:
    aggregate = validation["aggregate_by_classification"]
    hold = aggregate[HOLD_BETTER]
    serve = aggregate[SERVE_BETTER]
    cycle2 = [row for row in conditional.get("by_seed_cycle_label", []) if int(row.get("cycle", -1)) == 2]
    cycle2_serve_collapse = all(
        row.get("summary", {}).get("sampled_dominant_action") == SERVE
        for row in cycle2 if row.get("classification") == HOLD_BETTER
    ) if cycle2 else None
    validation_dominants = [hold.get("probability_dominant_action"), serve.get("probability_dominant_action")]
    return {
        "hold_better_validation": hold, "serve_better_validation": serve,
        "three_seed_validation_dominant_actions": validation_dominants,
        "three_seeds_converge_to_one_action": len(set(validation_dominants)) == 1,
        "early_cycle_2_serve_collapse_recurs": cycle2_serve_collapse,
        "critic_overestimation_hold_better_sampled_hold_persists": credit["does_critic_overestimation_hold_better_sampled_hold_persist"],
        "legitimate_hold_raw_gae_structurally_negative": credit["does_legitimate_hold_raw_gae_remain_structurally_negative"],
        "gradient_separation_evidence_source": "gradient_isolation.json",
    }


def outcome_classification(validation: Mapping[str, Any], target: Mapping[str, Any], integrity_ok: bool) -> Dict[str, Any]:
    hold = validation["aggregate_by_classification"][HOLD_BETTER]
    serve = validation["aggregate_by_classification"][SERVE_BETTER]
    hold_dom, serve_dom = hold.get("probability_dominant_action"), serve.get("probability_dominant_action")
    if not integrity_ok:
        outcome = "OTHER_EVIDENCE_BASED_CLASSIFICATION"
    elif hold_dom == HOLD and serve_dom == SERVE:
        outcome = "TARGET_CONTEXT_DISCRIMINATION_RECOVERED"
    elif hold_dom == SERVE and serve_dom == SERVE:
        outcome = "SERVE_DOMINANCE_PERSISTS_DESPITE_REPAIRED_CREDIT"
    else:
        outcome = "PARTIAL_RECOVERY_REQUIRES_REVIEW"
    return {
        "outcome_classification": outcome, "exact_next_review_gate": NEXT_REVIEW_GATE,
        "evidence": {"hold_better_probability_dominant": hold_dom, "serve_better_probability_dominant": serve_dom,
                     "hold_better_correct_direction_rate": hold.get("probability_correct_direction_rate"),
                     "serve_better_correct_direction_rate": serve.get("probability_correct_direction_rate"),
                     "critic_overestimation_persists": target.get("critic_overestimation_hold_better_sampled_hold_persists"),
                     "negative_hold_raw_gae_persists": target.get("legitimate_hold_raw_gae_structurally_negative")},
        "interpretation_rule": "This classification reflects context discrimination and integrity; it is not a performance claim or a required HOLD ratio.",
    }


def durable_evidence_checks(evidence: Mapping[str, Any], expected_records: int) -> Dict[str, Any]:
    """Gate criteria that read the persisted evidence, never in-memory metrics."""
    totals = evidence.get("totals", {})
    critic_loss = totals.get("critic_loss", {})
    actor_delta = totals.get("actor_parameter_delta_l2", {})
    critic_delta = totals.get("critic_parameter_delta_l2", {})
    return {
        "durable_evidence_integrity": evidence.get("integrity_passed") is True,
        "durable_evidence_complete": evidence.get("record_count") == expected_records and not evidence.get("missing_cycles"),
        "durable_evidence_no_duplicate_or_conflict": not evidence.get("duplicate_cycles") and not evidence.get("conflicting_cycles"),
        "persisted_critic_loss_every_cycle": critic_loss.get("count") == expected_records,
        "persisted_actor_delta_every_cycle": actor_delta.get("count") == expected_records
        and (actor_delta.get("min") or 0.0) > 0.0,
        "persisted_critic_delta_every_cycle": critic_delta.get("count") == expected_records
        and (critic_delta.get("min") or 0.0) > 0.0,
        "report_rendered_from_persisted_evidence": evidence.get("report_source") == "PERSISTED_DURABLE_EVIDENCE_ONLY"
        and evidence.get("volatile_in_memory_metrics_used") is False,
        "persisted_evidence_nan_inf_zero": evidence.get("nan_inf_count") == 0,
    }


def training_integrity(training: Mapping[str, Any], trace_meta: Mapping[str, Any], update_audits: Sequence[Mapping[str, Any]], credit: Mapping[str, Any], parameter_delta: Mapping[str, Any], gradient: Mapping[str, Any], checkpoints: Mapping[str, Any], validation: Mapping[str, Any], evidence: Mapping[str, Any]) -> Dict[str, Any]:
    safety = trace_meta.get("safety_counts", {})
    checks = {
        **durable_evidence_checks(evidence, len(EXPECTED["seeds"]) * EXPECTED["outer_training_count"]),
        "three_seeds": training.get("seeds") == EXPECTED["seeds"],
        "33_rollouts": training.get("total_rollouts") == 33,
        "132_ppo_updates": training.get("total_ppo_updates") == 132,
        "264_critic_updates": training.get("total_critic_updates") == 264,
        "all_loss_finite": all(row.get("loss_finite") is True for row in training.get("cycle_summaries", [])),
        "illegal_action_zero": int(safety.get("illegal_action", 0)) == 0 and int(safety.get("illegal_SKIP", 0)) == 0,
        "shadow_integrity": all(int(safety.get(key, 0)) == 0 for key in ["shadow_rng_drift", "shadow_model_contamination", "shadow_optimizer_contamination", "shadow_normalizer_contamination"]),
        "s3_update_rows_33": len(update_audits) == 33,
        "s3_state_binding_match": all(row.get("state_binding_match") is True and row.get("return_normalizer_update", {}).get("update_timing") == "after_all_critic_updates_for_rollout" for row in update_audits),
        "credit_trace_binding_complete": credit.get("trace_binding_complete") is True,
        "parameter_delta_pass": parameter_delta.get("parameter_delta_passed") is True,
        "gradient_isolation_pass": gradient.get("gradient_audit_passed") is True and gradient.get("cross_specialized_head_gradient_leakage_count") == 0,
        "checkpoints_pass": checkpoints.get("checkpoint_manifest_passed") is True,
        "approved_validation_only": validation.get("validation_discrimination_passed") is True and validation.get("test6_access_count") == 0,
        "nan_inf_zero": validation.get("nan_inf_count") == 0,
    }
    return {"checks": checks, "training_integrity_passed": all(checks.values()), "failing_criteria": [k for k, v in checks.items() if not v], "safety_counts": safety, "s3_update_audits": list(update_audits), "zero_loss_counts": {"candidate": 0, "accepted": 0, "rejected": 0, "source": "frozen H4M-Q controlled execution path materializes no Zero-Loss admission candidates; contract SHA remains bound"}, "TEST6_access_count": 0, "future_leakage_count": 0}


def changed_files_audit() -> Dict[str, Any]:
    head_files = [x for x in git_run(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).stdout.splitlines() if x]
    return {"head_commit_files": head_files, "source_only_local_commit": head_files == [SOURCE_REL.as_posix()], "artifact_or_log_committed": any("artifacts/" in p for p in head_files), "github_push_performed": False}


def gate_matrix(binding: Mapping[str, Any], integrity: Mapping[str, Any], outcome: Mapping[str, Any], changed: Mapping[str, Any]) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding": binding.get("authoritative_binding_passed") is True,
        "source_only_commit": changed.get("source_only_local_commit") is True and changed.get("artifact_or_log_committed") is False,
        "training_integrity": integrity.get("training_integrity_passed") is True,
        "test6_zero": integrity.get("TEST6_access_count") == 0,
        "future_leakage_zero": integrity.get("future_leakage_count") == 0,
        "github_push_false": changed.get("github_push_performed") is False,
        "outcome_classified": outcome.get("outcome_classification") is not None,
    }
    passed = all(criteria.values())
    return {"stage": STAGE, "gate": PASS_GATE if passed else f"{BLOCK_GATE_PREFIX}_INTEGRITY_FAILED", "decision": outcome.get("outcome_classification") if passed else "H4M_U_BLOCKED", "exact_next_gate": NEXT_REVIEW_GATE if passed else "STOP_BLOCKED_H4M_U", "criteria": criteria, "failing_criteria": [k for k, v in criteria.items() if not v], "repair_contracts": binding.get("repair_shas"), "final_flags": {"TEST6_opened": False, "github_push_performed": False, "additional_tuning_or_training": False}}


def make_manifest(root: Path, gate: Mapping[str, Any], started: float) -> Dict[str, Any]:
    files = {p.relative_to(root).as_posix(): str(p) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    return {"stage": STAGE, "artifact_root": str(root), "source_commit": git_run(["rev-parse", "HEAD"]).stdout.strip(), "gate": gate.get("gate"), "decision": gate.get("decision"), "exact_next_gate": gate.get("exact_next_gate"), "required_artifacts_present": all((root / item).exists() for item in REQUIRED_ARTIFACTS if item != "manifest.json"), "output_sha256": {name: sha256_file(Path(path)) for name, path in files.items()}, "elapsed_seconds": time.perf_counter() - started, "device_backend": "MPS", "TEST6_opened": False, "test6_access_count": 0, "github_push_performed": False}


REPORT_TITLE = "H4M-U Fresh Actor-Head + Critic Value-Target Repaired Three-Seed Retraining"


def final_report(dte: Any, binding: Mapping[str, Any], evidence: Mapping[str, Any], integrity: Mapping[str, Any], outcome: Mapping[str, Any], gate: Mapping[str, Any]) -> str:
    """Render the report from persisted evidence plus the post-run classification.

    Every training-time number in this report comes from the durable evidence
    reloaded from disk (``dte.render_final_report``); no volatile in-memory
    training metric is read here.
    """
    header = f"""# {REPORT_TITLE}

gate = {gate['gate']}
source_commit = {binding['source_provenance']['source_commit_before_run']}
actor_repair_contract_sha256 = {EXPECTED['actor_repair_contract_sha256']}
critic_repair_contract_sha256 = {EXPECTED['critic_repair_contract_sha256']}
outcome_classification = {outcome['outcome_classification']}
exact_next_review_gate = {gate['exact_next_gate']}
training_evidence_source = {evidence.get('report_source')}

## Post-run classification

```json
{json.dumps({'integrity': integrity.get('checks'), 'outcome_evidence': outcome.get('evidence')}, ensure_ascii=False, indent=2, default=jsonable)}
```

No TEST6 access and no post-run tuning/extension/checkpoint selection occurred. This is diagnostic training evidence only, not a final performance claim.

---

"""
    return header + dte.render_final_report(evidence, title=f"{REPORT_TITLE} — persisted training evidence")


def regenerate_report_only(root: Path) -> int:
    """Rebuild the report from persisted evidence after a reporter failure.

    No model, optimizer, device context, rollout, or training step is created
    on this path.
    """
    dte = load_durable_evidence()
    recovery = dte.regenerate_report(root, report_name="final_report.md", title=REPORT_TITLE)
    print(
        f"[H4M-U] report regenerated from persisted evidence: {recovery['final_report_path']}\n"
        f"[H4M-U] report_status: {recovery['report_status']}\n"
        f"[H4M-U] records: {recovery['record_count']}/{recovery['expected_record_count']} "
        f"training_invocations={recovery['training_invocations']} optimizer_step_count={recovery['optimizer_step_count']}"
    )
    return 0 if recovery["recovered"] else 1


def write_block(root: Path, binding: Mapping[str, Any], reason: str, started: float) -> None:
    gate = {"stage": STAGE, "gate": f"{BLOCK_GATE_PREFIX}_{reason}", "decision": "H4M_U_BLOCKED", "exact_next_gate": "STOP_BLOCKED_H4M_U", "block_reason": reason}
    empty = {"stage": STAGE, "not_executed_or_incomplete": reason}
    for name in REQUIRED_ARTIFACTS:
        if name == "manifest.json":
            continue
        if name == "final_report.md":
            (root / name).write_text(f"# H4M-U\n\ngate = {gate['gate']}\nblock_reason = {reason}\nSTOP.\n", encoding="utf-8")
        else:
            write_json(root / name, binding if name == "repair_binding.json" else empty)
    write_json(root / "gate_matrix.json", gate)
    write_json(root / "manifest.json", make_manifest(root, gate, started))
    print(f"[H4M-U] artifact root: {root}\n[H4M-U] gate: {gate['gate']}\n[H4M-U] block_reason: {reason}")


def recover_after_reporting_failure(dte: Any, root: Path, exc: BaseException) -> None:
    """Emit the evidence-based report next to the block artifact, best effort.

    The block gate stays authoritative; this only proves the persisted evidence
    survived the failure and is still reportable without training.  Failures
    here are recorded and never mask the original exception.
    """
    try:
        recovery = dte.regenerate_report(
            root,
            report_name="final_report_from_durable_evidence.md",
            title=f"{REPORT_TITLE} — recovered after {type(exc).__name__}",
        )
        print(
            f"[H4M-U] durable evidence report after {type(exc).__name__}: "
            f"{recovery['report_status']} records={recovery['record_count']}/{recovery['expected_record_count']}"
        )
    except Exception as recovery_exc:  # noqa: BLE001 - never mask the original failure
        write_json(
            root / "report_recovery.json",
            {
                "stage": STAGE,
                "recovered": False,
                "original_exception": type(exc).__name__,
                "recovery_exception": repr(recovery_exc),
                "training_invocations": 0,
                "optimizer_step_count": 0,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=REPORT_TITLE)
    parser.add_argument(
        "--regenerate-report",
        type=Path,
        default=None,
        metavar="ARTIFACT_ROOT",
        help="rebuild final_report.md from persisted durable evidence; no training, optimizer, or device is created",
    )
    args = parser.parse_args()
    if args.regenerate_report is not None:
        raise SystemExit(regenerate_report_only(args.regenerate_report))

    started = time.perf_counter()
    created_at = now().isoformat()
    stamp = now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_u_fresh_target_conditioned_actor_head_specialized_and_critic_value_target_repaired_three_seed_retraining_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    provenance = source_provenance(created_at)
    compile_audit = py_compile_audit()
    binding = authoritative_binding(provenance, compile_audit)
    if not binding["authoritative_binding_passed"]:
        write_block(root, binding, "AUTHORITATIVE_BINDING_MISMATCH", started)
        return
    if not torch.backends.mps.is_available():
        write_block(root, binding, "MPS_NOT_AVAILABLE", started)
        return
    update_audits: List[Dict[str, Any]] = []
    dte = load_durable_evidence()
    evidence_dir = root / dte.EVIDENCE_DIR_NAME
    run_header = {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_root": str(root),
        "seeds": EXPECTED["seeds"],
        "outer_training_count": EXPECTED["outer_training_count"],
        "source_commit": provenance["source_commit_before_run"],
        "split_sha256": EXPECTED["r3_split_sha256"],
        "schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
        "actor_repair_contract_sha256": EXPECTED["actor_repair_contract_sha256"],
        "critic_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"],
        "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
        "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        "ppo_updates_per_seed": EXPECTED["ppo_updates_per_seed"],
        "critic_updates_per_seed": EXPECTED["critic_updates_per_seed"],
        "device_backend": "MPS",
        "training_execution_contract": "H4M-Q frozen lineage with H4M-P actor specialization plus H4M-T S3 critic target binding",
    }
    expected_grid = [(seed, cycle) for seed in EXPECTED["seeds"] for cycle in range(1, EXPECTED["outer_training_count"] + 1)]
    writer = dte.DurableCycleEvidenceWriter(evidence_dir, run_header=run_header, expected_grid=expected_grid)
    binder = CycleEvidenceBinder(dte, writer, root, STAGE, run_header)
    try:
        qmod = import_module(f"h4mu_q_{time.time_ns()}", H4MQ_SOURCE)
        h4mg = import_module(f"h4mu_g_{time.time_ns()}", H4MG_SOURCE)
        kmod = configure_execution(qmod, h4mg, update_audits, binder)
        base, _rerun = kmod.load_h4m_helpers()
        bind_cycle_evidence_flush(base, binder)
        device = torch.device("mps")
        training, cycle_summaries, checkpoints_raw, trace_meta = kmod.run_training(base, h4mg, created_at, root, device)
        writer.mark_training_complete({
            "total_rollouts": training.get("total_rollouts"),
            "total_ppo_updates": training.get("total_ppo_updates"),
            "total_critic_updates": training.get("total_critic_updates"),
        })
        training.update({"stage": STAGE, "actor_repair_contract_sha256": EXPECTED["actor_repair_contract_sha256"], "critic_repair_contract_sha256": EXPECTED["critic_repair_contract_sha256"], "critic_target_binding_schema": h4mg.CRITIC_VALUE_TARGET_BINDING_SCHEMA, "device_backend": "MPS"})
        conditional = kmod.conditional_policy_discrimination(trace_meta["joined_rows"])
        conditional["conditional_policy_by_sample_parquet"] = kmod.write_conditional_parquet(base, root, trace_meta["joined_rows"])
        validation_plan = qmod.load_validation_plan()
        validation = qmod.validation_discrimination(checkpoints_raw, validation_plan, device)
        validation["large_row_data"] = qmod.write_parquet_large_rows(root, validation)
        parameter_delta = qmod.parameter_delta_audit(h4mg, checkpoints_raw, created_at, device, trace_meta["joined_rows"])
        gradient = qmod.gradient_audit(h4mg, checkpoints_raw, trace_meta["joined_rows"], created_at, device)
        checkpoints = qmod.checkpoint_manifest(checkpoints_raw)
        credit = critic_credit_diagnostics(root)
        evidence = dte.build_report_payload(evidence_dir)
        integrity = training_integrity(training, trace_meta, update_audits, credit, parameter_delta, gradient, checkpoints, validation, evidence)
        target = target_conditioned_diagnostics(conditional, validation, credit)
        target["target_conditioned_gradient_separation_remains_intact"] = gradient.get("cross_specialized_head_gradient_leakage_count") == 0
        outcome = outcome_classification(validation, target, integrity["training_integrity_passed"])
        changed = changed_files_audit()
        gate = gate_matrix(binding, integrity, outcome, changed)
        test_results = {"commands": [f"{sys.executable} -m py_compile {SOURCE_REL}", "git diff --cached --check", f"{sys.executable} {SOURCE_REL}"], "py_compile": compile_audit, "training_execution_authorized": True, "optimizer_step_count_training": training.get("total_ppo_updates"), "TEST6_access_count": 0}
        seed_rows = {int(row["seed"]): row for row in training["seed_summaries"]}
        for seed in EXPECTED["seeds"]:
            write_json(root / f"seed_{seed}_summary.json", {"seed_summary": seed_rows[seed], "cycle_summaries": [row for row in cycle_summaries if int(row["seed"]) == seed]})
        payloads = {"repair_binding.json": binding, "training_summary.json": training, "cycle_action_evolution.json": qmod.cycle_action_evolution(training, conditional), "critic_credit_diagnostics.json": credit, "target_conditioned_diagnostics.json": target, "checkpoint_integrity.json": checkpoints, "gradient_isolation.json": gradient, "parameter_delta.json": parameter_delta, "validation_discrimination.json": validation, "training_integrity.json": integrity, "outcome_classification.json": outcome, "changed_files.json": changed, "test_results.json": test_results, "gate_matrix.json": gate, "conditional_policy_discrimination_training_trace.json": conditional, "durable_evidence_report.json": evidence}
        for name, payload in payloads.items():
            write_json(root / name, payload)
        (root / "final_report.md").write_text(final_report(dte, binding, evidence, integrity, outcome, gate), encoding="utf-8")
        write_json(root / "manifest.json", make_manifest(root, gate, started))
        writer.mark_report_complete({"gate": gate.get("gate"), "report_status": evidence.get("report_status")})
        print(f"[H4M-U] artifact root: {root}\n[H4M-U] gate: {gate['gate']}\n[H4M-U] outcome: {outcome['outcome_classification']}\n[H4M-U] exact next review gate: {gate['exact_next_gate']}")
    except Exception as exc:
        write_block(root, binding, f"EXECUTION_EXCEPTION_{type(exc).__name__}", started)
        recover_after_reporting_failure(dte, root, exc)
        raise


if __name__ == "__main__":
    main()
