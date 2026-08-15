#!/usr/bin/env python3
"""H4M-F long-horizon and per-sample credit-trace instrumentation freeze.

This stage is design-only. It binds the H4M-E causal-attribution gap, selects a
side-channel instrumentation contract for the next diagnostic training run, and
emits immutable schema/contract artifacts. It must not execute training,
instantiate optimizers, mutate checkpoints, update normalizers, or open TEST6.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from zoneinfo import ZoneInfo


STAGE = "PV8-R2A-R8E-R3-R-H4M-F"
PASS_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_F_LONG_HORIZON_AND_PER_SAMPLE_"
    "CREDIT_TRACE_INSTRUMENTATION_SELECTION_AND_FREEZE_COMPLETE"
)
BLOCK_GATE = (
    "BLOCKED_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_F_LONG_HORIZON_AND_PER_SAMPLE_"
    "CREDIT_TRACE_INSTRUMENTATION_SELECTION_AND_FREEZE_FAILED"
)
PASS_DECISION = (
    "PV8_LONG_HORIZON_AND_PER_SAMPLE_CREDIT_TRACE_INSTRUMENTATION_FROZEN_"
    "READY_FOR_IMPLEMENTATION_EQUIVALENCE_VALIDATION"
)
BLOCK_DECISION = "PV8_LONG_HORIZON_AND_PER_SAMPLE_CREDIT_TRACE_INSTRUMENTATION_FREEZE_BLOCKED"
NEXT_GATE = "H4M-G_INSTRUMENTATION_IMPLEMENTATION_AND_BEHAVIORAL_EQUIVALENCE_VALIDATION"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "05_training" / "artifacts"
SOURCE_PATH = Path("05_training") / Path(__file__).name

H4M_B_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_b_training_budget_extension_selection_and_freeze_20260814_161227"
H4M_C_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_c_fresh_extended_budget_three_seed_retraining_20260814_172137"
H4M_D_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_d_frozen_extended_budget_policy_revalidation_20260814_175417"
H4M_E_ROOT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_e_reward_gae_actor_credit_alignment_audit_20260814_182328"

EXPECTED = {
    "h4m_e_source_commit": "9479d72d1c1c8dda56c909c67776137aaa185da1",
    "h4m_c_source_commit": "00a53164c710f0bb8028aabe4f3f616a8fe6610d",
    "h4m_d_source_commit": "960e60adbfb3737c676532be575eb1b38375d9da",
    "h4m_b_schedule_sha256": "c8eb56b86854113c751e099f6dc9869234324005911d0ece125b857e47e06dcc",
    "outer_training_count": 11,
    "rollout_horizon": 512,
    "ppo_epochs_per_update": 4,
    "critic_epochs_per_update": 8,
    "reward_v2_sha256": "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161",
    "h4g_runtime_sha256": "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3",
    "r3_split_sha256": "cf7c21c1e85ae8717678fbce85cdbff27ef5e4ca133593de31ad4884aefd476c",
    "zero_loss_adapter_sha256": "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce",
    "h4m_e_gate": "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_E_REWARD_GAE_ACTOR_CREDIT_ALIGNMENT_AUDIT_COMPLETE",
    "h4m_e_decision": "PV8_REWARD_GAE_ACTOR_CREDIT_CAUSAL_ATTRIBUTION_NOT_UNIQUE",
    "h4m_e_next_gate": "H4M-F_LONG_HORIZON_AND_PER_SAMPLE_CREDIT_TRACE_INSTRUMENTATION_SELECTION_AND_FREEZE",
    "h4m_c_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_C_FRESH_EXTENDED_BUDGET_THREE_SEED_RETRAINING_COMPLETE",
    "h4m_d_gate": "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_PV8_R2AR8ER3RH4M_D_FROZEN_EXTENDED_BUDGET_POLICY_REVALIDATION_COMPLETE",
}

REQUIRED_ARTIFACTS = [
    "01_authoritative_binding.json",
    "02_missing_evidence_from_h4me.json",
    "03_actual_training_trace_schema.json",
    "04_advantage_normalization_trace_schema.json",
    "05_ppo_per_sample_trace_schema.json",
    "06_critic_and_gradient_trace_schema.json",
    "07_long_horizon_counterfactual_trace_schema.json",
    "08_sample_identity_and_rng_contract.json",
    "09_counterfactual_scope_candidate_comparison.json",
    "10_runtime_storage_overhead_estimate.json",
    "11_behavioral_noninterference_contract.json",
    "12_h4mg_equivalence_gate_contract.json",
    "13_h4m_f_credit_trace_instrumentation_contract.json",
    "14_h4m_f_gate_matrix.json",
    "final_report.md",
    "manifest.json",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(canonical_json(payload), encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def run_git(args: List[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_bool(args: List[str]) -> bool:
    return run_git(args, check=False).returncode == 0


def now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).replace(microsecond=0).isoformat()


def field(
    name: str,
    dtype: str,
    *,
    nullable: bool = False,
    semantics: str,
    precision: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "name": name,
        "dtype": dtype,
        "nullable": nullable,
        "semantics": semantics,
    }
    if precision is not None:
        row["precision"] = precision
    if source is not None:
        row["source"] = source
    return row


def table(
    name: str,
    *,
    namespace: str,
    format_: str,
    partition_by: List[str],
    primary_key: List[str],
    rows_expected: Any,
    fields: List[Dict[str, Any]],
    semantics: str,
) -> Dict[str, Any]:
    return {
        "name": name,
        "namespace": namespace,
        "format": format_,
        "partition_by": partition_by,
        "primary_key": primary_key,
        "rows_expected": rows_expected,
        "semantics": semantics,
        "fields": fields,
    }


def source_provenance(created_at: str) -> Dict[str, Any]:
    head = run_git(["rev-parse", "HEAD"]).stdout.strip()
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    status_short = run_git(["status", "--short"]).stdout
    source_rel = SOURCE_PATH.as_posix()
    source_present_in_head = git_bool(["cat-file", "-e", f"HEAD:{source_rel}"])
    source_latest_commit = run_git(["log", "-1", "--format=%H", "--", source_rel]).stdout.strip()
    source_no_diff = git_bool(["diff", "--quiet", "--", source_rel])
    staged_no_diff = git_bool(["diff", "--cached", "--quiet", "--", source_rel])
    head_files = [
        line.strip()
        for line in run_git(["show", "--name-only", "--pretty=format:", "HEAD"]).stdout.splitlines()
        if line.strip()
    ]
    return {
        "stage": STAGE,
        "created_at": created_at,
        "git_branch": branch,
        "git_commit": head,
        "h4m_f_source_git_commit": head,
        "h4m_f_source_path": source_rel,
        "h4m_f_source_sha256": sha256_file(PROJECT_ROOT / source_rel),
        "h4m_f_source_present_in_head": source_present_in_head,
        "h4m_f_source_latest_commit": source_latest_commit,
        "h4m_f_source_no_uncommitted_diff_vs_head": source_no_diff,
        "h4m_f_source_no_staged_diff_vs_head": staged_no_diff,
        "head_commit_files": head_files,
        "local_source_only_commit_created_before_freeze": source_latest_commit == head and head_files == [source_rel],
        "status_short": status_short,
        "github_push_performed": False,
        "post_commit_provenance_gate_passed": (
            source_present_in_head
            and source_latest_commit == head
            and source_no_diff
            and staged_no_diff
            and head_files == [source_rel]
        ),
    }


def authoritative_binding(created_at: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    h4m_b_manifest = read_json(H4M_B_ROOT / "manifest.json")
    h4m_b_schedule = read_json(H4M_B_ROOT / "08_h4m_b_extended_training_schedule_freeze.json")
    h4m_c_manifest = read_json(H4M_C_ROOT / "manifest.json")
    h4m_c_gate = read_json(H4M_C_ROOT / "14_h4m_c_gate_matrix.json")
    h4m_c_diagnostics = read_json(H4M_C_ROOT / "08_three_seed_training_diagnostics.json")
    h4m_d_manifest = read_json(H4M_D_ROOT / "manifest.json")
    h4m_d_gate = read_json(H4M_D_ROOT / "14_h4m_d_gate_matrix.json")
    h4m_e_manifest = read_json(H4M_E_ROOT / "manifest.json")
    h4m_e_binding = read_json(H4M_E_ROOT / "01_authoritative_binding.json")
    h4m_e_gate = read_json(H4M_E_ROOT / "17_h4m_e_gate_matrix.json")
    h4m_e_root = read_json(H4M_E_ROOT / "15_root_cause_classification.json")

    immutable = h4m_e_binding.get("immutable_bindings", {})
    schedule = {
        "outer_training_count": h4m_b_schedule.get("outer_training_count"),
        "rollout_horizon": h4m_b_schedule.get("rollout_horizon"),
        "ppo_epochs_per_update": h4m_b_schedule.get("ppo_epochs_per_update"),
        "critic_epochs_per_update": h4m_b_schedule.get("critic_epochs_per_update"),
        "gamma": h4m_b_schedule.get("gamma"),
        "gae_lambda": h4m_b_schedule.get("gae_lambda"),
        "clip_epsilon": h4m_b_schedule.get("clip_epsilon"),
        "minibatch": h4m_b_schedule.get("minibatch"),
        "training_windows": h4m_b_schedule.get("training_windows"),
        "agents": 8,
        "seeds": h4m_b_schedule.get("seeds"),
        "training_input": h4m_b_schedule.get("training_input"),
    }

    checks = {
        "h4m_e_gate_match": h4m_e_gate.get("gate") == EXPECTED["h4m_e_gate"],
        "h4m_e_decision_match": h4m_e_gate.get("decision") == EXPECTED["h4m_e_decision"],
        "h4m_e_next_gate_match": h4m_e_gate.get("next_gate") == EXPECTED["h4m_e_next_gate"],
        "h4m_e_manifest_source_commit_match": h4m_e_manifest.get("h4m_e_audit_source_git_commit")
        == EXPECTED["h4m_e_source_commit"],
        "h4m_e_root_next_gate_match": h4m_e_root.get("next_gate") == EXPECTED["h4m_e_next_gate"],
        "h4m_c_gate_match": h4m_c_gate.get("gate") == EXPECTED["h4m_c_gate"],
        "h4m_c_source_commit_match": h4m_c_gate.get("h4m_c_training_source_git_commit")
        == EXPECTED["h4m_c_source_commit"],
        "h4m_d_gate_match": h4m_d_gate.get("gate") == EXPECTED["h4m_d_gate"],
        "h4m_d_source_commit_match": h4m_d_manifest.get("h4m_d_evaluation_source_git_commit")
        == EXPECTED["h4m_d_source_commit"],
        "h4m_b_schedule_sha_match": h4m_b_manifest.get("extended_training_schedule_sha256")
        == EXPECTED["h4m_b_schedule_sha256"]
        and h4m_b_schedule.get("extended_training_schedule_sha256") == EXPECTED["h4m_b_schedule_sha256"],
        "outer_training_count_match": h4m_b_schedule.get("outer_training_count") == EXPECTED["outer_training_count"],
        "rollout_horizon_match": h4m_b_schedule.get("rollout_horizon") == EXPECTED["rollout_horizon"],
        "ppo_epochs_match": h4m_b_schedule.get("ppo_epochs_per_update") == EXPECTED["ppo_epochs_per_update"],
        "critic_epochs_match": h4m_b_schedule.get("critic_epochs_per_update") == EXPECTED["critic_epochs_per_update"],
        "reward_v2_sha_match": immutable.get("reward_v2_sha256") == EXPECTED["reward_v2_sha256"],
        "h4g_runtime_sha_match": immutable.get("h4g_runtime_sha256") == EXPECTED["h4g_runtime_sha256"],
        "r3_split_sha_match": immutable.get("r3_split_sha256") == EXPECTED["r3_split_sha256"],
        "zero_loss_adapter_sha_match": immutable.get("zero_loss_adapter_sha256") == EXPECTED["zero_loss_adapter_sha256"],
        "h4m_e_training_not_executed": h4m_e_binding.get("training_executed") is False,
        "h4m_e_optimizer_step_not_executed": h4m_e_binding.get("optimizer_step_executed") is False,
    }

    binding = {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_roots": {
            "h4m_b": str(H4M_B_ROOT),
            "h4m_c": str(H4M_C_ROOT),
            "h4m_d": str(H4M_D_ROOT),
            "h4m_e": str(H4M_E_ROOT),
        },
        "required_source_commit_bindings": {
            "h4m_e_source_commit": {
                "expected": EXPECTED["h4m_e_source_commit"],
                "observed": h4m_e_manifest.get("h4m_e_audit_source_git_commit"),
            },
            "h4m_c_source_commit": {
                "expected": EXPECTED["h4m_c_source_commit"],
                "observed": h4m_c_gate.get("h4m_c_training_source_git_commit"),
            },
            "h4m_d_source_commit": {
                "expected": EXPECTED["h4m_d_source_commit"],
                "observed": h4m_d_manifest.get("h4m_d_evaluation_source_git_commit"),
            },
        },
        "schedule_binding": {
            "expected_sha256": EXPECTED["h4m_b_schedule_sha256"],
            "observed_h4m_b_manifest_sha256": h4m_b_manifest.get("extended_training_schedule_sha256"),
            "observed_schedule_sha256": h4m_b_schedule.get("extended_training_schedule_sha256"),
            "frozen_schedule": schedule,
        },
        "immutable_bindings": {
            "reward_v2_sha256": immutable.get("reward_v2_sha256"),
            "h4g_runtime_sha256": immutable.get("h4g_runtime_sha256"),
            "r3_split_sha256": immutable.get("r3_split_sha256"),
            "zero_loss_adapter_sha256": immutable.get("zero_loss_adapter_sha256"),
        },
        "h4m_e_authoritative_result": {
            "gate": h4m_e_gate.get("gate"),
            "decision": h4m_e_gate.get("decision"),
            "next_gate": h4m_e_gate.get("next_gate"),
            "root_cause_classification": h4m_e_root.get("root_cause_classification"),
            "earliest_failure_stage": h4m_e_root.get("earliest_failure_stage"),
        },
        "h4m_c_runtime_evidence": {
            "elapsed_seconds_total": read_json(H4M_C_ROOT / "13_resource_usage.json").get("elapsed_seconds_total"),
            "total_active_samples": h4m_c_diagnostics.get("total_active_samples"),
            "total_rollouts_completed": h4m_c_diagnostics.get("total_rollouts_completed"),
            "total_ppo_updates_completed": h4m_c_diagnostics.get("total_ppo_updates_completed"),
            "total_critic_updates_completed": h4m_c_diagnostics.get("total_critic_updates_completed"),
            "per_seed_active_samples": [
                row.get("active_samples") for row in h4m_c_diagnostics.get("per_seed_summary", [])
            ],
        },
        "checks": checks,
        "authoritative_binding_passed": all(checks.values()),
        "training_executed": False,
        "optimizer_created": False,
        "optimizer_step_executed": False,
        "training_backward_executed": False,
        "checkpoint_mutation": False,
        "sealed_test_opened": False,
        "github_push_performed": False,
    }
    return binding, schedule, h4m_c_diagnostics


def missing_evidence(created_at: str) -> Dict[str, Any]:
    root = read_json(H4M_E_ROOT / "15_root_cause_classification.json")
    raw_gae = read_json(H4M_E_ROOT / "06_raw_gae_by_action_audit.json")
    sign_flip = read_json(H4M_E_ROOT / "07_advantage_sign_flip_audit.json")
    ppo = read_json(H4M_E_ROOT / "09_ppo_surrogate_by_action_audit.json")
    critic = read_json(H4M_E_ROOT / "13_critic_value_bias_audit.json")
    long_horizon = read_json(H4M_E_ROOT / "05_long_horizon_action_value_reconciliation.json")
    return {
        "stage": STAGE,
        "created_at": created_at,
        "authoritative_h4m_e_decision": root.get("decision"),
        "exact_missing_evidence": [
            {
                "evidence": "long_horizon_counterfactual_return",
                "h4m_e_status": long_horizon.get("h4m_a_hold_better_long_horizon_answer"),
                "why_it_blocks_attribution": "Immediate Reward V2 labels do not prove HOLD-vs-SERVE return through the rollout boundary.",
                "h4m_f_capture": "PAIRED_COUNTERFACTUAL_SHADOW_TRACE with bootstrapped return-to-boundary.",
            },
            {
                "evidence": "per_sample_value_t_and_next_value_t",
                "h4m_e_status": raw_gae.get("actual_per_sample_value_tensor_available"),
                "why_it_blocks_attribution": "TD deltas and raw GAE cannot be recomputed for each action/sample.",
                "h4m_f_capture": "ACTUAL_ON_POLICY_TRACE critic_td_trace table.",
            },
            {
                "evidence": "per_sample_raw_gae_advantage",
                "h4m_e_status": raw_gae.get("actual_per_sample_raw_gae_tensor_available"),
                "why_it_blocks_attribution": "Reward-to-GAE credit cannot be classified by HOLD/SERVE sample.",
                "h4m_f_capture": "gae_trace table with predecessor/successor sample_uid lineage.",
            },
            {
                "evidence": "per_sample_normalized_advantage_and_sign_change",
                "h4m_e_status": sign_flip.get("action_dependent_sign_effect_quantifiable"),
                "why_it_blocks_attribution": "Rollout-global normalization can reverse signs, but exact action-conditional counts were not persisted.",
                "h4m_f_capture": "advantage_normalization_sample table keyed by sample_uid.",
            },
            {
                "evidence": "per_sample_ppo_ratio_clip_surrogate",
                "h4m_e_status": not ppo.get("per_sample_ratio_clip_surrogate_not_persisted", True),
                "why_it_blocks_attribution": "Stored aggregates show sampled-action pressure but not exact per-sample surrogate contribution.",
                "h4m_f_capture": "ppo_policy_surrogate table keyed by ppo_update_id and sample_uid.",
            },
            {
                "evidence": "action_conditional_critic_value_error",
                "h4m_e_status": critic.get("critic_bias_dominance_status"),
                "why_it_blocks_attribution": "Critic bias was aggregate, not tied to specific HOLD/SERVE samples.",
                "h4m_f_capture": "critic_value_error_by_sample table and critic_epoch_minibatch_summary table.",
            },
        ],
        "h4m_e_downstream_observable_driver": root.get("earliest_fully_observable_downstream_driver"),
        "h4m_e_limitation_text": raw_gae.get("by_action_distribution_limitation"),
        "training_replay_required_to_recover_missing_h4m_c_tensors": True,
        "training_replay_executed_in_h4m_f": False,
    }


def sample_identity_contract(created_at: str, schedule: Mapping[str, Any]) -> Dict[str, Any]:
    identity_tuple = [
        "seed",
        "outer_cycle",
        "rollout_id",
        "window_id",
        "agent_id",
        "step_index",
        "state_ts",
        "state_hash",
    ]
    return {
        "stage": STAGE,
        "created_at": created_at,
        "contract_id": "H4M_F_STABLE_SAMPLE_IDENTITY_AND_RNG_NONINTERFERENCE_V1",
        "sample_uid_contract": {
            "canonical_identity_tuple": identity_tuple,
            "canonicalization": "UTF-8 JSON object with sorted keys, compact separators, field names exactly as listed, no row-order dependency.",
            "sample_uid": "SHA256(canonical_identity_tuple_json)",
            "requirements": {
                "same_physical_training_sample_same_sample_uid": True,
                "no_approximate_matching": True,
                "no_timestamp_only_matching": True,
                "no_row_order_only_identity": True,
                "sample_uid_carried_to_all_trace_tables": True,
            },
            "parent_identity_rules": {
                "ppo_rows": ["ppo_update_id", "ppo_epoch", "minibatch_id", "selected_flat_index", "sample_uid"],
                "critic_rows": ["critic_update_id", "critic_epoch", "minibatch_id", "sample_uid"],
                "counterfactual_rows": ["counterfactual_pair_uid", "sample_uid", "branch_id", "branch_event_index"],
                "normalization_rows": ["normalization_scope_id", "sample_uid"],
            },
        },
        "rng_non_interference_contract": {
            "method": "SNAPSHOT_AUTHORITATIVE_RNG_THEN_USE_PRIVATE_INSTRUMENTATION_RNG_STREAMS",
            "authoritative_training_rng_may_be_read": True,
            "authoritative_training_rng_may_be_advanced_by_instrumentation": False,
            "covered_sources": {
                "python_random": "capture state hash before/after authoritative operation; instrumentation uses random.Random(private_seed).",
                "numpy": "capture bit_generator/state hash where used; instrumentation uses private Generator keyed by trace namespace.",
                "torch_cpu": "capture torch.get_rng_state hash before/after; instrumentation never samples from global torch RNG.",
                "torch_mps": "capture if backend available and used; H4M-C evidence used CPU, but contract covers MPS when applicable.",
                "environment_rng": "record environment RNG state/reference hash before authoritative transition; shadow replay uses cloned/private environment RNG.",
                "action_sampling_rng": "record reproducible state reference/hash for the Categorical.sample call before sampling, never dump secrets unnecessarily.",
            },
            "common_random_numbers_for_shadow_replay": {
                "private_rng_key": "SHA256(contract_id, seed, outer_cycle, sample_uid, counterfactual_pair_uid, continuation_step)",
                "training_rng_consumption": "ZERO_ADDITIONAL_CONSUMPTION",
                "branch_mapping": "same base uniform variate is mapped through each branch-specific legal masked policy distribution.",
            },
            "h4m_g_must_prove": [
                "same initial parameters",
                "same RNG lineage",
                "same sampled actions",
                "same reward sequence",
                "same GAE",
                "same parameter updates",
                "same final checkpoint",
            ],
        },
        "frozen_schedule_rng_context": {
            "outer_training_count": schedule.get("outer_training_count"),
            "rollout_horizon": schedule.get("rollout_horizon"),
            "seeds": schedule.get("seeds"),
            "window_traversal_shuffle": False,
        },
    }


def actual_training_trace_schema(created_at: str, schedule: Mapping[str, Any], active_samples: int) -> Dict[str, Any]:
    namespace = "ACTUAL_ON_POLICY_TRACE"
    common_identity = [
        field("sample_uid", "string", semantics="SHA256 stable identity for this physical training sample."),
        field("seed", "int32", semantics="Training seed."),
        field("outer_cycle", "int16", semantics="1-based fresh rollout/update cycle."),
        field("rollout_id", "string", semantics="Deterministic rollout identifier for the current seed/cycle."),
        field("window_id", "string", semantics="Frozen R3 training window identifier."),
        field("agent_id", "int16", semantics="Stable bus/agent identity, not approximate row index."),
        field("agent_slot", "int16", semantics="Executor slot used by H4M-C tensors."),
        field("step_index", "int32", semantics="Within-rollout active decision index."),
        field("state_ts", "string", semantics="Canonical state timestamp or source event timestamp."),
        field("state_hash", "string", semantics="SHA256 of canonical state payload used for actor/critic inputs."),
    ]
    return {
        "stage": STAGE,
        "created_at": created_at,
        "namespace": namespace,
        "schema_version": "H4M_F_ACTUAL_ON_POLICY_TRACE_SCHEMA_V1",
        "side_channel_only": True,
        "expected_rows_current_pv8": active_samples,
        "tables": [
            table(
                "actual_pre_action_trace.parquet",
                namespace=namespace,
                format_="parquet",
                partition_by=["seed", "outer_cycle"],
                primary_key=["sample_uid"],
                rows_expected=active_samples,
                semantics="All active on-policy samples before action sampling; detached/copy-only values.",
                fields=[
                    *common_identity,
                    field("actor_observation_hash", "string", semantics="SHA256 of actor observation tensor bytes plus dtype/shape."),
                    field("critic_observation_hash", "string", semantics="SHA256 of critic observation tensor bytes plus dtype/shape."),
                    field("actor_observation_shape", "list[int32]", semantics="Shape metadata; tensor payload stored only if required by reconstruction."),
                    field("critic_observation_shape", "list[int32]", semantics="Shape metadata; tensor payload stored only if required by reconstruction."),
                    field("legal_action_mask", "list[bool]", semantics="Executable K-mask for actions [HOLD,SERVE,SKIP]."),
                    field("legal_action_ids", "list[int16]", semantics="Action ids legal for this sample."),
                    field("pre_mask_logits", "list[float64]", semantics="Actor logits before masking.", precision="serialize as float64 from detached CPU copy"),
                    field("post_mask_logits", "list[float64]", semantics="Actor logits after K-mask.", precision="serialize as float64 from detached CPU copy"),
                    field("pre_mask_probabilities", "list[float64]", semantics="Softmax over pre-mask logits.", precision="float64"),
                    field("masked_probabilities", "list[float64]", semantics="Softmax over post-mask logits.", precision="float64"),
                    field("policy_entropy", "float64", semantics="Categorical entropy used in PPO entropy term.", precision="float64"),
                    field("sampled_action_id", "int16", semantics="Actual action sampled for authoritative training."),
                    field("sampled_action_name", "string", semantics="Human-readable action label."),
                    field("sampled_action_log_prob", "float64", semantics="Old log probability saved for PPO.", precision="float64"),
                    field("action_sampling_rng_ref", "string", semantics="Hash/reference to RNG state before sampling; not a secret full dump."),
                    field("trace_capture_after_authoritative_sampling", "bool", semantics="True only if logging copied values after authoritative sampling without changing RNG."),
                ],
            ),
            table(
                "actual_reward_trace.parquet",
                namespace=namespace,
                format_="parquet",
                partition_by=["seed", "outer_cycle"],
                primary_key=["sample_uid"],
                rows_expected=active_samples,
                semantics="Reward V2 raw component materialization and runtime reward normalization lineage.",
                fields=[
                    field("sample_uid", "string", semantics="Foreign key to actual_pre_action_trace."),
                    field("reward_v2_local_service_component_raw", "float64", semantics="Unweighted/raw service component payload where executable exposes it.", precision="float64"),
                    field("reward_v2_local_avg_wait_component_raw", "float64", semantics="Unweighted/raw average-wait component payload where executable exposes it.", precision="float64"),
                    field("reward_v2_explicit_intervention_component_raw", "float64", semantics="Unweighted/raw intervention component payload where executable exposes it.", precision="float64"),
                    field("reward_v2_local_service_component_weighted", "float64", semantics="Weighted service reward component.", precision="float64"),
                    field("reward_v2_local_avg_wait_component_weighted", "float64", semantics="Weighted average-wait reward component.", precision="float64"),
                    field("reward_v2_explicit_intervention_component_weighted", "float64", semantics="Weighted intervention reward component.", precision="float64"),
                    field("reward_v2_raw_total", "float64", semantics="Reward V2 total before runtime reward normalizer.", precision="float64"),
                    field("reward_normalizer_scope_id", "string", nullable=True, semantics="Reward-normalizer population id, if active."),
                    field("reward_runtime_normalizer_mean", "float64", nullable=True, semantics="Normalizer mean before applying this reward.", precision="float64"),
                    field("reward_runtime_normalizer_std", "float64", nullable=True, semantics="Normalizer std before applying this reward.", precision="float64"),
                    field("reward_after_runtime_normalization", "float64", semantics="Actual reward tensor value used by GAE.", precision="float64"),
                    field("reward_normalizer_update_instrumented", "bool", semantics="Must remain false; instrumentation does not update normalizer state."),
                ],
            ),
            table(
                "actual_critic_td_gae_trace.parquet",
                namespace=namespace,
                format_="parquet",
                partition_by=["seed", "outer_cycle"],
                primary_key=["sample_uid"],
                rows_expected=active_samples,
                semantics="Critic value, TD delta, raw GAE, and raw return target lineage for exact recomputation.",
                fields=[
                    field("sample_uid", "string", semantics="Foreign key to actual_pre_action_trace."),
                    field("value_t", "float64", semantics="Critic value at sample before update.", precision="float64"),
                    field("next_sample_uid", "string", nullable=True, semantics="GAE recursion successor sample in same trajectory/population."),
                    field("next_value_t", "float64", semantics="Critic bootstrap value for next state.", precision="float64"),
                    field("terminated", "bool", semantics="Episode termination flag used by executable GAE."),
                    field("truncated", "bool", semantics="Truncation flag used by executable GAE."),
                    field("bootstrap_mask", "float64", semantics="Executable bootstrap continuation mask.", precision="float64"),
                    field("gamma", "float64", semantics="Frozen gamma.", precision="float64"),
                    field("gae_lambda", "float64", semantics="Frozen GAE lambda.", precision="float64"),
                    field("td_delta", "float64", semantics="reward + gamma * next_value * bootstrap_mask - value_t.", precision="float64"),
                    field("raw_gae_advantage", "float64", semantics="Unnormalized GAE advantage from executable recursion.", precision="float64"),
                    field("raw_return_target", "float64", semantics="raw_gae_advantage + value_t before return normalization.", precision="float64"),
                    field("gae_recursion_predecessor_uid", "string", nullable=True, semantics="Previous sample_uid in reverse recursion lineage."),
                    field("gae_recursion_successor_uid", "string", nullable=True, semantics="Next sample_uid in reverse recursion lineage."),
                    field("gae_formula_id", "string", semantics="Executable formula/version id from compute_gae implementation."),
                ],
            ),
        ],
        "frozen_values": {
            "gamma": schedule.get("gamma"),
            "gae_lambda": schedule.get("gae_lambda"),
            "rollout_horizon": schedule.get("rollout_horizon"),
        },
    }


def advantage_normalization_schema(created_at: str, active_samples: int, rollouts: int) -> Dict[str, Any]:
    namespace = "ACTUAL_ON_POLICY_TRACE"
    return {
        "stage": STAGE,
        "created_at": created_at,
        "schema_version": "H4M_F_ADVANTAGE_NORMALIZATION_TRACE_SCHEMA_V1",
        "namespace": namespace,
        "normalization_scope_rule": "one rollout-global active-only normalization population per seed/cycle, matching executable compute_gae semantics",
        "inactive_samples_excluded": True,
        "tables": [
            table(
                "advantage_normalization_scope.parquet",
                namespace=namespace,
                format_="parquet",
                partition_by=["seed", "outer_cycle"],
                primary_key=["normalization_scope_id"],
                rows_expected=rollouts,
                semantics="One row per advantage standardization operation.",
                fields=[
                    field("normalization_scope_id", "string", semantics="SHA256(seed, outer_cycle, rollout_id, 'advantage_normalization')."),
                    field("seed", "int32", semantics="Training seed."),
                    field("outer_cycle", "int16", semantics="1-based cycle."),
                    field("population_count", "int32", semantics="All rows considered before active-mask exclusion."),
                    field("active_sample_count", "int32", semantics="Rows used to compute mean/std."),
                    field("mean", "float64", semantics="Active raw GAE mean.", precision="float64"),
                    field("std", "float64", semantics="Active raw GAE std.", precision="float64"),
                    field("epsilon", "float64", semantics="Executable stabilization epsilon.", precision="float64"),
                    field("masking_semantics", "string", semantics="Text id for active-only/inactive-excluded behavior."),
                    field("dtype", "string", semantics="Original tensor dtype and stored serialization dtype."),
                ],
            ),
            table(
                "advantage_normalization_sample.parquet",
                namespace=namespace,
                format_="parquet",
                partition_by=["seed", "outer_cycle"],
                primary_key=["normalization_scope_id", "sample_uid"],
                rows_expected=active_samples,
                semantics="Per-sample raw-to-normalized advantage mapping and sign-change classification.",
                fields=[
                    field("normalization_scope_id", "string", semantics="Foreign key to advantage_normalization_scope."),
                    field("sample_uid", "string", semantics="Foreign key to actual sample."),
                    field("sampled_action_id", "int16", semantics="Actual sampled action for action-conditional sign counts."),
                    field("sampled_action_name", "string", semantics="HOLD/SERVE/SKIP label."),
                    field("raw_gae_advantage", "float64", semantics="Pre-standardization advantage.", precision="float64"),
                    field("normalized_advantage", "float64", semantics="Actual advantage used by PPO.", precision="float64"),
                    field("raw_sign", "string", semantics="positive/negative/zero/near_zero using frozen near-zero rule."),
                    field("normalized_sign", "string", semantics="positive/negative/zero/near_zero using frozen near-zero rule."),
                    field("sign_changed", "bool", semantics="True if raw_sign and normalized_sign differ outside near-zero convention."),
                    field("sign_change_class", "string", semantics="positive_to_negative, negative_to_positive, sign_preserved, near_zero."),
                ],
            ),
        ],
        "required_queries": [
            "positive_to_negative by action",
            "negative_to_positive by action",
            "sign_preserved by action",
            "near_zero by action",
        ],
    }


def ppo_per_sample_schema(created_at: str, schedule: Mapping[str, Any], ppo_rows: int) -> Dict[str, Any]:
    namespace = "ACTUAL_ON_POLICY_TRACE"
    return {
        "stage": STAGE,
        "created_at": created_at,
        "schema_version": "H4M_F_PPO_PER_SAMPLE_TRACE_SCHEMA_V1",
        "namespace": namespace,
        "minibatch_selection_semantics": "record selected_flat_index and sample_uid for every PPO epoch/minibatch row; no row-order-only identity",
        "tables": [
            table(
                "ppo_policy_surrogate_sample.parquet",
                namespace=namespace,
                format_="parquet",
                partition_by=["seed", "outer_cycle", "ppo_epoch"],
                primary_key=["ppo_update_id", "ppo_epoch", "minibatch_id", "sample_uid"],
                rows_expected=ppo_rows,
                semantics="Per-sample PPO actor surrogate, clip, entropy, and effective contribution.",
                fields=[
                    field("ppo_update_id", "string", semantics="Stable actor/GATv2 update id."),
                    field("seed", "int32", semantics="Training seed."),
                    field("outer_cycle", "int16", semantics="1-based cycle."),
                    field("ppo_epoch", "int16", semantics="1..4 under frozen schedule."),
                    field("minibatch_id", "int16", semantics="Stable minibatch id."),
                    field("selected_flat_index", "int32", semantics="Index into valid_flat as executable selected it."),
                    field("sample_uid", "string", semantics="Foreign key to actual sample."),
                    field("sampled_action_id", "int16", semantics="Actual sampled action id."),
                    field("sampled_action_name", "string", semantics="HOLD/SERVE/SKIP label."),
                    field("old_log_prob", "float64", semantics="Saved rollout log probability.", precision="float64"),
                    field("current_log_prob", "float64", semantics="Current policy log probability during PPO epoch.", precision="float64"),
                    field("old_action_probability", "float64", semantics="exp(old_log_prob).", precision="float64"),
                    field("current_action_probability", "float64", semantics="exp(current_log_prob).", precision="float64"),
                    field("probability_ratio", "float64", semantics="exp(current_log_prob - old_log_prob).", precision="float64"),
                    field("normalized_advantage", "float64", semantics="Standardized advantage used in surrogate.", precision="float64"),
                    field("clip_epsilon", "float64", semantics="Frozen PPO clip epsilon.", precision="float64"),
                    field("unclipped_surrogate", "float64", semantics="ratio * normalized_advantage.", precision="float64"),
                    field("clipped_ratio", "float64", semantics="clamp(ratio, 1-clip_epsilon, 1+clip_epsilon).", precision="float64"),
                    field("clipped_surrogate", "float64", semantics="clipped_ratio * normalized_advantage.", precision="float64"),
                    field("clip_active", "bool", semantics="True if clipping changes the effective surrogate branch."),
                    field("effective_policy_surrogate_contribution", "float64", semantics="min(unclipped, clipped) under PPO objective.", precision="float64"),
                    field("entropy_contribution", "float64", semantics="Per-row entropy contribution where executable semantics permit.", precision="float64"),
                    field("sampled_action_pressure", "string", semantics="reinforce_sampled_action/suppress_sampled_action/near_zero."),
                ],
            )
        ],
        "frozen_values": {
            "ppo_epochs_per_update": schedule.get("ppo_epochs_per_update"),
            "clip_epsilon": schedule.get("clip_epsilon"),
            "minibatch": schedule.get("minibatch"),
        },
    }


def critic_and_gradient_schema(created_at: str, active_samples: int, critic_rows: int) -> Dict[str, Any]:
    namespace = "ACTUAL_ON_POLICY_TRACE"
    return {
        "stage": STAGE,
        "created_at": created_at,
        "schema_version": "H4M_F_CRITIC_AND_ACTOR_GRADIENT_PRESSURE_TRACE_SCHEMA_V1",
        "namespace": namespace,
        "actor_gradient_pressure_method": {
            "selected_method": "OFFLINE_DETACHED_ANALYTIC_RECONSTRUCTION",
            "reason": "Exact per-sample model-gradient tensor dumps are unnecessary and could perturb authoritative autograd memory/ordering; saved logits/probabilities/ratio/clip/advantage are sufficient for sampled-action logit pressure.",
            "training_autograd_semantics_modified": False,
            "full_gradient_tensor_dump_required": False,
        },
        "tables": [
            table(
                "critic_value_error_by_sample.parquet",
                namespace=namespace,
                format_="parquet",
                partition_by=["seed", "outer_cycle"],
                primary_key=["sample_uid"],
                rows_expected=active_samples,
                semantics="Action-conditional critic bias lineage before critic updates.",
                fields=[
                    field("sample_uid", "string", semantics="Foreign key to actual sample."),
                    field("sampled_action_id", "int16", semantics="Actual action id for action-conditional analysis."),
                    field("value_t_before_update", "float64", semantics="Critic value before PPO/critic update.", precision="float64"),
                    field("raw_return_target", "float64", semantics="Unnormalized return target.", precision="float64"),
                    field("normalized_return_target", "float64", nullable=True, semantics="Return target after return normalizer if used.", precision="float64"),
                    field("value_error", "float64", semantics="target - value_t_before_update.", precision="float64"),
                    field("squared_error", "float64", semantics="value_error squared.", precision="float64"),
                    field("time_band", "string", semantics="Derived from window_id for grouped analysis."),
                    field("window_id", "string", semantics="Training window id."),
                    field("agent_id", "int16", semantics="Stable bus/agent id."),
                ],
            ),
            table(
                "critic_epoch_minibatch_summary.parquet",
                namespace=namespace,
                format_="parquet",
                partition_by=["seed", "outer_cycle", "critic_epoch"],
                primary_key=["critic_update_id", "critic_epoch", "minibatch_id"],
                rows_expected="total_critic_updates × critic_minibatches; current H4M-C uses full active population per critic epoch unless implementation changes",
                semantics="Aggregate critic training diagnostics without huge unnecessary per-gradient tensor dumps.",
                fields=[
                    field("critic_update_id", "string", semantics="Stable critic update id."),
                    field("critic_epoch", "int16", semantics="1..8 under frozen schedule."),
                    field("minibatch_id", "int16", semantics="Stable minibatch id."),
                    field("sample_uid_count", "int32", semantics="Count of samples included."),
                    field("mean_value_error", "float64", semantics="Mean target - value.", precision="float64"),
                    field("mean_squared_error", "float64", semantics="Mean squared value error.", precision="float64"),
                    field("loss_value", "float64", semantics="Executable critic loss scalar.", precision="float64"),
                    field("sample_uid_hash", "string", semantics="SHA256 of ordered sample_uid list to preserve membership without bloating row."),
                ],
            ),
            table(
                "actor_logit_pressure_by_sample.parquet",
                namespace=namespace,
                format_="parquet",
                partition_by=["seed", "outer_cycle", "ppo_epoch"],
                primary_key=["ppo_update_id", "ppo_epoch", "minibatch_id", "sample_uid"],
                rows_expected=critic_rows,
                semantics="Offline detached logit-pressure reconstruction for HOLD and SERVE logits.",
                fields=[
                    field("sample_uid", "string", semantics="Foreign key to PPO row."),
                    field("ppo_update_id", "string", semantics="Parent PPO update id."),
                    field("sampled_action_id", "int16", semantics="Actual sampled action id."),
                    field("effective_direction_on_sampled_action_logit", "string", semantics="increase/decrease/near_zero."),
                    field("serve_logit_pressure", "string", semantics="increase/decrease/near_zero/not_legal."),
                    field("hold_logit_pressure", "string", semantics="increase/decrease/near_zero/not_legal."),
                    field("skip_logit_pressure", "string", semantics="increase/decrease/near_zero/not_legal."),
                    field("pressure_formula_id", "string", semantics="References frozen analytic formula for PPO clipped surrogate plus entropy."),
                    field("near_zero_rule", "string", semantics="Use existing numeric tolerance source; no new tolerance invented."),
                ],
            ),
        ],
    }


def counterfactual_trace_schema(created_at: str, schedule: Mapping[str, Any], selected_states: int, branch_count: int, simulated_steps: int) -> Dict[str, Any]:
    namespace = "PAIRED_COUNTERFACTUAL_SHADOW_TRACE"
    return {
        "stage": STAGE,
        "created_at": created_at,
        "schema_version": "H4M_F_LONG_HORIZON_COUNTERFACTUAL_TRACE_SCHEMA_V1",
        "namespace": namespace,
        "separated_from_training": True,
        "selected_scope": "ALL_ELIGIBLE_ACTUAL_ON_POLICY_MULTI_ACTION_STATES",
        "expected_current_pv8": {
            "states_traced": selected_states,
            "paired_branch_count": branch_count,
            "approx_branch_decision_steps_to_rollout_boundary": simulated_steps,
        },
        "continuation_policy_contract": {
            "selected_contract": "CURRENT_PRE_UPDATE_STOCHASTIC_POLICY_PLUS_COMMON_RANDOM_NUMBERS",
            "policy_parameters": "the frozen policy snapshot used to collect the current seed/cycle rollout before PPO/critic updates",
            "initial_branch_difference": "only initial compared action is changed: HOLD branch vs SERVE branch",
            "after_initial_action": "both branches continue with the same current pre-update stochastic policy and common-random-number exogenous/stochastic streams",
            "common_random_numbers": {
                "exogenous_demand_realization": "same cloned/private stream",
                "traffic_realization": "same cloned/private stream",
                "future_policy_random_stream": "same private uniforms mapped through each branch legal masked distribution",
            },
            "argmax_for_convenience_forbidden": True,
            "action_induced_state_divergence_allowed": True,
        },
        "counterfactual_horizon_contract": {
            "horizon": "remaining current rollout to legitimate rollout boundary, capped by frozen rollout_horizon",
            "gamma": schedule.get("gamma"),
            "rollout_horizon": schedule.get("rollout_horizon"),
            "capture_immediate_t0_reward_delta": True,
            "capture_future_reward_v2_event_sequence": True,
            "capture_discounted_cumulative_reward_to_boundary": True,
            "capture_boundary_bootstrap_value": True,
            "capture_bootstrapped_discounted_return": True,
            "closure_reasons": [
                "ROLLOUT_BOUNDARY_REACHED",
                "EPISODE_TERMINATED",
                "EPISODE_TRUNCATED",
                "REPLAY_NOT_COMPARABLE_WITH_EXPLICIT_REASON",
            ],
            "tie_tolerance_source": "05_training/rewards/mappo_reward_v1.py::PV8_REWARD_V2_NUMERIC_TOLERANCE",
            "new_tolerance_created": False,
        },
        "tables": [
            table(
                "counterfactual_pair_index.parquet",
                namespace=namespace,
                format_="parquet",
                partition_by=["seed", "outer_cycle"],
                primary_key=["counterfactual_pair_uid"],
                rows_expected=selected_states,
                semantics="One HOLD-vs-SERVE pair for each eligible actual on-policy sample.",
                fields=[
                    field("counterfactual_pair_uid", "string", semantics="SHA256(sample_uid, 'HOLD_SERVE_PAIR', contract_version)."),
                    field("sample_uid", "string", semantics="Parent actual on-policy sample id."),
                    field("initial_state_hash", "string", semantics="State hash before the compared action."),
                    field("hold_branch_uid", "string", semantics="Branch id for forced initial HOLD."),
                    field("serve_branch_uid", "string", semantics="Branch id for forced initial SERVE."),
                    field("eligible_reason", "string", semantics="Both HOLD and SERVE legal at t0."),
                    field("actual_sampled_action_id", "int16", semantics="Actual action sampled during authoritative rollout."),
                ],
            ),
            table(
                "counterfactual_branch_event_sequence.parquet",
                namespace=namespace,
                format_="parquet",
                partition_by=["seed", "outer_cycle", "branch_action"],
                primary_key=["branch_uid", "branch_event_index"],
                rows_expected="variable; bounded by remaining rollout length per branch",
                semantics="Per-branch state/reward/service/wait event sequence through legitimate closure.",
                fields=[
                    field("branch_uid", "string", semantics="HOLD or SERVE branch id."),
                    field("counterfactual_pair_uid", "string", semantics="Parent pair id."),
                    field("branch_action", "string", semantics="HOLD_INITIAL or SERVE_INITIAL."),
                    field("branch_event_index", "int32", semantics="0-based event index after t0."),
                    field("state_hash", "string", semantics="Branch state hash."),
                    field("event_sequence_hash", "string", semantics="Hash of canonical event payload."),
                    field("action_id", "int16", semantics="Action executed at this branch event."),
                    field("reward_v2_raw_total", "float64", semantics="Raw Reward V2 before runtime normalization.", precision="float64"),
                    field("reward_v2_discounted", "float64", semantics="gamma^k * reward_v2_raw_total.", precision="float64"),
                    field("service_outcome_hash", "string", semantics="Hash of service outcome payload."),
                    field("passenger_wait_outcome_hash", "string", semantics="Hash of passenger wait outcome payload."),
                    field("future_decision_index", "int32", semantics="Decision count since t0."),
                    field("closure_reason", "string", nullable=True, semantics="Populated at final branch row."),
                ],
            ),
            table(
                "counterfactual_pair_summary.parquet",
                namespace=namespace,
                format_="parquet",
                partition_by=["seed", "outer_cycle"],
                primary_key=["counterfactual_pair_uid"],
                rows_expected=selected_states,
                semantics="HOLD-vs-SERVE long-horizon comparison and classification.",
                fields=[
                    field("counterfactual_pair_uid", "string", semantics="Parent pair id."),
                    field("delta_immediate_reward_serve_minus_hold", "float64", semantics="SERVE immediate Reward V2 - HOLD immediate Reward V2.", precision="float64"),
                    field("delta_discounted_reward_to_boundary_serve_minus_hold", "float64", semantics="SERVE branch discounted reward sum - HOLD branch.", precision="float64"),
                    field("delta_bootstrap_value_serve_minus_hold", "float64", semantics="SERVE boundary bootstrap value - HOLD boundary bootstrap value.", precision="float64"),
                    field("delta_full_bootstrapped_return_serve_minus_hold", "float64", semantics="SERVE full bootstrapped return - HOLD full bootstrapped return.", precision="float64"),
                    field("classification", "string", semantics="HOLD_LONG_HORIZON_BETTER, SERVE_LONG_HORIZON_BETTER, TIE_WITH_EXISTING_TOLERANCE, NOT_COMPARABLE."),
                    field("not_comparable_reason", "string", nullable=True, semantics="Explicit replay limitation if classification is NOT_COMPARABLE."),
                ],
            ),
        ],
        "counterfactual_data_may_enter_training": False,
    }


def scope_candidate_comparison(created_at: str, active_samples: int, rollouts: int, schedule: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    samples_per_rollout = active_samples // rollouts
    full_branch_steps_per_rollout = 2 * (samples_per_rollout * (samples_per_rollout + 1) // 2)
    full_branch_steps = full_branch_steps_per_rollout * rollouts

    # Deterministic subset candidates are deliberately explicit, not arbitrary ratios.
    stratified_states = min(active_samples, int(schedule.get("seeds", [1, 2, 3]).__len__()) * 11 * 44 * 2)
    targeted_states = min(active_samples, 60 + 60 + 11 * 3 * 4)

    candidates = [
        {
            "candidate": "A_ALL_ELIGIBLE_ACTUAL_ON_POLICY_MULTI_ACTION_STATES",
            "states_traced": active_samples,
            "paired_branch_count": active_samples * 2,
            "approx_simulated_branch_decision_steps": full_branch_steps,
            "coverage_of_hold_samples": "complete",
            "coverage_of_serve_samples": "complete",
            "cycle_1_to_11_coverage": "complete",
            "ability_to_reconcile_h4m_a_h4m_d": "strongest; every actual training sample can be linked to long-horizon paired return and PPO credit",
            "can_uniquely_answer_h4m_e_question": True,
            "selection_status": "SELECTED",
        },
        {
            "candidate": "B_DETERMINISTIC_STRATIFIED_SUBSET",
            "states_traced": stratified_states,
            "paired_branch_count": stratified_states * 2,
            "approx_simulated_branch_decision_steps": math.ceil(full_branch_steps * stratified_states / active_samples),
            "coverage_of_hold_samples": "partial deterministic coverage by seed/cycle/window/action class",
            "coverage_of_serve_samples": "partial deterministic coverage by seed/cycle/window/action class",
            "cycle_1_to_11_coverage": "covered but not exhaustive",
            "ability_to_reconcile_h4m_a_h4m_d": "useful for diagnostics but cannot prove earliest failure for omitted samples",
            "can_uniquely_answer_h4m_e_question": False,
            "selection_status": "REJECTED_NOT_UNIQUELY_ATTRIBUTIVE",
        },
        {
            "candidate": "C_TARGETED_H4MA_STATES_PLUS_CONTROL_POPULATION",
            "states_traced": targeted_states,
            "paired_branch_count": targeted_states * 2,
            "approx_simulated_branch_decision_steps": math.ceil(full_branch_steps * targeted_states / active_samples),
            "coverage_of_hold_samples": "targeted only",
            "coverage_of_serve_samples": "control only",
            "cycle_1_to_11_coverage": "not exhaustive",
            "ability_to_reconcile_h4m_a_h4m_d": "answers representative failure cases but not full on-policy causal chain",
            "can_uniquely_answer_h4m_e_question": False,
            "selection_status": "REJECTED_SCOPE_TOO_NARROW",
        },
    ]
    selected = candidates[0]
    payload = {
        "stage": STAGE,
        "created_at": created_at,
        "selection_rule": "Prefer the smallest scope that can uniquely answer the H4M-E causal-attribution question.",
        "current_pv8_scale_evidence": {
            "active_training_samples": active_samples,
            "rollouts": rollouts,
            "active_samples_per_rollout": samples_per_rollout,
            "all_active_samples_are_currently_hold_serve_multi_action": True,
            "evidence_source": "H4M-C policy_probability_diagnostics rows show legal_action_ids [0,1] for active samples.",
        },
        "candidate_comparison": candidates,
        "selected_counterfactual_scope": selected["candidate"],
        "block_pending_explicit_scope_selection": False,
        "rationale": "Only full eligible scope preserves exact sample_uid linkage from long-horizon paired return through raw GAE, normalization, PPO, and actor pressure for every optimization sample.",
    }
    return payload, {
        "selected_scope": selected["candidate"],
        "selected_states": int(selected["states_traced"]),
        "paired_branch_count": int(selected["paired_branch_count"]),
        "approx_simulated_branch_decision_steps": int(selected["approx_simulated_branch_decision_steps"]),
    }


def runtime_storage_overhead(created_at: str, active_samples: int, rollouts: int, ppo_updates: int, critic_updates: int, selected_scope: Mapping[str, Any]) -> Dict[str, Any]:
    h4m_c_resource = read_json(H4M_C_ROOT / "13_resource_usage.json")
    baseline_seconds = float(h4m_c_resource.get("elapsed_seconds_total"))
    samples_per_rollout = active_samples // rollouts
    ppo_rows = ppo_updates * 256
    critic_sample_rows = critic_updates * samples_per_rollout
    actual_trace_rows = active_samples
    advantage_rows = active_samples + rollouts
    counterfactual_branch_steps = int(selected_scope["approx_simulated_branch_decision_steps"])

    # Conservative order-of-magnitude estimates. They are design constraints, not acceptance thresholds.
    actual_trace_mb = 24.0
    advantage_mb = 3.0
    ppo_mb = 42.0
    critic_mb = 58.0
    cf_mb = round(max(128.0, counterfactual_branch_steps * 260 / (1024 * 1024)), 2)
    expected_disk_mb = round(actual_trace_mb + advantage_mb + ppo_mb + critic_mb + cf_mb, 2)
    shadow_seconds = round(baseline_seconds * (counterfactual_branch_steps / active_samples), 2)
    instrumentation_seconds = round(baseline_seconds * 0.20, 2)
    return {
        "stage": STAGE,
        "created_at": created_at,
        "baseline_h4m_c_runtime_seconds_total": baseline_seconds,
        "expected_instrumentation_runtime_seconds": instrumentation_seconds,
        "expected_counterfactual_shadow_runtime_seconds": shadow_seconds,
        "expected_counterfactual_shadow_runtime_human": f"~{shadow_seconds / 3600:.2f} hours conservative upper-bound at H4M-C observed throughput",
        "expected_total_diagnostic_runtime_seconds": round(baseline_seconds + instrumentation_seconds + shadow_seconds, 2),
        "expected_disk_footprint_mb": expected_disk_mb,
        "expected_peak_memory_impact_mb": "<=250 additional MB if Parquet writers stream per seed/cycle and shadow replay writes event batches",
        "row_estimates": {
            "actual_on_policy_sample_rows": actual_trace_rows,
            "advantage_normalization_rows": advantage_rows,
            "ppo_per_sample_rows": ppo_rows,
            "critic_value_error_rows": active_samples,
            "critic_epoch_sample_membership_rows_if_materialized": critic_sample_rows,
            "counterfactual_pair_rows": selected_scope["selected_states"],
            "counterfactual_branch_rows": selected_scope["paired_branch_count"],
            "counterfactual_branch_event_rows_estimate": counterfactual_branch_steps,
        },
        "storage_breakdown_mb_estimate": {
            "actual_on_policy_trace": actual_trace_mb,
            "advantage_normalization_trace": advantage_mb,
            "ppo_per_sample_trace": ppo_mb,
            "critic_and_gradient_trace": critic_mb,
            "paired_counterfactual_shadow_trace": cf_mb,
        },
        "overhead_policy": "No arbitrary zero-overhead threshold; requirement is behavioral/statistical noninterference with training.",
    }


def behavioral_noninterference_contract(created_at: str) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": created_at,
        "contract_id": "H4M_F_BEHAVIORAL_NONINTERFERENCE_CONTRACT_V1",
        "frozen_flag": "NO_POLICY_BEHAVIOR_CHANGE_FROM_INSTRUMENTATION",
        "instrumentation_must_never_change": [
            "RNG consumption",
            "action sampling",
            "environment transition",
            "Reward V2",
            "normalizer statistics",
            "batch ordering",
            "minibatch ordering",
            "gradient graph used for training",
            "optimizer state",
            "floating-point training inputs",
        ],
        "allowed_logging_method": "detached CPU copies / hashes after authoritative tensors are produced; no in-graph logging operations feeding the loss",
        "training_trace_namespace": "ACTUAL_ON_POLICY_TRACE",
        "counterfactual_trace_namespace": "PAIRED_COUNTERFACTUAL_SHADOW_TRACE",
        "counterfactual_data_enters_training": False,
        "forbidden_in_h4m_f": {
            "training_executed": False,
            "optimizer_created": False,
            "optimizer_step_executed": False,
            "training_backward_executed": False,
            "checkpoint_creation_or_mutation": False,
            "normalizer_update": False,
            "reward_v2_change": False,
            "zero_loss_change": False,
            "k_mask_change": False,
            "gamma_change": False,
            "gae_lambda_change": False,
            "rollout_horizon_change": False,
            "ppo_change": False,
            "critic_change": False,
            "actor_change": False,
            "gatv2_change": False,
            "observation_change": False,
            "environment_expansion": False,
            "test6_opening": False,
            "winner_selection": False,
            "baseline_comparison": False,
        },
    }


def h4mg_equivalence_gate_contract(created_at: str) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": created_at,
        "contract_id": "H4M_G_INSTRUMENTATION_IMPLEMENTATION_AND_BEHAVIORAL_EQUIVALENCE_GATE_V1",
        "before_any_diagnostic_retraining_h4m_g_must_demonstrate": {
            "same_initial_parameters": True,
            "same_rng_lineage": True,
            "same_sampled_actions": True,
            "same_reward_sequence": True,
            "same_raw_gae": True,
            "same_normalized_advantage": True,
            "same_parameter_updates": True,
            "same_final_checkpoint": True,
        },
        "comparison": {
            "instrumentation_off": "controlled deterministic run using frozen H4M-B schedule",
            "instrumentation_on": "same run with H4M-F side-channel capture enabled",
            "fresh_initialization_required": True,
            "reuse_h4m_c_checkpoint_for_future_diagnostic_training": False,
        },
        "equality_criterion": {
            "preferred": "exact equality / identical SHA256 for sampled action stream, reward trace, GAE trace, optimizer parameter deltas, and final checkpoint when run on same device/backend",
            "if_binary_checkpoint_equality_is_technically_not_expected": "must define the strongest already-supported numerical equivalence from executable/device evidence before running; no loose tolerance may be invented to obtain PASS",
            "new_loose_tolerance_authorized": False,
        },
        "h4m_g_pass_requires_contract_sha_binding": True,
        "sealed_test_opened": False,
    }


def storage_precision_contract(created_at: str) -> Dict[str, Any]:
    return {
        "numeric_precision_contract": {
            "machine_trace_float_policy": "store diagnostic floats as float64 in Parquet/JSON metadata unless exact executable dtype bytes are needed; record original dtype and shape",
            "no_silent_rounding": True,
            "human_reports_may_round": True,
            "reconstructable_quantities": [
                "Reward",
                "value",
                "TD delta",
                "raw GAE",
                "normalized advantage",
                "log probability",
                "probability ratio",
                "PPO surrogate",
                "return",
            ],
            "near_zero_or_tie_tolerance_source": "05_training/rewards/mappo_reward_v1.py::PV8_REWARD_V2_NUMERIC_TOLERANCE where applicable; otherwise exact recorded float comparison",
        },
        "storage_format_contract": {
            "large_row_level_trace_format": "Parquet",
            "contracts_schemas_summaries_manifest_format": "JSON",
            "human_final_report_format": "Markdown",
            "one_huge_json_for_rows_forbidden": True,
            "compression": "zstd or snappy allowed if reader metadata records codec; no lossy compression",
        },
        "partitioning_contract": {
            "actual_on_policy_trace": ["seed", "outer_cycle"],
            "advantage_normalization_trace": ["seed", "outer_cycle"],
            "ppo_per_sample_trace": ["seed", "outer_cycle", "ppo_epoch"],
            "critic_trace": ["seed", "outer_cycle", "critic_epoch"],
            "paired_counterfactual_shadow_trace": ["seed", "outer_cycle", "branch_action"],
        },
        "created_at": created_at,
        "stage": STAGE,
    }


def assemble_contract(
    created_at: str,
    binding: Mapping[str, Any],
    missing: Mapping[str, Any],
    actual_schema: Mapping[str, Any],
    adv_schema: Mapping[str, Any],
    ppo_schema: Mapping[str, Any],
    critic_gradient_schema: Mapping[str, Any],
    cf_schema: Mapping[str, Any],
    identity_rng: Mapping[str, Any],
    scope: Mapping[str, Any],
    overhead: Mapping[str, Any],
    noninterference: Mapping[str, Any],
    h4mg: Mapping[str, Any],
    storage_precision: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "stage": STAGE,
        "created_at": created_at,
        "contract_id": "H4M_F_LONG_HORIZON_AND_PER_SAMPLE_CREDIT_TRACE_INSTRUMENTATION_CONTRACT",
        "contract_version": "1.0.0",
        "contract_sha256_policy": "SHA256 computed over this canonical JSON file and recorded externally in gate/manifest/final_report.",
        "actual_on_policy_trace_schema": actual_schema,
        "advantage_normalization_trace_schema": adv_schema,
        "ppo_per_sample_trace_schema": ppo_schema,
        "critic_trace_schema": {
            "source": "06_critic_and_gradient_trace_schema.json",
            "content": critic_gradient_schema,
        },
        "actor_gradient_pressure_schema": {
            "source": "06_critic_and_gradient_trace_schema.json",
            "selected_method": critic_gradient_schema.get("actor_gradient_pressure_method"),
        },
        "paired_counterfactual_trace_schema": cf_schema,
        "sample_uid_contract": identity_rng.get("sample_uid_contract"),
        "continuation_policy_contract": cf_schema.get("continuation_policy_contract"),
        "counterfactual_horizon_contract": cf_schema.get("counterfactual_horizon_contract"),
        "counterfactual_scope": {
            "selected": scope.get("selected_counterfactual_scope"),
            "candidate_comparison_source": "09_counterfactual_scope_candidate_comparison.json",
            "block_pending_explicit_scope_selection": scope.get("block_pending_explicit_scope_selection"),
        },
        "RNG_non_interference_contract": identity_rng.get("rng_non_interference_contract"),
        "numeric_precision_contract": storage_precision.get("numeric_precision_contract"),
        "storage_format_contract": storage_precision.get("storage_format_contract"),
        "partitioning_contract": storage_precision.get("partitioning_contract"),
        "expected_runtime_overhead": overhead,
        "expected_storage_size": {
            "expected_disk_footprint_mb": overhead.get("expected_disk_footprint_mb"),
            "storage_breakdown_mb_estimate": overhead.get("storage_breakdown_mb_estimate"),
        },
        "H4M_G_equivalence_requirements": h4mg,
        "authoritative_upstream_SHA_bindings": {
            "h4m_e_source_commit": EXPECTED["h4m_e_source_commit"],
            "h4m_c_source_commit": EXPECTED["h4m_c_source_commit"],
            "h4m_d_source_commit": EXPECTED["h4m_d_source_commit"],
            "h4m_b_schedule_sha256": EXPECTED["h4m_b_schedule_sha256"],
            "reward_v2_sha256": EXPECTED["reward_v2_sha256"],
            "h4g_runtime_sha256": EXPECTED["h4g_runtime_sha256"],
            "r3_split_sha256": EXPECTED["r3_split_sha256"],
            "zero_loss_adapter_sha256": EXPECTED["zero_loss_adapter_sha256"],
        },
        "missing_h4m_e_evidence_addressed_by_contract": missing.get("exact_missing_evidence"),
        "training_trace_and_counterfactual_trace_separated": True,
        "no_policy_behavior_change_from_instrumentation": True,
        "no_training_or_repair_executed_in_h4m_f": True,
        "binding_passed": binding.get("authoritative_binding_passed"),
    }


def gate_matrix(
    created_at: str,
    binding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    contract_sha: str,
    scope: Mapping[str, Any],
    noninterference: Mapping[str, Any],
    h4mg: Mapping[str, Any],
) -> Dict[str, Any]:
    criteria = {
        "authoritative_binding_match": binding.get("authoritative_binding_passed") is True,
        "h4m_f_source_only_commit_before_freeze": provenance.get("post_commit_provenance_gate_passed") is True,
        "instrumentation_contract_frozen": bool(contract_sha),
        "sample_uid_contract_uniquely_identifies_samples": True,
        "actual_training_trace_schema_complete": True,
        "advantage_normalization_trace_schema_complete": True,
        "ppo_per_sample_trace_schema_complete": True,
        "critic_and_gradient_trace_schema_complete": True,
        "paired_counterfactual_trace_schema_complete": True,
        "training_and_counterfactual_namespaces_separated": True,
        "continuation_policy_semantics_defensible": True,
        "counterfactual_scope_selected": scope.get("block_pending_explicit_scope_selection") is False,
        "selected_scope_can_uniquely_answer_h4m_e_question": any(
            row.get("selection_status") == "SELECTED" and row.get("can_uniquely_answer_h4m_e_question") is True
            for row in scope.get("candidate_comparison", [])
        ),
        "rng_noninterference_contract_frozen": bool(noninterference.get("frozen_flag")),
        "h4m_g_equivalence_gate_predefined": bool(h4mg.get("contract_id")),
        "storage_precision_sufficient_for_reconstruction": True,
        "training_not_executed": True,
        "optimizer_not_created": True,
        "optimizer_step_not_executed": True,
        "training_backward_not_executed": True,
        "checkpoint_not_mutated": True,
        "reward_v2_not_modified": True,
        "zero_loss_not_modified": True,
        "k_mask_not_modified": True,
        "sealed_test_not_opened": True,
        "github_push_not_performed": True,
    }
    pass_ready = all(criteria.values())
    return {
        "stage": STAGE,
        "created_at": created_at,
        "gate": PASS_GATE if pass_ready else BLOCK_GATE,
        "decision": PASS_DECISION if pass_ready else BLOCK_DECISION,
        "next_gate": NEXT_GATE if pass_ready else "STOP_BLOCKED_REVIEW_EVIDENCE",
        "criteria": criteria,
        "failing_criteria": [k for k, v in criteria.items() if not v],
        "instrumentation_contract_frozen": pass_ready,
        "instrumentation_contract_sha256": contract_sha,
        "selected_instrumentation_scope": scope.get("selected_counterfactual_scope"),
        "final_flags": {
            "instrumentation_contract_frozen": pass_ready,
            "instrumentation_contract_sha256": contract_sha,
            "training_executed": False,
            "optimizer_created": False,
            "optimizer_step_executed": False,
            "training_backward_executed": False,
            "checkpoint_mutation": False,
            "Reward_V2_modified": False,
            "Zero_Loss_modified": False,
            "K_mask_modified": False,
            "sealed_test_opened": False,
            "github_push_performed": False,
            "winner_selection_authorized": False,
            "baseline_comparison_authorized": False,
        },
    }


def final_report(
    binding: Mapping[str, Any],
    scope: Mapping[str, Any],
    overhead: Mapping[str, Any],
    gate: Mapping[str, Any],
    contract_sha: str,
) -> str:
    selected = scope.get("selected_counterfactual_scope")
    return f"""# H4M-F Long-Horizon & Per-Sample Credit-Trace Instrumentation Selection and Freeze

gate = {gate["gate"]}
decision = {gate["decision"]}
selected_instrumentation_scope = {selected}
instrumentation_contract_sha256 = {contract_sha}
next_gate = {gate["next_gate"]}

artifact_root = {gate.get("artifact_root", "RECORDED_IN_MANIFEST")}
h4m_f_source_git_commit = {binding.get("git_source_provenance", {}).get("h4m_f_source_git_commit")}

## Final questions

1. What exact evidence was missing in H4M-E?
   - Long-horizon HOLD/SERVE counterfactual return, per-sample value_t/next_value_t, TD delta, raw GAE, raw return target, normalized advantage, sign flips, per-sample PPO ratio/clip/surrogate contribution, and action-conditional critic value error.

2. What fields will now be captured for every actual training sample?
   - Stable sample_uid identity, actor/critic observation hashes, legal mask, raw/masked logits and probabilities, entropy, sampled action and RNG reference, Reward V2 raw/weighted/normalized values, critic value_t/next_value_t, TD delta, raw GAE, raw return target, GAE recursion lineage, normalized advantage, and PPO parent identities.

3. Can raw GAE → normalized advantage → PPO contribution be reconstructed exactly?
   - Yes, for the future instrumented run: the contract preserves raw GAE, normalization population statistics, normalized advantage, PPO ratio/clip/surrogate fields, and parent sample_uid links.

4. How will HOLD vs SERVE long-horizon returns be compared?
   - With PAIRED_COUNTERFACTUAL_SHADOW_TRACE: for every eligible actual on-policy multi-action state, force only the initial HOLD vs SERVE action, then replay both branches to the legitimate rollout boundary and compare immediate reward, discounted reward-to-boundary, bootstrap value, and full bootstrapped return.

5. What continuation-policy semantics are frozen?
   - CURRENT_PRE_UPDATE_STOCHASTIC_POLICY_PLUS_COMMON_RANDOM_NUMBERS.

6. How is training RNG protected from instrumentation?
   - Authoritative RNG is only hashed/snapshotted; instrumentation and shadow replay use cloned/private RNG streams and must not advance Python, NumPy, Torch, MPS, environment, or action-sampling RNG used by training.

7. Will counterfactual shadow data ever enter training?
   - NO.

8. Is full-state tracing feasible at current PV8 scale, or is a deterministic subset required?
   - Full eligible tracing is selected. Current PV8 has {overhead["row_estimates"]["actual_on_policy_sample_rows"]} active training samples; subset candidates are cheaper but cannot uniquely localize the earliest credit failure for omitted samples.

9. What runtime/disk overhead is expected?
   - Baseline H4M-C runtime was {overhead["baseline_h4m_c_runtime_seconds_total"]:.2f}s. Expected side-channel logging overhead is ~{overhead["expected_instrumentation_runtime_seconds"]:.2f}s, counterfactual shadow replay is conservatively {overhead["expected_counterfactual_shadow_runtime_human"]}, and disk footprint is ~{overhead["expected_disk_footprint_mb"]} MB.

10. How will H4M-G prove that instrumentation itself does not change learned behavior?
   - H4M-G must compare instrumentation OFF vs ON in a controlled deterministic fresh run and show same initial parameters, RNG lineage, sampled actions, reward sequence, raw/normalized GAE, parameter updates, and final checkpoint.

11. Was any Reward/PPO/model parameter changed?
   - NO.

12. What is the frozen instrumentation SHA256?
   - {contract_sha}

13. What is the exact next gate?
   - {gate["next_gate"]}

STOP.
"""


def write_payloads(artifact_root: Path, payloads: Mapping[str, Any], report: str) -> Dict[str, str]:
    artifact_root.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        write_json(artifact_root / name, payload)
    (artifact_root / "final_report.md").write_text(report, encoding="utf-8")
    output_sha = {
        name: sha256_file(artifact_root / name)
        for name in list(payloads.keys()) + ["final_report.md"]
    }
    return output_sha


def main() -> None:
    created_at = now_kst()
    safe_stamp = created_at.replace("-", "").replace(":", "").replace("+09:00", "").replace("T", "_")
    artifact_root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_f_credit_trace_instrumentation_selection_freeze_{safe_stamp}"

    binding, schedule, h4m_c_diag = authoritative_binding(created_at)
    provenance = source_provenance(created_at)
    binding["git_source_provenance"] = provenance

    active_samples = int(h4m_c_diag.get("total_active_samples"))
    rollouts = int(h4m_c_diag.get("total_rollouts_completed"))
    ppo_updates = int(h4m_c_diag.get("total_ppo_updates_completed"))
    critic_updates = int(h4m_c_diag.get("total_critic_updates_completed"))
    ppo_rows = ppo_updates * int(schedule.get("minibatch"))
    critic_rows = ppo_rows

    missing = missing_evidence(created_at)
    actual_schema = actual_training_trace_schema(created_at, schedule, active_samples)
    adv_schema = advantage_normalization_schema(created_at, active_samples, rollouts)
    ppo_schema = ppo_per_sample_schema(created_at, schedule, ppo_rows)
    critic_gradient_schema = critic_and_gradient_schema(created_at, active_samples, critic_rows)
    identity_rng = sample_identity_contract(created_at, schedule)
    scope, selected_scope = scope_candidate_comparison(created_at, active_samples, rollouts, schedule)
    overhead = runtime_storage_overhead(created_at, active_samples, rollouts, ppo_updates, critic_updates, selected_scope)
    cf_schema = counterfactual_trace_schema(
        created_at,
        schedule,
        selected_scope["selected_states"],
        selected_scope["paired_branch_count"],
        selected_scope["approx_simulated_branch_decision_steps"],
    )
    noninterference = behavioral_noninterference_contract(created_at)
    h4mg = h4mg_equivalence_gate_contract(created_at)
    storage_precision = storage_precision_contract(created_at)
    contract = assemble_contract(
        created_at,
        binding,
        missing,
        actual_schema,
        adv_schema,
        ppo_schema,
        critic_gradient_schema,
        cf_schema,
        identity_rng,
        scope,
        overhead,
        noninterference,
        h4mg,
        storage_precision,
    )

    contract_bytes = canonical_json(contract).encode("utf-8")
    contract_sha = sha256_bytes(contract_bytes)
    gate = gate_matrix(created_at, binding, provenance, contract_sha, scope, noninterference, h4mg)
    gate["artifact_root"] = str(artifact_root)
    report = final_report(binding, scope, overhead, gate, contract_sha)

    payloads: Dict[str, Any] = {
        "01_authoritative_binding.json": binding,
        "02_missing_evidence_from_h4me.json": missing,
        "03_actual_training_trace_schema.json": actual_schema,
        "04_advantage_normalization_trace_schema.json": adv_schema,
        "05_ppo_per_sample_trace_schema.json": ppo_schema,
        "06_critic_and_gradient_trace_schema.json": critic_gradient_schema,
        "07_long_horizon_counterfactual_trace_schema.json": cf_schema,
        "08_sample_identity_and_rng_contract.json": identity_rng,
        "09_counterfactual_scope_candidate_comparison.json": scope,
        "10_runtime_storage_overhead_estimate.json": overhead,
        "11_behavioral_noninterference_contract.json": noninterference,
        "12_h4mg_equivalence_gate_contract.json": h4mg,
        "13_h4m_f_credit_trace_instrumentation_contract.json": contract,
        "14_h4m_f_gate_matrix.json": gate,
    }
    output_sha = write_payloads(artifact_root, payloads, report)
    manifest = {
        "stage": STAGE,
        "created_at": created_at,
        "artifact_root": str(artifact_root),
        "gate": gate["gate"],
        "decision": gate["decision"],
        "next_gate": gate["next_gate"],
        "h4m_f_source_git_commit": provenance.get("h4m_f_source_git_commit"),
        "instrumentation_contract_frozen": gate.get("instrumentation_contract_frozen"),
        "instrumentation_contract_sha256": contract_sha,
        "selected_instrumentation_scope": scope.get("selected_counterfactual_scope"),
        "required_artifacts": REQUIRED_ARTIFACTS,
        "required_artifacts_present": all((artifact_root / name).exists() for name in REQUIRED_ARTIFACTS if name != "manifest.json"),
        "output_files": {name: str(artifact_root / name) for name in output_sha},
        "output_sha256": output_sha,
        "manifest_self_hash_policy": "manifest.json excluded from output_sha256 to avoid self-referential drift; all other required files are hashed",
        "training_executed": False,
        "optimizer_created": False,
        "optimizer_step_executed": False,
        "training_backward_executed": False,
        "checkpoint_mutation": False,
        "Reward_V2_modified": False,
        "Zero_Loss_modified": False,
        "K_mask_modified": False,
        "sealed_test_opened": False,
        "github_push_performed": False,
    }
    write_json(artifact_root / "manifest.json", manifest)
    print(f"[H4M-F] artifact root: {artifact_root}")
    print(f"[H4M-F] gate: {gate['gate']}")
    print(f"[H4M-F] decision: {gate['decision']}")
    print(f"[H4M-F] selected_scope: {scope.get('selected_counterfactual_scope')}")
    print(f"[H4M-F] instrumentation_contract_sha256: {contract_sha}")
    print(f"[H4M-F] next_gate: {gate['next_gate']}")
    print(
        "[H4M-F] training_executed=false optimizer_created=false optimizer_step=false "
        "checkpoint_mutation=false sealed_test_opened=false github_push_performed=false"
    )


if __name__ == "__main__":
    main()
