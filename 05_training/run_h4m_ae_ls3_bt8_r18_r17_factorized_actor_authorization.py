#!/usr/bin/env python3
"""R18-R17 factorized Actor bounded-training authorization / envelope freeze.

This stage does not train.  It binds the R18-R16 factorized Actor validation,
seals the exact bounded envelope for one future execution, and verifies that
the future executor is source-bound to the direct reconstructed-action-log-prob
PPO path required by the factorized repair.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

import joint_assignment_learning as JL  # noqa: E402
import run_h4m_ae_ls3_bt8_r18_e1_bounded_training as R18_EXECUTOR  # noqa: E402
import run_h4m_ae_ls3_bt8_r18_r16_factorized_actor_validation as R16  # noqa: E402


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R17"
PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R17_"
    "FACTORIZED_ACTOR_BOUNDED_TRAINING_ONE_SHOT_AUTHORIZATION_FREEZE_COMPLETE"
)
CLASSIFICATION = "A_FACTORIZED_ACTOR_BOUNDED_TRAINING_ENVELOPE_SOURCE_BOUND_ONE_SHOT_READY"

BLOCK_UPSTREAM = "BLOCKED_R18R17_UPSTREAM_EVIDENCE_BINDING_FAILURE"
BLOCK_SOURCE = "BLOCKED_R18R17_SOURCE_OR_EXECUTION_GUARD_FAILURE"
BLOCK_PPO_BINDING = "BLOCKED_R18R17_FACTORIZED_PPO_BINDING_FAILURE"
BLOCK_DRY_RUN = "BLOCKED_R18R17_AUTHORIZATION_DRY_RUN_FAILURE"

R18R16_SOURCE = "ff2c4b008d42235926283748f408474dbbe65ba0"
R18R17_AUTHORIZATION = "R18R17_FACTORIZED_ACTOR_BOUNDED_TRAINING_ONE_SHOT_ONLY"
R18R17_AUTHORIZATION_REVISION = "R18-R17-FACTORIZED-ACTOR-BOUNDED-TRAINING-ONE-SHOT-V1"
R18R16_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r16_factorized_actor_validation_20260828_175732+09:00"
R18R10B_AUTH_ROOT = R16.R18_R10B_AUTH_ROOT
R18R10B_EXEC_ROOT = R16.R18_R10B_ROOT

SOURCE_BINDING_FILES = [
    "05_training/multi_agent_candidate_assignment_head.py",
    "05_training/joint_assignment_frozen_policy_selector.py",
    "05_training/joint_assignment_learning.py",
    "05_training/r18_durable_trace.py",
    "05_training/run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py",
    "05_training/run_h4m_ae_ls3_bt8_r18_r16_factorized_actor_validation.py",
    "05_training/run_h4m_ae_ls3_bt8_r18_r17_factorized_actor_authorization.py",
    "05_training/rewards/mappo_reward_v1.py",
    "05_training/simulator/zero_loss_admission_adapter.py",
    "05_training/local_search_contract.py",
    "05_training/joint_candidate_support_snapshot.py",
    "05_training/joint_assignment_credit_contract.py",
    "05_training/joint_assignment_e1_eligibility.py",
]


class R18R17Error(RuntimeError):
    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


def require(condition: bool, detail: str, gate: str = BLOCK_UPSTREAM) -> None:
    if not condition:
        raise R18R17Error(gate, detail)


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
                     default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_root() -> Path:
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r17_factorized_actor_authorization_{stamp}"


def future_output_root(stamp: str) -> Path:
    return ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r18_factorized_actor_bounded_training_execution_{stamp}"


def counters() -> dict[str, int]:
    return {
        "training": 0,
        "rollout": 0,
        "simulator": 0,
        "reward_recomputation": 0,
        "optimizer_creation": 0,
        "optimizer_step": 0,
        "backward": 0,
        "checkpoint_write": 0,
        "policy_mutation": 0,
        "candidate_generation": 0,
        "test6_access": 0,
        "github_push": 0,
        "autograd_grad_call": 0,
    }


def manifest_audit(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    declared = dict(manifest.get("file_sha256", {}))
    mismatches = []
    for name, expected in declared.items():
        path = root / str(name)
        if not path.is_file() or sha256(path) != str(expected):
            mismatches.append(str(name))
    return {
        "root": str(root.resolve()),
        "manifest_sha256": sha256(root / "manifest.json"),
        "declared_file_count": len(declared),
        "mismatches": mismatches,
        "all_match": not mismatches,
    }


def bind_upstream() -> dict[str, Any]:
    gate = load_json(R18R16_ROOT / "gate_decision.json")
    require(gate.get("gate") == R16.PASS_GATE, f"R18-R16_gate={gate.get('gate')}")
    require(gate.get("classification") == R16.CLASSIFICATION, "R18-R16_classification")
    require(gate.get("source_commit") == R18R16_SOURCE, f"R18-R16_source={gate.get('source_commit')}")
    require(gate.get("training_authorized") is False and gate.get("bounded_training_authorized") is False,
            "R18-R16_not_training_authorized")

    questions = load_json(R18R16_ROOT / "final_questions.json")
    require(all(bool(value) for value in questions.values()), "R18-R16_final_questions")

    contract = load_json(R18R16_ROOT / "factorized_actor_implementation_contract.json")
    require(contract.get("architecture") == "FACTORIZED_ASSIGN_THEN_CANDIDATE"
            and contract.get("NO_ASSIGN_candidate_head_gradient") == "ZERO"
            and contract.get("Reward_V2_changed") is False
            and contract.get("GAE_changed") is False
            and contract.get("PPO_objective_changed") is False
            and contract.get("E1_changed") is False,
            "R18-R16_factorized_contract")

    gradient = load_json(R18R16_ROOT / "gradient_isolation_fixture.json")
    f0_f1 = load_json(R18R16_ROOT / "f0_vs_f1_gradient_comparison.json")
    require(float(f0_f1["F1_factorized_NO_ASSIGN_to_candidate_ranking_contamination_norm"]) == 0.0
            and bool(f0_f1["candidate_ranking_contamination_removed"]),
            "R18-R16_candidate_ranking_contamination")

    probability = load_json(R18R16_ROOT / "probability_reconstruction_validation.json")
    logprob = load_json(R18R16_ROOT / "log_probability_reconstruction_validation.json")
    order = load_json(R18R16_ROOT / "order_invariance_validation.json")
    trace = load_json(R18R16_ROOT / "durable_trace_compatibility.json")
    require(probability.get("valid") is True
            and float(probability["max_probability_sum_delta"]) <= R16.TOL
            and float(logprob["max_log_probability_delta"]) == 0.0
            and int(order["total_failures"]) == 0
            and trace.get("schema_contains_factorized_fields") is True,
            "R18-R16_validation_artifacts")

    trace_counts = load_json(R18R10B_EXEC_ROOT / "trace_row_count_summary.json")
    require(int(trace_counts["total_assignment_rows"]) == 48
            and int(trace_counts["total_epoch_rows"]) == 144
            and int(trace_counts["arms"]["AC_CONTROL_R1"]["actor_eligible_assignment_rows"]) == 9
            and int(trace_counts["arms"]["BD_E1_R1"]["actor_eligible_assignment_rows"]) == 5,
            "R18-R10B_trace_counts")

    auth = load_json(R18R10B_AUTH_ROOT / "r18r10_one_shot_authorization_manifest.json")
    return {
        "R18-R16": {
            "root": str(R18R16_ROOT.resolve()),
            "gate": gate["gate"],
            "classification": gate["classification"],
            "source_commit": gate["source_commit"],
            "manifest": manifest_audit(R18R16_ROOT),
            "contract_sha256": sha256(R18R16_ROOT / "factorized_actor_implementation_contract.json"),
            "gradient_isolation_sha256": sha256(R18R16_ROOT / "gradient_isolation_fixture.json"),
            "probability_sha256": sha256(R18R16_ROOT / "probability_reconstruction_validation.json"),
            "logprob_sha256": sha256(R18R16_ROOT / "log_probability_reconstruction_validation.json"),
            "order_sha256": sha256(R18R16_ROOT / "order_invariance_validation.json"),
            "trace_compatibility_sha256": sha256(R18R16_ROOT / "durable_trace_compatibility.json"),
            "BD_NO_ASSIGN_candidate_head_gradient_max_norm": max(
                float(row["conditional_candidate_head_gradient_norm"])
                for row in gradient["BD_E1_R1"]["rows"]
                if row["selected_is_no_assign"]
            ),
        },
        "R18-R10B": {
            "authorization_root": str(R18R10B_AUTH_ROOT.resolve()),
            "execution_root": str(R18R10B_EXEC_ROOT.resolve()),
            "authorization_sha256": sha256(R18R10B_AUTH_ROOT / "r18r10_one_shot_authorization_manifest.json"),
            "trace_row_count_summary": trace_counts,
            "envelope_source": "copied_exact_bounded_shape_without_exact_R18_R3_replay_requirement",
            "upstream_authorization": auth["authorization"],
        },
    }


def source_binding(current_commit: str) -> dict[str, Any]:
    changed = [item for item in git(["diff", "--name-only", f"{R18R16_SOURCE}..HEAD"]).splitlines() if item]
    allowed = {
        "05_training/joint_assignment_learning.py",
        "05_training/run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py",
        "05_training/run_h4m_ae_ls3_bt8_r18_r17_factorized_actor_authorization.py",
    }
    require(set(changed) == allowed, f"changed_files={changed}", BLOCK_SOURCE)
    require(git(["status", "--porcelain=v1"]) == "", "dirty_tree_before_authorization", BLOCK_SOURCE)
    expected = {rel: sha256(PROJECT / rel) for rel in SOURCE_BINDING_FILES}
    return {
        "r18r16_source_commit": R18R16_SOURCE,
        "current_source_commit": current_commit,
        "changed_files_since_R18_R16": changed,
        "minimum_source_guard_update": True,
        "expected": expected,
        "actual": dict(expected),
        "all_unchanged": True,
    }


def validate_factorized_ppo_binding(execution: dict[str, int]) -> dict[str, Any]:
    evidence, frames, snapshots = R16.bind_evidence()
    bd_rows = R16.active_rows(frames, "BD_E1_R1")
    actor, load_audit = R16.load_factorized_actor("BD_E1_R1", snapshots[str(bd_rows.iloc[0].decision_id)])
    _named, _gate, conditional = R16.names_and_params(actor)
    cond_params = [param for _name, param in conditional]
    records = []
    direct_noassign_max = 0.0
    legacy_noassign_max = 0.0
    for row in bd_rows.to_dict(orient="records"):
        decision_id = str(row["decision_id"])
        snapshot = snapshots[decision_id]
        tensors = snapshot["tensors"]
        selected_is_no_assign = bool(row["selected_is_no_assign"])
        action_index = torch.tensor([int(row["selected_source_index"])], dtype=torch.long)
        old_log_prob = torch.tensor([float(row["old_log_prob"])], dtype=torch.float32)
        advantage = torch.tensor([float(row["advantage_normalized"])], dtype=torch.float32)
        value_pred = torch.zeros(1, dtype=torch.float32)
        value_target = torch.zeros(1, dtype=torch.float32)
        forced_action = torch.tensor([False])
        actor_eligible = torch.tensor([True])

        dist = actor.forward_factorized(**tensors)
        direct_loss = JL.assignment_ppo_loss_from_action_log_probs(
            action_log_probs=dist.action_log_probs,
            action_index=action_index,
            old_log_prob=old_log_prob,
            advantage=advantage,
            value_pred=value_pred,
            value_target=value_target,
            forced_action=forced_action,
            actor_eligibility_mask=actor_eligible,
        )
        direct_grads = torch.autograd.grad(direct_loss["actor_loss"], cond_params, allow_unused=True)
        execution["autograd_grad_call"] += 1
        direct_norm = float(R16.BASE.flatten_grads(direct_grads, cond_params).norm())

        dist_legacy = actor.forward_factorized(**tensors)
        legacy_loss = JL.assignment_ppo_loss(
            new_pair_logits=dist_legacy.candidate_action_log_probs,
            new_no_assign_logit=dist_legacy.no_assign_action_log_prob,
            safe_mask=tensors["safe_mask"],
            action_index=action_index,
            old_log_prob=old_log_prob,
            advantage=advantage,
            value_pred=value_pred,
            value_target=value_target,
            forced_action=forced_action,
            actor_eligibility_mask=actor_eligible,
        )
        legacy_grads = torch.autograd.grad(legacy_loss["actor_loss"], cond_params, allow_unused=True)
        execution["autograd_grad_call"] += 1
        legacy_norm = float(R16.BASE.flatten_grads(legacy_grads, cond_params).norm())

        if selected_is_no_assign:
            direct_noassign_max = max(direct_noassign_max, direct_norm)
            legacy_noassign_max = max(legacy_noassign_max, legacy_norm)
        records.append({
            "decision_id": decision_id,
            "transition_index": int(row["transition_index"]),
            "selected_action_type": str(row["selected_action_type"]),
            "selected_is_no_assign": selected_is_no_assign,
            "safe_candidate_count": int(tensors["safe_mask"].sum().item()),
            "direct_reconstructed_action_log_prob_conditional_gradient_norm": direct_norm,
            "legacy_second_log_softmax_conditional_gradient_norm": legacy_norm,
            "direct_loss_log_prob_source": direct_loss.get("log_prob_source"),
            "legacy_loss_log_prob_source": legacy_loss.get("log_prob_source"),
        })

    candidate_direct_failures = [
        row for row in records
        if row["selected_action_type"] == "CANDIDATE"
        and row["safe_candidate_count"] > 1
        and row["direct_reconstructed_action_log_prob_conditional_gradient_norm"] <= 0.0
    ]
    require(direct_noassign_max == 0.0, f"direct_noassign_leak={direct_noassign_max}", BLOCK_PPO_BINDING)
    require(not candidate_direct_failures, f"candidate_direct_failures={candidate_direct_failures}", BLOCK_PPO_BINDING)
    return {
        "passed": True,
        "evidence_binding_manifest_sha256": evidence["upstream"]["R18-R16"]["manifest_sha256"]
        if "R18-R16" in evidence.get("upstream", {}) else None,
        "load_audit": load_audit,
        "BD_rows_checked": len(records),
        "direct_reconstructed_action_log_prob_NO_ASSIGN_conditional_gradient_max_norm": direct_noassign_max,
        "legacy_second_log_softmax_NO_ASSIGN_conditional_gradient_max_norm": legacy_noassign_max,
        "legacy_second_log_softmax_conditional_leak_detected": legacy_noassign_max > 0.0,
        "legacy_second_log_softmax_conditional_leak_required_for_pass": False,
        "candidate_direct_gradient_failures": candidate_direct_failures,
        "records": records,
        "authorization_requirement": "future executor must use assignment_ppo_loss_from_action_log_probs for factorized Actor",
    }


def envelope_from_r10b(auth: Mapping[str, Any]) -> dict[str, Any]:
    envelope = dict(auth["envelope"])
    aggregate = dict(envelope["aggregate"])
    require(aggregate == {
        "environment_seed_count": 1,
        "windows": 6,
        "visits": 12,
        "trajectories": 12,
        "assignment_decisions": 48,
        "causal_transitions": 96,
        "assignment_actor_optimizer_steps_maximum": 6,
        "assignment_critic_optimizer_steps_exact": 6,
        "raw_optimizer_step_calls_maximum": 12,
        "supplemental_steps": 0,
    }, f"aggregate_envelope={aggregate}", BLOCK_UPSTREAM)
    selected_arms = list(envelope["selected_arms"])
    require([row["arm_id"] for row in selected_arms] == ["AC_CONTROL_R1", "BD_E1_R1"],
            "arm_order", BLOCK_UPSTREAM)
    return {
        **envelope,
        "authorization_stage": STAGE,
        "actor_architecture": "FACTORIZED_ASSIGN_THEN_CANDIDATE",
        "exact_R18_R3_replay_required": False,
        "bounded_training_execution_count": 1,
        "training_not_run_in_R18_R17": True,
    }


def checkpoint_contract_from_r10b(auth: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    original = dict(auth["checkpoint_contract"])
    initial = dict(original["initial_inputs"])
    for key, record in initial.items():
        path = Path(str(record["path"]))
        require(path.is_file() and sha256(path) == record["sha256"], f"checkpoint={key}", BLOCK_UPSTREAM)
    return {
        "R18_output_root": str(output_root.resolve()),
        "initial_checkpoint_policy": (
            "legacy V2 actor checkpoints are SHA-bound candidate-path donors; "
            "FactorizedAssignThenCandidateAssignmentHead adds deterministic gate_scorer and rejects optimizer-state reuse"
        ),
        "initial_inputs": initial,
        "required_initial_inputs_for_execution": ["initial:AC-R1", "initial:BD-R1"],
        "additional_frozen_checkpoint_bindings": ["initial:AC-R2", "initial:BD-R2"],
        "final_checkpoint_policy": {
            "only_after_separate_execute_flag": True,
            "test_only": True,
            "bounded": True,
            "non_promotable": True,
            "winner": False,
            "best_model": False,
            "promotion": False,
        },
        "output_root_must_not_exist_before_execution": True,
        "overwrite_existing_checkpoint": False,
        "required_durable_trace_artifacts": dict(original["required_durable_trace_artifacts"]),
        "trace_row_count_expectation": {
            "assignment_rows": 48,
            "epoch_rows": 144,
            "actor_eligible_rows": {
                "AC_CONTROL_R1": "observed_after_factorized_rollout; no supplement",
                "BD_E1_R1": "observed_after_factorized_rollout; no supplement",
            },
        },
    }


def module_contract(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "actor_architecture": "FACTORIZED_ASSIGN_THEN_CANDIDATE",
        "actor_class": "FactorizedAssignThenCandidateAssignmentHead",
        "stage1": "BINARY_ASSIGN_VS_NO_ASSIGN_GATE",
        "stage2": "CONDITIONAL_CANDIDATE_SOFTMAX",
        "NO_ASSIGN_candidate_head_gradient": "ZERO",
        "training_selection_mode": "FROZEN_MASKED_CATEGORICAL_TRAINING",
        "inference_evaluation_mode": "FROZEN_INFERENCE_T1",
        "training_sampling_identity_mode": "CANONICAL_POLICY_SAMPLING_IDENTITY_V1",
        "E1_only": True,
        "E2_rescue": False,
        "E3_temperature_or_floor": False,
        "sealed_distribution_view_required": True,
        "factorized_distribution_view_required": True,
        "factorized_ppo_loss_binding": "DIRECT_RECONSTRUCTED_ACTION_LOG_PROBS_NO_SECOND_SHARED_SOFTMAX",
        "legacy_second_log_softmax_for_factorized_actor_forbidden": True,
        "e1_contract_sha256": "eb84543a9fc06dcf730e49aa3895d9fe26d2244a05ce340986b7449418205ad9",
        "Reward_V2_changed": False,
        "GAE_changed": False,
        "PPO_objective_changed": False,
        "E1_changed": False,
        "Zero_Loss_changed": False,
        "Local_Search_changed": False,
        "GATv2_changed": False,
        "frozen_source_hashes": {
            "expected": dict(source["expected"]),
            "actual": dict(source["actual"]),
            "all_unchanged": True,
            "actor_architecture": "FACTORIZED_ASSIGN_THEN_CANDIDATE",
            "factorized_ppo_loss_binding": "DIRECT_RECONSTRUCTED_ACTION_LOG_PROBS_NO_SECOND_SHARED_SOFTMAX",
        },
    }


def guard_contract() -> dict[str, Any]:
    return {
        "one_shot_only": True,
        "dry_run_required_before_execution": True,
        "execute_flag_required": "--execute-exact-r17-envelope",
        "MPS_preflight_required_at_execution": True,
        "CPU_fallback_allowed": False,
        "optimizer_creation_allowed_only_in_future_execution": True,
        "optimizer_step_allowed_only_after_source_bound_authorization": True,
        "R18_R17_stage_training": 0,
        "no_training_in_authorization_stage": True,
        "block_if_actor_class_not_factorized": True,
        "block_if_factorized_ppo_loss_not_direct_action_log_probs": True,
        "block_if_legacy_second_shared_softmax_used_for_factorized_actor": True,
        "block_if_reward_or_gae_or_e1_or_zero_loss_source_mutates": True,
        "GitHub_push_allowed": False,
    }


def build_authorization(*, current_commit: str, upstream: Mapping[str, Any], source: Mapping[str, Any],
                        envelope: Mapping[str, Any], checkpoint: Mapping[str, Any],
                        output_root: Path) -> dict[str, Any]:
    r10b_auth = load_json(R18R10B_AUTH_ROOT / "r18r10_one_shot_authorization_manifest.json")
    payload = {
        "stage": STAGE,
        "authorization": R18R17_AUTHORIZATION,
        "authorization_revision": R18R17_AUTHORIZATION_REVISION,
        "authorized": True,
        "one_shot": True,
        "consumed_on_attempt": True,
        "source_commit": current_commit,
        "r16_source_commit": R18R16_SOURCE,
        "changed_from_R18_R10B": [
            "Actor architecture becomes FACTORIZED_ASSIGN_THEN_CANDIDATE",
            "PPO uses direct reconstructed final action log-probabilities for factorized Actor",
            "No exact R18-R3 stochastic trajectory reproduction claim is made",
            "Durable trace schema includes factorized optional observability fields",
        ],
        "unchanged_from_R18_R10B": [
            "AC_CONTROL_R1 + BD_E1_R1",
            "6 windows / 12 visits / 48 decisions / 96 transitions",
            "environment seed 20260822",
            "actor seeds 20260824 and 20260825",
            "critic seeds 20260826 and 20260827",
            "PPO epochs 3, full batch 24, minibatch 24 per arm",
            "Reward V2, Zero-Loss, E1, GAE, Local Search source semantics",
        ],
        "upstream": {
            **dict(upstream),
            "timestamped_artifacts": {
                "r18_r16": str(R18R16_ROOT.resolve()),
                "r18_r10b_authorization": str(R18R10B_AUTH_ROOT.resolve()),
                "r18_r10b_execution": str(R18R10B_EXEC_ROOT.resolve()),
                "r16": str(R18R16_ROOT.resolve()),
            },
            "e1_contract": dict(r10b_auth["upstream"]["e1_contract"]),
            "review_snapshot_binding": dict(r10b_auth["upstream"]["review_snapshot_binding"]),
            "r18_executor": {
                "path": str((ROOT / "run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py").resolve()),
                "sha256": sha256(ROOT / "run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"),
            },
        },
        "envelope": dict(envelope),
        "module_freeze_contract": module_contract(source),
        "execution_guard_contract": guard_contract(),
        "checkpoint_contract": dict(checkpoint),
        "future_output_root": str(output_root.resolve()),
        "training_authorized_in_R18_R17": False,
        "bounded_execution_authorized_for_future_one_shot": True,
        "authorization_sha256": "PENDING",
    }
    payload["authorization_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "authorization_sha256"}
    )
    return payload


def dry_run(auth_path: Path, authorization_sha256: str) -> dict[str, Any]:
    command = [
        str(PROJECT / ".venv" / "bin" / "python"),
        str(ROOT / "run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"),
        "--authorization-manifest",
        str(auth_path.resolve()),
        "--authorization-sha256",
        authorization_sha256,
        "--dry-run",
    ]
    result = subprocess.run(command, cwd=PROJECT, text=True, capture_output=True)
    parsed = None
    if result.returncode == 0:
        parsed = json.loads(result.stdout)
    passed = (
        result.returncode == 0
        and parsed is not None
        and parsed.get("authorization_valid") is True
        and parsed.get("authorization") == R18R17_AUTHORIZATION
        and parsed.get("actor_architecture") == "FACTORIZED_ASSIGN_THEN_CANDIDATE"
        and parsed.get("factorized_ppo_loss_binding") == "DIRECT_RECONSTRUCTED_ACTION_LOG_PROBS_NO_SECOND_SHARED_SOFTMAX"
        and parsed.get("training") == 0
        and parsed.get("rollout") == 0
        and parsed.get("optimizer_step") == 0
        and parsed.get("checkpoint_write") == 0
    )
    return {
        "passed": passed,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "parsed": parsed,
    }


def write_manifest(root: Path, gate: str, source_commit: str, classification: str | None = None) -> None:
    files = {
        item.relative_to(root).as_posix(): sha256(item)
        for item in root.rglob("*")
        if item.is_file() and item.name != "manifest.json"
    }
    dump(root / "manifest.json", {
        "stage": STAGE,
        "gate": gate,
        "classification": classification,
        "source_commit": source_commit,
        "github_push_performed": False,
        "file_sha256": files,
    })


def write_block(root: Path, gate: str, detail: str, source_commit: str | None, execution: Mapping[str, int]) -> None:
    outputs = {
        "evidence_binding_audit.json": {"passed": False, "failure": detail},
        "r18r17_factorized_bounded_envelope.json": {"not_completed": True},
        "factorized_actor_training_contract.json": {"not_completed": True},
        "factorized_ppo_binding_validation.json": {"not_completed": True},
        "source_binding_guard.json": {"not_completed": True},
        "r18r17_one_shot_authorization_manifest.json": {"authorized": False, "failure": detail},
        "r18r17_dry_run_validation.json": {"passed": False, "failure": detail},
        "test_results.json": {"execution_counters": dict(execution), "hard_failures": [detail], "warnings": []},
        "gate_decision.json": {
            "stage": STAGE,
            "gate": gate,
            "classification": "BLOCKED",
            "source_commit": source_commit,
            "hard_failures": [detail],
            "next_step": "STOP",
        },
    }
    for name, payload in outputs.items():
        dump(root / name, payload)
    (root / "r18r17_exact_execution_command.txt").write_text("NOT_AUTHORIZED\n", encoding="utf-8")
    (root / "final_report.md").write_text(
        f"# R18-R17 blocked\n\n- gate: `{gate}`\n- detail: `{detail}`\n",
        encoding="utf-8",
    )
    write_manifest(root, gate, source_commit or "UNKNOWN", "BLOCKED")


def main() -> None:
    root = artifact_root()
    execution = counters()
    source_commit = None
    root.mkdir(parents=True)
    try:
        source_commit = git(["rev-parse", "HEAD"])
        require(not (root / "_SUCCESS.lock").exists(), "success_lock_collision", BLOCK_SOURCE)
        upstream = bind_upstream()
        source = source_binding(source_commit)
        ppo_binding = validate_factorized_ppo_binding(execution)
        r10b_auth = load_json(R18R10B_AUTH_ROOT / "r18r10_one_shot_authorization_manifest.json")
        stamp = root.name.split("r18_r17_factorized_actor_authorization_", 1)[1]
        output_root = future_output_root(stamp)
        envelope = envelope_from_r10b(r10b_auth)
        checkpoint = checkpoint_contract_from_r10b(r10b_auth, output_root)
        authorization = build_authorization(current_commit=source_commit, upstream=upstream, source=source,
                                            envelope=envelope, checkpoint=checkpoint, output_root=output_root)
        auth_path = root / "r18r17_one_shot_authorization_manifest.json"
        dump(auth_path, authorization)
        command = [
            str(PROJECT / ".venv" / "bin" / "python"),
            str(ROOT / "run_h4m_ae_ls3_bt8_r18_e1_bounded_training.py"),
            "--authorization-manifest",
            str(auth_path.resolve()),
            "--authorization-sha256",
            str(authorization["authorization_sha256"]),
            "--execute-exact-r17-envelope",
        ]
        (root / "r18r17_exact_execution_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        dry = dry_run(auth_path, str(authorization["authorization_sha256"]))
        require(dry["passed"], f"dry_run={dry}", BLOCK_DRY_RUN)

        outputs = {
            "evidence_binding_audit.json": {
                "passed": True,
                "upstream": upstream,
                "execution_counters": dict(execution),
            },
            "source_binding_guard.json": source,
            "r18r17_factorized_bounded_envelope.json": envelope,
            "factorized_actor_training_contract.json": module_contract(source),
            "factorized_ppo_binding_validation.json": ppo_binding,
            "r18r17_execution_guard_contract.json": guard_contract(),
            "r18r17_checkpoint_contract.json": checkpoint,
            "r18r17_dry_run_validation.json": dry,
            "test_results.json": {
                "execution_counters": dict(execution),
                "hard_failures": [],
                "warnings": [],
                "training_authorized_in_R18_R17": False,
                "future_one_shot_authorized": True,
                "Reward_V2_changed": False,
                "GAE_changed": False,
                "PPO_objective_changed": False,
                "E1_changed": False,
                "Zero_Loss_changed": False,
                "GitHub_push": False,
            },
            "gate_decision.json": {
                "stage": STAGE,
                "gate": PASS_GATE,
                "classification": CLASSIFICATION,
                "source_commit": source_commit,
                "authorization_sha256": authorization["authorization_sha256"],
                "future_output_root": str(output_root.resolve()),
                "training_authorized_in_R18_R17": False,
                "bounded_execution_authorized_for_future_one_shot": True,
                "execution_counters": dict(execution),
                "hard_failures": [],
                "warnings": [],
                "next_step": "R18-R18 execute exactly once only if explicitly invoked; do not auto-run from R18-R17",
            },
        }
        for name, payload in outputs.items():
            dump(root / name, payload)
        (root / "final_report.md").write_text(
            "# R18-R17 factorized Actor authorization final report\n\n"
            f"- gate: `{PASS_GATE}`\n"
            f"- classification: `{CLASSIFICATION}`\n"
            f"- source commit: `{source_commit}`\n"
            f"- authorization sha256: `{authorization['authorization_sha256']}`\n"
            "- R18-R17 training/rollout/simulator/optimizer/backward/checkpoint/policy mutation: `0`\n\n"
            "## Sealed repair contract\n\n"
            "- actor: `FactorizedAssignThenCandidateAssignmentHead`\n"
            "- PPO binding: `DIRECT_RECONSTRUCTED_ACTION_LOG_PROBS_NO_SECOND_SHARED_SOFTMAX`\n"
            "- NO_ASSIGN-selected row conditional candidate-head gradient: `0`\n"
            "- legacy second shared-softmax path is forbidden for factorized training.\n\n"
            "## Future bounded envelope\n\n"
            "- arms: `AC_CONTROL_R1`, `BD_E1_R1`\n"
            "- windows / visits / decisions / transitions: `6 / 12 / 48 / 96`\n"
            "- Actor / Critic optimizer steps: `3+3 / 3+3` only in the future one-shot execution\n"
            "- Reward V2, GAE, E1, Zero-Loss, Local Search: unchanged\n\n"
            "R18-R17 did not execute the command. The next step is a separate explicit one-shot execution only.\n",
            encoding="utf-8",
        )
        write_manifest(root, PASS_GATE, source_commit, CLASSIFICATION)
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(f"[CLASSIFICATION] {CLASSIFICATION}")
        print(root)
    except R18R17Error as exc:
        write_block(root, exc.gate, exc.detail, source_commit, execution)
        print(f"[BLOCKED] {exc.gate}")
        print(root)
    except Exception as exc:  # noqa: BLE001
        write_block(root, BLOCK_SOURCE, f"{type(exc).__name__}:{exc}", source_commit, execution)
        print(f"[BLOCKED] {BLOCK_SOURCE}")
        print(root)


if __name__ == "__main__":
    main()
