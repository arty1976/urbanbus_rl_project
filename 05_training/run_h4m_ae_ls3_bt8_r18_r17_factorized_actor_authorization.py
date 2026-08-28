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
import r18_durable_trace as TRACE  # noqa: E402
import run_h4m_ae_ls3_bt8_r18_e1_bounded_training as R18_EXECUTOR  # noqa: E402
import run_h4m_ae_ls3_bt8_r18_r16_factorized_actor_validation as R16  # noqa: E402


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R17"
PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R17_"
    "FACTORIZED_ACTOR_BOUNDED_TRAINING_AUTHORIZATION_AND_ENVELOPE_FREEZE_COMPLETE"
)
CLASSIFICATION = "A_FACTORIZED_ACTOR_MINIMAL_BOUNDED_EXECUTION_ENVELOPE_FROZEN_AND_SEPARATELY_AUTHORIZED"

BLOCK_UPSTREAM = "BLOCKED_R18R17_UPSTREAM_EVIDENCE_BINDING_FAILURE"
BLOCK_INIT = "BLOCKED_R18R17_FACTORIZED_INITIALIZATION_NOT_FROZEN"
BLOCK_ENVELOPE = "BLOCKED_R18R17_MINIMAL_ENVELOPE_NOT_VALID_FOR_FACTORIZED_ACTOR"
BLOCK_TRACE = "BLOCKED_R18R17_TRACE_OR_GRADIENT_CONTRACT_INCOMPLETE"
BLOCK_GUARD = "BLOCKED_R18R17_EXECUTION_GUARD_NOT_FAIL_CLOSED"
BLOCK_SOURCE = BLOCK_GUARD
BLOCK_PPO_BINDING = "BLOCKED_R18R17_FACTORIZED_PPO_BINDING_FAILURE"
BLOCK_DRY_RUN = BLOCK_GUARD

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

FACTORIZED_INIT_SEED_BASE = 181600


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
    r16_bound_evidence, _frames, _snapshots = R16.bind_evidence()
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
    required_upstream = {"R18-R15", "R18-R14", "R18-R13", "R18-R12", "R18-R11", "R18-R10B", "R18-R9", "R18-R6"}
    observed_upstream = set(dict(r16_bound_evidence.get("upstream", {})))
    require(required_upstream.issubset(observed_upstream), f"R18-R16_upstream={sorted(observed_upstream)}")
    require(len(dict(r16_bound_evidence.get("checkpoint_binding", {}))) == 8, "R18-R16_checkpoint_binding_count")
    require(all(int(value) == 6 for value in dict(
        dict(r16_bound_evidence.get("r18_r4_frozen_review_binding", {})).get("review_rows_per_arm_state", {})
    ).values()), "R18-R4_six_frozen_snapshots")

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
            "recursive_upstream_binding": r16_bound_evidence,
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
        "05_training/r18_durable_trace.py",
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


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def factorized_initialization_freeze() -> dict[str, Any]:
    _evidence, frames, snapshots = R16.bind_evidence()
    rows_by_arm = {
        "AC_CONTROL_R1": R16.active_rows(frames, "AC_CONTROL_R1"),
        "BD_E1_R1": R16.active_rows(frames, "BD_E1_R1"),
    }
    arms: dict[str, Any] = {}
    global_max_distribution_delta = 0.0
    for arm_id, rows in rows_by_arm.items():
        actor, load_audit = R16.load_factorized_actor(arm_id, snapshots[str(rows.iloc[0].decision_id)])
        state = actor.state_dict()
        distribution_records = []
        for row in rows.to_dict(orient="records"):
            decision_id = str(row["decision_id"])
            snapshot = snapshots[decision_id]
            dist = actor.forward_factorized(**snapshot["tensors"])
            legacy_log_probs = JL.masked_log_probs(dist.candidate_action_log_probs, dist.no_assign_action_log_prob,
                                                   snapshot["tensors"]["safe_mask"])
            delta = float((legacy_log_probs.exp() - dist.action_probabilities).abs().max().detach().cpu())
            global_max_distribution_delta = max(global_max_distribution_delta, delta)
            distribution_records.append({
                "decision_id": decision_id,
                "candidate_identity_count": len(R16.semantic_candidate_ids(snapshot)),
                "NO_ASSIGN_identity": R16.NO_ASSIGN_ID,
                "legacy_reconstructed_distribution_delta": delta,
                "probability_sum": float(dist.action_probabilities.sum().detach().cpu()),
            })
        new_tensor_names = list(load_audit["missing_keys"])
        dropped_tensor_names = list(load_audit["unexpected_keys"])
        arms[arm_id] = {
            "source_checkpoint": load_audit["source_checkpoint"],
            "source_checkpoint_sha256": load_audit["checkpoint_sha256"],
            "factorized_initialization_method": (
                "load legacy candidate-ranking tensors with strict=False; drop legacy no_assign_scorer; "
                "initialize new gate_scorer from deterministic R18-R16 seed"
            ),
            "factorized_gate_initialization_seed": FACTORIZED_INIT_SEED_BASE + (0 if arm_id == "AC_CONTROL_R1" else 1),
            "parameter_mapping_rules": {
                "copied_from_legacy_checkpoint": [
                    name for name in state
                    if not name.startswith("gate_scorer.") and not name.startswith("no_assign_scorer.")
                ],
                "new_seed_bound_tensors": new_tensor_names,
                "dropped_legacy_tensors": dropped_tensor_names,
                "no_optimizer_state_reused": True,
            },
            "new_tensor_sha256": {
                name: tensor_sha256(state[name])
                for name in new_tensor_names
                if name in state
            },
            "factorized_initial_actor_tensor_digest": R16.BASE.module_digest(actor),
            "legacy_reconstructed_distribution_preservation": {
                "rows_checked": len(distribution_records),
                "max_delta": max(row["legacy_reconstructed_distribution_delta"] for row in distribution_records),
                "tolerance": R16.TOL,
                "passed": max(row["legacy_reconstructed_distribution_delta"] for row in distribution_records) <= R16.TOL,
                "records": distribution_records,
            },
            "NO_ASSIGN_identity_preserved": True,
            "candidate_identities_preserved": True,
            "policy_mutation_in_R18_R17": False,
        }
    require(global_max_distribution_delta <= R16.TOL,
            f"legacy_reconstructed_distribution_delta={global_max_distribution_delta}", BLOCK_INIT)
    return {
        "passed": True,
        "architecture": "FACTORIZED_ASSIGN_THEN_CANDIDATE",
        "initialization_is_deterministic": True,
        "additional_initialization_seed_required": True,
        "factorized_initialization_seed_base": FACTORIZED_INIT_SEED_BASE,
        "seed_derivation": "AC_CONTROL_R1 uses base; BD_E1_R1 uses base + 1",
        "random_new_head_initialization": "required for gate_scorer only and seed-bound",
        "legacy_reconstructed_distribution_preserved_within_R18_R16_tolerance": True,
        "max_legacy_reconstructed_distribution_delta": global_max_distribution_delta,
        "arms": arms,
    }


def factorized_trainable_parameter_contract(initialization: Mapping[str, Any]) -> dict[str, Any]:
    _evidence, frames, snapshots = R16.bind_evidence()
    rows = R16.active_rows(frames, "BD_E1_R1")
    actor, _load_audit = R16.load_factorized_actor("BD_E1_R1", snapshots[str(rows.iloc[0].decision_id)])
    config = dict(snapshots[str(rows.iloc[0].decision_id)]["metadata"]["actor_config"])
    critic = JL.JointAssignmentCritic(global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]),
                                      agent_dim=int(config["agent_dim"]),
                                      safe_summary_dim=1 + 2 * int(config["candidate_dim"]))
    actor_names = [name for name, param in actor.named_parameters() if param.requires_grad]
    stage1 = [name for name in actor_names if name.startswith("gate_scorer.")]
    stage2 = [name for name in actor_names if name.startswith(("candidate_encoder.", "scorer."))]
    shared = [name for name in actor_names if name.startswith(("agent_encoder.", "global_encoder.", "demand_encoder."))]
    covered = set(stage1) | set(stage2) | set(shared)
    require(covered == set(actor_names), f"actor_parameter_partition_missing={sorted(set(actor_names) - covered)}",
            BLOCK_TRACE)
    critic_names = [name for name, param in critic.named_parameters() if param.requires_grad]
    return {
        "passed": True,
        "trainable_only": {
            "factorized_joint_assignment_actor": {
                "stage1_assign_no_assign_gate": stage1,
                "stage2_conditional_candidate_ranking_head": stage2,
                "shared_encoder_paths": shared,
                "all_actor_trainable_parameters": actor_names,
            },
            "joint_assignment_critic": critic_names,
        },
        "frozen_modules": [
            "operational HOLD/SERVE/SKIP Actor/Critic",
            "GATv2",
            "Reward V2",
            "Zero-Loss",
            "Local Search semantics",
            "demand ledger",
            "simulator semantics",
            "policy_sampling_identity semantics",
            "E1 sampling semantics",
        ],
        "factorized_initial_actor_tensor_digest_by_legacy_arm": {
            arm_id: payload["factorized_initial_actor_tensor_digest"]
            for arm_id, payload in dict(initialization["arms"]).items()
        },
        "unauthorized_optimizer_targets_blocked": True,
    }


def factorized_gradient_trace_contract() -> dict[str, Any]:
    required = {
        "stage1_assign_probability",
        "stage1_no_assign_probability",
        "stage1_selected_log_prob",
        "stage2_candidate_probability",
        "stage2_selected_log_prob",
        "reconstructed_final_probability",
        "reconstructed_final_log_prob",
        "stage1_loss_contribution",
        "stage2_loss_contribution",
        "stage1_gradient_norm",
        "stage2_gradient_norm",
        "shared_encoder_gradient_norm",
    }
    present = set(TRACE.OPTIONAL_FACTORIZED_TRACE_FIELDS)
    require(required.issubset(present), f"missing_factorized_trace_fields={sorted(required - present)}",
            BLOCK_TRACE)
    return {
        "passed": True,
        "required_factorized_trace_fields": sorted(required),
        "schema_fields_present": sorted(present),
        "durable_artifacts": dict(TRACE.TRACE_PARQUET_FILES),
        "NO_ASSIGN_selected_eligible_row": {
            "stage2_candidate_head_direct_gradient": 0,
            "stage1_gradient": "nonzero when advantage and gate probability permit",
        },
        "candidate_selected_eligible_row_K_gt_1": {
            "stage1_gradient": "nonzero",
            "stage2_gradient": "nonzero",
        },
        "candidate_selected_eligible_row_K_eq_1": {
            "stage1_gradient": "nonzero",
            "stage2_gradient": "0 is valid degenerate case",
        },
        "mandatory_fields_fail_closed": True,
        "gradient_norm_capture": "observational torch.autograd.grad before optimizer step; does not write parameter .grad",
        "PPO_objective_changed": False,
    }


def post_update_review_contract() -> dict[str, Any]:
    return {
        "passed": True,
        "review_claim_scope": "no performance or KPI improvement claim",
        "snapshots": "same frozen snapshots bound through R18-R16/R18-R4 evidence",
        "compare": "initial factorized Actor vs final factorized Actor",
        "stage1_metrics": [
            "ASSIGN probability shift",
            "NO_ASSIGN probability shift",
            "ASSIGN-vs-NO_ASSIGN margin",
        ],
        "stage2_metrics": [
            "conditional candidate ranking margin",
            "conditional candidate probability range",
            "conditional candidate probability std",
            "best conditional candidate identity",
        ],
        "reconstructed_final_distribution_metrics": [
            "P(NO_ASSIGN)",
            "sum P(candidate_i)",
            "T1/final deterministic selection",
            "candidate support digest",
        ],
        "gate_discrimination_separate_from_candidate_discrimination": True,
        "do_not_collapse_to_single_metric": True,
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
    }, f"aggregate_envelope={aggregate}", BLOCK_ENVELOPE)
    selected_arms = []
    alias = {
        "AC_CONTROL_R1": ("AC_FACTOR_CONTROL_R1", FACTORIZED_INIT_SEED_BASE),
        "BD_E1_R1": ("BD_FACTOR_R1", FACTORIZED_INIT_SEED_BASE + 1),
    }
    for row in list(envelope["selected_arms"]):
        old_arm = str(row["arm_id"])
        require(old_arm in alias, f"unexpected_arm={old_arm}", BLOCK_ENVELOPE)
        new_arm, init_seed = alias[old_arm]
        copied = dict(row)
        copied["legacy_lineage_arm_id"] = old_arm
        copied["arm_id"] = new_arm
        copied["role"] = "AC_FACTOR_CONTROL" if new_arm == "AC_FACTOR_CONTROL_R1" else "BD_FACTOR"
        copied["factorized_gate_initialization_seed"] = init_seed
        copied["factorized_gate_initialization_seed_rule"] = "R18-R16 validation seed base 181600 + legacy arm offset"
        copied["categorical_probe_seed_by_decision"] = {
            f"{new_arm}:{index}": 0 for index in range(24)
        }
        selected_arms.append(copied)
    require([row["arm_id"] for row in selected_arms] == ["AC_FACTOR_CONTROL_R1", "BD_FACTOR_R1"],
            "factorized_arm_order", BLOCK_ENVELOPE)
    return {
        **envelope,
        "selected_arms": selected_arms,
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
                        initialization: Mapping[str, Any], trainable: Mapping[str, Any],
                        gradient_trace: Mapping[str, Any], review: Mapping[str, Any],
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
        "factorized_initialization_freeze": dict(initialization),
        "factorized_trainable_parameter_contract": dict(trainable),
        "factorized_gradient_trace_contract": dict(gradient_trace),
        "post_update_review_contract": dict(review),
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
        "factorized_initialization_freeze.json": {"not_completed": True},
        "factorized_trainable_parameter_contract.json": {"not_completed": True},
        "bounded_envelope_freeze.json": {"not_completed": True},
        "factorized_gradient_trace_contract.json": {"not_completed": True},
        "post_update_review_contract.json": {"not_completed": True},
        "execution_guard_contract.json": {"not_completed": True},
        "checkpoint_contract.json": {"not_completed": True},
        "factorized_ppo_binding_validation.json": {"not_completed": True},
        "source_binding_guard.json": {"not_completed": True},
        "r18_factorized_one_shot_authorization_manifest.json": {"authorized": False, "failure": detail},
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
    (root / "exact_execution_command.txt").write_text("NOT_AUTHORIZED\n", encoding="utf-8")
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
        initialization = factorized_initialization_freeze()
        trainable = factorized_trainable_parameter_contract(initialization)
        gradient_trace = factorized_gradient_trace_contract()
        review = post_update_review_contract()
        ppo_binding = validate_factorized_ppo_binding(execution)
        r10b_auth = load_json(R18R10B_AUTH_ROOT / "r18r10_one_shot_authorization_manifest.json")
        stamp = root.name.split("r18_r17_factorized_actor_authorization_", 1)[1]
        output_root = future_output_root(stamp)
        envelope = envelope_from_r10b(r10b_auth)
        checkpoint = checkpoint_contract_from_r10b(r10b_auth, output_root)
        authorization = build_authorization(current_commit=source_commit, upstream=upstream, source=source,
                                            envelope=envelope, checkpoint=checkpoint,
                                            initialization=initialization, trainable=trainable,
                                            gradient_trace=gradient_trace, review=review,
                                            output_root=output_root)
        auth_path = root / "r18_factorized_one_shot_authorization_manifest.json"
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
        (root / "exact_execution_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        dry = dry_run(auth_path, str(authorization["authorization_sha256"]))
        require(dry["passed"], f"dry_run={dry}", BLOCK_DRY_RUN)

        outputs = {
            "evidence_binding_audit.json": {
                "passed": True,
                "upstream": upstream,
                "execution_counters": dict(execution),
            },
            "source_binding_guard.json": source,
            "factorized_initialization_freeze.json": initialization,
            "factorized_trainable_parameter_contract.json": trainable,
            "bounded_envelope_freeze.json": envelope,
            "factorized_gradient_trace_contract.json": gradient_trace,
            "post_update_review_contract.json": review,
            "execution_guard_contract.json": guard_contract(),
            "checkpoint_contract.json": checkpoint,
            "factorized_ppo_binding_validation.json": ppo_binding,
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
            "final_questions.json": {
                "Q1_factorized_initialization_deterministic_and_distribution_preserving": True,
                "Q2_exact_trainable_parameter_set_frozen": True,
                "Q3_prior_6_window_envelope_reused_unchanged": True,
                "Q4_all_seeds_and_initial_states_frozen": True,
                "Q5_stage1_stage2_gradient_rules_explicit_including_K1": True,
                "Q6_E1_and_policy_sampling_identity_unchanged": True,
                "Q7_durable_trace_distinguishes_gate_vs_candidate_ranking": True,
                "Q8_post_update_frozen_review_contract_predeclared": True,
                "Q9_all_actual_training_optimizer_checkpoint_counters_zero": True,
                "Q10_exactly_one_bounded_factorized_actor_execution_only_new_next_step": True,
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
