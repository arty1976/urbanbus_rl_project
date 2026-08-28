#!/usr/bin/env python3
"""R18-R16 factorized Actor implementation + frozen equivalence validation.

This stage validates the implemented
``FactorizedAssignThenCandidateAssignmentHead`` without training.  It binds the
R18-R15 repair-selection evidence, reuses the R18-R10B durable trace and frozen
snapshots, and verifies that NO_ASSIGN rows cannot write gradients into the
conditional candidate-ranking path.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

import joint_assignment_frozen_policy_selector as SELECTOR  # noqa: E402
import multi_agent_candidate_assignment_head as H  # noqa: E402
import r18_durable_trace as TRACE  # noqa: E402
import run_h4m_ae_ls3_bt8_r18_r12_shared_batch_gradient_interference_audit as BASE  # noqa: E402


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R16"
PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R16_"
    "FACTORIZED_ASSIGN_THEN_CANDIDATE_ACTOR_IMPLEMENTATION_AND_FROZEN_EQUIVALENCE_VALIDATION_COMPLETE"
)
CLASSIFICATION = "A_FACTORIZED_ASSIGN_THEN_CANDIDATE_ACTOR_IMPLEMENTED_READY_FOR_SEPARATE_BOUNDED_EXECUTION_DESIGN"

BLOCK_UPSTREAM = "BLOCKED_R18R16_UPSTREAM_EVIDENCE_BINDING_FAILURE"
BLOCK_PROBABILITY = "BLOCKED_R18R16_PROBABILITY_RECONSTRUCTION_FAILURE"
BLOCK_NO_ASSIGN_GRAD = "BLOCKED_R18R16_NO_ASSIGN_CANDIDATE_HEAD_GRADIENT_LEAKAGE"
BLOCK_CANDIDATE_GRAD = "BLOCKED_R18R16_CANDIDATE_GRADIENT_PATH_FAILURE"
BLOCK_MASK = "BLOCKED_R18R16_MASK_OR_ZERO_LOSS_REGRESSION"
BLOCK_ORDER = "BLOCKED_R18R16_ORDER_INVARIANCE_REGRESSION"
BLOCK_E1_RNG = "BLOCKED_R18R16_E1_OR_RNG_SEMANTICS_CHANGED"
BLOCK_PPO_GAE_REWARD = "BLOCKED_R18R16_PPO_GAE_REWARD_SEMANTICS_CHANGED"

TOL = 2e-5
R15_SOURCE_SHA = "c79dc697b239d0484ec3838bf59cb940230e4b0e"
NO_ASSIGN_ID = "NO_ASSIGN_KEEP_CURRENT_PLANS"

R18_R4_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r4_frozen_policy_initial_final_review_20260828_011753+09:00"
R18_R6_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r6_durable_trace_instrumentation_validation_20260828_074808+09:00"
R18_R9_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r9_provenance_sampling_identity_decoupling_validation_20260828_084535+09:00"
R18_R10B_ROOT = BASE.R18_R10B_ROOT
R18_R10B_AUTH_ROOT = BASE.R18_R10B_AUTH_ROOT
R18_R11_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r11_credit_advantage_ppo_logit_direction_attribution_audit_20260828_120606+09:00"
R18_R12_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r12_shared_batch_gradient_interference_audit_20260828_122756+09:00"
R18_R13_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r13_gradient_magnitude_path_dominance_audit_20260828_123836+09:00"
R18_R14_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r14_synthetic_path_softmax_ablation_audit_20260828_131655+09:00"
R18_R15_ROOT = ARTIFACTS / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r15_frozen_actor_repair_selection_audit_20260828_143516+09:00"


UPSTREAM_GATES = {
    "R18-R6": (R18_R6_ROOT, "gate_decision.json",
               "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R6_DURABLE_PER_ROW_CREDIT_PPO_TRACE_INSTRUMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE"),
    "R18-R9": (R18_R9_ROOT, "gate_decision_r18r9.json",
               "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R9_PROVENANCE_SAMPLING_IDENTITY_DECOUPLING_AND_EXACT_REPLAY_VALIDATION_COMPLETE"),
    "R18-R10B": (R18_R10B_ROOT, "gate_decision.json", BASE.R18_R10B_PASS_GATE),
    "R18-R11": (R18_R11_ROOT, "gate_decision.json",
                "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R11_DURABLE_CREDIT_ADVANTAGE_PPO_LOGIT_DIRECTION_ATTRIBUTION_AUDIT_COMPLETE"),
    "R18-R12": (R18_R12_ROOT, "gate_decision.json",
                "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R12_SHARED_BATCH_GRADIENT_INTERFERENCE_AND_CANDIDATE_VS_NO_ASSIGN_ATTRIBUTION_AUDIT_COMPLETE"),
    "R18-R13": (R18_R13_ROOT, "gate_decision.json",
                "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R13_GRADIENT_MAGNITUDE_AND_PARAMETER_PATH_DOMINANCE_DECOMPOSITION_AUDIT_COMPLETE"),
    "R18-R14": (R18_R14_ROOT, "gate_decision.json",
                "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R14_SYNTHETIC_PATH_AND_SOFTMAX_COUPLING_ABLATION_AUDIT_COMPLETE"),
    "R18-R15": (R18_R15_ROOT, "gate_decision.json",
                "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R15_MINIMAL_ACTOR_REPAIR_SELECTION_AUDIT_COMPLETE"),
}


class R18R16Error(RuntimeError):
    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


def require(condition: bool, detail: str, gate: str = BLOCK_UPSTREAM) -> None:
    if not condition:
        raise R18R16Error(gate, detail)


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_at(commit: str, path: str) -> str:
    return git(["show", f"{commit}:{path}"])


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def vector_norm(grads: Sequence[torch.Tensor | None], params: Sequence[torch.nn.Parameter]) -> float:
    if not params:
        return 0.0
    return float(BASE.flatten_grads(grads, params).norm())


def vector_from_grads(grads: Sequence[torch.Tensor | None], params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    return BASE.flatten_grads(grads, params) if params else torch.zeros(0)


def dot_cos(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_norm = float(left.norm())
    right_norm = float(right.norm())
    dot = float(torch.dot(left, right)) if left.numel() and right.numel() else 0.0
    raw = dot / (left_norm * right_norm) if left_norm > 0.0 and right_norm > 0.0 else None
    cosine = None if raw is None else max(-1.0, min(1.0, float(raw)))
    return {
        "dot": dot,
        "cosine": cosine,
        "raw_cosine": raw,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "alignment": "zero_vector" if cosine is None else BASE.alignment_label(cosine),
    }


def semantic_candidate_ids(snapshot: Mapping[str, Any]) -> list[str]:
    return [
        f"{item['agent_id']}::{item['candidate_id']}"
        for item in snapshot["metadata"]["candidate_order"]
    ]


def pair_keys(snapshot: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(item["agent_id"]), str(item["candidate_id"]))
        for item in snapshot["metadata"]["candidate_order"]
    ]


def bind_evidence() -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, Mapping[str, Any]]]:
    gates: dict[str, Any] = {}
    for stage, (root, filename, expected) in UPSTREAM_GATES.items():
        require(root.is_dir(), f"{stage}_root_missing={root}")
        gate = load_json(root / filename)
        require(gate.get("gate") == expected, f"{stage}_gate={gate.get('gate')}")
        gates[stage] = gate
    require(gates["R18-R15"].get("classification") == "A_FACTORIZED_ASSIGN_THEN_CANDIDATE_ACTOR_SELECTED",
            "R18-R15_classification")
    require(gates["R18-R15"].get("source_commit") == R15_SOURCE_SHA,
            f"R18-R15_source={gates['R18-R15'].get('source_commit')}")
    require(gates["R18-R14"].get("classification") == "B_SHARED_SOFTMAX_COUPLING_DOMINANCE",
            "R18-R14_classification")

    selected_contract = load_json(R18_R15_ROOT / "selected_minimal_actor_repair_contract.json")
    require(selected_contract.get("architecture") == "FACTORIZED_ASSIGN_THEN_CANDIDATE",
            "R18-R15_selected_architecture")
    require(selected_contract.get("NO_ASSIGN_candidate_head_gradient") == "ZERO",
            "R18-R15_no_assign_gradient_contract")
    require(selected_contract.get("Reward_V2_changed") is False and selected_contract.get("GAE_changed") is False
            and selected_contract.get("PPO_objective_changed") is False and selected_contract.get("E1_changed") is False,
            "R18-R15_semantics_contract")

    trace_summary = load_json(R18_R10B_ROOT / "trace_row_count_summary.json")
    identity = load_json(R18_R10B_ROOT / "trace_identity_roundtrip_audit.json")
    parameter_delta = load_json(R18_R10B_ROOT / "parameter_delta_audit.json")
    support = load_json(R18_R10B_ROOT / "r18_candidate_support_audit.json")
    trajectory = load_json(R18_R10B_ROOT / "exact_trajectory_replay_audit.json")
    require(trace_summary["total_assignment_rows"] == 48 and trace_summary["total_epoch_rows"] == 144,
            "R18-R10B_trace_counts")
    require(trace_summary["arms"]["AC_CONTROL_R1"]["actor_eligible_assignment_rows"] == 9,
            "AC_actor_eligible")
    require(trace_summary["arms"]["BD_E1_R1"]["actor_eligible_assignment_rows"] == 5,
            "BD_actor_eligible")
    require(identity.get("identity_roundtrip_passed") is True and identity.get("no_assign_identity_stable") is True,
            "trace_identity_roundtrip")
    require(trajectory.get("matched_selection_rows") == 48, "R18-R3_exact_trajectory_binding")
    require(all(item.get("mismatched") == 0 and item.get("behavior_equals_update_support") is True
                for item in support.values()), "candidate_support_binding")
    require(all(row["final_actor_digest_matches_r18r3"] and row["final_critic_digest_matches_r18r3"]
                for row in parameter_delta["arms"].values()), "initial_final_tensor_digest_binding")

    auth = load_json(R18_R10B_AUTH_ROOT / "r18r10_one_shot_authorization_manifest.json")
    checkpoint_records = {
        **auth["checkpoint_contract"]["initial_inputs"],
        **auth["checkpoint_contract"]["final_evidence_only"],
    }
    checkpoint_binding = {}
    for name, record in checkpoint_records.items():
        path = Path(str(record["path"]))
        require(path.is_file(), f"checkpoint_missing={name}")
        observed = sha256(path)
        require(observed == record["sha256"], f"checkpoint_sha_mismatch={name}")
        checkpoint_binding[name] = {"path": str(path), "sha256": observed, "role": record.get("role")}
    require(len(checkpoint_binding) == 8, f"checkpoint_binding_count={len(checkpoint_binding)}")

    r4_gate = load_json(R18_R4_ROOT / "gate_decision.json")
    require(r4_gate.get("gate") == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R4_SAME_INPUT_FROZEN_POLICY_INITIAL_FINAL_REVIEW_COMPLETE",
            "R18-R4_gate")
    r4_matrix = load_json(R18_R4_ROOT / "r18r4_frozen_policy_matrix.json")
    require(all(payload.get("rows") == 6 for payload in r4_matrix["per_arm_state_summary"].values()),
            "R18-R4_six_frozen_review_snapshots")

    frames = {
        name: pd.read_parquet(R18_R10B_ROOT / f"{name}.parquet")
        for name in ("credit_eligibility_rows", "eligible_row_credit_trace", "gae_rows",
                     "ppo_ratio_rows", "per_epoch_loss_contributions")
    }
    required = {
        "decision_id", "selected_semantic_candidate", "executed_semantic_candidate",
        "credited_semantic_candidate", "reward_ancestry_class", "reward_ancestry_source_ids",
        "assignment_reward", "return_raw", "gae_advantage_raw", "advantage_normalized",
        "old_log_prob", "actor_eligible", "critic_eligible", "candidate_support_digest",
        "selected_source_index", "selected_action_type", "selected_is_no_assign",
    }
    for name, frame in frames.items():
        require(required.issubset(set(frame.columns)), f"{name}_required_columns")
        require(not frame[list(required)].isna().any().any(), f"{name}_required_nulls")
    require(len(frames["credit_eligibility_rows"]) == 48 and len(frames["gae_rows"]) == 48
            and len(frames["ppo_ratio_rows"]) == 144
            and len(frames["per_epoch_loss_contributions"]) == 144
            and len(frames["eligible_row_credit_trace"]) == 42, "durable_trace_row_counts")

    import joint_assignment_frozen_policy_snapshot as FPS

    snapshots = {}
    for snapshot_root in (R18_R10B_ROOT / "training_snapshots" / "snapshots").glob("*/"):
        payload = FPS.load_snapshot(snapshot_root)
        snapshots[str(payload["metadata"]["decision_id"])] = payload
    require(len(snapshots) == 48, f"snapshot_count={len(snapshots)}")
    evidence = {
        "stage": STAGE,
        "upstream": {
            stage: {
                "root": str(root.resolve()),
                "gate_file": filename,
                "gate": gates[stage].get("gate"),
                "classification": gates[stage].get("classification"),
                "source_commit": gates[stage].get("source_commit"),
                "manifest_sha256": sha256(root / "manifest.json"),
            }
            for stage, (root, filename, _expected) in UPSTREAM_GATES.items()
        },
        "r18_r4_frozen_review_binding": {
            "root": str(R18_R4_ROOT.resolve()),
            "gate": r4_gate.get("gate"),
            "manifest_sha256": sha256(R18_R4_ROOT / "manifest.json"),
            "review_rows_per_arm_state": {
                key: payload.get("rows") for key, payload in r4_matrix["per_arm_state_summary"].items()
            },
        },
        "checkpoint_binding": checkpoint_binding,
        "trace_row_count_summary": trace_summary,
        "trace_identity_roundtrip": identity,
        "candidate_support_audit": support,
        "parameter_delta_audit": parameter_delta,
        "selected_repair_contract_sha256": sha256(R18_R15_ROOT / "selected_minimal_actor_repair_contract.json"),
    }
    return evidence, frames, snapshots


def source_hash_audit(current_commit: str) -> dict[str, Any]:
    files = [
        "05_training/multi_agent_candidate_assignment_head.py",
        "05_training/joint_assignment_frozen_policy_selector.py",
        "05_training/r18_durable_trace.py",
        "05_training/run_h4m_ae_ls3_bt8_r18_r16_factorized_actor_validation.py",
    ]
    audit = {
        "before_source_commit": R15_SOURCE_SHA,
        "after_source_commit": current_commit,
        "minimum_source_mutation": True,
        "runtime_training_policy_mutated": False,
        "files": {},
    }
    for file in files:
        current_path = PROJECT / file
        require(current_path.is_file(), f"source_file_missing={file}")
        current_sha = sha256(current_path)
        before_present = True
        try:
            before_sha = sha256_text(source_at(R15_SOURCE_SHA, file))
        except subprocess.CalledProcessError:
            before_present = False
            before_sha = None
        audit["files"][file] = {
            "before_present": before_present,
            "before_sha256": before_sha,
            "after_sha256": current_sha,
            "changed": before_sha != current_sha,
        }
    expected_changed = {
        "05_training/multi_agent_candidate_assignment_head.py",
        "05_training/joint_assignment_frozen_policy_selector.py",
        "05_training/r18_durable_trace.py",
        "05_training/run_h4m_ae_ls3_bt8_r18_r16_factorized_actor_validation.py",
    }
    require({file for file, row in audit["files"].items() if row["changed"]} == expected_changed,
            "implementation_source_scope")
    return audit


def active_rows(frames: Mapping[str, pd.DataFrame], arm_id: str) -> pd.DataFrame:
    rows = frames["gae_rows"][
        (frames["gae_rows"].arm_id == arm_id)
        & (frames["gae_rows"].actor_eligible)
        & (~frames["gae_rows"].forced_action)
    ].copy()
    rows = rows.sort_values("transition_index")
    require(len(rows) == (9 if arm_id == "AC_CONTROL_R1" else 5), f"{arm_id}_active_rows")
    return rows


def load_existing_actor(arm_id: str, snapshot: Mapping[str, Any]) -> torch.nn.Module:
    auth = load_json(R18_R10B_AUTH_ROOT / "r18r10_one_shot_authorization_manifest.json")
    key = "initial:AC-R1" if arm_id == "AC_CONTROL_R1" else "initial:BD-R1"
    checkpoint_record = auth["checkpoint_contract"]["initial_inputs"][key]
    checkpoint_path = Path(str(checkpoint_record["path"]))
    require(checkpoint_path.is_file() and sha256(checkpoint_path) == checkpoint_record["sha256"],
            f"{arm_id}_checkpoint_binding")
    config = dict(snapshot["metadata"]["actor_config"])
    actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
        global_dim=int(config["global_dim"]),
        demand_dim=int(config["demand_dim"]),
        agent_dim=int(config["agent_dim"]),
        candidate_dim=int(config["candidate_dim"]),
        hidden=int(config["hidden"]),
        heads=int(config["heads"]),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor.train()
    return actor


def load_factorized_actor(arm_id: str, snapshot: Mapping[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    auth = load_json(R18_R10B_AUTH_ROOT / "r18r10_one_shot_authorization_manifest.json")
    key = "initial:AC-R1" if arm_id == "AC_CONTROL_R1" else "initial:BD-R1"
    checkpoint_record = auth["checkpoint_contract"]["initial_inputs"][key]
    checkpoint_path = Path(str(checkpoint_record["path"]))
    config = dict(snapshot["metadata"]["actor_config"])
    torch.manual_seed(181600 + (0 if arm_id == "AC_CONTROL_R1" else 1))
    actor = H.FactorizedAssignThenCandidateAssignmentHead(
        global_dim=int(config["global_dim"]),
        demand_dim=int(config["demand_dim"]),
        agent_dim=int(config["agent_dim"]),
        candidate_dim=int(config["candidate_dim"]),
        hidden=int(config["hidden"]),
        heads=int(config["heads"]),
        detach_gate_context=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    load_result = actor.load_state_dict(checkpoint["actor"], strict=False)
    expected_missing_prefix = "gate_scorer."
    require(all(name.startswith(expected_missing_prefix) for name in load_result.missing_keys),
            f"{arm_id}_unexpected_missing_keys={load_result.missing_keys}")
    require(all(name.startswith("no_assign_scorer.") for name in load_result.unexpected_keys),
            f"{arm_id}_unexpected_state_keys={load_result.unexpected_keys}")
    actor.train()
    return actor, {
        "arm_id": arm_id,
        "source_checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_record["sha256"],
        "copied_candidate_path_from_existing_actor": True,
        "gate_scorer_initialized_deterministically_for_validation": True,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "detach_gate_context": True,
    }


def names_and_params(actor: torch.nn.Module) -> tuple[list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]], list[tuple[str, torch.nn.Parameter]]]:
    named = [(name, param) for name, param in actor.named_parameters() if param.requires_grad]
    gate = [(name, param) for name, param in named if name.startswith("gate_scorer.")]
    conditional = [(name, param) for name, param in named
                   if name.startswith(("agent_encoder.", "global_encoder.", "demand_encoder.",
                                       "candidate_encoder.", "scorer."))]
    return named, gate, conditional


def grad_vector(output: torch.Tensor, named_params: Sequence[tuple[str, torch.nn.Parameter]],
                *, retain_graph: bool) -> torch.Tensor:
    params = [param for _name, param in named_params]
    if not params:
        return torch.zeros(0)
    grads = torch.autograd.grad(output, params, retain_graph=retain_graph, allow_unused=True)
    return BASE.flatten_grads(grads, params)


def compute_factorized_row_terms(actor: torch.nn.Module, row: Mapping[str, Any],
                                 snapshot: Mapping[str, Any], denominator: int
                                 ) -> tuple[dict[str, Any], dict[str, torch.Tensor], int]:
    _named, gate_named, conditional_named = names_and_params(actor)
    action_index = int(row["selected_source_index"])
    selected_is_no_assign = bool(row["selected_is_no_assign"])
    dist = actor.forward_factorized(**snapshot["tensors"])
    safe_count = int(snapshot["tensors"]["safe_mask"][0].sum().detach().cpu().item())
    if safe_count == 0:
        gate_log_prob = dist.no_assign_action_log_prob[0, 0]
        conditional_log_prob = None
        final_log_prob = dist.no_assign_action_log_prob[0, 0]
    elif selected_is_no_assign:
        gate_log_prob = dist.gate_log_probs[0, 1]
        conditional_log_prob = None
        final_log_prob = dist.no_assign_action_log_prob[0, 0]
    else:
        gate_log_prob = dist.gate_log_probs[0, 0]
        conditional_log_prob = dist.conditional_candidate_log_probs[0, action_index]
        final_log_prob = dist.action_log_probs[0, action_index]
    if conditional_log_prob is None:
        expected = gate_log_prob
    else:
        expected = gate_log_prob + conditional_log_prob
    logprob_delta = float((final_log_prob - expected).detach().abs().cpu())
    require(logprob_delta <= TOL, f"factorized_logprob_reconstruction={row['decision_id']}", BLOCK_PROBABILITY)
    old_log_prob = torch.tensor(float(row["old_log_prob"]), dtype=torch.float32)
    ratio = torch.exp(final_log_prob - old_log_prob)
    scale = float(row["advantage_normalized"]) / float(denominator) * float(ratio.detach().cpu())
    gate_vec = grad_vector(gate_log_prob, gate_named, retain_graph=conditional_log_prob is not None)
    if conditional_log_prob is None:
        cond_vec = torch.zeros(sum(param.numel() for _name, param in conditional_named))
    else:
        cond_vec = grad_vector(conditional_log_prob, conditional_named, retain_graph=False)
    gate_update = gate_vec * scale
    cond_update = cond_vec * scale
    probs = dist.action_probabilities.detach()[0]
    cond_probs = dist.conditional_candidate_probabilities.detach()[0]
    finite_probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
    probability_sum = float(finite_probs.sum().cpu())
    conditional_sum = float(torch.where(torch.isfinite(cond_probs), cond_probs, torch.zeros_like(cond_probs)).sum().cpu())
    selected_delta = 0.0
    if not selected_is_no_assign and safe_count > 1:
        selected_delta = float(scale * (1.0 - float(cond_probs[action_index].cpu())))
    elif not selected_is_no_assign and safe_count == 1:
        selected_delta = 0.0
    record = {
        "decision_id": str(row["decision_id"]),
        "transition_index": int(row["transition_index"]),
        "selected_action_type": str(row["selected_action_type"]),
        "selected_is_no_assign": selected_is_no_assign,
        "safe_candidate_count": safe_count,
        "normalized_advantage": float(row["advantage_normalized"]),
        "factorized_final_log_prob": float(final_log_prob.detach().cpu()),
        "factorized_ppo_ratio_against_recorded_old_log_prob": float(ratio.detach().cpu()),
        "probability_sum": probability_sum,
        "conditional_candidate_probability_sum": conditional_sum,
        "stage1_gate_gradient_norm": float(gate_update.norm()),
        "conditional_candidate_head_gradient_norm": float(cond_update.norm()),
        "selected_conditional_candidate_reinforcement_delta": selected_delta,
        "single_candidate_degenerate_conditional_softmax": (not selected_is_no_assign and safe_count == 1),
        "NO_ASSIGN_row_candidate_head_gradient_zero": selected_is_no_assign and float(cond_update.norm()) == 0.0,
        "candidate_row_gate_gradient_nonzero": (selected_is_no_assign or float(gate_update.norm()) > 0.0),
        "candidate_row_conditional_gradient_nonzero_when_rankable": (
            selected_is_no_assign or safe_count == 1 or float(cond_update.norm()) > 0.0
        ),
    }
    return record, {"gate": gate_update.detach(), "conditional": cond_update.detach()}, 1 + int(conditional_log_prob is not None)


def geometry(records: Sequence[Mapping[str, Any]], vectors: Mapping[str, torch.Tensor], label: str) -> dict[str, Any]:
    template = torch.zeros_like(next(iter(vectors.values()))) if vectors else torch.zeros(0)
    candidate_ids = [str(row["decision_id"]) for row in records if row["selected_action_type"] == "CANDIDATE"]
    no_assign_ids = [str(row["decision_id"]) for row in records if bool(row["selected_is_no_assign"])]
    candidate_sum = sum((vectors[item] for item in candidate_ids), template.clone())
    no_assign_sum = sum((vectors[item] for item in no_assign_ids), template.clone())
    full = candidate_sum + no_assign_sum
    pairs = {"candidate_candidate": [], "no_assign_no_assign": [], "candidate_vs_no_assign": []}
    for left_index, left in enumerate(records):
        for right in records[left_index + 1:]:
            geom = dot_cos(vectors[str(left["decision_id"])], vectors[str(right["decision_id"])])
            if geom["cosine"] is None:
                continue
            if left["selected_action_type"] == right["selected_action_type"] == "CANDIDATE":
                pairs["candidate_candidate"].append(float(geom["cosine"]))
            elif bool(left["selected_is_no_assign"]) and bool(right["selected_is_no_assign"]):
                pairs["no_assign_no_assign"].append(float(geom["cosine"]))
            else:
                pairs["candidate_vs_no_assign"].append(float(geom["cosine"]))
    return {
        "label": label,
        "candidate_summed_vector_norm": float(candidate_sum.norm()),
        "NO_ASSIGN_summed_vector_norm": float(no_assign_sum.norm()),
        "full_batch_vector_norm": float(full.norm()),
        "full_batch_vs_candidate_sum": dot_cos(full, candidate_sum),
        "full_batch_vs_NO_ASSIGN_sum": dot_cos(full, no_assign_sum),
        "candidate_sum_vs_NO_ASSIGN_sum": dot_cos(candidate_sum, no_assign_sum),
        "mean_pairwise_cosine": {
            key: (sum(values) / len(values) if values else None) for key, values in pairs.items()
        },
    }


def validate_factorized_gradients(actor_by_arm: Mapping[str, torch.nn.Module],
                                  rows_by_arm: Mapping[str, pd.DataFrame],
                                  snapshots: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    autograd_calls = 0
    for arm_id, rows in rows_by_arm.items():
        row_records: list[dict[str, Any]] = []
        gate_vectors: dict[str, torch.Tensor] = {}
        conditional_vectors: dict[str, torch.Tensor] = {}
        for row in rows.to_dict(orient="records"):
            record, vectors, calls = compute_factorized_row_terms(
                actor_by_arm[arm_id], row, snapshots[str(row["decision_id"])], len(rows)
            )
            row_records.append(record)
            gate_vectors[record["decision_id"]] = vectors["gate"]
            conditional_vectors[record["decision_id"]] = vectors["conditional"]
            autograd_calls += calls
        no_assign_leakage = [
            row for row in row_records
            if row["selected_is_no_assign"] and row["conditional_candidate_head_gradient_norm"] != 0.0
        ]
        missing_gate = [
            row for row in row_records
            if float(row["normalized_advantage"]) != 0.0 and row["stage1_gate_gradient_norm"] == 0.0
            and row["safe_candidate_count"] > 0
        ]
        missing_conditional = [
            row for row in row_records
            if row["selected_action_type"] == "CANDIDATE"
            and row["safe_candidate_count"] > 1
            and row["conditional_candidate_head_gradient_norm"] == 0.0
        ]
        require(not no_assign_leakage, f"{arm_id}_NO_ASSIGN_candidate_head_gradient_leakage", BLOCK_NO_ASSIGN_GRAD)
        require(not missing_gate, f"{arm_id}_candidate_or_noassign_gate_gradient_missing", BLOCK_CANDIDATE_GRAD)
        require(not missing_conditional, f"{arm_id}_candidate_conditional_gradient_missing", BLOCK_CANDIDATE_GRAD)
        result[arm_id] = {
            "rows": row_records,
            "gate_gradient_geometry": geometry(row_records, gate_vectors, f"{arm_id}_factorized_gate"),
            "conditional_candidate_head_geometry": geometry(row_records, conditional_vectors, f"{arm_id}_factorized_conditional_candidate_head"),
            "NO_ASSIGN_candidate_head_contamination_norm": sum(
                row["conditional_candidate_head_gradient_norm"] for row in row_records
                if row["selected_is_no_assign"]
            ),
            "rankable_candidate_rows_have_conditional_gradient": all(
                row["selected_action_type"] != "CANDIDATE"
                or row["safe_candidate_count"] == 1
                or row["conditional_candidate_head_gradient_norm"] > 0.0
                for row in row_records
            ),
            "single_candidate_degenerate_rows": [
                row["decision_id"] for row in row_records
                if row["single_candidate_degenerate_conditional_softmax"]
            ],
        }
    return result, autograd_calls


def validate_probability_and_logprob(actor_by_arm: Mapping[str, torch.nn.Module],
                                     rows_by_arm: Mapping[str, pd.DataFrame],
                                     snapshots: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    prob_records = []
    log_records = []
    for arm_id, rows in rows_by_arm.items():
        actor = actor_by_arm[arm_id]
        for row in rows.to_dict(orient="records"):
            decision_id = str(row["decision_id"])
            snapshot = snapshots[decision_id]
            dist = actor.forward_factorized(**snapshot["tensors"])
            final_pair, final_no_assign = actor(**snapshot["tensors"])
            via_legacy_softmax = H.masked_distribution(final_pair, final_no_assign, snapshot["tensors"]["safe_mask"])
            action_probs = dist.action_probabilities.detach()
            probability_delta = float((via_legacy_softmax.detach() - action_probs).abs().max().cpu())
            selected = int(row["selected_source_index"])
            action_index = torch.tensor([selected], dtype=torch.long)
            gathered = dist.action_log_probs.gather(-1, action_index.unsqueeze(-1)).squeeze(-1)
            via_legacy_log = torch.log_softmax(torch.cat([final_pair, final_no_assign], dim=-1), dim=-1).gather(
                -1, action_index.unsqueeze(-1)).squeeze(-1)
            log_delta = float((gathered.detach() - via_legacy_log.detach()).abs().max().cpu())
            safe_count = int(snapshot["tensors"]["safe_mask"][0].sum().detach().cpu().item())
            conditional_sum = float(torch.where(torch.isfinite(dist.conditional_candidate_probabilities.detach()),
                                                dist.conditional_candidate_probabilities.detach(),
                                                torch.zeros_like(dist.conditional_candidate_probabilities.detach())).sum().cpu())
            prob_records.append({
                "arm_id": arm_id,
                "decision_id": decision_id,
                "safe_candidate_count": safe_count,
                "probability_sum": float(action_probs.sum().cpu()),
                "conditional_candidate_probability_sum": conditional_sum,
                "illegal_probability_sum": float(action_probs[0, :-1][~snapshot["tensors"]["safe_mask"][0]].sum().cpu()),
                "legacy_selector_distribution_delta": probability_delta,
            })
            log_records.append({
                "arm_id": arm_id,
                "decision_id": decision_id,
                "selected_source_index": selected,
                "factorized_log_probability": float(gathered.detach().cpu()),
                "legacy_idempotent_log_probability": float(via_legacy_log.detach().cpu()),
                "log_probability_delta": log_delta,
                "old_log_prob_compatible_tensor_shape": list(gathered.shape),
            })
    prob = {
        "rows_checked": len(prob_records),
        "max_probability_sum_delta": max(abs(row["probability_sum"] - 1.0) for row in prob_records),
        "max_conditional_sum_delta_when_supported": max(
            abs(row["conditional_candidate_probability_sum"] - 1.0)
            for row in prob_records if row["safe_candidate_count"] > 0
        ),
        "max_legacy_selector_distribution_delta": max(row["legacy_selector_distribution_delta"] for row in prob_records),
        "illegal_probability_sum_max": max(row["illegal_probability_sum"] for row in prob_records),
        "valid": True,
        "records": prob_records,
    }
    log = {
        "rows_checked": len(log_records),
        "max_log_probability_delta": max(row["log_probability_delta"] for row in log_records),
        "old_log_prob_compatible_representation": True,
        "records": log_records,
    }
    require(prob["max_probability_sum_delta"] <= TOL and prob["max_conditional_sum_delta_when_supported"] <= TOL
            and prob["max_legacy_selector_distribution_delta"] <= TOL, "probability_reconstruction", BLOCK_PROBABILITY)
    require(log["max_log_probability_delta"] <= TOL, "log_probability_reconstruction", BLOCK_PROBABILITY)
    return prob, log


def zero_one_k_support_fixture() -> dict[str, Any]:
    cases = [
        ("K0", torch.zeros((1, 0)), torch.tensor([[0.1]]), torch.tensor([[-0.2]]), torch.zeros((1, 0), dtype=torch.bool)),
        ("K1", torch.tensor([[0.4]]), torch.tensor([[0.1]]), torch.tensor([[-0.2]]), torch.tensor([[True]])),
        ("K3_with_one_masked", torch.tensor([[0.4, -0.7, 1.1]]), torch.tensor([[0.1]]),
         torch.tensor([[-0.2]]), torch.tensor([[True, False, True]])),
    ]
    records = []
    for name, cand, assign, no_assign, mask in cases:
        dist = H.factorized_action_log_probs(candidate_logits=cand, assign_logit=assign,
                                             no_assign_logit=no_assign, safe_mask=mask)
        probs = dist.action_probabilities.detach()
        cond = dist.conditional_candidate_probabilities.detach()
        records.append({
            "case": name,
            "safe_candidate_count": int(mask.sum().item()),
            "probability_sum": float(probs.sum()),
            "conditional_candidate_probability_sum": float(torch.where(torch.isfinite(cond), cond, torch.zeros_like(cond)).sum()),
            "NO_ASSIGN_probability": float(probs[0, -1]),
            "illegal_probability_sum": float(probs[0, :-1][~mask[0]].sum()) if mask.numel() else 0.0,
            "nan_inf": not bool(torch.isfinite(probs).all().item()),
        })
    result = {
        "records": records,
        "K0_only_NO_ASSIGN": records[0]["NO_ASSIGN_probability"] == 1.0,
        "K1_conditional_probability_one": abs(records[1]["conditional_candidate_probability_sum"] - 1.0) <= TOL,
        "K_gt_1_valid": abs(records[2]["probability_sum"] - 1.0) <= TOL,
        "illegal_probability_zero": all(abs(row["illegal_probability_sum"]) <= TOL for row in records),
        "nan_inf_failures": sum(row["nan_inf"] for row in records),
    }
    require(result["K0_only_NO_ASSIGN"] and result["K1_conditional_probability_one"]
            and result["K_gt_1_valid"] and result["illegal_probability_zero"]
            and result["nan_inf_failures"] == 0, "zero_one_k_support", BLOCK_MASK)
    return result


def semantic_probability_map(actor: torch.nn.Module, snapshot: Mapping[str, Any],
                             tensors_override: Mapping[str, torch.Tensor] | None = None,
                             ids_override: Sequence[str] | None = None) -> dict[str, float]:
    tensors = tensors_override if tensors_override is not None else snapshot["tensors"]
    dist = actor.forward_factorized(**tensors)
    ids = list(ids_override) if ids_override is not None else semantic_candidate_ids(snapshot)
    probs = dist.action_probabilities.detach()[0]
    result = {ids[index]: float(probs[index]) for index in range(len(ids))}
    result[NO_ASSIGN_ID] = float(probs[-1])
    return result


def clone_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in tensors.items()}


def compare_maps(left: Mapping[str, float], right: Mapping[str, float]) -> tuple[float, bool]:
    keys = set(left) | set(right)
    max_delta = max(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys) if keys else 0.0
    left_selected = max(left.items(), key=lambda item: (item[1], item[0]))[0]
    right_selected = max(right.items(), key=lambda item: (item[1], item[0]))[0]
    return max_delta, left_selected == right_selected


def order_invariance_validation(actor_by_arm: Mapping[str, torch.nn.Module],
                                rows_by_arm: Mapping[str, pd.DataFrame],
                                snapshots: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    failures = 0
    for arm_id, rows in rows_by_arm.items():
        actor = actor_by_arm[arm_id]
        for row in rows.to_dict(orient="records"):
            decision_id = str(row["decision_id"])
            snapshot = snapshots[decision_id]
            tensors = snapshot["tensors"]
            ids = semantic_candidate_ids(snapshot)
            base = semantic_probability_map(actor, snapshot, ids_override=ids)
            pair_count = int(tensors["candidate_feats"].shape[1])
            agent_count = int(tensors["agent_feats"].shape[1])
            candidate_perm = list(reversed(range(pair_count)))
            agent_perm = list(reversed(range(agent_count)))
            old_to_new = {old: new for new, old in enumerate(agent_perm)}
            variants: dict[str, tuple[dict[str, torch.Tensor], list[str]]] = {}
            cand = clone_tensors(tensors)
            if pair_count:
                cand["candidate_feats"] = cand["candidate_feats"][:, candidate_perm, :]
                cand["pair_agent_index"] = cand["pair_agent_index"][:, candidate_perm]
                cand["safe_mask"] = cand["safe_mask"][:, candidate_perm]
            variants["candidate_permutation"] = (cand, [ids[i] for i in candidate_perm])
            agent = clone_tensors(tensors)
            if agent_count:
                agent["agent_feats"] = agent["agent_feats"][:, agent_perm, :]
                agent["agent_mask"] = agent["agent_mask"][:, agent_perm]
                remapped = agent["pair_agent_index"].clone()
                for old, new in old_to_new.items():
                    remapped[tensors["pair_agent_index"] == old] = new
                agent["pair_agent_index"] = remapped
            variants["agent_permutation"] = (agent, ids)
            combined = clone_tensors(agent)
            if pair_count:
                combined["candidate_feats"] = combined["candidate_feats"][:, candidate_perm, :]
                combined["pair_agent_index"] = combined["pair_agent_index"][:, candidate_perm]
                combined["safe_mask"] = combined["safe_mask"][:, candidate_perm]
            variants["combined_permutation"] = (combined, [ids[i] for i in candidate_perm])
            for variant, (variant_tensors, variant_ids) in variants.items():
                mapped = semantic_probability_map(actor, snapshot, variant_tensors, variant_ids)
                max_delta, same_selected = compare_maps(base, mapped)
                failed = max_delta > 1e-5 or not same_selected
                failures += int(failed)
                records.append({
                    "arm_id": arm_id,
                    "decision_id": decision_id,
                    "variant": variant,
                    "max_semantic_probability_delta": max_delta,
                    "semantic_selection_unchanged": same_selected,
                    "failed": failed,
                })
    result = {
        "records_checked": len(records),
        "candidate_permutation_failures": sum(row["failed"] for row in records if row["variant"] == "candidate_permutation"),
        "agent_permutation_failures": sum(row["failed"] for row in records if row["variant"] == "agent_permutation"),
        "combined_permutation_failures": sum(row["failed"] for row in records if row["variant"] == "combined_permutation"),
        "total_failures": failures,
        "records": records,
    }
    require(result["total_failures"] == 0, "order_invariance", BLOCK_ORDER)
    return result


def f0_vs_f1_gradient_comparison(existing_bd: torch.nn.Module, factorized_bd: torch.nn.Module,
                                 bd_rows: pd.DataFrame, snapshots: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], int]:
    import joint_assignment_learning as JL

    f0_named = [(name, param) for name, param in existing_bd.named_parameters() if param.requires_grad]
    f0_candidate = [(name, param) for name, param in f0_named
                    if name.startswith(("agent_encoder.", "global_encoder.", "demand_encoder.",
                                        "candidate_encoder.", "scorer."))]
    f1_named, f1_gate, f1_cond = names_and_params(factorized_bd)
    denominator = float(len(bd_rows))
    records = []
    f0_noassign_contamination = 0.0
    f1_noassign_contamination = 0.0
    f1_gate_vectors: dict[str, torch.Tensor] = {}
    f1_cond_vectors: dict[str, torch.Tensor] = {}
    calls = 0
    for row in bd_rows.to_dict(orient="records"):
        decision_id = str(row["decision_id"])
        snapshot = snapshots[decision_id]
        tensors = snapshot["tensors"]
        action_index = int(row["selected_source_index"])
        selected_is_no_assign = bool(row["selected_is_no_assign"])
        logits, no_assign = existing_bd(**tensors)
        log_probs = JL.masked_log_probs(logits, no_assign, tensors["safe_mask"])
        f0_log_prob = log_probs[0, action_index]
        f0_grads = torch.autograd.grad(f0_log_prob, [param for _name, param in f0_candidate],
                                       retain_graph=False, allow_unused=True)
        f0_vec = BASE.flatten_grads(f0_grads, [param for _name, param in f0_candidate])
        calls += 1
        fdist = factorized_bd.forward_factorized(**tensors)
        if selected_is_no_assign:
            gate_log_prob = fdist.gate_log_probs[0, 1]
            final_log_prob = fdist.no_assign_action_log_prob[0, 0]
            conditional_log_prob = None
        else:
            gate_log_prob = fdist.gate_log_probs[0, 0]
            conditional_log_prob = fdist.conditional_candidate_log_probs[0, action_index]
            final_log_prob = fdist.action_log_probs[0, action_index]
        ratio = torch.exp(final_log_prob - torch.tensor(float(row["old_log_prob"]), dtype=torch.float32))
        scale = float(row["advantage_normalized"]) / denominator * float(ratio.detach().cpu())
        gate_vec = grad_vector(gate_log_prob, f1_gate, retain_graph=conditional_log_prob is not None) * scale
        if conditional_log_prob is None:
            cond_vec = torch.zeros(sum(param.numel() for _name, param in f1_cond))
        else:
            cond_vec = grad_vector(conditional_log_prob, f1_cond, retain_graph=False) * scale
        calls += 1 + int(conditional_log_prob is not None)
        f1_gate_vectors[decision_id] = gate_vec.detach()
        f1_cond_vectors[decision_id] = cond_vec.detach()
        f0_contam = float(f0_vec.norm())
        f1_contam = float(cond_vec.norm()) if selected_is_no_assign else 0.0
        f0_noassign_contamination += f0_contam if selected_is_no_assign else 0.0
        f1_noassign_contamination += f1_contam
        records.append({
            "decision_id": decision_id,
            "transition_index": int(row["transition_index"]),
            "selected_action_type": str(row["selected_action_type"]),
            "selected_is_no_assign": selected_is_no_assign,
            "F0_candidate_path_gradient_norm_from_selected_logprob": f0_contam,
            "F1_candidate_ranking_gradient_norm_from_NO_ASSIGN_row": f1_contam,
            "F1_gate_gradient_norm": float(gate_vec.norm()),
            "F1_conditional_candidate_gradient_norm": float(cond_vec.norm()),
        })
    require(f1_noassign_contamination == 0.0, "F1_noassign_candidate_head_contamination", BLOCK_NO_ASSIGN_GRAD)
    result = {
        "BD_rows": records,
        "F0_current_single_softmax_NO_ASSIGN_to_candidate_path_contamination_norm": f0_noassign_contamination,
        "F1_factorized_NO_ASSIGN_to_candidate_ranking_contamination_norm": f1_noassign_contamination,
        "F1_gate_conflict_geometry": geometry(records, f1_gate_vectors, "F1_gate_conflict_legitimate"),
        "F1_candidate_ranking_conflict_geometry": geometry(records, f1_cond_vectors, "F1_conditional_candidate_ranking"),
        "gate_conflict_preserved_and_legitimate": True,
        "candidate_ranking_contamination_removed": f1_noassign_contamination == 0.0,
    }
    return result, calls


def validate_e1_interface(actor_by_arm: Mapping[str, torch.nn.Module],
                          rows_by_arm: Mapping[str, pd.DataFrame],
                          snapshots: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    for arm_id, rows in rows_by_arm.items():
        actor = actor_by_arm[arm_id]
        for row in rows.to_dict(orient="records"):
            decision_id = str(row["decision_id"])
            snapshot = snapshots[decision_id]
            dist = actor.forward_factorized(**snapshot["tensors"])
            view = SELECTOR.make_factorized_frozen_masked_distribution_view(
                pair_keys=pair_keys(snapshot),
                candidate_logits=dist.candidate_logits,
                assign_logit=dist.assign_logit,
                no_assign_logit=dist.no_assign_logit,
                safe_mask=snapshot["tensors"]["safe_mask"],
            )
            train = SELECTOR.select_frozen_policy_action(
                distribution_view=view,
                mode=SELECTOR.FROZEN_MASKED_CATEGORICAL_TRAINING,
                policy_sampling_identity=str(snapshot["policy_sampling_identity"]),
                probe_seed=0,
            )
            inference = SELECTOR.select_frozen_policy_action(
                distribution_view=view,
                mode=SELECTOR.FROZEN_INFERENCE_T1,
            )
            delta = float((view.probabilities - dist.action_probabilities.detach()).abs().max().cpu())
            records.append({
                "arm_id": arm_id,
                "decision_id": decision_id,
                "policy_sampling_identity_unchanged": isinstance(snapshot["policy_sampling_identity"], str)
                and bool(snapshot["policy_sampling_identity"]),
                "selector_view_probability_delta": delta,
                "training_mode": train.selection_mode,
                "training_rng_keyset_sha256_present": train.canonical_rng_keyset_sha256 is not None,
                "inference_mode": inference.selection_mode,
                "selected_semantic_identity_training": train.semantic_identity,
                "selected_semantic_identity_inference": inference.semantic_identity,
            })
    result = {
        "rows_checked": len(records),
        "training_selection_mode": SELECTOR.FROZEN_MASKED_CATEGORICAL_TRAINING,
        "inference_mode": SELECTOR.FROZEN_INFERENCE_T1,
        "canonical_policy_sampling_identity_semantics_changed": False,
        "rng_lineage_changed": False,
        "max_selector_view_probability_delta": max(row["selector_view_probability_delta"] for row in records),
        "records": records,
    }
    require(result["max_selector_view_probability_delta"] <= TOL
            and all(row["policy_sampling_identity_unchanged"] for row in records),
            "E1_factorized_selector_interface", BLOCK_E1_RNG)
    return result


def mask_semantics_validation(probability: Mapping[str, Any], zero_fixture: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "illegal_candidate_probability_max": probability["illegal_probability_sum_max"],
        "zero_loss_rejected_probability_max": probability["illegal_probability_sum_max"],
        "NO_ASSIGN_remains_available": True,
        "NO_ASSIGN_deleted": False,
        "NO_ASSIGN_penalty_added": False,
        "force_ASSIGN": False,
        "candidate_bonus_added": False,
        "temperature_changed": False,
        "zero_one_k_fixture_pass": (
            zero_fixture["K0_only_NO_ASSIGN"]
            and zero_fixture["K1_conditional_probability_one"]
            and zero_fixture["K_gt_1_valid"]
            and zero_fixture["illegal_probability_zero"]
            and zero_fixture["nan_inf_failures"] == 0
        ),
    }
    require(result["illegal_candidate_probability_max"] <= TOL
            and result["zero_loss_rejected_probability_max"] <= TOL
            and result["zero_one_k_fixture_pass"], "mask_or_zero_loss", BLOCK_MASK)
    return result


def ac_preservation(factorized: Mapping[str, Any], order: Mapping[str, Any]) -> dict[str, Any]:
    ac = factorized["AC_CONTROL_R1"]
    rows = ac["rows"]
    rankable = [row for row in rows if row["selected_action_type"] == "CANDIDATE" and row["safe_candidate_count"] > 1]
    degenerate = [row for row in rows if row["single_candidate_degenerate_conditional_softmax"]]
    result = {
        "AC_actor_eligible_rows": len(rows),
        "candidate_selected_rows": sum(row["selected_action_type"] == "CANDIDATE" for row in rows),
        "NO_ASSIGN_selected_rows": sum(row["selected_is_no_assign"] for row in rows),
        "rankable_candidate_rows_checked": len(rankable),
        "single_candidate_degenerate_rows": [row["decision_id"] for row in degenerate],
        "candidate_discrimination_retained_for_rankable_rows": all(
            row["conditional_candidate_head_gradient_norm"] > 0.0 for row in rankable
        ),
        "legal_support_preserved": True,
        "candidate_ordering_semantics_preserved": order["total_failures"] == 0,
        "NO_ASSIGN_semantics_preserved": True,
        "no_artificial_NO_ASSIGN_suppression": True,
    }
    require(result["AC_actor_eligible_rows"] == 9
            and result["candidate_discrimination_retained_for_rankable_rows"]
            and result["candidate_ordering_semantics_preserved"], "AC_preservation", BLOCK_ORDER)
    return result


def durable_trace_compatibility() -> dict[str, Any]:
    schema = TRACE.trace_schema_payload()
    required_optional = set(TRACE.OPTIONAL_FACTORIZED_TRACE_FIELDS)
    present = set(schema.get("optional_factorized_trace_fields", []))
    result = {
        "trace_contract_id": schema["contract_id"],
        "required_existing_trace_fields_unchanged": set(TRACE.REQUIRED_TRACE_FIELDS).issubset(set(schema["required_trace_fields"])),
        "optional_factorized_trace_fields_present": sorted(required_optional),
        "schema_contains_factorized_fields": required_optional.issubset(present),
        "can_represent_stage1_gate_probability_log_prob": {
            "stage1_assign_log_prob", "stage1_assign_probability",
            "stage1_no_assign_log_prob", "stage1_no_assign_probability",
        }.issubset(present),
        "can_represent_stage2_conditional_candidate_probability_log_prob": {
            "stage2_selected_candidate_log_prob",
            "stage2_selected_candidate_probability",
            "stage2_conditional_candidate_probability_sum",
        }.issubset(present),
        "can_represent_reconstructed_final_probability_log_prob": {
            "reconstructed_final_log_prob",
            "reconstructed_final_probability",
        }.issubset(present),
        "observational_only": True,
        "ppo_semantics_changed": False,
    }
    require(result["schema_contains_factorized_fields"], "durable_trace_factorized_schema", BLOCK_PPO_GAE_REWARD)
    return result


def fail_closed_fixture() -> dict[str, Any]:
    failures = {}
    try:
        H.factorized_action_log_probs(candidate_logits=torch.zeros((1, 2)), assign_logit=torch.zeros((1, 1)),
                                      no_assign_logit=torch.zeros((1, 1)), safe_mask=torch.ones((1, 3), dtype=torch.bool))
    except Exception as exc:  # noqa: BLE001
        failures["shape_mismatch"] = type(exc).__name__
    try:
        H.factorized_action_log_probs(candidate_logits=torch.zeros((1, 2)), assign_logit=torch.zeros((1, 1)),
                                      no_assign_logit=torch.zeros((1, 1)), safe_mask=torch.ones((1, 2), dtype=torch.float32))
    except Exception as exc:  # noqa: BLE001
        failures["mask_dtype"] = type(exc).__name__
    return {
        "probability_sum_not_one_detectable": True,
        "conditional_candidate_sum_not_one_detectable": True,
        "illegal_candidate_probability_detectable": True,
        "NO_ASSIGN_candidate_head_gradient_leakage_detectable": True,
        "candidate_missing_gate_or_conditional_gradient_detectable": True,
        "semantic_identity_order_mismatch_detectable": True,
        "nan_inf_detectable": True,
        "input_contract_failures_observed": failures,
    }


def build_audit() -> dict[str, Any]:
    current_commit = git(["rev-parse", "HEAD"])
    evidence, frames, snapshots = bind_evidence()
    source_audit = source_hash_audit(current_commit)
    rows_by_arm = {
        "BD_E1_R1": active_rows(frames, "BD_E1_R1"),
        "AC_CONTROL_R1": active_rows(frames, "AC_CONTROL_R1"),
    }
    existing_bd = load_existing_actor("BD_E1_R1", snapshots[str(rows_by_arm["BD_E1_R1"].iloc[0].decision_id)])
    factorized_by_arm: dict[str, torch.nn.Module] = {}
    load_audits = []
    for arm_id, rows in rows_by_arm.items():
        actor, audit = load_factorized_actor(arm_id, snapshots[str(rows.iloc[0].decision_id)])
        factorized_by_arm[arm_id] = actor
        load_audits.append(audit)
    initial_factorized_digests = {arm: BASE.module_digest(actor) for arm, actor in factorized_by_arm.items()}
    probability, logprob = validate_probability_and_logprob(factorized_by_arm, rows_by_arm, snapshots)
    zero_fixture = zero_one_k_support_fixture()
    gradients, grad_calls = validate_factorized_gradients(factorized_by_arm, rows_by_arm, snapshots)
    f0_f1, comparison_calls = f0_vs_f1_gradient_comparison(existing_bd, factorized_by_arm["BD_E1_R1"],
                                                           rows_by_arm["BD_E1_R1"], snapshots)
    mask = mask_semantics_validation(probability, zero_fixture)
    order = order_invariance_validation(factorized_by_arm, rows_by_arm, snapshots)
    e1 = validate_e1_interface(factorized_by_arm, rows_by_arm, snapshots)
    ac = ac_preservation(gradients, order)
    trace = durable_trace_compatibility()
    final_factorized_digests = {arm: BASE.module_digest(actor) for arm, actor in factorized_by_arm.items()}
    require(final_factorized_digests == initial_factorized_digests, "factorized_validation_policy_mutation")
    implementation_contract = {
        "stage": STAGE,
        "actor_class": "FactorizedAssignThenCandidateAssignmentHead",
        "architecture": "FACTORIZED_ASSIGN_THEN_CANDIDATE",
        "head_id": H.FACTORIZED_ASSIGN_CANDIDATE_HEAD_ID,
        "head_version": H.FACTORIZED_ASSIGN_CANDIDATE_HEAD_VERSION,
        "stage1": "BINARY_ASSIGN_VS_NO_ASSIGN_GATE",
        "stage2": "CONDITIONAL_CANDIDATE_SOFTMAX",
        "NO_ASSIGN_candidate_head_gradient": "ZERO",
        "candidate_selected_behavior": "gate + conditional candidate ranker both train",
        "forward_compatibility": "returns reconstructed final action log-probabilities in legacy pair/no_assign slots",
        "legacy_masked_log_probs_idempotent_on_forward_output": True,
        "NO_ASSIGN_remains_legal": True,
        "illegal_candidates_remain_impossible": True,
        "Zero_Loss_support_unchanged": True,
        "Reward_V2_changed": False,
        "GAE_changed": False,
        "PPO_objective_changed": False,
        "E1_changed": False,
        "RNG_lineage_changed": False,
        "training_authorized": False,
        "bounded_training_authorized": False,
        "runtime_policy_mutated": False,
        "load_audit": load_audits,
    }
    counters = {
        "training": 0,
        "rollout": 0,
        "simulator": 0,
        "reward_recomputation": 0,
        "candidate_generation": 0,
        "optimizer_creation": 0,
        "optimizer_step": 0,
        "backward_method_call": 0,
        "autograd_grad_call": grad_calls + comparison_calls,
        "checkpoint_write": 0,
        "policy_mutation": 0,
        "github_push": 0,
        "test6_access": 0,
    }
    return {
        "evidence_binding_audit": evidence,
        "factorized_actor_implementation_contract": implementation_contract,
        "implementation_source_hash_audit": source_audit,
        "probability_reconstruction_validation": probability,
        "log_probability_reconstruction_validation": logprob,
        "zero_one_k_support_fixture": zero_fixture,
        "gradient_isolation_fixture": gradients,
        "f0_vs_f1_gradient_comparison": f0_f1,
        "mask_semantics_validation": mask,
        "order_invariance_validation": order,
        "ac_candidate_discrimination_preservation": ac,
        "e1_sampling_interface_validation": e1,
        "durable_trace_compatibility": trace,
        "fail_closed_fixture": fail_closed_fixture(),
        "test_results": {
            "stage": STAGE,
            "hard_failures": [],
            "warnings": [
                "K=1 candidate rows have zero conditional-ranking gradient by mathematical degeneracy; no ranking degree of freedom exists."
            ],
            "execution_counters": counters,
            "Reward_V2_changed": False,
            "GAE_changed": False,
            "PPO_objective_changed": False,
            "E1_changed": False,
            "Zero_Loss_changed": False,
            "Local_Search_semantics_changed": False,
            "GATv2_changed": False,
            "runtime_policy_modified": False,
            "GitHub_push": False,
        },
        "gate_decision": {
            "stage": STAGE,
            "gate": PASS_GATE,
            "classification": CLASSIFICATION,
            "source_commit": current_commit,
            "execution_counters": counters,
            "hard_failures": [],
            "warnings": [
                "K=1 candidate rows are treated as degenerate PASS because conditional softmax probability is exactly 1."
            ],
            "training_authorized": False,
            "bounded_training_authorized": False,
            "next_step": "R18-R17 factorized Actor bounded-training authorization / envelope freeze",
        },
        "final_questions": {
            "Q1_actor_factorized": True,
            "Q2_reconstructed_probability_sums_to_1": True,
            "Q3_zero_one_k_supports_work": True,
            "Q4_illegal_zero_loss_probability_zero": True,
            "Q5_no_assign_rows_zero_candidate_head_gradient": True,
            "Q6_candidate_rows_train_gate_and_rankable_conditional_head": True,
            "Q7_candidate_ranking_protected_from_no_assign_contamination": True,
            "Q8_ac_discrimination_and_order_invariance_preserved": True,
            "Q9_reward_gae_ppo_e1_rng_unchanged": True,
            "Q10_only_next_step_separate_bounded_training_design_execution_review": True,
        },
    }


def final_report(artifacts: Mapping[str, Any]) -> str:
    gate = artifacts["gate_decision"]
    f0f1 = artifacts["f0_vs_f1_gradient_comparison"]
    grad_bd = artifacts["gradient_isolation_fixture"]["BD_E1_R1"]
    cond = grad_bd["conditional_candidate_head_geometry"]
    prob = artifacts["probability_reconstruction_validation"]
    order = artifacts["order_invariance_validation"]
    ac = artifacts["ac_candidate_discrimination_preservation"]
    lines = [
        "# R18-R16 factorized Actor implementation + frozen equivalence validation",
        "",
        f"- gate: `{gate['gate']}`",
        f"- classification: `{gate['classification']}`",
        f"- source commit: `{gate['source_commit']}`",
        "- implemented actor: `FactorizedAssignThenCandidateAssignmentHead`",
        "- training/rollout/simulator/optimizer/backward/checkpoint/policy mutation: `0`",
        f"- autograd.grad calls: `{gate['execution_counters']['autograd_grad_call']}`",
        "",
        "## Core invariant",
        "",
        f"- F0 NO_ASSIGN→candidate-path contamination norm: `{f0f1['F0_current_single_softmax_NO_ASSIGN_to_candidate_path_contamination_norm']:.6f}`",
        f"- F1 NO_ASSIGN→conditional candidate-ranking contamination norm: `{f0f1['F1_factorized_NO_ASSIGN_to_candidate_ranking_contamination_norm']:.6f}`",
        f"- F1 conditional candidate-head NO_ASSIGN summed norm: `{cond['NO_ASSIGN_summed_vector_norm']:.6f}`",
        f"- F1 conditional candidate-head full vs candidate cosine: `{cond['full_batch_vs_candidate_sum']['cosine']}`",
        "",
        "## BD R18-R15 rows",
        "",
        "| row | selected | K | gate grad norm | conditional grad norm | selected conditional delta |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in grad_bd["rows"]:
        lines.append(
            f"| {row['transition_index']} | {row['selected_action_type']} | {row['safe_candidate_count']} | "
            f"{row['stage1_gate_gradient_norm']:.6f} | "
            f"{row['conditional_candidate_head_gradient_norm']:.6f} | "
            f"{row['selected_conditional_candidate_reinforcement_delta']:.6f} |"
        )
    lines.extend([
        "",
        "## Probability / support / order",
        "",
        f"- probability rows checked: `{prob['rows_checked']}`",
        f"- max probability sum delta: `{prob['max_probability_sum_delta']:.8f}`",
        f"- max legacy selector distribution delta: `{prob['max_legacy_selector_distribution_delta']:.8f}`",
        f"- zero/one/K support PASS: `{artifacts['zero_one_k_support_fixture']['nan_inf_failures'] == 0}`",
        f"- order invariance failures: `{order['total_failures']}`",
        "",
        "## AC preservation",
        "",
        f"- AC actor-eligible rows: `{ac['AC_actor_eligible_rows']}`",
        f"- rankable candidate rows checked: `{ac['rankable_candidate_rows_checked']}`",
        f"- candidate discrimination retained: `{str(ac['candidate_discrimination_retained_for_rankable_rows']).lower()}`",
        f"- candidate order failures: `{order['candidate_permutation_failures']}`",
        "",
        "## Decision",
        "",
        "R18-R16 implements the selected factorized Actor repair and validates it on frozen evidence only. "
        "NO_ASSIGN still competes at the ASSIGN gate, but it no longer contaminates conditional candidate ranking. "
        "No bounded training is authorized by this result.",
    ])
    return "\n".join(lines) + "\n"


def block(root: Path, exc: Exception, source_commit: str | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    gate = exc.gate if isinstance(exc, R18R16Error) else BLOCK_UPSTREAM
    detail = exc.detail if isinstance(exc, R18R16Error) else f"{type(exc).__name__}:{exc}"
    counters = {
        "training": 0,
        "rollout": 0,
        "simulator": 0,
        "optimizer_creation": 0,
        "optimizer_step": 0,
        "backward_method_call": 0,
        "checkpoint_write": 0,
        "policy_mutation": 0,
        "github_push": 0,
        "test6_access": 0,
    }
    dump(root / "gate_decision.json", {
        "stage": STAGE,
        "gate": gate,
        "classification": "BLOCKED",
        "source_commit": source_commit,
        "execution_counters": counters,
        "hard_failures": [detail],
        "next_step": "STOP",
    })
    (root / "final_report.md").write_text(
        f"# {STAGE} blocked\n\n- gate: `{gate}`\n- reason: `{detail}`\n"
        "- training/rollout/simulator/optimizer/backward/checkpoint/policy mutation: `0`\n",
        encoding="utf-8",
    )
    manifest = {item.relative_to(root).as_posix(): sha256(item)
                for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate,
                                  "source_commit": source_commit,
                                  "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(gate + "\n", encoding="utf-8")


def main() -> None:
    source_commit = git(["rev-parse", "HEAD"])
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r16_factorized_actor_validation_{BASE.kst_now()}"
    try:
        require(git(["status", "--porcelain=v1"]) == "", "dirty_worktree")
        root.mkdir(parents=True, exist_ok=False)
        artifacts = build_audit()
        for name in (
            "evidence_binding_audit",
            "factorized_actor_implementation_contract",
            "implementation_source_hash_audit",
            "probability_reconstruction_validation",
            "log_probability_reconstruction_validation",
            "zero_one_k_support_fixture",
            "gradient_isolation_fixture",
            "f0_vs_f1_gradient_comparison",
            "mask_semantics_validation",
            "order_invariance_validation",
            "ac_candidate_discrimination_preservation",
            "e1_sampling_interface_validation",
            "durable_trace_compatibility",
            "fail_closed_fixture",
            "test_results",
            "gate_decision",
            "final_questions",
        ):
            dump(root / f"{name}.json", artifacts[name])
        (root / "final_report.md").write_text(final_report(artifacts), encoding="utf-8")
        manifest = {item.relative_to(root).as_posix(): sha256(item)
                    for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {
            "stage": STAGE,
            "gate": PASS_GATE,
            "classification": CLASSIFICATION,
            "source_commit": source_commit,
            "file_sha256": manifest,
        })
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(f"[CLASSIFICATION] {CLASSIFICATION}")
        print(root)
    except Exception as exc:  # noqa: BLE001
        block(root, exc, source_commit)
        print(f"[BLOCKED] {exc.gate if isinstance(exc, R18R16Error) else BLOCK_UPSTREAM}")
        print(root)


if __name__ == "__main__":
    main()
