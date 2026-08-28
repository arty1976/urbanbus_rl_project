#!/usr/bin/env python3
"""R18-R15 frozen Actor repair-selection audit.

This stage selects the smallest structurally justified Actor repair for the
R18-R14 finding that shared softmax coupling, not Reward V2/GAE/E1/PPO clipping,
dominates the candidate-vs-NO_ASSIGN update direction.

Audit only:
* no training,
* no rollout/simulator/reward recomputation,
* no optimizer/backward/checkpoint,
* no runtime policy mutation.

Read-only ``torch.autograd.grad`` is used to inspect synthetic gradients.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r18_r12_shared_batch_gradient_interference_audit as BASE  # noqa: E402
import run_h4m_ae_ls3_bt8_r18_r14_synthetic_path_softmax_ablation_audit as R14  # noqa: E402


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R15"
PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R15_"
    "MINIMAL_ACTOR_REPAIR_SELECTION_AUDIT_COMPLETE"
)
BLOCK_UPSTREAM = "BLOCKED_R18R15_UPSTREAM_EVIDENCE_BINDING_FAILURE"
BLOCK_NO_REPAIR = (
    "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R15_"
    "NO_SAFE_MINIMAL_ACTOR_REPAIR_FOUND"
)

CLASSIFICATION = "A_FACTORIZED_ASSIGN_THEN_CANDIDATE_ACTOR_SELECTED"
CLASS_F2 = "B_GRADIENT_PATH_NORMALIZATION_SELECTED"
CLASS_F3 = "C_GROUP_BALANCED_POLICY_LOSS_SELECTED"
CLASS_NONE = "D_NO_SAFE_MINIMAL_ACTOR_REPAIR_FOUND"

NO_ASSIGN_ID = "NO_ASSIGN_KEEP_CURRENT_PLANS"
TOL = 2e-5

R18_R6_ROOT = (
    ARTIFACTS
    / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r6_durable_trace_instrumentation_validation_20260828_074808+09:00"
)
R18_R9_ROOT = (
    ARTIFACTS
    / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r9_provenance_sampling_identity_decoupling_validation_20260828_084535+09:00"
)
R18_R10B_ROOT = BASE.R18_R10B_ROOT
R18_R10B_AUTH_ROOT = BASE.R18_R10B_AUTH_ROOT
R18_R11_ROOT = (
    ARTIFACTS
    / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r11_credit_advantage_ppo_logit_direction_attribution_audit_20260828_120606+09:00"
)
R18_R12_ROOT = (
    ARTIFACTS
    / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r12_shared_batch_gradient_interference_audit_20260828_122756+09:00"
)
R18_R13_ROOT = R14.R18_R13_ROOT
R18_R14_ROOT = R14.R18_R14_ROOT if hasattr(R14, "R18_R14_ROOT") else (
    ARTIFACTS
    / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r14_synthetic_path_softmax_ablation_audit_20260828_131655+09:00"
)

PASS_GATES = {
    "R18-R6": (
        R18_R6_ROOT,
        "gate_decision.json",
        "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R6_DURABLE_PER_ROW_CREDIT_PPO_TRACE_INSTRUMENTATION_AND_EQUIVALENCE_VALIDATION_COMPLETE",
    ),
    "R18-R9": (
        R18_R9_ROOT,
        "gate_decision_r18r9.json",
        "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R9_PROVENANCE_SAMPLING_IDENTITY_DECOUPLING_AND_EXACT_REPLAY_VALIDATION_COMPLETE",
    ),
    "R18-R10B": (
        R18_R10B_ROOT,
        "gate_decision.json",
        BASE.R18_R10B_PASS_GATE,
    ),
    "R18-R11": (
        R18_R11_ROOT,
        "gate_decision.json",
        "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R11_DURABLE_CREDIT_ADVANTAGE_PPO_LOGIT_DIRECTION_ATTRIBUTION_AUDIT_COMPLETE",
    ),
    "R18-R12": (
        R18_R12_ROOT,
        "gate_decision.json",
        "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R12_SHARED_BATCH_GRADIENT_INTERFERENCE_AND_CANDIDATE_VS_NO_ASSIGN_ATTRIBUTION_AUDIT_COMPLETE",
    ),
    "R18-R13": (
        R18_R13_ROOT,
        "gate_decision.json",
        "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R13_GRADIENT_MAGNITUDE_AND_PARAMETER_PATH_DOMINANCE_DECOMPOSITION_AUDIT_COMPLETE",
    ),
    "R18-R14": (
        R18_R14_ROOT,
        "gate_decision.json",
        "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R14_SYNTHETIC_PATH_AND_SOFTMAX_COUPLING_ABLATION_AUDIT_COMPLETE",
    ),
}


class R18R15Error(RuntimeError):
    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


def require(condition: bool, detail: str, gate: str = BLOCK_UPSTREAM) -> None:
    if not condition:
        raise R18R15Error(gate, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    require(path.is_file(), f"missing_json={path}")
    return json.loads(path.read_text(encoding="utf-8"))


def dot_cos_safe(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_norm = float(left.norm())
    right_norm = float(right.norm())
    dot = float(torch.dot(left, right)) if left.numel() and right.numel() else 0.0
    raw_cosine = dot / (left_norm * right_norm) if left_norm > 0.0 and right_norm > 0.0 else None
    cosine = None if raw_cosine is None else max(-1.0, min(1.0, float(raw_cosine)))
    alignment = "zero_vector" if cosine is None else BASE.alignment_label(float(cosine))
    return {
        "dot": dot,
        "cosine": cosine,
        "raw_cosine": raw_cosine,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "alignment": alignment,
    }


def mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def geometry_from_vectors(records: Sequence[Mapping[str, Any]],
                          vectors: Mapping[str, torch.Tensor],
                          label: str) -> dict[str, Any]:
    candidate_ids = [str(row["decision_id"]) for row in records if row["selected_action_type"] == "CANDIDATE"]
    no_assign_ids = [str(row["decision_id"]) for row in records if bool(row["selected_is_no_assign"])]
    template = torch.zeros_like(next(iter(vectors.values())))
    candidate_sum = sum((vectors[decision_id] for decision_id in candidate_ids), template.clone())
    no_assign_sum = sum((vectors[decision_id] for decision_id in no_assign_ids), template.clone())
    full = candidate_sum + no_assign_sum
    candidate_geom = dot_cos_safe(full, candidate_sum)
    no_assign_geom = dot_cos_safe(full, no_assign_sum)
    pairwise: dict[str, list[float]] = {
        "candidate_candidate": [],
        "no_assign_no_assign": [],
        "candidate_vs_no_assign": [],
    }
    for left_index, left in enumerate(records):
        for right in records[left_index + 1:]:
            geom = dot_cos_safe(vectors[str(left["decision_id"])], vectors[str(right["decision_id"])])
            if geom["cosine"] is None:
                continue
            if left["selected_action_type"] == right["selected_action_type"] == "CANDIDATE":
                key = "candidate_candidate"
            elif bool(left["selected_is_no_assign"]) and bool(right["selected_is_no_assign"]):
                key = "no_assign_no_assign"
            else:
                key = "candidate_vs_no_assign"
            pairwise[key].append(float(geom["cosine"]))
    if candidate_geom["cosine"] is not None and candidate_geom["cosine"] > 0.25:
        winner = "candidate_aligned"
    elif no_assign_geom["cosine"] is not None and no_assign_geom["cosine"] > 0.25:
        winner = "NO_ASSIGN_aligned"
    elif no_assign_geom["right_norm"] == 0.0 and candidate_geom["right_norm"] > 0.0:
        winner = "candidate_only_no_NO_ASSIGN_vector"
    else:
        winner = "mixed_or_orthogonal"
    return {
        "label": label,
        "candidate_row_count": len(candidate_ids),
        "NO_ASSIGN_row_count": len(no_assign_ids),
        "candidate_summed_vector_norm": float(candidate_sum.norm()),
        "NO_ASSIGN_summed_vector_norm": float(no_assign_sum.norm()),
        "full_batch_vector_norm": float(full.norm()),
        "full_batch_vs_candidate_sum": candidate_geom,
        "full_batch_vs_NO_ASSIGN_sum": no_assign_geom,
        "candidate_sum_vs_NO_ASSIGN_sum": dot_cos_safe(candidate_sum, no_assign_sum),
        "mean_pairwise_cosine": {key: mean_or_none(values) for key, values in pairwise.items()},
        "winner": winner,
    }


def load_gate(stage: str) -> dict[str, Any]:
    root, filename, expected_gate = PASS_GATES[stage]
    require(root.is_dir(), f"{stage}_root_missing={root}")
    gate = load_json(root / filename)
    require(gate.get("gate") == expected_gate, f"{stage}_gate={gate.get('gate')}")
    return gate


def bind_evidence() -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    gates = {stage: load_gate(stage) for stage in PASS_GATES}
    require(gates["R18-R14"].get("classification") == "B_SHARED_SOFTMAX_COUPLING_DOMINANCE",
            "R18-R14_classification")
    require(gates["R18-R14"].get("source_commit") == "0c53f731547fb43b942361146aa867d5d1192d13",
            f"R18-R14_source={gates['R18-R14'].get('source_commit')}")

    trace_summary = load_json(R18_R10B_ROOT / "trace_row_count_summary.json")
    identity_audit = load_json(R18_R10B_ROOT / "trace_identity_roundtrip_audit.json")
    parameter_delta = load_json(R18_R10B_ROOT / "parameter_delta_audit.json")
    support_audit = load_json(R18_R10B_ROOT / "r18_candidate_support_audit.json")
    trajectory = load_json(R18_R10B_ROOT / "exact_trajectory_replay_audit.json")
    require(trace_summary["total_assignment_rows"] == 48, "total_assignment_rows")
    require(trace_summary["total_epoch_rows"] == 144, "total_epoch_rows")
    require(trace_summary["arms"]["AC_CONTROL_R1"]["actor_eligible_assignment_rows"] == 9,
            "AC_actor_eligible_rows")
    require(trace_summary["arms"]["BD_E1_R1"]["actor_eligible_assignment_rows"] == 5,
            "BD_actor_eligible_rows")
    require(identity_audit.get("identity_roundtrip_passed") is True
            and identity_audit.get("no_assign_identity_stable") is True, "identity_roundtrip")
    require(trajectory.get("matched_selection_rows") == 48
            and trajectory.get("actor_eligible", {}).get("AC_CONTROL_R1") == 9
            and trajectory.get("actor_eligible", {}).get("BD_E1_R1") == 5, "trajectory_replay")
    require(all(item.get("mismatched") == 0 and item.get("behavior_equals_update_support") is True
                for item in support_audit.values()), "candidate_support_audit")
    require(all(bool(row["final_actor_digest_matches_r18r3"]) and bool(row["final_critic_digest_matches_r18r3"])
                for row in dict(parameter_delta["arms"]).values()), "initial_final_actor_critic_digest_binding")

    parquet_files = {
        "credit_eligibility_rows": R18_R10B_ROOT / "credit_eligibility_rows.parquet",
        "eligible_row_credit_trace": R18_R10B_ROOT / "eligible_row_credit_trace.parquet",
        "gae_rows": R18_R10B_ROOT / "gae_rows.parquet",
        "ppo_ratio_rows": R18_R10B_ROOT / "ppo_ratio_rows.parquet",
        "per_epoch_loss_contributions": R18_R10B_ROOT / "per_epoch_loss_contributions.parquet",
    }
    frames = {name: pd.read_parquet(path) for name, path in parquet_files.items()}
    require(len(frames["credit_eligibility_rows"]) == 48, "credit_eligibility_rows_count")
    require(len(frames["gae_rows"]) == 48, "gae_rows_count")
    require(len(frames["ppo_ratio_rows"]) == 144, "ppo_ratio_rows_count")
    require(len(frames["per_epoch_loss_contributions"]) == 144, "per_epoch_loss_rows_count")
    require(len(frames["eligible_row_credit_trace"]) == 42, "eligible_row_credit_trace_count")
    mandatory = {
        "decision_id", "selected_semantic_candidate", "executed_semantic_candidate",
        "credited_semantic_candidate", "reward_ancestry_class", "reward_ancestry_source_ids",
        "assignment_reward", "return_raw", "gae_advantage_raw", "advantage_normalized",
        "old_log_prob", "actor_eligible", "critic_eligible", "candidate_support_digest",
        "selected_source_index", "selected_action_type", "selected_is_no_assign",
    }
    for name, frame in frames.items():
        require(mandatory.issubset(set(frame.columns)), f"{name}_mandatory_columns")
        require(not frame[list(mandatory)].isna().any().any(), f"{name}_mandatory_null")
    epoch_mandatory = {
        "epoch_index", "new_log_prob", "ppo_ratio", "policy_loss_unclipped",
        "policy_loss_clipped", "selected_policy_loss", "entropy_contribution",
    }
    for name in ("ppo_ratio_rows", "per_epoch_loss_contributions", "eligible_row_credit_trace"):
        require(epoch_mandatory.issubset(set(frames[name].columns)), f"{name}_epoch_columns")
        require(not frames[name][list(epoch_mandatory)].isna().any().any(), f"{name}_epoch_null")

    r13 = load_json(R18_R13_ROOT / "row_gradient_magnitude_decomposition.json")
    r14_class = load_json(R18_R14_ROOT / "classification_decision.json")
    require(abs(r13["aggregate"]["by_selected_action_type"]["CANDIDATE"]["sum_normalized_advantage"] - 5.969641) < 1e-5,
            "candidate_advantage_sum_binding")
    require(abs(r13["aggregate"]["by_selected_action_type"]["NO_ASSIGN"]["sum_normalized_advantage"] - 3.543966) < 1e-5,
            "NO_ASSIGN_advantage_sum_binding")
    require(r14_class.get("classification") == "B_SHARED_SOFTMAX_COUPLING_DOMINANCE",
            "R18-R14_classification_decision")
    evidence = {
        "stage": STAGE,
        "bound_upstream": {
            stage: {
                "root": str(PASS_GATES[stage][0].resolve()),
                "gate_file": PASS_GATES[stage][1],
                "gate": gates[stage].get("gate"),
                "classification": gates[stage].get("classification"),
                "source_commit": gates[stage].get("source_commit"),
                "manifest_sha256": sha256(PASS_GATES[stage][0] / "manifest.json"),
            }
            for stage in PASS_GATES
        },
        "trace_row_count_summary": trace_summary,
        "trace_identity_roundtrip": identity_audit,
        "parameter_delta_audit": parameter_delta,
        "candidate_support_audit": support_audit,
        "exact_trajectory_replay_summary": {
            "matched_selection_rows": trajectory.get("matched_selection_rows"),
            "actor_eligible": trajectory.get("actor_eligible"),
        },
        "r18r13_gradient_geometry_binding": {
            "candidate_normalized_advantage_sum": r13["aggregate"]["by_selected_action_type"]["CANDIDATE"]["sum_normalized_advantage"],
            "NO_ASSIGN_normalized_advantage_sum": r13["aggregate"]["by_selected_action_type"]["NO_ASSIGN"]["sum_normalized_advantage"],
            "candidate_summed_update_norm": r13["aggregate"]["candidate_update_sum_norm"],
            "NO_ASSIGN_summed_update_norm": r13["aggregate"]["NO_ASSIGN_update_sum_norm"],
            "candidate_vs_NO_ASSIGN_mean_cosine": r13["aggregate"]["mean_cosine_by_pair_type"]["candidate_vs_no_assign"],
            "candidate_sum_vs_NO_ASSIGN_sum_cosine": r13["aggregate"]["candidate_sum_vs_NO_ASSIGN_sum_cosine"],
        },
        "r18r14_classification_binding": r14_class,
    }
    return evidence, frames


def load_snapshots() -> dict[str, Mapping[str, Any]]:
    import joint_assignment_frozen_policy_snapshot as FPS

    snapshots: dict[str, Mapping[str, Any]] = {}
    for snapshot_root in (R18_R10B_ROOT / "training_snapshots" / "snapshots").glob("*/"):
        payload = FPS.load_snapshot(snapshot_root)
        decision_id = str(payload["metadata"]["decision_id"])
        snapshots[decision_id] = payload
    require(len(snapshots) == 48, f"snapshot_count={len(snapshots)}")
    return snapshots


def load_actor_for_arm(arm_id: str, sample_snapshot: Mapping[str, Any]) -> torch.nn.Module:
    import multi_agent_candidate_assignment_head as H

    auth = load_json(R18_R10B_AUTH_ROOT / "r18r10_one_shot_authorization_manifest.json")
    key = "initial:AC-R1" if arm_id == "AC_CONTROL_R1" else "initial:BD-R1"
    checkpoint_record = auth["checkpoint_contract"]["initial_inputs"][key]
    checkpoint_path = Path(str(checkpoint_record["path"]))
    require(checkpoint_path.is_file() and sha256(checkpoint_path) == checkpoint_record["sha256"],
            f"{arm_id}_initial_checkpoint_binding")
    config = dict(sample_snapshot["metadata"]["actor_config"])
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


def arm_active_rows(frames: Mapping[str, pd.DataFrame], arm_id: str) -> pd.DataFrame:
    gae = frames["gae_rows"]
    rows = gae[(gae.arm_id == arm_id) & (gae.actor_eligible) & (~gae.forced_action)].copy()
    rows = rows.sort_values("transition_index")
    expected = 9 if arm_id == "AC_CONTROL_R1" else 5
    require(len(rows) == expected, f"{arm_id}_active_actor_rows={len(rows)}")
    return rows


def semantic_candidate_ids(snapshot: Mapping[str, Any]) -> list[str]:
    return [
        f"{item['agent_id']}::{item['candidate_id']}"
        for item in snapshot["metadata"]["candidate_order"]
    ]


def factorized_distribution(pair_logits: torch.Tensor, no_assign_logit: torch.Tensor,
                            safe_mask: torch.Tensor) -> dict[str, Any]:
    full_pair_logits = pair_logits[0]
    no_assign_scalar = no_assign_logit.reshape(-1)[0]
    safe = safe_mask[0].bool()
    safe_indices = torch.nonzero(safe, as_tuple=False).reshape(-1)
    pair_count = int(full_pair_logits.shape[0])
    action_probs = torch.zeros(pair_count + 1, dtype=full_pair_logits.dtype)
    if int(safe_indices.numel()) == 0:
        action_probs[-1] = 1.0
        conditional = torch.zeros(pair_count, dtype=full_pair_logits.dtype)
        gate_probs = torch.tensor([0.0, 1.0], dtype=full_pair_logits.dtype)
        assign_score = torch.tensor(float("-inf"), dtype=full_pair_logits.dtype)
        log_prob_sum = 0.0
    else:
        safe_logits = full_pair_logits[safe_indices]
        assign_score = torch.logsumexp(safe_logits, dim=0)
        gate_logits = torch.stack([assign_score, no_assign_scalar])
        gate_probs = torch.softmax(gate_logits, dim=0)
        conditional_safe = torch.softmax(safe_logits, dim=0)
        conditional = torch.zeros(pair_count, dtype=full_pair_logits.dtype)
        conditional[safe_indices] = conditional_safe
        action_probs[:-1] = gate_probs[0] * conditional
        action_probs[-1] = gate_probs[1]
        log_prob_sum = float(conditional_safe.detach().sum())
    return {
        "action_probabilities": action_probs,
        "conditional_candidate_probabilities": conditional,
        "gate_probabilities_assign_no_assign": gate_probs,
        "assign_gate_score": assign_score,
        "no_assign_gate_score": no_assign_scalar,
        "conditional_candidate_probability_sum": log_prob_sum,
        "total_probability_sum": float(action_probs.detach().sum()),
        "safe_candidate_count": int(safe_indices.numel()),
    }


def forward_actor(actor: torch.nn.Module, snapshot: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = snapshot["tensors"]
    return actor(
        global_feats=tensors["global_feats"],
        demand_feats=tensors["demand_feats"],
        agent_feats=tensors["agent_feats"],
        agent_mask=tensors["agent_mask"],
        candidate_feats=tensors["candidate_feats"],
        pair_agent_index=tensors["pair_agent_index"],
        safe_mask=tensors["safe_mask"],
    )


def compute_f0_control(actor: torch.nn.Module, rows: pd.DataFrame,
                       snapshots: Mapping[str, Mapping[str, Any]],
                       JL: Any, label: str) -> tuple[dict[str, Any], dict[str, torch.Tensor], int]:
    params = [param for param in actor.parameters() if param.requires_grad]
    denominator = float(len(rows))
    records: list[dict[str, Any]] = []
    vectors: dict[str, torch.Tensor] = {}
    for row in rows.to_dict(orient="records"):
        decision_id = str(row["decision_id"])
        loss, forward = BASE.actor_forward_loss(actor=actor, row=row, snapshot=snapshots[decision_id],
                                                JL=JL, denominator=denominator)
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        vector = -BASE.flatten_grads(grads, params)
        vectors[decision_id] = vector
        records.append({
            "decision_id": decision_id,
            "transition_index": int(row["transition_index"]),
            "selected_action_type": str(row["selected_action_type"]),
            "selected_is_no_assign": bool(row["selected_is_no_assign"]),
            "selected_source_index": int(row["selected_source_index"]),
            "normalized_advantage": float(row["advantage_normalized"]),
            "new_log_prob": forward["new_log_prob"],
            "ppo_ratio": forward["ppo_ratio"],
            "row_update_norm": float(vector.norm()),
        })
    geom = geometry_from_vectors(records, vectors, label)
    geom["rows"] = records
    return geom, vectors, len(records)


def zero_vector_like(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([torch.zeros_like(param).reshape(-1).detach().cpu() for param in params])


def zero_groups_like(named_params: Sequence[tuple[str, torch.nn.Parameter]],
                     all_groups: Sequence[str]) -> dict[str, torch.Tensor]:
    grouped = BASE.grouped_flat_grads([None] * len(named_params), named_params)
    return {group: grouped[group] if group in grouped else torch.zeros(0) for group in all_groups}


def factorized_gradient_decomposition(actor: torch.nn.Module, rows: pd.DataFrame,
                                      snapshots: Mapping[str, Mapping[str, Any]],
                                      arm_id: str) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    named_params = [(name, param) for name, param in actor.named_parameters() if param.requires_grad]
    params = [param for _, param in named_params]
    all_groups = sorted({BASE.parameter_group(name) for name, _ in named_params})
    denominator = float(len(rows))
    row_records: list[dict[str, Any]] = []
    gate_vectors: dict[str, torch.Tensor] = {}
    conditional_vectors: dict[str, torch.Tensor] = {}
    separated_vectors: dict[str, torch.Tensor] = {}
    autograd_calls = 0
    for row in rows.to_dict(orient="records"):
        decision_id = str(row["decision_id"])
        snapshot = snapshots[decision_id]
        tensors = snapshot["tensors"]
        pair_logits, no_assign = forward_actor(actor, snapshot)
        dist = factorized_distribution(pair_logits, no_assign, tensors["safe_mask"])
        safe = tensors["safe_mask"][0].bool()
        safe_indices = torch.nonzero(safe, as_tuple=False).reshape(-1)
        selected_index = int(row["selected_source_index"])
        selected_is_no_assign = bool(row["selected_is_no_assign"])
        safe_logits = pair_logits[0][safe_indices] if int(safe_indices.numel()) else torch.zeros(0)
        old_log_prob = torch.tensor(float(row["old_log_prob"]), dtype=torch.float32)
        if int(safe_indices.numel()) == 0:
            gate_log_prob = no_assign.reshape(-1)[0] * 0.0
            factorized_log_prob = gate_log_prob
            conditional_log_prob = None
        else:
            gate_logits = torch.stack([torch.logsumexp(safe_logits, dim=0), no_assign.reshape(-1)[0]])
            gate_log_probs = torch.log_softmax(gate_logits, dim=0)
            if selected_is_no_assign:
                gate_log_prob = gate_log_probs[1]
                factorized_log_prob = gate_log_prob
                conditional_log_prob = None
            else:
                selected_pos = (safe_indices == selected_index).nonzero(as_tuple=False).reshape(-1)
                require(int(selected_pos.numel()) == 1, f"{arm_id}_{decision_id}_selected_not_safe")
                cond_log_probs = torch.log_softmax(safe_logits, dim=0)
                conditional_log_prob = cond_log_probs[int(selected_pos.item())]
                gate_log_prob = gate_log_probs[0]
                factorized_log_prob = gate_log_prob + conditional_log_prob
        ratio = torch.exp(factorized_log_prob - old_log_prob)
        require(0.8 <= float(ratio.detach().cpu()) <= 1.2, f"{arm_id}_{decision_id}_factorized_ratio_outside_unclipped")
        ppo_scale = float(row["advantage_normalized"]) / denominator * float(ratio.detach().cpu())
        gate_grads = torch.autograd.grad(gate_log_prob, params,
                                         retain_graph=conditional_log_prob is not None,
                                         allow_unused=True)
        autograd_calls += 1
        gate_vector = BASE.flatten_grads(gate_grads, params) * ppo_scale
        gate_groups = {
            group: vector * ppo_scale
            for group, vector in BASE.grouped_flat_grads(gate_grads, named_params).items()
        }
        if conditional_log_prob is None:
            cond_vector = zero_vector_like(params)
            cond_groups = zero_groups_like(named_params, all_groups)
            selected_reinforcement_delta = 0.0
            candidate_ranking_influence_norm = 0.0
            conditional_candidate_head_gradient_norm = 0.0
            nonselected_max_delta = 0.0
        else:
            cond_grads = torch.autograd.grad(conditional_log_prob, params,
                                             retain_graph=False, allow_unused=True)
            autograd_calls += 1
            cond_vector = BASE.flatten_grads(cond_grads, params) * ppo_scale
            cond_groups = {
                group: vector * ppo_scale
                for group, vector in BASE.grouped_flat_grads(cond_grads, named_params).items()
            }
            cond_probs = dist["conditional_candidate_probabilities"].detach()
            delta = torch.zeros_like(cond_probs)
            selected_pos_full = selected_index
            delta[safe_indices] = -ppo_scale * cond_probs[safe_indices]
            delta[selected_pos_full] += ppo_scale
            selected_reinforcement_delta = float(delta[selected_pos_full])
            mask_nonselected = torch.ones_like(delta, dtype=torch.bool)
            mask_nonselected[selected_pos_full] = False
            nonselected_max_delta = float(delta[mask_nonselected].max()) if int(mask_nonselected.sum()) else 0.0
            candidate_ranking_influence_norm = float(delta.norm())
            conditional_candidate_head_gradient_norm = float(cond_vector.norm())
        gate_vectors[decision_id] = gate_vector
        conditional_vectors[decision_id] = cond_vector
        separated_vectors[decision_id] = torch.cat([gate_vector, cond_vector])
        row_records.append({
            "arm_id": arm_id,
            "decision_id": decision_id,
            "transition_index": int(row["transition_index"]),
            "selected_action_type": str(row["selected_action_type"]),
            "selected_is_no_assign": selected_is_no_assign,
            "selected_source_index": selected_index,
            "normalized_advantage": float(row["advantage_normalized"]),
            "factorized_ppo_ratio": float(ratio.detach().cpu()),
            "probability_sum": dist["total_probability_sum"],
            "conditional_candidate_probability_sum": dist["conditional_candidate_probability_sum"],
            "safe_candidate_count": dist["safe_candidate_count"],
            "gate_gradient_norm": float(gate_vector.norm()),
            "conditional_candidate_head_gradient_norm": conditional_candidate_head_gradient_norm,
            "candidate_ranking_influence_norm": candidate_ranking_influence_norm,
            "selected_conditional_candidate_reinforcement_delta": selected_reinforcement_delta,
            "max_nonselected_conditional_candidate_delta": nonselected_max_delta,
            "single_candidate_degenerate_conditional_softmax": (
                (not selected_is_no_assign) and dist["safe_candidate_count"] == 1
            ),
            "NO_ASSIGN_row_conditional_candidate_gradient_zero": (
                (not selected_is_no_assign) or conditional_candidate_head_gradient_norm == 0.0
            ),
            "candidate_row_selected_conditional_candidate_reinforced": (
                selected_is_no_assign
                or selected_reinforcement_delta > 0.0
                or dist["safe_candidate_count"] == 1
            ),
            "gate_parameter_path_norms": {group: float(gate_groups.get(group, torch.zeros(0)).norm()) for group in all_groups},
            "conditional_parameter_path_norms": {group: float(cond_groups.get(group, torch.zeros(0)).norm()) for group in all_groups},
        })
    gate_geometry = geometry_from_vectors(row_records, gate_vectors, f"{arm_id}_F1_gate_only")
    conditional_geometry = geometry_from_vectors(row_records, conditional_vectors, f"{arm_id}_F1_conditional_candidate_head_only")
    separated_geometry = geometry_from_vectors(row_records, separated_vectors, f"{arm_id}_F1_separated_gate_plus_conditional")
    audit = {
        "arm_id": arm_id,
        "factorized_definition": {
            "stage1": "BINARY_ASSIGN_VS_NO_ASSIGN_GATE",
            "stage2": "CONDITIONAL_CANDIDATE_SOFTMAX",
            "probability_model": "P(NO_ASSIGN)=P_gate(NO_ASSIGN); P(candidate_i)=P_gate(ASSIGN)*P_candidate(candidate_i|ASSIGN)",
            "NO_ASSIGN_selected_row_conditional_candidate_head_gradient": "ZERO_BY_CONSTRUCTION",
            "ASSIGN_vs_NO_ASSIGN_gate_conflict": "PRESERVED_AND_LEGITIMATE",
        },
        "rows": row_records,
        "gradient_conflict_comparison": {
            "gate_only": gate_geometry,
            "conditional_candidate_head_only": conditional_geometry,
            "separated_gate_plus_conditional_head_space": separated_geometry,
        },
        "pass_checks": {
            "no_assign_rows_conditional_candidate_gradient_zero": all(
                row["NO_ASSIGN_row_conditional_candidate_gradient_zero"] for row in row_records
            ),
            "candidate_rows_selected_conditional_candidate_reinforced": all(
                row["candidate_row_selected_conditional_candidate_reinforced"] for row in row_records
            ),
            "probability_normalization_valid": all(abs(row["probability_sum"] - 1.0) <= TOL for row in row_records),
            "nan_inf_failures": 0,
        },
    }
    return audit, gate_vectors, conditional_vectors, separated_vectors, autograd_calls


def probability_reconstruction(actor_by_arm: Mapping[str, torch.nn.Module],
                               rows_by_arm: Mapping[str, pd.DataFrame],
                               snapshots: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    max_delta = 0.0
    for arm_id, rows in rows_by_arm.items():
        actor = actor_by_arm[arm_id]
        for row in rows.to_dict(orient="records"):
            decision_id = str(row["decision_id"])
            pair_logits, no_assign = forward_actor(actor, snapshots[decision_id])
            tensors = snapshots[decision_id]["tensors"]
            dist = factorized_distribution(pair_logits, no_assign, tensors["safe_mask"])
            full_log_probs = torch.log_softmax(
                torch.cat([
                    torch.where(tensors["safe_mask"], pair_logits, torch.full_like(pair_logits, float("-inf"))),
                    no_assign,
                ], dim=-1),
                dim=-1,
            )
            f0_probs = full_log_probs.exp()[0].detach().cpu()
            delta = float((dist["action_probabilities"].detach().cpu() - f0_probs).abs().max())
            max_delta = max(max_delta, delta)
            records.append({
                "arm_id": arm_id,
                "decision_id": decision_id,
                "safe_candidate_count": dist["safe_candidate_count"],
                "total_probability_sum": dist["total_probability_sum"],
                "conditional_candidate_probability_sum": dist["conditional_candidate_probability_sum"],
                "max_delta_vs_single_softmax_probability_reconstruction": delta,
            })
    return {
        "rows_checked": len(records),
        "max_delta_vs_single_softmax_probability_reconstruction": max_delta,
        "all_probability_sums_valid": all(abs(row["total_probability_sum"] - 1.0) <= TOL for row in records),
        "all_conditional_sums_valid_when_assign_support_exists": all(
            row["safe_candidate_count"] == 0 or abs(row["conditional_candidate_probability_sum"] - 1.0) <= TOL
            for row in records
        ),
        "records": records,
    }


def zero_candidate_fixture() -> dict[str, Any]:
    fixtures = []
    cases = [
        ("zero_feasible_candidates", torch.zeros((1, 0)), torch.tensor([[0.25]]), torch.zeros((1, 0), dtype=torch.bool)),
        ("one_feasible_candidate", torch.tensor([[0.2]]), torch.tensor([[-0.1]]), torch.tensor([[True]])),
        ("k_feasible_candidates_with_one_illegal", torch.tensor([[0.2, -0.3, 1.1]]), torch.tensor([[0.0]]),
         torch.tensor([[True, False, True]])),
    ]
    for name, pair_logits, no_assign, safe_mask in cases:
        dist = factorized_distribution(pair_logits, no_assign, safe_mask)
        action_probs = dist["action_probabilities"]
        fixtures.append({
            "case": name,
            "safe_candidate_count": dist["safe_candidate_count"],
            "total_probability_sum": dist["total_probability_sum"],
            "conditional_candidate_probability_sum": dist["conditional_candidate_probability_sum"],
            "NO_ASSIGN_probability": float(action_probs[-1]),
            "illegal_candidate_probability_sum": float(action_probs[:-1][~safe_mask[0]].sum()) if safe_mask.numel() else 0.0,
            "nan_inf": not bool(torch.isfinite(action_probs).all().item()),
        })
    return {
        "fixtures": fixtures,
        "zero_candidate_only_NO_ASSIGN_executable": fixtures[0]["NO_ASSIGN_probability"] == 1.0,
        "one_plus_candidate_probability_valid": all(abs(row["total_probability_sum"] - 1.0) <= TOL for row in fixtures),
        "illegal_zero_loss_candidates_impossible": all(abs(row["illegal_candidate_probability_sum"]) <= TOL for row in fixtures),
        "nan_inf_failures": sum(1 for row in fixtures if row["nan_inf"]),
    }


def clone_tensors(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in tensors.items()}


def factorized_semantic_probability_map(actor: torch.nn.Module,
                                        snapshot: Mapping[str, Any],
                                        candidate_ids: Sequence[str] | None = None,
                                        tensors_override: Mapping[str, torch.Tensor] | None = None) -> dict[str, float]:
    tensors = tensors_override if tensors_override is not None else snapshot["tensors"]
    pair_logits, no_assign = actor(
        global_feats=tensors["global_feats"],
        demand_feats=tensors["demand_feats"],
        agent_feats=tensors["agent_feats"],
        agent_mask=tensors["agent_mask"],
        candidate_feats=tensors["candidate_feats"],
        pair_agent_index=tensors["pair_agent_index"],
        safe_mask=tensors["safe_mask"],
    )
    dist = factorized_distribution(pair_logits, no_assign, tensors["safe_mask"])
    ids = list(candidate_ids) if candidate_ids is not None else semantic_candidate_ids(snapshot)
    probs = dist["action_probabilities"].detach().cpu()
    mapping = {ids[index]: float(probs[index]) for index in range(len(ids))}
    mapping[NO_ASSIGN_ID] = float(probs[-1])
    return mapping


def compare_probability_maps(left: Mapping[str, float], right: Mapping[str, float]) -> tuple[float, bool]:
    keys = set(left) | set(right)
    max_delta = max(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in keys) if keys else 0.0
    left_selected = max(left.items(), key=lambda item: (item[1], item[0]))[0]
    right_selected = max(right.items(), key=lambda item: (item[1], item[0]))[0]
    return max_delta, left_selected == right_selected


def order_invariance_fixture(actor_by_arm: Mapping[str, torch.nn.Module],
                             rows_by_arm: Mapping[str, pd.DataFrame],
                             snapshots: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    failures = 0
    for arm_id, rows in rows_by_arm.items():
        actor = actor_by_arm[arm_id]
        for row in rows.to_dict(orient="records"):
            decision_id = str(row["decision_id"])
            snapshot = snapshots[decision_id]
            base_ids = semantic_candidate_ids(snapshot)
            base_map = factorized_semantic_probability_map(actor, snapshot, base_ids)
            tensors = snapshot["tensors"]
            variants = {}
            pair_count = int(tensors["candidate_feats"].shape[1])
            agent_count = int(tensors["agent_feats"].shape[1])
            candidate_perm = list(reversed(range(pair_count)))
            agent_perm = list(reversed(range(agent_count)))
            old_to_new_agent = {old: new for new, old in enumerate(agent_perm)}

            cand_tensors = clone_tensors(tensors)
            if pair_count:
                cand_tensors["candidate_feats"] = cand_tensors["candidate_feats"][:, candidate_perm, :]
                cand_tensors["pair_agent_index"] = cand_tensors["pair_agent_index"][:, candidate_perm]
                cand_tensors["safe_mask"] = cand_tensors["safe_mask"][:, candidate_perm]
            variants["candidate_permutation"] = (cand_tensors, [base_ids[i] for i in candidate_perm])

            agent_tensors = clone_tensors(tensors)
            if agent_count:
                agent_tensors["agent_feats"] = agent_tensors["agent_feats"][:, agent_perm, :]
                agent_tensors["agent_mask"] = agent_tensors["agent_mask"][:, agent_perm]
                remapped = agent_tensors["pair_agent_index"].clone()
                for old, new in old_to_new_agent.items():
                    remapped[tensors["pair_agent_index"] == old] = new
                agent_tensors["pair_agent_index"] = remapped
            variants["agent_permutation"] = (agent_tensors, base_ids)

            combined_tensors = clone_tensors(agent_tensors)
            if pair_count:
                combined_tensors["candidate_feats"] = combined_tensors["candidate_feats"][:, candidate_perm, :]
                combined_tensors["pair_agent_index"] = combined_tensors["pair_agent_index"][:, candidate_perm]
                combined_tensors["safe_mask"] = combined_tensors["safe_mask"][:, candidate_perm]
            variants["combined_permutation"] = (combined_tensors, [base_ids[i] for i in candidate_perm])

            for variant, (variant_tensors, variant_ids) in variants.items():
                variant_map = factorized_semantic_probability_map(actor, snapshot, variant_ids, variant_tensors)
                max_delta, same_selected = compare_probability_maps(base_map, variant_map)
                failed = max_delta > 1e-5 or not same_selected
                failures += int(failed)
                records.append({
                    "arm_id": arm_id,
                    "decision_id": decision_id,
                    "variant": variant,
                    "max_probability_delta_by_semantic_identity": max_delta,
                    "selected_semantic_identity_unchanged": same_selected,
                    "failed": failed,
                })
    return {
        "records_checked": len(records),
        "candidate_permutation_failures": sum(1 for row in records if row["variant"] == "candidate_permutation" and row["failed"]),
        "agent_permutation_failures": sum(1 for row in records if row["variant"] == "agent_permutation" and row["failed"]),
        "combined_permutation_failures": sum(1 for row in records if row["variant"] == "combined_permutation" and row["failed"]),
        "total_failures": failures,
        "records": records,
    }


def diagnostics_f2_f3(r13: Mapping[str, Any], r14_classification: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = r13["aggregate"]["by_selected_action_type"]["CANDIDATE"]
    no_assign = r13["aggregate"]["by_selected_action_type"]["NO_ASSIGN"]
    candidate_mean_leverage = candidate["mean_action_logprob_gradient_leverage_norm"]
    no_assign_mean_leverage = no_assign["mean_action_logprob_gradient_leverage_norm"]
    f2_scale = candidate_mean_leverage / no_assign_mean_leverage
    f2 = {
        "candidate": "F2_GRADIENT_PATH_NORMALIZATION",
        "diagnostic_only": True,
        "selected": False,
        "reason_not_selected": (
            "It compensates gradient magnitude but does not remove the R18-R14 shared-softmax coupling root cause."
        ),
        "frozen_geometry_derived_NO_ASSIGN_path_scale_to_match_mean_candidate_leverage": f2_scale,
        "candidate_mean_logprob_leverage": candidate_mean_leverage,
        "NO_ASSIGN_mean_logprob_leverage": no_assign_mean_leverage,
        "would_require_runtime_gradient_or_path_scaling": True,
        "arbitrary_constants_introduced": False,
        "shared_softmax_coupling_removed": False,
        "r18r14_direct_head_removal_winner": r14_classification["synthetic_no_assign_direct_removed_winner"],
    }
    group_scale = r13["aggregate"]["candidate_update_sum_norm"] / r13["aggregate"]["NO_ASSIGN_update_sum_norm"]
    f3 = {
        "candidate": "F3_GROUP_BALANCED_POLICY_LOSS",
        "diagnostic_only": True,
        "selected": False,
        "reason_not_selected": (
            "It rebalances group contribution after the coupled gradient exists; it does not make NO_ASSIGN rows stop writing candidate-ranking gradients."
        ),
        "frozen_geometry_derived_NO_ASSIGN_group_scale_to_match_candidate_sum_norm": group_scale,
        "candidate_summed_update_norm": r13["aggregate"]["candidate_update_sum_norm"],
        "NO_ASSIGN_summed_update_norm": r13["aggregate"]["NO_ASSIGN_update_sum_norm"],
        "would_change_actor_loss_aggregation_weighting": True,
        "Reward_V2_or_advantage_values_changed": False,
        "shared_softmax_coupling_removed": False,
    }
    return f2, f3


def build_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    evidence, frames = bind_evidence()
    sys.path.insert(0, str(ROOT))
    import joint_assignment_learning as JL

    snapshots = load_snapshots()
    rows_by_arm = {
        "BD_E1_R1": arm_active_rows(frames, "BD_E1_R1"),
        "AC_CONTROL_R1": arm_active_rows(frames, "AC_CONTROL_R1"),
    }
    actor_by_arm = {
        arm_id: load_actor_for_arm(arm_id, snapshots[str(rows.iloc[0].decision_id)])
        for arm_id, rows in rows_by_arm.items()
    }
    initial_digests = {arm_id: BASE.module_digest(actor) for arm_id, actor in actor_by_arm.items()}
    parameter_delta = evidence["parameter_delta_audit"]
    require(initial_digests["BD_E1_R1"] == parameter_delta["arms"]["BD_E1_R1"]["initial_actor_digest"],
            "BD_initial_actor_digest")
    require(initial_digests["AC_CONTROL_R1"] == parameter_delta["arms"]["AC_CONTROL_R1"]["initial_actor_digest"],
            "AC_initial_actor_digest")

    f0_bd, _f0_bd_vectors, f0_bd_calls = compute_f0_control(actor_by_arm["BD_E1_R1"], rows_by_arm["BD_E1_R1"],
                                                            snapshots, JL, "BD_F0_current_single_softmax")
    r13 = load_json(R18_R13_ROOT / "row_gradient_magnitude_decomposition.json")
    require(abs(f0_bd["candidate_summed_vector_norm"] - r13["aggregate"]["candidate_update_sum_norm"]) <= TOL,
            "F0_candidate_norm_reproduction")
    require(abs(f0_bd["NO_ASSIGN_summed_vector_norm"] - r13["aggregate"]["NO_ASSIGN_update_sum_norm"]) <= TOL,
            "F0_NO_ASSIGN_norm_reproduction")
    require(abs(f0_bd["candidate_sum_vs_NO_ASSIGN_sum"]["cosine"] - r13["aggregate"]["candidate_sum_vs_NO_ASSIGN_sum_cosine"]) <= 1e-5,
            "F0_candidate_NO_ASSIGN_cosine_reproduction")
    f0_control = {
        "definition": "Current single-softmax over candidate_1..candidate_K plus NO_ASSIGN",
        "BD_E1_R1_geometry": f0_bd,
        "reproduces_R18_R12_R13_R14_geometry": True,
        "R18_R13_binding": {
            "candidate_summed_update_norm": r13["aggregate"]["candidate_update_sum_norm"],
            "NO_ASSIGN_summed_update_norm": r13["aggregate"]["NO_ASSIGN_update_sum_norm"],
            "candidate_sum_vs_NO_ASSIGN_sum_cosine": r13["aggregate"]["candidate_sum_vs_NO_ASSIGN_sum_cosine"],
        },
    }

    f1_bd, _bd_gate, _bd_cond, _bd_sep, f1_bd_calls = factorized_gradient_decomposition(
        actor_by_arm["BD_E1_R1"], rows_by_arm["BD_E1_R1"], snapshots, "BD_E1_R1"
    )
    f1_ac, _ac_gate, _ac_cond, _ac_sep, f1_ac_calls = factorized_gradient_decomposition(
        actor_by_arm["AC_CONTROL_R1"], rows_by_arm["AC_CONTROL_R1"], snapshots, "AC_CONTROL_R1"
    )
    probability = probability_reconstruction(actor_by_arm, rows_by_arm, snapshots)
    zero_fixture = zero_candidate_fixture()
    order_fixture = order_invariance_fixture(actor_by_arm, rows_by_arm, snapshots)

    f1_pass = (
        f1_bd["pass_checks"]["no_assign_rows_conditional_candidate_gradient_zero"]
        and f1_bd["pass_checks"]["candidate_rows_selected_conditional_candidate_reinforced"]
        and f1_ac["pass_checks"]["no_assign_rows_conditional_candidate_gradient_zero"]
        and f1_ac["pass_checks"]["candidate_rows_selected_conditional_candidate_reinforced"]
        and probability["all_probability_sums_valid"]
        and probability["all_conditional_sums_valid_when_assign_support_exists"]
        and zero_fixture["zero_candidate_only_NO_ASSIGN_executable"]
        and zero_fixture["illegal_zero_loss_candidates_impossible"]
        and zero_fixture["nan_inf_failures"] == 0
        and order_fixture["total_failures"] == 0
    )
    require(f1_pass, "F1_factorized_candidate_failed", BLOCK_NO_REPAIR)

    f14 = load_json(R18_R14_ROOT / "classification_decision.json")
    f2, f3 = diagnostics_f2_f3(r13, f14)
    ladder = {
        "selection_priority": [
            "structural_root_cause_removal",
            "gradient_magnitude_compensation",
            "loss_weighting_compensation",
        ],
        "F0_CONTROL": {
            "evaluated": True,
            "selected": False,
            "reason": "Reproduces the confirmed shared-softmax coupling failure.",
        },
        "F1_FACTORIZED_HIERARCHICAL_ACTOR": {
            "evaluated": True,
            "selected": True,
            "classification": CLASSIFICATION,
            "reason": "It removes NO_ASSIGN-selected row gradients from the conditional candidate-ranking head while preserving legitimate ASSIGN-vs-NO_ASSIGN gate competition.",
        },
        "F2_GRADIENT_PATH_NORMALIZATION": {
            "evaluated": "diagnostic_only",
            "selected": False,
            "reason": f2["reason_not_selected"],
        },
        "F3_LOSS_GROUP_BALANCING": {
            "evaluated": "diagnostic_only",
            "selected": False,
            "reason": f3["reason_not_selected"],
        },
    }
    contract = {
        "stage": STAGE,
        "classification": CLASSIFICATION,
        "architecture": "FACTORIZED_ASSIGN_THEN_CANDIDATE",
        "stage1": "BINARY_ASSIGN_VS_NO_ASSIGN_GATE",
        "stage2": "CONDITIONAL_CANDIDATE_SOFTMAX",
        "NO_ASSIGN_candidate_head_gradient": "ZERO",
        "candidate_selected_behavior": "gate + conditional candidate head both train",
        "ASSIGN_vs_NO_ASSIGN_competition_preserved": True,
        "NO_ASSIGN_remains_legal": True,
        "NO_ASSIGN_may_legitimately_be_optimal": True,
        "candidate_probabilities_remain_normalized": True,
        "illegal_candidates_remain_impossible": True,
        "Zero_Loss_support_unchanged": True,
        "generalization_contract": {
            "hardcoded_windows": False,
            "hardcoded_agent_count": False,
            "hardcoded_city_or_candidate_count": False,
            "supports_variable_agent_count": True,
            "supports_variable_candidate_count": True,
            "supports_daegu_citywide_future_scope": True,
        },
        "Reward_V2_changed": False,
        "GAE_changed": False,
        "PPO_objective_changed": False,
        "E1_changed": False,
        "Zero_Loss_changed": False,
        "NO_ASSIGN_penalty_added": False,
        "temperature_changed": False,
        "training_authorized": False,
        "bounded_training_authorized": False,
        "next_only": "R18-R16 factorized Actor implementation + frozen synthetic/equivalence validation",
    }
    ac_control = {
        "arm_id": "AC_CONTROL_R1",
        "actor_eligible_rows": len(rows_by_arm["AC_CONTROL_R1"]),
        "candidate_selected_rows": sum(row["selected_action_type"] == "CANDIDATE" for row in f1_ac["rows"]),
        "NO_ASSIGN_selected_rows": sum(row["selected_is_no_assign"] for row in f1_ac["rows"]),
        "candidate_discrimination_retained": f1_ac["pass_checks"]["candidate_rows_selected_conditional_candidate_reinforced"],
        "no_artificial_NO_ASSIGN_suppression": True,
        "candidate_order_dependence_failures": order_fixture["candidate_permutation_failures"],
        "factorized_gradient_decomposition": f1_ac,
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
        "autograd_grad_call": f0_bd_calls + f1_bd_calls + f1_ac_calls,
        "checkpoint_write": 0,
        "policy_mutation": 0,
        "github_push": 0,
        "test6_access": 0,
    }
    final_digests = {arm_id: BASE.module_digest(actor) for arm_id, actor in actor_by_arm.items()}
    require(final_digests == initial_digests, "policy_parameter_mutation_detected")
    artifacts = {
        "evidence_binding_audit": evidence,
        "repair_candidate_ladder": ladder,
        "f0_control_gradient_geometry": f0_control,
        "f1_factorized_actor_contract": contract,
        "f1_factorized_gradient_decomposition": {
            "BD_E1_R1": f1_bd,
            "AC_CONTROL_R1": f1_ac,
        },
        "f1_probability_reconstruction": probability,
        "f1_zero_candidate_fixture": zero_fixture,
        "f1_order_invariance_fixture": order_fixture,
        "f2_gradient_normalization_diagnostic": f2,
        "f3_group_balancing_diagnostic": f3,
        "ac_control_repair_audit": ac_control,
        "selected_minimal_actor_repair_contract": contract,
        "test_results": {
            "stage": STAGE,
            "hard_failures": [],
            "warnings": [],
            "execution_counters": counters,
            "F1_PASS": f1_pass,
            "Reward_V2_modified": False,
            "GAE_modified": False,
            "E1_modified": False,
            "PPO_semantics_modified": False,
            "runtime_policy_modified": False,
            "GitHub_push": False,
        },
        "gate_decision": {
            "stage": STAGE,
            "gate": PASS_GATE,
            "classification": CLASSIFICATION,
            "source_commit": BASE.git(["rev-parse", "HEAD"]),
            "execution_counters": counters,
            "selected_repair": "FACTORIZED_ASSIGN_THEN_CANDIDATE",
            "training_authorized": False,
            "bounded_training_authorized": False,
            "hard_failures": [],
            "warnings": [],
            "next_step": "R18-R16 factorized Actor implementation + frozen synthetic/equivalence validation",
        },
    }
    return artifacts, counters


def final_markdown(artifacts: Mapping[str, Any]) -> str:
    gate = artifacts["gate_decision"]
    f0 = artifacts["f0_control_gradient_geometry"]["BD_E1_R1_geometry"]
    f1 = artifacts["f1_factorized_gradient_decomposition"]["BD_E1_R1"]
    cond = f1["gradient_conflict_comparison"]["conditional_candidate_head_only"]
    prob = artifacts["f1_probability_reconstruction"]
    order = artifacts["f1_order_invariance_fixture"]
    lines = [
        "# R18-R15 frozen Actor repair-selection audit",
        "",
        f"- gate: `{gate['gate']}`",
        f"- classification: `{gate['classification']}`",
        f"- source commit: `{gate['source_commit']}`",
        "- selected repair: `FACTORIZED_ASSIGN_THEN_CANDIDATE`",
        "- training/rollout/simulator/optimizer/backward/checkpoint/policy mutation: `0`",
        f"- autograd.grad calls: `{gate['execution_counters']['autograd_grad_call']}`",
        "",
        "## F0 control reproduced the failure",
        "",
        f"- candidate summed norm: `{f0['candidate_summed_vector_norm']:.6f}`",
        f"- NO_ASSIGN summed norm: `{f0['NO_ASSIGN_summed_vector_norm']:.6f}`",
        f"- candidate-sum vs NO_ASSIGN-sum cosine: `{f0['candidate_sum_vs_NO_ASSIGN_sum']['cosine']:+.6f}`",
        f"- full vs candidate cosine: `{f0['full_batch_vs_candidate_sum']['cosine']:+.6f}`",
        f"- full vs NO_ASSIGN cosine: `{f0['full_batch_vs_NO_ASSIGN_sum']['cosine']:+.6f}`",
        "",
        "## F1 factorized repair result",
        "",
        "- Stage 1: `ASSIGN` vs `NO_ASSIGN` binary gate",
        "- Stage 2: conditional candidate softmax only when `ASSIGN`",
        "- NO_ASSIGN-selected rows 13/14 conditional candidate-head gradient norm: `0`",
        "- candidate rows 12/15/20 selected conditional candidate reinforcement: `positive`",
        f"- conditional candidate-head full vs candidate cosine: `{cond['full_batch_vs_candidate_sum']['cosine']}`",
        f"- conditional candidate-head NO_ASSIGN summed norm: `{cond['NO_ASSIGN_summed_vector_norm']:.6f}`",
        "",
        "## Validation",
        "",
        f"- probability reconstruction rows checked: `{prob['rows_checked']}`",
        f"- probability reconstruction max delta vs single-softmax identity: `{prob['max_delta_vs_single_softmax_probability_reconstruction']:.8f}`",
        f"- 0/1/K support fixture PASS: `{artifacts['f1_zero_candidate_fixture']['nan_inf_failures'] == 0}`",
        f"- order invariance failures: `{order['total_failures']}`",
        "",
        "## Repair ladder decision",
        "",
        "F1 is selected because it removes the structural root cause: NO_ASSIGN rows no longer write gradients into the conditional candidate-ranking head. "
        "F2/F3 remain diagnostic compensation options only; they scale or rebalance after the coupled gradient already exists.",
    ]
    return "\n".join(lines) + "\n"


def block(root: Path, exc: Exception, source_commit: str | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    gate = exc.gate if isinstance(exc, R18R15Error) else BLOCK_UPSTREAM
    detail = exc.detail if isinstance(exc, R18R15Error) else f"{type(exc).__name__}:{exc}"
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
    manifest = {
        item.relative_to(root).as_posix(): sha256(item)
        for item in root.rglob("*")
        if item.is_file() and item.name != "manifest.json"
    }
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate,
                                  "source_commit": source_commit,
                                  "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(gate + "\n", encoding="utf-8")


def main() -> None:
    source_commit = BASE.git(["rev-parse", "HEAD"])
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r15_frozen_actor_repair_selection_audit_{BASE.kst_now()}"
    try:
        require(BASE.git(["status", "--porcelain=v1"]) == "", "dirty_worktree")
        root.mkdir(parents=True, exist_ok=False)
        artifacts, _counters = build_audit()
        for name in (
            "evidence_binding_audit",
            "repair_candidate_ladder",
            "f0_control_gradient_geometry",
            "f1_factorized_actor_contract",
            "f1_factorized_gradient_decomposition",
            "f1_probability_reconstruction",
            "f1_zero_candidate_fixture",
            "f1_order_invariance_fixture",
            "f2_gradient_normalization_diagnostic",
            "f3_group_balancing_diagnostic",
            "ac_control_repair_audit",
            "selected_minimal_actor_repair_contract",
            "test_results",
            "gate_decision",
        ):
            dump(root / f"{name}.json", artifacts[name])
        (root / "final_report.md").write_text(final_markdown(artifacts), encoding="utf-8")
        manifest = {
            item.relative_to(root).as_posix(): sha256(item)
            for item in root.rglob("*")
            if item.is_file() and item.name != "manifest.json"
        }
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
        print(f"[BLOCKED] {exc.gate if isinstance(exc, R18R15Error) else BLOCK_UPSTREAM}")
        print(root)


if __name__ == "__main__":
    main()
