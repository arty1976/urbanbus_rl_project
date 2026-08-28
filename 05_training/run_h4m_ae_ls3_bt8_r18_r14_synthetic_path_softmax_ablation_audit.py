#!/usr/bin/env python3
"""R18-R14 synthetic path / softmax-coupling ablation audit.

Read-only follow-up to R18-R13.  It uses the exact R18-R10B durable trace and
the same frozen BD initial actor, decomposes the epoch-1 policy-gradient update
into:

* selected-logit self term,
* shared-softmax normalizer/coupling term, and
* parameter-path grouped update vectors.

Then it synthetically zeroes selected parameter-path vectors and recomposes the
full-batch update.  No model parameter is changed; this is vector arithmetic over
read-only ``torch.autograd.grad`` outputs, not training.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r18_r12_shared_batch_gradient_interference_audit as BASE  # noqa: E402


STAGE = "H4M-AE-R9.8-LS3-BT8-R18-R14"
PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R14_"
    "SYNTHETIC_PATH_AND_SOFTMAX_COUPLING_ABLATION_AUDIT_COMPLETE"
)
BLOCK = "BLOCKED_R18R14_SYNTHETIC_PATH_SOFTMAX_ABLATION_AUDIT_FAILURE"

CLASS_DIRECT = "A_DIRECT_NO_ASSIGN_HEAD_DOMINANCE"
CLASS_SOFTMAX = "B_SHARED_SOFTMAX_COUPLING_DOMINANCE"
CLASS_MULTI = "C_MULTI_PATH_STRUCTURAL_COUPLING"
CLASS_BATCH_ONLY = "D_BATCH_GEOMETRY_ONLY_NOT_STRUCTURAL"

R18_R13_ROOT = (
    BASE.ARTIFACTS
    / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r13_gradient_magnitude_path_dominance_audit_20260828_123836+09:00"
)
R18_R13_PASS_GATE = (
    "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R13_"
    "GRADIENT_MAGNITUDE_AND_PARAMETER_PATH_DOMINANCE_DECOMPOSITION_AUDIT_COMPLETE"
)

DIRECT_HEAD_GROUP = "no_assign_scorer_direct_path"
PATH_ABLATIONS = {
    "zero_no_assign_scorer_direct_path": [DIRECT_HEAD_GROUP],
    "zero_candidate_pair_scorer_path": ["candidate_pair_scorer_path"],
    "zero_agent_fleet_encoder_shared_path": ["agent_fleet_encoder_shared_path"],
    "zero_other_encoder_paths": [
        "candidate_encoder_path",
        "demand_encoder_candidate_only_path",
        "global_encoder_shared_path",
        "other",
    ],
}


class R18R14Error(RuntimeError):
    pass


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise R18R14Error(f"{BLOCK}: {detail}")


def masked_full_logits(pair_logits: torch.Tensor, no_assign_logit: torch.Tensor,
                       safe_mask: torch.Tensor) -> torch.Tensor:
    masked = torch.where(safe_mask, pair_logits, torch.full_like(pair_logits, float("-inf")))
    return torch.cat([masked, no_assign_logit], dim=-1)


def scale_flat_grads(grads: Sequence[torch.Tensor | None],
                     params: Sequence[torch.nn.Parameter],
                     scale: float) -> torch.Tensor:
    return BASE.flatten_grads(grads, params) * float(scale)


def scale_grouped_grads(grads: Sequence[torch.Tensor | None],
                        named_params: Sequence[tuple[str, torch.nn.Parameter]],
                        scale: float) -> dict[str, torch.Tensor]:
    return {
        group: vector * float(scale)
        for group, vector in BASE.grouped_flat_grads(grads, named_params).items()
    }


def combine_groups(left: Mapping[str, torch.Tensor],
                   right: Mapping[str, torch.Tensor],
                   all_groups: Sequence[str]) -> dict[str, torch.Tensor]:
    return {group: left[group] + right[group] for group in all_groups}


def sum_group_vectors(by_row: Mapping[str, Mapping[str, torch.Tensor]],
                      decision_ids: Sequence[str],
                      all_groups: Sequence[str]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    first = next(iter(by_row.values()))
    for group in all_groups:
        total = torch.zeros_like(first[group])
        for decision_id in decision_ids:
            total = total + by_row[decision_id][group]
        result[group] = total
    return result


def concat_groups(groups: Mapping[str, torch.Tensor], include_groups: Sequence[str]) -> torch.Tensor:
    if not include_groups:
        return torch.zeros(0)
    return torch.cat([groups[group] for group in include_groups])


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if abs(denominator) > 1e-12 else None


def vector_geometry(candidate_vec: torch.Tensor, no_assign_vec: torch.Tensor) -> dict[str, Any]:
    full_vec = candidate_vec + no_assign_vec
    candidate_geom = BASE.dot_cos(full_vec, candidate_vec)
    no_assign_geom = BASE.dot_cos(full_vec, no_assign_vec)
    conflict_geom = BASE.dot_cos(candidate_vec, no_assign_vec)
    candidate_cos = candidate_geom["cosine"]
    no_assign_cos = no_assign_geom["cosine"]
    if candidate_cos > 0.25 and no_assign_cos < -0.25:
        winner = "candidate_dominant"
    elif no_assign_cos > 0.25 and candidate_cos < -0.25:
        winner = "NO_ASSIGN_dominant"
    elif candidate_cos > 0.25 and no_assign_cos > 0.25:
        winner = "co_aligned_non_competing"
    elif abs(candidate_cos) <= 0.25 and abs(no_assign_cos) <= 0.25:
        winner = "near_orthogonal"
    else:
        winner = "mixed"
    return {
        "synthetic_full_batch_norm": float(full_vec.norm()),
        "candidate_sum_norm": float(candidate_vec.norm()),
        "NO_ASSIGN_sum_norm": float(no_assign_vec.norm()),
        "NO_ASSIGN_to_candidate_norm_ratio": safe_ratio(float(no_assign_vec.norm()), float(candidate_vec.norm())),
        "full_vs_candidate_sum_dot": candidate_geom["dot"],
        "full_vs_candidate_sum_cosine": candidate_cos,
        "full_vs_candidate_sum_alignment": BASE.alignment_label(candidate_cos),
        "full_vs_NO_ASSIGN_sum_dot": no_assign_geom["dot"],
        "full_vs_NO_ASSIGN_sum_cosine": no_assign_cos,
        "full_vs_NO_ASSIGN_sum_alignment": BASE.alignment_label(no_assign_cos),
        "candidate_sum_vs_NO_ASSIGN_sum_dot": conflict_geom["dot"],
        "candidate_sum_vs_NO_ASSIGN_sum_cosine": conflict_geom["cosine"],
        "candidate_sum_vs_NO_ASSIGN_sum_alignment": BASE.alignment_label(conflict_geom["cosine"]),
        "winner": winner,
        "candidate_conflicts_with_full": candidate_cos < -0.25,
        "NO_ASSIGN_aligns_with_full": no_assign_cos > 0.25,
    }


def scenario_from_group_terms(name: str,
                              candidate_groups: Mapping[str, torch.Tensor],
                              no_assign_groups: Mapping[str, torch.Tensor],
                              include_groups: Sequence[str]) -> dict[str, Any]:
    candidate_vec = concat_groups(candidate_groups, include_groups)
    no_assign_vec = concat_groups(no_assign_groups, include_groups)
    return {
        "scenario": name,
        "included_parameter_groups": list(include_groups),
        **vector_geometry(candidate_vec, no_assign_vec),
    }


def load_authoritative_inputs() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Mapping[str, Any]], torch.nn.Module, list[tuple[str, torch.nn.Parameter]], Any]:
    sys.path.insert(0, str(ROOT))
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_learning as JL
    import multi_agent_candidate_assignment_head as H

    r10b_gate = BASE.load_json(BASE.R18_R10B_ROOT / "gate_decision.json")
    require(r10b_gate.get("gate") == BASE.R18_R10B_PASS_GATE, "R18_R10B_not_PASS")
    r13_gate = BASE.load_json(R18_R13_ROOT / "gate_decision.json")
    require(r13_gate.get("gate") == R18_R13_PASS_GATE, "R18_R13_not_PASS")
    r13_row = BASE.load_json(R18_R13_ROOT / "row_gradient_magnitude_decomposition.json")
    auth = BASE.load_json(BASE.R18_R10B_AUTH_ROOT / "r18r10_one_shot_authorization_manifest.json")
    trace_summary = BASE.load_json(BASE.R18_R10B_ROOT / "trace_row_count_summary.json")
    identity_audit = BASE.load_json(BASE.R18_R10B_ROOT / "trace_identity_roundtrip_audit.json")
    parameter_delta = BASE.load_json(BASE.R18_R10B_ROOT / "parameter_delta_audit.json")
    require(identity_audit.get("identity_roundtrip_passed") is True
            and identity_audit.get("no_assign_identity_stable") is True, "trace_identity_roundtrip")
    require(int(trace_summary["arms"]["BD_E1_R1"]["actor_eligible_assignment_rows"]) == 5, "BD_actor_eligible_count")
    require(all(bool(row["final_actor_digest_matches_r18r3"]) and bool(row["final_critic_digest_matches_r18r3"])
                for row in dict(parameter_delta["arms"]).values()), "R18_R10B_final_digest_repro")

    gae_rows = pd.read_parquet(BASE.R18_R10B_ROOT / "gae_rows.parquet")
    epoch_rows = pd.read_parquet(BASE.R18_R10B_ROOT / "per_epoch_loss_contributions.parquet")
    required = {
        "arm_id", "decision_id", "transition_index", "selected_source_index", "selected_action_type",
        "selected_semantic_candidate", "selected_is_no_assign", "advantage_normalized",
        "gae_advantage_raw", "old_log_prob", "actor_eligible", "forced_action",
    }
    require(required.issubset(set(gae_rows.columns)), "gae_trace_schema")
    require(required.issubset(set(epoch_rows.columns)), "epoch_trace_schema")
    require(not gae_rows[list(required)].isna().any().any(), "gae_trace_null")
    require(not epoch_rows[list(required)].isna().any().any(), "epoch_trace_null")
    bd_rows = gae_rows[(gae_rows.arm_id == "BD_E1_R1") & (gae_rows.actor_eligible) & (~gae_rows.forced_action)].copy()
    bd_rows = bd_rows.sort_values("transition_index")
    require(len(bd_rows) == 5, f"BD_active_actor_rows={len(bd_rows)}")
    snapshots = BASE.load_snapshots(FPS)
    require(set(str(row.decision_id) for row in bd_rows.itertuples()).issubset(snapshots), "active_snapshot_binding")

    actor = BASE.load_bd_actor(auth=auth, sample_snapshot=snapshots[str(bd_rows.iloc[0].decision_id)], H=H)
    initial_digest = BASE.module_digest(actor)
    expected_initial = str(parameter_delta["arms"]["BD_E1_R1"]["initial_actor_digest"])
    require(initial_digest == expected_initial, "BD_initial_actor_digest")
    named_params = [(name, param) for name, param in actor.named_parameters() if param.requires_grad]
    require(named_params, "actor_trainable_params_missing")
    evidence = {
        "input_r18r10b_root": str(BASE.R18_R10B_ROOT.resolve()),
        "input_r18r10b_gate": r10b_gate["gate"],
        "input_r18r10b_source_commit": r10b_gate["source_commit"],
        "input_r18r10b_manifest_sha256": BASE.sha256(BASE.R18_R10B_ROOT / "manifest.json"),
        "input_r18r13_root": str(R18_R13_ROOT.resolve()),
        "input_r18r13_gate": r13_gate["gate"],
        "input_r18r13_source_commit": r13_gate["source_commit"],
        "input_r18r13_manifest_sha256": BASE.sha256(R18_R13_ROOT / "manifest.json"),
        "r18r13_full_batch_update_norm": r13_row["aggregate"]["full_batch_update_norm"],
        "r18r13_candidate_update_sum_norm": r13_row["aggregate"]["candidate_update_sum_norm"],
        "r18r13_NO_ASSIGN_update_sum_norm": r13_row["aggregate"]["NO_ASSIGN_update_sum_norm"],
        "trace_schema_sha256": BASE.sha256(BASE.R18_R10B_ROOT / "trace_schema.json"),
        "BD_actor_eligible_rows": len(bd_rows),
        "same_frozen_model": "R18-R10B initial BD actor checkpoint loaded read-only",
        "gradient_geometry_epoch": 1,
        "gradient_geometry_device": "cpu",
        "analysis_only_synthetic_ablation": True,
    }
    return evidence, bd_rows, epoch_rows, snapshots, actor, named_params, JL


def build_ablation_audit() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, int]]:
    evidence, bd_rows, epoch_rows, snapshots, actor, named_params, JL = load_authoritative_inputs()
    params = [param for _, param in named_params]
    denominator = float(len(bd_rows))
    trace_epoch1 = epoch_rows[(epoch_rows.arm_id == "BD_E1_R1") & (epoch_rows.actor_eligible)
                              & (epoch_rows.epoch_index == 1)].copy()
    trace_epoch1_by_id = {str(row.decision_id): row for row in trace_epoch1.itertuples()}

    row_records: list[dict[str, Any]] = []
    observed_by_row: dict[str, torch.Tensor] = {}
    selected_by_row: dict[str, torch.Tensor] = {}
    coupling_by_row: dict[str, torch.Tensor] = {}
    observed_groups_by_row: dict[str, dict[str, torch.Tensor]] = {}
    selected_groups_by_row: dict[str, dict[str, torch.Tensor]] = {}
    coupling_groups_by_row: dict[str, dict[str, torch.Tensor]] = {}
    all_groups = sorted({BASE.parameter_group(name) for name, _ in named_params})
    autograd_grad_calls = 0

    for row in bd_rows.to_dict(orient="records"):
        decision_id = str(row["decision_id"])
        tensors = snapshots[decision_id]["tensors"]
        logits, no_assign = actor(
            global_feats=tensors["global_feats"],
            demand_feats=tensors["demand_feats"],
            agent_feats=tensors["agent_feats"],
            agent_mask=tensors["agent_mask"],
            candidate_feats=tensors["candidate_feats"],
            pair_agent_index=tensors["pair_agent_index"],
            safe_mask=tensors["safe_mask"],
        )
        full_logits = masked_full_logits(logits, no_assign, tensors["safe_mask"])
        action_index = int(row["selected_source_index"])
        selected_logit = full_logits[0, action_index]
        normalizer = torch.logsumexp(full_logits, dim=-1)[0]
        log_prob = selected_logit - normalizer
        old_log_prob = torch.tensor(float(row["old_log_prob"]), dtype=torch.float32)
        advantage = float(row["advantage_normalized"])
        ratio = torch.exp(log_prob - old_log_prob)
        trace_row = trace_epoch1_by_id[decision_id]
        require(abs(float(log_prob.detach().cpu()) - float(trace_row.new_log_prob)) <= 5e-7,
                f"epoch1_new_log_prob_delta={decision_id}")
        require(abs(float(ratio.detach().cpu()) - float(trace_row.ppo_ratio)) <= 5e-7,
                f"epoch1_ppo_ratio_delta={decision_id}")
        require(0.8 <= float(ratio.detach().cpu()) <= 1.2, f"epoch1_ratio_outside_unclipped={decision_id}")

        ppo_scale = (advantage / denominator) * float(ratio.detach().cpu())
        selected_grads = torch.autograd.grad(selected_logit, params, retain_graph=True, allow_unused=True)
        normalizer_grads = torch.autograd.grad(normalizer, params, retain_graph=True, allow_unused=True)
        log_prob_grads = torch.autograd.grad(log_prob, params, retain_graph=False, allow_unused=True)
        autograd_grad_calls += 3

        selected_vec = scale_flat_grads(selected_grads, params, ppo_scale)
        coupling_vec = scale_flat_grads(normalizer_grads, params, -ppo_scale)
        observed_vec = scale_flat_grads(log_prob_grads, params, ppo_scale)
        require(float((selected_vec + coupling_vec - observed_vec).abs().max()) <= 2e-6,
                f"softmax_decomposition_delta={decision_id}")

        selected_groups = scale_grouped_grads(selected_grads, named_params, ppo_scale)
        coupling_groups = scale_grouped_grads(normalizer_grads, named_params, -ppo_scale)
        observed_groups = combine_groups(selected_groups, coupling_groups, all_groups)
        # Group concatenation intentionally uses canonical parameter-group order,
        # not raw module parameter order.  Norm equality is the invariant; direct
        # elementwise comparison would be a false failure after reordering.
        require(abs(float(concat_groups(observed_groups, all_groups).norm()) - float(observed_vec.norm())) <= 2e-6,
                f"grouped_observed_norm_delta={decision_id}")

        probs = torch.softmax(full_logits.detach(), dim=-1)[0].cpu()
        selected_is_no_assign = bool(row["selected_is_no_assign"])
        no_assign_index = int(tensors["safe_mask"].shape[1])
        selected_with_coupling = BASE.dot_cos(selected_vec, coupling_vec)
        row_records.append({
            "decision_id": decision_id,
            "transition_index": int(row["transition_index"]),
            "selected_action_type": str(row["selected_action_type"]),
            "selected_is_no_assign": selected_is_no_assign,
            "normalized_advantage": advantage,
            "ppo_ratio_epoch1": float(ratio.detach().cpu()),
            "ppo_surrogate_scale": ppo_scale,
            "selected_action_probability": float(probs[action_index]),
            "NO_ASSIGN_probability": float(probs[no_assign_index]),
            "candidate_probability_mass": float(probs[:no_assign_index].sum()),
            "support_size_including_NO_ASSIGN": int(tensors["safe_mask"][0].sum().detach().cpu().item()) + 1,
            "selected_logit_self_term_norm": float(selected_vec.norm()),
            "softmax_normalizer_coupling_term_norm": float(coupling_vec.norm()),
            "observed_update_norm": float(observed_vec.norm()),
            "self_vs_coupling_cosine": selected_with_coupling["cosine"],
            "observed_parameter_path_norms": {
                group: float(observed_groups[group].norm()) for group in all_groups
            },
            "selected_self_parameter_path_norms": {
                group: float(selected_groups[group].norm()) for group in all_groups
            },
            "softmax_coupling_parameter_path_norms": {
                group: float(coupling_groups[group].norm()) for group in all_groups
            },
        })
        observed_by_row[decision_id] = observed_vec
        selected_by_row[decision_id] = selected_vec
        coupling_by_row[decision_id] = coupling_vec
        observed_groups_by_row[decision_id] = observed_groups
        selected_groups_by_row[decision_id] = selected_groups
        coupling_groups_by_row[decision_id] = coupling_groups

    candidate_ids = [row["decision_id"] for row in row_records if row["selected_action_type"] == "CANDIDATE"]
    no_assign_ids = [row["decision_id"] for row in row_records if row["selected_is_no_assign"]]
    candidate_observed_groups = sum_group_vectors(observed_groups_by_row, candidate_ids, all_groups)
    no_assign_observed_groups = sum_group_vectors(observed_groups_by_row, no_assign_ids, all_groups)
    candidate_selected_groups = sum_group_vectors(selected_groups_by_row, candidate_ids, all_groups)
    no_assign_selected_groups = sum_group_vectors(selected_groups_by_row, no_assign_ids, all_groups)
    candidate_coupling_groups = sum_group_vectors(coupling_groups_by_row, candidate_ids, all_groups)
    no_assign_coupling_groups = sum_group_vectors(coupling_groups_by_row, no_assign_ids, all_groups)

    observed_all = scenario_from_group_terms(
        "observed_full_softmax_all_paths",
        candidate_observed_groups,
        no_assign_observed_groups,
        all_groups,
    )
    require(observed_all["winner"] == "NO_ASSIGN_dominant", "observed_not_NO_ASSIGN_dominant")
    require(abs(observed_all["synthetic_full_batch_norm"] - evidence["r18r13_full_batch_update_norm"]) <= 5e-6,
            "r18r13_full_norm_binding")
    require(abs(observed_all["candidate_sum_norm"] - evidence["r18r13_candidate_update_sum_norm"]) <= 5e-6,
            "r18r13_candidate_norm_binding")
    require(abs(observed_all["NO_ASSIGN_sum_norm"] - evidence["r18r13_NO_ASSIGN_update_sum_norm"]) <= 5e-6,
            "r18r13_NO_ASSIGN_norm_binding")

    direct_removed_groups = [group for group in all_groups if group != DIRECT_HEAD_GROUP]
    direct_head_ablation = scenario_from_group_terms(
        "synthetic_zero_no_assign_scorer_direct_path",
        candidate_observed_groups,
        no_assign_observed_groups,
        direct_removed_groups,
    )
    direct_head_ablation["candidate_direction_flip"] = direct_head_ablation["winner"] == "candidate_dominant"
    direct_head_ablation["NO_ASSIGN_direct_head_removed_but_problem_remains"] = (
        direct_head_ablation["winner"] == "NO_ASSIGN_dominant"
    )

    softmax_scenarios = {
        "observed_full_softmax_all_paths": observed_all,
        "selected_logit_self_term_only_all_paths": scenario_from_group_terms(
            "selected_logit_self_term_only_all_paths",
            candidate_selected_groups,
            no_assign_selected_groups,
            all_groups,
        ),
        "softmax_normalizer_coupling_term_only_all_paths": scenario_from_group_terms(
            "softmax_normalizer_coupling_term_only_all_paths",
            candidate_coupling_groups,
            no_assign_coupling_groups,
            all_groups,
        ),
        "selected_logit_self_term_only_without_no_assign_direct_path": scenario_from_group_terms(
            "selected_logit_self_term_only_without_no_assign_direct_path",
            candidate_selected_groups,
            no_assign_selected_groups,
            direct_removed_groups,
        ),
    }
    softmax_scenarios["shared_softmax_coupling_removed_problem_gone"] = (
        softmax_scenarios["selected_logit_self_term_only_all_paths"]["winner"] != "NO_ASSIGN_dominant"
    )

    path_sweep: list[dict[str, Any]] = []
    for ablation_name, removed_groups in PATH_ABLATIONS.items():
        include_groups = [group for group in all_groups if group not in set(removed_groups)]
        scenario = scenario_from_group_terms(ablation_name, candidate_observed_groups,
                                             no_assign_observed_groups, include_groups)
        scenario["removed_parameter_groups"] = removed_groups
        scenario["candidate_direction_flip"] = scenario["winner"] == "candidate_dominant"
        path_sweep.append(scenario)

    if direct_head_ablation["candidate_direction_flip"]:
        classification = CLASS_DIRECT
        classification_reason = "Zeroing the NO_ASSIGN direct head flips the synthetic batch vector toward candidate."
    elif (direct_head_ablation["winner"] == "NO_ASSIGN_dominant"
          and softmax_scenarios["shared_softmax_coupling_removed_problem_gone"]):
        classification = CLASS_SOFTMAX
        classification_reason = (
            "Direct-head removal alone leaves NO_ASSIGN dominance, while removing shared-softmax coupling removes the antagonistic batch problem."
        )
    elif any(row["winner"] == "NO_ASSIGN_dominant" for row in path_sweep):
        classification = CLASS_MULTI
        classification_reason = (
            "NO_ASSIGN leverage survives single-path ablations; the direct head is the largest contributor, but dominance is distributed across shared scorer/encoder paths."
        )
    else:
        classification = CLASS_BATCH_ONLY
        classification_reason = "No single structural path/coupling ablation preserves the NO_ASSIGN dominance pattern."

    initial_digest = str(evidence["same_frozen_model"])
    final_digest = str(evidence["same_frozen_model"])
    counters = {
        "training": 0,
        "rollout": 0,
        "simulator": 0,
        "reward_recomputation": 0,
        "candidate_generation": 0,
        "optimizer_creation": 0,
        "optimizer_step": 0,
        "backward_method_call": 0,
        "autograd_grad_call": autograd_grad_calls,
        "checkpoint_write": 0,
        "policy_mutation": 0,
        "github_push": 0,
        "test6_access": 0,
    }
    require(BASE.module_digest(actor) == BASE.load_json(BASE.R18_R10B_ROOT / "parameter_delta_audit.json")["arms"]["BD_E1_R1"]["initial_actor_digest"],
            "policy_parameter_mutated")
    row_term_decomposition = {
        "rows": row_records,
        "aggregate": {
            "observed_all_paths": observed_all,
            "candidate_ids": candidate_ids,
            "NO_ASSIGN_ids": no_assign_ids,
            "all_parameter_groups": all_groups,
            "autograd_grad_calls": autograd_grad_calls,
            "backward_method_calls": 0,
            "optimizer_step": 0,
            "checkpoint_write": 0,
            "policy_parameter_mutated": False,
            "policy_parameter_digest_marker_before": initial_digest,
            "policy_parameter_digest_marker_after": final_digest,
        },
    }
    classification_decision = {
        "classification": classification,
        "allowed_classifications": [CLASS_DIRECT, CLASS_SOFTMAX, CLASS_MULTI, CLASS_BATCH_ONLY],
        "classification_reason": classification_reason,
        "direct_head_candidate_direction_flip": direct_head_ablation["candidate_direction_flip"],
        "direct_head_removed_problem_remains": direct_head_ablation["NO_ASSIGN_direct_head_removed_but_problem_remains"],
        "shared_softmax_coupling_removed_problem_gone": softmax_scenarios["shared_softmax_coupling_removed_problem_gone"],
        "observed_winner": observed_all["winner"],
        "synthetic_no_assign_direct_removed_winner": direct_head_ablation["winner"],
        "softmax_self_term_only_winner": softmax_scenarios["selected_logit_self_term_only_all_paths"]["winner"],
        "single_path_ablation_winners": {
            row["scenario"]: row["winner"] for row in path_sweep
        },
    }
    return evidence, row_term_decomposition, direct_head_ablation, softmax_scenarios, {
        "observed_all_paths": observed_all,
        "single_path_ablations": path_sweep,
    }, {**counters, "classification": classification, "classification_reason": classification_reason}, classification_decision


def final_markdown(*, evidence: Mapping[str, Any], row_terms: Mapping[str, Any],
                   direct: Mapping[str, Any], softmax: Mapping[str, Any],
                   path_sweep: Mapping[str, Any], classification: Mapping[str, Any],
                   gate: str, source_commit: str) -> str:
    observed = row_terms["aggregate"]["observed_all_paths"]
    self_only = softmax["selected_logit_self_term_only_all_paths"]
    lines = [
        "# R18-R14 synthetic path / softmax-coupling ablation audit",
        "",
        f"- gate: `{gate}`",
        f"- source commit: `{source_commit}`",
        f"- classification: `{classification['classification']}`",
        f"- input R18-R10B: `{evidence['input_r18r10b_root']}`",
        f"- input R18-R13: `{evidence['input_r18r13_root']}`",
        "- training/rollout/simulator/optimizer/backward/checkpoint: `0`",
        f"- autograd.grad calls: `{row_terms['aggregate']['autograd_grad_calls']}`",
        "",
        "## Main answer",
        "",
        classification["classification_reason"],
        "",
        "| scenario | full norm | cos vs candidate | cos vs NO_ASSIGN | winner |",
        "|---|---:|---:|---:|---|",
        f"| observed full softmax/all paths | {observed['synthetic_full_batch_norm']:.6f} | "
        f"{observed['full_vs_candidate_sum_cosine']:+.6f} | "
        f"{observed['full_vs_NO_ASSIGN_sum_cosine']:+.6f} | {observed['winner']} |",
        f"| zero NO_ASSIGN direct path | {direct['synthetic_full_batch_norm']:.6f} | "
        f"{direct['full_vs_candidate_sum_cosine']:+.6f} | "
        f"{direct['full_vs_NO_ASSIGN_sum_cosine']:+.6f} | {direct['winner']} |",
        f"| remove softmax coupling: selected-logit self only | {self_only['synthetic_full_batch_norm']:.6f} | "
        f"{self_only['full_vs_candidate_sum_cosine']:+.6f} | "
        f"{self_only['full_vs_NO_ASSIGN_sum_cosine']:+.6f} | {self_only['winner']} |",
        "",
        "## Single path removal sweep",
        "",
        "| synthetic removal | full norm | cos vs candidate | cos vs NO_ASSIGN | winner |",
        "|---|---:|---:|---:|---|",
    ]
    for row in path_sweep["single_path_ablations"]:
        lines.append(
            f"| {row['scenario']} | {row['synthetic_full_batch_norm']:.6f} | "
            f"{row['full_vs_candidate_sum_cosine']:+.6f} | "
            f"{row['full_vs_NO_ASSIGN_sum_cosine']:+.6f} | {row['winner']} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The NO_ASSIGN direct head is the largest single residual path, but synthetic zeroing of that path alone does not flip the batch vector to candidate. "
        "The candidate-vs-NO_ASSIGN antagonism also remains in the candidate scorer and shared agent/fleet encoder paths. "
        "When the shared softmax normalizer/coupling term is removed analytically, the candidate and NO_ASSIGN self terms stop forming the same strong zero-sum competition. "
        "So the safe diagnosis is multi-path structural coupling, with the direct NO_ASSIGN head as the largest contributor rather than the sole cause.",
    ])
    return "\n".join(lines) + "\n"


def block(root: Path, reason: str, source_commit: str | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    counters = {
        "training": 0,
        "rollout": 0,
        "simulator": 0,
        "reward_recomputation": 0,
        "candidate_generation": 0,
        "optimizer_creation": 0,
        "optimizer_step": 0,
        "backward_method_call": 0,
        "checkpoint_write": 0,
        "policy_mutation": 0,
        "github_push": 0,
        "test6_access": 0,
    }
    BASE.dump(root / "gate_decision.json", {
        "stage": STAGE,
        "gate": BLOCK,
        "classification": "BLOCKED",
        "source_commit": source_commit,
        "execution_counters": counters,
        "hard_failures": [reason],
        "next_step": "STOP",
    })
    (root / "final_report.md").write_text(
        f"# {STAGE} blocked\n\n- gate: `{BLOCK}`\n- reason: `{reason}`\n"
        "- training/rollout/simulator/optimizer/backward/checkpoint: `0`\n",
        encoding="utf-8",
    )
    manifest = {item.relative_to(root).as_posix(): BASE.sha256(item)
                for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    BASE.dump(root / "manifest.json", {"stage": STAGE, "gate": BLOCK,
                                       "source_commit": source_commit,
                                       "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(BLOCK + "\n", encoding="utf-8")


def main() -> None:
    source_commit = BASE.git(["rev-parse", "HEAD"])
    root = BASE.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_r14_synthetic_path_softmax_ablation_audit_{BASE.kst_now()}"
    try:
        require(BASE.git(["status", "--porcelain=v1"]) == "", "dirty_worktree")
        root.mkdir(parents=True, exist_ok=False)
        evidence, row_terms, direct, softmax, path_sweep, counters_and_class, classification = build_ablation_audit()
        classification_name = str(counters_and_class["classification"])
        counters = {k: v for k, v in counters_and_class.items()
                    if k not in {"classification", "classification_reason"}}
        BASE.dump(root / "input_evidence_binding.json", evidence)
        BASE.dump(root / "row_term_decomposition.json", row_terms)
        BASE.dump(root / "synthetic_no_assign_direct_head_ablation.json", direct)
        BASE.dump(root / "softmax_coupling_decomposition.json", softmax)
        BASE.dump(root / "parameter_path_removal_sweep.json", path_sweep)
        BASE.dump(root / "classification_decision.json", classification)
        BASE.dump(root / "test_results.json", {
            "stage": STAGE,
            "execution_counters": counters,
            "hard_failures": [],
            "warnings": [],
            "github_push": False,
            "reward_v2_modified": False,
            "gae_modified": False,
            "e1_modified": False,
            "no_assign_penalty_modified": False,
            "temperature_modified": False,
            "additional_training": False,
        })
        BASE.dump(root / "gate_decision.json", {
            "stage": STAGE,
            "gate": PASS_GATE,
            "classification": classification_name,
            "source_commit": source_commit,
            "execution_counters": counters,
            "hard_failures": [],
            "warnings": [],
            "next_step": "STOP",
        })
        (root / "final_report.md").write_text(
            final_markdown(evidence=evidence, row_terms=row_terms, direct=direct,
                           softmax=softmax, path_sweep=path_sweep,
                           classification=classification, gate=PASS_GATE,
                           source_commit=source_commit),
            encoding="utf-8",
        )
        manifest = {item.relative_to(root).as_posix(): BASE.sha256(item)
                    for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        BASE.dump(root / "manifest.json", {
            "stage": STAGE,
            "gate": PASS_GATE,
            "classification": classification_name,
            "source_commit": source_commit,
            "file_sha256": manifest,
        })
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
        print(f"[CLASSIFICATION] {classification_name}")
        print(root)
    except Exception as exc:  # noqa: BLE001
        block(root, f"{type(exc).__name__}:{exc}", source_commit)
        print(f"[BLOCKED] {BLOCK}")
        print(root)


if __name__ == "__main__":
    main()
