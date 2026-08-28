#!/usr/bin/env python3
"""Durable per-row credit/PPO trace helpers for the R18 bounded executor.

This module is intentionally observational: it serializes already-computed
rollout, GAE, normalization, and PPO quantities.  It does not create an
optimizer, run backward, recompute rewards, regenerate candidates, or change the
loss tensors used by the executor.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch


TRACE_CONTRACT_ID = "R18_R6_DURABLE_PER_ROW_CREDIT_PPO_TRACE_V1"
NO_ASSIGN_SEMANTIC_IDENTITY = "NO_ASSIGN_KEEP_CURRENT_PLANS"

TRACE_PARQUET_FILES = {
    "eligible_row_credit_trace": "eligible_row_credit_trace.parquet",
    "credit_eligibility_rows": "credit_eligibility_rows.parquet",
    "gae_rows": "gae_rows.parquet",
    "ppo_ratio_rows": "ppo_ratio_rows.parquet",
    "per_epoch_loss_contributions": "per_epoch_loss_contributions.parquet",
}

REQUIRED_TRACE_FIELDS = [
    "arm_id",
    "window_id",
    "time_band",
    "trajectory_id",
    "decision_id",
    "selected_semantic_candidate",
    "executed_semantic_candidate",
    "credited_semantic_candidate",
    "NO_ASSIGN_semantic_identity",
    "candidate_support_digest",
    "selected_source_index",
    "reward_ancestry_class",
    "reward_ancestry_source_ids",
    "assignment_reward",
    "return_raw",
    "gae_advantage_raw",
    "advantage_normalized",
    "old_log_prob",
    "new_log_prob",
    "ppo_ratio",
    "clip_low",
    "clip_high",
    "was_clipped",
    "actor_eligible",
    "critic_eligible",
    "policy_loss_unclipped",
    "policy_loss_clipped",
    "selected_policy_loss",
    "entropy_contribution",
    "epoch_index",
    "selected_candidate_logit_pre",
    "selected_candidate_probability_pre",
    "NO_ASSIGN_logit_pre",
    "NO_ASSIGN_probability_pre",
    "selected_candidate_logit_post_epoch",
    "selected_candidate_probability_post_epoch",
    "NO_ASSIGN_logit_post_epoch",
    "NO_ASSIGN_probability_post_epoch",
    "relative_logit_delta_vs_NO_ASSIGN",
    "relative_probability_delta_vs_NO_ASSIGN",
    "per_row_gradient_vector_persisted",
]

OPTIONAL_FACTORIZED_TRACE_FIELDS = [
    "factorized_actor_contract_id",
    "stage1_assign_log_prob",
    "stage1_assign_probability",
    "stage1_no_assign_log_prob",
    "stage1_no_assign_probability",
    "stage1_selected_log_prob",
    "stage2_selected_candidate_log_prob",
    "stage2_selected_candidate_probability",
    "stage2_candidate_probability",
    "stage2_selected_log_prob",
    "stage2_conditional_candidate_probability_sum",
    "reconstructed_final_log_prob",
    "reconstructed_final_probability",
    "stage1_loss_contribution",
    "stage2_loss_contribution",
    "stage1_gradient_norm",
    "stage2_gradient_norm",
    "shared_encoder_gradient_norm",
]

STABLE_IDENTITY_FIELDS = [
    "arm_id",
    "environment_seed",
    "window_id",
    "time_band",
    "trajectory_id",
    "decision_id",
    "transition_index",
    "semantic_selected_candidate_id",
]


class TraceContractError(RuntimeError):
    """Fail-closed error for a mandatory trace contract omission."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def trace_schema_payload() -> dict[str, Any]:
    return {
        "contract_id": TRACE_CONTRACT_ID,
        "schema_version": 1,
        "stable_identity_fields": list(STABLE_IDENTITY_FIELDS),
        "required_trace_fields": list(REQUIRED_TRACE_FIELDS),
        "optional_factorized_trace_fields": list(OPTIONAL_FACTORIZED_TRACE_FIELDS),
        "artifact_files": dict(TRACE_PARQUET_FILES),
        "serialization_notes": {
            "reward_ancestry_source_ids": "JSON array string, not recomputed",
            "per_row_gradient_vector_persisted": False,
            "probabilities": "observational detached forward quantities",
            "factorized_fields": "optional R18-R16 observational ASSIGN gate / conditional candidate / reconstructed final log-probability quantities",
        },
        "invariants": {
            "instrumentation_only": True,
            "training_semantics_changed": False,
            "ppo_semantics_changed": False,
            "gae_semantics_changed": False,
            "reward_v2_changed": False,
            "zero_loss_changed": False,
            "candidate_semantics_changed": False,
            "no_optimizer_or_backward_inside_trace_module": True,
        },
    }


def contract_payload(*, source_commit: str | None = None, source_sha256: Mapping[str, Any] | None = None,
                     training_authorized: bool = False, bounded_rerun_authorized: bool = False) -> dict[str, Any]:
    return {
        "contract_id": TRACE_CONTRACT_ID,
        "instrumentation_only": True,
        "training_semantics_changed": False,
        "PPO_semantics_changed": False,
        "GAE_semantics_changed": False,
        "Reward_V2_changed": False,
        "Zero_Loss_changed": False,
        "candidate_semantics_changed": False,
        "training_authorized": bool(training_authorized),
        "bounded_rerun_authorized": bool(bounded_rerun_authorized),
        "training_allowed": False,
        "rollout_allowed": False,
        "simulator_allowed": False,
        "candidate_generation_allowed": False,
        "reward_recomputation_allowed": False,
        "optimizer_creation_allowed": False,
        "optimizer_step_allowed": False,
        "backward_allowed": False,
        "checkpoint_write_allowed": False,
        "policy_mutation_allowed": False,
        "per_row_gradient_vector_persisted": False,
        "new_source_commit": source_commit,
        "new_source_sha256": dict(source_sha256 or {}),
        "future_trace_artifacts": dict(TRACE_PARQUET_FILES),
        "optional_factorized_trace_fields": list(OPTIONAL_FACTORIZED_TRACE_FIELDS),
    }


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise TraceContractError(code, detail)


def _nonnull(value: Any, code: str, detail: str) -> Any:
    _require(value is not None and str(value) != "", code, detail)
    return value


def _finite_float(value: Any, code: str, detail: str) -> float:
    try:
        result = float(value)
    except Exception as exc:  # noqa: BLE001
        raise TraceContractError(code, detail) from exc
    _require(math.isfinite(result), code, detail)
    return result


def _json_array(values: Sequence[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False, sort_keys=True, default=str)


def ppo_ratio_from_log_probs(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor) -> torch.Tensor:
    """Return the exact PPO ratio formula used by ``assignment_ppo_loss``."""
    return torch.exp(new_log_prob - old_log_prob.detach())


def classify_clip(*, ratio: float, advantage: float, clip_epsilon: float) -> bool:
    """PPO clip branch actually selected by min(unclipped, clipped)."""
    low, high = 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon)
    return bool((float(advantage) > 0.0 and float(ratio) > high) or (float(advantage) < 0.0 and float(ratio) < low))


def _transition(row: Mapping[str, Any]) -> Any:
    transition = row.get("t")
    _require(transition is not None, "TRACE_TRANSITION_MISSING", "rollout row has no assignment transition")
    return transition


def _plan(row: Mapping[str, Any]) -> Any:
    plan = row.get("plan")
    _require(plan is not None, "TRACE_PLAN_MISSING", "rollout row has no candidate plan")
    return plan


def _raw_plan_ids(plan: Any) -> tuple[str, str, str]:
    return str(plan.selected_candidate_id), str(plan.applied_candidate_id), str(plan.credited_candidate_id)


def _candidate_semantic(*, transition: Any, raw_candidate_id: str, plan: Any) -> str:
    if raw_candidate_id == NO_ASSIGN_SEMANTIC_IDENTITY or bool(getattr(plan, "no_assign", False)):
        return NO_ASSIGN_SEMANTIC_IDENTITY
    agent = getattr(transition, "selected_agent_id", None)
    if agent is None:
        events = list(getattr(plan, "events", ()))
        agent = events[0].get("agent_id") if events else None
    _require(agent is not None and str(agent) != "", "TRACE_SELECTED_AGENT_MISSING", str(getattr(transition, "assignment_step_id", "")))
    return f"{agent}::{raw_candidate_id}"


def _identity_digest(row: Mapping[str, Any]) -> str:
    return canonical_sha256({field: row[field] for field in STABLE_IDENTITY_FIELDS})


def _support_digest_from_snapshot(row: Mapping[str, Any]) -> str | None:
    snapshot = row.get("snapshot")
    if snapshot is not None and hasattr(snapshot, "snapshot_digest"):
        return str(snapshot.snapshot_digest)
    loaded = row.get("loaded")
    if isinstance(loaded, Mapping):
        metadata = loaded.get("metadata", {})
        if isinstance(metadata, Mapping) and metadata.get("candidate_support_digest"):
            return str(metadata["candidate_support_digest"])
    return None


def _reward_ancestry_source_ids(*, plan: Any, transition: Any) -> list[str]:
    values: list[str] = []
    for value in (
        getattr(plan, "transition_id", None),
        getattr(plan, "candidate_plan_digest", None),
        getattr(plan, "applied_plan_digest", None),
        getattr(transition, "pre_state_digest", None),
        getattr(transition, "next_state_digest", None),
    ):
        if value is not None and str(value) != "":
            values.append(str(value))
    provenance = getattr(transition, "provenance", {})
    if isinstance(provenance, Mapping):
        for item in provenance.get("operational", []):
            if isinstance(item, Mapping):
                digest = item.get("causal_state_digest")
                if digest is not None and str(digest) != "":
                    values.append(str(digest))
    return values


def build_base_trace_rows(*, model: Mapping[str, Any], prepared: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Join rollout rows, E1 rows, and GAE rows without recomputing them."""
    rollout_rows = list(prepared.get("rows", []))
    e1_rows = list(prepared.get("e1_rows", []))
    gae_rows = list(prepared.get("gae_rows", []))
    _require(len(rollout_rows) == len(e1_rows) == len(gae_rows) and len(rollout_rows) > 0,
             "TRACE_PREPARED_ROW_COUNT_MISMATCH", str(model.get("arm_id", model.get("cell_id", ""))))

    batch = prepared.get("batch", {})
    old_log_prob = batch.get("old_log_prob") if isinstance(batch, Mapping) else None
    action_index = batch.get("action_index") if isinstance(batch, Mapping) else None
    forced_action = batch.get("forced_action") if isinstance(batch, Mapping) else None
    advantage = batch.get("advantage") if isinstance(batch, Mapping) else None
    value_target = batch.get("value_target") if isinstance(batch, Mapping) else None
    actor_mask = batch.get("actor_eligibility_mask") if isinstance(batch, Mapping) else None
    tensors_present = all(isinstance(item, torch.Tensor) for item in (old_log_prob, action_index, forced_action, advantage, value_target, actor_mask))

    arm_id = str(model.get("arm_id", model.get("cell_id", "")))
    _require(bool(arm_id), "TRACE_ARM_ID_MISSING", "model arm_id/cell_id")
    environment_seed = int(model.get("environment_seed", -1))
    _require(environment_seed >= 0, "TRACE_ENVIRONMENT_SEED_MISSING", arm_id)

    result: list[dict[str, Any]] = []
    ancestry_owners: dict[str, str] = {}
    seen_decisions: set[str] = set()
    for index, (rollout_row, e1_row, gae_row) in enumerate(zip(rollout_rows, e1_rows, gae_rows)):
        transition = _transition(rollout_row)
        plan = _plan(rollout_row)
        decision_id = str(_nonnull(getattr(transition, "assignment_step_id", None), "TRACE_STABLE_DECISION_ID_MISSING", arm_id))
        _require(decision_id not in seen_decisions, "TRACE_DUPLICATE_DECISION_ID", decision_id)
        seen_decisions.add(decision_id)
        _require(str(e1_row.get("decision_id", "")) == decision_id and str(gae_row.get("decision_id", "")) == decision_id,
                 "TRACE_DECISION_JOIN_MISMATCH", decision_id)

        selected_id, executed_id, credited_id = _raw_plan_ids(plan)
        _require(selected_id == executed_id == credited_id, "TRACE_SELECTED_EXECUTED_CREDITED_IDENTITY_MISMATCH", decision_id)

        selected_semantic = _candidate_semantic(transition=transition, raw_candidate_id=selected_id, plan=plan)
        executed_semantic = _candidate_semantic(transition=transition, raw_candidate_id=executed_id, plan=plan)
        credited_semantic = _candidate_semantic(transition=transition, raw_candidate_id=credited_id, plan=plan)
        _require(selected_semantic == executed_semantic == credited_semantic,
                 "TRACE_SELECTED_EXECUTED_CREDITED_IDENTITY_MISMATCH", decision_id)

        provenance = getattr(transition, "provenance", {})
        _require(isinstance(provenance, Mapping), "TRACE_PROVENANCE_MISSING", decision_id)
        candidate_support_digest = str(_nonnull(provenance.get("candidate_support_digest"), "TRACE_CANDIDATE_SUPPORT_DIGEST_MISSING", decision_id))
        snapshot_digest = _support_digest_from_snapshot(rollout_row)
        if snapshot_digest is not None:
            _require(snapshot_digest == candidate_support_digest, "TRACE_CANDIDATE_SUPPORT_DIGEST_MISMATCH", decision_id)

        raw_gae = _finite_float(gae_row.get("raw_gae"), "TRACE_RAW_GAE_MISSING", decision_id)
        normalized = _finite_float(gae_row.get("normalized_advantage"), "TRACE_NORMALIZED_ADVANTAGE_MISSING", decision_id)
        reward_component = _finite_float(e1_row.get("reward_gae_component"), "TRACE_REWARD_ANCESTRY_MISSING", decision_id)
        reward_ancestry_class = str(_nonnull(e1_row.get("category"), "TRACE_REWARD_ANCESTRY_CLASS_MISSING", decision_id))
        actor_eligible = bool(e1_row.get("actor_eligible_e1"))
        critic_eligible = bool(e1_row.get("critic_eligible_e1", True))
        if actor_eligible:
            _require(reward_component != 0.0, "TRACE_REWARD_ANCESTRY_MISSING_FOR_ACTOR_ELIGIBLE_ROW", decision_id)
        ancestry_ids = _reward_ancestry_source_ids(plan=plan, transition=transition)
        if actor_eligible:
            _require(bool(ancestry_ids), "TRACE_REWARD_ANCESTRY_SOURCE_IDS_MISSING", decision_id)
            owner_key = ancestry_ids[0]
            previous = ancestry_owners.get(owner_key)
            _require(previous is None, "TRACE_DUPLICATE_REWARD_ANCESTRY_OWNERSHIP", f"{previous}->{decision_id}")
            ancestry_owners[owner_key] = decision_id

        old_log = _finite_float(getattr(transition, "old_log_prob", None), "TRACE_OLD_LOG_PROB_MISSING", decision_id)
        batch_old_log: float | None = None
        batch_action_index: int | None = None
        batch_forced: bool | None = None
        if tensors_present:
            batch_old_log = float(old_log_prob.detach().cpu()[index])
            _require(math.isclose(batch_old_log, old_log, rel_tol=0.0, abs_tol=1e-6), "TRACE_OLD_LOG_PROB_BATCH_MISMATCH", decision_id)
            batch_action_index = int(action_index.detach().cpu()[index])
            batch_forced = bool(forced_action.detach().cpu()[index])
            _require(math.isclose(float(advantage.detach().cpu()[index]), normalized, rel_tol=0.0, abs_tol=1e-6),
                     "TRACE_NORMALIZED_ADVANTAGE_BATCH_MISMATCH", decision_id)
            _require(math.isclose(float(value_target.detach().cpu()[index]), float(gae_row.get("critic_target")), rel_tol=0.0, abs_tol=1e-6),
                     "TRACE_RETURN_BATCH_MISMATCH", decision_id)
            _require(bool(actor_mask.detach().cpu()[index]) == actor_eligible, "TRACE_ACTOR_ELIGIBILITY_BATCH_MISMATCH", decision_id)

        selected_source_index = int(getattr(transition, "action_index"))
        no_assign_source_index = int(getattr(transition, "no_assign_index"))
        base = {
            "arm_id": arm_id,
            "environment_seed": environment_seed,
            "window_id": str(getattr(transition, "window_id")),
            "time_band": str(provenance.get("time_band", rollout_row.get("time_band", ""))),
            "trajectory_id": str(_nonnull(provenance.get("trajectory_id"), "TRACE_TRAJECTORY_ID_MISSING", decision_id)),
            "decision_id": decision_id,
            "transition_index": int(index),
            "semantic_selected_candidate_id": selected_semantic,
            "selected_semantic_candidate": selected_semantic,
            "executed_semantic_candidate": executed_semantic,
            "credited_semantic_candidate": credited_semantic,
            "NO_ASSIGN_semantic_identity": NO_ASSIGN_SEMANTIC_IDENTITY,
            "NO_ASSIGN_source_index": no_assign_source_index,
            "candidate_support_digest": candidate_support_digest,
            "action_support_digest": str(getattr(transition, "action_support_digest", "")),
            "selected_source_index": selected_source_index,
            "ppo_batch_action_index": batch_action_index,
            "selected_raw_candidate_id": selected_id,
            "executed_raw_candidate_id": executed_id,
            "credited_raw_candidate_id": credited_id,
            "selected_agent_id": None if bool(getattr(transition, "selected_is_no_assign", False)) else str(getattr(transition, "selected_agent_id")),
            "selected_is_no_assign": bool(getattr(transition, "selected_is_no_assign", False)),
            "selected_action_type": str(rollout_row.get("action_type", e1_row.get("selected_action_type", ""))),
            "reward_ancestry_class": reward_ancestry_class,
            "reward_ancestry_source_ids": _json_array(ancestry_ids),
            "assignment_reward": _finite_float(gae_row.get("reward"), "TRACE_ASSIGNMENT_REWARD_MISSING", decision_id),
            "return_raw": _finite_float(gae_row.get("critic_target"), "TRACE_RETURN_MISSING", decision_id),
            "gae_advantage_raw": raw_gae,
            "advantage_normalized": normalized,
            "old_log_prob": old_log,
            "actor_eligible": actor_eligible,
            "critic_eligible": critic_eligible,
            "forced_action": bool(getattr(transition, "forced_action", batch_forced if batch_forced is not None else False)),
            "per_row_gradient_vector_persisted": False,
        }
        base["identity_digest"] = _identity_digest(base)
        result.append(base)
    return result


def _selected_tensor_values(*, values: torch.Tensor, no_assign: torch.Tensor, action_index: torch.Tensor) -> torch.Tensor:
    row_index = torch.arange(values.shape[0], device=values.device)
    no_assign_column = values.shape[1]
    selected = torch.empty((values.shape[0],), dtype=values.dtype, device=values.device)
    is_no_assign = action_index == no_assign_column
    if bool((~is_no_assign).any().item()):
        selected[~is_no_assign] = values[row_index[~is_no_assign], action_index[~is_no_assign]]
    if bool(is_no_assign.any().item()):
        selected[is_no_assign] = no_assign[is_no_assign, 0]
    return selected


def actor_batch_forward_no_grad(actor: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Detached observational actor forward after an epoch update."""
    with torch.no_grad():
        logits, no_assign = actor(
            global_feats=batch["global_feats"],
            demand_feats=batch["demand_feats"],
            agent_feats=batch["agent_feats"],
            agent_mask=batch["agent_mask"],
            candidate_feats=batch["candidate_feats"],
            pair_agent_index=batch["pair_agent_index"],
            safe_mask=batch["safe_mask"],
        )
    return logits.detach(), no_assign.detach()


def build_epoch_trace_rows(*, model: Mapping[str, Any], prepared: Mapping[str, Any], epoch_index: int,
                           loss: Mapping[str, Any], logits_pre: torch.Tensor, no_assign_pre: torch.Tensor,
                           logits_post: torch.Tensor, no_assign_post: torch.Tensor, JL: Any,
                           clip_epsilon: float | None = None) -> list[dict[str, Any]]:
    """Build per-epoch PPO trace rows from existing loss outputs."""
    batch = prepared["batch"]
    base_rows = build_base_trace_rows(model=model, prepared=prepared)
    epsilon = float(clip_epsilon if clip_epsilon is not None else JL.CC.PPO_CLIP_EPSILON)
    low, high = 1.0 - epsilon, 1.0 + epsilon

    for key in ("ratio", "new_log_prob", "unclipped_objective_per_row", "clipped_objective_per_row",
                "policy_loss_per_row", "entropy_per_row", "actor_row_weight"):
        _require(key in loss and isinstance(loss[key], torch.Tensor), "TRACE_PPO_LOSS_FIELD_MISSING", key)

    old_log_prob = batch["old_log_prob"].detach()
    action_index = batch["action_index"].detach()
    advantage = batch["advantage"].detach()
    safe_mask = batch["safe_mask"].detach()
    recomputed_ratio = ppo_ratio_from_log_probs(loss["new_log_prob"].detach(), old_log_prob)
    _require(torch.equal(loss["ratio"].detach().cpu(), recomputed_ratio.detach().cpu()),
             "TRACE_PPO_RATIO_RUNTIME_MISMATCH", f"epoch={epoch_index}")

    log_probs_pre = JL.masked_log_probs(logits_pre.detach(), no_assign_pre.detach(), safe_mask)
    log_probs_post = JL.masked_log_probs(logits_post.detach(), no_assign_post.detach(), safe_mask)
    selected_logit_pre = _selected_tensor_values(values=logits_pre.detach(), no_assign=no_assign_pre.detach(), action_index=action_index)
    selected_logit_post = _selected_tensor_values(values=logits_post.detach(), no_assign=no_assign_post.detach(), action_index=action_index)
    selected_prob_pre = log_probs_pre.gather(-1, action_index.unsqueeze(-1)).squeeze(-1).exp()
    selected_prob_post = log_probs_post.gather(-1, action_index.unsqueeze(-1)).squeeze(-1).exp()
    no_assign_col = safe_mask.shape[1]
    no_assign_prob_pre = log_probs_pre[:, no_assign_col].exp()
    no_assign_prob_post = log_probs_post[:, no_assign_col].exp()

    actor_denominator = float(loss.get("actor_denominator", 1.0))
    _require(math.isfinite(actor_denominator) and actor_denominator >= 1.0, "TRACE_ACTOR_DENOMINATOR_INVALID", str(epoch_index))
    entropy_coef = float(loss.get("entropy_coef", 0.0))
    optional_factorized = loss.get("factorized_trace", {})
    if optional_factorized is None:
        optional_factorized = {}
    _require(isinstance(optional_factorized, Mapping), "TRACE_FACTORIZED_TRACE_INVALID", str(epoch_index))

    rows: list[dict[str, Any]] = []
    for index, base in enumerate(base_rows):
        ratio = _finite_float(loss["ratio"].detach().cpu()[index], "TRACE_PPO_RATIO_MISSING", base["decision_id"])
        adv = _finite_float(advantage.detach().cpu()[index], "TRACE_NORMALIZED_ADVANTAGE_MISSING", base["decision_id"])
        unclipped_objective = _finite_float(loss["unclipped_objective_per_row"].detach().cpu()[index], "TRACE_POLICY_LOSS_MISSING", base["decision_id"])
        clipped_objective = _finite_float(loss["clipped_objective_per_row"].detach().cpu()[index], "TRACE_POLICY_LOSS_MISSING", base["decision_id"])
        selected_policy_loss = _finite_float(loss["policy_loss_per_row"].detach().cpu()[index], "TRACE_POLICY_LOSS_MISSING", base["decision_id"])
        actor_row_weight = _finite_float(loss["actor_row_weight"].detach().cpu()[index], "TRACE_ACTOR_ROW_WEIGHT_MISSING", base["decision_id"])
        selected_logit_delta = float(selected_logit_post.detach().cpu()[index]) - float(selected_logit_pre.detach().cpu()[index])
        no_assign_logit_delta = float(no_assign_post.detach().cpu()[index, 0]) - float(no_assign_pre.detach().cpu()[index, 0])
        selected_probability_delta = float(selected_prob_post.detach().cpu()[index]) - float(selected_prob_pre.detach().cpu()[index])
        no_assign_probability_delta = float(no_assign_prob_post.detach().cpu()[index]) - float(no_assign_prob_pre.detach().cpu()[index])
        row = {
            **base,
            "epoch_index": int(epoch_index),
            "new_log_prob": _finite_float(loss["new_log_prob"].detach().cpu()[index], "TRACE_NEW_LOG_PROB_MISSING", base["decision_id"]),
            "ppo_ratio": ratio,
            "clip_low": low,
            "clip_high": high,
            "was_clipped": classify_clip(ratio=ratio, advantage=adv, clip_epsilon=epsilon),
            "policy_loss_unclipped": -unclipped_objective,
            "policy_loss_clipped": -clipped_objective,
            "selected_policy_loss": selected_policy_loss,
            "selected_policy_loss_contribution_to_actor_loss": selected_policy_loss * actor_row_weight / actor_denominator,
            "entropy_contribution": float(loss["entropy_per_row"].detach().cpu()[index]) * actor_row_weight / actor_denominator,
            "entropy_loss_contribution": -entropy_coef * float(loss["entropy_per_row"].detach().cpu()[index]) * actor_row_weight / actor_denominator,
            "actor_row_weight": actor_row_weight,
            "actor_denominator": actor_denominator,
            "selected_candidate_logit_pre": float(selected_logit_pre.detach().cpu()[index]),
            "selected_candidate_probability_pre": float(selected_prob_pre.detach().cpu()[index]),
            "NO_ASSIGN_logit_pre": float(no_assign_pre.detach().cpu()[index, 0]),
            "NO_ASSIGN_probability_pre": float(no_assign_prob_pre.detach().cpu()[index]),
            "selected_candidate_logit_post_epoch": float(selected_logit_post.detach().cpu()[index]),
            "selected_candidate_probability_post_epoch": float(selected_prob_post.detach().cpu()[index]),
            "NO_ASSIGN_logit_post_epoch": float(no_assign_post.detach().cpu()[index, 0]),
            "NO_ASSIGN_probability_post_epoch": float(no_assign_prob_post.detach().cpu()[index]),
            "relative_logit_delta_vs_NO_ASSIGN": selected_logit_delta - no_assign_logit_delta,
            "relative_probability_delta_vs_NO_ASSIGN": selected_probability_delta - no_assign_probability_delta,
        }
        for field in OPTIONAL_FACTORIZED_TRACE_FIELDS:
            if field not in optional_factorized:
                continue
            value = optional_factorized[field]
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    row[field] = _finite_float(value.detach().cpu(), f"TRACE_{field.upper()}_INVALID", base["decision_id"])
                else:
                    row[field] = _finite_float(value.detach().cpu()[index], f"TRACE_{field.upper()}_INVALID", base["decision_id"])
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                row[field] = _finite_float(value[index], f"TRACE_{field.upper()}_INVALID", base["decision_id"])
            else:
                row[field] = str(value) if field == "factorized_actor_contract_id" else _finite_float(value, f"TRACE_{field.upper()}_INVALID", base["decision_id"])
        rows.append(row)
    return rows


def identity_roundtrip_audit(*, base_rows: Sequence[Mapping[str, Any]], epoch_rows: Sequence[Mapping[str, Any]],
                             expected_epochs: int = 3) -> dict[str, Any]:
    base_by_digest = {str(row["identity_digest"]): dict(row) for row in base_rows}
    _require(len(base_by_digest) == len(base_rows), "TRACE_IDENTITY_DIGEST_NOT_UNIQUE", "")
    epoch_sets: dict[int, set[str]] = {}
    for row in epoch_rows:
        digest = str(row.get("identity_digest", ""))
        epoch = int(row.get("epoch_index", -1))
        _require(digest in base_by_digest, "TRACE_EPOCH_IDENTITY_MISMATCH", str(row.get("decision_id", "")))
        for field in STABLE_IDENTITY_FIELDS:
            _require(row.get(field) == base_by_digest[digest].get(field), "TRACE_EPOCH_IDENTITY_MISMATCH", f"{field}:{row.get('decision_id', '')}")
        epoch_sets.setdefault(epoch, set()).add(digest)
    expected = set(base_by_digest)
    missing_by_epoch = {epoch: sorted(expected - values) for epoch, values in epoch_sets.items()}
    extra_epochs = sorted(epoch for epoch in epoch_sets if epoch < 1 or epoch > expected_epochs)
    _require(set(epoch_sets) == set(range(1, expected_epochs + 1)) and not any(missing_by_epoch.values()) and not extra_epochs,
             "TRACE_EPOCH_IDENTITY_MISMATCH", f"epochs={sorted(epoch_sets)}")
    no_assign_values = {str(row.get("NO_ASSIGN_semantic_identity")) for row in list(base_rows) + list(epoch_rows)}
    _require(no_assign_values == {NO_ASSIGN_SEMANTIC_IDENTITY}, "TRACE_NO_ASSIGN_IDENTITY_MISMATCH", str(no_assign_values))
    return {
        "contract_id": TRACE_CONTRACT_ID,
        "base_row_count": len(base_rows),
        "epoch_row_count": len(epoch_rows),
        "epochs": sorted(epoch_sets),
        "expected_epochs": expected_epochs,
        "stable_identity_fields": list(STABLE_IDENTITY_FIELDS),
        "identity_roundtrip_passed": True,
        "no_assign_identity": NO_ASSIGN_SEMANTIC_IDENTITY,
        "no_assign_identity_stable": True,
        "missing_by_epoch": missing_by_epoch,
    }


def row_count_summary(*, base_rows: Sequence[Mapping[str, Any]], epoch_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        arm = str(row["arm_id"])
        by_arm.setdefault(arm, {"assignment_rows": 0, "actor_eligible_assignment_rows": 0, "critic_eligible_assignment_rows": 0})
        by_arm[arm]["assignment_rows"] += 1
        by_arm[arm]["actor_eligible_assignment_rows"] += int(bool(row["actor_eligible"]))
        by_arm[arm]["critic_eligible_assignment_rows"] += int(bool(row["critic_eligible"]))
    for arm in by_arm:
        arm_epoch_rows = [row for row in epoch_rows if str(row["arm_id"]) == arm]
        epochs = sorted({int(row["epoch_index"]) for row in arm_epoch_rows})
        by_arm[arm]["epochs"] = epochs
        by_arm[arm]["ppo_ratio_rows"] = len(arm_epoch_rows)
        by_arm[arm]["eligible_credit_trace_rows"] = sum(bool(row["actor_eligible"]) for row in arm_epoch_rows)
        by_arm[arm]["expected_ppo_ratio_rows"] = by_arm[arm]["assignment_rows"] * len(epochs)
        by_arm[arm]["expected_eligible_credit_trace_rows"] = by_arm[arm]["actor_eligible_assignment_rows"] * len(epochs)
    return {
        "contract_id": TRACE_CONTRACT_ID,
        "arms": by_arm,
        "total_assignment_rows": len(base_rows),
        "total_epoch_rows": len(epoch_rows),
        "total_actor_eligible_assignment_rows": sum(bool(row["actor_eligible"]) for row in base_rows),
        "total_eligible_credit_trace_rows": sum(bool(row["actor_eligible"]) for row in epoch_rows),
    }


def write_trace_artifacts(*, root: Path, models: Mapping[str, Mapping[str, Any]], prepared: Mapping[str, Any],
                          training: Mapping[str, Any]) -> dict[str, Any]:
    """Persist the future bounded-run trace artifacts atomically at artifact level."""
    root.mkdir(parents=True, exist_ok=True)
    base_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for arm_id, model in models.items():
        prepared_cell = prepared[str(arm_id)]
        training_cell = training[str(arm_id)]
        arm_base = build_base_trace_rows(model=model, prepared=prepared_cell)
        arm_epoch = list(training_cell.get("epoch_trace_rows", []))
        _require(bool(arm_epoch), "TRACE_EPOCH_ROWS_MISSING", str(arm_id))
        identity_roundtrip_audit(base_rows=arm_base, epoch_rows=arm_epoch, expected_epochs=3)
        base_rows.extend(arm_base)
        epoch_rows.extend(arm_epoch)
    audit = identity_roundtrip_audit(base_rows=base_rows, epoch_rows=epoch_rows, expected_epochs=3)
    summary = row_count_summary(base_rows=base_rows, epoch_rows=epoch_rows)

    credit_rows = [dict(row) for row in base_rows]
    gae_rows = [dict(row) for row in base_rows]
    ppo_rows = [dict(row) for row in epoch_rows]
    loss_rows = [dict(row) for row in epoch_rows]
    eligible_rows = [dict(row) for row in epoch_rows if bool(row["actor_eligible"])]

    outputs = {
        TRACE_PARQUET_FILES["eligible_row_credit_trace"]: eligible_rows,
        TRACE_PARQUET_FILES["credit_eligibility_rows"]: credit_rows,
        TRACE_PARQUET_FILES["gae_rows"]: gae_rows,
        TRACE_PARQUET_FILES["ppo_ratio_rows"]: ppo_rows,
        TRACE_PARQUET_FILES["per_epoch_loss_contributions"]: loss_rows,
    }
    for filename, rows in outputs.items():
        path = root / filename
        _require(not path.exists(), "TRACE_ARTIFACT_OVERWRITE_FORBIDDEN", filename)
        pd.DataFrame(rows).to_parquet(path, index=False)

    schema = trace_schema_payload()
    (root / "trace_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "trace_row_count_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "trace_identity_roundtrip_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "trace_schema": schema,
        "trace_row_count_summary": summary,
        "trace_identity_roundtrip_audit": audit,
        "artifact_files": dict(TRACE_PARQUET_FILES),
        "durable_trace_persisted": True,
    }
