"""R18-R6 durable trace instrumentation regression tests.

These tests use synthetic rows and already-existing pure math functions only.
They do not execute rollout, simulator, optimizer creation, backward, optimizer
step, checkpoint write, or policy mutation.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joint_assignment_credit_contract as CC  # noqa: E402
import joint_assignment_f1_execution_contract as FC  # noqa: E402
import joint_assignment_learning as JL  # noqa: E402
import r18_durable_trace as TRACE  # noqa: E402


class SnapshotStub:
    def __init__(self, digest: str) -> None:
        self.snapshot_digest = digest


def _transition(*, index: int, candidate: bool = True, no_assign: bool = False, reward: float = 1.0,
                raw_gae: float = 0.5, support_digest: str | None = None) -> SimpleNamespace:
    support_digest = support_digest or f"support-{index}"
    return SimpleNamespace(
        assignment_step_id=f"decision-{index}",
        window_id="window-0",
        selected_is_no_assign=no_assign,
        selected_agent_id=None if no_assign else f"agent-{index}",
        selected_candidate_id=None if no_assign else f"cand-{index}",
        action_index=1 if no_assign else 0,
        no_assign_index=1,
        old_log_prob=-0.4 - index * 0.01,
        forced_action=False,
        pre_state_digest=f"pre-{index}",
        next_state_digest=f"next-{index}",
        action_support_digest=f"action-support-{index}",
        provenance={
            "trajectory_id": "traj-0",
            "time_band": "night",
            "candidate_support_digest": support_digest,
            "operational": [{"causal_state_digest": f"causal-{index}-0"}, {"causal_state_digest": f"causal-{index}-1"}],
        },
        assignment_discounted_reward=reward,
        raw_gae=raw_gae,
    )


def _plan(index: int, *, no_assign: bool = False, transition_id: str | None = None) -> SimpleNamespace:
    candidate_id = TRACE.NO_ASSIGN_SEMANTIC_IDENTITY if no_assign else f"cand-{index}"
    return SimpleNamespace(
        selected_candidate_id=candidate_id,
        applied_candidate_id=candidate_id,
        credited_candidate_id=candidate_id,
        transition_id=transition_id or f"plan-transition-{index}",
        candidate_plan_digest=f"candidate-plan-{index}",
        applied_plan_digest=f"applied-plan-{index}",
        events=({"agent_id": f"agent-{index}", "source_state_digest": f"pre-{index}"},),
        next_state=SimpleNamespace(state_digest=f"next-{index}"),
        no_assign=no_assign,
    )


def _prepared(*, rows: int = 3) -> tuple[dict[str, object], dict[str, object]]:
    rollout_rows = []
    e1_rows = []
    gae_rows = []
    for index in range(rows):
        no_assign = index == rows - 1
        support_digest = f"support-{index}"
        transition = _transition(index=index, no_assign=no_assign, support_digest=support_digest)
        plan = _plan(index, no_assign=no_assign)
        rollout_rows.append({
            "t": transition,
            "plan": plan,
            "action_type": "NO_ASSIGN" if no_assign else "CANDIDATE",
            "time_band": "night",
            "snapshot": SnapshotStub(support_digest),
        })
        e1_rows.append({
            "decision_id": transition.assignment_step_id,
            "replicate_id": "AC_CONTROL_R1",
            "data_role": "TRAIN",
            "identity_chain_valid": True,
            "selected_action_type": "NO_ASSIGN" if no_assign else "CANDIDATE",
            "selected_candidate_id": plan.selected_candidate_id,
            "applied_candidate_id": plan.applied_candidate_id,
            "credited_candidate_id": plan.credited_candidate_id,
            "reward": float(index + 1),
            "reward_gae_component": float(index + 1),
            "raw_gae": float(index + 0.5),
            "critic_eligible_e1": True,
            "category": "NO_ASSIGN_REWARD_SUPPORTED" if no_assign else "DIRECT_CANDIDATE_REWARD",
            "actor_eligible_e1": True,
        })
        gae_rows.append({
            "decision_id": transition.assignment_step_id,
            "trajectory_id": "traj-0",
            "window_id": "window-0",
            "reward": float(index + 1),
            "critic_target": float(index + 2),
            "td_residual": float(index + 0.25),
            "raw_gae": float(index + 0.5),
            "normalized_advantage": float(index - 1),
            "reward_gae_component": float(index + 1),
            "critic_bootstrap_gae_component": 0.0,
        })
    max_pairs = 1
    batch = {
        "global_feats": torch.zeros((rows, 8)),
        "demand_feats": torch.zeros((rows, 6)),
        "agent_feats": torch.zeros((rows, 2, 4)),
        "agent_mask": torch.ones((rows, 2), dtype=torch.bool),
        "candidate_feats": torch.zeros((rows, max_pairs, 3)),
        "pair_agent_index": torch.zeros((rows, max_pairs), dtype=torch.long),
        "safe_mask": torch.ones((rows, max_pairs), dtype=torch.bool),
        "action_index": torch.tensor([1 if index == rows - 1 else 0 for index in range(rows)], dtype=torch.long),
        "old_log_prob": torch.tensor([row["t"].old_log_prob for row in rollout_rows], dtype=torch.float32),
        "forced_action": torch.zeros((rows,), dtype=torch.bool),
        "advantage": torch.tensor([row["normalized_advantage"] for row in gae_rows], dtype=torch.float32),
        "value_target": torch.tensor([row["critic_target"] for row in gae_rows], dtype=torch.float32),
        "actor_eligibility_mask": torch.ones((rows,), dtype=torch.bool),
    }
    return {"arm_id": "AC_CONTROL_R1", "environment_seed": 20260822}, {
        "rows": rollout_rows,
        "e1_rows": e1_rows,
        "gae_rows": gae_rows,
        "batch": batch,
    }


def _loss_for(prepared: dict[str, object], *, logits: torch.Tensor, no_assign: torch.Tensor) -> dict[str, object]:
    batch = prepared["batch"]
    return JL.assignment_ppo_loss(
        new_pair_logits=logits,
        new_no_assign_logit=no_assign,
        safe_mask=batch["safe_mask"],
        action_index=batch["action_index"],
        old_log_prob=batch["old_log_prob"],
        advantage=batch["advantage"],
        value_pred=torch.zeros((len(prepared["rows"]),), dtype=torch.float32),
        value_target=batch["value_target"],
        forced_action=batch["forced_action"],
        actor_eligibility_mask=batch["actor_eligibility_mask"],
    )


def test_identity_roundtrip_selected_executed_credited_ppo_identity() -> None:
    model, prepared = _prepared(rows=3)
    base = TRACE.build_base_trace_rows(model=model, prepared=prepared)
    assert all(row["selected_semantic_candidate"] == row["executed_semantic_candidate"] == row["credited_semantic_candidate"] for row in base)
    logits = torch.tensor([[0.2], [0.1], [0.0]], dtype=torch.float32)
    no_assign = torch.tensor([[0.0], [0.3], [0.5]], dtype=torch.float32)
    loss = _loss_for(prepared, logits=logits, no_assign=no_assign)
    epoch_rows = []
    for epoch in range(1, 4):
        epoch_rows.extend(TRACE.build_epoch_trace_rows(
            model=model, prepared=prepared, epoch_index=epoch, loss=loss,
            logits_pre=logits, no_assign_pre=no_assign, logits_post=logits, no_assign_post=no_assign, JL=JL))
    audit = TRACE.identity_roundtrip_audit(base_rows=base, epoch_rows=epoch_rows)
    assert audit["identity_roundtrip_passed"] is True
    assert audit["epoch_row_count"] == 9


def test_gae_trace_preserves_known_runtime_gae() -> None:
    transitions = []
    rewards = [1.0, 0.5, -0.25]
    values = [0.2, 0.1, -0.1]
    next_values = [0.1, -0.1, 0.0]
    for index, reward in enumerate(rewards):
        transitions.append(CC.AssignmentTransition(
            assignment_step_id=f"gae-{index}",
            decision_group_id=f"gae-{index}",
            episode_id="episode",
            window_id="window",
            decision_ts=index,
            next_assignment_ts=None if index == 2 else index + 1,
            delta_operational_steps=1,
            pre_state_digest=f"pre-{index}",
            next_state_digest=f"next-{index}",
            safe_pair_ids=[("agent", f"cand-{index}")],
            safe_pair_mask=[True],
            no_assign_index=1,
            selected_agent_id="agent",
            selected_candidate_id=f"cand-{index}",
            selected_is_no_assign=False,
            valid_action_count=2,
            forced_action=False,
            old_log_prob=-0.5,
            old_value=values[index],
            team_reward_sequence=[reward],
            assignment_discounted_reward=reward,
            terminated=index == 2,
            truncated=False,
            policy_version="test",
            credit_contract_version=CC.CONTRACT_VERSION,
            seed=1,
            provenance={"trajectory_id": "episode", "time_band": "night", "candidate_support_digest": f"support-{index}"},
        ))
    gae = JL.compute_assignment_gae(transitions, values, next_values)
    model, prepared = _prepared(rows=3)
    for index, row in enumerate(prepared["gae_rows"]):
        row["decision_id"] = prepared["rows"][index]["t"].assignment_step_id
        row["raw_gae"] = gae["assignment_advantage"][index]
        row["critic_target"] = gae["assignment_return"][index]
        row["normalized_advantage"] = float(index)
        prepared["batch"]["advantage"][index] = float(index)
        prepared["batch"]["value_target"][index] = gae["assignment_return"][index]
    traced = TRACE.build_base_trace_rows(model=model, prepared=prepared)
    assert [row["gae_advantage_raw"] for row in traced] == pytest.approx(gae["assignment_advantage"])
    assert [row["return_raw"] for row in traced] == pytest.approx(gae["assignment_return"])


def test_advantage_normalization_preservation_exact() -> None:
    raw = [float(index - 12) for index in range(24)]
    normalized = FC.normalize_advantages_for_replicate(raw)
    assert len(normalized) == 24
    assert abs(sum(normalized)) < 1e-12
    replayed = list(normalized)
    assert replayed == normalized


def test_ppo_ratio_preservation_matches_runtime_quantity() -> None:
    old = torch.tensor([-0.4, -0.2, -1.0], dtype=torch.float32)
    new = torch.tensor([-0.3, -0.5, -1.1], dtype=torch.float32)
    assert torch.equal(TRACE.ppo_ratio_from_log_probs(new, old), torch.exp(new - old))


def test_clip_classification_inside_lower_upper() -> None:
    assert TRACE.classify_clip(ratio=1.0, advantage=1.0, clip_epsilon=0.2) is False
    assert TRACE.classify_clip(ratio=0.7, advantage=-1.0, clip_epsilon=0.2) is True
    assert TRACE.classify_clip(ratio=1.3, advantage=1.0, clip_epsilon=0.2) is True
    assert TRACE.classify_clip(ratio=0.7, advantage=1.0, clip_epsilon=0.2) is False
    assert TRACE.classify_clip(ratio=1.3, advantage=-1.0, clip_epsilon=0.2) is False


def test_per_epoch_rows_retain_identity_for_epochs_one_two_three() -> None:
    model, prepared = _prepared(rows=2)
    logits = torch.tensor([[0.2], [0.1]], dtype=torch.float32)
    no_assign = torch.tensor([[0.0], [0.3]], dtype=torch.float32)
    loss = _loss_for(prepared, logits=logits, no_assign=no_assign)
    rows = []
    for epoch in (1, 2, 3):
        rows.extend(TRACE.build_epoch_trace_rows(
            model=model, prepared=prepared, epoch_index=epoch, loss=loss,
            logits_pre=logits, no_assign_pre=no_assign, logits_post=logits, no_assign_post=no_assign, JL=JL))
    summary = TRACE.row_count_summary(base_rows=TRACE.build_base_trace_rows(model=model, prepared=prepared), epoch_rows=rows)
    assert summary["arms"]["AC_CONTROL_R1"]["epochs"] == [1, 2, 3]
    assert summary["arms"]["AC_CONTROL_R1"]["ppo_ratio_rows"] == 6


def test_no_assign_identity_is_explicit_and_stable() -> None:
    model, prepared = _prepared(rows=2)
    base = TRACE.build_base_trace_rows(model=model, prepared=prepared)
    no_assign_rows = [row for row in base if row["selected_is_no_assign"]]
    assert len(no_assign_rows) == 1
    assert no_assign_rows[0]["selected_semantic_candidate"] == TRACE.NO_ASSIGN_SEMANTIC_IDENTITY
    assert no_assign_rows[0]["NO_ASSIGN_semantic_identity"] == TRACE.NO_ASSIGN_SEMANTIC_IDENTITY


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        (lambda prepared: setattr(prepared["rows"][0]["t"], "assignment_step_id", ""), "TRACE_STABLE_DECISION_ID_MISSING"),
        (lambda prepared: setattr(prepared["rows"][0]["plan"], "credited_candidate_id", "other"), "TRACE_SELECTED_EXECUTED_CREDITED_IDENTITY_MISMATCH"),
        (lambda prepared: prepared["e1_rows"][0].update({"reward_gae_component": 0.0}), "TRACE_REWARD_ANCESTRY_MISSING_FOR_ACTOR_ELIGIBLE_ROW"),
        (lambda prepared: setattr(prepared["rows"][1]["plan"], "transition_id", prepared["rows"][0]["plan"].transition_id), "TRACE_DUPLICATE_REWARD_ANCESTRY_OWNERSHIP"),
        (lambda prepared: prepared["gae_rows"][0].pop("raw_gae"), "TRACE_RAW_GAE_MISSING"),
        (lambda prepared: prepared["gae_rows"][0].pop("normalized_advantage"), "TRACE_NORMALIZED_ADVANTAGE_MISSING"),
        (lambda prepared: setattr(prepared["rows"][0]["t"], "old_log_prob", float("nan")), "TRACE_OLD_LOG_PROB_MISSING"),
        (lambda prepared: setattr(prepared["rows"][0]["snapshot"], "snapshot_digest", "tampered"), "TRACE_CANDIDATE_SUPPORT_DIGEST_MISMATCH"),
    ],
)
def test_mandatory_omissions_fail_closed(mutator, code: str) -> None:
    model, prepared = _prepared(rows=2)
    prepared = copy.deepcopy(prepared)
    mutator(prepared)
    with pytest.raises(TRACE.TraceContractError) as exc:
        TRACE.build_base_trace_rows(model=model, prepared=prepared)
    assert exc.value.code == code


def test_epoch_identity_mismatch_fails_closed() -> None:
    model, prepared = _prepared(rows=2)
    logits = torch.tensor([[0.2], [0.1]], dtype=torch.float32)
    no_assign = torch.tensor([[0.0], [0.3]], dtype=torch.float32)
    loss = _loss_for(prepared, logits=logits, no_assign=no_assign)
    base = TRACE.build_base_trace_rows(model=model, prepared=prepared)
    epoch_rows = []
    for epoch in (1, 2, 3):
        epoch_rows.extend(TRACE.build_epoch_trace_rows(
            model=model, prepared=prepared, epoch_index=epoch, loss=loss,
            logits_pre=logits, no_assign_pre=no_assign, logits_post=logits, no_assign_post=no_assign, JL=JL))
    epoch_rows[0]["decision_id"] = "tampered"
    with pytest.raises(TRACE.TraceContractError) as exc:
        TRACE.identity_roundtrip_audit(base_rows=base, epoch_rows=epoch_rows)
    assert exc.value.code == "TRACE_EPOCH_IDENTITY_MISMATCH"
