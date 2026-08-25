#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-CR2/CR3 joint-assignment critic, duration-aware GAE, PPO interface.

Three pieces, all separate from the operational MAPPO stack:

    JointAssignmentCritic      V_joint(S_k) before the assignment is chosen
    compute_assignment_gae     GAE over assignment transitions, duration aware
    assignment_ppo_loss        PPO objective over the frozen rollout action support

`assignment_ppo_loss` only computes; it never touches a parameter.  The single
place that steps an optimizer is `apply_assignment_update`, and it is guarded by
`require_capability("training")` before any gradient is computed.  BT1 stepped the
optimizers directly and so bypassed that check entirely; routing every update
through one guarded entrypoint is what closes that hole.

Critic input discipline
-----------------------
The critic predicts the value of the state *before* selection, so it never sees
the selected agent, the selected candidate, the post-selection action identity,
future reward, or future state.  It sees the global context, the fleet, the
demand, and an aggregate summary of the safe set -- how many options exist and
what they look like in aggregate, never which one was taken.

Boundary discipline
-------------------
This project has been bitten by cross-window advantage contamination before, so
GAE resets hard at every episode and window boundary and at every terminated or
truncated row.  A window's advantages depend on that window alone; concatenating
or reordering windows cannot move them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

import joint_assignment_credit_contract as CC
import simulator_authorization as _authz

CRITIC_ID = "JOINT_ASSIGNMENT_CRITIC_V1"
PPO_ID = "JOINT_ASSIGNMENT_PPO_INTERFACE_V1"
NEG_INF = float("-inf")

FORBIDDEN_CRITIC_INPUTS = ("selected_agent_id", "selected_candidate_id",
                           "post_selection_action_identity", "future_reward", "future_state")

CRITIC_CONTRACT = {
    "critic_id": CRITIC_ID,
    "predicts": "V_joint(S_k), the value of the state before assignment selection",
    "consumes": ["global/GATv2 context", "decision-group demand context",
                 "active agent set state", "fleet context", "safe-pair-set aggregate summary"],
    "must_not_consume": list(FORBIDDEN_CRITIC_INPUTS),
    "selected_action_leakage": False,
    "future_leakage": False,
    "permutation_safe_agent_encoding": True,
    "agent_id_as_positional_preference": False,
    "separate_from_operational_critic": True,
}

PPO_CONTRACT = {
    "ppo_id": PPO_ID,
    "clip_epsilon": CC.PPO_CLIP_EPSILON,
    "gamma": CC.GAMMA, "gae_lambda": CC.GAE_LAMBDA,
    "constants_source": "existing authoritative MAPPO configuration",
    "constants_invented": False,
    "action_support": "frozen rollout evidence, never regenerated",
    "old_log_prob_source": "rollout evidence, never recomputed after policy mutation",
    "advantage_detached_from_critic_graph": True,
    "forced_rows_actor_weight": 0.0,
    "forced_rows_entropy_weight": 0.0,
    "forced_rows_in_actor_denominator": False,
    "entropy_over": "safe pairs plus NO_ASSIGN only",
    "optimizer_step_called": False,
    "optimizer_step_requires_capability": "training",
    "optimizer_step_entrypoint": "joint_assignment_learning.apply_assignment_update",
}

TRAINING_GUARD_SITE = "joint_assignment_learning.py::apply_assignment_update"


class JointAssignmentCritic(nn.Module):
    """Centralized value for an assignment state, blind to the action taken."""

    def __init__(self, *, global_dim: int, demand_dim: int, agent_dim: int,
                 safe_summary_dim: int, hidden: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.agent_project = nn.Linear(agent_dim, hidden)
        self.agent_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.agent_norm = nn.LayerNorm(hidden)
        self.trunk = nn.Sequential(
            nn.Linear(global_dim + demand_dim + hidden + safe_summary_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, *, global_feats: torch.Tensor, demand_feats: torch.Tensor,
                agent_feats: torch.Tensor, agent_mask: torch.Tensor,
                safe_summary: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.agent_project(agent_feats))
        attended, _ = self.agent_attention(hidden, hidden, hidden,
                                           key_padding_mask=~agent_mask, need_weights=False)
        agent_ctx = self.agent_norm(hidden + attended)
        mask_f = agent_mask.unsqueeze(-1).to(agent_ctx.dtype)
        fleet = (agent_ctx * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.trunk(torch.cat([global_feats, demand_feats, fleet, safe_summary], dim=-1)).squeeze(-1)


def safe_set_summary(candidate_feats: torch.Tensor, safe_mask: torch.Tensor) -> torch.Tensor:
    """Aggregate description of the safe set: size and shape, never identity.

    Deliberately order-free and selection-free, so the critic learns from how many
    options exist and what they look like, not from which one was chosen.
    """
    mask = safe_mask.unsqueeze(-1).to(candidate_feats.dtype)
    count = mask.sum(dim=1)
    total = (candidate_feats * mask).sum(dim=1)
    mean = total / count.clamp(min=1.0)
    masked = candidate_feats.masked_fill(~safe_mask.unsqueeze(-1), NEG_INF)
    best = masked.max(dim=1).values
    best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))
    return torch.cat([count, mean, best], dim=-1)


def compute_assignment_gae(transitions: Sequence[CC.AssignmentTransition],
                           values: Sequence[float], next_values: Sequence[float],
                           *, gamma: float = CC.GAMMA, lam: float = CC.GAE_LAMBDA
                           ) -> Dict[str, List[float]]:
    """Duration-aware GAE over assignment transitions, reset at every boundary.

    Transitions are grouped by (episode, window) and processed backwards inside
    each group only, so no advantage can reach across a boundary.  The bootstrap
    factor is gamma ** delta_operational_steps, which is what makes this a
    semi-Markov high-level step rather than a one-step MDP.
    """
    if not (len(transitions) == len(values) == len(next_values)):
        raise CC.CreditContractError("GAE_INPUT_LENGTH_MISMATCH")
    advantages = [0.0] * len(transitions)
    returns = [0.0] * len(transitions)
    residuals = [0.0] * len(transitions)
    groups: Dict[Tuple[str, str], List[int]] = {}
    for i, t in enumerate(transitions):
        groups.setdefault((t.episode_id, t.window_id), []).append(i)
    for _, idx in groups.items():
        idx.sort(key=lambda i: transitions[i].decision_ts)
        running = 0.0
        for pos in reversed(range(len(idx))):
            i = idx[pos]
            t = transitions[i]
            last_in_group = pos == len(idx) - 1
            nonterminal = 0.0 if (t.terminated or t.truncated or last_in_group) else 1.0
            gamma_k = CC.bootstrap_discount(t.delta_operational_steps, gamma)
            delta = t.assignment_discounted_reward + gamma_k * next_values[i] * nonterminal - values[i]
            running = delta + gamma_k * lam * nonterminal * running
            residuals[i] = delta
            advantages[i] = running
            returns[i] = running + values[i]
    return {"assignment_advantage": advantages, "assignment_return": returns,
            "assignment_td_residual": residuals}


class AssignmentRewardNormalizer:
    """Same algorithm as the authoritative MAPPO normalizer, separate state.

    Window and clip match `mappo_runner.RewardNormalizer`; the running statistics
    are this subsystem's own, so assignment credit can never perturb the
    operational actor's normalization.
    """

    def __init__(self, window_size: int = 1000, clip_value: float = 10.0) -> None:
        self.window_size = int(window_size)
        self.clip_value = float(clip_value)
        self.buffer: List[float] = []
        self.shares_state_with_operational_actor = False

    def normalize(self, rewards: Sequence[float]) -> List[float]:
        self.buffer.extend(float(r) for r in rewards)
        self.buffer = self.buffer[-self.window_size:]
        if len(self.buffer) < 2:
            return [float(r) for r in rewards]
        mean = sum(self.buffer) / len(self.buffer)
        var = sum((x - mean) ** 2 for x in self.buffer) / len(self.buffer)
        std = max(var ** 0.5, 1e-8)
        return [max(-self.clip_value, min(self.clip_value, (float(r) - mean) / std)) for r in rewards]


def masked_log_probs(pair_logits: torch.Tensor, no_assign_logit: torch.Tensor,
                     safe_mask: torch.Tensor) -> torch.Tensor:
    logits = torch.where(safe_mask, pair_logits, torch.full_like(pair_logits, NEG_INF))
    return torch.log_softmax(torch.cat([logits, no_assign_logit], dim=-1), dim=-1)


def assignment_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Entropy over the valid action set only; masked entries contribute nothing."""
    p = log_probs.exp()
    safe_log = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
    return -(p * safe_log).sum(-1)


def assignment_ppo_loss(*, new_pair_logits: torch.Tensor, new_no_assign_logit: torch.Tensor,
                        safe_mask: torch.Tensor, action_index: torch.Tensor,
                        old_log_prob: torch.Tensor, advantage: torch.Tensor,
                        value_pred: torch.Tensor, value_target: torch.Tensor,
                        forced_action: torch.Tensor,
                        actor_eligibility_mask: Optional[torch.Tensor] = None,
                        clip_epsilon: float = CC.PPO_CLIP_EPSILON,
                        entropy_coef: float = 0.0) -> Dict[str, Any]:
    """PPO objective over the frozen rollout support.  No parameter is updated.

    Forced rows -- where NO_ASSIGN was the only legal action -- contribute zero to
    the actor and entropy terms and are kept out of the actor denominator, so a
    batch full of them cannot silently shrink the policy gradient. They still
    train the critic.

    ``actor_eligibility_mask`` is an optional, explicit actor-aggregation mask.
    It is for a separately bound credit-eligibility contract only: it never
    changes logits, legal support, old log-probabilities, PPO ratios, clipping,
    advantages, values, targets, or critic loss.  An all-false mask is a valid
    explicit actor skip (with a finite critic loss), rather than a divide-by-zero
    or an implicit fallback to all rows.
    """
    log_probs = masked_log_probs(new_pair_logits, new_no_assign_logit, safe_mask)
    new_log_prob = log_probs.gather(-1, action_index.unsqueeze(-1)).squeeze(-1)
    # Advantage is rollout evidence; detaching keeps the actor gradient from
    # leaking backwards into the critic.
    adv = advantage.detach()
    ratio = torch.exp(new_log_prob - old_log_prob.detach())
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
    per_row = -torch.min(unclipped, clipped)
    if actor_eligibility_mask is None:
        eligible = torch.ones_like(forced_action, dtype=torch.bool)
        eligibility_supplied = False
    else:
        if actor_eligibility_mask.dtype != torch.bool:
            raise ValueError("ACTOR_ELIGIBILITY_MASK_MUST_BE_BOOL")
        if actor_eligibility_mask.shape != forced_action.shape:
            raise ValueError("ACTOR_ELIGIBILITY_MASK_SHAPE_MISMATCH")
        if actor_eligibility_mask.device != forced_action.device:
            raise ValueError("ACTOR_ELIGIBILITY_MASK_DEVICE_MISMATCH")
        eligible = actor_eligibility_mask
        eligibility_supplied = True
    active_bool = (~forced_action) & eligible
    active = active_bool.to(per_row.dtype)
    active_count = active.sum()
    denom = active_count.clamp(min=1.0)
    actor_loss = (per_row * active).sum() / denom
    ent = assignment_entropy(log_probs)
    entropy_term = (ent * active).sum() / denom
    critic_loss = ((value_pred - value_target.detach()) ** 2).mean()
    return {
        "actor_loss": actor_loss - entropy_coef * entropy_term,
        "policy_loss": actor_loss,
        "entropy": entropy_term,
        "critic_loss": critic_loss,
        "ratio": ratio,
        "new_log_prob": new_log_prob,
        "unclipped_objective_per_row": unclipped,
        "clipped_objective_per_row": clipped,
        "policy_loss_per_row": per_row,
        "entropy_per_row": ent,
        "actor_row_weight": active,
        "actor_denominator": float(denom.item()),
        "actor_eligible_rows": int(active_count.item()),
        "actor_ineligible_rows": int((~eligible).sum().item()),
        "actor_update_skipped": bool(active_count.item() == 0),
        "actor_eligibility_mask_supplied": eligibility_supplied,
        "forced_rows": int(forced_action.sum().item()),
        "non_forced_rows": int((~forced_action).sum().item()),
        "advantage_detached": not adv.requires_grad,
        "optimizer_step_called": False,
    }


def apply_assignment_update(*, loss: Dict[str, Any], actor: nn.Module, critic: nn.Module,
                            actor_optimizer: Any, critic_optimizer: Any,
                            max_grad_norm: Optional[float] = None) -> Dict[str, Any]:
    """The only sanctioned way to step the joint-assignment optimizers.

    BT1 revealed that the training capability was granted but never checked on
    this path: R9.8 wired `require_capability("training")` onto the legacy PPO
    entrypoints, and this newer optimizer path had no guard at all.  Owning the
    optimizer step here closes that hole -- the check runs before any gradient is
    computed, so an unauthorized caller cannot move a single parameter.

    Callers must not step the optimizers themselves; doing so would route around
    the guard exactly the way BT1 accidentally did.
    """
    _authz.require_capability("training", site=TRAINING_GUARD_SITE)
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    total = loss["actor_loss"] + loss["critic_loss"]
    total.backward()
    actor_grad = float(sum(p.grad.abs().sum() for p in actor.parameters() if p.grad is not None))
    critic_grad = float(sum(p.grad.abs().sum() for p in critic.parameters() if p.grad is not None))
    actor_norm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in actor.parameters()
                                      if p.grad is not None)))
    critic_norm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in critic.parameters()
                                       if p.grad is not None)))
    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    actor_optimizer.step()
    critic_optimizer.step()
    return {"capability_checked": "training", "optimizer_step_called": True,
            "actor_grad_abs_sum": actor_grad, "critic_grad_abs_sum": critic_grad,
            "actor_grad_norm": actor_norm, "critic_grad_norm": critic_norm,
            "total_loss": float(total.detach())}
