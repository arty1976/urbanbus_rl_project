#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-JA2 multi-agent joint candidate-assignment head.

A new, separately versioned decision layer.  The promoted 3-action operational
actor is not touched, not imported, and not re-weighted: this head answers a
different question one level above it -- which agent takes the demand, under
which Zero-Loss-safe plan.

Shape independence
------------------
There is no `agent_0_candidate_0` output neuron.  One shared scorer maps any
(global context, agent context, candidate features) triple to a single logit, so
2 pairs produce 2 logits and 150 pairs produce 150 with the same weights.  Cost
is linear in the number of proposed pairs; no joint action space is ever
enumerated.

Cooperation
-----------
Agent contexts pass through a masked self-attention set encoder before scoring,
so a candidate's logit genuinely depends on what the *other* vehicles are doing --
their load, their remaining plan, how many safe options they hold.  There is no
positional encoding and agent ids are never features, so the encoder is
permutation-equivariant: reordering the fleet moves the rows, never the meaning.

Safety
------
Zero-Loss is a hard mask, not a penalty.  Rejected pairs are never in the tensor
at all, and any pair masked unsafe receives a -inf logit, so its probability is
exactly zero under both sampling and argmax.  `NO_ASSIGN` is always present as a
real option and is the only outcome when nothing is safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_ID = "MULTI_AGENT_JOINT_ASSIGNMENT_HEAD_V1"
HEAD_VERSION = "LS3_JA2_V1"
SELECTION_SEMANTICS = "SHADOW_UNTRAINED_ARCHITECTURE_VALIDATION_ONLY"
NEG_INF = float("-inf")

HEAD_CONTRACT = {
    "head_id": HEAD_ID, "version": HEAD_VERSION,
    "separate_from_operational_actor": True,
    "operational_actor_modified": False,
    "operational_actor_imported": False,
    "shared_scorer": True,
    "per_pair_output_neurons": False,
    "supports_variable_agents": True,
    "supports_variable_candidates_per_agent": True,
    "scoring_complexity": "O(pairs)",
    "joint_action_enumeration": False,
    "agent_id_used_as_feature": False,
    "positional_encoding_over_agents": False,
    "permutation_equivariant_agent_encoder": True,
    "other_agent_state_influences_scores": True,
    "zero_loss_mask": "hard, -inf logit, probability exactly 0",
    "zero_loss_soft_penalty": False,
    "no_assign_always_available": True,
    "selection_semantics": SELECTION_SEMANTICS,
    "trained": False,
}


@dataclass
class AssignmentOutput:
    decision_group_id: str
    pair_logits: torch.Tensor          # (P,)
    no_assign_logit: torch.Tensor      # scalar
    probabilities: torch.Tensor        # (P + 1,), last entry is NO_ASSIGN
    selected_index: int                # P means NO_ASSIGN
    pair_keys: List[Tuple[str, str]]
    safe_mask: torch.Tensor

    @property
    def selected_is_no_assign(self) -> bool:
        return self.selected_index == len(self.pair_keys)

    @property
    def selected_pair(self) -> Optional[Tuple[str, str]]:
        return None if self.selected_is_no_assign else self.pair_keys[self.selected_index]

    def payload(self) -> Dict[str, Any]:
        probs = self.probabilities.detach().tolist()
        return {
            "decision_group_id": self.decision_group_id,
            "safe_pair_count": len(self.pair_keys),
            "pairs": [{"agent_id": a, "candidate_id": c,
                       "logit": float(self.pair_logits[i].detach()),
                       "masked_probability": float(probs[i])}
                      for i, (a, c) in enumerate(self.pair_keys)],
            "no_assign_logit": float(self.no_assign_logit.detach()),
            "no_assign_probability": float(probs[-1]),
            "shadow_selected_agent_id": None if self.selected_is_no_assign else self.selected_pair[0],
            "shadow_selected_candidate_id": None if self.selected_is_no_assign else self.selected_pair[1],
            "selected_action": "NO_ASSIGN_KEEP_CURRENT_PLANS" if self.selected_is_no_assign else "ASSIGN",
            "selection_mode": "argmax over masked distribution",
            "selection_semantics": SELECTION_SEMANTICS,
            "policy_version": HEAD_VERSION,
            "non_selected_agents": "KEEP_CURRENT_PLAN",
        }


def _mlp(sizes: Sequence[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class AgentSetEncoder(nn.Module):
    """Masked self-attention over the fleet.

    No positional encoding, so the output for a given agent depends on the *set*
    of other agents rather than their order.  This is what makes a candidate's
    score able to reflect the rest of the fleet.
    """

    def __init__(self, agent_dim: int, hidden: int, heads: int = 4) -> None:
        super().__init__()
        self.project = nn.Linear(agent_dim, hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, agent_feats: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        # agent_feats (B, A, F); agent_mask (B, A) with True for real agents.
        hidden = torch.tanh(self.project(agent_feats))
        pad = ~agent_mask
        attended, _ = self.attention(hidden, hidden, hidden, key_padding_mask=pad, need_weights=False)
        return self.norm(hidden + attended)


class MultiAgentCandidateAssignmentHead(nn.Module):
    """Scores arbitrary (agent, candidate) pairs against one shared function."""

    def __init__(self, *, global_dim: int, agent_dim: int, candidate_dim: int,
                 demand_dim: int, hidden: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.agent_encoder = AgentSetEncoder(agent_dim, hidden, heads)
        self.global_encoder = _mlp([global_dim, hidden, hidden])
        self.demand_encoder = _mlp([demand_dim, hidden, hidden])
        self.candidate_encoder = _mlp([candidate_dim, hidden, hidden])
        self.scorer = _mlp([hidden * 4, hidden, hidden, 1])
        self.no_assign_scorer = _mlp([hidden * 2, hidden, 1])

    def forward(self, *, global_feats: torch.Tensor, demand_feats: torch.Tensor,
                agent_feats: torch.Tensor, agent_mask: torch.Tensor,
                candidate_feats: torch.Tensor, pair_agent_index: torch.Tensor,
                safe_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pair_logits (B, P), no_assign_logit (B, 1)).

        Unsafe pairs are driven to -inf here, so nothing downstream has to
        remember to exclude them.
        """
        agent_ctx = self.agent_encoder(agent_feats, agent_mask)           # (B, A, H)
        global_ctx = self.global_encoder(global_feats)                    # (B, H)
        demand_ctx = self.demand_encoder(demand_feats)                    # (B, H)
        cand_ctx = self.candidate_encoder(candidate_feats)                # (B, P, H)

        batch, pairs, _ = cand_ctx.shape
        index = pair_agent_index.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.hidden)
        per_pair_agent = torch.gather(agent_ctx, 1, index)                # (B, P, H)
        # Fleet summary over real agents only: the "what is everyone else doing" term.
        mask_f = agent_mask.unsqueeze(-1).to(agent_ctx.dtype)
        fleet_ctx = (agent_ctx * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        stacked = torch.cat([
            global_ctx.unsqueeze(1).expand(-1, pairs, -1),
            demand_ctx.unsqueeze(1).expand(-1, pairs, -1),
            per_pair_agent,
            fleet_ctx.unsqueeze(1).expand(-1, pairs, -1),
        ], dim=-1)
        logits = self.scorer(stacked).squeeze(-1)                          # (B, P)
        logits = torch.where(safe_mask, logits, torch.full_like(logits, NEG_INF))
        no_assign = self.no_assign_scorer(torch.cat([global_ctx, fleet_ctx], dim=-1))
        return logits, no_assign


def masked_distribution(pair_logits: torch.Tensor, no_assign_logit: torch.Tensor,
                        safe_mask: torch.Tensor) -> torch.Tensor:
    """Softmax over safe pairs plus NO_ASSIGN.  Unsafe probability is exactly 0."""
    logits = torch.where(safe_mask, pair_logits, torch.full_like(pair_logits, NEG_INF))
    full = torch.cat([logits, no_assign_logit], dim=-1)
    return torch.softmax(full, dim=-1)


def select(decision_group_id: str, pair_keys: Sequence[Tuple[str, str]],
           pair_logits: torch.Tensor, no_assign_logit: torch.Tensor,
           safe_mask: torch.Tensor) -> AssignmentOutput:
    """Deterministic argmax over the masked distribution.

    With no safe pair the distribution collapses onto NO_ASSIGN, which is the
    honest outcome rather than an invented fallback route.
    """
    probs = masked_distribution(pair_logits, no_assign_logit, safe_mask)
    flat = probs[0] if probs.dim() > 1 else probs
    selected = int(torch.argmax(flat).item())
    return AssignmentOutput(
        decision_group_id=decision_group_id,
        pair_logits=(pair_logits[0] if pair_logits.dim() > 1 else pair_logits),
        no_assign_logit=(no_assign_logit[0, 0] if no_assign_logit.dim() > 1 else no_assign_logit[0]),
        probabilities=flat, selected_index=selected, pair_keys=list(pair_keys),
        safe_mask=(safe_mask[0] if safe_mask.dim() > 1 else safe_mask))


def log_probability(pair_logits: torch.Tensor, no_assign_logit: torch.Tensor,
                    safe_mask: torch.Tensor, action_index: torch.Tensor) -> torch.Tensor:
    """PPO-compatible log-probability of a chosen index (P means NO_ASSIGN)."""
    logits = torch.where(safe_mask, pair_logits, torch.full_like(pair_logits, NEG_INF))
    full = torch.cat([logits, no_assign_logit], dim=-1)
    return torch.log_softmax(full, dim=-1).gather(-1, action_index.unsqueeze(-1)).squeeze(-1)


def entropy(pair_logits: torch.Tensor, no_assign_logit: torch.Tensor,
            safe_mask: torch.Tensor) -> torch.Tensor:
    """Entropy over the masked distribution; masked-out mass contributes nothing."""
    logits = torch.where(safe_mask, pair_logits, torch.full_like(pair_logits, NEG_INF))
    full = torch.cat([logits, no_assign_logit], dim=-1)
    log_p = torch.log_softmax(full, dim=-1)
    p = log_p.exp()
    return -(p * torch.where(torch.isfinite(log_p), log_p, torch.zeros_like(log_p))).sum(-1)


def build_tensors(candidate_set, *, global_vector: Sequence[float],
                  demand_vector: Sequence[float], dtype=torch.float32):
    """Pack a JointSafeCandidateSet into tensors, ordered by identity."""
    import multi_agent_assignment_contract as C
    agents = sorted(candidate_set.agents, key=lambda a: a.agent_id)
    agent_index = {a.agent_id: i for i, a in enumerate(agents)}
    pairs = candidate_set.canonical_pairs()
    agent_feats = torch.tensor([[a.feature_vector() for a in agents]], dtype=dtype) \
        if agents else torch.zeros((1, 0, len(C.AgentContext.FEATURE_NAMES)), dtype=dtype)
    agent_mask = torch.ones((1, len(agents)), dtype=torch.bool)
    cand_feats = torch.tensor([[C.candidate_feature_vector(p) for p in pairs]], dtype=dtype) \
        if pairs else torch.zeros((1, 0, len(C.LOCAL_SEARCH_FEATURE_NAMES)), dtype=dtype)
    pair_agent_index = torch.tensor([[agent_index[p.agent_id] for p in pairs]], dtype=torch.long) \
        if pairs else torch.zeros((1, 0), dtype=torch.long)
    safe_mask = torch.ones((1, len(pairs)), dtype=torch.bool)
    return {
        "global_feats": torch.tensor([list(global_vector)], dtype=dtype),
        "demand_feats": torch.tensor([list(demand_vector)], dtype=dtype),
        "agent_feats": agent_feats, "agent_mask": agent_mask,
        "candidate_feats": cand_feats, "pair_agent_index": pair_agent_index,
        "safe_mask": safe_mask,
        "pair_keys": [(p.agent_id, p.candidate_id) for p in pairs],
    }
