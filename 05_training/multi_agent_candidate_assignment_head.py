#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3 multi-agent joint candidate-assignment heads.

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

Versioning
----------
``MultiAgentCandidateAssignmentHead`` is the historical LS3-JA2 V1 class. It
is intentionally retained for preserved BT6/BT8 checkpoint replay. BT8-R3 adds
the separately versioned ``CandidateSensitiveMultiAgentCandidateAssignmentHead``
V2 repair. V2 makes the smallest possible data-flow change: the already
computed, row-aligned candidate context is concatenated into the shared pair
scorer. Historical V1 checkpoints remain replayable with V1 but are never
silently reshaped or padded into V2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_ID = "MULTI_AGENT_JOINT_ASSIGNMENT_HEAD_V1"
HEAD_VERSION = "LS3_JA2_V1"
CANDIDATE_SENSITIVE_HEAD_ID = "MULTI_AGENT_JOINT_ASSIGNMENT_HEAD_V2_CANDIDATE_CONTEXT"
CANDIDATE_SENSITIVE_HEAD_VERSION = "LS3_BT8_R3_V2"
FACTORIZED_ASSIGN_CANDIDATE_HEAD_ID = "MULTI_AGENT_JOINT_ASSIGNMENT_HEAD_V3_FACTORIZED_ASSIGN_THEN_CANDIDATE"
FACTORIZED_ASSIGN_CANDIDATE_HEAD_VERSION = "LS3_BT8_R18_R16_V3"
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

CANDIDATE_SENSITIVE_HEAD_CONTRACT = {
    **HEAD_CONTRACT,
    "head_id": CANDIDATE_SENSITIVE_HEAD_ID,
    "version": CANDIDATE_SENSITIVE_HEAD_VERSION,
    "historical_v1_preserved": True,
    "candidate_context_reaches_pair_scorer": True,
    "pair_scorer_inputs": ["global_ctx", "demand_ctx", "matched_agent_ctx",
                           "fleet_ctx", "row_aligned_candidate_ctx"],
    "candidate_list_position_used": False,
    "candidate_rank_used": False,
    "legacy_actor_checkpoint_strict_compatible": False,
    "no_assign_scorer_changed": False,
    "trained": False,
}

FACTORIZED_ASSIGN_THEN_CANDIDATE_HEAD_CONTRACT = {
    **CANDIDATE_SENSITIVE_HEAD_CONTRACT,
    "head_id": FACTORIZED_ASSIGN_CANDIDATE_HEAD_ID,
    "version": FACTORIZED_ASSIGN_CANDIDATE_HEAD_VERSION,
    "architecture": "FACTORIZED_ASSIGN_THEN_CANDIDATE",
    "stage1": "BINARY_ASSIGN_VS_NO_ASSIGN_GATE",
    "stage2": "CONDITIONAL_CANDIDATE_SOFTMAX",
    "probability_model": {
        "P(NO_ASSIGN)": "P_gate(NO_ASSIGN)",
        "P(candidate_i)": "P_gate(ASSIGN) * P_candidate(candidate_i | ASSIGN)",
    },
    "no_assign_candidate_head_gradient": "ZERO",
    "legacy_direct_no_assign_scorer_removed_from_action_softmax": True,
    "no_assign_scorer_changed": True,
    "candidate_selected_behavior": "gate + conditional candidate ranker both train",
    "assign_vs_no_assign_gate_competition_preserved": True,
    "shared_softmax_candidate_no_assign_coupling_removed": True,
    "no_assign_remains_available": True,
    "zero_feasible_candidates": "P(NO_ASSIGN)=1 without empty-softmax",
    "one_feasible_candidate": "conditional candidate probability is 1",
    "variable_candidate_count": True,
    "variable_agent_count": True,
    "reward_v2_changed": False,
    "gae_changed": False,
    "ppo_objective_changed": False,
    "e1_sampling_rule_changed": False,
    "zero_loss_changed": False,
    "trained": False,
}


class ActorInputContractError(ValueError):
    """Fail-closed repaired-Actor input contract violation."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


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


@dataclass(frozen=True)
class FactorizedAssignmentDistribution:
    """A reconstructed action distribution for ASSIGN→candidate factorization.

    ``candidate_action_log_probs`` and ``no_assign_action_log_prob`` are already
    final action log-probabilities.  Passing them through the historical
    ``masked_log_probs``/selector softmax is mathematically idempotent because
    they reconstruct a normalized distribution.
    """

    candidate_logits: torch.Tensor
    assign_logit: torch.Tensor
    no_assign_logit: torch.Tensor
    safe_mask: torch.Tensor
    gate_log_probs: torch.Tensor
    conditional_candidate_log_probs: torch.Tensor
    candidate_action_log_probs: torch.Tensor
    no_assign_action_log_prob: torch.Tensor

    @property
    def action_log_probs(self) -> torch.Tensor:
        return torch.cat([self.candidate_action_log_probs, self.no_assign_action_log_prob], dim=-1)

    @property
    def action_probabilities(self) -> torch.Tensor:
        return self.action_log_probs.exp()

    @property
    def gate_probabilities(self) -> torch.Tensor:
        return self.gate_log_probs.exp()

    @property
    def conditional_candidate_probabilities(self) -> torch.Tensor:
        return self.conditional_candidate_log_probs.exp()


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


class CandidateSensitiveMultiAgentCandidateAssignmentHead(MultiAgentCandidateAssignmentHead):
    """BT8-R3 V2: V1 plus row-aligned candidate context in the pair scorer.

    V1 is not edited because its historical checkpoints are authoritative
    evidence. The sole representation change in V2 is the scorer input width
    and the fifth concatenated context block. NO_ASSIGN remains a function of
    global and fleet context only.
    """

    def __init__(self, *, global_dim: int, agent_dim: int, candidate_dim: int,
                 demand_dim: int, hidden: int = 128, heads: int = 4) -> None:
        super().__init__(global_dim=global_dim, agent_dim=agent_dim,
                         candidate_dim=candidate_dim, demand_dim=demand_dim,
                         hidden=hidden, heads=heads)
        self.global_dim = global_dim
        self.agent_dim = agent_dim
        self.candidate_dim = candidate_dim
        self.demand_dim = demand_dim
        # R1 repair: all contexts already have width H, so no projection layer
        # is necessary. The V1 4H scorer remains available only on the V1 class.
        self.scorer = _mlp([hidden * 5, hidden, hidden, 1])

    @staticmethod
    def _require(condition: bool, code: str) -> None:
        if not condition:
            raise ActorInputContractError(code)

    def _validate_inputs(self, *, global_feats: torch.Tensor,
                         demand_feats: torch.Tensor, agent_feats: torch.Tensor,
                         agent_mask: torch.Tensor, candidate_feats: torch.Tensor,
                         pair_agent_index: torch.Tensor,
                         safe_mask: torch.Tensor) -> None:
        tensors = (global_feats, demand_feats, agent_feats, agent_mask,
                   candidate_feats, pair_agent_index, safe_mask)
        self._require(all(isinstance(value, torch.Tensor) for value in tensors),
                      "ACTOR_INPUT_NOT_TENSOR")
        self._require(global_feats.ndim == 2 and demand_feats.ndim == 2,
                      "ACTOR_GLOBAL_OR_DEMAND_SHAPE_INVALID")
        self._require(agent_feats.ndim == 3 and agent_mask.ndim == 2,
                      "ACTOR_AGENT_SHAPE_INVALID")
        self._require(candidate_feats.ndim == 3 and pair_agent_index.ndim == 2
                      and safe_mask.ndim == 2, "ACTOR_PAIR_SHAPE_INVALID")
        batch = global_feats.shape[0]
        self._require(batch > 0 and demand_feats.shape[0] == batch
                      and agent_feats.shape[0] == batch
                      and candidate_feats.shape[0] == batch,
                      "ACTOR_BATCH_SHAPE_MISMATCH")
        self._require(agent_feats.shape[:2] == agent_mask.shape,
                      "ACTOR_AGENT_MASK_SHAPE_MISMATCH")
        self._require(candidate_feats.shape[:2] == pair_agent_index.shape
                      and pair_agent_index.shape == safe_mask.shape,
                      "ACTOR_PAIR_MAPPING_SHAPE_MISMATCH")
        self._require(global_feats.shape[-1] == self.global_dim
                      and demand_feats.shape[-1] == self.demand_dim
                      and agent_feats.shape[-1] == self.agent_dim
                      and candidate_feats.shape[-1] == self.candidate_dim,
                      "ACTOR_FEATURE_DIMENSION_MISMATCH")
        self._require(agent_mask.dtype == torch.bool,
                      "ACTOR_AGENT_MASK_DTYPE_INVALID")
        self._require(safe_mask.dtype == torch.bool,
                      "ACTOR_SAFE_MASK_DTYPE_INVALID")
        self._require(pair_agent_index.dtype == torch.long,
                      "ACTOR_PAIR_AGENT_INDEX_DTYPE_INVALID")
        feature_tensors = (global_feats, demand_feats, agent_feats, candidate_feats)
        self._require(all(value.dtype.is_floating_point for value in feature_tensors),
                      "ACTOR_FEATURE_DTYPE_INVALID")
        self._require(len({value.dtype for value in feature_tensors}) == 1,
                      "ACTOR_FEATURE_DTYPE_MISMATCH")
        self._require(len({value.device for value in tensors}) == 1,
                      "ACTOR_INPUT_DEVICE_MISMATCH")
        self._require(agent_feats.shape[1] > 0
                      and bool(agent_mask.any(dim=1).all().item()),
                      "ACTOR_ACTIVE_AGENT_REQUIRED")
        self._require(all(bool(torch.isfinite(value).all().item())
                          for value in feature_tensors),
                      "ACTOR_NONFINITE_FEATURE_INPUT")
        if pair_agent_index.numel():
            self._require(int(pair_agent_index.min().item()) >= 0
                          and int(pair_agent_index.max().item()) < agent_feats.shape[1],
                          "ACTOR_PAIR_AGENT_INDEX_OUT_OF_RANGE")
            mapped_active = torch.gather(agent_mask, 1, pair_agent_index)
            self._require(bool((mapped_active | ~safe_mask).all().item()),
                          "ACTOR_SAFE_PAIR_MAPPED_TO_INACTIVE_AGENT")

    def forward(self, *, global_feats: torch.Tensor, demand_feats: torch.Tensor,
                agent_feats: torch.Tensor, agent_mask: torch.Tensor,
                candidate_feats: torch.Tensor, pair_agent_index: torch.Tensor,
                safe_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return candidate-sensitive pair logits and unchanged NO_ASSIGN logit."""
        self._validate_inputs(global_feats=global_feats, demand_feats=demand_feats,
                              agent_feats=agent_feats, agent_mask=agent_mask,
                              candidate_feats=candidate_feats,
                              pair_agent_index=pair_agent_index,
                              safe_mask=safe_mask)
        agent_ctx = self.agent_encoder(agent_feats, agent_mask)           # (B, A, H)
        global_ctx = self.global_encoder(global_feats)                    # (B, H)
        demand_ctx = self.demand_encoder(demand_feats)                    # (B, H)
        cand_ctx = self.candidate_encoder(candidate_feats)                # (B, P, H)

        _, pairs, _ = cand_ctx.shape
        index = pair_agent_index.unsqueeze(-1).expand(-1, -1, self.hidden)
        per_pair_agent = torch.gather(agent_ctx, 1, index)                # (B, P, H)
        mask_f = agent_mask.unsqueeze(-1).to(agent_ctx.dtype)
        fleet_ctx = (agent_ctx * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)

        stacked = torch.cat([
            global_ctx.unsqueeze(1).expand(-1, pairs, -1),
            demand_ctx.unsqueeze(1).expand(-1, pairs, -1),
            per_pair_agent,
            fleet_ctx.unsqueeze(1).expand(-1, pairs, -1),
            cand_ctx,
        ], dim=-1)
        logits = self.scorer(stacked).squeeze(-1)                          # (B, P)
        logits = torch.where(safe_mask, logits, torch.full_like(logits, NEG_INF))
        no_assign = self.no_assign_scorer(torch.cat([global_ctx, fleet_ctx], dim=-1))
        return logits, no_assign


def factorized_action_log_probs(*, candidate_logits: torch.Tensor, assign_logit: torch.Tensor,
                                no_assign_logit: torch.Tensor, safe_mask: torch.Tensor
                                ) -> FactorizedAssignmentDistribution:
    """Return final action log-probabilities for ASSIGN→candidate factorization.

    The stage-2 softmax only sees legal/Zero-Loss-feasible candidates.  If a row
    has no feasible candidate, the distribution collapses to NO_ASSIGN with
    probability one and no empty candidate softmax is evaluated.
    """
    if candidate_logits.ndim != 2 or safe_mask.ndim != 2:
        raise ActorInputContractError("FACTORIZED_CANDIDATE_LOGIT_MASK_SHAPE_INVALID")
    if candidate_logits.shape != safe_mask.shape:
        raise ActorInputContractError("FACTORIZED_CANDIDATE_LOGIT_MASK_SHAPE_MISMATCH")
    if safe_mask.dtype != torch.bool:
        raise ActorInputContractError("FACTORIZED_SAFE_MASK_DTYPE_INVALID")
    if assign_logit.ndim == 1:
        assign_logit = assign_logit.reshape(-1, 1)
    if no_assign_logit.ndim == 1:
        no_assign_logit = no_assign_logit.reshape(-1, 1)
    if assign_logit.shape != no_assign_logit.shape or assign_logit.shape != (candidate_logits.shape[0], 1):
        raise ActorInputContractError("FACTORIZED_GATE_LOGIT_SHAPE_INVALID")
    if candidate_logits.dtype != assign_logit.dtype or candidate_logits.dtype != no_assign_logit.dtype:
        raise ActorInputContractError("FACTORIZED_LOGIT_DTYPE_MISMATCH")
    if candidate_logits.device != assign_logit.device or candidate_logits.device != no_assign_logit.device:
        raise ActorInputContractError("FACTORIZED_LOGIT_DEVICE_MISMATCH")

    masked_candidates = torch.where(safe_mask, candidate_logits, torch.full_like(candidate_logits, NEG_INF))
    conditional_rows: list[torch.Tensor] = []
    action_rows: list[torch.Tensor] = []
    gate_rows: list[torch.Tensor] = []
    for index in range(candidate_logits.shape[0]):
        safe = safe_mask[index]
        if bool(safe.any().item()):
            gate_log_probs = torch.log_softmax(torch.cat([assign_logit[index], no_assign_logit[index]], dim=0), dim=0)
            conditional = torch.log_softmax(masked_candidates[index], dim=0)
            candidate_action = torch.where(safe, gate_log_probs[0] + conditional,
                                           torch.full_like(conditional, NEG_INF))
            no_assign_action = gate_log_probs[1].reshape(1)
            conditional_rows.append(conditional)
            action_rows.append(torch.cat([candidate_action, no_assign_action], dim=0))
            gate_rows.append(gate_log_probs)
        else:
            conditional = torch.full_like(masked_candidates[index], NEG_INF)
            candidate_action = torch.full_like(masked_candidates[index], NEG_INF)
            no_assign_action = torch.zeros((1,), dtype=candidate_logits.dtype, device=candidate_logits.device)
            gate_log_probs = torch.stack([
                torch.full((), NEG_INF, dtype=candidate_logits.dtype, device=candidate_logits.device),
                torch.zeros((), dtype=candidate_logits.dtype, device=candidate_logits.device),
            ])
            conditional_rows.append(conditional)
            action_rows.append(torch.cat([candidate_action, no_assign_action], dim=0))
            gate_rows.append(gate_log_probs)

    action_log_probs = torch.stack(action_rows, dim=0)
    return FactorizedAssignmentDistribution(
        candidate_logits=masked_candidates,
        assign_logit=assign_logit,
        no_assign_logit=no_assign_logit,
        safe_mask=safe_mask,
        gate_log_probs=torch.stack(gate_rows, dim=0),
        conditional_candidate_log_probs=torch.stack(conditional_rows, dim=0),
        candidate_action_log_probs=action_log_probs[:, :-1],
        no_assign_action_log_prob=action_log_probs[:, -1:],
    )


def factorized_masked_distribution(*, candidate_logits: torch.Tensor, assign_logit: torch.Tensor,
                                   no_assign_logit: torch.Tensor, safe_mask: torch.Tensor) -> torch.Tensor:
    """Probability view for the factorized action distribution."""
    return factorized_action_log_probs(candidate_logits=candidate_logits, assign_logit=assign_logit,
                                       no_assign_logit=no_assign_logit, safe_mask=safe_mask).action_probabilities


def factorized_log_probability(*, candidate_logits: torch.Tensor, assign_logit: torch.Tensor,
                               no_assign_logit: torch.Tensor, safe_mask: torch.Tensor,
                               action_index: torch.Tensor) -> torch.Tensor:
    """Gather PPO-compatible log-probability from the reconstructed distribution."""
    dist = factorized_action_log_probs(candidate_logits=candidate_logits, assign_logit=assign_logit,
                                       no_assign_logit=no_assign_logit, safe_mask=safe_mask)
    return dist.action_log_probs.gather(-1, action_index.unsqueeze(-1)).squeeze(-1)


class FactorizedAssignThenCandidateAssignmentHead(CandidateSensitiveMultiAgentCandidateAssignmentHead):
    """R18-R16 V3 actor: binary ASSIGN gate plus conditional candidate ranker.

    The public ``forward`` remains compatible with existing selector/PPO call
    sites by returning reconstructed final action log-probabilities in the old
    ``(pair_logits, no_assign_logit)`` slots.  The historical
    ``masked_distribution``/``masked_log_probs`` operations are idempotent on
    those normalized log-probabilities, while gradients still flow through the
    factorized computational graph.

    Gate context is detached from the candidate-ranking encoders, so a
    NO_ASSIGN-selected row trains the gate but cannot write a gradient into the
    conditional candidate-ranking path.
    """

    def __init__(self, *, global_dim: int, agent_dim: int, candidate_dim: int,
                 demand_dim: int, hidden: int = 128, heads: int = 4,
                 detach_gate_context: bool = True) -> None:
        super().__init__(global_dim=global_dim, agent_dim=agent_dim,
                         candidate_dim=candidate_dim, demand_dim=demand_dim,
                         hidden=hidden, heads=heads)
        self.detach_gate_context = bool(detach_gate_context)
        self.no_assign_scorer = nn.Identity()
        self.gate_scorer = _mlp([hidden * 2, hidden, 2])

    def forward_factorized(self, *, global_feats: torch.Tensor,
                           demand_feats: torch.Tensor, agent_feats: torch.Tensor,
                           agent_mask: torch.Tensor, candidate_feats: torch.Tensor,
                           pair_agent_index: torch.Tensor,
                           safe_mask: torch.Tensor) -> FactorizedAssignmentDistribution:
        self._validate_inputs(global_feats=global_feats, demand_feats=demand_feats,
                              agent_feats=agent_feats, agent_mask=agent_mask,
                              candidate_feats=candidate_feats,
                              pair_agent_index=pair_agent_index,
                              safe_mask=safe_mask)
        agent_ctx = self.agent_encoder(agent_feats, agent_mask)
        global_ctx = self.global_encoder(global_feats)
        demand_ctx = self.demand_encoder(demand_feats)
        cand_ctx = self.candidate_encoder(candidate_feats)

        _, pairs, _ = cand_ctx.shape
        index = pair_agent_index.unsqueeze(-1).expand(-1, -1, self.hidden)
        per_pair_agent = torch.gather(agent_ctx, 1, index) if pairs else cand_ctx.new_zeros(cand_ctx.shape)
        mask_f = agent_mask.unsqueeze(-1).to(agent_ctx.dtype)
        fleet_ctx = (agent_ctx * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        stacked = torch.cat([
            global_ctx.unsqueeze(1).expand(-1, pairs, -1),
            demand_ctx.unsqueeze(1).expand(-1, pairs, -1),
            per_pair_agent,
            fleet_ctx.unsqueeze(1).expand(-1, pairs, -1),
            cand_ctx,
        ], dim=-1)
        candidate_logits = self.scorer(stacked).squeeze(-1)
        candidate_logits = torch.where(safe_mask, candidate_logits, torch.full_like(candidate_logits, NEG_INF))
        gate_context = torch.cat([global_ctx, fleet_ctx], dim=-1)
        if self.detach_gate_context:
            gate_context = gate_context.detach()
        gate_logits = self.gate_scorer(gate_context)
        return factorized_action_log_probs(candidate_logits=candidate_logits,
                                           assign_logit=gate_logits[:, 0:1],
                                           no_assign_logit=gate_logits[:, 1:2],
                                           safe_mask=safe_mask)

    def forward(self, *, global_feats: torch.Tensor, demand_feats: torch.Tensor,
                agent_feats: torch.Tensor, agent_mask: torch.Tensor,
                candidate_feats: torch.Tensor, pair_agent_index: torch.Tensor,
                safe_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward_factorized(global_feats=global_feats, demand_feats=demand_feats,
                                       agent_feats=agent_feats, agent_mask=agent_mask,
                                       candidate_feats=candidate_feats,
                                       pair_agent_index=pair_agent_index,
                                       safe_mask=safe_mask)
        return dist.candidate_action_log_probs, dist.no_assign_action_log_prob


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
