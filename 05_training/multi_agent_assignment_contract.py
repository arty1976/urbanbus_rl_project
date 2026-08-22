#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-JA0/JA1 multi-agent joint service-assignment contract.

The earlier reading -- each bus picks one of its own Local Search candidates --
is retired.  The canonical decision is cooperative and joint:

    every agent proposes service-plan candidates
            -> Zero-Loss filters each (agent, candidate) pair
            -> the whole safe set goes to one selector
            -> the selector answers *which agent* serves the demand and
               *under which plan*, seeing the whole city at once

So the selector's job is `COOPERATIVE_MULTI_AGENT_SERVICE_ASSIGNMENT`, not path
choice.  Nearest vehicle does not win by construction, and lowest Local Search
cost does not win by construction: both are inputs to a learned decision that
may prefer a farther bus to preserve coverage or to keep a bus free for demand
it expects nearby.

Two decision levels stay separate
---------------------------------
    Level 1  cooperative assignment  which agent, which safe service plan
    Level 2  operational control     HOLD / SERVE / CONDITIONAL_SKIP along the
                                     plan an agent already has

Level 2 is the promoted 3-action actor and is untouched here.  Its three actions
are never reinterpreted as candidate slots; this layer is additional, not a
replacement, and the two action spaces are never merged.

Identity
--------
A candidate alone is not a decision.  The canonical identity is the pair
`(agent_id, candidate_id)`, because the same plan proposed by two vehicles is two
different decisions with different consequences.

NO_ASSIGN
---------
Every decision group carries `NO_ASSIGN` -- keep every vehicle on its current
plan -- as a real, always-available option.  It is mandatory when no safe pair
exists, and no fallback route is ever invented to avoid it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

CONTRACT_ID = "MULTI_AGENT_JOINT_SERVICE_ASSIGNMENT_CONTRACT_V1"
CONTRACT_VERSION = "LS3_JA0_V1"

ASSIGNMENT_SEMANTICS = "COOPERATIVE_MULTI_AGENT_SERVICE_ASSIGNMENT"
NO_ASSIGN = "NO_ASSIGN_KEEP_CURRENT_PLANS"
NON_SELECTED_AGENT_OUTCOME = "KEEP_CURRENT_PLAN"
SELECTION_SEMANTICS = "SHADOW_UNTRAINED_ARCHITECTURE_VALIDATION_ONLY"

OPERATIONAL_HEAD_ACTIONS = ("HOLD", "SERVE", "CONDITIONAL_SKIP")

ROLE_CONTRACT = {
    "contract_id": CONTRACT_ID,
    "assignment_semantics": ASSIGNMENT_SEMANTICS,
    "canonical_identity": ["agent_id", "candidate_id"],
    "candidate_id_alone_is_a_decision": False,
    "roles": {
        "gatv2": "shared city graph / demand / spatial context",
        "local_search": "per-agent feasible service-plan candidate generator",
        "zero_loss": "hard admission filter at epsilon 0.0, per (agent, candidate)",
        "joint_assignment_head": "selects which agent serves the demand, under which safe plan",
        "operational_head": "HOLD / SERVE / CONDITIONAL_SKIP along an already assigned plan",
        "simulator": "the only component that changes state",
    },
    "levels": {"level_1": "cooperative service assignment",
               "level_2": "operational control on the assigned plan"},
    "action_spaces_merged": False,
    "operational_head_actions": list(OPERATIONAL_HEAD_ACTIONS),
    "operational_head_reinterpreted_as_candidates": False,
    "nearest_vehicle_hardcoded": False,
    "lowest_local_search_cost_hardcoded": False,
    "local_search_score_is_final_authority": False,
    "no_assign_option": NO_ASSIGN,
    "non_selected_agents": NON_SELECTED_AGENT_OUTCOME,
    "zero_loss_as_reward": False,
    "zero_loss_as_soft_penalty": False,
    "unsafe_pair_selectable": False,
    "python_builtin_hash_used": False,
}

# Everything scale-shaped is configuration; nothing here assumes a fleet size,
# a candidate count or a district.
SCALE_CONTRACT = {
    "agent_count": "variable",
    "candidates_per_agent": "variable",
    "decision_groups": "variable, schema is multi-group capable",
    "agent_count_hardcoded": False,
    "candidate_count_hardcoded": False,
    "district_hardcoded": False,
    "scoring_complexity": "linear in the number of proposed (agent, candidate) pairs",
    "joint_action_enumeration": False,
}


class AssignmentContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def pair_identity(*, decision_group_id: str, agent_id: str, candidate_id: str) -> str:
    payload = "|".join([CONTRACT_ID, CONTRACT_VERSION, decision_group_id, str(agent_id),
                        str(candidate_id)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AgentCandidatePair:
    """One (agent, candidate) decision, with the evidence behind it."""

    decision_group_id: str
    agent_id: str
    candidate_id: str
    operational_state_id: str
    request_identity: str
    pickup_stop_id: str
    dropoff_stop_id: str
    pickup_position: Any
    dropoff_position: Any
    plan_stop_ids: Tuple[str, ...]
    local_search_features: Mapping[str, float]
    zero_loss_status: str
    onboard_passenger_count: int
    max_delta_eta_sec: Optional[int]
    zero_loss_evidence_digest: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def pair_id(self) -> str:
        return pair_identity(decision_group_id=self.decision_group_id,
                             agent_id=self.agent_id, candidate_id=self.candidate_id)

    @property
    def is_safe(self) -> bool:
        return self.zero_loss_status == "PASS"

    def payload(self) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id, "decision_group_id": self.decision_group_id,
            "agent_id": self.agent_id, "candidate_id": self.candidate_id,
            "operational_state_id": self.operational_state_id,
            "request_identity": self.request_identity,
            "pickup_stop_id": self.pickup_stop_id, "dropoff_stop_id": self.dropoff_stop_id,
            "pickup_position": self.pickup_position, "dropoff_position": self.dropoff_position,
            "plan_stop_ids": list(self.plan_stop_ids),
            "local_search_features": dict(self.local_search_features),
            "zero_loss_status": self.zero_loss_status,
            "onboard_passenger_count": self.onboard_passenger_count,
            "max_delta_eta_sec": self.max_delta_eta_sec,
            "zero_loss_evidence_digest": self.zero_loss_evidence_digest,
            "is_safe": self.is_safe, "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class AgentContext:
    """Per-agent state the selector may condition on.  No agent id as a feature."""

    agent_id: str
    current_stop_id: str
    onboard_passenger_count: int
    plan_length: int
    remaining_plan_seconds: float
    candidate_availability: int

    def feature_vector(self) -> List[float]:
        return [float(self.onboard_passenger_count), float(self.plan_length),
                float(self.remaining_plan_seconds), float(self.candidate_availability)]

    FEATURE_NAMES = ("onboard_passenger_count", "plan_length", "remaining_plan_seconds",
                     "candidate_availability")


@dataclass
class JointSafeCandidateSet:
    """Only Zero-Loss PASS pairs are selectable; rejects are kept for audit only."""

    decision_group_id: str
    agents: List[AgentContext]
    safe_pairs: List[AgentCandidatePair]
    rejected_pairs: List[AgentCandidatePair]
    demand_context: Mapping[str, Any] = field(default_factory=dict)
    global_context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unsafe = [p.pair_id for p in self.safe_pairs if not p.is_safe]
        if unsafe:
            raise AssignmentContractError("UNSAFE_PAIR_IN_SELECTABLE_SET", str(unsafe[:3]))
        overlap = {p.pair_id for p in self.safe_pairs} & {p.pair_id for p in self.rejected_pairs}
        if overlap:
            raise AssignmentContractError("PAIR_IN_BOTH_SETS", str(sorted(overlap)[:3]))

    @property
    def safe_pair_count(self) -> int:
        return len(self.safe_pairs)

    @property
    def agent_count(self) -> int:
        return len(self.agents)

    def canonical_pairs(self) -> List[AgentCandidatePair]:
        """Ordering derives from identity, never from arrival order."""
        return sorted(self.safe_pairs, key=lambda p: (p.agent_id, p.candidate_id))

    def payload(self) -> Dict[str, Any]:
        return {
            "decision_group_id": self.decision_group_id,
            "agent_count": self.agent_count, "safe_pair_count": self.safe_pair_count,
            "rejected_pair_count": len(self.rejected_pairs),
            "agents": [{"agent_id": a.agent_id, "current_stop_id": a.current_stop_id,
                        "onboard_passenger_count": a.onboard_passenger_count,
                        "plan_length": a.plan_length,
                        "remaining_plan_seconds": a.remaining_plan_seconds,
                        "candidate_availability": a.candidate_availability}
                       for a in self.agents],
            "safe_pairs": [p.payload() for p in self.canonical_pairs()],
            "rejected_pairs": [p.payload() for p in self.rejected_pairs],
            "no_assign_option": NO_ASSIGN,
            "demand_context": dict(self.demand_context),
            "selectable_intersect_rejected": [],
        }


LOCAL_SEARCH_FEATURE_NAMES = ("distance_m", "time_sec", "generalized_cost", "hop_count",
                              "pickup_position", "dropoff_position", "onboard_passenger_count",
                              "max_delta_eta_sec")


def candidate_feature_vector(pair: AgentCandidatePair) -> List[float]:
    feats = dict(pair.local_search_features)
    return [
        float(feats.get("distance_m", 0.0)), float(feats.get("time_sec", 0.0)),
        float(feats.get("generalized_cost", 0.0)), float(feats.get("hop_count", 0.0)),
        float(pair.pickup_position if isinstance(pair.pickup_position, (int, float)) else 0.0),
        float(pair.dropoff_position if isinstance(pair.dropoff_position, (int, float)) else 0.0),
        float(pair.onboard_passenger_count),
        float(pair.max_delta_eta_sec if pair.max_delta_eta_sec is not None else 0.0),
    ]


def set_digest(candidate_set: JointSafeCandidateSet) -> str:
    import json
    return hashlib.sha256(json.dumps(candidate_set.payload(), sort_keys=True,
                                     default=str).encode("utf-8")).hexdigest()
