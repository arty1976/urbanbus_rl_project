"""BT8-R4 candidate-plan execution and credit-identity contract.

This module does not generate candidates, evaluate Zero-Loss, compute Reward
V2, execute a simulator, or perform optimization.  It binds an already-frozen
safe Local-Search candidate to an explicit, disposable applied route-plan
digest so a future Joint Assignment rollout cannot collapse candidate identity
to an agent-only SERVE action before credit attribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import joint_candidate_support_snapshot as SS


CONTRACT_ID = "LS3_BT8_R4_CANDIDATE_PLAN_EXECUTION_AND_CREDIT_IDENTITY_V1"
CONTRACT_VERSION = "BT8_R4_V1"


class CandidatePlanExecutionError(RuntimeError):
    """Candidate-plan identity or causal binding violation."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise CandidatePlanExecutionError(code, detail)


def canonical_digest(payload: Mapping[str, Any]) -> str:
    return SS.canonical_sha256(dict(payload))


@dataclass(frozen=True)
class CandidatePlanBinding:
    """An evaluated candidate and its concrete, identity-bearing route plan."""

    agent_id: str
    candidate_id: str
    source_candidate_digest: str
    source_state_digest: str
    base_plan_stop_ids: tuple[str, ...]
    candidate_path_stop_ids: tuple[str, ...]
    pickup_position: int
    dropoff_position: int
    candidate_plan_digest: str

    @classmethod
    def from_evidence(cls, row: SS.CandidateEvidence) -> "CandidatePlanBinding":
        local = row.payload.get("local_search")
        _require(isinstance(local, Mapping), "CANDIDATE_PLAN_LOCAL_SEARCH_EVIDENCE_MISSING")
        path = tuple(str(value) for value in local.get("path_stop_ids", ()))
        base = tuple(str(value) for value in row.payload.get("plan_stop_ids", ()))
        pickup = int(row.payload.get("pickup_position", -1))
        dropoff = int(row.payload.get("dropoff_position", -1))
        _require(len(path) >= 2, "CANDIDATE_PLAN_PATH_INVALID")
        _require(len(base) >= 2, "CANDIDATE_PLAN_BASE_ROUTE_INVALID")
        _require(0 <= pickup <= dropoff < len(base), "CANDIDATE_PLAN_INSERTION_POSITION_INVALID")
        _require(base[pickup] == path[0] and base[dropoff] == path[-1],
                 "CANDIDATE_PLAN_ENDPOINT_ROUTE_MISMATCH")
        _require(row.candidate_id == row.source_candidate_id,
                 "CANDIDATE_PLAN_IDENTITY_TAMPERED")
        digest = canonical_digest({
            "contract": CONTRACT_ID, "agent_id": row.agent_id,
            "candidate_id": row.candidate_id,
            "source_candidate_digest": row.source_candidate_digest,
            "source_state_digest": row.source_state_digest,
            "base_plan_stop_ids": list(base), "candidate_path_stop_ids": list(path),
            "pickup_position": pickup, "dropoff_position": dropoff,
        })
        return cls(agent_id=str(row.agent_id), candidate_id=str(row.candidate_id),
                   source_candidate_digest=str(row.source_candidate_digest),
                   source_state_digest=str(row.source_state_digest),
                   base_plan_stop_ids=base, candidate_path_stop_ids=path,
                   pickup_position=pickup, dropoff_position=dropoff,
                   candidate_plan_digest=digest)


@dataclass(frozen=True)
class AppliedCandidatePlan:
    """Pure disposable plan application result; source state is never mutated."""

    agent_id: str
    candidate_id: str
    candidate_plan_digest: str
    applied_plan_stop_ids: tuple[str, ...]
    applied_plan_digest: str
    applied_state_digest: str


def apply_on_disposable_route(binding: CandidatePlanBinding) -> AppliedCandidatePlan:
    """Splice the selected Local-Search path into the declared route interval.

    The endpoints are validated against the frozen base route.  This is the
    future execution authority for candidate-plan identity: a candidate is not
    represented by an agent-only action, and no source state object is touched.
    """
    base, path = binding.base_plan_stop_ids, binding.candidate_path_stop_ids
    applied = base[:binding.pickup_position] + path + base[binding.dropoff_position + 1:]
    _require(len(applied) >= 2, "APPLIED_CANDIDATE_PLAN_EMPTY")
    plan_digest = canonical_digest({"contract": CONTRACT_ID, "agent_id": binding.agent_id,
                                    "candidate_id": binding.candidate_id,
                                    "candidate_plan_digest": binding.candidate_plan_digest,
                                    "applied_plan_stop_ids": list(applied)})
    state_digest = canonical_digest({"contract": CONTRACT_ID,
                                     "source_state_digest": binding.source_state_digest,
                                     "applied_plan_digest": plan_digest})
    return AppliedCandidatePlan(agent_id=binding.agent_id, candidate_id=binding.candidate_id,
        candidate_plan_digest=binding.candidate_plan_digest, applied_plan_stop_ids=applied,
        applied_plan_digest=plan_digest, applied_state_digest=state_digest)


def binding_for_selected_pair(snapshot: SS.JointCandidateSupportSnapshot, *, agent_id: str,
                              candidate_id: str) -> CandidatePlanBinding:
    try:
        snapshot.assert_selectable(agent_id=agent_id, candidate_id=candidate_id)
    except SS.CandidateSupportError as exc:
        # Keep the public R4 execution boundary fail-closed on one explicit
        # error type.  The original support-contract code is retained so an
        # audit can still distinguish the precise rejection reason.
        raise CandidatePlanExecutionError(exc.code, str(exc)) from exc
    selected = next((row for row in snapshot.safe
                     if row.agent_id == str(agent_id) and row.candidate_id == str(candidate_id)), None)
    _require(selected is not None, "SELECTED_CANDIDATE_PLAN_MISSING")
    return CandidatePlanBinding.from_evidence(selected)


def candidate_plan_credit_row(*, decision_id: str, agent_id: str, candidate_id: str,
                              applied: AppliedCandidatePlan, transition_id: str,
                              trajectory_id: str, team_reward: float | None = None,
                              critic_target: float | None = None,
                              gae_advantage: float | None = None,
                              policy_gradient_contribution: float | None = None) -> dict[str, Any]:
    """Build an identity-complete future credit row without computing credit."""
    _require(agent_id == applied.agent_id and candidate_id == applied.candidate_id,
             "SELECTED_APPLIED_CANDIDATE_MISMATCH")
    _require(bool(decision_id) and bool(transition_id) and bool(trajectory_id),
             "CREDIT_IDENTITY_REQUIRED_FIELD_MISSING")
    return {
        "contract_id": CONTRACT_ID, "decision_id": str(decision_id),
        "agent_id": str(agent_id), "candidate_id": str(candidate_id),
        "candidate_plan_digest": applied.candidate_plan_digest,
        "applied_plan_digest": applied.applied_plan_digest,
        "transition_id": str(transition_id), "trajectory_id": str(trajectory_id),
        "Team Reward": team_reward, "critic target": critic_target,
        "GAE advantage": gae_advantage,
        "policy-gradient contribution": policy_gradient_contribution,
    }


def validate_candidate_plan_credit_row(row: Mapping[str, Any]) -> None:
    required = ("decision_id", "agent_id", "candidate_id", "candidate_plan_digest",
                "applied_plan_digest", "transition_id", "trajectory_id")
    _require(all(bool(row.get(name)) for name in required),
             "CREDIT_IDENTITY_REQUIRED_FIELD_MISSING")
    _require(row.get("contract_id") == CONTRACT_ID, "CREDIT_IDENTITY_CONTRACT_MISMATCH")
    _require(row.get("candidate_plan_digest") != row.get("applied_plan_digest"),
             "CANDIDATE_PLAN_EXECUTION_COLLAPSE")


CANDIDATE_PLAN_CREDIT_CONTRACT = {
    "contract_id": CONTRACT_ID,
    "version": CONTRACT_VERSION,
    "identity_chain": "rollout candidate == applied candidate plan == credit-attributed candidate",
    "required_future_row_fields": ["decision_id", "agent_id", "candidate_id",
        "candidate_plan_digest", "applied_plan_digest", "transition_id", "trajectory_id",
        "Team Reward", "critic target", "GAE advantage", "policy-gradient contribution"],
    "reward_v2_modified": False,
    "zero_loss_recomputed_at_credit": False,
    "source_state_mutation_allowed": False,
    "disposable_shadow_application_only": True,
}
