"""Immutable candidate-plan-aware authoritative state bridge for BT8-R4A.

It owns only the selected agent's route-plan state.  It does not change Reward
V2, Zero-Loss, Local Search, GATv2, the operational causal KPI bridge, or PPO.
Every selected plan was already materialized in a frozen candidate snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import joint_candidate_plan_execution as PE
import joint_candidate_support_snapshot as SS
import simulator_authorization as AUTH


CONTRACT_ID = "LS3_BT8_R4A_CANDIDATE_PLAN_CAUSAL_BRIDGE_V1"
NO_ASSIGN = SS.NO_ASSIGN


class CandidatePlanBridgeError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise CandidatePlanBridgeError(code, detail)


def digest(payload: Mapping[str, Any]) -> str:
    return SS.canonical_sha256(dict(payload))


@dataclass(frozen=True)
class CandidatePlanState:
    agent_id: str
    version: int
    route_plan_stop_ids: tuple[str, ...]
    source_state_digest: str
    state_digest: str

    @classmethod
    def from_binding(cls, binding: PE.CandidatePlanBinding, *, version: int = 0) -> "CandidatePlanState":
        route = tuple(binding.base_plan_stop_ids)
        state_digest = digest({"contract": CONTRACT_ID, "agent_id": binding.agent_id, "version": int(version),
                               "route_plan_stop_ids": list(route), "source_state_digest": binding.source_state_digest})
        return cls(agent_id=binding.agent_id, version=int(version), route_plan_stop_ids=route,
                   source_state_digest=binding.source_state_digest, state_digest=state_digest)


@dataclass(frozen=True)
class ImmutableCandidatePlan:
    decision_id: str
    agent_id: str
    candidate_id: str
    candidate_plan: PE.CandidatePlanBinding
    candidate_plan_digest: str
    source_state_version: int
    source_state_digest: str
    zero_loss_evidence_digest: str
    candidate_feature_digest: str
    candidate_support_digest: str

    @classmethod
    def from_snapshot(cls, *, decision_id: str, snapshot: SS.JointCandidateSupportSnapshot,
                      agent_id: str, candidate_id: str, source_state_version: int) -> "ImmutableCandidatePlan":
        try:
            snapshot.assert_selectable(agent_id=agent_id, candidate_id=candidate_id)
        except SS.CandidateSupportError as exc:
            raise CandidatePlanBridgeError(exc.code, str(exc)) from exc
        evidence = next((row for row in snapshot.safe if row.agent_id == agent_id and row.candidate_id == candidate_id), None)
        require(evidence is not None, "SELECTED_CANDIDATE_MISSING_FROM_FROZEN_SNAPSHOT")
        binding = PE.CandidatePlanBinding.from_evidence(evidence)
        features = {name: float(value) for name, value in sorted(evidence.local_search_features.items())}
        return cls(decision_id=str(decision_id), agent_id=str(agent_id), candidate_id=str(candidate_id),
                   candidate_plan=binding, candidate_plan_digest=binding.candidate_plan_digest,
                   source_state_version=int(source_state_version), source_state_digest=binding.source_state_digest,
                   zero_loss_evidence_digest=evidence.zero_loss_evidence_digest,
                   candidate_feature_digest=digest(features), candidate_support_digest=snapshot.snapshot_digest)


@dataclass(frozen=True)
class PurePlanTransition:
    next_state: CandidatePlanState
    transition_id: str
    selected_candidate_id: str
    applied_candidate_id: str
    credited_candidate_id: str
    candidate_plan_digest: str
    applied_plan_digest: str
    events: tuple[Mapping[str, Any], ...]
    no_assign: bool
    serve_fallback_used: bool = False


def _state_from_route(source: CandidatePlanState, route: tuple[str, ...]) -> CandidatePlanState:
    version = source.version + 1
    state_digest = digest({"contract": CONTRACT_ID, "agent_id": source.agent_id, "version": version,
                           "route_plan_stop_ids": list(route), "source_state_digest": source.source_state_digest})
    return CandidatePlanState(agent_id=source.agent_id, version=version, route_plan_stop_ids=route,
                              source_state_digest=source.source_state_digest, state_digest=state_digest)


def apply_candidate_plan_pure(source_state: CandidatePlanState, candidate_plan: ImmutableCandidatePlan) -> PurePlanTransition:
    """Pure route-plan application; ``source_state`` is never mutated."""
    binding = candidate_plan.candidate_plan
    require(source_state.agent_id == candidate_plan.agent_id == binding.agent_id, "CANDIDATE_PLAN_AGENT_MISMATCH")
    require(source_state.version == candidate_plan.source_state_version, "SOURCE_STATE_VERSION_MISMATCH")
    require(source_state.source_state_digest == candidate_plan.source_state_digest, "SOURCE_STATE_DIGEST_MISMATCH")
    require(source_state.route_plan_stop_ids == binding.base_plan_stop_ids, "SOURCE_ROUTE_PLAN_MISMATCH")
    require(candidate_plan.candidate_plan_digest == binding.candidate_plan_digest, "CANDIDATE_PLAN_DIGEST_MISMATCH")
    route = binding.base_plan_stop_ids[:binding.pickup_position] + binding.candidate_path_stop_ids + binding.base_plan_stop_ids[binding.dropoff_position + 1:]
    require(len(route) >= 2, "APPLIED_CANDIDATE_PLAN_EMPTY")
    next_state = _state_from_route(source_state, route)
    applied_digest = digest({"contract": CONTRACT_ID, "agent_id": source_state.agent_id,
                             "candidate_id": candidate_plan.candidate_id, "candidate_plan_digest": candidate_plan.candidate_plan_digest,
                             "applied_route_plan_stop_ids": list(route), "next_state_digest": next_state.state_digest})
    transition_id = digest({"contract": CONTRACT_ID, "decision_id": candidate_plan.decision_id,
                            "candidate_id": candidate_plan.candidate_id, "source_state_digest": source_state.state_digest,
                            "next_state_digest": next_state.state_digest, "applied_plan_digest": applied_digest})
    event = {"transition_id": transition_id, "decision_id": candidate_plan.decision_id, "agent_id": source_state.agent_id,
             "candidate_id": candidate_plan.candidate_id, "candidate_plan_digest": candidate_plan.candidate_plan_digest,
             "applied_plan_digest": applied_digest, "source_state_digest": source_state.state_digest,
             "next_state_digest": next_state.state_digest, "no_assign": False, "serve_fallback_used": False}
    return PurePlanTransition(next_state=next_state, transition_id=transition_id,
        selected_candidate_id=candidate_plan.candidate_id, applied_candidate_id=candidate_plan.candidate_id,
        credited_candidate_id=candidate_plan.candidate_id, candidate_plan_digest=candidate_plan.candidate_plan_digest,
        applied_plan_digest=applied_digest, events=(event,), no_assign=False)


def apply_no_assign_pure(source_state: CandidatePlanState, *, decision_id: str) -> PurePlanTransition:
    """Explicit NO_ASSIGN transition: retain route, advance only state version."""
    next_state = _state_from_route(source_state, source_state.route_plan_stop_ids)
    candidate_digest = digest({"contract": CONTRACT_ID, "decision_id": decision_id, "candidate_id": NO_ASSIGN,
                               "source_state_digest": source_state.state_digest})
    applied_digest = digest({"contract": CONTRACT_ID, "candidate_id": NO_ASSIGN,
                             "retained_route_plan_stop_ids": list(source_state.route_plan_stop_ids),
                             "next_state_digest": next_state.state_digest})
    transition_id = digest({"contract": CONTRACT_ID, "decision_id": decision_id, "candidate_id": NO_ASSIGN,
                            "source_state_digest": source_state.state_digest, "next_state_digest": next_state.state_digest})
    event = {"transition_id": transition_id, "decision_id": decision_id, "agent_id": source_state.agent_id,
             "candidate_id": NO_ASSIGN, "candidate_plan_digest": candidate_digest, "applied_plan_digest": applied_digest,
             "source_state_digest": source_state.state_digest, "next_state_digest": next_state.state_digest,
             "no_assign": True, "serve_fallback_used": False}
    return PurePlanTransition(next_state=next_state, transition_id=transition_id,
        selected_candidate_id=NO_ASSIGN, applied_candidate_id=NO_ASSIGN, credited_candidate_id=NO_ASSIGN,
        candidate_plan_digest=candidate_digest, applied_plan_digest=applied_digest, events=(event,), no_assign=True)


class CandidatePlanAuthoritativeBridge:
    """Atomic live state holder; all validation precedes exactly one assignment."""

    def __init__(self, *, initial_state: CandidatePlanState, snapshot: SS.JointCandidateSupportSnapshot) -> None:
        self._state = initial_state
        self._snapshot = snapshot
        self._committed_transition_ids: set[str] = set()
        self._committed_decision_ids: set[str] = set()

    @property
    def state(self) -> CandidatePlanState:
        return self._state

    def _commit(self, transition: PurePlanTransition, *, decision_id: str) -> PurePlanTransition:
        require(str(decision_id) not in self._committed_decision_ids, "DOUBLE_AUTHORITATIVE_COMMIT")
        require(transition.transition_id not in self._committed_transition_ids, "DOUBLE_AUTHORITATIVE_COMMIT")
        require(transition.selected_candidate_id == transition.applied_candidate_id == transition.credited_candidate_id,
                "SELECTED_APPLIED_CREDITED_IDENTITY_MISMATCH")
        require(transition.serve_fallback_used is False, "SERVE_FALLBACK_FORBIDDEN")
        # The only live mutation in this module occurs after every invariant.
        self._state = transition.next_state
        self._committed_transition_ids.add(transition.transition_id)
        self._committed_decision_ids.add(str(decision_id))
        return transition

    def commit_candidate(self, *, candidate: ImmutableCandidatePlan, expected_source_version: int,
                         expected_source_digest: str, expected_support_digest: str,
                         expected_candidate_plan_digest: str) -> PurePlanTransition:
        AUTH.require_capability(AUTH.SIMULATOR_EXECUTION, site="joint_candidate_plan_causal_bridge.py::commit_candidate")
        require(candidate.decision_id not in self._committed_decision_ids, "DOUBLE_AUTHORITATIVE_COMMIT")
        require(expected_source_version == self._state.version, "SOURCE_STATE_VERSION_MISMATCH")
        require(expected_source_digest == self._state.state_digest, "SOURCE_STATE_DIGEST_MISMATCH")
        require(expected_support_digest == self._snapshot.snapshot_digest == candidate.candidate_support_digest,
                "CANDIDATE_SNAPSHOT_DIGEST_MISMATCH")
        require(expected_candidate_plan_digest == candidate.candidate_plan_digest, "CANDIDATE_PLAN_DIGEST_MISMATCH")
        try:
            self._snapshot.assert_selectable(agent_id=candidate.agent_id, candidate_id=candidate.candidate_id)
        except SS.CandidateSupportError as exc:
            raise CandidatePlanBridgeError(exc.code, str(exc)) from exc
        return self._commit(apply_candidate_plan_pure(self._state, candidate), decision_id=candidate.decision_id)

    def commit_no_assign(self, *, decision_id: str, expected_source_version: int,
                         expected_source_digest: str, expected_support_digest: str) -> PurePlanTransition:
        AUTH.require_capability(AUTH.SIMULATOR_EXECUTION, site="joint_candidate_plan_causal_bridge.py::commit_no_assign")
        require(str(decision_id) not in self._committed_decision_ids, "DOUBLE_AUTHORITATIVE_COMMIT")
        require(expected_source_version == self._state.version, "SOURCE_STATE_VERSION_MISMATCH")
        require(expected_source_digest == self._state.state_digest, "SOURCE_STATE_DIGEST_MISMATCH")
        require(expected_support_digest == self._snapshot.snapshot_digest, "CANDIDATE_SNAPSHOT_DIGEST_MISMATCH")
        return self._commit(apply_no_assign_pure(self._state, decision_id=decision_id), decision_id=decision_id)


CANDIDATE_PLAN_BRIDGE_CONTRACT = {
    "contract_id": CONTRACT_ID,
    "pure_transition": "apply_candidate_plan_pure(source_state, candidate_plan) -> next_state/events/applied_plan_digest",
    "authoritative_commit": "capability + source/snapshot/Zero-Loss/digest validation -> pure transition -> one atomic commit",
    "candidate_regeneration_after_selection": False,
    "serve_fallback": False,
    "source_state_mutation": False,
    "no_assign": "explicit route-retaining transition with its own record",
    "identity_chain": "selected candidate_id == applied candidate_id == credited candidate_id",
}
