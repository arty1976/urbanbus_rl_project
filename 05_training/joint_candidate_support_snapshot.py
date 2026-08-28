#!/usr/bin/env python3
"""Immutable, fail-closed agent×candidate support snapshots.

This module is deliberately only an adapter between existing authorities:
Local Search proposes route candidates, the frozen Zero-Loss adapter labels each
pair PASS or REJECT, and this module freezes the resulting joint support for a
single decision.  It does not generate routes, evaluate Zero-Loss, select an
action, mutate state, or contain training code.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple


NO_ASSIGN = "NO_ASSIGN_KEEP_CURRENT_PLANS"
PASS = "PASS"
REJECT = "REJECT"


class CandidateSupportError(RuntimeError):
    """A malformed or regenerated decision support is never silently repaired."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     allow_nan=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateEvidence:
    """One evaluated local-search candidate for one agent.

    ``candidate_id`` must be the Local Search identity itself.  The agent id is
    identity metadata for the pair and may not be inserted into a numeric
    feature, cost, rank, or score by this adapter.
    """

    agent_id: str
    candidate_id: str
    source_candidate_id: str
    source_candidate_digest: str
    source_generalized_cost: float
    local_search_features: Mapping[str, float]
    zero_loss_status: str
    zero_loss_evidence_digest: str
    source_state_digest: str
    payload: Mapping[str, Any]

    def canonical_payload(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "candidate_id": self.candidate_id,
            "source_candidate_id": self.source_candidate_id,
            "source_candidate_digest": self.source_candidate_digest,
            "source_generalized_cost": float(self.source_generalized_cost),
            "local_search_features": {str(k): float(v)
                                      for k, v in sorted(self.local_search_features.items())},
            "zero_loss_status": self.zero_loss_status,
            "zero_loss_evidence_digest": self.zero_loss_evidence_digest,
            "source_state_digest": self.source_state_digest,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class JointCandidateSupportSnapshot:
    """A materialized support that may be replayed but never regenerated."""

    decision_group_id: str
    no_assign_option: str
    safe: Tuple[CandidateEvidence, ...]
    rejected: Tuple[CandidateEvidence, ...]
    raw_candidate_ids_by_agent: Tuple[Tuple[str, Tuple[str, ...]], ...]
    source_state_digests: Tuple[Tuple[str, str], ...]
    snapshot_digest: str

    @property
    def all_evaluated(self) -> Tuple[CandidateEvidence, ...]:
        return self.safe + self.rejected

    def canonical_pairs(self) -> Tuple[CandidateEvidence, ...]:
        return tuple(sorted(self.safe, key=lambda row: (row.agent_id, row.candidate_id)))

    def replay_guard(self, *, support_digest: str, regeneration_requested: bool = False) -> None:
        if regeneration_requested:
            raise CandidateSupportError("CANDIDATE_REGENERATION_FORBIDDEN_AT_REPLAY")
        if str(support_digest) != self.snapshot_digest:
            raise CandidateSupportError("ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE")

    def assert_selectable(self, *, agent_id: str, candidate_id: str) -> None:
        key = (str(agent_id), str(candidate_id))
        if key not in {(row.agent_id, row.candidate_id) for row in self.safe}:
            raise CandidateSupportError("REJECTED_OR_UNKNOWN_CANDIDATE_FORCED_SELECTION", repr(key))

    def payload(self) -> Dict[str, Any]:
        return {
            "decision_group_id": self.decision_group_id,
            "no_assign_option": self.no_assign_option,
            "safe": [row.canonical_payload() for row in self.canonical_pairs()],
            "rejected": [row.canonical_payload() for row in
                         sorted(self.rejected, key=lambda row: (row.agent_id, row.candidate_id))],
            "raw_candidate_ids_by_agent": {
                agent: list(ids) for agent, ids in self.raw_candidate_ids_by_agent
            },
            "source_state_digests": dict(self.source_state_digests),
            "snapshot_digest": self.snapshot_digest,
        }


def _finite(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise CandidateSupportError("NON_FINITE_CANDIDATE_FEATURE", name)
    return out


def _normalized_raw_ids(raw_candidate_ids_by_agent: Mapping[str, Iterable[str]]) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    out = []
    for agent_id, candidate_ids in raw_candidate_ids_by_agent.items():
        ids = tuple(sorted(str(item) for item in candidate_ids))
        if len(ids) != len(set(ids)):
            raise CandidateSupportError("DUPLICATE_LOCAL_SEARCH_CANDIDATE_ID", str(agent_id))
        out.append((str(agent_id), ids))
    return tuple(sorted(out))


def _validate_evidence(rows: Sequence[CandidateEvidence], raw: Mapping[str, Tuple[str, ...]]) -> None:
    keys = [(row.agent_id, row.candidate_id) for row in rows]
    if len(keys) != len(set(keys)):
        raise CandidateSupportError("DUPLICATE_AGENT_CANDIDATE_PAIR")
    for row in rows:
        if row.candidate_id != row.source_candidate_id:
            raise CandidateSupportError("CANDIDATE_ID_TAMPERED", row.agent_id)
        local_search_payload = row.payload.get("local_search")
        if not isinstance(local_search_payload, Mapping) \
                or canonical_sha256(local_search_payload) != row.source_candidate_digest:
            raise CandidateSupportError("CANDIDATE_SOURCE_EVIDENCE_TAMPERED", row.candidate_id)
        if row.agent_id not in raw or row.candidate_id not in raw[row.agent_id]:
            raise CandidateSupportError("CANDIDATE_NOT_FROM_DECLARED_LOCAL_SEARCH_SUPPORT", row.agent_id)
        source_cost = _finite("source_generalized_cost", row.source_generalized_cost)
        recorded_cost = _finite("generalized_cost", row.local_search_features.get("generalized_cost", source_cost))
        if recorded_cost != source_cost:
            raise CandidateSupportError("GENERALIZED_COST_SOURCE_MISMATCH", row.candidate_id)
        if row.zero_loss_status not in (PASS, REJECT):
            raise CandidateSupportError("ZERO_LOSS_STATUS_INVALID", row.zero_loss_status)
        if not row.zero_loss_evidence_digest or not row.source_state_digest:
            raise CandidateSupportError("CANDIDATE_EVIDENCE_INCOMPLETE", row.candidate_id)

    evaluated = {agent: {row.candidate_id for row in rows if row.agent_id == agent} for agent in raw}
    for agent, ids in raw.items():
        if evaluated.get(agent, set()) != set(ids):
            raise CandidateSupportError("FORCED_CANDIDATE_TRUNCATION_OR_REGENERATION", agent)


def freeze_joint_support(*, decision_group_id: str, no_assign_option: str,
                         records: Sequence[CandidateEvidence],
                         raw_candidate_ids_by_agent: Mapping[str, Iterable[str]],
                         source_state_digests: Mapping[str, str]) -> JointCandidateSupportSnapshot:
    """Validate and freeze the full pre-Zero-Loss support.

    There is no top-k selection here.  Every raw Local Search candidate must be
    evaluated exactly once, and Zero-Loss rejects are retained only as audit
    evidence.  This is the boundary that makes later candidate regeneration or
    a silent one-option truncation fail closed.
    """
    if no_assign_option != NO_ASSIGN:
        raise CandidateSupportError("MISSING_OR_INVALID_NO_ASSIGN")
    raw_items = _normalized_raw_ids(raw_candidate_ids_by_agent)
    raw = dict(raw_items)
    if not raw:
        raise CandidateSupportError("NO_AGENT_CANDIDATE_SUPPORT")
    _validate_evidence(records, raw)
    safe = tuple(sorted((row for row in records if row.zero_loss_status == PASS),
                        key=lambda row: (row.agent_id, row.candidate_id)))
    rejected = tuple(sorted((row for row in records if row.zero_loss_status == REJECT),
                            key=lambda row: (row.agent_id, row.candidate_id)))
    source_items = tuple(sorted((str(agent), str(digest))
                                for agent, digest in source_state_digests.items()))
    if set(agent for agent, _ in source_items) != set(raw):
        raise CandidateSupportError("SOURCE_STATE_DIGEST_AGENT_MISMATCH")
    basis = {
        "decision_group_id": str(decision_group_id),
        "no_assign_option": no_assign_option,
        "safe": [row.canonical_payload() for row in safe],
        "rejected": [row.canonical_payload() for row in rejected],
        "raw_candidate_ids_by_agent": {agent: list(ids) for agent, ids in raw_items},
        "source_state_digests": dict(source_items),
    }
    return JointCandidateSupportSnapshot(
        decision_group_id=str(decision_group_id), no_assign_option=no_assign_option,
        safe=safe, rejected=rejected, raw_candidate_ids_by_agent=raw_items,
        source_state_digests=source_items, snapshot_digest=canonical_sha256(basis),
    )
