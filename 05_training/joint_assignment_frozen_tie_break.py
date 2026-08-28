"""Order-invariant exact-tie selection for frozen Joint Assignment inference.

The learned actor scores are never changed here.  This utility only chooses a
canonical physical action when two or more *legal* actions have exactly equal
best scores.  It is intentionally not used by training-time rollout code.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


TIE_BREAK_CONTRACT_ID = "LS3_BT7_R1_EXACT_TIE_CANONICAL_ACTION_IDENTITY_V1"
NO_ASSIGN_IDENTITY = "NO_ASSIGN_KEEP_CURRENT_PLANS"


class FrozenTieBreakError(RuntimeError):
    """Selection evidence malformed enough that replay must fail closed."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class CanonicalAction:
    kind: str
    agent_id: str | None
    candidate_id: str | None
    source_index: int
    score: float
    identity_digest: str

    @property
    def is_no_assign(self) -> bool:
        return self.kind == "NO_ASSIGN"


@dataclass(frozen=True)
class TieBreakSelection:
    selected: CanonicalAction
    best_score: float
    tie_set: tuple[CanonicalAction, ...]
    exact_tie: bool


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise FrozenTieBreakError(code, detail)


def canonical_identity_digest(*, agent_id: str | None, candidate_id: str | None, no_assign: bool = False) -> str:
    """Stable opaque identity, never an array position, rank, or agent ordinal."""
    if no_assign:
        payload = f"{TIE_BREAK_CONTRACT_ID}|NO_ASSIGN|{NO_ASSIGN_IDENTITY}"
    else:
        _require(bool(agent_id) and bool(candidate_id), "MISSING_CANONICAL_CANDIDATE_IDENTITY")
        payload = f"{TIE_BREAK_CONTRACT_ID}|ASSIGN|agent={agent_id}|candidate={candidate_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_actions(*, candidate_ids: Sequence[Mapping[str, Any]], pair_scores: Sequence[float],
                      safe_mask: Sequence[bool], no_assign_score: float,
                      expected_identity_digests: Sequence[str] | None = None) -> tuple[CanonicalAction, ...]:
    """Build legal actions, excluding masked candidates before comparing scores."""
    _require(len(candidate_ids) == len(pair_scores) == len(safe_mask), "TIE_BREAK_ACTION_SHAPE_MISMATCH")
    _require(math.isfinite(float(no_assign_score)), "TIE_BREAK_NONFINITE_NO_ASSIGN_SCORE")
    seen = set()
    actions = []
    for index, (row, score, safe) in enumerate(zip(candidate_ids, pair_scores, safe_mask)):
        _require(isinstance(row, Mapping), "MISSING_CANONICAL_CANDIDATE_IDENTITY", str(index))
        agent_id, candidate_id = row.get("agent_id"), row.get("candidate_id")
        digest = canonical_identity_digest(agent_id=agent_id, candidate_id=candidate_id)
        if expected_identity_digests is not None:
            _require(index < len(expected_identity_digests) and expected_identity_digests[index] == digest,
                     "CANONICAL_IDENTITY_DIGEST_TAMPERED", str(index))
        key = (str(agent_id), str(candidate_id))
        _require(key not in seen, "DUPLICATE_CANONICAL_CANDIDATE_IDENTITY", repr(key))
        seen.add(key)
        if bool(safe):
            _require(math.isfinite(float(score)), "TIE_BREAK_NONFINITE_LEGAL_SCORE", str(index))
            actions.append(CanonicalAction("ASSIGN", str(agent_id), str(candidate_id), index, float(score), digest))
    no_assign = CanonicalAction("NO_ASSIGN", None, None, len(candidate_ids), float(no_assign_score),
                                canonical_identity_digest(agent_id=None, candidate_id=None, no_assign=True))
    actions.append(no_assign)
    return tuple(actions)


def select_exact_tie(*, candidate_ids: Sequence[Mapping[str, Any]], pair_scores: Sequence[float],
                     safe_mask: Sequence[bool], no_assign_score: float,
                     expected_identity_digests: Sequence[str] | None = None) -> TieBreakSelection:
    """T1: exact equality only; non-tie scores always retain their unique winner."""
    actions = canonical_actions(candidate_ids=candidate_ids, pair_scores=pair_scores, safe_mask=safe_mask,
                                no_assign_score=no_assign_score, expected_identity_digests=expected_identity_digests)
    best = max(row.score for row in actions)
    # No tolerance: T1 intentionally refuses to treat a positive model-score
    # margin as a tie, regardless of how small it may appear.
    tie_set = tuple(row for row in actions if row.score == best)
    selected = min(tie_set, key=lambda row: row.identity_digest)
    return TieBreakSelection(selected=selected, best_score=best, tie_set=tie_set, exact_tie=len(tie_set) > 1)
