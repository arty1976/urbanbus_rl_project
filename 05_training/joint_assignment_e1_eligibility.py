#!/usr/bin/env python3
"""BT8-R8 E1 reward-ancestry eligibility boundary for joint Actor loss.

This module binds the immutable BT8-R7 selection contract and translates its
persisted, identity-bound credit rows into a per-row Actor eligibility mask.
It does not calculate rewards, GAE, normalization, PPO ratios, logits, or any
optimizer update.  Critic eligibility is deliberately all-or-nothing over the
finite rows: a non-finite row is an error, never a silent filter.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


E1_CONTRACT_ID = "LS3_BT8_R7_E1_REWARD_ANCESTRY_ACTOR_ELIGIBILITY_V1"
E1_CONTRACT_SHA256 = "eb84543a9fc06dcf730e49aa3895d9fe26d2244a05ce340986b7449418205ad9"
NO_ASSIGN = "NO_ASSIGN_KEEP_CURRENT_PLANS"

ACTOR_ELIGIBLE_CATEGORIES = frozenset({
    "DIRECT_CANDIDATE_REWARD",
    "TEMPORALLY_PROPAGATED_CANDIDATE_REWARD",
    "NO_ASSIGN_REWARD_SUPPORTED",
})
ACTOR_INELIGIBLE_CATEGORIES = frozenset({
    "CRITIC_ONLY_NO_REWARD_ANCESTRY",
    "INSUFFICIENT_IDENTITY_CREDIT",
})
KNOWN_CATEGORIES = ACTOR_ELIGIBLE_CATEGORIES | ACTOR_INELIGIBLE_CATEGORIES


class E1EligibilityError(RuntimeError):
    """Fail-closed error for an E1 authority or row-integrity violation."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}:{detail}" if detail else code)
        self.code = code


def _require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise E1EligibilityError(code, detail)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def bind_e1_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the R7 E1 semantic contract and its canonical digest exactly."""
    payload = dict(contract)
    declared = payload.pop("sha256", None)
    _require(declared == E1_CONTRACT_SHA256, "E1_CONTRACT_SHA256_MISMATCH")
    _require(canonical_sha256(payload) == declared, "E1_CONTRACT_CANONICAL_DIGEST_MISMATCH")
    _require(payload.get("contract_id") == E1_CONTRACT_ID, "E1_CONTRACT_ID_MISMATCH")
    _require(payload.get("selected_option") == "E1", "E1_OPTION_NOT_SELECTED")
    _require(set(payload.get("actor_eligible_categories", [])) == ACTOR_ELIGIBLE_CATEGORIES,
             "E1_ELIGIBLE_CATEGORY_CONTRACT_MISMATCH")
    _require(set(payload.get("actor_excluded_categories", [])) == ACTOR_INELIGIBLE_CATEGORIES,
             "E1_INELIGIBLE_CATEGORY_CONTRACT_MISMATCH")
    _require(payload.get("critic_rows") == "all finite persisted rows", "E1_CRITIC_SCOPE_MISMATCH")
    _require(payload.get("normalization") == "N0 retained; no normalizer change selected",
             "E1_NORMALIZATION_CONTRACT_MISMATCH")
    invariants = payload.get("invariants", {})
    _require(invariants.get("reward_v2") == "unchanged" and invariants.get("legal_support") == "frozen"
             and invariants.get("old_log_prob") == "unchanged for retained rows"
             and invariants.get("ppo_ratio_and_clipping") == "unchanged for retained rows"
             and invariants.get("no_assign_penalty") == "none"
             and invariants.get("actor_critic_architecture") == "unchanged"
             and invariants.get("candidate_regeneration") is False,
             "E1_IMMUTABLE_INVARIANT_MISMATCH")
    return {**payload, "sha256": declared}


def category_from_row(row: Mapping[str, Any]) -> str:
    """Recompute the only categories recognized by the R7 E1 contract."""
    if not bool(row.get("identity_chain_valid")):
        return "INSUFFICIENT_IDENTITY_CREDIT"
    action = str(row.get("selected_action_type"))
    reward = float(row.get("reward"))
    reward_component = float(row.get("reward_gae_component"))
    raw_gae = float(row.get("raw_gae"))
    if action == "CANDIDATE" and reward != 0.0:
        return "DIRECT_CANDIDATE_REWARD"
    if action == "CANDIDATE" and reward == 0.0 and reward_component != 0.0:
        return "TEMPORALLY_PROPAGATED_CANDIDATE_REWARD"
    if action == "NO_ASSIGN" and reward_component != 0.0:
        return "NO_ASSIGN_REWARD_SUPPORTED"
    if reward_component == 0.0 and raw_gae != 0.0:
        return "CRITIC_ONLY_NO_REWARD_ANCESTRY"
    return "INSUFFICIENT_IDENTITY_CREDIT"


def _validate_identity(row: Mapping[str, Any]) -> None:
    decision_id = str(row.get("decision_id", ""))
    _require(bool(decision_id), "E1_DECISION_ID_MISSING")
    selected = str(row.get("selected_candidate_id", ""))
    applied = str(row.get("applied_candidate_id", ""))
    credited = str(row.get("credited_candidate_id", ""))
    _require(bool(row.get("identity_chain_valid")), "E1_INSUFFICIENT_IDENTITY_CREDIT", decision_id)
    _require(selected == applied == credited, "E1_CREDIT_IDENTITY_CHAIN_MISMATCH", decision_id)
    if str(row.get("selected_action_type")) == "NO_ASSIGN":
        _require(selected == NO_ASSIGN, "E1_NO_ASSIGN_IDENTITY_MISMATCH", decision_id)
    else:
        _require(bool(selected) and selected != NO_ASSIGN, "E1_CANDIDATE_IDENTITY_MISSING", decision_id)


@dataclass(frozen=True)
class E1EligibilityMask:
    replicate_id: str
    decision_ids: tuple[str, ...]
    actor_eligible: tuple[bool, ...]
    critic_eligible: tuple[bool, ...]
    category_counts: Mapping[str, int]
    evidence_digest: str

    @property
    def actor_eligible_count(self) -> int:
        return sum(self.actor_eligible)

    @property
    def actor_ineligible_count(self) -> int:
        return len(self.actor_eligible) - self.actor_eligible_count


def build_e1_eligibility_mask(rows: Sequence[Mapping[str, Any]], *, expected_replicate_id: str) -> E1EligibilityMask:
    """Build one seed-local E1 mask; mixed seeds/review rows fail closed."""
    _require(bool(rows), "E1_EMPTY_ROW_SET")
    decision_ids: list[str] = []
    actor_mask: list[bool] = []
    critic_mask: list[bool] = []
    category_counts = {category: 0 for category in sorted(KNOWN_CATEGORIES)}
    canonical_rows: list[dict[str, Any]] = []
    for row in rows:
        decision_id = str(row.get("decision_id", ""))
        replicate_id = str(row.get("replicate_id", ""))
        _require(replicate_id == expected_replicate_id, "E1_SEED_MIXING_OR_REPLICATE_MISMATCH", decision_id)
        _require("REVIEW" not in decision_id.upper() and str(row.get("data_role", "TRAIN")) == "TRAIN",
                 "E1_REVIEW_ROW_IN_ACTOR_LOSS", decision_id)
        _require(decision_id not in decision_ids, "E1_DUPLICATE_DECISION_ID", decision_id)
        _validate_identity(row)
        numeric = ("reward", "reward_gae_component", "raw_gae")
        _require(all(math.isfinite(float(row.get(name))) for name in numeric),
                 "E1_NONFINITE_CREDIT_ROW", decision_id)
        category = str(row.get("category", ""))
        _require(category in KNOWN_CATEGORIES, "E1_UNKNOWN_CREDIT_CATEGORY", decision_id)
        derived = category_from_row(row)
        _require(derived == category, "E1_ANCESTRY_LABEL_TAMPER", decision_id)
        _require(category != "INSUFFICIENT_IDENTITY_CREDIT", "E1_INSUFFICIENT_IDENTITY_CREDIT", decision_id)
        actor_eligible = category in ACTOR_ELIGIBLE_CATEGORIES
        _require(bool(row.get("actor_eligible_e1")) == actor_eligible,
                 "E1_ACTOR_ELIGIBILITY_FLAG_MISMATCH", decision_id)
        _require(bool(row.get("critic_eligible_e1")), "E1_CRITIC_ELIGIBILITY_MISMATCH", decision_id)
        decision_ids.append(decision_id)
        actor_mask.append(actor_eligible)
        critic_mask.append(True)
        category_counts[category] += 1
        canonical_rows.append({
            "decision_id": decision_id,
            "replicate_id": replicate_id,
            "selected_action_type": str(row.get("selected_action_type")),
            "selected_candidate_id": str(row.get("selected_candidate_id")),
            "applied_candidate_id": str(row.get("applied_candidate_id")),
            "credited_candidate_id": str(row.get("credited_candidate_id")),
            "category": category,
            "reward": float(row.get("reward")),
            "reward_gae_component": float(row.get("reward_gae_component")),
            "raw_gae": float(row.get("raw_gae")),
        })
    return E1EligibilityMask(
        replicate_id=expected_replicate_id,
        decision_ids=tuple(decision_ids),
        actor_eligible=tuple(actor_mask),
        critic_eligible=tuple(critic_mask),
        category_counts={key: value for key, value in category_counts.items() if value},
        evidence_digest=canonical_sha256(canonical_rows),
    )
