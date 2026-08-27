"""Explicit R16 action-selection boundary for frozen Joint Assignment policy.

This module intentionally leaves the historical Joint Assignment head and T1
implementation byte-identical.  It exposes the two R16-authorized semantic
modes instead:

* ``FROZEN_INFERENCE_T1`` delegates to the existing exact-tie selector.
* ``FROZEN_MASKED_CATEGORICAL_TRAINING`` uses the *same* masked probability
  distribution only for the R15 generic deadlock predicate.  Its categorical
  draw is an identity-keyed exponential race, so candidate or agent list order
  and global RNG state cannot change a semantic outcome.

This is an action-selection utility, not an execution authorization.  It does
not create an optimizer, mutate a model, generate candidates, or advance a
simulator.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Sequence

import torch

import joint_assignment_frozen_tie_break as TIE
import multi_agent_candidate_assignment_head as H


FROZEN_INFERENCE_T1 = "FROZEN_INFERENCE_T1"
FROZEN_MASKED_CATEGORICAL_TRAINING = "FROZEN_MASKED_CATEGORICAL_TRAINING"
R15_E1_CONTRACT_ID = "LS3_BT8_R15_E1_FROZEN_POLICY_MASKED_CATEGORICAL_SAMPLING_V1"
R15_E1_REPAIR_LEVEL = "E1"


class SelectionModeError(RuntimeError):
    """Fail-closed selector input or mode violation."""


@dataclass(frozen=True)
class FrozenPolicySelection:
    """One action drawn from the original frozen masked policy distribution."""

    selection_mode: str
    categorical_triggered: bool
    trigger_reason: str
    selected: TIE.CanonicalAction
    deterministic_selection: TIE.TieBreakSelection
    probabilities: torch.Tensor
    log_probability: torch.Tensor
    canonical_rng_keyset_sha256: str | None
    pair_logits: torch.Tensor
    no_assign_logit: torch.Tensor
    safe_mask: torch.Tensor

    @property
    def selected_index(self) -> int:
        return int(self.selected.source_index)

    @property
    def selected_is_no_assign(self) -> bool:
        return bool(self.selected.is_no_assign)

    @property
    def selected_pair(self) -> tuple[str, str] | None:
        if self.selected_is_no_assign:
            return None
        return (str(self.selected.agent_id), str(self.selected.candidate_id))

    @property
    def semantic_identity(self) -> str:
        return TIE.NO_ASSIGN_IDENTITY if self.selected_is_no_assign else f"{self.selected.agent_id}::{self.selected.candidate_id}"


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SelectionModeError(code)


def _single_pair_row(value: torch.Tensor, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name}_NOT_TENSOR")
    if value.ndim == 1:
        return value.unsqueeze(0)
    _require(value.ndim == 2 and value.shape[0] == 1, f"{name}_SINGLE_DECISION_SHAPE_REQUIRED")
    return value


def _single_no_assign(value: torch.Tensor) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), "NO_ASSIGN_LOGIT_NOT_TENSOR")
    if value.ndim == 0:
        return value.reshape(1, 1)
    if value.ndim == 1 and value.numel() == 1:
        return value.reshape(1, 1)
    _require(value.ndim == 2 and tuple(value.shape) == (1, 1), "NO_ASSIGN_LOGIT_SINGLE_DECISION_SHAPE_REQUIRED")
    return value


def _single_probability_row(value: torch.Tensor) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), "MASKED_PROBABILITIES_NOT_TENSOR")
    if value.ndim == 1:
        return value.unsqueeze(0)
    _require(value.ndim == 2 and value.shape[0] == 1, "MASKED_PROBABILITIES_SINGLE_DECISION_SHAPE_REQUIRED")
    return value


def _candidate_rows(pair_keys: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, pair in enumerate(pair_keys):
        _require(isinstance(pair, tuple) and len(pair) == 2, f"PAIR_KEY_INVALID:{index}")
        agent_id, candidate_id = str(pair[0]), str(pair[1])
        _require(bool(agent_id) and bool(candidate_id), f"PAIR_KEY_IDENTITY_MISSING:{index}")
        rows.append({"agent_id": agent_id, "candidate_id": candidate_id})
    return rows


def _stable_uniform(*, snapshot_identity: str, action: TIE.CanonicalAction, probe_seed: int) -> float:
    """R15's exact semantic RNG lineage; it never consumes global RNG state."""
    payload = "|".join((
        R15_E1_CONTRACT_ID,
        str(snapshot_identity),
        str(action.kind),
        str(action.agent_id or "NO_ASSIGN"),
        str(action.candidate_id or TIE.NO_ASSIGN_IDENTITY),
        str(probe_seed),
        R15_E1_REPAIR_LEVEL,
    ))
    raw = hashlib.sha256(payload.encode("utf-8")).digest()
    mantissa = int.from_bytes(raw[:8], "big") >> 11
    return (mantissa + 0.5) / float(1 << 53)


def _canonical_rng_keyset_sha(*, snapshot_identity: str, actions: Sequence[TIE.CanonicalAction], probe_seed: int) -> str:
    rows = [{
        "contract_id": R15_E1_CONTRACT_ID,
        "repair_level": R15_E1_REPAIR_LEVEL,
        "snapshot_identity": str(snapshot_identity),
        "probe_seed": int(probe_seed),
        "kind": action.kind,
        "agent_id": action.agent_id,
        "candidate_id": action.candidate_id,
        "identity_digest": action.identity_digest,
    } for action in sorted(actions, key=lambda action: (action.kind, str(action.agent_id or ""), str(action.candidate_id or "")))]
    raw = repr(rows).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _categorical_exponential_race(*, actions: Sequence[TIE.CanonicalAction], probabilities: torch.Tensor,
                                  snapshot_identity: str, probe_seed: int) -> TIE.CanonicalAction:
    contenders: list[tuple[float, str, TIE.CanonicalAction]] = []
    for action in actions:
        probability = float(probabilities[int(action.source_index)].detach().cpu())
        _require(math.isfinite(probability) and probability >= 0.0, "NONFINITE_OR_NEGATIVE_POLICY_PROBABILITY")
        if probability <= 0.0:
            continue
        uniform = _stable_uniform(snapshot_identity=snapshot_identity, action=action, probe_seed=probe_seed)
        _require(0.0 < uniform < 1.0, "CANONICAL_RNG_OUT_OF_RANGE")
        contenders.append((-math.log(uniform) / probability, action.identity_digest, action))
    _require(bool(contenders), "EMPTY_POSITIVE_POLICY_SUPPORT")
    return min(contenders, key=lambda row: (row[0], row[1]))[2]


def _bound_training_rng(*, snapshot_identity: str | None, probe_seed: int | None) -> tuple[str, int]:
    _require(isinstance(snapshot_identity, str) and bool(snapshot_identity), "TRAINING_SNAPSHOT_IDENTITY_REQUIRED")
    _require(isinstance(probe_seed, int) and not isinstance(probe_seed, bool), "TRAINING_PROBE_SEED_REQUIRED")
    return snapshot_identity, int(probe_seed)


def select_frozen_policy_action(*, pair_keys: Sequence[tuple[str, str]], pair_logits: torch.Tensor,
                                no_assign_logit: torch.Tensor, safe_mask: torch.Tensor,
                                mode: str = FROZEN_INFERENCE_T1,
                                snapshot_identity: str | None = None,
                                probe_seed: int | None = None,
                                masked_probabilities: torch.Tensor | None = None) -> FrozenPolicySelection:
    """Select exactly one semantic action from the pre-existing frozen policy.

    The training mode is deliberately not an unconditional sampler.  It is the
    R15 E1 deadlock branch: categorical sampling happens only when the same
    T1 deterministic selection is ``NO_ASSIGN`` *and* a legal candidate exists.
    No action receives added mass; ``NO_ASSIGN`` remains in the full sampling
    distribution and can still be selected.  A caller that has already obtained
    the frozen masked distribution from the actor-forward selection path may
    pass it through ``masked_probabilities``.  That view is validated and used
    unchanged; it is never re-normalized or reweighted.  This avoids a second
    device kernel becoming a needless source of numerical variation.
    """
    _require(mode in {FROZEN_INFERENCE_T1, FROZEN_MASKED_CATEGORICAL_TRAINING}, "UNKNOWN_SELECTION_MODE")
    pairs = _single_pair_row(pair_logits, "PAIR_LOGITS")
    mask = _single_pair_row(safe_mask, "SAFE_MASK")
    no_assign = _single_no_assign(no_assign_logit)
    _require(pairs.shape == mask.shape, "PAIR_LOGIT_MASK_SHAPE_MISMATCH")
    _require(mask.dtype == torch.bool, "SAFE_MASK_DTYPE_INVALID")
    _require(pairs.dtype.is_floating_point and no_assign.dtype.is_floating_point, "LOGIT_DTYPE_INVALID")
    _require(pairs.device == mask.device == no_assign.device and pairs.dtype == no_assign.dtype, "LOGIT_DEVICE_OR_DTYPE_MISMATCH")
    _require(len(pair_keys) == int(pairs.shape[1]), "PAIR_KEY_LOGIT_SHAPE_MISMATCH")

    candidate_ids = _candidate_rows(pair_keys)
    safe_logits = pairs[mask]
    _require(bool(torch.isfinite(safe_logits).all().item()) and bool(torch.isfinite(no_assign).all().item()),
             "NONFINITE_LEGAL_LOGIT")
    if masked_probabilities is None:
        probabilities = H.masked_distribution(pairs, no_assign, mask)[0]
    else:
        provided = _single_probability_row(masked_probabilities)
        _require(provided.shape == (1, pairs.shape[1] + 1), "MASKED_PROBABILITIES_SHAPE_MISMATCH")
        _require(provided.device == pairs.device and provided.dtype == pairs.dtype, "MASKED_PROBABILITIES_DEVICE_OR_DTYPE_MISMATCH")
        probabilities = provided[0]
    _require(bool(torch.isfinite(probabilities).all().item()), "NONFINITE_MASKED_PROBABILITY")
    _require(bool((probabilities >= 0).all().item()), "NEGATIVE_MASKED_PROBABILITY")
    _require(bool((probabilities[:-1][~mask[0]] == 0).all().item()), "UNSAFE_ACTION_HAS_POLICY_MASS")

    try:
        actions = TIE.canonical_actions(
            candidate_ids=candidate_ids,
            pair_scores=pairs[0].detach().cpu().tolist(),
            safe_mask=mask[0].detach().cpu().tolist(),
            no_assign_score=float(no_assign[0, 0].detach().cpu()),
        )
        deterministic = TIE.select_exact_tie(
            candidate_ids=candidate_ids,
            pair_scores=pairs[0].detach().cpu().tolist(),
            safe_mask=mask[0].detach().cpu().tolist(),
            no_assign_score=float(no_assign[0, 0].detach().cpu()),
        )
    except TIE.FrozenTieBreakError as exc:
        raise SelectionModeError(f"T1_CANONICALIZATION_FAILED:{exc.code}") from exc

    selected = deterministic.selected
    triggered = False
    trigger_reason = "INFERENCE_T1"
    keyset_sha: str | None = None
    if mode == FROZEN_MASKED_CATEGORICAL_TRAINING:
        snapshot_value, seed_value = _bound_training_rng(snapshot_identity=snapshot_identity, probe_seed=probe_seed)
        legal_candidate_count = sum(not action.is_no_assign for action in actions)
        if deterministic.selected.is_no_assign and legal_candidate_count > 0:
            selected = _categorical_exponential_race(actions=actions, probabilities=probabilities,
                                                     snapshot_identity=snapshot_value, probe_seed=seed_value)
            triggered = True
            trigger_reason = "R15_E1_DEADLOCK_TRIGGER"
        else:
            trigger_reason = "R15_E1_NO_DEADLOCK_TRIGGER"
        keyset_sha = _canonical_rng_keyset_sha(snapshot_identity=snapshot_value, actions=actions, probe_seed=seed_value)

    selected_probability = probabilities[int(selected.source_index)]
    _require(bool(torch.isfinite(selected_probability).item()) and float(selected_probability.detach().cpu()) > 0.0,
             "SELECTED_ACTION_HAS_NONPOSITIVE_PROBABILITY")
    # This is deliberately the log of the same frozen softmax result exposed to
    # the caller.  It cannot describe a separately normalized rescue policy.
    log_probability = torch.log(selected_probability)
    return FrozenPolicySelection(
        selection_mode=mode,
        categorical_triggered=triggered,
        trigger_reason=trigger_reason,
        selected=selected,
        deterministic_selection=deterministic,
        probabilities=probabilities,
        log_probability=log_probability,
        canonical_rng_keyset_sha256=keyset_sha,
        pair_logits=pairs[0],
        no_assign_logit=no_assign[0],
        safe_mask=mask[0],
    )
