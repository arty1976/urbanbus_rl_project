"""Focused R16 guards for frozen-policy categorical action selection.

These tests exercise only the pure action-selection boundary.  They do not
load checkpoints, create candidates, run a rollout, or create an optimizer.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joint_assignment_frozen_policy_selector as S  # noqa: E402
import joint_assignment_frozen_tie_break as TIE  # noqa: E402
import multi_agent_candidate_assignment_head as H  # noqa: E402


def _inputs(*, pair_keys: list[tuple[str, str]] | None = None,
            pair_logits: list[float] | None = None,
            no_assign_logit: float = 2.0,
            safe_mask: list[bool] | None = None) -> dict[str, object]:
    keys = pair_keys or [
        ("agent-02", "candidate-b"),
        ("agent-01", "candidate-a"),
        ("agent-03", "candidate-c"),
    ]
    logits = pair_logits or [-1.0, -1.5, -2.0]
    mask = safe_mask or [True] * len(keys)
    return {
        "pair_keys": keys,
        "pair_logits": torch.tensor([logits], dtype=torch.float32),
        "no_assign_logit": torch.tensor([[no_assign_logit]], dtype=torch.float32),
        "safe_mask": torch.tensor([mask], dtype=torch.bool),
    }


def _select(values: dict[str, object], *, mode: str = S.FROZEN_INFERENCE_T1,
            snapshot_identity: str | None = None, probe_seed: int | None = None):
    return S.select_frozen_policy_action(
        pair_keys=values["pair_keys"],
        pair_logits=values["pair_logits"],
        no_assign_logit=values["no_assign_logit"],
        safe_mask=values["safe_mask"],
        mode=mode,
        snapshot_identity=snapshot_identity,
        probe_seed=probe_seed,
    )


def _identity(result) -> str:
    return "NO_ASSIGN_KEEP_CURRENT_PLANS" if result.selected_is_no_assign else f"{result.selected_pair[0]}::{result.selected_pair[1]}"


def _reference_t1(values: dict[str, object]):
    return TIE.select_exact_tie(
        candidate_ids=[{"agent_id": agent_id, "candidate_id": candidate_id}
                       for agent_id, candidate_id in values["pair_keys"]],
        pair_scores=values["pair_logits"][0].tolist(),
        safe_mask=values["safe_mask"][0].tolist(),
        no_assign_score=float(values["no_assign_logit"][0, 0]),
    )


def _distribution(values: dict[str, object]) -> torch.Tensor:
    return H.masked_distribution(
        values["pair_logits"], values["no_assign_logit"], values["safe_mask"],
    )[0]


def test_inference_mode_defaults_to_exact_frozen_t1_and_preserves_distribution() -> None:
    # A candidate exact tie must use the historical semantic T1 winner rather
    # than a positional argmax winner.
    values = _inputs(pair_logits=[1.0, 1.0, -4.0], no_assign_logit=-3.0)
    before_logits = values["pair_logits"].clone()
    before_no_assign = values["no_assign_logit"].clone()
    before_mask = values["safe_mask"].clone()
    reference = _reference_t1(values)

    default = _select(values)
    explicit = _select(values, mode=S.FROZEN_INFERENCE_T1)
    source_probability = _distribution(values)

    assert default.selection_mode == S.FROZEN_INFERENCE_T1
    assert explicit.selection_mode == S.FROZEN_INFERENCE_T1
    assert default.categorical_triggered is False
    assert explicit.categorical_triggered is False
    assert default.selected_index == explicit.selected_index == reference.selected.source_index
    assert _identity(default) == _identity(explicit)
    assert default.selected_pair == explicit.selected_pair == (reference.selected.agent_id, reference.selected.candidate_id)
    assert torch.equal(default.probabilities, source_probability)
    assert torch.equal(explicit.probabilities, source_probability)
    assert math.isclose(float(default.log_probability), float(torch.log(source_probability[default.selected_index])), abs_tol=0.0)
    assert torch.equal(values["pair_logits"], before_logits)
    assert torch.equal(values["no_assign_logit"], before_no_assign)
    assert torch.equal(values["safe_mask"], before_mask)


def test_deadlocked_training_mode_is_exact_categorical_with_source_log_probability() -> None:
    # T1 picks NO_ASSIGN, but every candidate remains legal and has its original
    # nonzero policy mass.  R15's E1 trigger therefore applies in training mode.
    values = _inputs(pair_logits=[-1.0, -1.5, -2.0], no_assign_logit=2.0)
    reference = _reference_t1(values)
    source_probability = _distribution(values)
    assert reference.selected.is_no_assign is True
    assert float(source_probability[-1]) > 0.0
    assert all(float(value) > 0.0 for value in source_probability[:-1])

    observed = []
    for seed in range(32):
        result = _select(values, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                         snapshot_identity="snapshot-deadlock", probe_seed=seed)
        observed.append(_identity(result))
        assert result.selection_mode == S.FROZEN_MASKED_CATEGORICAL_TRAINING
        assert result.categorical_triggered is True
        assert torch.equal(result.probabilities, source_probability)
        assert result.selected_index in range(len(values["pair_keys"]) + 1)
        assert math.isclose(float(result.log_probability),
                            float(torch.log(source_probability[result.selected_index])), abs_tol=0.0)
    assert any(value != "NO_ASSIGN_KEEP_CURRENT_PLANS" for value in observed)


def test_training_request_without_deadlock_remains_deterministic_t1() -> None:
    values = _inputs(pair_logits=[4.0, -1.0, -2.0], no_assign_logit=0.0)
    reference = _reference_t1(values)
    source_probability = _distribution(values)
    assert reference.selected.is_no_assign is False

    for seed in range(8):
        result = _select(values, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                         snapshot_identity="snapshot-ac-control", probe_seed=seed)
        assert result.categorical_triggered is False
        assert result.selected_index == reference.selected.source_index
        assert _identity(result) == f"{reference.selected.agent_id}::{reference.selected.candidate_id}"
        assert torch.equal(result.probabilities, source_probability)
        assert math.isclose(float(result.log_probability),
                            float(torch.log(source_probability[result.selected_index])), abs_tol=0.0)


def test_identity_keyed_deadlock_sampling_is_permutation_invariant_and_does_not_touch_global_rng() -> None:
    values = _inputs(pair_logits=[-1.0, -1.5, -2.0], no_assign_logit=2.0)
    # Different input orders represent the same semantic action support.
    candidate_order = [2, 0, 1]
    agent_order = [1, 2, 0]
    combined_order = [1, 0, 2]

    def permuted(order: list[int]) -> dict[str, object]:
        return _inputs(
            pair_keys=[values["pair_keys"][index] for index in order],
            pair_logits=[float(values["pair_logits"][0, index]) for index in order],
            no_assign_logit=float(values["no_assign_logit"][0, 0]),
            safe_mask=[bool(values["safe_mask"][0, index]) for index in order],
        )

    candidate = permuted(candidate_order)
    agent = permuted(agent_order)
    combined = permuted(combined_order)

    torch.manual_seed(20260827)
    before_rng = torch.random.get_rng_state().clone()
    base = [_identity(_select(values, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                              snapshot_identity="snapshot-permutation", probe_seed=seed)) for seed in range(32)]
    after_rng = torch.random.get_rng_state().clone()
    assert torch.equal(before_rng, after_rng)

    for alternate in (candidate, agent, combined):
        selected = [_identity(_select(alternate, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                                      snapshot_identity="snapshot-permutation", probe_seed=seed)) for seed in range(32)]
        assert selected == base


def test_same_seed_replays_exactly_and_different_seeds_can_diversify() -> None:
    values = _inputs(pair_logits=[-1.0, -1.5, -2.0], no_assign_logit=2.0)
    first = _select(values, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                    snapshot_identity="snapshot-rng", probe_seed=11)
    replay = _select(values, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                     snapshot_identity="snapshot-rng", probe_seed=11)
    assert _identity(first) == _identity(replay)
    assert first.selected_index == replay.selected_index
    assert float(first.log_probability) == float(replay.log_probability)

    identities = {
        _identity(_select(values, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                          snapshot_identity="snapshot-rng", probe_seed=seed))
        for seed in range(32)
    }
    assert len(identities) > 1


def test_no_assign_only_and_masked_candidates_preserve_legality_without_fabricated_exploration() -> None:
    values = _inputs(pair_logits=[100.0, 99.0, 98.0], no_assign_logit=-3.0,
                     safe_mask=[False, False, False])
    source_probability = _distribution(values)
    assert torch.equal(source_probability, torch.tensor([0.0, 0.0, 0.0, 1.0]))

    for seed in range(8):
        result = _select(values, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                         snapshot_identity="snapshot-noassign-only", probe_seed=seed)
        assert result.categorical_triggered is False
        assert result.selected_is_no_assign is True
        assert result.selected_index == len(values["pair_keys"])
        assert _identity(result) == "NO_ASSIGN_KEEP_CURRENT_PLANS"
        assert torch.equal(result.probabilities, source_probability)
        assert float(result.log_probability) == 0.0

    # A huge unsafe logit must never receive probability or selection mass.
    mixed = _inputs(pair_logits=[100.0, -1.0, -2.0], no_assign_logit=2.0,
                    safe_mask=[False, True, True])
    mixed_probability = _distribution(mixed)
    assert float(mixed_probability[0]) == 0.0
    for seed in range(32):
        result = _select(mixed, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                         snapshot_identity="snapshot-masked", probe_seed=seed)
        assert result.selected_index != 0
        assert math.isfinite(float(result.log_probability))


def test_unknown_or_underbound_mode_fails_closed() -> None:
    values = _inputs(pair_logits=[-1.0, -1.5, -2.0], no_assign_logit=2.0)
    with pytest.raises(S.SelectionModeError):
        _select(values, mode="UNAUTHORIZED_MODE", snapshot_identity="snapshot", probe_seed=1)
    with pytest.raises(S.SelectionModeError):
        _select(values, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                snapshot_identity=None, probe_seed=1)
    with pytest.raises(S.SelectionModeError):
        _select(values, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                snapshot_identity="snapshot", probe_seed=None)
    duplicate = _inputs(pair_keys=[("agent-01", "candidate-a"), ("agent-01", "candidate-a")],
                        pair_logits=[-1.0, -2.0], no_assign_logit=2.0)
    with pytest.raises(S.SelectionModeError):
        _select(duplicate, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                snapshot_identity="snapshot", probe_seed=1)
