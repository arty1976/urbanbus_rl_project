"""Focused BT8-R3 architecture fixtures. No simulator, optimizer, or training."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import joint_assignment_frozen_policy_snapshot as FPS
import multi_agent_candidate_assignment_head as H


ROOT = Path(__file__).resolve().parent
A1_CHECKPOINT = (ROOT / "artifacts" /
    "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_a1_bounded_training_20260822_225122+09:00" /
    "bt8_a1_joint_assignment_checkpoint.pt")


def actor() -> H.CandidateSensitiveMultiAgentCandidateAssignmentHead:
    torch.manual_seed(20260823)
    return H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
        global_dim=8, demand_dim=6, agent_dim=4, candidate_dim=8,
        hidden=32, heads=4).eval()


def inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(91)
    candidate = torch.randn(1, 4, 8)
    candidate[0, 0, 3] = 2.0
    candidate[0, 1] = candidate[0, 0]
    candidate[0, 1, 3] = 3.0  # same-agent, observed-style hop_count delta
    return {
        "global_feats": torch.randn(1, 8),
        "demand_feats": torch.randn(1, 6),
        "agent_feats": torch.randn(1, 3, 4),
        "agent_mask": torch.tensor([[True, True, True]]),
        "candidate_feats": candidate,
        "pair_agent_index": torch.tensor([[0, 0, 1, 2]], dtype=torch.long),
        "safe_mask": torch.tensor([[True, True, True, False]]),
    }


def cloned(payload: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in payload.items()}


def forward(model: torch.nn.Module, payload: dict[str, torch.Tensor]):
    return model(**payload)


def test_v1_root_cause_is_preserved_and_v2_repairs_only_scorer_dataflow() -> None:
    old = H.MultiAgentCandidateAssignmentHead(global_dim=8, demand_dim=6,
        agent_dim=4, candidate_dim=8, hidden=32, heads=4)
    repaired = actor()
    assert old.scorer[0].in_features == 32 * 4
    assert repaired.scorer[0].in_features == 32 * 5
    assert old.no_assign_scorer[0].in_features == repaired.no_assign_scorer[0].in_features == 32 * 2
    assert H.CANDIDATE_SENSITIVE_HEAD_CONTRACT["candidate_context_reaches_pair_scorer"]
    assert not H.CANDIDATE_SENSITIVE_HEAD_CONTRACT["candidate_list_position_used"]


def test_candidate_sensitivity_row_swap_observed_replacement_zeroing_and_gradient() -> None:
    model, payload = actor(), inputs()
    base, base_no_assign = forward(model, payload)

    swapped = cloned(payload)
    swapped["candidate_feats"][:, [0, 1]] = payload["candidate_feats"][:, [1, 0]]
    swapped_logits, swapped_no_assign = forward(model, swapped)
    torch.testing.assert_close(swapped_logits[:, :2], base[:, [1, 0]], rtol=0, atol=1e-7)
    torch.testing.assert_close(swapped_logits[:, 2:], base[:, 2:], rtol=0, atol=1e-7)
    torch.testing.assert_close(swapped_no_assign, base_no_assign, rtol=0, atol=0)
    assert not torch.equal(swapped_logits[:, :2], base[:, :2])

    replaced = cloned(payload)
    replaced["candidate_feats"][:, 0] = payload["candidate_feats"][:, 1]
    replaced_logits, replaced_no_assign = forward(model, replaced)
    torch.testing.assert_close(replaced_logits[:, 0], base[:, 1], rtol=0, atol=1e-7)
    assert float((replaced_logits[:, 0] - base[:, 0]).abs().max().detach()) > 0
    torch.testing.assert_close(replaced_no_assign, base_no_assign, rtol=0, atol=0)

    zeroed = cloned(payload)
    zeroed["candidate_feats"] = torch.zeros_like(payload["candidate_feats"])
    zeroed_logits, zeroed_no_assign = forward(model, zeroed)
    assert float((zeroed_logits[:, :3] - base[:, :3]).abs().max().detach()) > 0
    torch.testing.assert_close(zeroed_no_assign, base_no_assign, rtol=0, atol=0)

    differentiable = cloned(payload)
    differentiable["candidate_feats"] = differentiable["candidate_feats"].requires_grad_(True)
    logits, _ = forward(model, differentiable)
    gradient = torch.autograd.grad(logits[0, 0], differentiable["candidate_feats"])[0]
    assert float(gradient[0, 0].abs().sum().detach()) > 0
    assert float(gradient[0, 0, 3].abs().detach()) > 0


def test_candidate_agent_and_combined_permutation_contract() -> None:
    model, payload = actor(), inputs()
    base, base_no_assign = forward(model, payload)

    candidate_perm = torch.tensor([2, 0, 3, 1])
    candidate = cloned(payload)
    for name in ("candidate_feats", "pair_agent_index", "safe_mask"):
        candidate[name] = candidate[name][:, candidate_perm]
    candidate_logits, candidate_no_assign = forward(model, candidate)
    torch.testing.assert_close(candidate_logits, base[:, candidate_perm], rtol=0, atol=1e-7)
    torch.testing.assert_close(candidate_no_assign, base_no_assign, rtol=0, atol=0)

    agent_perm = torch.tensor([2, 0, 1])
    inverse = torch.empty_like(agent_perm)
    inverse[agent_perm] = torch.arange(len(agent_perm))
    agent = cloned(payload)
    agent["agent_feats"] = agent["agent_feats"][:, agent_perm]
    agent["agent_mask"] = agent["agent_mask"][:, agent_perm]
    agent["pair_agent_index"] = inverse[agent["pair_agent_index"]]
    agent_logits, agent_no_assign = forward(model, agent)
    torch.testing.assert_close(agent_logits, base, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(agent_no_assign, base_no_assign, rtol=1e-5, atol=1e-6)

    combined = cloned(agent)
    for name in ("candidate_feats", "pair_agent_index", "safe_mask"):
        combined[name] = combined[name][:, candidate_perm]
    combined_logits, combined_no_assign = forward(model, combined)
    torch.testing.assert_close(combined_logits, base[:, candidate_perm], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(combined_no_assign, base_no_assign, rtol=1e-5, atol=1e-6)


def test_no_assign_only_and_mixed_support_preserve_hard_mask_semantics() -> None:
    model, payload = actor(), inputs()
    forced = cloned(payload)
    forced["safe_mask"][:] = False
    forced_logits, forced_no_assign = forward(model, forced)
    forced_probs = H.masked_distribution(forced_logits, forced_no_assign, forced["safe_mask"])
    assert bool(torch.isneginf(forced_logits).all())
    assert torch.equal(forced_probs[:, :-1], torch.zeros_like(forced_probs[:, :-1]))
    assert torch.equal(forced_probs[:, -1], torch.ones_like(forced_probs[:, -1]))

    mixed_logits, mixed_no_assign = forward(model, payload)
    mixed_probs = H.masked_distribution(mixed_logits, mixed_no_assign, payload["safe_mask"])
    assert bool(torch.isneginf(mixed_logits[~payload["safe_mask"]]).all())
    assert torch.equal(mixed_probs[:, 3], torch.zeros_like(mixed_probs[:, 3]))
    assert bool(torch.isfinite(mixed_no_assign).all())
    assert float(mixed_probs[:, -1].min().detach()) > 0


@pytest.mark.parametrize(("mutation", "code"), [
    ("pair_agent_index", "ACTOR_PAIR_AGENT_INDEX_OUT_OF_RANGE"),
    ("safe_mask", "ACTOR_SAFE_MASK_DTYPE_INVALID"),
    ("agent_mask", "ACTOR_SAFE_PAIR_MAPPED_TO_INACTIVE_AGENT"),
    ("nan", "ACTOR_NONFINITE_FEATURE_INPUT"),
    ("inf", "ACTOR_NONFINITE_FEATURE_INPUT"),
])
def test_invalid_inputs_fail_closed(mutation: str, code: str) -> None:
    model, payload = actor(), inputs()
    if mutation == "pair_agent_index":
        payload["pair_agent_index"][0, 0] = 99
    elif mutation == "safe_mask":
        payload["safe_mask"] = payload["safe_mask"].float()
    elif mutation == "agent_mask":
        payload["agent_mask"][0, 0] = False
    elif mutation == "nan":
        payload["candidate_feats"][0, 0, 0] = float("nan")
    else:
        payload["candidate_feats"][0, 0, 0] = float("inf")
    with pytest.raises(H.ActorInputContractError) as caught:
        forward(model, payload)
    assert caught.value.code == code


def test_duplicate_candidate_identity_fails_upstream_snapshot_contract() -> None:
    with pytest.raises(FPS.FrozenPolicySnapshotError) as caught:
        FPS._normal_pair_ids([("agent-1", "candidate-1"),
                              ("agent-1", "candidate-1")])
    assert caught.value.code == "SNAPSHOT_DUPLICATE_CANDIDATE_IDENTITY"


def test_old_actor_checkpoint_strict_load_is_incompatible_without_padding() -> None:
    payload = torch.load(A1_CHECKPOINT, map_location="cpu", weights_only=False)
    repaired = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
        global_dim=8, demand_dim=6, agent_dim=4, candidate_dim=8,
        hidden=128, heads=4)
    with pytest.raises(RuntimeError, match="size mismatch for scorer.0.weight"):
        repaired.load_state_dict(payload["actor"], strict=True)
    legacy_shape = tuple(payload["actor"]["scorer.0.weight"].shape)
    repaired_shape = tuple(repaired.state_dict()["scorer.0.weight"].shape)
    assert legacy_shape == (128, 512)
    assert repaired_shape == (128, 640)
    assert legacy_shape != repaired_shape
