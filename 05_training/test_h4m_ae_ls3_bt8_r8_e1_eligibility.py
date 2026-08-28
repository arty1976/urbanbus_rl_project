#!/usr/bin/env python3
"""Focused BT8-R8 E1 fixtures; all are CPU test-only and never step an optimizer."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joint_assignment_e1_eligibility as E1  # noqa: E402
import joint_assignment_learning as JL  # noqa: E402


R7 = ROOT / "artifacts" / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility_20260825_124641+09:00"


def _inputs(*, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    pair = torch.tensor([[0.2, -0.1], [0.3, 0.7], [-0.4, 0.6]], dtype=torch.float32,
                        requires_grad=requires_grad)
    no_assign = torch.tensor([[0.0], [0.1], [0.2]], dtype=torch.float32,
                             requires_grad=requires_grad)
    safe = torch.tensor([[True, True], [True, True], [True, True]])
    action = torch.tensor([0, 2, 1], dtype=torch.long)
    log_probs = JL.masked_log_probs(pair.detach(), no_assign.detach(), safe)
    old = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
    return {
        "new_pair_logits": pair,
        "new_no_assign_logit": no_assign,
        "safe_mask": safe,
        "action_index": action,
        "old_log_prob": old,
        "advantage": torch.tensor([1.0, -0.5, 0.75]),
        "value_pred": torch.tensor([0.2, -0.1, 0.4], requires_grad=requires_grad),
        "value_target": torch.tensor([0.0, 0.3, -0.2]),
        "forced_action": torch.tensor([False, False, False]),
    }


def _candidate_row(*, decision_id: str, category: str, reward: float, reward_component: float,
                   raw_gae: float, replicate: str = "F1_R1") -> dict[str, object]:
    return {
        "decision_id": decision_id,
        "replicate_id": replicate,
        "selected_action_type": "CANDIDATE",
        "selected_candidate_id": "candidate-a",
        "applied_candidate_id": "candidate-a",
        "credited_candidate_id": "candidate-a",
        "identity_chain_valid": True,
        "reward": reward,
        "reward_gae_component": reward_component,
        "raw_gae": raw_gae,
        "category": category,
        "actor_eligible_e1": category in E1.ACTOR_ELIGIBLE_CATEGORIES,
        "critic_eligible_e1": True,
    }


def test_all_eligible_e1_is_exactly_legacy_e0() -> None:
    inputs = _inputs()
    legacy = JL.assignment_ppo_loss(**inputs)
    e1 = JL.assignment_ppo_loss(**inputs, actor_eligibility_mask=torch.ones(3, dtype=torch.bool))
    for key in ("actor_loss", "policy_loss", "entropy", "critic_loss", "ratio", "new_log_prob",
                "policy_loss_per_row", "entropy_per_row"):
        assert torch.equal(torch.as_tensor(legacy[key]), torch.as_tensor(e1[key]))
    assert e1["actor_eligible_rows"] == 3
    assert e1["actor_update_skipped"] is False


def test_mixed_mask_matches_explicit_masked_reference() -> None:
    inputs = _inputs()
    mask = torch.tensor([True, False, True])
    out = JL.assignment_ppo_loss(**inputs, actor_eligibility_mask=mask)
    weights = mask.to(out["policy_loss_per_row"].dtype)
    reference_policy = (out["policy_loss_per_row"] * weights).sum() / weights.sum()
    reference_entropy = (out["entropy_per_row"] * weights).sum() / weights.sum()
    reference_critic = ((inputs["value_pred"] - inputs["value_target"]) ** 2).mean()
    assert torch.equal(out["policy_loss"], reference_policy)
    assert torch.equal(out["entropy"], reference_entropy)
    assert torch.equal(out["critic_loss"], reference_critic)
    assert out["actor_eligible_rows"] == 2
    assert out["actor_ineligible_rows"] == 1


def test_zero_eligible_is_explicit_actor_skip_with_critic_loss() -> None:
    inputs = _inputs(requires_grad=True)
    out = JL.assignment_ppo_loss(**inputs, actor_eligibility_mask=torch.zeros(3, dtype=torch.bool))
    pair_grad, no_assign_grad = torch.autograd.grad(
        out["actor_loss"], (inputs["new_pair_logits"], inputs["new_no_assign_logit"]), allow_unused=False)
    assert out["actor_update_skipped"] is True
    assert out["actor_eligible_rows"] == 0
    assert float(out["actor_loss"].detach()) == 0.0
    assert float(out["entropy"].detach()) == 0.0
    assert float(out["critic_loss"].detach()) > 0.0
    assert float(pair_grad.abs().max()) == 0.0
    assert float(no_assign_grad.abs().max()) == 0.0


def test_ineligible_rows_have_zero_actor_logit_gradient() -> None:
    inputs = _inputs(requires_grad=True)
    mask = torch.tensor([True, False, True])
    out = JL.assignment_ppo_loss(**inputs, actor_eligibility_mask=mask)
    pair_grad, no_assign_grad = torch.autograd.grad(
        out["actor_loss"], (inputs["new_pair_logits"], inputs["new_no_assign_logit"]), allow_unused=False)
    assert float(pair_grad[~mask].abs().max()) == 0.0
    assert float(no_assign_grad[~mask].abs().max()) == 0.0
    assert float(pair_grad[mask].abs().sum() + no_assign_grad[mask].abs().sum()) > 0.0


def test_invalid_mask_shape_or_dtype_fails_closed() -> None:
    inputs = _inputs()
    with pytest.raises(ValueError, match="ACTOR_ELIGIBILITY_MASK_MUST_BE_BOOL"):
        JL.assignment_ppo_loss(**inputs, actor_eligibility_mask=torch.tensor([1, 1, 1]))
    with pytest.raises(ValueError, match="ACTOR_ELIGIBILITY_MASK_SHAPE_MISMATCH"):
        JL.assignment_ppo_loss(**inputs, actor_eligibility_mask=torch.ones(2, dtype=torch.bool))


def test_r7_contract_and_preserved_row_distribution_bind_exactly() -> None:
    contract = json.loads((R7 / "bt8r7_selected_minimal_contract.json").read_text())
    rows = json.loads((R7 / "bt8r7_credit_eligibility_rows.json").read_text())["rows"]
    bound = E1.bind_e1_contract(contract)
    masks = {rep: E1.build_e1_eligibility_mask([row for row in rows if row["replicate_id"] == rep],
                                                expected_replicate_id=rep)
             for rep in ("F1_R1", "F1_R2")}
    assert bound["sha256"] == E1.E1_CONTRACT_SHA256
    assert masks["F1_R1"].actor_eligible_count == 9
    assert masks["F1_R2"].actor_eligible_count == 0
    assert sum(mask.actor_eligible_count for mask in masks.values()) == 9
    assert sum(mask.actor_ineligible_count for mask in masks.values()) == 39
    assert sum(sum(mask.critic_eligible) for mask in masks.values()) == 48


def test_label_digest_seed_review_and_identity_tampering_fail_closed() -> None:
    direct = _candidate_row(decision_id="BT8_F1:F1_R1:0:0", category="DIRECT_CANDIDATE_REWARD",
                            reward=1.0, reward_component=1.0, raw_gae=1.0)
    bad_label = copy.deepcopy(direct)
    bad_label["category"] = "CRITIC_ONLY_NO_REWARD_ANCESTRY"
    bad_label["actor_eligible_e1"] = False
    with pytest.raises(E1.E1EligibilityError, match="E1_ANCESTRY_LABEL_TAMPER"):
        E1.build_e1_eligibility_mask([bad_label], expected_replicate_id="F1_R1")
    bad_identity = copy.deepcopy(direct)
    bad_identity["identity_chain_valid"] = False
    with pytest.raises(E1.E1EligibilityError, match="E1_INSUFFICIENT_IDENTITY_CREDIT"):
        E1.build_e1_eligibility_mask([bad_identity], expected_replicate_id="F1_R1")
    review = copy.deepcopy(direct)
    review["data_role"] = "REVIEW"
    with pytest.raises(E1.E1EligibilityError, match="E1_REVIEW_ROW_IN_ACTOR_LOSS"):
        E1.build_e1_eligibility_mask([review], expected_replicate_id="F1_R1")
    other_seed = copy.deepcopy(direct)
    other_seed["replicate_id"] = "F1_R2"
    other_seed["decision_id"] = "BT8_F1:F1_R2:0:0"
    with pytest.raises(E1.E1EligibilityError, match="E1_SEED_MIXING_OR_REPLICATE_MISMATCH"):
        E1.build_e1_eligibility_mask([direct, other_seed], expected_replicate_id="F1_R1")
    contract = json.loads((R7 / "bt8r7_selected_minimal_contract.json").read_text())
    contract["sha256"] = "0" * 64
    with pytest.raises(E1.E1EligibilityError, match="E1_CONTRACT_SHA256_MISMATCH"):
        E1.bind_e1_contract(contract)


def test_identity_aligned_permutation_is_invariant() -> None:
    direct = _candidate_row(decision_id="BT8_F1:F1_R1:0:0", category="DIRECT_CANDIDATE_REWARD",
                            reward=1.0, reward_component=1.0, raw_gae=1.0)
    critic_only = _candidate_row(decision_id="BT8_F1:F1_R1:0:1", category="CRITIC_ONLY_NO_REWARD_ANCESTRY",
                                 reward=0.0, reward_component=0.0, raw_gae=0.2)
    original = E1.build_e1_eligibility_mask([direct, critic_only], expected_replicate_id="F1_R1")
    reversed_rows = E1.build_e1_eligibility_mask([critic_only, direct], expected_replicate_id="F1_R1")
    assert dict(zip(original.decision_ids, original.actor_eligible)) == dict(zip(reversed_rows.decision_ids, reversed_rows.actor_eligible))


def test_r8_runner_contains_no_training_or_optimizer_path() -> None:
    source = (ROOT / "run_h4m_ae_ls3_bt8_r8_e1_eligibility_validation.py").read_text()
    for forbidden in ("optimizer.step(", "apply_assignment_update(", "require_capability(",
                      "adapter.step(", "factory.build(", "save_checkpoint"):
        assert forbidden not in source
