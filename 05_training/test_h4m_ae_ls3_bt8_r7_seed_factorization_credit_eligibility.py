"""Focused pure tests for BT8-R7's read-only selection gate."""

from __future__ import annotations

import inspect
import sys

sys.path.insert(0, "05_training")

import run_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility as R7


def test_eligibility_categories_follow_reward_ancestry_and_identity() -> None:
    assert R7.eligibility_category(action_type="CANDIDATE", reward=1.0, reward_component=1.0,
                                   identity_chain_valid=True, raw_gae=1.0) == "DIRECT_CANDIDATE_REWARD"
    assert R7.eligibility_category(action_type="CANDIDATE", reward=0.0, reward_component=0.1,
                                   identity_chain_valid=True, raw_gae=0.1) == "TEMPORALLY_PROPAGATED_CANDIDATE_REWARD"
    assert R7.eligibility_category(action_type="NO_ASSIGN", reward=0.0, reward_component=0.1,
                                   identity_chain_valid=True, raw_gae=0.1) == "NO_ASSIGN_REWARD_SUPPORTED"
    assert R7.eligibility_category(action_type="NO_ASSIGN", reward=0.0, reward_component=0.0,
                                   identity_chain_valid=True, raw_gae=-0.1) == "CRITIC_ONLY_NO_REWARD_ANCESTRY"
    assert R7.eligibility_category(action_type="CANDIDATE", reward=1.0, reward_component=1.0,
                                   identity_chain_valid=False, raw_gae=1.0) == "INSUFFICIENT_IDENTITY_CREDIT"


def test_standardize_is_scope_local_and_empty_scope_is_explicit() -> None:
    values, audit = R7.standardize([1.0, 3.0])
    assert values == [-1.0, 1.0]
    assert audit["row_count"] == 2
    assert R7.standardize([])[0] == []
    assert R7.standardize([])[1]["mode"] == "NO_ROWS"


def test_action_direction_does_not_add_action_specific_bonus() -> None:
    assert R7.action_direction("CANDIDATE", 1.0) == "CANDIDATE_RELATIVE_TO_NO_ASSIGN"
    assert R7.action_direction("CANDIDATE", -1.0) == "NO_ASSIGN_RELATIVE_TO_CANDIDATE"
    assert R7.action_direction("NO_ASSIGN", 1.0) == "NO_ASSIGN_RELATIVE_TO_CANDIDATE"
    assert R7.action_direction("NO_ASSIGN", -1.0) == "CANDIDATE_RELATIVE_TO_NO_ASSIGN"


def test_fixed_gae_keeps_reward_and_bootstrap_components_separate() -> None:
    rows = [
        {"decision_id": "BT8_F1:F1_R1:0:0", "trajectory_id": "t", "nonterminal": True, "reward": 1.0},
        {"decision_id": "BT8_F1:F1_R1:0:1", "trajectory_id": "t", "nonterminal": False, "reward": 0.0},
    ]
    values = {"BT8_F1:F1_R1:0:0": 0.0, "BT8_F1:F1_R1:0:1": 0.0}
    result = R7.fixed_gae(rows, values, gamma=0.99, lam=0.95)
    assert result[0]["raw_gae"] == result[0]["reward_gae_component"] == 1.0
    assert result[1]["raw_gae"] == 0.0
    assert max(abs(row["raw_additivity_residual"]) for row in result) == 0.0


def test_factorization_classification_needs_both_exact_factors() -> None:
    actor = {"A_candidate": 6, "B_no_assign": 6, "canonical_rows": 6}
    critic = {"C_all_negative": True, "D_all_positive": True}
    lineages = {"R1_reward_rows": 4, "R2_reward_rows": 0}
    result = R7.classify_factorization(actor, critic, lineages)
    assert result == {"actor_seed_effect": "ACTOR_INIT_DOMINANT", "critic_seed_effect": "CRITIC_INIT_DOMINANT",
                      "interaction": "ACTOR_CRITIC_INTERACTION"}


def test_e1_contract_excludes_only_unsupported_actor_credit_and_keeps_critic_rows() -> None:
    rows = [
        {"category": "DIRECT_CANDIDATE_REWARD", "actor_eligible_e1": True, "identity_chain_valid": True},
        {"category": "CRITIC_ONLY_NO_REWARD_ANCESTRY", "actor_eligible_e1": False, "identity_chain_valid": True},
    ]
    outcome = R7.select_contract(rows, {"F1_R1": {}, "F1_R2": {}})
    selected = outcome["selected_contract"]
    assert selected["selected_option"] == "E1"
    assert selected["normalization"] == "N0 retained; no normalizer change selected"
    assert outcome["options"][1]["actor_loss_rows"] == 1
    assert outcome["options"][1]["critic_loss_rows"] == 2


def test_runner_has_no_training_rollout_or_parameter_update_call() -> None:
    source = inspect.getsource(R7)
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "apply_assignment_update(" not in source
    assert "adapter.step(" not in source
    assert "factory.build(" not in source
    assert "require_capability(" not in source
