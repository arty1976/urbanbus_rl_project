"""Focused non-executing guards for the BT8-R14 frozen factor review.

The R14 executor is deliberately a read-only, frozen-inference audit.  These
tests exercise only its small pure classification helpers; they do not load a
checkpoint, access MPS, or create a rollout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r14_s3_frozen_factor_review as R14  # noqa: E402


def _classification(value: object) -> str:
    """Accept the executor's auditable mapping (or a future string shorthand)."""
    if isinstance(value, str):
        return value
    assert isinstance(value, dict)
    classification = value.get("classification")
    assert isinstance(classification, str)
    return classification


def _effect_row(*, family_mismatch: bool, selection_mismatch: bool, no_assign_delta: float,
                candidate_margin_delta: float | None, entropy_delta: float) -> dict[str, object]:
    return {
        "snapshot_digest": "same-preserved-snapshot",
        "action_family_mismatch": family_mismatch,
        "selection_mismatch": selection_mismatch,
        "max_abs_logit_delta": 0.0,
        "max_abs_probability_delta": 0.0,
        "no_assign_minus_best_pair_delta": no_assign_delta,
        "candidate_margin_delta": candidate_margin_delta,
        "entropy_delta": entropy_delta,
    }


def _deadlock_row(*, cell_id: str, feasible: bool, action: str, ancestry: bool,
                  eligible: bool) -> dict[str, object]:
    return {
        "cell_id": cell_id,
        "feasible_candidate_exists": feasible,
        "selected_action_type": action,
        "reward_ancestry_supported": ancestry,
        "actor_eligible_e1": eligible,
    }


def test_source_scope_and_exact_gate_are_frozen() -> None:
    assert R14.SOURCE_FILES == {
        "05_training/run_h4m_ae_ls3_bt8_r14_s3_frozen_factor_review.py",
        "05_training/test_h4m_ae_ls3_bt8_r14_s3_frozen_factor_review.py",
    }
    assert R14.PASS_GATE == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R14_S3_SAME_INPUT_FROZEN_POLICY_FACTOR_REVIEW_COMPLETE"
    assert R14.S3_CONTRACT_SHA256 == "66e2fb35de3aa767780d9f0f001d774919e059e580f4410fe1d01bd96b185393"


def test_state_dict_exactness_is_bitwise_not_numerical_tolerance() -> None:
    base = {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([3.0])}
    assert R14.state_dicts_exact(base, {key: value.clone() for key, value in base.items()}) is True
    assert R14.state_dicts_exact(base, {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([3.0000002])}) is False
    assert R14.state_dicts_exact(base, {"weight": torch.tensor([1.0, 2.0])}) is False


def test_environment_effect_is_low_only_for_exact_same_actor_outputs() -> None:
    rows = [
        _effect_row(family_mismatch=False, selection_mismatch=False, no_assign_delta=0.0,
                    candidate_margin_delta=0.0, entropy_delta=0.0),
        _effect_row(family_mismatch=False, selection_mismatch=False, no_assign_delta=0.0,
                    candidate_margin_delta=None, entropy_delta=0.0),
    ]
    result = R14.environment_effect(rows, actor_states_exact=True)
    assert _classification(result) == "ENVIRONMENT_EFFECT_LOW"


def test_environment_effect_marks_action_family_divergence_material_without_a_tolerance() -> None:
    rows = [
        _effect_row(family_mismatch=True, selection_mismatch=True, no_assign_delta=1.0e-12,
                    candidate_margin_delta=0.0, entropy_delta=0.0),
    ]
    result = R14.environment_effect(rows, actor_states_exact=False)
    assert _classification(result) == "ENVIRONMENT_EFFECT_MATERIAL"


def test_exploration_deadlock_requires_the_full_feasible_no_assign_credit_chain() -> None:
    confirmed = R14.exploration_deadlock([
        _deadlock_row(cell_id="BD-R1", feasible=True, action="NO_ASSIGN", ancestry=False, eligible=False)
        for _ in range(24)
    ])
    assert _classification(confirmed) == "INITIAL_POLICY_EXPLORATION_DEADLOCK_CONFIRMED"
    assert confirmed["feasible_candidate_decisions"] == confirmed["no_assign_selections"] == 24
    assert confirmed["candidate_plan_executions"] == confirmed["reward_ancestry_rows"] == confirmed["actor_eligible_rows"] == 0

    credit_limited = R14.exploration_deadlock([
        _deadlock_row(cell_id="AC-R1", feasible=True, action="CANDIDATE", ancestry=False, eligible=False)
        for _ in range(24)
    ])
    assert _classification(credit_limited) == "CREDIT_GENERATION_LIMITATION"


def test_frozen_review_counters_prohibit_all_execution_capabilities() -> None:
    value = R14.counters()
    for key in (
        "training", "optimizer_step", "causal_rollout", "candidate_generation",
        "candidate_regeneration", "local_search_rerun", "zero_loss_reevaluation",
        "parameter_mutation", "checkpoint_write", "checkpoint_mutation",
        "review_optimizer_rows", "future_leakage", "test6_access", "github_push",
    ):
        assert value[key] == 0


def test_source_contains_no_training_or_mutating_execution_path() -> None:
    source = (ROOT / "run_h4m_ae_ls3_bt8_r14_s3_frozen_factor_review.py").read_text(encoding="utf-8")
    for forbidden in (
        "optimizer.step(", ".backward(", "require_capability(", "adapter.step(",
        "factory.build(", "candidate_plan_authoritative", "compute_reward_v2(",
    ):
        assert forbidden not in source
