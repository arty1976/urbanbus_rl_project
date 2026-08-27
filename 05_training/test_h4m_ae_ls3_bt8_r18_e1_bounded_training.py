"""Static guards for the R18 executor; these tests never execute training."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r18_e1_bounded_training as R18  # noqa: E402


def test_r18_rollout_uses_sealed_r16_selector_and_not_historical_argmax() -> None:
    source = inspect.getsource(R18._rollout_arm)
    assert "make_frozen_masked_distribution_view" in source
    assert "select_frozen_policy_action" in source
    assert "FROZEN_MASKED_CATEGORICAL_TRAINING" in source
    assert "H.select" not in source
    assert "torch.multinomial" not in source
    assert "masked_log_probs" not in source


def test_r18_train_path_asserts_support_before_every_epoch_and_requires_e1_rows() -> None:
    source = inspect.getsource(R18._train_arm)
    assert "behavior_equals_update_support" in source
    assert "ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE" in source
    assert "actor_eligible" in source
    assert "ineligible_actor_gradient" in source


def test_r18_captures_same_support_t1_review_deltas_without_optimizer_exposure() -> None:
    source = inspect.getsource(R18._frozen_review_replay) + inspect.getsource(R18._learning_counters) + inspect.getsource(R18.execute)
    assert "TIE_BREAK_CONTRACT_ID" in source
    assert "interpretation_performed" in source
    assert "logit_probability_shift_after_authorized_update" in source
    assert "duplicate_reward_ancestry" in source


def test_exact_execute_flag_is_required_by_parser(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        R18._parse_args(["--authorization-manifest", str(tmp_path / "a.json"), "--authorization-sha256", "a"])
    parsed = R18._parse_args(["--authorization-manifest", str(tmp_path / "a.json"), "--authorization-sha256", "a", "--dry-run"])
    assert parsed.dry_run is True
    assert parsed.execute_exact_r17_envelope is False


def test_dry_run_is_zero_execution() -> None:
    report = R18.dry_run_report({"source_commit": "source"})
    assert report["training"] == report["rollout"] == report["optimizer_step"] == report["checkpoint_write"] == 0
