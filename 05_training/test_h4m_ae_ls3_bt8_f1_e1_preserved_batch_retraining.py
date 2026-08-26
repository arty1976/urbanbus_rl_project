#!/usr/bin/env python3
"""Read-only source and accounting fixtures for BT8-F1-E1."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joint_assignment_frozen_policy_snapshot as FPS  # noqa: E402
import run_h4m_ae_ls3_bt8_f1_e1_preserved_batch_retraining as E1RUN  # noqa: E402
import run_h4m_ae_ls3_bt8_r9_retraining_authority_selection as R9  # noqa: E402


def test_exact_budget_enforces_r1_r2_asymmetric_actor_steps() -> None:
    r1 = E1RUN.ExactBudget("F1_R1", allowed_actor_steps=3, allowed_critic_steps=3)
    r2 = E1RUN.ExactBudget("F1_R2", allowed_actor_steps=0, allowed_critic_steps=3)
    ids = [f"BT8_F1:F1_R1:{window}:0" for window in range(6)]
    repeated_ids = [f"BT8_F1:F1_R1:{window}:{step}" for window in range(6) for step in range(4)]
    trajectories = [f"T{window}" for window in range(6) for _ in range(4)]
    r1.bind_batch(decision_ids=repeated_ids, trajectories=trajectories)
    r2.bind_batch(decision_ids=[value.replace("R1", "R2") for value in repeated_ids], trajectories=trajectories)
    assert len(ids) == 6  # Makes the intended 6x4 fixture shape explicit.
    for _ in range(3):
        r1.begin_cycle(); r1.actor_step(); r1.critic_step()
        r2.begin_cycle(); r2.critic_step()
    r1.finalize(); r2.finalize()
    assert (r1.actor_steps, r1.critic_steps) == (3, 3)
    assert (r2.actor_steps, r2.critic_steps) == (0, 3)


def test_budget_blocks_fourth_actor_or_critic_step_before_mutation() -> None:
    budget = E1RUN.ExactBudget("F1_R1", allowed_actor_steps=3, allowed_critic_steps=3)
    for _ in range(3):
        budget.actor_step(); budget.critic_step()
    with pytest.raises(E1RUN.F1E1Error, match="E1_ACTOR_STEP_OVERRUN"):
        budget.actor_step()
    with pytest.raises(E1RUN.F1E1Error, match="E1_CRITIC_STEP_OVERRUN"):
        budget.critic_step()


def test_actual_r9_preserved_evidence_matches_execution_scope() -> None:
    audit = R9.audit_preserved_batch(loader=FPS)
    assert audit["p1_evidence_complete"] is True
    assert audit["training_collection"]["snapshot_count"] == 48
    assert audit["review_collection"]["snapshot_count"] == 6
    assert audit["replicates"]["F1_R1"]["e1"] == {"actor_eligible": 9, "actor_ineligible": 15, "critic_eligible": 24}
    assert audit["replicates"]["F1_R2"]["e1"] == {"actor_eligible": 0, "actor_ineligible": 24, "critic_eligible": 24}


def test_execution_source_has_no_rollout_or_candidate_regeneration_path() -> None:
    source = (ROOT / "run_h4m_ae_ls3_bt8_f1_e1_preserved_batch_retraining.py").read_text()
    for forbidden in ("adapter.step(", "factory.build(", "compute_reward_v2(", "selected_plan(",
                      "candidate_plan_authoritative", "Local Search"):
        assert forbidden not in source


def test_frozen_executor_source_scope_is_closed() -> None:
    assert E1RUN.SOURCE_FILES == {
        "05_training/run_h4m_ae_ls3_bt8_f1_e1_preserved_batch_retraining.py",
        "05_training/test_h4m_ae_ls3_bt8_f1_e1_preserved_batch_retraining.py",
    }
