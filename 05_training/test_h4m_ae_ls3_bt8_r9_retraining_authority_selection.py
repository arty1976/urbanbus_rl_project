#!/usr/bin/env python3
"""Focused read-only fixtures for BT8-R9 P1 authorization selection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joint_assignment_frozen_policy_snapshot as FPS  # noqa: E402
import run_h4m_ae_ls3_bt8_r9_retraining_authority_selection as R9  # noqa: E402


def test_transition_ledger_parser_keeps_exact_persisted_fields() -> None:
    text = "AssignmentTransition(assignment_step_id='BT8_F1:F1_R1:0:0', old_log_prob=-0.6011579632759094, seed=20260822)"
    assert R9.parse_transition_ledger(text) == {
        "decision_id": "BT8_F1:F1_R1:0:0", "old_log_prob": -0.6011579632759094, "seed": 20260822,
    }


@pytest.mark.parametrize("text", [
    "AssignmentTransition(assignment_step_id='x', seed=1)",
    "AssignmentTransition(assignment_step_id='x', old_log_prob=nan, seed=1)",
    "AssignmentTransition(assignment_step_id='x', old_log_prob=0.0, seed=1, seed=2)",
])
def test_transition_ledger_parser_fails_closed_for_missing_nonfinite_or_ambiguous_fields(text: str) -> None:
    with pytest.raises(R9.R9Error):
        R9.parse_transition_ledger(text)


def test_p1_is_preferred_over_p2_when_complete_and_on_policy_bound() -> None:
    selected = R9.select_retraining_mode(p1_complete=True, on_policy_bound=True, p2_complete=True)
    assert selected["selected_mode"] == "P1"
    assert selected["classification"] == R9.PASS_CLASS


def test_missing_p1_does_not_imply_unauthorized_p2() -> None:
    selected = R9.select_retraining_mode(p1_complete=False, on_policy_bound=False, p2_complete=False)
    assert selected["selected_mode"] == "BLOCKED"


def test_frozen_e1_budget_has_no_compensating_actor_steps() -> None:
    budget = R9.expected_budget()
    assert budget["aggregate"] == {
        "ppo_epochs_per_replicate": 3,
        "full_batch_size_per_replicate": 24,
        "minibatch_size_per_replicate": 24,
        "actor_optimizer_steps": 3,
        "critic_optimizer_steps": 6,
        "raw_optimizer_step_calls": 9,
        "extra_compensating_steps": 0,
    }
    r1, r2 = budget["replicates"]
    assert (r1["actor_eligible_rows"], r1["actor_ineligible_rows"], r1["actor_optimizer_steps"]) == (9, 15, 3)
    assert (r2["actor_eligible_rows"], r2["actor_ineligible_rows"], r2["actor_optimizer_steps"]) == (0, 24, 0)
    assert r2["zero_eligible_actor_behavior"] == "EXPLICIT_SKIP"


def test_actual_f1_preserved_batch_is_complete_for_p1_without_replay_or_rollout() -> None:
    audit = R9.audit_preserved_batch(loader=FPS)
    assert audit["p1_evidence_complete"] is True
    assert audit["training_collection"]["snapshot_count"] == 48
    assert audit["review_collection"]["snapshot_count"] == 6
    for replicate_id, expected_eligible in (("F1_R1", 9), ("F1_R2", 0)):
        row = audit["replicates"][replicate_id]
        assert row["rows"]["train"] == 24
        assert row["rows"]["trajectory_count"] == 6
        assert row["old_log_probability"]["count"] == 24
        assert row["initial_actor_critic_checkpoint"]["all_sha_equal"] is True
        assert row["action_support_mask"]["all_selected_actions_legal"] is True
        assert row["candidate_identity"]["selected_applied_credited_exact"] is True
        assert row["e1"]["actor_eligible"] == expected_eligible


def test_actual_f1_p1_on_policy_binding_is_exact() -> None:
    audit = R9.audit_preserved_batch(loader=FPS)
    on_policy = R9.audit_on_policy_binding(completeness=audit)
    assert on_policy["p1_on_policy_binding_pass"] is True
    assert all(row["same_sha_exact"] for row in on_policy["replicates"].values())
    assert all(row["old_log_probability_rows"] == 24 for row in on_policy["replicates"].values())


def test_r9_source_has_no_training_execution_path() -> None:
    source = (ROOT / "run_h4m_ae_ls3_bt8_r9_retraining_authority_selection.py").read_text()
    for forbidden in ("optimizer.step(", "apply_assignment_update(", "require_capability(",
                      "adapter.step(", "torch.save(", "save_checkpoint"):
        assert forbidden not in source
