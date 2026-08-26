"""Focused, no-execution contracts for BT8-R12."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection as R12  # noqa: E402


def _cells() -> list[dict[str, object]]:
    return [
        {"cell_id": "AC-R1", "environment_replicate": "F1_R1", "actor_initial_checkpoint_sha256": "a" * 64},
        {"cell_id": "AC-R2", "environment_replicate": "F1_R2", "actor_initial_checkpoint_sha256": "a" * 64},
        {"cell_id": "BD-R1", "environment_replicate": "F1_R1", "actor_initial_checkpoint_sha256": "b" * 64},
        {"cell_id": "BD-R2", "environment_replicate": "F1_R2", "actor_initial_checkpoint_sha256": "b" * 64},
    ]


def test_m0_cross_cells_fail_on_policy_and_m2_is_the_only_uniform_selection() -> None:
    mode = R12.select_on_policy_mode(_cells(), {"F1_R1": "a" * 64, "F1_R2": "b" * 64})
    assert mode["selected_mode"] == "M2"
    assert mode["M0"]["all_four_cells_on_policy"] is False
    assert mode["M0"]["diagonal_cells_strict_match"] is True
    assert mode["M0"]["cross_cells_strict_match"] is False
    assert mode["M1"]["uniform_execution_origin"] is False
    assert mode["M2"]["execution_authorized"] is False


def test_mode_selection_fails_closed_on_incomplete_cell_matrix() -> None:
    with pytest.raises(R12.R12Error, match=R12.ON_POLICY_BLOCK):
        R12.select_on_policy_mode(_cells()[:3], {"F1_R1": "a" * 64, "F1_R2": "b" * 64})


def test_cell_budget_preserves_exact_four_cell_bounds_and_zero_eligibility_rule() -> None:
    budget = R12.cell_update_budget(_cells(), [f"w{index}" for index in range(6)])
    assert len(budget["per_cell"]) == 4
    assert budget["aggregate_authorized_maximum"] == {
        "train_windows_per_cell": 6,
        "decisions": 96,
        "trajectories": 24,
        "causal_transitions": 192,
        "actor_optimizer_steps_maximum": 12,
        "critic_optimizer_steps_exact": 12,
        "raw_optimizer_step_calls_maximum": 24,
        "supplemental_steps": 0,
    }
    zero = budget["per_cell"][0]["actor_eligibility"]["eligible_zero"]
    assert zero == {"actor_optimizer_steps": 0, "status": "ACTOR_NOT_TRAINED_INSUFFICIENT_CREDIT"}


def test_review_support_reuses_same_three_inputs_per_environment() -> None:
    collection = {
        "collection_digest": "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e",
        "entries": [
            {"seed": 20260822, "decision_index": index, "snapshot_digest": f"a{index}", "window_id": f"r1-{index}"}
            for index in range(3)
        ] + [
            {"seed": 20260823, "decision_index": index, "snapshot_digest": f"b{index}", "window_id": f"r2-{index}"}
            for index in range(3)
        ],
    }
    result = R12.review_support_contract(collection, _cells())
    assert result["unique_snapshot_count"] == 6
    assert result["environment_contracts"]["F1_R1"]["cells_reusing_exact_same_inputs"] == ["AC-R1", "BD-R1"]
    assert result["environment_contracts"]["F1_R2"]["cells_reusing_exact_same_inputs"] == ["AC-R2", "BD-R2"]


def test_source_has_no_execution_path_and_declares_only_r12_files() -> None:
    source = (ROOT / "run_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection.py").read_text(encoding="utf-8")
    for forbidden in ("optimizer.step(", ".backward(", "require_capability(", "adapter.step(", "torch.save(", "torch.load(",
                      "commit_candidate(", "commit_no_assign(", "factory.build("):
        assert forbidden not in source
    assert R12.SOURCE_FILES == {
        "05_training/run_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection.py",
        "05_training/test_h4m_ae_ls3_bt8_r12_s3_execution_authority_selection.py",
    }
