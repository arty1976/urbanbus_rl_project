"""Focused source-level guards for the BT8-R13 four-cell executor.

These are deliberately non-executing tests.  The actual MPS/capability-gated
causal run is authorized only by the R13 command-line executor after its
source-only commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r13_s3_four_cell_execution as R13  # noqa: E402


def test_source_scope_and_exact_gate_constants_are_frozen() -> None:
    assert R13.SOURCE_FILES == {
        "05_training/run_h4m_ae_ls3_bt8_r13_s3_four_cell_execution.py",
        "05_training/test_h4m_ae_ls3_bt8_r13_s3_four_cell_execution.py",
    }
    assert R13.PASS_GATE == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R13_S3_FOUR_CELL_ON_POLICY_CAUSAL_EXECUTION_COMPLETE"
    assert R13.S3_CONTRACT_SHA256 == "66e2fb35de3aa767780d9f0f001d774919e059e580f4410fe1d01bd96b185393"
    assert R13.E1_CONTRACT_SHA256 == "eb84543a9fc06dcf730e49aa3895d9fe26d2244a05ce340986b7449418205ad9"


def test_cell_budget_rejects_every_overrun_before_a_mutation() -> None:
    budget = R13.CellBudget("AC-R1", allowed_actor_steps=3)
    for index in range(24):
        budget.decision(f"d{index}")
    with pytest.raises(R13.R13Error, match="decision_overrun"):
        budget.decision("d24")
    for index in range(6):
        budget.trajectory(f"t{index}")
    with pytest.raises(R13.R13Error, match="trajectory_overrun"):
        budget.trajectory("t6")
    for _ in range(48):
        budget.transition()
    with pytest.raises(R13.R13Error, match="transition_overrun"):
        budget.transition()
    for _ in range(3):
        budget.cycle(); budget.actor_step(); budget.critic_step()
    budget.finalize()
    with pytest.raises(R13.R13Error, match="actor_step_overrun"):
        budget.actor_step()


def test_zero_eligible_cell_has_explicit_zero_actor_budget() -> None:
    budget = R13.CellBudget("BD-R2", allowed_actor_steps=0)
    with pytest.raises(R13.R13Error, match="actor_step_overrun"):
        budget.actor_step()


def test_review_contract_groups_the_same_three_snapshots_by_environment() -> None:
    collection = {
        "entries": [
            {"seed": 20260822, "decision_index": index, "snapshot_digest": f"r1-{index}"}
            for index in range(3)
        ] + [
            {"seed": 20260823, "decision_index": index, "snapshot_digest": f"r2-{index}"}
            for index in range(3)
        ]
    }
    grouped = R13._review_entries_by_environment(collection)
    assert [row["snapshot_digest"] for row in grouped["F1_R1"]] == ["r1-0", "r1-1", "r1-2"]
    assert [row["snapshot_digest"] for row in grouped["F1_R2"]] == ["r2-0", "r2-1", "r2-2"]
    bad = {"entries": collection["entries"][:-1] + [{"seed": 7, "decision_index": 2, "snapshot_digest": "bad"}]}
    with pytest.raises(R13.R13Error, match="review_seed"):
        R13._review_entries_by_environment(bad)


def test_cell_model_and_optimizer_identity_is_not_shared() -> None:
    def model() -> dict[str, object]:
        actor, critic = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
        return {"actor": actor, "critic": critic, "actor_opt": torch.optim.Adam(actor.parameters()),
                "critic_opt": torch.optim.Adam(critic.parameters())}

    cells = {name: model() for name in ("AC-R1", "AC-R2", "BD-R1", "BD-R2")}
    evidence = R13.assert_model_independence(cells)
    assert evidence["independent"] is True
    assert evidence["parameter_object_alias_count"] == 0
    cells["AC-R2"]["actor"] = cells["AC-R1"]["actor"]
    with pytest.raises(R13.R13Error, match="aliasing"):
        R13.assert_model_independence(cells)


def test_training_selector_and_review_t1_scopes_are_explicitly_separated() -> None:
    source = (ROOT / "run_h4m_ae_ls3_bt8_r13_s3_four_cell_execution.py").read_text(encoding="utf-8")
    assert "H.select is the frozen training-time selection path" in source
    assert "T1 not used in training-time rollout" in source
    assert "TIE.TIE_BREAK_CONTRACT_ID" in source
    assert "historical_f1_training_collection_reused\": False" in source
    assert "candidate_regeneration_during_ppo" in source
    assert "E1.build_e1_eligibility_mask" in source


def test_normalization_summary_is_cell_local_and_has_no_mask_dependency() -> None:
    result = R13._normalization_summary([1.0, 2.0, 3.0], [-1.0, 0.0, 1.0])
    assert result["scope"] == "cell-local 24 train rows before E1 mask"
    assert result["row_count"] == 3
    assert result["normalized_mean"] == 0.0
