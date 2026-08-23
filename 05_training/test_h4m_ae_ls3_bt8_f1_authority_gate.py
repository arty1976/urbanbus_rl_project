"""Focused BT8-F1 authority-contract tests; they never run MPS or training."""

from __future__ import annotations

import sys

import pytest

sys.path.insert(0, "05_training")

import joint_assignment_f1_execution_contract as FC
import run_h4m_ae_ls3_bt8_f1_bounded_training as F1


def test_rejects_nonexact_full_batch_partition() -> None:
    with pytest.raises(FC.F1ExecutionContractError) as caught:
        FC.validate_contract(ppo={**FC.PPO_PARTITION, "minibatch_size": 12})
    assert caught.value.code == "F1_PPO_FULL_BATCH_EPOCH_INVALID"


def test_fixed_replicate_totals_are_exact() -> None:
    FC.validate_contract()
    assert sum(int(row["decisions"]) for row in FC.REPLICATES) == 48
    assert sum(int(row["trajectories"]) for row in FC.REPLICATES) == 12
    assert sum(int(row["ppo_update_cycles"]) for row in FC.REPLICATES) == 6


def test_execution_source_scope_is_closed() -> None:
    assert F1.SOURCE_FILES == {
        "05_training/run_h4m_ae_ls3_bt8_f1_bounded_training.py",
        "05_training/test_h4m_ae_ls3_bt8_f1_authority_gate.py",
    }


def test_f1_slot_mapping_scopes_to_the_frozen_source_groups() -> None:
    eligible = {
        "A": [("AGENT_005", 0, 1), ("AGENT_007", 0, 1)],
        "B": [("AGENT_001", 0, 1), ("AGENT_011", 0, 1), ("AGENT_019", 0, 1), ("AGENT_023", 0, 1)],
        "C": [("AGENT_003", 0, 1), ("AGENT_009", 0, 1), ("AGENT_017", 0, 1)],
    }
    mapping = F1.f1_agent_slot_mapping(eligible=eligible, source_groups=["A", "B", "C"], slots=8)
    assert len({mapping[agent] for agent, _, _ in eligible["A"]}) == 2
    assert len({mapping[agent] for agent, _, _ in eligible["B"]}) == 4
    assert len({mapping[agent] for agent, _, _ in eligible["C"]}) == 3
