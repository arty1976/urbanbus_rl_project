"""Focused pure guards for the R17 authorization/envelope selection."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r17_e1_bounded_training_authorization as R17  # noqa: E402


def _cells():
    return [
        {"cell_id": "AC-R1", "environment_seed": 20260822, "actor_seed": 20260824, "critic_seed": 20260826, "initialization_replicate": "F1_R1"},
        {"cell_id": "AC-R2", "environment_seed": 20260823, "actor_seed": 20260824, "critic_seed": 20260826, "initialization_replicate": "F1_R1"},
        {"cell_id": "BD-R1", "environment_seed": 20260822, "actor_seed": 20260825, "critic_seed": 20260827, "initialization_replicate": "F1_R2"},
        {"cell_id": "BD-R2", "environment_seed": 20260823, "actor_seed": 20260825, "critic_seed": 20260827, "initialization_replicate": "F1_R2"},
    ]


def _records():
    return [{"window_id": value, "source_group": f"source-{index}", "time_band": R17._time_band(value)}
            for index, value in enumerate(R17.WINDOWS)]


def _r14():
    return {"cells": {"AC-R1": {"candidate_plan_executions": 24, "actor_eligible_rows": 9},
                      "BD-R1": {"feasible_candidate_decisions": 24, "no_assign_selections": 24, "actor_eligible_rows": 0}}}


def _probe_rows():
    rows = []
    for band in R17.TIME_BANDS:
        rows.append({"cell_id": "BD-R1", "probe_seed": 0, "selected_is_no_assign": False, "time_band": band})
    return rows


def test_selects_two_arm_24_row_minimum_without_changing_frozen_ppo_scope() -> None:
    envelope = R17.select_minimal_envelope(r12_cells=_cells(), r12_windows=list(R17.WINDOWS),
                                            r4_window_records=_records(), r14_deadlock=_r14(), r16_probe_rows=_probe_rows())
    assert [row["arm_id"] for row in envelope["selected_arms"]] == ["AC_CONTROL_R1", "BD_E1_R1"]
    assert envelope["aggregate"] == {
        "environment_seed_count": 1, "windows": 6, "visits": 12, "trajectories": 12,
        "assignment_decisions": 48, "causal_transitions": 96, "assignment_actor_optimizer_steps_maximum": 6,
        "assignment_critic_optimizer_steps_exact": 6, "raw_optimizer_step_calls_maximum": 12, "supplemental_steps": 0,
    }
    assert all(len(row["categorical_probe_seed_by_decision"]) == 24 for row in envelope["selected_arms"])
    assert all(set(row["categorical_probe_seed_by_decision"].values()) == {0} for row in envelope["selected_arms"])


def test_rejects_envelope_without_all_band_bd_exposure() -> None:
    with pytest.raises(R17.R17Error, match=R17.NO_SAFE_BLOCK):
        R17.select_minimal_envelope(r12_cells=_cells(), r12_windows=list(R17.WINDOWS), r4_window_records=_records(),
                                    r14_deadlock=_r14(), r16_probe_rows=_probe_rows()[:-1])


def test_freeze_contract_keeps_e1_training_and_t1_inference_separate() -> None:
    contract = R17.module_freeze_contract(source_hashes={"actual": {"selector": "x"}})
    assert contract["training_selection_mode"] == "FROZEN_MASKED_CATEGORICAL_TRAINING"
    assert contract["inference_evaluation_mode"] == "FROZEN_INFERENCE_T1"
    assert contract["E1_only"] is True
    assert contract["E2_rescue"] is False
    assert contract["E3_temperature_or_floor"] is False
