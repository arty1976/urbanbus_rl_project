from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
MODULE_PATH = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r3_holdout_evaluation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dl6d_r3_ho1", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_holdout_candidate_revalidation_matches_locked_c1():
    module = load_module()
    audit = module.candidate_hashes()
    assert audit["candidate_id"] == "R3_C1_NORMALIZATION_ONLY"
    assert audit["repair_level"] == "NORMALIZATION_ONLY"
    assert audit["three_way_candidate_hash_match"] is True
    assert audit["reward_formula_changed"] is False
    assert audit["reward_weights_changed"] is False
    assert audit["window_settlement_added"] is False
    assert audit["hinge_added"] is False
    assert audit["direct_action_id_penalty_added"] is False


def test_holdout_seal_revalidation_uses_pre_holdout_hashes():
    module = load_module()
    audit = module.seal_revalidation()
    assert audit["seal_unchanged"] is True
    assert audit["sealed_lock_unchanged"] is True
    assert audit["harmful_ids_unchanged"] is True
    assert audit["beneficial_ids_unchanged"] is True
    assert audit["tolerance_unchanged"] is True
    assert audit["beneficial_gate_unchanged"] is True


def test_holdout_preopen_cardinality_and_original_seal_unopened():
    module = load_module()
    state = module.preopen_state()
    assert state["holdout_opened_lock_exists_before_run"] is False
    assert state["holdout_opened_before_run"] is False
    assert state["holdout_open_count_before_run"] == 0
    assert state["sealed_harmful_id_count"] == 26
    assert state["sealed_beneficial_id_count"] == 26


def test_holdout_rows_reproduce_expected_c1_outcome_without_mutation():
    module = load_module()
    candidate = module.candidate_hashes()
    harmful, beneficial = module.load_holdout_rows(candidate["normalization_scale_correction"])
    h_summary = module.harmful_summary(harmful)
    b_summary = module.beneficial_summary(beneficial)
    pattern, _table = module.harm_pattern(harmful)
    assert len(harmful) == 26
    assert len(beneficial) == 26
    assert h_summary["harmful_misalignment_count"] == 1
    assert h_summary["maximum_post_c1_margin"] > 0
    assert b_summary["beneficial_retained_count"] == 26
    assert b_summary["beneficial_candidate_untouched_by_c1"] is True
    assert pattern["unique_harmful_kpi_patterns"] == 1
