from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
MODULE_PATH = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r3_pre_holdout_audit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dl6d_r3_ph", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pre_holdout_selects_single_authoritative_c1_artifact():
    module = load_module()
    selected, payload = module.select_r3_artifact()
    assert selected is not None
    assert payload["valid_authoritative_count"] == 1
    assert selected.name == "prompt5_e01_dl6d_r3_reward_service_alignment_repair_20260802_112731"


def test_pre_holdout_c0_exact_counts_reproduce_r2():
    module = load_module()
    selected, _payload = module.select_r3_artifact()
    audit, table = module.c0_exact_counts(selected)
    assert audit["all_exact_counts_match"] is True
    assert bool(table["exact_match"].all())
    assert dict(zip(table["metric_name"], table["actual_count"]))["problem_registry_rows"] == 86
    assert dict(zip(table["metric_name"], table["actual_count"]))["thirty_minute_skip_better"] == 524


def test_pre_holdout_candidate_three_way_hash_matches():
    module = load_module()
    selected, _payload = module.select_r3_artifact()
    audit = module.candidate_hash_audit(selected)
    assert audit["selected_candidate"] == "R3_C1_NORMALIZATION_ONLY"
    assert audit["selected_repair_level"] == "NORMALIZATION_ONLY"
    assert audit["three_way_candidate_hash_match"] is True
    assert audit["candidate_locked"] is True


def test_pre_holdout_root_cause_evidence_is_quantitative_and_consistent():
    module = load_module()
    selected, _payload = module.select_r3_artifact()
    evidence, signal, margin, case = module.root_cause_evidence(selected)
    assert evidence["primary_cause"] == "NORMALIZATION_SCALE_MISMATCH"
    assert "TEMPORAL_DILUTION" in evidence["secondary_causes"]
    assert evidence["raw_service_harm_total_pre_c1"] > 0
    assert evidence["post_normalization_service_harm_total_c1"] > evidence["post_normalization_service_harm_total_pre_c1"]
    assert evidence["harmful_design_positive_margin_pre_c1"] == 60
    assert evidence["harmful_design_positive_margin_c1"] == 0
    assert evidence["beneficial_design_positive_margin_pre_c1"] == 60
    assert evidence["beneficial_design_positive_margin_c1"] == 60
    assert evidence["candidate_mechanism_matches_root_cause"] is True
    assert evidence["evidence_sufficient_for_manual_review"] is True
    assert len(signal) == 3
    assert len(margin) == 2
    assert bool(case["case_evidence_consistent"].all())
