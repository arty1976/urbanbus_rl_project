from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
MODULE_PATH = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r3_reward_service_alignment_repair.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dl6d_r3", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_problem_registry_reproduces_86_rows():
    module = load_module()
    frame = module.load_base_frame()
    registry, payload = module.registry_from_frame(frame)
    assert len(registry) == 86
    assert payload["actual_problem_rows"] == 86


def test_prepare_split_cardinalities_and_no_holdout_leakage():
    module = load_module()
    frame = module.load_base_frame()
    registry, _payload = module.registry_from_frame(frame)
    design_h, holdout_h, pool, matches, design_b, split, matching = module.build_splits(frame, registry)
    assert len(design_h) == 60
    assert len(holdout_h) == 26
    assert len(design_b) == 60
    assert matching["beneficial_pool_count"] >= 86
    assert len(matches) == 86
    assert not set(design_h["row_id"]).intersection(set(holdout_h["row_id"]))
    assert split["holdout_outcomes_hidden_in_prepare_design"] is True


def test_c1_normalization_only_passes_design_gate_without_c2():
    module = load_module()
    frame = module.load_base_frame()
    registry, _payload = module.registry_from_frame(frame)
    design_h, _holdout_h, _pool, _matches, design_b, _split, _matching = module.build_splits(frame, registry)
    candidates, _results, summary = module.c1_design(frame, design_h, design_b)
    c1 = candidates["c1"]
    assert c1["formula_changed"] is False
    assert c1["weight_changed"] is False
    assert c1["window_settlement_added"] is False
    assert c1["action_id_penalty_added"] is False
    assert c1["design_service_misalignment_count"] == 0
    assert c1["design_beneficial_retention_count"] == 60
    assert c1["design_gate_passed"] is True
    assert summary["selected_candidate_id"] == "R3_C1_NORMALIZATION_ONLY"
    assert summary["c2_entry_locked_pending_user_command"] is True


def test_c1_micro_scenarios_all_pass_and_no_direct_action_penalty():
    module = load_module()
    frame = module.load_base_frame()
    registry, _payload = module.registry_from_frame(frame)
    design_h, _holdout_h, _pool, _matches, design_b, _split, _matching = module.build_splits(frame, registry)
    candidates, _results, _summary = module.c1_design(frame, design_h, design_b)
    micro, payload = module.micro_scenarios(candidates["c1"]["normalization_scale_correction"])
    action, double, temporal, scope, _logging = module.static_audits(candidates["c1"]["normalization_scale_correction"])
    assert payload["reward_micro_scenarios_passed"] == 12
    assert bool(micro["passed"].all())
    assert action["direct_action_id_penalty_detected"] is False
    assert double["reward_double_counting_detected"] is False
    assert temporal["temporal_credit_after_repair_valid"] is True
    assert scope["closed_loop_multi_agent_reward_alignment_verified"] is False
