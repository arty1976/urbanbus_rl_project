from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
MODULE_PATH = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_pa1a_provenance_and_dynamics_feasibility.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dl6d_pa1a", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dl6d_pa1a"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_path_b_decision_is_user_fixed():
    module = load_module()
    record = module.path_b_decision_record()
    assert record["original_research_intent"] == "REAL_SIMULATOR_DYNAMICS_THIRTY_MINUTE_BRANCH"
    assert record["adjudication_path"] == "PATH_B"
    assert record["path_decision_source"] == "USER"
    assert record["reduced_form_proxy_was_intended_modeling_choice"] is False
    assert record["decision_revisitable_in_pa1a"] is False


def test_r1_thirty_minute_audit_has_no_transition_calls():
    module = load_module()
    table, audit = module.proxy_provenance()
    assert audit["function_name"] == "thirty_minute_audit"
    assert audit["simulator_transition_call_count"] == 0
    assert audit["r1_reduced_form_proxy_confirmed"] is True
    assert (table["classification"] == "DERIVED_INVARIANT").sum() >= 2
    assert "service_harm_excess_native_harm_pattern" in set(table["value_name"])


def test_validation_and_test_audits_do_not_allow_row_access():
    module = load_module()
    seal = module.validation_seal_revalidation()
    validation = module.validation_untouched_audit(seal)
    test = module.test_holdout_untouched_audit()
    assert validation["validation_row_level_feature_access_allowed"] is False
    assert validation["validation_row_level_access_count"] == 0
    assert validation["validation_branch_count"] == 0
    assert validation["validation_reward_computation_count"] == 0
    assert test["prior_test_row_access_allowed"] is False
    assert test["test_sealed_holdout_rows_read"] == 0
    assert test["test_sealed_holdout_touched"] is False


def test_relabeling_registry_only_does_not_apply_to_existing_artifacts():
    module = load_module()
    contamination, _summary = module.contamination_scope()
    registry, table = module.relabeling_registry(contamination)
    assert registry["relabeling_applied_to_existing_artifacts"] is False
    assert registry["relabeling_registry_only"] is True
    assert registry["future_publication_must_use_corrected_terms"] is True
    assert "30-minute rollout" in set(table["original_term"])


def test_adjudicate_artifact_root_must_be_absolute(tmp_path):
    module = load_module()
    try:
        module.validate_artifact_root(Path("relative/path"), "adjudicate")
    except ValueError as exc:
        assert "--artifact-root must be an absolute path" in str(exc)
    else:
        raise AssertionError("relative artifact root must fail")

    root = tmp_path / "pa1a"
    resolved = module.validate_artifact_root(root, "adjudicate")
    assert resolved == root
    assert root.exists()


def test_repository_stub_sweep_covers_dl6b_and_engine_internals():
    module = load_module()
    table, audit, engine_stub = module.repository_stub_signature_sweep()
    assert audit["repository_wide_stub_sweep_added_by_user_revision"] is True
    assert audit["dl6b_signature_count"] > 0
    assert audit["transition_engine_modulo_or_stable_int_signature_count"] == 0
    assert engine_stub["engine_stable_int_or_modulo_5_found"] is False
    assert "TRANSITION_ENGINE" in set(table["target_role"])


def test_engine_static_api_action_and_horizon_contracts():
    module = load_module()
    engine = module.transition_engine_api_audit()
    assert engine["transition_function_name"] == "advance_vehicle_time_budget"
    assert engine["transition_function_execution_count"] == 0
    assert engine["reset_function_execution_count"] == 0
    action_table, action = module.action_mapping_audit()
    assert action["action_mapping"] == "READY"
    assert set(action_table["dl6c_action_name"]) == {
        "HOLD_CURRENT_POSITION",
        "SERVE_AND_MOVE_TO_NEXT_STOP",
        "CONDITIONAL_SKIP_EMPTY_STOP",
    }
    horizon = module.horizon_contract()
    assert horizon["decision_interval_value"] == 60.0
    assert horizon["total_steps"] == 30
    assert horizon["extra_30m_after_pulse_forbidden"] is True


def test_engine_gap_gate_is_partial_not_ready():
    module = load_module()
    _cap_table, axes = module.capability_inventory()
    assert axes["transition_api"] == "READY"
    assert axes["action_mapping"] == "READY"
    assert axes["branch_state_isolation"] == "PARTIAL"
    assert axes["exogenous_interface"] == "PARTIAL"
    assert axes["reward_kpi_extraction"] == "PARTIAL"
    gap_table, gaps = module.gap_repair_registry(axes)
    assert gaps["engine_repair_required"] is True
    assert gap_table["must_complete_before_state_feasibility"].sum() >= 1
