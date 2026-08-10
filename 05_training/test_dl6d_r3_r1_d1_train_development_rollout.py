from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
MODULE_PATH = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_r3_r1_d1_train_development_rollout.py"


def load_module():
    spec = importlib.util.spec_from_file_location("dl6d_r3_r1_d1", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dl6d_r3_r1_d1"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_artifact_root_is_required_and_absolute(tmp_path):
    module = load_module()
    relative = Path("05_training/artifacts/not_absolute")
    try:
        module.validate_artifact_root(relative, "calibrate")
    except ValueError as exc:
        assert "--artifact-root must be an absolute path" in str(exc)
    else:
        raise AssertionError("relative artifact root must fail closed")

    root = tmp_path / "d1_artifact"
    resolved = module.validate_artifact_root(root, "calibrate")
    assert resolved == root
    assert root.exists()


def test_scale_grid_is_separated_from_scale_production():
    module = load_module()
    audit = module.scale_derivation_prohibition_audit()
    assert audit["hypothetical_scale_grid_allowed_in_analyze"] is True
    assert audit["hypothetical_scale_grid_generated"] is False
    assert audit["hypothetical_scale_grid_count"] == 0
    assert audit["selected_scale_produced"] is False
    assert audit["recommended_scale_produced"] is False
    assert audit["candidate_scale_produced"] is False
    assert audit["headroom_factor_produced"] is False
    assert audit["new_scale_value_produced"] is False
    assert audit["scale_values_produced"] == 0
    assert audit["sensitivity_equation_locked"] == "post_margin(scale) = baseline_margin - scale * service_harm_excess_native"
    assert audit["tail_concentration_zero_denominator_rule"]["value"] is None


def test_prior_test_reference_is_not_reuse():
    module = load_module()
    audit = module.test_holdout_non_reuse_audit()
    assert audit["prior_test_outcome_reference_used"] is True
    assert audit["prior_test_rows_reexecuted"] is False
    assert audit["prior_test_rows_refit"] is False
    assert audit["prior_test_used_for_candidate_selection"] is False
    assert audit["prior_test_used_for_validation"] is False
    assert audit["prior_test_used_for_scale_grid"] is False
    assert audit["test_holdout_reused"] is False
    assert audit["existing_86_refit"] is False


def test_validation_access_scope_is_split_from_outcome_access():
    module = load_module()
    audit = module.validation_untouched_audit({"validation_seal_intact": True})
    assert audit["validation_inventory_hash_access_allowed"] is True
    assert audit["validation_precomputed_covariate_summary_access_allowed"] is True
    assert audit["validation_row_level_feature_access_allowed"] is False
    assert audit["validation_outcome_access_allowed"] is False
    assert audit["validation_reward_access_allowed"] is False
    assert audit["validation_harm_label_access_allowed"] is False
    assert audit["validation_30m_branch_execution_count"] == 0


def test_chunk_plan_has_250_row_calibration_chunk():
    module = load_module()
    train = module.load_train_skip_valid_sorted()
    plan = module.build_chunk_execution_plan(train)
    first = plan.iloc[0]
    assert len(train) == 5523
    assert first["chunk_id"] == "D1C0001"
    assert first["row_count"] == 250
    assert first["expected_branch_count"] == 750
    assert bool(first["is_calibration_chunk"]) is True
    assert int(plan["row_count"].sum()) == 5523
    assert int(plan["expected_branch_count"].sum()) == 16569


def test_covariate_reinterpretation_lowers_86_language():
    module = load_module()
    covariate = module.covariate_shift_reinterpretation()
    assert covariate["i0_artifact_modified"] is False
    assert covariate["decision_context_shift_classification"] == "NEGLIGIBLE"
    assert covariate["validation_capacity_highest_tier"] == "CAPACITY_50_PLAUSIBLE"
    assert covariate["validation_capacity_86_plausible"] is False
    assert covariate["train_validation_exchangeability_verified"] is False
    assert "arithmetic projection only" in covariate["validation_projection_interpretation"]


def test_ho1_gate_reference_does_not_require_manifest():
    module = load_module()
    _upstream, manifests = module.upstream_validation()
    assert manifests["artifacts"]["ho1"]["gate_reference_only"] is True
    assert manifests["artifacts"]["ho1"]["manifest_required_for_d1_gate"] is False
    assert manifests["upstream_manifest_reconciliation_passed"] is True
