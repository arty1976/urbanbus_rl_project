"""Read-only helper and source-scope tests for BT8-R11."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection as R11  # noqa: E402


def test_zero_eligibility_policy_rejects_trained_policy_interpretation() -> None:
    policy = R11.zero_eligibility_policy({"F1_R1": 9, "F1_R2": 0})
    assert policy["selected_option"] == "V1"
    assert policy["zero_eligible_replicates"] == ["F1_R2"]
    assert "ACTOR_NOT_TRAINED_INSUFFICIENT_CREDIT" in policy["actor_not_trained_rule"]


def test_s3_is_selected_only_when_all_confirmed_factors_need_control() -> None:
    checkpoints = {"F1_R1": "a", "F1_R2": "b"}
    selected = R11.select_repair(actor_classification="ACTOR_INIT_DOMINANT", critic_material=True, zero_eligible=True,
                                 initial_checkpoints=checkpoints)
    assert selected["selected_contract"]["selected_option"] == "S3"
    assert selected["selected_contract"]["execution_authorized"] is False
    with pytest.raises(R11.R11Error, match="BLOCKED_SEED_REPAIR_SELECTION_UNRESOLVED"):
        R11.select_repair(actor_classification="MIXED", critic_material=True, zero_eligible=True, initial_checkpoints=checkpoints)


def test_actor_support_factorization_detects_clean_actor_init_dominance() -> None:
    def row(actor: str, support: str, window: str, family: str, gap: float) -> dict[str, object]:
        return {"actor_label": actor, "support_label": support, "window_id": window, "candidate_support_digest": support,
                "action_family": family, "no_assign_minus_best_pair": gap, "candidate_only_top1_top2_margin": 0.1,
                "entropy": 0.5, "feasible_no_assign": family == "NO_ASSIGN", "exact_tie": False}
    matrix = {}
    for actor, family, gap in (("A", "CANDIDATE", -.2), ("B", "NO_ASSIGN", .2)):
        for support in ("R1", "R2"):
            matrix[f"{actor}×{support}"] = [row(actor, support, window, family, gap) for window in ("w0", "w1", "w2")]
    result = R11.actor_support_factorization(matrix)
    assert result["classification"] == "ACTOR_INIT_DOMINANT"
    assert result["actor_effect_same_support"]["action_family_mismatch_count"] == 6
    assert result["support_effect_same_actor"]["action_family_mismatch_count"] == 0


def test_source_has_no_training_or_transition_execution_path() -> None:
    source = (ROOT / "run_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection.py").read_text(encoding="utf-8")
    for forbidden in ("optimizer.step(", ".backward(", "require_capability(", "adapter.step(", "factory.build(",
                      "candidate_plan_authoritative", "compute_reward_v2("):
        assert forbidden not in source
    assert R11.SOURCE_FILES == {
        "05_training/run_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection.py",
        "05_training/test_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection.py",
    }


def test_r6_artifact_path_is_not_shadowed_by_the_r6_audit_module() -> None:
    source = (ROOT / "run_h4m_ae_ls3_bt8_r11_seed_divergence_repair_selection.py").read_text(encoding="utf-8")
    assert "as R6MOD" in source
    assert "R6MOD.frozen_hashes(BT6)" in source
    assert "R6 / \"bt8r6_replicate1_credit_trace.json\"" in source
