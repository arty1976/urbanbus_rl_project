"""Focused non-executing guards for R18-R5 attribution evidence handling."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r18_r5_credit_to_logit_direction_audit as R5  # noqa: E402


def test_gate_and_expected_actor_eligible_counts_are_frozen() -> None:
    assert R5.PASS_GATE == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R5_POST_UPDATE_CREDIT_TO_LOGIT_DIRECTION_ATTRIBUTION_AUDIT_COMPLETE"
    assert R5.EXPECTED_ELIGIBLE == {"AC_CONTROL_R1": 9, "BD_E1_R1": 5}
    assert R5.R18_R3_TRAINING_SOURCE == "7bd0e2e223779455e6112584abd0b5cb6441c228"
    assert R5.R18_R4_REVIEW_SOURCE == "be2683e68eea94699a16604a60d7cea3848a39f4"


def test_source_scope_is_limited_to_r18_r5_files() -> None:
    assert R5.SOURCE_FILES == {
        "05_training/run_h4m_ae_ls3_bt8_r18_r5_credit_to_logit_direction_audit.py",
        "05_training/test_h4m_ae_ls3_bt8_r18_r5_credit_to_logit_direction_audit.py",
    }


def test_row_evidence_availability_fails_closed_when_required_classes_are_missing(tmp_path: Path) -> None:
    result = R5.row_evidence_availability(tmp_path)
    assert result["row_level_trace_possible_without_forbidden_reconstruction"] is False
    assert set(result["missing_required_evidence_classes"]) == set(R5.REQUIRED_R18_R3_ROW_EVIDENCE)
    assert result["expected_actor_eligible_rows"]["total"] == 14


def test_row_evidence_availability_passes_only_when_every_class_has_a_file(tmp_path: Path) -> None:
    for alternatives in R5.REQUIRED_R18_R3_ROW_EVIDENCE.values():
        (tmp_path / alternatives[0]).write_text("placeholder", encoding="utf-8")
    result = R5.row_evidence_availability(tmp_path)
    assert result["row_level_trace_possible_without_forbidden_reconstruction"] is True
    assert result["missing_required_evidence_classes"] == []


def test_review_counters_prohibit_training_and_mutating_paths() -> None:
    value = R5.counters()
    for key in (
        "training", "mps_training", "rollout", "simulator_execution", "candidate_generation",
        "reward_recomputation", "optimizer_creation", "optimizer_step", "backward",
        "checkpoint_write", "policy_mutation", "test6_access", "github_push",
        "checkpoint_mutation",
    ):
        assert value[key] == 0


def test_source_contains_no_forbidden_execution_calls() -> None:
    source = (ROOT / "run_h4m_ae_ls3_bt8_r18_r5_credit_to_logit_direction_audit.py").read_text(encoding="utf-8")
    for forbidden in (
        "optimizer.step(", ".backward(", "require_capability(", "adapter.step(",
        "factory.build(", "compute_reward_v2(", "torch.save(",
    ):
        assert forbidden not in source
