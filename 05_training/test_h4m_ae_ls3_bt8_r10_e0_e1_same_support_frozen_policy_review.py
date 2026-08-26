"""Read-only helper and scope tests for BT8-R10."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r10_e0_e1_same_support_frozen_policy_review as R10  # noqa: E402


def _row(*, window: str, family: str, no_assign_gap: float, candidate_gap: float | None, entropy: float) -> dict[str, object]:
    return {
        "window_id": window,
        "decision_id": f"decision-{window}",
        "snapshot_digest": f"snap-{window}",
        "input_identity": {"candidate_support_digest": f"support-{window}"},
        "selected_action_family": family,
        "selected_identity": "NO_ASSIGN" if family == "NO_ASSIGN" else ("A", "C"),
        "feasible_no_assign": family == "NO_ASSIGN",
        "exact_tie": False,
        "no_assign_minus_best_pair": no_assign_gap,
        "candidate_only_top1_top2_margin": candidate_gap,
        "entropy": entropy,
        "pair_logits": [0.2],
        "no_assign_logit": 0.2 + no_assign_gap,
        "probabilities": [0.5, 0.5],
    }


def test_seed_divergence_requires_uniform_evidence_for_reduced_label() -> None:
    e0 = {"same_window_rows": 3, "action_family_mismatch_count": 3,
          "mean_abs_no_assign_minus_best_pair_gap": 3.0, "mean_abs_candidate_margin_gap": 2.0,
          "mean_abs_entropy_gap": 1.0}
    reduced = {**e0, "action_family_mismatch_count": 2, "mean_abs_no_assign_minus_best_pair_gap": 2.0,
               "mean_abs_candidate_margin_gap": 1.0, "mean_abs_entropy_gap": 0.5}
    mixed = {**e0, "action_family_mismatch_count": 2, "mean_abs_no_assign_minus_best_pair_gap": 4.0}
    assert R10.seed_divergence_classification(e0, reduced) == "SEED_DIVERGENCE_REDUCED"
    assert R10.seed_divergence_classification(e0, mixed) == "SEED_DIVERGENCE_PERSISTS"


def test_seed_view_does_not_compare_candidate_identity_on_different_support() -> None:
    r1 = [_row(window="w0", family="ASSIGN", no_assign_gap=-.2, candidate_gap=.1, entropy=.5)]
    r2 = [_row(window="w0", family="NO_ASSIGN", no_assign_gap=.2, candidate_gap=.2, entropy=.6)]
    r2[0]["input_identity"] = {"candidate_support_digest": "other"}
    audit = R10.seed_view(r1, r2)
    assert audit["rows"][0]["selected_identity_equal_when_comparable"] is None
    assert audit["rows"][0]["candidate_identity_comparison"] == "EXACT_ONLY_IF_SUPPORT_EQUAL"


def test_r2_immutability_requires_state_logits_probabilities_and_selection_exact() -> None:
    base = _row(window="w0", family="ASSIGN", no_assign_gap=-.1, candidate_gap=.2, entropy=.5)
    base |= {"pair_logits": [.2], "no_assign_logit": .1, "probabilities": [.52, .48]}
    e0 = {**base, "selected_identity": "NO_ASSIGN", "selected_action_family": "NO_ASSIGN", "pair_logits": [.0], "no_assign_logit": .3,
          "probabilities": [.43, .57]}
    good = R10.r2_immutability(initial_rows=[base], e0_rows=[e0], e1_rows=[dict(base)], initial_actor_sha="same",
                                e0_actor_sha="different", e1_actor_sha="same", state_tensors_exact=True)
    assert good["initial_e1_immutability"]["passed"] is True
    changed = R10.r2_immutability(initial_rows=[base], e0_rows=[e0], e1_rows=[{**base, "probabilities": [.51, .49]}], initial_actor_sha="same",
                                   e0_actor_sha="different", e1_actor_sha="same", state_tensors_exact=True)
    assert changed["initial_e1_immutability"]["passed"] is False


def test_r1_summary_reports_no_assign_change_without_quality_claim() -> None:
    e0 = [_row(window="w0", family="NO_ASSIGN", no_assign_gap=.3, candidate_gap=.1, entropy=.5)]
    e1 = [_row(window="w0", family="ASSIGN", no_assign_gap=-.2, candidate_gap=.2, entropy=.6)]
    initial = [dict(e1[0])]
    review = R10.r1_review({"initial": initial, "e0_final": e0, "e1_final": e1})
    assert review["e0_to_e1"]["feasible_no_assign"]["direction"] == "DECREASED"
    assert review["no_performance_interpretation"] is True


def test_source_scope_has_no_training_or_transition_execution_path() -> None:
    source = (ROOT / "run_h4m_ae_ls3_bt8_r10_e0_e1_same_support_frozen_policy_review.py").read_text(encoding="utf-8")
    for forbidden in ("optimizer.step(", ".backward(", "require_capability(", "adapter.step(", "factory.build(",
                      "candidate_plan_authoritative", "compute_reward_v2("):
        assert forbidden not in source
    assert R10.SOURCE_FILES == {
        "05_training/run_h4m_ae_ls3_bt8_r10_e0_e1_same_support_frozen_policy_review.py",
        "05_training/test_h4m_ae_ls3_bt8_r10_e0_e1_same_support_frozen_policy_review.py",
    }
