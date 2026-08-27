"""Focused non-executing guards for the R18-R4 frozen policy review."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_h4m_ae_ls3_bt8_r18_r4_frozen_policy_initial_final_review as R4  # noqa: E402


def _row(*, snapshot: str = "snap", support: str = "support", ids: list[str] | None = None,
         logits: list[float] | None = None, probs: list[float] | None = None, state: str = "initial") -> dict[str, object]:
    identities = ids or ["a::one", "b::two"]
    pair_logits = logits or [1.0, 0.5]
    probabilities = probs or [0.5, 0.3, 0.2]
    return {
        "arm_id": "AC_CONTROL_R1",
        "checkpoint_state": state,
        "decision_id": "d0",
        "window_id": "w0",
        "time_band": "offpeak",
        "environment_seed": 20260822,
        "snapshot_digest": snapshot,
        "candidate_support_digest": support,
        "candidate_identities": identities,
        "safe_mask_sha256": "mask",
        "support_size": len(identities),
        "selected_identity": identities[0],
        "selected_action_family": "CANDIDATE",
        "best_pair_identity": identities[0],
        "best_pair_probability": probabilities[0],
        "pair_logits": pair_logits,
        "no_assign_logit": 0.0,
        "probabilities": probabilities,
        "no_assign_probability": probabilities[-1],
        "best_pair_minus_no_assign": pair_logits[0],
        "candidate_only_top1_top2_logit_margin": pair_logits[0] - pair_logits[1],
        "candidate_only_top1_top2_probability_margin": probabilities[0] - probabilities[1],
        "candidate_only_logit_range": max(pair_logits) - min(pair_logits),
        "candidate_only_probability_range": max(probabilities[:-1]) - min(probabilities[:-1]),
        "candidate_only_logit_std": 0.25,
        "candidate_rank_by_identity": R4.rank_map(pair_logits, identities),
        "entropy": 1.0,
    }


def test_source_scope_and_gate_are_frozen() -> None:
    assert R4.SOURCE_FILES == {
        "05_training/run_h4m_ae_ls3_bt8_r18_r4_frozen_policy_initial_final_review.py",
        "05_training/test_h4m_ae_ls3_bt8_r18_r4_frozen_policy_initial_final_review.py",
    }
    assert R4.PASS_GATE == "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_R4_SAME_INPUT_FROZEN_POLICY_INITIAL_FINAL_REVIEW_COMPLETE"
    assert R4.REVIEW_COLLECTION_DIGEST == "6c811022a5df4b3966ac14fce750f8bdd840a65a50e48285fe0c97ce157b889e"


def test_same_snapshot_compare_rejects_cross_digest_or_support_mutation() -> None:
    left = _row()
    right = _row(state="final")
    assert R4.compare_same_snapshot(left, right)["selection_changed"] is False

    changed_snapshot = _row(snapshot="other", state="final")
    try:
        R4.compare_same_snapshot(left, changed_snapshot)
    except R4.R18R4Error as exc:
        assert exc.code == R4.BINDING_BLOCK
    else:  # pragma: no cover
        raise AssertionError("cross snapshot compare should fail closed")

    changed_support = _row(support="other", state="final")
    try:
        R4.compare_same_snapshot(left, changed_support)
    except R4.R18R4Error as exc:
        assert exc.code == R4.BINDING_BLOCK
    else:  # pragma: no cover
        raise AssertionError("support mutation should fail closed")


def test_candidate_discrimination_deltas_are_exact_directional_counts() -> None:
    initial = _row(logits=[1.0, 0.5], probs=[0.5, 0.3, 0.2])
    final = _row(logits=[1.4, 0.4], probs=[0.6, 0.25, 0.15], state="final")
    delta = R4.compare_same_snapshot(initial, final)
    assert R4.direction(delta["candidate_margin_delta"]) == "INCREASED"
    assert R4.direction(delta["candidate_range_delta"]) == "INCREASED"
    summary = R4.delta_summary([delta])
    assert summary["candidate_margin_direction_counts"]["INCREASED"] == 1
    assert summary["candidate_range_direction_counts"]["INCREASED"] == 1
    assert summary["selection_changed_count"] == 0


def test_classification_separates_selection_change_from_distribution_shift() -> None:
    shifted = R4.delta_summary([R4.compare_same_snapshot(_row(), _row(logits=[1.1, 0.5], state="final"))])
    assert R4.classification_from({"AC_CONTROL_R1": shifted}) == "B_R18_R3_POLICY_DISTRIBUTION_SHIFT_WITH_STABLE_T1_SELECTIONS"

    initial = _row()
    final = _row(state="final")
    final["selected_identity"] = "NO_ASSIGN"
    final["selected_action_family"] = "NO_ASSIGN"
    changed = R4.delta_summary([R4.compare_same_snapshot(initial, final)])
    assert R4.classification_from({"AC_CONTROL_R1": changed}) == "A_R18_R3_POLICY_SELECTION_CHANGED_ON_FIXED_REVIEW_SNAPSHOTS"


def test_frozen_review_counters_prohibit_execution_capabilities() -> None:
    value = R4.counters()
    for key in (
        "training", "mps_training", "optimizer_creation", "optimizer_step", "backward", "loss_update",
        "causal_rollout", "simulator_execution", "candidate_generation", "candidate_regeneration",
        "local_search_rerun", "zero_loss_reevaluation", "reward_settlement", "parameter_mutation",
        "checkpoint_write", "checkpoint_mutation", "review_optimizer_rows", "future_leakage",
        "test6_access", "github_push",
    ):
        assert value[key] == 0


def test_source_contains_no_training_rollout_or_mutating_execution_path() -> None:
    source = (ROOT / "run_h4m_ae_ls3_bt8_r18_r4_frozen_policy_initial_final_review.py").read_text(encoding="utf-8")
    for forbidden in (
        "optimizer.step(", ".backward(", "require_capability(", "adapter.step(",
        "factory.build(", "compute_reward_v2(", "torch.save(",
    ):
        assert forbidden not in source
