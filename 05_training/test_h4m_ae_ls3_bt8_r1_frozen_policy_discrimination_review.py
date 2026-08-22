"""Read-only fixture coverage for BT8-R1 authority and physical classification."""

import run_h4m_ae_ls3_bt8_r1_frozen_policy_discrimination_review as r1


def test_a1_and_previous_support_multisets_are_directly_comparable() -> None:
    import json
    from collections import Counter
    a1 = json.loads((r1.A1 / "bt8_a1_candidate_support_audit.json").read_text())["decisions"]
    old = json.loads((r1.BT6 / "bt6_candidate_support_audit.json").read_text())["decisions"]
    assert Counter(map(r1.support_signature, a1)) == Counter(map(r1.support_signature, old))


def test_quantiles_do_not_introduce_a_tolerance() -> None:
    assert r1.quantiles([0.0, 1.0])["min"] == 0.0
    assert r1.T1_TOLERANCE == 0.0


def test_margin_comparison_uses_only_multi_candidate_states() -> None:
    rows = [
        {"support_size": 1, "exact_tie": False, "top_score_margin": 3.0, "entropy": 0.1, "top1_probability": 0.9, "top2_probability": 0.1, "top_probability_margin": 0.8},
        {"support_size": 2, "exact_tie": False, "top_score_margin": 0.2, "entropy": 0.6, "top1_probability": 0.55, "top2_probability": 0.45, "top_probability_margin": 0.1},
    ]
    scoped = r1.multi_candidate_score_summary(rows)
    assert scoped["unique_winner_states"]["states"] == 1
    assert scoped["unique_winner_states"]["positive_top1_top2_score_margin"]["max"] == 0.2
