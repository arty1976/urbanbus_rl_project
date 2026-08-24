"""Pure helper tests for the BT8-R5 frozen review."""

from __future__ import annotations

import sys

sys.path.insert(0, "05_training")

import run_h4m_ae_ls3_bt8_r5_same_support_review as R5


def test_tie_change_classification_is_fail_conservative() -> None:
    base = {"meaningfully_distinct_multi_candidate_states": 2, "exact_ties": 1}
    assert R5.discrimination_change(base, {**base, "exact_ties": 0}) == "IMPROVED"
    assert R5.discrimination_change(base, {**base, "exact_ties": 2}) == "WORSENED"
    assert R5.discrimination_change(base, dict(base)) == "UNCHANGED"


def test_no_meaningful_support_is_insufficient() -> None:
    assert R5.discrimination_change({"meaningfully_distinct_multi_candidate_states": 0, "exact_ties": 0},
                                    {"meaningfully_distinct_multi_candidate_states": 0, "exact_ties": 0}) == "INSUFFICIENT_REVIEW_SAMPLES"


def test_quantiles_do_not_invent_empty_margin_values() -> None:
    assert R5.quantiles([]) == {"min": None, "median": None, "mean": None, "max": None}
