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
