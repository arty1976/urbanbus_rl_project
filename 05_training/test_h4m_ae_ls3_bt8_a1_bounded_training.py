"""Focused source-level lock for the BT8-A1 execution envelope."""

import run_h4m_ae_ls3_bt8_a1_bounded_training as a1


def test_a1_frozen_envelope_is_exact_and_complete() -> None:
    envelope, windows, selection = a1.load_a1_envelope()
    checks = a1.validate_a1_envelope(envelope, windows)
    assert all(checks.values()), checks
    assert selection["selected_label"] == "A1_small_extension"
    assert selection["selection_authorization_at_design_time"] == "NOT_GRANTED"
    assert selection["must_not_execute_automatically_at_design_time"] is True
    assert a1.A1_EXECUTOR_V1_SOURCE == "1e3a394fa436ca23c56dc94a24873731c2a4551e"
