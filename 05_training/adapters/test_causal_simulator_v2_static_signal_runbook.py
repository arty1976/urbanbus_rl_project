from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = PROJECT_ROOT / "05_training" / "adapters" / "causal_simulator_v2_static_signal_runbook.md"
PROJECT_LOG = PROJECT_ROOT / "project_log.md"


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_runbook_exists_and_has_guardrails() -> None:
    assert_true(RUNBOOK.exists(), f"missing runbook: {RUNBOOK}")
    text = RUNBOOK.read_text(encoding="utf-8-sig")

    required = [
        "Step 97",
        "Step 98",
        "Step 99",
        "Step 100",
        "Step 101",
        "Step 102",
        "Step 103",
        "performance_claim_allowed = false",
        "causal_performance_claim_allowed = false",
        "red_light_delay_claim_allowed = false",
        "green_time_claim_allowed = false",
        "cycle_length_claim_allowed = false",
        "signal_delay_risk_proxy",
        "canonical_kpi_aggregator.py",
        "Step 105",
    ]

    for phrase in required:
        assert_true(phrase in text, f"missing runbook phrase: {phrase}")


def test_project_log_updated() -> None:
    assert_true(PROJECT_LOG.exists(), f"missing project log: {PROJECT_LOG}")
    text = PROJECT_LOG.read_text(encoding="utf-8-sig")

    required = [
        "STEP_104_CAUSAL_SIMULATOR_V2_STATIC_SIGNAL_LOG_START",
        "Step 97~103",
        "node_rows = 4116",
        "edge_rows = 5484",
        "performance_claim_allowed = false",
        "CausalSimulatorAdapter v2",
        "canonical_kpi_aggregator.py",
        "Step 105",
    ]

    for phrase in required:
        assert_true(phrase in text, f"missing project_log phrase: {phrase}")


def main() -> None:
    test_runbook_exists_and_has_guardrails()
    test_project_log_updated()
    print("[OK] Step 104 project log and runbook self-test PASS")


if __name__ == "__main__":
    main()
