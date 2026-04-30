from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "05_training" / "run_causal_simulator_v2_canonical_kpi_smoke.py"
CONTRACT_DOC = ROOT / "05_training" / "adapters" / "causal_simulator_v2_canonical_kpi_smoke_contract.md"


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def make_window_rollup_fixture(input_root: Path) -> None:
    seed_dir = input_root / "C2_STATIC_SIGNAL" / "rollouts" / "seed_101"
    seed_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([
        {
            "condition_id": "C2_STATIC_SIGNAL",
            "seed": 101,
            "window_id": "C2_STATIC_SIGNAL_seed101_w000",
            "state_ts": "step_0000",
            "service_date": "2026-01-01",
            "time_band": "offpeak",
            "evaluation_horizon_minutes": 30,
            "headway_mean_seconds": 480.0,
            "headway_std_seconds": 60.0,
            "headway_sample_count": 5,
            "bunching_event_count": 1,
            "headway_event_count": 4,
            "wait_total_passenger_seconds": 2400.0,
            "wait_passenger_count": 10,
            "ontime_event_count": 3,
            "schedulable_arrival_count": 4,
            "intervention_count": 8,
            "decision_step_count": 32,
            "energy_proxy_total": 88.0,
            "source_mode": "static_signal_causal_v2_scaffold_smoke_nonperformance_v1",
            "qwen_trigger_rate": 0.0,
            "effective_replay_step_minutes": 60.0,
            "performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "dynamic_signal_phase_claim_allowed": False,
        }
    ])
    df.to_parquet(seed_dir / "window_rollup.parquet", index=False)


def test_doc_exists() -> None:
    assert_true(CONTRACT_DOC.exists(), "contract doc missing")
    text = CONTRACT_DOC.read_text(encoding="utf-8-sig")
    for phrase in [
        "canonical_kpi_aggregator.py",
        "window_rollup.parquet",
        "kpi_by_window.parquet",
        "causal_comparison_allowed",
        "Step 104",
    ]:
        assert_true(phrase in text, f"missing phrase: {phrase}")


def test_canonical_kpi_smoke() -> None:
    input_root = ROOT / "artifacts" / "step103_selftest_input"
    output_root = ROOT / "artifacts" / "step103_selftest_output"
    make_window_rollup_fixture(input_root)

    cmd = [
        sys.executable,
        str(RUNNER),
        "--project-root",
        str(ROOT),
        "--input-root",
        str(input_root),
        "--output-root",
        str(output_root),
    ]

    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("Step 103 canonical KPI smoke failed")

    manifest_path = output_root / "canonical_kpi_smoke_manifest.json"
    kpi_window_path = output_root / "canonical_eval" / "kpi_by_window.parquet"
    kpi_seed_path = output_root / "canonical_eval" / "kpi_by_seed.parquet"
    kpi_overall_path = output_root / "canonical_eval" / "kpi_overall.json"
    scenario_index_path = output_root / "scenario_index.parquet"

    for path in [manifest_path, kpi_window_path, kpi_seed_path, kpi_overall_path, scenario_index_path]:
        assert_true(path.exists(), f"missing output: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert_true(manifest["aggregation_summary"]["row_counts"]["kpi_by_window"] == 1, "bad kpi_by_window row count")
    assert_true(manifest["aggregation_summary"]["causal_comparison_allowed"] is False, "causal comparison should be false")
    assert_true(manifest["claim_guardrails"]["performance_claim_allowed"] is False, "performance claim guardrail broken")

    kpi_window = pd.read_parquet(kpi_window_path)
    assert_true(kpi_window["strict_canonical"].all(), "strict_canonical should be true")
    assert_true(not kpi_window["causal_comparison_allowed"].any(), "causal comparison should be false")
    assert_true("avg_wait_seconds" in kpi_window.columns, "avg_wait_seconds missing")
    assert_true(float(kpi_window["avg_wait_seconds"].iloc[0]) == 240.0, "avg wait computation mismatch")

    # Prepared input should have parseable state_ts.
    prepared_rollup = next((output_root / "prepared_input").rglob("window_rollup.parquet"))
    prep_df = pd.read_parquet(prepared_rollup)
    pd.to_datetime(prep_df["state_ts"], errors="raise")


def main() -> None:
    test_doc_exists()
    test_canonical_kpi_smoke()
    print("[OK] Step 103 causal-v2 canonical KPI smoke self-test PASS")


if __name__ == "__main__":
    main()
