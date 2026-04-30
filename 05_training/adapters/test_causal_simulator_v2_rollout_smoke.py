from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "05_training" / "run_causal_simulator_v2_rollout_smoke.py"
CONTRACT_MD = ROOT / "05_training" / "adapters" / "causal_simulator_v2_rollout_writer_contract.md"

FORBIDDEN = {
    "red_light_delay_seconds",
    "green_time_seconds",
    "cycle_length_seconds",
    "phase_sequence",
    "signal_offset_seconds",
    "real_time_signal_state",
    "queue_discharge_rate",
}


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def make_fixture(root: Path) -> tuple[Path, Path, Path]:
    signal_dir = root / "signal_features_v2"
    contract_dir = root / "causal_simulator_v2_contract"
    signal_dir.mkdir(parents=True, exist_ok=True)
    contract_dir.mkdir(parents=True, exist_ok=True)

    node_df = pd.DataFrame([
        {"node_uid":"STOP:1","node_index":0,"signal_count_250m":2,"nearest_signal_distance_m":12.0,"pedestrian_signal_count_250m":1,"blink_signal_ratio_250m":0.5,"controlled_signal_ratio_250m":1.0,"signal_delay_risk_proxy":1.2,"intersection_complexity_proxy":2.3},
        {"node_uid":"STOP:2","node_index":1,"signal_count_250m":1,"nearest_signal_distance_m":90.0,"pedestrian_signal_count_250m":0,"blink_signal_ratio_250m":0.0,"controlled_signal_ratio_250m":1.0,"signal_delay_risk_proxy":0.8,"intersection_complexity_proxy":1.1},
        {"node_uid":"STOP:3","node_index":2,"signal_count_250m":4,"nearest_signal_distance_m":5.0,"pedestrian_signal_count_250m":2,"blink_signal_ratio_250m":0.25,"controlled_signal_ratio_250m":1.0,"signal_delay_risk_proxy":1.8,"intersection_complexity_proxy":3.2},
    ])
    edge_df = pd.DataFrame([
        {"src_idx":0,"dst_idx":1,"distance_m":120.0,"edge_signal_count":3,"edge_signal_density_per_km":25.0,"edge_nearest_signal_distance_m":12.0,"edge_control_complexity_proxy":1.7},
        {"src_idx":1,"dst_idx":2,"distance_m":200.0,"edge_signal_count":5,"edge_signal_density_per_km":25.0,"edge_nearest_signal_distance_m":5.0,"edge_control_complexity_proxy":2.4},
    ])

    node_path = signal_dir / "node_signal_features.parquet"
    edge_path = signal_dir / "edge_signal_features.parquet"
    contract_path = contract_dir / "causal_simulator_v2_input_contract.json"

    node_df.to_parquet(node_path, index=False)
    edge_df.to_parquet(edge_path, index=False)

    contract = {
        "artifact_version": "causal_simulator_v2_input_contract_draft_step100",
        "step": 100,
        "feature_classes": {"dynamic_signal_phase_features": {"admission": "rejected", "columns": sorted(FORBIDDEN)}},
        "claim_guardrails": {
            "trained_model": False,
            "performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "dynamic_signal_phase_claim_allowed": False,
            "red_light_delay_claim_allowed": False,
            "green_time_claim_allowed": False,
            "cycle_length_claim_allowed": False,
        },
    }
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    return contract_path, node_path, edge_path


def test_contract_doc() -> None:
    assert_true(CONTRACT_MD.exists(), "contract doc missing")
    text = CONTRACT_MD.read_text(encoding="utf-8-sig")
    for phrase in [
        "raw_events.parquet",
        "window_rollup.parquet",
        "performance_claim_allowed = false",
        "red_light_delay_seconds",
        "Step 103",
    ]:
        assert_true(phrase in text, f"missing phrase: {phrase}")


def test_rollout_smoke_writer() -> None:
    contract, node, edge = make_fixture(ROOT / "artifacts" / "step102_selftest_input")
    output_root = ROOT / "artifacts" / "step102_selftest_output"

    cmd = [
        sys.executable,
        str(RUNNER),
        "--project-root",
        str(ROOT),
        "--conditions",
        "C2_STATIC_SIGNAL,C2_NOOP",
        "--seeds",
        "1,2",
        "--num-agents",
        "3",
        "--episode-steps",
        "4",
        "--baseline-bus-count",
        "4",
        "--output-root",
        str(output_root),
        "--contract-path",
        str(contract),
        "--node-signal-features",
        str(node),
        "--edge-signal-features",
        str(edge),
    ]

    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("Step 102 rollout smoke writer failed")

    manifest_path = output_root / "rollout_manifest.json"
    assert_true(manifest_path.exists(), "manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert_true(manifest["run_count"] == 4, "bad run_count")
    assert_true(manifest["claim_guardrails"]["performance_claim_allowed"] is False, "claim guardrail broken")

    raw_total = 0
    rollup_total = 0
    for run in manifest["runs"]:
        raw_path = Path(run["raw_events"])
        rollup_path = Path(run["window_rollup"])
        assert_true(raw_path.exists(), f"raw_events missing: {raw_path}")
        assert_true(rollup_path.exists(), f"window_rollup missing: {rollup_path}")

        raw_df = pd.read_parquet(raw_path)
        rollup_df = pd.read_parquet(rollup_path)
        raw_total += len(raw_df)
        rollup_total += len(rollup_df)

        for forbidden in FORBIDDEN:
            assert_true(forbidden not in raw_df.columns, f"forbidden raw column: {forbidden}")
            assert_true(forbidden not in rollup_df.columns, f"forbidden rollup column: {forbidden}")

        assert_true(bool(rollup_df["performance_claim_allowed"].any()) is False, "performance claim should remain false")
        assert_true(bool(rollup_df["dynamic_signal_phase_claim_allowed"].any()) is False, "phase claim should remain false")
        assert_true("source_mode" in rollup_df.columns, "source_mode missing")
        assert_true("smoke" in str(rollup_df["source_mode"].iloc[0]), "source_mode must include smoke")

    assert_true(raw_total == 48, f"bad raw_total: {raw_total}")
    assert_true(rollup_total == 4, f"bad rollup_total: {rollup_total}")


def main() -> None:
    test_contract_doc()
    test_rollout_smoke_writer()
    print("[OK] Step 102 CausalSimulatorAdapter v2 rollout writer smoke self-test PASS")


if __name__ == "__main__":
    main()
