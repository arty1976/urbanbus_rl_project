from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_DIR = PROJECT_ROOT / "05_training" / "adapters"
BUILDER = ADAPTERS_DIR / "build_signal_features_for_causal_simulator_v2.py"
CONTRACT_MD = ADAPTERS_DIR / "signal_feature_builder_contract_v2.md"

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


def make_fixture(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)

    signal_csv = root / "toy_signal.csv"
    signal_df = pd.DataFrame([
        {"signal_id": "S1", "signal_type": "차량신호", "latitude": 35.8700, "longitude": 128.6000},
        {"signal_id": "S2", "signal_type": "보행신호", "latitude": 35.8705, "longitude": 128.6005},
        {"signal_id": "S3", "signal_type": "점멸신호", "latitude": 35.8750, "longitude": 128.6050},
    ])
    signal_df.to_csv(signal_csv, index=False, encoding="utf-8-sig")

    node_path = root / "toy_nodes.parquet"
    node_df = pd.DataFrame([
        {"node_uid": "STOP:1", "node_index": 0, "latitude": 35.8700, "longitude": 128.6000},
        {"node_uid": "STOP:2", "node_index": 1, "latitude": 35.8706, "longitude": 128.6006},
        {"node_uid": "STOP:3", "node_index": 2, "latitude": 35.8800, "longitude": 128.6100},
    ])
    node_df.to_parquet(node_path, index=False)

    edge_path = root / "toy_edges.parquet"
    edge_df = pd.DataFrame([
        {"src_idx": 0, "dst_idx": 1, "distance_m": 120.0},
        {"src_idx": 1, "dst_idx": 2, "distance_m": 1400.0},
    ])
    edge_df.to_parquet(edge_path, index=False)

    return {
        "signal_csv": signal_csv,
        "node_path": node_path,
        "edge_path": edge_path,
        "output_dir": root / "out",
    }


def test_files_exist() -> None:
    assert_true(BUILDER.exists(), f"missing: {BUILDER}")
    assert_true(CONTRACT_MD.exists(), f"missing: {CONTRACT_MD}")


def test_contract_guardrails() -> None:
    text = CONTRACT_MD.read_text(encoding="utf-8-sig")
    for phrase in [
        "red_light_delay_seconds",
        "green_time_seconds",
        "cycle_length_seconds",
        "정적 신호 인프라 feature",
        "performance_claim_allowed=false",
    ]:
        assert_true(phrase in text, f"missing contract phrase: {phrase}")


def test_builder_smoke() -> None:
    fixture_root = PROJECT_ROOT / "artifacts" / "step98_selftest"
    fx = make_fixture(fixture_root)

    cmd = [
        sys.executable,
        str(BUILDER),
        "--signal-csv",
        str(fx["signal_csv"]),
        "--node-source",
        str(fx["node_path"]),
        "--edge-source",
        str(fx["edge_path"]),
        "--output-dir",
        str(fx["output_dir"]),
    ]

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("Step 98 builder smoke failed")

    node_out = fx["output_dir"] / "node_signal_features.parquet"
    edge_out = fx["output_dir"] / "edge_signal_features.parquet"
    contract_out = fx["output_dir"] / "tensor_signal_feature_contract_v2.json"
    quality_out = fx["output_dir"] / "signal_feature_quality_report.json"

    for p in [node_out, edge_out, contract_out, quality_out]:
        assert_true(p.exists(), f"missing output: {p}")

    node_df = pd.read_parquet(node_out)
    edge_df = pd.read_parquet(edge_out)

    required_node_cols = [
        "node_uid",
        "node_index",
        "signal_count_250m",
        "nearest_signal_distance_m",
        "pedestrian_signal_count_250m",
        "blink_signal_ratio_250m",
        "controlled_signal_ratio_250m",
        "signal_delay_risk_proxy",
        "intersection_complexity_proxy",
        "signal_feature_quality_flag",
    ]
    for c in required_node_cols:
        assert_true(c in node_df.columns, f"node output missing column: {c}")

    required_edge_cols = [
        "src_idx",
        "dst_idx",
        "distance_m",
        "edge_signal_count",
        "edge_signal_density_per_km",
        "edge_nearest_signal_distance_m",
        "edge_control_complexity_proxy",
        "edge_signal_feature_quality_flag",
    ]
    for c in required_edge_cols:
        assert_true(c in edge_df.columns, f"edge output missing column: {c}")

    for forbidden in FORBIDDEN:
        assert_true(forbidden not in node_df.columns, f"forbidden node column present: {forbidden}")
        assert_true(forbidden not in edge_df.columns, f"forbidden edge column present: {forbidden}")

    contract = json.loads(contract_out.read_text(encoding="utf-8"))
    assert_true(contract["artifact_version"] == "tensor_signal_feature_contract_v2", "bad contract version")
    assert_true(contract["claim_guardrails"]["performance_claim_allowed"] is False, "performance claim guardrail broken")
    assert_true(contract["claim_guardrails"]["red_light_delay_claim_allowed"] is False, "red-light delay claim guardrail broken")

    quality = json.loads(quality_out.read_text(encoding="utf-8"))
    assert_true(quality["step_98_ready"] is True, "step_98_ready must be true")
    assert_true(quality["forbidden_dynamic_signal_fields_absent"] is True, "forbidden fields flag must be true")


def main() -> None:
    test_files_exist()
    test_contract_guardrails()
    test_builder_smoke()
    print("[OK] Step 98 signal feature builder self-test PASS")


if __name__ == "__main__":
    main()
