from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_DIR = PROJECT_ROOT / "05_training" / "adapters"
BUILDER = ADAPTERS_DIR / "build_causal_simulator_v2_contract_from_signal_features.py"
CONTRACT_MD = ADAPTERS_DIR / "causal_simulator_v2_input_contract.md"
INTEGRATION_MD = ADAPTERS_DIR / "tensor_signal_availability_integration_v2.md"

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


def make_fixture(root: Path) -> tuple[Path, Path]:
    signal_dir = root / "signal_features_v2"
    preflight_dir = root / "signal_features_v2_preflight"
    signal_dir.mkdir(parents=True, exist_ok=True)
    preflight_dir.mkdir(parents=True, exist_ok=True)

    node_df = pd.DataFrame([
        {
            "node_uid": "STOP:1",
            "node_index": 0,
            "signal_count_100m": 1,
            "signal_count_250m": 2,
            "signal_count_500m": 3,
            "nearest_signal_distance_m": 12.0,
            "pedestrian_signal_count_250m": 1,
            "blink_signal_ratio_250m": 0.5,
            "controlled_signal_ratio_250m": 1.0,
            "signal_delay_risk_proxy": 1.2,
            "intersection_complexity_proxy": 2.3,
            "signal_feature_quality_flag": "ok",
        },
        {
            "node_uid": "STOP:2",
            "node_index": 1,
            "signal_count_100m": 0,
            "signal_count_250m": 1,
            "signal_count_500m": 2,
            "nearest_signal_distance_m": 90.0,
            "pedestrian_signal_count_250m": 0,
            "blink_signal_ratio_250m": 0.0,
            "controlled_signal_ratio_250m": 1.0,
            "signal_delay_risk_proxy": 0.8,
            "intersection_complexity_proxy": 1.1,
            "signal_feature_quality_flag": "ok",
        },
    ])
    edge_df = pd.DataFrame([
        {
            "src_idx": 0,
            "dst_idx": 1,
            "distance_m": 120.0,
            "edge_signal_count": 3,
            "edge_signal_density_per_km": 25.0,
            "edge_nearest_signal_distance_m": 12.0,
            "edge_control_complexity_proxy": 1.7,
            "edge_signal_feature_quality_flag": "ok",
        }
    ])

    node_df.to_parquet(signal_dir / "node_signal_features.parquet", index=False)
    edge_df.to_parquet(signal_dir / "edge_signal_features.parquet", index=False)

    (signal_dir / "tensor_signal_feature_contract_v2.json").write_text(
        json.dumps({"artifact_version": "tensor_signal_feature_contract_v2"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (signal_dir / "signal_feature_quality_report.json").write_text(
        json.dumps({"artifact_version": "signal_feature_quality_report_v1", "step_98_ready": True}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (preflight_dir / "daegu_signal_csv_preflight_report.json").write_text(
        json.dumps({"artifact_version": "daegu_signal_csv_preflight_report_v2", "step_99_ready": True}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return signal_dir, preflight_dir


def test_files_exist() -> None:
    assert_true(BUILDER.exists(), f"missing: {BUILDER}")
    assert_true(CONTRACT_MD.exists(), f"missing: {CONTRACT_MD}")
    assert_true(INTEGRATION_MD.exists(), f"missing: {INTEGRATION_MD}")


def test_docs_have_guardrails() -> None:
    text = CONTRACT_MD.read_text(encoding="utf-8-sig") + "\n" + INTEGRATION_MD.read_text(encoding="utf-8-sig")
    for phrase in [
        "red_light_delay_seconds",
        "green_time_seconds",
        "cycle_length_seconds",
        "performance_claim_allowed",
        "signal_delay_risk_proxy",
        "CausalSimulatorAdapter v2",
    ]:
        assert_true(phrase in text, f"missing phrase: {phrase}")


def test_builder_smoke() -> None:
    fixture_root = PROJECT_ROOT / "artifacts" / "step100_selftest"
    signal_dir, preflight_dir = make_fixture(fixture_root)
    output_dir = fixture_root / "out"

    cmd = [
        sys.executable,
        str(BUILDER),
        "--signal-features-dir",
        str(signal_dir),
        "--preflight-dir",
        str(preflight_dir),
        "--output-dir",
        str(output_dir),
    ]

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("Step 100 builder smoke failed")

    contract_path = output_dir / "causal_simulator_v2_input_contract.json"
    manifest_path = output_dir / "causal_simulator_v2_feature_manifest.json"
    report_path = output_dir / "causal_simulator_v2_contract_report.md"

    for p in [contract_path, manifest_path, report_path]:
        assert_true(p.exists(), f"missing output: {p}")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert_true(contract["step"] == 100, "bad step")
    assert_true(contract["row_counts"]["node_signal_features"] == 2, "bad node row count")
    assert_true(contract["row_counts"]["edge_signal_features"] == 1, "bad edge row count")
    assert_true(contract["claim_guardrails"]["performance_claim_allowed"] is False, "performance claim guardrail broken")
    assert_true(contract["claim_guardrails"]["dynamic_signal_phase_claim_allowed"] is False, "dynamic phase guardrail broken")

    for forbidden in FORBIDDEN:
        assert_true(forbidden in manifest["forbidden_dynamic_signal_fields"], f"missing forbidden field: {forbidden}")

    assert_true("signal_delay_risk_proxy" in manifest["proxy_columns"], "proxy column missing")
    assert_true(manifest["admission_summary"]["dynamic_signal_phase_features"] == "rejected", "dynamic admission must be rejected")


def main() -> None:
    test_files_exist()
    test_docs_have_guardrails()
    test_builder_smoke()
    print("[OK] Step 100 causal simulator v2 contract self-test PASS")


if __name__ == "__main__":
    main()
