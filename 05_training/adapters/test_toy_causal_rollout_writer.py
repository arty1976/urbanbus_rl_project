from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "05_training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


REQUIRED_RAW_EVENT_COLUMNS = {
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "time_band",
    "agent_id",
    "action",
    "action_name",
    "position_before",
    "position_after",
    "waiting_passenger_cnt",
    "passenger_served_count",
    "distance_m",
    "hold_seconds",
    "acceleration_event_count",
    "energy_proxy_total",
    "intervention_applied",
    "terminated",
    "truncated",
    "source_mode",
    "causal_comparison_allowed",
}

REQUIRED_WINDOW_ROLLUP_COLUMNS = {
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "service_date",
    "time_band",
    "evaluation_horizon_minutes",
    "qwen_trigger_rate",
    "effective_replay_step_minutes",
    "headway_mean_seconds",
    "headway_std_seconds",
    "headway_sample_count",
    "bunching_event_count",
    "headway_event_count",
    "wait_total_passenger_seconds",
    "wait_passenger_count",
    "ontime_event_count",
    "schedulable_arrival_count",
    "intervention_count",
    "decision_step_count",
    "energy_proxy_total",
    "source_mode",
    "control_step_minutes",
    "causal_comparison_allowed",
    "simulator_adapter_version",
    "graph_scope",
    "num_agents",
    "num_nodes",
    "passenger_demand_generated",
    "passenger_served_count",
}


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def run_writer(output_root: Path) -> None:
    py = Path(sys.executable)
    cmd = [
        str(py),
        str(TRAINING_DIR / "run_toy_causal_rollout.py"),
        "--output-root",
        str(output_root),
        "--conditions",
        "A,A90",
        "--seeds",
        "1,2",
        "--time-bands",
        "peak,offpeak",
        "--windows-per-time-band",
        "1",
        "--num-agents",
        "8",
        "--steps",
        "2",
        "--clean",
    ]

    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise RuntimeError("run_toy_causal_rollout.py failed")

    print(completed.stdout)


def test_writer_outputs() -> None:
    output_root = ROOT / "artifacts" / "phase2_toy_causal_v1_selftest"
    if output_root.exists():
        shutil.rmtree(output_root)

    run_writer(output_root)

    root_manifest_path = output_root / "run_manifest.json"
    scenario_index_path = output_root / "scenario_index.parquet"

    assert_true(root_manifest_path.exists(), "root manifest missing")
    assert_true(scenario_index_path.exists(), "scenario_index missing")

    root_manifest = load_json(root_manifest_path)
    assert_true(root_manifest["causal_comparison_allowed"] is True, "root causal flag must be true")
    assert_true(root_manifest["conditions"] == ["A", "A90"], "condition list mismatch")
    assert_true(root_manifest["seeds"] == [1, 2], "seed list mismatch")
    assert_true(root_manifest["scenario_count"] == 2, "scenario_count mismatch")
    assert_true(root_manifest["total_window_rollup_rows"] == 8, "rollup row total mismatch")
    assert_true(root_manifest["total_raw_event_rows"] == 128, "raw event row total mismatch")

    scenario_df = pd.read_parquet(scenario_index_path)
    assert_true(len(scenario_df) == 2, "scenario_index row count mismatch")
    assert_true(set(scenario_df["time_band"]) == {"peak", "offpeak"}, "scenario time bands mismatch")

    all_rollups = []
    all_raw = []

    for condition_id in ["A", "A90"]:
        for seed in [1, 2]:
            seed_dir = output_root / condition_id / "rollouts" / f"seed_{seed:03d}"
            raw_path = seed_dir / "raw_events.parquet"
            rollup_path = seed_dir / "window_rollup.parquet"
            manifest_path = seed_dir / "run_manifest.json"

            assert_true(raw_path.exists(), f"raw_events missing: {raw_path}")
            assert_true(rollup_path.exists(), f"window_rollup missing: {rollup_path}")
            assert_true(manifest_path.exists(), f"seed manifest missing: {manifest_path}")

            raw_df = pd.read_parquet(raw_path)
            rollup_df = pd.read_parquet(rollup_path)
            manifest = load_json(manifest_path)

            assert_true(REQUIRED_RAW_EVENT_COLUMNS.issubset(raw_df.columns), "raw schema mismatch")
            assert_true(REQUIRED_WINDOW_ROLLUP_COLUMNS.issubset(rollup_df.columns), "rollup schema mismatch")

            assert_true(len(raw_df) == 32, "raw row count per condition/seed mismatch")
            assert_true(len(rollup_df) == 2, "rollup row count per condition/seed mismatch")
            assert_true(bool(raw_df["causal_comparison_allowed"].all()), "raw causal flag false")
            assert_true(bool(rollup_df["causal_comparison_allowed"].all()), "rollup causal flag false")
            assert_true(all(str(x).startswith("causal_") for x in raw_df["source_mode"].unique()), "raw source_mode mismatch")
            assert_true(all(str(x).startswith("causal_") for x in rollup_df["source_mode"].unique()), "rollup source_mode mismatch")
            assert_true((rollup_df["evaluation_horizon_minutes"] == 30).all(), "rollup horizon mismatch")
            assert_true((rollup_df["control_step_minutes"] == 30).all(), "rollup control granularity mismatch")
            assert_true((rollup_df["qwen_trigger_rate"] == 0.0).all(), "qwen_trigger_rate must be zero")
            assert_true((rollup_df["decision_step_count"] > 0).all(), "decision_step_count must be positive")

            assert_true(manifest["causal_comparison_allowed"] is True, "seed manifest causal flag mismatch")
            assert_true(manifest["raw_event_rows"] == 32, "seed manifest raw rows mismatch")
            assert_true(manifest["window_rollup_rows"] == 2, "seed manifest rollup rows mismatch")

            all_raw.append(raw_df)
            all_rollups.append(rollup_df)

    raw_all = pd.concat(all_raw, ignore_index=True)
    rollup_all = pd.concat(all_rollups, ignore_index=True)

    assert_true(set(raw_all["condition_id"].astype(str)) == {"A", "A90"}, "raw condition set mismatch")
    assert_true(set(rollup_all["condition_id"].astype(str)) == {"A", "A90"}, "rollup condition set mismatch")

    # Causal sanity: action choices should have visible position changes and nonzero energy.
    assert_true((raw_all["position_before"] != raw_all["position_after"]).any(), "no movement recorded")
    assert_true((raw_all["energy_proxy_total"] > 0).any(), "no energy usage recorded")
    assert_true((rollup_all["passenger_demand_generated"] >= rollup_all["passenger_served_count"]).all(), "served exceeds generated")

    print("[OK] Step 78 toy causal rollout writer self-test PASS")


def main() -> None:
    test_writer_outputs()


if __name__ == "__main__":
    main()
