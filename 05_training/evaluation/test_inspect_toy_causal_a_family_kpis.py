from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "05_training"

PHASE2_12_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
]

BOUNDED_0_1 = {
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "passenger_service_rate",
    "fleet_reduction_ratio",
}


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def run_cmd(cmd: list[str], cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if completed.returncode != 0:
        print("[STDOUT]")
        print(completed.stdout)
        print("[STDERR]")
        print(completed.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    print(completed.stdout)


def test_inspect_toy_causal_a_family_kpis() -> None:
    py = Path(sys.executable)

    # Rebuild the upstream A-family canonical artifacts first.
    run_cmd(
        [str(py), str(TRAINING_DIR / "adapters" / "test_toy_causal_a_family_matrix.py")],
        cwd=ROOT,
    )

    canonical_root = ROOT / "artifacts" / "phase2_toy_causal_a_family_matrix_selftest" / "canonical_eval"
    output_root = ROOT / "artifacts" / "phase2_toy_causal_a_family_matrix_selftest" / "inspection"

    if output_root.exists():
        shutil.rmtree(output_root)

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "evaluation" / "inspect_toy_causal_a_family_kpis.py"),
            "--canonical-root",
            str(canonical_root),
            "--output-root",
            str(output_root),
            "--conditions",
            "A,A90,A80,A70",
        ],
        cwd=ROOT,
    )

    condition_summary_path = output_root / "condition_summary.csv"
    delta_path = output_root / "condition_vs_A_delta.csv"
    time_band_summary_path = output_root / "time_band_summary.csv"
    seed_summary_path = output_root / "seed_summary.csv"
    report_path = output_root / "inspection_report.json"

    for path in [
        condition_summary_path,
        delta_path,
        time_band_summary_path,
        seed_summary_path,
        report_path,
    ]:
        assert_true(path.exists(), f"missing inspection output: {path}")

    condition_summary = pd.read_csv(condition_summary_path)
    delta_df = pd.read_csv(delta_path)
    time_band_summary = pd.read_csv(time_band_summary_path)
    seed_summary = pd.read_csv(seed_summary_path)
    report = load_json(report_path)

    assert_true(len(condition_summary) == 4, "condition_summary must have 4 rows")
    assert_true(set(condition_summary["condition_id"].astype(str)) == {"A", "A90", "A80", "A70"}, "condition set mismatch")
    assert_true(len(delta_df) == 36, "condition_vs_A_delta must have 3*12 rows")
    assert_true(len(time_band_summary) == 12, "time_band_summary must have 4*3 rows")
    assert_true(len(seed_summary) == 12, "seed_summary must have 4*3 rows")

    assert_true(report["overall_has_all_12_kpis"] is True, "overall must include all 12 KPIs")
    assert_true(report["causal_comparison_allowed"] is True, "causal flag must be true")
    assert_true(report["row_counts"]["condition_summary"] == 4, "report condition count mismatch")
    assert_true(report["row_counts"]["condition_vs_A_delta"] == 36, "report delta count mismatch")
    assert_true(report["claim_boundary"] == "toy_causal_sanity_only_not_paper_performance_claim", "claim boundary mismatch")

    for kpi in PHASE2_12_KPIS:
        mean_col = f"{kpi}_mean"
        assert_true(mean_col in condition_summary.columns, f"missing condition summary column: {mean_col}")
        valid_count = int(pd.to_numeric(condition_summary[mean_col], errors="coerce").notna().sum())
        assert_true(valid_count > 0, f"condition summary has no valid values for {kpi}")

    for bounded in BOUNDED_0_1:
        mean_col = f"{bounded}_mean"
        values = pd.to_numeric(condition_summary[mean_col], errors="coerce").dropna()
        assert_true(bool(((values >= 0.0) & (values <= 1.0)).all()), f"{bounded} must be bounded")

    fleet = dict(zip(condition_summary["condition_id"].astype(str), condition_summary["fleet_reduction_ratio_mean"]))
    assert_true(abs(float(fleet["A"]) - 0.0) < 1e-9, "A fleet_reduction_ratio should be 0.0")
    assert_true(0.09 <= float(fleet["A90"]) <= 0.11, "A90 fleet_reduction_ratio should be about 0.1")
    assert_true(0.19 <= float(fleet["A80"]) <= 0.21, "A80 fleet_reduction_ratio should be about 0.2")
    assert_true(0.29 <= float(fleet["A70"]) <= 0.31, "A70 fleet_reduction_ratio should be about 0.3")

    assert_true(set(delta_df["kpi"].astype(str)) == set(PHASE2_12_KPIS), "delta KPI set mismatch")
    assert_true(set(delta_df["condition_id"].astype(str)) == {"A90", "A80", "A70"}, "delta condition set mismatch")

    print("[OK] Step 84 toy causal A-family 12-KPI inspector self-test PASS")


def main() -> None:
    test_inspect_toy_causal_a_family_kpis()


if __name__ == "__main__":
    main()
