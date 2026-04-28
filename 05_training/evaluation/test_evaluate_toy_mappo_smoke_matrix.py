from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "05_training"

PHASE2_12_KPIS = {
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
}


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def safe_rmtree_for_windows(path: Path) -> Path:
    if not path.exists():
        return path

    def onexc(func, target, exc_info):
        try:
            import os
            os.chmod(target, 0o700)
            func(target)
        except Exception:
            pass

    for attempt in range(5):
        try:
            try:
                shutil.rmtree(path, onexc=onexc)
            except TypeError:
                shutil.rmtree(path, onerror=onexc)
            return path
        except PermissionError:
            time.sleep(0.5 + attempt * 0.5)

    fallback = path.with_name(f"{path.name}_retry_{int(time.time())}")
    print(f"[WARN] could not remove locked artifact directory: {path}")
    print(f"[WARN] using fallback output directory instead: {fallback}")
    return fallback


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def run_cmd(cmd: list[str], cwd: Path) -> str:
    completed = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if completed.returncode != 0:
        print("[STDOUT]")
        print(completed.stdout)
        print("[STDERR]")
        print(completed.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    return completed.stdout


def test_toy_mappo_smoke_matrix_evaluator() -> None:
    py = Path(sys.executable)

    output_root = ROOT / "artifacts" / "phase2_toy_mappo_smoke_matrix_eval_selftest"
    output_root = safe_rmtree_for_windows(output_root)

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "evaluation" / "evaluate_toy_mappo_smoke_matrix.py"),
            "--output-root",
            str(output_root),
            "--conditions",
            "A,A90,A80,A70",
            "--seed",
            "93",
            "--num-agents",
            "8",
            "--steps",
            "4",
            "--time-bands",
            "peak,offpeak,night",
            "--windows-per-time-band",
            "1",
            "--action-mode",
            "argmax",
            "--device",
            "cpu",
            "--clean",
        ],
        cwd=ROOT,
    )

    manifest_path = output_root / "matrix_evaluation_manifest.json"
    summary_path = output_root / "matrix_summary.csv"
    combined_root = output_root / "matrix_canonical_eval"
    inspection_root = output_root / "matrix_inspection"

    for path in [
        manifest_path,
        summary_path,
        combined_root / "kpi_by_window.parquet",
        combined_root / "kpi_by_seed.parquet",
        combined_root / "kpi_by_time_band.parquet",
        combined_root / "kpi_overall.json",
        inspection_root / "condition_summary.csv",
        inspection_root / "condition_vs_A_delta.csv",
        inspection_root / "inspection_report.json",
    ]:
        assert_true(path.exists(), f"missing expected matrix output: {path}")

    manifest = load_json(manifest_path)
    assert_true(manifest["artifact_version"] == "toy_mappo_smoke_matrix_evaluation_v1_step93", "manifest artifact version mismatch")
    assert_true(manifest["conditions"] == ["A", "A90", "A80", "A70"], "condition order mismatch")
    assert_true(manifest["reward_version"] == "mappo_reward_v1", "reward version mismatch")
    assert_true(manifest["trained_model"] is False, "trained_model must be false")
    assert_true(manifest["performance_claim_allowed"] is False, "performance claim must be false")
    assert_true(manifest["smoke_evaluation_only"] is True, "smoke_evaluation_only must be true")
    assert_true(manifest["matrix_smoke_evaluation_only"] is True, "matrix smoke flag must be true")
    assert_true(manifest["causal_comparison_allowed"] is True, "causal flag must be true")
    assert_true(set(manifest["reward_metric_keys"]) == PHASE2_12_KPIS, "12-KPI key mismatch")
    assert_true("not a paper-level performance" in manifest["claim_boundary"], "claim boundary missing")
    assert_true(int(manifest["kpi_by_window_rows"]) == 12, "combined kpi_by_window rows must be 12")

    summary_df = pd.read_csv(summary_path)
    assert_true(set(summary_df["condition_id"].astype(str)) == {"A", "A90", "A80", "A70"}, "summary condition set mismatch")
    assert_true(len(summary_df) == 4, "matrix summary must have 4 rows")
    assert_true(bool((summary_df["performance_claim_allowed"] == False).all()), "summary performance claims must be false")
    assert_true(bool((summary_df["smoke_evaluation_only"] == True).all()), "summary smoke flags must be true")

    kpi_by_window = pd.read_parquet(combined_root / "kpi_by_window.parquet")
    assert_true(len(kpi_by_window) == 12, "combined kpi_by_window must have 12 rows")
    assert_true(set(kpi_by_window["condition_id"].astype(str)) == {"A", "A90", "A80", "A70"}, "canonical condition set mismatch")
    for kpi in PHASE2_12_KPIS:
        assert_true(kpi in kpi_by_window.columns, f"combined canonical missing KPI: {kpi}")

    condition_summary = pd.read_csv(inspection_root / "condition_summary.csv")
    delta_df = pd.read_csv(inspection_root / "condition_vs_A_delta.csv")
    report = load_json(inspection_root / "inspection_report.json")

    assert_true(len(condition_summary) == 4, "condition_summary must have 4 rows")
    assert_true(len(delta_df) == 36, "condition_vs_A_delta must have 36 rows")
    assert_true(report["overall_has_all_12_kpis"] is True, "inspection report must include all 12 KPIs")
    assert_true(report["causal_comparison_allowed"] is True, "inspection causal flag must be true")

    for condition_id in ["A", "A90", "A80", "A70"]:
        condition_manifest_path = output_root / "conditions" / condition_id / "evaluation_manifest.json"
        assert_true(condition_manifest_path.exists(), f"condition manifest missing: {condition_id}")
        condition_manifest = load_json(condition_manifest_path)
        assert_true(condition_manifest["condition_id"] == condition_id, f"{condition_id} manifest condition mismatch")
        assert_true(condition_manifest["performance_claim_allowed"] is False, f"{condition_id} performance claim must be false")
        assert_true(condition_manifest["smoke_evaluation_only"] is True, f"{condition_id} smoke flag must be true")

    print("[OK] Step 93 toy MAPPO smoke matrix evaluator self-test PASS")


def main() -> None:
    test_toy_mappo_smoke_matrix_evaluator()


if __name__ == "__main__":
    main()
