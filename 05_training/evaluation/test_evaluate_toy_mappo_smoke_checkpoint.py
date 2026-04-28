from __future__ import annotations

import json
import os
import math
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
    return completed.stdout


def safe_rmtree_for_windows(path: Path) -> Path:
    if not path.exists():
        return path

    def onexc(func, target, exc_info):
        try:
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

    fallback = path.with_name(f"{path.name}_retry_{os.getpid()}_{int(time.time())}")
    print(f"[WARN] could not remove locked artifact directory: {path}")
    print(f"[WARN] using fallback output directory instead: {fallback}")
    return fallback


def test_evaluate_toy_mappo_smoke_checkpoint() -> None:
    py = Path(sys.executable)

    training_root = ROOT / "artifacts" / "phase2_toy_mappo_smoke_eval_training_selftest"
    eval_root = ROOT / "artifacts" / "phase2_toy_mappo_smoke_checkpoint_eval_selftest"

    training_root = safe_rmtree_for_windows(training_root)
    eval_root = safe_rmtree_for_windows(eval_root)

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "train_toy_causal_mappo_smoke.py"),
            "--output-root",
            str(training_root),
            "--condition-id",
            "A",
            "--seed",
            "91",
            "--num-agents",
            "8",
            "--steps",
            "4",
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--clean",
        ],
        cwd=ROOT,
    )

    checkpoint_path = training_root / "checkpoints" / "toy_mappo_smoke_checkpoint.pt"
    assert_true(checkpoint_path.exists(), "training checkpoint missing")

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "evaluation" / "evaluate_toy_mappo_smoke_checkpoint.py"),
            "--checkpoint",
            str(checkpoint_path),
            "--output-root",
            str(eval_root),
            "--condition-id",
            "A",
            "--seed",
            "91",
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

    manifest_path = eval_root / "evaluation_manifest.json"
    raw_path = eval_root / "rollouts" / "A" / "rollouts" / "seed_091" / "raw_events.parquet"
    rollup_path = eval_root / "rollouts" / "A" / "rollouts" / "seed_091" / "window_rollup.parquet"
    reward_trace_path = eval_root / "rollouts" / "A" / "rollouts" / "seed_091" / "reward_trace.csv"
    canonical_root = eval_root / "canonical_eval"
    inspection_root = eval_root / "inspection"

    for path in [
        manifest_path,
        raw_path,
        rollup_path,
        reward_trace_path,
        canonical_root / "kpi_by_window.parquet",
        canonical_root / "kpi_by_seed.parquet",
        canonical_root / "kpi_by_time_band.parquet",
        canonical_root / "kpi_overall.json",
        inspection_root / "condition_summary.csv",
        inspection_root / "inspection_report.json",
    ]:
        assert_true(path.exists(), f"missing expected output: {path}")

    manifest = load_json(manifest_path)
    assert_true(manifest["artifact_version"] == "toy_mappo_smoke_checkpoint_evaluation_v1_step91", "manifest artifact version mismatch")
    assert_true(manifest["reward_version"] == "mappo_reward_v1", "reward version mismatch")
    assert_true(manifest["trained_model"] is False, "trained_model must be false")
    assert_true(manifest["performance_claim_allowed"] is False, "performance claim must be false")
    assert_true(manifest["smoke_evaluation_only"] is True, "smoke_evaluation_only must be true")
    assert_true(manifest["causal_comparison_allowed"] is True, "causal flag must be true")
    assert_true(set(manifest["reward_metric_keys"]) == PHASE2_12_KPIS, "12-KPI key mismatch")
    assert_true("not a paper-level performance claim" in manifest["note"], "claim boundary note missing")
    assert_true(int(manifest["raw_event_rows"]) > 0, "raw_event_rows must be positive")
    assert_true(int(manifest["window_rollup_rows"]) == 3, "window_rollup_rows must be 3")
    assert_true(int(manifest["reward_trace_rows"]) > 0, "reward_trace_rows must be positive")
    assert_true(math.isfinite(float(manifest["reward_summary"]["reward_total_mean"])), "reward mean must be finite")

    rollup_df = pd.read_parquet(rollup_path)
    assert_true(len(rollup_df) == 3, "rollup rows must be 3")
    for kpi in PHASE2_12_KPIS:
        assert_true(kpi in rollup_df.columns, f"rollup missing KPI: {kpi}")

    kpi_by_window = pd.read_parquet(canonical_root / "kpi_by_window.parquet")
    assert_true(len(kpi_by_window) == 3, "canonical kpi_by_window rows must be 3")
    for kpi in PHASE2_12_KPIS:
        assert_true(kpi in kpi_by_window.columns, f"canonical output missing KPI: {kpi}")

    inspection_report = load_json(inspection_root / "inspection_report.json")
    assert_true(inspection_report["overall_has_all_12_kpis"] is True, "inspection must include all 12 KPIs")
    assert_true(inspection_report["claim_boundary"] == "toy_causal_sanity_only_not_paper_performance_claim", "inspection claim boundary mismatch")

    print("[OK] Step 91 toy MAPPO smoke checkpoint evaluator self-test PASS")


def main() -> None:
    test_evaluate_toy_mappo_smoke_checkpoint()


if __name__ == "__main__":
    main()
