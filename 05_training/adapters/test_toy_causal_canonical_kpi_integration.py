from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "05_training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


SHARED_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
]


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
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


def build_toy_contract(contract_path: Path) -> None:
    payload = {
        "artifact_version": "phase2_toy_causal_contract_v1_step79",
        "phase": "Phase-2",
        "condition_id": "PHASE2_TOY_CAUSAL",
        "condition_name": "toy_causal_canonical_integration",
        "qwen_train": False,
        "qwen_inference": False,
        "evaluation_horizon_minutes": 30,
        "seeds": [1, 2],
        "time_bands": ["peak", "offpeak", "night"],
        "shared_kpis": SHARED_KPIS,
        "fairness_constraints": {
            "same_initial_state": True,
            "same_exogenous_events": True,
            "same_eval_window": True,
        },
        "status": "toy_causal_contract_for_step79_selftest",
        "note": (
            "This contract is for Phase 2 toy causal simulator canonical KPI "
            "integration validation only. It is not a paper-level performance claim."
        ),
    }
    dump_json(contract_path, payload)



def safe_rmtree_for_windows(path: Path) -> Path:
    # Windows can keep parquet/canonical_eval folders locked through Explorer,
    # antivirus, or previous Python handles. Retry first; if still locked,
    # use a fresh retry output path so the test can proceed.
    if not path.exists():
        return path

    def on_error(func, target, exc_info):
        try:
            os.chmod(target, 0o700)
            func(target)
        except Exception:
            pass

    for attempt in range(5):
        try:
            try:
                shutil.rmtree(path, onexc=on_error)
            except TypeError:
                shutil.rmtree(path, onerror=on_error)
            return path
        except PermissionError:
            time.sleep(0.5 + attempt * 0.5)

    fallback = path.with_name(f"{path.name}_retry_{os.getpid()}_{int(time.time())}")
    print(f"[WARN] could not remove locked artifact directory: {path}")
    print(f"[WARN] using fallback output directory instead: {fallback}")
    return fallback


def test_toy_causal_canonical_kpi_integration() -> None:
    py = Path(sys.executable)

    output_root = ROOT / "artifacts" / "phase2_toy_causal_v1_canonical_selftest"
    rollout_root = output_root / "rollouts"
    canonical_root = output_root / "canonical_eval"
    contract_path = output_root / "toy_causal_contract.json"

    output_root = safe_rmtree_for_windows(output_root)

    # Re-bind derived paths after possible fallback output_root replacement.
    rollout_root = output_root / "rollouts"
    canonical_root = output_root / "canonical_eval"
    contract_path = output_root / "toy_causal_contract.json"

    output_root.mkdir(parents=True, exist_ok=True)
    build_toy_contract(contract_path)

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "run_toy_causal_rollout.py"),
            "--output-root",
            str(rollout_root),
            "--conditions",
            "A,A90",
            "--seeds",
            "1,2",
            "--time-bands",
            "peak,offpeak,night",
            "--windows-per-time-band",
            "1",
            "--num-agents",
            "8",
            "--steps",
            "2",
            "--clean",
        ],
        cwd=ROOT,
    )

    scenario_index_path = rollout_root / "scenario_index.parquet"
    assert_true(scenario_index_path.exists(), "scenario_index.parquet missing")

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "evaluation" / "canonical_kpi_aggregator.py"),
            "--mode",
            "official_rollup",
            "--contract",
            str(contract_path),
            "--input-root",
            str(rollout_root),
            "--scenario-index",
            str(scenario_index_path),
            "--output-root",
            str(canonical_root),
            "--smoke",
        ],
        cwd=ROOT,
    )

    kpi_by_window_path = canonical_root / "kpi_by_window.parquet"
    kpi_by_seed_path = canonical_root / "kpi_by_seed.parquet"
    kpi_by_time_band_path = canonical_root / "kpi_by_time_band.parquet"
    kpi_overall_path = canonical_root / "kpi_overall.json"
    aggregation_manifest_path = canonical_root / "aggregation_manifest.json"

    for path in [
        kpi_by_window_path,
        kpi_by_seed_path,
        kpi_by_time_band_path,
        kpi_overall_path,
        aggregation_manifest_path,
    ]:
        assert_true(path.exists(), f"canonical output missing: {path}")

    window_df = pd.read_parquet(kpi_by_window_path)
    seed_df = pd.read_parquet(kpi_by_seed_path)
    time_band_df = pd.read_parquet(kpi_by_time_band_path)
    overall = load_json(kpi_overall_path)
    manifest = load_json(aggregation_manifest_path)

    assert_true(len(window_df) == 12, "kpi_by_window row count mismatch")
    assert_true(len(seed_df) == 4, "kpi_by_seed row count mismatch")
    assert_true(len(time_band_df) == 12, "kpi_by_time_band row count mismatch")

    assert_true(set(window_df["condition_id"].astype(str)) == {"A", "A90"}, "window condition set mismatch")
    assert_true(set(seed_df["condition_id"].astype(str)) == {"A", "A90"}, "seed condition set mismatch")
    assert_true(set(window_df["seed"].astype(int)) == {1, 2}, "window seed set mismatch")
    assert_true(set(window_df["time_band"].astype(str)) == {"peak", "offpeak", "night"}, "time band set mismatch")

    assert_true(bool(window_df["strict_canonical"].all()), "strict_canonical must be true")
    assert_true(set(window_df["computation_mode"].astype(str)) == {"official_rollup"}, "computation_mode mismatch")
    assert_true(bool(window_df["causal_comparison_allowed"].all()), "window causal flag must be true")
    assert_true(bool(seed_df["causal_comparison_allowed"].all()), "seed causal flag must be true")
    assert_true(bool(time_band_df["causal_comparison_allowed"].all()), "time-band causal flag must be true")
    assert_true(overall["causal_comparison_allowed"] is True, "overall causal flag must be true")
    assert_true(manifest["causal_comparison_allowed"] is True, "manifest causal flag must be true")
    assert_true(manifest["strict_canonical"] is True, "manifest strict_canonical must be true")

    for kpi in SHARED_KPIS:
        assert_true(kpi in window_df.columns, f"missing KPI in kpi_by_window: {kpi}")
        valid_count = int(pd.to_numeric(window_df[kpi], errors="coerce").notna().sum())
        assert_true(valid_count > 0, f"KPI has no valid window values: {kpi}")
        assert_true(kpi in overall["kpis"], f"missing KPI in overall json: {kpi}")

    for bounded in ["bunching_rate", "on_time_rate", "intervention_rate"]:
        s = pd.to_numeric(window_df[bounded], errors="coerce").dropna()
        assert_true(bool(((s >= 0.0) & (s <= 1.0)).all()), f"{bounded} must be within [0, 1]")

    source_modes = sorted(window_df["source_mode"].astype(str).unique().tolist())
    assert_true(source_modes == ["causal_toy_suseong_v1"], f"unexpected source_modes: {source_modes}")

    assert_true(
        int(overall["n_windows_total"]) == 12,
        "overall n_windows_total mismatch",
    )
    assert_true(
        int(overall["n_condition_seed_pairs"]) == 4,
        "overall n_condition_seed_pairs mismatch",
    )

    print("[OK] Step 79 toy causal canonical KPI integration self-test PASS")
    print(f"[OK] rollout_root       : {rollout_root}")
    print(f"[OK] canonical_root     : {canonical_root}")
    print(f"[OK] kpi_by_window rows : {len(window_df)}")
    print(f"[OK] kpi_by_seed rows   : {len(seed_df)}")
    print(f"[OK] kpi_by_band rows   : {len(time_band_df)}")
    print(f"[OK] causal_allowed     : {overall['causal_comparison_allowed']}")


def main() -> None:
    test_toy_causal_canonical_kpi_integration()


if __name__ == "__main__":
    main()
