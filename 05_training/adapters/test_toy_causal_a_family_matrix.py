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


CONDITIONS = ["A", "A90", "A80", "A70"]
SEEDS = [1, 2, 3]
TIME_BANDS = ["peak", "offpeak", "night"]
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


def validation_get(payload: dict, key: str):
    if key in payload:
        return payload[key]

    preferred_sections = [
        "summary",
        "input_summary",
        "validation_summary",
        "official_input_summary",
        "official_rollup_input_validation",
        "input_validation",
    ]
    for section in preferred_sections:
        value = payload.get(section)
        if isinstance(value, dict) and key in value:
            return value[key]

    for value in payload.values():
        if isinstance(value, dict) and key in value:
            return value[key]

    raise KeyError(f"{key} not found in validation json; top_level_keys={sorted(payload.keys())}")


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
    # Remove an artifact tree on Windows. If the directory is locked,
    # return a fresh retry path instead of failing the self-test.
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


def build_contract(contract_path: Path) -> None:
    payload = {
        "artifact_version": "phase2_toy_causal_a_family_matrix_contract_v1_step81",
        "phase": "Phase-2",
        "condition_id": "PHASE2_TOY_CAUSAL_A_FAMILY",
        "condition_name": "toy_causal_a_family_full_smoke_matrix",
        "qwen_train": False,
        "qwen_inference": False,
        "evaluation_horizon_minutes": 30,
        "seeds": SEEDS,
        "time_bands": TIME_BANDS,
        "shared_kpis": SHARED_KPIS,
        "fairness_constraints": {
            "same_initial_state": True,
            "same_exogenous_events": True,
            "same_eval_window": True,
        },
        "matrix": {
            "conditions": CONDITIONS,
            "expected_condition_seed_pairs": len(CONDITIONS) * len(SEEDS),
            "expected_windows_per_condition_seed": len(TIME_BANDS),
        },
        "claim_boundary": (
            "Toy causal A-family matrix validates Phase 2 wiring only. "
            "It is not a trained MAPPO performance claim."
        ),
    }
    dump_json(contract_path, payload)


def assert_rollout_outputs(rollout_root: Path) -> None:
    root_manifest_path = rollout_root / "run_manifest.json"
    scenario_index_path = rollout_root / "scenario_index.parquet"

    assert_true(root_manifest_path.exists(), "rollout root manifest missing")
    assert_true(scenario_index_path.exists(), "scenario_index missing")

    root_manifest = load_json(root_manifest_path)
    scenario_df = pd.read_parquet(scenario_index_path)

    assert_true(root_manifest["conditions"] == CONDITIONS, "rollout conditions mismatch")
    assert_true(root_manifest["seeds"] == SEEDS, "rollout seeds mismatch")
    assert_true(root_manifest["scenario_count"] == 3, "scenario_count must be 3")
    assert_true(root_manifest["total_window_rollup_rows"] == 36, "total rollup rows mismatch")
    assert_true(root_manifest["total_raw_event_rows"] == 576, "total raw rows mismatch")
    assert_true(root_manifest["causal_comparison_allowed"] is True, "rollout causal flag must be true")

    assert_true(len(scenario_df) == 3, "scenario_index row count mismatch")
    assert_true(set(scenario_df["time_band"].astype(str)) == set(TIME_BANDS), "scenario time band mismatch")

    for condition_id in CONDITIONS:
        for seed in SEEDS:
            seed_dir = rollout_root / condition_id / "rollouts" / f"seed_{seed:03d}"
            raw_path = seed_dir / "raw_events.parquet"
            rollup_path = seed_dir / "window_rollup.parquet"
            manifest_path = seed_dir / "run_manifest.json"

            assert_true(raw_path.exists(), f"raw_events missing: {raw_path}")
            assert_true(rollup_path.exists(), f"window_rollup missing: {rollup_path}")
            assert_true(manifest_path.exists(), f"seed manifest missing: {manifest_path}")

            raw_df = pd.read_parquet(raw_path)
            rollup_df = pd.read_parquet(rollup_path)
            manifest = load_json(manifest_path)

            assert_true(len(raw_df) == 48, "raw rows per condition/seed mismatch")
            assert_true(len(rollup_df) == 3, "rollup rows per condition/seed mismatch")
            assert_true(bool(raw_df["causal_comparison_allowed"].all()), "raw causal flag false")
            assert_true(bool(rollup_df["causal_comparison_allowed"].all()), "rollup causal flag false")
            assert_true(set(rollup_df["time_band"].astype(str)) == set(TIME_BANDS), "rollup time bands mismatch")
            assert_true((rollup_df["evaluation_horizon_minutes"] == 30).all(), "rollup horizon mismatch")
            assert_true((rollup_df["qwen_trigger_rate"] == 0.0).all(), "qwen_trigger_rate must be zero")
            assert_true((rollup_df["decision_step_count"] > 0).all(), "decision_step_count must be positive")
            assert_true(all(str(x).startswith("causal_") for x in rollup_df["source_mode"].unique()), "rollup source_mode must be causal")

            assert_true(manifest["condition_id"] == condition_id, "seed manifest condition mismatch")
            assert_true(manifest["seed"] == seed, "seed manifest seed mismatch")
            assert_true(manifest["raw_event_rows"] == 48, "seed manifest raw rows mismatch")
            assert_true(manifest["window_rollup_rows"] == 3, "seed manifest rollup rows mismatch")
            assert_true(manifest["causal_comparison_allowed"] is True, "seed manifest causal flag mismatch")


def assert_canonical_outputs(canonical_root: Path) -> None:
    kpi_by_window_path = canonical_root / "kpi_by_window.parquet"
    kpi_by_seed_path = canonical_root / "kpi_by_seed.parquet"
    kpi_by_time_band_path = canonical_root / "kpi_by_time_band.parquet"
    kpi_overall_path = canonical_root / "kpi_overall.json"
    manifest_path = canonical_root / "aggregation_manifest.json"
    validation_path = canonical_root / "official_rollup_input_validation.json"

    for path in [
        kpi_by_window_path,
        kpi_by_seed_path,
        kpi_by_time_band_path,
        kpi_overall_path,
        manifest_path,
        validation_path,
    ]:
        assert_true(path.exists(), f"canonical output missing: {path}")

    window_df = pd.read_parquet(kpi_by_window_path)
    seed_df = pd.read_parquet(kpi_by_seed_path)
    time_band_df = pd.read_parquet(kpi_by_time_band_path)
    overall = load_json(kpi_overall_path)
    manifest = load_json(manifest_path)
    validation = load_json(validation_path)

    assert_true(len(window_df) == 36, "kpi_by_window row count mismatch")
    assert_true(len(seed_df) == 12, "kpi_by_seed row count mismatch")
    assert_true(len(time_band_df) == 36, "kpi_by_time_band row count mismatch")

    assert_true(set(window_df["condition_id"].astype(str)) == set(CONDITIONS), "window condition set mismatch")
    assert_true(set(seed_df["condition_id"].astype(str)) == set(CONDITIONS), "seed condition set mismatch")
    assert_true(set(time_band_df["condition_id"].astype(str)) == set(CONDITIONS), "time-band condition set mismatch")
    assert_true(set(window_df["seed"].astype(int)) == set(SEEDS), "window seed set mismatch")
    assert_true(set(window_df["time_band"].astype(str)) == set(TIME_BANDS), "window time band mismatch")

    assert_true(bool(window_df["strict_canonical"].all()), "strict_canonical must be true")
    assert_true(set(window_df["computation_mode"].astype(str)) == {"official_rollup"}, "computation_mode mismatch")
    assert_true(bool(window_df["causal_comparison_allowed"].all()), "window causal flag must be true")
    assert_true(bool(seed_df["causal_comparison_allowed"].all()), "seed causal flag must be true")
    assert_true(bool(time_band_df["causal_comparison_allowed"].all()), "time-band causal flag must be true")
    assert_true(overall["causal_comparison_allowed"] is True, "overall causal flag must be true")
    assert_true(manifest["causal_comparison_allowed"] is True, "manifest causal flag must be true")
    assert_true(manifest["strict_canonical"] is True, "manifest strict_canonical must be true")

    assert_true(int(overall["n_windows_total"]) == 36, "overall window count mismatch")
    assert_true(int(overall["n_condition_seed_pairs"]) == 12, "overall condition-seed count mismatch")

    assert_true(int(validation_get(validation, "row_count")) == 36, "validation row_count mismatch")
    assert_true(set(validation_get(validation, "condition_ids")) == set(CONDITIONS), "validation condition ids mismatch")
    assert_true(set(validation_get(validation, "seed_values")) == set(SEEDS), "validation seed values mismatch")
    assert_true(int(validation_get(validation, "window_count")) == 3, "validation window_count mismatch")
    assert_true(set(validation_get(validation, "time_bands")) == set(TIME_BANDS), "validation time band mismatch")

    source_modes = sorted(window_df["source_mode"].astype(str).unique().tolist())
    assert_true(source_modes == ["causal_toy_suseong_v1"], f"unexpected source_modes: {source_modes}")

    for kpi in SHARED_KPIS:
        assert_true(kpi in window_df.columns, f"missing KPI in kpi_by_window: {kpi}")
        valid_count = int(pd.to_numeric(window_df[kpi], errors="coerce").notna().sum())
        assert_true(valid_count > 0, f"KPI has no valid window values: {kpi}")
        assert_true(kpi in overall["kpis"], f"missing KPI in overall json: {kpi}")

    for bounded in ["bunching_rate", "on_time_rate", "intervention_rate"]:
        s = pd.to_numeric(window_df[bounded], errors="coerce").dropna()
        assert_true(bool(((s >= 0.0) & (s <= 1.0)).all()), f"{bounded} must be within [0, 1]")


def test_toy_causal_a_family_full_smoke_matrix() -> None:
    py = Path(sys.executable)

    output_root = ROOT / "artifacts" / "phase2_toy_causal_a_family_matrix_selftest"
    rollout_root = output_root / "rollouts"
    canonical_root = output_root / "canonical_eval"
    contract_path = output_root / "toy_causal_a_family_contract.json"

    output_root = safe_rmtree_for_windows(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    build_contract(contract_path)

    run_cmd(
        [
            str(py),
            str(TRAINING_DIR / "run_toy_causal_rollout.py"),
            "--output-root",
            str(rollout_root),
            "--conditions",
            ",".join(CONDITIONS),
            "--seeds",
            ",".join(str(x) for x in SEEDS),
            "--time-bands",
            ",".join(TIME_BANDS),
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

    assert_rollout_outputs(rollout_root)

    scenario_index_path = rollout_root / "scenario_index.parquet"

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

    assert_canonical_outputs(canonical_root)

    print("[OK] Step 81 toy causal A-family full smoke matrix PASS")
    print(f"[OK] output_root       : {output_root}")
    print(f"[OK] rollout_root      : {rollout_root}")
    print(f"[OK] canonical_root    : {canonical_root}")
    print(f"[OK] conditions        : {CONDITIONS}")
    print(f"[OK] seeds             : {SEEDS}")
    print("[OK] condition_seed_runs: 12")
    print("[OK] kpi_by_window rows: 36")
    print("[OK] causal_allowed    : True")


def main() -> None:
    test_toy_causal_a_family_full_smoke_matrix()


if __name__ == "__main__":
    main()
