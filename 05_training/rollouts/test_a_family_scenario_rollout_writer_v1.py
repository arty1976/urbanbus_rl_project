"""
Self-test for run_a_family_scenario_rollout_writer_v1.py.

Run:
    python .\05_training\rollouts\test_a_family_scenario_rollout_writer_v1.py
"""

import shutil
from pathlib import Path

from run_a_family_scenario_rollout_writer_v1 import run_writer


def test_self_test_placeholder_writer():
    out = Path("artifacts/step27_a_family_scenario_writer_test_placeholder")

    if out.exists():
        shutil.rmtree(out)

    manifest = run_writer(
        scenario_index_path=None,
        output_root=out,
        limit=2,
        conditions=["A", "A90", "A80", "A70"],
        seeds=[1, 2],
        policy_kind="placeholder",
        checkpoint_path=None,
        require_existing_checkpoint=False,
        baseline_bus_count=10.0,
        eval_horizon_minutes=30,
        write_parquet_files=False,
        self_test=True,
    )

    assert manifest["passed"] is True
    assert (out / "summary.json").exists()
    assert (out / "combined_a_family_window_rollup.csv").exists()
    assert (out / "A" / "rollouts" / "seed_001" / "window_rollup.csv").exists()
    assert (out / "A90" / "rollouts" / "seed_002" / "window_rollup.csv").exists()


def test_self_test_mappo_writer_without_existing_checkpoint_check():
    out = Path("artifacts/step27_a_family_scenario_writer_test_mappo")

    if out.exists():
        shutil.rmtree(out)

    manifest = run_writer(
        scenario_index_path=None,
        output_root=out,
        limit=1,
        conditions=["A", "A90", "A80", "A70"],
        seeds=[1],
        policy_kind="mappo",
        checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
        require_existing_checkpoint=False,
        baseline_bus_count=10.0,
        eval_horizon_minutes=30,
        write_parquet_files=False,
        self_test=True,
    )

    assert manifest["passed"] is True
    assert (out / "summary.json").exists()


def test_mappo_writer_rejects_missing_checkpoint_path():
    out = Path("artifacts/step27_a_family_scenario_writer_test_fail")

    try:
        run_writer(
            scenario_index_path=None,
            output_root=out,
            limit=1,
            conditions=["A"],
            seeds=[1],
            policy_kind="mappo",
            checkpoint_path="",
            require_existing_checkpoint=False,
            baseline_bus_count=10.0,
            eval_horizon_minutes=30,
            write_parquet_files=False,
            self_test=True,
        )
    except FileNotFoundError:
        return

    raise AssertionError("missing MAPPO checkpoint_path was not rejected")


def main():
    tests = [
        test_self_test_placeholder_writer,
        test_self_test_mappo_writer_without_existing_checkpoint_check,
        test_mappo_writer_rejects_missing_checkpoint_path,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] a_family_scenario_rollout_writer_v1 self-test passed")


if __name__ == "__main__":
    main()
