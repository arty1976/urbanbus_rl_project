"""
Self-test for Step 37 A-family policy interface scenario writer.

Run:
    python ./05_training/rollouts/test_a_family_policy_interface_scenario_writer_v1.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from run_a_family_policy_interface_scenario_writer_v1 import run_writer


def test_step37_self_test_writer() -> None:
    out = Path("artifacts/step37_policy_interface_scenario_writer_test")

    if out.exists():
        shutil.rmtree(out)

    manifest = run_writer(
        scenario_index_path=None,
        output_root=out,
        limit=2,
        conditions=["A", "A90", "A80", "A70"],
        seeds=[1],
        checkpoint_path=None,
        use_existing_checkpoint=False,
        baseline_bus_count=10.0,
        eval_horizon_minutes=30,
        write_parquet_files=True,
        self_test=True,
    )

    assert manifest["passed"] is True
    assert manifest["action_summary"]["passed"] is True
    assert manifest["action_summary"]["policy_sources"] == ["mappo_policy"]

    for source_mode in manifest["action_summary"]["source_modes"]:
        assert source_mode.startswith("causal_")
        assert "_mappo_policy_v1" in source_mode
        assert "placeholder" not in source_mode
        assert "stub" not in source_mode
        assert "smoke" not in source_mode

    assert (out / "combined_policy_interface_window_rollup.parquet").exists()
    assert (out / "A80" / "rollouts" / "seed_001" / "window_rollup.parquet").exists()


def test_step37_uses_existing_checkpoint_metadata() -> None:
    out = Path("artifacts/step37_policy_interface_scenario_writer_existing_ckpt_test")

    if out.exists():
        shutil.rmtree(out)

    # First run creates a contract metadata checkpoint.
    first = run_writer(
        scenario_index_path=None,
        output_root=out / "first",
        limit=1,
        conditions=["A80"],
        seeds=[1],
        checkpoint_path=None,
        use_existing_checkpoint=False,
        baseline_bus_count=10.0,
        eval_horizon_minutes=30,
        write_parquet_files=False,
        self_test=True,
    )

    ckpt = Path(first["checkpoint_path"])
    assert ckpt.exists()

    second = run_writer(
        scenario_index_path=None,
        output_root=out / "second",
        limit=1,
        conditions=["A80"],
        seeds=[1],
        checkpoint_path=ckpt,
        use_existing_checkpoint=True,
        baseline_bus_count=10.0,
        eval_horizon_minutes=30,
        write_parquet_files=False,
        self_test=True,
    )

    assert second["passed"] is True
    assert second["checkpoint_path"] == str(ckpt)


def test_step37_rejects_missing_existing_checkpoint() -> None:
    out = Path("artifacts/step37_policy_interface_scenario_writer_missing_ckpt_test")

    try:
        run_writer(
            scenario_index_path=None,
            output_root=out,
            limit=1,
            conditions=["A80"],
            seeds=[1],
            checkpoint_path=Path("artifacts/does_not_exist/missing.metadata.json"),
            use_existing_checkpoint=True,
            baseline_bus_count=10.0,
            eval_horizon_minutes=30,
            write_parquet_files=False,
            self_test=True,
        )
    except FileNotFoundError:
        return

    raise AssertionError("missing existing checkpoint metadata was not rejected")


def main() -> None:
    tests = [
        test_step37_self_test_writer,
        test_step37_uses_existing_checkpoint_metadata,
        test_step37_rejects_missing_existing_checkpoint,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] Step 37 A-family policy interface scenario writer self-test passed")


if __name__ == "__main__":
    main()
