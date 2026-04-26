"""
Self-test for run_a_family_bridge_smoke_v1.py.

Run:
    python ./05_training/rollouts\test_a_family_bridge_smoke_runner_v1.py
"""

import shutil
from pathlib import Path

from run_a_family_bridge_smoke_v1 import run_smoke


def test_placeholder_smoke_runner():
    out = Path("artifacts/step26_a_family_bridge_smoke_test_placeholder")

    if out.exists():
        shutil.rmtree(out)

    manifest = run_smoke(
        output_root=out,
        policy_kind="placeholder",
        checkpoint_path=None,
        require_existing_checkpoint=False,
        write_parquet=False,
    )

    assert manifest["passed"] is True
    assert (out / "summary.json").exists()
    assert (out / "a_family_rows.jsonl").exists()
    assert (out / "a_family_rows.csv").exists()


def test_mappo_smoke_runner_without_existing_checkpoint_check():
    out = Path("artifacts/step26_a_family_bridge_smoke_test_mappo")

    if out.exists():
        shutil.rmtree(out)

    manifest = run_smoke(
        output_root=out,
        policy_kind="mappo",
        checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
        require_existing_checkpoint=False,
        write_parquet=False,
    )

    assert manifest["passed"] is True
    assert (out / "summary.json").exists()


def test_mappo_smoke_runner_rejects_missing_checkpoint_path():
    out = Path("artifacts/step26_a_family_bridge_smoke_test_fail")

    try:
        run_smoke(
            output_root=out,
            policy_kind="mappo",
            checkpoint_path="",
            require_existing_checkpoint=False,
            write_parquet=False,
        )
    except FileNotFoundError:
        return

    raise AssertionError("missing MAPPO checkpoint_path was not rejected")


def main():
    tests = [
        test_placeholder_smoke_runner,
        test_mappo_smoke_runner_without_existing_checkpoint_check,
        test_mappo_smoke_runner_rejects_missing_checkpoint_path,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] a_family_bridge_smoke_runner_v1 self-test passed")


if __name__ == "__main__":
    main()
