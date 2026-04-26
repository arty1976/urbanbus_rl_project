"""
Self-test for run_causal_rollout_a_family_bridge_v1.py.

Run:
    python ./05_training/test_run_causal_rollout_a_family_bridge_v1.py
"""

import shutil
from argparse import Namespace
from pathlib import Path

from run_causal_rollout_a_family_bridge_v1 import run_entry


def make_args(**overrides):
    base = {
        "scenario_index": "artifacts/baseline_v1/B1_noop/scenario_index.parquet",
        "output_root": "artifacts/step28_root_a_family_bridge_test",
        "limit": 2,
        "conditions": "A,A90,A80,A70",
        "seeds": "1,2",
        "policy_kind": "placeholder",
        "checkpoint_path": "artifacts/experiment_A_v1/checkpoints/best.pt",
        "require_existing_checkpoint": False,
        "baseline_bus_count": 10.0,
        "eval_horizon_minutes": 30,
        "write_parquet": False,
        "self_test": True,
    }
    base.update(overrides)
    return Namespace(**base)


def test_root_entry_placeholder_self_test():
    out = Path("artifacts/step28_root_a_family_bridge_test_placeholder")

    if out.exists():
        shutil.rmtree(out)

    args = make_args(
        output_root=str(out),
        policy_kind="placeholder",
        self_test=True,
    )

    manifest = run_entry(args)

    assert manifest["passed"] is True
    assert (out / "entry_manifest.json").exists()
    assert (out / "manifest.json").exists()
    assert (out / "summary.json").exists()


def test_root_entry_mappo_without_existing_checkpoint_check():
    out = Path("artifacts/step28_root_a_family_bridge_test_mappo")

    if out.exists():
        shutil.rmtree(out)

    args = make_args(
        output_root=str(out),
        policy_kind="mappo",
        checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
        require_existing_checkpoint=False,
        self_test=True,
    )

    manifest = run_entry(args)

    assert manifest["passed"] is True
    assert manifest["policy_kind"] == "mappo"


def test_root_entry_mappo_missing_checkpoint_rejected():
    out = Path("artifacts/step28_root_a_family_bridge_test_fail")

    args = make_args(
        output_root=str(out),
        policy_kind="mappo",
        checkpoint_path="",
        require_existing_checkpoint=False,
        self_test=True,
    )

    try:
        run_entry(args)
    except FileNotFoundError:
        return

    raise AssertionError("missing MAPPO checkpoint_path was not rejected")


def main():
    tests = [
        test_root_entry_placeholder_self_test,
        test_root_entry_mappo_without_existing_checkpoint_check,
        test_root_entry_mappo_missing_checkpoint_rejected,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] run_causal_rollout_a_family_bridge_v1 self-test passed")


if __name__ == "__main__":
    main()
