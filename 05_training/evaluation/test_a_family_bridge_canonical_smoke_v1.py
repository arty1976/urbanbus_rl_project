"""
Self-test for Step 30 A-family bridge canonical smoke.

Run:
    python ./05_training/evaluation\test_a_family_bridge_canonical_smoke_v1.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from run_a_family_bridge_canonical_smoke_v1 import run_step30


def test_step30_placeholder_smoke() -> None:
    out = Path("artifacts/step30_a_family_bridge_canonical_smoke_test")

    if out.exists():
        shutil.rmtree(out)

    manifest = run_step30(
        scenario_index=Path("artifacts/baseline_v1/B1_noop/scenario_index.parquet"),
        contract=Path("artifacts/baseline_v1/baseline_contract.json"),
        output_root=out,
        limit=2,
        seeds="1",
        policy_kind="placeholder",
        checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
        write_parquet=True,
        smoke=True,
    )

    assert manifest["passed"] is True
    assert (out / "step30_manifest.json").exists()
    assert (out / "rollout" / "A" / "canonical_eval" / "kpi_by_window.parquet").exists()
    assert (out / "rollout" / "A90" / "canonical_eval" / "kpi_by_window.parquet").exists()
    assert (out / "rollout" / "A80" / "canonical_eval" / "kpi_by_window.parquet").exists()
    assert (out / "rollout" / "A70" / "canonical_eval" / "kpi_by_window.parquet").exists()


def main() -> None:
    tests = [
        test_step30_placeholder_smoke,
    ]

    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")

    print("[OK] Step 30 A-family bridge canonical smoke self-test passed")


if __name__ == "__main__":
    main()
