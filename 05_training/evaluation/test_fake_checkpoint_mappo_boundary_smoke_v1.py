"""
Self-test for Step 34 fake checkpoint MAPPO boundary smoke.

Run:
    python ./05_training/evaluation/test_fake_checkpoint_mappo_boundary_smoke_v1.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from run_fake_checkpoint_mappo_boundary_smoke_v1 import run_step34


def test_step34_fake_checkpoint_mappo_boundary_smoke() -> None:
    out = Path("artifacts/step34_fake_checkpoint_mappo_boundary_smoke_test")

    if out.exists():
        shutil.rmtree(out)

    manifest = run_step34(
        scenario_index=Path("artifacts/baseline_v1/B1_noop/scenario_index.parquet"),
        contract=Path("artifacts/baseline_v1/baseline_contract.json"),
        output_root=out,
        limit=2,
        seeds="1",
        write_parquet=True,
        smoke=True,
    )

    assert manifest["passed"] is True
    assert Path(manifest["fake_checkpoint"]).exists()
    assert manifest["boundary_summary"]["policy_sources"] == ["mappo_policy"]

    for source_mode in manifest["boundary_summary"]["source_modes"]:
        assert source_mode.startswith("causal_")
        assert "_mappo_policy_v1" in source_mode
        assert "placeholder" not in source_mode
        assert "stub" not in source_mode
        assert "smoke" not in source_mode

    for result in manifest["canonical_results"]:
        assert result["causal_comparison_allowed"] is True


def main() -> None:
    test_step34_fake_checkpoint_mappo_boundary_smoke()
    print("[PASS] test_step34_fake_checkpoint_mappo_boundary_smoke")
    print("[OK] Step 34 fake checkpoint MAPPO boundary smoke self-test passed")


if __name__ == "__main__":
    main()
