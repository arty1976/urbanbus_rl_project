"""
Self-test for Step 38 policy-interface canonical KPI smoke.

Run:
    python ./05_training/evaluation/test_policy_interface_canonical_smoke_v1.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from run_policy_interface_canonical_smoke_v1 import run_step38


def test_step38_policy_interface_canonical_smoke() -> None:
    out = Path("artifacts/step38_policy_interface_canonical_smoke_test")

    if out.exists():
        shutil.rmtree(out)

    manifest = run_step38(
        scenario_index=Path("artifacts/baseline_v1/B1_noop/scenario_index.parquet"),
        contract=Path("artifacts/baseline_v1/baseline_contract.json"),
        output_root=out,
        limit=2,
        seeds="1",
        write_parquet=True,
        smoke=True,
    )

    assert manifest["passed"] is True
    assert manifest["writer"]["field_summary"]["passed"] is True
    assert manifest["canonical_summary"]["passed"] is True

    assert (out / "step38_manifest.json").exists()
    assert (out / "rollout" / "A" / "canonical_eval" / "kpi_by_window.parquet").exists()
    assert (out / "rollout" / "A90" / "canonical_eval" / "kpi_by_window.parquet").exists()
    assert (out / "rollout" / "A80" / "canonical_eval" / "kpi_by_window.parquet").exists()
    assert (out / "rollout" / "A70" / "canonical_eval" / "kpi_by_window.parquet").exists()

    for item in manifest["canonical_summary"]["conditions"]:
        assert item["passed"] is True


def main() -> None:
    test_step38_policy_interface_canonical_smoke()
    print("[PASS] test_step38_policy_interface_canonical_smoke")
    print("[OK] Step 38 policy-interface canonical KPI smoke self-test passed")


if __name__ == "__main__":
    main()
