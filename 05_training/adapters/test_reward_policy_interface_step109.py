#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from reward_policy_interface_step109 import run_step109


def _make_step108_fixture(root: Path) -> None:
    rows = []
    for condition_id in ["A", "A90"]:
        for seed in [1, 2]:
            for route_idx in [0, 1]:
                rows.append({
                    "condition_id": condition_id,
                    "seed": seed,
                    "window_id": f"{condition_id}_s{seed}_r{route_idx}",
                    "reward_service": 1.0,
                    "reward_wait": -0.2,
                    "reward_long_wait": -0.1,
                    "reward_energy": -0.05,
                    "reward_fleet": 0.02,
                    "reward_total": 0.67,
                    "passenger_service_rate": 0.8,
                    "passenger_wait_p95_seconds": 420.0,
                    "energy_proxy_per_passenger": 1.2,
                    "fleet_reduction_ratio": 0.1 if condition_id == "A90" else 0.0,
                    "passenger_demand_generated": 100,
                    "passenger_served_count": 80,
                    "queue_demand_proxy": True,
                    "queue_demand_observed": False,
                    "actual_passenger_wait_observed": False,
                    "actual_headway_observed": False,
                    "actual_arrival_departure_time_observed": False,
                    "actual_dwell_observed": False,
                    "reward_scaffold_only": True,
                    "reward_weights_are_final": False,
                    "reward_formula_finalized": False,
                    "final_reward_design_claim_allowed": False,
                    "paper_level_claim_allowed": False,
                    "causal_performance_claim_allowed": False,
                    "causal_comparison_allowed": False,
                })

    df = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(root / "reward_by_window_step108.parquet", index=False)
    except Exception:
        df.to_csv(root / "reward_by_window_step108.csv", index=False, encoding="utf-8-sig")


def test_step109_selftest() -> None:
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        input_root = base / "step108"
        output_root = base / "out"
        _make_step108_fixture(input_root)

        manifest = run_step109(input_root=input_root, output_root=output_root, clean_output=True)

        assert manifest["audit_status"] == "PASS"
        assert manifest["summary"]["row_count"] == 8
        assert manifest["claim_guards"]["reward_scaffold_only"] is True
        assert manifest["claim_guards"]["train_with_this_reward_allowed"] is False
        assert manifest["claim_guards"]["final_reward_design_claim_allowed"] is False
        assert manifest["claim_guards"]["causal_performance_claim_allowed"] is False
        assert (output_root / "policy_reward_interface_step109.csv").exists()
        assert (output_root / "policy_reward_interface_schema_step109.json").exists()


def main() -> int:
    test_step109_selftest()
    print("[OK] Step 109 reward-policy interface scaffold self-test PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
