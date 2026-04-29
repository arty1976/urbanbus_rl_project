#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-test for Step 108 queue/demand reward wiring scaffold."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from queue_demand_reward_wiring_step108 import (
    REQUIRED_12_KPIS,
    REWARD_COMPONENT_COLUMNS,
    Step108Paths,
    run_step108,
)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def build_synthetic_input() -> pd.DataFrame:
    rows = []
    for condition in ["A", "A90"]:
        for seed in [1, 2]:
            for route_idx in [0, 1]:
                service_rate = 0.78 if condition == "A" else 0.72
                fleet_reduction = 0.0 if condition == "A" else 0.10
                rows.append({
                    "condition_id": condition,
                    "seed": seed,
                    "window_id": f"{condition}_{seed}_{route_idx}",
                    "state_ts": "2026-04-29T09:00:00+09:00",
                    "service_date": "2026-04-29",
                    "time_band": "peak",
                    "source_mode": "route_aware_queue_demand_scaffold_step106_non_causal",
                    "cv_headway": 0.12 + route_idx * 0.01,
                    "avg_wait_seconds": 300.0 + route_idx * 20 + seed,
                    "bunching_rate": 0.10 + route_idx * 0.02,
                    "on_time_rate": 0.70 - route_idx * 0.01,
                    "intervention_rate": 0.20,
                    "energy_proxy": 80.0 + route_idx,
                    "passenger_demand_generated": 100 + route_idx * 10,
                    "passenger_served_count": int((100 + route_idx * 10) * service_rate),
                    "passenger_service_rate": service_rate,
                    "passenger_wait_p95_seconds": 620.0 + route_idx * 30,
                    "energy_proxy_per_passenger": 1.1 + route_idx * 0.1,
                    "fleet_reduction_ratio": fleet_reduction,
                    "queue_demand_observed": False,
                    "queue_demand_proxy": True,
                    "actual_passenger_wait_observed": False,
                    "actual_headway_observed": False,
                    "actual_arrival_departure_time_observed": False,
                    "actual_dwell_observed": False,
                    "causal_comparison_allowed": False,
                    "paper_level_claim_allowed": False,
                    "causal_performance_claim_allowed": False,
                })
    return pd.DataFrame(rows)


def test_step108_success() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        input_root = root / "step107"
        output_root = root / "step108"
        input_path = input_root / "canonical_eval" / "kpi_by_window_queue_demand_preserved.parquet"
        df = build_synthetic_input()
        write_table(df, input_path)

        result = run_step108(Step108Paths(input_root=input_root, output_root=output_root))
        assert result["audit_status"] == "PASS"
        assert result["reward_status"] == "READY_FOR_STEP109_REWARD_POLICY_INTERFACE_SCAFFOLD"
        assert result["guards"]["reward_scaffold_only"] is True
        assert result["guards"]["final_reward_design_claim_allowed"] is False
        assert result["guards"]["causal_performance_claim_allowed"] is False

        reward_path = output_root / "reward_by_window_step108.parquet"
        if reward_path.exists():
            reward_df = pd.read_parquet(reward_path)
        else:
            reward_df = pd.read_csv(reward_path.with_suffix(".csv"))

        assert len(reward_df) == len(df)
        for col in REQUIRED_12_KPIS:
            assert col in reward_df.columns
        for col in REWARD_COMPONENT_COLUMNS:
            assert col in reward_df.columns
            assert pd.to_numeric(reward_df[col], errors="coerce").notna().all()
        assert bool(reward_df["reward_scaffold_only"].astype(bool).all())
        assert not bool(reward_df["final_reward_design_claim_allowed"].astype(bool).any())
        assert not bool(reward_df["paper_level_claim_allowed"].astype(bool).any())


def test_missing_kpi_blocks() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        input_root = root / "step107"
        output_root = root / "step108"
        input_path = input_root / "canonical_eval" / "kpi_by_window_queue_demand_preserved.parquet"
        df = build_synthetic_input().drop(columns=["passenger_wait_p95_seconds"])
        write_table(df, input_path)
        try:
            run_step108(Step108Paths(input_root=input_root, output_root=output_root))
        except RuntimeError as exc:
            assert "passenger_wait_p95_seconds" in str(exc)
        else:
            raise AssertionError("missing KPI should block Step 108")


def main() -> None:
    test_step108_success()
    test_missing_kpi_blocks()
    print("[OK] Step 108 queue/demand reward wiring self-test PASS")


if __name__ == "__main__":
    main()
