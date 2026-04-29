#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-test for Step 106 minimal queue/demand scaffold."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from minimal_queue_demand_step106 import run_step106


def make_synthetic_step102(root: Path) -> tuple[Path, Path]:
    raw_rows = []
    for condition in ["A", "A70"]:
        for seed in [1, 2]:
            window_id = f"w_{condition}_{seed}"
            for step in range(1, 5):
                for agent_id in range(2):
                    raw_rows.append({
                        "artifact_version": "synthetic_step102",
                        "condition_id": condition,
                        "seed": seed,
                        "window_id": window_id,
                        "scenario_index": 0,
                        "state_ts": "2026-04-29T09:00:00+09:00",
                        "service_date": "2026-04-29",
                        "time_band": "peak",
                        "evaluation_horizon_minutes": 30,
                        "step_index": step,
                        "agent_id": agent_id,
                        "route_id": "R1",
                        "direction_id": "1",
                        "route_stop_count": 6,
                        "stop_index": min(5, step + agent_id),
                        "stop_order": min(6, step + agent_id + 1),
                        "stop_id": f"S{min(5, step + agent_id)}",
                        "next_stop_id": f"S{min(5, step + agent_id + 1)}",
                        "at_terminal": step == 4,
                        "action": "hold" if (step == 2 and agent_id == 0) else "advance",
                        "reward": -1.0,
                        "terminated": step == 4,
                        "truncated": False,
                        "source_mode": "route_aware_minimal_scaffold_step102_noncausal",
                        "paper_level_claim_allowed": False,
                        "causal_performance_claim_allowed": False,
                    })
    raw = pd.DataFrame(raw_rows)

    rollup_rows = []
    for condition in ["A", "A70"]:
        for seed in [1, 2]:
            window_id = f"w_{condition}_{seed}"
            active = 2 if condition == "A" else 1
            rollup_rows.append({
                "condition_id": condition,
                "seed": seed,
                "window_id": window_id,
                "state_ts": "2026-04-29T09:00:00+09:00",
                "service_date": "2026-04-29",
                "time_band": "peak",
                "evaluation_horizon_minutes": 30,
                "qwen_trigger_rate": 0.0,
                "effective_replay_step_minutes": 1.0,
                "headway_mean_seconds": 120.0,
                "headway_std_seconds": 30.0,
                "headway_sample_count": 4,
                "bunching_event_count": 1,
                "headway_event_count": 4,
                "wait_total_passenger_seconds": 2400.0,
                "wait_passenger_count": 20,
                "ontime_event_count": 3,
                "schedulable_arrival_count": 4,
                "intervention_count": 1,
                "decision_step_count": 8,
                "energy_proxy_total": 10.0,
                "source_mode": "route_aware_minimal_scaffold_step102_noncausal",
                "cv_headway": 0.25,
                "avg_wait_seconds": 120.0,
                "bunching_rate": 0.25,
                "on_time_rate": 0.75,
                "intervention_rate": 0.125,
                "energy_proxy": 10.0,
                "passenger_demand_generated": 20,
                "passenger_served_count": 10,
                "passenger_service_rate": 0.5,
                "passenger_wait_p95_seconds": 180.0,
                "energy_proxy_per_passenger": 1.0,
                "fleet_reduction_ratio": 0.0 if condition == "A" else 0.5,
                "route_id": "R1",
                "direction_id": "1",
                "route_stop_count": 6,
                "baseline_bus_count": 2,
                "active_bus_count": active,
                "strict_canonical": False,
                "causal_comparison_allowed": False,
                "scaffold_only": True,
                "actual_headway_observed": False,
                "actual_arrival_departure_time_observed": False,
                "actual_dwell_observed": False,
            })
    rollup = pd.DataFrame(rollup_rows)

    raw_path = root / "raw_events.csv"
    rollup_path = root / "window_rollup.csv"
    raw.to_csv(raw_path, index=False, encoding="utf-8")
    rollup.to_csv(rollup_path, index=False, encoding="utf-8")
    return raw_path, rollup_path


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        raw_path, rollup_path = make_synthetic_step102(root)
        out = root / "out"
        manifest = run_step106(
            raw_events_path=raw_path,
            window_rollup_path=rollup_path,
            output_root=out,
        )

        assert manifest["audit_status"] == "PASS"
        assert manifest["scaffold_status"] == "READY_FOR_STEP107_QUEUE_DEMAND_CANONICAL_KPI"
        assert manifest["row_counts"]["raw_event_rows"] == 32
        assert manifest["row_counts"]["window_rollup_rows"] == 4
        assert manifest["queue_summary"]["passenger_demand_generated_total"] > 0
        assert manifest["queue_summary"]["passenger_served_count_total"] >= 0
        assert manifest["claim_guards"]["causal_performance_claim_allowed"] is False
        assert manifest["claim_guards"]["queue_demand_observed"] is False
        assert manifest["claim_guards"]["queue_demand_proxy"] is True

        rollup_out = pd.read_csv(out / "window_rollup_queue_demand.csv")
        assert "passenger_service_rate" in rollup_out.columns
        assert "passenger_wait_p95_seconds" in rollup_out.columns
        assert "max_queue_depth" in rollup_out.columns
        assert bool((rollup_out["passenger_service_rate"] >= 0).all())
        assert bool((rollup_out["passenger_service_rate"] <= 1).all())
        assert not bool(rollup_out["causal_performance_claim_allowed"].any())

        readiness = pd.read_csv(out / "queue_demand_readiness_step106.csv")
        assert bool(readiness["passed"].astype(bool).all())

    print("[OK] Step 106 minimal queue/demand scaffold self-test PASS")


if __name__ == "__main__":
    main()
