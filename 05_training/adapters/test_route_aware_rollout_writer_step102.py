#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-test for Step 102 route-aware rollout writer scaffold."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from route_aware_rollout_writer_step102 import (
    Step102Paths,
    run_step102,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_synthetic_route_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"route_id": "R1", "direction_id": "0", "stop_id": "S1", "ordered_stop_sequence": 1, "route_no": "101", "stop_name": "Alpha"},
        {"route_id": "R1", "direction_id": "0", "stop_id": "S2", "ordered_stop_sequence": 2, "route_no": "101", "stop_name": "Beta"},
        {"route_id": "R1", "direction_id": "0", "stop_id": "S3", "ordered_stop_sequence": 3, "route_no": "101", "stop_name": "Gamma"},
        {"route_id": "R1", "direction_id": "0", "stop_id": "S4", "ordered_stop_sequence": 4, "route_no": "101", "stop_name": "Delta"},
        {"route_id": "R2", "direction_id": "1", "stop_id": "S5", "ordered_stop_sequence": 1, "route_no": "202", "stop_name": "One"},
        {"route_id": "R2", "direction_id": "1", "stop_id": "S6", "ordered_stop_sequence": 2, "route_no": "202", "stop_name": "Two"},
        {"route_id": "R2", "direction_id": "1", "stop_id": "S7", "ordered_stop_sequence": 3, "route_no": "202", "stop_name": "Three"},
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        contract_path = root / "contract.json"
        route_csv = root / "getbs02.csv"
        out = root / "out"

        write_json(contract_path, {
            "artifact_version": "synthetic_step100_contract",
            "contract_status": "READY_FOR_ROUTE_AWARE_MINIMAL_SCAFFOLD",
            "claim_guards": {
                "paper_level_claim_allowed": False,
                "causal_performance_claim_allowed": False,
            },
            "blocked_actual_observed_fields": [
                "actual_headway",
                "actual_arrival_departure_time",
                "actual_dwell",
            ],
        })
        write_synthetic_route_csv(route_csv)

        manifest = run_step102(
            Step102Paths(contract_json=contract_path, route_sequence_csv=route_csv, output_root=out),
            conditions=["A", "A90"],
            seeds=[1, 2],
            max_routes=2,
            baseline_bus_count=3,
            max_steps=5,
            service_date="2023-01-01",
            state_ts="2023-01-01T05:00:00+09:00",
            time_band="offpeak",
            evaluation_horizon_minutes=30,
        )

        assert manifest["audit_status"] == "PASS"
        assert manifest["row_counts"]["scenarios"] == 8
        assert manifest["row_counts"]["window_rollup"] == 8
        assert manifest["row_counts"]["raw_events"] > 0
        assert manifest["claim_guards"]["paper_level_claim_allowed"] is False
        assert manifest["claim_guards"]["causal_performance_claim_allowed"] is False
        assert (out / "raw_events.csv").exists()
        assert (out / "window_rollup.csv").exists()
        assert (out / "route_aware_rollout_writer_manifest.json").exists()
        assert (out / "route_aware_rollout_writer_report.md").exists()

        rollup_rows = read_csv_rows(out / "window_rollup.csv")
        assert len(rollup_rows) == 8
        for row in rollup_rows:
            assert row["source_mode"] == "route_aware_minimal_scaffold_step102_noncausal"
            assert row["not_actual_headway"] == "True"
            assert row["actual_headway_observed"] == "False"
            assert row["actual_arrival_departure_time_observed"] == "False"
            assert row["actual_dwell_observed"] == "False"
            assert row["paper_level_claim_allowed"] == "False"
            assert row["causal_performance_claim_allowed"] == "False"
            for col in [
                "cv_headway",
                "avg_wait_seconds",
                "bunching_rate",
                "on_time_rate",
                "intervention_rate",
                "energy_proxy",
                "passenger_demand_generated",
                "passenger_served_count",
                "passenger_service_rate",
                "passenger_wait_p95_seconds",
                "energy_proxy_per_passenger",
                "fleet_reduction_ratio",
            ]:
                assert col in row

        print("[OK] Step 102 route-aware rollout writer scaffold self-test PASS")
        print(f"[OK] synthetic output root: {out}")


if __name__ == "__main__":
    main()
