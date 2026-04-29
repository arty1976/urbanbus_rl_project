#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-test for Step 99-E BIS API integration audit.

The test creates small synthetic CSV files in a temporary directory:
- getBs02: 3 route-stop rows
- getPos02: 2 rows, 1 match and 1 mismatch
- getRealtime02 ETA: 2 rows, 1 match and 1 mismatch
- getRealtime02 headway candidate: 1 matching row

No API key, DB, or external artifact is required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from analyze_bis_api_integration_step99e import AuditPaths, run_audit


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected={expected!r}, actual={actual!r}")


def assert_almost(actual: float, expected: float, message: str, eps: float = 1e-9) -> None:
    if abs(float(actual) - float(expected)) > eps:
        raise AssertionError(f"{message}: expected={expected!r}, actual={actual!r}")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        getbs = root / "getbs02_route_stop_sequence_normalized.csv"
        getpos = root / "getpos02_trajectory_candidate_timeseries.csv"
        getpos_summary = root / "getpos02_trajectory_candidate_summary.csv"
        eta = root / "getrealtime02_eta_normalized.csv"
        headway = root / "getrealtime02_headway_candidate.csv"
        out = root / "out"

        write_csv(
            getbs,
            [
                {
                    "route_id": "R1",
                    "direction_id": "1",
                    "stop_id": "S1",
                    "ordered_stop_sequence": 1,
                    "route_no": "101",
                    "stop_name": "Alpha",
                },
                {
                    "route_id": "R1",
                    "direction_id": "1",
                    "stop_id": "S2",
                    "ordered_stop_sequence": 2,
                    "route_no": "101",
                    "stop_name": "Beta",
                },
                {
                    "route_id": "R2",
                    "direction_id": "0",
                    "stop_id": "S9",
                    "ordered_stop_sequence": 1,
                    "route_no": "202",
                    "stop_name": "Gamma",
                },
            ],
        )

        write_csv(
            getpos,
            [
                {
                    "sample_index": 0,
                    "route_id": "R1",
                    "direction_id": "1",
                    "bus_id_candidate": "BUS-1",
                    "current_stop_id": "S1",
                    "current_stop_order": 1,
                    "x_pos": 1.0,
                    "y_pos": 2.0,
                    "event_time_raw": "2026-04-29T09:00:00+09:00",
                },
                {
                    "sample_index": 1,
                    "route_id": "R1",
                    "direction_id": "1",
                    "bus_id_candidate": "BUS-2",
                    "current_stop_id": "S404",
                    "current_stop_order": 4,
                    "x_pos": 3.0,
                    "y_pos": 4.0,
                    "event_time_raw": "2026-04-29T09:01:00+09:00",
                },
            ],
        )

        write_csv(
            getpos_summary,
            [
                {"vehicle_group_count": 2, "repeated_vehicle_count": 1, "trajectory_candidate_count": 1}
            ],
        )

        write_csv(
            eta,
            [
                {"route_id": "R1", "direction_id": "1", "stop_id": "S2", "eta_seconds": 120},
                {"route_id": "R1", "direction_id": "1", "stop_id": "S404", "eta_seconds": 300},
            ],
        )

        write_csv(
            headway,
            [
                {"route_id": "R1", "direction_id": "1", "stop_id": "S1", "eta_headway_seconds": 180}
            ],
        )

        report = run_audit(
            AuditPaths(
                getbs02_path=getbs,
                getpos02_timeseries_path=getpos,
                getpos02_summary_path=getpos_summary,
                getrealtime02_eta_path=eta,
                getrealtime02_headway_path=headway,
                output_root=out,
            ),
            min_match_coverage=0.50,
            min_exact_order_coverage=0.50,
        )

        assert_equal(report["getbs02_base"]["raw_route_stop_rows"], 3, "getBs02 row count")
        assert_equal(report["getpos02_to_getbs02"]["row_count"], 2, "getPos02 row count")
        assert_equal(report["getpos02_to_getbs02"]["matched_rows"], 1, "getPos02 matched rows")
        assert_almost(report["getpos02_to_getbs02"]["match_coverage"], 0.5, "getPos02 coverage")

        assert_equal(report["getpos02_order_match"]["exact_order_match_count"], 1, "exact order matches")
        assert_equal(report["getpos02_order_match"]["nonzero_order_delta_count"], 0, "nonzero order deltas")
        assert_equal(report["getpos02_order_match"]["missing_route_stop_match_count"], 1, "missing route-stop matches")

        assert_equal(report["getrealtime02_to_getbs02"]["row_count"], 2, "ETA row count")
        assert_equal(report["getrealtime02_to_getbs02"]["matched_rows"], 1, "ETA matched rows")
        assert_almost(report["getrealtime02_to_getbs02"]["match_coverage"], 0.5, "ETA coverage")

        assert_equal(report["headway_candidate_to_getbs02"]["row_count"], 1, "headway row count")
        assert_equal(report["headway_candidate_to_getbs02"]["matched_rows"], 1, "headway matched rows")
        assert_almost(report["headway_candidate_to_getbs02"]["match_coverage"], 1.0, "headway coverage")

        assert_equal(report["classification_update"]["actual_headway"], "not_observed", "actual_headway guard")
        assert_equal(
            report["claim_guards"]["additional_api_calls_performed"],
            False,
            "additional API call guard",
        )

        expected_files = [
            "bis_api_integration_report.json",
            "bis_api_integration_report.md",
            "getpos02_to_getbs02_match.csv",
            "getrealtime02_to_getbs02_match.csv",
            "headway_candidate_to_getbs02_match.csv",
            "classification_update_step99e.json",
        ]
        for name in expected_files:
            if not (out / name).exists():
                raise AssertionError(f"expected output file missing: {out / name}")

        payload = json.loads((out / "classification_update_step99e.json").read_text(encoding="utf-8"))
        assert_equal(payload["claim_guards"]["paper_level_claim_allowed"], False, "paper claim guard")
        assert_equal(payload["claim_guards"]["causal_performance_claim_allowed"], False, "causal claim guard")

        print("[OK] Step 99-E BIS API integration audit self-test PASS")
        print(f"[OK] synthetic output root: {out}")


if __name__ == "__main__":
    main()
