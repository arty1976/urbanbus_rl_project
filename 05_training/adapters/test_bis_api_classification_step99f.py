#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Self-test for Step 99-F classification consolidation.

The test creates synthetic Step 99-E JSON outputs, runs the consolidation,
and verifies that:
- route-aware fields are upgraded.
- candidate fields remain labeled as candidates.
- actual headway/arrival/dwell remain not_observed.
- paper/causal claim guards remain false.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from consolidate_bis_api_classification_step99f import ConsolidationPaths, run_consolidation


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def assert_equal(actual, expected, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected={expected!r}, actual={actual!r}")


def assert_true(value, message: str) -> None:
    if not value:
        raise AssertionError(message)


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        update_path = root / "classification_update_step99e.json"
        report_path = root / "bis_api_integration_report.json"
        out = root / "out"

        classification_update = {
            "route_id": "observed_candidate_full_collection_234_of_238_routes",
            "direction_id": "observed_candidate_full_collection_234_of_238_routes_cross_confirmed",
            "ordered_stop_sequence": "observed_candidate_full_collection_234_of_238_routes",
            "bus_id_or_vehicle_no": "repeated_sample_observed_candidate_from_getPos02_vhcNo2",
            "live_position_xy": "repeated_sample_observed_candidate_from_getPos02_xPos_yPos",
            "current_route_sequence": "repeated_sample_observed_candidate_from_getPos02_seq",
            "current_stop_id": "repeated_sample_observed_candidate_from_getPos02_bsId",
            "vehicle_trajectory": "trajectory_candidate_from_repeated_getPos02_sampling",
            "getRealtime02_eta": "observed_candidate_cross_matched_to_getBs02",
            "eta_based_headway": "candidate_from_getRealtime02_cross_matched_to_getBs02",
            # Deliberately try to over-upgrade these; Step 99-F must force them back.
            "actual_headway": "observed",
            "actual_arrival_departure_time": "observed",
            "actual_dwell": "observed",
        }

        write_json(
            update_path,
            {
                "artifact_version": "bis_api_integration_step99e_v1",
                "step": "99-E",
                "classification_update": classification_update,
                "claim_guards": {
                    "db_write_performed": False,
                    "tensor_db_overwrite_performed": False,
                    "additional_api_calls_performed": False,
                    "paper_level_claim_allowed": False,
                    "causal_performance_claim_allowed": False,
                },
                "source_evidence": {
                    "getBs02": {"success_routes": 234, "auth_error_routes": 4},
                    "getPos02": {"trajectory_candidate_count": 248},
                    "getRealtime02": {"normalized_eta_rows_expected_from_step99d": 58},
                },
            },
        )

        write_json(
            report_path,
            {
                "artifact_version": "bis_api_integration_step99e_v1",
                "step": "99-E",
                "audit_status": "PASS",
                "claim_guards": {
                    "db_write_performed": False,
                    "tensor_db_overwrite_performed": False,
                    "additional_api_calls_performed": False,
                    "paper_level_claim_allowed": False,
                    "causal_performance_claim_allowed": False,
                },
                "getpos02_to_getbs02": {"row_count": 1452, "matched_rows": 1452, "missing_match_rows": 0, "match_coverage": 1.0},
                "getrealtime02_to_getbs02": {"row_count": 58, "matched_rows": 58, "missing_match_rows": 0, "match_coverage": 1.0},
                "headway_candidate_to_getbs02": {"row_count": 23, "matched_rows": 23, "missing_match_rows": 0, "match_coverage": 1.0},
                "route_direction_consistency": {
                    "getbs02_route_direction_count": 234,
                    "getpos02_route_direction_count": 12,
                    "getrealtime02_route_direction_count": 6,
                    "intersection_all_three_count": 6,
                },
                "source_row_counts": {
                    "getbs02_raw": 20508,
                    "getpos02_timeseries_raw": 1452,
                    "getrealtime02_eta_raw": 58,
                    "getrealtime02_headway_raw": 23,
                },
            },
        )

        result = run_consolidation(
            ConsolidationPaths(
                step99e_update_path=update_path,
                step99e_report_path=report_path,
                step97_report_path=None,
                output_root=out,
            )
        )

        assert_equal(result["audit_status"], "PASS", "audit status")
        final = result["final_classification"]
        assert_equal(final["route_id"], "observed_candidate_full_collection_234_of_238_routes", "route_id upgrade")
        assert_equal(final["direction_id"], "observed_candidate_full_collection_234_of_238_routes_cross_confirmed", "direction_id upgrade")
        assert_equal(final["ordered_stop_sequence"], "observed_candidate_full_collection_234_of_238_routes", "sequence upgrade")
        assert_equal(final["eta_based_headway"], "candidate_from_getRealtime02_cross_matched_to_getBs02", "eta headway candidate")
        assert_equal(final["actual_headway"], "not_observed", "actual_headway hard guard")
        assert_equal(final["actual_arrival_departure_time"], "not_observed", "arrival/departure hard guard")
        assert_equal(final["actual_dwell"], "not_observed", "dwell hard guard")
        assert_equal(result["claim_guards"]["paper_level_claim_allowed"], False, "paper claim guard")
        assert_equal(result["claim_guards"]["causal_performance_claim_allowed"], False, "causal claim guard")

        patch = result["step98_causal_simulator_v2_contract_patch"]
        assert_true("route_id" in patch["minimum_route_aware_v2_required_fields"], "contract patch route_id")
        assert_true("eta_based_headway" in patch["optional_api_candidate_fields"], "contract patch ETA candidate")
        assert_equal(
            patch["contract_rules"]["actual_headway_must_remain_not_observed"],
            True,
            "contract hard rule for actual_headway",
        )

        for name in [
            "classification_update_consolidated_step99f.json",
            "classification_update_consolidated_step99f.md",
            "field_classification_matrix_step99f.csv",
            "step98_causal_simulator_v2_contract_patch.json",
        ]:
            assert_true((out / name).exists(), f"expected output file missing: {name}")

        written = json.loads((out / "classification_update_consolidated_step99f.json").read_text(encoding="utf-8"))
        assert_equal(written["audit_status"], "PASS", "written report status")

        print("[OK] Step 99-F classification consolidation self-test PASS")
        print(f"[OK] synthetic output root: {out}")


if __name__ == "__main__":
    main()
