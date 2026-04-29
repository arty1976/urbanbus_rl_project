#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-test for Step 100 causal simulator v2 contract update."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from build_causal_simulator_v2_contract_step100 import (
    BLOCKED_ACTUAL_OBSERVED_FIELDS,
    CANONICAL_12_KPIS,
    REQUIRED_ROUTE_AWARE_FIELDS,
    Step100Paths,
    run_step100,
)


def dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def synthetic_patch() -> dict:
    return {
        "artifact_version": "step98_causal_simulator_v2_contract_patch_from_step99f_v1",
        "minimum_route_aware_v2_required_fields": {
            "route_id": "observed_candidate_full_collection_234_of_238_routes",
            "direction_id": "observed_candidate_full_collection_234_of_238_routes_cross_confirmed",
            "ordered_stop_sequence": "observed_candidate_full_collection_234_of_238_routes",
            "node_uid": "observed_from_tensor_db",
            "node_index": "observed_from_tensor_db",
            "edge_index": "observed_from_tensor_db",
            "edge_distance_m": "observed_from_tensor_db",
            "edge_time_sec": "observed_from_tensor_db",
            "boardings_recent": "observed_proxy_from_tensor_db",
            "alightings_recent": "observed_proxy_from_tensor_db",
            "waiting_passenger_cnt": "observed_proxy_from_tensor_db",
            "hour_sin_cos": "observed_from_tensor_db",
            "is_peak": "observed_from_tensor_db",
        },
        "optional_api_candidate_fields": {
            "bus_id_or_vehicle_no": "repeated_sample_observed_candidate_from_getPos02_vhcNo2",
            "live_position_xy": "repeated_sample_observed_candidate_from_getPos02_xPos_yPos",
            "current_route_sequence": "repeated_sample_observed_candidate_from_getPos02_seq",
            "current_stop_id": "repeated_sample_observed_candidate_from_getPos02_bsId",
            "vehicle_trajectory": "trajectory_candidate_from_repeated_getPos02_sampling",
            "getRealtime02_eta": "observed_candidate_cross_matched_to_getBs02",
            "eta_based_headway": "candidate_from_getRealtime02_cross_matched_to_getBs02",
        },
        "proxy_or_estimated_only_fields": {
            "passenger_wait_age_distribution": "missing",
            "actual_passenger_wait_p95_seconds": "not_observed",
            "actual_passenger_service_rate": "not_observed",
        },
        "blocked_actual_observed_fields": {
            # Intentionally wrong to verify Step 100 hard guard.
            "actual_headway": "observed_by_mistake",
            "actual_arrival_departure_time": "not_observed",
            "actual_dwell": "not_observed",
            "vehicle_load": "missing",
            "left_behind_passengers": "missing",
        },
        "contract_rules": {
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    }


def synthetic_report() -> dict:
    return {
        "artifact_version": "bis_api_classification_consolidation_step99f_v1",
        "audit_status": "PASS",
        "upstream_step99e_integration_summary": {
            "getpos02_to_getbs02": {"match_coverage": 1.0, "row_count": 1452, "matched_rows": 1452},
            "getrealtime02_to_getbs02": {"match_coverage": 1.0, "row_count": 58, "matched_rows": 58},
            "headway_candidate_to_getbs02": {"match_coverage": 1.0, "row_count": 23, "matched_rows": 23},
        },
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        patch_path = root / "step98_causal_simulator_v2_contract_patch.json"
        report_path = root / "classification_update_consolidated_step99f.json"
        output_root = root / "out"
        dump(patch_path, synthetic_patch())
        dump(report_path, synthetic_report())

        contract = run_step100(
            Step100Paths(
                step99f_patch_path=patch_path,
                step99f_report_path=report_path,
                output_root=output_root,
            )
        )

        assert contract["contract_status"] == "READY_WITH_WARNINGS", contract["contract_status"]
        assert contract["claim_guards"]["paper_level_claim_allowed"] is False
        assert contract["claim_guards"]["causal_performance_claim_allowed"] is False
        assert contract["transition_contract"]["headway_model"]["actual_headway_observed"] is False
        assert "eta_based_headway" in contract["field_groups"]["optional_api_candidate_inputs"]
        assert contract["field_groups"]["blocked_actual_observed_inputs"]["actual_headway"] == "not_observed"
        assert set(REQUIRED_ROUTE_AWARE_FIELDS).issubset(contract["field_groups"]["required_route_aware_inputs"].keys())
        assert contract["kpi_contract"]["shared_kpis"] == CANONICAL_12_KPIS
        assert set(BLOCKED_ACTUAL_OBSERVED_FIELDS).issubset(contract["field_groups"]["blocked_actual_observed_inputs"].keys())

        for filename in [
            "causal_simulator_v2_contract.json",
            "causal_simulator_v2_contract.md",
            "causal_simulator_v2_field_readiness_matrix.csv",
            "causal_simulator_v2_contract_manifest.json",
        ]:
            assert (output_root / filename).exists(), filename

    print("[OK] Step 100 causal simulator v2 contract self-test PASS")


if __name__ == "__main__":
    main()
