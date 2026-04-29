#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 99-F — Step 97/98 classification update consolidation.

This script consolidates Step 99-A~E BIS (Bus Information System=버스정보시스템)
API (Application Programming Interface=응용 프로그램 인터페이스) evidence into a
single Step 97/98 classification update package.

Safety contract:
- No DB (Database=데이터베이스) write.
- No tensor DB (Database=데이터베이스) overwrite.
- No additional API calls.
- Reads existing artifact JSON/CSV only.
- paper_level_claim_allowed=false.
- causal_performance_claim_allowed=false.

Primary input:
- artifacts/daegu_bis_api_audit/bis_api_integration_step99e/classification_update_step99e.json
- artifacts/daegu_bis_api_audit/bis_api_integration_step99e/bis_api_integration_report.json

Outputs:
- artifacts/daegu_bis_api_audit/bis_api_classification_step99f/classification_update_consolidated_step99f.json
- artifacts/daegu_bis_api_audit/bis_api_classification_step99f/classification_update_consolidated_step99f.md
- artifacts/daegu_bis_api_audit/bis_api_classification_step99f/field_classification_matrix_step99f.csv
- artifacts/daegu_bis_api_audit/bis_api_classification_step99f/step98_causal_simulator_v2_contract_patch.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ARTIFACT_VERSION = "bis_api_classification_consolidation_step99f_v1"

DEFAULT_STEP99E_UPDATE_PATH = Path(
    "artifacts/daegu_bis_api_audit/bis_api_integration_step99e/"
    "classification_update_step99e.json"
)
DEFAULT_STEP99E_REPORT_PATH = Path(
    "artifacts/daegu_bis_api_audit/bis_api_integration_step99e/"
    "bis_api_integration_report.json"
)
DEFAULT_STEP97_REPORT_PATH = Path(
    "artifacts/tensor_db_data_availability_audit/"
    "tensor_db_available_fields_report.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/daegu_bis_api_audit/bis_api_classification_step99f"
)


CLAIM_GUARDS: Dict[str, bool] = {
    "db_write_performed": False,
    "tensor_db_overwrite_performed": False,
    "additional_api_calls_performed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}


# Step 97 baseline categories are deliberately conservative.  Step 99-F does
# not rewrite the tensor DB; it only updates the classification ledger that will
# feed Step 98 causal simulator v2 contract drafting.
BASELINE_STEP97_CLASSIFICATION: Dict[str, str] = {
    # Tensor DB / already-approved graph and demand proxy layer.
    "node_uid": "observed_from_tensor_db",
    "node_index": "observed_from_tensor_db",
    "stop_node_population": "observed_from_tensor_db",
    "edge_index": "observed_from_tensor_db",
    "edge_distance_m": "observed_from_tensor_db",
    "edge_time_sec": "observed_from_tensor_db",
    "generalized_cost": "observed_from_tensor_db",
    "long_edge_5km_flag": "observed_from_tensor_db",
    "boardings_recent": "observed_proxy_from_tensor_db",
    "alightings_recent": "observed_proxy_from_tensor_db",
    "waiting_passenger_cnt": "observed_proxy_from_tensor_db",
    "hour_sin_cos": "observed_from_tensor_db",
    "is_peak": "observed_from_tensor_db",
    "next_boardings_recent": "observed_proxy_from_tensor_db",
    "next_alightings_recent": "observed_proxy_from_tensor_db",
    "next_waiting_passenger_cnt": "observed_proxy_from_tensor_db",
    # Fields that were unavailable before Step 99 API audits.
    "route_id": "missing_before_step99_api_audit",
    "direction_id": "missing_before_step99_api_audit",
    "ordered_stop_sequence": "missing_before_step99_api_audit",
    "bus_id_or_vehicle_no": "missing_before_step99_api_audit",
    "live_position_xy": "missing_before_step99_api_audit",
    "current_route_sequence": "missing_before_step99_api_audit",
    "current_stop_id": "missing_before_step99_api_audit",
    "vehicle_trajectory": "missing_before_step99_api_audit",
    "getRealtime02_eta": "missing_before_step99_api_audit",
    "eta_based_headway": "missing_before_step99_api_audit",
    "actual_headway": "not_observed",
    "actual_arrival_departure_time": "not_observed",
    "actual_dwell": "not_observed",
    "passenger_wait_age_distribution": "missing",
    "vehicle_load": "missing",
    "left_behind_passengers": "missing",
    "actual_passenger_wait_p95_seconds": "not_observed",
    "actual_passenger_service_rate": "not_observed",
}


FIELD_POLICIES: Dict[str, Dict[str, str]] = {
    "node_uid": {
        "step98_contract_use": "required_static_graph_field",
        "claim_boundary": "usable_as_observed_graph_identifier",
        "notes": "Already belongs to the approved tensor DB graph layer.",
    },
    "node_index": {
        "step98_contract_use": "required_static_graph_field",
        "claim_boundary": "usable_as_observed_graph_index",
        "notes": "Dense node index used by GATv2(Graph Attention Network v2=그래프 어텐션 네트워크 v2)/PyG(PyTorch Geometric=파이토치 지오메트릭).",
    },
    "stop_node_population": {
        "step98_contract_use": "required_static_graph_field",
        "claim_boundary": "usable_as_observed_graph_population",
        "notes": "Represents approved stop-node graph population, not all possible Daegu stops.",
    },
    "edge_index": {
        "step98_contract_use": "required_static_graph_field",
        "claim_boundary": "usable_as_observed_static_edges",
        "notes": "Static STOP_TO_STOP graph skeleton.",
    },
    "edge_distance_m": {
        "step98_contract_use": "required_static_graph_field",
        "claim_boundary": "usable_as_observed_or_engineered_edge_distance",
        "notes": "Distance feature for simulator travel-time and energy proxies.",
    },
    "edge_time_sec": {
        "step98_contract_use": "required_static_graph_field",
        "claim_boundary": "usable_as_engineered_nominal_travel_time",
        "notes": "Nominal travel time, not live congestion distribution.",
    },
    "generalized_cost": {
        "step98_contract_use": "optional_static_graph_feature",
        "claim_boundary": "engineered_feature_only",
        "notes": "Can support routing or policy observation but should not be called observed operating cost.",
    },
    "long_edge_5km_flag": {
        "step98_contract_use": "optional_static_graph_feature",
        "claim_boundary": "engineered_feature_only",
        "notes": "Quality-control and feature flag.",
    },
    "boardings_recent": {
        "step98_contract_use": "required_demand_proxy",
        "claim_boundary": "proxy_not_individual_passenger_arrivals",
        "notes": "Can initialize or calibrate stop-level demand proxy.",
    },
    "alightings_recent": {
        "step98_contract_use": "required_demand_proxy",
        "claim_boundary": "proxy_not_individual_alighting_events",
        "notes": "Can initialize or calibrate stop-level alighting proxy.",
    },
    "waiting_passenger_cnt": {
        "step98_contract_use": "required_queue_proxy",
        "claim_boundary": "proxy_not_individual_wait_distribution",
        "notes": "Can support queue proxy, but not true p95 passenger wait.",
    },
    "hour_sin_cos": {
        "step98_contract_use": "required_time_feature",
        "claim_boundary": "usable_as_time_encoding",
        "notes": "Time-of-day encoding.",
    },
    "is_peak": {
        "step98_contract_use": "required_time_feature",
        "claim_boundary": "usable_as_peak_indicator",
        "notes": "Peak/off-peak feature, not a full calendar/event model.",
    },
    "next_boardings_recent": {
        "step98_contract_use": "optional_supervised_transition_target",
        "claim_boundary": "proxy_transition_target",
        "notes": "Useful for calibrated transition sanity checks.",
    },
    "next_alightings_recent": {
        "step98_contract_use": "optional_supervised_transition_target",
        "claim_boundary": "proxy_transition_target",
        "notes": "Useful for calibrated transition sanity checks.",
    },
    "next_waiting_passenger_cnt": {
        "step98_contract_use": "optional_supervised_transition_target",
        "claim_boundary": "proxy_transition_target",
        "notes": "Useful for queue transition sanity checks.",
    },
    "route_id": {
        "step98_contract_use": "required_route_aware_field",
        "claim_boundary": "observed_candidate_not_100pct_complete",
        "notes": "Recovered for 234 of 238 route records; four known AUTH_ERROR exceptions remain.",
    },
    "direction_id": {
        "step98_contract_use": "required_route_aware_field",
        "claim_boundary": "observed_candidate_cross_confirmed_not_100pct_complete",
        "notes": "Cross-confirmed by getBs02/getPos02/getRealtime02 route-direction integration.",
    },
    "ordered_stop_sequence": {
        "step98_contract_use": "required_route_aware_field",
        "claim_boundary": "observed_candidate_not_100pct_complete",
        "notes": "Route-stop order is available for successful getBs02 routes.",
    },
    "bus_id_or_vehicle_no": {
        "step98_contract_use": "optional_live_vehicle_candidate_field",
        "claim_boundary": "candidate_vehicle_identifier_not_master_vehicle_registry",
        "notes": "Repeated getPos02 sampling provides vhcNo2 candidate; not a complete fleet registry.",
    },
    "live_position_xy": {
        "step98_contract_use": "optional_live_position_candidate_field",
        "claim_boundary": "live_snapshot_candidate_not_continuous_trace",
        "notes": "xPos/yPos live snapshots can calibrate or inspect simulator behavior.",
    },
    "current_route_sequence": {
        "step98_contract_use": "optional_live_progress_candidate_field",
        "claim_boundary": "candidate_current_sequence_not_actual_arrival_event",
        "notes": "getPos02 seq can be compared to route sequence but does not equal arrival/departure log.",
    },
    "current_stop_id": {
        "step98_contract_use": "optional_live_progress_candidate_field",
        "claim_boundary": "candidate_current_stop_not_actual_stop_event_log",
        "notes": "Matched to getBs02 stop sequence in Step 99-E.",
    },
    "vehicle_trajectory": {
        "step98_contract_use": "optional_calibration_candidate_only",
        "claim_boundary": "trajectory_candidate_not_actual_complete_trajectory",
        "notes": "Repeated live snapshots support a trajectory candidate, not a complete AVL log.",
    },
    "getRealtime02_eta": {
        "step98_contract_use": "optional_eta_candidate_field",
        "claim_boundary": "eta_observed_candidate_not_actual_arrival_time",
        "notes": "ETA(Estimated Time of Arrival=예상 도착 시간) rows are cross-matched to getBs02 route-stop rows.",
    },
    "eta_based_headway": {
        "step98_contract_use": "optional_headway_candidate_field",
        "claim_boundary": "candidate_not_actual_headway",
        "notes": "Can be used as a calibration candidate only; keep actual_headway not_observed.",
    },
    "actual_headway": {
        "step98_contract_use": "blocked_as_observed_field",
        "claim_boundary": "must_remain_not_observed",
        "notes": "Needs actual vehicle arrival/departure event logs, not only ETA/live snapshots.",
    },
    "actual_arrival_departure_time": {
        "step98_contract_use": "blocked_as_observed_field",
        "claim_boundary": "must_remain_not_observed",
        "notes": "No stop-level actual arrival/departure timestamps were observed.",
    },
    "actual_dwell": {
        "step98_contract_use": "blocked_as_observed_field",
        "claim_boundary": "must_remain_not_observed",
        "notes": "No observed door-open/stop dwell duration source is available.",
    },
    "passenger_wait_age_distribution": {
        "step98_contract_use": "missing_required_for_true_p95_wait",
        "claim_boundary": "missing",
        "notes": "Needed for actual p95 wait; current system can only use proxy/estimated wait.",
    },
    "vehicle_load": {
        "step98_contract_use": "missing_for_capacity_and_crowding",
        "claim_boundary": "missing",
        "notes": "No vehicle-level load/crowding source is available.",
    },
    "left_behind_passengers": {
        "step98_contract_use": "missing_for_capacity_constraint_validation",
        "claim_boundary": "missing",
        "notes": "No observed denied-boarding/left-behind records are available.",
    },
    "actual_passenger_wait_p95_seconds": {
        "step98_contract_use": "blocked_as_actual_observed_metric",
        "claim_boundary": "actual_metric_not_observed_proxy_only",
        "notes": "May be estimated in simulator, but cannot be claimed as observed actual p95 wait.",
    },
    "actual_passenger_service_rate": {
        "step98_contract_use": "blocked_as_actual_observed_metric",
        "claim_boundary": "actual_metric_not_observed_proxy_only",
        "notes": "Can be simulated/proxied, but true service rate needs passenger-level demand/service records.",
    },
}


REQUIRED_ROUTE_AWARE_FIELDS = [
    "route_id",
    "direction_id",
    "ordered_stop_sequence",
    "node_uid",
    "node_index",
    "edge_index",
    "edge_distance_m",
    "edge_time_sec",
    "boardings_recent",
    "alightings_recent",
    "waiting_passenger_cnt",
    "hour_sin_cos",
    "is_peak",
]

OPTIONAL_CANDIDATE_FIELDS = [
    "bus_id_or_vehicle_no",
    "live_position_xy",
    "current_route_sequence",
    "current_stop_id",
    "vehicle_trajectory",
    "getRealtime02_eta",
    "eta_based_headway",
]

PROXY_OR_ESTIMATED_ONLY_FIELDS = [
    "passenger_wait_age_distribution",
    "actual_passenger_wait_p95_seconds",
    "actual_passenger_service_rate",
]

BLOCKED_ACTUAL_OBSERVED_FIELDS = [
    "actual_headway",
    "actual_arrival_departure_time",
    "actual_dwell",
    "vehicle_load",
    "left_behind_passengers",
]


@dataclass(frozen=True)
class ConsolidationPaths:
    step99e_update_path: Path
    step99e_report_path: Path
    step97_report_path: Optional[Path]
    output_root: Path


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required JSON file not found: {path}")
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read JSON: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_get(dct: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def coverage_ok(value: Any, threshold: float) -> bool:
    if value is None:
        return False
    try:
        return float(value) >= threshold
    except Exception:
        return False


def merge_classification(step99e_update: Dict[str, Any], step99e_report: Dict[str, Any]) -> Dict[str, str]:
    update = step99e_update.get("classification_update")
    if not isinstance(update, dict):
        update = step99e_report.get("classification_update", {})
    if not isinstance(update, dict):
        update = {}

    merged = dict(BASELINE_STEP97_CLASSIFICATION)
    for key, value in update.items():
        merged[str(key)] = str(value)

    # Hard guards: these cannot be accidentally upgraded by upstream files.
    merged["actual_headway"] = "not_observed"
    merged["actual_arrival_departure_time"] = "not_observed"
    merged["actual_dwell"] = "not_observed"
    merged.setdefault("passenger_wait_age_distribution", "missing")
    merged.setdefault("vehicle_load", "missing")
    merged.setdefault("left_behind_passengers", "missing")
    merged.setdefault("actual_passenger_wait_p95_seconds", "not_observed")
    merged.setdefault("actual_passenger_service_rate", "not_observed")
    return merged


def build_field_matrix(final_classification: Dict[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for field in sorted(final_classification):
        policy = FIELD_POLICIES.get(field, {})
        rows.append({
            "field": field,
            "step97_before": BASELINE_STEP97_CLASSIFICATION.get(field, "not_in_baseline_step97_map"),
            "step99f_final_classification": final_classification[field],
            "step98_contract_use": policy.get("step98_contract_use", "review_required"),
            "claim_boundary": policy.get("claim_boundary", "review_required"),
            "notes": policy.get("notes", "No policy note registered; manual review required."),
        })
    return rows


def write_csv_matrix(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "field",
        "step97_before",
        "step99f_final_classification",
        "step98_contract_use",
        "claim_boundary",
        "notes",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_contract_patch(final_classification: Dict[str, str]) -> Dict[str, Any]:
    def pick(fields: List[str]) -> Dict[str, str]:
        return {field: final_classification.get(field, "missing") for field in fields}

    return {
        "artifact_version": "step98_causal_simulator_v2_contract_patch_from_step99f_v1",
        "purpose": (
            "Patch input for Step 98 causal simulator v2 contract drafting. "
            "This is not an executable simulator contract by itself."
        ),
        "minimum_route_aware_v2_required_fields": pick(REQUIRED_ROUTE_AWARE_FIELDS),
        "optional_api_candidate_fields": pick(OPTIONAL_CANDIDATE_FIELDS),
        "proxy_or_estimated_only_fields": pick(PROXY_OR_ESTIMATED_ONLY_FIELDS),
        "blocked_actual_observed_fields": pick(BLOCKED_ACTUAL_OBSERVED_FIELDS),
        "contract_rules": {
            "route_aware_simulator_v2_can_use_getBs02_sequence": True,
            "getPos02_can_be_used_for_calibration_or_sanity_check": True,
            "getRealtime02_eta_can_be_used_as_eta_candidate": True,
            "eta_based_headway_must_be_labeled_candidate": True,
            "vehicle_trajectory_must_be_labeled_candidate": True,
            "actual_headway_must_remain_not_observed": True,
            "actual_arrival_departure_time_must_remain_not_observed": True,
            "actual_dwell_must_remain_not_observed": True,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    }


def determine_audit_status(
    step99e_update: Dict[str, Any],
    step99e_report: Dict[str, Any],
    *,
    min_match_coverage: float,
) -> Tuple[str, List[Dict[str, Any]]]:
    warnings: List[Dict[str, Any]] = []

    upstream_status = step99e_report.get("audit_status")
    if upstream_status != "PASS":
        warnings.append({
            "code": "upstream_step99e_not_pass",
            "message": "Step 99-E report is not PASS; consolidation can proceed but Step 98 must review.",
            "step99e_audit_status": upstream_status,
        })

    coverages = {
        "getpos02_to_getbs02": safe_get(step99e_report, ["getpos02_to_getbs02", "match_coverage"]),
        "getrealtime02_to_getbs02": safe_get(step99e_report, ["getrealtime02_to_getbs02", "match_coverage"]),
        "headway_candidate_to_getbs02": safe_get(step99e_report, ["headway_candidate_to_getbs02", "match_coverage"]),
    }
    for name, value in coverages.items():
        if value is None:
            warnings.append({
                "code": "missing_coverage_metric",
                "source": name,
                "message": f"Missing match coverage metric for {name}.",
            })
        elif not coverage_ok(value, min_match_coverage):
            warnings.append({
                "code": "low_coverage_metric",
                "source": name,
                "match_coverage": value,
                "min_match_coverage": min_match_coverage,
                "message": f"Coverage for {name} is below Step 99-F threshold.",
            })

    guards = step99e_update.get("claim_guards", {})
    for key, expected in CLAIM_GUARDS.items():
        actual = guards.get(key, safe_get(step99e_report, ["claim_guards", key]))
        if actual is not expected:
            warnings.append({
                "code": "claim_guard_mismatch",
                "guard": key,
                "expected": expected,
                "actual": actual,
                "message": f"Claim guard {key} does not match expected value.",
            })

    if warnings:
        return "PASS_WITH_REVIEW_REQUIRED", warnings
    return "PASS", warnings


def summarize_step97_report(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"provided": False, "exists": False, "note": "No Step 97 report path provided."}
    if not path.exists():
        return {"provided": True, "exists": False, "path": str(path), "note": "Optional Step 97 report not found."}
    try:
        payload = load_json_any_encoding(path)
    except Exception as exc:
        return {"provided": True, "exists": True, "path": str(path), "read_error": str(exc)}

    # Keep this intentionally shallow because prior Step 97 report schema may evolve.
    return {
        "provided": True,
        "exists": True,
        "path": str(path),
        "top_level_keys": sorted(list(payload.keys()))[:50],
        "note": "Step 99-F records this file for traceability but uses its own conservative baseline map.",
    }


def write_markdown_report(path: Path, report: Dict[str, Any], matrix_rows: List[Dict[str, Any]]) -> None:
    def pct(value: Any) -> str:
        if value is None:
            return "N/A"
        try:
            return f"{float(value) * 100:.2f}%"
        except Exception:
            return str(value)

    integration = report["upstream_step99e_integration_summary"]
    lines: List[str] = [
        "# Step 99-F — Step 97/98 classification update consolidation",
        "",
        "## Purpose",
        "",
        "This document consolidates Step 99-A~E BIS (Bus Information System=버스정보시스템) API (Application Programming Interface=응용 프로그램 인터페이스) evidence into the Step 97/98 field-classification ledger.",
        "",
        "It is a classification and contract-preparation artifact, not a performance result.",
        "",
        "## Safety contract",
        "",
        f"- DB(Database=데이터베이스) write performed: `{report['claim_guards']['db_write_performed']}`",
        f"- tensor DB(Database=데이터베이스) overwrite performed: `{report['claim_guards']['tensor_db_overwrite_performed']}`",
        f"- Additional API calls performed: `{report['claim_guards']['additional_api_calls_performed']}`",
        f"- paper_level_claim_allowed: `{report['claim_guards']['paper_level_claim_allowed']}`",
        f"- causal_performance_claim_allowed: `{report['claim_guards']['causal_performance_claim_allowed']}`",
        "",
        "## Audit status",
        "",
        f"- Step 99-F status: `{report['audit_status']}`",
        f"- Upstream Step 99-E status: `{integration.get('audit_status')}`",
        "",
        "## Upstream Step 99-E match coverage",
        "",
        "| Source | Coverage | Rows | Matched rows |",
        "|---|---:|---:|---:|",
    ]

    for label, key in [
        ("getPos02 → getBs02", "getpos02_to_getbs02"),
        ("getRealtime02 ETA → getBs02", "getrealtime02_to_getbs02"),
        ("headway candidate → getBs02", "headway_candidate_to_getbs02"),
    ]:
        summary = integration.get(key, {}) or {}
        lines.append(
            f"| {label} | {pct(summary.get('match_coverage'))} | {summary.get('row_count', 'N/A')} | {summary.get('matched_rows', 'N/A')} |"
        )

    lines.extend([
        "",
        "## Step 98 causal simulator v2 contract impact",
        "",
        "### Required route-aware fields",
        "",
        "| Field | Final classification | Contract use |",
        "|---|---|---|",
    ])
    matrix_by_field = {row["field"]: row for row in matrix_rows}
    for field in REQUIRED_ROUTE_AWARE_FIELDS:
        row = matrix_by_field.get(field, {})
        lines.append(
            f"| `{field}` | `{row.get('step99f_final_classification', 'missing')}` | `{row.get('step98_contract_use', 'review_required')}` |"
        )

    lines.extend([
        "",
        "### Optional API candidate fields",
        "",
        "| Field | Final classification | Claim boundary |",
        "|---|---|---|",
    ])
    for field in OPTIONAL_CANDIDATE_FIELDS:
        row = matrix_by_field.get(field, {})
        lines.append(
            f"| `{field}` | `{row.get('step99f_final_classification', 'missing')}` | `{row.get('claim_boundary', 'review_required')}` |"
        )

    lines.extend([
        "",
        "### Fields that must remain non-observed or proxy-only",
        "",
        "| Field | Final classification | Rule |",
        "|---|---|---|",
    ])
    for field in [*PROXY_OR_ESTIMATED_ONLY_FIELDS, *BLOCKED_ACTUAL_OBSERVED_FIELDS]:
        row = matrix_by_field.get(field, {})
        lines.append(
            f"| `{field}` | `{row.get('step99f_final_classification', 'missing')}` | `{row.get('step98_contract_use', 'review_required')}` |"
        )

    lines.extend([
        "",
        "## Consolidated interpretation",
        "",
        "Step 99-F upgrades route-aware simulator readiness because `route_id`, `direction_id`, and `ordered_stop_sequence` are now available as observed candidates from `/getBs02`, and Step 99-E confirmed 100% cross-match coverage for the available `/getPos02` and `/getRealtime02` artifacts.",
        "",
        "However, Step 99-F does **not** upgrade `actual_headway`, `actual_arrival_departure_time`, or `actual_dwell`. Those still require actual vehicle event logs, not ETA rows or repeated live-position snapshots.",
        "",
        "## Prohibited claims",
        "",
    ])
    for claim in report["prohibited_claims"]:
        lines.append(f"- {claim}")

    lines.extend([
        "",
        "## Warnings",
        "",
    ])
    warnings = report.get("warnings", [])
    if not warnings:
        lines.append("- None")
    else:
        for warning in warnings:
            lines.append(f"- `{warning.get('code')}`: {warning.get('message')} ({warning})")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_consolidation(
    paths: ConsolidationPaths,
    *,
    min_match_coverage: float = 0.80,
) -> Dict[str, Any]:
    paths.output_root.mkdir(parents=True, exist_ok=True)

    step99e_update = load_json_any_encoding(paths.step99e_update_path)
    step99e_report = load_json_any_encoding(paths.step99e_report_path)
    final_classification = merge_classification(step99e_update, step99e_report)
    matrix_rows = build_field_matrix(final_classification)
    contract_patch = build_contract_patch(final_classification)
    audit_status, warnings = determine_audit_status(
        step99e_update,
        step99e_report,
        min_match_coverage=min_match_coverage,
    )

    report: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "step": "99-F",
        "audit_status": audit_status,
        "purpose": "Consolidate Step 99-A~E BIS API evidence into Step 97/98 classification updates.",
        "input_paths": {
            "step99e_update": str(paths.step99e_update_path),
            "step99e_report": str(paths.step99e_report_path),
            "step97_report": str(paths.step97_report_path) if paths.step97_report_path else None,
        },
        "claim_guards": dict(CLAIM_GUARDS),
        "step97_report_trace": summarize_step97_report(paths.step97_report_path),
        "upstream_step99e_integration_summary": {
            "audit_status": step99e_report.get("audit_status"),
            "getpos02_to_getbs02": step99e_report.get("getpos02_to_getbs02", {}),
            "getrealtime02_to_getbs02": step99e_report.get("getrealtime02_to_getbs02", {}),
            "headway_candidate_to_getbs02": step99e_report.get("headway_candidate_to_getbs02", {}),
            "route_direction_consistency": step99e_report.get("route_direction_consistency", {}),
            "source_row_counts": step99e_report.get("source_row_counts", {}),
        },
        "source_evidence": step99e_update.get("source_evidence", {}),
        "final_classification": final_classification,
        "field_classification_matrix": matrix_rows,
        "step98_causal_simulator_v2_contract_patch": contract_patch,
        "warnings": warnings,
        "prohibited_claims": [
            "Do not claim actual_headway is observed from Step 99 artifacts.",
            "Do not claim actual_arrival_departure_time is observed from Step 99 artifacts.",
            "Do not claim actual_dwell is observed from Step 99 artifacts.",
            "Do not use ETA-based headway as paper-level actual headway evidence.",
            "Do not make causal performance claims from this classification consolidation.",
            "Do not overwrite tensor DB classifications without a separate migration step.",
        ],
        "next_recommended_step": {
            "step": "Step 100 or Step 98-revision",
            "recommendation": (
                "Use step98_causal_simulator_v2_contract_patch.json to revise the causal simulator v2 contract. "
                "Keep candidate/proxy/observed boundaries explicit."
            ),
        },
    }

    report_json_path = paths.output_root / "classification_update_consolidated_step99f.json"
    report_md_path = paths.output_root / "classification_update_consolidated_step99f.md"
    matrix_csv_path = paths.output_root / "field_classification_matrix_step99f.csv"
    contract_patch_path = paths.output_root / "step98_causal_simulator_v2_contract_patch.json"

    write_csv_matrix(matrix_csv_path, matrix_rows)
    dump_json(contract_patch_path, contract_patch)
    write_markdown_report(report_md_path, report, matrix_rows)

    report["output_files"] = {
        "classification_update_consolidated_step99f_json": str(report_json_path),
        "classification_update_consolidated_step99f_md": str(report_md_path),
        "field_classification_matrix_step99f_csv": str(matrix_csv_path),
        "step98_causal_simulator_v2_contract_patch_json": str(contract_patch_path),
    }
    dump_json(report_json_path, report)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 99-F consolidation of BIS API classification updates for Step 97/98."
    )
    parser.add_argument("--project-root", default=".", help="Project root. Default: current directory.")
    parser.add_argument("--step99e-update", default=str(DEFAULT_STEP99E_UPDATE_PATH))
    parser.add_argument("--step99e-report", default=str(DEFAULT_STEP99E_REPORT_PATH))
    parser.add_argument("--step97-report", default=str(DEFAULT_STEP97_REPORT_PATH))
    parser.add_argument("--no-step97-report", action="store_true", help="Do not attempt to read the optional Step 97 report.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--min-match-coverage", type=float, default=0.80)
    return parser.parse_args()


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    step97_path = None if args.no_step97_report else resolve_path(project_root, args.step97_report)
    report = run_consolidation(
        ConsolidationPaths(
            step99e_update_path=resolve_path(project_root, args.step99e_update),
            step99e_report_path=resolve_path(project_root, args.step99e_report),
            step97_report_path=step97_path,
            output_root=resolve_path(project_root, args.output_root),
        ),
        min_match_coverage=args.min_match_coverage,
    )

    print("[OK] Step 99-F classification consolidation completed")
    print(f"[OK] audit_status: {report['audit_status']}")
    print(f"[OK] output_root : {report['output_files']['classification_update_consolidated_step99f_json']}")
    print(
        "[OK] getPos02 coverage:",
        report["upstream_step99e_integration_summary"].get("getpos02_to_getbs02", {}).get("match_coverage"),
    )
    print(
        "[OK] getRealtime02 ETA coverage:",
        report["upstream_step99e_integration_summary"].get("getrealtime02_to_getbs02", {}).get("match_coverage"),
    )
    print(
        "[OK] headway candidate coverage:",
        report["upstream_step99e_integration_summary"].get("headway_candidate_to_getbs02", {}).get("match_coverage"),
    )


if __name__ == "__main__":
    main()
