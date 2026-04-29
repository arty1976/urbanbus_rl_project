#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 100 — causal simulator v2 contract update.

This script converts the Step 99-F consolidated classification patch into a
formal causal simulator v2 contract artifact.

Safety contract:
- No DB (Database=데이터베이스) write.
- No tensor DB (Database=데이터베이스) overwrite.
- No additional API (Application Programming Interface=응용 프로그램 인터페이스) calls.
- No simulator rollout execution.
- paper_level_claim_allowed=false.
- causal_performance_claim_allowed=false.

Default inputs:
- artifacts/daegu_bis_api_audit/bis_api_classification_step99f/
  step98_causal_simulator_v2_contract_patch.json
- artifacts/daegu_bis_api_audit/bis_api_classification_step99f/
  classification_update_consolidated_step99f.json

Outputs:
- artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/
  causal_simulator_v2_contract.json
- artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/
  causal_simulator_v2_contract.md
- artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/
  causal_simulator_v2_field_readiness_matrix.csv
- artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/
  causal_simulator_v2_contract_manifest.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ARTIFACT_VERSION = "causal_simulator_v2_contract_step100_v1"
CONTRACT_VERSION = "causal_simulator_v2_route_aware_minimal_v1"

DEFAULT_STEP99F_PATCH_PATH = Path(
    "artifacts/daegu_bis_api_audit/bis_api_classification_step99f/"
    "step98_causal_simulator_v2_contract_patch.json"
)
DEFAULT_STEP99F_REPORT_PATH = Path(
    "artifacts/daegu_bis_api_audit/bis_api_classification_step99f/"
    "classification_update_consolidated_step99f.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100"
)

CLAIM_GUARDS: Dict[str, bool] = {
    "db_write_performed": False,
    "tensor_db_overwrite_performed": False,
    "additional_api_calls_performed": False,
    "simulator_rollout_executed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
}

# Step 100 must keep these fields blocked even if an upstream file is edited by
# mistake.  They require actual event logs, not ETA (Estimated Time of
# Arrival=예상 도착 시간) rows or repeated live snapshots.
BLOCKED_ACTUAL_OBSERVED_FIELDS = [
    "actual_headway",
    "actual_arrival_departure_time",
    "actual_dwell",
    "vehicle_load",
    "left_behind_passengers",
]

PROXY_OR_ESTIMATED_ONLY_FIELDS = [
    "passenger_wait_age_distribution",
    "actual_passenger_wait_p95_seconds",
    "actual_passenger_service_rate",
]

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

OPTIONAL_API_CANDIDATE_FIELDS = [
    "bus_id_or_vehicle_no",
    "live_position_xy",
    "current_route_sequence",
    "current_stop_id",
    "vehicle_trajectory",
    "getRealtime02_eta",
    "eta_based_headway",
]

CANONICAL_12_KPIS = [
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
]

KPI_CLAIM_BOUNDARIES: Dict[str, str] = {
    "cv_headway": "simulated_or_eta_candidate_until_actual_arrival_events_exist",
    "avg_wait_seconds": "queue_proxy_or_simulated_metric_not_observed_individual_wait",
    "bunching_rate": "simulated_or_candidate_metric_not_observed_actual_bunching",
    "on_time_rate": "simulated_metric_unless_schedule_and_actual_events_are_added",
    "intervention_rate": "simulator_policy_metric",
    "energy_proxy": "engineered_proxy_metric",
    "passenger_demand_generated": "simulator_generated_or_tensor_proxy_demand",
    "passenger_served_count": "simulator_generated_service_count",
    "passenger_service_rate": "simulated_or_proxy_metric_not_actual_service_rate",
    "passenger_wait_p95_seconds": "estimated_from_queue_age_buckets_or_simulation_not_observed_actual_p95",
    "energy_proxy_per_passenger": "engineered_proxy_metric",
    "fleet_reduction_ratio": "experimental_condition_metric",
}

FIELD_GROUP_LABELS = {
    "minimum_route_aware_v2_required_fields": "required_route_aware_input",
    "optional_api_candidate_fields": "optional_api_candidate",
    "proxy_or_estimated_only_fields": "proxy_or_estimated_only",
    "blocked_actual_observed_fields": "blocked_actual_observed",
}


@dataclass(frozen=True)
class Step100Paths:
    step99f_patch_path: Path
    step99f_report_path: Optional[Path]
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


def read_optional_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None or not path.exists():
        return None
    return load_json_any_encoding(path)


def classification_is_usable_for_required_field(classification: str) -> bool:
    value = str(classification).strip().lower()
    if not value:
        return False
    blocked_prefixes = ("missing", "not_observed", "blocked")
    return not value.startswith(blocked_prefixes)


def normalize_patch(raw_patch: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Return a normalized Step 99-F patch and safety warnings."""
    warnings: List[Dict[str, Any]] = []
    patch = dict(raw_patch)

    for group in FIELD_GROUP_LABELS:
        value = patch.get(group)
        if not isinstance(value, dict):
            patch[group] = {}
            warnings.append({
                "code": "missing_or_invalid_patch_group",
                "group": group,
                "message": f"Patch group {group} is missing or not a dictionary.",
            })

    # Preserve Step 100 hard guard.  This makes the contract safer than blindly
    # trusting an upstream patch that might have been hand-edited.
    blocked = dict(patch.get("blocked_actual_observed_fields", {}))
    for field in BLOCKED_ACTUAL_OBSERVED_FIELDS:
        old = blocked.get(field)
        if old not in (None, "not_observed", "missing", "blocked"):
            warnings.append({
                "code": "blocked_field_was_forced_back_to_not_observed",
                "field": field,
                "upstream_classification": old,
                "message": f"{field} cannot be upgraded by Step 100; forcing not_observed.",
            })
        blocked[field] = "not_observed" if field.startswith("actual_") or field in {"vehicle_load", "left_behind_passengers"} else "missing"
    patch["blocked_actual_observed_fields"] = blocked

    proxy_only = dict(patch.get("proxy_or_estimated_only_fields", {}))
    for field in PROXY_OR_ESTIMATED_ONLY_FIELDS:
        proxy_only.setdefault(field, "proxy_or_estimated_only")
    patch["proxy_or_estimated_only_fields"] = proxy_only

    required = dict(patch.get("minimum_route_aware_v2_required_fields", {}))
    for field in REQUIRED_ROUTE_AWARE_FIELDS:
        required.setdefault(field, "missing")
    patch["minimum_route_aware_v2_required_fields"] = required

    optional = dict(patch.get("optional_api_candidate_fields", {}))
    for field in OPTIONAL_API_CANDIDATE_FIELDS:
        optional.setdefault(field, "missing")
    patch["optional_api_candidate_fields"] = optional

    return patch, warnings


def build_field_readiness_rows(patch: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    for group_key, group_label in FIELD_GROUP_LABELS.items():
        group = patch.get(group_key, {}) or {}
        for field, classification in sorted(group.items()):
            seen.add((group_key, field))
            if group_key == "minimum_route_aware_v2_required_fields":
                readiness = "ready" if classification_is_usable_for_required_field(classification) else "blocked"
            elif group_key == "blocked_actual_observed_fields":
                readiness = "blocked_as_actual_observation"
            elif group_key == "proxy_or_estimated_only_fields":
                readiness = "proxy_only"
            else:
                readiness = "optional_candidate" if classification_is_usable_for_required_field(classification) else "optional_missing"

            rows.append({
                "field": field,
                "field_group": group_label,
                "classification": classification,
                "readiness": readiness,
                "contract_requirement": requirement_text(group_key, field),
            })

    # Add KPI (Key Performance Indicator=핵심 성과 지표) rows separately.
    for kpi in CANONICAL_12_KPIS:
        rows.append({
            "field": kpi,
            "field_group": "canonical_12_kpi",
            "classification": KPI_CLAIM_BOUNDARIES[kpi],
            "readiness": "contract_metric_defined",
            "contract_requirement": "Metric may be emitted by simulator v2 under its stated claim boundary.",
        })

    return rows


def requirement_text(group_key: str, field: str) -> str:
    if group_key == "minimum_route_aware_v2_required_fields":
        return "Required for minimal route-aware causal simulator v2 scaffold."
    if group_key == "optional_api_candidate_fields":
        return "Optional calibration/sanity-check candidate; not required for first scaffold."
    if group_key == "proxy_or_estimated_only_fields":
        return "May only be simulated or estimated; do not call it an observed actual field."
    if group_key == "blocked_actual_observed_fields":
        return "Blocked as observed input until actual event/load records are added."
    return "Review required."


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "field",
        "field_group",
        "classification",
        "readiness",
        "contract_requirement",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def determine_contract_status(rows: List[Dict[str, Any]], warnings: List[Dict[str, Any]]) -> str:
    required_rows = [r for r in rows if r["field_group"] == "required_route_aware_input"]
    blocked_required = [r for r in required_rows if r["readiness"] != "ready"]
    if blocked_required:
        return "REVIEW_REQUIRED_REQUIRED_FIELDS_BLOCKED"
    if warnings:
        return "READY_WITH_WARNINGS"
    return "READY_FOR_ROUTE_AWARE_MINIMAL_SCAFFOLD"


def build_contract(
    patch: Dict[str, Any],
    step99f_report: Optional[Dict[str, Any]],
    rows: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    status = determine_contract_status(rows, warnings)
    source_coverage = {}
    if step99f_report:
        source_coverage = step99f_report.get("upstream_step99e_integration_summary", {}) or {}

    return {
        "artifact_version": ARTIFACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "step": "100",
        "contract_status": status,
        "purpose": (
            "Formal Step 100 causal simulator v2 contract generated from the Step 99-F "
            "classification patch. This is a contract/update artifact, not a simulator implementation."
        ),
        "claim_guards": dict(CLAIM_GUARDS),
        "source_trace": {
            "step99f_patch_artifact_version": patch.get("artifact_version"),
            "step99f_report_artifact_version": step99f_report.get("artifact_version") if step99f_report else None,
            "step99f_report_audit_status": step99f_report.get("audit_status") if step99f_report else None,
            "upstream_step99e_integration_summary": source_coverage,
        },
        "field_groups": {
            "required_route_aware_inputs": patch["minimum_route_aware_v2_required_fields"],
            "optional_api_candidate_inputs": patch["optional_api_candidate_fields"],
            "proxy_or_estimated_only_inputs": patch["proxy_or_estimated_only_fields"],
            "blocked_actual_observed_inputs": patch["blocked_actual_observed_fields"],
        },
        "state_contract": build_state_contract(),
        "action_contract": build_action_contract(),
        "transition_contract": build_transition_contract(),
        "kpi_contract": build_kpi_contract(),
        "rollout_writer_contract": build_rollout_writer_contract(),
        "validation_gates": build_validation_gates(),
        "warnings": warnings,
        "prohibited_claims": [
            "Do not claim actual_headway is observed from Step 99/100 artifacts.",
            "Do not claim actual_arrival_departure_time is observed from Step 99/100 artifacts.",
            "Do not claim actual_dwell is observed from Step 99/100 artifacts.",
            "Do not claim ETA-based headway is actual headway.",
            "Do not claim causal performance improvement from this contract update.",
            "Do not claim paper-level performance results until a real causal simulator and evaluation run are complete.",
        ],
        "next_recommended_step": {
            "step": "Step 101",
            "name": "causal simulator v2 adapter scaffold",
            "goal": (
                "Implement a minimal route-aware adapter that consumes this contract while preserving candidate/proxy/blocked boundaries."
            ),
        },
    }


def build_state_contract() -> Dict[str, Any]:
    return {
        "state_schema_version": "causal_simulator_v2_state_schema_v1",
        "static_graph_state": {
            "required_fields": ["node_uid", "node_index", "edge_index", "edge_distance_m", "edge_time_sec"],
            "description": "Approved tensor DB static graph skeleton used for route-aware movement and nominal travel time.",
        },
        "route_sequence_state": {
            "required_fields": ["route_id", "direction_id", "ordered_stop_sequence"],
            "description": "Route-stop sequence from getBs02 observed candidates; complete for successful 234/238 route records only.",
            "known_exceptions_policy": "Keep known AUTH_ERROR route exceptions out of required full-coverage claims.",
        },
        "stop_demand_queue_state": {
            "required_fields": ["boardings_recent", "alightings_recent", "waiting_passenger_cnt", "hour_sin_cos", "is_peak"],
            "description": "Tensor DB demand and queue proxies. These are not passenger-level actual wait records.",
        },
        "vehicle_agent_state": {
            "minimal_required_fields": ["agent_id", "route_id", "direction_id", "current_stop_order", "active_bus_mask"],
            "optional_candidate_fields": ["bus_id_or_vehicle_no", "live_position_xy", "current_route_sequence", "current_stop_id"],
            "description": "Simulator-internal vehicle state may be calibrated against getPos02 candidates but is not an observed complete AVL log.",
        },
        "time_context_state": {
            "required_fields": ["state_ts", "service_date", "time_band", "hour_sin_cos", "is_peak"],
            "description": "Time context for demand/queue transition. Calendar/event/weather effects remain out of scope.",
        },
    }


def build_action_contract() -> Dict[str, Any]:
    return {
        "action_space_version": "bus_control_action_v2_route_aware_minimal",
        "action_type": "discrete_or_small_discrete_hold_control",
        "minimal_actions": [
            {
                "action_id": 0,
                "name": "NOOP",
                "meaning": "Continue nominal route progression without holding intervention.",
            },
            {
                "action_id": 1,
                "name": "HOLD_SHORT",
                "meaning": "Apply bounded short holding action at current stop/order.",
                "suggested_hold_seconds": 30,
            },
            {
                "action_id": 2,
                "name": "HOLD_LONG",
                "meaning": "Apply bounded long holding action at current stop/order.",
                "suggested_hold_seconds": 60,
            },
        ],
        "disabled_until_future_evidence": [
            {
                "name": "SKIP_STOP",
                "reason": "Requires stronger passenger service and left-behind passenger modeling before being safe for evaluation.",
            },
            {
                "name": "EXPRESS_REROUTE",
                "reason": "Requires road network, schedule, and operational feasibility constraints that are not in Step 100.",
            },
        ],
    }


def build_transition_contract() -> Dict[str, Any]:
    return {
        "transition_schema_version": "causal_simulator_v2_transition_contract_v1",
        "step_granularity_minutes": 30,
        "route_progression": {
            "uses_ordered_stop_sequence": True,
            "uses_nominal_edge_time_sec": True,
            "can_use_getPos02_for_calibration": True,
            "claim_boundary": "Route progression is simulated; repeated getPos02 snapshots are calibration candidates only.",
        },
        "demand_queue_update": {
            "uses_boardings_recent_proxy": True,
            "uses_alightings_recent_proxy": True,
            "uses_waiting_passenger_cnt_proxy": True,
            "individual_passenger_arrival_timestamps_observed": False,
            "claim_boundary": "Passenger wait and service metrics are simulated/proxy metrics until passenger-level records exist.",
        },
        "headway_model": {
            "may_use_eta_based_headway_candidate": True,
            "actual_headway_observed": False,
            "claim_boundary": "eta_based_headway is a candidate/calibration signal, not observed actual headway.",
        },
        "dwell_model": {
            "actual_dwell_observed": False,
            "allowed_mode": "engineered_or_simulated_from_boarding_alighting_proxy",
            "claim_boundary": "Do not report dwell as observed actual dwell.",
        },
    }


def build_kpi_contract() -> Dict[str, Any]:
    return {
        "kpi_schema_version": "canonical_12_kpi_for_causal_simulator_v2_v1",
        "shared_kpis": CANONICAL_12_KPIS,
        "claim_boundaries": dict(KPI_CLAIM_BOUNDARIES),
        "paper_level_use": "forbidden_until_real_causal_validation",
        "canonical_aggregator_requirement": "Downstream canonical KPI aggregator must preserve all 12 KPI columns and non-claim metadata.",
    }


def build_rollout_writer_contract() -> Dict[str, Any]:
    return {
        "rollout_schema_version": "causal_simulator_v2_rollout_schema_v1",
        "required_outputs": [
            "raw_events.parquet",
            "window_rollup.parquet",
            "run_manifest.json",
            "contract_snapshot.json",
        ],
        "required_raw_event_columns": [
            "state_ts",
            "time_band",
            "agent_id",
            "route_id",
            "direction_id",
            "current_stop_order",
            "action",
            "hold_seconds",
            "waiting_passenger_cnt_proxy",
            "reward_total",
            "terminated",
            "truncated",
        ],
        "required_window_rollup_columns": [
            "condition_id",
            "seed",
            "window_id",
            "state_ts",
            "service_date",
            "time_band",
            "evaluation_horizon_minutes",
            *CANONICAL_12_KPIS,
            "source_mode",
            "paper_level_claim_allowed",
            "causal_performance_claim_allowed",
        ],
        "non_claim_metadata_required": True,
    }


def build_validation_gates() -> Dict[str, Any]:
    return {
        "gate_1_required_field_readiness": "All required_route_aware_inputs must be ready or observed/proxy/candidate usable.",
        "gate_2_blocked_field_guard": "actual_headway, actual_arrival_departure_time, and actual_dwell must remain blocked/not_observed.",
        "gate_3_claim_guard": "paper_level_claim_allowed and causal_performance_claim_allowed must remain false for contract generation.",
        "gate_4_rollout_schema": "Any Step 101/102 adapter must emit raw_events and window_rollup with the required schema.",
        "gate_5_canonical_kpi": "All 12 KPI fields must be preserved through canonical aggregation before any comparison table is produced.",
    }


def write_markdown(path: Path, contract: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = [
        "# Step 100 — Causal simulator v2 contract update",
        "",
        "## Purpose",
        "",
        "This document formalizes the causal simulator v2 contract after Step 99-F classification consolidation.",
        "",
        "It is a contract/update artifact, not a simulator implementation and not a performance result.",
        "",
        "## Contract status",
        "",
        f"- Contract version: `{contract['contract_version']}`",
        f"- Contract status: `{contract['contract_status']}`",
        "",
        "## Safety guards",
        "",
    ]
    for key, value in contract["claim_guards"].items():
        lines.append(f"- `{key}`: `{value}`")

    lines.extend([
        "",
        "## Required route-aware inputs",
        "",
        "| Field | Classification | Readiness |",
        "|---|---|---|",
    ])
    for row in rows:
        if row["field_group"] == "required_route_aware_input":
            lines.append(
                f"| `{row['field']}` | `{row['classification']}` | `{row['readiness']}` |"
            )

    lines.extend([
        "",
        "## Optional API candidate inputs",
        "",
        "| Field | Classification | Readiness |",
        "|---|---|---|",
    ])
    for row in rows:
        if row["field_group"] == "optional_api_candidate":
            lines.append(
                f"| `{row['field']}` | `{row['classification']}` | `{row['readiness']}` |"
            )

    lines.extend([
        "",
        "## Proxy-only and blocked fields",
        "",
        "| Field | Group | Classification | Readiness |",
        "|---|---|---|---|",
    ])
    for row in rows:
        if row["field_group"] in {"proxy_or_estimated_only", "blocked_actual_observed"}:
            lines.append(
                f"| `{row['field']}` | `{row['field_group']}` | `{row['classification']}` | `{row['readiness']}` |"
            )

    lines.extend([
        "",
        "## KPI (Key Performance Indicator=핵심 성과 지표) contract",
        "",
        "The simulator v2 rollout path must preserve the 12-KPI schema:",
        "",
    ])
    for kpi in CANONICAL_12_KPIS:
        lines.append(f"- `{kpi}` — {KPI_CLAIM_BOUNDARIES[kpi]}")

    lines.extend([
        "",
        "## Critical interpretation",
        "",
        "Step 100 allows route-aware simulator v2 design to consume `/getBs02` route-stop sequence fields and to use `/getPos02` and `/getRealtime02` outputs as optional candidate/calibration signals.",
        "",
        "Step 100 does **not** allow `eta_based_headway` to become actual observed headway. `actual_headway`, `actual_arrival_departure_time`, and `actual_dwell` remain non-observed.",
        "",
        "## Prohibited claims",
        "",
    ])
    for claim in contract["prohibited_claims"]:
        lines.append(f"- {claim}")

    lines.extend([
        "",
        "## Next recommended step",
        "",
        f"- `{contract['next_recommended_step']['step']}`: {contract['next_recommended_step']['name']}",
        f"- Goal: {contract['next_recommended_step']['goal']}",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def run_step100(paths: Step100Paths) -> Dict[str, Any]:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    raw_patch = load_json_any_encoding(paths.step99f_patch_path)
    step99f_report = read_optional_json(paths.step99f_report_path)

    patch, warnings = normalize_patch(raw_patch)
    rows = build_field_readiness_rows(patch)
    contract = build_contract(patch, step99f_report, rows, warnings)

    contract_json_path = paths.output_root / "causal_simulator_v2_contract.json"
    contract_md_path = paths.output_root / "causal_simulator_v2_contract.md"
    matrix_csv_path = paths.output_root / "causal_simulator_v2_field_readiness_matrix.csv"
    manifest_json_path = paths.output_root / "causal_simulator_v2_contract_manifest.json"

    write_csv(matrix_csv_path, rows)
    write_markdown(contract_md_path, contract, rows)
    dump_json(contract_json_path, contract)

    manifest = {
        "artifact_version": "causal_simulator_v2_contract_manifest_step100_v1",
        "step": "100",
        "contract_status": contract["contract_status"],
        "input_files": {
            "step99f_patch": str(paths.step99f_patch_path),
            "step99f_report": str(paths.step99f_report_path) if paths.step99f_report_path else None,
        },
        "output_files": {
            "contract_json": str(contract_json_path),
            "contract_md": str(contract_md_path),
            "field_readiness_matrix_csv": str(matrix_csv_path),
            "manifest_json": str(manifest_json_path),
        },
        "claim_guards": dict(CLAIM_GUARDS),
        "warnings": warnings,
    }
    dump_json(manifest_json_path, manifest)
    contract["output_files"] = manifest["output_files"]
    return contract


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 100 causal simulator v2 contract from Step 99-F patch.")
    parser.add_argument("--project-root", default=".", help="Project root. Default: current directory.")
    parser.add_argument("--step99f-patch", default=str(DEFAULT_STEP99F_PATCH_PATH))
    parser.add_argument("--step99f-report", default=str(DEFAULT_STEP99F_REPORT_PATH))
    parser.add_argument("--no-step99f-report", action="store_true", help="Skip optional Step 99-F consolidated report read.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    step99f_report_path = None if args.no_step99f_report else resolve_path(project_root, args.step99f_report)
    contract = run_step100(
        Step100Paths(
            step99f_patch_path=resolve_path(project_root, args.step99f_patch),
            step99f_report_path=step99f_report_path,
            output_root=resolve_path(project_root, args.output_root),
        )
    )
    print("[OK] Step 100 causal simulator v2 contract update completed")
    print(f"[OK] contract_status: {contract['contract_status']}")
    print(f"[OK] output_root    : {Path(contract['output_files']['contract_json']).parent}")
    print(f"[OK] contract_json  : {contract['output_files']['contract_json']}")
    print(f"[OK] contract_md    : {contract['output_files']['contract_md']}")


if __name__ == "__main__":
    main()
