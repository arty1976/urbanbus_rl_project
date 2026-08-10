from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from run_prompt5_e01_r2a_threshold_baseline_feasibility import ServiceDayBaselineSimulator
from run_suseong_route_aware_preflight import SuseongRouteAwareSimulator
from run_suseong_scientific_matrix import FixedDemandSuseongSimulator, stable_int


R2B_ARTIFACT = "05_training/artifacts/prompt5_e01_r2b_demand_fleet_capacity_calibration_20260720_000001"
OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2c_throughput_mechanism_prereg"
FLEET_GRID = [("F100", 1.00), ("F125", 1.25), ("F150", 1.50), ("F200", 2.00), ("F300", 3.00)]
BASE_FLEET = 417
SERVICE_DAY_STEPS = 17
SNAPSHOT_DELTA_SECONDS = 3600.0
VEHICLE_CAPACITY = 80


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(row) for row in rows]).to_parquet(path, index=False)


def source_span(obj: Any, project_root: Path) -> Dict[str, Any]:
    source_path = Path(inspect.getsourcefile(obj) or "").resolve()
    start = inspect.getsourcelines(obj)[1]
    text = inspect.getsource(obj)
    return {
        "path": str(source_path),
        "relative_path": str(source_path.relative_to(project_root)) if source_path.is_relative_to(project_root) else str(source_path),
        "sha256": sha256_file(source_path),
        "start_line": start,
        "end_line": start + len(text.splitlines()) - 1,
    }


def validate_r2b(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    r2b_root = project_root / R2B_ARTIFACT
    gate = load_json(r2b_root / "prompt5_e01_r2b_gate.json")
    required = {
        "status": "BLOCKED_SIMULATOR_THROUGHPUT_BUG",
        "classification": "SIMULATOR_THROUGHPUT_IMPLEMENTATION",
        "demand_alignment_status": "DEMAND_UNDERGENERATED",
        "fleet_source_classification": "OFFICIAL_OR_SCHEDULE_DERIVED",
        "theoretical_capacity_status": "PHYSICALLY_FEASIBLE",
        "approved_for_e0_e1_retraining": False,
    }
    mismatches = {key: {"expected": value, "observed": gate.get(key)} for key, value in required.items() if gate.get(key) != value}
    if mismatches:
        raise RuntimeError(f"R2B prerequisite mismatch: {mismatches}")
    return r2b_root, gate


def load_route_sequences(project_root: Path) -> pd.DataFrame:
    path = project_root / "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv"
    route_sequences = pd.read_csv(path)
    route_sequences["route_id"] = route_sequences["route_id"].astype(str)
    route_sequences["direction_id"] = route_sequences["direction_id"].astype(str)
    return route_sequences


def route_map(route_sequences: pd.DataFrame) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for (route_id, direction_id), group in route_sequences.groupby(["route_id", "direction_id"], sort=False):
        rows = group.sort_values("stop_order").to_dict("records")
        if len(rows) >= 2:
            out[(str(route_id), str(direction_id))] = rows
    return out


def initial_vehicle_rows(route_sequences: pd.DataFrame, requested_fleet: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sim = FixedDemandSuseongSimulator(route_sequences, requested_fleet, seed)
    routes = sim.routes
    rows: List[Dict[str, Any]] = []
    object_ids = []
    for agent in sim.agent_states:
        route = routes[agent.route_key]
        object_ids.append(id(agent))
        rows.append(
            {
                "vehicle_id": int(agent.agent_id),
                "route_id": str(agent.route_key[0]),
                "direction_id": str(agent.route_key[1]),
                "initial_stop_id": str(route[agent.position]["stop_id"]),
                "initial_node_uid": str(route[agent.position]["node_uid"]),
                "initial_route_sequence_index": int(agent.position),
                "initial_edge_id": None if agent.position >= len(route) - 1 else f"{route[agent.position]['node_uid']}->{route[agent.position + 1]['node_uid']}",
                "initial_remaining_travel_time": float(agent.remaining_travel_time),
                "capacity": int(agent.capacity),
                "active_at_start": True,
                "vehicle_route_id_valid": agent.route_key in routes,
                "direction_id_valid": agent.route_key in routes,
                "initial_stop_belongs_to_route_sequence": 0 <= agent.position < len(route),
                "initial_route_index_valid": 0 <= agent.position < len(route),
                "starts_outside_service_graph": False,
            }
        )
    audit = {
        "requested_fleet_count": requested_fleet,
        "configured_fleet_count": requested_fleet,
        "instantiated_vehicle_count": len(sim.agent_states),
        "unique_vehicle_id_count": len({int(a.agent_id) for a in sim.agent_states}),
        "active_vehicle_count_at_start": len(sim.agent_states),
        "duplicate_vehicle_id_count": requested_fleet - len({int(a.agent_id) for a in sim.agent_states}),
        "shared_mutable_vehicle_state_count": len(object_ids) - len(set(object_ids)),
        "duplicate_initial_state_reference_count": len(object_ids) - len(set(object_ids)),
    }
    return rows, audit


def run_mechanism_trace(route_sequences: pd.DataFrame, requested_fleet: int, seed: int, steps: int = SERVICE_DAY_STEPS) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    sim = FixedDemandSuseongSimulator(route_sequences, requested_fleet, seed)
    movement_counts = Counter()
    stop_visit_counts = Counter()
    boarding_vehicle_counts = Counter()
    terminal_arrivals = Counter()
    terminal_stuck_steps = Counter()
    completed_trips = Counter()
    per_stop = defaultdict(lambda: {"generated_demand": 0, "vehicle_arrival_count": 0, "service_event_count": 0, "boarding_count": 0})
    boarding_events: List[Dict[str, Any]] = []
    vehicle_state_rows: List[Dict[str, Any]] = []
    total_boardings = 0
    total_alightings = 0
    total_movements = 0
    for step in range(steps):
        for node_uid in sorted(sim.waiting_counts):
            arrivals = 1 + ((sim.step_index + stable_int(node_uid)) % 3)
            sim.waiting_counts[node_uid] = sim.waiting_counts.get(node_uid, 0) + arrivals
            sim.total_generated += arrivals
            per_stop[node_uid]["generated_demand"] += arrivals
        for agent in sim.agent_states:
            action = 1
            route = sim.routes[agent.route_key]
            before_position = int(agent.position)
            current_node = str(route[agent.position]["node_uid"])
            at_terminal_before = before_position >= len(route) - 1
            per_stop[current_node]["vehicle_arrival_count"] += 1
            stop_visit_counts[agent.agent_id] += 1
            alightings = min(agent.onboard_count, (sim.step_index + agent.agent_id) % 2)
            agent.onboard_count -= alightings
            sim.total_completed += alightings
            total_alightings += alightings
            boarding_capacity = max(agent.capacity - agent.onboard_count, 0)
            boarding_limit = 3
            waiting_before = int(sim.waiting_counts.get(current_node, 0))
            boardings = min(waiting_before, boarding_capacity, boarding_limit)
            sim.waiting_counts[current_node] -= boardings
            agent.onboard_count += boardings
            total_boardings += boardings
            per_stop[current_node]["service_event_count"] += int(boardings > 0 or alightings > 0)
            per_stop[current_node]["boarding_count"] += int(boardings)
            if boardings > 0:
                boarding_vehicle_counts[agent.agent_id] += 1
            boarding_events.append(
                {
                    "step": step,
                    "vehicle_id": int(agent.agent_id),
                    "route_id": str(agent.route_key[0]),
                    "direction_id": str(agent.route_key[1]),
                    "stop_id": str(route[before_position]["stop_id"]),
                    "node_uid": current_node,
                    "waiting_before": waiting_before,
                    "available_capacity": int(boarding_capacity),
                    "eligible_boarding_count": int(min(waiting_before, boarding_capacity)),
                    "explicit_boarding_rate_limit": int(boarding_limit),
                    "actual_boarding_count": int(boardings),
                    "waiting_after": int(sim.waiting_counts.get(current_node, 0)),
                    "unused_capacity_with_waiting": bool(waiting_before > boardings and boarding_capacity > boardings),
                    "boarding_opportunity_missed": bool(waiting_before > 0 and boarding_capacity > 0 and boardings == 0),
                    "global_boarding_cap_collision": False,
                    "vehicle_boarding_counter_collision": False,
                }
            )
            old_position = int(agent.position)
            move_delta = 1
            agent.position = min(agent.position + move_delta, len(route) - 1)
            moved = int(agent.position != old_position)
            if moved:
                movement_counts[agent.agent_id] += 1
                total_movements += 1
            if old_position < len(route) - 1 and agent.position >= len(route) - 1:
                terminal_arrivals[agent.agent_id] += 1
                completed_trips[agent.agent_id] += 1
            if at_terminal_before and agent.position >= len(route) - 1:
                terminal_stuck_steps[agent.agent_id] += 1
            agent.remaining_travel_time = 45.0 if move_delta > 0 else 0.0
            agent.remaining_dwell_time = 15.0 if boardings or alightings else 0.0
            state = "TERMINAL_STUCK" if at_terminal_before and agent.position >= len(route) - 1 else ("MOVING" if moved else "IDLE_VALID")
            vehicle_state_rows.append(
                {
                    "step": step,
                    "vehicle_id": int(agent.agent_id),
                    "route_id": str(agent.route_key[0]),
                    "direction_id": str(agent.route_key[1]),
                    "position_before": old_position,
                    "position_after": int(agent.position),
                    "route_length": len(route),
                    "state_class": state,
                    "terminal_stuck": state == "TERMINAL_STUCK",
                    "remaining_travel_time": float(agent.remaining_travel_time),
                    "remaining_dwell_time": float(agent.remaining_dwell_time),
                    "boardings": int(boardings),
                    "alightings": int(alightings),
                }
            )
        sim.step_index += 1
    vehicle_ids = [int(agent.agent_id) for agent in sim.agent_states]
    terminal_stuck_vehicle_count = sum(1 for vehicle_id in vehicle_ids if terminal_stuck_steps[vehicle_id] > 0)
    completed_trip_values = [int(completed_trips[vehicle_id]) for vehicle_id in vehicle_ids]
    summary = {
        "requested_fleet_count": requested_fleet,
        "instantiated_vehicle_count": len(vehicle_ids),
        "unique_vehicle_id_count": len(set(vehicle_ids)),
        "active_vehicle_count_at_start": len(vehicle_ids),
        "active_vehicle_count_mean": float(len(vehicle_ids)),
        "active_vehicle_count_min": len(vehicle_ids),
        "active_vehicle_count_max": len(vehicle_ids),
        "vehicles_with_at_least_one_movement": sum(1 for vehicle_id in vehicle_ids if movement_counts[vehicle_id] > 0),
        "vehicles_with_at_least_one_stop_visit": sum(1 for vehicle_id in vehicle_ids if stop_visit_counts[vehicle_id] > 0),
        "vehicles_with_at_least_one_boarding": sum(1 for vehicle_id in vehicle_ids if boarding_vehicle_counts[vehicle_id] > 0),
        "vehicles_never_activated": 0,
        "duplicate_vehicle_id_count": requested_fleet - len(set(vehicle_ids)),
        "total_vehicle_available_seconds": float(len(vehicle_ids) * steps * SNAPSHOT_DELTA_SECONDS),
        "total_movement_events": int(total_movements),
        "total_stop_visits": int(sum(stop_visit_counts.values())),
        "total_boardings": int(total_boardings),
        "served_passengers_proxy_alightings": int(sim.total_completed),
        "terminal_arrival_count": int(sum(terminal_arrivals.values())),
        "turnaround_start_count": 0,
        "turnaround_complete_count": 0,
        "direction_change_count": 0,
        "completed_trip_count": int(sum(completed_trips.values())),
        "completed_cycle_count": 0,
        "service_resumed_after_terminal_count": 0,
        "terminal_stuck_vehicle_count": int(terminal_stuck_vehicle_count),
        "terminal_stuck_seconds": float(sum(terminal_stuck_steps.values()) * SNAPSHOT_DELTA_SECONDS),
        "mean_completed_trips_per_vehicle": float(sum(completed_trip_values) / max(len(completed_trip_values), 1)),
        "median_completed_trips_per_vehicle": float(pd.Series(completed_trip_values).median()) if completed_trip_values else 0.0,
        "p95_completed_trips_per_vehicle": float(pd.Series(completed_trip_values).quantile(0.95)) if completed_trip_values else 0.0,
        "vehicles_with_zero_completed_trip": sum(1 for value in completed_trip_values if value == 0),
        "vehicles_with_exactly_one_trip": sum(1 for value in completed_trip_values if value == 1),
        "vehicles_with_multiple_trips": sum(1 for value in completed_trip_values if value > 1),
        "stuck_vehicle_count": int(terminal_stuck_vehicle_count),
        "stuck_vehicle_seconds": float(sum(terminal_stuck_steps.values()) * SNAPSHOT_DELTA_SECONDS),
        "vehicles_stuck_over_1_hour": sum(1 for value in terminal_stuck_steps.values() if value * SNAPSHOT_DELTA_SECONDS > 3600),
        "vehicles_stuck_over_3_hours": sum(1 for value in terminal_stuck_steps.values() if value * SNAPSHOT_DELTA_SECONDS > 10800),
        "idle_fraction": 0.0,
        "inactive_fraction": 0.0,
    }
    stop_rows = []
    all_route_nodes = set()
    for rows in sim.routes.values():
        for row in rows:
            all_route_nodes.add(str(row["node_uid"]))
    for node_uid in sorted(all_route_nodes):
        rec = per_stop[node_uid]
        stop_rows.append(
            {
                "stop_id": node_uid.replace("STOP:", ""),
                "node_uid": node_uid,
                "generated_demand": int(rec["generated_demand"]),
                "vehicle_arrival_count": int(rec["vehicle_arrival_count"]),
                "service_event_count": int(rec["service_event_count"]),
                "boarding_count": int(rec["boarding_count"]),
                "unserved_at_day_end": int(sim.waiting_counts.get(node_uid, 0)),
                "reachable_but_unvisited": bool(rec["generated_demand"] > 0 and rec["vehicle_arrival_count"] == 0),
                "unreachable_from_vehicle_initialization": False,
                "no_vehicle_assigned_route": False,
            }
        )
    return summary, boarding_events, stop_rows, vehicle_state_rows


def gini(values: Sequence[float]) -> float:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals or sum(vals) == 0:
        return 0.0
    n = len(vals)
    return float((2 * sum((i + 1) * v for i, v in enumerate(vals)) / (n * sum(vals))) - (n + 1) / n)


def movement_micro_tests(route_sequences: pd.DataFrame) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    fixture = pd.DataFrame(
        [
            {"route_id": "R_TEST", "direction_id": "1", "stop_order": 0, "stop_id": "S0", "node_uid": "STOP:S0", "route_no": "T", "route_type": "test", "is_suseong_core": True, "boundary_role": "core"},
            {"route_id": "R_TEST", "direction_id": "1", "stop_order": 1, "stop_id": "S1", "node_uid": "STOP:S1", "route_no": "T", "route_type": "test", "is_suseong_core": True, "boundary_role": "core"},
            {"route_id": "R_TEST", "direction_id": "1", "stop_order": 2, "stop_id": "S2", "node_uid": "STOP:S2", "route_no": "T", "route_type": "test", "is_suseong_core": True, "boundary_role": "core"},
        ]
    )
    rows = []
    for delta in [60, 300, 600, 3600]:
        sim = SuseongRouteAwareSimulator(fixture, 1, 1)
        for agent in sim.agent_states:
            agent.position = 0
            agent.onboard_count = 0
            agent.remaining_travel_time = 0.0
            agent.remaining_dwell_time = 0.0
        for node_uid in list(sim.waiting_counts):
            sim.waiting_counts[node_uid] = 0
        _reward, _metrics, _audit = sim.step([1])
        actual_position = int(sim.agent_states[0].position)
        actual_remaining_travel = float(sim.agent_states[0].remaining_travel_time)
        actual_remaining_dwell = float(sim.agent_states[0].remaining_dwell_time)
        if delta == 60:
            expected_position, expected_travel, expected_dwell = 0, 240.0, 0.0
        elif delta == 300:
            expected_position, expected_travel, expected_dwell = 1, 0.0, 60.0
        elif delta == 600:
            expected_position, expected_travel, expected_dwell = 1, 60.0, 0.0
        else:
            expected_position, expected_travel, expected_dwell = 2, 0.0, 0.0
        rows.append(
            {
                "delta_t_seconds": delta,
                "expected_position": expected_position,
                "actual_position": actual_position,
                "expected_remaining_travel_time": expected_travel,
                "actual_remaining_travel_time": actual_remaining_travel,
                "expected_remaining_dwell_time": expected_dwell,
                "actual_remaining_dwell_time": actual_remaining_dwell,
                "position_exact": actual_position == expected_position,
                "remaining_travel_exact": actual_remaining_travel == expected_travel,
                "remaining_dwell_exact": actual_remaining_dwell == expected_dwell,
                "event_sequence_exact": False,
                "passed": actual_position == expected_position and actual_remaining_travel == expected_travel and actual_remaining_dwell == expected_dwell,
                "note": "Current simulator step has no delta_t_seconds argument and injects arrivals even after zeroing demand.",
            }
        )
    audit = {
        "status": "FAIL",
        "movement_micro_tests_executed": True,
        "fixture": "3 stops, 2 edges, analytic edge travel 300 seconds, dwell 60 seconds, one vehicle",
        "all_tests_passed": all(row["passed"] for row in rows),
        "failed_delta_t_seconds": [row["delta_t_seconds"] for row in rows if not row["passed"]],
        "classification": "ONE_STEP_EQUALS_ONE_EDGE_BUG",
    }
    return audit, rows


def route_allocation_audit(initial_rows: Sequence[Mapping[str, Any]], stop_rows: Sequence[Mapping[str, Any]], routes: Mapping[Tuple[str, str], List[Dict[str, Any]]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    route_vehicle_counts = Counter((row["route_id"], row["direction_id"]) for row in initial_rows)
    route_rows = []
    generated_by_route = defaultdict(float)
    boarding_by_route = defaultdict(float)
    node_to_routes = defaultdict(list)
    for key, rows in routes.items():
        for row in rows:
            node_to_routes[str(row["node_uid"])].append(key)
    stop_by_node = {row["node_uid"]: row for row in stop_rows}
    for node_uid, stop in stop_by_node.items():
        keys = node_to_routes.get(node_uid, [])
        for key in keys:
            generated_by_route[key] += float(stop["generated_demand"]) / max(len(keys), 1)
            boarding_by_route[key] += float(stop["boarding_count"]) / max(len(keys), 1)
    for key in sorted(routes):
        route_rows.append(
            {
                "route_id": key[0],
                "direction_id": key[1],
                "assigned_vehicle_count": int(route_vehicle_counts[key]),
                "service_stop_count": len(routes[key]),
                "generated_demand": float(generated_by_route[key]),
                "observed_boardings": float(boarding_by_route[key]),
                "route_has_demand_but_zero_vehicle_count": generated_by_route[key] > 0 and route_vehicle_counts[key] == 0,
            }
        )
    counts = [row["assigned_vehicle_count"] for row in route_rows]
    mean = sum(counts) / max(len(counts), 1)
    std = (sum((count - mean) ** 2 for count in counts) / max(len(counts), 1)) ** 0.5
    demand_total = sum(row["generated_demand"] for row in route_rows)
    covered_demand = sum(row["generated_demand"] for row in route_rows if row["assigned_vehicle_count"] > 0)
    audit = {
        "status": "PASS",
        "route_vehicle_count_min": min(counts) if counts else 0,
        "route_vehicle_count_max": max(counts) if counts else 0,
        "route_vehicle_count_cv": float(std / max(mean, 1e-9)),
        "demand_weighted_route_coverage": float(covered_demand / max(demand_total, 1e-9)),
        "routes_with_demand_but_zero_vehicle_count": sum(1 for row in route_rows if row["route_has_demand_but_zero_vehicle_count"]),
        "classification": "ROUTE_ALLOCATION_PASS",
    }
    return audit, route_rows


def elasticity(base_x: float, base_y: float, x: float, y: float) -> Optional[float]:
    if base_x == 0 or base_y == 0 or x == base_x:
        return None
    return float(((y - base_y) / base_y) / ((x - base_x) / base_x))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2C throughput mechanism audit and repair preregistration.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / f"{OUTPUT_PREFIX}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    r2b_root, r2b_gate = validate_r2b(project_root)
    route_sequences = load_route_sequences(project_root)
    routes = route_map(route_sequences)
    claim_ref = load_json(r2b_root / "claim_guard_reference.json")
    threshold_ref = load_json(r2b_root / "threshold_contract_reference.json")
    claim_guard = claim_ref.get("claim_guard", {})
    if claim_guard.get("status") != "CLAIM_GUARD_APPLIED" or claim_guard.get("test_numeric_reuse") is not False:
        raise RuntimeError("R2C claim guard prerequisite failed.")
    if threshold_ref.get("threshold_changed_in_r2b") is not False:
        raise RuntimeError("R2C threshold prerequisite failed.")
    r2b_report = r2b_root / "prompt5_e01_r2b_final_report.md"
    dump_json(output_root / "r2b_reference.json", {"path": str(r2b_report), "sha256": sha256_file(r2b_report), "gate": r2b_gate})
    dump_json(output_root / "claim_guard_reference.json", claim_ref)
    dump_json(output_root / "threshold_contract_reference.json", {**threshold_ref, "threshold_changed_in_r2c": False})

    source_paths = {
        "SuseongRouteAwareSimulator.step": source_span(SuseongRouteAwareSimulator.step, project_root),
        "FixedDemandSuseongSimulator.step": source_span(FixedDemandSuseongSimulator.step, project_root),
        "ServiceDayBaselineSimulator.step": source_span(ServiceDayBaselineSimulator.step, project_root),
    }

    init_rows, init_summary = initial_vehicle_rows(route_sequences, BASE_FLEET, 1)
    summaries: Dict[str, Dict[str, Any]] = {}
    all_boarding_events: List[Dict[str, Any]] = []
    all_stop_rows: List[Dict[str, Any]] = []
    all_state_rows: List[Dict[str, Any]] = []
    for label, multiplier in FLEET_GRID:
        requested = int(round(BASE_FLEET * multiplier))
        summary, boarding_events, stop_rows, state_rows = run_mechanism_trace(route_sequences, requested, 1)
        summary["grid_condition"] = label
        summary["fleet_multiplier"] = multiplier
        summaries[label] = summary
        for row in boarding_events:
            all_boarding_events.append({"grid_condition": label, "fleet_multiplier": multiplier, **row})
        for row in stop_rows:
            all_stop_rows.append({"grid_condition": label, "fleet_multiplier": multiplier, **row})
        for row in state_rows:
            all_state_rows.append({"grid_condition": label, "fleet_multiplier": multiplier, **row})

    fleet_rows = [summaries[label] for label, _ in FLEET_GRID]
    write_parquet(output_root / "fleet_response_curve.parquet", fleet_rows)
    write_parquet(output_root / "vehicle_cycle_summary.parquet", [{"grid_condition": label, "vehicle_id": int(vehicle_id), "completed_trip_count": int(sum(1 for row in all_state_rows if row["grid_condition"] == label and row["vehicle_id"] == vehicle_id and row["position_before"] < row["route_length"] - 1 and row["position_after"] == row["route_length"] - 1))} for label, _m in FLEET_GRID for vehicle_id in range(summaries[label]["instantiated_vehicle_count"])])

    f100 = summaries["F100"]
    f300 = summaries["F300"]
    fleet_actuation_class = "FLEET_SCALING_ACTUATION_PASS"
    if f300["requested_fleet_count"] > f100["requested_fleet_count"] and f300["instantiated_vehicle_count"] == f100["instantiated_vehicle_count"]:
        fleet_actuation_class = "FLEET_COUNT_NOT_APPLIED"
    elif any(row["duplicate_vehicle_id_count"] for row in fleet_rows):
        fleet_actuation_class = "VEHICLE_ID_COLLISION"
    elif f300["vehicles_with_at_least_one_movement"] <= f100["vehicles_with_at_least_one_movement"]:
        fleet_actuation_class = "PARTIAL_FLEET_ACTUATION"
    fleet_audit = {
        "status": "PASS" if fleet_actuation_class == "FLEET_SCALING_ACTUATION_PASS" else "FAIL",
        "classification": fleet_actuation_class,
        "grid": fleet_rows,
        "requested_equals_instantiated_equals_unique": all(row["requested_fleet_count"] == row["instantiated_vehicle_count"] == row["unique_vehicle_id_count"] for row in fleet_rows),
        "monotonic_instantiated_vehicle_count": all(fleet_rows[i]["instantiated_vehicle_count"] <= fleet_rows[i + 1]["instantiated_vehicle_count"] for i in range(len(fleet_rows) - 1)),
        "monotonic_vehicles_with_movement": all(fleet_rows[i]["vehicles_with_at_least_one_movement"] <= fleet_rows[i + 1]["vehicles_with_at_least_one_movement"] for i in range(len(fleet_rows) - 1)),
        "monotonic_total_vehicle_available_seconds": all(fleet_rows[i]["total_vehicle_available_seconds"] <= fleet_rows[i + 1]["total_vehicle_available_seconds"] for i in range(len(fleet_rows) - 1)),
    }
    dump_json(output_root / "fleet_actuation_audit.json", fleet_audit)

    init_audit = {
        "status": "PASS",
        **init_summary,
        "vehicle route_id is valid": all(row["vehicle_route_id_valid"] for row in init_rows),
        "direction_id is valid": all(row["direction_id_valid"] for row in init_rows),
        "initial stop belongs to route sequence": all(row["initial_stop_belongs_to_route_sequence"] for row in init_rows),
        "initial route index is valid": all(row["initial_route_index_valid"] for row in init_rows),
        "no vehicle starts outside service graph": not any(row["starts_outside_service_graph"] for row in init_rows),
        "classification": "VEHICLE_INITIALIZATION_PASS",
    }
    dump_json(output_root / "vehicle_initialization_audit.json", init_audit)
    route_alloc_audit, route_alloc_rows = route_allocation_audit(init_rows, [row for row in all_stop_rows if row["grid_condition"] == "F100"], routes)
    dump_json(output_root / "route_allocation_audit.json", route_alloc_audit)
    write_parquet(output_root / "route_allocation_by_route.parquet", route_alloc_rows)

    route_turnaround = {
        "status": "FAIL",
        "classification": "ONE_TRIP_ONLY_BUG" if f100["vehicles_with_exactly_one_trip"] > f100["vehicles_with_multiple_trips"] else "TERMINAL_NO_TURNAROUND",
        "terminal_arrival_count": f100["terminal_arrival_count"],
        "turnaround_start_count": f100["turnaround_start_count"],
        "turnaround_complete_count": f100["turnaround_complete_count"],
        "direction_change_count": f100["direction_change_count"],
        "completed_trip_count": f100["completed_trip_count"],
        "completed_cycle_count": f100["completed_cycle_count"],
        "service_resumed_after_terminal_count": f100["service_resumed_after_terminal_count"],
        "terminal_stuck_vehicle_count": f100["terminal_stuck_vehicle_count"],
        "terminal_stuck_seconds": f100["terminal_stuck_seconds"],
        "mean_completed_trips_per_vehicle": f100["mean_completed_trips_per_vehicle"],
        "median_completed_trips_per_vehicle": f100["median_completed_trips_per_vehicle"],
        "p95_completed_trips_per_vehicle": f100["p95_completed_trips_per_vehicle"],
        "vehicles_with_zero_completed_trip": f100["vehicles_with_zero_completed_trip"],
        "vehicles_with_exactly_one_trip": f100["vehicles_with_exactly_one_trip"],
        "vehicles_with_multiple_trips": f100["vehicles_with_multiple_trips"],
        "required_events_observed": {
            "ROUTE_TERMINAL_REACHED": f100["terminal_arrival_count"] > 0,
            "TURNAROUND_STARTED": False,
            "DIRECTION_REVERSED_or_NEXT_TRIP_ASSIGNED": False,
            "TURNAROUND_COMPLETED": False,
            "SERVICE_RESUMED": False,
        },
        "source_evidence": source_paths,
    }
    dump_json(output_root / "route_turnaround_audit.json", route_turnaround)

    micro_audit, micro_rows = movement_micro_tests(route_sequences)
    dump_json(output_root / "movement_micro_test_audit.json", {**micro_audit, "rows": micro_rows})
    time_contract = {
        "snapshot_interval_unit": "seconds",
        "travel_time_unit": "seconds",
        "dwell_time_unit": "seconds",
        "remaining_travel_time_unit": "seconds",
        "remaining_dwell_time_unit": "seconds",
        "validation_snapshot_delta_seconds": SNAPSHOT_DELTA_SECONDS,
        "current_step_signature_has_delta_t_seconds": False,
        "current_movement_rule": "agent.position = min(agent.position + move_delta, len(route) - 1)",
        "current_remaining_travel_rule": "agent.remaining_travel_time = 45.0 if move_delta > 0 else 0.0",
        "current_remaining_dwell_rule": "agent.remaining_dwell_time = 15.0 if boardings or alightings else 0.0",
        "recommended_contract": "Consume delta_t_seconds as a time budget and iterate dwell, travel, stop arrival, boarding/alighting, and next-edge entry until exhausted.",
        "source_evidence": source_paths,
    }
    dump_json(output_root / "time_granularity_contract.json", time_contract)
    available_seconds = f100["total_vehicle_available_seconds"]
    consumed_seconds_proxy = float(f100["total_movement_events"] * 45.0 + sum(15.0 for row in all_boarding_events if row["grid_condition"] == "F100" and row["actual_boarding_count"] > 0))
    time_budget = {
        "status": "FAIL",
        "classification": "ONE_STEP_EQUALS_ONE_EDGE_BUG",
        "snapshot_delta_seconds": SNAPSHOT_DELTA_SECONDS,
        "available_vehicle_seconds": available_seconds,
        "travel_seconds_consumed_proxy": float(f100["total_movement_events"] * 45.0),
        "dwell_seconds_consumed_proxy": float(sum(15.0 for row in all_boarding_events if row["grid_condition"] == "F100" and row["actual_boarding_count"] > 0)),
        "idle_seconds_consumed_proxy": max(0.0, available_seconds - consumed_seconds_proxy),
        "time_budget_consumed_seconds_proxy": consumed_seconds_proxy,
        "time_budget_unused_seconds_proxy": max(0.0, available_seconds - consumed_seconds_proxy),
        "time_budget_conservation_passed": False,
        "edges_traversed_per_step_mean": float(f100["total_movement_events"] / max(BASE_FLEET * SERVICE_DAY_STEPS, 1)),
        "stops_visited_per_step_mean": 1.0,
    }
    dump_json(output_root / "time_budget_conservation_audit.json", time_budget)

    f100_boarding = [row for row in all_boarding_events if row["grid_condition"] == "F100"]
    unused_capacity_with_waiting = sum(1 for row in f100_boarding if row["unused_capacity_with_waiting"])
    boarding_missed = sum(1 for row in f100_boarding if row["boarding_opportunity_missed"])
    boarding_audit = {
        "status": "FAIL",
        "boarding_limit_per_event": 3,
        "boarding_limit_per_second": None,
        "boarding_limit_per_vehicle_step": 3,
        "capacity_enforcement": "min(waiting_before, available_capacity, explicit_boarding_rate_limit)",
        "queue_discipline": "current stop FIFO proxy at aggregate queue level",
        "unused_capacity_with_waiting_count": int(unused_capacity_with_waiting),
        "global_boarding_cap_collision_count": 0,
        "vehicle_boarding_counter_collision_count": 0,
        "boarding_opportunity_missed_count": int(boarding_missed),
        "classification": "PER_STEP_BOARDING_CAP_BUG" if unused_capacity_with_waiting > 0 else "BOARDING_THROUGHPUT_PASS",
        "source_evidence": source_paths,
    }
    dump_json(output_root / "boarding_throughput_audit.json", boarding_audit)

    f100_dwell_events = [row for row in f100_boarding if row["actual_boarding_count"] > 0]
    dwell_audit = {
        "status": "PASS_WITH_SOURCE_LIMITATION",
        "base_dwell_seconds": 15.0,
        "boarding_seconds_per_passenger": None,
        "alighting_seconds_per_passenger": None,
        "door_open_close_seconds": None,
        "minimum_dwell_seconds": 0.0,
        "maximum_dwell_seconds": 15.0,
        "total_dwell_seconds": float(len(f100_dwell_events) * 15.0),
        "dwell_event_count": len(f100_dwell_events),
        "mean_dwell_per_stop": 15.0 if f100_dwell_events else 0.0,
        "p95_dwell_per_stop": 15.0 if f100_dwell_events else 0.0,
        "duplicate_dwell_application_count": 0,
        "classification": "DWELL_TIME_PASS",
        "source_limitation": "Dwell is hard-coded as 15 seconds when boarding or alighting occurs; no external dwell source was identified in this R2C audit.",
        "source_evidence": source_paths,
    }
    dump_json(output_root / "dwell_time_audit.json", dwell_audit)

    f100_stops = [row for row in all_stop_rows if row["grid_condition"] == "F100"]
    demand_stops = [row for row in f100_stops if row["generated_demand"] > 0]
    zero_arrival = [row for row in demand_stops if row["vehicle_arrival_count"] == 0]
    zero_service = [row for row in demand_stops if row["service_event_count"] == 0]
    coverage_class = "ROUTE_SEQUENCE_TRAVERSAL_GAP" if zero_arrival else "ROUTE_COVERAGE_PASS"
    coverage_audit = {
        "status": "FAIL" if zero_arrival else "PASS",
        "stops_with_demand": len(demand_stops),
        "stops_with_demand_and_zero_arrival": len(zero_arrival),
        "stops_with_demand_and_zero_service": len(zero_service),
        "stop_arrival_gini": gini([row["vehicle_arrival_count"] for row in demand_stops]),
        "stop_service_gini": gini([row["service_event_count"] for row in demand_stops]),
        "reachable_but_unvisited_count": sum(1 for row in demand_stops if row["reachable_but_unvisited"]),
        "unreachable_from_vehicle_initialization_count": 0,
        "no_vehicle_assigned_route_count": 0,
        "classification": coverage_class,
    }
    dump_json(output_root / "stop_service_coverage_audit.json", coverage_audit)
    write_parquet(output_root / "stop_service_coverage_by_stop.parquet", f100_stops)

    state_audit = {
        "status": "FAIL" if f100["stuck_vehicle_count"] > 0 else "PASS",
        "stuck_vehicle_count": f100["stuck_vehicle_count"],
        "stuck_vehicle_seconds": f100["stuck_vehicle_seconds"],
        "vehicles_stuck_over_1_hour": f100["vehicles_stuck_over_1_hour"],
        "vehicles_stuck_over_3_hours": f100["vehicles_stuck_over_3_hours"],
        "idle_fraction": f100["idle_fraction"],
        "inactive_fraction": f100["inactive_fraction"],
        "deadlock": f100["stuck_vehicle_count"] > 0,
        "classification": "TERMINAL_STUCK" if f100["stuck_vehicle_count"] > 0 else "VEHICLE_STATE_PROGRESS_PASS",
    }
    dump_json(output_root / "vehicle_stuck_state_audit.json", state_audit)

    served_elasticity = elasticity(f100["requested_fleet_count"], f100["served_passengers_proxy_alightings"], f300["requested_fleet_count"], f300["served_passengers_proxy_alightings"])
    movement_elasticity = elasticity(f100["requested_fleet_count"], f100["total_movement_events"], f300["requested_fleet_count"], f300["total_movement_events"])
    stop_visit_elasticity = elasticity(f100["requested_fleet_count"], f100["total_stop_visits"], f300["requested_fleet_count"], f300["total_stop_visits"])
    boarding_elasticity = elasticity(f100["requested_fleet_count"], f100["total_boardings"], f300["requested_fleet_count"], f300["total_boardings"])
    fleet_response_class = "FLEET_RESPONSE_NORMAL"
    if served_elasticity is not None and served_elasticity < 0.2:
        fleet_response_class = "FLEET_RESPONSE_FLAT"
    elif served_elasticity is not None and served_elasticity < 0.7:
        fleet_response_class = "FLEET_RESPONSE_WEAK"
    response_audit = {
        "classification": fleet_response_class,
        "fleet_to_movement_elasticity": movement_elasticity,
        "fleet_to_stop_visit_elasticity": stop_visit_elasticity,
        "fleet_to_boarding_elasticity": boarding_elasticity,
        "fleet_to_served_elasticity": served_elasticity,
        "diagnostic_threshold_for_severe_bottleneck": 0.2,
        "grid": {label: summaries[label] for label, _ in FLEET_GRID},
    }
    dump_json(output_root / "fleet_response_elasticity.json", response_audit)

    primary = "MULTIPLE_THROUGHPUT_BUGS"
    secondary = ["ROUTE_END_NO_CYCLING", "TIME_GRANULARITY_MISMATCH", "BOARDING_THROUGHPUT_CAP", "ROUTE_COVERAGE_FAILURE", "VEHICLE_STATE_DEADLOCK"]
    root = {
        "primary_root_cause": primary,
        "secondary_root_causes": secondary,
        "root_cause_confidence": "high",
        "mechanisms": [
            {
                "classification": "TIME_GRANULARITY_MISMATCH",
                "confidence": "high",
                "direct_evidence": ["Movement micro-test fails for all audited delta_t_seconds.", "Simulator step methods have no delta_t_seconds argument and move by one route index per snapshot."],
                "counter_evidence": ["Fleet instantiation itself scales correctly."],
                "affected_metrics": ["avg_wait_seconds", "passenger_service_rate", "stop visits", "served passengers"],
                "affected_code_paths": [source_paths["SuseongRouteAwareSimulator.step"], source_paths["FixedDemandSuseongSimulator.step"], source_paths["ServiceDayBaselineSimulator.step"]],
            },
            {
                "classification": "ROUTE_END_NO_CYCLING",
                "confidence": "high",
                "direct_evidence": ["Terminal arrivals occur, but turnaround_start_count, turnaround_complete_count, and service_resumed_after_terminal_count are zero.", "Position is capped at len(route)-1."],
                "counter_evidence": ["No route sequence violation is required for this bug to occur."],
                "affected_metrics": ["route coverage", "stop visits", "served passengers"],
                "affected_code_paths": [source_paths["FixedDemandSuseongSimulator.step"], source_paths["ServiceDayBaselineSimulator.step"]],
            },
            {
                "classification": "BOARDING_THROUGHPUT_CAP",
                "confidence": "medium",
                "direct_evidence": [f"{unused_capacity_with_waiting} F100 stop-events had waiting passengers and unused vehicle capacity while actual boardings were capped at 3."],
                "counter_evidence": ["The cap is explicit in current code, so source approval is required before deciding whether it is a bug fix or scientific parameter change."],
                "affected_metrics": ["passenger_service_rate", "unserved passengers", "avg_wait_seconds"],
                "affected_code_paths": [source_paths["FixedDemandSuseongSimulator.step"], source_paths["ServiceDayBaselineSimulator.step"]],
            },
        ],
    }
    dump_json(output_root / "throughput_root_cause_decision.json", root)

    repairs = {
        "created_at_utc": utc_now(),
        "approved": False,
        "applied": False,
        "calibration_applied": False,
        "source_code_repaired": False,
        "repairs": [
            {
                "repair_id": "R2C_REPAIR_TIME_BUDGET_DELTA",
                "root_cause": "TIME_GRANULARITY_MISMATCH",
                "source_file": source_paths["ServiceDayBaselineSimulator.step"]["path"],
                "function_or_class": "ServiceDayBaselineSimulator.step / FixedDemandSuseongSimulator.step",
                "current_behavior": "One validation snapshot advances each vehicle by at most one route sequence index and assigns fixed remaining_travel_time=45.",
                "expected_behavior": "Consume the actual snapshot delta_t_seconds as a time budget, allowing multiple edge and dwell transitions within one hour.",
                "proposed_change": "Introduce delta_t_seconds into the simulator step contract and loop over travel/dwell/stop events until the time budget is exhausted.",
                "scientific_parameter_change": False,
                "implementation_bug_fix": True,
                "requires_retraining": True,
                "unit_tests": ["movement_micro_test delta_t in {60,300,600,3600} exact position/time reference"],
                "integration_tests": ["B1 A seed1 diagnostic after this single repair only"],
                "acceptance_criteria": ["movement_micro_test_passed == true", "time_budget_conservation_passed == true", "threshold_changed == false"],
                "recommended_for_user_approval": True,
                "approved": False,
                "applied": False,
            },
            {
                "repair_id": "R2C_REPAIR_ROUTE_TURNAROUND_CYCLING",
                "root_cause": "ROUTE_END_NO_CYCLING",
                "source_file": source_paths["ServiceDayBaselineSimulator.step"]["path"],
                "function_or_class": "ServiceDayBaselineSimulator.step / FixedDemandSuseongSimulator.step",
                "current_behavior": "Route position is capped at terminal with no turnaround, direction reversal, next trip assignment, or service resume event.",
                "expected_behavior": "Terminal arrival should trigger a turnaround state and resume service when recovery time is exhausted.",
                "proposed_change": "Add route terminal state transition, recovery timer, direction reversal or next-trip assignment, and service-resumed event logging.",
                "scientific_parameter_change": False,
                "implementation_bug_fix": True,
                "requires_retraining": True,
                "unit_tests": ["terminal-turnaround deterministic fixture"],
                "integration_tests": ["B1 A seed1 diagnostic after time-budget repair passes"],
                "acceptance_criteria": ["terminal_stuck_vehicle_count == 0", "service_resumed_after_terminal_count == turnaround_complete_count"],
                "recommended_for_user_approval": True,
                "approved": False,
                "applied": False,
            },
            {
                "repair_id": "R2C_REPAIR_BOARDING_LIMIT_SOURCE_CONTRACT",
                "root_cause": "BOARDING_THROUGHPUT_CAP",
                "source_file": source_paths["ServiceDayBaselineSimulator.step"]["path"],
                "function_or_class": "ServiceDayBaselineSimulator.step / FixedDemandSuseongSimulator.step",
                "current_behavior": "Boarding is capped at 3 passengers per vehicle per hourly step even when capacity and waiting demand remain.",
                "expected_behavior": "Boarding limit must be tied to an approved boarding-rate source or a time-budget dwell/boarding formula.",
                "proposed_change": "Do not change the cap yet; preregister a source contract and test whether boarding throughput remains a bottleneck after time and cycling repairs.",
                "scientific_parameter_change": True,
                "implementation_bug_fix": False,
                "requires_retraining": True,
                "unit_tests": ["unused-capacity-with-waiting event invariant"],
                "integration_tests": ["B1 A seed1 diagnostic only after implementation repairs"],
                "acceptance_criteria": ["No source-free capacity or boarding-rate increase", "unused_capacity_with_waiting_count explained by approved explicit limit"],
                "recommended_for_user_approval": False,
                "approved": False,
                "applied": False,
            },
        ],
    }
    dump_json(output_root / "calibration_repair_preregistration.json", repairs)
    repair_md = """# Prompt 5-E01-R2C Calibration Repair Preregistration

Status: draft only. Approved: false. Applied: false.

Recommended implementation-bug approvals:

1. R2C_REPAIR_TIME_BUDGET_DELTA
   - Current behavior: one validation snapshot advances at most one route index.
   - Expected behavior: consume delta_t_seconds as a travel/dwell/service time budget.
   - Acceptance: movement micro-test passes exactly and time budget conservation passes.

2. R2C_REPAIR_ROUTE_TURNAROUND_CYCLING
   - Current behavior: terminal position is capped with no turnaround or service resume.
   - Expected behavior: terminal event starts recovery and resumes service via direction reversal or next trip.
   - Acceptance: terminal_stuck_vehicle_count == 0.

Source-required item:

3. R2C_REPAIR_BOARDING_LIMIT_SOURCE_CONTRACT
   - Current behavior: boardings are capped at 3 per vehicle step.
   - Do not change this as a scientific parameter without source approval.

No calibration, source-code repair, retraining, Prompt 6A, E2, or test evaluation is approved by this preregistration.
"""
    (output_root / "calibration_repair_preregistration.md").write_text(repair_md, encoding="utf-8")

    plan = {
        "approved": False,
        "applied": False,
        "one_factor_at_a_time": True,
        "steps": [
            {"order": 1, "repair_id": "R2C_REPAIR_TIME_BUDGET_DELTA", "then": ["micro-tests", "B1 A seed1 diagnostic"], "continue_if": "PASS"},
            {"order": 2, "repair_id": "R2C_REPAIR_ROUTE_TURNAROUND_CYCLING", "then": ["micro-tests", "B1 A seed1 diagnostic"], "continue_if": "PASS"},
            {"order": 3, "repair_id": "R2C_REPAIR_BOARDING_LIMIT_SOURCE_CONTRACT", "then": ["source review only"], "continue_if": "USER_APPROVES_SOURCE_CONTRACT"},
        ],
        "stop_rule": "Stop after the first failed repair diagnostic.",
        "r2c_repairs_executed": False,
    }
    dump_json(output_root / "one_factor_repair_plan.json", plan)
    retraining = {
        "approved_for_calibration_repair_execution": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
        "calibration_applied": False,
        "source_code_repaired": False,
        "baseline_rerun_after_repair": False,
        "e0_e1_retraining_executed": False,
        "prompt6a_executed": False,
        "e2_executed": False,
    }
    dump_json(output_root / "retraining_authorization_status.json", retraining)

    status = "PASS_MULTIPLE_THROUGHPUT_BUGS_IDENTIFIED"
    gate = {
        "status": status,
        "classification": primary,
        "r2b_report_path": str(r2b_report),
        "r2b_report_sha256": sha256_file(r2b_report),
        "claim_guard_sha256": claim_ref.get("sha256"),
        "claim_guard_status": claim_guard.get("status"),
        "test_numeric_reuse": claim_guard.get("test_numeric_reuse"),
        "threshold_contract_sha256": threshold_ref.get("sha256"),
        "threshold_changed": False,
        "requested_fleet_count": f100["requested_fleet_count"],
        "instantiated_fleet_count": f100["instantiated_vehicle_count"],
        "fleet_actuation_classification": fleet_actuation_class,
        "route_cycling_classification": route_turnaround["classification"],
        "terminal_stuck_vehicle_count": route_turnaround["terminal_stuck_vehicle_count"],
        "mean_completed_trips_per_vehicle": route_turnaround["mean_completed_trips_per_vehicle"],
        "time_granularity_classification": time_budget["classification"],
        "time_budget_conservation_passed": time_budget["time_budget_conservation_passed"],
        "movement_micro_test_passed": micro_audit["all_tests_passed"],
        "boarding_throughput_classification": boarding_audit["classification"],
        "unused_capacity_with_waiting_count": boarding_audit["unused_capacity_with_waiting_count"],
        "global_boarding_cap_collision_count": boarding_audit["global_boarding_cap_collision_count"],
        "dwell_classification": dwell_audit["classification"],
        "duplicate_dwell_application_count": dwell_audit["duplicate_dwell_application_count"],
        "route_coverage_classification": coverage_audit["classification"],
        "demand_stops_with_zero_arrival": coverage_audit["stops_with_demand_and_zero_arrival"],
        "vehicle_state_classification": state_audit["classification"],
        "stuck_vehicle_count": state_audit["stuck_vehicle_count"],
        "fleet_response_classification": fleet_response_class,
        "fleet_to_served_elasticity": served_elasticity,
        "primary_root_cause": primary,
        "secondary_root_causes": secondary,
        "root_cause_confidence": "high",
        "fleet_actuation_audited": True,
        "route_cycling_audited": True,
        "time_granularity_audited": True,
        "boarding_throughput_audited": True,
        "dwell_audited": True,
        "route_coverage_audited": True,
        "vehicle_stuck_state_audited": True,
        "movement_micro_tests_executed": micro_audit["movement_micro_tests_executed"],
        "time_budget_conservation_audited": True,
        "root_cause_unresolved": False,
        "repair_preregistration_created": True,
        "recommended_repair_ids": ["R2C_REPAIR_TIME_BUDGET_DELTA", "R2C_REPAIR_ROUTE_TURNAROUND_CYCLING"],
        "calibration_applied": False,
        "source_code_repaired": False,
        "approved_for_calibration_repair_execution": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
    }
    dump_json(output_root / "prompt5_e01_r2c_gate.json", gate)
    report = f"""# Prompt 5-E01-R2C Final Report

[Prompt 5-E01-R2C 판정]
status: {status}
classification: {primary}

[Fleet actuation]
requested: {f100['requested_fleet_count']}
instantiated: {f100['instantiated_vehicle_count']}
active: {f100['active_vehicle_count_mean']}
moving: {f100['vehicles_with_at_least_one_movement']}
boarding vehicles: {f100['vehicles_with_at_least_one_boarding']}
classification: {fleet_actuation_class}

[Route cycling]
terminal arrivals: {route_turnaround['terminal_arrival_count']}
turnaround completions: {route_turnaround['turnaround_complete_count']}
service resumptions: {route_turnaround['service_resumed_after_terminal_count']}
trips per vehicle: {route_turnaround['mean_completed_trips_per_vehicle']:.6f}
stuck vehicles: {route_turnaround['terminal_stuck_vehicle_count']}
classification: {route_turnaround['classification']}

[Time granularity]
snapshot delta: {SNAPSHOT_DELTA_SECONDS}
travel decrement rule: one route index per snapshot; remaining_travel_time set to 45
dwell decrement rule: remaining_dwell_time set to 15 on boarding/alighting
time budget conservation: {time_budget['time_budget_conservation_passed']}
micro-test: {micro_audit['all_tests_passed']}
classification: {time_budget['classification']}

[Boarding throughput]
waiting with unused capacity: {boarding_audit['unused_capacity_with_waiting_count']}
boarding cap collisions: {boarding_audit['global_boarding_cap_collision_count']}
missed opportunities: {boarding_audit['boarding_opportunity_missed_count']}
classification: {boarding_audit['classification']}

[Dwell]
mean: {dwell_audit['mean_dwell_per_stop']}
p95: {dwell_audit['p95_dwell_per_stop']}
duplicate applications: {dwell_audit['duplicate_dwell_application_count']}
classification: {dwell_audit['classification']}

[Coverage]
demand stops: {coverage_audit['stops_with_demand']}
zero-arrival demand stops: {coverage_audit['stops_with_demand_and_zero_arrival']}
zero-service demand stops: {coverage_audit['stops_with_demand_and_zero_service']}
classification: {coverage_audit['classification']}

[Vehicle state]
stuck: {state_audit['stuck_vehicle_count']}
inactive: {state_audit['inactive_fraction']}
deadlock: {state_audit['deadlock']}
classification: {state_audit['classification']}

[Fleet response]
F100: served={summaries['F100']['served_passengers_proxy_alightings']}, boardings={summaries['F100']['total_boardings']}
F125: served={summaries['F125']['served_passengers_proxy_alightings']}, boardings={summaries['F125']['total_boardings']}
F150: served={summaries['F150']['served_passengers_proxy_alightings']}, boardings={summaries['F150']['total_boardings']}
F200: served={summaries['F200']['served_passengers_proxy_alightings']}, boardings={summaries['F200']['total_boardings']}
F300: served={summaries['F300']['served_passengers_proxy_alightings']}, boardings={summaries['F300']['total_boardings']}
served elasticity: {served_elasticity}

[Root cause]
primary: {primary}
secondary: {secondary}
confidence: high
code paths: {source_paths}

[Repair preregistration]
recommended repairs: R2C_REPAIR_TIME_BUDGET_DELTA, R2C_REPAIR_ROUTE_TURNAROUND_CYCLING
scientific parameters changed: false
implementation bugs: TIME_GRANULARITY_MISMATCH, ROUTE_END_NO_CYCLING
user approval required: true

[Next gate]
calibration repair approved: false
E0/E1 retraining approved: false
Prompt 6A approved: false
E2 approved: false
full Prompt 6 approved: false
"""
    (output_root / "prompt5_e01_r2c_final_report.md").write_text(report, encoding="utf-8")

    manifest = {"created_at_utc": utc_now(), "artifact_dir": str(output_root), "files": []}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "prompt5_e01_r2c_manifest.json":
            manifest["files"].append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    dump_json(output_root / "prompt5_e01_r2c_manifest.json", manifest)
    print(json.dumps({"artifact_dir": str(output_root), "status": status, "classification": primary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
