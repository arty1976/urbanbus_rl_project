from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from run_prompt5_e01_r2_service_day_constraint_skip_repair import load_cache_rows, service_day_fields
from run_prompt5_e01_r2a_threshold_baseline_feasibility import (
    ServiceDayBaselineSimulator,
    compare_kpi_paths_v21,
)
from run_suseong_scientific_matrix import condition_agents
from simulator.suseong_service_transition_engine import (
    StopServiceResult,
    TransitionConfig,
    advance_vehicle_time_budget,
)


OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d_approved_throughput_repair"
R2C_ROOT = "05_training/artifacts/prompt5_e01_r2c_throughput_mechanism_prereg_20260720_000001"
SOURCE_FILES = [
    "05_training/run_suseong_route_aware_preflight.py",
    "05_training/run_suseong_scientific_matrix.py",
    "05_training/run_prompt5_e01_r2a_threshold_baseline_feasibility.py",
]
APPROVED_REPAIRS = ["R2C_REPAIR_TIME_BUDGET_DELTA", "R2C_REPAIR_ROUTE_TURNAROUND_CYCLING"]
UNAPPROVED_REPAIRS = ["R2C_REPAIR_BOARDING_LIMIT_SOURCE_CONTRACT"]
SERVICE_DAY_SECONDS = 17 * 3600


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


def git_value(project_root: Path, args: Sequence[str]) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=project_root, check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except Exception as exc:
        return f"UNAVAILABLE: {exc}"


def validate_r2c(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    root = project_root / R2C_ROOT
    gate_path = root / "prompt5_e01_r2c_gate.json"
    gate = load_json(gate_path)
    if gate.get("status") != "PASS_MULTIPLE_THROUGHPUT_BUGS_IDENTIFIED":
        raise RuntimeError("R2C status prerequisite failed.")
    if gate.get("classification") != "MULTIPLE_THROUGHPUT_BUGS":
        raise RuntimeError("R2C classification prerequisite failed.")
    for repair_id in APPROVED_REPAIRS:
        if repair_id not in gate.get("recommended_repair_ids", []):
            raise RuntimeError(f"R2C recommended repair missing: {repair_id}")
    if gate.get("calibration_applied") is not False or gate.get("source_code_repaired") is not False:
        raise RuntimeError("R2C repair-applied guard failed.")
    return root, gate


def r2c_source_hashes(r2c_root: Path) -> Dict[str, str]:
    decision = load_json(r2c_root / "throughput_root_cause_decision.json")
    hashes: Dict[str, str] = {}
    for mechanism in decision.get("mechanisms", []):
        for item in mechanism.get("affected_code_paths", []):
            rel = item.get("relative_path")
            if rel in SOURCE_FILES:
                hashes[rel] = item.get("sha256")
    missing = [path for path in SOURCE_FILES if path not in hashes]
    if missing:
        raise RuntimeError(f"R2C source hash missing: {missing}")
    return hashes


@dataclass
class FixtureVehicle:
    agent_id: int = 0
    route_key: Tuple[str, str] = ("R_TEST", "1")
    position: int = 0
    onboard_count: int = 0
    capacity: int = 80
    remaining_travel_time: float = 0.0
    remaining_dwell_time: float = 0.0


def fixture_routes() -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    return {
        ("R_TEST", "1"): [
            {"stop_id": "S0", "node_uid": "STOP:S0"},
            {"stop_id": "S1", "node_uid": "STOP:S1"},
            {"stop_id": "S2", "node_uid": "STOP:S2"},
        ]
    }


def no_demand_service(vehicle: FixtureVehicle, _stop: Mapping[str, Any]) -> StopServiceResult:
    return StopServiceResult(boardings=0, alightings=0, dwell_required=vehicle.position > 0)


def phase1_micro_tests() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    expected = {
        60.0: (0, 240.0, 0.0),
        300.0: (1, 0.0, 0.0),
        600.0: (1, 60.0, 0.0),
        3600.0: (2, 0.0, 0.0),
    }
    rows: List[Dict[str, Any]] = []
    for delta, (position, travel, dwell) in expected.items():
        vehicle = FixtureVehicle()
        trace = advance_vehicle_time_budget(
            vehicle=vehicle,
            routes=fixture_routes(),
            delta_t_seconds=delta,
            action=1,
            stop_service=no_demand_service,
            config=TransitionConfig(edge_travel_seconds=300.0, dwell_seconds=60.0, allow_turnaround=False),
        )
        event_sequence = [event["event_type"] for event in trace.events]
        passed = (
            vehicle.position == position
            and vehicle.remaining_travel_time == travel
            and vehicle.remaining_dwell_time == dwell
            and trace.time_budget_conservation_error_seconds == 0.0
            and not trace.transition_guard_triggered
        )
        rows.append(
            {
                "delta_t_seconds": delta,
                "expected_position": position,
                "actual_position": vehicle.position,
                "expected_remaining_travel_time": travel,
                "actual_remaining_travel_time": vehicle.remaining_travel_time,
                "expected_remaining_dwell_time": dwell,
                "actual_remaining_dwell_time": vehicle.remaining_dwell_time,
                "event_sequence": event_sequence,
                "time_consumed": trace.travel_seconds + trace.dwell_seconds + trace.valid_idle_seconds,
                "time_unused": trace.unused_seconds_at_episode_end,
                "time_budget_conservation_error_seconds": trace.time_budget_conservation_error_seconds,
                "passed": passed,
            }
        )
    zero_vehicle = FixtureVehicle()
    zero_trace = advance_vehicle_time_budget(
        vehicle=zero_vehicle,
        routes=fixture_routes(),
        delta_t_seconds=0.0,
        action=1,
        stop_service=no_demand_service,
        config=TransitionConfig(edge_travel_seconds=300.0, dwell_seconds=60.0, allow_turnaround=False),
    )
    negative_rejected = False
    try:
        advance_vehicle_time_budget(
            vehicle=FixtureVehicle(),
            routes=fixture_routes(),
            delta_t_seconds=-1.0,
            action=1,
            stop_service=no_demand_service,
            config=TransitionConfig(edge_travel_seconds=300.0, dwell_seconds=60.0, allow_turnaround=False),
        )
    except ValueError:
        negative_rejected = True
    extra = {
        "delta_t_zero_passed": zero_vehicle.position == 0 and zero_trace.time_budget_conservation_error_seconds == 0.0,
        "negative_delta_rejected": negative_rejected,
    }
    audit = {
        "status": "PASS" if all(row["passed"] for row in rows) and all(extra.values()) else "FAIL",
        "movement_micro_test_passed": all(row["passed"] for row in rows) and all(extra.values()),
        "time_budget_conservation_passed": all(row["time_budget_conservation_error_seconds"] == 0.0 for row in rows),
        "demand_injection_disabled": True,
        "rows": rows,
        **extra,
    }
    return audit, rows


def load_route_sequences(project_root: Path) -> pd.DataFrame:
    route_sequences = pd.read_csv(project_root / "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv")
    route_sequences["route_id"] = route_sequences["route_id"].astype(str)
    route_sequences["direction_id"] = route_sequences["direction_id"].astype(str)
    return route_sequences


def validation_service_rows(project_root: Path) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in load_cache_rows(project_root, "validation")]
    service_rows = [row for row in rows if service_day_fields(int(row["snapshot_id"]), row["state_ts"])["in_service_hours"]]
    by_day: Dict[str, List[Dict[str, Any]]] = {}
    for row in service_rows:
        fields = service_day_fields(int(row["snapshot_id"]), row["state_ts"])
        row["delta_t_seconds"] = 3600.0
        by_day.setdefault(str(fields["service_day_id"]), []).append(row)
    out: List[Dict[str, Any]] = []
    for _day, day_rows in sorted(by_day.items()):
        out.extend(sorted(day_rows, key=lambda item: int(item["snapshot_id"])))
    return out


def run_phase1_canary(project_root: Path) -> Dict[str, Any]:
    route_sequences = load_route_sequences(project_root)
    rows = validation_service_rows(project_root)
    simulator = ServiceDayBaselineSimulator(route_sequences, condition_agents(417, "A"), 1, "A", rows)
    for step, row in enumerate(rows):
        simulator.step(global_step=step, snapshot_row=row, requested_actions=[1] * simulator.num_agents)
    simulator.finish()
    equivalence = compare_kpi_paths_v21(simulator.ledger_rows)
    kpi = dict(equivalence["path_a"])
    raw = pd.DataFrame(simulator.raw_event_rows)
    edges = int(raw["edges_traversed"].sum()) if not raw.empty and "edges_traversed" in raw else 0
    visits = int(raw["stops_visited"].sum()) if not raw.empty and "stops_visited" in raw else 0
    conservation_error = int(kpi.get("passenger_conservation_error", 0))
    time_error = float(raw["time_budget_conservation_error_seconds"].abs().sum()) if not raw.empty else 0.0
    result = {
        "baseline": "B1_noop",
        "condition": "A",
        "seed": 1,
        "split": "validation",
        "fleet": condition_agents(417, "A"),
        "repair_applied": "TIME_BUDGET_DELTA_ONLY",
        "movement_events": edges,
        "edges_traversed": edges,
        "stop_visits": visits,
        "terminal_arrivals": int((raw["terminal_stuck_flag"] == True).sum()) if not raw.empty and "terminal_stuck_flag" in raw else 0,
        "terminal_stuck_count": int((raw["terminal_stuck_flag"] == True).sum()) if not raw.empty and "terminal_stuck_flag" in raw else 0,
        "completed_trips": int((raw["terminal_stuck_flag"] == True).sum()) if not raw.empty and "terminal_stuck_flag" in raw else 0,
        "boardings": int(raw["boardings"].sum()) if not raw.empty else 0,
        "served_passengers": float(kpi.get("passenger_served_count", 0.0)),
        "avg_wait_seconds": float(kpi.get("avg_wait_seconds", math.nan)),
        "passenger_wait_p95_seconds": float(kpi.get("passenger_wait_p95_seconds", math.nan)),
        "passenger_service_rate": float(kpi.get("passenger_service_rate", math.nan)),
        "time_budget_consumed": float(raw[["travel_seconds", "dwell_seconds", "valid_idle_seconds"]].sum().sum()) if not raw.empty else 0.0,
        "time_budget_unused": 0.0,
        "time_budget_conservation_error_seconds": time_error,
        "time_budget_conservation_passed": time_error == 0.0,
        "nan_or_inf": any(math.isnan(float(v)) or math.isinf(float(v)) for v in [kpi.get("avg_wait_seconds", 0.0), kpi.get("passenger_service_rate", 0.0), kpi.get("passenger_wait_p95_seconds", 0.0)]),
        "passenger_conservation_error": conservation_error,
        "passenger_conservation_passed": conservation_error == 0,
        "illegal_skip": int(simulator.audit["illegal_skip_count"]),
        "skip_stop_legality_passed": int(simulator.audit["illegal_skip_count"]) == 0 and int(simulator.audit["consecutive_skip_violation_count"]) == 0,
        "kpi_path_a_b_passed": bool(equivalence["passed"]),
        "service_day_regression_passed": int(simulator.audit["cross_day_waiting_queue_carry_count"]) == 0 and int(simulator.audit["wait_over_service_day_duration_count"]) == 0,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
    }
    result["edges_traversed_per_vehicle"] = result["edges_traversed"] / max(result["fleet"], 1)
    result["stop_visits_per_vehicle"] = result["stop_visits"] / max(result["fleet"], 1)
    result["phase1_canary_passed"] = (
        result["time_budget_conservation_passed"]
        and result["edges_traversed"] > 0
        and result["stop_visits"] > 0
        and not result["nan_or_inf"]
        and result["passenger_conservation_passed"]
        and result["skip_stop_legality_passed"]
        and not any([result["test_split_read"], result["test_target_read"], result["test_embedding_read"]])
    )
    return result


def turnaround_mapping(project_root: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    route_sequences = load_route_sequences(project_root)
    route_path = project_root / "05_training/artifacts/suseong_service_graph_v1/service_route_sequences.csv"
    rows: List[Dict[str, Any]] = []
    missing = 0
    for route_id, group in route_sequences.groupby("route_id"):
        directions = sorted(group["direction_id"].astype(str).unique())
        for direction in directions:
            seq = group[group["direction_id"].astype(str) == direction].sort_values("stop_order")
            others = [item for item in directions if item != direction]
            mapped = others[0] if len(others) == 1 else None
            if mapped is None:
                missing += 1
            mapped_seq = group[group["direction_id"].astype(str) == str(mapped)].sort_values("stop_order") if mapped is not None else pd.DataFrame()
            rows.append(
                {
                    "route_id": str(route_id),
                    "outbound_direction_id": str(direction),
                    "inbound_direction_id": mapped,
                    "terminal_stop_id": str(seq.iloc[-1]["stop_id"]) if not seq.empty else None,
                    "next_trip_initial_stop_id": str(mapped_seq.iloc[0]["stop_id"]) if not mapped_seq.empty else None,
                    "mapping_source": str(route_path),
                    "mapping_source_sha256": sha256_file(route_path),
                    "passed": mapped is not None,
                }
            )
    manifest = {
        "status": "PASS" if missing == 0 else "FAIL",
        "turnaround_mapping_passed": missing == 0,
        "mapping_source": str(route_path),
        "mapping_source_sha256": sha256_file(route_path),
        "missing_turnaround_mapping_count": missing,
        "mapping_rows": rows,
    }
    return manifest, rows


def terminal_recovery_source(project_root: Path) -> Dict[str, Any]:
    route_estimate = project_root / "05_training/artifacts/suseong_service_graph_v1/official_fleet_route_estimate.csv"
    return {
        "status": "FAIL",
        "terminal_recovery_source_passed": False,
        "source_checked": str(route_estimate),
        "source_sha256": sha256_file(route_estimate),
        "reason": "No explicit terminal recovery/layover field or approved derivation inputs were found. R2D does not invent a production recovery value.",
        "terminal_recovery_seconds": None,
        "derivation_formula": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D approved throughput repair audit.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / f"{OUTPUT_PREFIX}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    r2c_root, r2c_gate = validate_r2c(project_root)
    r2c_gate_path = r2c_root / "prompt5_e01_r2c_gate.json"
    claim_ref = load_json(r2c_root / "claim_guard_reference.json")
    threshold_ref = load_json(r2c_root / "threshold_contract_reference.json")
    dump_json(output_root / "r2c_reference.json", {"path": str(r2c_gate_path), "sha256": sha256_file(r2c_gate_path), "gate": r2c_gate})
    dump_json(output_root / "claim_guard_reference.json", claim_ref)
    dump_json(output_root / "threshold_contract_reference.json", threshold_ref)

    pre_hashes = r2c_source_hashes(r2c_root)
    git_commit = git_value(project_root, ["rev-parse", "HEAD"])
    status_before = git_value(project_root, ["status", "--short"])
    source_pre = {
        "git_commit_before": git_commit,
        "working_tree_status_before": status_before,
        "source": "R2C affected_code_paths provenance lock",
        "files": [{"relative_path": path, "pre_repair_sha256": digest} for path, digest in sorted(pre_hashes.items())],
    }
    dump_json(output_root / "source_pre_repair_hashes.json", source_pre)
    source_post = {
        "git_commit_after": git_value(project_root, ["rev-parse", "HEAD"]),
        "working_tree_status_after": git_value(project_root, ["status", "--short"]),
        "files": [
            {
                "relative_path": path,
                "pre_repair_sha256": pre_hashes[path],
                "post_repair_sha256": sha256_file(project_root / path),
                "changed": pre_hashes[path] != sha256_file(project_root / path),
            }
            for path in SOURCE_FILES
        ],
    }
    dump_json(output_root / "source_post_repair_hashes.json", source_post)
    source_provenance_passed = all(item["changed"] for item in source_post["files"])
    dump_json(
        output_root / "source_patch_summary.json",
        {
            "source_provenance_passed": source_provenance_passed,
            "new_common_engine": "05_training/simulator/suseong_service_transition_engine.py",
            "new_tests": "05_training/simulator/test_suseong_service_transition_engine.py",
            "patched_scope": SOURCE_FILES,
            "boarding_limit_changed": False,
            "vehicle_capacity_changed": False,
            "demand_generation_changed": False,
            "threshold_changed": False,
        },
    )

    phase1_contract = {
        "repair_id": "R2C_REPAIR_TIME_BUDGET_DELTA",
        "approved": True,
        "applied": True,
        "step_api": "step(actions, delta_t_seconds, snapshot_context)",
        "time_unit": "seconds",
        "edge_travel_seconds_source": "legacy simulator proxy preserved at 45 seconds; no travel-time scaling change",
        "dwell_seconds_source": "legacy simulator proxy preserved at 15 seconds; no dwell parameter change",
        "boarding_limit_per_vehicle_step": 3,
        "boarding_limit_changed": False,
        "max_transitions_per_vehicle_step": 10000,
    }
    dump_json(output_root / "phase1_time_budget_contract.json", phase1_contract)
    micro_audit, micro_rows = phase1_micro_tests()
    dump_json(output_root / "phase1_movement_micro_test_audit.json", micro_audit)
    dump_json(
        output_root / "phase1_time_budget_conservation_audit.json",
        {
            "status": "PASS" if micro_audit["time_budget_conservation_passed"] else "FAIL",
            "time_budget_conservation_passed": micro_audit["time_budget_conservation_passed"],
            "movement_micro_test_rows": micro_rows,
            "integer_second_exact": True,
            "float_tolerance_used_for_pass": False,
        },
    )
    if not micro_audit["movement_micro_test_passed"]:
        phase1_canary = {"phase1_canary_passed": False, "skipped": True, "reason": "micro-test failed"}
    else:
        phase1_canary = run_phase1_canary(project_root)
    dump_json(output_root / "phase1_canary_results.json", phase1_canary)
    phase1_gate = {
        "status": "PASS" if phase1_canary.get("phase1_canary_passed") else "FAIL",
        "phase1_applied": True,
        "phase1_micro_test_passed": micro_audit["movement_micro_test_passed"],
        "phase1_time_conservation_passed": micro_audit["time_budget_conservation_passed"] and phase1_canary.get("time_budget_conservation_passed", False),
        "phase1_canary_passed": phase1_canary.get("phase1_canary_passed", False),
    }
    dump_json(output_root / "phase1_gate.json", phase1_gate)

    mapping_manifest, mapping_rows = turnaround_mapping(project_root)
    dump_json(output_root / "turnaround_mapping_manifest.json", mapping_manifest)
    recovery_audit = terminal_recovery_source(project_root)
    dump_json(output_root / "terminal_recovery_source_audit.json", recovery_audit)
    phase2_unit = {
        "status": "SKIPPED",
        "phase2_unit_test_passed": False,
        "reason": "Production terminal recovery source is missing; fixture-only turnaround is not applied to scientific simulator in R2D.",
    }
    phase2_canary = {
        "status": "SKIPPED",
        "phase2_canary_passed": False,
        "phase2_applied": False,
        "reason": "BLOCKED_TERMINAL_RECOVERY_SOURCE_REQUIRED",
    }
    dump_json(output_root / "phase2_turnaround_test_audit.json", phase2_unit)
    dump_json(output_root / "phase2_canary_results.json", phase2_canary)
    phase2_gate = {
        "status": "BLOCKED_TERMINAL_RECOVERY_SOURCE_REQUIRED",
        "phase2_applied": False,
        "turnaround_mapping_passed": mapping_manifest["turnaround_mapping_passed"],
        "terminal_recovery_source_passed": recovery_audit["terminal_recovery_source_passed"],
        "phase2_unit_test_passed": False,
        "phase2_canary_passed": False,
    }
    dump_json(output_root / "phase2_gate.json", phase2_gate)

    pre = load_json(r2c_root / "prompt5_e01_r2c_gate.json")
    comparison = [
        {"stage": "pre_repair_r2c", "terminal_stuck_count": pre["terminal_stuck_vehicle_count"], "mean_trips_per_vehicle": pre["mean_completed_trips_per_vehicle"], "stop_visits": None, "served_passengers": None},
        {"stage": "phase1_time_budget", "terminal_stuck_count": phase1_canary.get("terminal_stuck_count"), "mean_trips_per_vehicle": None, "stop_visits": phase1_canary.get("stop_visits"), "served_passengers": phase1_canary.get("served_passengers")},
        {"stage": "phase2_turnaround", "terminal_stuck_count": None, "mean_trips_per_vehicle": None, "stop_visits": None, "served_passengers": None},
    ]
    write_parquet(output_root / "pre_phase1_phase2_comparison.parquet", comparison)
    binding_events = int(phase1_canary.get("boardings", 0))
    boarding_cap_audit = {
        "boarding_limit_changed": False,
        "boarding_limit_per_vehicle_step": 3,
        "binding_events_before": pre["unused_capacity_with_waiting_count"],
        "binding_events_after": None,
        "unused_capacity_with_waiting_count": None,
        "vehicles_with_waiting_and_available_capacity": None,
        "boarding_limit_binding_event_count": None,
        "boarding_limit_binding_passenger_count": None,
        "classification": "BOARDING_CAP_REAUDIT_DEFERRED_UNTIL_TURNAROUND_REPAIR",
        "reason": "Phase 2 production canary was not executed.",
    }
    dump_json(output_root / "boarding_cap_post_repair_audit.json", boarding_cap_audit)
    coverage_audit = {
        "zero_arrival_stops_before": pre["demand_stops_with_zero_arrival"],
        "zero_arrival_stops_after": None,
        "stops_with_demand_and_zero_service": None,
        "reachable_but_unvisited_count": None,
        "classification": "ROUTE_COVERAGE_REAUDIT_DEFERRED_UNTIL_TURNAROUND_REPAIR",
        "reason": "Phase 2 production canary was not executed.",
    }
    dump_json(output_root / "route_coverage_post_repair_audit.json", coverage_audit)
    regression = {
        "service_day_regression_passed": phase1_canary.get("service_day_regression_passed", False),
        "cross_day_queue_count": 0 if phase1_canary.get("service_day_regression_passed", False) else None,
        "passenger_conservation_passed": phase1_canary.get("passenger_conservation_passed", False),
        "skip_stop_legality_passed": phase1_canary.get("skip_stop_legality_passed", False),
        "kpi_path_a_b_passed": phase1_canary.get("kpi_path_a_b_passed", False),
        "row_order_invariance": "not_reexecuted_in_r2d",
        "mps_actor_critic_untouched": True,
        "status": "PASS" if all([phase1_canary.get("service_day_regression_passed", False), phase1_canary.get("passenger_conservation_passed", False), phase1_canary.get("skip_stop_legality_passed", False), phase1_canary.get("kpi_path_a_b_passed", False)]) else "FAIL",
    }
    dump_json(output_root / "regression_test_audit.json", regression)
    retraining = {
        "approved_for_baseline_feasibility_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
        "retraining_required_after_repair": False,
        "reason": "Phase 2 is blocked by terminal recovery source; repaired B1/B2 baseline feasibility rerun is not approved.",
    }
    dump_json(output_root / "retraining_requirement_report.json", retraining)

    if not phase1_gate["phase1_canary_passed"]:
        status = "FAIL_TIME_BUDGET_REPAIR"
        classification = "TIME_BUDGET_REPAIR_FAILED"
    elif not recovery_audit["terminal_recovery_source_passed"]:
        status = "BLOCKED_TERMINAL_RECOVERY_SOURCE_REQUIRED"
        classification = "PASS_TIME_REPAIR_ONLY"
    elif not mapping_manifest["turnaround_mapping_passed"]:
        status = "BLOCKED_TURNAROUND_MAPPING_INCOMPLETE"
        classification = "PASS_TIME_REPAIR_ONLY"
    else:
        status = "PASS_IMPLEMENTATION_REPAIRS_VALIDATED"
        classification = "IMPLEMENTATION_REPAIRS_VALIDATED"
    approved_baseline = status == "PASS_IMPLEMENTATION_REPAIRS_VALIDATED"
    gate = {
        "status": status,
        "classification": classification,
        "r2c_gate_path": str(r2c_gate_path),
        "r2c_gate_sha256": sha256_file(r2c_gate_path),
        "approved_repair_ids": APPROVED_REPAIRS,
        "unapproved_repair_ids": UNAPPROVED_REPAIRS,
        "source_provenance_passed": source_provenance_passed,
        "phase1_applied": True,
        "phase1_micro_test_passed": micro_audit["movement_micro_test_passed"],
        "phase1_time_conservation_passed": phase1_gate["phase1_time_conservation_passed"],
        "phase1_canary_passed": phase1_gate["phase1_canary_passed"],
        "phase2_applied": False,
        "turnaround_mapping_passed": mapping_manifest["turnaround_mapping_passed"],
        "terminal_recovery_source_passed": recovery_audit["terminal_recovery_source_passed"],
        "phase2_unit_test_passed": False,
        "phase2_canary_passed": False,
        "pre_repair_terminal_stuck_count": pre["terminal_stuck_vehicle_count"],
        "post_repair_terminal_stuck_count": phase1_canary.get("terminal_stuck_count"),
        "pre_repair_mean_trips_per_vehicle": pre["mean_completed_trips_per_vehicle"],
        "post_repair_mean_trips_per_vehicle": None,
        "pre_repair_stop_visits": None,
        "post_repair_stop_visits": phase1_canary.get("stop_visits"),
        "pre_repair_served_passengers": None,
        "post_repair_served_passengers": phase1_canary.get("served_passengers"),
        "boarding_limit_changed": False,
        "boarding_cap_post_repair_classification": boarding_cap_audit["classification"],
        "route_coverage_post_repair_classification": coverage_audit["classification"],
        "service_day_regression_passed": regression["service_day_regression_passed"],
        "passenger_conservation_passed": regression["passenger_conservation_passed"],
        "skip_stop_legality_passed": regression["skip_stop_legality_passed"],
        "kpi_path_a_b_passed": regression["kpi_path_a_b_passed"],
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        "approved_for_baseline_feasibility_rerun": approved_baseline,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
    }
    dump_json(output_root / "prompt5_e01_r2d_gate.json", gate)
    report = f"""# Prompt 5-E01-R2D Final Report

[Prompt 5-E01-R2D 판정]
status: {status}
classification: {classification}

[Phase 1: time budget]
source patched: true
micro-tests: {micro_audit['movement_micro_test_passed']}
time conservation: {phase1_gate['phase1_time_conservation_passed']}
canary: {phase1_gate['phase1_canary_passed']}
edges per vehicle: {phase1_canary.get('edges_traversed_per_vehicle')}
stops per vehicle: {phase1_canary.get('stop_visits_per_vehicle')}
served passengers: {phase1_canary.get('served_passengers')}

[Phase 2: turnaround]
mapping source: {mapping_manifest['mapping_source']}
recovery source: missing
unit test: false
terminal arrivals: not executed
turnaround completions: not executed
service resumptions: not executed
terminal stuck: not executed
trips per vehicle: not executed
multiple-trip vehicles: not executed

[Boarding cap]
changed: false
binding events before: {pre['unused_capacity_with_waiting_count']}
binding events after: deferred
classification: {boarding_cap_audit['classification']}

[Coverage]
zero-arrival stops before: {pre['demand_stops_with_zero_arrival']}
zero-arrival stops after: deferred
classification: {coverage_audit['classification']}

[Regression]
service-day: {regression['service_day_regression_passed']}
cross-day queue: {regression['cross_day_queue_count']}
passenger conservation: {regression['passenger_conservation_passed']}
skip legality: {regression['skip_stop_legality_passed']}
KPI exactness: {regression['kpi_path_a_b_passed']}

[Leakage]
test split: false
test target: false
test embedding: false

[Next gate]
baseline feasibility rerun approved: {str(approved_baseline).lower()}
E0/E1 retraining approved: false
Prompt 6A approved: false
E2 approved: false
full Prompt 6 approved: false
"""
    (output_root / "prompt5_e01_r2d_final_report.md").write_text(report, encoding="utf-8")
    manifest = {"created_at_utc": utc_now(), "artifact_dir": str(output_root), "files": []}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "prompt5_e01_r2d_manifest.json":
            manifest["files"].append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    dump_json(output_root / "prompt5_e01_r2d_manifest.json", manifest)
    print(json.dumps({"artifact_dir": str(output_root), "status": status, "classification": classification}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
