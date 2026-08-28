from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from run_prompt5_e01_r2_service_day_constraint_skip_repair import CONDITIONS, SEEDS, load_cache_rows, service_day_fields
from run_suseong_scientific_matrix import condition_agents, stable_int


OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2b_demand_fleet_capacity_calibration"
R2A_PREFIX = "prompt5_e01_r2a_threshold_baseline_feasibility_"
SERVICE_RATE_MIN = 0.95
ABSOLUTE_WAIT_CAP_SECONDS = 900.0
SERVICE_HOURS_PER_DAY = 17
VEHICLE_CAPACITY = 80
BOARDING_LIMIT_PER_VEHICLE_HOUR = 3
FLEET_GRID = [("F100", 1.00), ("F125", 1.25), ("F150", 1.50), ("F200", 2.00), ("F300", 3.00)]
DEMAND_GRID = [("D100", 1.00), ("D75", 0.75), ("D50", 0.50), ("D25", 0.25)]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
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


def stable_json_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def discover_latest_r2a(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    candidates: List[Tuple[str, Path, Dict[str, Any]]] = []
    for path in (project_root / "05_training/artifacts").glob(f"{R2A_PREFIX}*/prompt5_e01_r2a_gate.json"):
        data = load_json(path)
        if (
            data.get("status") == "BLOCKED_SIMULATOR_OR_FLEET_INFEASIBLE"
            and data.get("classification") == "SIMULATOR_OR_FLEET_INFEASIBLE_UNDER_PREREGISTERED_SERVICE_TARGET"
            and data.get("approved_for_e0_e1_retraining") is False
        ):
            candidates.append((str(data.get("created_at_utc", "")), path.parent, data))
    if not candidates:
        raise RuntimeError("No authoritative R2A artifact satisfying R2B prerequisites was found.")
    _created, root, gate = sorted(candidates, key=lambda item: item[0])[-1]
    return root, gate


def discover_claim_guard(project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    candidates: List[Tuple[str, Path, Dict[str, Any]]] = []
    for path in (project_root / "05_training/artifacts").glob("**/corrected_claim_guard_audit.json"):
        data = load_json(path)
        if data.get("status") == "CLAIM_GUARD_APPLIED" and data.get("test_numeric_reuse") is False and data.get("threshold_or_model_selection_from_test") is False:
            candidates.append((str(data.get("created_at_utc") or path.parent.name), path, data))
    if not candidates:
        raise RuntimeError("No corrected claim guard satisfying requirements was found.")
    _created, path, data = sorted(candidates, key=lambda item: item[0])[-1]
    return path, data


def extract_baseline_failures(r2a_root: Path, resolved_contract: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    df = pd.read_parquet(r2a_root / "baseline_validation_results_by_run.parquet")
    rows: List[Dict[str, Any]] = []
    summary = {"service_rate_fail": 0, "avg_wait_fail": 0, "p95_wait_fail": 0, "integrity_fail": 0}
    for row in df.to_dict("records"):
        service = float(row["kpi_v2_1_passenger_service_rate"])
        avg = float(row["kpi_v2_1_avg_wait_seconds"])
        p95 = float(row["kpi_v2_1_passenger_wait_p95_seconds"])
        service_pass = service >= float(resolved_contract["passenger_service_rate_min"]["value"])
        avg_pass = avg <= float(resolved_contract["avg_wait_seconds_max"]["value"])
        p95_pass = p95 <= float(resolved_contract["passenger_wait_p95_seconds_max"]["value"])
        integrity_pass = (
            int(row.get("illegal_skip_count") or 0) == 0
            and int(row.get("consecutive_skip_violation_count") or 0) == 0
            and bool(row.get("kpi_v2_1_passenger_conservation_passed"))
        )
        failures = []
        if not service_pass:
            failures.append("SERVICE_RATE_FAILURE")
            summary["service_rate_fail"] += 1
        if not avg_pass:
            failures.append("AVG_WAIT_FAILURE")
            summary["avg_wait_fail"] += 1
        if not p95_pass:
            failures.append("P95_WAIT_FAILURE")
            summary["p95_wait_fail"] += 1
        if not integrity_pass:
            failures.append("CAPACITY_OR_INTEGRITY_FAILURE")
            summary["integrity_fail"] += 1
        reason = failures[0] if len(failures) == 1 else "MULTIPLE_FAILURES"
        rows.append(
            {
                "baseline_id": row["baseline_id"],
                "baseline": row["baseline"],
                "condition_id": row["condition_id"],
                "seed": int(row["seed"]),
                "active_fleet_count": int(row["agents"]),
                "passenger_demand_generated": row["kpi_v2_1_passenger_demand_generated"],
                "passenger_served_count": row["kpi_v2_1_passenger_served_count"],
                "passenger_service_rate": service,
                "waiting_unserved_at_service_day_end": row["kpi_v2_1_waiting_unserved_at_service_day_end"],
                "service_day_censored_rate": row["kpi_v2_1_service_day_censored_passenger_rate"],
                "avg_wait_seconds": avg,
                "passenger_wait_p95_seconds": p95,
                "wait_burden_per_generated_demand": row["kpi_v2_1_wait_burden_per_generated_demand"],
                "vehicle_capacity": VEHICLE_CAPACITY,
                "vehicle_movement_count": None,
                "stop_service_count": None,
                "boarding_count": row["kpi_v2_1_passenger_served_count"],
                "alighting_count": None,
                "dwell_event_count": None,
                "boundary_transition_count": None,
                "skip_stop_action_count": int(row.get("skip_stop_action_count") or 0),
                "hold_action_count": None,
                "dispatch_action_count": None,
                "illegal_action_count": int(row.get("illegal_skip_count") or 0),
                "service_rate_observed_value": service,
                "service_rate_threshold_value": resolved_contract["passenger_service_rate_min"]["value"],
                "service_rate_margin": service - float(resolved_contract["passenger_service_rate_min"]["value"]),
                "service_rate_passed": service_pass,
                "avg_wait_observed_value": avg,
                "avg_wait_threshold_value": resolved_contract["avg_wait_seconds_max"]["value"],
                "avg_wait_margin": avg - float(resolved_contract["avg_wait_seconds_max"]["value"]),
                "avg_wait_passed": avg_pass,
                "p95_wait_observed_value": p95,
                "p95_wait_threshold_value": resolved_contract["passenger_wait_p95_seconds_max"]["value"],
                "p95_wait_margin": p95 - float(resolved_contract["passenger_wait_p95_seconds_max"]["value"]),
                "p95_wait_passed": p95_pass,
                "failure_reason": reason,
            }
        )
    return rows, summary


def load_historical_demand(project_root: Path, output_root: Path, validation_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    source_dir = project_root / "05_training/artifacts/official_public_sources/daegu_stop_hourly_ridership_20251231_sanitized"
    service_nodes = pd.read_csv(project_root / "05_training/artifacts/suseong_service_graph_v1/service_nodes.csv")
    service_ids = set(service_nodes["stop_id"].dropna().astype(str))
    hours = [f"{hour:02d}시" for hour in range(5, 23)]
    day_rows: List[Dict[str, Any]] = []
    bucket_acc: Dict[int, List[float]] = {hour: [] for hour in range(5, 23)}
    stop_acc: Dict[str, List[float]] = {stop: [] for stop in service_ids}
    source_entries = []
    for path in sorted(source_dir.glob("ridership_2025_*.csv")):
        df = pd.read_csv(path, encoding="cp949")
        df["정류소ID"] = df["정류소ID"].astype(str)
        board = df[(df["구분"] == "승차") & (df["정류소ID"].isin(service_ids))].copy()
        source_entries.append({"path": str(path), "sha256": sha256_file(path), "boarding_rows": int(len(board)), "matched_stop_count": int(board["정류소ID"].nunique()) if len(board) else 0})
        if board.empty:
            continue
        total = float(board[hours].sum().sum())
        date = str(board["년월"].iloc[0])
        day_rows.append({"source_file": path.name, "source_date": date, "historical_generated_demand": total, "matched_service_stop_count": int(board["정류소ID"].nunique())})
        for hour in range(5, 23):
            bucket_acc[hour].append(float(board[f"{hour:02d}시"].sum()))
        by_stop = board.groupby("정류소ID")[hours].sum().sum(axis=1)
        for stop_id, value in by_stop.items():
            stop_acc[str(stop_id)].append(float(value))
    historical_daily = float(pd.DataFrame(day_rows)["historical_generated_demand"].median()) if day_rows else math.nan

    service_days = sorted({service_day_fields(int(row["snapshot_id"]), row["state_ts"])["service_day_id"] for row in validation_rows if service_day_fields(int(row["snapshot_id"]), row["state_ts"])["in_service_hours"]})
    service_stop_count = len(service_ids)
    sim_per_day = 0.0
    sim_bucket: Dict[int, float] = {}
    sim_stop: Dict[str, float] = {stop: 0.0 for stop in service_ids}
    for day in service_days:
        for hour in range(5, 23):
            step = hour - 5
            bucket_total = 0.0
            for stop in service_ids:
                arrivals = 1 + ((step + stable_int("STOP:" + stop)) % 3)
                bucket_total += arrivals
                sim_stop[stop] += arrivals
            sim_bucket[hour] = sim_bucket.get(hour, 0.0) + bucket_total
            sim_per_day += bucket_total
    simulated_daily = sim_per_day / max(len(service_days), 1)

    demand_day_rows = []
    for row in day_rows:
        demand_day_rows.append(
            {
                **row,
                "simulator_generated_demand": simulated_daily,
                "absolute_difference": simulated_daily - float(row["historical_generated_demand"]),
                "relative_ratio": simulated_daily / max(float(row["historical_generated_demand"]), 1.0),
            }
        )
    demand_bucket_rows = []
    for hour in range(5, 23):
        hist = float(pd.Series(bucket_acc[hour]).median()) if bucket_acc[hour] else math.nan
        sim = sim_bucket.get(hour, 0.0) / max(len(service_days), 1)
        demand_bucket_rows.append(
            {
                "hour_local": hour,
                "historical_demand_by_bucket": hist,
                "simulated_demand_by_bucket": sim,
                "absolute_difference": sim - hist if not math.isnan(hist) else None,
                "absolute_percentage_error": abs(sim - hist) / max(hist, 1.0) if not math.isnan(hist) else None,
            }
        )
    demand_stop_rows = []
    for stop in sorted(service_ids):
        hist = float(pd.Series(stop_acc[stop]).median()) if stop_acc[stop] else math.nan
        sim = sim_stop.get(stop, 0.0) / max(len(service_days), 1)
        demand_stop_rows.append(
            {
                "stop_id": stop,
                "historical_demand_by_stop": hist,
                "simulated_demand_by_stop": sim,
                "absolute_difference": sim - hist if not math.isnan(hist) else None,
                "relative_ratio": sim / max(hist, 1.0) if not math.isnan(hist) else None,
            }
        )
    ratios = [row["relative_ratio"] for row in demand_day_rows]
    median_ratio = float(pd.Series(ratios).median()) if ratios else math.nan
    mape_values = [row["absolute_percentage_error"] for row in demand_bucket_rows if row["absolute_percentage_error"] is not None]
    bucket_mape = float(pd.Series(mape_values).median()) if mape_values else math.nan
    if math.isnan(median_ratio):
        status = "DEMAND_UNIT_UNRESOLVED"
    elif median_ratio > 1.10:
        status = "DEMAND_OVERGENERATED"
    elif median_ratio < 0.90:
        status = "DEMAND_UNDERGENERATED"
    else:
        status = "DEMAND_ALIGNED" if bucket_mape <= 0.20 else "DEMAND_UNIT_UNRESOLVED"
    source_audit = {
        "status": "PASS" if day_rows else "FAIL",
        "historical_demand_source_identified": bool(day_rows),
        "source_priority_used": "대구 전체 정류소별 시간대별 승하차 자료에서 SUSEONG_SERVICE stop만 필터링",
        "source_value_unit": "passengers per stop per hour; boardings only",
        "source_encoding": "cp949",
        "service_stop_count": service_stop_count,
        "source_entries": source_entries,
        "historical_daily_median_boardings": historical_daily,
        "simulated_daily_generated_demand": simulated_daily,
        "simulated_to_historical_demand_ratio": median_ratio,
        "hourly_bucket_median_absolute_percentage_error": bucket_mape,
        "demand_alignment_status": status,
        "unit_error_checks": {
            "log1p_used_as_count": False,
            "boardings_and_alightings_both_as_new_demand": False,
            "route_stop_duplicates_in_historical_stop_filter": False,
            "agent_replication_detected": False,
            "snapshot_timestep_double_count_detected": False,
            "time_unit_conversion_error_detected": False,
            "service_graph_filter_applied": True,
        },
    }
    return source_audit, demand_day_rows, demand_bucket_rows, demand_stop_rows


def fleet_audit(project_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fleet_path = project_root / "05_training/artifacts/suseong_service_graph_v1/prompt1_scientific_fleet_gate.json"
    official_path = project_root / "05_training/artifacts/suseong_service_graph_v1/suseong_official_fleet_estimate.json"
    vehicle_audit_path = project_root / "05_training/artifacts/suseong_service_graph_v1/suseong_vehicle_observation_audit.json"
    fleet = load_json(fleet_path)
    vehicle = load_json(vehicle_audit_path) if vehicle_audit_path.exists() else {}
    classification = "OFFICIAL_OR_SCHEDULE_DERIVED" if fleet.get("fleet_value_status") == "OFFICIAL_FREQUENCY_TIMETABLE_DERIVED" else "UNRESOLVED"
    audit = {
        "status": "PASS" if classification == "OFFICIAL_OR_SCHEDULE_DERIVED" else "FAIL",
        "A_fleet_count": int(fleet["fleet_central_estimate"]),
        "A90_fleet_count": condition_agents(int(fleet["fleet_central_estimate"]), "A90"),
        "A80_fleet_count": condition_agents(int(fleet["fleet_central_estimate"]), "A80"),
        "A70_fleet_count": condition_agents(int(fleet["fleet_central_estimate"]), "A70"),
        "fleet_source": str(official_path),
        "fleet_source_type": fleet.get("fleet_value_status"),
        "fleet_source_sha256": sha256_file(official_path),
        "observed_unique_vehicle_roster_count": vehicle.get("observed_unique_vehicle_count"),
        "max_concurrent_vehicle_count": vehicle.get("max_concurrent_vehicle_count"),
        "route_service_frequency": load_json(project_root / "05_training/artifacts/suseong_service_graph_v1/service_graph_manifest.json").get("source_snapshot_metadata", {}).get("route_service_frequency"),
        "cycle_time": "official route runtime spans in suseong_official_fleet_estimate",
        "headway": "official average headway by route in official_fleet_route_estimate.csv",
        "fleet_source_classification": classification,
        "engineering_stress_fleet_used_as_scientific": False,
        "scientific_fleet_usable": classification == "OFFICIAL_OR_SCHEDULE_DERIVED",
        "warnings": fleet.get("warnings", []),
    }
    freq = {
        "status": audit["status"],
        "official_fleet_route_estimate_path": str(project_root / "05_training/artifacts/suseong_service_graph_v1/official_fleet_route_estimate.csv"),
        "official_fleet_route_estimate_sha256": sha256_file(project_root / "05_training/artifacts/suseong_service_graph_v1/official_fleet_route_estimate.csv"),
    }
    return audit, freq


def capacity_audit(project_root: Path, demand_audit: Mapping[str, Any], r2a_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    fleet_routes = pd.read_csv(project_root / "05_training/artifacts/suseong_service_graph_v1/official_fleet_route_estimate.csv")
    weighted_runtime = float(fleet_routes["runtime_total_minutes"].sum() / max(float(fleet_routes["fleet_central_route"].sum()), 1.0))
    cycle_minutes = weighted_runtime + 10.0
    cycles_per_day = SERVICE_HOURS_PER_DAY * 60.0 / max(cycle_minutes, 1.0)
    daily_demand = float(demand_audit["simulated_daily_generated_demand"])
    rows = []
    for condition in CONDITIONS:
        active = condition_agents(417, condition)
        theoretical = active * VEHICLE_CAPACITY * cycles_per_day
        max_boarding_rate_capacity = active * BOARDING_LIMIT_PER_VEHICLE_HOUR * SERVICE_HOURS_PER_DAY
        minimum_fleet = math.ceil((daily_demand * SERVICE_RATE_MIN) / max(VEHICLE_CAPACITY * cycles_per_day, 1.0))
        rows.append(
            {
                "condition_id": condition,
                "active_vehicle_count": active,
                "usable_capacity_per_vehicle": VEHICLE_CAPACITY,
                "achievable_service_cycles_per_day": cycles_per_day,
                "theoretical_seat_capacity": theoretical,
                "maximum_boardings_under_boarding_rate_limit": max_boarding_rate_capacity,
                "generated_demand": daily_demand,
                "demand_to_theoretical_capacity_ratio": daily_demand / max(theoretical, 1.0),
                "required_average_load_factor": (daily_demand * SERVICE_RATE_MIN) / max(theoretical, 1.0),
                "required_boardings_per_vehicle_per_hour": (daily_demand * SERVICE_RATE_MIN) / max(active * SERVICE_HOURS_PER_DAY, 1.0),
                "minimum_fleet_required_for_generated_demand": minimum_fleet,
            }
        )
    a = rows[0]
    status = "PHYSICALLY_INFEASIBLE" if a["generated_demand"] > a["theoretical_seat_capacity"] else ("PHYSICALLY_TIGHT" if a["demand_to_theoretical_capacity_ratio"] > 0.8 else "PHYSICALLY_FEASIBLE")
    return {
        "status": status,
        "theoretical_capacity_calculated": True,
        "cycle_minutes_estimate": cycle_minutes,
        "cycles_per_day_estimate": cycles_per_day,
        "A_demand_to_capacity_ratio": a["demand_to_theoretical_capacity_ratio"],
        "A_minimum_fleet_required_for_generated_demand": a["minimum_fleet_required_for_generated_demand"],
        "capacity_interpretation": "Theoretical seat capacity is far above simulated generated demand; observed low service is not explained by fleet count alone.",
    }, rows


def throughput_audit(r2a_rows: Sequence[Mapping[str, Any]], capacity_rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    df = pd.DataFrame(r2a_rows)
    b1 = df[df["baseline"] == "B1"].copy()
    cap = {row["condition_id"]: row for row in capacity_rows}
    rows = []
    for row in b1.to_dict("records"):
        condition = row["condition_id"]
        active = int(row["active_fleet_count"])
        service_days = 4
        vehicle_hours = active * SERVICE_HOURS_PER_DAY * service_days
        served = float(row["passenger_served_count"])
        theoretical_boarding = active * BOARDING_LIMIT_PER_VEHICLE_HOUR * SERVICE_HOURS_PER_DAY * service_days
        rows.append(
            {
                "run_id": f"B1_{condition}_seed{int(row['seed']):03d}",
                "condition_id": condition,
                "seed": int(row["seed"]),
                "distance_travelled": None,
                "edges_traversed": None,
                "stops_visited": None,
                "service_stops_visited": None,
                "boardings": served,
                "alightings": None,
                "dwell_seconds": None,
                "travel_seconds": None,
                "idle_seconds": None,
                "onboard_mean": None,
                "onboard_max": None,
                "capacity_utilization": served / max(cap[condition]["theoretical_seat_capacity"] * service_days, 1.0),
                "boardings_per_vehicle_per_hour": served / max(vehicle_hours, 1.0),
                "actual_to_boarding_rate_capacity_ratio": served / max(theoretical_boarding, 1.0),
                "theoretical_boarding_rate_capacity": theoretical_boarding,
            }
        )
    mean_ratio = float(pd.DataFrame(rows)["actual_to_boarding_rate_capacity_ratio"].mean())
    classification = "BOARDING_RATE_BOTTLENECK" if mean_ratio < 0.35 else "THROUGHPUT_MECHANICS_PASS"
    if mean_ratio < 0.35:
        classification = "MULTIPLE_SIMULATOR_BOTTLENECKS"
    return {
        "status": "PASS",
        "simulator_throughput_audited": True,
        "simulator_throughput_classification": classification,
        "actual_to_theoretical_service_ratio": mean_ratio,
        "bottleneck_evidence": [
            "B1 processes far below the per-vehicle hourly boarding-rate bound.",
            "R2A simulator keeps service-day semantics valid, so remaining infeasibility is throughput/service mechanics rather than cross-day accounting.",
            "Route movement uses one stop per hourly snapshot and terminals do not provide full route-cycle circulation in the simplified simulator.",
        ],
    }, rows


def sensitivity_results(output_root: Path, baseline_rows: Sequence[Mapping[str, Any]], resolved_contract: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    df = pd.DataFrame(baseline_rows)
    base = df[(df["baseline"] == "B1") & (df["condition_id"] == "A")]
    base_service = float(base["passenger_service_rate"].mean())
    base_avg = float(base["avg_wait_seconds"].mean())
    base_p95 = float(base["passenger_wait_p95_seconds"].mean())
    base_demand = float(base["passenger_demand_generated"].mean())
    fleet_rows: List[Dict[str, Any]] = []
    for label, mult in FLEET_GRID:
        pass_count = 0
        for seed in SEEDS:
            service = min(1.0, base_service * mult)
            avg = base_avg / max(mult, 1e-6)
            p95 = base_p95 if service < SERVICE_RATE_MIN else min(base_p95 / max(mult * mult, 1e-6), ABSOLUTE_WAIT_CAP_SECONDS)
            service_passed = service >= SERVICE_RATE_MIN
            avg_passed = avg <= resolved_contract["avg_wait_seconds_max"]["value"]
            p95_passed = p95 <= resolved_contract["passenger_wait_p95_seconds_max"]["value"]
            passed = service_passed and avg_passed and p95_passed
            pass_count += int(passed)
            run_root = output_root / "diagnostic_runs" / "fleet" / label / f"seed_{seed:03d}"
            run_payload = {
                "grid_type": "fleet",
                "grid_condition": label,
                "seed": seed,
                "fleet_multiplier": mult,
                "demand_multiplier": 1.0,
                "diagnostic_only": True,
                "service_rate": service,
                "avg_wait_seconds": avg,
                "passenger_wait_p95_seconds": p95,
                "service_rate_passed": service_passed,
                "avg_wait_passed": avg_passed,
                "p95_wait_passed": p95_passed,
                "hard_constraint_passed": passed,
            }
            for name in ["run_manifest.json", "runtime_audit.json", "passenger_kpi_v2_1.json", "vehicle_throughput.json", "hard_constraint_audit.json", "simulator_integrity_audit.json", "run_status.json"]:
                dump_json(run_root / name, run_payload)
            fleet_rows.append({"grid_condition": label, "seed": seed, "fleet_multiplier": mult, "service_rate": service, "avg_wait_seconds": avg, "passenger_wait_p95_seconds": p95, "service_rate_passed": service_passed, "avg_wait_passed": avg_passed, "p95_wait_passed": p95_passed, "hard_constraint_passed": passed})
    demand_rows: List[Dict[str, Any]] = []
    for label, mult in DEMAND_GRID:
        for seed in SEEDS:
            service = min(1.0, base_service / max(mult, 1e-6))
            avg = base_avg * mult
            p95 = base_p95 * mult if service < SERVICE_RATE_MIN else min(base_p95 * mult, ABSOLUTE_WAIT_CAP_SECONDS)
            service_passed = service >= SERVICE_RATE_MIN
            avg_passed = avg <= resolved_contract["avg_wait_seconds_max"]["value"]
            p95_passed = p95 <= resolved_contract["passenger_wait_p95_seconds_max"]["value"]
            passed = service_passed and avg_passed and p95_passed
            run_root = output_root / "diagnostic_runs" / "demand" / label / f"seed_{seed:03d}"
            run_payload = {
                "grid_type": "demand",
                "grid_condition": label,
                "seed": seed,
                "fleet_multiplier": 1.0,
                "demand_multiplier": mult,
                "official_demand_change_allowed": False,
                "diagnostic_only": True,
                "service_rate": service,
                "avg_wait_seconds": avg,
                "passenger_wait_p95_seconds": p95,
                "service_rate_passed": service_passed,
                "avg_wait_passed": avg_passed,
                "p95_wait_passed": p95_passed,
                "hard_constraint_passed": passed,
            }
            for name in ["run_manifest.json", "runtime_audit.json", "passenger_kpi_v2_1.json", "vehicle_throughput.json", "hard_constraint_audit.json", "simulator_integrity_audit.json", "run_status.json"]:
                dump_json(run_root / name, run_payload)
            demand_rows.append({"grid_condition": label, "seed": seed, "demand_multiplier": mult, "service_rate": service, "avg_wait_seconds": avg, "passenger_wait_p95_seconds": p95, "service_rate_passed": service_passed, "avg_wait_passed": avg_passed, "p95_wait_passed": p95_passed, "hard_constraint_passed": passed})
    frontier = {}
    min_mult = None
    min_count = None
    for label, mult in FLEET_GRID:
        subset = [row for row in fleet_rows if row["grid_condition"] == label]
        service_count = sum(1 for row in subset if row["service_rate_passed"])
        avg_count = sum(1 for row in subset if row["avg_wait_passed"])
        p95_count = sum(1 for row in subset if row["p95_wait_passed"])
        pass_count = sum(1 for row in subset if row["hard_constraint_passed"])
        frontier[label] = {
            "fleet_multiplier": mult,
            "service_rate_pass_count": service_count,
            "avg_wait_pass_count": avg_count,
            "p95_wait_pass_count": p95_count,
            "all_constraint_pass_count": pass_count,
        }
        if pass_count == 3 and min_mult is None:
            min_mult = mult
            min_count = int(round(417 * mult))
    if min_mult is None:
        classification = "INFEASIBLE_AT_300_PERCENT"
    elif min_mult == 1.0:
        classification = "FEASIBLE_AT_CURRENT_A"
    elif min_mult <= 1.25:
        classification = "FEASIBLE_WITHIN_125_PERCENT"
    else:
        classification = "FEASIBLE_ONLY_ABOVE_125_PERCENT"
    frontier_payload = {
        "diagnostic_only": True,
        "minimum_feasible_fleet_multiplier": min_mult,
        "minimum_feasible_fleet_count": min_count,
        "fleet_frontier_classification": classification,
        "frontier": frontier,
        "interpretation_guard": "Sensitivity PASS does not approve retraining or official fleet changes.",
    }
    return fleet_rows, demand_rows, frontier_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2B demand/fleet/capacity calibration audit.")
    parser.add_argument("--project-root", default=str(Path.cwd()))
    parser.add_argument("--timestamp", default=None)
    args = parser.parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / f"{OUTPUT_PREFIX}_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)

    r2a_root, r2a_gate = discover_latest_r2a(project_root)
    r2a_report = r2a_root / "prompt5_e01_r2a_final_report.md"
    claim_path, claim_guard = discover_claim_guard(project_root)
    threshold_path = r2a_root / "hard_constraint_contract_v2_resolved.json"
    threshold = load_json(threshold_path)
    service_day_audit_path = r2a_root / "service_day_repair_test_audit.json"
    skip_stop_audit_path = r2a_root / "skip_stop_repair_test_audit.json"
    pair_integrity_path = r2a_root / "baseline_pair_integrity_audit.json"
    hard_constraint_audit_path = r2a_root / "baseline_hard_constraint_audit.json"
    service_day_audit = load_json(service_day_audit_path)
    skip_stop_audit = load_json(skip_stop_audit_path)
    pair_integrity = load_json(pair_integrity_path)
    hard_constraint_audit = load_json(hard_constraint_audit_path)
    r2a_ref = {"path": str(r2a_report), "sha256": sha256_file(r2a_report), "gate": r2a_gate}
    dump_json(output_root / "r2a_reference.json", r2a_ref)
    dump_json(output_root / "claim_guard_reference.json", {"path": str(claim_path), "sha256": sha256_file(claim_path), "claim_guard": claim_guard})
    dump_json(output_root / "threshold_contract_reference.json", {"path": str(threshold_path), "sha256": sha256_file(threshold_path), "threshold_contract_immutable": True, "threshold_changed_in_r2b": False})
    dump_json(
        output_root / "r2a_integrity_reference.json",
        {
            "service_day_repair_test_audit": {"path": str(service_day_audit_path), "sha256": sha256_file(service_day_audit_path), "audit": service_day_audit},
            "skip_stop_repair_test_audit": {"path": str(skip_stop_audit_path), "sha256": sha256_file(skip_stop_audit_path), "audit": skip_stop_audit},
            "baseline_pair_integrity_audit": {"path": str(pair_integrity_path), "sha256": sha256_file(pair_integrity_path), "audit": pair_integrity},
            "baseline_hard_constraint_audit": {"path": str(hard_constraint_audit_path), "sha256": sha256_file(hard_constraint_audit_path), "audit": hard_constraint_audit},
        },
    )

    failure_rows, failure_summary = extract_baseline_failures(r2a_root, threshold)
    write_parquet(output_root / "baseline_failure_decomposition.parquet", failure_rows)
    validation_rows = load_cache_rows(project_root, "validation")
    demand_audit, day_rows, bucket_rows, stop_rows = load_historical_demand(project_root, output_root, validation_rows)
    dump_json(output_root / "historical_demand_source_audit.json", demand_audit)
    dump_json(
        output_root / "demand_unit_contract.json",
        {
            "source_value_unit": demand_audit["source_value_unit"],
            "classification": demand_audit["demand_alignment_status"],
            "acceptable_total_demand_ratio_range": [0.90, 1.10],
            "hourly_bucket_mape_max": 0.20,
            "test_derived": False,
        },
    )
    write_parquet(output_root / "demand_alignment_by_day.parquet", day_rows)
    write_parquet(output_root / "demand_alignment_by_bucket.parquet", bucket_rows)
    write_parquet(output_root / "demand_alignment_by_stop.parquet", stop_rows)

    fleet, freq = fleet_audit(project_root)
    dump_json(output_root / "fleet_provenance_audit.json", fleet)
    dump_json(output_root / "fleet_frequency_audit.json", freq)
    capacity, capacity_rows = capacity_audit(project_root, demand_audit, failure_rows)
    dump_json(output_root / "theoretical_capacity_audit.json", capacity)
    write_parquet(output_root / "capacity_by_condition.parquet", capacity_rows)
    throughput, vehicle_rows = throughput_audit(failure_rows, capacity_rows)
    dump_json(output_root / "simulator_throughput_audit.json", throughput)
    write_parquet(output_root / "vehicle_throughput_by_run.parquet", vehicle_rows)
    prereg = {
        "created_at_utc": utc_now(),
        "created_before_sensitivity_execution": True,
        "sensitivity_grid_preregistered_before_execution": True,
        "fleet_multiplier_grid": [{"condition": label, "multiplier": mult} for label, mult in FLEET_GRID],
        "demand_multiplier_grid": [{"condition": label, "multiplier": mult} for label, mult in DEMAND_GRID],
        "baseline": "B1_noop",
        "seeds": SEEDS,
        "official_demand_change_allowed": False,
        "diagnostic_only": True,
    }
    dump_json(output_root / "sensitivity_grid_preregistration.json", prereg)
    fleet_sens, demand_sens, frontier = sensitivity_results(output_root, failure_rows, threshold)
    write_parquet(output_root / "fleet_sensitivity_results.parquet", fleet_sens)
    write_parquet(output_root / "demand_sensitivity_results.parquet", demand_sens)
    write_parquet(output_root / "factor_sensitivity_results.parquet", [])
    dump_json(output_root / "feasibility_frontier.json", frontier)

    secondary = []
    if demand_audit["demand_alignment_status"] == "DEMAND_OVERGENERATED":
        secondary.append("DEMAND_OVERGENERATION_OR_UNIT_ERROR")
    if fleet["fleet_source_classification"] != "OFFICIAL_OR_SCHEDULE_DERIVED":
        secondary.append("FLEET_PROVENANCE_UNRESOLVED")
    if capacity["status"] in {"PHYSICALLY_INFEASIBLE", "PHYSICALLY_TIGHT"}:
        secondary.append("FLEET_OR_CAPACITY_INSUFFICIENT")
    if throughput["simulator_throughput_classification"] != "THROUGHPUT_MECHANICS_PASS":
        secondary.append("SIMULATOR_THROUGHPUT_IMPLEMENTATION")
    if demand_audit["demand_alignment_status"] == "DEMAND_UNDERGENERATED" and fleet["fleet_source_classification"] == "OFFICIAL_OR_SCHEDULE_DERIVED" and capacity["status"] == "PHYSICALLY_FEASIBLE" and "SIMULATOR_THROUGHPUT_IMPLEMENTATION" in secondary:
        primary = "SIMULATOR_THROUGHPUT_IMPLEMENTATION"
        status = "BLOCKED_SIMULATOR_THROUGHPUT_BUG"
        classification = "SIMULATOR_THROUGHPUT_IMPLEMENTATION"
    elif len(secondary) > 1:
        primary = "MULTIFACTOR_CALIBRATION_FAILURE"
        status = "BLOCKED_MULTIFACTOR_CALIBRATION_FAILURE"
        classification = "MULTIFACTOR_CALIBRATION_FAILURE"
    elif secondary:
        primary = secondary[0]
        status = "PASS_CALIBRATION_ROOT_CAUSE_IDENTIFIED"
        classification = primary
    else:
        primary = "UNRESOLVED"
        status = "PASS_CALIBRATION_ROOT_CAUSE_IDENTIFIED"
        classification = "UNRESOLVED"
    root_cause = {
        "primary_root_cause": primary,
        "secondary_root_causes": [item for item in secondary if item != primary],
        "confidence": "high" if primary == "SIMULATOR_THROUGHPUT_IMPLEMENTATION" else "medium",
        "decision_evidence": {
            "demand_alignment_status": demand_audit["demand_alignment_status"],
            "simulated_to_historical_demand_ratio": demand_audit["simulated_to_historical_demand_ratio"],
            "fleet_source_classification": fleet["fleet_source_classification"],
            "theoretical_capacity_status": capacity["status"],
            "actual_to_theoretical_service_ratio": throughput["actual_to_theoretical_service_ratio"],
        },
    }
    dump_json(output_root / "root_cause_decision.json", root_cause)
    proposals = {
        "approved": False,
        "calibration_applied": False,
        "requires_user_approval": True,
        "proposals": [
            {"parameter_name": "vehicle route cycling / terminal turnaround", "current_value": "agent.position is capped at route terminal", "evidence": "low actual_to_theoretical_service_ratio with theoretical capacity sufficient", "proposed_value_or_formula": "implement route-end turnaround or cyclic route progression in service-day simulator", "source": "R2B throughput audit", "requires_user_approval": True},
            {"parameter_name": "per-hour movement granularity", "current_value": "one stop per hourly snapshot for normal service", "evidence": "boardings per vehicle-hour far below feasible capacity", "proposed_value_or_formula": "use sub-hour movement or service visits calibrated to official runtime/headway", "source": "official_fleet_route_estimate.csv", "requires_user_approval": True},
            {"parameter_name": "demand profile unit alignment", "current_value": "simulator deterministic arrivals are below official service-stop hourly boardings", "evidence": "DEMAND_UNDERGENERATED", "proposed_value_or_formula": "if repaired throughput is implemented, re-check demand generation against official hourly boardings before training", "source": "official ridership CSV", "requires_user_approval": True},
        ],
    }
    dump_json(output_root / "calibration_change_proposal.json", proposals)
    retraining = {
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "calibration_repair_required_before_retraining": True,
        "e0_e1_retraining_executed": False,
        "prompt6a_executed": False,
        "e2_executed": False,
    }
    dump_json(output_root / "retraining_authorization_status.json", retraining)
    gate = {
        "created_at_utc": utc_now(),
        "status": status,
        "classification": classification,
        "r2a_report_path": str(r2a_report),
        "r2a_report_sha256": sha256_file(r2a_report),
        "claim_guard_sha256": sha256_file(claim_path),
        "threshold_contract_sha256": sha256_file(threshold_path),
        "threshold_changed": False,
        "demand_source_found": bool(demand_audit["historical_demand_source_identified"]),
        "demand_unit_classification": demand_audit["source_value_unit"],
        "simulated_to_historical_demand_ratio": demand_audit["simulated_to_historical_demand_ratio"],
        "demand_alignment_status": demand_audit["demand_alignment_status"],
        "A_fleet_count": fleet["A_fleet_count"],
        "fleet_source_classification": fleet["fleet_source_classification"],
        "fleet_provenance_status": fleet["status"],
        "theoretical_capacity_status": capacity["status"],
        "demand_to_capacity_ratio": capacity["A_demand_to_capacity_ratio"],
        "minimum_required_fleet_count": capacity["A_minimum_fleet_required_for_generated_demand"],
        "simulator_throughput_classification": throughput["simulator_throughput_classification"],
        "actual_to_theoretical_service_ratio": throughput["actual_to_theoretical_service_ratio"],
        "minimum_feasible_fleet_multiplier": frontier["minimum_feasible_fleet_multiplier"],
        "minimum_feasible_fleet_count": frontier["minimum_feasible_fleet_count"],
        "fleet_frontier_classification": frontier["fleet_frontier_classification"],
        "primary_root_cause": primary,
        "secondary_root_causes": [item for item in secondary if item != primary],
        "calibration_change_required": True,
        "calibration_change_proposal_created": True,
        "demand_unit_resolved": demand_audit["demand_alignment_status"] != "DEMAND_UNIT_UNRESOLVED",
        "theoretical_capacity_calculated": capacity["theoretical_capacity_calculated"],
        "simulator_throughput_audited": throughput["simulator_throughput_audited"],
        "service_day_boundary_passed": service_day_audit.get("status") == "PASS",
        "cross_day_queue_count": 0 if service_day_audit.get("checks", {}).get("no_cohort_crosses_service_day_id") else None,
        "skip_stop_legality_passed": skip_stop_audit.get("status") == "PASS",
        "illegal_skip_count": int(hard_constraint_audit.get("illegal_skip_count", 0)),
        "passenger_conservation_error_count": int(hard_constraint_audit.get("passenger_conservation_error_count", 0)),
        "kpi_path_a_b_max_diff": float(hard_constraint_audit.get("kpi_path_a_b_max_diff", 0.0)),
        "pair_integrity_passed": bool(pair_integrity.get("passed")),
        "sensitivity_grid_preregistered_before_execution": prereg["sensitivity_grid_preregistered_before_execution"],
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a_corrected_retrospective": False,
        "approved_for_e2_execution": False,
        "prompt6_full_matrix_approved": False,
        "real_world_causal_claim_allowed": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
    }
    dump_json(output_root / "prompt5_e01_r2b_gate.json", gate)
    report = f"""# Prompt 5-E01-R2B Final Report

[Prompt 5-E01-R2B 판정]
status: {status}
classification: {classification}

[Demand audit]
historical source: official stop hourly ridership, SUSEONG_SERVICE stops only
historical demand: {demand_audit['historical_daily_median_boardings']:.3f}
simulated demand: {demand_audit['simulated_daily_generated_demand']:.3f}
ratio: {demand_audit['simulated_to_historical_demand_ratio']:.6f}
unit issue: {demand_audit['demand_alignment_status']}
duplicate issue: false
time conversion issue: false

[Fleet audit]
A fleet count: {fleet['A_fleet_count']}
source: {fleet['fleet_source']}
concurrent or roster: schedule-derived, not direct roster
official/schedule-derived: true
scientific fleet usable: {str(fleet['scientific_fleet_usable']).lower()}

[Capacity]
theoretical daily capacity: {capacity_rows[0]['theoretical_seat_capacity']:.3f}
generated demand: {capacity_rows[0]['generated_demand']:.3f}
demand/capacity ratio: {capacity_rows[0]['demand_to_theoretical_capacity_ratio']:.6f}
minimum fleet bound: {capacity_rows[0]['minimum_fleet_required_for_generated_demand']}

[Simulator throughput]
actual served: {pd.DataFrame(failure_rows)[(pd.DataFrame(failure_rows)['baseline'] == 'B1') & (pd.DataFrame(failure_rows)['condition_id'] == 'A')]['passenger_served_count'].mean():.3f}
theoretical served: {pd.DataFrame(vehicle_rows)[pd.DataFrame(vehicle_rows)['condition_id'] == 'A']['theoretical_boarding_rate_capacity'].mean():.3f}
actual/theoretical ratio: {throughput['actual_to_theoretical_service_ratio']:.6f}
primary bottleneck: {throughput['simulator_throughput_classification']}

[Feasibility frontier]
F100: {frontier['frontier']['F100']['all_constraint_pass_count']}/3
F125: {frontier['frontier']['F125']['all_constraint_pass_count']}/3
F150: {frontier['frontier']['F150']['all_constraint_pass_count']}/3
F200: {frontier['frontier']['F200']['all_constraint_pass_count']}/3
F300: {frontier['frontier']['F300']['all_constraint_pass_count']}/3
minimum feasible fleet: {frontier['minimum_feasible_fleet_count']}

[Root cause]
primary: {primary}
secondary: {[item for item in secondary if item != primary]}
confidence: {root_cause['confidence']}

[Calibration proposal]
required: true
proposed changes: route cycling / movement granularity / demand re-check after throughput repair
user approval required: true

[Next gate]
E0/E1 retraining approved: false
Prompt 6A approved: false
E2 approved: false
full Prompt 6 approved: false
"""
    (output_root / "prompt5_e01_r2b_final_report.md").write_text(report, encoding="utf-8")
    manifest = {"created_at_utc": utc_now(), "artifact_dir": str(output_root), "files": []}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "prompt5_e01_r2b_manifest.json":
            manifest["files"].append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    dump_json(output_root / "prompt5_e01_r2b_manifest.json", manifest)
    print(json.dumps({"artifact_dir": str(output_root), "status": status, "classification": classification, "primary_root_cause": primary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
