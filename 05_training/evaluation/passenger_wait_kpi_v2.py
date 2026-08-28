from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


REQUIRED_LEDGER_FIELDS = [
    "cohort_id",
    "snapshot_id",
    "condition_id",
    "seed",
    "request_step",
    "request_time_seconds",
    "passenger_count",
    "boarding_step",
    "boarding_time_seconds",
    "service_completed",
    "remaining_unserved_at_horizon",
    "censored_at_horizon",
    "wait_seconds",
]


def stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def make_cohort_id(*, split: str, snapshot_id: int, condition_id: str, seed: int, node_uid: str, request_step: int, ordinal: int) -> str:
    payload = {
        "split": split,
        "snapshot_id": int(snapshot_id),
        "condition_id": str(condition_id),
        "seed": int(seed),
        "node_uid": str(node_uid),
        "request_step": int(request_step),
        "ordinal": int(ordinal),
    }
    return "cohort_" + stable_hash(payload)[:20]


def weighted_percentile(values: Sequence[Tuple[float, float]], percentile: float) -> Optional[float]:
    if not values:
        return None
    if percentile < 0 or percentile > 100:
        raise ValueError("percentile must be between 0 and 100")
    rows = sorted((float(v), float(w)) for v, w in values if float(w) > 0.0)
    if not rows:
        return None
    total = sum(w for _, w in rows)
    threshold = total * (percentile / 100.0)
    running = 0.0
    for value, weight in rows:
        running += weight
        if running >= threshold:
            return value
    return rows[-1][0]


def validate_ledger_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    missing_key_count = 0
    duplicate_key_count = 0
    seen = set()
    invalid_passenger_count = 0
    invalid_wait_count = 0
    for row in rows:
        missing = [key for key in REQUIRED_LEDGER_FIELDS if key not in row]
        if missing:
            missing_key_count += 1
            continue
        key = (row["cohort_id"], row["condition_id"], int(row["seed"]))
        if key in seen:
            duplicate_key_count += 1
        seen.add(key)
        if float(row["passenger_count"]) < 0:
            invalid_passenger_count += 1
        wait_seconds = row.get("wait_seconds")
        if wait_seconds is not None and float(wait_seconds) < 0:
            invalid_wait_count += 1
    return {
        "row_count": len(rows),
        "missing_key_count": missing_key_count,
        "duplicate_key_count": duplicate_key_count,
        "invalid_passenger_count": invalid_passenger_count,
        "invalid_wait_count": invalid_wait_count,
        "passed": missing_key_count == 0 and duplicate_key_count == 0 and invalid_passenger_count == 0 and invalid_wait_count == 0,
    }

def compute_passenger_wait_kpi_v2(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    validation = validate_ledger_rows(rows)
    if not validation["passed"]:
        raise ValueError(f"ledger validation failed: {validation}")

    generated = sum(float(row["passenger_count"]) for row in rows)
    boarded_rows = [row for row in rows if bool(row["service_completed"])]
    censored_rows = [row for row in rows if bool(row["censored_at_horizon"])]
    served = sum(float(row["passenger_count"]) for row in boarded_rows)
    waiting_at_horizon = sum(float(row["passenger_count"]) for row in censored_rows)
    explicitly_cancelled = 0.0

    zero_boarding_flag = served == 0.0
    zero_demand_flag = generated == 0.0
    avg_wait_seconds = None
    if not zero_boarding_flag:
        avg_wait_seconds = sum(float(row["passenger_count"]) * float(row["wait_seconds"]) for row in boarded_rows) / served

    service_rate = None if zero_demand_flag else served / generated
    p95_values = [(float(row["wait_seconds"]), float(row["passenger_count"])) for row in rows]
    p95 = weighted_percentile(p95_values, 95.0)
    wait_burden = None if zero_demand_flag else sum(float(row["passenger_count"]) * float(row["wait_seconds"]) for row in rows) / generated

    conservation_error = generated - served - waiting_at_horizon - explicitly_cancelled
    invalid_kpi_count = 0
    if service_rate is not None and not (0.0 <= service_rate <= 1.0):
        invalid_kpi_count += 1
    if avg_wait_seconds is not None and avg_wait_seconds < 0.0:
        invalid_kpi_count += 1
    if p95 is not None and p95 < 0.0:
        invalid_kpi_count += 1
    if p95 is not None and avg_wait_seconds is not None and p95 < avg_wait_seconds:
        invalid_kpi_count += 1

    return {
        "passenger_demand_generated": generated,
        "passenger_served_count": served,
        "waiting_at_horizon": waiting_at_horizon,
        "explicitly_cancelled": explicitly_cancelled,
        "passenger_service_rate": service_rate,
        "avg_wait_seconds": avg_wait_seconds,
        "zero_boarding_flag": zero_boarding_flag,
        "zero_demand_flag": zero_demand_flag,
        "passenger_wait_p95_seconds": p95,
        "p95_contains_censored_passengers": bool(censored_rows),
        "censored_passenger_count": waiting_at_horizon,
        "censored_passenger_rate": None if zero_demand_flag else waiting_at_horizon / generated,
        "wait_burden_per_generated_demand": wait_burden,
        "passenger_conservation_error": conservation_error,
        "passenger_conservation_passed": abs(conservation_error) == 0.0,
        "invalid_kpi_count": invalid_kpi_count,
        "ledger_validation": validation,
    }


def compute_passenger_wait_kpi_v2_reference(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    required = set(REQUIRED_LEDGER_FIELDS)
    keys = set()
    for row in rows:
        if not required.issubset(row.keys()):
            raise ValueError("missing ledger keys")
        key = (row["cohort_id"], row["condition_id"], int(row["seed"]))
        if key in keys:
            raise ValueError("duplicate ledger key")
        keys.add(key)

    generated = 0.0
    served = 0.0
    waiting = 0.0
    boarded_wait_sum = 0.0
    all_wait_values: List[Tuple[float, float]] = []
    all_wait_sum = 0.0
    censored = 0.0
    for row in rows:
        count = float(row["passenger_count"])
        wait = float(row["wait_seconds"])
        generated += count
        all_wait_sum += count * wait
        all_wait_values.append((wait, count))
        if bool(row["service_completed"]):
            served += count
            boarded_wait_sum += count * wait
        if bool(row["censored_at_horizon"]):
            waiting += count
            censored += count

    avg = None if served == 0.0 else boarded_wait_sum / served
    rate = None if generated == 0.0 else served / generated
    p95 = weighted_percentile(all_wait_values, 95.0)
    burden = None if generated == 0.0 else all_wait_sum / generated
    conservation = generated - served - waiting
    invalid = 0
    if rate is not None and not (0.0 <= rate <= 1.0):
        invalid += 1
    if avg is not None and avg < 0.0:
        invalid += 1
    if p95 is not None and p95 < 0.0:
        invalid += 1
    if p95 is not None and avg is not None and p95 < avg:
        invalid += 1
    return {
        "passenger_demand_generated": generated,
        "passenger_served_count": served,
        "waiting_at_horizon": waiting,
        "explicitly_cancelled": 0.0,
        "passenger_service_rate": rate,
        "avg_wait_seconds": avg,
        "zero_boarding_flag": served == 0.0,
        "zero_demand_flag": generated == 0.0,
        "passenger_wait_p95_seconds": p95,
        "p95_contains_censored_passengers": censored > 0.0,
        "censored_passenger_count": censored,
        "censored_passenger_rate": None if generated == 0.0 else censored / generated,
        "wait_burden_per_generated_demand": burden,
        "passenger_conservation_error": conservation,
        "passenger_conservation_passed": abs(conservation) == 0.0,
        "invalid_kpi_count": invalid,
    }


def compare_kpi_paths(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    a = compute_passenger_wait_kpi_v2(rows)
    b = compute_passenger_wait_kpi_v2_reference(rows)
    keys = [
        "passenger_demand_generated",
        "passenger_served_count",
        "passenger_service_rate",
        "avg_wait_seconds",
        "passenger_wait_p95_seconds",
        "wait_burden_per_generated_demand",
    ]
    diffs: Dict[str, Optional[float]] = {}
    max_abs_diff = 0.0
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            diffs[key] = None if av is None and bv is None else float("inf")
        else:
            diffs[key] = abs(float(av) - float(bv))
            max_abs_diff = max(max_abs_diff, diffs[key] or 0.0)
    return {
        "path_a": a,
        "path_b": b,
        "diffs": diffs,
        "max_abs_diff": max_abs_diff,
        "passed": max_abs_diff == 0.0 and all(value not in (float("inf"),) for value in diffs.values()),
    }
