from __future__ import annotations

import random

import pytest

from passenger_wait_kpi_v2 import compare_kpi_paths, compute_passenger_wait_kpi_v2, validate_ledger_rows


def _rows():
    return [
        {
            "cohort_id": "a",
            "snapshot_id": 1,
            "condition_id": "A",
            "seed": 1,
            "request_step": 0,
            "request_time_seconds": 0.0,
            "passenger_count": 2.0,
            "boarding_step": 1,
            "boarding_time_seconds": 60.0,
            "service_completed": True,
            "remaining_unserved_at_horizon": False,
            "censored_at_horizon": False,
            "wait_seconds": 60.0,
        },
        {
            "cohort_id": "b",
            "snapshot_id": 1,
            "condition_id": "A",
            "seed": 1,
            "request_step": 1,
            "request_time_seconds": 60.0,
            "passenger_count": 1.0,
            "boarding_step": 3,
            "boarding_time_seconds": 180.0,
            "service_completed": True,
            "remaining_unserved_at_horizon": False,
            "censored_at_horizon": False,
            "wait_seconds": 120.0,
        },
        {
            "cohort_id": "c",
            "snapshot_id": 2,
            "condition_id": "A",
            "seed": 1,
            "request_step": 2,
            "request_time_seconds": 120.0,
            "passenger_count": 1.0,
            "boarding_step": None,
            "boarding_time_seconds": None,
            "service_completed": False,
            "remaining_unserved_at_horizon": True,
            "censored_at_horizon": True,
            "wait_seconds": 180.0,
        },
    ]


def test_wait_kpi_v2_weighted_metrics_and_censoring() -> None:
    kpi = compute_passenger_wait_kpi_v2(_rows())
    assert kpi["passenger_demand_generated"] == 4.0
    assert kpi["passenger_served_count"] == 3.0
    assert kpi["passenger_service_rate"] == 0.75
    assert kpi["avg_wait_seconds"] == 80.0
    assert kpi["passenger_wait_p95_seconds"] == 180.0
    assert kpi["p95_contains_censored_passengers"] is True
    assert kpi["passenger_conservation_passed"] is True


def test_path_a_b_equivalence_and_permutation_invariance() -> None:
    rows = _rows()
    base = compare_kpi_paths(rows)
    assert base["passed"] is True
    shuffled = list(rows)
    random.Random(1).shuffle(shuffled)
    assert compare_kpi_paths(shuffled)["path_a"]["avg_wait_seconds"] == base["path_a"]["avg_wait_seconds"]


def test_duplicate_ledger_key_is_blocked() -> None:
    rows = _rows()
    validation = validate_ledger_rows(rows + [dict(rows[0])])
    assert validation["duplicate_key_count"] == 1
    with pytest.raises(ValueError):
        compute_passenger_wait_kpi_v2(rows + [dict(rows[0])])
