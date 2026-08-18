#!/usr/bin/env python3
"""H4M-AE-R3.1 measured passenger wait tail accounting validation (Vp0-Vp12).

Contract validation only: no training, no optimizer, no performance ranking, no
A/B1/B2 comparison, no demand calibration, no TEST6 access.  Vp7/Vp8 are
synthetic fixture-policy incentive controls, not evaluation arms.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import test_h4m_ae_r3_causal_kpi_bridge as r3

TRAINING_ROOT = Path(__file__).resolve().parent
LEGACY_FALLBACK_MULTIPLIER = 1.65
TOL = 1e-9


def drive(adapter: Any, actions: List[int], steps: int) -> None:
    adapter.reset()
    n = len(adapter.state["vehicles"])
    for i in range(steps):
        act = {a: actions[(i + a) % len(actions)] for a in range(n)}
        mask = {a: [True, True, True] for a in range(n)}
        targets = {a: 1 for a in range(n)}
        adapter.step(act, legal_mask=mask, target_ids=targets, provenance={"policy_source_mode": "validation_fixture"})


def run_validations() -> Dict[str, Any]:
    bridge = r3.imp("bridge", r3.BRIDGE)
    qmod = r3.imp("h4mq", r3.H4MQ)
    dl1 = r3.imp("dl1", r3.DL1)
    dl4 = r3.imp("dl4", r3.DL4)
    energy_mod = r3.imp("energy", r3.ENERGY)
    agg = r3.imp("agg", r3.AGGREGATOR)
    plan, sample_graph, config, data, device, window = r3.build_env(bridge, qmod, dl1, dl4)
    demand = r3.demand_fields_for(window["window_id"])
    agents = len(plan["agent_indices"]) if isinstance(plan.get("agent_indices"), list) else 4

    def adapter(seed: int = 1):
        return bridge.PV8CausalKpiAdapter(window=window, num_agents=agents, seed=seed, demand_fields=demand)

    checks: Dict[str, Any] = {}

    # -- VpA authoritative arrival_schedule unit / weight semantics (spec 4) ---
    base = adapter()
    base.reset()
    stops = base.state["stops"]
    schedule_total = sum(len(s.arrival_schedule) for s in stops.values())
    weight_field = any(hasattr(s, n) for s in stops.values() for n in ("arrival_weights", "arrival_multiplicity"))
    checks["VpA_arrival_entry_semantics"] = {
        "declared": bridge.ARRIVAL_ENTRY_SEMANTICS,
        "schedule_entry_total": schedule_total,
        "passenger_demand_generated": int(base.accounting.passenger_demand_generated),
        "multiplicity_field_present": weight_field,
        "entries_are_scalar_timestamps": all(isinstance(v, float) for s in stops.values() for v in s.arrival_schedule),
        "percentile_family": "EXACT_UNWEIGHTED_EMPIRICAL",
        "passed": (
            bridge.ARRIVAL_ENTRY_SEMANTICS == "ONE_ENTRY_ONE_SIMULATED_PASSENGER_UNIT_WEIGHT"
            and schedule_total == int(base.accounting.passenger_demand_generated)
            and not weight_field
        ),
    }

    # -- Vp0 carry-in / initial-wait population boundary audit -----------------
    initial_waiting = sum(s.waiting_count for s in stops.values())
    outside = [v for s in stops.values() for v in s.arrival_schedule if v < 0.0 or v > base.horizon_seconds]
    if initial_waiting == 0 and not outside:
        case = "A_NO_INITIAL_WAITING_QUEUE"
    elif not outside:
        case = "B_CARRY_IN_WITH_KNOWN_ARRIVAL_TS"
    else:
        case = "C_CARRY_IN_WITH_UNKNOWN_ARRIVAL_TS"
    checks["Vp0_carry_in_population_boundary"] = {
        "case": case,
        "declared": bridge.CARRY_IN_CASE,
        "initial_waiting_count": int(initial_waiting),
        "arrivals_outside_horizon": len(outside),
        "carry_in_timestamp_status": "NOT_APPLICABLE_NO_CARRY_IN" if case.startswith("A") else "AUTHORITATIVE_ARRIVAL_TS_PRESENT",
        "fail_closed_triggered": case.startswith("C"),
        "passed": not case.startswith("C") and case == bridge.CARRY_IN_CASE,
    }

    # main measured scenario (mixed policy)
    mixed = adapter()
    drive(mixed, [1, 0, 1], 12)
    pop = mixed.finalize_wait_population()
    canonical = pop["canonical_p95"]
    row = mixed.window_rollup_row(
        condition_id="A",
        provenance={"policy_source_mode": "validation_fixture", "checkpoint_path": "n/a"},
        energy_model=energy_mod,
        baseline_bus_count=agents,
    )

    # -- Vp1 completed ledger sum == wait_total_passenger_seconds --------------
    ledger_sum = float(sum(pop["completed_wait_ledger"]))
    checks["Vp1_completed_sum_matches_wait_total"] = {
        "completed_ledger_sum": ledger_sum,
        "wait_total_passenger_seconds": row["wait_total_passenger_seconds"],
        "abs_diff": abs(ledger_sum - row["wait_total_passenger_seconds"]),
        "passed": abs(ledger_sum - row["wait_total_passenger_seconds"]) <= 1e-6,
    }

    # -- Vp2 completed count vs wait_passenger_count and passenger_served_count
    completed_n = len(pop["completed_wait_ledger"])
    served_n = int(row["passenger_served_count"])
    same_contract = completed_n == served_n
    checks["Vp2_completed_count_contract"] = {
        "completed_count": completed_n,
        "wait_passenger_count": int(row["wait_passenger_count"]),
        "passenger_served_count": served_n,
        "contracts_identical": same_contract,
        "post_horizon_boarding_count": pop["post_horizon_boarding_count"],
        "documented_reason": (
            "identical population contract"
            if same_contract
            else (
                "passenger_served_count keeps the R3-frozen definition (every SERVE boarding on the simulator "
                "vehicle clock). The wait population is closed at the fixed evaluation horizon end as R3.1 "
                "section 2.2 requires, so boardings whose board_ts exceeds horizon_end are censored instead of "
                "completed. The gap equals post_horizon_boarding_count and is caused by the uncalibrated demand "
                "scale deferred to R4; no existing KPI definition is altered by R3.1."
            )
        ),
        "existing_kpi_semantics_contradicted": False,
        "passed": (
            int(row["wait_passenger_count"]) == completed_n
            and (same_contract or completed_n + pop["post_horizon_boarding_count"] == served_n)
        ),
    }

    # -- Vp3 censored count is the exact complement ----------------------------
    censored_n = len(pop["censored_wait_ledger"])
    checks["Vp3_censored_count_complement"] = {
        "population_count": canonical["population_count"],
        "completed_count": completed_n,
        "censored_count": censored_n,
        "censored_share": canonical["censored_share"],
        "demand_generated": pop["demand_generated"],
        "reconciles_demand": pop["population_reconciles_demand"],
        "passed": (
            censored_n == canonical["population_count"] - completed_n
            and completed_n + censored_n == pop["demand_generated"]
        ),
    }

    # -- Vp4 every censored duration == horizon_end - authoritative arrival_ts -
    horizon = mixed.horizon_seconds
    from collections import Counter

    arrival_pool = Counter(round(v, 6) for s in mixed.state["stops"].values() for v in s.arrival_schedule)
    implied = Counter(round(horizon - c, 6) for c in pop["censored_wait_ledger"])
    unmatched = {a: n for a, n in implied.items() if arrival_pool.get(a, 0) < n}
    in_range = all(0.0 <= c <= horizon for c in pop["censored_wait_ledger"])
    exact = not unmatched
    checks["Vp4_censored_duration_exact"] = {
        "censor_reference": bridge.CENSOR_REFERENCE,
        "horizon_end_seconds": horizon,
        "all_derived_from_horizon_end": exact,
        "unmatched_implied_arrival_count": len(unmatched),
        "in_window_arrivals_bounded": in_range,
        "max_censored_seconds": max(pop["censored_wait_ledger"]) if censored_n else None,
        "carry_in_may_exceed_horizon": bridge.CARRY_IN_CASE != "A_NO_INITIAL_WAITING_QUEUE",
        "passed": exact and in_range,
    }

    # -- Vp5 median <= p95 <= max ----------------------------------------------
    checks["Vp5_median_le_p95_le_max"] = {
        "median_seconds": canonical["median_seconds"],
        "p95_seconds": canonical["value_seconds"],
        "max_seconds": canonical["max_seconds"],
        "support": canonical["p95_support"],
        "p95_support_includes_censored_observation": canonical["p95_support_includes_censored_observation"],
        "p95_exact_support_is_censored": canonical["p95_exact_support_is_censored"],
        "interpolated_between_two_support_points": canonical["p95_interpolated_between_two_support_points"],
        "passed": canonical["median_seconds"] <= canonical["value_seconds"] <= canonical["max_seconds"],
    }

    # -- Vp6 determinism: same seed + same actions -> identical ledger and p95 -
    repeat = adapter()
    drive(repeat, [1, 0, 1], 12)
    rep = repeat.finalize_wait_population()
    probe = (
        "import sys,json;sys.path.insert(0,%r);"
        "import test_h4m_ae_r3_causal_kpi_bridge as r3;"
        "b=r3.imp('bridge',r3.BRIDGE);"
        "a=b.PV8CausalKpiAdapter(window=%s,num_agents=%d,seed=1,demand_fields=%s);a.reset();"
        "print(json.dumps([s.arrival_schedule for s in a.state['stops'].values()]))"
    ) % (str(TRAINING_ROOT), repr(dict(window)), agents, repr(demand))
    outs = [
        subprocess.run(
            [sys.executable, "-c", probe], cwd=TRAINING_ROOT,
            env=dict(os.environ, PYTHONHASHSEED=salt), capture_output=True, text=True, check=True,
        ).stdout.strip()
        for salt in ("0", "12345")
    ]
    cross_process = outs[0] == outs[1] and json.loads(outs[0]) == [s.arrival_schedule for s in stops.values()]
    checks["Vp6_deterministic_ledger_and_p95"] = {
        "completed_equal": rep["completed_wait_ledger"] == pop["completed_wait_ledger"],
        "censored_equal": rep["censored_wait_ledger"] == pop["censored_wait_ledger"],
        "p95_equal": rep["canonical_p95"]["value_seconds"] == canonical["value_seconds"],
        "cross_process_hash_salts": ["0", "12345"],
        "cross_process_identical": cross_process,
        "passed": (
            rep["completed_wait_ledger"] == pop["completed_wait_ledger"]
            and rep["censored_wait_ledger"] == pop["censored_wait_ledger"]
            and rep["canonical_p95"]["value_seconds"] == canonical["value_seconds"]
            and cross_process
        ),
    }

    # -- Vp7 positive control: persistent service denial worsens the tail ------
    denial = adapter()
    drive(denial, [0], 12)
    denial_pop = denial.finalize_wait_population()
    denial_p95 = denial_pop["canonical_p95"]
    checks["Vp7_service_denial_positive_control"] = {
        "control_policy": "PERSISTENT_HOLD_NO_SERVE",
        "control_censored_count": len(denial_pop["censored_wait_ledger"]),
        "mixed_censored_count": censored_n,
        "control_p95_seconds": denial_p95["value_seconds"],
        "mixed_p95_seconds": canonical["value_seconds"],
        "passed": (
            len(denial_pop["censored_wait_ledger"]) >= censored_n
            and denial_p95["value_seconds"] >= canonical["value_seconds"]
        ),
    }

    # -- Vp8 same demand realization: immediate SERVE <= persistent HOLD -------
    serve = adapter()
    drive(serve, [1], 12)
    serve_pop = serve.finalize_wait_population()
    serve_p95 = serve_pop["canonical_p95"]
    same_demand = serve_pop["demand_generated"] == denial_pop["demand_generated"]
    checks["Vp8_serve_vs_hold_ordering"] = {
        "same_demand_realization": same_demand,
        "immediate_serve_p95_seconds": serve_p95["value_seconds"],
        "persistent_hold_p95_seconds": denial_p95["value_seconds"],
        "immediate_serve_completed": len(serve_pop["completed_wait_ledger"]),
        "persistent_hold_completed": len(denial_pop["completed_wait_ledger"]),
        "passed": same_demand and serve_p95["value_seconds"] <= denial_p95["value_seconds"],
    }

    # -- Vp9 avg*1.65 and fillna(0.0) unreachable on the promoted causal path --
    import pandas as pd

    raw = pd.DataFrame([dict(row, input_source_path="validation_fixture")])
    frame = agg.compute_official_kpi_by_window(raw)
    emitted = float(frame["passenger_wait_p95_seconds"].iloc[0])
    avg_wait = float(frame["avg_wait_seconds"].iloc[0])
    stripped = agg.compute_official_kpi_by_window(raw.drop(columns=["passenger_wait_p95_seconds"]))
    legacy_value = float(stripped["passenger_wait_p95_seconds"].iloc[0])
    nulled = raw.copy()
    nulled.loc[:, "passenger_wait_p95_seconds"] = np.nan
    fillna_value = float(agg.compute_official_kpi_by_window(nulled)["passenger_wait_p95_seconds"].iloc[0])
    checks["Vp9_synthetic_and_fillna_unreachable"] = {
        "rollup_emits_measured_p95": row["passenger_wait_p95_seconds"] is not None,
        "aggregator_p95": emitted,
        "measured_p95": canonical["value_seconds"],
        "legacy_multiplier_branch_value_if_column_absent": legacy_value,
        "legacy_branch_equals_avg_times_1_65": abs(legacy_value - avg_wait * LEGACY_FALLBACK_MULTIPLIER) <= 1e-6,
        "fillna_zero_value_if_column_null": fillna_value,
        "promoted_path_reaches_multiplier_branch": False,
        "promoted_path_reaches_fillna_branch": False,
        "fallback_used_flag": row["wait_tail_fallback_used"],
        "legacy_behaviour_status": "KNOWN_GUARDED_LEGACY_BEHAVIOUR_UNCHANGED_BY_R3_1",
        "passed": (
            row["passenger_wait_p95_seconds"] is not None
            and row["wait_tail_measured"] is True
            and row["wait_tail_fallback_used"] is False
            and abs(emitted - float(canonical["value_seconds"])) <= TOL
            and abs(legacy_value - avg_wait * LEGACY_FALLBACK_MULTIPLIER) <= 1e-6
            and abs(emitted - legacy_value) > 1e-6
        ),
    }

    # -- Vp10 percentile method frozen and reproducible ------------------------
    independent = float(
        np.percentile(np.asarray(pop["completed_wait_ledger"] + pop["censored_wait_ledger"], dtype=float), 95.0, method="linear")
    )
    checks["Vp10_percentile_method_frozen"] = {
        "percentile_q": canonical["percentile_q"],
        "percentile_method": canonical["percentile_method"],
        "library": canonical["percentile_library"],
        "numpy_version": np.__version__,
        "weighting": canonical["weighting"],
        "bridge_value": canonical["value_seconds"],
        "independent_recomputation": independent,
        "passed": (
            canonical["percentile_q"] == 95.0
            and canonical["percentile_method"] == "linear"
            and canonical["weighting"] == "UNWEIGHTED_UNIT_MULTIPLICITY"
            and abs(canonical["value_seconds"] - independent) <= TOL
        ),
    }

    # -- Vp11 served-only tail is diagnostic only ------------------------------
    diag = pop["served_only_p95_diagnostic"]
    canonical_columns = [c for c in frame.columns if "p95" in c]
    checks["Vp11_served_only_noncanonical"] = {
        "canonical": diag["canonical"],
        "promotion_use_allowed": diag["promotion_use_allowed"],
        "performance_claim_allowed": diag["performance_claim_allowed"],
        "diagnostic_value_seconds": diag["value_seconds"],
        "canonical_population": canonical["population_definition"],
        "p95_columns_in_canonical_frame": canonical_columns,
        "passed": (
            diag["canonical"] is False
            and diag["promotion_use_allowed"] is False
            and diag["performance_claim_allowed"] is False
            and canonical["population_definition"] == "SERVED_COMPLETED_PLUS_CENSORED_LOWER_BOUND"
            and canonical_columns == ["passenger_wait_p95_seconds"]
        ),
    }

    # -- Vp12 higher-level p95 never averages window percentiles ---------------
    second = adapter(seed=2)
    drive(second, [1, 2, 0], 12)
    second_pop = second.finalize_wait_population()
    pooled = bridge.pooled_wait_p95([pop, second_pop])
    union = (
        pop["completed_wait_ledger"] + pop["censored_wait_ledger"]
        + second_pop["completed_wait_ledger"] + second_pop["censored_wait_ledger"]
    )
    expected = float(np.percentile(np.asarray(union, dtype=float), 95.0, method="linear"))
    mean_of_window = float(np.mean([canonical["value_seconds"], second_pop["canonical_p95"]["value_seconds"]]))
    checks["Vp12_no_percentile_averaging"] = {
        "aggregation": pooled["aggregation"],
        "pooled_value_seconds": pooled["value_seconds"],
        "expected_pooled_value_seconds": expected,
        "prohibited_mean_of_window_p95": mean_of_window,
        "pooled_population_count": pooled["population_count"],
        "higher_level_p95_aggregation_status": "POOLED_LEDGER_RECOMPUTATION_AVAILABLE",
        "passed": (
            pooled["aggregation"] == "POOLED_RAW_PASSENGER_LEDGER"
            and abs(pooled["value_seconds"] - expected) <= TOL
            and pooled["population_count"] == len(union)
            and abs(pooled["value_seconds"] - mean_of_window) > TOL
        ),
    }

    # -- VpB empty population resolves to NOT_APPLICABLE (spec 7) --------------
    empty = bridge.measured_wait_p95([], [])
    checks["VpB_empty_population_not_applicable"] = {
        "status": empty["status"],
        "value_seconds": empty["value_seconds"],
        "passed": empty["status"] == "NOT_APPLICABLE" and empty["value_seconds"] is None,
    }

    # -- VpC frozen avg-wait semantics preserved (spec 2.1) --------------------
    ledger_mean = float(np.mean(pop["completed_wait_ledger"])) if pop["completed_wait_ledger"] else 0.0
    censored_leak = abs(row["wait_total_passenger_seconds"] - ledger_sum) > 1e-6
    checks["VpC_avg_wait_semantics_preserved"] = {
        "definition": bridge.KPI_FIELD_PROVENANCE["avg_wait_seconds"],
        "aggregator_avg_wait_seconds": avg_wait,
        "completed_ledger_mean": ledger_mean,
        "censored_leaked_into_mean": censored_leak,
        "formula_unchanged": "wait_total_passenger_seconds / wait_passenger_count",
        "passed": not censored_leak and abs(avg_wait - ledger_mean) <= 1e-6,
    }

    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-R3.1",
        "window_id": str(window.get("window_id")),
        "checks": checks,
        "measured_p95_seconds": canonical["value_seconds"],
        "all_passed": all(bool(c["passed"]) for c in checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    result = run_validations()
    if args.json_output:
        args.json_output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
    failed = [k for k, v in result["checks"].items() if not v["passed"]]
    if failed:
        raise SystemExit(f"[FAIL] H4M-AE-R3.1 measured wait tail validation failed: {failed}")
    print("[PASS] H4M-AE-R3.1 measured passenger wait tail validation passed (Vp0-Vp12 + VpA/VpB/VpC)")


if __name__ == "__main__":
    main()
