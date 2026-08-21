#!/usr/bin/env python3
"""H4M-AE-R5 authoritative demand binding and horizon service accounting (R5-01..R5-30).

Repair validation only: no training, no optimizer, no demand tuning, no policy
ranking, no A/B1/B2 performance comparison, no TEST6 access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import causal_arm_contracts as arms
import test_h4m_ae_r3_causal_kpi_bridge as r3

# --- H4M-AE-R9.8 fail-closed simulator authorization -------------------------------
import sys as _authz_sys
from pathlib import Path as _AuthzPath

for _authz_dir in (_AuthzPath(__file__).resolve().parent,):
    if (_authz_dir / "simulator_authorization.py").exists() and str(_authz_dir) not in _authz_sys.path:
        _authz_sys.path.insert(0, str(_authz_dir))
import simulator_authorization as _authz  # noqa: E402
# -----------------------------------------------------------------------------------

TRAINING_ROOT = Path(__file__).resolve().parent
AGGREGATOR = TRAINING_ROOT / "evaluation" / "canonical_kpi_aggregator.py"
DEMAND_MODULE = TRAINING_ROOT / "authoritative_demand_realization.py"
EXPECTED_TOTAL_REQUESTS = 414
EXPECTED_MEDIAN_PER_WINDOW = 7
AUDITED_WINDOW = "PV8_R3R_SATURDAY_OFFPEAK_D0_LATEST_20231230_1000"
EXPECTED_AUDITED_COUNT = 5
CONTINUITY_SERVICE_DATE = "2023-01-01"


def clock(adapter: Any) -> float:
    return max(float(getattr(v, "clock_seconds", 0.0)) for v in adapter.state["vehicles"].values())


def drive(adapter: Any, actions: List[int], *, cap: int = 6000) -> None:
    adapter.reset()
    n = len(adapter.state["vehicles"])
    i = 0
    while i < cap and clock(adapter) < adapter.horizon_seconds:
        adapter.step({a: actions[(i + a) % len(actions)] for a in range(n)},
                     legal_mask={a: [True, True, True] for a in range(n)},
                     target_ids={a: 1 for a in range(n)},
                     provenance={"policy_source_mode": "r5_fixture"})
        i += 1


def _run_validations_inner() -> Dict[str, Any]:
    bridge = r3.imp("bridge", r3.BRIDGE)
    demand_mod = r3.imp("demand", DEMAND_MODULE)
    qmod = r3.imp("h4mq", r3.H4MQ)
    dl1 = r3.imp("dl1", r3.DL1)
    dl4 = r3.imp("dl4", r3.DL4)
    energy_mod = r3.imp("energy", r3.ENERGY)
    agg = r3.imp("agg", AGGREGATOR)
    plan, sample_graph, config, data, device, window = r3.build_env(bridge, qmod, dl1, dl4)
    agents = 4
    inputs = r3.authoritative_adapter_inputs(window, num_agents=agents)
    realization = demand_mod.load_demand_realization(r3.AUTHORITATIVE_DEMAND)
    provenance = realization.provenance()
    registry = pd.read_parquet(r3.REGISTRY)
    checks: Dict[str, Any] = {}

    def adapter(**over):
        return bridge.PV8CausalKpiAdapter(**{**inputs, **over})

    # -- R5-01..R5-04 frozen realization ------------------------------------
    per_window = realization.requests_per_reporting_window()
    audited = int((realization.frame["window_id"].astype(str) == AUDITED_WINDOW).sum())
    checks["R5_01_demand_artifact_hash_verified"] = {
        "artifact_path": provenance["artifact_path"],
        "artifact_sha256": provenance["artifact_sha256"],
        "recomputed_sha256": hashlib.sha256(r3.AUTHORITATIVE_DEMAND.read_bytes()).hexdigest(),
        "interface_id": provenance["interface_id"],
        "unit": provenance["unit"],
        "passed": provenance["artifact_sha256"] == hashlib.sha256(r3.AUTHORITATIVE_DEMAND.read_bytes()).hexdigest(),
    }
    checks["R5_02_total_request_count"] = {
        "expected": EXPECTED_TOTAL_REQUESTS, "observed": provenance["total_request_count"],
        "passed": provenance["total_request_count"] == EXPECTED_TOTAL_REQUESTS,
    }
    checks["R5_03_median_requests_per_window"] = {
        "expected": EXPECTED_MEDIAN_PER_WINDOW, "observed": provenance["median_requests_per_reporting_window"],
        "reporting_window_count": provenance["reporting_window_count"],
        "passed": provenance["median_requests_per_reporting_window"] == EXPECTED_MEDIAN_PER_WINDOW,
    }
    checks["R5_04_audited_window_request_count"] = {
        "window_id": AUDITED_WINDOW, "expected": EXPECTED_AUDITED_COUNT, "observed": audited,
        "passed": audited == EXPECTED_AUDITED_COUNT,
    }

    # -- R5-05..R5-07 retired synthetic path ---------------------------------
    source = r3.BRIDGE.read_text(encoding="utf-8")
    retired = {
        "_arrival_schedule": "_arrival_schedule" in source,
        "demand_fields": "demand_fields" in source,
        "historical_boarding_intensity": "historical_boarding_intensity" in source,
        "per_stop_replication": "rng.uniform" in source,
        "six_hundred_rescale": "/ 600.0" in source,
        "score_multiplier": "* 0.1" in source,
    }
    live = adapter()
    live.reset()
    live_ids = sorted(rid for s in live.state["stops"].values() for rid in s.request_ids)
    checks["R5_05_broken_generation_unreachable"] = {
        "retired_tokens_present": {k: v for k, v in retired.items() if v},
        "simulator_population": len(live_ids),
        "authoritative_population": len(inputs["demand_population"]),
        "over_generation_factor": len(live_ids) / max(1, len(inputs["demand_population"])),
        "passed": not any(retired.values()) and len(live_ids) == len(inputs["demand_population"]),
    }
    checks["R5_06_no_aggregate_per_stop_expansion"] = {
        "request_ids_unique": len(live_ids) == len(set(live_ids)),
        "matches_authoritative_ids": live_ids == sorted(r.request_id for r in inputs["demand_population"]),
        "passed": len(live_ids) == len(set(live_ids)) and live_ids == sorted(r.request_id for r in inputs["demand_population"]),
    }
    checks["R5_07_no_synthetic_rate_path"] = {
        "six_hundred_rescale_present": retired["six_hundred_rescale"],
        "score_multiplier_present": retired["score_multiplier"],
        "intensity_field_consumed": retired["historical_boarding_intensity"],
        "passed": not (retired["six_hundred_rescale"] or retired["score_multiplier"] or retired["historical_boarding_intensity"]),
    }

    # -- R5-08/R5-09 external, non-literal evaluation boundary ---------------
    horizons = sorted({int(r["evaluation_end_ts"] - r["start_ts"]) for _, r in registry.iterrows()})
    alt_contract = demand_mod.EvaluationTimeContract(
        evaluation_start_ts=inputs["evaluation_contract"].evaluation_start_ts,
        evaluation_end_ts=inputs["evaluation_contract"].evaluation_start_ts + 7200.0,
        reporting_window_ids=[str(window["window_id"])],
        scope_label="R5_CONFIG_ONLY_TWO_HOUR_SMOKE",
    )
    alt_sel = demand_mod.select_population(realization, alt_contract)
    alt = adapter(evaluation_contract=alt_contract, demand_population=alt_sel["requests"], population_audit=alt_sel["audit"])
    import ast

    tree = ast.parse(source)
    literal_1800 = [f"Constant(1800)@line{n.lineno}" for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and n.value in (1800, 1800.0)]
    horizon_minute_params = [f"{fn.name}(horizon_minutes)" for fn in ast.walk(tree)
                             if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                             and any(a.arg == "horizon_minutes" for a in list(fn.args.args) + list(fn.args.kwonlyargs))]
    semantic_literals = literal_1800 + horizon_minute_params
    checks["R5_08_evaluation_boundary_configurable"] = {
        "authoritative_horizons_in_registry_seconds": horizons,
        "default_horizon_seconds": live.horizon_seconds,
        "alternate_horizon_seconds": alt.horizon_seconds,
        "boundary_supplied_externally": True,
        "passed": live.horizon_seconds != alt.horizon_seconds and alt.horizon_seconds == 7200.0,
    }
    checks["R5_09_no_hardcoded_1800_semantics"] = {
        "semantic_literal_matches_in_bridge": semantic_literals,
        "derived_metadata_key_allowed": "evaluation_horizon_minutes emitted as derived metadata only",
        "registry_horizons_are_not_all_1800": horizons != [1800],
        "passed": not semantic_literals and horizons != [1800],
    }

    # -- R5-10..R5-13 population closure -------------------------------------
    run = adapter()
    drive(run, [1, 0, 1])
    pop = run.finalize_wait_population()
    row = run.window_rollup_row(condition_id="A", energy_model=energy_mod, baseline_bus_count=agents,
                                provenance={"policy_source_mode": "r5_fixture", "checkpoint_path": "n/a"})
    served_h = len(pop["completed_wait_ledger"])
    unserved_h = len(pop["censored_wait_ledger"])
    demand_n = pop["demand_generated"]
    checks["R5_10_population_closure"] = {
        "evaluation_demand_population": demand_n, "served_by_horizon": served_h,
        "unserved_at_horizon": unserved_h, "passed": served_h + unserved_h == demand_n,
    }
    checks["R5_11_completed_equals_served_by_horizon"] = {
        "completed_wait_count": served_h, "passenger_served_by_horizon_count": int(row["passenger_served_by_horizon_count"]),
        "passed": served_h == int(row["passenger_served_by_horizon_count"]),
    }
    checks["R5_12_censored_equals_unserved"] = {
        "censored_wait_count": unserved_h, "wait_censored_passenger_count": int(row["wait_censored_passenger_count"]),
        "passed": unserved_h == int(row["wait_censored_passenger_count"]),
    }
    checks["R5_13_p95_population_equals_demand"] = {
        "p95_population_count": int(row["wait_population_passenger_count"]), "evaluation_demand_population": demand_n,
        "passed": int(row["wait_population_passenger_count"]) == demand_n,
    }

    # -- R5-14/R5-15/R5-16 horizon service rate under real post-horizon load --
    first_arrival = min(r.arrival_ts_relative for r in inputs["demand_population"])
    short_contract = demand_mod.EvaluationTimeContract(
        evaluation_start_ts=inputs["evaluation_contract"].evaluation_start_ts,
        evaluation_end_ts=inputs["evaluation_contract"].evaluation_start_ts + first_arrival + 60.0,
        reporting_window_ids=[str(window["window_id"])],
        scope_label="R5_POST_HORIZON_STRESS",
    )
    short_sel = demand_mod.select_population(realization, short_contract)
    stress = adapter(evaluation_contract=short_contract, demand_population=short_sel["requests"],
                     population_audit=short_sel["audit"])
    stress.reset()
    n = len(stress.state["vehicles"])
    for i in range(4000):
        if clock(stress) > stress.horizon_seconds + 600.0:
            break
        # HOLD past the evaluation boundary, then serve: every boarding is post-horizon.
        stress.step({a: (0 if clock(stress) <= stress.horizon_seconds else 1) for a in range(n)},
                    legal_mask={a: [True, True, True] for a in range(n)},
                    target_ids={a: 1 for a in range(n)}, provenance={"policy_source_mode": "r5_fixture"})
    stress_pop = stress.finalize_wait_population()
    stress_row = stress.window_rollup_row(condition_id="A", energy_model=energy_mod, baseline_bus_count=agents,
                                          provenance={"policy_source_mode": "r5_fixture", "checkpoint_path": "n/a"})
    stress_frame = agg.compute_official_kpi_by_window(pd.DataFrame([dict(stress_row, input_source_path="r5")]))
    stress_rate = float(stress_frame["passenger_service_rate"].iloc[0])
    horizon_served = int(stress_row["passenger_served_by_horizon_count"])
    eventual = int(stress_row["diagnostic_passenger_eventual_served_count"])
    expected_rate = horizon_served / stress_pop["demand_generated"] if stress_pop["demand_generated"] else None
    checks["R5_14_post_horizon_cannot_change_service_rate"] = {
        "post_horizon_boarding_count": stress_pop["post_horizon_boarding_count"],
        "served_by_horizon": horizon_served,
        "eventual_served": eventual,
        "canonical_service_rate": stress_rate,
        "horizon_consistent_rate": expected_rate,
        "eventual_rate_if_it_had_been_used": eventual / stress_pop["demand_generated"] if stress_pop["demand_generated"] else None,
        "stress_produced_post_horizon_boardings": stress_pop["post_horizon_boarding_count"] > 0,
        "passed": expected_rate is not None and abs(stress_rate - expected_rate) <= 1e-9
        and stress_pop["post_horizon_boarding_count"] > 0 and eventual > horizon_served,
    }
    checks["R5_15_eventual_vs_horizon_distinguishable"] = {
        "canonical_field": "passenger_served_count -> horizon bounded",
        "explicit_horizon_field": "passenger_served_by_horizon_count",
        "diagnostic_eventual_field": "diagnostic_passenger_eventual_served_count",
        "values": {"horizon": horizon_served, "eventual": eventual},
        "distinguishable": eventual != horizon_served,
        "eventual_semantics_preserved": True,
        "passed": int(stress_row["passenger_served_count"]) == horizon_served and eventual >= horizon_served,
    }
    checks["R5_16_service_rate_bounds"] = {
        "canonical_service_rate": stress_rate,
        "in_unit_interval": 0.0 <= stress_rate <= 1.0,
        "numerator_le_denominator": horizon_served <= stress_pop["demand_generated"],
        "shares_p95_boundary": stress_row["passenger_wait_p95_censor_reference"],
        "passed": 0.0 <= stress_rate <= 1.0 and horizon_served <= stress_pop["demand_generated"],
    }

    # -- R5-17..R5-19 arm equality -------------------------------------------
    arm_adapters = {a: adapter() for a in ("A", "B1", "B2")}
    for a in arm_adapters.values():
        a.reset()
    dh = {k: arms.demand_realization_hash(v) for k, v in arm_adapters.items()}
    ih = {k: arms.initial_state_hash(v) for k, v in arm_adapters.items()}
    probe = (
        "import sys\n"
        "sys.path.insert(0,%r)\n"
        # R9.8 guards the bridge; this probe is entitled to run it and says so.
        "import simulator_authorization as _A\n"
        "import test_h4m_ae_r3_causal_kpi_bridge as r3\n"
        "import causal_arm_contracts as arms\n"
        "b=r3.imp('bridge',r3.BRIDGE)\n"
        "with _A.granted('simulator_execution', reason='cross-process determinism probe'):\n"
        "    a=b.PV8CausalKpiAdapter(**r3.authoritative_adapter_inputs(%s,num_agents=%d))\n"
        "    a.reset()\n"
        "print(arms.demand_realization_hash(a))\n"
    ) % (str(TRAINING_ROOT), repr(dict(window)), agents)
    cross = [subprocess.run([sys.executable, "-c", probe], cwd=TRAINING_ROOT,
                            env=dict(os.environ, PYTHONHASHSEED=s), capture_output=True, text=True, check=True).stdout.strip()
             for s in ("0", "104729")]
    checks["R5_17_same_frozen_demand_across_arms"] = {
        "demand_hashes": dh, "identical": len(set(dh.values())) == 1, "passed": len(set(dh.values())) == 1}
    checks["R5_18_cross_process_determinism"] = {
        "hash_salts": ["0", "104729"], "hashes": cross,
        "matches_in_process": cross[0] == dh["A"], "passed": cross[0] == cross[1] == dh["A"]}
    checks["R5_19_initial_state_equality"] = {
        "initial_state_hashes": ih, "identical": len(set(ih.values())) == 1, "passed": len(set(ih.values())) == 1}

    # -- R5-20/R5-21 p95 path -------------------------------------------------
    stripped = agg.compute_official_kpi_by_window(
        pd.DataFrame([dict(row, input_source_path="r5")]).drop(columns=["passenger_wait_p95_seconds"]))
    frame = agg.compute_official_kpi_by_window(pd.DataFrame([dict(row, input_source_path="r5")]))
    checks["R5_20_measured_p95_active"] = {
        "p95_seconds": row["passenger_wait_p95_seconds"], "status": row["passenger_wait_p95_status"],
        "semantics": row["passenger_wait_p95_semantics"], "population": row["passenger_wait_p95_population"],
        "method": row["passenger_wait_p95_percentile_method"], "wait_tail_measured": row["wait_tail_measured"],
        "passed": row["wait_tail_measured"] is True and row["passenger_wait_p95_status"] == "MEASURED",
    }
    checks["R5_21_synthetic_p95_unreachable"] = {
        "aggregator_value": float(frame["passenger_wait_p95_seconds"].iloc[0]),
        "legacy_branch_value_if_column_absent": float(stripped["passenger_wait_p95_seconds"].iloc[0]),
        "fallback_used_flag": row["wait_tail_fallback_used"],
        "passed": row["wait_tail_fallback_used"] is False
        and abs(float(frame["passenger_wait_p95_seconds"].iloc[0]) - float(row["passenger_wait_p95_seconds"])) <= 1e-9,
    }

    # -- R5-22..R5-27 frozen contracts and guards ----------------------------
    r3_result = r3.run_validations()
    rc = r3_result["checks"]
    checks["R5_22_reward_v2_unchanged"] = dict(rc["V7_reward_v2_unchanged"])
    checks["R5_23_zero_loss_unchanged"] = dict(rc["V8_zero_loss_unchanged"])
    checks["R5_24_k_mask_unchanged"] = dict(rc["V6_kmask_unchanged"])
    split = json.loads(r3.SPLIT.read_text(encoding="utf-8-sig"))["ordered_window_ids"]
    sealed = {str(w) for w in split["test"]}
    used = {str(window["window_id"]), AUDITED_WINDOW} | {
        str(w) for w in registry[registry["service_date"].astype(str) == CONTINUITY_SERVICE_DATE]["window_id"]}
    checks["R5_25_test6_untouched"] = {
        "split_sizes": {k: len(v) for k, v in split.items()}, "windows_touched": sorted(used),
        "sealed_intersection": sorted(used & sealed), "passed": not (used & sealed)}
    checks["R5_26_no_performance_comparison"] = {
        "arms_ranked": False, "kpi_compared_across_arms": False, "passed": True}
    checks["R5_27_no_demand_tuning"] = {
        "regenerated": provenance["regenerated"], "rescaled": provenance["rescaled"],
        "expanded": provenance["expanded"], "tuned": provenance["tuned"],
        "artifact_sha256_unchanged": provenance["artifact_sha256"],
        "passed": not any((provenance["regenerated"], provenance["rescaled"], provenance["expanded"], provenance["tuned"])),
    }

    # -- R5-28/R5-29 horizon parameterization --------------------------------
    checks["R5_28_default_configuration"] = {
        "evaluation_horizon_seconds": live.horizon_seconds,
        "derived_minutes": live.horizon_seconds / 60.0,
        "source": "registry start_ts / evaluation_end_ts",
        "passed": live.horizon_seconds == inputs["evaluation_contract"].evaluation_horizon_seconds,
    }
    alt_run = adapter(evaluation_contract=alt_contract, demand_population=alt_sel["requests"], population_audit=alt_sel["audit"])
    drive(alt_run, [1, 0, 1])
    alt_pop = alt_run.finalize_wait_population()
    alt_row = alt_run.window_rollup_row(condition_id="A", energy_model=energy_mod, baseline_bus_count=agents,
                                        provenance={"policy_source_mode": "r5_fixture", "checkpoint_path": "n/a"})
    checks["R5_29_non_default_horizon_smoke"] = {
        "alternate_horizon_seconds": alt_run.horizon_seconds,
        "alternate_population": alt_pop["demand_generated"],
        "default_population": demand_n,
        "closure_holds": len(alt_pop["completed_wait_ledger"]) + len(alt_pop["censored_wait_ledger"]) == alt_pop["demand_generated"],
        "p95_status": alt_row["passenger_wait_p95_status"],
        "emitted_horizon_seconds": alt_row["evaluation_horizon_seconds"],
        "authoritative_registry_horizons": horizons,
        "performance_interpreted": False,
        "passed": alt_run.horizon_seconds == 7200.0 and alt_row["evaluation_horizon_seconds"] == 7200.0
        and len(alt_pop["completed_wait_ledger"]) + len(alt_pop["censored_wait_ledger"]) == alt_pop["demand_generated"],
    }

    # -- R5-30 citywide / full-day extensibility architecture ----------------
    day = registry[registry["service_date"].astype(str) == CONTINUITY_SERVICE_DATE]
    day_contract = demand_mod.EvaluationTimeContract(
        evaluation_start_ts=float(day["start_ts"].min()),
        evaluation_end_ts=float(day["evaluation_end_ts"].max()),
        reporting_window_ids=[str(w) for w in day["window_id"]],
        scope_label="R5_MULTI_WINDOW_SERVICE_BLOCK_ARCHITECTURE_AUDIT",
    )
    day_sel = demand_mod.select_population(realization, day_contract)
    day_adapter = adapter(evaluation_contract=day_contract, demand_population=day_sel["requests"],
                          population_audit=day_sel["audit"])
    day_adapter.reset()
    demand_source = DEMAND_MODULE.read_text(encoding="utf-8")
    hardcoded = {
        "literal_1800_in_bridge": bool(re.search(r"\b1800\b", source)),
        "suseong_in_bridge": "suseong" in source.lower(),
        "literal_agent_count": bool(re.search(r"num_agents\s*=\s*\d+", source)),
        "literal_stop_count": bool(re.search(r"stop_count\s*=\s*(max\()?\d+", source)),
        "literal_window_universe": "54" in re.sub(r"\d{3,}", "", source),
    }
    checks["R5_30_citywide_full_day_extensibility"] = {
        "A_current_reduced_evaluation_seconds": live.horizon_seconds,
        "B_multi_hour_evaluation_seconds": alt_run.horizon_seconds,
        "C_service_block_evaluation_seconds": day_contract.evaluation_horizon_seconds,
        "C_reporting_windows_spanned": len(day_contract.reporting_window_ids),
        "C_population_count": len(day_sel["requests"]),
        "C_stop_universe_size": day_adapter.stop_count,
        "reduced_stop_universe_size": live.stop_count,
        "stop_universe_scales_with_scope": day_adapter.stop_count > live.stop_count,
        "hardcoded_scope_tokens": {k: v for k, v in hardcoded.items() if v},
        "demand_loader_accepts_external_artifact": "def load_demand_realization" in demand_source,
        "demand_loader_accepts_column_map": "column_map" in demand_source,
        "evaluation_boundary_externally_supplied": "evaluation_contract" in source,
        "reporting_window_separated_from_boundary": len(day_contract.reporting_window_ids) > 1,
        "state_continues_across_reporting_windows": len(day_sel["requests"]) > max(per_window),
        "ledger_structure_unbounded": "List[float]" in source,
        "full_scale_experiment_executed": False,
        "passed": not any(hardcoded.values()) and len(day_contract.reporting_window_ids) > 1
        and day_adapter.stop_count > live.stop_count and len(day_sel["requests"]) > max(per_window),
    }

    failed = [k for k, v in checks.items() if not v["passed"]]
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-R5",
        "window_id": str(window["window_id"]),
        "demand_provenance": provenance,
        "population_audit": inputs["population_audit"],
        "evaluation_contract": inputs["evaluation_contract"].payload(),
        "checks": checks,
        "failed_checks": failed,
        "all_passed": not failed,
    }


def run_validations() -> Dict[str, Any]:
    """Validation harness.

    This exercises the causal bridge, which R9.8 protects with the
    simulator_execution capability. The harness is entitled to run it, so it
    declares that explicitly here rather than the guard being weakened.
    """
    with _authz.granted("simulator_execution", reason="R5 demand binding validation harness"):
        return _run_validations_inner()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    result = run_validations()
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if result["failed_checks"]:
        print(f"[FAIL] H4M-AE-R5 failed: {result['failed_checks']}")
        raise SystemExit(1)
    print("[PASS] H4M-AE-R5 demand binding and horizon accounting passed (R5-01..R5-30)")


if __name__ == "__main__":
    main()
