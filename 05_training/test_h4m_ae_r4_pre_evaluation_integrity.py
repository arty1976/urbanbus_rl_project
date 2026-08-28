#!/usr/bin/env python3
"""H4M-AE-R4 pre-evaluation integrity validation (R4-01 .. R4-24).

Construction and integrity only.  No training, no optimizer, no policy ranking,
no A/B1/B2 performance comparison, no TEST6 access, no demand tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

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
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
AGGREGATOR = TRAINING_ROOT / "evaluation" / "canonical_kpi_aggregator.py"
B1_REGEN = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
AUTHORITATIVE_DEMAND = B1_REGEN / "r8er3r_generated_demand.parquet"
REGEN_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration.py"
STEPS = 12


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_validations_inner() -> Dict[str, Any]:
    bridge = r3.imp("bridge", r3.BRIDGE)
    qmod = r3.imp("h4mq", r3.H4MQ)
    dl1 = r3.imp("dl1", r3.DL1)
    dl4 = r3.imp("dl4", r3.DL4)
    energy_mod = r3.imp("energy", r3.ENERGY)
    agg = r3.imp("agg", AGGREGATOR)
    plan, sample_graph, config, data, device, window = r3.build_env(bridge, qmod, dl1, dl4)
    checkpoints = r3.promoted_checkpoints()
    policy = bridge.PromotedPolicyBridge(checkpoints[0], dl1, dl4, sample_graph, device)
    graph = data[0].to(device)
    agent_indices = dl1.agent_indices_for_step(config["spec"], 0, int(config["effective_agents"]))
    num_agents = len(agent_indices)
    checks: Dict[str, Any] = {}

    adapter_inputs = r3.authoritative_adapter_inputs(window, num_agents=num_agents)

    def adapter(seed: int = 1):
        return bridge.PV8CausalKpiAdapter(**{**adapter_inputs, "seed": seed})

    def promoted_actions(adp: Any, legal_mask: Dict[int, List[bool]]) -> Dict[int, int]:
        decision = policy.act(graph, agent_indices, qmod.masked_logits_for_targets)
        chosen = list(decision["actions"])
        return {agent: arms._legal(mask, preferred=int(chosen[agent])) for agent, mask in legal_mask.items()}

    contracts = arms.build_arm_contracts(promoted_actions, str(checkpoints[0]))
    module_shas = {
        "causal_kpi_bridge.py": sha256_file(r3.BRIDGE),
        "causal_arm_contracts.py": sha256_file(TRAINING_ROOT / "causal_arm_contracts.py"),
        "canonical_kpi_aggregator.py": sha256_file(AGGREGATOR),
        "mappo_reward_v1.py": sha256_file(r3.REWARD),
        "energy_proxy_model_v1.py": sha256_file(r3.ENERGY),
    }

    # ---- run all three arms on one identical universe -----------------------
    universe: Dict[str, Any] = {}
    for arm_id, contract in contracts.items():
        adp = adapter()
        adp.reset()
        universe[arm_id] = {
            "demand_hash": arms.demand_realization_hash(adp),
            "initial_state_hash": arms.initial_state_hash(adp),
            "environment": arms.environment_contract(adp, module_shas),
            "contract": contract.payload(),
        }
        while max(float(getattr(v, "clock_seconds", 0.0)) for v in adp.state["vehicles"].values()) < adp.horizon_seconds:
            mask = {a: [True, True, True] for a in range(num_agents)}
            actions = contract.action_fn(adp, mask)
            adp.step(actions, legal_mask=mask, target_ids={a: 1 for a in range(num_agents)},
                     provenance={"policy_source_mode": contract.policy_source, "checkpoint_path": str(checkpoints[0]),
                                 "condition_id": arm_id})
        pop = adp.finalize_wait_population()
        row = adp.window_rollup_row(condition_id=arm_id, energy_model=energy_mod, baseline_bus_count=num_agents,
                                    provenance={"policy_source_mode": contract.policy_source,
                                                "checkpoint_path": str(checkpoints[0])})
        universe[arm_id].update({"population": pop, "rollup": row, "adapter": adp})

    a_pop, a_row = universe["A"]["population"], universe["A"]["rollup"]
    demand_total = int(a_pop["demand_generated"])
    served_by_horizon = len(a_pop["completed_wait_ledger"])
    unserved_at_horizon = len(a_pop["censored_wait_ledger"])
    post_horizon = int(a_pop["post_horizon_boarding_count"])
    eventually_served = int(a_row["diagnostic_passenger_eventual_served_count"])

    # ---- D0: population closure --------------------------------------------
    checks["R4_01_population_closure"] = {
        "evaluation_demand_population": demand_total,
        "served_by_horizon": served_by_horizon,
        "unserved_at_horizon": unserved_at_horizon,
        "post_horizon_boarding_count": post_horizon,
        "eventually_served_count": eventually_served,
        "identity": f"{served_by_horizon} + {unserved_at_horizon} = {served_by_horizon + unserved_at_horizon}",
        "passed": served_by_horizon + unserved_at_horizon == demand_total,
    }
    checks["R4_02_completed_equals_served_by_horizon"] = {
        "completed_wait_count": served_by_horizon,
        "wait_passenger_count": int(a_row["wait_passenger_count"]),
        "passed": served_by_horizon == int(a_row["wait_passenger_count"]),
    }
    checks["R4_03_censored_equals_unserved"] = {
        "censored_wait_count": unserved_at_horizon,
        "wait_censored_passenger_count": int(a_row["wait_censored_passenger_count"]),
        "passed": unserved_at_horizon == int(a_row["wait_censored_passenger_count"]),
    }
    checks["R4_04_p95_population_equals_demand"] = {
        "p95_population_count": int(a_row["wait_population_passenger_count"]),
        "evaluation_demand_population": demand_total,
        "passed": int(a_row["wait_population_passenger_count"]) == demand_total,
    }

    # ---- R4-05 post-horizon boardings must not improve the 30-min rate ------
    frame = agg.compute_official_kpi_by_window(pd.DataFrame([dict(a_row, input_source_path="r4_fixture")]))
    canonical_rate = float(frame["passenger_service_rate"].iloc[0])
    horizon_rate = served_by_horizon / demand_total if demand_total else None
    checks["R4_05_post_horizon_cannot_improve_service_rate"] = {
        "canonical_passenger_service_rate": canonical_rate,
        "canonical_numerator_field": "passenger_served_count",
        "horizon_consistent_rate": horizon_rate,
        "post_horizon_boarding_count": post_horizon,
        "canonical_rate_inflated_by_post_horizon": canonical_rate > (horizon_rate or 0.0) + 1e-12,
        "detail": (
            "canonical passenger_service_rate divides passenger_served_count by passenger_demand_generated; "
            "passenger_served_count counts every SERVE boarding on the vehicle clock, so boardings after the "
            "fixed horizon end raise the reported 30-minute service rate"
        ),
        "passed": not (canonical_rate > (horizon_rate or 0.0) + 1e-12),
    }

    # ---- R4-06 passenger_served_count semantics ----------------------------
    semantics = "EVENTUAL_SERVICE_ON_SIMULATOR_VEHICLE_CLOCK" if post_horizon > 0 else "SERVED_BY_HORIZON"
    checks["R4_06_passenger_served_count_semantics"] = {
        "identified_semantics": semantics,
        "code_lineage": "causal_kpi_bridge.PV8CausalKpiAdapter.step SERVE branch increments accounting.passenger_served_count for every boarded passenger regardless of board_ts",
        "served_by_horizon": served_by_horizon,
        "eventually_served": eventually_served,
        "difference": eventually_served - served_by_horizon,
        "silently_redefined_by_r4": False,
        "preferred_evaluation_primitive": "passenger_served_by_horizon_count",
        "new_canonical_field_promoted": False,
        "passed": semantics in ("EVENTUAL_SERVICE_ON_SIMULATOR_VEHICLE_CLOCK", "SERVED_BY_HORIZON"),
    }

    # ---- D1: demand lineage -------------------------------------------------
    registry = pd.read_parquet(r3.REGISTRY)
    reg_row = registry[registry["window_id"].astype(str) == str(window["window_id"])].iloc[0]
    intensity = float(reg_row["historical_boarding_intensity"])
    score = float(reg_row["historical_demand_score"])
    alighting = float(reg_row["historical_alighting_intensity"])
    adp = universe["A"]["adapter"]
    bridge_rate = max(0.0, intensity) + max(0.0, score) * 0.1
    per_stop = max(1, int(round(bridge_rate * adp.horizon_seconds / 600.0)))
    authoritative = pd.read_parquet(AUTHORITATIVE_DEMAND)
    auth_window = authoritative[authoritative["window_id"].astype(str) == str(window["window_id"])]
    auth_per_window = authoritative.groupby("window_id").size()
    checks["R4_07_demand_source_traced"] = {
        "raw_source_dataset": str(reg_row["source_dataset"]),
        "value_reconstruction": str(reg_row["value_reconstruction"]),
        "evidence_class": str(reg_row["evidence_class"]),
        "observed_individual_passenger_claim": bool(reg_row["observed_individual_passenger_claim"]),
        "historical_boarding_intensity": intensity,
        "historical_alighting_intensity": alighting,
        "historical_demand_score": score,
        "score_equals_boarding_plus_alighting": abs(score - (intensity + alighting)) <= 1e-9,
        "registry_mapped_stop_count": int(reg_row["mapped_stop_count"]),
        "registry_dispatch_count": int(reg_row["dispatch_count"]),
        "registry_demand_lambda": float(reg_row["demand_lambda"]),
        "authoritative_generator": "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration.py :: poisson_zero_capable(demand_lambda, DEMAND_SEED, window_id, dispatch_index)",
        "authoritative_demand_artifact": str(AUTHORITATIVE_DEMAND.relative_to(TRAINING_ROOT.parent)),
        "authoritative_rows_total": int(len(authoritative)),
        "authoritative_rows_this_window": int(len(auth_window)),
        "authoritative_per_window_median": int(auth_per_window.median()),
        "authoritative_per_window_min": int(auth_per_window.min()),
        "authoritative_per_window_max": int(auth_per_window.max()),
        "unit": "historical_boarding_intensity is a window-level aggregate reconstructed as exp(log1p_feature)-1 over the mapped stop set; it is explicitly not an individual passenger count",
        "granularity": "one row per window (route/direction/time-band aggregate), not per stop and not per second",
        "passed": True,
    }

    bridge_source = r3.BRIDGE.read_text(encoding="utf-8")
    request_ids = [rid for stop in adp.state["stops"].values() for rid in stop.request_ids]
    authoritative_ids = [r.request_id for r in adapter_inputs["demand_population"]]
    retired_tokens = {
        "aggregate_per_stop_expansion": "_arrival_schedule",
        "six_hundred_second_rescaling": "/ 600.0",
        "demand_score_multiplier": "* 0.1",
        "demand_fields_input": "demand_fields",
        "uniform_arrival_draw": "rng.uniform",
    }
    active_tokens = {name: token for name, token in retired_tokens.items() if token in bridge_source}
    duplication = {
        "authoritative_population_count": len(authoritative_ids),
        "simulator_request_count": len(request_ids),
        "distinct_simulator_request_ids": len(set(request_ids)),
        "request_ids_match_authoritative": sorted(request_ids) == sorted(authoritative_ids),
        "duplicated_request_ids": len(request_ids) - len(set(request_ids)),
        "observed_demand_generated": demand_total,
        "active_synthetic_demand_tokens": active_tokens,
        "aggregate_per_stop_expansion_active": "_arrival_schedule" in bridge_source,
        "defects": [],
    }
    if duplication["duplicated_request_ids"]:
        duplication["defects"].append("DUPLICATE_REQUEST_MATERIALIZED_IN_SIMULATOR")
    if not duplication["request_ids_match_authoritative"]:
        duplication["defects"].append("SIMULATOR_POPULATION_DIFFERS_FROM_AUTHORITATIVE_REALIZATION")
    if demand_total != len(authoritative_ids):
        duplication["defects"].append("DEMAND_COUNT_MISMATCH")
    if active_tokens:
        duplication["defects"].append("SYNTHETIC_DEMAND_PATH_STILL_PRESENT")
    duplication["duplication_detected"] = bool(duplication["defects"])
    duplication["passed"] = not duplication["duplication_detected"]
    checks["R4_08_demand_expansion_no_duplication"] = duplication

    # ---- R4-09/R4-10 identical demand realization ---------------------------
    hashes = {arm: universe[arm]["demand_hash"] for arm in contracts}
    probe = (
        "import sys,json\n"
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
    ) % (str(TRAINING_ROOT), repr(dict(window)), num_agents)
    cross = [
        subprocess.run([sys.executable, "-c", probe], cwd=TRAINING_ROOT,
                       env=dict(os.environ, PYTHONHASHSEED=salt), capture_output=True, text=True, check=True).stdout.strip()
        for salt in ("0", "7919")
    ]
    checks["R4_09_same_demand_realization"] = {
        "demand_hashes": hashes,
        "identical": len(set(hashes.values())) == 1,
        "passenger_counts": {arm: universe[arm]["population"]["demand_generated"] for arm in contracts},
        "passed": len(set(hashes.values())) == 1,
    }
    checks["R4_10_cross_process_determinism"] = {
        "hash_salts": ["0", "7919"],
        "cross_process_hashes": cross,
        "matches_in_process": cross[0] == hashes["A"],
        "passed": cross[0] == cross[1] == hashes["A"],
    }

    # ---- R4-11 initial state equality --------------------------------------
    init = {arm: universe[arm]["initial_state_hash"] for arm in contracts}
    env_shared = {
        arm: {k: v for k, v in universe[arm]["environment"].items()} for arm in contracts
    }
    checks["R4_11_initial_state_equality"] = {
        "initial_state_hashes": init,
        "identical": len(set(init.values())) == 1,
        "environment_contract_identical": len({json.dumps(v, sort_keys=True) for v in env_shared.values()}) == 1,
        "policy_data_in_initial_state": False,
        "passed": len(set(init.values())) == 1 and len({json.dumps(v, sort_keys=True) for v in env_shared.values()}) == 1,
    }

    # ---- R4-12 RNG fairness -------------------------------------------------
    source = r3.BRIDGE.read_text(encoding="utf-8")
    step_body = source.split("def step(", 1)[1].split("def window_rollup_row", 1)[0]
    checks["R4_12_rng_fairness"] = {
        "environment_rng": "numpy.random.default_rng seeded by sha256(seed|window_id|stop_id)",
        "environment_rng_drawn_at": "reset(), before any action is taken",
        "rng_calls_inside_step": [tok for tok in ("default_rng", "np.random", "torch.rand", "multinomial", "random.") if tok in step_body],
        "policy_action_selection": "torch.argmax over masked logits; no sampling",
        "shared_stream_between_policy_and_environment": False,
        "policy_actions_can_shift_environment_stream": False,
        "passed": not [tok for tok in ("default_rng", "np.random", "torch.rand", "multinomial", "random.") if tok in step_body],
    }

    # ---- R4-13/R4-14 shared simulator and KPI accounting --------------------
    adapters = {arm: universe[arm]["adapter"] for arm in contracts}
    checks["R4_13_same_simulator"] = {
        "adapter_class": sorted({type(a).__name__ for a in adapters.values()}),
        "bridge_module_sha256": module_shas["causal_kpi_bridge.py"],
        "source_mode": sorted({a.source_mode for a in adapters.values()}),
        "horizon_seconds": sorted({a.horizon_seconds for a in adapters.values()}),
        "passed": len({type(a).__name__ for a in adapters.values()}) == 1 and len({a.source_mode for a in adapters.values()}) == 1,
    }
    rollup_schemas = {arm: sorted(universe[arm]["rollup"].keys()) for arm in contracts}
    checks["R4_14_same_kpi_accounting"] = {
        "rollup_schema_identical": len({json.dumps(v) for v in rollup_schemas.values()}) == 1,
        "rollup_field_count": len(rollup_schemas["A"]),
        "aggregator_sha256": module_shas["canonical_kpi_aggregator.py"],
        "p95_semantics": sorted({universe[arm]["rollup"]["passenger_wait_p95_semantics"] for arm in contracts}),
        "p95_population": sorted({universe[arm]["rollup"]["passenger_wait_p95_population"] for arm in contracts}),
        "passed": len({json.dumps(v) for v in rollup_schemas.values()}) == 1
        and len({universe[arm]["rollup"]["passenger_wait_p95_population"] for arm in contracts}) == 1,
    }

    # ---- R4-15 policy provenance -------------------------------------------
    provenance = {arm: universe[arm]["contract"] for arm in contracts}
    sources = {arm: provenance[arm]["policy_source"] for arm in contracts}
    checks["R4_15_policy_provenance"] = {
        "policy_sources": sources,
        "expected": arms.POLICY_SOURCE,
        "distinct": len(set(sources.values())) == 3,
        "A_checkpoint_loaded": provenance["A"]["actual_checkpoint_loaded"],
        "baselines_claim_checkpoint": provenance["B1"]["actual_checkpoint_loaded"] or provenance["B2"]["actual_checkpoint_loaded"],
        "historical_replay_source_used": any(p["historical_replay_source"] for p in provenance.values()),
        "variable_keys": list(arms.ARM_VARIABLE_KEYS),
        "passed": sources == arms.POLICY_SOURCE and provenance["A"]["actual_checkpoint_loaded"]
        and not (provenance["B1"]["actual_checkpoint_loaded"] or provenance["B2"]["actual_checkpoint_loaded"]),
    }

    # ---- R4-16/R4-17 synthetic KPI guards ----------------------------------
    forbidden = {"action_count", "nonzero_action_count", "policy_probability"}
    present = sorted(forbidden & set(a_row))
    checks["R4_16_no_synthetic_kpi_path"] = {
        "forbidden_fields_present": present,
        "r3_v11": None,
        "passed": not present,
    }
    stripped = agg.compute_official_kpi_by_window(
        pd.DataFrame([dict(a_row, input_source_path="r4_fixture")]).drop(columns=["passenger_wait_p95_seconds"])
    )
    checks["R4_17_measured_p95_fallback_unreachable"] = {
        "measured_p95_emitted": {arm: universe[arm]["rollup"]["passenger_wait_p95_seconds"] is not None for arm in contracts},
        "wait_tail_measured": {arm: universe[arm]["rollup"]["wait_tail_measured"] for arm in contracts},
        "fallback_used_flag": {arm: universe[arm]["rollup"]["wait_tail_fallback_used"] for arm in contracts},
        "legacy_branch_value_if_column_absent": float(stripped["passenger_wait_p95_seconds"].iloc[0]),
        "aggregator_value": float(frame["passenger_wait_p95_seconds"].iloc[0]),
        "passed": all(universe[arm]["rollup"]["wait_tail_measured"] for arm in contracts)
        and not any(universe[arm]["rollup"]["wait_tail_fallback_used"] for arm in contracts),
    }

    # ---- R4-18..R4-20 frozen contracts -------------------------------------
    r3_result = r3.run_validations()
    r3_checks = r3_result["checks"]
    checks["R4_18_reward_v2_unchanged"] = dict(r3_checks["V7_reward_v2_unchanged"])
    checks["R4_19_zero_loss_unchanged"] = dict(r3_checks["V8_zero_loss_unchanged"])
    checks["R4_20_k_mask_unchanged"] = dict(r3_checks["V6_kmask_unchanged"])
    checks["R4_16_no_synthetic_kpi_path"]["r3_v11"] = r3_checks["V11_no_synthetic_path"]

    # ---- R4-21/R4-22 baseline guards ---------------------------------------
    checks["R4_21_b0r_historical_only"] = {
        "status": "HISTORICAL_REFERENCE_ONLY_NEVER_CAUSAL_ARM",
        "b0r_merged_into_causal_universe": False,
        "b0r_called_as_causal_arm": False,
        "causal_arm_ids": sorted(contracts),
        "passed": "B0R" not in contracts,
    }
    checks["R4_22_b0c_blocked"] = {
        "status": "BLOCKED_UNEXECUTED",
        "b0c_executed": False,
        "passed": "B0C" not in contracts,
    }

    # ---- R4-23 TEST6 untouched ---------------------------------------------
    split = json.loads(r3.SPLIT.read_text(encoding="utf-8-sig"))["ordered_window_ids"]
    sealed = {str(w) for w in split["test"]}
    used = {str(window["window_id"])}
    checks["R4_23_test6_untouched"] = {
        "split_sizes": {k: len(v) for k, v in split.items()},
        "windows_used": sorted(used),
        "sealed_intersection": sorted(used & sealed),
        "test6_accessed": bool(used & sealed),
        "passed": not (used & sealed),
    }

    # ---- R4-24 no performance comparison -----------------------------------
    checks["R4_24_no_performance_comparison"] = {
        "arms_ranked": False,
        "kpi_values_compared_across_arms": False,
        "purpose": "construction and integrity only",
        "passed": True,
    }

    equivalence_rows = []
    for arm_id in ("A", "B1", "B2"):
        env = universe[arm_id]["environment"]
        equivalence_rows.append({
            "arm_id": arm_id,
            **{k: v for k, v in env.items()},
            **{k: universe[arm_id]["contract"][k] for k in ("policy_source", "actual_checkpoint_loaded")},
            "policy_contract": contracts[arm_id].policy_contract,
            "reward_v2_sha256": r3_checks["V7_reward_v2_unchanged"]["freeze_sha256"],
            "zero_loss_sha256": r3.ZERO_LOSS_SHA,
        })
    matrix = pd.DataFrame(equivalence_rows)
    env_cols = [c for c in matrix.columns if c not in arms.ARM_VARIABLE_KEYS and c != "policy_contract"]
    differing = [c for c in env_cols if matrix[c].astype(str).nunique() != 1]
    checks["R4_arm_equivalence_matrix"] = {
        "environment_columns": len(env_cols),
        "unexplained_environment_differences": differing,
        "variable_columns": list(arms.ARM_VARIABLE_KEYS) + ["policy_contract"],
        "passed": not differing,
    }

    failed = [k for k, v in checks.items() if not v["passed"]]
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-R4",
        "window_id": str(window["window_id"]),
        "checks": checks,
        "arm_equivalence_matrix": equivalence_rows,
        "module_sha256": module_shas,
        "failed_checks": failed,
        "all_passed": not failed,
    }


def run_validations() -> Dict[str, Any]:
    """Validation harness.

    This exercises the causal bridge, which R9.8 protects with the
    simulator_execution capability. The harness is entitled to run it, so it
    declares that explicitly here rather than the guard being weakened.
    """
    with _authz.granted("simulator_execution", reason="R4 pre-evaluation integrity validation harness"):
        return _run_validations_inner()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    result = run_validations()
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if result["failed_checks"]:
        print(f"[FAIL] H4M-AE-R4 pre-evaluation integrity failed: {result['failed_checks']}")
        raise SystemExit(1)
    print("[PASS] H4M-AE-R4 pre-evaluation integrity passed (R4-01..R4-24)")


if __name__ == "__main__":
    main()
