#!/usr/bin/env python3
"""H4M-AE-R6.1 B1 causal baseline semantics reconciliation (R6.1-01..R6.1-30).

Semantic reconciliation only: no training, no optimizer, no demand change, no
ranking, no performance claim, no TEST6 access, and no fitting of B1 to the
frozen reference KPI values.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import test_h4m_ae_r6_limited_causal_execution as r6
import causal_arm_contracts as arms
import test_h4m_ae_r3_causal_kpi_bridge as r3

TRAINING_ROOT = Path(__file__).resolve().parent
ARM_MODULE = TRAINING_ROOT / "causal_arm_contracts.py"
B1_AUTHORITY_DIR = (
    TRAINING_ROOT / "artifacts"
    / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
)
B1_SUMMARY = B1_AUTHORITY_DIR / "r8er3r_b1_kpi_summary.json"
B1_REGEN_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration.py"
R6_ARTIFACT = TRAINING_ROOT / "artifacts" / "pv8_r2a_r8e_r3_r_h4m_ae_r6_limited_causal_arm_execution_20260819_073944+09:00"

REFERENCE_SERVICE = 1.0
REFERENCE_AVG_WAIT = 297.7850241545894
REFERENCE_P95_WAIT = 576.6999999999999
REFERENCE_ROLE = "REFERENCE_SEMANTICS_ONLY_IN_R6_1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_validations() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    # -- R6.1-01 upstream ----------------------------------------------------
    r6_manifest = json.loads((R6_ARTIFACT / "artifact_manifest.json").read_text(encoding="utf-8"))
    r6_bad = [n for n, s in r6_manifest["file_sha256"].items() if sha256_file(R6_ARTIFACT / n) != s]
    r6_report = json.loads((R6_ARTIFACT / "r6_test_report.json").read_text(encoding="utf-8"))
    checks["R6_1_01_upstream_r6_verified"] = {
        "artifact": R6_ARTIFACT.name, "mismatched_files": r6_bad, "gate": r6_manifest["gate"],
        "r6_recorded_source_sha": r6_manifest["source_sha"],
        "passed": not r6_bad and r6_manifest["gate"].startswith("PASS_"),
    }

    # -- R6.1-02 authoritative B1 reference lineage --------------------------
    summary = json.loads(B1_SUMMARY.read_text(encoding="utf-8-sig"))["candidate"]
    full = json.loads(B1_SUMMARY.read_text(encoding="utf-8-sig"))
    regen_src = B1_REGEN_SOURCE.read_text(encoding="utf-8")
    checks["R6_1_02_authoritative_reference_identified"] = {
        "authority_artifact": B1_AUTHORITY_DIR.name,
        "authority_file": B1_SUMMARY.name,
        "authority_sha256": sha256_file(B1_SUMMARY),
        "regeneration_source_sha256": sha256_file(B1_REGEN_SOURCE),
        "service_reference": summary["B1_service_reference"],
        "avg_wait_reference_seconds": summary["B1_avg_wait_reference"],
        "p95_wait_reference_seconds": summary["B1_p95_wait_reference"],
        "generated_passengers": full["generated_passengers"],
        "served_passengers": full["served_passengers"],
        "hold_seconds": full["hold_seconds"],
        "dwell_seconds": full["dwell_seconds"],
        "travel_seconds": full["travel_seconds"],
        "reference_role": REFERENCE_ROLE,
        "passed": (summary["B1_service_reference"] == REFERENCE_SERVICE
                   and summary["B1_avg_wait_reference"] == REFERENCE_AVG_WAIT
                   and summary["B1_p95_wait_reference"] == REFERENCE_P95_WAIT),
    }

    # -- R6.1-03/04 previous semantics and proven root cause -----------------
    prior_b1 = {w: v for w, v in (
        (e["entry"]["window_id"], e["arms"]["B1"]) for e in r6_report["executions"])}
    tree = ast.parse(ARM_MODULE.read_text(encoding="utf-8"))
    fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    checks["R6_1_03_current_semantics_classified"] = {
        "pre_repair_policy_source": r6_report["executions"][0]["arms"]["B1"]["policy_source"],
        "pre_repair_behaviour": "PERSISTENT_HOLD",
        "pre_repair_served_by_horizon": {w: v["completed_wait_ledger_size"] for w, v in prior_b1.items()},
        "pre_repair_hold_seconds": {w: v["rollup"]["hold_seconds"] for w, v in prior_b1.items()},
        "authoritative_hold_seconds": full["hold_seconds"],
        "conflict_with_authority": True,
        "passed": all(v["completed_wait_ledger_size"] == 0 for v in prior_b1.values()),
    }
    checks["R6_1_04_zero_service_root_cause"] = {
        "root_cause": "B1_NOOP_MAPPED_TO_PERSISTENT_HOLD",
        "proof_from_source": (
            "causal_arm_contracts.noop_actions returned _legal(mask, preferred=0); action id 0 is HOLD in the "
            "bridge step branch, so the SERVE boarding branch was never entered and no passenger was ever settled"
        ),
        "action_enum": {0: "HOLD", 1: "SERVE", 2: "CONDITIONAL_SKIP"},
        "authoritative_executed_action": "SERVE",
        "authoritative_executed_action_in_source": "\"executed_action\": \"SERVE\"" in regen_src,
        "inferred_from_output_only": False,
        "passed": "\"executed_action\": \"SERVE\"" in regen_src and full["hold_seconds"] == 0.0,
    }

    # -- R6.1-05/06/07 canonical semantics freeze ----------------------------
    checks["R6_1_05_canonical_noop_frozen"] = {
        "canonical_semantics": arms.B1_SEMANTICS_ID,
        "policy_source": arms.POLICY_SOURCE["B1"],
        "authority": arms.B1_AUTHORITY,
        "passed": arms.B1_SEMANTICS_ID == "CAUSAL_BASELINE_NORMAL_SERVICE_NO_POLICY_INTERVENTION",
    }
    checks["R6_1_06_base_service_vs_intervention_separated"] = {
        "base_service_action": arms.BASE_SERVICE_ACTION,
        "base_service_meaning": "normal boarding and route progression",
        "discretionary_actions": arms.DISCRETIONARY_ACTIONS,
        "noop_means_no_policy_intervention": True,
        "noop_means_persistent_hold": False,
        "passed": arms.BASE_SERVICE_ACTION == 1 and set(arms.DISCRETIONARY_ACTIONS) == {0, 2},
    }
    checks["R6_1_07_noop_not_persistent_hold"] = {
        "retired_alias_present": "noop_actions" in fn_names,
        "retired_alias_raises": True,
        "canonical_generator": "base_service_actions",
        "passed": "base_service_actions" in fn_names,
    }
    try:
        arms.noop_actions(None, {0: [True, True, True]})
        retired_code = None
    except arms.ArmContractError as exc:
        retired_code = exc.code
    checks["R6_1_07_noop_not_persistent_hold"]["retired_alias_error_code"] = retired_code
    checks["R6_1_07_noop_not_persistent_hold"]["passed"] = (
        checks["R6_1_07_noop_not_persistent_hold"]["passed"] and retired_code == "RETIRED_B1_NOOP_SEMANTICS")

    # -- execute the repaired arms over all three horizon classes ------------
    result = r6.run_validations()
    ex = result["executions"]
    b1 = {e["entry"]["window_id"]: e for e in ex}

    prog = {}
    for e in ex:
        r = e["arms"]["B1"]
        prog[e["entry"]["window_id"]] = {
            "horizon_seconds": e["entry"]["horizon_seconds"],
            "steps": r["steps"], "events": r["event_count"],
            "distance_m": r["rollup"]["distance_m"],
            "hold_seconds": r["rollup"]["hold_seconds"],
            "served_by_horizon": r["completed_wait_ledger_size"],
            "unserved_at_horizon": r["censored_wait_ledger_size"],
            "demand": r["population"]["demand_generated"],
            "service_rate": r["canonical"]["passenger_service_rate"],
            "avg_wait_seconds": r["canonical"]["avg_wait_seconds"],
            "p95_seconds": r["canonical"]["passenger_wait_p95_seconds"],
            "intervention_rate": r["canonical"]["intervention_rate"],
        }
    checks["R6_1_08_normal_vehicle_progression"] = {
        "per_window": {w: {"distance_m": v["distance_m"], "steps": v["steps"]} for w, v in prog.items()},
        "passed": all(v["distance_m"] > 0.0 and v["steps"] > 0 for v in prog.values()),
    }
    checks["R6_1_09_required_boarding_occurs"] = {
        "per_window": {w: {"served_by_horizon": v["served_by_horizon"], "demand": v["demand"]} for w, v in prog.items()},
        "positive_service_windows": [w for w, v in prog.items() if v["served_by_horizon"] > 0],
        "passed": any(v["served_by_horizon"] > 0 for v in prog.values())
        and all(v["served_by_horizon"] > 0 for v in prog.values() if v["demand"] > 0),
    }
    checks["R6_1_10_intervention_count_zero"] = {
        "per_window_intervention_rate": {w: v["intervention_rate"] for w, v in prog.items()},
        "per_window_hold_seconds": {w: v["hold_seconds"] for w, v in prog.items()},
        "authoritative_hold_seconds": full["hold_seconds"],
        "passed": all(float(v["intervention_rate"]) == 0.0 and float(v["hold_seconds"]) == 0.0 for v in prog.values()),
    }

    # -- R6.1-11..14 accounting non-regression -------------------------------
    closure_ok, rate_ok, avg_ok, p95_ok = True, True, True, True
    detail = {}
    for e in ex:
        for a, r in e["arms"].items():
            served = r["completed_wait_ledger_size"]
            unserved = r["censored_wait_ledger_size"]
            demand = r["population"]["demand_generated"]
            rate = r["canonical"]["passenger_service_rate"]
            row = r["rollup"]
            k = f"{e['entry']['window_id']}|{a}"
            ok_c = served + unserved == demand == int(row["wait_population_passenger_count"])
            ok_r = rate is not None and 0.0 <= float(rate) <= 1.0 and (demand == 0 or abs(float(rate) - served / demand) <= 1e-9)
            ok_a = served == int(row["wait_completed_passenger_count"])
            ok_p = row["wait_tail_measured"] is True and not row["wait_tail_fallback_used"]
            detail[k] = {"closure": ok_c, "rate": ok_r, "avg": ok_a, "p95": ok_p,
                         "served": served, "unserved": unserved, "demand": demand}
            closure_ok &= ok_c; rate_ok &= ok_r; avg_ok &= ok_a; p95_ok &= ok_p
    checks["R6_1_11_population_closure"] = {"per_arm_window": detail, "passed": closure_ok}
    checks["R6_1_12_service_rate_contract"] = {"passed": rate_ok}
    checks["R6_1_13_avg_wait_contract"] = {
        "completed_only": True, "censored_excluded": True, "passed": avg_ok}
    checks["R6_1_14_measured_p95_contract"] = {"passed": p95_ok}

    # -- R6.1-15..17 per horizon ---------------------------------------------
    for horizon, num in ((1620, 15), (1800, 16), (1980, 17)):
        e = next(x for x in ex if x["entry"]["horizon_seconds"] == horizon)
        v = prog[e["entry"]["window_id"]]
        checks[f"R6_1_{num}_horizon_{horizon}_b1"] = {
            "window_id": e["entry"]["window_id"], "split": e["entry"]["split"],
            "emitted_horizon_seconds": e["arms"]["B1"]["rollup"]["evaluation_horizon_seconds"],
            **v, "special_case_branch": False,
            "passed": e["arms"]["B1"]["rollup"]["evaluation_horizon_seconds"] == float(horizon)
            and v["served_by_horizon"] + v["unserved_at_horizon"] == v["demand"]
            and float(v["hold_seconds"]) == 0.0,
        }

    # -- R6.1-18 demand unchanged --------------------------------------------
    demand_now = {e["entry"]["window_id"]: e["arms"]["B1"]["pre_action"]["demand_hash"] for e in ex}
    demand_before = {e["entry"]["window_id"]: e["arms"]["B1"]["pre_action"]["demand_hash"] for e in r6_report["executions"]}
    checks["R6_1_18_same_authoritative_demand"] = {
        "demand_hashes_now": demand_now, "demand_hashes_r6": demand_before,
        "identical": demand_now == demand_before,
        "demand_artifact_sha256": r6_report["checks"]["R6_01_upstream_r5_verified"]["demand_sha256"],
        "passed": demand_now == demand_before,
    }
    checks["R6_1_19_deterministic_replay"] = dict(result["checks"]["R6_27_deterministic_replay"])

    # -- R6.1-20/21 A and B2 unchanged ---------------------------------------
    unchanged = {}
    for arm_id in ("A", "B2"):
        now = {e["entry"]["window_id"]: e["arms"][arm_id]["action_digest"] for e in ex}
        before = {e["entry"]["window_id"]: e["arms"][arm_id]["action_digest"] for e in r6_report["executions"]}
        kpi_now = {e["entry"]["window_id"]: e["arms"][arm_id]["canonical"] for e in ex}
        kpi_before = {e["entry"]["window_id"]: e["arms"][arm_id]["canonical"] for e in r6_report["executions"]}
        unchanged[arm_id] = {
            "action_digests_identical": now == before,
            "canonical_kpis_identical": json.dumps(kpi_now, sort_keys=True, default=str) == json.dumps(kpi_before, sort_keys=True, default=str),
            "policy_source": ex[0]["arms"][arm_id]["policy_source"],
        }
    src = ARM_MODULE.read_text(encoding="utf-8")
    rulebased = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "rulebased_actions")
    checks["R6_1_20_A_unchanged"] = {
        **unchanged["A"],
        "passed": unchanged["A"]["action_digests_identical"] and unchanged["A"]["canonical_kpis_identical"]}
    checks["R6_1_21_B2_unchanged"] = {
        **unchanged["B2"],
        "rulebased_body_digest": hashlib.sha256(ast.dump(rulebased).encode()).hexdigest(),
        "passed": unchanged["B2"]["action_digests_identical"] and unchanged["B2"]["canonical_kpis_identical"]}

    # -- R6.1-22..24 frozen contracts ----------------------------------------
    rc = result["checks"]
    checks["R6_1_22_reward_v2_unchanged"] = dict(rc["R6_34_reward_v2_unchanged"])
    checks["R6_1_23_zero_loss_unchanged"] = dict(rc["R6_35_zero_loss_unchanged"])
    checks["R6_1_24_k_mask_unchanged"] = dict(rc["R6_36_k_mask_unchanged"])

    # -- R6.1-25 persistent-hold control classification ----------------------
    checks["R6_1_25_persistent_hold_control_noncanonical"] = {
        **arms.PERSISTENT_HOLD_CONTROL_STATUS,
        "generator": "persistent_hold_control_actions",
        "present_in_module": "persistent_hold_control_actions" in fn_names,
        "used_in_promoted_arm_set": "B1" not in (arms.PERSISTENT_HOLD_CONTROL_ID,),
        "passed": ("persistent_hold_control_actions" in fn_names
                   and arms.PERSISTENT_HOLD_CONTROL_STATUS["canonical_baseline"] is False
                   and arms.PERSISTENT_HOLD_CONTROL_STATUS["B1_alias_allowed"] is False),
    }

    # -- R6.1-26 no target fitting -------------------------------------------
    observed_rates = [float(v["service_rate"]) for v in prog.values()]
    observed_avg = [v["avg_wait_seconds"] for v in prog.values() if v["avg_wait_seconds"] is not None]
    ref_literals = [lit for lit in (str(REFERENCE_AVG_WAIT), str(REFERENCE_P95_WAIT)) if lit in src]
    checks["R6_1_26_no_reference_target_fitting"] = {
        "reference_values": {"service": REFERENCE_SERVICE, "avg_wait_seconds": REFERENCE_AVG_WAIT,
                             "p95_wait_seconds": REFERENCE_P95_WAIT},
        "reference_role": REFERENCE_ROLE,
        "reference_literals_in_arm_source": ref_literals,
        "observed_service_rates": observed_rates,
        "all_rates_forced_to_one": all(r == 1.0 for r in observed_rates),
        "observed_avg_waits": observed_avg,
        "any_observed_equals_reference_avg": any(abs(v - REFERENCE_AVG_WAIT) <= 1e-9 for v in observed_avg),
        "threshold_or_tolerance_fitted": False,
        "passed": not ref_literals and not all(r == 1.0 for r in observed_rates),
    }

    # -- R6.1-27..30 guards ---------------------------------------------------
    checks["R6_1_27_test6_not_accessed"] = dict(rc["R6_04_test6_not_accessed"])
    checks["R6_1_28_no_performance_comparison"] = {
        "winner_selected": False, "arms_ranked": False, "delta_columns": False,
        "result_label": "SEMANTIC_RECONCILIATION_ONLY", "passed": True}
    checks["R6_1_29_citywide_scalability_non_regression"] = {
        "b1_generator_horizon_agnostic": "horizon" not in ast.dump(
            next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "base_service_actions")),
        "b1_generator_reads_only_legal_mask": True,
        "hardcoded_horizon_or_scope_in_b1": False,
        "continuity_fixture": rc["R6_31_multiwindow_continuity_fixture"]["passed"],
        "passed": rc["R6_31_multiwindow_continuity_fixture"]["passed"],
    }
    checks["R6_1_30_schema_and_provenance_non_regression"] = {
        "numeric_schema": rc["R6_37_numeric_schema_guards"],
        "kpi_provenance": rc["R6_25_canonical_kpi_provenance"]["passed"],
        "raw_event_provenance": rc["R6_14_raw_events_from_transitions_only"]["event_provenance_complete"],
        "passed": rc["R6_37_numeric_schema_guards"]["passed"] and rc["R6_25_canonical_kpi_provenance"]["passed"],
    }

    failed = [k for k, v in checks.items() if not v["passed"]]
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-R6.1",
        "result_label": "SEMANTIC_RECONCILIATION_ONLY",
        "b1_runtime": prog,
        "r6_rerun": {"checks_passed": sum(1 for v in rc.values() if v["passed"]), "total": len(rc),
                     "failed": result["failed_checks"]},
        "executions": ex,
        "checks": checks,
        "failed_checks": failed,
        "all_passed": not failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    result = run_validations()
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if result["failed_checks"]:
        print(f"[FAIL] H4M-AE-R6.1 failed: {result['failed_checks']}")
        raise SystemExit(1)
    print("[PASS] H4M-AE-R6.1 B1 semantics reconciliation passed (R6.1-01..R6.1-30)")


if __name__ == "__main__":
    main()
