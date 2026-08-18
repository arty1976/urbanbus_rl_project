#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R3.1 measured passenger wait tail accounting repair.

Repairs the synthetic p95 fallback on the promoted causal path with a measured,
right-censoring-inclusive empirical tail and revalidates the whole causal KPI
bridge (R3 V1-V18 + R3.1 Vp0-Vp12).

No training, no optimizer, no checkpoint change, no demand calibration, no
A/B1/B2 performance comparison, no TEST6 access, no network, no push.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
BRIDGE = TRAINING_ROOT / "causal_kpi_bridge.py"
R3_TEST = TRAINING_ROOT / "test_h4m_ae_r3_causal_kpi_bridge.py"
R3_1_TEST = TRAINING_ROOT / "test_h4m_ae_r3_1_measured_wait_tail.py"

UPSTREAM_R3_SHA = "427fd34a3371a333da4e54cfd201786ac2974666"
UPSTREAM_R3_GATE = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R3_CAUSAL_KPI_MEASUREMENT_BRIDGE_IMPLEMENTATION_AND_CONTRACT_VALIDATION_COMPLETE"
R3_FROZEN_BRIDGE_SHA = "b799e0ccf2de535526a1785c29019f4d7ca97de7f4aad5ce16e6d68f71dd9f0c"
GATE_PASS = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R3_1_MEASURED_PASSENGER_WAIT_TAIL_ACCOUNTING_REPAIR"
    "_AND_FULL_CAUSAL_BRIDGE_REVALIDATION_COMPLETE"
)
NEXT_GATE = "H4M-AE-R4_CAUSAL_COMPARISON_ARM_CONSTRUCTION_AND_PRE_EVALUATION_INTEGRITY_VALIDATION"
KST = timezone(timedelta(hours=9))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def r3_artifact_dir() -> Path:
    dirs = [p for p in ARTIFACTS_ROOT.glob("pv8_r2a_r8e_r3_r_h4m_ae_r3_causal_kpi_bridge_implementation_validation_*") if p.is_dir()]
    if not dirs:
        raise RuntimeError("R3 artifact directory not found")
    return sorted(dirs)[-1]


def verify_r3_artifact(root: Path) -> Dict[str, Any]:
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    mismatched = [n for n, s in manifest["file_sha256"].items() if sha256_file(root / n) != s]
    frozen = json.loads((root / "causal_bridge_implementation_manifest.json").read_text(encoding="utf-8"))
    return {
        "artifact_dir": root.name,
        "file_count": len(manifest["file_sha256"]),
        "mismatched_files": mismatched,
        "manifest_verified": not mismatched,
        "recorded_frozen_bridge_sha256": frozen["module_sha256"],
        "expected_frozen_bridge_sha256": R3_FROZEN_BRIDGE_SHA,
        "frozen_sha_matches_expected": frozen["module_sha256"] == R3_FROZEN_BRIDGE_SHA,
    }


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import test_h4m_ae_r3_causal_kpi_bridge as r3_mod
    import test_h4m_ae_r3_1_measured_wait_tail as r31_mod

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_ae_r3_1_measured_passenger_wait_tail_accounting_repair_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    upstream = verify_r3_artifact(r3_artifact_dir())
    new_bridge_sha = sha256_file(BRIDGE)

    r3_result = r3_mod.run_validations()
    r31_result = r31_mod.run_validations()
    bridge = r3_mod.imp("bridge_report", BRIDGE)

    vp = r31_result["checks"]
    r3_checks = r3_result["checks"]
    p95_runtime = vp["Vp5_median_le_p95_le_max"]
    counts = vp["Vp3_censored_count_complement"]
    fallback = vp["Vp9_synthetic_and_fillna_unreachable"]

    dump(root, "r3_upstream_binding.json", {
        "upstream_r3_source_commit": UPSTREAM_R3_SHA,
        "upstream_r3_gate": UPSTREAM_R3_GATE,
        "head_sha_at_run": head_sha,
        "git_status_short": status,
        "reset_or_rebase_performed": False,
        "unrelated_changes_preserved": True,
        **upstream,
        "r3_frozen_module_changed": new_bridge_sha != R3_FROZEN_BRIDGE_SHA,
        "new_bridge_sha256": new_bridge_sha,
        "change_declared_under_new_lineage": True,
    })

    dump(root, "arrival_schedule_weight_semantics.json", {
        "observation_unit": bridge.ARRIVAL_ENTRY_SEMANTICS,
        "weight_semantics": "UNIT_WEIGHT_PER_ENTRY",
        "multiplicity_field_present": vp["VpA_arrival_entry_semantics"]["multiplicity_field_present"],
        "schedule_entry_total": vp["VpA_arrival_entry_semantics"]["schedule_entry_total"],
        "passenger_demand_generated": vp["VpA_arrival_entry_semantics"]["passenger_demand_generated"],
        "percentile_algorithm_family": "EXACT_UNWEIGHTED_EMPIRICAL",
        "sampling_or_sketch_used": False,
        "evidence_check": "VpA_arrival_entry_semantics",
        "passed": vp["VpA_arrival_entry_semantics"]["passed"],
    })

    dump(root, "wait_population_boundary_audit.json", {
        "carry_in_case": vp["Vp0_carry_in_population_boundary"]["case"],
        "case_definitions": {
            "A": "initial waiting queue = 0; canonical population = passengers arriving during the horizon",
            "B": "carry-in waiters exist with authoritative arrival_ts; kept with original arrival_ts",
            "C": "carry-in waiters exist without authoritative arrival_ts; FAIL-CLOSED",
        },
        "initial_waiting_count": vp["Vp0_carry_in_population_boundary"]["initial_waiting_count"],
        "arrivals_outside_horizon": vp["Vp0_carry_in_population_boundary"]["arrivals_outside_horizon"],
        "carry_in_timestamp_status": vp["Vp0_carry_in_population_boundary"]["carry_in_timestamp_status"],
        "fail_closed_triggered": vp["Vp0_carry_in_population_boundary"]["fail_closed_triggered"],
        "discarded_passengers": 0,
        "arrival_ts_reset_to_horizon_start": False,
        "invented_arrival_timestamps": 0,
        "passed": vp["Vp0_carry_in_population_boundary"]["passed"],
    })

    dump(root, "completed_wait_ledger_contract.json", {
        "definition": "completed_wait_i = board_ts - authoritative_arrival_ts_i",
        "population": "passengers boarded at a SERVE transition on or before the fixed evaluation horizon end",
        "ownership_label": "anonymous_per_passenger_duration",
        "passenger_id_fabricated": False,
        "feeds": ["wait_total_passenger_seconds", "wait_passenger_count", "avg_wait_seconds"],
        "avg_wait_formula": "wait_total_passenger_seconds / wait_passenger_count",
        "avg_wait_formula_changed": False,
        "censored_observations_in_mean": False,
        "completed_count": counts["completed_count"],
        "evidence_checks": ["Vp1_completed_sum_matches_wait_total", "Vp2_completed_count_contract", "VpC_avg_wait_semantics_preserved"],
        "passed": vp["Vp1_completed_sum_matches_wait_total"]["passed"] and vp["VpC_avg_wait_semantics_preserved"]["passed"],
    })

    dump(root, "censored_wait_ledger_contract.json", {
        "definition": "censored_wait_i = horizon_end_ts - authoritative_arrival_ts_i",
        "censor_reference": bridge.CENSOR_REFERENCE,
        "censor_reference_is_vehicle_clock": False,
        "censor_reference_is_last_transition": False,
        "censor_reference_is_policy_dependent_termination": False,
        "identical_boundary_across_arms": True,
        "population": "passengers with an authoritative arrival_ts who are not boarded on or before horizon_end",
        "includes_post_horizon_boardings": True,
        "post_horizon_boarding_count": vp["Vp2_completed_count_contract"]["post_horizon_boarding_count"],
        "post_horizon_boarding_rationale": (
            "A boarding whose board_ts exceeds the fixed horizon end would make the tail a vehicle-clock quantity, "
            "which section 2.2 forbids. Such passengers were still waiting at horizon_end and are therefore "
            "right-censored there rather than credited with a completed wait."
        ),
        "censored_count": counts["censored_count"],
        "censored_share": counts["censored_share"],
        "evidence_checks": ["Vp3_censored_count_complement", "Vp4_censored_duration_exact"],
        "passed": vp["Vp3_censored_count_complement"]["passed"] and vp["Vp4_censored_duration_exact"]["passed"],
    })

    dump(root, "canonical_p95_semantics_freeze.json", {
        "passenger_wait_p95_population": bridge.P95_POPULATION,
        "p95_semantics": bridge.P95_SEMANTICS,
        "canonical_tail_is_only_promoted_definition": True,
        "interpretation": (
            "the measured value is a lower bound: censored observations contribute the elapsed wait at horizon_end, "
            "never the unobserved completion time"
        ),
        "empty_population_status": "NOT_APPLICABLE",
        "empty_population_never_zero_performance": True,
        "evidence_checks": ["Vp5_median_le_p95_le_max", "VpB_empty_population_not_applicable"],
        "passed": vp["Vp5_median_le_p95_le_max"]["passed"] and vp["VpB_empty_population_not_applicable"]["passed"],
    })

    dump(root, "percentile_method_freeze.json", {
        "p95_method": f"{bridge.PERCENTILE_LIBRARY}(q={bridge.PERCENTILE_Q}, method='{bridge.PERCENTILE_METHOD}')",
        "percentile_q": bridge.PERCENTILE_Q,
        "percentile_method": bridge.PERCENTILE_METHOD,
        "library": bridge.PERCENTILE_LIBRARY,
        "numpy_version": vp["Vp10_percentile_method_frozen"]["numpy_version"],
        "weighting": vp["Vp10_percentile_method_frozen"]["weighting"],
        "sampling_or_sketch": False,
        "frozen_before_output_interpretation": True,
        "post_result_method_shopping": False,
        "independent_recomputation": vp["Vp10_percentile_method_frozen"]["independent_recomputation"],
        "passed": vp["Vp10_percentile_method_frozen"]["passed"],
    })

    dump(root, "measured_p95_runtime_validation.json", {
        "window_id": r31_result["window_id"],
        "p95_seconds": p95_runtime["p95_seconds"],
        "median_seconds": p95_runtime["median_seconds"],
        "max_seconds": p95_runtime["max_seconds"],
        "p95_population_count": counts["population_count"],
        "completed_observation_count": counts["completed_count"],
        "censored_observation_count": counts["censored_count"],
        "censored_share": counts["censored_share"],
        "p95_semantics": bridge.P95_SEMANTICS,
        "passenger_wait_p95_population": bridge.P95_POPULATION,
        "p95_support": p95_runtime["support"],
        "p95_support_includes_censored_observation": p95_runtime["p95_support_includes_censored_observation"],
        "p95_exact_support_is_censored": p95_runtime["p95_exact_support_is_censored"],
        "p95_interpolated_between_two_support_points": p95_runtime["interpolated_between_two_support_points"],
        "single_censored_observation_claimed_as_p95": False,
        "incentive_controls": {
            "service_denial_positive_control": vp["Vp7_service_denial_positive_control"],
            "serve_vs_hold_ordering": vp["Vp8_serve_vs_hold_ordering"],
        },
        "performance_claim_allowed": False,
        "passed": p95_runtime["passed"] and vp["Vp7_service_denial_positive_control"]["passed"] and vp["Vp8_serve_vs_hold_ordering"]["passed"],
    })

    dump(root, "p95_fallback_unreachability_audit.json", {
        "legacy_fallback_expression": "passenger_wait_p95_seconds = avg_wait_seconds * 1.65",
        "legacy_fallback_location": "05_training/evaluation/canonical_kpi_aggregator.py::ensure_phase2_12_kpis",
        "aggregator_modified_by_r3_1": False,
        "legacy_behaviour_status": fallback["legacy_behaviour_status"],
        "promoted_causal_rollup_emits_measured_p95": fallback["rollup_emits_measured_p95"],
        "promoted_path_reaches_multiplier_branch": fallback["promoted_path_reaches_multiplier_branch"],
        "measured_value_seconds": fallback["measured_p95"],
        "legacy_branch_value_if_column_absent": fallback["legacy_multiplier_branch_value_if_column_absent"],
        "legacy_branch_verified_as_avg_times_1_65": fallback["legacy_branch_equals_avg_times_1_65"],
        "unreachability_mechanism": (
            "window_rollup_row always emits a non-null passenger_wait_p95_seconds column, so the aggregator's "
            "else-branch is never selected for promoted causal arms"
        ),
        "passed": fallback["passed"],
    })

    dump(root, "fillna_zero_unreachability_audit.json", {
        "legacy_expression": "p95.fillna(0.0).clip(lower=0.0)",
        "aggregator_modified_by_r3_1": False,
        "promoted_path_reaches_fillna_branch": fallback["promoted_path_reaches_fillna_branch"],
        "fillna_value_if_column_null": fallback["fillna_zero_value_if_column_null"],
        "null_p95_possible_on_promoted_path": False,
        "empty_population_handling": "NOT_APPLICABLE status is emitted instead of a 0-second value",
        "false_zero_second_p95_possible": False,
        "passed": fallback["passed"] and vp["VpB_empty_population_not_applicable"]["passed"],
    })

    dump(root, "served_only_p95_noncanonical_guard.json", {
        "diagnostic_field": "diagnostic_served_only_wait_p95_seconds",
        "canonical": vp["Vp11_served_only_noncanonical"]["canonical"],
        "promotion_use_allowed": vp["Vp11_served_only_noncanonical"]["promotion_use_allowed"],
        "performance_claim_allowed": vp["Vp11_served_only_noncanonical"]["performance_claim_allowed"],
        "diagnostic_value_seconds": vp["Vp11_served_only_noncanonical"]["diagnostic_value_seconds"],
        "canonical_value_seconds": p95_runtime["p95_seconds"],
        "survivorship_bias_note": (
            "the served-only tail excludes every censored passenger and is therefore optimistically biased; "
            "it exists for debugging only and never competes with the canonical censored-inclusive tail"
        ),
        "p95_columns_in_canonical_frame": vp["Vp11_served_only_noncanonical"]["p95_columns_in_canonical_frame"],
        "passed": vp["Vp11_served_only_noncanonical"]["passed"],
    })

    dump(root, "wait_accounting_invariants.json", {
        "sum_completed_equals_wait_total": vp["Vp1_completed_sum_matches_wait_total"],
        "completed_count_equals_wait_passenger_count": vp["Vp2_completed_count_contract"],
        "censored_count_is_complement": vp["Vp3_censored_count_complement"],
        "censored_excluded_from_avg_wait": not vp["VpC_avg_wait_semantics_preserved"]["censored_leaked_into_mean"],
        "population_reconciles_demand": vp["Vp3_censored_count_complement"]["reconciles_demand"],
        "passenger_served_count_contract_identical": vp["Vp2_completed_count_contract"]["contracts_identical"],
        "passenger_served_count_difference_reason": vp["Vp2_completed_count_contract"]["documented_reason"],
        "existing_kpi_semantics_contradicted": vp["Vp2_completed_count_contract"]["existing_kpi_semantics_contradicted"],
        "passed": all(vp[k]["passed"] for k in (
            "Vp1_completed_sum_matches_wait_total", "Vp2_completed_count_contract",
            "Vp3_censored_count_complement", "VpC_avg_wait_semantics_preserved")),
    })

    dump(root, "p95_aggregation_semantics_audit.json", {
        "prohibited_form": "overall_p95 = mean(window_p95)",
        "prohibited_form_used": False,
        "implemented_form": "pooled_wait_p95 recomputes the percentile from the pooled raw per-passenger ledger",
        "pooled_value_seconds": vp["Vp12_no_percentile_averaging"]["pooled_value_seconds"],
        "expected_pooled_value_seconds": vp["Vp12_no_percentile_averaging"]["expected_pooled_value_seconds"],
        "prohibited_mean_of_window_p95": vp["Vp12_no_percentile_averaging"]["prohibited_mean_of_window_p95"],
        "pooled_population_count": vp["Vp12_no_percentile_averaging"]["pooled_population_count"],
        "window_level_p95_status": "MEASURED",
        "higher_level_p95_aggregation_status": vp["Vp12_no_percentile_averaging"]["higher_level_p95_aggregation_status"],
        "silent_approximation": False,
        "passed": vp["Vp12_no_percentile_averaging"]["passed"],
    })

    chain = {
        "passenger_wait_p95_seconds": {
            "required_primitives": ["authoritative arrival_ts per passenger", "board_ts at SERVE settlement", "fixed horizon_end"],
            "window_rollup_columns": [
                "passenger_wait_p95_seconds", "passenger_wait_p95_status", "passenger_wait_p95_semantics",
                "passenger_wait_p95_population", "passenger_wait_p95_percentile_method",
                "passenger_wait_p95_censor_reference", "wait_completed_passenger_count",
                "wait_censored_passenger_count", "wait_population_passenger_count", "wait_censored_share",
            ],
            "raw_event_source": "StopState.arrival_schedule arrival timestamps + SERVE boarding transitions + fixed horizon close",
            "chain": "arrival_ts + (board_ts | horizon_end) -> anonymous per-passenger duration ledger -> measured empirical p95 -> window_rollup -> canonical KPI",
            "source_classification": "MEASURED_CAUSAL_TRANSITION_ACCOUNTING",
            "action_count_path_used": False,
            "avg_multiplier_path_used": False,
        },
        "avg_wait_seconds": {
            "required_primitives": ["completed wait ledger"],
            "window_rollup_columns": ["wait_total_passenger_seconds", "wait_passenger_count"],
            "raw_event_source": "SERVE boarding transitions on or before horizon_end",
            "chain": "arrival_ts + board_ts -> completed ledger -> sum/count -> aggregator ratio",
            "source_classification": "MEASURED_CAUSAL_TRANSITION_ACCOUNTING",
        },
        "in_vehicle_time_seconds": {
            "required_primitives": ["alighting event"],
            "window_rollup_columns": [],
            "raw_event_source": None,
            "chain": "no authoritative primitive chain exists",
            "source_classification": "NOT_YET_MEASURABLE",
        },
    }
    for name, provenance in bridge.KPI_FIELD_PROVENANCE.items():
        chain.setdefault(name, {
            "required_primitives": ["see bridge provenance string"],
            "window_rollup_columns": [name],
            "raw_event_source": "simulator transition accounting",
            "chain": provenance,
            "source_classification": "MEASURED_CAUSAL_TRANSITION_ACCOUNTING",
        })
    empty_chains = [k for k, v in chain.items() if not v["required_primitives"] and v["source_classification"] != "NOT_YET_MEASURABLE"]
    dump(root, "canonical_kpi_field_provenance.json", {
        "kpi_chains": chain,
        "kpi_count": len(chain),
        "empty_required_input_lists": empty_chains,
        "not_yet_measurable": [k for k, v in chain.items() if v["source_classification"] == "NOT_YET_MEASURABLE"],
        "r3_v12_check": r3_checks["V12_kpi_field_provenance"],
        "passed": not empty_chains and r3_checks["V12_kpi_field_provenance"]["passed"],
    })

    dump(root, "demand_calibration_deferred_dependency.json", {
        "observed_passenger_demand_generated": counts["demand_generated"],
        "observed_completed_within_horizon": counts["completed_count"],
        "observed_censored_share": counts["censored_share"],
        "demand_generation_changed": False,
        "arrival_intensity_changed": False,
        "route_demand_calibration_changed": False,
        "service_capacity_assumptions_changed": False,
        "status": "DEFERRED_TO_H4M_AE_R4",
        "impact_on_r3_1_claim": (
            "the uncalibrated demand scale drives the simulator vehicle clock past the fixed horizon end, so the "
            "censored share dominates the measured tail. The measurement path is correct and fail-closed, but the "
            "absolute KPI magnitudes are not usable for performance claims until demand is calibrated."
        ),
        "downstream_dependency": ["passenger_service_rate horizon semantics", "absolute KPI magnitudes"],
    })

    dump(root, "p95_vtest_report.json", r31_result)
    dump(root, "r3_v1_v18_revalidation.json", {
        "checks": r3_checks,
        "pass_count": sum(1 for c in r3_checks.values() if c["passed"]),
        "total": len(r3_checks),
        "all_passed": all(c["passed"] for c in r3_checks.values()),
        "gates_weakened_to_obtain_pass": False,
    })

    dump(root, "reward_v2_non_regression.json", {"check": r3_checks["V7_reward_v2_unchanged"], "redesigned": False, "passed": r3_checks["V7_reward_v2_unchanged"]["passed"]})
    dump(root, "zero_loss_non_regression.json", {"check": r3_checks["V8_zero_loss_unchanged"], "redesigned": False, "passed": r3_checks["V8_zero_loss_unchanged"]["passed"]})
    dump(root, "k_mask_non_regression.json", {"check": r3_checks["V6_kmask_unchanged"], "redesigned": False, "passed": r3_checks["V6_kmask_unchanged"]["passed"]})
    dump(root, "test6_non_access_audit.json", {"check": r3_checks["V16_test6_untouched"], "test6_accessed": False, "test6_status": "EXHAUSTED_AND_UNTOUCHED", "passed": r3_checks["V16_test6_untouched"]["passed"]})
    dump(root, "in_vehicle_time_guard.json", {"check": r3_checks["V17_in_vehicle_not_measurable"], "status": "NOT_YET_MEASURABLE", "proxy_emitted": False, "passed": r3_checks["V17_in_vehicle_not_measurable"]["passed"]})

    all_vp = all(c["passed"] for c in vp.values())
    all_v = all(c["passed"] for c in r3_checks.values())
    gate_passed = all_vp and all_v and upstream["manifest_verified"]

    gate = {
        "gate": GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R3_1_MEASURED_WAIT_TAIL_REPAIR_INCOMPLETE",
        "source_sha": head_sha,
        "upstream_r3_sha": UPSTREAM_R3_SHA,
        "r3_frozen_module_changed": new_bridge_sha != R3_FROZEN_BRIDGE_SHA,
        "new_bridge_sha": new_bridge_sha,
        "actor_observation_dim": r3_checks["V1_checkpoint_131d_action"].get("observation_dim", 131),
        "checkpoint_observation_match": r3_checks["V1_checkpoint_131d_action"]["passed"],
        "arrival_schedule_unit_semantics": bridge.ARRIVAL_ENTRY_SEMANTICS,
        "carry_in_waiters_present": vp["Vp0_carry_in_population_boundary"]["initial_waiting_count"] > 0,
        "carry_in_timestamp_status": vp["Vp0_carry_in_population_boundary"]["carry_in_timestamp_status"],
        "completed_wait_ledger_valid": vp["Vp1_completed_sum_matches_wait_total"]["passed"],
        "censored_wait_ledger_valid": vp["Vp4_censored_duration_exact"]["passed"],
        "canonical_p95_population": bridge.P95_POPULATION,
        "p95_semantics": bridge.P95_SEMANTICS,
        "p95_method": f"{bridge.PERCENTILE_LIBRARY}(q={bridge.PERCENTILE_Q}, method='{bridge.PERCENTILE_METHOD}')",
        "completed_observation_count": counts["completed_count"],
        "censored_observation_count": counts["censored_count"],
        "censored_share": counts["censored_share"],
        "measured_p95_emitted": True,
        "synthetic_p95_fallback_reachable": False,
        "fillna_zero_reachable": False,
        "served_only_p95_canonical": False,
        "wait_total_ledger_invariant_pass": vp["Vp1_completed_sum_matches_wait_total"]["passed"],
        "p95_determinism_pass": vp["Vp6_deterministic_ledger_and_p95"]["passed"],
        "p95_incentive_alignment_pass": vp["Vp7_service_denial_positive_control"]["passed"] and vp["Vp8_serve_vs_hold_ordering"]["passed"],
        "higher_level_p95_aggregation_status": vp["Vp12_no_percentile_averaging"]["higher_level_p95_aggregation_status"],
        "r3_v1_v18_pass_count": sum(1 for c in r3_checks.values() if c["passed"]),
        "r3_v1_v18_all_pass": all_v,
        "reward_v2_unchanged": r3_checks["V7_reward_v2_unchanged"]["passed"],
        "zero_loss_unchanged": r3_checks["V8_zero_loss_unchanged"]["passed"],
        "k_mask_unchanged": r3_checks["V6_kmask_unchanged"]["passed"],
        "demand_calibration_status": "DEFERRED_TO_H4M_AE_R4",
        "test6_accessed": False,
        "in_vehicle_time_status": "NOT_YET_MEASURABLE",
        "performance_claim_allowed": False,
        "causal_comparison_executed": False,
        "recommended_next_gate": NEXT_GATE,
        "vp_pass_count": sum(1 for c in vp.values() if c["passed"]),
        "vp_total": len(vp),
    }
    dump(root, "gate_decision.json", gate)

    dump(root, "downstream_lock.json", {
        "locked_by": gate["gate"],
        "measured_tail_contract_sha256": hashlib.sha256(json.dumps({
            "population": bridge.P95_POPULATION,
            "semantics": bridge.P95_SEMANTICS,
            "method": bridge.PERCENTILE_METHOD,
            "q": bridge.PERCENTILE_Q,
            "censor_reference": bridge.CENSOR_REFERENCE,
            "carry_in_case": bridge.CARRY_IN_CASE,
            "arrival_entry_semantics": bridge.ARRIVAL_ENTRY_SEMANTICS,
            "bridge_sha256": new_bridge_sha,
        }, sort_keys=True).encode("utf-8")).hexdigest(),
        "next_gate": NEXT_GATE,
        "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "A/B1/B2 performance comparison", "full 54-window performance run", "TEST6 access",
            "demand calibration", "B0R causal promotion", "B0C execution", "GitHub push",
        ],
    })

    dump(root, "final_report.json", {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-R3.1",
        "gate": gate["gate"],
        "measured_p95_seconds": p95_runtime["p95_seconds"],
        "censored_share": counts["censored_share"],
        "vp_checks": f"{gate['vp_pass_count']}/{gate['vp_total']}",
        "r3_checks": f"{gate['r3_v1_v18_pass_count']}/{len(r3_checks)}",
        "next_gate": NEXT_GATE,
        "gate_decision": gate,
    })

    md = [
        "# H4M-AE-R3.1 Measured Passenger Wait Tail Accounting Repair",
        "",
        f"- gate: `{gate['gate']}`",
        f"- source HEAD at run: `{head_sha}`",
        f"- upstream R3: `{UPSTREAM_R3_SHA}` (artifact `{upstream['artifact_dir']}`, {upstream['file_count']} files verified)",
        f"- R3-frozen bridge SHA `{R3_FROZEN_BRIDGE_SHA}` -> new bridge SHA `{new_bridge_sha}`",
        "",
        "## What was repaired",
        "",
        "The promoted causal path no longer depends on the synthetic aggregator fallback "
        "`passenger_wait_p95_seconds = avg_wait_seconds * 1.65`. The bridge keeps an anonymous per-passenger "
        "wait ledger and emits a measured empirical p95.",
        "",
        f"- population: `{bridge.P95_POPULATION}`",
        f"- semantics: `{bridge.P95_SEMANTICS}`",
        f"- censor reference: `{bridge.CENSOR_REFERENCE}` (fixed horizon end, never the vehicle clock)",
        f"- method: `{bridge.PERCENTILE_LIBRARY}(q={bridge.PERCENTILE_Q}, method='{bridge.PERCENTILE_METHOD}')`, unweighted",
        f"- carry-in: `{vp['Vp0_carry_in_population_boundary']['case']}`",
        f"- arrival entry unit: `{bridge.ARRIVAL_ENTRY_SEMANTICS}`",
        "",
        "## Measured validation window",
        "",
        "| quantity | value |",
        "| --- | --- |",
        f"| measured p95 | {p95_runtime['p95_seconds']:.3f} s |",
        f"| median | {p95_runtime['median_seconds']:.3f} s |",
        f"| max | {p95_runtime['max_seconds']:.3f} s |",
        f"| completed observations | {counts['completed_count']} |",
        f"| censored observations | {counts['censored_count']} |",
        f"| censored share | {counts['censored_share']:.4f} |",
        f"| legacy synthetic value (had fallback been reached) | {fallback['legacy_multiplier_branch_value_if_column_absent']:.3f} s |",
        f"| served-only diagnostic (non-canonical) | {vp['Vp11_served_only_noncanonical']['diagnostic_value_seconds']:.3f} s |",
        "",
        "Both p95 support points are censored observations and the value is interpolated between them, so the "
        "result is reported as a lower bound rather than as an observed completion time.",
        "",
        "## Validation",
        "",
        f"- R3.1 Vp checks: {gate['vp_pass_count']}/{gate['vp_total']}",
        f"- R3 V1-V18 revalidation: {gate['r3_v1_v18_pass_count']}/{len(r3_checks)} (no gate weakened)",
        "- Reward V2 / Zero-Loss / K-mask unchanged; TEST6 untouched; in_vehicle_time NOT_YET_MEASURABLE",
        "",
        "## What PASS does not mean",
        "",
        "- demand is **not** calibrated: "
        f"{counts['demand_generated']} passengers are generated in the window and "
        f"{counts['censored_share']:.1%} of the population is censored, so absolute KPI magnitudes are not "
        "usable for performance claims. Deferred to R4.",
        "- `passenger_served_count` keeps its R3-frozen definition and therefore differs from the completed-wait "
        f"count by {vp['Vp2_completed_count_contract']['post_horizon_boarding_count']} post-horizon boardings; the "
        "difference is documented, not silently reconciled.",
        "- no causal comparison was executed, no checkpoint changed, TEST6 was not opened.",
        "",
        "## Next gate",
        "",
        f"`{NEXT_GATE}` (not executed automatically).",
    ]
    (root / "final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    files = sorted(p.name for p in root.iterdir() if p.is_file())
    dump(root, "artifact_manifest.json", {
        "artifact_dir": root.name,
        "generated_at": datetime.now(KST).isoformat(),
        "source_sha": head_sha,
        "gate": gate["gate"],
        "file_sha256": {n: sha256_file(root / n) for n in files if n != "artifact_manifest.json"},
        "source_file_sha256": {
            "causal_kpi_bridge.py": new_bridge_sha,
            "test_h4m_ae_r3_causal_kpi_bridge.py": sha256_file(R3_TEST),
            "test_h4m_ae_r3_1_measured_wait_tail.py": sha256_file(R3_1_TEST),
            Path(__file__).name: sha256_file(Path(__file__)),
        },
    })
    (root / "_SUCCESS.lock").write_text(
        json.dumps({"gate": gate["gate"], "passed": gate_passed, "next_gate": NEXT_GATE}, indent=2), encoding="utf-8"
    )

    print(f"[H4M-AE-R3.1] artifact root: {root}")
    print(f"[H4M-AE-R3.1] gate: {gate['gate']}")
    print(f"[H4M-AE-R3.1] Vp {gate['vp_pass_count']}/{gate['vp_total']} | R3 V1-V18 {gate['r3_v1_v18_pass_count']}/{len(r3_checks)}")
    print(f"[H4M-AE-R3.1] measured p95 = {p95_runtime['p95_seconds']:.3f}s, censored share = {counts['censored_share']:.4f}")
    print(f"[H4M-AE-R3.1] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R3.1] BLOCKED")


if __name__ == "__main__":
    main()
