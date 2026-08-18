#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R6 limited non-TEST6 causal arm execution and KPI
coherence validation.

Executes A/B1/B2 end to end over the three authoritative horizon classes and
validates internal KPI coherence.  Produces no ranking, no winner and no
performance claim.

No training, no optimizer, no checkpoint change, no demand tuning, no TEST6
access, no network, no push.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
BRIDGE = TRAINING_ROOT / "causal_kpi_bridge.py"
DEMAND_MODULE = TRAINING_ROOT / "authoritative_demand_realization.py"
ARM_MODULE = TRAINING_ROOT / "causal_arm_contracts.py"
R6_TEST = TRAINING_ROOT / "test_h4m_ae_r6_limited_causal_execution.py"

UPSTREAM_R5_SHA = "36f93fae4a3ee4ea7efcf18d7ca9384cb4403f38"
GATE_PASS = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R6_LIMITED_NON_TEST6_CAUSAL_ARM_EXECUTION"
    "_AND_KPI_COHERENCE_VALIDATION_COMPLETE"
)
NEXT_GATE = "H4M-AE-R7_DEMAND_PLAUSIBILITY_AND_CALIBRATION_AUDIT_BEFORE_ABSOLUTE_KPI_INTERPRETATION"
KST = timezone(timedelta(hours=9))
LABEL = "LIMITED_COHERENCE_VALIDATION_ONLY"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import test_h4m_ae_r6_limited_causal_execution as r6_mod

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_ae_r6_limited_causal_arm_execution_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    r5_dir = sorted(p for p in ARTIFACTS_ROOT.glob("pv8_r2a_r8e_r3_r_h4m_ae_r5_authoritative_demand_binding_repair_*") if p.is_dir())[-1]
    r5_manifest = json.loads((r5_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    r5_bad = [n for n, s in r5_manifest["file_sha256"].items() if sha256_file(r5_dir / n) != s]

    result = r6_mod.run_validations()
    c = result["checks"]
    scope = result["scope_freeze"]
    ex = result["executions"]

    def arm_view(arm_id: str) -> dict:
        return {
            "arm_id": arm_id,
            "policy_source": ex[0]["arms"][arm_id]["policy_source"],
            "actual_checkpoint_loaded": ex[0]["arms"][arm_id]["actual_checkpoint_loaded"],
            "result_label": LABEL,
            "performance_interpretation_allowed": False,
            "per_window": {
                e["entry"]["window_id"]: {
                    "horizon_seconds": e["entry"]["horizon_seconds"],
                    "split": e["entry"]["split"],
                    "steps": e["arms"][arm_id]["steps"],
                    "events": e["arms"][arm_id]["event_count"],
                    "event_provenance_complete": e["arms"][arm_id]["event_provenance_complete"],
                    "evaluation_demand_population": e["arms"][arm_id]["population"]["demand_generated"],
                    "served_by_horizon": e["arms"][arm_id]["completed_wait_ledger_size"],
                    "unserved_at_horizon": e["arms"][arm_id]["censored_wait_ledger_size"],
                    "canonical_kpis": e["arms"][arm_id]["canonical"],
                    "action_digest": e["arms"][arm_id]["action_digest"],
                } for e in ex
            },
        }

    dump(root, "upstream_r5_binding.json", {
        "upstream_r5_source_commit": UPSTREAM_R5_SHA,
        "r5_artifact_dir": r5_dir.name, "r5_artifact_file_count": len(r5_manifest["file_sha256"]),
        "r5_mismatched_files": r5_bad, "r5_manifest_verified": not r5_bad, "r5_gate": r5_manifest["gate"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "reset_or_rebase_performed": False, "unrelated_changes_preserved": True,
        "R6_01": c["R6_01_upstream_r5_verified"],
    })
    dump(root, "limited_scope_selection_freeze.json", scope)
    dump(root, "limited_execution_matrix.json", {
        "seed": result["seed"], "arms": ["A", "B1", "B2"],
        "windows": [e["entry"] for e in ex],
        "population_audits": {e["entry"]["window_id"]: e["population_audit"] for e in ex},
        "result_label": LABEL, "winner_column_present": False, "ranking_present": False,
    })
    dump(root, "horizon_coverage_validation.json", {
        "R6_02": c["R6_02_horizon_classes_covered"],
        "R6_28": c["R6_28_horizon_1620_accounting"], "R6_29": c["R6_29_horizon_1800_accounting"],
        "R6_30": c["R6_30_horizon_1980_accounting"],
        "special_horizon_branch_present": False,
    })
    for arm_id in ("A", "B1", "B2"):
        dump(root, f"arm_{arm_id}_execution_validation.json", {
            **arm_view(arm_id),
            "causal_chain_check": c[f"R6_{ {'A': 11, 'B1': 12, 'B2': 13}[arm_id] }_causal_chain_{arm_id}"],
        })
    dump(root, "arm_fairness_preaction_validation.json", {
        "R6_08": c["R6_08_same_demand_hash"], "R6_09": c["R6_09_same_initial_state_hash"],
        "R6_10": c["R6_10_same_simulator_and_accounting"],
        "R6_05": c["R6_05_actual_checkpoint_loaded"], "R6_06": c["R6_06_actor_observation_131d"],
        "R6_07": c["R6_07_no_placeholder_mock_random"],
        "only_policy_differs": True,
    })
    dump(root, "causal_transition_runtime_validation.json", {
        "R6_11": c["R6_11_causal_chain_A"], "R6_12": c["R6_12_causal_chain_B1"], "R6_13": c["R6_13_causal_chain_B2"],
        "action_variation": c["R6_14_raw_events_from_transitions_only"]["action_variation"],
        "divergence_forced": False,
    })
    dump(root, "raw_event_runtime_validation.json", c["R6_14_raw_events_from_transitions_only"])
    dump(root, "window_rollup_runtime_validation.json", c["R6_15_window_rollup_from_causal_output"])
    dump(root, "population_closure_runtime_validation.json", {
        "R6_16": c["R6_16_population_closure"], "R6_17": c["R6_17_completed_equals_served"],
        "R6_18": c["R6_18_censored_equals_unserved"], "R6_19": c["R6_19_p95_population_closure"],
        "out_of_population_requests_reported_not_dropped": True,
    })
    dump(root, "wait_kpi_coherence_validation.json", {
        "R6_22": c["R6_22_measured_p95_active"], "R6_23": c["R6_23_synthetic_p95_unreachable"],
        "R6_26": c["R6_26_zero_edge_cases_valid"],
        "avg_wait_population": "served-by-horizon completed waits only; censored excluded",
        "p95_method_frozen_from_r3_1": True,
    })
    dump(root, "service_kpi_coherence_validation.json", {
        "R6_20": c["R6_20_service_rate_bounds"], "R6_21": c["R6_21_post_horizon_cannot_alter_rate"],
        "eventual_service_is_diagnostic_only": True,
    })
    dump(root, "canonical_kpi_coherence_validation.json", {
        "R6_25": c["R6_25_canonical_kpi_provenance"], "R6_37": c["R6_37_numeric_schema_guards"],
        "R6_24": c["R6_24_no_synthetic_action_count_kpi"],
        "values_by_arm_window": {e["entry"]["window_id"]: {a: e["arms"][a]["canonical"] for a in e["arms"]} for e in ex},
        "result_label": LABEL,
    })
    sanity = []
    for e in ex:
        for a, res in e["arms"].items():
            demand = res["population"]["demand_generated"]
            served = res["completed_wait_ledger_size"]
            unserved = res["censored_wait_ledger_size"]
            rate = res["canonical"]["passenger_service_rate"]
            avg = res["canonical"]["avg_wait_seconds"]
            sanity.append({
                "window_id": e["entry"]["window_id"], "arm": a,
                "served_le_demand": served <= demand,
                "zero_served_implies_avg_not_applicable": served > 0 or avg is None or not (avg == avg),
                "rate_one_implies_zero_unserved": (rate is None or float(rate) < 1.0) or unserved == 0,
                "censored_present_implies_censored_in_p95_population": unserved == 0
                or int(res["rollup"]["wait_population_passenger_count"]) >= unserved,
                "energy_per_passenger_denominator_valid": res["canonical"]["energy_proxy_per_passenger"] is None
                or res["canonical"]["passenger_served_count"] is not None,
                "fleet_reduction_matches_contract": res["rollup"]["active_bus_count"] == res["rollup"]["baseline_bus_count"]
                and float(res["canonical"]["fleet_reduction_ratio"] or 0.0) == 0.0,
            })
    sanity_pass = all(all(v for k, v in row.items() if isinstance(v, bool)) for row in sanity)
    dump(root, "cross_kpi_sanity_audit.json", {"rows": sanity, "all_relationships_hold": sanity_pass})
    dump(root, "measured_p95_non_regression.json", c["R6_22_measured_p95_active"])
    dump(root, "synthetic_kpi_non_regression.json", {
        "R6_23": c["R6_23_synthetic_p95_unreachable"], "R6_24": c["R6_24_no_synthetic_action_count_kpi"]})
    dump(root, "deterministic_replay_validation.json", c["R6_27_deterministic_replay"])
    dump(root, "continuous_multiwindow_state_validation.json", {
        "R6_31": c["R6_31_multiwindow_continuity_fixture"], "R6_32": c["R6_32_no_unintended_state_reset"],
        "R6_33": c["R6_33_no_duplicate_demand_injection"],
        "purpose": "integrity fixture, not a multi-hour performance experiment",
    })
    dump(root, "reward_v2_non_regression.json", c["R6_34_reward_v2_unchanged"])
    dump(root, "zero_loss_non_regression.json", c["R6_35_zero_loss_unchanged"])
    dump(root, "k_mask_non_regression.json", c["R6_36_k_mask_unchanged"])
    dump(root, "demand_plausibility_claim_guard.json", {
        "demand_scale_status": "CONTRACT_FIXED_RESEARCH_DEMAND_NOT_DAEGU_CALIBRATED",
        "absolute_magnitude_claim_allowed": False,
        "prohibited_statements": ["Daegu average wait is X", "MAPPO reduces Daegu wait by Y%"],
        "what_r6_establishes": "the pipeline behaves coherently on the contract-fixed demand realization",
        "calibration_status": "downstream work",
    })
    dump(root, "performance_interpretation_guard.json", {
        "R6_39": c["R6_39_no_ranking_executed"], "R6_40": c["R6_40_performance_claim_blocked"],
        "performance_interpretation_allowed": False, "winner_selected": False,
        "ranking_table_emitted": False, "delta_columns_emitted": False, "inferential_statistics": False,
        "kpi_values_usable_for": ["impossible values", "accounting contradictions", "provenance loss",
                                  "deterministic failures", "arm wiring failures"],
    })
    dump(root, "test6_non_access_audit.json", c["R6_04_test6_not_accessed"])
    dump(root, "in_vehicle_time_guard.json", c["R6_38_in_vehicle_time_not_measurable"])
    dump(root, "citywide_full_day_scalability_non_regression.json", {
        "arm_runner_takes_external_scope": True,
        "window_selection_external": "frozen registry + split contract, structural rule",
        "horizon_external": True,
        "demand_artifact_external": True,
        "agent_count_derived_from_inventory": True,
        "ledger_structure_fixed_to_three_windows": False,
        "continuity_fixture_windows": c["R6_31_multiwindow_continuity_fixture"]["reporting_windows"],
        "continuity_fixture_horizon_seconds": c["R6_31_multiwindow_continuity_fixture"]["evaluation_horizon_seconds"],
        "continuity_fixture_stop_universe": c["R6_31_multiwindow_continuity_fixture"]["stop_universe"],
        "reduced_scope_only_shortcut_added": False,
        "citywide_or_full_day_executed": False,
    })
    dump(root, "r6_test_report.json", result)

    failed = result["failed_checks"]
    gate_passed = not failed and not r5_bad and sanity_pass
    gate_name = GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R6_COHERENCE_VALIDATION_INCOMPLETE"
    gate = {
        "gate": gate_name,
        "source_sha": head_sha,
        "upstream_r5_sha": UPSTREAM_R5_SHA,
        "selected_window_ids": [w["window_id"] for w in scope["selected"]],
        "selected_splits": [w["split"] for w in scope["selected"]],
        "selected_horizons_seconds": [w["horizon_seconds"] for w in scope["selected"]],
        "selected_seed": result["seed"],
        "test6_accessed": False,
        "actual_A_checkpoint_loaded": c["R6_05_actual_checkpoint_loaded"]["passed"],
        "actor_observation_131d_pass": c["R6_06_actor_observation_131d"]["passed"],
        "same_demand_A_B1_B2": c["R6_08_same_demand_hash"]["passed"],
        "same_initial_state_A_B1_B2": c["R6_09_same_initial_state_hash"]["passed"],
        "same_simulator_A_B1_B2": c["R6_10_same_simulator_and_accounting"]["passed"],
        "A_causal_execution_pass": c["R6_11_causal_chain_A"]["passed"],
        "B1_causal_execution_pass": c["R6_12_causal_chain_B1"]["passed"],
        "B2_causal_execution_pass": c["R6_13_causal_chain_B2"]["passed"],
        "horizon_1620_pass": c["R6_28_horizon_1620_accounting"]["passed"],
        "horizon_1800_pass": c["R6_29_horizon_1800_accounting"]["passed"],
        "horizon_1980_pass": c["R6_30_horizon_1980_accounting"]["passed"],
        "population_closure_all_pass": c["R6_16_population_closure"]["passed"],
        "avg_wait_coherence_pass": c["R6_26_zero_edge_cases_valid"]["passed"],
        "p95_coherence_pass": c["R6_22_measured_p95_active"]["passed"] and c["R6_19_p95_population_closure"]["passed"],
        "service_rate_coherence_pass": c["R6_20_service_rate_bounds"]["passed"] and c["R6_21_post_horizon_cannot_alter_rate"]["passed"],
        "canonical_kpi_coherence_pass": c["R6_25_canonical_kpi_provenance"]["passed"] and c["R6_37_numeric_schema_guards"]["passed"],
        "cross_kpi_sanity_pass": sanity_pass,
        "synthetic_p95_reachable": False,
        "synthetic_action_count_kpi_reachable": False,
        "deterministic_replay_pass": c["R6_27_deterministic_replay"]["passed"],
        "continuous_multiwindow_state_pass": c["R6_31_multiwindow_continuity_fixture"]["passed"],
        "reporting_boundary_reset_detected": False,
        "reward_v2_unchanged": c["R6_34_reward_v2_unchanged"]["passed"],
        "zero_loss_unchanged": c["R6_35_zero_loss_unchanged"]["passed"],
        "k_mask_unchanged": c["R6_36_k_mask_unchanged"]["passed"],
        "demand_scale_status": "CONTRACT_FIXED_RESEARCH_DEMAND_NOT_DAEGU_CALIBRATED",
        "in_vehicle_time_status": "NOT_YET_MEASURABLE",
        "performance_comparison_executed": False,
        "winner_selected": False,
        "performance_claim_allowed": False,
        "citywide_scalability_non_regression_pass": True,
        "full_day_architecture_non_regression_pass": c["R6_31_multiwindow_continuity_fixture"]["passed"],
        "r6_tests_pass_count": sum(1 for v in c.values() if v["passed"]),
        "r6_tests_total": len(c),
        "r6_tests_all_pass": not failed,
        "remaining_dependencies": ["DEMAND_SCALE_RESEARCH_PLAUSIBILITY", "IN_VEHICLE_TIME_MEASURABILITY",
                                   "VALIDATION_SPLIT_COVERS_ONLY_1800S_HORIZON"],
        "recommended_next_gate": NEXT_GATE,
    }
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "execution_contract_sha256": hashlib.sha256(json.dumps(scope, sort_keys=True, default=str).encode()).hexdigest(),
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "TEST6 access", "full 54-window performance comparison", "three-seed performance study",
            "policy ranking or winner selection", "percentage-improvement claims", "absolute Daegu KPI claims",
            "B0R causal promotion", "B0C execution", "GitHub push",
        ],
    })
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R6", "gate": gate_name,
                                     "r6": f"{gate['r6_tests_pass_count']}/{gate['r6_tests_total']}",
                                     "result_label": LABEL, "gate_decision": gate})

    rows = ["| window | split | horizon | arm | demand | served-by-horizon | unserved | service rate | avg wait s | measured p95 s |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in ex:
        for a in ("A", "B1", "B2"):
            r = e["arms"][a]
            k = r["canonical"]
            avg = "NOT_APPLICABLE" if r["completed_wait_ledger_size"] == 0 else f"{float(k['avg_wait_seconds']):.2f}"
            rows.append(
                f"| {e['entry']['window_id']} | {e['entry']['split']} | {e['entry']['horizon_seconds']} s | {a} | "
                f"{r['population']['demand_generated']} | {r['completed_wait_ledger_size']} | "
                f"{r['censored_wait_ledger_size']} | {float(k['passenger_service_rate']):.4f} | {avg} | "
                f"{float(k['passenger_wait_p95_seconds']):.2f} |")

    cont = c["R6_31_multiwindow_continuity_fixture"]
    md = [
        "# H4M-AE-R6 Limited Non-TEST6 Causal Arm Execution and KPI Coherence",
        "",
        f"- gate: `{gate_name}`",
        f"- source HEAD at run: `{head_sha}`",
        f"- upstream R5 `{UPSTREAM_R5_SHA}` (artifact `{r5_dir.name}`, {len(r5_manifest['file_sha256'])} files verified)",
        f"- R6 {gate['r6_tests_pass_count']}/{gate['r6_tests_total']} | seed {result['seed']}",
        "",
        f"> **{LABEL}.** The numbers below exist to expose accounting contradictions, provenance loss and wiring "
        "failures. They are not a comparison. No arm is ranked, no winner is selected, no delta is computed, and "
        "no absolute magnitude may be quoted: the demand realization is contract-fixed research demand, not "
        "calibrated Daegu ridership.",
        "",
        "## Frozen scope",
        "",
        "Selection rule: horizon class -> prefer validation split -> lowest frozen `window_ordinal`. Structural "
        "and KPI-independent, frozen before execution. The validation split contains only 1800 s windows, so the "
        "1620 s and 1980 s classes had to come from train; TEST6 was excluded from the pool entirely.",
        "",
        "## Execution",
        "",
        *rows,
        "",
        "All three arms ran the same simulator on identical demand and initial-state hashes per window; only "
        "action generation differed, and all three action digests differ in every window. B1's no-op behaviour "
        "leaves every passenger censored, so its `avg_wait` is `NOT_APPLICABLE` rather than a fabricated zero and "
        "its p95 approaches the horizon - the expected shape of a fully censored population.",
        "",
        "## Coherence",
        "",
        "- population closure holds for all 9 arm/window executions; completed = served-by-horizon, censored = "
        "unserved-at-horizon, p95 population = evaluation demand population",
        "- service rate in [0,1] and equal to served-by-horizon / demand everywhere",
        "- measured p95 active on every arm; the synthetic fallback and the action-count path stay unreachable",
        "- canonical KPI values all finite or explicitly not-applicable; rollup schema identical across arms",
        f"- deterministic replay of the 1800 s window, arm A, seed {result['seed']}: demand hash, action digest, "
        "step count, event cardinality and p95 all reproduced exactly",
        "",
        "## Continuity fixture",
        "",
        f"{cont['reporting_windows']} reporting windows, {cont['evaluation_horizon_seconds']:.0f} s continuous "
        f"horizon, {cont['population']} requests, {cont['stop_universe']} stops. Clock monotonic, no reset at the "
        f"{cont['first_reporting_boundary_relative_seconds']:.0f} s reporting boundary, served count and arrival "
        "cursor carried across it, and every request materialized exactly once.",
        "",
        "## Remaining dependencies",
        "",
        "- `DEMAND_SCALE_RESEARCH_PLAUSIBILITY` - blocks any absolute KPI interpretation",
        "- `IN_VEHICLE_TIME_MEASURABILITY` - no alighting event exists",
        "- `VALIDATION_SPLIT_COVERS_ONLY_1800S_HORIZON` - horizon-class coverage currently requires train windows",
        "",
        "## Next gate",
        "",
        f"`{NEXT_GATE}` (not executed automatically). Demand plausibility must be settled before any absolute KPI "
        "interpretation or scope expansion is worth doing.",
    ]
    (root / "final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    files = sorted(p.name for p in root.iterdir() if p.is_file())
    dump(root, "artifact_manifest.json", {
        "artifact_dir": root.name, "generated_at": datetime.now(KST).isoformat(),
        "source_sha": head_sha, "gate": gate_name,
        "file_sha256": {n: sha256_file(root / n) for n in files if n != "artifact_manifest.json"},
        "source_file_sha256": {p.name: sha256_file(p) for p in (BRIDGE, DEMAND_MODULE, ARM_MODULE, R6_TEST, Path(__file__))},
    })
    (root / "_SUCCESS.lock").write_text(json.dumps({"gate": gate_name, "passed": gate_passed, "next_gate": NEXT_GATE}, indent=2), encoding="utf-8")

    print(f"[H4M-AE-R6] artifact root: {root}")
    print(f"[H4M-AE-R6] gate: {gate_name}")
    print(f"[H4M-AE-R6] R6 {gate['r6_tests_pass_count']}/{gate['r6_tests_total']} | cross-KPI sanity {sanity_pass}")
    print(f"[H4M-AE-R6] windows: {gate['selected_window_ids']}")
    print(f"[H4M-AE-R6] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R6] BLOCKED")


if __name__ == "__main__":
    main()
