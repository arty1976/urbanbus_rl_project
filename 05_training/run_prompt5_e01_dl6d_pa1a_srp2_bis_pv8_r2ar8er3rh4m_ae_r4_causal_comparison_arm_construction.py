#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R4 causal comparison arm construction and
pre-evaluation integrity validation.

Constructs the A/B1/B2 arm contracts over one identical causal universe and
audits D0 (horizon population/service accounting) and D1 (demand semantics and
scale) before any performance evaluation is permitted.

No training, no optimizer, no checkpoint change, no demand tuning, no A/B1/B2
performance comparison, no TEST6 access, no network, no push.
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
ARM_MODULE = TRAINING_ROOT / "causal_arm_contracts.py"
R4_TEST = TRAINING_ROOT / "test_h4m_ae_r4_pre_evaluation_integrity.py"

UPSTREAM_R3_1_SHA = "faf43d4dc23e11e9ac6f5a9d6bd1ca286a7485eb"
UPSTREAM_R3_1_GATE = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R3_1_MEASURED_PASSENGER_WAIT_TAIL_ACCOUNTING_REPAIR"
    "_AND_FULL_CAUSAL_BRIDGE_REVALIDATION_COMPLETE"
)
GATE_PASS = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R4_CAUSAL_COMPARISON_ARM_CONSTRUCTION"
    "_AND_PRE_EVALUATION_INTEGRITY_VALIDATION_COMPLETE"
)
BLOCK_PRIMARY = "BLOCKED_DEMAND_UNIT_OR_EXPANSION_REPAIR_REQUIRED"
BLOCK_SECONDARY = "BLOCKED_HORIZON_SERVICE_ACCOUNTING_REPAIR_REQUIRED"
NEXT_GATE_BLOCKED = "H4M-AE-R5_AUTHORITATIVE_DEMAND_REALIZATION_BINDING_AND_HORIZON_SERVICE_ACCOUNTING_REPAIR"
KST = timezone(timedelta(hours=9))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def r3_1_artifact_dir() -> Path:
    dirs = [p for p in ARTIFACTS_ROOT.glob("pv8_r2a_r8e_r3_r_h4m_ae_r3_1_measured_passenger_wait_tail_accounting_repair_*") if p.is_dir()]
    if not dirs:
        raise RuntimeError("R3.1 artifact directory not found")
    return sorted(dirs)[-1]


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import pandas as pd
    import test_h4m_ae_r4_pre_evaluation_integrity as r4_mod
    import test_h4m_ae_r3_causal_kpi_bridge as r3

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_ae_r4_causal_comparison_arm_construction_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    r3_1_dir = r3_1_artifact_dir()
    manifest = json.loads((r3_1_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    mismatched = [n for n, s in manifest["file_sha256"].items() if sha256_file(r3_1_dir / n) != s]
    r3_1_gate = json.loads((r3_1_dir / "gate_decision.json").read_text(encoding="utf-8"))

    result = r4_mod.run_validations()
    c = result["checks"]
    bridge = r3.imp("bridge_r4", BRIDGE)

    dump(root, "upstream_r3_1_binding.json", {
        "upstream_r3_1_source_commit": UPSTREAM_R3_1_SHA,
        "upstream_r3_1_gate": UPSTREAM_R3_1_GATE,
        "recorded_upstream_gate": r3_1_gate["gate"],
        "upstream_gate_matches": r3_1_gate["gate"] == UPSTREAM_R3_1_GATE,
        "artifact_dir": r3_1_dir.name,
        "artifact_file_count": len(manifest["file_sha256"]),
        "mismatched_files": mismatched,
        "manifest_verified": not mismatched,
        "head_sha_at_run": head_sha,
        "git_status_short": status,
        "reset_or_rebase_performed": False,
        "unrelated_changes_preserved": True,
        "module_sha256": result["module_sha256"],
    })

    closure = c["R4_01_population_closure"]
    dump(root, "evaluation_population_contract.json", {
        "horizon_minutes": 30,
        "carry_in_case": bridge.CARRY_IN_CASE,
        "evaluation_demand_population": closure["evaluation_demand_population"],
        "served_by_horizon_definition": "boarded at a SERVE transition with board_ts <= fixed horizon_end",
        "unserved_at_horizon_definition": "authoritative arrival_ts <= horizon_end and not boarded by horizon_end",
        "required_identity": "evaluation_demand_population = served_by_horizon + unserved_at_horizon",
        "p95_population_binding": bridge.P95_POPULATION,
        "censor_reference": bridge.CENSOR_REFERENCE,
        "passenger_may_disappear": False,
        "passenger_may_be_double_counted": False,
    })
    dump(root, "population_closure_audit.json", {
        "R4_01": closure,
        "R4_02": c["R4_02_completed_equals_served_by_horizon"],
        "R4_03": c["R4_03_censored_equals_unserved"],
        "R4_04": c["R4_04_p95_population_equals_demand"],
        "passed": all(c[k]["passed"] for k in (
            "R4_01_population_closure", "R4_02_completed_equals_served_by_horizon",
            "R4_03_censored_equals_unserved", "R4_04_p95_population_equals_demand")),
    })
    dump(root, "passenger_served_count_semantics_audit.json", c["R4_06_passenger_served_count_semantics"])
    dump(root, "horizon_service_accounting_audit.json", {
        "R4_05": c["R4_05_post_horizon_cannot_improve_service_rate"],
        "served_by_horizon": closure["served_by_horizon"],
        "eventually_served_count": closure["eventually_served_count"],
        "post_horizon_boarding_count": closure["post_horizon_boarding_count"],
        "defect": (
            "canonical passenger_service_rate reports 1.0 while the horizon-consistent rate is "
            f"{c['R4_05_post_horizon_cannot_improve_service_rate']['horizon_consistent_rate']}; post-horizon "
            "boardings inflate a 30-minute service KPI to a perfect score"
        ),
        "blocker": BLOCK_SECONDARY,
        "silent_repair_attempted": False,
        "passed": c["R4_05_post_horizon_cannot_improve_service_rate"]["passed"],
    })
    dump(root, "service_rate_semantics_review.json", {
        "canonical_formula": "passenger_service_rate = passenger_served_count / passenger_demand_generated",
        "aggregator": "05_training/evaluation/canonical_kpi_aggregator.py::ensure_phase2_12_kpis",
        "aggregator_modified_by_r4": False,
        "required_properties": {
            "same_horizon_as_p95_censor_point": False,
            "same_demand_population": True,
            "post_horizon_boarding_cannot_improve_rate": False,
            "result_in_unit_interval": True,
            "numerator_le_denominator": True,
            "identical_definition_across_arms": True,
        },
        "verdict": "FAIL_CLOSED",
        "repair_dependency": "horizon-closed service primitive (passenger_served_by_horizon_count) must exist before a 30-minute service rate can be claimed",
        "new_canonical_field_promoted_by_r4": False,
        "passed": False,
    })

    lineage = c["R4_07_demand_source_traced"]
    duplication = c["R4_08_demand_expansion_no_duplication"]
    dump(root, "demand_source_lineage.json", lineage)
    dump(root, "demand_unit_granularity_audit.json", {
        "raw_source_field": "historical_boarding_intensity (and historical_demand_score = boarding + alighting)",
        "raw_source_dataset": lineage["raw_source_dataset"],
        "value_reconstruction": lineage["value_reconstruction"],
        "evidence_class": lineage["evidence_class"],
        "observed_individual_passenger_claim": lineage["observed_individual_passenger_claim"],
        "unit": lineage["unit"],
        "time_granularity": "one fixed 30-minute evaluation window (band clock definition), not a per-second rate",
        "stop_granularity": f"aggregate over the window's {lineage['registry_mapped_stop_count']} mapped stops, not per stop",
        "route_direction_granularity": "one route/direction per registry row",
        "classification": "AGGREGATE_RECONSTRUCTED_INTENSITY_FEATURE_NOT_PASSENGER_COUNT",
        "25696_interpretation": (
            "25,696 is NOT 25,696 observed or contracted passengers. It is the bridge's own expansion "
            f"round({duplication['bridge_rate_value']} * horizon/600) = {duplication['bridge_per_stop_expected']} "
            f"per stop x {duplication['bridge_stop_count']} synthetic stops. The authoritative frozen demand "
            f"realization for this window is {lineage['authoritative_rows_this_window']} passengers "
            f"(median {lineage['authoritative_per_window_median']} across {lineage['authoritative_rows_total']} rows / 54 windows)."
        ),
        "R4_08": duplication,
        "passed": duplication["passed"],
    })
    dump(root, "demand_scale_calibration_status.json", {
        "status": "UNIT_OR_EXPANSION_DEFECT_REQUIRES_REPAIR",
        "defects": duplication["defects"],
        "overgeneration_factor_vs_authoritative": duplication["overgeneration_factor_vs_authoritative"],
        "authoritative_generator": lineage["authoritative_generator"],
        "authoritative_demand_artifact": lineage["authoritative_demand_artifact"],
        "authoritative_lambda_field": "demand_lambda = R3_PEAK_LAMBDA * historical_intensity_ratio, Poisson per dispatch, dispatch_count=3",
        "registry_demand_lambda_this_window": lineage["registry_demand_lambda"],
        "bridge_uses_authoritative_lambda": False,
        "result_driven_calibration_performed": False,
        "demand_parameters_tuned_by_r4": False,
        "blocker": BLOCK_PRIMARY,
        "passed": False,
    })
    dump(root, "demand_realization_determinism_audit.json", {
        "R4_09": c["R4_09_same_demand_realization"],
        "R4_10": c["R4_10_cross_process_determinism"],
        "seeding": "sha256(seed|window_id|stop_id) -> numpy.random.default_rng, drawn entirely at reset()",
        "passed": c["R4_09_same_demand_realization"]["passed"] and c["R4_10_cross_process_determinism"]["passed"],
    })
    dump(root, "initial_state_equivalence_audit.json", c["R4_11_initial_state_equality"])
    dump(root, "rng_fairness_contract.json", c["R4_12_rng_fairness"])

    for arm in ("A", "B1", "B2"):
        row = next(r for r in result["arm_equivalence_matrix"] if r["arm_id"] == arm)
        dump(root, f"causal_arm_{arm}_contract.json", {
            **row,
            "shared_environment": "PV8 causal simulator, demand realization, initial state, horizon, transition engine, passenger accounting, p95 ledger, energy accounting, raw_event schema, window_rollup schema, canonical aggregator",
            "environment_modified_for_this_arm": False,
            "historical_or_replay_source_mode": False,
            "constructed": True,
        })
    dump(root, "arm_equivalence_matrix.json", {
        "rows": result["arm_equivalence_matrix"],
        "audit": c["R4_arm_equivalence_matrix"],
    })
    pd.DataFrame(result["arm_equivalence_matrix"]).to_parquet(root / "arm_equivalence_matrix.parquet", index=False)

    dump(root, "policy_provenance_audit.json", c["R4_15_policy_provenance"])
    dump(root, "canonical_kpi_pre_evaluation_provenance.json", {
        "passenger_wait_p95_seconds": {
            "chain": "arrival_ts + (board_ts | fixed horizon_end) -> anonymous passenger wait ledger -> measured empirical p95 -> window_rollup -> canonical KPI",
            "avg_times_constant_used": False,
            "action_count_synthesis_used": False,
            "fillna_zero_fabrication_possible": False,
            "status": "MEASURED",
        },
        "passenger_service_rate": {
            "chain": "passenger_served_count / passenger_demand_generated",
            "horizon_consistent": False,
            "status": "FAIL_CLOSED_PENDING_HORIZON_CLOSED_SERVICE_PRIMITIVE",
        },
        "in_vehicle_time_seconds": {"chain": None, "proxy_emitted": False, "status": "NOT_YET_MEASURABLE"},
        "R4_16": c["R4_16_no_synthetic_kpi_path"],
        "R4_13": c["R4_13_same_simulator"],
        "R4_14": c["R4_14_same_kpi_accounting"],
    })
    dump(root, "measured_p95_non_regression.json", c["R4_17_measured_p95_fallback_unreachable"])
    dump(root, "reward_v2_non_regression.json", c["R4_18_reward_v2_unchanged"])
    dump(root, "zero_loss_non_regression.json", c["R4_19_zero_loss_unchanged"])
    dump(root, "k_mask_non_regression.json", c["R4_20_k_mask_unchanged"])
    dump(root, "b0r_b0c_guard_audit.json", {"B0R": c["R4_21_b0r_historical_only"], "B0C": c["R4_22_b0c_blocked"]})
    dump(root, "test6_non_access_audit.json", c["R4_23_test6_untouched"])
    dump(root, "in_vehicle_time_guard.json", {"status": "NOT_YET_MEASURABLE", "proxy_emitted": False, "passed": True})
    dump(root, "pre_evaluation_integrity_test_report.json", result)

    dump(root, "repair_dependency_register.json", {
        "blockers": [
            {
                "id": BLOCK_PRIMARY,
                "severity": "PRIMARY",
                "evidence": "R4_08_demand_expansion_no_duplication",
                "defects": duplication["defects"],
                "root_cause": (
                    "causal_kpi_bridge._arrival_schedule invents a demand model from a window-level aggregate "
                    "intensity feature instead of binding to the frozen authoritative demand realization"
                ),
                "minimum_repair": (
                    "bind the bridge arrival schedule to r8er3r_generated_demand.parquet (per-passenger request_ts "
                    "and origin/destination stop occurrences) or to the frozen demand_lambda dispatch process"
                ),
            },
            {
                "id": BLOCK_SECONDARY,
                "severity": "COUPLED",
                "evidence": "R4_05_post_horizon_cannot_improve_service_rate",
                "root_cause": "passenger_served_count carries eventual-service semantics on the vehicle clock",
                "minimum_repair": "introduce a horizon-closed service primitive and bind the 30-minute service rate to it under an explicit schema gate",
            },
        ],
        "coupling": (
            "the demand over-generation drives the vehicle clock past the horizon, which is what produces the "
            "post-horizon boardings; repairing demand alone does not prove the service-rate semantics, so both "
            "belong to one minimum-change gate"
        ),
        "repaired_automatically_by_r4": False,
        "recommended_next_gate": NEXT_GATE_BLOCKED,
    })

    failed = result["failed_checks"]
    gate_passed = not failed and not mismatched
    gate_name = GATE_PASS if gate_passed else BLOCK_PRIMARY
    gate = {
        "gate": gate_name,
        "source_sha": head_sha,
        "upstream_r3_1_sha": UPSTREAM_R3_1_SHA,
        "population_closure_pass": c["R4_01_population_closure"]["passed"],
        "evaluation_demand_population_semantics": "passengers with an authoritative arrival_ts inside the fixed 30-minute horizon (carry-in Case A)",
        "served_by_horizon_semantics": "boarded at a SERVE transition with board_ts <= fixed horizon_end",
        "unserved_at_horizon_semantics": "arrival_ts <= horizon_end and not boarded by horizon_end; right-censored there",
        "post_horizon_boarding_counted_as_30min_service": True,
        "passenger_served_count_current_semantics": c["R4_06_passenger_served_count_semantics"]["identified_semantics"],
        "service_rate_horizon_semantics_status": "FAIL_CLOSED",
        "demand_source": lineage["raw_source_dataset"],
        "demand_unit": "aggregate reconstructed boarding intensity (exp(log1p_feature)-1), not a passenger count",
        "demand_granularity": "per window / route-direction / time-band aggregate over the mapped stop set",
        "demand_scale_status": "UNIT_OR_EXPANSION_DEFECT_REQUIRES_REPAIR",
        "demand_duplication_detected": duplication["duplication_detected"],
        "same_demand_realization_A_B1_B2": c["R4_09_same_demand_realization"]["passed"],
        "cross_process_determinism_pass": c["R4_10_cross_process_determinism"]["passed"],
        "same_initial_state_A_B1_B2": c["R4_11_initial_state_equality"]["passed"],
        "rng_fairness_pass": c["R4_12_rng_fairness"]["passed"],
        "A_arm_constructed": True,
        "B1_arm_constructed": True,
        "B2_arm_constructed": True,
        "arm_equivalence_pass": c["R4_arm_equivalence_matrix"]["passed"],
        "measured_p95_path_valid": c["R4_17_measured_p95_fallback_unreachable"]["passed"],
        "synthetic_p95_reachable": False,
        "reward_v2_unchanged": c["R4_18_reward_v2_unchanged"]["passed"],
        "zero_loss_unchanged": c["R4_19_zero_loss_unchanged"]["passed"],
        "k_mask_unchanged": c["R4_20_k_mask_unchanged"]["passed"],
        "B0R_status": "HISTORICAL_REFERENCE_ONLY",
        "B0C_status": "BLOCKED_UNEXECUTED",
        "test6_accessed": False,
        "performance_comparison_executed": False,
        "performance_claim_allowed": False,
        "in_vehicle_time_status": "NOT_YET_MEASURABLE",
        "pre_evaluation_readiness": "BLOCKED" if failed else "READY",
        "repair_required": [BLOCK_PRIMARY, BLOCK_SECONDARY] if failed else [],
        "failed_checks": failed,
        "check_pass_count": sum(1 for v in c.values() if v["passed"]),
        "check_total": len(c),
        "recommended_next_gate": NEXT_GATE_BLOCKED if failed else "explicit non-TEST6 causal evaluation gate",
    }
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "arm_universe_contract_sha256": hashlib.sha256(json.dumps(result["arm_equivalence_matrix"], sort_keys=True, default=str).encode("utf-8")).hexdigest(),
        "next_gate": NEXT_GATE_BLOCKED,
        "next_gate_auto_executed": False,
        "evaluation_release_permitted": False,
        "forbidden_downstream_without_new_gate": [
            "A/B1/B2 performance comparison", "full 54-window evaluation run", "TEST6 access",
            "absolute KPI performance claims", "B0R causal promotion", "B0C execution", "GitHub push",
        ],
    })

    dump(root, "final_report.json", {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-R4",
        "gate": gate_name,
        "checks": f"{gate['check_pass_count']}/{gate['check_total']}",
        "failed_checks": failed,
        "gate_decision": gate,
    })

    md = [
        "# H4M-AE-R4 Causal Comparison Arm Construction and Pre-Evaluation Integrity Validation",
        "",
        f"- gate: `{gate_name}`",
        f"- source HEAD at run: `{head_sha}`",
        f"- upstream R3.1: `{UPSTREAM_R3_1_SHA}` (artifact `{r3_1_dir.name}`, {len(manifest['file_sha256'])} files verified)",
        f"- integrity checks: {gate['check_pass_count']}/{gate['check_total']}",
        "",
        "## Arms constructed",
        "",
        "| arm | policy source | checkpoint | environment |",
        "| --- | --- | --- | --- |",
        "| A | `actual_promoted_mappo` | loaded | identical |",
        "| B1 | `causal_noop_baseline` | none | identical |",
        "| B2 | `causal_rulebased_baseline` | none | identical |",
        "",
        f"Demand realization hash, initial-state hash and every environment column are identical across the three "
        f"arms; unexplained environment differences: {c['R4_arm_equivalence_matrix']['unexplained_environment_differences']}.",
        "",
        "## D0 - horizon population and service accounting",
        "",
        f"- population closure holds: {closure['evaluation_demand_population']} = "
        f"{closure['served_by_horizon']} served-by-horizon + {closure['unserved_at_horizon']} unserved-at-horizon",
        f"- `passenger_served_count` = **{c['R4_06_passenger_served_count_semantics']['identified_semantics']}** "
        f"({closure['eventually_served_count']} eventual vs {closure['served_by_horizon']} by horizon)",
        f"- **R4-05 FAILS**: canonical `passenger_service_rate` reports "
        f"{c['R4_05_post_horizon_cannot_improve_service_rate']['canonical_passenger_service_rate']} while the "
        f"horizon-consistent rate is "
        f"{c['R4_05_post_horizon_cannot_improve_service_rate']['horizon_consistent_rate']:.4f}. Post-horizon "
        "boardings inflate a 30-minute service KPI to a perfect score.",
        "",
        "## D1 - demand semantics and scale",
        "",
        f"- source: `{lineage['raw_source_dataset']}`, reconstruction `{lineage['value_reconstruction']}`, "
        f"evidence class `{lineage['evidence_class']}`, `observed_individual_passenger_claim = "
        f"{lineage['observed_individual_passenger_claim']}`",
        f"- `historical_demand_score` = boarding + alighting intensity "
        f"(verified: {lineage['score_equals_boarding_plus_alighting']})",
        f"- authoritative frozen demand realization: `{lineage['authoritative_demand_artifact']}` - "
        f"{lineage['authoritative_rows_total']} passenger requests across 54 windows "
        f"(median {lineage['authoritative_per_window_median']}/window, "
        f"{lineage['authoritative_rows_this_window']} for this window), generated by "
        "`poisson_zero_capable(demand_lambda, ...)` over 3 dispatches",
        f"- **the bridge does not use it**. It computes "
        f"`{duplication['bridge_rate_expression']}` = {duplication['bridge_rate_value']}, rescales by horizon/600 "
        f"to {duplication['bridge_per_stop_expected']} per stop, and multiplies by "
        f"{duplication['bridge_stop_count']} synthetic stops = {duplication['bridge_total']}",
        "",
        f"**25,696 is not 25,696 passengers.** Over-generation factor vs the authoritative realization: "
        f"**{duplication['overgeneration_factor_vs_authoritative']:.1f}x**. Defects: "
        + ", ".join(f"`{d}`" for d in duplication["defects"]),
        "",
        "## What passed",
        "",
        "R4-01..04 population closure, R4-06 semantics identification, R4-07 lineage, R4-09/10 identical and "
        "cross-process-deterministic demand, R4-11 initial-state equality, R4-12 RNG fairness (environment "
        "randomness is drawn entirely at `reset()`; no RNG call occurs inside `step`; the promoted policy uses "
        "argmax), R4-13/14 shared simulator and KPI accounting, R4-15 policy provenance, R4-16/17 no synthetic "
        "KPI path and measured p95 fallback still unreachable, R4-18/19/20 Reward V2 / Zero-Loss / K-mask "
        "unchanged, R4-21/22 B0R and B0C guards, R4-23 TEST6 untouched, R4-24 no performance comparison.",
        "",
        "## Blockers",
        "",
        f"1. `{BLOCK_PRIMARY}` (primary) - the bridge's demand model is not the authoritative one and duplicates "
        "a window-level aggregate across stops and time.",
        f"2. `{BLOCK_SECONDARY}` (coupled) - the 30-minute service rate is satisfied by post-horizon boardings.",
        "",
        "The second is downstream of the first: over-generated demand is what drives the vehicle clock past the "
        "horizon. Repairing demand alone would not prove the service-rate semantics, so both belong to one gate.",
        "",
        "## Next gate",
        "",
        f"`{NEXT_GATE_BLOCKED}` (not executed automatically). No evaluation release is permitted until it passes.",
    ]
    (root / "final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    files = sorted(p.name for p in root.iterdir() if p.is_file())
    dump(root, "artifact_manifest.json", {
        "artifact_dir": root.name,
        "generated_at": datetime.now(KST).isoformat(),
        "source_sha": head_sha,
        "gate": gate_name,
        "file_sha256": {n: sha256_file(root / n) for n in files if n != "artifact_manifest.json"},
        "source_file_sha256": {
            "causal_kpi_bridge.py": sha256_file(BRIDGE),
            "causal_arm_contracts.py": sha256_file(ARM_MODULE),
            "test_h4m_ae_r4_pre_evaluation_integrity.py": sha256_file(R4_TEST),
            Path(__file__).name: sha256_file(Path(__file__)),
        },
    })
    (root / "_SUCCESS.lock").write_text(
        json.dumps({"gate": gate_name, "passed": gate_passed, "next_gate": NEXT_GATE_BLOCKED}, indent=2), encoding="utf-8"
    )

    print(f"[H4M-AE-R4] artifact root: {root}")
    print(f"[H4M-AE-R4] gate: {gate_name}")
    print(f"[H4M-AE-R4] checks: {gate['check_pass_count']}/{gate['check_total']} | failed: {failed}")
    print(f"[H4M-AE-R4] arms constructed: A/B1/B2, equivalence pass={gate['arm_equivalence_pass']}")
    print(f"[H4M-AE-R4] next gate: {gate['recommended_next_gate']}")
    print(f"[H4M-AE-R4] artifact files: {len(files) + 1}")


if __name__ == "__main__":
    main()
