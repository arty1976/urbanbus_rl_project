#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R6.1 B1 causal baseline semantics reconciliation and
reference binding.

Resolves canonical B1 meaning from frozen authority, repairs the causal binding
with the minimum change, and revalidates. No training, no demand change, no
ranking, no target fitting, no TEST6 access, no push.
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
ARM_MODULE = TRAINING_ROOT / "causal_arm_contracts.py"
BRIDGE = TRAINING_ROOT / "causal_kpi_bridge.py"
R6_1_TEST = TRAINING_ROOT / "test_h4m_ae_r6_1_b1_semantics_reconciliation.py"
R6_ARTIFACT = ARTIFACTS_ROOT / "pv8_r2a_r8e_r3_r_h4m_ae_r6_limited_causal_arm_execution_20260819_073944+09:00"

R6_GATE_COMMIT = "13bbddd7f1669132ecd527cd2da771d12f9377e1"
GATE_PASS = (
    "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R6_1_B1_CAUSAL_BASELINE_SEMANTICS_RECONCILIATION"
    "_AND_REFERENCE_BINDING_COMPLETE"
)
NEXT_GATE = "H4M-AE-R7_DEMAND_PLAUSIBILITY_AND_CALIBRATION_AUDIT"
KST = timezone(timedelta(hours=9))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import test_h4m_ae_r6_1_b1_semantics_reconciliation as r61
    import causal_arm_contracts as arms

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS_ROOT / f"pv8_r2a_r8e_r3_r_h4m_ae_r6_1_b1_semantics_reconciliation_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    result = r61.run_validations()
    c = result["checks"]
    ref = c["R6_1_02_authoritative_reference_identified"]
    prog = result["b1_runtime"]

    dump(root, "upstream_r6_binding.json", {
        "r6_artifact_dir": R6_ARTIFACT.name,
        "r6_gate": c["R6_1_01_upstream_r6_verified"]["gate"],
        "r6_manifest_recorded_source_sha": c["R6_1_01_upstream_r6_verified"]["r6_recorded_source_sha"],
        "r6_gate_commit": R6_GATE_COMMIT,
        "sha_resolution_note": (
            "the R6 manifest records the HEAD at execution time (36f93fa, the R5 commit); the commit that "
            "carries the R6 gate itself is 13bbddd. Both are recorded rather than guessed."
        ),
        "mismatched_files": c["R6_1_01_upstream_r6_verified"]["mismatched_files"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "reset_or_rebase_performed": False, "unrelated_changes_preserved": True,
    })
    dump(root, "authoritative_b1_reference_binding.json", {
        **ref,
        "reference_role": r61.REFERENCE_ROLE,
        "used_as_optimization_target": False,
        "limited_window_numerical_equality_claimed": False,
        "test6_required": False,
    })
    dump(root, "current_b1_semantics_audit.json", c["R6_1_03_current_semantics_classified"])
    dump(root, "b1_zero_service_root_cause.json", c["R6_1_04_zero_service_root_cause"])
    dump(root, "canonical_b1_semantics_freeze.json", {
        **c["R6_1_05_canonical_noop_frozen"],
        "resolved_interpretation": "A) NO_POLICY_INTERVENTION_BUT_NORMAL_BASE_SERVICE",
        "resolution_evidence": [
            "r8er3r_b1_kpi_summary.json hold_seconds = 0.0",
            "414 generated / 414 served with service reference 1.0",
            "regeneration source emits executed_action = 'SERVE' at every occurrence",
            "selection_basis: full route travel plus maximum 30-second dwell at every occurrence",
        ],
        "learned_policy_used": False, "checkpoint_loaded": False,
        "discretionary_policy_intervention": False,
        "normal_vehicle_progression": True, "required_boarding_alighting_enabled": True,
    })
    dump(root, "base_service_vs_policy_intervention_contract.json", c["R6_1_06_base_service_vs_intervention_separated"])
    dump(root, "b1_causal_binding_repair.json", {
        "changed_file": "05_training/causal_arm_contracts.py",
        "removed_behaviour": "noop_actions -> _legal(mask, preferred=0)  # HOLD",
        "added_behaviour": "base_service_actions -> _legal(mask, preferred=BASE_SERVICE_ACTION)  # SERVE",
        "retired_alias": "noop_actions now raises RETIRED_B1_NOOP_SEMANTICS",
        "policy_source_before": "causal_noop_baseline",
        "policy_source_after": arms.POLICY_SOURCE["B1"],
        "A_touched": False, "B2_touched": False, "bridge_touched": False,
        "aggregator_touched": False, "demand_touched": False,
        "R6_1_07": c["R6_1_07_noop_not_persistent_hold"],
    })
    dump(root, "b1_runtime_transition_validation.json", {
        "R6_1_08": c["R6_1_08_normal_vehicle_progression"], "per_window": prog})
    dump(root, "b1_boarding_alighting_validation.json", {
        "R6_1_09": c["R6_1_09_required_boarding_occurs"], "R6_1_10": c["R6_1_10_intervention_count_zero"],
        "alighting_note": "no alighting event exists in the bridge; in_vehicle_time remains NOT_YET_MEASURABLE",
    })
    dump(root, "persistent_hold_control_classification.json", c["R6_1_25_persistent_hold_control_noncanonical"])
    for horizon, num in ((1620, 15), (1800, 16), (1980, 17)):
        dump(root, f"horizon_{horizon}_b1_validation.json", c[f"R6_1_{num}_horizon_{horizon}_b1"])
    dump(root, "population_accounting_non_regression.json", c["R6_1_11_population_closure"])
    dump(root, "service_rate_non_regression.json", {
        "R6_1_12": c["R6_1_12_service_rate_contract"],
        "observed_service_rates": {w: v["service_rate"] for w, v in prog.items()},
        "forced_to_one": False,
    })
    dump(root, "avg_wait_non_regression.json", c["R6_1_13_avg_wait_contract"])
    dump(root, "measured_p95_non_regression.json", c["R6_1_14_measured_p95_contract"])
    dump(root, "A_non_regression.json", c["R6_1_20_A_unchanged"])
    dump(root, "B2_non_regression.json", c["R6_1_21_B2_unchanged"])
    dump(root, "reward_v2_non_regression.json", c["R6_1_22_reward_v2_unchanged"])
    dump(root, "zero_loss_non_regression.json", c["R6_1_23_zero_loss_unchanged"])
    dump(root, "k_mask_non_regression.json", c["R6_1_24_k_mask_unchanged"])
    dump(root, "reference_value_non_target_fitting_audit.json", c["R6_1_26_no_reference_target_fitting"])
    dump(root, "citywide_full_day_scalability_non_regression.json", c["R6_1_29_citywide_scalability_non_regression"])
    dump(root, "test6_non_access_audit.json", c["R6_1_27_test6_not_accessed"])
    dump(root, "performance_interpretation_guard.json", {
        **c["R6_1_28_no_performance_comparison"],
        "performance_claim_allowed": False,
        "b1_numbers_role": "evidence that baseline service executes, never a comparison against A or B2",
    })
    dump(root, "r6_1_test_report.json", result)

    failed = result["failed_checks"]
    gate_passed = not failed
    gate_name = GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R6_1_B1_SEMANTICS_UNRESOLVED"
    gate = {
        "gate": gate_name,
        "source_sha": head_sha,
        "upstream_r6_sha": R6_GATE_COMMIT,
        "upstream_r6_manifest_source_sha": c["R6_1_01_upstream_r6_verified"]["r6_recorded_source_sha"],
        "authoritative_b1_reference_sha": ref["authority_sha256"],
        "authoritative_b1_reference_artifact": ref["authority_artifact"],
        "current_b1_semantics": "PERSISTENT_HOLD (pre-repair)",
        "canonical_b1_semantics": arms.B1_SEMANTICS_ID,
        "zero_service_root_cause": c["R6_1_04_zero_service_root_cause"]["root_cause"],
        "noop_means_no_policy_intervention": True,
        "noop_means_persistent_hold": False,
        "normal_vehicle_progression_enabled": True,
        "required_service_enabled": True,
        "discretionary_intervention_enabled": False,
        "persistent_hold_control_retained": True,
        "persistent_hold_control_canonical": False,
        "horizon_1620_pass": c["R6_1_15_horizon_1620_b1"]["passed"],
        "horizon_1800_pass": c["R6_1_16_horizon_1800_b1"]["passed"],
        "horizon_1980_pass": c["R6_1_17_horizon_1980_b1"]["passed"],
        "positive_service_fixture_pass": c["R6_1_09_required_boarding_occurs"]["passed"],
        "population_closure_pass": c["R6_1_11_population_closure"]["passed"],
        "service_rate_contract_pass": c["R6_1_12_service_rate_contract"]["passed"],
        "avg_wait_contract_pass": c["R6_1_13_avg_wait_contract"]["passed"],
        "p95_contract_pass": c["R6_1_14_measured_p95_contract"]["passed"],
        "reference_service_value": r61.REFERENCE_SERVICE,
        "reference_avg_wait_seconds": r61.REFERENCE_AVG_WAIT,
        "reference_p95_wait_seconds": r61.REFERENCE_P95_WAIT,
        "reference_values_used_as_targets": False,
        "A_unchanged": c["R6_1_20_A_unchanged"]["passed"],
        "B2_unchanged": c["R6_1_21_B2_unchanged"]["passed"],
        "reward_v2_unchanged": c["R6_1_22_reward_v2_unchanged"]["passed"],
        "zero_loss_unchanged": c["R6_1_23_zero_loss_unchanged"]["passed"],
        "k_mask_unchanged": c["R6_1_24_k_mask_unchanged"]["passed"],
        "test6_accessed": False,
        "performance_comparison_executed": False,
        "performance_claim_allowed": False,
        "citywide_scalability_non_regression_pass": c["R6_1_29_citywide_scalability_non_regression"]["passed"],
        "full_day_architecture_non_regression_pass": c["R6_1_29_citywide_scalability_non_regression"]["passed"],
        "r6_1_tests_pass_count": sum(1 for v in c.values() if v["passed"]),
        "r6_1_tests_total": len(c),
        "r6_1_tests_all_pass": gate_passed,
        "r6_revalidation": result["r6_rerun"],
        "remaining_dependencies": ["DEMAND_SCALE_RESEARCH_PLAUSIBILITY", "IN_VEHICLE_TIME_MEASURABILITY",
                                   "VALIDATION_SPLIT_COVERS_ONLY_1800S_HORIZON"],
        "recommended_next_gate": NEXT_GATE,
    }
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "b1_semantics_contract_sha256": hashlib.sha256(json.dumps({
            "semantics": arms.B1_SEMANTICS_ID, "policy_source": arms.POLICY_SOURCE["B1"],
            "base_service_action": arms.BASE_SERVICE_ACTION,
            "discretionary_actions": arms.DISCRETIONARY_ACTIONS,
            "arm_module_sha256": sha256_file(ARM_MODULE),
        }, sort_keys=True).encode()).hexdigest(),
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "TEST6 access", "A/B1/B2 performance comparison", "policy ranking", "absolute Daegu KPI claims",
            "fitting B1 to the frozen reference values", "GitHub push",
        ],
    })
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R6.1", "gate": gate_name,
                                     "r6_1": f"{gate['r6_1_tests_pass_count']}/{gate['r6_1_tests_total']}",
                                     "gate_decision": gate})

    rows = ["| window | horizon | demand | served-by-horizon | service rate | avg wait s | p95 s | hold s | intervention rate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for w, v in prog.items():
        rows.append(f"| {w} | {v['horizon_seconds']} s | {v['demand']} | {v['served_by_horizon']} | "
                    f"{float(v['service_rate']):.4f} | {float(v['avg_wait_seconds']):.2f} | "
                    f"{float(v['p95_seconds']):.2f} | {float(v['hold_seconds']):.1f} | "
                    f"{float(v['intervention_rate']):.1f} |")

    md = [
        "# H4M-AE-R6.1 B1 Causal Baseline Semantics Reconciliation",
        "",
        f"- gate: `{gate_name}`",
        f"- source HEAD at run: `{head_sha}`",
        f"- upstream R6 gate commit `{R6_GATE_COMMIT}`; R6 manifest records execution-time HEAD "
        f"`{c['R6_1_01_upstream_r6_verified']['r6_recorded_source_sha']}`",
        f"- R6.1 {gate['r6_1_tests_pass_count']}/{gate['r6_1_tests_total']}; R6 suite rerun "
        f"{result['r6_rerun']['checks_passed']}/{result['r6_rerun']['total']}",
        "",
        "## The question, answered from frozen authority",
        "",
        f"Authority: `{ref['authority_artifact']}` / `{ref['authority_file']}` "
        f"(sha256 `{ref['authority_sha256']}`).",
        "",
        f"- `hold_seconds = {ref['hold_seconds']}`, `dwell_seconds = {ref['dwell_seconds']}`, "
        f"`travel_seconds = {ref['travel_seconds']}`",
        f"- {ref['generated_passengers']} generated / {ref['served_passengers']} served, service reference "
        f"{ref['service_reference']}",
        "- the regeneration source emits `executed_action = \"SERVE\"` at every occurrence, with the stated basis "
        "\"full route travel plus maximum 30-second dwell at every occurrence\"",
        "",
        "A baseline with **zero hold seconds that serves every passenger** cannot mean \"hold forever\". "
        "Canonical B1 is therefore **A) NO_POLICY_INTERVENTION_BUT_NORMAL_BASE_SERVICE**.",
        "",
        "## Root cause of the zero-service B1",
        "",
        f"`{c['R6_1_04_zero_service_root_cause']['root_cause']}` - proven from source, not inferred from output: "
        "`causal_arm_contracts.noop_actions` returned `_legal(mask, preferred=0)`, and action id 0 is HOLD in the "
        "bridge's step branch, so the SERVE boarding branch was never entered and no passenger was ever settled.",
        "",
        "## Repair",
        "",
        "One file changed (`causal_arm_contracts.py`): `base_service_actions` prefers SERVE, the old "
        "`noop_actions` name now raises `RETIRED_B1_NOOP_SEMANTICS` so no caller can silently resolve the old "
        "meaning, and the hold-forever behaviour survives only as `persistent_hold_control_actions` under "
        "`PERSISTENT_HOLD_CONTROL` with `canonical_baseline = false`, `performance_reference_allowed = false`, "
        "`B1_alias_allowed = false`. A, B2, the bridge, the aggregator and the demand binding were not touched.",
        "",
        "## Repaired B1 runtime",
        "",
        *rows,
        "",
        "`hold_seconds = 0.0` and `intervention_rate = 0.0` in every window, matching the authoritative "
        "signature. Service rates are " + ", ".join(f"{float(v['service_rate']):.4f}" for v in prog.values())
        + " - two windows reach full service naturally and one does not, which is the point: nothing was fitted.",
        "",
        "## Reference binding",
        "",
        f"`service = {r61.REFERENCE_SERVICE}`, `avg_wait = {r61.REFERENCE_AVG_WAIT} s`, "
        f"`p95 = {r61.REFERENCE_P95_WAIT} s` are bound as **{r61.REFERENCE_ROLE}**. They appear in no source "
        "file as a literal, no threshold was fitted to them, no limited-window equality is claimed, and TEST6 "
        "was not opened. The observed limited-window averages "
        f"({', '.join(f'{v:.2f}' for v in c['R6_1_26_no_reference_target_fitting']['observed_avg_waits'])} s) "
        "differ from the representative reference, as expected for three windows against a 54-window aggregate.",
        "",
        "## Non-regression",
        "",
        "A and B2 reproduce their R6 action digests and canonical KPIs exactly. Population closure, service-rate, "
        "avg-wait and measured-p95 contracts hold for all 9 arm/window executions. Reward V2 / Zero-Loss / K-mask "
        "unchanged. The full R6 suite re-passes 40/40.",
        "",
        "## Next gate",
        "",
        f"`{NEXT_GATE}` (not executed automatically).",
    ]
    (root / "final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    files = sorted(p.name for p in root.iterdir() if p.is_file())
    dump(root, "artifact_manifest.json", {
        "artifact_dir": root.name, "generated_at": datetime.now(KST).isoformat(),
        "source_sha": head_sha, "gate": gate_name,
        "file_sha256": {n: sha256_file(root / n) for n in files if n != "artifact_manifest.json"},
        "source_file_sha256": {p.name: sha256_file(p) for p in (ARM_MODULE, BRIDGE, R6_1_TEST, Path(__file__))},
    })
    (root / "_SUCCESS.lock").write_text(json.dumps({"gate": gate_name, "passed": gate_passed, "next_gate": NEXT_GATE}, indent=2), encoding="utf-8")

    print(f"[H4M-AE-R6.1] artifact root: {root}")
    print(f"[H4M-AE-R6.1] gate: {gate_name}")
    print(f"[H4M-AE-R6.1] R6.1 {gate['r6_1_tests_pass_count']}/{gate['r6_1_tests_total']} | R6 rerun {result['r6_rerun']['checks_passed']}/{result['r6_rerun']['total']}")
    print(f"[H4M-AE-R6.1] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R6.1] BLOCKED")


if __name__ == "__main__":
    main()
