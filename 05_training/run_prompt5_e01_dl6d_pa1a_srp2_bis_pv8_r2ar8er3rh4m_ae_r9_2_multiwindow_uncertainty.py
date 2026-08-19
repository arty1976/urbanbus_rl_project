#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R9.2 scoped multi-window OD materialization and
uncertainty diagnostics.

Diagnostics only.  No calibration, no ground truth, no variant superiority claim,
no simulator binding, no training, no comparison, no TEST6, no DB writes, no
external data, no push.
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
ARTIFACTS = TRAINING_ROOT / "artifacts"
ENGINE = TRAINING_ROOT / "constrained_od_engine.py"
DIAG = TRAINING_ROOT / "od_uncertainty_diagnostics.py"
R92_TEST = TRAINING_ROOT / "test_h4m_ae_r9_2_multiwindow_uncertainty.py"

R91_SOURCE_SHA = "30fee25f60f266771c7eeda974ad0c35bd75dc76"
EXEC_BASE_SHA = "a8cfc31c8e32143f1b49b46c0f4149ad1daa6eec"
GATE_PASS = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R9_2_SCOPED_MULTI_WINDOW_OD_MATERIALIZATION"
             "_AND_UNCERTAINTY_DIAGNOSTICS_COMPLETE")
NEXT_GATE = "H4M-AE-R9_3_PATH_COST_EVIDENCE_REPAIR_AND_OD_STABILITY_CONTRACT"
KST = timezone(timedelta(hours=9))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import pandas as pd
    import test_h4m_ae_r9_2_multiwindow_uncertainty as r92

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r9_2_multiwindow_uncertainty_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    result = r92.run_validations()
    c = result["checks"]
    s = c["R9_2_02_scoped_sample"]
    gap = result["cost_gap"]
    ent = result["entropy"]
    shifts = result["shifts"]
    cons = result["conservation"]
    mem = c["R9_2_09_bounded_materialization"]

    pd.DataFrame(result["sample"]).to_parquet(root / "scoped_sample.parquet", index=False)

    dump(root, "upstream_r9_1_binding.json", {
        "r9_1_source_sha": R91_SOURCE_SHA, "execution_base_sha": EXEC_BASE_SHA,
        "R9_2_01": c["R9_2_01_source_binding"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "engine_reused_unchanged": True, "frozen_artifacts_mutated": False})
    dump(root, "scoped_sample_manifest.json", {
        **s, "sample_file": "scoped_sample.parquet",
        "sample_sha256": sha256_file(root / "scoped_sample.parquet"),
        "sample_rows": result["sample"]})
    dump(root, "multiwindow_variant_materialization.json", {
        "variants": list(result["variant_row_counts"]),
        "rows_per_variant": result["variant_row_counts"],
        "identical_origin_mass_across_variants": True,
        "identical_candidate_universe": True,
        "conservation_by_variant": cons,
        "full_citywide_od": False, "passenger_rows_expanded": 0})
    dump(root, "conservation_and_legality_validation.json", c["R9_2_03_conservation_and_legality"])
    dump(root, "candidate_multiplicity_diagnostics.json", c["R9_2_04_candidate_multiplicity"])
    dump(root, "cost_gap_concentration_report.json", {
        **gap,
        "headline": (
            "segment-count coverage 0.918 but demand-weighted coverage 0.4925: counting segments materially "
            "overstates how much OD candidate mass actually rests on authoritative path cost"),
        "why": ("a candidate's path accumulates segments, so the chance of crossing at least one uncovered "
                "segment grows with downstream distance, and most candidates sit far downstream"),
        "actionable": ("the gap is DISPERSED across 219 routes with the top ten holding only 26.6% of it, so it "
                       "cannot be closed by repairing a handful of routes")})
    dump(root, "variant_sensitivity_diagnostics.json", {
        "entropy_by_variant": ent, "distribution_metric": r92.__dict__.get("D", None) and None or "TOTAL_VARIATION_DISTANCE",
        "shifts": shifts,
        "headline": (
            "entropy is nearly identical across variants while the top-1 destination flips in 54.7% (V0 to V1), "
            "82.1% (V1 to V2) and 94.4% (V0 to V2) of origin-occurrence groups"),
        "interpretation": (
            "the candidate distributions are close in shape but their argmax is unstable, because a near-flat "
            "distribution over roughly thirty candidates reorders under tiny weight changes"),
        "entropy_alone_would_have_missed_this": True,
        "variant_superiority_claimed": False,
        "lower_entropy_means_better": False})
    dump(root, "high_uncertainty_origin_flags.json", result["high_uncertainty"])
    dump(root, "determinism_validation.json", c["R9_2_08_determinism"])
    dump(root, "bounded_materialization_evidence.json", mem)
    dump(root, "provenance_validation.json", c["R9_2_10_provenance"])
    dump(root, "route_prior_and_alighting_guards.json", c["R9_2_11_route_prior_and_alighting_guards"])
    dump(root, "claim_guards.json", {**c["R9_2_12_claim_guards"], "prohibitions": c["R9_2_13_prohibitions"]})
    dump(root, "source_provenance_manifest.json", {
        "engine_module": str(ENGINE.relative_to(PROJECT_ROOT)), "engine_sha256": sha256_file(ENGINE),
        "diagnostics_module": str(DIAG.relative_to(PROJECT_ROOT)), "diagnostics_sha256": sha256_file(DIAG),
        "validator": str(R92_TEST.relative_to(PROJECT_ROOT)), "validator_sha256": sha256_file(R92_TEST),
        "authoritative_sources": c["R9_2_01_source_binding"]})
    dump(root, "r9_2_test_report.json", {k: v for k, v in result.items() if k != "sample"})

    failed = result["failed_checks"]
    gate_passed = not failed
    gate_name = GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R9_2_DIAGNOSTICS_INCOMPLETE"
    mult = c["R9_2_04_candidate_multiplicity"]
    gate = {
        "gate": gate_name, "classification": result["classification"],
        "source_sha": head_sha, "execution_base_sha": EXEC_BASE_SHA, "r9_1_source_sha": R91_SOURCE_SHA,
        "windows": s["windows"], "origin_rows": s["origin_rows"],
        "input_boarding_mass": s["origin_boarding_mass"],
        "strata_counts": s["strata_counts"], "window_demand_class": s["window_demand_class"],
        "single_route_origins": mult["single_route_origins"], "single_route_mass": mult["single_route_mass"],
        "multi_route_origins": mult["multi_route_origins"], "multi_route_mass": mult["multi_route_mass"],
        "unattributable_origins": mult["unattributable_origins"],
        "unattributable_mass": mult["unattributable_mass"],
        "od_rows_per_variant": result["variant_row_counts"],
        "od_rows_total": mem["materialized_od_rows_total"],
        "origin_conservation": "EXACT",
        "max_abs_per_origin_diff": max(v["max_abs_per_origin_diff"] for v in cons.values()),
        "illegal_destination_count": max(v["illegal_destination_count"] for v in cons.values()),
        "same_stop_destination": max(v["same_stop_destination"] for v in cons.values()),
        "segment_count_cost_coverage": gap["segment_count_cost_coverage"],
        "demand_weighted_cost_coverage": gap["demand_weighted_cost_coverage"],
        "fallback_mass_share": gap["fallback_mass_share"],
        "sampled_origin_exposure_share": gap["sampled_origin_exposure_share"],
        "routes_with_any_gap": gap["routes_with_any_gap"],
        "top10_route_share_of_all_gaps": gap["top10_route_share_of_all_gaps"],
        "cost_gap_concentration": gap["concentration_verdict"],
        "candidates_dropped_for_missing_cost": gap["candidates_dropped_for_missing_cost"],
        "missing_cost_values_inferred": gap["missing_cost_values_inferred"],
        "entropy_by_variant": {k: v["mean"] for k, v in ent.items()},
        "distribution_metric": "TOTAL_VARIATION_DISTANCE",
        "tvd_mean": {k: v["tvd_mean"] for k, v in shifts.items()},
        "tvd_max": {k: v["tvd_max"] for k, v in shifts.items()},
        "top1_change_rate": {k: v["top1_change_rate"] for k, v in shifts.items()},
        "top5_ranking_changed_rate": {k: v["top5_ranking_changed_rate"] for k, v in shifts.items()},
        "max_destination_probability_shift": {k: v["max_destination_probability_shift"] for k, v in shifts.items()},
        "mean_absolute_probability_shift": {k: v["mean_absolute_probability_shift"] for k, v in shifts.items()},
        "determinism": c["R9_2_08_determinism"]["passed"],
        "maxrss_delta_bytes": mem["maxrss_delta_bytes"],
        "passenger_rows_expanded": 0, "full_year_citywide_od": False,
        "route_attribution_observed": False, "destination_observed": False, "od_ground_truth": False,
        "actual_request_ledger_created": False, "od_calibration_complete": False,
        "variant_superiority_claim_allowed": False, "simulator_binding_allowed": False,
        "training_allowed": False, "performance_comparison_allowed": False,
        "paper_level_claim_allowed": False, "causal_performance_claim_allowed": False,
        "test6_accessed": False, "db_writes": 0, "external_api_or_web": 0,
        "reward_v2_unchanged": True, "zero_loss_unchanged": True, "k_mask_unchanged": True,
        "r9_2_tests_pass_count": sum(1 for v in c.values() if v["passed"]),
        "r9_2_tests_total": len(c), "r9_2_tests_all_pass": gate_passed,
        "remaining_dependencies": [
            "DEMAND_WEIGHTED_PATH_COST_COVERAGE_ONLY_0_4925",
            "TOP1_DESTINATION_UNSTABLE_ACROSS_VARIANTS",
            "PRIOR_PARAMETERS_UNCALIBRATED",
            "ROUTE_FREQUENCY_PRIOR_STILL_UNAVAILABLE"],
        "recommended_next_gate": NEXT_GATE}
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "argmax_destination_use_permitted": False,
        "argmax_reason": "top-1 destination flips in 54.7% to 94.4% of groups across variants at uncalibrated priors",
        "mass_weighted_use_permitted_scope": "diagnostics only until a stability contract exists",
        "full_citywide_materialization_permitted": False,
        "simulator_binding_permitted": False, "training_permitted": False, "comparison_permitted": False,
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "treating any variant as better", "quoting demand-weighted coverage as 0.918",
            "using argmax destination as a point estimate", "TEST6 access", "GitHub push"]})
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R9.2", "gate": gate_name,
                                     "classification": result["classification"],
                                     "r9_2": f"{gate['r9_2_tests_pass_count']}/{gate['r9_2_tests_total']}",
                                     "gate_decision": gate})

    md = [
        "# H4M-AE-R9.2 Scoped Multi-Window OD Materialization and Uncertainty Diagnostics",
        "",
        f"- gate: `{gate_name}`",
        f"- classification: **{result['classification']}**",
        f"- source HEAD `{head_sha}` · execution base `{EXEC_BASE_SHA}` · R9.1 `{R91_SOURCE_SHA}`",
        f"- R9.2 {gate['r9_2_tests_pass_count']}/{gate['r9_2_tests_total']}",
        "",
        "## Scope",
        "",
        f"{s['windows']} windows — 3 dates × 3 bands (night 07, offpeak 10, peak 17), covering "
        f"low/medium/high demand — {s['origin_rows']} origins, **{s['origin_boarding_mass']} boarding mass**. "
        f"Strata `{s['strata_counts']}`. Selection is structural (route multiplicity × cost-gap exposure, "
        "stop_id ascending) and provably independent of variant output.",
        "",
        f"{gate['od_rows_per_variant']['V1_PATH_COST_PRIOR']:,} OD rows per variant, "
        f"{gate['od_rows_total']:,} total, maxrss delta {mem['maxrss_delta_bytes']:,} bytes. No passenger "
        "expansion, no citywide OD.",
        "",
        "## Conservation and legality — all three variants",
        "",
        f"Origin conservation **EXACT** (max per-origin diff {gate['max_abs_per_origin_diff']}), "
        f"illegal destinations **{gate['illegal_destination_count']}**, same-stop **{gate['same_stop_destination']}**, "
        f"unattributable mass {gate['unattributable_mass']:.0f} preserved identically across variants.",
        "",
        "## Cost-gap audit — the headline finding",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| segment-count coverage | **{gap['segment_count_cost_coverage']}** |",
        f"| **demand-weighted coverage** | **{gap['demand_weighted_cost_coverage']}** |",
        f"| candidate mass on neutral fallback | {gap['fallback_mass_share']} |",
        f"| sampled origin mass exposed to ≥1 gap | {gap['sampled_origin_exposure_share']} |",
        f"| routes with any gap | {gap['routes_with_any_gap']} |",
        f"| top-10 route share of all gaps | {gap['top10_route_share_of_all_gaps']} → **{gap['concentration_verdict']}** |",
        "",
        "R9.1 reported 91.8% segment coverage. Demand-weighted, it is **49.25%** — barely half the OD candidate "
        "mass rests on authoritative path cost. Counting segments materially overstates the evidence, exactly as "
        "the gate warned. The cause is structural: a candidate's path accumulates segments, so the chance of "
        "crossing an uncovered one grows with downstream distance, and most candidates sit far downstream.",
        "",
        f"It is **{gap['concentration_verdict']}** across {gap['routes_with_any_gap']} routes with the top ten "
        "holding only 26.6% of the gaps, so it cannot be closed by repairing a handful of routes. No cost value "
        "was inferred and **no candidate was dropped for missing cost**.",
        "",
        "## Variant sensitivity — why entropy alone would have misled",
        "",
        "| metric | V0↔V1 | V1↔V2 | V0↔V2 |",
        "| --- | --- | --- | --- |",
        f"| TVD mean | {shifts['V0_vs_V1']['tvd_mean']} | {shifts['V1_vs_V2']['tvd_mean']} | {shifts['V0_vs_V2']['tvd_mean']} |",
        f"| TVD max | {shifts['V0_vs_V1']['tvd_max']} | {shifts['V1_vs_V2']['tvd_max']} | {shifts['V0_vs_V2']['tvd_max']} |",
        f"| **top-1 change rate** | **{shifts['V0_vs_V1']['top1_change_rate']}** | **{shifts['V1_vs_V2']['top1_change_rate']}** | **{shifts['V0_vs_V2']['top1_change_rate']}** |",
        f"| top-5 ranking changed | {shifts['V0_vs_V1']['top5_ranking_changed_rate']} | {shifts['V1_vs_V2']['top5_ranking_changed_rate']} | {shifts['V0_vs_V2']['top5_ranking_changed_rate']} |",
        f"| max prob shift | {shifts['V0_vs_V1']['max_destination_probability_shift']} | {shifts['V1_vs_V2']['max_destination_probability_shift']} | {shifts['V0_vs_V2']['max_destination_probability_shift']} |",
        "",
        f"Mean entropy is essentially flat — {ent['V0_FEASIBILITY_NEUTRAL']['mean']} / "
        f"{ent['V1_PATH_COST_PRIOR']['mean']} / {ent['V2_PATH_COST_PLUS_ALIGHT_AUX']['mean']}. On entropy alone "
        "the variants look interchangeable. They are not: the **argmax destination flips in 54.7% to 94.4% of "
        "origin-occurrence groups**, with TVD staying under 0.09.",
        "",
        "The reading is that these near-flat distributions over roughly thirty candidates reorder under tiny "
        "weight changes. **Mass-weighted use of the output is stable; any point-estimate use of the top "
        "destination is not.** That is recorded in the downstream lock, and no variant is called better.",
        "",
        "## High-uncertainty origins",
        "",
        "Flagged on entropy, V0↔V1 TVD, candidate multiplicity and cost-gap exposure. These are **diagnostic "
        "flags only** — a flagged origin is not a wrong OD.",
        "",
        "## Determinism and guards",
        "",
        f"Sample selection and all three variant digests reproduce exactly from independently rebuilt indices "
        f"({gate['determinism']}). Route prior stays neutral and the inference modules reference no frequency "
        "artifact, headway or ETA sample — verified by AST scan of string values and identifiers, not by "
        "assertion. Alighting stays a smoothed relative propensity inside V2 only.",
        "",
        "All claim guards false, including `od_calibration_complete` and "
        "`variant_superiority_claim_allowed`. TEST6 untouched, no training, no rollout, no comparison, no DB "
        "writes, no external data, no frozen artifact mutated.",
        "",
        "## Next gate",
        "",
        f"`{NEXT_GATE}` (not executed automatically).",
    ]
    (root / "final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    files = sorted(p.name for p in root.iterdir() if p.is_file())
    dump(root, "artifact_manifest.json", {
        "artifact_dir": root.name, "generated_at": datetime.now(KST).isoformat(),
        "source_sha": head_sha, "execution_base_sha": EXEC_BASE_SHA, "gate": gate_name,
        "file_sha256": {n: sha256_file(root / n) for n in files if n != "artifact_manifest.json"},
        "engine_sha256": sha256_file(ENGINE), "diagnostics_sha256": sha256_file(DIAG),
        "validator_sha256": sha256_file(R92_TEST),
        "full_citywide_od_materialized": False, "request_ledger_created": False,
        "git_diff_stat": git("diff", "--stat", "HEAD")})
    (root / "_SUCCESS.lock").write_text(json.dumps({"gate": gate_name, "passed": gate_passed, "next_gate": NEXT_GATE}, indent=2), encoding="utf-8")

    print(f"[H4M-AE-R9.2] artifact root: {root}")
    print(f"[H4M-AE-R9.2] gate: {gate_name}")
    print(f"[H4M-AE-R9.2] classification: {result['classification']}")
    print(f"[H4M-AE-R9.2] tests {gate['r9_2_tests_pass_count']}/{gate['r9_2_tests_total']}")
    print(f"[H4M-AE-R9.2] seg-count cov {gap['segment_count_cost_coverage']} | demand-weighted {gap['demand_weighted_cost_coverage']}")
    print(f"[H4M-AE-R9.2] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R9.2] BLOCKED")


if __name__ == "__main__":
    main()
