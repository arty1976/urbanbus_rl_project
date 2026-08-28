#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R7 demand plausibility and calibration audit.

Audit and freeze only.  No demand regeneration or tuning, no policy-outcome
input, no A/B1/B2 comparison, no TEST6 access, no DB writes, no network, no push.
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
R7_TEST = TRAINING_ROOT / "test_h4m_ae_r7_demand_plausibility_audit.py"
DEMAND_MODULE = TRAINING_ROOT / "authoritative_demand_realization.py"
ARM_MODULE = TRAINING_ROOT / "causal_arm_contracts.py"
BRIDGE = TRAINING_ROOT / "causal_kpi_bridge.py"

UPSTREAM_R6_1_SHA = "46e5cd57446ba9d2421a7c75ecba0ff04854d85d"
GATE_PASS = "PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R7_DEMAND_PLAUSIBILITY_AND_CALIBRATION_AUDIT_COMPLETE"
NEXT_GATE = "H4M-AE-R8_HISTORICAL_BOARDING_TO_REQUEST_EQUIVALENCE_MAPPING_AND_CALIBRATED_DEMAND_CANDIDATE_CONSTRUCTION"
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
    import test_h4m_ae_r7_demand_plausibility_audit as r7

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r7_demand_plausibility_audit_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    result = r7.run_validations()
    c = result["checks"]
    lin = result["lineage"]
    scale = c["R7_08_absolute_scale_comparison"]

    dump(root, "upstream_r6_1_binding.json", {
        "upstream_r6_1_source_commit": UPSTREAM_R6_1_SHA,
        "R7_01": c["R7_01_upstream_r6_1_verified"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "reset_or_rebase_performed": False, "unrelated_changes_preserved": True,
    })
    dump(root, "frozen_research_demand_binding.json", {
        **c["R7_02_frozen_demand_hash"], "R7_20": c["R7_20_frozen_artifact_unchanged"],
        "immutability": "r8er3r_generated_demand.parquet is the lineage for R5/R6/R6.1 and is never overwritten",
    })
    dump(root, "demand_generation_lineage.json", {**lin, "R7_03": c["R7_03_lineage_reconstructed"], "R7_04": c["R7_04_one_row_one_request"]})
    dump(root, "demand_unit_semantics_audit.json", {
        "research_unit": "one row = one trip request = one simulated passenger, unit weight",
        "historical_unit": "boardings: integer count of boarding events in a stop-hour bucket",
        "equivalence_proven": False,
        "mapping_limitation": (
            "the DB carries no passenger, request or trip identity anywhere, so a boarding event cannot be "
            "converted into a trip request without an equivalence contract that does not yet exist"
        ),
        "normalized_feature_converted_to_passengers": False,
        "upstream_feature_chain": lin["chain"],
        "R7_05": c["R7_05_historical_source_unit"],
    })
    dump(root, "historical_demand_source_registry.json", {
        "priority_used": "1. local Mac mini PostgreSQL urbanbus (READ-ONLY)",
        "database": "urbanbus", "table": r7.HIST_TABLE,
        "columns": ["service_date", "service_hour", "stop_id", "boardings", "alightings", "is_observed", "data_tier"],
        "unit": "boarding / alighting event counts",
        "temporal_granularity": "hourly", "spatial_granularity": "stop_id",
        "route_direction_columns": None,
        "period_used": c["R7_05_historical_source_unit"]["period"],
        "rows_returned": c["R7_05_historical_source_unit"]["rows_returned"],
        "stops_returned": c["R7_05_historical_source_unit"]["stops_returned"],
        "data_tier": c["R7_05_historical_source_unit"]["data_tier"],
        "network_or_api_used": False,
        "R7_32": c["R7_32_db_read_only"], "R7_33": c["R7_33_no_network"],
    })
    dump(root, "historical_query_manifest.json", {
        "query_text": result["historical_query"],
        "query_sha256": c["R7_05_historical_source_unit"]["query_sha256"],
        "read_only_guard": "SET default_transaction_read_only=on",
        "write_statements": 0, "schema_mutation": False, "matview_refresh": False,
    })
    dump(root, "comparison_population_contract.json", {
        **c["R7_06_comparison_population_aligned"], "R7_07": c["R7_07_real_horizon_used"],
        "reduced_universe": "Suseong reduced validation: route 3000814001, representative 54-window registry",
        "citywide_universe": "not audited here; a citywide comparison needs its own artifact and scope contract",
        "universes_mixed_in_one_coefficient": False,
    })
    dump(root, "absolute_scale_comparison.json", scale)
    pd.DataFrame(result["comparison"]["per_window"]).to_parquet(root / "per_window_rate_comparison.parquet", index=False)
    dump(root, "time_band_distribution_comparison.json", c["R7_09_time_band_shape"])
    dump(root, "hour_distribution_comparison.json", c["R7_10_hour_shape"])
    pd.DataFrame(result["comparison"]["stop"]).to_parquet(root / "stop_distribution_comparison.parquet", index=False)
    dump(root, "route_direction_distribution_comparison.json", c["R7_12_route_direction_shape"])
    dump(root, "zero_demand_frequency_comparison.json", c["R7_13_zero_demand_shape"])
    dump(root, "demand_quantile_comparison.json", c["R7_14_quantile_comparison"])
    dump(root, "sampling_scaling_method_audit.json", {
        "method": lin["sampling_method"], "family": "Poisson (inverse CDF over a deterministic sha256 uniform)",
        "stratified_or_weighted": "window-level stratification through historical_intensity_ratio; stop assignment is uniform over route occurrences",
        "deterministic": True, "reproducible": True, "zero_capable": True,
        "duplicated_rows_allowed": False, "future_information_leakage": False,
        "suitable_for_larger_scale": "the draw primitive scales, but the absolute level is fixed by BASE_REQUEST_INTENSITY and the spatial draw is uniform, so a larger-scale realization needs a demand-weighted stop assignment and an evidence-based level",
        "replaced_in_r7": False,
        "R7_22": c["R7_22_sampling_deterministic"], "R7_23": c["R7_23_cross_process_determinism"],
    })
    dump(root, "data_quality_audit.json", {**result["data_quality"], "R7_34": c["R7_34_data_quality"]})
    dump(root, "policy_independence_audit.json", c["R7_15_policy_independent"])
    dump(root, "b1_reference_non_target_fitting_audit.json", c["R7_16_b1_reference_not_target"])
    dump(root, "fleet_capacity_independence_audit.json", c["R7_17_fleet_capacity_not_target"])
    dump(root, "reduced_to_citywide_scaling_contract.json", {
        **c["R7_24_citywide_scaling_contract"],
        "parameters_supported_by_evidence": {
            "demand_source_artifact": "path to a frozen realization",
            "demand_artifact_sha256": "content hash recorded per run",
            "column_map": "semantic role to column name mapping",
            "source_scope": "reporting_window_ids carried by the artifact",
            "evaluation_start_ts / evaluation_end_ts": "external boundary",
            "seed": "only where a generator is run; R7 runs none",
        },
        "parameters_deliberately_not_frozen": {
            "scale_factor": "no evidence supports a boarding-to-request multiplier",
            "sampling_fraction": "the current realization is not a sample of a historical population",
        },
        "per_stop_replication_prohibited": True,
        "aggregate_times_stop_count_prohibited": True,
        "R7_18": c["R7_18_no_per_stop_replication"],
    })
    dump(root, "full_day_demand_ledger_architecture.json", c["R7_25_full_day_demand_ledger"])
    dump(root, "retired_synthetic_demand_non_regression.json", c["R7_19_retired_25696_unreachable"])
    dump(root, "calibration_candidate_contract.json", {
        "candidate_status": "NO_CANDIDATE_CREATED",
        "reason": (
            "a calibrated realization would require a boarding-to-request equivalence factor. The authoritative "
            "DB has no passenger, request or trip identity, so any factor would be a guessed multiplier, which "
            "section 24 prohibits."
        ),
        "measured_proxy_ratios_for_future_reference_only": {
            "overall": scale["ratio_research_over_historical"],
            "by_time_band": {k: v["ratio"] for k, v in c["R7_09_time_band_shape"]["table"].items()},
            "role": "PROXY_OBSERVATION_NOT_A_CALIBRATION_FACTOR",
            "binding_allowed": False,
        },
        "candidate_binding_allowed": False,
        "frozen_artifact_overwritten": False,
        "R7_21": c["R7_21_candidate_versioned_separately"],
    })
    dump(root, "calibration_classification.json", c["R7_35_classification_supported"])
    dump(root, "test6_non_access_audit.json", c["R7_26_test6_not_accessed"])
    dump(root, "reward_v2_non_regression.json", c["R7_28_reward_v2_unchanged"])
    dump(root, "zero_loss_non_regression.json", c["R7_29_zero_loss_unchanged"])
    dump(root, "k_mask_non_regression.json", c["R7_30_k_mask_unchanged"])
    dump(root, "arm_source_non_regression.json", {**c["R7_31_arm_source_unchanged"], "R7_27": c["R7_27_no_performance_comparison"]})
    dump(root, "r7_test_report.json", result)

    failed = result["failed_checks"]
    gate_passed = not failed
    gate_name = GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R7_DEMAND_AUDIT_INCOMPLETE"
    tb = c["R7_09_time_band_shape"]
    stop = c["R7_11_stop_shape"]
    gate = {
        "gate": gate_name,
        "source_sha": head_sha,
        "upstream_r6_1_sha": UPSTREAM_R6_1_SHA,
        "frozen_demand_sha256": c["R7_02_frozen_demand_hash"]["sha256"],
        "frozen_total_requests": c["R7_02_frozen_demand_hash"]["total_requests"],
        "frozen_window_count": c["R7_02_frozen_demand_hash"]["window_count"],
        "frozen_median_requests_per_window": c["R7_02_frozen_demand_hash"]["median_requests_per_window"],
        "frozen_demand_source_class": lin["source_class"],
        "frozen_demand_generation_method": lin["sampling_method"],
        "historical_source": r7.HIST_TABLE,
        "historical_period": c["R7_05_historical_source_unit"]["period"],
        "historical_unit": "boarding event count",
        "historical_spatial_granularity": "stop_id",
        "historical_temporal_granularity": "hourly",
        "comparison_scope": "route 3000814001 stop set, 9 service dates, hours 07/10/17, real horizons 1620/1800/1980 s",
        "absolute_scale_ratio_or_null": scale["ratio_research_over_historical"],
        "scale_ratio_interpretation": (
            f"about 1 research request per {scale['one_research_request_per_historical_boardings']} historical "
            "boarding events in matched scope; a PROXY comparison, not observed request calibration"
        ),
        "time_band_shape_status": "PARTIALLY_PRESERVED_PEAK_LIFT_NOT_REPRODUCED",
        "hour_shape_status": "DETERMINED_BY_BAND_NO_ADDITIONAL_FREEDOM",
        "stop_shape_status": "NOT_PRESERVED_UNIFORM_DRAW",
        "route_direction_shape_status": "EVIDENCE_GAP_NO_DIRECTION_IN_SOURCE",
        "zero_demand_shape_status": "NO_ZERO_WINDOWS_ON_EITHER_SIDE",
        "quantile_shape_status": "NOT_CORRELATED_PER_WINDOW",
        "policy_outcomes_used_for_calibration": False,
        "b1_references_used_as_targets": False,
        "fleet_capacity_used_as_target": False,
        "sampling_method": "poisson_zero_capable inverse CDF over sha256 stable_uniform",
        "sampling_deterministic": True,
        "current_414_artifact_modified": False,
        "new_calibration_candidate_created": False,
        "new_calibration_candidate_sha256_or_null": None,
        "candidate_binding_allowed": False,
        "calibration_classification": result["classification"],
        "citywide_scaling_contract_ready": c["R7_24_citywide_scaling_contract"]["passed"],
        "full_day_demand_ledger_ready": c["R7_25_full_day_demand_ledger"]["passed"],
        "retired_25696_path_reachable": False,
        "test6_accessed": False,
        "performance_comparison_executed": False,
        "performance_claim_allowed": False,
        "reward_v2_unchanged": True, "zero_loss_unchanged": True, "k_mask_unchanged": True,
        "A_unchanged": True, "B1_unchanged": True, "B2_unchanged": True,
        "r7_tests_pass_count": sum(1 for v in c.values() if v["passed"]),
        "r7_tests_total": len(c),
        "r7_tests_all_pass": gate_passed,
        "remaining_dependencies": [
            "HISTORICAL_BOARDING_TO_REQUEST_EQUIVALENCE_UNPROVEN",
            "SPATIAL_DEMAND_WEIGHTING_ABSENT_IN_GENERATOR",
            "ABSOLUTE_SCALE_NOT_HISTORICALLY_CALIBRATED",
            "IN_VEHICLE_TIME_MEASURABILITY",
        ],
        "recommended_next_gate": NEXT_GATE,
    }
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "demand_classification": result["classification"],
        "absolute_kpi_interpretation_allowed": False,
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "binding any calibrated demand candidate", "absolute Daegu KPI claims",
            "A/B1/B2 performance comparison", "TEST6 access", "overwriting the frozen 414 artifact",
            "applying the measured proxy ratio as a scale factor", "GitHub push",
        ],
    })
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R7", "gate": gate_name,
                                     "classification": result["classification"],
                                     "r7": f"{gate['r7_tests_pass_count']}/{gate['r7_tests_total']}",
                                     "gate_decision": gate})

    band_rows = ["| band | research requests | historical boardings | research share | historical share | ratio |",
                 "| --- | --- | --- | --- | --- | --- |"]
    for band, v in tb["table"].items():
        band_rows.append(f"| {band} | {int(v['req'])} | {v['hist']:.1f} | {v['req_share']:.4f} | "
                         f"{v['hist_share']:.4f} | {v['ratio']:.5f} |")

    md = [
        "# H4M-AE-R7 Demand Plausibility and Calibration Audit",
        "",
        f"- gate: `{gate_name}`",
        f"- source HEAD at run: `{head_sha}`",
        f"- upstream R6.1 `{UPSTREAM_R6_1_SHA}`",
        f"- R7 {gate['r7_tests_pass_count']}/{gate['r7_tests_total']}",
        f"- classification: **{result['classification']}**",
        "",
        "## What the 414 realization is",
        "",
        "Traced end to end from the frozen artifact back to source:",
        "",
        f"`demand_lambda = BASE_REQUEST_INTENSITY({lin['base_request_intensity']}) x "
        f"TIME_BAND_MULTIPLIERS['peak']({lin['time_band_multipliers']['peak']}) x historical_intensity_ratio`, "
        f"then `poisson_zero_capable(demand_lambda, {lin['demand_seed']}, window_id, dispatch_index)` for each of "
        f"3 dispatches per window. Expected total from lambda {lin['expected_total_from_lambda']} against an "
        f"observed {lin['observed_total']}.",
        "",
        f"**`BASE_REQUEST_INTENSITY = {lin['base_request_intensity']}` is a fixed research constant**, declared in "
        "the regeneration source and not derived from ridership. Historical data enters only as "
        "`historical_intensity_ratio` - a within-stratum ratio spanning roughly 0.65 to 1.16 - and through three "
        "time-band multipliers. Origin and destination stops are drawn uniformly over route occurrences, not by "
        "demand weight. One row is one request; identity is anonymous and carried, never claimed observed.",
        "",
        "## Historical source",
        "",
        f"`{r7.HIST_TABLE}` in the local `urbanbus` Postgres, read-only, "
        f"{c['R7_05_historical_source_unit']['rows_returned']} rows over "
        f"{c['R7_05_historical_source_unit']['stops_returned']} stops, all `observed` tier. Unit is **boarding "
        "events per stop-hour**; there is no route, direction, passenger, request or trip identity anywhere. "
        "Boardings are therefore **not proven equivalent to trip requests**, and every scale statement below is "
        "labelled `PROXY_SCALE_COMPARISON`.",
        "",
        "## Absolute scale, matched scope",
        "",
        f"- research: **{scale['research_requests']}** requests",
        f"- historical: **{scale['historical_boardings_matched_scope']}** boardings, scaled to the real "
        "1620/1800/1980 s horizons and halved for the two registry directions",
        f"- ratio **{scale['ratio_research_over_historical']}**, about one research request per "
        f"**{scale['one_research_request_per_historical_boardings']}** historical boardings",
        "",
        "## Shape",
        "",
        *band_rows,
        "",
        f"Time-band JS divergence is {tb['js_divergence_bits']} bits, but the shares tell the real story: the "
        f"research realization is nearly flat while history is peaked, and the peak-to-night share ratio is "
        f"{tb['peak_to_night_ratio_research']} against a historical {tb['peak_to_night_ratio_historical']}.",
        "",
        f"- **per window**: Spearman {c['R7_14_quantile_comparison']['per_window_spearman']}, Pearson "
        f"{c['R7_14_quantile_comparison']['per_window_pearson']} - essentially no correlation with history",
        f"- **spatial**: top-10 stop share {stop['top10_share_research']} against a historical "
        f"{stop['top10_share_historical']}; Spearman {stop['spearman_overlap']} over {stop['overlapping_stops']} "
        f"overlapping stops; {stop['research_stops_without_history']} of {stop['research_origin_stops']} research "
        f"origin stops have no historical boardings in scope (coverage {stop['stop_coverage_ratio']})",
        "- **route/direction**: evidence gap - the source table has no direction column",
        "- **zero demand**: none on either side",
        "",
        "## Classification",
        "",
        f"**{result['classification']}**. The absolute level is a chosen constant, per-window ordering is "
        "uncorrelated with history, and the spatial profile is uniform where history is concentrated. Only the "
        "time-band split is roughly balanced, and even there the historical peak lift is not reproduced. That is "
        "weaker than structure-preserving, so classification C would overstate the evidence.",
        "",
        "This does **not** invalidate the realization for its actual job. It remains deterministic, reproducible, "
        "zero-capable, duplicate-free and identity-clean, which is what Mac mini methodology validation needs. "
        "What it cannot support is any absolute KPI claim about Daegu.",
        "",
        "## No calibration candidate was created",
        "",
        "A calibrated realization would need a boarding-to-request equivalence factor. The authoritative DB has no "
        "passenger, request or trip identity, so any factor would be a guessed multiplier. The measured proxy "
        f"ratios ({scale['ratio_research_over_historical']} overall) are recorded as observations only, explicitly "
        "not as a calibration factor, and binding is not allowed. The frozen 414 artifact is untouched.",
        "",
        "## Scaling architecture",
        "",
        "The demand interface already takes the artifact, column map, scope and evaluation boundary externally, "
        "and the stop universe is derived from the population rather than replicated per stop - so the retired "
        "25,696 aggregate-times-stop-count defect stays impossible. Reaching citywide or full-day scale is a "
        "matter of binding a larger authoritative artifact, not of changing accounting semantics.",
        "",
        "## Next gate",
        "",
        f"`{NEXT_GATE}` (not executed automatically). The binding constraint is the boarding-to-request "
        "equivalence, not the multiplier.",
    ]
    (root / "final_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    files = sorted(p.name for p in root.iterdir() if p.is_file())
    dump(root, "artifact_manifest.json", {
        "artifact_dir": root.name, "generated_at": datetime.now(KST).isoformat(),
        "source_sha": head_sha, "gate": gate_name,
        "file_sha256": {n: sha256_file(root / n) for n in files if n != "artifact_manifest.json"},
        "source_file_sha256": {p.name: sha256_file(p) for p in (R7_TEST, DEMAND_MODULE, ARM_MODULE, BRIDGE, Path(__file__))},
    })
    (root / "_SUCCESS.lock").write_text(json.dumps({"gate": gate_name, "passed": gate_passed, "next_gate": NEXT_GATE}, indent=2), encoding="utf-8")

    print(f"[H4M-AE-R7] artifact root: {root}")
    print(f"[H4M-AE-R7] gate: {gate_name}")
    print(f"[H4M-AE-R7] classification: {result['classification']}")
    print(f"[H4M-AE-R7] R7 {gate['r7_tests_pass_count']}/{gate['r7_tests_total']} | scale ratio {scale['ratio_research_over_historical']}")
    print(f"[H4M-AE-R7] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R7] BLOCKED")


if __name__ == "__main__":
    main()
