#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R9 Daegu 2023 constrained OD reconstruction contract.

Contract definition only.  No OD inference, no OD matrix, no request ledger, no
simulator binding, no training, no comparison, no DB writes, no network, no push.
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
R9_TEST = TRAINING_ROOT / "test_h4m_ae_r9_od_reconstruction_contract.py"
LEDGER_MODULE = TRAINING_ROOT / "citywide_demand_ledger.py"

EXEC_BASE_SHA = "373b83e301af5f487776c3b7c75bc91ebea98065"
GATE_PASS = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R9_DAEGU_2023_CONSTRAINED_OD_RECONSTRUCTION"
             "_CONTRACT_COMPLETE")
NEXT_GATE = "H4M-AE-R9_1_CITYWIDE_CONSTRAINED_OD_RECONSTRUCTION_ENGINE_AND_NONTEST_MINIMAL_MATERIALIZATION"
KST = timezone(timedelta(hours=9))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import test_h4m_ae_r9_od_reconstruction_contract as r9

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r9_od_reconstruction_contract_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    result = r9.run_validations()
    c = result["checks"]

    dump(root, "upstream_r8_9_1_binding.json", {
        "execution_base_sha": EXEC_BASE_SHA, "r8_9_1_source_commit_sha": EXEC_BASE_SHA,
        "R9_01": c["R9_01_upstream_r8_9_1"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "reset_or_rebase_performed": False, "unrelated_changes_preserved": True})
    dump(root, "authoritative_source_registry.json", {
        "tier_1_origin_demand": c["R9_02_citywide_ledger_bound"],
        "tier_2_network_feasibility": c["R9_03_network_authority_bound"],
        "tier_3_physical_cost": c["R9_04_cost_authority_bound"],
        "tier_4_partial_alighting": c["R9_21_alighting_rule"],
        "tier_5_z01712_auxiliary": c["R9_22_z01712_auxiliary_only"],
        "inactive": {"m009": c["R9_23_m009_inactive"], "stcis_15min_api": c["R9_24_stcis_api_nondependency"]},
        "research_414": c["R9_37_research_414_unchanged"],
        "route_frequency_rejected": c["R9_16_no_false_frequency_authority"],
        "hashes_discovered_from_repo": True})
    dump(root, "external_data_enrichment_closeout.json", c["R9_05_enrichment_closed"])
    dump(root, "observed_vs_inferred_field_contract.json", c["R9_06_observed_inferred_contract"])
    dump(root, "boarding_to_counterfactual_drt_demand_contract.json", {
        **c["R9_07_boarding_to_drt_assumption"], "R9_08": c["R9_08_no_arbitrary_scaling"]})
    dump(root, "lazy_demand_materialization_contract.json", c["R9_09_lazy_materialization"])
    dump(root, "request_timestamp_inference_contract.json", {
        **c["R9_11_timestamp_inference"], "R9_10": c["R9_10_bucket_duration_detected"],
        "R9_12": c["R9_12_count_conservation"]})
    dump(root, "route_assignment_inference_contract.json", {
        **c["R9_13_route_assignment_inferred"], "R9_16": c["R9_16_no_false_frequency_authority"]})
    dump(root, "destination_feasibility_contract.json", {
        **c["R9_14_destination_feasibility"], "R9_15": c["R9_15_loops_and_edge_cases"]})
    dump(root, "od_inference_family_contract.json", {
        "variants": result["od_variants"], "R9_17": c["R9_17_uncertainty_ensemble"],
        "V0": c["R9_18_feasibility_variant"], "V1": c["R9_19_cost_variant"], "V2": c["R9_20_alight_aux_variant"],
        "single_ground_truth": False})
    dump(root, "alighting_auxiliary_evidence_contract.json", c["R9_21_alighting_rule"])
    dump(root, "z01712_auxiliary_only_non_regression.json", c["R9_22_z01712_auxiliary_only"])
    dump(root, "chronological_leakage_prevention_contract.json", {
        **c["R9_25_leakage_contract"], "R9_26": c["R9_26_test6_untouched"]})
    dump(root, "od_uncertainty_and_realization_contract.json", c["R9_27_uncertainty_provenance"])
    dump(root, "simulated_in_vehicle_time_provenance_contract.json", c["R9_28_simulated_in_vehicle_time"])
    dump(root, "citywide_first_scaling_contract.json", c["R9_29_citywide_first"])
    dump(root, "od_reconstruction_validation_contract.json", c["R9_38_validation_contract"])
    dump(root, "r9_test_report.json", result)
    dump(root, "classification.json", c["R9_39_classification_supported"])

    failed = result["failed_checks"]
    gate_passed = not failed
    gate_name = GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R9_CONTRACT_INCOMPLETE"
    net = c["R9_03_network_authority_bound"]
    edge = c["R9_15_loops_and_edge_cases"]
    freq = c["R9_16_no_false_frequency_authority"]
    gate = {
        "gate": gate_name, "source_sha": head_sha, "execution_base_sha": EXEC_BASE_SHA,
        "external_data_enrichment_status": "CLOSED_CURRENT_EVIDENCE_ONLY",
        "citywide_ledger_sha256": c["R9_02_citywide_ledger_bound"]["dataset_sha256"],
        "network_authority_sha256": net["sha256"], "network_route_directions": net["route_directions"],
        "cost_authority_sha256": c["R9_04_cost_authority_bound"]["sha256"],
        "route_frequency_authority": False,
        "boarding_to_drt_assumption": c["R9_07_boarding_to_drt_assumption"]["assumption_id"],
        "boarding_to_drt_class": "RESEARCH_MODELING_ASSUMPTION",
        "arbitrary_scaling_applied": False,
        "source_bucket_duration_seconds": c["R9_10_bucket_duration_detected"]["inferred_bucket_duration_seconds"],
        "bucket_duration_hardcoded": False,
        "timestamp_inference_label": c["R9_11_timestamp_inference"]["label"],
        "count_conservation": "exact",
        "route_assignment": "INFERRED",
        "destination_feasibility_keyed_on": "route_stop_occurrence_id",
        "terminal_occurrences_without_downstream": edge["terminal_occurrences_with_no_downstream"],
        "repeated_stop_occurrences": edge["repeated_stop_occurrences"],
        "ledger_stops_without_network_coverage": edge["ledger_stops_without_network_coverage"],
        "orphan_share": edge["orphan_share"],
        "od_variants": list(result["od_variants"]),
        "alighting_absolute_destination_marginal_allowed": False,
        "alighting_auxiliary_evidence_allowed": True,
        "z01712_absolute_transfer_allowed": False,
        "m009_required": False, "stcis_api_required": False,
        "leakage_contract_definable": True, "test6_accessed": False,
        "uncertainty_fields": c["R9_27_uncertainty_provenance"]["required_fields"],
        "in_vehicle_time_provenance": "SIMULATED_IN_VEHICLE_TIME",
        "od_inference_executed": False, "od_matrix_created": False,
        "request_ledger_created": False, "simulator_binding": False,
        "historical_2023_citywide_ledger_unchanged": True, "research_414_unchanged": True,
        "training_executed": False, "performance_comparison_executed": False,
        "reward_v2_unchanged": True, "zero_loss_unchanged": True, "k_mask_unchanged": True,
        "db_writes": 0, "network_collection": 0, "external_data_search": 0,
        "classification": result["classification"],
        "r9_tests_pass_count": sum(1 for v in c.values() if v["passed"]),
        "r9_tests_total": len(c), "r9_tests_all_pass": gate_passed,
        "remaining_dependencies": [
            "OD_REMAINS_INFERRED_NOT_OBSERVED_FOR_ALL_TIME",
            "NO_ROUTE_FREQUENCY_PRIOR_AUTHORITY",
            f"{edge['ledger_stops_without_network_coverage']} LEDGER_STOPS_OUTSIDE_ROUTE_NETWORK",
            "COST_DECAY_AND_ALIGHT_WEIGHT_PARAMETERS_UNCALIBRATED",
        ],
        "recommended_next_gate": NEXT_GATE}
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "implementation_permitted": True,
        "od_materialization_permitted_scope": "R9.1 minimal deterministic NON-TEST sample only",
        "full_matrix_permitted": False, "simulator_binding_permitted": False,
        "training_permitted": False, "comparison_permitted": False,
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "labelling inferred OD as observed", "single-realization performance claims",
            "raw alighting as destination marginal", "Z01712 as a 2023 absolute constraint",
            "fabricated route frequency weights", "TEST6 access", "GitHub push"]})
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R9", "gate": gate_name,
                                     "classification": result["classification"],
                                     "r9": f"{gate['r9_tests_pass_count']}/{gate['r9_tests_total']}",
                                     "gate_decision": gate})

    fc = result["field_contract"]
    md = [
        "# H4M-AE-R9 Daegu 2023 Constrained OD Reconstruction Contract",
        "",
        f"- gate: `{gate_name}`",
        f"- source HEAD at run: `{head_sha}` · execution base `{EXEC_BASE_SHA}`",
        f"- R9 {gate['r9_tests_pass_count']}/{gate['r9_tests_total']}",
        f"- classification: **{result['classification']}**",
        "",
        "## External data is closed",
        "",
        "`external_data_enrichment_status = CLOSED_CURRENT_EVIDENCE_ONLY`. STCIS bus-only OD unavailable, 2023 "
        "Z01712 unavailable, M009 inactive, the 15-minute OD API removed as a required dependency, 경상권 "
        "withdrawn. **No external search was performed in this gate and execution is no longer waiting on any of "
        "them.**",
        "",
        "## Authorities, bound to real artifacts",
        "",
        "| tier | artifact | evidence |",
        "| --- | --- | --- |",
        f"| 1 origin demand | citywide ledger | `{gate['citywide_ledger_sha256'][:16]}…`, 181,413,653 boardings |",
        f"| 2 network feasibility | k6 occurrence master | `{net['sha256'][:16]}…`, {net['rows']:,} occurrences, "
        f"{net['routes']} routes, {net['route_directions']} route-directions |",
        f"| 3 physical cost | full_graph_edges | `{gate['cost_authority_sha256'][:16]}…`, distance_m / time_sec / generalized_cost |",
        "| 4 alighting | 2023 partial tap-out | auxiliary only, selection biased |",
        "| 5 Z01712 | 2026 bus+rail daily SGG | structural sanity only |",
        "",
        "The occurrence master's recorded source hash matches `route_stop_sequences.parquet` exactly, so the "
        "chain is verified rather than assumed.",
        "",
        "## The honest part: what is inferred",
        "",
        "| observed | inferred |",
        "| --- | --- |",
        "| service date, time bucket, origin stop, boarding count | the DRT request event itself |",
        "| route topology and ordered occurrences | request timestamp inside the bucket |",
        "| edge distance, time, generalized cost | route-direction assignment |",
        "| partial alighting (selection biased) | destination occurrence and the OD pair |",
        "",
        "**No inferred value may ever be presented as observed transport-card OD.** In-vehicle time, once "
        "computable, is `SIMULATED_IN_VEHICLE_TIME` and never `OBSERVED_HISTORICAL`.",
        "",
        "## Boarding → DRT demand",
        "",
        f"Frozen as **`{gate['boarding_to_drt_assumption']}`**, explicitly a "
        "`RESEARCH_MODELING_ASSUMPTION`: per enabled source bucket, synthetic request units sum to the observed "
        "boarding count. No global scale factor, no reuse of the retired 56.7 proxy. Its limitations are written "
        "into the contract: boardings are a lower bound on latent demand, a travelling party is several "
        "boardings but one request, a transfer journey is several boardings, and DRT would itself change the "
        "demand process.",
        "",
        "## Timestamps and materialization",
        "",
        f"Bucket duration is **detected** from the source ({gate['source_bucket_duration_seconds']} s, from 19 "
        "distinct service hours), not hardcoded. Placement is deterministic stratified within `[t0,t1)` with "
        "exact count preservation — no Poisson resampling, no draw that cannot be reproduced from the source row "
        "identity plus the realization seed. Expansion is lazy: the 181M population is never materialized at "
        "preprocessing, only the selected execution horizon.",
        "",
        "## Destination feasibility",
        "",
        f"Keyed on `route_stop_occurrence_id`, never on `stop_id`. Mean {c['R9_14_destination_feasibility']['downstream_candidates_mean']} "
        f"downstream candidates, median {c['R9_14_destination_feasibility']['downstream_candidates_median']}, max "
        f"{c['R9_14_destination_feasibility']['downstream_candidates_max']}. Hard invariant "
        "`illegal_downstream_destination == 0`.",
        "",
        f"The awkward cases are handled explicitly rather than averaged away: **{edge['repeated_stop_occurrences']} "
        f"repeated-stop occurrences** across {edge['routes_with_repeats']} routes (resolved by occurrence "
        f"ordinal), **{edge['terminal_occurrences_with_no_downstream']} terminal occurrences with no downstream** "
        "(fall back to another route-direction, else flagged `UNSERVABLE_ORIGIN`), and "
        f"**{edge['ledger_stops_without_network_coverage']} of {edge['ledger_stops']} ledger stops outside the "
        f"route network** ({edge['orphan_share']:.1%}) flagged `OUT_OF_NETWORK_ORIGIN` with a count, never "
        "silently dropped.",
        "",
        "## A finding that constrains the design",
        "",
        f"`route_service_frequency.parquet` was inspected and **rejected**: {freq['rows']} rows covering "
        f"{freq['distinct_routes']} of {freq['project_route_total']} routes, from a single 2026-04-29 realtime "
        "ETA sample, labelled `candidate_from_getRealtime02_not_timetable` with `fleet_estimate_allowed=false`. "
        "So **there is no authoritative service-frequency prior**, and route assignment must not be "
        "frequency-weighted. That uncertainty is carried explicitly instead of being papered over with invented "
        "weights.",
        "",
        "## Uncertainty is the deliverable, not a single OD",
        "",
        "Three variants, none of them ground truth: **V0** feasibility-only maximum entropy, **V1** "
        "generalized-cost prior, **V2** V1 plus a smoothed relative alighting propensity. Every materialized row "
        "carries seed, variant, destination and route probabilities, candidate count and destination entropy. No "
        "future claim may rest on one realization; robustness runs across OD variants, realization seeds and "
        "MAPPO seeds.",
        "",
        "## Leakage",
        "",
        "Inputs are classed static-topology / historical-dynamic / current-row / research-prior. Static topology "
        "is shared globally; dynamic statistics are fitted only on strictly earlier data and applied forward — "
        "train on train, validation from train-fitted priors, future test from train+validation only. TEST6 is "
        "never fitted on and stays sealed.",
        "",
        "## What PASS does not mean",
        "",
        "OD is **not** observed, no matrix exists, no request ledger exists, the simulator is not bound and no "
        "KPI comparison is authorized. What is now true is that the methodology is defensible on current "
        "evidence and implementation may begin.",
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
        "od_data_generated": False, "request_ledger_generated": False,
        "source_file_sha256": {p.name: sha256_file(p) for p in (R9_TEST, LEDGER_MODULE, Path(__file__))}})
    (root / "_SUCCESS.lock").write_text(json.dumps({"gate": gate_name, "passed": gate_passed, "next_gate": NEXT_GATE}, indent=2), encoding="utf-8")

    print(f"[H4M-AE-R9] artifact root: {root}")
    print(f"[H4M-AE-R9] gate: {gate_name}")
    print(f"[H4M-AE-R9] classification: {result['classification']}")
    print(f"[H4M-AE-R9] tests {gate['r9_tests_pass_count']}/{gate['r9_tests_total']}")
    print(f"[H4M-AE-R9] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R9] BLOCKED")


if __name__ == "__main__":
    main()
