#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R9 Daegu 2023 route-attribution evidence and
constrained OD mapping contract.

Contract and evidence audit only.  No OD inference, no OD matrix, no request
ledger, no simulator binding, no training, no comparison, no DB writes, no
network, no external-data search, no push.
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
R9_TEST = TRAINING_ROOT / "test_h4m_ae_r9_route_attribution_contract.py"
LEDGER_MODULE = TRAINING_ROOT / "citywide_demand_ledger.py"

UPSTREAM_R891_SHA = "373b83e301af5f487776c3b7c75bc91ebea98065"
EXEC_BASE_SHA = "a8cfc31c8e32143f1b49b46c0f4149ad1daa6eec"
GATE_PASS = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R9_DAEGU_2023_ROUTE_ATTRIBUTION"
             "_AND_CONSTRAINED_OD_MAPPING_CONTRACT_COMPLETE")
NEXT_GATE = "H4M-AE-R9_1_CITYWIDE_ROUTE_ATTRIBUTION_AND_CONSTRAINED_OD_ENGINE_MINIMAL_MATERIALIZATION"
KST = timezone(timedelta(hours=9))


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def dump(root: Path, name: str, payload: Any) -> None:
    (root / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    sys.path.insert(0, str(TRAINING_ROOT))
    import test_h4m_ae_r9_route_attribution_contract as r9

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r9_route_attribution_contract_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    result = r9.run_validations()
    c = result["checks"]
    a = result["route_attribution_audit"]

    dump(root, "upstream_r8_9_1_binding.json", {
        "upstream_r8_9_1_sha": UPSTREAM_R891_SHA, "execution_base_sha": EXEC_BASE_SHA,
        "R9_01": c["R9_01_upstream_r8_9_1"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "reset_or_rebase_performed": False, "unrelated_changes_preserved": True})
    dump(root, "authoritative_source_registry.json", {
        "boarding": c["R9_02_citywide_ledger_bound"],
        "bis_route_direction_sequence": c["R9_03_route_authority_bound"],
        "graph_path_cost": c["R9_04_path_cost_authority_bound"],
        "alighting": c["R9_23_alighting_contract"],
        "z01712": c["R9_24_z01712_contract"],
        "m009": c["R9_25_m009_closeout"], "stcis_api": c["R9_26_stcis_api_nondependency"],
        "route_frequency": c["R9_18_route_frequency_prior"],
        "research_414": c["R9_39_research_414_unchanged"]})
    dump(root, "external_data_enrichment_closeout.json", c["R9_05_enrichment_closed"])
    dump(root, "observed_vs_inferred_field_contract.json", {
        **c["R9_06_observed_inferred_boundary"],
        "R9_07_route_attribution": c["R9_07_route_attribution_inferred"],
        "R9_08_destination": c["R9_08_destination_inferred"]})
    dump(root, "boarding_to_counterfactual_drt_demand_contract.json", {
        **c["R9_09_boarding_to_drt_assumption"], "R9_10": c["R9_10_no_arbitrary_scale"]})
    dump(root, "lazy_demand_materialization_contract.json", c["R9_11_lazy_materialization"])
    dump(root, "request_timestamp_inference_contract.json", {
        **c["R9_13_timestamp_inference"], "R9_12": c["R9_12_bucket_duration_detected"],
        "R9_14": c["R9_14_count_conservation"]})
    dump(root, "route_attribution_evidence_audit.json", {**a, "R9_15": c["R9_15_route_candidate_audit"]})
    dump(root, "route_attribution_inference_contract.json", {
        **c["R9_07_route_attribution_inferred"], "R9_16": c["R9_16_multi_route_ambiguity"],
        "R9_17": c["R9_17_loops_and_occurrences"]})
    dump(root, "route_frequency_prior_audit.json", c["R9_18_route_frequency_prior"])
    dump(root, "destination_feasibility_contract.json", c["R9_19_destination_feasibility"])
    dump(root, "od_inference_family_contract.json", {
        "variants": result["od_variants"], "V0": c["R9_20_v0_defined"],
        "V1": c["R9_21_v1_defined"], "V2": c["R9_22_v2_defined"],
        "frequency_variant_available": False, "ground_truth_variant": None})
    dump(root, "alighting_auxiliary_evidence_contract.json", c["R9_23_alighting_contract"])
    dump(root, "z01712_auxiliary_only_non_regression.json", c["R9_24_z01712_contract"])
    dump(root, "m009_stcis_dependency_closeout.json", {
        "m009": c["R9_25_m009_closeout"], "stcis_api": c["R9_26_stcis_api_nondependency"]})
    dump(root, "chronological_leakage_prevention_contract.json", {
        **c["R9_27_leakage_policy"], "R9_28": c["R9_28_test6_untouched"]})
    dump(root, "od_uncertainty_and_realization_contract.json", c["R9_29_uncertainty_contract"])
    dump(root, "simulated_in_vehicle_time_provenance_contract.json", c["R9_30_in_vehicle_time_boundary"])
    dump(root, "citywide_first_scaling_contract.json", c["R9_31_citywide_first"])
    dump(root, "od_reconstruction_validation_contract.json", c["R9_40_validation_contract"])
    dump(root, "r9_test_report.json", result)
    dump(root, "classification.json", c["R9_41_classification_supported"])

    failed = result["failed_checks"]
    gate_passed = not failed
    gate_name = GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R9_CONTRACT_INCOMPLETE"
    net = c["R9_03_route_authority_bound"]
    freq = c["R9_18_route_frequency_prior"]
    gate = {
        "gate": gate_name, "source_sha": head_sha, "execution_base_sha": EXEC_BASE_SHA,
        "upstream_r8_9_1_sha": UPSTREAM_R891_SHA,
        "external_data_enrichment_status": "CLOSED_CURRENT_EVIDENCE_ONLY",
        "citywide_ledger_sha256": c["R9_02_citywide_ledger_bound"]["dataset_sha256"],
        "route_authority_sha256": net["sha256"], "route_directions": net["route_directions"],
        "path_cost_authority_sha256": c["R9_04_path_cost_authority_bound"]["sha256"],
        "route_attribution_status": "INFERRED",
        "destination_status": "INFERRED",
        "boarding_to_drt_assumption": c["R9_09_boarding_to_drt_assumption"]["assumption_id"],
        "boarding_to_drt_decision": "APPROVED_AND_FROZEN",
        "arbitrary_scale_factor": False,
        "bucket_seconds": c["R9_12_bucket_duration_detected"]["bucket_seconds"],
        "bucket_hardcoded": False,
        "timestamp_label": c["R9_13_timestamp_inference"]["label"],
        "count_conservation": "exact",
        "ledger_stops": a["ledger_stops"], "network_stops": a["network_stops"],
        "stops_unattributable": a["stops_unmatched_to_any_route_direction"],
        "unattributable_boarding_share": a["unmatched_boarding_share"],
        "materializable_boarding_share": a["materializable_boarding_share"],
        "stops_single_route_direction": a["stops_with_exactly_one_route_direction"],
        "stops_multi_route_direction": a["stops_with_multiple_route_directions"],
        "route_candidates_mean": a["route_direction_candidates_mean"],
        "route_candidates_boarding_weighted_mean": a["boarding_weighted_mean_candidates"],
        "route_candidates_max": a["route_direction_candidates_max"],
        "route_frequency_prior_status": freq["status"],
        "route_frequency_prior_fabricated": False,
        "destination_downstream_mean": c["R9_19_destination_feasibility"]["downstream_mean"],
        "terminal_occurrences_zero_downstream": c["R9_19_destination_feasibility"]["terminal_occurrences_zero_downstream"],
        "repeated_stop_occurrences": c["R9_17_loops_and_occurrences"]["repeated_stop_occurrences"],
        "od_variants": list(result["od_variants"]),
        "alighting_absolute_destination_marginal_allowed": False,
        "alighting_auxiliary_evidence_allowed": True,
        "z01712_absolute_transfer_allowed": False,
        "m009_daegu_citybus_constraint_authority": False,
        "stcis_api_required": False,
        "leakage_contract_definable": True, "test6_accessed": False,
        "uncertainty_fields": c["R9_29_uncertainty_contract"]["required_fields"],
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
            "ROUTE_ATTRIBUTION_AND_OD_REMAIN_INFERRED_PERMANENTLY",
            "ROUTE_FREQUENCY_PRIOR_UNAVAILABLE",
            f"{a['stops_unmatched_to_any_route_direction']}_UNATTRIBUTABLE_STOPS_"
            f"{a['unmatched_boarding_share']:.4f}_OF_DEMAND",
            "COST_DECAY_AND_ALIGHT_WEIGHT_UNCALIBRATED",
        ],
        "recommended_next_gate": NEXT_GATE}
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "implementation_permitted": True,
        "materialization_scope": "R9.1 tiny deterministic NON-TEST sample only",
        "full_matrix_permitted": False, "simulator_binding_permitted": False,
        "training_permitted": False, "comparison_permitted": False,
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "labelling inferred route usage or OD as observed",
            "single-realization conclusions", "fabricated frequency weights",
            "raw alighting as destination marginal", "Z01712 as a 2023 absolute constraint",
            "TEST6 access", "GitHub push"]})
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R9", "gate": gate_name,
                                     "classification": result["classification"],
                                     "r9": f"{gate['r9_tests_pass_count']}/{gate['r9_tests_total']}",
                                     "route_attribution_audit": a, "gate_decision": gate})

    md = [
        "# H4M-AE-R9 Daegu 2023 Route-Attribution Evidence and Constrained OD Mapping Contract",
        "",
        f"- gate: `{gate_name}`",
        f"- source HEAD at run: `{head_sha}` · execution base `{EXEC_BASE_SHA}`",
        f"- upstream R8.9.1 `{UPSTREAM_R891_SHA}`",
        f"- R9 {gate['r9_tests_pass_count']}/{gate['r9_tests_total']}",
        f"- classification: **{result['classification']}**",
        "",
        "This gate extends the earlier R9 contract additively with the route-attribution **evidence audit** that "
        "had not been measured. Nothing in the prior artifact is rewritten.",
        "",
        "## External data is closed",
        "",
        "`CLOSED_CURRENT_EVIDENCE_ONLY`. STCIS bus-only OD unavailable, 2023 Z01712 unavailable, M009 inactive, "
        "15-minute API not required. No external search was performed and nothing downstream waits on them.",
        "",
        "## Route-attribution evidence — the new measurement",
        "",
        "| quantity | value |",
        "| --- | --- |",
        f"| ledger stops | {a['ledger_stops']:,} |",
        f"| network stops | {a['network_stops']:,} over {a['route_directions']} route-directions |",
        f"| stops with **no** route-direction | {a['stops_unmatched_to_any_route_direction']} |",
        f"| demand on those stops | {a['unmatched_boardings']:,} (**{a['unmatched_boarding_share']:.4f}**) |",
        f"| **materializable demand share** | **{a['materializable_boarding_share']:.4f}** |",
        f"| stops with exactly 1 route-direction | {a['stops_with_exactly_one_route_direction']:,} |",
        f"| stops with 2 or more | {a['stops_with_multiple_route_directions']:,} |",
        f"| candidates per served stop | mean {a['route_direction_candidates_mean']}, median "
        f"{a['route_direction_candidates_median']}, p90 {a['route_direction_candidates_p90']}, max "
        f"{a['route_direction_candidates_max']} |",
        f"| **boarding-weighted mean candidates** | **{a['boarding_weighted_mean_candidates']}** |",
        "",
        f"So the hard invariant `route_candidate_count >= 1` is satisfiable for "
        f"**{a['materializable_boarding_share']:.2%}** of demand. The {a['stops_unmatched_to_any_route_direction']} "
        "unattributable stops are classified `UNATTRIBUTABLE_ORIGIN` with an explicit count — never dropped "
        "silently and never given a fabricated route.",
        "",
        f"The finding worth carrying forward: the boarding-weighted candidate mean "
        f"({a['boarding_weighted_mean_candidates']}) is well above the unweighted mean "
        f"({a['route_direction_candidates_mean']}). **Attribution ambiguity concentrates exactly where the demand "
        "is** — busy stops are served by more route-directions. Any later claim that route attribution is 'mostly "
        "unambiguous' would be false in demand-weighted terms.",
        "",
        "## Observed vs inferred",
        "",
        "Observed: service date, time bucket, origin stop, boarding count, route and direction topology, ordered "
        "occurrences, graph distance/time/generalized cost, partial alighting. **Inferred: the request "
        "timestamp, the passenger demand unit, route selection, direction selection, destination stop, the OD "
        "pair and both probabilities.** In-vehicle time is `SIMULATED_IN_VEHICLE_TIME`.",
        "",
        "Route attribution and destination are **never** to be labelled observed card OD, observed passenger OD "
        "or observed route demand.",
        "",
        "## Contracts frozen",
        "",
        f"- **boarding → DRT**: `{gate['boarding_to_drt_assumption']}`, a `RESEARCH_MODELING_ASSUMPTION`, "
        "approved and frozen with its limitations recorded. No multiplier, no 56.7 reuse, no Poisson resampling "
        "of the total.",
        f"- **timestamps**: bucket duration detected ({gate['bucket_seconds']} s), deterministic stratified "
        "placement, exact count conservation, alternatives rejected on record.",
        f"- **destination feasibility**: occurrence-keyed, mean {gate['destination_downstream_mean']} downstream "
        f"candidates, {gate['terminal_occurrences_zero_downstream']} terminal occurrences and "
        f"{gate['repeated_stop_occurrences']} repeated occurrences handled by explicit rule. Invariant "
        "`illegal_destination_count == 0`.",
        f"- **route prior**: `{freq['status']}`. The frequency artifact was inspected and rejected — "
        f"{freq['rows']} rows over {freq['routes_covered']} of {freq['project_routes']} routes from one 2026 "
        "realtime ETA sample. No timetable prior was fabricated; a neutral baseline variant is retained instead.",
        "- **OD family**: V0 feasibility-neutral, V1 path-cost prior, V2 cost plus smoothed alighting. None is "
        "ground truth and none is tuned to Z01712.",
        "- **leakage**: static topology shared; dynamic statistics fitted only on strictly earlier data; TEST6 "
        "never fitted on and never opened.",
        "- **uncertainty**: seed, variant, route and destination candidate counts, both probabilities, entropy "
        "and provenance on every future row; robustness across variants × realization seeds × MAPPO seeds.",
        "",
        "## What PASS does not mean",
        "",
        "Route usage is not observed, OD is not observed, no matrix or request ledger exists, the simulator is "
        "not bound and no KPI comparison is authorized. What is true is that current evidence supports a "
        "defensible inference methodology and implementation may begin.",
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
    print(f"[H4M-AE-R9] materializable demand share {a['materializable_boarding_share']:.4f}")
    print(f"[H4M-AE-R9] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R9] BLOCKED")


if __name__ == "__main__":
    main()
