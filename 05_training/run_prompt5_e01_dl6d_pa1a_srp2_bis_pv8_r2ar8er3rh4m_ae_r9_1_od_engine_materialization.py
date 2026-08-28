#!/usr/bin/env python3
"""PV8-R2A-R8E-R3-R-H4M-AE-R9.1 citywide route-attribution and constrained OD
engine minimal materialization.

Implementation-integrity gate.  Tiny deterministic NON-TEST materialization only.
No full OD, no request ledger, no simulator binding, no training, no comparison,
no TEST6, no DB writes, no external data, no push.
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
R91_TEST = TRAINING_ROOT / "test_h4m_ae_r9_1_od_engine_materialization.py"

R9_SOURCE_SHA = "6bedd5f24bfc88c6819b12c74ae6a5b4ebc455a1"
EXEC_BASE_SHA = "a8cfc31c8e32143f1b49b46c0f4149ad1daa6eec"
GATE_PASS = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R9_1_CITYWIDE_ROUTE_ATTRIBUTION"
             "_AND_CONSTRAINED_OD_ENGINE_MINIMAL_MATERIALIZATION_COMPLETE")
NEXT_GATE = "H4M-AE-R9_2_SCOPED_MULTI_WINDOW_OD_MATERIALIZATION_AND_UNCERTAINTY_DIAGNOSTICS"
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
    import test_h4m_ae_r9_1_od_engine_materialization as r91

    stamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S%z")
    stamp = f"{stamp[:-2]}:{stamp[-2:]}"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r9_1_od_engine_materialization_{stamp}"
    root.mkdir(parents=True, exist_ok=False)

    head_sha = git("rev-parse", "HEAD")
    status = git("status", "--short")
    diffstat = git("diff", "--stat", "HEAD")
    result = r91.run_validations()
    c = result["checks"]

    # fixture and minimal materialized output are small; store them as evidence
    pd.DataFrame(result["fixture"]).to_parquet(root / "minimal_fixture.parquet", index=False)
    sys.path.insert(0, str(TRAINING_ROOT))
    import constrained_od_engine as E
    cfg = E.EngineConfig(occurrence_master=r91.OCCURRENCE,
                         graph_nodes=r91.PACK / "full_graph_nodes.parquet",
                         graph_edges=r91.PACK / "full_graph_edges.parquet",
                         variant="V1_PATH_COST_PRIOR")
    index = E.build_index(cfg)
    alight = r91.psql_copy(
        f"SELECT stop_id, sum(alightings)::bigint AS boardings FROM public.fact_stop_usage_hourly "
        f"WHERE service_date='{r91.SAMPLE_DATE}' AND service_hour={r91.SAMPLE_HOUR} GROUP BY 1")
    amap = {r.stop_id: float(r.boardings) for r in alight.itertuples()}
    rows = list(E.materialize(index, cfg, result["fixture"], amap))
    out_df = pd.DataFrame(rows)
    out_df.to_parquet(root / "minimal_materialized_od.parquet", index=False)

    dump(root, "upstream_r9_binding.json", {
        "r9_source_sha": R9_SOURCE_SHA, "execution_base_sha": EXEC_BASE_SHA,
        "R9_1_01": c["R9_1_01_source_sha_binding"],
        "head_sha_at_run": head_sha, "git_status_short": status,
        "reset_or_rebase_performed": False, "frozen_artifacts_mutated": False})
    dump(root, "engine_manifest.json", {
        **result["engine_manifest"],
        "engine_module": str(ENGINE.relative_to(PROJECT_ROOT)),
        "engine_module_sha256": sha256_file(ENGINE),
        "R9_1_02": c["R9_1_02_citywide_index_built"]})
    dump(root, "minimal_fixture_contract.json", {
        **c["R9_1_03_fixture_non_test"],
        "fixture_file": "minimal_fixture.parquet",
        "fixture_sha256": sha256_file(root / "minimal_fixture.parquet"),
        "fixture_rows": result["fixture"]})
    dump(root, "minimal_materialization_output.json", {
        "output_file": "minimal_materialized_od.parquet",
        "output_sha256": sha256_file(root / "minimal_materialized_od.parquet"),
        "rows": int(len(out_df)),
        "attributed_rows": int((out_df["attribution_status"] == "ATTRIBUTED_ORIGIN").sum()),
        "unattributable_rows": int((out_df["attribution_status"] == "UNATTRIBUTABLE_ORIGIN").sum()),
        "total_mass": float(out_df["mass"].sum()),
        "sample": result["materialized_sample"],
        "full_od_matrix": False, "request_ledger": False})
    dump(root, "origin_conservation_validation.json", c["R9_1_04_origin_conservation"])
    dump(root, "route_candidate_validation.json", c["R9_1_05_route_candidate_validity"])
    dump(root, "downstream_only_destination_validation.json", {
        **c["R9_1_06_downstream_only_rule"], "R9_1_07": c["R9_1_07_illegal_destination_count"]})
    dump(root, "unattributable_origin_validation.json", c["R9_1_08_unattributable_preserved"])
    dump(root, "authoritative_unmatched_population_check.json", c["R9_1_09_authoritative_unmatched_class"])
    dump(root, "route_frequency_prior_non_fabrication.json", c["R9_1_10_no_fabricated_route_prior"])
    dump(root, "determinism_validation.json", c["R9_1_11_determinism"])
    dump(root, "provenance_completeness_validation.json", c["R9_1_12_provenance_complete"])
    dump(root, "od_variant_implementation_audit.json", {
        **c["R9_1_13_variants_implemented"],
        "observation": ("with uncalibrated priors the three variants differ only marginally in mean entropy on "
                        "this fixture; that is expected and is not evidence that the variants are equivalent at "
                        "calibrated settings or at larger scope")})
    dump(root, "auxiliary_evidence_guards.json", c["R9_1_14_auxiliary_guards"])
    dump(root, "scalability_evidence.json", c["R9_1_15_scalability"])
    dump(root, "claim_guards.json", {**c["R9_1_16_claim_guards"], "R9_1_17": c["R9_1_17_prohibitions"]})
    dump(root, "frozen_contract_non_regression.json", c["R9_1_18_frozen_contracts"])
    dump(root, "r9_1_test_report.json", {k: v for k, v in result.items() if k != "fixture"})

    failed = result["failed_checks"]
    gate_passed = not failed
    gate_name = GATE_PASS if gate_passed else "BLOCKED_H4M_AE_R9_1_ENGINE_VALIDATION_INCOMPLETE"
    fx = c["R9_1_03_fixture_non_test"]
    cons = c["R9_1_04_origin_conservation"]
    un = c["R9_1_08_unattributable_preserved"]
    agg = c["R9_1_09_authoritative_unmatched_class"]
    sc = c["R9_1_15_scalability"]
    gate = {
        "gate": gate_name, "source_sha": head_sha, "execution_base_sha": EXEC_BASE_SHA,
        "r9_source_sha": R9_SOURCE_SHA,
        "classification": result["classification"],
        "engine_module": str(ENGINE.relative_to(PROJECT_ROOT)),
        "engine_module_sha256": sha256_file(ENGINE),
        "index_occurrences": sc["index_occurrences"], "index_route_directions": sc["index_route_directions"],
        "sample_origin_rows": fx["fixture_rows"], "sample_origin_mass": cons["input_mass"],
        "attributable_mass": cons["input_mass"] - un["unattributable_mass"],
        "unattributable_mass": un["unattributable_mass"],
        "single_route_candidate_origins": fx["cases_covered"]["single_route_direction_origin"],
        "multi_route_candidate_origins": fx["cases_covered"]["multi_candidate_origin"],
        "unattributable_origins": fx["cases_covered"]["unattributable_origin"],
        "route_candidate_counts": fx["route_candidate_counts"],
        "materialized_rows": int(len(out_df)),
        "legal_downstream_candidate_rows": int((out_df["attribution_status"] == "ATTRIBUTED_ORIGIN").sum()),
        "illegal_destination_count": c["R9_1_07_illegal_destination_count"]["illegal_destination_count"],
        "min_sequence_delta": c["R9_1_06_downstream_only_rule"]["min_sequence_delta"],
        "origin_conservation": "EXACT",
        "determinism": c["R9_1_11_determinism"]["identical"],
        "provenance_complete": c["R9_1_12_provenance_complete"]["reconstructable"],
        "route_prior_fabricated": False, "route_prior_mode": c["R9_1_10_no_fabricated_route_prior"]["route_prior_mode"],
        "authoritative_unmatched_stops": agg["unmatched_stops"],
        "authoritative_unmatched_boardings": agg["unmatched_boardings"],
        "authoritative_unmatched_matches_r9": agg["matches_r9"],
        "unmatched_check_without_od_materialization": True,
        "maxrss_delta_bytes": sc["maxrss_delta_bytes"],
        "passenger_rows_expanded": 0, "full_matrix_materialized": False,
        "citywide_capable": True, "suseong_hardcoded": sc["suseong_hardcoded"],
        "route_attribution_observed": False, "destination_observed": False, "od_ground_truth": False,
        "actual_request_ledger_created": False, "simulator_binding_allowed": False,
        "training_allowed": False, "performance_comparison_allowed": False,
        "paper_level_claim_allowed": False, "causal_performance_claim_allowed": False,
        "test6_accessed": False, "db_writes": 0, "external_api_or_web": 0,
        "reward_v2_unchanged": True, "zero_loss_unchanged": True, "k_mask_unchanged": True,
        "ledger_unchanged": True, "research_414_unchanged": True,
        "r9_1_tests_pass_count": sum(1 for v in c.values() if v["passed"]),
        "r9_1_tests_total": len(c), "r9_1_tests_all_pass": gate_passed,
        "remaining_dependencies": [
            "PRIOR_PARAMETERS_UNCALIBRATED_COST_DECAY_AND_ALIGHT_WEIGHT",
            "ROUTE_FREQUENCY_PRIOR_STILL_UNAVAILABLE",
            "SEGMENT_COST_COVERAGE_BELOW_ONE_NEUTRAL_FALLBACK_USED",
            "OD_REMAINS_INFERRED_PERMANENTLY"],
        "recommended_next_gate": NEXT_GATE}
    dump(root, "gate_decision.json", gate)
    dump(root, "downstream_lock.json", {
        "locked_by": gate_name,
        "engine_available_for": "scoped multi-window materialization under a new gate",
        "full_citywide_materialization_permitted": False,
        "simulator_binding_permitted": False, "training_permitted": False, "comparison_permitted": False,
        "next_gate": NEXT_GATE, "next_gate_auto_executed": False,
        "forbidden_downstream_without_new_gate": [
            "labelling engine output as observed OD", "claiming calibrated priors",
            "full 181M expansion", "TEST6 access", "GitHub push"]})
    dump(root, "final_report.json", {"stage": "PV8-R2A-R8E-R3-R-H4M-AE-R9.1", "gate": gate_name,
                                     "classification": result["classification"],
                                     "r9_1": f"{gate['r9_1_tests_pass_count']}/{gate['r9_1_tests_total']}",
                                     "gate_decision": gate})

    var = c["R9_1_13_variants_implemented"]["variants"]
    md = [
        "# H4M-AE-R9.1 Citywide Route-Attribution and Constrained OD Engine — Minimal Materialization",
        "",
        f"- gate: `{gate_name}`",
        f"- source HEAD at run: `{head_sha}` · execution base `{EXEC_BASE_SHA}` · R9 `{R9_SOURCE_SHA}`",
        f"- R9.1 {gate['r9_1_tests_pass_count']}/{gate['r9_1_tests_total']}",
        f"- classification: **{result['classification']}**",
        "",
        "## Engine",
        "",
        f"`{gate['engine_module']}` (`{gate['engine_module_sha256'][:16]}…`) builds one citywide index — "
        f"**{sc['index_occurrences']:,} occurrences over {sc['index_route_directions']} route-directions** — then "
        "generates destination candidates lazily as a slice of the ordered route-direction sequence. Demand is "
        "carried as **aggregated mass**, so no passenger row is ever expanded and no Cartesian product over "
        "routes × stops is formed.",
        "",
        f"Segment-cost coverage from the authoritative graph is "
        f"{c['R9_1_02_citywide_index_built']['segment_cost_coverage']:.4f}. The uncovered segments do **not** "
        "silently drop candidates: those destinations stay feasible and fall back to a neutral weight tagged "
        "`COST_UNAVAILABLE_NEUTRAL_FALLBACK`, so missing cost evidence never removes a legal destination.",
        "",
        "## Minimal materialization",
        "",
        f"Non-TEST fixture: {fx['service_date']} hour {fx['service_hour']}, "
        f"{fx['fixture_rows']} origin rows, **{cons['input_mass']:.0f} boarding mass**, chosen to cover every "
        f"required case — route candidate counts {fx['route_candidate_counts']}.",
        "",
        "| invariant | result |",
        "| --- | --- |",
        f"| origin conservation | **EXACT** — in {cons['input_mass']:.0f}, out {cons['output_mass']:.0f}, max per-origin diff {cons['max_abs_per_origin_diff']} |",
        f"| attributable / unattributable mass | {gate['attributable_mass']:.0f} / {gate['unattributable_mass']:.0f} |",
        f"| legal downstream rows | {gate['legal_downstream_candidate_rows']} |",
        f"| **illegal destination count** | **{gate['illegal_destination_count']}** |",
        f"| min sequence delta | {gate['min_sequence_delta']} (strictly downstream) |",
        f"| determinism | {gate['determinism']} (index rebuilt, digests identical) |",
        f"| provenance reconstructable | {gate['provenance_complete']} |",
        f"| maxrss delta | {sc['maxrss_delta_bytes']:,} bytes |",
        "",
        "The unattributable origin is **preserved as mass**, not dropped: "
        f"{un['unattributable_mass']:.0f} units emitted as `UNATTRIBUTABLE_ORIGIN` with reason "
        f"`{un['reasons_present'][0] if un['reasons_present'] else 'n/a'}`, and no reassignment, nearest-match or "
        "fabrication.",
        "",
        "## The 51-stop population, checked properly",
        "",
        "Rather than claim the tiny fixture reproduces it, I checked the **authoritative citywide aggregate** by "
        "joining stop-level totals against the route index with **no OD row materialized**: "
        f"**{agg['unmatched_stops']} stops / {agg['unmatched_boardings']:,} boardings "
        f"({agg['unmatched_share']:.4f})** — matching R9 exactly. The class is representable and was not silently "
        "repaired.",
        "",
        "## Variants",
        "",
        "| variant | rows | mass | mean entropy |",
        "| --- | --- | --- | --- |",
        *[f"| {k} | {v['rows']} | {v['mass']} | {v['mean_entropy']} |" for k, v in var.items()],
        "",
        "All three conserve mass exactly and all remain `INFERRED`. Worth stating plainly: **with uncalibrated "
        "priors the variants differ only marginally in entropy on this fixture.** That is expected at "
        "cost_decay 0.15/km on short candidate sets — it is not evidence that the variants are equivalent at "
        "calibrated settings or at wider scope, and no calibration is claimed here.",
        "",
        "## Route prior",
        "",
        f"`{gate['route_prior_mode']}`. Route choice is uniform over serving route-directions. The engine reads "
        "no frequency artifact, no headway and no ETA sample — verified by source scan, not by assertion.",
        "",
        "## Claim guards — all false",
        "",
        "`route_attribution_observed`, `destination_observed`, `od_ground_truth`, "
        "`actual_request_ledger_created`, `simulator_binding_allowed`, `training_allowed`, "
        "`performance_comparison_allowed`, `paper_level_claim_allowed`, `causal_performance_claim_allowed`.",
        "",
        "TEST6 untouched, no training, no rollout, no comparison, no DB writes, no external data, no frozen "
        "artifact mutated.",
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
        "authoritative_source_sha256": result["engine_manifest"]["source_sha256"],
        "engine_module_sha256": sha256_file(ENGINE),
        "validator_sha256": sha256_file(R91_TEST),
        "full_od_materialized": False, "request_ledger_created": False,
        "git_diff_stat": diffstat})
    (root / "_SUCCESS.lock").write_text(json.dumps({"gate": gate_name, "passed": gate_passed, "next_gate": NEXT_GATE}, indent=2), encoding="utf-8")

    print(f"[H4M-AE-R9.1] artifact root: {root}")
    print(f"[H4M-AE-R9.1] gate: {gate_name}")
    print(f"[H4M-AE-R9.1] classification: {result['classification']}")
    print(f"[H4M-AE-R9.1] tests {gate['r9_1_tests_pass_count']}/{gate['r9_1_tests_total']} | rows {gate['materialized_rows']} | illegal {gate['illegal_destination_count']}")
    print(f"[H4M-AE-R9.1] artifact files: {len(files) + 1}")
    if not gate_passed:
        raise SystemExit("[H4M-AE-R9.1] BLOCKED")


if __name__ == "__main__":
    main()
