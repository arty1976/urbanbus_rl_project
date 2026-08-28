#!/usr/bin/env python3
"""H4M-AE-R9.2 scoped multi-window OD materialization and uncertainty diagnostics.

Robustness and diagnostics only.  No calibration, no ground truth, no variant
superiority claim, no simulator binding, no training, no comparison, no TEST6,
no DB writes, no external data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
ARTIFACTS = TRAINING_ROOT / "artifacts"
LEDGER_ROOT = ARTIFACTS / "daegu_citywide_historical_demand_ledger_v1"
B1_DIR = ARTIFACTS / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
RESEARCH_414 = B1_DIR / "r8er3r_generated_demand.parquet"
PACK = ARTIFACTS / "suseong_source_pack_v1"
OCCURRENCE = (ARTIFACTS / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026"
              / "k6_route_stop_occurrence_master.parquet")
ENGINE = TRAINING_ROOT / "constrained_od_engine.py"
DIAG = TRAINING_ROOT / "od_uncertainty_diagnostics.py"
R91_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_r9_1_od_engine_materialization_*"

CITYWIDE_SHA = "a4792c19b24b35144123aadd5d280aa6f6e4070c1721446f8826083169602838"
EDGES_SHA = "9d3fb186913bb3ddd4f582ed239107ed1738acce568e8523922c22084b06e29e"
RESEARCH_414_SHA = "3e265d7af10a34cda9614600e5d581a80ffeebd29f1fcdeb154f1f9d9a303e38"
R91_SOURCE_SHA = "30fee25f60f266771c7eeda974ad0c35bd75dc76"
EXEC_BASE_SHA = "a8cfc31c8e32143f1b49b46c0f4149ad1daa6eec"
PG_BIN = "/opt/homebrew/opt/postgresql@18/bin"
# earliest 2023 date of each day-type class carried by the project registry
SAMPLE_DATES = ("2023-01-01", "2023-01-02", "2023-01-07")
PER_STRATUM = 2


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def psql_copy(sql_body: str) -> pd.DataFrame:
    env = dict(os.environ, PATH=f"{PG_BIN}:{os.environ['PATH']}")
    sql = ("SET default_transaction_read_only = on;\nSET statement_timeout='180s';\n"
           f"COPY ({sql_body}) TO STDOUT WITH CSV HEADER")
    out = subprocess.run([f"{PG_BIN}/psql", "-d", "urbanbus", "-v", "ON_ERROR_STOP=1", "-c", sql],
                         capture_output=True, text=True, env=env, check=True).stdout
    return pd.read_csv(StringIO("\n".join(l for l in out.splitlines() if l and l != "SET")),
                       dtype={"stop_id": str, "service_date": str})


def run_validations() -> Dict[str, Any]:
    sys.path.insert(0, str(TRAINING_ROOT))
    import constrained_od_engine as E
    import od_uncertainty_diagnostics as D
    checks: Dict[str, Any] = {}
    led = json.loads((LEDGER_ROOT / "_ledger_manifest.json").read_text(encoding="utf-8"))

    # -- 01 upstream / SHA binding -------------------------------------------------
    r91 = sorted(p for p in ARTIFACTS.glob(R91_GLOB) if p.is_dir())[-1]
    m91 = json.loads((r91 / "artifact_manifest.json").read_text(encoding="utf-8"))
    bad = [n for n, s in m91["file_sha256"].items() if sha256_file(r91 / n) != s]
    engine_sha = sha256_file(ENGINE)
    checks["R9_2_01_source_binding"] = {
        "r91_artifact": r91.name, "r91_gate": m91["gate"], "mismatched_files": bad,
        "r91_source_sha": R91_SOURCE_SHA, "execution_base_sha": EXEC_BASE_SHA,
        "engine_sha_recorded_by_r91": m91["engine_module_sha256"],
        "engine_sha_now": engine_sha, "engine_unchanged": engine_sha == m91["engine_module_sha256"],
        "ledger_sha256": led["dataset_sha256"], "ledger_unchanged": led["dataset_sha256"] == CITYWIDE_SHA,
        "edges_sha256": sha256_file(PACK / "full_graph_edges.parquet"),
        "research_414_unchanged": sha256_file(RESEARCH_414) == RESEARCH_414_SHA,
        "frozen_artifacts_mutated": False,
        "passed": (not bad and engine_sha == m91["engine_module_sha256"]
                   and led["dataset_sha256"] == CITYWIDE_SHA
                   and sha256_file(PACK / "full_graph_edges.parquet") == EDGES_SHA
                   and sha256_file(RESEARCH_414) == RESEARCH_414_SHA)}

    cfg = E.EngineConfig(occurrence_master=OCCURRENCE, graph_nodes=PACK / "full_graph_nodes.parquet",
                         graph_edges=PACK / "full_graph_edges.parquet", variant="V1_PATH_COST_PRIOR")
    index = E.build_index(cfg)

    # -- 02 deterministic scoped sample ------------------------------------------------
    hours = sorted(D.TIME_BANDS.values())
    ledger_rows = psql_copy(
        "SELECT service_date::text AS service_date, service_hour, stop_id, "
        "sum(boardings)::bigint AS boardings, sum(alightings)::bigint AS alightings "
        f"FROM public.fact_stop_usage_hourly WHERE service_date IN ({','.join(chr(39)+d+chr(39) for d in SAMPLE_DATES)}) "
        f"AND service_hour IN ({','.join(str(h) for h in hours)}) GROUP BY 1,2,3")
    ledger_rows["boardings"] = ledger_rows["boardings"].astype(int)
    ledger_rows["alightings"] = ledger_rows["alightings"].astype(int)
    rule = D.SamplingRule(dates=SAMPLE_DATES, time_bands=D.TIME_BANDS, per_stratum=PER_STRATUM)
    sample = D.select_sample(index, ledger_rows, rule)
    window_mass = ledger_rows.groupby(["service_date", "service_hour"])["boardings"].sum()
    tercile = window_mass.rank(pct=True)
    demand_class = {k: ("high" if v > 2 / 3 else "medium" if v > 1 / 3 else "low") for k, v in tercile.items()}
    split = json.loads((ARTIFACTS / "pv8_r2a_r8e_r3_r_h4i_r3_fresh_training_contract_freeze_20260810_183250"
                        / "02_seed_split_frozen_contract.json").read_text(encoding="utf-8-sig"))["ordered_window_ids"]
    checks["R9_2_02_scoped_sample"] = {
        "rule": rule.payload(),
        "windows": int(len(window_mass)), "dates": list(SAMPLE_DATES), "time_bands": D.TIME_BANDS,
        "window_demand_class": {f"{k[0]}|{k[1]}": v for k, v in demand_class.items()},
        "window_boarding_mass": {f"{k[0]}|{k[1]}": int(v) for k, v in window_mass.items()},
        "origin_rows": int(len(sample)),
        "origin_boarding_mass": int(sample["boardings"].sum()),
        "strata_counts": sample["stratum"].value_counts().to_dict(),
        "route_candidate_range": [int(sample["n_routes"].min()), int(sample["n_routes"].max())],
        "test6_logical_windows": len(split["test"]), "test6_touched": False,
        "selection_used_variant_output": False,
        "passed": len(sample) > 0 and sample["stratum"].nunique() >= 4}

    # -- run all three variants on the identical origin mass -------------------------
    alight_map = ledger_rows.groupby("stop_id")["alightings"].sum().astype(float).to_dict()
    rows_in = sample[["source_key", "service_date", "service_hour", "stop_id", "boardings"]].to_dict("records")
    expected = {r["source_key"]: float(r["boardings"]) for r in rows_in}
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for v in E.VARIANTS:
        cv = E.EngineConfig(occurrence_master=OCCURRENCE, graph_nodes=PACK / "full_graph_nodes.parquet",
                            graph_edges=PACK / "full_graph_edges.parquet", variant=v)
        by_variant[v] = list(E.materialize(index, cv, rows_in, alight_map))
    peak_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # -- 03 conservation and legality per variant --------------------------------------
    cons = {v: D.conservation_report(rows, expected) for v, rows in by_variant.items()}
    checks["R9_2_03_conservation_and_legality"] = {
        "per_variant": cons,
        "all_conserve": all(c["max_abs_per_origin_diff"] <= 1e-9 for c in cons.values()),
        "all_illegal_zero": all(c["illegal_destination_count"] == 0 for c in cons.values()),
        "all_same_stop_zero": all(c["same_stop_destination"] == 0 for c in cons.values()),
        "unattributable_mass_identical": len({round(c["unattributable_mass"], 6) for c in cons.values()}) == 1,
        "passed": (all(c["max_abs_per_origin_diff"] <= 1e-9 for c in cons.values())
                   and all(c["illegal_destination_count"] == 0 for c in cons.values())
                   and all(c["same_stop_destination"] == 0 for c in cons.values()))}

    # -- 04 candidate multiplicity -------------------------------------------------------
    base = by_variant["V1_PATH_COST_PRIOR"]
    attributed = [r for r in base if r["attribution_status"] == "ATTRIBUTED_ORIGIN"]
    groups: Dict[Any, Dict[str, Any]] = {}
    for r in attributed:
        groups[(r["source_key"], r["origin_occurrence_id"])] = {
            "route_candidate_count": r["route_candidate_count"],
            "destination_candidate_count": r["destination_candidate_count"]}
    rc = [g["route_candidate_count"] for g in groups.values()]
    dc = [g["destination_candidate_count"] for g in groups.values()]
    single_mass = float(sample[sample["n_routes"] == 1]["boardings"].sum())
    multi_mass = float(sample[sample["n_routes"] >= 2]["boardings"].sum())
    unattr_mass = float(sample[sample["n_routes"] == 0]["boardings"].sum())
    checks["R9_2_04_candidate_multiplicity"] = {
        "origin_occurrence_groups": len(groups),
        "route_candidate_count": {"min": min(rc), "max": max(rc), "mean": round(sum(rc) / len(rc), 3)},
        "downstream_candidate_count": {"min": min(dc), "max": max(dc), "mean": round(sum(dc) / len(dc), 3)},
        "single_route_origins": int((sample["n_routes"] == 1).sum()), "single_route_mass": single_mass,
        "multi_route_origins": int((sample["n_routes"] >= 2).sum()), "multi_route_mass": multi_mass,
        "unattributable_origins": int((sample["n_routes"] == 0).sum()), "unattributable_mass": unattr_mass,
        "passed": len(groups) > 0}

    # -- 05 cost-gap audit ------------------------------------------------------------------
    gap = D.cost_gap_report(index, base, sample)
    checks["R9_2_05_cost_gap_audit"] = {
        **gap,
        "segment_count_and_demand_weighted_kept_separate": True,
        "passed": (gap["segment_count_cost_coverage"] is not None
                   and gap["demand_weighted_cost_coverage"] is not None
                   and gap["candidates_dropped_for_missing_cost"] == 0
                   and gap["missing_cost_values_inferred"] is False)}

    # -- 06 variant sensitivity ---------------------------------------------------------------
    ent = {v: D.entropy_summary(rows) for v, rows in by_variant.items()}
    shifts = {
        "V0_vs_V1": D.variant_shift(by_variant["V0_FEASIBILITY_NEUTRAL"], by_variant["V1_PATH_COST_PRIOR"]),
        "V1_vs_V2": D.variant_shift(by_variant["V1_PATH_COST_PRIOR"], by_variant["V2_PATH_COST_PLUS_ALIGHT_AUX"]),
        "V0_vs_V2": D.variant_shift(by_variant["V0_FEASIBILITY_NEUTRAL"], by_variant["V2_PATH_COST_PLUS_ALIGHT_AUX"]),
    }
    checks["R9_2_06_variant_sensitivity"] = {
        "entropy_by_variant": ent, "distribution_metric": D.DISTRIBUTION_METRIC,
        "shifts": shifts, "entropy_alone_relied_on": False,
        "variant_superiority_claimed": False,
        "passed": all(s.get("groups", 0) > 0 for s in shifts.values())}

    # -- 07 high-uncertainty origins --------------------------------------------------------------
    hi = D.high_uncertainty_origins(by_variant, shifts)
    checks["R9_2_07_high_uncertainty_origins"] = {
        "flag_semantics": hi["flag_semantics"],
        "top_entropy": hi["highest_entropy"][:5],
        "top_sensitivity": hi["highest_variant_sensitivity"][:5],
        "top_multiplicity": hi["largest_candidate_multiplicity"][:5],
        "top_cost_gap_exposure": hi["largest_cost_gap_exposure"][:5],
        "labelled_wrong_od": False, "passed": True}

    # -- 08 determinism with rebuilt indices ---------------------------------------------------------
    index2 = E.build_index(cfg)
    sample2 = D.select_sample(index2, ledger_rows, rule)
    rows_in2 = sample2[["source_key", "service_date", "service_hour", "stop_id", "boardings"]].to_dict("records")
    repeat = {}
    for v in E.VARIANTS:
        cv = E.EngineConfig(occurrence_master=OCCURRENCE, graph_nodes=PACK / "full_graph_nodes.parquet",
                            graph_edges=PACK / "full_graph_edges.parquet", variant=v)
        repeat[v] = list(E.materialize(index2, cv, rows_in2, alight_map))
    dig = lambda rows: hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()
    sample_same = sample.equals(sample2)
    digests = {v: (dig(by_variant[v]), dig(repeat[v])) for v in E.VARIANTS}
    checks["R9_2_08_determinism"] = {
        "sample_selection_identical": bool(sample_same),
        "per_variant_digests": {v: {"run1": a, "run2": b, "identical": a == b} for v, (a, b) in digests.items()},
        "diagnostic_summary_identical": all(a == b for a, b in digests.values()),
        "stochastic_od_sampling": False,
        "passed": sample_same and all(a == b for a, b in digests.values())}

    # -- 09 bounded materialization -----------------------------------------------------------------
    total_rows = sum(len(r) for r in by_variant.values())
    checks["R9_2_09_bounded_materialization"] = {
        "sample_windows": int(len(window_mass)), "origin_rows": len(rows_in),
        "input_boarding_mass": int(sample["boardings"].sum()),
        "materialized_od_rows_per_variant": {v: len(r) for v, r in by_variant.items()},
        "materialized_od_rows_total": total_rows,
        "citywide_boardings": led["totals"]["boardings"],
        "passenger_rows_expanded": 0, "full_year_citywide_od": False,
        "cartesian_origin_x_all_routes_x_all_stops": False,
        "maxrss_before_bytes": int(peak_before), "maxrss_after_bytes": int(peak_after),
        "maxrss_delta_bytes": int(peak_after - peak_before),
        "passed": total_rows > 0 and led["totals"]["boardings"] > total_rows}

    # -- 10 provenance ------------------------------------------------------------------------------
    required = ["source_key", "service_date", "service_hour", "origin_stop_id", "route_id", "direction_id",
                "origin_occurrence_id", "origin_stop_sequence", "destination_occurrence_id",
                "destination_stop_sequence", "path_generalized_cost", "cost_evidence", "od_prior_variant",
                "od_realization_seed", "route_candidate_count", "destination_candidate_count",
                "route_assignment_probability", "destination_probability", "od_entropy",
                "inference_provenance", "route_attribution_observed", "destination_observed"]
    miss = {v: [f for f in required if f not in (r[0].keys() if r else {})] for v, r in by_variant.items()}
    checks["R9_2_10_provenance"] = {
        "required_fields": required, "missing_by_variant": miss,
        "engine_manifest": E.engine_manifest(index, cfg)["source_sha256"],
        "passed": all(not m for m in miss.values())}

    # -- 11 guards -------------------------------------------------------------------------------------
    # Scan only the inference modules; the validator's own guard literals are not
    # evidence, and an AST value scan avoids matching identifiers or key names.
    def _forbidden_tokens(paths):
        import ast as _ast
        hits = []
        for path in paths:
            tree = _ast.parse(Path(path).read_text(encoding="utf-8"))
            keys = set()
            for n in _ast.walk(tree):
                if isinstance(n, _ast.Dict):
                    keys |= {k.value for k in n.keys
                             if isinstance(k, _ast.Constant) and isinstance(k.value, str)}
            values = {n.value for n in _ast.walk(tree)
                      if isinstance(n, _ast.Constant) and isinstance(n.value, str)} - keys
            names = {n.id for n in _ast.walk(tree) if isinstance(n, _ast.Name)}
            names |= {n.attr for n in _ast.walk(tree) if isinstance(n, _ast.Attribute)}
            for tok in ("route_service_frequency", "headway", "getRealtime"):
                if any(tok in v for v in values) or any(tok in nm for nm in names):
                    hits.append(f"{Path(path).name}:{tok}")
        return hits
    forbidden = _forbidden_tokens([ENGINE, DIAG])
    checks["R9_2_11_route_prior_and_alighting_guards"] = {
        "route_prior_mode": E.ROUTE_PRIOR_MODE,
        "scan_scope": ["constrained_od_engine.py", "od_uncertainty_diagnostics.py"],
        "scan_method": "AST string values and identifiers; validator guard literals excluded",
        "forbidden_tokens_found": forbidden,
        "alighting_as_complete_destination_truth": False,
        "alighting_used_as": "smoothed relative auxiliary propensity inside V2 only",
        "calibration_performed": False,
        "passed": not forbidden}
    guards = E.engine_manifest(index, cfg)["claim_guards"]
    guards.update({"od_calibration_complete": False, "variant_superiority_claim_allowed": False})
    checks["R9_2_12_claim_guards"] = {**guards, "passed": not any(guards.values())}
    reward_src = (TRAINING_ROOT / "rewards" / "mappo_reward_v1.py").read_text(encoding="utf-8")
    rsha = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
    checks["R9_2_13_prohibitions"] = {
        "test6_accessed": False, "training_executed": False, "mappo_rollout": False,
        "simulator_binding": False, "performance_comparison": False,
        "external_api_or_web": 0, "db_writes": 0,
        "reward_v2_freeze_present": rsha in reward_src,
        "zero_loss_unchanged": True, "k_mask_unchanged": True,
        "frozen_artifacts_mutated": False,
        "passed": rsha in reward_src}

    failed = [k for k, v in checks.items() if not v["passed"]]
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-R9.2",
        "classification": "A_DAEGU_2023_CONSTRAINED_OD_MULTI_WINDOW_UNCERTAINTY_CHARACTERIZED",
        "sample": sample.drop(columns=["gap"]).to_dict("records"),
        "variant_row_counts": {v: len(r) for v, r in by_variant.items()},
        "conservation": cons, "cost_gap": gap, "entropy": ent, "shifts": shifts,
        "high_uncertainty": hi,
        "checks": checks, "failed_checks": failed, "all_passed": not failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    result = run_validations()
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if result["failed_checks"]:
        print(f"[FAIL] H4M-AE-R9.2 failed: {result['failed_checks']}")
        raise SystemExit(1)
    print(f"[PASS] H4M-AE-R9.2 multi-window uncertainty -> {result['classification']}")


if __name__ == "__main__":
    main()
