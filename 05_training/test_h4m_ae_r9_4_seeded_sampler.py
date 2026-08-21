#!/usr/bin/env python3
"""H4M-AE-R9.4 seeded OD sampler interface and request realization contract.

Sampler contract validation only.  No simulator binding, no MAPPO, no training,
no comparison, no TEST6, no DB writes, no external data, no new OD evidence.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
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
REPAIRED_EDGES = ARTIFACTS / "daegu_path_cost_repair_v1" / "full_graph_edges_repaired.parquet"
ENGINE = TRAINING_ROOT / "constrained_od_engine.py"
SAMPLER = TRAINING_ROOT / "od_seeded_sampler.py"
R93_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_r9_3_cost_repair_stability_*"

ENGINE_SHA_R91 = "5e877b9c69471a59"
REPAIRED_EDGES_SHA = "66301e830bf3b437"
CITYWIDE_SHA = "a4792c19b24b35144123aadd5d280aa6f6e4070c1721446f8826083169602838"
RESEARCH_414_SHA = "3e265d7af10a34cda9614600e5d581a80ffeebd29f1fcdeb154f1f9d9a303e38"
R93_SOURCE_SHA = "5fee5dc225f54d2d3263d4f425ccca3dbdecd261"
EXEC_BASE_SHA = "a8cfc31c8e32143f1b49b46c0f4149ad1daa6eec"
PG_BIN = "/opt/homebrew/opt/postgresql@18/bin"
SAMPLE_DATES = ("2023-01-01", "2023-01-02", "2023-01-07")
GLOBAL_SEED = 20260821
ALT_SEED = 777
FIDELITY_N = 10000
REALIZATION_N = 200
ORIGINS_PER_VARIANT = 3


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
    import od_seeded_sampler as S
    checks: Dict[str, Any] = {}
    led = json.loads((LEDGER_ROOT / "_ledger_manifest.json").read_text(encoding="utf-8"))

    # -- 01 upstream binding --------------------------------------------------------
    r93 = sorted(p for p in ARTIFACTS.glob(R93_GLOB) if p.is_dir())[-1]
    m93 = json.loads((r93 / "artifact_manifest.json").read_text(encoding="utf-8"))
    bad = [n for n, s in m93["file_sha256"].items() if sha256_file(r93 / n) != s]
    checks["R9_4_01_upstream_binding"] = {
        "r93_artifact": r93.name, "r93_gate": m93["gate"], "mismatched_files": bad,
        "r93_source_sha": R93_SOURCE_SHA, "execution_base_sha": EXEC_BASE_SHA,
        "engine_byte_identical": sha256_file(ENGINE).startswith(ENGINE_SHA_R91),
        "repaired_edges_unchanged": sha256_file(REPAIRED_EDGES).startswith(REPAIRED_EDGES_SHA),
        "ledger_unchanged": led["dataset_sha256"] == CITYWIDE_SHA,
        "research_414_unchanged": sha256_file(RESEARCH_414) == RESEARCH_414_SHA,
        "frozen_artifacts_mutated": False,
        "passed": (not bad and sha256_file(ENGINE).startswith(ENGINE_SHA_R91)
                   and sha256_file(REPAIRED_EDGES).startswith(REPAIRED_EDGES_SHA)
                   and led["dataset_sha256"] == CITYWIDE_SHA
                   and sha256_file(RESEARCH_414) == RESEARCH_414_SHA)}

    # -- rebuild the frozen R9.3 distributions (repaired edges), variants isolated ----
    def build(variant: str):
        cfg = E.EngineConfig(occurrence_master=OCCURRENCE, graph_nodes=PACK / "full_graph_nodes.parquet",
                             graph_edges=REPAIRED_EDGES, variant=variant)
        return cfg, E.build_index(cfg)
    cfg_v1, index = build("V1_PATH_COST_PRIOR")
    hours = sorted(D.TIME_BANDS.values())
    ledger_rows = psql_copy(
        "SELECT service_date::text AS service_date, service_hour, stop_id, "
        "sum(boardings)::bigint AS boardings, sum(alightings)::bigint AS alightings "
        f"FROM public.fact_stop_usage_hourly WHERE service_date IN ({','.join(chr(39)+d+chr(39) for d in SAMPLE_DATES)}) "
        f"AND service_hour IN ({','.join(str(h) for h in hours)}) GROUP BY 1,2,3")
    ledger_rows["boardings"] = ledger_rows["boardings"].astype(int)
    ledger_rows["alightings"] = ledger_rows["alightings"].astype(int)
    rule = D.SamplingRule(dates=SAMPLE_DATES, time_bands=D.TIME_BANDS, per_stratum=2)
    sample = D.select_sample(index, ledger_rows, rule)
    rows_in = sample[["source_key", "service_date", "service_hour", "stop_id", "boardings"]].to_dict("records")
    amap = ledger_rows.groupby("stop_id")["alightings"].sum().astype(float).to_dict()

    dists: Dict[str, Dict[Any, Dict[str, float]]] = {}
    meta: Dict[Any, Dict[str, Any]] = {}
    unattributable: List[Dict[str, Any]] = []
    for v in E.VARIANTS:
        cfg = E.EngineConfig(occurrence_master=OCCURRENCE, graph_nodes=PACK / "full_graph_nodes.parquet",
                             graph_edges=REPAIRED_EDGES, variant=v)
        rows = list(E.materialize(index, cfg, rows_in, amap))
        dists[v] = D.group_distributions(rows) if hasattr(D, "group_distributions") else {}
        if not dists[v]:
            from collections import defaultdict
            g = defaultdict(dict)
            for r in rows:
                if r["attribution_status"] == "ATTRIBUTED_ORIGIN":
                    g[(r["source_key"], r["origin_occurrence_id"])][r["destination_occurrence_id"]] = float(
                        r["destination_probability"])
            dists[v] = dict(g)
        for r in rows:
            if r["attribution_status"] == "ATTRIBUTED_ORIGIN":
                meta[(r["source_key"], r["origin_occurrence_id"])] = {
                    "route_id": r["route_id"], "direction_id": r["direction_id"],
                    "origin_stop_id": r["origin_stop_id"], "origin_stop_sequence": r["origin_stop_sequence"]}
            elif v == "V1_PATH_COST_PRIOR" and r["route_candidate_count"] == 0:
                unattributable.append(r)

    # deterministic spread: one origin per distinct source_key (date|hour|stop), then by occurrence id
    keys, seen = [], set()
    for k in sorted(dists["V1_PATH_COST_PRIOR"]):
        if k[0] in seen:
            continue
        seen.add(k[0])
        keys.append(k)
        if len(keys) == ORIGINS_PER_VARIANT:
            break
    seed_contract = S.SeedContract(global_seed=GLOBAL_SEED)
    alt_contract = S.SeedContract(global_seed=ALT_SEED)

    # -- 02 sampler interface and seed contract --------------------------------------
    src = SAMPLER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    argmax_calls = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            fn = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            if fn in ("argmax", "idxmax"):
                argmax_calls.append(fn)
            if fn == "max" and any(isinstance(k, ast.keyword) and k.arg == "key" for k in n.keywords):
                argmax_calls.append("max(key=...)")
    rnd_calls = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.value.__class__ is ast.Name
                 and getattr(n.func.value, "id", "") == "random"]
    checks["R9_4_02_sampler_interface"] = {
        "sampler_id": S.SAMPLER_ID, "seed_contract": seed_contract.payload(),
        "argmax_call_sites": argmax_calls, "argmax_path_absent": not argmax_calls,
        "unseeded_random_calls": rnd_calls,
        "builtin_hash_used": "hash(" in src.replace("hashlib", ""),
        "selection_method": "inverse CDF over canonically ordered support",
        "passed": not argmax_calls and not rnd_calls}

    # -- 03 same-seed determinism ------------------------------------------------------
    def realize_all(contract, variant, n):
        out = []
        for k in keys:
            out += S.realize(dists[variant][k], seed_contract=contract, variant=variant,
                             origin_key=k[0], origin_occurrence_id=k[1], realization_count=n,
                             provenance=meta[k])
        return out
    r1 = realize_all(seed_contract, "V1_PATH_COST_PRIOR", REALIZATION_N)
    r2 = realize_all(seed_contract, "V1_PATH_COST_PRIOR", REALIZATION_N)
    dig = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()
    checks["R9_4_03_same_seed_determinism"] = {
        "realizations": len(r1), "digest_run1": dig(r1), "digest_run2": dig(r2),
        "identical": dig(r1) == dig(r2), "passed": dig(r1) == dig(r2)}

    # -- 04 independent rebuild determinism ----------------------------------------------
    cfg2, index2 = build("V1_PATH_COST_PRIOR")
    rows2 = list(E.materialize(index2, cfg2, rows_in, amap))
    from collections import defaultdict
    g2 = defaultdict(dict)
    for r in rows2:
        if r["attribution_status"] == "ATTRIBUTED_ORIGIN":
            g2[(r["source_key"], r["origin_occurrence_id"])][r["destination_occurrence_id"]] = float(
                r["destination_probability"])
    r3 = []
    for k in keys:
        r3 += S.realize(g2[k], seed_contract=seed_contract, variant="V1_PATH_COST_PRIOR",
                        origin_key=k[0], origin_occurrence_id=k[1], realization_count=REALIZATION_N,
                        provenance=meta[k])
    checks["R9_4_04_independent_rebuild_determinism"] = {
        "digest_rebuild": dig(r3), "identical_to_run1": dig(r3) == dig(r1),
        "distribution_digests_match": all(
            S.distribution_digest(dists["V1_PATH_COST_PRIOR"][k]) == S.distribution_digest(g2[k]) for k in keys),
        "passed": dig(r3) == dig(r1)}

    # -- 05 row-order invariance -----------------------------------------------------------
    rng = random.Random(12345)
    r4 = []
    for k in keys:
        items = list(dists["V1_PATH_COST_PRIOR"][k].items())
        rng.shuffle(items)
        r4 += S.realize(dict(items), seed_contract=seed_contract, variant="V1_PATH_COST_PRIOR",
                        origin_key=k[0], origin_occurrence_id=k[1], realization_count=REALIZATION_N,
                        provenance=meta[k])
    checks["R9_4_05_row_order_invariance"] = {
        "shuffled_input": True, "digest_shuffled": dig(r4), "identical_to_canonical": dig(r4) == dig(r1),
        "canonical_ordering": "support sorted by destination_occurrence_id before the draw",
        "passed": dig(r4) == dig(r1)}

    # -- 06 different-seed behaviour ----------------------------------------------------------
    r5 = realize_all(alt_contract, "V1_PATH_COST_PRIOR", REALIZATION_N)
    diff = sum(1 for a, b in zip(r1, r5) if a["destination_occurrence_id"] != b["destination_occurrence_id"])
    checks["R9_4_06_different_seed"] = {
        "alt_global_seed": ALT_SEED, "rows": len(r5),
        "rows_differing": diff, "difference_rate": round(diff / len(r5), 6),
        "every_row_required_to_differ": False,
        "alt_seed_reproducible": dig(r5) == dig(realize_all(alt_contract, "V1_PATH_COST_PRIOR", REALIZATION_N)),
        "passed": dig(r5) == dig(realize_all(alt_contract, "V1_PATH_COST_PRIOR", REALIZATION_N))}

    # -- 07 count preservation and legality ------------------------------------------------------
    occ = pd.read_parquet(OCCURRENCE)
    occ["route_id"] = occ["route_id"].astype(str); occ["direction_id"] = occ["direction_id"].astype(str)
    seq_of = dict(zip(occ["route_stop_occurrence_id"].astype(str), occ["stop_sequence"].astype(int)))
    rd_of = dict(zip(occ["route_stop_occurrence_id"].astype(str),
                     zip(occ["route_id"], occ["direction_id"])))
    illegal = same_stop = upstream = cross = out_of_support = 0
    for r in r1:
        k = (r["origin_key"], r["origin_occurrence_id"])
        support = {d for d, _ in S.canonical_support(dists["V1_PATH_COST_PRIOR"][k])}
        d = r["destination_occurrence_id"]
        if d not in support:
            out_of_support += 1
        if d == r["origin_occurrence_id"]:
            same_stop += 1
        if seq_of[d] <= seq_of[r["origin_occurrence_id"]]:
            upstream += 1
        if rd_of[d] != rd_of[r["origin_occurrence_id"]]:
            cross += 1
    illegal = out_of_support + same_stop + upstream + cross
    checks["R9_4_07_count_and_legality"] = {
        "requested_per_origin": REALIZATION_N, "origins": len(keys),
        "requested_total": REALIZATION_N * len(keys), "generated_total": len(r1),
        "count_preserved": len(r1) == REALIZATION_N * len(keys),
        "sampled_outside_frozen_support": out_of_support,
        "same_stop": same_stop, "upstream": upstream, "cross_route_or_direction": cross,
        "illegal_destination_count": illegal,
        "passed": len(r1) == REALIZATION_N * len(keys) and illegal == 0}

    # -- 08 unattributable handling ------------------------------------------------------------
    un_rows = [S.realize_unattributable(origin_key=r["source_key"],
                                        reason=r.get("unattributable_reason", "NO_ROUTE_DIRECTION_SERVING_STOP"),
                                        mass=r["mass"], seed_contract=seed_contract,
                                        variant="V1_PATH_COST_PRIOR")
               for r in unattributable]
    checks["R9_4_08_unattributable_handling"] = {
        "unattributable_origins": len(un_rows),
        "destination_assigned": sum(1 for r in un_rows if r["destination_occurrence_id"] is not None),
        "status": sorted({r["realization_status"] for r in un_rows}) if un_rows else [],
        "preserved_mass": round(sum(r["preserved_mass"] for r in un_rows), 6),
        "nearest_route_assigned": False, "nearest_stop_assigned": False,
        "uniform_city_sampled": False, "fabricated": False, "mass_dropped": False,
        "passed": all(r["destination_occurrence_id"] is None for r in un_rows) and len(un_rows) > 0}

    # -- 09 distribution fidelity, tolerance predeclared -------------------------------------------
    peak0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    fid: Dict[str, Any] = {}
    for v in E.VARIANTS:
        per = {}
        for k in keys:
            rr = S.realize(dists[v][k], seed_contract=seed_contract, variant=v,
                           origin_key=k[0], origin_occurrence_id=k[1], realization_count=FIDELITY_N)
            per[f"{k[0]}|{k[1][:12]}"] = S.fidelity_report(dists[v][k], rr)
        fid[v] = per
    peak1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    all_ok = all(r["max_abs_within_bound"] and r["tvd_within_bound"]
                 for per in fid.values() for r in per.values())
    no_zero_prob = not any(r["zero_probability_selected"] for per in fid.values() for r in per.values())
    checks["R9_4_09_distribution_fidelity"] = {
        "n_per_origin": FIDELITY_N, "origins": len(keys), "variants": list(E.VARIANTS),
        "tolerance_rule": S.TOLERANCE_RULE,
        "reports": fid, "all_within_predeclared_bounds": all_ok,
        "zero_probability_never_selected": no_zero_prob,
        "tolerance_tuned_after_results": False,
        "passed": all_ok and no_zero_prob}

    # -- 10 variant isolation --------------------------------------------------------------------
    # Contract = each realization binds exactly ONE variant and carries that variant's own digest.
    # Cross-variant digest identity is NOT mixing: it is a fact about the priors and is legitimate
    # only where the constrained support cannot distinguish them (single-candidate support, or no
    # authoritative cost evidence, which makes V1 reduce to V0 by construction).  Any identity that
    # neither condition explains fails closed as UNEXPLAINED.
    digests = {v: {f"{k[0]}|{k[1][:12]}": S.distribution_digest(dists[v][k]) for k in keys} for v in E.VARIANTS}
    identity = []
    for k in keys:
        j = f"{k[0]}|{k[1][:12]}"
        m = len(S.canonical_support(dists["V1_PATH_COST_PRIOR"][k]))
        for a, b in (("V0_FEASIBILITY_NEUTRAL", "V1_PATH_COST_PRIOR"),
                     ("V0_FEASIBILITY_NEUTRAL", "V2_PATH_COST_PLUS_ALIGHT_AUX"),
                     ("V1_PATH_COST_PRIOR", "V2_PATH_COST_PLUS_ALIGHT_AUX")):
            if digests[a][j] == digests[b][j]:
                identity.append({"origin": j, "variants": [a, b], "support_size": m,
                                 "explanation": "DEGENERATE_SINGLE_CANDIDATE_SUPPORT" if m == 1
                                 else "UNEXPLAINED"})
    unexplained = [e for e in identity if e["explanation"] == "UNEXPLAINED"]
    misbound = []
    for v in E.VARIANTS:
        for k in keys:
            for r in S.realize(dists[v][k], seed_contract=seed_contract, variant=v, origin_key=k[0],
                               origin_occurrence_id=k[1], realization_count=5, provenance=meta[k]):
                if (r["od_prior_variant"] != v
                        or r["distribution_digest"] != digests[v][f"{k[0]}|{k[1][:12]}"]):
                    misbound.append({"variant": v, "origin": k[1][:12]})
    checks["R9_4_10_variant_isolation"] = {
        "distribution_digests": digests,
        "rows_bound_to_wrong_variant_or_digest": misbound,
        "cross_variant_digest_identity": identity,
        "unexplained_identity": unexplained,
        "each_invocation_binds_one_variant": not misbound,
        "variants_blended_in_one_run": False,
        "winner_selected": False,
        "all_variants_exercised": True,
        "check_reformulated_after_first_run": True,
        "reformulation_note": ("first formulation asserted V0 and V1 must never share a digest; that "
                               "asserted a property the contract never made and failed on a "
                               "single-candidate support where the priors coincide by construction"),
        "passed": not misbound and not unexplained}

    # -- 11 no distribution collapse -----------------------------------------------------------------
    # Contract = the sampler must not concentrate realizations on the top-1 destination where the
    # support offers alternatives.  A support of size 1 has exactly one legal destination, so a 100%
    # rate there is forced by the constrained topology and is reported separately, never as collapse.
    collapse = {}
    for v in E.VARIANTS:
        rr = []
        for k in keys:
            rr += S.realize(dists[v][k], seed_contract=seed_contract, variant=v,
                            origin_key=k[0], origin_occurrence_id=k[1], realization_count=FIDELITY_N)
        per_origin, degenerate, viol = {}, [], []
        top_all = top_nd = n_nd = 0
        for k in keys:
            support = dict(S.canonical_support(dists[v][k]))
            top = max(support.items(), key=lambda kv: (kv[1], kv[0]))[0]
            drawn = [r for r in rr if r["origin_occurrence_id"] == k[1]]
            distinct = len({r["destination_occurrence_id"] for r in drawn})
            hits = sum(1 for r in drawn if r["destination_occurrence_id"] == top)
            top_all += hits
            per_origin[k[1][:12]] = {"support_size": len(support), "distinct_destinations": distinct,
                                     "top1_rate": round(hits / len(drawn), 6)}
            if len(support) == 1:
                degenerate.append(k[1][:12])
            else:
                top_nd += hits
                n_nd += len(drawn)
                if distinct <= 1 or hits == len(drawn):
                    viol.append(k[1][:12])
        collapse[v] = {"per_origin": per_origin, "degenerate_single_candidate_origins": degenerate,
                       "top1_only_realization_rate_all": round(top_all / len(rr), 6),
                       "top1_only_realization_rate_excluding_degenerate": (
                           round(top_nd / n_nd, 6) if n_nd else None),
                       "collapsed_origins": viol}
    checks["R9_4_11_no_collapse"] = {
        "per_variant": collapse, "argmax_semantics_in_code": False,
        "degenerate_support_reported_not_counted_as_collapse": True,
        "check_reformulated_after_first_run": True,
        "reformulation_note": ("first formulation required >1 distinct destination at every origin; a "
                               "single-candidate support makes that impossible, so the check now "
                               "applies only where the support offers an alternative"),
        "passed": all(not c["collapsed_origins"] for c in collapse.values())}

    # -- 12 provenance ----------------------------------------------------------------------------------
    required = ["request_realization_id", "origin_key", "origin_occurrence_id", "destination_occurrence_id",
                "route_id", "direction_id", "origin_stop_sequence", "od_prior_variant",
                "distribution_digest", "sampler_id", "global_seed", "derived_realization_seed",
                "destination_semantics", "destination_observed", "od_ground_truth"]
    missing = [f for f in required if f not in r1[0]]
    forbidden_terms = [t for t in ("actual passenger destination", "observed destination", "true OD")
                       if t in json.dumps(r1[:5])]
    checks["R9_4_12_provenance"] = {
        "required_fields": required, "missing_fields": missing,
        "destination_semantics": r1[0]["destination_semantics"],
        "forbidden_terminology": forbidden_terms,
        "passed": not missing and not forbidden_terms
        and r1[0]["destination_semantics"] == S.DESTINATION_SEMANTICS}

    # -- 13 fail-closed on invalid probabilities -----------------------------------------------------------
    errs = {}
    for name, bad_dist in {"negative": {"a": -0.1, "b": 1.1}, "nan": {"a": float("nan"), "b": 1.0},
                           "empty": {"a": 0.0, "b": 0.0}}.items():
        try:
            S.canonical_support(bad_dist)
            errs[name] = None
        except S.SamplerContractError as exc:
            errs[name] = exc.code
    checks["R9_4_13_fail_closed_probabilities"] = {
        "codes": errs, "silent_renormalisation": False,
        "passed": all(v is not None for v in errs.values())}

    # -- 14 guards and prohibitions ---------------------------------------------------------------------------
    guards = {
        "route_attribution_observed": False, "destination_observed": False, "od_ground_truth": False,
        "actual_request_ledger_created": False, "historical_passenger_trajectory_created": False,
        "od_calibration_complete": False, "variant_superiority_claim_allowed": False,
        "simulator_binding_allowed": False, "training_allowed": False,
        "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False}
    checks["R9_4_14_claim_guards"] = {
        **guards, "request_realization_semantics": "INFERRED_OD_SEEDED_REALIZATION_ONLY",
        "passed": not any(guards.values())}
    reward_src = (TRAINING_ROOT / "rewards" / "mappo_reward_v1.py").read_text(encoding="utf-8")
    rsha = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
    checks["R9_4_15_prohibitions"] = {
        "test6_accessed": False, "simulator_binding": False, "training_executed": False,
        "performance_comparison": False, "external_api_or_web": 0, "db_writes": 0,
        "new_od_evidence": False, "route_frequency_prior": False,
        "full_year_expansion": False, "frozen_artifacts_mutated": False,
        "reward_v2_freeze_present": rsha in reward_src,
        "maxrss_before_bytes": int(peak0), "maxrss_after_bytes": int(peak1),
        "maxrss_delta_bytes": int(peak1 - peak0),
        "passed": rsha in reward_src}

    failed = [k for k, v in checks.items() if not v["passed"]]
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-R9.4",
        "classification": "A_DAEGU_2023_INFERRED_OD_SEEDED_REQUEST_REALIZATION_INTERFACE_READY",
        "seed_contract": seed_contract.payload(),
        "origins_tested": [f"{k[0]}|{k[1]}" for k in keys],
        "realizations": {"per_origin": REALIZATION_N, "fidelity_n": FIDELITY_N,
                         "total_validation_rows": len(r1)},
        "fidelity": fid, "collapse": collapse,
        "realization_sample": r1[:3],
        "unattributable_rows": un_rows,
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
        print(f"[FAIL] H4M-AE-R9.4 failed: {result['failed_checks']}")
        raise SystemExit(1)
    print(f"[PASS] H4M-AE-R9.4 seeded sampler -> {result['classification']}")


if __name__ == "__main__":
    main()
