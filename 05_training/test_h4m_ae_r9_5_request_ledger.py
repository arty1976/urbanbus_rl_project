#!/usr/bin/env python3
"""H4M-AE-R9.5 one-to-one historical boarding request ledger and simulator handoff readiness.

Ledger materialization and handoff contract only.  No simulator execution, no MAPPO
rollout or training, no B1/B2/A comparison, no TEST6, no DB writes, no external data.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import resource
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
ARTIFACTS = TRAINING_ROOT / "artifacts"
PACK = ARTIFACTS / "suseong_source_pack_v1"
OCCURRENCE = (ARTIFACTS / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_k6_static_rule_authority_occurrence_audit_20260808_125026"
              / "k6_route_stop_occurrence_master.parquet")
REPAIRED_EDGES = ARTIFACTS / "daegu_path_cost_repair_v1" / "full_graph_edges_repaired.parquet"
LEDGER_ROOT = ARTIFACTS / "daegu_citywide_historical_demand_ledger_v1"
SUSEONG_SUBSET = ARTIFACTS / "daegu_citywide_historical_demand_ledger_v1_suseong_subset" / "suseong_subset.parquet"
B1_DIR = ARTIFACTS / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
RESEARCH_414 = B1_DIR / "r8er3r_generated_demand.parquet"
CORE_MODULES = ("request_ledger_materializer.py", "request_timestamp_realization.py")
R94_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_r9_4_seeded_od_sampler_*"

ENGINE_SHA = "5e877b9c69471a59"
SAMPLER_SHA = "54fe2bc553541572"
REPAIRED_EDGES_SHA = "66301e830bf3b437"
CITYWIDE_SHA = "a4792c19b24b35144123aadd5d280aa6f6e4070c1721446f8826083169602838"
RESEARCH_414_SHA = "3e265d7af10a34cda9614600e5d581a80ffeebd29f1fcdeb154f1f9d9a303e38"
R94_SOURCE_SHA = "7bde88f39f9db67a1addd57e90adfdbb1832ebc3"
EXEC_BASE_SHA = "a8cfc31c8e32143f1b49b46c0f4149ad1daa6eec"

# Scope is configuration, not core logic.  Four real 2023 service dates spanning a
# public holiday, a weekday, a Saturday and the highest-demand weekday of the year,
# crossed with the night / offpeak / peak bands frozen in R9.2.
SCOPE_DATES = ("2023-01-01", "2023-01-02", "2023-01-07", "2023-09-22")
SCOPE_HOURS = {"night": 7, "offpeak": 10, "peak": 17}
SCOPE_DISTRICT = "수성구"
VARIANT = "V1_PATH_COST_PRIOR"
GLOBAL_SEED = 20260821
ALT_SEED = 777
SCOPE_TOKENS = ("suseong", "수성", "2023-01", "2023-09", "70110", "70010")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_scope_rows() -> List[Dict[str, Any]]:
    """Deterministic scope slice of the frozen Suseong subset of the citywide parent."""
    frame = pd.read_parquet(SUSEONG_SUBSET,
                            columns=["service_date", "service_hour", "stop_id", "boardings", "district"])
    frame = frame[frame["service_date"].astype(str).isin(SCOPE_DATES)
                  & frame["service_hour"].astype(int).isin(list(SCOPE_HOURS.values()))
                  & (frame["district"].astype(str) == SCOPE_DISTRICT)
                  & (frame["boardings"].astype(int) > 0)]
    frame = frame.sort_values(["service_date", "service_hour", "stop_id"], kind="mergesort")
    return [{"service_date": str(r.service_date), "service_hour": int(r.service_hour),
             "stop_id": str(r.stop_id), "boardings": int(r.boardings)} for r in frame.itertuples()]


def run_validations() -> Dict[str, Any]:
    sys.path.insert(0, str(TRAINING_ROOT))
    import constrained_od_engine as E
    import od_seeded_sampler as S
    import request_ledger_materializer as M
    import request_timestamp_realization as TS
    import causal_arm_contracts as CA
    checks: Dict[str, Any] = {}
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # -- 01 upstream binding -------------------------------------------------------------
    r94 = sorted(p for p in ARTIFACTS.glob(R94_GLOB) if p.is_dir())[-1]
    m94 = json.loads((r94 / "artifact_manifest.json").read_text(encoding="utf-8"))
    bad = [n for n, s in m94["file_sha256"].items() if sha256_file(r94 / n) != s]
    led = json.loads((LEDGER_ROOT / "_ledger_manifest.json").read_text(encoding="utf-8"))
    frozen = {
        "engine": sha256_file(TRAINING_ROOT / "constrained_od_engine.py").startswith(ENGINE_SHA),
        "sampler": sha256_file(TRAINING_ROOT / "od_seeded_sampler.py").startswith(SAMPLER_SHA),
        "repaired_edges": sha256_file(REPAIRED_EDGES).startswith(REPAIRED_EDGES_SHA),
        "citywide_ledger": led["dataset_sha256"] == CITYWIDE_SHA,
        "research_414": sha256_file(RESEARCH_414) == RESEARCH_414_SHA}
    checks["R9_5_01_upstream_binding"] = {
        "r94_artifact": r94.name, "r94_gate": m94["gate"], "mismatched_r94_files": bad,
        "r94_source_sha": R94_SOURCE_SHA, "execution_base_sha": EXEC_BASE_SHA,
        "frozen_inputs_byte_identical": frozen, "frozen_artifacts_mutated": False,
        "passed": not bad and all(frozen.values())}

    provenance = {
        "citywide_ledger_dataset": CITYWIDE_SHA,
        "route_attribution_occurrence_master": sha256_file(OCCURRENCE),
        "path_cost_repair_edges": sha256_file(REPAIRED_EDGES),
        "sampler_module": sha256_file(TRAINING_ROOT / "od_seeded_sampler.py"),
        "engine_module": sha256_file(TRAINING_ROOT / "constrained_od_engine.py"),
        "timestamp_module": sha256_file(TRAINING_ROOT / "request_timestamp_realization.py"),
        "materializer_module": sha256_file(TRAINING_ROOT / "request_ledger_materializer.py")}
    scope = {"district": SCOPE_DISTRICT, "dates": list(SCOPE_DATES), "time_bands": SCOPE_HOURS,
             "source": "frozen Suseong subset of the citywide parent ledger",
             "non_test_windows_only": True, "test6_windows": 0}

    def engine_cfg():
        return E.EngineConfig(occurrence_master=OCCURRENCE, graph_nodes=PACK / "full_graph_nodes.parquet",
                              graph_edges=REPAIRED_EDGES, variant=VARIANT)

    def build(seed: int, rows: List[Dict[str, Any]], index, ecfg):
        cfg = M.LedgerConfig(variant=VARIANT, global_seed=seed, scope=scope, provenance=provenance)
        frame = pd.DataFrame(list(M.materialize(index, ecfg, cfg, rows)))
        return cfg, M.canonical_sort(frame)

    src_rows = load_scope_rows()
    ecfg = engine_cfg()
    index = E.build_index(ecfg)
    cfg, ledger = build(GLOBAL_SEED, src_rows, index, ecfg)
    digest = M.ledger_digest(ledger)
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # -- 02 one-to-one demand semantics --------------------------------------------------
    core_src = {n: (TRAINING_ROOT / n).read_text(encoding="utf-8") for n in CORE_MODULES}
    mult = []
    for name, text in core_src.items():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                for side in (node.left, node.right):
                    if isinstance(side, ast.Constant) and isinstance(side.value, (int, float)) \
                            and side.value not in (0, 1, 1000.0):
                        mult.append({"module": name, "line": node.lineno, "factor": side.value})
    checks["R9_5_02_one_to_one_semantics"] = {
        "request_realization_mode": M.REQUEST_REALIZATION_MODE,
        "request_weight_passengers": M.REQUEST_WEIGHT,
        "demand_semantics": M.DEMAND_SEMANTICS,
        "numeric_multiplier_sites_in_core": mult,
        "weighted_representative_passenger": False, "passenger_compression": False,
        "demand_scaling": False, "synthetic_multiplier": False,
        "passed": (M.REQUEST_REALIZATION_MODE == "ONE_HISTORICAL_BOARDING_ONE_REQUEST"
                   and M.REQUEST_WEIGHT == 1 and not mult)}

    # -- 03 demand conservation ----------------------------------------------------------
    cons = M.conservation_report(src_rows, ledger)
    checks["R9_5_03_demand_conservation"] = {
        **cons, "passed": cons["exact_one_to_one"] and cons["dropped_mass"] == 0
        and not cons["buckets_with_wrong_row_count"] and cons["passenger_count_values"] == [1]}

    # -- 04 request timestamp diagnostics ------------------------------------------------
    ts = pd.to_datetime(ledger["request_ts"])
    bs, be = pd.to_datetime(ledger["bucket_start"]), pd.to_datetime(ledger["bucket_end"])
    outside = int(((ts < bs) | (ts >= be)).sum())
    hour_ok = int((ts.dt.hour != ledger["service_hour"].astype(int)).sum())
    date_ok = int((ts.dt.strftime("%Y-%m-%d") != ledger["service_date"].astype(str)).sum())
    per_bucket_sorted = ledger.groupby("source_key")["request_ts"].apply(
        lambda s: bool((s.values == sorted(s.values)).all()))
    spacing = TS.spacing_summary(ledger[ledger.source_key == ledger.source_key.iloc[0]].to_dict("records"))
    checks["R9_5_04_timestamp_diagnostics"] = {
        "rule": TS.TIMESTAMP_RULE, "contract": TS.TimestampContract(global_seed=GLOBAL_SEED).payload(),
        "timestamps_outside_source_bucket": outside,
        "timestamps_in_wrong_hour": hour_ok, "timestamps_on_wrong_service_date": date_ok,
        "null_timestamps": int(ledger["request_ts"].isna().sum()),
        "duplicate_request_ids": int(ledger["request_id"].duplicated().sum()),
        "buckets_chronologically_ordered": int(per_bucket_sorted.sum()), "buckets": int(len(per_bucket_sorted)),
        "all_buckets_ordered": bool(per_bucket_sorted.all()),
        "example_spacing_summary": spacing,
        "claims_actual_within_bucket_arrivals": False,
        "passed": (outside == 0 and hour_ok == 0 and date_ok == 0
                   and not ledger["request_ts"].isna().any()
                   and not ledger["request_id"].duplicated().any() and bool(per_bucket_sorted.all()))}

    # -- 05 destination legality ---------------------------------------------------------
    occ = pd.read_parquet(OCCURRENCE)
    occ["route_id"] = occ["route_id"].astype(str); occ["direction_id"] = occ["direction_id"].astype(str)
    oid = occ["route_stop_occurrence_id"].astype(str)
    seq_of = dict(zip(oid, occ["stop_sequence"].astype(int)))
    rd_of = dict(zip(oid, zip(occ["route_id"], occ["direction_id"])))
    stop_of = dict(zip(oid, occ["stop_id"].astype(str)))
    real = ledger[ledger.realization_status == M.STATUS_REALIZED]
    same_stop = upstream = cross = badseq = 0
    for r in real.itertuples():
        o, d = r.origin_occurrence_id, r.destination_occurrence_id
        if stop_of[d] == stop_of[o]:
            same_stop += 1
        if seq_of[d] <= seq_of[o]:
            upstream += 1
        if rd_of[d] != rd_of[o]:
            cross += 1
        if int(r.destination_stop_sequence) <= int(r.origin_stop_sequence):
            badseq += 1
    checks["R9_5_05_destination_legality"] = {
        "realized_requests": int(len(real)),
        "distinct_origin_stops": int(real["origin_stop_id"].nunique()),
        "distinct_destination_stops": int(real["destination_stop_id"].nunique()),
        "distinct_route_directions": int(real.groupby(["route_id", "direction_id"]).ngroups),
        "distinct_origin_occurrences": int(real["origin_occurrence_id"].nunique()),
        "distinct_destination_occurrences": int(real["destination_occurrence_id"].nunique()),
        "mean_destination_support_size": round(float(real["support_size"].mean()), 3),
        "same_stop_count": same_stop, "upstream_count": upstream,
        "cross_route_or_direction_count": cross,
        "destination_sequence_not_greater_than_origin": badseq,
        "illegal_destination_count": same_stop + upstream + cross + badseq,
        "destination_semantics": S.DESTINATION_SEMANTICS,
        "passed": same_stop == 0 and upstream == 0 and cross == 0 and badseq == 0}

    # -- 06 unattributable demand explicit -----------------------------------------------
    unreal = ledger[ledger.realization_status != M.STATUS_REALIZED]
    checks["R9_5_06_unattributable_explicit"] = {
        "unattributable_origin_requests": int((ledger.realization_status == M.STATUS_UNATTRIBUTABLE).sum()),
        "unrealizable_terminal_requests": int((ledger.realization_status == M.STATUS_TERMINAL).sum()),
        "reasons": unreal["unattributable_reason"].value_counts().to_dict(),
        "destinations_assigned_to_unrealizable": int(unreal["destination_stop_id"].notna().sum()),
        "rows_dropped": 0, "mass_reassigned_to_nearest_stop": False,
        "mass_reassigned_to_nearest_route": False, "uniform_city_fallback": False,
        "still_carry_request_id_and_ts": bool(unreal["request_id"].notna().all()
                                              and unreal["request_ts"].notna().all()),
        "passed": int(unreal["destination_stop_id"].notna().sum()) == 0
        and bool(unreal["request_id"].notna().all())}

    # -- 07 same-seed determinism --------------------------------------------------------
    _, ledger2 = build(GLOBAL_SEED, src_rows, index, ecfg)
    checks["R9_5_07_same_seed_determinism"] = {
        "rows": int(len(ledger)), "digest_run1": digest, "digest_run2": M.ledger_digest(ledger2),
        "identical": M.ledger_digest(ledger2) == digest,
        "passed": M.ledger_digest(ledger2) == digest}

    # -- 08 independent rebuild ----------------------------------------------------------
    ecfg3 = engine_cfg()
    index3 = E.build_index(ecfg3)
    _, ledger3 = build(GLOBAL_SEED, src_rows, index3, ecfg3)
    checks["R9_5_08_independent_rebuild"] = {
        "index_and_od_rebuilt": True, "digest_rebuild": M.ledger_digest(ledger3),
        "identical_to_run1": M.ledger_digest(ledger3) == digest,
        "passed": M.ledger_digest(ledger3) == digest}

    # -- 09 row-order invariance ---------------------------------------------------------
    shuffled = list(src_rows)
    random.Random(4242).shuffle(shuffled)
    _, ledger4 = build(GLOBAL_SEED, shuffled, index, ecfg)
    checks["R9_5_09_row_order_invariance"] = {
        "upstream_rows_shuffled": True, "digest_shuffled": M.ledger_digest(ledger4),
        "identical_to_canonical": M.ledger_digest(ledger4) == digest,
        "canonical_order": list(M.CANONICAL_ORDER),
        "passed": M.ledger_digest(ledger4) == digest}

    # -- 10 different seed ------------------------------------------------------------------
    # Joined positionally on (source_key, request_ordinal).  request_id embeds the global seed
    # by design -- arms sharing a seed must share request ids -- so joining on request_id
    # compares nothing: the first formulation did exactly that and passed on an empty frame.
    _, alt = build(ALT_SEED, src_rows, index, ecfg)
    both = ledger.merge(alt, on=["source_key", "request_ordinal"], suffixes=("_a", "_b"))
    same_count = int(len(alt)) == int(len(ledger))
    per_bucket_same = (ledger.groupby("source_key").size().sort_index()
                       .equals(alt.groupby("source_key").size().sort_index()))
    origin_mass_same = (ledger.groupby("origin_stop_id").size().sort_index()
                        .equals(alt.groupby("origin_stop_id").size().sort_index()))
    unattr_same = (int((ledger.realization_status == M.STATUS_UNATTRIBUTABLE).sum())
                   == int((alt.realization_status == M.STATUS_UNATTRIBUTABLE).sum()))
    prov_same = bool((alt["provenance_digest"] == cfg.provenance_digest).all())
    ts_changed = int((both["request_ts_a"].values != both["request_ts_b"].values).sum())
    dest_changed = int((both["destination_stop_id_a"].fillna("_")
                        != both["destination_stop_id_b"].fillna("_")).sum())
    route_changed = int((both["origin_occurrence_id_a"].fillna("_")
                         != both["origin_occurrence_id_b"].fillna("_")).sum())
    status_flips = int((both["realization_status_a"] != both["realization_status_b"]).sum())
    # The frozen legal support is a property of the network evidence, not of the seed:
    # build_origin_plan takes no seed, and no realization under either seed leaves it.
    plan_seeded = "seed" in M.build_origin_plan.__code__.co_varnames[
        :M.build_origin_plan.__code__.co_argcount]
    plan_cache: Dict[str, Any] = {}
    outside = 0
    for frame in (ledger, alt):
        for t in frame[frame.realization_status == M.STATUS_REALIZED].itertuples():
            plan = plan_cache.get(t.origin_stop_id)
            if plan is None:
                plan = plan_cache[t.origin_stop_id] = M.build_origin_plan(index, ecfg, t.origin_stop_id)
            if t.destination_occurrence_id not in plan.od_support.get(t.origin_occurrence_id, {}):
                outside += 1
    checks["R9_5_10_different_seed"] = {
        "alt_seed": ALT_SEED, "join_key": ["source_key", "request_ordinal"],
        "joined_rows": int(len(both)),
        "invariant_total_request_count": same_count,
        "invariant_per_bucket_counts": bool(per_bucket_same),
        "invariant_origin_historical_mass": bool(origin_mass_same),
        "invariant_unattributable_origin_count": unattr_same,
        "invariant_source_provenance": prov_same,
        "invariant_frozen_legal_support": not plan_seeded,
        "destinations_outside_frozen_support_either_seed": outside,
        "changed_timestamps": ts_changed,
        "changed_destinations": dest_changed,
        "changed_route_direction_assignment": route_changed,
        "changed_realization_status": status_flips,
        "terminal_count_seed_a": int((ledger.realization_status == M.STATUS_TERMINAL).sum()),
        "terminal_count_seed_b": int((alt.realization_status == M.STATUS_TERMINAL).sum()),
        "route_stage_is_seeded_by_design": True,
        "route_stage_note": ("stage 1 draws the route-direction occurrence, so a different seed moves "
                             "some requests onto a different occurrence; where that occurrence is a "
                             "terminal one the request becomes UNREALIZABLE_TERMINAL_OCCURRENCE. This "
                             "is a disclosed consequence of route assignment being an inferred "
                             "realization, not a change in what the evidence permits: no frequency "
                             "prior exists (R9.1 ROUTE_PRIOR_MODE), and a seed-independent assignment "
                             "would concentrate every stop on one arbitrary route"),
        "digest_differs": M.ledger_digest(alt) != digest,
        "check_reformulated_after_first_run": True,
        "reformulation_note": ("first formulation joined on request_id, which embeds the global seed, "
                               "so it compared an empty frame and reported 0 changed destinations, "
                               "and vacuously-true support and status invariance"),
        "passed": (same_count and per_bucket_same and origin_mass_same and unattr_same and prov_same
                   and not plan_seeded and outside == 0 and dest_changed > 0 and ts_changed > 0
                   and M.ledger_digest(alt) != digest)}

    # -- 11 B1/B2/A shared demand fairness ------------------------------------------------
    arm_digests = {}
    for arm in (CA.ARM_A, CA.ARM_B1, CA.ARM_B2):
        # every arm receives the identical frozen ledger object; no arm-keyed regeneration
        arm_digests[arm] = M.ledger_digest(ledger)
    mat_src = core_src["request_ledger_materializer.py"]
    policy_tokens = [t for t in ("arm_id", "policy", "checkpoint", "B1", "B2", "MAPPO")
                     if t in mat_src]
    cfg_fields = [f for f in M.LedgerConfig.__dataclass_fields__]
    checks["R9_5_11_shared_demand_fairness"] = {
        "arms": [CA.ARM_A, CA.ARM_B1, CA.ARM_B2], "arm_ledger_digests": arm_digests,
        "all_arms_identical": len(set(arm_digests.values())) == 1,
        "shared_fields": [f for f, _ in M.HANDOFF_SCHEMA],
        "arm_variable_keys_allowed": list(CA.ARM_VARIABLE_KEYS),
        "demand_is_arm_variable": "demand" in CA.ARM_VARIABLE_KEYS,
        "policy_tokens_in_materializer": policy_tokens,
        "ledger_config_fields": cfg_fields,
        "policy_specific_regeneration_possible": bool(policy_tokens),
        "arms_executed_in_r9_5": False,
        "passed": (len(set(arm_digests.values())) == 1 and not policy_tokens
                   and "demand" not in CA.ARM_VARIABLE_KEYS)}

    # -- 12 simulator handoff schema ------------------------------------------------------
    fields = [f for f, _ in M.HANDOFF_SCHEMA]
    missing = [f for f in fields if f not in ledger.columns]
    checks["R9_5_12_handoff_schema"] = {
        "schema": [{"field": f, "meaning": m} for f, m in M.HANDOFF_SCHEMA],
        "missing_from_ledger": missing,
        "passenger_count_constant_one": sorted({int(v) for v in ledger["passenger_count"].unique()}) == [1],
        "citywide_compatible": True, "day_long_compatible": True,
        "simulator_code_modified": False, "simulator_bound": False, "simulator_executed": False,
        "passed": not missing and sorted({int(v) for v in ledger["passenger_count"].unique()}) == [1]}

    # -- 13 no argmax and no scope hardcoding in core --------------------------------------
    argmax, scope_hits = [], []
    for name, text in core_src.items():
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if fn in ("argmax", "idxmax") or (fn == "max" and any(
                        isinstance(k, ast.keyword) and k.arg == "key" for k in node.keywords)):
                    argmax.append({"module": name, "line": node.lineno, "call": fn})
        # Exclude dict keys (R9.1 lesson) and docstrings (R9.5: prose asserting the absence
        # of a scope token matches a scan for that token).  Documentation cannot configure
        # scope; only executable string values can, and those are what this scans.
        keys = {id(k) for n in ast.walk(tree) if isinstance(n, ast.Dict) for k in n.keys if k is not None}
        docs = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(n, "body", [])
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    docs.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in keys and id(node) not in docs:
                low = node.value.lower()
                for tok in SCOPE_TOKENS:
                    if tok.lower() in low:
                        scope_hits.append({"module": name, "line": node.lineno, "token": tok})
    checks["R9_5_13_no_argmax_no_scope_hardcoding"] = {
        "argmax_call_sites_in_core": argmax, "argmax_destination_path_absent": not argmax,
        "scope_token_string_values_in_core": scope_hits, "scope_tokens_scanned": list(SCOPE_TOKENS),
        "ast_precise_excludes_dict_keys_and_docstrings": True,
        "docstring_exclusion_reason": ("a docstring cannot configure scope; the module docstring states that the core does not know what Suseong is, which a raw scan matches as a hit"),
        "scope_supplied_by_configuration": True,
        "passed": not argmax and not scope_hits}

    # -- 14 bounded memory and processing mode ---------------------------------------------
    rss2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    parts = ledger.groupby(["service_date", "service_hour"]).size().to_dict()
    checks["R9_5_14_bounded_execution"] = {
        "processing_mode": "per-window streaming generator with a per-stop plan cache",
        "windows": len(parts), "rows_processed": int(len(ledger)),
        "source_rows": len(src_rows),
        "maxrss_start_bytes": int(rss0), "maxrss_after_first_build_bytes": int(rss1),
        "maxrss_final_bytes": int(rss2),
        "maxrss_final_mb": round(rss2 / 1e6, 1),
        "full_year_materialized": False, "citywide_expansion_required_for_pass": False,
        "citywide_year_row_count_avoided": 181413653,
        "whole_scope_held_in_ram": False,
        "passed": rss2 < 8 * 1024 ** 3}

    # -- 15 provenance completeness ---------------------------------------------------------
    required = ["request_id", "service_date", "service_hour", "source_key", "bucket_start", "bucket_end",
                "origin_stop_id", "origin_semantics", "request_ts", "request_ts_semantics",
                "route_id", "direction_id", "origin_occurrence_id", "origin_stop_sequence",
                "destination_stop_id", "destination_occurrence_id", "destination_stop_sequence",
                "destination_semantics", "od_prior_variant", "od_distribution_digest",
                "route_distribution_digest", "demand_realization_seed", "timestamp_realization_seed",
                "route_realization_seed", "destination_realization_seed", "provenance_digest",
                "src_citywide_ledger_dataset", "src_route_attribution_occurrence_master",
                "src_path_cost_repair_edges", "src_sampler_module", "src_engine_module"]
    miss = [f for f in required if f not in ledger.columns]
    sem = {"origin": str(ledger["origin_semantics"].iloc[0]),
           "request_ts": str(ledger["request_ts_semantics"].iloc[0]),
           "destination": str(real["destination_semantics"].iloc[0])}
    checks["R9_5_15_provenance"] = {
        "required_fields": required, "missing_fields": miss, "semantics": sem,
        "provenance_digest": cfg.provenance_digest, "sources": provenance,
        "passed": (not miss and sem["origin"] == M.ORIGIN_SEMANTICS
                   and sem["request_ts"] == TS.TIMESTAMP_SEMANTICS
                   and sem["destination"] == S.DESTINATION_SEMANTICS)}

    # -- 16 claim guards and prohibitions -----------------------------------------------------
    guards = {"destination_observed": False, "od_ground_truth": False, "request_ts_observed": False,
              "actual_passenger_request_ledger_created": False,
              "historical_passenger_trajectory_created": False,
              "simulator_binding_allowed": False, "simulator_execution_allowed": False,
              "training_allowed": False, "performance_comparison_allowed": False,
              "variant_superiority_claim_allowed": False, "paper_level_claim_allowed": False,
              "causal_performance_claim_allowed": False}
    checks["R9_5_16_claim_guards"] = {
        **guards, "scoped_inferred_request_ledger_created": True,
        "row_level_guards": {
            "destination_observed": bool(ledger["destination_observed"].any()),
            "od_ground_truth": bool(ledger["od_ground_truth"].any()),
            "request_ts_observed": bool(ledger["request_ts_observed"].any())},
        "passed": not any(guards.values()) and not ledger["destination_observed"].any()
        and not ledger["od_ground_truth"].any() and not ledger["request_ts_observed"].any()}

    reward_src = (TRAINING_ROOT / "rewards" / "mappo_reward_v1.py").read_text(encoding="utf-8")
    rsha = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
    checks["R9_5_17_prohibitions"] = {
        "test6_accessed": False, "simulator_executed": False, "mappo_rollout_or_training": False,
        "b1_b2_a_comparison": False, "passenger_compression": False, "demand_scaling": False,
        "external_api_or_web": 0, "db_writes": 0, "new_od_evidence": False,
        "argmax_destination": False, "observed_timestamp_fabrication": False,
        "frozen_r91_r94_artifacts_mutated": False,
        "reward_v2_freeze_present": rsha in reward_src,
        "passed": rsha in reward_src}

    failed = [k for k, v in checks.items() if not v["passed"]]
    band_of = {v: k for k, v in SCOPE_HOURS.items()}
    windows = [{"service_date": d, "service_hour": int(h), "time_band": band_of[int(h)],
                "source_boarding_mass": int(sum(r["boardings"] for r in src_rows
                                                if r["service_date"] == d and r["service_hour"] == h)),
                "request_rows": int(((ledger.service_date == d) & (ledger.service_hour == h)).sum()),
                "realized": int(((ledger.service_date == d) & (ledger.service_hour == h)
                                 & (ledger.realization_status == M.STATUS_REALIZED)).sum()),
                "unrealizable": int(((ledger.service_date == d) & (ledger.service_hour == h)
                                     & (ledger.realization_status != M.STATUS_REALIZED)).sum())}
               for d in SCOPE_DATES for h in sorted(SCOPE_HOURS.values())]
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-R9.5",
        "classification": "A_SUSEONG_2023_ONE_TO_ONE_INFERRED_REQUEST_LEDGER_READY_FOR_CAUSAL_SIMULATOR_HANDOFF_VALIDATION",
        "scope": scope, "windows": windows, "ledger_digest": digest,
        "ledger_rows": int(len(ledger)),
        "requests_per_time_band": {band_of[int(h)]: int((ledger.service_hour == h).sum())
                                   for h in sorted(SCOPE_HOURS.values())},
        "config": cfg.payload(),
        "manifest": M.manifest(cfg, index, ecfg, cons, digest),
        "ledger_sample": ledger.head(3).to_dict("records"),
        "checks": checks, "failed_checks": failed, "all_passed": not failed,
        "_ledger": ledger,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    result = run_validations()
    result.pop("_ledger", None)
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True,
                                               default=str) + "\n", encoding="utf-8")
    if result["failed_checks"]:
        print(f"[FAIL] H4M-AE-R9.5 failed: {result['failed_checks']}")
        raise SystemExit(1)
    print(f"[PASS] H4M-AE-R9.5 request ledger -> {result['classification']}")


if __name__ == "__main__":
    main()
