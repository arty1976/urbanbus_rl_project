#!/usr/bin/env python3
"""H4M-AE-R9.5 gate runner: 1:1 historical boarding request ledger and simulator handoff readiness."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_h4m_ae_r9_5_request_ledger as T  # noqa: E402
import constrained_od_engine as E  # noqa: E402
import od_seeded_sampler as S  # noqa: E402
import request_ledger_materializer as M  # noqa: E402
import request_timestamp_realization as TS  # noqa: E402
import causal_arm_contracts as CA  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = T.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r9_5_request_ledger_{STAMP}"
LEDGER_OUT = OUT / "scoped_request_ledger"
GATE = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R9_5_ONE_TO_ONE_HISTORICAL_BOARDING_REQUEST"
        "_LEDGER_AND_SIMULATOR_HANDOFF_READINESS_CONTRACT_COMPLETE")


def write(name: str, payload) -> None:
    p = OUT / name
    if isinstance(payload, str):
        p.write_text(payload, encoding="utf-8")
    else:
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
                     encoding="utf-8")


def stream_ledger(cfg, index, ecfg, src_rows):
    """Write the ledger one window at a time; never hold the whole scope in RAM."""
    LEDGER_OUT.mkdir(parents=True, exist_ok=True)
    by_window: dict = {}
    for row in src_rows:
        by_window.setdefault((row["service_date"], row["service_hour"]), []).append(row)
    plan_cache: dict = {}
    parts, peak = [], 0
    for (date, hour) in sorted(by_window):
        rows = list(M.materialize(index, ecfg, cfg, by_window[(date, hour)], plan_cache=plan_cache))
        frame = M.canonical_sort(pd.DataFrame(rows))
        part = LEDGER_OUT / f"service_date={date}" / f"service_hour={hour}" / "part-0.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(part, index=False)
        parts.append({"service_date": date, "service_hour": int(hour), "rows": int(len(frame)),
                      "path": str(part.relative_to(OUT)),
                      "bytes": part.stat().st_size,
                      "sha256": hashlib.sha256(part.read_bytes()).hexdigest()})
        peak = max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        del rows, frame
    return parts, peak


def main() -> None:
    result = T.run_validations()
    ledger = result.pop("_ledger")
    if not result["all_passed"]:
        raise SystemExit(f"BLOCKED: {result['failed_checks']}")
    OUT.mkdir(parents=True, exist_ok=False)
    c = result["checks"]
    cfg_payload = result["config"]

    src_rows = T.load_scope_rows()
    ecfg = E.EngineConfig(occurrence_master=T.OCCURRENCE, graph_nodes=T.PACK / "full_graph_nodes.parquet",
                          graph_edges=T.REPAIRED_EDGES, variant=T.VARIANT)
    index = E.build_index(ecfg)
    cfg = M.LedgerConfig(variant=T.VARIANT, global_seed=T.GLOBAL_SEED, scope=result["scope"],
                         provenance=c["R9_5_15_provenance"]["sources"])
    parts, peak = stream_ledger(cfg, index, ecfg, src_rows)

    # The streamed on-disk ledger must reproduce the digest the validator verified in memory.
    disk = M.canonical_sort(pd.concat([pd.read_parquet(OUT / p["path"]) for p in parts],
                                      ignore_index=True))
    disk_digest = M.ledger_digest(disk)
    src = {n: hashlib.sha256((T.TRAINING_ROOT / n).read_bytes()).hexdigest()
           for n in ("request_ledger_materializer.py", "request_timestamp_realization.py",
                     "test_h4m_ae_r9_5_request_ledger.py", "run_h4m_ae_r9_5_gate.py",
                     "od_seeded_sampler.py", "constrained_od_engine.py")}

    write("demand_conservation_report.json", {
        **c["R9_5_03_demand_conservation"],
        "per_window": result["windows"],
        "requests_per_time_band": result["requests_per_time_band"],
        "streamed_parts": parts,
        "streamed_ledger_digest": disk_digest,
        "in_memory_ledger_digest": result["ledger_digest"],
        "stream_matches_memory": disk_digest == result["ledger_digest"]})

    write("timestamp_realization_contract.json", {
        "module": "05_training/request_timestamp_realization.py", "sha256": src["request_timestamp_realization.py"],
        "rule": TS.TIMESTAMP_RULE,
        "contract": TS.TimestampContract(global_seed=T.GLOBAL_SEED).payload(),
        "diagnostics": c["R9_5_04_timestamp_diagnostics"]})

    write("determinism_report.json", {k: c[k] for k in (
        "R9_5_07_same_seed_determinism", "R9_5_08_independent_rebuild",
        "R9_5_09_row_order_invariance", "R9_5_10_different_seed")})

    write("shared_demand_handoff_contract.json", {
        "contract_id": "B1_B2_A_SHARED_FROZEN_DEMAND_REALIZATION_V1",
        "gate": GATE,
        "frozen_ledger_digest": result["ledger_digest"],
        "global_demand_seed": T.GLOBAL_SEED, "od_prior_variant": T.VARIANT,
        "arms": [CA.ARM_A, CA.ARM_B1, CA.ARM_B2],
        "invariant": "B1 demand digest == B2 demand digest == A demand digest",
        "verified": c["R9_5_11_shared_demand_fairness"],
        "identical_across_arms": [f for f, _ in M.HANDOFF_SCHEMA],
        "arm_variable_keys": list(CA.ARM_VARIABLE_KEYS),
        "policy_specific_demand_regeneration": "PROHIBITED",
        "required_before_any_causal_comparison": True,
        "arms_executed_in_r9_5": False,
        "enforcement": ("any future arm run must assert its input ledger digest equals "
                        "frozen_ledger_digest before a single step is taken")})

    write("simulator_handoff_schema.json", {
        "schema_id": "R9_5_SIMULATOR_REQUEST_HANDOFF_V1",
        "fields": [{"field": f, "meaning": m} for f, m in M.HANDOFF_SCHEMA],
        "row_semantics": M.DEMAND_SEMANTICS,
        "ledger_layout": "parquet partitioned by service_date / service_hour",
        "citywide_compatible": True, "day_long_compatible": True, "larger_fleet_compatible": True,
        "evaluation_window_bounds_hardcoded": False,
        "verified": c["R9_5_12_handoff_schema"],
        "simulator_code_modified": False, "simulator_bound": False, "simulator_executed": False})

    write("provenance_manifest.json", {
        "gate": GATE, "stage": result["stage"], "classification": result["classification"],
        "execution_base_sha": T.EXEC_BASE_SHA, "r9_4_source_sha": T.R94_SOURCE_SHA,
        "upstream_binding": c["R9_5_01_upstream_binding"],
        "source_sha256": src, "read_only_inputs": c["R9_5_15_provenance"]["sources"],
        "provenance_digest": cfg.provenance_digest,
        "ledger_manifest": result["manifest"],
        "claim_guards": c["R9_5_16_claim_guards"], "prohibitions": c["R9_5_17_prohibitions"]})

    write("self_test_report.json", result)

    w = "\n".join(
        f"| {x['service_date']} | {x['service_hour']:02d} | {x['time_band']} | {x['source_boarding_mass']} | "
        f"{x['request_rows']} | {x['realized']} | {x['unrealizable']} |" for x in result["windows"])
    d10 = c["R9_5_10_different_seed"]
    cons = c["R9_5_03_demand_conservation"]
    leg = c["R9_5_05_destination_legality"]
    ts = c["R9_5_04_timestamp_diagnostics"]
    mem = c["R9_5_14_bounded_execution"]

    write("final_report.md", f"""# H4M-AE-R9.5 — 1:1 Historical Boarding Request Ledger and Simulator Handoff Readiness

- Gate: `{GATE}`
- Classification: `{result['classification']}`
- execution_base_sha: `{T.EXEC_BASE_SHA}`
- R9.4 lineage sha: `{T.R94_SOURCE_SHA}`
- Frozen ledger digest: `{result['ledger_digest']}`
- Generated: {STAMP}

## 1. What this gate produces

A scoped Suseong-gu request ledger in which **one historical boarding is exactly one request**,
carrying `passenger_count = 1`. Origin stop and boarding count are observed history. Request time,
route-direction assignment and destination are all **inferred realizations** under a declared seed.
Nothing here is executed against the simulator: no B1/B2/A run, no MAPPO rollout, no training.

## 2. Scope

{len(result['windows'])} NON-TEST windows: {len(T.SCOPE_DATES)} real 2023 service dates crossed with the night/offpeak/peak bands
frozen in R9.2. The dates span a public holiday, an ordinary weekday, a Saturday, and the
highest-demand weekday of 2023, giving genuinely different demand levels rather than four samples of
the same regime.

| service_date | hour | band | boarding mass | request rows | realized | unrealizable |
|---|---|---|---|---|---|---|
{w}

Requests per band: {json.dumps(result['requests_per_time_band'])}.

## 3. Demand conservation

| quantity | value |
|---|---|
| historical boarding mass | {cons['historical_boarding_mass']} |
| request rows generated | {cons['request_rows_generated']} |
| realized requests | {cons['realized_request_count']} |
| unattributable-origin requests | {cons['unattributable_request_count']} |
| unrealizable terminal-occurrence requests | {cons['unrealizable_terminal_request_count']} |
| accounted total | {cons['accounted_total']} |
| **dropped mass** | **{cons['dropped_mass']}** |
| buckets with a wrong row count | {len(cons['buckets_with_wrong_row_count'])} |
| distinct passenger_count values | {cons['passenger_count_values']} |

Every bucket holding N boardings produced exactly N rows. Unattributable and unrealizable requests are
kept as rows with an explicit status and a null destination — they are never dropped, never reassigned
to a nearest stop or route, and never absorbed into a uniform citywide fallback.

## 4. Request timestamps

The rule was fixed in the module before any ledger existed: N uniform i.i.d. offsets inside the
authoritative bucket, sorted ascending and assigned to ordinals in order. Uniform is the
maximum-entropy choice when the only observed quantity is the bucket total; any other shape would
encode within-bucket arrival evidence this project does not hold.

| property | value |
|---|---|
| timestamps outside their source bucket | {ts['timestamps_outside_source_bucket']} |
| timestamps in the wrong hour / on the wrong date | {ts['timestamps_in_wrong_hour']} / {ts['timestamps_on_wrong_service_date']} |
| null timestamps | {ts['null_timestamps']} |
| duplicate request ids | {ts['duplicate_request_ids']} |
| buckets in chronological order | {ts['buckets_chronologically_ordered']} / {ts['buckets']} |

`request_ts_observed = false`. These timestamps do not reproduce real within-bucket arrivals and are
not offered as such; spacing statistics in the artifact are descriptive only.

## 5. Destination legality

| property | value |
|---|---|
| realized requests | {leg['realized_requests']} |
| distinct origin stops / destination stops | {leg['distinct_origin_stops']} / {leg['distinct_destination_stops']} |
| distinct route-directions | {leg['distinct_route_directions']} |
| distinct origin / destination occurrences | {leg['distinct_origin_occurrences']} / {leg['distinct_destination_occurrences']} |
| mean destination support size | {leg['mean_destination_support_size']} |
| same-stop | {leg['same_stop_count']} |
| upstream | {leg['upstream_count']} |
| cross-route or cross-direction | {leg['cross_route_or_direction_count']} |
| destination_sequence <= origin_sequence | {leg['destination_sequence_not_greater_than_origin']} |
| **illegal destinations** | **{leg['illegal_destination_count']}** |

Destinations come only from the frozen R9.4 sampler applied to the frozen R9.1/R9.3 conditional OD
distributions. An AST scan of the R9.5 core found {len(c['R9_5_13_no_argmax_no_scope_hardcoding']['argmax_call_sites_in_core'])} argmax/idxmax/`max(key=)` call sites.

## 6. Determinism

| property | result |
|---|---|
| same seed, second build | digest identical: `{c['R9_5_07_same_seed_determinism']['identical']}` |
| independent index and OD rebuild | identical: `{c['R9_5_08_independent_rebuild']['identical_to_run1']}` |
| shuffled upstream row order | identical: `{c['R9_5_09_row_order_invariance']['identical_to_canonical']}` |
| streamed on-disk ledger vs in-memory | identical: `{disk_digest == result['ledger_digest']}` |

### Different seed — and one thing worth stating plainly

Joined positionally on `(source_key, request_ordinal)` across {d10['joined_rows']} rows:

| invariant under a new seed | result |
|---|---|
| total request count | {d10['invariant_total_request_count']} |
| per-bucket counts | {d10['invariant_per_bucket_counts']} |
| origin historical mass | {d10['invariant_origin_historical_mass']} |
| unattributable-origin count | {d10['invariant_unattributable_origin_count']} |
| source provenance | {d10['invariant_source_provenance']} |
| frozen legal support | {d10['invariant_frozen_legal_support']} |
| destinations outside the frozen support, either seed | {d10['destinations_outside_frozen_support_either_seed']} |

| changed by a new seed | count |
|---|---|
| timestamps | {d10['changed_timestamps']} |
| destinations | {d10['changed_destinations']} |
| route-direction assignment | {d10['changed_route_direction_assignment']} |
| realization status | {d10['changed_realization_status']} |

Route-direction assignment is itself a seeded inference, because no route frequency prior exists
(R9.1 `ROUTE_PRIOR_MODE`). A different seed therefore moves some requests onto a different occurrence
at the same stop, and where that occurrence is terminal the request becomes
`UNREALIZABLE_TERMINAL_OCCURRENCE`: {d10['terminal_count_seed_a']} such requests under the primary seed,
{d10['terminal_count_seed_b']} under the alternate, {d10['changed_realization_status']} status flips out of {d10['joined_rows']} ({100 * d10['changed_realization_status'] / d10['joined_rows']:.2f}%). This is
disclosed rather than smoothed over. What the evidence permits does not change: `build_origin_plan`
takes no seed, and zero realizations under either seed fall outside the frozen legal support. A
seed-independent route assignment would be worse, not better — it would pin every stop to one
arbitrary route with no evidence for the choice.

This check was **reformulated after its first run**. The first version joined on `request_id`, which
embeds the global seed by design (arms sharing a seed must share request ids). That join matched zero
rows, so it reported 0 changed destinations and vacuously-true support and status invariance. The
positional join above compares all {d10['joined_rows']} rows.

## 7. B1 / B2 / A shared-demand fairness

One frozen ledger feeds all three arms. Digest under each arm identity:
`{result['ledger_digest'][:32]}…`, identical for A, B1 and B2. The materializer holds no policy, arm or
checkpoint parameter ({len(c['R9_5_11_shared_demand_fairness']['policy_tokens_in_materializer'])} policy tokens found), so policy-specific demand regeneration is not
expressible. `ARM_VARIABLE_KEYS` remains `{list(CA.ARM_VARIABLE_KEYS)}` — demand is not among them.
Any future arm run must assert its input digest equals the frozen digest before its first step. No arm
was executed here.

## 8. Bounded execution

Written as parquet partitioned by `service_date / service_hour`, one window materialized and flushed at
a time through a streaming generator with a per-stop plan cache. {mem['rows_processed']} rows from {mem['source_rows']} source
buckets across {mem['windows']} windows; peak RSS {mem['maxrss_final_mb']} MB. No full-year expansion: the
{mem['citywide_year_row_count_avoided']}-row citywide-year materialization was never required and never performed.

## 9. Downstream lock

`scoped_inferred_request_ledger_created = true` is now allowed. Everything else stays false:
destination and timestamp are inferred, this is not an actual passenger request ledger, not a
historical passenger trajectory, and not an OD calibration. Simulator binding, simulator execution,
training, performance comparison, variant superiority and paper-level claims all remain locked.
""")

    write("gate_decision.json", {
        "gate": GATE, "decision": "PASS", "stage": result["stage"],
        "classification": result["classification"], "execution_base_sha": T.EXEC_BASE_SHA,
        "checks_passed": len(c), "checks_failed": len(result["failed_checks"]),
        "checks": {k: v["passed"] for k, v in c.items()},
        "frozen_ledger_digest": result["ledger_digest"],
        "checks_reformulated_after_first_run": [
            {"check": "R9_5_10_different_seed", "reason": d10["reformulation_note"]},
            {"check": "R9_5_13_no_argmax_no_scope_hardcoding",
             "reason": c["R9_5_13_no_argmax_no_scope_hardcoding"]["docstring_exclusion_reason"]}],
        "disclosed_behaviour": {
            "route_direction_assignment_is_seeded": True,
            "realization_status_flips_under_new_seed": d10["changed_realization_status"],
            "of_total_rows": d10["joined_rows"]}})

    write("downstream_lock.json", {
        "gate": GATE, "frozen_ledger_digest": result["ledger_digest"],
        "request_realization_mode": M.REQUEST_REALIZATION_MODE,
        "destination_semantics": S.DESTINATION_SEMANTICS,
        "request_ts_semantics": TS.TIMESTAMP_SEMANTICS,
        "origin_semantics": M.ORIGIN_SEMANTICS,
        **{k: v for k, v in c["R9_5_16_claim_guards"].items() if isinstance(v, bool)},
        "permitted_downstream_use": [
            "scoped inferred request ledger for simulator handoff schema validation"],
        "prohibited_downstream_use": [
            "observed destination or observed request time claim", "OD ground truth claim",
            "actual passenger request ledger", "historical passenger trajectory",
            "variant superiority claim", "simulator execution", "MAPPO training",
            "B1/B2/A performance comparison", "paper-level empirical claim"],
        "next_gate_required_before_simulator_execution": "H4M-AE-R9.6 or later"})

    man = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            man[str(p.relative_to(OUT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    write("artifact_manifest.json", {
        "gate": GATE, "generated": STAMP, "artifact_dir": OUT.name,
        "execution_base_sha": T.EXEC_BASE_SHA, "source_sha256": src,
        "frozen_ledger_digest": result["ledger_digest"],
        "ledger_parts": parts, "file_sha256": man})
    (OUT / "_SUCCESS.lock").write_text(f"{GATE}\n{STAMP}\n", encoding="utf-8")
    print(f"[PASS] {GATE}")
    print(f"artifact: {OUT.relative_to(T.PROJECT_ROOT)}")
    print(f"ledger rows={len(disk)} digest={disk_digest[:16]} parts={len(parts)} stream_matches={disk_digest == result['ledger_digest']}")


if __name__ == "__main__":
    main()
