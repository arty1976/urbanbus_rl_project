#!/usr/bin/env python3
"""H4M-AE-R9.6 gate runner: terminal-occurrence feasibility repair, stable request
identity, and simulator demand handoff validation."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import authoritative_demand_realization as ADR  # noqa: E402
import causal_arm_contracts as CA  # noqa: E402
import constrained_od_engine as E  # noqa: E402
import od_seeded_sampler as S  # noqa: E402
import request_identity as RI  # noqa: E402
import request_ledger_materializer as M  # noqa: E402
import request_timestamp_realization as TS  # noqa: E402
import simulator_demand_handoff as H  # noqa: E402
import terminal_feasibility_filter as F  # noqa: E402
import test_h4m_ae_r9_5_request_ledger as R95  # noqa: E402
import test_h4m_ae_r9_6_feasibility_identity_handoff as T  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = T.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r9_6_feasibility_identity_handoff_{STAMP}"
LEDGER_OUT = OUT / "scoped_request_ledger_v2"
GATE = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R9_6_TERMINAL_OCCURRENCE_FEASIBILITY_REPAIR"
        "_STABLE_REQUEST_IDENTITY_AND_SIMULATOR_DEMAND_HANDOFF_VALIDATION_COMPLETE")


def write(name: str, payload) -> None:
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        p.write_text(payload, encoding="utf-8")
    else:
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
                     encoding="utf-8")


def stream_ledger(cfg, index, ecfg, src_rows):
    """One window materialized and flushed at a time; the scope is never held whole."""
    LEDGER_OUT.mkdir(parents=True, exist_ok=True)
    by_window: dict = {}
    for row in src_rows:
        by_window.setdefault((row["service_date"], row["service_hour"]), []).append(row)
    plan_cache: dict = {}
    parts, peak = [], 0
    for (date, hour) in sorted(by_window):
        frame = M.canonical_sort(pd.DataFrame(
            list(M.materialize(index, ecfg, cfg, by_window[(date, hour)], plan_cache=plan_cache))))
        part = LEDGER_OUT / f"service_date={date}" / f"service_hour={hour}" / "part-0.parquet"
        part.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(part, index=False)
        parts.append({"service_date": date, "service_hour": int(hour), "rows": int(len(frame)),
                      "path": str(part.relative_to(OUT)), "bytes": part.stat().st_size,
                      "sha256": hashlib.sha256(part.read_bytes()).hexdigest()})
        peak = max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        del frame
    return parts, peak


def main() -> None:
    result = T.run_validations()
    ledger = result.pop("_ledger")
    before = result.pop("_before")
    if not result["all_passed"]:
        raise SystemExit(f"BLOCKED: {result['failed_checks']}")
    OUT.mkdir(parents=True, exist_ok=False)
    c = result["checks"]

    src_rows = R95.load_scope_rows()
    ecfg = E.EngineConfig(occurrence_master=T.OCCURRENCE, graph_nodes=T.PACK / "full_graph_nodes.parquet",
                          graph_edges=T.REPAIRED_EDGES, variant=R95.VARIANT)
    index = E.build_index(ecfg)
    cfg = M.LedgerConfig(variant=R95.VARIANT, global_seed=R95.GLOBAL_SEED, scope=result["scope"],
                         provenance=c["R9_6_16_provenance"]["sources"], feasibility_filter=True)
    parts, peak = stream_ledger(cfg, index, ecfg, src_rows)
    disk = M.canonical_sort(pd.concat([pd.read_parquet(OUT / p["path"]) for p in parts],
                                      ignore_index=True))
    disk_digest = M.ledger_digest(disk)

    view = H.project(disk, OUT / "simulator_handoff_view" / "handoff_view.parquet")
    realization = H.load_only(view.path)
    per_window, requests = {}, []
    for window in sorted(disk["window_id"].astype(str).unique()):
        selected = ADR.select_population(realization, H.window_contract(disk, window),
                                         reporting_window_ids=[window])
        per_window[window] = selected["audit"]
        requests += selected["requests"]
    loaded_digest = H.loaded_demand_digest(requests)

    src = {n: hashlib.sha256((T.TRAINING_ROOT / n).read_bytes()).hexdigest() for n in
           ("terminal_feasibility_filter.py", "request_identity.py", "simulator_demand_handoff.py",
            "request_ledger_materializer.py", "request_timestamp_realization.py",
            "test_h4m_ae_r9_6_feasibility_identity_handoff.py", "run_h4m_ae_r9_6_gate.py",
            "od_seeded_sampler.py", "constrained_od_engine.py")}

    write("terminal_feasibility_audit.json", {
        **result["feasibility_audit"], "verified": c["R9_6_02_feasibility_audit"],
        "filter_justification": c["R9_6_03_filter_is_feasibility_only"]})

    write("stable_request_identity_contract.json", {
        "module": "05_training/request_identity.py", "sha256": src["request_identity.py"],
        "contract": RI.IdentityContract(realization_version=cfg.realization_version).payload(),
        "verified": c["R9_6_06_stable_identity"],
        "why": ("R9.5 carried one identifier that folded the global seed into it, so the same "
                "historical boarding could not be followed across seeds and a different-seed "
                "comparison joined zero rows")})

    write("before_after_demand_accounting.json", {
        "same_windows_and_seeds": True,
        "before_filter": result["conservation_before"], "after_filter": result["conservation_after"],
        "summary": c["R9_6_04_before_after"],
        "per_window": result["windows"],
        "streamed_parts": parts, "streamed_ledger_digest": disk_digest,
        "in_memory_ledger_digest": result["ledger_digest"],
        "stream_matches_memory": disk_digest == result["ledger_digest"]})

    write("different_seed_identity_stability.json", c["R9_6_07_different_seed"])

    write("determinism_report.json", {k: c[k] for k in (
        "R9_6_08_same_seed_determinism", "R9_6_09_independent_rebuild",
        "R9_6_10_row_order_invariance")} | {
        "streamed_on_disk_matches_memory": disk_digest == result["ledger_digest"],
        "double_load_identical": c["R9_6_12_handoff_accounting"]["double_load_digest_identical"]})

    write("simulator_demand_handoff_schema.json", {
        "handoff_id": H.HANDOFF_ID, "mode": H.MODE,
        "module": "05_training/simulator_demand_handoff.py", "sha256": src["simulator_demand_handoff.py"],
        "column_map": H.COLUMN_MAP, "invariant_fields": list(H.INVARIANT_FIELDS),
        "target_interface": ADR.DEMAND_INTERFACE_ID,
        "ledger_fields": [{"field": f, "meaning": m} for f, m in M.HANDOFF_SCHEMA],
        "view_path": str((OUT / "simulator_handoff_view" / "handoff_view.parquet").relative_to(OUT)),
        "view_sha256": view.sha256,
        "verified": c["R9_6_11_handoff_schema"],
        "execution_guards": H.EXECUTION_GUARDS})

    write("handoff_load_validation.json", {
        "accounting": c["R9_6_12_handoff_accounting"],
        "evaluation_time_contract": c["R9_6_14_evaluation_time_contract"],
        "per_window_audit": per_window,
        "loaded_demand_digest": loaded_digest,
        "interface_capability": c["R9_6_13_interface_capability"]})

    write("fairness_digest_contract.json", {
        "contract_id": "B1_B2_A_SHARED_FROZEN_DEMAND_REALIZATION_V2",
        "gate": GATE, "supersedes": "B1_B2_A_SHARED_FROZEN_DEMAND_REALIZATION_V1",
        "frozen_ledger_digest": result["ledger_digest"],
        "frozen_loaded_demand_digest": loaded_digest,
        "global_demand_seed": R95.GLOBAL_SEED, "od_prior_variant": R95.VARIANT,
        "feasibility_filter_version": F.FILTER_VERSION,
        "realization_version": cfg.realization_version,
        "arms": [CA.ARM_A, CA.ARM_B1, CA.ARM_B2],
        "invariant": "B1 demand digest == B2 demand digest == A demand digest",
        "verified": c["R9_6_15_fairness_contract"],
        "policy_specific_demand_regeneration": "PROHIBITED",
        "enforcement": ("a future arm run must assert both the ledger digest and the loaded-demand "
                        "digest before its first step"),
        "arms_executed": False})

    write("provenance_manifest.json", {
        "gate": GATE, "stage": result["stage"], "classification": result["classification"],
        "execution_base_sha": T.EXEC_BASE_SHA, "r9_5_source_sha": T.R95_SOURCE_SHA,
        "r9_5_ledger_digest": T.R95_LEDGER_DIGEST,
        "frozen_upstream": c["R9_6_01_frozen_upstream"],
        "source_sha256": src, "read_only_inputs": c["R9_6_16_provenance"]["sources"],
        "provenance_integrity": c["R9_6_16_provenance"],
        "ledger_manifest": result["manifest"],
        "claim_guards": c["R9_6_17_claim_guards"], "prohibitions": c["R9_6_18_prohibitions"]})

    write("self_test_report.json", result)

    b, a = c["R9_6_04_before_after"]["before"], c["R9_6_04_before_after"]["after"]
    full_after = result["conservation_after"]
    residual = c["R9_6_04_before_after"]["residual_unrealizable_after"]
    fa = result["feasibility_audit"]
    d7 = c["R9_6_07_different_seed"]
    acc = c["R9_6_12_handoff_accounting"]
    etc = c["R9_6_14_evaluation_time_contract"]
    cap = c["R9_6_13_interface_capability"]
    wrows = "\n".join(
        f"| {x['window_id']} | {x['time_band']} | {x['requests']} | {x['realized']} | "
        f"{x['handoff_visible']} | {x['horizon_seconds']:.0f} |" for x in result["windows"])

    write("final_report.md", f"""# H4M-AE-R9.6 — Terminal-Occurrence Feasibility Repair, Stable Request Identity, Simulator Demand Handoff

- Gate: `{GATE}`
- Classification: `{result['classification']}`
- execution_base_sha: `{T.EXEC_BASE_SHA}`
- R9.5 lineage sha: `{T.R95_SOURCE_SHA}`
- R9.6 ledger digest: `{result['ledger_digest']}`
- Loaded-demand digest: `{loaded_digest}`
- Generated: {STAMP}

## 1. Terminal-occurrence feasibility

The question was whether a route-direction occurrence offering **zero legal downstream destination**
can be the latent route choice of a passenger who *historically boarded* at that stop. It cannot. The
boarding is observed history — somebody got on a bus there — and an occurrence with no downstream stop
gives that passenger nowhere to ride. Excluding it follows from the observation plus the frozen
occurrence topology. It is a feasibility constraint, not a preference: no frequency prior, no headway,
no ETA, no external source, no performance outcome.

| classification | occurrences |
|---|---|
| `LEGAL_DOWNSTREAM_SUPPORT` | {fa['occurrence_classification']['LEGAL_DOWNSTREAM_SUPPORT']} |
| `ZERO_DOWNSTREAM_TERMINAL_OCCURRENCE` | {fa['occurrence_classification']['ZERO_DOWNSTREAM_TERMINAL_OCCURRENCE']} |
| `OTHER_NO_LEGAL_DOWNSTREAM_SUPPORT` | {fa['occurrence_classification']['OTHER_NO_LEGAL_DOWNSTREAM_SUPPORT']} |
| occurrences removed | {fa['occurrences_removed']} |
| **occurrences with legal support removed** | **{fa['occurrences_with_legal_support_removed']}** |

Every removal is a genuine terminal: there were no non-terminal zero-support cases to argue about. At
stop level, {fa['stop_classification'].get('HAS_LEGAL_CANDIDATE', 0)} stops keep at least one legal candidate,
{fa['stop_classification'].get('NO_ROUTE_DIRECTION_SERVING_STOP', 0)} have no serving route-direction at all, and
{fa['stop_classification'].get('ALL_CANDIDATES_ZERO_DOWNSTREAM_SUPPORT', 0)} stop — `{fa['stops_with_every_candidate_zero_downstream'][0]['stop_id']}`, carrying
{fa['stops_with_every_candidate_zero_downstream'][0]['boarding_mass']} boarding — has every candidate terminal. That boarding is **not** fabricated onto a route:
it stays an explicit request identity with `{F.REASON_ALL_ZERO}`.

## 2. Before / after, same windows and same seeds

| quantity | before | after | delta |
|---|---|---|---|
| historical boarding mass | {b['historical_boarding_mass']} | {a['historical_boarding_mass']} | 0 |
| request rows | {b['request_rows_generated']} | {a['request_rows_generated']} | 0 |
| realized | {b['realized_request_count']} | {a['realized_request_count']} | +{a['realized_request_count'] - b['realized_request_count']} |
| unattributable origin | {b['unattributable_request_count']} | {a['unattributable_request_count']} | 0 |
| unrealizable terminal | {b['unrealizable_terminal_request_count']} | {a['unrealizable_terminal_request_count']} | {a['unrealizable_terminal_request_count'] - b['unrealizable_terminal_request_count']} |
| other unrealizable | {b['other_unrealizable_count']} | {a['other_unrealizable_count']} | +{a['other_unrealizable_count'] - b['other_unrealizable_count']} |
| **dropped mass** | **{b['dropped_mass']}** | **{a['dropped_mass']}** | 0 |

The before column reproduces the R9.5 result exactly ({b['realized_request_count']} / {b['unattributable_request_count']} / {b['unrealizable_terminal_request_count']}), which is the
regression evidence that the filter, and not some incidental change, produced the difference.

Terminal unrealizable was **not forced to zero as a target**: {residual} requests remain unrealizable
after the repair ({a['unattributable_request_count']} with no serving route-direction, {a['other_unrealizable_count']} with every candidate terminal), and
{c['R9_6_04_before_after']['destinations_fabricated_to_reach_full_realization']} destinations were fabricated to close the gap.

## 3. Stable identity

`historical_request_key` = sha256(source bucket key, request ordinal). No seed, no variant, no
realization digest, so the same historical boarding keeps one key under every seed.
`request_realization_id` = sha256(stable key, global seed, realization version) names one *draw*.
{full_after['stable_request_identities']} stable identities for {a['historical_boarding_mass']} boardings,
{full_after['duplicate_stable_identities']} duplicates. AST scan: {len(c['R9_6_06_stable_identity']['builtin_hash_call_sites'])} builtin `hash()` call sites,
{len(c['R9_6_06_stable_identity']['seed_references_in_stable_derivation'])} seed references inside the stable derivation.

## 4. Different seed — the R9.5 defect is gone

Joined on `historical_request_key` across {d7['joined_rows']} rows ({'non-vacuous' if d7['join_is_non_vacuous'] else 'VACUOUS'}), key populations identical.

| quantity | change rate |
|---|---|
| request timestamp | {d7['timestamp_change_rate']:.4f} |
| route-direction | {d7['route_direction_change_rate']:.4f} |
| destination | {d7['destination_change_rate']:.4f} |
| **realization status** | **{d7['realization_status_change_rate']:.4f}** |

Terminal status flips {d7['terminal_status_flip_count']}, unattributable status flips {d7['unattributable_status_flip_count']}. R9.5 flipped realization status on
201 rows (0.44%) because a seeded route draw could land on a terminal occurrence; with those
occurrences no longer in the candidate set, that cannot happen and the rate is exactly zero. What the
seed still moves is what it should move: time, route-direction and destination. Historical boarding
count, key population, origin evidence, source bucket and passenger_count are all invariant.

## 5. Determinism

Same seed, independent rebuild and shuffled upstream row order all reproduce
`{result['ledger_digest'][:32]}…`. The streamed on-disk ledger matches the in-memory digest
(`{disk_digest == result['ledger_digest']}`), and loading the handoff view twice yields the identical loaded-demand digest
(`{acc['double_load_digest_identical']}`).

## 6. Simulator demand handoff — load only

| window | band | requests | realized | handoff visible | horizon (s) |
|---|---|---|---|---|---|
{wrows}

| accounting | value |
|---|---|
| ledger request count | {acc['ledger_request_count']} |
| handoff-visible | {acc['handoff_visible_request_count']} |
| explicitly rejected by interface | {acc['explicitly_rejected_by_interface_count']} |
| **silent loss** | **{acc['silent_loss']}** |
| duplicate realization ids / stable keys | {acc['duplicate_realization_ids']} / {acc['duplicate_stable_keys']} |
| fields changed by the handoff | {sum(acc['field_invariance']['changed_values_per_field'].values())} |

`request_ts`, origin, destination and passenger_count pass through unchanged. Nothing was stepped:
no action selected, no vehicle moved, no boarding executed, no reward or KPI computed.

### Interface blocker, documented not repaired

`{cap['blocker']['code']}` — {cap['blocker']['requests_affected']} requests affected. `DemandRequest.destination_stop_id`
applies `str()` to the column value, so a request with no legal destination arrives as the *text* of a
missing value rather than as an absent destination. All {cap['unrealizable_requests_carried_into_handoff']} unrealizable requests are carried and
{cap['unrealizable_requests_deleted']} were deleted, but a consumer cannot tell "no destination" from a stop id without also reading
`realization_status`. This is left for the next gate rather than repaired here, because R9.6 must not
invent simulator semantics.

## 7. EvaluationTimeContract alignment

All {etc['window_count']} windows carry a horizon of {etc['distinct_horizon_seconds']} seconds read from the authoritative bucket, never
hardcoded. {etc['requests_outside_source_bucket']} requests fall outside their source bucket, {etc['requests_before_evaluation_start']} before the evaluation start,
{etc['requests_after_evaluation_end']} after the end; every window partition closes. Relative arrivals span
{etc['min_relative_arrival']}s to {etc['max_relative_arrival']}s with {etc['negative_relative_arrivals']} negative and {etc['future_leakage']} beyond the horizon. The existing causal
time contract was used as-is and not reshaped to fit the ledger.

## 8. Fairness

One frozen ledger digest and one loaded-demand digest for A, B1 and B2. The materializer holds
{len(c['R9_6_15_fairness_contract']['policy_tokens_in_materializer'])} policy/arm/checkpoint tokens, so policy-specific regeneration is not expressible, and
`demand` is not in `ARM_VARIABLE_KEYS`. No arm was executed.

## 9. Guards

`scoped_inferred_request_ledger_created`, `simulator_demand_handoff_schema_validated` and
`simulator_demand_load_validation_complete` are now true. `simulator_binding_allowed` stays **false**:
this gate validated a schema and a load, which is not authorization to execute.
""")

    write("gate_decision.json", {
        "gate": GATE, "decision": "PASS", "stage": result["stage"],
        "classification": result["classification"], "execution_base_sha": T.EXEC_BASE_SHA,
        "checks_passed": len(c), "checks_failed": len(result["failed_checks"]),
        "checks": {k: v["passed"] for k, v in c.items()},
        "r9_6_ledger_digest": result["ledger_digest"], "loaded_demand_digest": loaded_digest,
        "r9_5_ledger_digest_preserved": T.R95_LEDGER_DIGEST,
        "before_after": c["R9_6_04_before_after"],
        "checks_reformulated_after_first_run": [
            {"check": "R9_6_03_filter_is_feasibility_only",
             "reason": c["R9_6_03_filter_is_feasibility_only"]["scan_note"]},
            {"check": "R9_6_06_stable_identity",
             "reason": ("a raw text scan for a builtin hash call matched the docstring sentence "
                        "stating that builtin hash() cannot back a persistent identity; the scan "
                        "is now an AST call-site scan")}],
        "open_blocker": c["R9_6_13_interface_capability"]["blocker"]})

    write("downstream_lock.json", {
        "gate": GATE, "frozen_ledger_digest": result["ledger_digest"],
        "frozen_loaded_demand_digest": loaded_digest,
        "request_realization_mode": M.REQUEST_REALIZATION_MODE,
        "destination_semantics": S.DESTINATION_SEMANTICS,
        "request_ts_semantics": TS.TIMESTAMP_SEMANTICS,
        "origin_semantics": M.ORIGIN_SEMANTICS,
        **{k: v for k, v in c["R9_6_17_claim_guards"].items() if isinstance(v, bool)},
        "permitted_downstream_use": [
            "scoped inferred request ledger validated at the simulator demand boundary",
            "schema and load validation of the demand handoff"],
        "prohibited_downstream_use": [
            "simulator execution", "policy action or stepping", "MAPPO training",
            "B1/B2/A performance comparison", "reward or KPI computation",
            "observed destination or observed request time claim", "OD ground truth claim",
            "variant superiority claim", "paper-level empirical claim"],
        "open_interface_blocker": c["R9_6_13_interface_capability"]["blocker"],
        "next_gate_required_before_simulator_execution": "H4M-AE-R9.7 or later"})

    man = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            man[str(p.relative_to(OUT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    write("artifact_manifest.json", {
        "gate": GATE, "generated": STAMP, "artifact_dir": OUT.name,
        "execution_base_sha": T.EXEC_BASE_SHA, "source_sha256": src,
        "r9_6_ledger_digest": result["ledger_digest"], "loaded_demand_digest": loaded_digest,
        "ledger_parts": parts, "handoff_view_sha256": view.sha256,
        "peak_rss_bytes": int(peak), "file_sha256": man})
    (OUT / "_SUCCESS.lock").write_text(f"{GATE}\n{STAMP}\n", encoding="utf-8")
    print(f"[PASS] {GATE}")
    print(f"artifact: {OUT.relative_to(T.PROJECT_ROOT)}")
    print(f"ledger rows={len(disk)} digest={disk_digest[:16]} stream_matches={disk_digest == result['ledger_digest']}")
    print(f"loaded_demand rows={len(requests)} digest={loaded_digest[:16]}")


if __name__ == "__main__":
    main()
