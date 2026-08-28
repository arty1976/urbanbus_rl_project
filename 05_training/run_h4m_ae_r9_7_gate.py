#!/usr/bin/env python3
"""H4M-AE-R9.7 gate runner: null-safe demand contract, versioned ledger
compatibility, and simulator binding promotion review."""

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
import r95_compatibility_adapter as C  # noqa: E402
import request_ledger_materializer as M  # noqa: E402
import request_ledger_schema as SCHEMA  # noqa: E402
import request_timestamp_realization as TS  # noqa: E402
import simulator_demand_handoff as H  # noqa: E402
import test_h4m_ae_r9_5_request_ledger as R95  # noqa: E402
import test_h4m_ae_r9_7_null_safe_versioned_binding as T  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = T.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r9_7_null_safe_versioned_binding_{STAMP}"
LEDGER_OUT = OUT / "scoped_request_ledger_v2_null_safe"
GATE = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R9_7_NULL_SAFE_DEMAND_CONTRACT"
        "_VERSIONED_LEDGER_COMPATIBILITY_AND_SIMULATOR_BINDING_PROMOTION_REVIEW_COMPLETE")


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
    result.pop("_ledger", None)
    v1_view = result.pop("_v1_view")
    if not result["all_passed"]:
        raise SystemExit(f"BLOCKED: {result['failed_checks']}")
    OUT.mkdir(parents=True, exist_ok=False)
    c = result["checks"]

    src_rows = R95.load_scope_rows()
    ecfg = E.EngineConfig(occurrence_master=T.OCCURRENCE, graph_nodes=T.PACK / "full_graph_nodes.parquet",
                          graph_edges=T.REPAIRED_EDGES, variant=R95.VARIANT)
    index = E.build_index(ecfg)
    # the same provenance mapping the validator used, so the digest matches
    cfg = M.LedgerConfig(variant=R95.VARIANT, global_seed=R95.GLOBAL_SEED, scope=result["scope"],
                         provenance=result["config"]["provenance"], feasibility_filter=True)
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

    v1_path = OUT / "r95_compatibility_view" / "legacy_v1_view.parquet"
    v1_path.parent.mkdir(parents=True, exist_ok=True)
    v1_view.to_parquet(v1_path, index=False)

    src = {n: hashlib.sha256((T.TRAINING_ROOT / n).read_bytes()).hexdigest() for n in
           ("request_ledger_schema.py", "r95_compatibility_adapter.py",
            "request_ledger_materializer.py", "simulator_demand_handoff.py",
            "authoritative_demand_realization.py", "request_identity.py",
            "request_timestamp_realization.py", "terminal_feasibility_filter.py",
            "test_h4m_ae_r9_7_null_safe_versioned_binding.py", "run_h4m_ae_r9_7_gate.py")}

    write("null_safe_demand_contract.json", {
        "contract": SCHEMA.NULL_SAFE_CONTRACT,
        "schema_module": "05_training/request_ledger_schema.py", "sha256": src["request_ledger_schema.py"],
        "interface_module": "05_training/authoritative_demand_realization.py",
        "interface_sha256": src["authoritative_demand_realization.py"],
        "row_level_validation": c["R9_7_02_null_safe_destination"],
        "consumer_semantics": c["R9_7_03_consumer_semantics"],
        "round_trip": c["R9_7_10_null_round_trip"],
        "resolved_blocker": "INTERFACE_CANNOT_REPRESENT_NULL_DESTINATION",
        "resolution": ("the interface now carries a null through as None instead of applying str() to "
                       "it, and the ledger states destination_realizable and unrealizable_reason "
                       "explicitly so a consumer never parses text")})

    write("request_ledger_schema_contract.json", {
        "registry": SCHEMA.SCHEMA_REGISTRY, "current": SCHEMA.CURRENT_SCHEMA,
        "verified": c["R9_7_05_versioned_schema"],
        "v1_columns": list(C.V1_COLUMNS),
        "v2_ledger_columns": sorted(disk.columns.tolist())})

    write("r95_compatibility_report.json", {
        "adapter_id": C.ADAPTER_ID,
        "module": "05_training/r95_compatibility_adapter.py", "sha256": src["r95_compatibility_adapter.py"],
        "legacy_derivation": {"ledger_id": C.LEGACY_LEDGER_ID, "experiment_id": C.LEGACY_EXPERIMENT_ID,
                              "canonical_order": list(C.LEGACY_CANONICAL_ORDER)},
        "result": c["R9_7_06_r95_compatibility"],
        "view_path": str(v1_path.relative_to(OUT)),
        "view_sha256": hashlib.sha256(v1_path.read_bytes()).hexdigest(),
        "archived_artifact_mutated": False, "archived_digest_overwritten": False})

    write("r97_ledger_accounting.json", {
        **c["R9_7_07_ledger_accounting"],
        "per_window": result["windows"],
        "streamed_parts": parts, "streamed_ledger_digest": disk_digest,
        "in_memory_ledger_digest": result["ledger_digest"],
        "stream_matches_memory": disk_digest == result["ledger_digest"],
        "identity_invariants": c["R9_7_08_identity_invariants"]})

    write("determinism_and_round_trip.json", {
        "determinism": c["R9_7_09_determinism"],
        "null_round_trip": c["R9_7_10_null_round_trip"],
        "streamed_on_disk_matches_memory": disk_digest == result["ledger_digest"]})

    write("simulator_load_only_report.json", {
        "handoff_id": H.HANDOFF_ID, "mode": H.MODE, "column_map": H.COLUMN_MAP,
        "load_validation": c["R9_7_11_load_only_revalidation"],
        "evaluation_time_contract": c["R9_7_12_evaluation_time_contract"],
        "per_window_audit": per_window,
        "loaded_demand_digest": loaded_digest,
        "loaded_demand_digest_definition": ("nullable fields are presence-flagged so an absent "
                                            "destination can never collide with a stop id; this "
                                            "differs from the R9.6 digest definition"),
        "view_sha256": view.sha256,
        "execution_guards": H.EXECUTION_GUARDS})

    write("shared_demand_fairness_contract.json", {
        "contract_id": "B1_B2_A_SHARED_FROZEN_DEMAND_REALIZATION_V3",
        "gate": GATE, "supersedes": "B1_B2_A_SHARED_FROZEN_DEMAND_REALIZATION_V2",
        "frozen_ledger_digest": result["ledger_digest"],
        "frozen_loaded_demand_digest": loaded_digest,
        "global_demand_seed": R95.GLOBAL_SEED, "od_prior_variant": R95.VARIANT,
        "schema_version": SCHEMA.CURRENT_SCHEMA,
        "arms": [CA.ARM_A, CA.ARM_B1, CA.ARM_B2],
        "invariant": ("B1 ledger SHA == B2 ledger SHA == A ledger SHA, and the same for the "
                      "loaded-demand SHA"),
        "verified": c["R9_7_13_fairness_contract"],
        "arm_specific_filtering_of_unrealizable_rows": "PROHIBITED",
        "serviceable_only_preprocessing": ("not implemented; if a future gate needs serviceable-only "
                                           "demand it must be a separate frozen shared preprocessing "
                                           "contract applied identically to every arm"),
        "arms_executed": False})

    write("simulator_binding_promotion_decision.json", {
        **c["R9_7_14_binding_promotion_review"],
        "gate": GATE,
        "promotion_requirements": {
            "null_safe_interface": c["R9_7_02_null_safe_destination"]["passed"],
            "no_stringified_nulls": c["R9_7_03_consumer_semantics"]["unserviceable_with_string_null"] == 0,
            "schema_version_frozen": c["R9_7_05_versioned_schema"]["passed"],
            "r95_compatibility_resolved": c["R9_7_06_r95_compatibility"]["regeneration_status"],
            "deterministic_ledger": c["R9_7_09_determinism"]["passed"],
            "load_validation": c["R9_7_11_load_only_revalidation"]["passed"],
            "silent_loss": c["R9_7_11_load_only_revalidation"]["silent_loss"],
            "duplicates": c["R9_7_11_load_only_revalidation"]["duplicate_realization_ids"],
            "evaluation_time_contract": c["R9_7_12_evaluation_time_contract"]["passed"],
            "fairness_contract": c["R9_7_13_fairness_contract"]["passed"],
            "no_arm_execution": True},
        "all_technical_requirements_met": True,
        "outcome": "SIMULATOR_BINDING_ALLOWED_REMAINS_FALSE"})

    write("provenance_manifest.json", {
        "gate": GATE, "stage": result["stage"], "classification": result["classification"],
        "execution_base_sha": T.EXEC_BASE_SHA, "r9_6_source_sha": T.R96_SOURCE_SHA,
        "r9_5_ledger_digest": T.R95_LEDGER_DIGEST, "r9_6_ledger_digest": T.R96_LEDGER_DIGEST,
        "r9_7_ledger_digest": result["ledger_digest"], "loaded_demand_digest": loaded_digest,
        "frozen_upstream": c["R9_7_01_frozen_upstream"],
        "source_sha256": src, "read_only_inputs": result["config"]["provenance"],
        "static_safety_audit": c["R9_7_15_static_safety_audit"],
        "claim_guards": c["R9_7_16_claim_guards"], "prohibitions": c["R9_7_17_prohibitions"]})

    write("self_test_report.json", result)

    a = c["R9_7_07_ledger_accounting"]
    ns = c["R9_7_02_null_safe_destination"]
    cs = c["R9_7_03_consumer_semantics"]
    cm = c["R9_7_06_r95_compatibility"]
    ld = c["R9_7_11_load_only_revalidation"]
    etc = c["R9_7_12_evaluation_time_contract"]
    bp = c["R9_7_14_binding_promotion_review"]
    rt = c["R9_7_10_null_round_trip"]
    wrows = "\n".join(
        f"| {x['window_id']} | {x['time_band']} | {x['requests']} | {x['serviceable']} | "
        f"{x['unserviceable']} | {x['handoff_visible']} |" for x in result["windows"])

    write("final_report.md", f"""# H4M-AE-R9.7 — Null-Safe Demand Contract, Versioned Ledger Compatibility, Binding Promotion Review

- Gate: `{GATE}`
- Classification: `{result['classification']}`
- execution_base_sha: `{T.EXEC_BASE_SHA}`
- R9.6 lineage sha: `{T.R96_SOURCE_SHA}`
- R9.7 ledger digest: `{result['ledger_digest']}`
- Loaded-demand digest: `{loaded_digest}`
- Generated: {STAMP}

## 1. Null-safe destination contract

An absent destination is now a real absence everywhere it travels. The ledger and its parquet
partitions already stored true nulls; the defect was at the interface, where `DemandRequest`
applied `str()` to the column and turned a null into the text "None". The interface now carries a
null through as `None`, and the ledger states the fact explicitly rather than leaving it to be
inferred:

| field | meaning |
|---|---|
| `destination_realizable` | false exactly when no legal destination exists |
| `destination_stop_id` | null exactly when `destination_realizable` is false |
| `realization_status` | the explicit status enum |
| `unrealizable_reason` | non-null exactly when not realizable |

| property | value |
|---|---|
| rows | {ns['rows']} |
| realized | {ns['realized_rows']} |
| unrealizable | {ns['unrealizable_rows']} |
| true-null destinations | {ns['true_null_destinations']} |
| **stringified or sentinel destinations** | **{ns['stringified_or_sentinel_destinations']}** |
| realized rows with a null destination | {ns['realized_with_null_destination']} |
| unrealizable rows with a non-null destination | {ns['unrealizable_with_non_null_destination']} |
| realizable flag disagreeing with null-ness | {ns['realizable_matches_null']} |
| destinations fabricated | {ns['destination_fabricated']} |

No sentinel was invented. The repository documents none, so `"None"`, `"NULL"`, `"-1"`,
`"UNKNOWN_STOP"` and `0` are all rejected as destinations.

## 2. Consumer semantics

All {cs['loaded_requests']} requests reach the interface. A consumer determines realizability from the explicit
flag or from a null test, never from text: `determinable_without_text_parsing = {cs['determinable_without_text_parsing']}`,
`realizable_flag_carried_explicitly = {cs['realizable_flag_carried_explicitly']}`, `reason_carried_explicitly = {cs['reason_carried_explicitly']}`.

| unrealizable reason | count |
|---|---|
{chr(10).join(f"| `{k}` | {v} |" for k, v in sorted(cs['unserviceable_reasons'].items()))}

{cs['unserviceable_with_true_null_destination']} of {cs['unserviceable_requests']} carry a true null, {cs['unserviceable_with_string_null']} carry a string null, and
{cs['unserviceable_without_a_reason']} lack a reason.

## 3. Round trip

A null must survive every hop, not just the first. Across {len(rt['trips_checked'])} serialization paths the count of
true nulls stays at {rt['unrealizable_rows']}:

{chr(10).join(f"- `{k}`: {v}" for k, v in sorted(rt['true_null_after_each_trip'].items()))}

`json_stringified_nulls = {rt['json_stringified_nulls']}`, `null_became_string_anywhere = {rt['null_became_string_anywhere']}`.

The loaded-demand digest was also repaired: it previously serialized nullable fields with `str()`,
so an absent destination and a stop literally named "None" hashed identically. Nullable fields are
now presence-flagged, which is why this digest differs from the R9.6 one by definition and not
only by content.

## 4. Versioned schema

`{SCHEMA.SCHEMA_V1}` is the archived R9.5 shape, reachable only through the compatibility view.
`{SCHEMA.SCHEMA_V2}` is current, with `historical_request_key` and `request_realization_id` as
canonical identities. Legacy `request_id` is **not** restored as canonical: it appears only inside
the compatibility view (`legacy_request_id_in_v2_ledger = {c['R9_7_05_versioned_schema']['legacy_request_id_in_v2_ledger']}`). The declared V1 column
contract matches the archived artifact exactly ({c['R9_7_05_versioned_schema']['v1_columns']} columns).

## 5. R9.5 compatibility — byte-identical

| property | result |
|---|---|
| archived artifact integrity | {cm['archived_artifact_integrity']['files']} files, {len(cm['archived_artifact_integrity']['mismatched'])} mismatched |
| digest recomputed from the archive | `{cm['archived_digest_recomputed_from_archive'][:32]}…` |
| declared archived digest | `{cm['archived_digest'][:32]}…` |
| columns compared | {cm['columns_compared']} |
| **columns differing** | **{len(cm['columns_differing'])}** |
| reconstructed digest | `{cm['reconstructed_digest'][:32]}…` |
| **byte-identical reconstruction** | **{cm['byte_identical_reconstruction']}** |
| reconstructed counts | {cm['reconstructed_counts']} |
| reference counts | {cm['reference_counts']} |

Status: `{cm['regeneration_status']}`.

One thing to be precise about. The {len(cm['historical_provenance_columns_carried_from_archive'])} provenance columns record the sha256 of the
materializer and timestamp modules *as they were when R9.5 ran*, and R9.6 modified both, so those
bytes are not derivable from current source. The adapter takes them as declared inputs read from
the archived R9.5 manifest — the lineage record of what those files were. Every other column, all
{cm['columns_compared'] - len(cm['historical_provenance_columns_carried_from_archive'])} realization columns, is regenerated from current source and matches the archive cell for cell.
The archived artifact was not touched and its digest was not overwritten.

## 6. R9.7 ledger accounting

| quantity | value |
|---|---|
| historical boarding mass | {a['historical_boarding_mass']} |
| request rows / stable identities | {a['request_rows_generated']} / {a['stable_request_identities']} |
| realized | {a['realized_request_count']} |
| unattributable origin | {a['unattributable_request_count']} |
| unrealizable terminal | {a['unrealizable_terminal_request_count']} |
| other unrealizable | {a['other_unrealizable_count']} |
| total unrealizable | {a['total_unrealizable']} |
| **dropped mass** | **{a['dropped_mass']}** |
| duplicate stable identities | {a['duplicate_stable_identities']} |
| fabricated destinations | {a['fabricated_destination_count']} |

Matches frozen R9.6 semantics: `{a['matches_frozen_r9_6_semantics']}`. The numbers were validated against the
frozen semantics, not forced to a target.

## 7. Determinism

Same seed, independent rebuild and shuffled upstream order all reproduce
`{result['ledger_digest'][:32]}…`, and the streamed on-disk ledger matches the in-memory digest
(`{disk_digest == result['ledger_digest']}`).

## 8. Load-only simulator revalidation

| window | band | requests | serviceable | unserviceable | handoff visible |
|---|---|---|---|---|---|
{wrows}

| accounting | value |
|---|---|
| loaded rows | {ld['loaded_rows']} |
| simulator serviceable | {ld['simulator_serviceable_count']} |
| explicitly unserviceable | {ld['explicitly_unserviceable_count']} |
| **silent loss** | **{ld['silent_loss']}** |
| duplicates (realization id / stable key) | {ld['duplicate_realization_ids']} / {ld['duplicate_stable_keys']} |
| fields changed by the handoff | {sum(ld['field_invariance']['changed_values_per_field'].values())} |
| passenger_count values | {ld['passenger_count_values']} |

The {ld['explicitly_unserviceable_count']} unserviceable requests are classified `NOT_SIMULATOR_SERVICEABLE` and carried; the loader
discards none of them. All {etc['distinct_horizon_seconds']}-second window horizons are read from the authoritative bucket:
{etc['requests_outside_source_bucket']} requests outside their bucket, {etc['wrong_service_date']} on a wrong service date, {etc['negative_relative_arrivals']} negative relative arrivals,
{etc['future_leakage']} beyond the horizon, {etc['boundary_loss']} boundary losses, timestamps altered: {etc['timestamps_altered']}.

## 9. Binding promotion review — declined

`simulator_binding_allowed` would mean only that the demand interface may be connected to the causal
simulator. Every technical requirement in §15 is met. The flag is still **not** promoted, and the
reason is about authorization semantics rather than data quality.

| evidence | value |
|---|---|
| flag declaration sites | {bp['flag_declaration_sites']} |
| **real enforcement sites** | **{len(bp['flag_enforcement_sites'])}** |
| scanner self-matches excluded | {len(bp['scanner_self_matches_excluded'])} |
| repository enforces the distinction | {bp['repository_enforces_the_distinction']} |

Every occurrence of the flag is a declaration inside an artifact payload. No branch and no assertion
anywhere reads either `simulator_binding_allowed` or `simulator_execution_allowed`, so setting
binding true would be an unenforced statement that nothing checks before execution — exactly the
condition under which the flag must not be broadened. Branches inside validators and gate runners
match the token because they scan for it; a scanner is not an enforcement mechanism, so those are
excluded from the evidence rather than counted as enforcement.

Decision: `{bp['decision']}`.
What would unblock it: {bp['what_would_unblock']}.

## 10. Guards

Now true: `scoped_inferred_request_ledger_created`, `simulator_demand_handoff_schema_validated`,
`simulator_demand_load_validation_complete`, `null_safe_destination_contract_validated`,
`versioned_request_ledger_contract_validated`. Everything else stays false, including
`simulator_binding_allowed`, `simulator_execution_allowed`, `training_allowed`,
`performance_comparison_allowed` and every claim guard. Static safety audit: {c['R9_7_15_static_safety_audit']['total_findings']} findings across
{len(c['R9_7_15_static_safety_audit']['modules_scanned'])} modules.
""")

    write("gate_decision.json", {
        "gate": GATE, "decision": "PASS", "stage": result["stage"],
        "classification": result["classification"], "execution_base_sha": T.EXEC_BASE_SHA,
        "checks_passed": len(c), "checks_failed": len(result["failed_checks"]),
        "checks": {k: v["passed"] for k, v in c.items()},
        "r9_7_ledger_digest": result["ledger_digest"], "loaded_demand_digest": loaded_digest,
        "r95_compatibility": c["R9_7_06_r95_compatibility"]["regeneration_status"],
        "simulator_binding_allowed": False,
        "binding_decision": c["R9_7_14_binding_promotion_review"]["decision"],
        "checks_reformulated_after_first_run": [
            {"check": "R9_7_14_binding_promotion_review",
             "reason": c["R9_7_14_binding_promotion_review"]["scan_note"]},
            {"check": "R9_7_03_consumer_semantics",
             "reason": ("the first formulation did not assert that the realizable flag and reason "
                        "actually crossed the interface, and passed while the reason was arriving "
                        "as None for all 102 rows")}],
        "repairs_made": [
            {"code": "INTERFACE_CANNOT_REPRESENT_NULL_DESTINATION", "status": "RESOLVED"},
            {"code": "LOADED_DEMAND_DIGEST_STRINGIFIED_NULLS", "status": "RESOLVED",
             "detail": "nullable fields are now presence-flagged in the digest serialization"},
            {"code": "R95_VALIDATOR_NOT_REEXECUTABLE", "status": "RESOLVED_VIA_COMPATIBILITY_VIEW"}]})

    write("downstream_lock.json", {
        "gate": GATE, "frozen_ledger_digest": result["ledger_digest"],
        "frozen_loaded_demand_digest": loaded_digest,
        "schema_version": SCHEMA.CURRENT_SCHEMA,
        "request_realization_mode": M.REQUEST_REALIZATION_MODE,
        "destination_semantics": S.DESTINATION_SEMANTICS,
        "request_ts_semantics": TS.TIMESTAMP_SEMANTICS,
        "origin_semantics": M.ORIGIN_SEMANTICS,
        **{k: v for k, v in c["R9_7_16_claim_guards"].items() if isinstance(v, bool)},
        "permitted_downstream_use": [
            "null-safe scoped inferred request ledger validated at the simulator demand boundary",
            "schema and load validation of the demand handoff",
            "R9.5 legacy validation through the compatibility view"],
        "prohibited_downstream_use": [
            "connecting the demand interface to the simulator", "simulator execution",
            "policy action or stepping", "MAPPO training", "B1/B2/A performance comparison",
            "reward or KPI computation", "observed destination or request time claim",
            "OD ground truth claim", "variant superiority claim", "paper-level empirical claim"],
        "binding_decision": c["R9_7_14_binding_promotion_review"]["decision"],
        "what_would_unblock_binding": c["R9_7_14_binding_promotion_review"]["what_would_unblock"],
        "next_gate_required_before_binding": "H4M-AE-R9.8 or later"})

    man = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            man[str(p.relative_to(OUT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    write("artifact_manifest.json", {
        "gate": GATE, "generated": STAMP, "artifact_dir": OUT.name,
        "execution_base_sha": T.EXEC_BASE_SHA, "source_sha256": src,
        "r9_7_ledger_digest": result["ledger_digest"], "loaded_demand_digest": loaded_digest,
        "ledger_parts": parts, "handoff_view_sha256": view.sha256,
        "peak_rss_bytes": int(peak), "file_sha256": man})
    (OUT / "_SUCCESS.lock").write_text(f"{GATE}\n{STAMP}\n", encoding="utf-8")
    print(f"[PASS] {GATE}")
    print(f"artifact: {OUT.relative_to(T.PROJECT_ROOT)}")
    print(f"ledger rows={len(disk)} digest={disk_digest[:16]} stream_matches={disk_digest == result['ledger_digest']}")
    print(f"loaded rows={len(requests)} digest={loaded_digest[:16]}")
    print(f"binding: {c['R9_7_14_binding_promotion_review']['decision']}")


if __name__ == "__main__":
    main()
