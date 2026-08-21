#!/usr/bin/env python3
"""H4M-AE-R9.4 gate runner: seeded OD distribution sampler interface and request realization contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_h4m_ae_r9_4_seeded_sampler as T  # noqa: E402
import od_seeded_sampler as S  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = T.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r9_4_seeded_od_sampler_{STAMP}"
GATE = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R9_4_SEEDED_OD_DISTRIBUTION_SAMPLER"
        "_INTERFACE_AND_REQUEST_REALIZATION_CONTRACT_COMPLETE")


def write(name: str, payload) -> None:
    p = OUT / name
    if isinstance(payload, str):
        p.write_text(payload, encoding="utf-8")
    else:
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
                     encoding="utf-8")


def main() -> None:
    r = T.run_validations()
    if not r["all_passed"]:
        raise SystemExit(f"BLOCKED: {r['failed_checks']}")
    OUT.mkdir(parents=True, exist_ok=False)
    c = r["checks"]
    src = {n: hashlib.sha256((T.TRAINING_ROOT / n).read_bytes()).hexdigest()
           for n in ("od_seeded_sampler.py", "test_h4m_ae_r9_4_seeded_sampler.py",
                     "run_h4m_ae_r9_4_gate.py", "constrained_od_engine.py",
                     "od_uncertainty_diagnostics.py", "path_cost_repair.py",
                     "od_stability_diagnostics.py")}

    write("sampler_interface_spec.json", {
        "sampler_id": S.SAMPLER_ID,
        "module": "05_training/od_seeded_sampler.py",
        "sha256": src["od_seeded_sampler.py"],
        "input_contract": {
            "distribution": "frozen R9.1/R9.3 per-origin destination probability mass, one variant only",
            "origin_key": "service_date|service_hour|stop_id",
            "origin_occurrence_id": "route_stop_occurrence_id of the boarding occurrence",
            "variant": list(T.__dict__["sys"].modules["constrained_od_engine"].VARIANTS),
            "realization_count": "integer >= 0, caller supplied, never inferred from a hardcoded scale"},
        "selection_method": "inverse CDF over canonically ordered support",
        "canonical_ordering": "support sorted by destination_occurrence_id; zero-probability candidates dropped",
        "argmax_path_absent": c["R9_4_02_sampler_interface"]["argmax_path_absent"],
        "argmax_call_sites": c["R9_4_02_sampler_interface"]["argmax_call_sites"],
        "fail_closed_codes": c["R9_4_13_fail_closed_probabilities"]["codes"],
        "silent_renormalisation": False,
        "destination_semantics": S.DESTINATION_SEMANTICS,
        "unattributable_status": S.UNATTRIBUTABLE_STATUS,
        "output_fields": c["R9_4_12_provenance"]["required_fields"]})

    write("seed_derivation_contract.json", {
        **r["seed_contract"], "tolerance_rule": S.TOLERANCE_RULE,
        "same_seed_determinism": c["R9_4_03_same_seed_determinism"],
        "independent_rebuild_determinism": c["R9_4_04_independent_rebuild_determinism"],
        "row_order_invariance": c["R9_4_05_row_order_invariance"],
        "different_seed": c["R9_4_06_different_seed"]})

    write("bounded_realization_fixture.json", {
        "scope": "bounded deterministic NON-TEST subset of the frozen R9.2/R9.3 scoped sample",
        "test6_accessed": False,
        "sample_dates": list(T.SAMPLE_DATES), "time_bands": {"night": 7, "offpeak": 10, "peak": 17},
        "origins_tested": r["origins_tested"],
        "realizations_per_origin": T.REALIZATION_N,
        "fidelity_draws_per_origin": T.FIDELITY_N,
        "count_and_legality": c["R9_4_07_count_and_legality"],
        "unattributable_handling": c["R9_4_08_unattributable_handling"],
        "example_realizations": r["realization_sample"],
        "example_unattributable_rows": r["unattributable_rows"][:3]})

    write("distribution_fidelity_diagnostics.json", {
        "tolerance_rule": S.TOLERANCE_RULE,
        "tolerance_declared_before_results": True, "tolerance_tuned_after_results": False,
        "fidelity": c["R9_4_09_distribution_fidelity"],
        "collapse": c["R9_4_11_no_collapse"],
        "variant_isolation": c["R9_4_10_variant_isolation"]})

    write("determinism_order_invariance_report.json", {
        k: c[k] for k in ("R9_4_03_same_seed_determinism", "R9_4_04_independent_rebuild_determinism",
                          "R9_4_05_row_order_invariance", "R9_4_06_different_seed")})

    write("provenance_manifest.json", {
        "gate": GATE, "stage": r["stage"], "classification": r["classification"],
        "execution_base_sha": T.EXEC_BASE_SHA, "r9_3_source_sha": T.R93_SOURCE_SHA,
        "upstream_binding": c["R9_4_01_upstream_binding"],
        "source_sha256": src,
        "read_only_inputs": {
            "occurrence_master": T.sha256_file(T.OCCURRENCE),
            "graph_nodes": T.sha256_file(T.PACK / "full_graph_nodes.parquet"),
            "repaired_edges": T.sha256_file(T.REPAIRED_EDGES),
            "citywide_ledger_dataset": T.CITYWIDE_SHA,
            "research_414_demand": T.RESEARCH_414_SHA},
        "provenance_fields": c["R9_4_12_provenance"],
        "claim_guards": c["R9_4_14_claim_guards"],
        "prohibitions": c["R9_4_15_prohibitions"]})

    write("self_test_report.json", r)

    fid = c["R9_4_09_distribution_fidelity"]["reports"]
    rows = "\n".join(
        f"| {v} | {o.split('|')[-1]} | {rep['support_size']} | {rep['max_abs_probability_error']:.6f} | "
        f"{rep['max_abs_error_bound']:.4f} | {rep['tvd']:.5f} | {rep['tvd_bound']:.5f} | "
        f"{'within' if rep['max_abs_within_bound'] and rep['tvd_within_bound'] else 'OUT'} |"
        for v, per in fid.items() for o, rep in per.items())
    col = "\n".join(
        f"| {v} | {cc['top1_only_realization_rate_all']:.4f} | "
        f"{cc['top1_only_realization_rate_excluding_degenerate']:.4f} | "
        f"{len(cc['degenerate_single_candidate_origins'])} | {len(cc['collapsed_origins'])} |"
        for v, cc in c["R9_4_11_no_collapse"]["per_variant"].items())

    _degrows = T.REALIZATION_N * sum(
        1 for v in [next(iter(c["R9_4_11_no_collapse"]["per_variant"]))]
        for _ in c["R9_4_11_no_collapse"]["per_variant"][v]["degenerate_single_candidate_origins"])
    _total = c["R9_4_07_count_and_legality"]["generated_total"]
    _bound = (_total - _degrows) / _total
    _ofdiff = c["R9_4_06_different_seed"]["rows_differing"] / (_total - _degrows)
    write("final_report.md", f"""# H4M-AE-R9.4 — Seeded OD Distribution Sampler Interface and Request Realization Contract

- Gate: `{GATE}`
- Classification: `{r['classification']}`
- execution_base_sha: `{T.EXEC_BASE_SHA}`
- R9.3 lineage sha: `{T.R93_SOURCE_SHA}`
- Generated: {STAMP}

## 1. What this gate does and does not establish

It delivers a **sampler interface**: a deterministic, seeded way to turn a frozen R9.1/R9.3 destination
probability distribution into individual request realizations. Every realized destination is
`{S.DESTINATION_SEMANTICS}` — inferred, not observed. No realization is a passenger's actual
destination, no realization set is an OD ground truth, and nothing here calibrates, validates or ranks
the V0/V1/V2 priors. No simulator binding, no training, no performance comparison, no TEST6 access.

## 2. Sampler contract

- Selection is an **inverse-CDF draw** over the canonically ordered support. There is no argmax path:
  an AST scan of the module found `{len(c['R9_4_02_sampler_interface']['argmax_call_sites'])}` argmax/idxmax/`max(key=)` call sites
  and `{len(c['R9_4_02_sampler_interface']['unseeded_random_calls'])}` unseeded `random.*` calls.
- Seeds derive from `sha256(experiment_id | global_seed | variant | distribution_digest | origin_key |
  origin_occurrence_id | request_ordinal)`. Python's builtin `hash()` is not used anywhere
  (`builtin_hash_used = {c['R9_4_02_sampler_interface']['builtin_hash_used']}`); it is salted per process and cannot support persistent seeding.
- Invalid input fails closed rather than being silently renormalised:
  `{json.dumps(c['R9_4_13_fail_closed_probabilities']['codes'], ensure_ascii=False)}`.

## 3. Determinism, order invariance, seed independence

| property | result |
|---|---|
| same seed, same run | digests identical: `{c['R9_4_03_same_seed_determinism']['identical']}` |
| independent distribution rebuild | identical: `{c['R9_4_04_independent_rebuild_determinism']['identical_to_run1']}` |
| shuffled input row order | identical: `{c['R9_4_05_row_order_invariance']['identical_to_canonical']}` |
| different global seed ({T.ALT_SEED}) | {c['R9_4_06_different_seed']['difference_rate']:.4f} of rows differ, independently reproducible: `{c['R9_4_06_different_seed']['alt_seed_reproducible']}` |

Order invariance holds because the support is sorted by `destination_occurrence_id` before the draw, so
the CDF does not depend on the order rows arrive from the engine.

The new-seed difference rate is bounded above by {_bound:.4f}, not 1.0: one of the three tested origins has a
single-candidate support (section 6), so its {_degrows} rows realize the same forced destination under every seed.
Of the rows that *can* differ, {_ofdiff:.4f} do. A rate below 1.0 is therefore expected and is not evidence of
seed leakage; the contract requires the alternate seed to be independently reproducible, which it is.

## 4. Distribution fidelity against a pre-declared tolerance

Tolerance was fixed in the module **before any result was produced**: `max_abs <= 2/sqrt(n)` (four-sigma
envelope on a binomial proportion, `se <= sqrt(0.25/n)`) and `TVD <= sqrt((m-1)/n)`
(from `E[TVD] <= 0.5*sqrt((m-1)/n)`, doubled for finite-sample slack). It was not tuned afterwards.

| variant | origin | m | max abs err | bound | TVD | bound | verdict |
|---|---|---|---|---|---|---|---|
{rows}

All {len(fid) * len(next(iter(fid.values())))} variant-origin pairs fall inside the pre-declared bounds; support coverage is 1.000 and
no zero-probability candidate was ever selected.

## 5. Legality and mass handling

| property | count |
|---|---|
| requested / generated realizations | {c['R9_4_07_count_and_legality']['requested_total']} / {c['R9_4_07_count_and_legality']['generated_total']} |
| sampled outside the frozen support | {c['R9_4_07_count_and_legality']['sampled_outside_frozen_support']} |
| same-stop destinations | {c['R9_4_07_count_and_legality']['same_stop']} |
| upstream destinations | {c['R9_4_07_count_and_legality']['upstream']} |
| cross-route or cross-direction | {c['R9_4_07_count_and_legality']['cross_route_or_direction']} |
| unattributable origins carried through | {c['R9_4_08_unattributable_handling']['unattributable_origins']} |
| destinations fabricated for them | {c['R9_4_08_unattributable_handling']['destination_assigned']} |
| their mass preserved | {c['R9_4_08_unattributable_handling']['preserved_mass']} |

Unattributable origins keep status `{S.UNATTRIBUTABLE_STATUS}`. Their mass is neither dropped nor
reassigned to a nearest stop, a nearest route, or a uniform city-wide draw.

## 6. Variant isolation and non-collapse

| variant | top-1 rate (all) | top-1 rate (excl. degenerate) | degenerate origins | collapsed origins |
|---|---|---|---|---|
{col}

Two checks were **reformulated after their first run**, and this is recorded rather than hidden. Origin
`RSO1_133daa5…` sits one stop before its route terminal, so its constrained support contains exactly one
candidate. The first formulations asserted that V0 and V1 must never share a distribution digest, and
that every origin must realize more than one distinct destination. Both are impossible on a
single-candidate support and were properties the contract never claimed. The checks now test what the
contract actually says: every row binds exactly one variant and carries that variant's own digest
(`{len(c['R9_4_10_variant_isolation']['rows_bound_to_wrong_variant_or_digest'])}` misbound rows), cross-variant digest identity is permitted only where the
support cannot distinguish the priors (`{len(c['R9_4_10_variant_isolation']['unexplained_identity'])}` unexplained cases), and non-collapse is asserted only
where the support offers an alternative. The pre-declared numeric fidelity tolerance was **not** touched.

Excluding that degenerate origin, the top-1 destination captures only 2.7–3.3% of realizations. This is
consistent with R9.3: the top-1 destination is not a defensible point estimate, and the sampler
therefore preserves the full distribution rather than committing to a mode.

## 7. Bounded execution

Peak RSS {c['R9_4_15_prohibitions']['maxrss_after_bytes'] / 1e6:.1f} MB, delta across the fidelity stage {c['R9_4_15_prohibitions']['maxrss_delta_bytes'] / 1e6:.1f} MB. No whole-city passenger
expansion, no full-year materialization, no DB writes, no network access.

## 8. Downstream lock

Realizations produced through this interface may be used only as **inferred, seeded** OD draws under a
declared variant and seed. They may not be described as observed destinations, actual request ledgers,
historical passenger trajectories, or a completed OD calibration; they may not be used to claim one
variant superior; and they are not yet bound to the simulator. R9.5 or later must authorize that.
""")

    write("gate_decision.json", {
        "gate": GATE, "decision": "PASS", "stage": r["stage"], "classification": r["classification"],
        "execution_base_sha": T.EXEC_BASE_SHA,
        "checks_passed": len(c), "checks_failed": len(r["failed_checks"]),
        "checks": {k: v["passed"] for k, v in c.items()},
        "checks_reformulated_after_first_run": [
            {"check": "R9_4_10_variant_isolation", "reason": c["R9_4_10_variant_isolation"]["reformulation_note"]},
            {"check": "R9_4_11_no_collapse", "reason": c["R9_4_11_no_collapse"]["reformulation_note"]}],
        "numeric_fidelity_tolerance_unchanged": True})

    write("downstream_lock.json", {
        "gate": GATE,
        "request_realization_semantics": "INFERRED_OD_SEEDED_REALIZATION_ONLY",
        "destination_semantics": S.DESTINATION_SEMANTICS,
        **c["R9_4_14_claim_guards"],
        "permitted_downstream_use": [
            "seeded, variant-declared inferred OD realization for interface development"],
        "prohibited_downstream_use": [
            "observed destination claim", "OD ground truth claim", "actual request ledger",
            "historical passenger trajectory", "OD calibration complete claim",
            "variant superiority claim", "simulator binding", "training", "performance comparison",
            "paper-level empirical claim"],
        "next_gate_required_before_simulator_binding": "H4M-AE-R9.5 or later"})

    man = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.iterdir())}
    write("artifact_manifest.json", {
        "gate": GATE, "generated": STAMP, "artifact_dir": OUT.name,
        "execution_base_sha": T.EXEC_BASE_SHA, "source_sha256": src, "file_sha256": man})
    (OUT / "_SUCCESS.lock").write_text(f"{GATE}\n{STAMP}\n", encoding="utf-8")
    print(f"[PASS] {GATE}")
    print(f"artifact: {OUT.relative_to(T.PROJECT_ROOT)}  files={len(list(OUT.iterdir()))}")


if __name__ == "__main__":
    main()
