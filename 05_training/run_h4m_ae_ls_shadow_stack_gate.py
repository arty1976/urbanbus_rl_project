#!/usr/bin/env python3
"""H4M-AE-R9.8-LS0..LS3 gate runner.

Writes LS0 and LS1 artifacts.  If LS2 blocks, no LS2-result or LS3 artifact is
written -- only the blocker determination, as the execution model requires.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_search_contract as LS  # noqa: E402
import test_h4m_ae_ls_shadow_stack as T  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = T.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_local_search_shadow_stack_{STAMP}"
GATE_PASS = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_LOCAL_SEARCH_ZERO_LOSS_MAPPO"
             "_SHADOW_STACK_READY_FOR_CAUSAL_ABLATION_REVIEW")
SOURCES = ("local_search_contract.py", "test_h4m_ae_ls_shadow_stack.py",
           "run_h4m_ae_ls_shadow_stack_gate.py")


def write(name: str, payload) -> None:
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        p.write_text(payload, encoding="utf-8")
    else:
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
                     encoding="utf-8")


def main() -> None:
    result = T.run_validations()
    sets = result.pop("_sets", None)
    if result["failed_checks"]:
        raise SystemExit(f"UNEXPECTED FAILURE: {result['failed_checks']}")
    OUT.mkdir(parents=True, exist_ok=False)
    c = result["checks"]
    blocker = result["blocker"]
    src = {n: hashlib.sha256((T.TRAINING_ROOT / n).read_bytes()).hexdigest() for n in SOURCES}

    write("ls0_local_search_contract.json", {
        "module": "05_training/local_search_contract.py", "sha256": src["local_search_contract.py"],
        "role_contract": LS.ROLE_CONTRACT,
        "candidate_schema": sorted(LS.Candidate.__dataclass_fields__),
        "search_config_defaults": LS.SearchConfig().payload(),
        "route_direction_semantics": LS.ROUTE_DIRECTION_SEMANTICS,
        "not_available_marker": LS.NOT_AVAILABLE,
        "verified": c["LS0_02_interface_contract"],
        "frozen_upstream": c["LS0_01_frozen_upstream"]})

    if sets:
        payloads = [s.payload() for s in sorted(sets, key=lambda s: s.historical_request_key)]
        write("ls1_shadow_candidates.json", {
            "note": "candidates that would have been available; no simulator action was taken",
            "sample_rule": T.SAMPLE_RULE,
            "candidate_set_digest": result["candidate_set_digest"],
            "shadow_input_digest": result["shadow_input_digest"],
            "candidate_sets": payloads})

    write("ls1_validation_report.json", {
        "shadow_candidate_validation": c["LS1_03_shadow_candidate_validation"],
        "determinism": c["LS1_04_determinism"],
        "frozen_demand_integrity": c["LS1_05_frozen_demand_integrity"]})

    write("ls2_blocker_determination.json", {
        **c["LS2_06_zero_loss_feasibility"],
        "stage_status": result["stage_status"],
        "ls3_artifacts_written": False,
        "reason_ls3_not_attempted": "the execution model stops at the first blocked stage"})

    write("safety_and_authorization.json", {
        **c["SAFETY_99_shadow_and_authorization"],
        "static_audit": {
            "local_search_simulator_mutation_calls": c["LS0_02_interface_contract"]["simulator_mutation_calls"],
            "local_search_simulator_imports": c["LS0_02_interface_contract"]["simulator_imports"],
            "builtin_hash_calls": c["LS0_02_interface_contract"]["builtin_hash_calls"],
            "unseeded_rng_calls": c["LS0_02_interface_contract"]["unseeded_rng_calls"]}})

    write("provenance_manifest.json", {
        "stage": result["stage"], "classification": result["classification"],
        "stopped_at": result["stopped_at"], "stage_status": result["stage_status"],
        "r9_7_source_sha": T.R97_SOURCE_SHA, "r9_8_source_sha": T.R98_SOURCE_SHA,
        "r9_7_ledger_digest": T.R97_LEDGER_DIGEST,
        "r9_7_loaded_demand_digest": T.R97_LOADED_DIGEST,
        "shadow_input_digest": result["shadow_input_digest"],
        "candidate_set_digest": result.get("candidate_set_digest"),
        "source_sha256": src,
        "frozen_upstream": c["LS0_01_frozen_upstream"],
        "authorization": result["authorization"]})

    write("self_test_report.json", result)

    a = c["LS1_03_shadow_candidate_validation"]
    d = c["LS1_04_determinism"]
    fi = c["LS1_05_frozen_demand_integrity"]
    sa = c["SAFETY_99_shadow_and_authorization"]
    findings = "\n".join(
        f"- **{f['id']}** {'(blocking)' if f['blocking'] else '(not blocking)'} — {f['fact']}"
        for f in blocker["findings"])
    why = "\n".join(f"- {w}" for w in blocker["why_not_repaired_automatically"])
    decisions = "\n".join(f"{i}. {q}" for i, q in enumerate(blocker["research_decisions_required"], 1))
    settled = "\n".join(f"- {s}" for s in blocker["not_blocking_and_already_settled"])

    write("final_report.md", f"""# H4M-AE-R9.8-LS0..LS3 — Local Search / Zero-Loss / MAPPO Shadow Stack

- Stage status: **LS0 {result['stage_status']['LS0']} → LS1 {result['stage_status']['LS1']} → LS2 {result['stage_status']['LS2']} → LS3 {result['stage_status']['LS3']}**
- Classification: `{result['classification']}`
- Stopped at: **{result['stopped_at']}**
- Generated: {STAMP}

## 1. Gate status

LS0 and LS1 pass. **LS2 is blocked** and LS3 was therefore not attempted, in line with the execution
model: the first blocked stage stops the run and no later artifact is written.

## 2. Where it stops, and why

`{blocker['code']}` — {blocker['title']}

{findings}

The epsilon question resolved cleanly and needs no decision: the frozen adapter refuses to construct
with any value other than 0.0, so Zero-Loss stays exactly as frozen. What has no answer in the
repository is the *state* Zero-Loss measures against. The rule is about existing onboard passengers
losing time, and the frozen R9.7 layer is a demand ledger with no vehicle, no onboard passenger and no
planned path. Every `ServiceObligationStateMachine` in the repository is built by hand in a test as a
toy fixture on stops `S0..S4` with passengers `P0..Pn` on vehicle `V0` — nothing bound to R9.7 demand
or to a real Daegu stop id.

### Why this was not repaired automatically

{why}

### What needs a research decision

{decisions}

### Already settled, not blocking

{settled}

## 3. LS0 — interface contract

Local Search is frozen as a candidate generator. It may rank; ranking is explicitly not action
authority (`ranking_is_final_action_authority = {LS.ROLE_CONTRACT['ranking_is_final_action_authority']}`), and final authority stays with
MAPPO after Zero-Loss. Static audit of the module: {len(c['LS0_02_interface_contract']['simulator_mutation_calls'])} simulator mutation calls,
{len(c['LS0_02_interface_contract']['simulator_imports'])} simulator imports, {len(c['LS0_02_interface_contract']['builtin_hash_calls'])} builtin `hash()` calls, {len(c['LS0_02_interface_contract']['unseeded_rng_calls'])} unseeded RNG calls. Identity is sha256 over
the path; ordering is canonical on cost, hop count and id. Every bound is configuration — no district,
agent count, route or horizon is baked in.

`route_direction_semantics = {LS.ROUTE_DIRECTION_SEMANTICS}`. A request's
inferred route travels as provenance; paths are searched over the authoritative graph, so a candidate
may leave the historical route entirely.

## 4. LS1 — shadow candidate validation

Sample declared before results: the first {a['requests_sampled']} realizable R9.7 rows by ascending
`historical_request_key`, 0 TEST6 rows.

| property | value |
|---|---|
| requests sampled | {a['requests_sampled']} |
| candidates generated | {a['candidates_generated']} |
| requests with ≥1 candidate | {a['requests_with_candidates']} |
| zero-candidate requests | {a['zero_candidate_requests']} ({a['zero_candidate_rate']:.3f}) |
| candidates per request (min/mean/max) | {a['candidates_per_request']['min']} / {a['candidates_per_request']['mean']} / {a['candidates_per_request']['max']} |
| **duplicate candidate ids** | **{a['duplicate_candidate_ids']}** |
| **illegal graph edges** | **{a['illegal_graph_edges']}** |
| disconnected or revisiting paths | {a['disconnected_or_revisiting_paths']} |
| pickup/dropoff order violations | {a['pickup_dropoff_order_violations']} |
| origin / destination mutations | {a['origin_mutations']} / {a['destination_mutations']} |
| request_ts / passenger identity mutations | {a['request_ts_mutations']} / {a['passenger_identity_mutations']} |
| generation time | {a['generation_seconds']}s |

Zero-candidate reasons: {json.dumps(a['zero_candidate_reasons'])}. The 35% zero-candidate rate is a
property of the authoritative graph, not a tuning failure: it holds 6,607 directed edges over 4,116
nodes, roughly 1.6 out-edges per node, so many origin-destination pairs have no path inside the
declared depth and radius. The rate is reported as measured rather than tuned away, and 8 requests
name a stop absent from the graph entirely.

Insertion positions are `{a['insertion_positions']}` throughout, which is the same missing
vehicle state that blocks LS2.

**Determinism.** Rebuild identical: {d['rebuild_identical']}. Shuffled request order identical:
{d['shuffled_input_identical']}. Shuffled graph edge rows identical: {d['shuffled_graph_rows_identical']}.
Candidate set digest `{d['digest'][:32]}…`.

## 5. Frozen integrity and fairness

R9.7 is untouched: {fi['identities']} identities, {fi['realized']} realized, {fi['unrealizable']} unrealizable,
matching the frozen counts ({fi['matches_frozen_counts']}). No demand, OD or timestamp was regenerated.
One shadow input digest `{fi['shadow_input_digest'][:32]}…` covers every layer, so the only thing that
would differ across research layers is candidate processing.

## 6. Shadow guarantee

| quantity | count |
|---|---|
| simulator step / reset | {sa['simulator_step']} / {sa['simulator_reset']} |
| clock advancement / vehicle movement | {sa['clock_advancement']} / {sa['vehicle_movement']} |
| boarding or alighting | {sa['boarding_or_alighting']} |
| reward settlement / KPI comparison | {sa['reward_settlement']} / {sa['kpi_performance_comparison']} |
| optimizer step / checkpoint write | {sa['optimizer_step']} / {sa['checkpoint_write']} |
| simulator state mutation | {sa['simulator_state_mutation']} |
| execution capability ever granted | {sa['execution_capability_ever_granted']} |

Peak RSS {result['maxrss_final_bytes'] / 1e6:.0f} MB, runtime {result['runtime_seconds']}s.

## 7. Authorization

Unchanged and unweakened: `simulator_binding_allowed = {result['authorization']['simulator_binding_allowed']}`,
`simulator_execution_allowed = {result['authorization']['simulator_execution_allowed']}`,
`training_allowed = {result['authorization']['training_allowed']}`,
`performance_comparison_allowed = {result['authorization']['performance_comparison_allowed']}`,
`causal_performance_claim_allowed = {result['authorization']['causal_performance_claim_allowed']}`.

## 8. Next step

Not LS4. The blocker above has to be answered first, and both questions are research decisions rather
than implementation work: what authoritative source supplies vehicle and onboard state for the R9.7
scope, and whether a counterfactual advance on a discarded deep copy sits inside `simulator_execution`
or deserves a narrower capability of its own.
""")

    write("gate_decision.json", {
        "gate_if_complete": GATE_PASS,
        "decision": "BLOCKED",
        "stopped_at": result["stopped_at"],
        "stage_status": result["stage_status"],
        "classification": result["classification"],
        "blocker_code": blocker["code"],
        "blocker_title": blocker["title"],
        "blocking_findings": [f["id"] for f in blocker["findings"] if f["blocking"]],
        "research_decisions_required": blocker["research_decisions_required"],
        "checks": {k: v["passed"] for k, v in c.items()},
        "ls3_artifacts_written": False,
        "authorization": result["authorization"],
        "arm_executions": 0, "simulator_state_mutations": 0})

    man = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            man[str(p.relative_to(OUT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    write("artifact_manifest.json", {
        "generated": STAMP, "artifact_dir": OUT.name, "decision": "BLOCKED",
        "source_sha256": src, "file_sha256": man})
    (OUT / "_BLOCKED.lock").write_text(
        f"{blocker['code']}\nstopped_at={result['stopped_at']}\n{STAMP}\n", encoding="utf-8")
    print(f"[BLOCKED at {result['stopped_at']}] {blocker['code']}")
    print(f"artifact: {OUT.relative_to(T.PROJECT_ROOT)}")
    print(f"stages: {result['stage_status']}")


if __name__ == "__main__":
    main()
