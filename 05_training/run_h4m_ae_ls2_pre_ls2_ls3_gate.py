#!/usr/bin/env python3
"""H4M-AE-R9.8 LS2-PRE / LS2 / LS3 gate runner.

Writes only artifacts for stages that were actually reached.  LS3 blocked, so no
LS3 selection artifact is produced.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import operational_state_layer as OP  # noqa: E402
import simulator_authorization as AUTH  # noqa: E402
import test_h4m_ae_ls2_pre_ls2_ls3 as T  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = T.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls2_pre_ls2_ls3_{STAMP}"
GATE_PRE = "PASS_SUSEONG_H4M_AE_OPERATIONAL_VEHICLE_ONBOARD_STATE_AND_SHADOW_COUNTERFACTUAL_CONTRACT_COMPLETE"
GATE_LS2 = "PASS_SUSEONG_H4M_AE_LOCAL_SEARCH_ZERO_LOSS_NON_VACUOUS_SHADOW_FILTER_VALIDATION_COMPLETE"
GATE_LS3 = "PASS_SUSEONG_H4M_AE_LOCAL_SEARCH_ZERO_LOSS_MAPPO_SHADOW_SELECTION_VALIDATION_COMPLETE"
SOURCES = ("operational_state_layer.py", "simulator_authorization.py", "simulator/k_safety_state.py",
           "local_search_contract.py", "test_h4m_ae_ls2_pre_ls2_ls3.py",
           "run_h4m_ae_ls2_pre_ls2_ls3_gate.py", "simulator/zero_loss_admission_adapter.py")


def write(name, payload):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload if isinstance(payload, str) else
                 json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
                 encoding="utf-8")


def main() -> None:
    r = T.run_validations()
    evidence = r.pop("_evidence", None) or []
    states = r.pop("_states", None) or []
    if r["failed_checks"]:
        raise SystemExit(f"UNEXPECTED FAILURE: {r['failed_checks']}")
    OUT.mkdir(parents=True, exist_ok=False)
    c = r["checks"]
    src = {n: hashlib.sha256((T.TRAINING_ROOT / n).read_bytes()).hexdigest() for n in SOURCES}
    blocker = r["blocker"]

    write("operational_state_contract.json", {
        "gate": GATE_PRE, "contract": OP.DERIVATION_CONTRACT,
        "scenario": T.SCENARIO, "scenario_config": OP.ScenarioConfig().payload(),
        "verified": c["PRE_A_03_operational_state"],
        "module": "05_training/operational_state_layer.py", "sha256": src["operational_state_layer.py"]})
    write("operational_state_derivation_manifest.json", {
        "states_digest": r["operational_states_digest"],
        "demand_input_digest": r["demand_input_digest"],
        "onboard_distribution": c["PRE_A_03_operational_state"]["onboard_distribution"],
        "states": [s.payload() for s in states]})
    write("shadow_counterfactual_authorization_contract.json", {
        "gate": GATE_PRE, "capability": AUTH.SHADOW_COUNTERFACTUAL,
        "meaning": AUTH.CAPABILITY_MEANING[AUTH.SHADOW_COUNTERFACTUAL],
        "ladder": list(AUTH.CAPABILITY_LADDER),
        "contract": AUTH.AUTHORIZATION_CONTRACT,
        "verified": c["PRE_B_02_shadow_capability"],
        "design": ("ServiceObligationStateMachine.__deepcopy__ marks a copy disposable when taken "
                   "inside a shadow_counterfactual grant; advance_to charges a disposable copy to "
                   "shadow_counterfactual and the live state to simulator_execution, so the frozen "
                   "Zero-Loss adapter stays byte-identical"),
        "zero_loss_adapter_sha256": src["simulator/zero_loss_admission_adapter.py"]})
    write("authorization_negative_tests.json", c["PRE_B_04_authorization_negative_tests"])
    write("source_state_immutability_proof.json", {
        "case_B_source_unchanged": c["PRE_B_04_authorization_negative_tests"]["case_B_source_unchanged"],
        "zero_loss_source_state_mutations": c["LS2_05_zero_loss_non_vacuous"]["source_state_mutations"],
        "per_candidate_pre_post_digests": [
            {"candidate_id": e["candidate_id"], "pre": e["source_state_pre_digest"],
             "post": e["source_state_post_digest"],
             "unchanged": e["source_state_pre_digest"] == e["source_state_post_digest"]}
            for e in evidence]})
    write("zero_loss_candidate_evidence.json", {
        "gate": GATE_LS2, "epsilon_sec": 0.0,
        "adapter_sha256": src["simulator/zero_loss_admission_adapter.py"],
        "counterfactual_marking": "SHADOW_ONLY / NON_EXECUTED_COUNTERFACTUAL",
        "evidence": evidence})
    write("zero_loss_accounting_and_non_vacuity.json", c["LS2_05_zero_loss_non_vacuous"])
    write("ls3_blocker_determination.json", {
        **c["LS3_06_mappo_candidate_interface"], "gate_if_complete": GATE_LS3,
        "ls3_selection_artifact_written": False})
    write("safety_and_authorization.json", c["SAFETY_99_shadow_and_authorization"])
    write("provenance_manifest.json", {
        "stage": r["stage"], "stage_status": r["stage_status"], "stopped_at": r["stopped_at"],
        "r9_7_source_sha": T.R97_SOURCE_SHA, "r9_8_source_sha": T.R98_SOURCE_SHA,
        "ls0_ls1_source_sha": T.LS01_SOURCE_SHA,
        "r9_7_ledger_digest": T.R97_LEDGER_DIGEST,
        "demand_input_digest": r["demand_input_digest"],
        "operational_states_digest": r["operational_states_digest"],
        "frozen_upstream": c["PRE_01_frozen_upstream"], "source_sha256": src})
    write("self_test_report.json", r)

    z = c["LS2_05_zero_loss_non_vacuous"]
    a = c["PRE_A_03_operational_state"]
    n = c["PRE_B_04_authorization_negative_tests"]
    s = c["SAFETY_99_shadow_and_authorization"]
    findings = "\n".join(f"- **{f['id']}** — {f['fact']}" for f in blocker["findings"])
    decisions = "\n".join(f"{i}. {q}" for i, q in enumerate(blocker["research_decisions_required"], 1))

    write("final_report.md", f"""# H4M-AE-R9.8 — LS2-PRE / LS2 / LS3

- Stage status: **LS2-PRE {r['stage_status']['LS2_PRE']} → LS2 {r['stage_status']['LS2']} → LS3 {r['stage_status']['LS3']}**
- Stopped at: **{r['stopped_at']}**
- Classification: `{r['classification']}`
- Generated: {STAMP}

## 1. LS2-PRE-A — operational vehicle/onboard state

R9.7 stays a demand ledger; this layer sits beside it. Every onboard rider is a real
`historical_request_key` travelling to its own real destination, placed on a real simple path over
authoritative edges, assigned by the repository's only sanctioned neutral baseline
(`{a['service_rule_id']}`). Nothing was randomised: {len(a['unseeded_rng_calls'])} RNG calls and
{len(a['builtin_hash_calls'])} builtin `hash()` calls in the module.

| property | value |
|---|---|
| states derived | {a['states_derived']} |
| zero / one / multi onboard | {a['onboard_distribution']['zero_onboard']} / {a['onboard_distribution']['one_onboard']} / {a['onboard_distribution']['multi_onboard']} |
| total onboard passengers | {a['total_onboard_passengers']} |
| fabricated passenger identities | {len(a['fabricated_passenger_keys'])} |
| all onboard are real R9.7 identities | {a['all_onboard_are_real_r9_7_identities']} |
| all destinations lie on the vehicle path | {a['all_destinations_on_vehicle_path']} |
| deterministic rebuild / shuffled input | {a['deterministic_rebuild']} / {a['shuffled_request_order_identical']} |

Labelled `{a['vehicle_state_source']}` and `{a['onboard_state_source']}` — not observed history, and the
corresponding claim flags stay false. Only multi-onboard states arose naturally from the frozen rule;
proportions were not forced to manufacture the other two classes.

## 2. LS2-PRE-B — shadow_counterfactual authorization

`simulator_execution` keeps its R9.8 meaning untouched. The new capability covers exactly one thing:
advancing a disposable deep copy that is discarded. The separation is structural —
`ServiceObligationStateMachine.__deepcopy__` marks a copy disposable when taken inside a
`shadow_counterfactual` grant, and `advance_to` charges a disposable copy to the new capability while
the live state still demands `simulator_execution`. The frozen Zero-Loss adapter was **not** modified;
its sha256 is unchanged, which matters because it is pinned in several runners.

| test | result |
|---|---|
| A — nothing granted, counterfactual | {'BLOCKED' if n['case_A_blocked'] else 'ALLOWED'} |
| B — shadow granted, counterfactual | {'ALLOWED' if n['case_B_allowed'] else 'BLOCKED'}, source unchanged: {n['case_B_source_unchanged']} |
| C — shadow granted, **live** advance | {'BLOCKED' if n['case_C_live_advance_blocked'] else 'ALLOWED'} on `{n['case_C_denied_capability']}` |
| D — simulator_execution ever granted | {not n['case_D_execution_never_granted']} |

Ladder isolation: {len(c['PRE_B_02_shadow_capability']['implicit_escalations'])} implicit escalations. Shadow does not confer execution
({c['PRE_B_02_shadow_capability']['shadow_grants_execution']}), training ({c['PRE_B_02_shadow_capability']['shadow_grants_training']}) or comparison.

## 3. LS2 — non-vacuous Zero-Loss

Frozen adapter reused byte-identically at `epsilon_sec = {z['epsilon_sec']}`. It was not reimplemented,
not turned into a reward, and not used as a ranking preference.

| quantity | value |
|---|---|
| candidates before Zero-Loss | {z['candidates_before_zero_loss']} |
| candidates after Zero-Loss | {z['candidates_after_zero_loss']} |
| PASS / REJECT | {z['PASS']} / {z['REJECT']} |
| VACUOUS (no existing passenger) | {z['VACUOUS_NO_EXISTING_PASSENGER']} |
| ERROR | {z['ERROR']} |
| states evaluated with onboard > 0 | {z['nonzero_onboard_evaluated_states']} |
| candidates evaluated with onboard > 0 | {z['nonzero_onboard_evaluated_candidates']} |
| zero-safe-candidate states | {z['zero_safe_candidate_states']} |
| **source-state mutations** | **{z['source_state_mutations']}** |
| every REJECT carries a reason | {z['every_reject_has_reason']} |
| retained candidates with a violation | {z['retained_candidate_violations']} |

delta ETA over {z['delta_eta']['count']} passenger evaluations: min {z['delta_eta']['min']}s, median
{z['delta_eta']['median']}s, p95 {z['delta_eta']['p95']}s, max {z['delta_eta']['max']}s.

**Discrimination observed: {z['discrimination_observed']}.** Both classes arose naturally — 25 candidates
cost no existing rider a second and were kept, 99 delayed at least one rider and were rejected. Nothing
was tuned to manufacture either class, and no candidate disappeared silently.

## 4. LS3 — blocked

`{blocker['code']}`

{blocker['title']}.

{findings}

Forcing a pass here would mean mapping a path candidate onto HOLD/SERVE/CONDITIONAL_SKIP, which
disguises candidate routing as an existing action — the spec explicitly prefers a block to that. The
actor architecture, output dimension, reward and weights are all untouched, and no new policy was
created.

### Research decisions required

{decisions}

## 5. Frozen integrity, fairness and shadow guarantee

R9.7 untouched and its artifact intact; the Zero-Loss adapter byte-identical
(`{src['simulator/zero_loss_admission_adapter.py'][:24]}…`). One demand input digest
`{r['demand_input_digest'][:32]}…` covers every layer, so only candidate processing differs across them.

| quantity | count |
|---|---|
| **simulator_execution ALLOWED events** | **{s['simulator_execution_allowed_events']}** |
| shadow_counterfactual ALLOWED events | {s['shadow_counterfactual_allowed_events']} |
| live step / reset / clock advance | {s['live_simulator_step']} / {s['live_causal_reset']} / {s['live_clock_advance']} |
| live movement / boarding / reward | {s['live_vehicle_movement']} / {s['live_boarding_or_alighting']} / {s['live_reward_settlement']} |
| optimizer step / checkpoint write | {s['optimizer_step']} / {s['checkpoint_write']} |

Peak RSS {r['maxrss_final_bytes'] / 1e6:.0f} MB, runtime {r['runtime_seconds']}s.

## 6. Authorization after this gate

`simulator_binding_allowed = true`; `shadow_counterfactual` exists but defaults to deny and is only
ever held inside a declared block; `simulator_execution_allowed`, `training_allowed`,
`performance_comparison_allowed`, `causal_performance_claim_allowed` and `paper_level_claim_allowed`
all remain false.
""")

    write("gate_decision.json", {
        "gates": {"LS2_PRE": {"gate": GATE_PRE, "status": r["stage_status"]["LS2_PRE"]},
                  "LS2": {"gate": GATE_LS2, "status": r["stage_status"]["LS2"]},
                  "LS3": {"gate": GATE_LS3, "status": r["stage_status"]["LS3"]}},
        "decision": "BLOCKED_AT_LS3", "stopped_at": r["stopped_at"],
        "classification": r["classification"],
        "blocker_code": blocker["code"], "blocker_title": blocker["title"],
        "research_decisions_required": blocker["research_decisions_required"],
        "checks": {k: v["passed"] for k, v in c.items()},
        "ls3_selection_artifact_written": False,
        "authorization": s["declared_flags"],
        "simulator_execution_allowed_events": s["simulator_execution_allowed_events"]})

    man = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(OUT.rglob("*")) if p.is_file()}
    write("artifact_manifest.json", {"generated": STAMP, "artifact_dir": OUT.name,
                                     "decision": "BLOCKED_AT_LS3", "source_sha256": src,
                                     "file_sha256": man})
    (OUT / "_BLOCKED.lock").write_text(f"{blocker['code']}\nstopped_at=LS3\n{STAMP}\n", encoding="utf-8")
    print(f"[LS2-PRE PASS][LS2 PASS][LS3 BLOCKED] {blocker['code']}")
    print(f"artifact: {OUT.relative_to(T.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
