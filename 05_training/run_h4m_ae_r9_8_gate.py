#!/usr/bin/env python3
"""H4M-AE-R9.8 gate runner: simulator authorization enforcement and binding promotion."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulator_authorization as AUTH  # noqa: E402
import simulator_binding as BIND  # noqa: E402
import test_h4m_ae_r9_8_authorization_enforcement as T  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = T.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_r9_8_authorization_enforcement_{STAMP}"
GATE = ("PASS_SUSEONG_DL6D_PA1A_R8E_R3_RH4M_AE_R9_8_SIMULATOR_AUTHORIZATION_ENFORCEMENT"
        "_AND_BINDING_PROMOTION_GATE_COMPLETE")

SOURCES = ("simulator_authorization.py", "simulator_binding.py",
           "test_h4m_ae_r9_8_authorization_enforcement.py", "run_h4m_ae_r9_8_gate.py",
           "causal_kpi_bridge.py", "adapters/causal_simulator_adapter.py",
           "adapters/causal_simulator_v2_adapter.py", "adapters/historical_replay_adapter.py",
           "adapters/route_aware_minimal_simulator_step101.py",
           "simulator/dynamics_multiagent_orchestrator.py",
           "simulator/suseong_service_transition_engine.py", "simulator/k_safety_state.py")


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
    if not result["all_passed"]:
        raise SystemExit(f"BLOCKED: {result['failed_checks']}")
    OUT.mkdir(parents=True, exist_ok=False)
    c = result["checks"]
    src = {n: hashlib.sha256((T.TRAINING_ROOT / n).read_bytes()).hexdigest() for n in SOURCES}
    promotable = result["simulator_binding_allowed"]

    write("authorization_contract.json", {
        "module": "05_training/simulator_authorization.py", "sha256": src["simulator_authorization.py"],
        "contract": AUTH.AUTHORIZATION_CONTRACT,
        "capabilities": {c_: AUTH.CAPABILITY_MEANING[c_] for c_ in AUTH.CAPABILITY_LADDER},
        "enforcement_api": ["require_capability", "check_capability", "granted"],
        "verified": c["R9_8_02_authorization_contract"],
        "ladder_isolation": c["R9_8_03_ladder_isolation"]})

    write("binding_entrypoint_contract.json", {
        "module": "05_training/simulator_binding.py", "sha256": src["simulator_binding.py"],
        "surface_contract": BIND.SURFACE_CONTRACT,
        "read_only_left_unguarded_by_design": True,
        "handoff_module_left_byte_identical": True,
        "handoff_reason": ("simulator_demand_handoff sha256 is recorded in the R9.7 ledger provenance, "
                           "so touching it would move the frozen ledger digest"),
        "verified": c["R9_8_08_enforcement_site_audit"]["audits"]["simulator_binding"]})

    write("execution_enforcement_audit.json", {
        "causal_mutation_entrypoints": [
            {"module": m, "class": cl or None, "function": f} for m, cl, f in T.CAUSAL_MUTATION_ENTRYPOINTS],
        "audit": c["R9_8_08_enforcement_site_audit"]["audits"]["simulator_execution"],
        "no_bypass": c["R9_8_09_no_bypass"]})

    write("training_and_comparison_enforcement.json", {
        "training": c["R9_8_10_training_guard"],
        "performance_comparison": c["R9_8_11_comparison_guard"],
        "policy_independence": c["R9_8_12_policy_independence"]})

    write("negative_authorization_tests.json", {
        "case_A_nothing_granted": c["R9_8_04_negative_case_A"],
        "case_B_binding_only": c["R9_8_05_negative_case_B"],
        "case_C_probe_only": c["R9_8_06_case_C_probe_only"]})

    write("state_immutability_proof.json", c["R9_8_07_state_immutability_on_denial"])

    write("enforcement_site_audit.json", c["R9_8_08_enforcement_site_audit"])

    write("binding_promotion_decision.json", {
        **c["R9_8_15_binding_promotion"], "gate": GATE,
        "supersedes": "BINDING_PROMOTION_BLOCKED_BY_AUTHORIZATION_SEMANTICS (R9.7)",
        "what_changed_since_r9_7": ("R9.7 found 14 declaration sites and 0 enforcement sites; R9.8 adds "
                                    "a central fail-closed module and wires it into every causal "
                                    "mutation entrypoint, the live binding path, the training path and "
                                    "the comparison path")})

    write("provenance_manifest.json", {
        "gate": GATE, "stage": result["stage"], "classification": result["classification"],
        "execution_base_sha": T.EXEC_BASE_SHA, "r9_7_source_sha": T.R97_SOURCE_SHA,
        "r9_7_ledger_digest": T.R97_LEDGER_DIGEST, "loaded_demand_digest": T.R97_LOADED_DIGEST,
        "frozen_upstream": c["R9_8_01_frozen_upstream"],
        "source_sha256": src,
        "demand_regression": c["R9_8_13_demand_regression"],
        "fairness_regression": c["R9_8_14_fairness_regression"],
        "claim_guards": c["R9_8_16_claim_guards"], "prohibitions": c["R9_8_17_prohibitions"]})

    write("self_test_report.json", result)

    ea = c["R9_8_08_enforcement_site_audit"]
    nb = c["R9_8_09_no_bypass"]
    ca, cb, cc = (c["R9_8_04_negative_case_A"], c["R9_8_05_negative_case_B"],
                  c["R9_8_06_case_C_probe_only"])
    im = c["R9_8_07_state_immutability_on_denial"]
    dr = c["R9_8_13_demand_regression"]
    bp = c["R9_8_15_binding_promotion"]
    pr = c["R9_8_17_prohibitions"]
    ladder_rows = "\n".join(
        f"| `{cap}` | {AUTH.CAPABILITY_MEANING[cap]} | "
        f"{'yes' if cap == 'simulator_binding' and promotable else 'no'} |"
        for cap in AUTH.CAPABILITY_LADDER)

    write("final_report.md", f"""# H4M-AE-R9.8 — Simulator Authorization Enforcement and Binding Promotion

- Gate: `{GATE}`
- Classification: `{result['classification']}`
- execution_base_sha: `{T.EXEC_BASE_SHA}`
- R9.7 lineage sha: `{T.R97_SOURCE_SHA}`
- Generated: {STAMP}

## 1. What changed

R9.7 found the authorization flags had 14 declaration sites and **zero** enforcement sites: they were
statements in artifact payloads that no branch ever read. R9.8 makes them a mechanism. One canonical
module holds the capability ladder, every capability defaults to deny, and a grant is an explicit,
block-scoped, thread-local context manager that requires a written reason. There is no environment
variable and no global mutable switch.

| capability | meaning | granted after this gate |
|---|---|---|
{ladder_rows}

## 2. Enforcement sites

| capability | production entrypoints protected |
|---|---|
| `simulator_execution` | {ea['production_enforcement_per_capability']['simulator_execution']} |
| `simulator_binding` | {ea['production_enforcement_per_capability']['simulator_binding']} |
| `training` | {ea['production_enforcement_per_capability']['training']} |
| `performance_comparison` | {ea['production_enforcement_per_capability']['performance_comparison']} |

{ea['site_classification']['production']} production modules carry enforcement; {ea['site_classification']['test_or_scanner']} test/scanner module is excluded from the count
because a validator that scans for a guard is not a guard.

**No-bypass audit:** {nb['protected_mutation_entrypoints']} of {nb['causal_mutation_entrypoints']} causal mutation entrypoints are protected, with
**{nb['unprotected_count']} unprotected**. Every guard is the first statement of its function body
(`guard_before_first_mutation = {nb['guard_before_first_mutation']}`), so the check cannot be reached after a mutation. The
protected set spans four simulator adapters, the causal KPI bridge, the multi-agent orchestrator, the
service transition engine and the K-safety clock — not an outer runner.

## 3. Negative tests

**Case A — nothing granted.** Live binding {'BLOCKED' if ca['binding_blocked'] else 'ALLOWED'}, simulator step {'BLOCKED' if ca['step_blocked'] else 'ALLOWED'},
state mutated: {ca['simulator_state_mutated']}.

**Case B — binding granted, execution denied.** This is the separation test.
Binding {'ALLOWED' if cb['binding_allowed'] else 'BLOCKED'} and it bound {cb['bound_demand']['bound_request_count']} demand identities
({cb['serviceable']} serviceable, {cb['unserviceable']} unserviceable). The simulator step was still {'BLOCKED' if cb['step_blocked'] else 'ALLOWED'},
state mutated: {cb['simulator_state_mutated']}, and binding did not implicitly enable execution
(`{cb['binding_implicitly_enabled_execution']}`).

**Case C — both granted, nothing run.** A pre-mutation probe reports that binding
({cc['probe_binding_would_be_admitted']}) and execution ({cc['probe_execution_would_be_admitted']}) would be admitted, while training ({not cc['probe_training_still_denied']}) and
comparison ({not cc['probe_comparison_still_denied']}) stay denied. No `step`, `reset` or `advance` was called:
`simulator_advanced = {cc['simulator_advanced']}`, state digest unchanged (`{cc['state_unchanged']}`).

## 4. A denied call changes nothing

Measured on a fake simulator and on the real causal bridge, before and after denial:

| quantity | value |
|---|---|
| denied outcomes | {im['denied_outcomes']} |
| fake simulator state digest equal | {im['fake_simulator']['state_unchanged']} |
| causal bridge state digest equal | {im['causal_bridge']['state_unchanged']} |
| time advance | {im['time_advance']} |
| vehicle movement | {im['vehicle_movement']} |
| boarding / alighting | {im['boarding']} / {im['alighting']} |
| reward rows / event rows | {im['reward_rows']} / {im['event_rows']} |
| checkpoint writes | {im['checkpoint_writes']} |

The digest walks the object's own attribute dictionary rather than a hand-listed subset, so a mutation
nobody anticipated would still show up.

## 5. The ladder does not leak

Granting each capability alone and asking every other capability what it answers produced
{len(c['R9_8_03_ladder_isolation']['implicit_escalations'])} implicit escalations. Concretely: binding does not confer execution
({c['R9_8_03_ladder_isolation']['binding_implies_execution']}), execution does not confer training ({c['R9_8_03_ladder_isolation']['execution_implies_training']}), training does not confer
comparison ({c['R9_8_03_ladder_isolation']['training_implies_comparison']}), and comparison does not confer a causal claim
({c['R9_8_03_ladder_isolation']['comparison_implies_claim']}). Training stays denied even with execution granted
({c['R9_8_10_training_guard']['denied_even_with_execution_granted']}), and comparison stays denied even with execution and training granted
({c['R9_8_11_comparison_guard']['denied_even_with_execution_and_training']}).

## 6. Policy independence

{len(c['R9_8_12_policy_independence']['policy_branches_in_authorization'])} policy branches exist in the authorization module. All three arms receive the identical
answer: {c['R9_8_12_policy_independence']['per_arm_denial']}. No arm can take a different path to a mutation.

## 7. Regression

| property | result |
|---|---|
| R9.7 ledger digest unchanged | {dr['ledger_digest_unchanged']} |
| loaded-demand digest unchanged | {dr['loaded_digest_unchanged']} |
| request identities / realized / dropped | {dr['identities']} / {dr['realized']} / {dr['dropped_mass']} |
| full R9.7 validator still passes | {dr['r9_7_all_checks_passed']} |
| B1/B2/A digests identical | {c['R9_8_14_fairness_regression']['all_identical']} |

The binding entrypoint lives in its own module precisely to keep this true: the handoff module's
sha256 is recorded in the R9.7 ledger's provenance, so editing it would have moved the frozen digest.
Separating live binding from read-only handoff also turns the §3 distinction into a module boundary
rather than a comment.

## 8. Nothing was executed

`simulator_execution` was ALLOWED exactly {pr['execution_capability_allowed_events']} time in this entire gate, at
`causal_kpi_bridge.py::reset`, to construct the fixture whose immutability under denial is then
proven. No `step` call was ever admitted. Zero arms executed, zero optimizer steps, zero checkpoint
writes, zero rewards, zero KPIs, no TEST6, no DB writes, no external calls.

## 9. Decision

`{bp['decision']}`

`simulator_binding_allowed = {bp['simulator_binding_allowed']}` now means exactly one thing: {bp['meaning']}.
It does not mean {', '.join(bp['does_not_mean'])}. Those remain false, and they are now false in a way
that is enforced rather than merely written down.
""")

    write("gate_decision.json", {
        "gate": GATE, "decision": "PASS", "stage": result["stage"],
        "classification": result["classification"], "execution_base_sha": T.EXEC_BASE_SHA,
        "checks_passed": len(c), "checks_failed": len(result["failed_checks"]),
        "checks": {k: v["passed"] for k, v in c.items()},
        "simulator_binding_allowed": promotable,
        "simulator_execution_allowed": False, "training_allowed": False,
        "performance_comparison_allowed": False, "causal_performance_claim_allowed": False,
        "binding_decision": bp["decision"],
        "production_enforcement_per_capability": ea["production_enforcement_per_capability"],
        "unprotected_mutation_entrypoints": nb["unprotected_count"],
        "arm_executions": 0,
        "execution_capability_allowed_events": pr["execution_capability_allowed_events"],
        "side_effect": {
            "code": "SIMULATOR_EXECUTION_NOW_FAILS_CLOSED_REPOSITORY_WIDE",
            "detail": ("every causal mutation entrypoint now denies by default; five R3-R6 validation "
                       "harnesses that legitimately exercise the causal bridge were updated to declare "
                       "an explicit grant, and all five still pass"),
            "harnesses_updated": [
                "test_h4m_ae_r3_causal_kpi_bridge.py", "test_h4m_ae_r3_1_measured_wait_tail.py",
                "test_h4m_ae_r4_pre_evaluation_integrity.py",
                "test_h4m_ae_r5_demand_binding_and_horizon_accounting.py",
                "test_h4m_ae_r6_limited_causal_execution.py"]}})

    write("downstream_lock.json", {
        "gate": GATE,
        "simulator_binding_allowed": promotable,
        "simulator_binding_meaning": bp["meaning"],
        "simulator_execution_allowed": False, "training_allowed": False,
        "performance_comparison_allowed": False, "causal_performance_claim_allowed": False,
        "paper_level_claim_allowed": False,
        **{k: v for k, v in c["R9_8_16_claim_guards"].items() if isinstance(v, bool)},
        "enforcement": ("capabilities are enforced by simulator_authorization.require_capability at "
                        "every production entrypoint, not merely declared in artifacts"),
        "permitted_downstream_use": [
            "attaching the frozen demand ledger to the causal simulator input under an explicit "
            "simulator_binding grant"],
        "prohibited_downstream_use": [
            "advancing the simulator", "policy action or stepping", "MAPPO training",
            "optimizer step or checkpoint write", "B1/B2/A performance comparison",
            "reward or KPI computation", "causal performance claim", "paper-level empirical claim"],
        "next_gate_required_before_execution": "H4M-AE-R9.9 or later"})

    man = {}
    for p in sorted(OUT.rglob("*")):
        if p.is_file():
            man[str(p.relative_to(OUT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    write("artifact_manifest.json", {
        "gate": GATE, "generated": STAMP, "artifact_dir": OUT.name,
        "execution_base_sha": T.EXEC_BASE_SHA, "source_sha256": src, "file_sha256": man})
    (OUT / "_SUCCESS.lock").write_text(f"{GATE}\n{STAMP}\n", encoding="utf-8")
    print(f"[PASS] {GATE}")
    print(f"artifact: {OUT.relative_to(T.PROJECT_ROOT)}")
    print(f"binding={promotable} execution=False training=False comparison=False")
    print(f"protected mutation entrypoints={nb['protected_mutation_entrypoints']}/{nb['causal_mutation_entrypoints']} "
          f"unprotected={nb['unprotected_count']} arm_executions=0")


if __name__ == "__main__":
    main()
