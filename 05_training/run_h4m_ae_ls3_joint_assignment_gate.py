#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-JA0..JA3 gate runner."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import multi_agent_assignment_contract as C  # noqa: E402
import multi_agent_candidate_assignment_head as H  # noqa: E402
import test_h4m_ae_ls3_joint_assignment as T  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = T.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_joint_assignment_{STAMP}"
GATES = {
    "JA0": "PASS_SUSEONG_H4M_AE_LS3_JA0_MULTI_AGENT_JOINT_ASSIGNMENT_CONTRACT_COMPLETE",
    "JA1": "PASS_SUSEONG_H4M_AE_LS3_JA1_AGENT_CANDIDATE_SET_AND_INFORMATION_SHARING_CONTRACT_COMPLETE",
    "JA2": "PASS_SUSEONG_H4M_AE_LS3_JA2_ZERO_LOSS_SAFE_MULTI_AGENT_JOINT_ASSIGNMENT_SHADOW_FORWARD_COMPLETE",
    "JA3": ("PASS_SUSEONG_H4M_AE_LS3_MULTI_AGENT_LOCAL_SEARCH_ZERO_LOSS_JOINT_MAPPO_ASSIGNMENT"
            "_ARCHITECTURE_COMPLETE_TRAINING_STILL_LOCKED"),
}
SOURCES = ("multi_agent_assignment_contract.py", "multi_agent_candidate_assignment_head.py",
           "test_h4m_ae_ls3_joint_assignment.py", "run_h4m_ae_ls3_joint_assignment_gate.py",
           "operational_state_layer.py", "local_search_contract.py",
           "simulator/zero_loss_admission_adapter.py", T.OPERATIONAL_ACTOR_MODULE)


def write(name, payload):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload if isinstance(payload, str) else
                 json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
                 encoding="utf-8")


def main() -> None:
    r = T.run_validations()
    groups = r.pop("_groups", []) or []
    outputs = r.pop("_outputs", []) or []
    if r["failed_checks"]:
        raise SystemExit(f"BLOCKED: {r['failed_checks']}")
    OUT.mkdir(parents=True, exist_ok=False)
    c = r["checks"]
    src = {n: hashlib.sha256((T.TRAINING_ROOT / n).read_bytes()).hexdigest() for n in SOURCES}

    write("multi_agent_assignment_contract.json", {
        "gate": GATES["JA0"], "contract": C.ROLE_CONTRACT, "scale_contract": C.SCALE_CONTRACT,
        "verified": c["JA0_01_contract"], "sha256": src["multi_agent_assignment_contract.py"]})
    write("joint_candidate_set_schema.json", {
        "gate": GATES["JA1"],
        "canonical_identity": ["agent_id", "candidate_id"],
        "pair_fields": sorted(C.AgentCandidatePair.__dataclass_fields__),
        "agent_context_features": list(C.AgentContext.FEATURE_NAMES),
        "candidate_features": list(C.LOCAL_SEARCH_FEATURE_NAMES),
        "no_assign_option": C.NO_ASSIGN,
        "decision_groups": [g.payload() for g in groups],
        "verified": c["JA1_02_agent_candidate_representation"]})
    write("permutation_invariance_report.json", c["JA1_03_permutation_invariance"])
    write("information_sharing_report.json", c["JA1_04_information_sharing"])
    write("joint_assignment_head_contract.json", {
        "gate": GATES["JA2"], "head_contract": H.HEAD_CONTRACT,
        "module": "05_training/multi_agent_candidate_assignment_head.py",
        "sha256": src["multi_agent_candidate_assignment_head.py"],
        "forward_validation": c["JA2_06_real_forward"]})
    write("shadow_joint_assignment_selection.json", {
        "gate": GATES["JA2"], "selection_semantics": H.SELECTION_SEMANTICS,
        "policy_quality_claimed": False,
        "structural_tests": c["JA2_05_structural_selection"],
        "shadow_selections": [o.payload() for o in outputs]})
    write("compute_scaling_report.json", c["JA2_07_compute_scaling"])
    write("mappo_ppo_integration_readiness.json", {
        "gate": GATES["JA3"], **c["JA3_08_ppo_readiness"],
        "operational_head_integrity": c["JA3_09_operational_head_survives"]})
    write("safety_and_authorization.json", c["SAFETY_99_shadow_and_authorization"])
    write("provenance_manifest.json", {
        "stage": r["stage"], "stage_status": r["stage_status"],
        "architecture_status": r["architecture_status"], "training_status": r["training_status"],
        "r9_7_source_sha": T.R97_SOURCE_SHA, "r9_8_source_sha": T.R98_SOURCE_SHA,
        "ls0_ls1_source_sha": T.LS01_SOURCE_SHA, "ls2_source_sha": T.LS2_SOURCE_SHA,
        "demand_input_digest": r["demand_input_digest"], "source_sha256": src})
    write("self_test_report.json", r)

    j1, jp, ji = (c["JA1_02_agent_candidate_representation"], c["JA1_03_permutation_invariance"],
                  c["JA1_04_information_sharing"])
    j2, jf, jsc = c["JA2_05_structural_selection"], c["JA2_06_real_forward"], c["JA2_07_compute_scaling"]
    j3, sf = c["JA3_08_ppo_readiness"], c["SAFETY_99_shadow_and_authorization"]
    scale_rows = "\n".join(f"| {g['agents']} | {g['pairs']} | {g['forward_seconds']:.4f} |"
                           for g in jsc["grid"])
    unresolved = "\n".join(f"{i}. {q}" for i, q in enumerate(j3["credit_assignment_unresolved"], 1))

    write("final_report.md", f"""# H4M-AE-R9.8 LS3-JA0..JA3 — Multi-Agent Joint Service Assignment

- Stages: **JA0 {r['stage_status']['JA0']} · JA1 {r['stage_status']['JA1']} · JA2 {r['stage_status']['JA2']} · JA3 {r['stage_status']['JA3']}**
- Architecture: `{r['architecture_status']}` · Training: `{r['training_status']}`
- Classification: `{r['classification']}`
- Generated: {STAMP}

## 1. Canonical interpretation

The decision is cooperative and joint: every agent proposes service plans, Zero-Loss filters each
(agent, candidate) pair, and one selector sees the whole safe set and answers *which agent* takes the
demand *under which plan*. The canonical identity is the pair, not the candidate — the same plan
proposed by two vehicles is two different decisions. Nearest vehicle does not win by construction and
lowest Local Search cost does not win by construction; both are inputs, neither is the rule.

Two levels stay separate and their action spaces are never merged:

    Level 1  cooperative assignment   which agent, which safe plan   (new head)
    Level 2  operational control      HOLD / SERVE / CONDITIONAL_SKIP (promoted actor, untouched)

## 2. JA1 — agent × candidate representation

| quantity | value |
|---|---|
| decision groups | {j1['decision_groups']} |
| distinct agents represented | {j1['distinct_agents_represented']} |
| agents per group (min/max) | {j1['agents_per_group']['min']} / {j1['agents_per_group']['max']} |
| pairs before Zero-Loss | {j1['pairs_before_zero_loss']} |
| **safe pairs after Zero-Loss** | **{j1['safe_pairs_after_zero_loss']}** |
| rejected pairs | {j1['rejected_pairs']} |
| safe-pair counts observed | {j1['safe_pair_counts_observed']} |
| zero-safe groups | {j1['zero_safe_groups']} |
| selectable ∩ rejected | {j1['selectable_intersect_rejected']} |
| source-state mutations | {j1['source_state_mutations']} |

Variable arity is real, not asserted: groups carry 2–4 agents and 0–3 safe pairs.

**Permutation safety.** Agent reorder across {jp['agent_reorder_groups_tested']} groups: max identity-aligned logit deviation
{jp['max_agent_reorder_logit_deviation']:.2e}. Candidate reorder across {jp['candidate_reorder_groups_tested']} groups: {jp['max_candidate_reorder_logit_deviation']:.2e}. Both far inside the
{jp['tolerance']} tolerance — float noise, not preference.

**Cooperation is demonstrated, not claimed.** Perturbing *another* agent's onboard load shifted the
target candidate's logit in {ji['groups_with_nonzero_shift']} of {ji['groups_probed']} probed groups (shift {ji['min_logit_shift_from_other_agent_state']:.2e}–{ji['max_logit_shift_from_other_agent_state']:.2e}).
Scores do not depend only on the candidate's own vehicle.

## 3. JA2 — safe joint selection

Injected-logit structural tests (interface validation, not performance evidence):

| property | result |
|---|---|
| unsafe best candidate not selected | {j2['injected_logit_tests']['unsafe_best_candidate_not_selected']} (its probability {j2['injected_logit_tests']['unsafe_probability']}) |
| highest valid injected maps to correct pair | {j2['injected_logit_tests']['highest_valid_injected_maps_to_correct_pair']} |
| zero-safe selects NO_ASSIGN | {j2['injected_logit_tests']['zero_safe_selects_no_assign']} |
| reorder preserves identity result | {j2['injected_logit_tests']['reorder_preserves_identity_result']} |

On the {j2['real_groups_evaluated']} real groups: **unsafe selections {j2['unsafe_selection_count']}**, **duplicate assignments {j2['duplicate_assignment_count']}**,
zero-safe groups all select NO_ASSIGN ({j2['zero_safe_groups_select_no_assign']}). Non-selected agents remain
`{j2['non_selected_agents_outcome']}`; no plan was mutated.

Real forward under a fixed seed: deterministic init {jf['deterministic_init_under_fixed_seed']}, probabilities sum to 1
{jf['probabilities_sum_to_one']}, unsafe probability mass {jf['unsafe_probability_mass']}, gradient-capable {jf['gradient_capable']}
(measured on a group with {jf['probe_safe_pair_count']} safe pairs — a one-option distribution has log-prob identically
zero and would show no gradient regardless of the head). Which candidate it prefers is **not**
interpreted: the weights are untrained.

## 4. Compute scaling — `{jsc['label']}`

| agents | pairs | forward (s) |
|---|---|---|
{scale_rows}

{jsc['growth_pairs']:.0f}× more pairs cost {jsc['growth_time']:.1f}× more time. Scoring is linear in proposed pairs; no joint action
space is enumerated, and no exponential blow-up appears. Not research-performance evidence.

## 5. JA3 — PPO integration readiness

Classification: **{j3['classification']}**, not vanilla MAPPO ({j3['vanilla_mappo']}) — it is PPO-compatible
and cooperative over a shared context, but its action space is a variable set of (agent, candidate)
pairs rather than a per-agent discrete action. The interface exists: masked log-probability, entropy,
a hard candidate mask, and a ratio over the same masked index space.

**Training stays locked.** Reward V2 is frozen and present, but it does not define assignment credit,
and four questions are genuinely unresolved:

{unresolved}

Deciding any of these would set new research meaning, so `{j3['training_status']}`. This does not
downgrade the architecture, which passes on its own terms.

The promoted operational actor is intact: 3 actions, unchanged, and the new head does not import it
({c['JA3_09_operational_head_survives']['imported_by_joint_head']}).

## 6. Shadow guarantee and authorization

| quantity | count |
|---|---|
| **simulator_execution ALLOWED events** | **{sf['simulator_execution_allowed_events']}** |
| shadow_counterfactual ALLOWED events | {sf['shadow_counterfactual_allowed_events']} |
| source-state mutations | {sf['source_state_mutations']} |
| live step / reset / movement / boarding | {sf['live_simulator_step']} / {sf['live_causal_reset']} / {sf['live_vehicle_movement']} / {sf['live_boarding_or_alighting']} |
| optimizer step / checkpoint write | {sf['optimizer_step']} / {sf['checkpoint_write']} |

`simulator_binding_allowed = true`; `shadow_counterfactual` default-deny and block-scoped; execution,
training, comparison and causal-claim all false. Runtime {r['runtime_seconds']}s, peak RSS
{r['maxrss_final_bytes'] / 1e6:.0f} MB.
""")

    write("gate_decision.json", {
        "gates": {k: {"gate": GATES[k], "status": r["stage_status"][k]} for k in GATES},
        "architecture_status": r["architecture_status"],
        "training_status": r["training_status"],
        "classification": r["classification"],
        "checks": {k: v["passed"] for k, v in c.items()},
        "unsafe_selection_count": j2["unsafe_selection_count"],
        "duplicate_assignment_count": j2["duplicate_assignment_count"],
        "simulator_execution_allowed_events": sf["simulator_execution_allowed_events"],
        "authorization": sf["declared_flags"],
        "credit_assignment_unresolved": j3["credit_assignment_unresolved"]})

    man = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(OUT.rglob("*")) if p.is_file()}
    write("artifact_manifest.json", {"generated": STAMP, "artifact_dir": OUT.name,
                                     "source_sha256": src, "file_sha256": man})
    (OUT / "_SUCCESS.lock").write_text(f"{GATES['JA3']}\n{STAMP}\n", encoding="utf-8")
    print(f"[JA0-JA3 architecture PASS] {r['classification']}")
    print(f"[training] {r['training_status']}")
    print(f"artifact: {OUT.relative_to(T.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
