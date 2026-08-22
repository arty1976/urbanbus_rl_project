#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-CR0..CR4 gate runner."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_assignment_credit_contract as CC  # noqa: E402
import joint_assignment_learning as JL  # noqa: E402
import test_h4m_ae_ls3_joint_assignment_credit as T  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = T.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_joint_assignment_credit_{STAMP}"
GATES = {
    "CR0": "PASS_SUSEONG_H4M_AE_LS3_CR0_JOINT_ASSIGNMENT_CREDIT_REWARD_CONTRACT_FROZEN",
    "CR1": "PASS_SUSEONG_H4M_AE_LS3_CR1_ASSIGNMENT_TRANSITION_AND_FROZEN_ACTION_SUPPORT_BUFFER_COMPLETE",
    "CR2": "PASS_SUSEONG_H4M_AE_LS3_CR2_CENTRALIZED_JOINT_CRITIC_AND_DURATION_AWARE_GAE_COMPLETE",
    "CR3": "PASS_SUSEONG_H4M_AE_LS3_CR3_JOINT_ASSIGNMENT_PPO_INTERFACE_AND_GRADIENT_OWNERSHIP_COMPLETE",
    "CR4": "PASS_SUSEONG_H4M_AE_LS3_CR4_JOINT_ASSIGNMENT_TRAINING_READINESS_COMPLETE_TRAINING_STILL_LOCKED",
}
SOURCES = ("joint_assignment_credit_contract.py", "joint_assignment_learning.py",
           "test_h4m_ae_ls3_joint_assignment_credit.py",
           "run_h4m_ae_ls3_joint_assignment_credit_gate.py",
           "multi_agent_candidate_assignment_head.py", "multi_agent_assignment_contract.py",
           "simulator/zero_loss_admission_adapter.py", "rewards/mappo_reward_v1.py")


def write(name, payload):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload if isinstance(payload, str) else
                 json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
                 encoding="utf-8")


def main() -> None:
    r = T.run_validations()
    if r["failed_checks"]:
        raise SystemExit(f"BLOCKED: {r['failed_checks']}")
    OUT.mkdir(parents=True, exist_ok=False)
    c = r["checks"]
    src = {n: hashlib.sha256((T.TRAINING_ROOT / n).read_bytes()).hexdigest() for n in SOURCES}

    write("joint_assignment_credit_contract_v1.json", {
        "gate": GATES["CR0"], "contract": CC.CREDIT_CONTRACT,
        "contract_digest": CC.contract_digest(), "verified": c["CR0_01_credit_contract"]})
    write("joint_assignment_team_reward_report.json", c["CR0_02_team_reward"])
    write("duration_aware_discount_validation.json", c["CR0_03_duration_discount"])
    write("assignment_transition_schema.json", {
        "gate": GATES["CR1"],
        "fields": sorted(CC.AssignmentTransition.__dataclass_fields__),
        "verified": c["CR1_04_transition_and_support"]})
    write("action_support_freeze_validation.json", c["CR1_04_transition_and_support"])
    write("joint_assignment_critic_contract.json", {
        "gate": GATES["CR2"], "contract": JL.CRITIC_CONTRACT, "verified": c["CR2_05_critic"]})
    write("assignment_gae_validation.json", c["CR2_06_assignment_gae"])
    write("window_boundary_contamination_report.json", {
        "cross_window_advantage_contamination": c["CR2_06_assignment_gae"]["cross_window_advantage_contamination"],
        "window_shuffle_max_deviation": c["CR2_06_assignment_gae"]["window_shuffle_max_deviation"],
        "resets_at": CC.CREDIT_CONTRACT["boundary"]["gae_resets_at"]})
    write("joint_assignment_ppo_interface_report.json", {
        "gate": GATES["CR3"], "contract": JL.PPO_CONTRACT, "verified": c["CR3_07_ppo_interface"]})
    write("forced_action_validation.json", {
        "rule": CC.CREDIT_CONTRACT["forced_action"],
        "forced_rows": c["CR3_07_ppo_interface"]["forced_rows"],
        "forced_only_policy_loss": c["CR3_07_ppo_interface"]["forced_only_policy_loss"],
        "forced_only_entropy": c["CR3_07_ppo_interface"]["forced_only_entropy"],
        "actor_denominator": c["CR3_07_ppo_interface"]["actor_denominator"]})
    write("gradient_ownership_report.json", c["CR3_08_gradient_ownership"])
    write("authorization_no_side_effect_report.json", {
        **c["CR4_10_readiness"], "source_scanner": c["CR4_09_source_scanner"]})
    write("frozen_upstream_regression_report.json", {
        "reward_v2_freeze_present": c["CR0_01_credit_contract"]["reward_v2_freeze_present"],
        "reward_v2_runtime_binding_sha256": CC.REWARD_V2_RUNTIME_BINDING_SHA,
        "zero_loss_adapter_byte_identical": c["CR0_01_credit_contract"]["zero_loss_adapter_byte_identical"],
        "zero_loss_epsilon": CC.ZERO_LOSS_EPSILON_SEC,
        "r9_7_artifact_mismatched": c["CR0_01_credit_contract"]["r9_7_artifact_mismatched"],
        "ja3_artifact_mismatched": c["CR0_01_credit_contract"]["ja3_artifact_mismatched"],
        "source_sha256": src})
    write("self_test_report.json", r)

    a, d = c["CR0_02_team_reward"], c["CR0_03_duration_discount"]
    b, g = c["CR1_04_transition_and_support"], c["CR2_06_assignment_gae"]
    p, o = c["CR3_07_ppo_interface"], c["CR3_08_gradient_ownership"]
    rd = c["CR4_10_readiness"]

    write("final_report.md", f"""# H4M-AE-R9.8 LS3-CR0..CR4 — Joint Assignment Credit Contract and PPO Readiness

- Stages: **CR0 {r['stage_status']['CR0']} · CR1 {r['stage_status']['CR1']} · CR2 {r['stage_status']['CR2']} · CR3 {r['stage_status']['CR3']} · CR4 {r['stage_status']['CR4']}**
- Classification: `{r['classification']}`
- Credit contract: `{CC.CONTRACT_ID}` digest `{r['credit_contract_digest'][:32]}…`
- **training_allowed = {r['training_allowed']}**
- Generated: {STAMP}

## 1. What was frozen

Reward V2 is untouched. This is a new attribution layer *over* frozen Reward V2 outputs: it decides
which already-computed rewards belong to which assignment decision, never how they are computed.

    team_reward_t = mean( RewardV2_t(agent_i) for active agent_i )

Arithmetic mean over active agents only. Fixture A: rewards 1/2/3 give **{a['fixture_A_mean_of_1_2_3']}**, and an inactive
agent carrying 99 is excluded ({a['fixture_A_inactive_excluded']}). Fixture B: the same reward distribution at 6, 30 and
126 active agents gives {list(a['fixture_B_scale_invariance'].values())} — scale does not drift with fleet size. Zero active
agents fails closed with `{a['zero_active_agents_error']}` rather than being defined as zero.

## 2. Assignment interval and duration-aware credit

Credit runs over `[T_k, T_(k+1))` as a semi-Markov high-level step, using the repository's own
`gamma = {CC.GAMMA}` and `gae_lambda = {CC.GAE_LAMBDA}` — resolved from the authoritative MAPPO config, not invented.

Fixture C with rewards 1/2/3: `R_assign` = **{d['computed_return']}** against expected {d['expected_return']} (exact: {d['return_exact']}),
and `Gamma_assign` for three operational steps = **{d['computed']}** = gamma³ (exact: {d['bootstrap_exact']}).
With one operational step per assignment it reduces to plain gamma ({d['delta_1_reduces_to_gamma']}); no simulator
cadence was changed to make that happen. A non-positive duration raises `{d['non_positive_duration_error']}`.

## 3. Frozen action support

The safe set is rollout evidence, not something regenerated at update time — regeneration could change
the candidate count, mask or denominator and silently invalidate every PPO ratio.

| property | result |
|---|---|
| unsafe pair in stored support | rejected: `{b['unsafe_pair_in_support_error']}` |
| support mutated between rollout and update | rejected: `{b['action_support_mutation_error']}` |
| reorder gives the same identity digest | {b['reorder_digest_stable']} |
| Local Search / Zero-Loss rerun at update | {b['local_search_rerun_at_update']} / {b['zero_loss_rerun_at_update']} |
| shared advantage storage with operational head | {b['shared_advantage_storage']} |
| buffer | {b['buffer_stats']['transitions']} rows, {b['buffer_stats']['forced_rows']} forced, {b['buffer_stats']['non_forced_rows']} non-forced, {b['buffer_stats']['unsafe_rows_in_support']} unsafe |

## 4. Critic and GAE

`JointAssignmentCritic` predicts V_joint(S_k) *before* selection. Its forward signature contains
**{len(c['CR2_05_critic']['selected_action_leakage_arguments'])}** leakage arguments — no selected agent, no selected candidate, no post-selection
identity, no future reward or state. The safe set reaches it only as an aggregate summary
({c['CR2_05_critic']['safe_summary_dim']} dims: count, mean, best), never as identity.

**Cross-window contamination = {g['cross_window_advantage_contamination']}.** Window A's advantages are identical whether computed
alone or concatenated with a window whose rewards are 50× larger, and shuffling window order moves
them by {g['window_shuffle_max_deviation']}. GAE resets at every episode and window boundary and at every terminated or
truncated row. This project has been bitten by exactly this before, so it is a hard gate rather than
an assertion.

## 5. PPO interface and gradient ownership

| property | result |
|---|---|
| ratios finite | {p['ratio_finite']} — {p['ratio_values']} |
| forced / non-forced rows | {p['forced_rows']} / {p['non_forced_rows']} |
| actor denominator | {p['actor_denominator']} (non-forced rows only) |
| forced-row policy loss / entropy | {p['forced_only_policy_loss']} / {p['forced_only_entropy']} |
| advantage detached from critic graph | {p['advantage_detached']} |
| old_log_prob recomputed after mutation | {p['old_log_prob_recomputed_after_policy_mutation']} |

A forced row — NO_ASSIGN as the only legal action — has log-prob identically 0 and no policy gradient
to give. It contributes zero to actor and entropy and is kept out of the denominator, so a batch full
of them cannot silently shrink the gradient; it still trains the critic. That zero is structural, not
an architecture failure.

**Gradient ownership.** Joint actor gradient nonzero ({o['joint_actor_gradient_nonzero']}), joint critic gradient
{o['joint_critic_gradient']:.1f}, and the frozen components receive exactly nothing:
GATv2 {o['gatv2_gradient']}, operational actor {o['operational_actor_gradient']}, operational critic {o['operational_critic_gradient']}. Their parameter
digests are unchanged ({o['frozen_parameter_digests_unchanged']}). `optimizer.step()` count **{o['optimizer_step_count']}**.

## 6. Readiness and what is still missing

Source scan of the new modules: **{c['CR4_09_source_scanner']['total_findings']}** findings — no `simulator_execution` grant, no training
grant, no live simulator call, no optimizer step, no checkpoint write, no builtin `hash()`.

The JA3 shadow artifact supplies {rd['ja3_decision_groups']} decision groups and {rd['ja3_safe_pairs']} safe pairs, but it carries no
causal rollout, so there are no real Reward V2 sequences to attribute:
`{rd['real_reward_status']}`. Numeric credit behaviour is therefore proven on structural fixtures
labelled `{rd['structural_fixtures_label']}`, and {rd['rewards_fabricated']} rewards were fabricated. Those fixtures are not
mixed into any research result.

## 7. Authorization

`simulator_execution` ALLOWED events **{rd['simulator_execution_allowed_events']}**, training ALLOWED events **{rd['training_allowed_events']}**,
optimizer steps {rd['optimizer_step_count']}, checkpoint writes {rd['checkpoint_writes']}, live steps/resets/movements
{rd['live_simulator_step']}/{rd['live_causal_reset']}/{rd['live_vehicle_movement']}. `training_allowed` stays **false**: the credit path is ready for a
separately authorized bounded training gate, and no training was performed.

Runtime {r['runtime_seconds']}s, peak RSS {r['maxrss_final_bytes'] / 1e6:.0f} MB.
""")

    write("gate_decision.json", {
        "gates": {k: {"gate": GATES[k], "status": r["stage_status"][k]} for k in GATES},
        "classification": r["classification"],
        "credit_contract_digest": r["credit_contract_digest"],
        "training_allowed": False,
        "checks": {k: v["passed"] for k, v in c.items()},
        "cross_window_contamination": g["cross_window_advantage_contamination"],
        "optimizer_step_count": 0, "checkpoint_writes": 0,
        "simulator_execution_allowed_events": rd["simulator_execution_allowed_events"],
        "real_reward_status": rd["real_reward_status"]})

    man = {str(p_.relative_to(OUT)): hashlib.sha256(p_.read_bytes()).hexdigest()
           for p_ in sorted(OUT.rglob("*")) if p_.is_file()}
    write("manifest.json", {"generated": STAMP, "artifact_dir": OUT.name,
                            "source_sha256": src, "file_sha256": man})
    (OUT / "_SUCCESS.lock").write_text(f"{GATES['CR4']}\n{STAMP}\n", encoding="utf-8")
    print(f"[CR0-CR4 PASS] {r['classification']}")
    print(f"artifact: {OUT.relative_to(T.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
