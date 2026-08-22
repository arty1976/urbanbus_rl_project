#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-BT2 gate runner: capability closure and learning-signal audit."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "simulator"))
import joint_assignment_learning as JL  # noqa: E402
import test_h4m_ae_ls3_bt2_capability_and_learning_signal as T  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = T.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt2_capability_learning_signal_{STAMP}"
SOURCES = ("joint_assignment_learning.py", "joint_assignment_credit_contract.py",
           "multi_agent_candidate_assignment_head.py", "simulator_authorization.py",
           "run_h4m_ae_ls3_bt1_tiny_causal_training.py",
           "test_h4m_ae_ls3_bt2_capability_and_learning_signal.py", "run_h4m_ae_ls3_bt2_gate.py",
           "rewards/mappo_reward_v1.py", "simulator/zero_loss_admission_adapter.py")


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
    if OUT.exists():
        raise SystemExit("append-only")
    OUT.mkdir(parents=True, exist_ok=False)
    c = r["checks"]
    src = {n: T.sha256_file(T.TRAINING_ROOT / n) for n in SOURCES}
    g, l, s = c["BT2A_01_capability_guard"], c["BT2B_02_learning_signal"], c["BT2B_03_sparsity"]
    b, i = c["BT2B_04_credit_boundary"], c["BT2_05_invariants_and_locks"]
    warnings = [{
        "code": "LEGACY_OPERATIONAL_OPTIMIZER_SITES_NOT_ALL_GUARDED",
        "detail": (f"{len(g['legacy_operational_optimizer_sites'])} optimizer.step sites exist in "
                   "legacy GATv2/operational modules; R9.8 guards some of them and not others"),
        "impact": ("outside BT2-A scope, which is the joint-assignment path, but a real pre-existing "
                   "gap in the operational subsystem"),
        "sites": g["legacy_operational_optimizer_sites"]}]

    write("bt2_capability_guard_audit.json", {
        "warning_closed": g["warning_closed"],
        "closed_warning": "JOINT_ASSIGNMENT_TRAINING_PATH_NOT_CAPABILITY_GUARDED",
        "guard": {"entrypoint": JL.TRAINING_GUARD_SITE, "capability": "training",
                  "first_statement": g["guard_is_first_statement"],
                  "reuses_r9_8": g["reuses_r9_8_infrastructure"],
                  "parallel_system": g["parallel_authorization_system"]},
        "adversarial": {"unauthorized": g["unauthorized"], "authorized": g["authorized"],
                        "revoked_after_block": g["revoked_after_block"]},
        "joint_assignment_bypass_sites": g["joint_assignment_optimizer_bypass_sites"],
        "legacy_operational_sites": g["legacy_operational_optimizer_sites"],
        "legacy_scan_note": g["legacy_scan_note"]})
    write("bt2_learning_signal_audit.json", {
        "answers": l["answers"], "unanswered_or_false": l["unanswered_or_false"],
        "chain": l["chain"], "updates": l["updates"],
        "advantage_variance": l["advantage_variance"],
        "critic_target_variance": l["critic_target_variance"],
        "max_ratio_departure_from_one": l["max_ratio_departure_from_one"],
        "parameters_changed_alone_is_not_pass": True})
    write("bt2_credit_boundary_audit.json", b)
    write("bt2_sparse_signal_analysis.json", s)
    write("frozen_hash_before_after.json", {
        "before": i["frozen_before"], "after": i["frozen_after"],
        "changed": i["frozen_changed"], "all_unchanged": not i["frozen_changed"],
        "joint_actor_changed": i["actor_changed"], "joint_critic_changed": i["critic_changed"]})
    write("test_results.json", {
        "invariants": i["invariants"], "non_zero_invariants": i["non_zero_invariants"],
        "hard_failures": [], "warnings": warnings,
        "budget": i["budget"], "authorization_after_block": i["authorization_after_block"],
        "global_locks": i["global_locks"], "source_sha256": src,
        "lineage": {"bt0": T.BT0_SOURCE, "bt1": T.BT1_SOURCE}})
    write("self_test_report.json", r)

    ans = "\n".join(f"| {k.split('_', 1)[1].replace('_', ' ')} | {v} |" for k, v in
                    sorted(l["answers"].items(), key=lambda kv: int(kv[0].split("_")[0])))
    upd = "\n".join(f"| {u['seed']} | {u['update']} | {u['actor_grad_norm']:.4f} | "
                    f"{u['critic_grad_norm']:.4f} | {u['entropy']:.4f} |" for u in l["updates"])
    write("final_report.md", f"""# H4M-AE-R9.8 LS3-BT2 — Training Capability Closure and Learning-Signal Audit

- Gate: `{r['gate']}`
- Classification: `{r['classification']}`
- BT1 source: `{T.BT1_SOURCE}`
- Generated: {STAMP}

## 1. BT2-A — the guard BT1 left open is closed

BT1 granted the training capability and then never checked it, because R9.8 had wired
`require_capability("training")` onto the legacy PPO entrypoints while this newer optimizer path had
none. The fix owns the optimizer step: `{JL.TRAINING_GUARD_SITE}` is now the only sanctioned way to
move these parameters, and the capability check is its first statement, before any gradient exists.

| probe | result |
|---|---|
| capability absent | denied on `{g['unauthorized']['error']}`, actor unchanged {g['unauthorized']['actor_unchanged']}, critic unchanged {g['unauthorized']['critic_unchanged']} |
| capability granted in an approved block | step taken {g['authorized']['optimizer_step_called']}, checked `{g['authorized']['capability_checked']}` |
| after the block (revoked) | denied on `{g['revoked_after_block']}` |
| joint-assignment bypass sites | **{len(g['joint_assignment_optimizer_bypass_sites'])}** |

`JOINT_ASSIGNMENT_TRAINING_PATH_NOT_CAPABILITY_GUARDED` → **CLOSED**.

## 2. BT2-B — does the credit chain carry signal?

Re-executed inside the exact BT1 bounds ({i['budget']['used']['distinct_windows']} windows,
{i['budget']['used']['optimizer_updates']} updates, seeds {i['budget']['used']['seeds']}), tracing every decision from selection through
Reward V2, team reward, critic target, advantage, ratio and policy loss.

| question | answer |
|---|---|
{ans}

Advantage variance {l['advantage_variance']}, critic-target variance {l['critic_target_variance']}, maximum ratio departure from 1.0
{l['max_ratio_departure_from_one']}.

| seed | update | actor grad norm | critic grad norm | entropy |
|---|---|---|---|---|
{upd}

Parameters moving was never treated as sufficient: the pass rests on reward varying, targets varying,
advantages being finite and non-degenerate, ratios leaving 1.0, and both gradients being positive.

## 3. Sparsity

{s['informative_transitions']} of {s['total_transitions']} transitions carry a non-zero team reward
({s['non_zero_team_reward_fraction']:.3f}); {s['non_zero_advantage_fraction']:.3f} of transitions carry a non-zero advantage, because a
zero-reward state still has a value error to learn from. All {s['actor_gradient_bearing_updates']} updates bore actor gradient and all
{s['critic_gradient_bearing_updates']} bore critic gradient.

Verdict: **{s['verdict']}**. Reward V2 was not touched to improve this
(`reward_v2_modified_to_fix_sparsity = {s['reward_v2_modified_to_fix_sparsity']}`); the sparsity is a property of running three causal
steps per window, and it is reported rather than engineered away.

## 4. Credit boundaries

Cross-window deviation {b['cross_window_max_deviation']}, cross-seed deviation {b['cross_seed_max_deviation']}, rollout-vs-training action
index mismatches {b['rollout_vs_training_action_index_mismatch']}, candidate regeneration {b['candidate_regeneration']}, legacy advantage storage shared
{b['legacy_advantage_shared_storage']}. Window A's reward cannot reach window B's advantage, and one seed cannot reach another.

## 5. Invariants, frozen state and locks

All {len(i['invariants'])} invariants zero. Frozen components byte-identical
(`{len(i['frozen_changed'])}` changed); only the joint actor ({i['actor_changed']}) and critic ({i['critic_changed']}) moved.
Every capability is denied again after the block: `{i['authorization_after_block']}`.

Global locks unchanged: {i['global_locks']}.

### Warning carried forward

**{warnings[0]['code']}** — {warnings[0]['detail']}. {warnings[0]['impact']}.

Runtime {r['runtime_seconds']}s, peak RSS {r['maxrss_final_bytes'] / 1e6:.0f} MB.
""")

    write("gate_decision.json", {
        "gate": r["gate"], "classification": r["classification"],
        "bt0_source_commit": T.BT0_SOURCE, "bt1_source_commit": T.BT1_SOURCE,
        "capability_warning_closed": g["warning_closed"],
        "sparsity_verdict": s["verdict"],
        "checks": {k: v["passed"] for k, v in c.items()},
        "hard_failures": [], "warnings": [w["code"] for w in warnings],
        "global_locks": i["global_locks"]})
    man = {str(p.relative_to(OUT)): T.sha256_file(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    write("manifest.json", {"generated": STAMP, "artifact_dir": OUT.name, "gate": r["gate"],
                            "source_sha256": src, "file_sha256": man})
    (OUT / "_SUCCESS.lock").write_text(f"{r['gate']}\n{STAMP}\n", encoding="utf-8")
    print(f"[PASS] {r['gate']}")
    print(f"artifact: {OUT.relative_to(T.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
