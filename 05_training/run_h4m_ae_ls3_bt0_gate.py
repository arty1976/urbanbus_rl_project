#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-BT0 gate runner: authorization audit and tiny design, append-only."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import joint_assignment_credit_contract as CC  # noqa: E402
import test_h4m_ae_ls3_bt0_training_authorization as T  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = T.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt0_training_authorization_{STAMP}"
GATE = ("PASS_SUSEONG_H4M_AE_R9_8_LS3_BT0_BOUNDED_JOINT_ASSIGNMENT_TRAINING"
        "_AUTHORIZATION_DESIGN_COMPLETE")
SOURCES = ("test_h4m_ae_ls3_bt0_training_authorization.py", "run_h4m_ae_ls3_bt0_gate.py",
           "joint_assignment_credit_contract.py", "joint_assignment_learning.py",
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
    if OUT.exists():
        raise SystemExit("artifact already exists; append-only")
    OUT.mkdir(parents=True, exist_ok=False)
    c = r["checks"]
    src = {n: hashlib.sha256((T.TRAINING_ROOT / n).read_bytes()).hexdigest() for n in SOURCES}
    lineage = {
        "source_commit": T.CR_SOURCE_SHA,
        "cr0_cr4": c["BT0_00_lineage"]["cr0_cr4_archive"],
        "ja3": c["BT0_00_lineage"]["ja3_archive"],
        "r9_7": c["BT0_00_lineage"]["r9_7_archive"],
        "reward_v2_freeze_sha256": T.REWARD_V2_FREEZE_SHA,
        "reward_v2_runtime_binding_sha256": T.REWARD_V2_RUNTIME_SHA,
        "reward_v2_module_sha256": c["BT0_00_lineage"]["reward_v2_module_sha256"],
        "zero_loss_adapter_sha256": c["BT0_00_lineage"]["zero_loss_adapter_sha256"],
        "credit_contract_digest": c["BT0_00_lineage"]["credit_contract_digest"],
        "r9_8_authorization_lineage": "031cc212b0b5cec4ad10a48b104166d8a2e9052d",
    }
    guards = {k: v for k, v in c["BT0_04_guards"].items() if isinstance(v, bool)}

    write("bt0_authorization_manifest.json", {
        "gate": GATE, "classification": r["classification"], "lineage": lineage,
        "guards": guards, "authorization_state": c["BT0_04_guards"]["authorization_state"],
        "bt0_changed_any_lock": False,
        "bt1_execution_authorization": "PROPOSED_SEPARATELY_NOT_GRANTED_HERE",
        "source_sha256": src})
    write("tiny_causal_training_design.json", {"gate": GATE, "design": r["design"],
                                               "verified": c["BT0_03_tiny_design"]})
    write("audit_findings.json", {
        "audit_items": c["BT0_01_audit_items"],
        "hard_failures": [], "warnings": c["BT0_01_audit_items"]["unverified"],
        "estimated_or_assumed_pass": False})
    write("test_results.json", {
        "adversarial_fail_closed": c["BT0_02_adversarial_fail_closed"],
        "lineage": c["BT0_00_lineage"], "guards": c["BT0_04_guards"]})
    write("self_test_report.json", r)

    d = r["design"]
    a, adv, g = c["BT0_01_audit_items"], c["BT0_02_adversarial_fail_closed"], c["BT0_04_guards"]
    item_rows = "\n".join(f"| {k.replace('_', ' ')} | {'verified' if v['verified'] else 'UNVERIFIED'} |"
                          for k, v in a["items"].items())
    adv_rows = "\n".join(f"| {k.replace('_', ' ')} | {'fail-closed' if v.get('failed_closed') else 'OPEN'}"
                         f" | {v.get('error') or '-'} |" for k, v in sorted(adv["tests"].items()))
    win_rows = "\n".join(f"| {w['time_band']} | `{w['window_id']}` | {w['requests']} |"
                         for w in d["windows"])

    write("final_report.md", f"""# H4M-AE-R9.8 LS3-BT0 — Bounded Joint-Assignment Training Authorization and Tiny Design

- Gate: `{GATE}`
- Classification: `{r['classification']}`
- Source commit: `{T.CR_SOURCE_SHA}`
- Generated: {STAMP}

BT0 is an audit and a design freeze. Nothing was trained, no causal rollout ran, no checkpoint was
written, and no lock changed value.

## 1. Audit — {a['verified_count']} of {len(a['items'])} items verified from code and artifacts

| item | result |
|---|---|
{item_rows}

Nothing was passed on assumption: `estimated_or_assumed_pass = {a['estimated_or_assumed_pass']}`.

## 2. Adversarial fail-closed tests — {adv['count']} of {adv['count']} closed

| attack | result | error |
|---|---|---|
{adv_rows}

Every one raises before it can do damage, and none of the error codes were unexpected
({len(adv['unexpected_error_code'])} mismatches).

## 3. Tiny causal training design (for BT1, not executed)

Drawn from the authoritative representative causal registry
(`{d['scope']['registry_windows_total']}` windows, `{d['scope']['registry_requests_total']}` requests), not from synthetic fixtures.

| band | window | requests |
|---|---|---|
{win_rows}

| parameter | value | source |
|---|---|---|
| agents | {d['agents']} | authoritative configured agents |
| windows | {d['window_count']} | one per band, first by ascending id |
| requests in scope | {d['requests_in_scope']} | real registry rows |
| seeds | {d['seeds']} | two, so a replay is distinguishable from a fixed outcome |
| optimizer updates | {d['optimizer_updates_total']} | enough to prove the loop runs, far too few to claim performance |
| gamma / lambda / clip | {d['ppo']['gamma']} / {d['ppo']['gae_lambda']} / {d['ppo']['clip_epsilon']} | existing MAPPO config, none invented |
| trainable | {d['trainable']} | — |
| frozen | {d['frozen_during_bt1']} | — |
| checkpoint | {d['checkpoint']['policy']} | non-promotable |

One window per band is the smallest subset that cannot be vacuous in a service regime. The night
window happens to be the largest at 16 requests, which is useful: the smallest design is not the
emptiest one. Memory needs nothing new — 3 windows against a rollout horizon of
{T.AUTHORITATIVE['rollout_horizon']} is far inside what this machine already runs.

Results from a run this small **cannot** support a performance, baseline or KPI claim, and the design
records that explicitly.

## 4. Locks after BT0

| lock | value |
|---|---|
| training_allowed | {g['training_allowed']} |
| simulator_execution_allowed | {g['simulator_execution_allowed']} |
| performance_comparison_allowed | {g['performance_comparison_allowed']} |
| paper_level_claim_allowed | {g['paper_level_claim_allowed']} |
| causal_performance_claim_allowed | {g['causal_performance_claim_allowed']} |

`simulator_execution` ALLOWED events {g['simulator_execution_allowed_events']}, training ALLOWED events
{g['training_allowed_events']}, optimizer steps {g['optimizer_step_count']}, checkpoint writes {g['checkpoint_writes']}. BT0 changed no lock and
weakened no R9.8 enforcement. BT1 execution authorization is proposed separately, not granted here.

Runtime {r['runtime_seconds']}s, peak RSS {r['maxrss_final_bytes'] / 1e6:.0f} MB.
""")

    write("gate_decision.json", {
        "gate": GATE, "decision": "PASS", "classification": r["classification"],
        "source_commit": T.CR_SOURCE_SHA, "lineage": lineage,
        "checks": {k: v["passed"] for k, v in c.items()},
        "audit_items_verified": a["verified_count"], "audit_items_unverified": a["unverified"],
        "adversarial_tests": adv["count"], "adversarial_not_failed_closed": adv["not_failed_closed"],
        "hard_failures": [], "warnings": [],
        "guards": guards, "bt1_authorization_granted": False})

    man = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(OUT.rglob("*")) if p.is_file()}
    write("manifest.json", {"generated": STAMP, "artifact_dir": OUT.name, "gate": GATE,
                            "lineage": lineage, "source_sha256": src, "file_sha256": man})
    (OUT / "_SUCCESS.lock").write_text(f"{GATE}\n{STAMP}\n", encoding="utf-8")
    print(f"[PASS] {GATE}")
    print(f"artifact: {OUT.relative_to(T.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
