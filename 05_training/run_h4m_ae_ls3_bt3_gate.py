#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-BT3 gate runner: guard closure and bounded scale design."""

from __future__ import annotations

import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "simulator"))
import test_h4m_ae_ls3_bt3_guard_closure_and_scale as T  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = T.ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt3_guard_closure_scale_design_{STAMP}"
SOURCES = ("test_h4m_ae_ls3_bt3_guard_closure_and_scale.py", "run_h4m_ae_ls3_bt3_gate.py",
           "joint_assignment_learning.py", "simulator_authorization.py",
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
    g, a = c["BT3A_01_guard_inventory"], c["BT3A_02_adversarial"]
    t, l = c["BT3B_03_temporal_credit"], c["BT3B_04_scale_ladder"]
    d, f = c["BT3B_05_selected_design"], c["BT3_06_locks_and_frozen"]

    write("bt3_training_guard_inventory.json", {**g, "adversarial": a})
    write("bt3_temporal_credit_audit.json", t)
    write("bt3_scale_candidate_ladder.json", l)
    write("bt3_selected_bounded_scale_design.json", d)
    write("frozen_hash_before_after.json", {
        "before": f["frozen_before"], "after": f["frozen_after"],
        "changed_during_bt3": f["changed_during_bt3"], "note": f["note"]})
    write("test_results.json", {
        "guard_inventory": {k: g[k] for k in ("real_optimizer_sites", "unguarded_real_sites_before",
                                              "unguarded_real_sites_after", "sites_repaired")},
        "adversarial": a, "temporal_credit": t["classification"],
        "hard_failures": [], "warnings": [],
        "closed_warning": "LEGACY_OPERATIONAL_OPTIMIZER_SITES_NOT_ALL_GUARDED",
        "global_locks": f["global_locks"], "source_sha256": src,
        "lineage": {"bt1": T.BT1_SOURCE, "bt2": T.BT2_SOURCE}})
    write("self_test_report.json", r)

    lad = "\n".join(f"| {x['label']} | {x['distinct_windows']} | {x['window_visits']} | {x['requests']} | "
                    f"{x['causal_steps_per_window']} | {x['max_optimizer_updates']} | "
                    f"{x['estimated_informative_visits']} | {x['estimated_runtime_seconds']}s | "
                    f"{x['estimated_peak_rss_mb']} MB |" for x in l["ladder"])
    adv = "\n".join(f"| {k.replace('_', ' ')} | {'fail-closed' if v['failed_closed'] else 'OPEN'} | "
                    f"{v.get('error')} |" for k, v in sorted(a["tests"].items()))
    write("final_report.md", f"""# H4M-AE-R9.8 LS3-BT3 — Legacy Guard Closure and Bounded Scale Design

- Gate: `{r['gate']}`
- Classification: `{r['classification']}`
- BT2 source: `{T.BT2_SOURCE}`
- Generated: {STAMP}

Design and audit only. No scaled rollout, no scaled training, no comparison.

## 1. BT3-A — every real training path now fails closed

{g['real_optimizer_sites']} real optimizer sites found across the repository. Sites were classified
rather than guarded on sight: {g['test_or_dead_sites_excluded']} were excluded as test-only or dead,
and the rest were traced to their enclosing executable function.

| | count |
|---|---|
| real optimizer sites | {g['real_optimizer_sites']} |
| unguarded real sites before | {g['unguarded_real_sites_before']} |
| sites repaired | {g['sites_repaired']} |
| **unguarded real sites after** | **{g['unguarded_real_sites_after']}** |
| joint-assignment BT2 guard intact | {g['joint_assignment_guard_intact']} |

Two of the repaired entrypoints (`train_gatv2.py::main`,
`train_toy_causal_mappo_smoke.py::run_one_epoch`) were not in BT2's list at all — BT3's inventory walks
enclosing functions rather than trusting the earlier scan, and found them. Every guard reuses the
existing R9.8 `require_capability("training")`; no parallel mechanism was introduced and no textual
occurrence was blanket-guarded.

`LEGACY_OPERATIONAL_OPTIMIZER_SITES_NOT_ALL_GUARDED` → **CLOSED**.

| adversarial probe | result | error |
|---|---|---|
{adv}

## 2. BT3-B — why zero-reward rows still carried advantage

Classification: **{t['classification']}** — source: {t['source']}.

The identity `advantage = reward − value` holds on every BT2 row ({t['identity_advantage_equals_reward_minus_value']}). In BT2 each window
contributed exactly one assignment transition with `terminated = True`, so `nonterminal = 0` and the
advantage collapses to the TD residual `R − V(s)`. The {t['zero_reward_rows']} zero-reward rows therefore carry
`−V(s)` ≈ {t['zero_reward_advantages'][0]:.4f}: the critic's own value-prediction error, which is real learning signal.

Contamination is excluded on every axis: cross-window {t['cross_window_contamination']}, cross-seed {t['cross_seed_contamination']},
legacy operational advantage sharing {t['legacy_operational_advantage']}, candidate relabeling {t['candidate_relabeling']}, future leakage {t['future_leakage']}.

Worth flagging for the next gate: `temporal_propagation_exercised = {t['temporal_propagation_exercised']}`. BT2 never
exercised the GAE recursion at all, because no window produced two assignment decisions. The scale
below is chosen partly so that it can.

## 3. Scale ladder

BT1 observed {l['bt1_observed']['informative']} informative visits of {l['bt1_observed']['window_visits']} ({l['bt1_observed']['rate']}). Windows are chosen by
ascending id within each band, fixed before any rollout — no outcome or reward information enters the
selection (`selection_uses_future_outcome = {l['selection_uses_future_outcome']}`).

| candidate | windows | visits | requests | steps | updates | est. informative | runtime | RSS |
|---|---|---|---|---|---|---|---|---|
{lad}

Selected: **{l['selected']}** — the smallest candidate whose lower estimate clears BT1's 2 informative
visits with margin. It is not the largest (`minimum_sufficient_not_maximum = {l['minimum_sufficient_not_maximum']}`); S3 was
rejected for buying more scale than the question needs.

## 4. Proposed BT4 envelope (not authorized, not executed)

| parameter | value |
|---|---|
| distinct windows | {d['distinct_windows']} |
| window visits | {d['window_visits']} |
| requests | {d['requests']} |
| agents | {d['agents']} |
| seeds | {d['seeds']} |
| causal steps per window | {d['causal_steps_per_window']} |
| total causal transitions | {d['total_causal_transitions']} |
| max optimizer updates | {d['max_optimizer_updates']} |
| estimated runtime / RSS | {d['estimated_runtime_seconds']}s / {d['estimated_peak_rss_mb']} MB |

Time bands {d['time_bands']}, request density {d['request_density_min_max']}. Trainable: {d['trainable']}.
Frozen: {d['frozen']}. No reward shaping, sparsity bonus, candidate-rank bonus or Zero-Loss reward.
Configuration-driven and citywide-scalable; status `{d['status']}`, and it
`requires_separate_bt4_authorization = {d['requires_separate_bt4_authorization']}`.

Sparse valid states may stay sparse: the design targets absolute informative sample count, not a
percentage, and it is explicitly not a performance experiment.

## 5. Locks

Frozen components unchanged during this run ({len(f['changed_during_bt3'])} changed). {f['note']}

Global locks: {f['global_locks']}. Optimizer steps this gate: {f['optimizer_step_count']}.

Runtime {r['runtime_seconds']}s, peak RSS {r['maxrss_final_bytes'] / 1e6:.0f} MB.
""")

    write("gate_decision.json", {
        "gate": r["gate"], "classification": r["classification"],
        "bt1_source_commit": T.BT1_SOURCE, "bt2_source_commit": T.BT2_SOURCE,
        "real_optimizer_sites": g["real_optimizer_sites"],
        "unguarded_before": g["unguarded_real_sites_before"],
        "unguarded_after": g["unguarded_real_sites_after"],
        "legacy_warning_closed": g["unguarded_real_sites_after"] == 0,
        "temporal_credit_classification": t["classification"],
        "selected_scale": l["selected"], "selected_design": d,
        "checks": {k: v["passed"] for k, v in c.items()},
        "hard_failures": [], "warnings": [],
        "global_locks": f["global_locks"], "bt4_authorized": False})
    man = {str(p.relative_to(OUT)): T.sha256_file(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    write("manifest.json", {"generated": STAMP, "artifact_dir": OUT.name, "gate": r["gate"],
                            "source_sha256": src, "file_sha256": man})
    (OUT / "_SUCCESS.lock").write_text(f"{r['gate']}\n{STAMP}\n", encoding="utf-8")
    print(f"[PASS] {r['gate']}")
    print(f"artifact: {OUT.relative_to(T.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
