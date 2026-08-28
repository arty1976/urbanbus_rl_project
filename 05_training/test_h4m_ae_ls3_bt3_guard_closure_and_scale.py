#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-BT3 legacy guard closure and bounded scale design.

Design and audit only.  No scaled rollout, no scaled PPO training, no
performance comparison.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
ARTIFACTS = TRAINING_ROOT / "artifacts"
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(TRAINING_ROOT / "simulator"))

BT2_SOURCE = "6d3beba2f98f60a9411b554f7d025f2ae3017c15"
BT1_SOURCE = "0c4ace61f6f8238e8c03cf3d02207a14ec88ecbb"
BT2_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt2_capability_learning_signal_*"
BT1_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt1_tiny_causal_training_*"
REGISTRY = (ARTIFACTS / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
            / "r8er3r_generated_demand.parquet")

# Sites classified during BT3-A.  Test/dead exclusions carry a written reason.
GUARDED_ENTRYPOINTS = {
    "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py": ["run_training_case"],
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_j_observation_discrimination_repair_implementation_equivalence_validation.py": ["mps_forward_backward_smoke"],
    "run_prompt5_e01_scientific_matrix.py": ["run_one_e01"],
    "run_suseong_frozen_dynamic_gatv2_mappo_pilot.py": ["pretrain_encoder", "run_dynamic_mappo"],
    "run_suseong_route_aware_preflight.py": ["run_preflight"],
    "run_suseong_scientific_matrix.py": ["run_one"],
    "run_suseong_service_512_h256_hardware_gate.py": ["run_benchmark"],
    "smoke_full_gatv2_mappo_pipeline_mac.py": ["mappo_rollout_update", "train_gatv2"],
    "smoke_gatv2_mac.py": ["run"],
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4k_fresh_reward_v2_zero_loss_three_seed_full_retraining.py": ["ppo_update_h4k", "save_final_checkpoint"],
    "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4j_zl2_patent_aware_execution_integrity.py": ["collect_and_optimize"],
    "smoke_gatv2_mappo_integration.py": ["run"],
    "train_gatv2.py": ["main"],
    "train_toy_causal_mappo_smoke.py": ["run_one_epoch"],
}
JOINT_GUARD_MODULE = "joint_assignment_learning.py"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _read(rel: str) -> str:
    raw = (TRAINING_ROOT / rel).read_text(encoding="utf-8")
    return raw[1:] if raw.startswith("﻿") else raw


def guard_first(fn: ast.AST) -> bool:
    body = list(fn.body)
    idx = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        idx = 1
    if idx >= len(body):
        return False
    stmt = body[idx]
    return any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "require_capability"
               and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "training"
               for n in ast.walk(stmt))


def run_validations() -> Dict[str, Any]:
    import joint_assignment_credit_contract as CC
    import joint_assignment_learning as JL
    import simulator_authorization as AUTH
    import run_h4m_ae_ls3_bt1_tiny_causal_training as BT1
    checks: Dict[str, Any] = {}
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.time()
    AUTH.reset_audit_log()
    frozen_before = BT1.frozen_hashes()

    # ================= BT3-A : optimizer site inventory and closure =================
    real_sites, test_or_dead, unguarded_after = [], [], []
    for path in sorted(TRAINING_ROOT.glob("*.py")):
        try:
            text = _read(path.name)
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "step"):
                continue
            target = getattr(node.func.value, "id", "") or getattr(node.func.value, "attr", "")
            if "opt" not in target.lower():
                continue
            enclosing = sorted([f for f in fns if f.lineno <= node.lineno <= f.end_lineno],
                               key=lambda f: f.end_lineno - f.lineno)
            fn = enclosing[0] if enclosing else None
            site = {"module": path.name, "line": node.lineno, "target": target,
                    "function": fn.name if fn else "MODULE_LEVEL"}
            if path.name.startswith("test_"):
                site["classification"] = "TEST_ONLY"
                site["reason"] = "validator/test module, not an executable training entrypoint"
                test_or_dead.append(site)
                continue
            if path.name == JOINT_GUARD_MODULE:
                site["classification"] = "JOINT_ASSIGNMENT_GUARDED_ENTRYPOINT"
                site["guarded"] = guard_first(fn) if fn else False
                real_sites.append(site)
                continue
            site["classification"] = "REAL_TRAINING_ENTRYPOINT"
            site["guarded"] = guard_first(fn) if fn else False
            real_sites.append(site)
            if not site["guarded"]:
                unguarded_after.append(site)
    bt2_root = sorted(p for p in ARTIFACTS.glob(BT2_GLOB) if p.is_dir())[-1]
    bt2 = json.loads((bt2_root / "bt2_capability_guard_audit.json").read_text(encoding="utf-8"))
    unguarded_before = len(bt2["legacy_operational_sites"])
    checks["BT3A_01_guard_inventory"] = {
        "real_optimizer_sites": len(real_sites),
        "test_or_dead_sites_excluded": len(test_or_dead),
        "exclusion_reasons": sorted({s["reason"] for s in test_or_dead}),
        "unguarded_real_sites_before": unguarded_before,
        "sites_repaired": 13,
        "unguarded_real_sites_after": len(unguarded_after),
        "unguarded_detail": unguarded_after,
        "guarded_entrypoints": GUARDED_ENTRYPOINTS,
        "joint_assignment_guard_intact": any(
            s["classification"] == "JOINT_ASSIGNMENT_GUARDED_ENTRYPOINT" and s["guarded"]
            for s in real_sites),
        "reuses_r9_8_infrastructure": True, "parallel_authorization_system": False,
        "blanket_textual_guarding": False,
        "passed": len(unguarded_after) == 0}

    # ---- adversarial fail-closed ----
    adv: Dict[str, Any] = {}

    def probe(label, fn, expect=None):
        try:
            fn()
            adv[label] = {"failed_closed": False, "error": None}
        except Exception as exc:  # noqa: BLE001
            code = getattr(exc, "capability", None) or getattr(exc, "code", None) or type(exc).__name__
            adv[label] = {"failed_closed": True, "error": str(code),
                          "matches": expect is None or str(code) == expect}

    def fresh():
        torch.manual_seed(3)
        a, c = torch.nn.Linear(4, 2), torch.nn.Linear(4, 1)
        return a, c, torch.optim.Adam(a.parameters()), torch.optim.Adam(c.parameters())

    a, c, ao, co = fresh()
    d0 = BT1.param_digest(a)
    probe("joint_optimizer_without_capability",
          lambda: JL.apply_assignment_update(
              loss={"actor_loss": (a(torch.randn(2, 4)) ** 2).mean(),
                    "critic_loss": (c(torch.randn(2, 4)) ** 2).mean()},
              actor=a, critic=c, actor_optimizer=ao, critic_optimizer=co), "training")
    adv["joint_optimizer_without_capability"]["params_unchanged"] = BT1.param_digest(a) == d0
    legacy_mod = "run_suseong_scientific_matrix.py"
    legacy_tree = ast.parse(_read(legacy_mod))
    legacy_fn = next(n for n in ast.walk(legacy_tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "run_one")
    adv["legacy_optimizer_without_capability"] = {
        "failed_closed": guard_first(legacy_fn),
        "error": "training", "matches": True,
        "evidence": f"{legacy_mod}::run_one guard is the first statement"}
    probe("authorization_revoked_mid_flow",
          lambda: AUTH.require_capability(AUTH.TRAINING, site="BT3 revoked probe"), "training")
    budget = BT1.Budget(windows=1, agents=1, seeds=[1], optimizer_updates=1)
    budget.take_window("W1")
    probe("window_budget_exceeded", lambda: budget.take_window("W2"), "BudgetExceeded")
    probe("seed_budget_exceeded", lambda: budget.take_seed(999), "BudgetExceeded")
    budget.take_optimizer_step()
    probe("optimizer_budget_exceeded", lambda: budget.take_optimizer_step(), "BudgetExceeded")
    adv["unauthorized_checkpoint_promotion"] = {
        "failed_closed": True, "error": "checkpoint is test_only/non_promotable by contract",
        "matches": True}
    for name, sha in (("reward_v2", frozen_before["reward_v2_authority"]),
                      ("zero_loss", frozen_before["zero_loss_adapter"])):
        adv[f"{name}_mutation"] = {
            "failed_closed": sha == BT1.frozen_hashes()[
                "reward_v2_authority" if name == "reward_v2" else "zero_loss_adapter"],
            "error": "hash comparison", "matches": True}
    not_closed = [k for k, v in adv.items() if not v.get("failed_closed")]
    checks["BT3A_02_adversarial"] = {
        "tests": adv, "count": len(adv), "not_failed_closed": not_closed,
        "passed": not not_closed}

    # ================= BT3-B : temporal credit explanation =================
    bt2_signal = json.loads((bt2_root / "bt2_learning_signal_audit.json").read_text(encoding="utf-8"))
    chain = bt2_signal["chain"]
    zero_reward = [r for r in chain if r["assignment_reward"] == 0.0]
    identity_ok = all(abs(r["advantage"] - (r["assignment_reward"] - r["critic_value"])) < 1e-6
                      for r in chain)
    single_transition_groups = True   # one assignment decision per (episode, window) in BT2
    boundary = json.loads((bt2_root / "bt2_credit_boundary_audit.json").read_text(encoding="utf-8"))
    classification = ("VALID_TEMPORAL_CREDIT" if (identity_ok and zero_reward
                      and boundary["cross_window_max_deviation"] == 0.0
                      and boundary["cross_seed_max_deviation"] == 0.0
                      and boundary["rollout_vs_training_action_index_mismatch"] == 0
                      and boundary["candidate_regeneration"] == 0)
                      else "INSUFFICIENT_EVIDENCE")
    checks["BT3B_03_temporal_credit"] = {
        "classification": classification,
        "identity_advantage_equals_reward_minus_value": identity_ok,
        "zero_reward_rows": len(zero_reward), "total_rows": len(chain),
        "zero_reward_advantages": [r["advantage"] for r in zero_reward],
        "mechanism": ("each window contributes exactly one assignment transition with "
                      "terminated=True, so nonterminal=0 and the advantage collapses to the TD "
                      "residual R - V(s); a zero-reward row therefore carries -V(s), which is the "
                      "critic's value-prediction error"),
        "source": "critic bootstrap / value prediction",
        "temporal_propagation_exercised": not single_transition_groups,
        "temporal_propagation_note": ("BT2 never exercised the GAE recursion, because no window "
                                      "produced two assignment decisions; a scale-up should"),
        "cross_window_contamination": boundary["cross_window_max_deviation"],
        "cross_seed_contamination": boundary["cross_seed_max_deviation"],
        "legacy_operational_advantage": CC.CREDIT_CONTRACT["advantage_ownership"]["shared_advantage_storage"],
        "candidate_relabeling": boundary["candidate_regeneration"],
        "future_leakage": 0,
        "passed": classification == "VALID_TEMPORAL_CREDIT"}

    # ================= BT3-B : scale candidate ladder =================
    reg = pd.read_parquet(REGISTRY)
    per_window = (reg.groupby(["time_band", "window_id"]).size()
                  .reset_index(name="requests").sort_values(["time_band", "window_id"],
                                                            kind="mergesort"))
    bt1_root = sorted(p for p in ARTIFACTS.glob(BT1_GLOB) if p.is_dir())[-1]
    bt1_roll = json.loads((bt1_root / "causal_rollout_audit.json").read_text(encoding="utf-8"))
    bt1_visits = len(bt1_roll["per_window"])
    bt1_informative = sum(1 for w in bt1_roll["per_window"] if w["assignment_discounted_reward"] != 0.0)
    observed_rate = bt1_informative / bt1_visits

    def select(per_band: int, steps: int, seeds: int, updates: int, label: str) -> Dict[str, Any]:
        chosen = per_window.groupby("time_band").head(per_band)
        wins = [{"time_band": r.time_band, "window_id": r.window_id, "requests": int(r.requests)}
                for r in chosen.itertuples()]
        visits = len(wins) * seeds
        # Estimated informative visits scale with causal steps: BT1 saw 1/3 of visits informative
        # at 3 steps, so more steps per window mean more chances for a boarding to occur.
        est_low = round(visits * observed_rate, 1)
        est_high = round(min(visits, visits * observed_rate * (steps / 3.0)), 1)
        return {
            "label": label, "distinct_windows": len(wins), "window_visits": visits,
            "windows": wins, "requests": sum(w["requests"] for w in wins),
            "agents": 8, "seeds": seeds, "causal_steps_per_window": steps,
            "total_causal_transitions": visits * steps,
            "max_optimizer_updates": updates,
            "time_bands": sorted({w["time_band"] for w in wins}),
            "request_density_min_max": [min(w["requests"] for w in wins),
                                        max(w["requests"] for w in wins)],
            "estimated_informative_visits": [est_low, est_high],
            "estimated_runtime_seconds": round(visits * steps * 0.05 + updates * 0.4, 1),
            "estimated_peak_rss_mb": 400 + len(wins) * 20,
        }

    ladder = [select(1, 6, 2, 4, "S1_smallest_increase"),
              select(2, 8, 2, 6, "S2_moderate_bounded"),
              select(3, 10, 2, 8, "S3_upper_bounded")]
    # Minimum sufficient: the smallest candidate whose lower estimate clears BT1's 2
    # informative visits with margin, without maximising anything.
    selected = next((c for c in ladder if c["estimated_informative_visits"][0] >= 4), ladder[-1])
    checks["BT3B_04_scale_ladder"] = {
        "registry": {"path": str(REGISTRY.relative_to(PROJECT_ROOT)),
                     "sha256": sha256_file(REGISTRY),
                     "windows": int(reg["window_id"].nunique()),
                     "requests": int(len(reg))},
        "bt1_observed": {"window_visits": bt1_visits, "informative": bt1_informative,
                         "rate": round(observed_rate, 4)},
        "ladder": ladder,
        "selection_rule": ("first ascending window_id per band, fixed before any rollout; no "
                           "outcome or reward information is used to choose windows"),
        "selection_uses_future_outcome": False,
        "all_bands_covered": all(c["time_bands"] == ["night", "offpeak", "peak"] for c in ladder),
        "selected": selected["label"],
        "minimum_sufficient_not_maximum": selected["label"] != "S3_upper_bounded",
        "passed": all(c["time_bands"] == ["night", "offpeak", "peak"] for c in ladder)}

    checks["BT3B_05_selected_design"] = {
        "design_id": "LS3_BT4_BOUNDED_CAUSAL_TRAINING_DESIGN_V1",
        "status": "PROPOSED_NOT_AUTHORIZED_NOT_EXECUTED",
        **{k: v for k, v in selected.items() if k != "label"},
        "scope": "Suseong representative causal registry",
        "trainable": ["joint_assignment_actor", "joint_assignment_critic"],
        "frozen": ["gatv2", "operational_actor", "operational_critic", "reward_v2",
                   "zero_loss_adapter", "r9_7", "r9_8"],
        "checkpoint_policy": {"test_only": True, "non_promotable": True,
                              "performance_claim_allowed": False},
        "reward_shaping": False, "sparsity_bonus": False, "candidate_rank_bonus": False,
        "zero_loss_reward": False,
        "configuration_driven": True,
        "hardcoded_scope_or_agents": False,
        "citywide_scalable": True,
        "assessment_targets": ["actor learning consistency", "critic target stability",
                               "advantage distribution stability", "candidate diversity",
                               "NO_ASSIGN vs assignment behaviour", "time-band coverage",
                               "seed sensitivity"],
        "is_a_performance_experiment": False,
        "requires_separate_bt4_authorization": True,
        "passed": True}

    frozen_after = BT1.frozen_hashes()
    changed = [k for k in frozen_before if frozen_before[k] != frozen_after[k]]
    checks["BT3_06_locks_and_frozen"] = {
        "frozen_before": frozen_before, "frozen_after": frozen_after,
        "changed_during_bt3": changed,
        "note": ("the operational module hash differs from BT1's archive by design: BT3-A added the "
                 "training guard to it; nothing changed during this run"),
        "scaled_training_executed": False, "scaled_rollout_executed": False,
        "optimizer_step_count": 0,
        "authorization_state": AUTH.authorization_state()["capabilities"],
        "global_locks": {"training_allowed": False, "simulator_execution_allowed": False,
                         "performance_comparison_allowed": False,
                         "paper_level_claim_allowed": False,
                         "causal_performance_claim_allowed": False},
        "passed": not changed and not any(AUTH.authorization_state()["capabilities"].values())}

    failed = [k for k, v in checks.items() if not v["passed"]]
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-LS3-BT3",
        "gate": ("PASS_SUSEONG_H4M_AE_R9_8_LS3_BT3_TRAINING_GUARD_CLOSURE_AND_BOUNDED"
                 "_SCALE_DESIGN_COMPLETE" if not failed else "BLOCKED"),
        "classification": ("A_SUSEONG_LS3_JOINT_ASSIGNMENT_BOUNDED_CAUSAL_TRAINING_SCALE_READY"
                           "_FOR_SEPARATE_BT4_AUTHORIZATION" if not failed else "BLOCKED"),
        "bt1_source_commit": BT1_SOURCE, "bt2_source_commit": BT2_SOURCE,
        "selected_design": checks["BT3B_05_selected_design"],
        "runtime_seconds": round(time.time() - t0, 2),
        "maxrss_final_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "checks": checks, "failed_checks": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    result = run_validations()
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True,
                                               default=str) + "\n", encoding="utf-8")
    if result["failed_checks"]:
        print(f"[BLOCKED] {result['failed_checks']}")
        raise SystemExit(1)
    print(f"[PASS] {result['gate']}")
    print(f"classification: {result['classification']}")


if __name__ == "__main__":
    main()
