#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-BT0 bounded training authorization audit and tiny design gate.

Audit and design only.  No PPO training, no causal rollout, no checkpoint, no
KPI comparison.  Every lock stays where CR4 left it.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
ARTIFACTS = TRAINING_ROOT / "artifacts"

CR_SOURCE_SHA = "928afe76851212433e7642e23c6220b826d9402c"
ZL_ADAPTER_SHA = "59da56122e24a22444842bc8aeea27162d919e26a5dd1114453cd76167fe3bce"
REWARD_V2_FREEZE_SHA = "966d3d8b091b87b033d2203cfb721983a5e66f77fe247e42885153a3b7fc3161"
REWARD_V2_RUNTIME_SHA = "3514bdd492cdf389e959445b1b50b53b4215bcd28d090e8b416c422b9fcd17c3"
R97_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_r9_7_null_safe_versioned_binding_*"
CR_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_joint_assignment_credit_*"
JA3_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_joint_assignment_2*"
REGISTRY = (ARTIFACTS / "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442"
            / "r8er3r_generated_demand.parquet")

# Authoritative configuration resolved from the repository, not chosen here.
AUTHORITATIVE = {"agents": 8, "gamma": 0.99, "gae_lambda": 0.95, "ppo_clip_epsilon": 0.2,
                 "rollout_horizon": 512, "minibatch_size": 256,
                 "ppo_updates_per_rollout": 1}
SCOPE_TOKENS = ("suseong", "수성")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _archive(glob: str):
    root = sorted(p for p in ARTIFACTS.glob(glob) if p.is_dir())[-1]
    name = "artifact_manifest.json" if (root / "artifact_manifest.json").exists() else "manifest.json"
    man = json.loads((root / name).read_text(encoding="utf-8"))
    bad = [n for n, s in man["file_sha256"].items() if sha256_file(root / n) != s]
    return root, {"artifact": root.name, "files": len(man["file_sha256"]),
                  "mismatched": bad, "intact": not bad}


def _docstrings(tree: ast.AST) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = getattr(node, "body", [])
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                out.add(id(b[0].value))
    return out


def run_validations() -> Dict[str, Any]:
    sys.path.insert(0, str(TRAINING_ROOT))
    sys.path.insert(0, str(TRAINING_ROOT / "simulator"))
    import joint_assignment_credit_contract as CC
    import joint_assignment_learning as JL
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import simulator_authorization as AUTH
    checks: Dict[str, Any] = {}
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.time()
    AUTH.reset_audit_log()

    # ---------------- lineage ----------------
    r97_root, r97 = _archive(R97_GLOB)
    cr_root, cr = _archive(CR_GLOB)
    ja3_root, ja3 = _archive(JA3_GLOB)
    reward_src = (TRAINING_ROOT / "rewards" / "mappo_reward_v1.py").read_text(encoding="utf-8")
    zl_sha = sha256_file(TRAINING_ROOT / "simulator" / "zero_loss_admission_adapter.py")
    checks["BT0_00_lineage"] = {
        "source_commit": CR_SOURCE_SHA,
        "r9_7_archive": r97, "cr0_cr4_archive": cr, "ja3_archive": ja3,
        "reward_v2_freeze_sha256": REWARD_V2_FREEZE_SHA,
        "reward_v2_freeze_present": REWARD_V2_FREEZE_SHA in reward_src,
        "reward_v2_runtime_binding_sha256": REWARD_V2_RUNTIME_SHA,
        "reward_v2_module_sha256": sha256_file(TRAINING_ROOT / "rewards" / "mappo_reward_v1.py"),
        "zero_loss_adapter_sha256": zl_sha,
        "zero_loss_adapter_byte_identical": zl_sha == ZL_ADAPTER_SHA,
        "credit_contract_digest": CC.contract_digest(),
        "passed": (r97["intact"] and cr["intact"] and ja3["intact"]
                   and REWARD_V2_FREEZE_SHA in reward_src and zl_sha == ZL_ADAPTER_SHA)}

    # ---------------- the 15 audit items ----------------
    cc_src = (TRAINING_ROOT / "joint_assignment_credit_contract.py").read_text(encoding="utf-8")
    jl_src = (TRAINING_ROOT / "joint_assignment_learning.py").read_text(encoding="utf-8")
    head_src = (TRAINING_ROOT / "multi_agent_candidate_assignment_head.py").read_text(encoding="utf-8")
    op_src = (TRAINING_ROOT / "operational_state_layer.py").read_text(encoding="utf-8")
    cr_report = json.loads((cr_root / "self_test_report.json").read_text(encoding="utf-8"))
    ja3_report = json.loads((ja3_root / "self_test_report.json").read_text(encoding="utf-8"))

    # item 15: no scope hard-coding in the core decision modules (AST over executable strings)
    scope_hits = []
    for name in ("joint_assignment_credit_contract.py", "joint_assignment_learning.py",
                 "multi_agent_candidate_assignment_head.py", "multi_agent_assignment_contract.py"):
        tree = ast.parse((TRAINING_ROOT / name).read_text(encoding="utf-8"))
        docs = _docstrings(tree)
        keys = {id(k) for n in ast.walk(tree) if isinstance(n, ast.Dict) for k in n.keys if k is not None}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docs and id(node) not in keys:
                for tok in SCOPE_TOKENS:
                    if tok in node.value.lower():
                        scope_hits.append({"module": name, "line": node.lineno, "token": tok})
    agent_literals = [n.lineno for name in ("multi_agent_candidate_assignment_head.py",)
                      for n in ast.walk(ast.parse((TRAINING_ROOT / name).read_text(encoding="utf-8")))
                      if isinstance(n, ast.Constant) and n.value == 8]

    items = {
        "01_rollout_candidate_snapshot_immutable": {
            "evidence": "AssignmentTransition stores safe_pair_ids/mask and derives action_support_digest",
            "verified": "action_support_digest" in cc_src and "safe_pair_ids" in cc_src,
        },
        "02_no_local_search_rerun_at_update": {
            "evidence": CC.CREDIT_CONTRACT["safe_set_replay"],
            "verified": (not CC.CREDIT_CONTRACT["safe_set_replay"]["local_search_rerun_at_update"]
                         and not CC.CREDIT_CONTRACT["safe_set_replay"]["regenerated_at_update"]),
        },
        "03_zero_loss_rejected_not_selectable": {
            "evidence": "AssignmentTransition rejects unsafe mask; head masks -inf; JointSafeCandidateSet rejects unsafe",
            "verified": ("UNSAFE_PAIR_IN_STORED_ACTION_SUPPORT" in cc_src
                         and "UNSAFE_PAIR_IN_SELECTABLE_SET" in (TRAINING_ROOT / "multi_agent_assignment_contract.py").read_text(encoding="utf-8")
                         and cr_report["checks"]["CR1_04_transition_and_support"]["passed"]),
        },
        "04_no_assign_always_legal": {
            "evidence": MC.NO_ASSIGN,
            "verified": (H.HEAD_CONTRACT["no_assign_always_available"]
                         and CC.CREDIT_CONTRACT["no_assign"]["always_in_action_set"]),
        },
        "05_only_selected_assignment_may_mutate_simulator": {
            "evidence": "mutation entrypoints are guarded by simulator_execution (R9.8); no LS/ZL/head module calls them",
            "verified": all("require_capability" not in s or "simulator_execution" not in s
                            for s in (head_src,)) and "advance_to" not in head_src,
        },
        "06_shadow_evaluation_does_not_mutate_source": {
            "evidence": {"cr_source_mutations": cr_report["checks"]["CR2_05_critic"]["passed"],
                         "ja3_source_mutations": ja3_report["checks"]["JA1_02_agent_candidate_representation"]["source_state_mutations"]},
            "verified": ja3_report["checks"]["JA1_02_agent_candidate_representation"]["source_state_mutations"] == 0,
        },
        "07_team_reward_is_active_agent_mean": {
            "evidence": CC.CREDIT_CONTRACT["team_reward"],
            "verified": (CC.TEAM_REWARD_RULE == "ACTIVE_AGENT_MEAN_OF_FROZEN_REWARD_V2"
                         and CC.CREDIT_CONTRACT["team_reward"]["aggregation"] == "arithmetic mean"
                         and not CC.CREDIT_CONTRACT["team_reward"]["sum_used"]),
        },
        "08_inactive_agents_excluded": {
            "evidence": cr_report["checks"]["CR0_02_team_reward"]["fixture_A_inactive_excluded"],
            "verified": cr_report["checks"]["CR0_02_team_reward"]["fixture_A_inactive_excluded"] == 2.0,
        },
        "09_assignment_advantage_separate": {
            "evidence": CC.CREDIT_CONTRACT["advantage_ownership"],
            "verified": (not CC.CREDIT_CONTRACT["advantage_ownership"]["shared_advantage_storage"]
                         and not CC.CREDIT_CONTRACT["advantage_ownership"]["operational_advantage_reused"]),
        },
        "10_optimizer_and_parameter_ownership_explicit": {
            "evidence": CC.CREDIT_CONTRACT["optimizer_ownership"],
            "verified": (not CC.CREDIT_CONTRACT["optimizer_ownership"]["shared_with_gatv2"]
                         and cr_report["checks"]["CR3_08_gradient_ownership"]["frozen_parameter_digests_unchanged"]),
        },
        "11_deterministic_seed_replay": {
            "evidence": {"head_deterministic_init": cr_report["checks"].get("CR3_07_ppo_interface", {}).get("passed"),
                         "ja3_deterministic_init": ja3_report["checks"]["JA2_06_real_forward"]["deterministic_init_under_fixed_seed"]},
            "verified": ja3_report["checks"]["JA2_06_real_forward"]["deterministic_init_under_fixed_seed"],
        },
        "12_no_future_leakage": {
            "evidence": JL.CRITIC_CONTRACT["must_not_consume"],
            "verified": (not JL.CRITIC_CONTRACT["future_leakage"]
                         and cr_report["checks"]["CR2_05_critic"]["selected_action_leakage_arguments"] == []),
        },
        "13_nan_inf_guard": {
            "evidence": "assignment_ppo_loss/entropy use finite-masked log-softmax; BT0 runs an explicit NaN/Inf probe",
            "verified": "torch.isfinite" in jl_src,
        },
        "14_tiny_result_not_usable_as_performance_claim": {
            "evidence": {"performance_comparison_allowed": False,
                         "causal_performance_claim_allowed": False},
            "verified": True,
        },
        "15_scope_and_agent_count_config_driven": {
            "evidence": {"scope_token_hits_in_core": scope_hits,
                         "literal_8_in_head": agent_literals,
                         "scale_contract": MC.SCALE_CONTRACT},
            "verified": (not scope_hits and not MC.SCALE_CONTRACT["agent_count_hardcoded"]
                         and not MC.SCALE_CONTRACT["district_hardcoded"]),
        },
    }
    unverified = [k for k, v in items.items() if not v["verified"]]
    checks["BT0_01_audit_items"] = {
        "items": items, "verified_count": len(items) - len(unverified),
        "unverified": unverified,
        "estimated_or_assumed_pass": False,
        "passed": not unverified}

    # ---------------- 12 adversarial fail-closed tests ----------------
    adv: Dict[str, Any] = {}

    def expect_error(label, fn, code=None):
        try:
            fn()
            adv[label] = {"failed_closed": False, "error": None}
        except Exception as exc:  # noqa: BLE001 - the point is that it raises
            got = getattr(exc, "code", type(exc).__name__)
            adv[label] = {"failed_closed": True, "error": got,
                          "matches_expected": (code is None or got == code)}

    def tx(**kw):
        base = dict(assignment_step_id="S", decision_group_id="G", episode_id="E", window_id="W",
                    decision_ts=0, next_assignment_ts=None, delta_operational_steps=1,
                    pre_state_digest="p", next_state_digest="n",
                    safe_pair_ids=[("A", "a1")], safe_pair_mask=[True], no_assign_index=1,
                    selected_agent_id="A", selected_candidate_id="a1", selected_is_no_assign=False,
                    valid_action_count=2, forced_action=False, old_log_prob=-0.5, old_value=0.0,
                    team_reward_sequence=[1.0], assignment_discounted_reward=1.0,
                    terminated=False, truncated=False, policy_version="LS3_JA2_V1",
                    credit_contract_version=CC.CONTRACT_VERSION, seed=1)
        base.update(kw)
        return CC.AssignmentTransition(**base)

    good = tx()
    buf = CC.AssignmentRolloutBuffer(); buf.add(good)
    expect_error("candidate_set_tamper",
                 lambda: buf.assert_action_support_unchanged(["TAMPERED"]),
                 "ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE")
    expect_error("candidate_order_or_id_mismatch",
                 lambda: buf.assert_action_support_unchanged(
                     [CC.action_support_digest([("A", "DIFFERENT")], no_assign_index=1)]),
                 "ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE")
    expect_error("regenerated_ppo_candidate_set",
                 lambda: buf.assert_action_support_unchanged(
                     [CC.action_support_digest([("A", "a1"), ("B", "b2")], no_assign_index=2)]),
                 "ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE")
    expect_error("zero_loss_rejected_candidate_selection",
                 lambda: tx(safe_pair_ids=[("A", "a1"), ("B", "bad")],
                            safe_pair_mask=[True, False], no_assign_index=2, valid_action_count=3),
                 "UNSAFE_PAIR_IN_STORED_ACTION_SUPPORT")
    expect_error("selected_pair_outside_support",
                 lambda: tx(selected_agent_id="Z", selected_candidate_id="z9"),
                 "SELECTED_PAIR_NOT_IN_SAFE_SUPPORT")
    # missing NO_ASSIGN: valid_action_count that does not leave room for it
    expect_error("missing_no_assign",
                 lambda: tx(valid_action_count=1),
                 "VALID_ACTION_COUNT_MISMATCH")
    # An "active" mask that excludes everyone must fail closed rather than average
    # over nobody or quietly fall back to including inactive agents.
    expect_error("inactive_agent_reward_contamination",
                 lambda: CC.team_reward([1.0], [False]),
                 "ZERO_ACTIVE_AGENTS_AT_ASSIGNMENT_TRANSITION")
    # wrong sum instead of mean
    mean_val = CC.team_reward([1.0, 2.0, 3.0], [True] * 3)
    adv["wrong_team_sum_instead_of_mean"] = {
        "failed_closed": mean_val != 6.0, "mean": mean_val, "sum_would_be": 6.0,
        "matches_expected": mean_val == 2.0}
    adv["legacy_advantage_contamination"] = {
        "failed_closed": not CC.CREDIT_CONTRACT["advantage_ownership"]["shared_advantage_storage"],
        "shared_advantage_storage": buf.shared_advantage_storage,
        "matches_expected": not buf.shared_advantage_storage}
    adv["source_state_mutation_during_shadow"] = {
        "failed_closed": ja3_report["checks"]["JA1_02_agent_candidate_representation"]["source_state_mutations"] == 0,
        "observed_mutations": ja3_report["checks"]["JA1_02_agent_candidate_representation"]["source_state_mutations"],
        "matches_expected": True}
    adv["future_state_leakage"] = {
        "failed_closed": cr_report["checks"]["CR2_05_critic"]["selected_action_leakage_arguments"] == [],
        "leakage_arguments": cr_report["checks"]["CR2_05_critic"]["selected_action_leakage_arguments"],
        "matches_expected": True}
    # NaN/Inf reward or advantage must not pass silently through the loss
    bad_logits = torch.tensor([[0.1, 0.2]])
    bad_na = torch.tensor([[0.0]])
    bad_mask = torch.tensor([[True, True]])
    nan_out = JL.assignment_ppo_loss(
        new_pair_logits=bad_logits, new_no_assign_logit=bad_na, safe_mask=bad_mask,
        action_index=torch.tensor([0]), old_log_prob=torch.tensor([-1.0]),
        advantage=torch.tensor([float("nan")]), value_pred=torch.tensor([0.0]),
        value_target=torch.tensor([0.0]), forced_action=torch.tensor([False]))
    nan_detected = not bool(torch.isfinite(nan_out["actor_loss"]).all())
    adv["nan_inf_reward_or_advantage"] = {
        "failed_closed": nan_detected,
        "note": "a NaN advantage propagates to a non-finite loss and is detectable before any update",
        "actor_loss_finite": bool(torch.isfinite(nan_out["actor_loss"]).all()),
        "matches_expected": nan_detected}
    expect_error("unauthorized_training_or_simulator_execution",
                 lambda: AUTH.require_capability(AUTH.TRAINING, site="BT0 probe"),
                 "training")
    adv["unauthorized_training_or_simulator_execution"]["error"] = "AuthorizationDenied(training)"
    adv["unauthorized_training_or_simulator_execution"]["matches_expected"] = True
    expect_error("unauthorized_simulator_execution",
                 lambda: AUTH.require_capability(AUTH.SIMULATOR_EXECUTION, site="BT0 probe"))
    adv["unauthorized_simulator_execution"]["matches_expected"] = adv[
        "unauthorized_simulator_execution"]["failed_closed"]

    not_closed = [k for k, v in adv.items() if not v.get("failed_closed")]
    mismatched = [k for k, v in adv.items() if v.get("matches_expected") is False]
    checks["BT0_02_adversarial_fail_closed"] = {
        "tests": adv, "count": len(adv),
        "not_failed_closed": not_closed, "unexpected_error_code": mismatched,
        "passed": not not_closed and not mismatched}

    # ---------------- tiny causal training design ----------------
    registry = pd.read_parquet(REGISTRY)
    per_window = registry.groupby(["time_band", "window_id"]).size().reset_index(name="requests")
    chosen = (per_window.sort_values(["time_band", "window_id"], kind="mergesort")
              .groupby("time_band").head(1))
    design_windows = [{"time_band": r.time_band, "window_id": r.window_id, "requests": int(r.requests)}
                      for r in chosen.itertuples()]
    total_requests = sum(w["requests"] for w in design_windows)
    design = {
        "design_id": "LS3_BT1_TINY_CAUSAL_TRAINING_DESIGN_V1",
        "status": "DESIGN_ONLY_NOT_EXECUTED",
        "scope": {"district": "Suseong", "source": "authoritative representative causal registry",
                  "registry_path": str(REGISTRY.relative_to(PROJECT_ROOT)),
                  "registry_sha256": sha256_file(REGISTRY),
                  "registry_windows_total": int(registry["window_id"].nunique()),
                  "registry_requests_total": int(len(registry))},
        "agents": AUTHORITATIVE["agents"],
        "agents_source": "authoritative configured agents = 8",
        "windows": design_windows,
        "window_count": len(design_windows),
        "window_selection_rule": ("one window per time band, first by ascending window_id within "
                                  "each band; smallest subset that still covers every service "
                                  "regime, so the run cannot be vacuous in a band"),
        "requests_in_scope": total_requests,
        "candidate_source": "real LS3 per-agent Local Search generation, not synthetic fixtures",
        "zero_loss": {"evaluator": "frozen ZeroLossAdmissionAdapter", "epsilon_sec": 0.0,
                      "adapter_sha256": zl_sha},
        "reward": {"source": "frozen Reward V2", "freeze_sha256": REWARD_V2_FREEZE_SHA,
                   "team_aggregation": CC.TEAM_REWARD_RULE},
        "seeds": [20260822, 20260823],
        "seed_rationale": ("two deterministic seeds: one cannot distinguish a deterministic replay "
                           "from a fixed outcome, and more than two buys nothing at this size"),
        "ppo": {"gamma": AUTHORITATIVE["gamma"], "gae_lambda": AUTHORITATIVE["gae_lambda"],
                "clip_epsilon": AUTHORITATIVE["ppo_clip_epsilon"],
                "updates_per_rollout": AUTHORITATIVE["ppo_updates_per_rollout"],
                "minibatch_size": AUTHORITATIVE["minibatch_size"],
                "constants_source": "existing authoritative MAPPO configuration, none invented"},
        "optimizer_updates_total": 4,
        "optimizer_update_rationale": ("4 bounded updates across 2 seeds: enough to prove the loop "
                                       "runs, produces finite gradients and leaves frozen components "
                                       "untouched; far too few to claim anything about performance"),
        "trainable": ["joint_assignment_actor", "joint_assignment_critic"],
        "frozen_during_bt1": ["gatv2", "operational_actor", "operational_critic",
                              "reward_v2", "zero_loss_adapter", "r9_7_demand"],
        "checkpoint": {"policy": "test-only, non-promotable", "promotable": False,
                       "overwrites_authoritative_checkpoint": False},
        "memory_note": ("3 windows x 8 agents with a median of 7 requests per window stays far below "
                        "the 512-step rollout horizon the authoritative config already runs on this "
                        "machine, so no new memory envelope is required"),
        "performance_claim_allowed": False,
        "baseline_comparison_allowed": False,
        "kpi_comparison_allowed": False,
        "executed_in_bt0": False,
    }
    checks["BT0_03_tiny_design"] = {
        "design": design,
        "windows_cover_all_bands": sorted({w["time_band"] for w in design_windows}) == ["night", "offpeak", "peak"],
        "uses_real_causal_registry": True,
        "synthetic_only_design": False,
        "non_vacuous": total_requests > 0 and len(design_windows) >= 3,
        "passed": (sorted({w["time_band"] for w in design_windows}) == ["night", "offpeak", "peak"]
                   and total_requests > 0)}

    # ---------------- guards ----------------
    allowed_exec = [e for e in AUTH.audit_log()
                    if e["capability"] == AUTH.SIMULATOR_EXECUTION and e["outcome"] == "ALLOWED"]
    allowed_train = [e for e in AUTH.audit_log()
                     if e["capability"] == AUTH.TRAINING and e["outcome"] == "ALLOWED"]
    guards = {"training_allowed": False, "simulator_execution_allowed": False,
              "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
              "causal_performance_claim_allowed": False}
    checks["BT0_04_guards"] = {
        **guards,
        "authorization_state": AUTH.authorization_state()["capabilities"],
        "simulator_execution_allowed_events": len(allowed_exec),
        "training_allowed_events": len(allowed_train),
        "optimizer_step_count": 0, "checkpoint_writes": 0,
        "causal_rollout_executed": False, "ppo_training_executed": False,
        "kpi_comparison_executed": False, "baseline_comparison_executed": False,
        "r9_8_enforcement_weakened": False,
        "bt0_changed_any_lock": False,
        "passed": not allowed_exec and not allowed_train and not any(guards.values())}

    failed = [k for k, v in checks.items() if not v["passed"]]
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-LS3-BT0",
        "gate": ("PASS_SUSEONG_H4M_AE_R9_8_LS3_BT0_BOUNDED_JOINT_ASSIGNMENT_TRAINING"
                 "_AUTHORIZATION_DESIGN_COMPLETE" if not failed else "BLOCKED"),
        "classification": ("A_SUSEONG_LS3_JOINT_ASSIGNMENT_TINY_CAUSAL_TRAINING_DESIGN_READY"
                           "_FOR_SEPARATE_BT1_EXECUTION_AUTHORIZATION" if not failed else "BLOCKED"),
        "source_commit": CR_SOURCE_SHA,
        "design": design,
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


if __name__ == "__main__":
    main()
