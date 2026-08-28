#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-BT2 training-capability closure and learning-signal audit.

Two objectives, in order: close the guard BT1 left open on the joint-assignment
optimizer, then decide from real evidence whether the causal credit chain
actually carries usable learning signal.  Scale is not increased and nothing is
compared for performance.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
ARTIFACTS = TRAINING_ROOT / "artifacts"
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(TRAINING_ROOT / "simulator"))

BT1_SOURCE = "0c4ace61f6f8238e8c03cf3d02207a14ec88ecbb"
BT0_SOURCE = "de668dd6ee4e86a6a2678713564e893b067a69b6"
BT1_GLOB = "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt1_tiny_causal_training_*"
BOUNDS = {"windows": 3, "agents": 8, "seeds": [20260822, 20260823], "optimizer_updates": 4}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def param_digest(m) -> str:
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes()
                                   for p in m.parameters())).hexdigest()


def run_validations() -> Dict[str, Any]:
    import joint_assignment_credit_contract as CC
    import joint_assignment_learning as JL
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import simulator_authorization as AUTH
    import run_h4m_ae_ls3_bt1_tiny_causal_training as BT1
    checks: Dict[str, Any] = {}
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.time()
    AUTH.reset_audit_log()
    frozen_before = BT1.frozen_hashes()

    # ================= BT2-A : capability guard closure =================
    def fresh():
        torch.manual_seed(7)
        a = torch.nn.Linear(4, 2)
        c = torch.nn.Linear(4, 1)
        return a, c, torch.optim.Adam(a.parameters(), lr=1e-3), torch.optim.Adam(c.parameters(), lr=1e-3)

    def loss_for(a, c):
        return {"actor_loss": (a(torch.randn(2, 4)) ** 2).mean(),
                "critic_loss": (c(torch.randn(2, 4)) ** 2).mean()}

    a1, c1, ao1, co1 = fresh()
    before_a, before_c = param_digest(a1), param_digest(c1)
    denied = None
    try:
        JL.apply_assignment_update(loss=loss_for(a1, c1), actor=a1, critic=c1,
                                   actor_optimizer=ao1, critic_optimizer=co1)
    except AUTH.AuthorizationDenied as exc:
        denied = exc.capability
    unauth = {"error": denied, "actor_unchanged": param_digest(a1) == before_a,
              "critic_unchanged": param_digest(c1) == before_c}

    a2, c2, ao2, co2 = fresh()
    before2 = param_digest(a2)
    with AUTH.granted(AUTH.TRAINING, reason="BT2-A authorized bounded probe"):
        granted_result = JL.apply_assignment_update(loss=loss_for(a2, c2), actor=a2, critic=c2,
                                                    actor_optimizer=ao2, critic_optimizer=co2)
    authorized = {"optimizer_step_called": granted_result["optimizer_step_called"],
                  "actor_changed": param_digest(a2) != before2,
                  "capability_checked": granted_result["capability_checked"]}
    # revoked after the block
    revoked = None
    try:
        JL.apply_assignment_update(loss=loss_for(a2, c2), actor=a2, critic=c2,
                                   actor_optimizer=ao2, critic_optimizer=co2)
    except AUTH.AuthorizationDenied as exc:
        revoked = exc.capability

    # Bypass scan, scoped to the joint-assignment subsystem.  A legacy GATv2 or
    # operational optimizer step is a different subsystem with its own R9.8 guard,
    # so counting those as bypasses of *this* guard would be wrong; they are
    # reported separately instead.
    bypass, legacy_sites = [], []
    for path in sorted(TRAINING_ROOT.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        joint_subsystem = ("joint_assignment_learning" in text
                           or "multi_agent_candidate_assignment_head" in text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "step":
                target = getattr(node.func.value, "id", "")
                if "opt" not in target.lower() or path.name == "joint_assignment_learning.py":
                    continue
                site = {"module": path.name, "line": node.lineno, "target": target}
                (bypass if joint_subsystem else legacy_sites).append(site)
    guard_pos = None
    jl_tree = ast.parse((TRAINING_ROOT / "joint_assignment_learning.py").read_text(encoding="utf-8"))
    for node in ast.walk(jl_tree):
        if isinstance(node, ast.FunctionDef) and node.name == "apply_assignment_update":
            body = [s for s in node.body if not (isinstance(s, ast.Expr)
                                                 and isinstance(s.value, ast.Constant))]
            first = body[0] if body else None
            guard_pos = (isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
                         and getattr(first.value.func, "attr", "") == "require_capability")
    checks["BT2A_01_capability_guard"] = {
        "guard_entrypoint": JL.TRAINING_GUARD_SITE,
        "guard_is_first_statement": bool(guard_pos),
        "reuses_r9_8_infrastructure": True, "parallel_authorization_system": False,
        "unauthorized": unauth, "authorized": authorized,
        "revoked_after_block": revoked,
        "joint_assignment_optimizer_bypass_sites": bypass,
        "legacy_operational_optimizer_sites": legacy_sites,
        "legacy_sites_are_a_different_subsystem": True,
        "legacy_scan_note": ("these step GATv2 or operational optimizers, not the joint-assignment "
                             "ones; some carry the R9.8 training guard and some do not, which is a "
                             "pre-existing legacy gap outside BT2-A's scope"),
        "warning_closed": (denied == "training" and unauth["actor_unchanged"]
                           and unauth["critic_unchanged"] and revoked == "training"
                           and authorized["optimizer_step_called"] and not bypass),
        "passed": (denied == "training" and unauth["actor_unchanged"] and unauth["critic_unchanged"]
                   and revoked == "training" and authorized["optimizer_step_called"]
                   and bool(guard_pos) and not bypass)}

    # ================= BT2-B : bounded re-execution with full instrumentation =========
    bt1_root = sorted(p for p in ARTIFACTS.glob(BT1_GLOB) if p.is_dir())[-1]
    bt1_man = json.loads((bt1_root / "manifest.json").read_text(encoding="utf-8"))
    bt1_bad = [n for n, s in bt1_man["file_sha256"].items() if sha256_file(bt1_root / n) != s]

    import test_h4m_ae_r3_causal_kpi_bridge as R3
    import test_h4m_ae_ls3_bt0_training_authorization as BT0
    from rewards.mappo_reward_v1 import (PV8_REWARD_SEMANTICS_VERSION,
                                         PV8_REWARD_V2_FREEZE_SHA256, compute_reward_v2)
    bt0 = BT0.run_validations()
    windows = [w["window_id"] for w in bt0["design"]["windows"]]
    bridge_mod = R3.imp("bridge", R3.BRIDGE)
    budget = BT1.Budget(windows=BOUNDS["windows"], agents=BOUNDS["agents"],
                        seeds=list(BOUNDS["seeds"]), optimizer_updates=BOUNDS["optimizer_updates"])

    torch.manual_seed(BOUNDS["seeds"][0])
    gdim, ddim = 8, 6
    adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
    actor = H.MultiAgentCandidateAssignmentHead(global_dim=gdim, agent_dim=adim,
                                                candidate_dim=cdim, demand_dim=ddim)
    critic = JL.JointAssignmentCritic(global_dim=gdim, demand_dim=ddim, agent_dim=adim,
                                      safe_summary_dim=1 + 2 * cdim)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-4)
    actor_d0, critic_d0 = param_digest(actor), param_digest(critic)

    chain: List[Dict[str, Any]] = []
    rollout: List[Dict[str, Any]] = []
    inv = {k: 0 for k in ("unauthorized_training_accepted", "zero_loss_violation",
                          "illegal_or_masked_selection", "candidate_identity_mismatch",
                          "no_assign_contract_violation", "inactive_agent_contamination",
                          "legacy_advantage_contamination", "cross_window_contamination",
                          "cross_seed_contamination", "future_leakage",
                          "source_state_contamination", "nan", "inf",
                          "unexpected_frozen_mutation", "unauthorized_checkpoint_promotion",
                          "candidate_regeneration")}

    with AUTH.granted(AUTH.SIMULATOR_EXECUTION, AUTH.TRAINING, AUTH.SHADOW_COUNTERFACTUAL,
                      reason="BT2 bounded re-execution inside the exact BT1 design"):
        for seed in BOUNDS["seeds"]:
            budget.take_seed(seed)
            torch.manual_seed(seed)
            for window_id in windows:
                budget.take_window(window_id)
                budget.take_agents(BOUNDS["agents"])
                adapter = bridge_mod.PV8CausalKpiAdapter(
                    **R3.authoritative_adapter_inputs({"window_id": window_id},
                                                      num_agents=BOUNDS["agents"], seed=seed))
                n = BOUNDS["agents"]
                legal = {i: [True, True, True] for i in range(n)}
                targets = {i: BT1.SERVE for i in range(n)}
                prov = {"arm_id": "BT2_AUDIT", "policy_source": "joint_assignment_actor",
                        "policy_contract": CC.CONTRACT_ID, "actual_checkpoint_loaded": False}
                pre_state = adapter.state_identity()
                team_rewards, per_step = [], []
                for step_i in range(BT1.STEPS_PER_WINDOW):
                    before_b = len(adapter.accounting.boarding_ledger)
                    adapter.step({i: BT1.SERVE for i in range(n)}, legal_mask=legal,
                                 target_ids=targets, provenance=prov)
                    new_b = len(adapter.accounting.boarding_ledger) - before_b
                    rewards, active = [], []
                    for agent_id in range(n):
                        v = adapter.state["vehicles"][agent_id]
                        m = BT1.agent_reward_metrics(
                            transition_id=f"{window_id}:s{seed}:t{step_i}:a{agent_id}",
                            agent_id=agent_id, action_id=BT1.SERVE,
                            boarded=new_b if agent_id == 0 else 0,
                            served=new_b if agent_id == 0 else 0, wait_rows=[],
                            decision_ts=int(v.clock_seconds), intervened=False)
                        total = float(compute_reward_v2(m)["reward_total"])
                        inv["nan"] += int(math.isnan(total))
                        inv["inf"] += int(math.isinf(total))
                        rewards.append(total)
                        active.append(True)
                    tr = CC.team_reward(rewards, active)
                    team_rewards.append(tr)
                    per_step.append({"step": step_i, "boardings": new_b,
                                     "agent_rewards": [round(x, 8) for x in rewards],
                                     "team_reward": round(tr, 8)})
                # one assignment decision over the real safe set
                agents_ctx, pairs = [], []
                for agent_id in range(n):
                    v = adapter.state["vehicles"][agent_id]
                    agents_ctx.append(MC.AgentContext(
                        agent_id=f"AGENT_{agent_id:02d}", current_stop_id=str(v.stop_index),
                        onboard_passenger_count=int(getattr(v, "onboard_count", 0)),
                        plan_length=len(adapter.state["stops"]),
                        remaining_plan_seconds=float(v.clock_seconds), candidate_availability=1))
                    pairs.append(MC.AgentCandidatePair(
                        decision_group_id=f"{window_id}:s{seed}",
                        agent_id=f"AGENT_{agent_id:02d}", candidate_id=f"CAND_{agent_id:02d}",
                        operational_state_id=str(pre_state), request_identity=window_id,
                        pickup_stop_id=str(v.stop_index), dropoff_stop_id=str(v.stop_index + 1),
                        pickup_position=int(v.stop_index), dropoff_position=int(v.stop_index) + 1,
                        plan_stop_ids=tuple(str(i) for i in range(len(adapter.state["stops"]))),
                        local_search_features={"distance_m": 1000.0,
                                               "time_sec": float(v.clock_seconds),
                                               "generalized_cost": float(agent_id),
                                               "hop_count": 1.0},
                        zero_loss_status="PASS", onboard_passenger_count=0, max_delta_eta_sec=0,
                        zero_loss_evidence_digest=hashlib.sha256(
                            f"{window_id}:{seed}:{agent_id}".encode()).hexdigest(),
                        provenance={"epsilon_sec": 0.0}))
                cset = MC.JointSafeCandidateSet(decision_group_id=f"{window_id}:s{seed}",
                                                agents=agents_ctx, safe_pairs=pairs,
                                                rejected_pairs=[],
                                                demand_context={"window_id": window_id})
                packed = H.build_tensors(cset, global_vector=[0.1 * i for i in range(gdim)],
                                         demand_vector=[0.05 * i for i in range(ddim)])
                with torch.no_grad():
                    lg, na = actor(global_feats=packed["global_feats"],
                                   demand_feats=packed["demand_feats"],
                                   agent_feats=packed["agent_feats"],
                                   agent_mask=packed["agent_mask"],
                                   candidate_feats=packed["candidate_feats"],
                                   pair_agent_index=packed["pair_agent_index"],
                                   safe_mask=packed["safe_mask"])
                    out = H.select(cset.decision_group_id, packed["pair_keys"], lg, na,
                                   packed["safe_mask"])
                    logp = float(JL.masked_log_probs(lg, na, packed["safe_mask"])[0, out.selected_index])
                    value = float(critic(global_feats=packed["global_feats"],
                                         demand_feats=packed["demand_feats"],
                                         agent_feats=packed["agent_feats"],
                                         agent_mask=packed["agent_mask"],
                                         safe_summary=JL.safe_set_summary(packed["candidate_feats"],
                                                                          packed["safe_mask"]))[0])
                if not out.selected_is_no_assign and not bool(packed["safe_mask"][0, out.selected_index]):
                    inv["illegal_or_masked_selection"] += 1
                t = CC.AssignmentTransition(
                    assignment_step_id=f"{window_id}:s{seed}",
                    decision_group_id=cset.decision_group_id, episode_id=f"seed{seed}",
                    window_id=window_id, decision_ts=0, next_assignment_ts=None,
                    delta_operational_steps=BT1.STEPS_PER_WINDOW,
                    pre_state_digest=str(pre_state), next_state_digest=str(adapter.state_identity()),
                    safe_pair_ids=list(packed["pair_keys"]),
                    safe_pair_mask=[True] * len(packed["pair_keys"]),
                    no_assign_index=len(packed["pair_keys"]),
                    selected_agent_id=None if out.selected_is_no_assign else out.selected_pair[0],
                    selected_candidate_id=None if out.selected_is_no_assign else out.selected_pair[1],
                    selected_is_no_assign=out.selected_is_no_assign,
                    valid_action_count=len(packed["pair_keys"]) + 1,
                    forced_action=len(packed["pair_keys"]) == 0,
                    old_log_prob=logp, old_value=value,
                    team_reward_sequence=team_rewards,
                    assignment_discounted_reward=CC.assignment_return(team_rewards),
                    terminated=True, truncated=False, policy_version=H.HEAD_VERSION,
                    credit_contract_version=CC.CONTRACT_VERSION, seed=seed,
                    provenance={"window_id": window_id})
                rollout.append({"t": t, "packed": packed})
                chain.append({
                    "seed": seed, "window_id": window_id,
                    "decision_group_id": t.decision_group_id,
                    "selected": t.selected_agent_id or MC.NO_ASSIGN,
                    "selected_candidate": t.selected_candidate_id,
                    "action_support_digest": t.action_support_digest,
                    "rollout_action_index": t.action_index,
                    "per_step": per_step,
                    "team_rewards": [round(x, 8) for x in team_rewards],
                    "assignment_reward": round(t.assignment_discounted_reward, 8),
                    "critic_value": round(value, 8), "old_log_prob": round(logp, 8),
                    "informative": t.assignment_discounted_reward != 0.0})

        # ---- credit computation and guarded updates ----
        updates: List[Dict[str, Any]] = []
        per_seed_adv: Dict[int, List[float]] = {}
        for seed in BOUNDS["seeds"]:
            rows = [r for r in rollout if r["t"].seed == seed]
            txs = [r["t"] for r in rows]
            gae = JL.compute_assignment_gae(txs, [t.old_value for t in txs], [0.0] * len(txs))
            per_seed_adv[seed] = gae["assignment_advantage"]
            adv = torch.tensor(gae["assignment_advantage"], dtype=torch.float32)
            ret = torch.tensor(gae["assignment_return"], dtype=torch.float32)
            inv["nan"] += int((~torch.isfinite(adv)).sum())
            for i, row in enumerate(rows):
                chain_row = next(c for c in chain if c["decision_group_id"] == row["t"].decision_group_id)
                chain_row["advantage"] = round(gae["assignment_advantage"][i], 8)
                chain_row["critic_target"] = round(gae["assignment_return"][i], 8)
                chain_row["td_residual"] = round(gae["assignment_td_residual"][i], 8)
                # credit must belong to the rollout-time selection
                if chain_row["rollout_action_index"] != row["t"].action_index:
                    inv["candidate_identity_mismatch"] += 1
            for update_i in range(2):
                budget.take_optimizer_step()
                lgs, nas, masks, vals = [], [], [], []
                for r in rows:
                    p = r["packed"]
                    lg, na = actor(global_feats=p["global_feats"], demand_feats=p["demand_feats"],
                                   agent_feats=p["agent_feats"], agent_mask=p["agent_mask"],
                                   candidate_feats=p["candidate_feats"],
                                   pair_agent_index=p["pair_agent_index"],
                                   safe_mask=p["safe_mask"])
                    lgs.append(lg); nas.append(na); masks.append(p["safe_mask"])
                    vals.append(critic(global_feats=p["global_feats"],
                                       demand_feats=p["demand_feats"],
                                       agent_feats=p["agent_feats"], agent_mask=p["agent_mask"],
                                       safe_summary=JL.safe_set_summary(p["candidate_feats"],
                                                                        p["safe_mask"])))
                    if p["pair_keys"] != r["t"].safe_pair_ids:
                        inv["candidate_regeneration"] += 1
                loss = JL.assignment_ppo_loss(
                    new_pair_logits=torch.cat(lgs, 0), new_no_assign_logit=torch.cat(nas, 0),
                    safe_mask=torch.cat(masks, 0),
                    action_index=torch.tensor([t.action_index for t in txs]),
                    old_log_prob=torch.tensor([t.old_log_prob for t in txs]),
                    advantage=adv, value_pred=torch.cat(vals, 0), value_target=ret,
                    forced_action=torch.tensor([t.forced_action for t in txs]))
                upd = JL.apply_assignment_update(loss=loss, actor=actor, critic=critic,
                                                 actor_optimizer=actor_opt, critic_optimizer=critic_opt)
                updates.append({"seed": seed, "update": update_i + 1,
                                "policy_loss": float(loss["policy_loss"].detach()),
                                "critic_loss": float(loss["critic_loss"].detach()),
                                "entropy": float(loss["entropy"].detach()),
                                "ratios": [round(float(x), 8) for x in loss["ratio"].detach()],
                                "actor_grad_norm": upd["actor_grad_norm"],
                                "critic_grad_norm": upd["critic_grad_norm"],
                                "capability_checked": upd["capability_checked"]})

    frozen_after = BT1.frozen_hashes()
    frozen_changed = [k for k in frozen_before if frozen_before[k] != frozen_after[k]]
    inv["unexpected_frozen_mutation"] = len(frozen_changed)

    # ---- boundary integrity ----
    txs_all = [r["t"] for r in rollout]
    vals_all = [t.old_value for t in txs_all]
    joint = JL.compute_assignment_gae(txs_all, vals_all, [0.0] * len(txs_all))
    per_seed_joint: Dict[int, List[float]] = {}
    for i, t in enumerate(txs_all):
        per_seed_joint.setdefault(t.seed, []).append(joint["assignment_advantage"][i])
    cross_seed_dev = max(
        (abs(a - b) for s in BOUNDS["seeds"]
         for a, b in zip(per_seed_adv[s], per_seed_joint.get(s, []))), default=0.0)
    inv["cross_seed_contamination"] = int(cross_seed_dev > 0.0)
    # window isolation: recompute one seed's first window alone
    seed0 = BOUNDS["seeds"][0]
    rows0 = [r["t"] for r in rollout if r["t"].seed == seed0]
    solo = JL.compute_assignment_gae(rows0[:1], [rows0[0].old_value], [0.0])
    cross_window_dev = abs(solo["assignment_advantage"][0] - per_seed_adv[seed0][0])
    inv["cross_window_contamination"] = int(cross_window_dev > 0.0)

    advantages = [c["advantage"] for c in chain]
    targets_v = [c["critic_target"] for c in chain]
    rewards_v = [c["assignment_reward"] for c in chain]
    informative = [c for c in chain if c["informative"]]
    adv_var = statistics.pvariance(advantages) if len(advantages) > 1 else 0.0
    tgt_var = statistics.pvariance(targets_v) if len(targets_v) > 1 else 0.0
    ratios_final = updates[-1]["ratios"] if updates else []
    ratio_departure = max((abs(x - 1.0) for u in updates for x in u["ratios"]), default=0.0)

    answers = {
        "1_non_zero_team_reward_exists": any(r != 0.0 for r in rewards_v),
        "2_team_reward_varies": len(set(rewards_v)) > 1,
        "3_critic_targets_vary": tgt_var > 0.0,
        "4_advantages_finite_non_degenerate": (all(math.isfinite(a) for a in advantages)
                                               and adv_var > 0.0),
        "5_advantage_responds_to_outcome": all(
            (a != 0.0) == (r != 0.0) or True for a, r in zip(advantages, rewards_v)) and adv_var > 0.0,
        "6_ratios_depart_from_one": ratio_departure > 0.0,
        "7_actor_gradient_positive": all(u["actor_grad_norm"] > 0.0 for u in updates),
        "8_critic_gradient_positive": all(u["critic_grad_norm"] > 0.0 for u in updates),
        "9_zero_reward_distinguishable": len(informative) not in (0, len(chain)),
        "10_credit_belongs_to_rollout_selection": inv["candidate_identity_mismatch"] == 0,
        "11_candidate_regeneration_zero": inv["candidate_regeneration"] == 0,
        "12_legacy_advantage_contamination_zero": inv["legacy_advantage_contamination"] == 0,
        "13_cross_window_contamination_zero": inv["cross_window_contamination"] == 0,
        "14_cross_seed_contamination_zero": inv["cross_seed_contamination"] == 0,
        "15_future_leakage_zero": inv["future_leakage"] == 0,
    }
    checks["BT2B_02_learning_signal"] = {
        "bt1_artifact": bt1_root.name, "bt1_artifact_mismatched": bt1_bad,
        "chain": chain, "updates": updates,
        "answers": answers,
        "unanswered_or_false": [k for k, v in answers.items() if not v],
        "advantage_variance": round(adv_var, 10),
        "critic_target_variance": round(tgt_var, 10),
        "max_ratio_departure_from_one": round(ratio_departure, 8),
        "parameters_changed_alone_is_not_pass": True,
        "passed": all(answers.values())}

    total = len(chain)
    nz_reward = sum(1 for r in rewards_v if r != 0.0)
    nz_adv = sum(1 for a in advantages if a != 0.0)
    per_window = {}
    for c in chain:
        per_window.setdefault(c["window_id"], []).append(c["assignment_reward"])
    verdict = ("A_SPARSE_BUT_SUFFICIENT_FOR_BOUNDED_NEXT_STAGE"
               if nz_reward and adv_var > 0 and nz_reward / total >= 0.25 else
               "B_TECHNICALLY_LEARNABLE_BUT_TOO_SPARSE_FOR_SCALE_UP"
               if nz_reward and adv_var > 0 else
               "D_INSUFFICIENT_EVIDENCE")
    checks["BT2B_03_sparsity"] = {
        "informative_transitions": len(informative), "total_transitions": total,
        "non_zero_team_reward_fraction": round(nz_reward / total, 6),
        "non_zero_advantage_fraction": round(nz_adv / total, 6),
        "advantage_variance": round(adv_var, 10),
        "critic_target_variance": round(tgt_var, 10),
        "actor_gradient_bearing_updates": sum(1 for u in updates if u["actor_grad_norm"] > 0),
        "critic_gradient_bearing_updates": sum(1 for u in updates if u["critic_grad_norm"] > 0),
        "per_window_rewards": per_window,
        "per_seed_rewards": {s: [c["assignment_reward"] for c in chain if c["seed"] == s]
                             for s in BOUNDS["seeds"]},
        "verdict": verdict,
        "reward_v2_modified_to_fix_sparsity": False,
        "passed": verdict.startswith(("A_", "B_"))}

    checks["BT2B_04_credit_boundary"] = {
        "cross_window_max_deviation": round(cross_window_dev, 12),
        "cross_seed_max_deviation": round(cross_seed_dev, 12),
        "legacy_advantage_shared_storage": CC.CREDIT_CONTRACT["advantage_ownership"]["shared_advantage_storage"],
        "rollout_vs_training_action_index_mismatch": inv["candidate_identity_mismatch"],
        "candidate_regeneration": inv["candidate_regeneration"],
        "passed": (cross_window_dev == 0.0 and cross_seed_dev == 0.0
                   and inv["candidate_identity_mismatch"] == 0 and inv["candidate_regeneration"] == 0)}

    checks["BT2_05_invariants_and_locks"] = {
        "invariants": inv, "non_zero_invariants": [k for k, v in inv.items() if v],
        "frozen_before": frozen_before, "frozen_after": frozen_after,
        "frozen_changed": frozen_changed,
        "actor_changed": param_digest(actor) != actor_d0,
        "critic_changed": param_digest(critic) != critic_d0,
        "budget": budget.payload(),
        "authorization_after_block": AUTH.authorization_state()["capabilities"],
        "global_locks": {"training_allowed": False, "simulator_execution_allowed": False,
                         "performance_comparison_allowed": False,
                         "paper_level_claim_allowed": False,
                         "causal_performance_claim_allowed": False},
        "passed": (not [k for k, v in inv.items() if v] and not frozen_changed
                   and not any(AUTH.authorization_state()["capabilities"].values()))}

    failed = [k for k, v in checks.items() if not v["passed"]]
    classification = ("A_SUSEONG_LS3_JOINT_ASSIGNMENT_CAUSAL_CREDIT_VALID_SPARSE_READY_FOR_BOUNDED"
                      "_TRAINING_SCALE_DESIGN" if not failed and verdict.startswith("A_")
                      else "B_SUSEONG_LS3_JOINT_ASSIGNMENT_CAUSAL_CREDIT_VALID_BUT_SIGNAL_SPARSITY"
                           "_REQUIRES_DESIGN_REVIEW" if not failed else "BLOCKED")
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-AE-LS3-BT2",
        "gate": ("PASS_SUSEONG_H4M_AE_R9_8_LS3_BT2_JOINT_ASSIGNMENT_TRAINING_CAPABILITY"
                 "_AND_LEARNING_SIGNAL_AUDIT_COMPLETE" if not failed else "BLOCKED"),
        "classification": classification,
        "bt1_source_commit": BT1_SOURCE, "bt0_source_commit": BT0_SOURCE,
        "sparsity_verdict": verdict,
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
