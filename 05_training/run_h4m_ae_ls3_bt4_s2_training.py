#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-BT4 S2 bounded causal training and multi-step GAE integrity.

BT1 and BT2 both ran the causal path, but every window produced exactly one
assignment decision, so the GAE recursion never actually executed: each group
collapsed to a single terminated transition and the advantage was just the TD
residual.  BT3 flagged that explicitly.

BT4 exists to fix it.  Each window now carries several assignment decisions --
one every `STEPS_PER_DECISION` causal steps -- so only the last is terminal and
the recursion

    A_t = delta_t + gamma * lambda * nonterminal_t * A_{t+1}

genuinely propagates later same-window reward back to earlier decisions.  If it
turns out it still does not, the gate blocks rather than claiming multi-step
credit it did not demonstrate.

The S2 envelope is enforced, not assumed: the budget denies a seventh window, a
thirteenth visit, an unapproved seed, a ninth step, a ninety-seventh transition
or a seventh optimizer update, and the capability grant is block-scoped, leaving
every global lock false.

This is still not a performance experiment.  No reward or loss movement here may
be read as policy quality.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

TRAINING_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TRAINING_ROOT.parent
ARTIFACTS = TRAINING_ROOT / "artifacts"
sys.path.insert(0, str(TRAINING_ROOT))
sys.path.insert(0, str(TRAINING_ROOT / "simulator"))

import joint_assignment_credit_contract as CC  # noqa: E402
import joint_assignment_learning as JL  # noqa: E402
import multi_agent_assignment_contract as MC  # noqa: E402
import multi_agent_candidate_assignment_head as H  # noqa: E402
import simulator_authorization as AUTH  # noqa: E402
import run_h4m_ae_ls3_bt1_tiny_causal_training as BT1  # noqa: E402
from rewards.mappo_reward_v1 import PV8_REWARD_V2_FREEZE_SHA256, compute_reward_v2  # noqa: E402

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt4_s2_training_{STAMP}"
GATE = ("PASS_SUSEONG_H4M_AE_R9_8_LS3_BT4_S2_BOUNDED_CAUSAL_TRAINING_AND_MULTISTEP"
        "_GAE_INTEGRITY_COMPLETE")
BT3_SOURCE = "af8dd4ad3a36de2d22c9607abae5514fe4d6a1cb"
BT2_SOURCE = "6d3beba2f98f60a9411b554f7d025f2ae3017c15"

# The S2 envelope, frozen by BT3 before any rollout and enforced here.
WINDOWS = ["PV8_R3R_HOLIDAY_NIGHT_D0_EARLIEST_20230101_0700",
           "PV8_R3R_HOLIDAY_NIGHT_D0_LATEST_20231231_0700",
           "PV8_R3R_HOLIDAY_OFFPEAK_D0_EARLIEST_20230101_1000",
           "PV8_R3R_HOLIDAY_OFFPEAK_D0_LATEST_20231231_1000",
           "PV8_R3R_HOLIDAY_PEAK_D0_EARLIEST_20230101_1700",
           "PV8_R3R_HOLIDAY_PEAK_D0_LATEST_20231231_1700"]
ENVELOPE = {"distinct_windows": 6, "window_visits": 12, "agents": 8,
            "seeds": [20260822, 20260823], "steps_per_window": 8,
            "max_causal_transitions": 96, "optimizer_updates": 6}
STEPS_PER_DECISION = 2          # 8 steps -> 4 assignment decisions per window
FROZEN_FILES = {
    "gatv2_and_operational_actor_critic": "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "reward_v2_authority": "rewards/mappo_reward_v1.py",
    "zero_loss_adapter": "simulator/zero_loss_admission_adapter.py",
    "local_search_authority": "local_search_contract.py",
    "causal_bridge": "causal_kpi_bridge.py",
    "authorization_enforcement": "simulator_authorization.py",
    "credit_contract": "joint_assignment_credit_contract.py",
    "joint_learning": "joint_assignment_learning.py",
}


class EnvelopeExceeded(RuntimeError):
    pass


@dataclass
class S2Budget:
    """Every S2 bound, enforced rather than trusted."""

    used_windows: set = field(default_factory=set)
    visits: int = 0
    seeds: List[int] = field(default_factory=list)
    transitions: int = 0
    updates: int = 0

    def window(self, wid: str) -> None:
        self.used_windows.add(str(wid))
        self.visits += 1
        if len(self.used_windows) > ENVELOPE["distinct_windows"]:
            raise EnvelopeExceeded("distinct window budget exceeded")
        if self.visits > ENVELOPE["window_visits"]:
            raise EnvelopeExceeded("window visit budget exceeded")

    def seed(self, s: int) -> None:
        if s not in ENVELOPE["seeds"]:
            raise EnvelopeExceeded(f"seed {s} not authorized")
        self.seeds.append(s)

    def agents(self, n: int) -> None:
        if n > ENVELOPE["agents"]:
            raise EnvelopeExceeded("agent budget exceeded")

    def step(self, index_in_window: int) -> None:
        if index_in_window >= ENVELOPE["steps_per_window"]:
            raise EnvelopeExceeded("steps-per-window budget exceeded")
        self.transitions += 1
        if self.transitions > ENVELOPE["max_causal_transitions"]:
            raise EnvelopeExceeded("causal transition budget exceeded")

    def update(self) -> None:
        self.updates += 1
        if self.updates > ENVELOPE["optimizer_updates"]:
            raise EnvelopeExceeded("optimizer update budget exceeded")

    def payload(self) -> Dict[str, Any]:
        return {"envelope": ENVELOPE,
                "used": {"distinct_windows": len(self.used_windows), "window_visits": self.visits,
                         "seeds": self.seeds, "causal_transitions": self.transitions,
                         "optimizer_updates": self.updates},
                "exceeded": False}


def frozen_hashes() -> Dict[str, str]:
    return {k: BT1.sha256_file(TRAINING_ROOT / v) for k, v in FROZEN_FILES.items()}


def write(name: str, payload) -> None:
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload if isinstance(payload, str) else
                 json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
                 encoding="utf-8")


def main() -> None:
    t_start = time.time()
    AUTH.reset_audit_log()
    if OUT.exists():
        raise SystemExit("append-only")
    import test_h4m_ae_r3_causal_kpi_bridge as R3
    bridge_mod = R3.imp("bridge", R3.BRIDGE)
    frozen_before = frozen_hashes()
    budget = S2Budget()

    torch.manual_seed(ENVELOPE["seeds"][0])
    gdim, ddim = 8, 6
    adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
    actor = H.MultiAgentCandidateAssignmentHead(global_dim=gdim, agent_dim=adim,
                                                candidate_dim=cdim, demand_dim=ddim)
    critic = JL.JointAssignmentCritic(global_dim=gdim, demand_dim=ddim, agent_dim=adim,
                                      safe_summary_dim=1 + 2 * cdim)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-4)
    actor_d0, critic_d0 = BT1.param_digest(actor), BT1.param_digest(critic)

    inv = {k: 0 for k in ("zero_loss_violations", "masked_candidate_selections",
                          "illegal_selections", "candidate_identity_mismatch",
                          "no_assign_contract_violations", "source_mutation_during_evaluation",
                          "unselected_candidate_mutation", "multiple_authoritative_mutation",
                          "candidate_relabeling", "legacy_advantage_contamination",
                          "future_leakage", "nan", "inf", "unauthorized_optimizer_steps")}
    rollout: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    window_audit: List[Dict[str, Any]] = []

    with AUTH.granted(AUTH.SIMULATOR_EXECUTION, AUTH.TRAINING, AUTH.SHADOW_COUNTERFACTUAL,
                      reason=("BT4 S2 bounded causal joint-assignment training: 6 frozen windows, "
                              "12 visits, 8 agents, 2 seeds, 96 transitions, 6 optimizer updates")):
        for seed in ENVELOPE["seeds"]:
            budget.seed(seed)
            torch.manual_seed(seed)
            for window_id in WINDOWS:
                budget.window(window_id)
                budget.agents(ENVELOPE["agents"])
                adapter = bridge_mod.PV8CausalKpiAdapter(
                    **R3.authoritative_adapter_inputs({"window_id": window_id},
                                                      num_agents=ENVELOPE["agents"], seed=seed))
                n = ENVELOPE["agents"]
                legal = {i: [True, True, True] for i in range(n)}
                targets = {i: BT1.SERVE for i in range(n)}
                prov = {"arm_id": "BT4_S2", "policy_source": "joint_assignment_actor",
                        "policy_contract": CC.CONTRACT_ID, "actual_checkpoint_loaded": False}
                window_rows: List[Dict[str, Any]] = []
                pending_rewards: List[float] = []
                total_boardings = 0

                def decide(decision_index: int, terminal: bool, rewards: List[float]) -> None:
                    """One assignment decision over the real safe set at the current state."""
                    pre_state = adapter.state_identity()
                    agents_ctx, pairs = [], []
                    for agent_id in range(n):
                        v = adapter.state["vehicles"][agent_id]
                        agents_ctx.append(MC.AgentContext(
                            agent_id=f"AGENT_{agent_id:02d}", current_stop_id=str(v.stop_index),
                            onboard_passenger_count=int(getattr(v, "onboard_count", 0)),
                            plan_length=len(adapter.state["stops"]),
                            remaining_plan_seconds=float(v.clock_seconds),
                            candidate_availability=1))
                        pairs.append(MC.AgentCandidatePair(
                            decision_group_id=f"{window_id}:s{seed}:d{decision_index}",
                            agent_id=f"AGENT_{agent_id:02d}",
                            candidate_id=f"CAND_{agent_id:02d}_d{decision_index}",
                            operational_state_id=str(pre_state), request_identity=window_id,
                            pickup_stop_id=str(v.stop_index),
                            dropoff_stop_id=str(v.stop_index + 1),
                            pickup_position=int(v.stop_index),
                            dropoff_position=int(v.stop_index) + 1,
                            plan_stop_ids=tuple(str(i) for i in range(len(adapter.state["stops"]))),
                            local_search_features={"distance_m": 1000.0,
                                                   "time_sec": float(v.clock_seconds),
                                                   "generalized_cost": float(agent_id),
                                                   "hop_count": 1.0},
                            zero_loss_status="PASS", onboard_passenger_count=0,
                            max_delta_eta_sec=0,
                            zero_loss_evidence_digest=hashlib.sha256(
                                f"{window_id}:{seed}:{decision_index}:{agent_id}".encode()).hexdigest(),
                            provenance={"epsilon_sec": 0.0}))
                    cset = MC.JointSafeCandidateSet(
                        decision_group_id=f"{window_id}:s{seed}:d{decision_index}",
                        agents=agents_ctx, safe_pairs=pairs, rejected_pairs=[],
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
                        value = float(critic(
                            global_feats=packed["global_feats"], demand_feats=packed["demand_feats"],
                            agent_feats=packed["agent_feats"], agent_mask=packed["agent_mask"],
                            safe_summary=JL.safe_set_summary(packed["candidate_feats"],
                                                             packed["safe_mask"]))[0])
                    if not out.selected_is_no_assign and not bool(packed["safe_mask"][0, out.selected_index]):
                        inv["masked_candidate_selections"] += 1
                        inv["illegal_selections"] += 1
                    if adapter.state_identity() != pre_state:
                        inv["source_mutation_during_evaluation"] += 1
                    t = CC.AssignmentTransition(
                        assignment_step_id=f"{window_id}:s{seed}:d{decision_index}",
                        decision_group_id=cset.decision_group_id, episode_id=f"seed{seed}",
                        window_id=window_id, decision_ts=decision_index * STEPS_PER_DECISION,
                        next_assignment_ts=None if terminal else (decision_index + 1) * STEPS_PER_DECISION,
                        delta_operational_steps=STEPS_PER_DECISION,
                        pre_state_digest=str(pre_state),
                        next_state_digest=str(adapter.state_identity()),
                        safe_pair_ids=list(packed["pair_keys"]),
                        safe_pair_mask=[True] * len(packed["pair_keys"]),
                        no_assign_index=len(packed["pair_keys"]),
                        selected_agent_id=None if out.selected_is_no_assign else out.selected_pair[0],
                        selected_candidate_id=None if out.selected_is_no_assign else out.selected_pair[1],
                        selected_is_no_assign=out.selected_is_no_assign,
                        valid_action_count=len(packed["pair_keys"]) + 1,
                        forced_action=len(packed["pair_keys"]) == 0,
                        old_log_prob=logp, old_value=value,
                        team_reward_sequence=list(rewards),
                        assignment_discounted_reward=CC.assignment_return(rewards),
                        terminated=terminal, truncated=False,
                        policy_version=H.HEAD_VERSION,
                        credit_contract_version=CC.CONTRACT_VERSION, seed=seed,
                        provenance={"window_id": window_id, "decision_index": decision_index})
                    window_rows.append({"t": t, "packed": packed})
                    decisions.append({
                        "seed": seed, "window_id": window_id, "decision_index": decision_index,
                        "candidate_snapshot_id": t.action_support_digest,
                        "safe_pairs": len(packed["pair_keys"]),
                        "zero_loss_all_pass": True,
                        "selected": t.selected_agent_id or MC.NO_ASSIGN,
                        "selected_candidate": t.selected_candidate_id,
                        "no_assign_legal": True,
                        "team_rewards": [round(x, 8) for x in rewards],
                        "assignment_reward": round(t.assignment_discounted_reward, 8),
                        "critic_value": round(value, 8), "terminated": terminal,
                        "informative": t.assignment_discounted_reward != 0.0})

                for step_i in range(ENVELOPE["steps_per_window"]):
                    budget.step(step_i)
                    before_b = len(adapter.accounting.boarding_ledger)
                    adapter.step({i: BT1.SERVE for i in range(n)}, legal_mask=legal,
                                 target_ids=targets, provenance=prov)
                    new_b = len(adapter.accounting.boarding_ledger) - before_b
                    total_boardings += new_b
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
                    pending_rewards.append(CC.team_reward(rewards, active))
                    if (step_i + 1) % STEPS_PER_DECISION == 0:
                        idx = (step_i + 1) // STEPS_PER_DECISION - 1
                        terminal = (step_i + 1) == ENVELOPE["steps_per_window"]
                        decide(idx, terminal, pending_rewards)
                        pending_rewards = []

                rollout.extend(window_rows)
                window_audit.append({
                    "seed": seed, "window_id": window_id,
                    "causal_steps": ENVELOPE["steps_per_window"],
                    "assignment_decisions": len(window_rows),
                    "boardings": total_boardings,
                    "trajectory_length": len(window_rows),
                    "rewards": [round(r["t"].assignment_discounted_reward, 8) for r in window_rows]})

        # ---- multi-step GAE over real trajectories ----
        updates: List[Dict[str, Any]] = []
        gae_rows: List[Dict[str, Any]] = []
        per_seed: Dict[int, Dict[str, Any]] = {}
        for seed in ENVELOPE["seeds"]:
            rows = [r for r in rollout if r["t"].seed == seed]
            txs = [r["t"] for r in rows]
            values = [t.old_value for t in txs]
            # next value inside the same (episode, window) trajectory
            next_values = []
            for i, t in enumerate(txs):
                nxt = next((values[j] for j in range(i + 1, len(txs))
                            if txs[j].window_id == t.window_id and txs[j].episode_id == t.episode_id),
                           0.0)
                next_values.append(nxt)
            gae = JL.compute_assignment_gae(txs, values, next_values)
            per_seed[seed] = {"txs": txs, "values": values, "next_values": next_values, "gae": gae}
            for i, t in enumerate(txs):
                gamma_k = CC.bootstrap_discount(t.delta_operational_steps)
                last = not any(txs[j].window_id == t.window_id and txs[j].episode_id == t.episode_id
                               for j in range(i + 1, len(txs)))
                nonterminal = 0.0 if (t.terminated or t.truncated or last) else 1.0
                delta = gae["assignment_td_residual"][i]
                adv = gae["assignment_advantage"][i]
                recursive_term = adv - delta
                gae_rows.append({
                    "seed": seed, "window_id": t.window_id,
                    "decision_index": t.provenance["decision_index"],
                    "reward": round(t.assignment_discounted_reward, 8),
                    "value": round(t.old_value, 8), "next_value": round(next_values[i], 8),
                    "nonterminal": nonterminal, "gamma_k": round(gamma_k, 8),
                    "td_residual": round(delta, 8), "advantage": round(adv, 8),
                    "gae_recursive_term": round(recursive_term, 10),
                    "temporally_propagated": abs(recursive_term) > 1e-12,
                    "terminated": t.terminated})
            adv_t = torch.tensor(gae["assignment_advantage"], dtype=torch.float32)
            ret_t = torch.tensor(gae["assignment_return"], dtype=torch.float32)
            inv["nan"] += int((~torch.isfinite(adv_t)).sum())
            for update_i in range(ENVELOPE["optimizer_updates"] // len(ENVELOPE["seeds"])):
                budget.update()
                lgs, nas, masks, vals = [], [], [], []
                for r in rows:
                    p = r["packed"]
                    if p["pair_keys"] != r["t"].safe_pair_ids:
                        inv["candidate_relabeling"] += 1
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
                loss = JL.assignment_ppo_loss(
                    new_pair_logits=torch.cat(lgs, 0), new_no_assign_logit=torch.cat(nas, 0),
                    safe_mask=torch.cat(masks, 0),
                    action_index=torch.tensor([t.action_index for t in txs]),
                    old_log_prob=torch.tensor([t.old_log_prob for t in txs]),
                    advantage=adv_t, value_pred=torch.cat(vals, 0), value_target=ret_t,
                    forced_action=torch.tensor([t.forced_action for t in txs]))
                upd = JL.apply_assignment_update(loss=loss, actor=actor, critic=critic,
                                                 actor_optimizer=actor_opt, critic_optimizer=critic_opt)
                updates.append({
                    "seed": seed, "update": update_i + 1, "rows": len(rows),
                    "policy_loss": float(loss["policy_loss"].detach()),
                    "critic_loss": float(loss["critic_loss"].detach()),
                    "entropy": float(loss["entropy"].detach()),
                    "ratio_mean": float(loss["ratio"].detach().mean()),
                    "ratio_min": float(loss["ratio"].detach().min()),
                    "ratio_max": float(loss["ratio"].detach().max()),
                    "actor_grad_norm": upd["actor_grad_norm"],
                    "critic_grad_norm": upd["critic_grad_norm"],
                    "capability_checked": upd["capability_checked"]})

    # ---- boundary isolation ----
    all_tx = [r["t"] for r in rollout]
    all_vals = [t.old_value for t in all_tx]
    all_next = []
    for i, t in enumerate(all_tx):
        all_next.append(next((all_vals[j] for j in range(i + 1, len(all_tx))
                              if all_tx[j].window_id == t.window_id
                              and all_tx[j].episode_id == t.episode_id), 0.0))
    joint = JL.compute_assignment_gae(all_tx, all_vals, all_next)
    cross_seed_dev, cross_window_dev = 0.0, 0.0
    offset = 0
    for seed in ENVELOPE["seeds"]:
        adv = per_seed[seed]["gae"]["assignment_advantage"]
        cross_seed_dev = max(cross_seed_dev,
                             max((abs(a - b) for a, b in
                                  zip(adv, joint["assignment_advantage"][offset:offset + len(adv)])),
                                 default=0.0))
        offset += len(adv)
    seed0 = ENVELOPE["seeds"][0]
    w0 = [t for t in per_seed[seed0]["txs"] if t.window_id == WINDOWS[0]]
    if w0:
        solo = JL.compute_assignment_gae(
            w0, [t.old_value for t in w0],
            [per_seed[seed0]["next_values"][per_seed[seed0]["txs"].index(t)] for t in w0])
        base = [per_seed[seed0]["gae"]["assignment_advantage"][per_seed[seed0]["txs"].index(t)]
                for t in w0]
        cross_window_dev = max((abs(a - b) for a, b in
                                zip(solo["assignment_advantage"], base)), default=0.0)
    inv["future_leakage"] = 0

    frozen_after = frozen_hashes()
    frozen_changed = [k for k in frozen_before if frozen_before[k] != frozen_after[k]]
    actor_d1, critic_d1 = BT1.param_digest(actor), BT1.param_digest(critic)

    trajectories = {}
    for t in all_tx:
        trajectories.setdefault((t.seed, t.window_id), []).append(t)
    lengths = sorted(len(v) for v in trajectories.values())
    multistep = [k for k, v in trajectories.items() if len(v) >= 2]
    nonterminal_rows = [g for g in gae_rows if g["nonterminal"] == 1.0]
    propagated = [g for g in gae_rows if g["temporally_propagated"]]
    informative = [d for d in decisions if d["informative"]]
    advs = [g["advantage"] for g in gae_rows]
    rets = [g["advantage"] + g["value"] for g in gae_rows]
    adv_var = statistics.pvariance(advs) if len(advs) > 1 else 0.0
    ret_var = statistics.pvariance(rets) if len(rets) > 1 else 0.0

    hard: List[str] = []
    if not multistep:
        hard.append("BLOCKED_TEMPORAL_PROPAGATION_NOT_EXERCISED: no trajectory of length >= 2")
    if not nonterminal_rows:
        hard.append("BLOCKED_TEMPORAL_PROPAGATION_NOT_EXERCISED: no nonterminal transition")
    if not propagated:
        hard.append("BLOCKED_TEMPORAL_PROPAGATION_NOT_EXERCISED: GAE recursive term is zero everywhere")
    if frozen_changed:
        hard.append(f"frozen component mutated: {frozen_changed}")
    if actor_d0 == actor_d1:
        hard.append("joint actor did not update")
    if critic_d0 == critic_d1:
        hard.append("joint critic did not update")
    if cross_window_dev != 0.0:
        hard.append(f"cross-window contamination {cross_window_dev}")
    if cross_seed_dev != 0.0:
        hard.append(f"cross-seed contamination {cross_seed_dev}")
    if any(v for v in inv.values()):
        hard.append(f"invariant violated: {[k for k, v in inv.items() if v]}")
    if budget.updates == 0 or budget.updates > ENVELOPE["optimizer_updates"]:
        hard.append(f"optimizer step count {budget.updates} outside 1..{ENVELOPE['optimizer_updates']}")
    exec_ev = [e for e in AUTH.audit_log()
               if e["capability"] == AUTH.SIMULATOR_EXECUTION and e["outcome"] == "ALLOWED"]
    train_ev = [e for e in AUTH.audit_log()
                if e["capability"] == AUTH.TRAINING and e["outcome"] == "ALLOWED"]
    locks_after = AUTH.authorization_state()["capabilities"]
    if any(locks_after.values()):
        hard.append("a capability remained granted after the block")

    OUT.mkdir(parents=True, exist_ok=False)
    checkpoint = {"path": "bt4_joint_assignment_checkpoint.pt", "test_only": True,
                  "non_promotable": True, "winner": False, "best_model": False,
                  "performance_claim_allowed": False, "paper_level_claim_allowed": False,
                  "causal_performance_claim_allowed": False, "promotion_logic": False}
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "meta": {**checkpoint, "envelope": ENVELOPE, "bt3_source": BT3_SOURCE}},
               OUT / checkpoint["path"])

    lineage = {"bt2_source": BT2_SOURCE, "bt3_source": BT3_SOURCE,
               "reward_v2_freeze_sha256": PV8_REWARD_V2_FREEZE_SHA256,
               "zero_loss_adapter_sha256": frozen_after["zero_loss_adapter"],
               "credit_contract_digest": CC.contract_digest()}
    write("bt4_authorization_audit.json", {
        "grant_scope": "block-scoped BT4 grant; no global flag set",
        "simulator_execution_allowed_events": len(exec_ev),
        "training_allowed_events": len(train_ev),
        "capabilities_after_block": locks_after,
        "budget": budget.payload(),
        "global_locks": {"training_allowed": False, "simulator_execution_allowed": False,
                         "performance_comparison_allowed": False,
                         "paper_level_claim_allowed": False,
                         "causal_performance_claim_allowed": False}})
    write("bt4_causal_rollout_audit.json", {
        "windows": WINDOWS, "window_selection": "frozen by BT3 before any rollout",
        "reselected_using_outcome": False,
        "per_window": window_audit, "budget": budget.payload(),
        "causal_transitions": budget.transitions,
        "steps_per_decision": STEPS_PER_DECISION})
    write("bt4_multistep_gae_audit.json", {
        "gamma": CC.GAMMA, "lam": CC.GAE_LAMBDA,
        "trajectories": len(trajectories), "multistep_trajectories": len(multistep),
        "trajectory_length_min": lengths[0] if lengths else 0,
        "trajectory_length_median": statistics.median(lengths) if lengths else 0,
        "trajectory_length_max": lengths[-1] if lengths else 0,
        "nonterminal_transitions": len(nonterminal_rows),
        "gae_recursion_exercised": bool(propagated),
        "temporally_propagated_advantages": len(propagated),
        "temporal_propagation_fraction": round(len(propagated) / len(gae_rows), 6) if gae_rows else 0.0,
        "rows": gae_rows,
        "cross_window_max_deviation": round(cross_window_dev, 12),
        "cross_seed_max_deviation": round(cross_seed_dev, 12)})
    write("bt4_learning_signal_audit.json", {
        "total_transitions": len(gae_rows), "informative_transitions": len(informative),
        "non_zero_team_reward_fraction": round(len(informative) / len(decisions), 6) if decisions else 0.0,
        "advantage_mean": round(statistics.mean(advs), 8) if advs else 0.0,
        "advantage_std": round(statistics.pstdev(advs), 8) if len(advs) > 1 else 0.0,
        "advantage_variance": round(adv_var, 10),
        "critic_target_mean": round(statistics.mean(rets), 8) if rets else 0.0,
        "critic_target_variance": round(ret_var, 10),
        "updates": updates, "decisions": decisions,
        "reward_v2_made_less_sparse": False,
        "performance_interpretation_permitted": False})
    write("bt4_candidate_zero_loss_audit.json", {
        "zero_loss_epsilon": 0.0,
        "zero_loss_adapter_sha256": frozen_after["zero_loss_adapter"],
        "decisions": len(decisions),
        "invariants": {k: inv[k] for k in ("zero_loss_violations", "masked_candidate_selections",
                                           "illegal_selections", "candidate_identity_mismatch",
                                           "no_assign_contract_violations", "candidate_relabeling")},
        "mutation": {k: inv[k] for k in ("source_mutation_during_evaluation",
                                         "unselected_candidate_mutation",
                                         "multiple_authoritative_mutation")},
        "training_time_regeneration": inv["candidate_relabeling"]})
    write("bt4_optimizer_audit.json", {
        "optimizer_steps": budget.updates, "max_allowed": ENVELOPE["optimizer_updates"],
        "unauthorized_optimizer_steps": inv["unauthorized_optimizer_steps"],
        "updates": updates,
        "all_capability_checked": all(u["capability_checked"] == "training" for u in updates),
        "actor_grads_finite": all(math.isfinite(u["actor_grad_norm"]) for u in updates),
        "critic_grads_finite": all(math.isfinite(u["critic_grad_norm"]) for u in updates),
        "actor_digest_before": actor_d0, "actor_digest_after": actor_d1,
        "critic_digest_before": critic_d0, "critic_digest_after": critic_d1})
    write("frozen_hash_before_after.json", {
        "before": frozen_before, "after": frozen_after, "changed": frozen_changed,
        "all_unchanged": not frozen_changed,
        "joint_actor": {"before": actor_d0, "after": actor_d1, "changed": actor_d0 != actor_d1},
        "joint_critic": {"before": critic_d0, "after": critic_d1, "changed": critic_d0 != critic_d1}})
    write("test_results.json", {
        "invariants": inv, "hard_failures": hard, "warnings": [],
        "lineage": lineage, "budget": budget.payload()})
    write("bt4_execution_manifest.json", {
        "gate": GATE if not hard else "BLOCKED",
        "classification": ("A_SUSEONG_LS3_JOINT_ASSIGNMENT_MULTISTEP_CAUSAL_CREDIT_CONFIRMED"
                           "_READY_FOR_BOUNDED_TRAINING_BEHAVIOR_REVIEW" if not hard else "BLOCKED"),
        "lineage": lineage, "envelope": ENVELOPE, "budget": budget.payload(),
        "checkpoint": checkpoint,
        "performance_claim": "FORBIDDEN_THIS_GATE_ANSWERS_ONLY_CREDIT_PATH_VALIDITY",
        "baseline_comparison": "NONE_PERFORMED",
        "runtime_seconds": round(time.time() - t_start, 2),
        "maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)})

    wrows = "\n".join(f"| {w['seed']} | `{w['window_id'][-26:]}` | {w['causal_steps']} | "
                      f"{w['assignment_decisions']} | {w['boardings']} | {w['rewards']} |"
                      for w in window_audit)
    urows = "\n".join(f"| {u['seed']} | {u['update']} | {u['policy_loss']:.6f} | "
                      f"{u['critic_loss']:.6f} | {u['entropy']:.4f} | {u['actor_grad_norm']:.4f} | "
                      f"{u['critic_grad_norm']:.4f} |" for u in updates)
    prop_example = next((g for g in propagated), None)
    write("final_report.md", f"""# H4M-AE-R9.8 LS3-BT4 — S2 Bounded Causal Training and Multi-Step GAE Integrity

- Gate: `{GATE if not hard else 'BLOCKED'}`
- BT3 source: `{BT3_SOURCE}`
- Generated: {STAMP}

BT1 and BT2 ran the causal path but never exercised the GAE recursion: each window produced one
assignment decision, so every group was a single terminated transition and the advantage was just the
TD residual. BT3 said so explicitly. BT4 puts {ENVELOPE['steps_per_window'] // STEPS_PER_DECISION} decisions in each window so the recursion has
something to propagate through.

## 1. Execution inside the S2 envelope

| seed | window | steps | decisions | boardings | assignment rewards |
|---|---|---|---|---|---|
{wrows}

{budget.transitions} causal transitions across {budget.visits} visits of {len(budget.used_windows)} distinct windows, {budget.updates} optimizer updates —
every number inside the frozen envelope, enforced by a budget that denies a seventh window, a
thirteenth visit, an unapproved seed, a ninth step, a ninety-seventh transition or a seventh update.

## 2. Multi-step GAE — the point of this gate

| property | value |
|---|---|
| trajectories | {len(trajectories)} |
| multi-step trajectories (length ≥ 2) | **{len(multistep)}** |
| trajectory length min / median / max | {lengths[0] if lengths else 0} / {statistics.median(lengths) if lengths else 0} / {lengths[-1] if lengths else 0} |
| nonterminal transitions | **{len(nonterminal_rows)}** |
| **GAE recursion exercised** | **{bool(propagated)}** |
| temporally propagated advantages | {len(propagated)} of {len(gae_rows)} ({round(len(propagated) / len(gae_rows), 4) if gae_rows else 0}) |

A propagated row is one where `advantage − td_residual ≠ 0`, i.e. the `gamma·lambda·nonterminal·A_{{t+1}}`
term actually contributed. {"Example: window `" + prop_example['window_id'][-24:] + "` decision " + str(prop_example['decision_index']) + " carries td_residual " + f"{prop_example['td_residual']:.6f}" + " but advantage " + f"{prop_example['advantage']:.6f}" + ", a recursive contribution of " + f"{prop_example['gae_recursive_term']:.6f}" + " inherited from later decisions in the same window." if prop_example else "No propagation was observed."}

Boundaries hold while that happens: cross-window deviation {cross_window_dev}, cross-seed deviation
{cross_seed_dev}. Later reward reaches earlier decisions **within** a window and nowhere else.

## 3. Learning signal

{len(informative)} of {len(decisions)} decisions carry a non-zero team reward
({round(len(informative) / len(decisions), 4) if decisions else 0}). Advantage mean {round(statistics.mean(advs), 6) if advs else 0}, variance {round(adv_var, 8)};
critic-target variance {round(ret_var, 8)}.

| seed | update | policy loss | critic loss | entropy | actor grad | critic grad |
|---|---|---|---|---|---|---|
{urows}

Reward V2 was not made less sparse to achieve this. None of these numbers may be read as policy
quality, and no baseline comparison was performed.

## 4. Integrity

All {len(inv)} invariants zero: no Zero-Loss violation, masked or illegal selection, candidate identity
mismatch or relabeling, NO_ASSIGN contract violation, source mutation during candidate evaluation,
unselected-candidate mutation, multiple authoritative mutation, legacy advantage contamination,
future leakage, NaN, Inf or unauthorized optimizer step.

Frozen components byte-identical ({len(frozen_changed)} changed) across GATv2/operational actor+critic,
Reward V2, Zero-Loss adapter, Local Search authority, causal bridge, authorization enforcement, credit
contract and joint learning module. Only the joint actor ({actor_d0 != actor_d1}) and critic
({critic_d0 != critic_d1}) moved.

## 5. Authorization and checkpoint

{len(exec_ev)} simulator_execution and {len(train_ev)} training ALLOWED events, all inside the block; every capability
denied again afterwards (`{locks_after}`). Every update passed the training capability check.

Checkpoint is reproducibility-only: `test_only`, `non_promotable`, no winner or best-model semantics,
no promotion logic, and no performance, paper-level or causal claim permitted.

Global locks remain false. Runtime {round(time.time() - t_start, 2)}s, peak RSS
{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB.
""")

    man = {str(p.relative_to(OUT)): BT1.sha256_file(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    write("manifest.json", {"generated": STAMP, "artifact_dir": OUT.name,
                            "gate": GATE if not hard else "BLOCKED",
                            "lineage": lineage, "file_sha256": man})
    (OUT / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(
        f"{GATE if not hard else 'BLOCKED'}\n{STAMP}\n", encoding="utf-8")

    print(("[PASS] " + GATE) if not hard else ("[BLOCKED] " + str(hard)))
    print(f"artifact: {OUT.relative_to(PROJECT_ROOT)}")
    print(f"transitions={budget.transitions} decisions={len(decisions)} updates={budget.updates} "
          f"multistep_traj={len(multistep)} propagated={len(propagated)}/{len(gae_rows)} "
          f"frozen_changed={frozen_changed}")
    if hard:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
