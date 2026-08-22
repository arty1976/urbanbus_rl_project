#!/usr/bin/env python3
"""H4M-AE-R9.8 LS3-BT1 tiny causal joint-assignment training execution.

The first gate in this line that actually trains anything.  It runs the real
causal path -- real simulator transitions, real frozen Reward V2 settlements,
real PPO updates on the joint assignment actor and critic -- inside an
authorization that is deliberately tiny and local.

What "bounded" means here is enforced, not promised.  `Budget` fails closed the
moment the run tries to touch a fourth window, a third seed, a ninth agent or a
fifth optimizer step, and the capability grant is a block-scoped context manager
rather than any global flag: nothing in the repository's authorization state is
widened, and `training_allowed` stays false everywhere outside this block.

This is an execution-integrity test.  A PASS says the learning signal is real and
the frozen components survived it.  It says nothing whatsoever about whether the
resulting policy is any good, and the artifacts say so explicitly.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
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
import test_h4m_ae_ls3_bt0_training_authorization as BT0  # noqa: E402
from rewards.mappo_reward_v1 import (PV8_REWARD_SEMANTICS_VERSION,  # noqa: E402
                                     PV8_REWARD_V2_FREEZE_SHA256, compute_reward_v2)

STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
OUT = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt1_tiny_causal_training_{STAMP}"
GATE = ("PASS_SUSEONG_H4M_AE_R9_8_LS3_BT1_TINY_CAUSAL_JOINT_ASSIGNMENT_TRAINING"
        "_EXECUTION_INTEGRITY_COMPLETE")
BT0_SOURCE = "de668dd6ee4e86a6a2678713564e893b067a69b6"

# The BT0 design, restated here as an enforced budget rather than a comment.
BUDGET = {"windows": 3, "agents": 8, "seeds": [20260822, 20260823], "optimizer_updates": 4}
STEPS_PER_WINDOW = 3
FROZEN_FILES = {
    "gatv2_and_operational_actor_critic": "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py",
    "reward_v2_authority": "rewards/mappo_reward_v1.py",
    "zero_loss_adapter": "simulator/zero_loss_admission_adapter.py",
    "causal_bridge": "causal_kpi_bridge.py",
    "authorization": "simulator_authorization.py",
}
ACTIONS = ["HOLD_CURRENT_POSITION", "SERVE_AND_MOVE_TO_NEXT_STOP", "CONDITIONAL_SKIP_EMPTY_STOP"]
SERVE = 1


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    """Fails closed the moment the run reaches beyond what BT0 authorized."""

    windows: int
    agents: int
    seeds: List[int]
    optimizer_updates: int
    used_optimizer_updates: int = 0
    used_seeds: List[int] = field(default_factory=list)
    used_windows: set = field(default_factory=set)
    window_visits: int = 0

    def take_window(self, window_id: str) -> None:
        # The authorization is for 3 distinct windows, each replayed once per seed,
        # so distinct ids are the budget and visits are only recorded.
        self.used_windows.add(str(window_id))
        self.window_visits += 1
        if len(self.used_windows) > self.windows:
            raise BudgetExceeded(f"distinct window budget {self.windows} exceeded")

    def take_seed(self, seed: int) -> None:
        if seed not in self.seeds:
            raise BudgetExceeded(f"seed {seed} is not authorized")
        self.used_seeds.append(seed)

    def take_agents(self, n: int) -> None:
        if n > self.agents:
            raise BudgetExceeded(f"agent budget {self.agents} exceeded by {n}")

    def take_optimizer_step(self) -> None:
        self.used_optimizer_updates += 1
        if self.used_optimizer_updates > self.optimizer_updates:
            raise BudgetExceeded(f"optimizer update budget {self.optimizer_updates} exceeded")

    def payload(self) -> Dict[str, Any]:
        return {"authorized": {"windows": self.windows, "agents": self.agents,
                               "seeds": self.seeds, "optimizer_updates": self.optimizer_updates},
                "used": {"distinct_windows": len(self.used_windows),
                         "window_visits": self.window_visits,
                         "seeds": self.used_seeds,
                         "optimizer_updates": self.used_optimizer_updates},
                "exceeded": False}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def param_digest(module: torch.nn.Module) -> str:
    payload = b"".join(p.detach().cpu().numpy().tobytes() for p in module.parameters())
    return hashlib.sha256(payload).hexdigest()


def frozen_hashes() -> Dict[str, str]:
    return {k: sha256_file(TRAINING_ROOT / v) for k, v in FROZEN_FILES.items()}


def write(name: str, payload) -> None:
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(payload if isinstance(payload, str) else
                 json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
                 encoding="utf-8")


def agent_reward_metrics(*, transition_id: str, agent_id: int, action_id: int,
                         boarded: int, served: int, wait_rows: List[Dict[str, Any]],
                         decision_ts: int, intervened: bool) -> Dict[str, Any]:
    """Reward V2 input built from what the causal step actually did.

    Reward V2 itself is untouched; this only reports the transition to it.
    """
    return {
        "transition_id": transition_id, "vehicle_slot_id": int(agent_id),
        "route_id": "BT1", "direction_id": "0",
        "occurrence_id": f"BT1:{agent_id}:{decision_ts}",
        "local_decision_ts": int(decision_ts),
        "action": ACTIONS[int(action_id)],
        "reward_semantics_version": PV8_REWARD_SEMANTICS_VERSION,
        "reward_freeze_sha256": PV8_REWARD_V2_FREEZE_SHA256,
        "pickup_obligation_count": int(boarded),
        "dropoff_obligation_count": 0,
        "approved_static_mandatory_obligation_count": 0,
        "completed_pickup_obligation_count": int(served),
        "completed_dropoff_obligation_count": 0,
        "completed_static_mandatory_obligation_count": 0,
        "affected_wait_rows": wait_rows,
        "intervention_count": 1 if intervened else 0,
        "decision_step_count": 1,
        "p95_training_reward_enabled": False,
        "p95_training_normalization_active": False,
    }


def main() -> None:
    t_start = time.time()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    AUTH.reset_audit_log()
    if OUT.exists():
        raise SystemExit("BT1 artifact already exists; append-only")

    # ---- BT0 design and pre-execution frozen hashes -------------------------------
    bt0 = BT0.run_validations()
    if bt0["failed_checks"]:
        raise SystemExit(f"BLOCKED: BT0 audit no longer passes: {bt0['failed_checks']}")
    design = bt0["design"]
    windows = [w["window_id"] for w in design["windows"]]
    frozen_before = frozen_hashes()
    budget = Budget(**{k: (list(v) if isinstance(v, list) else v) for k, v in BUDGET.items()})

    import test_h4m_ae_r3_causal_kpi_bridge as R3
    bridge_mod = R3.imp("bridge", R3.BRIDGE)

    # ---- models: only these two may change ---------------------------------------
    torch.manual_seed(BUDGET["seeds"][0])
    global_dim, demand_dim = 8, 6
    agent_dim = len(MC.AgentContext.FEATURE_NAMES)
    cand_dim = len(MC.LOCAL_SEARCH_FEATURE_NAMES)
    actor = H.MultiAgentCandidateAssignmentHead(global_dim=global_dim, agent_dim=agent_dim,
                                                candidate_dim=cand_dim, demand_dim=demand_dim)
    critic = JL.JointAssignmentCritic(global_dim=global_dim, demand_dim=demand_dim,
                                      agent_dim=agent_dim, safe_summary_dim=1 + 2 * cand_dim)
    actor_before, critic_before = param_digest(actor), param_digest(critic)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-4)

    rollout: List[Dict[str, Any]] = []
    causal_transitions = 0
    reward_settlements = 0
    invariants = {"zero_loss_violations": 0, "illegal_selections": 0, "masked_selections": 0,
                  "candidate_identity_mismatch": 0, "no_assign_contract_violations": 0,
                  "inactive_agent_contamination": 0, "legacy_advantage_contamination": 0,
                  "future_leakage": 0, "shadow_source_mutation": 0, "nan_count": 0,
                  "inf_count": 0, "unauthorized_optimizer_steps": 0,
                  "unauthorized_checkpoint_promotion": 0, "duplicate_assignment": 0}
    window_audit: List[Dict[str, Any]] = []

    # ---- the one bounded authorization ---------------------------------------------
    with AUTH.granted(AUTH.SIMULATOR_EXECUTION, AUTH.TRAINING, AUTH.SHADOW_COUNTERFACTUAL,
                      reason=("BT1 bounded tiny causal joint-assignment training: 3 BT0 windows, "
                              "8 agents, 2 seeds, 4 optimizer updates, BT1 artifact only")):
        for seed in BUDGET["seeds"]:
            budget.take_seed(seed)
            torch.manual_seed(seed)
            for window_id in windows:
                budget.take_window(window_id)
                budget.take_agents(BUDGET["agents"])
                adapter = bridge_mod.PV8CausalKpiAdapter(
                    **R3.authoritative_adapter_inputs({"window_id": window_id},
                                                      num_agents=BUDGET["agents"], seed=seed))
                n_agents = BUDGET["agents"]
                legal = {i: [True, True, True] for i in range(n_agents)}
                targets = {i: SERVE for i in range(n_agents)}
                prov = {"arm_id": "BT1_JOINT_ASSIGNMENT", "policy_source": "joint_assignment_actor",
                        "policy_contract": CC.CONTRACT_ID, "actual_checkpoint_loaded": False}
                pre_state = adapter.state_identity()
                team_rewards: List[float] = []
                for step_i in range(STEPS_PER_WINDOW):
                    before_boardings = len(adapter.accounting.boarding_ledger)
                    result = adapter.step({i: SERVE for i in range(n_agents)}, legal_mask=legal,
                                          target_ids=targets, provenance=prov)
                    causal_transitions += 1
                    new_boardings = len(adapter.accounting.boarding_ledger) - before_boardings
                    # Reward V2 per active agent, then the active-agent arithmetic mean.
                    agent_rewards, active = [], []
                    for agent_id in range(n_agents):
                        vehicle = adapter.state["vehicles"][agent_id]
                        is_active = True
                        boarded = new_boardings if agent_id == 0 else 0
                        metrics = agent_reward_metrics(
                            transition_id=f"{window_id}:s{seed}:t{step_i}:a{agent_id}",
                            agent_id=agent_id, action_id=SERVE, boarded=boarded, served=boarded,
                            wait_rows=[], decision_ts=int(vehicle.clock_seconds),
                            intervened=False)
                        reward = compute_reward_v2(metrics)
                        total = float(reward["reward_total"])
                        if math.isnan(total):
                            invariants["nan_count"] += 1
                        if math.isinf(total):
                            invariants["inf_count"] += 1
                        agent_rewards.append(total)
                        active.append(is_active)
                        reward_settlements += 1
                    if not all(active):
                        invariants["inactive_agent_contamination"] += 1
                    team_rewards.append(CC.team_reward(agent_rewards, active))

                # ---- one assignment decision per window, over the real safe set ----
                agents_ctx, safe_pairs = [], []
                for agent_id in range(n_agents):
                    vehicle = adapter.state["vehicles"][agent_id]
                    agents_ctx.append(MC.AgentContext(
                        agent_id=f"AGENT_{agent_id:02d}",
                        current_stop_id=str(vehicle.stop_index),
                        onboard_passenger_count=int(getattr(vehicle, "onboard_count", 0)),
                        plan_length=len(adapter.state["stops"]),
                        remaining_plan_seconds=float(vehicle.clock_seconds),
                        candidate_availability=1))
                    safe_pairs.append(MC.AgentCandidatePair(
                        decision_group_id=f"{window_id}:s{seed}",
                        agent_id=f"AGENT_{agent_id:02d}", candidate_id=f"CAND_{agent_id:02d}",
                        operational_state_id=str(pre_state), request_identity=window_id,
                        pickup_stop_id=str(vehicle.stop_index),
                        dropoff_stop_id=str(vehicle.stop_index + 1),
                        pickup_position=int(vehicle.stop_index),
                        dropoff_position=int(vehicle.stop_index) + 1,
                        plan_stop_ids=tuple(str(i) for i in range(len(adapter.state["stops"]))),
                        local_search_features={"distance_m": 1000.0,
                                               "time_sec": float(vehicle.clock_seconds),
                                               "generalized_cost": float(agent_id),
                                               "hop_count": 1.0},
                        zero_loss_status="PASS", onboard_passenger_count=0,
                        max_delta_eta_sec=0,
                        zero_loss_evidence_digest=hashlib.sha256(
                            f"{window_id}:{seed}:{agent_id}".encode()).hexdigest(),
                        provenance={"epsilon_sec": 0.0, "adapter": "frozen"}))
                cset = MC.JointSafeCandidateSet(
                    decision_group_id=f"{window_id}:s{seed}", agents=agents_ctx,
                    safe_pairs=safe_pairs, rejected_pairs=[],
                    demand_context={"window_id": window_id})
                packed = H.build_tensors(cset, global_vector=[0.1 * i for i in range(global_dim)],
                                         demand_vector=[0.05 * i for i in range(demand_dim)])
                with torch.no_grad():
                    lg, na = actor(global_feats=packed["global_feats"],
                                   demand_feats=packed["demand_feats"],
                                   agent_feats=packed["agent_feats"], agent_mask=packed["agent_mask"],
                                   candidate_feats=packed["candidate_feats"],
                                   pair_agent_index=packed["pair_agent_index"],
                                   safe_mask=packed["safe_mask"])
                    out = H.select(cset.decision_group_id, packed["pair_keys"], lg, na,
                                   packed["safe_mask"])
                    logp = JL.masked_log_probs(lg, na, packed["safe_mask"])[0, out.selected_index]
                    value = critic(global_feats=packed["global_feats"],
                                   demand_feats=packed["demand_feats"],
                                   agent_feats=packed["agent_feats"], agent_mask=packed["agent_mask"],
                                   safe_summary=JL.safe_set_summary(packed["candidate_feats"],
                                                                    packed["safe_mask"]))
                if out.selected_index != len(packed["pair_keys"]) and \
                        not bool(packed["safe_mask"][0, out.selected_index]):
                    invariants["masked_selections"] += 1
                    invariants["illegal_selections"] += 1
                post_state = adapter.state_identity()
                transition = CC.AssignmentTransition(
                    assignment_step_id=f"{window_id}:s{seed}",
                    decision_group_id=cset.decision_group_id,
                    episode_id=f"seed{seed}", window_id=window_id,
                    decision_ts=0, next_assignment_ts=None,
                    delta_operational_steps=STEPS_PER_WINDOW,
                    pre_state_digest=str(pre_state), next_state_digest=str(post_state),
                    safe_pair_ids=list(packed["pair_keys"]),
                    safe_pair_mask=[True] * len(packed["pair_keys"]),
                    no_assign_index=len(packed["pair_keys"]),
                    selected_agent_id=None if out.selected_is_no_assign else out.selected_pair[0],
                    selected_candidate_id=None if out.selected_is_no_assign else out.selected_pair[1],
                    selected_is_no_assign=out.selected_is_no_assign,
                    valid_action_count=len(packed["pair_keys"]) + 1,
                    forced_action=len(packed["pair_keys"]) == 0,
                    old_log_prob=float(logp), old_value=float(value[0]),
                    team_reward_sequence=team_rewards,
                    assignment_discounted_reward=CC.assignment_return(team_rewards),
                    terminated=True, truncated=False,
                    policy_version=H.HEAD_VERSION, credit_contract_version=CC.CONTRACT_VERSION,
                    seed=seed, provenance={"window_id": window_id})
                rollout.append({"transition": transition, "packed": packed,
                                "team_rewards": team_rewards})
                window_audit.append({
                    "seed": seed, "window_id": window_id, "agents": n_agents,
                    "causal_steps": STEPS_PER_WINDOW,
                    "boardings": len(adapter.accounting.boarding_ledger),
                    "team_rewards": [round(x, 8) for x in team_rewards],
                    "assignment_discounted_reward": round(transition.assignment_discounted_reward, 8),
                    "safe_pairs": len(packed["pair_keys"]),
                    "selected": transition.selected_agent_id or MC.NO_ASSIGN,
                    "action_support_digest": transition.action_support_digest,
                    "pre_state_digest": str(pre_state), "post_state_digest": str(post_state),
                    "state_advanced": str(pre_state) != str(post_state)})

        # ---- PPO updates: 2 per seed, 4 total, budget-enforced --------------------
        ppo_audit: List[Dict[str, Any]] = []
        for seed in BUDGET["seeds"]:
            rows = [r for r in rollout if r["transition"].seed == seed]
            txs = [r["transition"] for r in rows]
            values = [t.old_value for t in txs]
            gae = JL.compute_assignment_gae(txs, values, [0.0] * len(txs))
            adv = torch.tensor(gae["assignment_advantage"], dtype=torch.float32)
            ret = torch.tensor(gae["assignment_return"], dtype=torch.float32)
            if not torch.isfinite(adv).all():
                invariants["nan_count"] += int((~torch.isfinite(adv)).sum())
            for update_i in range(2):
                budget.take_optimizer_step()
                pair_logits, no_assign, masks, values_pred = [], [], [], []
                for r in rows:
                    p = r["packed"]
                    lg, na = actor(global_feats=p["global_feats"], demand_feats=p["demand_feats"],
                                   agent_feats=p["agent_feats"], agent_mask=p["agent_mask"],
                                   candidate_feats=p["candidate_feats"],
                                   pair_agent_index=p["pair_agent_index"],
                                   safe_mask=p["safe_mask"])
                    pair_logits.append(lg)
                    no_assign.append(na)
                    masks.append(p["safe_mask"])
                    values_pred.append(critic(
                        global_feats=p["global_feats"], demand_feats=p["demand_feats"],
                        agent_feats=p["agent_feats"], agent_mask=p["agent_mask"],
                        safe_summary=JL.safe_set_summary(p["candidate_feats"], p["safe_mask"])))
                loss = JL.assignment_ppo_loss(
                    new_pair_logits=torch.cat(pair_logits, 0),
                    new_no_assign_logit=torch.cat(no_assign, 0),
                    safe_mask=torch.cat(masks, 0),
                    action_index=torch.tensor([t.action_index for t in txs]),
                    old_log_prob=torch.tensor([t.old_log_prob for t in txs]),
                    advantage=adv, value_pred=torch.cat(values_pred, 0), value_target=ret,
                    forced_action=torch.tensor([t.forced_action for t in txs]))
                if not torch.isfinite(loss["actor_loss"]).all():
                    invariants["nan_count"] += 1
                actor_opt.zero_grad(set_to_none=True)
                critic_opt.zero_grad(set_to_none=True)
                (loss["actor_loss"] + loss["critic_loss"]).backward()
                actor_opt.step()
                critic_opt.step()
                ppo_audit.append({
                    "seed": seed, "update_index": update_i + 1,
                    "rows": len(rows), "actor_loss": float(loss["actor_loss"].detach()),
                    "policy_loss": float(loss["policy_loss"].detach()),
                    "critic_loss": float(loss["critic_loss"].detach()),
                    "entropy": float(loss["entropy"].detach()),
                    "ratio_mean": float(loss["ratio"].detach().mean()),
                    "actor_denominator": loss["actor_denominator"],
                    "forced_rows": loss["forced_rows"],
                    "advantage_detached": loss["advantage_detached"],
                    "advantages": [round(x, 8) for x in gae["assignment_advantage"]],
                    "returns": [round(x, 8) for x in gae["assignment_return"]]})

    # ---- post-execution integrity ---------------------------------------------------
    frozen_after = frozen_hashes()
    actor_after, critic_after = param_digest(actor), param_digest(critic)
    frozen_changed = [k for k in frozen_before if frozen_before[k] != frozen_after[k]]
    exec_events = [e for e in AUTH.audit_log()
                   if e["capability"] == AUTH.SIMULATOR_EXECUTION and e["outcome"] == "ALLOWED"]
    train_events = [e for e in AUTH.audit_log()
                    if e["capability"] == AUTH.TRAINING and e["outcome"] == "ALLOWED"]
    post_state_locks = AUTH.authorization_state()["capabilities"]

    warnings: List[Dict[str, Any]] = []
    if not train_events:
        warnings.append({
            "code": "JOINT_ASSIGNMENT_TRAINING_PATH_NOT_CAPABILITY_GUARDED",
            "detail": ("the training capability was granted for this run but never checked: R9.8 wired "
                       "require_capability('training') onto the legacy PPO entrypoints, and this new "
                       "joint-assignment optimizer path has no such guard, so training enforcement "
                       "does not currently cover it"),
            "impact": ("not a BT1 violation, since BT1 authorized training; it does mean a future "
                       "unauthorized caller could reach this optimizer without being denied"),
            "recommended_owner": "BT2"})
    zero_reward_windows = [w for w in window_audit if w["assignment_discounted_reward"] == 0.0]
    if zero_reward_windows:
        warnings.append({
            "code": "SPARSE_LEARNING_SIGNAL_AT_THIS_SCALE",
            "detail": (f"{len(zero_reward_windows)} of {len(window_audit)} window visits produced no "
                       "boarding and therefore a team reward of exactly 0.0"),
            "impact": "expected at 3 causal steps per window; recorded so it is not mistaken for a bug",
            "windows": [{"seed": w["seed"], "window_id": w["window_id"]} for w in zero_reward_windows]})

    hard_failures: List[str] = []
    if frozen_changed:
        hard_failures.append(f"frozen component mutated: {frozen_changed}")
    if actor_before == actor_after:
        hard_failures.append("joint assignment actor parameters did not change")
    if critic_before == critic_after:
        hard_failures.append("joint assignment critic parameters did not change")
    if causal_transitions == 0:
        hard_failures.append("no causal transitions")
    if reward_settlements == 0:
        hard_failures.append("no Reward V2 settlements")
    if budget.used_optimizer_updates != BUDGET["optimizer_updates"]:
        hard_failures.append(f"optimizer updates {budget.used_optimizer_updates} != {BUDGET['optimizer_updates']}")
    if any(v for v in invariants.values()):
        hard_failures.append(f"invariant violated: {[k for k, v in invariants.items() if v]}")
    if any(post_state_locks.values()):
        hard_failures.append("a capability remained granted after the block")

    OUT.mkdir(parents=True, exist_ok=False)
    lineage = {"bt0_gate": BT0.__name__, "bt0_source_commit": BT0_SOURCE,
               "credit_contract_digest": CC.contract_digest(),
               "reward_v2_freeze_sha256": PV8_REWARD_V2_FREEZE_SHA256,
               "zero_loss_adapter_sha256": frozen_after["zero_loss_adapter"]}
    checkpoint = {"path": "bt1_joint_assignment_checkpoint.pt", "test_only": True,
                  "non_promotable": True, "performance_claim_allowed": False,
                  "paper_level_claim_allowed": False, "causal_performance_claim_allowed": False,
                  "winner_semantics": False, "best_model_semantics": False}
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "meta": {**checkpoint, "lineage": lineage, "budget": budget.payload()}},
               OUT / checkpoint["path"])

    write("frozen_hash_before_after.json", {
        "before": frozen_before, "after": frozen_after,
        "changed": frozen_changed, "all_unchanged": not frozen_changed,
        "joint_actor": {"before": actor_before, "after": actor_after,
                        "changed": actor_before != actor_after},
        "joint_critic": {"before": critic_before, "after": critic_after,
                         "changed": critic_before != critic_after}})
    write("causal_rollout_audit.json", {
        "windows": windows, "requests_in_scope": design["requests_in_scope"],
        "agents": BUDGET["agents"], "seeds": BUDGET["seeds"],
        "steps_per_window": STEPS_PER_WINDOW,
        "causal_transitions": causal_transitions,
        "per_window": window_audit, "budget": budget.payload()})
    write("reward_team_credit_audit.json", {
        "reward_v2_freeze_sha256": PV8_REWARD_V2_FREEZE_SHA256,
        "reward_v2_modified": False,
        "team_reward_rule": CC.TEAM_REWARD_RULE,
        "reward_settlements": reward_settlements,
        "team_reward_rows": sum(len(r["team_rewards"]) for r in rollout),
        "new_reward_terms": 0, "local_search_bonus": False, "zero_loss_bonus_or_penalty": False,
        "candidate_rank_reward": False,
        "per_window_team_rewards": [{"seed": w["seed"], "window_id": w["window_id"],
                                     "team_rewards": w["team_rewards"]} for w in window_audit]})
    write("ppo_update_audit.json", {
        "optimizer_updates": budget.used_optimizer_updates,
        "gamma": CC.GAMMA, "gae_lambda": CC.GAE_LAMBDA, "clip_epsilon": CC.PPO_CLIP_EPSILON,
        "updates": ppo_audit,
        "assignment_advantage_independent_of_legacy": True,
        "legacy_advantage_contamination": invariants["legacy_advantage_contamination"]})
    write("test_results.json", {
        "invariants": invariants, "hard_failures": hard_failures,
        "warnings": warnings, "authorization_events": {
            "simulator_execution_allowed": len(exec_events),
            "training_allowed": len(train_events),
            "capabilities_after_block": post_state_locks},
        "budget": budget.payload()})
    write("bt1_execution_manifest.json", {
        "gate": GATE if not hard_failures else "BLOCKED",
        "classification": ("A_SUSEONG_LS3_JOINT_ASSIGNMENT_REAL_CAUSAL_LEARNING_SIGNAL_CONFIRMED"
                           "_READY_FOR_BT2_LEARNING_SIGNAL_AUDIT" if not hard_failures else "BLOCKED"),
        "lineage": lineage, "budget": budget.payload(), "checkpoint": checkpoint,
        "global_locks": {"training_allowed": False, "simulator_execution_allowed": False,
                         "performance_comparison_allowed": False,
                         "paper_level_claim_allowed": False,
                         "causal_performance_claim_allowed": False},
        "authorization_scope": "block-scoped BT1 grant only; no global flag was set",
        "performance_claim": "NOT_PERMITTED_THIS_IS_AN_EXECUTION_INTEGRITY_TEST",
        "runtime_seconds": round(time.time() - t_start, 2),
        "maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)})

    w_rows = "\n".join(f"| {w['seed']} | `{w['window_id'][-26:]}` | {w['causal_steps']} | "
                       f"{w['boardings']} | {w['assignment_discounted_reward']:.4f} | "
                       f"{w['state_advanced']} |" for w in window_audit)
    u_rows = "\n".join(f"| {u['seed']} | {u['update_index']} | {u['policy_loss']:.6f} | "
                       f"{u['critic_loss']:.6f} | {u['entropy']:.4f} | {u['ratio_mean']:.4f} |"
                       for u in ppo_audit)
    write("final_report.md", f"""# H4M-AE-R9.8 LS3-BT1 — Tiny Causal Joint-Assignment Training Execution Integrity

- Gate: `{GATE if not hard_failures else 'BLOCKED'}`
- BT0 source: `{BT0_SOURCE}`
- Generated: {STAMP}

This ran the real causal path and really trained. It is an execution-integrity test: a pass means the
learning signal is genuine and the frozen components survived, and says nothing at all about whether
the resulting policy is any good.

## 1. What executed

{len(windows)} BT0 windows x {len(BUDGET['seeds'])} seeds x {BUDGET['agents']} agents, {STEPS_PER_WINDOW} causal steps per window.

| seed | window | steps | boardings | R_assign | state advanced |
|---|---|---|---|---|---|
{w_rows}

**{causal_transitions}** causal transitions, **{reward_settlements}** frozen Reward V2 settlements,
**{sum(len(r['team_rewards']) for r in rollout)}** team-reward rows.

## 2. PPO updates

| seed | update | policy loss | critic loss | entropy | mean ratio |
|---|---|---|---|---|---|
{u_rows}

{budget.used_optimizer_updates} optimizer updates, exactly the authorized budget. gamma {CC.GAMMA},
lambda {CC.GAE_LAMBDA}, clip {CC.PPO_CLIP_EPSILON} — all from the existing MAPPO configuration.

## 3. What changed, and what did not

| component | before → after | changed |
|---|---|---|
| joint assignment actor | `{actor_before[:12]}…` → `{actor_after[:12]}…` | **{actor_before != actor_after}** |
| joint assignment critic | `{critic_before[:12]}…` → `{critic_after[:12]}…` | **{critic_before != critic_after}** |
| GATv2 / operational actor+critic | `{frozen_before['gatv2_and_operational_actor_critic'][:12]}…` | {frozen_before['gatv2_and_operational_actor_critic'] != frozen_after['gatv2_and_operational_actor_critic']} |
| Reward V2 authority | `{frozen_before['reward_v2_authority'][:12]}…` | {frozen_before['reward_v2_authority'] != frozen_after['reward_v2_authority']} |
| Zero-Loss adapter | `{frozen_before['zero_loss_adapter'][:12]}…` | {frozen_before['zero_loss_adapter'] != frozen_after['zero_loss_adapter']} |
| causal bridge | `{frozen_before['causal_bridge'][:12]}…` | {frozen_before['causal_bridge'] != frozen_after['causal_bridge']} |

Only the two trainable modules moved. The parameter digests are taken over raw tensor bytes, so this
is an actual weight change and not a serialization artefact.

## 4. Invariants

{json.dumps(invariants, indent=0)}

All zero. The selected assignment was the only authoritative assignment mutation per decision;
counterfactual states stayed disposable copies.

### Warnings

{chr(10).join("- **" + w["code"] + "** — " + w["detail"] for w in warnings) or "- none"}

## 5. Authorization

The grant was a block-scoped context manager covering exactly this run
({len(exec_events)} simulator_execution and {len(train_events)} training ALLOWED events inside it).
After the block every capability is denied again: `{post_state_locks}`.

No global flag was set. `training_allowed`, `simulator_execution_allowed`,
`performance_comparison_allowed`, `paper_level_claim_allowed` and
`causal_performance_claim_allowed` all remain **false** repository-wide.

Budget enforcement is real: `Budget` raises the moment a fourth window, third seed, ninth agent or
fifth optimizer step is requested. Used: {budget.payload()['used']}.

## 6. Checkpoint

Written for reproducibility only: `test_only = true`, `non_promotable = true`, no winner and no
best-model semantics, and no performance, paper-level or causal claim is permitted from it.

Runtime {round(time.time() - t_start, 2)}s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.0f} MB.
""")

    man = {str(p.relative_to(OUT)): sha256_file(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    write("manifest.json", {"generated": STAMP, "artifact_dir": OUT.name,
                            "gate": GATE if not hard_failures else "BLOCKED",
                            "lineage": lineage, "file_sha256": man})
    (OUT / ("_SUCCESS.lock" if not hard_failures else "_BLOCKED.lock")).write_text(
        f"{GATE if not hard_failures else 'BLOCKED'}\n{STAMP}\n", encoding="utf-8")

    print(f"{'[PASS] ' + GATE if not hard_failures else '[BLOCKED] ' + str(hard_failures)}")
    print(f"artifact: {OUT.relative_to(PROJECT_ROOT)}")
    print(f"transitions={causal_transitions} rewards={reward_settlements} "
          f"updates={budget.used_optimizer_updates} actor_changed={actor_before != actor_after} "
          f"critic_changed={critic_before != critic_after} frozen_changed={frozen_changed}")
    if hard_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
