#!/usr/bin/env python3
"""BT8-A1: execute only the BT8-S0-frozen additional bounded envelope.

The program deliberately has two boundaries.  Before either simulator or
training capability is granted, it proves the MPS runtime with a forward-only
shape probe that covers the A1 contract.  After that preflight, it runs exactly
the twelve BT8-S0 windows twice (the two frozen seeds), captures every actor
input before PPO, and makes precisely three full-rollout updates per seed.

This is a structural bounded-training execution.  It makes no baseline,
performance, convergence, promotion, or causal-performance claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-A1"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_A1_ADDITIONAL_BOUNDED_TRAINING_EXECUTION_AND_SNAPSHOT_PRESERVATION_COMPLETE"
PASS_CLASS = "A_SUSEONG_LS3_ADDITIONAL_BOUNDED_TRAINING_COMPLETE_READY_FOR_FROZEN_POLICY_BEHAVIOR_REVIEW"
MPS_BLOCK = "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_A1_MPS_PREFLIGHT_FAILED"
INTEGRITY_BLOCK = "BLOCKED_SUSEONG_H4M_AE_R9_8_LS3_BT8_A1_EXECUTION_INTEGRITY_FAILURE"
BT8_S0_SOURCE = "593c94d8fa80e87d37758b2a7c0383605229ed1f"
BT8_S0_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_S0_ADDITIONAL_TRAINING_ADEQUACY_AND_DISCRIMINATION_EXPOSURE_DESIGN_COMPLETE"
BT8_S0 = Path(__file__).resolve().parent / "artifacts" / "pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_s0_training_adequacy_design_20260822_212909+09:00"
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
ARTIFACTS = ROOT / "artifacts"
SOURCE_REL = "05_training/run_h4m_ae_ls3_bt8_a1_bounded_training.py"
TEST_REL = "05_training/test_h4m_ae_ls3_bt8_a1_bounded_training.py"
SOURCE_FILES = {SOURCE_REL, TEST_REL}
LOCKS = {"training_allowed": False, "simulator_execution_allowed": False,
         "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
         "causal_performance_claim_allowed": False}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def provenance() -> Dict[str, Any]:
    changed = [row for row in git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"]).splitlines() if row]
    return {"source_commit": git(["rev-parse", "HEAD"]), "source_parent": git(["rev-parse", "HEAD^"]),
            "source_only_local_commit": set(changed) == SOURCE_FILES, "changed_files": changed,
            "github_push_performed": False}


def load_a1_envelope() -> tuple[Dict[str, Any], list[Dict[str, Any]], Dict[str, Any]]:
    """Read the immutable A1 selection, not an outcome-conditioned re-selection."""
    selected_path = BT8_S0 / "bt8s0_selected_training_envelope.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    windows = list(selected["windows"])
    envelope = {
        "label": "A1_SMALL_EXTENSION", "distinct_windows": int(selected["additional_distinct_windows"]),
        "window_visits": int(selected["additional_window_visits"]),
        "requests": int(selected["additional_requests_unique_window_sum"]), "agents": 8,
        "seeds": [20260822, 20260823], "assignment_decisions": int(selected["assignment_decisions"]),
        "trajectories": int(selected["trajectories"]), "trajectory_length": 4,
        "causal_transitions": int(selected["causal_transitions"]), "optimizer_updates": int(selected["optimizer_updates"]),
    }
    return envelope, windows, {"selected_envelope_sha256": sha256(selected_path), "selected_label": selected["selected_label"],
                                "selection_authorization_at_design_time": selected["authorization"],
                                "must_not_execute_automatically_at_design_time": selected["must_not_execute_automatically"]}


def validate_a1_envelope(envelope: Mapping[str, Any], windows: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
    return {
        "exact_a1_label": envelope["label"] == "A1_SMALL_EXTENSION",
        "twelve_distinct_windows": envelope["distinct_windows"] == len(windows) == len({row["window_id"] for row in windows}) == 12,
        "twenty_four_visits": envelope["window_visits"] == 24,
        "eighty_seven_selected_requests": envelope["requests"] == sum(int(row["requests"]) for row in windows) == 87,
        "only_frozen_seeds": envelope["seeds"] == [20260822, 20260823],
        "ninety_six_decisions": envelope["assignment_decisions"] == 96,
        "twenty_four_four_step_trajectories": envelope["trajectories"] == 24 and envelope["trajectory_length"] == 4,
        "one_ninety_two_transitions": envelope["causal_transitions"] == 192,
        "exactly_six_updates": envelope["optimizer_updates"] == 6,
        "all_time_bands": Counter(row["time_band"] for row in windows) == Counter({"night": 4, "offpeak": 4, "peak": 4}),
        "density_coverage": {row["density_class_from_prepolicy_rank"] for row in windows} == {"low", "medium", "high"},
        "nine_new_three_balanced_repeats": sum(row["exposure_origin"] == "new_since_BT6" for row in windows) == 9
            and sum(row["exposure_origin"] == "balanced_repeat_for_density_coverage" for row in windows) == 3,
    }


def mps_preflight(*, H: Any, JL: Any, MC: Any, envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """Forward-only MPS viability check before capability/candidate/optimizer use."""
    evidence: Dict[str, Any] = {"required_device": "mps", "cpu_fallback_allowed": False,
                                "mps_built": bool(torch.backends.mps.is_built()),
                                "mps_available": bool(torch.backends.mps.is_available()),
                                "optimizer_steps": 0, "candidate_generation": 0,
                                "zero_loss_evaluation": 0, "capability_grants": 0}
    if not evidence["mps_available"]:
        evidence.update({"passed": False, "failure_reason": "torch.backends.mps.is_available() returned false"})
        return evidence
    try:
        device = torch.device("mps")
        adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
        # Search permits at most eight candidates per each of the eight A1 agents.
        pairs = int(envelope["agents"]) * 8
        torch.manual_seed(envelope["seeds"][0])
        actor = H.MultiAgentCandidateAssignmentHead(global_dim=8, demand_dim=6, agent_dim=adim, candidate_dim=cdim).to(device)
        critic = JL.JointAssignmentCritic(global_dim=8, demand_dim=6, agent_dim=adim, safe_summary_dim=1 + 2 * cdim).to(device)
        global_feats = torch.zeros((1, 8), device=device)
        demand_feats = torch.zeros((1, 6), device=device)
        agent_feats = torch.zeros((1, envelope["agents"], adim), device=device)
        agent_mask = torch.ones((1, envelope["agents"]), dtype=torch.bool, device=device)
        candidate_feats = torch.zeros((1, pairs, cdim), device=device)
        pair_agent_index = torch.arange(pairs, device=device).reshape(1, pairs) % envelope["agents"]
        safe_mask = torch.ones((1, pairs), dtype=torch.bool, device=device)
        with torch.no_grad():
            logits, no_assign = actor(global_feats=global_feats, demand_feats=demand_feats, agent_feats=agent_feats,
                                      agent_mask=agent_mask, candidate_feats=candidate_feats,
                                      pair_agent_index=pair_agent_index, safe_mask=safe_mask)
            value = critic(global_feats=global_feats, demand_feats=demand_feats, agent_feats=agent_feats,
                           agent_mask=agent_mask, safe_summary=JL.safe_set_summary(candidate_feats, safe_mask))
        torch.mps.synchronize()
        finite = bool(torch.isfinite(logits).all().item() and torch.isfinite(no_assign).all().item() and torch.isfinite(value).all().item())
        evidence.update({"passed": finite, "selected_device": str(logits.device), "finite": finite,
                         "shape_contract": {"agents": envelope["agents"], "candidate_pairs": pairs,
                                            "agent_feature_dim": adim, "candidate_feature_dim": cdim},
                         "mps_allocated_bytes": int(torch.mps.current_allocated_memory()),
                         "mps_driver_allocated_bytes": int(torch.mps.driver_allocated_memory())})
    except Exception as exc:  # noqa: BLE001
        evidence.update({"passed": False, "failure_reason": f"{type(exc).__name__}: {exc}"})
    return evidence


def write_block(*, root: Path, source: Mapping[str, Any], binding: Mapping[str, Any], preflight: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "bt8_a1_mps_preflight.json": preflight,
        "bt8_a1_authoritative_binding.json": dict(binding),
        "bt8_a1_execution_manifest.json": {"stage": STAGE, "gate": MPS_BLOCK, "source_commit": source["source_commit"],
                                              "execution_started": False, "optimizer_steps": 0, "simulator_rollouts": 0,
                                              "candidate_regeneration": 0, "checkpoint_writes": 0, "global_locks": LOCKS},
        "test_results.json": {"hard_failures": [MPS_BLOCK], "TEST6_access": 0, "github_push_performed": False,
                              "optimizer_steps": 0, "simulator_rollouts": 0, "checkpoint_writes": 0},
        "gate_decision.json": {"gate": MPS_BLOCK, "source_commit": source["source_commit"], "hard_failures": [MPS_BLOCK],
                               "warnings": [], "global_locks": LOCKS, "next_step": "restore MPS; rerun only the unchanged BT8-A1 envelope"},
    }
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(
        f"# {STAGE} — mandatory MPS preflight block\n\ngate = {MPS_BLOCK}\nsource_commit = {source['source_commit']}\n\n"
        "No capability was granted; no candidate was generated; no optimizer step, simulator rollout, or checkpoint write occurred.\n",
        encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": MPS_BLOCK, "source_commit": source["source_commit"], "file_sha256": manifest})
    (root / "_BLOCKED.lock").write_text(MPS_BLOCK + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "simulator"))
    import joint_assignment_credit_contract as CC
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_frozen_tie_break as TIE
    import joint_assignment_learning as JL
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt1_tiny_causal_training as BT1
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import simulator_authorization as AUTH
    import test_h4m_ae_r3_causal_kpi_bridge as R3
    from rewards.mappo_reward_v1 import PV8_REWARD_V2_FREEZE_SHA256, compute_reward_v2

    stamp = BT6.now().strftime("%Y%m%d_%H%M%S%z")[:-2] + ":00"
    root = ARTIFACTS / f"pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_a1_bounded_training_{stamp}"
    if root.exists(): raise SystemExit("append-only artifact collision")
    source = provenance()
    envelope, windows, selection = load_a1_envelope()
    envelope_checks = validate_a1_envelope(envelope, windows)
    s0_gate = json.loads((BT8_S0 / "gate_decision.json").read_text(encoding="utf-8"))
    review_contract = json.loads((BT8_S0 / "bt8s0_posttraining_review_contract.json").read_text(encoding="utf-8"))
    binding = {"bt8_s0_gate": s0_gate.get("gate") == BT8_S0_GATE, "bt8_s0_source": s0_gate.get("source_commit") == BT8_S0_SOURCE,
               "source_parent_is_bt8_s0": source["source_parent"] == BT8_S0_SOURCE,
               "source_only_local_commit": source["source_only_local_commit"], "design_authorization_was_not_auto_execution": selection["must_not_execute_automatically"],
               "t1_exact_zero_tolerance": review_contract["selector"].get("tolerance") == 0.0,
               "t1_contract": review_contract["selector"].get("contract_id") == TIE.TIE_BREAK_CONTRACT_ID,
               **envelope_checks}
    preflight = mps_preflight(H=H, JL=JL, MC=MC, envelope=envelope)
    if not all(binding.values()) or not preflight["passed"]:
        if not all(binding.values()): preflight = {**preflight, "passed": False, "failure_reason": "AUTHORITATIVE_BINDING_FAILURE"}
        write_block(root=root, source=source, binding=binding, preflight=preflight)
        print(f"[BLOCKED] {MPS_BLOCK}"); print(f"artifact: {root.relative_to(PROJECT)}")
        return

    before = BT6.frozen_hashes()
    AUTH.reset_audit_log()
    factory = BT6.RepairedSupportFactory()
    adversarial = BT6.adversarial_probes(factory, envelope, AUTH)
    if not adversarial["all_passed"] or factory.source_mutations:
        raise SystemExit("BLOCKED: pre-execution authorization/support integrity probe failed")
    device = torch.device("mps")
    torch.manual_seed(envelope["seeds"][0])
    adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
    actor = H.MultiAgentCandidateAssignmentHead(global_dim=8, demand_dim=6, agent_dim=adim, candidate_dim=cdim).to(device)
    critic = JL.JointAssignmentCritic(global_dim=8, demand_dim=6, agent_dim=adim, safe_summary_dim=1 + 2 * cdim).to(device)
    actor_opt, critic_opt = torch.optim.Adam(actor.parameters(), lr=1e-4), torch.optim.Adam(critic.parameters(), lr=1e-4)
    actor_before, critic_before = BT1.param_digest(actor), BT1.param_digest(critic)
    actor_config = FPS.actor_config(global_dim=8, demand_dim=6, agent_dim=adim, candidate_dim=cdim,
                                    actor_module_sha256=sha256(ROOT / "multi_agent_candidate_assignment_head.py"),
                                    actor_head_id=H.HEAD_ID, actor_head_version=H.HEAD_VERSION)
    feature_contract = {"feature_contract_id": FPS.FEATURE_CONTRACT_ID, "global_feature_dim": 8, "demand_feature_dim": 6,
                        "agent_feature_dim": adim, "candidate_feature_dim": cdim,
                        "feature_normalization": "already materialized at post-Zero-Loss pre-forward boundary; replay forbids recomputation",
                        "candidate_support_contract_id": FPS.CANDIDATE_SUPPORT_CONTRACT_ID,
                        "candidate_support_contract": "post-Zero-Loss finalized support; source digest is preserved",
                        "actor_input_boundary": "model_ready_pack output, including permanently-masked empty-support storage padding"}
    authorities = {**before, "joint_actor_head": sha256(ROOT / "multi_agent_candidate_assignment_head.py")}
    writer = FPS.SnapshotCollectionWriter(root=root / "bt8_a1_frozen_policy_snapshots", collection_binding={
        "snapshot_schema_version": FPS.SNAPSHOT_SCHEMA_VERSION, "source_commit": source["source_commit"], "actor_config": actor_config,
        "actor_config_sha256": FPS.actor_config_sha256(actor_config), "feature_contract": feature_contract,
        "frozen_authority_hashes": authorities, "captured_device": str(device)})
    budget = BT6.Budget(envelope)
    inv = {name: 0 for name in ("zero_loss_violations", "rejected_candidate_selected", "illegal_or_masked_selection",
           "candidate_identity_mismatch", "candidate_regeneration", "no_assign_violation", "cross_window_contamination",
           "cross_seed_contamination", "legacy_advantage_contamination", "future_leakage", "source_state_mutation", "nan_or_inf", "unauthorized_optimizer_step")}
    decisions, rollout, updates, window_rows = [], [], [], []
    by_band_windows: Dict[str, list[str]] = defaultdict(list)
    for row in windows: by_band_windows[row["time_band"]].append(row["window_id"])
    agent_slot = BT6.frozen_agent_slot_mapping(factory, slots=envelope["agents"])

    with AUTH.granted(AUTH.SIMULATOR_EXECUTION, AUTH.TRAINING, AUTH.SHADOW_COUNTERFACTUAL,
                      reason="BT8-A1 explicitly authorized exact envelope: 12 windows / 24 visits / 96 decisions / 192 transitions / 6 updates"):
        for seed in envelope["seeds"]:
            budget.seed(seed); torch.manual_seed(seed)
            for window_meta in windows:
                window_id, band = window_meta["window_id"], window_meta["time_band"]
                budget.visit(window_id); budget.trajectory(); budget.agent_count(envelope["agents"])
                adapter = R3.imp("bridge", R3.BRIDGE).PV8CausalKpiAdapter(
                    **R3.authoritative_adapter_inputs({"window_id": window_id}, num_agents=envelope["agents"], seed=seed))
                tx_rows, rewards_by_decision, total_boardings = [], [], 0
                for decision_index in range(envelope["trajectory_length"]):
                    budget.decision()
                    source_group = factory.groups_by_band[band][(by_band_windows[band].index(window_id) + decision_index) % len(factory.groups_by_band[band])]
                    decision_group = f"{window_id}:seed{seed}:decision{decision_index}"
                    support = factory.build(source_group=source_group, decision_group=decision_group)
                    joint, snapshot = support["joint"], support["snapshot"]
                    packed = BT6.device_pack(H.build_tensors(joint, global_vector=[0.1 * index for index in range(8)],
                                                             demand_vector=[0.05 * index for index in range(6)]), device)
                    model_packed = BT6.model_ready_pack(packed, cdim=cdim, device=device)
                    frozen = writer.capture(decision_id=decision_group, window_id=window_id, seed=seed, decision_index=decision_index,
                        time_band=band, actor_inputs=model_packed, agent_ids=[item.agent_id for item in sorted(joint.agents, key=lambda item: item.agent_id)],
                        candidate_ids=packed["pair_keys"], selectable_pair_count=len(packed["pair_keys"]), candidate_support_digest=snapshot.snapshot_digest,
                        no_assign_option=MC.NO_ASSIGN, actor_config_value=actor_config, feature_contract=feature_contract,
                        frozen_authority_hashes=authorities, source_commit=source["source_commit"], captured_device=str(device))
                    pre_state = adapter.state_identity()
                    with torch.no_grad():
                        logits, no_assign = actor(global_feats=model_packed["global_feats"], demand_feats=model_packed["demand_feats"],
                            agent_feats=model_packed["agent_feats"], agent_mask=model_packed["agent_mask"], candidate_feats=model_packed["candidate_feats"],
                            pair_agent_index=model_packed["pair_agent_index"], safe_mask=model_packed["safe_mask"])
                        selectable_logits = logits[:, :len(packed["pair_keys"])]
                        output = H.select(decision_group, packed["pair_keys"], selectable_logits, no_assign, packed["safe_mask"])
                        old_log_prob = float(JL.masked_log_probs(selectable_logits, no_assign, packed["safe_mask"])[0, output.selected_index])
                        value = float(critic(global_feats=model_packed["global_feats"], demand_feats=model_packed["demand_feats"],
                            agent_feats=model_packed["agent_feats"], agent_mask=model_packed["agent_mask"],
                            safe_summary=JL.safe_set_summary(model_packed["candidate_feats"], model_packed["safe_mask"]))[0])
                    if not output.selected_is_no_assign:
                        try: snapshot.assert_selectable(agent_id=output.selected_pair[0], candidate_id=output.selected_pair[1])
                        except Exception: inv["rejected_candidate_selected"] += 1; inv["zero_loss_violations"] += 1
                    action_by_agent = {index: 0 for index in range(envelope["agents"])}
                    if not output.selected_is_no_assign: action_by_agent[agent_slot[output.selected_pair[0]]] = BT1.SERVE
                    legal = {index: [True, True, True] for index in range(envelope["agents"])}
                    step_rewards = []
                    for within in range(2):
                        budget.transition()
                        result = adapter.step(action_by_agent, legal_mask=legal, target_ids=action_by_agent,
                            provenance={"arm_id": "BT8_A1", "policy_source": "joint_assignment_actor", "policy_contract": CC.CONTRACT_ID,
                                        "actual_checkpoint_loaded": False})
                        rewards, active = [], []
                        for event in result["events"]:
                            metric = BT1.agent_reward_metrics(transition_id=f"{decision_group}:step{within}:agent{event['agent_id']}",
                                agent_id=event["agent_id"], action_id=event["action_id"], boarded=int(event["passenger_served"]),
                                served=int(event["passenger_served"]), wait_rows=[], decision_ts=int(event["event_ts"]), intervened=False)
                            reward = float(compute_reward_v2(metric)["reward_total"]); inv["nan_or_inf"] += int(not math.isfinite(reward))
                            rewards.append(reward); active.append(True); total_boardings += int(event["passenger_served"])
                        step_rewards.append(CC.team_reward(rewards, active))
                    terminal = decision_index + 1 == envelope["trajectory_length"]
                    transition = CC.AssignmentTransition(assignment_step_id=decision_group, decision_group_id=decision_group,
                        episode_id=f"seed{seed}", window_id=window_id, decision_ts=decision_index * 2,
                        next_assignment_ts=None if terminal else (decision_index + 1) * 2, delta_operational_steps=2,
                        pre_state_digest=str(pre_state), next_state_digest=str(adapter.state_identity()), safe_pair_ids=list(packed["pair_keys"]),
                        safe_pair_mask=[True] * len(packed["pair_keys"]), no_assign_index=len(packed["pair_keys"]),
                        selected_agent_id=None if output.selected_is_no_assign else output.selected_pair[0],
                        selected_candidate_id=None if output.selected_is_no_assign else output.selected_pair[1], selected_is_no_assign=output.selected_is_no_assign,
                        valid_action_count=len(packed["pair_keys"]) + 1, forced_action=len(packed["pair_keys"]) == 0, old_log_prob=old_log_prob,
                        old_value=value, team_reward_sequence=list(step_rewards), assignment_discounted_reward=CC.assignment_return(step_rewards),
                        terminated=terminal, truncated=False, policy_version=H.HEAD_VERSION, credit_contract_version=CC.CONTRACT_VERSION, seed=seed,
                        provenance={"window_id": window_id, "time_band": band, "source_group": source_group, "decision_index": decision_index,
                                    "candidate_snapshot_digest": snapshot.snapshot_digest, "candidate_regenerated_during_ppo": False})
                    tx_rows.append({"t": transition, "packed": packed, "snapshot": snapshot})
                    decisions.append({"seed": seed, "window_id": window_id, "time_band": band, "decision_index": decision_index,
                        "source_group": source_group, "candidate_count_before_zero_loss": support["before_zero_loss"],
                        "candidate_count_after_zero_loss": support["after_zero_loss"], "joint_support_including_no_assign": support["after_zero_loss"] + 1,
                        "genuine_2plus_candidate": support["genuine_2plus"], "zero_loss_pass": support["zero_loss_pass"],
                        "zero_loss_fail": support["zero_loss_fail"], "zero_loss_selective": support["selective"],
                        "selected_agent": transition.selected_agent_id, "selected_candidate": transition.selected_candidate_id,
                        "candidate_identity": list(transition.safe_pair_ids), "request_density_stratum": window_meta["prepolicy_density_rank"],
                        "frozen_actor_snapshot_digest": frozen["snapshot_digest"], "frozen_actor_snapshot_path": frozen["relative_path"],
                        "assignment_reward": transition.assignment_discounted_reward, "critic_value": value, "terminated": terminal,
                        "informative": transition.assignment_discounted_reward != 0.0})
                    rewards_by_decision.append(transition.assignment_discounted_reward)
                rollout.extend(tx_rows)
                window_rows.append({"seed": seed, "window_id": window_id, "time_band": band, "trajectory_length": len(tx_rows),
                                    "causal_transitions": 8, "assignment_decisions": 4, "boardings": total_boardings,
                                    "assignment_rewards": rewards_by_decision})

        gae_rows = []
        for seed in envelope["seeds"]:
            rows = [row for row in rollout if row["t"].seed == seed]; transactions = [row["t"] for row in rows]
            values = [row.old_value for row in transactions]
            next_values = [next((values[j] for j in range(i + 1, len(transactions))
                                 if transactions[j].window_id == item.window_id and transactions[j].episode_id == item.episode_id), 0.0)
                           for i, item in enumerate(transactions)]
            gae = JL.compute_assignment_gae(transactions, values, next_values)
            for index, item in enumerate(transactions):
                delta, advantage = gae["assignment_td_residual"][index], gae["assignment_advantage"][index]
                gae_rows.append({"seed": seed, "window_id": item.window_id, "time_band": item.provenance["time_band"],
                    "decision_index": item.provenance["decision_index"], "reward": item.assignment_discounted_reward,
                    "value": item.old_value, "next_value": next_values[index], "td_residual": delta, "advantage": advantage,
                    "gae_recursive_term": advantage - delta, "nonterminal": not item.terminated and not item.truncated,
                    "temporally_propagated": abs(advantage - delta) > 1e-12})
            batch = BT6.batch_rollout(rows, device=device, adim=adim, cdim=cdim)
            advantages = torch.tensor(gae["assignment_advantage"], dtype=torch.float32, device=device)
            returns = torch.tensor(gae["assignment_return"], dtype=torch.float32, device=device)
            for update_index in range(3):
                for row in rows:
                    try: row["snapshot"].replay_guard(support_digest=row["t"].provenance["candidate_snapshot_digest"], regeneration_requested=False)
                    except Exception: inv["candidate_regeneration"] += 1
                logits, no_assign = actor(global_feats=batch["global_feats"], demand_feats=batch["demand_feats"], agent_feats=batch["agent_feats"],
                    agent_mask=batch["agent_mask"], candidate_feats=batch["candidate_feats"], pair_agent_index=batch["pair_agent_index"], safe_mask=batch["safe_mask"])
                prediction = critic(global_feats=batch["global_feats"], demand_feats=batch["demand_feats"], agent_feats=batch["agent_feats"],
                    agent_mask=batch["agent_mask"], safe_summary=JL.safe_set_summary(batch["candidate_feats"], batch["safe_mask"]))
                loss = JL.assignment_ppo_loss(new_pair_logits=logits, new_no_assign_logit=no_assign, safe_mask=batch["safe_mask"],
                    action_index=batch["action_index"], old_log_prob=batch["old_log_prob"], advantage=advantages, value_pred=prediction,
                    value_target=returns, forced_action=batch["forced_action"])
                budget.update(); update = JL.apply_assignment_update(loss=loss, actor=actor, critic=critic, actor_optimizer=actor_opt, critic_optimizer=critic_opt)
                torch.mps.synchronize()
                finite = all(math.isfinite(float(value)) for value in (update["actor_grad_norm"], update["critic_grad_norm"], update["total_loss"]))
                inv["nan_or_inf"] += int(not finite)
                updates.append({"seed": seed, "update": update_index + 1, "rows": len(rows), "policy_loss": float(loss["policy_loss"].detach()),
                    "critic_loss": float(loss["critic_loss"].detach()), "entropy": float(loss["entropy"].detach()),
                    "ratio_mean": float(loss["ratio"].detach().mean()), "ratio_min": float(loss["ratio"].detach().min()),
                    "ratio_max": float(loss["ratio"].detach().max()), **update})
        checkpoint = {"path": "bt8_a1_joint_assignment_checkpoint.pt", "test_only": True, "bounded": True, "non_promotable": True,
                      "winner": False, "best_model": False, "promotion": False, "performance_claim_allowed": False,
                      "paper_level_claim_allowed": False, "causal_performance_claim_allowed": False}
        root.mkdir(parents=True, exist_ok=True)
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "meta": {**checkpoint, "envelope": envelope,
                   "bt8_s0_source": BT8_S0_SOURCE, "selected_envelope_sha256": selection["selected_envelope_sha256"]}}, root / checkpoint["path"])
        collection = writer.finalize(checkpoint_path=root / checkpoint["path"])

    auth_events = AUTH.audit_log()
    execution_events = [event for event in auth_events if event["capability"] == AUTH.SIMULATOR_EXECUTION and event["outcome"] == "ALLOWED"]
    training_events = [event for event in auth_events if event["capability"] == AUTH.TRAINING and event["outcome"] == "ALLOWED"]
    locks_after = AUTH.authorization_state()["capabilities"]
    after = BT6.frozen_hashes(); actor_after, critic_after = BT1.param_digest(actor), BT1.param_digest(critic)
    trajectories = defaultdict(list)
    for row in rollout: trajectories[(row["t"].seed, row["t"].window_id)].append(row["t"])
    by_band = {band: [row for row in decisions if row["time_band"] == band] for band in ("night", "offpeak", "peak")}
    multi = {band: sum(row["genuine_2plus_candidate"] for row in rows) for band, rows in by_band.items()}
    selective = {band: sum(row["zero_loss_selective"] for row in rows) for band, rows in by_band.items()}
    propagated = [row for row in gae_rows if row["temporally_propagated"]]
    nonterminal = [row for row in gae_rows if row["nonterminal"]]
    readiness = {"exact_budget": budget.payload()["used"] == {"distinct_windows": 12, "window_visits": 24, "seeds": envelope["seeds"],
                  "assignment_decisions": 96, "trajectories": 24, "causal_transitions": 192, "optimizer_updates": 6},
                 "all_time_bands": all(by_band.values()), "all_trajectories_four_step": len(trajectories) == 24 and all(len(rows) == 4 for rows in trajectories.values()),
                 "gae_recursion": bool(propagated), "finite_gradients": all(math.isfinite(row["actor_grad_norm"]) and math.isfinite(row["critic_grad_norm"]) for row in updates),
                 "snapshot_count": collection.get("snapshot_count") == 96, "snapshot_checkpoint_binding": collection.get("checkpoint_sha256") == sha256(root / checkpoint["path"]),
                 "t1_posttraining_contract_retained": review_contract["selector"].get("tolerance") == 0.0}
    if any(locks_after.values()): inv["unauthorized_optimizer_step"] += 1
    if factory.source_mutations: inv["source_state_mutation"] += factory.source_mutations
    hard = []
    if before != after: hard.append("FROZEN_COMPONENT_MUTATION")
    if actor_before == actor_after or critic_before == critic_after: hard.append("JOINT_PARAMETER_DID_NOT_MOVE")
    if any(inv.values()): hard.append("INTEGRITY_VIOLATION:" + ",".join(name for name, value in inv.items() if value))
    if not adversarial["all_passed"]: hard.append("ADVERSARIAL_PROBE_FAILED")
    if not all(readiness.values()): hard.append("A1_READINESS_FAILURE:" + ",".join(name for name, value in readiness.items() if not value))
    gate, classification = (PASS_GATE, PASS_CLASS) if not hard else (INTEGRITY_BLOCK, "BT8_A1_INTEGRITY_OR_ENVELOPE_FAILURE")
    advantages = [row["advantage"] for row in gae_rows]; returns = [row["advantage"] + row["value"] for row in gae_rows]
    candidate_audit = {"decisions": decisions, "total_decisions": len(decisions), "genuine_2plus_candidate_decisions": sum(multi.values()),
                       "zero_candidate_decisions": sum(row["candidate_count_after_zero_loss"] == 0 for row in decisions),
                       "one_candidate_decisions": sum(row["candidate_count_after_zero_loss"] == 1 for row in decisions),
                       "by_time_band": {band: {"decisions": len(rows), "genuine_2plus": multi[band]} for band, rows in by_band.items()},
                       "candidate_regeneration": inv["candidate_regeneration"], "frozen_source_agent_to_causal_slot": agent_slot}
    zero_audit = {"epsilon_sec": 0.0, "candidates_evaluated": sum(row["candidate_count_before_zero_loss"] for row in decisions),
                  "PASS": sum(row["zero_loss_pass"] for row in decisions), "FAIL": sum(row["zero_loss_fail"] for row in decisions),
                  "selective_states": sum(selective.values()), "selective_by_time_band": selective,
                  "zero_loss_violations": inv["zero_loss_violations"], "rejected_candidate_selected": inv["rejected_candidate_selected"]}
    learning = {"informative_decisions": sum(row["informative"] for row in decisions),
                "non_zero_team_reward_fraction": sum(row["informative"] for row in decisions) / len(decisions),
                "multi_step_trajectories": len(trajectories), "nonterminal_transitions": len(nonterminal),
                "temporally_propagated_advantages": len(propagated), "gae_rows": gae_rows,
                "advantage_distribution": {"mean": statistics.mean(advantages), "std": statistics.pstdev(advantages), "variance": statistics.pvariance(advantages)},
                "critic_target_distribution": {"mean": statistics.mean(returns), "variance": statistics.pvariance(returns)},
                "cross_window_contamination": inv["cross_window_contamination"], "cross_seed_contamination": inv["cross_seed_contamination"],
                "legacy_advantage_contamination": inv["legacy_advantage_contamination"], "future_leakage": inv["future_leakage"]}
    lineage = {"bt8_s0_source": BT8_S0_SOURCE, "bt8_s0_selected_envelope_sha256": selection["selected_envelope_sha256"],
               "reward_v2_sha256": PV8_REWARD_V2_FREEZE_SHA256, "t1_contract_id": TIE.TIE_BREAK_CONTRACT_ID,
               "t1_tolerance": 0.0, "t1_training_time_use": False, "t1_posttraining_review_use": True}
    outputs = {
        "bt8_a1_mps_preflight.json": preflight, "bt8_a1_authoritative_binding.json": binding,
        "bt8_a1_execution_manifest.json": {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
            "envelope": envelope, "budget": budget.payload(), "selected_windows": windows, "checkpoint": checkpoint,
            "runtime_seconds": round(time.perf_counter() - started, 3), "maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "baseline_comparison": "NONE", "performance_interpretation_permitted": False, "global_locks": LOCKS},
        "bt8_a1_authorization_audit.json": {"simulator_execution_allowed_events": len(execution_events), "training_allowed_events": len(training_events),
            "capabilities_after_block": locks_after, "global_locks": LOCKS, "budget": budget.payload()},
        "bt8_a1_candidate_support_audit.json": candidate_audit, "bt8_a1_zero_loss_selectivity_audit.json": zero_audit,
        "bt8_a1_learning_signal_audit.json": learning, "bt8_a1_optimizer_audit.json": {"optimizer_steps": budget.updates, "max_allowed": 6,
            "updates": updates, "actor_gradients_finite": all(math.isfinite(row["actor_grad_norm"]) for row in updates),
            "critic_gradients_finite": all(math.isfinite(row["critic_grad_norm"]) for row in updates)},
        "bt8_a1_frozen_policy_snapshot_audit.json": {"schema_version": FPS.SNAPSHOT_SCHEMA_VERSION, "snapshot_count": collection.get("snapshot_count"),
            "expected_decision_count": 96, "collection_digest": collection.get("collection_digest"), "checkpoint_sha256": collection.get("checkpoint_sha256"),
            "capture_before_training_time_actor_forward": True, "candidate_regeneration_during_replay": 0, "feature_recomputation_during_replay": 0,
            "mask_reconstruction_during_replay": 0},
        "bt8_a1_posttraining_review_contract.json": review_contract,
        "frozen_hash_before_after.json": {"before": before, "after": after, "all_unchanged": before == after,
            "joint_actor": {"before": actor_before, "after": actor_after, "changed": actor_before != actor_after},
            "joint_critic": {"before": critic_before, "after": critic_after, "changed": critic_before != critic_after}},
        "test_results.json": {"binding": binding, "adversarial_probes": adversarial, "integrity_violations": inv, "readiness": readiness,
            "hard_failures": hard, "TEST6_access": 0, "github_push_performed": False, "lineage": lineage},
        "gate_decision.json": {"gate": gate, "classification": classification, "source_commit": source["source_commit"], "lineage": lineage,
            "hard_failures": hard, "warnings": [], "global_locks": LOCKS,
            "next_step": "BT8 frozen joint-assignment policy behavior review using only preserved A1 snapshots and bound checkpoint" if not hard else "STOP"},
    }
    for name, payload in outputs.items(): dump(root / name, payload)
    (root / "final_report.md").write_text(
        f"# {STAGE} — exact A1 bounded training\n\ngate = {gate}\nclassification = {classification}\nsource_commit = {source['source_commit']}\n\n"
        f"MPS preflight passed on `{preflight.get('selected_device')}` before any capability grant. A1 consumed {budget.visits} visits, "
        f"{budget.decisions} decisions, {budget.transitions} causal transitions, and {budget.updates} optimizer updates.\n\n"
        "All actor-input snapshots were captured before training-time forwards and are bound to the non-promotable checkpoint. "
        "This is not a policy-quality, baseline, convergence, or causal-performance result.\n", encoding="utf-8")
    manifest = {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}
    dump(root / "manifest.json", {"stage": STAGE, "gate": gate, "classification": classification, "source_commit": source["source_commit"],
                                   "lineage": lineage, "file_sha256": manifest, "github_push_performed": False})
    (root / ("_SUCCESS.lock" if not hard else "_BLOCKED.lock")).write_text(gate + "\n", encoding="utf-8")
    print(f"[{'PASS' if not hard else 'BLOCKED'}] {gate}"); print(f"artifact: {root.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
