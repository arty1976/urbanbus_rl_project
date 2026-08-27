#!/usr/bin/env python3
"""R18 exact executor for the one-shot R17 E1 bounded-training authorization.

This source is intentionally inert unless invoked with both the SHA-bound R17
manifest and ``--execute-exact-r17-envelope``.  ``--dry-run`` is read-only and
is the only mode R17 may invoke.  The actual execution path is new rather than
a retrofit of historical R13/F1 execution: rollout selection always crosses
the sealed R16 frozen-policy distribution boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE = "H4M-AE-R9.8-LS3-BT8-R18"
PASS_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R18_E1_MINIMAL_BOUNDED_TRAINING_AND_CAUSAL_LEARNING_PATH_EVIDENCE_COMPLETE"
AUTH_BLOCK = "BLOCKED_R18_AUTHORIZATION_OR_BINDING_FAILURE"
INTEGRITY_BLOCK = "BLOCKED_R18_EXECUTION_INTEGRITY_FAILURE"
MPS_BLOCK = "BLOCKED_MPS_EXECUTION_ENVIRONMENT_UNAVAILABLE"
BD_CREDIT_BLOCK = "BLOCKED_R18_BD_E1_CAUSAL_LEARNING_PATH_NOT_OBSERVED"
EXPECTED_R17_GATE = "PASS_SUSEONG_H4M_AE_R9_8_LS3_BT8_R17_E1_BOUNDED_TRAINING_EXECUTION_AUTHORIZATION_AND_ENVELOPE_FREEZE_COMPLETE"
R16_SOURCE = "90a8a69227c1166c8d311d54f53fe9795dfeb7cd"

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


class R18Error(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def require(condition: bool, code: str, detail: str = "") -> None:
    if not condition:
        raise R18Error(code, detail)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def git(args: Sequence[str]) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, text=True, capture_output=True, check=True).stdout.strip()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-manifest", required=True, type=Path)
    parser.add_argument("--authorization-sha256", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute-exact-r17-envelope", action="store_true")
    return parser.parse_args(argv)


def load_authorization(path: Path, supplied_sha256: str) -> dict[str, Any]:
    require(path.is_file(), AUTH_BLOCK, f"authorization_missing={path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual = canonical_sha256({key: value for key, value in payload.items() if key != "authorization_sha256"})
    require(payload.get("authorization_sha256") == supplied_sha256 == actual, AUTH_BLOCK, "authorization_sha256")
    require(payload.get("stage") == "H4M-AE-R9.8-LS3-BT8-R17" and payload.get("authorized") is True
            and payload.get("authorization") == "R18_EXACT_ONE_SHOT_E1_BOUNDED_TRAINING_ONLY", AUTH_BLOCK, "authorization_schema")
    upstream = dict(payload.get("upstream", {}))
    executor = dict(upstream.get("r18_executor", {}))
    require(executor.get("path") == str(Path(__file__).resolve()) and executor.get("sha256") == sha256(Path(__file__)),
            AUTH_BLOCK, "r18_executor_source_binding")
    require(payload.get("r16_source_commit") == R16_SOURCE, AUTH_BLOCK, "r16_source")
    envelope = dict(payload.get("envelope", {}))
    aggregate = dict(envelope.get("aggregate", {}))
    require(aggregate == {"environment_seed_count": 1, "windows": 6, "visits": 12, "trajectories": 12,
                          "assignment_decisions": 48, "causal_transitions": 96,
                          "assignment_actor_optimizer_steps_maximum": 6,
                          "assignment_critic_optimizer_steps_exact": 6,
                          "raw_optimizer_step_calls_maximum": 12, "supplemental_steps": 0},
            AUTH_BLOCK, "aggregate_envelope")
    arms = list(envelope.get("selected_arms", []))
    require([str(row.get("arm_id")) for row in arms] == ["AC_CONTROL_R1", "BD_E1_R1"], AUTH_BLOCK, "arm_order")
    for arm in arms:
        require(int(arm.get("assignment_decisions", -1)) == 24 and int(arm.get("trajectories", -1)) == 6
                and int(arm.get("trajectory_length", -1)) == 4 and int(arm.get("ppo_epochs", -1)) == 3
                and int(arm.get("full_batch_size", -1)) == int(arm.get("minibatch_size", -1)) == 24,
                AUTH_BLOCK, f"arm_scope={arm.get('arm_id')}")
        seeds = dict(arm.get("categorical_probe_seed_by_decision", {}))
        require(len(seeds) == 24 and set(seeds.values()) == {0}, AUTH_BLOCK, f"categorical_seed_map={arm.get('arm_id')}")
    module = dict(payload.get("module_freeze_contract", {}))
    require(module.get("training_selection_mode") == "FROZEN_MASKED_CATEGORICAL_TRAINING"
            and module.get("inference_evaluation_mode") == "FROZEN_INFERENCE_T1"
            and module.get("E1_only") is True and module.get("E2_rescue") is False and module.get("E3_temperature_or_floor") is False
            and module.get("sealed_distribution_view_required") is True
            and module.get("e1_contract_sha256") == "eb84543a9fc06dcf730e49aa3895d9fe26d2244a05ce340986b7449418205ad9", AUTH_BLOCK, "selector_freeze")
    checkpoints = dict(dict(payload.get("checkpoint_contract", {})).get("initial_inputs", {}))
    require(set(checkpoints) == {"initial:AC-R1", "initial:AC-R2", "initial:BD-R1", "initial:BD-R2"}, AUTH_BLOCK, "checkpoint_set")
    for arm in arms:
        key = str(arm.get("initial_checkpoint_key"))
        entry = dict(checkpoints.get(key, {}))
        checkpoint_path = Path(str(entry.get("path", "")))
        require(entry.get("role") == "allowed_initial_input" and checkpoint_path.is_file()
                and sha256(checkpoint_path) == entry.get("sha256"), AUTH_BLOCK, f"initial_checkpoint={key}")
    output_root = Path(str(dict(payload.get("checkpoint_contract", {})).get("R18_output_root", "")))
    require(output_root.parent.is_dir() and output_root.name.startswith("pv8_r2a_r8e_r3_r_h4m_ae_ls3_bt8_r18_"), AUTH_BLOCK, "output_root")
    return payload


def dry_run_report(auth: Mapping[str, Any]) -> dict[str, Any]:
    """Pure manifest/source verification; no MPS probe, models, or artifacts."""
    return {
        "stage": STAGE, "authorization_valid": True, "r17_source_commit": auth["source_commit"],
        "r18_executor_sha256": sha256(Path(__file__)), "training": 0, "rollout": 0,
        "optimizer_step": 0, "checkpoint_write": 0, "policy_mutation": 0,
        "next_action_requires_explicit_execute_flag": True,
    }


def _module_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for _, tensor in sorted(module.state_dict().items()):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _optimizer_summary(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    return {"fresh_empty": len(state["state"]) == 0, "state_item_count": len(state["state"]),
            "parameter_group_count": len(state["param_groups"])}


def _counters() -> dict[str, int]:
    return {
        "training": 0, "mps_training": 0, "causal_rollout": 0, "simulator_step": 0,
        "authoritative_candidate_generation": 0, "candidate_regeneration_after_selection": 0,
        "candidate_regeneration_during_ppo": 0, "local_search_rerun_during_ppo": 0,
        "zero_loss_reevaluation_during_ppo": 0, "actor_optimizer_step": 0, "critic_optimizer_step": 0,
        "raw_optimizer_step": 0, "unauthorized_optimizer_step": 0, "checkpoint_write": 0,
        "checkpoint_promotion": 0, "review_optimizer_rows": 0, "review_batch_inclusion": 0,
        "review_regeneration": 0, "future_leakage": 0, "cross_window_gae": 0,
        "duplicate_reward_ancestry": 0, "test6_access": 0, "github_push": 0, "nan_or_inf": 0,
        "candidate_identity_mismatch": 0, "candidate_plan_execution_collapse": 0, "serve_fallback": 0,
        "zero_loss_violation": 0, "illegal_or_masked_selection": 0,
        "source_state_mutation_during_shadow_evaluation": 0, "action_support_mutation": 0,
    }


def _load_models(*, arms: Sequence[Mapping[str, Any]], checkpoints: Mapping[str, Any], config: Mapping[str, Any],
                 H: Any, JL: Any, device: torch.device) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    identities: set[int] = set()
    for arm in arms:
        key, arm_id = str(arm["initial_checkpoint_key"]), str(arm["arm_id"])
        checkpoint_path = Path(str(checkpoints[key]["path"]))
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        require(isinstance(payload, Mapping) and set(payload).issuperset({"actor", "critic", "meta"})
                and not any("optimizer" in str(name).lower() for name in payload), AUTH_BLOCK, f"checkpoint_schema={arm_id}")
        torch.manual_seed(int(arm["actor_seed"]))
        actor = H.CandidateSensitiveMultiAgentCandidateAssignmentHead(
            global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]), agent_dim=int(config["agent_dim"]),
            candidate_dim=int(config["candidate_dim"]), hidden=int(config["hidden"]), heads=int(config["heads"])).to(device)
        actor.load_state_dict(payload["actor"], strict=True)
        torch.manual_seed(int(arm["critic_seed"]))
        critic = JL.JointAssignmentCritic(global_dim=int(config["global_dim"]), demand_dim=int(config["demand_dim"]),
                                           agent_dim=int(config["agent_dim"]), safe_summary_dim=1 + 2 * int(config["candidate_dim"])).to(device)
        critic.load_state_dict(payload["critic"], strict=True)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-4)
        require(_optimizer_summary(actor_opt)["fresh_empty"] and _optimizer_summary(critic_opt)["fresh_empty"], AUTH_BLOCK, f"optimizer={arm_id}")
        parameter_ids = [id(parameter) for parameter in actor.parameters()] + [id(parameter) for parameter in critic.parameters()]
        require(not identities.intersection(parameter_ids), AUTH_BLOCK, f"parameter_aliasing={arm_id}")
        identities.update(parameter_ids)
        models[arm_id] = {**dict(arm), "actor": actor, "critic": critic, "actor_opt": actor_opt, "critic_opt": critic_opt,
                          "actor_initial_digest": _module_digest(actor), "critic_initial_digest": _module_digest(critic),
                          "behavior_checkpoint_sha256": str(checkpoints[key]["sha256"]), "checkpoint_path": str(checkpoint_path)}
    return models


def _rollout_arm(*, model: Mapping[str, Any], frozen: Mapping[str, str], root: Path, snapshot_entries: list[dict[str, Any]],
                 H: Any, JL: Any, CC: Any, CB: Any, PE: Any, MC: Any, BT1: Any, BT6: Any, R3: Any,
                 F1MOD: Any, FPS: Any, S: Any, compute_reward_v2: Any, config: Mapping[str, Any],
                 features: Mapping[str, Any], device: torch.device, counters: dict[str, int]) -> dict[str, Any]:
    """Fresh causal rollout whose action selection is the sealed R16 E1 path."""
    arm_id = str(model["arm_id"])
    factory = BT6.RepairedSupportFactory()
    windows = list(model["train_window_records"])
    agent_slots = F1MOD.f1_agent_slot_mapping(eligible=factory.eligible, source_groups=[str(row["source_group"]) for row in windows], slots=8)
    rows: list[dict[str, Any]] = []
    support_checks: list[dict[str, Any]] = []
    for window_index, window in enumerate(windows):
        trajectory_id = f"R18:{arm_id}:{window['window_id']}"
        adapter = R3.imp("bridge", R3.BRIDGE).PV8CausalKpiAdapter(
            **R3.authoritative_adapter_inputs({"window_id": window["window_id"]}, num_agents=8, seed=int(model["environment_seed"])))
        for decision_index in range(4):
            sequence_index = window_index * 4 + decision_index
            decision_id = f"R18:{arm_id}:{sequence_index}"
            probe_seed = int(dict(model["categorical_probe_seed_by_decision"])[f"{arm_id}:{sequence_index}"])
            support = factory.build(source_group=window["source_group"], decision_group=decision_id)
            counters["authoritative_candidate_generation"] += 1
            packed, ready = F1MOD.model_pack(support=support, H=H, BT6=BT6, cdim=int(config["candidate_dim"]), device=device)
            captured = F1MOD.snapshot(root=root / "training_snapshots", store=snapshot_entries, FPS=FPS,
                                      decision_id=decision_id, window=window, seed=int(model["environment_seed"]), index=decision_index,
                                      packed=packed, model_packed=ready, support=support, MC=MC, config=config, features=features,
                                      frozen=frozen, source_commit=git(["rev-parse", "HEAD"]), device=device)
            loaded = FPS.load_snapshot(captured["directory"])
            meta, tensors = loaded["metadata"], loaded["tensors"]
            pair_keys = [(str(pair["agent_id"]), str(pair["candidate_id"])) for pair in meta["candidate_ids"]]
            require(loaded["snapshot_digest"] == captured["digest"] and pair_keys == list(packed["pair_keys"])
                    and meta["candidate_support_digest"] == support["snapshot"].snapshot_digest
                    and torch.equal(tensors["safe_mask"], ready["safe_mask"].detach().cpu()),
                    INTEGRITY_BLOCK, f"rollout_snapshot_roundtrip={decision_id}")
            model["actor"].eval(); model["critic"].eval()
            with torch.no_grad():
                raw, no_assign = model["actor"](global_feats=ready["global_feats"], demand_feats=ready["demand_feats"],
                                                   agent_feats=ready["agent_feats"], agent_mask=ready["agent_mask"],
                                                   candidate_feats=ready["candidate_feats"], pair_agent_index=ready["pair_agent_index"],
                                                   safe_mask=ready["safe_mask"])
                logits, mask = raw[:, :len(packed["pair_keys"])], ready["safe_mask"][:, :len(packed["pair_keys"])]
                view = S.make_frozen_masked_distribution_view(pair_keys=packed["pair_keys"], pair_logits=logits,
                                                               no_assign_logit=no_assign, safe_mask=mask)
                output = S.select_frozen_policy_action(distribution_view=view, mode=S.FROZEN_MASKED_CATEGORICAL_TRAINING,
                                                        snapshot_identity=captured["digest"], probe_seed=probe_seed)
                value = float(model["critic"](global_feats=ready["global_feats"], demand_feats=ready["demand_feats"],
                                                agent_feats=ready["agent_feats"], agent_mask=ready["agent_mask"],
                                                safe_summary=JL.safe_set_summary(ready["candidate_feats"], ready["safe_mask"]))[0])
            old_log = float(output.log_probability.detach().cpu())
            require(math.isfinite(old_log) and math.isfinite(value) and int(output.selected_index) < len(pair_keys) + 1,
                    INTEGRITY_BLOCK, f"selection_nonfinite_or_index={decision_id}")
            selected_type = "NO_ASSIGN" if output.selected_is_no_assign else "CANDIDATE"
            if not output.selected_is_no_assign:
                support["snapshot"].assert_selectable(agent_id=output.selected_pair[0], candidate_id=output.selected_pair[1])
            plan = F1MOD.selected_plan(snapshot_value=support["snapshot"], output=output, decision_id=decision_id, CB=CB, PE=PE)
            selected, applied, credited = str(plan.selected_candidate_id), str(plan.applied_candidate_id), str(plan.credited_candidate_id)
            identity_match = selected == applied == credited
            counters["candidate_identity_mismatch"] += int(not identity_match)
            counters["serve_fallback"] += int(bool(plan.serve_fallback_used))
            require(identity_match and not bool(plan.serve_fallback_used), INTEGRITY_BLOCK, f"candidate_identity={decision_id}")
            actions = {agent: 0 for agent in range(8)}
            if not output.selected_is_no_assign:
                require(output.selected_pair[0] in agent_slots, INTEGRITY_BLOCK, f"agent_slot={decision_id}")
                actions[int(agent_slots[output.selected_pair[0]])] = BT1.SERVE
            rewards: list[float] = []
            operational: list[dict[str, Any]] = []
            for operation in range(2):
                result = adapter.step(actions, legal_mask={agent: [True, True, True] for agent in actions}, target_ids=actions,
                                      provenance={"arm_id": arm_id, "policy_source": "R16_sealed_E1_masked_categorical",
                                                  "candidate_plan_transition_id": plan.transition_id,
                                                  "candidate_plan_digest": plan.candidate_plan_digest, "applied_plan_digest": plan.applied_plan_digest})
                agent_rewards = []
                for event in result["events"]:
                    metric = BT1.agent_reward_metrics(transition_id=f"{decision_id}:{operation}:{event['agent_id']}",
                                                      agent_id=event["agent_id"], action_id=event["action_id"],
                                                      boarded=int(event["passenger_served"]), served=int(event["passenger_served"]),
                                                      wait_rows=[], decision_ts=int(event["event_ts"]), intervened=False)
                    reward = float(compute_reward_v2(metric)["reward_total"])
                    require(math.isfinite(reward), INTEGRITY_BLOCK, f"reward_nonfinite={decision_id}")
                    agent_rewards.append(reward)
                rewards.append(CC.team_reward(agent_rewards, [True] * len(agent_rewards)))
                operational.append({"step": operation, "team_reward": rewards[-1], "causal_state_digest": adapter.state_identity()})
                counters["causal_rollout"] += 1; counters["simulator_step"] += 1
            terminal = decision_index == 3
            transition = CC.AssignmentTransition(
                assignment_step_id=decision_id, decision_group_id=decision_id, episode_id=arm_id, window_id=window["window_id"],
                decision_ts=decision_index * 2, next_assignment_ts=None if terminal else (decision_index + 1) * 2,
                delta_operational_steps=2, pre_state_digest=plan.events[0]["source_state_digest"], next_state_digest=plan.next_state.state_digest,
                safe_pair_ids=list(packed["pair_keys"]), safe_pair_mask=[True] * len(packed["pair_keys"]), no_assign_index=len(packed["pair_keys"]),
                selected_agent_id=None if output.selected_is_no_assign else output.selected_pair[0],
                selected_candidate_id=None if output.selected_is_no_assign else output.selected_pair[1],
                selected_is_no_assign=bool(output.selected_is_no_assign), valid_action_count=len(packed["pair_keys"]) + 1,
                forced_action=len(packed["pair_keys"]) == 0, old_log_prob=old_log, old_value=value,
                team_reward_sequence=rewards, assignment_discounted_reward=CC.assignment_return(rewards), terminated=terminal,
                truncated=False, policy_version=H.CANDIDATE_SENSITIVE_HEAD_VERSION, credit_contract_version=CC.CONTRACT_VERSION,
                seed=int(model["environment_seed"]), provenance={"trajectory_id": trajectory_id, "arm_id": arm_id,
                    "time_band": window["time_band"], "candidate_support_digest": support["snapshot"].snapshot_digest,
                    "candidate_plan_digest": plan.candidate_plan_digest, "applied_plan_digest": plan.applied_plan_digest,
                    "selected_candidate_id": selected, "applied_candidate_id": applied, "credited_candidate_id": credited,
                    "candidate_regenerated_during_ppo": False, "operational": operational, "no_assign": plan.no_assign,
                    "selection_mode": output.selection_mode, "categorical_triggered": output.categorical_triggered,
                    "frozen_distribution_view_sha256": view.binding_sha256, "canonical_rng_keyset_sha256": output.canonical_rng_keyset_sha256,
                    "probe_seed": probe_seed})
            support_checks.append({"decision_id": decision_id, "snapshot_digest": captured["digest"], "support_digest": support["snapshot"].snapshot_digest,
                                   "pair_keys": pair_keys, "safe_mask_sha256": hashlib.sha256(tensors["safe_mask"].numpy().tobytes()).hexdigest(),
                                   "selected_index": int(output.selected_index), "old_log_probability": old_log})
            rows.append({"t": transition, "packed": packed, "snapshot": support["snapshot"], "loaded": loaded, "plan": plan,
                         "snapshot_digest": captured["digest"], "action_type": selected_type, "time_band": window["time_band"],
                         "categorical_triggered": bool(output.categorical_triggered), "selected_is_no_assign": bool(output.selected_is_no_assign)})
    require(len(rows) == 24 and len({row["t"].assignment_step_id for row in rows}) == 24, INTEGRITY_BLOCK, f"rollout_scope={arm_id}")
    counters["source_state_mutation_during_shadow_evaluation"] += int(factory.source_mutations)
    return {"rows": rows, "support_checks": support_checks, "factory_source_mutations": int(factory.source_mutations)}


def _assert_support_roundtrip(*, rows: Sequence[Mapping[str, Any]], CC: Any) -> dict[str, Any]:
    buffer = CC.AssignmentRolloutBuffer()
    update_supports: list[str] = []
    for row in rows:
        transition, loaded = row["t"], row["loaded"]
        metadata = loaded["metadata"]
        pairs = [(str(item["agent_id"]), str(item["candidate_id"])) for item in metadata["candidate_ids"]]
        require(pairs == list(transition.safe_pair_ids) and metadata["candidate_support_digest"] == transition.action_support_digest,
                INTEGRITY_BLOCK, f"ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE:{transition.assignment_step_id}")
        buffer.add(transition)
        update_supports.append(str(metadata["candidate_support_digest"]))
    try:
        result = buffer.assert_action_support_unchanged(update_supports)
    except Exception as exc:  # noqa: BLE001
        raise R18Error(INTEGRITY_BLOCK, f"ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE:{exc}") from exc
    return dict(result)


def _train_arm(*, model: Mapping[str, Any], prepared: Mapping[str, Any], support_guard: Mapping[str, Any],
               JL: Any, AUTH: Any, device: torch.device, counters: dict[str, int]) -> dict[str, Any]:
    """Three full-batch E1 PPO epochs with independent Actor/Critic steps."""
    batch, rows = prepared["batch"], list(prepared["rows"])
    active = batch["actor_eligibility_mask"] & ~batch["forced_action"]
    active_count = int(active.sum().item())
    require(active_count >= 1, BD_CREDIT_BLOCK if str(model["arm_id"]) == "BD_E1_R1" else INTEGRITY_BLOCK,
            f"actor_eligible={model['arm_id']}")
    actor, critic, actor_opt, critic_opt = model["actor"], model["critic"], model["actor_opt"], model["critic_opt"]
    updates: list[dict[str, Any]] = []
    ratios: list[list[float]] = [[] for _ in rows]
    for epoch in range(3):
        require(support_guard.get("behavior_equals_update_support") is True and int(support_guard.get("checked", 0)) == 24,
                INTEGRITY_BLOCK, "ACTION_SUPPORT_MUTATED_BETWEEN_ROLLOUT_AND_UPDATE")
        for row in rows:
            row["snapshot"].replay_guard(support_digest=row["t"].provenance["candidate_support_digest"], regeneration_requested=False)
        actor.train(); critic.train()
        logits, no_assign = actor(global_feats=batch["global_feats"], demand_feats=batch["demand_feats"], agent_feats=batch["agent_feats"],
                                  agent_mask=batch["agent_mask"], candidate_feats=batch["candidate_feats"],
                                  pair_agent_index=batch["pair_agent_index"], safe_mask=batch["safe_mask"])
        value = critic(global_feats=batch["global_feats"], demand_feats=batch["demand_feats"], agent_feats=batch["agent_feats"],
                       agent_mask=batch["agent_mask"], safe_summary=JL.safe_set_summary(batch["candidate_feats"], batch["safe_mask"]))
        require(bool(torch.isfinite(logits[batch["safe_mask"]]).all().item() and torch.isfinite(no_assign).all().item()
                     and torch.isfinite(value).all().item()), INTEGRITY_BLOCK, f"nonfinite_ppo_output={model['arm_id']}")
        loss = JL.assignment_ppo_loss(new_pair_logits=logits, new_no_assign_logit=no_assign, safe_mask=batch["safe_mask"],
                                      action_index=batch["action_index"], old_log_prob=batch["old_log_prob"], advantage=batch["advantage"],
                                      value_pred=value, value_target=batch["value_target"], forced_action=batch["forced_action"],
                                      actor_eligibility_mask=batch["actor_eligibility_mask"])
        require(int(loss["actor_eligible_rows"]) == active_count and not bool(loss["actor_update_skipped"]), INTEGRITY_BLOCK, "e1_actor_mask")
        logits.retain_grad(); no_assign.retain_grad()
        AUTH.require_capability(AUTH.TRAINING, site=f"{STAGE}:{model['arm_id']}:actor:{epoch + 1}")
        actor_opt.zero_grad(set_to_none=True); loss["actor_loss"].backward()
        inactive = ~active
        inactive_grad = max(float(logits.grad[inactive].detach().abs().max().cpu()) if bool(inactive.any()) else 0.0,
                            float(no_assign.grad[inactive].detach().abs().max().cpu()) if bool(inactive.any()) else 0.0)
        eligible_grad = float(logits.grad[active].detach().abs().sum().cpu()) + float(no_assign.grad[active].detach().abs().sum().cpu())
        require(inactive_grad == 0.0 and eligible_grad > 0.0, INTEGRITY_BLOCK, f"ineligible_actor_gradient={model['arm_id']}")
        actor_norm = float(torch.sqrt(sum((parameter.grad.detach() ** 2).sum() for parameter in actor.parameters() if parameter.grad is not None)).detach().cpu())
        actor_opt.step(); counters["actor_optimizer_step"] += 1; counters["raw_optimizer_step"] += 1
        AUTH.require_capability(AUTH.TRAINING, site=f"{STAGE}:{model['arm_id']}:critic:{epoch + 1}")
        critic_opt.zero_grad(set_to_none=True); loss["critic_loss"].backward()
        critic_norm = float(torch.sqrt(sum((parameter.grad.detach() ** 2).sum() for parameter in critic.parameters() if parameter.grad is not None)).detach().cpu())
        require(critic_norm > 0.0, INTEGRITY_BLOCK, f"critic_gradient={model['arm_id']}")
        critic_opt.step(); counters["critic_optimizer_step"] += 1; counters["raw_optimizer_step"] += 1
        torch.mps.synchronize()
        finite = all(math.isfinite(number) for number in (actor_norm, critic_norm, float(loss["actor_loss"].detach().cpu()), float(loss["critic_loss"].detach().cpu())))
        counters["nan_or_inf"] += int(not finite)
        for index, ratio in enumerate(loss["ratio"].detach().cpu().tolist()):
            ratios[index].append(float(ratio))
        updates.append({"epoch": epoch + 1, "actor_eligible_rows": active_count, "actor_loss": float(loss["actor_loss"].detach().cpu()),
                        "critic_loss": float(loss["critic_loss"].detach().cpu()), "actor_gradient_norm": actor_norm,
                        "critic_gradient_norm": critic_norm, "ineligible_logit_gradient_max_abs": inactive_grad,
                        "ratio_min": min(float(x) for x in loss["ratio"].detach().cpu()), "ratio_max": max(float(x) for x in loss["ratio"].detach().cpu())})
    require(max(row["ineligible_logit_gradient_max_abs"] for row in updates) == 0.0, INTEGRITY_BLOCK, "ineligible_gradient_leakage")
    return {"updates": updates, "ratios": ratios, "actor_final_digest": _module_digest(actor), "critic_final_digest": _module_digest(critic),
            "actor_changed": _module_digest(actor) != str(model["actor_initial_digest"]), "critic_changed": _module_digest(critic) != str(model["critic_initial_digest"])}


def _write_final_checkpoint(path: Path, model: Mapping[str, Any], training: Mapping[str, Any]) -> dict[str, Any]:
    require(not path.exists(), INTEGRITY_BLOCK, f"checkpoint_overwrite={path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"kind": "r18_exact_e1_final_bounded_evidence", "arm_id": model["arm_id"], "test_only": True, "bounded": True,
            "non_promotable": True, "winner": False, "best_model": False, "promotion": False,
            "initial_actor_digest": model["actor_initial_digest"], "final_actor_digest": training["actor_final_digest"],
            "initial_critic_digest": model["critic_initial_digest"], "final_critic_digest": training["critic_final_digest"]}
    torch.save({"actor": {name: tensor.detach().cpu() for name, tensor in model["actor"].state_dict().items()},
                "critic": {name: tensor.detach().cpu() for name, tensor in model["critic"].state_dict().items()}, "meta": meta}, path)
    return {"path": str(path), "sha256": sha256(path), **meta}


def _block(root: Path, code: str, detail: str, counters: Mapping[str, Any], auth: Mapping[str, Any] | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    outputs = {"r18_execution_manifest.json": {"authorized": False, "failure": detail},
               "r18_learning_path_audit.json": {"not_completed": True}, "r18_candidate_support_audit.json": {"not_completed": True},
               "r18_checkpoint_manifest.json": {"not_completed": True},
               "test_results.json": {"execution_counters": dict(counters), "hard_failures": [detail]},
               "gate_decision.json": {"stage": STAGE, "gate": code, "classification": "BLOCKED",
                                      "source_commit": auth.get("source_commit") if auth else None, "hard_failures": [detail], "next_step": "STOP"}}
    for name, value in outputs.items(): dump(root / name, value)
    (root / "final_report.md").write_text(f"# R18 blocked\n\n- gate: `{code}`\n- detail: `{detail}`\n", encoding="utf-8")
    manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
    dump(root / "manifest.json", {"stage": STAGE, "gate": code, "file_sha256": manifest})


def execute(auth: Mapping[str, Any]) -> None:
    """Execute only the SHA-bound one-shot R17 envelope; no adaptive extension."""
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "simulator"))
    import joint_assignment_credit_contract as CC
    import joint_assignment_e1_eligibility as E1
    import joint_assignment_f1_execution_contract as FC
    import joint_assignment_frozen_policy_selector as S
    import joint_assignment_frozen_policy_snapshot as FPS
    import joint_assignment_learning as JL
    import joint_candidate_plan_causal_bridge as CB
    import joint_candidate_plan_execution as PE
    import multi_agent_assignment_contract as MC
    import multi_agent_candidate_assignment_head as H
    import run_h4m_ae_ls3_bt1_tiny_causal_training as BT1
    import run_h4m_ae_ls3_bt6_postrepair_r2_training as BT6
    import run_h4m_ae_ls3_bt8_f1_bounded_training as F1MOD
    import run_h4m_ae_ls3_bt8_r17_e1_bounded_training_authorization as R17
    import run_h4m_ae_ls3_bt8_r7_seed_factorization_credit_eligibility as R7MOD
    import simulator_authorization as AUTH
    import test_h4m_ae_r3_causal_kpi_bridge as R3
    from rewards.mappo_reward_v1 import compute_reward_v2

    checkpoint_contract = dict(auth["checkpoint_contract"])
    root = Path(str(checkpoint_contract["R18_output_root"]))
    require(not root.exists(), AUTH_BLOCK, f"output_already_exists={root}")
    counters = _counters()
    try:
        require(git(["rev-parse", "HEAD"]) == str(auth["source_commit"]) and git(["status", "--porcelain=v1"]) == "",
                AUTH_BLOCK, "unexpected_source_mutation")
        r16_path = Path(str(dict(auth["upstream"])["timestamped_artifacts"]["r16"]))
        current_frozen = R17._source_hash_binding(r16_path)
        bound_frozen = dict(dict(auth["module_freeze_contract"])["frozen_source_hashes"])
        require(current_frozen == bound_frozen, AUTH_BLOCK, "frozen_source_mutation")
        e1_contract_entry = dict(dict(auth["upstream"])["e1_contract"])
        e1_runtime_path = Path(str(e1_contract_entry["runtime_path"]))
        e1_selection_path = Path(str(e1_contract_entry["selection_path"]))
        require(e1_runtime_path.is_file() and sha256(e1_runtime_path) == e1_contract_entry["runtime_file_sha256"]
                and e1_selection_path.is_file() and sha256(e1_selection_path) == e1_contract_entry["selection_file_sha256"]
                and e1_contract_entry["contract_sha256"] == E1.E1_CONTRACT_SHA256, AUTH_BLOCK, "e1_contract_file")
        r7_contract = json.loads(e1_selection_path.read_text(encoding="utf-8"))
        E1.bind_e1_contract(r7_contract)
        require(bool(torch.backends.mps.is_built()) and bool(torch.backends.mps.is_available()), MPS_BLOCK, "mps_unavailable")
        device = torch.device("mps:0")
        preflight = {"mps_built": True, "mps_available": True, "device": str(device), "cpu_fallback": 0}
        adim, cdim = len(MC.AgentContext.FEATURE_NAMES), len(MC.LOCAL_SEARCH_FEATURE_NAMES)
        config, features = F1MOD.actor_config(FPS, H, adim, cdim), F1MOD.feature_contract(FPS, adim, cdim)
        require(int(config["candidate_dim"]) == 8 and int(config["global_dim"]) == 8 and int(config["demand_dim"]) == 6, AUTH_BLOCK, "v2_config")
        arms = list(dict(auth["envelope"])["selected_arms"])
        checkpoints = dict(checkpoint_contract["initial_inputs"])
        models = _load_models(arms=arms, checkpoints=checkpoints, config=config, H=H, JL=JL, device=device)
        frozen_before = dict(dict(auth["module_freeze_contract"])["frozen_source_hashes"]["actual"])
        root.mkdir(parents=True)
        snapshots: list[dict[str, Any]] = []
        rollouts: dict[str, Any] = {}
        prepared: dict[str, Any] = {}
        AUTH.reset_audit_log()
        with AUTH.granted(AUTH.SIMULATOR_EXECUTION, AUTH.TRAINING, AUTH.SHADOW_COUNTERFACTUAL,
                          reason="R18 exact one-shot SHA-bound R17 E1 bounded envelope"):
            for arm_id, model in models.items():
                torch.manual_seed(int(model["environment_seed"]))
                rollout = _rollout_arm(model=model, frozen=frozen_before, root=root, snapshot_entries=snapshots,
                                       H=H, JL=JL, CC=CC, CB=CB, PE=PE, MC=MC, BT1=BT1, BT6=BT6, R3=R3,
                                       F1MOD=F1MOD, FPS=FPS, S=S, compute_reward_v2=compute_reward_v2, config=config, features=features,
                                       device=device, counters=counters)
                support_guard = _assert_support_roundtrip(rows=rollout["rows"], CC=CC)
                prepared_cell = __import__("run_h4m_ae_ls3_bt8_r13_s3_four_cell_execution").build_credit_and_batch(
                    rollout=rollout, model={**model, "cell_id": arm_id}, JL=JL, CC=CC, E1=E1, R7MOD=R7MOD, FC=FC,
                    BT6=BT6, device=device, adim=adim, cdim=cdim)
                prepared_cell["support_guard"] = support_guard
                rollouts[arm_id], prepared[arm_id] = rollout, prepared_cell
            bd = prepared["BD_E1_R1"]
            bd_rows = rollouts["BD_E1_R1"]["rows"]
            by_band = {band: [row for row in bd_rows if row["time_band"] == band] for band in ("night", "offpeak", "peak")}
            require(all(any(not row["selected_is_no_assign"] for row in rows) and any(row["categorical_triggered"] for row in rows) for rows in by_band.values()),
                    BD_CREDIT_BLOCK, "bd_time_band_non_no_assign_or_sampling")
            require(sum(not row["selected_is_no_assign"] for row in bd_rows) > 0 and any(float(row["t"].assignment_discounted_reward) != 0.0 for row in bd_rows)
                    and int(bd["active_actor_rows"]) >= 1, BD_CREDIT_BLOCK, "bd_candidate_reward_or_ancestry")
            training = {arm_id: _train_arm(model=model, prepared=prepared[arm_id], support_guard=prepared[arm_id]["support_guard"],
                                            JL=JL, AUTH=AUTH, device=device, counters=counters)
                        for arm_id, model in models.items()}
            counters["training"] = 1; counters["mps_training"] = 1
        require((counters["actor_optimizer_step"], counters["critic_optimizer_step"], counters["raw_optimizer_step"], counters["causal_rollout"]) == (6, 6, 12, 96),
                INTEGRITY_BLOCK, "exact_budget")
        require(not any(AUTH.authorization_state()["capabilities"].values()), INTEGRITY_BLOCK, "capability_not_revoked")
        final_checkpoints = {arm_id: _write_final_checkpoint(root / "final_checkpoints" / f"{arm_id}_final.pt", model, training[arm_id])
                             for arm_id, model in models.items()}
        counters["checkpoint_write"] += len(final_checkpoints)
        require(counters["checkpoint_write"] == 2, INTEGRITY_BLOCK, "checkpoint_count")
        require(all(value == 0 for key, value in counters.items() if key in {"candidate_regeneration_after_selection", "candidate_regeneration_during_ppo", "local_search_rerun_during_ppo", "zero_loss_reevaluation_during_ppo", "future_leakage", "cross_window_gae", "duplicate_reward_ancestry", "test6_access", "github_push", "nan_or_inf", "candidate_identity_mismatch", "candidate_plan_execution_collapse", "serve_fallback", "zero_loss_violation", "illegal_or_masked_selection", "source_state_mutation_during_shadow_evaluation", "action_support_mutation"}), INTEGRITY_BLOCK, "integrity_counter")
        output = {
            "r18_execution_manifest.json": {"authorization_manifest_sha256": auth["authorization_sha256"], "source_commit": auth["source_commit"],
                                             "preflight": preflight, "arms": arms, "counters": counters, "selector": "R16 sealed E1 only"},
            "r18_learning_path_audit.json": {arm_id: {"actor_eligible": prepared[arm_id]["active_actor_rows"],
                                                        "reward_ancestry": sum(bool(row["actor_eligible_e1"]) for row in prepared[arm_id]["e1_rows"]),
                                                        "updates": training[arm_id]["updates"], "actor_changed": training[arm_id]["actor_changed"],
                                                        "critic_changed": training[arm_id]["critic_changed"]} for arm_id in models},
            "r18_candidate_support_audit.json": {arm_id: prepared[arm_id]["support_guard"] for arm_id in models},
            "r18_checkpoint_manifest.json": {"final": final_checkpoints, "test_only": True, "bounded": True, "non_promotable": True,
                                               "winner": False, "best_model": False, "promotion": False},
            "test_results.json": {"execution_counters": counters, "hard_failures": [], "warnings": [], "github_push": False},
            "gate_decision.json": {"stage": STAGE, "gate": PASS_GATE, "classification": "A_R18_E1_CAUSAL_LEARNING_PATH_OBSERVED",
                                   "source_commit": auth["source_commit"], "next_step": "separate frozen-policy review only"},
        }
        for name, value in output.items(): dump(root / name, value)
        (root / "final_report.md").write_text(f"# R18 final report\n\n- gate: `{PASS_GATE}`\n- source: `{auth['source_commit']}`\n", encoding="utf-8")
        manifest = {item.relative_to(root).as_posix(): sha256(item) for item in root.rglob("*") if item.is_file() and item.name != "manifest.json"}
        dump(root / "manifest.json", {"stage": STAGE, "gate": PASS_GATE, "source_commit": auth["source_commit"], "file_sha256": manifest})
        (root / "_SUCCESS.lock").write_text(PASS_GATE + "\n", encoding="utf-8")
        print(f"[PASS] {PASS_GATE}")
    except R18Error as exc:
        _block(root=root, code=exc.code, detail=str(exc), counters=counters, auth=auth)
        print(f"[BLOCKED] {exc.code}")
    except Exception as exc:  # noqa: BLE001
        _block(root=root, code=INTEGRITY_BLOCK, detail=f"{type(exc).__name__}:{exc}", counters=counters, auth=auth)
        print(f"[BLOCKED] {INTEGRITY_BLOCK}")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    auth = load_authorization(args.authorization_manifest, args.authorization_sha256)
    if args.dry_run:
        print(json.dumps(dry_run_report(auth), ensure_ascii=False, sort_keys=True))
        return
    execute(auth)


if __name__ == "__main__":
    main()
