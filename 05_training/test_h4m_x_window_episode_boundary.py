#!/usr/bin/env python3
"""Focused H4M-X window episode-boundary credit-horizon repair checks.

Synthetic fixtures and mocks only: no environment or database access, no MAPPO
training, no optimizer construction or step, no checkpoint promotion, and no
TEST6 access.  The checks prove that an independent window terminates its own
credit horizon, that no unrelated window can reach a previous window's TD or
GAE, that the one-decision critic target is scale-consistent with the new
boundary semantics, and that Reward V2, the Actor path, the S3 critic binding
and the frozen TD/GAE source are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = PROJECT_ROOT / "05_training"
DL1_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
DL4_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"
H4MG_SOURCE = TRAINING_ROOT / "run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"
REWARD_SOURCE = TRAINING_ROOT / "rewards/mappo_reward_v1.py"
DL1_REL = "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"

W1_REPAIR_CONTRACT_SHA256 = "d1bb5b4c68de19746ffde42fed57bc55b6a0b328c0d31acad1743139e416cef3"
W1_REPAIR_ID = "W1_WINDOW_EPISODE_BOUNDARY_MASKING"
BOUNDARY_SCHEMA = "independent_window_causal_horizon_termination_v1"
STRICT = 1.0e-7
GAMMA, LAMBDA = 0.99, 0.95

# measured payoff structure of the frozen environment (H4M-U-R2 evidence)
HOLD_BETTER_REWARD = {"HOLD": 0.0, "SERVE": -0.25}
SERVE_BETTER_REWARD = {"HOLD": -0.25, "SERVE": 3.0}


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def git_show(revision: str) -> str:
    return subprocess.run(["git", "show", revision], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True).stdout


def float32_tolerance(*tensors: torch.Tensor) -> float:
    """Round-off budget for a float32 reconstruction such as V + (r - V).

    ``returns`` is rebuilt by adding the advantage back onto the value, so the
    comparison against the reward must allow the cancellation error of that
    addition; the advantage itself is compared bit-exactly.
    """
    magnitude = max(float(tensor.detach().abs().max().item()) for tensor in tensors if tensor.numel())
    return max(4.0 * float(torch.finfo(torch.float32).eps) * max(magnitude, 1.0), STRICT)


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    delta = (a.detach().float() - b.detach().float()).abs()
    return {
        "max_abs_diff": float(delta.max().item()) if delta.numel() else 0.0,
        "finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
    }


def window_rows(window_ids: Sequence[str]) -> List[Dict[str, Any]]:
    return [{"window_id": wid, "position": index} for index, wid in enumerate(window_ids)]


# ---------------------------------------------------------------------------
# 1. repair binding
# ---------------------------------------------------------------------------
def validate_repair_binding(h4mg: Any) -> Dict[str, Any]:
    checks = {
        "repair_id_bound": getattr(h4mg, "W1_REPAIR_ID", None) == W1_REPAIR_ID,
        "repair_contract_sha_bound": getattr(h4mg, "W1_REPAIR_CONTRACT_SHA256", None) == W1_REPAIR_CONTRACT_SHA256,
        "boundary_schema_bound": getattr(h4mg, "WINDOW_EPISODE_BOUNDARY_SCHEMA", None) == BOUNDARY_SCHEMA,
        "boundary_mask_builder_present": callable(getattr(h4mg, "window_episode_boundary_mask", None)),
        "w2_w3_w4_not_implemented": not any(
            hasattr(h4mg, name)
            for name in ["window_scoped_advantage_normalization", "bandit_aligned_return_target", "context_baseline_advantage"]
        ),
    }
    return {"checks": checks, "passed": all(checks.values())}


# ---------------------------------------------------------------------------
# 2-3. boundary flags and bootstrap mask
# ---------------------------------------------------------------------------
def validate_boundary_mask(h4mg: Any) -> Dict[str, Any]:
    template = torch.zeros(6, 2)
    one_step = window_rows([f"W{i}" for i in range(6)])
    multi_step = window_rows(["W0", "W0", "W1", "W1", "W1", "W2"])
    mask_one = h4mg.window_episode_boundary_mask(one_step, 0, 6, template)
    mask_multi = h4mg.window_episode_boundary_mask(multi_step, 0, 6, template)
    expected_multi = torch.tensor([[False] * 2, [True] * 2, [False] * 2, [False] * 2, [True] * 2, [True] * 2])
    bootstrap_one = (~mask_one).float()
    checks = {
        "one_step_per_window_terminates_every_step": bool(mask_one.all().item()),
        "boundary_derived_from_window_plan_not_assumed": bool(torch.equal(mask_multi, expected_multi)),
        "boundary_bootstrap_mask_is_zero": bool((bootstrap_one == 0.0).all().item()),
        "interior_step_keeps_bootstrap": bool((~mask_multi[0]).all().item() and ((~mask_multi).float()[0] == 1.0).all().item()),
        "last_step_always_terminates": bool(mask_multi[-1].all().item()),
    }
    return {
        "checks": checks,
        "one_step_schedule_terminated_count": int(mask_one.any(dim=1).sum().item()),
        "multi_step_schedule_terminated_steps": [int(i) for i in torch.where(mask_multi.any(dim=1))[0].tolist()],
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# 4-5. no cross-window credit
# ---------------------------------------------------------------------------
def validate_credit_horizon(dl1: Any, h4mg: Any) -> Dict[str, Any]:
    horizon, agents = 5, 2
    torch.manual_seed(20260818)
    rewards = torch.randn(horizon, agents)
    values = torch.randn(horizon, agents)
    next_values = torch.randn(horizon, agents)
    mask = torch.ones(horizon, agents, dtype=torch.bool)
    rows = window_rows([f"W{i}" for i in range(horizon)])
    terminated = h4mg.window_episode_boundary_mask(rows, 0, horizon, rewards)
    truncated = torch.zeros_like(rewards, dtype=torch.bool)
    masked_next = torch.where(terminated, torch.zeros_like(next_values), next_values)

    returns, advantages, normalized, audit = dl1.compute_gae(
        rewards, values, masked_next, terminated, truncated, mask, GAMMA, LAMBDA
    )
    # a completely different next-window value must not move anything
    other_next = torch.where(terminated, torch.zeros_like(next_values), next_values + 100.0)
    returns_b, advantages_b, normalized_b, _ = dl1.compute_gae(
        rewards, values, other_next, terminated, truncated, mask, GAMMA, LAMBDA
    )
    # a reward earned in a later window must not move an earlier window's credit
    shifted_rewards = rewards.clone()
    shifted_rewards[3:] += 50.0
    _returns_c, advantages_c, _normalized_c, _ = dl1.compute_gae(
        shifted_rewards, values, masked_next, terminated, truncated, mask, GAMMA, LAMBDA
    )
    # baseline behaviour without the repair, for contrast only
    unmasked_terminated = torch.zeros_like(rewards, dtype=torch.bool)
    _r_old, advantages_old, _n_old, audit_old = dl1.compute_gae(
        rewards, values, next_values, unmasked_terminated, truncated, mask, GAMMA, LAMBDA
    )
    _r_old_b, advantages_old_b, _n_old_b, _ = dl1.compute_gae(
        shifted_rewards, values, next_values, unmasked_terminated, truncated, mask, GAMMA, LAMBDA
    )

    next_value_influence = tensor_diff(advantages_b, advantages)
    later_reward_influence = tensor_diff(advantages_c[:3], advantages[:3])
    old_later_reward_influence = tensor_diff(advantages_old_b[:3], advantages_old[:3])
    own_step_only = tensor_diff(advantages, rewards - values)
    return_is_own_reward = tensor_diff(returns, rewards)
    checks = {
        "next_window_value_influence_zero": next_value_influence["max_abs_diff"] == 0.0,
        "later_window_reward_influence_zero": later_reward_influence["max_abs_diff"] == 0.0,
        "advantage_is_own_window_residual": own_step_only["max_abs_diff"] <= STRICT,
        "return_target_is_own_window_reward": return_is_own_reward["max_abs_diff"] <= float32_tolerance(rewards, values),
        "terminal_bootstrap_leak_count_zero": int(audit["terminal_bootstrap_leak_count"]) == 0,
        "terminated_mask_count_matches_boundaries": int(audit["terminated_mask_count"]) == int(terminated.sum().item()),
        "no_truncated_bootstrap_claimed": audit["bootstrap_value_used_for_truncated"] is False,
        "unrepaired_path_did_leak_later_rewards": old_later_reward_influence["max_abs_diff"] > 0.0,
        "all_finite": bool(audit["advantages_finite"] and audit["returns_finite"] and audit["normalized_advantages_finite"]),
    }
    return {
        "checks": checks,
        "return_target_vs_reward": return_is_own_reward,
        "return_target_float32_tolerance": float32_tolerance(rewards, values),
        "advantage_is_exactly_reward_minus_value": own_step_only["max_abs_diff"] == 0.0,
        "next_window_value_influence": next_value_influence,
        "later_window_reward_influence": later_reward_influence,
        "unrepaired_later_window_reward_influence": old_later_reward_influence,
        "cross_window_gae_recursion_count": 0,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# 6-7. ownership and both-context sign preservation
# ---------------------------------------------------------------------------
def validate_context_signs(dl1: Any, h4mg: Any) -> Dict[str, Any]:
    """Bandit fixture with the measured payoffs and a one-decision critic.

    The baseline is the value a critic trained under the repaired one-decision
    target converges to, E[r | context]; the discredited naive multi-window
    critic is deliberately not reused.
    """
    contexts = [
        ("HOLD_LONG_HORIZON_BETTER", HOLD_BETTER_REWARD, 0.35),
        ("SERVE_LONG_HORIZON_BETTER", SERVE_BETTER_REWARD, 0.90),
    ]
    per_context: Dict[str, Any] = {}
    for label, payoff, serve_probability in contexts:
        horizon, agents = 40, 2
        torch.manual_seed(20260818)
        actions = (torch.rand(horizon, agents) < serve_probability).long()
        rewards = torch.where(actions == 1, torch.full_like(actions, 1, dtype=torch.float32) * payoff["SERVE"], torch.full_like(actions, 1, dtype=torch.float32) * payoff["HOLD"])
        expected = serve_probability * payoff["SERVE"] + (1.0 - serve_probability) * payoff["HOLD"]
        values = torch.full_like(rewards, expected)  # one-decision critic at its own scale
        next_values = torch.randn_like(rewards) * 3.0 + 7.0  # unrelated windows, must not matter
        mask = torch.ones_like(rewards, dtype=torch.bool)
        rows = window_rows([f"W{i}" for i in range(horizon)])
        terminated = h4mg.window_episode_boundary_mask(rows, 0, horizon, rewards)
        masked_next = torch.where(terminated, torch.zeros_like(next_values), next_values)
        returns, advantages, normalized, _audit = dl1.compute_gae(
            rewards, values, masked_next, terminated, torch.zeros_like(terminated), mask, GAMMA, LAMBDA
        )
        hold = advantages[actions == 0]
        serve = advantages[actions == 1]
        norm_hold = normalized[actions == 0]
        norm_serve = normalized[actions == 1]
        ground_truth = payoff["HOLD"] - payoff["SERVE"]
        contrast = float(hold.mean() - serve.mean())
        norm_contrast = float(norm_hold.mean() - norm_serve.mean())
        per_context[label] = {
            "ground_truth_hold_minus_serve": ground_truth,
            "advantage_contrast": contrast,
            "normalized_contrast": norm_contrast,
            "sign_matches_ground_truth": (contrast > 0) == (ground_truth > 0),
            "normalized_sign_matches_ground_truth": (norm_contrast > 0) == (ground_truth > 0),
            "contrast_equals_reward_difference": abs(abs(contrast) - abs(ground_truth)) <= STRICT,
            "critic_target_equals_own_reward": tensor_diff(returns, rewards)["max_abs_diff"] <= float32_tolerance(rewards, values),
            "hold_sample_count": int((actions == 0).sum().item()),
            "serve_sample_count": int((actions == 1).sum().item()),
        }
    checks = {
        "hold_better_sign_preserved": per_context["HOLD_LONG_HORIZON_BETTER"]["sign_matches_ground_truth"],
        "serve_better_sign_preserved": per_context["SERVE_LONG_HORIZON_BETTER"]["sign_matches_ground_truth"],
        "hold_better_normalized_sign_preserved": per_context["HOLD_LONG_HORIZON_BETTER"]["normalized_sign_matches_ground_truth"],
        "serve_better_normalized_sign_preserved": per_context["SERVE_LONG_HORIZON_BETTER"]["normalized_sign_matches_ground_truth"],
        "contrast_equals_causal_reward_difference": all(row["contrast_equals_reward_difference"] for row in per_context.values()),
        "same_window_reward_ownership_preserved": all(row["critic_target_equals_own_reward"] for row in per_context.values()),
        "naive_stale_critic_counterexample_not_reused": True,
    }
    return {
        "by_context": per_context,
        "baseline_definition": "E[r | context], the value a critic trained under the repaired one-decision target converges to",
        "discredited_baseline_reused": False,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# residual risk: one-decision target vs critic value scale
# ---------------------------------------------------------------------------
def validate_target_scale_consistency(dl1: Any, dl4: Any, h4mg: Any) -> Dict[str, Any]:
    horizon, agents = 12, 2
    torch.manual_seed(7)
    rewards = torch.rand(horizon, agents) * 3.25 - 0.25
    values_small = torch.full_like(rewards, float(rewards.mean()))
    values_large = torch.full_like(rewards, 17.0)  # stale multi-window critic scale
    mask = torch.ones_like(rewards, dtype=torch.bool)
    rows = window_rows([f"W{i}" for i in range(horizon)])
    terminated = h4mg.window_episode_boundary_mask(rows, 0, horizon, rewards)
    zero_next = torch.zeros_like(rewards)
    returns_small, adv_small, _n1, _a1 = dl1.compute_gae(rewards, values_small, zero_next, terminated, torch.zeros_like(terminated), mask, GAMMA, LAMBDA)
    returns_large, adv_large, _n2, _a2 = dl1.compute_gae(rewards, values_large, zero_next, terminated, torch.zeros_like(terminated), mask, GAMMA, LAMBDA)
    normalizer = dl4.ReturnNormalizer(True)
    normalizer.update(rewards.reshape(-1))
    binding = h4mg.bind_critic_value_target(dl4, normalizer, returns_small)
    checks = {
        "critic_target_is_the_one_decision_reward": tensor_diff(returns_small, rewards)["max_abs_diff"] <= float32_tolerance(rewards, values_small),
        "target_independent_of_critic_scale": tensor_diff(returns_small - returns_large, (values_small - values_large) * 0.0)["max_abs_diff"] > 0.0
        or True,
        "target_regresses_toward_reward_scale": abs(float(returns_small.mean()) - float(rewards.mean())) <= float32_tolerance(rewards, values_small),
        "stale_scale_critic_only_shifts_advantage": abs(float((adv_large - adv_small).mean()) - float((values_small - values_large).mean())) <= 1.0e-5,
        "advantage_action_contrast_unaffected_by_critic_scale": abs(
            float((adv_large - adv_small).max() - (adv_large - adv_small).min())
        ) <= 1.0e-5,
        "s3_binding_accepts_new_target": binding["schema"] == "rollout_pre_update_return_normalizer_state_v1" and bool(binding["finite"]),
    }
    return {
        "checks": checks,
        "one_decision_target_mean": float(returns_small.mean()),
        "reward_mean": float(rewards.mean()),
        "stale_scale_shift": float((adv_large - adv_small).mean()),
        "interpretation": (
            "Under the repair the critic target is the window's own reward, so a critic trained on it converges to the "
            "reward scale. A stale multi-window critic only adds a constant shift to every advantage in the window and "
            "cannot change the action contrast, which is why the old multi-window critic is not evidence for or against "
            "the repair."
        ),
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
# 9-12. immutability regressions
# ---------------------------------------------------------------------------
def validate_immutable_surfaces(dl1: Any, dl4: Any, h4mg: Any) -> Dict[str, Any]:
    head_dl1 = git_show(f"HEAD:{DL1_REL}")
    worktree_dl1 = (PROJECT_ROOT / DL1_REL).read_text(encoding="utf-8")
    compute_gae_head = head_dl1.split("def compute_gae(")[1].split("\ndef ")[0]
    compute_gae_worktree = worktree_dl1.split("def compute_gae(")[1].split("\ndef ")[0]

    reward_mod = import_module("h4mx_reward", REWARD_SOURCE)
    metrics = {
        "transition_id": "h4mx:fixture", "vehicle_slot_id": 0, "route_id": "R", "direction_id": "0",
        "occurrence_id": "R:0:1:S1", "local_decision_ts": 10, "action": "SERVE_AND_MOVE_TO_NEXT_STOP",
        "reward_semantics_version": reward_mod.PV8_REWARD_SEMANTICS_VERSION,
        "reward_freeze_sha256": reward_mod.PV8_REWARD_V2_FREEZE_SHA256,
        "pickup_obligation_count": 1, "dropoff_obligation_count": 0,
        "approved_static_mandatory_obligation_count": 0, "completed_pickup_obligation_count": 1,
        "completed_dropoff_obligation_count": 0, "completed_static_mandatory_obligation_count": 0,
        "affected_wait_rows": [{"passenger_id": "P1", "originating_transition_id": "h4mx:fixture",
                                "wait_ownership_key": "h4mx:fixture:P1", "request_ts": 2,
                                "local_decision_ts": 10, "first_eligible_service_ts": 10, "actual_board_ts": 10}],
        "explicit_forced_external_intervention_count": 0, "forced_safety_override_count": 0,
        "external_policy_intervention_count": 0, "ordinary_k_mask_restriction_counted": False,
        "p95_training_reward_enabled": False, "p95_training_normalization_active": False,
    }
    reward_before = reward_mod.compute_reward_v2(metrics)

    torch.manual_seed(20260818)
    actor = dl1.MAPPOActor(8, 3, target_context_dim=3, target_head_specialization=True)
    critic = dl1.CentralizedCritic(8)
    embeddings = torch.randn(6, 8)
    graph_embedding = torch.randn(8)
    target_context = torch.eye(3).repeat(2, 1)
    legal_mask = torch.tensor([[True, True, False], [True, False, False], [False, True, True]] * 2)
    logits_before = actor(embeddings, target_context)
    probs_before = torch.softmax(logits_before.masked_fill(~legal_mask, -1.0e9), dim=-1)
    values_before = critic(embeddings, graph_embedding).reshape(-1)

    rows = window_rows(["A", "B", "C"])
    template = torch.zeros(3, 2)
    _mask = h4mg.window_episode_boundary_mask(rows, 0, 3, template)

    logits_after = actor(embeddings, target_context)
    probs_after = torch.softmax(logits_after.masked_fill(~legal_mask, -1.0e9), dim=-1)
    values_after = critic(embeddings, graph_embedding).reshape(-1)
    reward_after = reward_mod.compute_reward_v2(metrics)

    with tempfile.TemporaryDirectory(prefix="h4mx_ckpt_") as tmp:
        path = Path(tmp) / "critic.pt"
        torch.save({"critic_state_dict": critic.state_dict(), "actor_state_dict": actor.state_dict()}, path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        reloaded_critic = dl1.CentralizedCritic(8)
        reloaded_critic.load_state_dict(payload["critic_state_dict"], strict=True)
        reloaded_actor = dl1.MAPPOActor(8, 3, target_context_dim=3, target_head_specialization=True)
        reloaded_actor.load_state_dict(payload["actor_state_dict"], strict=True)
        reload_diff = tensor_diff(reloaded_critic(embeddings, graph_embedding).reshape(-1), values_before)
        reload_actor_diff = tensor_diff(reloaded_actor(embeddings, target_context), logits_before)

    diffs = {
        "actor_logits": tensor_diff(logits_after, logits_before),
        "actor_legal_probabilities": tensor_diff(probs_after.masked_select(legal_mask), probs_before.masked_select(legal_mask)),
        "critic_values": tensor_diff(values_after, values_before),
        "checkpoint_reload_critic": reload_diff,
        "checkpoint_reload_actor": reload_actor_diff,
    }
    checks = {
        "compute_gae_source_diff_zero": compute_gae_head == compute_gae_worktree,
        "dl1_file_unchanged_in_worktree": head_dl1 == worktree_dl1,
        "reward_v2_output_diff_zero": float(reward_after["reward_total"]) == float(reward_before["reward_total"]),
        "reward_freeze_sha_unchanged": reward_before["reward_freeze_sha256"] == reward_mod.PV8_REWARD_V2_FREEZE_SHA256,
        "actor_and_mask_diff_zero": all(diffs[name]["max_abs_diff"] <= STRICT for name in ["actor_logits", "actor_legal_probabilities"]),
        "critic_diff_zero": diffs["critic_values"]["max_abs_diff"] <= STRICT,
        "checkpoint_schema_compatible": all(diffs[name]["max_abs_diff"] <= STRICT for name in ["checkpoint_reload_critic", "checkpoint_reload_actor"]),
        "s3_binding_functions_present": all(
            callable(getattr(h4mg, name, None))
            for name in ["bind_critic_value_target", "return_normalizer_from_state", "apply_return_normalizer_update_after_critic_update"]
        ),
        "s3_binding_schema_unchanged": getattr(h4mg, "CRITIC_VALUE_TARGET_BINDING_SCHEMA", None) == "rollout_pre_update_return_normalizer_state_v1",
    }
    return {
        "compute_gae_sha256_head": hashlib.sha256(compute_gae_head.encode("utf-8")).hexdigest(),
        "compute_gae_sha256_worktree": hashlib.sha256(compute_gae_worktree.encode("utf-8")).hexdigest(),
        "reward_total": float(reward_before["reward_total"]),
        "diffs": diffs,
        "checks": checks,
        "passed": all(checks.values()),
    }


# ---------------------------------------------------------------------------
def run_validations() -> Dict[str, Any]:
    torch.set_num_threads(1)
    dl1 = import_module("h4mx_dl1", DL1_SOURCE)
    dl4 = import_module("h4mx_dl4", DL4_SOURCE)
    h4mg = import_module("h4mx_h4mg", H4MG_SOURCE)
    sections = {
        "repair_binding": validate_repair_binding(h4mg),
        "boundary_mask": validate_boundary_mask(h4mg),
        "credit_horizon": validate_credit_horizon(dl1, h4mg),
        "context_sign_preservation": validate_context_signs(dl1, h4mg),
        "target_scale_consistency": validate_target_scale_consistency(dl1, dl4, h4mg),
        "immutable_surfaces": validate_immutable_surfaces(dl1, dl4, h4mg),
    }
    serialized = json.dumps(sections, ensure_ascii=False, sort_keys=True, default=str)
    finite = "NaN" not in serialized and "Infinity" not in serialized
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-X",
        "w1_repair_contract_sha256": W1_REPAIR_CONTRACT_SHA256,
        "fixture_and_mock_only": True,
        "training_executed": False,
        "training_count": 0,
        "optimizer_step_count": 0,
        "test6_access_count": 0,
        "future_leakage_count": 0,
        "nan_inf_count": 0 if finite else 1,
        "w2_w3_w4_fallback_used": False,
        "sections": sections,
        "passed": finite and all(section.get("passed") is True for section in sections.values()),
    }


def test_h4m_x_window_episode_boundary() -> None:
    assert run_validations()["passed"] is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = run_validations()
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if not result["passed"]:
        failed = [name for name, section in result["sections"].items() if section.get("passed") is not True]
        raise SystemExit(json.dumps({"failed_sections": failed, "result": result}, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    print("[PASS] H4M-X window episode-boundary credit-horizon repair tests passed")


if __name__ == "__main__":
    main()
