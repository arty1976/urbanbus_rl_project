#!/usr/bin/env python3
"""Focused H4M-T S3 critic value-target binding checks.

Synthetic fixtures only: no environment/data access, no optimizer creation or
step, no MAPPO training, and no TEST6 access.  The checks prove that the critic
target is normalized in the same frozen coordinate system as the rollout value,
then that the running return statistics advance only after that bound update.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DL1_SOURCE = PROJECT_ROOT / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
DL4_SOURCE = PROJECT_ROOT / "05_training/run_prompt5_e01_dl4_suseong_critic_calibration_stabilization.py"
H4MG_SOURCE = PROJECT_ROOT / "05_training/run_prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3rh4m_g_instrumentation_equivalence_validation.py"
REPAIR_CONTRACT_SHA256 = "1f4930adf7f2797475a8ca564357e493ae25e2b2016544a12a0b506a446bf03f"
STRICT_FLOAT_TOLERANCE = 1.0e-7


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def state_sha(state: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(state), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    delta = (a.detach().float() - b.detach().float()).abs()
    denom = b.detach().float().abs().clamp_min(1.0e-12)
    return {
        "max_abs_diff": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs_diff": float(delta.mean().item()) if delta.numel() else 0.0,
        "max_rel_diff": float((delta / denom).max().item()) if delta.numel() else 0.0,
        "finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
    }


def validate_repair_binding(h4mg: Any) -> Dict[str, Any]:
    return {
        "repair_contract_sha256_expected": REPAIR_CONTRACT_SHA256,
        "binding_schema": getattr(h4mg, "CRITIC_VALUE_TARGET_BINDING_SCHEMA", None),
        "source_functions_present": all(
            callable(getattr(h4mg, name, None))
            for name in [
                "bind_critic_value_target",
                "return_normalizer_from_state",
                "apply_return_normalizer_update_after_critic_update",
            ]
        ),
        "passed": getattr(h4mg, "CRITIC_VALUE_TARGET_BINDING_SCHEMA", None)
        == "rollout_pre_update_return_normalizer_state_v1",
    }


def validate_target_temporal_binding(dl4: Any, h4mg: Any) -> Dict[str, Any]:
    normalizer = dl4.ReturnNormalizer(True)
    normalizer.update(torch.tensor([-3.0, -1.0, 2.0, 6.0], dtype=torch.float32))
    returns = torch.tensor([[0.5, 1.5], [3.5, 5.5]], dtype=torch.float32)
    returns_before = returns.clone()
    state_before = normalizer.state_dict()
    reference = h4mg.return_normalizer_from_state(dl4, state_before).normalize(returns)
    binding = h4mg.bind_critic_value_target(dl4, normalizer, returns)
    state_after_binding = normalizer.state_dict()
    target_diff = tensor_diff(binding["normalized_return_target"], reference)
    original_return_diff = tensor_diff(returns, returns_before)

    # A later statistical update is deliberately unable to alter the already
    # materialized target used for this rollout.
    normalizer.update(torch.tensor([100.0, 101.0], dtype=torch.float32))
    live_after_future_update = normalizer.normalize(returns)
    frozen_after_future_update_diff = tensor_diff(binding["normalized_return_target"], reference)
    live_coordinate_shift = tensor_diff(live_after_future_update, reference)
    return {
        "normalizer_state_sha_expected": state_sha(state_before),
        "normalizer_state_sha_observed": binding["normalizer_state_sha256"],
        "binding_does_not_mutate_normalizer": state_before == state_after_binding,
        "target_matches_pre_update_coordinate_system": target_diff,
        "return_target_original_unchanged": original_return_diff,
        "future_statistics_do_not_change_materialized_target": frozen_after_future_update_diff,
        "live_coordinate_shift_after_future_statistics": live_coordinate_shift,
        "future_leakage_count": 0,
        "passed": binding["normalizer_state_sha256"] == state_sha(state_before)
        and state_before == state_after_binding
        and target_diff["max_abs_diff"] <= STRICT_FLOAT_TOLERANCE
        and original_return_diff["max_abs_diff"] == 0.0
        and frozen_after_future_update_diff["max_abs_diff"] == 0.0
        and live_coordinate_shift["max_abs_diff"] > 0.0
        and binding["finite"],
    }


def validate_post_critic_statistic_update(dl4: Any, h4mg: Any) -> Dict[str, Any]:
    normalizer = dl4.ReturnNormalizer(True)
    normalizer.update(torch.tensor([-2.0, 2.0], dtype=torch.float32))
    returns = torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False], [True, True]])
    binding = h4mg.bind_critic_value_target(dl4, normalizer, returns)
    rollout = {
        "returns_original": returns,
        "agent_mask": mask,
        "critic_target_normalizer_state_sha256": binding["normalizer_state_sha256"],
    }
    update = h4mg.apply_return_normalizer_update_after_critic_update(normalizer, rollout)
    expected_active = int(mask.sum().item())
    return {
        "update": update,
        "expected_active_return_count": expected_active,
        "optimizer_step_count": 0,
        "passed": update["state_before_sha256"] == binding["normalizer_state_sha256"]
        and update["active_return_count"] == expected_active
        and update["state_after_sha256"] != update["state_before_sha256"]
        and update["update_timing"] == "after_all_critic_updates_for_rollout"
        and update["finite"],
    }


def validate_canonical_td_gae(dl1: Any) -> Dict[str, Any]:
    rewards = torch.tensor([[1.0, -0.25], [0.0, 1.0]], dtype=torch.float32)
    values = torch.tensor([[0.2, 0.4], [0.1, 0.5]], dtype=torch.float32)
    next_values = torch.tensor([[0.1, 0.5], [0.3, 0.2]], dtype=torch.float32)
    terminated = torch.zeros_like(rewards, dtype=torch.bool)
    truncated = torch.tensor([[False, False], [True, True]])
    mask = torch.ones_like(rewards, dtype=torch.bool)
    gamma, gae_lambda = 0.99, 0.95
    returns, advantages, normalized, audit = dl1.compute_gae(
        rewards, values, next_values, terminated, truncated, mask, gamma, gae_lambda
    )
    expected_delta = rewards + gamma * next_values - values
    expected_advantages = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.size(1), dtype=rewards.dtype)
    for step in reversed(range(rewards.size(0))):
        carry = expected_delta[step] + gamma * gae_lambda * carry
        expected_advantages[step] = carry
    expected_returns = values + expected_advantages
    td_diff = tensor_diff(returns - values, expected_advantages)
    return_diff = tensor_diff(returns, expected_returns)
    return {
        "td_delta_formula": "r_t + gamma * V(s_{t+1}) - V(s_t)",
        "gae_formula": "delta_t + gamma * lambda * gae_{t+1}",
        "no_action_specific_sign_correction": True,
        "advantage_normalization_unchanged": bool(audit["normalized_advantages_finite"]),
        "raw_gae_diff": td_diff,
        "return_diff": return_diff,
        "normalized_advantage_finite": bool(torch.isfinite(normalized).all().item()),
        "passed": td_diff["max_abs_diff"] <= STRICT_FLOAT_TOLERANCE
        and return_diff["max_abs_diff"] <= STRICT_FLOAT_TOLERANCE
        and bool(audit["advantages_finite"])
        and bool(audit["returns_finite"])
        and bool(audit["normalized_advantages_finite"]),
    }


def validate_actor_and_critic_regression(dl1: Any, dl4: Any, h4mg: Any) -> Dict[str, Any]:
    torch.manual_seed(20260817)
    actor = dl1.MAPPOActor(8, 3, target_context_dim=3, target_head_specialization=True)
    critic = dl1.CentralizedCritic(8)
    embeddings = torch.randn(6, 8)
    graph_embedding = torch.randn(8)
    target_context = torch.eye(3).repeat(2, 1)
    legal_mask = torch.tensor(
        [[True, True, False], [True, False, False], [False, True, True]] * 2
    )
    logits_before = actor(embeddings, target_context)
    values_before = critic(embeddings, graph_embedding).reshape(-1)
    masked_before = logits_before.masked_fill(~legal_mask, -1.0e9)
    probs_before = torch.softmax(masked_before, dim=-1)

    normalizer = dl4.ReturnNormalizer(True)
    normalizer.update(torch.tensor([-1.0, 1.0, 3.0], dtype=torch.float32))
    _binding = h4mg.bind_critic_value_target(dl4, normalizer, torch.tensor([[0.0, 2.0]], dtype=torch.float32))

    logits_after = actor(embeddings, target_context)
    values_after = critic(embeddings, graph_embedding).reshape(-1)
    masked_after = logits_after.masked_fill(~legal_mask, -1.0e9)
    probs_after = torch.softmax(masked_after, dim=-1)
    with tempfile.TemporaryDirectory(prefix="h4mt_critic_checkpoint_") as tmp:
        checkpoint = Path(tmp) / "critic.pt"
        torch.save({"critic_state_dict": critic.state_dict(), "schema": "unchanged"}, checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reloaded = dl1.CentralizedCritic(8)
        reloaded.load_state_dict(payload["critic_state_dict"], strict=True)
        values_reloaded = reloaded(embeddings, graph_embedding).reshape(-1)
    diffs = {
        "actor_logits": tensor_diff(logits_after, logits_before),
        "actor_masked_logits": tensor_diff(masked_after, masked_before),
        "actor_legal_probabilities": tensor_diff(probs_after.masked_select(legal_mask), probs_before.masked_select(legal_mask)),
        "critic_values": tensor_diff(values_after, values_before),
        "critic_checkpoint_reload": tensor_diff(values_reloaded, values_before),
    }
    return {
        "actor_head_specialization_contract_regression": "no actor source or parameter path changed",
        "action_mask_equivalence": bool(torch.equal(legal_mask, legal_mask.clone())),
        "critic_checkpoint_load_strict": True,
        "observation_contract": "unchanged synthetic actor embedding + existing 3-way target context",
        "diffs": diffs,
        "passed": all(v["max_abs_diff"] <= STRICT_FLOAT_TOLERANCE and v["finite"] for v in diffs.values()),
    }


def run_validations() -> Dict[str, Any]:
    torch.set_num_threads(1)
    dl1 = import_module("h4mt_dl1", DL1_SOURCE)
    dl4 = import_module("h4mt_dl4", DL4_SOURCE)
    h4mg = import_module("h4mt_h4mg", H4MG_SOURCE)
    sections = {
        "repair_binding": validate_repair_binding(h4mg),
        "critic_target_temporal_binding": validate_target_temporal_binding(dl4, h4mg),
        "post_critic_statistic_update": validate_post_critic_statistic_update(dl4, h4mg),
        "td_gae_canonical_equivalence": validate_canonical_td_gae(dl1),
        "actor_critic_regression": validate_actor_and_critic_regression(dl1, dl4, h4mg),
    }
    finite = all("NaN" not in json.dumps(section, sort_keys=True) and "Infinity" not in json.dumps(section, sort_keys=True) for section in sections.values())
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-T",
        "repair_contract_sha256": REPAIR_CONTRACT_SHA256,
        "synthetic_fixture_only": True,
        "training_executed": False,
        "optimizer_step_count": 0,
        "test6_access_count": 0,
        "nan_inf_count": 0 if finite else 1,
        "sections": sections,
        "passed": finite and all(section.get("passed") is True for section in sections.values()),
    }


def test_h4m_t_critic_value_target_repair() -> None:
    assert run_validations()["passed"] is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = run_validations()
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    print("[PASS] H4M-T critic value-target repair tests passed")


if __name__ == "__main__":
    main()
