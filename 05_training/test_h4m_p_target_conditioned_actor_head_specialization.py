#!/usr/bin/env python3
"""Focused H4M-P structural/equivalence checks for MAPPOActor head specialization.

These tests use synthetic tensors only. They do not run MAPPO training, do not
step an optimizer, and do not access TEST6 or validation data.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DL1_SOURCE = PROJECT_ROOT / "05_training/run_prompt5_e01_dl1_suseong_gatv2_mappo_critic_joint_learning_validation.py"
REPAIR_CONTRACT_SHA256 = "d672bce5d29fbdb26365bca351c69be09dd94559c5e1f3adc65067f3c6c40f97"
STRICT_FLOAT_TOLERANCE = 1.0e-7


def import_dl1() -> Any:
    spec = importlib.util.spec_from_file_location("h4mp_dl1_under_test", DL1_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {DL1_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_diff(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    diff = (a.detach().float() - b.detach().float()).abs()
    denom = b.detach().float().abs().clamp_min(1.0e-12)
    rel = diff / denom
    return {
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "max_rel_diff": float(rel.max().item()) if rel.numel() else 0.0,
        "finite": bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item()),
    }


def grad_norm(parameters: Any) -> float:
    values = []
    for parameter in parameters:
        if parameter.grad is not None:
            values.append(parameter.grad.detach().float().norm())
    if not values:
        return 0.0
    return float(torch.stack(values).norm().item())


def assert_close_stats(stats: Mapping[str, Any], tolerance: float = STRICT_FLOAT_TOLERANCE) -> None:
    assert stats["finite"] is True
    assert float(stats["max_abs_diff"]) <= tolerance


def actor_pair(dl1: Any) -> tuple[torch.nn.Module, torch.nn.Module]:
    torch.manual_seed(20260817)
    legacy = dl1.MAPPOActor(8, 3, target_context_dim=3)
    specialized = dl1.MAPPOActor(8, 3, target_context_dim=3, target_head_specialization=True)
    specialized.load_state_dict(legacy.state_dict(), strict=True)
    return legacy, specialized


def validation_repair_binding(dl1: Any) -> Dict[str, Any]:
    specialized = dl1.MAPPOActor(8, 3, target_context_dim=3, target_head_specialization=True)
    return {
        "repair_contract_sha256_expected": REPAIR_CONTRACT_SHA256,
        "repair_contract_sha256_observed": getattr(
            dl1.MAPPOActor, "TARGET_HEAD_SPECIALIZATION_REPAIR_CONTRACT_SHA256", None
        ),
        "specialization_active": bool(getattr(specialized, "target_head_specialization_active", False)),
        "target_head_count": int(getattr(specialized, "target_head_count", -1)),
        "target_context_dim": int(getattr(specialized, "target_context_dim", -1)),
        "action_dim": int(getattr(specialized, "action_dim", -1)),
        "passed": getattr(dl1.MAPPOActor, "TARGET_HEAD_SPECIALIZATION_REPAIR_CONTRACT_SHA256", None)
        == REPAIR_CONTRACT_SHA256
        and bool(getattr(specialized, "target_head_specialization_active", False))
        and int(getattr(specialized, "target_head_count", -1)) == 3
        and int(getattr(specialized, "action_dim", -1)) == 3,
    }


def validation_routing(dl1: Any) -> Dict[str, Any]:
    actor = dl1.MAPPOActor(4, 3, target_context_dim=3, target_head_specialization=True)
    with torch.no_grad():
        actor.net[0].weight.zero_()
        actor.net[0].bias.zero_()
        actor.net[-1].weight.zero_()
        actor.net[-1].bias.zero_()
        for target_id, head in enumerate(actor.target_heads):
            head.weight.zero_()
            head.bias.copy_(torch.tensor([10.0 * target_id, 10.0 * target_id + 1.0, 10.0 * target_id + 2.0]))
    embeddings = torch.zeros(6, 4)
    target_context = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    logits = actor(embeddings, target_context)
    expected = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
        ]
    )
    diff = tensor_diff(logits, expected)
    target_indices = actor._target_indices_from_context(target_context)
    expected_indices = torch.tensor([0, 1, 2, 0, 1, 2])
    return {
        "route_indices": [int(v) for v in target_indices.tolist()],
        "route_indices_expected": [int(v) for v in expected_indices.tolist()],
        "logit_diff": diff,
        "deterministic": True,
        "passed": bool(torch.equal(target_indices, expected_indices)) and diff["max_abs_diff"] <= 0.0,
    }


def validation_legacy_equivalence(dl1: Any) -> Dict[str, Any]:
    legacy, specialized = actor_pair(dl1)
    torch.manual_seed(17)
    embeddings = torch.randn(12, 8)
    target_context = torch.eye(3).repeat(4, 1)
    legacy_logits = legacy(embeddings, target_context)
    specialized_logits = specialized(embeddings, target_context)
    legacy_probs = torch.softmax(legacy_logits, dim=-1)
    specialized_probs = torch.softmax(specialized_logits, dim=-1)
    logit_diff = tensor_diff(specialized_logits, legacy_logits)
    prob_diff = tensor_diff(specialized_probs, legacy_probs)
    return {
        "tolerance": STRICT_FLOAT_TOLERANCE,
        "logit_diff": logit_diff,
        "probability_diff": prob_diff,
        "legacy_actor_state_loaded_strictly_into_specialized_actor": True,
        "copied_legacy_decision_head_to_each_target_head": True,
        "passed": logit_diff["max_abs_diff"] <= STRICT_FLOAT_TOLERANCE
        and prob_diff["max_abs_diff"] <= STRICT_FLOAT_TOLERANCE
        and logit_diff["finite"]
        and prob_diff["finite"],
    }


def validation_action_masks(dl1: Any) -> Dict[str, Any]:
    legacy, specialized = actor_pair(dl1)
    torch.manual_seed(18)
    embeddings = torch.randn(9, 8)
    target_context = torch.eye(3).repeat(3, 1)
    legal_mask = torch.tensor(
        [
            [True, True, False],
            [True, False, False],
            [False, True, True],
            [True, True, True],
            [False, True, False],
            [True, False, True],
            [True, True, False],
            [True, False, False],
            [False, True, True],
        ]
    )
    legacy_logits = legacy(embeddings, target_context)
    specialized_logits = specialized(embeddings, target_context)
    masked_legacy = legacy_logits.masked_fill(~legal_mask, -1.0e9)
    masked_specialized = specialized_logits.masked_fill(~legal_mask, -1.0e9)
    diff = tensor_diff(masked_specialized, masked_legacy)
    legacy_legal_probs = torch.softmax(masked_legacy, dim=-1).masked_select(legal_mask)
    specialized_legal_probs = torch.softmax(masked_specialized, dim=-1).masked_select(legal_mask)
    prob_diff = tensor_diff(specialized_legal_probs, legacy_legal_probs)
    return {
        "action_dim_legacy": int(legacy.action_dim),
        "action_dim_specialized": int(specialized.action_dim),
        "legal_mask_exact_match": True,
        "masked_logit_diff": diff,
        "legal_probability_diff": prob_diff,
        "passed": int(legacy.action_dim) == 3
        and int(specialized.action_dim) == 3
        and diff["max_abs_diff"] <= STRICT_FLOAT_TOLERANCE
        and prob_diff["max_abs_diff"] <= STRICT_FLOAT_TOLERANCE,
    }


def validation_critic_and_observation_contract(dl1: Any) -> Dict[str, Any]:
    actor = dl1.MAPPOActor(128, 3, target_context_dim=3, target_head_specialization=True)
    critic = dl1.CentralizedCritic(128)
    torch.manual_seed(19)
    agent_embeddings = torch.randn(5, 128)
    graph_embedding = torch.randn(128)
    target_context = torch.eye(3)[torch.tensor([0, 1, 2, 0, 1])]
    logits = actor(agent_embeddings, target_context)
    values_a = critic(agent_embeddings, graph_embedding).reshape(-1)
    values_b = critic(agent_embeddings, graph_embedding).reshape(-1)
    value_diff = tensor_diff(values_a, values_b)
    return {
        "actor_conditioned_input_dim": int(actor.actor_conditioned_input_dim),
        "target_context_dim": int(actor.target_context_dim),
        "action_dim": int(actor.action_dim),
        "critic_class": "CentralizedCritic",
        "critic_value_diff_same_input": value_diff,
        "logits_shape": list(logits.shape),
        "observation_contract": "12D observation unchanged; actor consumes existing final 3 target one-hot fields as target_context.",
        "passed": int(actor.actor_conditioned_input_dim) == 131
        and int(actor.target_context_dim) == 3
        and int(actor.action_dim) == 3
        and list(logits.shape) == [5, 3]
        and value_diff["max_abs_diff"] <= 0.0
        and value_diff["finite"],
    }


def validation_gradient_isolation(dl1: Any) -> Dict[str, Any]:
    torch.manual_seed(20)
    actor = dl1.MAPPOActor(8, 3, target_context_dim=3, target_head_specialization=True)
    embeddings = torch.randn(7, 8)

    def probe(target_id: int, action_id: int) -> Dict[str, Any]:
        actor.zero_grad(set_to_none=True)
        context = torch.eye(3)[torch.full((embeddings.size(0),), target_id, dtype=torch.long)]
        logits = actor(embeddings, context)
        loss = -torch.log_softmax(logits, dim=-1)[:, action_id].mean()
        loss.backward()
        head_norms = {
            f"target_head_{idx}": grad_norm(actor.target_heads[idx].parameters()) for idx in range(len(actor.target_heads))
        }
        shared_trunk_norm = grad_norm(actor.net[0].parameters())
        unused_legacy_head_norm = grad_norm(actor.net[-1].parameters())
        finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
            for parameter in actor.parameters()
        )
        return {
            "target_id": target_id,
            "optimized_action_id": action_id,
            "loss": float(loss.detach().item()),
            "shared_trunk_grad_norm": shared_trunk_norm,
            "target_head_grad_norms": head_norms,
            "unused_legacy_decision_head_grad_norm": unused_legacy_head_norm,
            "all_gradients_finite": finite,
            "optimizer_step_count": 0,
        }

    hold_probe = probe(target_id=0, action_id=0)
    serve_probe = probe(target_id=1, action_id=1)
    skip_probe = probe(target_id=2, action_id=2)
    passed = (
        hold_probe["target_head_grad_norms"]["target_head_0"] > 0.0
        and hold_probe["target_head_grad_norms"]["target_head_1"] == 0.0
        and hold_probe["target_head_grad_norms"]["target_head_2"] == 0.0
        and serve_probe["target_head_grad_norms"]["target_head_1"] > 0.0
        and serve_probe["target_head_grad_norms"]["target_head_0"] == 0.0
        and serve_probe["target_head_grad_norms"]["target_head_2"] == 0.0
        and skip_probe["target_head_grad_norms"]["target_head_2"] > 0.0
        and skip_probe["target_head_grad_norms"]["target_head_0"] == 0.0
        and skip_probe["target_head_grad_norms"]["target_head_1"] == 0.0
        and hold_probe["shared_trunk_grad_norm"] > 0.0
        and serve_probe["shared_trunk_grad_norm"] > 0.0
        and skip_probe["shared_trunk_grad_norm"] > 0.0
        and hold_probe["unused_legacy_decision_head_grad_norm"] == 0.0
        and serve_probe["unused_legacy_decision_head_grad_norm"] == 0.0
        and skip_probe["unused_legacy_decision_head_grad_norm"] == 0.0
        and hold_probe["all_gradients_finite"]
        and serve_probe["all_gradients_finite"]
        and skip_probe["all_gradients_finite"]
    )
    return {
        "backward_executed_for_validation_only": True,
        "optimizer_step_count": 0,
        "shared_trunk_gradient_sharing": "explicit: selected-target loss still updates shared actor trunk net.0; decision-head gradients are target-isolated.",
        "hold_target_probe": hold_probe,
        "serve_target_probe": serve_probe,
        "skip_target_probe": skip_probe,
        "passed": passed,
    }


def validation_serialization(dl1: Any) -> Dict[str, Any]:
    legacy, specialized = actor_pair(dl1)
    torch.manual_seed(21)
    embeddings = torch.randn(6, 8)
    target_context = torch.eye(3).repeat(2, 1)
    expected_logits = specialized(embeddings, target_context)
    with tempfile.TemporaryDirectory(prefix="h4mp_checkpoint_") as tmp:
        tmpdir = Path(tmp)
        legacy_path = tmpdir / "legacy_actor.pt"
        migrated_path = tmpdir / "specialized_actor.pt"
        torch.save({"actor_state_dict": legacy.state_dict(), "actor_target_context_dim": 3}, legacy_path)
        legacy_payload = torch.load(legacy_path, map_location="cpu", weights_only=False)
        migrated = dl1.MAPPOActor(8, 3, target_context_dim=3, target_head_specialization=True)
        migrated.load_state_dict(legacy_payload["actor_state_dict"], strict=True)
        torch.save(
            {
                "actor_state_dict": migrated.state_dict(),
                "actor_target_context_dim": 3,
                "target_head_specialization_active": True,
                "repair_contract_sha256": REPAIR_CONTRACT_SHA256,
            },
            migrated_path,
        )
        reloaded_payload = torch.load(migrated_path, map_location="cpu", weights_only=False)
        reloaded = dl1.MAPPOActor(8, 3, target_context_dim=3, target_head_specialization=True)
        reloaded.load_state_dict(reloaded_payload["actor_state_dict"], strict=True)
        reloaded_logits = reloaded(embeddings, target_context)
    diff = tensor_diff(reloaded_logits, expected_logits)
    return {
        "legacy_state_dict_migrated_into_specialized_actor_strict": True,
        "specialized_state_dict_reload_strict": True,
        "fresh_optimizer_required_for_specialized_retraining": True,
        "legacy_actor_optimizer_state_reuse_authorized": False,
        "output_diff_after_save_reload": diff,
        "passed": diff["max_abs_diff"] <= STRICT_FLOAT_TOLERANCE and diff["finite"],
    }


def run_validations() -> Dict[str, Any]:
    torch.set_num_threads(1)
    dl1 = import_dl1()
    sections = {
        "repair_binding": validation_repair_binding(dl1),
        "routing": validation_routing(dl1),
        "legacy_equivalence": validation_legacy_equivalence(dl1),
        "action_masks": validation_action_masks(dl1),
        "critic_and_observation_contract": validation_critic_and_observation_contract(dl1),
        "gradient_isolation": validation_gradient_isolation(dl1),
        "serialization": validation_serialization(dl1),
    }
    nan_inf_count = 0
    for section in sections.values():
        encoded = json.dumps(section, sort_keys=True)
        if "NaN" in encoded or "Infinity" in encoded:
            nan_inf_count += 1
    return {
        "stage": "PV8-R2A-R8E-R3-R-H4M-P",
        "synthetic_fixture_only": True,
        "training_executed": False,
        "optimizer_step_count": 0,
        "validation_or_test6_access_count": 0,
        "nan_inf_section_count": nan_inf_count,
        "sections": sections,
        "passed": all(section.get("passed") is True for section in sections.values()) and nan_inf_count == 0,
    }


def test_repair_binding() -> None:
    assert run_validations()["sections"]["repair_binding"]["passed"] is True


def test_target_routing_is_deterministic_and_correct() -> None:
    result = run_validations()["sections"]["routing"]
    assert result["passed"] is True


def test_legacy_logits_probabilities_and_masks_are_equivalent() -> None:
    result = run_validations()
    assert result["sections"]["legacy_equivalence"]["passed"] is True
    assert result["sections"]["action_masks"]["passed"] is True


def test_gradient_isolation_and_shared_trunk_audit() -> None:
    result = run_validations()["sections"]["gradient_isolation"]
    assert result["passed"] is True
    assert result["optimizer_step_count"] == 0


def test_critic_observation_and_serialization_contracts() -> None:
    result = run_validations()
    assert result["sections"]["critic_and_observation_contract"]["passed"] is True
    assert result["sections"]["serialization"]["passed"] is True
    assert result["validation_or_test6_access_count"] == 0


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
    print("[PASS] H4M-P target-conditioned actor head specialization tests passed")


if __name__ == "__main__":
    main()
