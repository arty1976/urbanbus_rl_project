"""
policy_inference_boundary_v1.py
===============================

Actual MAPPO policy inference boundary for UrbanBus RL.

MAPPO means Multi-Agent Proximal Policy Optimization.
Qwen must remain disabled for A/A90/A80/A70 in the current phase.

Purpose
-------
This module creates a hard boundary between:

1. placeholder policy outputs
2. replay/non-causal validation outputs
3. actual causal MAPPO policy inference outputs

This file does NOT implement neural network inference yet.
It implements the guardrail that prevents actual MAPPO conditions from
silently falling back to placeholder policy.

Core rule
---------
If condition_id is A/A90/A80/A70 and policy_kind is mappo,
checkpoint_path must be provided.

If require_existing_checkpoint=True, the checkpoint file must exist.

No placeholder fallback is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


try:
    from policy_registry_v1 import (
        default_actual_mappo_spec,
        default_placeholder_spec,
        make_policy_spec,
        normalize_condition_id,
    )
except Exception:
    # Allows import when package context is different.
    from .policy_registry_v1 import (
        default_actual_mappo_spec,
        default_placeholder_spec,
        make_policy_spec,
        normalize_condition_id,
    )


A_FAMILY_CONDITIONS = {
    "A",
    "A90",
    "A80",
    "A70",
}


@dataclass(frozen=True)
class PolicyInferenceRequest:
    condition_id: str
    policy_kind: str
    checkpoint_path: Optional[str] = None
    qwen_enabled: bool = False
    causal: bool = False
    require_existing_checkpoint: bool = False


@dataclass(frozen=True)
class PolicyInferenceBoundary:
    condition_id: str
    policy_kind: str
    policy_source: str
    source_mode: str
    checkpoint_path: Optional[str]
    qwen_trigger_rate: float
    causal_claim_allowed: bool
    placeholder_allowed: bool
    actual_mappo_inference_ready: bool
    failure_if_used_for_claim: Optional[str]
    notes: str


def is_a_family(condition_id: Any) -> bool:
    return normalize_condition_id(condition_id) in A_FAMILY_CONDITIONS


def checkpoint_exists(checkpoint_path: Optional[str]) -> bool:
    if not checkpoint_path:
        return False
    return Path(checkpoint_path).exists()


def build_policy_inference_boundary(
    request: PolicyInferenceRequest,
) -> PolicyInferenceBoundary:
    cid = normalize_condition_id(request.condition_id)
    kind = str(request.policy_kind).strip().lower()

    if cid in A_FAMILY_CONDITIONS and request.qwen_enabled:
        raise ValueError("A/A90/A80/A70 must keep qwen_enabled=False in the current phase")

    if kind == "mappo":
        if not request.checkpoint_path:
            raise FileNotFoundError(
                "Actual MAPPO inference requires checkpoint_path. "
                "No placeholder fallback is allowed."
            )

        if request.require_existing_checkpoint and not checkpoint_exists(request.checkpoint_path):
            raise FileNotFoundError(
                f"MAPPO checkpoint does not exist: {request.checkpoint_path}"
            )

        spec = default_actual_mappo_spec(
            condition_id=cid,
            checkpoint_path=request.checkpoint_path,
            require_existing_checkpoint=request.require_existing_checkpoint,
        )

        return PolicyInferenceBoundary(
            condition_id=cid,
            policy_kind="mappo",
            policy_source=spec["policy_source"],
            source_mode=spec["source_mode"],
            checkpoint_path=request.checkpoint_path,
            qwen_trigger_rate=0.0,
            causal_claim_allowed=True,
            placeholder_allowed=False,
            actual_mappo_inference_ready=True,
            failure_if_used_for_claim=None,
            notes=(
                "Actual MAPPO policy boundary accepted. "
                "Neural network inference implementation is still connected later."
            ),
        )

    if kind == "placeholder":
        if request.causal:
            raise ValueError("Placeholder policy cannot be used for causal rollout claims")

        spec = default_placeholder_spec(cid)

        return PolicyInferenceBoundary(
            condition_id=cid,
            policy_kind="placeholder",
            policy_source=spec["policy_source"],
            source_mode=spec["source_mode"],
            checkpoint_path=None,
            qwen_trigger_rate=0.0,
            causal_claim_allowed=False,
            placeholder_allowed=True,
            actual_mappo_inference_ready=False,
            failure_if_used_for_claim="placeholder policy is smoke/contract only",
            notes="Placeholder policy accepted only for smoke/contract validation.",
        )

    # For baseline policy kinds, delegate validation to registry.
    spec = make_policy_spec(
        condition_id=cid,
        policy_kind=kind,
        checkpoint_path=request.checkpoint_path,
        qwen_enabled=request.qwen_enabled,
        causal=request.causal,
        require_existing_checkpoint=request.require_existing_checkpoint,
    )

    return PolicyInferenceBoundary(
        condition_id=cid,
        policy_kind=kind,
        policy_source=spec.policy_source,
        source_mode=spec.source_mode,
        checkpoint_path=spec.checkpoint_path,
        qwen_trigger_rate=0.0,
        causal_claim_allowed=False,
        placeholder_allowed=False,
        actual_mappo_inference_ready=False,
        failure_if_used_for_claim="baseline or non-MAPPO policy is not an actual MAPPO causal claim",
        notes=spec.notes,
    )


def boundary_to_dict(boundary: PolicyInferenceBoundary) -> Dict[str, Any]:
    return asdict(boundary)


def boundary_to_rollout_provenance(boundary: PolicyInferenceBoundary) -> Dict[str, Any]:
    """
    Convert boundary info into fields that must be written into window_rollup.parquet.
    """
    return {
        "policy_source": boundary.policy_source,
        "policy_checkpoint_path": boundary.checkpoint_path,
        "source_mode": boundary.source_mode,
        "qwen_trigger_rate": boundary.qwen_trigger_rate,
    }


def validate_boundary_for_rollout_claim(boundary: PolicyInferenceBoundary) -> None:
    """
    Raise if a boundary is not eligible for paper-level causal performance claims.
    """
    if not boundary.causal_claim_allowed:
        raise ValueError(
            boundary.failure_if_used_for_claim
            or "this policy boundary is not eligible for causal performance claims"
        )

    if boundary.policy_kind != "mappo":
        raise ValueError("causal performance claim requires policy_kind=mappo")

    if boundary.placeholder_allowed:
        raise ValueError("placeholder policy cannot be used for causal performance claims")

    if not boundary.actual_mappo_inference_ready:
        raise ValueError("actual MAPPO inference is not ready")

    if boundary.qwen_trigger_rate != 0.0:
        raise ValueError("A-family causal MAPPO claim requires qwen_trigger_rate=0.0")


def make_a_family_actual_boundary(
    condition_id: str,
    checkpoint_path: str,
    require_existing_checkpoint: bool = False,
) -> Dict[str, Any]:
    request = PolicyInferenceRequest(
        condition_id=condition_id,
        policy_kind="mappo",
        checkpoint_path=checkpoint_path,
        qwen_enabled=False,
        causal=True,
        require_existing_checkpoint=require_existing_checkpoint,
    )
    boundary = build_policy_inference_boundary(request)
    return boundary_to_dict(boundary)


def make_a_family_placeholder_boundary(condition_id: str) -> Dict[str, Any]:
    request = PolicyInferenceRequest(
        condition_id=condition_id,
        policy_kind="placeholder",
        checkpoint_path=None,
        qwen_enabled=False,
        causal=False,
        require_existing_checkpoint=False,
    )
    boundary = build_policy_inference_boundary(request)
    return boundary_to_dict(boundary)


def boundary_contract() -> Dict[str, Any]:
    return {
        "artifact_version": "policy_inference_boundary_v1",
        "rules": {
            "actual_mappo_requires_checkpoint_path": True,
            "actual_mappo_missing_checkpoint_fails": True,
            "placeholder_never_used_for_causal_claim": True,
            "a_family_qwen_trigger_rate_zero": True,
            "rollout_must_record_policy_source": True,
            "rollout_must_record_policy_checkpoint_path": True,
            "rollout_must_record_source_mode": True,
            "rollout_must_record_qwen_trigger_rate": True,
        },
        "a_family_conditions": sorted(A_FAMILY_CONDITIONS),
        "required_rollout_provenance_fields": [
            "policy_source",
            "policy_checkpoint_path",
            "source_mode",
            "qwen_trigger_rate",
        ],
        "actual_mappo_source_mode_examples": [
            "causal_A_mappo_policy_v1",
            "causal_A90_mappo_policy_v1",
            "causal_A80_mappo_policy_v1",
            "causal_A70_mappo_policy_v1",
        ],
        "placeholder_source_mode_examples": [
            "stub_A_placeholder_smoke",
            "stub_A90_placeholder_smoke",
            "stub_A80_placeholder_smoke",
            "stub_A70_placeholder_smoke",
        ],
    }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.print_contract:
        print(json.dumps(boundary_contract(), ensure_ascii=False, indent=2))
        return 0

    if args.self_test:
        # Placeholder is allowed only for smoke/contract.
        placeholder = build_policy_inference_boundary(
            PolicyInferenceRequest(
                condition_id="A",
                policy_kind="placeholder",
                checkpoint_path=None,
                qwen_enabled=False,
                causal=False,
                require_existing_checkpoint=False,
            )
        )
        assert placeholder.placeholder_allowed is True
        assert placeholder.causal_claim_allowed is False

        # Actual MAPPO with a declared checkpoint path is accepted when existence check is disabled.
        actual = build_policy_inference_boundary(
            PolicyInferenceRequest(
                condition_id="A",
                policy_kind="mappo",
                checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
                qwen_enabled=False,
                causal=True,
                require_existing_checkpoint=False,
            )
        )
        assert actual.actual_mappo_inference_ready is True
        assert actual.causal_claim_allowed is True
        validate_boundary_for_rollout_claim(actual)

        # Missing checkpoint path must fail.
        try:
            build_policy_inference_boundary(
                PolicyInferenceRequest(
                    condition_id="A",
                    policy_kind="mappo",
                    checkpoint_path="",
                    qwen_enabled=False,
                    causal=True,
                    require_existing_checkpoint=False,
                )
            )
        except FileNotFoundError:
            print("[OK] missing checkpoint_path correctly rejected")
        else:
            print("[FAIL] missing checkpoint_path was not rejected")
            return 1

        # Placeholder cannot be used as causal.
        try:
            build_policy_inference_boundary(
                PolicyInferenceRequest(
                    condition_id="A",
                    policy_kind="placeholder",
                    checkpoint_path=None,
                    qwen_enabled=False,
                    causal=True,
                    require_existing_checkpoint=False,
                )
            )
        except ValueError:
            print("[OK] causal placeholder correctly rejected")
        else:
            print("[FAIL] causal placeholder was not rejected")
            return 1

        # Qwen must remain disabled for A.
        try:
            build_policy_inference_boundary(
                PolicyInferenceRequest(
                    condition_id="A",
                    policy_kind="mappo",
                    checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
                    qwen_enabled=True,
                    causal=True,
                    require_existing_checkpoint=False,
                )
            )
        except ValueError:
            print("[OK] A-family Qwen intervention correctly rejected")
        else:
            print("[FAIL] A-family Qwen intervention was not rejected")
            return 1

        print("[OK] policy_inference_boundary_v1 self-test passed")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
