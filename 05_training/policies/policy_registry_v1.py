"""
policy_registry_v1.py
=====================

Policy registry contract for UrbanBus MAPPO rollout.

MAPPO means Multi-Agent Proximal Policy Optimization.
Qwen must remain disabled for A/A90/A80/A70 in the current phase.

Purpose
-------
This registry separates placeholder policies from actual MAPPO inference.

Important rule:
Actual MAPPO policies must require a checkpoint path.
If checkpoint_path is missing, the registry must fail loudly.
It must never silently fall back to a placeholder policy.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


VALID_POLICY_KINDS = {
    "noop",
    "rulebased",
    "placeholder",
    "mappo",
}


A_FAMILY_CONDITIONS = {
    "A",
    "A90",
    "A80",
    "A70",
}


BASELINE_CONDITIONS = {
    "B0",
    "B0R",
    "B1",
    "B2",
}


@dataclass(frozen=True)
class PolicySpec:
    condition_id: str
    policy_kind: str
    policy_source: str
    source_mode: str
    qwen_enabled: bool
    checkpoint_required: bool
    checkpoint_path: Optional[str]
    causal_required_for_claims: bool
    notes: str


def normalize_condition_id(condition_id: Any) -> str:
    return str(condition_id).strip().upper()


def normalize_policy_kind(policy_kind: Any) -> str:
    return str(policy_kind).strip().lower()


def is_a_family_condition(condition_id: Any) -> bool:
    return normalize_condition_id(condition_id) in A_FAMILY_CONDITIONS


def is_baseline_condition(condition_id: Any) -> bool:
    return normalize_condition_id(condition_id) in BASELINE_CONDITIONS


def build_source_mode(
    condition_id: str,
    policy_kind: str,
    causal: bool,
    version: str = "v1",
) -> str:
    cid = normalize_condition_id(condition_id)
    kind = normalize_policy_kind(policy_kind)

    if causal:
        return f"causal_{cid}_{kind}_policy_{version}"

    if kind == "placeholder":
        return f"stub_{cid}_{kind}_smoke"

    if kind == "noop":
        return f"replay_{cid}_{kind}_noncausal"

    if kind == "rulebased":
        return f"replay_{cid}_{kind}_noncausal"

    return f"noncausal_{cid}_{kind}_{version}"


def validate_policy_request(
    condition_id: Any,
    policy_kind: Any,
    checkpoint_path: Optional[str] = None,
    qwen_enabled: bool = False,
    causal: bool = False,
    require_existing_checkpoint: bool = False,
) -> None:
    cid = normalize_condition_id(condition_id)
    kind = normalize_policy_kind(policy_kind)

    if kind not in VALID_POLICY_KINDS:
        raise ValueError(f"unsupported policy_kind: {policy_kind}")

    if is_a_family_condition(cid) and qwen_enabled:
        raise ValueError("A/A90/A80/A70 must have qwen_enabled=False in the current phase")

    if cid == "A" and kind == "placeholder" and causal:
        raise ValueError("A causal rollout cannot use placeholder policy")

    if cid in {"A", "A90", "A80", "A70"} and kind == "mappo":
        if not checkpoint_path:
            raise FileNotFoundError(
                "Actual MAPPO policy requires checkpoint_path. "
                "Refusing to fall back to placeholder policy."
            )

        if require_existing_checkpoint and not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"MAPPO checkpoint does not exist: {checkpoint_path}"
            )

    if kind == "mappo" and not causal:
        raise ValueError("Actual MAPPO policy should be used with causal=True")

    if kind == "placeholder" and checkpoint_path:
        raise ValueError("Placeholder policy must not carry a MAPPO checkpoint_path")


def make_policy_spec(
    condition_id: Any,
    policy_kind: Any,
    checkpoint_path: Optional[str] = None,
    qwen_enabled: bool = False,
    causal: bool = False,
    version: str = "v1",
    require_existing_checkpoint: bool = False,
) -> PolicySpec:
    cid = normalize_condition_id(condition_id)
    kind = normalize_policy_kind(policy_kind)

    validate_policy_request(
        condition_id=cid,
        policy_kind=kind,
        checkpoint_path=checkpoint_path,
        qwen_enabled=qwen_enabled,
        causal=causal,
        require_existing_checkpoint=require_existing_checkpoint,
    )

    source_mode = build_source_mode(
        condition_id=cid,
        policy_kind=kind,
        causal=causal,
        version=version,
    )

    if kind == "mappo":
        policy_source = "mappo_policy"
        checkpoint_required = True
        notes = "actual MAPPO inference; checkpoint required; no placeholder fallback"
    elif kind == "placeholder":
        policy_source = "placeholder_policy"
        checkpoint_required = False
        notes = "contract/smoke only; not valid for paper-level performance claims"
    elif kind == "rulebased":
        policy_source = "rulebased_policy"
        checkpoint_required = False
        notes = "rule-based baseline policy"
    elif kind == "noop":
        policy_source = "noop_policy"
        checkpoint_required = False
        notes = "no-op baseline policy"
    else:
        raise ValueError(f"unsupported policy_kind: {kind}")

    return PolicySpec(
        condition_id=cid,
        policy_kind=kind,
        policy_source=policy_source,
        source_mode=source_mode,
        qwen_enabled=bool(qwen_enabled),
        checkpoint_required=bool(checkpoint_required),
        checkpoint_path=checkpoint_path,
        causal_required_for_claims=True,
        notes=notes,
    )


def spec_to_dict(spec: PolicySpec) -> Dict[str, Any]:
    return asdict(spec)


def default_placeholder_spec(condition_id: Any) -> Dict[str, Any]:
    spec = make_policy_spec(
        condition_id=condition_id,
        policy_kind="placeholder",
        checkpoint_path=None,
        qwen_enabled=False,
        causal=False,
        version="v1",
    )
    return spec_to_dict(spec)


def default_actual_mappo_spec(
    condition_id: Any,
    checkpoint_path: str,
    require_existing_checkpoint: bool = False,
) -> Dict[str, Any]:
    spec = make_policy_spec(
        condition_id=condition_id,
        policy_kind="mappo",
        checkpoint_path=checkpoint_path,
        qwen_enabled=False,
        causal=True,
        version="v1",
        require_existing_checkpoint=require_existing_checkpoint,
    )
    return spec_to_dict(spec)


def default_noop_spec() -> Dict[str, Any]:
    spec = make_policy_spec(
        condition_id="B1",
        policy_kind="noop",
        checkpoint_path=None,
        qwen_enabled=False,
        causal=False,
        version="v1",
    )
    return spec_to_dict(spec)


def default_rulebased_spec() -> Dict[str, Any]:
    spec = make_policy_spec(
        condition_id="B2",
        policy_kind="rulebased",
        checkpoint_path=None,
        qwen_enabled=False,
        causal=False,
        version="v1",
    )
    return spec_to_dict(spec)


def registry_contract() -> Dict[str, Any]:
    return {
        "artifact_version": "policy_registry_v1",
        "valid_policy_kinds": sorted(VALID_POLICY_KINDS),
        "a_family_conditions": sorted(A_FAMILY_CONDITIONS),
        "baseline_conditions": sorted(BASELINE_CONDITIONS),
        "rules": {
            "actual_mappo_requires_checkpoint": True,
            "no_placeholder_fallback_for_mappo": True,
            "a_family_qwen_disabled": True,
            "actual_mappo_requires_causal_source_mode": True,
            "placeholder_is_smoke_only": True,
        },
        "examples": {
            "A_placeholder": default_placeholder_spec("A"),
            "A_actual_mappo": default_actual_mappo_spec(
                condition_id="A",
                checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
                require_existing_checkpoint=False,
            ),
            "B1_noop": default_noop_spec(),
            "B2_rulebased": default_rulebased_spec(),
        },
    }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.print_contract:
        print(json.dumps(registry_contract(), ensure_ascii=False, indent=2))
        return 0

    if args.self_test:
        _ = default_placeholder_spec("A")
        _ = default_actual_mappo_spec(
            "A",
            "artifacts/experiment_A_v1/checkpoints/best.pt",
            require_existing_checkpoint=False,
        )
        _ = default_noop_spec()
        _ = default_rulebased_spec()

        try:
            default_actual_mappo_spec("A", "", require_existing_checkpoint=False)
        except FileNotFoundError:
            print("[OK] missing MAPPO checkpoint correctly rejected")
        else:
            print("[FAIL] missing MAPPO checkpoint was not rejected")
            return 1

        try:
            make_policy_spec(
                condition_id="A",
                policy_kind="placeholder",
                checkpoint_path=None,
                qwen_enabled=False,
                causal=True,
            )
        except ValueError:
            print("[OK] causal placeholder correctly rejected")
        else:
            print("[FAIL] causal placeholder was not rejected")
            return 1

        try:
            make_policy_spec(
                condition_id="A",
                policy_kind="mappo",
                checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
                qwen_enabled=True,
                causal=True,
            )
        except ValueError:
            print("[OK] A-family Qwen intervention correctly rejected")
        else:
            print("[FAIL] A-family Qwen intervention was not rejected")
            return 1

        print("[OK] policy_registry_v1 self-test passed")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
