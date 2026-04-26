"""
a_family_policy_interface_bridge_v1.py
======================================

Step 36 bridge:
Connect MAPPO policy interface action dict to A-family rollout row.

MAPPO means Multi-Agent Proximal Policy Optimization.
KPI means Key Performance Indicator.
Qwen remains disabled for A/A90/A80/A70 in the current phase.

Purpose
-------
Step 35 defined the actual MAPPO inference interface contract:

checkpoint metadata
-> observation dict
-> action dict

Step 36 verifies that the action dict can be applied to the existing
A-family rollout row structure.

Important
---------
This still does not run neural MAPPO inference.
It uses the Step 35 conservative mock action to validate the action contract.
No performance claim is allowed from this step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


THIS = Path(__file__).resolve()
TRAINING_DIR = THIS.parents[1]
ROLLOUTS_DIR = TRAINING_DIR / "rollouts"
POLICIES_DIR = TRAINING_DIR / "policies"
REWARDS_DIR = TRAINING_DIR / "rewards"

for p in (ROLLOUTS_DIR, POLICIES_DIR, REWARDS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


from a_family_rollout_bridge_v1 import (  # noqa: E402
    build_a_family_rollout_row,
    sample_base_row,
)
from rollout_schema_v1 import validate_rollout_rows  # noqa: E402
from mappo_policy_interface_v1 import (  # noqa: E402
    run_policy_interface_once,
    write_sample_metadata,
)


A_FAMILY = {"A", "A90", "A80", "A70"}


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_condition_id(condition_id: Any) -> str:
    return str(condition_id).strip().upper()


def derive_interface_observation_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Convert existing rollout raw fields into Step 35 observation fields.

    Required by mappo_policy_interface_v1:
    - avg_wait_seconds
    - bunching_rate
    - on_time_rate
    - energy_proxy_per_passenger
    """
    out = dict(row)

    wait_total = as_float(out.get("wait_total_passenger_seconds"), 0.0)
    wait_count = max(1.0, as_float(out.get("wait_passenger_count"), 1.0))
    avg_wait = as_float(out.get("avg_wait_seconds"), wait_total / wait_count)

    bunching_events = as_float(out.get("bunching_event_count"), 0.0)
    headway_events = max(1.0, as_float(out.get("headway_event_count"), 1.0))
    bunching_rate = as_float(out.get("bunching_rate"), bunching_events / headway_events)

    ontime_events = as_float(out.get("ontime_event_count"), 0.0)
    schedulable = max(1.0, as_float(out.get("schedulable_arrival_count"), 1.0))
    on_time_rate = as_float(out.get("on_time_rate"), ontime_events / schedulable)

    energy_pp = as_float(out.get("energy_proxy_per_passenger"), 0.0)

    if energy_pp <= 0.0:
        energy_proxy = as_float(out.get("energy_proxy"), 0.0)
        if energy_proxy <= 0.0:
            energy_proxy = as_float(out.get("energy_proxy_total"), 0.0)
        served = max(1.0, as_float(out.get("passenger_served_count"), wait_count))
        energy_pp = energy_proxy / served

    out["avg_wait_seconds"] = float(avg_wait)
    out["bunching_rate"] = float(bunching_rate)
    out["on_time_rate"] = float(on_time_rate)
    out["energy_proxy_per_passenger"] = float(energy_pp)

    if "qwen_trigger_rate" not in out or out["qwen_trigger_rate"] is None:
        out["qwen_trigger_rate"] = 0.0

    return out


def apply_policy_action_to_rollout_row(
    row: Mapping[str, Any],
    action: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Apply Step 35 action dict to rollout row.

    These fields become the policy provenance and action-control columns
    that future causal simulator integration can consume.
    """
    out = dict(row)

    condition_id = normalize_condition_id(action.get("condition_id"))

    if condition_id not in A_FAMILY:
        raise ValueError(f"invalid A-family action condition_id: {condition_id}")

    out["condition_id"] = condition_id
    out["policy_source"] = action["policy_source"]
    out["source_mode"] = action["source_mode"]
    out["qwen_trigger_rate"] = float(action["qwen_trigger_rate"])
    out["active_bus_count"] = float(action["active_bus_count"])

    out["action_version"] = action["action_version"]
    out["dispatch_delta"] = float(action["dispatch_delta"])
    out["hold_seconds"] = float(action["hold_seconds"])
    out["skip_stop_flag"] = bool(action["skip_stop_flag"])
    out["target_headway_ratio"] = float(action["target_headway_ratio"])
    out["policy_debug"] = action.get("policy_debug", {})

    if "placeholder" in str(out["policy_source"]).lower():
        raise ValueError("placeholder policy_source appeared in MAPPO interface row")

    source_mode = str(out["source_mode"])
    if not source_mode.startswith("causal_") or "_mappo_policy_v1" not in source_mode:
        raise ValueError(f"invalid MAPPO source_mode: {source_mode}")

    if "placeholder" in source_mode.lower() or "stub" in source_mode.lower() or "smoke" in source_mode.lower():
        raise ValueError(f"placeholder/stub/smoke marker appeared in source_mode: {source_mode}")

    if abs(float(out["qwen_trigger_rate"])) > 1e-12:
        raise ValueError("qwen_trigger_rate must remain 0.0")

    return out


def build_a_family_row_with_policy_interface(
    *,
    base_row: Mapping[str, Any],
    condition_id: str,
    checkpoint_path: str | Path,
    require_existing_checkpoint: bool = True,
) -> Dict[str, Any]:
    """
    Build A-family rollout row and then apply Step 35 MAPPO policy action.

    The initial row is produced by Step 25 bridge.
    The final policy fields are overwritten by Step 35 action output.
    """
    cid = normalize_condition_id(condition_id)

    if cid not in A_FAMILY:
        raise ValueError(f"condition_id must be one of {sorted(A_FAMILY)}, got {condition_id}")

    row = build_a_family_rollout_row(
        base_row,
        condition_id=cid,
        policy_kind="mappo",
        checkpoint_path=str(checkpoint_path),
        require_existing_checkpoint=bool(require_existing_checkpoint),
        strict_claim_validation=True,
    )

    observation_row = derive_interface_observation_fields(row)

    interface_result = run_policy_interface_once(
        checkpoint_path=checkpoint_path,
        rollout_row=observation_row,
    )

    action = interface_result["action"]

    out = apply_policy_action_to_rollout_row(row, action)

    out["policy_interface_version"] = interface_result["interface_version"]
    out["policy_action_debug"] = json.dumps(action.get("policy_debug", {}), ensure_ascii=False)
    out["performance_claim_allowed"] = False

    validation = validate_rollout_rows([out], strict=True)

    if not validation["passed"]:
        raise ValueError(f"rollout schema validation failed after policy action apply: {validation}")

    return out


def write_contract_checkpoint(path: str | Path) -> Path:
    """
    Write trained_model=True metadata for interface contract validation.

    This is still not a real neural checkpoint.
    It is a contract metadata file for Step 36 testing.
    """
    return write_sample_metadata(path, trained_model=True)


def sample_step36_row(condition_id: str = "A80") -> Dict[str, Any]:
    ckpt = write_contract_checkpoint(
        Path("artifacts/step36_policy_interface_bridge/sample_mappo_checkpoint_metadata.json")
    )

    return build_a_family_row_with_policy_interface(
        base_row=sample_base_row(),
        condition_id=condition_id,
        checkpoint_path=ckpt,
        require_existing_checkpoint=True,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--checkpoint-path",
        default="artifacts/step36_policy_interface_bridge/sample_mappo_checkpoint_metadata.json",
    )
    parser.add_argument(
        "--condition-id",
        default="A80",
        choices=["A", "A90", "A80", "A70"],
    )

    args = parser.parse_args()

    if args.self_test:
        ckpt = write_contract_checkpoint(args.checkpoint_path)
        row = build_a_family_row_with_policy_interface(
            base_row=sample_base_row(),
            condition_id=args.condition_id,
            checkpoint_path=ckpt,
            require_existing_checkpoint=True,
        )
        print(json.dumps(row, ensure_ascii=False, indent=2))
        print("[OK] a_family_policy_interface_bridge_v1 self-test passed")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
