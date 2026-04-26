"""
a_family_rollout_bridge_v1.py
=============================

Bridge module for A/A90/A80/A70 rollout rows.

This module connects:
- Step 21 reward contract
- Step 22 rollout schema contract
- Step 23 policy registry contract
- Step 24 policy inference boundary

MAPPO means Multi-Agent Proximal Policy Optimization.
KPI means Key Performance Indicator.
Qwen must remain disabled for A/A90/A80/A70 in the current phase.

This module does not run the simulator.
It prepares and validates one extended window_rollup row.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


_THIS = Path(__file__).resolve()
_TRAINING_DIR = _THIS.parents[1]
_POLICIES_DIR = _TRAINING_DIR / "policies"
_REWARDS_DIR = _TRAINING_DIR / "rewards"
_ROLLOUTS_DIR = _TRAINING_DIR / "rollouts"

for _p in (_POLICIES_DIR, _REWARDS_DIR, _ROLLOUTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


from policy_inference_boundary_v1 import (  # noqa: E402
    PolicyInferenceRequest,
    boundary_to_rollout_provenance,
    build_policy_inference_boundary,
    validate_boundary_for_rollout_claim,
)
from rollout_schema_v1 import (  # noqa: E402
    augment_rollout_row,
    validate_rollout_rows,
)
from mappo_reward_v1 import compute_total_reward  # noqa: E402


A_FAMILY_CONDITIONS = {"A", "A90", "A80", "A70"}


def normalize_condition_id(condition_id: Any) -> str:
    return str(condition_id).strip().upper()


def _float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _default_active_bus_ratio(condition_id: str) -> float:
    cid = normalize_condition_id(condition_id)
    if cid == "A":
        return 1.0
    if cid == "A90":
        return 0.9
    if cid == "A80":
        return 0.8
    if cid == "A70":
        return 0.7
    raise ValueError(f"unsupported A-family condition_id: {condition_id}")


def build_a_family_rollout_row(
    base_row: Mapping[str, Any],
    *,
    condition_id: str,
    policy_kind: str,
    checkpoint_path: Optional[str] = None,
    require_existing_checkpoint: bool = False,
    baseline_bus_count: Optional[float] = None,
    active_bus_count: Optional[float] = None,
    reward_config: Optional[Mapping[str, Any]] = None,
    strict_claim_validation: bool = False,
) -> Dict[str, Any]:
    """
    Build one extended rollout row for A/A90/A80/A70.

    policy_kind:
    - "placeholder": smoke/contract only
    - "mappo": actual MAPPO boundary; checkpoint_path required
    """
    cid = normalize_condition_id(condition_id)

    if cid not in A_FAMILY_CONDITIONS:
        raise ValueError(f"condition_id must be one of {sorted(A_FAMILY_CONDITIONS)}, got {condition_id}")

    kind = str(policy_kind).strip().lower()
    causal = kind == "mappo"

    request = PolicyInferenceRequest(
        condition_id=cid,
        policy_kind=kind,
        checkpoint_path=checkpoint_path,
        qwen_enabled=False,
        causal=causal,
        require_existing_checkpoint=require_existing_checkpoint,
    )
    boundary = build_policy_inference_boundary(request)

    if strict_claim_validation:
        validate_boundary_for_rollout_claim(boundary)

    row = dict(base_row)
    row["condition_id"] = cid
    row.update(boundary_to_rollout_provenance(boundary))

    if baseline_bus_count is None:
        baseline_bus_count = _float(row.get("baseline_bus_count"), default=None)
    if baseline_bus_count is None:
        baseline_bus_count = _float(row.get("active_bus_count"), default=10.0)

    if active_bus_count is None:
        active_bus_count = round(float(baseline_bus_count) * _default_active_bus_ratio(cid), 6)

    row["baseline_bus_count"] = float(baseline_bus_count)
    row["active_bus_count"] = float(active_bus_count)

    if row["baseline_bus_count"] <= 0:
        raise ValueError("baseline_bus_count must be positive")

    fleet_reduction_ratio = (row["baseline_bus_count"] - row["active_bus_count"]) / row["baseline_bus_count"]
    fleet_reduction_ratio = max(0.0, min(1.0, fleet_reduction_ratio))
    row["fleet_reduction_ratio"] = float(fleet_reduction_ratio)

    if row.get("energy_proxy") is None and row.get("energy_proxy_total") is not None:
        row["energy_proxy"] = _float(row.get("energy_proxy_total"), default=None)

    row = augment_rollout_row(row)

    reward = compute_total_reward(row, config=reward_config)
    row.update(
        {
            "reward_total": reward["reward_total"],
            "reward_service": reward["reward_service"],
            "reward_avg_wait": reward["reward_avg_wait"],
            "reward_long_wait": reward["reward_long_wait"],
            "reward_on_time": reward["reward_on_time"],
            "reward_bunching": reward["reward_bunching"],
            "reward_energy": reward["reward_energy"],
            "reward_fleet": reward["reward_fleet"],
            "reward_constraint": reward["reward_constraint"],
            "constraint_violation_flags": reward["constraint_violation_flags"],
        }
    )

    validation = validate_rollout_rows([row], strict=True)

    if not validation["passed"]:
        raise ValueError(f"rollout row validation failed: {validation}")

    return row


def sample_base_row() -> Dict[str, Any]:
    return {
        "seed": 1,
        "window_id": "sample_window_001",
        "state_ts": "2023-01-01T08:00:00+09:00",
        "service_date": "2023-01-01",
        "time_band": "peak",
        "evaluation_horizon_minutes": 30,
        "headway_mean_seconds": 600.0,
        "headway_std_seconds": 90.0,
        "headway_sample_count": 6,
        "bunching_event_count": 0,
        "headway_event_count": 5,
        "wait_total_passenger_seconds": 12000.0,
        "wait_passenger_count": 40,
        "ontime_event_count": 4,
        "schedulable_arrival_count": 5,
        "intervention_count": 1,
        "decision_step_count": 5,
        "energy_proxy_total": 360.0,
        "passenger_demand_generated": 40,
        "passenger_served_count": 39,
        "passenger_service_rate": None,
        "passenger_wait_p95_seconds": 540.0,
        "long_wait_passenger_count": 2,
        "energy_proxy": None,
        "energy_proxy_per_passenger": None,
        "active_bus_count": 10,
        "baseline_bus_count": 10,
        "fleet_reduction_ratio": None,
        "policy_source": None,
        "policy_checkpoint_path": None,
        "source_mode": None,
        "qwen_trigger_rate": 0.0,
        "effective_replay_step_minutes": 5.0,
        "reward_total": 0.0,
        "reward_service": 0.0,
        "reward_avg_wait": 0.0,
        "reward_long_wait": 0.0,
        "reward_on_time": 0.0,
        "reward_bunching": 0.0,
        "reward_energy": 0.0,
        "reward_fleet": 0.0,
        "reward_constraint": 0.0,
    }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--print-sample", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        base = sample_base_row()

        placeholder = build_a_family_rollout_row(
            base,
            condition_id="A",
            policy_kind="placeholder",
            checkpoint_path=None,
            strict_claim_validation=False,
        )
        assert placeholder["source_mode"] == "stub_A_placeholder_smoke"
        assert placeholder["policy_source"] == "placeholder_policy"

        actual = build_a_family_rollout_row(
            base,
            condition_id="A80",
            policy_kind="mappo",
            checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
            require_existing_checkpoint=False,
            strict_claim_validation=True,
        )
        assert actual["source_mode"] == "causal_A80_mappo_policy_v1"
        assert actual["policy_source"] == "mappo_policy"
        assert abs(actual["fleet_reduction_ratio"] - 0.2) < 1e-9
        assert actual["qwen_trigger_rate"] == 0.0

        try:
            build_a_family_rollout_row(
                base,
                condition_id="A",
                policy_kind="mappo",
                checkpoint_path="",
                require_existing_checkpoint=False,
            )
        except FileNotFoundError:
            print("[OK] missing MAPPO checkpoint_path correctly rejected")
        else:
            print("[FAIL] missing MAPPO checkpoint_path was not rejected")
            return 1

        print("[OK] a_family_rollout_bridge_v1 self-test passed")
        return 0

    if args.print_sample:
        row = build_a_family_rollout_row(
            sample_base_row(),
            condition_id="A90",
            policy_kind="mappo",
            checkpoint_path="artifacts/experiment_A_v1/checkpoints/best.pt",
            require_existing_checkpoint=False,
            strict_claim_validation=True,
        )
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
