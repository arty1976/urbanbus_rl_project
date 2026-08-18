#!/usr/bin/env python3
"""A/B1/B2 causal comparison arm contracts (H4M-AE-R4).

Every arm shares one causal universe: the same PV8 simulator, demand
realization, initial state, horizon, transition engine, passenger accounting,
p95 ledger semantics and canonical aggregation path.  Only action generation
differs.

Construction and integrity contracts only: this module never ranks arms and
never computes a performance comparison.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

ARM_A = "A"
ARM_B1 = "B1"
ARM_B2 = "B2"

POLICY_SOURCE = {
    ARM_A: "actual_promoted_mappo",
    ARM_B1: "causal_noop_baseline",
    ARM_B2: "causal_rulebased_baseline",
}

# The environment side of every arm is identical by contract; only these keys
# may differ between arms.
ARM_VARIABLE_KEYS = ("arm_id", "policy_source", "policy_contract", "actual_checkpoint_loaded")

HISTORICAL_SOURCE_MODES = ("historical", "legacy", "stub", "replay", "smoke", "noncausal", "non_causal")


class ArmContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def demand_realization_payload(adapter: Any) -> Dict[str, Any]:
    """Exogenous demand realization, captured before any action is taken."""
    stops = adapter.state["stops"]
    return {
        "window_id": str(adapter.window.get("window_id")),
        "seed": int(adapter.seed),
        "horizon_seconds": float(adapter.horizon_seconds),
        "stop_count": int(adapter.stop_count),
        "passenger_count": int(adapter.accounting.passenger_demand_generated),
        "arrival_schedule": {int(s.stop_id): [round(v, 9) for v in s.arrival_schedule] for s in stops.values()},
        "carry_in_queue": {int(s.stop_id): [round(v, 9) for v in s.queue_arrival_ts] for s in stops.values()},
    }


def demand_realization_hash(adapter: Any) -> str:
    return _digest(demand_realization_payload(adapter))


def initial_state_payload(adapter: Any) -> Dict[str, Any]:
    """Canonical pre-action state.  Contains no policy-specific data."""
    stops = adapter.state["stops"]
    vehicles = adapter.state["vehicles"]
    return {
        "demand_realization": demand_realization_payload(adapter),
        "vehicles": sorted(
            (
                {
                    "agent_id": int(v.agent_id),
                    "stop_index": int(v.stop_index),
                    "clock_seconds": round(float(getattr(v, "clock_seconds", 0.0)), 9),
                    "hold_seconds": round(float(v.hold_seconds), 9),
                    "distance_m": round(float(getattr(v, "distance_m", 0.0)), 9),
                }
                for v in vehicles.values()
            ),
            key=lambda r: r["agent_id"],
        ),
        "waiting_counts": {int(s.stop_id): int(s.waiting_count) for s in stops.values()},
        "next_arrival_index": {int(s.stop_id): int(s.next_arrival_index) for s in stops.values()},
        "accounting": {
            "completed_wait_ledger": list(adapter.accounting.completed_wait_ledger),
            "censored_wait_ledger": list(adapter.accounting.censored_wait_ledger),
            "boarding_ledger": list(adapter.accounting.boarding_ledger),
            "passenger_served_count": int(adapter.accounting.passenger_served_count),
        },
        "num_agents": int(adapter.num_agents),
        "source_mode": adapter.source_mode,
    }


def initial_state_hash(adapter: Any) -> str:
    return _digest(initial_state_payload(adapter))


def _legal(mask: Sequence[bool], preferred: int, fallback: int = 0) -> int:
    if preferred < len(mask) and bool(mask[preferred]):
        return preferred
    if fallback < len(mask) and bool(mask[fallback]):
        return fallback
    for index, allowed in enumerate(mask):
        if allowed:
            return index
    raise ArmContractError("NO_LEGAL_ACTION", "the K-mask left no admissible action")


def noop_actions(adapter: Any, legal_mask: Mapping[int, Sequence[bool]]) -> Dict[int, int]:
    """B1: never intervene.  HOLD wherever HOLD is legal."""
    return {agent: _legal(mask, preferred=0) for agent, mask in legal_mask.items()}


def rulebased_actions(adapter: Any, legal_mask: Mapping[int, Sequence[bool]]) -> Dict[int, int]:
    """B2: deterministic rule -- SERVE a stop with waiting passengers, else HOLD.

    Reads only simulator state that every arm shares; no policy network, no
    randomness, no KPI feedback.
    """
    stops = adapter.state["stops"]
    actions: Dict[int, int] = {}
    for agent, mask in legal_mask.items():
        vehicle = adapter.state["vehicles"][agent]
        stop = stops[vehicle.stop_index % adapter.stop_count]
        actions[agent] = _legal(mask, preferred=1 if stop.waiting_count > 0 else 0)
    return actions


@dataclass(frozen=True)
class ArmContract:
    arm_id: str
    policy_source: str
    policy_contract: str
    actual_checkpoint_loaded: bool
    action_fn: Callable[..., Dict[int, int]]

    def payload(self) -> Dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "policy_source": self.policy_source,
            "policy_contract": self.policy_contract,
            "actual_checkpoint_loaded": self.actual_checkpoint_loaded,
            "placeholder_fallback_used": False,
            "mock_action_used": False,
            "random_fallback_used": False,
            "historical_replay_source": False,
        }


def build_arm_contracts(promoted_action_fn: Callable[..., Dict[int, int]], checkpoint_path: str) -> Dict[str, ArmContract]:
    return {
        ARM_A: ArmContract(ARM_A, POLICY_SOURCE[ARM_A], f"promoted checkpoint {checkpoint_path}", True, promoted_action_fn),
        ARM_B1: ArmContract(ARM_B1, POLICY_SOURCE[ARM_B1], "HOLD wherever legal; never intervenes", False, noop_actions),
        ARM_B2: ArmContract(ARM_B2, POLICY_SOURCE[ARM_B2], "SERVE when the current stop has waiting passengers, else HOLD", False, rulebased_actions),
    }


def environment_contract(adapter: Any, module_shas: Mapping[str, str]) -> Dict[str, Any]:
    """The half of the universe that must be byte-identical across arms."""
    if any(token in str(adapter.source_mode).lower() for token in HISTORICAL_SOURCE_MODES):
        raise ArmContractError("HISTORICAL_SOURCE_MODE_IN_CAUSAL_ARM", str(adapter.source_mode))
    return {
        "window_id": str(adapter.window.get("window_id")),
        "seed": int(adapter.seed),
        "horizon_seconds": float(adapter.horizon_seconds),
        "num_agents": int(adapter.num_agents),
        "stop_count": int(adapter.stop_count),
        "source_mode": adapter.source_mode,
        "demand_realization_hash": demand_realization_hash(adapter),
        "initial_state_hash": initial_state_hash(adapter),
        **{f"module_sha256::{k}": v for k, v in sorted(module_shas.items())},
    }
