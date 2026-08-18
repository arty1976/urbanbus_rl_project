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
    ARM_B1: "causal_baseline_normal_service_no_policy_intervention",
    ARM_B2: "causal_rulebased_baseline",
}

# H4M-AE-R6.1: canonical B1 means "no discretionary policy intervention", not
# "hold forever".  The frozen representative B1 regeneration executes SERVE at
# every occurrence with hold_seconds = 0.0, so normal base service is part of
# the baseline, and only discretionary HOLD/SKIP counts as intervention.
B1_SEMANTICS_ID = "CAUSAL_BASELINE_NORMAL_SERVICE_NO_POLICY_INTERVENTION"
B1_AUTHORITY = (
    "prompt5_e01_dl6d_pa1a_srp2_bis_pv8_r2ar8er3r_representative_b1_regeneration_20260809_200442: "
    "r8er3r_b1_kpi_summary.json hold_seconds=0.0, 414 generated / 414 served; "
    "regeneration source emits executed_action='SERVE' at every occurrence"
)
BASE_SERVICE_ACTION = 1  # SERVE: normal boarding and route progression
DISCRETIONARY_ACTIONS = {0: "HOLD", 2: "CONDITIONAL_SKIP"}

PERSISTENT_HOLD_CONTROL_ID = "PERSISTENT_HOLD_CONTROL"
PERSISTENT_HOLD_CONTROL_STATUS = {
    "control_id": PERSISTENT_HOLD_CONTROL_ID,
    "canonical_baseline": False,
    "performance_reference_allowed": False,
    "B1_alias_allowed": False,
    "role": "negative control for tail/censoring detectors only",
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
        "evaluation_start_ts": float(adapter.evaluation_start_ts),
        "evaluation_end_ts": float(adapter.evaluation_end_ts),
        "evaluation_horizon_seconds": float(adapter.horizon_seconds),
        "stop_count": int(adapter.stop_count),
        "demand_artifact_sha256": adapter.demand_provenance.get("artifact_sha256"),
        "passenger_count": int(adapter.accounting.passenger_demand_generated),
        "request_ids": {int(s.stop_id): list(s.request_ids) for s in stops.values()},
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
            "passenger_eventual_served_count": int(adapter.accounting.passenger_eventual_served_count),
            "passenger_served_by_horizon_count": int(adapter.accounting.passenger_served_by_horizon_count),
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


def base_service_actions(adapter: Any, legal_mask: Mapping[int, Sequence[bool]]) -> Dict[int, int]:
    """Canonical B1: normal baseline service, zero discretionary intervention.

    The vehicle progresses along its route and serves every occurrence, exactly
    as the frozen representative B1 regeneration does.  No HOLD and no
    CONDITIONAL_SKIP is ever chosen by policy; a non-service action can only
    appear if the K-mask makes SERVE illegal, which is a legality constraint
    rather than a discretionary decision.
    """
    return {agent: _legal(mask, preferred=BASE_SERVICE_ACTION) for agent, mask in legal_mask.items()}


def persistent_hold_control_actions(adapter: Any, legal_mask: Mapping[int, Sequence[bool]]) -> Dict[int, int]:
    """Non-canonical negative control: hold forever.  Never a B1 alias."""
    return {agent: _legal(mask, preferred=0) for agent, mask in legal_mask.items()}


# Retired alias kept only so no caller silently resolves the old meaning.
def noop_actions(adapter: Any, legal_mask: Mapping[int, Sequence[bool]]) -> Dict[int, int]:
    raise ArmContractError(
        "RETIRED_B1_NOOP_SEMANTICS",
        "no-op no longer means persistent HOLD; use base_service_actions for canonical B1 "
        "or persistent_hold_control_actions for the non-canonical control",
    )


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
            "learned_policy_used": self.arm_id == ARM_A,
            "discretionary_policy_intervention": self.arm_id != ARM_B1,
            "normal_vehicle_progression": True,
            "required_boarding_alighting_enabled": True,
        }


def build_arm_contracts(promoted_action_fn: Callable[..., Dict[int, int]], checkpoint_path: str) -> Dict[str, ArmContract]:
    return {
        ARM_A: ArmContract(ARM_A, POLICY_SOURCE[ARM_A], f"promoted checkpoint {checkpoint_path}", True, promoted_action_fn),
        ARM_B1: ArmContract(
            ARM_B1, POLICY_SOURCE[ARM_B1],
            "normal baseline service: SERVE every occurrence, zero discretionary HOLD/SKIP",
            False, base_service_actions,
        ),
        ARM_B2: ArmContract(ARM_B2, POLICY_SOURCE[ARM_B2], "SERVE when the current stop has waiting passengers, else HOLD", False, rulebased_actions),
    }


def environment_contract(adapter: Any, module_shas: Mapping[str, str]) -> Dict[str, Any]:
    """The half of the universe that must be byte-identical across arms."""
    if any(token in str(adapter.source_mode).lower() for token in HISTORICAL_SOURCE_MODES):
        raise ArmContractError("HISTORICAL_SOURCE_MODE_IN_CAUSAL_ARM", str(adapter.source_mode))
    return {
        "window_id": str(adapter.window.get("window_id")),
        "seed": int(adapter.seed),
        "evaluation_start_ts": float(adapter.evaluation_start_ts),
        "evaluation_end_ts": float(adapter.evaluation_end_ts),
        "evaluation_horizon_seconds": float(adapter.horizon_seconds),
        "num_agents": int(adapter.num_agents),
        "stop_count": int(adapter.stop_count),
        "demand_artifact_sha256": adapter.demand_provenance.get("artifact_sha256"),
        "demand_population_count": int(len(adapter.demand_population)),
        "source_mode": adapter.source_mode,
        "demand_realization_hash": demand_realization_hash(adapter),
        "initial_state_hash": initial_state_hash(adapter),
        **{f"module_sha256::{k}": v for k, v in sorted(module_shas.items())},
    }
