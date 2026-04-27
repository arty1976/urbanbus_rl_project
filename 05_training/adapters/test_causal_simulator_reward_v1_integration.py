from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "05_training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


from adapters.causal_simulator_adapter import (  # noqa: E402
    ACTION_DISPATCH,
    ACTION_HOLD,
    ACTION_SKIP,
    CausalSimulatorAdapter,
)


PHASE2_12_KPIS = {
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
}

REQUIRED_REWARD_INFO_KEYS = {
    "reward_version",
    "reward_claim_boundary",
    "reward_total",
    "reward_components",
    "reward_debug",
    "reward_metrics",
}

EXPECTED_REWARD_VERSION = "mappo_reward_v1"
EXPECTED_CLAIM_BOUNDARY = "toy_causal_training_reward_contract_not_paper_performance_claim"


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_adapter(seed: int = 87, condition_id: str = "A") -> CausalSimulatorAdapter:
    return CausalSimulatorAdapter(
        {
            "condition_id": condition_id,
            "seed": seed,
            "num_agents": 8,
            "max_steps": 2,
            "control_step_minutes": 30,
            "evaluation_horizon_minutes": 30,
        }
    )


def reset_adapter(adapter: CausalSimulatorAdapter, seed: int = 87):
    return adapter.reset(
        seed=seed,
        scenario_config={
            "window_id": "step87_reward_v1_integration",
            "state_ts": "2024-01-01T08:00:00Z",
            "time_band": "peak",
        },
    )


def dispatch_dominant_actions(num_agents: int = 8) -> Dict[int, int]:
    actions = {agent_id: ACTION_DISPATCH for agent_id in range(num_agents)}
    # Keep one mild intervention so reward/debug sees a realistic mixed action pattern.
    actions[num_agents - 1] = ACTION_SKIP
    return actions


def all_hold_actions(num_agents: int = 8) -> Dict[int, int]:
    return {agent_id: ACTION_HOLD for agent_id in range(num_agents)}


def run_one_step(actions: Dict[int, int], seed: int = 87, condition_id: str = "A"):
    adapter = make_adapter(seed=seed, condition_id=condition_id)
    reset_adapter(adapter, seed=seed)
    result = adapter.step(actions)
    return adapter, result


def assert_reward_info_contract(result) -> None:
    info = result.info

    missing = REQUIRED_REWARD_INFO_KEYS - set(info.keys())
    assert_true(not missing, f"StepResult.info missing reward keys: {sorted(missing)}")

    assert_true(info["reward_version"] == EXPECTED_REWARD_VERSION, "reward_version mismatch")
    assert_true(
        info["reward_claim_boundary"] == EXPECTED_CLAIM_BOUNDARY,
        "reward_claim_boundary mismatch",
    )

    reward_total = float(info["reward_total"])
    assert_true(math.isfinite(reward_total), "reward_total must be finite")

    assert_true(isinstance(info["reward_components"], dict), "reward_components must be dict")
    assert_true(isinstance(info["reward_debug"], dict), "reward_debug must be dict")
    assert_true(isinstance(info["reward_metrics"], dict), "reward_metrics must be dict")

    metrics = info["reward_metrics"]
    missing_kpis = PHASE2_12_KPIS - set(metrics.keys())
    assert_true(not missing_kpis, f"reward_metrics missing 12-KPI fields: {sorted(missing_kpis)}")

    for kpi in PHASE2_12_KPIS:
        value = metrics[kpi]
        assert_true(value is not None, f"reward_metrics[{kpi}] is None")
        try:
            numeric = float(value)
        except Exception as exc:
            raise AssertionError(f"reward_metrics[{kpi}] is not numeric: {value!r}") from exc
        assert_true(math.isfinite(numeric), f"reward_metrics[{kpi}] must be finite")

    for bounded in [
        "bunching_rate",
        "on_time_rate",
        "intervention_rate",
        "passenger_service_rate",
        "fleet_reduction_ratio",
    ]:
        value = float(metrics[bounded])
        assert_true(0.0 <= value <= 1.0, f"{bounded} must be in [0, 1], got {value}")

    assert_true(info.get("causal_comparison_allowed") is True, "causal_comparison_allowed must stay true")


def assert_team_reward_broadcast(result) -> None:
    reward_total = float(result.info["reward_total"])
    assert_true(len(result.rewards) == 8, "expected 8 agent rewards")
    for agent_id, value in result.rewards.items():
        assert_true(
            abs(float(value) - reward_total) < 1e-9,
            f"agent {agent_id} reward must equal reward_total",
        )


def test_reward_info_contract_and_broadcast() -> None:
    _, result = run_one_step(dispatch_dominant_actions())
    assert_reward_info_contract(result)
    assert_team_reward_broadcast(result)


def test_reward_is_deterministic_for_same_seed_and_actions() -> None:
    _, result_a = run_one_step(dispatch_dominant_actions(), seed=87)
    _, result_b = run_one_step(dispatch_dominant_actions(), seed=87)

    assert_true(
        abs(float(result_a.info["reward_total"]) - float(result_b.info["reward_total"])) < 1e-9,
        "reward_total must be deterministic for same seed and same actions",
    )

    for kpi in PHASE2_12_KPIS:
        va = float(result_a.info["reward_metrics"][kpi])
        vb = float(result_b.info["reward_metrics"][kpi])
        assert_true(abs(va - vb) < 1e-9, f"reward_metrics[{kpi}] must be deterministic")


def test_all_hold_is_worse_than_dispatch_dominant() -> None:
    _, dispatch_result = run_one_step(dispatch_dominant_actions(), seed=88)
    _, hold_result = run_one_step(all_hold_actions(), seed=88)

    dispatch_reward = float(dispatch_result.info["reward_total"])
    hold_reward = float(hold_result.info["reward_total"])

    assert_true(
        hold_reward < dispatch_reward,
        f"all-hold reward should be lower than dispatch-dominant reward; "
        f"hold={hold_reward}, dispatch={dispatch_reward}",
    )

    hold_metrics = hold_result.info["reward_metrics"]
    dispatch_metrics = dispatch_result.info["reward_metrics"]

    assert_true(
        float(hold_metrics["passenger_service_rate"]) <= float(dispatch_metrics["passenger_service_rate"]),
        "all-hold should not have better passenger_service_rate than dispatch-dominant",
    )


def test_claim_boundary_is_non_paper_claim() -> None:
    _, result = run_one_step(dispatch_dominant_actions())
    assert_true(result.info["causal_comparison_allowed"] is True, "causal flag should be true")
    assert_true(
        result.info["reward_claim_boundary"] == EXPECTED_CLAIM_BOUNDARY,
        "reward claim boundary must explicitly prohibit paper-level performance claim",
    )
    assert_true(
        "not_paper_performance_claim" in result.info["reward_claim_boundary"],
        "claim boundary must include not_paper_performance_claim",
    )


def main() -> None:
    tests = [
        test_reward_info_contract_and_broadcast,
        test_reward_is_deterministic_for_same_seed_and_actions,
        test_all_hold_is_worse_than_dispatch_dominant,
        test_claim_boundary_is_non_paper_claim,
    ]

    for test in tests:
        test()
        print(f"[OK] {test.__name__}")

    print("[OK] Step 87 causal simulator reward v1 integration self-test PASS")


if __name__ == "__main__":
    main()
