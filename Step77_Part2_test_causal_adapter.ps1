$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project"
if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$TestPy = @'
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "05_training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))


from simulator_adapter_interface import load_adapter_class  # noqa: E402
from adapters.causal_simulator_adapter import (  # noqa: E402
    ACTION_DISPATCH,
    ACTION_HOLD,
    ACTION_SKIP,
    CausalSimulatorAdapter,
)


REQUIRED_OBS_KEYS = {
    "actor_obs",
    "critic_obs",
    "edge_index",
    "edge_attr",
    "active_bus_mask",
    "action_mask",
    "time_band",
    "state_ts",
    "agent_ids",
}

REQUIRED_RAW_EVENT_COLUMNS = {
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "time_band",
    "agent_id",
    "action",
    "action_name",
    "position_before",
    "position_after",
    "waiting_passenger_cnt",
    "passenger_served_count",
    "distance_m",
    "hold_seconds",
    "acceleration_event_count",
    "energy_proxy_total",
    "intervention_applied",
    "terminated",
    "truncated",
    "source_mode",
    "causal_comparison_allowed",
}

REQUIRED_WINDOW_ROLLUP_COLUMNS = {
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "service_date",
    "time_band",
    "evaluation_horizon_minutes",
    "qwen_trigger_rate",
    "effective_replay_step_minutes",
    "headway_mean_seconds",
    "headway_std_seconds",
    "headway_sample_count",
    "bunching_event_count",
    "headway_event_count",
    "wait_total_passenger_seconds",
    "wait_passenger_count",
    "ontime_event_count",
    "schedulable_arrival_count",
    "intervention_count",
    "decision_step_count",
    "energy_proxy_total",
    "source_mode",
    "control_step_minutes",
    "causal_comparison_allowed",
    "simulator_adapter_version",
    "graph_scope",
    "num_agents",
    "num_nodes",
}

REQUIRED_KPIS = {
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
}


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_fixed_policy(actions_by_step):
    adapter = CausalSimulatorAdapter({
        "condition_id": "A",
        "seed": 77,
        "num_agents": 8,
        "max_steps": len(actions_by_step),
    })

    obs = adapter.reset(
        seed=77,
        scenario_config={
            "window_id": "step77_toy_window",
            "state_ts": "2024-01-01T08:00:00Z",
            "time_band": "peak",
        },
    )

    for actions in actions_by_step:
        result = adapter.step(actions)
        obs = result.obs

    return adapter, obs


def test_import_and_interface_load() -> None:
    cls = load_adapter_class("adapters.causal_simulator_adapter.CausalSimulatorAdapter")
    assert_true(cls.__name__ == "CausalSimulatorAdapter", "load_adapter_class failed")


def test_reset_observation_contract() -> None:
    adapter = CausalSimulatorAdapter({"num_agents": 8})
    obs = adapter.reset(seed=1, scenario_config={"time_band": "offpeak"})

    assert_true(REQUIRED_OBS_KEYS.issubset(set(obs.keys())), "ObsDict missing required keys")
    assert_true(obs["actor_obs"].shape == (8, 16), "actor_obs shape mismatch")
    assert_true(obs["critic_obs"].shape == (1, 64), "critic_obs shape mismatch")
    assert_true(obs["active_bus_mask"].shape == (8,), "active_bus_mask shape mismatch")
    assert_true(obs["action_mask"].shape == (8, 3), "action_mask shape mismatch")

    assert_true(adapter.num_agents == 8, "num_agents mismatch")
    assert_true(adapter.action_space["action_dim"] == 3, "action_dim must be 3")
    assert_true(adapter.control_step_minutes == 30, "control_step_minutes must be 30")

    graph = adapter.get_graph_skeleton()
    assert_true(6 <= graph.num_nodes <= 10, "toy graph must have 6 to 10 nodes")
    assert_true(graph.edge_index.shape[0] == 2, "edge_index must have shape [2, E]")
    assert_true(graph.edge_attr.shape[1] == 4, "edge_attr must have 4 columns")


def test_causal_action_changes_next_state() -> None:
    dispatch_actions = {i: ACTION_DISPATCH for i in range(8)}
    hold_actions = {i: ACTION_HOLD for i in range(8)}

    adapter_dispatch, obs_dispatch = run_fixed_policy([dispatch_actions])
    adapter_hold, obs_hold = run_fixed_policy([hold_actions])

    positions_dispatch = adapter_dispatch.state.bus_positions.copy()
    positions_hold = adapter_hold.state.bus_positions.copy()
    queues_dispatch = adapter_dispatch.state.queues.copy()
    queues_hold = adapter_hold.state.queues.copy()

    assert_true(
        not np.array_equal(positions_dispatch, positions_hold),
        "different actions must produce different bus positions",
    )

    assert_true(
        not np.allclose(queues_dispatch, queues_hold),
        "different actions must produce different passenger queues",
    )

    assert_true(
        not np.allclose(obs_dispatch["actor_obs"], obs_hold["actor_obs"]),
        "different actions must produce different next observations",
    )


def test_deterministic_same_seed_same_actions() -> None:
    actions_1 = {i: ACTION_DISPATCH for i in range(8)}
    actions_2 = {
        0: ACTION_HOLD,
        1: ACTION_DISPATCH,
        2: ACTION_SKIP,
        3: ACTION_DISPATCH,
        4: ACTION_HOLD,
        5: ACTION_DISPATCH,
        6: ACTION_SKIP,
        7: ACTION_DISPATCH,
    }

    adapter_a, obs_a = run_fixed_policy([actions_1, actions_2])
    adapter_b, obs_b = run_fixed_policy([actions_1, actions_2])

    assert_true(np.allclose(adapter_a.state.queues, adapter_b.state.queues), "queues not deterministic")
    assert_true(np.array_equal(adapter_a.state.bus_positions, adapter_b.state.bus_positions), "positions not deterministic")
    assert_true(np.allclose(obs_a["actor_obs"], obs_b["actor_obs"]), "obs not deterministic")
    assert_true(adapter_a.compute_kpis() == adapter_b.compute_kpis(), "KPI dict not deterministic")


def test_step_info_and_kpis() -> None:
    actions = {
        0: ACTION_HOLD,
        1: ACTION_DISPATCH,
        2: ACTION_SKIP,
        3: ACTION_DISPATCH,
        4: ACTION_HOLD,
        5: ACTION_DISPATCH,
        6: ACTION_SKIP,
        7: ACTION_DISPATCH,
    }

    adapter = CausalSimulatorAdapter({"num_agents": 8, "max_steps": 2})
    adapter.reset(seed=5, scenario_config={"time_band": "peak", "window_id": "kpi_test"})
    result = adapter.step(actions)

    assert_true(result.info["control_step_minutes"] == 30, "step info control_step_minutes mismatch")
    assert_true(result.info["causal_comparison_allowed"] is True, "causal comparison flag must be true")
    assert_true(result.info["source_mode"].startswith("causal_"), "source_mode must start with causal_")

    kpis = adapter.compute_kpis()

    assert_true(REQUIRED_KPIS.issubset(set(kpis.keys())), "compute_kpis missing shared KPI keys")
    assert_true(kpis["avg_wait_seconds"] >= 0, "avg_wait_seconds must be non-negative")
    assert_true(0 <= kpis["bunching_rate"] <= 1, "bunching_rate must be bounded")
    assert_true(0 <= kpis["on_time_rate"] <= 1, "on_time_rate must be bounded")
    assert_true(0 <= kpis["intervention_rate"] <= 1, "intervention_rate must be bounded")
    assert_true(kpis["energy_proxy"] >= 0, "energy_proxy must be non-negative")


def test_raw_event_and_window_rollup_schema() -> None:
    mixed_actions = {
        0: ACTION_HOLD,
        1: ACTION_DISPATCH,
        2: ACTION_SKIP,
        3: ACTION_DISPATCH,
        4: ACTION_HOLD,
        5: ACTION_DISPATCH,
        6: ACTION_SKIP,
        7: ACTION_DISPATCH,
    }

    adapter, _ = run_fixed_policy([mixed_actions, mixed_actions])

    raw_events = adapter.get_raw_events()
    assert_true(len(raw_events) == 16, "expected num_agents * steps raw events")
    assert_true(REQUIRED_RAW_EVENT_COLUMNS.issubset(set(raw_events[0].keys())), "raw event schema missing columns")
    assert_true(all(r["causal_comparison_allowed"] for r in raw_events), "raw causal flag must be true")
    assert_true(all(str(r["source_mode"]).startswith("causal_") for r in raw_events), "raw source_mode must be causal")

    rollup = adapter.build_window_rollup_row()

    assert_true(REQUIRED_WINDOW_ROLLUP_COLUMNS.issubset(set(rollup.keys())), "window rollup schema missing columns")
    assert_true(rollup["evaluation_horizon_minutes"] == 30, "rollup horizon must be 30")
    assert_true(rollup["control_step_minutes"] == 30, "rollup control granularity must be 30")
    assert_true(rollup["causal_comparison_allowed"] is True, "rollup causal flag must be true")
    assert_true(str(rollup["source_mode"]).startswith("causal_"), "rollup source_mode must be causal")
    assert_true(rollup["decision_step_count"] > 0, "decision_step_count must be positive")
    assert_true(rollup["energy_proxy_total"] >= 0, "energy_proxy_total must be non-negative")
    assert_true(
        rollup["passenger_demand_generated"] >= rollup["passenger_served_count"],
        "served cannot exceed generated in this toy test",
    )


def test_invalid_agent_count_is_blocked() -> None:
    blocked = False
    try:
        CausalSimulatorAdapter({"num_agents": 4})
    except ValueError:
        blocked = True
    assert_true(blocked, "num_agents below 5 must be blocked")

    blocked = False
    try:
        CausalSimulatorAdapter({"num_agents": 11})
    except ValueError:
        blocked = True
    assert_true(blocked, "num_agents above 10 must be blocked")


def main() -> None:
    tests = [
        test_import_and_interface_load,
        test_reset_observation_contract,
        test_causal_action_changes_next_state,
        test_deterministic_same_seed_same_actions,
        test_step_info_and_kpis,
        test_raw_event_and_window_rollup_schema,
        test_invalid_agent_count_is_blocked,
    ]

    for test in tests:
        test()
        print(f"[OK] {test.__name__}")

    print("[OK] Step 77 causal simulator adapter contract self-test PASS")


if __name__ == "__main__":
    main()
'@

$TestPath = ".\05_training\adapters\test_causal_simulator_adapter_contract.py"
$TestPy | Set-Content -Path $TestPath -Encoding UTF8

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

Write-Host "[INFO] python = $py"

& $py -m py_compile `
    ".\05_training\adapters\causal_simulator_adapter.py" `
    ".\05_training\adapters\test_causal_simulator_adapter_contract.py"

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 77 py_compile failed"
}

& $py ".\05_training\adapters\test_causal_simulator_adapter_contract.py"

if ($LASTEXITCODE -ne 0) {
    throw "[FAIL] Step 77 causal simulator adapter contract self-test failed"
}

Write-Host ""
Write-Host "[DONE] Step 77 causal simulator adapter contract complete."
Write-Host "[OUTPUT]"
Write-Host "  05_training\adapters\causal_simulator_adapter_contract.md"
Write-Host "  05_training\adapters\causal_simulator_adapter.py"
Write-Host "  05_training\adapters\test_causal_simulator_adapter_contract.py"
