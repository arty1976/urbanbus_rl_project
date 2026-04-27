# Step 77 ??Phase 2 Causal Simulator Adapter Contract

## Purpose

This document defines the first Phase 2 causal simulator adapter contract for `urbanbus_rl_project`.

The purpose of Step 77 is not to claim paper-level performance.

The purpose is to prove the minimum causal chain:

```text
action_t -> simulator dynamics -> state_{t+1} -> KPI
```

## Historical Replay vs Causal Simulator

### HistoricalReplayAdapter

HistoricalReplayAdapter is valid for contract validation, shape validation, checkpoint path validation, and policy path smoke validation.

However, it is non-causal.

```text
action_t does not change state_{t+1}
```

Therefore, historical replay cannot support causal performance claims.

### CausalSimulatorAdapter

CausalSimulatorAdapter must satisfy:

```text
same initial state
+ same exogenous demand seed
+ different action sequence
=> different next state
=> different KPI path
```

Only adapters satisfying this property may set:

```text
causal_comparison_allowed = true
```

## Initial Scope

Step 77 uses a toy/minimal simulator.

Initial scope:

```text
spatial scope: Suseong-gu toy corridor, Beomeo-Manchon style
nodes: 6 to 10 stops
agents: 5 to 10 bus agents
control granularity: 30 minutes
```

## Required Interface

The adapter must remain compatible with `SimulatorAdapterInterface`.

Required methods:

```text
reset(seed=None, scenario_config=None) -> ObsDict
step(actions) -> StepResult
get_graph_skeleton() -> GraphSkeleton
compute_kpis(trajectory=None) -> dict
close() -> None
```

Required properties:

```text
num_agents
observation_space
action_space
```

## Action Space

Initial discrete action space:

```text
0 = hold
1 = dispatch
2 = skip
```

## Minimum Action Effects

### hold

```text
bus position unchanged
passenger queues continue growing
hold_seconds increases
idle energy increases
intervention_count increases
```

### dispatch

```text
bus moves to next stop
destination queue decreases by served passenger count
arrival/headway event is recorded
distance energy and acceleration energy increase
```

### skip

```text
bus skips next stop and moves two stops forward
skipped stop queue is not served
destination queue may be served
distance energy increases more than dispatch
intervention_count increases
```

## Raw Event Schema

Minimum raw event columns:

```text
condition_id
seed
window_id
state_ts
time_band
agent_id
action
action_name
position_before
position_after
waiting_passenger_cnt
passenger_served_count
distance_m
hold_seconds
acceleration_event_count
energy_proxy_total
intervention_applied
terminated
truncated
source_mode
causal_comparison_allowed
```

## Window Rollup Schema

Minimum canonical-style window rollup columns:

```text
condition_id
seed
window_id
state_ts
service_date
time_band
evaluation_horizon_minutes
qwen_trigger_rate
effective_replay_step_minutes
headway_mean_seconds
headway_std_seconds
headway_sample_count
bunching_event_count
headway_event_count
wait_total_passenger_seconds
wait_passenger_count
ontime_event_count
schedulable_arrival_count
intervention_count
decision_step_count
energy_proxy_total
source_mode
```

Recommended metadata columns:

```text
control_step_minutes
causal_comparison_allowed
simulator_adapter_version
graph_scope
num_agents
num_nodes
passenger_demand_generated
passenger_served_count
```

## Shared KPI Keys

The adapter-level convenience KPI output must include:

```text
cv_headway
avg_wait_seconds
bunching_rate
on_time_rate
intervention_rate
energy_proxy
```

Official comparison must still go through the shared canonical KPI aggregator.

## Causal Claim Gate

`causal_comparison_allowed=true` is allowed only when all conditions are true:

```text
1. source_mode starts with causal_
2. step(actions) changes future simulator state
3. passenger queue update is action-sensitive
4. bus position/headway update is action-sensitive
5. same seed and same action sequence are deterministic
6. different valid action sequences produce different next states
7. canonical raw/window rollup schema is preserved
```

Step 77 validates only the contract and toy dynamics.

It does not make a paper-level performance claim.
