# Step 152 — H200 Monitoring Dashboard Contract

## Purpose

Step 152 defines a monitoring-only observability contract for H200-side reward ablation work.
It does **not** release training, reward ablation, or actual execution.

This step creates the first safe monitoring layer before any H200 actual run is allowed:

1. TensorBoard-compatible scalar naming contract.
2. JSONL run metric event schema.
3. GPU CSV metric schema.
4. Run status JSON schema.
5. Safety gate schema for early stop / abort decisions.
6. Explicit prohibition of live mutation of reward weights or experiment variables.

## Folder policy

Observability files belong under:

```text
05_training/observability/
```

Generated outputs belong under:

```text
artifacts/observability/h200_monitoring_dashboard_contract_step152/
```

Reason:

- `05_training/rewards/` is reserved for reward specifications, release manifests, and reward ablation guard chains.
- `05_training/observability/` is cross-cutting infrastructure for monitoring, logging, and dashboard contracts.
- `artifacts/observability/` keeps generated monitoring outputs separate from reward outputs.

## Contract status

```text
contract_status = OBSERVABILITY_CONTRACT_READY_MONITORING_ONLY_STILL_LOCKED
```

This means:

- monitoring contract is ready;
- sample logging schema is ready;
- training is still locked;
- actual reward ablation is still locked;
- dashboard must not mutate live experiment variables.

## Required lock values

The manifest must keep the following values:

```text
actual_execution_allowed = false
actual_execution_released = false
train_allowed = false
operator_approval_recorded = false
operator_approval_granted = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```

## Dashboard mode

```text
dashboard_mode = MONITORING_ONLY
control_policy = NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY
```

Allowed dashboard actions:

- display run status;
- display training metrics;
- display KPI metrics;
- display GPU/system metrics;
- display safety gate status;
- display checkpoint status;
- recommend abort when a hard safety gate fails.

Forbidden dashboard actions:

- changing reward weights in the middle of a run;
- changing learning rate in the middle of a run unless the run contract already defines a scheduler;
- changing seed in the middle of a run;
- changing condition_id, reward_id, fleet ratio, or Qwen flags during a run;
- promoting any run to paper-level claim status;
- opening actual reward ablation release.

If a parameter must be adjusted, it must be written into the **next-run config**, not live-mutated.

## Required metric groups

### 1. run_identity

Required fields:

```text
run_id
condition_id
reward_id
seed
git_commit
project_root
started_at_utc
```

### 2. training_metrics

Recommended TensorBoard scalar names:

```text
train/policy_loss
train/value_loss
train/entropy
train/kl_divergence
train/grad_norm
train/learning_rate
train/reward_mean
train/reward_std
train/episode_length_mean
```

### 3. traffic_kpis

Recommended TensorBoard scalar names:

```text
kpi/avg_wait_seconds
kpi/passenger_service_rate
kpi/passenger_wait_p95_seconds
kpi/bunching_rate
kpi/on_time_rate
kpi/energy_proxy_per_passenger
kpi/fleet_reduction_ratio
kpi/intervention_rate
kpi/cv_headway
```

### 4. safety_gates

Required safety fields:

```text
nan_detected
inf_detected
grad_norm_exceeded
reward_explosion_detected
loss_explosion_detected
checkpoint_validation_passed
abort_recommended
```

### 5. system_metrics

Recommended GPU fields:

```text
gpu_index
gpu_name
gpu_utilization_pct
gpu_memory_used_mb
gpu_memory_total_mb
gpu_power_draw_w
gpu_temperature_c
ecc_error_count
```

### 6. claim_guards

Required claim guard fields:

```text
trained_model
actual_results
winner_selected
trainable_reward_promoted
paper_level_claim_allowed
causal_performance_claim_allowed
```

## Generated files

The Step 152 generator should produce:

```text
h200_monitoring_dashboard_contract_step152_manifest.json
run_status_step152_sample.json
metrics_events_step152_sample.jsonl
gpu_metrics_step152_sample.csv
tensorboard_scalar_contract_step152.json
```

## Safety interpretation

A PASS from Step 152 means only this:

```text
The H200 monitoring and observability contract is ready.
```

It does **not** mean:

```text
actual reward ablation is released;
training is allowed;
H200 actual execution has started;
any paper-level result exists;
any causal performance claim is allowed.
```
