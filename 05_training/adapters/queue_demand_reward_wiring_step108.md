# Step 108 — Queue/Demand Reward Wiring Scaffold

## Purpose

Step 108 connects the Step 107 preserved 12-KPI window output to a scaffold reward calculation. The goal is to verify the wiring path:

```text
kpi_by_window_queue_demand_preserved.parquet
→ reward component terms
→ reward_by_window_step108.parquet
→ reward summaries
```

This is **not** the final MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) reward design.

## Inputs

Default input:

```text
artifacts/daegu_bis_api_audit/queue_demand_canonical_kpi_step107/canonical_eval/kpi_by_window_queue_demand_preserved.parquet
```

Required 12 KPI (Key Performance Indicator=핵심 성과 지표) columns:

```text
cv_headway
avg_wait_seconds
bunching_rate
on_time_rate
intervention_rate
energy_proxy
passenger_demand_generated
passenger_served_count
passenger_service_rate
passenger_wait_p95_seconds
energy_proxy_per_passenger
fleet_reduction_ratio
```

## Temporary scaffold reward components

The Step 108 script computes:

```text
reward_service
reward_wait
reward_long_wait
reward_ontime
reward_bunching
reward_energy
reward_fleet
reward_constraint
reward_total
```

The weights are explicitly marked scaffold-only:

```text
reward_scaffold_only = true
reward_weights_are_final = false
reward_formula_finalized = false
final_reward_design_claim_allowed = false
```

## Claim guards

Step 108 must keep:

```text
queue_demand_observed = false
queue_demand_proxy = true
actual_passenger_wait_observed = false
actual_headway_observed = false
actual_arrival_departure_time_observed = false
actual_dwell_observed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```

## Outputs

```text
artifacts/daegu_bis_api_audit/queue_demand_reward_wiring_step108/reward_by_window_step108.parquet
artifacts/daegu_bis_api_audit/queue_demand_reward_wiring_step108/reward_by_window_step108.csv
artifacts/daegu_bis_api_audit/queue_demand_reward_wiring_step108/reward_by_seed_step108.csv
artifacts/daegu_bis_api_audit/queue_demand_reward_wiring_step108/reward_by_condition_step108.csv
artifacts/daegu_bis_api_audit/queue_demand_reward_wiring_step108/reward_overall_step108.json
artifacts/daegu_bis_api_audit/queue_demand_reward_wiring_step108/queue_demand_reward_readiness_step108.csv
artifacts/daegu_bis_api_audit/queue_demand_reward_wiring_step108/queue_demand_reward_wiring_step108_manifest.json
artifacts/daegu_bis_api_audit/queue_demand_reward_wiring_step108/queue_demand_reward_wiring_step108_report.md
```

## Commands

```powershell
python -m py_compile `
  05_training/adapters/queue_demand_reward_wiring_step108.py `
  05_training/adapters/test_queue_demand_reward_wiring_step108.py

python 05_training/adapters/test_queue_demand_reward_wiring_step108.py

python 05_training/adapters/queue_demand_reward_wiring_step108.py
```

## Next step

Step 109 should connect the reward output to a policy-interface scaffold without treating the temporary reward values as final training rewards.
