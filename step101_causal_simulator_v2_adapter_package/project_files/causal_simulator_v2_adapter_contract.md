# Step 101 — CausalSimulatorAdapter v2 Scaffold

Step 101 creates a static-signal-aware adapter scaffold.

It reads:

```text
artifacts/causal_simulator_v2_contract/causal_simulator_v2_input_contract.json
artifacts/signal_features_v2/node_signal_features.parquet
artifacts/signal_features_v2/edge_signal_features.parquet
```

It adds static signal context to observations:

```text
signal_count_250m
nearest_signal_distance_m
pedestrian_signal_count_250m
blink_signal_ratio_250m
controlled_signal_ratio_250m
signal_delay_risk_proxy
intersection_complexity_proxy
edge_signal_count
edge_signal_density_per_km
edge_control_complexity_proxy
```

It does **not** create:

```text
red_light_delay_seconds
green_time_seconds
cycle_length_seconds
phase_sequence
signal_offset_seconds
real_time_signal_state
queue_discharge_rate
```

Claim guardrails remain false:

```text
trained_model=false
performance_claim_allowed=false
causal_performance_claim_allowed=false
dynamic_signal_phase_claim_allowed=false
```

Next step:

```text
Step 102 — Connect CausalSimulatorAdapter v2 to rollout writer smoke path
```
