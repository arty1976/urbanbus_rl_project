# A-family Rollout Policy Integration Plan v1

## 1. Purpose

This document defines how A/A90/A80/A70 rollout writers should integrate MAPPO policy action sources.

This is an integration plan, not the final rollout writer replacement.

The goal is to prevent the following mistakes:

1. mock/stub actions being silently treated as MAPPO actions;
2. smoke checkpoints being treated as actual trained checkpoints;
3. Qwen-enabled checkpoints entering the A-family pure MAPPO path;
4. policy provenance metadata being dropped before canonical KPI aggregation;
5. non-causal replay outputs being used for causal performance claims.

## 2. A-family conditions

The A-family conditions are:

| condition_id | Meaning |
|---|---|
| A | pure MAPPO baseline |
| A90 | A-family variant, same MAPPO policy interface |
| A80 | A-family variant, same MAPPO policy interface |
| A70 | A-family variant, same MAPPO policy interface |

For this integration stage, all A-family conditions are Qwen-disabled:

```text
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
```

## 3. Allowed policy input modes

Rollout writers may support three policy input modes.

### 3.1 mock_smoke

For development only.

- No checkpoint required.
- May use conservative mock action.
- Must set `policy_source = mock_policy`.
- Must set `mock_action_used = true`.
- Must set `placeholder_fallback_used = true` or explicitly indicate mock path.
- Must never be claim-ready.

### 3.2 mappo_smoke

For boundary validation only.

- Requires checkpoint path.
- Uses `MAPPOPolicyActionSource`.
- Uses validator mode `smoke`.
- Requires `checkpoint_loaded = true`.
- Requires `checkpoint_validator_ran = true`.
- Must set `policy_source = mappo_policy`.
- Must set `mock_action_used = false`.
- Must set `placeholder_fallback_used = false`.
- Must never be actual performance claim-ready.

### 3.3 mappo_actual

For H200-trained checkpoint only.

- Requires checkpoint path.
- Uses `MAPPOPolicyActionSource`.
- Uses validator mode `actual`.
- Requires `trained_model = true`.
- Requires `performance_claim_allowed = true`.
- Requires `checkpoint_loaded = true`.
- Requires `checkpoint_validator_ran = true`.
- Requires Qwen-disabled metadata.
- May become actual-policy-claim-ready.
- May become causal-policy-claim-ready only when the simulator source is causal.

## 4. Rollout writer target flow

Target flow for A-family rollout writer:

```text
scenario window
→ simulator.reset(seed, scenario_config)
→ obs
→ policy action source selected by --policy-source-mode
→ actions + policy metadata
→ simulator.step(actions)
→ raw_events rows include policy metadata
→ window_rollup rows include policy metadata summary
→ canonical KPI aggregator receives source_mode/policy metadata
```

## 5. Metadata fields that must propagate

Rollout outputs should retain these fields:

```text
policy_metadata_version
condition_id
source_mode
policy_source
policy_action_source_version
policy_action_source_mode
checkpoint_path
checkpoint_validation_mode
checkpoint_validator_ran
checkpoint_loaded
trained_model
performance_claim_allowed
placeholder_fallback_used
mock_action_used
qwen_train
qwen_inference
qwen_trigger_rate
reward_version
energy_proxy_model_version
k_dist_kwh_per_m
k_acc_kwh_per_event
k_idle_kwh_per_sec
actual_policy_claim_ready
causal_policy_claim_ready
```

## 6. Claim rules

Paper-level actual MAPPO policy claim requires:

```text
policy_source = mappo_policy
checkpoint_validator_ran = true
checkpoint_loaded = true
trained_model = true
performance_claim_allowed = true
mock_action_used = false
placeholder_fallback_used = false
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
reward_version = mappo_reward_v1
energy_proxy_model_version = daegu_energy_proxy_v1
K_DIST/K_ACC/K_IDLE constants match contract
```

Causal performance claim additionally requires:

```text
causal simulator = true
source_mode starts with causal_
same window/demand across A/A90/A80/A70
canonical KPI aggregation pass
```

## 7. Step 51 boundary

Step 51 does not replace rollout writers yet.

Step 51 creates:

- integration plan document;
- A-family policy integration helper;
- self-test validating allowed and forbidden combinations.

Actual rollout writer modification starts after this step.
