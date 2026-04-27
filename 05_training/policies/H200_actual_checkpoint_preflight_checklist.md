# H200 Actual MAPPO Checkpoint Preflight Checklist v1

## 1. Purpose

This checklist defines the minimum conditions for accepting an H200-trained MAPPO checkpoint as an actual policy checkpoint.

This document is not for smoke checkpoints.

Smoke checkpoints can validate code boundaries, but they must never be used for actual performance claims.

## 2. Acronyms

- MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화)
- KPI (Key Performance Indicator=핵심성과지표)
- CTDE (Centralized Training Decentralized Execution=중앙집중 학습 분산 실행)

## 3. Required checkpoint acceptance rules

An H200 checkpoint can enter `mappo_actual` only when all of the following are true.

```text
checkpoint file exists
checkpoint contract v1 is satisfied
validate_mappo_checkpoint.py --mode actual PASS
trained_model = true
performance_claim_allowed = true
condition_id = A
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
reward_version = mappo_reward_v1
rollout_schema_version = rollout_schema_v1
policy_interface_version = mappo_policy_interface_v1
action_space_version = bus_control_action_v1
observation_space_version = urbanbus_observation_v1
energy_proxy_model_version = daegu_energy_proxy_v1
energy_proxy_unit = kwh_equivalent
k_dist_kwh_per_m = 0.0012
k_acc_kwh_per_event = 0.1800
k_idle_kwh_per_sec = 0.0080
actor_obs_dim matches adapter contract
critic_obs_dim matches adapter contract
action_dim matches adapter contract
model_state_dict strict-loads into NeuralMAPPOInferenceAdapter
```

## 4. Actual policy claim rule

Actual policy claim can be considered only when:

```text
policy_source = mappo_policy
policy_source_mode = mappo_actual
checkpoint_validation_mode = actual
checkpoint_validator_ran = true
checkpoint_loaded = true
trained_model = true
performance_claim_allowed = true
mock_action_used = false
placeholder_fallback_used = false
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
actual_policy_claim_ready = true
```

## 5. Causal claim rule

Actual checkpoint alone is not enough for causal performance claims.

Causal claim additionally requires:

```text
causal simulator adapter
source_mode starts with causal_
same initial state across A/A90/A80/A70
same exogenous events across A/A90/A80/A70
same evaluation window across A/A90/A80/A70
canonical KPI aggregation PASS
post-canonical policy metadata validation PASS
causal_policy_claim_ready = true
```

HistoricalReplayAdapter remains non-causal.

Therefore, even if an H200 checkpoint passes actual preflight, a rollout through HistoricalReplayAdapter remains:

```text
actual policy path possible
causal performance claim not allowed
```

## 6. Required command sequence

Recommended sequence after H200 training creates `best.pt`:

```powershell
$ckpt = "C:\path\to\best.pt"

python .\05_training\policies\validate_mappo_checkpoint.py `
  --checkpoint $ckpt `
  --mode actual `
  --device cpu `
  --json-output .\artifacts\experiment_A_v1\h200_actual_preflight\actual_checkpoint_validation_report.json

python .\05_training\policies\h200_actual_checkpoint_preflight_checklist.py `
  --checkpoint-path $ckpt `
  --device cpu `
  --run-validator `
  --json-output .\artifacts\experiment_A_v1\h200_actual_preflight\h200_actual_preflight_report.json
```

After this passes, the next stage may prepare `mappo_actual` matrix execution.

## 7. Explicit non-claim cases

The following cases must never be used for actual performance claims.

```text
trained_model = false
performance_claim_allowed = false
checkpoint_validation_mode = smoke
policy_source_mode = mappo_smoke
mock_action_used = true
placeholder_fallback_used = true
qwen_train = true
qwen_inference = true
qwen_trigger_rate != 0.0
energy_proxy_model_version != daegu_energy_proxy_v1
K_DIST/K_ACC/K_IDLE mismatch
HistoricalReplayAdapter output used as causal comparison
```

## 8. Step boundary

Step 58 creates the checklist and preflight guard.

It does not train MAPPO.

It does not create a real H200 checkpoint.

It does not claim performance.

It prepares the acceptance gate for the future actual checkpoint.
