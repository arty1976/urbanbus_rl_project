# MAPPO Checkpoint Contract v1

## 1. Purpose

This document defines the checkpoint contract for actual MAPPO inference in `urbanbus_rl_project`.

MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) checkpoints must satisfy this contract before they can be treated as actual trained policy artifacts.

This contract exists to prevent the following mistakes:

1. treating a fake/mock/smoke checkpoint as an actual trained policy;
2. mixing Qwen-enabled and Qwen-disabled conditions;
3. loading a checkpoint whose observation/action dimensions do not match the inference adapter;
4. making paper-level performance claims from scaffold-only artifacts;
5. losing reward/energy proxy version information required for reproducibility.

## 2. Scope

This contract applies to Experiment A:

- condition_id: `A`
- qwen_train: `false`
- qwen_inference: `false`
- qwen_trigger_rate: `0.0`

A checkpoint that violates any of these conditions must not be accepted for Experiment A inference.

## 3. Required top-level keys

Every actual MAPPO checkpoint must be a Python dictionary saved with `torch.save`.

Required keys:

| Key | Required value / rule |
|---|---|
| `artifact_version` | `mappo_policy_checkpoint_v1` |
| `contract_version` | `mappo_checkpoint_contract_v1` |
| `condition_id` | `A` |
| `qwen_train` | `false` |
| `qwen_inference` | `false` |
| `qwen_trigger_rate` | `0.0` |
| `policy_architecture` | `ActorCriticMLP` |
| `actor_obs_dim` | integer, default expected `16` |
| `critic_obs_dim` | integer, default expected `64` |
| `action_dim` | integer, default expected `2` |
| `hidden_dim` | integer, default expected `128` |
| `shared_policy` | `true` |
| `ctde_enabled` | `true` |
| `model_state_dict` | PyTorch model state dictionary |
| `training_seed` | integer |
| `git_commit` | non-empty string |
| `created_at_utc` | non-empty UTC timestamp string |
| `trained_model` | boolean |
| `performance_claim_allowed` | boolean |
| `reward_version` | `mappo_reward_v1` |
| `rollout_schema_version` | `rollout_schema_v1` |
| `policy_interface_version` | `mappo_policy_interface_v1` |
| `action_space_version` | `bus_control_action_v1` |
| `observation_space_version` | `urbanbus_observation_v1` |
| `energy_proxy_model_version` | `daegu_energy_proxy_v1` |
| `energy_proxy_unit` | `kwh_equivalent` |
| `k_dist_kwh_per_m` | `0.0012` |
| `k_acc_kwh_per_event` | `0.1800` |
| `k_idle_kwh_per_sec` | `0.0080` |

CTDE (Centralized Training Decentralized Execution=중앙집중 학습 분산 실행) must remain enabled for this contract.

## 4. Actual mode vs smoke mode

The validator supports two modes.

### 4.1 actual mode

`actual` mode is for H200-trained MAPPO checkpoints.

Rules:

- `trained_model` must be `true`.
- `performance_claim_allowed` must be `true`.
- no fake/mock/stub/smoke/placeholder marker may be present.
- `model_state_dict` must load into the neural MAPPO adapter architecture without missing or unexpected keys.

Only checkpoints that pass `actual` mode may be used for paper-level performance claims.

### 4.2 smoke mode

`smoke` mode is for boundary validation only.

Rules:

- `trained_model` may be `false`.
- `performance_claim_allowed` must be `false` for fake/mock/scaffold checkpoints.
- fake checkpoint files may be used only to test strict loading boundaries.
- smoke-mode artifacts must never be used for performance claims.

## 5. Energy proxy constants

The checkpoint must pin the Daegu energy proxy model constants:

energy_kwh_equiv =
  0.0012 * distance_m
  + 0.1800 * acceleration_event_count
  + 0.0080 * hold_seconds

Then:

energy_proxy_per_passenger =
  energy_kwh_equiv / max(passenger_served_count, epsilon)

The energy proxy model itself will be implemented later as `daegu_energy_proxy_v1`.

## 6. Acceptance rule

A checkpoint is accepted as an actual MAPPO checkpoint only if:

1. all required keys exist;
2. all required constants match exactly;
3. Qwen is disabled;
4. the policy dimensions match the inference adapter;
5. the `model_state_dict` loads cleanly into the neural MAPPO model;
6. `trained_model = true`;
7. `performance_claim_allowed = true`;
8. no fake/mock/stub/smoke/placeholder markers exist.

If any rule fails, the inference path must STOP instead of falling back to a placeholder policy.
