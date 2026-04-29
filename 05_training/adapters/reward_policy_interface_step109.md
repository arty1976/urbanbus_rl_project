# Step 109 — Reward-policy interface scaffold

## Purpose

Step 109 converts the Step 108 scaffold reward output into a policy-facing reward interface artifact.

This is **not** the final MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) reward design.

## Inputs

Default input:

```text
artifacts/daegu_bis_api_audit/queue_demand_reward_wiring_step108/reward_by_window_step108.parquet
```

CSV fallback is supported.

## Outputs

Default output directory:

```text
artifacts/daegu_bis_api_audit/reward_policy_interface_step109/
```

Generated files:

```text
policy_reward_interface_step109.parquet
policy_reward_interface_step109.csv
policy_reward_interface_summary_step109.json
policy_reward_interface_schema_step109.json
policy_reward_interface_readiness_step109.csv
reward_policy_interface_step109_manifest.json
reward_policy_interface_step109_report.md
```

## Required guard

The interface intentionally keeps:

```text
reward_scaffold_only = true
reward_weights_are_final = false
reward_formula_finalized = false
final_reward_design_claim_allowed = false
train_with_this_reward_allowed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
canonical_causal_comparison_allowed = false
```

## Interpretation

Allowed claim:

```text
Step 108 scaffold reward rows can be converted into a policy-facing interface artifact.
```

Forbidden claims:

```text
This is the final reward design.
This reward is ready for actual MAPPO training.
This reward can support paper-level performance claims.
This reward proves causal performance.
```

## Run

```powershell
python -m py_compile `
  05_training/adapters/reward_policy_interface_step109.py `
  05_training/adapters/test_reward_policy_interface_step109.py

python 05_training/adapters/test_reward_policy_interface_step109.py

python 05_training/adapters/reward_policy_interface_step109.py
```

## Next step

Step 110 should either:

1. connect this interface to a mock policy runner while preserving scaffold guards, or
2. open the final reward-design review gate that decides which components, weights, scaling, and constraints become the actual MAPPO training reward.
