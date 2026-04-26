# A-family Rollout Bridge v1

## Purpose

This bridge connects the reward contract, rollout schema, policy registry, and policy inference boundary.

It is a safe sidecar module created before modifying run_causal_rollout.py.

## Covered Conditions

- A
- A90
- A80
- A70

## Rules

- A uses 100 percent active bus count.
- A90 uses 90 percent active bus count.
- A80 uses 80 percent active bus count.
- A70 uses 70 percent active bus count.
- All A-family rows must keep qwen_trigger_rate equal to 0.0.
- Actual MAPPO rows require checkpoint_path.
- Placeholder rows are smoke/contract only.
- Placeholder rows must not be used for paper-level performance claims.

## Output

The bridge returns one extended window_rollup row with:

- policy provenance fields
- fleet fields
- derived passenger service fields
- derived energy per passenger field
- reward component fields
- validation through rollout_schema_v1
