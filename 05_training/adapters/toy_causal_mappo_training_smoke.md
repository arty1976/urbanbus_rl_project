# Step 88 ??Toy Causal MAPPO Training Smoke Scaffold

## Purpose

Step 88 adds a minimal MAPPO-style training smoke scaffold for the Phase 2 toy causal simulator.

This is not full MAPPO training.

This is not a paper-level performance result.

The purpose is to validate this wiring:

```text
CausalSimulatorAdapter
-> toy MAPPO actor/critic
-> rollout buffer
-> reward_v1 team reward
-> one tiny policy update
-> smoke checkpoint
-> training_manifest.json
```

## Claim Boundary

All Step 88 outputs must retain:

```text
trained_model = false
performance_claim_allowed = false
smoke_training_only = true
```

The checkpoint is a smoke artifact, not a trained policy.

## Reward Path

`CausalSimulatorAdapter.step()` emits the 12-KPI reward metrics and `mappo_reward_v1` reward debug payload.

The training scaffold broadcasts the team reward to all agents and performs one small update only to verify gradients, losses, checkpoint serialization, and deterministic reward tracing.

## Outputs

```text
artifacts/phase2_toy_causal_mappo_smoke/
  training_manifest.json
  training_trace.csv
  checkpoints/toy_mappo_smoke_checkpoint.pt
```

## Next Step

Recommended Step 89:

```text
Smoke checkpoint contract validator
```

or

```text
Toy causal MAPPO smoke rollout evaluator
```
