# Step 91 ??Toy MAPPO Smoke Checkpoint Evaluation

## Purpose

Step 91 evaluates the Step 88 toy MAPPO smoke checkpoint on the Phase 2 toy causal adapter.

This is not a trained-model performance evaluation.

The goal is to validate the smoke evaluation wiring:

```text
smoke checkpoint
-> toy causal adapter rollout
-> raw_events.parquet / window_rollup.parquet
-> canonical KPI aggregation
-> 12-KPI inspector
-> evaluation_manifest.json
```

## Claim Boundary

All Step 91 outputs must keep:

```text
trained_model = false
performance_claim_allowed = false
smoke_evaluation_only = true
```

The result is only a smoke evaluation artifact. It must not be used as a paper-level performance claim.

## Outputs

```text
artifacts/phase2_toy_mappo_smoke_checkpoint_eval/
  evaluation_manifest.json
  toy_mappo_smoke_eval_contract.json
  rollouts/
  canonical_eval/
  inspection/
```

## Next Step

Recommended Step 92:

```text
Phase 2 toy causal train/eval runbook
```

or

```text
Step 86~91 project_log.md record
```
