# Step 129 ??Actual Reward Ablation Runbook

## Purpose

Step 129 prepares the runbook for future actual reward ablation execution.

This step does not execute reward ablation, does not start MAPPO (Multi-Agent Proximal Policy Optimization=?ㅼ쨷 ?먯씠?꾪듃 洹쇱젒 ?뺤콉 理쒖쟻?? training, does not select a winner, and does not promote any reward.

## Current Status

```text
ACTUAL_ABLATION_RUNBOOK_READY_NOT_EXECUTED
```

Current guards:

```text
actual_execution_started = false
actual_results = false
winner_selected = false
trainable_reward_promoted = false
train_with_this_reward_allowed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```

## Execution Matrix

Future actual ablation requires:

```text
R0/R1/R2/R3/R4/R5
횞 A/A90/A80/A70
횞 seed 1,2,3
= 72 runs
```

## Operator Preflight

Before actual execution:

1. Git state must be clean or explicitly documented.
2. Step 111~128 artifacts must be committed.
3. Step 119 dry-run plan must exist.
4. Step 120 execution preflight must pass.
5. Step 121 execution manifest must exist.
6. Step 126 actual-result ingestion preflight must be available.
7. H200 or approved execution environment must be documented.
8. R0~R5 candidate definitions must match Step 115.
9. B1_noop normalization reference must be preserved.
10. Step 114 hard constraints must be active.

## Per-Run Required Outputs

Each future actual run must produce:

- `run_config.json`
- training/evaluation log
- canonical KPI output path
- reward candidate ID
- condition ID
- seed
- checkpoint path
- checkpoint SHA256
- trained model flag
- actual result flag
- hard constraint report
- `kpi_by_window.parquet`
- `kpi_by_seed.parquet`
- `kpi_overall.json`

## Hard Stop Conditions

Stop immediately if:

- template/no-op/dry-run/smoke-only markers appear in actual results
- checkpoint path or checkpoint hash is missing
- `trained_model=false`
- `actual_result=false`
- any selected candidate violates hard constraints
- any candidate/condition/seed row is missing
- paper or causal claim flags become true before later evidence gates
- winner is selected before actual-result scorecard evaluation

## Post-Run Required Sequence

After future actual execution:

1. collect all 72 actual outputs
2. build actual result bundle
3. run Step 126 ingestion preflight
4. populate Step 125 selection scorecard
5. prepare a new promotion decision package
6. only then consider trainable reward promotion

## Important

This runbook prepares the execution procedure only. It is not permission to execute.
