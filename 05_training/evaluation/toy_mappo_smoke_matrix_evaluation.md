# Step 93 ??Toy MAPPO Smoke Matrix Evaluation

## Purpose

Step 93 extends the Step 91 single-condition smoke checkpoint evaluator to the A-family matrix:

```text
A / A90 / A80 / A70
```

This is still a smoke evaluation.

It is not a paper-level performance comparison.

## What It Validates

The Step 93 matrix evaluator checks that the same smoke checkpoint can pass the evaluation pipeline across all A-family toy conditions:

```text
smoke checkpoint
-> condition-specific toy causal rollout
-> raw window_rollup with 12-KPI schema
-> condition-level canonical KPI aggregation
-> combined matrix canonical output
-> A-family 12-KPI inspector
-> matrix_evaluation_manifest.json
```

## Required Claim Boundary

All outputs must keep:

```text
trained_model = false
performance_claim_allowed = false
smoke_evaluation_only = true
matrix_smoke_evaluation_only = true
```

## Outputs

```text
artifacts/phase2_toy_mappo_smoke_matrix_eval/
  matrix_evaluation_manifest.json
  matrix_summary.csv
  matrix_training/
  conditions/
    A/
    A90/
    A80/
    A70/
  matrix_canonical_eval/
  matrix_inspection/
```

## Interpretation

Allowed:

```text
The toy MAPPO smoke evaluation pipeline runs across A/A90/A80/A70.
The 12-KPI schema is preserved across the matrix.
The inspector can produce A-family delta tables for smoke outputs.
```

Not allowed:

```text
A70 outperforms A in real operation.
The MAPPO policy has converged.
The result is suitable for a paper performance table.
```

## Next Step

Recommended Step 94:

```text
Step 91~93 project_log.md record and runbook update
```
