# Step 81 ??Toy Causal A-Family Full Smoke Matrix

## Goal

Step 81 expands the Phase 2 toy causal simulator validation from a small A/A90 sample to the full A-family smoke matrix:

```text
A, A90, A80, A70
x seeds 1,2,3
= 12 condition-seed runs
```

This validates the experimental matrix wiring before connecting trained MAPPO policy checkpoints.

## Inputs

The test uses:

```text
05_training/run_toy_causal_rollout.py
05_training/adapters/causal_simulator_adapter.py
05_training/evaluation/canonical_kpi_aggregator.py
```

## Expected Outputs

```text
artifacts/phase2_toy_causal_a_family_matrix_selftest/
  rollouts/
    scenario_index.parquet
    A/rollouts/seed_001/window_rollup.parquet
    ...
    A70/rollouts/seed_003/window_rollup.parquet

  canonical_eval/
    kpi_by_window.parquet
    kpi_by_seed.parquet
    kpi_by_time_band.parquet
    kpi_overall.json
    aggregation_manifest.json
```

## Success Criteria

- 4 A-family conditions are generated.
- 3 seeds are generated.
- 12 condition-seed rollout directories exist.
- 36 window rollup rows are generated.
- 576 raw event rows are generated.
- canonical KPI aggregation passes in `official_rollup` mode.
- `causal_comparison_allowed=true` is preserved in all canonical outputs.
- shared KPI columns are valid.

## Claim Boundary

This is still toy causal simulator validation.

It is not a trained MAPPO result.

It is not a paper-level performance claim.

The result only proves that the Phase 2 A-family causal evaluation matrix can be generated, materialized, and canonicalized.
