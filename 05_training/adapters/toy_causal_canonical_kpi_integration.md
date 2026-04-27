# Step 79 ??Toy Causal Rollout to Canonical KPI Integration

## Goal

Step 79 connects the Phase 2 toy causal simulator outputs to the existing canonical KPI aggregator.

Input:

```text
artifacts/phase2_toy_causal_v1_canonical_selftest/rollouts/**/window_rollup.parquet
```

Aggregator mode:

```text
official_rollup
```

Expected canonical outputs:

```text
kpi_by_window.parquet
kpi_by_seed.parquet
kpi_by_time_band.parquet
kpi_overall.json
aggregation_manifest.json
```

## Claim Boundary

This step proves that toy causal simulator outputs can pass through the canonical KPI path.

It does not prove real-world performance.

It does not make a paper-level performance claim.

## Required Success Criteria

- `window_rollup.parquet` is discovered recursively.
- shared KPI columns are computed.
- `source_mode=causal_toy_suseong_v1` is preserved.
- `causal_comparison_allowed=true` is preserved in canonical outputs.
- `strict_canonical=true` is set.
- `kpi_overall.json` reports causal comparison allowed.
