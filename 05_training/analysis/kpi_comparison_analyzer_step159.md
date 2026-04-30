# Step 159 — KPI Comparison Analyzer Scaffold

## Purpose

Step 159 introduces a **scaffold-only** KPI comparison analyzer.
It loads canonical KPI tables, computes baseline-relative deltas per condition,
and emits CSV / JSON / Markdown artifacts for human review.

This step is **not** an actual performance comparison.
It does not release training, reward ablation, actual results, or paper-level claims.

## Folder policy

```text
05_training/analysis/
```

Generated outputs:

```text
artifacts/analysis/kpi_comparison_analyzer_step159/
```

`artifacts/` is git-ignored. Outputs must not be committed.

## Contract status

```text
comparison_tool_status = KPI_COMPARISON_ANALYZER_READY_SCAFFOLD_STILL_LOCKED
```

This means:

- The comparison tool scaffold is ready for offline review on local data.
- Comparison itself is allowed (scaffold-level, not paper-level).
- Training, actual reward ablation, winner selection, and paper claims remain locked.
- The tool must not write to any database.
- The tool must not mutate live experiment variables.

## Inputs

The generator accepts canonical KPI rows in two ways:

1. `--input-csv <path>` — read a CSV with the canonical KPI schema.
2. `--input-parquet <path>` — read a parquet with the canonical KPI schema.
3. `--sample-mode` — synthesize a built-in sample DataFrame for self-test.

Required meta columns:

```text
condition_id
seed
time_band
window_count
```

KPI columns (12 canonical):

| KPI | Direction |
|-----|-----------|
| cv_headway | lower_is_better |
| avg_wait_seconds | lower_is_better |
| bunching_rate | lower_is_better |
| intervention_rate | lower_is_better |
| energy_proxy | lower_is_better |
| passenger_wait_p95_seconds | lower_is_better |
| energy_proxy_per_passenger | lower_is_better |
| on_time_rate | higher_is_better |
| passenger_demand_generated | higher_is_better |
| passenger_served_count | higher_is_better |
| passenger_service_rate | higher_is_better |
| fleet_reduction_ratio | higher_is_better |

In default (non-strict) mode, at least 6 of these 12 KPI columns must be present.
In `--strict` mode, all 12 KPI columns must be present.

The `--baseline-condition` flag selects the reference group (default `B0`).
Other conditions are compared against the baseline mean per KPI.

## Delta calculation

For each (group, KPI) cell:

```text
condition_mean = mean(group[kpi])
baseline_mean  = mean(baseline_rows[kpi])
raw_delta      = condition_mean - baseline_mean
pct_delta      = raw_delta / abs(baseline_mean) * 100   if baseline_mean != 0 else null
improvement_score = -raw_delta if direction == lower else +raw_delta
```

Three group axes are emitted:

- by condition: aggregate across seeds and time_bands.
- by seed: aggregate across time_bands, group by (condition, seed).
- by time_band: aggregate across seeds, group by (condition, time_band).

## Outputs

```text
artifacts/analysis/kpi_comparison_analyzer_step159/
  kpi_comparison_manifest_step159.json
  kpi_comparison_summary_step159.csv
  kpi_delta_by_condition_step159.csv
  kpi_delta_by_seed_step159.csv
  kpi_delta_by_time_band_step159.csv
  kpi_comparison_report_step159.md
  claim_guard_status_step159.json
```

## Locked guard values

`claim_guard_status_step159.json` must keep the following values:

```text
comparison_tool_status        = KPI_COMPARISON_ANALYZER_READY_SCAFFOLD_STILL_LOCKED
comparison_allowed            = true
winner_selected               = false
trainable_reward_promoted     = false
actual_results                = false
actual_execution_allowed      = false
actual_execution_released     = false
train_allowed                 = false
paper_level_claim_allowed     = false
causal_performance_claim_allowed = false
live_mutation_allowed         = false
db_write_allowed              = false
h200_required_for_actual_claim = true
next_gate                     = WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RESULT
```

## Validator

`validate_kpi_comparison_analyzer_step159.py` checks:

- the manifest reports the locked scaffold status,
- all locked guard values are correct,
- output CSV / Markdown / JSON files exist and are non-empty,
- the baseline condition appears in the summary,
- delta CSVs have at least one row.

## Self-test

`test_kpi_comparison_analyzer_step159.py` covers:

1. sample-mode end-to-end PASS,
2. tampered guard with `train_allowed=true` is rejected,
3. tampered guard with `winner_selected=true` is rejected,
4. strict mode with missing KPIs is rejected,
5. missing baseline condition is rejected,
6. baseline mean = 0 produces null `pct_delta` without crash.

## Prohibited claims

This artifact does **not** authorize:

- training,
- actual reward ablation execution,
- winner selection or trainable reward promotion,
- causal performance claims,
- paper-level claims,
- database writes,
- mutation of live experiment configuration.

## Next gate

```text
WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RESULT
```

The comparison tool stays at scaffold status until real H200 actual KPI rows arrive.
