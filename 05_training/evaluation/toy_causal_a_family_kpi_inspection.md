# Step 84 ??Toy Causal A-Family 12-KPI Result Inspector

## Goal

Step 84 reads the Phase 2 toy causal canonical KPI outputs and produces human-readable inspection tables.

Input directory:

```text
artifacts/phase2_toy_causal_a_family_matrix_selftest/canonical_eval
```

Expected input files:

```text
kpi_by_window.parquet
kpi_by_seed.parquet
kpi_by_time_band.parquet
kpi_overall.json
```

Output directory:

```text
artifacts/phase2_toy_causal_a_family_matrix_selftest/inspection
```

Generated files:

```text
condition_summary.csv
condition_vs_A_delta.csv
time_band_summary.csv
seed_summary.csv
inspection_report.json
```

## 12 KPI Schema

The inspector validates the Phase 2 12-KPI schema:

```text
cv_headway
avg_wait_seconds
bunching_rate
on_time_rate
intervention_rate
energy_proxy
passenger_demand_generated
passenger_served_count
passenger_service_rate
passenger_wait_p95_seconds
energy_proxy_per_passenger
fleet_reduction_ratio
```

## Interpretation Boundary

This is a toy causal sanity report.

It is not a paper-level performance claim.

The report only checks whether A/A90/A80/A70 differences are materialized, canonicalized, and inspectable through the 12-KPI schema.
