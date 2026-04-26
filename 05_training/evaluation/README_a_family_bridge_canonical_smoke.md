# A-family Bridge Canonical KPI Smoke v1

## Purpose

This Step 30 smoke test validates the end-to-end path:

    run_causal_rollout.py --a-family-bridge
    -> A/A90/A80/A70 window_rollup files
    -> canonical_kpi_aggregator.py official_rollup
    -> condition-level canonical_eval outputs

## Important

This is not a performance claim.

In placeholder mode, outputs are valid only for contract and pipeline validation.

## Default command

    python .\05_training\evaluation\run_a_family_bridge_canonical_smoke_v1.py --write-parquet --smoke --self-test

## Outputs

Default output root:

    artifacts/step30_a_family_bridge_canonical_smoke

Main files:

- step30_manifest.json
- rollout/entry_manifest.json
- rollout/summary.json
- rollout/A/canonical_eval/kpi_by_window.parquet
- rollout/A90/canonical_eval/kpi_by_window.parquet
- rollout/A80/canonical_eval/kpi_by_window.parquet
- rollout/A70/canonical_eval/kpi_by_window.parquet

## Interpretation

Success means that the A-family bridge-generated rollout rows can be consumed by the canonical KPI aggregator.

It does not mean that MAPPO has been trained.
It does not mean that A/A90/A80/A70 performance is paper-ready.
