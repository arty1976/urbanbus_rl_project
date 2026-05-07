# Step 166 — Patent Evidence Report v2 Schema

## Purpose

Step 166 generates a patent-facing evidence report by combining:

1. Step 160 Zero-Loss Pickup Evidence Reporter outputs
2. Step 165 attempt-specific route/path attention filter outputs

The report is designed to document, in a reproducible non-claim bundle:

- how many pickup attempts were evaluated;
- how many attempts satisfied the zero-loss ETA condition for the already-boarded passenger;
- how real GATv2 attention evidence was distributed over attempt-specific route/path segments;
- whether attention rows were matched by exact route/path edge overlap, route/path node overlap, or fallback.

## Non-claim boundary

All Step 166 outputs must keep the following flags locked:

```text
simulation_evidence_only = true
actual_operational_claim_allowed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
train_allowed = false
```

The report is evidence-pipeline output only. It is not an actual operational performance claim and is not a causal performance claim.

## Required inputs

### A. Step 160 Zero-Loss manifest

`zero_loss_evidence_manifest.json`

Required fields:

- `audit_status = PASS`
- `bundle_status = ZERO_LOSS_PICKUP_EVIDENCE_BUNDLE_READY_NONCLAIM`
- `summary.total_pickup_attempts`
- `summary.zero_loss_success_count`
- `summary.zero_loss_success_rate`
- `output_files.eta_delta_by_attempt`
- `output_files.zero_loss_by_condition`
- `output_files.attention_mass_by_segment`
- `output_files.top_attention_edges`

### B. Step 165 path-filter manifest

`attempt_route_path_attention_filter_manifest.json`

Required fields:

- `audit_status = PASS`
- `bundle_status = ATTEMPT_ROUTE_PATH_ATTENTION_FILTERED_FOR_ZERO_LOSS_EVIDENCE_NONCLAIM`
- `connection_mode = attempt_route_path_edge_filter_step165`
- `row_counts.pickup_attempt_count`
- `row_counts.filtered_attention_row_count`
- `row_counts.attempt_count_with_filtered_attention`
- `row_counts.fallback_attempt_count`
- `row_counts.exact_edge_overlap_count`
- `row_counts.node_overlap_count`
- `output_files.gatv2_attention`
- `output_files.attention_path_mass_by_attempt`
- `output_files.attempt_route_path_filter_report`

## Step 166 outputs

Output root contains:

```text
patent_evidence_report_v2.md
patent_evidence_summary_v2.json
zero_loss_attempt_attention_summary_v2.csv
attention_mass_by_attempt_path_v2.csv
attention_mass_by_success_path_v2.csv
top_filtered_attention_edges_v2.csv
patent_evidence_report_v2_manifest.json
```

Optional plots may be generated when matplotlib is available:

```text
zero_loss_attention_mass_by_success_path_v2.png
zero_loss_delta_vs_attention_mass_v2.png
```

## Key output tables

### `zero_loss_attempt_attention_summary_v2.csv`

One row per pickup attempt. Key columns:

- `attempt_id`
- `condition_id`
- `seed`
- `route_id`
- `direction_id`
- `decision`
- `delta_eta_existing_passenger_sec`
- `zero_loss_success`
- `filtered_attention_row_count`
- `edge_overlap_count`
- `node_overlap_count`
- `fallback_row_count`
- `existing_passenger_path_attention_mass`
- `candidate_pickup_path_attention_mass`
- `candidate_dropoff_path_attention_mass`
- `shared_path_attention_mass`
- `unrelated_attention_mass`
- `real_gatv2conv_attention_extracted_all`

### `attention_mass_by_success_path_v2.csv`

Aggregates filtered attention mass by zero-loss success status and path segment.

### `top_filtered_attention_edges_v2.csv`

Top-k filtered GATv2 attention rows by attempt, preserving:

- `path_segment`
- `filter_match_type`
- `attention_weight`
- `src_node`
- `dst_node`
- `layer_id`
- `head_id`

## Interpretation rule

A zero-loss success means:

```text
eta_with_new_pickup_sec - eta_without_new_pickup_sec <= zero_loss_epsilon_sec
```

inside the configured simulator/counterfactual pipeline. It does not mean actual field operation observed zero arrival-time loss.

