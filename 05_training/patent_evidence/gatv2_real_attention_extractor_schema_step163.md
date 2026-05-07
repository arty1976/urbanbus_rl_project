# Step 163 — GATv2 Real Attention Extractor Contract

## Purpose

Step 163 defines the contract for extracting **real GATv2Conv attention weights** from a PyTorch Geometric GATv2 encoder path and converting them into the `gatv2_attention` evidence table used by the Zero-Loss Pickup patent evidence pipeline.

This step upgrades the Step 161 placeholder/proxy attention rows into a reproducible extraction contract:

```text
PyG Data snapshot / sample graph
→ GATv2Conv(return_attention_weights=True)
→ snapshot-level attention edge table
→ attempt-level gatv2_attention table compatible with Step 160
```

## Non-claim boundary

Step 163 is an attention extraction contract and smoke validator. It does **not** make the following claims:

```text
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
actual_operational_claim_allowed = false
trained_model_claim_allowed = false
train_allowed = false
```

Even when real `GATv2Conv` attention weights are extracted, the result is still non-claim evidence until it is tied to an approved trained checkpoint, actual route-aware rollout, and validated evidence manifest.

## Terms

- GATv2 (Graph Attention Network v2=그래프 어텐션 네트워크 v2)
- PyG (PyTorch Geometric=파이토치 지오메트릭)
- ETA (Estimated Time of Arrival=예상 도착 시간)
- KPI (Key Performance Indicator=핵심 성과 지표)

## Inputs

### Mode: `sample`

Creates a deterministic small PyG graph and sample pickup attempts for contract validation.

Required:

```text
--mode sample
--output-root <path>
```

Optional:

```text
--condition-id A
--seed 1
--top-k-per-attempt 12
--file-format csv|parquet
```

### Mode: `from-pyg-pt`

Loads one PyG `Data` snapshot and extracts attention weights.

Required:

```text
--mode from-pyg-pt
--pyg-pt <path-to-data.pt>
--output-root <path>
```

Optional:

```text
--checkpoint <path-to-checkpoint.pt>
--attempt-events <pickup_attempt_events.csv|parquet>
--state-ts <timestamp>
--top-k-per-attempt 32
--file-format csv|parquet
```

## Required output files

```text
gatv2_real_attention_edges.csv/parquet
  Snapshot-level real attention rows extracted from GATv2Conv.

gatv2_attention.csv/parquet
  Attempt-level Step 160/161 compatible attention evidence rows.
  Generated when attempt events are supplied or when sample mode is used.

attention_quality_report.json
  Attention extraction quality summary.

gatv2_real_attention_extractor_manifest.json
  Hashes, counts, guard flags, and extraction status.
```

## Snapshot-level required columns

`gatv2_real_attention_edges` must include:

```text
state_ts
layer_id
head_id
edge_rank
src_node
dst_node
src_idx
dst_idx
attention_weight
attention_source
real_gatv2conv_attention_extracted
```

Recommended optional columns:

```text
edge_id
distance_m
time_sec
generalized_cost
long_edge_5km_flag
is_self_loop
```

## Step 160-compatible attempt-level required columns

`gatv2_attention` must include:

```text
attempt_id
state_ts
layer_id
head_id
src_node
dst_node
attention_weight
```

Recommended optional columns:

```text
edge_id
path_segment
route_id
direction_id
distance_m
time_sec
generalized_cost
attention_source
real_gatv2conv_attention_extracted
```

## Quality rules

The validator checks:

1. Manifest exists.
2. Non-claim flags remain locked.
3. Snapshot attention output exists and is non-empty.
4. Required columns exist.
5. `attention_weight` is finite and non-negative.
6. If `--require-real-gatv2conv` is used, `real_gatv2conv_attention_extracted=true` must be present in the manifest.
7. If attempt-level output exists, it must satisfy Step 160-compatible attention columns.

## Why Step 163 matters for the patent evidence

Step 160 and Step 161 already prove that Zero-Loss Pickup evidence can be summarized from:

```text
pickup_attempt_events
eta_counterfactual
gatv2_attention
```

But Step 161 can generate proxy attention rows for smoke testing. Step 163 creates the path that replaces proxy attention with **actual GATv2Conv attention weights**. This is essential for the patent evidence sentence:

```text
The zero-loss pickup decision is supported by graph-attention weights extracted from the GATv2 encoder over the route-aware urban bus graph.
```

## Next step

Recommended next step:

```text
Step 164 — Attach real GATv2 attention extractor to Step 161/160 evidence bundle
```

Step 164 should connect actual route-aware rollout attempts, real PyG snapshots, and Step 163 attention output into one evidence bundle.
