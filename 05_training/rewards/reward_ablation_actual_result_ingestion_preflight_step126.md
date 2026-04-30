# Step 126 ??Reward Ablation Actual Result Ingestion Preflight

Step 126 defines the preflight rules for future actual reward ablation result bundles.

This step does not ingest actual results, does not select a winner, and does not allow training.

Required future matrix:

```text
R0/R1/R2/R3/R4/R5 횞 A/A90/A80/A70 횞 seeds 1,2,3 = 72 rows
```

The future actual bundle must contain actual result detail CSV, summary CSV, hard-constraint violations CSV, canonical evaluation manifest, checkpoint validation manifest, and bundle manifest.

Current guards remain:

- `ingestion_allowed_now = false`
- `actual_results_ingested = false`
- `winner_selected = false`
- `train_with_this_reward_allowed = false`
- `paper_level_claim_allowed = false`
- `causal_performance_claim_allowed = false`
