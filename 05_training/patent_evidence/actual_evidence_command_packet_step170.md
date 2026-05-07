# Step 170 Actual Evidence Command Packet

## Purpose

Step 170 creates a dry-run command packet for executing the Zero-Loss Pickup evidence chain with actual route-aware rollout inputs.

The packet fixes this execution order:

1. Step 162 — route-aware rollout adapter
2. Step 161 — pickup attempt / ETA counterfactual writer
3. Step 163 — GATv2 real attention extractor
4. Step 165 — attempt-specific route/path attention filter
5. Step 160 — Zero-Loss Pickup evidence reporter
6. Step 166 — Patent evidence report v2

## Non-claim guard

This step does not execute an actual evidence run by default. It generates a dry-run operator packet only.

The following guards must remain false:

- `actual_evidence_execution_allowed`
- `command_execution_allowed`
- `paper_level_claim_allowed`
- `causal_performance_claim_allowed`
- `actual_operational_claim_allowed`
- `train_allowed`
- `winner_selected`

## Why this is needed

Steps 160–166 created the evidence-generation components. Step 170 prevents ad-hoc manual command assembly by generating a reproducible command packet from a Step 169 readiness manifest.

## Output

- `actual_evidence_command_packet_step170.md`
- `actual_evidence_command_sequence_step170.json`
- `actual_evidence_command_packet_step170.ps1`
- `actual_evidence_command_packet_manifest_step170.json`
