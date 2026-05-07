# Step 174 — Patent Attorney Review Packet Index

## Purpose

Step 174 creates a review packet index for a patent attorney based on the Zero-Loss Pickup Threshold evidence pipeline developed in Step 160 through Step 173.

This step does **not** provide a legal opinion, novelty opinion, filing decision, or enforceability opinion. It only organizes the technical materials, evidence pipeline, claim-drafting packet, prior-art differentiation notes, and review questions for attorney review.

## Inputs

Recommended inputs:

- Step 172 final zero-loss patent evidence handoff index
- Step 173 zero-loss patent claim drafting packet
- Step 160 zero-loss pickup evidence reporter
- Step 165 attempt-specific route/path attention filter
- Step 166 patent evidence report v2
- Step 167 actual route-aware rollout evidence runbook
- Step 169 readiness checklist
- Step 170 command packet
- Step 171 release checklist

## Outputs

The generator writes:

- `patent_attorney_review_packet_index_step174.md`
- `patent_attorney_review_packet_index_step174.json`
- `patent_attorney_attachment_index_step174.csv`
- `patent_attorney_review_questions_step174.csv`
- `patent_attorney_review_requested_items_step174.csv`
- `patent_attorney_review_manifest_step174.json`

## Guard policy

The following must remain false:

- `paper_level_claim_allowed`
- `causal_performance_claim_allowed`
- `actual_operational_claim_allowed`
- `actual_evidence_execution_allowed`
- `command_execution_allowed`
- `train_allowed`
- `legal_novelty_opinion_provided`
- `filing_ready_without_attorney_review`

## Intended attorney review questions

1. Whether the independent claim should emphasize a computer-implemented dispatch method, a public transit control system, or a non-transitory computer-readable medium.
2. Whether “zero-loss” should be claimed as exactly zero seconds or as zero within simulator/time-resolution tolerance.
3. Whether the GATv2 attention evidence should be in the independent claim or reserved for dependent claims.
4. Whether the manifest/preflight/reproducibility workflow should be part of the same application or separated into another filing.
5. Whether ride-sharing detour/delay prior art requires narrowing the claim to public-bus route-aware causal simulation and graph-attention evidence.
