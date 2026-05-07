# Step 173 — Zero-Loss Patent Claim Drafting Packet

## Purpose

This step creates a **technical drafting packet** for the Zero-Loss Pickup Threshold patent concept.
It is not a legal opinion, not a novelty opinion, and not a permission to make paper-level or operational claims.

The packet translates the Step 160~172 evidence pipeline into patent-drafting material:

- technical problem statement,
- proposed inventive concept,
- independent claim draft candidates,
- dependent claim draft candidates,
- evidence-to-claim mapping,
- prior-art differentiation matrix,
- non-claim and legal-review guard status.

## Scope

The drafting packet focuses on the first patent family:

```text
Zero-Loss Pickup Threshold for ride-sharing / public-bus-like route-aware dispatch.
```

The technical core is:

```text
A candidate pickup is accepted only when the counterfactual ETA loss of already-boarded passenger(s)
is zero or within a configured simulator-resolution epsilon, and the decision is accompanied by a
GATv2 attention evidence bundle mapped to the route/path of the candidate pickup.
```

## Inputs expected from previous steps

The packet assumes the following pipeline exists:

```text
Step 160  Zero-Loss Pickup Evidence Reporter
Step 161  Route-Aware Pickup Attempt Event Writer
Step 162  Route-Aware Rollout Adapter
Step 163  GATv2 Real Attention Extractor
Step 164  Real Attention Pipeline Connector
Step 165  Attempt-Specific Route/Path Attention Filter
Step 166  Patent Evidence Report v2
Step 167  Actual Route-Aware Rollout Evidence Runbook
Step 168  Project Log Update
Step 169  Actual Evidence Input Readiness Checklist
Step 170  Actual Evidence Command Packet
Step 171  Actual Evidence Execution Release Checklist
Step 172  Final Zero-Loss Patent Evidence Handoff Index
```

## Non-claim guard

This step must keep the following locked:

```text
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
actual_operational_claim_allowed = false
actual_evidence_execution_allowed = false
command_execution_allowed = false
train_allowed = false
legal_novelty_opinion_provided = false
filing_ready_without_attorney_review = false
```

## Output files

The generator writes:

```text
zero_loss_patent_claim_drafting_packet_step173.md
zero_loss_patent_claim_drafting_packet_step173.json
zero_loss_claim_element_matrix_step173.csv
zero_loss_dependent_claim_candidates_step173.csv
zero_loss_evidence_to_claim_mapping_step173.csv
zero_loss_prior_art_differentiation_matrix_step173.csv
zero_loss_claim_drafting_manifest_step173.json
```

## Important warning

This packet is useful for organizing technical disclosure, but a registered patent attorney should still perform:

- claim charting,
- KIPRIS / Google Patents / WIPO / USPTO / EPO prior-art search,
- novelty / inventive-step review,
- jurisdiction-specific claim revision,
- final filing strategy.
