# Step 175 — Patent Packet Export Checklist

Project: `urbanbus_rl_project`

Purpose: define what should be exported for attorney review before any patent filing action.

This checklist is **not** a patent filing package and is **not** a legal novelty opinion. It is a technical export-control checklist for the Zero-Loss Pickup Threshold evidence and drafting materials produced in Steps 160–174.

## Export principle

Only stable source documents, generated attorney-review indices, claim drafting matrices, and evidence-mapping materials should be included in the attorney packet.

Generated smoke-test artifacts under `artifacts/**_selftest`, temporary integration outputs, Python cache folders, local `.latest.json` pointers, large model checkpoints, local environment folders, and private credentials must be excluded unless an explicit later release manifest says otherwise.

## Primary include candidates

1. Step 172 final Zero-Loss patent evidence handoff index.
2. Step 173 Zero-Loss patent claim drafting packet.
3. Step 174 patent attorney review packet index.
4. Step 160–171 schemas and source-level runbooks necessary to understand how the evidence pipeline is produced.
5. Project log section documenting Steps 160–168.

## Primary exclusions

1. `artifacts/patent_evidence/*_selftest/**`
2. `artifacts/patent_evidence/*_integration_selftest/**`
3. `.venv/**`, `__pycache__/**`, `.pytest_cache/**`
4. Local `*.latest.json` pointers.
5. Raw credentials, API keys, private database URLs, and machine-local absolute paths.
6. Large model checkpoints unless a later attorney packet explicitly requests them as hash-only references or sealed evidence.

## Non-claim guard

The packet remains locked:

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

## Expected generated outputs

Running `patent_packet_export_checklist_step175.py` creates:

```text
patent_packet_export_checklist_step175.md
patent_packet_export_checklist_step175.json
patent_packet_export_include_index_step175.csv
patent_packet_export_exclusion_index_step175.csv
patent_packet_export_open_questions_step175.csv
patent_packet_export_manifest_step175.json
```
