# Step 171 — Actual Evidence Execution Release Checklist

## Purpose

Step 171 defines the operator approval gate for promoting a Step 170 dry-run command packet into an executable Zero-Loss evidence run.

This step **does not** create paper-level, causal performance, or actual operational claims.

It only answers:

> Are the actual-evidence input readiness manifest and dry-run command packet sufficiently validated, and has an operator explicitly approved execution?

## Scope

Input candidates:

- Step 169 readiness manifest
- Step 170 dry-run command packet manifest
- operator approval decision
- release manifest commit/push status

Output:

- `actual_evidence_execution_release_checklist_manifest.json`
- `actual_evidence_execution_release_checklist.md`

## Release condition

`actual_evidence_execution_allowed = true` only if all conditions hold:

1. Step 169 readiness manifest has `ready_for_actual_like_execution = true`
2. Step 170 command packet has `audit_status = PASS`
3. Step 170 command packet has `dry_run_only = true`
4. Step 170 command packet has `command_count >= 6`
5. Step 170 command packet has `command_execution_allowed = false`
6. Operator approval is recorded
7. `operator_approval_granted = true`
8. Release manifest is recorded as committed/pushed
9. hard failures are zero

Even if execution is released, claim guards remain locked:

```text
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
actual_operational_claim_allowed = false
```

## Non-claim policy

Step 171 releases an evidence-generation run only. It does not release scientific claims.

Actual claim release requires later review of produced Step 166 report, source manifests, run hashes, checkpoint provenance, and legal/patent review.
