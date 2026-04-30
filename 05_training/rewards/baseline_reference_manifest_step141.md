# Step 141 B-group baseline reference manifest

This step records B0R/B1/B2 as local baseline references for later comparison with A-family MAPPO outputs.

This step does not execute MAPPO and does not permit causal or paper-level claims.

## Status

- audit_status: `PASS`
- baseline_reference_status: `B_GROUP_BASELINE_REFERENCE_READY`
- ready_baseline_count: `3`
- required_baseline_count: `3`
- causal_comparison_allowed: `False`
- paper_level_claim_allowed: `False`

## Baseline references

| baseline | role | status | root |
|---|---|---|---|
| B0R | historical_replay_reference | READY_AS_NONCAUSAL_BASELINE_REFERENCE | `C:\Users\ryujo\urbanbus_rl_project\artifacts\baseline_v1\B0_historical\canonical_eval` |
| B1 | no_op_reference | READY_AS_NONCAUSAL_BASELINE_REFERENCE | `C:\Users\ryujo\urbanbus_rl_project\artifacts\baseline_v1\B1_noop\canonical_eval` |
| B2 | rule_based_reference | READY_AS_NONCAUSAL_BASELINE_REFERENCE | `C:\Users\ryujo\urbanbus_rl_project\artifacts\baseline_v1\B2_rulebased\canonical_eval` |

## Guard

- B0R/B1/B2 are reference baselines, not reward candidates.
- A-family MAPPO reward ablation remains separate.
- Historical/replay/stub outputs remain non-causal references unless rerun through a validated causal simulator.
