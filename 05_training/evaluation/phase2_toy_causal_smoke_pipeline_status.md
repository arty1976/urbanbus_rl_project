# Phase 2 Toy Causal Smoke Pipeline Status Report

## Purpose

This document summarizes the current status of the Phase 2 toy causal smoke pipeline after Steps 77 through 94.

It separates:

```text
what has been structurally validated
what has not yet been validated
what claims are still prohibited
what the next research gate should be
```

This report is not a performance report.

It is a gate/status report for reproducibility and research planning.

## Current Status Summary

The Phase 2 toy causal smoke pipeline is structurally connected.

The following path has passed smoke validation:

```text
CausalSimulatorAdapter
-> 12-KPI reward_v1
-> toy MAPPO smoke training
-> smoke checkpoint
-> checkpoint validator
-> single-condition smoke evaluation
-> A-family smoke matrix evaluation
-> canonical KPI aggregation
-> 12-KPI inspector
```

KPI means Key Performance Indicator=?듭떖?깃낵吏??

MAPPO means Multi-Agent Proximal Policy Optimization=?ㅼ쨷 ?먯씠?꾪듃 洹쇱젒 ?뺤콉 理쒖쟻??

## Completed Gates

| Gate | Step | Status | Meaning |
|---|---:|---|---|
| Causal adapter contract | 77 | PASS | Toy causal adapter follows the SimulatorAdapterInterface contract. |
| Toy rollout writer | 78 | PASS | Toy causal rollout can emit raw event and window rollup outputs. |
| Canonical KPI integration | 79 | PASS | Toy causal output can enter canonical KPI aggregation. |
| A-family smoke matrix | 81 | PASS | A/A90/A80/A70 toy causal matrix can be generated. |
| 12-KPI schema extension | 83 | PASS | Phase 2 official 12-KPI schema is wired. |
| 12-KPI inspector | 84 | PASS | Condition/time/seed summaries and A-baseline deltas can be inspected. |
| Project log for schema/inspector | 85 | PASS | Step 83~84 work is recorded. |
| Reward v1 contract | 86 | PASS | 12-KPI MAPPO reward contract is defined and tested. |
| Adapter reward wiring | 87 | PASS | CausalSimulatorAdapter.step emits reward_v1 team reward and debug info. |
| Toy MAPPO training smoke | 88 | PASS | Tiny actor/critic rollout/update/checkpoint smoke path works. |
| Smoke checkpoint validator | 89 | PASS | Smoke checkpoint contract is independently validated. |
| Project log for reward/training | 90 | PASS | Step 86~89 work is recorded. |
| Single-condition checkpoint evaluation | 91 | PASS | Smoke checkpoint can be evaluated through rollout/canonical/inspector. |
| Train/eval runbook | 92 | PASS | Phase 2 toy train/eval runbook is recorded. |
| A-family matrix smoke evaluation | 93 | PASS | A/A90/A80/A70 smoke evaluation matrix passes. |
| Project log/runbook for matrix | 94 | PASS | Step 93 work is recorded. |

## Validated Technical Path

The following structural path is now validated:

```text
1. Environment reset
2. Agent action sampling
3. Toy causal step dynamics
4. 12-KPI reward metrics
5. mappo_reward_v1 reward_total
6. Team-level reward broadcast
7. Tiny MAPPO-style policy/value update
8. Smoke checkpoint save/load
9. Checkpoint contract validation
10. Evaluation rollout from checkpoint
11. Raw event and window rollup generation
12. Canonical KPI aggregation
13. A-family 12-KPI inspection
14. Non-claim metadata propagation
```

## Validated Non-Claim Flags

All smoke artifacts must keep:

```text
trained_model = false
performance_claim_allowed = false
smoke_training_only = true       # training/checkpoint artifacts
smoke_evaluation_only = true     # evaluation artifacts
matrix_smoke_evaluation_only = true # matrix evaluation artifacts
```

These flags are not optional. They are part of the scientific boundary of the pipeline.

## What Has Been Validated

The following statements are currently allowed:

```text
The Phase 2 toy causal adapter satisfies the local interface contract.
The 12-KPI schema can flow through toy causal rollout, canonical aggregation, and inspection.
The 12-KPI reward contract can compute a finite reward_total and reward_debug payload.
The adapter can broadcast reward_v1 as a team-level reward.
A toy MAPPO-style smoke training loop can execute one tiny rollout/update path.
The smoke checkpoint can be serialized, validated, reloaded, and evaluated.
The A/A90/A80/A70 smoke matrix can pass through canonical aggregation and A-family inspection.
```

## What Has Not Been Validated

The following statements are not yet validated:

```text
MAPPO policy convergence
Real operational improvement in Daegu
Generalization across full-year demand
Full 4,000-stop network performance
Real bus dispatch feasibility
Real passenger waiting-time distribution
Real traffic congestion response
Actual H200 training performance
Paper-level performance comparison
```

## Prohibited Claims

The current outputs must not be used to claim:

```text
A70 is better than A in real operation.
A90/A80/A70 produce real fleet reduction benefits.
The MAPPO policy has learned a deployable policy.
The smoke checkpoint is a trained model.
The result is suitable for a paper performance table.
The toy simulator represents the full Daegu transit system.
```

## Current Scientific Interpretation

The correct interpretation is:

```text
The Phase 2 toy causal smoke pipeline is structurally wired, reproducible, and guarded by non-claim metadata.
```

This means the system is ready for the next design gate, not ready for performance claims.

## Current Artifacts and Source Files

Main source files:

```text
05_training/adapters/causal_simulator_adapter.py
05_training/rewards/mappo_reward_v1.py
05_training/train_toy_causal_mappo_smoke.py
05_training/policies/validate_toy_mappo_smoke_checkpoint.py
05_training/evaluation/evaluate_toy_mappo_smoke_checkpoint.py
05_training/evaluation/evaluate_toy_mappo_smoke_matrix.py
05_training/evaluation/inspect_toy_causal_a_family_kpis.py
```

Main documentation files:

```text
05_training/rewards/README_reward_contract.md
05_training/adapters/causal_simulator_adapter_contract.md
05_training/adapters/toy_causal_mappo_training_smoke.md
05_training/policies/toy_mappo_smoke_checkpoint_contract.md
05_training/evaluation/toy_mappo_smoke_checkpoint_evaluation.md
05_training/evaluation/toy_mappo_smoke_matrix_evaluation.md
05_training/evaluation/phase2_toy_causal_train_eval_runbook.md
project_log.md
```

## Recommended Next Gate

The next recommended gate is Step 96:

```text
Toy causal simulator realism gap analysis
```

Reason:

```text
Before moving from toy smoke validation to H200 actual training or larger causal simulation, the gap between the toy dynamics and real Daegu bus operations must be explicitly documented.
```

## Decision After Step 96

After Step 96, choose one of the following:

```text
Path A: Causal simulator v2 contract
Path B: H200 actual MAPPO reward_v1 integration
Path C: Paper/proposal documentation update
```

Recommended order:

```text
Step 96: realism gap analysis
Step 97: causal simulator v2 contract
Step 98: H200 actual integration gate
```
