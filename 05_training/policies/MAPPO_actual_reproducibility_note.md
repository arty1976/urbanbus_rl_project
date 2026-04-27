# MAPPO actual execution reproducibility note

## 1. Purpose

This note summarizes the reproducibility and claim-boundary design for the A-family actual MAPPO execution path.

MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) actual results must be separated from smoke, mock, placeholder, and historical replay-only outputs. This document records the guard sequence created in Steps 58-62 and explains how the resulting artifacts should be interpreted in a paper, report, or experiment log.

---

## 2. Scope

This document covers:

```text
Step 58: H200 actual checkpoint preflight
Step 59: A-family actual execution plan
Step 60: actual matrix runner guard
Step 61: final H200 actual MAPPO runbook
Step 62: actual checkpoint arrival workflow
```

This document does not claim that a trained H200 checkpoint already exists. It only defines the reproducible gate structure that must be passed before an actual checkpoint can be used.

---

## 3. Core reproducibility principle

The experiment separates four levels of evidence.

| Level | Meaning | Paper claim allowed |
|---|---|---|
| smoke | Code path and schema validation only | No performance claim |
| actual checkpoint preflight | Checkpoint is structurally valid and trained | Actual checkpoint readiness claim |
| actual policy path dry-run | Plan and command sequence are valid | Execution readiness claim |
| causal simulator evaluation | Actions affect future states | Causal performance claim |

HistoricalReplayAdapter can support contract validation and actual policy path validation, but it cannot support causal performance comparison because actions do not change the next state.

---

## 4. Step 58: H200 actual checkpoint preflight

Step 58 defines the minimum requirements for accepting a checkpoint as actual MAPPO.

Required gates:

```text
checkpoint file exists
trained_model = true
performance_claim_allowed = true
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
reward_version = mappo_reward_v1
energy_proxy_model_version = daegu_energy_proxy_v1
k_dist_kwh_per_m = 0.0012
k_acc_kwh_per_event = 0.1800
k_idle_kwh_per_sec = 0.0080
validate_mappo_checkpoint.py --mode actual PASS
```

If any gate fails, the checkpoint remains blocked.

---

## 5. Step 59: A-family actual execution plan

Step 59 generates the actual execution plan only after a PASS preflight report.

Fixed matrix:

```text
conditions = A, A90, A80, A70
seeds = 1, 2, 3
run_count = 12
policy_source_mode = mappo_actual
checkpoint_validation_mode = actual
```

Blocked cases:

```text
preflight report is BLOCKED
checkpoint path mismatch
non A-family condition included
causal claim requested on HistoricalReplayAdapter
```

---

## 6. Step 60: Matrix runner guard

Step 60 validates that the plan can only be used in a controlled execution sequence.

Required command order per run:

```text
rollout_command
metadata_validation_command
canonical_command
post_canonical_validation_command
```

The dry-run report must include:

```text
run_count
command_count
expected_outputs
actual_claim_guard
causal_claim_guard
```

Important Step 60 fix:

```text
causal_claim_guard.require_causal_claim uses strict OR semantics.
If either the top-level plan or any run-level guard requires causal claim,
the runner treats require_causal_claim as true.
```

This prevents a weaker per-run false value from overriding a stronger top-level causal guard.

---

## 7. Step 61: Final H200 actual MAPPO runbook

Step 61 defines the human operator procedure before actual execution.

Required review before execute:

```text
Step 58 preflight report PASS
Step 59 plan READY_TO_EXECUTE
Step 60 dry-run DRY_RUN_VALIDATED
checkpoint path consistency confirmed
run_count = 12
command_count = 48
A/A90/A80/A70 only
seeds = 1,2,3
Qwen disabled
```

The operator must understand that HistoricalReplayAdapter results are non-causal.

---

## 8. Step 62: Checkpoint arrival workflow

Step 62 links checkpoint arrival to the gate chain.

Required order:

```text
checkpoint existence check
Step 58 actual preflight
preflight PASS check
checkpoint_path consistency check
Step 59 execution plan generation
READY_TO_EXECUTE check
Step 60 dry-run
DRY_RUN_VALIDATED check
arrival workflow report
```

Successful Step 62 status:

```text
arrival_workflow_status = READY_FOR_MANUAL_EXECUTE
preflight_status = PASS
plan_status = READY_TO_EXECUTE
dry_run_status = DRY_RUN_VALIDATED
```

This means manual execution is ready. It does not mean causal performance has been proven.

---

## 9. Policy metadata preservation

The actual pipeline must preserve policy metadata through canonical KPI aggregation.

KPI (Key Performance Indicator=핵심 성과 지표) outputs must retain enough metadata to distinguish actual policy outputs from smoke or placeholder outputs.

Required metadata fields include:

```text
policy_source
policy_source_mode
checkpoint_path
checkpoint_validation_mode
checkpoint_validator_ran
checkpoint_loaded
trained_model
performance_claim_allowed
mock_action_used
placeholder_fallback_used
qwen_train
qwen_inference
qwen_trigger_rate
reward_version
energy_proxy_model_version
actual_policy_claim_ready
causal_policy_claim_ready
```

---

## 10. Correct interpretation language

Allowed:

```text
The A-family actual MAPPO checkpoint arrival workflow passed through dry-run validation.
```

Allowed:

```text
The actual MAPPO policy execution path was validated under the non-causal replay adapter.
```

Not allowed:

```text
The A-family actual MAPPO policy causally outperformed baselines.
```

Not allowed:

```text
Historical replay proves causal performance improvement.
```

Causal performance claims require a Phase 2 causal simulator adapter where policy actions affect future states.

---

## 11. Recommended paper wording

The following wording is safe for a methods or reproducibility section.

```text
Before actual MAPPO evaluation, we introduced a multi-stage execution gate that separates checkpoint validity, execution readiness, and causal performance claims. A checkpoint is accepted only when it passes an actual-mode preflight validating trained_model=true, performance_claim_allowed=true, Qwen-disabled metadata, reward version, energy proxy version, and fixed energy coefficients. The accepted checkpoint is then converted into a fixed A-family execution plan over A, A90, A80, and A70 conditions with seeds 1, 2, and 3. The matrix runner validates command order and metadata preservation in dry-run mode before any actual execution. Results produced under the historical replay adapter are interpreted only as non-causal execution-path validation; causal performance claims are deferred to the future simulator-backed causal adapter.
```

---

## 12. Current status

As of Step 65:

```text
Step 58 = complete
Step 59 = complete
Step 60 = complete
Step 61 = complete
Step 62 = complete
Step 63 = project log update complete
Step 64 = neural strict checkpoint leftovers stashed
```

Remaining local issue:

```text
.claude/worktrees entries may remain modified due to nested gitlink/worktree pointer state.
These are not part of the MAPPO actual execution gate and should not be mixed into research commits.
```

---

## 13. Next recommended work

Next notebook-safe tasks:

```text
1. H200 training runbook draft
2. actual checkpoint folder contract
3. final local validation bundle
4. Phase 2 causal simulator adapter design
```
