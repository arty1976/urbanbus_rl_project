# A-family MAPPO Actual Execution Plan v1

## 1. Purpose

This document defines the execution plan for running A/A90/A80/A70 with `mappo_actual`.

This step does not train MAPPO.

This step does not execute the actual matrix.

This step creates the guarded plan that will be used only after an H200-trained checkpoint passes actual checkpoint preflight.

## 2. Acronyms

- MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화)
- KPI (Key Performance Indicator=핵심성과지표)
- H200 (NVIDIA H200 GPU server=엔비디아 H200 그래픽처리장치 서버)

## 3. Required inputs

The actual execution plan requires:

```text
checkpoint_path
h200_actual_checkpoint_preflight_report.json
conditions = A,A90,A80,A70
seeds = 1,2,3 or approved seed set
policy_source_mode = mappo_actual
checkpoint_validation_mode = actual
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
```

## 4. Preflight report acceptance rules

The plan can be created only if the H200 actual checkpoint preflight report says:

```text
status = PASS
actual_checkpoint_ready = true
actual_policy_claim_ready_candidate = true
checkpoint file exists
trained_model = true
performance_claim_allowed = true
validate_mappo_checkpoint.py --mode actual PASS
```

If any of these are false, the plan must be BLOCKED.

## 5. Actual policy claim rule

When the actual matrix is eventually executed, each rollout row must satisfy:

```text
policy_source = mappo_policy
policy_source_mode = mappo_actual
checkpoint_validation_mode = actual
checkpoint_validator_ran = true
checkpoint_loaded = true
trained_model = true
performance_claim_allowed = true
mock_action_used = false
placeholder_fallback_used = false
actual_policy_claim_ready = true
```

## 6. Causal claim rule

Actual checkpoint execution and causal performance claim are different.

If the simulator adapter is HistoricalReplayAdapter or any replay adapter, then:

```text
actual policy path may run
causal performance claim is not allowed
```

Causal claim requires a future causal simulator adapter.

## 7. Target execution sequence

After H200 checkpoint preflight passes:

```text
1. Load h200_actual_preflight_report.json
2. Validate checkpoint_path matches report checkpoint_path
3. Build A/A90/A80/A70 x seed execution matrix
4. For each condition and seed:
   run_a_family_policy_rollout_smoke.py
     --policy-source-mode mappo_actual
     --checkpoint-validation-mode actual
5. Validate every window_rollup.parquet with validate_window_rollup_policy_metadata.py
6. Run canonical_kpi_aggregator.py official_rollup
7. Validate post-canonical kpi_by_window.parquet metadata
8. Write actual_execution_manifest.json
```

## 8. Step boundary

Step 59 creates the guarded execution plan.

Step 60 may implement the actual matrix runner.

Step 61 may write the H200 training/runbook.

Step 62 may run the real H200 checkpoint through the actual matrix after checkpoint is available.
