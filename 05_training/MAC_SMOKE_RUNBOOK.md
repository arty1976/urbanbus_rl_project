# Mac Smoke Runbook

This runbook covers the local Mac sequence after the Windows transfer package has been copied and verified.

## Current roots

- Source repository: `/Users/arty/Documents/Codex/urbanbus_rl_project`
- Verified transfer package: `/Users/arty/urbanbus_transfer_20260716_153509`
- GATv2 tensor artifact: `/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/dataset_full_20260422_084243`

`/Users/arty/urbanbus_rl_project` does not currently exist, so commands below use the cloned source repository path.

## Environment

```bash
cd /Users/arty/Documents/Codex/urbanbus_rl_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-mac-smoke.in
```

Minimum dependency groups:

- Tensor loading and GATv2 smoke: `torch`, `torch-geometric`
- Dataset rebuild from PostgreSQL: `pandas`, `sqlalchemy`, `psycopg2-binary`
- MAPPO toy smoke: `torch`, `numpy`
- Parquet-backed replay utilities: `pandas`, `pyarrow`

## Tensor availability

The full GATv2 PyG tensor artifact has been copied into the project.

```bash
find /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/dataset_full_20260422_084243 -name '*.pt' | wc -l
```

Expected count: `6570`.

## Rebuild tensors from database

Required source files:

- `/Users/arty/Documents/Codex/urbanbus_rl_project/04_model_inputs/create_gatv2_training_views.sql`
- `/Users/arty/Documents/Codex/urbanbus_rl_project/04_model_inputs/materialize_gatv2_training_table.sql`
- `/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/build_gatv2_dataset.py`

Example build command after `URBANBUS_DB_URL` is configured:

```bash
cd /Users/arty/Documents/Codex/urbanbus_rl_project/05_training
python build_gatv2_dataset.py \
  --max-snapshots 8 \
  --out-dir /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_gatv2_smoke
```

## Smoke sequence

Set `DATASET_DIR` to a directory containing `.pt` files or `train/val/test` subdirectories.

```bash
cd /Users/arty/Documents/Codex/urbanbus_rl_project
source .venv/bin/activate
export DATASET_DIR=/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/dataset_full_20260422_084243
```

Tensor sample loading:

```bash
python 05_training/smoke_gatv2_mac.py --dataset-dir "$DATASET_DIR" --split train --mode load --max-files 2
```

CPU GATv2 forward:

```bash
python 05_training/smoke_gatv2_mac.py --dataset-dir "$DATASET_DIR" --split train --mode forward --device cpu --max-files 2
```

MPS GATv2 forward/backward:

```bash
python 05_training/smoke_gatv2_mac.py --dataset-dir "$DATASET_DIR" --split train --mode backward --device cpu --max-files 2
```

One epoch GATv2 training:

```bash
python 05_training/smoke_gatv2_mac.py --dataset-dir "$DATASET_DIR" --split train --mode train --device cpu --epochs 1 --max-files 8
```

Short MAPPO rollout:

```bash
python 05_training/train_toy_causal_mappo_smoke.py \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_toy_mappo_smoke \
  --steps 4 \
  --epochs 1 \
  --device cpu \
  --clean
```

Tiny GATv2-MAPPO integration smoke:

```bash
python 05_training/smoke_gatv2_mappo_integration.py --dataset-dir "$DATASET_DIR" --split train --device cpu --max-files 2
```

Full GATv2 -> embedding -> MAPPO -> checkpoint -> KPI pipeline smoke:

```bash
python 05_training/smoke_full_gatv2_mappo_pipeline_mac.py \
  --dataset-dir "$DATASET_DIR" \
  --split train \
  --snapshots 128 \
  --gat-batch-size 1 \
  --gat-hidden 32 \
  --rollout-horizon 64 \
  --ppo-update-epochs 2 \
  --minibatch-size 128 \
  --num-agents 8 \
  --seed 1 \
  --condition-id A \
  --preferred-device mps \
  --clean
```

512-snapshot MPS validation:

```bash
python 05_training/smoke_full_gatv2_mappo_pipeline_mac.py \
  --dataset-dir "$DATASET_DIR" \
  --split train \
  --val-split val \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_512_snapshots \
  --snapshots 512 \
  --val-snapshots 64 \
  --gat-batch-size 1 \
  --gat-hidden 32 \
  --gat-epochs 1 \
  --rollout-horizon 64 \
  --ppo-update-epochs 2 \
  --minibatch-size 128 \
  --num-agents 8 \
  --seed 1 \
  --condition-id A \
  --preferred-device mps \
  --clean
```

Verified result path:

```text
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_512_snapshots/pipeline_manifest.json
```

512-snapshot / horizon-128 strict-MPS validation:

```bash
python 05_training/smoke_full_gatv2_mappo_pipeline_mac.py \
  --dataset-dir "$DATASET_DIR" \
  --split train \
  --val-split val \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_512_h128 \
  --snapshots 512 \
  --val-snapshots 64 \
  --gat-batch-size 1 \
  --gat-hidden 32 \
  --gat-epochs 1 \
  --gat-grad-clip-norm 5.0 \
  --rollout-horizon 128 \
  --ppo-update-epochs 2 \
  --minibatch-size 128 \
  --mappo-grad-clip-norm 0.5 \
  --ppo-clip-epsilon 0.2 \
  --num-agents 8 \
  --seed 1 \
  --condition-id A \
  --preferred-device mps \
  --require-gpu \
  --clean
```

Verified result path:

```text
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_512_h128/pipeline_manifest.json
```

512-snapshot / 16-agent / horizon-128 strict-MPS validation:

```bash
python 05_training/smoke_full_gatv2_mappo_pipeline_mac.py \
  --dataset-dir "$DATASET_DIR" \
  --split train \
  --val-split val \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_512_a16_h128 \
  --snapshots 512 \
  --val-snapshots 64 \
  --gat-batch-size 1 \
  --gat-hidden 32 \
  --gat-epochs 1 \
  --gat-grad-clip-norm 5.0 \
  --rollout-horizon 128 \
  --ppo-update-epochs 2 \
  --minibatch-size 128 \
  --mappo-grad-clip-norm 0.5 \
  --ppo-clip-epsilon 0.2 \
  --num-agents 16 \
  --seed 1 \
  --condition-id A \
  --preferred-device mps \
  --require-gpu \
  --clean
```

Verified result path:

```text
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_512_a16_h128/pipeline_manifest.json
```

1000-snapshot / 16-agent / horizon-128 strict-MPS validation:

```bash
python 05_training/smoke_full_gatv2_mappo_pipeline_mac.py \
  --dataset-dir "$DATASET_DIR" \
  --split train \
  --val-split val \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a16_h128 \
  --snapshots 1000 \
  --val-snapshots 100 \
  --gat-batch-size 1 \
  --gat-hidden 32 \
  --gat-epochs 1 \
  --gat-grad-clip-norm 5.0 \
  --rollout-horizon 128 \
  --ppo-update-epochs 2 \
  --minibatch-size 128 \
  --mappo-grad-clip-norm 0.5 \
  --ppo-clip-epsilon 0.2 \
  --num-agents 16 \
  --seed 1 \
  --condition-id A \
  --preferred-device mps \
  --require-gpu \
  --clean
```

Verified result path:

```text
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a16_h128/pipeline_manifest.json
```

1000-snapshot / 32-agent / horizon-128 strict-MPS validation:

```bash
python 05_training/smoke_full_gatv2_mappo_pipeline_mac.py \
  --dataset-dir "$DATASET_DIR" \
  --split train \
  --val-split val \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a32_h128 \
  --snapshots 1000 \
  --val-snapshots 100 \
  --gat-batch-size 1 \
  --gat-hidden 32 \
  --gat-epochs 1 \
  --gat-grad-clip-norm 5.0 \
  --rollout-horizon 128 \
  --ppo-update-epochs 2 \
  --minibatch-size 128 \
  --mappo-grad-clip-norm 0.5 \
  --ppo-clip-epsilon 0.2 \
  --num-agents 32 \
  --seed 1 \
  --condition-id A \
  --preferred-device mps \
  --require-gpu \
  --clean
```

Verified result path:

```text
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a32_h128/pipeline_manifest.json
```

32-agent actor-loss audit:

```text
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a32_h128/actor_loss_audit.json
```

PASS criteria:

- `actual_device`/selected device is `mps`.
- `gpu_blocked = false` and `cpu_fallback_used = false`.
- `observed_unique_agent_count = 32`.
- `expected_agent_decisions = observed_agent_decisions = 4096`.
- `policy_logits_shape = [128, 32, 5]`.
- `policy_actions_shape = actor_ratio_shape = actor_surrogate_shape = [128, 32]`.
- `actor_ratio_numel = actor_surrogate_numel = actor_entropy_numel = 4096`.
- `team_advantage_count_before_broadcast = 128`.
- `actor_advantage_count_after_broadcast = 4096`.
- `actor_loss_reduction_axes = ["time", "agent"]`.
- `critic_loss_reduction_axes = ["time"]`.
- `checkpoint_reload_ok = true`.
- `checkpoint_reload_logits_max_abs_diff = 0.0`.
- `checkpoint_reload_value_max_abs_diff = 0.0`.
- `canonical_kpi_count = 12` and `canonical_kpi_missing = []`.

1000-snapshot / 64-agent / horizon-128 strict-MPS validation:

```bash
unset PYTORCH_ENABLE_MPS_FALLBACK
python 05_training/smoke_full_gatv2_mappo_pipeline_mac.py \
  --dataset-dir "$DATASET_DIR" \
  --split train \
  --val-split val \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128 \
  --snapshots 1000 \
  --val-snapshots 100 \
  --gat-batch-size 1 \
  --gat-hidden 32 \
  --gat-epochs 1 \
  --gat-grad-clip-norm 5.0 \
  --rollout-horizon 128 \
  --ppo-update-epochs 2 \
  --minibatch-size 128 \
  --mappo-grad-clip-norm 0.5 \
  --ppo-clip-epsilon 0.2 \
  --num-agents 64 \
  --seed 1 \
  --condition-id A \
  --preferred-device mps \
  --require-gpu \
  --clean
```

Verified result path:

```text
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128/pipeline_manifest.json
```

64-agent audit checks:

```bash
jq '.actual_device, .strict_mps, .cpu_fallback_used, .mappo.expected_agent_decisions, .mappo.observed_agent_decisions, .mappo.actor_loss_audit.actor_ratio_numel, .mappo.actor_loss_audit.actor_elements_processed_across_all_ppo_epochs' \
  /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128/pipeline_manifest.json

jq '.rollout_total_seconds, .rollout_trace_recording_seconds, .actor_inference_mode, .trace_rows_written' \
  /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128/rollout_profile.json

jq '.summary' \
  /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128/memory_profile.json

jq '.mappo.checkpoint_reload_ok, .mappo.checkpoint_reload_logits_max_abs_diff, .mappo.checkpoint_reload_value_max_abs_diff, .canonical_kpi_count, .canonical_kpi_missing' \
  /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128/pipeline_manifest.json
```

PASS criteria:

- `actual_device = "mps"`.
- `strict_mps = true`.
- `cpu_fallback_allowed = false` and `cpu_fallback_used = false`.
- `gpu_blocked = false`.
- `configured_num_agents = observed_unique_agent_count = 64`.
- `unique_agent_ids = [0, ..., 63]`.
- `expected_agent_decisions = observed_agent_decisions = 8192`.
- `policy_logits_shape = [128, 64, 5]`.
- `policy_actions_shape = actor_ratio_shape = actor_surrogate_shape = [128, 64]`.
- `policy_action_element_count = policy_old_logprob_element_count = policy_new_logprob_element_count = policy_entropy_element_count = 8192`.
- `actor_ratio_numel = actor_surrogate_numel = actor_entropy_numel = 8192`.
- `unique_rollout_actor_elements = 8192`.
- `actor_elements_processed_across_all_ppo_epochs = 16384`.
- `team_advantage_count_before_broadcast = 128`.
- `actor_advantage_count_after_broadcast = 8192`.
- `critic_value_target_count = 128`.
- `actor_loss_reduction_axes = ["time", "agent"]`.
- `critic_loss_reduction_axes = ["time"]`.
- `nan_detected = false` and `inf_detected = false`.
- `checkpoint_reload_ok = true`.
- `checkpoint_reload_logits_max_abs_diff = 0.0`.
- `checkpoint_reload_value_max_abs_diff = 0.0`.
- `canonical_kpi_count = 12` and `canonical_kpi_missing = []`.
- `swap_used_mb_max = 0.0`.

64-agent AUDIT/BENCHMARK trace-mode split:

Use AUDIT mode for contract and regression checks. It keeps the full agent decision CSV.

```bash
unset PYTORCH_ENABLE_MPS_FALLBACK
python 05_training/smoke_full_gatv2_mappo_pipeline_mac.py \
  --dataset-dir "$DATASET_DIR" \
  --split train \
  --val-split val \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_audit \
  --snapshots 1000 \
  --val-snapshots 100 \
  --gat-batch-size 1 \
  --gat-hidden 32 \
  --gat-epochs 1 \
  --gat-grad-clip-norm 5.0 \
  --rollout-horizon 128 \
  --ppo-update-epochs 2 \
  --minibatch-size 128 \
  --mappo-grad-clip-norm 0.5 \
  --ppo-clip-epsilon 0.2 \
  --num-agents 64 \
  --seed 1 \
  --condition-id A \
  --preferred-device mps \
  --trace-mode audit \
  --require-gpu \
  --clean
```

Use BENCHMARK mode for scale/performance measurement. It does not write the 8,192-row `agent_decision_trace.csv`; it writes tensor hashes, tensor summary, and sampled trace JSON.

```bash
unset PYTORCH_ENABLE_MPS_FALLBACK
python 05_training/smoke_full_gatv2_mappo_pipeline_mac.py \
  --dataset-dir "$DATASET_DIR" \
  --split train \
  --val-split val \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_benchmark_run1 \
  --snapshots 1000 \
  --val-snapshots 100 \
  --gat-batch-size 1 \
  --gat-hidden 32 \
  --gat-epochs 1 \
  --gat-grad-clip-norm 5.0 \
  --rollout-horizon 128 \
  --ppo-update-epochs 2 \
  --minibatch-size 128 \
  --mappo-grad-clip-norm 0.5 \
  --ppo-clip-epsilon 0.2 \
  --num-agents 64 \
  --seed 1 \
  --condition-id A \
  --preferred-device mps \
  --trace-mode benchmark \
  --require-gpu \
  --clean
```

Mode comparison:

```bash
python 05_training/compare_gatv2_mappo_trace_modes.py \
  --audit-manifest /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_audit/pipeline_manifest.json \
  --benchmark-manifest /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_benchmark_run1/pipeline_manifest.json \
  --benchmark-manifest /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_benchmark_run2/pipeline_manifest.json \
  --previous-manifest /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128/pipeline_manifest.json \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_mode_comparison
```

Observed full end-to-end result:

- AUDIT and BENCHMARK both passed Strict-MPS execution.
- BENCHMARK reduced rollout from `5.401755s` to `0.880265s`.
- BENCHMARK decision throughput improved from `1516.54/s` to `9306.29/s`.
- Full separate-process tensor equivalence did not pass because MPS GATv2 train/export produced different embedding hashes across runs even with the same seed and identical initial parameters.

Fixed-embedding trace-mode control:

```bash
python 05_training/smoke_full_gatv2_mappo_pipeline_mac.py \
  --dataset-dir "$DATASET_DIR" \
  --split train \
  --val-split val \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_benchmark_fixed_embedding \
  --snapshots 1000 \
  --val-snapshots 100 \
  --gat-batch-size 1 \
  --gat-hidden 32 \
  --gat-epochs 1 \
  --gat-grad-clip-norm 5.0 \
  --precomputed-embedding-path /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_audit/embeddings/gatv2_graph_embeddings.pt \
  --rollout-horizon 128 \
  --ppo-update-epochs 2 \
  --minibatch-size 128 \
  --mappo-grad-clip-norm 0.5 \
  --ppo-clip-epsilon 0.2 \
  --num-agents 64 \
  --seed 1 \
  --condition-id A \
  --preferred-device mps \
  --trace-mode benchmark \
  --require-gpu \
  --clean
```

Fixed-embedding comparison path:

```text
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_mode_comparison_fixed_embedding/mode_comparison.json
```

Fixed-embedding PASS values:

- `embedding_hash_match = true`.
- `action_hash_match = true`.
- `old_logprob_hash_match = true`.
- `reward_hash_match = true`.
- `done_hash_match = true`.
- `final_actor_state_match = true`.
- `final_critic_state_match = true`.
- `canonical_kpi_match = true`.
- actor/critic/KPI absolute diffs are `0.0`.
- rollout speedup is `5.911x`.
- trace overhead is `4.500544s`, or `83.08%` of AUDIT rollout time.

Operational rule:

- Contract and regression validation: use `--trace-mode audit`.
- Performance and scale validation: use `--trace-mode benchmark`.
- 128-agent and larger tests should default to BENCHMARK mode, with AUDIT mode reserved for milestone contract checks.

1000-snapshot / 128-agent / horizon-128 fixed-embedding BENCHMARK validation:

This validates MAPPO agent scaling with a canonical fixed embedding. It does not validate fresh GATv2 cross-process determinism.

```bash
unset PYTORCH_ENABLE_MPS_FALLBACK
python 05_training/smoke_full_gatv2_mappo_pipeline_mac.py \
  --dataset-dir "$DATASET_DIR" \
  --split train \
  --val-split val \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_fixed_embedding_mappo_1000_a128_h128_benchmark \
  --snapshots 1000 \
  --val-snapshots 100 \
  --gat-batch-size 1 \
  --gat-hidden 32 \
  --gat-epochs 1 \
  --gat-grad-clip-norm 5.0 \
  --precomputed-embedding-path /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_audit/embeddings/gatv2_graph_embeddings.pt \
  --rollout-horizon 128 \
  --ppo-update-epochs 2 \
  --minibatch-size 128 \
  --mappo-grad-clip-norm 0.5 \
  --ppo-clip-epsilon 0.2 \
  --num-agents 128 \
  --seed 1 \
  --condition-id A \
  --preferred-device mps \
  --trace-mode benchmark \
  --require-gpu \
  --clean
```

Verified result path:

```text
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_fixed_embedding_mappo_1000_a128_h128_benchmark/pipeline_manifest.json
```

64-vs-128 scaling comparison:

```bash
python 05_training/compare_mappo_agent_scaling.py \
  --source-comparison /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_mode_comparison_fixed_embedding/mode_comparison.json \
  --source-manifest /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_benchmark_fixed_embedding/pipeline_manifest.json \
  --target-manifest /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_fixed_embedding_mappo_1000_a128_h128_benchmark/pipeline_manifest.json \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_fixed_embedding_mappo_1000_a64_vs_a128_h128_benchmark_comparison
```

128-agent PASS criteria:

- `trace_mode = "benchmark"`.
- `actual_device = "mps"` and `strict_mps = true`.
- `cpu_fallback_used = false` and `gpu_blocked = false`.
- `configured_num_agents = observed_unique_agent_count = 128`.
- `expected_agent_decisions = observed_agent_decisions = 16384`.
- `policy_logits_shape = [128, 128, 5]`.
- `policy_actions_shape = actor_ratio_shape = actor_surrogate_shape = [128, 128]`.
- `policy_action_element_count = policy_old_logprob_element_count = policy_new_logprob_element_count = policy_entropy_element_count = 16384`.
- `actor_ratio_numel = actor_surrogate_numel = actor_entropy_numel = actor_advantage_numel = 16384`.
- `unique_rollout_actor_elements = 16384`.
- `actor_elements_processed_across_all_ppo_epochs = 32768`.
- `team_advantage_count_before_broadcast = 128`.
- `actor_advantage_count_after_broadcast = 16384`.
- `critic_value_target_count = 128`.
- `checkpoint_reload_ok = true`.
- `checkpoint_reload_logits_max_abs_diff = 0.0`.
- `checkpoint_reload_value_max_abs_diff = 0.0`.
- `canonical_kpi_count = 12` and `canonical_kpi_missing = []`.
- `swap_used_mb_max = 0.0`.

Observed 128-agent scaling gate:

- `status = PASS`.
- `rollout_total_multiplier = 1.0205` versus fixed-embedding 64-agent benchmark.
- `ppo_update_multiplier = 1.3113`.
- `actor_inference_multiplier = 2.2707`; classified as superlinear, but absolute actor inference time is still `0.046593s`.
- `decisions_per_second` increased by `95.97%`.
- `MPS current delta = 0.0 MB`, `MPS driver delta = 16.0 MB`, `RSS delta = 1.797 MB`, `swap delta = 0.0 MB`.
- `approved_for_256_agents = true`.

1000-snapshot / 256-agent / horizon-128 fixed-embedding BENCHMARK validation:

This validates MAPPO agent scaling with the canonical fixed embedding. It does not validate fresh GATv2 cross-process determinism, policy convergence, or policy performance improvement.

Precondition gate:

```text
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_fixed_embedding_mappo_1000_a64_vs_a128_h128_benchmark_comparison/scaling_gate.json
```

Required precondition values:

- `source_scale.agents = 64`.
- `target_scale.agents = 128`.
- `execution.status = PASS`.
- `execution.actual_device = "mps"`.
- `execution.cpu_fallback_used = false`.
- `execution.nan_detected = false` and `execution.inf_detected = false`.
- `execution.swap_used_mb_max = 0.0`.
- `actor_contract.observed_agents = 128`.
- `actor_contract.observed_decisions = 16384`.
- `actor_contract.processed_actor_elements = 32768`.
- `stability.checkpoint_reload_ok = true`.
- `stability.reload_logits_diff = 0.0`.
- `stability.reload_value_diff = 0.0`.
- `stability.canonical_kpi_count = 12`.
- `next_scale.approved_for_256_agents = true`.
- `next_scale.blocking_reasons = []`.

Run command:

```bash
unset PYTORCH_ENABLE_MPS_FALLBACK
python 05_training/smoke_full_gatv2_mappo_pipeline_mac.py \
  --dataset-dir "$DATASET_DIR" \
  --split train \
  --val-split val \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_fixed_embedding_mappo_1000_a256_h128_benchmark \
  --snapshots 1000 \
  --val-snapshots 100 \
  --gat-batch-size 1 \
  --gat-hidden 32 \
  --gat-epochs 1 \
  --gat-grad-clip-norm 5.0 \
  --precomputed-embedding-path /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_full_pipeline_1000_a64_h128_audit/embeddings/gatv2_graph_embeddings.pt \
  --rollout-horizon 128 \
  --ppo-update-epochs 2 \
  --minibatch-size 128 \
  --mappo-grad-clip-norm 0.5 \
  --ppo-clip-epsilon 0.2 \
  --num-agents 256 \
  --seed 1 \
  --condition-id A \
  --preferred-device mps \
  --trace-mode benchmark \
  --require-gpu \
  --clean
```

Verified result path:

```text
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_fixed_embedding_mappo_1000_a256_h128_benchmark/pipeline_manifest.json
```

128-vs-256 scaling comparison:

```bash
python 05_training/compare_mappo_agent_scaling.py \
  --source-comparison /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_fixed_embedding_mappo_1000_a64_vs_a128_h128_benchmark_comparison/scale_comparison.json \
  --source-manifest /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_fixed_embedding_mappo_1000_a128_h128_benchmark/pipeline_manifest.json \
  --target-manifest /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_fixed_embedding_mappo_1000_a256_h128_benchmark/pipeline_manifest.json \
  --output-root /Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_fixed_embedding_mappo_1000_a128_vs_a256_h128_benchmark_comparison
```

256-agent PASS criteria:

- `trace_mode = "benchmark"`.
- `actual_device = "mps"` and `strict_mps = true`.
- `cpu_fallback_used = false` and `gpu_blocked = false`.
- `configured_num_agents = observed_unique_agent_count = 256`.
- `expected_agent_decisions = observed_agent_decisions = 32768`.
- `policy_logits_shape = [128, 256, 5]`.
- `policy_actions_shape = actor_ratio_shape = actor_surrogate_shape = [128, 256]`.
- `policy_action_element_count = policy_old_logprob_element_count = policy_new_logprob_element_count = policy_entropy_element_count = 32768`.
- `actor_ratio_numel = actor_surrogate_numel = actor_entropy_numel = actor_advantage_numel = 32768`.
- `unique_rollout_actor_elements = 32768`.
- `actor_elements_processed_across_all_ppo_epochs = 65536`.
- `team_advantage_count_before_broadcast = 128`.
- `actor_advantage_count_after_broadcast = 32768`.
- `critic_value_target_count = 128`.
- `checkpoint_reload_ok = true`.
- `checkpoint_reload_logits_max_abs_diff = 0.0`.
- `checkpoint_reload_value_max_abs_diff = 0.0`.
- `canonical_kpi_count = 12` and `canonical_kpi_missing = []`.
- `swap_used_mb_max = 0.0`.
- `full_agent_decision_trace_generated = false`; Benchmark evidence is `rollout_tensor_hashes.json`, `rollout_tensor_summary.json`, `sampled_decision_trace.json`, `actor_loss_audit.json`, and `ppo_update_profile.json`.

Observed 256-agent scaling gate:

- `status = PASS`.
- `observed_agents = 256`.
- `observed_decisions = 32768`.
- `unique_actor_elements = 32768`.
- `actor_elements_processed_across_all_ppo_epochs = 65536`.
- `team_advantage_count_before_broadcast = 128`.
- `actor_advantage_count_after_broadcast = 32768`.
- `critic_value_target_count = 128`.
- `rollout_total_seconds = 0.974521`.
- `rollout_online_seconds = 0.974521`.
- `ppo_update_seconds = 0.887824`.
- `actor_inference_seconds = 0.023669`.
- `actor_inference_calls = 128`.
- `agents_per_actor_call = 256`.
- `actor_inference_share_percent = 2.4288`.
- `agent_decisions_per_second_total = 33624.725801`.
- `checkpoint_reload_logits_max_abs_diff = 0.0`.
- `checkpoint_reload_value_max_abs_diff = 0.0`.
- `MPS current = 0.024 MB`, `MPS driver = 76.031 MB`, `RSS = 1278.0 MB`, `swap = 0.0 MB`.
- `128 -> 256 rollout_total_multiplier = 1.0420`.
- `128 -> 256 rollout_online_multiplier = 1.0420`.
- `128 -> 256 actor_inference_multiplier = 0.5080`; classified as sublinear.
- `128 -> 256 action_sampling_multiplier = 0.9159`; classified as sublinear.
- `128 -> 256 simulator_step_multiplier = 1.3230`; classified as sublinear.
- `128 -> 256 ppo_update_multiplier = 0.9304`; classified as sublinear.
- `decision throughput change = +91.94%`.
- `MPS current delta = 0.0 MB`, `MPS driver delta = 24.031 MB`, `RSS delta = 18.234 MB`, `swap delta = 0.0 MB`.
- `approved_for_512_agents = true`.

Actual SUSEONG_SERVICE / 512-agent / horizon-256 hardware gate:

This gate is intentionally stricter than the fixed-embedding synthetic MAPPO scaling tests. It requires a DB-grounded `SUSEONG_SERVICE` graph, full-graph-to-service-node embedding mapping, route-aware causal simulator preflights, and then the final 512-agent benchmark. The existing synthetic smoke runner must not be used as a substitute.

Current gate result:

```text
status = BLOCKED
capacity_assessment = NOT_EVALUATED
validation_scope = suseong_service_graph_fixed_embedding_mappo_scaling
graph_scope = SUSEONG_SERVICE
embedding_mode = fixed
simulator_mode = suseong_route_aware_causal_service_graph
trace_mode = benchmark
configured agents = 512
rollout horizon = 256
```

Passed prerequisites:

- 256-agent fixed-embedding scaling gate passed.
- `approved_for_512_agents = true`.
- canonical fixed embedding tensor SHA-256 matches the 256-agent manifest.
- canonical fixed embedding tensor SHA-256: `5a0c3d2b080bacd640290c25fcde853820414d0aea6d6002b835c11793aa5f88`.
- canonical fixed embedding shape: `[1000, 32]`.
- canonical fixed embedding dtype: `torch.float32`.

Blocking reasons:

- No read-only DB connection information is available in the Mac environment: `URBANBUS_DB_DSN`, `DATABASE_URL`, and `PG*` variables are unset.
- No existing `suseong_service_graph_v1` source artifact was available before this blocked gate.
- `SUSEONG_CORE` stop identification was not executed because DB schema/source inspection is unavailable.
- `SUSEONG_SERVICE` route-direction sequence and boundary gateway graph were not constructed.
- The fixed-embedding manifest does not include a full graph node ordering path or `full_graph_node_id_hash`, so service-node to embedding-row mapping cannot be proven.
- Preflight A and B were not run because service graph construction and embedding mapping are blocked.

Blocked gate artifacts:

```text
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/suseong_service_graph_v1/service_graph_manifest.json
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_suseong_service_fixed_embedding_mappo_1000_a512_h256_benchmark/pipeline_manifest.json
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_suseong_service_a512_h256_hardware_gate/hardware_gate.json
/Users/arty/Documents/Codex/urbanbus_rl_project/05_training/artifacts/mac_suseong_service_a512_h256_hardware_gate/capacity_assessment.json
```

Next unblock step:

```text
Provide a read-only DB DSN or prebuilt DB-grounded suseong_service_graph_v1 artifacts,
including service graph node/edge/route sequence files and full-graph node ordering
metadata. Then rerun service graph connectivity, embedding mapping, Preflight A,
Preflight B, and only then the 512-agent final benchmark.
```

## Known blockers

- 128-snapshot, 512-snapshot, 512-snapshot/horizon-128, 512-snapshot/16-agent/horizon-128, 1000-snapshot/16-agent/horizon-128, 1000-snapshot/32-agent/horizon-128, and 1000-snapshot/64-agent/horizon-128 full pipeline tests passed on MPS in approved native runs. If a restricted shell reports MPS unavailable, rerun the pipeline in the approved project environment and check `pipeline_manifest.json`.
- No PostgreSQL dump was transferred, so rebuilding tensors from scratch still needs a local DB restore path.
- The existing `05_training/train_gatv2.py` remains Windows-path-based; use `05_training/smoke_gatv2_mac.py` for Mac smoke tests.
