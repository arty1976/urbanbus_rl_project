# Step 154 — H200 nvidia-smi GPU Monitor Logger Scaffold

## Purpose

Step 154 defines a monitoring-only GPU logging scaffold for H200-side experiments.

This step does **not** release training.  
This step does **not** permit actual reward ablation execution.  
This step does **not** permit live mutation of reward weights, learning rate, batch size, or any other training control variable.

It only defines how to observe GPU state using `nvidia-smi` when the H200 server is available.

## Status

```text
monitor_status = NVIDIA_SMI_GPU_MONITOR_SCAFFOLD_READY_MONITORING_ONLY_STILL_LOCKED
monitor_mode = MONITORING_ONLY
control_policy = NO_LIVE_MUTATION_CONFIG_BASED_NEXT_RUN_ONLY
actual_execution_allowed = false
actual_execution_released = false
train_allowed = false
live_mutation_allowed = false
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```

## Why this step exists

Step 152 fixed the overall H200 observability contract.  
Step 153 created a TensorBoard + JSONL metric logger integration scaffold.  
Step 154 adds a GPU telemetry scaffold around `nvidia-smi`.

The main goal is to ensure that future H200 training runs produce enough GPU telemetry to diagnose:

- GPU utilization
- GPU memory pressure
- power draw
- temperature
- ECC (Error-Correcting Code=오류 정정 코드) error counters when available
- driver / CUDA (Compute Unified Device Architecture=엔비디아 GPU 연산 플랫폼) visibility
- per-run GPU observation provenance

## Local notebook behavior

The local notebook may not have NVIDIA GPU access.  
Therefore Step 154 must pass even when `nvidia-smi` is unavailable.

If `nvidia-smi` is unavailable, the scaffold records:

```text
nvidia_smi_status = nvidia_smi_unavailable
gpu_observation_source = schema_sample_fallback
```

This is expected on a non-H200 local environment and is not a failure.

## H200 behavior

On H200, the same logger can probe `nvidia-smi` and write GPU rows using the fixed schema.

The intended one-shot probe command is:

```bash
python 05_training/observability/h200_nvidia_smi_gpu_monitor_step154.py \
  --project-root /workspace/urbanbus_rl_project \
  --output-root artifacts/observability/h200_nvidia_smi_gpu_monitor_step154_h200_probe \
  --probe-once
```

This remains monitoring-only. It does not start training.

## Output files

The generator writes:

```text
artifacts/observability/h200_nvidia_smi_gpu_monitor_step154/
  h200_nvidia_smi_gpu_monitor_step154_manifest.json
  gpu_metrics_step154_sample.csv
  gpu_metrics_step154_sample.jsonl
  gpu_monitor_status_step154.json
  nvidia_smi_query_contract_step154.json
```

## Fixed CSV / JSONL columns

The GPU metrics row schema is:

```text
timestamp_utc
run_id
condition_id
reward_id
seed
global_step
gpu_index
gpu_name
gpu_uuid
driver_version
cuda_version
utilization_gpu_pct
utilization_memory_pct
memory_total_mib
memory_used_mib
memory_free_mib
temperature_gpu_c
power_draw_w
power_limit_w
pcie_link_gen_current
pcie_link_width_current
ecc_volatile_uncorrected
ecc_aggregate_uncorrected
nvidia_smi_available
nvidia_smi_status
source_mode
monitoring_only
actual_execution_allowed
train_allowed
live_mutation_allowed
```

## Safety rules

1. The monitor cannot set training variables.
2. The monitor cannot enable `train_allowed`.
3. The monitor cannot enable `actual_execution_allowed`.
4. The monitor cannot enable `actual_execution_released`.
5. The monitor cannot make paper-level or causal-performance claims.
6. Missing local `nvidia-smi` is a warning/state, not a failure.
7. Missing required telemetry schema columns is a failure.

## Next step

Recommended next step:

```text
Step 155 — H200 observability runbook and dashboard launch commands
```

Step 155 should document how to run:

- TensorBoard
- JSONL metric tailing
- `nvidia-smi` monitor probe
- optional SSH (Secure Shell=보안 셸) port forwarding
- safe stop / abort observation procedures
