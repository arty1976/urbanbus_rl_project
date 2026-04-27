# Step 66 — H200 MAPPO training runbook draft

## 1. Purpose

This runbook defines how to start actual MAPPO training on an H200 server.

H200 (NVIDIA H200 GPU=엔비디아 H200 그래픽 처리 장치) training is the step that should eventually produce a real `best_mappo.pt` checkpoint. That checkpoint must later pass Step 58 actual preflight, Step 59 execution plan generation, Step 60 dry-run, Step 61 operator review, and Step 62 checkpoint arrival workflow before it can be used as an actual MAPPO artifact.

MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) training must remain separate from smoke, fake, scaffold, placeholder, and historical replay-only validation.

This document is a runbook draft. It does not itself implement full training.

---

## 2. Current boundary

The current local notebook pipeline has completed the actual execution gates, but the real H200-trained checkpoint does not yet exist.

Current state:

```text
Step 58 = actual checkpoint preflight gate exists
Step 59 = A-family actual execution plan gate exists
Step 60 = actual matrix runner guard exists
Step 61 = final actual execution runbook exists
Step 62 = actual checkpoint arrival workflow exists
Step 63 = project log updated
Step 64 = neural strict checkpoint leftovers stashed
Step 65 = actual execution reproducibility note exists
```

Step 66 prepares the H200 training side of the workflow.

---

## 3. Training objective

The first H200 training objective is to produce a checkpoint for condition A.

Condition A:

```text
condition_id = A
condition_name = pure_mappo_baseline
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
policy_source_mode = mappo_actual
reward_version = mappo_reward_v1
energy_proxy_model_version = daegu_energy_proxy_v1
```

The trained checkpoint must eventually be usable for A-family evaluation:

```text
A
A90
A80
A70
```

A90/A80/A70 are evaluation conditions using active bus ratio reduction. They must not silently change Qwen, reward, data split, or checkpoint contract.

---

## 4. Non-negotiable training invariants

The following must be fixed before training starts.

```text
git commit hash pinned
dataset artifact pinned
baseline contract pinned
experiment A contract pinned
reward_version = mappo_reward_v1
energy_proxy_model_version = daegu_energy_proxy_v1
k_dist_kwh_per_m = 0.0012
k_acc_kwh_per_event = 0.1800
k_idle_kwh_per_sec = 0.0080
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
seeds = 1,2,3
same_initial_state = true
same_exogenous_events = true
same_eval_window = true
```

Training must not use wall-clock time as the learning budget. Training budget must be defined by env step count, episode count, or update count.

---

## 5. Required repository state on H200

The H200 server must use a clean, pinned project checkout.

Recommended commands:

```bash
git clone https://github.com/arty1976/urbanbus_rl_project.git
cd urbanbus_rl_project
git checkout <PINNED_COMMIT_HASH>
git status --short
```

Expected:

```text
git status --short = empty
```

If local H200 changes are required, they must be committed or documented before running training.

---

## 6. Required environment report

Before training, write an environment report.

Minimum fields:

```text
hostname
os
python_version
torch_version
cuda_available
cuda_version
gpu_count
gpu_names
driver_version
nvidia_smi_output
git_commit
created_at_utc
```

Recommended file:

```text
artifacts/h200_mappo_training/<run_id>/environment_report.json
```

---

## 7. Python environment setup

Recommended environment setup:

```bash
python -m venv 05_training/.venv
source 05_training/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r 05_training/requirements.txt
```

If H200 uses CUDA-specific PyTorch wheels, install the correct torch package for the server before running the project tests.

CUDA (Compute Unified Device Architecture=통합 병렬 연산 장치 아키텍처) availability must be verified.

Example check:

```bash
python - <<'PY'
import torch
print("torch_version =", torch.__version__)
print("cuda_available =", torch.cuda.is_available())
print("cuda_device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

Training must not start if CUDA is unavailable on the H200 training server.

---

## 8. Required local validation on H200 before training

Run the policy and checkpoint gate tests on the H200 checkout before starting training.

Recommended validation:

```bash
python -m py_compile \
  05_training/policies/validate_mappo_checkpoint.py \
  05_training/policies/h200_actual_checkpoint_preflight_checklist.py \
  05_training/policies/a_family_mappo_actual_execution_plan.py \
  05_training/policies/run_a_family_mappo_actual_matrix_from_plan.py \
  05_training/policies/run_h200_actual_checkpoint_arrival_workflow.py

python 05_training/policies/test_validate_mappo_checkpoint.py
python 05_training/policies/test_h200_actual_checkpoint_preflight_checklist.py
python 05_training/policies/test_a_family_mappo_actual_execution_plan.py
python 05_training/policies/test_run_a_family_mappo_actual_matrix_from_plan.py
python 05_training/policies/test_h200_actual_mappo_final_runbook.py
python 05_training/policies/test_h200_actual_checkpoint_arrival_workflow.py
python 05_training/policies/test_mappo_actual_reproducibility_note.py
```

All tests must pass before training.

---

## 9. Required input artifacts

Training must receive or verify these inputs.

```text
artifacts/baseline_v1/baseline_contract.json
artifacts/experiment_A_v1/experiment_A_contract.json
scenario_index.parquet
GATv2 encoder artifact or embedding cache
MAPPO reward configuration
MAPPO checkpoint contract
```

GATv2 (Graph Attention Network version 2=그래프 어텐션 네트워크 버전 2) artifacts must be pinned. If the H200 server trains MAPPO using exported GATv2 embeddings, the exact embedding cache path and hash must be recorded.

---

## 10. Recommended H200 training output structure

Use one immutable run directory per training attempt.

```text
artifacts/h200_mappo_training/<run_id>/
├── environment_report.json
├── training_config.json
├── training_command.txt
├── git_commit.txt
├── logs/
│   ├── train_stdout.log
│   ├── train_stderr.log
│   └── train_metrics.jsonl
├── checkpoints/
│   ├── last_mappo.pt
│   ├── best_mappo.pt
│   └── checkpoint_manifest.json
├── validation/
│   ├── validate_mappo_checkpoint_smoke_report.json
│   └── validate_mappo_checkpoint_actual_report.json
└── README_run_summary.md
```

Only `best_mappo.pt` that passes actual validation may be promoted to Step 58.

---

## 11. Training command template

The exact training entrypoint may change as the implementation matures. The command must nevertheless expose the following contract.

Required arguments:

```text
--condition-id A
--experiment-contract artifacts/experiment_A_v1/experiment_A_contract.json
--baseline-contract artifacts/baseline_v1/baseline_contract.json
--output-root artifacts/h200_mappo_training/<run_id>
--seeds 1 2 3
--device cuda
--qwen-train false
--qwen-inference false
--qwen-trigger-rate 0.0
--reward-version mappo_reward_v1
--energy-proxy-model-version daegu_energy_proxy_v1
--k-dist-kwh-per-m 0.0012
--k-acc-kwh-per-event 0.1800
--k-idle-kwh-per-sec 0.0080
```

Draft command shape:

```bash
python 05_training/train_mappo_actual.py \
  --condition-id A \
  --experiment-contract artifacts/experiment_A_v1/experiment_A_contract.json \
  --baseline-contract artifacts/baseline_v1/baseline_contract.json \
  --output-root artifacts/h200_mappo_training/<run_id> \
  --seeds 1 2 3 \
  --device cuda \
  --qwen-train false \
  --qwen-inference false \
  --qwen-trigger-rate 0.0 \
  --reward-version mappo_reward_v1 \
  --energy-proxy-model-version daegu_energy_proxy_v1 \
  --k-dist-kwh-per-m 0.0012 \
  --k-acc-kwh-per-event 0.1800 \
  --k-idle-kwh-per-sec 0.0080
```

If `train_mappo_actual.py` does not yet exist, this command is a contract target, not an executable command.

---

## 12. Checkpoint content contract

The training output checkpoint must include the fields required by the MAPPO checkpoint contract.

Minimum expected fields:

```text
artifact_version
condition_id
qwen_train
qwen_inference
qwen_trigger_rate
policy_architecture
actor_obs_dim
critic_obs_dim
action_dim
hidden_dim
shared_policy
ctde_enabled
model_state_dict
training_seed
git_commit
created_at_utc
reward_version
rollout_schema_version
policy_interface_version
action_space_version
observation_space_version
energy_proxy_model_version
energy_proxy_unit
k_dist_kwh_per_m
k_acc_kwh_per_event
k_idle_kwh_per_sec
trained_model
performance_claim_allowed
```

For actual checkpoint promotion:

```text
trained_model = true
performance_claim_allowed = true
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
```

---

## 13. Immediate post-training validation

After `best_mappo.pt` is produced, validate it before copying it to the notebook or execution machine.

Recommended command:

```bash
python 05_training/policies/validate_mappo_checkpoint.py \
  --checkpoint artifacts/h200_mappo_training/<run_id>/checkpoints/best_mappo.pt \
  --mode actual
```

Then run Step 58 preflight:

```bash
python 05_training/policies/h200_actual_checkpoint_preflight_checklist.py \
  --checkpoint-path artifacts/h200_mappo_training/<run_id>/checkpoints/best_mappo.pt \
  --mode actual \
  --report-path artifacts/h200_mappo_training/<run_id>/validation/h200_actual_checkpoint_preflight_report.json
```

The checkpoint must not be used if either command fails.

---

## 14. Transfer back to notebook or execution machine

When transferring `best_mappo.pt`, also transfer:

```text
environment_report.json
training_config.json
training_command.txt
git_commit.txt
checkpoint_manifest.json
validate_mappo_checkpoint_actual_report.json
h200_actual_checkpoint_preflight_report.json
README_run_summary.md
```

Do not transfer only the `.pt` file without metadata.

---

## 15. Arrival workflow after transfer

After the checkpoint arrives on the notebook or execution machine, run Step 62.

Example PowerShell:

```powershell
$ErrorActionPreference = "Stop"
Set-Location "C:\Users\ryujo\urbanbus_rl_project"

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

& $py ".\05_training\policies\run_h200_actual_checkpoint_arrival_workflow.py" `
  --checkpoint-path ".\artifacts\h200_mappo_training\<run_id>\checkpoints\best_mappo.pt" `
  --output-root ".\artifacts\experiment_A_v1\h200_actual_arrival\<run_id>"
```

Expected success:

```text
arrival_workflow_status = READY_FOR_MANUAL_EXECUTE
```

This still does not imply causal performance improvement.

---

## 16. Failure handling

Stop immediately if any of these occurs.

```text
CUDA unavailable
git commit not pinned
experiment contract missing
baseline contract missing
qwen_train is true
qwen_inference is true
qwen_trigger_rate is not 0.0
reward_version mismatch
energy_proxy_model_version mismatch
energy coefficients mismatch
checkpoint missing required fields
trained_model is false
performance_claim_allowed is false
validate_mappo_checkpoint.py --mode actual fails
Step 58 preflight fails
```

Do not patch around failed gates. Fix the upstream issue and rerun.

---

## 17. Correct interpretation

Allowed:

```text
H200 training produced a checkpoint that passed actual-mode structural validation.
```

Allowed:

```text
The checkpoint is ready for Step 62 arrival workflow and Step 60 dry-run guarded execution.
```

Not allowed:

```text
The checkpoint proves causal superiority over baselines.
```

Causal superiority requires Phase 2 causal simulator evaluation.

---

## 18. Next implementation target

After this draft runbook, the next implementation target is one of the following.

```text
1. train_mappo_actual.py entrypoint contract
2. actual checkpoint folder contract validator
3. H200 environment_report generator
4. Phase 2 causal simulator adapter design
```
