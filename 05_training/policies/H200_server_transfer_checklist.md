# Step 72 — H200 server transfer checklist

## 1. Purpose

This checklist defines what must be transferred to the H200 server before actual MAPPO training begins, and what must be transferred back after training.

H200 (NVIDIA H200 GPU=엔비디아 H200 그래픽 처리 장치) is used for actual MAPPO training. MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) training outputs must not be treated as actual experiment results until they pass the checkpoint, folder, preflight, plan, dry-run, and arrival workflow gates.

This checklist is an operator handoff document. It does not run training.

---

## 2. Current local gate status

The following local notebook gates are already defined.

```text
Step 58: H200 actual checkpoint preflight
Step 59: A-family mappo_actual execution plan
Step 60: mappo_actual matrix runner guard
Step 61: H200 actual MAPPO final runbook
Step 62: actual checkpoint arrival workflow
Step 65: MAPPO actual reproducibility note
Step 66: H200 MAPPO training runbook
Step 67: H200 environment report generator
Step 68: H200 actual checkpoint folder validator
Step 69: train_mappo_actual.py CLI contract scaffold
Step 70: H200 training local validation bundle
Step 71: project_log.md update for Steps 66-70
```

The H200 transfer must preserve this gate chain.

---

## 3. Commit pinning before transfer

Before transferring to H200, pin the exact git commit.

On the local notebook:

```powershell
git status --short
git log --oneline -5
git rev-parse HEAD
```

Recommended rule:

```text
Only transfer or clone a committed state.
Do not manually copy a dirty working tree to H200.
```

Known exception:

```text
.claude/worktrees may show nested gitlink/worktree pointer changes.
Those entries are not part of the MAPPO training handoff and must not be mixed into research commits.
```

---

## 4. Recommended H200 checkout

On the H200 server:

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

If the H200 server requires local changes, document them before training.

---

## 5. Minimum source files that must exist on H200

The H200 checkout must contain these files.

```text
05_training/train_mappo_actual.py
05_training/mappo_runner.py
05_training/simulator_adapter_interface.py
05_training/adapters/historical_replay_adapter.py

05_training/policies/validate_mappo_checkpoint.py
05_training/policies/h200_actual_checkpoint_preflight_checklist.py
05_training/policies/a_family_mappo_actual_execution_plan.py
05_training/policies/run_a_family_mappo_actual_matrix_from_plan.py
05_training/policies/run_h200_actual_checkpoint_arrival_workflow.py
05_training/policies/generate_h200_environment_report.py
05_training/policies/validate_h200_actual_checkpoint_folder.py
05_training/policies/run_h200_training_local_validation_bundle.py

05_training/policies/H200_mappo_training_runbook.md
05_training/policies/H200_actual_checkpoint_folder_contract.md
05_training/policies/H200_training_local_validation_bundle.md
05_training/policies/MAPPO_actual_reproducibility_note.md
```

---

## 6. Minimum artifacts that must be available on H200

The H200 server needs the contracts and dataset artifacts used by MAPPO training.

Required local-to-H200 artifacts:

```text
artifacts/baseline_v1/baseline_contract.json
artifacts/experiment_A_v1/experiment_A_contract.json
artifacts/baseline_v1/B1_noop/scenario_index.parquet
artifacts/baseline_v1/B2_rulebased/scenario_index.parquet
```

Depending on the final training implementation, H200 may also need:

```text
artifacts/gatv2_v1/best_mappo.pt
artifacts/gatv2_v1/artifact_contract.json
artifacts/gatv2_v1/snapshot_index.parquet
artifacts/gatv2_v1/embeddings/
artifacts/gatv2_v1/predictions/
```

If the GATv2 artifact directory does not yet exist, this must be treated as a training blocker or replaced with an explicitly documented stub/smoke input. Smoke input cannot support actual performance claims.

GATv2 (Graph Attention Network version 2=그래프 어텐션 네트워크 버전 2) artifacts must be version-pinned if used by MAPPO.

---

## 7. H200 Python environment setup

On H200:

```bash
python -m venv 05_training/.venv
source 05_training/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r 05_training/requirements.txt
```

If the H200 server requires a CUDA-specific PyTorch installation, install the correct torch build before project validation.

CUDA (Compute Unified Device Architecture=통합 병렬 연산 장치 아키텍처) must be available for actual H200 training.

---

## 8. H200 environment report command

Run this before training.

```bash
python 05_training/policies/generate_h200_environment_report.py \
  --project-root . \
  --output-path artifacts/h200_mappo_training/<run_id>/environment_report.json \
  --require-cuda \
  --require-h200-name \
  --require-clean-git
```

Expected:

```text
report_status = PASS
cuda_available = true
gpu_count > 0
gpu_names contains H200
repo_dirty = false
```

---

## 9. Local validation bundle command on H200

Run this after checkout and environment setup.

```bash
python 05_training/policies/run_h200_training_local_validation_bundle.py \
  --project-root . \
  --report-path artifacts/h200_local_validation_bundle/step70_h200_execute_report.json \
  --python-exe "$(which python)" \
  --execute
```

Expected:

```text
bundle_status = EXECUTION_VALIDATED
executed = true
```

If this fails, do not start training.

---

## 10. Training CLI dry-run contract on H200

Before implementing or launching full training, run the Step 69 dry-run contract.

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
  --k-idle-kwh-per-sec 0.0080 \
  --dry-run-contract
```

Expected dry-run contract state:

```text
trained_model = false
performance_claim_allowed = false
checkpoint_status = not_trained
```

This command validates the CLI and folder scaffold only. It is not actual training.

---

## 11. Actual training implementation gate

Full training may start only after the dry-run contract is replaced or extended by an implemented training path.

Before full training, confirm that the implementation can write:

```text
training_config.json
training_command.txt
git_commit.txt
logs/train_stdout.log
logs/train_stderr.log
logs/train_metrics.jsonl
checkpoints/best_mappo.pt
checkpoints/checkpoint_manifest.json
validation/validate_mappo_checkpoint_actual_report.json
validation/h200_actual_checkpoint_preflight_report.json
README_run_summary.md
```

The final `best_mappo.pt` must satisfy:

```text
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
```

---

## 12. Post-training validation on H200

After `best_mappo.pt` is produced, run:

```bash
python 05_training/policies/validate_mappo_checkpoint.py \
  --checkpoint artifacts/h200_mappo_training/<run_id>/checkpoints/best_mappo.pt \
  --mode actual
```

Then run:

```bash
python 05_training/policies/h200_actual_checkpoint_preflight_checklist.py \
  --checkpoint-path artifacts/h200_mappo_training/<run_id>/checkpoints/best_mappo.pt \
  --mode actual \
  --report-path artifacts/h200_mappo_training/<run_id>/validation/h200_actual_checkpoint_preflight_report.json
```

Then validate the full folder:

```bash
python 05_training/policies/validate_h200_actual_checkpoint_folder.py \
  --run-root artifacts/h200_mappo_training/<run_id>
```

Expected:

```text
folder_contract_status = PASS
ready_for_step62_arrival_workflow = true
```

---

## 13. Required return package from H200 to notebook

After training, transfer the full run folder back, not only the checkpoint.

Required return package:

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
│   ├── best_mappo.pt
│   └── checkpoint_manifest.json
├── validation/
│   ├── validate_mappo_checkpoint_actual_report.json
│   ├── h200_actual_checkpoint_preflight_report.json
│   └── h200_actual_checkpoint_folder_contract_report.json
└── README_run_summary.md
```

Do not copy only:

```text
best_mappo.pt
```

A checkpoint without metadata is not sufficient for actual promotion.

---

## 14. Notebook arrival workflow after return

On the notebook or execution machine:

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

Expected:

```text
arrival_workflow_status = READY_FOR_MANUAL_EXECUTE
```

---

## 15. Claim boundary

Allowed after Step 72 preparation:

```text
The H200 transfer checklist and training handoff are ready.
```

Allowed after successful H200 training and Step 62 arrival:

```text
The H200-trained MAPPO checkpoint passed actual-mode arrival workflow up to dry-run validation.
```

Not allowed:

```text
The policy causally outperformed baselines.
```

Causal performance claims require Phase 2 causal simulator evaluation.

---

## 16. Stop conditions

Stop if any of the following occurs.

```text
git commit is not pinned
H200 checkout is dirty
CUDA unavailable
H200 GPU not detected when --require-h200-name is used
Step 70 execute bundle fails on H200
Step 69 dry-run contract fails
baseline_contract.json missing
experiment_A_contract.json missing
GATv2 artifacts missing without documented substitute
qwen_train is true
qwen_inference is true
qwen_trigger_rate is not 0.0
reward_version mismatch
energy_proxy_model_version mismatch
energy coefficient mismatch
best_mappo.pt missing after training
validate_mappo_checkpoint.py --mode actual fails
Step 58 preflight fails
Step 68 folder contract fails
```

Do not patch around a failed gate. Fix the upstream source and rerun.
