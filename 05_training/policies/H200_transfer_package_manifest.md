# Step 74 — H200 transfer package manifest

## 1. Purpose

This document defines the transfer package manifest for moving the UrbanBus MAPPO training workflow from the local notebook to the H200 server.

H200 (NVIDIA H200 GPU=엔비디아 H200 그래픽 처리 장치) transfer must be based on a pinned git commit, a known set of source files, required contract artifacts, and explicit H200-side commands.

MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) training outputs must not be interpreted as actual results until they later pass the checkpoint folder contract, actual checkpoint validator, Step 58 preflight, Step 59 plan, Step 60 dry-run, Step 61 runbook review, and Step 62 arrival workflow.

---

## 2. Manifest modes

### Planning mode

Planning mode creates a manifest even if some training artifacts are not yet available.

```text
transfer_manifest_status = PLANNING_MANIFEST_READY
strict_artifacts = false
```

This mode is useful before the final H200 transfer.

### Strict artifact mode

Strict artifact mode blocks if required transfer artifacts are missing.

```text
transfer_manifest_status = TRANSFER_READY
strict_artifacts = true
```

This mode should be used immediately before copying or cloning to H200.

---

## 3. Required source files

The manifest records the source files needed for H200 training handoff.

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
05_training/policies/generate_h200_transfer_package_manifest.py

05_training/policies/H200_server_transfer_checklist.md
05_training/policies/H200_mappo_training_runbook.md
05_training/policies/H200_actual_checkpoint_folder_contract.md
05_training/policies/H200_training_local_validation_bundle.md
05_training/policies/MAPPO_actual_reproducibility_note.md
```

---

## 4. Required contract artifacts

The following artifacts are required for actual transfer.

```text
artifacts/baseline_v1/baseline_contract.json
artifacts/experiment_A_v1/experiment_A_contract.json
artifacts/baseline_v1/B1_noop/scenario_index.parquet
artifacts/baseline_v1/B2_rulebased/scenario_index.parquet
```

If these are missing, planning mode may still write a manifest, but strict artifact mode must block.

---

## 5. Conditional GATv2 artifacts

Depending on the final MAPPO training implementation, the H200 server may need exported GATv2 artifacts.

GATv2 (Graph Attention Network version 2=그래프 어텐션 네트워크 버전 2) candidate transfer paths:

```text
artifacts/gatv2_v1/best_mappo.pt
artifacts/gatv2_v1/artifact_contract.json
artifacts/gatv2_v1/snapshot_index.parquet
artifacts/gatv2_v1/embeddings/
artifacts/gatv2_v1/predictions/
```

If these do not exist, the manifest must clearly mark them as missing conditional artifacts.

Smoke or placeholder GATv2 artifacts cannot support actual performance claims.

---

## 6. Required H200 command pack

The manifest must include the H200-side command pack.

```text
git checkout <PINNED_COMMIT_HASH>
generate_h200_environment_report.py --require-cuda --require-h200-name --require-clean-git
run_h200_training_local_validation_bundle.py --execute
train_mappo_actual.py --dry-run-contract
validate_mappo_checkpoint.py --mode actual
h200_actual_checkpoint_preflight_checklist.py --mode actual
validate_h200_actual_checkpoint_folder.py
run_h200_actual_checkpoint_arrival_workflow.py
```

---

## 7. Required return package

After H200 training, the full run folder must return to the notebook.

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

Do not return only `best_mappo.pt`.

---

## 8. Claim boundary

Passing Step 74 means:

```text
The H200 transfer package manifest is ready.
```

It does not mean:

```text
A trained H200 checkpoint exists.
The MAPPO policy causally outperformed baselines.
```

Causal performance claims require Phase 2 causal simulator evaluation.
