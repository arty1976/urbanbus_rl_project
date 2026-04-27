# Step 68 — H200 actual checkpoint folder contract

## 1. Purpose

This contract defines the minimum folder structure required before an H200-trained MAPPO checkpoint can be moved into the actual execution pipeline.

MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) actual promotion must not be based on `best_mappo.pt` alone. The checkpoint must be accompanied by environment metadata, training configuration, command record, git commit, checkpoint manifest, validation reports, and a run summary.

---

## 2. Required folder structure

Expected root:

```text
artifacts/h200_mappo_training/<run_id>/
```

Required structure:

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
│   └── h200_actual_checkpoint_preflight_report.json
└── README_run_summary.md
```

---

## 3. Required metadata values

The folder is valid only if the metadata preserves the actual MAPPO contract.

```text
condition_id = A
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
reward_version = mappo_reward_v1
energy_proxy_model_version = daegu_energy_proxy_v1
k_dist_kwh_per_m = 0.0012
k_acc_kwh_per_event = 0.1800
k_idle_kwh_per_sec = 0.0080
trained_model = true
performance_claim_allowed = true
```

---

## 4. Validation reports

The folder must include both:

```text
validation/validate_mappo_checkpoint_actual_report.json
validation/h200_actual_checkpoint_preflight_report.json
```

Both reports must be PASS-like and must reference the same `best_mappo.pt` path when a checkpoint path is present.

---

## 5. Claim boundary

Passing this folder validator means:

```text
The H200 checkpoint package is complete enough to enter Step 62 arrival workflow.
```

It does not mean:

```text
The MAPPO policy causally outperformed baselines.
```

Causal performance claims require Phase 2 causal simulator evaluation.
