# Step 102 CausalSimulatorAdapter v2 Rollout Writer Smoke Package

압축을 반드시 풀고, 압축푼 폴더 안의 `Step102_CausalSimulatorV2Rollout_Setup.ps1`을 실행하세요.

## 실행

```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project"

powershell -ExecutionPolicy Bypass -File "압축푼경로\Step102_CausalSimulatorV2Rollout_Setup.ps1" -RunSelfTest
```

## 실제 artifact rollout smoke만 재실행

```powershell
powershell -ExecutionPolicy Bypass -File "압축푼경로\Step102_RunActualRolloutSmoke.ps1"
```

## 생성 파일

```text
05_training/adapters/causal_simulator_v2_rollout_writer_contract.md
05_training/adapters/test_causal_simulator_v2_rollout_smoke.py
05_training/run_causal_simulator_v2_rollout_smoke.py
```

## 생성 artifact

```text
artifacts/causal_simulator_v2_rollout_smoke/rollout_manifest.json
artifacts/causal_simulator_v2_rollout_smoke/C2_STATIC_SIGNAL/rollouts/seed_101/raw_events.parquet
artifacts/causal_simulator_v2_rollout_smoke/C2_STATIC_SIGNAL/rollouts/seed_101/window_rollup.parquet
```
