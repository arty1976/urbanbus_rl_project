# Step 101 CausalSimulatorAdapter v2 Scaffold Package

압축을 반드시 풀고, 압축푼 폴더 안의 `Step101_CausalSimulatorV2Adapter_Setup.ps1`을 실행하세요.
단일 PS1만 프로젝트 루트로 복사하면 `project_files`를 찾지 못합니다.

## 실행

```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project"

powershell -ExecutionPolicy Bypass -File "압축푼경로\Step101_CausalSimulatorV2Adapter_Setup.ps1" -RunSelfTest
```

## 실제 artifact smoke만 재실행

```powershell
powershell -ExecutionPolicy Bypass -File "압축푼경로\Step101_RunActualAdapterSmoke.ps1"
```

## 생성 파일

```text
05_training/adapters/causal_simulator_v2_adapter_contract.md
05_training/adapters/causal_simulator_v2_adapter.py
05_training/adapters/test_causal_simulator_v2_adapter.py
05_training/run_causal_simulator_v2_adapter_smoke.py
```
