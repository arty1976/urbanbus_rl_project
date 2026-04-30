# Step 99 Daegu Signal CSV Preflight Package

## 실행

```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project"

powershell -ExecutionPolicy Bypass -File "압축푼경로\Step99_DaeguSignalCsvPreflight_Setup.ps1" -RunSelfTest
```

## DB 기반 실제 feature 생성

```powershell
powershell -ExecutionPolicy Bypass -File "압축푼경로\Step99_RunBuilderWithDb.ps1"
```

`URBANBUS_DB_DSN`이 없으면 `-DbUrl`을 직접 넘기세요.

## 생성 파일

```text
05_training/adapters/daegu_signal_csv_preflight_v2.md
05_training/adapters/run_daegu_signal_csv_preflight_v2.py
05_training/adapters/patch_signal_feature_builder_for_daegu_csv_v2.py
05_training/adapters/test_daegu_signal_csv_preflight_v2.py
```

기존 Step 98 builder도 대구 신호등 CSV 한국어 컬럼 지원을 위해 patch됩니다.

```text
05_training/adapters/build_signal_features_for_causal_simulator_v2.py
```
