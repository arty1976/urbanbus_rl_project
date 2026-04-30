# Step 103 Causal-v2 Canonical KPI Smoke Package

압축을 반드시 풀고, 압축푼 폴더 안의 `Step103_CausalV2CanonicalKpi_Setup.ps1`을 실행하세요.

## 실행

```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project"

powershell -ExecutionPolicy Bypass -File "압축푼경로\Step103_CausalV2CanonicalKpi_Setup.ps1" -RunSelfTest
```

## 실제 Step 102 artifact에 대해 재실행

```powershell
powershell -ExecutionPolicy Bypass -File "압축푼경로\Step103_RunActualCanonicalKpiSmoke.ps1"
```

## 생성 파일

```text
05_training/adapters/causal_simulator_v2_canonical_kpi_smoke_contract.md
05_training/adapters/test_causal_simulator_v2_canonical_kpi_smoke.py
05_training/run_causal_simulator_v2_canonical_kpi_smoke.py
```

## 생성 artifact

```text
artifacts/causal_simulator_v2_canonical_kpi_smoke/contract/c2_static_signal_canonical_contract.json
artifacts/causal_simulator_v2_canonical_kpi_smoke/prepared_input/...
artifacts/causal_simulator_v2_canonical_kpi_smoke/scenario_index.parquet
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_eval/kpi_by_window.parquet
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_eval/kpi_by_seed.parquet
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_eval/kpi_by_time_band.parquet
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_eval/kpi_overall.json
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_kpi_smoke_manifest.json
```
