# Step 104 — Causal Simulator v2 Static Signal Runbook

## 0. 목적

이 runbook은 Step 97~103에서 구축한 **static signal-aware causal simulator v2 smoke pipeline**을 재현하고 해석하기 위한 문서이다.

현재 결과는 성능 실험이 아니다.  
현재 결과는 **정적 신호 인프라 feature가 causal simulator v2 pipeline에 연결되는지 확인한 smoke validation**이다.

---

## 1. 현재 완료된 흐름

```text
Step 97:
Tensor DB + signal CSV data availability audit

Step 98:
Signal feature builder for causal simulator v2

Step 99:
Actual Daegu signal CSV + tensor DB signal feature generation

Step 100:
Causal simulator v2 input contract integration

Step 101:
CausalSimulatorAdapter v2 scaffold

Step 102:
CausalSimulatorAdapter v2 rollout writer smoke

Step 103:
Causal simulator v2 rollout → canonical KPI aggregator smoke
```

전체 연결 흐름:

```text
대구 신호등 CSV
→ node/edge static signal features
→ causal simulator v2 input contract
→ CausalSimulatorAdapter v2
→ raw_events.parquet
→ window_rollup.parquet
→ canonical_kpi_aggregator.py
→ kpi_by_window.parquet / kpi_by_seed.parquet / kpi_overall.json
```

---

## 2. Step별 의미

### Step 97 — Data availability audit

목적:

- tensor DB(Database=데이터베이스)에 직접 있는 데이터 분류
- signal CSV(Comma-Separated Values=쉼표 구분값)에서 직접 얻을 수 있는 데이터 분류
- tensor DB + signal CSV 결합으로 생성 가능한 feature 분류
- proxy로만 가능한 데이터와 현재 불가능한 데이터 분류

핵심 결론:

```text
신호등 CSV는 동적 신호 phase simulator 원천이 아니다.
정적 signal infrastructure feature source로만 사용한다.
```

---

### Step 98 — Signal feature builder

목적:

- 신호등 CSV를 node/edge graph와 결합하는 builder 작성
- synthetic self-test 통과

생성 가능 feature:

```text
signal_count_100m
signal_count_250m
signal_count_500m
nearest_signal_distance_m
pedestrian_signal_count_250m
blink_signal_ratio_250m
controlled_signal_ratio_250m
edge_signal_count
edge_signal_density_per_km
edge_nearest_signal_distance_m
```

proxy feature:

```text
signal_delay_risk_proxy
intersection_complexity_proxy
edge_control_complexity_proxy
```

---

### Step 99 — Actual Daegu signal CSV feature generation

목적:

- 실제 대구 신호등 CSV를 프로젝트 data source로 복사
- CSV preflight 수행
- DB node/edge graph와 결합
- 실제 node/edge signal feature 산출

생성 산출물:

```text
artifacts/signal_features_v2/node_signal_features.parquet
artifacts/signal_features_v2/edge_signal_features.parquet
artifacts/signal_features_v2/tensor_signal_feature_contract_v2.json
artifacts/signal_features_v2/signal_feature_quality_report.json
```

성공 기준:

```text
node_rows = 4116
edge_rows = 5484
```

주의:

```text
node_index 컬럼명 차이로 인해 실제 DB의 node_idx를 node_index로 alias하는 patch를 적용했다.
```

---

### Step 100 — Causal simulator v2 input contract

목적:

- Step 99 feature를 causal simulator v2 입력 계약으로 통합
- 허용 feature와 금지 feature를 계약 JSON/manifest로 고정

생성 산출물:

```text
artifacts/causal_simulator_v2_contract/causal_simulator_v2_input_contract.json
artifacts/causal_simulator_v2_contract/causal_simulator_v2_feature_manifest.json
artifacts/causal_simulator_v2_contract/causal_simulator_v2_contract_report.md
```

---

### Step 101 — CausalSimulatorAdapter v2 scaffold

목적:

- Step 100 contract 읽기
- node/edge signal feature artifact 읽기
- observation에 static signal context 연결
- reset/step 가능성 확인

생성 파일:

```text
05_training/adapters/causal_simulator_v2_adapter.py
05_training/adapters/test_causal_simulator_v2_adapter.py
05_training/run_causal_simulator_v2_adapter_smoke.py
```

실제 smoke 결과:

```text
artifacts/causal_simulator_v2_adapter_smoke/status.json
```

성공 기준:

```text
graph nodes = 4116
graph edges = 5484
```

---

### Step 102 — Rollout writer smoke path

목적:

- CausalSimulatorAdapter v2 step 결과를 실험 산출물로 저장
- raw_events와 window_rollup 생성

생성 산출물:

```text
artifacts/causal_simulator_v2_rollout_smoke/rollout_manifest.json
artifacts/causal_simulator_v2_rollout_smoke/C2_STATIC_SIGNAL/rollouts/seed_101/raw_events.parquet
artifacts/causal_simulator_v2_rollout_smoke/C2_STATIC_SIGNAL/rollouts/seed_101/window_rollup.parquet
```

의미:

```text
raw_events.parquet:
agent별 step 기록

window_rollup.parquet:
canonical KPI 집계기로 넘길 수 있는 window 단위 요약
```

---

### Step 103 — Canonical KPI aggregator smoke

목적:

- Step 102의 `window_rollup.parquet`를 `canonical_kpi_aggregator.py`에 연결
- canonical KPI 산출물이 생성되는지 확인

생성 산출물:

```text
artifacts/causal_simulator_v2_canonical_kpi_smoke/prepared_input/
artifacts/causal_simulator_v2_canonical_kpi_smoke/scenario_index.parquet
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_eval/kpi_by_window.parquet
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_eval/kpi_by_seed.parquet
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_eval/kpi_by_time_band.parquet
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_eval/kpi_overall.json
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_kpi_smoke_manifest.json
```

성공 기준:

```text
kpi_by_window rows = 1
kpi_by_seed rows = 1
causal_allowed = False
SMOKE PASS
```

---

## 3. Claim guardrails

현재 반드시 유지해야 하는 상태:

```text
trained_model = false
performance_claim_allowed = false
causal_performance_claim_allowed = false
dynamic_signal_phase_claim_allowed = false
red_light_delay_claim_allowed = false
green_time_claim_allowed = false
cycle_length_claim_allowed = false
daegu_citywide_performance_claim_allowed = false
fleet_reduction_claim_allowed = false
```

---

## 4. 허용되는 표현

사용 가능:

```text
static signal infrastructure-aware causal simulator v2 smoke pipeline이 연결되었다.
대구 신호등 CSV 기반 node/edge static signal feature가 생성되었다.
CausalSimulatorAdapter v2가 static signal feature를 observation/edge_attr로 읽을 수 있다.
rollout smoke 결과가 canonical KPI aggregator까지 연결되었다.
```

---

## 5. 금지되는 표현

금지:

```text
실제 대구 전역 버스 운영 성능이 검증되었다.
MAPPO 정책이 신호등을 고려해 성능 향상을 보였다.
fleet reduction 효과가 입증되었다.
실제 red-light delay가 반영되었다.
실제 green time/cycle length/signal phase가 반영되었다.
```

---

## 6. 현재 feature 해석

정확한 해석:

```text
signal_count_250m:
정류장 주변 250m 내 신호등 수

nearest_signal_distance_m:
가장 가까운 신호등까지 거리

signal_delay_risk_proxy:
신호 인프라가 많고 가까울수록 지연 위험이 높을 수 있다는 정적 proxy

intersection_complexity_proxy:
주변 신호 인프라 복잡도 proxy
```

부정확한 해석:

```text
signal_delay_risk_proxy = 실제 신호 지연 시간
nearest_signal_distance_m = 실제 적색 정지 위치
edge_signal_count = 실제 주행 경로상 모든 신호등 수
```

---

## 7. 재실행 순서

### 7.1 실제 신호 feature 생성 재실행

```powershell
powershell -ExecutionPolicy Bypass -File ".\step99_daegu_signal_csv_preflight_package\Step99_RunBuilderWithDb.ps1"
```

### 7.2 Step 100 contract 재생성

```powershell
powershell -ExecutionPolicy Bypass -File ".\Step100_CausalSimulatorV2Contract_Standalone.ps1" -RunSelfTest
```

### 7.3 Step 101 adapter smoke 재실행

```powershell
powershell -ExecutionPolicy Bypass -File ".\step101_causal_simulator_v2_adapter_package\Step101_RunActualAdapterSmoke.ps1"
```

### 7.4 Step 102 rollout smoke 재실행

```powershell
powershell -ExecutionPolicy Bypass -File ".\step102_causal_simulator_v2_rollout_package\Step102_RunActualRolloutSmoke.ps1"
```

### 7.5 Step 103 canonical KPI smoke 재실행

```powershell
powershell -ExecutionPolicy Bypass -File ".\step103_causal_v2_canonical_kpi_package\Step103_RunActualCanonicalKpiSmoke.ps1"
```

---

## 8. 다음 추천 단계

```text
Step 105:
Causal v2 canonical KPI smoke inspector

목표:
- Step 103의 kpi_by_window/kpi_by_seed/kpi_overall 내용을 사람이 읽기 쉽게 요약
- guardrail이 false인지 다시 확인
- canonical KPI 6종과 Phase 2 확장 KPI 후보가 어떤 상태인지 점검
```
