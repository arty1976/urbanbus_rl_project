# Step 100 — Causal Simulator v2 Input Contract

## 0. 목적

Step 100의 목적은 Step 97~99에서 확인한 tensor DB(Database=데이터베이스) + 대구 신호등 CSV(Comma-Separated Values=쉼표 구분값) 기반 feature를 `causal simulator v2`의 입력 계약으로 고정하는 것이다.

이 계약은 성능 주장 문서가 아니다. 현재 결과는 여전히 toy/smoke 및 정적 feature preparation 단계이다.

## 1. 입력 feature family

### 1.1 Tensor DB direct features

허용:

- graph skeleton
- node identity
- edge identity
- edge distance/time/cost
- snapshot time context
- observed demand context
- scenario/window metadata

### 1.2 Static signal infrastructure features

허용:

- `signal_count_100m`
- `signal_count_250m`
- `signal_count_500m`
- `nearest_signal_distance_m`
- `pedestrian_signal_count_250m`
- `blink_signal_ratio_250m`
- `controlled_signal_ratio_250m`
- `edge_signal_count`
- `edge_signal_density_per_km`
- `edge_nearest_signal_distance_m`

### 1.3 Explicit proxy features

허용하되 반드시 proxy로 표기:

- `signal_delay_risk_proxy`
- `intersection_complexity_proxy`
- `edge_control_complexity_proxy`

이 값들은 실제 신호 지연 측정값이 아니다.

## 2. 금지 feature

현재 데이터만으로는 아래 feature를 causal simulator v2에 넣으면 안 된다.

- `red_light_delay_seconds`
- `green_time_seconds`
- `cycle_length_seconds`
- `phase_sequence`
- `signal_offset_seconds`
- `real_time_signal_state`
- `queue_discharge_rate`
- `lane_level_turning_movement`

## 3. Causal simulator v2 admission rule

| feature class | admission |
|---|---|
| tensor_db_direct | allowed |
| signal_csv_direct_static | allowed |
| tensor_signal_derived_static | allowed |
| explicit_proxy | allowed with `_proxy` suffix |
| dynamic_signal_phase | rejected |
| red_light_delay / green_time / cycle length | rejected |

## 4. Claim guardrail

아래 값은 false로 유지한다.

```json
{
  "trained_model": false,
  "performance_claim_allowed": false,
  "causal_performance_claim_allowed": false,
  "dynamic_signal_phase_claim_allowed": false,
  "red_light_delay_claim_allowed": false,
  "green_time_claim_allowed": false,
  "cycle_length_claim_allowed": false,
  "daegu_citywide_performance_claim_allowed": false,
  "fleet_reduction_claim_allowed": false
}
```

## 5. Step 100 산출물

```text
artifacts/causal_simulator_v2_contract/causal_simulator_v2_input_contract.json
artifacts/causal_simulator_v2_contract/causal_simulator_v2_feature_manifest.json
artifacts/causal_simulator_v2_contract/causal_simulator_v2_contract_report.md
```

## 6. 다음 단계

Step 101에서는 이 계약을 기반으로 `CausalSimulatorAdapter v2` scaffold를 작성한다.
