# Step 100 — Tensor DB + Signal Feature Availability Integration

## 0. 목적

Step 97은 데이터 사용 가능성을 분류했고, Step 98은 정적 신호 feature builder를 만들었으며, Step 99는 실제 대구 신호등 CSV와 tensor graph를 결합하여 feature 산출물을 생성했다.

Step 100은 이 세 단계를 통합하여 다음을 고정한다.

1. 어떤 feature가 실제 산출물에 존재하는가
2. 어떤 feature가 causal simulator v2에 투입 가능한가
3. 어떤 feature는 proxy로만 허용되는가
4. 어떤 feature는 현재 금지되어야 하는가
5. 다음 CausalSimulatorAdapter v2가 읽어야 할 계약은 무엇인가

## 1. 현재 완료 상태

| Step | 상태 |
|---|---|
| Step 97 | Tensor DB + signal CSV data availability audit 완료 |
| Step 98 | Signal feature builder self-test 완료 |
| Step 99 | 실제 대구 신호등 CSV 기반 node/edge signal feature 생성 완료 |

Step 99 기준 산출물:

```text
artifacts/signal_features_v2/node_signal_features.parquet
artifacts/signal_features_v2/edge_signal_features.parquet
artifacts/signal_features_v2/tensor_signal_feature_contract_v2.json
artifacts/signal_features_v2/signal_feature_quality_report.json
```

## 2. 통합 원칙

허용:

- 정적 위치 기반 신호 수
- 주변 신호 밀도
- 가장 가까운 신호까지 거리
- 보행 신호 수
- 점멸 신호 비율
- 제어 신호 비율
- `_proxy` suffix가 붙은 복잡도/위험도 proxy

금지:

- red-light delay
- green time
- cycle length
- phase sequence
- signal offset
- real-time signal state
- queue discharge rate

## 3. Causal simulator v2에서의 사용 위치

| 위치 | 허용 여부 | 설명 |
|---|---|---|
| actor observation context | 가능 | 정책이 정적 인프라 난이도를 인식 |
| critic/global context | 가능 | 환경 복잡도 반영 |
| edge_attr extension | 가능 | graph edge signal context |
| reward 직접항 | 원칙적 보류 | proxy 오용 위험 |
| 동적 phase simulator | 금지 | phase/timing 자료 없음 |

## 4. 다음 작업

Step 101:

```text
CausalSimulatorAdapter v2 scaffold
```

목표:

- Step 100 계약을 읽는다.
- node/edge signal feature artifact를 로드한다.
- observation에 정적 signal context를 붙인다.
- 동적 신호 지연을 생성하지 않는다.
- performance_claim_allowed=false 상태를 유지한다.
