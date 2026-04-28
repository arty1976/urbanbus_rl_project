# Step 97 — Tensor DB + Signal CSV Data Availability Audit

## 0. 목적

이 문서는 `causal simulator v2` 계약을 바로 작성하기 전에, 현재 사용 가능한 데이터의 한계를 먼저 고정하기 위한 audit 문서이다.

핵심 질문은 다음이다.

1. 현재 tensor DB(Database=데이터베이스)에 직접 존재하는 데이터는 무엇인가?
2. 대구 신호등 CSV(Comma-Separated Values=쉼표 구분값)에서 직접 얻을 수 있는 데이터는 무엇인가?
3. tensor DB + signal CSV 결합으로 생성 가능한 feature는 무엇인가?
4. proxy로만 가능한 데이터는 무엇인가?
5. 현재 절대 만들 수 없는 데이터는 무엇인가?
6. causal simulator v2에 넣어도 되는 데이터와 넣으면 안 되는 데이터는 무엇인가?

---

## 1. 현재 claim guardrail

현재 결과는 toy causal smoke validation이다.

| 항목 | 값 |
|---|---|
| `trained_model` | `false` |
| `performance_claim_allowed` | `false` |
| `smoke_training_only` | `true` |
| `smoke_evaluation_only` | `true` |
| `matrix_smoke_evaluation_only` | `true` |
| 논문 성능 주장 | 금지 |
| 대구 전역 성능 주장 | 금지 |
| 실제 fleet reduction 주장 | 금지 |

따라서 Step 97의 산출물도 **성능 증명 문서가 아니라 데이터 사용 가능성 분류 문서**이다.

---

## 2. Tensor DB에 직접 존재하는 데이터

현재 GATv2(Graph Attention Network version 2=그래프 어텐션 네트워크 버전 2) / MAPPO(Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) 경로에서 직접 사용할 수 있는 데이터는 아래와 같다.

### 2.1 Node identity / graph population

| 데이터 | 사용 가능성 | 설명 |
|---|---|---|
| `node_uid` | 직접 가능 | 정류장 노드 고유 ID(Identifier=식별자) |
| `node_index` | 직접 가능 | PyG(PyTorch Geometric=파이토치 지오메트릭) dense index |
| `num_nodes` | 직접 가능 | 현재 canonical 기준 4,116 노드 |
| node mask | 직접 가능 | 관측 row가 없는 노드 구분 |

### 2.2 Static edge skeleton

| 데이터 | 사용 가능성 | 설명 |
|---|---|---|
| `src_idx` | 직접 가능 | edge source node index |
| `dst_idx` | 직접 가능 | edge destination node index |
| `distance_m` | 직접 가능 | 정류장 간 거리 |
| `time_sec` | 직접 가능 | 평균 속도 가정 기반 이동 시간 |
| `generalized_cost` | 직접 가능 | graph edge cost |
| `long_edge_5km_flag` | 직접 가능 | 장거리 edge flag |

### 2.3 Snapshot demand / time context

| 데이터 | 사용 가능성 | 설명 |
|---|---|---|
| `boardings_recent_log` | 직접 가능 | 현재 승차 수요 log feature |
| `alightings_recent_log` | 직접 가능 | 현재 하차 수요 log feature |
| `waiting_passenger_cnt_log` | 직접 가능 | 현재 대기 승객 log feature |
| `hour_sin` | 직접 가능 | 시간 cyclic encoding |
| `hour_cos` | 직접 가능 | 시간 cyclic encoding |
| `is_peak` | 직접 가능 | 첨두 여부 |
| `next_boardings_recent` | 직접 가능 | supervised target |
| `next_alightings_recent` | 직접 가능 | supervised target |
| `next_waiting_passenger_cnt` | 직접 가능 | supervised target |

### 2.4 Scenario / evaluation context

| 데이터 | 사용 가능성 | 설명 |
|---|---|---|
| `state_ts` | 직접 가능 | 상태 시각 |
| `next_state_ts` | 직접 가능 | 다음 관측 시각 |
| `service_date` | 직접 가능 | 운행일 |
| `time_band` | 직접 가능 | peak/offpeak/night |
| `window_id` | 직접 가능 | 평가 window index |
| 12-KPI schema | 부분 가능 | toy causal smoke pipeline 기준으로 schema 확장됨 |

---

## 3. Signal CSV에서 직접 얻을 수 있는 데이터

신호등 CSV는 **동적 신호 phase simulation 원천이 아니라 정적 signal infrastructure source**로 취급한다.

### 3.1 직접 사용 가능한 후보

| 데이터 | 사용 가능성 | 설명 |
|---|---|---|
| signal ID | 직접 가능 | 신호등 개체 식별자 |
| signal name/type | 직접 가능 | 신호등 명칭 또는 종류 |
| latitude/longitude | 직접 가능 | 위경도 좌표가 있는 경우 |
| x/y projected coordinate | 직접 가능 | EPSG:5187 등 투영 좌표가 있는 경우 |
| pedestrian signal flag | 직접 가능 | 보행자 신호 여부가 컬럼으로 있거나 type에서 추론 가능한 경우 |
| blink signal flag | 직접 가능 | 점멸 신호 여부 |
| controlled signal flag | 직접 가능 | 제어 신호 여부 |
| installation location | 직접 가능 | 설치 위치 텍스트 |

### 3.2 직접 사용하면 안 되는 해석

CSV에 신호등 위치만 있다고 해서 아래 값을 직접 얻었다고 보면 안 된다.

| 데이터 | 판정 |
|---|---|
| red-light delay | 불가능 |
| green time | 불가능 |
| cycle length | 불가능 |
| phase sequence | 불가능 |
| signal offset | 불가능 |
| 실시간 신호 상태 | 불가능 |
| queue discharge rate | 불가능 |

---

## 4. Tensor DB + signal CSV 결합으로 생성 가능한 feature

Step 98에서 만들 수 있는 feature는 다음 범위까지이다.

### 4.1 Node-level signal infrastructure feature

| Feature | 생성 가능성 | causal simulator v2 투입 |
|---|---|---|
| `signal_count_100m` | 가능 | 가능 |
| `signal_count_250m` | 가능 | 가능 |
| `signal_count_500m` | 가능 | 가능 |
| `nearest_signal_distance_m` | 가능 | 가능 |
| `pedestrian_signal_count_250m` | 가능 | 가능 |
| `blink_signal_ratio_250m` | 가능 | 가능 |
| `controlled_signal_ratio_250m` | 가능 | 가능 |
| `signal_density_per_km` | 가능 | 가능 |

### 4.2 Edge-level signal infrastructure feature

| Feature | 생성 가능성 | causal simulator v2 투입 |
|---|---|---|
| `edge_signal_count` | 가능 | 가능 |
| `edge_signal_density_per_km` | 가능 | 가능 |
| `edge_nearest_signal_distance_m` | 가능 | 가능 |
| `edge_pedestrian_signal_count` | 가능 | 가능 |
| `edge_controlled_signal_ratio` | 가능 | 가능 |

단, edge-level feature는 edge geometry가 없으면 `src/dst node 주변 신호의 합성 proxy`로 시작해야 한다.

---

## 5. Proxy로만 가능한 데이터

아래 feature는 실제 측정값이 아니라 proxy로만 허용한다.

| Feature | 의미 | 조건 |
|---|---|---|
| `signal_delay_risk_proxy` | 신호가 많아 지연 위험이 높을 가능성 | 반드시 `_proxy` suffix 유지 |
| `intersection_complexity_proxy` | 교차로/신호 밀집 복잡도 | 실제 교차로 용량으로 주장 금지 |
| `edge_control_complexity_proxy` | edge 주변 제어 인프라 복잡도 | red-light delay로 해석 금지 |
| `stop_access_friction_proxy` | 정류장 접근 마찰도 | 보행 신호 존재 기반 proxy |

---

## 6. 현재 절대 불가능한 데이터

현재 tensor DB + static signal CSV만으로는 아래 데이터를 만들 수 없다.

| 데이터 | 이유 |
|---|---|
| `red_light_delay_seconds` | phase/timing/arrival trajectory 없음 |
| `green_time_seconds` | 신호 현시 자료 없음 |
| `cycle_length_seconds` | 신호 운영 주기 없음 |
| `signal_offset_seconds` | 신호 연동 offset 자료 없음 |
| `real_time_signal_state` | 실시간 신호 상태 없음 |
| `queue_discharge_rate` | 차로별 queue/flow 관측 없음 |
| `lane_level_turning_movement` | 차로·회전 교통량 없음 |
| `actual_vehicle_trajectory` | 버스 GPS(Global Positioning System=위성항법시스템) trajectory 없음 |
| `incident_event_stream` | 사고 이벤트 원천 없음 |
| `weather_event_stream` | 기상 이벤트 원천 없음 |

---

## 7. Causal simulator v2 투입 허용/금지 기준

### 7.1 넣어도 되는 데이터

- tensor DB direct feature
- signal CSV direct static infrastructure feature
- tensor DB + signal CSV 결합으로 만든 static neighborhood feature
- `_proxy` suffix가 붙은 명시적 proxy feature

### 7.2 아직 넣으면 안 되는 데이터

- red-light delay
- green time
- cycle length
- signal phase
- offset
- real-time signal state
- dynamic queue discharge
- phase-aware travel time

---

## 8. Step 98 signal feature builder 입력/출력 contract 초안

### 8.1 입력

| 입력 | 필수 여부 | 설명 |
|---|---|---|
| `--signal-csv` | 필수 | 대구 신호등 CSV 경로 |
| `--db-url` | 선택 | tensor DB DSN(Data Source Name=데이터 원본 이름) |
| `--node-source` | 필수 | node table/view 또는 parquet |
| `--edge-source` | 필수 | edge table/view 또는 parquet |
| `--coord-crs` | 필수 | `EPSG:4326`, `EPSG:5187`, 또는 manual |

### 8.2 출력

| 출력 | 설명 |
|---|---|
| `node_signal_features.parquet` | node-level static signal features |
| `edge_signal_features.parquet` | edge-level static signal features |
| `tensor_signal_feature_contract_v2.json` | feature 정의 및 source hash |
| `signal_feature_quality_report.json` | 결측/좌표/매칭 품질 보고 |

### 8.3 최소 feature set

```text
node_signal_features:
- node_uid
- node_index
- signal_count_250m
- nearest_signal_distance_m
- pedestrian_signal_count_250m
- blink_signal_ratio_250m
- controlled_signal_ratio_250m
- signal_feature_quality_flag

edge_signal_features:
- src_idx
- dst_idx
- edge_signal_count
- edge_signal_density_per_km
- edge_nearest_signal_distance_m
- edge_signal_feature_quality_flag
```

---

## 9. Step 97 최종 판정

현재 신호 CSV는 **동적 신호 phase simulator 원천이 아니라 정적 signal infrastructure feature source**로 사용한다.

따라서 Step 98은 다음처럼 진행한다.

```text
Step 98:
Signal feature builder for causal simulator v2

목표:
signal CSV를 tensor graph node/edge에 결합하여
static signal infrastructure feature를 만든다.

금지:
red-light delay, green time, cycle length, phase, offset을 임의 생성하지 않는다.
```
