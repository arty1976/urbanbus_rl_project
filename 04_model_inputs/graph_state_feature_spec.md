# Graph State Feature Specification (Phase 5)

## 📌 개요

본 문서는 UrbanBus RL 프로젝트 Phase 5의 **상태 공간(State Space)** 입력을 공식적으로 고정하고 명세하기 위한 문서입니다. 이 규격은 `graph_state_timeslice` 테이블을 기반으로 GAT 및 RL 에이전트가 학습할 PyTorch 데이터셋(Tensor) 생성의 기준점이 됩니다.

**[주의 사항]**

* **Node Temporal Feature 중심**: 현재 Phase 5는 `graph_edge_master` 연결성을 이용한 Graph Structural Feature(구조 특성)보다, 노드의 시간에 따른 흐름을 예측/분석하는 **Node Temporal Feature(노드 시계열 특성)** 중심으로 설계되었습니다.
* **노드 타입 상수화**: `node_type`은 현재 사실상 `STOP` 단일값이므로, 모델 입력시 상수(Constant)로 취급하거나 제외가 가능합니다.
* **대리 지표 활용**: `waiting_passenger_cnt`는 실제 정류장에서 관측된 대기열(Queue)이 아니라, 이전에 누적된 승하차 차이를 기반으로 추론한 프록시(Proxy) 근사값입니다.

---

## 🏗️ 1. 즉시 사용 가능 핵심 입력 (Group A)

현재 `graph_state_timeslice` 에 100% 무결성으로 채워져 있으며 (결측치 없음), **즉시 모델 입력 자질(Feature)** 로 활용되는 핵심 컬럼입니다.

| Column Name | Meaning | SQL Type | Feature Type | Preprocessing (추천) | Null | Use? |
| --- | --- | --- | --- | --- | --- | --- |
| `boardings_recent` | 버킷 내 총 승차 인원 | INT | Continuous | Min-Max 스케일링 혹은 Log1p | 불가 | O |
| `alightings_recent` | 버킷 내 총 하차 인원 | INT | Continuous | Min-Max 스케일링 혹은 Log1p | 불가 | O |
| `waiting_passenger_cnt` | 잔여 대기 인원 프록시 | INT | Continuous | Min-Max 스케일링 혹은 Log1p | 불가 | O |
| `hour_of_day` | 측정 시간대 (0~23) | INT | Categorical | One-hot 인코딩 또는 Sin/Cos 주기 임베딩 | 불가 | O (Context) |
| `day_of_week` | 요일 (0:일 ~ 6:토) | INT | Categorical | One-hot 인코딩 또는 Sin/Cos 주기 임베딩 | 불가 | O (Context) |
| `is_peak` | 출퇴근 등 첨두시간 여부 | BOOLEAN | Boolean | 0 또는 1 매핑 | 불가 | O (Context) |

---

## ⏸️ 2. 후속 확장용 보류 컬럼 (Group B)

스키마에는 존재하나 현재 소스 부재, 알고리즘 구현 대기 등으로 인해 `NULL` 값 또는 Placeholder 상태인 컬럼들입니다. 현재 PyTorch Tensor 변환 **입력에서 제외(수동 차단)** 되어야 합니다.

| Column Name | Meaning | Feature Target | 연계 계획 | Use? |
| --- | --- | --- | --- | --- |
| `predicted_demand_10m` | 단기(10분) 수요 예측량 | Continuous | RL 보상/예측 서브 타릿으로 후속 확장 | X |
| `predicted_demand_30m` | 중기(30분) 수요 예측량 | Continuous | RL 보상/예측 서브 타릿으로 후속 확장 | X |
| `nearest_bus_eta_sec` | 최근접 버스 도착 예상 시간 | Continuous | 실시간 위치 서버 연동 후 확장 | X |
| `active_bus_cnt_nearby` | 근방 활성화 버스 대수 | Continuous | 실시간 위치 서버 연동 후 확장 | X |
| `link_travel_time_sec` | 링크 통행 시간 | Continuous | 도로 소통 정보 연동 후 확장 | X |
| `link_speed_kmh` | 링크 통행 속도 | Continuous | 도로 소통 정보 연동 후 확장 | X |
| `link_congestion_index` | 링크 혼잡도 인덱스 | Continuous / Ordered | 도로 소통 정보 연동 후 확장 | X |
| `link_flow_proxy` | 간선 대기열 흐름 프록시 | Continuous | 도로 소통 정보 연동 후 확장 | X |
| `incident_flag` | 돌발상황(사고/공사) 플래그 | Boolean | 공공데이터 통제망 연계 후 확장 | X |

---

## 🔑 3. 메타 / 키 컬럼 (Group C)

관계를 매핑하고 데이터를 필터링하기 위한 메타데이터입니다. 모델 학습의 물리적 입력 텐서(`x`)로는 들어가지 않으나, 미니배치 분할 / 타임스텝 분리를 위해 데이터 로더에서 사용됩니다.

| Column Name | Meaning | 로더 활용 방안 |
| --- | --- | --- |
| `state_ts` | 텐서의 `t` 인덱스(시간축) 기반값 | 시계열의 Timeline 순서 고정 |
| `node_uid` | 정류장/링크 고유 식별자 | Graph Nodes 리스트 인덱싱(Embedding 매핑) |
| `node_type` | 노드 분류 (현재 'STOP' 전용) | GAT Heterogeneous Graph Type 라우팅 (보류) |
| `created_at` | 적재 로그 (시스템용) | 학습 사용 안 함 |
| `updated_at` | 적재 로그 (시스템용) | 학습 사용 안 함 |
