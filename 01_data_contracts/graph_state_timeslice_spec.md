# Graph State Timeslice Spec

## 1. 목적
본 문서는 통합 그래프 마스터(Graph Master) 위에 시간축 상태 데이터를 정의하는 `graph_state_timeslice` 테이블의 설계를 정의한다. 이 테이블은 GAT(Graph Attention Network) 및 RL(Reinforcement Learning) 모델의 입력 상태(State)로 사용되는 핵심 피처 레이어이다.

## 2. 시간 해상도 (Temporal Resolution)
- **기본 시간 버킷**: **10분** (`state_ts`)
- **버킷 산정 방식**: `YYYY-MM-DD HH:MM:00` 형식의 TIMESTAMP (시작 시각)
- **Hourly Allocation (시간당 할당)**: 
    - 현재 원천 데이터(`fact_stop_usage_hourly`)가 1시간 단위인 경우, 해당 1시간의 값을 각 10분 버킷에 동일 비율로 배분(예: `board_count / 6`)하여 적재한다.
    - 이를 통해 10분 단위의 **파생 버킷(Derived Buckets)**을 생성하며, 향후 실시간 집계 파이프라인이 도입되면 이를 실제 관측값으로 대체한다.

## 3. 노드 유형별 상태 해석
노드 유형(`node_type`)에 따라 동일한 컬럼도 다르게 해석될 수 있다.

### 3.1 STOP 노드 (수요 측면)
- **의미**: 승객의 대기, 승차, 하차가 일어나는 지점의 서비스 상태.
- **주요 컬럼**: `waiting_passenger_cnt`, `boardings_recent`, `alightings_recent`.

### 3.2 LINK 노드 (공급/이동 측면)
- **의미**: 버스 차량이 이동 중인 물리적 구간의 교통 소통 상태.
- **주요 컬럼**: `link_travel_time_sec`, `link_speed_kmh`, `link_congestion_index`.

---

## 4. 테이블 정의: public.graph_state_timeslice

| 컬럼명 | 타입 | 설명 | 종류 |
| :--- | :--- | :--- | :--- |
| **state_ts** | TIMESTAMPTZ | 상태 버킷 시작 시각 (PK) | Key |
| **node_uid** | TEXT | 통합 노드 식별자 (PK, FK) | Key |
| node_type | TEXT | 노드 유형 (STOP, LINK) | Attribute |
| waiting_passenger_cnt | INTEGER | 현재 정류장 대기 추정 인원 | STOP Metric |
| boardings_recent | INTEGER | 직전 10분간 승차 인원 (Hourly Allocation) | STOP Metric |
| alightings_recent | INTEGER | 직전 10분간 하차 인원 (Hourly Allocation) | STOP Metric |
| predicted_demand_10m | INTEGER | 향후 10분 수요 예측값 (AI 모델용) | AI Feature |
| predicted_demand_30m | INTEGER | 향후 30분 수요 예측값 (AI 모델용) | AI Feature |
| nearest_bus_eta_sec | INTEGER | 가장 가까운 버스 도착 예정 시간 (초) | Real-time |
| active_bus_cnt_nearby | INTEGER | 인근 운행 중인 버스 대수 | Real-time |
| link_travel_time_sec | NUMERIC | 링크 통과 소요 시간 (초) | LINK Metric |
| link_speed_kmh | NUMERIC | 링크 평균 주행 속도 (km/h) | LINK Metric |
| link_congestion_index | NUMERIC | 링크 혼잡 지표 (0.0~1.0) | LINK Metric |
| link_flow_proxy | NUMERIC | 링크 통행량 대리 지표 | LINK Metric |
| incident_flag | BOOLEAN | 사고/정체 이벤트 발생 여부 | Status |
| hour_of_day | INTEGER | 시간대 (0~23) | Time Feature |
| day_of_week | INTEGER | 요일 (0:일, 6:토) | Time Feature |
| is_peak | BOOLEAN | 첨두 시간대(Peak Time) 여부 | Time Feature |
| created_at | TIMESTAMPTZ | 레코드 생성 시각 | System |

---

## 5. 후속 연계 계획
1. **Passenger Demand**: 현재 1시간 단위 집계를 10분 단위 실시간 결제 정보와 동기화.
2. **Traffic State**: 링크별 실시간 속도(VDS, C-ITS 소스) 연계.
3. **Vehicle State**: 버스 위치 정보(BMS/BIS)를 통한 실시간 ETA 및 주변 차량 수 계산.
4. **Predictive Layer**: 수요 예측 모델의 출력을 `predicted_demand_*` 컬럼에 백필.
