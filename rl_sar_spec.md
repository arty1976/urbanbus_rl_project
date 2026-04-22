# rl_sar_spec.md

## 1. 문서 목적

본 문서는 Urban Bus GAT-RL (Graph Attention Network-Reinforcement Learning=그래프 어텐션 네트워크-강화학습) 연구 파이프라인의 SAR (State-Action-Reward=상태-행동-보상) 명세를 고정한다.

본 명세는 현재 승인된 데이터 기반을 전제로 한다.

- `graph_state_timeslice`
- `rl_state_training_base`

현재 범위는 `STOP` 노드 중심 상태(state=상태) 구성과 전이(transition=전이) 학습 준비 단계까지다.  
`graph_edge_master` 는 아직 복구되지 않았으므로, 본 문서는 다음 둘을 구분한다.

1. **현재 즉시 실행 가능한 명세**
2. **향후 그래프 확장 명세**

---

## 2. 현재 확정 데이터 기반

### 2.1 승인 완료 상위 테이블

#### `graph_state_timeslice`

- row count: `21,615,863`
- readiness: `APPROVED`
- `gat_rl_input_ready = YES`

#### `rl_state_training_base`

- row count: `21,612,482`
- readiness: `APPROVED`
- `rl_training_base_ready = YES`

### 2.2 핵심 해석

`next_state_ts` 는 strict one-hour future(정확히 1시간 뒤 미래 시각)가 아니다.  
의미는 **next observed timestamp(다음 관측 시각)** 이다.

따라서 상태 전이 간격은 고정 1시간이 아니라 운영 관측 기반의 불규칙 간격이며, 반드시 시간 간격 변수로 모델에 반영해야 한다.

### 2.3 운영 버킷 해석

- 하루 active service buckets(운영 버킷) 수: `19`
- 운행 시간대: 대략 `05:00 ~ 23:00`
- 야간 공백 존재

즉 환경(environment=환경)은 연속적인 24시간 고정 시간축이라기보다 **운영 관측 시퀀스** 로 해석해야 한다.

---

## 3. 본 문서의 범위

본 문서는 아래를 고정한다.

- 현재 학습 가능한 상태(state=상태) 정의
- 향후 정책(policy=정책) 설계를 위한 행동(action=행동) 정의
- 향후 RL (Reinforcement Learning=강화학습) 최적화를 위한 보상(reward=보상) 정의
- 전이(transition=전이) 해석
- PyTorch (Python Torch=파이썬 기반 딥러닝 프레임워크) 입력 데이터셋 계약

본 문서는 아직 아래를 최종 확정하지 않는다.

- 버스 단위 행동 로그
- 전체 그래프 엣지(edge=엣지) 위상
- 라우팅 시뮬레이터 내부 구조
- 다중 에이전트 rollout(rollout=시뮬레이션 전개) 실행 로직

---

## 4. 환경 정의

### 4.1 의사결정 과정

환경은 MDP (Markov Decision Process=마르코프 결정 과정) 스타일의 순차 의사결정 구조로 본다. 다만 관측 간격은 불규칙하다.

- **episode(에피소드)** = 하루 운행일
- **step(스텝)** = 하나의 관측 운영 버킷
- **state transition(상태 전이)** = `state_ts -> next_state_ts`
- **time gap(시간 간격)** = `state_ts` 와 `next_state_ts` 사이 실제 경과 시간

### 4.2 전이 간격 변수

아래 파생 변수는 필수다.

- `delta_t_min = minutes(next_state_ts - state_ts)`
- `delta_t_hr = hours(next_state_ts - state_ts)`

`next_state_ts` 가 다음 관측 시각을 뜻하므로, 이 변수 없이는 야간 공백이나 비균등 전이를 잘못 해석하게 된다.

---

## 5. 상태(State=상태) 명세

### 5.1 상태 단위

현재 상태 단위는 다음과 같다.

- 하나의 `STOP` 노드
- 하나의 관측 시각
- 하나의 다음 관측 시각과 연결된 전이 행

형식적으로 현재 상태는 아래와 같다.

\[
s_{i,t}
\]

여기서

- \( i \) = stop node(정류소 노드)
- \( t \) = observed timestamp(관측 시각)

---

### 5.2 현재 원시 상태 컬럼

`rl_state_training_base` 의 현재 승인 컬럼은 아래와 같다.

- `state_ts`
- `next_state_ts`
- `node_uid`
- `node_type`
- `boardings_recent`
- `alightings_recent`
- `waiting_passenger_cnt`
- `hour_of_day`
- `day_of_week`
- `is_peak`
- `next_boardings_recent`
- `next_alightings_recent`
- `next_waiting_passenger_cnt`

---

### 5.3 현재 모델 입력 상태 벡터

권장 상태 특성 벡터(feature vector=특성 벡터)는 아래와 같다.

\[
x_{i,t} =
[
\log(1 + boardings\_recent),
\log(1 + alightings\_recent),
\log(1 + waiting\_passenger\_cnt),
hour\_sin,
hour\_cos,
dow\_sin,
dow\_cos,
is\_peak,
delta\_t\_hr,
node\_type\_stop
]
\]

#### 5.3.1 특성 정의

##### 수요 규모 특성

- `log1p_boardings_recent = log(1 + boardings_recent)`
- `log1p_alightings_recent = log(1 + alightings_recent)`
- `log1p_waiting_passenger_cnt = log(1 + waiting_passenger_cnt)`

##### 순환 시간 특성

- `hour_sin = sin(2π * hour_of_day / 24)`
- `hour_cos = cos(2π * hour_of_day / 24)`
- `dow_sin = sin(2π * day_of_week / 7)`
- `dow_cos = cos(2π * day_of_week / 7)`

##### 피크 여부

- `is_peak ∈ {0, 1}`

##### 불규칙 전이 간격

- `delta_t_hr = extract(epoch from (next_state_ts - state_ts)) / 3600`

##### 노드 타입 표시

- `node_type_stop = 1` for current `STOP` rows

---

### 5.4 특성 우선순위 해석

현재 상태 벡터는 모든 입력이 동일 중요도를 갖는 평면 목록이 아니다.  
본 연구의 현재 승인 우선순위는 아래와 같이 해석한다.

1. `waiting_passenger_cnt`
2. `is_peak`
3. `hour_of_day`, `day_of_week`
4. `boardings_recent`
5. `alightings_recent`

#### 5.4.1 `waiting_passenger_cnt`

이 변수는 **primary operational pressure signal(주요 운영 압력 신호)** 이다.  
즉 시민 불편의 직접 표현이며, 현재 stop-level(정류소 단위) 상태에서 가장 중요한 핵심 특성이다.

#### 5.4.2 `is_peak`

이 변수는 단순 시간 플래그가 아니라 **congestion-context correction flag(혼잡 문맥 보정 플래그)** 다.  
출퇴근 시간대와 같은 병목 상황에서 모델 반응을 달리하게 만드는 보정 역할을 한다.

#### 5.4.3 `hour_of_day`, `day_of_week`

이 둘은 단순 캘린더 정보가 아니라 **anticipatory temporal predictors(선제 예측 시간 변수)** 다.  
즉 승객 호출이 명시되기 전에 반복 패턴을 학습해 선행 이동을 가능하게 하는 핵심 시간 문맥이다.

#### 5.4.4 `boardings_recent`

이 변수는 **realized-demand confirmation feature(실현 수요 확인 특성)** 다.  
이미 발생한 수요 규모를 파악하는 보조 지표이며, 대기 수요 자체보다 우선순위는 낮다.

#### 5.4.5 `alightings_recent`

이 변수는 **minimum capacity-awareness feature(최소 용량 인지 특성)** 다.  
향후 좌석 여유가 언제 생기는지에 대한 간접 정보를 주지만, 현재 핵심 5개 중 상대적 중요도는 가장 낮다.

---

### 5.5 상태 설계 원칙

#### 5.5.1 왜 `log1p`

`boardings_recent`, `alightings_recent`, `waiting_passenger_cnt` 는 긴 꼬리 분포를 가지므로 `log1p` 변환이 분산을 안정화하고 극단값 지배를 줄인다.

#### 5.5.2 왜 cyclic encoding(순환 인코딩)

`hour_of_day`, `day_of_week` 는 순환 변수이므로 정수 그대로보다 `sin/cos` 표현이 더 적절하다.

#### 5.5.3 왜 `delta_t_hr`

`next_state_ts` 는 다음 관측 시각이므로, 고정 간격 전제로 해석하면 야간 공백과 비균등 전이를 잘못 이해하게 된다.

---

### 5.6 상태 가중 해석 지침

위 우선순위는 데이터베이스(DB=Database=데이터베이스) 단계에서 입력값에 임의의 곱셈을 넣으라는 뜻이 아니다.

우선순위는 아래 방식으로 반영한다.

- supervised learning(지도학습) 손실 가중치
- ablation study(제거 실험) 우선순위
- feature importance(특성 중요도) 추적
- attention analysis(어텐션 분석)
- 향후 reward shaping(보상 형성) 정렬

권장 해석은 아래와 같다.

- `waiting_passenger_cnt` 는 가장 강하게 추적하고 보호해야 하는 변수다.
- `is_peak` 는 혼잡 반응을 조정하는 문맥 게이트로 본다.
- `hour_of_day`, `day_of_week` 는 선제 이동을 위한 시간 prior(prior=사전 정보) 로 유지한다.
- `boardings_recent`, `alightings_recent` 는 보조 운영 맥락 변수로 유지한다.

---

## 6. 다음 상태 타깃 명세

현재 승인된 다음 상태 타깃은 아래 3개다.

\[
y_{i,t} =
[
next\_boardings\_recent,
next\_alightings\_recent,
next\_waiting\_passenger\_cnt
]
\]

이것은 다음 관측 시각의 정류소 단위 수요 상태를 의미한다.

### 6.1 타깃 의미

- `next_boardings_recent` = 다음 관측 시각 탑승 수요
- `next_alightings_recent` = 다음 관측 시각 하차 수요
- `next_waiting_passenger_cnt` = 다음 관측 시각 대기 승객 수

### 6.2 현재 학습 해석

현재 단계에서 이 베이스는 full policy-learning base(완전 정책학습 베이스) 가 아니라 **transition-learning base(전이학습 베이스)** 로 해석하는 것이 맞다.

즉 현재 직접 학습 목표는 아래다.

> 현재 관측 정류소 상태로부터 다음 관측 시각의 정류소 수요 상태를 예측한다.

---

## 7. 행동(Action=행동) 명세

### 7.1 현재 상태

현재 승인 데이터에는 버스 단위 행동 로그가 없다.  
따라서 현재 행동 명세는 **개념적으로 고정** 되지만 아직 full RL training(완전 강화학습 훈련) 용으로 즉시 실행되지는 않는다.

### 7.2 향후 행동 단위

향후 에이전트(agent=에이전트)는 아래처럼 정의한다.

- **one bus = one agent(버스 1대 = 에이전트 1개)**

시각 \( t \) 에서의 행동은 아래와 같다.

\[
a_{b,t} \in C_{b,t}
\]

여기서

- \( b \) = bus agent(버스 에이전트)
- \( C_{b,t} \) = 시각 \( t \) 에서 가능한 후보 정류소 집합

### 7.3 권장 행동 의미

권장 행동은 아래로 고정한다.

- **다음 목표 정류소 선택**

즉 행동 표현은 아래와 같다.

- `action_type = target_node_selection`
- `action_value = target_node_uid`

### 7.4 후보 행동 집합

향후 후보 집합은 아래를 포함할 수 있다.

- 현재 정류소 유지
- 인접 정류소 이동
- 수요 집중 정류소로 재배치
- 그래프 제약 기반 도달 가능 노드 집합

### 7.5 이 행동 설계를 택하는 이유

본 연구는 route-free on-demand public bus(노선 없는 온디맨드 공공버스) 구조를 목표로 한다.  
따라서 핵심 제어는 노선 번호 선택이 아니라 **동적 목표 정류소 선택** 이다.

이 구조는 아래와 잘 맞는다.

- graph-based routing(그래프 기반 라우팅)
- action mask(행동 마스크)
- 이동 시간 제약 기반 dispatch(배차)
- 향후 multi-agent control(다중 에이전트 제어)

---

## 8. 보상(Reward=보상) 명세

### 8.1 최종 연구용 보상

최종 RL (Reinforcement Learning=강화학습) 보상은 아래 연구 목표와 일치해야 한다.

- 대기시간 감소
- 승차 소요시간 감소
- 필요 버스 대수 감소
- 에너지 감소

권장 보상식은 아래와 같다.

\[
r_t =
- \lambda_1 \cdot served\_request\_cnt_t
- \lambda_2 \cdot waiting\_passenger\_minutes_t
- \lambda_3 \cdot invehicle\_passenger\_minutes_t
- \lambda_4 \cdot deadhead\_vehicle\_minutes_t
- \lambda_5 \cdot active\_bus\_cnt_t
- \lambda_6 \cdot rejected\_request\_cnt_t
- \lambda_7 \cdot energy\_proxy_t
\]

### 8.2 초기 계수 권장값

- `λ1 = 1.00`
- `λ2 = 0.10`
- `λ3 = 0.05`
- `λ4 = 0.02`
- `λ5 = 0.03`
- `λ6 = 0.50`
- `λ7 = 0.02`

이 값은 초기값이며 시뮬레이션 기반으로 조정한다.

### 8.3 보상 항 의미

#### 양의 항

- `served_request_cnt_t`: 실제 처리된 요청 수

#### 패널티 항

- `waiting_passenger_minutes_t`: 누적 대기 불편
- `invehicle_passenger_minutes_t`: 누적 승차 시간 부담
- `deadhead_vehicle_minutes_t`: 공차 재배치 비용
- `active_bus_cnt_t`: 운영 중 차량 부담
- `rejected_request_cnt_t`: 미처리 수요
- `energy_proxy_t`: 에너지 소비 대리 지표

---

### 8.4 현재 단계 proxy reward(대리 보상)

현재 승인 데이터에는 완전한 버스 이동 로그가 없으므로 최종 보상을 직접 계산할 수 없다.

따라서 현 단계에서는 representation learning(표현 학습) 과 transition-aware scoring(전이 인지 점수화) 용 임시 보상을 아래처럼 둘 수 있다.

\[
r_t^{proxy}
=
0.3 \cdot \log(1 + next\_boardings\_recent)
- 0.1 \cdot \log(1 + next\_alightings\_recent)
- 1.2 \cdot \log(1 + next\_waiting\_passenger\_cnt)
- 0.1 \cdot \max(0, delta\_t\_hr - 1)
\]

### 8.5 임시 보상 해석

이 대리 보상은 `next_waiting_passenger_cnt` 에 가장 강한 패널티를 둔다.  
이는 현재 데이터 단계에서 가장 중요한 운영 압력이 대기 승객 압력이라는 해석과 일치한다.

이 보상은 **최종 운영 RL 보상** 이 아니며, 현재 전이 학습 단계의 임시 연구 신호일 뿐이다.

---

## 9. 전이(Transition=전이) 명세

### 9.1 현재 전이 튜플

현재 전이 튜플은 아래와 같다.

\[
(s_t, s_{t+1})
\]

여기서

- `s_t` = 현재 정류소 상태 특성
- `s_{t+1}` = 다음 관측 시각 타깃 특성

향후 행동이 연결되면 아래로 확장된다.

\[
(s_t, a_t, r_t, s_{t+1})
\]

즉 이것이 향후 SAR (State-Action-Reward=상태-행동-보상) 완성형 전이 계약이다.

### 9.2 중요한 전이 제약

전이는 observation-driven(관측 주도형) 이다.  
따라서 모든 학습과 평가는 아래 사실을 반드시 존중해야 한다.

- `next_state_ts` 는 `state_ts + 1 hour` 를 보장하지 않는다.

---

## 10. PyTorch 입력 데이터셋 계약

### 10.1 현재 row-level dataset(행 단위 데이터셋)

현재 한 학습 샘플은 아래를 노출해야 한다.

- `node_uid`
- `state_ts`
- `next_state_ts`
- `x`
- `y`
- `delta_t_hr`

#### 권장 `x`

- `log1p_boardings_recent`
- `log1p_alightings_recent`
- `log1p_waiting_passenger_cnt`
- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`
- `is_peak`
- `delta_t_hr`
- `node_type_stop`

#### 권장 `y`

- `next_boardings_recent`
- `next_alightings_recent`
- `next_waiting_passenger_cnt`

---

### 10.2 향후 snapshot-level graph dataset(시점 단위 그래프 데이터셋)

snapshot grouping(스냅샷 그룹화) 이후에는 한 샘플이 하나의 시각 그래프를 표현해야 한다.

권장 계약은 아래와 같다.

- `snapshot_id`
- `state_ts`
- `next_state_ts`
- `x: [num_nodes, feat_dim]`
- `y: [num_nodes, target_dim]`
- `node_uid: [num_nodes]`
- `edge_index: [2, E]`
- `edge_attr: [E, edge_feat_dim]`
- `global_context`
- `delta_t_hr`

현재 단계에서는 `edge_index` 가 비어 있을 수 있으나, 공식 그래프 위상은 `graph_edge_master` 복구 이후 확정한다.

---

## 11. 현재 학습 단계 해석

### 11.1 지금 가능한 것

현재 승인 데이터는 아래를 지원한다.

- stop-level transition modeling(정류소 단위 전이 모델링)
- feature engineering(특성 공학)
- temporal supervised learning(시계열 지도학습)
- node representation pretraining(노드 표현 사전학습)
- graph-ready dataset construction(그래프 대응 데이터셋 구성)

### 11.2 아직 완전하지 않은 것

현재 승인 데이터는 아래가 아직 부족하므로 full policy optimization(완전 정책 최적화) 을 바로 지원하지 않는다.

- 복구된 그래프 엣지 테이블 없음
- 버스 agent state(에이전트 상태) 테이블 없음
- 명시적 행동 로그 없음
- simulator reward rollup(시뮬레이터 보상 집계) 없음

### 11.3 최종 해석

따라서 현재 `rl_state_training_base` 는 아래로 해석한다.

- **transition training base(전이 학습 베이스)**
- **not yet full policy training base(아직 완전 정책학습 베이스는 아님)**

---

## 12. 검증 규칙

SAR (State-Action-Reward=상태-행동-보상) 파이프라인은 최소 아래 검증을 만족해야 한다.

### 12.1 상태 검증

- 음수 탑승 수 없음
- 음수 하차 수 없음
- 음수 대기 승객 수 없음
- 잘못된 시간 값 없음
- `delta_t_hr <= 0` 없음

### 12.2 전이 검증

- `next_state_ts` null 없음
- 다음 상태 타깃 null 없음
- time inversion(시간 역전) 없음

### 12.3 향후 행동 검증

- 모든 행동은 feasible candidate set(허용 후보 집합) 안에 있어야 한다.
- masked action(마스크된 행동) 은 학습과 평가에서 선택되면 안 된다.

### 12.4 향후 보상 검증

- 모든 보상 항은 원천 지표로 역추적 가능해야 한다.
- 에너지 및 차량 패널티는 시뮬레이터 로그로 재현 가능해야 한다.

---

## 13. 단계 연결

### 13.1 Phase 5 결과

Phase 5 는 clean stop-level transition base(정제된 정류소 단위 전이 베이스) 를 확립했다.

### 13.2 다음 즉시 산출물

다음 즉시 산출물은 아래다.

1. `rl_stop_transition_features`
2. `graph_snapshot_index`
3. PyTorch dataset loader contract(데이터셋 로더 계약)
4. baseline supervised transition model(기준 지도학습 전이 모델)
5. `graph_edge_master` 복구 설계

### 13.3 edge restoration(엣지 복구) 이후

`graph_edge_master` 복구 후에는 본 문서를 아래로 확장해야 한다.

- graph adjacency semantics(인접 관계 의미)
- edge travel-time attributes(엣지 이동시간 속성)
- bus agent state schema(버스 상태 스키마)
- action mask schema(행동 마스크 스키마)
- environment step logic(환경 전개 로직)
- simulator-based reward realization(시뮬레이터 기반 보상 실현)

---

## 14. 최종 고정 해석

현재 승인된 프로젝트 상태에서

- **State(상태)** 는 stop-level observed demand context(정류소 단위 관측 수요 문맥) 로 고정되며, 그중 `waiting_passenger_cnt` 를 가장 중요한 운영 신호로 본다.
- **Action(행동)** 은 bus agent(버스 에이전트) 의 다음 목표 정류소 선택으로 개념 고정되지만, 그래프 엣지와 버스 로그 복구 전까지는 실행형 정책 행동은 아니다.
- **Reward(보상)** 은 운영 최적화 방향으로 개념 고정되며, 현 단계에서는 대기 승객 감소에 가장 큰 비중을 둔 대리 보상 사용을 허용한다.

따라서 현재 프로젝트 단계는 아래다.

> **graph-ready transition learning stage(그래프 대응 전이 학습 단계)**

아직 아래 단계는 아니다.

> **full RL policy optimization stage(완전 강화학습 정책 최적화 단계)**

---

## 15. 버전

- document: `rl_sar_spec.md`
- version: `v0.2`
- status: `DRAFT FOR PROJECT FIXING`
- based on:
  - `graph_state_timeslice` approved
  - `rl_state_training_base` approved
  - feature priority note reflected
  - `graph_edge_master` pending restoration
