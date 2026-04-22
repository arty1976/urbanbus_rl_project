# Integrated Graph Master Spec

## 1. 목적

본 문서는 `route_link_sequence` 운영 기준을 기반으로, 대구 버스 연구용 **통합 그래프 마스터**의 현재 설계 기준을 정의한다.

현재 문서는 아래 3개 단계를 모두 반영한다.

1. **Phase 1**: LINK 중심 통합 그래프 마스터 골격 구축
2. **Phase 2**: STOP(Stop=정류장) 노드 적재 및 공간 근접 기반 1차 매핑 구축
3. **Phase 3**: `route_id + move_dir_code + sequence(sequence=순서)` 기준 2차 정밀 매핑 설계

목표는 다음과 같다.

1. 정류장 stop(정류장) 과 링크 link(도로 구간) 를 분리된 노드 유형으로 관리한다.
2. `route_id + move_dir_code + link_seq` 기준 방향성 `LINK_TO_LINK` 연결을 그래프 간선으로 승격한다.
3. 정류장과 링크 사이의 서비스 연결을 `STOP_TO_LINK`, `LINK_TO_STOP` 간선으로 관리한다.
4. 공간 근접 기반 1차 매핑과 노선/방향/순서 기반 2차 정밀 매핑을 분리하여 관리한다.
5. 향후 GAT (Graph Attention Network=그래프 어텐션 네트워크) 와 RL (Reinforcement Learning=강화학습) 입력으로 연결 가능한 `node_uid / edge_uid` 통합 키 구조를 유지한다.

---

## 2. 배경

직전 단계까지 다음이 완료되었다.

- `route_link_sequence` remediation(remediation=문제 시정) 완료
- legacy sample `route_id='1000'` 287행 제거 완료
- dedup(deduplication=중복 제거) 기준 승격 및 최종 검증 완료
- 최종 판정 APPROVED
- 운영 표준 문서화 완료
- 통합 그래프 마스터 1차 SQL 작성 완료
- STOP 노드 적재 및 공간 근접 기반 1차 매핑 Phase 2 설계 완료
- 정류장-링크 2차 정밀 매핑 Phase 3 설계 시작

따라서 현재 단계의 목표는 `route_link_sequence` 를 단순 노선 적재 결과로 보지 않고, **정류장 서비스와 차량 이동이 함께 표현되는 연구용 도시 버스 그래프**로 승격하는 것이다.

---

## 3. 설계 원칙

### 3.1 노선 중심이 아니라 이동 가능성 중심
고정노선 데이터는 출발점이지만, 연구의 목표는 **노선 없는 On-Demand(On-Demand=수요응답형) 대중버스 운영**이다.
따라서 통합 그래프 마스터는 노선 그 자체보다, 버스가 현재 상태에서 어디로 이동 가능한지를 우선적으로 표현해야 한다.

### 3.2 방향성 필수
`link_seq` 는 route 전체 단일 순번이 아니라 `move_dir_code` 별 순번으로 해석된다.
따라서 핵심 기준 키는 반드시 아래를 유지한다.

- `route_id`
- `move_dir_code`
- `link_seq`

### 3.3 stop 와 link 분리
정류장과 링크는 역할이 다르다.

- stop: 승객 서비스와 대기 수요의 기본 단위
- link: 차량 이동과 혼잡도의 기본 단위

따라서 두 개체를 분리된 노드 유형으로 유지해야, 이후 heterogeneous graph(Heterogeneous Graph=이질 그래프) 및 상태 테이블 설계가 자연스럽다.

### 3.4 저장된 `geom_5187` 직접 사용
거리 계산과 공간 매핑의 기본 좌표계는 **EPSG:5187** 이다.

원칙은 다음과 같다.

- `dim_stop.geom_5187` 직접 사용
- 링크 원천의 `geom_5187` 직접 사용
- 최종 매핑 로직에서 cross join(cross join=교차 조인) + `ST_Transform` 반복 호출 금지
- `geom_5187` 가 원천 테이블에 없을 경우, 먼저 영구 컬럼 추가 및 1회 백필 후 사용

### 3.5 1차 매핑과 2차 매핑 분리
정류장-링크 연결은 한 번에 최종 확정하지 않는다.

- **Phase 2**: 공간 근접 기반 `SNAP_NEAREST`
- **Phase 3**: 노선/방향/순서 기반 `ROUTE_INFERRED` 또는 `ROUTE_CONFIRMED`

즉, 1차는 "가장 가까운가", 2차는 "운영적으로 맞는가"를 본다.

---

## 4. 핵심 엔터티

### 4.1 graph_node_master
통합 노드 마스터.

대표 유형:
- `STOP`
- `LINK`
- `ZONE`
- `VIRTUAL_PICKUP`

대표 키:
- `node_uid` 예: `STOP:12345`, `LINK:8800123`

### 4.2 graph_edge_master
통합 간선 마스터.

대표 유형:
- `LINK_TO_LINK`
- `STOP_TO_LINK`
- `LINK_TO_STOP`
- `STOP_TO_STOP`
- `ZONE_TO_STOP`

대표 키:
- `edge_uid`

### 4.3 stop_link_mapping_master
정류장과 링크 사이의 연결 관계를 저장한다.

대표 `mapping_type`:
- `SNAP_NEAREST`
- `ROUTE_INFERRED`
- `ROUTE_CONFIRMED`
- `MANUAL_REVIEW`
- `MANUAL`

주요 관리 컬럼:
- `stop_id`
- `link_id`
- `distance_to_link_m`
- `confidence_score`
- `route_id`
- `move_dir_code`
- `is_primary_mapping`
- `is_active`

### 4.4 route_link_graph_edge_vw
`route_link_sequence` 로부터 동일 `route_id + move_dir_code` 내부의 연속 링크 간선을 생성하는 뷰.

### 4.5 link_master_candidate_vw
물리 링크 후보를 `link_id` 기준 1행으로 정리한 표준 뷰.

원칙:
- geometry(Geometry=공간도형) 동일성 기준이 아니라 `link_id` 기준 대표 1행
- `geom_5187` 보유
- 가능하면 `geom_4326` 도 함께 유지

---

## 5. 핵심 키 정책

### 5.1 route_link_sequence 기준 키
반드시 다음 기준을 유지한다.

- `(route_id, move_dir_code, link_seq)`

### 5.2 graph node key
- `node_uid = node_type || ':' || node_id`

예시:
- `STOP:702341`
- `LINK:1775701914078_000123`

### 5.3 graph edge key
간선은 사람이 읽을 수 있는 조합형 키를 기본으로 한다.

예시:
- `LINK_TO_LINK:R1000:0:12:8800123->8800456`
- `STOP_TO_LINK:702341->8800123`
- `LINK_TO_STOP:8800123->702341`

---

## 6. 데이터 흐름

### Phase 1. LINK 중심 골격 생성
1. `route_link_sequence` 에 등장하는 `link_id` 를 기준으로 `graph_node_master` 에 LINK 노드를 적재한다.
2. `route_link_graph_edge_vw` 를 이용하여 같은 `route_id + move_dir_code` 내부의 `(현재 link_seq -> 다음 link_seq)` 연결을 생성한다.
3. `LINK_TO_LINK` 간선을 `graph_edge_master` 에 적재한다.

### Phase 2. STOP 노드 적재 및 공간 근접 1차 매핑
1. 원천 테이블에 `geom_5187` 영구 컬럼 존재 여부를 점검한다.
2. 필요 시 `prepare_source_spatial_columns.sql` 로 컬럼 추가, 백필, 공간 인덱스, `ANALYZE` 를 수행한다.
3. `dim_stop` 를 `graph_node_master` 에 `STOP:{stop_id}` 규칙으로 적재한다.
4. `link_master_candidate_vw` 를 기준으로 stop-link 최근접 매핑을 수행한다.
5. 매핑 전략은 **150m 1차 + 500m fallback(fallback=대체 보완 경로)** 이다.
6. 정류장별 정확히 1건의 primary(primary=주매핑) 를 선택하며, 선택은 `row_number()` 기반으로 수행한다.
7. 결과를 `stop_link_mapping_master` 에 `mapping_type='SNAP_NEAREST'` 로 적재한다.
8. `STOP_TO_LINK`, `LINK_TO_STOP` 간선을 생성한다.

### Phase 3. route-aware 2차 정밀 매핑
1. `stop_route_sequence` 또는 이에 준하는 정류장-노선-방향-순서 데이터를 확보한다.
2. `route_link_sequence` 와 `route_id + move_dir_code` 축으로 결합한다.
3. stop 측 `stop_seq` 와 link 측 `link_seq` 정합성을 비교한다.
4. 공간 근접 기반 1차 매핑을 유지한 채, 운영적으로 더 타당한 후보를 `ROUTE_INFERRED` 또는 `ROUTE_CONFIRMED` 로 누적한다.
5. 필요 시 기존 service edge(service edge=서비스 간선) 를 비활성화하고 route-aware 기준 간선을 재생성한다.

---

## 7. 정류장-링크 매핑 정책

### 7.1 Phase 2: SNAP_NEAREST
기준:
- `geom_5187` 직접 사용
- 150m 내 후보 우선
- 150m 내 후보가 없으면 500m 내 fallback 후보 탐색
- 정류장별 최근접 1건을 primary 로 선택

권장 `confidence_score` 예시:
- `<= 15m` -> `1.00`
- `<= 30m` -> `0.95`
- `<= 50m` -> `0.85`
- `<= 100m` -> `0.70`
- `> 100m` -> `0.50`

### 7.2 Phase 3: ROUTE_INFERRED / ROUTE_CONFIRMED
보정 기준:
- 동일 `route_id`
- 동일 `move_dir_code`
- `stop_seq` 와 `link_seq` 의 상대 위치 정합성
- 거리 수준의 상식성
- 반대 방향 링크 여부

권장 품질 등급:
- `APPROVED`
- `HOLD`
- `MANUAL_REVIEW`

### 7.3 1차 결과 보존 원칙
`SNAP_NEAREST` 결과는 삭제하지 않는다.

이유:
- 공간 근접 기준의 원본 참조값 유지
- Phase 3 보정 결과와 비교 가능
- 추후 수동 보정 시 근거 데이터로 활용 가능

---

## 8. 검증 기준

### 8.1 순번 이상
같은 `route_id + move_dir_code` 내부에서 `link_seq` 가 역전되거나 비증가하면 안 된다.

### 8.2 고아 간선
`graph_edge_master` 의 모든 `src_node_uid`, `dst_node_uid` 는 `graph_node_master` 에 존재해야 한다.

### 8.3 정류장-링크 미매핑
`dim_stop` 기준 핵심 정류장은 최소 1건의 활성 primary mapping(primary mapping=기본 대표 매핑) 을 가져야 한다.

### 8.4 duplicate primary(primary=중복 주매핑) 금지
동일 `stop_id` 에 활성 primary 매핑이 2건 이상 존재하면 안 된다.

### 8.5 거리 분포 점검
다음 항목을 정기적으로 점검한다.

- average(avg=평균)
- median(median=중앙값)
- p95(p95=95백분위수)
- max(max=최대값)
- `distance_to_link_m > 100m` 건수
- `distance_to_link_m > 250m` 건수

### 8.6 route-aware 품질 점검
Phase 3 이후 아래를 점검한다.

- `route_id + move_dir_code` 일치율
- 기존 `SNAP_NEAREST` 대비 보정 건수
- 반대 방향 링크 의심 건수
- sequence 정합성 이상치
- `APPROVED / HOLD / MANUAL_REVIEW` 분포

---

## 9. 현재 단계의 범위와 비범위

### 포함 범위
- LINK 노드 생성
- `route_link_sequence` 기반 `LINK_TO_LINK` 간선 생성
- STOP 노드 적재
- stop-link 공간 근접 1차 매핑
- `STOP_TO_LINK`, `LINK_TO_STOP` 간선 생성
- route-aware 2차 정밀 매핑 설계 및 SQL 초안
- readiness / mapping quality 검증 SQL 작성

### 비포함 범위
- 수요 상태 시계열 적재
- 실제 링크 속성 거리/시간 완전 결합
- 차량 상태 테이블
- GAT 입력 텐서 변환
- MAPPO 정책 학습

즉, 현재 단계는 **통합 그래프 마스터 구축과 정류장 서비스 연결 확정 단계**까지다.

---

## 10. 실행 순서

권장 실행 순서는 아래와 같다.

1. `prepare_source_spatial_columns.sql`
2. `create_graph_node_master.sql`
3. `create_graph_edge_master.sql`
4. `create_stop_link_mapping_master.sql`
5. `create_route_link_graph_edge_view_and_load.sql`
6. `load_graph_master_phase_2.sql`
7. `graph_mapping_quality_check.sql`
8. `refine_stop_link_mapping_phase_3.sql`
9. `stop_link_mapping_phase_3_quality_check.sql`

---

## 11. 다음 단계

다음 단계는 아래 순서가 적절하다.

1. `graph_state_timeslice` 설계
2. 정류장 대기 수요, 링크 이동시간, 혼잡도, 시간대 상태를 그래프 키에 결합
3. GAT 입력용 adjacency(adjacency=인접관계) / feature(feature=특징) 추출 쿼리 설계
4. RL 상태 `s_t`, 행동 `a_t`, 보상 `r_t` 와의 연결 정의

---

## 12. 한 줄 정의

현재 통합 그래프 마스터 구축 단계의 본질은,

**검증 완료된 `route_link_sequence` 를 기반으로 LINK 이동 그래프와 STOP 서비스 그래프를 결합하고, 공간 근접 및 노선/방향/순서 정합성 기준을 함께 반영한 연구용 도시 버스 그래프 구조로 승격하는 것**이다.
