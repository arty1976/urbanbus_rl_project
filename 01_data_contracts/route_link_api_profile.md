# route_link_api_profile.md

## 1. 소스 개요

| 항목 | 값 |
|---|---|
| API 오퍼레이션 | `getLink02` |
| 서비스 Base URL | `https://apis.data.go.kr/6270000/dbmsapi02` |
| 전체 엔드포인트 | `https://apis.data.go.kr/6270000/dbmsapi02/getLink02` |
| 응답 포맷 | JSON (`resultType=json`) |
| 의미 | 노선별 경유 링크(도로 구간) 순서 정보 |
| 스냅샷 저장 경로 | `data/raw/daegu/api_snapshots/daegu_bus_api_YYYYMMDD_HHmmss.json` |
| 프로파일 작성일 | 2026-04-09 |

---

## 2. 컬럼 목록 및 매핑

| API 응답 컬럼 | staging 컬럼 | 타입 (staging) | 설명 | 비고 |
|---|---|---|---|---|
| `linkId` | `link_id_raw` | `text` | 링크(도로 구간) 식별자 | 후보 키; 형식 미확정 |
| `stNode` | `st_node_raw` | `text` | 링크 시작 노드 ID | 형식 미확정 |
| `edNode` | `ed_node_raw` | `text` | 링크 종료 노드 ID | 형식 미확정 |
| `gisDist` | `gis_dist_raw` | `text` | GIS 거리 (단위 미확정) | 숫자 변환은 staging 이후 |
| `moveDir` | `move_dir_raw` | `text` | 이동 방향 코드 | 코드값 도메인 미확정 |
| `linkSeq` | `link_seq_raw` | `text` | 노선 내 링크 순서(추정) | ⚠️ 아래 Fact/Assumption 참조. 후속 승격 시 `INTEGER` 변환 예정 |

> **원칙**: staging 테이블은 모든 컬럼을 `text`로 수신한다. 형변환은 analytical 레이어 승격 시 수행한다.

---

## 3. 후보 키 분석

| 후보 | 유일성 | 안정성 | 결론 |
|---|---|---|---|
| `linkId` 단독 | **미확인** — 노선마다 동일 linkId가 반복될 가능성 있음 | 미확인 | **후보 키(Candidate)**, 확정 불가 |
| `(routeId, linkSeq)` 복합 | routeId가 응답에 포함되면 복합 유일 가능성 높음 | routeId 응답 여부 확인 필요 | **후보 키(Candidate)** |
| `(linkId, routeId)` 복합 | 노선-링크 관계가 N:M이면 복합 필요 | 미확인 | **후보 키(Candidate)** |

> ⚠️ 현재 응답 컬럼에 `routeId`가 명시되지 않음.  
> 실제 응답 JSON을 열어 `routeId` 또는 `routeNo` 필드가 포함되는지 반드시 확인할 것.

---

## 4. Null 리스크

| 컬럼 | Null 가능성 | 근거 |
|---|---|---|
| `linkId` | 낮음 (추정) | 식별자 성격 |
| `stNode` | 중간 | 일부 구간에서 노드 미정의 가능성 |
| `edNode` | 중간 | 동일 |
| `gisDist` | 중간 | 거리 정보 미제공 구간 존재 가능 |
| `moveDir` | 높음 | 방향 정보가 선택적으로 제공될 가능성 |
| `linkSeq` | 낮음 (추정) | 순서 정보는 필수 성격 |

---

## 5. 중복 리스크

| 시나리오 | 리스크 |
|---|---|
| 동일 API 스냅샷 중복 적재 | `snapshot_file` + DB UNIQUE 미적용 시 발생 가능 |
| 여러 노선이 동일 `linkId` 공유 | `linkId` 단독 PK 사용 불가 이유 |
| 실시간/정기 수집 중복 | `snapshot_file` 컬럼으로 출처 추적 필요 |

---

## 6. linkSeq 해석 근거 및 한계

### Fact (검증된 사실)
- `getLink02` API의 명칭이 "노선별 경유 링크 순서 조회"임
- 응답 컬럼 중 `linkSeq`라는 이름이 존재함
- `stNode`(시작 노드)와 `edNode`(종료 노드)가 함께 반환되므로 링크의 방향성이 표현됨

### Assumption (가정 — 실제 데이터로 검증 필요)
- `linkSeq`가 동일 노선 내에서 1부터 순차 증가한다고 **가정**
- `linkSeq` 기준 정렬로 노선의 경유 순서를 재현할 수 있다고 **가정**
- `linkSeq`가 노선 방향(왕복/단방향)과 일치한다고 **가정**
- `moveDir`이 링크와 노선의 주행 방향 일치/반대를 나타낸다고 **가정**

### 검증 방법 (실제 데이터 수령 후 수행)
```sql
-- linkSeq 연속성 확인: gap이 있으면 순서 가정이 흔들림
SELECT route_id_inferred,
       MIN(link_seq_raw::int)  AS seq_min,
       MAX(link_seq_raw::int)  AS seq_max,
       COUNT(*)                AS cnt,
       MAX(link_seq_raw::int) - MIN(link_seq_raw::int) + 1 AS expected_cnt
FROM   stg_daegu_route_links_api
GROUP  BY route_id_inferred
HAVING COUNT(*) <> MAX(link_seq_raw::int) - MIN(link_seq_raw::int) + 1;
```

---

## 7. 승격(Promotion) 조건

아래 조건이 모두 충족될 때까지 `stg_daegu_route_links_api`에서 분석 레이어로 승격하지 않는다.

- [ ] `routeId` 컬럼 응답 포함 여부 확인
- [ ] `linkId` 단독 유일성 또는 복합 키 확정
- [ ] `linkSeq` 연속성 검증 쿼리 실행 및 결과 기록
- [ ] `moveDir` 코드 도메인 확인
- [ ] `gisDist` 단위(m / km) 확인
- [ ] Null 비율 실측 후 기록
