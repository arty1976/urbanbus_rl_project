---
name: stop_to_stop_edge_leg_aggregator
description: Builds STOP_TO_STOP directed edges in graph_edge_master by collapsing mixed stop + intermediate-node sequences in route_link_sequence into stop-to-stop "legs". Use this whenever building a GATv2 / GNN static graph for a city's bus network from per-city link sequence data — Daegu, Seoul, Busan, Daejeon, or any new city — even if the user doesn't explicitly mention "leg-aggregation". Triggers on prompts about low stop-to-stop coverage (<50%), route_link_sequence ↔ dim_stop mismatch, graph_edge_master bulk load, GATv2 static edge reconstruction, or "다른 도시 버스 노선 DB 구축". Also triggers when the user mentions bus network DB work for a new city. Bundled readiness template uses leg-aggregation denominator; ordinary "count link rows" coverage metrics are wrong for this problem shape and must not be used.
---

# Purpose

이 스킬은 도시 버스 노선 DB (Daegu / Seoul / Busan / Daejeon / …) 에서
`route_link_sequence` 원천 테이블로부터 GATv2 기반 정적 그래프의 핵심인
`graph_edge_master.STOP_TO_STOP` 엣지를 **leg-aggregation** 방식으로 생성하는
파이프라인을 강제합니다. 단순 link-pair 적재가 아니라, "한 stop 출발 ~ 다음
stop 도착" 까지의 모든 intermediate link 를 하나의 directed edge 로 접는
알고리즘을 보장합니다.

### 왜 이 접근이 필요한가 (one-line)
한국 지자체 대부분이 제공하는 `route_link_sequence` 류 테이블은 **정류장 노드와
도로·교차로 intermediate 노드가 섞여있는 link 시퀀스** 이다. 두 끝이 모두
`dim_stop` 에 있는 link 만 적재하면 실제 coverage 가 20~30% 대로 급락한다
(Daegu 사례: 22.84%). Leg-aggregation 은 이 소스의 본래 구조를 그대로 살려
99%+ coverage 를 회복시킨다.

---

## Core Rules

1. **Schema preservation**: `graph_edge_master` 의 기존 스키마 (`distance_m`
   numeric, `time_sec` numeric, `is_primary_edge` boolean, `edge_rank` smallint,
   `edge_uid` text unique, `is_active` boolean, `is_bidirectional` boolean) 를
   **절대 변경하지 않는다**. GATv2 static graph 계약이 깨지기 때문. 누락
   컬럼은 `ALTER TABLE ADD COLUMN IF NOT EXISTS` 만 허용.
2. **Mixed-sequence 전제**: `route_link_sequence.st_node_id` / `ed_node_id` 는
   정류장일 수도, intermediate 노드일 수도 있다. 이를 **가정하고 코드를
   작성**하라. "모든 node 가 stop 일 것" 이라는 기대는 한국 지자체 데이터에서
   거의 항상 거짓이다.
3. **Stop-to-stop directed edge only**: Intermediate 노드는 `graph_edge_master`
   에 **저장하지 않는다**. 필요하면 그래프-외부의 링크 테이블에 별도 관리.
4. **거리 계산 2단**: `coalesce(nullif(sum(gis_dist) within leg, 0),
   ST_Distance(src.geom_5187, dst.geom_5187))` — 누적 실제 링크 거리 우선,
   0/NULL 이면 stop 좌표 직선거리 fallback.
5. **통행시간**: `distance_m / 1000.0 / 20.0 * 3600.0` (평균 20 km/h 가정;
   도시마다 실측 평균이 다르면 이 상수만 교체).
6. **Edge 고유성**: `edge_uid = 'STOP_TO_STOP:' || route_id || ':' ||
   move_dir_code || ':' || src_stop || '->' || dst_stop`. 회차·순환 노선이
   같은 (route, dir, src, dst) 를 여러 leg 로 방문해도 **edge_uid 단위 dedup
   으로 가장 이른 leg 1건만 유지**해야 한다 (ON CONFLICT 21000 회피).
7. **Primary 선정**: 동일 `(src_stop, dst_stop)` 에 여러 노선 엣지가 있으면
   `row_number() over (partition by src_node_uid, dst_node_uid order by
   distance_m asc, route_id asc, move_dir_code asc, link_seq asc)` 로 rank
   부여, rank=1 → `is_primary_edge=true`.
8. **Idempotent**: 전체 파이프라인은 `ON CONFLICT (edge_uid) DO UPDATE` 로
   재실행 안전. Stale 엣지는 `DELETE` 아니라 `is_active=false`.
9. **Orphan 금지**: `dim_stop.geom_5187 is not null` 인 stop 만 엣지 생성
   대상. 매칭 실패 src/dst 는 전 단계에서 필터 (STAGE 4 inner join).

---

## Leg-aggregation 9-Stage Algorithm

이 순서를 지키면 회차 노선·U-turn·mixed sequence 모두 안전하게 처리된다.

```
STAGE 1  tmp_rls_tagged    : link 마다 (st_is_stop, ed_is_stop) 플래그 부여
                             (dim_stop LEFT JOIN 으로 판정)
STAGE 2  tmp_rls_legged    : (route_id, move_dir_code) partition 으로
                             link_seq 순회, st_is_stop=TRUE 마다 leg_id 증가
                             (window: sum(case when st_is_stop then 1 else 0
                              end) over (... rows unbounded preceding))
STAGE 3  tmp_leg_edges     : leg 단위 집계 —
                               src_stop_id = (array_agg(st_node_id order by
                                              link_seq))[1]       (반드시 stop)
                               dst_stop_id = (array_agg(ed_node_id order by
                                              link_seq)
                                              filter (where ed_is_stop))[1]
                               cum_gis_dist = sum(gis_dist) between
                                              seg_start_seq ~ seg_end_seq
STAGE 4  tmp_edge_src      : dim_stop.geom_5187 INNER JOIN (src, dst 양쪽),
                             distance_m / time_sec 계산, edge_uid 생성
STAGE 5  tmp_edge_dedup    : row_number() over (partition by edge_uid order by
                             link_seq asc, link_id asc) = 1 만 유지 (회차 노선
                             중복 edge_uid → ON CONFLICT 21000 방지)
STAGE 6  tmp_edge_ranked   : (src_node_uid, dst_node_uid) 단위 edge_rank
                             부여 (distance_m ASC)
STAGE 7  INSERT INTO graph_edge_master ...
           ON CONFLICT (edge_uid) DO UPDATE SET (모든 변경 가능 컬럼) + updated_at
STAGE 8  UPDATE g SET is_active=false WHERE edge_type='STOP_TO_STOP' AND
         NOT EXISTS (SELECT 1 FROM tmp_edge_ranked WHERE edge_uid = g.edge_uid)
         AND g.is_active = true    -- stale 처리
STAGE 9  AFTER 스냅샷 NOTICE (rows_after, primary_edges, inactive)
```

**핵심 통찰**: `leg_id` 는 "현재 링크까지 누적 st_is_stop=TRUE 횟수" 다. 한
link 가 새 stop 에서 시작하면 leg_id 가 증가하고, 그 stop 에서 출발해 다음
stop 에 도착할 때까지의 모든 intermediate link 가 같은 leg_id 를 공유한다.
이것이 `(partition by route_id, move_dir_code order by link_seq rows unbounded
preceding)` window 의 역할이다.

---

## Diagnostic Framework — 4 가설 (coverage 문제 시 이 순서로 기각)

Coverage 가 기대보다 낮을 때 (<90%) 는 **반드시 이 순서로** 원인을 분류해야
한다. 바로 Hypothesis D 로 점프하면 A/B/C 의 다른 케이스를 놓칠 수 있다.

### Hypothesis A — 다른 키 체계 (alt-key) 사용
- `dim_stop` 의 `node_uid`, `bis_id`, `bs_id`, `nid`, `mobile_no` 등 대체
  키 컬럼에 `route_link_sequence.st_node_id` 가 매칭되는지 탐색.
- 기각 기준: `dim_stop` 에 alt-key 컬럼이 없거나, 있어도 매칭 0건.

### Hypothesis B — 브리지/staging 테이블에 있음
- `stop_link_mapping_master`, `stg_*_stops_geo`, `bs_YYYYMMDD`,
  `graph_node_master`, `err_*_mapping_failed` 등 후보 테이블에서
  모든 text/numeric 컬럼 × missing_id 매칭률을 **dynamic EXECUTE** 로 프로빙.
- 기각 기준: 모든 (테이블 × 컬럼) 조합에서 매칭 0건.

### Hypothesis C — 원천 누락 (업스트림 데이터 문제)
- `dim_stop` 적재 시점에 누락된 정류장이 있는지, 도시 측 원천 Shapefile
  버전과 link 시퀀스 snapshot 날짜 정합성 확인.
- 흔치 않지만 snapshot drift 가 있으면 여기서 잡힘.

### Hypothesis D — Mixed stop + intermediate node sequence (가장 흔함)
- missing_id 가 `dim_stop` 매칭 실패인 게 아니라, **원래부터 stop 이 아닌
  도로/교차로 노드** 일 가능성.
- **확정 검증**: 핫스팟 노선 1개의 link 시퀀스를 `S→-, -→S, S→-, -→-` 패턴
  으로 태그해 교차 패턴이 보이면 확정. 전체 분포에서 `S→S < 50%` 이면 거의
  확실.
- 확정 시 → leg-aggregation 알고리즘 적용.

> **Daegu 사례** (2026-04-19): A 기각(dim_stop 7컬럼뿐) → B 기각(5×N cols
> 0건) → D 확정(S→S 32.41%, links_per_stop avg 1.84). v1 coverage 22.84%
> → v2 leg-coverage 99.91%.

---

## Readiness Verification (pass criteria)

| ID | 체크 | 기대값 |
|---|---|---|
| R1 | `stop_to_stop_total` | > 0 |
| R2 | `primary_edges` | > 0 AND ≤ total |
| R3 | `self_loop_cnt` (`src_node_uid = dst_node_uid`) | **0** |
| R4 | `orphan_src` / `orphan_dst` (`dim_stop` 미매칭) | **0 / 0** |
| R5 | `duplicate_primary_cnt` (동일 (src,dst) 에 primary 2개 이상) | **0** |
| R6 | `null_distance` / `null_time` / `zero_distance` | **0 / 0 / 0** |
| R7 | `max_dist_m` | < 20000 (도시 반경 상한) |
| R10 | **leg-based coverage_pct** | ≥ 95% (권장 99%+) |

**R10 중요**: 분모는 반드시 "leg-aggregation distinct legs" 이어야 한다.
"route_link_sequence link rows" 를 분모로 쓰면 mixed-sequence 에서 필연적으로
낮은 값 (Daegu 22.84%) 이 나오지만 이는 **측정 오류**이지 데이터 문제가
아니다. 번들된 `references/readiness_stop_to_stop_edge_template.sql` 을 그대로
적용하면 leg 분모가 자동 계산된다.

---

## Failure Modes (흔한 함정)

### F1. ON CONFLICT DO UPDATE 21000 에러
**증상**: `ON CONFLICT DO UPDATE 명령은 두번째 작업에는 부모의 영향을 주지
않음` (또는 영어: `ON CONFLICT DO UPDATE command cannot affect row a second
time`).
**원인**: 회차·순환 노선에서 같은 (route, dir, src, dst) 가 여러 leg 로
반복되어 같은 edge_uid 가 INSERT 한 번에 2회 이상 등장.
**처방**: STAGE 5 의 `tmp_edge_dedup` 을 절대 생략하지 않는다. `row_number()
over (partition by edge_uid order by link_seq asc, link_id asc) = 1` 필터로
가장 이른 leg 1건만 유지.

### F2. PowerShell NativeCommandError
**증상**: 파이프라인 런처가 `02_load` 단계에서 갑자기 중단. stderr 로
`NativeCommandError` 표시.
**원인**: `psql` 의 `RAISE NOTICE` / `\echo` 출력이 stderr 로 섞여 나오고,
PowerShell 5.1 및 7.x 의 기본 `$ErrorActionPreference = "Stop"` 정책이
native command 의 stderr 를 오류로 간주.
**처방**: 런처 상단에 `$ErrorActionPreference = "Continue"` 고정, 성패는
`$LASTEXITCODE` 로만 판정.

### F3. gis_dist = 0 stop pair
**증상**: 같은 위치에 표기된 정류장 쌍에서 distance_m = 0, time_sec = 0.
**처방**: `coalesce(nullif(cum_gis_dist, 0), ST_Distance(geom_5187))` 로
fallback. 그래도 0 이면 `dim_stop.geom_5187` 중복을 의심.

### F4. terminal leg 가 닫히지 않음
**증상**: 각 노선의 마지막 leg 에서 `dst_stop_id IS NULL` (ed_is_stop=TRUE
인 링크가 leg 안에 없음).
**처방**: STAGE 3 의 `seg_end_seq is not null` 필터로 해당 leg 는 버린다.
이는 데이터 손실이 아니라 "종점 이후의 꼬리 링크" 처리.

---

## Other-City Adaptation Checklist

다른 도시 버스 DB 에 적용할 때 **반드시 교체/확인** 해야 하는 항목:

1. **스키마 이름**: 도시별로 `dim_stop`, `route_link_sequence`, `bs_YYYYMMDD`,
   `link_YYYYMMDD`, `node_YYYYMMDD`, `graph_edge_master`, `graph_node_master`
   의 **실제 테이블명·컬럼명** 을 확인. Core Rules / 9-Stage 본문의 이름은
   Daegu 기준이므로 치환 필수.
2. **EPSG / CRS**: `geom_5187` 은 한국 중부 원점 TM 좌표계. 도시가 다른
   좌표계를 쓰면 (예: 동부 원점 5186) 컬럼명과 SRID 모두 교체.
3. **평균 속도**: 20 km/h 는 한국 도심 버스 평균. 도시별 실측 평균이 다르면
   time_sec 공식의 `20.0` 만 바꾼다.
4. **route_id 포맷**: Daegu 는 10자리 숫자 (`7361109008`). 서울은 5자리
   text (`N37`), 부산은 다른 포맷 등. 길이/타입 assumption 을 코드에 넣지
   않는다.
5. **move_dir_code 타입**: Daegu 는 text ('0' / '1'), 일부 도시는 smallint.
   `::integer` 캐스트 시점을 데이터 타입에 맞춰 조정.
6. **missing_id 패턴 학습**: 신규 도시 첫 pass 에서 반드시 D1~D3 진단을 돌려
   해당 도시의 "intermediate node 접두사 체계" 를 먼저 파악한다. Daegu 는
   `15xxx / 30007xxx / 73611xxx` 패턴이지만 도시마다 다르다.

---

## 번들 파일 — references/

### `references/readiness_stop_to_stop_edge_template.sql`
R0~R12 검증 쿼리 풀세트 (leg-aggregation 분모 포함). 치환 포인트는 파일 상단
주석에 `-- @CHANGE_FOR_CITY` 태그로 표시되어 있다. 다른 도시 적용 시 해당
라인만 수정하면 바로 사용 가능.

SKILL.md 본문으로는 파이프라인 SQL 을 **코드 그대로 복붙하지 않는다** — 9-stage
알고리즘 설명 + 위 번들 템플릿 파일을 참조하는 방식으로 파이프라인을 재생성
하도록 지시한다. 신규 도시에서는 템플릿에서 출발해 Daegu 의 `02_load_v2.sql`
(프로젝트 내 `graph_edge_master_pkg/`) 를 참조 구현으로 활용.

---

## 호출 프롬프트 예시

- "부산시 버스 노선 DB 에서 `route_link_sequence` 로 `graph_edge_master`
  STOP_TO_STOP 엣지를 만들어 줘. `stop_to_stop_edge_leg_aggregator` 스킬의
  9-stage leg-aggregation 알고리즘과 edge_uid dedup 규칙을 반드시 따라야 해."
- "서울시 GATv2 static graph 용 stop-to-stop 엣지 빌드 중인데 coverage 가
  30% 밖에 안 나와. `stop_to_stop_edge_leg_aggregator` 의 4-가설 진단
  프레임워크 (A alt-key → B bridge → C upstream → D mixed-sequence) 를
  순서대로 돌려서 원인 분류해 줘."
- "다른도시(대전) 적용할 건데 `stop_to_stop_edge_leg_aggregator` 의
  Other-City Adaptation Checklist 6개 항목을 먼저 확인하고, 각 항목별로
  확인 SQL 을 뽑아 줘."
- "leg-aggregation 끝냈으니 `stop_to_stop_edge_leg_aggregator` 번들의
  readiness_stop_to_stop_edge_template.sql 을 복사해 현재 DB 에 맞게
  치환해서 실행해 줘 — R10 leg-coverage ≥ 95% 목표."
