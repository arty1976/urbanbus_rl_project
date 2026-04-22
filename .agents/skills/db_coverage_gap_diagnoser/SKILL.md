---
name: db_coverage_gap_diagnoser
description: Readiness 쿼리 결과 coverage가 기대치 대비 크게 낮을 때(예: 95% 이상 기대였으나 22.84%), 원인을 "스키마/키/의미론" 레벨에서 체계적으로 진단하는 4-가설 프레임워크. 대구 graph_edge_master 22.84% → 99.91% 복원 과정에서 정립. 다른 도시 버스 노선 DB 구축 / 다른 조인 파이프라인에서도 coverage 붕괴 시 제일 먼저 호출할 것.
---

# Purpose

파이프라인 readiness 에서 `coverage_pct` 같은 핵심 적재율 지표가
**기대치 대비 큰 폭으로 낮게 나올 때** (예: 95%+ 기대 vs 실측 22.84%),
"그냥 데이터가 부족한 건지, 키가 안 맞는 건지, 아예 의미론이 어긋난 건지"
를 **4-가설 프레임워크** 로 단계적으로 좁혀 내리는 범용 진단 스킬.

이 스킬은 특정 테이블/컬럼명을 전제하지 않는다. 다른 도시 / 다른
조인 파이프라인에서도 coverage 붕괴 시 똑같이 적용할 수 있도록
**도시/유스케이스별 치환점만 바꿔서 재사용** 하는 것이 목적이다.

> 💡 이 스킬은 "왜 적재율이 낮은가?" 질문 전용이지, "적재율을 올리는
> 방법" 은 **가설이 확정된 뒤** 그에 맞는 다른 스킬
> (`stop_to_stop_edge_leg_aggregator`, `dim_stop_promotion_guard`,
> `spatial_graph_mapping_guard` 등) 으로 넘긴다. 진단 ↔ 수정은 분리.

---

# Core Framework — 4 Hypotheses

Coverage 가 낮을 때 원인은 반드시 아래 네 가지 중 하나 (혹은 조합).
**순서대로** 하나씩 기각/확정해 나간다.

| ID | 가설 | 한 줄 정의 | 전형 증상 |
|----|------|----------|---------|
| **A** | Alt-Key Mismatch | 양쪽 테이블은 같은 개념을 담지만 조인 키가 서로 다른 컬럼 (예: `stop_id` vs `bis_id`, `node_uid` vs `internal_code`) | missing 측 id 가 특정 prefix/길이로 일관됨. join 키 한 쪽에만 존재 |
| **B** | Bridge / Staging Table Exists | 원천과 타겟 사이에 **중간 매핑 테이블** 이 이미 있어서, 직접 조인이 아닌 브리지 경유가 정답 | staging/bridge/mapping 이름이 포함된 테이블 존재. 원천 id 가 브리지엔 있음 |
| **C** | Upstream Data Missing | 원천 자체가 커버리지를 못 채울 만큼 row 가 부족 (ETL 누락/로딩 실패/삭제) | 원천 row count 가 기대치에 한참 못 미침. 최근 ETL 로그 이상 |
| **D** | Semantic Mismatch | 조인 키도 맞고 브리지도 필요없지만, **데이터의 의미론 자체가 다름** (예: sequence 테이블이 "stop 간 이동" 이 아니라 "stop + 도로/교차로 혼합 시퀀스") | 데이터 분포에 "비 stop" 섞임. "한 단위 당 row 수" 가 기대치 대비 수 배 |

**핵심 규칙**: 가설은 **A → B → C → D** 순으로 시도한다. A 가 기각되어야 B 에 의미가 있고, C 는 A/B 가 모두 기각된 뒤의 "마지막으로 ETL 자체를 의심" 단계이며, D 는 제일 비용 큰 "재설계" 가설이므로 **반드시 마지막에**.

---

# Core Rules

1. **진단은 INSERT/UPDATE 를 하지 않는다.** 이 스킬이 만드는 SQL 은 **SELECT 전용** 이어야 한다. 절대 원천 테이블을 수정하지 않음.
2. **DO block + dynamic EXECUTE 로 후보를 훑는다.** 후보 테이블/컬럼이 여러 개일 때, 각각을 하드코딩하지 말고 `information_schema.columns` 조회 → PL/pgSQL `format()/EXECUTE` 로 순회 → 매칭 수 집계.
3. **"패턴 태깅" 으로 가설 D 를 빠르게 검증.** 의미론 가설 D 는 **분포 비율** 로 확정된다 (예: `S→S / S→- / -→S / -→-` 4-way partition 의 비율). 단일 핫스팟 노선 1개만 뽑아서 눈으로 패턴을 확인하는 게 제일 빠르다.
4. **결론은 증거 3종 세트로.** 가설 확정에는 ① 전체 모수 대비 분포 수치 ② 핫스팟 샘플 rows ③ 기각된 가설 목록 — 이 세 가지가 **같은 로그 블록** 안에 있어야 한다. 한 가지만 가지고 섣불리 결정하면 오진.
5. **진단 결과는 재현 가능한 SQL 파일로 보관한다.** `D_*.sql`, `D2_*.sql`, `D3_*.sql` 처럼 번호를 붙여 `{pipeline_pkg}/diagnose/` 또는 `references/` 에 commit. 구두 보고로 끝내지 않는다.
6. **수정 스킬에 바톤을 넘긴다.** 가설이 확정되면 이 스킬은 역할 종료. 수정은 A→`dim_stop_promotion_guard`, B→`stop_link_mapping_audit` (future), C→ETL runbook, D→`stop_to_stop_edge_leg_aggregator` (leg-aggregation 계열) 로 분기.

---

# 4-Stage Diagnostic Workflow

## Stage D1 — Key Pattern Analysis (가설 A 검증)

"실패 측 id 들이 어떤 공통 패턴을 갖는가?" 확인하는 단계.

수집 지표:
- `missing_id_count` : 조인 실패 rows
- `distinct_missing_id` : distinct 개수
- `id_length_dist` : 길이별 분포 (보통 길이 하나로 몰림)
- `prefix_dist` : 첫 3~5자 prefix 별 분포
- `sample_missing_ids` : 20~30개 샘플

**기각 조건**: 실패 id 들이 타겟 테이블 (e.g. `dim_stop`) 의 모든 컬럼 어디에도 안 보이면 → 가설 A **기각** (조인 키 문제가 아님).

**확정 조건**: 실패 id 가 `dim_stop.bis_id` 같은 alt-key 컬럼에 **대량으로** 보이면 → 가설 A **확정** → `dim_stop_promotion_guard` 로 alt-key 승격 후 재시도.

## Stage D2 — Dynamic Bridge/Staging Probe (가설 B 검증)

"원천과 타겟 사이에 이미 만들어진 매핑/브리지 테이블이 있는가?" 확인.

후보 테이블 이름 규칙:
- `%mapping%`, `%bridge%`, `%xref%`, `%lookup%`
- `stg_%`, `staging_%`, `raw_%`
- `err_%_failed` (ETL 실패 로그 — 거꾸로 매칭 힌트)

후보 컬럼 타입 필터:
- `information_schema.columns` 에서 `data_type IN ('text', 'varchar', 'character varying', 'bigint', 'integer', 'numeric')`

기법:
```sql
DO $$
DECLARE r record; hit_count bigint;
BEGIN
  FOR r IN
    SELECT table_schema, table_name, column_name
    FROM information_schema.columns
    WHERE table_name ~* '(mapping|bridge|staging|stg_|raw_)'
      AND data_type IN ('text','bigint','integer','character varying','numeric')
  LOOP
    EXECUTE format(
      'SELECT count(*) FROM %I.%I WHERE %I::text = ANY($1)',
      r.table_schema, r.table_name, r.column_name
    ) INTO hit_count USING missing_id_array;
    IF hit_count > 0 THEN
      RAISE NOTICE '%.%.%  -> % hits', r.table_schema, r.table_name, r.column_name, hit_count;
    END IF;
  END LOOP;
END $$;
```

**기각 조건**: 모든 후보 테이블 × 모든 후보 컬럼에 대해 매칭 수가 **전부 0** → 가설 B **기각**.

**확정 조건**: 특정 `(테이블, 컬럼)` 조합에서 매칭이 수십~수백건 이상 → 가설 B **확정** → 해당 브리지를 조인에 추가하는 방향.

## Stage D3 — Hotspot Semantic Pattern Tagging (가설 D 검증)

"한 노선/한 단위 의 row 들을 의미 태그 한 뒤 패턴을 보면, 의미가 기대랑 다른가?" 검증.

**Recipe (graph edges 예시)**:
```sql
-- 단일 (route_id, move_dir_code) 핫스팟 1개 고름 (link 수 상위)
WITH hot AS (
  SELECT route_id, move_dir_code, count(*) AS links
  FROM route_link_sequence
  GROUP BY 1,2
  ORDER BY links DESC LIMIT 1
),
tagged AS (
  SELECT r.link_seq, r.st_node_id, r.ed_node_id,
         CASE WHEN d1.stop_id IS NOT NULL THEN 'S' ELSE '-' END AS st_tag,
         CASE WHEN d2.stop_id IS NOT NULL THEN 'S' ELSE '-' END AS ed_tag
  FROM route_link_sequence r
  JOIN hot h ON h.route_id = r.route_id AND h.move_dir_code = r.move_dir_code
  LEFT JOIN dim_stop d1 ON d1.stop_id = r.st_node_id
  LEFT JOIN dim_stop d2 ON d2.stop_id = r.ed_node_id
)
SELECT st_tag || '->' || ed_tag AS pattern, count(*)
FROM tagged GROUP BY 1 ORDER BY 2 DESC;
```

**판정 기준**:
- `S->S` 가 95%+ → 가설 D **기각** (의미론은 stop-to-stop 맞음. 진짜 데이터가 부족한 것일 수 있음 → C 로)
- `S->-` / `-->S` / `-->` 가 섞여서 30%+ → 가설 D **확정** (시퀀스가 stop + intermediate node 혼합) → leg-aggregation 필요 → `stop_to_stop_edge_leg_aggregator` 로 이관.

**대구 실측 (2026-04-19)**: `S→S 32.41% / S→- 32.39% / -→S 23.01% / -→- 12.19%` → 가설 D 확정.

## Stage D4 — Evidence Aggregation

A/B/D 세 가지 가설 각각에 대해 아래 네 칸이 채워진 표를 만들고, 그 결과를 **`{pipeline_pkg}/diagnose/evidence_{YYYYMMDD}.md`** 에 보관.

| 가설 | 진단 방법 | 결과 요약 | 판정 |
|------|---------|--------|------|
| A    | D1 missing prefix → dim_stop 전 컬럼 스캔 | 1,482 missing, dim_stop 7컬럼 어디에도 부재 | **기각** |
| B    | D2 staging/bridge 5-테이블 dynamic probe | 5 테이블 × N 컬럼 전부 0 매치 | **기각** |
| D    | D3 핫스팟 노선 패턴 태깅 | S→S 32%, 혼합 섞임 | **확정** |
| C    | (A/B/D 로 충분히 설명됨) | - | 불필요 |

---

# Failure Modes

## F1. 증거 한 가지만으로 섣불리 가설 확정

**전형 증상**: "prefix 가 특정하니 alt-key 가설 A 다!" 라고 외치고 바로 alt-key 스크립트 작성. 그런데 실제로는 그 prefix 들이 dim_stop 어디에도 없어서 결국 D 였음.

**해결**: 가설 확정 전에 반드시 **"반대 증거" 도 찾아본다**. A 를 확정하려면 "alt-key 컬럼에서 실제로 매칭되는 건수 > 0" 까지 확인해야 함.

## F2. 브리지 테이블 이름으로만 찾고 컬럼 스캔 안 함

**전형 증상**: `stop_link_mapping_master` 를 이름 보고 "이게 브리지 맞네" 했는데, 정작 컬럼엔 missing id 가 없어서 무의미.

**해결**: D2 는 **"이름 + 컬럼 실제 매칭 수"** 모두 확인. 이름만으로는 판단 금물.

## F3. 핫스팟 하나로 가설 D 를 일반화

**전형 증상**: 노선 1개만 보고 "이 시퀀스는 혼합이네" 결정. 다른 노선은 깨끗한 stop-only 일 수도 있어서 오진.

**해결**: D3 는 **핫스팟 1개 + 전체 분포** 두 가지를 **같이** 확인. 전체 분포에서도 혼합이 30%+ 이면 그때 일반화.

## F4. 진단 SQL 을 저장 안 하고 터미널에서 날림

**전형 증상**: 한 달 뒤 다른 도시에서 같은 증상 발생했는데 재현이 안 됨.

**해결**: 진단 SQL 은 반드시 `D_*.sql`, `D2_*.sql`, `D3_*.sql` 번호를 매겨 `{pipeline_pkg}/diagnose/` 에 commit.

---

# Template / References

- **references/diagnose_template.sql**
  4-Stage (D1/D2/D3/D4) 풀세트. 도시/유스케이스별로 `-- @CHANGE_FOR_USE_CASE`
  표시된 라인만 치환. D1 key pattern, D2 dynamic probe, D3 hotspot tagging,
  D4 evidence aggregation 까지 한 파일에 통합.

---

# Use Case Adaptation Checklist

다른 도시 / 다른 파이프라인 적용 시 다음 항목만 바꾼다.

1. **원천 테이블명**: `route_link_sequence` / `stg_xxx` 등 → 해당 도시 스키마
2. **타겟 테이블명**: `graph_edge_master` → 타겟 엣지/adjacency 테이블
3. **조인 키 컬럼**: `stop_id` → 해당 도시의 stop PK (`bs_id`, `node_uid` 등)
4. **"stop 인가?" 판정 조건**: D3 의 tag CASE 를 해당 도시 dim 테이블 기준으로
5. **partition 키**: 엣지를 구성하는 단위 (`(route_id, move_dir_code)`) 는 도시 불변일 가능성 큼. 다르면 여기서 치환.
6. **핫스팟 선택 로직**: D3 의 `ORDER BY links DESC LIMIT 1` 은 상위 1개. 분포가 왜곡된 도시면 LIMIT 3~5 로 늘려 평균적 패턴 확인.

---

# Skill Relationships

- **선행 스킬**:
  - `data_contract_audit` — readiness 가 낮다는 걸 **최초 발견** 하는 게이트
  - `shape_inspection` / `schema_inspector` — 후보 테이블/컬럼 구조 사전 이해
- **후행 스킬 (가설별 분기)**:
  - 가설 A 확정 → `dim_stop_promotion_guard` (alt-key 승격)
  - 가설 B 확정 → 브리지-aware 조인 재작성 (future skill: `bridge_join_composer`)
  - 가설 C 확정 → ETL runbook 재실행
  - 가설 D 확정 → `stop_to_stop_edge_leg_aggregator` (leg-aggregation)
- **동시 활용**:
  - `spatial_graph_mapping_guard` — 공간 매핑 레벨의 coverage 문제인지 동시에 의심될 때

---

# Historical Context

**대구 2026-04-19 케이스** — graph_edge_master STOP_TO_STOP 22.84% → 99.91%:
- **D1** (D_diagnose_stop_id_mapping.sql): missing 1,482 / 전부 길이 10 / prefix `15xxx, 30007xxx, 73611xxx` → dim_stop 7컬럼 전수 스캔 → **전부 부재** → A 기각.
- **D2** (D2_probe_mapping_tables.sql): 5개 후보 브리지 테이블 × 전 text/numeric 컬럼 dynamic EXECUTE → **전부 0 매치** → B 기각.
- **D3** (D3_link_vs_stop_hypothesis.sql): 핫스팟 `7361109008:dir=1` 시퀀스 태깅 → S/- 교차 패턴 선명, 전체 분포 S→S 32%/S→- 32%/-→S 23%/-→- 12% → **D 확정**.
- 조치: `02_load v2` 로 leg-aggregation 재설계 → 99.91% 달성.

이 케이스 전체 과정이 곧 이 스킬의 표준 시나리오.

---

# 호출 프롬프트 예시

- "서울시 graph_edge_master 적재 후 coverage 가 45% 로 나왔어. `db_coverage_gap_diagnoser` 스킬의 4-가설 프레임워크로 D1→D2→D3 순서로 진단해줘."
- "부산시 stop_to_link 매핑 coverage 가 60% 대. alt-key 가설 A 를 먼저 확인하는 D1 스크립트 작성해줘 (`db_coverage_gap_diagnoser` 스킬)."
- "대전시 route_link_sequence 데이터가 들어왔는데 대구 때처럼 mixed sequence 일 수 있어. D3 핫스팟 태깅만 먼저 돌려서 패턴 확인해줘."
- "지난 대구 케이스에서 했던 D1/D2/D3 를 참고해서 현재 case 용으로 `-- @CHANGE_FOR_USE_CASE` 부분 치환한 diagnose_template.sql 만들어줘."
