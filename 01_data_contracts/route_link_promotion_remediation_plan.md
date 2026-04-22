# Route Link Promotion Remediation Plan

- 작성일: 2026-04-10
- 기준 문서: `01_data_contracts/route_link_promotion_result_report_filled.md`
- 원본 출력: `validation_output_utf8.txt`
- 상태: **차단(BLOCK) — 실행 전 수정 필요**

---

## 1. 현재 차단 사유 요약

| 순위 | check_name | 행 수 | 판정 | 비고 |
|---|---|---:|---|---|
| 1 | `unexpected_route_in_promoted` | 2 | FAIL | promoted 테이블 오염 확인 |
| 2 | `staging_dup_natural_key` | 38,960 | FAIL | 자연키 중복 적재 |
| 3 | `count_mismatch_by_route_dir` | 346 | FAIL | staging 건수 ≠ promoted 건수 |
| 4 | `key_gap_between_staging_and_promoted` | 38,960 | FAIL | staging 키가 promoted에 없음 |

PASS 항목: `missing_route_in_staging`, `promoted_dup_natural_key`, `promoted_continuity_break`, `link_id_collision`

---

## 2. unexpected_route_in_promoted — 정확한 식별

### 실제 출력 (validation_output_utf8.txt line 38981~38982)

```text
          check_name          | route_id | move_dir_code | min_link_seq | max_link_seq | row_count |       reason
------------------------------+----------+---------------+--------------+--------------+-----------+---------------------
 unexpected_route_in_promoted | 1000     | 0             |            1 |          144 |       144 | legacy_sample_route
 unexpected_route_in_promoted | 1000     | 1             |            1 |          143 |       143 | legacy_sample_route
(2 rows)
```

### 해석

| 항목 | 값 |
|---|---|
| 오염 route_id | `1000` (단일 route_id, 방향 2개) |
| move_dir_code | `0` (144건), `1` (143건) |
| 총 오염 행 수 | **287행** (144 + 143) |
| 원인 분류 | `legacy_sample_route` |

> **관찰**: route_id=`1000`은 `unexpected_route_in_staging`에도 등장하며, `stg_daegu_routes`에는 없는 레거시 샘플 데이터다.
> `route_link_sequence`에 이 데이터가 존재한다는 것은 이전 테스트 또는 초기 적재 시 잘못 승격된 것으로 보인다.

### 확인 필요 사항 (실행 전 검토)

```sql
-- route_link_sequence에서 route_id=1000 존재 확인 (읽기 전용)
SELECT route_id, move_dir_code, COUNT(*) AS row_count
FROM public.route_link_sequence
WHERE route_id = '1000'
GROUP BY route_id, move_dir_code
ORDER BY move_dir_code;
```

---

## 3. staging_dup_natural_key — 중복 원인 분석

### 실제 출력 대표 사례

```text
        check_name         |   route_id   | move_dir_code | link_seq | dup_count
---------------------------+--------------+---------------+----------+-----------
 staging_dup_natural_key   | 1000001000   | 0             |        1 |         3
 staging_dup_natural_key   | 1000001000   | 0             |        2 |         3
 ...
 staging_dup_natural_key   | 1000003000   | 0             |      113 |         3
 ...
 staging_dup_natural_key   | 7361111009   | 1             |      108 |         2
(38960 rows)
```

### 관찰

| 항목 | 값 |
|---|---|
| 총 중복 행 수 | 38,960 |
| dup_count 패턴 | 대부분 `3`, 일부 `2` (7361111009 등) |
| 영향 route_id 범위 | `1000001000` ~ `7361111009` (전 노선에 걸쳐 존재) |
| 자연키 | `(route_id, move_dir_code, link_seq)` |

### 중복 원인 가설

**가설 A: API 다중 호출 중복 적재 (가장 유력)**
- 동일 route_id를 여러 번 API 호출하여 staging에 append 방식으로 적재
- `fetch_bus_api.ps1` 또는 PowerShell 인제스트 스크립트가 멱등성(idempotency) 없이 실행된 경우
- dup_count=3은 동일 노선을 3회 적재했음을 시사

**가설 B: batch 분할 실행 중복**
- 노선을 batch로 나눠 실행하면서 일부 노선이 여러 batch에 중복 포함
- 일부 노선만 dup_count=2인 것이 이를 지지할 수 있음

**가설 C: staging 테이블 truncate 없이 재적재**
- 재실행 시 기존 데이터를 지우지 않고 INSERT만 반복

### 확인 필요 쿼리 (읽기 전용 제안)

```sql
-- 중복 분포 확인
SELECT dup_count, COUNT(DISTINCT route_id) AS affected_routes
FROM (
    SELECT route_id, move_dir_code, link_seq,
           COUNT(*) AS dup_count
    FROM public.stg_daegu_route_links_api
    GROUP BY route_id, move_dir_code, link_seq
    HAVING COUNT(*) > 1
) t
GROUP BY dup_count
ORDER BY dup_count;

-- 동일 route_id의 ingested_at 분포 확인 (적재 시점 확인)
SELECT route_id, COUNT(DISTINCT ingested_at) AS distinct_load_times, COUNT(*) AS total_rows
FROM public.stg_daegu_route_links_api
WHERE route_id = '1000001000'
GROUP BY route_id;
```

---

## 4. count_mismatch_by_route_dir — 분석

### 실제 출력 대표 사례

```text
         check_name          |  route_id  | move_dir_code | staging_distinct_cnt | promoted_cnt | diff
-----------------------------+------------+---------------+----------------------+--------------+------
 count_mismatch_by_route_dir | 1000001000 | 0             |                  144 |            0 | -144
 count_mismatch_by_route_dir | 1000001000 | 1             |                  143 |            0 | -143
 count_mismatch_by_route_dir | 1000002001 | 0             |                  126 |            0 | -126
 ...
 count_mismatch_by_route_dir | 7361111009 | 1             |                  108 |            0 | -108
(346 rows)
```

### 해석

| 항목 | 값 |
|---|---|
| 전체 불일치 건 수 | 346개 (route_id, move_dir_code) 조합 |
| `promoted_cnt` 값 | **전부 0** |
| `diff` 방향 | 전부 음수 (staging에만 있고 promoted에는 없음) |
| 결론 | 승격이 실질적으로 수행된 적이 없음을 재확인 |

> **해석**: `count_mismatch_by_route_dir`의 346건은 모두 `promoted_cnt=0`이다.
> 이는 독립적인 데이터 오류가 아니라 **승격 미실행의 결과**이다.
> staging dedup 완료 후 승격 실행 시 자동 해소될 가능성이 높다.

---

## 5. key_gap_between_staging_and_promoted — 분석

### 대표 사례

```text
              check_name                   |  route_id  | move_dir_code | link_seq |     issue_type
-------------------------------------------+------------+---------------+----------+--------------------
 key_gap_between_staging_and_promoted      | 7361110011 | 1             |      158 | missing_in_promoted
 ...
 key_gap_between_staging_and_promoted      | 7361111009 | 1             |      108 | missing_in_promoted
(38960 rows)
```

### 해석

- 전체 38,960행이 전부 `missing_in_promoted` 방향
- `missing_in_staging` 방향은 0건 (staging에 없는 키가 promoted에 있는 경우 없음)
- route_id=`1000`의 287행 오염을 제외하면 promoted는 사실상 비어있는 상태

---

## 6. 수정 전략 및 처리 순서

### 6-1. 처리 우선순위

```
[Step 1] route_link_sequence 오염 정리 (route_id=1000 제거)
[Step 2] stg_daegu_route_links_api 중복 해소 (dedup)
[Step 3] 검증 재실행 (route_link_promotion_readiness.sql)
[Step 4] Guard 포함 승격 실행
[Step 5] 검증 최종 확인
```

### 6-2. Step 1 — promoted 오염 정리 (제안안, 미실행)

```sql
-- [제안안 — 실행 전 반드시 검토]
-- 사전 확인: 삭제 대상 건수
SELECT COUNT(*) FROM public.route_link_sequence WHERE route_id = '1000';

-- 실제 삭제 (Step 3 재검증 전에 실행)
DELETE FROM public.route_link_sequence WHERE route_id = '1000';
-- 예상 삭제 건수: 287행 (move_dir_code=0: 144행, move_dir_code=1: 143행)
```

> **주의**: 삭제 전에 `route_link_sequence`에 다른 데이터가 있는지 반드시 확인.
> `promoted_distinct_routes_all=1`에서 해당 1건이 route_id=1000임을 검증 결과로 확인.

### 6-3. Step 2 — staging dedup 처리 전략

**옵션 A: dedup 뷰 생성 (추천 — 원본 불변, 안전)**

```sql
-- [제안안 — 미실행]
CREATE OR REPLACE VIEW public.stg_daegu_route_links_api_dedup AS
SELECT DISTINCT ON (route_id, move_dir_code, link_seq)
    *
FROM public.stg_daegu_route_links_api
ORDER BY route_id, move_dir_code, link_seq, ingested_at DESC;
-- ingested_at 컬럼 존재 가정 — 없으면 ctid 또는 임의 ORDER 사용
```

**옵션 B: 테이블 내 중복 직접 제거**

```sql
-- [제안안 — 미실행, 파괴적 연산 — 주의]
-- 중복 중 최신 1건만 남기고 삭제
DELETE FROM public.stg_daegu_route_links_api
WHERE ctid NOT IN (
    SELECT DISTINCT ON (route_id, move_dir_code, link_seq) ctid
    FROM public.stg_daegu_route_links_api
    ORDER BY route_id, move_dir_code, link_seq, ingested_at DESC
);
```

> **권장**: 옵션 A(뷰 방식)를 먼저 사용하여 승격 로직만 변경, 원본 보존.
> staging 원본 삭제는 승격 성공 확인 후 별도 결정.

### 6-4. Step 2 실행 전 확인 필요 사항

```sql
-- [읽기 전용 확인 쿼리]
-- 1. ingested_at 컬럼 존재 여부
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'stg_daegu_route_links_api'
  AND column_name = 'ingested_at';

-- 2. 전체 중복 규모 재확인
SELECT COUNT(*) AS dup_rows,
       COUNT(DISTINCT (route_id, move_dir_code, link_seq)) AS dup_key_combos
FROM public.stg_daegu_route_links_api
WHERE (route_id, move_dir_code, link_seq) IN (
    SELECT route_id, move_dir_code, link_seq
    FROM public.stg_daegu_route_links_api
    GROUP BY route_id, move_dir_code, link_seq
    HAVING COUNT(*) > 1
);
```

---

## 7. 재검증 순서

```
1. [확인] SELECT COUNT(*) FROM route_link_sequence WHERE route_id='1000'
   → 287이면 Step 1 실행 대상 확정

2. [실행] DELETE FROM route_link_sequence WHERE route_id='1000'

3. [확인] ingested_at 컬럼 유무 확인 후 dedup 전략(A/B) 결정

4. [실행] dedup 뷰 생성 또는 테이블 직접 정리 (결정에 따라)

5. [재실행] route_link_promotion_readiness.sql
   → staging_dup_natural_key = 0, unexpected_route_in_promoted = 0 확인

6. [실행] Guard 포함 승격 (route_link_promotion_execution_runbook.md 참조)

7. [최종 검증] route_link_promotion_readiness.sql 재실행
   → 전 항목 PASS/WARN 확인
```

---

## 8. 실행 전 금지사항

- 승격 실행 금지 (dedup 해소 전)
- staging 대량 삭제 금지 (dedup 전략 결정 전)
- 위 SQL을 확인 없이 실행 금지
- config 및 scripts 수정 금지
- raw 원본 데이터 덮어쓰기 금지

---

## 9. 관련 문서

| 문서 | 역할 |
|---|---|
| `01_data_contracts/route_link_promotion_result_report_filled.md` | 검증 결과 전체 기록 |
| `01_data_contracts/route_link_promotion_result_guide.md` | PASS/WARN/FAIL 판정 기준 |
| `02_ingest_jobs/route_link_promotion_execution_runbook.md` | 승격 실행 절차 |
| `03_validation_queries/route_link_promotion_readiness.sql` | 재검증 SQL |
| `validation_output_utf8.txt` | 현재 검증 출력 원본 |
