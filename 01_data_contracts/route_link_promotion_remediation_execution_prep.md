# Route Link Promotion Remediation Execution Prep

- 작성일: 2026-04-10
- 문서 성격: **실행 전 최종 확인 결과 및 행동 강령**
- 상태: **수정 적용 대기 (읽기 전용 확인 완료)**

---

## 1. 읽기 전용 확인 결과 요약

### 1-1. route_id='1000' 오염 행 수 및 분포 (promoted 테이블)
`public.route_link_sequence` 테이블 내 레거시 샘플 노선(`1000`)의 데이터 규모를 확정했습니다.

- **총 오염 행 수**: `287 rows`
- **방향별 분포**:
  - `move_dir_code = '0'`: `144 rows`
  - `move_dir_code = '1'`: `143 rows`
- **조치 요건**: 승격 롤백 및 정리를 위해 정확히 이 287행만 삭제되어야 함.

### 1-2. Staging 테이블 스키마 확인
`public.stg_daegu_route_links_api` 테이블의 실제 스키마(`ingested_at` 및 주요 식별자)를 재확인했습니다.

- **`ingested_at` 존재 여부**: 없음. 대신 **`loaded_at`** (timestamptz) 컬럼이 존재함.
- **주요 식별자 컬럼명 주의**: 원시값이므로 `route_id_raw`, `move_dir_raw`, `link_seq_raw` 형태임.

---

## 2. Dedup(중복 제거) 방식 최종 선택

**선택**: **옵션 A (Dedup 뷰 생성 전략)** 및 **승격 SQL(`route_link_promotion_readiness.sql`, `promote_route_link_sequence.sql`)의 소스 교체 방식** 채택.

### 결정 사유
1. **원본 보존**: 원시 적재 테이블인 `stg_daegu_route_links_api`의 원본 데이터(특히 `snapshot_file` 이력)를 파괴하지 않음.
2. **안전성 보장**: 대규모 `DELETE` 문(파괴적 연산) 실행에 따른 락(lock), 인덱스 단편화, 그리고 휴먼 에러의 위험을 원천 차단함.

### 적용 설계안 (미실행)
```sql
CREATE OR REPLACE VIEW public.vw_stg_daegu_route_links_api_dedup AS
SELECT DISTINCT ON (route_id_raw, move_dir_raw, link_seq_raw)
    *
FROM public.stg_daegu_route_links_api
ORDER BY route_id_raw, move_dir_raw, link_seq_raw, loaded_at DESC;
```
> 이후 추출, 변환 로직에서 `stg_daegu_route_links_api` 대신 위 뷰(`vw_stg_daegu_route_links_api_dedup`)를 참조하도록 변경해야 합니다.

---

## 3. 다음 단계 가이드 (준비 단계)

> **주의**: 현재는 읽기 전용 확인만 완료된 "준비 단계"이며, 실제 적용 쿼리를 작성/구현하는 다음 단계로 넘어가야 합니다. **아직 승격 실행은 절대 금지됩니다.**

**다음 실행 단계 목표:**
1. **정리 실행**: `route_link_sequence` 에서 `route_id = '1000'` 데이터(287행)를 `DELETE`.
2. **Dedup 적용**: `vw_stg_daegu_route_links_api_dedup` 뷰를 생성.
3. **로직 연동**: 검증 및 승격 SQL 쿼리들이 위 dedup 뷰를 참조하도록 `.sql` 문서 업데이트.

---

**[엄수 사항]**
- **승격 실행 금지**: Dedup이 완전히 검증되기 전까지 `run_promote_route_link_sequence.ps1` 등을 실행해서는 안 됩니다.
- **staging 대량 삭제 금지**: 원본 테이블에 직접적인 `DELETE` 연산을 배제합니다.
