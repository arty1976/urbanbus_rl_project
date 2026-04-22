# Route Link Operational Checklist (운영 체크리스트)

이 문서는 `route_link_sequence` 테이블로의 데이터 승격(Promotion) 작업을 반복 수행할 때 데이터 무결성을 보장하기 위한 표준 운영 체크리스트입니다.

## 1. 실행 전 확인 (Pre-run Verification)
- [ ] `public.stg_daegu_route_links_api` 테이블에 신규 데이터가 정상적으로 적재되었는가?
- [ ] `public.vw_stg_daegu_route_links_api_dedup` 뷰(View)가 존재하며, 조회 시 중복이 제거된 행이 반환되는가?
- [ ] 승격 대상 `route_id` 리스트가 확보되었는가?

## 2. 검증 단계 (Validation Phase)
- [ ] **운영 기준 검증 SQL 실행**: `03_validation_queries/route_link_promotion_readiness_dedup.sql`
- [ ] **결과 분석**:
    - [ ] `eligible_route_count_status` 가 **PASS** 인가?
    - [ ] `staging_dup_natural_key` 가 **0행** 인가?
    - [ ] `unexpected_route_in_promoted` 가 **0행** 인가? (1000번 노선 등 오염 확인)

## 3. 승격 단계 (Promotion Phase)
- [ ] (필요 시) `route_link_sequence` 에서 기존 오염 데이터(`route_id = '1000'`) 삭제를 먼저 수행하였는가?
- [ ] **운영 기준 승격 SQL 실행**: `02_ingest_jobs/promote_route_link_sequence_dedup.sql`
- [ ] SQL 실행 로그에서 `INSERT` 건수가 예상 범위 내에 있는가?

## 4. 사후 검증 (Post-run Verification)
- [ ] **검증 SQL 재실행**: `03_validation_queries/route_link_promotion_readiness_dedup.sql`
- [ ] **결과 분석**:
    - [ ] `count_mismatch_by_route_dir` 가 **0행** 인가?
    - [ ] `key_gap_between_staging_and_promoted` 가 **0행** 인가?
    - [ ] `promoted_continuity_break` 가 **0행** 인가?

## 5. 실패 시 중단 조건 (Termination Conditions)
- [ ] **상세 쿼리 결과( 상세 쿼리 1~9 ) 중 단 하나라도 0행이 아닌 FAIL 결과가 나오면 작업을 즉시 중단하고 원인을 파악한다.**
- [ ] 특히 `dup_natural_key` 나 `continuity_break` 발생 시에는 분석 레이어의 신뢰도가 파괴되므로 수리(Remediation) 전까지 승격을 금지한다.

## 6. 산출물 저장 (Artifacts)
- [ ] 실행 결과(Summary 및 상세 결과)를 `01_data_contracts/` 하위 리포트 문서(템플릿 기준)로 작성하여 저장한다.
- [ ] 최종 판정(승인/보류/차단)을 팀에 공유한다.
