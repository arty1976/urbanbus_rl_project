# Route Link Promotion Execution Runbook

## 목적
- `stg_daegu_route_links_api` 에서 `route_link_sequence` 승격 전후 실행 절차를 표준화하기 위한 문서입니다.

## 실행 전 체크리스트
- 예외 노선 CSV 확인: `data/processed/daegu/excluded_route_links_api.csv`
- 수집 상태 문서 확인: `01_data_contracts/route_links_collection_status.md`
- 판독 기준표 확인: `01_data_contracts/route_link_promotion_result_guide.md`
- 결과 리포트 템플릿 확인: `01_data_contracts/route_link_promotion_result_report_template.md`
- 검증 SQL 확인: `03_validation_queries/route_link_promotion_readiness.sql`
- 승격 실행 전 DB 연결 정보 확인
- scripts/configs 수정 없이 진행

## 검증만 실행하는 절차
```powershell
psql -X -h localhost -p 5432 -U postgres -d urbanbus -f .\03_validation_queries\route_link_promotion_readiness.sql
```
- 결과를 `01_data_contracts/route_link_promotion_result_report_template.md` 에 기록합니다.

## Guard 포함 승격 실행 절차
```powershell
.\scripts\run_promote_route_link_sequence.ps1 `
  -DbName "urbanbus" `
  -DbUser "postgres" `
  -DbHost "localhost" `
  -DbPort 5432 `
  -PsqlPath "C:\Program Files\PostgreSQL\18\bin\psql.exe"
```
- Guard 검증 실패 시 승격을 중단해야 합니다.

## 실행 직후 확인 사항
- summary 결과 확인
- unexpected_route_in_promoted 확인
- count_mismatch_by_route_dir 확인
- key_gap_between_staging_and_promoted 확인
- promoted_continuity_break 확인
- link_id_collision_between_staging_and_promoted 확인

## 결과 리포트 작성
- `01_data_contracts/route_link_promotion_result_report_template.md` 사용
- summary 기대값과 실제값 비교
- 상세 쿼리별 행 수 및 판정 기록
- 최종 상태를 승인 / 보류 / 차단 중 하나로 표기

## 최종 판정 기준
- 승인: 실패 쿼리 없음, promoted 오염 없음, 주요 불일치 없음
- 보류: staging legacy sample `1000` 등 경고성 이슈만 존재
- 차단: promoted 오염, count mismatch, key gap, continuity break, link_id 충돌, 자연키 중복 중 하나라도 존재

## 차단 시 후속 조치
- 결과 리포트에 차단 사유 기록
- 원인 쿼리 재확인
- source/staging/promoted 범위 재대조
- 필요 시 승격 로직 수정 후 재검증
- 수정 전 무단 실행 금지
