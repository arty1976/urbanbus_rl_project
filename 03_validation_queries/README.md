# Data Validation Queries

이 폴더는 데이터 전처리가 끝난 도시 버스 데이터가 데이터베이스에 올바르게 적재되었는지 검증하기 위한 SQL 스크립트들을 포함하고 있습니다.
PostgreSQL 환경에 맞추어 작성되었으며, 각 쿼리는 데이터를 시각적으로 확인하거나 비정상적인 건수를 추출하도록 설계되었습니다. 테이블 내용이 비어있을 땐 결과가 없는 정상 상태로 반환됩니다.

## 파일 구성 및 실행 순서

데이터의 기본 무결성을 먼저 확인한 뒤, 테이블 간의 관계 무결성을 검증하는 것을 권장합니다.

### 1단계: 기본 무결성 검증 (`basic_checks.sql`)
개별 테이블 단위로 필수 조건들이 잘 지켜지고 있는지 확인합니다.
- 식별자(PK 성격의 컬럼)의 중복 여부 (`dim_stop`, `dim_route`)
- 필수 데이터인 공간 좌표(`latitude`, `longitude`)의 누락 여부
- 팩트 데이터의 논리적 오류 검사 (예: `board_count`, `alight_count`, `transfer_count`에서의 음수 발생)

### 2단계: 관계 및 무결성 검증 (`join_checks.sql`)
연결 테이블(`bridge_route_stop`)과 차원 테이블 간의 조인 관계 및 순서의 논리적 오류를 점검합니다.
- 참조 무결성 오류(차원 테이블에 없는 ID를 참조하는 경우)
- 정류장 순서(`stop_sequence`)의 연속성 문제(결측치)
- 관계가 맺어지지 않은(누락된) 데이터의 전체적인 요약 수치 확인

### 3단계: 노선 링크 승격 검증 (Operational Standard)
- **운영 기준본**: [route_link_promotion_readiness_dedup.sql](./route_link_promotion_readiness_dedup.sql)
  - `vw_stg_daegu_route_links_api_dedup` 뷰를 사용하여 중복 제거가 반영된 무결성을 검증합니다.
- **레거시 (Legacy/Audit)**: `route_link_promotion_readiness.sql` (최초 검증 로직 보존용)
- **주요 검증 항목**:
  - 원천 `stg_daegu_routes` 기준 234개 노선 정합성 확인
  - 자연키 `(route_id, move_dir_code, link_seq)` 기준 중복, 누락, 건수 불일치 확인
  - `link_id` 충돌 및 노선 연속성 단절 검사

### 4단계: 승격 결과 판독 기준표 (`../01_data_contracts/route_link_promotion_result_guide.md`)
`route_link_promotion_readiness.sql` 실행 결과를 PASS / WARN / FAIL 로 해석하는 기준표입니다.
- summary 기대값(238 / 4 / 234 / 234 / 234) 확인
- `unexpected_route_in_staging` 에서 legacy sample `1000` 잔존 시 WARN 처리
- promoted 오염, 자연키 중복, 건수 불일치, key gap, 연속성 단절, `link_id` 충돌은 모두 FAIL 처리

### 5단계: 결과 리포트 작성 (`../01_data_contracts/route_link_promotion_result_report_template.md`)
실제 검증 결과를 붙여서 최종 판정(승인 / 보류 / 차단)을 남기는 템플릿입니다.
- summary 실제값과 판정을 한 번에 기록
- 상세 쿼리별 행 수, 판정, 비고를 표로 정리
- 결과 원문을 그대로 붙여 재검토 가능하게 유지
- 최종 상태와 후속 조치를 즉시 기록

## 실행 방법 및 체크리스트
1. **표준 운영 절차**: [운영 체크리스트](../01_data_contracts/route_link_operational_checklist.md)를 따라 순차적으로 실행합니다.
2. **실행 환경**: 사용 중인 DB 클라이언트나 `psql`에서 위의 순서대로 SQL 파일을 실행합니다.
3. **판독**: 결과 행(Row)이 반환된다면 데이터에 문제가 있음을 의미합니다. `_dedup.sql` 버전은 예외 4개 노선과 legacy sample `route_id=1000` 을 필터링하여 정확한 운영 상태를 보여줍니다.
