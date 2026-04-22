# Route Link Promotion Result Guide

이 문서는 `03_validation_queries/route_link_promotion_readiness.sql` 실행 결과를 어떻게 판독할지 정리한 기준표입니다.

## 적용 범위
- 원천 기준 노선 수: 238
- 예외 노선: 4
  - `4040006020`
  - `4040006021`
  - `4040007001`
  - `4040007009`
- 유효 검증 대상 노선 수: 234
- legacy sample route: `1000`

본 기준표는 **예외 4개 제외 + legacy sample `1000` 제외** 규칙을 전제로 합니다.

## 최종 판정 규칙
- **PASS**
  - summary 수치가 기대값과 일치한다.
  - `unexpected_route_in_promoted` 포함 상세 오류 쿼리가 모두 0행이다.
- **WARN**
  - `unexpected_route_in_staging` 결과에 `route_id='1000'` 1건만 남아 있다.
  - 즉, staging 에만 legacy sample 이 남아 있고 promoted 결과는 오염되지 않았다.
- **FAIL**
  - summary 기대값이 어긋난다.
  - `unexpected_route_in_promoted` 결과가 1행 이상이다.
  - 중복, 누락, 건수 불일치, 연속성 단절, `link_id` 충돌이 1행 이상이다.
  - `unexpected_route_in_staging` 결과에 `1000` 이외 route 가 나타난다.

## Summary 판독 기준

| section | 기대값 | 판독 |
|---|---:|---|
| `source_routes_total` | 238 | 원천 `stg_daegu_routes` 기준 전체 노선 수 |
| `excluded_route_count` | 4 | 예외 노선 수 |
| `eligible_source_routes` | 234 | 실제 승격 검증 대상 노선 수 |
| `staging_effective_distinct_routes` | 234 | staging 에서 예외 4개와 `1000` 제외 후 남는 유효 노선 수 |
| `promoted_distinct_routes_all` | 234 | promoted 에 예외 노선과 `1000` 이 없어야 하므로 최종 합격 기준은 234 |
| `promoted_effective_distinct_routes` | 234 | promoted 유효 노선 수 |
| `eligible_route_count_status` | `PASS` | 검증 대상 노선 수 자체가 올바르게 계산되었는지 확인 |

### Summary 해석 규칙
- `staging_effective_distinct_routes < 234`
  - staging 누락이 있으므로 `missing_route_in_staging` 결과를 먼저 본다.
- `promoted_distinct_routes_all > 234`
  - 예외 노선 또는 legacy sample 이 promoted 에 섞였을 가능성이 높다.
- `promoted_effective_distinct_routes < 234`
  - 승격 누락 또는 승격 중 일부 route/move_dir 누락 가능성이 높다.
- `eligible_route_count_status = FAIL`
  - 전제 자체가 틀렸으므로 다른 상세 결과보다 먼저 원천 scope 를 재확인한다.

## 상세 쿼리 판독 기준표

| check_name | 정상 기준 | 판정 | 의미 | 우선 조치 |
|---|---|---|---|---|
| `missing_route_in_staging` | 0행 | FAIL | 원천 대상 234개 중 staging 에 안 들어온 노선 존재 | 수집/적재 로그 재확인, 예외 노선 재분류 필요 여부 점검 |
| `unexpected_route_in_staging` | 0행이 이상적 | `1000`만 있으면 WARN, 그 외는 FAIL | staging 에 원천 대상 외 route 존재 | `1000`이면 legacy residue 로 분리 관리, 그 외 route 는 source/staging 매핑 점검 |
| `staging_dup_natural_key` | 0행 | FAIL | `(route_id, move_dir_code, link_seq)` 중복 | 원천 snapshot 중복 또는 적재 중복 확인 |
| `unexpected_route_in_promoted` | 0행 | FAIL | promoted 에 예외 4개 또는 `1000` 존재 | 승격 범위 필터/정리 절차 재점검 |
| `promoted_dup_natural_key` | 0행 | FAIL | promoted 자연키 중복 | upsert 키, 제약조건, 기존 잔존 데이터 점검 |
| `count_mismatch_by_route_dir` | 0행 | FAIL | route + 방향 단위 건수가 staging 과 promoted 사이에 다름 | 승격 누락/중복 또는 방향 분리 처리 오류 점검 |
| `key_gap_between_staging_and_promoted` | 0행 | FAIL | 특정 자연키가 promoted 에 없거나 extra 로 존재 | 누락 건/초과 건을 route 단위로 역추적 |
| `promoted_continuity_break` | 0행 | FAIL | `link_seq` 연속성이 끊김 | 순번 생성/승격 로직 및 원천 정합성 재검토 |
| `link_id_collision_between_staging_and_promoted` | 0행 | FAIL | 같은 자연키인데 `link_id` 값이 서로 다름 | 승격 업데이트 로직 또는 원천 파싱 값 점검 |

## 승인 기준

### 승격 결과 승인 가능
아래를 모두 만족하면 승인 가능합니다.
1. summary 기대값이 전부 일치한다.
2. `unexpected_route_in_promoted` 가 0행이다.
3. 중복, 누락, 건수 불일치, 연속성 단절, `link_id` 충돌이 모두 0행이다.
4. `unexpected_route_in_staging` 도 0행이거나, 남아 있어도 `route_id='1000'` 1건뿐이다.

### 승격 보류
아래 경우는 보류 후 정리 권장입니다.
- `unexpected_route_in_staging` 에 `1000` 만 남아 있는 경우
- promoted 결과는 정상이나 staging 정리 상태를 더 깔끔히 맞추고 싶은 경우

### 즉시 차단
아래 경우는 승격 결과를 승인하면 안 됩니다.
- `unexpected_route_in_promoted` 결과 존재
- `count_mismatch_by_route_dir` 결과 존재
- `key_gap_between_staging_and_promoted` 결과 존재
- `promoted_continuity_break` 결과 존재
- `link_id_collision_between_staging_and_promoted` 결과 존재

## 권장 판독 순서
1. summary 확인
2. `unexpected_route_in_promoted` 확인
3. `count_mismatch_by_route_dir` 확인
4. `key_gap_between_staging_and_promoted` 확인
5. 중복/연속성/`link_id` 충돌 확인
6. 마지막으로 `unexpected_route_in_staging` 에 legacy sample `1000` 잔존 여부 확인
