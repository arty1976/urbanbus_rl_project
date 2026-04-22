---
name: route_link_promotion_guard
description: Safeguards the promotion of bus routes from staging to analytical sequences, enforcing composite key integrity and preventing data contamination.
---

# Purpose

이 스킬은 API로 임시 적재된 버스 노선 링크(`stg_{RegionName}_route_links_api`)를 정규 분석용 테이블(`route_link_sequence`)로 승격(Promotion) 시킬 때, 데이터 이빠짐(Gap), 키 충돌(Collision), 그리고 이전 레거시 오염 데이터의 유입을 철저하게 방어(Guard)하기 위해 사용됩니다.

## Core Rules

- **Composite Key Integrity**: 승격 대상 데이터는 반드시 `(route_id, move_dir_code, link_seq)`의 복합키(Composite Key) 조합에서 절대 중복이 발생해서는 안 됩니다.
- **Legacy Contamination Prevention**: 더미 데이터 또는 오염된 샘플 데이터(예: `route_id='1000'`)가 승격 대상에 혼입되지 않도록 Pre-check해야 합니다.
- **Continuous Sequence**: 노선의 `link_seq`는 끊기거나 비어있는 구간(Gap) 없이 연속되어야 합니다.
- **Upsert Execution**: 적재 방식은 `ON CONFLICT (route_id, move_dir_code, link_seq) DO UPDATE` 패턴의 Upsert를 지향합니다.
- **Result Template Usage**: 승격 검증 결과는 반드시 `route_link_promotion_result_report_template.md` 양식에 맞춰 PASS/WARN/FAIL을 판독하여 문서화해야 합니다.

## Pre-validation (승격 전 검사)

승격 명령을 수행하기 전 다음을 점검합니다:

1. 대상 `route_id`에 Null 값이나 타입 오류(Integer 기대 ↔ Text 입력)가 없는지 확인 (`Type Casting` 보장).
2. Staging 테이블에서 기대하지 않은 노선(Legacy sample 등)이 발견될 경우 WARN 로그를 발행하고 승격 대상에서 제외할 것.
3. 대상 집합 내의 Natural Key(원천 데이터의 링크 순번 등)에 중복이 발생하면 에러(FAIL) 처리 후 중단할 것.

## Post-validation (승격 후 검사)

승격이 완료된 후 `route_link_promotion_readiness.sql` (혹은 이와 유사한 종합 검증 쿼리)를 돌려 다음 지표들을 산출해야 합니다.

1. `promoted_rows_count`: 정상적으로 삽입/업데이트된 행의 수.
2. `pk_violations`: 대상 복합키 기준 중복 건수 (반드시 0이어야 함).
3. `sequence_gaps`: 순번(sequence)이 끊어진 건수 (반드시 0이어야 함).
4. `count_mismatches`: Staging 건수와 승격된 건수가 다름을 의미함 (반드시 0이어야 함).

## 판독 기준 (Readiness Guide)

검증 결과를 바탕으로 다음 기준에 따라 **최종 판정(APPROVED / HOLD / BLOCKED)**을 내려야 합니다.

- **PASS (APPROVED)**: 모든 pk_violations, sequence_gaps, mismatch가 `0` 건일 때.
- **WARN (HOLD)**: Staging에 예기치 않은 데이터(legacy)가 발견되었으나 승격본(Promoted)에는 전이되지 않았거나 무시 가능할 때 (운영자 리뷰 대기).
- **FAIL (BLOCKED)**: 단 1건이라도 PK 중복, 연속성 단절, 링크 아이디 충돌이 발생했을 때. 이 경우 즉시 파이프라인을 멈추고 원인을 분석할 것.

## 호출 프롬프트 예시

- "대구 노선 30건을 승격스크립트를 태울 건데, 적재 전후로 `route_link_promotion_guard` 규칙을 적용해서 점검해줘."
- "어제 승격시킨 부산 데이터에 대해 `route_link_promotion_readiness.sql`을 돌렸어. 출력된 결과를 보고 Guard 스킬 문서의 판독 기준에 따라 리포트 템플릿을 채워줘."
