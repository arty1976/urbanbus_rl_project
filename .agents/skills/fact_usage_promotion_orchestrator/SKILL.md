---
name: fact_usage_promotion_orchestrator
description: Orchestrates and audits the aggregation of synthetic or real card trips into hourly fact usages, ensuring zero reconciliation discrepancies.
---

# Purpose

이 스킬은 API로 유입된 로우 레벨 단위(Record, Trip)의 통행 데이터(Staging)를 분석용 팩트 테이블(`fact_stop_usage_hourly`)로 승격 및 집계시킬 때, 단 한 명의 여객 수요도 유실 혹은 부풀려 지지 않았는지 원장 검증(Reconciliation)을 강제(Orchestrate)하기 위해 사용됩니다.

## Core Rules

- **Source-Agnostic Key Design**: 여러 원천 데이터(합성/실측, 2023년/2025년)가 혼합될 수 있으므로 집계 테이블의 구분자(`source_system`, `data_tier`, `is_observed`)가 로직에 명확하게 분리 파티셔닝되어야 합니다.
- **Absolute Reconciliation**: Staging 테이블의 총 탑승객 수(예: `UTZTN_NOPE` 합) 와 Fact 테이블의 `SUM(boardings)`의 합이 수학적으로 무조건 100% (오차 0) 일치해야 합니다.
- **Negative Value Strict Ban**: 승하차 수단이나 잔여 승객 등 파생 집계 과정에서 어떠한 경우에도 음수(-) 값이 발생해서는 안 됩니다. 파생 로직 계산 시 `GREATEST(sum, 0)` 등으로 방어하거나 에러 처리해야 합니다.

## Workflow

1. **Aggregation Generation**: 통행 개별 건(`stg_daegu_transport_card_usage_synth_trip` 등)의 일자(`service_date`), 버킷 시간(`service_hour`), 정류장(`stop_id`)을 Group By 요소로 묶는 SQL 승격 구문 점검.
2. **Promote Exec**: `fact_stop_usage_hourly` 대상 INSERT 수행 수행 시 복합 Primary Key 에 위배되지 않도록 `ON CONFLICT` 로직 확보.
3. **Cross-Layer Integrity Check**: 집계 후 반드시 양 레이어의 합산을 비교하는 대사 작업(Reconciliation Query) 수행.

## Failure Rules

에이전트는 다음 발생 건수가 **단 1건이라도 존재하면** 승격 결과를 인정하지 않고 원복 조치 혹은 보고해야 합니다.

- `Total_Diff(Source_SUM - Fact_SUM) != 0`
- `Negative_Boardings_Count > 0`
- `Hourly_Bucket_Out_Of_Range` (service_hour가 0~23 구간을 벗어난 경우)

## 호출 프롬프트 예시

- "2025년 합성 통행 데이터를 `fact_stop_usage_hourly`로 올릴 거야. `fact_usage_promotion_orchestrator` 스킬의 룰을 반영해서 집계 SQL의 무결성을 리뷰해 줘."
- "승격 후 `reconcile_usage_sums_2025.sql`을 돌렸는데, Source와 Fact 합계 간 미스매치가 떨어졌어. 오케스트레이터의 가이드라인에 따라 원인(Null 취급이나 시간 변환 유실 등)을 추적해라."
