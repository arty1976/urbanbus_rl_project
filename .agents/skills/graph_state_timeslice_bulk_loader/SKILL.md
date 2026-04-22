---
name: graph_state_timeslice_bulk_loader
description: Prescribes rigid architectural transactions and constraints handling for massive (+20M rows) ingestion processes like Phase 4 Graph State tables to prevent IO thrashing.
---

# Purpose

이 스킬은 연간 통행량 등 초거대 단위(Phase 4: 약 2천만~수천만 로우)의 데이터를 그래프 상태 공간(`graph_state_timeslice`) 테이블로 부어 넣을 때 발생할 수 있는 DB Lock 정지 및 `DataFileExtend`(디스크 병목) 현상을 원천 방지하기 위한 구조적 아키텍처 규칙을 제시합니다.

## Core Rules

- **Strict Transaction Enclosure**: 대량 적재 시 DDL/DML 컨트롤은 반드시 외부 SQL 수준의 단일 트랜잭션 블록(`BEGIN;` ~ `COMMIT;`) 하에서 수행되어 예기치 못한 중간 실패 시 무결성을 확보해야 합니다. 내부 `DO $$ BEGIN` 컨텍스트는 가급적 사용하지 않습니다.
- **Index Suspension Technique**: 대량 데이터를 적재하는 중 `B-Tree` 업데이트 오버헤드를 막기 위해, 초기화 시점과 삽입 전 시점 사이 **타겟 테이블 `TRUNCATE` -> PK(Primary Key) 제약 조건 일시 삭제 -> 보조 인덱스(Index) 일시 삭제**가 순차적으로 이루어져야 합니다.
- **Bulk Push First**: 아무 제약이 없는 빈 깡통 상태의 테이블로 `SELECT ... INSERT` 쿼리를 밀어 넣어 속도를 극대화해야 합니다.
- **Post-Index Rebuild**: `INSERT` 완료된 즉시(동일 트랜잭션 내에서) `ALTER TABLE ADD PRIMARY KEY` 및 `CREATE INDEX` 로 인덱스를 일괄 재생성해야 합니다.
- **Parent Protect**: 종속 테이블(`graph_state_timeslice`)을 `TRUNCATE` 하되, 외래 키로 참조하는 부모 룩업 테이블(`graph_node_master`)의 강제 `TRUNCATE` 나 파괴 행위는 절대 엄금합니다. 부모 테이블 추가가 필요하다면 `LEFT JOIN`에 기반해 `IS NULL` 인 누락건(gap)만 `INSERT` 합니다.

## Transformation Constraints

스니펫이나 자동화 SQL 생성 시 에이전트가 흔히 범하기 쉬운 스키마 오류를 다음과 같이 방지합니다:
- `is_active` 나 `MAX(service_date)` 필터, `boardings / 6` 등은 프로젝트의 요구 명세에 명확히 표기되지 않은 한 본 스킬 내에서 원천 금지입니다.
- 복합 시간 키 생성이 필요하다면 임시 테이블의 교차 조인(cross-join) 루프를 돌지 않고, 정수와 일자가 결합하는 포맷 `(service_date::timestamp + service_hour * interval '1 hour')`을 원칙으로 삼습니다.

## Failure Rules

- 스크립트 리뷰 시 5천만 레코드 삽입 대상이면서 외래키/PK의 DROP/ADD 문이 트랜잭션으로 묶여있지 않다면 즉시 수정을 지시해야 합니다.
- TRUNCATE 대상 테이블이 부모 노드 테이블을 포함하고 있으면 거절(Reject)합니다.

## 호출 프롬프트 예시

- "2023년대구 이용량 데이터 2100만건을 상태 테이블로 올릴 거다. 코드가 멈추지 않게 `graph_state_timeslice_bulk_loader` 스킬 규칙(Option B)을 적용해 SQL 포맷을 리팩터링 해 줘."
- "내가 제시한 Phase 4 재적재 코드(`load_graph.sql`)에서 Transaction, TRUNCATE, Index 순서가 벌크 로더 스킬 가이드에 정확히 들어맞는지 안전 점검을 수행해 줘."
