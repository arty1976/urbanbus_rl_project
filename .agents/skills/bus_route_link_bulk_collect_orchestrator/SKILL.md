---
name: bus_route_link_bulk_collect_orchestrator
description: Region-agnostic orchestrator skill for bulk collecting and loading bus route links, managing manifests, and identifying missing routes.
---

# Purpose

이 스킬은 단건 적재가 아니라, 특정 지역의 전체 노선 목록을 기반으로 일괄 수집(Bulk Collect) 및 적재를 자동화하고 오케스트레이션(Orchestration)하기 위해 사용됩니다.

## Core Rules

- **Decoding Key**: API `serviceKey`는 반드시 설정 파일에 정의된 일반 인증키(Decoding 키)를 사용해야 합니다.
- **Dynamic Route List**: 수집 대상 `route_id` 목록은 사용자가 넘겨주는 것이 아니라 데이터베이스(예: `public.stg_{RegionName}_routes`)에서 추출해야 합니다.
- **Dynamic URL Construction**: API 호출 URL은 하드코딩하지 않고 `endpointBaseUrl + operationPath`의 조합으로 구성합니다.
- **Strict Metadata Propagation**: `route_id_raw`는 반드시 요청 파라미터로 사용된 `RouteId` 메타데이터 기준이어야 합니다.
- **Snapshot & Manifest Required**: 수신된 원본(raw snapshot) JSON 파일 저장과 상태 로깅(`manifest.csv`)을 필히 수반해야 합니다.
- **State Separation**: manifest 기록 시 수집 상태(`collect_status`)와 적재 상태(`load_status`)를 명백히 분리 기록해야 합니다.
- **Missing Route Detection**: 적재 종료 후, 원천 스키마(source route 수)와 staging에 적재된(loaded route 수) 결과 수를 비교하여 누락된 `route_id` 목록을 자동 산출해야 합니다.

## Input Parameters

| Parameter | Mandatory | Default Value | Description |
| Parameter | Mandatory | Default Value | Description |
| --- | --- | --- | --- |
| `RegionName` | Yes | - | Target region (e.g., 'seoul', 'daegu'). |
| `ConfigPath` | No | `configs/secrets.json` | API keys and endpoint settings. |
| `DbName` | No | `urbanbus` | PostgreSQL database name. |
| `DbUser` | No | `postgres` | Database user. |
| `DbHost` | No | `localhost` | Database host. |
| `DbPort` | No | `5432` | Database port. |
| `PsqlPath` | No | (Depends on System) | PostgreSQL client executable path. |
| `MaxRetries` | No | `2` | Number of times to retry failed requests. |
| `RetryDelaySeconds` | No | `2` | Wait time between retries. |
| `SleepMilliseconds` | No | `500` | Wait time between successful calls to avoid rate limits. |
| `MaxRoutes` | No | `0` (No limit) | Upper limit of routes to process (useful for testing or batching). |

## Execution Steps

1. **Config Load**: `ConfigPath`를 읽어 API 인증키, Endpoint, Parameter Name 등을 초기화합니다.
2. **Route Identification**: DB에 접속해 해당 지역의 중복 없는 노선 ID(`route_id`) 목록을 추출합니다.
3. **Capacity Control**: `MaxRoutes` 값이 설정된 경우 상한을 적용해 목록을 타겟팅합니다.
4. **API Invocation Loop**:
   - 추출된 `route_id`마다 대상 API를 호출합니다.
   - 응답 결과를 `{RegionName}_link_{route_id}_{timestamp}.json` 형태로 저장(Snapshot)합니다.
5. **Logging (Collect)**: 호출이 성공적이면 `collect_status=SUCCESS`, 아니면 `FAILED` 및 `error_message`를 `manifest.csv`에 기록합니다.
6. **Delegated Loading**: (호출 성공 시) 하위 로더 스크립트(`load_route_links_api.ps1`)를 호출하여 해당 Snapshot의 물리적 DB 적재를 수행합니다.
7. **Logging (Load)**: 적재 스크립트 결과를 기반으로 `manifest.csv`의 `load_status` 값을 업데이트합니다.
8. **Reconciliation**: 루프 종료 후 DB 쿼리로 `source_routes` 수와 적재가 확인된 `loaded_routes` 개수 스냅샷을 비교합니다.
9. **Missing Report**: 차이가 발생할 경우(적재되지 않은 노선), 그 누락된 `missing_route_id` 목록을 명시적으로 보고합니다.

## Verification Rules

1. **Source Routes Count**: DB에서 추출한 해당 지역의 전체 노선 수량이 터미널에 명시될 것.
2. **Loaded Routes Count**: 최종 루프 종료 후, 대상 staging 테이블 내 존재하는 고유 `route_id_raw` 건수가 출력될 것.
3. **Missing Route IDs Query**: DB 집합 연산(Except 등)을 통해 실제 누락된 노선이 식별될 것.
4. **Manifest State**: `manifest.csv`의 마지막 기록 상태가 중단 없이 점검 및 유지되었는지 확인.
5. **Sample Row Check**: 노선 ID 단위의 임의 데이터를 샘플 조회해 `row_count` 값이 기대대로인지 교차 점검할 것.

## Failure Rules

- **psql Not Found**: `psql` 실행 파일을 환경 내에서 찾을 수 없으면 바로 실패 리포트.
- **Config Initialization Fail**: API 키 등 설정 정보가 누락되면 수집 진입 전 중단.
- **JSON Path Mismatch**: 응답 데이터로 넘어온 JSON 아이템 패스가 일치하지 않음이 감지되면 즉시 실패 처리.
- **Empty Items (API 200)**: [운영 가정] API 응답이 성공(HTTP 200)하더라도 아이템 배열이 빈 값인 예외 노선은, 적재 없이 통과하되 반드시 별도 보고 대상("NO_ITEMS" 등)으로 기록될 것.
- **Duplicate Snapshot**: 이미 적재된 스냅샷에 대해 중복 호출을 하게 될 경우, 중복 데이터 예방을 위해 적재를 거부하고 정지.

## Success Criteria

- 대상이 된 `route_id` 목록의 끝까지 프로세스가 성공적으로 순회(Loop) 될 것.
- 발생한 모든 API JSON 응답이 디스크에 안전하게 스냅샷으로 저장될 것.
- [소스 노선 수량] vs [적재 완료 수량]의 차분 결과가 명확한 요약 보고서로 생성될 것.
- 단 한 건 이라도 적재가 누락된 `route_id`가 존재할 시, 해당 목록이 명시적으로 출력될 것.

## 호출 프롬프트 예시

- **[대구 Bulk 수집 수행]**:
  "대구시 전체 노선 링크 정보를 수집해줘. `bus_route_link_bulk_collect_orchestrator` 스킬을 이용해서 목록 추출부터 적재 완료, 누락 노선 비교 보고까지 알아서 수행해."

- **[부분 배치(Chunk) 수집 (MaxRoutes 지정)]**:
  "`bus_route_link_bulk_collect_orchestrator` 스킬을 참조하여 중배치 수집을 테스트할 거야. 대전시 상한 값 `MaxRoutes=300`을 적용해서 부분 수집을 돌려보고 누락이 생기면 알려줘."

- **[누락 노선 재확인]**:
  "어제 수집에서 누락 보고된 10건의 노선들에 대해서 오류 원인(Empty items 등)을 파악하고 재수집을 수행해줘. 오케스트레이터 스킬의 루프를 사용하여 재점검해라."

- **[타 지역 수집 확장 적용]**:
  "기울여진 설정파일 정보를 이용해 광주시 전체 노선을 일괄 수집 시작해. 스킬 오케스트레이터의 단계에 따르고, 적재가 끝나면 missing report를 뽑아줘."
