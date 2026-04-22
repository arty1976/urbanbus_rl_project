---
name: bus_route_link_ingest_guard
description: Region-agnostic ingestion skill for bus route links with strict metadata enforcement, duplicate protection, and automated verification.
---

# Purpose

Standardize the ingestion of bus route link snapshots (JSON or CSV) into PostgreSQL staging tables. This skill ensures data integrity by decoupling internal route identifiers from unstable provider fields and prevents environment-specific DDL failures on Windows.

# Core Rules

- **Metadata Authority**: `route_id_raw` must derived EXCLUSIVELY from the request parameter `RouteId`.
- **Source Ignorance**: Ignore any `routeId` or equivalent field within the source JSON/CSV body to prevent vendor-side mislabeling.
- **No Fallbacks**: Never use `UNKNOWN` as a fallback for route identification. If `RouteId` is missing, the process must stop.
- **Idempotency**: Always verify table existence before DDL. Skip DDL if the table exists to avoid encoding/index conflicts.
- **Generic Naming**: All assets must use the `{RegionName}` variable. Never hardcode specific city names.

# Input Parameters

| Parameter | Mandatory | Default Value | Description |
|---|---|---|---|
| `RegionName` | Yes | - | Target region (e.g., 'seoul', 'busan'). |
| `RouteId` | Yes | - | The stable route identifier to inject for all rows. |
| `InputFile` | Yes | - | Local path to the source JSON or CSV file. |
| `ItemsPath` | No | `response.body.items.item` | Dot-notation path to the items array (JSON only). |
| `TargetTable` | No | `public.stg_{RegionName}_route_links_api` | Destination staging table. |
| `DbName` | No | `urbanbus` | PostgreSQL database name. |
| `DbUser` | No | `postgres` | Database user. |
| `DbHost` | No | `localhost` | Database host. |
| `DbPort` | No | `5432` | Database port. |

# Selection & Transformation

### JSON Mode
- Validates existence of array at `ItemsPath`.
- Maps fields to `_raw` columns: `linkId`, `stNode`, `edNode`, `gisDist`, `moveDir`, `linkSeq`.

### CSV Mode
- **Required Headers**: `linkId`, `stNode`, `edNode`, `gisDist`, `moveDir`, `linkSeq`.
- **Validation**: If any header is missing, report the specific missing column names and stop.

# Execution Steps

1. **Environmental Check**: Verify `InputFile` exists and `psql` is accessible.
2. **Schema Protection**: Query `pg_tables` for `TargetTable`.
3. **Conditional DDL**: Execute the regional DDL script ONLY if the table is missing. If exists, log "Staging table already exists. Skipping DDL."
4. **Duplicate Protection**: Query row count for `snapshot_file = '{FileName}'`. If > 0, stop with "SKIP" status.
5. **Atomic Load**: 
   - Transform input into a temporary CSV with exactly 8 columns (including injected `RouteId`).
   - Use `psql \copy` for atomic ingestion.
6. **Active Verification**: Run the 5 standard validation queries.

# Verification Queries

Execute and report the count for each of the following:

1. **Metadata Consistency**:
   `SELECT count(*) FROM {TargetTable} WHERE route_id_raw = '{RouteId}';`
2. **Snapshot Integrity**:
   `SELECT count(*) FROM {TargetTable} WHERE snapshot_file = '{FileName}';`
3. **Invalid Metadata Check** (Should be 0):
   `SELECT count(*) FROM {TargetTable} WHERE route_id_raw = 'UNKNOWN';`
4. **Data Completeness Check** (Should be 0):
   `SELECT count(*) FROM {TargetTable} WHERE link_id_raw IS NULL OR link_id_raw = '';`
5. **Sequence Sample**:
   `SELECT route_id_raw, link_id_raw, link_seq_raw FROM {TargetTable} WHERE snapshot_file = '{FileName}' ORDER BY link_seq_raw::INT LIMIT 10;`

# Failure Rules

- **JSON Path Mismatch**: Report the actual JSON structure found at the root and stop.
- **CSV Header Mismatch**: Report the list of missing headers and stop.
- **psql Error**: Report the command executed, the exit code, and the full stderr output, then stop.
- **Connection Error**: If the database is unreachable, report connection parameters used and stop.

# Success Criteria

- The `TargetTable` contains all rows from the file.
- `route_id_raw` is populated correctly for every row using the `RouteId` parameter.
- No `UNKNOWN` or null values exist in the `link_id_raw` column.
- The `snapshot_file` column correctly reflects the source filename.

# 호출 프롬프트 예시

- **서울시 데이터 적재**:
  "서울 버스 노선 '1000'번 링크 데이터를 `data/raw/seoul/api_snapshots/response_seoul_1000.csv` 에서 적재해줘. `bus_route_link_ingest_guard` 스킬을 사용하고 스키마는 `public`을 사용해."

- **부산시 JSON 데이터 적재 (커스텀 경로)**:
  "부산 버스 '210'번 데이터를 적재해. 파일은 `route_210.json` 인데 JSON 아이템 경로가 `result.data`야. `bus_route_link_ingest_guard` 스킬의 `ItemsPath` 파라미터를 조절해서 실행해."

- **인천시 복구 적재**:
  "`bus_route_link_ingest_guard` 스킬로 인천 노선 '77' 데이터를 `incheon_77.csv` 에서 적재해. 이미 테이블이 있을 테니 DDL은 건너뛰고 중복 체크 후 적재해줘."

- **대전시 적재 및 정밀 검증**:
  "대전 '305'번 데이터를 `response_305.csv` 에서 적재하고, `bus_route_link_ingest_guard` 스킬의 5가지 검증 쿼리 결과를 테이블 형태로 보고해줘."
