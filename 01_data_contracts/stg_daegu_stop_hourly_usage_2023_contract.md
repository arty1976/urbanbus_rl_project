# Data Contract: stg_daegu_stop_hourly_usage_2023

## Status
- **Layer**: Staging (Raw to DB)
- **Source**: `data/raw/daegu/traffic_cards/stop_hourly_usage_2023/.../정류소별시간대별승하차인원조회_20230101_20231231.csv`
- **Format**: CSV, UTF-8, Comma-delimited

## Schema Definition
| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| 일자 | DATE | No | Service date (YYYY-MM-DD) |
| 정류소명 | VARCHAR | No | Stop name (Informational) |
| 정류소ID | VARCHAR | No | Official Stop ID (Primary Join Key) |
| 모바일ID | VARCHAR | Yes | Mobile/BMS Stop ID (Fallback Join Key) |
| 행정구역 | VARCHAR | Yes | Administrative area |
| 구분 | VARCHAR | No | '승차' (Boarding) or '하차' (Alighting) |
| 05시 ~ 23시 | INTEGER | Yes | Hourly counts |
| 합계 | INTEGER | Yes | Row total count |

## Integrity Rules
1. **Unpivoting**: Columns `05시` to `23시` must be unpivoted into `service_hour` (int) and `amount` (int).
2. **Key mapping**: `정류소ID` is the preferred key. `모바일ID` is fallback.
3. **Values**: Negative values must be treated as 0 or NULL.

## Promotion Readiness
- Total rows expected: ~2.1M (Based on file inspection)
- Partitioning suggested: By `일자` if table grows larger.
