# Data Contract: stg_daegu_stop_usage_2025_monthly

## Status
- **Layer**: Staging (Profile/Validation)
- **Source**: `data/raw/daegu/traffic_cards/stop_hourly_usage_2023/대구광역시_정류소별 시간대별 승하차인원(2025년)/*.csv`
- **Format**: CSV, Likely CP949 or UTF-8

## Schema Definition
| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| 년월 | VARCHAR | No | Month (YYYYMM) |
| 정류소명 | VARCHAR | No | Stop name |
| 정류소ID | VARCHAR | No | Official Stop ID |
| 구분 | VARCHAR | No | '승차' (Boarding) or '하차' (Alighting) |
| 05시 ~ 23시 | INTEGER | Yes | Hourly counts |

## Integrity Rules
1. **Unpivoting**: Columns `05시` to `23시` must be unpivoted into `service_hour` (int) and `amount` (int).
2. **Aggregation**: This data is monthly, so `service_date` in `fact` will be set to the 1st of the month or a special flag record.
3. **Purpose**: Used for high-level pattern profiling and 2023-2025 trend analysis.
