SET client_encoding = 'UTF8';
-- ============================================================
-- File: 03_validation_queries/rl_stop_transition_features_readiness.sql
-- Purpose:
--   Readiness validation for public.rl_stop_transition_features
--
-- Validation scope:
--   1) row existence / null safety
--   2) non-negative demand counts
--   3) transition timing integrity
--   4) overnight gap / terminal transition consistency
--   5) service bucket structure sanity
--   6) final approval decision
-- ============================================================
with base as (
    select *
    from public.rl_stop_transition_features
),
core as (
    select count(*) as row_count,
        -- null checks
        count(*) filter (
            where state_ts is null
        ) as null_state_ts_cnt,
        count(*) filter (
            where next_state_ts is null
        ) as null_next_state_ts_cnt,
        count(*) filter (
            where node_uid is null
        ) as null_node_uid_cnt,
        count(*) filter (
            where node_type is null
        ) as null_node_type_cnt,
        count(*) filter (
            where boardings_recent is null
        ) as null_boardings_recent_cnt,
        count(*) filter (
            where alightings_recent is null
        ) as null_alightings_recent_cnt,
        count(*) filter (
            where waiting_passenger_cnt is null
        ) as null_waiting_passenger_cnt_cnt,
        count(*) filter (
            where next_boardings_recent is null
        ) as null_next_boardings_recent_cnt,
        count(*) filter (
            where next_alightings_recent is null
        ) as null_next_alightings_recent_cnt,
        count(*) filter (
            where next_waiting_passenger_cnt is null
        ) as null_next_waiting_passenger_cnt_cnt,
        count(*) filter (
            where service_date is null
        ) as null_service_date_cnt,
        count(*) filter (
            where service_bucket_seq is null
        ) as null_service_bucket_seq_cnt,
        count(*) filter (
            where buckets_per_day_actual is null
        ) as null_buckets_per_day_actual_cnt,
        -- non-negative checks
        count(*) filter (
            where boardings_recent < 0
        ) as negative_boardings_recent_cnt,
        count(*) filter (
            where alightings_recent < 0
        ) as negative_alightings_recent_cnt,
        count(*) filter (
            where waiting_passenger_cnt < 0
        ) as negative_waiting_passenger_cnt_cnt,
        count(*) filter (
            where next_boardings_recent < 0
        ) as negative_next_boardings_recent_cnt,
        count(*) filter (
            where next_alightings_recent < 0
        ) as negative_next_alightings_recent_cnt,
        count(*) filter (
            where next_waiting_passenger_cnt < 0
        ) as negative_next_waiting_passenger_cnt_cnt,
        -- raw timing checks
        count(*) filter (
            where delta_t_hr_raw is null
        ) as null_delta_t_hr_raw_cnt,
        count(*) filter (
            where delta_t_hr_raw <= 0
        ) as non_positive_delta_t_hr_raw_cnt,
        min(delta_t_hr_raw) as min_delta_t_hr_raw,
        max(delta_t_hr_raw) as max_delta_t_hr_raw,
        avg(delta_t_hr_raw) as avg_delta_t_hr_raw,
        -- effective timing checks
        count(*) filter (
            where is_overnight_gap = 0
                and delta_t_hr_effective is null
        ) as unexpected_null_delta_t_hr_effective_cnt,
        count(*) filter (
            where is_overnight_gap = 1
                and delta_t_hr_effective is not null
        ) as overnight_gap_nonnull_effective_delta_cnt,
        count(*) filter (
            where delta_t_hr_effective <= 0
        ) as non_positive_delta_t_hr_effective_cnt,
        -- calendar / transition checks
        count(*) filter (
            where next_state_ts <= state_ts
        ) as time_inversion_cnt,
        count(*) filter (
            where is_cross_calendar_day = 1
                and next_service_date <= service_date
        ) as invalid_cross_day_order_cnt,
        -- categorical checks
        count(*) filter (
            where node_type <> 'STOP'
        ) as non_stop_node_type_cnt,
        count(*) filter (
            where node_type_stop <> 1
        ) as invalid_node_type_stop_cnt,
        count(*) filter (
            where hour_of_day < 0
                or hour_of_day > 23
        ) as invalid_hour_of_day_cnt,
        count(*) filter (
            where day_of_week < 0
                or day_of_week > 6
        ) as invalid_day_of_week_cnt,
        count(*) filter (
            where is_peak::text not in ('0', '1', 'true', 'false')
        ) as invalid_is_peak_cnt
    from base
),
bucket_daily as (
    select service_date,
        max(service_bucket_seq) as max_service_bucket_seq,
        min(service_bucket_seq) as min_service_bucket_seq,
        max(buckets_per_day_actual) as buckets_per_day_actual,
        count(distinct state_ts) as distinct_state_ts_cnt,
        case when service_date = max(service_date) over() then 1 else 0 end as is_last_service_date
    from base
    group by service_date
),
bucket_daily_checks as (
    select count(*) as active_days,
        min(max_service_bucket_seq) as min_buckets_per_day,
        max(max_service_bucket_seq) as max_buckets_per_day,
        avg(max_service_bucket_seq::numeric) as avg_buckets_per_day,
        count(*) filter (
            where min_service_bucket_seq <> 1
        ) as invalid_min_bucket_seq_days,
        count(*) filter (
            where max_service_bucket_seq <> buckets_per_day_actual
              and not (is_last_service_date = 1 and max_service_bucket_seq = buckets_per_day_actual - 1)
        ) as inconsistent_bucket_seq_days,
        count(*) filter (
            where distinct_state_ts_cnt <> buckets_per_day_actual
              and not (is_last_service_date = 1 and distinct_state_ts_cnt = buckets_per_day_actual - 1)
        ) as inconsistent_distinct_state_ts_days
    from bucket_daily
),
overnight as (
    select count(*) filter (
            where is_overnight_gap = 1
        ) as overnight_gap_cnt,
        count(*) filter (
            where is_terminal_transition = 1
        ) as terminal_transition_cnt,
        count(*) filter (
            where is_overnight_gap = 1
                and is_terminal_transition <> 1
        ) as overnight_not_terminal_cnt,
        count(*) filter (
            where is_terminal_transition = 1
                and is_overnight_gap <> 1
        ) as terminal_not_overnight_cnt,
        count(*) filter (
            where is_overnight_gap = 1
                and is_cross_calendar_day <> 1
        ) as overnight_without_cross_day_cnt,
        count(*) filter (
            where is_overnight_gap = 1
                and next_service_bucket_seq <> 1
        ) as overnight_next_bucket_not_1_cnt,
        count(*) filter (
            where is_overnight_gap = 1
                and delta_t_hr_raw < 4
        ) as overnight_gap_under_threshold_cnt
    from base
),
snapshot_consistency as (
    select count(*) as total_snapshot_pairs,
        count(*) filter (
            where state_ts is not null
                and next_state_ts is not null
        ) as valid_snapshot_pairs
    from (
            select distinct state_ts,
                next_state_ts
            from base
        ) s
),
final_eval as (
    select c.row_count,
        -- 주요 결과
        bdc.active_days,
        bdc.min_buckets_per_day,
        bdc.max_buckets_per_day,
        round(bdc.avg_buckets_per_day, 4) as avg_buckets_per_day,
        o.overnight_gap_cnt,
        o.terminal_transition_cnt,
        c.min_delta_t_hr_raw,
        c.max_delta_t_hr_raw,
        round(c.avg_delta_t_hr_raw::numeric, 6) as avg_delta_t_hr_raw,
        -- null / 음수 / 구조 오류 총합
        (
            c.null_state_ts_cnt + c.null_next_state_ts_cnt + c.null_node_uid_cnt + c.null_node_type_cnt + c.null_boardings_recent_cnt + c.null_alightings_recent_cnt + c.null_waiting_passenger_cnt_cnt + c.null_next_boardings_recent_cnt + c.null_next_alightings_recent_cnt + c.null_next_waiting_passenger_cnt_cnt + c.null_service_date_cnt + c.null_service_bucket_seq_cnt + c.null_buckets_per_day_actual_cnt
        ) as total_null_issue_cnt,
        (
            c.negative_boardings_recent_cnt + c.negative_alightings_recent_cnt + c.negative_waiting_passenger_cnt_cnt + c.negative_next_boardings_recent_cnt + c.negative_next_alightings_recent_cnt + c.negative_next_waiting_passenger_cnt_cnt
        ) as total_negative_issue_cnt,
        (
            c.non_positive_delta_t_hr_raw_cnt + c.unexpected_null_delta_t_hr_effective_cnt + c.overnight_gap_nonnull_effective_delta_cnt + c.non_positive_delta_t_hr_effective_cnt + c.time_inversion_cnt + c.invalid_cross_day_order_cnt
        ) as total_transition_issue_cnt,
        (
            c.non_stop_node_type_cnt + c.invalid_node_type_stop_cnt + c.invalid_hour_of_day_cnt + c.invalid_day_of_week_cnt + c.invalid_is_peak_cnt + bdc.invalid_min_bucket_seq_days + bdc.inconsistent_bucket_seq_days + bdc.inconsistent_distinct_state_ts_days + o.overnight_not_terminal_cnt + o.terminal_not_overnight_cnt + o.overnight_without_cross_day_cnt + o.overnight_next_bucket_not_1_cnt + o.overnight_gap_under_threshold_cnt
        ) as total_structure_issue_cnt,
        -- 개별 오류 세부
        c.null_state_ts_cnt,
        c.null_next_state_ts_cnt,
        c.null_node_uid_cnt,
        c.null_node_type_cnt,
        c.null_boardings_recent_cnt,
        c.null_alightings_recent_cnt,
        c.null_waiting_passenger_cnt_cnt,
        c.null_next_boardings_recent_cnt,
        c.null_next_alightings_recent_cnt,
        c.null_next_waiting_passenger_cnt_cnt,
        c.null_service_date_cnt,
        c.null_service_bucket_seq_cnt,
        c.null_buckets_per_day_actual_cnt,
        c.negative_boardings_recent_cnt,
        c.negative_alightings_recent_cnt,
        c.negative_waiting_passenger_cnt_cnt,
        c.negative_next_boardings_recent_cnt,
        c.negative_next_alightings_recent_cnt,
        c.negative_next_waiting_passenger_cnt_cnt,
        c.null_delta_t_hr_raw_cnt,
        c.non_positive_delta_t_hr_raw_cnt,
        c.unexpected_null_delta_t_hr_effective_cnt,
        c.overnight_gap_nonnull_effective_delta_cnt,
        c.non_positive_delta_t_hr_effective_cnt,
        c.time_inversion_cnt,
        c.invalid_cross_day_order_cnt,
        c.non_stop_node_type_cnt,
        c.invalid_node_type_stop_cnt,
        c.invalid_hour_of_day_cnt,
        c.invalid_day_of_week_cnt,
        c.invalid_is_peak_cnt,
        bdc.invalid_min_bucket_seq_days,
        bdc.inconsistent_bucket_seq_days,
        bdc.inconsistent_distinct_state_ts_days,
        o.overnight_not_terminal_cnt,
        o.terminal_not_overnight_cnt,
        o.overnight_without_cross_day_cnt,
        o.overnight_next_bucket_not_1_cnt,
        o.overnight_gap_under_threshold_cnt,
        sc.total_snapshot_pairs,
        sc.valid_snapshot_pairs,
        case
            when c.row_count = 0 then 'BLOCKED'
            when (
                c.null_state_ts_cnt + c.null_next_state_ts_cnt + c.null_node_uid_cnt + c.null_node_type_cnt + c.null_boardings_recent_cnt + c.null_alightings_recent_cnt + c.null_waiting_passenger_cnt_cnt + c.null_next_boardings_recent_cnt + c.null_next_alightings_recent_cnt + c.null_next_waiting_passenger_cnt_cnt + c.null_service_date_cnt + c.null_service_bucket_seq_cnt + c.null_buckets_per_day_actual_cnt + c.negative_boardings_recent_cnt + c.negative_alightings_recent_cnt + c.negative_waiting_passenger_cnt_cnt + c.negative_next_boardings_recent_cnt + c.negative_next_alightings_recent_cnt + c.negative_next_waiting_passenger_cnt_cnt + c.null_delta_t_hr_raw_cnt + c.non_positive_delta_t_hr_raw_cnt + c.unexpected_null_delta_t_hr_effective_cnt + c.overnight_gap_nonnull_effective_delta_cnt + c.non_positive_delta_t_hr_effective_cnt + c.time_inversion_cnt + c.invalid_cross_day_order_cnt + c.non_stop_node_type_cnt + c.invalid_node_type_stop_cnt + c.invalid_hour_of_day_cnt + c.invalid_day_of_week_cnt + c.invalid_is_peak_cnt + bdc.invalid_min_bucket_seq_days + bdc.inconsistent_bucket_seq_days + bdc.inconsistent_distinct_state_ts_days + o.overnight_not_terminal_cnt + o.terminal_not_overnight_cnt + o.overnight_without_cross_day_cnt + o.overnight_next_bucket_not_1_cnt + o.overnight_gap_under_threshold_cnt
            ) = 0 then 'APPROVED'
            else 'REVIEW_REQUIRED'
        end as final_decision,
        case
            when c.row_count = 0 then 'NO'
            when (
                c.null_state_ts_cnt + c.null_next_state_ts_cnt + c.null_node_uid_cnt + c.null_node_type_cnt + c.null_boardings_recent_cnt + c.null_alightings_recent_cnt + c.null_waiting_passenger_cnt_cnt + c.null_next_boardings_recent_cnt + c.null_next_alightings_recent_cnt + c.null_next_waiting_passenger_cnt_cnt + c.null_service_date_cnt + c.null_service_bucket_seq_cnt + c.null_buckets_per_day_actual_cnt + c.negative_boardings_recent_cnt + c.negative_alightings_recent_cnt + c.negative_waiting_passenger_cnt_cnt + c.negative_next_boardings_recent_cnt + c.negative_next_alightings_recent_cnt + c.negative_next_waiting_passenger_cnt_cnt + c.null_delta_t_hr_raw_cnt + c.non_positive_delta_t_hr_raw_cnt + c.unexpected_null_delta_t_hr_effective_cnt + c.overnight_gap_nonnull_effective_delta_cnt + c.non_positive_delta_t_hr_effective_cnt + c.time_inversion_cnt + c.invalid_cross_day_order_cnt + c.non_stop_node_type_cnt + c.invalid_node_type_stop_cnt + c.invalid_hour_of_day_cnt + c.invalid_day_of_week_cnt + c.invalid_is_peak_cnt + bdc.invalid_min_bucket_seq_days + bdc.inconsistent_bucket_seq_days + bdc.inconsistent_distinct_state_ts_days + o.overnight_not_terminal_cnt + o.terminal_not_overnight_cnt + o.overnight_without_cross_day_cnt + o.overnight_next_bucket_not_1_cnt + o.overnight_gap_under_threshold_cnt
            ) = 0 then 'YES'
            else 'NO'
        end as rl_stop_transition_features_ready
    from core c
        cross join bucket_daily_checks bdc
        cross join overnight o
        cross join snapshot_consistency sc
)
select *
from final_eval;