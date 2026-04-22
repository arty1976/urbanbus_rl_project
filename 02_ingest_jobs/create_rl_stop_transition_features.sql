SET client_encoding = 'UTF8';
-- ============================================================
-- File: 02_ingest_jobs/create_rl_stop_transition_features.sql
-- Purpose:
--   Build model-ready stop transition features from
--   public.rl_state_training_base
--
-- Design decisions:
--   1) Keep raw elapsed time as delta_t_hr_raw
--   2) Explicitly flag overnight service gaps
--   3) Mark overnight-crossing rows as terminal transitions
--   4) Provide delta_t_hr_effective for RL-friendly usage
--      (NULL on overnight gaps)
-- ============================================================
create or replace view public.rl_stop_transition_features as with base as (
        select state_ts,
            next_state_ts,
            node_uid,
            node_type,
            boardings_recent,
            alightings_recent,
            waiting_passenger_cnt,
            hour_of_day,
            day_of_week,
            is_peak,
            next_boardings_recent,
            next_alightings_recent,
            next_waiting_passenger_cnt
        from public.rl_state_training_base
        where node_type = 'STOP'
    ),
    -- ------------------------------------------------------------
    -- Build a timestamp index at the service-bucket level.
    -- We use distinct timestamps so that all nodes sharing the
    -- same state_ts receive the same service_bucket_seq.
    -- ------------------------------------------------------------
    timestamp_index as (
        select ts::date as service_date,
            ts as bucket_ts,
            dense_rank() over (
                partition by ts::date
                order by ts
            ) as service_bucket_seq,
            count(*) over (partition by ts::date) as buckets_per_day_actual
        from (
                select distinct state_ts as ts
                from base
                union
                select distinct next_state_ts as ts
                from base
            ) u
    ),
    enriched as (
        select b.state_ts,
            b.next_state_ts,
            b.node_uid,
            b.node_type,
            -- raw counts
            b.boardings_recent,
            b.alightings_recent,
            b.waiting_passenger_cnt,
            b.next_boardings_recent,
            b.next_alightings_recent,
            b.next_waiting_passenger_cnt,
            -- temporal raw columns
            b.hour_of_day,
            b.day_of_week,
            b.is_peak,
            -- service-day metadata
            cur_idx.service_date as service_date,
            nxt_idx.service_date as next_service_date,
            cur_idx.service_bucket_seq,
            nxt_idx.service_bucket_seq as next_service_bucket_seq,
            cur_idx.buckets_per_day_actual,
            -- actual elapsed time
            extract(
                epoch
                from (b.next_state_ts - b.state_ts)
            ) / 60.0 as delta_t_min_raw,
            extract(
                epoch
                from (b.next_state_ts - b.state_ts)
            ) / 3600.0 as delta_t_hr_raw,
            -- day-crossing metadata
            case
                when cur_idx.service_date <> nxt_idx.service_date then 1
                else 0
            end as is_cross_calendar_day
        from base b
            left join timestamp_index cur_idx on b.state_ts = cur_idx.bucket_ts
            left join timestamp_index nxt_idx on b.next_state_ts = nxt_idx.bucket_ts
    ),
    finalized as (
        select e.state_ts,
            e.next_state_ts,
            e.node_uid,
            e.node_type,
            -- service timeline identifiers
            e.service_date,
            e.next_service_date,
            e.service_bucket_seq,
            e.next_service_bucket_seq,
            e.buckets_per_day_actual,
            -- current raw state
            e.boardings_recent,
            e.alightings_recent,
            e.waiting_passenger_cnt,
            e.hour_of_day,
            e.day_of_week,
            e.is_peak,
            -- next raw targets
            e.next_boardings_recent,
            e.next_alightings_recent,
            e.next_waiting_passenger_cnt,
            -- transformed state features
            ln(1 + e.boardings_recent) as boardings_recent_log,
            ln(1 + e.alightings_recent) as alightings_recent_log,
            ln(1 + e.waiting_passenger_cnt) as waiting_passenger_cnt_log,
            sin(2 * pi() * e.hour_of_day / 24.0) as hour_sin,
            cos(2 * pi() * e.hour_of_day / 24.0) as hour_cos,
            sin(2 * pi() * e.day_of_week / 7.0) as dow_sin,
            cos(2 * pi() * e.day_of_week / 7.0) as dow_cos,
            case
                when e.node_type = 'STOP' then 1
                else 0
            end as node_type_stop,
            -- raw elapsed time always preserved
            e.delta_t_min_raw,
            e.delta_t_hr_raw,
            e.is_cross_calendar_day,
            -- overnight gap:
            -- current day last bucket -> next day first bucket
            -- plus a sufficiently large gap
            case
                when e.is_cross_calendar_day = 1
                and e.service_bucket_seq = e.buckets_per_day_actual
                and e.next_service_bucket_seq = 1
                and e.delta_t_hr_raw >= 4 then 1
                else 0
            end as is_overnight_gap,
            -- terminal transition for RL episode boundary
            case
                when e.is_cross_calendar_day = 1
                and e.service_bucket_seq = e.buckets_per_day_actual
                and e.next_service_bucket_seq = 1
                and e.delta_t_hr_raw >= 4 then 1
                else 0
            end as is_terminal_transition
        from enriched e
    )
select state_ts,
    next_state_ts,
    node_uid,
    node_type,
    -- service metadata
    service_date,
    next_service_date,
    service_bucket_seq,
    next_service_bucket_seq,
    buckets_per_day_actual,
    -- raw state
    boardings_recent,
    alightings_recent,
    waiting_passenger_cnt,
    hour_of_day,
    day_of_week,
    is_peak,
    -- transformed features
    boardings_recent_log,
    alightings_recent_log,
    waiting_passenger_cnt_log,
    hour_sin,
    hour_cos,
    dow_sin,
    dow_cos,
    node_type_stop,
    -- transition timing
    delta_t_min_raw,
    delta_t_hr_raw,
    case
        when is_overnight_gap = 1 then null
        else delta_t_hr_raw
    end as delta_t_hr_effective,
    is_cross_calendar_day,
    is_overnight_gap,
    is_terminal_transition,
    -- next-state targets
    next_boardings_recent,
    next_alightings_recent,
    next_waiting_passenger_cnt
from finalized;