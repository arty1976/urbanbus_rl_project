-- ============================================================
-- create_rl_state_training_base.sql
-- 목적: GAT/RL 모형 학습을 위해 시각 t와 다음 관측 시점(t+1) 상태를 쌍(Pair)으로 묶는 Training Base 구축
-- 기준: graph_state_timeslice 대상 LEAD() 윈도우 함수 사용
-- ============================================================
-- [중요 해석 가이드]
-- 1. next_state_ts 는 strict one-hour future(정확히 1시간 뒤)가 아니라 'next observed timestamp(다음 관측 시점)'을 의미합니다.
-- 2. 하루 19개 운영 버킷(05~23시) 기준으로 심야/야간 공백이 존재할 수 있으며(예: 23:00 -> 다음날 05:00), 
--    현재 Phase 5 단계에서는 이 연속성 단절을 그대로 인정하고 학습 Pair로 사용합니다.
-- ============================================================

DROP TABLE IF EXISTS public.rl_state_training_base;

CREATE TABLE public.rl_state_training_base AS
WITH state_transitions AS (
    SELECT
        state_ts,
        LEAD(state_ts) OVER w AS next_state_ts,
        node_uid,
        node_type,
        
        -- 현재 시점 (t) 입력(Input) 변수
        boardings_recent,
        alightings_recent,
        waiting_passenger_cnt,
        hour_of_day,
        day_of_week,
        is_peak,
        
        -- 다음 시점 (t+1) 타깃(Target) 변수
        LEAD(boardings_recent) OVER w AS next_boardings_recent,
        LEAD(alightings_recent) OVER w AS next_alightings_recent,
        LEAD(waiting_passenger_cnt) OVER w AS next_waiting_passenger_cnt
        
    FROM public.graph_state_timeslice
    WHERE node_type = 'STOP'
    WINDOW w AS (PARTITION BY node_uid ORDER BY state_ts)
)
-- 마지막 타임스텝(t+1 타깃이 존재하지 않는 시점)은 학습 Pair 구성 불가로 제외
SELECT *
FROM state_transitions
WHERE next_state_ts IS NOT NULL;

-- 조회 성능 향상을 위한 최적화 인덱스 생성
ALTER TABLE public.rl_state_training_base ADD PRIMARY KEY (node_uid, state_ts);
CREATE INDEX idx_rl_train_ts ON public.rl_state_training_base (state_ts);
