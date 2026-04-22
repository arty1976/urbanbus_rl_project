-- ============================================================
-- create_graph_state_timeslice.sql
-- 목적: 통합 그래프 시계열 상태 테이블 생성
-- 기준: 10분 단위 버킷, node_uid 중심 설계
-- 작성일: 2026-04-10
-- ============================================================

CREATE TABLE IF NOT EXISTS public.graph_state_timeslice (
    state_ts                TIMESTAMPTZ NOT NULL,
    node_uid                TEXT NOT NULL,
    node_type               TEXT NOT NULL,

    -- STOP 관련 수요 지표
    waiting_passenger_cnt   INTEGER DEFAULT 0,
    boardings_recent        INTEGER DEFAULT 0,
    alightings_recent       INTEGER DEFAULT 0,

    -- 수요예측 지표 (AI 모델 연계용)
    predicted_demand_10m    INTEGER DEFAULT 0,
    predicted_demand_30m    INTEGER DEFAULT 0,

    -- 실시간 버스 상태 (BIS/BMS 연계용)
    nearest_bus_eta_sec     INTEGER DEFAULT NULL,
    active_bus_cnt_nearby   INTEGER DEFAULT NULL,

    -- LINK 관련 소통 지표
    link_travel_time_sec    NUMERIC(10,2) DEFAULT NULL,
    link_speed_kmh          NUMERIC(10,2) DEFAULT NULL,
    link_congestion_index   NUMERIC(5,4) DEFAULT NULL,
    link_flow_proxy         NUMERIC(10,2) DEFAULT NULL,

    -- 상태 플래그 및 환경 피처
    incident_flag           BOOLEAN DEFAULT FALSE,
    hour_of_day             INTEGER,
    day_of_week             INTEGER,
    is_peak                 BOOLEAN DEFAULT FALSE,

    created_at              TIMESTAMPTZ DEFAULT NOW(),
    updated_at              TIMESTAMPTZ DEFAULT NOW(),

    -- 제약 조건
    PRIMARY KEY (state_ts, node_uid),
    CONSTRAINT fk_graph_state_timeslice_node
        FOREIGN KEY (node_uid) REFERENCES public.graph_node_master(node_uid),
    CONSTRAINT ck_graph_state_timeslice_hour
        CHECK (hour_of_day >= 0 AND hour_of_day <= 23),
    CONSTRAINT ck_graph_state_timeslice_dow
        CHECK (day_of_week >= 0 AND day_of_week <= 6)
);

-- 성능 최적화를 위한 인덱스
CREATE INDEX IF NOT EXISTS idx_graph_state_ts_type ON public.graph_state_timeslice (state_ts, node_type);
CREATE INDEX IF NOT EXISTS idx_graph_state_node_uid ON public.graph_state_timeslice (node_uid);

COMMENT ON TABLE public.graph_state_timeslice IS '통합 그래프 시계열 상태 테이블. 10분 단위 상태 피처를 저장한다.';
COMMENT ON COLUMN public.graph_state_timeslice.state_ts IS '상태 버킷 시작 시각 (10분 단위)';
COMMENT ON COLUMN public.graph_state_timeslice.boardings_recent IS '직전 10분간 승차 인원 (Hourly Allocation 적용 가능)';
COMMENT ON COLUMN public.graph_state_timeslice.link_speed_kmh IS '링크 평균 주행 속도 (km/h)';
