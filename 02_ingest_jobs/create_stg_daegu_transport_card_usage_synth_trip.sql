-- ============================================================
-- create_stg_daegu_transport_card_usage_synth_trip.sql
-- 목적: 합성 교통카드 개별 통행 데이터를 저장하는 Staging 테이블 생성
-- 특이사항: record_hash를 통한 중복 방지 및 원본 JSON 보존
-- ============================================================

CREATE TABLE IF NOT EXISTS public.stg_daegu_transport_card_usage_synth_trip (
    -- 1. Metadata Fields
    record_hash      VARCHAR(64) PRIMARY KEY, -- 데이터 무결성 및 중복 제거용 해시
    batch_id         VARCHAR(50),             -- 적재 배치 ID (YYYYMMDD_HHMMSS)
    source_ref       VARCHAR(255),            -- 원천 정보 참조
    raw_payload_json JSONB,                   -- API 응답 원본 (디버깅용)
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 2. API Fields (Trip-level)
    opr_ymd          VARCHAR(8),              -- 운행 일자
    ride_dt          VARCHAR(14),             -- 승차 일시
    goff_dt          VARCHAR(14),             -- 하차 일시
    rte_id           VARCHAR(50),             -- 노선 식별자
    ride_sttn_id     VARCHAR(50),             -- 승차 정류장 ID
    goff_sttn_id     VARCHAR(50),             -- 하차 정류장 ID
    utztn_nope       INTEGER,                 -- 이용 인원 (tz_nope 아님)
    msv_intrpl_yn    CHAR(1),                 -- 결측 보간 여부 (Y/N)
    vr_card_no       VARCHAR(100),            -- 가상 카드 번호
    card_se_cd       VARCHAR(10),             -- 카드 구분 코드
    trnf_cnt         INTEGER,                 -- 환승 횟수
    users_type_cd    VARCHAR(10),             -- 사용자 유형 코드
    utztn_dstnc      NUMERIC(15, 2),          -- 이용 거리
    brdg_hr          INTEGER,                 -- 체류 시간
    ride_ctpv_cd     VARCHAR(10),             -- 승차 시도 코드
    goff_ctpv_cd     VARCHAR(10),             -- 하차 시도 코드
    clcln_bzmn_id    VARCHAR(50),             -- 정산 사업자 ID
    clcln_bzmn_trfc_mns_cd VARCHAR(10)        -- 정산 사업자 교통 수단 코드
);

-- 인덱스 추가 (조회 및 중복 체크 성능 향상)
CREATE INDEX IF NOT EXISTS idx_stg_daegu_synth_trip_opr_ymd ON public.stg_daegu_transport_card_usage_synth_trip (opr_ymd);
CREATE INDEX IF NOT EXISTS idx_stg_daegu_synth_trip_batch_id ON public.stg_daegu_transport_card_usage_synth_trip (batch_id);

COMMENT ON TABLE public.stg_daegu_transport_card_usage_synth_trip IS '대구 교통카드 이용 합성 데이터 (Trip-level) 스테이징 테이블';
