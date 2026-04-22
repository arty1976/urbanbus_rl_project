-- ============================================================
-- dim_stop 생성 및 데이터 적재 SQL
-- 대상: 오직 dim_stop 테이블만 재생성 및 적재함
-- 원천 테이블: public.bs_20250903 (DB 적재 검증 완료: 5,705건)
-- 생성 좌표계: geom_5187 (원본 EPSG:5187, 한국 TM 중부원점)
--             geom_4326 (WGS84, ST_Transform 변환)
-- longitude / latitude: ST_Transform 기반 계산값
--
-- [전제 조건]
-- 1. PostGIS extension이 urbanbus DB에 활성화되어 있어야 함.
-- 2. public.bs_20250903 테이블이 DB에 적재되어 있어야 함. (5,705건 확인)
-- 3. (Assumption) bs_20250903.geometry 컬럼의 점 데이터가 TM 중부원점(5187) 기반임을 전제함.
--    안전한 변환을 위해 명시적으로 ST_SetSRID(geometry, 5187)을 적용함.
-- ============================================================

-- [사전 점검] 원천 테이블의 geometry SRID 확인
-- SELECT Find_SRID('public', 'bs_20250903', 'geometry');
-- SELECT ST_SRID(geometry), COUNT(*) FROM public.bs_20250903 GROUP BY ST_SRID(geometry);

-- ============================================================
-- STEP 1. 기존 테이블 삭제 (재실행 안전성 보장)
-- 의도: 본 파일은 dim_stop을 재생성하는 스크립트이므로, 기존 테이블 및 의존성 뷰 등을 정리(CASCADE)
-- ============================================================
DROP TABLE IF EXISTS public.dim_stop CASCADE;

-- ============================================================
-- STEP 2. dim_stop 테이블 생성
-- ============================================================
CREATE TABLE public.dim_stop (
    stop_id    VARCHAR(50)   PRIMARY KEY,
    stop_name  VARCHAR(100)  NOT NULL,     -- 참고: stop_name은 조인 키가 아니라 단순 표시용(전시용) 속성임
    longitude  NUMERIC(12, 8),             -- WGS84 경도 (ST_Transform 계산값)
    latitude   NUMERIC(12, 8),             -- WGS84 위도 (ST_Transform 계산값)
    geom_5187  GEOMETRY(Point, 5187),      -- 명시적 SRID 5187 지정 원본 좌표
    geom_4326  GEOMETRY(Point, 4326),      -- WGS84 변환 좌표
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- STEP 3. 데이터 적재
-- ============================================================
INSERT INTO public.dim_stop (
    stop_id,
    stop_name,
    longitude,
    latitude,
    geom_5187,
    geom_4326
)
SELECT
    bs_id::VARCHAR(50)                                                   AS stop_id,
    bs_nm::VARCHAR(100)                                                  AS stop_name,
    ST_X(ST_Transform(ST_SetSRID(geometry, 5187), 4326))::NUMERIC(12, 8) AS longitude,
    ST_Y(ST_Transform(ST_SetSRID(geometry, 5187), 4326))::NUMERIC(12, 8) AS latitude,
    ST_SetSRID(geometry, 5187)                                           AS geom_5187,
    ST_Transform(ST_SetSRID(geometry, 5187), 4326)                       AS geom_4326
FROM public.bs_20250903
WHERE bs_id IS NOT NULL;  -- Fact: bs_id null 0건이므로 전량 적재 예상

-- ============================================================
-- STEP 4. 공간 인덱스 생성
-- ============================================================
CREATE INDEX idx_dim_stop_geom_5187 ON public.dim_stop USING GIST (geom_5187);
CREATE INDEX idx_dim_stop_geom_4326 ON public.dim_stop USING GIST (geom_4326);

-- ============================================================
-- STEP 5. 검증 SQL (CREATE 및 INSERT 완료 후 별도 실행)
-- ============================================================

/*
-- [검증 1] 행 수 확인 (기대값: 5705)
SELECT COUNT(*) AS total_rows FROM public.dim_stop;

-- [검증 2] PK 무결성 (기대값: 모두 0)
SELECT 'stop_id_null'      AS check_type, COUNT(*) AS cnt FROM public.dim_stop WHERE stop_id IS NULL
UNION ALL
SELECT 'stop_id_duplicate' AS check_type, COUNT(*) AS cnt
FROM (SELECT stop_id FROM public.dim_stop GROUP BY stop_id HAVING COUNT(*) > 1) d;

-- [검증 3] 이름 중복 경고 (경고 지표, 250건 예상 - 조인 금지 근거)
SELECT 'stop_name_duplicate_warning' AS check_type, COUNT(*) AS cnt
FROM (SELECT stop_name FROM public.dim_stop GROUP BY stop_name HAVING COUNT(*) > 1) d;

-- [검증 4] 경위도 유효 범위 확인 (기대값: 모두 0 / 대구 기준 범위 적용)
-- (Assumption: 대구광역시 경도 128.3~128.9, 위도 35.6~36.2)
SELECT 'longitude_out_of_range' AS check_type, COUNT(*) AS cnt
FROM public.dim_stop WHERE longitude NOT BETWEEN 128.0 AND 129.5
UNION ALL
SELECT 'latitude_out_of_range'  AS check_type, COUNT(*) AS cnt
FROM public.dim_stop WHERE latitude NOT BETWEEN 35.0 AND 37.0;

-- [검증 5] 공간 무결성 확인 (기대값: 모두 0)
SELECT 'geom_4326_null_or_invalid' AS check_type, COUNT(*) AS cnt
FROM public.dim_stop WHERE geom_4326 IS NULL OR NOT ST_IsValid(geom_4326)
UNION ALL
SELECT 'geom_5187_null_or_invalid' AS check_type, COUNT(*) AS cnt
FROM public.dim_stop WHERE geom_5187 IS NULL OR NOT ST_IsValid(geom_5187);
*/
