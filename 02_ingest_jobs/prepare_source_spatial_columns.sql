-- ============================================================
-- prepare_source_spatial_columns.sql
-- 목적: 공간 매핑 성능 최적화를 위해 원천 테이블에 EPSG:5187 영구 컬럼 추가
-- 작업 대상: public.dim_stop, public.gis_route_link
-- 작성일: 2026-04-10
-- ============================================================

DO $$
BEGIN
    -- 1. dim_stop 공간 컬럼 보강
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'dim_stop' AND column_name = 'geom_5187'
    ) THEN
        ALTER TABLE public.dim_stop ADD COLUMN geom_5187 geometry(Point, 5187);
        RAISE NOTICE 'Added geom_5187 to dim_stop';
    END IF;

    -- 2. gis_route_link 공간 컬럼 보강 (LineString)
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = 'gis_route_link' AND column_name = 'geom_5187'
    ) THEN
        ALTER TABLE public.gis_route_link ADD COLUMN geom_5187 geometry(LineString, 5187);
        RAISE NOTICE 'Added geom_5187 to gis_route_link';
    END IF;
END $$;

-- ------------------------------------------------------------
-- 백필 (Backfill)
-- ------------------------------------------------------------

-- dim_stop 백필 (기존 geom 4326 기반)
UPDATE public.dim_stop 
SET geom_5187 = ST_Transform(geom, 5187) 
WHERE geom_5187 IS NULL AND geom IS NOT NULL;

-- gis_route_link 백필
UPDATE public.gis_route_link 
SET geom_5187 = ST_Transform(geom, 5187) 
WHERE geom_5187 IS NULL AND geom IS NOT NULL;

-- ------------------------------------------------------------
-- 인덱스 (Spatial Indexes)
-- ------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_dim_stop_geom_5187 ON public.dim_stop USING GIST (geom_5187);
CREATE INDEX IF NOT EXISTS idx_gis_route_link_geom_5187 ON public.gis_route_link USING GIST (geom_5187);

ANALYZE public.dim_stop;
ANALYZE public.gis_route_link;
