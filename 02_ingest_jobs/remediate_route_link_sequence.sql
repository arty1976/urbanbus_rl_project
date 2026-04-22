-- ==============================================================================
-- 프로젝트: UrbanBus RL 
-- 파일 설명: route_link_sequence 테이블의 레거시 오염 데이터 제거 및 중복 제거 뷰 생성
-- 작성일: 2026-04-14
-- ==============================================================================

-- 1. Analytical 테이블에서 레거시 샘플 데이터(route_id='1000') 제거
-- 예상 삭제 건수: 287행
DELETE FROM public.route_link_sequence
WHERE route_id = '1000';

-- 2. Staging 테이블의 중복을 제거하여 최신 데이터만 제공하는 뷰 생성
-- 원본 테이블: stg_daegu_route_links_api (38,960건의 중복 존재)
-- 전략: (route_id_raw, move_dir_raw, link_seq_raw) 기준으로 가장 최근(loaded_at DESC) 데이터 1건만 선택
CREATE OR REPLACE VIEW public.vw_stg_daegu_route_links_api_dedup AS
SELECT DISTINCT ON (route_id_raw, move_dir_raw, link_seq_raw)
    *
FROM public.stg_daegu_route_links_api
ORDER BY route_id_raw, move_dir_raw, link_seq_raw, loaded_at DESC;

-- 3. 검증 결과 리포트용 로그 (옵션)
-- SELECT count(*) FROM public.route_link_sequence WHERE route_id = '1000'; -- 기대값: 0
-- SELECT count(*) FROM public.vw_stg_daegu_route_links_api_dedup; -- 기대값: 유효 데이터 총합
