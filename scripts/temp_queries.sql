-- scripts/temp_queries.sql
-- 1. 삭제 전 재확인
SELECT route_id, move_dir_code, COUNT(*) AS row_count
FROM public.route_link_sequence
WHERE route_id = '1000'
GROUP BY route_id, move_dir_code
ORDER BY move_dir_code;

SELECT COUNT(*) AS total_rows
FROM public.route_link_sequence
WHERE route_id = '1000';

-- 2. promoted 오염 정리 실행
DELETE FROM public.route_link_sequence
WHERE route_id = '1000';

-- 3. 삭제 후 재확인
SELECT COUNT(*) AS remaining_rows
FROM public.route_link_sequence
WHERE route_id = '1000';

-- 4. dedup 뷰 생성
CREATE OR REPLACE VIEW public.vw_stg_daegu_route_links_api_dedup AS
SELECT DISTINCT ON (route_id_raw, move_dir_raw, link_seq_raw)
       route_id_raw AS route_id,
       move_dir_raw AS move_dir_code,
       link_seq_raw AS link_seq,
       route_id_raw,
       link_id_raw,
       st_node_raw,
       ed_node_raw,
       gis_dist_raw,
       move_dir_raw,
       link_seq_raw,
       snapshot_file,
       loaded_at
FROM public.stg_daegu_route_links_api
ORDER BY route_id_raw, move_dir_raw, link_seq_raw, loaded_at DESC;
