-- 02_ingest_jobs/promote_route_link_sequence_dedup.sql
-- Promote data from staging (dedup view) to route_link_sequence with strict validation.

INSERT INTO public.route_link_sequence (
    route_id, 
    move_dir_code, 
    link_seq, 
    link_id, 
    st_node_id, 
    ed_node_id, 
    gis_dist, 
    snapshot_file
)
SELECT 
    route_id_raw AS route_id,
    move_dir_raw AS move_dir_code,
    link_seq_raw::INTEGER AS link_seq,
    link_id_raw AS link_id,
    st_node_raw AS st_node_id,
    ed_node_raw AS ed_node_id,
    -- Cast to numeric only if it matches numeric pattern
    CASE 
        WHEN gis_dist_raw ~ '^\d+(\.\d+)?$' THEN gis_dist_raw::NUMERIC 
        ELSE NULL 
    END AS gis_dist,
    snapshot_file
FROM public.vw_stg_daegu_route_links_api_dedup
WHERE 1=1
  -- [STRICT VALIDATION RULES]
  AND link_seq_raw ~ '^\d+$'              -- Must be integer
  AND route_id_raw IS NOT NULL            -- No null route_id
  AND route_id_raw NOT IN ('', 'UNKNOWN') -- No empty/unknown route_id
  AND move_dir_raw IS NOT NULL            -- No null move_dir
  AND move_dir_raw != ''                  -- No empty move_dir
  AND link_id_raw != ''                   -- No empty link_id

  -- [EXCLUSIONS]
  AND route_id_raw <> '1000'              -- Exclude legacy sample route
  AND route_id_raw NOT IN (               -- Exclude routes with known API response issues
      '4040006020', '4040006021', '4040007001', '4040007009'
  )

  -- [SCOPE FILTER EXAMPLES]
  -- Uncomment one of these to limit scope for testing:
  -- AND route_id_raw = '1000'
  -- AND snapshot_file = 'response_1775701914078.csv'
  
ON CONFLICT (route_id, move_dir_code, link_seq) 
DO UPDATE SET 
    link_id = EXCLUDED.link_id,
    st_node_id = EXCLUDED.st_node_id,
    ed_node_id = EXCLUDED.ed_node_id,
    gis_dist = EXCLUDED.gis_dist,
    snapshot_file = EXCLUDED.snapshot_file,
    created_at = NOW();
