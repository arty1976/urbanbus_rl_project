-- =============================================================================
-- create_stg_daegu_route_links_api.sql
-- DDL for Daegu Bus API getLink02 response staging table
--
-- Rules:
--   - All API response columns are raw text.
--   - route_id_raw is injected from request metadata, not response body.
--   - snapshot_file ensures traceability to the original JSON file.
-- =============================================================================

CREATE TABLE IF NOT EXISTS stg_daegu_route_links_api (
    route_id_raw  text,          -- Injected from request parameter
    link_id_raw   text,          -- linkId
    st_node_raw   text,          -- stNode
    ed_node_raw   text,          -- edNode
    gis_dist_raw  text,          -- gisDist
    move_dir_raw  text,          -- moveDir
    link_seq_raw  text,          -- linkSeq
    
    snapshot_file text NOT NULL, -- Traceability
    loaded_at     timestamptz    DEFAULT now()
);

-- Index for snapshot load deduplication
CREATE INDEX IF NOT EXISTS idx_stg_route_links_snapshot
    ON stg_daegu_route_links_api (snapshot_file);

-- Index for link_id querying
CREATE INDEX IF NOT EXISTS idx_stg_route_links_link_id
    ON stg_daegu_route_links_api (link_id_raw);
