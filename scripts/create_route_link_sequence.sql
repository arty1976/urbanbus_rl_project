-- scripts/create_route_link_sequence.sql
-- Create refined table for route link sequences with proper data types.

CREATE TABLE IF NOT EXISTS public.route_link_sequence (
    route_id        TEXT NOT NULL,
    move_dir_code   TEXT NOT NULL, -- moveDir raw code; domain not confirmed yet
    link_seq        INTEGER NOT NULL,
    link_id         TEXT NOT NULL,
    st_node_id      TEXT NULL,
    ed_node_id      TEXT NULL,
    gis_dist        NUMERIC NULL,
    snapshot_file   TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (route_id, move_dir_code, link_seq)
);

-- Index for efficient lookup during validation
CREATE INDEX IF NOT EXISTS idx_route_link_seq_route_dir 
ON public.route_link_sequence (route_id, move_dir_code);
