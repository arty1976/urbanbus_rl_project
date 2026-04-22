from sqlalchemy import create_engine, text

DBURL = "postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus"
e = create_engine(DBURL)

with e.begin() as conn:
    conn.execute(text("DROP VIEW IF EXISTS public.gatv2_edge_primary_active"))

    conn.execute(text("""
        CREATE VIEW public.gatv2_edge_primary_active AS
        SELECT
            b.edge_uid,
            b.src_node_uid,
            b.dst_node_uid,
            b.src_node_index,
            b.dst_node_index,
            b.route_id,
            b.move_dir_code,
            b.distance_m::double precision AS distance_m,
            b.time_sec::double precision AS time_sec,
            b.edge_rank,
            b.distance_km::double precision AS distance_km,
            b.time_min::double precision AS time_min,
            b.distance_m_log::double precision AS distance_m_log,
            b.time_sec_log::double precision AS time_sec_log,
            b.src_node_index::integer AS src_idx,
            b.dst_node_index::integer AS dst_idx,
            b.time_sec::double precision AS generalized_cost,
            CASE
                WHEN b.distance_m >= 5000 THEN 1.0
                ELSE 0.0
            END::double precision AS long_edge_5km_flag
        FROM public.gatv2_edge_primary_active_base b
    """))

print("OK: recreated compatibility view with numeric casts")
