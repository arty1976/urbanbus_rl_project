from sqlalchemy import create_engine, inspect, text

DBURL = "postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus"
e = create_engine(DBURL)

insp = inspect(e)
views = set(insp.get_view_names(schema="public"))

with e.begin() as conn:
    if "gatv2_edge_primary_active_base" not in views:
        conn.execute(text("ALTER VIEW public.gatv2_edge_primary_active RENAME TO gatv2_edge_primary_active_base"))
    else:
        conn.execute(text("DROP VIEW IF EXISTS public.gatv2_edge_primary_active"))

    conn.execute(text("""
        CREATE VIEW public.gatv2_edge_primary_active AS
        SELECT
            b.*,
            b.src_node_index AS src_idx,
            b.dst_node_index AS dst_idx,
            b.time_sec::double precision AS generalized_cost,
            (b.distance_m >= 5000)::boolean AS long_edge_5km_flag
        FROM public.gatv2_edge_primary_active_base b
    """))
print("OK: created compatibility view public.gatv2_edge_primary_active")
