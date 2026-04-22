create table if not exists stg_daegu_stops_geo (
  stop_name_ko text,
  stop_name_en text,
  sido text,
  gugun text,
  dong text,
  lon double precision,
  lat double precision,
  pass_route_count integer,
  pass_routes text,
  source_file text,
  loaded_at timestamp default now()
);

create table if not exists stg_daegu_routes (
  route_type text,
  route_id bigint,
  route_no text,
  route_dir text,
  origin_stop_id bigint,
  origin_stop_name text,
  dest_stop_id bigint,
  dest_stop_name text,
  source_file text,
  loaded_at timestamp default now()
);

create table if not exists stg_daegu_shape_registry (
  layer_name text,
  file_group_date date,
  shp_path text,
  dbf_path text,
  prj_path text,
  shx_path text,
  source_zip text,
  loaded_at timestamp default now()
);
