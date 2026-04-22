-- ============================================================
-- create_graph_node_master.sql
-- 목적: 통합 그래프 노드 마스터 생성
-- 기준: stop / link / zone / virtual pickup 을 단일 node_uid 체계로 관리
-- 작성일: 2026-04-10
-- ============================================================

create extension if not exists postgis;

create table if not exists public.graph_node_master (
    node_uid            text primary key,
    node_type           text not null,
    node_id             text not null,
    node_name           text null,

    route_id            text null,
    move_dir_code       integer null,

    lon                 numeric(12,8) null,
    lat                 numeric(12,8) null,
    geom_4326           geometry(Point, 4326) null,
    geom_5187           geometry(Point, 5187) null,

    is_active           boolean not null default true,
    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now(),

    constraint ck_graph_node_master_node_type
        check (node_type in ('STOP', 'LINK', 'ZONE', 'VIRTUAL_PICKUP'))
);

create index if not exists idx_graph_node_master_type_id
    on public.graph_node_master (node_type, node_id);

create index if not exists idx_graph_node_master_route_dir
    on public.graph_node_master (route_id, move_dir_code);

create index if not exists idx_graph_node_master_active
    on public.graph_node_master (is_active);

create index if not exists idx_graph_node_master_geom_4326
    on public.graph_node_master using gist (geom_4326);

create index if not exists idx_graph_node_master_geom_5187
    on public.graph_node_master using gist (geom_5187);

comment on table public.graph_node_master is
'통합 그래프 노드 마스터. STOP, LINK, ZONE, VIRTUAL_PICKUP 유형을 공통 node_uid 체계로 관리한다.';

comment on column public.graph_node_master.node_uid is
'통합 노드 키. 예: STOP:12345, LINK:8800123';

comment on column public.graph_node_master.node_type is
'노드 유형: STOP / LINK / ZONE / VIRTUAL_PICKUP';

comment on column public.graph_node_master.node_id is
'원천 도메인 식별자. stop_id 또는 link_id 등';

comment on column public.graph_node_master.route_id is
'노선 의존 노드의 경우 참고용 route_id. 노선 없는 운영 단계에서는 null 허용';

comment on column public.graph_node_master.move_dir_code is
'방향 코드. route_link_sequence 운영 기준과 정합되도록 유지';
