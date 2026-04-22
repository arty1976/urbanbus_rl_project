-- ============================================================
-- create_graph_edge_master.sql
-- 목적: 통합 그래프 간선 마스터 생성
-- 기준: link 간 이동, stop-link 연결, stop-stop 보행 연결 등을 공통 간선으로 관리
-- 작성일: 2026-04-10
-- ============================================================

create table if not exists public.graph_edge_master (
    edge_uid            text primary key,

    src_node_uid        text not null,
    dst_node_uid        text not null,

    src_node_type       text not null,
    src_node_id         text not null,
    dst_node_type       text not null,
    dst_node_id         text not null,

    edge_type           text not null,
    route_id            text null,
    move_dir_code       integer null,

    link_id             text null,
    link_seq            integer null,
    stop_id             text null,
    stop_seq            integer null,

    distance_m          numeric(12,3) null,
    time_sec            numeric(12,3) null,
    generalized_cost    numeric(12,3) null,

    is_bidirectional    boolean not null default false,
    is_active           boolean not null default true,

    created_at          timestamptz not null default now(),
    updated_at          timestamptz not null default now(),

    constraint ck_graph_edge_master_edge_type
        check (edge_type in (
            'LINK_TO_LINK',
            'STOP_TO_LINK',
            'LINK_TO_STOP',
            'STOP_TO_STOP',
            'ZONE_TO_STOP'
        ))
);

create index if not exists idx_graph_edge_master_src
    on public.graph_edge_master (src_node_uid);

create index if not exists idx_graph_edge_master_dst
    on public.graph_edge_master (dst_node_uid);

create index if not exists idx_graph_edge_master_type
    on public.graph_edge_master (edge_type);

create index if not exists idx_graph_edge_master_route_dir_seq
    on public.graph_edge_master (route_id, move_dir_code, link_seq, stop_seq);

create index if not exists idx_graph_edge_master_active
    on public.graph_edge_master (is_active);

comment on table public.graph_edge_master is
'통합 그래프 간선 마스터. 물리 이동 간선과 서비스 연결 간선을 공통 구조로 저장한다.';

comment on column public.graph_edge_master.edge_uid is
'통합 간선 키. 예: LINK_TO_LINK:R1000:0:12:8800123->8800456';

comment on column public.graph_edge_master.edge_type is
'간선 유형: LINK_TO_LINK / STOP_TO_LINK / LINK_TO_STOP / STOP_TO_STOP / ZONE_TO_STOP';

comment on column public.graph_edge_master.generalized_cost is
'거리, 시간, 혼잡도, 운영 제약을 반영한 일반화 비용';
