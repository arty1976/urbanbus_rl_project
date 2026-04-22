-- ============================================================
-- create_stop_link_mapping_master.sql
-- 목적: 정류장과 링크의 연결 관계를 관리하는 매핑 마스터 생성
-- 기준: 공간 근접, 노선-방향 추론, 수작업 보정 결과를 함께 관리
-- 작성일: 2026-04-10
-- ============================================================

create table if not exists public.stop_link_mapping_master (
    mapping_id            bigserial primary key,
    stop_id               text not null,
    link_id               text not null,

    mapping_type          text not null,
    distance_to_link_m    numeric(12,3) null,
    projection_ratio      numeric(10,6) null,
    confidence_score      numeric(10,6) null,

    route_id              text null,
    move_dir_code         integer null,

    is_primary_mapping    boolean not null default true,
    is_active             boolean not null default true,
    created_at            timestamptz not null default now(),
    updated_at            timestamptz not null default now(),

    constraint ck_stop_link_mapping_master_mapping_type
        check (mapping_type in ('SNAP_NEAREST', 'ROUTE_INFERRED', 'MANUAL'))
);

create unique index if not exists ux_stop_link_mapping_master_natural
    on public.stop_link_mapping_master (
        stop_id, link_id, coalesce(route_id, ''), coalesce(move_dir_code, -1)
    );

create index if not exists idx_stop_link_mapping_master_stop
    on public.stop_link_mapping_master (stop_id);

create index if not exists idx_stop_link_mapping_master_link
    on public.stop_link_mapping_master (link_id);

create index if not exists idx_stop_link_mapping_master_primary_active
    on public.stop_link_mapping_master (is_primary_mapping, is_active);

create index if not exists idx_stop_link_mapping_master_route_dir
    on public.stop_link_mapping_master (route_id, move_dir_code);

comment on table public.stop_link_mapping_master is
'정류장과 링크 간 연결 관계를 저장하는 마스터. 공간 snap, 노선 추론, 수작업 보정 결과를 함께 관리한다.';

comment on column public.stop_link_mapping_master.mapping_type is
'SNAP_NEAREST / ROUTE_INFERRED / MANUAL';

comment on column public.stop_link_mapping_master.projection_ratio is
'링크 시작점~끝점 중 정류장 투영 위치 비율(0~1)';

comment on column public.stop_link_mapping_master.confidence_score is
'매핑 신뢰도 점수. 자동 생성 후 수작업 검수 우선순위 산정에 활용';
