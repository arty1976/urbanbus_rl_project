---
name: spatial_graph_mapping_guard
description: Guards the spatial projection and mapping rules between STOP and LINK nodes, maintaining strict node-edge integrity and coordinate constraints.
---

# Purpose

이 스킬은 정류장(STOP)과 도로망(LINK) 간의 물리적 위치 데이터를 기반으로 상호 연결하는 Phase 2/Phase 3 작업을 진행할 때, 공간 좌표 연산 오남용 방지 및 매핑 무결성(Node-Edge Integrity)을 강제하는 방어선(Guard) 역할을 합니다.

## Core Rules

- **Native Projection Only**: 거리 계산은 반드시 데이터베이스에 영구 보존된 `geom_5187` (EPSG:5187) 컬럼을 직접 사용하여 측정합니다. View 내부에서의 임시 `ST_Transform` 남용은 금지합니다.
- **Two-Tier Radius Fallback**: 1차 공간 매핑(`SNAP_NEAREST`)은 150m 반경 우선 탐색을 원칙으로 하며, 미탐지(Unmapped)된 정류장에 한해서만 예외적으로 500m Fallback 탐색을 허용합니다.
- **Single Primary Resolution**: 반경 내에 여러 링크가 걸리더라도, `row_number() OVER (PARTITION BY stop_id ORDER BY distance_m)` 방식을 사용해 무조건 최단 거리 1개만을 Primary 맵핑 데이터로 채택해야 합니다. Primary의 중복은 절대 금물입니다.
- **Route-Aware Refinement**: Phase 3 보정 작업 시 단순히 거리가 가까운 링크가 아니라, 노선의 연속성(`route_id, move_dir_code, sequence`) 정보에 기반하여 주행 경로상 합당한 링크로 수정(Refinement)이 이루어져야 합니다.

## Verification Queries

공간 맵핑 스크립트를 작성하거나 수행 후, 다음 기준을 에이전트가 검사/보고해야 합니다.

1. **Mapping Density**:
   - `SELECT COUNT(*) FROM stop_link_mapping_master WHERE distance_m > 100` : **경고(WARN)** 발생 및 수량 체크.
   - `SELECT COUNT(*) FROM stop_link_mapping_master WHERE distance_m > 250` : **위험(FAIL)** 수준으로 간주하여 수동 검토(MANUAL_REVIEW) 대상으로 분류.
2. **Duplicate Primary Mappings**:
   - 동일 `stop_id`에 대해 `is_primary = true` 인 링크가 두 개 이상인 경우가 없는지 `GROUP BY HAVING count > 1` 쿼리로 확인 (0건 필수).
3. **Ghost Nodes Check**:
   - `graph_node_master`에 등록된 전체 STOP 노드 수 대비 `stop_link_mapping_master`에 한 번도 등장하지 않는(Unmapped) 노드 규모 파악.

## Failure Rules

- **Geometry SRID Mismatch**: 타겟 테이블의 좌표계가 타겟 공간 데이터 스펙(예: 5187)이 아닐 경우 즉각 연산 중지 및 보고.
- **Node-Edge Foreign Key Drop**: 매핑 결과물이 `STOP_TO_LINK` 혹은 `LINK_TO_STOP` 에지(간선)로 형성될 때 참조될 수 없는 유령 노드(id 부재)가 있을 경우 FAILED.

## 호출 프롬프트 예시

- "대전시 Phase 2 정류장 맵핑 SQL을 작성해 줘. 반드시 `spatial_graph_mapping_guard` 스킬의 2-Tier Fallback 전략과 `geom_5187` 영구 컬럼 사용 규칙을 따라야 해."
- "어제 수행된 매핑 결과물(stop_link_mapping_master)에서 `spatial_graph_mapping_guard`의 검증 쿼리들을 실행하고 품질 경고(100m, 250m) 대상 리스트를 뽑아줘."
