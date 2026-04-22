# 대구광역시 공간정보 레이어 프로파일 (Shapefile)

본 프로파일은 대구광역시 정류장, 노드, 도로 링크 형상 정보를 담은 Shapefile 구조 및 데이터 웨어하우스(Dimension / Bridge 테이블) 승격 가능성을 분석한 문서입니다.

## 1. 정류장 레이어 (bs_20250903)
* **File path**: `data/raw/daegu/shapes/extracted/2025-09-03/bs_20250903.shp`
* **Geometry type**: `Point`
* **Row count**: 5,705
* **Columns**: `gid`, `app_bg_dt`, `app_ed_dt`, `bs_id`, `bs_nm`, `bs_eng_nm`, `ngis_x_pos`, `ngis_y_pos`, `bsinfo_cnt`, `node_id`, `node_k_cd`, `route_no`, `bs_icon_no`, `bs_t_cd`, `dataconnar`, `geometry`
* **Candidate ID fields**: `bs_id` (Primary Key 후보), `node_id`
* **Risks and ambiguities**: 
  * `bs_id`의 결측치(Null) 중복 건수 모두 0건으로 완벽한 고유 식별자(PK) 보장.
  * 단, `bs_nm`(정류소 이름)의 중복 건수는 334건에 달하므로, **정류소 이름(`bs_nm`) 구별 및 조인은 절대 금지됨** (Data Contract Rule 준수 필).
* **Promotion assessment**:
  * **dim_stop (Dimension)**: 승격 조건 충족 (Fact). `bs_id`를 PK로 활용하여 즉각적인 마스터 테이블 생성이 가능합니다.

## 2. 노드 레이어 (node_20250903)
* **File path**: `data/raw/daegu/shapes/extracted/2025-09-03/node_20250903.shp`
* **Geometry type**: `Point`
* **Row count**: 8,067
* **Columns**: `gid`, `node_id`, `node_nm`, `node_eng_n`, `ngis_x_pos`, `ngis_y_pos`, `app_bg_dt`, `app_ed_dt`, `node_k_cd`, `dataconnar`, `geometry`
* **Candidate ID fields**: `node_id` (주요 교차지점 특화 고유 ID)
* **Risks and ambiguities**: `bs`의 `node_id` 값들과 결합하여 확장할 때 활용성이 크리라 예상됩니다.

## 3. 도로 링크 레이어 (link_20250903)
* **File path**: `data/raw/daegu/shapes/extracted/2025-09-03/link_20250903.shp`
* **Geometry type**: `LineString`
* **Row count**: 9,927
* **Columns**: `gid`, `link_id`, `link_nm`, `app_bg_dt`, `app_ed_dt`, `shape_len`, `dataconnar`, `geometry`
* **Candidate ID fields**: `link_id` (물리적 구간 고유 ID)

---

## 4. 데이터 모델 승격(Promotion) 분석 결론 총합

### 4.1. `dim_stop` (정류점 Dimension)
* **상태**: **승격 가능**
* **근거 (Fact)**: `bs` 레이어의 `bs_id`가 중복과 결측치 없이 고유함을 실측하여 검증 완료했습니다. 좌표 정보와 이름을 묶어 즉시 사용 가능한 상태입니다.

### 4.2. `dim_route` (노선 Dimension)
* **상태**: **공간정보 3개 파일만으로는 승격 불가, 그러나 `stg_daegu_routes` 기준으로는 승격 가능**
* **초안 컬럼 제안** (stg_daegu_routes 가정):
  - `route_id` (Primary Key)
  - `route_name`
  - `operation_type` (고정 노선 / 동적 DRT 등)
  - `description`
* **근거 (Assumption & Fact 혼재)**: 주어진 Shapefile 3종에서는 단일화된 노선 엔티티 정보가 파편화(`route_no` 텍스트 필드 등에 의존)되어 있어 차원(Dimension)화하기 부적합합니다. 하지만 원천 스키마 역할을 하는 개별 API/Staging 엔티티(`stg_daegu_routes`)가 마련되어 있다면 이를 통해 승격이 가능합니다.

### 4.3. `bridge_route_stop` (노선-정류장 브릿지 테이블)
* **상태**: **현재 승격 불가**
* **근거 (Assumption)**: 브릿지 테이블이 제 역할을 하려면 한 노선 내 정류장 운행 순서를 보장하는 `stop_sequence` 데이터가 필수적입니다. 현재 분석한 공간 레이어에서는 이를 입증하거나 재구성할 순서 정보 값이 부재하므로 레이어만으로는 승격이 불가능하다고 판단합니다.
