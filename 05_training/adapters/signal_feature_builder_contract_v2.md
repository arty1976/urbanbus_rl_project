# Step 98 — Signal Feature Builder for Causal Simulator v2

## 0. 목적

Step 98의 목적은 Step 97에서 분류한 사용 가능 데이터 중 **정적 신호 인프라 feature**만을 causal simulator v2 입력 후보로 변환하는 것이다.

이 단계는 신호등 CSV(Comma-Separated Values=쉼표 구분값)를 읽어 다음 산출물을 생성한다.

- `node_signal_features.parquet`
- `edge_signal_features.parquet`
- `tensor_signal_feature_contract_v2.json`
- `signal_feature_quality_report.json`

---

## 1. 중요한 금지선

신호등 CSV는 동적 신호 현시 자료가 아니다.

따라서 아래 값은 생성하지 않는다.

| 금지 feature | 이유 |
|---|---|
| `red_light_delay_seconds` | 신호 phase, 차량 arrival trajectory 없음 |
| `green_time_seconds` | 현시 시간 자료 없음 |
| `cycle_length_seconds` | 신호 주기 자료 없음 |
| `signal_offset_seconds` | 연동 offset 자료 없음 |
| `real_time_signal_state` | 실시간 신호 상태 자료 없음 |
| `queue_discharge_rate` | 차로별 queue/flow 자료 없음 |

허용되는 것은 아래와 같은 정적 인프라 feature뿐이다.

| 허용 feature | 설명 |
|---|---|
| `signal_count_250m` | 정류장 주변 250m 내 신호 수 |
| `nearest_signal_distance_m` | 가장 가까운 신호까지 거리 |
| `pedestrian_signal_count_250m` | 주변 보행 신호 수 |
| `blink_signal_ratio_250m` | 주변 점멸 신호 비율 |
| `controlled_signal_ratio_250m` | 주변 제어 신호 비율 |
| `edge_signal_count` | edge 양끝 또는 edge 주변 신호 수 proxy |
| `edge_signal_density_per_km` | edge 길이 대비 신호 밀도 |

---

## 2. 입력 계약

### 2.1 필수 입력

| 인자 | 설명 |
|---|---|
| `--signal-csv` | 대구 신호등 CSV 경로 |
| `--output-dir` | 산출물 저장 디렉터리 |

### 2.2 node/edge source 입력 방식

둘 중 하나를 사용한다.

#### 방식 A: Parquet 입력

| 인자 | 설명 |
|---|---|
| `--node-source` | `node_uid`, `node_index`, 좌표 컬럼이 있는 parquet |
| `--edge-source` | `src_idx`, `dst_idx`, `distance_m`이 있는 parquet |

#### 방식 B: DB 입력

| 인자 | 설명 |
|---|---|
| `--db-url` | PostgreSQL DSN(Data Source Name=데이터 원본 이름) |
| `--node-source-db` | 기본값 `public.gatv2_node_master_active` |
| `--edge-source-db` | 기본값 `public.gatv2_edge_primary_active` |

DB 방식은 `gatv2_node_master_active.node_uid = 'STOP:' || dim_stop.stop_id` 조인을 시도하고, `dim_stop.geom_5187`에서 node 좌표를 읽는 것을 기본 가정으로 둔다.

---

## 3. 출력 계약

### 3.1 node_signal_features.parquet

필수 컬럼:

```text
node_uid
node_index
signal_count_100m
signal_count_250m
signal_count_500m
nearest_signal_distance_m
pedestrian_signal_count_250m
blink_signal_ratio_250m
controlled_signal_ratio_250m
signal_delay_risk_proxy
intersection_complexity_proxy
signal_feature_quality_flag
```

### 3.2 edge_signal_features.parquet

필수 컬럼:

```text
src_idx
dst_idx
distance_m
edge_signal_count
edge_signal_density_per_km
edge_nearest_signal_distance_m
edge_control_complexity_proxy
edge_signal_feature_quality_flag
```

### 3.3 tensor_signal_feature_contract_v2.json

필수 내용:

```text
source_csv_path
source_csv_hash
coordinate_mode
join_method
feature_definitions
unavailable_dynamic_signal_fields
claim_guardrails
```

### 3.4 signal_feature_quality_report.json

필수 내용:

```text
signal_csv_row_count
node_row_count
edge_row_count
node_feature_row_count
edge_feature_row_count
coordinate_mode
step_98_ready
warnings
```

---

## 4. proxy feature 원칙

`signal_delay_risk_proxy`, `intersection_complexity_proxy`, `edge_control_complexity_proxy`는 실제 측정값이 아니다.

이 feature들은 다음 의미로만 사용한다.

- 신호 인프라가 많은 지역일수록 운행 복잡도가 높을 수 있다는 약한 사전정보
- policy observation context
- simulator 환경 난이도 proxy

논문 또는 보고서에서 이것을 실제 신호 지연 시간으로 주장하면 안 된다.

---

## 5. Step 98 성공 판정

Step 98 self-test는 다음을 확인한다.

1. builder script가 문법 오류 없이 compile되는가
2. synthetic node/edge/signal CSV에서 4개 산출물이 생성되는가
3. node output에 필수 컬럼이 있는가
4. edge output에 필수 컬럼이 있는가
5. 금지 feature가 output에 섞이지 않았는가
6. claim guardrail이 `performance_claim_allowed=false`로 유지되는가

---
