# PyTorch Geometric Dataset Contract — GATv2 Transition Learning

> 본 문서는 urbanbus_rl_project 의 **Graph-ready transition learning** 단계에서,
> DB 뷰 5종 (`04_model_inputs/create_gatv2_training_views.sql` 산출물) 으로부터
> PyTorch Geometric (PyG) 입력 텐서를 **어떻게 구성해야 하는지** 공식 계약한다.
>
> - 상위 단계 산출물: `graph_edge_master` (STOP_TO_STOP v2, 99.91% coverage),
>   `rl_stop_transition_features` (APPROVED), `graph_state_timeslice` (APPROVED).
> - 본 계약이 고정된 뒤, 실제 파이썬 구현은 `05_training/build_gatv2_dataset.py` 에서 수행.
> - 현 단계는 **transition (t → t+1) 예측** 이며 **RL 정책 학습은 아직 아님**.
> - dynamic traffic edge / link 피처는 범위 **밖** (후속 단계).

---

## 1. 전체 데이터 모델

### 1.1 입력 뷰 → PyG 객체 매핑

| DB 뷰 | PyG 대응 | 주 기능 |
|------|---------|--------|
| `public.gatv2_node_master_active` | **x** 행 순서 정의, `num_nodes` | 노드 집합 고정 + `node_index` (0-based dense) 부여 |
| `public.gatv2_edge_primary_active` | **edge_index**, **edge_attr** | 정적 그래프 스켈레톤 — snapshot 공통 |
| `public.gatv2_snapshot_stop_features_train` | **x** 성분값 + **y** 성분값 (per snapshot) | time-varying per-node feature / target |
| `public.gatv2_snapshot_summary` | 로더 메타 | split 기준 시간범위, 예상 배치수 |
| `public.gatv2_edge_summary` | 토폴로지 sanity | num_nodes/edges 대조 (torch shape 검증) |

### 1.2 PyG 표현 — Snapshot-level Data 객체

한 `state_ts` 당 `torch_geometric.data.Data` 하나를 만든다 (아래 "Option A").
샘플 효율이 필요해지면 Option B 로 전환.

```text
Data(
    x           : FloatTensor [num_nodes, F_x]
    edge_index  : LongTensor  [2, num_edges]
    edge_attr   : FloatTensor [num_edges, F_e]
    y           : FloatTensor [num_nodes, F_y]
    node_mask   : BoolTensor  [num_nodes]            # 해당 snapshot 에 학습 row 있는 노드만 True
    state_ts    : Tensor      []  (metadata)         # time-like key
    delta_t_hr  : FloatTensor [num_nodes]            # per-row effective elapsed time
)
```

### 1.3 Option A vs Option B

| Option | 한 줄 정의 | 장점 | 단점 | 권장 |
|--------|---------|-----|-----|------|
| **A** | 1 snapshot = 1 Data, `x`/`y` 는 `[num_nodes, ...]` 로 풀 패딩 | 구현 단순, edge_index 재사용 | 전 노드 대비 실제 학습 row 밀도가 낮으면 loss mask 필수 | **1차 학습 권장** |
| **B** | 1 row = 1 mini-sample, edge_index 는 DataLoader collate 시점에서 배치 공유 | 밀도 낮은 snapshot 에서 비용 절감 | edge_index broadcasting 구현 필요 | 데이터셋 커질 때 |

현 단계 (dataset 약 2,000 만 row 급) 는 **Option A + `node_mask`** 를 사용한다.

---

## 2. 노드 인덱싱 계약

### 2.1 권위 있는 인덱스 소스
- `gatv2_node_master_active.node_index` (0-based dense, `num_nodes - 1` 까지 연속).
- 파이썬에서 load 순서: **먼저 node_master_active 전량 로드** → `dict[node_uid] = node_index`.
- 이후 edge / snapshot 뷰를 읽을 때 `node_index` 컬럼만 신뢰 (자체 해시 금지).

### 2.2 불변 원칙
- `node_uid` 집합이 바뀌지 않는 한 `node_index` 는 세션 간 **고정**.
- `graph_edge_master` 에 stop 이 추가/제거되어 node 집합이 바뀌는 경우 **dataset 전체 재빌드** — 부분 갱신 금지 (PyG shape contract 깨짐).

### 2.3 PyG 변환 규약
```python
# node_master_df  : pandas.DataFrame from gatv2_node_master_active
node_master_df = node_master_df.sort_values("node_index").reset_index(drop=True)
assert (node_master_df["node_index"].values == np.arange(len(node_master_df))).all()
num_nodes = int(node_master_df["num_nodes"].iloc[0])
uid_to_idx = dict(zip(node_master_df["node_uid"], node_master_df["node_index"]))
```

---

## 3. edge_index / edge_attr 계약

### 3.1 텐서 구성
```python
edge_src = torch.as_tensor(edge_df["src_node_index"].values, dtype=torch.long)
edge_dst = torch.as_tensor(edge_df["dst_node_index"].values, dtype=torch.long)
edge_index = torch.stack([edge_src, edge_dst], dim=0)       # shape [2, E]
```
- 방향성 있음 (`is_bidirectional = false` 원칙). 역방향 edge 는 생성하지 않음.
- GATv2 에서 역방향 정보가 필요하면 **모델 레이어 안** 에서 `add_reverse=True` / `to_undirected()` 처리. **뷰에서 두 방향을 생성하지 않는다.**

### 3.2 edge_attr 컬럼 순서 (F_e = 4)
| idx | 컬럼 | 전처리 |
|-----|-----|-------|
| 0 | `distance_km` | raw (이미 / 1000) |
| 1 | `time_min` | raw (이미 / 60) |
| 2 | `distance_m_log` | `ln(1 + distance_m)` |
| 3 | `time_sec_log` | `ln(1 + time_sec)` |

> scaling (Standard / MinMax) 은 모델 파이프라인 안의 `torch.nn.BatchNorm1d` /
> 별도 전처리 레이어에서 수행. **뷰 레벨에선 하지 않는다** — 학습/추론 분포 드리프트 보호.

### 3.3 불변 체크
- `edge_index.max() == num_nodes - 1` 반드시 참.
- `orphan_index_cnt` (readiness V2) 는 반드시 0.

---

## 4. x (node feature) 계약

### 4.1 컬럼 순서 (F_x = 10)
| idx | 컬럼 | 타입 | 전처리 메모 |
|-----|-----|------|-----------|
| 0 | `boardings_recent_log` | float | log1p 이미 적용됨 |
| 1 | `alightings_recent_log` | float | log1p 이미 적용됨 |
| 2 | `waiting_passenger_cnt_log` | float | log1p 이미 적용됨 |
| 3 | `hour_sin` | float | ∈ [-1, 1] |
| 4 | `hour_cos` | float | ∈ [-1, 1] |
| 5 | `dow_sin` | float | ∈ [-1, 1] |
| 6 | `dow_cos` | float | ∈ [-1, 1] |
| 7 | `is_peak` | bool→float | 0/1 캐스팅 |
| 8 | `node_type_stop` | int→float | 항상 1 (현 데이터셋) — 다형그래프 대비 보존 |
| 9 | `delta_t_hr_effective` | float | effective 전이 시간 (NULL → fill 0 금지, snapshot 제외로 해결) |

### 4.2 Null / NaN 처리
- `delta_t_hr_effective` 는 `is_overnight_gap=1` 일 때 NULL 이지만, **training 뷰는 이미 overnight/terminal 제외** 하므로 학습 row 에는 NULL 이 없어야 한다.
- 만약 NaN 이 발견되면 **snapshot drop** (fill 아님). 원천 rl_stop_transition_features 의 readiness 를 재검증.

### 4.3 raw 컬럼 (입력 아님, 로그·디버깅용)
`boardings_recent`, `alightings_recent`, `waiting_passenger_cnt`, `hour_of_day`, `day_of_week` 은 raw 로 포함되어 있지만 **x 텐서엔 넣지 않음**. 필요 시 metadata 로만 보관.

---

## 5. y (target) 계약

### 5.1 컬럼 순서 (F_y = 3)
| idx | 컬럼 | 손실 | 메모 |
|-----|-----|-----|------|
| 0 | `next_boardings_recent_log` | MSE (log space) | 실 사용 시 `expm1` 복원 후 MAE 계산 |
| 1 | `next_alightings_recent_log` | MSE (log space) | 동상 |
| 2 | `next_waiting_passenger_cnt_log` | MSE (log space) | RL 보상에서 pressure signal 의 원전, 중요도 ↑ |

### 5.2 Multi-task weights (기본값)
- 균등 1:1:1 로 시작. `next_waiting_passenger_cnt_log` 은 튜닝 후 w=2 까지 상향 가능 (SAR 명세에서 primary pressure signal 로 지정됨).

### 5.3 학습 vs 평가 지표
- **학습 loss**: log space MSE (분포 skew 대응).
- **평가 지표**: `expm1` 로 복원 후 MAE / RMSE / MAPE — 운영 해석용.

---

## 6. 학습 샘플 필터링 규칙

DB 뷰 `gatv2_snapshot_stop_features_train` 단계에서 이미 적용된 필터:
- `node_type = 'STOP'` (현재 단일이지만 heterogeneous 대비 명시)
- `is_overnight_gap = 0`
- `is_terminal_transition = 0`
- `node_uid ∈ gatv2_node_master_active` (고립 노드 제외)

PyG 쪽에서 **추가로 적용할 필터는 없다** — 이중 배제는 계약 위반. 만약 샘플이 이상하게 많으면 DB 뷰 readiness 부터 재확인.

---

## 7. 데이터셋 분할 (Split) 규칙

### 7.1 시간 기반 split (권장)
2023-01 ~ 2023-12 의 365일 데이터를 **시간축을 따라** 쪼갠다. **무작위 셔플 금지** — 시간적 누수.

| Split | 기간 | 목적 |
|-------|------|-----|
| train | 2023-01-01 ~ 2023-10-31 | 학습 |
| val   | 2023-11-01 ~ 2023-11-30 | 조기종료 / HP 튜닝 |
| test  | 2023-12-01 ~ 2023-12-31 | 최종 평가 |

### 7.2 구현 쪽 주의
- `state_ts::date` (`service_date`) 기준으로 cutoff.
- DataLoader `shuffle=True` 는 **train split 안에서만** 허용.

---

## 8. 배치 전략

### 8.1 Option A (권장 초기값)
- 1 snapshot → 1 `Data` 객체.
- `DataLoader(batch_size=B_snapshots)` → `torch_geometric.loader.DataLoader` 가 block-diagonal 로 edge_index concat.
- 초기 B_snapshots = 8. 메모리 프로파일 후 조정.

### 8.2 edge_index 중복
- 동일 그래프 skeleton 이 모든 snapshot 에서 반복되므로 DataLoader batch 안에서 edge_index 는 실제로 **B× 복제** 된다. 메모리 이슈 발생 시 Option B 또는 PyG `temporal_data` 로 전환.

---

## 9. 코드 스켈레톤 (참고 — 정식 구현은 build_gatv2_dataset.py)

```python
import pandas as pd, torch
from sqlalchemy import create_engine
from torch_geometric.data import Data

X_COLS = [
    "boardings_recent_log", "alightings_recent_log", "waiting_passenger_cnt_log",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "is_peak", "node_type_stop", "delta_t_hr_effective",
]
Y_COLS = [
    "next_boardings_recent_log",
    "next_alightings_recent_log",
    "next_waiting_passenger_cnt_log",
]
E_COLS = ["distance_km", "time_min", "distance_m_log", "time_sec_log"]

def load_static_graph(eng):
    nodes = pd.read_sql(
        "select node_uid, node_index, num_nodes from public.gatv2_node_master_active "
        "order by node_index", eng)
    edges = pd.read_sql(
        "select src_node_index, dst_node_index, " + ", ".join(E_COLS)
        + " from public.gatv2_edge_primary_active", eng)
    num_nodes = int(nodes["num_nodes"].iloc[0])
    edge_index = torch.as_tensor(
        edges[["src_node_index", "dst_node_index"]].values.T, dtype=torch.long)
    edge_attr = torch.as_tensor(edges[E_COLS].values, dtype=torch.float32)
    return num_nodes, edge_index, edge_attr, nodes

def build_snapshot(df_ts, num_nodes):
    x = torch.zeros((num_nodes, len(X_COLS)), dtype=torch.float32)
    y = torch.zeros((num_nodes, len(Y_COLS)), dtype=torch.float32)
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    idx = torch.as_tensor(df_ts["node_index"].values, dtype=torch.long)
    x[idx] = torch.as_tensor(df_ts[X_COLS].astype(float).values, dtype=torch.float32)
    y[idx] = torch.as_tensor(df_ts[Y_COLS].astype(float).values, dtype=torch.float32)
    mask[idx] = True
    return x, y, mask
```

---

## 10. 불변성 / 검증 체크리스트 (구현 단계에서 반드시 통과)

| # | 조건 | 확인 방법 |
|---|------|---------|
| C1 | `x.shape == (num_nodes, 10)` | `assert` |
| C2 | `y.shape == (num_nodes, 3)` | `assert` |
| C3 | `edge_index.max() == num_nodes - 1` | `assert` |
| C4 | `edge_attr.shape[1] == 4` | `assert` |
| C5 | `node_mask.sum() == len(df_ts)` | snapshot 당 row 수 = mask=True 수 |
| C6 | snapshot 수 == `gatv2_snapshot_summary.distinct_snapshots` | 집계 대조 |
| C7 | total rows == `gatv2_snapshot_summary.total_training_rows` | 집계 대조 |
| C8 | NaN/Inf 검사 | `torch.isfinite(x).all() and torch.isfinite(y).all()` |

---

## 11. 범위 밖 (Out of Scope)

| 항목 | 현재 상태 |
|------|---------|
| dynamic traffic edge feature (속도/혼잡) | **다음 페이즈**. graph_state_timeslice 에 link_* 컬럼 NULL. |
| heterogeneous graph (STOP + LINK 혼합) | 현재 STOP only. node_type_stop 컬럼은 확장 대비 보존만. |
| RL policy gradient / environment | 현재는 transition 예측 supervised. RL 은 별도 문서 (`rl_sar_spec.md`). |
| temporal attention over snapshots | 현 stage 에서 각 snapshot 독립 처리. time-aware aggregation 은 후속. |
| bidirectional edge duplication | 정책 상 단방향 유지. 필요 시 모델 안에서 처리. |

---

## 12. 버전 / 변경 이력

| 날짜 | 버전 | 변경 | 사유 |
|------|-----|------|-----|
| 2026-04-19 | v1 | 최초 작성 | graph_edge_master v2 approve 직후 GATv2 입력 파이프라인 착수 |
