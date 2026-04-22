# project_log.md 삽입용 초안 — 2026-04-19 GATv2 입력 레이어

> 아래 블록을 `C:\Users\ryujo\urbanbus_rl_project\project_log.md` 의
> **동일 2026-04-19 섹션 내부**, "graph_edge_master STOP_TO_STOP 복원 — leg-aggregation
> 도입" 블록 **아래** 에 이어서 붙여 주세요. (날짜 헤더 `## 📅 2026-04-19`
> 는 새로 만들지 말고 기존 것 공유.)
>
> 2026-04-19 21:51 실행 완료. 모든 수치는 실측치로 채움.

---

### GATv2 입력 레이어 구성 — 5개 view + PyG dataset contract [APPROVED]

`graph_edge_master STOP_TO_STOP v2` 가 승인된 직후, GATv2 학습 파이프라인
(PyTorch Geometric) 이 읽을 **DB 뷰 5종** 과 **dataset 계약 문서** 를 확정하여,
**"static graph skeleton + per-snapshot node features"** 라는 dual layer 를
DB 레벨에서 고정했다. readiness 는 2026-04-19 21:51 ~ 23:58 (7,664.4 s) 에
걸쳐 실행되어 모든 hard gate 를 통과했다.

#### 1. 생성/수정 파일

| 파일 | 역할 |
|------|------|
| `04_model_inputs/create_gatv2_training_views.sql` [NEW] | 5개 view 생성 + 8개 readiness 체크 (single-file idempotent) |
| `04_model_inputs/run_gatv2_views.ps1` [NEW] | psql 러너. ASCII-only (PS 5.1 CP949 안전), `$LASTEXITCODE` 판정 |
| `05_training/pytorch_geometric_dataset_contract.md` [NEW] | PyG `Data` 객체 컬럼 순서, split 규칙, invariant C1~C8 |

#### 2. 생성된 view 5종

| View | 역할 | PyG 대응 |
|------|-----|---------|
| `public.gatv2_node_master_active` | primary+active STOP_TO_STOP 엣지 참여 STOP 노드. 0-based dense `node_index` | `num_nodes`, **x** 행 순서 |
| `public.gatv2_edge_primary_active` | edge_index 재료 + `distance_km/time_min/*_log` (F_e=4) | **edge_index**, **edge_attr** |
| `public.gatv2_snapshot_stop_features_train` | per-(state_ts, node) 학습 row. overnight/terminal 전이 제외 | **x** (F_x=10) + **y** (F_y=3) |
| `public.gatv2_snapshot_summary` | snapshot 수 / 시간 범위 / rows 분포 | 로더 메타 / split 기준 |
| `public.gatv2_edge_summary` | 그래프 스켈레톤 통계 | shape sanity |

#### 3. 주요 설계 원칙

- **노드 모집단**: primary+active STOP_TO_STOP 엣지 참여 STOP 만. 고립/terminal-only stop 제외.
- **엣지 모집단**: `edge_type='STOP_TO_STOP' AND is_primary_edge AND is_active`.
- **node_index**: `node_uid` 오름차순 `row_number()-1`. node_uid set 불변이면 세션 간 고정.
- **학습 제외 규칙**: `is_overnight_gap=1 OR is_terminal_transition=1` 전량 컷.
- **Target 변환**: `next_*_recent` 에 `log1p` 동시 노출 — MSE in log space, 평가시 `expm1` 복원.
- **edge_attr**: `distance_km`, `time_min`, `distance_m_log`, `time_sec_log` 4열. Scaling 은 뷰 아님.
- **방향성**: `is_bidirectional=false` 유지. 역방향 엣지는 모델 내부에서 처리.
- **범위 밖**: dynamic traffic edge / heterogeneous graph / RL policy — 모두 이번 단계 아님.

#### 4. Readiness 실측 결과 ([APPROVED])

| ID | 지표 | 측정값 | 기준 | 판정 |
|----|------|-------|------|------|
| V1 | num_nodes | **4,116** | ≤ 5,705 (graph_node_master STOP seed) | ✅ |
| V1 | index_gap_check_zero_ok | **0** | = 0 | ✅ |
| V2 | num_edges | **5,484** | = graph_edge_master v2 primary_edges | ✅ |
| V2 | distinct_src / distinct_dst | 4,096 / 4,089 | < num_nodes | ✅ |
| V2 | orphan_index_cnt / self_loop_cnt | **0 / 0** | = 0 | ✅ |
| V2 | rank1_cnt / non_rank1_cnt | **5,484 / 0** | 100% primary | ✅ |
| V3 | distinct_route_dirs | 301 | — | 보고 |
| V3 | distance (min/avg/p50/p95/max) m | 6.88 / 571.4 / 404.2 / 1,475.8 / 17,438.65 | 대구 반경 17km 급 정상 | ✅ |
| V3 | time_sec (min/avg/max) | 1.24 / 102.86 / 3,138.96 | 정상 (17km @ 20km/h ≈ 3060s) | ✅ |
| V4 | **total_training_rows** | **20,289,600** | ≤ 20,490,432 (overnight/terminal 제외) | ✅ |
| V4 | orphan_node_index_cnt | **0** | = 0 | ✅ |
| V4 | overnight_leak_cnt / terminal_leak_cnt | **0 / 0** | = 0 | ✅ |
| V4 | non_stop_leak_cnt | **0** | = 0 | ✅ |
| V5 | distinct_snapshots | **6,570** | ≤ 6,935 | ✅ |
| V5 | min_state_ts / max_state_ts | 2023-01-01 05:00 / 2023-12-31 22:00 | 2023 전구간 | ✅ |
| V5 | rows_per_snapshot (min/avg/max) | 2,848 / 3,088.22 / 3,157 | 균일 분포 | ✅ |
| V6 | nodes_with_training_rows | 3,333 | 노드 모집단 4,116 중 81% | 보고 |
| V6 | rows_per_node (min/avg/max) | 18 / 6,087.49 / 6,570 | max=distinct_snapshots 일치 | ✅ |
| V7 | nodes_without_any_training_row | **783** | — | ⚠️ (§4-A 참고) |
| V8 | negative_target_cnt | **0** | = 0 | ✅ |
| V8 | next_boardings (min/max/avg) | 0 / 750 / 8.83 | — | 보고 |
| V8 | max_next_waiting | 674 | — | 보고 |

**산술 대조**: `avg_rows_per_snapshot × distinct_snapshots`
= 3,088.22 × 6,570 = 20,289,605 ≈ **20,289,600** (total_training_rows). 두 뷰 간 정합 ✓

##### §4-A.  V7 = 783 노드 해석

4,116개 노드 중 783개 (≈19%) 는 그래프 토폴로지(엣지 양끝)에는 존재하지만 2023년
365일 동안 승하차 기록이 0 이어서 `rl_stop_transition_features` 에 한 행도 없다.
이는 고립(isolated) 이 아니라 **passenger-activity-zero stop** 이다. 대구 외곽
/ 시즌성 / 2023년 중 개설된 정류장 패턴과 부합. PyG `Data` 에서 `node_mask`
= False 로 자동 처리되므로 추가 조치 불필요 — dataset contract §1.2 에 이미 명세됨.

#### 5. 숨은 발견 — 뷰 쿼리 지연 2시간 7분

`create_views.log` 총 실행 7,664.4s. 원인은 `rl_stop_transition_features` 자체가
**뷰** 이고, 그 위의 `gatv2_snapshot_stop_features_train` 도 뷰이기 때문에 매
쿼리마다 21.6M row CTE 가 재전개된다. 학습 시 PyG DataLoader 가 이 뷰를
반복 스캔하면 I/O 병목. **다음 단계 진행 전 materialization 결정 필요**:

- **옵션 A (권장)**: `CREATE TABLE gatv2_snapshot_stop_features_train_mat AS SELECT * FROM gatv2_snapshot_stop_features_train;`
  → `(state_ts, node_index)` 복합 인덱스. 한 번 ~2시간 걸리지만 이후는 O(1) 조회.
- **옵션 B**: 날짜 기반 시간 윈도우로 잘라 읽는 파이썬 로더.

#### 6. 다음 단계 — `05_training/build_gatv2_dataset.py` 진행 가능 여부

**YES — 블로킹 해제**. 단 선결 과제 1 건 권장:

| # | 과제 | 비용 | 블로킹? |
|---|------|-----|---------|
| 1 | `gatv2_snapshot_stop_features_train` materialize (§5 옵션 A) | ~2h (1회) | **권장** (성능 문제로 학습 못 돌면 어차피 해야 함) |
| 2 | `build_gatv2_dataset.py` 스켈레톤 (contract §9) | - | 없음 |
| 3 | invariant C1~C8 assert | - | 없음 |
| 4 | 시간축 split (train 01-10 / val 11 / test 12) | - | 없음 |

#### 7. 부산물 / 정리

- 뷰 DDL 은 `CREATE OR REPLACE VIEW` idempotent. 재실행 안전.
- 뷰 생성 자체는 수 초지만, readiness 체크 (R1~R8) 가 20M row 반복 집계라 시간 대부분 차지.
- `05_training/` 디렉터리 신규 도입. 앞으로 `build_gatv2_dataset.py`, `train_gatv2.py`,
  `eval_gatv2.py`, `gatv2_model.py` 가 이 폴더 안에 자리잡을 예정.
- `run_gatv2_views.ps1` 은 ASCII-only 로 재정비 (PS 5.1 CP949 이슈로 초기 한글 버전이
  파싱 실패 → 영문으로 치환, 인코딩 공격면 제거).
