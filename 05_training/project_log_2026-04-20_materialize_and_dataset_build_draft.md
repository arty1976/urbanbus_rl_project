# project_log.md 삽입용 초안 — 2026-04-20 GATv2 학습 데이터셋 준비

> 아래 블록을 `C:\Users\ryujo\urbanbus_rl_project\project_log.md` 의
> **새 날짜 헤더 `## 📅 2026-04-20` 아래** 에 붙여 주세요.
> (전날 2026-04-19 의 GATv2 inputs 블록 직후 시간순으로 위치.)
>
> 2026-04-20 기록. 모든 수치는 실측치.

---

### 학습 뷰 materialize + PyG dataset 빌더 — 환경/배선 모두 통과 [APPROVED]

전날 확정한 5개 view 위에 두 개의 파이프라인 단계를 더해 GATv2 학습 직전까지의
경로를 끝까지 배선 완료. 슬로우 뷰 한 번을 매트리얼라이즈드 테이블로 고정하고,
PyG `Data` 빌더를 작성하여 smoke test 에서 invariant 8종이 모두 통과함을 확인.

#### 1. 생성/수정 파일

| 파일 | 역할 |
|------|------|
| `04_model_inputs/materialize_gatv2_training_table.sql` [NEW] | view → 테이블 변환 + 3개 인덱스 + VACUUM ANALYZE + parity check |
| `04_model_inputs/run_materialize.ps1` [NEW] | psql 러너. ASCII-only (PS 5.1 CP949 안전), `$LASTEXITCODE` 판정 |
| `05_training/build_gatv2_dataset.py` [NEW] | PyG `Data` 객체 빌더. mat-table 자동 감지, time-based split, invariant C1~C8 assert |
| `05_training/run_build_dataset.ps1` [NEW] | venv 부트스트랩 + torch CPU 휠 + 빌더 실행 + 로그 |
| `05_training/requirements.txt` [NEW] | torch / torch_geometric / pandas / sqlalchemy / psycopg2 |

#### 2. Materialization 실측 (전체 13,803.4 s ≈ 3h 50m)

| 단계 | 시간 | 메모 |
|------|------|------|
| MAT-0 drop (IF EXISTS) | 0.01 s | 첫 실행이라 NOTICE skip |
| **MAT-1 CTAS (20.29M rows)** | **5,549.8 s (1h 32m 29s)** | 사전 추정 ~2h 보다 빠름 |
| MAT-2 row count | 1,404.3 s (23m 24s) | cold cache full scan |
| MAT-3 idx (state_ts, node_index) | 484.9 s (8m 4s) | 학습 hot path 복합 인덱스 |
| MAT-3 idx (service_date) | 306.6 s (5m 6s) | split 필터용 |
| MAT-3 idx (node_index) | 285.2 s (4m 45s) | per-node 분석용 |
| MAT-4 VACUUM ANALYZE | 900.4 s (15m) | planner 통계 |
| MAT-5 parity check | 4,872.1 s (1h 21m 12s) | view 측 쿼리가 재전개 (§5 발견) |

**parity_check 결과**:

```
 mat_rows | view_total_rows | mat_snapshots | view_snapshots | parity_check
----------+-----------------+---------------+----------------+--------------
 20289600 |        20289600 |          6570 |           6570 | OK
```

전날 readiness 의 V4/V5 수치 (`total_training_rows=20,289,600`, `distinct_snapshots=6,570`)
와 **정확히 일치**. 두 뷰 (`gatv2_snapshot_summary` ↔ mat table) 간 정합 확인.

#### 3. 인덱스 설계

학습 시 PyG DataLoader 가 snapshot 단위로 읽는 패턴 (`WHERE state_ts = ?`) 에
맞춰 `(state_ts, node_index)` 복합 인덱스를 hot path 로 지정. 부수적으로
`service_date` (split 필터), `node_index` (per-node 분석) 보조 인덱스 추가.
이후 모든 학습 쿼리는 btree seek 로 즉답 (수백 ms 이내 예상).

#### 4. PyG Dataset Builder 설계 — `build_gatv2_dataset.py`

contract `pytorch_geometric_dataset_contract.md` §10 invariant C1~C8 을 코드
레벨에서 `assert` 로 강제. 주요 함수:

| 함수 | 역할 |
|------|------|
| `_resolve_training_source(engine, prefer_mat)` | `_mat` 테이블 존재 시 자동 선택, 없으면 view fallback |
| `load_static_graph(engine)` | (num_nodes, edge_index, edge_attr, nodes_df) 반환. C3/C4 assert |
| `load_snapshot_list(engine, source, cfg)` | state_ts 그룹화. `--max-snapshots` smoke cap 지원 |
| `load_snapshot_rows(engine, source, state_ts)` | snapshot 단위 dataframe |
| `build_snapshot_data(df_ts, num_nodes, state_ts)` | x/y/node_mask 패딩 + C5/C8 assert |
| `_split(service_date, cfg)` | train (01-10) / val (11) / test (12) date 기반 라우팅 |
| `run_build(cfg)` | 메인 루프. 30초마다 진행 로그, `.pt` 4개 + `build_report.json` 저장 |

CLI: `--db-url`, `--out-dir`, `--no-mat`, `--max-snapshots N`, `--dry-run`.
DB URL 기본값은 `PG*` 환경변수에서 조립.

#### 5. Smoke Test 결과 — 환경 + 데이터 경로 모두 OK

```
[env ] torch 2.5.1+cpu / pyg 2.6.1 / pandas 2.3.3
[cfg]  use_materialized=True, max_snapshots=8, dry_run=True
[db]   snapshot_summary: distinct_snapshots=6570, total_training_rows=20,289,600
[db]   edge_summary    : num_nodes=4116, num_edges=5484
[graph] num_nodes=4116  num_edges=5484
[DONE] dataset build  (1665.4s)
```

| 검증 항목 | 측정값 | 기대 | 판정 |
|----------|-------|-----|------|
| torch 설치 | 2.5.1+cpu | ≥2.1 | ✅ |
| torch_geometric 설치 | 2.6.1 | ≥2.4 | ✅ |
| pandas 설치 | 2.3.3 | ≥2.0 | ✅ |
| sqlalchemy/psycopg2 import | OK | OK | ✅ |
| mat table 자동 감지 | True | True | ✅ |
| num_nodes (graph) | 4116 | 4116 | ✅ |
| num_edges (graph) | 5484 | 5484 | ✅ |
| C1 x.shape == (4116, 10) | OK | OK | ✅ |
| C2 y.shape == (4116, 3) | OK | OK | ✅ |
| C3 edge_index.max() == 4115 | OK | OK | ✅ |
| C4 edge_attr.shape[1] == 4 | OK | OK | ✅ |
| C5 node_mask.sum() == rows | OK | OK | ✅ |
| C8 isfinite(x) / isfinite(y) | OK | OK | ✅ |

1,665.4 s 중 대부분은 **venv 부트스트랩 + torch CPU 휠 다운로드** 일회성 비용.
이후 실행부터는 venv 재사용으로 즉시 시작.

#### 6. 숨은 발견

##### §6-A. PostgreSQL 18 + PowerShell 5.1 NOTICE 표시 잡음

psql 18.3 의 `DROP TABLE IF EXISTS` NOTICE 가 PS 5.1 콘솔에서 `NativeCommandError`
빨간 블록으로 렌더링됨. `$ErrorActionPreference = "Continue"` 가 걸려 있어
**실행은 계속됨** (실측: MAT-0 NOTICE 발생 → MAT-1 CTAS 1h 32m 정상 수행 →
MAT-5 parity OK 까지 자동 완료). 콘솔 빨간색 = 시각적 잡음, 로그 파일에는
정상 텍스트로 저장됨. 향후 동일 패턴 발생 시 무시 가능.

##### §6-B. parity check 가 1h 21m 걸린 이유

MAT-5 의 `(SELECT total_training_rows FROM public.gatv2_snapshot_summary)` 가
**원본 view chain 을 재전개** 하여 20.29M row CTE 를 다시 돌림. mat table
직접 쿼리는 인덱스로 즉답이지만, 비교 대상이 view 측이라 한 번 더 풀스캔.
**학습 경로에서는 발생하지 않음** — `build_gatv2_dataset.py` 는 mat table 만
참조함.

##### §6-C. PowerShell 5.1 CP949 vs UTF-8 호환

전날 ASCII-only 로 정비한 PS1 두 개 (`run_materialize.ps1`, `run_build_dataset.ps1`)
가 인코딩 이슈 없이 실행됨. 향후 `05_training/` 의 모든 PS1 은 ASCII-only 정책
유지. 한글 메시지는 `.md` / `.sql` (UTF-8 처리 가능) 에서만 사용.

#### 7. 다음 단계 — 풀 빌드 → GATv2 학습 코드

| # | 과제 | 비용 | 블로킹? |
|---|------|------|---------|
| 1 | `run_build_dataset.ps1` 풀 실행 (smoke 인자 제거) | 예상 ~1-2h | YES (학습용 .pt 4개 생성) |
| 2 | `build_report.json` 검증 (C6/C7 합산 일치 확인) | 즉시 | YES |
| 3 | `05_training/gatv2_model.py` GATv2 인코더 정의 | - | NO |
| 4 | `05_training/train_gatv2.py` 학습 루프 (MSE in log space) | - | NO |
| 5 | `05_training/eval_gatv2.py` 평가 (expm1 복원 + RMSE/MAE) | - | NO |

#### 8. 부산물 / 정리

- materialize SQL / runner 둘 다 `DROP TABLE IF EXISTS` 시작이라 **재실행
  완전 안전** (idempotent). 단 재실행 시 또 ~3h 50m 소요.
- `05_training/.venv/` 가 생성됨. `.gitignore` 미포함 시 추가 권장 (대용량 venv).
- `05_training/data/gatv2_dataset/` 디렉터리 (out-dir) 는 빌드 시 자동 생성.
  `.pt` 파일들도 `.gitignore` 권장 (수백 MB 단위 예상).
- snapshot 별 진행은 `logs_build_<timestamp>/build.log` 에 30초마다 기록되므로
  중간 실패 시 어느 state_ts 에서 멈췄는지 즉시 파악 가능.

---

### 누적 진척 매트릭스 갱신 (2026-04-20 23:59 기준)

| 레이어 | 상태 |
|-------|------|
| graph_state_timeslice | APPROVED |
| rl_state_training_base | APPROVED |
| rl_stop_transition_features | APPROVED |
| graph_edge_master STOP_TO_STOP v2 | APPROVED |
| GATv2 5-view layer | APPROVED |
| **gatv2_snapshot_stop_features_train_mat** | **APPROVED (parity OK)** |
| **build_gatv2_dataset.py** | **smoke test PASSED** |
| GATv2 model / train / eval | TODO |

GATv2 학습 코드 작성으로 즉시 진행 가능.
