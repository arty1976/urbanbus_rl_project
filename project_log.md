## 📅 2026-04-26 - Phase 2 causal simulator skeleton, fleet sensitivity, extended KPI 분석 체계 확정

### 1. Phase 2 causal simulator 구조 확장
Phase 1 full-year replay-backed canonical validation 통과 이후, Phase 2에서는 `HistoricalReplayAdapter` 기반 non-causal replay 검증과 별도로 정책 action이 다음 상태에 영향을 주는 `CausalSimulatorAdapter` 경로를 확장하였다.

- `B0_historical_observed`는 실제 관측 기반 non-causal reference로 보존.
- `B0R_current_ops_reconstructed`를 새로 분리하여 현재 고정노선 운영을 causal simulator 안에서 재구성하는 기준선으로 정의.
- B0R/B1/B2/A 계열 모두 `causal_comparison_allowed=true`가 되도록 canonical output 경로 검증.
- `run_causal_rollout.py`와 `canonical_kpi_aggregator.py`를 통해 causal rollout -> window_rollup -> canonical_eval 경로를 유지.

### 2. Bunching 판정 기준 개선
기존 고정 180초 기준만으로는 night/offpeak/peak의 target headway 차이를 반영하기 어려워 다음 기준으로 개선하였다.

```text
effective_bunching_threshold = max(180, target_headway_seconds * 0.75)
```

Step 13 결과에서 B1의 bunching_rate는 night < offpeak < peak 순으로 증가했고, B2는 B1 대비 bunching_rate를 낮추는 방향으로 작동하였다. 이는 target-headway-ratio 기반 bunching policy가 조건 차이에 민감하게 반응함을 의미한다.

### 3. 2023 observed 기반 empirical shared demand profile 적용
임시 수요 배율을 사용하지 않고 2023년 대구 observed feature table에서 직접 time_band별 empirical demand profile을 산출하였다.

- source relation: `public.gatv2_snapshot_stop_features_train_mat`
- 사용 컬럼: `boardings_recent_log`, `waiting_passenger_cnt_log`, `alightings_recent_log`
- demand_intensity 계산식:

```text
demand_intensity =
  0.6 * mean_log1p_boardings_recent
+ 0.4 * mean_log1p_waiting_passenger_cnt
```

산출된 arrival_multiplier_by_time_band는 다음과 같다.

| time_band | demand_intensity | multiplier_vs_night |
|---|---:|---:|
| night | 1.259334 | 1.000000 |
| offpeak | 0.962699 | 0.764451 |
| peak | 1.231103 | 0.977582 |

time_band의 KST 기준 정의도 검증하였다.

| time_band | KST 기준 시간 |
|---|---|
| peak | 07, 08, 09, 17, 18, 19 |
| offpeak | 06, 10, 11, 12, 13, 14, 15, 16 |
| night | 05, 20, 21, 22 |

이 profile은 `shared_exogenous_demand_profile_v1.yaml`로 저장되었고, 모든 causal 조건이 같은 수요 profile을 사용하도록 연결하였다.

### 4. Fleet sensitivity 정책 도입
A_pure_mappo가 B0R보다 모든 KPI를 압도해야 한다는 가정 대신, 다음 연구 가설을 실험 계약에 반영하였다.

> A_pure_mappo는 B0R 대비 on_time_rate가 약간 낮더라도, 더 적은 active_bus_count와 낮은 energy_proxy로 유사한 passenger_service_rate를 유지할 수 있다면 성공으로 볼 수 있다.

이를 위해 A 계열 fleet variants를 정의하였다.

| 조건 | 의미 | fleet_ratio_vs_b0r |
|---|---|---:|
| A | A_pure_mappo_fleet_100 | 1.00 |
| A90 | A_pure_mappo_fleet_90 | 0.90 |
| A80 | A_pure_mappo_fleet_80 | 0.80 |
| A70 | A_pure_mappo_fleet_70 | 0.70 |

성공 조건은 다음 방향으로 정의하였다.

- avg_wait_seconds <= 1.10 × B0R
- passenger_service_rate >= 0.98 × B0R
- on_time_rate drop <= 10 percentage points
- passenger_wait_p95_seconds <= 1.20 × B0R
- bunching_rate increase <= 5 percentage points
- energy_proxy는 B0R보다 감소해야 함
- A90/A80/A70은 실제 fleet reduction을 달성해야 함

### 5. Extended KPI 산출 경로 추가
기존 shared KPI 6개는 유지하고, fleet 및 passenger service quality를 해석하기 위한 extended KPI를 추가하였다.

기존 shared KPI:
- cv_headway
- avg_wait_seconds
- bunching_rate
- on_time_rate
- intervention_rate
- energy_proxy

추가 extended KPI:
- active_bus_count
- base_num_agents_b0r
- fleet_ratio_vs_b0r
- fleet_reduction_ratio
- passengers_served
- passenger_demand_generated
- passenger_service_rate
- passenger_wait_p95_seconds
- energy_proxy_per_passenger
- intervention_events

`canonical_kpi_aggregator.py`는 optional extended KPI를 보존하도록 수정되었다.

### 6. Shared demand generation fairness 수정
Step 18에서 A90/A80/A70의 active_bus_count가 줄면서 `passenger_demand_generated`도 함께 줄어드는 문제가 발견되었다. 이는 fleet reduction 실험에서 공정성 위반이므로 Step 19에서 수정하였다.

수정 후 같은 seed/window/time_band에서는 B0R/B1/B2/A/A90/A80/A70 모두 동일한 generated demand를 받는다.

| window_id | time_band | passenger_demand_generated | label_count |
|---|---|---:|---:|
| 1 | night | 520.0 | 7 |
| 2 | offpeak | 506.0 | 7 |
| 3 | peak | 523.0 | 7 |

이제 fleet-reduced variants는 같은 수요를 받으면서 더 적은 active_bus_count로 대응하므로, passenger_service_rate와 wait-time KPI를 공정하게 해석할 수 있다.

### 7. Step 20 KPI relationship analyzer 생성
`05_training/evaluation/analyze_kpi_relationships.py`를 생성하여 다음 분석 결과를 산출하였다.

- KPI label summary
- Pearson correlation matrix
- Spearman correlation matrix
- strong Spearman correlations
- B0R-relative service constraints
- Pareto frontier
- markdown report
- summary JSON

생성 파일:
- `artifacts/baseline_v2_causal/analysis/kpi_label_summary.csv`
- `artifacts/baseline_v2_causal/analysis/kpi_correlation_pearson.csv`
- `artifacts/baseline_v2_causal/analysis/kpi_correlation_spearman.csv`
- `artifacts/baseline_v2_causal/analysis/kpi_strong_spearman_correlations.csv`
- `artifacts/baseline_v2_causal/analysis/b0r_relative_service_constraints.csv`
- `artifacts/baseline_v2_causal/analysis/kpi_pareto_frontier.csv`
- `artifacts/baseline_v2_causal/analysis/kpi_relationship_report.md`
- `artifacts/baseline_v2_causal/analysis/kpi_relationship_analysis_summary.json`

### 8. Step 20 주요 해석
Step 20 label summary에서 B0R은 서비스 품질이 가장 높고, A는 B0R 대비 energy_proxy와 energy_proxy_per_passenger를 줄이지만 avg_wait_seconds와 p95 wait가 증가하였다.

| label | avg_wait_seconds | p95_wait | service_rate | on_time_rate | energy_proxy | fleet_reduction |
|---|---:|---:|---:|---:|---:|---:|
| B0R | 4.021270 | 3.107143 | 1.000000 | 1.000000 | 480.0 | 0.0 |
| A | 5.701756 | 49.118880 | 0.999044 | 0.928889 | 390.0 | 0.0 |
| A90 | 24.315487 | 82.747620 | 0.927327 | 0.925926 | 351.0 | 0.1 |
| A80 | 41.812523 | 372.750000 | 0.868431 | 0.931944 | 312.0 | 0.2 |
| A70 | 60.062176 | 571.666667 | 0.810445 | 0.923810 | 273.0 | 0.3 |
| B1 | 31.238253 | 133.822220 | 0.991595 | 0.436667 | 300.0 | 0.0 |
| B2 | 68.945827 | 1234.000000 | 0.894052 | 0.858889 | 571.0 | 0.0 |

B0R-relative constraint evaluation에서는 모든 조건이 `success_under_fleet_policy=False`로 나타났다. 현재 A 계열은 아직 placeholder policy이며 학습된 MAPPO 결과가 아니므로 성능 결론으로 해석하지 않는다. 다만 실험 구조가 A 정책이 해결해야 할 trade-off를 명확히 드러낸다.

### 9. Qwen 역할 경계
KPI relationship analyzer와 Qwen의 역할은 분리한다.

- KPI analyzer: deterministic numeric analysis 도구. 공식 KPI 수치, 상관관계, Pareto frontier, B0R-relative constraint를 계산한다.
- Qwen: analyzer 결과를 읽고 원인 가설, reward shaping 후보, 다음 ablation 설계를 제안하는 해석자 역할을 수행한다.
- Qwen은 공식 KPI 값을 수정하거나 사후 tuning을 수행하지 않는다.
- A_pure_mappo 조건에서는 Qwen 개입이 없어야 하며, Qwen 개입 실험은 별도 D 조건에서만 허용한다.

### 10. 다음 단계
다음 작업은 MAPPO reward 설계와 학습 연결이다.

우선순위:
1. reward에 avg_wait, p95 wait, passenger_service_rate, energy_proxy, bunching/on_time penalty를 반영.
2. A/A90/A80/A70 placeholder를 실제 MAPPO policy output으로 교체.
3. 1 seed × 3 window smoke를 넘어 다중 seed 및 더 많은 windows로 확장.
4. analyzer 결과를 Qwen 해석 입력으로 사용하되, 공식 KPI 산출과 분리.
5. A90/A80/A70의 fleet reduction frontier를 정식 실험으로 평가.

중요 caveat:
현재 A/A90/A80/A70 결과는 학습된 MAPPO 성능 결과가 아니라 skeleton-stage placeholder 결과다. 따라서 논문에는 성능 결론이 아니라 실험 체계, 공정성 제약, KPI 분석 도구의 검증 결과로 기록한다.

---


# 🚌 UrbanBus RL Project - 작업 일지 (Project Journal)

이 문서는 프로젝트의 진행 상황을 일자별로 기록하는 통합 로그입니다.  
**"오늘 한 일 저장"** 요청 시 최신 날짜가 상단에 추가됩니다.

---
## 📅 2026-04-24
### Canonical KPI Aggregator 배선 완료 및 B1 Replay-backed Rollout 성공 (Phase 1)

오늘 작업에서는 B1 No-op baseline의 실행기를 스텁에서 실제 리플레이 기반으로 확장하고, B0/B1/B2/[A] 전 조건이 공통 KPI 집계 경로에 연결되도록 canonical aggregator 파이프라인을 정비했다. 특히 B1은 이제 단순 스텁이 아닌 어댑터 기반의 실제 롤아웃 데이터를 생성한다.

#### 1. B1 No-op Rollout Writer 구현 및 실행 완료 (Phase 1)
- **파일**: `05_training/run_b1_noop_rollout.py` 고도화 완료.
- **기능**: `HistoricalReplayAdapter`를 로드하여 각 시나리오 윈도우에 대해 `reset()` -> `step()` 과정을 수행하고 실측 데이터를 모사한 아티팩트를 생성함.
- **산출물**: 
  - `raw_events.parquet`: 에이전트별 행동 및 보상 기록 (비인과 리플레이).
  - `window_rollup.parquet`: 표준 집계기용 KPI 요약 데이터.
  - `status.json` / `run_manifest.json`: 비인과성 경고 및 메타데이터 포함.
- **검증**: Seed 001에 대해 128개 윈도우 롤아웃 실행 및 `canonical_kpi_aggregator.py` 통과 확인 (**Real-ish PASS**).

#### 2. Simulator Adapter 인터페이스 확정 및 Phase 1 Smoke Validation 체계 구축
- **Simulator Adapter 인터페이스 설계**: `05_training/simulator_adapter_interface.py` 정의. CTDE 기반 `ObsDict`, Multi-Agent `StepResult`, `GraphSkeleton` 확장 포함.
- **HistoricalReplayAdapter 구현**: Phase 1 전용 비인과적(Non-causal) 리플레이 어댑터 구축 및 smoke validation 용도 제한 명시.
- **MAPPO Runner 고도화**: 어댑터 동적 로드 및 `reset()`, `get_graph_skeleton()` 호출 실측 기능 보강.

#### 3. canonical_kpi_aggregator.py 실파일 생성 및 실행 경로 정착
- 생성 파일: `05_training/evaluation/canonical_kpi_aggregator.py`
- 실행 모드 2종 검증: `legacy_b0_passthrough`, `official_rollup`.
- 공통 출력 5종(`kpi_by_window.parquet` 등) 생성 확인.

#### 4. 현재 판정 및 제약 사항
- **상태**: 
  - B0: **completed** (Legacy Pass)
  - B1: **completed** (Real-ish Pass via Historical Replay)
  - B2/A: **smoke PASS** (via Stub rollup)
- **한계**: 현재 모든 PASS는 평가 계약 및 파일 경로 규약의 정합성 검증 성공을 의미하며, **비인과적 리플레이 기반**이므로 실제 성능 비교 자료로 사용 불가.

#### 5. 다음 단계 (Next Steps)
- B2 rule-based real rollout 실행기(Runner) 작성 및 `HistoricalReplayAdapter` 연결.
- [A] MAPPO runner가 실제 `window_rollup.parquet`를 생성하도록 어댑터 연결.
- 모든 조건이 리플레이 기반 실측 데이터를 생성하면 전체 파이프라인을 **Real-ish PASS**로 승격.
- Phase 2 인과 시뮬레이터(Causal Simulator) 어댑터 준비.

---
## 📅 2026-04-23
### B0/B1/B2 Baseline 완성 및 [A] MAPPO 학습 골격(Stub) 구축

오늘 작업에서는 강화학습 시뮬레이션을 위한 기준선(Baseline) 프레임워크를 정립하고, B0(Historical) 지표를 데이터베이스 및 Parquet 형태로 확보했습니다. 더불어 시뮬레이터가 부재한 현재 상태를 반영해 B1(No-op) 및 B2(Rule-based)의 실행 전 메타데이터를 마련했으며, 최종적으로 [A] 순수 MAPPO 모델의 러너(Runner) 골격을 구성했습니다.

#### 1. B0 Historical Baseline 완료
- **생성 완료**: `public.baseline_b0_historical_kpi_by_window`, `public.baseline_b0_historical_kpi_metadata`
- **6개 KPI 계약 컬럼 고정**: `cv_headway`, `avg_wait_seconds`, `bunching_rate`, `on_time_rate`, `intervention_rate`, `energy_proxy`
- **데이터 상태**:
  - 채워진 값: `avg_wait_seconds`, `intervention_rate`
  - NULL 유지: `cv_headway`, `bunching_rate`, `on_time_rate`, `energy_proxy`
- **검증 결과**:
  - `row_count = 6570`
  - `min_state_ts = 2023-01-01 05:00:00+09`
  - `max_state_ts = 2023-12-31 22:00:00+09`
  - `wait_filled_rows = 6570`
  - `cv_headway_null_rows = 6570`
  - `zero_intervention_rows = 6570`

#### 2. B0 Artifact Export 완료
- **생성 산출물**:
  - `artifacts/baseline_v1/B0_historical/kpi_by_window.parquet`
  - `artifacts/baseline_v1/B0_historical/metadata.json`

#### 3. B1 No-op Baseline 준비 완료
- **생성 산출물**:
  - `artifacts/baseline_v1/B1_noop/scenario_index.parquet`
  - `artifacts/baseline_v1/B1_noop/policy_config.json`
  - `artifacts/baseline_v1/B1_noop/rollouts/seed_001` ~ `seed_003` (내부에 `run_manifest.json` 생성 완료)
  - `run_b1_noop_rollout.py` 골격 생성 완료
- **현재 상태**: `prepared_not_executed` (사유: replay simulator adapter 부재)

#### 4. B2 Rule-based Baseline 준비 완료
- **생성 산출물**:
  - `artifacts/baseline_v1/B2_rulebased/scenario_index.parquet`
  - `artifacts/baseline_v1/B2_rulebased/policy_config.json`
  - `artifacts/baseline_v1/B2_rulebased/rollouts/seed_001` ~ `seed_003`
- **현재 상태**: `prepared_not_executed` (사유: replay simulator adapter 부재)
- **B2 Rule Params 확정**:
  - `target_headway_seconds = 600`
  - `low_headway_threshold_seconds = 360`
  - `high_headway_threshold_seconds = 900`
  - `max_hold_seconds = 120`
  - `allow_skip = true`

#### 5. baseline_contract.json 확정
- **생성 산출물**: `artifacts/baseline_v1/baseline_contract.json`
- **공통 계약**:
  - `evaluation_horizon_minutes = 30`
  - `seeds = [1, 2, 3]`
  - `time_bands = [peak, offpeak, night]`
  - `shared_kpis` = 6종 고정
  - `fairness_constraints`:
    - `same_initial_state = true`
    - `same_exogenous_events = true`
    - `same_eval_window = true`
- **Baseline 상태 스냅샷**: `B0 = completed`, `B1 = prepared_not_executed`, `B2 = prepared_not_executed`

#### 6. [A] pure_mappo_baseline 계약 연결 완료
- **조건 명세**:
  - `condition_id = A`
  - `qwen_train = false`
  - `qwen_inference = false`
- **생성 산출물**: `experiment_A_contract.json`
- **현재 상태**: `contract_linked_not_trained`

#### 7. MAPPO 러너 골격(Stub) 생성 완료
- **생성 파일**:
  - `05_training/mappo_runner.py`
  - `05_training/run_experiment_A_stub.py`
  - `05_training/policies/mappo_policy_stub.py`
- **Seed별 실행 디렉터리 및 산출물**:
  - `artifacts/experiment_A_v1/runs/seed_001` ~ `seed_003` 생성 완료
  - 각 seed별 `run_manifest.json`, `status.json`, `checkpoint_stub.json` 산출물 포함
- **현재 상태**: `status: "adapter_missing"` (사유: simulator adapter 미구현이므로 정상 동작임)

#### 8. MAPPO 러너 설계 체크리스트 확정
- CTDE(Centralized Training Decentralized Execution=중앙집중 학습 분산 실행) 구조 분리
- PyG(PyTorch Geometric=파이토치 지오메트릭) Batch.from_data_list 그래프 배치
- 활성 버스 마스킹
- edge_index 롤아웃 버퍼 제외
- GAE(Generalized Advantage Estimation=일반화 이점 추정)에서 terminated / truncated 분리
- 첨두 적응형 엔트로피 계수
- grad_norm_clip = 0.5
- rng_state 포함 체크포인트
- qwen_trigger_rate 로깅, 단 [A]에서는 0.0 강제
- shared_policy = true 기본값
- GATv2 freeze → 점진 해제 3단계
- reward normalization
- KL divergence monitoring + early stopping
- H200 multi-GPU는 현재 인터페이스만 설계, 본구현은 보류

#### 9. Troubleshooting & Today Notes
- **Windows PowerShell(PowerShell=마이크로소프트 명령행 셸) 붙여넣기형 patch workflow 정착**: 직접 수정 대신 스크립트를 통한 텍스트 조작 파이프라인 안착.
- **psql.exe 탐색 및 PATH(Path=실행 경로 환경변수) 이슈 해결**: PATH 미인식 문제를 자동 탐색 스크립트로 해결하고, DB(Database=데이터베이스) 사용자 `ryujo` 인증 실패를 `PGUSER`/`PGPASSWORD` 환경변수 방식으로 복구.
- **SQL 데이터 타입 오류 수정**: `integer` vs `boolean`의 `COALESCE` 오류 수정.
- **인코딩 & 파일 시스템 이슈 회피**: 
  - `preview.sql` UTF-8 BOM(Byte Order Mark=문자 인코딩 표시 바이트) 문제 확인 및 우회.
  - config 디렉터리 없음으로 인한 YAML 생성 실패 복구.
  - JSON UTF-8 BOM 에러 발생 시 `utf-8-sig` 읽기 옵션 적용.
  - PowerShell 내 here-string 중첩 시 내부 파이썬/Bash 변수가 null 로 평가되는 문제 파악 및 회피.

#### 10. 다음 단계 (Next Steps)
- `simulator_adapter_interface.py` 골격 생성
- replay simulator adapter 명세 확정
- B1/B2 real rollout 실행기 연결
- [A] 실제 MAPPO train/eval runner 확장

---
## 📅 2026-04-22
### GATv2 Training Smoke Test Binding 및 신규 스킬 제정

- dataset_full_20260422_084243 successfully bound to train_gatv2.py
- Samsung Galaxy Book 5 Pro CPU-based training smoke test PASS
- 128 files / batch_size 1 / 3 epochs stability test PASS
- node-level GATv2 forward, loss, backward, optimizer step verified
- laptop is sufficient for pipeline validation, but full-scale training should be migrated to H200 server
- **새 스킬 `gatv2_training_smoke_test_binding` 추가됨** (`05_training/skills/gatv2_training_smoke_test_binding.md`)

---
## 📅 2026-04-21
### GATv2 전체 데이터셋 빌드 성공 및 파이프라인 승인 [APPROVED]

GATv2 학습을 위한 1년 치(2023년) 전수 데이터셋 빌드를 완료하고, 생성된 아티팩트의 무결성을 최종 승인함.

#### 1. 주요 성과
- **run_build_dataset.ps1** full dataset save PASS
- **snap 6570/6570** completed (1년 치 전체 타임슬라이스 변환 완료)
- **build_report.json** generated successfully
- **full dataset build** completed in 4470.9s (74m 31s)
- **end-to-end artifact generation pipeline** approved

#### 2. 다음 단계 (Next Step)
- `dataset_full` 출력을 `train_gatv2.py`에 바인딩하고 학습 smoke test 실행

---
## 📅 2026-04-20
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

### RL Baseline & Rollout 진척도 (2026-04-24 기준)

| 레이어 / 태스크 | 상태 |
|-------|------|
| simulator_adapter_interface | **APPROVED** |
| HistoricalReplayAdapter | **APPROVED (Phase 1)** |
| B0 Historical Baseline | **COMPLETED (Legacy Pass)** |
| B1 No-op Baseline (Replay) | **COMPLETED (Real-ish Pass)** |
| B2 Rule-based Baseline | Smoke PASS (Stub) |
| [A] Pure MAPPO Baseline | Smoke PASS (Stub) |
| Canonical KPI Aggregator | **APPROVED (official_rollup)** |

---
## 📅 2026-04-19
### graph_edge_master STOP_TO_STOP 복원 — leg-aggregation 도입

`graph_edge_master` 의 `STOP_TO_STOP` 엣지를 `route_link_sequence` + `dim_stop`
로부터 재생성하는 패키지 (`graph_edge_master_pkg/`) 를 v2 로 고도화하여,
coverage 22.84% → **99.91%** (leg 기준) 로 복구.

#### 1. v1 관찰 — 초기 22.84% 커버리지 ([수용 불가])
- `01_alter_graph_edge_master.sql` / `02_load_...sql` / `03_..._readiness.sql`
  3-단계 파이프라인을 first pass 로 실행.
- 구조 지표(R3~R6) 는 전부 0 으로 깨끗했으나 **R10 coverage_pct = 22.84%**.
- 동시에 R5 에서 `ON CONFLICT DO UPDATE` 를 같은 edge_uid 로 두 번 건드려
  `21000` 에러 → STAGE 4.5 `tmp_edge_dedup (row_number() over edge_uid)` 추가로 해결.
- 이 시점엔 coverage 저하를 "dim_stop 매핑 부재" 로 가정함.

#### 2. 진단 3단계 (D1 / D2 / D3) — 가설 A, B 기각 → 가설 D 확정
- **D1** (`D_diagnose_stop_id_mapping.sql`): missing_id 1,482개, 전부 길이 10,
  숫자 prefix `15xxx / 30007xxx / 73611xxx…` 로 분포. `dim_stop` 은 7개 컬럼
  뿐이라 alt-key 매핑 여지 없음 → **가설 A (다른 키 사용) 기각**.
- **D2** (`D2_probe_mapping_tables.sql`): 5개 후보 브리지/staging 테이블
  (`stop_link_mapping_master`, `stg_daegu_stops_geo`, `bs_20250903`,
  `graph_node_master`, `err_daegu_stop_usage_mapping_failed`) × 모든 text/numeric
  컬럼에 대해 dynamic `EXECUTE` 로 매칭 수 측정 → **전부 0 매치**. 브리지
  테이블 가설 B 완전 기각.
- **D3** (`D3_link_vs_stop_hypothesis.sql`): 핫스팟 노선 `7361109008:dir=1`
  의 link 시퀀스를 `S→-, -→S, S→-, -→-` 패턴으로 태그해보니 **교차 패턴** 선명.
  전체 분포 `S→S 32.41% / S→- 32.39% / -→S 23.01% / -→- 12.19%`.
  노선당 `links_per_stop = 1.84` (avg) / 최대 수 십 단위. → **가설 D
  (route_link_sequence 는 stop + 도로/교차로 intermediate node 혼합 시퀀스)
  확정.**

#### 3. 02_load v2 — leg-aggregation 알고리즘
가설 D 에 맞춰 `02_load_...sql` 을 9-stage 파이프라인으로 전면 재작성:

```
STAGE 1  tmp_rls_tagged    : link 마다 (st_is_stop, ed_is_stop) 플래그
STAGE 2  tmp_rls_legged    : st_is_stop=TRUE 마다 leg_id 증가 (window sum)
STAGE 3  tmp_leg_edges     : leg 단위 src/dst stop, cum_gis_dist 집계
STAGE 4  tmp_edge_src      : dim_stop.geom_5187 join, distance_m/time_sec 계산
STAGE 5  tmp_edge_dedup    : edge_uid 단위 가장 이른 leg 1건만 유지
STAGE 6  tmp_edge_ranked   : (src,dst) 단위 edge_rank (distance_m ASC)
STAGE 7  INSERT ... ON CONFLICT DO UPDATE
STAGE 8  UPDATE ... SET is_active=false  (stale 처리, DELETE 아님)
STAGE 9  AFTER 스냅샷 NOTICE
```

- **핵심**: leg = "한 stop 출발 ~ 다음 stop 도착 직전" 까지의 모든 link.
  중간 intermediate node 는 `graph_edge_master` 에 저장하지 않음 (stop-to-stop
  directed edge only 원칙 유지).
- **거리 계산**: `coalesce(nullif(cum_gis_dist,0), ST_Distance(geom_5187))` —
  누적 실제 링크 거리 우선, 0/NULL 이면 직선거리 fallback.
- **시간 계산**: `distance_m / 1000.0 / 20.0 * 3600.0` (평균 20 km/h 가정).

#### 4. 03_readiness 의미론 수정
- **R10** 분모를 "source link rows" → "leg-aggregation distinct legs" 로 변경.
  `source_legs_total / source_legs_valid / source_legs_distinct / loaded_edges /
  missing / coverage_pct` 6-컬럼 출력.
- **R11** 의미를 "매칭 실패 stop_id" → "route_link_sequence 의 non-stop
  node_id 샘플 (의도된 intermediate node — `dim_stop` 에 없는 게 정상)" 으로 수정.

#### 5. v2 Readiness 결과 ([APPROVED])

| ID | 지표 | 측정값 | 판정 |
|---|---|---|---|
| R0 | stop_to_stop_total | **21,466** | ✅ |
| R0 | primary_edges | 5,484 | ✅ |
| R0 | inactive_edges | 0 | ✅ |
| R3 | self_loop_cnt | **0** | ✅ |
| R4 | orphan_src / orphan_dst | **0 / 0** | ✅ |
| R5 | duplicate_primary_cnt | **0** | ✅ |
| R6 | null_distance / null_time / zero_distance | **0 / 0 / 0** | ✅ |
| R7 | min_dist_m / max_dist_m | 6.88 / 17,438.65 | ✅ (< 20 km) |
| R7 | avg_dist_m / p50 / p95 | 494.7 / 381.7 / 1,234.9 | ✅ |
| R7 | over_5km / over_20km | 27 / 0 | ✅ |
| R8 | avg_time_sec / p50_time_sec | 89.0 / 68.7 | ✅ (1.48 min / 1.15 min) |
| R10 | **source_legs_distinct** | **21,485** | |
| R10 | **loaded_edges** | **21,466** | |
| R10 | **coverage_pct** | **99.91%** | ✅ |

- v1 22.84% → v2 **99.91%** — 4.37× 상승.
- 미적재 19건 (99.91% ↔ 100%) 은 `tmp_edge_src` STAGE 4 의 `dim_stop.geom_5187
  is not null` inner-join 조건에서 탈락한 지오메트리-결측 stop 으로 추정 (총
  21,485 legs 기준 0.09%). 별도 데이터 품질 티켓으로 분리해 추적.
- R11 샘플이 전부 `1500xxx` prefix — 예상대로 도로/교차로 node (대구시 `link` /
  `node` Shapefile 계열). 더 이상 "매칭 실패" 로 취급하지 않음.

#### 6. 부산물 / 정리
- **유지**: `graph_edge_master` 스키마 (`distance_m`, `time_sec`,
  `is_primary_edge`, `edge_rank`) 전혀 변경 없음. GATv2 static graph 표준을
  깨지 않음.
- **추가된 문서**: `graph_edge_master_pkg/D_diagnose_stop_id_mapping.sql`,
  `D2_probe_mapping_tables.sql`, `D3_link_vs_stop_hypothesis.sql`,
  `run_diagnose.ps1`.
- **README §7 changelog** 에 "02_load v2 배포" 항목 기록.
- PowerShell NativeCommandError (`psql` 의 RAISE NOTICE 가 stderr 로 나와
  `$ErrorActionPreference=Stop` 정책에서 전체 스크립트 중단) → 러너
  (`run_graph_edge_master.ps1`, `run_diagnose.ps1`) 에 `$ErrorActionPreference =
  "Continue"` 고정 + `$LASTEXITCODE` 로 성패 판정으로 교정.

#### 7. 다음 단계 (블로킹 해제됨)
- `graph_state_timeslice` 에 `node_uid = 'STOP:<stop_id>'` 기준 state features
  주입 — 이번에 복원한 STOP_TO_STOP 엣지가 GATv2 message-passing 구조의
  static skeleton 으로 사용됨.
- 19건 지오메트리-결측 stop 은 `dim_stop` 데이터 품질 후속 티켓으로.
- 여기까지 완료되면 비로소 **GATv2 모델 코드 작성 단계** 로 이행.

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


## 📅 2026-04-18
### RL 상태 전이 특성(Stop Transition Features) 및 SAR 명세 확정

프로젝트는 Phase 5의 물리적 데이터(t, t+1 페어링)를 넘어서, 실제 강화학습 모델에 주입할 **상태(State), 행동(Action), 보상(Reward) 명세**를 고정하고 이를 생성하는 파이프라인을 구축하였다.

#### 1. SAR 명세 고정 (`rl_sar_spec.md` [NEW])
- **상태(State)**: 정류소 단위 수요 문맥으로 정의. `waiting_passenger_cnt`를 가장 중요한 운영 압력 신호(Primary Pressure Signal)로 설정.
- **행동(Action)**: 버스 에이전트의 차기 목표 정류소 선택(Target node selection)으로 개념 고정.
- **보상(Reward)**: 대기 승객 패널티를 강화한 대리 보상(Proxy Reward) 설계.
- **학습 단계**: 현재를 'Graph-ready transition learning stage'로 규정 (전이 모델링 기반).

#### 2. 모델용 피처 뷰 생성 (`02_ingest_jobs/create_rl_stop_transition_features.sql` [NEW])
- **데이터 변환**: 수요 데이터의 로그 변환(`ln(1+x)`), 시간/요일의 순환 인코딩(sin/cos) 적용.
- **RL 에피소드 관리**: 야간 공백(4시간 이상)을 인지하여 `is_terminal_transition` 및 `is_overnight_gap` 플래그 주입.
- **유효 시간 간격**: 야간 공백을 제외한 실제 운영 전이 시간(`delta_t_hr_effective`) 산출.

#### 3. 피처 무결성 검증 및 실행 (`03_validation_queries/rl_stop_transition_features_readiness.sql` [NEW])
- **최종 판정**: `final_decision = APPROVED` / `rl_stop_transition_features_ready = YES`
- **검증 항목**: Null safety, 음수 수요 차단, 시간 역전 감지, 전이 일관성 체크.
- **실행 및 시정 사항 (Remediation)**:
    - **인코딩 대응**: 윈도우 `psql` 환경의 UTF8 인코딩 미스매치 해결 (`SET client_encoding`).
    - **타입 캐스팅**: `is_peak` (boolean) 타입과 정수 비교 충돌을 해결하기 위한 명시적 캐스팅 적용.
    - **경계 조건 처리**: 2023-12-31(최종일)의 마지막 버킷이 소스 상태로 존재하지 않아 발생하는 버킷 수 부족(18 vs 19) 현상을 시스템 경계 조건으로 해석하여, 마지막 날에 한해 `n-1` 버킷을 허용하도록 검증 로직 보완.

#### 4. 현재 데이터 상태
- `row_count`: **21,612,482**
- `rl_usable_row_count`: **20,490,432** (야간 공백 및 터미널 전이 제외 가용 데이터)
- `active_days`: 365일
- `final_status`: **APPROVED**

---

## 📅 2026-04-15
### Phase 4 복구 완료 · Phase 5 입력 규격 확정 완료 — 이력 정리

이 항목은 2026-04-14 작업 세션에서 달성한 Phase 4 / Phase 5 성과를 **사실 기반으로 상세 기록**한 보완 로그이다.

#### 현재 프로젝트 상태 요약

Phase 4(`graph_state_timeslice` 적재)와 Phase 5(`rl_state_training_base` 생성) 모두 최종 APPROVED를 달성하였다.
대구 2023 observed 기반 시계열 상태 레이어(21.6M행)와 다음 관측 시점 기반 학습 베이스(21.6M행)가 확보되었으며,
프로젝트는 데이터 복구·적재 단계를 졸업하고 **모델링 단계**로 진입하였다.

다음 단계는 아래와 같다.

1. 상태(State)·행동(Action)·보상(Reward) 명세 고정
2. PyTorch 입력 데이터셋 연결
3. 그래프 구조(edge) 복구 및 GAT/RL 학습 파이프라인 확장

---

## 📅 2026-04-14
### Phase 4 `graph_state_timeslice` 복구 완료 (APPROVED)

#### 결정사항 — 스키마 및 키 교훈

| 항목 | 확인 결과 |
| --- | --- |
| `fact_stop_usage_hourly` 실제 스키마 | `service_date(date)`, `service_hour(integer)`, `stop_id`, `boardings`, `alightings` |
| `service_hour` 타입 | **정수 시간대**(0–23), timestamp 아님 |
| `state_ts` 생성 규칙 | `service_date + service_hour` 조합으로 timestamp 파생 |
| 노드 마스터 | `graph_node` 테이블은 존재하지 않음 → 실제 마스터는 `graph_node_master` |
| 간선 마스터 | `graph_edge_master` 는 현재 0행 → edge 기반 복구 불가 |
| STOP 노드 시드 | `dim_stop` 기준으로 `graph_node_master` 에 `STOP` 노드 **5,705건** 시드 완료 |

#### 실행결과 — `graph_state_timeslice` 적재 검증

| 지표 | 값 |
| --- | --- |
| `row_cnt` | **21,615,863** |
| `distinct_time_buckets` | 6,935 |
| `min_ts` | `2023-01-01 05:00:00+09` |
| `max_ts` | `2023-12-31 23:00:00+09` |
| `buckets_per_day` | 19 |
| `active_days` | 365 |
| `orphan_cnt` | 0 |
| `negative_boardings` | 0 |
| `invalid_hour` | 0 |

**Readiness 최종 판정**: `APPROVED` / `gat_rl_input_ready = YES`

#### Phase 4 수정/확정 파일

| 파일 | 핵심 수정 포인트 |
| --- | --- |
| `02_ingest_jobs/create_graph_state_timeslice.sql` | UTF-8 정리, `graph_node_master` 참조, `state_ts = service_date + service_hour` |
| `02_ingest_jobs/load_graph_state_timeslice_initial.sql` | `boardings/alightings` 실제 컬럼명 반영, 2023 observed only 필터 |
| `03_validation_queries/graph_state_timeslice_readiness.sql` | 0분모 방어(division-by-zero guard) 추가 |

---

### Phase 5 입력 규격 확정 완료 (APPROVED)

#### 결정사항 — 피처 분류

`graph_state_timeslice` 기준으로 RL 입력 특성(feature) 분류를 완료하였다.

#### 즉시 사용 가능 핵심 입력 (Core Features)

| 피처 | 비고 |
| --- | --- |
| `boardings_recent` | 현재 시점 승차 |
| `alightings_recent` | 현재 시점 하차 |
| `waiting_passenger_cnt` | 대기 승객 수 |
| `hour_of_day` | 시간대 (0–23) |
| `day_of_week` | 요일 (0–6) |
| `is_peak` | 첨두시 여부 |

#### 후속 확장 보류 컬럼

| 피처 | 보류 사유 |
| --- | --- |
| `predicted_demand_10m` | 예측 모델 미구축 |
| `predicted_demand_30m` | 예측 모델 미구축 |
| `nearest_bus_eta_sec` | 실시간 위치 원천 없음 |
| `active_bus_cnt_nearby` | 실시간 위치 원천 없음 |
| `link_travel_time_sec` | `graph_edge_master` 0행 |
| `link_speed_kmh` | `graph_edge_master` 0행 |
| `link_congestion_index` | `graph_edge_master` 0행 |
| `link_flow_proxy` | `graph_edge_master` 0행 |
| `incident_flag` | 사고 원천 없음 |

#### Phase 5 신규 산출물

| 파일 | 역할 |
| --- | --- |
| `04_model_inputs/graph_state_feature_spec.md` | RL 상태 공간 피처 명세서 |
| `02_ingest_jobs/create_rl_state_training_base.sql` | 학습 베이스 DDL + `LEAD()` 기반 (t, t+1) 페어링 |
| `03_validation_queries/rl_state_training_base_readiness.sql` | 학습 베이스 무결성 검증 수트 |

#### 실행결과 — `rl_state_training_base` 생성 및 검증

| 지표 | 값 |
| --- | --- |
| `row_count` | **21,612,482** |

#### 확정 컬럼 구조

| 컬럼 | 설명 |
| --- | --- |
| `state_ts` | 현재 관측 시각 |
| `next_state_ts` | 다음 관측 시각 (next observed, strict +1h 아님) |
| `node_uid` | 노드 고유 ID |
| `node_type` | 노드 유형 |
| `boardings_recent` | 현재 승차 |
| `alightings_recent` | 현재 하차 |
| `waiting_passenger_cnt` | 현재 대기 승객 |
| `hour_of_day` | 시간대 |
| `day_of_week` | 요일 |
| `is_peak` | 첨두시 |
| `next_boardings_recent` | 다음 시점 승차 |
| `next_alightings_recent` | 다음 시점 하차 |
| `next_waiting_passenger_cnt` | 다음 시점 대기 승객 |

> **해석**: `next_state_ts`는 strict one-hour future가 아니라 **next observed timestamp**(다음 관측 시각)이다.  
> 하루 19개 운영 버킷(05~23시) 기준으로 야간 공백이 존재할 수 있다.

#### Phase 5 Readiness 검증 결과

| 검사 항목 | 결과 |
| --- | --- |
| `time_inversion_cnt` | 0 |
| `null_next_state_ts_cnt` | 0 |
| `null_next_boardings_cnt` | 0 |
| `null_next_alightings_cnt` | 0 |
| `null_next_waiting_cnt` | 0 |
| 모든 음수 검사 | 0 |
| `invalid_hour_cnt` | 0 |

**최종 판정**: `final_decision = APPROVED` / `rl_training_base_ready = YES`

---

### 정보 관리 및 스킬(Agent Skill) 자산 최신화

- **신규 수집 오케스트레이터 스킬 제정**: 노선 링크 전체 수집 자동화를 지휘하는 `bus_route_link_bulk_collect_orchestrator` 스킬 문서(.agents/skills) 신규 제정.
  - 기존 단일 노선 적재 보호 장치(`bus_route_link_ingest_guard`)와 철저히 역할을 분리하여, API 루프 제어, Manifest 로깅 (수집/적재 분리), Missing Route 산출을 전담하도록 설계.
- **원천 문서 및 스펙 검증**: NotebookLM을 연동하여 프로젝트 설계서 및 관련 연구 문서를 교차 검증하고, 기존 UrbanBus RL 관련 자료를 프로젝트 컨텍스트에 동기화.
- **프로젝트 지배구조(Governance) 강화**: `project_log.md`를 최신화하여 지난 48시간의 기술적 결정 사항 및 작업 성과를 통합 기록.
- **작업 현황 동기화**: `02_ingest_jobs`와 `03_validation_queries`의 신규 파일 및 태스크 상태를 점검하고 우선순위 재정렬.

### 파이프라인 안전망(Guard/Orchestrator) 스킬 전면 분할 (COMPLETED)

- **4대 핵심 에이전트 스킬 신규 제정**: 데이터 파이프라인 각 전환 단계의 병목과 무결성 위협을 막기 위해 4개의 특수 스킬을 분할 신설.
  - `route_link_promotion_guard`: Staging → Sequence 승격 시 복합키 충돌 및 Sequence 단절 방어.
  - `spatial_graph_mapping_guard`: 150m-500m Fallback 기반 STOP-LINK 공간 매핑, `geom_5187` 원천 좌표계 적용 의무화.
  - `fact_usage_promotion_orchestrator`: 전체 통행 총합 대사(Reconciliation) 및 파생 연산 중 음수 수요(Negative) 이상치 유입 차단.
  - `graph_state_timeslice_bulk_loader`: 수천만 건 수준의 대용량 IO 트랜잭션 마비(Locking)를 막기 위한 아키텍처(TRUNCATE+인덱스 지연 생성) 가이드.

### 적재 성능 초고도화 (COMPLETED)

- 단일 트랜잭션(`BEGIN`~`COMMIT`) 내에서 `TRUNCATE` → 인덱스/PK 일시 제거 → 대량 삽입(Bulk Insert) → 인덱스/PK 재생성 구조를 적용하여, 2,160만 건의 상태 데이터를 `DataFileExtend` IO 병목 및 Lock 지연 없이 고속 적재 상용화에 성공.

---

## 📅 2026-04-13 - 대구 버스 이용량 파이프라인 안정화

### 대구 버스 이용량(Usage) 데이터 파이프라인 안정화

- **이용량 데이터 적재 완료**:
  - 대구 버스 이용 실적 데이터(2023년 전체 및 2025년 월별 데이터)의 Staging 적재 파이프라인 구축.
  - 윈도우 인코딩(UHC/EUC-KR) 처리 및 SQL 실행 환경(PowerShell 환경 변수 등) 최적화를 통해 적재 안정성 확보.
- **Staging Gate 검증 통과**:
  - `stg_daegu_stop_usage_*` 테이블에 대한 데이터 타입, 필수값, 중복 체크 프로세스 가동.
- **Fact 테이블 승격 및 집계**:
  - `promote_fact_stop_usage_hourly_from_2023_file.sql` 등을 통해 원천 통행 데이터를 1시간 단위 정류장별 승하차량 데이터로 변환/적재.
- **데이터 정합성 검증(Reconciliation)**:
  - `reconcile_usage_sums_2023.sql` 및 `reconcile_usage_sums_2025_monthly.sql`을 구현하여 Staging 총합과 Fact 총합 간의 불일치 여부를 0건으로 검증.

---

## 📅 2026-04-10 - Graph State Timeslice 설계

### Integrated Graph Master Phase 4 — Graph State Timeslice 설계 및 구축

#### 상태 테이블 설계

- `public.graph_state_timeslice` 표준 스키마 확정.
  - (state_ts, node_uid) 복합 PK 체계 및 10분 단위 시간 해상도 채택.
  - STOP(수요)과 LINK(소통) 노드 통합 관리를 위한 단일 상태 테이블 구조.

#### Hourly Allocation 전략

- 1시간 단위 수요 데이터(`fact_stop_usage_hourly`)를 10분 단위 파생 버킷으로 배분하는 로직 구현.
- `boardings_recent`, `alightings_recent` 지표에 대해 1/6 할당 적용.

#### 신규 파일 생성

- `01_data_contracts/graph_state_timeslice_spec.md`: 상태 컬럼 정의 및 해석 가이드.
- `02_ingest_jobs/create_graph_state_timeslice.sql`: 상태 테이블 DDL.
- `02_ingest_jobs/load_graph_state_timeslice_initial.sql`: 최신 가용일(service_date) 기준 초기 적재 스크립트.
- `03_validation_queries/graph_state_timeslice_readiness.sql`: 적재 정합성 및 결측치 검증 수트.

#### 핵심 결정 사항

- LINK 노드의 실시간 상태값(속도, ETA 등)은 현재 소스 부재로 인해 `NULL + TODO` 처리하며 향후 파이프라인 연계 예정.
- 요일(`day_of_week`), 첨두시간(`is_peak`) 등 AI 학습용 피처 컬럼 기본 포함.

---

## 📅 2026-04-11 - Integrated Graph Master Bootstrap & Phase 4 Status Report

### 완료 사항
- **통합 부트스트랩 실행**: `bootstrap_graph_master.ps1`을 통한 Phase 1~4 일괄 구축 시도.
- **DB 계정 및 접속 최적화**: `ryujo` -> `postgres` 사용자 전환 및 `urbanbus` DB 타겟팅 안정화.
- **Phase 1~3 APPROVED**: 노드/간선/매핑 레이어의 모든 검증 게이트 통과 및 구축 완료.
- **Phase 4 HOLD**: `fact_stop_usage_hourly` 데이터 부재(0건)로 인한 시계열 상태 적재 유보.

### 현재 데이터베이스 상태
- `graph_node_master`: 적재 완료
- `graph_edge_master`: 적재 완료
- `stop_link_mapping_master`: 2차 정밀 보정까지 완료
- `fact_stop_usage_hourly`: 테이블 생성 완료 (데이터 대기 중)

### 기술 결정 사항
- **Single Session Auth**: psql의 인터랙티브 인증 이슈 해결을 위한 `PGPASSWORD` 환경 변수 기반 단일 세션 스크립트 채택.
- **Conditional Load**: 원천 데이터 부재 시 분석 테이블 생성을 건너뛰는 데이터 기반 제어 로직 확립.

### 향후 작업
- `fact_stop_usage_hourly` 데이터 적재 파이프라인 가동.
- 데이터 적재 후 Phase 4 단독 실행 및 최종 승인.
- `ROUTE_INFERRED`, `ROUTE_CONFIRMED`, `MANUAL_REVIEW` 매핑 유형 확장 반영
- 통합 그래프 설계 문서 `integrated_graph_master_spec.md` 를 Phase 2~3 기준으로 개정 완료

#### 신규/수정 파일
- 신규: `02_ingest_jobs/prepare_source_spatial_columns.sql`

---

## 📅 2026-04-11 - Trip-level Synthetic Card Data Pipeline Redesign [COMPLETED]

### 완료 사항
- **Ingestion 전략 수정 완료**: 합성 데이터 API가 개별 통행(Trip-level) 데이터임을 확인하고 2단계 적재 체계 구축.
- **Deduplication 메커니즘**: `record_hash` (SHA256) 기반의 Staging 중복 방지 로직 적용.
- **Cross-Layer 검증 설계**: Staging의 `UTZTN_NOPE` 총합과 Fact의 승하차량 총합 일치 여부를 검증하는 Readiness SQL 구현.
- **Unmapped ID 관리**: 매핑 실패한 정류장 ID 목록을 별도로 추출할 수 있는 쿼리 포함.

### 생성/수정 파일 목록
- `01_data_contracts/daegu_transport_card_synth_trip_api_spec.md` [NEW]
- `01_data_contracts/fact_stop_usage_hourly_spec.md` [NEW]
- `02_ingest_jobs/create_stg_daegu_transport_card_usage_synth_trip.sql` [NEW]
- `02_ingest_jobs/load_daegu_transport_card_usage_synth_trip.ps1` [NEW]
- `02_ingest_jobs/create_fact_stop_usage_hourly.sql` [MODIFY]
- `02_ingest_jobs/load_fact_stop_usage_hourly_from_synth_trip.sql` [NEW]
- `03_validation_queries/daegu_transport_card_synth_trip_readiness.sql` [NEW]
- `03_validation_queries/fact_stop_usage_hourly_from_synth_trip_readiness.sql` [NEW]

### 핵심 설계 결정
- **UTZTN_NOPE 집계**: 시간대별 집계 시 인원수 필드를 가중치로 합산하여 수요 정합성 확보.
- **Source-Agnostic 팩트**: `source_system`을 PK에 포함하여 다양한 원천 데이터(합성/실측)의 병행 관리가 가능하도록 설계.

---

## 📅 2026-04-10 - Integrated Graph Master Phase 2~3 설계 및 반영

#### 완료 사항
- Integrated Graph Master Phase 2 착수 및 반영 준비 완료
- 원천 테이블의 공간 컬럼 직접 사용 원칙을 운영 기준으로 확정
- `geom_5187` 영구 컬럼 기반 거리 계산 원칙 채택
- 원천 준비 단계와 그래프 적재 단계를 분리한 파이프라인으로 정리
- STOP(정류장) 노드 적재 및 정류장-링크 1차 공간 매핑 로직 설계 완료
- `150m` 1차 탐색 + `500m` fallback (fallback=대체 보완 경로) 전략 채택
- `row_number()` 기반 정류장별 단일 primary(primary=주매핑) 링크 선택 기준 확정
- `STOP_TO_LINK`, `LINK_TO_STOP` 서비스 간선 생성 기준 확정
- 정류장-링크 2차 정밀 매핑 Phase 3 초안 작성 완료
- `route_id + move_dir_code + sequence(sequence=순서)` 기반 route-aware(route-aware=노선 인지형) 보정 로직 초안 작성 완료
- 기존 `SNAP_NEAREST` 1차 매핑 보존 원칙 확정
- `ROUTE_INFERRED`, `ROUTE_CONFIRMED`, `MANUAL_REVIEW` 매핑 유형 확장 반영
- 통합 그래프 설계 문서 `integrated_graph_master_spec.md` 를 Phase 2~3 기준으로 개정 완료

#### 신규/수정 파일
- 신규: `02_ingest_jobs/prepare_source_spatial_columns.sql`
- 수정: `02_ingest_jobs/load_graph_master_phase_2.sql`
- 신규: `03_validation_queries/graph_mapping_quality_check.sql`
- 신규: `02_ingest_jobs/refine_stop_link_mapping_phase_3.sql`
- 신규: `03_validation_queries/stop_link_mapping_phase_3_quality_check.sql`
- 수정: `01_data_contracts/integrated_graph_master_spec.md`

#### 핵심 설계 결정
1. 거리 계산은 view 내부 임시 `ST_Transform` 이 아니라 저장된 `geom_5187` 직접 사용을 원칙으로 한다.
2. `geom_5187` 가 원천 테이블에 없을 경우, 먼저 영구 컬럼 추가 및 백필(backfill=기존 데이터 채우기) 후 사용한다.
3. 정류장-링크 1차 매핑은 공간 근접 기반 `SNAP_NEAREST` 로 수행한다.
4. 1차 매핑은 `150m` 우선 탐색, 미매핑 정류장에 한해 `500m` fallback 을 허용한다.
5. 정류장별 primary 링크 선택은 `row_number()` 기반 최근접 1건으로 수행한다.
6. 2차 정밀 매핑은 `route_id + move_dir_code + stop_seq/link_seq` 정합성을 반영하는 route-aware 보정 단계로 분리한다.
7. 기존 1차 매핑은 삭제하지 않고 보정 결과를 누적/승격하는 방식으로 관리한다.

#### 검증 기준
- `duplicate primary mappings = 0`
- `node-edge integrity 오류 = 0`
- `STOP_TO_LINK`, `LINK_TO_STOP` 간선 정상 생성
- `unmapped stops` 최소화
- `distance_to_link_m > 100m` 경고 건수 점검
- `distance_to_link_m > 250m` 수동 검토 대상 분리
- Phase 3에서 반대 방향 링크 의심 건수 및 sequence 이상치 점검

#### 현재 단계 판정
- Integrated Graph Master Phase 2: 구현/검증 진행 단계
- Integrated Graph Master Phase 3: 초안 작성 완료, 실행 및 결과 판독 대기

#### 다음 작업
- Phase 2 실행:
  1. `prepare_source_spatial_columns.sql`
  2. `load_graph_master_phase_2.sql`
  3. `graph_mapping_quality_check.sql`
- Phase 3 실행:
  4. `refine_stop_link_mapping_phase_3.sql`
  5. `stop_link_mapping_phase_3_quality_check.sql`
- 결과에 따라 `APPROVED / HOLD / MANUAL_REVIEW` 판정
- 이후 `graph_state_timeslice` 설계로 진입

---

## 📅 2026-04-10 - Integrated Graph Master Phase Kick-off

- route_link_sequence 운영 기준을 기반으로 통합 그래프 마스터 1차 설계 시작
- stop / link 분리 노드 유형 전략 채택
- 방향성 그래프 기준 키 `(route_id, move_dir_code, link_seq)` 유지 확정
- `graph_node_master`, `graph_edge_master`, `stop_link_mapping_master` DDL 초안 작성
- `route_link_graph_edge_vw` 및 LINK_TO_LINK 적재 SQL 초안 작성
- readiness 검증 SQL 초안 작성
- `integrated_graph_master_spec.md` 문서 초안 작성

*Notes:*
- 현 단계는 LINK 중심 1차 그래프 승격까지를 범위로 함
- STOP_TO_LINK / LINK_TO_STOP 간선과 상태 시계열 테이블은 후속 단계로 분리
- 실제 링크 거리 / 이동시간 / 혼잡도는 링크 원천 마스터 결합 시 보강 예정

---

## 📅 2026-04-10 - 최종 패키징 산출물 생성

- 운영 표준 문서 정리 완료
- 최종 산출물 패키징 완료
- 배포/보관용 최종 압축본 생성:
  - `urbanbus_rl_project_final_2026-04-10.zip`
- 본 압축본은 2026-04-10 기준 운영 승인 스냅샷으로 보관

---

## 📅 2026-04-10 - 긴급 데이터 정비(Remediation) 종료 및 운영 안정화 진입

- **Remediation 종료**: `route_id='1000'` 오염 제거 및 중복 데이터(dedup) 처리 완료.
- **최종 승인**: 39,247행 승격 후 정합성 검증 PASS 확인 및 최종 승인(**APPROVED**).
- **운영 기준 SQL 확정**:
  - 승격: `promote_route_link_sequence_dedup.sql`
  - 검증: `route_link_promotion_readiness_dedup.sql`
- **운영 프로세스 정립**:
  - `01_data_contracts/route_link_operational_checklist.md` [신규]: 배치 작업 표준 체크리스트 추가.
- **상태 전환**: 문제 시정 단계에서 정규 운영 및 패키징 단계로 전환.

---

## 📅 2026-04-10 - 승격 운영 절차 표준화 및 Runbook 작성 (실행 전)

- **`02_ingest_jobs/route_link_promotion_execution_runbook.md`** [신규]: `route_link_sequence` 승격 및 검증을 위한 표준 운영 절차(Runbook) 작성 완료.
- **`02_ingest_jobs/README.md`** 업데이트: 승격 실행 Runbook 연결 및 가이드 추가.
- **특이사항**: 실행 전 운영 절차 문서화를 완료하였으며, 실제 승격 실행은 아직 진행하지 않음.

---

## 📅 2026-04-10 - 백업 대비 시스템 변경점 분석 (Diff 리포트 요약)

#### 1. 노선 링크 승격 검증 프레임워크(Validation Framework) 완성
- **`03_validation_queries/route_link_promotion_readiness.sql`** [신규]: Staging → Promoted 승격 무결성을 검증하는 종합 검증 SQL 쿼리 추가.
- **`01_data_contracts/route_link_promotion_result_guide.md`** [신규]: 품질 검증 쿼리 결과(PASS/WARN/FAIL) 판독 기준표 추가.
- **`01_data_contracts/route_link_promotion_result_report_template.md`** [신규]: 최종 승인 여부를 기록하는 마크다운 리포트 양식 완료.
- **`03_validation_queries/README.md`**: 상기 시스템 명세 업데이트.

#### 2. 검증용 작업 도구 추가
- **`scripts/compare_with_zip.ps1`** [신규]: 백업 압축파일(`urbanbus_rl_project (2).zip`)과 최신 작업 폴더 간의 파일 단위 내용 비교(Diff) 자동화 스크립트 작성.

---

## 📅 2026-04-10 - 승격 결과 리포트 템플릿 신규 추가

#### 1. 승인 / 보류 / 차단 기록 템플릿 문서화
- **`01_data_contracts/route_link_promotion_result_report_template.md`** [신규]: `route_link_promotion_readiness.sql` 실행 결과를 복붙하여 최종 판정을 기록하는 결과 리포트 템플릿 작성.
  - summary 기대값 / 실제값 / 판정 표 포함.
  - 상세 쿼리별 행 수, 판정, 비고 기록 표 포함.
  - 결과 원문 보존 구간과 최종 상태(승인 / 보류 / 차단), 후속 조치, 승인 기록 섹션 포함.

#### 2. 검증 README 확장
- **`03_validation_queries/README.md`** 업데이트: 결과 리포트 템플릿 문서 위치 및 사용 목적 추가.

---

## 📅 2026-04-10 - 파이프라인 안정화 및 전체 노선 승격 준비 작업

#### 1. PowerShell 파이프라인 기술적 부채 해결
- **`scripts/run_promote_route_link_sequence.ps1`** 수정:
  - `psql` 호출 시 `NOTICE` 메시지가 PowerShell 종료 오류로 오처리되는 문제 해결.
  - `try/finally` 블록으로 `ErrorActionPreference` 상태 복구 보장.
  - `stderr` 출력을 Verbose 로그로 분류하여 가독성 개선.
- **`scripts/validate_route_link_sequence.sql`** 수정:
  - `route_id` 조인 시 발생하던 `text` vs `integer` 타입 불일치 오류 수정 (Type casting 적용).

#### 2. 검증 및 프로토타이핑
- **샘플 검증 성공**: `route_id=1000` 노선에 대해 `Pre-validation -> Promotion -> Post-validation -> Final Report` 전 과정 `PASS` 확인.
- **적재 현황 점검**:
  - `stg_daegu_route_links_api`: 현재 1개 노선(287건) 적재 상태.
  - `api_snapshots`: 1개 파일 존재 확인.
- **차기 과제**: 전체 노선 대상 데이터 수집(getLink02 API) 및 소배치(5-10개) 승격 테스트 준비.

---

## 📅 2026-04-10 - 노선 링크 승격 검증 판독 기준 가이드 정리

#### 1. 검증 결과 판독 기준표 문서화
- **`01_data_contracts/route_link_promotion_result_guide.md`** [신규]: `route_link_promotion_readiness.sql` 실행 결과를 PASS / WARN / FAIL 로 해석하는 기준표 작성.
  - summary 기대값(`238 / 4 / 234 / 234 / 234`) 명시.
  - `unexpected_route_in_staging` 에서 legacy sample `route_id='1000'` 잔존 시 WARN 으로 해석하는 운영 기준 정리.
  - promoted 오염, 자연키 중복, 건수 불일치, key gap, 연속성 단절, `link_id` 충돌은 모두 FAIL 로 분류.

#### 2. 검증 README 보강
- **`03_validation_queries/README.md`** 업데이트: 판독 기준표 문서 위치 및 사용 목적 추가.

---

## 📅 2026-04-09
### 대구 버스 API 수집/검증/승급 파이프라인 구축

#### 1. 데이터 수집 자동화 (Ingest)
- **`scripts/load_route_links_api_csv.ps1`** [신규]: getLink02 API 스냅샷 CSV → `stg_daegu_route_links_api` 적재 자동화.
  - 파일명 기반 중복 로드 방지.
  - 컬럼 매핑 및 `route_id` 주입.
  - `psql \copy`를 이용한 고속 로드.
- **`scripts/create_stg_daegu_route_links_api.sql`** [신규]: Staging 테이블 DDL.

#### 2. 검증 시스템 (Guard)
- **`scripts/validate_route_link_sequence.sql`** [신규]: 738줄 규모의 종합 검증 스크립트.
  - **Pre-validation**: 타입 체크, 필수값, Natural Key 중복.
  - **Post-validation**: PK 중복, 링크 순번 연속성 단절, Key Collision, Count Mismatch.
  - **Guard 기능**: 오류 발견 시 `RAISE EXCEPTION`으로 파이프라인 중단.
  - psql 변수(`route_id_raw`, `snapshot_file`, `phase`)로 스코프 제어 가능.

#### 3. 승급 파이프라인 (Promotion)
- **`scripts/create_route_link_sequence.sql`** [신규]: 분석용 테이블 DDL 및 제약조건.
- **`scripts/promote_route_link_sequence.sql`** [신규]: Staging → 분석용 테이블 Upsert 로직.
  - `ON CONFLICT (route_id, move_dir_code, link_seq) DO UPDATE` 적용.

#### 4. 파이프라인 오케스트레이션
- **`scripts/run_promote_route_link_sequence.ps1`** [신규]: 전체 워크플로우 자동화.
  - 5단계 실행: Scope 확인 → DDL → Pre-check → Promote → Post-check.
  - 특정 `route_id` 또는 `snapshot_file` 기준 Scoped Execution 지원.
  - 작업 로그 자동 생성 (`Start-Transcript`).

#### 5. 인프라
- **`mcp_config.json`** 수정: JSON 파싱 오류 복구 및 MCP 서버 정상화.
- **`.agents/skills/bus_route_link_ingest_guard`** [신규]: 노선 링크 적재 전용 스킬.
- **`.agents/skills/data-contract-audit-skill`** 업데이트.

---

## 📅 2026-04-08
### 공간 데이터 모델링 및 정류장 기초 데이터 적재

#### 1. 공간 데이터 프로파일링
- `bs_20250903`, `link_20250903`, `node_20250903` Shapefile 레이어 분석.
- Geometry 타입, SRID, Row Count, 후보 키 컬럼 문서화.
- **`01_data_contracts/shape_layer_profile.md`** [신규] 작성.

#### 2. 정류장 원천 데이터 DB 적재
- 윈도우 인코딩(UHC/UTF-8) 충돌 해결: `geopandas` + SQLAlchemy 방식 사용.
- 5,705건 정류장 데이터 → `public.bs_20250903` 적재 완료.
- 공간 메타데이터 EPSG:5187 일치 확인.

#### 3. dim_stop 차원 테이블 승격
- **`scripts/create_dim_stop.sql`** [신규]: 차원 테이블 생성 스크립트.
- `ST_SetSRID`로 좌표계 명시 부여 → 이중 좌표계(`geom_5187`, `geom_4326`) 및 경위도 추출.
- **무결성 검증 결과**:
  - PK(`stop_id`) NULL/중복: **0건** ✅
  - 위경도 대구 권역 이탈: **0건** ✅
  - 유효하지 않은 Geometry: **0건** ✅
  - `stop_name` 중복: **250건** → **이름 기반 조인 금지 원칙 수립** ⚠️

#### 4. 데이터 계약 및 검증 프레임워크
- **`01_data_contracts/keys.md`** [신규]: 키 정의 및 조인 규칙 문서화.
- **`01_data_contracts/source_registry.md`** [신규]: 원천 데이터 출처 등록.
- **`01_data_contracts/schema_draft.sql`** [신규]: PostgreSQL 스키마 초안.
- **`02_ingest_jobs/`** [신규]: 적재 스크립트 디렉토리 (`load_daegu_stops.ps1`, `fetch_bus_api.ps1`).
- **`03_validation_queries/`** [신규]: SQL 검증 쿼리 (`basic_checks.sql`, `join_checks.sql`).
- **`.agents/skills/shape-inspection-skill`** [신규]: 공간 레이어 검사 스킬.
- **`.agents/skills/data-contract-audit-skill`** [신규]: 데이터 계약 오딧 스킬.

---

---
## 📅 2026-04-26
### Phase 2 Step 21~31 MAPPO Reward and A-family Rollout Bridge

오늘 작업에서는 기존 A/A90/A80/A70 placeholder 정책을 실제 MAPPO 연결 준비 단계로 승격하기 위한 보상 계약, rollout schema, policy boundary, bridge runner, canonical KPI smoke 경로를 단계적으로 구축했다.

#### 1. Step 21 — MAPPO reward contract v1 추가
- 생성 경로: `05_training/rewards/`
- 핵심 파일:
  - `mappo_reward_v1.py`
  - `reward_config_v1.yaml`
  - `README_reward_contract.md`
  - `test_mappo_reward_v1.py`
- 설계 원칙:
  - 서비스 품질 우선
  - 평균 대기시간과 p95 long-wait penalty 분리
  - `energy_proxy` 단독 최적화 금지
  - `energy_proxy_per_passenger` 기준 에너지 효율 평가
  - `fleet_reduction_ratio`는 보너스이며 hard objective가 아님
  - A 계열에서 `qwen_trigger_rate=0.0` 강제

#### 2. Step 22 — rollout schema contract v1 추가
- 생성 경로: `05_training/rollouts/`
- 핵심 파일:
  - `rollout_schema_v1.py`
  - `README_rollout_schema.md`
  - `test_rollout_schema_v1.py`
- 확정된 extended rollout 필드:
  - `passenger_demand_generated`
  - `passenger_served_count`
  - `passenger_service_rate`
  - `passenger_wait_p95_seconds`
  - `energy_proxy_per_passenger`
  - `active_bus_count`
  - `baseline_bus_count`
  - `fleet_reduction_ratio`
  - `policy_source`
  - `policy_checkpoint_path`
  - `source_mode`
  - reward component fields

#### 3. Step 23 — policy registry contract v1 추가
- 생성 경로: `05_training/policies/`
- 핵심 파일:
  - `policy_registry_v1.py`
  - `README_policy_registry.md`
  - `test_policy_registry_v1.py`
- 정책 종류를 `noop`, `rulebased`, `placeholder`, `mappo`로 분리.
- 실제 MAPPO 정책은 checkpoint가 없으면 실패하도록 설계.
- placeholder가 actual MAPPO로 조용히 대체되는 fallback을 금지.

#### 4. Step 24 — policy inference boundary v1 추가
- 핵심 파일:
  - `policy_inference_boundary_v1.py`
  - `README_policy_inference_boundary.md`
  - `test_policy_inference_boundary_v1.py`
- A/A90/A80/A70에서 Qwen 개입 금지.
- 실제 MAPPO 추론 경로는 checkpoint path를 반드시 요구.
- 논문용 causal claim 가능 여부를 boundary 수준에서 분리.

#### 5. Step 25 — A-family rollout bridge v1 추가
- 핵심 파일:
  - `a_family_rollout_bridge_v1.py`
  - `README_a_family_rollout_bridge.md`
  - `test_a_family_rollout_bridge_v1.py`
- 한 개 scenario row를 A/A90/A80/A70 rollout row로 변환하는 bridge 생성.
- reward contract, rollout schema, policy boundary를 단일 row 생성 경로에서 통합.

#### 6. Step 26 — A-family bridge smoke runner v1 추가
- 핵심 파일:
  - `run_a_family_bridge_smoke_v1.py`
  - `README_a_family_bridge_smoke_runner.md`
  - `test_a_family_bridge_smoke_runner_v1.py`
- A/A90/A80/A70 4조건 row를 한 번에 생성하고 검증.
- smoke artifact는 Git 커밋 대상에서 제외.

#### 7. Step 27 — A-family scenario rollout writer v1 추가
- 핵심 파일:
  - `run_a_family_scenario_rollout_writer_v1.py`
  - `README_a_family_scenario_rollout_writer.md`
  - `test_a_family_scenario_rollout_writer_v1.py`
- `scenario_index.parquet`를 입력으로 받아 A/A90/A80/A70 조건별 `window_rollup` 구조를 생성.
- 실제 B1_noop scenario index 기반 smoke 통과.

#### 8. Step 28 — root A-family rollout bridge entrypoint 추가
- 핵심 파일:
  - `run_causal_rollout_a_family_bridge_v1.py`
  - `README_run_causal_rollout_a_family_bridge.md`
  - `test_run_causal_rollout_a_family_bridge_v1.py`
- `05_training/` 루트에서 A-family bridge를 실행하는 엔트리포인트 추가.
- 아직 기존 `run_causal_rollout.py`는 직접 수정하지 않는 안전 검증 단계로 사용.

#### 9. Step 29 — run_causal_rollout.py bridge dispatch 추가
- `run_causal_rollout.py`에 `--a-family-bridge` 옵션 기반 dispatch 추가.
- 기존 causal rollout 실행 경로는 유지.
- `--a-family-bridge`가 명시된 경우에만 Step 28 bridge entrypoint로 위임.

#### 10. Step 30 — A-family bridge canonical KPI smoke 추가
- 핵심 파일:
  - `run_a_family_bridge_canonical_smoke_v1.py`
  - `README_a_family_bridge_canonical_smoke.md`
  - `test_a_family_bridge_canonical_smoke_v1.py`
- 검증 경로:
  - `run_causal_rollout.py --a-family-bridge`
  - A/A90/A80/A70 `window_rollup` 생성
  - `canonical_kpi_aggregator.py --mode official_rollup`
  - condition별 `canonical_eval` 산출물 생성
- 결과: smoke self-test와 CLI smoke 모두 PASS.

#### 11. Step 31 — cleanup and log update
- Step 26~30 smoke artifact 디렉터리 정리.
- Windows path docstring으로 인한 `SyntaxWarning: invalid escape sequence` 제거.
- Step 21~30 집중 regression test 재실행.
- `project_log.md`에 Phase 2 reward/bridge 구축 내역 반영.

#### 현재 의미
A 계열 정책은 아직 실제 학습된 MAPPO inference가 아니다. 그러나 이제 다음 경계가 모두 마련되었다.

1. reward contract
2. rollout schema contract
3. policy registry
4. policy inference boundary
5. A-family rollout bridge
6. scenario-index writer
7. root bridge entrypoint
8. `run_causal_rollout.py --a-family-bridge` dispatch
9. canonical KPI aggregator smoke path

#### 다음 단계
- 실제 MAPPO checkpoint가 생성되면 `--policy-kind mappo --checkpoint-path ...` 경로로 A/A90/A80/A70 rollout을 실행한다.
- 이후 `run_causal_rollout.py` 내부의 placeholder A-family 경로를 actual MAPPO inference 경로로 단계적으로 교체한다.
- 논문용 성능 주장은 `source_mode=causal_*`이며 actual checkpoint가 연결된 결과에 한해 사용한다.

