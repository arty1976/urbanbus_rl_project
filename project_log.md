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
|---

## 📅 2026-04-27
### Step 54~56 policy metadata 검증, canonical KPI 보존, A-family MAPPO smoke matrix 구축

오늘 작업에서는 Step 52에서 생성되기 시작한 A-family rollout 산출물이 canonical KPI 집계까지 이동하는 동안 policy metadata가 손실되지 않는지 검증하고, A/A90/A80/A70 네 조건을 한 번에 실행하는 `mappo_smoke` matrix 경로를 구축했다.

핵심 목표는 다음과 같다.

- `window_rollup.parquet` 안에 policy metadata가 빠짐없이 들어 있는지 검증한다.
- canonical KPI 집계 이후 `kpi_by_window.parquet`에서도 policy metadata가 보존되는지 확인한다.
- A/A90/A80/A70 다조건 `mappo_smoke` rollout matrix를 생성한다.
- B2 같은 비 A-family 조건이 matrix에 섞이면 즉시 차단한다.
- smoke checkpoint 결과가 actual 성능 주장으로 오해되지 않도록 `actual_policy_claim_ready=false`, `causal_policy_claim_ready=false`를 유지한다.

#### 1. Step 54 — window_rollup policy metadata validator 생성

생성 파일:

- `05_training/policies/validate_window_rollup_policy_metadata.py`
- `05_training/policies/test_validate_window_rollup_policy_metadata.py`

Step 54에서는 `window_rollup.parquet`가 canonical KPI 집계로 넘어가기 전에 policy metadata가 빠지거나 오염되지 않았는지 자동 검증하는 validator를 만들었다.

검증 대상 핵심 필드:

```text
policy_metadata_version
condition_id
source_mode
policy_source
policy_action_source_version
policy_action_source_mode
checkpoint_path
checkpoint_validation_mode
checkpoint_validator_ran
checkpoint_loaded
trained_model
performance_claim_allowed
placeholder_fallback_used
mock_action_used
qwen_train
qwen_inference
qwen_trigger_rate
reward_version
energy_proxy_model_version
k_dist_kwh_per_m
k_acc_kwh_per_event
k_idle_kwh_per_sec
actual_policy_claim_ready
causal_policy_claim_ready
```

검증 규칙:

```text
mappo_policy row:
- checkpoint_validator_ran = true
- checkpoint_loaded = true
- mock_action_used = false
- placeholder_fallback_used = false
- qwen_train = false
- qwen_inference = false
- qwen_trigger_rate = 0.0
- reward_version = mappo_reward_v1
- energy_proxy_model_version = daegu_energy_proxy_v1
- K_DIST/K_ACC/K_IDLE 상수 일치

mock_policy row:
- allow_mock=true일 때만 허용
- mock_action_used = true
- placeholder_fallback_used = true
- actual_policy_claim_ready = false
- causal_policy_claim_ready = false
```

self-test 결과:

```text
mappo_smoke window_rollup 검증 PASS
mock_smoke + allow_mock 검증 PASS
mock_policy인데 allow_mock 없음 → FAIL 정상
mappo_smoke인데 require_actual_ready 적용 → FAIL 정상
qwen_trigger_rate = 0.5 → FAIL 정상
policy_source 컬럼 삭제 → FAIL 정상
mappo_policy row인데 mock_action_used = true → FAIL 정상
```

최종 결과:

```text
[OK] Step 54 window_rollup policy metadata self-test PASS
[DONE] Step 54 window_rollup policy metadata validator complete.
```

해석:

Step 54는 rollout 산출물이 canonical KPI 집계로 들어가기 전, 정책 출처 정보가 온전한지 확인하는 입구 검문소다.

#### 2. Step 55 — canonical KPI aggregation 이후 policy metadata 보존 검증

수정/생성 파일:

- `05_training/evaluation/canonical_kpi_aggregator.py`
- `05_training/policies/test_canonical_kpi_preserves_policy_metadata.py`

Step 55에서는 `canonical_kpi_aggregator.py`가 `window_rollup.parquet`를 읽고 `kpi_by_window.parquet`를 만들 때 policy metadata를 보존하는지 검증했다.

초기 실패:

처음 실행에서는 rollout 단계의 `window_rollup.parquet` metadata 검증은 PASS했지만, canonical output인 `kpi_by_window.parquet` 검증은 FAIL했다.

원인:

```text
canonical_kpi_aggregator.py가 KPI 계산 후 out_cols를 다시 고르면서
policy metadata columns를 kpi_by_window.parquet에 포함하지 않았음
```

즉, 아래 필드들이 canonical output에서 사라졌다.

```text
policy_source
checkpoint_loaded
checkpoint_validator_ran
mock_action_used
placeholder_fallback_used
qwen_train
qwen_inference
reward_version
energy_proxy_model_version
actual_policy_claim_ready
causal_policy_claim_ready
```

수정:

`canonical_kpi_aggregator.py`에 `POLICY_METADATA_PRESERVE_COLUMNS`를 추가하고, `compute_official_kpi_by_window()`에서 KPI 필수 컬럼 외에도 입력에 존재하는 policy metadata columns를 output columns에 보존하도록 수정했다.

보존 대상 예:

```text
policy_metadata_version
policy_source
policy_action_source_version
policy_action_source_mode
checkpoint_path
checkpoint_validation_mode
checkpoint_validator_ran
checkpoint_loaded
trained_model
performance_claim_allowed
placeholder_fallback_used
mock_action_used
qwen_train
qwen_inference
reward_version
energy_proxy_model_version
k_dist_kwh_per_m
k_acc_kwh_per_event
k_idle_kwh_per_sec
actual_policy_claim_ready
causal_policy_claim_ready
policy_action_count
policy_nonzero_action_count
distance_m
acceleration_event_count
hold_seconds
passenger_served_count
energy_proxy_per_passenger
```

재실행 결과:

```text
A + mappo_smoke rollout 생성 PASS
pre-canonical window_rollup metadata validation PASS
canonical_kpi_aggregator.py official_rollup PASS
post-canonical kpi_by_window metadata validation PASS
```

최종 결과:

```text
[OK] Step 55 canonical policy metadata preservation self-test PASS
[DONE] Step 55 canonical KPI policy metadata preservation complete.
```

해석:

Step 55는 canonical KPI 집계 후에도 policy provenance가 사라지지 않도록 만든 단계다. 이제 KPI 결과를 볼 때도 해당 수치가 mock인지, MAPPO smoke인지, actual checkpoint인지, Qwen이 섞였는지 추적할 수 있다.

#### 3. Step 56 — A/A90/A80/A70 mappo_smoke rollout matrix 생성

생성 파일:

- `05_training/run_a_family_mappo_smoke_matrix.py`
- `05_training/policies/test_a_family_mappo_smoke_matrix.py`

Step 56에서는 A-family 네 조건을 한 번에 실행하는 `mappo_smoke` matrix runner를 만들었다.

대상 조건:

```text
A
A90
A80
A70
```

runner의 역할:

```text
smoke checkpoint 생성 또는 입력 checkpoint 사용
→ A/A90/A80/A70 조건 반복
→ 각 조건별 mappo_smoke rollout 생성
→ 각 window_rollup.parquet policy metadata 검증
→ canonical KPI aggregation 실행
→ canonical kpi_by_window.parquet policy metadata 검증
→ matrix_manifest.json / status.json 저장
```

실행 성공 경로:

```text
A, A90, A80, A70
+ seed 56
+ mappo_smoke
+ smoke checkpoint
→ rollout 4개 생성
→ window_rollup metadata validation
→ canonical KPI aggregation
→ post-canonical metadata validation
→ PASS
```

의도적 실패 테스트:

```text
--conditions A,B2
→ B2는 A-family 조건이 아니므로 RuntimeError 발생
→ expected failure로 정상 처리
```

B2가 거부되는 이유:

```text
Step 56 matrix는 A-family 전용이다.
허용 조건은 A, A90, A80, A70뿐이다.
B2는 rule-based baseline이므로 이 matrix에 섞이면 안 된다.
```

최종 결과:

```text
[OK] Step 56 A-family mappo_smoke matrix self-test PASS
[DONE] Step 56 A-family mappo_smoke rollout matrix complete.
```

#### 4. Step 54~56 이후 구조

현재 A-family smoke inference/evaluation 경로는 다음과 같다.

```text
MAPPO smoke checkpoint
→ MAPPOPolicyActionSource
→ A/A90/A80/A70 rollout writer
→ window_rollup.parquet
→ validate_window_rollup_policy_metadata.py
→ canonical_kpi_aggregator.py
→ kpi_by_window.parquet
→ post-canonical metadata validation
→ matrix_manifest.json / status.json
```

#### 5. 현재 policy metadata 보존 상태

이제 다음 값들이 rollout과 canonical KPI 양쪽에서 유지된다.

```text
policy_source = mappo_policy
source_mode = noncausal_*_mappo_policy_smoke_v1
checkpoint_loaded = true
checkpoint_validator_ran = true
mock_action_used = false
placeholder_fallback_used = false
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
reward_version = mappo_reward_v1
energy_proxy_model_version = daegu_energy_proxy_v1
actual_policy_claim_ready = false
causal_policy_claim_ready = false
```

#### 6. 연구적으로 중요한 점

Step 54~56은 성능을 낸 단계가 아니라, 성능 결과가 나중에 들어올 때 실험 오염을 막는 계층을 구축한 단계다.

방지하는 문제:

```text
window_rollup 단계에서 policy metadata 누락
canonical KPI 집계 과정에서 policy metadata 손실
mock action이 MAPPO action으로 오해되는 문제
smoke checkpoint가 actual checkpoint로 오해되는 문제
Qwen 개입 checkpoint가 A-family 경로에 섞이는 문제
B2 rule-based baseline이 A-family matrix에 섞이는 문제
noncausal smoke 결과가 causal claim으로 오해되는 문제
```

#### 7. 현재 판정

```text
Step 54: COMPLETED
Step 55: COMPLETED
Step 56: COMPLETED

window_rollup policy metadata validator: READY
canonical KPI policy metadata preservation: READY
A-family mappo_smoke matrix runner: READY

actual H200 trained checkpoint: NOT YET
actual performance claim: NOT YET
causal performance claim: NOT YET
```

#### 8. 다음 단계 후보

다음 단계는 아래 순서가 적절하다.

1. Step 57 기록 커밋 및 GitHub push
2. Step 58 H200 actual checkpoint preflight checklist 작성
3. Step 59 A-family mappo_actual execution plan 작성
4. Step 60 actual checkpoint가 들어왔을 때 Step 56 matrix를 mappo_actual로 승격하는 경로 준비
5. Step 61 causal simulator adapter 준비 전, noncausal/smoke claim guard 재점검

---
## ?뱟 2026-04-28
### Phase 2 Toy MAPPO Smoke Matrix Evaluation ??Step 93 ?꾨즺

?대쾲 ?묒뾽?먯꽌??Step 91???⑥씪 `A` condition smoke checkpoint evaluation??A-family matrix ?꾩껜濡??뺤옣?덈떎. 利?媛숈? Step 88 smoke checkpoint瑜?湲곗??쇰줈 `A`, `A90`, `A80`, `A70` 議곌굔?먯꽌 媛곴컖 toy causal evaluation rollout???섑뻾?섍퀬, 洹?寃곌낵瑜??섎굹??matrix-level canonical KPI 諛?inspector output?쇰줈 ?듯빀?덈떎.

?대쾲 ?④퀎???ъ쟾??smoke evaluation?대떎. ?곕씪????寃곌낵???ㅼ젣 MAPPO ?깅뒫 鍮꾧탳???쇰Ц ?깅뒫?쒓? ?꾨땲?? 紐⑹쟻? ??議곌굔??紐⑤몢 媛숈? 12-KPI schema? 媛숈? non-claim boundary濡?evaluation pipeline???듦낵?섎뒗吏 ?뺤씤?섎뒗 寃껋씠??

#### 1. Step 93 ??Toy MAPPO smoke matrix evaluator ?꾨즺

- **?앹꽦/?섏젙 ?뚯씪**
  - `05_training/evaluation/evaluate_toy_mappo_smoke_matrix.py`
  - `05_training/evaluation/test_evaluate_toy_mappo_smoke_matrix.py`
  - `05_training/evaluation/toy_mappo_smoke_matrix_evaluation.md`
  - `05_training/evaluation/evaluate_toy_mappo_smoke_checkpoint.py`

- **?듭떖 紐⑹쟻**
  - Step 88 smoke checkpoint瑜???踰??앹꽦?섍굅???낅젰 checkpoint瑜??ъ슜?쒕떎.
  - `A/A90/A80/A70` 議곌굔蹂꾨줈 Step 91 evaluator瑜??몄텧?쒕떎.
  - 媛?議곌굔?먯꽌 `raw_events.parquet`, `window_rollup.parquet`, canonical KPI output???앹꽦?쒕떎.
  - 議곌굔蹂?canonical output???⑹퀜 `matrix_canonical_eval`??留뚮뱺??
  - ?듯빀??matrix canonical output?먯꽌 A-family inspector瑜??ㅽ뻾?쒕떎.
  - 理쒖쥌 `matrix_evaluation_manifest.json`怨?`matrix_summary.csv`瑜??앹꽦?쒕떎.

- **寃利앸맂 寃쎈줈**
  ```text
  Step 88 smoke checkpoint
  -> A / A90 / A80 / A70 condition-level evaluation
  -> condition-level raw_events / window_rollup
  -> condition-level canonical KPI
  -> matrix_canonical_eval
  -> A-family 12-KPI inspector
  -> matrix_evaluation_manifest.json
  ```

- **self-test 寃곌낵**
  - `conditions = ['A', 'A90', 'A80', 'A70']`
  - `kpi_by_window_rows = 12`
  - `condition_summary = 4 rows`
  - `condition_vs_A_delta = 36 rows`
  - `12_kpis = validated`
  - `causal_allowed = true`
  - `performance_claim_allowed = false`
  - `smoke_evaluation_only = true`
  - `matrix_smoke_evaluation_only = true`

#### 2. Step 93 以?諛쒓껄?섍퀬 ?닿껐??臾몄젣

##### 2.1 condition-level inspector ?몄텧 ?꾩튂 臾몄젣

珥덇린 Step 93 援ы쁽?먯꽌??媛?condition??Step 91 evaluator濡??몄텧???? `A90`, `A80`, `A70` ?⑤룆 canonical output????댁꽌??A-family inspector瑜??ㅽ뻾?섎젮怨??덈떎. 洹몃윭??A-family inspector??`A` baseline???꾩슂?섍린 ?뚮Ц??`A90` ?⑤룆 ?낅젰?먯꽌???ㅼ쓬 ?ㅻ쪟媛 諛쒖깮?덈떎.

```text
RuntimeError: condition_summary must include A baseline condition
```

?닿껐:
- Step 91 evaluator??`--skip-inspector` ?듭뀡??異붽??덈떎.
- Step 93 matrix evaluator??議곌굔蹂??⑤룆 evaluation ?④퀎?먯꽌??inspector瑜?嫄대꼫?대떎.
- 留덉?留됱뿉 `A/A90/A80/A70`??紐⑤몢 ?⑹튇 `matrix_canonical_eval`?먯꽌留?inspector瑜??ㅽ뻾?쒕떎.

??援ъ“媛 留욌뒗 ?댁쑀:
```text
議곌굔蹂?evaluation = rollout + canonical KPI源뚯?留??앹꽦
matrix-level evaluation = A baseline???ы븿???듯빀 inspector ?ㅽ뻾
```

##### 2.2 `--skip-inspector` ?듭뀡 ?꾩튂 ?ㅻ쪟 ?섏젙

以묎컙 ?⑥튂?먯꽌 `--skip-inspector`媛 ?섎せ?댁꽌 `train_toy_causal_mappo_smoke.py` ?몄텧遺??遺숈뿀?? ??training script???대떦 ?듭뀡???뚯? 紐삵븯誘濡??ㅼ쓬 ?ㅻ쪟媛 諛쒖깮?덈떎.

```text
train_toy_causal_mappo_smoke.py: error: unrecognized arguments: --skip-inspector
```

?닿껐:
- training command?먯꽌??`--skip-inspector`瑜??쒓굅?덈떎.
- condition-level Step 91 evaluator command?먮쭔 `--skip-inspector`瑜?遺숈씠?꾨줉 ?섏젙?덈떎.

#### 3. ?꾩옱 ?섎?? ?쒓퀎

Step 93 ?꾨즺濡??꾨옒 寃쎈줈媛 ?ロ삍??

```text
12-KPI reward_v1
-> toy causal adapter reward
-> toy MAPPO smoke training
-> smoke checkpoint validation
-> single-condition smoke checkpoint evaluation
-> A-family smoke matrix evaluation
-> matrix-level canonical KPI
-> matrix-level inspector
```

?ㅻ쭔 ?꾩쭅 ?ㅼ쓬? ?섎??섏? ?딅뒗??

```text
A70???ㅼ젣濡?A蹂대떎 ?깅뒫??醫뗫떎.
MAPPO policy媛 ?섎졃?덈떎.
?援??꾩뿭 causal simulator?먯꽌 ?댁쁺 ?깅뒫??寃利앸릱??
?쇰Ц ?깅뒫?쒖뿉 ?ｌ쓣 ???덈떎.
```

?꾩옱 ?덉슜?섎뒗 ?댁꽍? ?ㅼ쓬?대떎.

```text
Phase 2 toy causal train/eval smoke matrix pipeline??援ъ“?곸쑝濡??곌껐?섏뿀??
A/A90/A80/A70 議곌굔???숈씪??12-KPI schema? non-claim boundary濡?evaluation pipeline???듦낵?쒕떎.
matrix-level inspector媛 A baseline 湲곕컲 delta table???앹꽦?????덈떎.
```

#### 4. ?ㅼ쓬 ?④퀎

異붿쿇 ?ㅼ쓬 ?④퀎??Step 95?대떎.

- **Step 95 ??Phase 2 toy causal smoke pipeline final status report**
  - Step 77~93 ?꾩껜瑜????μ쭨由?status report濡??뺣━?쒕떎.
  - ?쒖셿猷뚮맂 gate?? ?쒖븘吏?湲덉??섎뒗 claim?? ?쒕떎??H200/causal simulator ?뺤옣 吏꾩엯 議곌굔?앹쓣 紐낆떆?쒕떎.
  - ?댄썑遺?곕뒗 toy smoke媛 ?꾨땲???ㅼ젣 causal simulator ?뺤옣 ?먮뒗 H200 actual integration?쇰줈 ?섏뼱媛덉? 寃곗젙?쒕떎.

???
- **Step 95 ??Toy causal simulator realism gap analysis**
  - ?꾩옱 toy dynamics媛 ?ㅼ젣 ?援?踰꾩뒪 ?댁쁺怨??ㅻⅨ ?먯쓣 紐⑸줉?뷀븳??
  - ?ㅼ쓬 causal simulator v2?먯꽌 諛섎뱶??蹂닿컯??dynamics瑜??뺤쓽?쒕떎.

---
## ?뱟 2026-04-28
### Phase 2 Toy MAPPO Smoke Checkpoint Evaluation ??Step 91 ?꾨즺

?대쾲 ?묒뾽?먯꽌??Step 88?먯꽌 ?앹꽦??toy MAPPO smoke checkpoint瑜??ㅼ떆 遺덈윭? Phase 2 toy causal adapter?먯꽌 ?됯? rollout???섑뻾?섍퀬, 洹?寃곌낵瑜?canonical KPI aggregation 諛?12-KPI inspector源뚯? ?곌껐?덈떎.

?대쾲 ?④퀎???ъ쟾??smoke evaluation ?④퀎?? 利? ??寃곌낵???ㅼ젣 MAPPO ?깅뒫, ?援??꾩뿭 ?깅뒫, ?쇰Ц ?깅뒫?쒖뿉 ?ｌ쓣 ???덈뒗 寃곌낵瑜??섎??섏? ?딅뒗?? 紐⑹쟻? training smoke checkpoint媛 evaluation pipeline??援ъ“?곸쑝濡??듦낵?섎뒗吏 寃利앺븯??寃껋씠??

#### 1. Step 91 ??Toy MAPPO smoke checkpoint evaluator ?꾨즺

- **?앹꽦/?섏젙 ?뚯씪**
  - `05_training/evaluation/evaluate_toy_mappo_smoke_checkpoint.py`
  - `05_training/evaluation/test_evaluate_toy_mappo_smoke_checkpoint.py`
  - `05_training/evaluation/toy_mappo_smoke_checkpoint_evaluation.md`
  - `05_training/evaluation/inspect_toy_causal_a_family_kpis.py`

- **?듭떖 紐⑹쟻**
  - Step 88 smoke checkpoint瑜?遺덈윭?⑤떎.
  - toy causal adapter?먯꽌 evaluation rollout???섑뻾?쒕떎.
  - `raw_events.parquet`? `window_rollup.parquet`瑜??앹꽦?쒕떎.
  - canonical KPI aggregator濡?12-KPI 怨듭떇 吏묎퀎瑜??섑뻾?쒕떎.
  - Step 84 inspector濡?condition summary, time-band summary, seed summary, inspection report瑜??앹꽦?쒕떎.
  - 理쒖쥌 `evaluation_manifest.json`??non-claim boundary瑜?湲곕줉?쒕떎.

- **寃利앸맂 寃쎈줈**
  ```text
  Step 88 smoke checkpoint
  -> toy causal adapter evaluation rollout
  -> raw_events.parquet
  -> window_rollup.parquet
  -> canonical KPI aggregation
  -> 12-KPI inspector
  -> evaluation_manifest.json
  ```

- **self-test 寃곌낵**
  - `condition_ids = ['A']`
  - `window_rows = 3`
  - `seed_rows = 1`
  - `time_band_rows = 3`
  - `12_kpis = validated`
  - `causal_allowed = true`
  - `performance_claim_allowed = false`
  - `smoke_evaluation_only = true`

#### 2. Step 91 以?諛쒓껄?섍퀬 ?닿껐??臾몄젣

Step 91 怨쇱젙?먯꽌 ??媛吏 compatibility issue瑜??닿껐?덈떎.

##### 2.1 ?⑥씪 condition inspector compatibility

Step 84 inspector???먮옒 A-family 鍮꾧탳, 利?`A/A90/A80/A70` ?꾩껜 議곌굔???ㅼ뼱?ㅻ뒗 ?곹솴??湲곗??쇰줈 ?묒꽦?섏뼱 ?덉뿀?? 洹몃윭??Step 91 smoke evaluation? smoke checkpoint ?섎굹瑜??됯??섎뒗 ?④퀎??`condition_id = A` ?섎굹留??ㅼ뼱?붾떎.

???뚮Ц??`condition_vs_A_delta.csv`媛 鍮꾩뼱 ?덇퀬, 鍮?DataFrame?먯꽌 `kpi` column???쎌쑝???섎㈃???ㅻ쪟媛 諛쒖깮?덈떎.

?닿껐:
- `condition_vs_A_delta.csv`媛 鍮꾩뼱??schema瑜??좎??섎룄濡??섏젙
- `build_report()`媛 empty delta table???덉쟾?섍쾶 泥섎━?섎룄濡??섏젙
- ?⑥씪 condition evaluation?먯꽌??inspector媛 ?뺤긽 report瑜?留뚮뱾 ???덈룄濡?蹂닿컯

##### 2.2 raw window_rollup 12-KPI 蹂댁옣

canonical KPI aggregation? 12-KPI瑜?蹂댁젙???듦낵?덉?留? Step 91 evaluator媛 ??ν븳 raw `window_rollup.parquet`?먮뒗 ?쇰? KPI, ?뱁엳 `fleet_reduction_ratio`媛 鍮좎쭏 ???덉뿀??

?닿껐:
- evaluator媛 `StepResult.info["reward_metrics"]`?먯꽌 諛쏆? 12-KPI瑜?raw `window_rollup` row??蹂묓빀?섎룄濡??섏젙
- ?댁젣 raw window_rollup, canonical output, inspector output??紐⑤몢 媛숈? 12-KPI schema瑜??좎??쒕떎.

#### 3. ?꾩옱 ?섎?? ?쒓퀎

Step 91 ?꾨즺濡??꾨옒 寃쎈줈媛 ?ロ삍??

```text
12-KPI reward_v1
-> toy causal adapter step reward
-> toy MAPPO smoke training
-> smoke checkpoint
-> checkpoint validator
-> smoke checkpoint evaluation rollout
-> canonical KPI aggregation
-> inspector
```

?섏?留??꾩쭅 ?ㅼ쓬???섎??섏? ?딅뒗??

```text
?ㅼ젣 MAPPO policy媛 ?깅뒫???덈떎.
A condition???ㅼ젣 ?援?踰꾩뒪 ?댁쁺?먯꽌 ?곗닔?섎떎.
H200 蹂명븰??寃곌낵媛 寃利앸릱??
?쇰Ц ?깅뒫?쒖뿉 ?ｌ쓣 ???덈떎.
```

?꾩옱 ?덉슜?섎뒗 ?댁꽍? ?ㅼ쓬?대떎.

```text
Phase 2 toy causal train/eval smoke pipeline??援ъ“?곸쑝濡??곌껐?섏뿀??
12-KPI reward metrics媛 adapter, training, checkpoint, validator, evaluator, canonical aggregator, inspector源뚯? ?꾨떖?쒕떎.
?앹꽦??checkpoint? evaluation output? ?덉쟾?섍쾶 smoke artifact濡??쇰꺼留곷맂??
```

#### 4. ?ㅼ쓬 ?④퀎

異붿쿇 ?ㅼ쓬 ?④퀎??Step 92 runbook ?뺣━ ??Step 93?대떎.

- **Step 92**
  - Phase 2 toy causal train/eval runbook ?묒꽦
  - Step 86~91 ?꾩껜 ?ㅽ뻾 ?쒖꽌? non-claim boundary ?뺣━
  - ?ㅼ쓬 ?④퀎 吏꾩엯 議곌굔 紐낆떆

- **Step 93**
  - Toy causal MAPPO smoke matrix evaluator
  - A/A90/A80/A70 議곌굔蹂?smoke checkpoint evaluation??媛숈? schema濡??뺤옣
  - ?? ?ъ쟾??`performance_claim_allowed=false` ?좎?

---
## ?뱟 2026-04-28
### Phase 2 12-KPI Reward 諛?Toy MAPPO Smoke Scaffold ??Step 86~89 ?꾨즺

?ㅻ뒛 ?묒뾽?먯꽌??Phase 2 toy causal simulator媛 ?⑥닚 dynamics 寃利앹쓣 ?섏뼱, 12-KPI 湲곕컲 reward contract? 理쒖냼 MAPPO smoke training 寃쎈줈源뚯? ?곌껐?섎뒗吏瑜?寃利앺뻽?? ?대쾲 援ш컙? ?ㅼ젣 ?쇰Ц ?깅뒫 二쇱옣?대굹 H200 蹂명븰?듭씠 ?꾨땲?? causal toy simulator ?꾩뿉??reward wiring, training smoke, checkpoint contract媛 ?덉쟾?섍쾶 ?묐룞?섎뒗吏瑜??뺤씤?섎뒗 ?④퀎??

#### 1. Step 86 ??12-KPI 湲곕컲 MAPPO reward v1 contract ?꾨즺

- **?앹꽦/?섏젙 ?뚯씪**
  - `05_training/rewards/__init__.py`
  - `05_training/rewards/mappo_reward_v1.py`
  - `05_training/rewards/reward_config_v1.yaml`
  - `05_training/rewards/README_reward_contract.md`
  - `05_training/rewards/test_mappo_reward_v1.py`

- **?듭떖 紐⑹쟻**
  - Phase 2 怨듭떇 12醫?KPI瑜??낅젰?쇰줈 諛쏅뒗 MAPPO reward contract瑜?怨좎젙?덈떎.
  - reward???쒕퉬???덉쭏??1?쒖쐞, ?먮꼫吏 ?⑥쑉怨?fleet reduction??2?쒖쐞濡??먮룄濡??ㅺ퀎?덈떎.
  - `fleet_reduction_ratio`??蹂대꼫???깃꺽?대ŉ, `passenger_service_rate`媛 臾대꼫吏硫?fleet bonus媛 臾댄슚?붾릺?꾨줉 ?덈떎.
  - `energy_proxy` ?⑤룆???꾨땲??`energy_proxy_per_passenger`瑜?以묒떖?쇰줈 ?먮꼫吏 ?⑥쑉???됯??섎룄濡??덈떎.
  - `passenger_wait_p95_seconds`???됯퇏 ?湲곗떆媛꾨낫??媛뺥븯寃?踰뚯젏?뷀뻽??

- **?듭떖 諛섑솚 援ъ“**
  - `reward_total`
  - `reward_components`
  - `reward_debug`
  - `reward_version = mappo_reward_v1`
  - `reward_claim_boundary = toy_causal_training_reward_contract_not_paper_performance_claim`

- **寃利?寃곌낵**
  - `python -m py_compile` ?듦낵
  - `test_mappo_reward_v1.py` self-test ?듦낵
  - low service rate, high p95 wait, energy efficiency, fleet bonus secondary constraint, missing KPI, NaN/inf 諛⑹뼱 寃利??꾨즺

#### 2. Step 87 ??CausalSimulatorAdapter.step() reward_v1 ?곌껐 ?꾨즺

- **?앹꽦/?섏젙 ?뚯씪**
  - `05_training/adapters/causal_simulator_adapter.py`
  - `05_training/adapters/test_causal_simulator_reward_v1_integration.py`
  - `05_training/adapters/test_causal_simulator_adapter_contract.py`
  - `05_training/adapters/causal_simulator_adapter_contract.md`

- **?듭떖 紐⑹쟻**
  - `CausalSimulatorAdapter.step()`媛 12-KPI reward metrics瑜?援ъ꽦?섍퀬, `mappo_reward_v1.compute_total_reward()`瑜??몄텧?섎룄濡??곌껐?덈떎.
  - 怨꾩궛??`reward_total`? 紐⑤뱺 agent?먭쾶 媛숈? team-level scalar reward濡?broadcast?쒕떎.
  - `StepResult.info`?먮뒗 reward debugging怨?claim boundary ?뺣낫媛 ?ㅼ뼱媛꾨떎.

- **StepResult.info reward fields**
  - `reward_version`
  - `reward_claim_boundary`
  - `reward_total`
  - `reward_components`
  - `reward_debug`
  - `reward_metrics`

- **寃利?寃곌낵**
  - reward info contract ?듦낵
  - 紐⑤뱺 agent reward媛 `reward_total`怨?媛숈?吏 寃利??듦낵
  - 媛숈? seed + 媛숈? action sequence??deterministic reward 寃利??듦낵
  - all-hold action pattern??dispatch-dominant pattern蹂대떎 ??? reward瑜?諛쏅뒗吏 寃利??듦낵
  - `causal_comparison_allowed = true`???좎??섎릺, reward claim boundary??paper-level performance claim???꾨떂??紐낆떆

#### 3. Step 88 ??Toy causal MAPPO training smoke scaffold ?꾨즺

- **?앹꽦 ?뚯씪**
  - `05_training/train_toy_causal_mappo_smoke.py`
  - `05_training/adapters/test_toy_causal_mappo_training_smoke.py`
  - `05_training/adapters/toy_causal_mappo_training_smoke.md`

- **?듭떖 紐⑹쟻**
  - ?ㅼ젣 full MAPPO training???꾨땲?? 理쒖냼 training smoke 寃쎈줈瑜?寃利앺뻽??
  - actor/critic tiny torch model???ъ슜??rollout, reward collection, policy/value loss 怨꾩궛, optimizer step, checkpoint ??κ퉴吏 ?섑뻾?덈떎.
  - ??checkpoint???숈뒿 ?꾨즺 紐⑤뜽???꾨땲??smoke artifact??

- **寃利앸맂 寃쎈줈**
  ```text
  CausalSimulatorAdapter
  -> toy MAPPO actor/critic
  -> 12-KPI reward_v1 team reward
  -> rollout trace
  -> tiny policy/value update
  -> smoke checkpoint ???  -> training_manifest.json 湲곕줉
  -> deterministic reward sequence 寃利?  ```

- **?앹꽦 ?곗텧臾?*
  - `training_manifest.json`
  - `training_trace.csv`
  - `checkpoints/toy_mappo_smoke_checkpoint.pt`

- **寃利?寃곌낵**
  - `total_transitions = 32`
  - `reward_version = mappo_reward_v1`
  - `trained_model = false`
  - `performance_claim_allowed = false`
  - `smoke_training_only = true`
  - deterministic reward sequence 寃利??듦낵

#### 4. Step 89 ??Toy MAPPO smoke checkpoint contract validator ?꾨즺

- **?앹꽦 ?뚯씪**
  - `05_training/policies/validate_toy_mappo_smoke_checkpoint.py`
  - `05_training/policies/test_toy_mappo_smoke_checkpoint_validator.py`
  - `05_training/policies/toy_mappo_smoke_checkpoint_contract.md`

- **?듭떖 紐⑹쟻**
  - Step 88?먯꽌 ?앹꽦??checkpoint媛 ?쒗븰???꾨즺 紐⑤뜽?앹씠 ?꾨땲???쐓moke checkpoint?앸씪??怨꾩빟??吏?ㅻ뒗吏 ?낅┰ 寃利앺븳??
  - `training_manifest.json`, `training_trace.csv`, `toy_mappo_smoke_checkpoint.pt` ?ъ씠???꾩닔 metadata ?쇨??깆쓣 寃利앺븳??
  - 議곗옉??manifest?먯꽌 `performance_claim_allowed=true`媛 ?ㅼ뼱?ㅻ㈃ validator媛 ?뺥솗??嫄곕??섎뒗吏 negative test???섑뻾?덈떎.

- **寃利앸맂 ?듭떖 怨꾩빟**
  - `trained_model = false`
  - `performance_claim_allowed = false`
  - `smoke_training_only = true`
  - `causal_comparison_allowed = true`
  - `reward_version = mappo_reward_v1`
  - `reward_claim_boundary = toy_causal_training_reward_contract_not_paper_performance_claim`
  - 12-KPI `reward_metric_keys` 議댁옱
  - loss/reward/trace finite 寃利?
#### 5. ?꾩옱 ?섎?? ?쒓퀎

Step 86~89 ?꾨즺濡??꾨옒 寃쎈줈媛 ?ロ삍??

```text
12-KPI canonical metrics
-> mappo_reward_v1 reward contract
-> CausalSimulatorAdapter.step() team reward
-> toy MAPPO smoke rollout/update
-> smoke checkpoint
-> checkpoint contract validator
```

?ㅻ쭔 ??寃곌낵???꾩쭅 ?ㅼ쓬???섎??섏? ?딅뒗??

```text
?ㅼ젣 MAPPO policy媛 ?섎졃?덈떎.
?援??꾩뿭 causal simulator?먯꽌 ?깅뒫??寃利앸릱??
?쇰Ц ?깅뒫?쒖뿉 ?ｌ쓣 ???덈뒗 寃곌낵媛 ?섏솕??
H200 蹂명븰??checkpoint? ?숇벑?섎떎.
```

?꾩옱 ?덉슜?섎뒗 ?댁꽍? ?ㅼ쓬?대떎.

```text
Phase 2 toy causal environment?먯꽌 reward_v1怨?MAPPO smoke training path媛 援ъ“?곸쑝濡??곌껐?먮떎.
12-KPI reward metrics媛 adapter -> training -> checkpoint -> validator源뚯? ?꾨떖?쒕떎.
?앹꽦??checkpoint???덉쟾?섍쾶 smoke artifact濡??쇰꺼留곷맂??
```

#### 6. ?ㅼ쓬 ?④퀎

異붿쿇 ?ㅼ쓬 ?④퀎??Step 91?대떎.

- **Step 91 ??Toy causal MAPPO smoke rollout evaluator**
  - Step 88 smoke checkpoint瑜?遺덈윭? toy causal adapter?먯꽌 rollout ?됯?瑜??섑뻾?쒕떎.
  - evaluation output??`raw_events.parquet`, `window_rollup.parquet`, canonical KPI, inspector源뚯? ?곌껐?쒕떎.
  - ?? ?ъ쟾??smoke checkpoint 湲곕컲?대?濡?performance claim? 湲덉??쒕떎.

?먮뒗 ??덉쑝濡?Step 91???ㅼ쓬泥섎읆 ?≪쓣 ???덈떎.

- **Step 91 ??Phase 2 toy causal training/evaluation runbook**
  - Step 86~89??reward, training smoke, checkpoint validator, evaluation 怨꾪쉷????臾몄꽌濡??뺣━?쒕떎.
  - ?댄썑 H200 actual training ?곌껐 ?꾩뿉 ?꾩슂??gate瑜?怨좎젙?쒕떎.

---
## ?뱟 2026-04-27
### Phase 2 12-KPI Schema Extension 諛?A-Family Inspector ??Step 83~84 ?꾨즺

?ㅻ뒛 ?묒뾽?먯꽌??Phase 2 toy causal simulator??KPI 泥닿퀎瑜?湲곗〈 6醫?以묒떖?먯꽌 12醫?KPI schema濡??뺤옣?섍퀬, A/A90/A80/A70 議곌굔蹂?寃곌낵瑜??щ엺???댁꽍 媛?ν븳 ?쒖? JSON report濡??먭??????덈뒗 inspector瑜?異붽??덈떎.

?대쾲 援ш컙? ?ъ쟾??toy causal simulator 湲곕컲 sanity validation ?④퀎?? ?곕씪??paper-level performance claim? ?섏? ?딅뒗?? ???④퀎???섎?????2-KPI causal evaluation wiring???뺤긽?곸쑝濡??앹꽦쨌吏묎퀎쨌寃?щ맂?ㅲ앸뒗 寃껋쓣 ?뺤씤??寃껋씠??

#### 1. Step 83 ??Phase 2 12-KPI schema extension ?꾨즺

- **?듭떖 紐⑹쟻**
  - 湲곗〈 canonical shared KPI 6醫낆쓣 Phase 2 怨듭떇 12醫?KPI schema濡??뺤옣?쒕떎.
  - legacy 6-KPI 寃쎈줈? Phase 2 12-KPI 寃쎈줈瑜?紐⑤몢 吏?먰븯?꾨줉 canonical KPI aggregator瑜?怨꾩빟 湲곕컲?쇰줈 ?뺣━?쒕떎.
  - toy causal rollout?먯꽌 ?좉퇋 KPI 諛?fleet-ratio metadata媛 downstream canonical output源뚯? 蹂댁〈쨌怨꾩궛?섎룄濡??쒕떎.

- **Phase 2 怨듭떇 12醫?KPI**
  1. `cv_headway`
  2. `avg_wait_seconds`
  3. `bunching_rate`
  4. `on_time_rate`
  5. `intervention_rate`
  6. `energy_proxy`
  7. `passenger_demand_generated`
  8. `passenger_served_count`
  9. `passenger_service_rate`
  10. `passenger_wait_p95_seconds`
  11. `energy_proxy_per_passenger`
  12. `fleet_reduction_ratio`

- **異붽???蹂닿컯??怨꾩궛**
  - `passenger_service_rate = passenger_served_count / max(passenger_demand_generated, 1)`
  - `passenger_wait_p95_seconds`??toy causal simulator??deterministic wait proxy瑜?湲곕컲?쇰줈 怨꾩궛
  - `energy_proxy_per_passenger = energy_proxy_total / max(passenger_served_count, 1)`
  - `fleet_reduction_ratio = 1.0 - active_bus_count / max(baseline_bus_count, 1)`
  - A-family toy setting?먯꽌 baseline bus count??8?濡??먭퀬, 議곌굔蹂?active bus ratio瑜?諛섏쁺
    - `A`: 1.0
    - `A90`: 0.9
    - `A80`: 0.8
    - `A70`: 0.7

- **canonical KPI aggregator 蹂닿컯**
  - `canonical_kpi_aggregator.py`媛 contract??`shared_kpis` 湲곗??쇰줈 6-KPI legacy? 12-KPI Phase 2瑜?紐⑤몢 泥섎━?섎룄濡??섏젙
  - `official_rollup` output??`kpi_by_window.parquet`, `kpi_by_seed.parquet`, `kpi_by_time_band.parquet`, `kpi_overall.json`?먯꽌 12醫?KPI媛 紐⑤몢 寃利앸릺?꾨줉 蹂닿컯
  - 以묐났 KPI column???앷린??寃쎌슦?먮룄 ?덉쟾?섍쾶 泥섎━?섎룄濡?duplicate-safe column selection??異붽?
  - bounded KPI 寃利?踰붿쐞瑜??뺤옣
    - `bunching_rate`
    - `on_time_rate`
    - `intervention_rate`
    - `passenger_service_rate`
    - `fleet_reduction_ratio`

- **寃利?寃곌낵**
  - Step 79 怨꾩뿴 canonical integration self-test PASS
  - Step 81 A-family matrix self-test PASS
  - 12醫?KPI validation PASS
  - `causal_comparison_allowed = true` ?좎? ?뺤씤

#### 2. Step 84 ??Toy causal A-family 12-KPI result inspector ?꾨즺

- **?앹꽦 ?뚯씪**
  - `05_training/evaluation/inspect_toy_causal_a_family_kpis.py`
  - `05_training/evaluation/test_inspect_toy_causal_a_family_kpis.py`
  - `05_training/evaluation/toy_causal_a_family_kpi_inspection.md`

- **??븷**
  - Phase 2 toy causal A-family canonical output???щ엺???댁꽍 媛?ν븳 ?쒕줈 ?붿빟?쒕떎.
  - `A` 議곌굔??baseline?쇰줈 ?먭퀬 `A90`, `A80`, `A70`??12-KPI 蹂?붾웾怨?蹂?붿쑉??怨꾩궛?쒕떎.
  - 議곌굔蹂??붿빟, seed蹂??붿빟, time_band蹂??붿빟, A ?鍮?delta report瑜??앹꽦?쒕떎.

- **?낅젰**
  - `artifacts/phase2_toy_causal_a_family_matrix_selftest/canonical_eval/kpi_by_window.parquet`
  - `artifacts/phase2_toy_causal_a_family_matrix_selftest/canonical_eval/kpi_by_seed.parquet`
  - `artifacts/phase2_toy_causal_a_family_matrix_selftest/canonical_eval/kpi_by_time_band.parquet`
  - `artifacts/phase2_toy_causal_a_family_matrix_selftest/canonical_eval/kpi_overall.json`

- **?앹꽦 ?곗텧臾?*
  - `condition_summary.csv`
  - `condition_vs_A_delta.csv`
  - `time_band_summary.csv`
  - `seed_summary.csv`
  - `inspection_report.json`

- **寃利?寃곌낵**
  - `A/A90/A80/A70 횞 seeds 1,2,3` ?꾩껜 matrix ?ъ깮??PASS
  - `kpi_by_window` rows: 36
  - `kpi_by_seed` rows: 12
  - `kpi_by_time_band` rows: 36
  - `condition_summary` rows: 4
  - `condition_vs_A_delta` rows: 36
  - 12醫?KPI 紐⑤몢 validated
  - `causal_comparison_allowed = true` ?좎?
  - `claim_boundary = toy_causal_sanity_only_not_paper_performance_claim` 紐낆떆

#### 3. ?꾩옱 ?섎?? ?쒓퀎

Step 83~84 ?꾨즺濡??꾨옒 寃쎈줈媛 ?ロ삍??

```text
Phase 2 toy causal dynamics
-> 12-KPI window_rollup
-> canonical_kpi_aggregator.py official_rollup
-> 12-KPI kpi_by_window / kpi_by_seed / kpi_by_time_band / kpi_overall
-> A-family condition summary / A ?鍮?delta / time-band summary
```

?ㅻ쭔 ??寃곌낵???꾩쭅 toy causal simulator 湲곕컲?대떎.

?곕씪???ㅼ쓬 臾몄옣? ?꾩쭅 湲덉??쒕떎.

```text
A70???ㅼ젣 ?援?踰꾩뒪 ?댁쁺?먯꽌 ?깅뒫???곗닔?섎떎.
A90/A80/A70???ㅼ젣 MAPPO ?뺤콉?쇰줈 寃利앸릱??
?쇰Ц ?깅뒫?쒖뿉 諛붾줈 ?ｌ쓣 ???덈떎.
```

?꾩옱 ?덉슜?섎뒗 ?댁꽍? ?ㅼ쓬?대떎.

```text
Phase 2 toy causal simulator?먯꽌 12-KPI schema wiring???뺤긽 ?묐룞?쒕떎.
A-family 議곌굔蹂?fleet ratio媛 canonical KPI? inspector源뚯? ?꾨떖?쒕떎.
12-KPI 寃곌낵瑜?議곌굔蹂??쒓컙?蹂?seed蹂꾨줈 寃?ы븷 ???덈떎.
```

#### 4. ?ㅼ쓬 ?④퀎

?ㅼ쓬 異붿쿇 ?④퀎??Step 86?대떎.

- **Step 86 ??12-KPI 湲곕컲 MAPPO reward v1 ?곌껐**
  - `mappo_reward_v1.py` ?먮뒗 湲곗〈 reward module??12醫?KPI瑜??낅젰?쇰줈 諛쏆븘 reward components瑜?怨꾩궛?섎룄濡??쒕떎.
  - service quality瑜?1?쒖쐞, energy/fleet reduction??2?쒖쐞濡??먮뒗 reward contract瑜?怨좎젙?쒕떎.
  - `passenger_service_rate`, `passenger_wait_p95_seconds`, `energy_proxy_per_passenger`, `fleet_reduction_ratio`媛 reward debug output??紐낆떆?섎룄濡??쒕떎.

以묎컙??Step 85 ?꾩뿉??蹂寃쎈텇??commit/push?섍퀬, artifacts??而ㅻ컠?섏? ?딅뒗??

---
## ?뱟 2026-04-27
### Phase 2 Causal Simulator Adapter ?쒖옉 ??Step 77~79 ?꾨즺

?ㅻ뒛 ?묒뾽?먯꽌??Phase 1??historical replay ?쒓퀎瑜??섏뼱?? action??next state???ㅼ젣濡??곹뼢??二쇰뒗 理쒖냼 ?멸낵 ?쒕??덉씠??寃쎈줈瑜??댁뿀?? ?대쾲 援ш컙? ?쇰Ц ?깅뒫 二쇱옣???꾪븳 ?④퀎媛 ?꾨땲?? Phase 2 causal evaluation?쇰줈 吏꾩엯?섍린 ?꾪븳 adapter contract, toy dynamics, rollout writer, canonical KPI ?곌껐??寃利앺븯???④퀎??

#### 1. Step 77 ??Phase 2 causal simulator adapter contract ?묒꽦 諛?self-test ?듦낵

- **?앹꽦 ?뚯씪**
  - `05_training/adapters/causal_simulator_adapter_contract.md`
  - `05_training/adapters/causal_simulator_adapter.py`
  - `05_training/adapters/test_causal_simulator_adapter_contract.py`

- **?듭떖 寃곗젙**
  - `HistoricalReplayAdapter`??non-causal replay濡??좎??쒕떎.
  - historical replay?먯꽌??action??next state瑜?諛붽씀吏 ?딆쑝誘濡?causal performance claim? 遺덇??섎떎.
  - Phase 2??`CausalSimulatorAdapter`??`action_t -> state_{t+1} -> KPI` 寃쎈줈瑜?理쒖냼 ?섏??먯꽌 援ы쁽?쒕떎.
  - 1?④퀎 怨듦컙 踰붿쐞???援??꾩뿭???꾨땲???섏꽦援??듭떖 嫄곗젏, 踰붿뼱-留뚯큿 ?ㅽ???toy corridor濡??쒗븳?쒕떎.
  - agent ?섎뒗 5~10? 踰꾩뒪 ?섏??쇰줈 ?쒗븳?쒕떎.
  - control granularity??30遺꾩쑝濡?怨좎젙?쒕떎.

- **援ы쁽??理쒖냼 dynamics**
  - passenger queue update
  - bus position update
  - headway sample update
  - hold / dispatch / skip action effect
  - energy proxy update
  - intervention count / decision count update
  - `causal_comparison_allowed = true` metadata 遺??
- **self-test ?듦낵 ??ぉ**
  - `reset()` / `step(actions)` / `get_graph_skeleton()` / `compute_kpis()` 怨꾩빟 ?듦낵
  - 媛숈? seed + 媛숈? action sequence??deterministic
  - 媛숈? seed + ?ㅻⅨ action sequence??bus position, passenger queue, next observation???ㅻⅤ寃?留뚮벀
  - raw event schema 寃利?  - window rollup schema 寃利?  - 5~10 agent ?쒗븳 寃利?  - 30遺?control granularity 寃利?
#### 2. Step 78 ??Toy causal rollout writer ?앹꽦 諛?self-test ?듦낵

- **?앹꽦 ?뚯씪**
  - `05_training/run_toy_causal_rollout.py`
  - `05_training/adapters/test_toy_causal_rollout_writer.py`

- **??븷**
  - Step 77??toy causal simulator瑜??ㅼ젣 rollout artifact濡???ν븳??
  - 媛?scenario / condition / seed 議고빀?????`raw_events.parquet`? `window_rollup.parquet`瑜??앹꽦?쒕떎.
  - run manifest瑜??④퍡 ?앹꽦?섏뿬 causal flag, source mode, row count, horizon ?뺣낫瑜?湲곕줉?쒕떎.

- **寃利?寃곌낵**
  - conditions: `A`, `A90`
  - seeds: `1`, `2`
  - scenario_count: `2`
  - raw_event_rows: `128`
  - window_rollup_rows: `8`
  - self-test PASS

- **?섎?**
  - toy simulator媛 硫붾え由??대???dynamics 寃利앹쓣 ?섏뼱, canonical evaluation???곌껐 媛?ν븳 ?뚯씪 ?곗텧臾쇱쓣 留뚮뱾 ???덇쾶 ?섏뿀??

#### 3. Step 79 ??Toy causal rollout??canonical KPI aggregator???곌껐

- **?앹꽦 ?뚯씪**
  - `05_training/adapters/test_toy_causal_canonical_kpi_integration.py`
  - `05_training/adapters/toy_causal_canonical_kpi_integration.md`

- **??븷**
  - Step 78??`window_rollup.parquet`瑜?湲곗〈 `canonical_kpi_aggregator.py --mode official_rollup` 寃쎈줈???곌껐?쒕떎.
  - 湲곗〈 Phase 1?먯꽌 ?뺤갑??canonical KPI 吏묎퀎 寃쎈줈瑜?Phase 2 toy causal output?먮룄 洹몃?濡??곸슜?????덈뒗吏 寃利앺븳??

- **寃利?寃곌낵**
  - conditions: `A`, `A90`
  - seeds: `1`, `2`
  - windows: `3`
  - time_bands: `peak`, `offpeak`, `night`
  - raw_event_rows: `192`
  - window_rollup_rows: `12`
  - `kpi_by_window.parquet`: 12 rows
  - `kpi_by_seed.parquet`: 4 rows
  - `kpi_by_time_band.parquet`: 12 rows
  - `kpi_overall.json` ?앹꽦 ?뺤씤
  - `causal_comparison_allowed = true` ?좎? ?뺤씤
  - shared KPI 6醫?寃利??듦낵
  - official_rollup smoke PASS

#### 4. ?꾩옱 ?섎?? ?쒓퀎

?대쾲 Step 77~79濡?泥섏쓬?쇰줈 ?꾨옒 寃쎈줈媛 ?대졇??

```text
action_t
-> causal simulator dynamics
-> state_{t+1}
-> raw_events.parquet / window_rollup.parquet
-> canonical_kpi_aggregator.py
-> kpi_by_window / kpi_by_seed / kpi_overall
```

?ㅻ쭔 ??寃곌낵???꾩쭅 toy simulator 湲곕컲?대떎. ?곕씪???ㅼ젣 ?쇰Ц ?깅뒫 二쇱옣?쇰줈 ?ъ슜?섏? ?딅뒗?? ?꾩옱 ?④퀎???섎????ㅼ쓬?쇰줈 ?쒗븳?쒕떎.

- causal adapter contract 寃利?- 理쒖냼 toy dynamics 寃利?- canonical KPI aggregation ?곌껐 寃利?- Phase 2 causal simulator architecture 吏꾩엯 寃利?
#### 5. ?ㅼ쓬 ?④퀎

?ㅼ쓬 ?④퀎??Step 80 ?댄썑 ?ㅼ쓬 ??以??섎굹??

1. **Step 81 ??Toy causal A/A90/A80/A70 full smoke matrix**
   - A-family 議곌굔 4媛??꾩껜瑜?toy causal simulator ?꾩뿉???ㅽ뻾?쒕떎.
   - seeds 1,2,3源뚯? ?뺤옣?섏뿬 12媛?run 援ъ“瑜?寃利앺븳??

2. **Step 82 ??Project log 諛?GitHub push/status review**
   - Step 77~80源뚯???蹂寃쎌쓣 而ㅻ컠?섍퀬 origin/main??push?쒕떎.
   - artifacts???ъ깮??媛?ν븯誘濡?而ㅻ컠 ??곸뿉???쒖쇅?쒕떎.

---

## 📅 2026-04-27
### Step 51~52 A-family rollout policy integration plan 및 policy-source selectable rollout writer 구축

오늘 작업에서는 Step 48~49에서 만든 MAPPO policy action source bridge와 policy metadata propagation 계약을 실제 A-family rollout writer에 연결하기 위한 통합 계획과 smoke writer를 구축했다.

핵심 목표는 다음과 같다.

- A/A90/A80/A70 조건에서 어떤 policy source mode를 허용할지 명확히 정의한다.
- mock/stub action과 neural MAPPO action을 옵션으로 분리한다.
- checkpoint가 없거나 validator를 통과하지 못하면 fallback 없이 STOP한다.
- Qwen이 섞인 checkpoint가 A-family MAPPO 경로에 들어오지 못하게 한다.
- rollout 산출물에 policy source metadata를 남긴다.
- smoke 결과와 actual 성능 주장 가능 결과를 명확히 구분한다.

#### 1. Step 51 — A-family rollout policy integration plan 생성

생성 파일:

- `05_training/policies/a_family_rollout_policy_integration_plan.md`
- `05_training/policies/a_family_rollout_policy_integration.py`
- `05_training/policies/test_a_family_rollout_policy_integration.py`

Step 51에서는 실제 rollout writer를 수정하기 전에 A-family 조건의 policy source 선택 규칙을 먼저 계약으로 고정했다.

A-family 조건:

```text
A
A90
A80
A70
```

공통 규칙:

```text
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
```

허용 policy source mode:

```text
mock_smoke
mappo_smoke
mappo_actual
```

각 mode의 의미:

```text
mock_smoke:
- 개발용 mock action
- checkpoint 불필요
- allow_mock=true를 명시해야만 허용
- policy_source = mock_policy
- 성능 주장 불가

mappo_smoke:
- neural MAPPO action source 경로 검증용
- checkpoint 필요
- checkpoint_validation_mode = smoke
- checkpoint_loaded = true 필요
- checkpoint_validator_ran = true 필요
- policy_source = mappo_policy
- actual 성능 주장 불가

mappo_actual:
- H200 trained checkpoint 전용
- checkpoint 필요
- checkpoint_validation_mode = actual
- trained_model = true 필요
- performance_claim_allowed = true 필요
- Qwen 비활성 필요
- actual policy claim 가능 후보
- causal claim은 causal simulator가 있을 때만 가능
```

Step 51에서 정한 source_mode 예:

```text
A + mock_smoke:
noncausal_A_mock_policy_smoke_v1

A + mappo_smoke:
noncausal_A_mappo_policy_smoke_v1

A + mappo_actual + noncausal simulator:
noncausal_A_mappo_policy_actual_v1

A + mappo_actual + causal simulator:
causal_A_mappo_policy_v1
```

self-test에서 확인한 expected failure:

```text
B2는 A-family 조건이 아니므로 거부
mappo_smoke인데 checkpoint 없음 → 거부
mappo_actual인데 validation_mode=smoke → 거부
mock_smoke인데 allow_mock=false → 거부
qwen_train=true → 거부
qwen_trigger_rate=0.5 → 거부
```

최종 결과:

```text
[OK] Step 51 A-family policy integration plan self-test PASS
[DONE] Step 51 A-family rollout policy integration plan complete.
```

해석:

Step 51은 A/A90/A80/A70 rollout writer가 mock, MAPPO smoke, MAPPO actual을 헷갈리지 않고 안전하게 선택하도록 만든 정책 통합 설계도다.

#### 2. Step 52 — A-family policy-source selectable rollout smoke writer 생성

생성 파일:

- `05_training/run_a_family_policy_rollout_smoke.py`
- `05_training/policies/test_a_family_policy_rollout_smoke.py`

Step 52에서는 Step 51의 통합 계획을 실제 실행 가능한 smoke rollout writer로 구현했다.

새 runner 옵션:

```text
--condition-id A|A90|A80|A70
--policy-source-mode mock_smoke|mappo_smoke|mappo_actual
--checkpoint-path <path>
--checkpoint-validation-mode smoke|actual
--allow-mock
--seed <seed>
--device cpu|cuda
--output-root <path>
--scenario-index <path>
--scenario-row-index <index>
```

writer 실행 흐름:

```text
scenario window
→ simulator.reset(seed, scenario_config)
→ obs 생성
→ policy_source_mode 확인
→ mock action 또는 MAPPOPolicyActionSource 선택
→ actions 생성
→ simulator.step(actions)
→ policy metadata 생성
→ window_rollup.parquet 저장
→ status/manifest/actions/policy_metadata 저장
```

생성 산출물:

```text
status.json
run_manifest.json
actions.json
policy_metadata.json
window_rollup.parquet
```

성공 케이스:

```text
A + mock_smoke + allow_mock
→ PASS
→ policy_source = mock_policy
→ source_mode = noncausal_A_mock_policy_smoke_v1

A + mappo_smoke + smoke checkpoint
→ PASS
→ policy_source = mappo_policy
→ source_mode = noncausal_A_mappo_policy_smoke_v1

A90 + mappo_smoke + smoke checkpoint
→ PASS
→ policy_source = mappo_policy
→ source_mode = noncausal_A90_mappo_policy_smoke_v1
```

의도적 실패 케이스:

```text
mock_smoke인데 allow_mock 없음
→ FAIL 정상
→ 실수로 mock을 쓰는 것 방지

mappo_smoke인데 checkpoint 없음
→ FAIL 정상
→ checkpoint 없는 placeholder fallback 방지

mappo_actual인데 smoke checkpoint 사용
→ FAIL 정상
→ trained_model=true, performance_claim_allowed=true 필요

qwen_train=True checkpoint 사용
→ FAIL 정상
→ A-family Qwen 비활성 조건 보호
```

최종 결과:

```text
[OK] Step 52 A-family policy rollout smoke self-test PASS
[DONE] Step 52 A-family policy-source selectable rollout writer complete.
```

#### 3. Step 52의 중요한 의미

Step 52는 A-family rollout writer를 실제 neural MAPPO action source와 연결하기 위한 첫 실행 가능한 다리다.

이제 아래가 가능하다.

```text
mock smoke rollout 생성
mappo smoke rollout 생성
A/A90 조건별 source_mode 구분
policy_metadata.json 저장
window_rollup.parquet에 policy metadata 포함
checkpoint 없는 mappo path 차단
Qwen-invalid checkpoint 차단
actual mode에서 smoke checkpoint 차단
```

단, 현재는 여전히 smoke 단계다.

```text
실제 H200 trained checkpoint 기반 성능 주장: 아직 아님
causal performance comparison: 아직 아님
```

#### 4. 현재까지의 누적 구조

```text
Step 40:
neural MAPPO inference adapter scaffold 생성

Step 41:
MAPPO checkpoint contract v1 및 validator 생성

Step 42:
neural inference runner 앞에 strict checkpoint validator 연결

Step 43:
Daegu energy proxy model v1 독립 모듈 생성

Step 44:
Step 41~43 project_log 기록

Step 45:
MAPPO checkpoint builder와 runner checkpoint contract preview 연결

Step 46:
actual checkpoint preflight script 생성

Step 47:
Step 45~46 project_log 기록

Step 48:
MAPPO policy action source bridge 생성

Step 49:
policy source metadata propagation 계약 생성

Step 50:
Step 48~49 project_log 기록

Step 51:
A-family rollout policy integration plan 생성

Step 52:
A-family policy-source selectable rollout smoke writer 생성
```

#### 5. 연구적으로 중요한 점

Step 51~52는 실험 오염을 줄이는 provenance control layer를 실제 rollout smoke writer까지 끌어내린 단계다.

방지하는 문제:

```text
mock action을 실제 MAPPO action으로 착각
checkpoint 없는 placeholder fallback
smoke checkpoint를 actual checkpoint로 사용
Qwen 개입 checkpoint가 A-family MAPPO 경로에 섞임
source_mode 누락
policy metadata 누락
noncausal replay 결과를 causal claim으로 오해
```

이제 rollout 산출물에는 다음 정보가 남는다.

```text
어떤 condition인가?
어떤 policy_source_mode인가?
policy_source가 mock인가 mappo인가?
checkpoint를 로드했는가?
validator를 실행했는가?
mock_action_used인가?
placeholder_fallback_used인가?
Qwen trigger rate는 0인가?
energy proxy model version은 무엇인가?
actual_policy_claim_ready인가?
causal_policy_claim_ready인가?
```

#### 6. 현재 판정

```text
Step 51: COMPLETED
Step 52: COMPLETED
A-family policy integration plan: READY
A-family policy-source selectable rollout smoke writer: READY
mappo_smoke rollout path: READY
mappo_actual rollout path: 구조상 준비, 실제 H200 checkpoint 필요
actual H200 trained checkpoint: NOT YET
actual performance claim: NOT YET
causal performance claim: NOT YET
```

#### 7. 다음 단계 후보

다음 단계는 아래 순서가 적절하다.

1. Step 53 기록 커밋 및 GitHub push
2. Step 54 window_rollup policy metadata columns 검증기 생성
3. Step 55 canonical KPI aggregator가 policy metadata/source_mode를 보존하는지 smoke 검증
4. Step 56 A/A90/A80/A70 다조건 mappo_smoke rollout matrix 생성
5. Step 57 H200 actual checkpoint 생성 전 최종 checklist 작성

---

## 📅 2026-04-27
### Step 48~49 MAPPO policy action source bridge 및 policy metadata propagation 계약 구축

오늘 작업에서는 Step 46까지 구축한 checkpoint preflight pipeline 위에, 실제 rollout writer가 neural MAPPO action을 안전하게 받아올 수 있는 action source bridge와 policy metadata propagation 계약을 추가했다.

핵심 목표는 다음과 같다.

- A/A90/A80/A70 rollout writer가 mock/stub action 대신 neural MAPPO action을 같은 인터페이스로 받을 수 있게 준비한다.
- checkpoint가 없거나 validator를 통과하지 못하면 fallback 없이 STOP한다.
- Qwen이 섞인 checkpoint는 Experiment A-family 경로에서 차단한다.
- rollout/KPI 전 단계까지 policy source metadata를 같은 필드명으로 전달할 수 있게 표준화한다.
- 실제 paper-level claim 가능 여부를 metadata로 명확히 구분한다.

#### 1. Step 48 — MAPPO policy action source bridge 생성

생성 파일:

- `05_training/policies/mappo_policy_action_source.py`
- `05_training/policies/test_mappo_policy_action_source.py`

Step 48에서는 rollout writer가 직접 neural adapter와 validator를 모두 다루지 않아도 되도록 `MAPPOPolicyActionSource`를 만들었다.

기존 구조의 문제:

```text
rollout writer가 neural adapter를 직접 다루면
checkpoint validation, strict loading, Qwen 차단, metadata 기록이 분산될 위험이 있음
```

Step 48 이후 목표 구조:

```text
obs 생성
→ MAPPOPolicyActionSource.select_actions(obs)
→ actions 반환
→ simulator.step(actions)
→ metadata에 policy_source=mappo_policy 기록
```

`MAPPOPolicyActionSource`의 핵심 안전 규칙:

```text
checkpoint_path 없음 → STOP
checkpoint validator 실패 → STOP
actual mode에서 smoke checkpoint → STOP
qwen_train=true 또는 qwen_inference=true → STOP
qwen_trigger_rate != 0.0 → STOP
placeholder fallback 없음
mock action 없음
```

주요 metadata:

```text
policy_source = mappo_policy
policy_action_source_version = mappo_policy_action_source_v1
policy_action_source_mode = mappo_policy_actual_v1 또는 mappo_policy_smoke_v1
checkpoint_validator_ran = true
checkpoint_loaded = true
trained_model
performance_claim_allowed
placeholder_fallback_used = false
mock_action_used = false
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
reward_version = mappo_reward_v1
energy_proxy_model_version = daegu_energy_proxy_v1
```

검증 결과:

- smoke checkpoint + smoke mode:
  - action 생성 PASS
  - checkpoint validator 실행 확인
  - checkpoint loaded 확인
  - mock/placeholder fallback 미사용 확인
  - Qwen 비활성 확인
  - energy proxy model version 확인
- checkpoint 없음:
  - expected failure
  - fallback 없이 STOP 확인
- smoke checkpoint + actual mode:
  - expected failure
  - trained_model/performance_claim 조건 미충족으로 STOP 확인
- qwen_train=True checkpoint:
  - expected failure
  - Experiment A 조건 위반으로 STOP 확인

최종 출력:

```text
[OK] Step 48 MAPPO policy action source self-test PASS
[DONE] Step 48 MAPPO policy action source bridge complete.
```

해석:

Step 48은 “rollout writer가 나중에 MAPPO neural action을 안전하게 받아오는 단일 출입구”를 만든 단계다. 이 출입구는 checkpoint 검증, strict load, Qwen 차단, metadata 생성을 함께 수행한다.

#### 2. Step 49 — policy source metadata propagation 계약 생성

생성 파일:

- `05_training/policies/policy_source_metadata.py`
- `05_training/policies/test_policy_source_metadata.py`

Step 49에서는 Step 48의 action source가 만든 metadata를 rollout row, window rollup, canonical KPI 직전까지 같은 이름과 같은 의미로 전달할 수 있도록 표준화했다.

핵심 목적:

```text
정책 action이 진짜 MAPPO에서 왔는가?
mock/stub action이 아닌가?
checkpoint 검증을 거쳤는가?
Qwen이 섞이지 않았는가?
energy proxy 계약이 맞는가?
actual performance claim이 가능한 상태인가?
causal performance claim이 가능한 상태인가?
```

위 질문에 대한 답을 rollout metadata 필드로 남기는 것이 목적이다.

표준 metadata 필드:

```text
policy_metadata_version
condition_id
source_mode
policy_source
policy_action_source_version
policy_action_source_mode
checkpoint_path
checkpoint_validation_mode
checkpoint_validator_ran
checkpoint_loaded
trained_model
performance_claim_allowed
placeholder_fallback_used
mock_action_used
qwen_train
qwen_inference
qwen_trigger_rate
reward_version
energy_proxy_model_version
k_dist_kwh_per_m
k_acc_kwh_per_event
k_idle_kwh_per_sec
actual_policy_claim_ready
causal_policy_claim_ready
```

`actual_policy_claim_ready`가 true가 되려면:

```text
policy_source = mappo_policy
policy_action_source_version = mappo_policy_action_source_v1
checkpoint_validator_ran = true
checkpoint_loaded = true
trained_model = true
performance_claim_allowed = true
placeholder_fallback_used = false
mock_action_used = false
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
reward_version = mappo_reward_v1
energy_proxy_model_version = daegu_energy_proxy_v1
K_DIST/K_ACC/K_IDLE 상수 일치
```

`causal_policy_claim_ready`가 true가 되려면 위 조건에 더해:

```text
causal_simulator = true
source_mode가 causal_* 로 시작
```

해야 한다.

즉, 실제 trained checkpoint만 있어도 causal claim은 불가능하다.  
causal simulator 기반 rollout까지 필요하다.

검증 결과:

- smoke checkpoint 기반 metadata는 구조적으로 valid.
- 하지만 `trained_model=false`, `performance_claim_allowed=false`이므로 actual claim-ready는 false.
- noncausal smoke이므로 causal claim-ready도 false.
- `require_actual_ready=True`를 걸면 smoke metadata는 실패해야 하며 실제로 실패.
- `mock_action_used=true`는 실패.
- `placeholder_fallback_used=true`는 실패.
- `qwen_trigger_rate != 0.0`은 실패.
- energy proxy model mismatch는 실패.
- causal source_mode인데 causal readiness가 false인 경우 실패.
- batch validation에서 bad row 포함 시 invalid 처리 확인.

최종 출력:

```text
[OK] Step 49 policy source metadata propagation self-test PASS
[DONE] Step 49 policy source metadata propagation contract complete.
```

#### 3. Step 48~49 이후 구조

현재 actual MAPPO inference 준비 경로는 다음과 같다.

```text
H200 actual checkpoint 생성
→ preflight_mappo_checkpoint.py 검증
→ MAPPOPolicyActionSource 생성
→ obs 입력
→ checkpoint validator 재확인
→ neural adapter strict load
→ actions 생성
→ policy_source_metadata로 rollout-safe metadata 정규화
→ rollout row/window rollup/canonical KPI 전 단계로 전달
```

#### 4. 연구적으로 중요한 의미

Step 48~49는 단순 코드 편의 기능이 아니다.  
이 단계는 논문 결과 오염을 막는 provenance layer다.

방지하는 문제:

- mock action이 실제 MAPPO action처럼 섞이는 문제
- placeholder fallback이 조용히 사용되는 문제
- Qwen이 켜진 checkpoint가 A 조건 결과에 섞이는 문제
- checkpoint 검증 없이 action이 생성되는 문제
- smoke checkpoint 결과가 actual claim으로 해석되는 문제
- noncausal replay 결과가 causal performance claim으로 오해되는 문제
- energy proxy 상수/버전이 다른 결과가 같은 실험으로 묶이는 문제

이제 rollout 결과에는 다음 질문에 대한 근거가 남는다.

```text
이 action은 어디서 왔는가?
이 checkpoint는 검증됐는가?
실제 학습된 checkpoint인가?
성능 주장에 쓸 수 있는가?
causal claim이 가능한가?
Qwen이 개입했는가?
energy proxy 계약이 맞는가?
```

#### 5. 현재까지의 누적 상태

```text
Step 40:
neural MAPPO inference adapter scaffold 생성

Step 41:
MAPPO checkpoint contract v1 및 validator 생성

Step 42:
neural inference runner 앞에 strict checkpoint validator 연결

Step 43:
Daegu energy proxy model v1 독립 모듈 생성

Step 44:
Step 41~43 project_log 기록

Step 45:
MAPPO checkpoint builder와 runner checkpoint contract preview 연결

Step 46:
actual checkpoint preflight script 생성

Step 47:
Step 45~46 project_log 기록

Step 48:
MAPPO policy action source bridge 생성

Step 49:
policy source metadata propagation 계약 생성
```

#### 6. 현재 판정

- Step 48: COMPLETED
- Step 49: COMPLETED
- MAPPO action source bridge: READY
- policy metadata propagation contract: READY
- actual H200 trained checkpoint: NOT YET
- actual performance claim: NOT YET
- causal performance claim: NOT YET
- A/A90/A80/A70 rollout writer neural action integration: NOT YET

#### 7. 다음 단계 후보

다음 단계는 아래 순서가 적절하다.

1. Step 50 기록 커밋 및 GitHub push
2. Step 51 A/A90/A80/A70 rollout writer integration plan 작성
3. Step 52 A-family rollout writer에 policy action source 선택 옵션 추가
4. Step 53 policy metadata columns를 window_rollup/canonical KPI 입력 전 단계에 연결
5. Step 54 H200 actual checkpoint 생성 전 최종 preflight checklist 작성

---

## 📅 2026-04-27
### Step 45~46 MAPPO checkpoint builder 및 actual checkpoint preflight pipeline 구축

오늘 작업에서는 Step 41~43에서 확정한 MAPPO checkpoint contract, strict validator, Daegu energy proxy model을 실제 runner/checkpoint 저장 경로와 preflight 검증 경로에 연결했다.

핵심 목표는 다음과 같다.

- H200에서 실제 MAPPO checkpoint가 생성될 때 반드시 Step 41 계약 형식으로 저장되도록 준비한다.
- `mappo_runner.py`가 smoke checkpoint stub뿐 아니라 checkpoint contract preview를 함께 남기도록 한다.
- 실제 `best.pt`가 생겼을 때 한 번의 preflight 명령으로 actual checkpoint 사용 가능 여부를 검증할 수 있게 한다.
- fake/smoke/Qwen-invalid checkpoint가 actual inference 경로에 들어오지 못하게 한다.

#### 1. Step 45 — MAPPO checkpoint builder 및 runner checkpoint contract preview 연결

생성/수정 파일:

- `05_training/policies/mappo_checkpoint_builder.py`
- `05_training/policies/test_mappo_checkpoint_builder.py`
- `05_training/mappo_runner.py`

Step 45에서는 checkpoint 저장 포맷 생성기를 추가했다.

`mappo_checkpoint_builder.py`의 역할:

- Step 41의 checkpoint contract v1에 맞는 metadata 생성
- Step 43의 Daegu energy proxy model v1 상수 포함
- `model_state_dict` 포함 checkpoint payload 생성
- `torch.save` 가능한 checkpoint 저장 함수 제공
- runner smoke용 `checkpoint_contract_preview.json` 생성

주요 고정 값:

```text
artifact_version = mappo_policy_checkpoint_v1
contract_version = mappo_checkpoint_contract_v1
condition_id = A
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
policy_architecture = ActorCriticMLP
actor_obs_dim = 16
critic_obs_dim = 64
action_dim = 2
hidden_dim = 128
reward_version = mappo_reward_v1
rollout_schema_version = rollout_schema_v1
policy_interface_version = mappo_policy_interface_v1
action_space_version = bus_control_action_v1
observation_space_version = urbanbus_observation_v1
energy_proxy_model_version = daegu_energy_proxy_v1
energy_proxy_unit = kwh_equivalent
k_dist_kwh_per_m = 0.0012
k_acc_kwh_per_event = 0.1800
k_idle_kwh_per_sec = 0.0080
```

`mappo_runner.py` 변경 사항:

기존:

```text
checkpoint_stub.json만 저장
```

변경 후:

```text
checkpoint_stub.json 저장
checkpoint_contract_preview.json 저장
```

`checkpoint_contract_preview.json`은 실제 torch checkpoint가 아니다.  
`model_state_dict`도 없다.  
대신 H200에서 실제 checkpoint를 만들 때 반드시 만족해야 할 metadata 계약을 seed별 run directory에 남긴다.

주의 및 수정 사항:

초기 self-test에서 `run_experiment_A_stub.py`에 존재하지 않는 `--contract`, `--run-root` 인자를 넘겨 실패했다. 이는 checkpoint builder 본체 문제가 아니라 테스트 호출 방식 문제였다.

수정:

- CLI 호출 대신 `MAPPOExperimentRunner.write_checkpoint_stub()`를 직접 호출하도록 테스트 수정.
- 이후 runner preview 파일 생성 여부를 직접 확인하도록 변경.

추가로 `extra_metadata`에 `condition_id`, `shared_policy`, `ctde_enabled` 같은 checkpoint contract 정식 key와 겹치는 이름을 넣어 builder가 preview 생성을 거부할 수 있는 문제가 있었다.

수정:

```text
condition_id      → runner_condition_id
shared_policy     → runner_shared_policy
ctde_enabled      → runner_ctde_enabled
```

최종 검증 결과:

- builder가 checkpoint payload 생성 PASS
- smoke checkpoint 저장 PASS
- validator smoke mode PASS
- `model_state_dict` neural adapter load 가능 확인
- `checkpoint_contract_preview.json` 생성 확인
- preview 안에 `mappo_checkpoint_contract_v1`, `daegu_energy_proxy_v1`, Qwen 비활성 조건 포함 확인
- 최종 self-test PASS

최종 출력:

```text
[OK] Step 45 MAPPO checkpoint builder self-test PASS
[DONE] Step 45 MAPPO runner checkpoint contract format complete.
```

#### 2. Step 46 — actual checkpoint preflight script 생성

생성 파일:

- `05_training/policies/preflight_mappo_checkpoint.py`
- `05_training/policies/test_preflight_mappo_checkpoint.py`

Step 46에서는 나중에 H200에서 실제 `best.pt`가 생겼을 때 사용할 preflight 검증 스크립트를 만들었다.

`preflight_mappo_checkpoint.py`가 수행하는 일:

1. checkpoint 파일 존재 확인
2. Step 41 checkpoint validator 실행
3. actual/smoke mode에 따른 조건 확인
4. Step 42 neural inference strict runner 실행
5. checkpoint가 neural adapter에 실제로 로드되는지 확인
6. Qwen 비활성 조건 확인
7. preflight manifest 저장

실제 H200 checkpoint 사용 예:

```powershell
python .\05_training\policies\preflight_mappo_checkpoint.py `
  --checkpoint .\artifacts\experiment_A_v1\checkpoints\best.pt `
  --mode actual `
  --device cuda `
  --seed 1
```

Preflight 산출물:

- `preflight_checkpoint_validation_report.json`
- `neural_strict_load/seed_xxx/status.json`
- `preflight_manifest.json`

검증 모드:

```text
smoke mode:
- boundary validation용
- trained_model=false 허용
- performance_claim_allowed=false 유지
- actual performance claim 금지

actual mode:
- H200 trained checkpoint 전용
- trained_model=true 필요
- performance_claim_allowed=true 필요
- fake/mock/stub/smoke marker 금지
```

Step 46 self-test 구성:

- smoke checkpoint를 smoke mode로 검사 → PASS
- 같은 smoke checkpoint를 actual mode로 검사 → FAIL이 맞음
- qwen_train=True checkpoint를 smoke mode로 검사 → FAIL이 맞음

중간 FAIL 해석:

중간의 `[FAIL]`은 오류가 아니라 의도된 차단 검증이다.

실제로 로그에서 다음이 확인되었다.

```text
smoke checkpoint + smoke mode:
  preflight PASS
  trained_model = False
  performance_claim_allowed = False
  actual_claim_allowed = False

smoke checkpoint + actual mode:
  FAIL
  reason = actual mode requires trained_model=true
           actual mode requires performance_claim_allowed=true

qwen_train=True checkpoint + smoke mode:
  FAIL
  reason = qwen_train mismatch: expected False, got True
```

최종 검증 결과:

```text
[OK] Step 46 checkpoint preflight self-test PASS
[DONE] Step 46 actual checkpoint preflight script complete.
```

#### 3. 현재 구조의 의미

Step 45~46 이후 checkpoint 흐름은 다음과 같다.

```text
mappo_runner.py
→ checkpoint contract preview 기록

H200 actual training
→ mappo_checkpoint_builder.py 형식으로 checkpoint 저장

preflight_mappo_checkpoint.py
→ checkpoint contract 검증
→ actual mode 검증
→ neural adapter strict load 검증
→ Qwen 비활성 확인
→ energy proxy 계약 확인
→ preflight manifest 저장

preflight PASS
→ actual neural MAPPO inference에 투입 가능
```

#### 4. 현재까지의 누적 상태

```text
Step 40:
neural MAPPO inference adapter scaffold 생성

Step 41:
MAPPO checkpoint contract v1 및 validator 생성

Step 42:
neural inference runner 앞에 strict checkpoint validator 연결

Step 43:
Daegu energy proxy model v1 독립 모듈 생성

Step 44:
Step 41~43 project_log 기록

Step 45:
MAPPO checkpoint builder와 runner checkpoint contract preview 연결

Step 46:
actual checkpoint preflight script 생성
```

#### 5. 연구적으로 중요한 점

이번 구조는 실제 성능 결과를 만들기 전, 실험 오염을 막는 안전장치다.

방지하는 문제:

- smoke checkpoint를 actual checkpoint로 착각하는 문제
- Qwen 개입 checkpoint가 Experiment A 결과에 섞이는 문제
- energy proxy 상수가 다른 checkpoint가 섞이는 문제
- reward/policy/action/observation schema version이 다른 checkpoint가 섞이는 문제
- neural adapter가 로드할 수 없는 checkpoint를 뒤늦게 발견하는 문제

이제 실제 H200 checkpoint가 생기면 단순히 파일만 확인하는 것이 아니라, 다음 기준을 모두 만족해야 한다.

```text
trained_model = true
performance_claim_allowed = true
condition_id = A
qwen_train = false
qwen_inference = false
qwen_trigger_rate = 0.0
model_state_dict load OK
energy_proxy_model_version = daegu_energy_proxy_v1
K_DIST/K_ACC/K_IDLE 상수 일치
preflight PASS
```

#### 6. 현재 판정

- Step 45: COMPLETED
- Step 46: COMPLETED
- actual H200 trained checkpoint: NOT YET
- actual performance claim: NOT YET
- preflight pipeline: READY
- smoke/fake checkpoint boundary validation: PASS
- Qwen-invalid checkpoint rejection: PASS

#### 7. 다음 단계 후보

다음 단계는 아래 순서가 적절하다.

1. Step 47 기록 커밋 및 GitHub push
2. Step 48 A/A90/A80/A70 rollout writer에서 mock/stub action을 neural adapter action으로 교체 준비
3. Step 49 actual policy source metadata propagation 점검
4. Step 50 H200 actual checkpoint training 전 preflight checklist 작성
5. Step 51 H200 서버에서 trained checkpoint 생성 후 `preflight_mappo_checkpoint.py --mode actual` 실행

---

## 📅 2026-04-27
### Step 41~43 MAPPO checkpoint contract, strict inference gate, Daegu energy proxy model 정착

오늘 작업에서는 Step 40에서 만든 neural MAPPO inference adapter scaffold 위에, 실제 H200 학습 checkpoint가 들어오기 전에 반드시 필요한 안전 계약과 검증 장치를 추가했다. 또한 대구버스 에너지 프록시 K 상수를 실제 계산 가능한 독립 모듈로 분리하였다.

이번 작업의 핵심은 다음과 같다.

- 아무 `.pt` 파일이나 actual MAPPO checkpoint로 인정하지 않는다.
- Qwen이 섞인 checkpoint는 Experiment A 조건에서 차단한다.
- fake/mock/smoke checkpoint는 actual mode에서 절대 통과하지 못하게 한다.
- energy proxy model version과 K 상수를 checkpoint 계약에 고정한다.
- energy proxy는 바로 reward에 넣지 않고, 먼저 KPI-side diagnostic lens로 검증한다.

#### 1. Step 41 — MAPPO checkpoint contract v1 정의 및 validator 생성

생성 파일:

- `05_training/policies/mappo_checkpoint_contract.md`
- `05_training/policies/validate_mappo_checkpoint.py`
- `05_training/policies/test_validate_mappo_checkpoint.py`

Step 41에서는 실제 MAPPO checkpoint로 인정할 수 있는 `.pt` 파일의 필수 조건을 계약으로 고정했다.

필수 조건:

- `artifact_version = mappo_policy_checkpoint_v1`
- `contract_version = mappo_checkpoint_contract_v1`
- `condition_id = A`
- `qwen_train = false`
- `qwen_inference = false`
- `qwen_trigger_rate = 0.0`
- `policy_architecture = ActorCriticMLP`
- `actor_obs_dim = 16`
- `critic_obs_dim = 64`
- `action_dim = 2`
- `hidden_dim = 128`
- `shared_policy = true`
- `ctde_enabled = true`
- `model_state_dict` 존재
- `trained_model` 명시
- `performance_claim_allowed` 명시
- `reward_version = mappo_reward_v1`
- `rollout_schema_version = rollout_schema_v1`
- `policy_interface_version = mappo_policy_interface_v1`
- `action_space_version = bus_control_action_v1`
- `observation_space_version = urbanbus_observation_v1`
- `energy_proxy_model_version = daegu_energy_proxy_v1`
- `energy_proxy_unit = kwh_equivalent`
- `k_dist_kwh_per_m = 0.0012`
- `k_acc_kwh_per_event = 0.1800`
- `k_idle_kwh_per_sec = 0.0080`

중요한 판정 원칙:

- smoke mode에서는 fake checkpoint가 boundary validation 용도로만 통과 가능하다.
- actual mode에서는 `trained_model = true`와 `performance_claim_allowed = true`가 반드시 필요하다.
- actual mode에서는 fake/mock/stub/smoke/placeholder/scaffold marker가 있으면 실패한다.
- Qwen 관련 값이 Experiment A 조건과 다르면 실패한다.
- energy proxy K 상수가 하나라도 다르면 실패한다.
- `model_state_dict`가 neural MAPPO adapter 구조에 로드되지 않으면 실패한다.

검증 결과:

- fake smoke checkpoint는 smoke mode에서 PASS.
- 같은 fake checkpoint는 actual mode에서 의도적으로 FAIL.
- `qwen_train = true` checkpoint는 FAIL.
- energy constant mismatch checkpoint는 FAIL.
- 최종 self-test PASS.

해석:

중간의 FAIL은 오류가 아니라 validator가 위험한 checkpoint를 제대로 거부했음을 의미한다. 즉, Step 41은 “checkpoint 형식과 실험 조건을 지키지 않는 파일을 actual MAPPO checkpoint로 인정하지 않는 안전장치”를 만든 단계다.

#### 2. Step 42 — neural inference smoke runner에 strict checkpoint validator 연결

수정/생성 파일:

- `05_training/run_experiment_A_neural_inference_smoke.py`
- `05_training/policies/test_neural_inference_strict_checkpoint.py`

Step 42에서는 Step 40의 neural MAPPO inference smoke runner와 Step 41의 checkpoint validator를 연결했다.

기존 Step 40 흐름:

```text
checkpoint_path 입력
→ neural adapter가 직접 torch.load 시도
```

Step 42 이후 흐름:

```text
checkpoint_path 입력
→ Step 41 validator 실행
→ validator PASS
→ neural adapter checkpoint 로드
→ action 생성
```

validator가 실패하면:

```text
checkpoint_path 입력
→ validator FAIL
→ 즉시 STOP
→ neural adapter 로드 안 함
→ placeholder fallback 없음
```

새 CLI 옵션:

- `--checkpoint-path`
- `--require-checkpoint`
- `--checkpoint-validation-mode actual|smoke`
- `--checkpoint-validation-report`
- `--skip-checkpoint-validator`

주의:

- `--skip-checkpoint-validator`는 developer escape hatch이며 actual claim에는 사용 금지.
- `actual` mode는 H200 trained checkpoint 전용.
- `smoke` mode는 boundary test 전용.

검증 결과:

- fake smoke checkpoint + smoke mode:
  - validator ran
  - checkpoint loaded
  - SMOKE PASS
- fake smoke checkpoint + actual mode:
  - trained_model/performance_claim/fake marker 이유로 FAIL
- qwen_train=True checkpoint + smoke mode:
  - Experiment A 조건 위반으로 FAIL
- 최종 self-test PASS.

해석:

Step 42는 neural MAPPO inference adapter 앞에 checkpoint 검문소를 세운 단계다. 이제 실제 H200 checkpoint가 들어와도 먼저 계약 검증을 통과해야만 inference 경로로 진입할 수 있다.

#### 3. Step 43 — Daegu energy proxy model v1 생성

생성 파일:

- `05_training/rewards/__init__.py`
- `05_training/rewards/energy_proxy_model_v1.py`
- `05_training/rewards/energy_proxy_config_daegu_v1.yaml`
- `05_training/rewards/README_energy_proxy_model.md`
- `05_training/rewards/test_energy_proxy_model_v1.py`

Step 43에서는 Step 41 checkpoint contract에 고정한 대구형 energy proxy 상수를 실제 계산 가능한 독립 모듈로 구현했다.

확정 상수:

```text
K_DIST = 0.0012 kWh/m
K_ACC  = 0.1800 kWh/event
K_IDLE = 0.0080 kWh/sec
```

계산식:

```text
energy_kwh_equiv =
  0.0012 * distance_m
  + 0.1800 * acceleration_event_count
  + 0.0080 * hold_seconds
```

승객 1인당 에너지 프록시:

```text
energy_proxy_per_passenger =
  energy_kwh_equiv / max(passenger_served_count, epsilon)
```

모듈 기능:

- `compute_energy_kwh_equiv`
- `compute_energy_proxy_per_passenger`
- `compute_energy_proxy_from_components`
- `compute_energy_proxy_for_record`
- `compute_energy_proxy_for_records`
- `summarize_energy_proxy_records`
- `expected_contract_fields`
- `validate_config_constants`

검증 결과:

- 기본 K 상수 검증 PASS.
- 0 energy case PASS.
- 샘플 계산식 PASS.
- record 기반 계산 PASS.
- batch summary PASS.
- 승객 수 0명 denominator epsilon 보호 PASS.
- 음수 입력 방어 PASS.
- 잘못된 K 상수 config 거부 PASS.
- 최종 self-test PASS.

중요 해석:

Daegu energy proxy model v1은 아직 MAPPO reward에 직접 편입하지 않았다. 현재 위치는 KPI-side diagnostic lens다.

즉, 먼저 B0/B1/B2/A rollout에서 에너지 프록시가 상식적인 범위로 계산되는지 확인하고, 정책 간 trade-off가 말이 되는지 검증한다. 사용할 만하다고 판정되면 이후 reward shaping 후보로 승격한다.

#### 4. 현재 연구 흐름상 의미

Step 41~43 이후의 구조는 다음과 같다.

```text
Step 40:
neural MAPPO inference adapter scaffold 생성

Step 41:
actual MAPPO checkpoint 인정 기준 정의

Step 42:
neural inference runner 앞에 checkpoint validator 연결

Step 43:
Daegu energy proxy model v1을 독립 계산 모듈로 구현
```

이로써 actual inference 준비 경로는 다음처럼 강화되었다.

```text
H200에서 trained MAPPO checkpoint 생성
→ checkpoint contract v1 만족 여부 검증
→ qwen 비활성 조건 확인
→ reward/rollout/policy/action/observation/energy model version 확인
→ model_state_dict 로드 가능성 확인
→ validator PASS인 checkpoint만 neural inference adapter에 진입
```

#### 5. 논문/실험 방어 관점의 의미

이번 구조를 통해 다음 위험을 줄였다.

- smoke checkpoint를 실제 결과로 오인하는 문제
- Qwen 개입 checkpoint가 A 조건 결과에 섞이는 문제
- reward version 또는 energy proxy version이 다른 checkpoint가 섞이는 문제
- observation/action dimension mismatch로 인한 잘못된 inference
- 에너지 프록시를 검증 없이 reward에 바로 넣어 모델이 이상하게 학습되는 문제

특히 energy proxy는 “대기시간은 줄었지만 에너지를 과도하게 쓴 것은 아닌가?”라는 리뷰어 질문에 대응하기 위한 핵심 방어 지표가 된다.

#### 6. 현재 판정

- Step 41: COMPLETED
- Step 42: COMPLETED
- Step 43: COMPLETED
- Actual H200 trained checkpoint: NOT YET
- Actual performance claim: NOT YET
- Energy proxy reward integration: NOT YET
- Energy proxy KPI-side validation: READY

#### 7. 다음 단계 후보

다음 단계는 아래 순서가 적절하다.

1. Step 44 기록 커밋 및 GitHub push
2. Step 45 `mappo_runner.py` checkpoint 저장 포맷을 checkpoint contract v1에 맞춤
3. Step 46 actual checkpoint preflight script 생성
4. Step 47 A/A90/A80/A70 rollout writer에서 mock/stub action을 neural adapter action으로 교체 준비
5. Step 48 H200 서버 actual training checkpoint 생성 후 strict actual-mode validation

---

## 📅 2026-04-26
### Step 40 actual neural MAPPO inference adapter 자리 생성

오늘 작업에서는 Step 39까지 완료된 actual-inference preparation progress 위에, Step 40으로 **actual neural MAPPO inference adapter**가 들어갈 자리를 만들었다.  
이번 단계의 목표는 아직 학습된 MAPPO 체크포인트를 붙이는 것이 아니라, 향후 실제 신경망 정책 체크포인트가 들어왔을 때 같은 인터페이스로 추론을 수행할 수 있는 **정식 어댑터 껍데기와 smoke 검증 경로**를 먼저 고정하는 것이다.

#### 1. 시작 상태
- Step 39까지 완료.
- `project_log.md`에 Step 32~39 actual-inference preparation progress 기록 완료.
- GitHub main 최신 기준 커밋:
  - `dfd24db Document MAPPO actual-inference preparation progress`
- 다음 목표:
  - Step 40부터 actual neural MAPPO inference adapter 자리 만들기.

#### 2. Step 40 신규 파일 생성
- 신규 파일:
  - `05_training/policies/mappo_neural_inference_adapter.py`
  - `05_training/run_experiment_A_neural_inference_smoke.py`
- 보조 생성/확인:
  - `05_training/policies/__init__.py`

#### 3. `mappo_neural_inference_adapter.py` 역할
- MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) 정책의 실제 추론 어댑터 자리 생성.
- `NeuralMAPPOInferenceAdapter` 클래스 추가.
- `ActorCriticMLP` 기반 actor-critic 구조 추가.
- 현재는 checkpoint가 없어도 smoke test가 가능하도록 random-initialized policy를 허용.
- 향후 실제 학습 checkpoint가 생기면 `--checkpoint-path`로 로드 가능하도록 설계.
- action mask, active bus mask, agent_ids를 모두 검증.
- Qwen 개입률은 [A] 조건 기준으로 항상 `0.0` 유지.

#### 4. `run_experiment_A_neural_inference_smoke.py` 역할
- Experiment A 계약 검증:
  - `condition_id = A`
  - `qwen_train = false`
  - `qwen_inference = false`
  - `evaluation_horizon_minutes = 30`
- `HistoricalReplayAdapter`를 통해 관측값을 받아 neural MAPPO adapter에 전달.
- adapter가 action을 생성하고, 이를 simulator adapter의 `step()`에 전달하는 경로를 검증.
- 산출물 생성 위치:
  - `artifacts/experiment_A_v1/neural_inference_smoke/seed_001/status.json`
  - `artifacts/experiment_A_v1/neural_inference_smoke/seed_001/actions_preview.json`
  - `artifacts/experiment_A_v1/neural_inference_smoke/seed_001/neural_inference_manifest.json`

#### 5. 중요한 해석
- 이번 Step 40은 **성능 평가 단계가 아니다.**
- 현재 HistoricalReplayAdapter는 비인과적 replay adapter이므로 action이 다음 상태를 실제로 바꾸지 않는다.
- 따라서 이번 산출물은 “actual neural MAPPO inference adapter가 들어갈 경로가 정상 배선되었는가”를 확인하는 smoke validation이다.
- 실제 causal performance comparison은 Phase 2 causal simulator adapter 이후에 가능하다.

#### 6. 현재 상태 판정
- Step 40 범위:
  - neural MAPPO inference adapter 자리 생성
  - checkpoint optional loading 구조 생성
  - Experiment A 계약 검증 연결
  - adapter → action 생성 → simulator step 호출 경로 smoke 가능
- 상태:
  - **Step 40 scaffold ready**
  - **actual checkpoint integration pending**
  - **causal evaluation not yet supported**

#### 7. 다음 단계
1. Step 40 스크립트 실행 결과 확인:
   - `SMOKE PASS`
   - `status.json`
   - `actions_preview.json`
   - `neural_inference_manifest.json`
2. Git 상태 확인:
   - `git status`
3. Step 40 파일 커밋:
   - 권장 커밋 메시지:
     - `Add neural MAPPO inference adapter smoke path`
4. 다음 Step 41 후보:
   - 실제 MAPPO policy checkpoint contract 정의
   - checkpoint 저장 키 표준화
   - neural adapter와 기존 `mappo_runner.py` 연결
   - A 조건 rollout writer에서 stub policy를 neural adapter로 교체하는 준비

---|---:|---:|
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
## 📅 2026-04-27
### H200 MAPPO training 준비 게이트 Step 66~70 완료

오늘 작업에서는 실제 H200 서버에서 MAPPO 학습을 시작하기 전 필요한 학습 준비 게이트를 추가로 정리했다. Step 58~62가 실제 checkpoint 도착 이후의 preflight, 실행 계획, dry-run, arrival workflow를 다루는 체계였다면, Step 66~70은 그 checkpoint를 만들기 위한 H200 training side의 계약과 검증 체계를 고정한 작업이다.

#### 1. Step 66 — H200 MAPPO training runbook draft 작성
- **생성 파일**
  - `05_training/policies/H200_mappo_training_runbook.md`
  - `05_training/policies/test_h200_mappo_training_runbook.py`
- **핵심 기능**
  - 실제 H200에서 MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) 학습을 시작하기 위한 절차서를 작성했다.
  - H200 checkout, Python environment, CUDA (Compute Unified Device Architecture=통합 병렬 연산 장치 아키텍처) 확인, local validation, input artifact, output folder 구조, post-training validation 절차를 문서화했다.
  - `best_mappo.pt`가 단독으로 actual artifact가 되는 것이 아니라, 환경 리포트·학습 설정·명령 기록·checkpoint manifest·검증 리포트와 함께 관리되어야 함을 명시했다.
- **중요 제한**
  - Step 66은 실제 학습 구현이 아니라 H200 학습 운영 절차서다.
  - 이 단계만으로 trained checkpoint가 생성되는 것은 아니다.

#### 2. Step 67 — H200 environment report generator 작성
- **생성 파일**
  - `05_training/policies/generate_h200_environment_report.py`
  - `05_training/policies/test_generate_h200_environment_report.py`
- **핵심 기능**
  - H200 학습 전 서버 환경을 JSON (JavaScript Object Notation=자바스크립트 객체 표기법) 리포트로 남기는 도구를 작성했다.
  - 기록 항목에는 hostname, OS, Python, torch, CUDA, GPU (Graphics Processing Unit=그래픽 처리 장치), `nvidia-smi`, git commit, repo dirty 여부, 주요 환경변수가 포함된다.
  - 실제 H200 서버에서는 `--require-cuda`, `--require-h200-name`, `--require-clean-git` 옵션으로 강제 검증할 수 있다.
- **의미**
  - 나중에 학습 결과가 달라졌을 때 서버 환경 차이, CUDA/PyTorch 버전 차이, git commit 차이를 추적할 수 있다.

#### 3. Step 68 — H200 actual checkpoint folder contract validator 작성
- **생성 파일**
  - `05_training/policies/H200_actual_checkpoint_folder_contract.md`
  - `05_training/policies/validate_h200_actual_checkpoint_folder.py`
  - `05_training/policies/test_validate_h200_actual_checkpoint_folder.py`
- **핵심 기능**
  - H200 학습 결과 폴더가 actual checkpoint 후보로 인정되기 위한 최소 구조를 검증한다.
  - `best_mappo.pt`만 있는 상태를 차단하고, 아래 파일들이 함께 있어야 한다.

```text
environment_report.json
training_config.json
training_command.txt
git_commit.txt
logs/train_stdout.log
logs/train_stderr.log
logs/train_metrics.jsonl
checkpoints/best_mappo.pt
checkpoints/checkpoint_manifest.json
validation/validate_mappo_checkpoint_actual_report.json
validation/h200_actual_checkpoint_preflight_report.json
README_run_summary.md
```

- **검증 조건**
  - `condition_id = A`
  - `qwen_train = false`
  - `qwen_inference = false`
  - `qwen_trigger_rate = 0.0`
  - `reward_version = mappo_reward_v1`
  - `energy_proxy_model_version = daegu_energy_proxy_v1`
  - `k_dist_kwh_per_m = 0.0012`
  - `k_acc_kwh_per_event = 0.1800`
  - `k_idle_kwh_per_sec = 0.0080`
  - `trained_model = true`
  - `performance_claim_allowed = true`

#### 4. Step 69 — train_mappo_actual.py CLI contract scaffold 작성
- **생성 파일**
  - `05_training/train_mappo_actual.py`
  - `05_training/policies/test_train_mappo_actual_contract.py`
- **핵심 기능**
  - 실제 full training 구현 전, H200에서 사용할 training entrypoint의 CLI (Command Line Interface=명령행 인터페이스) 계약을 고정했다.
  - `--dry-run-contract` 모드에서 필수 인자, A 조건 고정, Qwen 비활성화, reward/energy proxy 상수, seeds `1,2,3`, output folder scaffold 생성을 검증한다.
- **중요 제한**
  - Step 69는 실제 학습을 수행하지 않는다.
  - dry-run contract 모드에서는 `trained_model=false`, `performance_claim_allowed=false`로 기록된다.
  - 즉, Step 69 산출물은 Step 58 actual preflight로 승격될 수 없다.

#### 5. Step 70 — H200 training local validation bundle 작성
- **생성 파일**
  - `05_training/policies/H200_training_local_validation_bundle.md`
  - `05_training/policies/run_h200_training_local_validation_bundle.py`
  - `05_training/policies/test_h200_training_local_validation_bundle.py`
- **핵심 기능**
  - Step 58~69의 핵심 스크립트와 self-test가 로컬에서 존재하고 실행 가능한지 검증하는 bundle runner를 작성했다.
  - dry-run mode에서는 필수 파일 존재 여부와 command pack만 검증한다.
  - execute mode에서는 self-test command pack을 순서대로 실행할 수 있다.
- **확인된 결과**
  - `step70_dry_run_report.json` 생성 완료.
  - `bundle_status = DRY_RUN_VALIDATED`
  - `command_count = 11`
- **의미**
  - H200으로 코드를 옮기기 전, local notebook에서 training gate script bundle이 깨지지 않았는지 빠르게 확인할 수 있다.

#### 6. 현재 H200 training 준비 체계

```text
Step 66:
H200 MAPPO training runbook draft 작성

Step 67:
H200 environment report generator 작성

Step 68:
H200 actual checkpoint folder contract validator 작성

Step 69:
train_mappo_actual.py CLI contract scaffold 작성

Step 70:
H200 training local validation bundle 작성 및 dry-run report 생성
```

#### 7. 기존 actual execution gate와의 연결

Step 66~70은 checkpoint를 만들기 위한 H200 training side 준비이고, Step 58~62는 만들어진 checkpoint를 actual execution side로 승격하기 위한 게이트다.

```text
H200 training side:
Step 66 → Step 67 → Step 68 → Step 69 → Step 70

Actual execution side:
Step 58 → Step 59 → Step 60 → Step 61 → Step 62
```

두 흐름을 연결하면 다음과 같다.

```text
H200에서 학습 실행
→ environment_report.json 기록
→ training output folder 생성
→ folder contract 검증
→ validate_mappo_checkpoint.py --mode actual
→ Step 58 actual preflight
→ Step 59 A-family actual execution plan
→ Step 60 dry-run
→ Step 61 final runbook review
→ Step 62 arrival workflow
```

#### 8. 현재 claim boundary

- Step 66~70 PASS는 H200 training 준비 체계가 갖춰졌다는 뜻이다.
- Step 70 dry-run PASS는 local command bundle이 준비됐다는 뜻이다.
- 아직 실제 `best_mappo.pt` trained checkpoint가 생성된 것은 아니다.
- HistoricalReplayAdapter 기반 결과는 actual policy path validation은 가능하지만 causal performance claim은 불가하다.
- causal 성능 비교는 Phase 2 causal simulator adapter에서만 가능하다.

#### 9. 다음 단계

- Step 72 후보 1: `project_log.md` 이후 전체 상태 점검 및 GitHub push.
- Step 72 후보 2: H200 서버 이관용 run package checklist 작성.
- Step 72 후보 3: Phase 2 causal simulator adapter 설계 시작.
- 현재 로컬에서 가장 자연스러운 다음 작업은 H200 서버에 넘길 최소 파일·명령·산출물 목록을 정리하는 transfer checklist 작성이다.


---
## 📅 2026-04-27
### H200 actual MAPPO checkpoint 실행 게이트 Step 58~62 완료

오늘 작업에서는 A-family MAPPO actual 실행을 실제 H200 checkpoint 도착 이후 안전하게 수행하기 위한 최종 guard 체계를 완성했다. 핵심 목표는 smoke 결과와 actual 결과가 섞이지 않도록 막고, 실제 checkpoint가 들어왔을 때 preflight부터 실행 계획, dry-run 검증까지 일관된 절차로 연결하는 것이다.

#### 1. Step 58 — H200 actual checkpoint preflight checklist 확정
- **생성 파일**
  - `05_training/policies/H200_actual_checkpoint_preflight_checklist.md`
  - `05_training/policies/h200_actual_checkpoint_preflight_checklist.py`
  - `05_training/policies/test_h200_actual_checkpoint_preflight_checklist.py`
- **핵심 기능**
  - 실제 H200 학습 checkpoint가 `mappo_actual` 후보가 되기 위한 입구 검사를 고정했다.
  - checkpoint 파일 존재 여부, `trained_model=true`, `performance_claim_allowed=true`, Qwen 비활성화, reward/energy proxy 버전, 에너지 상수 일치 여부를 검사한다.
  - `validate_mappo_checkpoint.py --mode actual` 통과 여부를 actual 인정 조건으로 포함했다.
- **중요 판정**
  - smoke/fake/scaffold checkpoint는 actual 성능 주장용으로 승격될 수 없다.
  - HistoricalReplayAdapter 기반 실행은 actual policy path 검증은 가능하지만 causal performance claim은 불가하다.

#### 2. Step 59 — A-family mappo_actual execution plan 생성기 확정
- **생성 파일**
  - `05_training/policies/a_family_mappo_actual_execution_plan.md`
  - `05_training/policies/a_family_mappo_actual_execution_plan.py`
  - `05_training/policies/test_a_family_mappo_actual_execution_plan.py`
- **핵심 기능**
  - Step 58 preflight report가 PASS일 때만 A-family actual 실행 계획을 생성한다.
  - 대상 조건은 `A`, `A90`, `A80`, `A70`이며, seed는 `1, 2, 3`으로 고정한다.
  - 총 12개 run 계획을 생성한다.
- **차단 조건**
  - preflight report가 BLOCKED이면 차단.
  - 입력 checkpoint path와 preflight report 내부 checkpoint path가 다르면 차단.
  - B2 등 A-family가 아닌 조건이 섞이면 차단.
  - HistoricalReplayAdapter에서 causal claim을 요구하면 차단.

#### 3. Step 60 — mappo_actual matrix runner guard 검증 완료
- **생성 파일**
  - `05_training/policies/run_a_family_mappo_actual_matrix_from_plan.py`
  - `05_training/policies/test_run_a_family_mappo_actual_matrix_from_plan.py`
- **핵심 기능**
  - Step 59의 execution plan JSON을 입력으로 받아 실행 가능 여부를 검증한다.
  - self-test에서는 실제 H200 matrix를 실행하지 않고 dry-run command validation만 수행한다.
  - `READY_TO_EXECUTE` plan만 dry-run 또는 execute 모드로 진입할 수 있다.
  - 각 run의 명령 순서를 아래와 같이 강제한다.

```text
rollout_command
→ metadata_validation_command
→ canonical_command
→ post_canonical_validation_command
```

- **Step 60 중 발견 및 수정**
  - 최초 self-test에서 `causal_claim_guard.require_causal_claim=true`가 plan 최상위에 있을 때, run 내부의 false 값이 이를 덮어써 차단되지 않는 문제가 발견되었다.
  - 이를 strict OR semantics로 수정했다.
  - 이제 plan 최상위 또는 run 내부 중 하나라도 causal claim을 요구하면 true로 취급하며, HistoricalReplayAdapter 기반에서는 즉시 차단된다.

#### 4. Step 61 — H200 actual MAPPO final runbook 확정
- **생성 파일**
  - `05_training/policies/H200_actual_mappo_final_runbook.md`
  - `05_training/policies/test_h200_actual_mappo_final_runbook.py`
- **핵심 기능**
  - 실제 실행 직전 운영자가 따라야 할 최종 절차서를 작성했다.
  - Step 58 preflight, Step 59 plan, Step 60 dry-run, Step 60 execute의 순서를 문서화했다.
  - 실행 전 operator checklist를 추가했다.
- **해석 기준**
  - `A-family actual MAPPO checkpoint execution path validated under non-causal replay`는 허용된다.
  - `A-family actual MAPPO causally outperformed baselines`는 HistoricalReplayAdapter 기반에서는 금지된다.

#### 5. Step 62 — actual checkpoint arrival workflow 확정
- **생성 파일**
  - `05_training/policies/H200_actual_checkpoint_arrival_workflow.md`
  - `05_training/policies/run_h200_actual_checkpoint_arrival_workflow.py`
  - `05_training/policies/test_h200_actual_checkpoint_arrival_workflow.py`
- **핵심 기능**
  - 실제 H200 checkpoint가 도착했을 때 실행할 자동 연결 workflow를 만들었다.
  - 순서는 다음과 같다.

```text
checkpoint existence check
→ Step 58 actual preflight
→ preflight PASS check
→ checkpoint_path consistency check
→ Step 59 execution plan generation
→ READY_TO_EXECUTE check
→ Step 60 dry-run
→ DRY_RUN_VALIDATED check
→ arrival workflow report 작성
```

- **성공 상태**
  - `arrival_workflow_status = READY_FOR_MANUAL_EXECUTE`
  - `preflight_status = PASS`
  - `plan_status = READY_TO_EXECUTE`
  - `dry_run_status = DRY_RUN_VALIDATED`
- **중요 제한**
  - Step 62는 실제 12개 matrix run을 실행하지 않는다.
  - Step 62 PASS는 manual execute 준비 완료를 뜻하며, causal performance 증명을 뜻하지 않는다.

#### 6. 현재 확정된 actual 실행 체계

```text
Step 58:
실제 H200 checkpoint가 actual 자격이 있는지 preflight 검증

Step 59:
PASS checkpoint만으로 A/A90/A80/A70 × seeds 1,2,3 실행 계획 생성

Step 60:
READY_TO_EXECUTE plan만 dry-run/execute 가능한 matrix runner guard 검증

Step 61:
실제 실행 직전 운영자용 final runbook 확정

Step 62:
checkpoint 도착 시 preflight → plan → dry-run까지 자동 연결하는 arrival workflow 확정
```

#### 7. 현재 claim boundary

- actual policy path 검증 가능:
  - 실제 H200 checkpoint가 preflight를 통과하고, plan과 dry-run이 통과한 경우.
- causal performance claim 불가:
  - HistoricalReplayAdapter 또는 historical/replay/noncausal adapter 기반인 경우.
- causal 성능 비교는 Phase 2 causal simulator adapter가 준비된 뒤에만 가능하다.

#### 8. 다음 단계

- 실제 H200 checkpoint가 아직 없다면, 다음 단계는 H200 학습 실행 준비 또는 H200 training runbook 작성이다.
- 실제 checkpoint가 생기면 Step 62 arrival workflow를 실행한다.
- 별도로 남아 있는 작업트리 변경:
  - `.claude/worktrees/*`
  - `05_training/run_experiment_A_neural_inference_smoke.py`
  - `05_training/policies/test_neural_inference_strict_checkpoint.py`
  는 Step 63 커밋 대상에서 제외한다.


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


---
## 📅 2026-04-26
### Phase 2 Step 32~39 MAPPO Actual-Inference Preparation

오늘 작업에서는 A/A90/A80/A70 placeholder 기반 검증 경로를 실제 MAPPO 연결 준비 경로로 확장했다.  
MAPPO (Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) 실제 학습은 아직 H200 서버에서 수행하지 않았지만, checkpoint 존재 검증, policy interface, action field, canonical KPI 집계 경로까지의 연결 안전장치를 갤럭시북5 Pro에서 모두 사전 검증했다.

#### Step 32 — actual MAPPO checkpoint preflight 확인
- 명령:
  - `python .\05_training\run_causal_rollout.py --a-family-bridge --policy-kind mappo --checkpoint-path artifacts/experiment_A_v1/checkpoints/best.pt --require-existing-checkpoint --write-parquet`
- 결과:
  - checkpoint가 없을 때 `FileNotFoundError`로 중단됨.
  - placeholder fallback이 발생하지 않음을 확인.
- 의미:
  - actual MAPPO 경로가 checkpoint 존재 여부를 강제한다.
  - checkpoint 없는 상태에서 A/A90/A80/A70 결과를 실수로 생성하지 않는다.

#### Step 33 — friendly MAPPO checkpoint STOP message 추가
- 수정 파일:
  - `05_training/run_causal_rollout_a_family_bridge_v1.py`
  - `05_training/test_step33_friendly_checkpoint_stop.py`
- 기존 긴 traceback을 사람이 읽기 쉬운 STOP 메시지로 정리.
- 최종 출력 예:
  - `[STOP] MAPPO checkpoint does not exist: artifacts/experiment_A_v1/checkpoints/best.pt`
  - `This is expected before actual MAPPO training.`
  - `No placeholder fallback was used.`
- 의미:
  - 실제 학습 전에는 checkpoint 없음이 정상 STOP 상태임을 명확히 전달.
  - 운영 중 traceback 혼동을 줄임.

#### Step 34 — fake checkpoint MAPPO boundary smoke 추가
- 생성 파일:
  - `05_training/evaluation/run_fake_checkpoint_mappo_boundary_smoke_v1.py`
  - `05_training/evaluation/README_fake_checkpoint_mappo_boundary_smoke.md`
  - `05_training/evaluation/test_fake_checkpoint_mappo_boundary_smoke_v1.py`
- fake checkpoint를 만들어 `--policy-kind mappo --require-existing-checkpoint` 경로가 통과하는지 검증.
- 검증 항목:
  - `policy_source = mappo_policy`
  - `source_mode = causal_*_mappo_policy_v1`
  - placeholder/stub/smoke marker 없음
  - `qwen_trigger_rate = 0.0`
  - canonical KPI (Key Performance Indicator=핵심성과지표) 집계 통과
- 주의:
  - fake checkpoint는 성능 주장용이 아니다.
  - boundary와 artifact path 검증용이다.

#### Step 35 — MAPPO policy inference interface contract 추가
- 생성 파일:
  - `05_training/policies/mappo_policy_interface_v1.py`
  - `05_training/policies/README_mappo_policy_interface.md`
  - `05_training/policies/test_mappo_policy_interface_v1.py`
- 정의한 계약:
  - checkpoint metadata contract
  - observation dict contract
  - action dict contract
- 필수 checkpoint metadata:
  - `artifact_type = urbanbus_mappo_checkpoint`
  - `interface_version = mappo_policy_interface_v1`
  - `policy_kind = mappo`
  - `trained_model = true`
  - `action_space_version = bus_control_action_v1`
  - `observation_space_version = urbanbus_observation_v1`
- 현재 action은 conservative mock action이다.
- 의미:
  - 실제 H200 checkpoint가 생성되면 어떤 입출력 형식으로 붙일지 사전에 고정했다.

#### Step 36 — MAPPO policy interface를 A-family bridge row에 연결
- 생성 파일:
  - `05_training/rollouts/a_family_policy_interface_bridge_v1.py`
  - `05_training/rollouts/README_a_family_policy_interface_bridge.md`
  - `05_training/rollouts/test_a_family_policy_interface_bridge_v1.py`
- Step 35의 action dict를 실제 rollout row에 반영.
- 추가·검증된 action fields:
  - `action_version`
  - `dispatch_delta`
  - `hold_seconds`
  - `skip_stop_flag`
  - `target_headway_ratio`
  - `policy_debug`
  - `policy_action_debug`
  - `policy_interface_version`
  - `performance_claim_allowed`
- `performance_claim_allowed = false` 유지.
- 의미:
  - neural MAPPO가 아니더라도 action interface가 rollout row 구조에 안전하게 들어갈 수 있음을 확인.

#### Step 37 — policy interface scenario writer 추가
- 생성 파일:
  - `05_training/rollouts/run_a_family_policy_interface_scenario_writer_v1.py`
  - `05_training/rollouts/README_a_family_policy_interface_scenario_writer.md`
  - `05_training/rollouts/test_a_family_policy_interface_scenario_writer_v1.py`
- scenario_index 기반으로 다중 window에 policy interface action fields를 붙임.
- 검증 구조:
  - `scenario_index.parquet`
  - seed
  - A/A90/A80/A70
  - base rollout row
  - policy observation
  - action dict
  - extended `window_rollup`
- 의미:
  - 단일 row가 아니라 여러 window/condition에 대해 MAPPO interface row 생성이 가능해졌다.

#### Step 38 — policy-interface canonical KPI smoke 추가
- 생성 파일:
  - `05_training/evaluation/run_policy_interface_canonical_smoke_v1.py`
  - `05_training/evaluation/README_policy_interface_canonical_smoke.md`
  - `05_training/evaluation/test_policy_interface_canonical_smoke_v1.py`
- 검증 경로:
  - Step 37 policy-interface scenario writer
  - A/A90/A80/A70 window_rollup 생성
  - action fields 포함 확인
  - `canonical_kpi_aggregator.py --mode official_rollup`
  - condition별 canonical_eval 생성
- 결과:
  - `passed = True`
- 의미:
  - action fields가 붙은 MAPPO-interface rollout도 canonical KPI 집계 경로와 호환됨을 확인했다.

#### Step 39 — H200 MAPPO training/inference runbook 작성
- 생성 파일:
  - `05_training/runbooks/H200_MAPPO_training_runbook_v1.md`
  - `05_training/runbooks/h200_mappo_training_commands_v1.sh`
  - `05_training/runbooks/h200_mappo_inference_smoke_commands_v1.sh`
  - `05_training/runbooks/test_h200_mappo_runbook_v1.py`
- 목적:
  - 갤럭시북5 Pro에서 만든 검증 경로를 H200 서버 실제 학습·추론 절차로 이전하기 위한 runbook 작성.
- 포함 내용:
  - H200 git clone 절차
  - Python virtual environment 설정
  - GPU (Graphics Processing Unit=그래픽처리장치) / CUDA (Compute Unified Device Architecture=엔비디아 병렬 컴퓨팅 플랫폼) 확인
  - Step 21~38 regression test 실행
  - 실제 MAPPO checkpoint 출력 위치
  - checkpoint metadata sidecar 요구사항
  - inference smoke 명령
  - canonical KPI 집계 절차
- 실제 checkpoint 목표:
  - `artifacts/experiment_A_v1/checkpoints/best.pt`
  - `artifacts/experiment_A_v1/checkpoints/best.pt.metadata.json`

#### 현재 확정 상태
- A/A90/A80/A70은 여전히 Qwen 비활성화 상태다.
- actual MAPPO 경로는 checkpoint가 없으면 STOP한다.
- fake checkpoint는 boundary 검증용이며 성능 주장에 사용할 수 없다.
- policy interface 경로는 아직 conservative mock action이다.
- canonical KPI 집계 경로까지 mock-action rollout 호환성은 확인됐다.
- H200 서버에서 실제 trained checkpoint 생성 후 metadata sidecar가 필요하다.

#### 논문용 결과 사용 조건
논문용 성능 주장은 아래 조건이 모두 만족될 때만 가능하다.

1. `policy_source = mappo_policy`
2. `source_mode = causal_*_mappo_policy_v1`
3. checkpoint metadata의 `trained_model = true`
4. placeholder/stub/smoke marker 없음
5. `qwen_trigger_rate = 0.0`
6. A/A90/A80/A70이 동일 window에서 동일 `passenger_demand_generated` 사용
7. canonical KPI aggregation 통과
8. 실제 H200 학습 checkpoint 기반 결과

#### 다음 단계
Step 40부터는 새창에서 시작한다.

권장 목표:
- actual neural MAPPO inference adapter 자리 만들기
- 추천 파일:
  - `05_training/policies/mappo_neural_policy_adapter_v1.py`
  - `05_training/policies/README_mappo_neural_policy_adapter.md`
  - `05_training/policies/test_mappo_neural_policy_adapter_v1.py`
- conservative mock action을 바로 제거하지 않고, 별도 adapter에서 실제 H200 checkpoint loader를 받을 준비를 한다.







