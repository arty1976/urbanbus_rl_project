# Skill 03 — Claim guard와 metadata 보존 검증

## 목적

scaffold, smoke, replay, actual 결과가 서로 섞이지 않도록 metadata를 보존하고, 성능 주장 가능 여부를 자동으로 차단한다.

이 스킬은 canonical KPI(Key Performance Indicator=핵심 성과 지표) aggregation 이후에도 source_mode, policy_source, reward_scaffold_only, train_allowed 같은 중요한 provenance metadata가 사라지지 않도록 하는 데 사용한다.

## 언제 사용하나

- canonical KPI aggregator를 통과한 뒤 metadata가 사라지는 문제가 생겼을 때
- scaffold 결과와 actual 결과가 섞일 위험이 있을 때
- 논문용 성능 claim을 차단해야 할 때
- `noncausal` 문자열이 guard에서 누락될 위험이 있을 때
- pandas merge 후 `_x`, `_y` suffix 때문에 원래 컬럼명이 사라졌을 때

## 핵심 원칙

1. 결과 row는 항상 source provenance를 가져야 한다.
2. scaffold output은 actual claim으로 승격하면 안 된다.
3. reward scaffold는 train_with_this_reward_allowed=false를 유지해야 한다.
4. canonical aggregator가 컬럼을 drop하면 wrapper에서 보존본을 따로 만든다.
5. `noncausal` / `non_causal` / `non-causal` 표기를 통일한다.

## 필수 guard 컬럼

```text
paper_level_claim_allowed
causal_performance_claim_allowed
canonical_causal_comparison_allowed
actual_headway_observed
actual_arrival_departure_time_observed
actual_dwell_observed
actual_passenger_wait_observed
queue_demand_observed
queue_demand_proxy
reward_scaffold_only
train_with_this_reward_allowed
final_reward_design_claim_allowed
```

## 권장 기본값

```json
{
  "paper_level_claim_allowed": false,
  "causal_performance_claim_allowed": false,
  "canonical_causal_comparison_allowed": false,
  "actual_headway_observed": false,
  "actual_arrival_departure_time_observed": false,
  "actual_dwell_observed": false,
  "actual_passenger_wait_observed": false,
  "queue_demand_observed": false,
  "queue_demand_proxy": true,
  "reward_scaffold_only": true,
  "train_with_this_reward_allowed": false,
  "final_reward_design_claim_allowed": false
}
```

## 문제 패턴 1 — canonical aggregator가 metadata를 drop함

증상:

```text
policy_source, checkpoint_loaded, mock_action_used, energy_proxy_model_version 같은 컬럼이 kpi_by_window.parquet에서 사라짐
```

해결:

```text
1. 보존해야 할 metadata column list를 정의한다.
2. compute_official_kpi_by_window()의 return 직전 out_cols에 preserve_cols를 추가한다.
3. self-test에서 pre-canonical window_rollup과 post-canonical kpi_by_window의 metadata를 비교한다.
```

## 문제 패턴 2 — noncausal 표기가 guard에서 누락됨

증상:

```text
source_mode = route_aware_minimal_scaffold_step102_noncausal
canonical guard가 non_causal을 감지하지 못함
```

해결:

```text
staging copy 생성 시 noncausal → non_causal로 정규화한다.
```

## 문제 패턴 3 — pandas merge suffix collision

증상:

```text
preserved kpi_by_window missing extended KPI columns:
passenger_demand_generated, passenger_served_count ...
```

원인:

```text
canonical_df와 staged_df에 같은 컬럼이 이미 있어서 merge 후 _x / _y suffix가 붙음
```

해결:

```text
1. preserve_cols를 overlapping_preserve_cols와 merge_preserve_cols로 나눈다.
2. 이미 있는 컬럼은 combine_first로 staged value를 보완한다.
3. suffix column은 plain column으로 정리한다.
4. 최종적으로 원래 컬럼명이 존재하는지 검사한다.
```

## 문제 패턴 4 — nested manifest를 top-level로 읽음

증상:

```text
hard_failures가 발생하지만 실제 manifest에는 값이 존재함
```

해결:

```text
nested_get(payload, "canonical_summary.row_counts.kpi_by_window") 같은 안전한 nested accessor를 사용한다.
```

## 완료 기준

- metadata 보존 self-test PASS
- claim guard false 유지
- causal_allowed=false 유지
- reward_scaffold_only=true 유지
- train_with_this_reward_allowed=false 유지
- suffix collision 없음
- nested manifest 구조 지원

## 에이전트용 프롬프트

```text
canonical KPI와 reward-policy pipeline에서 claim guard와 metadata 보존을 검증하라.
scaffold/smoke/replay output이 actual 성능 claim으로 승격되지 않도록 paper_level_claim_allowed=false, causal_performance_claim_allowed=false, train_with_this_reward_allowed=false를 유지하라.
canonical aggregator나 merge 과정에서 metadata/extended KPI 컬럼이 사라지지 않게 preserve wrapper 또는 patch를 작성하라.
noncausal 표기는 non_causal로 정규화하고, pandas merge suffix collision과 nested manifest mismatch를 self-test로 방지하라.
```
