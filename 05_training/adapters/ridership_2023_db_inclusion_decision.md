# Step 105 — 2023 Ridership DB Inclusion Decision

## 0. 목적

이 문서는 2023년 정류소별 시간대별 승하차 CSV를 현재 tensor DB에 새로 넣을지, 이미 포함된 것으로 보고 재적재하지 않을지에 대한 의사결정을 기록한다.

## 1. 확인 결과

`public.stg_daegu_stop_usage_2023`에서 다음 값이 확인되었다.

```text
row_count = 2,102,115
date_count = 365
date_range = 2023-01-01 ~ 2023-12-31
stop_id_count = 3,386
usage_type_count = 2
boarding_total = 181,556,972
alighting_total = 70,567,680
overall_total = 252,124,652
이는 2023년 정류소별 시간대별 승하차 CSV의 기대 총량과 일치한다.

2. 판단
2023 승하차 CSV는 이미 DB의 staging source에 포함되어 있다.

따라서 현 단계에서는 같은 2023 CSV를 다시 import하지 않는다.

3. 결정
3.1 2023 데이터
status = already_loaded_in_db
source_table = public.stg_daegu_stop_usage_2023
action = do_not_reimport
tensor_rebuild_required_now = false

현재 tensor DB가 2023년 기준임은 다음 핵심 relation에서 확인되었다.

public.graph_state_timeslice.state_ts
2023-01-01 05:00:00+09:00 ~ 2023-12-31 23:00:00+09:00

public.rl_state_training_base.state_ts
2023-01-01 05:00:00+09:00 ~ 2023-12-31 22:00:00+09:00

따라서 현재 단계에서는 2023 승하차 CSV를 새로 넣거나 기존 tensor DB를 재생성하지 않는다.

3.2 2022 데이터
status = archive_for_future_cross_year_validation
action = do_not_merge_into_current_2023_tensor_db

2022년 자료는 가치가 있지만, 현재 tensor DB가 2023년 기준으로 닫혀 있으므로 현 단계에서는 합치지 않는다.

4. 하지 말아야 할 작업
- stg_daegu_stop_usage_2023에 2023 CSV 재적재
- fact_stop_usage_hourly에 같은 2023 데이터를 중복 insert
- 기존 tensor DB를 동일 원천으로 무작정 재생성
- 2022 자료를 현재 2023 tensor DB에 혼합
5. 허용되는 표현
2023 정류소별 시간대별 승하차 CSV와 동일한 row 수·기간·승차/하차 합계가 public.stg_daegu_stop_usage_2023에서 확인되었으므로, 해당 원천 데이터는 이미 DB에 적재된 것으로 판단한다.
6. 피해야 할 표현
모든 tensor row가 100% 해당 CSV에서 직접 왔다.

staging → fact → graph_state_timeslice → tensor 전체 lineage를 row-level로 끝까지 대조한 것은 아니기 때문에, 위 표현은 피한다.

7. 다음 단계

다음 작업은 ridership 재적재가 아니라 다음 중 하나다.

1. 현재 DB lineage를 문서화
2. fact_stop_usage_hourly와 tensor source table의 변환 규칙 확인
3. 2022 자료를 future cross-year validation 후보로 별도 보관
4. 기존 tensor DB는 보존하고, 필요할 때만 새 버전으로 rebuild

