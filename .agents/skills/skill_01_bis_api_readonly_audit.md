# Skill 01 — 대구 BIS API read-only 데이터 감사

## 목적

대구 BIS(Bus Information System=버스정보시스템) API(Application Programming Interface=응용 프로그램 인터페이스) 데이터를 새로 확보했을 때, 기존 DB(Database=데이터베이스)나 tensor DB(Database=데이터베이스)를 바로 덮어쓰지 않고 artifact 기반으로 안전하게 감사한다.

이 스킬은 `/getBs02`, `/getPos02`, `/getRealtime02` 같은 API 산출물을 route-aware causal simulator v2 입력 후보로 승격하기 전, observed / candidate / proxy / not_observed를 분리하는 데 사용한다.

## 언제 사용하나

- 새 API 데이터를 받았지만 아직 DB에 넣어도 되는지 확신이 없을 때
- route_id / direction_id / stop_id / stop_order 체계가 서로 맞는지 확인해야 할 때
- 실제 관측값과 추정 후보를 구분해야 할 때
- API schema가 문서와 다르게 내려올 가능성이 있을 때
- 논문 주장 전에 데이터 계보와 신뢰등급을 고정해야 할 때

## 입력

- API raw JSON(JSON=JavaScript Object Notation=자바스크립트 객체 표기법) 또는 normalized CSV(Comma-Separated Values=쉼표 구분 값)
- 기존 project DB route list 또는 artifact route list
- 이전 Step classification JSON/MD가 있으면 선택 입력

## 원칙

1. DB write 금지.
2. tensor DB overwrite 금지.
3. 검증 전 API 산출물을 학습 데이터에 직접 주입하지 않는다.
4. 산출물은 `artifacts/...` 아래에만 둔다.
5. API 호출 스크립트와 분석 스크립트를 분리한다.
6. actual observed와 candidate/proxy를 엄격히 구분한다.
7. paper_level_claim_allowed=false를 기본값으로 둔다.
8. causal_performance_claim_allowed=false를 기본값으로 둔다.

## 절차

### 1. API별 raw/normalized 산출물 존재 확인

확인할 대표 산출물:

```text
/getBs02
- route_id
- direction_id
- ordered_stop_sequence
- stop_id

/getPos02
- vhcNo2
- xPos
- yPos
- seq
- bsId
- moveDir
- arTime

/getRealtime02
- route_id
- stop_id
- ETA 후보
- ETA 기반 headway candidate
```

### 2. API별 classification 작성

예시:

```text
route_id = observed_candidate_full_collection_234_of_238_routes
direction_id = observed_candidate_full_collection_234_of_238_routes_cross_confirmed
ordered_stop_sequence = observed_candidate_full_collection_234_of_238_routes
bus_id_or_vehicle_no = repeated_sample_observed_candidate_from_getPos02_vhcNo2
live_position_xy = repeated_sample_observed_candidate_from_getPos02_xPos_yPos
getRealtime02_eta = observed_candidate_cross_matched_to_getBs02
eta_based_headway = candidate_from_getRealtime02_cross_matched_to_getBs02
actual_headway = not_observed
actual_arrival_departure_time = not_observed
actual_dwell = not_observed
```

### 3. Cross-match audit 수행

기준 table은 `/getBs02` route-stop sequence로 둔다.

검증 항목:

```text
getPos02 row → getBs02 route_id / direction_id / stop_id match
getRealtime02 ETA row → getBs02 route_id / direction_id / stop_id match
ETA-based headway candidate → getBs02 route-stop sequence match
```

coverage가 1.0이어도 actual로 승격하지 않는다. 이는 “같은 좌표계로 연결 가능”하다는 뜻이지, 실제 도착/출발 이벤트를 관측했다는 뜻이 아니다.

### 4. 통합 classification update 생성

생성 권장 파일:

```text
classification_update_consolidated_stepXX.json
classification_update_consolidated_stepXX.md
field_classification_matrix_stepXX.csv
causal_simulator_v2_contract_patch.json
```

### 5. claim guard 삽입

모든 report와 manifest에 아래를 넣는다.

```json
{
  "actual_headway": "not_observed",
  "actual_arrival_departure_time": "not_observed",
  "actual_dwell": "not_observed",
  "paper_level_claim_allowed": false,
  "causal_performance_claim_allowed": false
}
```

## 완료 기준

- API별 normalized rows 수가 기록됨
- coverage 계산이 완료됨
- known exception이 분리됨
- consolidated classification JSON/MD가 생성됨
- actual_headway / actual_arrival_departure_time / actual_dwell이 not_observed로 유지됨
- DB write와 tensor DB overwrite가 발생하지 않음

## 에이전트용 프롬프트

```text
대구 BIS API 산출물을 read-only 방식으로 감사한다.
DB write와 tensor DB overwrite는 금지한다.
입력 artifact만 읽고 route_id / direction_id / stop_id / ordered_stop_sequence 체계를 cross-match하라.
observed / candidate / proxy / not_observed를 분리하고, actual_headway / actual_arrival_departure_time / actual_dwell은 실제 이벤트 로그가 없으면 절대 observed로 승격하지 말라.
최종 산출물은 classification_update JSON/MD, field matrix CSV, causal simulator v2 contract patch로 만든다.
paper_level_claim_allowed=false와 causal_performance_claim_allowed=false를 모든 manifest에 기록하라.
```
