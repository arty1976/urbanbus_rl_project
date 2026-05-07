# Step 160 — Zero-Loss Pickup Evidence Reporter Schema

문서 상태: scaffold / non-claim evidence contract  
프로젝트: `urbanbus_rl_project`  
목적: 특허 명세서 보조 증거용 Zero-Loss Pickup Threshold 실험 리포트 생성 계약

---

## 0. 핵심 원칙

이 Step 160 산출물은 **성능 주장 도구가 아니다.**

허용되는 주장:

```text
route-aware simulator 기반 합승 후보 로그에서
기존 탑승 승객의 ETA(Estimated Time of Arrival=예상 도착 시간) 변화량이
0초 또는 지정 epsilon 이내인 시도를 집계하고,
그 시점의 GATv2(Graph Attention Network v2=그래프 어텐션 네트워크 v2)
attention weight(어텐션 가중치) 분포를 evidence bundle로 저장할 수 있다.
```

금지되는 주장:

```text
actual 운행에서 Zero-Loss 합승이 관측되었다.
actual headway가 개선되었다.
actual arrival/departure/dwell이 관측되었다.
paper-level 또는 causal performance claim이 가능하다.
```

따라서 모든 manifest는 아래 guard를 유지해야 한다.

```json
{
  "simulation_evidence_only": true,
  "actual_operational_claim_allowed": false,
  "paper_level_claim_allowed": false,
  "causal_performance_claim_allowed": false,
  "train_allowed": false
}
```

---

## 1. 입력 파일 1 — pickup attempt events

권장 파일명:

```text
pickup_attempt_events.parquet
```

지원 포맷:

```text
.parquet / .csv / .json / .jsonl
```

필수 컬럼:

| 컬럼 | 의미 |
|---|---|
| `attempt_id` | 합승 시도 고유 ID(Identifier=식별자) |
| `state_ts` | 판단 시점 timestamp |
| `condition_id` | A/A90/A80/A70 등 condition |
| `seed` | 재현 seed |
| `vehicle_id` | 차량 또는 agent 식별자 |
| `existing_passenger_id` | 먼저 탄 승객 식별자 |
| `new_passenger_id` | 신규 승객 식별자 |
| `route_id` | 노선 ID |
| `direction_id` | 방향 ID |
| `pickup_stop_id` | 신규 승객 pickup 정류장 |
| `dropoff_stop_id` | 신규 승객 dropoff 정류장 |
| `decision` | accepted / rejected / candidate 등 |

선택 컬럼:

| 컬럼 | 의미 |
|---|---|
| `time_band` | peak/offpeak/night |
| `window_id` | 평가 window |
| `policy_source` | mappo_policy / rulebased / scaffold 등 |
| `source_mode` | route-aware scaffold/source mode |

---

## 2. 입력 파일 2 — ETA counterfactual table

권장 파일명:

```text
eta_counterfactual.parquet
```

필수 컬럼:

| 컬럼 | 의미 |
|---|---|
| `attempt_id` | pickup attempt events와 join되는 ID |
| `existing_passenger_id` | 먼저 탄 승객 ID |
| `eta_without_new_pickup_sec` | 신규 승객을 태우지 않았을 때 기존 승객 ETA |
| `eta_with_new_pickup_sec` | 신규 승객을 태웠을 때 기존 승객 ETA |

선택 컬럼:

| 컬럼 | 의미 |
|---|---|
| `counterfactual_method` | ETA 계산 방식 |
| `simulator_version` | simulator 버전 |

Reporter는 아래 값을 계산한다.

```text
delta_eta_existing_passenger_sec =
  eta_with_new_pickup_sec - eta_without_new_pickup_sec

zero_loss_success =
  delta_eta_existing_passenger_sec <= zero_loss_epsilon_sec
```

기본 `zero_loss_epsilon_sec`는 0.0이다. 단 simulator 해상도가 초 단위보다 거칠다면 해당 해상도를 epsilon으로 기록할 수 있다.

---

## 3. 입력 파일 3 — GATv2 attention weights

권장 파일명:

```text
gatv2_attention.parquet
```

필수 컬럼:

| 컬럼 | 의미 |
|---|---|
| `attempt_id` | pickup attempt와 join되는 ID |
| `state_ts` | attention 산출 시점 |
| `layer_id` | GATv2 layer |
| `head_id` | attention head |
| `src_node` | source node |
| `dst_node` | destination node |
| `attention_weight` | attention weight |

선택 컬럼:

| 컬럼 | 의미 |
|---|---|
| `edge_id` | edge 식별자 |
| `distance_m` | edge 거리 |
| `time_sec` | edge 소요 시간 |
| `generalized_cost` | generalized cost |
| `path_segment` | existing_passenger_path / candidate_pickup_path / candidate_dropoff_path / unrelated |
| `route_id` | route ID |
| `direction_id` | direction ID |

---

## 4. 입력 파일 4 — run manifest

권장 파일명:

```text
run_manifest.json
```

권장 필드:

```json
{
  "artifact_version": "zero_loss_pickup_evidence_input_v1",
  "project": "urbanbus_rl_project",
  "git_commit": "<commit>",
  "dataset_artifact": "<dataset>",
  "checkpoint_path": "<path>",
  "checkpoint_sha256": "<sha256>",
  "gatv2_artifact_contract": "<path>",
  "simulator_version": "route_aware_simulator_v2_scaffold",
  "zero_loss_epsilon_sec": 0.0,
  "simulation_evidence_only": true,
  "actual_operational_claim_allowed": false,
  "paper_level_claim_allowed": false,
  "causal_performance_claim_allowed": false
}
```

---

## 5. 출력 bundle

Reporter는 output root 아래에 다음 파일을 생성한다.

```text
zero_loss_summary.json
zero_loss_by_condition.csv
eta_delta_by_attempt.csv
attention_mass_by_segment.csv
top_attention_edges.csv
patent_evidence_report.md
zero_loss_evidence_manifest.json
eta_delta_distribution.png              # matplotlib 가능 시
attention_mass_by_segment.png           # matplotlib 가능 시
```

---

## 6. Step 160 PASS 조건

Validator는 최소 아래 조건을 확인한다.

```text
1. zero_loss_evidence_manifest.json 존재
2. zero_loss_summary.json 존재
3. total_pickup_attempts > 0
4. zero_loss_success_count <= total_pickup_attempts
5. zero_loss_epsilon_sec >= 0
6. simulation_evidence_only = true
7. actual_operational_claim_allowed = false
8. paper_level_claim_allowed = false
9. causal_performance_claim_allowed = false
10. eta_delta_by_attempt.csv의 zero_loss_success가 epsilon 조건과 일치
11. top_attention_edges.csv가 비어 있지 않음
12. attention_mass_by_segment.csv가 비어 있지 않음
```

---

## 7. Step 160 이후

Step 160은 실제 claim을 여는 단계가 아니다. 다음 단계 후보는 다음이다.

```text
Step 161: route-aware simulator pickup attempt event writer
Step 162: ETA counterfactual pair generator
Step 163: GATv2 attention export hook
Step 164: patent evidence bundle integration test
```
