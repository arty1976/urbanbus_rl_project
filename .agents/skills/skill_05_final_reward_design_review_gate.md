# Skill 05 — Final reward design review gate

## 목적

Step 108/109에서 만든 scaffold reward를 실제 MAPPO(Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) 학습 reward로 착각하지 않도록 차단하고, 최종 reward 설계로 승격하기 전 필요한 검토 gate를 만든다.

이 스킬은 다음 단계인 Step 110에 바로 사용한다.

## 왜 필요한가

Step 108의 reward는 다음 목적만 가진다.

```text
12-KPI → reward component → policy interface 경로가 깨지지 않는지 확인
```

그러나 실제 학습 reward는 다음을 모두 고려해야 한다.

```text
서비스 품질 우선
긴 대기 승객 보호
미서비스 승객 방지
energy 절감 꼼수 방지
fleet reduction 보너스의 과최적화 방지
baseline normalization
hard constraint
ablation compatibility
```

따라서 최종 reward 설계 전까지는 `train_with_this_reward_allowed=false`가 유지되어야 한다.

## 최종 reward 설계 원칙

### 1. 서비스 품질 우선

버스를 줄이거나 에너지를 줄이는 것이 목적이지만, 그 과정에서 승객을 버리는 정책은 실패다.

우선순위:

```text
1. passenger_service_rate 유지
2. avg_wait_seconds 악화 방지
3. passenger_wait_p95_seconds 폭증 방지
4. bunching_rate 악화 방지
5. on_time_rate 악화 방지
6. 위 조건을 만족한 뒤 energy_proxy_per_passenger와 fleet_reduction_ratio 개선
```

### 2. energy 단독 최적화 금지

`energy_proxy`만 줄이면 버스를 적게 운행하는 정책이 유리해질 수 있다.

따라서 energy는 반드시 아래와 함께 해석한다.

```text
energy_proxy_per_passenger
passenger_service_rate
passenger_wait_p95_seconds
```

### 3. fleet reduction은 보너스

A90/A80/A70은 active bus count를 줄이는 조건이다. fleet_reduction_ratio를 너무 크게 주면 “버스를 줄였으니 성공”이라는 잘못된 최적화가 발생한다.

따라서 fleet reduction은 주 reward가 아니라 보조 bonus여야 한다.

### 4. p95 wait는 강한 penalty

평균 대기시간이 좋아도 일부 승객이 오래 기다리면 실제 대중교통 정책으로는 실패다.

따라서 passenger_wait_p95_seconds는 avg_wait_seconds보다 강한 보호 항목으로 둔다.

## 검토해야 할 reward term

후보 식:

```text
reward_t =
  + w_service * service_rate_score
  - w_avg_wait * avg_wait_penalty
  - w_long_wait * long_wait_penalty
  + w_ontime * on_time_score
  - w_bunching * bunching_penalty
  - w_energy * energy_per_passenger_penalty
  + w_fleet * fleet_reduction_bonus
  - w_safety * constraint_violation_penalty
```

검토 항목:

```text
service_rate_score
avg_wait_penalty
long_wait_penalty
on_time_score
bunching_penalty
energy_per_passenger_penalty
fleet_reduction_bonus
constraint_violation_penalty
```

## hard constraint 후보

```text
passenger_service_rate < 0.95 → 강한 penalty
avg_wait_seconds > baseline_avg_wait_seconds * 1.10 → penalty
passenger_wait_p95_seconds > baseline_p95_wait * 1.15 → penalty
bunching_rate가 baseline보다 악화 → penalty
on_time_rate가 baseline보다 하락 → penalty
qwen_trigger_rate != 0.0 in A condition → fail
```

## baseline normalization 후보

초기 학습 reward 기준:

```text
B1_noop
```

논문 평가 비교 기준:

```text
B0 historical
B1 noop
B2 rulebased
A/A90/A80/A70 MAPPO variants
```

## Step 110 gate가 확인해야 할 것

필수 입력:

```text
12-KPI schema exists
reward scaffold exists
policy reward interface exists
claim guard exists
baseline normalization plan exists
hard constraints are documented
reward weights are marked candidate, not final
```

필수 출력:

```text
final_reward_design_review_gate_step110.json
final_reward_design_review_gate_step110.md
reward_design_checklist_step110.csv
```

## 승격 조건

아래가 모두 충족되기 전까지 train_with_this_reward_allowed는 false이다.

```text
reward_formula_finalized = true
reward_weights_are_final = true
baseline_normalization_finalized = true
hard_constraints_finalized = true
reward_unit_tests_passed = true
ablation_compatibility_checked = true
paper_claim_guard_reviewed = true
```

## 강제 차단 조건

하나라도 해당하면 최종 reward 승격 금지:

```text
passenger_service_rate가 reward에서 빠짐
p95 wait penalty가 없음
energy 단독 최적화 구조
fleet reduction이 주 reward로 작동
actual passenger wait observed라고 잘못 표기
actual headway observed라고 잘못 표기
train_with_this_reward_allowed=true가 premature하게 설정됨
```

## 완료 기준

- reward 설계 검토 gate PASS
- 아직 최종 reward가 아니면 train_with_this_reward_allowed=false 유지
- reward candidate와 final reward 구분
- hard constraint와 normalization plan 문서화
- 다음 단계에서 실제 reward candidate 비교가 가능함

## 에이전트용 프롬프트

```text
Step 110 final reward design review gate를 작성하라.
Step 108/109의 scaffold reward가 최종 MAPPO reward로 오해되지 않도록 train_with_this_reward_allowed=false를 유지하라.
서비스 품질 우선, energy/fleet reduction 보조 원칙, p95 wait 보호, hard constraint, baseline normalization을 검토하는 checklist와 validator를 만들라.
reward_formula_finalized, reward_weights_are_final, baseline_normalization_finalized, hard_constraints_finalized가 모두 true가 아니면 최종 reward 승격을 차단하라.
```
