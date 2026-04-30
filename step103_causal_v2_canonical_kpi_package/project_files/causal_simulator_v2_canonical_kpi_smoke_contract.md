# Step 103 — CausalSimulatorAdapter v2 Rollout to Canonical KPI Aggregator Smoke

## 0. 목적

Step 103의 목적은 Step 102에서 생성한 `window_rollup.parquet`를 기존 `canonical_kpi_aggregator.py`의 `official_rollup` 모드에 연결하는 것이다.

즉 흐름은 다음과 같다.

```text
CausalSimulatorAdapter v2
→ raw_events.parquet
→ window_rollup.parquet
→ canonical_kpi_aggregator.py
→ kpi_by_window.parquet
→ kpi_by_seed.parquet
→ kpi_by_time_band.parquet
→ kpi_overall.json
```

---

## 1. 왜 필요한가

Step 102는 시뮬레이터 결과를 `window_rollup.parquet`로 저장했다.

하지만 연구 파이프라인에서 중요한 것은 최종적으로 모든 조건의 결과가 같은 KPI 집계기에서 같은 방식으로 계산되는 것이다.

따라서 Step 103은 다음을 확인한다.

1. Step 102 rollout 산출물이 official KPI 입력 형식에 맞는가
2. canonical KPI aggregator가 해당 파일을 읽을 수 있는가
3. 6개 shared KPI가 계산되는가
4. smoke/nonperformance source mode 때문에 causal comparison은 false로 남는가
5. 성능 주장 guardrail이 유지되는가

---

## 2. 중요한 변환

Step 102의 `state_ts`는 smoke용 문자열이다.

```text
step_0000
```

기존 `canonical_kpi_aggregator.py`는 `state_ts`를 datetime으로 파싱한다.

따라서 Step 103은 원본 `window_rollup.parquet`를 수정하지 않고, prepared input copy를 만든다.

```text
artifacts/causal_simulator_v2_canonical_kpi_smoke/prepared_input/...
```

prepared copy에서는 `state_ts`를 다음처럼 안전한 timestamp로 바꾼다.

```text
2026-01-01T00:00:00+00:00
```

---

## 3. Claim guardrail

Step 103 결과도 성능 주장이 아니다.

계속 유지:

```text
performance_claim_allowed = false
causal_performance_claim_allowed = false
dynamic_signal_phase_claim_allowed = false
```

`source_mode`에는 smoke/nonperformance가 들어가므로 aggregator 결과의 `causal_comparison_allowed`도 false여야 한다.

---

## 4. 생성 산출물

```text
artifacts/causal_simulator_v2_canonical_kpi_smoke/contract/c2_static_signal_canonical_contract.json
artifacts/causal_simulator_v2_canonical_kpi_smoke/prepared_input/...
artifacts/causal_simulator_v2_canonical_kpi_smoke/scenario_index.parquet
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_eval/kpi_by_window.parquet
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_eval/kpi_by_seed.parquet
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_eval/kpi_by_time_band.parquet
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_eval/kpi_overall.json
artifacts/causal_simulator_v2_canonical_kpi_smoke/canonical_kpi_smoke_manifest.json
```

---

## 5. 다음 단계

Step 104에서는 Step 97~103 내용을 project_log.md와 runbook에 기록한다.
