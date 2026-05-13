from __future__ import annotations

from datetime import datetime
from pathlib import Path


def main() -> None:
    project_log = Path("project_log.md")

    if not project_log.exists():
        raise SystemExit(f"[FAIL] missing project_log.md: {project_log}")

    marker = "### B0C causal shadow baseline validation progress"

    current = project_log.read_text(encoding="utf-8-sig")

    if marker in current:
        print("[OK] B0C validation progress section already exists")
        print("[OK] file:", project_log)
        return

    today = datetime.now().strftime("%Y-%m-%d")

    section = f"""

---

## 📅 {today}
### B0C causal shadow baseline validation progress

오늘 작업에서는 `B0C_causal_shadow_v1`을 생성하고,
B0R historical reference와 구분되는 synthetic causal-shadow baseline으로
정의한 뒤 simulator validation candidate 단계까지 진행했다.

#### 1. B0C baseline identity

B0C는 B0R을 대체하지 않는다.

- `B0R_historical_12kpi_compat`
  - 과거 실제 데이터 기반 historical replay / compatibility baseline
- `B0C_causal_shadow_v1`
  - Phase 2 simulator 내부에서 historical / no-intervention /
    full-fleet 조건을 재현하는 synthetic causal-shadow baseline

B0C의 핵심 정책 조건은 다음과 같다.

- `condition_id = B0C`
- `baseline_id = B0C_causal_shadow_v1`
- `policy_source = historical_shadow_policy`
- `action_mode = maintain_no_learned_intervention`
- `active_bus_ratio = 1.0`
- `fleet_reduction_ratio = 0.0`
- `qwen_train = false`
- `qwen_inference = false`
- `qwen_trigger_rate = 0.0`

#### 2. B0C-1 contract

생성 파일:

- `05_training/baselines/b0c_causal_shadow_baseline_contract.md`
- `05_training/baselines/b0c_causal_shadow_baseline_contract.json`
- `05_training/baselines/validate_b0c_causal_shadow_baseline_contract.py`

검증 결과:

- B0C contract validation PASS
- 12-KPI schema included
- claim guard false 유지

#### 3. B0C-2 rollout writer

생성 파일:

- `05_training/baselines/run_b0c_causal_shadow_rollout.py`
- `05_training/baselines/validate_b0c_causal_shadow_rollout.py`

생성 구조:

- seeds = `[1, 2, 3]`
- windows per seed = `6,570`
- total windows = `19,710`
- raw events per seed = `65,700`
- total raw events = `197,100`
- 12-KPI columns present

#### 4. B0C-3 canonical KPI aggregation

B0C rollout output을 `canonical_kpi_aggregator.py`의
`official_rollup` 모드에 연결했다.

생성 산출물:

- `kpi_by_window.parquet`
- `kpi_by_seed.parquet`
- `kpi_by_time_band.parquet`
- `kpi_overall.json`
- `aggregation_manifest.json`

중요 수정:

- B0C source mode:
  `noncausal_B0C_causal_shadow_scaffold_not_validated_v1`
- canonical aggregator가 이를 causal로 오판하지 않도록
  `noncausal` 및 `not_validated` pattern을 반영했다.

최종 guard:

- `causal_comparison_allowed = false`
- `paper_level_claim_allowed = false`

#### 5. B0C-4 calibration sanity check

생성 파일:

- `05_training/baselines/compare_b0r_b0c_calibration_sanity.py`

역할:

- B0R historical compatibility reference와
  B0C synthetic causal-shadow baseline의 KPI scale 차이를 기록한다.
- 이 단계는 성능 비교가 아니라 calibration gap 기록이다.

#### 6. B0C-5 simulator validation tolerance contract

생성 파일:

- `05_training/baselines/b0c_simulator_validation_tolerance_contract.md`
- `05_training/baselines/b0c_simulator_validation_tolerance_contract.json`
- `05_training/baselines/validate_b0c_simulator_validation_tolerance_contract.py`

주요 기준:

- condition_id = B0C
- seeds = `[1, 2, 3]`
- window rows = `19,710`
- windows per seed = `6,570`
- 12-KPI schema present
- qwen_trigger_rate = `0.0`
- intervention_rate = `0.0`
- fleet_reduction_ratio = `0.0`
- passenger_service_rate sanity range
- energy_proxy_per_passenger positive and finite

#### 7. B0C-6 simulator validation report

생성 파일:

- `05_training/baselines/build_b0c_simulator_validation_report.py`

검증 결과:

- `audit_status = PASS`
- `hard_failures = []`
- `warnings = []`
- `b0c_windows = 19,710`
- `b0c_seeds = [1, 2, 3]`
- `b0c_time_bands = [night, offpeak, peak]`
- `qwen_trigger_rate = 0.0`
- `b0c_causal_comparison_allowed_in_overall = false`
- `b0c_causal_comparison_allowed_in_manifest = false`

의미:

- B0C는 simulator sanity validation을 통과했다.
- 하지만 real-world operational improvement 증거는 아니다.

#### 8. B0C-7 promotion candidate manifest

생성 파일:

- `05_training/baselines/b0c_validation_promotion_candidate_manifest.py`
- `05_training/baselines/validate_b0c_validation_promotion_candidate_manifest.py`

최종 지위:

- `manifest_status = PROMOTION_CANDIDATE_STILL_LOCKED`
- `promotion_candidate = true`
- `promotion_released = false`
- `causal_comparison_allowed = false`
- `paper_level_claim_allowed = false`
- `operator_approval_required = true`
- `explicit_release_gate_required = true`

#### 9. 현재 연구적 의미

B0C의 현재 상태는 다음과 같다.

- simulator sanity validation PASS
- promotion candidate
- still locked
- not automatically released
- not paper-level evidence
- not real-world improvement proof

즉 B0C는 Phase 2 simulator 내부 비교 후보로 사용할 준비가 되었지만,
실제 causal comparison release는 아직 열리지 않았다.

#### 10. 유지해야 하는 guard

- `causal_comparison_allowed = false`
- `paper_level_claim_allowed = false`
- `actual_results = false`
- `winner_selected = false`
- `trainable_reward_promoted = false`
- `auto_promotion_allowed = false`
"""

    updated = current.rstrip() + section + "\n"
    project_log.write_text(updated, encoding="utf-8")

    print("[OK] project_log updated with B0C validation progress")
    print("[OK] file:", project_log)


if __name__ == "__main__":
    main()
