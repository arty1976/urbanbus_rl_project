from __future__ import annotations

from datetime import datetime
from pathlib import Path


def main() -> None:
    project_log = Path("project_log.md")

    if not project_log.exists():
        raise SystemExit(f"[FAIL] missing project_log.md: {project_log}")

    marker = "### Baseline reference registry locked"

    current = project_log.read_text(encoding="utf-8-sig")

    if marker in current:
        print("[OK] baseline reference registry section already exists")
        return

    today = datetime.now().strftime("%Y-%m-%d")

    section = f"""

---

## 📅 {today}
### Baseline reference registry locked

오늘 작업에서는 A-family actual MAPPO 결과가 나오기 전,
비교 기준선 집합을 하나의 registry로 고정했다.

#### 1. Registry 목적

`B0R`, `B1`, `B2_calibrated`, `B0C`의 역할과 claim guard를
하나의 기준 파일로 묶어 향후 A-family actual 결과와 비교할 때
기준선 정의가 흔들리지 않도록 했다.

#### 2. Registry 포함 baseline

- `B0R_historical_12kpi_compat`
  - 과거 실제 데이터 기반 historical replay / compatibility baseline
- `B1_noop`
  - no-op replay baseline
- `B2_rulebased_calibrated`
  - 보정된 rule-based replay baseline
- `B0C_causal_shadow_v1`
  - simulator sanity validation PASS 상태의 synthetic causal-shadow candidate
  - promotion candidate지만 still locked

#### 3. Registry 산출물

생성 산출물:

- `artifacts/baseline_v1/baseline_reference_registry/baseline_reference_registry.json`
- `artifacts/baseline_v1/baseline_reference_registry/baseline_reference_registry.md`

생성 코드:

- `05_training/baselines/build_baseline_reference_registry.py`
- `05_training/baselines/validate_baseline_reference_registry.py`

#### 4. 핵심 판정

- `baseline_side_ready = true`
- `actual_policy_side_ready = false`
- `requires_h200_actual_a_family_results = true`

즉 baseline reference side는 준비됐지만,
A-family actual MAPPO 결과는 아직 H200 actual training 전이므로 준비되지 않았다.

#### 5. 유지 guard

- `causal_comparison_allowed = false`
- `paper_level_claim_allowed = false`
- `actual_results = false`
- `winner_selected = false`
- `trainable_reward_promoted = false`
- `auto_promotion_allowed = false`

#### 6. 연구적 의미

현재까지 비교 기준선 데이터는 준비됐다.

그러나 이 registry는 baseline reference set을 잠근 것이며,
실제 MAPPO 성능 비교표나 논문 수준 causal claim을 허용하는 것은 아니다.

다음 큰 단계는 H200 actual A-family 결과가 생성된 뒤,
이 registry를 기준으로 A/A90/A80/A70 및 reward family 결과를 비교하는 것이다.
"""

    project_log.write_text(current.rstrip() + section + "\n", encoding="utf-8")

    print("[OK] project_log updated with baseline reference registry section")
    print("[OK] file:", project_log)


if __name__ == "__main__":
    main()
