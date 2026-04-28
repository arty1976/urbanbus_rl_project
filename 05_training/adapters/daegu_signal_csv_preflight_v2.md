# Step 99 — Daegu Signal CSV Preflight for Signal Feature Builder v2

## 목적

실제 대구 신호등 CSV를 Step 98 signal feature builder에 투입하기 전에 CSV 구조와 guardrail을 검증한다.

## 핵심 컬럼

| 컬럼 | 판정 |
|---|---|
| `위도` | 정적 위치 좌표 |
| `경도` | 정적 위치 좌표 |
| `신호등관리번호` | signal_id 후보 |
| `신호등구분` | signal type/code 후보 |
| `점멸등운영여부` | blink signal flag 후보 |
| `보행자작동신호기유무` | pedestrian signal flag 후보 |
| `신호제어방식` | controlled signal metadata/proxy 후보 |
| `신호시간결정방식` | timing metadata/proxy 후보 |

## Quarantine 대상

아래 컬럼은 dynamic phase data로 쓰지 않는다.

- `신호등화순서`
- `신호등화시간`
- `점멸등운영시작시각`
- `점멸등운영종료시각`

금지 feature:

- `red_light_delay_seconds`
- `green_time_seconds`
- `cycle_length_seconds`
- `phase_sequence`
- `signal_offset_seconds`
- `real_time_signal_state`

## 산출물

```text
artifacts/signal_features_v2_preflight/daegu_signal_csv_preflight_report.json
artifacts/signal_features_v2_preflight/daegu_signal_csv_preflight_report.md
```
