# urbanbus_rl_project reusable skills — 2026-04-29

어제·오늘 Step 99-A~109 작업 중 반복 사용 가치가 큰 절차를 스킬 문서로 추린 패키지입니다.

## 포함 스킬

1. `skill_01_bis_api_readonly_audit.md`
   - 대구 BIS API read-only 데이터 감사
   - DB/tensor DB overwrite 없이 artifact 기반으로 observed/candidate/proxy/not_observed 판정

2. `skill_02_route_aware_causal_v2_scaffold.md`
   - route-aware causal simulator v2 scaffold 구축
   - getBs02 route-stop sequence 기반 reset/step/rollout/raw_events/window_rollup 연결

3. `skill_03_claim_guard_and_metadata_preservation.md`
   - claim guard와 metadata 보존 검증
   - canonical KPI 이후 source/provenance/claim flag가 사라지지 않도록 방어

4. `skill_04_queue_demand_12kpi_reward_interface.md`
   - queue/demand 12-KPI reward-policy interface scaffold
   - 12-KPI preserved output → scaffold reward → policy interface 연결

5. `skill_05_final_reward_design_review_gate.md`
   - final reward design review gate
   - Step 110에서 scaffold reward를 실제 MAPPO reward로 승격하기 전 검토 gate

6. `skill_06_windows_git_artifact_hygiene.md`
   - Windows PowerShell + Git + artifact hygiene
   - git add ., artifacts 강제 커밋, push 상태 혼동, WinError 5 방지

## 사용법

새창에서 특정 작업을 이어갈 때 해당 skill 파일의 “에이전트용 프롬프트” 블록을 그대로 붙여넣으면 됩니다.

## 중요한 공통 guard

```text
actual_headway = not_observed
actual_arrival_departure_time = not_observed
actual_dwell = not_observed
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
reward_scaffold_only = true
train_with_this_reward_allowed = false
final_reward_design_claim_allowed = false
```
