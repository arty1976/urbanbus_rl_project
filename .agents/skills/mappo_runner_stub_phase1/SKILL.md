---
name: mappo_runner_stub_phase1
description: "[A] 순수 MAPPO 기준선을 위한 러너 골격 파일 및 디렉터리 트리를 구성합니다."
---

# mappo_runner_stub_phase1

## 언제 이 skill을 써야 하는가
- GATv2 등 그래프 기반 RL 모델을 적용한 `[A] pure_mappo_baseline` 훈련 코드의 뼈대(Stub)를 만들 때
- 본격적인 로직 구현 전, 필수 체크리스트 요소들을 선언적으로 확보하고 씨드별 출력 레이아웃을 생성할 때

## 전제 조건
- `experiment_A_contract.json` 이 준비된 상태
- 파이썬 환경 (PyTorch, PyG 등 의존성은 코드가 Stub이므로 로드 시점에는 없어도 무방하나, 구조는 맞춰야 함)

## 입력
- 러너 설계 체크리스트 14 항목

## 실행 순서

### 1. 러너 골격 파일 생성
다음 3개의 Python 스텁 파일을 PowerShell Here-String 패치 워크플로우를 이용해 생성합니다:
1. `05_training/mappo_runner.py`
2. `05_training/run_experiment_A_stub.py`
3. `05_training/policies/mappo_policy_stub.py`

이 파일들의 클래스나 함수 안에는 다음의 **러너 설계 체크리스트** 사항이 주석 또는 파라미터 초기화 형태로 명시적으로 반영되어야 합니다:
- [ ] CTDE (Centralized Training Decentralized Execution) 구조 분리 명시
- [ ] PyG `Batch.from_data_list`를 이용한 그래프 배치 슬롯 명시
- [ ] 활성 버스 마스킹(Active Bus Masking) 로직 위치
- [ ] `edge_index`는 롤아웃 버퍼에 복사하지 않음 (제외)
- [ ] GAE (Generalized Advantage Estimation) 연산 시 `terminated` / `truncated` 분리 처리
- [ ] 첨두 적응형 엔트로피 계수 (Adaptive Entropy Coefficient)
- [ ] `grad_norm_clip = 0.5` 설정
- [ ] `rng_state`가 포함된 체크포인트 저장 구조
- [ ] `qwen_trigger_rate` 로깅 슬롯 포함 (단 [A] 기준선에서는 `0.0`으로 강제)
- [ ] `shared_policy = True` (기본값)
- [ ] GATv2 Freeze → 점진 해제 3단계 주석 처리
- [ ] Reward Normalization 슬롯 포함
- [ ] KL Divergence 모니터링 및 Early Stopping 슬롯 포함
- [ ] H200 Multi-GPU 인터페이스는 남기되 실제 구현은 보류(NotImplemented) 상태로 둘 것

### 2. Seed별 출력 디렉터리 레이아웃 생성
`run_experiment_A_stub.py` 실행 시 자동으로 다음 디렉터리와 파일을 생성하게 합니다:
- `artifacts/experiment_A_v1/runs/seed_001/`
- `artifacts/experiment_A_v1/runs/seed_002/`
- `artifacts/experiment_A_v1/runs/seed_003/`

**각 디렉터리의 산출물**:
- `run_manifest.json` (실행 메타데이터)
- `status.json` (예: `status: "stub_initialized"`)
- `checkpoint_stub.json` (임시 체크포인트 스텁)

### 3. 파일 실행 및 검증
```powershell
python 05_training/run_experiment_A_stub.py
```

## 성공 판정
- `05_training` 하위에 3개의 `.py` 파일이 모두 생성됨
- 파이썬 파일 내에 체크리스트 14항목이 주석 또는 기본 파라미터로 명확히 인지됨
- `artifacts/experiment_A_v1/runs/` 아래에 3개의 Seed 디렉터리와 내부 3개의 JSON 매니페스트 파일이 정상 생성됨

## 실패 시 복구법 및 트러블슈팅 사례

### 1. Here-String 파이썬 코드 깨짐 현상
- **증상**: 파이썬 코드 내 `f"{var}"` 구조가 PowerShell Expandable String(`@" "@`) 안에서 평가되어 버려, 엉뚱한 값이나 에러를 유발.
- **해결법**: 반드시 Literal Here-String(`@' ... '@`)을 사용해 파이썬 코드를 텍스트 그대로 파일에 밀어넣습니다.

### 2. 하위 디렉터리(policies) 생성 누락
- **증상**: `05_training/policies/mappo_policy_stub.py`를 기록하려 할 때, `policies` 폴더가 없어서 `FileNotFoundError` (또는 PS Set-Content 에러) 발생.
- **해결법**: 파일 쓰기 전에 PowerShell 명령어로 `New-Item -Path "05_training/policies" -ItemType Directory -Force`를 선행하거나, 자동 생성하는 Python 스크립트 기반 생성기를 사용해야 합니다.
