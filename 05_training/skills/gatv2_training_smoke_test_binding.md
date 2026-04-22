---
name: gatv2_training_smoke_test_binding
description: 전체 저장 완료된 GATv2 dataset artifact를 학습 입력 경로로 바인딩하고, Windows PowerShell 기준 CPU 기반 node-level GATv2 학습 smoke test를 수행하는 실행 절차서
---

# GATv2 Training Smoke Test Binding

본 문서는 전체 저장 완료된 GATv2(Graph Attention Network v2) dataset(학습용 데이터 묶음) artifact(산출물 파일)를 학습 스크립트에 연결하고, CPU(Central Processing Unit) 기반 소규모 학습 smoke test(기본 기능만 빠르게 검증하는 시험 실행)를 수행하는 표준 절차서입니다.

## A. 목적 및 적용 범위
본 스킬은 기존 `gatv2_dataset_build_recovery`의 후속 단계 전용으로 사용됩니다. 기존 스킬이 데이터셋을 처음부터 빌드하고 복구하는 데 초점을 맞추었다면, 본 스킬은 완성된 데이터셋을 학습 스크립트(`train_gatv2.py`)에 안전하게 바인딩하고 정상 학습이 이뤄지는지 검증하는 "학습 연결 및 smoke test"에 집중합니다. 

## B. 전제 조건
1. `dataset_full` artifact가 이미 생성되어 있어야 합니다.
   - 대상 경로: `C:\Users\ryujo\urbanbus_rl_project\05_training\artifacts\dataset_full_20260422_084243`
2. PyTorch(파이토치 딥러닝 프레임워크), PyG(PyTorch Geometric), Python(파이썬) 환경이 정상적으로 구성되어 있어야 합니다.
3. 작업 환경은 Windows PowerShell 기준입니다.
4. 파이프라인 검증용으로 노트북(Samsung Galaxy Book 5 Pro)의 CPU에서 먼저 검증을 수행하고, 전체 본훈련은 H200 서버 등 고성능 장비로 넘기는 전략을 따릅니다.

## C. 디렉터리/파일 구조 확인 절차
데이터셋 빌드 후 artifact는 루트 디렉터리에 바로 `.pt` 파일이 있지 않고, split(학습·검증·시험 분할) 기준 하위 폴더 구조로 생성됩니다. 전체 snapshot(시점 단위 데이터 묶음) 수는 6570개이며, 이 중 `train` split에는 5476개의 `.pt` 파일이 존재합니다. 

```powershell
# 1. 하위 구조 확인
Get-ChildItem "C:\Users\ryujo\urbanbus_rl_project\05_training\artifacts\dataset_full_20260422_084243"

# 2. train split .pt 파일 개수 확인
(Get-ChildItem "C:\Users\ryujo\urbanbus_rl_project\05_training\artifacts\dataset_full_20260422_084243\train\*.pt").Count
```

## D. `train_gatv2.py` 생성 절차
초기 `train_gatv2.py`가 존재하지 않았으므로 최소 학습 스크립트를 새로 작성해야 합니다. 아래 스크립트는 `train` 폴더 내 `.pt` 파일들을 순회하며 node-level(노드 단위) 학습을 수행합니다. `torch.load(..., weights_only=False)`를 사용하며, shape mismatch(텐서 차원 불일치)를 방지하기 위해 graph-level이 아닌 node-level 회귀(regression)로 설정합니다.

```python
import os
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

# 학습 설정
DATASET_DIR = r"C:\Users\ryujo\urbanbus_rl_project\05_training\artifacts\dataset_full_20260422_084243\train"
DEVICE = "cpu"
EPOCHS = 1  # 기본 예시. (안정성 검증 시 EPOCHS = 3 사용)
BATCH_SIZE = 1  # (안정성 검증 시 BATCH_SIZE = 1 사용)
MAX_FILES = 10  # 빠른 테스트용. (안정성 검증 시 MAX_FILES = 128 사용)

class NodeLevelGATv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NodeLevelGATv2, self).__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=4, concat=False)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        return x

def main():
    model = NodeLevelGATv2(in_channels=10, hidden_channels=32, out_channels=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.pt')][:MAX_FILES]
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for f in files:
            file_path = os.path.join(DATASET_DIR, f)
            sample = torch.load(file_path, weights_only=False).to(DEVICE)
            
            optimizer.zero_grad()
            out = model(sample.x, sample.edge_index, sample.edge_attr)
            
            # shape 확인 출력 (최초 1회 등 로깅 용도)
            # print(f"sample.x: {sample.x.shape}, sample.y: {sample.y.shape}, edge_attr: {sample.edge_attr.shape}")
            
            # node_mask를 train_mask로 사용하여 train 노드만 loss 계산
            mask = sample.node_mask
            loss = criterion(out[mask], sample.y[mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(files):.4f}")

    print("SMOKE TEST PASS")

if __name__ == "__main__":
    main()
```

## E. 실제 발생한 대표 오류와 복구법

### 1. `train_gatv2.py` 파일 자체가 없음
- **증상:** 스크립트 실행 시 파일 경로를 찾을 수 없음.
- **원인:** 모델 학습 단계 코드가 아직 작성되지 않은 상태였음.
- **진단:** `05_training` 디렉터리 내에 `build_gatv2_dataset.py`만 있고 학습 스크립트 부재.
- **해결:** `train_gatv2.py` 최소 스크립트를 새로 생성하여 배치.

### 2. `.pt` 파일을 루트 폴더에서만 찾아 `No .pt files found` 발생
- **증상:** `len(files) == 0`으로 학습 루프가 동작하지 않음.
- **원인:** dataset artifact가 `train/val/test` 서브 폴더로 나뉘어 저장되었으나 루트 폴더를 스캔함.
- **진단:** 루트 폴더에는 `.pt` 파일이 없고 디렉터리만 존재.
- **해결:** 데이터를 읽는 경로를 `...dataset_full_20260422_084243\train`으로 명시적 변경.

### 3. PowerShell에 `MAX_FILES = 128` 같은 Python 변수 대입을 직접 입력하여 실패
- **증상:** `MAX_FILES : The term 'MAX_FILES' is not recognized as the name of a cmdlet` 등 에러.
- **원인:** PowerShell 환경에서 Python 내부 변수를 명령어로 직접 실행하려 시도.
- **진단:** 스크립트 파라미터가 아닌 Python 코드 내 상수를 외부 쉘 커맨드로 수정 불가.
- **해결:** 파일 내용을 직접 수정(에디터 이용 또는 치환 명령어 사용)하거나 `argparse` 등을 도입하도록 스크립트를 변경.

### 4. PowerShell 문자열 치환으로 Python 파일 안에 `` `n `` 이 들어가 SyntaxError 발생
- **증상:** 파이썬 실행 시 줄바꿈 문제로 `SyntaxError` 발생.
- **원인:** PowerShell에서 치환 후 저장할 때 특수문자 이스케이프(`` `n ``)가 파이썬 코드 문자열에 그대로 삽입됨.
- **진단:** 정규식 치환 시 PowerShell의 이스케이프 문법이 텍스트 치환에서 오작동.
- **해결:** `-replace` 사용 시 원시 문자열 처리 주의. 백업본(`.bak`)을 먼저 만들고 신중히 치환 로직 작성. 안전하게 IDE나 문서 편집기를 사용하는 것을 권장.

### 5. graph-level 출력으로 `pred=[2,3]`, `target=[8232,3]` mismatch 발생
- **증상:** Loss 계산 시 shape mismatch 발생.
- **원인:** 기존 PyG 예제 코드가 그래프 단위(graph-level) 예측용 `global_mean_pool`을 사용하고 있었음.
- **진단:** 우리 문제는 노드 단위의 수요(승차/하차/대기인원)를 예측하는 것이므로 출력 shape는 `[total_nodes, 3]`이어야 함.
- **해결:** `global_mean_pool` 제거 후 node-level GATv2로 아키텍처 변경, `pred`와 `target` 구조 일치화.

## F. 실행 명령 모음 (Copy & Paste)

```powershell
# 1. artifact 하위 폴더 구조 확인
Get-ChildItem "C:\Users\ryujo\urbanbus_rl_project\05_training\artifacts\dataset_full_20260422_084243"

# 2. train 분할 내 .pt 파일 개수 확인
(Get-ChildItem "C:\Users\ryujo\urbanbus_rl_project\05_training\artifacts\dataset_full_20260422_084243\train\*.pt").Count

# 3. train_gatv2.py 스크립트 백업 생성 및 복원
Copy-Item .\05_training\train_gatv2.py .\05_training\train_gatv2.py.bak
# 복원 시: Copy-Item .\05_training\train_gatv2.py.bak .\05_training\train_gatv2.py -Force

# 4. PowerShell을 이용한 Python 설정값 자동 치환 (예: MAX_FILES를 128로 변경)
(Get-Content .\05_training\train_gatv2.py) -replace 'MAX_FILES = \d+', 'MAX_FILES = 128' -replace 'EPOCHS = \d+', 'EPOCHS = 3' | Set-Content .\05_training\train_gatv2.py

# 5. train_gatv2.py 실행
& C:\Users\ryujo\urbanbus_rl_project\05_training\.venv\Scripts\python.exe .\05_training\train_gatv2.py
```

## G. 성공 판정 기준
- [ ] `train` split 폴더 내 `.pt` 파일 탐색 성공 여부
- [ ] PyG Data 객체의 `sample.x`, `sample.y`, `edge_attr` shape 출력 성공 여부
- [ ] `pred_shape == target_shape` 일치 확인 (node-level 출력)
- [ ] `loss` 정상 계산 여부
- [ ] `backward` (역전파) 성공 여부
- [ ] `optimizer.step()` 성공 여부
- [ ] 콘솔에 `SMOKE TEST PASS` 문자열 정상 출력 확인
- [ ] 삼성 갤럭시북 5 Pro (CPU 기준) 3 epoch 안정성 검증 통과

## H. 노트북 vs 서버 운영 가이드
- **삼성 갤럭시북 5 Pro**: 파이프라인이 코드부터 데이터, 텐서 단위까지 연결되는지 확인하는 "파이프라인 검증용"으로 충분합니다. `MAX_FILES = 128`, `BATCH_SIZE = 1` 수준의 검증 환경으로 활용하십시오.
- **H200 서버**: 전체 본훈련(full-scale training)은 노트북의 CPU로는 부하가 크므로, GPU 연산이 가능한 H200 서버 같은 고성능 장비로 이관하여 진행하는 것을 권장합니다.

## I. 재발 방지 원칙
1. Python 스크립트 파일 내용 수정 전, 반드시 `.bak` 백업을 생성합니다.
2. PowerShell 명령어와 Python 코드 대입 문법을 혼동하여 콘솔에 직접 대입하지 않습니다.
3. 데이터셋 스캔 시 split 구조(`train/val/test`) 여부를 가장 먼저 확인합니다.
4. 모델 아키텍처 작성 시, 본 과제가 node-level 예측인지 graph-level 예측인지 먼저 판단합니다.

## J. 프로젝트 로그용 최종 상태 문구
(작업 완료 후 `project_log.md` 등에 기록할 내용)

```
- dataset_full_20260422_084243 successfully bound to train_gatv2.py
- Samsung Galaxy Book 5 Pro CPU-based training smoke test PASS
- 128 files / batch_size 1 / 3 epochs stability test PASS
- node-level GATv2 forward, loss, backward, optimizer step verified
- laptop is sufficient for pipeline validation, but full-scale training should be migrated to H200 server
```
