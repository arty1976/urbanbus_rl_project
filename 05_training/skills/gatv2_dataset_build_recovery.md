---
name: gatv2_dataset_build_recovery
description: GATv2(Graph Attention Network v2) 데이터셋 빌드 시 발생하는 환경 오류, DB 연결 오류, 스키마 계약 불일치 및 성능 병목 문제를 진단하고 복구하는 실행 절차서(Runbook)
---

# 🚌 GATv2 Dataset Build: Recovery & Troubleshooting Guide

본 문서는 `urbanbus_rl_project`에서 GATv2(Graph Attention Network v2=그래프 어텐션 네트워크 2버전) 파이프라인의 데이터셋 생성을 수행할 때 발생하는 다양한 오류들을 신속하게 진단하고 복구하기 위한 **표준 복구 스킬(Skill)** 문서입니다. 
추상론을 배제하고 **Windows PowerShell(Windows PowerShell=윈도우 파워셸)** 기준의 재현 가능한 실제 명령과 로그에 바탕을 둔 Runbook(Runbook=실행 절차서) 및 Validation Checklist(Validation Checklist=검증 체크리스트)를 제공합니다.

---

## A. 문제 개요

`run_build_dataset.ps1` 실행을 시작으로 `build_gatv2_dataset.py`의 dry-run(dry-run=실제 저장 없이 시험 실행)이 성공하기까지의 과정은 단순한 문법 오류 수정을 넘어서는 문제였습니다.
본 장애는 단일 원인이 아니라 파이프라인 각 계층 간의 **연쇄적 계약 불일치(Contract Mismatch=계약 불일치)** 가 원인이었습니다.

1. **환경 계층**: PowerShell 파서 및 실행 정책 통제, Python 패키지 의존성 부재.
2. **연결 계층**: PostgreSQL(PostgreSQL=오픈소스 관계형 데이터베이스) 드라이버 호환성 한계와 인증(Authentication) 오류 오판.
3. **스키마 계층**: 원본 테이블과 뷰, 모델이 기대하는 컬럼명/데이터 타입 간의 규격 불일치.
4. **성능 계층**: 뷰의 CTE 반복 전개로 인한 I/O 병목 및 메모리 지연 문제.

위 4가지 계층의 문제를 하나씩 해결하여 최종적으로 ~4초대의 고속 dry-run 파이프라인을 완성한 과정을 아래에 정리합니다.

---

## B. 1차 실행/환경 오류 복구

최초 스크립트 실행 시 Windows 환경 특유의 파서 및 권한 문제, 인자 처리 문제가 발생합니다.

- **증상 1**: `param(...)` 관련 문법 오류 및 스크립트 실행 불가.
  - **원인**: 이전 스크립트에서 매개변수 선언부가 파일 최상단이 아닌 곳에 위치했거나, PowerShell 5.1의 구문 분석기와 충돌.
  - **해결**: 인자 파싱을 커스텀 반복문(while)으로 교체하고 표준 옵션 방식(`--max-snapshots`, `--dry-run`)으로 파싱 로직을 격리하여 해결.
- **증상 2**: `ExecutionPolicy (Execution Policy=스크립트 실행 정책)` 차단.
  - **원인**: Windows 기본 보안 정책으로 인해 서명되지 않은 셸 스크립트(.ps1) 실행 막힘.
  - **해결**: 관리자 권한 없이 현재 프로세스에 대해서만 실행 정책을 바이패스(Bypass).
    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
    ```
- **증상 3**: `build.log` 파일 잠금(Lock) 충돌.
  - **원인**: 파이프라인 로깅 시 Python 스크립트 내 로거와 PowerShell `Tee-Object`가 동일한 파일을 물고 있어 I/O 충돌 발생.
  - **해결**: Python의 애플리케이션 로그는 `build.log`에 저장하고, PowerShell 콘솔의 에러 스트림을 포함한 전체 출력은 `runner_console.log`로 별도 분리.

---

## C. Python 환경 복구

- **증상**: `ModuleNotFoundError: No module named 'torch'` 및 `torch_geometric` 찾을 수 없음.
  - **원인**: 시스템 기본 가상환경 또는 잘못된 전역 Python 인터프리터를 참조함.
  - **진단**: PowerShell에서 `Get-Command python` 또는 Python 경로 확인.
  - **해결**: 프로젝트 전용 또는 올바른 글로벌 인터프리터에 의존성을 명시적으로 설치.
    ```powershell
    # Windows 환경 인터프리터 경로를 명시하여 패키지 설치
    & "C:\Users\ryujo\AppData\Local\Programs\Python\Python312\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    & "C:\Users\ryujo\AppData\Local\Programs\Python\Python312\python.exe" -m pip install torch_geometric pandas sqlalchemy psycopg2 pg8000
    ```
- **주의점**: Windows 환경 특성 상 여러 Python 버전이 충돌할 수 있으므로, .ps1 래퍼 스크립트 내부에서 `$PythonExe` 변수에 명시적 경로(`C:\...\python.exe`)를 고정하여 환경 일관성을 강제합니다.

---

## D. PostgreSQL 연결 문제

- **증상**: Python 스크립트 DB 접속 시 `psycopg2`에서 `UnicodeDecodeError` 발생.
  - **진단**: 드라이버 오류인지 실제 서버 오류인지 판단하기 위해 DB 드라이버를 `pg8000`로 강제 지정(`--db-driver pg8000`)하여 로그를 확인.
  - **결과**: 인코딩 뒤에 숨어있던 실제 원인이 `28P01: password authentication failed for user "postgres"` (인증 실패) 였음을 발견.
- **원인**: 시스템 환경변수의 DSN과 실제 접속 비밀번호("siwoo")가 불일치했으며, `localhost` 사용 시 Windows IPv6(`::1`) / IPv4(`127.0.0.1`) 해상도 차이로 인해 pg_hba.conf가 인증 방식을 거부함.
- **해결**: 서버 주소에 `localhost` 대신 암묵적 변수가 없는 `127.0.0.1`을 명시하고, 사용자명/비밀번호를 조합한 DB URL(DB URL=데이터베이스 접속 주소) 인자를 명시적으로 전달.
  - **비밀번호 확인용 Python 한 줄 테스트**:
    ```powershell
    python -c "from sqlalchemy import create_engine; print(create_engine('postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus').connect().scalar(text('select 1')))"
    ```

---

## E. 스키마/뷰 계약 불일치 복구

DB 연결 후 실제 데이터를 당겨오는 `T2_load_graph_static` 등에서, PyG 모델이 요구하는 계약(컬럼명)과 실제 DB 뷰(View)의 컬럼명 규격이 맞지 않는 연쇄적 오류가 터졌습니다.

### 1. `gatv2_node_master_active`
- **문제**: 모델은 `node_idx`를 기대했으나 테이블에는 `node_index`로 정의됨.
- **해결**: 모델 스크립트나 매트리얼라이즈 단계에서 명시적으로 `node_index AS node_idx`로 조회하도록 알리아싱(Aliasing) 적용을 통해 계약 호환성 확보.

### 2. `gatv2_edge_primary_active`
- **문제**: 모델이 기대하는 노드 식별자가 `src_idx` / `dst_idx` 였으나 소스는 `src_node_index` / `dst_node_index` 였음. 또한 피처 컬럼으로 기대한 `generalized_cost`와 `long_edge_5km_flag`가 누락됨.
- **해결**: 해당 뷰의 SELECT 쿼리에 누락된 파생 특성을 상수로 채워 넣고 이름 불일치를 교정하는 별도 호환 뷰 패치 적용(임시 또는 Materialized 처리).

### 3. `gatv2_snapshot_summary`
- **문제**: 데이터 로더는 `snapshot_id`와 데이터 시간의 메타 정보를 담은 통계 테이블의 스냅샷 단위(snapshot-level=스냅샷 단위) 정보를 원했으나, 실제 뷰는 전체 테이블 통합(1행) 결과를 던지고 있었음.
- **해결**: `GROUP BY state_ts` 및 `DENSE_RANK() OVER (ORDER BY state_ts)` 윈도우 함수를 사용하여 시간별 타임슬라이스를 `snapshot_id`로 분해하는 올바른 뷰(View)로 완전 재작성 수행.

### 4. `gatv2_snapshot_stop_features_train_mat`
- **문제**: 원본 테이블과 그 위를 덮고 있는 뷰, 그리고 학습 속도를 위한 실체화 뷰(Materialized View)까지 3단계로 분리되는 과정에서 `node_idx` 변환 적용 누락이 잦음. 
- **해결**: 베이스 테이블(`_base`) 위로 일괄 변환 규칙(`node_index AS node_idx`)을 담은 매트리얼라이즈드 뷰 로직 래핑 후, 모든 조회를 이 실체화 뷰 하나로만 하도록 교통 정리 완료.

---

## F. Materialized View (뷰 실체화) 전략

- **`gatv2_snapshot_summary` 실체화 이유**
  `DENSE_RANK()`, `LEAD()` 등 윈도우 분석 함수가 매번 파이프라인 접근 시마다 수백만 건의 원천 테이블 스캔 및 정렬을 유발합니다. 이를 일회성 테이블로 굳혀 불필요한 반복 통계 스캔을 방지합니다.
- **`gatv2_snapshot_stop_features_train_mat` 실체화 이유**
  2,000만 건 규모의 조인과 필터가 뷰로 남아있으면, 딥러닝 미니배치가 `snapshot_id`로 잘라 읽을 때 매번(I/O 병목) 원본 쿼리(CTE)가 재실행됩니다. 이 I/O 참사를 막기 위해 인덱스가 포함된 물리적 Materialized View를 구축합니다.
- **운영 성능 개선 및 유지보수 규칙**
  - **효과**: 매트리얼라이즈 전략 이후 스냅샷 1개 로드 속도가 **초 단위에서 0.1초 미만으로** 수직 상승했습니다.
  - **운영 룰**: 데이터 갱신 시 `REFRESH MATERIALIZED VIEW public.gatv2_...` 수행. 컬럼 구조가 바뀔경우 `DROP` 후 재생성 모드(`-Rebuild`) 사용 권장.

---

## G. 성능 병목 진단 결과 

실제 벤치마크 테스트에서의 성능 향상 로그 수치 비교표입니다.

| 단계 | 적용 전 (초기 뷰 상태) | 적용 후 (최적화 완료 상태) |
|---|---|---|
| `T3_load_snapshot_summary` | 약 8 ~ 10 초 | **약 0.48 초** |
| `T4_fetch_snapshot_rows` | 약 1.5 ~ 2.5 초 | **약 0.09 ~ 0.10 초** |
| `max-snapshots 2 --dry-run` 총합 | N/A (I/O 병목 렉) | **약 4.2 ~ 4.5 초** |

- **해석**: 6,570개의 스냅샷을 1초에 1개씩 읽어서는 모델 학습에만 며칠이 소요됩니다. 0.09초 대역의 `fetch` 속도는 I/O 스캔 로직을 CTE 전개 대신 B-Tree Index Seek로 완벽히 전환했음을 의미합니다.
- **Cold run vs Warm run(cold run=캐시가 비어 있는 첫 실행, warm run=캐시가 따뜻한 반복 실행)**: 
  PostgreSQL 재시작 후 첫 벤치마크(Cold)에서는 `T4_fetch`가 디스크를 스윕하므로 1초에 가까울 수 있으나, 연속 실행(Warm) 시 메모리 캐싱(`shared_buffers`)을 타며 **0.09초 안정폭**으로 떨어집니다.

---

## H. PyTorch 경고(Warning) 정리 및 코드 패치

데이터셋 빌드 중 NumPy 관련 경고 및 이에 대응하는 패치 작업 이력입니다.

- **원인**: `pandas`의 `to_numpy()` 함수는 복사 비용을 줄이기 위해 읽기 전용(Read-only View) 메모리를 반환하는 경우가 잦습니다. 이를 `.float32` 상태로 `torch.tensor()`로 던지면 **non-writable NumPy array(non-writable NumPy array=쓰기 불가 넘파이 배열)** 경고가 발생합니다. PyTorch가 내부적으로 예측 불가능한 메모리 변조를 우려하는 것입니다.
- **라인 수정 흐름 (`x`, `y`, `node_mask`)**:
  단순 참조 대입에서 `.tolist()`를 거치거나 `np.array(..., copy=True)`를 강제하여 명시적으로 메모리를 분리해주어야 합니다.
  ```python
  # 변경 전 
  x[node_indices] = torch.from_numpy(x_vals)
  # 변경 후 (안전한 우회법)
  node_indices_t = torch.tensor(node_indices.tolist(), dtype=torch.long)
  x[node_indices_t] = torch.tensor(x_vals.tolist(), dtype=torch.float32)
  ```
- **자동 패치의 위험성 (SyntaxError / IndentationError 방지)**:
  정규식 기반 자동 문자열 치환(Replace) 방식으로 코드를 조작할 때 공백이나 탭이 어긋나 `IndentationError` 가 발생한 바 있습니다. 파이썬 문법의 치명성을 고려할 때, **원본 교체보다 임시 Python 패치 파일을 실행하여 코드를 파싱/수정**하거나 단단한 정규식을 사용하고, 작업 전에 **반드시 `.bak` 백업 파일 스냅샷을 만들어두는 것**이 월등히 안전합니다.

---

## I. 재발 방지 표준 절차 (Standard Procedure Checklist)

만약 앞서 다룬 환경, 연결, 스키마, 성능 관련 유사 문제가 재발한다면 **반드시 아래 체크리스트 순서대로 접근**하시기 바랍니다.

- [ ] **1. 실행 스크립트 문법 및 파라미터 확인**: `--help`를 통해 명령줄 인자를 정상 파싱하는지 확인.
- [ ] **2. PowerShell 실행 정책(Execution Policy) 확인**: `Set-ExecutionPolicy -Scope Process Bypass` 처리 여부.
- [ ] **3. Python 인터프리터 경로 확인**: `$PythonExe`가 정확히 설치된 전역/가상환경 경로를 지시하는지 점검. 패키지 설치 여부(torch, pyg, psycopg2 등).
- [ ] **4. DB 접속 단독 테스트**: JDBC URL과 로그인 정보, 호스트 주소를 Python 스니펫으로 1줄 실행하여 순수 네트워크/권한 이슈 여부 판단.
- [ ] **5. View, Table, Materialized View의 `relkind(relkind=PostgreSQL 객체 종류)` 확인**: `select relkind from pg_class` 를 통해 뷰('v'), 테이블('r'), 실체화 뷰('m')가 올바른 상태인지 확인.
- [ ] **6. 컬럼 계약(Column Contract) 확인**: 소스 테이블의 컬럼 이름과 PyTorch Builder 스크립트 상단의 `X_COLS`, `Y_COLS` 명칭이 정확히 합치하는지 점검. 
- [ ] **7. 성능 병목 검증 (Cold/Warm 확인)**: 전체 실행 대신 `--max-snapshots 5` 정도의 숏 레이스로 I/O 시간을 측정하고 1초 이상의 지연이 있다면 인덱스 구조나 뷰 스캔을 의심.

---

## J. 실행 명령 모음 (Copy & Paste)

현업에서 즉각적으로 복사하여 응용할 수 있는 관리 명령 모음입니다.

**1. 파이프라인 빠른 검증 (Dry-run)**
```powershell
.\run_build_dataset.ps1 --db-url "postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus" --out-dir ".\out_temp" --max-snapshots 2 --dry-run
```

**2. DB 연결 자격증명 단순 체크 (1줄 핑)**
```powershell
python -c "from sqlalchemy import create_engine, text; print('DB Ping:', create_engine('postgresql+pg8000://postgres:siwoo@127.0.0.1:5432/urbanbus').connect().scalar(text('SELECT 1')))"
```

**3. 현재 객체가 테이블인지 뷰인지(relkind) 확인**
```sql
SELECT c.relname, c.relkind 
FROM pg_class c 
JOIN pg_namespace n ON n.oid = c.relnamespace 
WHERE n.nspname = 'public' AND c.relname LIKE 'gatv2%';
-- r=table, v=view, m=materialized view
```

**4. Materialized View 전체 갱신 (Refresh 및 인덱스 재생성)**
```powershell
.\materialize_gatv2_snapshot_summary.ps1
.\materialize_gatv2_snapshot_rows.ps1
```

**5. 파이썬 코드 강제 패치 및 백업**
```powershell
Copy-Item .\build_gatv2_dataset.py .\build_gatv2_dataset.py.bak -Force
# (문제가 생기면 복원)
Copy-Item .\build_gatv2_dataset.py.bak .\build_gatv2_dataset.py -Force
```

**6. 병목 구간 로그 확인**
```powershell
Select-String -Path .\logs_build_*\build.log -Pattern "\[perf\].*state_ts=" | Select-Object -Last 10
```

---

## K. 프로젝트 최종 상태 보고

### 현재 최종 안정 상태
방향성 있는 문제 해결 과정을 통해 파이프라인은 현재 아래의 최고 효율 상태를 보증합니다.
- **전체 시퀀스 정상 동작**: `run_build_dataset.ps1 --db-url $dburl --max-snapshots 2 --dry-run` 정상 완료
- **속도 지표**: 
  - `T3_load_snapshot_summary`: **약 0.48초**
  - `T4_fetch_snapshot_rows`: **약 0.09초대**
  - 초기 환경 부트스트랩을 포함하여 총합 성능 **약 4~5초대** (최단시간 컷 통과)

### 남은 후속 작업
1. `--max-snapshots 100 --dry-run` 을 통한 대형 배치 확장 안전성 검증 수행.
2. `--dry-run` 인자 제거 후 하드디스크 I/O 시나리오(.pt 실제 아티팩트 저장 모드) 작동 점검.
3. 생성된 스냅샷 텐서 아티팩트를 GATv2 모델 학습 파이프라인(train_gatv2.py)에 고정 바인딩.
