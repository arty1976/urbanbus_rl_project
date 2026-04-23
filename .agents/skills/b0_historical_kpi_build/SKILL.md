---
name: b0_historical_kpi_build
description: "public.rl_stop_transition_features 기반으로 B0 Historical KPI 메타데이터 및 윈도우 집계 테이블을 생성합니다."
---

# b0_historical_kpi_build

## 언제 이 skill을 써야 하는가
- RL 환경의 Baseline 중 `[B0] Historical` 성능 지표를 DB에 구축해야 할 때
- `public.rl_stop_transition_features` 테이블이 완성된 후, B0 Baseline을 추출해야 할 때

## 전제 조건
- `public.rl_stop_transition_features` 테이블 생성 완료 및 데이터 검증 통과
- PostgreSQL 서버 구동 확인
- `psql.exe` 명령 사용 가능 환경 (Windows 환경 중심)

## 입력
- `create_b0_historical_kpi.sql` (KPI 생성 쿼리)
- `run_b0_historical_kpi.ps1` (PowerShell 실행 러너)

## 실행 순서

### 1. SQL 스크립트 작성
DB 테이블을 생성하는 `create_b0_historical_kpi.sql` 스크립트를 작성합니다.
이 과정에서 주의할 점:
- `integer/boolean` 타입의 `COALESCE` 오류 방지 (예: `COALESCE(bool_column, false)` 등 타입 일치)

### 2. PowerShell 러너 스크립트 작성
Windows 환경에서 `.ps1` 러너 스크립트를 작성합니다. 다음과 같은 패턴을 준수해야 합니다.
- **psql 자동 탐색**: `C:\Program Files\PostgreSQL\*\bin\psql.exe` 위치를 자동으로 찾아내는 로직 포함
- **환경 변수 방식 연결**: 스크립트 내 하드코딩 대신 `$env:PGHOST`, `$env:PGPORT`, `$env:PGUSER`, `$env:PGPASSWORD`, `$env:PGDATABASE` 방식으로 연결

예시 스크립트 구조:
```powershell
$env:PGUSER = "ryujo"
$env:PGDATABASE = "daegu_bus_db"
# ... set other env vars ...

$psql_paths = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue
if ($psql_paths.Count -eq 0) { throw "psql.exe not found" }
$psql = $psql_paths[0].FullName

& $psql -v ON_ERROR_STOP=1 -f "create_b0_historical_kpi.sql"
```

### 3. 러너 실행
```powershell
.\run_b0_historical_kpi.ps1
```

## 성공 판정
생성된 테이블(`public.baseline_b0_historical_kpi_metadata`)의 검증 쿼리 결과가 다음과 같아야 성공입니다:
- `row_count = 6570`
- `wait_filled_rows = 6570`
- `cv_headway_null_rows = 6570`
- `zero_intervention_rows = 6570`

## 실패 시 복구법 및 트러블슈팅 사례

### 1. psql PATH 미인식 문제
- **증상**: `psql` 명령어를 인식하지 못함.
- **해결법**: `Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe"`를 통해 절대 경로를 찾아내어 `& $psql` 형태로 호출합니다.

### 2. DB 사용자 인증 실패 (ryujo)
- **증상**: 환경 변수 미설정으로 기본 Windows 로그인 계정 등으로 접속 시도하여 실패.
- **해결법**: 러너 스크립트에서 `$env:PGUSER="ryujo"` 등 명시적으로 환경변수를 선언합니다.

### 3. integer vs boolean COALESCE 오류
- **증상**: `COALESCE` 함수 내에 integer와 boolean 등 호환되지 않는 타입을 혼용하여 쿼리 실패.
- **해결법**: 테이블 스키마에 맞춰 정확한 타입을 지정합니다. `COALESCE(cast(col as integer), 0)` 또는 `COALESCE(bool_col, false)`.

### 4. preview.sql BOM 문제
- **증상**: `preview.sql` 실행 시 `\ufeff` 등 UTF-8 BOM으로 인해 구문 오류(Syntax Error) 발생.
- **해결법**: PowerShell에서 파일 저장 시 UTF-8 (BOM 없음)으로 저장해야 합니다. `Set-Content` 시 Windows PowerShell 5.1의 기본 인코딩 문제에 유의하고, 가급적 붙여넣기형(Patch) 워크플로우를 활용합니다.
