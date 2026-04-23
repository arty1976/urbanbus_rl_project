---
name: b0_export_parquet
description: "baseline_b0_historical_kpi_by_window 테이블을 Parquet 및 JSON 메타데이터로 추출합니다."
---

# b0_export_parquet

## 언제 이 skill을 써야 하는가
- `b0_historical_kpi_build` 스킬 완료 후, DB에 존재하는 B0 KPI 데이터를 파일 시스템(Artifacts)으로 내보낼 때
- RL 실험 환경에서 B0 베이스라인을 불러올 수 있도록 Parquet와 Metadata JSON 포맷으로 저장해야 할 때

## 전제 조건
- DB에 `baseline_b0_historical_kpi_by_window` 및 관련 메타데이터 뷰가 존재함
- Python 환경이 구성되어 있고 필수 패키지가 설치됨

## 입력
- `export_b0_historical_kpi.py` (추출 스크립트)
- 데이터베이스 접속 정보

## 실행 순서

### 1. Python 패키지 설치
반드시 다음 패키지들을 설치합니다.
```powershell
pip install pandas sqlalchemy psycopg2-binary pyarrow
```

### 2. 내보내기 스크립트 작성 및 실행 (PowerShell 붙여넣기 방식)
파일 전체를 재작성하는 PowerShell Here-String 방식을 사용해 파이썬 스크립트를 생성합니다. 이 방식은 에디터를 직접 열지 않아도 되어 자동화 환경에 적합합니다.

```powershell
$script = @'
import pandas as pd
from sqlalchemy import create_engine
import json
import os

# 디렉터리 생성 (config 디렉터리 없음 등 오류 방지)
out_dir = "artifacts/baseline_v1/B0_historical"
os.makedirs(out_dir, exist_ok=True)

# DB 연결
engine = create_engine("postgresql://ryujo:your_password@localhost:5432/daegu_bus_db")

# 데이터 내보내기
df = pd.read_sql("SELECT * FROM public.baseline_b0_historical_kpi_by_window", engine)
df.to_parquet(f"{out_dir}/kpi_by_window.parquet", index=False)

# 메타데이터 내보내기
meta_df = pd.read_sql("SELECT * FROM public.baseline_b0_historical_kpi_metadata", engine)
meta_dict = meta_df.to_dict(orient="records")[0]
with open(f"{out_dir}/metadata.json", "w", encoding="utf-8") as f:
    json.dump(meta_dict, f, indent=4)
'@
$script | Set-Content -Path "export_b0_historical_kpi.py" -Encoding UTF8

python export_b0_historical_kpi.py
```

## 성공 판정
다음 파일들이 생성되었는지 확인합니다:
- `artifacts/baseline_v1/B0_historical/kpi_by_window.parquet`
- `artifacts/baseline_v1/B0_historical/metadata.json`

## 실패 시 복구법 및 트러블슈팅 사례

### 1. config 디렉터리 없음 에러
- **증상**: 파이썬 스크립트가 대상 폴더가 존재하지 않아 FileNotFoundError를 반환함.
- **해결법**: 파이썬 코드 내에 `os.makedirs(out_dir, exist_ok=True)`를 반드시 포함하여 출력 디렉터리를 선제적으로 확보합니다.

### 2. psycopg2-binary 누락
- **증상**: `create_engine` 호출 시 드라이버 오류 발생.
- **해결법**: 필수 패키지에 `psycopg2-binary`가 제대로 설치되어 있는지 확인합니다.

### 3. Here-String 중첩 오류
- **증상**: PowerShell의 `@' ... '@` 블록 내부에 또 다른 변수 기호나 예약어가 충돌하여 변수가 null이 되는 현상.
- **해결법**: Literal string인 `@' '@` 대신 Expandable string인 `@" "@`를 사용할 때는 파이썬 변수인 `$var` 형태가 PowerShell 변수로 평가되지 않도록 주의해야 합니다. 안전하게 파이썬 코드를 작성할 때는 `@' ... '@` (단일 따옴표 Here-String)을 권장합니다.
