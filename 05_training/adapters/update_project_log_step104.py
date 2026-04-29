from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

STEP104_MARKER_START = "<!-- STEP104_ROUTE_AWARE_V2_LOG_START -->"
STEP104_MARKER_END = "<!-- STEP104_ROUTE_AWARE_V2_LOG_END -->"
DEFAULT_RUNBOOK_REL = Path("05_training") / "adapters" / "route_aware_v2_pipeline_runbook_step104.md"
DEFAULT_README_REL = Path("05_training") / "adapters" / "project_log_runbook_update_step104.md"
DEFAULT_REPORT_ROOT = Path("artifacts") / "daegu_bis_api_audit" / "project_log_runbook_step104"

PROJECT_LOG_BODY = r'''
## 📅 2026-04-29
### Step 99-D~103 BIS API 통합 감사 및 route-aware causal simulator v2 scaffold 연결 완료

오늘 작업에서는 Step 97/98에서 제기된 데이터 가용성 문제를 API(Application Programming Interface=응용 프로그램 인터페이스) artifact 기반으로 재검증하고, 그 결과를 route-aware causal simulator v2의 최소 실행 파이프라인까지 연결했다. 이번 범위는 **DB(Database=데이터베이스) write 없음**, **tensor DB(Database=데이터베이스) overwrite 없음**, **추가 API 호출 없음**, **성능 주장 금지** 원칙을 유지한 구조 검증이다.

#### 1. Step 99-D — `/getRealtime02` ETA sampling audit 완료
- 생성/검증 파일:
  - `05_training/adapters/inspect_getrealtime02_eta_sampling.py`
  - `05_training/adapters/test_getrealtime02_eta_sampling.py`
  - `05_training/adapters/run_step99d_getrealtime02_eta_sampling.ps1`
- 주요 결과:
  - normalized ETA rows = 58
  - ETA-based headway candidates = 23
  - `getRealtime02_eta = observed_candidate`
  - `eta_based_headway = candidate_from_getRealtime02`
- 단, 다음 값들은 계속 `not_observed`로 유지:
  - `actual_headway`
  - `actual_arrival_departure_time`
  - `actual_dwell`

#### 2. Step 99-E — BIS API data source integration audit 완료
- 생성/검증 파일:
  - `05_training/adapters/analyze_bis_api_integration_step99e.py`
  - `05_training/adapters/test_bis_api_integration_step99e.py`
  - `05_training/adapters/bis_api_integration_audit_step99e.md`
- 주요 결과:
  - `audit_status = PASS`
  - `getPos02 coverage = 1.0`
  - `getRealtime02 ETA coverage = 1.0`
  - `headway candidate coverage = 1.0`
- 의미:
  - `/getPos02` 현재 위치 후보가 `/getBs02` route-stop sequence 위에 100% 매칭됨.
  - `/getRealtime02` ETA 후보도 `/getBs02` route-stop sequence 위에 100% 매칭됨.
  - ETA 기반 headway candidate도 route-aware simulator v2 입력 후보로 연결 가능함.

#### 3. Step 99-F — Step 97/98 classification update consolidation 완료
- 생성/검증 파일:
  - `05_training/adapters/consolidate_bis_api_classification_step99f.py`
  - `05_training/adapters/test_bis_api_classification_step99f.py`
  - `05_training/adapters/bis_api_classification_consolidation_step99f.md`
- 주요 결과:
  - `audit_status = PASS`
  - `getPos02 coverage = 1.0`
  - `getRealtime02 ETA coverage = 1.0`
  - `headway candidate coverage = 1.0`
- classification update 핵심:
  - `route_id = observed_candidate_full_collection_234_of_238_routes`
  - `direction_id = observed_candidate_full_collection_234_of_238_routes_cross_confirmed`
  - `ordered_stop_sequence = observed_candidate_full_collection_234_of_238_routes`
  - `bus_id_or_vehicle_no = repeated_sample_observed_candidate_from_getPos02_vhcNo2`
  - `live_position_xy = repeated_sample_observed_candidate_from_getPos02_xPos_yPos`
  - `current_route_sequence = repeated_sample_observed_candidate_from_getPos02_seq`
  - `current_stop_id = repeated_sample_observed_candidate_from_getPos02_bsId`
  - `getRealtime02_eta = observed_candidate_cross_matched_to_getBs02`
  - `eta_based_headway = candidate_from_getRealtime02_cross_matched_to_getBs02`
- 유지되는 방어선:
  - `actual_headway = not_observed`
  - `actual_arrival_departure_time = not_observed`
  - `actual_dwell = not_observed`
  - `paper_level_claim_allowed = false`
  - `causal_performance_claim_allowed = false`

#### 4. Step 100 — causal simulator v2 contract 갱신 완료
- 생성/검증 파일:
  - `05_training/adapters/build_causal_simulator_v2_contract_step100.py`
  - `05_training/adapters/test_causal_simulator_v2_contract_step100.py`
  - `05_training/adapters/causal_simulator_v2_contract_step100.md`
- 주요 결과:
  - `contract_status = READY_FOR_ROUTE_AWARE_MINIMAL_SCAFFOLD`
- 의미:
  - Step 99-F의 observed/candidate/proxy/not_observed 판정을 simulator v2 계약으로 승격.
  - `eta_based_headway`는 calibration candidate로만 사용.
  - `actual_headway`, `actual_arrival_departure_time`, `actual_dwell`은 계속 not_observed.

#### 5. Step 101 — route-aware minimal simulator scaffold 완료
- 생성/검증 파일:
  - `05_training/adapters/route_aware_minimal_simulator_step101.py`
  - `05_training/adapters/test_route_aware_minimal_simulator_step101.py`
  - `05_training/adapters/route_aware_minimal_simulator_step101.md`
- 주요 결과:
  - `audit_status = PASS`
  - `scaffold_status = READY_FOR_STEP102_ROLLOUT_WRITER_SCAFFOLD`
  - `trace_rows = 24`
- 의미:
  - `/getBs02`의 `route_id + direction_id + ordered_stop_sequence`를 기반으로 route table 생성.
  - bus-like agent를 route sequence 위에 초기화하고, `hold`/`advance` action에 따라 stop_order를 갱신하는 최소 scaffold 통과.

#### 6. Step 102 — route-aware rollout writer scaffold 완료
- 생성/검증 파일:
  - `05_training/adapters/route_aware_rollout_writer_step102.py`
  - `05_training/adapters/test_route_aware_rollout_writer_step102.py`
  - `05_training/adapters/route_aware_rollout_writer_step102.md`
- 주요 결과:
  - 기본 실행: `raw_events = 24`, `window_rollup = 1`, `parquet_ready = True`
  - A/A90/A80/A70 × seeds 1,2,3 × routes 2 matrix:
    - `raw_events = 480`
    - `window_rollup = 24`
    - `parquet_ready = True`
- 의미:
  - Step 101 trace를 `raw_events.parquet`와 `window_rollup.parquet` 형태로 변환.
  - canonical KPI(Key Performance Indicator=핵심 성과 지표) 집계기의 입력 형태로 보낼 준비 완료.

#### 7. Step 103 — route-aware rollout canonical KPI integration 완료
- 생성/검증 파일:
  - `05_training/adapters/route_aware_canonical_kpi_step103.py`
  - `05_training/adapters/test_route_aware_canonical_kpi_step103.py`
  - `05_training/adapters/route_aware_canonical_kpi_step103.md`
- 주요 결과:
  - `audit_status = PASS`
  - `integration_status = READY_FOR_STEP104_PROJECT_LOG_RUNBOOK_UPDATE`
  - `kpi_by_window = 24`
  - `kpi_by_seed = 12`
  - `kpi_by_time_band = 12`
  - `causal_allowed = False`
- 의미:
  - Step 102의 `window_rollup.parquet`가 기존 `canonical_kpi_aggregator.py`의 `official_rollup` 경로를 통과.
  - route-aware scaffold output이 canonical KPI 파이프라인에 연결 가능함을 확인.
  - 단, `causal_allowed = False`로 성능 주장 차단선은 유지.

#### 8. 현재 허용되는 주장
- `/getBs02`, `/getPos02`, `/getRealtime02` 산출물은 route_id / direction_id / stop_id 체계에서 상호 연결 가능하다.
- route-aware minimal simulator scaffold는 `/getBs02` ordered stop sequence 위에서 agent 이동 trace를 생성할 수 있다.
- route-aware rollout writer는 `raw_events`와 `window_rollup`을 생성할 수 있다.
- route-aware scaffold output은 canonical KPI aggregation까지 통과할 수 있다.

#### 9. 아직 금지되는 주장
- actual headway가 관측되었다.
- 실제 arrival/departure time이 관측되었다.
- 실제 dwell time이 관측되었다.
- A/A90/A80/A70의 실제 성능 차이를 주장할 수 있다.
- 논문 성능표에 넣을 수 있는 causal performance result가 나왔다.

#### 10. 다음 단계
- Step 105: route-aware v2 realism gap closure plan 작성.
- Step 106: passenger queue / demand proxy v2 설계.
- Step 107: ETA 기반 calibration hook 설계.
- Step 108: route-aware simulator adapter interface 정식화.
- Step 109: canonical 12-KPI 확장과 route-aware v2 연결 여부 재점검.
'''

PROJECT_LOG_SECTION = STEP104_MARKER_START + "\n" + PROJECT_LOG_BODY.strip() + "\n" + STEP104_MARKER_END + "\n"

RUNBOOK_CONTENT = r'''# Step 104 — Route-aware causal simulator v2 runbook

## Purpose

This runbook freezes the current Step 99-D through Step 103 route-aware v2 pipeline status.
It is a continuation guide, not a performance report.

The current pipeline proves that BIS API artifacts can be connected into a route-aware simulator scaffold and then passed through canonical KPI aggregation.
It does **not** prove real-world dispatch performance.

## Completed gates

| Step | Status | Output meaning |
|---|---:|---|
| Step 99-D | PASS | getRealtime02 ETA rows and ETA-based headway candidates collected as candidates |
| Step 99-E | PASS | getBs02, getPos02, getRealtime02 artifacts cross-match on route/direction/stop keys |
| Step 99-F | PASS | Step 97/98 classification updates consolidated |
| Step 100 | PASS | causal simulator v2 contract updated from classification patch |
| Step 101 | PASS | route-aware minimal simulator scaffold created |
| Step 102 | PASS | raw_events/window_rollup writer scaffold created |
| Step 103 | PASS | route-aware window_rollup passed through canonical KPI official_rollup |

## Canonical guardrails

The following fields must remain blocked until real observed event data is available:

```text
actual_headway = not_observed
actual_arrival_departure_time = not_observed
actual_dwell = not_observed
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```

`eta_based_headway` is a candidate/calibration signal only. It must not be upgraded to actual headway.

## Re-run commands

Run from project root:

```powershell
Set-Location "C:\Users\ryujo\urbanbus_rl_project"
```

### Step 99-E integration audit

```powershell
python -m py_compile `
  05_training/adapters/analyze_bis_api_integration_step99e.py `
  05_training/adapters/test_bis_api_integration_step99e.py

python 05_training/adapters/test_bis_api_integration_step99e.py
python 05_training/adapters/analyze_bis_api_integration_step99e.py
```

Expected:

```text
getPos02 coverage = 1.0
getRealtime02 ETA coverage = 1.0
headway candidate coverage = 1.0
```

### Step 99-F classification consolidation

```powershell
python -m py_compile `
  05_training/adapters/consolidate_bis_api_classification_step99f.py `
  05_training/adapters/test_bis_api_classification_step99f.py

python 05_training/adapters/test_bis_api_classification_step99f.py
python 05_training/adapters/consolidate_bis_api_classification_step99f.py
```

Expected:

```text
audit_status = PASS
actual_headway = not_observed
causal_performance_claim_allowed = false
```

### Step 100 contract update

```powershell
python -m py_compile `
  05_training/adapters/build_causal_simulator_v2_contract_step100.py `
  05_training/adapters/test_causal_simulator_v2_contract_step100.py

python 05_training/adapters/test_causal_simulator_v2_contract_step100.py
python 05_training/adapters/build_causal_simulator_v2_contract_step100.py
```

Expected:

```text
contract_status = READY_FOR_ROUTE_AWARE_MINIMAL_SCAFFOLD
```

### Step 101 route-aware minimal simulator scaffold

```powershell
python -m py_compile `
  05_training/adapters/route_aware_minimal_simulator_step101.py `
  05_training/adapters/test_route_aware_minimal_simulator_step101.py

python 05_training/adapters/test_route_aware_minimal_simulator_step101.py
python 05_training/adapters/route_aware_minimal_simulator_step101.py
```

Expected:

```text
scaffold_status = READY_FOR_STEP102_ROLLOUT_WRITER_SCAFFOLD
trace_rows = 24
```

### Step 102 route-aware rollout writer scaffold

```powershell
python -m py_compile `
  05_training/adapters/route_aware_rollout_writer_step102.py `
  05_training/adapters/test_route_aware_rollout_writer_step102.py

python 05_training/adapters/test_route_aware_rollout_writer_step102.py
python 05_training/adapters/route_aware_rollout_writer_step102.py `
  --conditions A,A90,A80,A70 `
  --seeds 1,2,3 `
  --max-routes 2 `
  --baseline-bus-count 3 `
  --max-steps 8
```

Expected:

```text
raw_events = 480
window_rollup = 24
parquet_ready = True
```

### Step 103 canonical KPI integration

```powershell
python -m py_compile `
  05_training/adapters/route_aware_canonical_kpi_step103.py `
  05_training/adapters/test_route_aware_canonical_kpi_step103.py

python 05_training/adapters/test_route_aware_canonical_kpi_step103.py
python 05_training/adapters/route_aware_canonical_kpi_step103.py
```

Expected:

```text
kpi_by_window = 24
kpi_by_seed = 12
kpi_by_time_band = 12
causal_allowed = False
```

## Commit sequence

Recommended commits:

```powershell
git add `
  .\05_training\adapters\route_aware_canonical_kpi_step103.py `
  .\05_training\adapters\test_route_aware_canonical_kpi_step103.py `
  .\05_training\adapters\route_aware_canonical_kpi_step103.md

git commit -m "Connect route-aware rollout to canonical KPI aggregation"
```

Then Step 104:

```powershell
git add `
  .\05_training\adapters\update_project_log_step104.py `
  .\05_training\adapters\test_project_log_update_step104.py `
  .\05_training\adapters\route_aware_v2_pipeline_runbook_step104.md `
  .\05_training\adapters\project_log_runbook_update_step104.md `
  .\project_log.md

git commit -m "Document route-aware causal simulator v2 pipeline progress"
```

Artifacts under `artifacts/` should normally stay uncommitted because they are reproducible local outputs.

## Next recommended steps

1. Step 105 — route-aware v2 realism gap closure plan.
2. Step 106 — passenger queue and demand proxy v2 design.
3. Step 107 — getRealtime02 ETA calibration hook design.
4. Step 108 — formal simulator adapter interface for route-aware v2.
5. Step 109 — 12-KPI contract alignment check for route-aware v2.
'''

README_CONTENT = r'''# Step 104 — Project log and route-aware v2 runbook update

This step records the completed Step 99-D through Step 103 route-aware causal simulator v2 scaffold progress.

It updates `project_log.md` idempotently and writes a continuation runbook.

## Files

- `update_project_log_step104.py`
- `test_project_log_update_step104.py`
- `route_aware_v2_pipeline_runbook_step104.md`
- `project_log_runbook_update_step104.md`

## Usage

```powershell
python -m py_compile `
  05_training/adapters/update_project_log_step104.py `
  05_training/adapters/test_project_log_update_step104.py

python 05_training/adapters/test_project_log_update_step104.py
python 05_training/adapters/update_project_log_step104.py
```

## Non-claim guard

The update explicitly preserves:

```text
actual_headway = not_observed
actual_arrival_departure_time = not_observed
actual_dwell = not_observed
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
```
'''


def replace_or_insert_section(existing: str, section: str) -> str:
    if STEP104_MARKER_START in existing and STEP104_MARKER_END in existing:
        start = existing.index(STEP104_MARKER_START)
        end = existing.index(STEP104_MARKER_END) + len(STEP104_MARKER_END)
        return existing[:start].rstrip() + "\n\n" + section.strip() + "\n\n" + existing[end:].lstrip()

    lines = existing.splitlines()
    insert_idx = 0
    for i, line in enumerate(lines[:30]):
        if line.strip() == "---":
            insert_idx = i + 1
            break
    else:
        insert_idx = min(len(lines), 1)
    new_lines = lines[:insert_idx] + ["", section.strip(), ""] + lines[insert_idx:]
    return "\n".join(new_lines).rstrip() + "\n"


def update_project_log(project_log_path: Path) -> Dict[str, Any]:
    project_log_path.parent.mkdir(parents=True, exist_ok=True)
    if project_log_path.exists():
        existing = project_log_path.read_text(encoding="utf-8-sig")
    else:
        existing = "# 🚌 UrbanBus RL Project - 작업 일지 (Project Journal)\n\n---\n"
    updated = replace_or_insert_section(existing, PROJECT_LOG_SECTION)
    project_log_path.write_text(updated, encoding="utf-8")
    return {
        "project_log_path": str(project_log_path),
        "marker_start_present": STEP104_MARKER_START in updated,
        "marker_end_present": STEP104_MARKER_END in updated,
        "project_log_bytes": project_log_path.stat().st_size,
    }


def write_text(path: Path, content: str) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size}


def write_manifest(output_root: Path, payload: Dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "project_log_runbook_step104_manifest.json"
    payload = dict(payload)
    payload["artifact_version"] = "project_log_runbook_step104_v1"
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 104 project log/runbook updater")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--project-log", default="project_log.md")
    parser.add_argument("--runbook", default=str(DEFAULT_RUNBOOK_REL))
    parser.add_argument("--readme", default=str(DEFAULT_README_REL))
    parser.add_argument("--output-root", default=str(DEFAULT_REPORT_ROOT))
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    project_log_path = root / args.project_log
    runbook_path = root / args.runbook
    readme_path = root / args.readme
    output_root = root / args.output_root

    log_info = update_project_log(project_log_path)
    runbook_info = write_text(runbook_path, RUNBOOK_CONTENT)
    readme_info = write_text(readme_path, README_CONTENT)

    manifest_payload = {
        "audit_status": "PASS",
        "step": "Step 104",
        "purpose": "Document Step 99-D through Step 103 route-aware causal simulator v2 scaffold progress",
        "project_log": log_info,
        "runbook": runbook_info,
        "readme": readme_info,
        "guardrails": {
            "actual_headway": "not_observed",
            "actual_arrival_departure_time": "not_observed",
            "actual_dwell": "not_observed",
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
        "next_steps": [
            "Step 105 route-aware v2 realism gap closure plan",
            "Step 106 passenger queue and demand proxy v2 design",
            "Step 107 getRealtime02 ETA calibration hook design",
            "Step 108 formal route-aware simulator adapter interface",
            "Step 109 12-KPI alignment check",
        ],
    }
    manifest_path = write_manifest(output_root, manifest_payload)

    print("[OK] Step 104 project log/runbook update completed")
    print("[OK] audit_status : PASS")
    print(f"[OK] project_log  : {project_log_path}")
    print(f"[OK] runbook      : {runbook_path}")
    print(f"[OK] readme       : {readme_path}")
    print(f"[OK] manifest     : {manifest_path}")


if __name__ == "__main__":
    main()
