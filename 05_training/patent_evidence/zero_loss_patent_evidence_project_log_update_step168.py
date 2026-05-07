from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


BEGIN_MARKER = "<!-- STEP168_ZERO_LOSS_PATENT_EVIDENCE_LOG_BEGIN -->"
END_MARKER = "<!-- STEP168_ZERO_LOSS_PATENT_EVIDENCE_LOG_END -->"
ARTIFACT_VERSION = "zero_loss_patent_evidence_project_log_update_step168_v1"


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_text_any(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read text file: {path}")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_section() -> str:
    return f"""{BEGIN_MARKER}
## 📅 2026-05-07
### Step 160~167 — Zero-Loss Pickup Threshold 특허 증거 파이프라인 구축 기록

이번 작업에서는 `Zero-Loss Pickup Threshold` 특허 출원 보조 자료를 만들기 위한 evidence pipeline(증거 파이프라인)을 구축했다. 목표는 실제 성능 주장이 아니라, 향후 실제 route-aware rollout 산출물이 준비되었을 때 다음 질문에 재현 가능하게 답하는 것이다.

```text
1. 몇 번의 합승 시도가 있었는가?
2. 그중 기존 탑승 승객의 ETA(Estimated Time of Arrival=예상 도착 시간) 손실이 0인 경우는 몇 번인가?
3. Zero-Loss 성공/실패 시점의 GATv2(Graph Attention Network v2=그래프 어텐션 네트워크 v2) attention weight(어텐션 가중치)는 어떤 route/path 주변에 분포했는가?
4. 해당 증거 묶음이 어떤 checkpoint/config/data/run manifest와 연결되는가?
```

#### 1. Step 160 — Zero-Loss Pickup Evidence Reporter

- 파일 계열: `05_training/patent_evidence/zero_loss_pickup_evidence_reporter_step160.py`
- 역할: `pickup_attempt_events`, `eta_counterfactual`, `gatv2_attention`, `run_manifest`를 입력으로 받아 Zero-Loss 성공률과 attention evidence bundle(증거 묶음)을 생성한다.
- 핵심 계산식:

```text
delta_eta_existing_passenger_sec =
  eta_with_new_pickup_sec - eta_without_new_pickup_sec

zero_loss_success =
  delta_eta_existing_passenger_sec <= zero_loss_epsilon_sec
```

- 산출물: JSON/CSV/PNG/Markdown 기반 특허 증거 리포트.
- guard: `paper_level_claim_allowed=false`, `causal_performance_claim_allowed=false` 유지.

#### 2. Step 161 — Route-Aware Pickup Attempt Event Writer

- 파일 계열: `05_training/patent_evidence/route_aware_pickup_attempt_event_writer_step161.py`
- 역할: route-aware simulator 산출물 또는 sample rollout에서 `pickup_attempt_events`, `eta_counterfactual`, `gatv2_attention`, `run_manifest` 입력 묶음을 생성한다.
- Step 160과 직접 연결되는 integration smoke(통합 구조 테스트)를 통과했다.
- 이 단계의 attention은 최초에는 proxy attention(대리 어텐션) 구조였으며, 실제 GATv2 attention은 Step 163 이후 별도 경로로 연결했다.

#### 3. Step 162 — Route-Aware Rollout Adapter

- 파일 계열: `05_training/patent_evidence/route_aware_rollout_adapter_step162.py`
- 역할: 기존 `raw_events.parquet` 또는 route-aware rollout output을 Step 161/160 입력 형식으로 정규화한다.
- 의미: 샘플 데이터 생성기 수준에서 한 단계 올라가, 실제 rollout artifact(롤아웃 산출물)를 특허 증거 파이프라인에 연결할 수 있는 adapter(어댑터)를 마련했다.
- 주의: self-test 값은 실제 10분 snapshot(스냅샷) 전체 결과가 아니라 구조 검증용 sample 결과이다.

#### 4. Step 163 — GATv2 Real Attention Extractor Contract

- 파일 계열: `05_training/patent_evidence/gatv2_real_attention_extractor_step163.py`
- 역할: PyG(PyTorch Geometric=파이토치 지오메트릭) `GATv2Conv`의 `return_attention_weights=True` 경로에서 실제 attention weight를 추출할 수 있는지 검증한다.
- 핵심 상태: `real_gatv2conv_attention_extracted=true`.
- 의미: 특허 증거 파이프라인이 proxy attention이 아니라 실제 GATv2 attention weight를 담을 수 있음을 확인했다.

#### 5. Step 164 — Real Attention Pipeline Connector

- 파일 계열: `05_training/patent_evidence/real_attention_pipeline_connector_step164.py`
- 역할: Step 163의 real GATv2 attention output을 Step 160 evidence pipeline 입력인 `gatv2_attention.csv`로 연결한다.
- connection mode: `snapshot_topk_projection_step164`.
- 한계: snapshot 전체 top-k attention을 attempt에 연결하는 방식이므로, attempt별 관련 edge/path 필터링은 Step 165에서 분리 수행했다.

#### 6. Step 165 — Attempt-Specific Route/Path Attention Filter

- 파일 계열: `05_training/patent_evidence/attempt_route_path_attention_filter_step165.py`
- 역할: 각 pickup attempt(합승 시도)의 `current_stop_id → pickup_stop_id → dropoff_stop_id` 경로와 관련 있는 real GATv2 attention row를 우선 선별한다.
- 필터 우선순위:

```text
1. exact_edge_overlap: route/path edge와 attention edge가 직접 일치
2. node_overlap: route/path node와 attention edge의 src/dst node가 겹침
3. fallback: 관련 edge/node를 찾지 못할 때만 제한적으로 전체 top-k 사용
```

- 의미: 특허 리포트에서 “해당 합승 시도와 관련 있는 route/path 주변의 attention evidence”를 제시할 수 있게 되었다.
- guard: self-test 결과가 있어도 실제 운영 성능 claim은 금지.

#### 7. Step 166 — Patent Evidence Report v2

- 파일 계열: `05_training/patent_evidence/patent_evidence_report_v2_step166.py`
- 역할: Step 160의 Zero-Loss 성공률/ETA delta와 Step 165의 path-segment attention evidence를 합쳐 특허 명세서에 넣기 쉬운 v2 리포트를 생성한다.
- 산출물 후보:

```text
patent_evidence_report_v2.md
patent_evidence_summary_v2.json
zero_loss_attempt_attention_summary_v2.csv
attention_mass_by_attempt_path_v2.csv
attention_mass_by_success_path_v2.csv
top_filtered_attention_edges_v2.csv
```

- 의미: “ETA 손실 0 성공 횟수”와 “그때의 real GATv2 route/path attention distribution(경로별 어텐션 분포)”를 한 리포트에 통합하는 구조가 준비되었다.

#### 8. Step 167 — Actual Route-Aware Rollout Evidence Runbook

- 파일 계열: `05_training/patent_evidence/actual_route_aware_rollout_evidence_runbook_step167.py`
- 역할: 실제 `raw_events.parquet`, `window_rollup.parquet`, route_stop_sequence, GATv2 checkpoint가 준비되었을 때 Step 162→161→163→165→160→166 순서로 실행하는 절차를 고정한다.
- 상태: template-only runbook이며, 실제 route-aware rollout evidence 실행은 아직 열리지 않았다.

#### 9. 현재 허용되는 주장

현재 허용되는 표현은 다음으로 제한한다.

```text
Step 160~167은 Zero-Loss Pickup Threshold 특허 증거 리포트 생성을 위한
파일 계약, 변환기, real GATv2 attention 추출 경로, attempt-specific path filter,
리포트 v2, 실제 실행 runbook을 구조적으로 연결했다.
```

#### 10. 현재 금지되는 주장

아래 표현은 아직 금지한다.

```text
실제 대구 버스 운영에서 Zero-Loss 합승 성공률이 검증되었다.
실제 H200 MAPPO(Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화) 학습 결과가 Zero-Loss 성능을 입증했다.
Step 160~167 self-test 수치를 논문 성능표 또는 특허 효과 수치로 사용한다.
현재 리포트가 actual causal performance(실제 인과 성능)를 증명한다.
```

#### 11. Guard state

```text
paper_level_claim_allowed = false
causal_performance_claim_allowed = false
actual_operational_claim_allowed = false
actual_route_aware_rollout_evidence_ready = false
train_allowed = false
```

#### 12. 다음 단계

- Step 168: project log update 완료.
- 다음 추천: Step 169 — actual evidence input readiness checklist.
  - 실제 `raw_events.parquet` / `window_rollup.parquet` / `route_stop_sequence` / GATv2 checkpoint / run manifest가 모두 준비되었는지 확인하는 실행 전 readiness gate를 만든다.

{END_MARKER}
"""


def insert_or_replace_section(original: str, section: str) -> str:
    if BEGIN_MARKER in original and END_MARKER in original:
        start = original.index(BEGIN_MARKER)
        end = original.index(END_MARKER) + len(END_MARKER)
        return original[:start].rstrip() + "\n\n" + section.strip() + "\n\n" + original[end:].lstrip()

    lines = original.splitlines()
    insert_at = 0
    # Put the newest journal entry after the first horizontal rule if present.
    for i, line in enumerate(lines[:20]):
        if line.strip() == "---":
            insert_at = i + 1
            break
    if insert_at == 0:
        return section.strip() + "\n\n" + original.lstrip()
    return "\n".join(lines[:insert_at]).rstrip() + "\n" + section.strip() + "\n\n" + "\n".join(lines[insert_at:]).lstrip() + "\n"


def validate_section_text(text: str) -> List[str]:
    required_tokens = [
        "Step 160",
        "Step 161",
        "Step 162",
        "Step 163",
        "Step 164",
        "Step 165",
        "Step 166",
        "Step 167",
        "paper_level_claim_allowed = false",
        "causal_performance_claim_allowed = false",
        "actual_operational_claim_allowed = false",
        "actual_route_aware_rollout_evidence_ready = false",
        "Zero-Loss Pickup Threshold",
        "GATv2(Graph Attention Network v2=그래프 어텐션 네트워크 v2)",
        "MAPPO(Multi-Agent Proximal Policy Optimization=다중 에이전트 근접 정책 최적화)",
    ]
    missing = [t for t in required_tokens if t not in text]
    return missing


def run_update(project_root: Path, output_root: Path, dry_run: bool = False) -> Dict[str, Any]:
    project_root = project_root.resolve()
    project_log = project_root / "project_log.md"
    if not project_log.exists():
        raise RuntimeError(f"project_log.md not found: {project_log}")

    output_root = (project_root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    original = read_text_any(project_log)
    section = build_section()
    missing_section = validate_section_text(section)
    if missing_section:
        raise RuntimeError(f"generated section missing required tokens: {missing_section}")

    updated = insert_or_replace_section(original, section)
    missing_updated = validate_section_text(updated)
    if missing_updated:
        raise RuntimeError(f"updated project log missing required tokens: {missing_updated}")

    section_path = output_root / "project_log_update_step168_section.md"
    write_text(section_path, section)

    if not dry_run:
        write_text(project_log, updated)

    manifest_path = output_root / "project_log_update_step168_manifest.json"
    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_status": "PASS",
        "log_update_status": "DRY_RUN_READY" if dry_run else "PROJECT_LOG_UPDATED",
        "project_root": str(project_root),
        "project_log": str(project_log),
        "output_root": str(output_root),
        "section_path": str(section_path),
        "step_range": "160-167",
        "documented_steps": [160, 161, 162, 163, 164, 165, 166, 167],
        "pipeline_name": "zero_loss_pickup_threshold_patent_evidence_pipeline",
        "summary": "Step 160~167 Zero-Loss patent evidence pipeline was documented in project_log.md with non-claim guards locked.",
        "template_only": False,
        "actual_route_aware_rollout_evidence_ready": False,
        "actual_operational_claim_allowed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "train_allowed": False,
        "performance_claim_allowed": False,
        "dry_run": bool(dry_run),
        "output_files": {
            "project_log_update_section": str(section_path),
            "manifest": str(manifest_path),
        },
    }
    dump_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 168 project log update for zero-loss patent evidence pipeline")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-root", default="artifacts/patent_evidence/zero_loss_patent_evidence_project_log_update_step168")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = run_update(Path(args.project_root), Path(args.output_root), dry_run=bool(args.dry_run))
    print("[OK] Step 168 zero-loss patent evidence project log update completed")
    print(f"[OK] audit_status              : {manifest['audit_status']}")
    print(f"[OK] log_update_status         : {manifest['log_update_status']}")
    print(f"[OK] project_log               : {manifest['project_log']}")
    print(f"[OK] manifest                  : {manifest['output_files']['manifest']}")
    print(f"[OK] paper_level_claim_allowed : {manifest['paper_level_claim_allowed']}")
    print(f"[OK] causal_performance_claim_allowed : {manifest['causal_performance_claim_allowed']}")


if __name__ == "__main__":
    main()
