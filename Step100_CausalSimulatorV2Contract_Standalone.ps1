param(
    [string]$ProjectRoot = "C:\Users\ryujo\urbanbus_rl_project",
    [switch]$RunSelfTest,
    [switch]$RunActualIntegration
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ProjectRoot)) {
    throw "[STOP] Project root not found: $ProjectRoot"
}
Set-Location $ProjectRoot

$AdaptersDir = ".\05_training\adapters"
New-Item -ItemType Directory -Force -Path $AdaptersDir | Out-Null

# ---------------------------------------------------------------------
# 1) causal_simulator_v2_input_contract.md
# ---------------------------------------------------------------------
$ContractMd = @'
# Step 100 — Causal Simulator v2 Input Contract

## 0. 목적

Step 100의 목적은 Step 97~99에서 확인한 tensor DB(Database=데이터베이스) + 대구 신호등 CSV(Comma-Separated Values=쉼표 구분값) 기반 feature를 `causal simulator v2`의 입력 계약으로 고정하는 것이다.

이 계약은 성능 주장 문서가 아니다. 현재 결과는 여전히 toy/smoke 및 정적 feature preparation 단계이다.

## 1. 입력 feature family

### 1.1 Tensor DB direct features

허용:

- graph skeleton
- node identity
- edge identity
- edge distance/time/cost
- snapshot time context
- observed demand context
- scenario/window metadata

### 1.2 Static signal infrastructure features

허용:

- `signal_count_100m`
- `signal_count_250m`
- `signal_count_500m`
- `nearest_signal_distance_m`
- `pedestrian_signal_count_250m`
- `blink_signal_ratio_250m`
- `controlled_signal_ratio_250m`
- `edge_signal_count`
- `edge_signal_density_per_km`
- `edge_nearest_signal_distance_m`

### 1.3 Explicit proxy features

허용하되 반드시 proxy로 표기:

- `signal_delay_risk_proxy`
- `intersection_complexity_proxy`
- `edge_control_complexity_proxy`

이 값들은 실제 신호 지연 측정값이 아니다.

## 2. 금지 feature

현재 데이터만으로는 아래 feature를 causal simulator v2에 넣으면 안 된다.

- `red_light_delay_seconds`
- `green_time_seconds`
- `cycle_length_seconds`
- `phase_sequence`
- `signal_offset_seconds`
- `real_time_signal_state`
- `queue_discharge_rate`
- `lane_level_turning_movement`

## 3. Causal simulator v2 admission rule

| feature class | admission |
|---|---|
| tensor_db_direct | allowed |
| signal_csv_direct_static | allowed |
| tensor_signal_derived_static | allowed |
| explicit_proxy | allowed with `_proxy` suffix |
| dynamic_signal_phase | rejected |
| red_light_delay / green_time / cycle length | rejected |

## 4. Claim guardrail

아래 값은 false로 유지한다.

```json
{
  "trained_model": false,
  "performance_claim_allowed": false,
  "causal_performance_claim_allowed": false,
  "dynamic_signal_phase_claim_allowed": false,
  "red_light_delay_claim_allowed": false,
  "green_time_claim_allowed": false,
  "cycle_length_claim_allowed": false,
  "daegu_citywide_performance_claim_allowed": false,
  "fleet_reduction_claim_allowed": false
}
```

## 5. Step 100 산출물

```text
artifacts/causal_simulator_v2_contract/causal_simulator_v2_input_contract.json
artifacts/causal_simulator_v2_contract/causal_simulator_v2_feature_manifest.json
artifacts/causal_simulator_v2_contract/causal_simulator_v2_contract_report.md
```

## 6. 다음 단계

Step 101에서는 이 계약을 기반으로 `CausalSimulatorAdapter v2` scaffold를 작성한다.
'@

$ContractMd | Set-Content -Path (Join-Path $AdaptersDir "causal_simulator_v2_input_contract.md") -Encoding UTF8

# ---------------------------------------------------------------------
# 2) tensor_signal_availability_integration_v2.md
# ---------------------------------------------------------------------
$IntegrationMd = @'
# Step 100 — Tensor DB + Signal Feature Availability Integration

## 0. 목적

Step 97은 데이터 사용 가능성을 분류했고, Step 98은 정적 신호 feature builder를 만들었으며, Step 99는 실제 대구 신호등 CSV와 tensor graph를 결합하여 feature 산출물을 생성했다.

Step 100은 이 세 단계를 통합하여 다음을 고정한다.

1. 어떤 feature가 실제 산출물에 존재하는가
2. 어떤 feature가 causal simulator v2에 투입 가능한가
3. 어떤 feature는 proxy로만 허용되는가
4. 어떤 feature는 현재 금지되어야 하는가
5. 다음 CausalSimulatorAdapter v2가 읽어야 할 계약은 무엇인가

## 1. 현재 완료 상태

| Step | 상태 |
|---|---|
| Step 97 | Tensor DB + signal CSV data availability audit 완료 |
| Step 98 | Signal feature builder self-test 완료 |
| Step 99 | 실제 대구 신호등 CSV 기반 node/edge signal feature 생성 완료 |

Step 99 기준 산출물:

```text
artifacts/signal_features_v2/node_signal_features.parquet
artifacts/signal_features_v2/edge_signal_features.parquet
artifacts/signal_features_v2/tensor_signal_feature_contract_v2.json
artifacts/signal_features_v2/signal_feature_quality_report.json
```

## 2. 통합 원칙

허용:

- 정적 위치 기반 신호 수
- 주변 신호 밀도
- 가장 가까운 신호까지 거리
- 보행 신호 수
- 점멸 신호 비율
- 제어 신호 비율
- `_proxy` suffix가 붙은 복잡도/위험도 proxy

금지:

- red-light delay
- green time
- cycle length
- phase sequence
- signal offset
- real-time signal state
- queue discharge rate

## 3. Causal simulator v2에서의 사용 위치

| 위치 | 허용 여부 | 설명 |
|---|---|---|
| actor observation context | 가능 | 정책이 정적 인프라 난이도를 인식 |
| critic/global context | 가능 | 환경 복잡도 반영 |
| edge_attr extension | 가능 | graph edge signal context |
| reward 직접항 | 원칙적 보류 | proxy 오용 위험 |
| 동적 phase simulator | 금지 | phase/timing 자료 없음 |

## 4. 다음 작업

Step 101:

```text
CausalSimulatorAdapter v2 scaffold
```

목표:

- Step 100 계약을 읽는다.
- node/edge signal feature artifact를 로드한다.
- observation에 정적 signal context를 붙인다.
- 동적 신호 지연을 생성하지 않는다.
- performance_claim_allowed=false 상태를 유지한다.
'@

$IntegrationMd | Set-Content -Path (Join-Path $AdaptersDir "tensor_signal_availability_integration_v2.md") -Encoding UTF8

# ---------------------------------------------------------------------
# 3) build_causal_simulator_v2_contract_from_signal_features.py
# ---------------------------------------------------------------------
$BuilderPy = @'
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


ALLOWED_NODE_SIGNAL_COLUMNS = [
    "node_uid",
    "node_index",
    "signal_count_100m",
    "signal_count_250m",
    "signal_count_500m",
    "nearest_signal_distance_m",
    "pedestrian_signal_count_250m",
    "blink_signal_ratio_250m",
    "controlled_signal_ratio_250m",
    "signal_delay_risk_proxy",
    "intersection_complexity_proxy",
    "signal_feature_quality_flag",
]

ALLOWED_EDGE_SIGNAL_COLUMNS = [
    "src_idx",
    "dst_idx",
    "distance_m",
    "edge_signal_count",
    "edge_signal_density_per_km",
    "edge_nearest_signal_distance_m",
    "edge_control_complexity_proxy",
    "edge_signal_feature_quality_flag",
]

FORBIDDEN_DYNAMIC_SIGNAL_FIELDS = [
    "red_light_delay_seconds",
    "green_time_seconds",
    "cycle_length_seconds",
    "phase_sequence",
    "signal_offset_seconds",
    "real_time_signal_state",
    "queue_discharge_rate",
    "lane_level_turning_movement",
]

PROXY_COLUMNS = [
    "signal_delay_risk_proxy",
    "intersection_complexity_proxy",
    "edge_control_complexity_proxy",
]


def load_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def assert_no_forbidden_columns(df: pd.DataFrame, name: str) -> None:
    bad = [c for c in FORBIDDEN_DYNAMIC_SIGNAL_FIELDS if c in df.columns]
    if bad:
        raise RuntimeError(f"{name} contains forbidden dynamic signal fields: {bad}")


def require_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required columns: {missing}")


def summarize_numeric(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for c in columns:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        out[c] = {
            "valid_count": int(s.notna().sum()),
            "null_count": int(s.isna().sum()),
            "min": float(s.min()) if s.notna().any() else None,
            "max": float(s.max()) if s.notna().any() else None,
            "mean": float(s.mean()) if s.notna().any() else None,
        }
    return out


def write_report_md(path: Path, contract: Dict[str, Any], manifest: Dict[str, Any]) -> None:
    lines = [
        "# Step 100 Causal Simulator v2 Contract Report",
        "",
        f"- artifact_version: `{contract['artifact_version']}`",
        f"- status: `{contract['status']}`",
        f"- node rows: `{contract['row_counts']['node_signal_features']}`",
        f"- edge rows: `{contract['row_counts']['edge_signal_features']}`",
        "",
        "## Admission Summary",
        "",
    ]

    for k, v in manifest["admission_summary"].items():
        lines.append(f"- `{k}`: `{v}`")

    lines.extend(["", "## Node Static Signal Features", ""])
    for c in manifest["node_feature_columns"]:
        lines.append(f"- `{c}`")

    lines.extend(["", "## Edge Static Signal Features", ""])
    for c in manifest["edge_feature_columns"]:
        lines.append(f"- `{c}`")

    lines.extend(["", "## Proxy Features", ""])
    for c in manifest["proxy_columns"]:
        lines.append(f"- `{c}`")

    lines.extend(["", "## Forbidden Dynamic Signal Fields", ""])
    for c in manifest["forbidden_dynamic_signal_fields"]:
        lines.append(f"- `{c}`")

    lines.extend(["", "## Claim Guardrails", ""])
    for k, v in contract["claim_guardrails"].items():
        lines.append(f"- `{k}`: `{v}`")

    lines.extend([
        "",
        "## Next Step",
        "",
        f"- Step {contract['next_step']['step']}: {contract['next_step']['title']}",
        f"- Goal: {contract['next_step']['goal']}",
    ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_contract(signal_features_dir: Path, preflight_dir: Path, output_dir: Path) -> Dict[str, Any]:
    node_path = signal_features_dir / "node_signal_features.parquet"
    edge_path = signal_features_dir / "edge_signal_features.parquet"
    signal_contract_path = signal_features_dir / "tensor_signal_feature_contract_v2.json"
    signal_quality_path = signal_features_dir / "signal_feature_quality_report.json"
    preflight_report_path = preflight_dir / "daegu_signal_csv_preflight_report.json"

    if not node_path.exists():
        raise RuntimeError(f"node signal features not found: {node_path}")
    if not edge_path.exists():
        raise RuntimeError(f"edge signal features not found: {edge_path}")

    node_df = pd.read_parquet(node_path)
    edge_df = pd.read_parquet(edge_path)

    require_columns(node_df, ALLOWED_NODE_SIGNAL_COLUMNS, "node_signal_features")
    require_columns(edge_df, ALLOWED_EDGE_SIGNAL_COLUMNS, "edge_signal_features")

    assert_no_forbidden_columns(node_df, "node_signal_features")
    assert_no_forbidden_columns(edge_df, "edge_signal_features")

    signal_contract = load_json_optional(signal_contract_path)
    signal_quality = load_json_optional(signal_quality_path)
    preflight_report = load_json_optional(preflight_report_path)

    node_quality_counts = (
        node_df["signal_feature_quality_flag"].astype(str).value_counts(dropna=False).to_dict()
        if "signal_feature_quality_flag" in node_df.columns
        else {}
    )
    edge_quality_counts = (
        edge_df["edge_signal_feature_quality_flag"].astype(str).value_counts(dropna=False).to_dict()
        if "edge_signal_feature_quality_flag" in edge_df.columns
        else {}
    )

    claim_guardrails = {
        "trained_model": False,
        "performance_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "dynamic_signal_phase_claim_allowed": False,
        "red_light_delay_claim_allowed": False,
        "green_time_claim_allowed": False,
        "cycle_length_claim_allowed": False,
        "daegu_citywide_performance_claim_allowed": False,
        "fleet_reduction_claim_allowed": False,
    }

    contract = {
        "artifact_version": "causal_simulator_v2_input_contract_draft_step100",
        "step": 100,
        "status": "draft_contract_from_static_signal_features",
        "source_artifacts": {
            "node_signal_features": str(node_path),
            "edge_signal_features": str(edge_path),
            "tensor_signal_feature_contract_v2": str(signal_contract_path),
            "signal_feature_quality_report": str(signal_quality_path),
            "daegu_signal_csv_preflight_report": str(preflight_report_path),
        },
        "row_counts": {
            "node_signal_features": int(len(node_df)),
            "edge_signal_features": int(len(edge_df)),
        },
        "feature_classes": {
            "tensor_db_direct": {
                "admission": "allowed",
                "usage": ["graph skeleton", "scenario index", "time context", "observed demand context"],
            },
            "static_signal_node_features": {
                "admission": "allowed",
                "columns": [c for c in ALLOWED_NODE_SIGNAL_COLUMNS if c in node_df.columns],
                "usage": ["actor_observation_context", "critic_global_context"],
            },
            "static_signal_edge_features": {
                "admission": "allowed",
                "columns": [c for c in ALLOWED_EDGE_SIGNAL_COLUMNS if c in edge_df.columns],
                "usage": ["edge_attr_extension", "critic_global_context"],
            },
            "explicit_proxy_features": {
                "admission": "allowed_with_proxy_suffix",
                "columns": [c for c in PROXY_COLUMNS if c in node_df.columns or c in edge_df.columns],
                "usage": ["environment_difficulty_proxy", "policy_context_only"],
                "warning": "Proxy columns are not measured delay, phase, green time, or real-time signal state.",
            },
            "dynamic_signal_phase_features": {
                "admission": "rejected",
                "columns": FORBIDDEN_DYNAMIC_SIGNAL_FIELDS,
            },
        },
        "causal_simulator_v2_usage_rules": {
            "allowed_usage": [
                "observation_context",
                "static_environment_context",
                "edge_attr_extension",
                "critic_context",
                "difficulty_proxy_with_explicit_suffix",
            ],
            "rejected_usage": [
                "measured_red_light_delay",
                "green_time_simulation",
                "cycle_length_simulation",
                "phase_sequence_simulation",
                "signal_offset_simulation",
                "real_time_signal_state_simulation",
            ],
            "reward_usage_policy": {
                "direct_reward_term_allowed": False,
                "reason": "Static signal proxies can bias reward interpretation. Use as context first; reward use requires a separate calibrated assumption document.",
            },
        },
        "quality_summary": {
            "node_quality_counts": {str(k): int(v) for k, v in node_quality_counts.items()},
            "edge_quality_counts": {str(k): int(v) for k, v in edge_quality_counts.items()},
            "node_numeric_summary": summarize_numeric(
                node_df,
                [
                    "signal_count_100m",
                    "signal_count_250m",
                    "signal_count_500m",
                    "nearest_signal_distance_m",
                    "pedestrian_signal_count_250m",
                    "blink_signal_ratio_250m",
                    "controlled_signal_ratio_250m",
                    "signal_delay_risk_proxy",
                    "intersection_complexity_proxy",
                ],
            ),
            "edge_numeric_summary": summarize_numeric(
                edge_df,
                [
                    "distance_m",
                    "edge_signal_count",
                    "edge_signal_density_per_km",
                    "edge_nearest_signal_distance_m",
                    "edge_control_complexity_proxy",
                ],
            ),
            "signal_quality_report_loaded": signal_quality is not None,
            "preflight_report_loaded": preflight_report is not None,
            "source_signal_contract_loaded": signal_contract is not None,
        },
        "claim_guardrails": claim_guardrails,
        "next_step": {
            "step": 101,
            "title": "CausalSimulatorAdapter v2 scaffold",
            "goal": "Read Step 100 contract and attach static signal features to observation context without dynamic signal phase claims.",
        },
    }

    feature_manifest = {
        "artifact_version": "causal_simulator_v2_feature_manifest_step100",
        "step": 100,
        "node_feature_columns": [c for c in ALLOWED_NODE_SIGNAL_COLUMNS if c in node_df.columns],
        "edge_feature_columns": [c for c in ALLOWED_EDGE_SIGNAL_COLUMNS if c in edge_df.columns],
        "proxy_columns": [c for c in PROXY_COLUMNS if c in node_df.columns or c in edge_df.columns],
        "forbidden_dynamic_signal_fields": FORBIDDEN_DYNAMIC_SIGNAL_FIELDS,
        "admission_summary": {
            "node_static_signal_features": "allowed",
            "edge_static_signal_features": "allowed",
            "proxy_features": "allowed_with_proxy_suffix",
            "dynamic_signal_phase_features": "rejected",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "causal_simulator_v2_input_contract.json", contract)
    write_json(output_dir / "causal_simulator_v2_feature_manifest.json", feature_manifest)
    write_report_md(output_dir / "causal_simulator_v2_contract_report.md", contract, feature_manifest)

    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-features-dir", default="artifacts/signal_features_v2")
    parser.add_argument("--preflight-dir", default="artifacts/signal_features_v2_preflight")
    parser.add_argument("--output-dir", default="artifacts/causal_simulator_v2_contract")
    args = parser.parse_args()

    contract = build_contract(
        signal_features_dir=Path(args.signal_features_dir),
        preflight_dir=Path(args.preflight_dir),
        output_dir=Path(args.output_dir),
    )

    print("[OK] Step 100 causal simulator v2 contract integration complete")
    print(f"[OK] output_dir: {args.output_dir}")
    print(f"[OK] node_rows : {contract['row_counts']['node_signal_features']}")
    print(f"[OK] edge_rows : {contract['row_counts']['edge_signal_features']}")
    print("[OK] performance_claim_allowed:", contract["claim_guardrails"]["performance_claim_allowed"])
    print("[OK] dynamic_signal_phase_claim_allowed:", contract["claim_guardrails"]["dynamic_signal_phase_claim_allowed"])


if __name__ == "__main__":
    main()
'@

$BuilderPy | Set-Content -Path (Join-Path $AdaptersDir "build_causal_simulator_v2_contract_from_signal_features.py") -Encoding UTF8

# ---------------------------------------------------------------------
# 4) test_causal_simulator_v2_contract_from_signal_features.py
# ---------------------------------------------------------------------
$TestPy = @'
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTERS_DIR = PROJECT_ROOT / "05_training" / "adapters"
BUILDER = ADAPTERS_DIR / "build_causal_simulator_v2_contract_from_signal_features.py"
CONTRACT_MD = ADAPTERS_DIR / "causal_simulator_v2_input_contract.md"
INTEGRATION_MD = ADAPTERS_DIR / "tensor_signal_availability_integration_v2.md"

FORBIDDEN = {
    "red_light_delay_seconds",
    "green_time_seconds",
    "cycle_length_seconds",
    "phase_sequence",
    "signal_offset_seconds",
    "real_time_signal_state",
    "queue_discharge_rate",
}


def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def make_fixture(root: Path) -> tuple[Path, Path]:
    signal_dir = root / "signal_features_v2"
    preflight_dir = root / "signal_features_v2_preflight"
    signal_dir.mkdir(parents=True, exist_ok=True)
    preflight_dir.mkdir(parents=True, exist_ok=True)

    node_df = pd.DataFrame([
        {
            "node_uid": "STOP:1",
            "node_index": 0,
            "signal_count_100m": 1,
            "signal_count_250m": 2,
            "signal_count_500m": 3,
            "nearest_signal_distance_m": 12.0,
            "pedestrian_signal_count_250m": 1,
            "blink_signal_ratio_250m": 0.5,
            "controlled_signal_ratio_250m": 1.0,
            "signal_delay_risk_proxy": 1.2,
            "intersection_complexity_proxy": 2.3,
            "signal_feature_quality_flag": "ok",
        },
        {
            "node_uid": "STOP:2",
            "node_index": 1,
            "signal_count_100m": 0,
            "signal_count_250m": 1,
            "signal_count_500m": 2,
            "nearest_signal_distance_m": 90.0,
            "pedestrian_signal_count_250m": 0,
            "blink_signal_ratio_250m": 0.0,
            "controlled_signal_ratio_250m": 1.0,
            "signal_delay_risk_proxy": 0.8,
            "intersection_complexity_proxy": 1.1,
            "signal_feature_quality_flag": "ok",
        },
    ])
    edge_df = pd.DataFrame([
        {
            "src_idx": 0,
            "dst_idx": 1,
            "distance_m": 120.0,
            "edge_signal_count": 3,
            "edge_signal_density_per_km": 25.0,
            "edge_nearest_signal_distance_m": 12.0,
            "edge_control_complexity_proxy": 1.7,
            "edge_signal_feature_quality_flag": "ok",
        }
    ])

    node_df.to_parquet(signal_dir / "node_signal_features.parquet", index=False)
    edge_df.to_parquet(signal_dir / "edge_signal_features.parquet", index=False)

    (signal_dir / "tensor_signal_feature_contract_v2.json").write_text(
        json.dumps({"artifact_version": "tensor_signal_feature_contract_v2"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (signal_dir / "signal_feature_quality_report.json").write_text(
        json.dumps({"artifact_version": "signal_feature_quality_report_v1", "step_98_ready": True}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (preflight_dir / "daegu_signal_csv_preflight_report.json").write_text(
        json.dumps({"artifact_version": "daegu_signal_csv_preflight_report_v2", "step_99_ready": True}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return signal_dir, preflight_dir


def test_files_exist() -> None:
    assert_true(BUILDER.exists(), f"missing: {BUILDER}")
    assert_true(CONTRACT_MD.exists(), f"missing: {CONTRACT_MD}")
    assert_true(INTEGRATION_MD.exists(), f"missing: {INTEGRATION_MD}")


def test_docs_have_guardrails() -> None:
    text = CONTRACT_MD.read_text(encoding="utf-8-sig") + "\n" + INTEGRATION_MD.read_text(encoding="utf-8-sig")
    for phrase in [
        "red_light_delay_seconds",
        "green_time_seconds",
        "cycle_length_seconds",
        "performance_claim_allowed",
        "signal_delay_risk_proxy",
        "CausalSimulatorAdapter v2",
    ]:
        assert_true(phrase in text, f"missing phrase: {phrase}")


def test_builder_smoke() -> None:
    fixture_root = PROJECT_ROOT / "artifacts" / "step100_selftest"
    signal_dir, preflight_dir = make_fixture(fixture_root)
    output_dir = fixture_root / "out"

    cmd = [
        sys.executable,
        str(BUILDER),
        "--signal-features-dir",
        str(signal_dir),
        "--preflight-dir",
        str(preflight_dir),
        "--output-dir",
        str(output_dir),
    ]

    completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise AssertionError("Step 100 builder smoke failed")

    contract_path = output_dir / "causal_simulator_v2_input_contract.json"
    manifest_path = output_dir / "causal_simulator_v2_feature_manifest.json"
    report_path = output_dir / "causal_simulator_v2_contract_report.md"

    for p in [contract_path, manifest_path, report_path]:
        assert_true(p.exists(), f"missing output: {p}")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert_true(contract["step"] == 100, "bad step")
    assert_true(contract["row_counts"]["node_signal_features"] == 2, "bad node row count")
    assert_true(contract["row_counts"]["edge_signal_features"] == 1, "bad edge row count")
    assert_true(contract["claim_guardrails"]["performance_claim_allowed"] is False, "performance claim guardrail broken")
    assert_true(contract["claim_guardrails"]["dynamic_signal_phase_claim_allowed"] is False, "dynamic phase guardrail broken")

    for forbidden in FORBIDDEN:
        assert_true(forbidden in manifest["forbidden_dynamic_signal_fields"], f"missing forbidden field: {forbidden}")

    assert_true("signal_delay_risk_proxy" in manifest["proxy_columns"], "proxy column missing")
    assert_true(manifest["admission_summary"]["dynamic_signal_phase_features"] == "rejected", "dynamic admission must be rejected")


def main() -> None:
    test_files_exist()
    test_docs_have_guardrails()
    test_builder_smoke()
    print("[OK] Step 100 causal simulator v2 contract self-test PASS")


if __name__ == "__main__":
    main()
'@

$TestPy | Set-Content -Path (Join-Path $AdaptersDir "test_causal_simulator_v2_contract_from_signal_features.py") -Encoding UTF8

Write-Host "[OK] Step 100 standalone files created:"
Write-Host " - .\05_training\adapters\causal_simulator_v2_input_contract.md"
Write-Host " - .\05_training\adapters\tensor_signal_availability_integration_v2.md"
Write-Host " - .\05_training\adapters\build_causal_simulator_v2_contract_from_signal_features.py"
Write-Host " - .\05_training\adapters\test_causal_simulator_v2_contract_from_signal_features.py"

if (Test-Path ".\05_training\.venv\Scripts\python.exe") {
    $py = (Resolve-Path ".\05_training\.venv\Scripts\python.exe").Path
} else {
    $py = "python"
}

Write-Host "[INFO] python = $py"

if ($RunSelfTest) {
    & $py -m py_compile `
      ".\05_training\adapters\build_causal_simulator_v2_contract_from_signal_features.py" `
      ".\05_training\adapters\test_causal_simulator_v2_contract_from_signal_features.py"

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 100 py_compile failed"
    }

    & $py ".\05_training\adapters\test_causal_simulator_v2_contract_from_signal_features.py"

    if ($LASTEXITCODE -ne 0) {
        throw "[FAIL] Step 100 self-test failed"
    }

    Write-Host "[DONE] Step 100 causal simulator v2 contract self-test complete."
}

if ($RunActualIntegration -or $RunSelfTest) {
    if (Test-Path ".\artifacts\signal_features_v2\node_signal_features.parquet") {
        & $py ".\05_training\adapters\build_causal_simulator_v2_contract_from_signal_features.py" `
          --signal-features-dir ".\artifacts\signal_features_v2" `
          --preflight-dir ".\artifacts\signal_features_v2_preflight" `
          --output-dir ".\artifacts\causal_simulator_v2_contract"

        if ($LASTEXITCODE -ne 0) {
            throw "[FAIL] Step 100 actual integration failed"
        }

        Write-Host "[DONE] Step 100 actual causal simulator v2 contract integration complete."
    } else {
        Write-Host "[WARN] actual signal feature artifacts not found; skipped actual integration."
        Write-Host "[WARN] expected: .\artifacts\signal_features_v2\node_signal_features.parquet"
    }
}
