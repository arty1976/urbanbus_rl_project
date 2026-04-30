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
