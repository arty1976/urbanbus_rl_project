from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


KPI_12 = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
]

BASE_6 = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
]

EXTENDED_6 = [
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
]


def load_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_parquet_optional(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def kpi_presence_from_overall(overall: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overall:
        return {
            "overall_exists": False,
            "available_kpis": [],
            "missing_kpis": KPI_12,
        }

    kpis = overall.get("kpis", {})
    available = [k for k in KPI_12 if k in kpis]
    missing = [k for k in KPI_12 if k not in kpis]

    return {
        "overall_exists": True,
        "available_kpis": available,
        "missing_kpis": missing,
        "kpi_means": {k: kpis.get(k, {}).get("mean") for k in available},
    }


def column_presence(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if df is None:
        return {
            "exists": False,
            "columns": [],
            "available_kpis": [],
            "missing_kpis": KPI_12,
            "row_count": 0,
        }

    cols = list(df.columns)
    available = [k for k in KPI_12 if k in cols]
    missing = [k for k in KPI_12 if k not in cols]

    valid_counts = {}
    null_counts = {}
    for k in available:
        s = df[k]
        valid_counts[k] = int(s.notna().sum())
        null_counts[k] = int(s.isna().sum())

    return {
        "exists": True,
        "row_count": int(len(df)),
        "columns": cols,
        "available_kpis": available,
        "missing_kpis": missing,
        "valid_counts": valid_counts,
        "null_counts": null_counts,
    }


def classify_gap(kpi: str, window_cols: List[str], rollup_cols: List[str]) -> Dict[str, Any]:
    source_cols = set(window_cols) | set(rollup_cols)

    if kpi in source_cols:
        return {
            "classification": "present",
            "action": "preserve_and_report",
            "reason": "KPI column already exists in B1_noop artifacts.",
        }

    if kpi == "passenger_service_rate":
        if {"passenger_served_count", "passenger_demand_generated"}.issubset(source_cols):
            return {
                "classification": "derivable",
                "action": "derive_as_passenger_served_count_over_passenger_demand_generated",
                "reason": "Both numerator and denominator exist.",
            }
        return {
            "classification": "not_derivable_current_artifact",
            "action": "leave_null_with_reason",
            "reason": "Requires passenger_served_count and passenger_demand_generated.",
        }

    if kpi == "energy_proxy_per_passenger":
        if "passenger_served_count" in source_cols and ("energy_proxy" in source_cols or "energy_proxy_total" in source_cols):
            return {
                "classification": "derivable",
                "action": "derive_as_energy_proxy_over_passenger_served_count",
                "reason": "Energy and served passenger count exist.",
            }
        return {
            "classification": "not_derivable_current_artifact",
            "action": "leave_null_with_reason",
            "reason": "Requires energy_proxy or energy_proxy_total plus passenger_served_count.",
        }

    if kpi == "fleet_reduction_ratio":
        return {
            "classification": "contract_fixed_candidate",
            "action": "can_set_to_0_0_only_with_contract_fixed_flag",
            "reason": "B1_noop is full-fleet no-op reference; this is a policy contract value, not an observed measurement.",
        }

    if kpi == "passenger_wait_p95_seconds":
        return {
            "classification": "not_derivable_current_artifact",
            "action": "leave_null_with_reason",
            "reason": "Cannot derive p95 passenger wait from average wait aggregate alone.",
        }

    if kpi in {"passenger_demand_generated", "passenger_served_count"}:
        return {
            "classification": "missing_source_quantity",
            "action": "do_not_impute_from_wait_passenger_count_without_explicit_contract",
            "reason": "Demand/served counts need explicit simulator or rollout semantics.",
        }

    return {
        "classification": "unknown_gap",
        "action": "manual_review",
        "reason": "No automatic rule defined.",
    }


def build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("# B1_noop 12-KPI Baseline Schema Gap Audit")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"- audit_status: `{report['audit_status']}`")
    lines.append(f"- hard_failure_count: `{len(report['hard_failures'])}`")
    lines.append(f"- warning_count: `{len(report['warnings'])}`")
    lines.append(f"- causal_comparison_allowed: `{report['claim_guards']['causal_comparison_allowed']}`")
    lines.append(f"- paper_level_claim_allowed: `{report['claim_guards']['paper_level_claim_allowed']}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("B1_noop remains usable as a non-causal no-op reference baseline.")
    lines.append("")
    lines.append("The 12-KPI warning is a schema completeness warning, not a hard failure.")
    lines.append("")
    lines.append("Before paper-level 12-KPI comparison, missing extended KPI fields must be either derived by explicit contract, kept null with reason, or regenerated from a richer rollout.")
    lines.append("")
    lines.append("## KPI Presence")
    lines.append("")
    lines.append("| KPI | kpi_by_window | window_rollup | gap_classification | action |")
    lines.append("|---|---:|---:|---|---|")

    window_available = set(report["kpi_by_window"]["available_kpis"])
    rollup_available = set(report["window_rollup"]["available_kpis"])

    for kpi in KPI_12:
        gap = report["gap_classification"][kpi]
        lines.append(
            f"| `{kpi}` | {kpi in window_available} | {kpi in rollup_available} | "
            f"`{gap['classification']}` | `{gap['action']}` |"
        )

    lines.append("")
    lines.append("## Required Follow-up")
    lines.append("")
    for item in report["recommended_next_actions"]:
        lines.append(f"- {item}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--b1-root",
        default="artifacts/baseline_v1/B1_noop",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/baseline_v1/B1_noop/schema_gap_audit",
    )
    args = parser.parse_args()

    b1_root = Path(args.b1_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    canonical_root = b1_root / "canonical_eval"
    overall_path = canonical_root / "kpi_overall.json"
    kpi_by_window_path = canonical_root / "kpi_by_window.parquet"

    window_rollup_paths = sorted(b1_root.rglob("window_rollup.parquet"))

    overall = load_json_optional(overall_path)
    kpi_by_window_df = read_parquet_optional(kpi_by_window_path)

    if window_rollup_paths:
        rollup_frames = [pd.read_parquet(p).assign(input_source_path=str(p)) for p in window_rollup_paths]
        window_rollup_df = pd.concat(rollup_frames, ignore_index=True)
    else:
        window_rollup_df = None

    hard_failures: List[str] = []
    warnings: List[str] = []

    if overall is None:
        hard_failures.append(f"missing kpi_overall.json: {overall_path}")

    if kpi_by_window_df is None:
        hard_failures.append(f"missing kpi_by_window.parquet: {kpi_by_window_path}")

    if window_rollup_df is None:
        warnings.append(f"no window_rollup.parquet found under {b1_root}")

    overall_presence = kpi_presence_from_overall(overall)
    window_presence = column_presence(kpi_by_window_df)
    rollup_presence = column_presence(window_rollup_df)

    window_cols = window_presence["columns"]
    rollup_cols = rollup_presence["columns"]

    gap_classification = {
        k: classify_gap(k, window_cols=window_cols, rollup_cols=rollup_cols)
        for k in KPI_12
    }

    missing_in_overall = overall_presence["missing_kpis"]
    if missing_in_overall:
        warnings.append(f"B1_noop kpi_overall is missing {len(missing_in_overall)} of 12 KPI fields: {missing_in_overall}")

    non_derivable = [
        k for k, v in gap_classification.items()
        if v["classification"] in {
            "not_derivable_current_artifact",
            "missing_source_quantity",
            "unknown_gap",
        }
    ]

    if non_derivable:
        warnings.append(f"non-derivable or source-missing KPI fields require explicit follow-up: {non_derivable}")

    claim_guards = {
        "causal_comparison_allowed": False,
        "paper_level_claim_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "b1_schema_gap_audit_only": True,
    }

    recommended_next_actions = [
        "Do not change causal_comparison_allowed; keep it false.",
        "Do not impute passenger_demand_generated or passenger_served_count from wait_passenger_count without an explicit contract.",
        "Keep passenger_wait_p95_seconds null unless raw passenger wait distribution or approved proxy is available.",
        "fleet_reduction_ratio may be set to 0.0 only in a separate compatibility artifact with source='contract_fixed'.",
        "For paper-level 12-KPI tables, generate a B1_noop_12kpi_compat artifact or regenerate B1 from a 12-KPI-capable simulator rollout.",
    ]

    audit_status = "PASS_WITH_SCHEMA_GAP_WARNINGS" if not hard_failures else "FAIL"

    report = {
        "artifact_version": "b1_noop_12kpi_schema_gap_audit_v1",
        "baseline_id": "B1_noop",
        "audit_status": audit_status,
        "b1_root": str(b1_root),
        "canonical_root": str(canonical_root),
        "input_files": {
            "kpi_overall": str(overall_path),
            "kpi_by_window": str(kpi_by_window_path),
            "window_rollups": [str(p) for p in window_rollup_paths],
        },
        "overall": overall_presence,
        "kpi_by_window": window_presence,
        "window_rollup": rollup_presence,
        "gap_classification": gap_classification,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "claim_guards": claim_guards,
        "recommended_next_actions": recommended_next_actions,
    }

    json_path = output_root / "b1_noop_12kpi_schema_gap_audit.json"
    md_path = output_root / "b1_noop_12kpi_schema_gap_audit.md"

    dump_json(json_path, report)
    md_path.write_text(build_markdown(report), encoding="utf-8")

    print("[OK] B1_noop 12-KPI schema gap audit completed")
    print(f"[OK] audit_status : {audit_status}")
    print(f"[OK] hard_failures: {len(hard_failures)}")
    print(f"[OK] warnings     : {len(warnings)}")
    print(f"[OK] report_json  : {json_path}")
    print(f"[OK] report_md    : {md_path}")

    if hard_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
