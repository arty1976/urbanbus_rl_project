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


BASELINES = {
    "B0R_historical_12kpi_compat": {
        "root": "artifacts/baseline_v1/B0R_historical_12kpi_compat",
        "role": "historical replay baseline, 12-KPI compatibility artifact",
    },
    "B1_noop": {
        "root": "artifacts/baseline_v1/B1_noop/canonical_eval",
        "role": "no-op replay baseline",
    },
    "B2_rulebased_calibrated": {
        "root": "artifacts/baseline_v1/B2_rulebased_calibrated/canonical_eval",
        "role": "calibrated rule-based replay baseline",
    },
}


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def fmt_value(value: Optional[float]) -> str:
    if value is None:
        return "null"
    try:
        v = float(value)
    except Exception:
        return str(value)

    if abs(v) >= 100:
        return f"{v:.2f}"
    return f"{v:.4f}"


def infer_kpi_status(baseline_id: str, kpi: str, mean: Optional[float], valid_count: int, null_count: int) -> str:
    if valid_count == 0:
        return "not_observed_or_null"

    if baseline_id == "B0R_historical_12kpi_compat":
        if kpi in {"avg_wait_seconds", "intervention_rate", "fleet_reduction_ratio"}:
            return "usable_with_b0r_compat_caution"
        return "not_observed_in_legacy_b0r_compat"

    if kpi in {
        "passenger_demand_generated",
        "passenger_served_count",
        "passenger_service_rate",
        "energy_proxy_per_passenger",
    }:
        if mean == 0:
            return "placeholder_zero_do_not_interpret_as_real_zero"

    if kpi == "fleet_reduction_ratio" and mean == 0:
        return "defined_baseline_zero_or_placeholder_by_condition"

    return "usable_replay_metric_noncausal"


def load_baseline(baseline_id: str, spec: Dict[str, str]) -> Dict[str, Any]:
    root = Path(spec["root"])
    overall_path = root / "kpi_overall.json"
    manifest_path = root / "aggregation_manifest.json"

    if not overall_path.exists():
        raise FileNotFoundError(f"missing kpi_overall.json for {baseline_id}: {overall_path}")

    overall = read_json(overall_path)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    kpis = overall.get("kpis", {})
    rows: List[Dict[str, Any]] = []

    for kpi in KPI_12:
        entry = kpis.get(kpi, {})
        mean = entry.get("mean")
        std = entry.get("std")
        valid_count = int(entry.get("valid_window_count", 0) or 0)
        null_count = int(entry.get("null_window_count", 0) or 0)

        rows.append({
            "baseline_id": baseline_id,
            "role": spec["role"],
            "kpi": kpi,
            "mean": mean,
            "std": std,
            "valid_window_count": valid_count,
            "null_window_count": null_count,
            "status": infer_kpi_status(baseline_id, kpi, mean, valid_count, null_count),
            "causal_comparison_allowed": bool(overall.get("causal_comparison_allowed", False)),
            "n_windows_total": int(overall.get("n_windows_total", 0) or 0),
            "n_condition_seed_pairs": int(overall.get("n_condition_seed_pairs", 0) or 0),
            "artifact_version": overall.get("artifact_version"),
            "mode": overall.get("mode", manifest.get("mode")),
        })

    return {
        "baseline_id": baseline_id,
        "root": str(root),
        "overall_path": str(overall_path),
        "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        "overall": overall,
        "manifest": manifest,
        "rows": rows,
    }


def build_markdown(summary_df: pd.DataFrame, payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Baseline Refresh Summary")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This summary compares B0R, B1, and calibrated B2 under the current 12-KPI schema.")
    lines.append("")
    lines.append("Important guard:")
    lines.append("")
    lines.append("```text")
    lines.append("causal_comparison_allowed = false")
    lines.append("paper_level_claim_allowed = false")
    lines.append("These are replay/compatibility baseline artifacts, not causal performance evidence.")
    lines.append("```")
    lines.append("")

    pivot = summary_df.pivot(index="kpi", columns="baseline_id", values="mean").reset_index()

    lines.append("## KPI Mean Table")
    lines.append("")
    columns = ["kpi"] + [c for c in pivot.columns if c != "kpi"]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in pivot.iterrows():
        values = [str(row["kpi"])]
        for c in columns[1:]:
            values.append(fmt_value(row.get(c)))
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("- B2 calibrated is now distinct from B1 no-op because intervention_rate is non-zero.")
    lines.append("- B0R is a 12-KPI compatibility artifact; unobserved legacy fields remain null or guarded.")
    lines.append("- Demand/service KPIs with zero means should not automatically be interpreted as real zero demand.")
    lines.append("- All three baselines remain non-causal replay/compatibility artifacts.")
    lines.append("")

    lines.append("## Baseline Metadata")
    lines.append("")
    lines.append("| baseline_id | windows | condition_seed_pairs | causal_allowed |")
    lines.append("| --- | ---: | ---: | --- |")
    for baseline_id, meta in payload["baseline_summary"].items():
        lines.append(
            f"| {baseline_id} | {meta['n_windows_total']} | "
            f"{meta['n_condition_seed_pairs']} | {meta['causal_comparison_allowed']} |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="artifacts/baseline_v1/baseline_refresh_summary",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    loaded = {
        baseline_id: load_baseline(baseline_id, spec)
        for baseline_id, spec in BASELINES.items()
    }

    rows: List[Dict[str, Any]] = []
    baseline_summary: Dict[str, Any] = {}

    for baseline_id, item in loaded.items():
        rows.extend(item["rows"])
        overall = item["overall"]
        baseline_summary[baseline_id] = {
            "root": item["root"],
            "overall_path": item["overall_path"],
            "manifest_path": item["manifest_path"],
            "n_windows_total": int(overall.get("n_windows_total", 0) or 0),
            "n_condition_seed_pairs": int(overall.get("n_condition_seed_pairs", 0) or 0),
            "causal_comparison_allowed": bool(overall.get("causal_comparison_allowed", False)),
            "artifact_version": overall.get("artifact_version"),
            "mode": overall.get("mode", item["manifest"].get("mode")),
        }

    summary_df = pd.DataFrame(rows)

    # Wide mean table for quick reading.
    mean_pivot = summary_df.pivot(index="kpi", columns="baseline_id", values="mean").reset_index()

    csv_path = output_root / "baseline_refresh_summary_long.csv"
    mean_csv_path = output_root / "baseline_refresh_summary_mean_table.csv"
    json_path = output_root / "baseline_refresh_summary.json"
    md_path = output_root / "baseline_refresh_summary.md"

    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    mean_pivot.to_csv(mean_csv_path, index=False, encoding="utf-8-sig")

    payload = {
        "artifact_version": "baseline_refresh_summary_v1",
        "kpi_schema": KPI_12,
        "baseline_summary": baseline_summary,
        "claim_guards": {
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "actual_results": False,
        },
        "important_notes": [
            "B2_rulebased_calibrated has non-zero intervention_rate and is no longer no-op leakage.",
            "B0R_historical_12kpi_compat is compatibility refresh, not strict official KPI recomputation.",
            "Demand/service KPI zeros in B1/B2 should be treated as placeholder/proxy-unavailable unless separately validated.",
            "All artifacts remain non-causal replay/compatibility baselines.",
        ],
        "output_files": {
            "long_csv": str(csv_path),
            "mean_table_csv": str(mean_csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
    }

    dump_json(json_path, payload)
    md_path.write_text(build_markdown(summary_df, payload), encoding="utf-8")

    # Smoke checks.
    hard_failures: List[str] = []

    expected = set(BASELINES.keys())
    got = set(summary_df["baseline_id"].unique())
    if got != expected:
        hard_failures.append(f"baseline_id mismatch expected={expected} got={got}")

    for baseline_id in expected:
        n_kpis = int((summary_df["baseline_id"] == baseline_id).sum())
        if n_kpis != len(KPI_12):
            hard_failures.append(f"{baseline_id} KPI count {n_kpis} != {len(KPI_12)}")

    b2_ir = summary_df[
        (summary_df["baseline_id"] == "B2_rulebased_calibrated")
        & (summary_df["kpi"] == "intervention_rate")
    ]["mean"].iloc[0]

    if b2_ir is None or float(b2_ir) <= 0:
        hard_failures.append("B2 calibrated intervention_rate is not positive")

    if any(meta["causal_comparison_allowed"] for meta in baseline_summary.values()):
        hard_failures.append("causal_comparison_allowed must remain false for all baselines")

    if hard_failures:
        payload["audit_status"] = "FAIL"
        payload["hard_failures"] = hard_failures
        dump_json(json_path, payload)
        raise SystemExit(f"[FAIL] {hard_failures}")

    payload["audit_status"] = "PASS"
    payload["hard_failures"] = []
    dump_json(json_path, payload)

    print("[OK] Baseline refresh summary completed")
    print("[OK] audit_status: PASS")
    print("[OK] output_root:", output_root)
    print("[OK] long_csv:", csv_path)
    print("[OK] mean_table_csv:", mean_csv_path)
    print("[OK] json:", json_path)
    print("[OK] markdown:", md_path)
    print("[OK] B2 calibrated intervention_rate:", float(b2_ir))


if __name__ == "__main__":
    main()
