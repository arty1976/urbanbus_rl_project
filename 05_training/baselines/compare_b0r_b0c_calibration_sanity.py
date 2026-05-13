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


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except Exception:
        return None
    if pd.isna(v):
        return None
    return v


def fmt(v: Optional[float]) -> str:
    if v is None:
        return "null"
    if abs(v) >= 100:
        return f"{v:.2f}"
    return f"{v:.4f}"


def kpi_mean(overall: Dict[str, Any], kpi: str) -> Optional[float]:
    return safe_float(overall.get("kpis", {}).get(kpi, {}).get("mean"))


def kpi_valid_count(overall: Dict[str, Any], kpi: str) -> int:
    return int(overall.get("kpis", {}).get(kpi, {}).get("valid_window_count", 0) or 0)


def classify_gap(kpi: str, b0r_mean: Optional[float], b0c_mean: Optional[float]) -> str:
    if b0r_mean is None and b0c_mean is None:
        return "not_comparable_both_null"

    if b0r_mean is None and b0c_mean is not None:
        return "b0r_not_observed_b0c_available"

    if b0r_mean is not None and b0c_mean is None:
        return "b0r_available_b0c_missing"

    assert b0r_mean is not None and b0c_mean is not None

    if b0r_mean == 0 and b0c_mean == 0:
        return "aligned_zero"

    if b0r_mean == 0 and b0c_mean != 0:
        return "baseline_zero_b0c_nonzero_check_definition"

    ratio = b0c_mean / b0r_mean

    # Loose sanity bands. These are not pass/fail scientific tolerances.
    if kpi in {"avg_wait_seconds", "passenger_wait_p95_seconds", "energy_proxy"}:
        if 0.50 <= ratio <= 1.50:
            return "roughly_same_scale"
        return "scale_gap_warning"

    if kpi in {"cv_headway", "bunching_rate", "on_time_rate"}:
        if 0.50 <= ratio <= 1.50:
            return "roughly_same_scale"
        return "scale_gap_warning"

    return "comparable_value_recorded"


def build_kpi_comparison(
    b0r_overall: Dict[str, Any],
    b0c_overall: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for kpi in KPI_12:
        b0r_mean = kpi_mean(b0r_overall, kpi)
        b0c_mean = kpi_mean(b0c_overall, kpi)

        if b0r_mean is not None and b0c_mean is not None:
            abs_delta = b0c_mean - b0r_mean
            ratio = None if b0r_mean == 0 else b0c_mean / b0r_mean
        else:
            abs_delta = None
            ratio = None

        rows.append({
            "kpi": kpi,
            "b0r_mean": b0r_mean,
            "b0c_mean": b0c_mean,
            "absolute_delta_b0c_minus_b0r": abs_delta,
            "ratio_b0c_over_b0r": ratio,
            "b0r_valid_window_count": kpi_valid_count(b0r_overall, kpi),
            "b0c_valid_window_count": kpi_valid_count(b0c_overall, kpi),
            "gap_status": classify_gap(kpi, b0r_mean, b0c_mean),
        })

    return pd.DataFrame(rows)


def build_time_band_comparison(b0r_window_path: Path, b0c_window_path: Path) -> pd.DataFrame:
    b0r = pd.read_parquet(b0r_window_path).copy()
    b0c = pd.read_parquet(b0c_window_path).copy()

    rows: List[Dict[str, Any]] = []

    for kpi in KPI_12:
        if kpi not in b0r.columns or kpi not in b0c.columns:
            continue

        # B0R compatibility files may contain pd.NA/object dtype columns.
        # Convert explicitly before groupby aggregation.
        b0r_tmp = b0r[["time_band", kpi]].copy()
        b0c_tmp = b0c[["time_band", kpi]].copy()

        b0r_tmp[kpi] = pd.to_numeric(b0r_tmp[kpi], errors="coerce")
        b0c_tmp[kpi] = pd.to_numeric(b0c_tmp[kpi], errors="coerce")

        b0r_grp = b0r_tmp.groupby("time_band", dropna=False)[kpi].mean()
        b0c_grp = b0c_tmp.groupby("time_band", dropna=False)[kpi].mean()

        time_bands = sorted(set(map(str, b0r_grp.index.tolist())) | set(map(str, b0c_grp.index.tolist())))

        for tb in time_bands:
            b0r_val = safe_float(b0r_grp.get(tb))
            b0c_val = safe_float(b0c_grp.get(tb))

            if b0r_val is not None and b0c_val is not None:
                delta = b0c_val - b0r_val
                ratio = None if b0r_val == 0 else b0c_val / b0r_val
            else:
                delta = None
                ratio = None

            rows.append({
                "time_band": tb,
                "kpi": kpi,
                "b0r_mean": b0r_val,
                "b0c_mean": b0c_val,
                "absolute_delta_b0c_minus_b0r": delta,
                "ratio_b0c_over_b0r": ratio,
                "gap_status": classify_gap(kpi, b0r_val, b0c_val),
            })

    return pd.DataFrame(rows)


def build_markdown(
    payload: Dict[str, Any],
    kpi_df: pd.DataFrame,
    time_band_df: pd.DataFrame,
) -> str:
    lines: List[str] = []

    lines.append("# B0R vs B0C Calibration Sanity Check")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("This report compares B0R_historical_12kpi_compat and B0C_causal_shadow_v1.")
    lines.append("")
    lines.append("B0R is a historical replay / compatibility reference. B0C is a synthetic causal-shadow baseline generated for Phase 2 simulator-internal comparison.")
    lines.append("")
    lines.append("## Claim Guard")
    lines.append("")
    lines.append("| Guard | Value |")
    lines.append("|---|---|")
    for k, v in payload["claim_guards"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Overall KPI Comparison")
    lines.append("")
    lines.append("| KPI | B0R mean | B0C mean | ratio B0C/B0R | status |")
    lines.append("|---|---:|---:|---:|---|")

    for _, row in kpi_df.iterrows():
        ratio = row["ratio_b0c_over_b0r"]
        ratio_text = "null" if pd.isna(ratio) else f"{float(ratio):.4f}"
        lines.append(
            f"| {row['kpi']} | {fmt(safe_float(row['b0r_mean']))} | "
            f"{fmt(safe_float(row['b0c_mean']))} | {ratio_text} | {row['gap_status']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- A scale gap is not an automatic failure at this stage.")
    lines.append("- A scale gap means B0C simulator assumptions differ from the historical B0R reference and must be calibrated before causal claims.")
    lines.append("- B0C remains not simulator-validated unless a later tolerance report passes.")
    lines.append("- B0C must not be presented as observed historical operation.")
    lines.append("")

    warning_rows = kpi_df[kpi_df["gap_status"].astype(str).str.contains("warning|not_observed|missing", case=False, regex=True)]
    if len(warning_rows):
        lines.append("## Warnings")
        lines.append("")
        for _, row in warning_rows.iterrows():
            lines.append(f"- {row['kpi']}: {row['gap_status']}")
        lines.append("")

    lines.append("## Output Status")
    lines.append("")
    lines.append(f"- audit_status: {payload['audit_status']}")
    lines.append(f"- hard_failures: {len(payload['hard_failures'])}")
    lines.append(f"- warnings: {len(payload['warnings'])}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--b0r-root",
        default="artifacts/baseline_v1/B0R_historical_12kpi_compat",
    )
    parser.add_argument(
        "--b0c-root",
        default="artifacts/baseline_v1/B0C_causal_shadow_v1/canonical_eval",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/baseline_v1/B0C_causal_shadow_v1/calibration_sanity",
    )
    args = parser.parse_args()

    b0r_root = Path(args.b0r_root)
    b0c_root = Path(args.b0c_root)
    output_root = Path(args.output_root)

    b0r_overall_path = b0r_root / "kpi_overall.json"
    b0c_overall_path = b0c_root / "kpi_overall.json"
    b0r_window_path = b0r_root / "kpi_by_window.parquet"
    b0c_window_path = b0c_root / "kpi_by_window.parquet"

    hard_failures: List[str] = []
    warnings: List[str] = []

    for label, path in [
        ("B0R overall", b0r_overall_path),
        ("B0C overall", b0c_overall_path),
        ("B0R window", b0r_window_path),
        ("B0C window", b0c_window_path),
    ]:
        if not path.exists():
            hard_failures.append(f"missing {label}: {path}")

    if hard_failures:
        payload = {
            "artifact_version": "b0r_vs_b0c_calibration_sanity_v1",
            "audit_status": "FAIL",
            "hard_failures": hard_failures,
            "warnings": warnings,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        dump_json(output_root / "b0r_vs_b0c_calibration_sanity.json", payload)
        raise SystemExit(f"[FAIL] {hard_failures}")

    b0r_overall = load_json(b0r_overall_path)
    b0c_overall = load_json(b0c_overall_path)

    if bool(b0r_overall.get("causal_comparison_allowed", False)):
        hard_failures.append("B0R causal_comparison_allowed must be false")

    if bool(b0c_overall.get("causal_comparison_allowed", False)):
        hard_failures.append("B0C causal_comparison_allowed must be false until simulator validation")

    kpi_df = build_kpi_comparison(b0r_overall, b0c_overall)

    for _, row in kpi_df.iterrows():
        status = str(row["gap_status"])
        if "scale_gap_warning" in status:
            warnings.append(f"{row['kpi']}: scale_gap_warning")
        if "not_observed" in status or "missing" in status:
            warnings.append(f"{row['kpi']}: {status}")

    time_band_df = build_time_band_comparison(b0r_window_path, b0c_window_path)

    audit_status = "FAIL" if hard_failures else ("PASS_WITH_WARNINGS" if warnings else "PASS")

    payload: Dict[str, Any] = {
        "artifact_version": "b0r_vs_b0c_calibration_sanity_v1",
        "audit_status": audit_status,
        "b0r_root": str(b0r_root),
        "b0c_root": str(b0c_root),
        "b0r_overall": str(b0r_overall_path),
        "b0c_overall": str(b0c_overall_path),
        "b0r_windows": int(b0r_overall.get("n_windows_total", 0) or 0),
        "b0c_windows": int(b0c_overall.get("n_windows_total", 0) or 0),
        "claim_guards": {
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "actual_results": False,
            "simulator_validation_required_before_claims": True,
        },
        "hard_failures": hard_failures,
        "warnings": warnings,
        "interpretation": {
            "b0r_role": "historical replay / compatibility reference",
            "b0c_role": "synthetic causal-shadow baseline, not observed historical operation",
            "scale_gap_meaning": "A scale gap records simulator calibration need. It is not a causal performance result.",
        },
        "output_files": {
            "summary_json": str(output_root / "b0r_vs_b0c_calibration_sanity.json"),
            "kpi_csv": str(output_root / "b0r_vs_b0c_kpi_comparison.csv"),
            "time_band_csv": str(output_root / "b0r_vs_b0c_time_band_comparison.csv"),
            "markdown": str(output_root / "b0r_vs_b0c_calibration_sanity.md"),
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)

    kpi_csv = output_root / "b0r_vs_b0c_kpi_comparison.csv"
    time_band_csv = output_root / "b0r_vs_b0c_time_band_comparison.csv"
    json_path = output_root / "b0r_vs_b0c_calibration_sanity.json"
    md_path = output_root / "b0r_vs_b0c_calibration_sanity.md"

    kpi_df.to_csv(kpi_csv, index=False, encoding="utf-8-sig")
    time_band_df.to_csv(time_band_csv, index=False, encoding="utf-8-sig")
    dump_json(json_path, payload)
    md_path.write_text(build_markdown(payload, kpi_df, time_band_df), encoding="utf-8")

    print("[OK] B0R vs B0C calibration sanity check completed")
    print("[OK] audit_status:", audit_status)
    print("[OK] hard_failures:", len(hard_failures))
    print("[OK] warnings:", len(warnings))
    print("[OK] output_root:", output_root)
    print("[OK] summary_json:", json_path)
    print("[OK] kpi_csv:", kpi_csv)
    print("[OK] time_band_csv:", time_band_csv)
    print("[OK] markdown:", md_path)

    if hard_failures:
        raise SystemExit(f"[FAIL] {hard_failures}")


if __name__ == "__main__":
    main()
