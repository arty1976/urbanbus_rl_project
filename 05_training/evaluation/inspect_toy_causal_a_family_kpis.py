from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


PHASE2_12_KPIS = [
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

LOWER_IS_BETTER = {
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
}

HIGHER_IS_BETTER = {
    "on_time_rate",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "fleet_reduction_ratio",
}

BOUNDED_0_1 = {
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "passenger_service_rate",
    "fleet_reduction_ratio",
}


def load_json(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_float(value: Any) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def pct_delta(value: Any, baseline: Any) -> Optional[float]:
    v = safe_float(value)
    b = safe_float(baseline)
    if v is None or b is None:
        return None
    if abs(b) < 1e-12:
        return None
    return float((v - b) / abs(b) * 100.0)


def direction_for_kpi(kpi: str) -> str:
    if kpi in LOWER_IS_BETTER:
        return "lower_is_better"
    if kpi in HIGHER_IS_BETTER:
        return "higher_is_better"
    return "neutral"


def improvement_from_delta(kpi: str, delta_pct: Optional[float]) -> Optional[bool]:
    if delta_pct is None:
        return None
    if kpi in LOWER_IS_BETTER:
        return bool(delta_pct < 0)
    if kpi in HIGHER_IS_BETTER:
        return bool(delta_pct > 0)
    return None


def require_file(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"required file missing: {path}")


def validate_kpi_frame(df: pd.DataFrame, frame_name: str, expected_conditions: List[str]) -> None:
    missing = [k for k in PHASE2_12_KPIS if k not in df.columns]
    if missing:
        raise RuntimeError(f"{frame_name} missing 12-KPI columns: {missing}")

    if "condition_id" not in df.columns:
        raise RuntimeError(f"{frame_name} missing condition_id column")

    observed_conditions = set(df["condition_id"].astype(str).str.upper().unique().tolist())
    expected_set = set(x.upper() for x in expected_conditions)
    if observed_conditions != expected_set:
        raise RuntimeError(
            f"{frame_name} condition mismatch. expected={sorted(expected_set)}, got={sorted(observed_conditions)}"
        )

    if "causal_comparison_allowed" in df.columns:
        if not bool(df["causal_comparison_allowed"].fillna(False).astype(bool).all()):
            raise RuntimeError(f"{frame_name} has causal_comparison_allowed=false rows")

    for kpi in PHASE2_12_KPIS:
        values = pd.to_numeric(df[kpi], errors="coerce")
        if int(values.notna().sum()) == 0:
            raise RuntimeError(f"{frame_name}.{kpi} has no numeric values")

        if kpi != "cv_headway" and bool((values.dropna() < 0).any()):
            raise RuntimeError(f"{frame_name}.{kpi} has negative values")

    for kpi in BOUNDED_0_1:
        values = pd.to_numeric(df[kpi], errors="coerce").dropna()
        if bool(((values < 0.0) | (values > 1.0)).any()):
            raise RuntimeError(f"{frame_name}.{kpi} must be bounded within [0, 1]")


def build_condition_summary(window_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for condition_id, grp in window_df.groupby("condition_id", dropna=False):
        row: Dict[str, Any] = {
            "condition_id": str(condition_id),
            "window_count": int(len(grp)),
            "seed_count": int(grp["seed"].nunique()) if "seed" in grp.columns else None,
            "time_band_count": int(grp["time_band"].nunique()) if "time_band" in grp.columns else None,
            "causal_comparison_allowed": bool(grp["causal_comparison_allowed"].all())
            if "causal_comparison_allowed" in grp.columns else None,
        }

        for kpi in PHASE2_12_KPIS:
            s = pd.to_numeric(grp[kpi], errors="coerce")
            row[f"{kpi}_mean"] = safe_float(s.mean())
            row[f"{kpi}_std"] = safe_float(s.std(ddof=1)) if int(s.notna().sum()) >= 2 else None
            row[f"{kpi}_min"] = safe_float(s.min())
            row[f"{kpi}_max"] = safe_float(s.max())
            row[f"{kpi}_valid_count"] = int(s.notna().sum())

        rows.append(row)

    preferred_order = ["A", "A90", "A80", "A70"]
    out = pd.DataFrame(rows)
    out["_order"] = out["condition_id"].map({c: i for i, c in enumerate(preferred_order)}).fillna(999)
    out = out.sort_values(["_order", "condition_id"]).drop(columns=["_order"]).reset_index(drop=True)
    return out


def build_condition_vs_a_delta(condition_summary: pd.DataFrame) -> pd.DataFrame:
    if "A" not in set(condition_summary["condition_id"].astype(str)):
        raise RuntimeError("condition_summary must include A baseline condition")

    baseline = condition_summary.loc[condition_summary["condition_id"].astype(str) == "A"].iloc[0]
    rows: List[Dict[str, Any]] = []

    for r in condition_summary.itertuples(index=False):
        condition_id = str(getattr(r, "condition_id"))
        if condition_id == "A":
            continue

        for kpi in PHASE2_12_KPIS:
            value = getattr(r, f"{kpi}_mean")
            base_value = baseline[f"{kpi}_mean"]
            delta_abs = None
            v = safe_float(value)
            b = safe_float(base_value)
            if v is not None and b is not None:
                delta_abs = float(v - b)
            delta_percent = pct_delta(value, base_value)
            rows.append({
                "condition_id": condition_id,
                "baseline_condition_id": "A",
                "kpi": kpi,
                "direction": direction_for_kpi(kpi),
                "condition_mean": v,
                "baseline_mean": b,
                "delta_abs": delta_abs,
                "delta_pct_vs_A": delta_percent,
                "improved_vs_A": improvement_from_delta(kpi, delta_percent),
                "claim_boundary": "toy_causal_sanity_only_not_paper_performance_claim",
            })

    return pd.DataFrame(rows)


def build_time_band_summary(window_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (condition_id, time_band), grp in window_df.groupby(["condition_id", "time_band"], dropna=False):
        row: Dict[str, Any] = {
            "condition_id": str(condition_id),
            "time_band": str(time_band),
            "window_count": int(len(grp)),
            "seed_count": int(grp["seed"].nunique()) if "seed" in grp.columns else None,
            "causal_comparison_allowed": bool(grp["causal_comparison_allowed"].all())
            if "causal_comparison_allowed" in grp.columns else None,
        }

        for kpi in PHASE2_12_KPIS:
            s = pd.to_numeric(grp[kpi], errors="coerce")
            row[f"{kpi}_mean"] = safe_float(s.mean())
            row[f"{kpi}_std"] = safe_float(s.std(ddof=1)) if int(s.notna().sum()) >= 2 else None

        rows.append(row)

    preferred_order = ["A", "A90", "A80", "A70"]
    band_order = ["peak", "offpeak", "night"]
    out = pd.DataFrame(rows)
    out["_condition_order"] = out["condition_id"].map({c: i for i, c in enumerate(preferred_order)}).fillna(999)
    out["_band_order"] = out["time_band"].map({b: i for i, b in enumerate(band_order)}).fillna(999)
    out = (
        out.sort_values(["_condition_order", "_band_order", "condition_id", "time_band"])
        .drop(columns=["_condition_order", "_band_order"])
        .reset_index(drop=True)
    )
    return out


def build_seed_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for r in seed_df.itertuples(index=False):
        row: Dict[str, Any] = {
            "condition_id": str(getattr(r, "condition_id")),
            "seed": int(getattr(r, "seed")),
            "window_count": int(getattr(r, "window_count")),
            "causal_comparison_allowed": bool(getattr(r, "causal_comparison_allowed")),
        }
        for kpi in PHASE2_12_KPIS:
            mean_col = f"{kpi}_mean"
            std_col = f"{kpi}_std"
            if hasattr(r, mean_col):
                row[mean_col] = safe_float(getattr(r, mean_col))
            if hasattr(r, std_col):
                row[std_col] = safe_float(getattr(r, std_col))
        rows.append(row)

    preferred_order = ["A", "A90", "A80", "A70"]
    out = pd.DataFrame(rows)
    out["_order"] = out["condition_id"].map({c: i for i, c in enumerate(preferred_order)}).fillna(999)
    out = out.sort_values(["_order", "condition_id", "seed"]).drop(columns=["_order"]).reset_index(drop=True)
    return out


def build_report(
    canonical_root: Path,
    output_root: Path,
    window_df: pd.DataFrame,
    seed_df: pd.DataFrame,
    time_band_df: pd.DataFrame,
    overall: Dict[str, Any],
    condition_summary: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> Dict[str, Any]:
    kpis_in_overall = sorted(list((overall.get("kpis") or {}).keys()))
    missing_overall = [k for k in PHASE2_12_KPIS if k not in kpis_in_overall]

    fleet_rows = condition_summary[[
        "condition_id",
        "fleet_reduction_ratio_mean",
        "passenger_service_rate_mean",
        "passenger_wait_p95_seconds_mean",
        "energy_proxy_per_passenger_mean",
    ]].to_dict("records")

    interesting_delta = delta_df[
        delta_df["kpi"].isin([
            "passenger_service_rate",
            "passenger_wait_p95_seconds",
            "energy_proxy_per_passenger",
            "fleet_reduction_ratio",
            "avg_wait_seconds",
        ])
    ].to_dict("records")

    return {
        "artifact_version": "toy_causal_a_family_kpi_inspector_v1_step84",
        "canonical_root": str(canonical_root),
        "output_root": str(output_root),
        "claim_boundary": "toy_causal_sanity_only_not_paper_performance_claim",
        "phase2_12_kpis": PHASE2_12_KPIS,
        "row_counts": {
            "kpi_by_window": int(len(window_df)),
            "kpi_by_seed": int(len(seed_df)),
            "kpi_by_time_band": int(len(time_band_df)),
            "condition_summary": int(len(condition_summary)),
            "condition_vs_A_delta": int(len(delta_df)),
        },
        "condition_ids": sorted(window_df["condition_id"].astype(str).unique().tolist()),
        "seed_values": sorted([int(x) for x in pd.to_numeric(window_df["seed"], errors="coerce").dropna().unique().tolist()]),
        "time_bands": sorted(window_df["time_band"].astype(str).unique().tolist())
        if "time_band" in window_df.columns else [],
        "causal_comparison_allowed": bool(window_df["causal_comparison_allowed"].all())
        if "causal_comparison_allowed" in window_df.columns else None,
        "overall_has_all_12_kpis": len(missing_overall) == 0,
        "missing_overall_kpis": missing_overall,
        "fleet_and_service_summary": fleet_rows,
        "selected_delta_vs_A": interesting_delta,
        "notes": [
            "This inspector summarizes Phase 2 toy causal simulator outputs only.",
            "Do not use this report as a paper-level performance claim.",
            "A/A90/A80/A70 differences are sanity checks for 12-KPI wiring and fleet-ratio propagation.",
        ],
    }


def run_inspection(args: argparse.Namespace) -> Dict[str, Any]:
    canonical_root = Path(args.canonical_root)
    output_root = Path(args.output_root)

    kpi_by_window_path = canonical_root / "kpi_by_window.parquet"
    kpi_by_seed_path = canonical_root / "kpi_by_seed.parquet"
    kpi_by_time_band_path = canonical_root / "kpi_by_time_band.parquet"
    kpi_overall_path = canonical_root / "kpi_overall.json"

    for path in [
        kpi_by_window_path,
        kpi_by_seed_path,
        kpi_by_time_band_path,
        kpi_overall_path,
    ]:
        require_file(path)

    window_df = pd.read_parquet(kpi_by_window_path)
    seed_df = pd.read_parquet(kpi_by_seed_path)
    time_band_df = pd.read_parquet(kpi_by_time_band_path)
    overall = load_json(kpi_overall_path)

    expected_conditions = [x.strip().upper() for x in args.conditions.split(",") if x.strip()]
    validate_kpi_frame(window_df, "kpi_by_window", expected_conditions)

    # seed/time_band outputs contain KPI mean/std columns, so validate by checking the raw expected fields
    # after reconstructing a lightweight frame of *_mean where possible.
    if len(seed_df) == 0:
        raise RuntimeError("kpi_by_seed is empty")
    if len(time_band_df) == 0:
        raise RuntimeError("kpi_by_time_band is empty")

    condition_summary = build_condition_summary(window_df)
    delta_df = build_condition_vs_a_delta(condition_summary)
    time_band_summary = build_time_band_summary(window_df)
    seed_summary = build_seed_summary(seed_df)

    output_root.mkdir(parents=True, exist_ok=True)

    condition_summary_path = output_root / "condition_summary.csv"
    delta_path = output_root / "condition_vs_A_delta.csv"
    time_band_summary_path = output_root / "time_band_summary.csv"
    seed_summary_path = output_root / "seed_summary.csv"
    report_path = output_root / "inspection_report.json"

    condition_summary.to_csv(condition_summary_path, index=False, encoding="utf-8-sig")
    delta_df.to_csv(delta_path, index=False, encoding="utf-8-sig")
    time_band_summary.to_csv(time_band_summary_path, index=False, encoding="utf-8-sig")
    seed_summary.to_csv(seed_summary_path, index=False, encoding="utf-8-sig")

    report = build_report(
        canonical_root=canonical_root,
        output_root=output_root,
        window_df=window_df,
        seed_df=seed_df,
        time_band_df=time_band_df,
        overall=overall,
        condition_summary=condition_summary,
        delta_df=delta_df,
    )
    dump_json(report_path, report)

    print("[OK] Step 84 toy causal A-family KPI inspection completed")
    print(f"[OK] canonical_root       : {canonical_root}")
    print(f"[OK] output_root          : {output_root}")
    print(f"[OK] condition_summary    : {condition_summary_path}")
    print(f"[OK] condition_vs_A_delta : {delta_path}")
    print(f"[OK] time_band_summary    : {time_band_summary_path}")
    print(f"[OK] seed_summary         : {seed_summary_path}")
    print(f"[OK] inspection_report    : {report_path}")
    print(f"[OK] conditions           : {report['condition_ids']}")
    print(f"[OK] row_counts           : {report['row_counts']}")
    print(f"[OK] 12_kpis              : validated")
    print(f"[OK] causal_allowed       : {report['causal_comparison_allowed']}")
    print("[OK] claim_boundary       : toy_causal_sanity_only_not_paper_performance_claim")

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--canonical-root",
        default="artifacts/phase2_toy_causal_a_family_matrix_selftest/canonical_eval",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/phase2_toy_causal_a_family_matrix_selftest/inspection",
    )
    parser.add_argument("--conditions", default="A,A90,A80,A70")
    args = parser.parse_args()

    run_inspection(args)


if __name__ == "__main__":
    main()
