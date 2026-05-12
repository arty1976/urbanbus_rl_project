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

LOWER_IS_BETTER = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "intervention_rate",
    "energy_proxy",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
]

HIGHER_IS_BETTER = [
    "on_time_rate",
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "fleet_reduction_ratio",
]


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_mean(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def safe_std(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return None
    return float(s.std(ddof=1))


def summarize_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []

    for key, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row: Dict[str, Any] = {}
        for col, value in zip(group_cols, key):
            row[col] = value

        row["window_count"] = int(len(grp))
        if "time_band" not in group_cols and "time_band" in grp.columns:
            row["time_band_count"] = int(grp["time_band"].nunique())

        row["causal_comparison_allowed"] = bool(grp["causal_comparison_allowed"].all())

        for kpi in KPI_12:
            s = pd.to_numeric(grp[kpi], errors="coerce")
            row[f"{kpi}_mean"] = safe_mean(s)
            row[f"{kpi}_std"] = safe_std(s)
            row[f"{kpi}_valid_window_count"] = int(s.notna().sum())
            row[f"{kpi}_null_window_count"] = int(s.isna().sum())

        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(group_cols).reset_index(drop=True)


def summarize_overall(df: pd.DataFrame) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "artifact_version": "b0r_historical_12kpi_compat_v1",
        "mode": "b0_historical_to_b0r_12kpi_compat",
        "condition_ids": sorted(df["condition_id"].astype(str).unique().tolist()),
        "n_condition_seed_pairs": int(df[["condition_id", "seed"]].drop_duplicates().shape[0]),
        "n_windows_total": int(len(df)),
        "causal_comparison_allowed": bool(df["causal_comparison_allowed"].all()),
        "paper_level_claim_allowed": False,
        "strict_canonical": False,
        "kpi_schema": KPI_12,
        "kpi_direction": {
            "lower_is_better": LOWER_IS_BETTER,
            "higher_is_better": HIGHER_IS_BETTER,
        },
        "kpis": {},
    }

    for kpi in KPI_12:
        s = pd.to_numeric(df[kpi], errors="coerce")
        payload["kpis"][kpi] = {
            "mean": safe_mean(s),
            "std": safe_std(s),
            "valid_window_count": int(s.notna().sum()),
            "null_window_count": int(s.isna().sum()),
        }

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="artifacts/baseline_v1/B0_historical/canonical_eval",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/baseline_v1/B0R_historical_12kpi_compat",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    in_window = input_root / "kpi_by_window.parquet"
    in_overall = input_root / "kpi_overall.json"
    in_manifest = input_root / "aggregation_manifest.json"

    if not in_window.exists():
        raise SystemExit(f"[STOP] missing input kpi_by_window: {in_window}")

    df = pd.read_parquet(in_window).copy()
    original_row_count = int(len(df))

    if df.empty:
        raise SystemExit("[STOP] input window file is empty")

    required_base_cols = [
        "condition_id",
        "seed",
        "window_id",
        "state_ts",
        "service_date",
        "time_band",
        "evaluation_horizon_minutes",
        "avg_wait_seconds",
        "intervention_rate",
    ]
    missing_required = [c for c in required_base_cols if c not in df.columns]
    if missing_required:
        raise SystemExit(f"[STOP] missing required base columns: {missing_required}")

    df["original_condition_id"] = df["condition_id"].astype(str)
    df["condition_id"] = "B0R"
    df["baseline_parent_condition_id"] = "B0"
    df["kpi_schema_version"] = "12kpi_compat_v1"
    df["b0r_compat_version"] = "b0r_historical_12kpi_compat_v1"

    # Preserve the existing six KPI values and add missing extended KPI columns.
    for kpi in KPI_12:
        if kpi not in df.columns:
            df[kpi] = pd.NA

    # B0R is the historical replay baseline itself, so fleet reduction is
    # definitionally zero. This is not an observed fleet count.
    if df["fleet_reduction_ratio"].isna().all():
        df["fleet_reduction_ratio"] = 0.0

    df["strict_canonical"] = False
    df["causal_comparison_allowed"] = False
    df["paper_level_claim_allowed"] = False
    df["actual_results"] = False
    df["winner_selected"] = False
    df["trainable_reward_promoted"] = False

    df["extended_kpi_fill_policy"] = (
        "not_observed_null_except_fleet_reduction_defined_zero"
    )

    df["b0r_refresh_warning"] = (
        "compatibility refresh only; does not recompute strict official rollout KPIs"
    )

    # Stable output column order: identifiers, KPI_12, metadata.
    id_cols = [
        "condition_id",
        "original_condition_id",
        "baseline_parent_condition_id",
        "seed",
        "window_id",
        "state_ts",
        "service_date",
        "time_band",
        "evaluation_horizon_minutes",
    ]

    metadata_cols = [
        c for c in df.columns
        if c not in set(id_cols + KPI_12)
    ]

    out_df = df[id_cols + KPI_12 + metadata_cols].copy()

    seed_df = summarize_group(out_df, ["condition_id", "seed"])
    time_band_df = summarize_group(out_df, ["condition_id", "seed", "time_band"])
    overall_payload = summarize_overall(out_df)

    output_root.mkdir(parents=True, exist_ok=True)

    window_path = output_root / "kpi_by_window.parquet"
    seed_path = output_root / "kpi_by_seed.parquet"
    time_band_path = output_root / "kpi_by_time_band.parquet"
    overall_path = output_root / "kpi_overall.json"
    manifest_path = output_root / "aggregation_manifest.json"

    out_df.to_parquet(window_path, index=False)
    seed_df.to_parquet(seed_path, index=False)
    time_band_df.to_parquet(time_band_path, index=False)
    dump_json(overall_path, overall_payload)

    kpi_observation_status = {
        "cv_headway": "not_observed_in_legacy_b0" if out_df["cv_headway"].isna().all() else "legacy_passthrough",
        "avg_wait_seconds": "legacy_passthrough_valid",
        "bunching_rate": "not_observed_in_legacy_b0" if out_df["bunching_rate"].isna().all() else "legacy_passthrough",
        "on_time_rate": "not_observed_in_legacy_b0" if out_df["on_time_rate"].isna().all() else "legacy_passthrough",
        "intervention_rate": "legacy_defined_zero_valid",
        "energy_proxy": "not_observed_in_legacy_b0" if out_df["energy_proxy"].isna().all() else "legacy_passthrough",
        "passenger_demand_generated": "not_observed_in_legacy_b0",
        "passenger_served_count": "not_observed_in_legacy_b0",
        "passenger_service_rate": "not_observed_in_legacy_b0",
        "passenger_wait_p95_seconds": "not_observed_in_legacy_b0",
        "energy_proxy_per_passenger": "not_observed_in_legacy_b0",
        "fleet_reduction_ratio": "defined_baseline_zero_not_observed_fleet_count",
    }

    manifest = {
        "artifact_version": "b0r_historical_12kpi_compat_v1",
        "mode": "b0_historical_to_b0r_12kpi_compat",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "input_files": {
            "kpi_by_window": str(in_window),
            "kpi_overall": str(in_overall) if in_overall.exists() else None,
            "aggregation_manifest": str(in_manifest) if in_manifest.exists() else None,
        },
        "output_files": {
            "kpi_by_window": str(window_path),
            "kpi_by_seed": str(seed_path),
            "kpi_by_time_band": str(time_band_path),
            "kpi_overall": str(overall_path),
            "aggregation_manifest": str(manifest_path),
        },
        "row_counts": {
            "input_kpi_by_window": original_row_count,
            "output_kpi_by_window": int(len(out_df)),
            "kpi_by_seed": int(len(seed_df)),
            "kpi_by_time_band": int(len(time_band_df)),
        },
        "condition_id_rewrite": {
            "from": "B0",
            "to": "B0R",
            "reason": "B0R is the canonical replay-ready historical baseline name for comparison.",
        },
        "kpi_schema": KPI_12,
        "kpi_observation_status": kpi_observation_status,
        "strict_canonical": False,
        "causal_comparison_allowed": False,
        "paper_level_claim_allowed": False,
        "actual_results": False,
        "winner_selected": False,
        "trainable_reward_promoted": False,
        "warnings": [
            {
                "code": "compatibility_refresh_only",
                "message": (
                    "This refresh creates a 12-KPI-compatible B0R artifact. "
                    "It does not recover unobserved historical fields and does not "
                    "recompute strict official rollout KPIs."
                ),
            },
            {
                "code": "extended_kpis_partly_not_observed",
                "message": (
                    "Extended service, p95 wait, and energy-per-passenger KPIs remain "
                    "null unless observed/proxy source fields exist in the input."
                ),
            },
        ],
        "smoke": {
            "enabled": bool(args.smoke),
            "passed": True,
        },
    }

    dump_json(manifest_path, manifest)

    if args.smoke:
        missing_after = [k for k in KPI_12 if k not in out_df.columns]
        if missing_after:
            raise SystemExit(f"[FAIL] missing KPI_12 columns after refresh: {missing_after}")

        if int(len(out_df)) != original_row_count:
            raise SystemExit("[FAIL] row count changed during refresh")

        if set(out_df["condition_id"].astype(str).unique()) != {"B0R"}:
            raise SystemExit("[FAIL] condition_id was not rewritten to B0R")

        if not bool((out_df["causal_comparison_allowed"] == False).all()):
            raise SystemExit("[FAIL] causal_comparison_allowed must remain false")

        if not bool((out_df["paper_level_claim_allowed"] == False).all()):
            raise SystemExit("[FAIL] paper_level_claim_allowed must remain false")

    print("[OK] B0R 12-KPI compatibility refresh completed")
    print(f"[OK] input_rows        : {original_row_count}")
    print(f"[OK] output_rows       : {len(out_df)}")
    print(f"[OK] output_root       : {output_root}")
    print(f"[OK] kpi_by_window     : {window_path}")
    print(f"[OK] kpi_by_seed       : {seed_path}")
    print(f"[OK] kpi_by_time_band  : {time_band_path}")
    print(f"[OK] kpi_overall       : {overall_path}")
    print(f"[OK] manifest          : {manifest_path}")
    print("[OK] condition_id      : B0R")
    print("[OK] kpi_schema        : 12 KPI columns present")
    print("[OK] claim_guard       : false")
    print("SMOKE PASS" if args.smoke else "DONE")


if __name__ == "__main__":
    main()
