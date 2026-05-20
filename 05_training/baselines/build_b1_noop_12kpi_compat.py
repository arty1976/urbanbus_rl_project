from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

KEY_CANDIDATES = [
    "condition_id",
    "seed",
    "window_id",
]

DEFAULT_KEEP_COLUMNS = [
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "service_date",
    "time_band",
    "evaluation_horizon_minutes",
]


def read_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
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


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_mean(series: pd.Series) -> Optional[float]:
    s = safe_numeric(series).dropna()
    if s.empty:
        return None
    return float(s.mean())


def safe_std(series: pd.Series) -> Optional[float]:
    s = safe_numeric(series).dropna()
    if len(s) < 2:
        return None
    return float(s.std(ddof=1))


def find_kpi_by_window(source_root: Path) -> Path:
    path = first_existing([
        source_root / "canonical_eval" / "kpi_by_window.parquet",
        source_root / "kpi_by_window.parquet",
    ])
    if path is None:
        raise FileNotFoundError(
            "B1 kpi_by_window.parquet not found. Expected one of: "
            f"{source_root / 'canonical_eval' / 'kpi_by_window.parquet'} or "
            f"{source_root / 'kpi_by_window.parquet'}"
        )
    return path


def read_window_rollups(source_root: Path) -> Tuple[Optional[pd.DataFrame], List[str]]:
    paths = sorted(source_root.rglob("window_rollup.parquet"))
    if not paths:
        return None, []

    frames: List[pd.DataFrame] = []
    for path in paths:
        df_one = pd.read_parquet(path).copy()
        df_one["compat_rollup_input_source_path"] = str(path)
        frames.append(df_one)

    return pd.concat(frames, ignore_index=True), [str(p) for p in paths]


def merge_rollup_candidates(window_df: pd.DataFrame, rollup_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if rollup_df is None:
        return window_df

    key_cols = [c for c in KEY_CANDIDATES if c in window_df.columns and c in rollup_df.columns]
    if len(key_cols) < 3:
        return window_df

    candidate_cols = [
        "passenger_demand_generated",
        "passenger_served_count",
        "passenger_service_rate",
        "passenger_wait_p95_seconds",
        "energy_proxy_per_passenger",
        "fleet_reduction_ratio",
        "energy_proxy_total",
        "compat_rollup_input_source_path",
    ]

    use_cols = key_cols + [c for c in candidate_cols if c in rollup_df.columns and c not in key_cols]
    if len(use_cols) == len(key_cols):
        return window_df

    # If multiple rollup rows exist per key, keep the first row because B1 canonical
    # window output is already the authoritative aggregation layer.
    rollup_small = rollup_df[use_cols].drop_duplicates(subset=key_cols, keep="first").copy()
    merged = window_df.merge(
        rollup_small,
        on=key_cols,
        how="left",
        suffixes=("", "__rollup"),
    )

    for col in candidate_cols:
        rollup_col = f"{col}__rollup"
        if rollup_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[rollup_col]
            else:
                merged[col] = merged[col].combine_first(merged[rollup_col])
            merged.drop(columns=[rollup_col], inplace=True)

    return merged


def init_compat_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for kpi in KPI_12:
        if kpi not in out.columns:
            out[kpi] = pd.NA
        out[f"{kpi}_source"] = pd.NA
        out[f"{kpi}_missing_reason"] = pd.NA

    return out


def mark_existing_or_missing(out: pd.DataFrame, kpi: str, source_label: str, missing_reason: str) -> None:
    s = out[kpi]
    mask = s.notna()
    out.loc[mask, f"{kpi}_source"] = source_label
    out.loc[~mask, f"{kpi}_source"] = "missing_null"
    out.loc[~mask, f"{kpi}_missing_reason"] = missing_reason


def derive_service_rate(out: pd.DataFrame) -> None:
    if "passenger_service_rate" not in out.columns:
        out["passenger_service_rate"] = pd.NA

    demand = safe_numeric(out["passenger_demand_generated"])
    served = safe_numeric(out["passenger_served_count"])

    can_derive = (
        out["passenger_service_rate"].isna()
        & demand.notna()
        & served.notna()
        & (demand > 0)
    )

    out.loc[can_derive, "passenger_service_rate"] = served[can_derive] / demand[can_derive]
    out.loc[can_derive, "passenger_service_rate_source"] = (
        "derived_from_passenger_served_count_over_passenger_demand_generated"
    )

    existing = out["passenger_service_rate"].notna() & out["passenger_service_rate_source"].isna()
    out.loc[existing, "passenger_service_rate_source"] = "preserved_from_source_artifact"

    missing = out["passenger_service_rate"].isna()
    out.loc[missing, "passenger_service_rate_source"] = "missing_null"
    out.loc[missing, "passenger_service_rate_missing_reason"] = (
        "requires passenger_demand_generated > 0 and passenger_served_count"
    )


def derive_energy_per_passenger(out: pd.DataFrame) -> None:
    if "energy_proxy_per_passenger" not in out.columns:
        out["energy_proxy_per_passenger"] = pd.NA

    energy = safe_numeric(out["energy_proxy"])
    if "energy_proxy_total" in out.columns:
        energy = energy.combine_first(safe_numeric(out["energy_proxy_total"]))

    served = safe_numeric(out["passenger_served_count"])

    can_derive = (
        out["energy_proxy_per_passenger"].isna()
        & energy.notna()
        & served.notna()
        & (served > 0)
    )

    out.loc[can_derive, "energy_proxy_per_passenger"] = energy[can_derive] / served[can_derive]
    out.loc[can_derive, "energy_proxy_per_passenger_source"] = (
        "derived_from_energy_proxy_over_passenger_served_count"
    )

    existing = (
        out["energy_proxy_per_passenger"].notna()
        & out["energy_proxy_per_passenger_source"].isna()
    )
    out.loc[existing, "energy_proxy_per_passenger_source"] = "preserved_from_source_artifact"

    missing = out["energy_proxy_per_passenger"].isna()
    out.loc[missing, "energy_proxy_per_passenger_source"] = "missing_null"
    out.loc[missing, "energy_proxy_per_passenger_missing_reason"] = (
        "requires energy_proxy or energy_proxy_total plus passenger_served_count > 0"
    )


def set_contract_fixed_fleet_reduction(out: pd.DataFrame) -> None:
    # B1_noop is the full-fleet no-op baseline. This is a contract value,
    # not a measured fleet optimization result.
    out["fleet_reduction_ratio"] = 0.0
    out["fleet_reduction_ratio_source"] = "contract_fixed_b1_noop_full_fleet"
    out["fleet_reduction_ratio_missing_reason"] = pd.NA


def apply_compat_policy(df: pd.DataFrame) -> pd.DataFrame:
    out = init_compat_columns(df)

    for kpi in BASE_6:
        mark_existing_or_missing(
            out,
            kpi,
            source_label="preserved_from_b1_noop_canonical_kpi",
            missing_reason="base KPI missing from B1 canonical artifact",
        )

    for kpi in ["passenger_demand_generated", "passenger_served_count"]:
        mark_existing_or_missing(
            out,
            kpi,
            source_label="preserved_from_source_artifact",
            missing_reason=(
                "not present in B1 source artifacts; do not impute from wait_passenger_count "
                "without explicit simulator contract"
            ),
        )

    derive_service_rate(out)

    # Do not create p95 from average wait. This must remain null unless source provides it.
    mark_existing_or_missing(
        out,
        "passenger_wait_p95_seconds",
        source_label="preserved_from_source_artifact",
        missing_reason=(
            "not derivable from aggregate average wait; requires passenger-level wait "
            "distribution or approved p95 proxy contract"
        ),
    )

    derive_energy_per_passenger(out)
    set_contract_fixed_fleet_reduction(out)

    out["compat_artifact_version"] = "b1_noop_12kpi_compat_v1"
    out["compat_baseline_id"] = "B1_noop_12kpi_compat"
    out["compat_source_baseline_id"] = "B1_noop"
    out["compat_schema"] = "KPI_12_SCHEMA_COMPAT"
    out["compat_is_observed_complete_12kpi"] = False
    out["causal_comparison_allowed"] = False
    out["paper_level_claim_allowed"] = False
    out["actual_results"] = False
    out["winner_selected"] = False
    out["trainable_reward_promoted"] = False

    return out


def aggregate_by_seed(window_df: pd.DataFrame) -> pd.DataFrame:
    if "seed" not in window_df.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    group_cols = ["seed"]
    if "condition_id" in window_df.columns:
        group_cols = ["condition_id", "seed"]

    for key, grp in window_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row: Dict[str, Any] = {
            "window_count": int(len(grp)),
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
        }

        for col_name, val in zip(group_cols, key):
            row[col_name] = val.item() if hasattr(val, "item") else val

        for kpi in KPI_12:
            s = safe_numeric(grp[kpi])
            row[f"{kpi}_mean"] = safe_mean(s)
            row[f"{kpi}_std"] = safe_std(s)
            row[f"{kpi}_valid_window_count"] = int(s.notna().sum())
            row[f"{kpi}_null_window_count"] = int(s.isna().sum())

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_by_time_band(window_df: pd.DataFrame) -> pd.DataFrame:
    if "time_band" not in window_df.columns:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    group_cols = ["time_band"]
    if "condition_id" in window_df.columns:
        group_cols = ["condition_id", "time_band"]
    if "seed" in window_df.columns:
        group_cols.append("seed")

    for key, grp in window_df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        row: Dict[str, Any] = {
            "window_count": int(len(grp)),
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
        }

        for col_name, val in zip(group_cols, key):
            row[col_name] = val.item() if hasattr(val, "item") else val

        for kpi in KPI_12:
            s = safe_numeric(grp[kpi])
            row[f"{kpi}_mean"] = safe_mean(s)
            row[f"{kpi}_std"] = safe_std(s)
            row[f"{kpi}_valid_window_count"] = int(s.notna().sum())
            row[f"{kpi}_null_window_count"] = int(s.isna().sum())

        rows.append(row)

    return pd.DataFrame(rows)


def build_overall(window_df: pd.DataFrame) -> Dict[str, Any]:
    kpis: Dict[str, Any] = {}

    for kpi in KPI_12:
        s = safe_numeric(window_df[kpi])
        kpis[kpi] = {
            "mean": safe_mean(s),
            "std": safe_std(s),
            "valid_window_count": int(s.notna().sum()),
            "null_window_count": int(s.isna().sum()),
            "source_counts": (
                window_df[f"{kpi}_source"].astype(str).value_counts(dropna=False).to_dict()
                if f"{kpi}_source" in window_df.columns
                else {}
            ),
        }

    return {
        "artifact_version": "b1_noop_12kpi_compat_v1",
        "mode": "schema_compatibility_artifact",
        "baseline_id": "B1_noop_12kpi_compat",
        "source_baseline_id": "B1_noop",
        "condition_ids": (
            sorted(window_df["condition_id"].astype(str).dropna().unique().tolist())
            if "condition_id" in window_df.columns
            else ["B1"]
        ),
        "n_windows_total": int(len(window_df)),
        "kpi_schema": KPI_12,
        "kpis": kpis,
        "claim_guards": {
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
        },
        "compatibility_statement": (
            "This artifact makes B1_noop readable under the 12-KPI schema. "
            "It is not a complete observed 12-KPI baseline and must not be used "
            "as causal or paper-level performance evidence."
        ),
    }


def build_missing_reason_report(window_df: pd.DataFrame, manifest: Dict[str, Any]) -> str:
    lines: List[str] = []

    lines.append("# B1_noop 12-KPI Compatibility Missing Reason Report")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append("- artifact_status: `SCHEMA_COMPAT_CREATED_WITH_GUARDS`")
    lines.append("- baseline_id: `B1_noop_12kpi_compat`")
    lines.append("- source_baseline_id: `B1_noop`")
    lines.append("- causal_comparison_allowed: `False`")
    lines.append("- paper_level_claim_allowed: `False`")
    lines.append("")
    lines.append("## Important Interpretation")
    lines.append("")
    lines.append("This artifact aligns B1_noop with the 12-KPI column schema.")
    lines.append("")
    lines.append("It does not turn missing KPI fields into observed measurements.")
    lines.append("")
    lines.append("## KPI Source and Missing Summary")
    lines.append("")
    lines.append("| KPI | valid_windows | null_windows | source_counts |")
    lines.append("|---|---:|---:|---|")

    for kpi in KPI_12:
        s = window_df[kpi]
        source_counts = (
            window_df[f"{kpi}_source"].astype(str).value_counts(dropna=False).to_dict()
            if f"{kpi}_source" in window_df.columns
            else {}
        )
        lines.append(
            f"| `{kpi}` | {int(s.notna().sum())} | {int(s.isna().sum())} | `{source_counts}` |"
        )

    lines.append("")
    lines.append("## Non-claim Guards")
    lines.append("")
    for key, value in manifest["claim_guards"].items():
        lines.append(f"- {key}: `{value}`")

    lines.append("")
    lines.append("## Files")
    lines.append("")
    for key, value in manifest["outputs"].items():
        lines.append(f"- {key}: `{value}`")

    return "\n".join(lines)


def validate_compat(window_df: pd.DataFrame) -> List[str]:
    hard_failures: List[str] = []

    missing_cols = [k for k in KPI_12 if k not in window_df.columns]
    if missing_cols:
        hard_failures.append(f"missing KPI columns: {missing_cols}")

    for kpi in KPI_12:
        for suffix in ("_source", "_missing_reason"):
            col = f"{kpi}{suffix}"
            if col not in window_df.columns:
                hard_failures.append(f"missing provenance column: {col}")

    if "fleet_reduction_ratio" in window_df.columns:
        vals = safe_numeric(window_df["fleet_reduction_ratio"]).dropna()
        if vals.empty or not bool((vals == 0.0).all()):
            hard_failures.append("fleet_reduction_ratio must be contract-fixed 0.0 for B1_noop_12kpi_compat")

    if "fleet_reduction_ratio_source" in window_df.columns:
        bad = window_df["fleet_reduction_ratio_source"].astype(str) != "contract_fixed_b1_noop_full_fleet"
        if bool(bad.any()):
            hard_failures.append("fleet_reduction_ratio_source must be contract_fixed_b1_noop_full_fleet")

    if "passenger_wait_p95_seconds_source" in window_df.columns:
        # p95 is allowed only if preserved from a source artifact. It must not be fabricated.
        fabricated = window_df["passenger_wait_p95_seconds_source"].astype(str).str.contains(
            "derived_from_avg_wait|imputed|filled_zero",
            case=False,
            regex=True,
            na=False,
        )
        if bool(fabricated.any()):
            hard_failures.append("passenger_wait_p95_seconds must not be imputed from avg_wait or filled with zero")

    for guard in [
        "causal_comparison_allowed",
        "paper_level_claim_allowed",
        "actual_results",
        "winner_selected",
        "trainable_reward_promoted",
    ]:
        if guard in window_df.columns and bool(window_df[guard].fillna(False).astype(bool).any()):
            hard_failures.append(f"{guard} must remain false")

    return hard_failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="artifacts/baseline_v1/B1_noop")
    parser.add_argument("--output-root", default="artifacts/baseline_v1/B1_noop_12kpi_compat")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"[STOP] output_root already exists and is not empty: {output_root}\n"
            "Use --overwrite only if you intentionally want to regenerate this artifact."
        )

    output_root.mkdir(parents=True, exist_ok=True)

    kpi_by_window_path = find_kpi_by_window(source_root)
    source_window_df = pd.read_parquet(kpi_by_window_path).copy()

    rollup_df, rollup_paths = read_window_rollups(source_root)
    merged = merge_rollup_candidates(source_window_df, rollup_df)
    compat_window_df = apply_compat_policy(merged)

    hard_failures = validate_compat(compat_window_df)

    seed_df = aggregate_by_seed(compat_window_df)
    time_band_df = aggregate_by_time_band(compat_window_df)
    overall = build_overall(compat_window_df)

    window_path = output_root / "kpi_by_window.parquet"
    seed_path = output_root / "kpi_by_seed.parquet"
    time_band_path = output_root / "kpi_by_time_band.parquet"
    overall_path = output_root / "kpi_overall.json"
    manifest_path = output_root / "compatibility_manifest.json"
    report_path = output_root / "missing_reason_report.md"

    compat_window_df.to_parquet(window_path, index=False)
    seed_df.to_parquet(seed_path, index=False)
    time_band_df.to_parquet(time_band_path, index=False)
    dump_json(overall_path, overall)

    manifest: Dict[str, Any] = {
        "artifact_version": "b1_noop_12kpi_compat_v1",
        "baseline_id": "B1_noop_12kpi_compat",
        "source_baseline_id": "B1_noop",
        "artifact_status": "SCHEMA_COMPAT_CREATED_WITH_GUARDS" if not hard_failures else "BLOCKED",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "inputs": {
            "kpi_by_window": str(kpi_by_window_path),
            "window_rollups": rollup_paths,
        },
        "outputs": {
            "kpi_by_window": str(window_path),
            "kpi_by_seed": str(seed_path),
            "kpi_by_time_band": str(time_band_path),
            "kpi_overall": str(overall_path),
            "compatibility_manifest": str(manifest_path),
            "missing_reason_report": str(report_path),
        },
        "row_counts": {
            "kpi_by_window": int(len(compat_window_df)),
            "kpi_by_seed": int(len(seed_df)),
            "kpi_by_time_band": int(len(time_band_df)),
        },
        "kpi_schema": KPI_12,
        "claim_guards": {
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "actual_results": False,
            "winner_selected": False,
            "trainable_reward_promoted": False,
        },
        "compatibility_rules": {
            "base_6_kpis": "preserve from B1_noop canonical KPI artifact",
            "passenger_demand_generated": "preserve if present, else null",
            "passenger_served_count": "preserve if present, else null",
            "passenger_service_rate": "derive only if demand and served counts are present",
            "passenger_wait_p95_seconds": "preserve only if present, else null; never fill with zero",
            "energy_proxy_per_passenger": "derive only if energy and served counts are present",
            "fleet_reduction_ratio": "contract-fixed 0.0 for B1_noop full-fleet no-op baseline",
        },
        "hard_failures": hard_failures,
        "warnings": [
            "This is a 12-KPI schema-compatible artifact, not a complete observed 12-KPI baseline.",
            "Do not use this artifact as causal or paper-level performance evidence.",
        ],
    }

    dump_json(manifest_path, manifest)
    report_path.write_text(build_missing_reason_report(compat_window_df, manifest), encoding="utf-8")

    print("[OK] Step B1-GAP-2 B1_noop_12kpi_compat artifact generation completed")
    print(f"[OK] artifact_status : {manifest['artifact_status']}")
    print(f"[OK] hard_failures   : {len(hard_failures)}")
    print(f"[OK] window_rows     : {len(compat_window_df)}")
    print(f"[OK] output_root     : {output_root}")
    print(f"[OK] manifest        : {manifest_path}")
    print(f"[OK] report          : {report_path}")

    if hard_failures:
        for failure in hard_failures:
            print(f"[FAIL] {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
