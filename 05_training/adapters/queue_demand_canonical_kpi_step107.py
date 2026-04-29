#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 107 — queue/demand canonical KPI integration scaffold.

Connects Step 106 queue/demand proxy rollups to the existing canonical
KPI (Key Performance Indicator=핵심 성과 지표) aggregator while preserving
queue/demand extension columns in a post-canonical window output.

Safety: no DB (Database=데이터베이스) write, no tensor DB overwrite, no extra
API (Application Programming Interface=응용 프로그램 인터페이스) calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

ARTIFACT_VERSION = "queue_demand_canonical_kpi_step107_v1"
CANONICAL_MODE = "official_rollup"
DEFAULT_INPUT_ROOT = Path("artifacts/daegu_bis_api_audit/minimal_queue_demand_step106")
DEFAULT_OUTPUT_ROOT = Path("artifacts/daegu_bis_api_audit/queue_demand_canonical_kpi_step107")
DEFAULT_AGGREGATOR = Path("05_training/evaluation/canonical_kpi_aggregator.py")

EXPECTED_SHARED_KPIS = [
    "cv_headway", "avg_wait_seconds", "bunching_rate",
    "on_time_rate", "intervention_rate", "energy_proxy",
]
QUEUE_DEMAND_EXTENDED_KPIS = [
    "passenger_demand_generated", "passenger_served_count", "passenger_service_rate",
    "passenger_wait_p95_seconds", "energy_proxy_per_passenger", "fleet_reduction_ratio",
]
CANONICAL_12_KPIS = [*EXPECTED_SHARED_KPIS, *QUEUE_DEMAND_EXTENDED_KPIS]
QUEUE_DEMAND_PRESERVE_COLUMNS = [
    "artifact_version", "queue_demand_model_version",
    *QUEUE_DEMAND_EXTENDED_KPIS,
    "passenger_left_behind_count", "max_queue_depth", "avg_queue_depth",
    "avg_queue_pressure", "unmet_demand_rate", "queue_event_count",
    "queue_demand_observed", "queue_demand_proxy", "actual_passenger_wait_observed",
    "actual_headway_observed", "actual_arrival_departure_time_observed",
    "actual_dwell_observed", "paper_level_claim_allowed",
    "causal_performance_claim_allowed", "scaffold_only", "performance_claim_allowed",
    "actual_policy_claim_ready", "causal_policy_claim_ready",
]
REQUIRED_STEP106_ROLLUP_COLUMNS = [
    "condition_id", "seed", "window_id", "state_ts", "service_date", "time_band",
    "evaluation_horizon_minutes", "headway_mean_seconds", "headway_std_seconds",
    "headway_sample_count", "bunching_event_count", "headway_event_count",
    "wait_total_passenger_seconds", "wait_passenger_count", "ontime_event_count",
    "schedulable_arrival_count", "intervention_count", "decision_step_count",
    "energy_proxy_total", "source_mode", *QUEUE_DEMAND_EXTENDED_KPIS,
]
CLAIM_GUARDS: Dict[str, bool] = {
    "db_write_performed": False,
    "tensor_db_overwrite_performed": False,
    "additional_api_calls_performed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "queue_demand_observed": False,
    "queue_demand_proxy": True,
    "actual_passenger_wait_observed": False,
    "actual_headway_observed": False,
    "actual_arrival_departure_time_observed": False,
    "actual_dwell_observed": False,
}

@dataclass(frozen=True)
class Step107Paths:
    input_root: Path
    output_root: Path
    canonical_aggregator: Path

    @property
    def source_window_rollup(self) -> Path:
        return self.input_root / "window_rollup_queue_demand.parquet"
    @property
    def staging_root(self) -> Path:
        return self.output_root / "canonical_input_staging"
    @property
    def staged_window_rollup(self) -> Path:
        return self.staging_root / "window_rollup.parquet"
    @property
    def contract_path(self) -> Path:
        return self.output_root / "step107_canonical_aggregator_contract.json"
    @property
    def canonical_output_root(self) -> Path:
        return self.output_root / "canonical_eval"
    @property
    def preserved_kpi_by_window_path(self) -> Path:
        return self.canonical_output_root / "kpi_by_window_queue_demand_preserved.parquet"
    @property
    def preserved_kpi_by_window_csv_path(self) -> Path:
        return self.canonical_output_root / "kpi_by_window_queue_demand_preserved.csv"
    @property
    def extension_summary_path(self) -> Path:
        return self.output_root / "queue_demand_kpi_extension_summary_step107.json"
    @property
    def readiness_csv_path(self) -> Path:
        return self.output_root / "queue_demand_canonical_readiness_step107.csv"
    @property
    def manifest_path(self) -> Path:
        return self.output_root / "queue_demand_canonical_kpi_step107_manifest.json"
    @property
    def report_path(self) -> Path:
        return self.output_root / "queue_demand_canonical_kpi_step107_report.md"

def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")

def table_exists(path: Path) -> bool:
    return path.exists() or path.with_suffix(".csv").exists()

def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        alt = path.with_suffix(".csv") if path.suffix.lower() == ".parquet" else path.with_suffix(".parquet")
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"input table not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table extension: {path}")

def write_table_with_csv_fallback(df: pd.DataFrame, parquet_path: Path, csv_path: Optional[Path] = None) -> Dict[str, Any]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = csv_path or parquet_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_written = False
    parquet_error = None
    try:
        df.to_parquet(parquet_path, index=False)
        parquet_written = True
    except Exception as exc:  # optional parquet engine may be absent in lightweight tests
        parquet_error = str(exc)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return {"parquet_path": str(parquet_path), "csv_path": str(csv_path), "parquet_written": parquet_written, "parquet_error": parquet_error}

def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

def normalize_source_mode(value: Any) -> str:
    text = str(value)
    lower = text.lower()
    if "noncausal" in lower and "non_causal" not in lower and "non-causal" not in lower:
        return text.replace("noncausal", "non_causal").replace("NONCAUSAL", "NON_CAUSAL")
    return text

def validate_step106_input(df: pd.DataFrame) -> Dict[str, Any]:
    missing = [c for c in REQUIRED_STEP106_ROLLUP_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Step 106 window_rollup_queue_demand missing required columns: {missing}")
    if df.empty:
        raise RuntimeError("Step 106 window_rollup_queue_demand is empty")
    numeric_required = [
        "seed", "evaluation_horizon_minutes", "headway_mean_seconds", "headway_std_seconds",
        "headway_sample_count", "bunching_event_count", "headway_event_count",
        "wait_total_passenger_seconds", "wait_passenger_count", "ontime_event_count",
        "schedulable_arrival_count", "intervention_count", "decision_step_count",
        "energy_proxy_total", *QUEUE_DEMAND_EXTENDED_KPIS,
    ]
    bad = [c for c in numeric_required if pd.to_numeric(df[c], errors="coerce").isna().any()]
    if bad:
        raise RuntimeError(f"Step 106 window_rollup has non-numeric values in: {bad}")
    for col in ["passenger_service_rate", "fleet_reduction_ratio"]:
        s = pd.to_numeric(df[col], errors="coerce")
        if bool((s < 0).any()) or bool((s > 1).any()):
            raise RuntimeError(f"Step 106 {col} must be within [0, 1]")
    if "queue_demand_proxy" in df.columns and not bool(df["queue_demand_proxy"].astype(bool).all()):
        raise RuntimeError("Step 106 queue_demand_proxy must be true for all rows")
    if "queue_demand_observed" in df.columns and bool(df["queue_demand_observed"].astype(bool).any()):
        raise RuntimeError("Step 106 queue_demand_observed must be false for all rows")
    return {
        "row_count": int(len(df)),
        "condition_ids": sorted(df["condition_id"].astype(str).str.upper().unique().tolist()),
        "seed_values": sorted([int(x) for x in pd.to_numeric(df["seed"], errors="raise").unique().tolist()]),
        "window_count": int(df["window_id"].nunique()),
        "time_bands": sorted(df["time_band"].astype(str).str.lower().unique().tolist()),
        "source_modes": sorted(df["source_mode"].astype(str).unique().tolist()),
        "demand_total": int(pd.to_numeric(df["passenger_demand_generated"], errors="coerce").fillna(0).sum()),
        "served_total": int(pd.to_numeric(df["passenger_served_count"], errors="coerce").fillna(0).sum()),
    }

def build_staged_input(paths: Step107Paths) -> Dict[str, Any]:
    df = read_table(paths.source_window_rollup).copy()
    summary = validate_step106_input(df)
    df["original_source_mode_step106"] = df["source_mode"].astype(str)
    df["source_mode"] = df["source_mode"].map(normalize_source_mode)
    normalized = sorted(df["source_mode"].astype(str).unique().tolist())
    write_info = write_table_with_csv_fallback(df, paths.staged_window_rollup, paths.staging_root / "window_rollup.csv")
    return {
        "input_summary": summary,
        "staged_window_rollup": str(paths.staged_window_rollup),
        "staged_row_count": int(len(df)),
        "staged_source_modes": normalized,
        "source_mode_normalized": summary["source_modes"] != normalized,
        "staged_write": write_info,
    }

def build_canonical_contract(paths: Step107Paths, input_summary: Mapping[str, Any]) -> Dict[str, Any]:
    contract = {
        "artifact_version": "step107_queue_demand_canonical_aggregator_contract_v1",
        "condition_id": "QUEUE_DEMAND_STEP107",
        "condition_name": "queue_demand_scaffold_canonical_integration",
        "qwen_train": False,
        "qwen_inference": False,
        "evaluation_horizon_minutes": 30,
        "time_bands": ["peak", "offpeak", "night"],
        "shared_kpis": list(EXPECTED_SHARED_KPIS),
        "extended_kpis_preserved_by_wrapper": list(QUEUE_DEMAND_EXTENDED_KPIS),
        "fairness_constraints": {"same_initial_state": True, "same_exogenous_events": True, "same_eval_window": True},
        "input_source": {
            "step106_input_root": str(paths.input_root),
            "staged_input_root": str(paths.staging_root),
            "input_row_count": int(input_summary.get("row_count", 0)),
            "condition_ids": list(input_summary.get("condition_ids", [])),
            "seeds": list(input_summary.get("seed_values", [])),
            "window_count": int(input_summary.get("window_count", 0)),
        },
        "claim_guards": dict(CLAIM_GUARDS),
        "note": "Stable six KPIs are computed by canonical aggregator; queue/demand extension columns are preserved by Step 107 wrapper.",
    }
    dump_json(paths.contract_path, contract)
    return contract

def run_canonical_aggregator(paths: Step107Paths, *, smoke: bool) -> Dict[str, Any]:
    if not paths.canonical_aggregator.exists():
        raise RuntimeError(f"canonical_kpi_aggregator.py not found: {paths.canonical_aggregator}")
    cmd = [sys.executable, str(paths.canonical_aggregator), "--mode", CANONICAL_MODE, "--contract", str(paths.contract_path), "--input-root", str(paths.staging_root), "--output-root", str(paths.canonical_output_root)]
    if smoke:
        cmd.append("--smoke")
    result = subprocess.run(cmd, capture_output=True, text=True)
    payload = {"command": cmd, "returncode": int(result.returncode), "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode != 0:
        raise RuntimeError(f"canonical KPI aggregator failed\ncmd={cmd}\nstdout={result.stdout}\nstderr={result.stderr}")
    return payload

def _merge_key_normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["condition_id"] = out["condition_id"].astype(str).str.upper()
    out["seed"] = pd.to_numeric(out["seed"], errors="raise").astype(int)
    out["window_id"] = out["window_id"].astype(str)
    return out

def preserve_queue_demand_columns(paths: Step107Paths) -> Dict[str, Any]:
    canonical_path = paths.canonical_output_root / "kpi_by_window.parquet"
    if not table_exists(canonical_path):
        raise RuntimeError(f"canonical kpi_by_window parquet/csv missing: {canonical_path}")

    canonical_df = _merge_key_normalize(read_table(canonical_path))
    staged_df = _merge_key_normalize(read_table(paths.staged_window_rollup))
    keys = ["condition_id", "seed", "window_id"]

    # Preserve queue/demand columns from Step 106, but avoid merge suffix collisions.
    # Some versions of canonical_kpi_aggregator.py may already preserve optional
    # extended KPI columns. If we merge the same names again, pandas creates
    # *_x / *_y columns and the canonical plain column name disappears.
    preserve_cols = [
        c for c in QUEUE_DEMAND_PRESERVE_COLUMNS
        if c in staged_df.columns and c not in keys
    ]

    preserve_df = staged_df[keys + preserve_cols].drop_duplicates(keys)
    if len(preserve_df) != len(staged_df):
        raise RuntimeError("Step 106 staged input has duplicate canonical keys")

    overlapping_preserve_cols = [
        c for c in preserve_cols
        if c in canonical_df.columns and c not in keys
    ]
    merge_preserve_cols = [
        c for c in preserve_cols
        if c not in canonical_df.columns and c not in keys
    ]

    merged = canonical_df.merge(
        preserve_df[keys + merge_preserve_cols],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # For columns already present in canonical_df, fill nulls from staged_df.
    # This covers aggregators that partially preserved optional extended columns.
    if overlapping_preserve_cols:
        overlap_df = preserve_df[keys + overlapping_preserve_cols]
        merged = merged.merge(
            overlap_df,
            on=keys,
            how="left",
            validate="one_to_one",
            suffixes=("", "__step106"),
        )
        for col in overlapping_preserve_cols:
            staged_col = f"{col}__step106"
            if staged_col in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[col].combine_first(merged[staged_col])
                    merged = merged.drop(columns=[staged_col])
                else:
                    merged = merged.rename(columns={staged_col: col})

    # Safety cleanup in case an older merge path already produced suffix columns.
    for col in QUEUE_DEMAND_EXTENDED_KPIS:
        candidates = [col, f"{col}_x", f"{col}_y", f"{col}__step106"]
        existing = [c for c in candidates if c in merged.columns]
        if col not in merged.columns and existing:
            merged[col] = merged[existing[0]]
        for c in existing:
            if c != col and c in merged.columns:
                merged[col] = merged[col].combine_first(merged[c])
                merged = merged.drop(columns=[c])

    missing = [c for c in QUEUE_DEMAND_EXTENDED_KPIS if c not in merged.columns]
    if missing:
        raise RuntimeError(f"preserved kpi_by_window missing extended KPI columns: {missing}")

    missing_values = {
        c: int(merged[c].isna().sum())
        for c in QUEUE_DEMAND_EXTENDED_KPIS
    }
    if any(v > 0 for v in missing_values.values()):
        raise RuntimeError(f"extended KPI columns have missing values after merge: {missing_values}")

    if "causal_comparison_allowed" not in merged.columns:
        raise RuntimeError("preserved canonical output missing causal_comparison_allowed")

    if bool(merged["causal_comparison_allowed"].all()):
        raise RuntimeError("preserved canonical output incorrectly allows causal comparison")

    for col in ["causal_performance_claim_allowed", "paper_level_claim_allowed", "queue_demand_observed"]:
        if col in merged.columns and bool(merged[col].astype(bool).any()):
            raise RuntimeError(f"{col} must remain false")

    if "queue_demand_proxy" in merged.columns and not bool(merged["queue_demand_proxy"].astype(bool).all()):
        raise RuntimeError("queue_demand_proxy must remain true")

    write_info = write_table_with_csv_fallback(
        merged,
        paths.preserved_kpi_by_window_path,
        paths.preserved_kpi_by_window_csv_path,
    )

    return {
        "preserved_output_files": {
            "kpi_by_window_queue_demand_preserved": str(paths.preserved_kpi_by_window_path),
            "kpi_by_window_queue_demand_preserved_csv": str(paths.preserved_kpi_by_window_csv_path),
        },
        "preserved_write": write_info,
        "preserved_row_count": int(len(merged)),
        "preserved_columns": preserve_cols,
        "overlapping_preserve_columns": overlapping_preserve_cols,
        "merge_preserve_columns": merge_preserve_cols,
        "extended_kpis_present": [c for c in QUEUE_DEMAND_EXTENDED_KPIS if c in merged.columns],
        "extended_kpi_missing_counts": missing_values,
        "causal_comparison_allowed": False,
        "condition_ids": sorted(merged["condition_id"].astype(str).unique().tolist()),
        "seed_values": sorted([int(x) for x in merged["seed"].unique().tolist()]),
    }

def build_extension_summary(paths: Step107Paths) -> Dict[str, Any]:
    df = read_table(paths.preserved_kpi_by_window_path)
    kpis: Dict[str, Dict[str, Any]] = {}
    for kpi in CANONICAL_12_KPIS:
        if kpi not in df.columns:
            kpis[kpi] = {"present": False}
            continue
        s = pd.to_numeric(df[kpi], errors="coerce")
        valid = s.dropna()
        kpis[kpi] = {"present": True, "valid_window_count": int(s.notna().sum()), "null_window_count": int(s.isna().sum()), "mean": None if valid.empty else float(valid.mean()), "min": None if valid.empty else float(valid.min()), "max": None if valid.empty else float(valid.max())}
    payload = {
        "artifact_version": ARTIFACT_VERSION,
        "summary_type": "queue_demand_12_kpi_extension_summary",
        "source_file": str(paths.preserved_kpi_by_window_path),
        "row_count": int(len(df)),
        "canonical_12_kpis": list(CANONICAL_12_KPIS),
        "all_12_kpis_present": all(kpis[k].get("present", False) for k in CANONICAL_12_KPIS),
        "all_12_kpis_have_valid_rows": all(kpis[k].get("valid_window_count", 0) > 0 for k in CANONICAL_12_KPIS),
        "causal_comparison_allowed": bool(df["causal_comparison_allowed"].all()) if "causal_comparison_allowed" in df.columns else None,
        "claim_guards": dict(CLAIM_GUARDS),
        "kpis": kpis,
    }
    dump_json(paths.extension_summary_path, payload)
    return payload

def read_canonical_outputs(paths: Step107Paths) -> Dict[str, Any]:
    expected_files = {
        "official_rollup_input_validation": paths.canonical_output_root / "official_rollup_input_validation.json",
        "kpi_by_window": paths.canonical_output_root / "kpi_by_window.parquet",
        "kpi_by_seed": paths.canonical_output_root / "kpi_by_seed.parquet",
        "kpi_by_time_band": paths.canonical_output_root / "kpi_by_time_band.parquet",
        "kpi_overall": paths.canonical_output_root / "kpi_overall.json",
        "aggregation_manifest": paths.canonical_output_root / "aggregation_manifest.json",
        "kpi_by_window_queue_demand_preserved": paths.preserved_kpi_by_window_path,
        "queue_demand_kpi_extension_summary": paths.extension_summary_path,
    }
    missing = [str(p) for p in expected_files.values() if p.suffix.lower() != ".parquet" and not p.exists()]
    missing += [str(p) for p in expected_files.values() if p.suffix.lower() == ".parquet" and not table_exists(p)]
    if missing:
        raise RuntimeError(f"Step 107 output files missing: {missing}")
    window_df = read_table(expected_files["kpi_by_window"])
    preserved_df = read_table(expected_files["kpi_by_window_queue_demand_preserved"])
    seed_df = read_table(expected_files["kpi_by_seed"])
    time_band_df = read_table(expected_files["kpi_by_time_band"])
    overall = load_json_any_encoding(expected_files["kpi_overall"])
    agg_manifest = load_json_any_encoding(expected_files["aggregation_manifest"])
    ext_summary = load_json_any_encoding(expected_files["queue_demand_kpi_extension_summary"])
    if len(window_df) != len(preserved_df):
        raise RuntimeError("preserved kpi_by_window row count must match canonical kpi_by_window")
    missing_base = [c for c in EXPECTED_SHARED_KPIS if c not in window_df.columns]
    missing_12 = [c for c in CANONICAL_12_KPIS if c not in preserved_df.columns]
    if missing_base or missing_12:
        raise RuntimeError(f"missing KPIs: base={missing_base}, preserved12={missing_12}")
    causal_allowed = bool(preserved_df["causal_comparison_allowed"].all())
    if causal_allowed:
        raise RuntimeError("Step 107 preserved output incorrectly allows causal comparison")
    return {
        "output_files": {k: str(v) for k, v in expected_files.items()},
        "row_counts": {"kpi_by_window": int(len(window_df)), "kpi_by_window_queue_demand_preserved": int(len(preserved_df)), "kpi_by_seed": int(len(seed_df)), "kpi_by_time_band": int(len(time_band_df))},
        "condition_ids": sorted(preserved_df["condition_id"].astype(str).unique().tolist()),
        "seed_values": sorted([int(x) for x in preserved_df["seed"].unique().tolist()]),
        "causal_comparison_allowed": causal_allowed,
        "aggregation_manifest_causal_allowed": bool(agg_manifest.get("causal_comparison_allowed", False)),
        "aggregation_warning_codes": [w.get("code") for w in agg_manifest.get("warnings", [])],
        "kpi_overall_causal_allowed": overall.get("causal_comparison_allowed"),
        "all_12_kpis_present": bool(ext_summary.get("all_12_kpis_present")),
        "all_12_kpis_have_valid_rows": bool(ext_summary.get("all_12_kpis_have_valid_rows")),
    }

def build_readiness_rows(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cs, ps, es, guards = manifest["canonical_summary"], manifest["preservation_summary"], manifest["extension_summary"], manifest["claim_guards"]
    return [
        {"check_name": "canonical_aggregator_passed", "passed": manifest["audit_status"] == "PASS", "observed_value": manifest["audit_status"], "note": "official_rollup executed"},
        {"check_name": "canonical_window_rows_positive", "passed": cs["row_counts"]["kpi_by_window"] > 0, "observed_value": cs["row_counts"]["kpi_by_window"], "note": "canonical rows exist"},
        {"check_name": "preserved_window_rows_match", "passed": cs["row_counts"]["kpi_by_window"] == ps["preserved_row_count"], "observed_value": ps["preserved_row_count"], "note": "preserved rows match canonical rows"},
        {"check_name": "all_12_kpis_present", "passed": bool(es["all_12_kpis_present"]), "observed_value": es["all_12_kpis_present"], "note": "base six + queue/demand six present"},
        {"check_name": "all_12_kpis_have_valid_rows", "passed": bool(es["all_12_kpis_have_valid_rows"]), "observed_value": es["all_12_kpis_have_valid_rows"], "note": "all 12 KPIs have valid rows"},
        {"check_name": "causal_comparison_blocked", "passed": cs["causal_comparison_allowed"] is False, "observed_value": cs["causal_comparison_allowed"], "note": "causal comparison remains blocked"},
        {"check_name": "paper_claim_blocked", "passed": guards["paper_level_claim_allowed"] is False, "observed_value": guards["paper_level_claim_allowed"], "note": "paper-level claims remain blocked"},
        {"check_name": "queue_demand_marked_proxy", "passed": guards["queue_demand_proxy"] is True and guards["queue_demand_observed"] is False, "observed_value": f"proxy={guards['queue_demand_proxy']}, observed={guards['queue_demand_observed']}", "note": "queue/demand remains proxy"},
    ]

def build_markdown_report(manifest: Mapping[str, Any]) -> str:
    cs, ps, es = manifest["canonical_summary"], manifest["preservation_summary"], manifest["extension_summary"]
    return f"""# Step 107 — Queue/Demand Canonical KPI Integration

## Status

- audit_status: `{manifest['audit_status']}`
- integration_status: `{manifest['integration_status']}`
- artifact_version: `{manifest['artifact_version']}`

## Canonical outputs

- canonical_output_root: `{manifest['canonical_output_root']}`
- kpi_by_window rows: `{cs['row_counts']['kpi_by_window']}`
- kpi_by_seed rows: `{cs['row_counts']['kpi_by_seed']}`
- kpi_by_time_band rows: `{cs['row_counts']['kpi_by_time_band']}`
- causal_comparison_allowed: `{cs['causal_comparison_allowed']}`

## Queue/demand preservation

- preserved file: `{ps['preserved_output_files']['kpi_by_window_queue_demand_preserved']}`
- preserved row count: `{ps['preserved_row_count']}`
- extended_kpis_present: `{ps['extended_kpis_present']}`
- all_12_kpis_present: `{es['all_12_kpis_present']}`
- all_12_kpis_have_valid_rows: `{es['all_12_kpis_have_valid_rows']}`

## Claim guards

- paper_level_claim_allowed: `{manifest['claim_guards']['paper_level_claim_allowed']}`
- causal_performance_claim_allowed: `{manifest['claim_guards']['causal_performance_claim_allowed']}`
- queue_demand_observed: `{manifest['claim_guards']['queue_demand_observed']}`
- queue_demand_proxy: `{manifest['claim_guards']['queue_demand_proxy']}`
- actual_passenger_wait_observed: `{manifest['claim_guards']['actual_passenger_wait_observed']}`
- actual_headway_observed: `{manifest['claim_guards']['actual_headway_observed']}`
- actual_arrival_departure_time_observed: `{manifest['claim_guards']['actual_arrival_departure_time_observed']}`
- actual_dwell_observed: `{manifest['claim_guards']['actual_dwell_observed']}`

## Interpretation

Step 107 confirms that Step 106 queue/demand proxy rollups can pass through
canonical KPI official_rollup while preserving queue/demand extension columns in
`kpi_by_window_queue_demand_preserved.parquet`.

This remains a scaffold/proxy validation. It does not validate real passenger
demand, real passenger wait, actual headway, actual arrival/departure time, or
actual dwell time. It must not be used for paper-level performance claims.
"""

def run_step107(paths: Step107Paths, *, smoke: bool = True, clean_output: bool = False) -> Dict[str, Any]:
    if clean_output and paths.output_root.exists():
        shutil.rmtree(paths.output_root)
    paths.output_root.mkdir(parents=True, exist_ok=True)
    staged = build_staged_input(paths)
    contract = build_canonical_contract(paths, staged["input_summary"])
    canonical_run = run_canonical_aggregator(paths, smoke=smoke)
    preservation_summary = preserve_queue_demand_columns(paths)
    extension_summary = build_extension_summary(paths)
    canonical_summary = read_canonical_outputs(paths)
    if not canonical_summary["all_12_kpis_present"] or canonical_summary["causal_comparison_allowed"]:
        raise RuntimeError("Step 107 readiness guard failed")
    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": "PASS",
        "integration_status": "READY_FOR_STEP108_QUEUE_DEMAND_REWARD_WIRING",
        "input_root": str(paths.input_root),
        "output_root": str(paths.output_root),
        "canonical_aggregator": str(paths.canonical_aggregator),
        "canonical_output_root": str(paths.canonical_output_root),
        "contract_path": str(paths.contract_path),
        "staged_input": {"staged_input_root": str(paths.staging_root), **staged},
        "canonical_contract": contract,
        "canonical_run": canonical_run,
        "preservation_summary": preservation_summary,
        "extension_summary": {"path": str(paths.extension_summary_path), "all_12_kpis_present": bool(extension_summary["all_12_kpis_present"]), "all_12_kpis_have_valid_rows": bool(extension_summary["all_12_kpis_have_valid_rows"]), "canonical_12_kpis": list(extension_summary["canonical_12_kpis"])},
        "canonical_summary": canonical_summary,
        "claim_guards": dict(CLAIM_GUARDS),
        "notes": [
            "Existing canonical KPI aggregator computes stable six KPIs.",
            "Step 107 wrapper preserves queue/demand extension columns after canonical aggregation.",
            "Output is scaffold/proxy only; causal performance and paper-level claims remain blocked.",
        ],
    }
    readiness_rows = build_readiness_rows(manifest)
    write_csv_rows(paths.readiness_csv_path, readiness_rows, ["check_name", "passed", "observed_value", "note"])
    if not all(bool(r["passed"]) for r in readiness_rows):
        manifest["audit_status"] = "FAIL"
        manifest["integration_status"] = "BLOCKED"
    dump_json(paths.manifest_path, manifest)
    paths.report_path.write_text(build_markdown_report(manifest), encoding="utf-8")
    return manifest

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 107 queue/demand canonical KPI integration")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--canonical-aggregator", default=str(DEFAULT_AGGREGATOR))
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    paths = Step107Paths(Path(args.input_root), Path(args.output_root), Path(args.canonical_aggregator))
    try:
        manifest = run_step107(paths, smoke=not args.no_smoke, clean_output=bool(args.clean_output))
    except Exception as exc:
        paths.output_root.mkdir(parents=True, exist_ok=True)
        dump_json(paths.manifest_path, {"artifact_version": ARTIFACT_VERSION, "audit_status": "FAIL", "integration_status": "BLOCKED", "error": str(exc), "claim_guards": dict(CLAIM_GUARDS)})
        print("[FAIL] Step 107 queue/demand canonical KPI integration failed", file=sys.stderr)
        print(f"[FAIL] reason: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("[OK] Step 107 queue/demand canonical KPI integration completed")
    print(f"[OK] audit_status      : {manifest['audit_status']}")
    print(f"[OK] integration_status: {manifest['integration_status']}")
    print(f"[OK] output_root       : {paths.output_root}")
    print(f"[OK] canonical_output  : {paths.canonical_output_root}")
    print(f"[OK] preserved_window  : {paths.preserved_kpi_by_window_path}")
    print(f"[OK] kpi_by_window     : {manifest['canonical_summary']['row_counts']['kpi_by_window']}")
    print(f"[OK] preserved_rows    : {manifest['preservation_summary']['preserved_row_count']}")
    print(f"[OK] all_12_kpis       : {manifest['extension_summary']['all_12_kpis_present']}")
    print(f"[OK] causal_allowed    : {manifest['canonical_summary']['causal_comparison_allowed']}")

if __name__ == "__main__":
    main()
