#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 108 — queue/demand reward wiring scaffold.

Reads Step 107 preserved 12-KPI window output and computes scaffold-only
reward components. This is NOT the final MAPPO (Multi-Agent Proximal Policy
Optimization=다중 에이전트 근접 정책 최적화) reward design.

Safety: no DB (Database=데이터베이스) write, no tensor DB overwrite, no extra
API (Application Programming Interface=응용 프로그램 인터페이스) calls.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

ARTIFACT_VERSION = "queue_demand_reward_wiring_step108_v1"
REWARD_MODEL_VERSION = "queue_demand_reward_scaffold_step108_v1"
DEFAULT_INPUT_ROOT = Path("artifacts/daegu_bis_api_audit/queue_demand_canonical_kpi_step107")
DEFAULT_OUTPUT_ROOT = Path("artifacts/daegu_bis_api_audit/queue_demand_reward_wiring_step108")

CANONICAL_6_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
]
QUEUE_DEMAND_6_KPIS = [
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
]
REQUIRED_12_KPIS = [*CANONICAL_6_KPIS, *QUEUE_DEMAND_6_KPIS]
KEY_COLUMNS = ["condition_id", "seed", "window_id"]
OPTIONAL_CONTEXT_COLUMNS = [
    "state_ts",
    "service_date",
    "time_band",
    "source_mode",
    "queue_demand_observed",
    "queue_demand_proxy",
    "actual_passenger_wait_observed",
    "actual_headway_observed",
    "actual_arrival_departure_time_observed",
    "actual_dwell_observed",
    "causal_comparison_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
]
REWARD_COMPONENT_COLUMNS = [
    "reward_service",
    "reward_wait",
    "reward_long_wait",
    "reward_ontime",
    "reward_bunching",
    "reward_energy",
    "reward_fleet",
    "reward_constraint",
    "reward_total",
]
CLAIM_GUARDS: Dict[str, bool] = {
    "db_write_performed": False,
    "tensor_db_overwrite_performed": False,
    "additional_api_calls_performed": False,
    "reward_scaffold_only": True,
    "final_reward_design_claim_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "queue_demand_observed": False,
    "queue_demand_proxy": True,
    "actual_passenger_wait_observed": False,
    "actual_headway_observed": False,
    "actual_arrival_departure_time_observed": False,
    "actual_dwell_observed": False,
}

DEFAULT_REWARD_CONFIG: Dict[str, Any] = {
    "reward_model_version": REWARD_MODEL_VERSION,
    "scaffold_only": True,
    "description": (
        "Temporary reward wiring constants for Step 108. These weights are used only "
        "to verify that 12-KPI queue/demand outputs can be converted into reward "
        "components. They are not final MAPPO training reward values."
    ),
    "weights": {
        "service": 3.0,
        "avg_wait_penalty": -2.0,
        "long_wait_penalty": -3.0,
        "on_time": 1.0,
        "bunching_penalty": -1.5,
        "energy_per_passenger_penalty": -1.0,
        "fleet_reduction_bonus": 0.5,
        "constraint_violation_penalty": -5.0,
    },
    "thresholds": {
        "min_service_rate": 0.60,
        "wait_p95_cap_seconds": 900.0,
        "max_bunching_rate": 0.30,
        "min_on_time_rate": 0.40,
    },
    "baseline_policy": {
        "avg_wait_seconds": "median_of_step107_input_or_cli_override",
        "energy_proxy_per_passenger": "median_of_step107_input_or_cli_override",
    },
}

@dataclass(frozen=True)
class Step108Paths:
    input_root: Path
    output_root: Path
    input_kpi_by_window: Optional[Path] = None

    @property
    def source_kpi_by_window(self) -> Path:
        if self.input_kpi_by_window is not None:
            return self.input_kpi_by_window
        return self.input_root / "canonical_eval" / "kpi_by_window_queue_demand_preserved.parquet"

    @property
    def reward_by_window_path(self) -> Path:
        return self.output_root / "reward_by_window_step108.parquet"

    @property
    def reward_by_window_csv_path(self) -> Path:
        return self.output_root / "reward_by_window_step108.csv"

    @property
    def reward_by_seed_path(self) -> Path:
        return self.output_root / "reward_by_seed_step108.csv"

    @property
    def reward_by_condition_path(self) -> Path:
        return self.output_root / "reward_by_condition_step108.csv"

    @property
    def reward_overall_path(self) -> Path:
        return self.output_root / "reward_overall_step108.json"

    @property
    def readiness_csv_path(self) -> Path:
        return self.output_root / "queue_demand_reward_readiness_step108.csv"

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "queue_demand_reward_wiring_step108_manifest.json"

    @property
    def report_path(self) -> Path:
        return self.output_root / "queue_demand_reward_wiring_step108_report.md"


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


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
    except Exception as exc:
        parquet_error = str(exc)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return {
        "parquet_path": str(parquet_path),
        "csv_path": str(csv_path),
        "parquet_written": parquet_written,
        "parquet_error": parquet_error,
    }


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return default
    if not math.isfinite(x):
        return default
    return x


def safe_median(series: pd.Series, fallback: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float(fallback)
    val = float(s.median())
    if not math.isfinite(val) or val <= 0:
        return float(fallback)
    return val


def assert_bool_guard(df: pd.DataFrame, col: str, expected: bool) -> None:
    if col not in df.columns:
        return
    values = df[col].astype(bool)
    if expected and not bool(values.all()):
        raise RuntimeError(f"{col} must be true for all rows")
    if not expected and bool(values.any()):
        raise RuntimeError(f"{col} must be false for all rows")


def validate_input_12kpi(df: pd.DataFrame) -> Dict[str, Any]:
    missing = [c for c in [*KEY_COLUMNS, *REQUIRED_12_KPIS] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Step 108 input missing required 12-KPI columns: {missing}")
    if df.empty:
        raise RuntimeError("Step 108 input kpi_by_window is empty")
    if df.duplicated(KEY_COLUMNS).any():
        dups = df.loc[df.duplicated(KEY_COLUMNS, keep=False), KEY_COLUMNS].head(10)
        raise RuntimeError(f"Step 108 input has duplicate canonical keys: {dups.to_dict('records')}")
    numeric_required = [*REQUIRED_12_KPIS]
    bad = [c for c in numeric_required if pd.to_numeric(df[c], errors="coerce").isna().any()]
    if bad:
        raise RuntimeError(f"Step 108 input has non-numeric/null KPI values in: {bad}")
    for col in ["passenger_service_rate", "fleet_reduction_ratio", "on_time_rate", "bunching_rate", "intervention_rate"]:
        s = pd.to_numeric(df[col], errors="coerce")
        if bool((s < 0).any()) or bool((s > 1).any()):
            raise RuntimeError(f"{col} must be within [0, 1]")
    assert_bool_guard(df, "queue_demand_observed", False)
    assert_bool_guard(df, "queue_demand_proxy", True)
    assert_bool_guard(df, "actual_passenger_wait_observed", False)
    assert_bool_guard(df, "actual_headway_observed", False)
    assert_bool_guard(df, "actual_arrival_departure_time_observed", False)
    assert_bool_guard(df, "actual_dwell_observed", False)
    assert_bool_guard(df, "paper_level_claim_allowed", False)
    assert_bool_guard(df, "causal_performance_claim_allowed", False)
    if "causal_comparison_allowed" in df.columns and bool(df["causal_comparison_allowed"].astype(bool).any()):
        raise RuntimeError("Step 108 input must not allow causal comparison")
    return {
        "row_count": int(len(df)),
        "condition_ids": sorted(df["condition_id"].astype(str).unique().tolist()),
        "seed_values": sorted([int(x) for x in pd.to_numeric(df["seed"], errors="raise").unique().tolist()]),
        "window_count": int(df["window_id"].nunique()),
        "all_12_kpis_present": True,
        "demand_total": int(pd.to_numeric(df["passenger_demand_generated"], errors="coerce").sum()),
        "served_total": int(pd.to_numeric(df["passenger_served_count"], errors="coerce").sum()),
        "mean_service_rate": float(pd.to_numeric(df["passenger_service_rate"], errors="coerce").mean()),
    }


def compute_reward_by_window(
    df: pd.DataFrame,
    reward_config: Mapping[str, Any],
    baseline_avg_wait_seconds: Optional[float] = None,
    baseline_energy_proxy_per_passenger: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    weights = dict(reward_config.get("weights", {}))
    thresholds = dict(reward_config.get("thresholds", {}))

    baseline_avg_wait = float(
        baseline_avg_wait_seconds
        if baseline_avg_wait_seconds is not None and baseline_avg_wait_seconds > 0
        else safe_median(df["avg_wait_seconds"], fallback=300.0)
    )
    baseline_energy_pp = float(
        baseline_energy_proxy_per_passenger
        if baseline_energy_proxy_per_passenger is not None and baseline_energy_proxy_per_passenger > 0
        else safe_median(df["energy_proxy_per_passenger"], fallback=1.0)
    )
    baseline_energy_pp = max(baseline_energy_pp, 1e-6)

    min_service_rate = float(thresholds.get("min_service_rate", 0.60))
    wait_p95_cap = max(float(thresholds.get("wait_p95_cap_seconds", 900.0)), 1e-6)
    max_bunching_rate = float(thresholds.get("max_bunching_rate", 0.30))
    min_on_time_rate = float(thresholds.get("min_on_time_rate", 0.40))

    out = df.copy()
    service_rate = pd.to_numeric(out["passenger_service_rate"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    avg_wait = pd.to_numeric(out["avg_wait_seconds"], errors="coerce").fillna(baseline_avg_wait).clip(lower=0.0)
    wait_p95 = pd.to_numeric(out["passenger_wait_p95_seconds"], errors="coerce").fillna(wait_p95_cap).clip(lower=0.0)
    on_time_rate = pd.to_numeric(out["on_time_rate"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    bunching_rate = pd.to_numeric(out["bunching_rate"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    energy_pp = pd.to_numeric(out["energy_proxy_per_passenger"], errors="coerce").fillna(baseline_energy_pp).clip(lower=0.0)
    fleet_ratio = pd.to_numeric(out["fleet_reduction_ratio"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    avg_wait_penalty = ((avg_wait / max(baseline_avg_wait, 1e-6)) - 1.0).clip(lower=0.0)
    long_wait_penalty = ((wait_p95 / wait_p95_cap) - 1.0).clip(lower=0.0)
    energy_penalty = energy_pp / baseline_energy_pp

    constraint_violation_count = (
        (service_rate < min_service_rate).astype(int)
        + (wait_p95 > wait_p95_cap).astype(int)
        + (bunching_rate > max_bunching_rate).astype(int)
        + (on_time_rate < min_on_time_rate).astype(int)
    )

    out["reward_service"] = float(weights.get("service", 3.0)) * service_rate
    out["reward_wait"] = float(weights.get("avg_wait_penalty", -2.0)) * avg_wait_penalty
    out["reward_long_wait"] = float(weights.get("long_wait_penalty", -3.0)) * long_wait_penalty
    out["reward_ontime"] = float(weights.get("on_time", 1.0)) * on_time_rate
    out["reward_bunching"] = float(weights.get("bunching_penalty", -1.5)) * bunching_rate
    out["reward_energy"] = float(weights.get("energy_per_passenger_penalty", -1.0)) * energy_penalty
    out["reward_fleet"] = float(weights.get("fleet_reduction_bonus", 0.5)) * fleet_ratio
    out["constraint_violation_count"] = constraint_violation_count.astype(int)
    out["reward_constraint"] = float(weights.get("constraint_violation_penalty", -5.0)) * constraint_violation_count
    out["reward_total"] = out[REWARD_COMPONENT_COLUMNS[:-1]].sum(axis=1)

    out["reward_model_version"] = str(reward_config.get("reward_model_version", REWARD_MODEL_VERSION))
    out["reward_scaffold_only"] = True
    out["final_reward_design_claim_allowed"] = False
    out["reward_weights_are_final"] = False
    out["reward_formula_finalized"] = False
    out["performance_claim_allowed"] = False
    out["paper_level_claim_allowed"] = False
    out["causal_performance_claim_allowed"] = False
    out["baseline_avg_wait_seconds_step108"] = baseline_avg_wait
    out["baseline_energy_proxy_per_passenger_step108"] = baseline_energy_pp
    out["wait_p95_cap_seconds_step108"] = wait_p95_cap
    out["min_service_rate_step108"] = min_service_rate

    # Keep key/context/KPI/reward columns first but preserve all other metadata afterward.
    preferred = [
        *KEY_COLUMNS,
        *[c for c in OPTIONAL_CONTEXT_COLUMNS if c in out.columns],
        *REQUIRED_12_KPIS,
        "constraint_violation_count",
        *REWARD_COMPONENT_COLUMNS,
        "reward_model_version",
        "reward_scaffold_only",
        "final_reward_design_claim_allowed",
        "reward_weights_are_final",
        "reward_formula_finalized",
        "performance_claim_allowed",
        "baseline_avg_wait_seconds_step108",
        "baseline_energy_proxy_per_passenger_step108",
        "wait_p95_cap_seconds_step108",
        "min_service_rate_step108",
    ]
    remaining = [c for c in out.columns if c not in preferred]
    out = out[[*preferred, *remaining]]

    # Finite checks.
    for col in [*REWARD_COMPONENT_COLUMNS, "constraint_violation_count"]:
        s = pd.to_numeric(out[col], errors="coerce")
        if s.isna().any() or not bool(s.map(lambda x: math.isfinite(float(x))).all()):
            raise RuntimeError(f"non-finite reward output in column: {col}")

    config_used = {
        "reward_model_version": str(reward_config.get("reward_model_version", REWARD_MODEL_VERSION)),
        "scaffold_only": True,
        "weights": weights,
        "thresholds": thresholds,
        "derived_baselines": {
            "avg_wait_seconds": baseline_avg_wait,
            "energy_proxy_per_passenger": baseline_energy_pp,
        },
    }
    return out, config_used


def summarize_rewards(reward_df: pd.DataFrame, input_summary: Mapping[str, Any], config_used: Mapping[str, Any]) -> Dict[str, Any]:
    by_seed_rows: List[Dict[str, Any]] = []
    for (condition_id, seed), grp in reward_df.groupby(["condition_id", "seed"], dropna=False):
        row: Dict[str, Any] = {
            "condition_id": str(condition_id),
            "seed": int(seed),
            "window_count": int(len(grp)),
        }
        for col in REWARD_COMPONENT_COLUMNS:
            s = pd.to_numeric(grp[col], errors="coerce")
            row[f"{col}_mean"] = float(s.mean())
            row[f"{col}_min"] = float(s.min())
            row[f"{col}_max"] = float(s.max())
        row["constraint_violation_count_sum"] = int(pd.to_numeric(grp["constraint_violation_count"], errors="coerce").sum())
        by_seed_rows.append(row)

    by_condition_rows: List[Dict[str, Any]] = []
    for condition_id, grp in reward_df.groupby("condition_id", dropna=False):
        row = {
            "condition_id": str(condition_id),
            "seed_count": int(grp["seed"].nunique()),
            "window_count": int(len(grp)),
            "reward_total_mean": float(pd.to_numeric(grp["reward_total"], errors="coerce").mean()),
            "reward_total_min": float(pd.to_numeric(grp["reward_total"], errors="coerce").min()),
            "reward_total_max": float(pd.to_numeric(grp["reward_total"], errors="coerce").max()),
            "passenger_service_rate_mean": float(pd.to_numeric(grp["passenger_service_rate"], errors="coerce").mean()),
            "constraint_violation_count_sum": int(pd.to_numeric(grp["constraint_violation_count"], errors="coerce").sum()),
        }
        by_condition_rows.append(row)

    reward_stats = {
        col: {
            "mean": float(pd.to_numeric(reward_df[col], errors="coerce").mean()),
            "min": float(pd.to_numeric(reward_df[col], errors="coerce").min()),
            "max": float(pd.to_numeric(reward_df[col], errors="coerce").max()),
        }
        for col in REWARD_COMPONENT_COLUMNS
    }
    return {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": "PASS",
        "reward_status": "READY_FOR_STEP109_REWARD_POLICY_INTERFACE_SCAFFOLD",
        "input_summary": dict(input_summary),
        "reward_model_version": str(config_used.get("reward_model_version", REWARD_MODEL_VERSION)),
        "reward_config_used": config_used,
        "reward_stats": reward_stats,
        "row_counts": {
            "reward_by_window": int(len(reward_df)),
            "reward_by_seed": int(len(by_seed_rows)),
            "reward_by_condition": int(len(by_condition_rows)),
        },
        "guards": dict(CLAIM_GUARDS),
        "non_claim_notes": [
            "Step 108 reward values are temporary scaffold wiring outputs, not final MAPPO training reward values.",
            "The reward weights are intentionally marked non-final.",
            "Queue/demand metrics remain proxy-only and cannot support paper-level or causal performance claims.",
        ],
        "by_seed_rows": by_seed_rows,
        "by_condition_rows": by_condition_rows,
    }


def write_report(paths: Step108Paths, overall: Mapping[str, Any]) -> None:
    stats = overall.get("reward_stats", {})
    lines = [
        "# Step 108 — Queue/Demand Reward Wiring Scaffold",
        "",
        "## Status",
        "",
        f"- audit_status: `{overall.get('audit_status')}`",
        f"- reward_status: `{overall.get('reward_status')}`",
        f"- reward_model_version: `{overall.get('reward_model_version')}`",
        "- reward_scaffold_only: `true`",
        "- final_reward_design_claim_allowed: `false`",
        "",
        "## Purpose",
        "",
        "This step verifies that the Step 107 preserved 12-KPI output can be converted into reward components.",
        "The reward values are temporary scaffold wiring outputs and are not final MAPPO training reward values.",
        "",
        "## Reward Components",
        "",
    ]
    for col in REWARD_COMPONENT_COLUMNS:
        s = stats.get(col, {})
        lines.append(f"- `{col}`: mean={s.get('mean')}, min={s.get('min')}, max={s.get('max')}")
    lines.extend([
        "",
        "## Claim Guards",
        "",
        "- queue_demand_observed: `false`",
        "- queue_demand_proxy: `true`",
        "- actual_passenger_wait_observed: `false`",
        "- actual_headway_observed: `false`",
        "- actual_arrival_departure_time_observed: `false`",
        "- actual_dwell_observed: `false`",
        "- paper_level_claim_allowed: `false`",
        "- causal_performance_claim_allowed: `false`",
        "",
        "## Outputs",
        "",
        f"- reward_by_window: `{paths.reward_by_window_path}`",
        f"- reward_by_seed: `{paths.reward_by_seed_path}`",
        f"- reward_by_condition: `{paths.reward_by_condition_path}`",
        f"- reward_overall: `{paths.reward_overall_path}`",
    ])
    paths.report_path.parent.mkdir(parents=True, exist_ok=True)
    paths.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_step108(
    paths: Step108Paths,
    clean_output: bool = False,
    baseline_avg_wait_seconds: Optional[float] = None,
    baseline_energy_proxy_per_passenger: Optional[float] = None,
) -> Dict[str, Any]:
    if clean_output and paths.output_root.exists():
        shutil.rmtree(paths.output_root)
    paths.output_root.mkdir(parents=True, exist_ok=True)

    input_df = read_table(paths.source_kpi_by_window)
    input_summary = validate_input_12kpi(input_df)
    reward_df, config_used = compute_reward_by_window(
        input_df,
        DEFAULT_REWARD_CONFIG,
        baseline_avg_wait_seconds=baseline_avg_wait_seconds,
        baseline_energy_proxy_per_passenger=baseline_energy_proxy_per_passenger,
    )

    write_info = write_table_with_csv_fallback(
        reward_df,
        paths.reward_by_window_path,
        paths.reward_by_window_csv_path,
    )
    overall = summarize_rewards(reward_df, input_summary, config_used)

    by_seed_df = pd.DataFrame(overall.pop("by_seed_rows"))
    by_condition_df = pd.DataFrame(overall.pop("by_condition_rows"))
    by_seed_df.to_csv(paths.reward_by_seed_path, index=False, encoding="utf-8-sig")
    by_condition_df.to_csv(paths.reward_by_condition_path, index=False, encoding="utf-8-sig")

    overall_with_files: Dict[str, Any] = dict(overall)
    overall_with_files["input_files"] = {
        "source_kpi_by_window": str(paths.source_kpi_by_window),
    }
    overall_with_files["output_files"] = {
        "reward_by_window": str(paths.reward_by_window_path),
        "reward_by_window_csv": str(paths.reward_by_window_csv_path),
        "reward_by_seed": str(paths.reward_by_seed_path),
        "reward_by_condition": str(paths.reward_by_condition_path),
        "reward_overall": str(paths.reward_overall_path),
        "readiness_csv": str(paths.readiness_csv_path),
        "manifest": str(paths.manifest_path),
        "report": str(paths.report_path),
    }
    overall_with_files["reward_by_window_write"] = write_info

    dump_json(paths.reward_overall_path, overall_with_files)

    readiness_rows = [
        {"check_id": "S108_INPUT_12_KPIS", "passed": True, "value": True, "message": "Step 107 preserved output contains all 12 KPI columns."},
        {"check_id": "S108_REWARD_COMPONENTS_FINITE", "passed": True, "value": True, "message": "All reward component columns are finite."},
        {"check_id": "S108_REWARD_SCAFFOLD_ONLY", "passed": True, "value": True, "message": "Reward values are explicitly scaffold-only."},
        {"check_id": "S108_FINAL_REWARD_CLAIM_BLOCKED", "passed": True, "value": False, "message": "Final reward design claim remains blocked."},
        {"check_id": "S108_CAUSAL_CLAIM_BLOCKED", "passed": True, "value": False, "message": "Causal performance claim remains blocked."},
    ]
    write_csv_rows(paths.readiness_csv_path, readiness_rows, ["check_id", "passed", "value", "message"])

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": "PASS",
        "reward_status": overall_with_files["reward_status"],
        "input_files": overall_with_files["input_files"],
        "output_files": overall_with_files["output_files"],
        "row_counts": overall_with_files["row_counts"],
        "guards": dict(CLAIM_GUARDS),
        "reward_config_used": config_used,
        "next_step": "Step 109 — reward policy interface scaffold",
    }
    dump_json(paths.manifest_path, manifest)
    write_report(paths, overall_with_files)

    return overall_with_files


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 108 queue/demand reward wiring scaffold")
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--input-kpi-by-window", default="")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--baseline-avg-wait-seconds", type=float, default=0.0)
    parser.add_argument("--baseline-energy-proxy-per-passenger", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_kpi = Path(args.input_kpi_by_window) if args.input_kpi_by_window else None
    paths = Step108Paths(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        input_kpi_by_window=input_kpi,
    )
    try:
        result = run_step108(
            paths,
            clean_output=bool(args.clean_output),
            baseline_avg_wait_seconds=(args.baseline_avg_wait_seconds or None),
            baseline_energy_proxy_per_passenger=(args.baseline_energy_proxy_per_passenger or None),
        )
    except Exception as exc:
        paths.output_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_version": ARTIFACT_VERSION,
            "audit_status": "FAIL",
            "reward_status": "BLOCKED",
            "reason": str(exc),
            "guards": dict(CLAIM_GUARDS),
        }
        dump_json(paths.manifest_path, failure)
        print("[FAIL] Step 108 queue/demand reward wiring failed")
        print(f"[FAIL] reason: {exc}")
        return 1

    print("[OK] Step 108 queue/demand reward wiring completed")
    print(f"[OK] audit_status    : {result['audit_status']}")
    print(f"[OK] reward_status   : {result['reward_status']}")
    print(f"[OK] output_root     : {paths.output_root}")
    print(f"[OK] reward_by_window: {paths.reward_by_window_path}")
    print(f"[OK] rows            : {result['row_counts']['reward_by_window']}")
    print(f"[OK] reward_scaffold : {result['guards']['reward_scaffold_only']}")
    print(f"[OK] final_claim     : {result['guards']['final_reward_design_claim_allowed']}")
    print(f"[OK] causal_allowed  : {result['guards']['causal_performance_claim_allowed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
