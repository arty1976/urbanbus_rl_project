#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 106 — minimal queue/demand scaffold for route-aware simulator v2.

This script consumes Step 102 route-aware rollout artifacts and adds a
minimal deterministic passenger demand / queue service process.

Inputs by default:
- artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102/raw_events.parquet
- artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102/window_rollup.parquet

Outputs by default:
- artifacts/daegu_bis_api_audit/minimal_queue_demand_step106/raw_events_queue_demand.csv
- artifacts/daegu_bis_api_audit/minimal_queue_demand_step106/raw_events_queue_demand.parquet
- artifacts/daegu_bis_api_audit/minimal_queue_demand_step106/window_rollup_queue_demand.csv
- artifacts/daegu_bis_api_audit/minimal_queue_demand_step106/window_rollup_queue_demand.parquet
- artifacts/daegu_bis_api_audit/minimal_queue_demand_step106/minimal_queue_demand_manifest.json
- artifacts/daegu_bis_api_audit/minimal_queue_demand_step106/minimal_queue_demand_report.md
- artifacts/daegu_bis_api_audit/minimal_queue_demand_step106/queue_demand_readiness_step106.csv

Safety contract:
- No DB (Database=데이터베이스) write.
- No tensor DB (Database=데이터베이스) overwrite.
- No additional API (Application Programming Interface=응용 프로그램 인터페이스) calls.
- Reads existing CSV/parquet artifacts only.
- This is a deterministic scaffold/proxy, not an observed passenger process.
- paper_level_claim_allowed=false.
- causal_performance_claim_allowed=false.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Step 106 requires pandas, consistent with prior project steps.") from exc


ARTIFACT_VERSION = "minimal_queue_demand_step106_v1"
QUEUE_DEMAND_MODEL_VERSION = "minimal_queue_demand_proxy_v1"
SOURCE_MODE = "route_aware_queue_demand_scaffold_step106_non_causal"

DEFAULT_STEP102_ROOT = Path("artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102")
DEFAULT_RAW_EVENTS = DEFAULT_STEP102_ROOT / "raw_events.parquet"
DEFAULT_WINDOW_ROLLUP = DEFAULT_STEP102_ROOT / "window_rollup.parquet"
DEFAULT_OUTPUT_ROOT = Path("artifacts/daegu_bis_api_audit/minimal_queue_demand_step106")

CLAIM_GUARDS: Dict[str, bool] = {
    "db_write_performed": False,
    "tensor_db_overwrite_performed": False,
    "additional_api_calls_performed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "queue_demand_observed": False,
    "queue_demand_proxy": True,
    "actual_headway_observed": False,
    "actual_arrival_departure_time_observed": False,
    "actual_dwell_observed": False,
    "actual_passenger_wait_observed": False,
}

CANONICAL_12_KPIS = [
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

REQUIRED_RAW_COLUMNS = [
    "condition_id",
    "seed",
    "window_id",
    "step_index",
    "agent_id",
    "route_id",
    "direction_id",
    "stop_id",
    "action",
]

REQUIRED_WINDOW_COLUMNS = [
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "service_date",
    "time_band",
    "evaluation_horizon_minutes",
    "headway_mean_seconds",
    "headway_std_seconds",
    "headway_sample_count",
    "bunching_event_count",
    "headway_event_count",
    "wait_total_passenger_seconds",
    "wait_passenger_count",
    "ontime_event_count",
    "schedulable_arrival_count",
    "intervention_count",
    "decision_step_count",
    "energy_proxy_total",
    "source_mode",
]


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read json: {path}")


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        # Fallback between parquet and CSV names.
        if path.suffix.lower() == ".parquet":
            alt = path.with_suffix(".csv")
            if alt.exists():
                return pd.read_csv(alt)
        if path.suffix.lower() == ".csv":
            alt = path.with_suffix(".parquet")
            if alt.exists():
                return pd.read_parquet(alt)
        raise FileNotFoundError(f"input table not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table extension: {path}")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def try_write_parquet(path: Path, df: pd.DataFrame) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        return {"written": True, "path": str(path), "error": ""}
    except Exception as exc:
        return {"written": False, "path": str(path), "error": str(exc)}


def stable_int(*parts: Any) -> int:
    text = "|".join(str(x) for x in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def to_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_input_frames(raw_df: pd.DataFrame, rollup_df: pd.DataFrame) -> Dict[str, Any]:
    missing_raw = [c for c in REQUIRED_RAW_COLUMNS if c not in raw_df.columns]
    missing_rollup = [c for c in REQUIRED_WINDOW_COLUMNS if c not in rollup_df.columns]
    if missing_raw:
        raise RuntimeError(f"raw_events missing required columns: {missing_raw}")
    if missing_rollup:
        raise RuntimeError(f"window_rollup missing required columns: {missing_rollup}")
    if raw_df.empty:
        raise RuntimeError("raw_events input is empty")
    if rollup_df.empty:
        raise RuntimeError("window_rollup input is empty")
    if rollup_df.duplicated(["condition_id", "seed", "window_id"]).any():
        raise RuntimeError("window_rollup has duplicate (condition_id, seed, window_id) rows")
    return {
        "raw_event_rows": int(len(raw_df)),
        "window_rollup_rows": int(len(rollup_df)),
        "condition_count": int(rollup_df["condition_id"].astype(str).nunique()),
        "seed_count": int(rollup_df["seed"].astype(int).nunique()),
        "window_count": int(rollup_df["window_id"].astype(str).nunique()),
    }


def demand_base_for_time_band(time_band: str) -> int:
    tb = str(time_band).strip().lower()
    if tb == "peak":
        return 5
    if tb == "night":
        return 1
    return 3


def condition_service_capacity_factor(condition_id: str) -> float:
    cid = str(condition_id).strip().upper()
    if cid == "A90":
        return 0.95
    if cid == "A80":
        return 0.90
    if cid == "A70":
        return 0.82
    return 1.00


def compute_event_demand(row: Mapping[str, Any], time_band: str) -> int:
    base = demand_base_for_time_band(time_band)
    h = stable_int(row.get("window_id"), row.get("step_index"), row.get("agent_id"), row.get("stop_id"))
    # Deterministic but non-constant demand.  This is a proxy, not observation.
    return int(max(0, base + (h % 4) - 1))


def compute_service_capacity(row: Mapping[str, Any], active_bus_count: int) -> int:
    action = str(row.get("action", "advance")).strip().lower()
    at_terminal = truthy(row.get("at_terminal", False))
    if action == "hold":
        return 0
    if at_terminal:
        return 2
    factor = condition_service_capacity_factor(str(row.get("condition_id", "A")))
    capacity = int(round((6 + max(0, active_bus_count - 1)) * factor))
    return max(1, capacity)


def p95(values: Sequence[float]) -> float:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return 0.0
    idx = int(math.ceil(0.95 * len(clean))) - 1
    idx = min(max(idx, 0), len(clean) - 1)
    return float(clean[idx])


def mean(values: Sequence[float], default: float = 0.0) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return float(default)
    return float(sum(clean) / len(clean))


def infer_window_lookup(rollup_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rollup_df.to_dict("records"):
        out[str(r["window_id"])] = dict(r)
    return out


def enrich_raw_events(raw_df: pd.DataFrame, rollup_df: pd.DataFrame) -> pd.DataFrame:
    window_lookup = infer_window_lookup(rollup_df)
    raw = raw_df.copy()
    # Stable sort so queue state is deterministic.
    sort_cols = [c for c in ["condition_id", "seed", "window_id", "step_index", "agent_id"] if c in raw.columns]
    raw = raw.sort_values(sort_cols).reset_index(drop=True)

    queue_state: Dict[Tuple[str, str], int] = {}
    wait_age_state: Dict[Tuple[str, str], float] = {}
    enriched: List[Dict[str, Any]] = []

    for row in raw.to_dict("records"):
        window_id = str(row.get("window_id"))
        stop_id = str(row.get("stop_id", ""))
        key = (window_id, stop_id)
        rollup = window_lookup.get(window_id, {})
        time_band = str(rollup.get("time_band", row.get("time_band", "offpeak")))
        active_bus_count = to_int(rollup.get("active_bus_count", 1), default=1)

        queue_before = int(queue_state.get(key, 0))
        demand_arrival = compute_event_demand(row, time_band)
        capacity_available = compute_service_capacity(row, active_bus_count)
        demand_available = queue_before + demand_arrival
        boarded = int(min(demand_available, capacity_available))
        queue_after = int(max(0, demand_available - boarded))
        left_behind = queue_after

        previous_age = float(wait_age_state.get(key, 0.0))
        if queue_after <= 0:
            wait_age_proxy_seconds = 0.0
        else:
            # Queue age increases with leftover passengers.  This is intentionally
            # conservative and deterministic, not a real passenger timestamp model.
            wait_age_proxy_seconds = float(min(3600.0, previous_age + 60.0 + queue_after * 12.0))

        queue_state[key] = queue_after
        wait_age_state[key] = wait_age_proxy_seconds

        queue_pressure = float(queue_after / max(1, capacity_available))
        service_rate_event = float(boarded / max(1, demand_available)) if demand_available > 0 else 1.0

        out = dict(row)
        out.update({
            "artifact_version": ARTIFACT_VERSION,
            "queue_demand_model_version": QUEUE_DEMAND_MODEL_VERSION,
            "source_mode": SOURCE_MODE,
            "demand_arrival_count": int(demand_arrival),
            "passenger_queue_before": int(queue_before),
            "service_capacity_available": int(capacity_available),
            "passenger_boarded_count": int(boarded),
            "passenger_queue_after": int(queue_after),
            "passenger_left_behind_count": int(left_behind),
            "event_service_rate": float(service_rate_event),
            "queue_pressure": float(queue_pressure),
            "wait_age_proxy_seconds": float(wait_age_proxy_seconds),
            "queue_demand_observed": False,
            "queue_demand_proxy": True,
            "actual_passenger_wait_observed": False,
            "actual_headway_observed": False,
            "actual_arrival_departure_time_observed": False,
            "actual_dwell_observed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "causal_comparison_allowed": False,
            "scaffold_only": True,
        })
        enriched.append(out)

    return pd.DataFrame(enriched)


def rebuild_window_rollup(enriched_raw: pd.DataFrame, rollup_df: pd.DataFrame) -> pd.DataFrame:
    base_lookup = infer_window_lookup(rollup_df)
    rows: List[Dict[str, Any]] = []

    for window_id, grp in enriched_raw.groupby("window_id", sort=True):
        base = dict(base_lookup[str(window_id)])
        demand_total = int(pd.to_numeric(grp["demand_arrival_count"], errors="coerce").fillna(0).sum())
        served_total = int(pd.to_numeric(grp["passenger_boarded_count"], errors="coerce").fillna(0).sum())
        left_behind_total = int(pd.to_numeric(grp["passenger_left_behind_count"], errors="coerce").fillna(0).sum())
        max_queue_depth = int(pd.to_numeric(grp["passenger_queue_after"], errors="coerce").fillna(0).max())
        avg_queue_depth = float(pd.to_numeric(grp["passenger_queue_after"], errors="coerce").fillna(0).mean())
        avg_queue_pressure = float(pd.to_numeric(grp["queue_pressure"], errors="coerce").fillna(0).mean())
        wait_values = pd.to_numeric(grp["wait_age_proxy_seconds"], errors="coerce").fillna(0).astype(float).tolist()
        service_rate = float(served_total / demand_total) if demand_total > 0 else 1.0
        wait_p95 = float(p95(wait_values))
        avg_wait_proxy = float(max(to_float(base.get("avg_wait_seconds", 0.0), 0.0), mean(wait_values, 0.0)))
        wait_passenger_count = int(max(1, demand_total + left_behind_total))
        wait_total = float(avg_wait_proxy * wait_passenger_count)

        energy_proxy_total = to_float(base.get("energy_proxy_total", base.get("energy_proxy", 0.0)), 0.0)
        energy_per_passenger = float(energy_proxy_total / max(1, served_total))
        unmet_demand_rate = float(max(0.0, 1.0 - service_rate))

        base.update({
            "artifact_version": ARTIFACT_VERSION,
            "queue_demand_model_version": QUEUE_DEMAND_MODEL_VERSION,
            "source_mode": SOURCE_MODE,
            "passenger_demand_generated": int(demand_total),
            "passenger_served_count": int(served_total),
            "passenger_service_rate": float(service_rate),
            "passenger_wait_p95_seconds": float(wait_p95),
            "avg_wait_seconds": float(avg_wait_proxy),
            "wait_total_passenger_seconds": float(wait_total),
            "wait_passenger_count": int(wait_passenger_count),
            "energy_proxy_per_passenger": float(energy_per_passenger),
            "passenger_left_behind_count": int(left_behind_total),
            "max_queue_depth": int(max_queue_depth),
            "avg_queue_depth": float(avg_queue_depth),
            "avg_queue_pressure": float(avg_queue_pressure),
            "unmet_demand_rate": float(unmet_demand_rate),
            "queue_event_count": int(len(grp)),
            "queue_demand_observed": False,
            "queue_demand_proxy": True,
            "actual_passenger_wait_observed": False,
            "actual_headway_observed": False,
            "actual_arrival_departure_time_observed": False,
            "actual_dwell_observed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "causal_comparison_allowed": False,
            "scaffold_only": True,
            "performance_claim_allowed": False,
            "actual_policy_claim_ready": False,
            "causal_policy_claim_ready": False,
        })
        # Keep aliases used by the canonical KPI aggregator.
        base["energy_proxy"] = to_float(base.get("energy_proxy", energy_proxy_total), energy_proxy_total)
        if "fleet_reduction_ratio" not in base:
            active = to_float(base.get("active_bus_count", 1), 1)
            baseline = max(1.0, to_float(base.get("baseline_bus_count", active), active))
            base["fleet_reduction_ratio"] = float(max(0.0, 1.0 - active / baseline))
        rows.append(base)

    out = pd.DataFrame(rows)
    missing_kpis = [c for c in CANONICAL_12_KPIS if c not in out.columns]
    if missing_kpis:
        raise RuntimeError(f"Step 106 rollup missing 12-KPI columns: {missing_kpis}")
    return out.sort_values(["condition_id", "seed", "window_id"]).reset_index(drop=True)


def build_readiness_rows(raw_df: pd.DataFrame, rollup_df: pd.DataFrame) -> List[Dict[str, Any]]:
    checks: List[Tuple[str, bool, Any, str]] = []
    checks.append(("raw_event_rows_positive", len(raw_df) > 0, int(len(raw_df)), "enriched raw events exist"))
    checks.append(("window_rollup_rows_positive", len(rollup_df) > 0, int(len(rollup_df)), "enriched window rollup exists"))
    checks.append(("demand_generated_positive", int(pd.to_numeric(rollup_df["passenger_demand_generated"], errors="coerce").fillna(0).sum()) > 0, int(pd.to_numeric(rollup_df["passenger_demand_generated"], errors="coerce").fillna(0).sum()), "demand proxy generated"))
    checks.append(("served_passenger_nonnegative", bool((pd.to_numeric(rollup_df["passenger_served_count"], errors="coerce").fillna(0) >= 0).all()), int(pd.to_numeric(rollup_df["passenger_served_count"], errors="coerce").fillna(0).sum()), "served passenger count nonnegative"))
    checks.append(("service_rate_range", bool(((pd.to_numeric(rollup_df["passenger_service_rate"], errors="coerce").fillna(-1) >= 0) & (pd.to_numeric(rollup_df["passenger_service_rate"], errors="coerce").fillna(2) <= 1)).all()), "0..1", "passenger_service_rate within [0, 1]"))
    checks.append(("wait_p95_nonnegative", bool((pd.to_numeric(rollup_df["passenger_wait_p95_seconds"], errors="coerce").fillna(0) >= 0).all()), float(pd.to_numeric(rollup_df["passenger_wait_p95_seconds"], errors="coerce").fillna(0).max()), "p95 wait proxy nonnegative"))
    checks.append(("claim_guard_false", not bool(rollup_df["causal_performance_claim_allowed"].any()), False, "causal performance claims remain blocked"))
    checks.append(("queue_demand_marked_proxy", bool(rollup_df["queue_demand_proxy"].all()) and not bool(rollup_df["queue_demand_observed"].any()), True, "queue/demand is marked as proxy, not observed"))
    rows = []
    for name, passed, observed, note in checks:
        rows.append({"check_name": name, "passed": bool(passed), "observed_value": observed, "note": note})
    return rows


def write_readiness_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check_name", "passed", "observed_value", "note"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def build_markdown_report(manifest: Mapping[str, Any]) -> str:
    rc = manifest["row_counts"]
    q = manifest["queue_summary"]
    guards = manifest["claim_guards"]
    return f"""# Step 106 — Minimal Queue/Demand Scaffold

## Status

- audit_status: `{manifest['audit_status']}`
- scaffold_status: `{manifest['scaffold_status']}`
- artifact_version: `{manifest['artifact_version']}`
- queue_demand_model_version: `{manifest['queue_demand_model_version']}`

## Inputs

- raw_events_input: `{manifest['input_files']['raw_events']}`
- window_rollup_input: `{manifest['input_files']['window_rollup']}`

## Outputs

- raw_events_queue_demand_csv: `{manifest['output_files']['raw_events_queue_demand_csv']}`
- raw_events_queue_demand_parquet: `{manifest['output_files'].get('raw_events_queue_demand_parquet', '')}`
- window_rollup_queue_demand_csv: `{manifest['output_files']['window_rollup_queue_demand_csv']}`
- window_rollup_queue_demand_parquet: `{manifest['output_files'].get('window_rollup_queue_demand_parquet', '')}`

## Row counts

- raw_event_rows: `{rc['raw_event_rows']}`
- window_rollup_rows: `{rc['window_rollup_rows']}`
- condition_count: `{rc['condition_count']}`
- seed_count: `{rc['seed_count']}`
- window_count: `{rc['window_count']}`

## Queue/demand proxy summary

- passenger_demand_generated_total: `{q['passenger_demand_generated_total']}`
- passenger_served_count_total: `{q['passenger_served_count_total']}`
- passenger_service_rate_mean: `{q['passenger_service_rate_mean']}`
- passenger_wait_p95_seconds_max: `{q['passenger_wait_p95_seconds_max']}`
- max_queue_depth: `{q['max_queue_depth']}`
- unmet_demand_rate_mean: `{q['unmet_demand_rate_mean']}`

## Claim guards

- paper_level_claim_allowed: `{guards['paper_level_claim_allowed']}`
- causal_performance_claim_allowed: `{guards['causal_performance_claim_allowed']}`
- queue_demand_observed: `{guards['queue_demand_observed']}`
- queue_demand_proxy: `{guards['queue_demand_proxy']}`
- actual_passenger_wait_observed: `{guards['actual_passenger_wait_observed']}`
- actual_headway_observed: `{guards['actual_headway_observed']}`
- actual_arrival_departure_time_observed: `{guards['actual_arrival_departure_time_observed']}`
- actual_dwell_observed: `{guards['actual_dwell_observed']}`

## Interpretation

Step 106 adds a deterministic minimal passenger queue/demand proxy on top of
Step 102 route-aware rollout artifacts.  It is suitable for Step 107 canonical
KPI integration or reward wiring smoke tests, but it is not an observed
passenger process and must not be used for paper-level performance claims.
"""


def run_step106(
    *,
    raw_events_path: Path,
    window_rollup_path: Path,
    output_root: Path,
) -> Dict[str, Any]:
    raw_df = read_table(raw_events_path)
    rollup_df = read_table(window_rollup_path)
    input_summary = validate_input_frames(raw_df, rollup_df)

    enriched_raw = enrich_raw_events(raw_df, rollup_df)
    enriched_rollup = rebuild_window_rollup(enriched_raw, rollup_df)

    readiness_rows = build_readiness_rows(enriched_raw, enriched_rollup)
    hard_failures = [r for r in readiness_rows if not bool(r["passed"])]
    audit_status = "PASS" if not hard_failures else "FAIL"
    scaffold_status = "READY_FOR_STEP107_QUEUE_DEMAND_CANONICAL_KPI" if audit_status == "PASS" else "BLOCKED"

    output_root.mkdir(parents=True, exist_ok=True)
    raw_csv = output_root / "raw_events_queue_demand.csv"
    raw_parquet = output_root / "raw_events_queue_demand.parquet"
    rollup_csv = output_root / "window_rollup_queue_demand.csv"
    rollup_parquet = output_root / "window_rollup_queue_demand.parquet"
    readiness_csv = output_root / "queue_demand_readiness_step106.csv"
    manifest_json = output_root / "minimal_queue_demand_manifest.json"
    report_md = output_root / "minimal_queue_demand_report.md"

    write_csv(raw_csv, enriched_raw)
    write_csv(rollup_csv, enriched_rollup)
    raw_parquet_status = try_write_parquet(raw_parquet, enriched_raw)
    rollup_parquet_status = try_write_parquet(rollup_parquet, enriched_rollup)
    write_readiness_csv(readiness_csv, readiness_rows)

    queue_summary = {
        "passenger_demand_generated_total": int(pd.to_numeric(enriched_rollup["passenger_demand_generated"], errors="coerce").fillna(0).sum()),
        "passenger_served_count_total": int(pd.to_numeric(enriched_rollup["passenger_served_count"], errors="coerce").fillna(0).sum()),
        "passenger_service_rate_mean": float(pd.to_numeric(enriched_rollup["passenger_service_rate"], errors="coerce").fillna(0).mean()),
        "passenger_wait_p95_seconds_max": float(pd.to_numeric(enriched_rollup["passenger_wait_p95_seconds"], errors="coerce").fillna(0).max()),
        "max_queue_depth": int(pd.to_numeric(enriched_rollup["max_queue_depth"], errors="coerce").fillna(0).max()),
        "unmet_demand_rate_mean": float(pd.to_numeric(enriched_rollup["unmet_demand_rate"], errors="coerce").fillna(0).mean()),
    }

    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "queue_demand_model_version": QUEUE_DEMAND_MODEL_VERSION,
        "audit_status": audit_status,
        "scaffold_status": scaffold_status,
        "input_files": {
            "raw_events": str(raw_events_path),
            "window_rollup": str(window_rollup_path),
        },
        "output_root": str(output_root),
        "output_files": {
            "raw_events_queue_demand_csv": str(raw_csv),
            "raw_events_queue_demand_parquet": str(raw_parquet) if raw_parquet_status.get("written") else "",
            "window_rollup_queue_demand_csv": str(rollup_csv),
            "window_rollup_queue_demand_parquet": str(rollup_parquet) if rollup_parquet_status.get("written") else "",
            "readiness_csv": str(readiness_csv),
            "manifest_json": str(manifest_json),
            "report_md": str(report_md),
        },
        "input_summary": input_summary,
        "row_counts": {
            "raw_event_rows": int(len(enriched_raw)),
            "window_rollup_rows": int(len(enriched_rollup)),
            "condition_count": int(enriched_rollup["condition_id"].astype(str).nunique()),
            "seed_count": int(enriched_rollup["seed"].astype(int).nunique()),
            "window_count": int(enriched_rollup["window_id"].astype(str).nunique()),
        },
        "queue_summary": queue_summary,
        "parquet_status": {
            "raw_events_queue_demand": raw_parquet_status,
            "window_rollup_queue_demand": rollup_parquet_status,
        },
        "readiness": {
            "hard_failures": int(len(hard_failures)),
            "checks": readiness_rows,
        },
        "claim_guards": dict(CLAIM_GUARDS),
        "notes": [
            "Queue/demand metrics are deterministic scaffold proxies, not observed passenger timestamps.",
            "This step prepares minimal queue/demand signals for Step 107 canonical KPI integration and later reward wiring.",
            "Causal performance and paper-level claims remain explicitly blocked.",
        ],
    }
    dump_json(manifest_json, manifest)
    report_md.write_text(build_markdown_report(manifest), encoding="utf-8")

    if audit_status != "PASS":
        raise SystemExit(1)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 106 minimal queue/demand scaffold")
    parser.add_argument("--raw-events", default=str(DEFAULT_RAW_EVENTS), help="Step 102 raw_events parquet/csv")
    parser.add_argument("--window-rollup", default=str(DEFAULT_WINDOW_ROLLUP), help="Step 102 window_rollup parquet/csv")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output directory")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        manifest = run_step106(
            raw_events_path=Path(args.raw_events),
            window_rollup_path=Path(args.window_rollup),
            output_root=Path(args.output_root),
        )
    except SystemExit:
        raise
    except Exception as exc:
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        failure = {
            "artifact_version": ARTIFACT_VERSION,
            "audit_status": "FAIL",
            "scaffold_status": "BLOCKED",
            "error": str(exc),
            "claim_guards": dict(CLAIM_GUARDS),
        }
        dump_json(output_root / "minimal_queue_demand_manifest.json", failure)
        print("[FAIL] Step 106 minimal queue/demand scaffold failed")
        print(f"[ERROR] {exc}")
        raise SystemExit(1)

    print("[OK] Step 106 minimal queue/demand scaffold completed")
    print(f"[OK] audit_status   : {manifest['audit_status']}")
    print(f"[OK] scaffold_status: {manifest['scaffold_status']}")
    print(f"[OK] output_root    : {manifest['output_root']}")
    print(f"[OK] raw_events     : {manifest['row_counts']['raw_event_rows']}")
    print(f"[OK] window_rollup  : {manifest['row_counts']['window_rollup_rows']}")
    print(f"[OK] demand_total   : {manifest['queue_summary']['passenger_demand_generated_total']}")
    print(f"[OK] served_total   : {manifest['queue_summary']['passenger_served_count_total']}")
    print(f"[OK] causal_allowed : {manifest['claim_guards']['causal_performance_claim_allowed']}")


if __name__ == "__main__":
    main()
