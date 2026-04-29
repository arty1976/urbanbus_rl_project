#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 103 — route-aware rollout canonical KPI integration scaffold.

This script connects the Step 102 route-aware rollout writer output to the
existing canonical KPI (Key Performance Indicator=핵심 성과 지표) aggregator.

Safety contract:
- No DB (Database=데이터베이스) write.
- No tensor DB (Database=데이터베이스) overwrite.
- No additional API (Application Programming Interface=응용 프로그램 인터페이스) calls.
- Reads existing Step 102 rollout artifacts only.
- Writes a staged copy of window_rollup.parquet with normalized non-causal
  source_mode values so the existing canonical aggregator keeps
  causal_comparison_allowed=false.
- This is an integration scaffold, not a validated causal performance result.
- paper_level_claim_allowed=false.
- causal_performance_claim_allowed=false.

Default input:
- artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102/window_rollup.parquet

Default output:
- artifacts/daegu_bis_api_audit/route_aware_canonical_kpi_step103/

Default aggregator:
- 05_training/evaluation/canonical_kpi_aggregator.py
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd


ARTIFACT_VERSION = "route_aware_canonical_kpi_step103_v1"
CANONICAL_MODE = "official_rollup"

DEFAULT_INPUT_ROOT = Path(
    "artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/daegu_bis_api_audit/route_aware_canonical_kpi_step103"
)
DEFAULT_AGGREGATOR = Path("05_training/evaluation/canonical_kpi_aggregator.py")

EXPECTED_SHARED_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
]

OPTIONAL_EXTENDED_KPIS = [
    "passenger_demand_generated",
    "passenger_served_count",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "fleet_reduction_ratio",
]

REQUIRED_STEP102_ROLLUP_COLUMNS = [
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

CLAIM_GUARDS: Dict[str, bool] = {
    "db_write_performed": False,
    "tensor_db_overwrite_performed": False,
    "additional_api_calls_performed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "actual_headway_observed": False,
    "actual_arrival_departure_time_observed": False,
    "actual_dwell_observed": False,
}


@dataclass(frozen=True)
class Step103Paths:
    input_root: Path
    output_root: Path
    canonical_aggregator: Path

    @property
    def source_window_rollup(self) -> Path:
        return self.input_root / "window_rollup.parquet"

    @property
    def source_manifest(self) -> Path:
        return self.input_root / "route_aware_rollout_writer_manifest.json"

    @property
    def staging_root(self) -> Path:
        return self.output_root / "canonical_input_staging"

    @property
    def staged_window_rollup(self) -> Path:
        return self.staging_root / "window_rollup.parquet"

    @property
    def contract_path(self) -> Path:
        return self.output_root / "step103_canonical_aggregator_contract.json"

    @property
    def canonical_output_root(self) -> Path:
        return self.output_root / "canonical_eval"

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "route_aware_canonical_kpi_step103_manifest.json"

    @property
    def report_path(self) -> Path:
        return self.output_root / "route_aware_canonical_kpi_step103_report.md"

    @property
    def readiness_csv_path(self) -> Path:
        return self.output_root / "canonical_output_readiness_step103.csv"


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def normalize_source_mode(value: Any) -> str:
    """Normalize source mode so canonical guard detects non-causal scaffold rows."""
    text = str(value)
    lowered = text.lower()
    # Existing canonical_kpi_aggregator.py catches "non_causal" and "non-causal"
    # but not the glued token "noncausal".  Normalize only that marker.
    if "noncausal" in lowered and "non_causal" not in lowered and "non-causal" not in lowered:
        return text.replace("noncausal", "non_causal").replace("NONCAUSAL", "NON_CAUSAL")
    return text


def validate_step102_input(df: pd.DataFrame) -> Dict[str, Any]:
    missing = [c for c in REQUIRED_STEP102_ROLLUP_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Step 102 window_rollup missing required columns: {missing}")
    if df.empty:
        raise RuntimeError("Step 102 window_rollup is empty")

    if "source_mode" not in df.columns:
        raise RuntimeError("Step 102 window_rollup must contain source_mode")

    source_modes = sorted(df["source_mode"].astype(str).unique().tolist())
    if not any("scaffold" in s.lower() or "noncausal" in s.lower() or "non_causal" in s.lower() for s in source_modes):
        raise RuntimeError(
            "Step 102 source_mode does not look like a scaffold/non-causal source; "
            f"source_modes={source_modes}"
        )

    numeric_required = [
        "seed",
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
    ]
    bad_numeric: List[str] = []
    for col in numeric_required:
        if pd.to_numeric(df[col], errors="coerce").isna().any():
            bad_numeric.append(col)
    if bad_numeric:
        raise RuntimeError(f"Step 102 window_rollup has non-numeric values in: {bad_numeric}")

    return {
        "row_count": int(len(df)),
        "condition_ids": sorted(df["condition_id"].astype(str).str.upper().unique().tolist()),
        "seed_values": sorted([int(x) for x in pd.to_numeric(df["seed"], errors="raise").unique().tolist()]),
        "window_count": int(df["window_id"].nunique()),
        "time_bands": sorted(df["time_band"].astype(str).str.lower().unique().tolist()),
        "source_modes": source_modes,
        "optional_extended_kpis_present": [c for c in OPTIONAL_EXTENDED_KPIS if c in df.columns],
    }


def build_staged_input(paths: Step103Paths) -> Dict[str, Any]:
    if not paths.source_window_rollup.exists():
        raise RuntimeError(f"Step 102 window_rollup.parquet not found: {paths.source_window_rollup}")

    df = pd.read_parquet(paths.source_window_rollup).copy()
    input_summary = validate_step102_input(df)

    df["original_source_mode_step102"] = df["source_mode"].astype(str)
    df["source_mode"] = df["source_mode"].map(normalize_source_mode)
    normalized_source_modes = sorted(df["source_mode"].astype(str).unique().tolist())

    paths.staging_root.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.staged_window_rollup, index=False)

    # CSV copy is useful for quick inspection without parquet tooling.
    df.to_csv(paths.staging_root / "window_rollup.csv", index=False, encoding="utf-8-sig")

    return {
        "input_summary": input_summary,
        "staged_window_rollup": str(paths.staged_window_rollup),
        "staged_row_count": int(len(df)),
        "staged_source_modes": normalized_source_modes,
        "source_mode_normalized": input_summary["source_modes"] != normalized_source_modes,
    }


def build_canonical_contract(paths: Step103Paths, input_summary: Mapping[str, Any]) -> Dict[str, Any]:
    contract = {
        "artifact_version": "step103_route_aware_canonical_aggregator_contract_v1",
        "condition_id": "ROUTE_AWARE_STEP103",
        "condition_name": "route_aware_minimal_scaffold_canonical_integration",
        "qwen_train": False,
        "qwen_inference": False,
        "evaluation_horizon_minutes": 30,
        "time_bands": ["peak", "offpeak", "night"],
        "shared_kpis": list(EXPECTED_SHARED_KPIS),
        "fairness_constraints": {
            "same_initial_state": True,
            "same_exogenous_events": True,
            "same_eval_window": True,
        },
        "input_source": {
            "step102_input_root": str(paths.input_root),
            "staged_input_root": str(paths.staging_root),
            "input_row_count": int(input_summary.get("row_count", 0)),
            "condition_ids": list(input_summary.get("condition_ids", [])),
            "seeds": list(input_summary.get("seed_values", [])),
            "window_count": int(input_summary.get("window_count", 0)),
        },
        "claim_guards": dict(CLAIM_GUARDS),
        "note": (
            "This contract exists only to run the existing canonical KPI aggregator on "
            "route-aware minimal scaffold outputs. It is not a performance-claim contract."
        ),
    }
    dump_json(paths.contract_path, contract)
    return contract


def run_canonical_aggregator(paths: Step103Paths, *, smoke: bool) -> Dict[str, Any]:
    if not paths.canonical_aggregator.exists():
        raise RuntimeError(f"canonical_kpi_aggregator.py not found: {paths.canonical_aggregator}")

    cmd = [
        sys.executable,
        str(paths.canonical_aggregator),
        "--mode",
        CANONICAL_MODE,
        "--contract",
        str(paths.contract_path),
        "--input-root",
        str(paths.staging_root),
        "--output-root",
        str(paths.canonical_output_root),
    ]
    if smoke:
        cmd.append("--smoke")

    result = subprocess.run(cmd, capture_output=True, text=True)
    payload = {
        "command": cmd,
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    if result.returncode != 0:
        raise RuntimeError(
            "canonical KPI aggregator failed\n"
            f"cmd={cmd}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return payload


def read_canonical_outputs(paths: Step103Paths) -> Dict[str, Any]:
    expected_files = {
        "official_rollup_input_validation": paths.canonical_output_root / "official_rollup_input_validation.json",
        "kpi_by_window": paths.canonical_output_root / "kpi_by_window.parquet",
        "kpi_by_seed": paths.canonical_output_root / "kpi_by_seed.parquet",
        "kpi_by_time_band": paths.canonical_output_root / "kpi_by_time_band.parquet",
        "kpi_overall": paths.canonical_output_root / "kpi_overall.json",
        "aggregation_manifest": paths.canonical_output_root / "aggregation_manifest.json",
    }
    missing = [str(p) for p in expected_files.values() if not p.exists()]
    if missing:
        raise RuntimeError(f"canonical KPI output files missing: {missing}")

    window_df = pd.read_parquet(expected_files["kpi_by_window"])
    seed_df = pd.read_parquet(expected_files["kpi_by_seed"])
    time_band_df = pd.read_parquet(expected_files["kpi_by_time_band"])
    overall = load_json_any_encoding(expected_files["kpi_overall"])
    agg_manifest = load_json_any_encoding(expected_files["aggregation_manifest"])

    if window_df.empty:
        raise RuntimeError("canonical kpi_by_window is empty")

    missing_kpis = [c for c in EXPECTED_SHARED_KPIS if c not in window_df.columns]
    if missing_kpis:
        raise RuntimeError(f"canonical kpi_by_window missing expected KPI columns: {missing_kpis}")

    if "causal_comparison_allowed" not in window_df.columns:
        raise RuntimeError("canonical kpi_by_window missing causal_comparison_allowed")

    causal_allowed = bool(window_df["causal_comparison_allowed"].all())
    if causal_allowed:
        raise RuntimeError(
            "canonical output incorrectly allows causal comparison for route-aware scaffold rows"
        )

    if not bool((window_df["strict_canonical"] == True).all()):  # noqa: E712 - pandas scalar comparison
        raise RuntimeError("canonical kpi_by_window strict_canonical must be true for official_rollup")

    valid_counts = {
        k: int(pd.to_numeric(window_df[k], errors="coerce").notna().sum())
        for k in EXPECTED_SHARED_KPIS
    }
    null_counts = {
        k: int(pd.to_numeric(window_df[k], errors="coerce").isna().sum())
        for k in EXPECTED_SHARED_KPIS
    }
    if any(v == 0 for v in valid_counts.values()):
        raise RuntimeError(f"one or more canonical KPIs have zero valid rows: {valid_counts}")

    return {
        "output_files": {k: str(v) for k, v in expected_files.items()},
        "row_counts": {
            "kpi_by_window": int(len(window_df)),
            "kpi_by_seed": int(len(seed_df)),
            "kpi_by_time_band": int(len(time_band_df)),
        },
        "condition_ids": sorted(window_df["condition_id"].astype(str).unique().tolist()),
        "seed_values": sorted([int(x) for x in window_df["seed"].unique().tolist()]),
        "window_count": int(window_df["window_id"].nunique()),
        "time_bands": sorted(window_df["time_band"].astype(str).unique().tolist()),
        "source_modes": sorted(window_df["source_mode"].astype(str).unique().tolist()),
        "causal_comparison_allowed": causal_allowed,
        "strict_canonical": bool(window_df["strict_canonical"].all()),
        "valid_kpi_counts": valid_counts,
        "null_kpi_counts": null_counts,
        "overall_kpis": sorted(list(overall.get("kpis", {}).keys())),
        "aggregation_manifest_causal_allowed": bool(agg_manifest.get("causal_comparison_allowed", True)),
        "warnings": agg_manifest.get("warnings", []),
    }


def build_readiness_rows(canonical_summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    valid_counts = canonical_summary.get("valid_kpi_counts", {})
    null_counts = canonical_summary.get("null_kpi_counts", {})
    for kpi in EXPECTED_SHARED_KPIS:
        rows.append({
            "field": kpi,
            "category": "canonical_kpi",
            "status": "PASS" if int(valid_counts.get(kpi, 0)) > 0 else "FAIL",
            "valid_row_count": int(valid_counts.get(kpi, 0)),
            "null_row_count": int(null_counts.get(kpi, 0)),
            "claim_scope": "scaffold_contract_validation_only",
        })
    rows.extend([
        {
            "field": "causal_comparison_allowed",
            "category": "claim_guard",
            "status": "PASS" if canonical_summary.get("causal_comparison_allowed") is False else "FAIL",
            "valid_row_count": "",
            "null_row_count": "",
            "claim_scope": "must_be_false_for_step103",
        },
        {
            "field": "strict_canonical",
            "category": "canonical_output_contract",
            "status": "PASS" if canonical_summary.get("strict_canonical") is True else "FAIL",
            "valid_row_count": "",
            "null_row_count": "",
            "claim_scope": "official_rollup_schema_only",
        },
    ])
    return rows


def build_report(payload: Mapping[str, Any]) -> str:
    input_summary = payload["input_summary"]
    canonical_summary = payload["canonical_summary"]
    guards = payload["claim_guards"]

    lines = [
        "# Step 103 — Route-aware Canonical KPI Integration Scaffold",
        "",
        "## Status",
        "",
        f"- audit_status: `{payload['audit_status']}`",
        f"- integration_status: `{payload['integration_status']}`",
        f"- artifact_version: `{payload['artifact_version']}`",
        "",
        "## Input",
        "",
        f"- input_root: `{payload['input_root']}`",
        f"- source_window_rollup_rows: `{input_summary['row_count']}`",
        f"- condition_ids: `{input_summary['condition_ids']}`",
        f"- seeds: `{input_summary['seed_values']}`",
        f"- window_count: `{input_summary['window_count']}`",
        f"- source_modes: `{input_summary['source_modes']}`",
        f"- optional_extended_kpis_present: `{input_summary['optional_extended_kpis_present']}`",
        "",
        "## Canonical Output",
        "",
        f"- canonical_output_root: `{payload['canonical_output_root']}`",
        f"- kpi_by_window rows: `{canonical_summary['row_counts']['kpi_by_window']}`",
        f"- kpi_by_seed rows: `{canonical_summary['row_counts']['kpi_by_seed']}`",
        f"- kpi_by_time_band rows: `{canonical_summary['row_counts']['kpi_by_time_band']}`",
        f"- overall_kpis: `{canonical_summary['overall_kpis']}`",
        f"- causal_comparison_allowed: `{canonical_summary['causal_comparison_allowed']}`",
        "",
        "## Claim Guards",
        "",
    ]
    for key, value in guards.items():
        lines.append(f"- {key}: `{str(value).lower()}`")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Step 103 proves that Step 102 route-aware scaffold outputs can enter the existing canonical KPI aggregator path.",
        "The result is valid for schema and pipeline validation only.",
        "It does not prove observed actual headway, actual arrival/departure time, actual dwell, or causal performance improvement.",
        "",
        "## Next Step",
        "",
        "Step 104 should update the project log/runbook and record that route-aware scaffold → canonical KPI integration is now connected.",
    ])
    return "\n".join(lines) + "\n"


def run_step103(paths: Step103Paths, *, smoke: bool = True, clean_output: bool = False) -> Dict[str, Any]:
    if clean_output and paths.output_root.exists():
        shutil.rmtree(paths.output_root)

    paths.output_root.mkdir(parents=True, exist_ok=True)

    staged_summary = build_staged_input(paths)
    input_summary = staged_summary["input_summary"]
    build_canonical_contract(paths, input_summary=input_summary)
    aggregator_run = run_canonical_aggregator(paths, smoke=smoke)
    canonical_summary = read_canonical_outputs(paths)

    readiness_rows = build_readiness_rows(canonical_summary)
    write_csv(
        paths.readiness_csv_path,
        readiness_rows,
        fieldnames=["field", "category", "status", "valid_row_count", "null_row_count", "claim_scope"],
    )

    failed = [r for r in readiness_rows if r.get("status") != "PASS"]
    audit_status = "PASS" if not failed else "FAIL"
    integration_status = (
        "READY_FOR_STEP104_PROJECT_LOG_RUNBOOK_UPDATE" if audit_status == "PASS" else "BLOCKED"
    )

    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "audit_status": audit_status,
        "integration_status": integration_status,
        "input_root": str(paths.input_root),
        "output_root": str(paths.output_root),
        "canonical_aggregator": str(paths.canonical_aggregator),
        "canonical_output_root": str(paths.canonical_output_root),
        "input_summary": input_summary,
        "staging_summary": staged_summary,
        "canonical_summary": canonical_summary,
        "aggregator_run": aggregator_run,
        "readiness_csv": str(paths.readiness_csv_path),
        "claim_guards": dict(CLAIM_GUARDS),
        "smoke": {"enabled": bool(smoke), "passed": audit_status == "PASS"},
        "notes": [
            "Step 103 uses a staged copy of window_rollup.parquet so source_mode noncausal markers are normalized to non_causal.",
            "This prevents route-aware scaffold output from being misinterpreted as causal-comparison-ready by legacy guard patterns.",
            "The canonical KPI output is for schema and integration validation only.",
        ],
    }

    dump_json(paths.manifest_path, manifest)
    paths.report_path.write_text(build_report(manifest), encoding="utf-8")

    if audit_status != "PASS":
        raise RuntimeError(f"Step 103 failed readiness checks: {failed}")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 103 route-aware rollout canonical KPI integration scaffold"
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--canonical-aggregator", default=str(DEFAULT_AGGREGATOR))
    parser.add_argument("--no-smoke", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Step103Paths(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        canonical_aggregator=Path(args.canonical_aggregator),
    )
    manifest = run_step103(
        paths,
        smoke=not bool(args.no_smoke),
        clean_output=bool(args.clean_output),
    )

    print("[OK] Step 103 route-aware canonical KPI integration completed")
    print(f"[OK] audit_status      : {manifest['audit_status']}")
    print(f"[OK] integration_status: {manifest['integration_status']}")
    print(f"[OK] output_root       : {paths.output_root}")
    print(f"[OK] canonical_output  : {paths.canonical_output_root}")
    print(f"[OK] kpi_by_window     : {manifest['canonical_summary']['row_counts']['kpi_by_window']}")
    print(f"[OK] kpi_by_seed       : {manifest['canonical_summary']['row_counts']['kpi_by_seed']}")
    print(f"[OK] kpi_by_time_band  : {manifest['canonical_summary']['row_counts']['kpi_by_time_band']}")
    print(f"[OK] causal_allowed    : {manifest['canonical_summary']['causal_comparison_allowed']}")
    print(f"[OK] manifest_json     : {paths.manifest_path}")


if __name__ == "__main__":
    main()
