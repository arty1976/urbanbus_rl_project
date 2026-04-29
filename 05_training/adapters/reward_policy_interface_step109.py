#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 109 — Reward-policy interface scaffold
===========================================

Purpose
-------
Convert Step 108 scaffold reward output into a policy-facing reward interface
artifact. This is a wiring/contract check only.

Important claim boundary
------------------------
The reward values are NOT final MAPPO reward design values.
They are temporary scaffold values used to verify that the 12-KPI route-aware
queue/demand pipeline can be consumed by later policy code.

This script must not:
- write to DB
- overwrite tensor DB
- call external APIs
- claim causal performance
- claim final reward design
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


REWARD_INTERFACE_VERSION = "reward_policy_interface_scaffold_step109"
DEFAULT_INPUT_ROOT = Path("artifacts/daegu_bis_api_audit/queue_demand_reward_wiring_step108")
DEFAULT_OUTPUT_ROOT = Path("artifacts/daegu_bis_api_audit/reward_policy_interface_step109")

EXPECTED_REWARD_COMPONENTS = [
    "reward_service",
    "reward_wait",
    "reward_long_wait",
    "reward_energy",
    "reward_fleet",
    "reward_total",
]

OPTIONAL_REWARD_COMPONENTS = [
    "reward_bunching",
    "reward_ontime",
    "reward_intervention",
    "reward_constraint",
]

POLICY_INTERFACE_COLUMNS = [
    "condition_id",
    "seed",
    "window_id",
    "reward_total_scaffold",
    "reward_service",
    "reward_wait",
    "reward_long_wait",
    "reward_energy",
    "reward_fleet",
    "reward_vector_json",
    "reward_source_step",
    "reward_interface_version",
    "reward_scaffold_only",
    "reward_weights_are_final",
    "reward_formula_finalized",
    "final_reward_design_claim_allowed",
    "train_with_this_reward_allowed",
    "paper_level_claim_allowed",
    "causal_performance_claim_allowed",
    "canonical_causal_comparison_allowed",
]


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def table_exists(path: Path) -> bool:
    return path.exists() or path.with_suffix(".csv").exists()


def read_table(path: Path) -> pd.DataFrame:
    if path.exists():
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"table not found: {path} or {csv_path}")


def write_table(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> Dict[str, Any]:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    info: Dict[str, Any] = {
        "parquet_path": str(parquet_path),
        "csv_path": str(csv_path),
        "parquet_written": False,
        "csv_written": False,
    }
    try:
        df.to_parquet(parquet_path, index=False)
        info["parquet_written"] = True
    except Exception as exc:
        info["parquet_error"] = repr(exc)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    info["csv_written"] = True
    return info


def as_bool_series(df: pd.DataFrame, col: str, default: bool) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def require_columns(df: pd.DataFrame, columns: List[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label} missing required columns: {missing}")


def find_step108_reward_table(input_root: Path) -> Path:
    candidates = [
        input_root / "reward_by_window_step108.parquet",
        input_root / "reward_by_window_step108.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Step 108 reward_by_window file not found. Expected one of: "
        + ", ".join(str(c) for c in candidates)
    )


def find_step108_manifest(input_root: Path) -> Optional[Path]:
    for c in [
        input_root / "queue_demand_reward_wiring_step108_manifest.json",
        input_root / "reward_overall_step108.json",
    ]:
        if c.exists():
            return c
    return None


def validate_step108_reward(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        raise RuntimeError("Step 108 reward_by_window is empty")

    require_columns(df, ["condition_id", "seed", "window_id", "reward_total"], "Step 108 reward_by_window")

    missing_components = [c for c in EXPECTED_REWARD_COMPONENTS if c not in df.columns]
    if missing_components:
        raise RuntimeError(f"Step 108 reward_by_window missing reward components: {missing_components}")

    numeric_cols = EXPECTED_REWARD_COMPONENTS + [c for c in OPTIONAL_REWARD_COMPONENTS if c in df.columns]
    numeric_summary: Dict[str, Dict[str, Optional[float]]] = {}
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().any():
            raise RuntimeError(f"reward component has non-numeric/null values: {c}")
        numeric_summary[c] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "mean": float(s.mean()),
        }

    # Hard guard: the reward must remain scaffold-only and not final.
    scaffold_only = as_bool_series(df, "reward_scaffold_only", True)
    if not bool(scaffold_only.all()):
        raise RuntimeError("reward_scaffold_only must be true for every row")

    final_design = as_bool_series(df, "final_reward_design_claim_allowed", False)
    if bool(final_design.any()):
        raise RuntimeError("final_reward_design_claim_allowed must remain false")

    causal_claim = as_bool_series(df, "causal_performance_claim_allowed", False)
    if bool(causal_claim.any()):
        raise RuntimeError("causal_performance_claim_allowed must remain false")

    paper_claim = as_bool_series(df, "paper_level_claim_allowed", False)
    if bool(paper_claim.any()):
        raise RuntimeError("paper_level_claim_allowed must remain false")

    causal_allowed = as_bool_series(df, "causal_comparison_allowed", False)
    if bool(causal_allowed.any()):
        raise RuntimeError("canonical causal_comparison_allowed must remain false")

    dup = df.duplicated(["condition_id", "seed", "window_id"], keep=False)
    if bool(dup.any()):
        examples = df.loc[dup, ["condition_id", "seed", "window_id"]].head(10).to_dict("records")
        raise RuntimeError(f"duplicate reward interface keys found: {examples}")

    return {
        "row_count": int(len(df)),
        "condition_ids": sorted(df["condition_id"].astype(str).unique().tolist()),
        "seed_values": sorted([int(x) for x in df["seed"].unique().tolist()]),
        "window_count": int(df[["condition_id", "seed", "window_id"]].drop_duplicates().shape[0]),
        "numeric_summary": numeric_summary,
        "claim_guards": {
            "reward_scaffold_only": True,
            "reward_weights_are_final": False,
            "reward_formula_finalized": False,
            "final_reward_design_claim_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "canonical_causal_comparison_allowed": False,
            "train_with_this_reward_allowed": False,
        },
    }


def build_reward_vector_json(row: pd.Series) -> str:
    vector: Dict[str, float] = {}
    for c in EXPECTED_REWARD_COMPONENTS + OPTIONAL_REWARD_COMPONENTS:
        if c in row.index:
            vector[c] = float(row[c])
    return json.dumps(vector, ensure_ascii=False, sort_keys=True)


def build_policy_interface(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["condition_id"] = df["condition_id"].astype(str)
    out["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    out["window_id"] = df["window_id"].astype(str)

    out["reward_total_scaffold"] = pd.to_numeric(df["reward_total"], errors="raise")
    for c in ["reward_service", "reward_wait", "reward_long_wait", "reward_energy", "reward_fleet"]:
        out[c] = pd.to_numeric(df[c], errors="raise") if c in df.columns else 0.0

    out["reward_vector_json"] = df.apply(build_reward_vector_json, axis=1)
    out["reward_source_step"] = "step108_queue_demand_reward_wiring"
    out["reward_interface_version"] = REWARD_INTERFACE_VERSION

    out["reward_scaffold_only"] = True
    out["reward_weights_are_final"] = False
    out["reward_formula_finalized"] = False
    out["final_reward_design_claim_allowed"] = False
    out["train_with_this_reward_allowed"] = False
    out["paper_level_claim_allowed"] = False
    out["causal_performance_claim_allowed"] = False
    out["canonical_causal_comparison_allowed"] = False

    # Preserve useful columns if Step 108 has them. These are metadata/proxy fields only.
    preserve_if_present = [
        "passenger_service_rate",
        "passenger_wait_p95_seconds",
        "energy_proxy_per_passenger",
        "fleet_reduction_ratio",
        "passenger_demand_generated",
        "passenger_served_count",
        "queue_demand_proxy",
        "queue_demand_observed",
        "actual_passenger_wait_observed",
        "actual_headway_observed",
        "actual_arrival_departure_time_observed",
        "actual_dwell_observed",
    ]
    for c in preserve_if_present:
        if c in df.columns:
            out[c] = df[c]

    require_columns(out, POLICY_INTERFACE_COLUMNS, "Step 109 policy interface")
    return out


def aggregate_interface(interface_df: pd.DataFrame) -> Dict[str, Any]:
    rows_by_condition: Dict[str, Any] = {}
    for condition_id, grp in interface_df.groupby("condition_id"):
        rewards = pd.to_numeric(grp["reward_total_scaffold"], errors="coerce")
        rows_by_condition[str(condition_id)] = {
            "row_count": int(len(grp)),
            "seed_count": int(grp["seed"].nunique()),
            "reward_total_scaffold_mean": float(rewards.mean()),
            "reward_total_scaffold_min": float(rewards.min()),
            "reward_total_scaffold_max": float(rewards.max()),
        }

    return {
        "row_count": int(len(interface_df)),
        "condition_ids": sorted(interface_df["condition_id"].astype(str).unique().tolist()),
        "seed_values": sorted([int(x) for x in interface_df["seed"].unique().tolist()]),
        "rows_by_condition": rows_by_condition,
        "claim_guards": {
            "reward_scaffold_only": True,
            "reward_weights_are_final": False,
            "reward_formula_finalized": False,
            "final_reward_design_claim_allowed": False,
            "train_with_this_reward_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "canonical_causal_comparison_allowed": False,
        },
    }


@dataclass
class Step109Paths:
    input_root: Path
    output_root: Path

    @property
    def interface_parquet(self) -> Path:
        return self.output_root / "policy_reward_interface_step109.parquet"

    @property
    def interface_csv(self) -> Path:
        return self.output_root / "policy_reward_interface_step109.csv"

    @property
    def summary_json(self) -> Path:
        return self.output_root / "policy_reward_interface_summary_step109.json"

    @property
    def schema_json(self) -> Path:
        return self.output_root / "policy_reward_interface_schema_step109.json"

    @property
    def readiness_csv(self) -> Path:
        return self.output_root / "policy_reward_interface_readiness_step109.csv"

    @property
    def manifest_json(self) -> Path:
        return self.output_root / "reward_policy_interface_step109_manifest.json"

    @property
    def report_md(self) -> Path:
        return self.output_root / "reward_policy_interface_step109_report.md"


def build_schema_payload(interface_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "artifact_version": REWARD_INTERFACE_VERSION,
        "interface_purpose": "policy-facing reward interface scaffold",
        "final_reward_design": False,
        "train_with_this_reward_allowed": False,
        "required_columns": POLICY_INTERFACE_COLUMNS,
        "available_columns": list(interface_df.columns),
        "reward_vector_json_components": EXPECTED_REWARD_COMPONENTS + OPTIONAL_REWARD_COMPONENTS,
        "claim_guards": {
            "reward_scaffold_only": True,
            "reward_weights_are_final": False,
            "reward_formula_finalized": False,
            "final_reward_design_claim_allowed": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    }


def build_readiness_matrix(summary: Dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "check_id": "S109_INTERFACE_ROWS_POSITIVE",
            "status": "PASS" if summary["row_count"] > 0 else "FAIL",
            "value": summary["row_count"],
            "required": "> 0",
        },
        {
            "check_id": "S109_REWARD_SCAFFOLD_ONLY",
            "status": "PASS",
            "value": True,
            "required": "true",
        },
        {
            "check_id": "S109_FINAL_REWARD_CLAIM_BLOCKED",
            "status": "PASS",
            "value": False,
            "required": "false",
        },
        {
            "check_id": "S109_TRAIN_WITH_THIS_REWARD_BLOCKED",
            "status": "PASS",
            "value": False,
            "required": "false",
        },
        {
            "check_id": "S109_CAUSAL_CLAIM_BLOCKED",
            "status": "PASS",
            "value": False,
            "required": "false",
        },
    ]
    return pd.DataFrame(rows)


def write_report(paths: Step109Paths, manifest: Dict[str, Any]) -> None:
    lines = [
        "# Step 109 — Reward-policy interface scaffold report",
        "",
        f"- audit_status: `{manifest['audit_status']}`",
        f"- interface_status: `{manifest['interface_status']}`",
        f"- input_reward_by_window: `{manifest['input_files']['reward_by_window']}`",
        f"- policy_reward_interface: `{manifest['output_files']['policy_reward_interface']}`",
        f"- rows: `{manifest['summary']['row_count']}`",
        f"- reward_scaffold_only: `{manifest['claim_guards']['reward_scaffold_only']}`",
        f"- reward_weights_are_final: `{manifest['claim_guards']['reward_weights_are_final']}`",
        f"- reward_formula_finalized: `{manifest['claim_guards']['reward_formula_finalized']}`",
        f"- final_reward_design_claim_allowed: `{manifest['claim_guards']['final_reward_design_claim_allowed']}`",
        f"- train_with_this_reward_allowed: `{manifest['claim_guards']['train_with_this_reward_allowed']}`",
        f"- causal_performance_claim_allowed: `{manifest['claim_guards']['causal_performance_claim_allowed']}`",
        "",
        "## Interpretation",
        "",
        "This step proves only that Step 108 scaffold reward rows can be converted into a policy-facing interface artifact.",
        "It does not define the final MAPPO reward and must not be used as a training objective for paper-level claims.",
        "",
        "## Next step",
        "",
        "Step 110 should define the final reward-design review gate or connect this interface to a mock policy runner while keeping scaffold guards active.",
    ]
    paths.report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_step109(input_root: Path, output_root: Path, clean_output: bool = False) -> Dict[str, Any]:
    paths = Step109Paths(input_root=input_root, output_root=output_root)

    if clean_output and output_root.exists():
        shutil.rmtree(output_root)

    paths.output_root.mkdir(parents=True, exist_ok=True)

    reward_path = find_step108_reward_table(input_root)
    manifest_path = find_step108_manifest(input_root)

    reward_df = read_table(reward_path)
    validation = validate_step108_reward(reward_df)
    interface_df = build_policy_interface(reward_df)

    interface_write = write_table(interface_df, paths.interface_parquet, paths.interface_csv)
    summary = aggregate_interface(interface_df)
    write_json(paths.summary_json, summary)

    schema_payload = build_schema_payload(interface_df)
    write_json(paths.schema_json, schema_payload)

    readiness_df = build_readiness_matrix(summary)
    readiness_df.to_csv(paths.readiness_csv, index=False, encoding="utf-8-sig")

    manifest: Dict[str, Any] = {
        "artifact_version": REWARD_INTERFACE_VERSION,
        "audit_status": "PASS",
        "interface_status": "READY_FOR_STEP110_POLICY_INTERFACE_MOCK_RUNNER_OR_REWARD_REVIEW",
        "input_files": {
            "reward_by_window": str(reward_path),
            "step108_manifest": str(manifest_path) if manifest_path else None,
        },
        "output_files": {
            "policy_reward_interface": str(paths.interface_parquet),
            "policy_reward_interface_csv": str(paths.interface_csv),
            "summary_json": str(paths.summary_json),
            "schema_json": str(paths.schema_json),
            "readiness_csv": str(paths.readiness_csv),
            "manifest_json": str(paths.manifest_json),
            "report_md": str(paths.report_md),
        },
        "input_validation": validation,
        "interface_write": interface_write,
        "summary": summary,
        "claim_guards": summary["claim_guards"],
        "notes": [
            "Step 109 reward interface is scaffold-only.",
            "The reward values are not final MAPPO reward design values.",
            "train_with_this_reward_allowed is intentionally false.",
            "No DB writes, tensor DB overwrites, or external API calls are performed.",
        ],
    }

    write_json(paths.manifest_json, manifest)
    write_report(paths, manifest)

    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 109 reward-policy interface scaffold")
    p.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--clean-output", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = run_step109(
            input_root=Path(args.input_root),
            output_root=Path(args.output_root),
            clean_output=bool(args.clean_output),
        )
    except Exception as exc:
        print("[FAIL] Step 109 reward-policy interface scaffold failed")
        print(f"[FAIL] reason: {exc}")
        return 1

    print("[OK] Step 109 reward-policy interface scaffold completed")
    print(f"[OK] audit_status   : {manifest['audit_status']}")
    print(f"[OK] interface_status: {manifest['interface_status']}")
    print(f"[OK] output_root     : {DEFAULT_OUTPUT_ROOT if Path(args.output_root) == DEFAULT_OUTPUT_ROOT else args.output_root}")
    print(f"[OK] interface_rows  : {manifest['summary']['row_count']}")
    print(f"[OK] scaffold_only   : {manifest['claim_guards']['reward_scaffold_only']}")
    print(f"[OK] train_allowed   : {manifest['claim_guards']['train_with_this_reward_allowed']}")
    print(f"[OK] final_claim     : {manifest['claim_guards']['final_reward_design_claim_allowed']}")
    print(f"[OK] causal_allowed  : {manifest['claim_guards']['causal_performance_claim_allowed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
