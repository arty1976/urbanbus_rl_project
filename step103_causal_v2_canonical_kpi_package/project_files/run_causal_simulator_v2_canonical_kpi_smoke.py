from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


EXPECTED_SHARED_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
]

REQUIRED_AGGREGATOR_OUTPUTS = [
    "official_rollup_input_validation.json",
    "kpi_by_window.parquet",
    "kpi_by_seed.parquet",
    "kpi_by_time_band.parquet",
    "kpi_overall.json",
    "aggregation_manifest.json",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_rollups(input_root: Path) -> List[Path]:
    return sorted(input_root.rglob("window_rollup.parquet"))


def build_contract(output_dir: Path, condition_id: str = "C2_STATIC_SIGNAL") -> Path:
    contract = {
        "artifact_version": "causal_v2_canonical_kpi_smoke_contract_step103",
        "condition_id": condition_id,
        "shared_kpis": EXPECTED_SHARED_KPIS,
        "evaluation_horizon_minutes": 30,
        "time_bands": ["peak", "offpeak", "night"],
        "fairness_constraints": {
            "same_initial_state": True,
            "same_exogenous_events": True,
            "same_eval_window": True,
        },
        "qwen_train": False,
        "qwen_inference": False,
        "claim_guardrails": {
            "trained_model": False,
            "performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "dynamic_signal_phase_claim_allowed": False,
            "red_light_delay_claim_allowed": False,
            "green_time_claim_allowed": False,
            "cycle_length_claim_allowed": False,
        },
        "source_mode_policy": {
            "expected_contains": ["smoke", "nonperformance"],
            "causal_comparison_allowed_expected": False,
        },
    }
    path = output_dir / "contract" / "c2_static_signal_canonical_contract.json"
    write_json(path, contract)
    return path


def normalize_rollup_for_aggregator(df: pd.DataFrame, idx: int) -> pd.DataFrame:
    out = df.copy()

    # Existing canonical_kpi_aggregator requires parseable datetimes.
    base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=30 * idx)
    out["state_ts"] = base_ts.isoformat()
    out["service_date"] = base_ts.date().isoformat()

    # Existing aggregator contract requires horizon=30.
    out["evaluation_horizon_minutes"] = 30

    # Step 102 already emits offpeak, but normalize defensively.
    if "time_band" not in out.columns:
        out["time_band"] = "offpeak"
    out["time_band"] = out["time_band"].astype(str).str.lower().replace({"scaffold": "offpeak"})

    # Keep qwen and replay metadata explicit.
    if "qwen_trigger_rate" not in out.columns:
        out["qwen_trigger_rate"] = 0.0
    if "effective_replay_step_minutes" not in out.columns:
        out["effective_replay_step_minutes"] = 60.0

    # Source mode must remain smoke/nonperformance so causal_comparison_allowed becomes false.
    if "source_mode" not in out.columns:
        out["source_mode"] = "static_signal_causal_v2_scaffold_smoke_nonperformance_v1"

    return out


def prepare_input(input_root: Path, prepared_root: Path) -> Dict[str, Any]:
    paths = discover_rollups(input_root)
    if not paths:
        raise RuntimeError(f"no window_rollup.parquet files found under input_root: {input_root}")

    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    scenario_rows: List[Dict[str, Any]] = []
    prepared_files: List[Dict[str, Any]] = []

    for i, src in enumerate(paths):
        df = pd.read_parquet(src)
        out_df = normalize_rollup_for_aggregator(df, idx=i)

        # Preserve relative folder shape where possible.
        rel_parent = src.parent.relative_to(input_root)
        dst_dir = prepared_root / rel_parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "window_rollup.parquet"
        out_df.to_parquet(dst, index=False)

        for row in out_df.itertuples(index=False):
            scenario_rows.append({
                "window_id": str(getattr(row, "window_id")),
                "state_ts": str(getattr(row, "state_ts")),
                "service_date": str(getattr(row, "service_date")),
                "time_band": str(getattr(row, "time_band")),
            })

        prepared_files.append({
            "source": str(src),
            "prepared": str(dst),
            "rows": int(len(out_df)),
        })

    scenario_df = pd.DataFrame(scenario_rows).drop_duplicates("window_id").reset_index(drop=True)
    scenario_index_path = prepared_root.parent / "scenario_index.parquet"
    scenario_df.to_parquet(scenario_index_path, index=False)

    return {
        "source_file_count": int(len(paths)),
        "prepared_files": prepared_files,
        "prepared_input_root": str(prepared_root),
        "scenario_index_path": str(scenario_index_path),
        "scenario_rows": int(len(scenario_df)),
    }


def run_aggregator(
    project_root: Path,
    contract_path: Path,
    prepared_input_root: Path,
    scenario_index_path: Path,
    canonical_output_root: Path,
) -> Dict[str, Any]:
    aggregator = project_root / "05_training" / "evaluation" / "canonical_kpi_aggregator.py"
    if not aggregator.exists():
        raise RuntimeError(f"canonical_kpi_aggregator.py not found: {aggregator}")

    cmd = [
        sys.executable,
        str(aggregator),
        "--mode",
        "official_rollup",
        "--contract",
        str(contract_path),
        "--input-root",
        str(prepared_input_root),
        "--scenario-index",
        str(scenario_index_path),
        "--output-root",
        str(canonical_output_root),
        "--smoke",
    ]

    completed = subprocess.run(
        cmd,
        cwd=str(project_root),
        text=True,
        capture_output=True,
    )

    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr)
        raise RuntimeError("canonical_kpi_aggregator official_rollup smoke failed")

    missing = [name for name in REQUIRED_AGGREGATOR_OUTPUTS if not (canonical_output_root / name).exists()]
    if missing:
        raise RuntimeError(f"canonical aggregator missing outputs: {missing}")

    manifest = json.loads((canonical_output_root / "aggregation_manifest.json").read_text(encoding="utf-8"))

    window_df = pd.read_parquet(canonical_output_root / "kpi_by_window.parquet")
    seed_df = pd.read_parquet(canonical_output_root / "kpi_by_seed.parquet")
    time_band_df = pd.read_parquet(canonical_output_root / "kpi_by_time_band.parquet")
    overall = json.loads((canonical_output_root / "kpi_overall.json").read_text(encoding="utf-8"))

    if bool(window_df["causal_comparison_allowed"].any()):
        raise RuntimeError("Step 103 expected causal_comparison_allowed=false for smoke/nonperformance source")

    for kpi in EXPECTED_SHARED_KPIS:
        if kpi not in window_df.columns:
            raise RuntimeError(f"kpi_by_window missing KPI column: {kpi}")

    return {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "canonical_output_root": str(canonical_output_root),
        "row_counts": {
            "kpi_by_window": int(len(window_df)),
            "kpi_by_seed": int(len(seed_df)),
            "kpi_by_time_band": int(len(time_band_df)),
        },
        "causal_comparison_allowed": bool(window_df["causal_comparison_allowed"].all()),
        "strict_canonical": bool(window_df["strict_canonical"].all()),
        "overall_n_windows_total": int(overall["n_windows_total"]),
        "aggregator_manifest_warning_count": int(len(manifest.get("warnings", []))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--input-root", default="artifacts/causal_simulator_v2_rollout_smoke")
    parser.add_argument("--output-root", default="artifacts/causal_simulator_v2_canonical_kpi_smoke")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = project_root / input_root

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    contract_path = build_contract(output_root)
    prepared_root = output_root / "prepared_input"
    prepare_summary = prepare_input(input_root, prepared_root)

    scenario_index_path = Path(prepare_summary["scenario_index_path"])
    canonical_output_root = output_root / "canonical_eval"

    aggregation_summary = run_aggregator(
        project_root=project_root,
        contract_path=contract_path,
        prepared_input_root=prepared_root,
        scenario_index_path=scenario_index_path,
        canonical_output_root=canonical_output_root,
    )

    manifest = {
        "artifact_version": "causal_v2_canonical_kpi_smoke_manifest_step103",
        "created_at_utc": now_utc_iso(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "contract_path": str(contract_path),
        "prepare_summary": prepare_summary,
        "aggregation_summary": aggregation_summary,
        "claim_guardrails": {
            "trained_model": False,
            "performance_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "dynamic_signal_phase_claim_allowed": False,
            "red_light_delay_claim_allowed": False,
            "green_time_claim_allowed": False,
            "cycle_length_claim_allowed": False,
        },
        "next_step": {
            "step": 104,
            "title": "Document Step 97-103 in project_log and runbook",
        },
    }

    manifest_path = output_root / "canonical_kpi_smoke_manifest.json"
    write_json(manifest_path, manifest)

    print("[OK] Step 103 causal-v2 canonical KPI smoke complete")
    print(f"[OK] input_root          : {input_root}")
    print(f"[OK] prepared_input_root : {prepared_root}")
    print(f"[OK] scenario_index      : {scenario_index_path}")
    print(f"[OK] canonical_eval      : {canonical_output_root}")
    print(f"[OK] manifest            : {manifest_path}")
    print(f"[OK] kpi_by_window rows  : {aggregation_summary['row_counts']['kpi_by_window']}")
    print(f"[OK] kpi_by_seed rows    : {aggregation_summary['row_counts']['kpi_by_seed']}")
    print(f"[OK] causal_allowed      : {aggregation_summary['causal_comparison_allowed']}")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
