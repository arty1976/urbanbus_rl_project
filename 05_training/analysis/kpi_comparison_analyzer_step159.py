from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


ARTIFACT_VERSION = "kpi_comparison_analyzer_step159_v1"
STEP_ID = "Step 159"
COMPARISON_TOOL_STATUS = "KPI_COMPARISON_ANALYZER_READY_SCAFFOLD_STILL_LOCKED"
DECISION = "SCAFFOLD_ONLY_NOT_ACTUAL_COMPARISON"
NEXT_GATE = "WAITING_FOR_REAL_H200_STEP149_EXPECT_H200_RESULT"


KPI_DIRECTIONS: Dict[str, str] = {
    "cv_headway": "lower_is_better",
    "avg_wait_seconds": "lower_is_better",
    "bunching_rate": "lower_is_better",
    "intervention_rate": "lower_is_better",
    "energy_proxy": "lower_is_better",
    "passenger_wait_p95_seconds": "lower_is_better",
    "energy_proxy_per_passenger": "lower_is_better",
    "on_time_rate": "higher_is_better",
    "passenger_demand_generated": "higher_is_better",
    "passenger_served_count": "higher_is_better",
    "passenger_service_rate": "higher_is_better",
    "fleet_reduction_ratio": "higher_is_better",
}

ALL_KPIS: List[str] = list(KPI_DIRECTIONS.keys())
REQUIRED_META_COLUMNS: List[str] = ["condition_id", "seed", "time_band", "window_count"]
MIN_KPI_COLUMNS_NON_STRICT = 6
DEFAULT_BASELINE_CONDITION = "B0"

LOCK_VALUES: Dict[str, Any] = {
    "comparison_tool_status": COMPARISON_TOOL_STATUS,
    "comparison_allowed": True,
    "winner_selected": False,
    "trainable_reward_promoted": False,
    "actual_results": False,
    "actual_execution_allowed": False,
    "actual_execution_released": False,
    "train_allowed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "live_mutation_allowed": False,
    "db_write_allowed": False,
    "h200_required_for_actual_claim": True,
    "next_gate": NEXT_GATE,
}

PROHIBITED_CLAIMS: List[str] = [
    "training",
    "actual reward ablation execution",
    "winner selection",
    "trainable reward promotion",
    "causal performance claim",
    "paper-level claim",
    "database write",
    "live mutation of experiment configuration",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def get_git_commit(project_root: Path) -> str:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
        if cp.returncode == 0:
            return cp.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def get_git_status_short(project_root: Path) -> str:
    try:
        cp = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(project_root),
            text=True,
            capture_output=True,
            check=False,
        )
        if cp.returncode == 0:
            return cp.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def build_sample_dataframe() -> pd.DataFrame:
    """Deterministic synthetic sample for self-test only."""

    conditions = ["B0", "B1", "B2", "A"]
    seeds = [1, 2, 3]
    time_bands = ["AM_PEAK", "PM_PEAK", "OFF_PEAK"]

    base_values: Dict[str, float] = {
        "cv_headway": 0.40,
        "avg_wait_seconds": 320.0,
        "bunching_rate": 0.18,
        "intervention_rate": 0.0,
        "energy_proxy": 1500.0,
        "passenger_wait_p95_seconds": 720.0,
        "energy_proxy_per_passenger": 0.85,
        "on_time_rate": 0.78,
        "passenger_demand_generated": 12000.0,
        "passenger_served_count": 11200.0,
        "passenger_service_rate": 0.93,
        "fleet_reduction_ratio": 0.0,
    }

    condition_offsets: Dict[str, Dict[str, float]] = {
        "B0": {k: 0.0 for k in base_values},
        "B1": {
            "cv_headway": 0.05,
            "avg_wait_seconds": 40.0,
            "bunching_rate": 0.04,
            "intervention_rate": 0.0,
            "energy_proxy": 50.0,
            "passenger_wait_p95_seconds": 80.0,
            "energy_proxy_per_passenger": 0.04,
            "on_time_rate": -0.03,
            "passenger_demand_generated": 0.0,
            "passenger_served_count": -200.0,
            "passenger_service_rate": -0.01,
            "fleet_reduction_ratio": 0.0,
        },
        "B2": {
            "cv_headway": -0.05,
            "avg_wait_seconds": -30.0,
            "bunching_rate": -0.03,
            "intervention_rate": 0.05,
            "energy_proxy": -40.0,
            "passenger_wait_p95_seconds": -50.0,
            "energy_proxy_per_passenger": -0.02,
            "on_time_rate": 0.04,
            "passenger_demand_generated": 0.0,
            "passenger_served_count": 200.0,
            "passenger_service_rate": 0.01,
            "fleet_reduction_ratio": 0.0,
        },
        "A": {
            "cv_headway": -0.10,
            "avg_wait_seconds": -60.0,
            "bunching_rate": -0.06,
            "intervention_rate": 0.10,
            "energy_proxy": -120.0,
            "passenger_wait_p95_seconds": -120.0,
            "energy_proxy_per_passenger": -0.06,
            "on_time_rate": 0.07,
            "passenger_demand_generated": 0.0,
            "passenger_served_count": 400.0,
            "passenger_service_rate": 0.02,
            "fleet_reduction_ratio": 0.10,
        },
    }

    band_offsets: Dict[str, float] = {"AM_PEAK": 1.05, "PM_PEAK": 1.10, "OFF_PEAK": 0.85}

    rows: List[Dict[str, Any]] = []
    for cond in conditions:
        for seed in seeds:
            for band in time_bands:
                row: Dict[str, Any] = {
                    "condition_id": cond,
                    "seed": seed,
                    "time_band": band,
                    "window_count": 96,
                }
                seed_jitter = 1.0 + (seed - 2) * 0.01
                for kpi, base in base_values.items():
                    offset = condition_offsets[cond][kpi]
                    val = (base + offset) * band_offsets[band] * seed_jitter
                    row[kpi] = float(val)
                rows.append(row)

    return pd.DataFrame(rows)


def load_input_dataframe(args: argparse.Namespace) -> Tuple[pd.DataFrame, str]:
    if args.input_csv:
        path = Path(args.input_csv)
        if not path.exists():
            raise SystemExit(f"[FAIL] input csv not found: {path}")
        return pd.read_csv(path), f"csv:{path}"
    if args.input_parquet:
        path = Path(args.input_parquet)
        if not path.exists():
            raise SystemExit(f"[FAIL] input parquet not found: {path}")
        return pd.read_parquet(path), f"parquet:{path}"
    if args.sample_mode:
        return build_sample_dataframe(), "sample_mode_builtin"
    raise SystemExit(
        "[FAIL] no input provided; pass --input-csv, --input-parquet, or --sample-mode"
    )


def validate_schema(
    df: pd.DataFrame, baseline_condition: str, strict: bool
) -> Tuple[List[str], List[str]]:
    missing_meta = [c for c in REQUIRED_META_COLUMNS if c not in df.columns]
    if missing_meta:
        raise SystemExit(f"[FAIL] missing required meta columns: {missing_meta}")

    present_kpis = [k for k in ALL_KPIS if k in df.columns]
    if strict and len(present_kpis) < len(ALL_KPIS):
        missing = [k for k in ALL_KPIS if k not in present_kpis]
        raise SystemExit(f"[FAIL] strict mode requires all 12 KPIs; missing: {missing}")
    if len(present_kpis) < MIN_KPI_COLUMNS_NON_STRICT:
        raise SystemExit(
            f"[FAIL] need at least {MIN_KPI_COLUMNS_NON_STRICT} KPIs; got {len(present_kpis)}"
        )

    conditions = sorted({str(c) for c in df["condition_id"].dropna().unique().tolist()})
    if baseline_condition not in conditions:
        raise SystemExit(
            f"[FAIL] baseline_condition '{baseline_condition}' not present; "
            f"available conditions: {conditions}"
        )

    return present_kpis, conditions


def safe_pct(raw: float, baseline: float) -> Optional[float]:
    if baseline is None:
        return None
    try:
        if pd.isna(baseline):
            return None
    except Exception:
        pass
    if abs(baseline) < 1e-12:
        return None
    return float(raw) / abs(float(baseline)) * 100.0


def improvement(raw: float, kpi: str) -> float:
    direction = KPI_DIRECTIONS.get(kpi, "higher_is_better")
    if direction == "lower_is_better":
        return -float(raw)
    return float(raw)


def _none_for_csv(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and math.isnan(value):
            return ""
    except Exception:
        pass
    return value


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _none_for_csv(row.get(k)) for k in fieldnames})


def compute_grouped_deltas(
    df: pd.DataFrame,
    group_keys: List[str],
    present_kpis: List[str],
    baseline_means: Dict[str, float],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped = df.groupby(group_keys, dropna=False)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        meta = {gk: key[i] for i, gk in enumerate(group_keys)}
        for kpi in present_kpis:
            cm = float(sub[kpi].mean())
            bm = baseline_means.get(kpi)
            if bm is None or pd.isna(bm):
                raw = None
                pct = None
                imp = None
            else:
                raw = cm - float(bm)
                pct = safe_pct(raw, float(bm))
                imp = improvement(raw, kpi)
            row: Dict[str, Any] = dict(meta)
            row.update(
                {
                    "kpi": kpi,
                    "direction": KPI_DIRECTIONS.get(kpi, "unknown"),
                    "condition_mean": cm,
                    "baseline_mean": bm,
                    "raw_delta": raw,
                    "pct_delta": pct,
                    "improvement_score": imp,
                    "n_rows": int(len(sub)),
                }
            )
            rows.append(row)
    return rows


def build_summary_rows(
    by_condition_rows: List[Dict[str, Any]],
    conditions: List[str],
    baseline_condition: str,
) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for cond in conditions:
        cond_rows = [r for r in by_condition_rows if r["condition_id"] == cond]
        n_kpis = len(cond_rows)
        n_improved = 0
        n_regressed = 0
        n_neutral = 0
        imp_scores: List[float] = []
        for r in cond_rows:
            imp = r.get("improvement_score")
            if imp is None:
                continue
            imp_scores.append(float(imp))
            if imp > 1e-12:
                n_improved += 1
            elif imp < -1e-12:
                n_regressed += 1
            else:
                n_neutral += 1
        mean_imp = float(sum(imp_scores) / len(imp_scores)) if imp_scores else None
        summary.append(
            {
                "condition_id": cond,
                "is_baseline": cond == baseline_condition,
                "n_kpis_compared": n_kpis,
                "n_kpis_improved": n_improved,
                "n_kpis_regressed": n_regressed,
                "n_kpis_neutral_or_null": n_neutral + (n_kpis - len(imp_scores)),
                "mean_improvement_score": mean_imp,
            }
        )
    return summary


def write_report(
    path: Path,
    baseline_condition: str,
    conditions: List[str],
    present_kpis: List[str],
    summary_rows: List[Dict[str, Any]],
    by_condition_rows: List[Dict[str, Any]],
) -> None:
    lines: List[str] = []
    lines.append("# Step 159 — KPI Comparison Report (Scaffold)")
    lines.append("")
    lines.append(
        "**This file is a scaffold-level comparison report. "
        "It is NOT a paper-level claim and NOT an actual performance comparison.**"
    )
    lines.append("")
    lines.append(f"- artifact_version: `{ARTIFACT_VERSION}`")
    lines.append(f"- comparison_tool_status: `{COMPARISON_TOOL_STATUS}`")
    lines.append(f"- decision: `{DECISION}`")
    lines.append(f"- next_gate: `{NEXT_GATE}`")
    lines.append(f"- generated_at_utc: `{utc_now()}`")
    lines.append("")

    lines.append("## Baseline and conditions")
    lines.append("")
    lines.append(f"- baseline_condition: `{baseline_condition}`")
    lines.append(f"- conditions: {conditions}")
    lines.append(f"- kpis_present: {present_kpis}")
    lines.append("")

    lines.append("## KPI directions")
    lines.append("")
    lines.append("| KPI | Direction |")
    lines.append("|-----|-----------|")
    for kpi in present_kpis:
        lines.append(f"| {kpi} | {KPI_DIRECTIONS.get(kpi, 'unknown')} |")
    lines.append("")

    lines.append("## Per-condition delta summary")
    lines.append("")
    lines.append(
        "| condition | baseline? | n_kpis | improved | regressed | neutral_or_null | mean_improvement |"
    )
    lines.append(
        "|-----------|-----------|--------|----------|-----------|-----------------|------------------|"
    )
    for r in summary_rows:
        mi = r.get("mean_improvement_score")
        mi_text = "null" if mi is None else f"{mi:.4f}"
        lines.append(
            f"| {r['condition_id']} | {r['is_baseline']} | {r['n_kpis_compared']} "
            f"| {r['n_kpis_improved']} | {r['n_kpis_regressed']} "
            f"| {r['n_kpis_neutral_or_null']} | {mi_text} |"
        )
    lines.append("")

    service_kpis = ["passenger_service_rate", "avg_wait_seconds", "passenger_wait_p95_seconds"]
    energy_kpis = ["energy_proxy", "energy_proxy_per_passenger", "fleet_reduction_ratio"]

    def render_focus(title: str, focus_kpis: List[str]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        present_focus = [k for k in focus_kpis if k in present_kpis]
        if not present_focus:
            lines.append("- (no relevant KPIs present in input)")
            lines.append("")
            return
        lines.append("| condition | kpi | baseline_mean | condition_mean | raw_delta | pct_delta | improvement |")
        lines.append("|-----------|-----|---------------|----------------|-----------|-----------|-------------|")
        for r in by_condition_rows:
            if r["kpi"] not in present_focus:
                continue
            pct = r.get("pct_delta")
            pct_text = "null" if pct is None else f"{pct:.2f}%"
            raw = r.get("raw_delta")
            raw_text = "null" if raw is None else f"{raw:.4f}"
            imp = r.get("improvement_score")
            imp_text = "null" if imp is None else f"{imp:.4f}"
            bm = r.get("baseline_mean")
            cm = r.get("condition_mean")
            bm_text = "null" if bm is None or (isinstance(bm, float) and math.isnan(bm)) else f"{bm:.4f}"
            cm_text = "null" if cm is None or (isinstance(cm, float) and math.isnan(cm)) else f"{cm:.4f}"
            lines.append(
                f"| {r['condition_id']} | {r['kpi']} | {bm_text} | {cm_text} "
                f"| {raw_text} | {pct_text} | {imp_text} |"
            )
        lines.append("")

    render_focus("Service quality guard summary", service_kpis)
    render_focus("Energy / fleet efficiency guard summary", energy_kpis)

    lines.append("## Prohibited claims")
    lines.append("")
    for claim in PROHIBITED_CLAIMS:
        lines.append(f"- not authorized: {claim}")
    lines.append("")

    lines.append("## Next gate")
    lines.append("")
    lines.append(f"`{NEXT_GATE}`")
    lines.append("")

    write_text(path, "\n".join(lines))


def build_manifest_payload(
    project_root: Path,
    output_root: Path,
    args: argparse.Namespace,
    input_descriptor: str,
    df: pd.DataFrame,
    baseline_condition: str,
    conditions: List[str],
    present_kpis: List[str],
    summary_rows: List[Dict[str, Any]],
    output_files: Dict[str, Path],
) -> Dict[str, Any]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "step": STEP_ID,
        "comparison_tool_status": COMPARISON_TOOL_STATUS,
        "decision": DECISION,
        "next_gate": NEXT_GATE,
        "generated_at_utc": utc_now(),
        "git_commit": get_git_commit(project_root),
        "git_status_short": get_git_status_short(project_root),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "input_descriptor": input_descriptor,
        "input_csv": args.input_csv,
        "input_parquet": args.input_parquet,
        "sample_mode": bool(args.sample_mode),
        "strict": bool(args.strict),
        "baseline_condition": baseline_condition,
        "conditions": conditions,
        "kpis_present": present_kpis,
        "kpi_directions": {k: KPI_DIRECTIONS[k] for k in present_kpis},
        "row_count": int(len(df)),
        "summary": summary_rows,
        "output_files": {k: str(v) for k, v in output_files.items()},
        **LOCK_VALUES,
    }


def build_guard_payload() -> Dict[str, Any]:
    payload = {
        "artifact_version": ARTIFACT_VERSION + "_guard",
        "step": STEP_ID,
        "decision": DECISION,
        "generated_at_utc": utc_now(),
        "prohibited_claims": list(PROHIBITED_CLAIMS),
    }
    payload.update(LOCK_VALUES)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output-root",
        default="artifacts/analysis/kpi_comparison_analyzer_step159",
    )
    parser.add_argument("--baseline-condition", default=DEFAULT_BASELINE_CONDITION)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--input-parquet", default=None)
    parser.add_argument("--sample-mode", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    df, input_descriptor = load_input_dataframe(args)
    present_kpis, conditions = validate_schema(df, args.baseline_condition, args.strict)

    baseline_df = df[df["condition_id"] == args.baseline_condition]
    if len(baseline_df) == 0:
        raise SystemExit(
            f"[FAIL] baseline_condition '{args.baseline_condition}' has zero rows"
        )

    baseline_means: Dict[str, float] = {}
    for kpi in present_kpis:
        baseline_means[kpi] = float(baseline_df[kpi].mean())

    by_condition_rows = compute_grouped_deltas(
        df, ["condition_id"], present_kpis, baseline_means
    )
    by_seed_rows = compute_grouped_deltas(
        df, ["condition_id", "seed"], present_kpis, baseline_means
    )
    by_time_band_rows = compute_grouped_deltas(
        df, ["condition_id", "time_band"], present_kpis, baseline_means
    )

    summary_rows = build_summary_rows(by_condition_rows, conditions, args.baseline_condition)

    summary_csv = output_root / "kpi_comparison_summary_step159.csv"
    delta_by_cond_csv = output_root / "kpi_delta_by_condition_step159.csv"
    delta_by_seed_csv = output_root / "kpi_delta_by_seed_step159.csv"
    delta_by_band_csv = output_root / "kpi_delta_by_time_band_step159.csv"
    report_md = output_root / "kpi_comparison_report_step159.md"
    manifest_path = output_root / "kpi_comparison_manifest_step159.json"
    guard_path = output_root / "claim_guard_status_step159.json"

    write_csv(
        summary_csv,
        summary_rows,
        [
            "condition_id",
            "is_baseline",
            "n_kpis_compared",
            "n_kpis_improved",
            "n_kpis_regressed",
            "n_kpis_neutral_or_null",
            "mean_improvement_score",
        ],
    )
    common_delta_fields = [
        "kpi",
        "direction",
        "condition_mean",
        "baseline_mean",
        "raw_delta",
        "pct_delta",
        "improvement_score",
        "n_rows",
    ]
    write_csv(
        delta_by_cond_csv,
        by_condition_rows,
        ["condition_id"] + common_delta_fields,
    )
    write_csv(
        delta_by_seed_csv,
        by_seed_rows,
        ["condition_id", "seed"] + common_delta_fields,
    )
    write_csv(
        delta_by_band_csv,
        by_time_band_rows,
        ["condition_id", "time_band"] + common_delta_fields,
    )

    write_report(
        report_md,
        args.baseline_condition,
        conditions,
        present_kpis,
        summary_rows,
        by_condition_rows,
    )

    output_files = {
        "manifest": manifest_path,
        "guard": guard_path,
        "summary_csv": summary_csv,
        "delta_by_condition_csv": delta_by_cond_csv,
        "delta_by_seed_csv": delta_by_seed_csv,
        "delta_by_time_band_csv": delta_by_band_csv,
        "report_md": report_md,
    }

    manifest = build_manifest_payload(
        project_root=project_root,
        output_root=output_root,
        args=args,
        input_descriptor=input_descriptor,
        df=df,
        baseline_condition=args.baseline_condition,
        conditions=conditions,
        present_kpis=present_kpis,
        summary_rows=summary_rows,
        output_files=output_files,
    )
    dump_json(manifest_path, manifest)
    dump_json(guard_path, build_guard_payload())

    print(f"[OK] {STEP_ID} KPI comparison analyzer scaffold completed")
    print(f"[OK] comparison_tool_status: {COMPARISON_TOOL_STATUS}")
    print(f"[OK] decision              : {DECISION}")
    print(f"[OK] baseline_condition    : {args.baseline_condition}")
    print(f"[OK] conditions            : {conditions}")
    print(f"[OK] kpis_present          : {len(present_kpis)} / 12")
    print(f"[OK] manifest              : {manifest_path}")
    print(f"[OK] guard                 : {guard_path}")
    print(f"[OK] next_gate             : {NEXT_GATE}")


if __name__ == "__main__":
    main()
