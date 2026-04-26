from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("artifacts/baseline_v2_causal/comparison_smoke/step19_shared_demand_fairness_by_window.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/baseline_v2_causal/analysis")

CORE_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "energy_proxy",
]

EXTENDED_KPIS = [
    "active_bus_count",
    "fleet_ratio_vs_b0r",
    "fleet_reduction_ratio",
    "passenger_demand_generated",
    "passengers_served",
    "passenger_service_rate",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "intervention_events",
]

ANALYSIS_KPIS = CORE_KPIS + EXTENDED_KPIS

LOWER_IS_BETTER = {
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "energy_proxy",
    "active_bus_count",
    "fleet_ratio_vs_b0r",
    "passenger_wait_p95_seconds",
    "energy_proxy_per_passenger",
    "intervention_events",
}

HIGHER_IS_BETTER = {
    "on_time_rate",
    "fleet_reduction_ratio",
    "passengers_served",
    "passenger_service_rate",
}

SUCCESS_CONSTRAINTS = {
    "max_avg_wait_increase_ratio_vs_b0r": 1.10,
    "min_passenger_service_rate_ratio_vs_b0r": 0.98,
    "max_on_time_rate_drop_percentage_points_vs_b0r": 10.0,
    "max_p95_wait_increase_ratio_vs_b0r": 1.20,
    "max_bunching_rate_increase_percentage_points_vs_b0r": 5.0,
    "require_energy_proxy_reduction_vs_b0r": True,
}


def load_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"[FAIL] input CSV not found: {path}")

    df = pd.read_csv(path)

    required = [
        "label",
        "condition_id",
        "seed",
        "window_id",
        "time_band",
        "source_mode",
        "causal_comparison_allowed",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[FAIL] input missing required columns: {missing}")

    missing_kpis = [c for c in ANALYSIS_KPIS if c not in df.columns]
    if missing_kpis:
        raise SystemExit(f"[FAIL] input missing KPI columns: {missing_kpis}")

    for col in ANALYSIS_KPIS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[ANALYSIS_KPIS].isna().any().any():
        bad = df[ANALYSIS_KPIS].columns[df[ANALYSIS_KPIS].isna().any()].tolist()
        raise SystemExit(f"[FAIL] non-numeric or null KPI values found: {bad}")

    if not bool(df["causal_comparison_allowed"].astype(bool).all()):
        raise SystemExit("[FAIL] causal_comparison_allowed must be true for all rows")

    return df


def write_correlation_matrices(df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    numeric = df[ANALYSIS_KPIS].copy()

    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")

    pearson_csv = out_dir / "kpi_correlation_pearson.csv"
    spearman_csv = out_dir / "kpi_correlation_spearman.csv"
    pearson_json = out_dir / "kpi_correlation_pearson.json"
    spearman_json = out_dir / "kpi_correlation_spearman.json"

    pearson.to_csv(pearson_csv, encoding="utf-8-sig")
    spearman.to_csv(spearman_csv, encoding="utf-8-sig")

    pearson_json.write_text(
        json.dumps(pearson.round(6).to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    spearman_json.write_text(
        json.dumps(spearman.round(6).to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "pearson_csv": str(pearson_csv),
        "spearman_csv": str(spearman_csv),
        "pearson_json": str(pearson_json),
        "spearman_json": str(spearman_json),
    }


def summarize_strong_correlations(df: pd.DataFrame, threshold: float = 0.70) -> pd.DataFrame:
    corr = df[ANALYSIS_KPIS].corr(method="spearman")

    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            value = float(corr.loc[a, b])
            if np.isfinite(value) and abs(value) >= threshold:
                rows.append(
                    {
                        "kpi_a": a,
                        "kpi_b": b,
                        "spearman_corr": value,
                        "direction": "positive" if value > 0 else "negative",
                        "abs_corr": abs(value),
                    }
                )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["abs_corr", "kpi_a", "kpi_b"], ascending=[False, True, True])
    return out


def aggregate_by_label(df: pd.DataFrame) -> pd.DataFrame:
    agg_map = {kpi: "mean" for kpi in ANALYSIS_KPIS}
    out = (
        df.groupby("label", as_index=False)
        .agg(agg_map)
        .sort_values("label")
        .reset_index(drop=True)
    )
    return out


def evaluate_against_b0r(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for keys, g in df.groupby(["seed", "window_id", "time_band"]):
        b0r = g[g["label"] == "B0R"]
        if b0r.empty:
            continue

        b0r_row = b0r.iloc[0]

        for _, row in g.iterrows():
            label = str(row["label"])
            if label == "B0R":
                continue

            avg_wait_ratio = float(row["avg_wait_seconds"] / max(float(b0r_row["avg_wait_seconds"]), 1e-9))
            service_rate_ratio = float(row["passenger_service_rate"] / max(float(b0r_row["passenger_service_rate"]), 1e-9))
            on_time_drop_pp = float((b0r_row["on_time_rate"] - row["on_time_rate"]) * 100.0)
            p95_wait_ratio = float(row["passenger_wait_p95_seconds"] / max(float(b0r_row["passenger_wait_p95_seconds"]), 1e-9))
            bunching_increase_pp = float((row["bunching_rate"] - b0r_row["bunching_rate"]) * 100.0)
            energy_reduction = float(row["energy_proxy"] < b0r_row["energy_proxy"])

            pass_avg_wait = avg_wait_ratio <= SUCCESS_CONSTRAINTS["max_avg_wait_increase_ratio_vs_b0r"]
            pass_service = service_rate_ratio >= SUCCESS_CONSTRAINTS["min_passenger_service_rate_ratio_vs_b0r"]
            pass_on_time = on_time_drop_pp <= SUCCESS_CONSTRAINTS["max_on_time_rate_drop_percentage_points_vs_b0r"]
            pass_p95 = p95_wait_ratio <= SUCCESS_CONSTRAINTS["max_p95_wait_increase_ratio_vs_b0r"]
            pass_bunching = bunching_increase_pp <= SUCCESS_CONSTRAINTS["max_bunching_rate_increase_percentage_points_vs_b0r"]
            pass_energy = bool(energy_reduction)

            success = all([
                pass_avg_wait,
                pass_service,
                pass_on_time,
                pass_p95,
                pass_bunching,
                pass_energy,
            ])

            rows.append(
                {
                    "seed": keys[0],
                    "window_id": keys[1],
                    "time_band": keys[2],
                    "label": label,
                    "fleet_reduction_ratio": float(row["fleet_reduction_ratio"]),
                    "avg_wait_ratio_vs_b0r": avg_wait_ratio,
                    "service_rate_ratio_vs_b0r": service_rate_ratio,
                    "on_time_drop_pp_vs_b0r": on_time_drop_pp,
                    "p95_wait_ratio_vs_b0r": p95_wait_ratio,
                    "bunching_increase_pp_vs_b0r": bunching_increase_pp,
                    "energy_proxy_delta_vs_b0r": float(row["energy_proxy"] - b0r_row["energy_proxy"]),
                    "energy_proxy_per_passenger_delta_vs_b0r": float(
                        row["energy_proxy_per_passenger"] - b0r_row["energy_proxy_per_passenger"]
                    ),
                    "pass_avg_wait_constraint": pass_avg_wait,
                    "pass_service_rate_constraint": pass_service,
                    "pass_on_time_constraint": pass_on_time,
                    "pass_p95_wait_constraint": pass_p95,
                    "pass_bunching_constraint": pass_bunching,
                    "pass_energy_reduction_constraint": pass_energy,
                    "success_under_fleet_policy": success,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["time_band", "label", "window_id"]).reset_index(drop=True)
    return out


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple Pareto frontier over label-level means.

    Objectives:
    - minimize avg_wait_seconds
    - minimize passenger_wait_p95_seconds
    - minimize energy_proxy
    - maximize passenger_service_rate
    - maximize fleet_reduction_ratio
    """
    label_df = aggregate_by_label(df)

    objectives = [
        ("avg_wait_seconds", "min"),
        ("passenger_wait_p95_seconds", "min"),
        ("energy_proxy", "min"),
        ("passenger_service_rate", "max"),
        ("fleet_reduction_ratio", "max"),
    ]

    rows = []
    for i, row_i in label_df.iterrows():
        dominated = False

        for j, row_j in label_df.iterrows():
            if i == j:
                continue

            no_worse_all = True
            strictly_better_any = False

            for col, direction in objectives:
                vi = float(row_i[col])
                vj = float(row_j[col])

                if direction == "min":
                    if vj > vi:
                        no_worse_all = False
                        break
                    if vj < vi:
                        strictly_better_any = True
                else:
                    if vj < vi:
                        no_worse_all = False
                        break
                    if vj > vi:
                        strictly_better_any = True

            if no_worse_all and strictly_better_any:
                dominated = True
                break

        item = row_i.to_dict()
        item["pareto_frontier"] = not dominated
        rows.append(item)

    out = pd.DataFrame(rows)
    out = out.sort_values(["pareto_frontier", "label"], ascending=[False, True])
    return out



def df_to_markdown_safe(df: pd.DataFrame) -> str:
    """
    Render a DataFrame for markdown reports without requiring pandas' optional
    tabulate dependency. If tabulate is unavailable, fall back to a fenced
    plain-text table.
    """
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def write_markdown_report(
    *,
    df: pd.DataFrame,
    label_summary: pd.DataFrame,
    b0r_eval: pd.DataFrame,
    strong_corr: pd.DataFrame,
    pareto: pd.DataFrame,
    out_path: Path,
) -> None:
    lines: List[str] = []

    lines.append("# Step 20 KPI Relationship Analysis Report\n")
    lines.append("\n")
    lines.append("## Scope\n")
    lines.append("\n")
    lines.append(
        "This report analyzes KPI relationships after Step 19 shared demand generation fairness was fixed. "
        "It is a deterministic post-hoc evaluation tool. It does not alter policies, rewards, or simulator state.\n"
    )
    lines.append("\n")
    lines.append("## Analyzer vs Qwen role separation\n")
    lines.append("\n")
    lines.append(
        "- KPI analyzer: computes correlations, trade-offs, B0R-relative constraints, and Pareto candidates.\n"
    )
    lines.append(
        "- Qwen: may later read these outputs to generate explanations, hypotheses, and experiment recommendations.\n"
    )
    lines.append(
        "- Qwen must not rewrite official KPI values or perform post-hoc tuning of evaluation results.\n"
    )
    lines.append("\n")

    lines.append("## Label-level KPI means\n\n")
    lines.append(df_to_markdown_safe(label_summary.round(6)))
    lines.append("\n\n")

    lines.append("## B0R-relative service-quality constraint evaluation\n\n")
    if b0r_eval.empty:
        lines.append("No B0R-relative rows were available.\n\n")
    else:
        display_cols = [
            "time_band",
            "label",
            "fleet_reduction_ratio",
            "avg_wait_ratio_vs_b0r",
            "service_rate_ratio_vs_b0r",
            "on_time_drop_pp_vs_b0r",
            "p95_wait_ratio_vs_b0r",
            "energy_proxy_delta_vs_b0r",
            "success_under_fleet_policy",
        ]
        lines.append(df_to_markdown_safe(b0r_eval[display_cols].round(6)))
        lines.append("\n\n")

    lines.append("## Strong Spearman correlations\n\n")
    if strong_corr.empty:
        lines.append("No absolute Spearman correlations above the threshold were found.\n\n")
    else:
        lines.append(df_to_markdown_safe(strong_corr.round(6)))
        lines.append("\n\n")

    lines.append("## Pareto frontier candidates\n\n")
    pareto_cols = [
        "label",
        "pareto_frontier",
        "avg_wait_seconds",
        "passenger_wait_p95_seconds",
        "energy_proxy",
        "passenger_service_rate",
        "fleet_reduction_ratio",
    ]
    lines.append(df_to_markdown_safe(pareto[pareto_cols].round(6)))
    lines.append("\n\n")

    lines.append("## Interpretation caution\n\n")
    lines.append(
        "Correlation is not causation. Causal interpretation must come from matched-condition comparisons "
        "under the same initial state, same exogenous demand, same seed, and same evaluation window.\n"
    )

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze KPI relationships and trade-offs.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--strong-corr-threshold", type=float, default=0.70)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_input(input_path)

    corr_paths = write_correlation_matrices(df, out_dir)

    label_summary = aggregate_by_label(df)
    label_summary_path = out_dir / "kpi_label_summary.csv"
    label_summary.to_csv(label_summary_path, index=False, encoding="utf-8-sig")

    strong_corr = summarize_strong_correlations(df, threshold=args.strong_corr_threshold)
    strong_corr_path = out_dir / "kpi_strong_spearman_correlations.csv"
    strong_corr.to_csv(strong_corr_path, index=False, encoding="utf-8-sig")

    b0r_eval = evaluate_against_b0r(df)
    b0r_eval_path = out_dir / "b0r_relative_service_constraints.csv"
    b0r_eval.to_csv(b0r_eval_path, index=False, encoding="utf-8-sig")

    pareto = pareto_frontier(df)
    pareto_path = out_dir / "kpi_pareto_frontier.csv"
    pareto.to_csv(pareto_path, index=False, encoding="utf-8-sig")

    report_path = out_dir / "kpi_relationship_report.md"
    write_markdown_report(
        df=df,
        label_summary=label_summary,
        b0r_eval=b0r_eval,
        strong_corr=strong_corr,
        pareto=pareto,
        out_path=report_path,
    )

    summary = {
        "status": "step20_kpi_relationship_analysis_pass",
        "input": str(input_path),
        "output_dir": str(out_dir),
        "row_count": int(len(df)),
        "labels": sorted(df["label"].astype(str).unique().tolist()),
        "time_bands": sorted(df["time_band"].astype(str).unique().tolist()),
        "analysis_kpis": ANALYSIS_KPIS,
        "strong_corr_threshold": float(args.strong_corr_threshold),
        "outputs": {
            **corr_paths,
            "label_summary_csv": str(label_summary_path),
            "strong_spearman_correlations_csv": str(strong_corr_path),
            "b0r_relative_service_constraints_csv": str(b0r_eval_path),
            "pareto_frontier_csv": str(pareto_path),
            "markdown_report": str(report_path),
        },
        "qwen_role_boundary": {
            "kpi_analyzer_role": "deterministic post-hoc numeric analysis",
            "qwen_allowed_role": "interpretation, hypothesis generation, experiment recommendation",
            "qwen_not_allowed_role": "rewriting official KPI values or post-hoc tuning evaluation outputs",
        },
    }

    summary_path = out_dir / "kpi_relationship_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("=== Step 20 KPI label summary ===")
    print(label_summary.round(6).to_string(index=False))

    print("")
    print("=== Step 20 B0R-relative constraint evaluation ===")
    if b0r_eval.empty:
        print("[WARN] no B0R-relative rows found")
    else:
        print(
            b0r_eval[
                [
                    "time_band",
                    "label",
                    "fleet_reduction_ratio",
                    "avg_wait_ratio_vs_b0r",
                    "service_rate_ratio_vs_b0r",
                    "on_time_drop_pp_vs_b0r",
                    "p95_wait_ratio_vs_b0r",
                    "energy_proxy_delta_vs_b0r",
                    "success_under_fleet_policy",
                ]
            ]
            .round(6)
            .to_string(index=False)
        )

    print("")
    print("=== Step 20 Pareto frontier ===")
    print(
        pareto[
            [
                "label",
                "pareto_frontier",
                "avg_wait_seconds",
                "passenger_wait_p95_seconds",
                "energy_proxy",
                "passenger_service_rate",
                "fleet_reduction_ratio",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    print("")
    print(f"[OK] wrote label summary    : {label_summary_path}")
    print(f"[OK] wrote Pearson corr     : {corr_paths['pearson_csv']}")
    print(f"[OK] wrote Spearman corr    : {corr_paths['spearman_csv']}")
    print(f"[OK] wrote strong corr      : {strong_corr_path}")
    print(f"[OK] wrote B0R constraints  : {b0r_eval_path}")
    print(f"[OK] wrote Pareto frontier  : {pareto_path}")
    print(f"[OK] wrote report           : {report_path}")
    print(f"[OK] wrote summary JSON     : {summary_path}")
    print("STEP 20 KPI RELATIONSHIP ANALYZER PASS")


if __name__ == "__main__":
    main()
