from __future__ import annotations

import argparse
import json
import math
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


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def get_kpi_mean(overall: Dict[str, Any], kpi: str) -> Optional[float]:
    return safe_float(overall.get("kpis", {}).get(kpi, {}).get("mean"))


def get_series_mean(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def add_check(
    rows: List[Dict[str, Any]],
    name: str,
    status: str,
    severity: str,
    observed: Any,
    expected: Any,
    message: str,
) -> None:
    rows.append({
        "check_name": name,
        "status": status,
        "severity": severity,
        "observed": observed,
        "expected": expected,
        "message": message,
    })


def check_range(
    rows: List[Dict[str, Any]],
    name: str,
    value: Optional[float],
    min_value: Optional[float],
    max_value: Optional[float],
    severity: str,
) -> None:
    if value is None:
        add_check(rows, name, "FAIL", severity, value, f"[{min_value}, {max_value}]", "value is missing or non-finite")
        return

    ok = True
    if min_value is not None and value < min_value:
        ok = False
    if max_value is not None and value > max_value:
        ok = False

    add_check(
        rows,
        name,
        "PASS" if ok else "FAIL",
        severity,
        value,
        f"[{min_value}, {max_value}]",
        "within tolerance" if ok else "outside tolerance",
    )


def check_exact(
    rows: List[Dict[str, Any]],
    name: str,
    value: Optional[float],
    expected: float,
    severity: str,
    atol: float = 1e-12,
) -> None:
    ok = value is not None and abs(value - expected) <= atol
    add_check(
        rows,
        name,
        "PASS" if ok else "FAIL",
        severity,
        value,
        expected,
        "exact match" if ok else "exact value mismatch",
    )


def load_baseline_root(root: Path, label: str) -> Dict[str, Any]:
    overall_path = root / "kpi_overall.json"
    window_path = root / "kpi_by_window.parquet"
    manifest_path = root / "aggregation_manifest.json"

    missing = []
    if not overall_path.exists():
        missing.append(str(overall_path))
    if not window_path.exists():
        missing.append(str(window_path))

    if missing:
        raise FileNotFoundError(f"{label} missing required files: {missing}")

    return {
        "overall_path": overall_path,
        "window_path": window_path,
        "manifest_path": manifest_path if manifest_path.exists() else None,
        "overall": load_json(overall_path),
        "window": pd.read_parquet(window_path).copy(),
        "manifest": load_json(manifest_path) if manifest_path.exists() else {},
    }


def collect_qwen_trigger_rate(b0c_canonical_df: pd.DataFrame, b0c_canonical_root: Path) -> Optional[float]:
    if "qwen_trigger_rate" in b0c_canonical_df.columns:
        return get_series_mean(b0c_canonical_df, "qwen_trigger_rate")

    # Fallback to original rollout files if canonical aggregation did not preserve the metadata.
    rollout_root = b0c_canonical_root.parent
    files = sorted(rollout_root.glob("rollouts/seed_*/window_rollup.parquet"))
    frames = []
    for p in files:
        df = pd.read_parquet(p)
        if "qwen_trigger_rate" in df.columns:
            frames.append(df[["qwen_trigger_rate"]].copy())

    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    return get_series_mean(merged, "qwen_trigger_rate")


def build_markdown(payload: Dict[str, Any], checks_df: pd.DataFrame, kpi_df: pd.DataFrame) -> str:
    lines: List[str] = []

    lines.append("# B0C Simulator Validation Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- audit_status: {payload['audit_status']}")
    lines.append(f"- hard_failures: {len(payload['hard_failures'])}")
    lines.append(f"- warnings: {len(payload['warnings'])}")
    lines.append("")
    lines.append("## Claim Guard")
    lines.append("")
    lines.append("| Guard | Value |")
    lines.append("|---|---|")
    for key, value in payload["claim_guards"].items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    lines.append("## KPI Means")
    lines.append("")
    lines.append("| KPI | B0C | B0R | B1 | B2 calibrated |")
    lines.append("|---|---:|---:|---:|---:|")

    for _, row in kpi_df.iterrows():
        def fmt(x):
            if pd.isna(x):
                return "null"
            try:
                v = float(x)
            except Exception:
                return str(x)
            if abs(v) >= 100:
                return f"{v:.2f}"
            return f"{v:.4f}"

        lines.append(
            f"| {row['kpi']} | {fmt(row['b0c_mean'])} | {fmt(row['b0r_mean'])} | "
            f"{fmt(row['b1_mean'])} | {fmt(row['b2_mean'])} |"
        )

    lines.append("")
    lines.append("## Validation Checks")
    lines.append("")
    lines.append("| Check | Status | Severity | Observed | Expected |")
    lines.append("|---|---|---|---:|---|")
    for _, row in checks_df.iterrows():
        lines.append(
            f"| {row['check_name']} | {row['status']} | {row['severity']} | "
            f"{row['observed']} | {row['expected']} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- PASS means B0C satisfies the scaffold-level simulator sanity checks, not real-world proof.")
    lines.append("- PASS_WITH_WARNINGS means B0C is structurally usable as a validation candidate, but calibration gaps remain.")
    lines.append("- FAIL means a guard, schema, structural, or hard tolerance rule failed.")
    lines.append("- causal_comparison_allowed must remain false until a later explicit release gate.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tolerance-contract",
        default="05_training/baselines/b0c_simulator_validation_tolerance_contract.json",
    )
    parser.add_argument(
        "--b0c-root",
        default="artifacts/baseline_v1/B0C_causal_shadow_v1/canonical_eval",
    )
    parser.add_argument(
        "--b0r-root",
        default="artifacts/baseline_v1/B0R_historical_12kpi_compat",
    )
    parser.add_argument(
        "--b1-root",
        default="artifacts/baseline_v1/B1_noop/canonical_eval",
    )
    parser.add_argument(
        "--b2-root",
        default="artifacts/baseline_v1/B2_rulebased_calibrated/canonical_eval",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/baseline_v1/B0C_causal_shadow_v1/simulator_validation",
    )
    args = parser.parse_args()

    tolerance_contract_path = Path(args.tolerance_contract)
    b0c_root = Path(args.b0c_root)
    b0r_root = Path(args.b0r_root)
    b1_root = Path(args.b1_root)
    b2_root = Path(args.b2_root)
    output_root = Path(args.output_root)

    checks: List[Dict[str, Any]] = []
    hard_failures: List[str] = []
    warnings: List[str] = []

    if not tolerance_contract_path.exists():
        raise SystemExit(f"[FAIL] missing tolerance contract: {tolerance_contract_path}")

    contract = load_json(tolerance_contract_path)

    b0c = load_baseline_root(b0c_root, "B0C")
    b0r = load_baseline_root(b0r_root, "B0R")
    b1 = load_baseline_root(b1_root, "B1")
    b2 = load_baseline_root(b2_root, "B2 calibrated")

    b0c_df = b0c["window"]
    b0c_overall = b0c["overall"]
    b0r_overall = b0r["overall"]
    b1_overall = b1["overall"]
    b2_overall = b2["overall"]

    required = contract.get("required_structure", {})
    expected_rows = int(required.get("expected_window_rows", 19710))
    expected_seeds = [int(x) for x in required.get("expected_seeds", [1, 2, 3])]
    expected_windows_per_seed = int(required.get("expected_windows_per_seed", 6570))
    expected_time_bands = sorted(str(x).lower() for x in required.get("expected_time_bands", ["peak", "offpeak", "night"]))

    # Structural checks.
    add_check(
        checks,
        "condition_id_is_B0C",
        "PASS" if set(b0c_df["condition_id"].astype(str).unique()) == {"B0C"} else "FAIL",
        "hard_fail",
        sorted(b0c_df["condition_id"].astype(str).unique().tolist()),
        ["B0C"],
        "B0C condition_id check",
    )

    add_check(
        checks,
        "window_rows",
        "PASS" if len(b0c_df) == expected_rows else "FAIL",
        "hard_fail",
        len(b0c_df),
        expected_rows,
        "B0C canonical window row count",
    )

    seed_values = sorted(pd.to_numeric(b0c_df["seed"], errors="coerce").dropna().astype(int).unique().tolist())
    add_check(
        checks,
        "seed_values",
        "PASS" if seed_values == expected_seeds else "FAIL",
        "hard_fail",
        seed_values,
        expected_seeds,
        "B0C seed set",
    )

    per_seed = b0c_df.groupby("seed").size().to_dict()
    per_seed_ok = all(int(per_seed.get(s, -1)) == expected_windows_per_seed for s in expected_seeds)
    add_check(
        checks,
        "windows_per_seed",
        "PASS" if per_seed_ok else "FAIL",
        "hard_fail",
        per_seed,
        {s: expected_windows_per_seed for s in expected_seeds},
        "B0C windows per seed",
    )

    time_bands = sorted(str(x).lower() for x in b0c_df["time_band"].dropna().unique().tolist())
    add_check(
        checks,
        "time_bands",
        "PASS" if time_bands == expected_time_bands else "FAIL",
        "hard_fail",
        time_bands,
        expected_time_bands,
        "B0C time-band set",
    )

    missing_kpis = [k for k in KPI_12 if k not in b0c_df.columns]
    add_check(
        checks,
        "kpi_12_schema_present",
        "PASS" if not missing_kpis else "FAIL",
        "hard_fail",
        missing_kpis,
        [],
        "All 12 KPI columns must exist",
    )

    # Claim guard checks.
    b0c_causal_allowed = bool(b0c_overall.get("causal_comparison_allowed", False))
    add_check(
        checks,
        "b0c_causal_comparison_allowed_false",
        "PASS" if b0c_causal_allowed is False else "FAIL",
        "hard_fail",
        b0c_causal_allowed,
        False,
        "B0C must remain non-claim until explicit release",
    )

    manifest_causal_allowed = bool(b0c["manifest"].get("causal_comparison_allowed", False))
    add_check(
        checks,
        "b0c_manifest_causal_comparison_allowed_false",
        "PASS" if manifest_causal_allowed is False else "FAIL",
        "hard_fail",
        manifest_causal_allowed,
        False,
        "B0C manifest must remain non-claim until explicit release",
    )

    # KPI means table.
    kpi_rows = []
    for kpi in KPI_12:
        kpi_rows.append({
            "kpi": kpi,
            "b0c_mean": get_kpi_mean(b0c_overall, kpi),
            "b0r_mean": get_kpi_mean(b0r_overall, kpi),
            "b1_mean": get_kpi_mean(b1_overall, kpi),
            "b2_mean": get_kpi_mean(b2_overall, kpi),
        })
    kpi_df = pd.DataFrame(kpi_rows)

    b0c_mean = {row["kpi"]: safe_float(row["b0c_mean"]) for _, row in kpi_df.iterrows()}
    b1_mean = {row["kpi"]: safe_float(row["b1_mean"]) for _, row in kpi_df.iterrows()}
    b2_mean = {row["kpi"]: safe_float(row["b2_mean"]) for _, row in kpi_df.iterrows()}

    # B1/B2 scale-band tolerance.
    rules = contract.get("tolerance_rules", {})

    for kpi in ["avg_wait_seconds", "cv_headway"]:
        rule = rules.get(kpi, {})
        refs = [x for x in [b1_mean.get(kpi), b2_mean.get(kpi)] if x is not None]
        value = b0c_mean.get(kpi)

        if len(refs) < 2 or value is None:
            add_check(checks, f"{kpi}_reference_scale_available", "FAIL", "warning", value, "B1/B2 means available", "missing reference or B0C value")
            continue

        lower = min(refs) * float(rule.get("lower_margin_multiplier", 0.7))
        upper = max(refs) * float(rule.get("upper_margin_multiplier", 1.3))
        check_range(checks, f"{kpi}_within_B1_B2_scale_band", value, lower, upper, rule.get("severity_if_outside", "warning"))

    # Direct range/exact tolerance checks.
    check_range(
        checks,
        "bunching_rate_range",
        b0c_mean.get("bunching_rate"),
        safe_float(rules.get("bunching_rate", {}).get("min")),
        safe_float(rules.get("bunching_rate", {}).get("max")),
        rules.get("bunching_rate", {}).get("severity_if_outside", "warning"),
    )

    check_range(
        checks,
        "on_time_rate_range",
        b0c_mean.get("on_time_rate"),
        safe_float(rules.get("on_time_rate", {}).get("min")),
        safe_float(rules.get("on_time_rate", {}).get("max")),
        rules.get("on_time_rate", {}).get("severity_if_outside", "warning"),
    )

    check_exact(
        checks,
        "intervention_rate_exact_zero",
        b0c_mean.get("intervention_rate"),
        0.0,
        rules.get("intervention_rate", {}).get("severity_if_outside", "hard_fail"),
    )

    check_range(
        checks,
        "passenger_service_rate_range",
        b0c_mean.get("passenger_service_rate"),
        safe_float(rules.get("passenger_service_rate", {}).get("min")),
        safe_float(rules.get("passenger_service_rate", {}).get("max")),
        rules.get("passenger_service_rate", {}).get("severity_if_outside", "hard_fail"),
    )

    avg_wait = b0c_mean.get("avg_wait_seconds")
    p95_wait = b0c_mean.get("passenger_wait_p95_seconds")
    if avg_wait is None or p95_wait is None:
        add_check(checks, "passenger_wait_p95_sanity", "FAIL", "warning", p95_wait, "p95 > avg_wait", "missing avg or p95")
    else:
        ratio = p95_wait / avg_wait if avg_wait > 0 else None
        p95_rule = rules.get("passenger_wait_p95_seconds", {})
        max_ratio = float(p95_rule.get("max_ratio_to_avg_wait", 2.5))
        ok = p95_wait > avg_wait and ratio is not None and ratio <= max_ratio
        add_check(
            checks,
            "passenger_wait_p95_sanity",
            "PASS" if ok else "FAIL",
            p95_rule.get("severity_if_outside", "warning"),
            {"p95": p95_wait, "avg": avg_wait, "ratio": ratio},
            f"p95 > avg_wait and ratio <= {max_ratio}",
            "p95 wait sanity",
        )

    energy_pp = b0c_mean.get("energy_proxy_per_passenger")
    energy_ok = energy_pp is not None and energy_pp > 0
    add_check(
        checks,
        "energy_proxy_per_passenger_positive_finite",
        "PASS" if energy_ok else "FAIL",
        rules.get("energy_proxy_per_passenger", {}).get("severity_if_outside", "hard_fail"),
        energy_pp,
        "> 0 and finite",
        "energy per passenger sanity",
    )

    check_exact(
        checks,
        "fleet_reduction_ratio_exact_zero",
        b0c_mean.get("fleet_reduction_ratio"),
        0.0,
        rules.get("fleet_reduction_ratio", {}).get("severity_if_outside", "hard_fail"),
    )

    qwen_rate = collect_qwen_trigger_rate(b0c_df, b0c_root)
    check_exact(
        checks,
        "qwen_trigger_rate_exact_zero",
        qwen_rate,
        0.0,
        rules.get("qwen_trigger_rate", {}).get("severity_if_outside", "hard_fail"),
    )

    # Observability guard.
    forbidden_observed_cols = [
        "actual_headway",
        "actual_arrival_departure_time",
        "actual_dwell",
        "passenger_level_observed_wait_distribution",
    ]

    for col in forbidden_observed_cols:
        if col in b0c_df.columns:
            non_null_count = int(b0c_df[col].notna().sum())
            add_check(
                checks,
                f"{col}_not_promoted",
                "PASS" if non_null_count == 0 else "FAIL",
                "hard_fail",
                non_null_count,
                0,
                f"{col} must not be promoted to observed value",
            )
        else:
            add_check(
                checks,
                f"{col}_not_promoted",
                "PASS",
                "hard_fail",
                "column_absent",
                "absent_or_all_null",
                f"{col} not present as observed field",
            )

    checks_df = pd.DataFrame(checks)

    for _, row in checks_df.iterrows():
        if row["status"] == "FAIL" and row["severity"] == "hard_fail":
            hard_failures.append(f"{row['check_name']}: {row['message']}")
        elif row["status"] == "FAIL":
            warnings.append(f"{row['check_name']}: {row['message']}")

    audit_status = "FAIL" if hard_failures else ("PASS_WITH_WARNINGS" if warnings else "PASS")

    output_root.mkdir(parents=True, exist_ok=True)

    checks_csv = output_root / "b0c_simulator_validation_checks.csv"
    kpi_csv = output_root / "b0c_simulator_validation_kpi_means.csv"
    json_path = output_root / "b0c_simulator_validation_report.json"
    md_path = output_root / "b0c_simulator_validation_report.md"

    checks_df.to_csv(checks_csv, index=False, encoding="utf-8-sig")
    kpi_df.to_csv(kpi_csv, index=False, encoding="utf-8-sig")

    payload = {
        "artifact_version": "b0c_simulator_validation_report_v1",
        "audit_status": audit_status,
        "baseline_id": "B0C_causal_shadow_v1",
        "condition_id": "B0C",
        "tolerance_contract": str(tolerance_contract_path),
        "inputs": {
            "b0c_root": str(b0c_root),
            "b0r_root": str(b0r_root),
            "b1_root": str(b1_root),
            "b2_root": str(b2_root),
        },
        "claim_guards": {
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "actual_results": False,
            "auto_promotion_allowed": False,
            "requires_later_explicit_release_gate": True,
        },
        "hard_failures": hard_failures,
        "warnings": warnings,
        "summary": {
            "b0c_windows": int(len(b0c_df)),
            "b0c_seeds": seed_values,
            "b0c_time_bands": time_bands,
            "qwen_trigger_rate": qwen_rate,
            "b0c_causal_comparison_allowed_in_overall": b0c_causal_allowed,
            "b0c_causal_comparison_allowed_in_manifest": manifest_causal_allowed,
        },
        "output_files": {
            "checks_csv": str(checks_csv),
            "kpi_means_csv": str(kpi_csv),
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "non_claim_statement": "This report is a simulator sanity validation report. It is not proof of real-world operational improvement.",
    }

    dump_json(json_path, payload)
    md_path.write_text(build_markdown(payload, checks_df, kpi_df), encoding="utf-8")

    print("[OK] B0C simulator validation report completed")
    print("[OK] audit_status:", audit_status)
    print("[OK] hard_failures:", len(hard_failures))
    print("[OK] warnings:", len(warnings))
    print("[OK] output_root:", output_root)
    print("[OK] report_json:", json_path)
    print("[OK] report_md:", md_path)
    print("[OK] checks_csv:", checks_csv)
    print("[OK] kpi_csv:", kpi_csv)

    if hard_failures:
        raise SystemExit(f"[FAIL] {hard_failures}")


if __name__ == "__main__":
    main()
