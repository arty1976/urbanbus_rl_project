import json
from pathlib import Path
import pandas as pd


EXPECTED_KPIS = [
    "cv_headway",
    "avg_wait_seconds",
    "bunching_rate",
    "on_time_rate",
    "intervention_rate",
    "energy_proxy",
]

REQUIRED_FILES = [
    "kpi_by_window.parquet",
    "kpi_by_seed.parquet",
    "kpi_by_time_band.parquet",
    "kpi_overall.json",
    "aggregation_manifest.json",
]

FULL_YEAR_EXPECTED = {
    "B0": 6570,
    "B1": 19710,
    "B2": 19710,
    "A": 19710,
}

CASES = [
    {
        "name": "B0_historical",
        "condition_id": "B0",
        "root": Path("artifacts/baseline_v1/B0_historical/canonical_eval"),
        "strict_canonical": False,
        "needs_validation_json": False,
    },
    {
        "name": "B1_noop",
        "condition_id": "B1",
        "root": Path("artifacts/baseline_v1/B1_noop/canonical_eval"),
        "strict_canonical": True,
        "needs_validation_json": True,
    },
    {
        "name": "B2_rulebased",
        "condition_id": "B2",
        "root": Path("artifacts/baseline_v1/B2_rulebased/canonical_eval"),
        "strict_canonical": True,
        "needs_validation_json": True,
    },
    {
        "name": "A_pure_mappo",
        "condition_id": "A",
        "root": Path("artifacts/experiment_A_v1/canonical_eval"),
        "strict_canonical": True,
        "needs_validation_json": True,
    },
]


class CheckError(RuntimeError):
    pass


def require(ok, msg):
    if not ok:
        raise CheckError(msg)


def load_json(path):
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise CheckError(f"failed to read json: {path}")


def dump_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_bool_values(series):
    values = []
    for value in series.dropna().tolist():
        text = str(value).strip().lower()
        if text in ("true", "1"):
            values.append(True)
        elif text in ("false", "0"):
            values.append(False)
        else:
            raise CheckError(f"cannot parse bool value: {value}")
    return sorted(set(values))


def check_case(case):
    name = case["name"]
    root = case["root"]
    condition_id = case["condition_id"]
    warnings = []

    require(root.exists(), f"{name}: canonical_eval folder not found: {root}")

    required_files = list(REQUIRED_FILES)
    if case["needs_validation_json"]:
        required_files.append("official_rollup_input_validation.json")

    missing = [x for x in required_files if not (root / x).exists()]
    require(not missing, f"{name}: missing files: {missing}")

    window_df = pd.read_parquet(root / "kpi_by_window.parquet")
    seed_df = pd.read_parquet(root / "kpi_by_seed.parquet")
    time_band_df = pd.read_parquet(root / "kpi_by_time_band.parquet")
    overall = load_json(root / "kpi_overall.json")
    manifest = load_json(root / "aggregation_manifest.json")

    require(len(window_df) > 0, f"{name}: kpi_by_window is empty")
    require(len(seed_df) > 0, f"{name}: kpi_by_seed is empty")
    require(len(time_band_df) > 0, f"{name}: kpi_by_time_band is empty")

    observed_conditions = sorted(window_df["condition_id"].astype(str).str.upper().unique().tolist())
    require(
        observed_conditions == [condition_id],
        f"{name}: condition_id mismatch. expected={[condition_id]} got={observed_conditions}",
    )

    require("evaluation_horizon_minutes" in window_df.columns, f"{name}: missing evaluation_horizon_minutes")
    require(bool((window_df["evaluation_horizon_minutes"] == 30).all()), f"{name}: horizon must be 30")

    observed_time_bands = sorted(window_df["time_band"].astype(str).str.lower().unique().tolist())
    require(
        set(observed_time_bands).issubset({"peak", "offpeak", "night"}),
        f"{name}: invalid time_band values: {observed_time_bands}",
    )

    for kpi in EXPECTED_KPIS:
        require(kpi in window_df.columns, f"{name}: missing KPI column: {kpi}")

    require("strict_canonical" in window_df.columns, f"{name}: missing strict_canonical")
    strict_values = parse_bool_values(window_df["strict_canonical"])
    require(
        strict_values == [case["strict_canonical"]],
        f"{name}: strict_canonical mismatch. expected={[case['strict_canonical']]} got={strict_values}",
    )

    require("causal_comparison_allowed" in window_df.columns, f"{name}: missing causal_comparison_allowed")
    causal_values = parse_bool_values(window_df["causal_comparison_allowed"])
    require(
        causal_values == [False],
        f"{name}: causal_comparison_allowed must be false now. got={causal_values}",
    )

    manifest_counts = manifest.get("row_counts", {})
    require(
        int(manifest_counts.get("kpi_by_window", -1)) == len(window_df),
        f"{name}: manifest window count mismatch. manifest={manifest_counts.get('kpi_by_window')} actual={len(window_df)}",
    )
    require(
        int(manifest_counts.get("kpi_by_seed", -1)) == len(seed_df),
        f"{name}: manifest seed count mismatch. manifest={manifest_counts.get('kpi_by_seed')} actual={len(seed_df)}",
    )
    require(
        int(manifest_counts.get("kpi_by_time_band", -1)) == len(time_band_df),
        f"{name}: manifest time_band count mismatch. manifest={manifest_counts.get('kpi_by_time_band')} actual={len(time_band_df)}",
    )

    require(
        int(overall.get("n_windows_total", -1)) == len(window_df),
        f"{name}: overall n_windows_total mismatch. overall={overall.get('n_windows_total')} actual={len(window_df)}",
    )

    input_validation_passed = None
    if case["needs_validation_json"]:
        validation = load_json(root / "official_rollup_input_validation.json")
        input_validation_passed = bool(validation.get("smoke", {}).get("passed", False))
        require(input_validation_passed, f"{name}: input validation smoke not passed")

    full_expected = FULL_YEAR_EXPECTED.get(condition_id)
    full_year_complete = bool(full_expected is not None and len(window_df) == full_expected)

    if full_expected is not None and len(window_df) != full_expected:
        warnings.append({
            "code": "partial_or_smoke_window_count",
            "message": (
                f"{name}: current rows={len(window_df)}, full-year expected rows={full_expected}. "
                "Allowed for smoke/limited validation."
            ),
        })

    return {
        "name": name,
        "condition_id": condition_id,
        "root": str(root),
        "window_rows": int(len(window_df)),
        "seed_rows": int(len(seed_df)),
        "time_band_rows": int(len(time_band_df)),
        "time_bands": observed_time_bands,
        "strict_canonical": case["strict_canonical"],
        "causal_comparison_allowed": False,
        "input_validation_passed": input_validation_passed,
        "full_year_expected_rows": full_expected,
        "full_year_complete": full_year_complete,
        "warnings": warnings,
        "kpi_valid_counts": {
            kpi: int(pd.to_numeric(window_df[kpi], errors="coerce").notna().sum())
            for kpi in EXPECTED_KPIS
        },
    }


def main():
    print(f"[OK] project_root: {Path.cwd()}")

    summaries = []
    warning_count = 0

    for case in CASES:
        summary = check_case(case)
        summaries.append(summary)
        warning_count += len(summary["warnings"])

        print(
            "[OK] {name:14s} condition={condition_id:2s} "
            "window_rows={window_rows:5d} seed_rows={seed_rows:2d} "
            "time_band_rows={time_band_rows:2d} strict={strict_canonical} "
            "causal={causal_comparison_allowed} full_year={full_year_complete}".format(**summary)
        )

        for warning in summary["warnings"]:
            print(f"[WARN] {warning['code']}: {warning['message']}")

    payload = {
        "artifact_version": "canonical_eval_summary_v1",
        "status": "PASS",
        "case_count": len(summaries),
        "warning_count": int(warning_count),
        "cases": summaries,
        "interpretation": (
            "All canonical outputs exist and match their manifests. "
            "causal_comparison_allowed=false means these are not causal performance comparison results yet. "
            "full_year_complete=false is allowed for smoke or limited validation."
        ),
    }

    out_path = Path("artifacts/canonical_eval_summary.json")
    dump_json(out_path, payload)

    print(f"[OK] summary_json: {out_path}")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()