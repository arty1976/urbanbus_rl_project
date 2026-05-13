from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0c-root", required=True)
    parser.add_argument("--expected-seeds", default="1,2,3")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    root = Path(args.b0c_root)
    expected_seeds = [int(x.strip()) for x in args.expected_seeds.split(",") if x.strip()]

    files = sorted(root.glob("rollouts/seed_*/window_rollup.parquet"))
    raw_files = sorted(root.glob("rollouts/seed_*/raw_events.parquet"))

    failures = []

    if len(files) != len(expected_seeds):
        failures.append(f"expected {len(expected_seeds)} window_rollup files, got {len(files)}")

    if len(raw_files) != len(expected_seeds):
        failures.append(f"expected {len(expected_seeds)} raw_events files, got {len(raw_files)}")

    frames = []
    for p in files:
        df = pd.read_parquet(p).copy()
        df["input_file"] = str(p)
        frames.append(df)

    if not frames:
        raise SystemExit("[FAIL] no B0C window_rollup files found")

    all_df = pd.concat(frames, ignore_index=True)

    for kpi in KPI_12:
        if kpi not in all_df.columns:
            failures.append(f"missing KPI column: {kpi}")

    if set(all_df["condition_id"].astype(str).unique()) != {"B0C"}:
        failures.append("condition_id must be B0C only")

    if "qwen_trigger_rate" in all_df.columns:
        if not bool((pd.to_numeric(all_df["qwen_trigger_rate"], errors="coerce") == 0.0).all()):
            failures.append("qwen_trigger_rate must be 0.0")

    if "intervention_rate" in all_df.columns:
        if not bool((pd.to_numeric(all_df["intervention_rate"], errors="coerce") == 0.0).all()):
            failures.append("B0C intervention_rate must be 0.0")

    if "active_bus_ratio" in all_df.columns:
        if not bool((pd.to_numeric(all_df["active_bus_ratio"], errors="coerce") == 1.0).all()):
            failures.append("active_bus_ratio must be 1.0")

    if "fleet_reduction_ratio" in all_df.columns:
        if not bool((pd.to_numeric(all_df["fleet_reduction_ratio"], errors="coerce") == 0.0).all()):
            failures.append("fleet_reduction_ratio must be 0.0")

    if "causal_comparison_allowed" in all_df.columns:
        if not bool((all_df["causal_comparison_allowed"] == False).all()):
            failures.append("causal_comparison_allowed must remain false")

    if "paper_level_claim_allowed" in all_df.columns:
        if not bool((all_df["paper_level_claim_allowed"] == False).all()):
            failures.append("paper_level_claim_allowed must remain false")

    seed_values = sorted(pd.to_numeric(all_df["seed"], errors="coerce").dropna().astype(int).unique().tolist())
    if seed_values != expected_seeds:
        failures.append(f"seed mismatch expected={expected_seeds} got={seed_values}")

    payload = {
        "artifact_version": "b0c_causal_shadow_rollout_validation_v1",
        "audit_status": "FAIL" if failures else "PASS",
        "b0c_root": str(root),
        "window_rollup_files": [str(p) for p in files],
        "raw_event_files": [str(p) for p in raw_files],
        "window_rows": int(len(all_df)),
        "seed_values": seed_values,
        "condition_ids": sorted(all_df["condition_id"].astype(str).unique().tolist()),
        "kpi_schema": KPI_12,
        "claim_guards": {
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "actual_results": False,
        },
        "hard_failures": failures,
    }

    out = Path(args.output_json)
    dump_json(out, payload)

    print("[OK] B0C rollout validation completed")
    print("[OK] audit_status:", payload["audit_status"])
    print("[OK] window_rows:", payload["window_rows"])
    print("[OK] output_json:", out)

    if failures:
        for f in failures:
            print("[FAIL]", f)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
