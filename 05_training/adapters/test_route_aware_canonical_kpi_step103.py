#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-test for Step 103 route-aware canonical KPI integration scaffold."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from route_aware_canonical_kpi_step103 import (
    Step103Paths,
    normalize_source_mode,
    run_step103,
)


MINI_CANONICAL_AGGREGATOR = r'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
import pandas as pd

EXPECTED_SHARED_KPIS = [
    "cv_headway", "avg_wait_seconds", "bunching_rate", "on_time_rate", "intervention_rate", "energy_proxy"
]


def dump_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def ratio(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = a / b
    return out.where(b > 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--input-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    with open(args.contract, "r", encoding="utf-8") as f:
        contract = json.load(f)
    if contract.get("shared_kpis") != EXPECTED_SHARED_KPIS:
        raise SystemExit("bad shared_kpis")

    in_path = Path(args.input_root) / "window_rollup.parquet"
    raw = pd.read_parquet(in_path)
    df = raw.copy()
    df["condition_id"] = df["condition_id"].astype(str).str.upper()
    df["seed"] = pd.to_numeric(df["seed"], errors="raise").astype(int)
    df["time_band"] = df["time_band"].astype(str).str.lower()
    df["cv_headway"] = ratio(df["headway_std_seconds"], df["headway_mean_seconds"])
    df["avg_wait_seconds"] = ratio(df["wait_total_passenger_seconds"], df["wait_passenger_count"])
    df["bunching_rate"] = ratio(df["bunching_event_count"], df["headway_event_count"])
    df["on_time_rate"] = ratio(df["ontime_event_count"], df["schedulable_arrival_count"])
    df["intervention_rate"] = ratio(df["intervention_count"], df["decision_step_count"])
    df["energy_proxy"] = pd.to_numeric(df["energy_proxy_total"], errors="coerce")
    df["strict_canonical"] = True
    df["computation_mode"] = "official_rollup"
    df["causal_comparison_allowed"] = ~df["source_mode"].astype(str).str.lower().str.contains(
        "historical|legacy|stub|replay|smoke|non_causal|non-causal", regex=True, na=False
    )
    df["qwen_trigger_rate"] = 0.0
    df["effective_replay_step_minutes"] = 60
    df["input_source_path"] = str(in_path)

    out_cols = [
        "condition_id", "seed", "window_id", "state_ts", "service_date", "time_band",
        "evaluation_horizon_minutes", *EXPECTED_SHARED_KPIS, "strict_canonical", "computation_mode",
        "causal_comparison_allowed", "source_mode", "qwen_trigger_rate", "effective_replay_step_minutes",
        "headway_mean_seconds", "headway_std_seconds", "headway_sample_count", "bunching_event_count",
        "headway_event_count", "wait_total_passenger_seconds", "wait_passenger_count", "ontime_event_count",
        "schedulable_arrival_count", "intervention_count", "decision_step_count", "energy_proxy_total",
        "input_source_path",
    ]
    window = df[out_cols].copy()
    seed = window.groupby(["condition_id", "seed"], as_index=False).agg(window_count=("window_id", "count"))
    time_band = window.groupby(["condition_id", "seed", "time_band"], as_index=False).agg(window_count=("window_id", "count"))
    overall = {
        "mode": "official_rollup",
        "n_windows_total": int(len(window)),
        "causal_comparison_allowed": bool(window["causal_comparison_allowed"].all()),
        "kpis": {k: {"valid_window_count": int(pd.to_numeric(window[k], errors="coerce").notna().sum())} for k in EXPECTED_SHARED_KPIS},
    }

    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "official_rollup_input_validation.json", {"smoke": {"passed": True}})
    window.to_parquet(out / "kpi_by_window.parquet", index=False)
    seed.to_parquet(out / "kpi_by_seed.parquet", index=False)
    time_band.to_parquet(out / "kpi_by_time_band.parquet", index=False)
    dump_json(out / "kpi_overall.json", overall)
    dump_json(out / "aggregation_manifest.json", {
        "causal_comparison_allowed": bool(window["causal_comparison_allowed"].all()),
        "warnings": [{"code": "non_causal_or_smoke_source"}],
    })
    print("SMOKE PASS")

if __name__ == "__main__":
    main()
'''


def write_synthetic_step102_rollup(input_root: Path) -> None:
    input_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in ["A", "A90"]:
        for seed in [1, 2]:
            for route_index in [0, 1]:
                rows.append({
                    "condition_id": condition,
                    "seed": seed,
                    "window_id": f"synthetic_{condition}_{seed}_{route_index}",
                    "state_ts": "2026-04-29T09:00:00+09:00",
                    "service_date": "2026-04-29",
                    "time_band": "peak" if route_index == 0 else "offpeak",
                    "evaluation_horizon_minutes": 30,
                    "headway_mean_seconds": 600.0 + route_index,
                    "headway_std_seconds": 120.0,
                    "headway_sample_count": 4,
                    "bunching_event_count": 1,
                    "headway_event_count": 4,
                    "wait_total_passenger_seconds": 12000.0,
                    "wait_passenger_count": 40,
                    "ontime_event_count": 3,
                    "schedulable_arrival_count": 4,
                    "intervention_count": 1 if condition != "A" else 0,
                    "decision_step_count": 8,
                    "energy_proxy_total": 100.0,
                    "source_mode": "route_aware_minimal_scaffold_step102_noncausal",
                    "passenger_demand_generated": 50,
                    "passenger_served_count": 45,
                    "passenger_service_rate": 0.9,
                    "passenger_wait_p95_seconds": 800.0,
                    "energy_proxy_per_passenger": 2.22,
                    "fleet_reduction_ratio": 0.1 if condition == "A90" else 0.0,
                })
    pd.DataFrame(rows).to_parquet(input_root / "window_rollup.parquet", index=False)
    with open(input_root / "route_aware_rollout_writer_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"audit_status": "PASS", "rollout_status": "READY_FOR_STEP103_CANONICAL_KPI_SCAFFOLD"}, f)


def main() -> None:
    assert normalize_source_mode("abc_noncausal_xyz") == "abc_non_causal_xyz"
    assert normalize_source_mode("abc_non_causal_xyz") == "abc_non_causal_xyz"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        input_root = root / "step102"
        output_root = root / "step103"
        aggregator = root / "canonical_kpi_aggregator.py"
        aggregator.write_text(MINI_CANONICAL_AGGREGATOR, encoding="utf-8")
        write_synthetic_step102_rollup(input_root)

        manifest = run_step103(
            Step103Paths(
                input_root=input_root,
                output_root=output_root,
                canonical_aggregator=aggregator,
            ),
            smoke=True,
            clean_output=True,
        )

        assert manifest["audit_status"] == "PASS"
        assert manifest["integration_status"] == "READY_FOR_STEP104_PROJECT_LOG_RUNBOOK_UPDATE"
        assert manifest["canonical_summary"]["row_counts"]["kpi_by_window"] == 8
        assert manifest["canonical_summary"]["causal_comparison_allowed"] is False
        assert (output_root / "canonical_eval" / "kpi_by_window.parquet").exists()
        assert (output_root / "canonical_output_readiness_step103.csv").exists()

    print("[OK] Step 103 route-aware canonical KPI integration self-test PASS")


if __name__ == "__main__":
    main()
