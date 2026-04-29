#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self-test for Step 107 queue/demand canonical KPI integration scaffold."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from queue_demand_canonical_kpi_step107 import Step107Paths, normalize_source_mode, run_step107, read_table, write_table_with_csv_fallback

MINI_CANONICAL_AGGREGATOR = r'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
import pandas as pd
EXPECTED_SHARED_KPIS = ["cv_headway", "avg_wait_seconds", "bunching_rate", "on_time_rate", "intervention_rate", "energy_proxy"]

def dump_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def read_table(path):
    path = Path(path)
    if not path.exists():
        alt = path.with_suffix(".csv") if path.suffix == ".parquet" else path.with_suffix(".parquet")
        if alt.exists(): path = alt
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

def write_table(df, parquet_path):
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass
    df.to_csv(parquet_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

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
    raw = read_table(in_path)
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
    df["causal_comparison_allowed"] = ~df["source_mode"].astype(str).str.lower().str.contains("historical|legacy|stub|replay|smoke|non_causal|non-causal", regex=True, na=False)
    df["qwen_trigger_rate"] = 0.0
    df["effective_replay_step_minutes"] = 60
    df["input_source_path"] = str(in_path)
    # Intentionally drops queue/demand extension columns to test Step 107 preservation wrapper.
    out_cols = ["condition_id", "seed", "window_id", "state_ts", "service_date", "time_band", "evaluation_horizon_minutes", *EXPECTED_SHARED_KPIS, "strict_canonical", "computation_mode", "causal_comparison_allowed", "source_mode", "qwen_trigger_rate", "effective_replay_step_minutes", "headway_mean_seconds", "headway_std_seconds", "headway_sample_count", "bunching_event_count", "headway_event_count", "wait_total_passenger_seconds", "wait_passenger_count", "ontime_event_count", "schedulable_arrival_count", "intervention_count", "decision_step_count", "energy_proxy_total", "input_source_path"]
    window = df[out_cols].copy()
    seed = window.groupby(["condition_id", "seed"], as_index=False).agg(window_count=("window_id", "count"))
    time_band = window.groupby(["condition_id", "seed", "time_band"], as_index=False).agg(window_count=("window_id", "count"))
    overall = {"mode": "official_rollup", "n_windows_total": int(len(window)), "causal_comparison_allowed": bool(window["causal_comparison_allowed"].all()), "kpis": {k: {"valid_window_count": int(pd.to_numeric(window[k], errors="coerce").notna().sum())} for k in EXPECTED_SHARED_KPIS}}
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    dump_json(out / "official_rollup_input_validation.json", {"smoke": {"passed": True}})
    write_table(window, out / "kpi_by_window.parquet")
    write_table(seed, out / "kpi_by_seed.parquet")
    write_table(time_band, out / "kpi_by_time_band.parquet")
    dump_json(out / "kpi_overall.json", overall)
    dump_json(out / "aggregation_manifest.json", {"causal_comparison_allowed": bool(window["causal_comparison_allowed"].all()), "warnings": [{"code": "non_causal_or_smoke_source"}]})
    print("SMOKE PASS")
if __name__ == "__main__":
    main()
'''

def write_synthetic_step106_rollup(input_root: Path) -> None:
    input_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in ["A", "A90"]:
        for seed in [1, 2]:
            for route_index in [0, 1]:
                demand = 70 + route_index
                served = 56 + route_index
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
                    "wait_total_passenger_seconds": 16000.0,
                    "wait_passenger_count": 50,
                    "ontime_event_count": 3,
                    "schedulable_arrival_count": 4,
                    "intervention_count": 1 if condition != "A" else 0,
                    "decision_step_count": 8,
                    "energy_proxy_total": 120.0,
                    "source_mode": "route_aware_queue_demand_scaffold_step106_non_causal",
                    "artifact_version": "minimal_queue_demand_step106_v1",
                    "queue_demand_model_version": "minimal_queue_demand_proxy_v1",
                    "passenger_demand_generated": demand,
                    "passenger_served_count": served,
                    "passenger_service_rate": served / demand,
                    "passenger_wait_p95_seconds": 900.0 + route_index,
                    "energy_proxy_per_passenger": 120.0 / served,
                    "fleet_reduction_ratio": 0.1 if condition == "A90" else 0.0,
                    "passenger_left_behind_count": demand - served,
                    "max_queue_depth": 10 + route_index,
                    "avg_queue_depth": 5.0 + route_index,
                    "avg_queue_pressure": 0.35 + route_index * 0.01,
                    "unmet_demand_rate": 1.0 - (served / demand),
                    "queue_event_count": 20,
                    "queue_demand_observed": False,
                    "queue_demand_proxy": True,
                    "actual_passenger_wait_observed": False,
                    "actual_headway_observed": False,
                    "actual_arrival_departure_time_observed": False,
                    "actual_dwell_observed": False,
                    "paper_level_claim_allowed": False,
                    "causal_performance_claim_allowed": False,
                    "scaffold_only": True,
                    "performance_claim_allowed": False,
                    "actual_policy_claim_ready": False,
                    "causal_policy_claim_ready": False,
                })
    df = pd.DataFrame(rows)
    write_table_with_csv_fallback(df, input_root / "window_rollup_queue_demand.parquet")
    with open(input_root / "minimal_queue_demand_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"audit_status": "PASS", "scaffold_status": "READY_FOR_STEP107_QUEUE_DEMAND_CANONICAL_KPI"}, f)

def main() -> None:
    assert normalize_source_mode("abc_noncausal_xyz") == "abc_non_causal_xyz"
    assert normalize_source_mode("abc_non_causal_xyz") == "abc_non_causal_xyz"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        input_root = root / "step106"
        output_root = root / "step107"
        aggregator = root / "canonical_kpi_aggregator.py"
        aggregator.write_text(MINI_CANONICAL_AGGREGATOR, encoding="utf-8")
        write_synthetic_step106_rollup(input_root)
        manifest = run_step107(Step107Paths(input_root, output_root, aggregator), smoke=True, clean_output=True)
        assert manifest["audit_status"] == "PASS"
        assert manifest["integration_status"] == "READY_FOR_STEP108_QUEUE_DEMAND_REWARD_WIRING"
        assert manifest["canonical_summary"]["row_counts"]["kpi_by_window"] == 8
        assert manifest["canonical_summary"]["causal_comparison_allowed"] is False
        assert manifest["extension_summary"]["all_12_kpis_present"] is True
        assert manifest["extension_summary"]["all_12_kpis_have_valid_rows"] is True
        preserved = read_table(output_root / "canonical_eval" / "kpi_by_window_queue_demand_preserved.parquet")
        for col in ["passenger_demand_generated", "passenger_served_count", "passenger_service_rate", "passenger_wait_p95_seconds", "energy_proxy_per_passenger", "fleet_reduction_ratio"]:
            assert col in preserved.columns
            assert preserved[col].notna().all()
        assert not bool(preserved["causal_comparison_allowed"].all())
        assert not bool(preserved["paper_level_claim_allowed"].astype(bool).any())
        assert bool(preserved["queue_demand_proxy"].astype(bool).all())
    print("[OK] Step 107 queue/demand canonical KPI integration self-test PASS")
if __name__ == "__main__":
    main()
