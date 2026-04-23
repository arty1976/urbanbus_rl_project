import json
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
b0_parquet = root / "artifacts" / "baseline_v1" / "B0_historical" / "kpi_by_window.parquet"
out_dir = root / "artifacts" / "baseline_v1" / "B2_rulebased"
rollouts_dir = out_dir / "rollouts"

if not b0_parquet.exists():
    raise SystemExit(f"B0 parquet not found: {b0_parquet}")

out_dir.mkdir(parents=True, exist_ok=True)
rollouts_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(b0_parquet)

required_cols = ["window_id", "state_ts", "service_date", "time_band"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns in B0 parquet: {missing}")

scenario_df = df[required_cols].copy()
scenario_df["policy_id"] = "B2_rulebased"
scenario_df["policy_name"] = "headway_threshold_rule"
scenario_df["eval_horizon_minutes"] = 30
scenario_df["rule_family"] = "holding_skip_threshold"

scenario_path = out_dir / "scenario_index.parquet"
scenario_df.to_parquet(scenario_path, index=False)

for seed in [1, 2, 3]:
    (rollouts_dir / f"seed_{seed:03d}").mkdir(parents=True, exist_ok=True)

policy_config = {
    "baseline_id": "B2_rulebased",
    "policy_name": "headway_threshold_rule",
    "definition": "threshold-based holding/skip heuristic baseline",
    "eval_horizon_minutes": 30,
    "seeds": [1, 2, 3],
    "same_initial_state": True,
    "same_exogenous_events": True,
    "same_eval_window": True,
    "rule_params": {
        "target_headway_seconds": 600,
        "low_headway_threshold_seconds": 360,
        "high_headway_threshold_seconds": 900,
        "max_hold_seconds": 120,
        "allow_skip": True
    },
    "source_b0_parquet": str(b0_parquet),
    "scenario_index_file": str(scenario_path),
    "output_rollouts_dir": str(rollouts_dir),
    "status": "prepared_not_executed",
    "note": "Real B2 rollout still requires a replay simulator adapter."
}

with open(out_dir / "policy_config.json", "w", encoding="utf-8") as f:
    json.dump(policy_config, f, ensure_ascii=False, indent=2)

print("[OK] Prepared B2 rulebased layout")
print(f"[OK] scenario_index: {scenario_path}")
print(f"[OK] policy_config : {out_dir / 'policy_config.json'}")
print(f"[OK] rollouts dir   : {rollouts_dir}")
print(f"[OK] windows        : {len(scenario_df)}")
