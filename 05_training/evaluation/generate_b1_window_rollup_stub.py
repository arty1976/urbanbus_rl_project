import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def load_json(path: Path):
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            pass
    raise RuntimeError(f"failed to read json: {path}")


def stable_int(seed: int, window_id: str) -> int:
    s = f"{seed}|{window_id}"
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def base_by_time_band(time_band: str):
    tb = str(time_band).strip().lower()
    if tb == "peak":
        return {
            "headway_mean": 540.0,
            "headway_std": 150.0,
            "headway_events": 6,
            "wait_avg": 360.0,
            "sched_arrivals": 6,
            "energy": 120.0,
        }
    if tb == "night":
        return {
            "headway_mean": 900.0,
            "headway_std": 90.0,
            "headway_events": 3,
            "wait_avg": 180.0,
            "sched_arrivals": 3,
            "energy": 70.0,
        }
    return {
        "headway_mean": 660.0,
        "headway_std": 110.0,
        "headway_events": 5,
        "wait_avg": 240.0,
        "sched_arrivals": 5,
        "energy": 90.0,
    }


def build_rollup(df: pd.DataFrame, seed: int, eval_horizon_minutes: int) -> pd.DataFrame:
    rows = []

    for r in df.itertuples(index=False):
        tb = str(r.time_band).strip().lower()
        base = base_by_time_band(tb)
        h = stable_int(seed, str(r.window_id))

        headway_events = int(base["headway_events"])
        headway_sample_count = headway_events + 1

        headway_mean_seconds = float(base["headway_mean"] + (h % 31) - 15 + seed * 3)
        headway_std_seconds = float(max(1.0, base["headway_std"] + (h % 21) - 10))

        wait_passenger_count = int(20 + (h % 41))
        avg_wait_seconds = float(base["wait_avg"] + (h % 31) - 15 + seed * 2)
        wait_total_passenger_seconds = float(avg_wait_seconds * wait_passenger_count)

        schedulable_arrival_count = int(base["sched_arrivals"])
        ontime_event_count = int(max(0, min(
            schedulable_arrival_count,
            round(schedulable_arrival_count * (0.70 + (h % 16) / 100.0))
        )))

        if tb == "peak":
            bunching_event_count = int((h % 3))
        elif tb == "offpeak":
            bunching_event_count = int((h % 2))
        else:
            bunching_event_count = 0

        bunching_event_count = min(bunching_event_count, headway_events)

        intervention_count = 0
        decision_step_count = 1
        energy_proxy_total = float(base["energy"] + (h % 23))

        rows.append({
            "condition_id": "B1",
            "seed": int(seed),
            "window_id": r.window_id,
            "state_ts": str(r.state_ts),
            "service_date": str(r.service_date),
            "time_band": tb,
            "evaluation_horizon_minutes": int(eval_horizon_minutes),
            "headway_mean_seconds": headway_mean_seconds,
            "headway_std_seconds": headway_std_seconds,
            "headway_sample_count": headway_sample_count,
            "bunching_event_count": bunching_event_count,
            "headway_event_count": headway_events,
            "wait_total_passenger_seconds": wait_total_passenger_seconds,
            "wait_passenger_count": wait_passenger_count,
            "ontime_event_count": ontime_event_count,
            "schedulable_arrival_count": schedulable_arrival_count,
            "intervention_count": intervention_count,
            "decision_step_count": decision_step_count,
            "energy_proxy_total": energy_proxy_total,
            "source_mode": "stub_b1_noop_smoke",
        })

    out = pd.DataFrame(rows)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--scenario-index", required=True)
    parser.add_argument("--b1-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    contract_path = Path(args.contract)
    scenario_index_path = Path(args.scenario_index)
    b1_root = Path(args.b1_root)

    if not contract_path.exists():
        raise SystemExit(f"contract not found: {contract_path}")
    if not scenario_index_path.exists():
        raise SystemExit(f"scenario_index not found: {scenario_index_path}")

    contract = load_json(contract_path)
    df = pd.read_parquet(scenario_index_path).copy()

    required = ["window_id", "state_ts", "service_date", "time_band"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"scenario_index missing columns: {missing}")

    df["time_band"] = df["time_band"].astype(str).str.strip().str.lower()

    valid_time_bands = {str(x).strip().lower() for x in contract.get("time_bands", [])}
    bad = sorted(set(df["time_band"]) - valid_time_bands)
    if bad:
        raise SystemExit(f"invalid time_band values in scenario_index: {bad}")

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    if df.empty:
        raise SystemExit("scenario_index is empty after applying limit")

    eval_horizon_minutes = int(contract["evaluation_horizon_minutes"])
    seeds = [int(x) for x in contract["seeds"]]

    total_written = 0

    for seed in seeds:
        seed_dir = b1_root / "rollouts" / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        out_df = build_rollup(df=df, seed=seed, eval_horizon_minutes=eval_horizon_minutes)
        out_path = seed_dir / "window_rollup.parquet"
        out_df.to_parquet(out_path, index=False)

        manifest_path = seed_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
        else:
            manifest = {
                "baseline_id": "B1_noop",
                "seed": seed,
            }

        manifest["window_rollup_path"] = str(out_path)
        manifest["window_rollup_mode"] = "stub_b1_noop_smoke"
        manifest["scenario_count_used"] = int(len(out_df))
        manifest["status"] = "stub_window_rollup_written"

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"[OK] seed={seed:03d} wrote: {out_path} rows={len(out_df)}")
        total_written += len(out_df)

    print(f"[OK] total_rows_written={total_written}")
    print("[OK] B1 window_rollup stub generation complete")


if __name__ == "__main__":
    main()
