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


def stable_int(*parts: str) -> int:
    s = "|".join(parts)
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)


def base_by_time_band(time_band: str):
    tb = str(time_band).strip().lower()
    if tb == "peak":
        return {
            "headway_mean": 560.0,
            "headway_std": 140.0,
            "headway_events": 6,
            "wait_avg": 340.0,
            "sched_arrivals": 6,
            "energy": 118.0,
        }
    if tb == "night":
        return {
            "headway_mean": 920.0,
            "headway_std": 85.0,
            "headway_events": 3,
            "wait_avg": 170.0,
            "sched_arrivals": 3,
            "energy": 68.0,
        }
    return {
        "headway_mean": 680.0,
        "headway_std": 105.0,
        "headway_events": 5,
        "wait_avg": 225.0,
        "sched_arrivals": 5,
        "energy": 88.0,
    }


def mode_adjust(condition_id: str):
    cid = str(condition_id).strip().upper()
    if cid == "B2":
        return {
            "source_mode": "stub_b2_rulebased_smoke",
            "headway_mean_delta": -25.0,
            "headway_std_delta": -12.0,
            "wait_avg_delta": -35.0,
            "bunching_bonus": -1,
            "ontime_bonus": 0.05,
            "intervention_count": 1,
            "decision_step_count": 1,
            "energy_delta": +10.0,
            "qwen_trigger_rate": 0.0,
            "effective_replay_step_minutes": 60.0,
        }
    if cid == "A":
        return {
            "source_mode": "stub_A_pure_mappo_smoke",
            "headway_mean_delta": -40.0,
            "headway_std_delta": -18.0,
            "wait_avg_delta": -55.0,
            "bunching_bonus": -1,
            "ontime_bonus": 0.08,
            "intervention_count": 1,
            "decision_step_count": 1,
            "energy_delta": +6.0,
            "qwen_trigger_rate": 0.0,
            "effective_replay_step_minutes": 60.0,
        }
    raise RuntimeError(f"unsupported condition_id: {condition_id}")


def build_rollup(df: pd.DataFrame, condition_id: str, seed: int, eval_horizon_minutes: int) -> pd.DataFrame:
    adj = mode_adjust(condition_id)
    rows = []

    for r in df.itertuples(index=False):
        tb = str(r.time_band).strip().lower()
        base = base_by_time_band(tb)
        h = stable_int(condition_id, str(seed), str(r.window_id))

        headway_events = int(base["headway_events"])
        headway_sample_count = headway_events + 1

        headway_mean_seconds = float(
            max(60.0, base["headway_mean"] + adj["headway_mean_delta"] + (h % 31) - 15 + seed * 2)
        )
        headway_std_seconds = float(
            max(1.0, base["headway_std"] + adj["headway_std_delta"] + (h % 21) - 10)
        )

        wait_passenger_count = int(20 + (h % 41))
        avg_wait_seconds = float(
            max(30.0, base["wait_avg"] + adj["wait_avg_delta"] + (h % 31) - 15 + seed * 2)
        )
        wait_total_passenger_seconds = float(avg_wait_seconds * wait_passenger_count)

        schedulable_arrival_count = int(base["sched_arrivals"])
        ontime_ratio = 0.70 + adj["ontime_bonus"] + ((h % 16) / 100.0)
        ontime_event_count = int(max(
            0,
            min(schedulable_arrival_count, round(schedulable_arrival_count * ontime_ratio))
        ))

        if tb == "peak":
            bunching_event_count = int((h % 3))
        elif tb == "offpeak":
            bunching_event_count = int((h % 2))
        else:
            bunching_event_count = 0

        bunching_event_count = max(0, bunching_event_count + int(adj["bunching_bonus"]))
        bunching_event_count = min(bunching_event_count, headway_events)

        intervention_count = int(adj["intervention_count"])
        decision_step_count = int(max(intervention_count, adj["decision_step_count"]))

        energy_proxy_total = float(max(0.0, base["energy"] + adj["energy_delta"] + (h % 23)))

        rows.append({
            "condition_id": str(condition_id).upper(),
            "seed": int(seed),
            "window_id": r.window_id,
            "state_ts": str(r.state_ts),
            "service_date": str(r.service_date),
            "time_band": tb,
            "evaluation_horizon_minutes": int(eval_horizon_minutes),
            "qwen_trigger_rate": float(adj["qwen_trigger_rate"]),
            "effective_replay_step_minutes": float(adj["effective_replay_step_minutes"]),
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
            "source_mode": adj["source_mode"],
        })

    return pd.DataFrame(rows)


def update_manifest(manifest_path: Path, payload: dict):
    if manifest_path.exists():
        manifest = load_json(manifest_path)
    else:
        manifest = {}

    manifest.update(payload)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--scenario-index", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--condition-id", required=True, choices=["B2", "A", "b2", "a"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    contract_path = Path(args.contract)
    scenario_index_path = Path(args.scenario_index)
    root = Path(args.root)
    condition_id = str(args.condition_id).upper()

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
        seed_dir = root / "rollouts" / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        out_df = build_rollup(
            df=df,
            condition_id=condition_id,
            seed=seed,
            eval_horizon_minutes=eval_horizon_minutes,
        )

        out_path = seed_dir / "window_rollup.parquet"
        out_df.to_parquet(out_path, index=False)

        manifest_payload = {
            "condition_id": condition_id,
            "seed": seed,
            "window_rollup_path": str(out_path),
            "window_rollup_mode": f"stub_{condition_id.lower()}_smoke",
            "scenario_count_used": int(len(out_df)),
            "status": "stub_window_rollup_written",
        }
        update_manifest(seed_dir / "run_manifest.json", manifest_payload)

        print(f"[OK] condition={condition_id} seed={seed:03d} wrote: {out_path} rows={len(out_df)}")
        total_written += len(out_df)

    print(f"[OK] condition={condition_id} total_rows_written={total_written}")


if __name__ == "__main__":
    main()
