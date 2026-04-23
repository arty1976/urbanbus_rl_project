import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--eval-horizon-minutes", type=int, default=30)
    parser.add_argument("--simulator-adapter", default="")
    args = parser.parse_args()

    scenario_index = Path(args.scenario_index)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not scenario_index.exists():
        raise SystemExit(f"scenario_index not found: {scenario_index}")

    df = pd.read_parquet(scenario_index)

    required = ["window_id", "state_ts", "service_date", "time_band"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    manifest = {
        "baseline_id": "B1_noop",
        "seed": args.seed,
        "eval_horizon_minutes": args.eval_horizon_minutes,
        "scenario_count": int(len(df)),
        "scenario_index": str(scenario_index),
        "output_dir": str(output_dir),
        "simulator_adapter": args.simulator_adapter,
        "status": "adapter_missing" if not args.simulator_adapter else "ready_to_run"
    }

    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    if not args.simulator_adapter:
        print("[STOP] No simulator adapter configured.")
        print("[STOP] B1 contract is prepared, but real rollouts cannot run yet.")
        print(f"[OK] wrote: {output_dir / 'run_manifest.json'}")
        return

    print("[TODO] simulator adapter path supplied, but adapter invocation is not yet implemented.")
    print(f"[OK] wrote: {output_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
