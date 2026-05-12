from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-root", required=True)
    parser.add_argument("--min-rate", type=float, default=0.02)
    parser.add_argument("--max-rate", type=float, default=0.20)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    root = Path(args.rollout_root)
    files = sorted(root.glob("rollouts/seed_*/window_rollup.parquet"))

    if not files:
        raise SystemExit(f"[STOP] no window_rollup.parquet found under: {root}")

    frames = []
    for p in files:
        df = pd.read_parquet(p).copy()
        df["input_file"] = str(p)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    required = ["intervention_count", "decision_step_count", "time_band"]
    missing = [c for c in required if c not in all_df.columns]
    if missing:
        raise SystemExit(f"[STOP] missing required columns: {missing}")

    intervention_count = pd.to_numeric(all_df["intervention_count"], errors="coerce").fillna(0).sum()
    decision_step_count = pd.to_numeric(all_df["decision_step_count"], errors="coerce").fillna(0).sum()

    if decision_step_count <= 0:
        raise SystemExit("[STOP] decision_step_count is zero")

    intervention_rate = float(intervention_count / decision_step_count)

    by_time_band = {}
    for tb, grp in all_df.groupby("time_band", dropna=False):
        ic = pd.to_numeric(grp["intervention_count"], errors="coerce").fillna(0).sum()
        dc = pd.to_numeric(grp["decision_step_count"], errors="coerce").fillna(0).sum()
        by_time_band[str(tb)] = {
            "window_count": int(len(grp)),
            "intervention_count": float(ic),
            "decision_step_count": float(dc),
            "intervention_rate": float(ic / dc) if dc > 0 else None,
        }

    status = "PASS"
    hard_failures = []

    if intervention_rate <= 0:
        status = "FAIL"
        hard_failures.append("intervention_rate_is_zero")

    if intervention_rate < args.min_rate:
        status = "FAIL"
        hard_failures.append(f"intervention_rate_below_min_{args.min_rate}")

    if intervention_rate > args.max_rate:
        status = "FAIL"
        hard_failures.append(f"intervention_rate_above_max_{args.max_rate}")

    payload = {
        "audit_status": status,
        "rollout_root": str(root),
        "files": [str(p) for p in files],
        "window_rows": int(len(all_df)),
        "intervention_count": float(intervention_count),
        "decision_step_count": float(decision_step_count),
        "intervention_rate": intervention_rate,
        "expected_rate_band": {
            "min": args.min_rate,
            "max": args.max_rate,
        },
        "by_time_band": by_time_band,
        "hard_failures": hard_failures,
        "claim_guards": {
            "causal_comparison_allowed": False,
            "paper_level_claim_allowed": False,
            "strict_canonical": False,
        },
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("[OK] B2 intervention-rate validation completed")
    print("[OK] audit_status:", status)
    print("[OK] window_rows:", len(all_df))
    print("[OK] intervention_rate:", intervention_rate)
    print("[OK] output_json:", out)

    if status != "PASS":
        raise SystemExit(f"[FAIL] {hard_failures}")


if __name__ == "__main__":
    main()
