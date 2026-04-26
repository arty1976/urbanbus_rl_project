"""
run_a_family_scenario_rollout_writer_v1.py
==========================================

Scenario-index based A-family rollout writer.

Purpose
-------
This sidecar writer converts a scenario_index.parquet file into extended
window_rollup.parquet files for A/A90/A80/A70.

It connects the Step 25 bridge to scenario windows without modifying
run_causal_rollout.py yet.

MAPPO means Multi-Agent Proximal Policy Optimization.
KPI means Key Performance Indicator.

Important
---------
This writer is still a bridge/integration writer.
It does not run a neural MAPPO model yet.
It exists to validate schema, reward, policy provenance, fleet sensitivity,
and shared demand fairness before patching run_causal_rollout.py.

Rules
-----
1. A/A90/A80/A70 share the same passenger_demand_generated for the same window.
2. A90/A80/A70 change active_bus_count only.
3. qwen_trigger_rate remains 0.0.
4. Actual MAPPO mode requires checkpoint_path.
5. Placeholder mode is smoke/contract only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


THIS = Path(__file__).resolve()
TRAINING_DIR = THIS.parents[1]
ROLLOUTS_DIR = TRAINING_DIR / "rollouts"
POLICIES_DIR = TRAINING_DIR / "policies"
REWARDS_DIR = TRAINING_DIR / "rewards"

for p in (ROLLOUTS_DIR, POLICIES_DIR, REWARDS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


from a_family_rollout_bridge_v1 import build_a_family_rollout_row  # noqa: E402
from rollout_schema_v1 import validate_rollout_rows, validate_same_demand_by_window  # noqa: E402


A_FAMILY_DEFAULT = ["A", "A90", "A80", "A70"]


def stable_int(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def parse_csv_list(value: str) -> List[str]:
    return [
        item.strip()
        for item in str(value).split(",")
        if item.strip()
    ]


def parse_seed_list(value: str) -> List[int]:
    out = []
    for item in parse_csv_list(value):
        out.append(int(item))
    return out


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def normalize_time_band(value: Any) -> str:
    text = str(value or "offpeak").strip().lower()
    if text not in {"peak", "offpeak", "night"}:
        return "offpeak"
    return text


def base_by_time_band(time_band: str) -> Dict[str, float]:
    tb = normalize_time_band(time_band)

    if tb == "peak":
        return {
            "headway_mean_seconds": 560.0,
            "headway_std_seconds": 130.0,
            "headway_event_count": 6,
            "headway_sample_count": 7,
            "wait_avg_seconds": 330.0,
            "schedulable_arrival_count": 6,
            "energy_proxy_total": 420.0,
            "demand_base": 80,
            "demand_span": 40,
            "p95_wait_seconds": 610.0,
        }

    if tb == "night":
        return {
            "headway_mean_seconds": 900.0,
            "headway_std_seconds": 90.0,
            "headway_event_count": 3,
            "headway_sample_count": 4,
            "wait_avg_seconds": 180.0,
            "schedulable_arrival_count": 3,
            "energy_proxy_total": 190.0,
            "demand_base": 20,
            "demand_span": 15,
            "p95_wait_seconds": 360.0,
        }

    return {
        "headway_mean_seconds": 680.0,
        "headway_std_seconds": 105.0,
        "headway_event_count": 5,
        "headway_sample_count": 6,
        "wait_avg_seconds": 230.0,
        "schedulable_arrival_count": 5,
        "energy_proxy_total": 310.0,
        "demand_base": 45,
        "demand_span": 30,
        "p95_wait_seconds": 460.0,
    }


def make_sample_scenario_rows() -> List[Dict[str, Any]]:
    return [
        {
            "window_id": "sample_peak_001",
            "state_ts": "2023-01-02T08:00:00+09:00",
            "service_date": "2023-01-02",
            "time_band": "peak",
        },
        {
            "window_id": "sample_offpeak_001",
            "state_ts": "2023-01-02T13:00:00+09:00",
            "service_date": "2023-01-02",
            "time_band": "offpeak",
        },
        {
            "window_id": "sample_night_001",
            "state_ts": "2023-01-02T21:00:00+09:00",
            "service_date": "2023-01-02",
            "time_band": "night",
        },
    ]


def load_scenario_rows(
    scenario_index_path: Optional[Path],
    limit: int,
    self_test: bool,
) -> List[Dict[str, Any]]:
    if self_test:
        rows = make_sample_scenario_rows()
        return rows[:limit] if limit > 0 else rows

    if scenario_index_path is None:
        raise FileNotFoundError("scenario_index_path is required unless --self-test is used")

    if not scenario_index_path.exists():
        raise FileNotFoundError(f"scenario_index not found: {scenario_index_path}")

    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError("pandas is required to read scenario_index.parquet") from exc

    df = pd.read_parquet(scenario_index_path).copy()

    if df.empty:
        raise RuntimeError(f"scenario_index is empty: {scenario_index_path}")

    if limit and limit > 0:
        df = df.head(limit).copy()

    rows = df.to_dict("records")

    normalized: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        out = dict(row)

        if "window_id" not in out or out.get("window_id") is None:
            out["window_id"] = f"window_{idx + 1:06d}"

        if "state_ts" not in out or out.get("state_ts") is None:
            raise RuntimeError("scenario row missing state_ts")

        if "service_date" not in out or out.get("service_date") is None:
            out["service_date"] = str(out["state_ts"])[:10]

        if "time_band" not in out or out.get("time_band") is None:
            out["time_band"] = "offpeak"

        out["window_id"] = str(out["window_id"])
        out["state_ts"] = str(out["state_ts"])
        out["service_date"] = str(out["service_date"])
        out["time_band"] = normalize_time_band(out["time_band"])

        normalized.append(out)

    return normalized


def build_base_row_from_scenario(
    scenario_row: Mapping[str, Any],
    *,
    seed: int,
    baseline_bus_count: float,
    eval_horizon_minutes: int,
) -> Dict[str, Any]:
    window_id = str(scenario_row.get("window_id"))
    time_band = normalize_time_band(scenario_row.get("time_band"))
    base = base_by_time_band(time_band)

    h = stable_int(window_id, time_band)

    if scenario_row.get("passenger_demand_generated") is not None:
        demand_generated = as_int(scenario_row.get("passenger_demand_generated"), default=0)
    elif scenario_row.get("shared_passenger_demand_generated") is not None:
        demand_generated = as_int(scenario_row.get("shared_passenger_demand_generated"), default=0)
    else:
        demand_generated = int(base["demand_base"] + (h % int(base["demand_span"])))

    demand_generated = max(0, int(demand_generated))

    missed = int(h % 3)
    passenger_served_count = max(0, demand_generated - missed)

    wait_avg = float(base["wait_avg_seconds"] + (h % 31) - 15)
    wait_total = float(wait_avg * max(1, passenger_served_count))

    headway_event_count = int(base["headway_event_count"])
    headway_sample_count = int(base["headway_sample_count"])

    if time_band == "peak":
        bunching_event_count = int(h % 2)
    else:
        bunching_event_count = 0

    schedulable_arrival_count = int(base["schedulable_arrival_count"])
    ontime_event_count = int(max(0, min(
        schedulable_arrival_count,
        round(schedulable_arrival_count * 0.80),
    )))

    return {
        "seed": int(seed),
        "window_id": window_id,
        "state_ts": str(scenario_row.get("state_ts")),
        "service_date": str(scenario_row.get("service_date")),
        "time_band": time_band,
        "evaluation_horizon_minutes": int(eval_horizon_minutes),

        "headway_mean_seconds": float(base["headway_mean_seconds"] + (h % 21) - 10),
        "headway_std_seconds": float(max(1.0, base["headway_std_seconds"] + (h % 15) - 7)),
        "headway_sample_count": headway_sample_count,
        "bunching_event_count": bunching_event_count,
        "headway_event_count": headway_event_count,

        "wait_total_passenger_seconds": wait_total,
        "wait_passenger_count": int(max(1, passenger_served_count)),

        "ontime_event_count": ontime_event_count,
        "schedulable_arrival_count": schedulable_arrival_count,

        "intervention_count": 1,
        "decision_step_count": 5,
        "energy_proxy_total": float(base["energy_proxy_total"] + (h % 41)),

        "passenger_demand_generated": int(demand_generated),
        "passenger_served_count": int(passenger_served_count),
        "passenger_service_rate": None,
        "passenger_wait_p95_seconds": float(base["p95_wait_seconds"] + (h % 41) - 20),
        "long_wait_passenger_count": int(h % 4),

        "energy_proxy": None,
        "energy_proxy_per_passenger": None,
        "active_bus_count": float(baseline_bus_count),
        "baseline_bus_count": float(baseline_bus_count),
        "fleet_reduction_ratio": None,

        "policy_source": None,
        "policy_checkpoint_path": None,
        "source_mode": None,
        "qwen_trigger_rate": 0.0,
        "effective_replay_step_minutes": 5.0,

        "reward_total": 0.0,
        "reward_service": 0.0,
        "reward_avg_wait": 0.0,
        "reward_long_wait": 0.0,
        "reward_on_time": 0.0,
        "reward_bunching": 0.0,
        "reward_energy": 0.0,
        "reward_fleet": 0.0,
        "reward_constraint": 0.0,
    }


def build_rows(
    scenario_rows: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str],
    seeds: Sequence[int],
    policy_kind: str,
    checkpoint_path: Optional[str],
    require_existing_checkpoint: bool,
    baseline_bus_count: float,
    eval_horizon_minutes: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for seed in seeds:
        for scenario_row in scenario_rows:
            base_row = build_base_row_from_scenario(
                scenario_row,
                seed=seed,
                baseline_bus_count=baseline_bus_count,
                eval_horizon_minutes=eval_horizon_minutes,
            )

            for condition_id in conditions:
                row = build_a_family_rollout_row(
                    base_row,
                    condition_id=condition_id,
                    policy_kind=policy_kind,
                    checkpoint_path=checkpoint_path,
                    require_existing_checkpoint=require_existing_checkpoint,
                    baseline_bus_count=baseline_bus_count,
                    active_bus_count=None,
                    strict_claim_validation=(policy_kind == "mappo"),
                )
                rows.append(row)

    return rows


def json_safe_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def json_safe_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append({key: json_safe_value(value) for key, value in row.items()})
    return out


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    safe = json_safe_rows(rows)

    if not safe:
        return

    fieldnames = sorted({key for row in safe for key in row.keys()})

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in safe:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        import pandas as pd
    except Exception:
        return False

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        safe = json_safe_rows(rows)
        df = pd.DataFrame(safe)
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def group_rows_by_condition_seed(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    grouped: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}

    for row in rows:
        condition_id = str(row.get("condition_id"))
        seed = int(row.get("seed"))

        grouped.setdefault(condition_id, {})
        grouped[condition_id].setdefault(seed, [])
        grouped[condition_id][seed].append(dict(row))

    return grouped


def write_condition_seed_outputs(
    output_root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    write_parquet_files: bool,
) -> Dict[str, Any]:
    grouped = group_rows_by_condition_seed(rows)
    files: Dict[str, Any] = {}

    for condition_id, by_seed in grouped.items():
        files.setdefault(condition_id, {})

        for seed, seed_rows in by_seed.items():
            seed_dir = output_root / condition_id / "rollouts" / f"seed_{seed:03d}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            csv_path = seed_dir / "window_rollup.csv"
            jsonl_path = seed_dir / "window_rollup.jsonl"
            parquet_path = seed_dir / "window_rollup.parquet"
            manifest_path = seed_dir / "run_manifest.json"

            write_csv(csv_path, seed_rows)
            write_jsonl(jsonl_path, seed_rows)

            parquet_written = False
            if write_parquet_files:
                parquet_written = write_parquet(parquet_path, seed_rows)

            manifest = {
                "artifact_version": "a_family_scenario_rollout_writer_seed_manifest_v1",
                "condition_id": condition_id,
                "seed": seed,
                "row_count": len(seed_rows),
                "window_rollup_csv": str(csv_path),
                "window_rollup_jsonl": str(jsonl_path),
                "window_rollup_parquet": str(parquet_path) if parquet_written else None,
                "parquet_written": parquet_written,
            }
            write_json(manifest_path, manifest)

            files[condition_id][f"seed_{seed:03d}"] = manifest

    return files


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    schema_summary = validate_rollout_rows(rows, strict=True)
    demand_summary = validate_same_demand_by_window(rows)

    condition_ids = sorted(set(str(row.get("condition_id")) for row in rows))
    seeds = sorted(set(int(row.get("seed")) for row in rows))
    source_modes = sorted(set(str(row.get("source_mode")) for row in rows))

    fleet_by_condition: Dict[str, Any] = {}

    for condition_id in condition_ids:
        condition_rows = [row for row in rows if str(row.get("condition_id")) == condition_id]
        if not condition_rows:
            continue

        first = condition_rows[0]
        fleet_by_condition[condition_id] = {
            "baseline_bus_count": first.get("baseline_bus_count"),
            "active_bus_count": first.get("active_bus_count"),
            "fleet_reduction_ratio": first.get("fleet_reduction_ratio"),
        }

    return {
        "artifact_version": "a_family_scenario_rollout_writer_summary_v1",
        "row_count": len(rows),
        "condition_ids": condition_ids,
        "seeds": seeds,
        "source_modes": source_modes,
        "schema_summary": schema_summary,
        "demand_fairness_summary": demand_summary,
        "fleet_by_condition": fleet_by_condition,
        "passed": bool(schema_summary["passed"]) and bool(demand_summary["passed"]),
    }


def run_writer(
    *,
    scenario_index_path: Optional[Path],
    output_root: Path,
    limit: int,
    conditions: Sequence[str],
    seeds: Sequence[int],
    policy_kind: str,
    checkpoint_path: Optional[str],
    require_existing_checkpoint: bool,
    baseline_bus_count: float,
    eval_horizon_minutes: int,
    write_parquet_files: bool,
    self_test: bool,
) -> Dict[str, Any]:
    scenario_rows = load_scenario_rows(
        scenario_index_path=scenario_index_path,
        limit=limit,
        self_test=self_test,
    )

    if policy_kind == "placeholder":
        effective_checkpoint_path = None
    else:
        effective_checkpoint_path = checkpoint_path

    rows = build_rows(
        scenario_rows=scenario_rows,
        conditions=conditions,
        seeds=seeds,
        policy_kind=policy_kind,
        checkpoint_path=effective_checkpoint_path,
        require_existing_checkpoint=require_existing_checkpoint,
        baseline_bus_count=baseline_bus_count,
        eval_horizon_minutes=eval_horizon_minutes,
    )

    summary = summarize(rows)

    output_root.mkdir(parents=True, exist_ok=True)

    combined_csv = output_root / "combined_a_family_window_rollup.csv"
    combined_jsonl = output_root / "combined_a_family_window_rollup.jsonl"
    combined_parquet = output_root / "combined_a_family_window_rollup.parquet"

    write_csv(combined_csv, rows)
    write_jsonl(combined_jsonl, rows)

    combined_parquet_written = False
    if write_parquet_files:
        combined_parquet_written = write_parquet(combined_parquet, rows)

    condition_seed_files = write_condition_seed_outputs(
        output_root=output_root,
        rows=rows,
        write_parquet_files=write_parquet_files,
    )

    summary_path = output_root / "summary.json"
    manifest_path = output_root / "manifest.json"

    write_json(summary_path, summary)

    manifest = {
        "artifact_version": "a_family_scenario_rollout_writer_manifest_v1",
        "scenario_index_path": str(scenario_index_path) if scenario_index_path else None,
        "output_root": str(output_root),
        "limit": limit,
        "conditions": list(conditions),
        "seeds": list(seeds),
        "policy_kind": policy_kind,
        "checkpoint_path": effective_checkpoint_path,
        "require_existing_checkpoint": require_existing_checkpoint,
        "baseline_bus_count": baseline_bus_count,
        "eval_horizon_minutes": eval_horizon_minutes,
        "write_parquet_files": write_parquet_files,
        "combined_files": {
            "csv": str(combined_csv),
            "jsonl": str(combined_jsonl),
            "parquet": str(combined_parquet) if combined_parquet_written else None,
        },
        "condition_seed_files": condition_seed_files,
        "summary_path": str(summary_path),
        "passed": bool(summary["passed"]),
    }

    write_json(manifest_path, manifest)

    if not summary["passed"]:
        raise RuntimeError(f"A-family scenario rollout writer validation failed: {summary}")

    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario-index",
        default="artifacts/baseline_v1/B1_noop/scenario_index.parquet",
    )
    parser.add_argument(
        "--output-root",
        default="artifacts/step27_a_family_scenario_rollout",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--conditions",
        default="A,A90,A80,A70",
    )
    parser.add_argument(
        "--seeds",
        default="1,2,3",
    )
    parser.add_argument(
        "--policy-kind",
        choices=["placeholder", "mappo"],
        default="placeholder",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="artifacts/experiment_A_v1/checkpoints/best.pt",
    )
    parser.add_argument(
        "--require-existing-checkpoint",
        action="store_true",
    )
    parser.add_argument(
        "--baseline-bus-count",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--eval-horizon-minutes",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--write-parquet",
        action="store_true",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    args = parser.parse_args()

    scenario_index_path: Optional[Path]
    if args.self_test:
        scenario_index_path = None
    else:
        scenario_index_path = Path(args.scenario_index)

    manifest = run_writer(
        scenario_index_path=scenario_index_path,
        output_root=Path(args.output_root),
        limit=int(args.limit),
        conditions=parse_csv_list(args.conditions),
        seeds=parse_seed_list(args.seeds),
        policy_kind=args.policy_kind,
        checkpoint_path=args.checkpoint_path,
        require_existing_checkpoint=bool(args.require_existing_checkpoint),
        baseline_bus_count=float(args.baseline_bus_count),
        eval_horizon_minutes=int(args.eval_horizon_minutes),
        write_parquet_files=bool(args.write_parquet),
        self_test=bool(args.self_test),
    )

    print("[OK] A-family scenario rollout writer completed")
    print(f"[OK] output_root : {manifest['output_root']}")
    print(f"[OK] summary     : {manifest['summary_path']}")
    print(f"[OK] passed      : {manifest['passed']}")

    if args.self_test:
        print("[OK] run_a_family_scenario_rollout_writer_v1 self-test passed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
