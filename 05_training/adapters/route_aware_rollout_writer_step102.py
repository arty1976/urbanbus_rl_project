#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 102 — route-aware rollout writer scaffold.

This script consumes the Step 101 route-aware minimal simulator scaffold and
writes canonical rollout-shaped artifacts:

- raw_events.csv / raw_events.parquet
- window_rollup.csv / window_rollup.parquet
- route_aware_rollout_writer_manifest.json
- route_aware_rollout_writer_report.md

Safety contract:
- No DB (Database=데이터베이스) write.
- No tensor DB (Database=데이터베이스) overwrite.
- No additional API (Application Programming Interface=응용 프로그램 인터페이스) calls.
- Reads existing CSV/JSON artifacts only.
- This is a scaffold writer, not a validated causal performance simulator.
- paper_level_claim_allowed=false.
- causal_performance_claim_allowed=false.

Default inputs:
- artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/
  causal_simulator_v2_contract.json
- artifacts/daegu_bis_api_audit/getbs02_bulk_collect/20260428_230936/
  getbs02_route_stop_sequence_normalized.csv

Default output:
- artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102/

The output folder intentionally contains a direct `window_rollup.parquet` path
so the existing canonical KPI (Key Performance Indicator=핵심 성과 지표)
aggregator can discover it in the next step.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ARTIFACT_VERSION = "route_aware_rollout_writer_step102_v1"
ROLLOUT_SCHEMA_VERSION = "route_aware_rollout_schema_step102_v1"
SOURCE_MODE = "route_aware_minimal_scaffold_step102_noncausal"

DEFAULT_CONTRACT_JSON = Path(
    "artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/"
    "causal_simulator_v2_contract.json"
)
DEFAULT_GETBS02_CSV = Path(
    "artifacts/daegu_bis_api_audit/getbs02_bulk_collect/20260428_230936/"
    "getbs02_route_stop_sequence_normalized.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/daegu_bis_api_audit/route_aware_rollout_writer_step102"
)

CLAIM_GUARDS: Dict[str, bool] = {
    "db_write_performed": False,
    "tensor_db_overwrite_performed": False,
    "additional_api_calls_performed": False,
    "paper_level_claim_allowed": False,
    "causal_performance_claim_allowed": False,
    "actual_headway_observed": False,
    "actual_arrival_departure_time_observed": False,
    "actual_dwell_observed": False,
}

CANONICAL_12_KPIS = [
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

REQUIRED_OFFICIAL_ROLLUP_COLUMNS = [
    "condition_id",
    "seed",
    "window_id",
    "state_ts",
    "service_date",
    "time_band",
    "evaluation_horizon_minutes",
    "headway_mean_seconds",
    "headway_std_seconds",
    "headway_sample_count",
    "bunching_event_count",
    "headway_event_count",
    "wait_total_passenger_seconds",
    "wait_passenger_count",
    "ontime_event_count",
    "schedulable_arrival_count",
    "intervention_count",
    "decision_step_count",
    "energy_proxy_total",
    "source_mode",
]

# Same-directory import.  The package is designed to be copied into
# 05_training/adapters next to the Step 101 scaffold.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

try:
    from route_aware_minimal_simulator_step101 import (  # type: ignore
        RouteAwareMinimalSimulatorStep101,
        RouteStopRecord,
        build_route_table,
        clean_str,
        load_json_any_encoding,
        read_route_sequence_csv,
        route_readiness_rows,
        summarize_route_readiness,
        validate_contract_for_step101,
    )
except Exception as exc:  # pragma: no cover - clearer runtime failure
    raise RuntimeError(
        "Step 102 requires route_aware_minimal_simulator_step101.py in the same directory. "
        "Copy both Step 101 and Step 102 files into 05_training/adapters."
    ) from exc


@dataclass(frozen=True)
class Step102Paths:
    contract_json: Path
    route_sequence_csv: Path
    output_root: Path


@dataclass(frozen=True)
class RolloutScenario:
    condition_id: str
    seed: int
    route_id: str
    direction_id: str
    scenario_index: int
    service_date: str
    state_ts: str
    time_band: str
    evaluation_horizon_minutes: int = 30

    @property
    def window_id(self) -> str:
        route = str(self.route_id).replace("/", "_")
        direction = str(self.direction_id).replace("/", "_")
        return (
            f"step102_{self.condition_id}_seed{self.seed:03d}_"
            f"route{route}_dir{direction}_scenario{self.scenario_index:03d}"
        )


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for x in parse_csv_list(text):
        values.append(int(x))
    if not values:
        raise ValueError("at least one seed is required")
    return values


def fleet_ratio_for_condition(condition_id: str) -> float:
    cid = str(condition_id).strip().upper()
    if cid == "A90":
        return 0.90
    if cid == "A80":
        return 0.80
    if cid == "A70":
        return 0.70
    return 1.00


def active_bus_count_for_condition(condition_id: str, baseline_bus_count: int) -> int:
    return max(1, int(round(float(baseline_bus_count) * fleet_ratio_for_condition(condition_id))))


def select_route_scenarios(
    route_table: Mapping[Tuple[str, str], List[RouteStopRecord]],
    *,
    max_routes: int,
    route_id: str = "",
    direction_id: str = "",
) -> List[Tuple[str, str]]:
    if route_id and direction_id:
        key = (str(route_id), str(direction_id))
        if key not in route_table:
            raise KeyError(f"requested route/direction not found: {key}")
        if len(route_table[key]) < 2:
            raise RuntimeError(f"requested route/direction has fewer than two stops: {key}")
        return [key]

    eligible = [key for key, rows in sorted(route_table.items()) if len(rows) >= 2]
    if not eligible:
        raise RuntimeError("no route-direction with at least two stops is available")
    return eligible[: max(1, int(max_routes))]


def choose_action(step_index: int, agent_id: int, seed: int, condition_id: str) -> str:
    """Deterministic scaffold policy.

    This is not a trained policy.  It simply creates a small mixture of
    advance/hold events so the rollout writer has non-empty action metadata.
    """
    cid = str(condition_id).strip().upper()
    hold_mod = 6 if cid == "A" else 5
    if (int(step_index) + int(agent_id) + int(seed)) % hold_mod == 0:
        return "hold"
    return "advance"


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float(default)


def safe_std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) < 2:
        return 0.0
    return float(statistics.pstdev(vals))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def try_write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Write parquet when pandas/pyarrow is available.

    The scaffold always writes CSV.  Parquet is attempted because the canonical
    KPI aggregator discovers `window_rollup.parquet` in the next step.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        return {
            "requested": True,
            "written": False,
            "reason": f"pandas import failed: {type(exc).__name__}: {exc}",
        }

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(list(rows)).to_parquet(path, index=False)
        return {"requested": True, "written": True, "path": str(path)}
    except Exception as exc:
        return {
            "requested": True,
            "written": False,
            "reason": f"parquet write failed: {type(exc).__name__}: {exc}",
        }


def validate_rollup_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise RuntimeError("window_rollup rows are empty")
    missing_required = sorted(
        col for col in REQUIRED_OFFICIAL_ROLLUP_COLUMNS
        if any(col not in row for row in rows)
    )
    if missing_required:
        raise RuntimeError(f"window_rollup missing required columns: {missing_required}")
    missing_kpis = sorted(
        col for col in CANONICAL_12_KPIS
        if any(col not in row for row in rows)
    )
    if missing_kpis:
        raise RuntimeError(f"window_rollup missing 12-KPI columns: {missing_kpis}")

    for row in rows:
        if bool(row.get("paper_level_claim_allowed")):
            raise RuntimeError("paper_level_claim_allowed must be false")
        if bool(row.get("causal_performance_claim_allowed")):
            raise RuntimeError("causal_performance_claim_allowed must be false")
        if bool(row.get("actual_headway_observed")):
            raise RuntimeError("actual_headway_observed must be false")
        if bool(row.get("actual_arrival_departure_time_observed")):
            raise RuntimeError("actual_arrival_departure_time_observed must be false")
        if bool(row.get("actual_dwell_observed")):
            raise RuntimeError("actual_dwell_observed must be false")
    return {
        "row_count": int(len(rows)),
        "condition_ids": sorted({str(row["condition_id"]) for row in rows}),
        "seed_values": sorted({int(row["seed"]) for row in rows}),
        "all_claim_guards_false": True,
        "required_official_columns_present": True,
        "canonical_12_kpis_present": True,
    }


def build_window_rollup_from_events(
    scenario: RolloutScenario,
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    route_stop_count: int,
    baseline_bus_count: int,
    active_bus_count: int,
) -> Dict[str, Any]:
    if not raw_rows:
        raise RuntimeError(f"raw rows are empty for scenario: {scenario.window_id}")

    by_step: Dict[int, List[int]] = {}
    hold_count = 0
    advance_count = 0
    terminal_count = 0
    reward_values: List[float] = []
    for row in raw_rows:
        step = int(row["step_index"])
        pos = int(row["stop_index"])
        by_step.setdefault(step, []).append(pos)
        if str(row.get("action", "")).lower() == "hold":
            hold_count += 1
        else:
            advance_count += 1
        if str(row.get("at_terminal", "")).lower() in {"true", "1"}:
            terminal_count += 1
        try:
            reward_values.append(float(row.get("reward", 0.0)))
        except Exception:
            pass

    stop_gap_samples: List[float] = []
    for positions in by_step.values():
        sorted_pos = sorted(int(x) for x in positions)
        for a, b in zip(sorted_pos[:-1], sorted_pos[1:]):
            gap = max(0, b - a)
            stop_gap_samples.append(float(gap))

    # Headway remains a route-stop gap proxy.  Convert stop gaps to seconds only
    # to satisfy the existing canonical rollup schema.  It is not observed.
    headway_seconds = [max(60.0, gap * 90.0) for gap in stop_gap_samples]
    if not headway_seconds:
        headway_seconds = [float(max(60, route_stop_count * 30))]

    headway_mean_seconds = safe_mean(headway_seconds, default=600.0)
    headway_std_seconds = safe_std(headway_seconds)
    cv_headway = headway_std_seconds / headway_mean_seconds if headway_mean_seconds > 0 else None

    headway_event_count = int(len(headway_seconds))
    bunching_event_count = int(sum(1 for x in headway_seconds if x <= 90.0))
    bunching_rate = float(bunching_event_count / headway_event_count) if headway_event_count > 0 else 0.0

    wait_passenger_count = int(max(1, route_stop_count * 2 + active_bus_count * 3))
    avg_wait_seconds = float(max(30.0, headway_mean_seconds / 2.0 + hold_count * 3.0))
    wait_total_passenger_seconds = float(avg_wait_seconds * wait_passenger_count)

    schedulable_arrival_count = int(max(1, headway_event_count))
    ontime_penalty = min(schedulable_arrival_count, int(round(hold_count / max(1, active_bus_count))))
    ontime_event_count = int(max(0, schedulable_arrival_count - ontime_penalty))
    on_time_rate = float(ontime_event_count / schedulable_arrival_count) if schedulable_arrival_count > 0 else 0.0

    decision_step_count = int(len(raw_rows))
    intervention_count = int(hold_count)
    intervention_rate = float(intervention_count / decision_step_count) if decision_step_count > 0 else 0.0

    passenger_demand_generated = int(wait_passenger_count + active_bus_count * len(by_step))
    passenger_served_count = int(min(passenger_demand_generated, max(1, advance_count * 5 + terminal_count)))
    passenger_service_rate = float(passenger_served_count / passenger_demand_generated) if passenger_demand_generated > 0 else 0.0
    passenger_wait_p95_seconds = float(avg_wait_seconds * 1.50 + intervention_count * 2.0)

    distance_m = float(advance_count * 350.0)
    acceleration_event_count = int(advance_count)
    hold_seconds = float(hold_count * 30.0)

    # Daegu energy proxy v1 constants used in earlier project contracts.
    k_dist = 0.0012
    k_acc = 0.1800
    k_idle = 0.0080
    energy_proxy_total = float(k_dist * distance_m + k_acc * acceleration_event_count + k_idle * hold_seconds)
    energy_proxy_per_passenger = float(energy_proxy_total / max(1, passenger_served_count))
    fleet_reduction_ratio = float(max(0.0, 1.0 - active_bus_count / max(1, baseline_bus_count)))

    row: Dict[str, Any] = {
        "condition_id": scenario.condition_id,
        "seed": int(scenario.seed),
        "window_id": scenario.window_id,
        "state_ts": scenario.state_ts,
        "service_date": scenario.service_date,
        "time_band": scenario.time_band,
        "evaluation_horizon_minutes": int(scenario.evaluation_horizon_minutes),
        "qwen_trigger_rate": 0.0,
        "effective_replay_step_minutes": 1.0,
        "headway_mean_seconds": headway_mean_seconds,
        "headway_std_seconds": headway_std_seconds,
        "headway_sample_count": int(headway_event_count),
        "bunching_event_count": bunching_event_count,
        "headway_event_count": headway_event_count,
        "wait_total_passenger_seconds": wait_total_passenger_seconds,
        "wait_passenger_count": wait_passenger_count,
        "ontime_event_count": ontime_event_count,
        "schedulable_arrival_count": schedulable_arrival_count,
        "intervention_count": intervention_count,
        "decision_step_count": decision_step_count,
        "energy_proxy_total": energy_proxy_total,
        "source_mode": SOURCE_MODE,
        # 12-KPI explicit columns.
        "cv_headway": cv_headway,
        "avg_wait_seconds": avg_wait_seconds,
        "bunching_rate": bunching_rate,
        "on_time_rate": on_time_rate,
        "intervention_rate": intervention_rate,
        "energy_proxy": energy_proxy_total,
        "passenger_demand_generated": passenger_demand_generated,
        "passenger_served_count": passenger_served_count,
        "passenger_service_rate": passenger_service_rate,
        "passenger_wait_p95_seconds": passenger_wait_p95_seconds,
        "energy_proxy_per_passenger": energy_proxy_per_passenger,
        "fleet_reduction_ratio": fleet_reduction_ratio,
        # Route-aware scaffold metadata.
        "route_id": scenario.route_id,
        "direction_id": scenario.direction_id,
        "route_stop_count": int(route_stop_count),
        "baseline_bus_count": int(baseline_bus_count),
        "active_bus_count": int(active_bus_count),
        "distance_m": distance_m,
        "acceleration_event_count": acceleration_event_count,
        "hold_seconds": hold_seconds,
        "reward_mean": safe_mean(reward_values, default=0.0),
        "rollout_schema_version": ROLLOUT_SCHEMA_VERSION,
        "policy_source": "deterministic_scaffold_policy",
        "policy_action_source_mode": "step102_deterministic_advance_hold",
        "trained_model": False,
        "checkpoint_loaded": False,
        "checkpoint_validator_ran": False,
        "mock_action_used": True,
        "placeholder_fallback_used": False,
        "actual_policy_claim_ready": False,
        "causal_policy_claim_ready": False,
        "strict_canonical": False,
        "causal_comparison_allowed": False,
        "scaffold_only": True,
        "not_actual_headway": True,
        "actual_headway_observed": False,
        "actual_arrival_departure_time_observed": False,
        "actual_dwell_observed": False,
        "paper_level_claim_allowed": False,
        "causal_performance_claim_allowed": False,
        "energy_proxy_model_version": "daegu_energy_proxy_v1",
        "k_dist_kwh_per_m": k_dist,
        "k_acc_kwh_per_event": k_acc,
        "k_idle_kwh_per_sec": k_idle,
    }
    return row


def run_one_scenario(
    route_table: Mapping[Tuple[str, str], List[RouteStopRecord]],
    *,
    contract: Mapping[str, Any],
    scenario: RolloutScenario,
    baseline_bus_count: int,
    max_steps: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    active_bus_count = active_bus_count_for_condition(scenario.condition_id, baseline_bus_count)
    sim = RouteAwareMinimalSimulatorStep101(
        route_table,
        contract=contract,
        num_agents=active_bus_count,
        max_steps=max_steps,
    )
    obs = sim.reset(seed=scenario.seed, scenario_config={
        "route_id": scenario.route_id,
        "direction_id": scenario.direction_id,
    })

    raw_rows: List[Dict[str, Any]] = []
    final_terminated = False
    final_truncated = False
    route_stop_count = int(obs.get("route_stop_count", 0))

    for _ in range(max_steps):
        step_for_action = int(sim.step_index) + 1
        actions = {
            agent_id: choose_action(step_for_action, agent_id, scenario.seed, scenario.condition_id)
            for agent_id in range(sim.num_agents)
        }
        result = sim.step(actions)
        final_terminated = bool(result.terminated)
        final_truncated = bool(result.truncated)
        applied = result.info.get("applied_actions", {})
        spacing = result.info.get("spacing_proxy", {})

        for agent in result.obs["agents"]:
            aid = int(agent["agent_id"])
            raw_rows.append({
                "artifact_version": ARTIFACT_VERSION,
                "rollout_schema_version": ROLLOUT_SCHEMA_VERSION,
                "condition_id": scenario.condition_id,
                "seed": int(scenario.seed),
                "window_id": scenario.window_id,
                "scenario_index": int(scenario.scenario_index),
                "state_ts": scenario.state_ts,
                "service_date": scenario.service_date,
                "time_band": scenario.time_band,
                "evaluation_horizon_minutes": int(scenario.evaluation_horizon_minutes),
                "step_index": int(result.obs["step_index"]),
                "agent_id": aid,
                "route_id": scenario.route_id,
                "direction_id": scenario.direction_id,
                "route_stop_count": int(result.obs["route_stop_count"]),
                "stop_index": int(agent["stop_index"]),
                "stop_order": int(agent["stop_order"]),
                "stop_id": agent["stop_id"],
                "next_stop_id": agent["next_stop_id"],
                "at_terminal": bool(agent["at_terminal"]),
                "action": str(applied.get(aid, actions.get(aid, "advance"))),
                "reward": float(result.rewards.get(aid, 0.0)),
                "terminated": bool(result.terminated),
                "truncated": bool(result.truncated),
                "mean_route_stop_gap_proxy": spacing.get("mean_route_stop_gap_proxy"),
                "cv_route_stop_gap_proxy": spacing.get("cv_route_stop_gap_proxy"),
                "source_mode": SOURCE_MODE,
                "policy_source": "deterministic_scaffold_policy",
                "mock_action_used": True,
                "trained_model": False,
                "scaffold_only": True,
                "not_actual_headway": True,
                "actual_headway_observed": False,
                "actual_arrival_departure_time_observed": False,
                "actual_dwell_observed": False,
                "paper_level_claim_allowed": False,
                "causal_performance_claim_allowed": False,
            })
        if result.terminated or result.truncated:
            break

    rollup_row = build_window_rollup_from_events(
        scenario,
        raw_rows,
        route_stop_count=route_stop_count,
        baseline_bus_count=baseline_bus_count,
        active_bus_count=active_bus_count,
    )
    rollup_row["terminated"] = final_terminated
    rollup_row["truncated"] = final_truncated
    return raw_rows, rollup_row


def build_scenarios(
    route_keys: Sequence[Tuple[str, str]],
    *,
    conditions: Sequence[str],
    seeds: Sequence[int],
    service_date: str,
    state_ts: str,
    time_band: str,
    evaluation_horizon_minutes: int,
) -> List[RolloutScenario]:
    scenarios: List[RolloutScenario] = []
    idx = 0
    for condition in conditions:
        cid = str(condition).strip().upper()
        if not cid:
            continue
        for seed in seeds:
            for route_id, direction_id in route_keys:
                idx += 1
                scenarios.append(RolloutScenario(
                    condition_id=cid,
                    seed=int(seed),
                    route_id=str(route_id),
                    direction_id=str(direction_id),
                    scenario_index=idx,
                    service_date=service_date,
                    state_ts=state_ts,
                    time_band=time_band,
                    evaluation_horizon_minutes=int(evaluation_horizon_minutes),
                ))
    if not scenarios:
        raise ValueError("no scenarios were generated")
    return scenarios


def build_markdown_report(manifest: Mapping[str, Any]) -> str:
    rc = manifest["row_counts"]
    return f"""# Step 102 — Route-aware Rollout Writer Scaffold

## Status

- audit_status: `{manifest['audit_status']}`
- rollout_status: `{manifest['rollout_status']}`
- artifact_version: `{manifest['artifact_version']}`
- rollout_schema_version: `{manifest['rollout_schema_version']}`

## Outputs

- raw_events_csv: `{manifest['output_files']['raw_events_csv']}`
- raw_events_parquet: `{manifest['output_files'].get('raw_events_parquet', '')}`
- window_rollup_csv: `{manifest['output_files']['window_rollup_csv']}`
- window_rollup_parquet: `{manifest['output_files'].get('window_rollup_parquet', '')}`

## Row Counts

- raw_event_rows: `{rc['raw_events']}`
- window_rollup_rows: `{rc['window_rollup']}`
- scenario_count: `{rc['scenarios']}`

## Safety Boundary

This step writes rollout-shaped files from a route-aware scaffold. It is still
not a validated causal simulator.

- DB (Database=데이터베이스) write performed: `{manifest['claim_guards']['db_write_performed']}`
- tensor DB (Database=데이터베이스) overwrite performed: `{manifest['claim_guards']['tensor_db_overwrite_performed']}`
- additional API (Application Programming Interface=응용 프로그램 인터페이스) calls performed: `{manifest['claim_guards']['additional_api_calls_performed']}`
- paper_level_claim_allowed: `{manifest['claim_guards']['paper_level_claim_allowed']}`
- causal_performance_claim_allowed: `{manifest['claim_guards']['causal_performance_claim_allowed']}`

## Claim Boundary

The generated `headway_mean_seconds` and `cv_headway` values are route-stop gap
proxies converted to seconds only to satisfy the current canonical rollout
schema. They are **not** actual observed headway. Actual arrival/departure time
and dwell time remain not observed.
"""


def run_step102(
    paths: Step102Paths,
    *,
    conditions: Sequence[str],
    seeds: Sequence[int],
    max_routes: int,
    baseline_bus_count: int,
    max_steps: int,
    service_date: str,
    state_ts: str,
    time_band: str,
    evaluation_horizon_minutes: int,
    route_id: str = "",
    direction_id: str = "",
) -> Dict[str, Any]:
    contract = load_json_any_encoding(paths.contract_json)
    contract_validation = validate_contract_for_step101(contract)
    records = read_route_sequence_csv(paths.route_sequence_csv)
    route_table = build_route_table(records)
    readiness = route_readiness_rows(route_table)
    readiness_summary = summarize_route_readiness(readiness)
    if readiness_summary["eligible_route_direction_count"] <= 0:
        raise RuntimeError("no route-direction sequence is eligible for Step 102")

    route_keys = select_route_scenarios(route_table, max_routes=max_routes, route_id=route_id, direction_id=direction_id)
    scenarios = build_scenarios(
        route_keys,
        conditions=conditions,
        seeds=seeds,
        service_date=service_date,
        state_ts=state_ts,
        time_band=time_band,
        evaluation_horizon_minutes=evaluation_horizon_minutes,
    )

    all_raw_rows: List[Dict[str, Any]] = []
    all_rollup_rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        raw_rows, rollup_row = run_one_scenario(
            route_table,
            contract=contract,
            scenario=scenario,
            baseline_bus_count=baseline_bus_count,
            max_steps=max_steps,
        )
        all_raw_rows.extend(raw_rows)
        all_rollup_rows.append(rollup_row)

    validation = validate_rollup_rows(all_rollup_rows)

    output_root = paths.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    raw_csv = output_root / "raw_events.csv"
    rollup_csv = output_root / "window_rollup.csv"
    raw_parquet = output_root / "raw_events.parquet"
    rollup_parquet = output_root / "window_rollup.parquet"
    manifest_json = output_root / "route_aware_rollout_writer_manifest.json"
    report_md = output_root / "route_aware_rollout_writer_report.md"

    write_csv(raw_csv, all_raw_rows)
    write_csv(rollup_csv, all_rollup_rows)
    raw_parquet_status = try_write_parquet(raw_parquet, all_raw_rows)
    rollup_parquet_status = try_write_parquet(rollup_parquet, all_rollup_rows)

    parquet_ready = bool(raw_parquet_status.get("written")) and bool(rollup_parquet_status.get("written"))
    rollout_status = "READY_FOR_STEP103_CANONICAL_KPI_SCAFFOLD" if parquet_ready else "PASS_CSV_ONLY_PARQUET_NOT_AVAILABLE"

    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "rollout_schema_version": ROLLOUT_SCHEMA_VERSION,
        "audit_status": "PASS",
        "rollout_status": rollout_status,
        "input_files": {
            "contract_json": str(paths.contract_json),
            "route_sequence_csv": str(paths.route_sequence_csv),
        },
        "output_files": {
            "manifest_json": str(manifest_json),
            "report_md": str(report_md),
            "raw_events_csv": str(raw_csv),
            "window_rollup_csv": str(rollup_csv),
            "raw_events_parquet": str(raw_parquet) if raw_parquet_status.get("written") else "",
            "window_rollup_parquet": str(rollup_parquet) if rollup_parquet_status.get("written") else "",
        },
        "parquet_status": {
            "raw_events": raw_parquet_status,
            "window_rollup": rollup_parquet_status,
        },
        "claim_guards": dict(CLAIM_GUARDS),
        "contract_validation": contract_validation,
        "route_sequence_readiness_summary": readiness_summary,
        "scenario_config": {
            "conditions": [str(x).strip().upper() for x in conditions],
            "seeds": [int(x) for x in seeds],
            "max_routes": int(max_routes),
            "selected_route_keys": [{"route_id": k[0], "direction_id": k[1]} for k in route_keys],
            "baseline_bus_count": int(baseline_bus_count),
            "max_steps": int(max_steps),
            "service_date": service_date,
            "state_ts": state_ts,
            "time_band": time_band,
            "evaluation_horizon_minutes": int(evaluation_horizon_minutes),
        },
        "row_counts": {
            "scenarios": int(len(scenarios)),
            "raw_events": int(len(all_raw_rows)),
            "window_rollup": int(len(all_rollup_rows)),
        },
        "window_rollup_validation": validation,
        "canonical_12_kpis": list(CANONICAL_12_KPIS),
        "not_observed_fields": [
            "actual_headway",
            "actual_arrival_departure_time",
            "actual_dwell",
        ],
        "next_step": "Step 103 — route-aware rollout canonical KPI integration scaffold",
    }
    dump_json(manifest_json, manifest)
    report_md.write_text(build_markdown_report(manifest), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 102 route-aware rollout writer scaffold")
    parser.add_argument("--contract-json", default=str(DEFAULT_CONTRACT_JSON))
    parser.add_argument("--route-sequence-csv", default=str(DEFAULT_GETBS02_CSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--conditions", default="A")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--max-routes", type=int, default=1)
    parser.add_argument("--baseline-bus-count", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--service-date", default="2023-01-01")
    parser.add_argument("--state-ts", default="2023-01-01T05:00:00+09:00")
    parser.add_argument("--time-band", default="offpeak")
    parser.add_argument("--evaluation-horizon-minutes", type=int, default=30)
    parser.add_argument("--route-id", default="")
    parser.add_argument("--direction-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_step102(
        Step102Paths(
            contract_json=Path(args.contract_json),
            route_sequence_csv=Path(args.route_sequence_csv),
            output_root=Path(args.output_root),
        ),
        conditions=parse_csv_list(args.conditions),
        seeds=parse_int_list(args.seeds),
        max_routes=args.max_routes,
        baseline_bus_count=args.baseline_bus_count,
        max_steps=args.max_steps,
        service_date=args.service_date,
        state_ts=args.state_ts,
        time_band=args.time_band,
        evaluation_horizon_minutes=args.evaluation_horizon_minutes,
        route_id=args.route_id,
        direction_id=args.direction_id,
    )
    print("[OK] Step 102 route-aware rollout writer scaffold completed")
    print(f"[OK] audit_status  : {manifest['audit_status']}")
    print(f"[OK] rollout_status: {manifest['rollout_status']}")
    print(f"[OK] output_root   : {Path(manifest['output_files']['manifest_json']).parent}")
    print(f"[OK] raw_events    : {manifest['row_counts']['raw_events']}")
    print(f"[OK] window_rollup : {manifest['row_counts']['window_rollup']}")
    print(f"[OK] parquet_ready : {bool(manifest['parquet_status']['window_rollup'].get('written'))}")


if __name__ == "__main__":
    main()
