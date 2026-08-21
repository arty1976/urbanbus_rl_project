#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 101 — route-aware minimal causal simulator scaffold.

This script is the first executable scaffold after the Step 100 causal
simulator v2 contract.  It builds a minimal route-aware simulator from the
/getBs02 route-stop sequence artifact only.

Safety contract:
- No DB (Database=데이터베이스) write.
- No tensor DB (Database=데이터베이스) overwrite.
- No additional API (Application Programming Interface=응용 프로그램 인터페이스) calls.
- Reads existing CSV/JSON artifacts only.
- This is a scaffold/smoke simulator, not a validated causal performance
  simulator.
- paper_level_claim_allowed=false.
- causal_performance_claim_allowed=false.

Default inputs:
- artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/
  causal_simulator_v2_contract.json
- artifacts/daegu_bis_api_audit/getbs02_bulk_collect/20260428_230936/
  getbs02_route_stop_sequence_normalized.csv

Outputs:
- artifacts/daegu_bis_api_audit/route_aware_minimal_simulator_step101/
  route_aware_minimal_simulator_manifest.json
- artifacts/daegu_bis_api_audit/route_aware_minimal_simulator_step101/
  route_aware_minimal_simulator_report.md
- artifacts/daegu_bis_api_audit/route_aware_minimal_simulator_step101/
  route_sequence_readiness_step101.csv
- artifacts/daegu_bis_api_audit/route_aware_minimal_simulator_step101/
  smoke_rollout_trace_step101.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --- H4M-AE-R9.8 fail-closed simulator authorization -------------------------------
import sys as _authz_sys
from pathlib import Path as _AuthzPath

for _authz_dir in (_AuthzPath(__file__).resolve().parent, _AuthzPath(__file__).resolve().parent.parent):
    if (_authz_dir / "simulator_authorization.py").exists():
        if str(_authz_dir) not in _authz_sys.path:
            _authz_sys.path.insert(0, str(_authz_dir))
        break
import simulator_authorization as _authz  # noqa: E402
# -----------------------------------------------------------------------------------


ARTIFACT_VERSION = "route_aware_minimal_simulator_step101_v1"
SCAFFOLD_VERSION = "route_aware_minimal_scaffold_v1"

DEFAULT_CONTRACT_JSON = Path(
    "artifacts/daegu_bis_api_audit/causal_simulator_v2_contract_step100/"
    "causal_simulator_v2_contract.json"
)
DEFAULT_GETBS02_CSV = Path(
    "artifacts/daegu_bis_api_audit/getbs02_bulk_collect/20260428_230936/"
    "getbs02_route_stop_sequence_normalized.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/daegu_bis_api_audit/route_aware_minimal_simulator_step101"
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

EXPECTED_CONTRACT_STATUS = "READY_FOR_ROUTE_AWARE_MINIMAL_SCAFFOLD"

# Column aliases are intentionally broad because raw public API outputs often
# use camelCase, while normalized audit artifacts use snake_case.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "route_id": (
        "route_id", "routeId", "ROUTE_ID", "bs_route_id", "bsRouteId",
        "routeid", "rout_id",
    ),
    "direction_id": (
        "direction_id", "direction", "dir", "dir_id", "route_dir",
        "moveDir", "move_dir", "DIRECTION_ID",
    ),
    "stop_id": (
        "stop_id", "bsId", "bs_id", "station_id", "node_id", "stopId",
        "STOP_ID", "bus_stop_id",
    ),
    "stop_order": (
        "stop_order", "ordered_stop_sequence", "order", "seq", "bsSeq",
        "stop_seq", "route_seq", "station_order", "current_stop_order",
    ),
    "route_no": (
        "route_no", "routeNo", "route_name", "routeName", "route_nm",
        "route_num", "line_no",
    ),
    "stop_name": (
        "stop_name", "bsNm", "bs_nm", "station_name", "stopNm",
        "name", "STOP_NAME",
    ),
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


# Optional compatibility with the project adapter interface.  The script must
# also run as a standalone file when copied into 05_training/adapters.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_TRAINING_ROOT = _THIS_FILE.parents[1] if len(_THIS_FILE.parents) >= 2 else None
if _PROJECT_TRAINING_ROOT is not None:
    sys.path.insert(0, str(_PROJECT_TRAINING_ROOT))

try:  # pragma: no cover - exercised only inside the full project tree
    from simulator_adapter_interface import GraphSkeleton  # type: ignore
except Exception:  # pragma: no cover
    GraphSkeleton = None  # type: ignore


@dataclass(frozen=True)
class RouteStopRecord:
    route_id: str
    direction_id: str
    stop_id: str
    stop_order: int
    route_no: str = ""
    stop_name: str = ""


@dataclass(frozen=True)
class RouteSequenceKey:
    route_id: str
    direction_id: str

    def as_key(self) -> Tuple[str, str]:
        return (self.route_id, self.direction_id)


@dataclass
class MinimalStepResult:
    obs: Dict[str, Any]
    rewards: Dict[int, float]
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


@dataclass
class RouteReadinessRow:
    route_id: str
    direction_id: str
    route_no: str
    stop_count: int
    min_stop_order: int
    max_stop_order: int
    duplicate_stop_order_count: int
    duplicate_stop_id_count: int
    has_at_least_two_stops: bool
    scaffold_eligible: bool


@dataclass
class Step101Paths:
    contract_json: Path
    route_sequence_csv: Path
    output_root: Path


def load_json_any_encoding(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required JSON file not found: {path}")
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"failed to read JSON: {path}")


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def norm_name(name: str) -> str:
    return str(name).strip().replace("\ufeff", "")


def resolve_column(field: str, columns: Sequence[str], required: bool = True) -> Optional[str]:
    lookup = {norm_name(c).lower(): c for c in columns}
    for alias in COLUMN_ALIASES[field]:
        key = norm_name(alias).lower()
        if key in lookup:
            return lookup[key]
    if required:
        raise ValueError(
            f"required column for {field!r} not found. "
            f"aliases={COLUMN_ALIASES[field]}, columns={list(columns)}"
        )
    return None


def clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_int(value: Any, field_name: str) -> int:
    text = clean_str(value)
    if text == "":
        raise ValueError(f"empty integer field: {field_name}")
    try:
        return int(float(text))
    except Exception as exc:
        raise ValueError(f"failed to parse integer field {field_name}={value!r}") from exc


def read_route_sequence_csv(path: Path) -> List[RouteStopRecord]:
    if not path.exists():
        raise FileNotFoundError(f"route sequence CSV not found: {path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        columns = [norm_name(c) for c in reader.fieldnames]
        route_col = resolve_column("route_id", columns)
        direction_col = resolve_column("direction_id", columns)
        stop_col = resolve_column("stop_id", columns)
        order_col = resolve_column("stop_order", columns)
        route_no_col = resolve_column("route_no", columns, required=False)
        stop_name_col = resolve_column("stop_name", columns, required=False)

        records: List[RouteStopRecord] = []
        for row_index, row in enumerate(reader, start=2):
            route_id = clean_str(row.get(route_col, ""))
            direction_id = clean_str(row.get(direction_col, ""))
            stop_id = clean_str(row.get(stop_col, ""))
            if not route_id or not direction_id or not stop_id:
                # Keep the loader strict for scaffold safety.  Empty route keys
                # should have been cleaned by the upstream audit.
                raise ValueError(
                    f"empty route/direction/stop at row {row_index}: "
                    f"route_id={route_id!r}, direction_id={direction_id!r}, stop_id={stop_id!r}"
                )
            records.append(RouteStopRecord(
                route_id=route_id,
                direction_id=direction_id,
                stop_id=stop_id,
                stop_order=parse_int(row.get(order_col, ""), "stop_order"),
                route_no=clean_str(row.get(route_no_col, "")) if route_no_col else "",
                stop_name=clean_str(row.get(stop_name_col, "")) if stop_name_col else "",
            ))

    if not records:
        raise ValueError(f"route sequence CSV contains zero rows: {path}")
    return records


def build_route_table(records: Iterable[RouteStopRecord]) -> Dict[Tuple[str, str], List[RouteStopRecord]]:
    table: Dict[Tuple[str, str], List[RouteStopRecord]] = {}
    for rec in records:
        table.setdefault((rec.route_id, rec.direction_id), []).append(rec)
    for key, rows in table.items():
        rows.sort(key=lambda r: (r.stop_order, r.stop_id))
    return table


def route_readiness_rows(route_table: Mapping[Tuple[str, str], List[RouteStopRecord]]) -> List[RouteReadinessRow]:
    out: List[RouteReadinessRow] = []
    for (route_id, direction_id), rows in sorted(route_table.items()):
        orders = [r.stop_order for r in rows]
        stops = [r.stop_id for r in rows]
        duplicate_order_count = len(orders) - len(set(orders))
        duplicate_stop_count = len(stops) - len(set(stops))
        has_two = len(rows) >= 2
        out.append(RouteReadinessRow(
            route_id=route_id,
            direction_id=direction_id,
            route_no=rows[0].route_no if rows else "",
            stop_count=len(rows),
            min_stop_order=min(orders) if orders else -1,
            max_stop_order=max(orders) if orders else -1,
            duplicate_stop_order_count=duplicate_order_count,
            duplicate_stop_id_count=duplicate_stop_count,
            has_at_least_two_stops=has_two,
            scaffold_eligible=bool(has_two and duplicate_order_count == 0),
        ))
    return out


def summarize_route_readiness(rows: Sequence[RouteReadinessRow]) -> Dict[str, Any]:
    eligible = [r for r in rows if r.scaffold_eligible]
    return {
        "route_direction_count": int(len(rows)),
        "eligible_route_direction_count": int(len(eligible)),
        "ineligible_route_direction_count": int(len(rows) - len(eligible)),
        "total_stop_rows": int(sum(r.stop_count for r in rows)),
        "duplicate_stop_order_route_direction_count": int(sum(1 for r in rows if r.duplicate_stop_order_count > 0)),
        "duplicate_stop_id_route_direction_count": int(sum(1 for r in rows if r.duplicate_stop_id_count > 0)),
        "min_stop_count": int(min((r.stop_count for r in rows), default=0)),
        "max_stop_count": int(max((r.stop_count for r in rows), default=0)),
    }


def write_readiness_csv(path: Path, rows: Sequence[RouteReadinessRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else [
        "route_id", "direction_id", "route_no", "stop_count",
        "min_stop_order", "max_stop_order", "duplicate_stop_order_count",
        "duplicate_stop_id_count", "has_at_least_two_stops", "scaffold_eligible",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


class RouteAwareMinimalSimulatorStep101:
    """Minimal route-aware scaffold over a single route-direction sequence.

    The simulator only proves that a bus-like agent can move monotonically along
    a /getBs02 ordered route-stop sequence.  It does not claim actual headway,
    dwell time, observed arrival/departure events, or causal performance.
    """

    def __init__(
        self,
        route_table: Mapping[Tuple[str, str], List[RouteStopRecord]],
        *,
        contract: Optional[Mapping[str, Any]] = None,
        num_agents: int = 3,
        max_steps: int = 8,
    ) -> None:
        self.route_table = {k: list(v) for k, v in route_table.items()}
        self.contract = dict(contract or {})
        self.num_agents_val = int(num_agents)
        self.max_steps = int(max_steps)
        if self.num_agents_val <= 0:
            raise ValueError("num_agents must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.current_key: Optional[Tuple[str, str]] = None
        self.current_route: List[RouteStopRecord] = []
        self.agent_positions: Dict[int, int] = {}
        self.step_index = 0
        self.rng = random.Random(0)

    @property
    def num_agents(self) -> int:
        return self.num_agents_val

    @property
    def observation_space(self) -> Dict[str, Any]:
        return {
            "type": "route_aware_minimal_observation_v1",
            "route_id": "str",
            "direction_id": "str",
            "agent_state": ["agent_id", "stop_index", "stop_id", "next_stop_id", "at_terminal"],
            "claim_guards": dict(CLAIM_GUARDS),
        }

    @property
    def action_space(self) -> Dict[str, Any]:
        return {
            "type": "discrete_or_string",
            "actions": {
                "0": "hold_one_step_scaffold_action",
                "1": "advance_to_next_stop_scaffold_action",
                "hold": "hold_one_step_scaffold_action",
                "advance": "advance_to_next_stop_scaffold_action",
            },
            "not_actual_dispatch_control": True,
        }

    def _select_route_key(self, scenario_config: Optional[Mapping[str, Any]]) -> Tuple[str, str]:
        scenario_config = scenario_config or {}
        route_id = clean_str(scenario_config.get("route_id"))
        direction_id = clean_str(scenario_config.get("direction_id"))
        if route_id and direction_id:
            key = (route_id, direction_id)
            if key not in self.route_table:
                raise KeyError(f"route_id/direction_id not found in route table: {key}")
            return key

        eligible = [k for k, rows in sorted(self.route_table.items()) if len(rows) >= 2]
        if not eligible:
            raise RuntimeError("no route-direction sequence with at least two stops is available")
        return eligible[0]

    def reset(self, seed: Optional[int] = None, scenario_config: Optional[dict] = None) -> Dict[str, Any]:
        _authz.require_capability("simulator_execution", site="adapters/route_aware_minimal_simulator_step101.py::reset")
        if seed is not None:
            self.rng.seed(int(seed))
        self.current_key = self._select_route_key(scenario_config)
        self.current_route = list(self.route_table[self.current_key])
        if len(self.current_route) < 2:
            raise RuntimeError(f"selected route-direction has fewer than two stops: {self.current_key}")

        route_len = len(self.current_route)
        spacing = max(1, route_len // max(1, self.num_agents_val))
        self.agent_positions = {
            agent_id: min(route_len - 1, agent_id * spacing)
            for agent_id in range(self.num_agents_val)
        }
        self.step_index = 0
        return self._make_obs()

    def _make_agent_obs(self, agent_id: int, stop_index: int) -> Dict[str, Any]:
        last = len(self.current_route) - 1
        stop_index = max(0, min(last, int(stop_index)))
        next_index = min(last, stop_index + 1)
        cur = self.current_route[stop_index]
        nxt = self.current_route[next_index]
        return {
            "agent_id": int(agent_id),
            "stop_index": int(stop_index),
            "stop_order": int(cur.stop_order),
            "stop_id": cur.stop_id,
            "stop_name": cur.stop_name,
            "next_stop_index": int(next_index),
            "next_stop_order": int(nxt.stop_order),
            "next_stop_id": nxt.stop_id,
            "at_terminal": bool(stop_index >= last),
        }

    def _make_obs(self) -> Dict[str, Any]:
        if self.current_key is None:
            raise RuntimeError("reset() must be called before observation is available")
        route_id, direction_id = self.current_key
        return {
            "artifact_version": ARTIFACT_VERSION,
            "scaffold_version": SCAFFOLD_VERSION,
            "route_id": route_id,
            "direction_id": direction_id,
            "route_stop_count": int(len(self.current_route)),
            "step_index": int(self.step_index),
            "agents": [
                self._make_agent_obs(agent_id, pos)
                for agent_id, pos in sorted(self.agent_positions.items())
            ],
            "claim_guards": dict(CLAIM_GUARDS),
            "data_boundaries": {
                "route_sequence_source": "getBs02_ordered_stop_sequence",
                "headway_source": "simulated_spacing_proxy_only",
                "eta_based_headway_used": False,
                "actual_headway_observed": False,
                "actual_arrival_departure_time_observed": False,
                "actual_dwell_observed": False,
            },
        }

    @staticmethod
    def _is_advance(action: Any) -> bool:
        if isinstance(action, str):
            return action.strip().lower() in {"1", "advance", "move", "go", "depart"}
        try:
            return int(action) == 1
        except Exception:
            return False

    def _spacing_proxy(self) -> Dict[str, Any]:
        positions = sorted(self.agent_positions.values())
        if len(positions) < 2:
            gaps: List[int] = []
        else:
            gaps = [int(b - a) for a, b in zip(positions[:-1], positions[1:])]
        if gaps:
            mean_gap = sum(gaps) / len(gaps)
            variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
            cv = math.sqrt(variance) / mean_gap if mean_gap > 0 else None
        else:
            mean_gap = None
            cv = None
        return {
            "route_stop_gap_samples": gaps,
            "mean_route_stop_gap_proxy": mean_gap,
            "cv_route_stop_gap_proxy": cv,
            "not_actual_headway": True,
        }

    def step(self, actions: Mapping[int, Any]) -> MinimalStepResult:
        _authz.require_capability("simulator_execution", site="adapters/route_aware_minimal_simulator_step101.py::step")
        if self.current_key is None:
            raise RuntimeError("reset() must be called before step()")

        last = len(self.current_route) - 1
        applied_actions: Dict[int, str] = {}
        for agent_id in range(self.num_agents_val):
            action = actions.get(agent_id, 1)
            if self._is_advance(action):
                self.agent_positions[agent_id] = min(last, self.agent_positions[agent_id] + 1)
                applied_actions[agent_id] = "advance"
            else:
                self.agent_positions[agent_id] = min(last, self.agent_positions[agent_id])
                applied_actions[agent_id] = "hold"

        self.step_index += 1
        terminated = all(pos >= last for pos in self.agent_positions.values())
        truncated = self.step_index >= self.max_steps and not terminated
        spacing = self._spacing_proxy()

        # A tiny bounded smoke reward.  It only discourages everyone from being
        # on the exact same stop.  It is not the final MAPPO reward.
        if spacing["mean_route_stop_gap_proxy"] is None:
            reward_value = 0.0
        else:
            cv = spacing["cv_route_stop_gap_proxy"] or 0.0
            reward_value = float(max(-1.0, min(0.0, -cv)))
        rewards = {agent_id: reward_value for agent_id in range(self.num_agents_val)}

        info = {
            "applied_actions": applied_actions,
            "spacing_proxy": spacing,
            "scaffold_step_only": True,
            "actual_policy_claim_ready": False,
            "causal_policy_claim_ready": False,
            "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "note": (
                "Step 101 verifies route-order movement only. It does not "
                "observe actual headway, dwell, or arrival/departure times."
            ),
        }
        return MinimalStepResult(
            obs=self._make_obs(),
            rewards=rewards,
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=info,
        )

    def get_graph_skeleton(self) -> Any:
        if self.current_key is None:
            # Select deterministic first route if get_graph_skeleton is called
            # before reset by a contract smoke test.
            self.reset(seed=0, scenario_config=None)
        num_nodes = len(self.current_route)
        edge_index = [[i for i in range(max(0, num_nodes - 1))], [i + 1 for i in range(max(0, num_nodes - 1))]]
        edge_attr = [[1.0, 1.0, 1.0, 0.0] for _ in range(max(0, num_nodes - 1))]
        if GraphSkeleton is not None:
            return GraphSkeleton(
                num_nodes=num_nodes,
                num_edges=max(0, num_nodes - 1),
                node_features_dim=4,
                edge_features_dim=4,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
        return {
            "num_nodes": num_nodes,
            "num_edges": max(0, num_nodes - 1),
            "node_features_dim": 4,
            "edge_features_dim": 4,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }

    def compute_kpis(self, trajectory: Optional[List[MinimalStepResult]] = None) -> Dict[str, Optional[float]]:
        # These remain mostly empty because Step 101 is a scaffold.  Official 12
        # KPI generation belongs to later rollout/canonical aggregation steps.
        out: Dict[str, Optional[float]] = {k: None for k in CANONICAL_12_KPIS}
        out["intervention_rate"] = 0.0
        return out

    def close(self) -> None:
        return None


def validate_contract_for_step101(contract: Mapping[str, Any]) -> Dict[str, Any]:
    warnings: List[Dict[str, Any]] = []
    status = clean_str(contract.get("contract_status"))
    if not status:
        status = clean_str(contract.get("status"))
    if status != EXPECTED_CONTRACT_STATUS:
        warnings.append({
            "code": "unexpected_contract_status",
            "expected": EXPECTED_CONTRACT_STATUS,
            "observed": status,
            "message": "Step 101 can still build a scaffold, but Step 100 status should be checked.",
        })

    guards = dict(contract.get("claim_guards", {}))
    for key in ("paper_level_claim_allowed", "causal_performance_claim_allowed"):
        if bool(guards.get(key, False)):
            raise RuntimeError(f"contract guard violation: {key} must be false for Step 101")

    blocked = contract.get("blocked_actual_observed_fields") or contract.get("field_readiness", {}).get("blocked_actual_observed_fields")
    # The exact Step 100 schema may store blocked fields in a list/dict.  We do
    # not require a specific shape here, but the manifest records whether it was
    # present.
    return {
        "contract_status": status,
        "contract_status_expected": EXPECTED_CONTRACT_STATUS,
        "contract_status_ok": bool(status == EXPECTED_CONTRACT_STATUS),
        "blocked_actual_observed_fields_present": bool(blocked),
        "warnings": warnings,
    }


def select_smoke_scenario(
    route_table: Mapping[Tuple[str, str], List[RouteStopRecord]],
    preferred_route_id: str = "",
    preferred_direction_id: str = "",
) -> Tuple[str, str]:
    if preferred_route_id and preferred_direction_id:
        key = (preferred_route_id, preferred_direction_id)
        if key not in route_table:
            raise KeyError(f"preferred route-direction not found: {key}")
        if len(route_table[key]) < 2:
            raise RuntimeError(f"preferred route-direction has fewer than two stops: {key}")
        return key
    eligible = [k for k, rows in sorted(route_table.items()) if len(rows) >= 2]
    if not eligible:
        raise RuntimeError("no scaffold-eligible route direction found")
    return eligible[0]


def write_smoke_trace(path: Path, trace_rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step_index", "agent_id", "route_id", "direction_id", "stop_index",
        "stop_order", "stop_id", "next_stop_id", "action", "terminated",
        "truncated", "reward", "not_actual_headway",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in trace_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def run_smoke_rollout(
    sim: RouteAwareMinimalSimulatorStep101,
    *,
    scenario: Tuple[str, str],
    seed: int,
    max_steps: int,
) -> Tuple[List[Dict[str, Any]], MinimalStepResult]:
    route_id, direction_id = scenario
    obs = sim.reset(seed=seed, scenario_config={"route_id": route_id, "direction_id": direction_id})
    trace: List[Dict[str, Any]] = []

    result = MinimalStepResult(obs=obs, rewards={}, terminated=False, truncated=False, info={})
    for _ in range(max_steps):
        actions = {agent_id: 1 for agent_id in range(sim.num_agents)}
        result = sim.step(actions)
        for agent in result.obs["agents"]:
            aid = int(agent["agent_id"])
            trace.append({
                "step_index": int(result.obs["step_index"]),
                "agent_id": aid,
                "route_id": route_id,
                "direction_id": direction_id,
                "stop_index": int(agent["stop_index"]),
                "stop_order": int(agent["stop_order"]),
                "stop_id": agent["stop_id"],
                "next_stop_id": agent["next_stop_id"],
                "action": "advance",
                "terminated": bool(result.terminated),
                "truncated": bool(result.truncated),
                "reward": float(result.rewards.get(aid, 0.0)),
                "not_actual_headway": True,
            })
        if result.terminated or result.truncated:
            break
    return trace, result


def build_markdown_report(manifest: Mapping[str, Any]) -> str:
    readiness = manifest["route_sequence_readiness_summary"]
    smoke = manifest["smoke_rollout"]
    contract = manifest["contract_validation"]
    return f"""# Step 101 — Route-aware Minimal Simulator Scaffold

## Status

- audit_status: `{manifest['audit_status']}`
- scaffold_status: `{manifest['scaffold_status']}`
- artifact_version: `{manifest['artifact_version']}`
- contract_status: `{contract.get('contract_status', '')}`

## Safety Boundary

This step is a scaffold/smoke test only.

- DB (Database=데이터베이스) write performed: `{manifest['claim_guards']['db_write_performed']}`
- tensor DB (Database=데이터베이스) overwrite performed: `{manifest['claim_guards']['tensor_db_overwrite_performed']}`
- additional API (Application Programming Interface=응용 프로그램 인터페이스) calls performed: `{manifest['claim_guards']['additional_api_calls_performed']}`
- paper_level_claim_allowed: `{manifest['claim_guards']['paper_level_claim_allowed']}`
- causal_performance_claim_allowed: `{manifest['claim_guards']['causal_performance_claim_allowed']}`

## Route Sequence Readiness

- route_direction_count: `{readiness['route_direction_count']}`
- eligible_route_direction_count: `{readiness['eligible_route_direction_count']}`
- total_stop_rows: `{readiness['total_stop_rows']}`
- duplicate_stop_order_route_direction_count: `{readiness['duplicate_stop_order_route_direction_count']}`

## Smoke Rollout

- selected_route_id: `{smoke['selected_route_id']}`
- selected_direction_id: `{smoke['selected_direction_id']}`
- num_agents: `{smoke['num_agents']}`
- trace_rows: `{smoke['trace_rows']}`
- terminated: `{smoke['terminated']}`
- truncated: `{smoke['truncated']}`

## Claim Boundary

Step 101 verifies only that agents can move monotonically over a getBs02 ordered
route-stop sequence.  It does **not** observe actual headway, actual
arrival/departure time, or actual dwell time.  ETA (Estimated Time of
Arrival=예상 도착 시간) and getPos02 live-position candidates remain calibration
or alignment candidates, not paper-level causal performance evidence.
"""


def run_step101(paths: Step101Paths, *, num_agents: int, max_steps: int, seed: int, route_id: str = "", direction_id: str = "") -> Dict[str, Any]:
    contract = load_json_any_encoding(paths.contract_json)
    contract_validation = validate_contract_for_step101(contract)

    records = read_route_sequence_csv(paths.route_sequence_csv)
    route_table = build_route_table(records)
    readiness = route_readiness_rows(route_table)
    readiness_summary = summarize_route_readiness(readiness)
    if readiness_summary["eligible_route_direction_count"] <= 0:
        raise RuntimeError("no route-direction sequence is eligible for Step 101 scaffold")

    scenario = select_smoke_scenario(route_table, preferred_route_id=route_id, preferred_direction_id=direction_id)
    sim = RouteAwareMinimalSimulatorStep101(
        route_table,
        contract=contract,
        num_agents=num_agents,
        max_steps=max_steps,
    )
    trace_rows, final_result = run_smoke_rollout(sim, scenario=scenario, seed=seed, max_steps=max_steps)

    output_root = paths.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    readiness_csv = output_root / "route_sequence_readiness_step101.csv"
    trace_csv = output_root / "smoke_rollout_trace_step101.csv"
    manifest_json = output_root / "route_aware_minimal_simulator_manifest.json"
    report_md = output_root / "route_aware_minimal_simulator_report.md"

    write_readiness_csv(readiness_csv, readiness)
    write_smoke_trace(trace_csv, trace_rows)

    route_rows = route_table[scenario]
    manifest: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "scaffold_version": SCAFFOLD_VERSION,
        "audit_status": "PASS",
        "scaffold_status": "READY_FOR_STEP102_ROLLOUT_WRITER_SCAFFOLD",
        "input_files": {
            "contract_json": str(paths.contract_json),
            "route_sequence_csv": str(paths.route_sequence_csv),
        },
        "output_files": {
            "manifest_json": str(manifest_json),
            "report_md": str(report_md),
            "route_sequence_readiness_csv": str(readiness_csv),
            "smoke_rollout_trace_csv": str(trace_csv),
        },
        "claim_guards": dict(CLAIM_GUARDS),
        "contract_validation": contract_validation,
        "route_sequence_readiness_summary": readiness_summary,
        "smoke_rollout": {
            "selected_route_id": scenario[0],
            "selected_direction_id": scenario[1],
            "selected_route_no": route_rows[0].route_no if route_rows else "",
            "selected_stop_count": int(len(route_rows)),
            "num_agents": int(num_agents),
            "max_steps": int(max_steps),
            "seed": int(seed),
            "trace_rows": int(len(trace_rows)),
            "terminated": bool(final_result.terminated),
            "truncated": bool(final_result.truncated),
            "final_step_index": int(final_result.obs.get("step_index", 0)),
            "actual_headway_observed": False,
            "actual_arrival_departure_time_observed": False,
            "actual_dwell_observed": False,
        },
        "next_step": "Step 102 — route-aware rollout writer scaffold",
    }
    dump_json(manifest_json, manifest)
    report_md.write_text(build_markdown_report(manifest), encoding="utf-8")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 101 route-aware minimal simulator scaffold")
    parser.add_argument("--contract-json", default=str(DEFAULT_CONTRACT_JSON))
    parser.add_argument("--route-sequence-csv", default=str(DEFAULT_GETBS02_CSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--route-id", default="")
    parser.add_argument("--direction-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_step101(
        Step101Paths(
            contract_json=Path(args.contract_json),
            route_sequence_csv=Path(args.route_sequence_csv),
            output_root=Path(args.output_root),
        ),
        num_agents=args.num_agents,
        max_steps=args.max_steps,
        seed=args.seed,
        route_id=args.route_id,
        direction_id=args.direction_id,
    )
    print("[OK] Step 101 route-aware minimal simulator scaffold completed")
    print(f"[OK] audit_status   : {manifest['audit_status']}")
    print(f"[OK] scaffold_status: {manifest['scaffold_status']}")
    print(f"[OK] output_root    : {Path(manifest['output_files']['manifest_json']).parent}")
    print(f"[OK] manifest_json  : {manifest['output_files']['manifest_json']}")
    print(f"[OK] trace_rows     : {manifest['smoke_rollout']['trace_rows']}")


if __name__ == "__main__":
    main()
