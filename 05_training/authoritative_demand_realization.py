#!/usr/bin/env python3
"""Authoritative demand realization interface (H4M-AE-R5).

The causal simulator consumes an already-frozen demand realization.  Demand is
input data: it is never regenerated, expanded, rescaled or redrawn here.

The interface is deliberately artifact-agnostic.  Scaling from the Mac mini
reduced Suseong validation to a Daegu citywide, full-service-day study is a
change of artifact and configuration, not of accounting semantics.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

DEMAND_INTERFACE_ID = "AUTHORITATIVE_FROZEN_DEMAND_REALIZATION_V1"
DEMAND_UNIT = "ONE_ROW_ONE_REQUEST_UNIT_WEIGHT"
IDENTITY_LABEL = "simulator_internal_anonymous_request_identity"

# Semantic role -> column name in the frozen artifact.  Only the mapping moves
# when a different authoritative artifact is bound.
DEFAULT_COLUMN_MAP: Dict[str, str] = {
    "request_id": "request_id",
    "passenger_id": "passenger_id",
    "request_ts": "request_ts",
    "reporting_window_id": "window_id",
    "origin_stop_id": "origin_stop_id",
    "destination_stop_id": "destination_stop_id",
    "route_id": "route_id",
    "direction_id": "direction_id",
    "destination_realizable": "destination_realizable",
    "realization_status": "realization_status",
    "unrealizable_reason": "unrealizable_reason",
}
OPTIONAL_ROLES = ("destination_stop_id", "route_id", "direction_id",
                  "destination_realizable", "realization_status", "unrealizable_reason")
REQUIRED_ROLES = tuple(r for r in DEFAULT_COLUMN_MAP if r not in OPTIONAL_ROLES)


class DemandContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class EvaluationTimeContract:
    """The single external evaluation boundary every horizon-sensitive path uses."""

    evaluation_start_ts: float
    evaluation_end_ts: float
    reporting_window_ids: Sequence[str] = ()
    scope_label: str = "UNSPECIFIED_SCOPE"

    def __post_init__(self) -> None:
        if not (self.evaluation_end_ts > self.evaluation_start_ts):
            raise DemandContractError("INVALID_EVALUATION_BOUNDARY", f"{self.evaluation_start_ts} -> {self.evaluation_end_ts}")

    @property
    def evaluation_horizon_seconds(self) -> float:
        return float(self.evaluation_end_ts) - float(self.evaluation_start_ts)

    @property
    def evaluation_horizon_minutes(self) -> float:
        """Derived metadata only; never an input to accounting."""
        return self.evaluation_horizon_seconds / 60.0

    def payload(self) -> Dict[str, Any]:
        return {
            "evaluation_start_ts": float(self.evaluation_start_ts),
            "evaluation_end_ts": float(self.evaluation_end_ts),
            "evaluation_horizon_seconds": self.evaluation_horizon_seconds,
            "evaluation_horizon_minutes_derived": self.evaluation_horizon_minutes,
            "reporting_window_ids": list(self.reporting_window_ids),
            "reporting_window_count": len(self.reporting_window_ids),
            "scope_label": self.scope_label,
        }


def _optional(value: Any) -> Optional[str]:
    """Carry a value without stringifying a null.

    A request with no legal destination must reach a consumer as an absent
    destination, not as the text of a missing value.  `str()` on a null would
    produce "None" or "nan", which a consumer cannot distinguish from a stop id.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


@dataclass(frozen=True)
class DemandRequest:
    """One authoritative request.  Identity is carried, never fabricated.

    `destination_stop_id` is None exactly when the request has no legal
    destination; `destination_realizable` states the same fact explicitly so a
    consumer never has to infer it.
    """

    request_id: str
    passenger_id: str
    arrival_ts_absolute: float
    arrival_ts_relative: float
    origin_stop_id: str
    reporting_window_id: str
    weight: float = 1.0
    route_id: Optional[str] = None
    direction_id: Optional[str] = None
    destination_stop_id: Optional[str] = None
    destination_realizable: bool = True
    realization_status: Optional[str] = None
    unrealizable_reason: Optional[str] = None


@dataclass
class DemandRealization:
    """A frozen demand artifact bound by path and content hash."""

    artifact_path: Path
    artifact_sha256: str
    frame: pd.DataFrame
    column_map: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_COLUMN_MAP))

    @property
    def total_request_count(self) -> int:
        return int(len(self.frame))

    def requests_per_reporting_window(self) -> pd.Series:
        return self.frame.groupby(self.column_map["reporting_window_id"]).size()

    def provenance(self) -> Dict[str, Any]:
        per_window = self.requests_per_reporting_window()
        return {
            "interface_id": DEMAND_INTERFACE_ID,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": self.artifact_sha256,
            "unit": DEMAND_UNIT,
            "identity_label": IDENTITY_LABEL,
            "observed_individual_identity_claim": False,
            "column_map": dict(self.column_map),
            "total_request_count": self.total_request_count,
            "reporting_window_count": int(per_window.size),
            "median_requests_per_reporting_window": float(per_window.median()) if per_window.size else None,
            "min_requests_per_reporting_window": int(per_window.min()) if per_window.size else None,
            "max_requests_per_reporting_window": int(per_window.max()) if per_window.size else None,
            "regenerated": False,
            "rescaled": False,
            "expanded": False,
            "tuned": False,
        }


def load_demand_realization(artifact_path: Path, column_map: Optional[Mapping[str, str]] = None) -> DemandRealization:
    path = Path(artifact_path)
    if not path.exists():
        raise DemandContractError("DEMAND_ARTIFACT_MISSING", str(path))
    mapping = dict(DEFAULT_COLUMN_MAP)
    mapping.update(column_map or {})
    frame = pd.read_parquet(path)
    missing = [role for role in REQUIRED_ROLES if mapping[role] not in frame.columns]
    if missing:
        raise DemandContractError("DEMAND_SCHEMA_INCOMPLETE", f"missing roles {missing}")
    ids = frame[mapping["request_id"]]
    if ids.duplicated().any():
        raise DemandContractError("DUPLICATE_REQUEST_ID", f"{int(ids.duplicated().sum())} duplicate rows")
    return DemandRealization(
        artifact_path=path,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        frame=frame,
        column_map=mapping,
    )


def select_population(
    realization: DemandRealization,
    contract: EvaluationTimeContract,
    *,
    reporting_window_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Requests belonging to the configured evaluation population.

    A request joins the population when its authoritative timestamp lies inside
    the evaluation boundary.  Requests outside the boundary are reported, never
    silently dropped, and never rewritten to fit.
    """
    cmap = realization.column_map
    frame = realization.frame
    scope = list(reporting_window_ids if reporting_window_ids is not None else contract.reporting_window_ids)
    if scope:
        frame = frame[frame[cmap["reporting_window_id"]].astype(str).isin({str(w) for w in scope})]
    scoped_total = int(len(frame))

    ts = pd.to_numeric(frame[cmap["request_ts"]], errors="coerce")
    if ts.isna().any():
        raise DemandContractError("NON_NUMERIC_REQUEST_TS", f"{int(ts.isna().sum())} rows")
    inside = (ts >= contract.evaluation_start_ts) & (ts <= contract.evaluation_end_ts)
    before = int((ts < contract.evaluation_start_ts).sum())
    after = int((ts > contract.evaluation_end_ts).sum())

    selected = frame[inside].sort_values([cmap["request_ts"], cmap["request_id"]], kind="mergesort")
    requests: List[DemandRequest] = []
    for _, row in selected.iterrows():
        absolute = float(row[cmap["request_ts"]])
        requests.append(
            DemandRequest(
                request_id=str(row[cmap["request_id"]]),
                passenger_id=str(row[cmap["passenger_id"]]),
                arrival_ts_absolute=absolute,
                arrival_ts_relative=absolute - float(contract.evaluation_start_ts),
                origin_stop_id=str(row[cmap["origin_stop_id"]]),
                reporting_window_id=str(row[cmap["reporting_window_id"]]),
                route_id=_optional(row[cmap["route_id"]]) if cmap.get("route_id") in frame.columns else None,
                direction_id=_optional(row[cmap["direction_id"]]) if cmap.get("direction_id") in frame.columns else None,
                destination_stop_id=_optional(row[cmap["destination_stop_id"]]) if cmap.get("destination_stop_id") in frame.columns else None,
                destination_realizable=(bool(row[cmap["destination_realizable"]])
                                        if cmap.get("destination_realizable") in frame.columns
                                        else _optional(row[cmap["destination_stop_id"]]) is not None
                                        if cmap.get("destination_stop_id") in frame.columns else True),
                realization_status=_optional(row[cmap["realization_status"]]) if cmap.get("realization_status") in frame.columns else None,
                unrealizable_reason=_optional(row[cmap["unrealizable_reason"]]) if cmap.get("unrealizable_reason") in frame.columns else None,
            )
        )
    if any(r.arrival_ts_relative < 0.0 for r in requests):
        raise DemandContractError("NEGATIVE_RELATIVE_ARRIVAL", "a selected request predates the evaluation start")
    if len(requests) + before + after != scoped_total:
        raise DemandContractError("POPULATION_PARTITION_MISMATCH", f"{len(requests)}+{before}+{after} != {scoped_total}")

    return {
        "requests": requests,
        "audit": {
            "scope_reporting_window_ids": [str(w) for w in scope],
            "scoped_request_count": scoped_total,
            "evaluation_population_count": len(requests),
            "requests_before_evaluation_start": before,
            "requests_after_evaluation_end": after,
            "partition_closes": len(requests) + before + after == scoped_total,
            "carry_in_present": before > 0,
            "carry_in_case": "B_CARRY_IN_WITH_KNOWN_ARRIVAL_TS" if before > 0 else "A_NO_INITIAL_WAITING_QUEUE",
            "population_rule": "evaluation_start_ts <= request_ts <= evaluation_end_ts",
            "discarded_silently": False,
            "timestamps_rewritten": False,
            **contract.payload(),
        },
    }


def stop_universe(requests: Sequence[DemandRequest], *, extra_stop_ids: Sequence[str] = ()) -> Dict[str, int]:
    """Deterministic stop index assignment from authoritative origin stops."""
    ids = sorted({r.origin_stop_id for r in requests} | {str(s) for s in extra_stop_ids})
    return {stop_id: index for index, stop_id in enumerate(ids)}
