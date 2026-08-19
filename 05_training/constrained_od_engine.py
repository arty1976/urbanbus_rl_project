#!/usr/bin/env python3
"""Citywide route-attribution and constrained OD engine (H4M-AE-R9.1).

Implements the R9 contract as a deterministic, config-driven engine.  It is
citywide-capable by construction: the network index is built once from the
occurrence master, destination candidates are generated lazily per origin, and
boarding demand is carried as aggregated mass rather than expanded passenger
rows.  Nothing here materialises a full OD matrix or a request ledger.

Every emitted value is INFERRED.  Route attribution and destination are never
observed, and no route-frequency prior is fabricated: none exists.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

ENGINE_ID = "CITYWIDE_CONSTRAINED_OD_ENGINE_V1"
VARIANTS = ("V0_FEASIBILITY_NEUTRAL", "V1_PATH_COST_PRIOR", "V2_PATH_COST_PLUS_ALIGHT_AUX")
ROUTE_PRIOR_MODE = "NEUTRAL_UNIFORM_ROUTE_FREQUENCY_PRIOR_UNAVAILABLE"
UNATTRIBUTABLE = "UNATTRIBUTABLE_ORIGIN"
ATTRIBUTED = "ATTRIBUTED_ORIGIN"


class EngineContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@dataclass(frozen=True)
class EngineConfig:
    """Everything scope- or scale-dependent lives here, never in the logic."""

    occurrence_master: Path
    graph_nodes: Path
    graph_edges: Path
    variant: str = "V1_PATH_COST_PRIOR"
    cost_decay_per_km: float = 0.15      # research prior, uncalibrated
    alight_weight: float = 1.0           # research prior, uncalibrated
    alight_smoothing: float = 1.0        # research prior, uncalibrated
    demand_realization_seed: int = 1
    district_scope: Optional[Sequence[str]] = None
    route_scope: Optional[Sequence[str]] = None
    stop_scope: Optional[Sequence[str]] = None
    chunk_size: int = 100_000

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise EngineContractError("UNKNOWN_VARIANT", self.variant)

    def payload(self) -> Dict[str, Any]:
        return {
            "engine_id": ENGINE_ID, "variant": self.variant,
            "cost_decay_per_km": self.cost_decay_per_km,
            "alight_weight": self.alight_weight, "alight_smoothing": self.alight_smoothing,
            "prior_parameters_calibrated": False,
            "demand_realization_seed": self.demand_realization_seed,
            "district_scope": list(self.district_scope) if self.district_scope else None,
            "route_scope": list(self.route_scope) if self.route_scope else None,
            "stop_scope": list(self.stop_scope) if self.stop_scope else None,
            "chunk_size": self.chunk_size,
            "route_prior_mode": ROUTE_PRIOR_MODE,
        }


@dataclass
class Occurrence:
    occurrence_id: str
    route_id: str
    direction_id: str
    stop_id: str
    stop_sequence: int
    position: int          # index within the ordered route-direction
    same_stop_ordinal: int
    is_repeated: bool
    endpoint_role: Optional[str]


@dataclass
class RouteNetworkIndex:
    """Built once; reused for every origin.  No Cartesian product is ever formed."""

    occurrences_by_stop: Dict[str, List[Occurrence]] = field(default_factory=dict)
    sequence_by_route_dir: Dict[Tuple[str, str], List[Occurrence]] = field(default_factory=dict)
    prefix_cost: Dict[Tuple[str, str], List[float]] = field(default_factory=dict)
    prefix_missing: Dict[Tuple[str, str], List[int]] = field(default_factory=dict)
    source_sha256: Dict[str, str] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)

    def route_candidates(self, stop_id: str) -> List[Occurrence]:
        return self.occurrences_by_stop.get(str(stop_id), [])

    def downstream_candidates(self, origin: Occurrence) -> List[Occurrence]:
        """Strictly downstream on the same route-direction.  Never upstream, never
        the same occurrence, never another route or direction."""
        seq = self.sequence_by_route_dir[(origin.route_id, origin.direction_id)]
        return seq[origin.position + 1:]

    def segment_cost(self, origin: Occurrence, destination: Occurrence) -> Tuple[Optional[float], bool]:
        key = (origin.route_id, origin.direction_id)
        pc, pm = self.prefix_cost[key], self.prefix_missing[key]
        if pm[destination.position] - pm[origin.position] > 0:
            return None, False
        return pc[destination.position] - pc[origin.position], True


def build_index(config: EngineConfig) -> RouteNetworkIndex:
    occ = pd.read_parquet(config.occurrence_master)
    occ["stop_id"] = occ["stop_id"].astype(str)
    occ["route_id"] = occ["route_id"].astype(str)
    occ["direction_id"] = occ["direction_id"].astype(str)
    if config.route_scope:
        occ = occ[occ["route_id"].isin({str(r) for r in config.route_scope})]
    occ = occ.sort_values(["route_id", "direction_id", "stop_sequence"], kind="mergesort")

    nodes = pd.read_parquet(config.graph_nodes)
    edges = pd.read_parquet(config.graph_edges)
    idx2stop = dict(zip(nodes["node_index"].astype(int), nodes["stop_id"].astype(str)))
    seg: Dict[Tuple[str, str], float] = {}
    for s, d, gc in zip(edges["src_idx"].astype(int), edges["dst_idx"].astype(int), edges["generalized_cost"]):
        a, b = idx2stop.get(s), idx2stop.get(d)
        if a and b:
            seg[(a, b)] = float(gc)

    index = RouteNetworkIndex(source_sha256={
        "occurrence_master": sha256_file(config.occurrence_master),
        "graph_nodes": sha256_file(config.graph_nodes),
        "graph_edges": sha256_file(config.graph_edges),
    })
    covered = missing = 0
    for key, group in occ.groupby(["route_id", "direction_id"], sort=True):
        records: List[Occurrence] = []
        for position, (_, r) in enumerate(group.iterrows()):
            o = Occurrence(
                occurrence_id=str(r["route_stop_occurrence_id"]), route_id=str(r["route_id"]),
                direction_id=str(r["direction_id"]), stop_id=str(r["stop_id"]),
                stop_sequence=int(r["stop_sequence"]), position=position,
                same_stop_ordinal=int(r.get("same_stop_occurrence_ordinal") or 1),
                is_repeated=bool(r.get("is_repeated_stop_occurrence")),
                endpoint_role=(None if pd.isna(r.get("endpoint_role")) else str(r.get("endpoint_role"))),
            )
            records.append(o)
            index.occurrences_by_stop.setdefault(o.stop_id, []).append(o)
        index.sequence_by_route_dir[key] = records
        pc, pm = [0.0], [0]
        for a, b in zip(records, records[1:]):
            cost = seg.get((a.stop_id, b.stop_id))
            if cost is None:
                missing += 1
                pm.append(pm[-1] + 1)
                pc.append(pc[-1])
            else:
                covered += 1
                pm.append(pm[-1])
                pc.append(pc[-1] + cost)
        index.prefix_cost[key] = pc
        index.prefix_missing[key] = pm
    index.stats = {
        "occurrences": int(len(occ)), "route_directions": len(index.sequence_by_route_dir),
        "indexed_stops": len(index.occurrences_by_stop),
        "segment_cost_covered": covered, "segment_cost_missing": missing,
        "segment_cost_coverage": round(covered / (covered + missing), 6) if covered + missing else None,
    }
    return index


def _alight_shape(index: RouteNetworkIndex, alight_by_stop: Dict[str, float],
                  candidates: Sequence[Occurrence], smoothing: float) -> List[float]:
    """Smoothed RELATIVE auxiliary propensity.  Never a destination marginal."""
    raw = [float(alight_by_stop.get(c.stop_id, 0.0)) for c in candidates]
    total = sum(raw) + smoothing * len(raw)
    if total <= 0:
        return [1.0] * len(raw)
    return [(v + smoothing) / total for v in raw]


def attribute_origin(index: RouteNetworkIndex, config: EngineConfig, *,
                     source_key: str, service_date: str, service_hour: int,
                     origin_stop_id: str, boarding_count: float,
                     alight_by_stop: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Distribute one boarding row as aggregated mass over feasible OD candidates.

    Returns either a single UNATTRIBUTABLE_ORIGIN row or one row per legal
    (origin occurrence, destination occurrence) pair.  Origin mass is conserved
    exactly in both cases.
    """
    stop = str(origin_stop_id)
    routes = index.route_candidates(stop)
    base = {
        "source_key": source_key, "service_date": service_date, "service_hour": int(service_hour),
        "origin_stop_id": stop, "origin_boarding_count": float(boarding_count),
        "od_prior_variant": config.variant, "od_realization_seed": config.demand_realization_seed,
        "route_prior_mode": ROUTE_PRIOR_MODE,
        "route_attribution_observed": False, "destination_observed": False,
        "inference_provenance": "INFERRED",
    }
    if not routes:
        return [{**base, "attribution_status": UNATTRIBUTABLE,
                 "unattributable_reason": "NO_ROUTE_DIRECTION_SERVING_STOP", "route_candidate_count": 0,
                 "destination_candidate_count": 0, "mass": float(boarding_count),
                 "route_id": None, "direction_id": None, "origin_occurrence_id": None,
                 "origin_stop_sequence": None, "destination_occurrence_id": None,
                 "destination_stop_id": None, "destination_stop_sequence": None,
                 "route_assignment_probability": None, "destination_probability": None,
                 "path_generalized_cost": None, "cost_evidence": "NOT_APPLICABLE",
                 "od_entropy": None}]

    # Route choice: uniform, because no frequency prior authority exists.
    per_route = float(boarding_count) / len(routes)
    rows: List[Dict[str, Any]] = []
    for origin in routes:
        dests = index.downstream_candidates(origin)
        if not dests:
            rows.append({**base, "attribution_status": UNATTRIBUTABLE,
                         "unattributable_reason": "TERMINAL_OCCURRENCE_NO_DOWNSTREAM",
                         "route_candidate_count": len(routes), "destination_candidate_count": 0,
                         "mass": per_route, "route_id": origin.route_id,
                         "direction_id": origin.direction_id,
                         "origin_occurrence_id": origin.occurrence_id,
                         "origin_stop_sequence": origin.stop_sequence,
                         "destination_occurrence_id": None, "destination_stop_id": None,
                         "destination_stop_sequence": None,
                         "route_assignment_probability": 1.0 / len(routes),
                         "destination_probability": None, "path_generalized_cost": None,
                         "cost_evidence": "NOT_APPLICABLE", "od_entropy": None})
            continue

        costs: List[Optional[float]] = []
        weights: List[float] = []
        for d in dests:
            cost, ok = index.segment_cost(origin, d)
            costs.append(cost)
            if config.variant == "V0_FEASIBILITY_NEUTRAL" or not ok:
                weights.append(1.0)
            else:
                weights.append(math.exp(-config.cost_decay_per_km * (cost / 1000.0)))
        if config.variant == "V2_PATH_COST_PLUS_ALIGHT_AUX":
            shape = _alight_shape(index, alight_by_stop or {}, dests, config.alight_smoothing)
            weights = [w * (1.0 + config.alight_weight * s) for w, s in zip(weights, shape)]
        total = sum(weights)
        if total <= 0:
            weights = [1.0] * len(dests)
            total = float(len(dests))
        probs = [w / total for w in weights]
        entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)

        for d, p, cost in zip(dests, probs, costs):
            if d.stop_sequence <= origin.stop_sequence or d.position <= origin.position:
                raise EngineContractError("ILLEGAL_DESTINATION",
                                          f"{origin.occurrence_id} -> {d.occurrence_id}")
            rows.append({**base, "attribution_status": ATTRIBUTED,
                         "route_candidate_count": len(routes),
                         "destination_candidate_count": len(dests),
                         "route_id": origin.route_id, "direction_id": origin.direction_id,
                         "origin_occurrence_id": origin.occurrence_id,
                         "origin_stop_sequence": origin.stop_sequence,
                         "destination_occurrence_id": d.occurrence_id,
                         "destination_stop_id": d.stop_id,
                         "destination_stop_sequence": d.stop_sequence,
                         "route_assignment_probability": 1.0 / len(routes),
                         "destination_probability": p,
                         "mass": per_route * p,
                         "path_generalized_cost": cost,
                         "cost_evidence": "OBSERVED_GRAPH_EDGE" if cost is not None else "COST_UNAVAILABLE_NEUTRAL_FALLBACK",
                         "od_entropy": entropy})
    return rows


def materialize(index: RouteNetworkIndex, config: EngineConfig,
                boarding_rows: Iterable[Dict[str, Any]],
                alight_by_stop: Optional[Dict[str, float]] = None) -> Iterator[Dict[str, Any]]:
    """Stream aggregated OD mass rows.  Nothing is accumulated citywide in RAM."""
    for row in boarding_rows:
        yield from attribute_origin(
            index, config,
            source_key=str(row["source_key"]), service_date=str(row["service_date"]),
            service_hour=int(row["service_hour"]), origin_stop_id=str(row["stop_id"]),
            boarding_count=float(row["boardings"]), alight_by_stop=alight_by_stop)


def engine_manifest(index: RouteNetworkIndex, config: EngineConfig) -> Dict[str, Any]:
    return {
        "engine_id": ENGINE_ID, "config": config.payload(),
        "source_sha256": index.source_sha256, "index_stats": index.stats,
        "citywide_capable": True, "suseong_hardcoded": False,
        "full_matrix_materialized": False, "passenger_rows_expanded": False,
        "claim_guards": {
            "route_attribution_observed": False, "destination_observed": False,
            "od_ground_truth": False, "actual_request_ledger_created": False,
            "simulator_binding_allowed": False, "training_allowed": False,
            "performance_comparison_allowed": False, "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
        },
    }
