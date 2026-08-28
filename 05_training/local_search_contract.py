#!/usr/bin/env python3
"""H4M-AE-R9.8-LS0 Local Search interface contract and candidate generator.

Local Search is a *candidate generator* and nothing else.  It proposes service
paths that would have been available; it never decides, never executes, and never
touches the simulator.  The ranking score it attaches is an ordering hint for
downstream layers, not authority to act:

    Local Search   proposes candidates
    Zero-Loss      removes candidates that would cost an existing rider time
    MAPPO          selects among what survives
    Simulator      is the only thing that may ever change state

This module imports no simulator, holds no RNG, and calls nothing that could
advance causal state.  Determinism comes from sha256 identity and a canonical
ordering, so the same inputs in any row order produce the same candidate set.

Route/direction provenance
--------------------------
A request's inferred route_id/direction_id travels with a candidate as
provenance only.  It is never used to constrain where a DRT vehicle may go:

    ROUTE_DIRECTION_SEMANTICS =
        INFERENCE_PROVENANCE_ONLY_NOT_DRT_ROUTE_CONSTRAINT

Paths are searched over the authoritative graph, so a candidate may leave the
historical fixed route entirely.

Inputs that do not exist
------------------------
Where the repository holds no authoritative source for an input, the generator
records NOT_AVAILABLE_FOR_LS_INPUT rather than inventing one.  Vehicle state and
onboard passengers are currently in that category for the frozen R9.7 scope, so
pickup/dropoff *insertion positions* are reported as unavailable while
origin-to-destination path candidates are still fully generated.
"""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

GENERATOR_ID = "LOCAL_SEARCH_CANDIDATE_GENERATOR_V1"
GENERATOR_VERSION = "LS0_V1"
ROUTE_DIRECTION_SEMANTICS = "INFERENCE_PROVENANCE_ONLY_NOT_DRT_ROUTE_CONSTRAINT"
NOT_AVAILABLE = "NOT_AVAILABLE_FOR_LS_INPUT"

ROLE_CONTRACT = {
    "generator_id": GENERATOR_ID,
    "role": "candidate generator",
    "may_rank_candidates": True,
    "ranking_is_final_action_authority": False,
    "may_execute_action": False,
    "may_mutate_simulator": False,
    "may_bypass_zero_loss": False,
    "may_bypass_mappo": False,
    "final_decision_authority": "MAPPO, after Zero-Loss filtering",
    "route_direction_semantics": ROUTE_DIRECTION_SEMANTICS,
    "uses_historical_route_as_drt_constraint": False,
    "deterministic_identity": "sha256",
    "python_builtin_hash_used": False,
    "unseeded_rng_used": False,
}


class LocalSearchContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class SearchConfig:
    """Every bound is configuration.  Nothing here is scope-specific.

    No district, agent count, route or horizon is baked in: the same contract
    serves a bounded Mac-mini validation and a Daegu citywide run by changing
    these values alone.
    """

    max_candidates: int = 8
    search_depth: int = 12
    search_radius_m: float = 20000.0
    insertion_limit: int = 4
    beam_width: int = 24
    timeout_expansions: int = 20000
    graph_scope: str = "AUTHORITATIVE_FULL_GRAPH"
    vehicle_scope: str = "CONFIGURED_BY_CALLER"

    def payload(self) -> Dict[str, Any]:
        return {
            "max_candidates": self.max_candidates, "search_depth": self.search_depth,
            "search_radius_m": self.search_radius_m, "insertion_limit": self.insertion_limit,
            "beam_width": self.beam_width, "timeout_expansions": self.timeout_expansions,
            "graph_scope": self.graph_scope, "vehicle_scope": self.vehicle_scope,
            "district_hardcoded": False, "agent_count_hardcoded": False,
            "route_hardcoded": False, "horizon_hardcoded": False,
            "citywide_scalable": True,
        }


@dataclass(frozen=True)
class GraphState:
    """Read-only authoritative graph: adjacency plus edge cost, nothing else."""

    adjacency: Mapping[int, Tuple[Tuple[int, float, float, float], ...]]
    stop_by_index: Mapping[int, str]
    index_by_stop: Mapping[str, int]
    source_sha256: str

    def edge_exists(self, src: int, dst: int) -> bool:
        return any(d == dst for d, _, _, _ in self.adjacency.get(src, ()))


@dataclass(frozen=True)
class VehicleState:
    """Vehicle identity and forward path, or an explicit statement of absence."""

    availability: str = NOT_AVAILABLE
    vehicle_id: Optional[str] = None
    current_stop_id: Optional[str] = None
    planned_path: Tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.availability != NOT_AVAILABLE


@dataclass(frozen=True)
class OnboardState:
    """Existing riders and their destinations, or an explicit statement of absence."""

    availability: str = NOT_AVAILABLE
    passengers: Tuple[Mapping[str, Any], ...] = ()

    @property
    def available(self) -> bool:
        return self.availability != NOT_AVAILABLE


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    historical_request_key: str
    request_realization_id: str
    vehicle_id: Optional[str]
    origin_stop_id: str
    destination_stop_id: str
    path_stop_ids: Tuple[str, ...]
    path_node_indices: Tuple[int, ...]
    pickup_insertion_position: Any
    dropoff_insertion_position: Any
    distance_m: float
    time_sec: float
    generalized_cost: float
    hop_count: int
    rank: int
    ranking_score: float
    feasibility: Mapping[str, Any]
    provenance: Mapping[str, Any]
    generator_version: str = GENERATOR_VERSION

    def payload(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "historical_request_key": self.historical_request_key,
            "request_realization_id": self.request_realization_id,
            "vehicle_id": self.vehicle_id,
            "origin_stop_id": self.origin_stop_id,
            "destination_stop_id": self.destination_stop_id,
            "path_stop_ids": list(self.path_stop_ids),
            "path_node_indices": list(self.path_node_indices),
            "pickup_insertion_position": self.pickup_insertion_position,
            "dropoff_insertion_position": self.dropoff_insertion_position,
            "distance_m": round(self.distance_m, 4), "time_sec": round(self.time_sec, 4),
            "generalized_cost": round(self.generalized_cost, 4),
            "hop_count": self.hop_count, "rank": self.rank,
            "ranking_score": round(self.ranking_score, 6),
            "ranking_is_final_action_authority": False,
            "feasibility": dict(self.feasibility),
            "provenance": dict(self.provenance),
            "generator_version": self.generator_version,
        }


@dataclass
class CandidateSet:
    historical_request_key: str
    candidates: List[Candidate] = field(default_factory=list)
    status: str = "GENERATED"
    reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def payload(self) -> Dict[str, Any]:
        return {
            "historical_request_key": self.historical_request_key,
            "status": self.status, "reason": self.reason,
            "candidate_count": len(self.candidates),
            "candidates": [c.payload() for c in self.candidates],
            "diagnostics": dict(self.diagnostics),
        }


def build_graph_state(edges, nodes, *, source_sha256: str) -> GraphState:
    """Adjacency from the authoritative edge table.  No edge is synthesised."""
    stop_by_index: Dict[int, str] = {}
    index_by_stop: Dict[str, int] = {}
    for idx, stop in zip(nodes["node_index"].astype(int), nodes["stop_id"].astype(str)):
        stop_by_index[int(idx)] = str(stop)
        index_by_stop.setdefault(str(stop), int(idx))
    adjacency: Dict[int, List[Tuple[int, float, float, float]]] = {}
    for src, dst, dist, tsec, cost in zip(
            edges["src_idx"].astype(int), edges["dst_idx"].astype(int),
            edges["distance_m"].astype(float), edges["time_sec"].astype(float),
            edges["generalized_cost"].astype(float)):
        adjacency.setdefault(int(src), []).append((int(dst), float(dist), float(tsec), float(cost)))
    # Canonical neighbour order makes expansion independent of table row order.
    frozen = {k: tuple(sorted(v, key=lambda t: (t[3], t[0]))) for k, v in adjacency.items()}
    return GraphState(adjacency=frozen, stop_by_index=stop_by_index,
                      index_by_stop=index_by_stop, source_sha256=source_sha256)


def candidate_identity(*, historical_request_key: str, path_node_indices: Sequence[int],
                       vehicle_id: Optional[str]) -> str:
    payload = "|".join([GENERATOR_ID, GENERATOR_VERSION, historical_request_key,
                        str(vehicle_id), ",".join(str(int(i)) for i in path_node_indices)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _search_paths(graph: GraphState, source: int, target: int,
                  config: SearchConfig) -> Tuple[List[Tuple[Tuple[int, ...], float, float, float]], Dict[str, Any]]:
    """Deterministic bounded beam search for distinct simple paths.

    Ties break on the node index, so the frontier order never depends on how the
    edge table happened to be ordered.  Expansion is capped by both depth and a
    total expansion budget, which is what keeps this bounded on any graph size.
    """
    frontier: List[Tuple[float, Tuple[int, ...], float, float]] = [(0.0, (source,), 0.0, 0.0)]
    found: List[Tuple[Tuple[int, ...], float, float, float]] = []
    # A frontier entry can be reached more than once, so the same complete path can
    # surface repeatedly; a candidate set must hold distinct paths.
    seen_paths: set = set()
    seen_frontier: set = set()
    expansions = 0
    truncated = False
    while frontier and len(found) < config.max_candidates:
        frontier.sort(key=lambda t: (t[0], t[1]))
        beam, frontier = frontier[:config.beam_width], frontier[config.beam_width:]
        nxt: List[Tuple[float, Tuple[int, ...], float, float]] = []
        for cost, path, dist, tsec in beam:
            if len(path) > config.search_depth:
                continue
            for dst, edge_dist, edge_time, edge_cost in graph.adjacency.get(path[-1], ()):
                if dst in path:
                    continue
                expansions += 1
                if expansions > config.timeout_expansions:
                    truncated = True
                    break
                new_dist = dist + edge_dist
                if new_dist > config.search_radius_m:
                    continue
                new = (cost + edge_cost, path + (dst,), new_dist, tsec + edge_time)
                if dst == target:
                    if new[1] not in seen_paths:
                        seen_paths.add(new[1])
                        found.append((new[1], new[2], new[3], new[0]))
                    if len(found) >= config.max_candidates:
                        break
                elif len(new[1]) <= config.search_depth and new[1] not in seen_frontier:
                    seen_frontier.add(new[1])
                    nxt.append(new)
            if truncated or len(found) >= config.max_candidates:
                break
        if truncated:
            break
        frontier.extend(nxt)
    found.sort(key=lambda t: (t[3], len(t[0]), t[0]))
    return found[:config.max_candidates], {"expansions": expansions, "budget_truncated": truncated}


def generate_candidates(*, vehicle_state: VehicleState, onboard_state: OnboardState,
                        pending_request: Mapping[str, Any], graph_state: GraphState,
                        search_config: SearchConfig) -> CandidateSet:
    """Propose service paths for one pending request.  Decides nothing."""
    key = str(pending_request["historical_request_key"])
    origin = str(pending_request["origin_stop_id"])
    destination = pending_request.get("destination_stop_id")
    result = CandidateSet(historical_request_key=key)
    result.diagnostics = {
        "vehicle_state_availability": vehicle_state.availability,
        "onboard_state_availability": onboard_state.availability,
        "insertion_positions_available": vehicle_state.available,
        "route_direction_semantics": ROUTE_DIRECTION_SEMANTICS,
    }

    if destination is None or str(destination) == "" or str(destination).lower() == "nan":
        result.status = "NO_CANDIDATE"
        result.reason = "REQUEST_HAS_NO_REALIZABLE_DESTINATION"
        return result
    destination = str(destination)
    src = graph_state.index_by_stop.get(origin)
    dst = graph_state.index_by_stop.get(destination)
    if src is None or dst is None:
        result.status = "NO_CANDIDATE"
        result.reason = "STOP_NOT_IN_AUTHORITATIVE_GRAPH"
        result.diagnostics["origin_in_graph"] = src is not None
        result.diagnostics["destination_in_graph"] = dst is not None
        return result
    if src == dst:
        result.status = "NO_CANDIDATE"
        result.reason = "ORIGIN_EQUALS_DESTINATION"
        return result

    paths, search_diag = _search_paths(graph_state, src, dst, search_config)
    result.diagnostics.update(search_diag)
    # Canonical dedupe: distinct node sequences only, in canonical cost order.
    deduped, seen = [], set()
    for entry in paths:
        if entry[0] in seen:
            continue
        seen.add(entry[0])
        deduped.append(entry)
    result.diagnostics["duplicate_paths_removed"] = len(paths) - len(deduped)
    paths = deduped
    if not paths:
        result.status = "NO_CANDIDATE"
        result.reason = "NO_FEASIBLE_PATH_WITHIN_SEARCH_BUDGET"
        return result

    # Insertion positions need a vehicle path.  None exists, so they are reported
    # as unavailable rather than guessed.
    insertion = NOT_AVAILABLE if not vehicle_state.available else 0
    for rank, (nodes, dist, tsec, cost) in enumerate(paths):
        stops = tuple(graph_state.stop_by_index[i] for i in nodes)
        result.candidates.append(Candidate(
            candidate_id=candidate_identity(historical_request_key=key, path_node_indices=nodes,
                                            vehicle_id=vehicle_state.vehicle_id),
            historical_request_key=key,
            request_realization_id=str(pending_request["request_realization_id"]),
            vehicle_id=vehicle_state.vehicle_id,
            origin_stop_id=origin, destination_stop_id=destination,
            path_stop_ids=stops, path_node_indices=tuple(int(i) for i in nodes),
            pickup_insertion_position=insertion, dropoff_insertion_position=insertion,
            distance_m=dist, time_sec=tsec, generalized_cost=cost,
            hop_count=len(nodes) - 1, rank=rank, ranking_score=cost,
            feasibility={
                "all_edges_authoritative": True, "simple_path": len(set(nodes)) == len(nodes),
                "within_search_radius_m": dist <= search_config.search_radius_m,
                "within_search_depth": len(nodes) - 1 <= search_config.search_depth,
                "origin_preserved": True, "destination_preserved": True,
                "insertion_positions": insertion,
            },
            provenance={
                "generator_id": GENERATOR_ID,
                "graph_source_sha256": graph_state.source_sha256,
                "request_ts": str(pending_request.get("request_ts")),
                "inferred_route_id": pending_request.get("route_id"),
                "inferred_direction_id": pending_request.get("direction_id"),
                "route_direction_semantics": ROUTE_DIRECTION_SEMANTICS,
                "search_config": search_config.payload(),
            },
        ))
    result.diagnostics["paths_found"] = len(paths)
    return result


def canonical_order(candidates: Iterable[Candidate]) -> List[Candidate]:
    """Ordering that never depends on how the inputs arrived."""
    return sorted(candidates, key=lambda c: (c.historical_request_key, round(c.generalized_cost, 6),
                                             c.hop_count, c.candidate_id))


def candidate_set_digest(sets: Sequence[CandidateSet]) -> str:
    import json
    payload = json.dumps([s.payload() for s in
                          sorted(sets, key=lambda s: s.historical_request_key)],
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
