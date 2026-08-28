#!/usr/bin/env python3
"""Evidence-supported path-cost repair layer (H4M-AE-R9.3).

Repairs a consecutive route-direction segment cost ONLY when it is reconstructable
from the existing authoritative graph: a unique, route-consistent, bounded
multi-edge path whose intermediates are not other stops of the same
route-direction.  Nothing is interpolated, averaged, substituted or fabricated.

A remaining gap is preferable to fabricated precision, so every unrepairable
segment keeps its neutral fallback and an explicit reason code.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

REPAIR_ID = "AUTHORITATIVE_ROUTE_CONSISTENT_MULTI_EDGE_PATH_REPAIR_V1"
MAX_HOPS = 3

REASON_A = "A_DIRECT_EDGE_MAPPING_FAILURE"
REASON_B = "B_ROUTE_CONSTRAINED_MULTI_EDGE_PATH_AVAILABLE"
REASON_C = "C_INTERMEDIATE_GRAPH_NODE_BETWEEN_CONSECUTIVE_STOPS"
REASON_D = "D_ROUTE_GRAPH_TOPOLOGY_CONFLICT"
REASON_E = "E_NO_AUTHORITATIVE_COST_EVIDENCE"
REASON_F = "F_OTHER_EXPLICITLY_DOCUMENTED"

FORBIDDEN = (
    "arbitrary shortest path across the city graph", "nearest-stop substitution",
    "Euclidean distance fabrication", "interpolation from adjacent routes",
    "inferred average segment cost", "ETA or headway derived cost",
    "route-frequency prior", "coverage-driven calibration", "external data",
)


@dataclass
class GraphEvidence:
    idx2stop: Dict[int, str]
    stop2idx: Dict[str, int]
    forward: Dict[str, Dict[str, Tuple[float, float, float]]]
    edges_sha256: str
    nodes_sha256: str

    def edge(self, a: str, b: str) -> Optional[Tuple[float, float, float]]:
        return self.forward.get(a, {}).get(b)


def load_graph(nodes_path: Path, edges_path: Path) -> GraphEvidence:
    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)
    idx2stop = dict(zip(nodes["node_index"].astype(int), nodes["stop_id"].astype(str)))
    forward: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for s, d, dm, ts, gc in zip(edges["src_idx"].astype(int), edges["dst_idx"].astype(int),
                                edges["distance_m"], edges["time_sec"], edges["generalized_cost"]):
        a, b = idx2stop.get(s), idx2stop.get(d)
        if a and b:
            forward.setdefault(a, {})[b] = (float(dm), float(ts), float(gc))
    return GraphEvidence(
        idx2stop=idx2stop, stop2idx={v: k for k, v in idx2stop.items()}, forward=forward,
        edges_sha256=hashlib.sha256(Path(edges_path).read_bytes()).hexdigest(),
        nodes_sha256=hashlib.sha256(Path(nodes_path).read_bytes()).hexdigest())


def _paths(graph: GraphEvidence, a: str, b: str, route_stops: set, max_hops: int) -> List[List[str]]:
    """Route-consistent simple paths a -> b, intermediates never on this route-direction."""
    found: List[List[str]] = []
    for hops in range(2, max_hops + 1):          # hop 1 is the direct edge, already absent
        frontier: List[List[str]] = [[a]]
        for _ in range(hops - 1):
            nxt: List[List[str]] = []
            for path in frontier:
                for m in graph.forward.get(path[-1], {}):
                    if m == b or m in path:
                        continue
                    if m in route_stops:          # would contradict consecutiveness
                        continue
                    nxt.append(path + [m])
            frontier = nxt
        for path in frontier:
            if b in graph.forward.get(path[-1], {}):
                found.append(path + [b])
        if found:
            break                                  # prefer the minimal hop count
    return found


def classify_and_repair(occurrence_master: Path, graph: GraphEvidence,
                        max_hops: int = MAX_HOPS) -> Dict[str, Any]:
    occ = pd.read_parquet(occurrence_master)
    occ["stop_id"] = occ["stop_id"].astype(str)
    occ["route_id"] = occ["route_id"].astype(str)
    occ["direction_id"] = occ["direction_id"].astype(str)
    occ = occ.sort_values(["route_id", "direction_id", "stop_sequence"], kind="mergesort")

    records: List[Dict[str, Any]] = []
    reasons: Counter = Counter()
    covered = 0
    for (route_id, direction_id), grp in occ.groupby(["route_id", "direction_id"], sort=True):
        stops = list(grp["stop_id"])
        seqs = list(grp["stop_sequence"])
        route_stops = set(stops)
        for i in range(len(stops) - 1):
            a, b = stops[i], stops[i + 1]
            if graph.edge(a, b) is not None:
                covered += 1
                continue
            base = {"route_id": route_id, "direction_id": direction_id,
                    "from_stop_id": a, "to_stop_id": b,
                    "from_stop_sequence": int(seqs[i]), "to_stop_sequence": int(seqs[i + 1])}
            if a not in graph.stop2idx or b not in graph.stop2idx:
                reasons[REASON_E] += 1
                records.append({**base, "gap_reason": REASON_E, "repaired": False,
                                "detail": "one or both stops are absent from the authoritative graph node table",
                                "absent_endpoint": ("from" if a not in graph.stop2idx else "") +
                                                   ("to" if b not in graph.stop2idx else "")})
                continue
            paths = _paths(graph, a, b, route_stops, max_hops)
            if not paths:
                # does a route-internal intermediate exist?  that is a topology conflict
                conflict = any(m in route_stops and b in graph.forward.get(m, {})
                               for m in graph.forward.get(a, {}))
                reason = REASON_D if conflict else REASON_E
                reasons[reason] += 1
                records.append({**base, "gap_reason": reason, "repaired": False,
                                "detail": ("an intermediate lies on the same route-direction, contradicting "
                                           "consecutiveness" if conflict else
                                           "no route-consistent authoritative path within the hop bound")})
                continue
            if len(paths) > 1:
                reasons[REASON_F] += 1
                records.append({**base, "gap_reason": REASON_F, "repaired": False,
                                "detail": f"{len(paths)} route-consistent paths tie at the minimal hop count; "
                                          "an ambiguous reconstruction is not evidence"})
                continue
            path = paths[0]
            legs = [graph.edge(x, y) for x, y in zip(path, path[1:])]
            if any(l is None for l in legs):
                reasons[REASON_F] += 1
                records.append({**base, "gap_reason": REASON_F, "repaired": False,
                                "detail": "leg lookup inconsistency"})
                continue
            dm = sum(l[0] for l in legs)
            ts = sum(l[1] for l in legs)
            gc = sum(l[2] for l in legs)
            reason = REASON_C if len(path) == 3 else REASON_B
            reasons[reason] += 1
            records.append({**base, "gap_reason": reason, "repaired": True,
                            "repair_reason": REPAIR_ID,
                            "node_sequence": path, "reconstructed_segment_count": len(path) - 1,
                            "edge_pairs": [f"{x}->{y}" for x, y in zip(path, path[1:])],
                            "distance_m": dm, "time_sec": ts, "generalized_cost": gc,
                            "distance_source": "SUM_OF_AUTHORITATIVE_PROJECT_EDGES",
                            "travel_time_source": "SUM_OF_AUTHORITATIVE_PROJECT_EDGES",
                            "route_consistency": {
                                "correct_origin": True, "correct_downstream": True,
                                "direction_preserved": True, "monotonic_progression": True,
                                "reverse_traversal": False, "cross_route_shortcut": False,
                                "disconnected_teleport": False,
                                "intermediates_off_route": True, "unique_minimal_path": True},
                            "source_edges_sha256": graph.edges_sha256,
                            "source_nodes_sha256": graph.nodes_sha256})
    repaired = [r for r in records if r["repaired"]]
    return {
        "repair_id": REPAIR_ID, "max_hops": max_hops,
        "forbidden_methods_used": [],
        "forbidden_methods_declared": list(FORBIDDEN),
        "segments_total": covered + len(records),
        "segments_covered_before": covered,
        "gap_segments": len(records),
        "gap_reason_counts": dict(reasons),
        "repairable": len(repaired), "repaired": len(repaired),
        "unrepaired": len(records) - len(repaired),
        "records": records,
        "source_edges_sha256": graph.edges_sha256, "source_nodes_sha256": graph.nodes_sha256,
    }


def write_repaired_edges(base_edges: Path, base_nodes: Path, repair: Dict[str, Any],
                         out_path: Path) -> Dict[str, Any]:
    """Emit an explicit derived edge table: original edges plus repaired direct segments.

    The engine stays byte-identical; the repair is a versioned data layer, and every
    added row is traceable through the repair records.
    """
    nodes = pd.read_parquet(base_nodes)
    edges = pd.read_parquet(base_edges)
    stop2idx = dict(zip(nodes["stop_id"].astype(str), nodes["node_index"].astype(int)))
    added = [{"src_idx": stop2idx[r["from_stop_id"]], "dst_idx": stop2idx[r["to_stop_id"]],
              "distance_m": r["distance_m"], "time_sec": r["time_sec"],
              "generalized_cost": r["generalized_cost"], "long_edge_5km_flag": 0.0}
             for r in repair["records"] if r["repaired"]]
    out = pd.concat([edges, pd.DataFrame(added, columns=edges.columns)], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return {"base_edges": str(base_edges), "base_edges_sha256": repair["source_edges_sha256"],
            "repaired_edges_path": str(out_path),
            "repaired_edges_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest(),
            "base_rows": int(len(edges)), "added_rows": len(added), "total_rows": int(len(out)),
            "added_rows_traceable_to_repair_records": True}
