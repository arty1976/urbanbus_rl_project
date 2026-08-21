#!/usr/bin/env python3
"""H4M-AE-R9.6 terminal-occurrence feasibility classification and filtering.

The question this layer answers, using frozen R9 semantics and nothing else:

    can a route-direction occurrence with zero legal downstream destination
    support be the latent route choice of a passenger who *historically boarded*
    at that stop?

It cannot.  The boarding is observed history: somebody got on a bus there.  A
route-direction occurrence with no downstream stop offers that passenger nowhere
to ride, so it cannot be the route-direction they boarded.  Removing it from the
candidate set is entailed by the observed boarding plus the frozen occurrence
topology.  It is a physical/OD feasibility constraint, not a statement that one
route is likelier than another.

Nothing here consults a route-frequency prior, a headway, an ETA, an external
source, or any performance outcome -- none of which this project holds authority
for (R9.1 froze ROUTE_PRIOR_MODE as
NEUTRAL_UNIFORM_ROUTE_FREQUENCY_PRIOR_UNAVAILABLE).  An occurrence with at least
one legal downstream destination is never removed, whatever its cost or shape.

When *every* candidate at a stop has zero downstream support the boarding cannot
be attributed at all.  It is never fabricated onto a route and never dropped: it
stays an explicit request identity carrying
ALL_CANDIDATES_ZERO_DOWNSTREAM_SUPPORT.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import constrained_od_engine as E

FILTER_ID = "TERMINAL_OCCURRENCE_ZERO_DOWNSTREAM_FEASIBILITY_FILTER_V1"
FILTER_VERSION = "R9_6_V1"

LEGAL = "LEGAL_DOWNSTREAM_SUPPORT"
ZERO_TERMINAL = "ZERO_DOWNSTREAM_TERMINAL_OCCURRENCE"
OTHER_ZERO = "OTHER_NO_LEGAL_DOWNSTREAM_SUPPORT"

REASON_ALL_ZERO = "ALL_CANDIDATES_ZERO_DOWNSTREAM_SUPPORT"
REASON_NO_ROUTE = "NO_ROUTE_DIRECTION_SERVING_STOP"

FILTER_CONTRACT = {
    "filter_id": FILTER_ID,
    "version": FILTER_VERSION,
    "basis": "observed boarding event plus frozen occurrence-level route topology",
    "removes": "only candidates with zero legal downstream destination support",
    "never_removes": "any occurrence holding at least one legal downstream destination",
    "route_frequency_prior_used": False,
    "terminal_occurrence_preference_assumed": False,
    "headway_used": False, "eta_used": False, "external_data_used": False,
    "performance_outcome_used": False,
    "is_a_preference_prior": False,
    "is_a_feasibility_constraint": True,
    "fabricates_destination_for_terminal_occurrence": False,
    "drops_request_mass": False,
}


def classify_candidate(index: E.RouteNetworkIndex, occurrence: E.Occurrence) -> str:
    """Classify one candidate occurrence by its legal downstream support."""
    if index.downstream_candidates(occurrence):
        return LEGAL
    sequence = index.sequence_by_route_dir[(occurrence.route_id, occurrence.direction_id)]
    # A zero-support occurrence that is not last in its route-direction would be a
    # topology anomaly rather than a terminal, and is reported under its own label.
    return ZERO_TERMINAL if occurrence.position == len(sequence) - 1 else OTHER_ZERO


def classify_stop(index: E.RouteNetworkIndex, stop_id: str) -> Dict[str, Any]:
    """Full feasibility picture for one boarding origin."""
    candidates = index.route_candidates(str(stop_id))
    if not candidates:
        return {"stop_id": str(stop_id), "candidate_count": 0, "legal": [], "removed": [],
                "classification": {}, "attributable": False, "reason": REASON_NO_ROUTE}
    by_class: Dict[str, List[str]] = {}
    legal, removed = [], []
    for occurrence in candidates:
        label = classify_candidate(index, occurrence)
        by_class.setdefault(label, []).append(occurrence.occurrence_id)
        (legal if label == LEGAL else removed).append(occurrence)
    return {
        "stop_id": str(stop_id), "candidate_count": len(candidates),
        "legal": legal, "removed": removed,
        "classification": {k: len(v) for k, v in sorted(by_class.items())},
        "removed_occurrence_ids": sorted(o.occurrence_id for o in removed),
        "attributable": bool(legal),
        "reason": None if legal else REASON_ALL_ZERO,
    }


def feasible_candidates(index: E.RouteNetworkIndex, stop_id: str) -> List[E.Occurrence]:
    """Candidate occurrences that an observed boarding could actually have used."""
    return classify_stop(index, stop_id)["legal"]


def audit(index: E.RouteNetworkIndex, stop_ids: Sequence[str],
          mass_by_stop: Dict[str, int]) -> Dict[str, Any]:
    """Occurrence-level and stop-level feasibility audit over a scoped stop set."""
    occ_counts: Dict[str, int] = {LEGAL: 0, ZERO_TERMINAL: 0, OTHER_ZERO: 0}
    stop_counts: Dict[str, int] = {}
    stop_mass: Dict[str, int] = {}
    all_zero, mixed = [], []
    for stop in sorted({str(s) for s in stop_ids}):
        info = classify_stop(index, stop)
        mass = int(mass_by_stop.get(stop, 0))
        for label, count in info["classification"].items():
            occ_counts[label] = occ_counts.get(label, 0) + count
        if info["candidate_count"] == 0:
            bucket = REASON_NO_ROUTE
        elif not info["attributable"]:
            bucket = REASON_ALL_ZERO
            all_zero.append({"stop_id": stop, "boarding_mass": mass,
                             "candidate_count": info["candidate_count"]})
        else:
            bucket = "HAS_LEGAL_CANDIDATE"
            if info["removed"]:
                mixed.append({"stop_id": stop, "boarding_mass": mass,
                              "legal": len(info["legal"]), "removed": len(info["removed"])})
        stop_counts[bucket] = stop_counts.get(bucket, 0) + 1
        stop_mass[bucket] = stop_mass.get(bucket, 0) + mass
    return {
        "filter_contract": FILTER_CONTRACT,
        "occurrence_classification": occ_counts,
        "occurrences_removed": occ_counts[ZERO_TERMINAL] + occ_counts[OTHER_ZERO],
        "occurrences_with_legal_support_removed": 0,
        "non_terminal_zero_support_occurrences": occ_counts[OTHER_ZERO],
        "stop_classification": stop_counts,
        "stop_boarding_mass": stop_mass,
        "stops_with_every_candidate_zero_downstream": all_zero,
        "stops_with_mixed_candidates": mixed,
        "total_boarding_mass": int(sum(int(v) for v in mass_by_stop.values())),
    }
