#!/usr/bin/env python3
"""Scoped multi-window OD materialization and uncertainty diagnostics (H4M-AE-R9.2).

Reuses the R9.1 engine unchanged.  Selection is structural and recorded; it never
depends on V0/V1/V2 output.  Nothing here calibrates a prior, ranks a variant as
better, or materialises citywide OD.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

DIAGNOSTICS_ID = "SCOPED_MULTI_WINDOW_OD_UNCERTAINTY_DIAGNOSTICS_V1"
# Project fixed clock definition carried from R3: night 07:00, offpeak 10:00, peak 17:00.
TIME_BANDS = {"night": 7, "offpeak": 10, "peak": 17}
DISTRIBUTION_METRIC = "TOTAL_VARIATION_DISTANCE"
TOP_K = 5


@dataclass(frozen=True)
class SamplingRule:
    """Structural, outcome-independent.  Recorded verbatim in the artifact."""

    dates: Sequence[str]
    time_bands: Dict[str, int]
    per_stratum: int
    strata: Sequence[str] = ("single_route_no_gap", "single_route_gap",
                             "multi_route_no_gap", "multi_route_gap")
    order_key: str = "stop_id ascending"
    depends_on_variant_output: bool = False

    def payload(self) -> Dict[str, Any]:
        return {
            "diagnostics_id": DIAGNOSTICS_ID,
            "dates": list(self.dates), "time_bands": dict(self.time_bands),
            "per_stratum": self.per_stratum, "strata": list(self.strata),
            "order_key": self.order_key,
            "selection_depends_on_variant_output": self.depends_on_variant_output,
            "cherry_picking": False,
            "date_rule": "earliest 2023 date of each day-type class carried by the project registry",
            "band_rule": "project fixed clock definition: night 07:00, offpeak 10:00, peak 17:00",
            "stratum_rule": "route-candidate multiplicity crossed with presence of at least one uncovered segment on any feasible path",
            "tie_break": "stop_id ascending",
        }


def origin_cost_gap_exposed(index: Any, stop_id: str) -> bool:
    """True when at least one feasible path from this origin crosses an uncovered segment."""
    for origin in index.route_candidates(stop_id):
        key = (origin.route_id, origin.direction_id)
        pm = index.prefix_missing[key]
        if pm[-1] - pm[origin.position] > 0:
            return True
    return False


def select_sample(index: Any, ledger_rows: pd.DataFrame, rule: SamplingRule) -> pd.DataFrame:
    """Deterministic stratified selection over authoritative boarding rows."""
    picked: List[Dict[str, Any]] = []
    for date in rule.dates:
        for band, hour in rule.time_bands.items():
            win = ledger_rows[(ledger_rows["service_date"] == date)
                              & (ledger_rows["service_hour"] == hour)
                              & (ledger_rows["boardings"] > 0)].copy()
            if win.empty:
                continue
            win = win.sort_values("stop_id", kind="mergesort")
            win["n_routes"] = win["stop_id"].map(lambda s: len(index.route_candidates(s)))
            win["gap"] = win["stop_id"].map(lambda s: origin_cost_gap_exposed(index, s))
            buckets = {
                "single_route_no_gap": win[(win["n_routes"] == 1) & (~win["gap"])],
                "single_route_gap": win[(win["n_routes"] == 1) & (win["gap"])],
                "multi_route_no_gap": win[(win["n_routes"] >= 2) & (~win["gap"])],
                "multi_route_gap": win[(win["n_routes"] >= 2) & (win["gap"])],
            }
            for name in rule.strata:
                for _, r in buckets[name].head(rule.per_stratum).iterrows():
                    picked.append({**r.to_dict(), "time_band": band, "stratum": name})
            unattr = win[win["n_routes"] == 0].head(1)
            for _, r in unattr.iterrows():
                picked.append({**r.to_dict(), "time_band": band, "stratum": "unattributable"})
    frame = pd.DataFrame(picked)
    if frame.empty:
        return frame
    frame["source_key"] = [f"{d}|{h}|{s}" for d, h, s in
                           zip(frame["service_date"], frame["service_hour"], frame["stop_id"])]
    return frame.sort_values(["service_date", "service_hour", "stop_id"], kind="mergesort").reset_index(drop=True)


def total_variation(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def _by_group(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Destination probability vectors keyed by (source_key, origin_occurrence_id)."""
    out: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["attribution_status"] != "ATTRIBUTED_ORIGIN":
            continue
        out[(r["source_key"], r["origin_occurrence_id"])][r["destination_occurrence_id"]] = float(r["destination_probability"])
    return out


def variant_shift(rows_a: Sequence[Dict[str, Any]], rows_b: Sequence[Dict[str, Any]],
                  top_k: int = TOP_K) -> Dict[str, Any]:
    ga, gb = _by_group(rows_a), _by_group(rows_b)
    common = sorted(set(ga) & set(gb))
    if not common:
        return {"groups": 0}
    tvds, max_shifts, mean_shifts = [], [], []
    top1_changed = 0
    topk_overlaps = []
    for key in common:
        p, q = ga[key], gb[key]
        tvds.append(total_variation(p, q))
        keys = set(p) | set(q)
        diffs = [abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys]
        max_shifts.append(max(diffs))
        mean_shifts.append(sum(diffs) / len(diffs))
        if max(p, key=lambda k: (p[k], k)) != max(q, key=lambda k: (q[k], k)):
            top1_changed += 1
        ta = {k for k, _ in sorted(p.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]}
        tb = {k for k, _ in sorted(q.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]}
        topk_overlaps.append(len(ta & tb) / max(1, len(ta | tb)))
    n = len(common)
    return {
        "metric": DISTRIBUTION_METRIC, "groups": n,
        "tvd_mean": round(sum(tvds) / n, 8), "tvd_max": round(max(tvds), 8),
        "tvd_p90": round(sorted(tvds)[int(0.9 * (n - 1))], 8),
        "top1_change_rate": round(top1_changed / n, 6), "top1_changed_groups": top1_changed,
        f"top{top_k}_jaccard_mean": round(sum(topk_overlaps) / n, 6),
        f"top{top_k}_ranking_changed_rate": round(sum(1 for o in topk_overlaps if o < 1.0) / n, 6),
        "max_destination_probability_shift": round(max(max_shifts), 8),
        "mean_absolute_probability_shift": round(sum(mean_shifts) / n, 8),
    }


def entropy_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [float(r["od_entropy"]) for r in rows
            if r["attribution_status"] == "ATTRIBUTED_ORIGIN" and r["od_entropy"] is not None]
    seen: Dict[Tuple[str, str], float] = {}
    for r in rows:
        if r["attribution_status"] == "ATTRIBUTED_ORIGIN" and r["od_entropy"] is not None:
            seen[(r["source_key"], r["origin_occurrence_id"])] = float(r["od_entropy"])
    g = sorted(seen.values())
    if not g:
        return {"groups": 0}
    return {"groups": len(g), "mean": round(sum(g) / len(g), 6),
            "min": round(g[0], 6), "max": round(g[-1], 6),
            "median": round(g[len(g) // 2], 6)}


def conservation_report(rows: Sequence[Dict[str, Any]], expected: Dict[str, float]) -> Dict[str, Any]:
    got: Dict[str, float] = defaultdict(float)
    for r in rows:
        got[r["source_key"]] += float(r["mass"])
    diffs = {k: abs(got.get(k, 0.0) - v) for k, v in expected.items()}
    attributed = [r for r in rows if r["attribution_status"] == "ATTRIBUTED_ORIGIN"]
    unattr = [r for r in rows if r["attribution_status"] == "UNATTRIBUTABLE_ORIGIN"]
    illegal = sum(1 for r in attributed
                  if r["destination_stop_sequence"] is None
                  or r["destination_stop_sequence"] <= r["origin_stop_sequence"])
    same = sum(1 for r in attributed if r["destination_occurrence_id"] == r["origin_occurrence_id"])
    return {
        "input_mass": round(sum(expected.values()), 6), "output_mass": round(sum(got.values()), 6),
        "max_abs_per_origin_diff": max(diffs.values()) if diffs else 0.0,
        "attributed_rows": len(attributed), "unattributable_rows": len(unattr),
        "unattributable_mass": round(sum(float(r["mass"]) for r in unattr), 6),
        "illegal_destination_count": illegal, "same_stop_destination": same,
        "upstream_destination": illegal,
    }


def cost_gap_report(index: Any, rows: Sequence[Dict[str, Any]],
                    sample: pd.DataFrame) -> Dict[str, Any]:
    """Segment-count coverage and demand-weighted coverage are kept strictly separate."""
    covered = missing = 0
    by_route: Counter = Counter()
    by_route_dir: Counter = Counter()
    for (route_id, direction_id), seq in index.sequence_by_route_dir.items():
        pm = index.prefix_missing[(route_id, direction_id)]
        gaps = pm[-1]
        segs = max(0, len(seq) - 1)
        covered += segs - gaps
        missing += gaps
        if gaps:
            by_route[route_id] += gaps
            by_route_dir[f"{route_id}|{direction_id}"] += gaps

    attributed = [r for r in rows if r["attribution_status"] == "ATTRIBUTED_ORIGIN"]
    total_mass = sum(float(r["mass"]) for r in attributed)
    fallback_mass = sum(float(r["mass"]) for r in attributed
                        if r["cost_evidence"] == "COST_UNAVAILABLE_NEUTRAL_FALLBACK")
    exposed_origins = {r["source_key"] for r in attributed
                       if r["cost_evidence"] == "COST_UNAVAILABLE_NEUTRAL_FALLBACK"}
    origin_mass = {r["source_key"]: float(r["origin_boarding_count"]) for r in rows}
    sample_mass = sum(set((k, v) for k, v in origin_mass.items()) and origin_mass.values())
    exposed_mass = sum(v for k, v in origin_mass.items() if k in exposed_origins)
    by_stop: Counter = Counter()
    by_band: Counter = Counter()
    band_of = dict(zip(sample["source_key"], sample["time_band"])) if not sample.empty else {}
    for r in attributed:
        if r["cost_evidence"] == "COST_UNAVAILABLE_NEUTRAL_FALLBACK":
            by_stop[r["origin_stop_id"]] += float(r["mass"])
            by_band[band_of.get(r["source_key"], "unknown")] += float(r["mass"])
    n_routes_with_gap = len(by_route)
    top_route_share = (sum(c for _, c in by_route.most_common(10)) / missing) if missing else None
    return {
        "segment_count_cost_coverage": round(covered / (covered + missing), 6) if covered + missing else None,
        "segments_covered": covered, "segments_missing": missing,
        "demand_weighted_cost_coverage": round(1.0 - fallback_mass / total_mass, 6) if total_mass else None,
        "candidate_mass_on_neutral_fallback": round(fallback_mass, 6),
        "candidate_mass_total": round(total_mass, 6),
        "fallback_mass_share": round(fallback_mass / total_mass, 6) if total_mass else None,
        "sampled_origin_mass": round(sample_mass, 6),
        "sampled_origin_mass_exposed_to_any_gap": round(exposed_mass, 6),
        "sampled_origin_exposure_share": round(exposed_mass / sample_mass, 6) if sample_mass else None,
        "routes_with_any_gap": n_routes_with_gap,
        "route_directions_with_any_gap": len(by_route_dir),
        "top_uncovered_routes": by_route.most_common(10),
        "top_uncovered_route_directions": by_route_dir.most_common(10),
        "top_fallback_origin_stops_by_mass": [(k, round(v, 4)) for k, v in by_stop.most_common(10)],
        "fallback_mass_by_time_band": {k: round(v, 4) for k, v in by_band.items()},
        "top10_route_share_of_all_gaps": round(top_route_share, 6) if top_route_share is not None else None,
        "concentration_verdict": ("CONCENTRATED" if (top_route_share or 0) >= 0.5 else "DISPERSED"),
        "missing_cost_values_inferred": False,
        "candidates_dropped_for_missing_cost": 0,
    }


def high_uncertainty_origins(rows_by_variant: Dict[str, Sequence[Dict[str, Any]]],
                             shifts: Dict[str, Any], top_n: int = 10) -> Dict[str, Any]:
    base = rows_by_variant["V1_PATH_COST_PRIOR"]
    per_group: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in base:
        if r["attribution_status"] != "ATTRIBUTED_ORIGIN":
            continue
        key = (r["source_key"], r["origin_occurrence_id"])
        per_group.setdefault(key, {
            "source_key": r["source_key"], "origin_stop_id": r["origin_stop_id"],
            "route_candidate_count": r["route_candidate_count"],
            "destination_candidate_count": r["destination_candidate_count"],
            "od_entropy": r["od_entropy"], "fallback_candidates": 0})
        if r["cost_evidence"] == "COST_UNAVAILABLE_NEUTRAL_FALLBACK":
            per_group[key]["fallback_candidates"] += 1
    groups = list(per_group.values())
    ga, gb = _by_group(rows_by_variant["V0_FEASIBILITY_NEUTRAL"]), _by_group(base)
    tvd_by_group = {k: total_variation(ga[k], gb[k]) for k in set(ga) & set(gb)}
    for key, g in per_group.items():
        g["v0_v1_tvd"] = round(tvd_by_group.get(key, 0.0), 8)
    return {
        "flag_semantics": "diagnostic only; a flagged origin is not a wrong OD",
        "highest_entropy": sorted(groups, key=lambda g: -g["od_entropy"])[:top_n],
        "highest_variant_sensitivity": sorted(groups, key=lambda g: -g["v0_v1_tvd"])[:top_n],
        "largest_candidate_multiplicity": sorted(
            groups, key=lambda g: -(g["route_candidate_count"] * g["destination_candidate_count"]))[:top_n],
        "largest_cost_gap_exposure": sorted(groups, key=lambda g: -g["fallback_candidates"])[:top_n],
    }
