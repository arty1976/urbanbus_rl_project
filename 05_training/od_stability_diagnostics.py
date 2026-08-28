#!/usr/bin/env python3
"""OD stability diagnostics (H4M-AE-R9.3).

Additive to the R9.2 diagnostics module, which is left byte-identical so its
artifact manifest stays verifiable.  Adds top-1 margin and bounded perturbation
stability.  Nothing here selects or tunes a weight.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

STABILITY_ID = "OD_DISTRIBUTION_STABILITY_DIAGNOSTICS_V1"
PERTURBATION_FACTORS = (0.95, 1.00, 1.05)


def group_distributions(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["attribution_status"] != "ATTRIBUTED_ORIGIN":
            continue
        out[(r["source_key"], r["origin_occurrence_id"])][r["destination_occurrence_id"]] = float(
            r["destination_probability"])
    return out


def margin_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Top-1 probability, top-2 probability and their margin per origin-occurrence."""
    groups = group_distributions(rows)
    top1: List[float] = []
    top2: List[float] = []
    margins: List[float] = []
    for probs in groups.values():
        ordered = sorted(probs.values(), reverse=True)
        p1 = ordered[0]
        p2 = ordered[1] if len(ordered) > 1 else 0.0
        top1.append(p1)
        top2.append(p2)
        margins.append(p1 - p2)
    if not margins:
        return {"groups": 0}
    srt = sorted(margins)
    n = len(srt)
    return {
        "groups": n,
        "top1_probability_mean": round(statistics.fmean(top1), 8),
        "top2_probability_mean": round(statistics.fmean(top2), 8),
        "top1_margin_mean": round(statistics.fmean(margins), 8),
        "top1_margin_median": round(statistics.median(margins), 8),
        "top1_margin_min": round(srt[0], 8),
        "top1_margin_max": round(srt[-1], 8),
        "top1_margin_p10": round(srt[int(0.10 * (n - 1))], 8),
        "top1_margin_p90": round(srt[int(0.90 * (n - 1))], 8),
        "groups_with_margin_below_1e-3": sum(1 for m in margins if m < 1e-3),
        "groups_with_margin_below_1e-2": sum(1 for m in margins if m < 1e-2),
    }


def perturbation_stability(materialize_fn, factors: Sequence[float] = PERTURBATION_FACTORS,
                           top_k: int = 5) -> Dict[str, Any]:
    """Bounded symmetric perturbation of the existing uncalibrated weight.

    materialize_fn(factor) -> rows.  No weight is selected, ranked or tuned; the
    only question asked is whether the top destination survives a small nudge.
    """
    runs = {f: group_distributions(materialize_fn(f)) for f in factors}
    base = runs[1.00]
    keys = sorted(set(base))
    for f in factors:
        keys = [k for k in keys if k in runs[f]]
    if not keys:
        return {"status": "PERTURBATION_TEST_NOT_APPLICABLE", "reason": "no comparable groups"}
    stable = 0
    overlaps: List[float] = []
    margin_deltas: List[float] = []
    for k in keys:
        tops = []
        for f in factors:
            p = runs[f][k]
            tops.append(max(p, key=lambda d: (p[d], d)))
        if len(set(tops)) == 1:
            stable += 1
        sets = []
        for f in factors:
            p = runs[f][k]
            sets.append({d for d, _ in sorted(p.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]})
        inter = set.intersection(*sets)
        union = set.union(*sets)
        overlaps.append(len(inter) / max(1, len(union)))
        ms = []
        for f in factors:
            o = sorted(runs[f][k].values(), reverse=True)
            ms.append(o[0] - (o[1] if len(o) > 1 else 0.0))
        margin_deltas.append(max(ms) - min(ms))
    n = len(keys)
    return {
        "status": "EVALUATED", "factors": list(factors), "groups": n,
        "top1_perturbation_stability_rate": round(stable / n, 6),
        "top1_flipped_groups": n - stable,
        f"top{top_k}_perturbation_overlap_mean": round(statistics.fmean(overlaps), 6),
        "margin_range_mean": round(statistics.fmean(margin_deltas), 8),
        "margin_range_max": round(max(margin_deltas), 8),
        "weight_selected_or_tuned": False,
    }


def evidence_interpretation(shifts: Dict[str, Any], margins: Dict[str, Any],
                            perturbation: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the three questions apart; only two of them are measurable here."""
    return {
        "INFORMATION_HAS_EFFECT": {
            "measured": True,
            "evidence": "adding cost and alighting evidence changes ranking and distribution",
            "top1_change_rates": {k: v.get("top1_change_rate") for k, v in shifts.items()},
        },
        "DISCRIMINATION_STRENGTH": {
            "measured": True,
            "evidence": "top-1 margin and perturbation stability describe how separated the leader is",
            "top1_margin_mean": margins.get("top1_margin_mean"),
            "top1_perturbation_stability_rate": perturbation.get("top1_perturbation_stability_rate"),
        },
        "GROUND_TRUTH_ACCURACY": {
            "measured": False,
            "reason": "no independent true destination label exists anywhere in current evidence",
            "measurable_in_r9_3": False,
        },
        "prohibited_inferences": [
            "high top-1 change rate means high accuracy",
            "low entropy means correct OD",
            "a variant with sharper margins is the better model",
        ],
    }


def stability_contract(margins: Dict[str, Any], perturbation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "contract_id": STABILITY_ID,
        "od_inference_output_is": "a probability mass distribution over legal downstream destinations",
        "od_inference_output_is_not": "an observed passenger destination",
        "top1_destination_is_not": "an allowed passenger-level truth assignment",
        "deterministic_argmax_destination_assignment_allowed": False,
        "distribution_carry_forward_allowed": True,
        "future_individual_request_interface": [
            "frozen OD probability distribution",
            "explicitly seeded stochastic sampler",
            "individual destination realization",
        ],
        "sampling_separated_from_inference": True,
        "implemented_in_r9_3": False,
        "promotion_requires": "an explicit future gate with independent destination evidence",
        "supporting_evidence": {"top1_margin": margins, "perturbation": perturbation},
    }
