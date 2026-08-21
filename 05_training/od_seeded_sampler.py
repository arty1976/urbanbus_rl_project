#!/usr/bin/env python3
"""Seeded OD distribution sampler and request realization interface (H4M-AE-R9.4).

Converts a frozen OD probability distribution into individual inferred
destination realizations without touching OD inference semantics.  There is no
argmax path anywhere in this module: selection is always an inverse-CDF draw over
the canonically ordered legal support.

A realization is an INFERRED_OD_SEEDED_REALIZATION.  It is never an observed
passenger destination and never a ground truth.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

SAMPLER_ID = "OD_SEEDED_CATEGORICAL_REALIZATION_SAMPLER_V1"
DESTINATION_SEMANTICS = "INFERRED_OD_SEEDED_REALIZATION"
UNATTRIBUTABLE_STATUS = "REALIZATION_NOT_ALLOWED_UNATTRIBUTABLE_ORIGIN"

# ---------------------------------------------------------------------------
# Finite-sample tolerance, declared here before any result is produced.
#
# For a categorical distribution estimated from n independent draws, each
# category proportion has standard error sqrt(p(1-p)/n) <= sqrt(0.25/n).  A
# four-sigma envelope on the worst category therefore gives
#     max_abs_error_bound = 4 * sqrt(0.25 / n) = 2 / sqrt(n)
# For total variation, E[TVD] <= 0.5 * sqrt((m-1)/n) for m categories; doubling
# that for finite-sample slack gives
#     tvd_bound = sqrt((m-1)/n)
# Both are functions of n and m only.  Neither may be adjusted after inspecting
# a result.
# ---------------------------------------------------------------------------
TOLERANCE_RULE = {
    "max_abs_error_bound": "2 / sqrt(n)",
    "tvd_bound": "sqrt((m - 1) / n)",
    "derivation": ("four-sigma envelope on a binomial proportion with se <= sqrt(0.25/n); "
                   "TVD bound from E[TVD] <= 0.5*sqrt((m-1)/n) doubled for finite-sample slack"),
    "declared_before_results": True,
    "post_hoc_tuning_allowed": False,
}


def max_abs_error_bound(n: int) -> float:
    return 2.0 / math.sqrt(n)


def tvd_bound(n: int, m: int) -> float:
    return math.sqrt(max(m - 1, 1) / n)


class SamplerContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class SeedContract:
    """Sampler randomness is derived, never drawn from a global RNG."""

    global_seed: int
    experiment_id: str = "H4M_AE_R9_4"

    def payload(self) -> Dict[str, Any]:
        return {
            "sampler_id": SAMPLER_ID, "experiment_id": self.experiment_id,
            "global_seed": self.global_seed,
            "derivation": "sha256(experiment_id | global_seed | variant | distribution_digest | origin_key | origin_occurrence_id | request_ordinal)",
            "hash_function": "hashlib.sha256",
            "python_builtin_hash_used": False,
            "global_rng_state_used": False,
            "unseeded_random_calls": 0,
            "depends_on_row_iteration_order": False,
            "separate_from_mappo_rng": True,
            "separate_from_environment_rng": True,
            "separate_from_training_rng": True,
            "same_experiment_replays_exactly": True,
            "different_seed_independently_reproducible": True,
        }

    def derive(self, *, variant: str, distribution_digest: str, origin_key: str,
               origin_occurrence_id: str, request_ordinal: int) -> int:
        payload = "|".join([self.experiment_id, str(self.global_seed), variant, distribution_digest,
                            origin_key, origin_occurrence_id, str(request_ordinal)])
        return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def canonical_support(distribution: Dict[str, float]) -> List[Tuple[str, float]]:
    """Canonical ordering makes the draw independent of input row order.

    Zero-probability candidates are dropped from the support so they can never be
    selected.  Negative or non-finite probabilities fail closed rather than being
    silently renormalised.
    """
    items: List[Tuple[str, float]] = []
    for key, value in distribution.items():
        p = float(value)
        if not math.isfinite(p):
            raise SamplerContractError("NON_FINITE_PROBABILITY", f"{key}={value!r}")
        if p < 0.0:
            raise SamplerContractError("NEGATIVE_PROBABILITY", f"{key}={value!r}")
        if p > 0.0:
            items.append((str(key), p))
    if not items:
        raise SamplerContractError("EMPTY_SUPPORT", "no candidate carries positive probability")
    total = math.fsum(p for _, p in items)
    if total <= 0.0:
        raise SamplerContractError("ZERO_TOTAL_MASS", "")
    return [(k, p / total) for k, p in sorted(items, key=lambda kv: kv[0])]


def distribution_digest(distribution: Dict[str, float]) -> str:
    support = canonical_support(distribution)
    return hashlib.sha256(json.dumps([[k, round(p, 12)] for k, p in support],
                                     sort_keys=True).encode("utf-8")).hexdigest()


def _inverse_cdf(support: Sequence[Tuple[str, float]], u: float) -> str:
    acc = 0.0
    for key, p in support:
        acc += p
        if u < acc:
            return key
    return support[-1][0]          # guards float accumulation only; never an argmax


def realize(distribution: Dict[str, float], *, seed_contract: SeedContract, variant: str,
            origin_key: str, origin_occurrence_id: str, realization_count: int,
            provenance: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Draw `realization_count` inferred destinations from one frozen distribution."""
    if realization_count < 0:
        raise SamplerContractError("NEGATIVE_REALIZATION_COUNT", str(realization_count))
    support = canonical_support(distribution)
    digest = distribution_digest(distribution)
    base = provenance or {}
    out: List[Dict[str, Any]] = []
    for ordinal in range(realization_count):
        seed = seed_contract.derive(variant=variant, distribution_digest=digest,
                                    origin_key=origin_key, origin_occurrence_id=origin_occurrence_id,
                                    request_ordinal=ordinal)
        u = (seed + 0.5) / float(2 ** 64)
        destination = _inverse_cdf(support, u)
        out.append({
            "request_realization_id": hashlib.sha256(
                f"{SAMPLER_ID}|{digest}|{origin_key}|{origin_occurrence_id}|{ordinal}|{seed_contract.global_seed}"
                .encode("utf-8")).hexdigest(),
            "origin_key": origin_key, "origin_occurrence_id": origin_occurrence_id,
            "destination_occurrence_id": destination,
            "request_ordinal": ordinal,
            "od_prior_variant": variant, "distribution_digest": digest,
            "sampler_id": SAMPLER_ID, "global_seed": seed_contract.global_seed,
            "derived_realization_seed": seed,
            "destination_semantics": DESTINATION_SEMANTICS,
            "destination_observed": False, "od_ground_truth": False,
            "support_size": len(support),
            **base,
        })
    return out


def realize_unattributable(*, origin_key: str, reason: str, mass: float,
                           seed_contract: SeedContract, variant: str) -> Dict[str, Any]:
    """An unattributable origin never receives a destination, and is never dropped."""
    return {
        "origin_key": origin_key, "origin_occurrence_id": None,
        "destination_occurrence_id": None,
        "realization_status": UNATTRIBUTABLE_STATUS,
        "unattributable_reason": reason, "preserved_mass": float(mass),
        "od_prior_variant": variant, "global_seed": seed_contract.global_seed,
        "destination_semantics": "NOT_APPLICABLE",
        "destination_observed": False, "od_ground_truth": False,
        "nearest_route_assigned": False, "nearest_stop_assigned": False,
        "uniform_city_sampled": False, "fabricated": False, "mass_dropped": False,
    }


def empirical_distribution(realizations: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for r in realizations:
        counts[r["destination_occurrence_id"]] = counts.get(r["destination_occurrence_id"], 0) + 1
    n = sum(counts.values())
    return {k: v / n for k, v in counts.items()} if n else {}


def fidelity_report(source: Dict[str, float], realizations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    support = dict(canonical_support(source))
    emp = empirical_distribution(realizations)
    n = len(realizations)
    m = len(support)
    keys = set(support) | set(emp)
    errors = {k: abs(support.get(k, 0.0) - emp.get(k, 0.0)) for k in keys}
    max_err = max(errors.values()) if errors else 0.0
    mean_err = (math.fsum(errors.values()) / len(errors)) if errors else 0.0
    tvd = 0.5 * math.fsum(errors.values())
    out_of_support = sorted(set(emp) - set(support))
    return {
        "n": n, "support_size": m,
        "max_abs_probability_error": round(max_err, 8),
        "mean_abs_probability_error": round(mean_err, 8),
        "tvd": round(tvd, 8),
        "max_abs_error_bound": round(max_abs_error_bound(n), 8),
        "tvd_bound": round(tvd_bound(n, m), 8),
        "max_abs_within_bound": max_err <= max_abs_error_bound(n),
        "tvd_within_bound": tvd <= tvd_bound(n, m),
        "support_coverage": round(len(set(emp) & set(support)) / m, 6) if m else None,
        "sampled_outside_support": out_of_support,
        "zero_probability_selected": bool(out_of_support),
        "tolerance_rule": TOLERANCE_RULE,
    }
