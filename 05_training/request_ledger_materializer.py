#!/usr/bin/env python3
"""H4M-AE-R9.5 one-to-one historical boarding request ledger materializer.

One historical boarding becomes exactly one request row.  There is no weighted
representative passenger, no compression of several boardings into one request,
no demand scaling and no synthetic multiplier: a bucket holding N boardings
produces exactly N rows, every one of them carrying passenger_count = 1.

Scale is handled by partitioning and streaming, never by compressing passengers.
The caller streams source rows window by window and each window is written as its
own part file, so no scope is ever accumulated whole in RAM.

The core is scope-agnostic.  Districts, dates, hours and stop sets arrive through
configuration; nothing here knows what Suseong is, and the same code path serves a
citywide, day-long, larger-fleet run unchanged.

Two realization stages sit on top of frozen R9 evidence, and neither invents new
evidence:

  stage 1  route-direction occurrence, drawn from the engine's own uniform route
           attribution prior (R9.1 ROUTE_PRIOR_MODE, no frequency prior exists)
  stage 2  destination, drawn by the frozen R9.4 sampler from the frozen R9.1/R9.3
           conditional OD distribution for the occurrence drawn in stage 1

Timestamps come from the R9.5 timestamp contract.  Origin and boarding count are
observed; route-direction, destination and timestamp are all inferred.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import constrained_od_engine as E
import od_seeded_sampler as S
import request_timestamp_realization as TS

LEDGER_ID = "SCOPED_ONE_TO_ONE_INFERRED_REQUEST_LEDGER_V1"
REQUEST_REALIZATION_MODE = "ONE_HISTORICAL_BOARDING_ONE_REQUEST"
REQUEST_WEIGHT = 1
ORIGIN_SEMANTICS = "HISTORICAL_BOARDING_ORIGIN"

STATUS_REALIZED = "REALIZED"
STATUS_UNATTRIBUTABLE = "UNATTRIBUTABLE_ORIGIN"
STATUS_TERMINAL = "UNREALIZABLE_TERMINAL_OCCURRENCE"

DEMAND_SEMANTICS = {
    "request_realization_mode": REQUEST_REALIZATION_MODE,
    "request_weight_passengers": REQUEST_WEIGHT,
    "boarding_count_semantics": TS.BOARDING_COUNT_SEMANTICS,
    "origin_semantics": ORIGIN_SEMANTICS,
    "request_ts_semantics": TS.TIMESTAMP_SEMANTICS,
    "destination_semantics": S.DESTINATION_SEMANTICS,
    "weighted_representative_passenger": False,
    "passenger_compression": False,
    "demand_scaling": False,
    "synthetic_passenger_multiplier": False,
    "research_intensity_rescaling": False,
    "scale_handled_by": "partitioning, streaming and chunked writes",
}

HANDOFF_SCHEMA: Tuple[Tuple[str, str], ...] = (
    ("request_id", "unique request identifier, stable under replay"),
    ("request_ts", "inferred realization time inside the authoritative bucket"),
    ("origin_stop_id", "observed historical boarding stop"),
    ("destination_stop_id", "inferred seeded OD realization; null when unrealizable"),
    ("passenger_count", "always 1; one historical boarding is one request"),
    ("route_id", "route attribution provenance from R9.1"),
    ("direction_id", "direction attribution provenance from R9.1"),
    ("origin_occurrence_id", "route-stop occurrence carrying the origin"),
    ("destination_occurrence_id", "route-stop occurrence carrying the destination"),
    ("origin_stop_sequence", "occurrence-level origin ordering"),
    ("destination_stop_sequence", "occurrence-level destination ordering"),
    ("realization_status", "REALIZED | UNATTRIBUTABLE_ORIGIN | UNREALIZABLE_TERMINAL_OCCURRENCE"),
    ("demand_realization_seed", "global demand seed governing the whole ledger"),
    ("provenance_digest", "digest over every authoritative source SHA behind the row"),
)


class MaterializerContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class LedgerConfig:
    """Everything scope-shaped lives here, so the core stays scope-agnostic."""

    variant: str
    global_seed: int
    scope: Dict[str, Any]                       # recorded, never interpreted by the core
    provenance: Dict[str, str]                  # source name -> sha256
    bucket_unit: str = "service_hour"
    experiment_id: str = "H4M_AE_R9_5"
    chunk_rows: int = 50_000

    def payload(self) -> Dict[str, Any]:
        return {
            "ledger_id": LEDGER_ID, "experiment_id": self.experiment_id,
            "variant": self.variant, "global_seed": self.global_seed,
            "bucket_unit": self.bucket_unit, "chunk_rows": self.chunk_rows,
            "scope": self.scope, "provenance": self.provenance,
            "scope_hardcoded_in_core": False,
            "citywide_capable": True, "day_long_capable": True, "larger_fleet_capable": True,
            "variant_choice_is_a_superiority_claim": False,
            **DEMAND_SEMANTICS,
        }

    @property
    def provenance_digest(self) -> str:
        joined = "|".join(f"{k}={self.provenance[k]}" for k in sorted(self.provenance))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass
class OriginPlan:
    """Frozen per-stop realization support, reused across every bucket at that stop."""

    route_support: Dict[str, float]                      # occurrence_id -> uniform probability
    route_digest: str
    occurrences: Dict[str, E.Occurrence]
    od_support: Dict[str, Dict[str, float]]              # occurrence_id -> {dest_occ_id: p}
    od_digest: Dict[str, str]
    destinations: Dict[str, E.Occurrence]
    terminal: frozenset
    unattributable_reason: Optional[str] = None


def build_origin_plan(index: E.RouteNetworkIndex, config: E.EngineConfig, stop_id: str,
                      alight_by_stop: Optional[Dict[str, float]] = None) -> OriginPlan:
    """Derive one stop's realization support from the frozen engine, at unit mass."""
    rows = E.attribute_origin(index, config, source_key="PLAN", service_date="1970-01-01",
                              service_hour=0, origin_stop_id=str(stop_id), boarding_count=1.0,
                              alight_by_stop=alight_by_stop)
    if len(rows) == 1 and rows[0]["attribution_status"] == E.UNATTRIBUTABLE \
            and rows[0]["route_candidate_count"] == 0:
        return OriginPlan(route_support={}, route_digest="", occurrences={}, od_support={},
                          od_digest={}, destinations={}, terminal=frozenset(),
                          unattributable_reason=rows[0]["unattributable_reason"])

    occurrences = {o.occurrence_id: o for o in index.route_candidates(str(stop_id))}
    route_support = {oid: 1.0 / len(occurrences) for oid in occurrences}
    od_support: Dict[str, Dict[str, float]] = {}
    destinations: Dict[str, E.Occurrence] = {}
    terminal = set()
    for row in rows:
        oid = row["origin_occurrence_id"]
        if row["attribution_status"] == E.UNATTRIBUTABLE:
            terminal.add(oid)
            continue
        od_support.setdefault(oid, {})[row["destination_occurrence_id"]] = float(row["destination_probability"])
    for origin in occurrences.values():
        for d in index.downstream_candidates(origin):
            destinations[d.occurrence_id] = d
    return OriginPlan(route_support, S.distribution_digest(route_support), occurrences,
                      od_support, {oid: S.distribution_digest(dist) for oid, dist in od_support.items()},
                      destinations, frozenset(terminal))


def _draw_route(plan: OriginPlan, contract: S.SeedContract, *, variant: str, source_key: str,
                ordinal: int) -> Tuple[str, int]:
    """Stage 1: inverse-CDF draw over the engine's uniform route attribution prior."""
    support = S.canonical_support(plan.route_support)
    seed = contract.derive(variant=variant, distribution_digest=plan.route_digest,
                           origin_key=source_key, origin_occurrence_id="ROUTE_STAGE",
                           request_ordinal=ordinal)
    u = (seed + 0.5) / float(2 ** 64)
    return S._inverse_cdf(support, u), seed


def realize_bucket(plan: OriginPlan, *, config: LedgerConfig, route_contract: S.SeedContract,
                   dest_contract: S.SeedContract, ts_contract: TS.TimestampContract,
                   service_date: str, service_hour: int, stop_id: str,
                   boarding_count: int) -> List[Dict[str, Any]]:
    """Turn one authoritative bucket into exactly `boarding_count` request rows."""
    n = int(boarding_count)
    if n < 0:
        raise MaterializerContractError("NEGATIVE_BOARDING_COUNT", str(n))
    if n == 0:
        return []
    source_key = f"{service_date}|{service_hour}|{stop_id}"
    stamps = TS.realize_bucket_timestamps(service_date=service_date, service_hour=service_hour,
                                          count=n, source_key=source_key, contract=ts_contract)
    common = {
        "service_date": str(service_date), "service_hour": int(service_hour),
        "source_key": source_key, "origin_stop_id": str(stop_id),
        "origin_semantics": ORIGIN_SEMANTICS,
        "boarding_count_semantics": TS.BOARDING_COUNT_SEMANTICS,
        "source_boarding_count": n,
        "passenger_count": REQUEST_WEIGHT,
        "od_prior_variant": config.variant,
        "demand_realization_seed": config.global_seed,
        "destination_observed": False, "od_ground_truth": False,
        "request_ts_observed": False,
        "provenance_digest": config.provenance_digest,
        **{f"src_{k}": v for k, v in sorted(config.provenance.items())},
    }

    # Unattributable origin: every request is preserved, none receives a destination.
    if plan.unattributable_reason is not None:
        return [{
            **common, **stamps[i],
            "request_id": _request_id(config, source_key, i),
            "realization_status": STATUS_UNATTRIBUTABLE,
            "unattributable_reason": plan.unattributable_reason,
            "route_id": None, "direction_id": None, "origin_occurrence_id": None,
            "origin_stop_sequence": None, "destination_occurrence_id": None,
            "destination_stop_id": None, "destination_stop_sequence": None,
            "route_distribution_digest": None, "od_distribution_digest": None,
            "route_realization_seed": None, "destination_realization_seed": None,
            "destination_semantics": "NOT_APPLICABLE", "support_size": 0,
        } for i in range(n)]

    # Stage 1 for every request, then one sampler call per drawn occurrence.
    assigned: Dict[str, List[int]] = {}
    route_seed: Dict[int, int] = {}
    for i in range(n):
        oid, seed = _draw_route(plan, route_contract, variant=config.variant,
                                source_key=source_key, ordinal=i)
        assigned.setdefault(oid, []).append(i)
        route_seed[i] = seed

    out: Dict[int, Dict[str, Any]] = {}
    for oid, ordinals in assigned.items():
        origin = plan.occurrences[oid]
        base = {
            **common, "route_id": origin.route_id, "direction_id": origin.direction_id,
            "origin_occurrence_id": oid, "origin_stop_sequence": origin.stop_sequence,
            "route_distribution_digest": plan.route_digest,
        }
        if oid in plan.terminal:
            for i in ordinals:
                out[i] = {**base, **stamps[i], "request_id": _request_id(config, source_key, i),
                          "realization_status": STATUS_TERMINAL,
                          "unattributable_reason": "TERMINAL_OCCURRENCE_NO_DOWNSTREAM",
                          "destination_occurrence_id": None, "destination_stop_id": None,
                          "destination_stop_sequence": None, "od_distribution_digest": None,
                          "route_realization_seed": route_seed[i],
                          "destination_realization_seed": None,
                          "destination_semantics": "NOT_APPLICABLE", "support_size": 0}
            continue
        # Stage 2 goes through the frozen R9.4 sampler and nothing else.
        draws = S.realize(plan.od_support[oid], seed_contract=dest_contract, variant=config.variant,
                          origin_key=source_key, origin_occurrence_id=oid,
                          realization_count=len(ordinals))
        for i, draw in zip(sorted(ordinals), draws):
            dest = plan.destinations[draw["destination_occurrence_id"]]
            out[i] = {**base, **stamps[i], "request_id": _request_id(config, source_key, i),
                      "realization_status": STATUS_REALIZED, "unattributable_reason": None,
                      "destination_occurrence_id": dest.occurrence_id,
                      "destination_stop_id": dest.stop_id,
                      "destination_stop_sequence": dest.stop_sequence,
                      "od_distribution_digest": draw["distribution_digest"],
                      "route_realization_seed": route_seed[i],
                      "destination_realization_seed": draw["derived_realization_seed"],
                      "destination_semantics": S.DESTINATION_SEMANTICS,
                      "support_size": draw["support_size"]}
    return [out[i] for i in range(n)]


def _request_id(config: LedgerConfig, source_key: str, ordinal: int) -> str:
    payload = f"{LEDGER_ID}|{config.experiment_id}|{config.global_seed}|{source_key}|{ordinal}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def materialize(index: E.RouteNetworkIndex, engine_config: E.EngineConfig, config: LedgerConfig,
                source_rows: Iterable[Dict[str, Any]],
                alight_by_stop: Optional[Dict[str, float]] = None,
                plan_cache: Optional[Dict[str, OriginPlan]] = None) -> Iterator[Dict[str, Any]]:
    """Stream request rows.  Only one bucket is ever held in memory at a time."""
    cache = plan_cache if plan_cache is not None else {}
    route_contract = S.SeedContract(global_seed=config.global_seed,
                                    experiment_id=f"{config.experiment_id}_ROUTE")
    dest_contract = S.SeedContract(global_seed=config.global_seed,
                                   experiment_id=f"{config.experiment_id}_DEST")
    ts_contract = TS.TimestampContract(global_seed=config.global_seed,
                                       experiment_id=f"{config.experiment_id}_TS",
                                       bucket_unit=config.bucket_unit)
    for row in source_rows:
        stop = str(row["stop_id"])
        if stop not in cache:
            cache[stop] = build_origin_plan(index, engine_config, stop, alight_by_stop)
        yield from realize_bucket(cache[stop], config=config, route_contract=route_contract,
                                  dest_contract=dest_contract, ts_contract=ts_contract,
                                  service_date=str(row["service_date"]),
                                  service_hour=int(row["service_hour"]), stop_id=stop,
                                  boarding_count=int(row["boardings"]))


CANONICAL_ORDER = ("service_date", "service_hour", "origin_stop_id", "request_ts", "request_id")


def canonical_sort(frame):
    """Canonical ledger ordering: chronological, then stable on request identity."""
    return frame.sort_values(list(CANONICAL_ORDER), kind="mergesort").reset_index(drop=True)


def ledger_digest(frame) -> str:
    """Digest of the canonical ledger content, independent of upstream row order."""
    ordered = canonical_sort(frame)
    cols = [c for c in sorted(ordered.columns)]
    return hashlib.sha256(ordered[cols].to_csv(index=False).encode("utf-8")).hexdigest()


def conservation_report(source_rows: List[Dict[str, Any]], frame) -> Dict[str, Any]:
    """Exact accounting: every historical boarding is one row, nothing is dropped."""
    mass = int(sum(int(r["boardings"]) for r in source_rows))
    rows = int(len(frame))
    status = frame["realization_status"].value_counts().to_dict()
    realized = int(status.get(STATUS_REALIZED, 0))
    unattr = int(status.get(STATUS_UNATTRIBUTABLE, 0))
    terminal = int(status.get(STATUS_TERMINAL, 0))
    per_bucket = frame.groupby("source_key").size().to_dict()
    expected = {f"{r['service_date']}|{r['service_hour']}|{r['stop_id']}": int(r["boardings"])
                for r in source_rows if int(r["boardings"]) > 0}
    mismatched = sorted(k for k, v in expected.items() if per_bucket.get(k, 0) != v)
    return {
        "historical_boarding_mass": mass,
        "request_rows_generated": rows,
        "realized_request_count": realized,
        "unattributable_request_count": unattr,
        "unrealizable_terminal_request_count": terminal,
        "accounted_total": realized + unattr + terminal,
        "dropped_mass": mass - rows,
        "exact_one_to_one": mass == rows and realized + unattr + terminal == mass,
        "buckets_with_wrong_row_count": mismatched,
        "passenger_count_values": sorted({int(v) for v in frame["passenger_count"].unique()}),
        "compression_applied": False, "scaling_applied": False,
        **DEMAND_SEMANTICS,
    }


def manifest(config: LedgerConfig, index: E.RouteNetworkIndex, engine_config: E.EngineConfig,
             conservation: Dict[str, Any], digest: str) -> Dict[str, Any]:
    return {
        "ledger_id": LEDGER_ID, "config": config.payload(),
        "engine_id": E.ENGINE_ID, "engine_config": engine_config.payload(),
        "sampler_id": S.SAMPLER_ID, "timestamp_rule_id": TS.RULE_ID,
        "engine_source_sha256": index.source_sha256,
        "handoff_schema": [{"field": f, "meaning": m} for f, m in HANDOFF_SCHEMA],
        "conservation": conservation, "ledger_digest": digest,
        "claim_guards": {
            "destination_observed": False, "od_ground_truth": False, "request_ts_observed": False,
            "actual_passenger_request_ledger_created": False,
            "historical_passenger_trajectory_created": False,
            "simulator_binding_allowed": False, "simulator_execution_allowed": False,
            "training_allowed": False, "performance_comparison_allowed": False,
            "variant_superiority_claim_allowed": False, "paper_level_claim_allowed": False,
            "causal_performance_claim_allowed": False,
            "scoped_inferred_request_ledger_created": True,
        },
    }
