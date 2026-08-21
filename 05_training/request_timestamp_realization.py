#!/usr/bin/env python3
"""H4M-AE-R9.5 deterministic within-bucket request timestamp realization.

The authoritative 2023 ledger observes a boarding *count* per
`service_date x service_hour x stop_id` bucket.  It does not observe when inside
that bucket any individual passenger boarded.  This module therefore realizes
timestamps; it never reconstructs them.

The realization rule is fixed here, before any ledger is produced and before any
downstream behaviour is inspected.  It is not selected by looking at results.

Rule
----
For a bucket holding N boardings, draw N i.i.d. uniform offsets on
[0, bucket_seconds), then sort ascending and assign them to request ordinals in
order.  Uniform is the maximum-entropy choice given that the only observed
quantity is the bucket total: any non-uniform within-bucket shape would encode
an arrival profile for which this project holds no evidence.  Sorting makes the
ordinal-to-timestamp map canonical, so the ledger is chronologically ordered
inside each bucket without a second sort pass.

Every offset satisfies 0 <= offset < bucket_seconds, so a realized timestamp can
never leave its authoritative bucket, never crosses the service date, and never
uses information from a later bucket.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date as _date, datetime, timedelta
from typing import Any, Dict, List

RULE_ID = "UNIFORM_WITHIN_AUTHORITATIVE_BUCKET_SORTED_ASCENDING_V1"
TIMESTAMP_SEMANTICS = "INFERRED_TIME_REALIZATION"
BOARDING_COUNT_SEMANTICS = "OBSERVED_HISTORICAL"

# Bucket width is resolved from the ledger's own declared grain, never assumed.
# The frozen ledger manifest declares service_hour with unit "hour of day 0-23".
BUCKET_UNIT_SECONDS = {"service_hour": 3600, "service_minute": 60, "service_day": 86400}

TIMESTAMP_RULE = {
    "rule_id": RULE_ID,
    "within_bucket_distribution": "uniform i.i.d. on [0, bucket_seconds)",
    "ordering": "offsets sorted ascending, then assigned to request ordinals in order",
    "justification": ("maximum entropy given that only the bucket total is observed; a non-uniform "
                      "profile would encode within-bucket arrival evidence this project does not hold"),
    "declared_before_results": True,
    "selected_after_inspecting_downstream_performance": False,
    "boarding_count_semantics": BOARDING_COUNT_SEMANTICS,
    "request_ts_semantics": TIMESTAMP_SEMANTICS,
    "request_ts_observed": False,
    "reproduces_actual_within_bucket_arrivals": False,
    "future_information_used": False,
}


class TimestampContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class TimestampContract:
    """Timestamp randomness is derived from the demand seed, never a global RNG."""

    global_seed: int
    experiment_id: str = "H4M_AE_R9_5_TS"
    bucket_unit: str = "service_hour"

    @property
    def bucket_seconds(self) -> int:
        if self.bucket_unit not in BUCKET_UNIT_SECONDS:
            raise TimestampContractError("UNKNOWN_BUCKET_UNIT", self.bucket_unit)
        return BUCKET_UNIT_SECONDS[self.bucket_unit]

    def payload(self) -> Dict[str, Any]:
        return {
            "rule_id": RULE_ID, "experiment_id": self.experiment_id,
            "global_seed": self.global_seed, "bucket_unit": self.bucket_unit,
            "bucket_seconds": self.bucket_seconds,
            "bucket_seconds_source": "ledger manifest grain declaration, not a literal assumption",
            "derivation": "sha256(experiment_id | global_seed | source_key | request_ordinal)",
            "hash_function": "hashlib.sha256",
            "python_builtin_hash_used": False,
            "global_rng_state_used": False,
            "unseeded_random_calls": 0,
            "depends_on_row_iteration_order": False,
            "separate_from_mappo_rng": True,
            "separate_from_training_rng": True,
            **TIMESTAMP_RULE,
        }

    def derive(self, *, source_key: str, request_ordinal: int) -> int:
        payload = "|".join([self.experiment_id, str(self.global_seed), source_key, str(request_ordinal)])
        return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def bucket_bounds(service_date: str, service_hour: int, contract: TimestampContract) -> tuple:
    """Absolute [start, end) of one authoritative bucket."""
    hour = int(service_hour)
    if hour < 0 or hour * contract.bucket_seconds >= 86400:
        raise TimestampContractError("BUCKET_OUTSIDE_SERVICE_DATE", f"{service_date} {service_hour}")
    start = datetime.combine(_date.fromisoformat(str(service_date)), datetime.min.time()) \
        + timedelta(seconds=hour * contract.bucket_seconds)
    return start, start + timedelta(seconds=contract.bucket_seconds)


def realize_bucket_timestamps(*, service_date: str, service_hour: int, count: int,
                              source_key: str, contract: TimestampContract) -> List[Dict[str, Any]]:
    """Realize exactly `count` inferred timestamps inside one authoritative bucket."""
    if count < 0:
        raise TimestampContractError("NEGATIVE_REQUEST_COUNT", str(count))
    start, end = bucket_bounds(service_date, service_hour, contract)
    span = float(contract.bucket_seconds)
    draws = []
    for ordinal in range(count):
        seed = contract.derive(source_key=source_key, request_ordinal=ordinal)
        offset = ((seed + 0.5) / float(2 ** 64)) * span
        draws.append((offset, seed))
    draws.sort(key=lambda t: (t[0], t[1]))
    out: List[Dict[str, Any]] = []
    for ordinal, (offset, seed) in enumerate(draws):
        ts = start + timedelta(seconds=offset)
        if not (start <= ts < end):
            raise TimestampContractError("TIMESTAMP_LEFT_AUTHORITATIVE_BUCKET", ts.isoformat())
        out.append({
            "request_ordinal": ordinal,
            "request_ts": ts.isoformat(sep=" ", timespec="milliseconds"),
            "request_ts_offset_seconds": round(offset, 6),
            "bucket_start": start.isoformat(sep=" "), "bucket_end": end.isoformat(sep=" "),
            "request_ts_semantics": TIMESTAMP_SEMANTICS,
            "request_ts_observed": False,
            "timestamp_realization_seed": seed,
            "timestamp_rule_id": RULE_ID,
        })
    return out


def spacing_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Descriptive only.  Never a claim that realized spacing matches real arrivals."""
    offs = sorted(float(r["request_ts_offset_seconds"]) for r in rows)
    if len(offs) < 2:
        return {"count": len(offs), "gaps": 0, "note": "insufficient rows for spacing"}
    gaps = [b - a for a, b in zip(offs, offs[1:])]
    gaps_sorted = sorted(gaps)
    q = lambda f: gaps_sorted[min(len(gaps_sorted) - 1, int(f * len(gaps_sorted)))]
    return {
        "count": len(offs), "gaps": len(gaps),
        "min_offset": round(offs[0], 3), "max_offset": round(offs[-1], 3),
        "mean_gap_seconds": round(sum(gaps) / len(gaps), 4),
        "min_gap_seconds": round(gaps_sorted[0], 4),
        "p50_gap_seconds": round(q(0.50), 4), "p95_gap_seconds": round(q(0.95), 4),
        "max_gap_seconds": round(gaps_sorted[-1], 4),
        "monotonic_non_decreasing": all(a <= b for a, b in zip(offs, offs[1:])),
        "interpretation": "descriptive spacing of an inferred realization; not observed arrivals",
    }
