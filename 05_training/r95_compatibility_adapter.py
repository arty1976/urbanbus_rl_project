#!/usr/bin/env python3
"""H4M-AE-R9.7 R9.5 backward-compatibility adapter.

R9.6 replaced the R9.5 identity (`request_id`, which folded the global seed into
itself) with `historical_request_key` and `request_realization_id`.  That left the
archived R9.5 artifact intact but its validator unable to re-run.

This adapter restores reproducible R9.5 validation without reviving V1 semantics
in the live path.  It projects a current V2 ledger back onto the frozen V1 shape:
it recomputes the legacy `request_id` from the exact R9.5 derivation, drops the
V2-only columns, and applies the V1 canonical ordering.  Nothing here changes the
archived artifact, and `request_id` is never restored as a canonical identity --
it exists only inside this view.

Provenance is the one thing the view cannot recompute.  A V1 row carries the
sha256 of the materializer and timestamp modules *as they existed when R9.5 ran*,
and R9.6 modified both.  Those bytes are not derivable from current source, so
the adapter takes them as declared inputs, read from the archived R9.5 manifest,
which is the lineage record of what those files were.  This is stated in the
result rather than hidden: the realization columns are regenerated, the historical
provenance columns are carried.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

ADAPTER_ID = "R9_5_LEGACY_LEDGER_COMPATIBILITY_VIEW_V1"

# The exact R9.5 derivation, reproduced verbatim so the view is a reconstruction
# rather than a re-implementation.
LEGACY_LEDGER_ID = "SCOPED_ONE_TO_ONE_INFERRED_REQUEST_LEDGER_V1"
LEGACY_EXPERIMENT_ID = "H4M_AE_R9_5"
LEGACY_CANONICAL_ORDER = ("service_date", "service_hour", "origin_stop_id", "request_ts", "request_id")

# Frozen V1 column contract, verified against the archived artifact by the gate.
V1_COLUMNS: Sequence[str] = (
    "boarding_count_semantics", "bucket_end", "bucket_start",
    "demand_realization_seed", "destination_observed", "destination_occurrence_id",
    "destination_realization_seed", "destination_semantics", "destination_stop_id",
    "destination_stop_sequence", "direction_id", "od_distribution_digest",
    "od_ground_truth", "od_prior_variant", "origin_occurrence_id",
    "origin_semantics", "origin_stop_id", "origin_stop_sequence",
    "passenger_count", "provenance_digest", "realization_status",
    "request_id", "request_ordinal", "request_ts",
    "request_ts_observed", "request_ts_offset_seconds", "request_ts_semantics",
    "route_distribution_digest", "route_id", "route_realization_seed",
    "service_date", "service_hour", "source_boarding_count",
    "source_key", "src_citywide_ledger_dataset", "src_engine_module",
    "src_materializer_module", "src_path_cost_repair_edges",
    "src_route_attribution_occurrence_master", "src_sampler_module",
    "src_timestamp_module", "support_size",
    "timestamp_realization_seed", "timestamp_rule_id", "unattributable_reason",
)

# Columns a V1 row carries that record the state of files at R9.5 execution time.
HISTORICAL_PROVENANCE_COLUMNS = tuple(c for c in V1_COLUMNS if c.startswith("src_")) + ("provenance_digest",)


class CompatibilityError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class LegacyProvenance:
    """R9.5-era provenance, declared from the archived manifest, never recomputed."""

    values: Mapping[str, str]
    provenance_digest: str
    source: str = "archived R9.5 provenance_manifest.json"

    def payload(self) -> Dict[str, Any]:
        return {"values": dict(self.values), "provenance_digest": self.provenance_digest,
                "source": self.source, "recomputed_from_current_source": False,
                "reason": ("a V1 row records the sha256 of the materializer and timestamp modules as "
                           "they were when R9.5 ran; R9.6 modified both, so those bytes cannot be "
                           "derived from current source")}


def legacy_request_id(*, source_key: str, request_ordinal: int, global_seed: int,
                      experiment_id: str = LEGACY_EXPERIMENT_ID) -> str:
    """Reproduce the R9.5 `request_id` derivation exactly."""
    payload = f"{LEGACY_LEDGER_ID}|{experiment_id}|{global_seed}|{source_key}|{request_ordinal}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def legacy_canonical_sort(frame: pd.DataFrame) -> pd.DataFrame:
    """The V1 canonical ordering, whose final tiebreak is the legacy identity."""
    return frame.sort_values(list(LEGACY_CANONICAL_ORDER), kind="mergesort").reset_index(drop=True)


def legacy_digest(frame: pd.DataFrame) -> str:
    """The V1 ledger digest, computed the way R9.5 computed it."""
    ordered = legacy_canonical_sort(frame)
    return hashlib.sha256(ordered[sorted(ordered.columns)].to_csv(index=False).encode("utf-8")).hexdigest()


def to_v1_view(ledger: pd.DataFrame, *, global_seed: int,
               provenance: LegacyProvenance) -> pd.DataFrame:
    """Project a V2 ledger onto the frozen V1 shape."""
    missing = [c for c in V1_COLUMNS
               if c not in ledger.columns and c != "request_id" and c not in HISTORICAL_PROVENANCE_COLUMNS]
    if missing:
        raise CompatibilityError("V2_LEDGER_MISSING_V1_COLUMNS", str(missing))
    view = ledger.copy()
    view["request_id"] = [
        legacy_request_id(source_key=str(sk), request_ordinal=int(ordinal), global_seed=global_seed)
        for sk, ordinal in zip(view["source_key"], view["request_ordinal"])]
    for column in HISTORICAL_PROVENANCE_COLUMNS:
        if column == "provenance_digest":
            view[column] = provenance.provenance_digest
        else:
            key = column[len("src_"):]
            if key not in provenance.values:
                raise CompatibilityError("LEGACY_PROVENANCE_MISSING_KEY", key)
            view[column] = provenance.values[key]
    return legacy_canonical_sort(view[list(V1_COLUMNS)])


def compare_to_archive(view: pd.DataFrame, archived: pd.DataFrame,
                       archived_digest: str) -> Dict[str, Any]:
    """Prove, column by column, what the view does and does not reproduce."""
    left = legacy_canonical_sort(view)
    right = legacy_canonical_sort(archived[list(V1_COLUMNS)])
    per_column = {}
    for column in V1_COLUMNS:
        a = left[column].astype("object").where(left[column].notna(), "MISSING").astype(str)
        b = right[column].astype("object").where(right[column].notna(), "MISSING").astype(str)
        per_column[column] = int((a.values != b.values).sum())
    reconstructed = legacy_digest(left)
    differing = sorted(c for c, n in per_column.items() if n)
    return {
        "adapter_id": ADAPTER_ID,
        "archived_rows": int(len(right)), "view_rows": int(len(left)),
        "row_count_matches": int(len(left)) == int(len(right)),
        "columns_compared": len(V1_COLUMNS),
        "columns_differing": differing,
        "differing_cell_counts": {c: per_column[c] for c in differing},
        "realization_columns_identical": not [c for c in differing
                                              if c not in HISTORICAL_PROVENANCE_COLUMNS],
        "reconstructed_digest": reconstructed,
        "archived_digest": archived_digest,
        "byte_identical_reconstruction": reconstructed == archived_digest,
        "archived_digest_overwritten": False,
    }
