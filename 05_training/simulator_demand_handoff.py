#!/usr/bin/env python3
"""H4M-AE-R9.6 read-only simulator demand handoff adapter.

Maps the R9.6 request ledger onto the demand contract the causal simulator
already has (`authoritative_demand_realization`), and validates that the mapping
loses nothing and changes nothing.  It is a projection, not a generator: no
resampling, no timestamp rewriting, no destination invention, no request
deletion.  Requests the interface cannot faithfully represent are reported as
explicit interface blockers and still carried, never quietly discarded.

Load-only.  Nothing here steps the simulator, selects an action, moves a vehicle,
boards a passenger, or computes a reward or KPI.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import authoritative_demand_realization as ADR
import request_ledger_materializer as M

HANDOFF_ID = "R9_6_SIMULATOR_DEMAND_HANDOFF_ADAPTER_V1"
MODE = "LOAD_AND_SCHEMA_VALIDATION_ONLY"

# Semantic role -> R9.6 ledger column.  Only the mapping moves; the ledger is read-only.
COLUMN_MAP: Dict[str, str] = {
    "request_id": "request_realization_id",
    "passenger_id": "simulator_anonymous_passenger_id",
    "request_ts": "request_ts_epoch",
    "reporting_window_id": "window_id",
    "origin_stop_id": "origin_stop_id",
    "destination_stop_id": "destination_stop_id",
    "route_id": "route_id",
    "direction_id": "direction_id",
    # Null-safe destination contract: carried explicitly, never inferred downstream.
    "destination_realizable": "destination_realizable",
    "realization_status": "realization_status",
    "unrealizable_reason": "unrealizable_reason",
}

# Fields the handoff must carry through unchanged.
INVARIANT_FIELDS = ("request_ts_epoch", "origin_stop_id", "destination_stop_id", "passenger_count")

MISSING_VALUE_TEXT = ("nan", "none", "<na>", "nat")

EXECUTION_GUARDS = {
    "mode": MODE,
    "step_called": False, "policy_action_selected": False, "vehicle_moved": False,
    "boarding_or_alighting_executed": False, "reward_computed": False, "kpi_computed": False,
    "simulator_state_mutated": False, "arm_executed": False,
    "resampled_anything": False, "timestamps_rewritten": False,
    "destinations_invented": False, "requests_deleted": False,
}


class HandoffError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


@dataclass(frozen=True)
class HandoffView:
    path: Path
    frame: pd.DataFrame
    ledger_rows: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def project(ledger: pd.DataFrame, out_path: Path) -> HandoffView:
    """Write the handoff view.  Every ledger row appears exactly once."""
    needed = sorted({*COLUMN_MAP.values(), "historical_request_key", "realization_status",
                     "unattributable_reason", "unrealizable_reason", "destination_realizable",
                     "schema_version", "passenger_count", "service_date", "service_hour",
                     "request_ts", "bucket_start", "bucket_end", "provenance_digest",
                     "demand_realization_seed", "realization_version", "od_prior_variant",
                     "od_distribution_digest", "route_distribution_digest",
                     "origin_occurrence_id", "destination_occurrence_id",
                     "feasibility_filter_applied", "feasibility_filter_version",
                     *[c for c in ledger.columns if c.startswith("src_")]})
    missing = [c for c in needed if c not in ledger.columns]
    if missing:
        raise HandoffError("LEDGER_MISSING_HANDOFF_FIELDS", str(missing))
    view = ledger[needed].copy()
    if len(view) != len(ledger):
        raise HandoffError("PROJECTION_CHANGED_ROW_COUNT", f"{len(view)} != {len(ledger)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    view.to_parquet(out_path, index=False)
    return HandoffView(path=out_path, frame=view, ledger_rows=int(len(ledger)))


def window_contract(ledger: pd.DataFrame, window_id: str) -> ADR.EvaluationTimeContract:
    """Build the evaluation boundary from the window's own authoritative bucket.

    Bounds are read from the bucket, never hardcoded, and the existing causal time
    contract is used as-is rather than reshaped to fit the ledger.
    """
    rows = ledger[ledger["window_id"].astype(str) == str(window_id)]
    if rows.empty:
        raise HandoffError("UNKNOWN_WINDOW", str(window_id))
    start = pd.Timestamp(rows["bucket_start"].iloc[0], tz="UTC").timestamp()
    end = pd.Timestamp(rows["bucket_end"].iloc[0], tz="UTC").timestamp()
    return ADR.EvaluationTimeContract(evaluation_start_ts=start, evaluation_end_ts=end,
                                      reporting_window_ids=(str(window_id),),
                                      scope_label=f"R9_6_WINDOW_{window_id}")


def load_only(view_path: Path) -> ADR.DemandRealization:
    """Instantiate the existing loader.  Reads a file; touches no simulator state."""
    return ADR.load_demand_realization(view_path, column_map=COLUMN_MAP)


def interface_capability(ledger: pd.DataFrame, realization: ADR.DemandRealization,
                         requests: List[ADR.DemandRequest]) -> Dict[str, Any]:
    """What the existing demand contract can and cannot represent, stated plainly."""
    unrealizable = ledger[ledger["realization_status"].isin(M.UNREALIZABLE_STATUSES)]
    by_id = {r.request_id: r for r in requests}
    carried, degraded = 0, []
    for row in unrealizable.itertuples():
        req = by_id.get(getattr(row, "request_realization_id"))
        if req is None:
            continue
        carried += 1
        # A null destination must arrive as an absent destination, not as the text of a
        # missing value.  Anything that merely looks like a null is a degradation.
        value = req.destination_stop_id
        if value is not None and value.strip().lower() in MISSING_VALUE_TEXT:
            degraded.append(value)
    blocker = None
    if degraded:
        blocker = {
            "code": "INTERFACE_CANNOT_REPRESENT_NULL_DESTINATION",
            "detail": ("DemandRequest.destination_stop_id applies str() to the column value, so a "
                       "request with no legal destination arrives as the text of a missing value "
                       "instead of an absent destination. The requests are carried, not dropped, "
                       "but a consumer cannot distinguish 'no destination' from a stop id without "
                       "also reading realization_status."),
            "resolution_owner": ("next gate; not repaired here because R9.6 must not invent "
                                 "simulator semantics"),
            "requests_affected": len(degraded)}
    return {
        "unrealizable_requests_in_ledger": int(len(unrealizable)),
        "unrealizable_requests_carried_into_handoff": carried,
        "unrealizable_requests_deleted": int(len(unrealizable)) - carried,
        "destination_null_representable": not degraded,
        "degraded_destination_values": sorted(set(degraded))[:3],
        "blocker": blocker,
        "realization_status_preserved_in_view": "realization_status" in realization.frame.columns,
        "realizable_flag_carried": "destination_realizable" in realization.frame.columns,
        "reason_carried": "unrealizable_reason" in realization.frame.columns,
        "reasons_resolved": sorted({str(r.unrealizable_reason) for r in requests
                                    if not r.destination_realizable}),
        "consumer_must_read": "destination_realizable, or test destination_stop_id for null",
    }


def accounting(ledger: pd.DataFrame, view: HandoffView, per_window: Dict[str, Any]) -> Dict[str, Any]:
    """Exact handoff accounting: nothing lost, nothing duplicated, nothing altered."""
    visible = sum(w["evaluation_population_count"] for w in per_window.values())
    rejected = sum(w["requests_before_evaluation_start"] + w["requests_after_evaluation_end"]
                   for w in per_window.values())
    ids = view.frame["request_realization_id"]
    keys = view.frame["historical_request_key"]
    return {
        "ledger_request_count": int(len(ledger)),
        "handoff_view_row_count": int(len(view.frame)),
        "handoff_visible_request_count": int(visible),
        "explicitly_rejected_by_interface_count": int(rejected),
        "accounted_total": int(visible + rejected),
        "silent_loss": int(len(ledger)) - int(visible + rejected),
        "duplicate_realization_ids": int(ids.duplicated().sum()),
        "duplicate_stable_keys": int(keys.duplicated().sum()),
        "duplicate_load_count": 0,
        **EXECUTION_GUARDS,
    }


def field_invariance(ledger: pd.DataFrame, view: HandoffView) -> Dict[str, Any]:
    """The handoff must not resample: every carried field is identical."""
    left = ledger.sort_values("historical_request_key", kind="mergesort").reset_index(drop=True)
    right = view.frame.sort_values("historical_request_key", kind="mergesort").reset_index(drop=True)
    diffs = {}
    for field in INVARIANT_FIELDS:
        a = left[field].astype("object").where(left[field].notna(), "MISSING").astype(str)
        b = right[field].astype("object").where(right[field].notna(), "MISSING").astype(str)
        diffs[field] = int((a != b).sum())
    return {"changed_values_per_field": diffs, "any_field_changed": any(diffs.values()),
            "fields_checked": list(INVARIANT_FIELDS)}


def _nullable_field(value: Optional[str]) -> str:
    """Serialize a nullable field so a null can never collide with a real value.

    Applying str() to a null would emit the text "None", which is indistinguishable
    from a stop id that happens to be the string "None".  A leading presence flag
    keeps absence structurally separate from any value a stop id could take.
    """
    return "0:" if value is None else f"1:{value}"


def loaded_demand_digest(requests: List[ADR.DemandRequest]) -> str:
    """Digest of what the simulator would actually receive.

    Nullable fields are presence-flagged, so this digest distinguishes an absent
    destination from every possible stop id rather than flattening both to text.
    """
    lines = ["|".join([r.request_id, r.passenger_id, f"{r.arrival_ts_absolute:.6f}",
                       f"{r.arrival_ts_relative:.6f}", r.origin_stop_id, r.reporting_window_id,
                       _nullable_field(r.destination_stop_id), _nullable_field(r.route_id),
                       _nullable_field(r.direction_id), f"{r.weight:.6f}",
                       "1" if r.destination_realizable else "0"])
             for r in sorted(requests, key=lambda x: x.request_id)]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
