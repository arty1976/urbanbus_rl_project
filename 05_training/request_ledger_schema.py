#!/usr/bin/env python3
"""H4M-AE-R9.7 versioned request-ledger schema and null-safe destination contract.

Two schema versions exist and both stay meaningful.

REQUEST_LEDGER_SCHEMA_V1
    The R9.5 shape.  Its canonical identity was a single `request_id` that folded
    the global seed into itself.  V1 is archived, not revived: nothing new is
    written in it, and it is reachable only through the R9.5 compatibility view.

REQUEST_LEDGER_SCHEMA_V2
    The R9.6/R9.7 shape.  Canonical identities are `historical_request_key`
    (stable across seeds) and `request_realization_id` (one draw).  `request_id`
    is deliberately NOT restored as a canonical identity.

Null-safe destination
---------------------
An absent destination is a real state, not a missing string.  A request that
cannot reach any legal destination carries a true null destination plus an
explicit boolean and an explicit reason, so a consumer never has to parse text
or infer intent from a sentinel:

    destination_realizable : bool
    destination_stop_id    : nullable stop id, null exactly when not realizable
    realization_status     : the status enum
    unrealizable_reason    : nullable reason, non-null exactly when not realizable

No sentinel stop id is invented.  The repository documents none, so none is used:
"None", "NULL", "-1", "UNKNOWN_STOP" and 0 are all forbidden as destinations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

SCHEMA_V1 = "REQUEST_LEDGER_SCHEMA_V1"
SCHEMA_V2 = "REQUEST_LEDGER_SCHEMA_V2"
CURRENT_SCHEMA = SCHEMA_V2

# Values that must never appear as a destination stop id.
FORBIDDEN_DESTINATION_SENTINELS = ("None", "NULL", "null", "NaN", "nan", "-1", "UNKNOWN_STOP", "0", "")
DOCUMENTED_SENTINEL = None      # the repository documents no destination sentinel

NULL_SAFE_CONTRACT = {
    "contract_id": "NULL_SAFE_DESTINATION_CONTRACT_V1",
    "absent_destination_representation": "true null, never a string and never a sentinel id",
    "fields": {
        "destination_realizable": "bool; false exactly when no legal destination exists",
        "destination_stop_id": "nullable stop id; null exactly when destination_realizable is false",
        "realization_status": "explicit status enum",
        "unrealizable_reason": "nullable reason; non-null exactly when destination_realizable is false",
    },
    "consumer_rule": ("read destination_realizable, or test destination_stop_id for null; never parse "
                      "text and never consult an unrelated field"),
    "forbidden_sentinels": list(FORBIDDEN_DESTINATION_SENTINELS),
    "documented_repository_sentinel": DOCUMENTED_SENTINEL,
    "sentinel_invented": False,
    "stringified_null_allowed": False,
}

SCHEMA_V1_CONTRACT = {
    "schema_version": SCHEMA_V1,
    "status": "ARCHIVED_R9_5",
    "canonical_identity": ["request_id"],
    "identity_includes_seed": True,
    "null_safe_destination_fields": False,
    "written_by_current_source": False,
    "reachable_via": "R9.5 compatibility view only",
}

SCHEMA_V2_CONTRACT = {
    "schema_version": SCHEMA_V2,
    "status": "CURRENT",
    "canonical_identity": ["historical_request_key", "request_realization_id"],
    "identity_includes_seed": {"historical_request_key": False, "request_realization_id": True},
    "legacy_request_id_restored_as_canonical": False,
    "null_safe_destination_fields": True,
    "null_safe_contract": NULL_SAFE_CONTRACT,
}

SCHEMA_REGISTRY = {SCHEMA_V1: SCHEMA_V1_CONTRACT, SCHEMA_V2: SCHEMA_V2_CONTRACT}


class SchemaContractError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


def is_absent(value: Any) -> bool:
    """True when a destination value represents 'no destination'.

    Only a real null counts.  A string that merely looks like a null is a defect,
    not an absent destination, and is reported as one rather than accepted.
    """
    if value is None:
        return True
    try:
        import pandas as pd
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def optional_text(value: Any) -> Optional[str]:
    """Carry a value across an interface boundary without stringifying a null."""
    return None if is_absent(value) else str(value)


def validate_null_safety(frame, *, realized_status: str,
                         statuses: Sequence[str] = ()) -> Dict[str, Any]:
    """Row-level proof that the null-safe contract holds across a ledger."""
    realized = frame[frame["realization_status"] == realized_status]
    unrealizable = frame[frame["realization_status"] != realized_status]
    dest = frame["destination_stop_id"]
    as_text = dest.astype("object")
    stringified = int(sum(1 for v in as_text if isinstance(v, str)
                          and v.strip() in FORBIDDEN_DESTINATION_SENTINELS))
    true_null = int(sum(1 for v in as_text if is_absent(v)))
    return {
        "contract": NULL_SAFE_CONTRACT,
        "rows": int(len(frame)),
        "realized_rows": int(len(realized)),
        "unrealizable_rows": int(len(unrealizable)),
        "true_null_destinations": true_null,
        "stringified_or_sentinel_destinations": stringified,
        "realized_with_null_destination": int(sum(1 for v in realized["destination_stop_id"] if is_absent(v))),
        "unrealizable_with_non_null_destination": int(
            sum(1 for v in unrealizable["destination_stop_id"] if not is_absent(v))),
        "realizable_flag_present": "destination_realizable" in frame.columns,
        "realizable_matches_null": (
            int((frame["destination_realizable"].astype(bool)
                 != dest.map(lambda v: not is_absent(v))).sum())
            if "destination_realizable" in frame.columns else None),
        "realized_missing_reason": int(realized["unrealizable_reason"].notna().sum())
        if "unrealizable_reason" in frame.columns else None,
        "unrealizable_missing_reason": int(unrealizable["unrealizable_reason"].isna().sum())
        if "unrealizable_reason" in frame.columns else None,
        "statuses_present": sorted(frame["realization_status"].unique().tolist()),
        "expected_statuses": list(statuses),
    }
