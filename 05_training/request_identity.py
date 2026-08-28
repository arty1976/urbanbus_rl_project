#!/usr/bin/env python3
"""H4M-AE-R9.6 request identity contract.

R9.5 carried one identifier that folded the global seed into it, so the same
historical boarding realization could not be followed across seeds.  R9.6 splits
identity in two.

historical_request_key
    The stable identity of one historical boarding within its authoritative
    bucket.  Derived only from evidence that does not move when a seed moves:
    the source bucket key and the request ordinal inside that bucket's observed
    boarding mass.  No seed, no variant, no realization digest.  The population
    of keys is therefore invariant across seeds, which is what makes a
    different-seed comparison non-vacuous.

request_realization_id
    The identity of one *inferred realization* of that boarding: the stable key
    bound to the seed and the realization version.  It identifies a draw, never
    a person.

Both use sha256.  Python's builtin hash() is salted per process and can never
back a persistent identity.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict

STABLE_IDENTITY_ID = "HISTORICAL_REQUEST_KEY_V1"
REALIZATION_IDENTITY_ID = "REQUEST_REALIZATION_ID_V1"

IDENTITY_CONTRACT = {
    "stable_identity": STABLE_IDENTITY_ID,
    "stable_derivation": "sha256(STABLE_IDENTITY_ID | source_key | request_ordinal)",
    "stable_inputs": ["source_key (service_date|service_hour|stop_id)", "request_ordinal"],
    "stable_includes_seed": False,
    "stable_includes_variant": False,
    "stable_includes_realization_digest": False,
    "realization_identity": REALIZATION_IDENTITY_ID,
    "realization_derivation": ("sha256(REALIZATION_IDENTITY_ID | historical_request_key | "
                               "global_seed | realization_version)"),
    "realization_includes_seed": True,
    "hash_function": "hashlib.sha256",
    "python_builtin_hash_used": False,
    "identifies_a_real_person": False,
    "same_boarding_same_key_under_all_seeds": True,
}


@dataclass(frozen=True)
class IdentityContract:
    realization_version: str

    def historical_request_key(self, *, source_key: str, request_ordinal: int) -> str:
        payload = "|".join([STABLE_IDENTITY_ID, str(source_key), str(int(request_ordinal))])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def request_realization_id(self, *, historical_request_key: str, global_seed: int) -> str:
        payload = "|".join([REALIZATION_IDENTITY_ID, historical_request_key,
                            str(int(global_seed)), self.realization_version])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def payload(self) -> Dict[str, Any]:
        return {**IDENTITY_CONTRACT, "realization_version": self.realization_version}
