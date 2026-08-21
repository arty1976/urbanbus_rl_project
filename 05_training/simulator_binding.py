#!/usr/bin/env python3
"""H4M-AE-R9.8 live demand-to-simulator binding.

This module exists to make one distinction structural rather than editorial.

`simulator_demand_handoff` is read-only: it projects a ledger, loads it through
the demand contract, and audits the result.  It touches no simulator, so it stays
unguarded and schema validation remains possible while every capability is denied.
Keeping it untouched also keeps the R9.7 ledger digest reproducible, because that
module's sha256 is recorded in the ledger's own provenance.

This module is the live path: the single point where a frozen demand ledger
becomes the input of a causal simulator.  It is guarded by `simulator_binding`.

Binding is not execution.  Holding `simulator_binding` attaches demand and
nothing else; it does not advance a step, and every simulator mutation entrypoint
enforces `simulator_execution` separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import authoritative_demand_realization as ADR
import simulator_authorization as _authz
import simulator_demand_handoff as H

BINDING_ID = "R9_8_LIVE_DEMAND_SIMULATOR_BINDING_V1"
LIVE_BINDING_SITE = "simulator_binding.py::bind_demand_to_simulator"

READ_ONLY_SURFACE = ("simulator_demand_handoff.project", "simulator_demand_handoff.load_only",
                     "simulator_demand_handoff.window_contract", "simulator_demand_handoff.accounting",
                     "simulator_demand_handoff.field_invariance",
                     "simulator_demand_handoff.interface_capability",
                     "simulator_demand_handoff.loaded_demand_digest")
LIVE_BINDING_SURFACE = ("simulator_binding.bind_demand_to_simulator",)

SURFACE_CONTRACT = {
    "binding_id": BINDING_ID,
    "read_only_surface": list(READ_ONLY_SURFACE),
    "read_only_requires_capability": None,
    "read_only_rationale": ("schema and load validation touch no simulator, so they stay available "
                            "while every capability is denied"),
    "live_surface": list(LIVE_BINDING_SURFACE),
    "live_requires_capability": "simulator_binding",
    "binding_implies_execution": False,
    "execution_capability_enforced_separately_at": "every simulator mutation entrypoint",
}


@dataclass(frozen=True)
class BoundDemand:
    """A frozen demand ledger attached to a simulator input, and nothing more."""

    requests: List[ADR.DemandRequest]
    ledger_digest: str
    loaded_demand_digest: str
    serviceable: int
    unserviceable: int
    capability: str = "simulator_binding"

    def payload(self) -> Dict[str, Any]:
        return {
            "binding_id": BINDING_ID,
            "bound_request_count": len(self.requests),
            "serviceable": self.serviceable, "unserviceable": self.unserviceable,
            "ledger_digest": self.ledger_digest,
            "loaded_demand_digest": self.loaded_demand_digest,
            "capability_required": self.capability,
            "execution_capability_required_separately": "simulator_execution",
            "simulator_advanced": False, "simulator_state_mutated": False,
        }


def bind_demand_to_simulator(requests: List[ADR.DemandRequest], *, ledger_digest: str,
                             simulator: Any = None) -> BoundDemand:
    """Attach a frozen demand ledger to the causal simulator input.

    Fail-closed on `simulator_binding`.  The check runs before anything is read
    from or written to `simulator`, so a denied bind leaves it exactly as it was.
    Nothing here steps, resets or advances the simulator: that needs
    `simulator_execution`, which the adapters enforce themselves.
    """
    _authz.require_capability("simulator_binding", site=LIVE_BINDING_SITE)
    serviceable = sum(1 for r in requests if r.destination_realizable)
    bound = BoundDemand(
        requests=list(requests), ledger_digest=ledger_digest,
        loaded_demand_digest=H.loaded_demand_digest(requests),
        serviceable=serviceable, unserviceable=len(requests) - serviceable)
    if simulator is not None:
        # Attaching demand to an input slot is not a causal mutation; the
        # simulator's own entrypoints stay guarded by simulator_execution.
        setattr(simulator, "bound_demand", bound)
    return bound
