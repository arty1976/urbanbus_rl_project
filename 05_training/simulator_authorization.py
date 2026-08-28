#!/usr/bin/env python3
"""H4M-AE-R9.8 canonical simulator authorization.

Until now the repository *declared* permissions inside artifact payloads and
nothing read them, so `simulator_binding_allowed = false` was a statement rather
than a control.  This module turns those statements into a mechanism.

Capabilities form a strict ladder, and holding one never confers the next:

    simulator_binding      attach the frozen demand ledger to the simulator input
    simulator_execution    advance causal simulator state at all
    training               update parameters or write a checkpoint
    performance_comparison run an official B1/B2/A causal comparison
    causal_performance_claim  publish a causal performance claim

Every capability defaults to deny.  Absent means deny, false means deny, and
only an explicit grant allows.  `require_capability` raises before the caller
performs any mutation, so a denied call leaves state untouched.

Granting is explicit, scoped and auditable: there is no environment variable and
no global mutable switch that a stray import could flip.  A caller that is
entitled to act says so in code, for the narrowest block that needs it:

    with granted("simulator_execution", reason="R9.9 authorized rollout"):
        adapter.step(actions)

The grant stack is thread-local, so one thread's authorization never leaks into
another.  Every grant and every denial is recorded in an in-process audit log
that a gate can read back as evidence.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

AUTHORIZATION_ID = "SIMULATOR_CAPABILITY_AUTHORIZATION_V1"

SIMULATOR_BINDING = "simulator_binding"
SHADOW_COUNTERFACTUAL = "shadow_counterfactual"
SIMULATOR_EXECUTION = "simulator_execution"
TRAINING = "training"
PERFORMANCE_COMPARISON = "performance_comparison"
CAUSAL_PERFORMANCE_CLAIM = "causal_performance_claim"

# Ordered strictly weakest to strongest.  The order documents the ladder; it is
# deliberately NOT used to imply anything: holding one never grants another.
CAPABILITY_LADDER: Tuple[str, ...] = (
    SIMULATOR_BINDING,
    SHADOW_COUNTERFACTUAL,
    SIMULATOR_EXECUTION,
    TRAINING,
    PERFORMANCE_COMPARISON,
    CAUSAL_PERFORMANCE_CLAIM,
)

CAPABILITY_MEANING: Dict[str, str] = {
    SIMULATOR_BINDING: "attach the frozen demand ledger to the causal simulator input",
    SHADOW_COUNTERFACTUAL: ("advance a disposable deep copy of state for a counterfactual that is "
                            "discarded; never the live authoritative state"),
    SIMULATOR_EXECUTION: "advance the live authoritative causal simulator state",
    TRAINING: "update parameters, step an optimizer, or write a checkpoint",
    PERFORMANCE_COMPARISON: "run an official B1/B2/A causal comparison",
    CAUSAL_PERFORMANCE_CLAIM: "publish a causal performance claim",
}

AUTHORIZATION_CONTRACT = {
    "authorization_id": AUTHORIZATION_ID,
    "default": "DENY",
    "absent_capability": "DENY",
    "explicit_false": "DENY",
    "allow_requires": "an explicit in-code grant",
    "ladder": list(CAPABILITY_LADDER),
    "meaning": dict(CAPABILITY_MEANING),
    "implicit_escalation": False,
    "shadow_counterfactual_is_not_execution": True,
    "shadow_counterfactual_may_touch_live_state": False,
    "simulator_execution_meaning_unchanged": True,
    "holding_one_capability_grants_another": False,
    "policy_independent": True,
    "arm_or_policy_specific_bypass": False,
    "environment_variable_override": False,
    "global_mutable_switch": False,
    "grant_scope": "thread-local, block-scoped context manager",
    "denial_timing": "before any state mutation",
}


class AuthorizationDenied(RuntimeError):
    """Raised before a protected action performs any mutation."""

    def __init__(self, capability: str, site: str = "", detail: str = "") -> None:
        self.capability = capability
        self.site = site
        self.detail = detail
        message = f"AUTHORIZATION_DENIED: {capability}"
        if site:
            message += f" at {site}"
        if detail:
            message += f" ({detail})"
        super().__init__(message)


@dataclass
class _AuditEntry:
    capability: str
    site: str
    outcome: str
    ts: float = field(default_factory=time.time)

    def payload(self) -> Dict[str, Any]:
        return {"capability": self.capability, "site": self.site, "outcome": self.outcome}


class _State(threading.local):
    def __init__(self) -> None:
        self.granted: List[Tuple[str, str]] = []


_STATE = _State()
_AUDIT: List[_AuditEntry] = []
_AUDIT_LOCK = threading.Lock()


def _record(capability: str, site: str, outcome: str) -> None:
    with _AUDIT_LOCK:
        _AUDIT.append(_AuditEntry(capability=capability, site=site, outcome=outcome))


def is_granted(capability: str) -> bool:
    """True only when this thread currently holds an explicit grant.

    No ladder logic: `simulator_execution` does not answer for `training`, and a
    capability nobody granted is simply absent, which is a denial.
    """
    if capability not in CAPABILITY_LADDER:
        return False
    return any(name == capability for name, _ in getattr(_STATE, "granted", []))


def require_capability(capability: str, *, site: str = "") -> None:
    """Fail closed.  Call this before the first mutation, never after."""
    if capability not in CAPABILITY_LADDER:
        _record(capability, site, "DENIED_UNKNOWN_CAPABILITY")
        raise AuthorizationDenied(capability, site, "unknown capability")
    if not is_granted(capability):
        _record(capability, site, "DENIED")
        raise AuthorizationDenied(capability, site, "no explicit grant is in scope")
    _record(capability, site, "ALLOWED")


def check_capability(capability: str, *, site: str = "") -> bool:
    """Non-raising probe.  Answers whether a call would be permitted, and does
    not itself authorize anything."""
    allowed = is_granted(capability)
    _record(capability, site, "PROBE_ALLOWED" if allowed else "PROBE_DENIED")
    return allowed


@contextmanager
def granted(*capabilities: str, reason: str) -> Iterator[None]:
    """Grant capabilities for one block, on this thread only.

    `reason` is required so an authorization always carries a written
    justification into the audit log.
    """
    if not reason or not reason.strip():
        raise AuthorizationDenied(",".join(capabilities), "granted", "a grant requires a reason")
    unknown = [c for c in capabilities if c not in CAPABILITY_LADDER]
    if unknown:
        raise AuthorizationDenied(",".join(unknown), "granted", "unknown capability")
    entries = [(c, reason) for c in capabilities]
    stack = getattr(_STATE, "granted", [])
    for capability in capabilities:
        _record(capability, "granted", f"GRANT_OPEN:{reason}")
    stack.extend(entries)
    _STATE.granted = stack
    try:
        yield
    finally:
        remaining = list(getattr(_STATE, "granted", []))
        for entry in entries:
            if entry in remaining:
                remaining.remove(entry)
        _STATE.granted = remaining
        for capability in capabilities:
            _record(capability, "granted", "GRANT_CLOSED")


def authorization_state() -> Dict[str, Any]:
    """The capability state as a gate should report it."""
    return {
        "authorization_id": AUTHORIZATION_ID,
        "capabilities": {c: is_granted(c) for c in CAPABILITY_LADDER},
        "contract": AUTHORIZATION_CONTRACT,
    }


def audit_log(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with _AUDIT_LOCK:
        entries = [e.payload() for e in _AUDIT]
    return entries[-limit:] if limit else entries


def reset_audit_log() -> None:
    with _AUDIT_LOCK:
        _AUDIT.clear()


def ladder_report() -> Dict[str, Any]:
    """Evidence that no capability implies any other.

    For each capability, grant it alone and record what every other capability
    answers.  Anything other than the granted one answering True would be an
    implicit escalation.
    """
    rows: Dict[str, Dict[str, bool]] = {}
    for capability in CAPABILITY_LADDER:
        with granted(capability, reason="ladder isolation probe"):
            rows[capability] = {other: is_granted(other) for other in CAPABILITY_LADDER}
    escalations = [
        {"granted": cap, "also_allowed": other}
        for cap, answers in rows.items()
        for other, allowed in answers.items()
        if allowed and other != cap
    ]
    return {"matrix": rows, "implicit_escalations": escalations,
            "no_implicit_escalation": not escalations,
            "ladder": list(CAPABILITY_LADDER)}


def protected(capability: str, site: str):
    """Decorator form for entrypoints whose whole body is a mutation."""
    def wrap(fn):
        def inner(*args: Any, **kwargs: Any):
            require_capability(capability, site=site)
            return fn(*args, **kwargs)
        inner.__name__ = getattr(fn, "__name__", "inner")
        inner.__doc__ = getattr(fn, "__doc__", None)
        inner.__wrapped__ = fn
        inner.__protected_capability__ = capability
        inner.__protected_site__ = site
        return inner
    return wrap


# --- state immutability evidence -------------------------------------------------

def state_digest(target: Any, *, fields: Sequence[str] = ()) -> str:
    """A digest of whatever mutable state an object exposes.

    Used to prove that a denied call changed nothing.  It walks the object's own
    attribute dictionary rather than a hand-listed subset, so a mutation the test
    author did not anticipate still shows up.
    """
    import hashlib
    import json

    def encode(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): encode(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, set):
            return sorted(str(v) for v in value)
        if hasattr(value, "__dict__"):
            return {k: encode(v) for k, v in sorted(vars(value).items()) if not k.startswith("__")}
        return repr(value)

    if fields:
        payload = {f: encode(getattr(target, f, None)) for f in fields}
    else:
        payload = {k: encode(v) for k, v in sorted(vars(target).items()) if not k.startswith("__")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=repr).encode("utf-8")).hexdigest()
