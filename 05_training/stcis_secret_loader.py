#!/usr/bin/env python3
"""Runtime-only loader for the STCIS Open API service key (H4M-AE-R8.7).

The key is read into process memory and never returned in any printable audit
payload.  There is deliberately no accessor that exposes the key's length, a
hash, a prefix or any fragment: the only thing this module will tell a caller
about the key is whether one was found.

The secret source directory is git-excluded and must never be committed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SECRET_DIR = PROJECT_ROOT / "stcis api"
_TOKEN = re.compile(r"[A-Za-z0-9%+/=_-]{30,}")


class SecretUnavailable(RuntimeError):
    pass


def secret_source_present() -> bool:
    return SECRET_DIR.is_dir() and any(SECRET_DIR.glob("*.md"))


def load_service_key() -> str:
    """Return the service key.  Callers must never log or persist the result."""
    if not secret_source_present():
        raise SecretUnavailable("STCIS secret source directory or markdown file is absent")
    source = sorted(SECRET_DIR.glob("*.md"))[0]
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("[") or "stcis.go.kr" in stripped:
            continue  # link lines carry the endpoint, not the standalone key
        matches = _TOKEN.findall(stripped)
        if matches:
            # a non-link line carries the key, possibly behind a short label
            return max(matches, key=len)
    raise SecretUnavailable("no standalone service-key token found in the secret source")


def endpoint_hint() -> Optional[str]:
    """The non-secret endpoint recorded alongside the key, with any query string dropped."""
    if not secret_source_present():
        return None
    source = sorted(SECRET_DIR.glob("*.md"))[0]
    for line in source.read_text(encoding="utf-8", errors="ignore").splitlines():
        found = re.search(r"https?://[^\s\)\]]+", line)
        if found:
            return found.group(0).split("?")[0]
    return None


def handling_audit() -> dict:
    """Structural, key-free description safe to persist in an artifact."""
    return {
        "secret_source_dir": str(SECRET_DIR.relative_to(PROJECT_ROOT)),
        "secret_source_present": secret_source_present(),
        "secret_source_git_excluded": True,
        "key_loaded_into_process_memory_only": True,
        "key_value_returned_to_artifact": False,
        "key_hash_recorded": False,
        "key_length_recorded": False,
        "key_fragment_recorded": False,
        "key_bearing_url_recorded": False,
        "environment_dumped": False,
        "endpoint_recorded_without_query_string": endpoint_hint(),
    }
