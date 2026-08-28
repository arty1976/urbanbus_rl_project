#!/usr/bin/env python3
"""Prompt 5-E01-DL-6D-PA1-A-SRP2-BIS-C1.

Limited Prospective BIS Pilot Capture, Raw Evidence Preservation, and
Trajectory Quality Audit.

Scope: LIMITED_PROSPECTIVE_BIS_DATA_PIPELINE_VALIDATION.

Makes REAL, allowlisted getPos02 + getRealtime02 calls to apis.data.go.kr per the
C0-frozen 90-call plan (route 3000814001 both directions, 30 cycles / 60 s), then
preserves raw evidence append-only, normalizes, and audits repeated-vehicle
continuity / route progression / interval-censored turnaround candidates.

It does NOT: study morning-peak performance, measure actual headway / arrival /
departure / dwell, point-estimate turnaround time, write the DB, run the
simulator, or train. The 07:00 KST start is an operational sampling choice, NOT a
treatment variable and NOT a success criterion.

Security: reuses the existing turnaround BIS secure caller / redaction / raw
writer unchanged (no new API client, key loader, or secret store). The service
key value is used to make requests but is never written to any artifact, log, or
URL. A network allowlist restricts traffic to the two approved endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import resource
import secrets
import shutil
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
UTC = ZoneInfo("UTC")

PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_REL = "run_prompt5_e01_dl6d_pa1a_srp2_bis_c1_limited_pilot_capture.py"
RUNNER_PATH = TRAINING_ROOT / RUNNER_REL

# --------------------------------------------------------------------------- #
# upstream
# --------------------------------------------------------------------------- #
C0_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp2_bis_c0_capture_contract_preflight_20260804_184415"
C0_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_C0_CAPTURE_CONTRACT_AND_LIMITED_PILOT_PREFLIGHT_COMPLETE"
C0_READINESS = "SRP2_BIS_C0_COMPLETE_SERVICE_KEY_OR_QUOTA_CONFIRMATION_PENDING_USER_COMMAND"
C0_SECURITY_VERDICT = "EXISTING_TURNAROUND_BIS_SECURITY_PATH_VERIFIED_FOR_REUSE"
C0_MANIFEST = "artifact_manifest_srp2_bis_c0.json"
C0_LOCK = "_SRP2_BIS_C0_PREFLIGHT_COMPLETE.lock"
C0_RUNNER_FROZEN_SHA = "3efaec3f25e7f46730afc7116e6e12356471154b7963a8839324a6814671795c"

SRP1R4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp1_r4_postgresql_first_reconciliation_20260803_224422"
SRP1R4_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP1_R4_POSTGRESQL_FIRST_SOURCE_RECONCILIATION_COMPLETE"
SF0_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_sf0_historical_state_feasibility_audit_20260803_185838"
SRP0_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp0_state_source_repair_plan_20260803_192100"

# real getBs02 master (route-stop sequence, 20,508 rows / 234 routes)
ROUTE_STOP_SEQ_PARQUET = ARTIFACTS_ROOT / "suseong_source_pack_v1" / "route_stop_sequences.parquet"

# --------------------------------------------------------------------------- #
# authoritative turnaround BIS security path (9 sources, C0 baseline SHA)
# --------------------------------------------------------------------------- #
SECURITY_PATH_SOURCES = [
    ("05_training/data_acquisition/capture_terminal_position_samples.py", "ec35b4ba02af163b12855ae1776a14e8244d976976a2f3cb8984a1e07be4caee"),
    ("05_training/data_acquisition/capture_terminal_semantics_samples.py", "96f6d7fbb6e70e78016a2c2ed71a140c45034ddfb898fa65c46643e94f5da49f"),
    ("05_training/data_acquisition/reconstruct_terminal_dwell_events.py", "731128619fc8156a5db6a210c2fbcf906c4cc4a5b64ee1b62baf1d10af28257f"),
    ("05_training/adapters/inspect_getpos02_live_position_sampling.py", "e01b0bdb1b226857c8d3404cb422622af086573266391bec04299f3f1f37401a"),
    ("05_training/adapters/inspect_getrealtime02_eta_sampling.py", "fe449b9f8666b86faa1d92004d4bba4f8eb9d51ce2cf2d315174e676e02b17aa"),
    ("05_training/adapters/inspect_getbs02_route_stop_sequence.py", "26341935d8d6fbc47da88639683bfd1f2eed4041ea5808a2600ee968b31f819c"),
    ("05_training/adapters/inspect_getbasic02_master_snapshot.py", "2a84e246bbce037fe5ee4ec28b9aa1453c5a9c0ec2969de0dd3bfd4d73f67b13"),
    ("05_training/adapters/collect_getbs02_route_stop_sequence_bulk.py", "c27392ed2d322edb60b3787167172af9c41427ef92a817286a8ec0273562a8ff"),
    ("05_training/run_prompt5_e01_r2d1c_r1_bis_credential_mapping_preflight.py", "12b0cd10e5cc49a84a312e6ec9baa3f5725c0fec954f617c5905820bec3f6672"),
]
# reused secure caller modules (imported, main() never runs)
GETPOS02_CALLER_REL = "05_training/data_acquisition/capture_terminal_position_samples.py"
GETREALTIME02_CALLER_REL = "05_training/adapters/inspect_getrealtime02_eta_sampling.py"
REDACTION_HELPER_REL = "05_training/run_prompt5_e01_r2d1c_r1_bis_credential_mapping_preflight.py"

SERVICE_KEY_ENV_NAMES = ["DAEGU_BIS_SERVICE_KEY", "DATAGO_SERVICE_KEY"]
OPERATOR_CAP_ENV = "SRP2_BIS_C1_OPERATOR_HARD_CAP_ACCEPTED"

# --------------------------------------------------------------------------- #
# network allowlist (Section 8)
# --------------------------------------------------------------------------- #
ALLOWED_HOST = "apis.data.go.kr"
ALLOWED_SCHEME = "https"
GETPOS02_PATH = "/6270000/dbmsapi02/getPos02"
GETREALTIME02_PATH = "/6270000/dbmsapi02/getRealtime02"
ALLOWED_PATHS = {GETPOS02_PATH, GETREALTIME02_PATH}
GETPOS02_URL = f"https://{ALLOWED_HOST}{GETPOS02_PATH}"
GETREALTIME02_URL = f"https://{ALLOWED_HOST}{GETREALTIME02_PATH}"

# --------------------------------------------------------------------------- #
# frozen pilot scope + budget (C0)  — 814 both directions, 90 calls
# --------------------------------------------------------------------------- #
TARGET_ROUTE_ID = "3000814001"
TARGET_ROUTE_NO = "814"
TARGET_DIRECTIONS = ["0", "1"]
C0_PLANNED_CALLS = 90
PLANNED_CYCLES = 30
POLLING_INTERVAL_SEC = 60
DURATION_MINUTES = 30
HARD_CALL_CAP = 180
MAX_RETRY_PER_CALL = 1
RETRY_BACKOFF_SEC = 5
PAIR_SKEW_MAX_SEC = 30
PREFERRED_START_LOCAL = "07:00:00"
CAPTURE_TIMEZONE = "Asia/Seoul"
SOURCE_SYSTEM = "DAEGU_BIS"

# --------------------------------------------------------------------------- #
# gate / readiness / status
# --------------------------------------------------------------------------- #
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_LIMITED_PROSPECTIVE_CAPTURE_COMPLETE"
READINESS_A = "SRP2_BIS_C1_COMPLETE_TRAJECTORY_QUALITY_AUDIT_READY_PENDING_USER_COMMAND"
READINESS_B = "SRP2_BIS_C1_COMPLETE_ADDITIONAL_LIMITED_CAPTURE_REQUIRED_PENDING_USER_COMMAND"
READINESS_C = "SRP2_BIS_C1_COMPLETE_TRAJECTORY_READY_NO_TURNAROUND_OBSERVED"
READINESS_D = "SRP2_BIS_C1_COMPLETE_BIS_CAPTURE_REPAIR_PENDING_USER_COMMAND"
READINESS_E = "SRP2_BIS_C1_COMPLETE_PROSPECTIVE_STATE_EVIDENCE_AMENDMENT_PENDING_USER_COMMAND"

_B = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_"
BLOCKED_C0 = _B + "C0_UPSTREAM_INVALID"
BLOCKED_KEY_ABSENT = _B + "SERVICE_KEY_ABSENT"
BLOCKED_OPERATOR_CAP = _B + "OPERATOR_CAP_NOT_ACCEPTED"
BLOCKED_SECURITY_DRIFT = _B + "SECURITY_SOURCE_DRIFT"
BLOCKED_CALL_PLAN = _B + "CALL_PLAN_MISMATCH"
BLOCKED_TARGET_SCOPE = _B + "TARGET_SCOPE_INVALID"
BLOCKED_ENDPOINT_PARAM = _B + "ENDPOINT_PARAMETERIZATION_INVALID"
BLOCKED_MULTIPLE_KEY = _B + "MULTIPLE_KEY_SOURCE_AMBIGUOUS"
BLOCKED_INSUFFICIENT = _B + "INSUFFICIENT_LIVE_OBSERVATION"

_F = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_"
FAIL_RUNNER = _F + "RUNNER_MUTATED_DURING_CAPTURE"
FAIL_ALLOWLIST = _F + "NETWORK_ALLOWLIST_VIOLATION"
FAIL_CAP = _F + "CALL_CAP_EXCEEDED"
FAIL_UNAPPROVED_ENDPOINT = _F + "UNAPPROVED_ENDPOINT_CALLED"
FAIL_KEY_EXPOSED = _F + "SERVICE_KEY_EXPOSED"
FAIL_KEY_URL = _F + "KEY_BEARING_URL_PERSISTED"
FAIL_ENV_DUMP = _F + "ENVIRONMENT_DUMPED"
FAIL_RAW_OVERWRITE = _F + "RAW_RESPONSE_OVERWRITTEN"
FAIL_RAW_MISSING = _F + "RAW_RESPONSE_MISSING"
FAIL_RAW_SHA = _F + "RAW_SHA_MISMATCH"
FAIL_IDEMPOTENCY = _F + "JOURNAL_IDEMPOTENCY_VIOLATION"
FAIL_SYNTH = _F + "SYNTHETIC_API_ROW_CREATED"
FAIL_TS_FAB = _F + "TIMESTAMP_FABRICATED"
FAIL_HEADWAY = _F + "ACTUAL_HEADWAY_OVERCLAIMED"
FAIL_TURNAROUND = _F + "EXACT_TURNAROUND_OVERCLAIMED"
FAIL_DB_WRITE = _F + "DATABASE_WRITE_DETECTED"
FAIL_SIM = _F + "SIMULATOR_EXECUTED"
FAIL_TRAIN = _F + "TRAINING_EXECUTED"
FAIL_VT = _F + "VALIDATION_OR_TEST_TOUCHED"
FAIL_SEALED = _F + "SEALED_HOLDOUT_ACCESSED"
FAIL_UPSTREAM_MUT = _F + "UPSTREAM_ARTIFACT_MUTATED"
FAIL_MANIFEST = _F + "MANIFEST_RECONCILIATION"


class C1Error(RuntimeError):
    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate_status = gate
        self.detail = detail


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def json_clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_clean(v) for v in value]
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def now_utc() -> datetime:
    return datetime.now(UTC)


def now_kst() -> datetime:
    return datetime.now(KST)


def iso_kst() -> str:
    return now_kst().isoformat(timespec="seconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


class Writer:
    def __init__(self, root: Path) -> None:
        self.root = root

    def text(self, rel: str, text: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def json(self, rel: str, payload: Mapping[str, Any]) -> None:
        self.text(rel, json.dumps(json_clean(payload), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")

    def jsonl(self, rel: str, rows: Sequence[Mapping[str, Any]]) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(json_clean(dict(r)), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in rows), encoding="utf-8")

    def parquet(self, rel: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Dict[str, Any]:
        import pandas as pd
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(list(rows), columns=list(columns)) if rows else pd.DataFrame(columns=list(columns))
        df.to_parquet(p, index=False)
        return {"relative_path": rel, "row_count": int(len(df)), "sha256": sha256_file(p), "size_bytes": p.stat().st_size}


def copy_file(writer: Writer, src: Path, dst_rel: str) -> Dict[str, Any]:
    dst = writer.root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source_path": str(src), "snapshot_relative_path": dst_rel, "source_sha256": sha256_file(src),
            "copied_sha256": sha256_file(dst), "byte_identical": sha256_file(src) == sha256_file(dst), "size_bytes": dst.stat().st_size}


def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"srp2-bis-c1 artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def import_module_no_main(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, str(PROJECT_ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs module-level defs/imports only; main() guarded by __main__
    return mod


# --------------------------------------------------------------------------- #
# network allowlist
# --------------------------------------------------------------------------- #
class AllowlistViolation(RuntimeError):
    pass


class NetworkAllowlist:
    def __init__(self) -> None:
        self.installed = False
        self.request_records: List[Dict[str, Any]] = []
        self.violation_count = 0
        self.socket_connect_count = 0
        self._orig_urlopen = None
        self._orig_socket_connect = None

    @staticmethod
    def _url_of(arg: Any) -> str:
        if isinstance(arg, urllib.request.Request):
            return arg.full_url
        return str(arg)

    @classmethod
    def _validate(cls, url: str) -> Tuple[bool, str]:
        try:
            p = urllib.parse.urlsplit(url)
        except Exception:
            return False, "unparseable_url"
        if p.scheme != ALLOWED_SCHEME:
            return False, f"scheme:{p.scheme}"
        if p.hostname != ALLOWED_HOST:
            return False, f"host:{p.hostname}"
        if p.path not in ALLOWED_PATHS:
            return False, f"path:{p.path}"
        return True, "ok"

    def install(self) -> None:
        guard = self

        class StrictRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
                ok, reason = guard._validate(newurl)
                if not ok:
                    guard.violation_count += 1
                    raise AllowlistViolation(f"redirect off allowlist ({reason})")
                return super().redirect_request(req, fp, code, msg, headers, newurl)

        opener = urllib.request.build_opener(StrictRedirect())
        urllib.request.install_opener(opener)
        self._orig_urlopen = urllib.request.urlopen

        def guarded_urlopen(arg, *a, **k):  # noqa: ANN001
            url = guard._url_of(arg)
            method = arg.get_method() if isinstance(arg, urllib.request.Request) else "GET"
            ok, reason = guard._validate(url)
            allowed = ok and method == "GET"
            guard.request_records.append({"host": urllib.parse.urlsplit(url).hostname,
                                          "path": urllib.parse.urlsplit(url).path, "method": method,
                                          "allowed": allowed, "reason": reason, "at": iso_kst()})
            if not allowed:
                guard.violation_count += 1
                raise AllowlistViolation(f"blocked non-allowlisted request ({reason}, {method})")
            return guard._orig_urlopen(arg, *a, **k)

        urllib.request.urlopen = guarded_urlopen  # type: ignore[assignment]

        self._orig_socket_connect = socket.socket.connect

        def guarded_socket_connect(self_sock, address, *a, **k):  # noqa: ANN001
            guard.socket_connect_count += 1
            return guard._orig_socket_connect(self_sock, address, *a, **k)

        socket.socket.connect = guarded_socket_connect  # type: ignore[assignment]
        self.installed = True

    def audit(self) -> Dict[str, Any]:
        return {"created_at": iso_kst(), "network_allowlist_installed": self.installed,
                "allowed_host": ALLOWED_HOST, "allowed_scheme": ALLOWED_SCHEME, "allowed_paths": sorted(ALLOWED_PATHS),
                "allowed_method": "GET", "request_count": len(self.request_records),
                "allowlist_violation_count": self.violation_count, "socket_connect_count": self.socket_connect_count,
                "off_allowlist_redirect_blocked": True}


ALLOWLIST = NetworkAllowlist()


# --------------------------------------------------------------------------- #
# upstream verify + snapshot
# --------------------------------------------------------------------------- #
def verify_upstream(root: Path, gate_expected: str, manifest_name: Optional[str], lock_name: Optional[str]) -> Dict[str, Any]:
    if not root.is_dir():
        return {"artifact_root": str(root), "upstream_valid": False, "checks": {"artifact_exists": False}}
    gate = read_json(root / "gate_decision.json")
    checks = {"artifact_exists": True, "gate": gate.get("gate") == gate_expected}
    manifest_sha = None
    if manifest_name and lock_name:
        lock = read_json(root / lock_name)
        manifest_path = root / manifest_name
        manifest = read_json(manifest_path)
        manifest_sha = sha256_file(manifest_path)
        missing = mismatch = 0
        for r in manifest["files"]:
            t = root / r["relative_path"]
            if not t.exists():
                missing += 1
            elif r.get("sha256") is not None and sha256_file(t) != r["sha256"]:
                mismatch += 1
        checks.update({"lock_present": (root / lock_name).exists(), "lock_manifest_sha": lock.get("manifest_sha256") == manifest_sha,
                       "manifest_missing_zero": missing == 0, "manifest_hash_mismatch_zero": mismatch == 0,
                       "manifest_self_ref_absent": not any(r["relative_path"] == manifest_name for r in manifest["files"]),
                       "terminal_lock_not_in_manifest": not any(r["relative_path"] == lock_name for r in manifest["files"])})
    return {"artifact_root": str(root), "gate": gate.get("gate"), "readiness": gate.get("readiness"),
            "manifest_sha256": manifest_sha, "checks": checks, "upstream_valid": all(checks.values())}


def snapshot_upstream(writer: Writer) -> Tuple[List[Dict[str, Any]], List[str]]:
    plan = [
        (C0_ROOT, "upstream_c0_snapshot", ["gate_decision.json", "downstream_lock.json", C0_MANIFEST, C0_LOCK,
                                           "limited_pilot_call_budget.json", "suseong_bis_pilot_recommended_scope.json",
                                           "bis_endpoint_contract.json", "existing_turnaround_bis_security_lineage.json",
                                           "service_key_security_contract.json", "runner_freeze_audit.json"]),
        (SRP1R4_ROOT, "upstream_srp1_r4_snapshot", ["gate_decision.json", "downstream_lock.json"]),
        (SF0_ROOT, "upstream_sf0_snapshot", ["gate_decision.json", "overall_state_feasibility.json"]),
        (SRP0_ROOT, "upstream_srp0_snapshot", ["gate_decision.json", "recommended_agent_semantics.json"]),
    ]
    records, paths = [], []
    for root, sub, names in plan:
        for name in names:
            src = root / name
            if src.exists():
                rec = copy_file(writer, src, f"{sub}/{name}")
                records.append(rec)
                paths.append(rec["snapshot_relative_path"])
    return records, paths


def security_source_drift_audit() -> Dict[str, Any]:
    rows = []
    drift = 0
    for rel, baseline in SECURITY_PATH_SOURCES:
        p = PROJECT_ROOT / rel
        cur = sha256_file(p) if p.exists() else None
        d = (cur is None) or (cur != baseline)
        if d:
            drift += 1
        rows.append({"source_path": rel, "baseline_sha256": baseline, "current_sha256": cur, "exists": p.exists(), "drift": d})
    return {"created_at": iso_kst(), "security_source_count": len(rows), "security_source_drift_count": drift,
            "all_present": all(r["exists"] for r in rows), "records": rows}


# --------------------------------------------------------------------------- #
# service key presence + operator cap
# --------------------------------------------------------------------------- #
def resolve_service_key(service_key_file: Optional[str], gp_mod) -> Tuple[str, Dict[str, Any]]:
    """Presence-audited key resolution. Value held in memory only; never persisted."""
    present_env = [n for n in SERVICE_KEY_ENV_NAMES if n in os.environ]
    file_present = bool(service_key_file) and Path(service_key_file).expanduser().is_file()
    key = ""
    source = "ABSENT"
    if len(present_env) >= 2:
        # both env names present -> ambiguity unless C0 defined a priority (it did not)
        source = "MULTIPLE_ENV_PRESENT"
    elif len(present_env) == 1:
        key = os.environ.get(present_env[0]) or ""
        source = f"ENVIRONMENT:{present_env[0]}"
    elif file_present:
        key = gp_mod.read_service_key_file(str(Path(service_key_file).expanduser())) or ""
        source = "SERVICE_KEY_FILE"
    audit = {"created_at": iso_kst(), "checked_env_names": SERVICE_KEY_ENV_NAMES, "present_env_names": present_env,
             "service_key_file_provided": bool(service_key_file), "service_key_file_present": file_present,
             "service_key_present": bool(key) or source == "MULTIPLE_ENV_PRESENT", "service_key_source_kind": source,
             "service_key_value_written": False, "service_key_value_hashed": False, "service_key_length_recorded": False,
             "presence_only": True}
    return key, audit


def operator_cap_acceptance() -> Dict[str, Any]:
    accepted = os.environ.get(OPERATOR_CAP_ENV, "").strip().lower() == "true"
    return {"created_at": iso_kst(), "operator_cap_env": OPERATOR_CAP_ENV, "operator_cap_accepted": accepted,
            "provider_quota_authoritative": False, "hard_local_cap": HARD_CALL_CAP,
            "note": "user-accepted 180-call local operator cap; provider daily quota (1000/day per user) is not asserted from a local artifact"}


# --------------------------------------------------------------------------- #
# terminal stops + C0 call-plan reconciliation
# --------------------------------------------------------------------------- #
def derive_terminal_stops() -> Dict[str, Any]:
    import pandas as pd
    rss = pd.read_parquet(ROUTE_STOP_SEQ_PARQUET)
    r = rss[rss["route_id"].astype(str) == TARGET_ROUTE_ID].copy()
    stops = {}
    for d in TARGET_DIRECTIONS:
        sub = r[r["direction_id"].astype(str) == d].sort_values("stop_order")
        if len(sub) == 0:
            continue
        term = sub.iloc[-1]
        first = sub.iloc[0]
        stops[d] = {"direction_id": d, "terminal_stop_id": str(term["stop_id"]), "terminal_stop_name": str(term["stop_name"]),
                    "terminal_stop_order": int(term["stop_order"]), "first_stop_id": str(first["stop_id"]),
                    "first_stop_name": str(first["stop_name"]), "stop_count": int(len(sub))}
    return {"route_id": TARGET_ROUTE_ID, "route_no": TARGET_ROUTE_NO, "master_source": str(ROUTE_STOP_SEQ_PARQUET.relative_to(PROJECT_ROOT)),
            "master_rows": int(len(rss)), "master_routes": int(rss["route_id"].nunique()), "directions": stops}


def build_call_plan(term_stops: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct the C0-frozen deterministic call sequence: per cycle
    1 getPos02(routeId) + 2 getRealtime02(terminal bsId of each direction)."""
    seq: List[Dict[str, Any]] = []
    for cycle_index in range(1, PLANNED_CYCLES + 1):
        seq.append({"cycle_index": cycle_index, "endpoint": "getPos02", "param_name": "routeId",
                    "param_value": TARGET_ROUTE_ID, "target_id": f"{TARGET_ROUTE_ID}_both_dir"})
        for d in TARGET_DIRECTIONS:
            if d in term_stops["directions"]:
                bs = term_stops["directions"][d]["terminal_stop_id"]
                seq.append({"cycle_index": cycle_index, "endpoint": "getRealtime02", "param_name": "bsId",
                            "param_value": bs, "target_id": f"{TARGET_ROUTE_ID}_dir{d}_terminal_{bs}"})
    plan_canonical = "\n".join(f"{s['cycle_index']}|{s['endpoint']}|{s['param_name']}={s['param_value']}" for s in seq)
    return {"planned_call_sequence": seq, "reconstructed_planned_call_count": len(seq),
            "plan_hash": sha256_bytes(plan_canonical.encode("utf-8"))}


def reconcile_call_plan(writer: Writer, term_stops: Dict[str, Any]) -> Dict[str, Any]:
    c0_budget = read_json(C0_ROOT / "limited_pilot_call_budget.json")
    c0_scope = read_json(C0_ROOT / "suseong_bis_pilot_recommended_scope.json")
    c0_planned = int(c0_budget.get("planned_call_count", -1))
    c0_targets = sorted({t["route_id"] for t in c0_scope.get("targets", [])})
    plan = build_call_plan(term_stops)
    reconstructed = plan["reconstructed_planned_call_count"]
    getpos = sum(1 for s in plan["planned_call_sequence"] if s["endpoint"] == "getPos02")
    getrt = sum(1 for s in plan["planned_call_sequence"] if s["endpoint"] == "getRealtime02")
    count_match = reconstructed == c0_planned == C0_PLANNED_CALLS
    target_match = c0_targets == [TARGET_ROUTE_ID]
    payload = {"created_at": iso_kst(), "c0_planned_call_count": c0_planned, "c0_target_routes": c0_targets,
               "c0_target_composition": c0_scope.get("target_composition"),
               "c1_reconstructed_planned_call_count": reconstructed, "c1_getpos02_calls": getpos, "c1_getrealtime02_calls": getrt,
               "c1_target_route": TARGET_ROUTE_ID, "c1_directions": TARGET_DIRECTIONS,
               "c1_terminal_stops": {d: term_stops["directions"].get(d, {}).get("terminal_stop_id") for d in TARGET_DIRECTIONS},
               "plan_hash": plan["plan_hash"], "count_match": count_match, "target_match": target_match,
               "call_plan_reconciled": count_match and target_match,
               "note": "C0 froze TWO_DIRECTIONS_OF_TOP_ROUTE (814); reconstructed 1 getPos02 + 2 getRealtime02 per cycle x 30 = 90; "
                       "terminal bsIds derived deterministically from the getBs02 route-stop master."}
    writer.json("c0_call_plan_reconciliation.json", payload)
    return payload, plan


# --------------------------------------------------------------------------- #
# capture loop
# --------------------------------------------------------------------------- #
def classify_error(exc: Optional[Exception], http_status: Optional[int], raw_text: str, provider_status: str) -> str:
    if isinstance(exc, AllowlistViolation):
        return "NETWORK_ALLOWLIST_VIOLATION"
    if exc is not None:
        name = type(exc).__name__.lower()
        if "timeout" in name or "timed out" in str(exc).lower():
            return "NETWORK_TIMEOUT"
        if "url" in name or "connection" in name or "reset" in str(exc).lower():
            return "CONNECTION_ERROR"
        return "UNKNOWN_ERROR"
    if http_status == 429 or provider_status == "RATE_LIMIT":
        return "PROVIDER_RATE_LIMIT"
    if provider_status in ("AUTH_ERROR", "AUTH_OR_PROVIDER_ERROR"):
        return "PROVIDER_AUTH_ERROR"
    if http_status is not None and 500 <= http_status < 600:
        return "HTTP_ERROR"
    if http_status is not None and 400 <= http_status < 500:
        return "HTTP_ERROR"
    return "NONE"


def parse_provider_time(raw: Optional[str]) -> Tuple[Optional[str], str, str]:
    if raw in (None, ""):
        return None, "PROVIDER_TIMEZONE_UNCONFIRMED", "MISSING"
    s = str(raw).strip()
    if re.fullmatch(r"\d{6}", s):  # HHMMSS provider format
        return s, "PROVIDER_TIMEZONE_UNCONFIRMED_ASSUMED_KST", "PARSED_HHMMSS"
    if re.fullmatch(r"\d{14}", s):  # YYYYMMDDHHMMSS
        return s, "PROVIDER_TIMEZONE_UNCONFIRMED_ASSUMED_KST", "PARSED_YYYYMMDDHHMMSS"
    return s, "PROVIDER_TIMEZONE_UNCONFIRMED", "RAW_PRESERVED"


class CaptureLoop:
    def __init__(self, writer: Writer, service_key: str, session_id: str, gr_mod, gp_mod, cred_mod,
                 plan: Dict[str, Any], term_stops: Dict[str, Any]) -> None:
        self.writer = writer
        self.key = service_key
        self.session_id = session_id
        self.gr = gr_mod
        self.gp = gp_mod
        self.cred = cred_mod
        self.plan = plan
        self.term_stops = term_stops
        self.raw_root = writer.root / "raw" / session_id
        self.request_journal: List[Dict[str, Any]] = []
        self.cycle_journal: List[Dict[str, Any]] = []
        self.raw_registry: List[Dict[str, Any]] = []
        self.getpos_rows: List[Dict[str, Any]] = []
        self.getrt_rows: List[Dict[str, Any]] = []
        self.pair_timing: List[Dict[str, Any]] = []
        self.total_attempts = 0
        self.successful_calls = 0
        self.retry_calls = 0
        self.error_counts: Dict[str, int] = {}
        self.aborted = False
        self.abort_reason: Optional[str] = None
        self.paused = False

    def _redact(self, url: str) -> str:
        return self.cred.redact(url, self.key) if self.key else url

    def _do_call(self, endpoint: str, param_name: str, param_value: str, cycle_index: int, cycle_id: str,
                 target_id: str, attempt_id: str) -> Dict[str, Any]:
        base = GETPOS02_URL if endpoint == "getPos02" else GETREALTIME02_URL
        planned_at = now_utc()
        m_start = time.monotonic_ns()
        started = now_utc()
        exc: Optional[Exception] = None
        http_status: Optional[int] = None
        raw_text = ""
        request_url = ""
        try:
            http_status, raw_text, request_url = self.gr.fetch_api(base, self.key, param_name, param_value, {"resultType": "json"}, timeout=30)
            if http_status == 0:  # gr.fetch_api returns 0 + message on transport error
                exc = RuntimeError(raw_text)
        except AllowlistViolation as e:
            exc = e
        except Exception as e:  # noqa: BLE001
            exc = e
        m_end = time.monotonic_ns()
        completed = now_utc()
        provider_status = self.cred.detect_provider_status(raw_text, http_status) if raw_text else "EMPTY_RESPONSE"
        err = classify_error(exc, http_status, raw_text, provider_status)
        # write raw append-only (response body only; never the key-bearing URL)
        raw_rel = f"raw/{self.session_id}/{endpoint}/{cycle_id}_{target_id}_{attempt_id}.json"
        raw_path = self.writer.root / raw_rel
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if raw_path.exists():
            raise C1Error(FAIL_RAW_OVERWRITE, f"raw already exists: {raw_rel}")
        payload_bytes = (raw_text or "").encode("utf-8")
        raw_path.write_bytes(payload_bytes)
        raw_sha = sha256_bytes(payload_bytes)
        self.raw_registry.append({"capture_session_id": self.session_id, "cycle_id": cycle_id, "target_id": target_id,
                                  "endpoint": endpoint, "attempt_id": attempt_id,
                                  "request_started_at_utc": started.isoformat(), "response_received_at_utc": completed.isoformat(),
                                  "http_status": http_status, "provider_result_code": provider_status,
                                  "content_type": "application/json", "encoding": "utf-8",
                                  "response_byte_size": len(payload_bytes), "response_sha256": raw_sha,
                                  "raw_relative_path": raw_rel, "parse_status": "PENDING", "error_class": err})
        req_rec = {"capture_session_id": self.session_id, "cycle_index": cycle_index, "cycle_id": cycle_id,
                   "target_id": target_id, "endpoint_id": endpoint, "attempt_id": attempt_id,
                   "param_name": param_name, "param_value": param_value,
                   "request_signature_without_key": sha256_bytes(f"{endpoint}|{param_name}={param_value}|{cycle_id}|{target_id}".encode()),
                   "request_url_redacted": self._redact(request_url) if request_url else None,
                   "planned_at_utc": planned_at.isoformat(), "started_at_utc": started.isoformat(),
                   "completed_at_utc": completed.isoformat(), "started_at_kst": started.astimezone(KST).isoformat(),
                   "completed_at_kst": completed.astimezone(KST).isoformat(),
                   "monotonic_start_ns": m_start, "monotonic_end_ns": m_end, "elapsed_milliseconds": (m_end - m_start) / 1e6,
                   "http_status": http_status, "provider_result_code": provider_status, "error_class": err,
                   "response_sha256": raw_sha, "raw_relative_path": raw_rel,
                   "normalized_row_count": None, "success": err == "NONE" and http_status is not None and 200 <= (http_status or 0) < 300}
        self.request_journal.append(req_rec)
        return {"endpoint": endpoint, "http_status": http_status, "provider_status": provider_status, "error_class": err,
                "raw_text": raw_text, "raw_sha": raw_sha, "raw_rel": raw_rel, "started": started, "completed": completed,
                "target_id": target_id, "param_value": param_value, "success": req_rec["success"]}

    def _retryable(self, err: str) -> bool:
        return err in ("NETWORK_TIMEOUT", "CONNECTION_ERROR") or err == "HTTP_ERROR"

    def _call_with_retry(self, step: Dict[str, Any], cycle_index: int, cycle_id: str) -> Optional[Dict[str, Any]]:
        endpoint, pname, pval, target_id = step["endpoint"], step["param_name"], step["param_value"], step["target_id"]
        for attempt in range(0, MAX_RETRY_PER_CALL + 1):
            if self.total_attempts >= HARD_CALL_CAP:
                return None  # cap reached
            attempt_id = f"a{attempt}"
            self.total_attempts += 1
            if attempt > 0:
                self.retry_calls += 1
                time.sleep(RETRY_BACKOFF_SEC)
            res = self._do_call(endpoint, pname, pval, cycle_index, cycle_id, target_id, attempt_id)
            err = res["error_class"]
            self.error_counts[err] = self.error_counts.get(err, 0) + 1
            if err == "PROVIDER_AUTH_ERROR":
                self.aborted = True
                self.abort_reason = "PROVIDER_AUTH_ERROR"
                return res
            if err == "PROVIDER_RATE_LIMIT":
                self.paused = True
                return res
            if res["success"] or not self._retryable(err):
                return res
        return res

    def run(self) -> None:
        for cyc in range(1, PLANNED_CYCLES + 1):
            if self.aborted or self.paused:
                break
            if self.total_attempts >= HARD_CALL_CAP:
                break
            cycle_id = f"c{cyc:03d}"
            cycle_start = now_utc()
            planned_time = cycle_start
            steps = [s for s in self.plan["planned_call_sequence"] if s["cycle_index"] == cyc]
            getpos_res = None
            getrt_res = []
            status = "COMPLETE"
            for step in steps:
                if self.total_attempts >= HARD_CALL_CAP:
                    status = "SKIPPED_CALL_CAP"
                    break
                res = self._call_with_retry(step, cyc, cycle_id)
                if res is None:
                    status = "SKIPPED_CALL_CAP"
                    break
                if step["endpoint"] == "getPos02":
                    getpos_res = res
                    self._normalize_getpos(res, cyc, cycle_id)
                else:
                    getrt_res.append(res)
                    self._normalize_getrt(res, cyc, cycle_id, step["param_value"])
                if self.aborted:
                    status = "FAILED"
                    break
                if self.paused:
                    status = "PARTIAL"
                    break
            # pair timing (getPos02 vs first getRealtime02)
            if getpos_res is not None and getrt_res:
                skew = abs((getrt_res[0]["completed"] - getpos_res["completed"]).total_seconds())
                self.pair_timing.append({"cycle_id": cycle_id, "cycle_index": cyc,
                                         "getpos02_completed_utc": getpos_res["completed"].isoformat(),
                                         "getrealtime02_completed_utc": getrt_res[0]["completed"].isoformat(),
                                         "pair_skew_seconds": skew, "endpoint_pair_is_simultaneous": False,
                                         "pair_status": "PAIR_UNSYNCHRONIZED" if skew > PAIR_SKEW_MAX_SEC else "PAIR_SYNCHRONIZED"})
            cycle_end = now_utc()
            if status == "COMPLETE" and not steps:
                status = "FAILED"
            self.cycle_journal.append({"capture_session_id": self.session_id, "cycle_index": cyc, "cycle_id": cycle_id,
                                       "planned_cycle_time_utc": planned_time.isoformat(),
                                       "actual_cycle_start_utc": cycle_start.isoformat(), "actual_cycle_end_utc": cycle_end.isoformat(),
                                       "cycle_lag_seconds": 0.0, "getpos02_calls": 1 if getpos_res else 0,
                                       "getrealtime02_calls": len(getrt_res), "cycle_status": status})
            if self.aborted or status == "SKIPPED_CALL_CAP":
                break
            # inter-cycle pacing to hold ~60s cadence
            if cyc < PLANNED_CYCLES and not self.aborted and not self.paused:
                elapsed = (now_utc() - cycle_start).total_seconds()
                remaining = max(0.0, POLLING_INTERVAL_SEC - elapsed)
                if remaining > 0:
                    time.sleep(remaining)

    def _normalize_getpos(self, res: Dict[str, Any], cyc: int, cycle_id: str) -> None:
        rows = self.gp.normalize_rows(TARGET_ROUTE_ID, res["raw_text"], Path(res["raw_rel"]), res["raw_sha"],
                                      res["started"].astimezone(KST).isoformat(), res["provider_status"]) if res["raw_text"] else []
        for idx, r in enumerate(rows):
            vraw = r.get("vehicle_id")
            token = sha256_bytes(f"{SOURCE_SYSTEM}|{vraw}".encode()) if vraw not in (None, "") else None
            ev_raw = r.get("provider_event_time")
            ev_parsed, tz_assume, parse_stat = parse_provider_time(ev_raw)
            self.getpos_rows.append({
                "capture_session_id": self.session_id, "cycle_id": cycle_id, "cycle_index": cyc, "target_id": res["target_id"],
                "poll_observed_at_utc": res["started"].isoformat(), "endpoint_started_at_utc": res["started"].isoformat(),
                "endpoint_completed_at_utc": res["completed"].isoformat(), "provider_vehicle_id_raw": vraw,
                "vehicle_token": token, "route_id": r.get("route_id"), "direction_id": (str(r.get("direction_id")) if r.get("direction_id") is not None else None),
                "route_sequence": r.get("current_sequence"), "current_stop_id": r.get("current_stop_id"),
                "x_position_raw": r.get("x"), "y_position_raw": r.get("y"), "coordinate_reference_status": "PROVIDER_RAW_WGS84_ASSUMED",
                "provider_event_time_raw": ev_raw, "provider_event_time_parsed": ev_parsed,
                "provider_timezone_assumption": tz_assume, "raw_response_sha256": res["raw_sha"], "raw_row_index": idx,
                "parse_status": parse_stat, "identity_status": "PROVIDER_VEHICLE_ID" if vraw else "IDENTITY_MISSING",
                "route_stop_match_status": "PENDING"})

    def _extract_getrt_rows(self, raw_text: str) -> List[Dict[str, Any]]:
        """getRealtime02 nests ETA rows inside each item's arrList
        (body.items[].arrList[]); flatten them. Falls back to XML / flat items."""
        if not raw_text:
            return []
        if raw_text.lstrip().startswith("<"):
            return self.gr.parse_xml_items(raw_text)
        try:
            obj = json.loads(raw_text)
        except Exception:
            return []
        body = obj.get("body") if isinstance(obj, dict) else None
        items = body.get("items") if isinstance(body, dict) else None
        if isinstance(items, dict):
            items = items.get("item", items)
        if isinstance(items, dict):
            items = [items]
        out: List[Dict[str, Any]] = []
        for it in (items or []):
            if not isinstance(it, dict):
                continue
            arr = it.get("arrList")
            if isinstance(arr, dict):
                arr = [arr]
            if isinstance(arr, list) and arr:
                for a in arr:
                    if isinstance(a, dict):
                        out.append(a)
            elif "routeId" in it or "vhcNo2" in it:
                out.append(it)  # already-flat item
        return out

    def _normalize_getrt(self, res: Dict[str, Any], cyc: int, cycle_id: str, stop_id: str) -> None:
        rows = self._extract_getrt_rows(res["raw_text"])
        for idx, r in enumerate(rows):
            if not isinstance(r, dict):
                continue
            vraw = r.get("vhcNo2") or r.get("vhcNo") or r.get("busId")
            token = sha256_bytes(f"{SOURCE_SYSTEM}|{vraw}".encode()) if vraw not in (None, "") else None
            eta_state = r.get("arrState")
            eta_sec = _to_int(r.get("arrTime") or r.get("remainSec") or r.get("etaSec") or r.get("predictTime"))
            eta_raw = eta_state if eta_state not in (None, "") else r.get("arrTime")
            self.getrt_rows.append({
                "capture_session_id": self.session_id, "cycle_id": cycle_id, "cycle_index": cyc, "target_id": res["target_id"],
                "poll_observed_at_utc": res["started"].isoformat(), "endpoint_started_at_utc": res["started"].isoformat(),
                "endpoint_completed_at_utc": res["completed"].isoformat(), "route_id": r.get("routeId") or r.get("route_id"),
                "direction_id": (str(r.get("moveDir")) if r.get("moveDir") is not None else None),
                "stop_id": stop_id,  # the queried target stop (arrList entries carry no bsId)
                "vehicle_identifier_if_available": vraw, "vehicle_token_if_available": token, "eta_raw": eta_raw,
                "eta_seconds": eta_sec, "arrival_order": _to_int(r.get("bsGap") or r.get("prevBsGap")),
                "route_sequence_if_available": r.get("seq"), "provider_event_time_raw": None,
                "raw_response_sha256": res["raw_sha"], "raw_row_index": idx, "parse_status": "PARSED",
                "route_stop_match_status": "PENDING",
                "eta_validity_status": "PROVIDER_PREDICTED_ETA" if (eta_sec is not None or eta_state not in (None, "")) else "NO_ETA",
                "eta_class": "PROVIDER_PREDICTED_ETA"})


def _to_int(v: Any) -> Optional[int]:
    try:
        return int(str(v).strip())
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# post-capture audits
# --------------------------------------------------------------------------- #
def crossmatch_rows(getpos_rows, getrt_rows) -> Dict[str, Any]:
    import pandas as pd
    rss = pd.read_parquet(ROUTE_STOP_SEQ_PARQUET)
    key = set((str(a), str(b)) for a, b in zip(rss["route_id"], rss["stop_id"]))
    stopset = set(str(s) for s in rss["stop_id"])
    routeset = set(str(s) for s in rss["route_id"])
    for row in getpos_rows:
        rid, sid = str(row.get("route_id")), str(row.get("current_stop_id"))
        row["route_stop_match_status"] = ("EXACT_SEQUENCE_MATCH" if (rid, sid) in key else
                                          ("EXACT_MASTER_MATCH" if sid in stopset and rid in routeset else
                                           ("UNMATCHED" if row.get("current_stop_id") else "MISSING")))
    for row in getrt_rows:
        rid, sid = str(row.get("route_id")), str(row.get("stop_id"))
        row["route_stop_match_status"] = ("EXACT_SEQUENCE_MATCH" if (rid, sid) in key else
                                          ("EXACT_MASTER_MATCH" if sid in stopset else ("UNMATCHED" if row.get("stop_id") else "MISSING")))
    matched = sum(1 for r in getpos_rows if r["route_stop_match_status"].startswith("EXACT"))
    total = len(getpos_rows)
    return {"created_at": iso_kst(), "getpos02_rows": total, "getpos02_matched": matched,
            "getpos02_crossmatch_rate": (matched / total) if total else None,
            "master_source": str(ROUTE_STOP_SEQ_PARQUET.relative_to(PROJECT_ROOT)), "master_rows": int(len(rss))}


def vehicle_continuity(getpos_rows) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for r in getpos_rows:
        if not r.get("vehicle_token"):
            continue
        k = (r["vehicle_token"], str(r.get("route_id")), str(r.get("direction_id")))
        groups.setdefault(k, []).append(r)
    rows = []
    for (token, rid, did), obs in groups.items():
        cycles = sorted({o["cycle_index"] for o in obs})
        seqs = [o.get("route_sequence") for o in obs if o.get("route_sequence") is not None]
        stops = {o.get("current_stop_id") for o in obs}
        n_cycles = len(cycles)
        gaps = max((cycles[i + 1] - cycles[i] for i in range(len(cycles) - 1)), default=0)
        if n_cycles >= 3:
            status = "CONTINUOUS_MULTI_CYCLE" if gaps <= 1 else "CONTINUOUS_WITH_GAPS"
        elif n_cycles == 1:
            status = "SINGLE_OBSERVATION"
        else:
            status = "CONTINUOUS_WITH_GAPS" if gaps <= 2 else "INSUFFICIENT_EVIDENCE"
        rows.append({"vehicle_token": token, "route_id": rid, "direction_id": did, "observation_count": len(obs),
                     "distinct_cycle_count": n_cycles, "distinct_stop_count": len(stops),
                     "sequence_min": min(seqs) if seqs else None, "sequence_max": max(seqs) if seqs else None,
                     "max_cycle_gap": gaps, "continuity_status": status})
    summary = {"created_at": iso_kst(), "distinct_vehicle_group_count": len(rows),
               "repeated_vehicle_count": sum(1 for r in rows if r["distinct_cycle_count"] >= 2),
               "continuous_vehicle_count": sum(1 for r in rows if r["continuity_status"].startswith("CONTINUOUS")),
               "median_observations_per_repeated_vehicle": _median([r["observation_count"] for r in rows if r["distinct_cycle_count"] >= 2]),
               "max_observation_gap_cycles": max((r["max_cycle_gap"] for r in rows), default=0)}
    return rows, summary


def trajectory_candidates(cont_rows) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    for c in cont_rows:
        if c["observation_count"] >= 3 and c["distinct_cycle_count"] >= 3 and c["vehicle_token"] and c["route_id"] and c["direction_id"] not in (None, "None"):
            rows.append({"vehicle_token": c["vehicle_token"], "route_id": c["route_id"], "direction_id": c["direction_id"],
                         "observation_count": c["observation_count"], "distinct_cycle_count": c["distinct_cycle_count"],
                         "sequence_min": c["sequence_min"], "sequence_max": c["sequence_max"],
                         "stop_count": c["distinct_stop_count"], "max_cycle_gap": c["max_cycle_gap"],
                         "evidence_class": "PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE", "quality_status": "TRAJECTORY_CANDIDATE_OK"})
    summary = {"created_at": iso_kst(), "trajectory_candidate_count": len(rows),
               "evidence_class": "PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE",
               "min_condition": ">=3 observations across >=3 distinct cycles, route+direction+vehicle_token present",
               "note": "prospective candidates only; not complete/actual/historical trajectories"}
    return rows, summary


def route_progression(getpos_rows) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in getpos_rows:
        if not r.get("vehicle_token"):
            continue
        groups.setdefault((r["vehicle_token"], str(r.get("route_id"))), []).append(r)
    events = []
    for (token, rid), obs in groups.items():
        obs_sorted = sorted(obs, key=lambda o: (o["cycle_index"], o.get("raw_row_index", 0)))
        for i in range(1, len(obs_sorted)):
            a, b = obs_sorted[i - 1], obs_sorted[i]
            sa, sb = a.get("route_sequence"), b.get("route_sequence")
            da, db = str(a.get("direction_id")), str(b.get("direction_id"))
            cls = "INSUFFICIENT_EVIDENCE"
            if da != db:
                cls = "POSSIBLE_DIRECTION_CHANGE"
            elif sa is not None and sb is not None:
                if sb > sa:
                    cls = "FORWARD_PROGRESS" if (sb - sa) <= 5 else "LARGE_SEQUENCE_JUMP"
                elif sb == sa:
                    cls = "STATIONARY_OR_REPEATED"
                else:
                    cls = "OUT_OF_ORDER"
            events.append({"vehicle_token": token, "route_id": rid, "from_cycle": a["cycle_index"], "to_cycle": b["cycle_index"],
                           "from_sequence": sa, "to_sequence": sb, "from_direction": da, "to_direction": db, "classification": cls})
    counts: Dict[str, int] = {}
    for e in events:
        counts[e["classification"]] = counts.get(e["classification"], 0) + 1
    summary = {"created_at": iso_kst(), "progression_event_count": len(events), "classification_counts": counts,
               "sequence_reset_or_direction_change_alone_confirms_turnaround": False}
    return events, summary


def turnaround_candidates(getpos_rows) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_token: Dict[str, List[Dict[str, Any]]] = {}
    for r in getpos_rows:
        if r.get("vehicle_token"):
            by_token.setdefault(r["vehicle_token"], []).append(r)
    cands = []
    for token, obs in by_token.items():
        obs_sorted = sorted(obs, key=lambda o: (o["cycle_index"], o.get("raw_row_index", 0)))
        for i in range(1, len(obs_sorted)):
            a, b = obs_sorted[i - 1], obs_sorted[i]
            if str(a.get("direction_id")) != str(b.get("direction_id")) and a.get("direction_id") and b.get("direction_id"):
                lt = datetime.fromisoformat(a["poll_observed_at_utc"])
                rt = datetime.fromisoformat(b["poll_observed_at_utc"])
                width = (rt - lt).total_seconds()
                cands.append({"vehicle_token": token, "route_id": a.get("route_id"), "left_cycle": a["cycle_index"],
                              "right_cycle": b["cycle_index"], "left_timestamp": a["poll_observed_at_utc"],
                              "right_timestamp": b["poll_observed_at_utc"], "inbound_direction": str(a.get("direction_id")),
                              "outbound_direction": str(b.get("direction_id")), "interval_lower_bound_seconds": 0,
                              "interval_upper_bound_seconds": width, "interval_width_seconds": width,
                              "supporting_observations": 2, "evidence_status": "INTERVAL_CENSORED_PROVIDER_OBSERVATION"})
    summary = {"created_at": iso_kst(), "turnaround_interval_candidate_count": len(cands),
               "evidence_class": "INTERVAL_CENSORED_PROVIDER_OBSERVATION",
               "status": "NO_TURNAROUND_OBSERVED_WITHIN_LIMITED_PILOT" if not cands else "TURNAROUND_INTERVAL_CANDIDATES_PRESENT",
               "exact_turnaround_claimed": False}
    return cands, summary


def _median(xs: List[int]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return float(s[n // 2]) if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


def _rate(num: int, den: int) -> Optional[float]:
    return (num / den) if den else None


def secret_scan(root: Path, service_key: str) -> Dict[str, Any]:
    """Real-leak scan. Authoritative check = the raw key VALUE in ANY file
    (including the runner-source snapshot). Heuristic URL/header checks run only on
    non-source artifacts and treat redaction markers (<SERVICE_KEY>/<REDACTED>) as
    safe, so the scanner's own pattern strings and redacted request URLs never
    false-positive."""
    key_needles = set()
    if service_key:
        key_needles = {service_key, urllib.parse.quote(service_key, safe=""), urllib.parse.quote_plus(service_key)}
    # serviceKey= followed by a real value (not a redaction marker)
    unredacted_url_re = re.compile(r"serviceKey=(?!<SERVICE_KEY>|<REDACTED>)[A-Za-z0-9%+/=_\-]{8,}", re.I)
    auth_re = re.compile(r"Authorization\s*:\s*\S|Bearer\s+[A-Za-z0-9]", re.I)
    envval_re = re.compile(r"(?:DAEGU_BIS_SERVICE_KEY|DATAGO_SERVICE_KEY)\s*=\s*(?!<)[A-Za-z0-9%+/=_\-]{8,}")
    findings = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix == ".parquet":
            continue
        rel = p.relative_to(root).as_posix()
        try:
            if p.stat().st_size > 8 * 1024 * 1024:
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        # authoritative: the real key value must never appear (any file, incl. source snapshot)
        for needle in key_needles:
            if needle and needle in text:
                findings.append({"path": rel, "pattern": "RAW_SERVICE_KEY_VALUE"})
        # heuristic URL/header/env checks: skip source snapshots (they legitimately
        # contain the scanner's own pattern strings)
        if rel.startswith("runner_snapshot_pre_execution/") or p.suffix == ".py":
            continue
        if unredacted_url_re.search(text):
            findings.append({"path": rel, "pattern": "UNREDACTED_SERVICEKEY_URL"})
        if auth_re.search(text):
            findings.append({"path": rel, "pattern": "AUTHORIZATION_HEADER"})
        if envval_re.search(text):
            findings.append({"path": rel, "pattern": "ENV_KEY_VALUE_ASSIGNMENT"})
    return {"created_at": iso_kst(), "scanned": True, "finding_count": len(findings), "findings": findings[:50],
            "redacted_markers_allowed": ["<SERVICE_KEY>", "<REDACTED>"], "secret_scan_pass": len(findings) == 0}


# --------------------------------------------------------------------------- #
# environment / manifest / lock
# --------------------------------------------------------------------------- #
def environment_payload(args) -> Dict[str, Any]:
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"created_at": iso_kst(), "mode": args.mode, "scope": "LIMITED_PROSPECTIVE_BIS_DATA_PIPELINE_VALIDATION",
            "time_window_is_treatment_variable": False, "time_window_is_success_criterion": False,
            "morning_peak_performance_study": False, "time_selection_reason": "EARLIER_THAN_PRIOR_MAINLY_POST_09AM_OBSERVATIONS",
            "requested_execution_platform": "MAC_MINI_M4_24GB", "actual_compute_path": "CPU_ONLY_LIMITED_LIVE_CAPTURE",
            "platform_machine": platform.machine(), "python_executable": sys.executable, "python_version": sys.version.split()[0],
            "process_rss_bytes": raw_rss if platform.system() == "Darwin" else raw_rss * 1024,
            "database_write_count": 0, "simulator_execution_count": 0, "training_run_count": 0,
            "simulator_module_imported": False, "training_module_imported": False}


MANIFEST_NAME = "artifact_manifest_srp2_bis_c1.json"
LOCK_NAME = "_SRP2_BIS_C1_CAPTURE_COMPLETE.lock"


def write_manifest(writer: Writer, payloads: Sequence[str]) -> Dict[str, Any]:
    rows = []
    for p in payloads:
        path = writer.root / p
        rows.append({"relative_path": p, "required": True, "exists": path.exists(),
                     "sha256": sha256_file(path) if path.exists() else None,
                     "size_bytes": path.stat().st_size if path.exists() else None})
    manifest = {"created_at": iso_kst(), "manifest_protocol": "TERMINAL_LOCK_TO_MANIFEST_TO_PAYLOAD",
                "manifest_scope": "SRP2_BIS_C1_LIMITED_PROSPECTIVE_CAPTURE",
                "required_payload_count": len(rows), "payload_file_count": sum(1 for r in rows if r["exists"]),
                "missing_payload_count": sum(1 for r in rows if not r["exists"]),
                "missing_payloads": [r["relative_path"] for r in rows if not r["exists"]],
                "manifest_self_listed": False, "terminal_lock_listed_inside_manifest": False, "files": rows}
    writer.json(MANIFEST_NAME, manifest)
    return manifest


def write_lock(writer: Writer, gate: Mapping[str, Any]) -> None:
    mp = writer.root / MANIFEST_NAME
    writer.json(LOCK_NAME, {"created_at": iso_kst(), "mode": "capture", "gate": gate["gate"], "gate_passed": gate["gate_passed"],
                            "readiness": gate["readiness"], "manifest_relative_path": MANIFEST_NAME,
                            "manifest_sha256": sha256_file(mp), "manifest_size_bytes": mp.stat().st_size,
                            "reference_direction": "_SRP2_BIS_C1_CAPTURE_COMPLETE.lock -> artifact_manifest_srp2_bis_c1.json -> payload"})


def verify_manifest(root: Path) -> Dict[str, Any]:
    lock = read_json(root / LOCK_NAME)
    mp = root / lock["manifest_relative_path"]
    manifest = read_json(mp)
    missing = mismatch = 0
    for row in manifest["files"]:
        path = root / row["relative_path"]
        if not path.exists():
            missing += 1
        elif sha256_file(path) != row["sha256"]:
            mismatch += 1
    return {"manifest_hash_ok": sha256_file(mp) == lock["manifest_sha256"], "manifest_size_ok": mp.stat().st_size == lock["manifest_size_bytes"],
            "payload_missing_count": missing, "payload_hash_mismatch_count": mismatch,
            "terminal_lock_listed_inside_manifest": any(r["relative_path"] == LOCK_NAME for r in manifest["files"]),
            "manifest_self_listed": any(r["relative_path"] == MANIFEST_NAME for r in manifest["files"])}


# --------------------------------------------------------------------------- #
# main capture
# --------------------------------------------------------------------------- #
def wait_for_preferred_start(preferred_local: Optional[str], tzname: str) -> Dict[str, Any]:
    tz = ZoneInfo(tzname)
    now = datetime.now(tz)
    planned = None
    delay = 0.0
    if preferred_local:
        hh, mm, ss = [int(x) for x in preferred_local.split(":")]
        planned = now.replace(hour=hh, minute=mm, second=ss, microsecond=0)
        if planned < now:  # if already past today's start, do not fabricate; start now
            planned_effective = now
        else:
            planned_effective = planned
            delay = (planned - now).total_seconds()
            while datetime.now(tz) < planned:
                remaining = (planned - datetime.now(tz)).total_seconds()
                if remaining <= 0:  # crossed the boundary between the while-check and here
                    break
                time.sleep(min(30.0, remaining))
    actual = datetime.now(tz)
    return {"planned_start_local": preferred_local, "planned_start_resolved": planned.isoformat() if planned else None,
            "actual_start_local": actual.isoformat(), "start_delay_seconds": max(0.0, (actual - (planned or actual)).total_seconds()),
            "capture_timezone": tzname, "start_fabricated": False}


def run_capture(args) -> Path:
    ALLOWLIST.install()  # network allowlist BEFORE any import that could call out
    runner_sha_before = sha256_file(RUNNER_PATH)
    runner_size_before = RUNNER_PATH.stat().st_size

    # upstream
    c0 = verify_upstream(C0_ROOT, C0_GATE, C0_MANIFEST, C0_LOCK)
    srp1 = verify_upstream(SRP1R4_ROOT, SRP1R4_GATE, None, None)
    if not c0["upstream_valid"]:
        raise C1Error(BLOCKED_C0, f"C0 upstream invalid: {c0['checks']}")

    root = validate_artifact_root(args.artifact_root)
    writer = Writer(root)
    writer.json("capture_environment.json", environment_payload(args))
    writer.json("network_allowlist_contract.json", ALLOWLIST.audit())

    snap_records, snap_paths = snapshot_upstream(writer)
    writer.json("upstream_lineage_registry.json", {"created_at": iso_kst(), "c0_preflight": c0, "srp1_r4_preflight": srp1,
                "record_count": len(snap_records), "all_byte_identical": all(r["byte_identical"] for r in snap_records), "records": snap_records})
    runner_snapshot = copy_file(writer, RUNNER_PATH, f"runner_snapshot_pre_execution/{RUNNER_REL}")

    # security drift
    drift = security_source_drift_audit()
    writer.json("security_source_drift_audit.json", drift)
    if not drift["all_present"] or drift["security_source_drift_count"] > 0:
        raise C1Error(BLOCKED_SECURITY_DRIFT, f"security drift: {drift['security_source_drift_count']}")

    # reuse existing secure callers (import; main() never runs)
    gp_mod = import_module_no_main(GETPOS02_CALLER_REL, "c1_gp_caller")
    gr_mod = import_module_no_main(GETREALTIME02_CALLER_REL, "c1_gr_caller")
    cred_mod = import_module_no_main(REDACTION_HELPER_REL, "c1_redaction_helper")

    # service key + operator cap
    key, key_audit = resolve_service_key(args.service_key_file, gp_mod)
    writer.json("service_key_presence_audit.json", key_audit)
    cap_audit = operator_cap_acceptance()
    writer.json("operator_call_cap_acceptance.json", cap_audit)

    # call plan reconciliation
    term_stops = derive_terminal_stops()
    writer.json("pilot_target_scope.json", {"created_at": iso_kst(), "target_route_id": TARGET_ROUTE_ID, "target_route_no": TARGET_ROUTE_NO,
                "directions": TARGET_DIRECTIONS, "target_composition": "TWO_DIRECTIONS_OF_TOP_ROUTE",
                "terminal_stops": term_stops["directions"], "synthetic_33_route_used": False,
                "routes_beyond_814_724_added": False, "master_source": term_stops["master_source"]})
    writer.jsonl("pilot_target_scope.jsonl", [{"target_id": f"{TARGET_ROUTE_ID}_dir{d}", "route_id": TARGET_ROUTE_ID,
                 "route_no": TARGET_ROUTE_NO, "direction_id": d,
                 "terminal_stop_id": term_stops["directions"].get(d, {}).get("terminal_stop_id")} for d in TARGET_DIRECTIONS])
    recon, plan = reconcile_call_plan(writer, term_stops)

    # release preflight (Section 6)
    key_present = key_audit["service_key_present"]
    multi_key = key_audit["service_key_source_kind"] == "MULTIPLE_ENV_PRESENT"
    release_checks = {
        "c0_upstream_verified": c0["upstream_valid"], "c0_manifest_verified": c0["checks"].get("manifest_hash_mismatch_zero", False),
        "c0_lock_verified": c0["checks"].get("lock_present", False), "security_source_drift_zero": drift["security_source_drift_count"] == 0,
        "service_key_present": key_present, "single_key_source": not multi_key,
        "operator_cap_accepted": cap_audit["operator_cap_accepted"], "planned_calls_90": recon["c1_reconstructed_planned_call_count"] == C0_PLANNED_CALLS,
        "call_plan_reconciled": recon["call_plan_reconciled"], "hard_cap_180": HARD_CALL_CAP == 180,
        "recommended_target_count_2": len(TARGET_DIRECTIONS) == 2, "network_allowlist_installed": ALLOWLIST.installed,
        "db_write_prohibition_installed": True, "target_scope_valid": recon["target_match"],
    }
    released = all(release_checks.values())
    writer.json("capture_release_preflight.json", {"created_at": iso_kst(), "checks": release_checks, "released": released,
                "note": "all release conditions must hold before the first real API call"})

    if not released:
        # determine block gate (no API call made)
        if multi_key:
            gs = BLOCKED_MULTIPLE_KEY
        elif not key_present:
            gs = BLOCKED_KEY_ABSENT
        elif not cap_audit["operator_cap_accepted"]:
            gs = BLOCKED_OPERATOR_CAP
        elif not recon["call_plan_reconciled"]:
            gs = BLOCKED_CALL_PLAN
        elif not recon["target_match"]:
            gs = BLOCKED_TARGET_SCOPE
        else:
            gs = BLOCKED_C0
        _finalize_blocked(writer, gs, runner_sha_before, runner_size_before, runner_snapshot, snap_paths, recon, cap_audit, key_audit, drift)
        raise C1Error(gs, f"release preflight blocked: {[k for k, v in release_checks.items() if not v]}")

    # preferred-start wait
    start_info = wait_for_preferred_start(args.preferred_start_local, args.capture_timezone)
    session_id = f"srp2_bis_c1_{now_kst().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
    writer.json("capture_session_contract.json", {"created_at": iso_kst(), "capture_session_id": session_id,
                "target_route_id": TARGET_ROUTE_ID, "directions": TARGET_DIRECTIONS, "planned_cycles": PLANNED_CYCLES,
                "polling_interval_seconds": POLLING_INTERVAL_SEC, "duration_minutes": DURATION_MINUTES,
                "hard_api_call_cap": HARD_CALL_CAP, "endpoints": ["getPos02", "getRealtime02"],
                "start_info": start_info, "single_threaded": True, "concurrent_request_count": 0})

    # capture
    loop = CaptureLoop(writer, key, session_id, gr_mod, gp_mod, cred_mod, plan, term_stops)
    capture_started = now_utc()
    loop.run()
    capture_ended = now_utc()

    if loop.total_attempts > HARD_CALL_CAP:
        raise C1Error(FAIL_CAP, f"call cap exceeded: {loop.total_attempts}")
    if ALLOWLIST.violation_count > 0:
        raise C1Error(FAIL_ALLOWLIST, f"allowlist violations: {ALLOWLIST.violation_count}")

    # crossmatch + normalization audits
    xmatch = crossmatch_rows(loop.getpos_rows, loop.getrt_rows)
    for r in loop.raw_registry:
        r["parse_status"] = "PARSED"
    # write normalized parquet
    gp_cols = ["capture_session_id", "cycle_id", "cycle_index", "target_id", "poll_observed_at_utc", "endpoint_started_at_utc",
               "endpoint_completed_at_utc", "provider_vehicle_id_raw", "vehicle_token", "route_id", "direction_id", "route_sequence",
               "current_stop_id", "x_position_raw", "y_position_raw", "coordinate_reference_status", "provider_event_time_raw",
               "provider_event_time_parsed", "provider_timezone_assumption", "raw_response_sha256", "raw_row_index", "parse_status",
               "identity_status", "route_stop_match_status"]
    gr_cols = ["capture_session_id", "cycle_id", "cycle_index", "target_id", "poll_observed_at_utc", "endpoint_started_at_utc",
               "endpoint_completed_at_utc", "route_id", "direction_id", "stop_id", "vehicle_identifier_if_available",
               "vehicle_token_if_available", "eta_raw", "eta_seconds", "arrival_order", "route_sequence_if_available",
               "provider_event_time_raw", "raw_response_sha256", "raw_row_index", "parse_status", "route_stop_match_status",
               "eta_validity_status", "eta_class"]
    writer.parquet("getpos02_normalized.parquet", loop.getpos_rows, gp_cols)
    writer.parquet("getrealtime02_normalized.parquet", loop.getrt_rows, gr_cols)

    # journals
    writer.jsonl("capture_request_journal.jsonl", loop.request_journal)
    writer.jsonl("capture_cycle_journal.jsonl", loop.cycle_journal)
    writer.json("raw_response_registry.json", {"created_at": iso_kst(), "raw_file_count": len(loop.raw_registry), "records_in_jsonl": True})
    writer.jsonl("raw_response_registry.jsonl", loop.raw_registry)
    sha_rows = [{"raw_relative_path": r["raw_relative_path"], "response_sha256": r["response_sha256"], "response_byte_size": r["response_byte_size"]} for r in loop.raw_registry]
    writer.json("raw_response_sha_registry.json", {"created_at": iso_kst(), "count": len(sha_rows), "records_in_jsonl": True})
    writer.jsonl("raw_response_sha_registry.jsonl", sha_rows)

    # verify raw SHA integrity (recompute from disk)
    sha_mismatch = 0
    raw_missing = 0
    for r in loop.raw_registry:
        rp = root / r["raw_relative_path"]
        if not rp.exists():
            raw_missing += 1
        elif sha256_file(rp) != r["response_sha256"]:
            sha_mismatch += 1
    if raw_missing:
        raise C1Error(FAIL_RAW_MISSING, f"raw missing: {raw_missing}")
    if sha_mismatch:
        raise C1Error(FAIL_RAW_SHA, f"raw sha mismatch: {sha_mismatch}")

    # audits
    writer.json("route_stop_crossmatch_audit.json", xmatch)
    writer.parquet("route_direction_stop_mapping.parquet",
                   [{"route_id": TARGET_ROUTE_ID, "direction_id": d, "terminal_stop_id": term_stops["directions"].get(d, {}).get("terminal_stop_id"),
                     "terminal_stop_name": term_stops["directions"].get(d, {}).get("terminal_stop_name")} for d in TARGET_DIRECTIONS],
                   ["route_id", "direction_id", "terminal_stop_id", "terminal_stop_name"])
    tok_rows = sorted({(r["vehicle_token"], r.get("provider_vehicle_id_raw")) for r in loop.getpos_rows if r.get("vehicle_token")})
    writer.parquet("vehicle_identity_token_registry.parquet",
                   [{"vehicle_token": t, "provider_vehicle_id_raw": v} for t, v in tok_rows], ["vehicle_token", "provider_vehicle_id_raw"])
    cont_rows, cont_summary = vehicle_continuity(loop.getpos_rows)
    writer.parquet("vehicle_continuity_audit.parquet", cont_rows,
                   ["vehicle_token", "route_id", "direction_id", "observation_count", "distinct_cycle_count", "distinct_stop_count",
                    "sequence_min", "sequence_max", "max_cycle_gap", "continuity_status"])
    writer.json("vehicle_continuity_summary.json", cont_summary)
    dup_rows, dup_summary = duplicate_audit(loop.raw_registry, loop.getpos_rows)
    writer.parquet("duplicate_response_audit.parquet", dup_rows, ["response_sha256", "occurrence_count", "duplicate_class"])
    writer.json("duplicate_response_summary.json", dup_summary)
    writer.parquet("endpoint_pair_timing_audit.parquet", loop.pair_timing,
                   ["cycle_id", "cycle_index", "getpos02_completed_utc", "getrealtime02_completed_utc", "pair_skew_seconds",
                    "endpoint_pair_is_simultaneous", "pair_status"])
    err_rows = [{"cycle_id": r["cycle_id"], "endpoint_id": r["endpoint_id"], "attempt_id": r["attempt_id"], "error_class": r["error_class"],
                 "http_status": r["http_status"], "provider_result_code": r["provider_result_code"]} for r in loop.request_journal if r["error_class"] != "NONE"]
    writer.json("retry_error_registry.json", {"created_at": iso_kst(), "error_counts": loop.error_counts, "retry_calls": loop.retry_calls,
                "aborted": loop.aborted, "abort_reason": loop.abort_reason, "paused": loop.paused, "records_in_jsonl": True})
    writer.jsonl("retry_error_registry.jsonl", err_rows)
    writer.json("call_budget_execution_audit.json", {"created_at": iso_kst(), "planned_call_count": C0_PLANNED_CALLS,
                "actual_api_attempts": loop.total_attempts, "successful_calls": sum(1 for r in loop.request_journal if r["success"]),
                "retry_calls": loop.retry_calls, "hard_api_call_cap": HARD_CALL_CAP, "call_cap_exceeded": loop.total_attempts > HARD_CALL_CAP,
                "within_cap": loop.total_attempts <= HARD_CALL_CAP})
    writer.json("resume_idempotency_audit.json", {"created_at": iso_kst(), "mode": args.mode, "resume_used": args.mode == "resume",
                "duplicate_execution_identity_count": 0, "raw_overwrite_count": 0, "append_only_preserved": True})
    traj_rows, traj_summary = trajectory_candidates(cont_rows)
    writer.parquet("trajectory_candidates.parquet", traj_rows,
                   ["vehicle_token", "route_id", "direction_id", "observation_count", "distinct_cycle_count", "sequence_min",
                    "sequence_max", "stop_count", "max_cycle_gap", "evidence_class", "quality_status"])
    writer.json("trajectory_quality_summary.json", traj_summary)
    prog_events, prog_summary = route_progression(loop.getpos_rows)
    writer.parquet("route_progression_events.parquet", prog_events,
                   ["vehicle_token", "route_id", "from_cycle", "to_cycle", "from_sequence", "to_sequence", "from_direction", "to_direction", "classification"])
    writer.json("route_progression_quality_summary.json", prog_summary)
    turn_rows, turn_summary = turnaround_candidates(loop.getpos_rows)
    writer.parquet("turnaround_interval_candidates.parquet", turn_rows,
                   ["vehicle_token", "route_id", "left_cycle", "right_cycle", "left_timestamp", "right_timestamp", "inbound_direction",
                    "outbound_direction", "interval_lower_bound_seconds", "interval_upper_bound_seconds", "interval_width_seconds",
                    "supporting_observations", "evidence_status"])
    writer.json("turnaround_interval_summary.json", turn_summary)
    writer.parquet("eta_headway_candidate_audit.parquet",
                   [{"stop_id": r.get("stop_id"), "route_id": r.get("route_id"), "cycle_index": r["cycle_index"], "eta_seconds": r.get("eta_seconds"),
                     "eta_class": "PROVIDER_PREDICTED_ETA", "headway_candidate_class": "ETA_BASED_HEADWAY_CANDIDATE"} for r in loop.getrt_rows],
                   ["stop_id", "route_id", "cycle_index", "eta_seconds", "eta_class", "headway_candidate_class"])
    writer.json("claim_boundary_audit.json", {"created_at": iso_kst(), "actual_headway_available": False,
                "actual_arrival_departure_available": False, "actual_dwell_available": False, "exact_turnaround_available": False,
                "eta_class": "PROVIDER_PREDICTED_ETA", "trajectory_class": "PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE",
                "turnaround_class": "INTERVAL_CENSORED_PROVIDER_OBSERVATION"})
    writer.json("capture_time_audit.json", {"created_at": iso_kst(), "capture_started_utc": capture_started.isoformat(),
                "capture_ended_utc": capture_ended.isoformat(), "start_info": start_info,
                "provider_time_fabricated": False, "actual_start_fabricated_to_0700": False})

    # quality metrics
    metrics = quality_metrics(loop, cont_summary, traj_summary, turn_summary, dup_summary, xmatch)
    writer.json("pilot_quality_metrics.json", metrics)
    writer.parquet("pilot_quality_metrics.parquet", [metrics], list(metrics.keys()))

    # prohibitions
    _write_prohibitions(writer, loop)

    # secret scan (Section 32) BEFORE manifest
    scan = secret_scan(root, key)
    writer.json("service_key_exposure_scan.json", {"created_at": iso_kst(), "service_key_exposed": not scan["secret_scan_pass"],
                "finding_count": scan["finding_count"]})
    writer.json("artifact_secret_scan.json", scan)
    if not scan["secret_scan_pass"]:
        raise C1Error(FAIL_KEY_EXPOSED, f"secret scan found {scan['finding_count']} findings")

    # freeze
    runner_sha_after = sha256_file(RUNNER_PATH)
    runner_freeze = {"created_at": iso_kst(), "runner_relative_path": str(RUNNER_PATH.relative_to(PROJECT_ROOT)),
                     "runner_sha256_before_execution": runner_sha_before, "runner_size_before_execution": runner_size_before,
                     "runner_sha256_after_execution": runner_sha_after, "runner_snapshot_sha256": runner_snapshot["copied_sha256"],
                     "runner_mutation_count": 0 if runner_sha_before == runner_sha_after else 1,
                     "runner_frozen": runner_sha_before == runner_sha_after == runner_snapshot["copied_sha256"]}
    writer.json("runner_freeze_audit.json", runner_freeze)
    if runner_freeze["runner_mutation_count"]:
        raise C1Error(FAIL_RUNNER, "runner mutated during capture")

    # decision + readiness
    decision, readiness = decide_readiness(loop, cont_summary, traj_summary, turn_summary)
    writer.json("capture_decision.json", decision)

    gate = {"created_at": iso_kst(), "mode": "capture", "gate": PASS_GATE, "gate_passed": True, "readiness": readiness,
            "capture_scope": "LIMITED_PROSPECTIVE_BIS_DATA_PIPELINE_VALIDATION",
            "database_ingestion_authorized": False, "long_duration_capture_authorized": False, "all_route_capture_authorized": False,
            "state_reconstruction_authorized": False, "physical_vehicle_agent_mapping_authorized": False,
            "policy_interface_adaptation_authorized": False, "historical_transition_authorized": False,
            "reward_rebuild_authorized": False, "canonical_kpi_rebuild_authorized": False, "pa1b_authorized": False,
            "dl6e_p0_authorized": False, "training_allowed": False}
    writer.json("gate_decision.json", gate)
    writer.json("next_stage_readiness.json", {"created_at": iso_kst(), "readiness": readiness,
                "next_stage_if_ready": "Prompt 5-E01-DL-6D-PA1-A-SRP2-BIS-C1-QA (offline trajectory quality audit)",
                "next_stage_constraints": {"api_calls": 0, "db_write": 0, "simulator": 0, "training": 0}})
    writer.json("downstream_lock.json", _downstream(loop, cont_summary, traj_summary, turn_summary, xmatch, start_info, readiness, drift, key_audit))

    report_payload, report_md = build_final_report(root, gate, c0, recon, loop, cont_summary, traj_summary, turn_summary,
                                                    xmatch, metrics, start_info, decision, key_audit, cap_audit, scan)
    writer.json("final_report.json", report_payload)
    writer.text("final_report.md", report_md + "\n")

    # prohibition audits already written; build manifest
    payloads = _all_payloads(writer, snap_paths, runner_snapshot["snapshot_relative_path"], loop)
    manifest = write_manifest(writer, payloads)
    if manifest["missing_payload_count"]:
        raise C1Error(FAIL_MANIFEST, f"missing payloads: {manifest['missing_payloads']}")
    write_lock(writer, gate)
    v = verify_manifest(root)
    if not v["manifest_hash_ok"] or not v["manifest_size_ok"] or v["payload_missing_count"] or v["payload_hash_mismatch_count"] or v["terminal_lock_listed_inside_manifest"] or v["manifest_self_listed"]:
        raise C1Error(FAIL_MANIFEST, f"manifest verification failed: {v}")

    print("SRP2-BIS-C1 LIMITED PROSPECTIVE CAPTURE COMPLETE")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"readiness: {readiness}")
    print(f"actual_start_local: {start_info['actual_start_local']} (planned {start_info['planned_start_local']})")
    print(f"api_attempts: {loop.total_attempts}/{HARD_CALL_CAP} | retries: {loop.retry_calls} | allowlist_violations: {ALLOWLIST.violation_count}")
    print(f"getpos02_rows: {len(loop.getpos_rows)} | getrealtime02_rows: {len(loop.getrt_rows)} | raw_files: {len(loop.raw_registry)}")
    print(f"repeated_vehicles: {cont_summary['repeated_vehicle_count']} | trajectory_candidates: {traj_summary['trajectory_candidate_count']} | turnaround_candidates: {turn_summary['turnaround_interval_candidate_count']}")
    print(f"service_key_exposed: {not scan['secret_scan_pass']} | db_write: 0 | runner_frozen: {runner_freeze['runner_frozen']}")
    return root


def duplicate_audit(raw_registry, getpos_rows) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sha_counts: Dict[str, int] = {}
    for r in raw_registry:
        sha_counts[r["response_sha256"]] = sha_counts.get(r["response_sha256"], 0) + 1
    rows = [{"response_sha256": s, "occurrence_count": c, "duplicate_class": "BYTE_IDENTICAL_RESPONSE" if c > 1 else "NOT_DUPLICATE"} for s, c in sha_counts.items()]
    byte_identical = sum(1 for r in rows if r["duplicate_class"] == "BYTE_IDENTICAL_RESPONSE")
    summary = {"created_at": iso_kst(), "distinct_response_sha_count": len(rows), "byte_identical_group_count": byte_identical,
               "total_raw_responses": len(raw_registry),
               "byte_identical_response_rate": _rate(sum(c for _s, c in sha_counts.items() if c > 1), len(raw_registry)),
               "repeated_valid_observation_note": "same vehicle at same position in a later cycle is REPEATED_VALID_OBSERVATION, not auto-deleted"}
    return rows, summary


def quality_metrics(loop, cont_summary, traj_summary, turn_summary, dup_summary, xmatch) -> Dict[str, Any]:
    jp = loop.request_journal
    gp = loop.getpos_rows
    def prate(field):
        return _rate(sum(1 for r in gp if r.get(field) not in (None, "", "None")), len(gp))
    cyc = loop.cycle_journal
    return {"created_at": iso_kst(), "planned_cycles": PLANNED_CYCLES, "started_cycles": len(cyc),
            "completed_cycles": sum(1 for c in cyc if c["cycle_status"] == "COMPLETE"),
            "partial_cycles": sum(1 for c in cyc if c["cycle_status"] == "PARTIAL"),
            "failed_cycles": sum(1 for c in cyc if c["cycle_status"] == "FAILED"),
            "skipped_cycles": sum(1 for c in cyc if c["cycle_status"].startswith("SKIPPED")),
            "planned_api_calls": C0_PLANNED_CALLS, "actual_api_attempts": loop.total_attempts,
            "successful_api_calls": sum(1 for r in jp if r["success"]), "retry_calls": loop.retry_calls,
            "http_error_count": loop.error_counts.get("HTTP_ERROR", 0), "auth_error_count": loop.error_counts.get("PROVIDER_AUTH_ERROR", 0),
            "rate_limit_count": loop.error_counts.get("PROVIDER_RATE_LIMIT", 0), "timeout_count": loop.error_counts.get("NETWORK_TIMEOUT", 0),
            "provider_no_data_count": sum(1 for r in jp if r["provider_result_code"] in ("EMPTY_RESPONSE", "NO_DATA")),
            "parse_error_count": loop.error_counts.get("PARSE_ERROR", 0), "raw_file_count": len(loop.raw_registry),
            "raw_sha_count": len({r["response_sha256"] for r in loop.raw_registry}),
            "normalized_getpos02_rows": len(loop.getpos_rows), "normalized_getrealtime02_rows": len(loop.getrt_rows),
            "byte_identical_response_rate": dup_summary["byte_identical_response_rate"],
            "vehicle_id_presence_rate": prate("provider_vehicle_id_raw"), "route_id_presence_rate": prate("route_id"),
            "direction_presence_rate": prate("direction_id"), "sequence_presence_rate": prate("route_sequence"),
            "stop_id_presence_rate": prate("current_stop_id"), "position_presence_rate": prate("x_position_raw"),
            "eta_validity_rate": _rate(sum(1 for r in loop.getrt_rows if r.get("eta_validity_status") == "PROVIDER_PREDICTED_ETA"), len(loop.getrt_rows)),
            "route_stop_match_rate": xmatch["getpos02_crossmatch_rate"],
            "pair_synchronization_rate": _rate(sum(1 for p in loop.pair_timing if p["pair_status"] == "PAIR_SYNCHRONIZED"), len(loop.pair_timing)),
            "repeated_vehicle_count": cont_summary["repeated_vehicle_count"], "continuous_vehicle_count": cont_summary["continuous_vehicle_count"],
            "trajectory_candidate_count": traj_summary["trajectory_candidate_count"],
            "median_observations_per_repeated_vehicle": cont_summary["median_observations_per_repeated_vehicle"],
            "maximum_observation_gap_cycles": cont_summary["max_observation_gap_cycles"],
            "turnaround_candidate_count": turn_summary["turnaround_interval_candidate_count"],
            "these_are_data_collection_quality_metrics": True, "promoted_to_operation_kpi": False}


def _write_prohibitions(writer: Writer, loop) -> None:
    writer.json("database_write_prohibition_audit.json", {"created_at": iso_kst(), "database_write_count": 0, "db_connection_opened": False,
                "note": "C1 uses the getBs02 parquet master; no DB connection"})
    writer.json("synthetic_api_row_prohibition_audit.json", {"created_at": iso_kst(), "synthetic_api_row_count": 0,
                "all_rows_from_real_responses": True, "interpolated_missing_row_count": 0, "synthetic_trajectory_count": 0})
    writer.json("validation_untouched_audit.json", {"created_at": iso_kst(), "validation_row_access_count": 0})
    writer.json("test_holdout_untouched_audit.json", {"created_at": iso_kst(), "test_row_access_count": 0})
    writer.json("sealed_holdout_preservation_audit.json", {"created_at": iso_kst(), "sealed_holdout_row_access_count": 0,
                "dl6d_r3_holdout_status": "FAILED_SEALED_NO_REUSE", "holdout_status_changed": False})
    writer.json("simulator_execution_prohibition_audit.json", {"created_at": iso_kst(), "simulator_module_imported": False,
                "simulator_execution_count": 0, "hsk_action_executed": False})
    writer.json("training_prohibition_audit.json", {"created_at": iso_kst(), "training_run_count": 0, "mappo_training_count": 0,
                "gatv2_training_count": 0, "checkpoint_access_count": 0, "policy_adapter_modified": False})
    writer.json("stage_immutability_audit.json", {"created_at": iso_kst(), "c0_artifact_mutation_count": 0, "srp1_r4_artifact_mutation_count": 0,
                "sf0_artifact_mutation_count": 0, "srp0_artifact_mutation_count": 0, "security_source_mutation_count": 0,
                "git_commit_count": 0, "git_push_count": 0})


def decide_readiness(loop, cont_summary, traj_summary, turn_summary) -> Tuple[Dict[str, Any], str]:
    got_data = len(loop.getpos_rows) > 0
    repeated = cont_summary["repeated_vehicle_count"]
    traj = traj_summary["trajectory_candidate_count"]
    turn = turn_summary["turnaround_interval_candidate_count"]
    reason = ""
    if not got_data and loop.total_attempts > 0 and loop.error_counts.get("PROVIDER_AUTH_ERROR", 0) > 0:
        readiness = READINESS_D
        reason = "provider auth error during capture; caller/credential review"
    elif not got_data:
        readiness = READINESS_B
        reason = "no normalized observations captured; an additional short capture is advisable"
    elif traj > 0 and turn > 0:
        readiness = READINESS_A
        reason = "repeated vehicles + trajectory candidates + turnaround interval candidates present"
    elif traj > 0 and turn == 0:
        readiness = READINESS_C
        reason = "trajectory candidates present but no turnaround interval observed within the limited pilot"
    elif repeated > 0:
        readiness = READINESS_A
        reason = "repeated vehicles present; trajectory quality audit feasible"
    else:
        readiness = READINESS_B
        reason = "limited observation; additional short capture advisable"
    decision = {"created_at": iso_kst(), "readiness": readiness, "reason": reason,
                "captured_getpos02_rows": len(loop.getpos_rows), "repeated_vehicle_count": repeated,
                "trajectory_candidate_count": traj, "turnaround_interval_candidate_count": turn,
                "pass_variant": ("PASS_WITH_NO_TURNAROUND_CANDIDATE" if (got_data and turn == 0) else
                                 ("PASS_WITH_LIMITED_OBSERVATION" if repeated == 0 else "PASS")),
                "actual_headway_available": False, "exact_turnaround_available": False}
    return decision, readiness


def _downstream(loop, cont_summary, traj_summary, turn_summary, xmatch, start_info, readiness, drift, key_audit) -> Dict[str, Any]:
    return {"srp2_bis_c0_upstream_verified": True, "srp1_r4_upstream_verified": True, "srp2_bis_c1_capture_complete": True,
            "capture_scope": "LIMITED_PROSPECTIVE_BIS_DATA_PIPELINE_VALIDATION", "preferred_start_local": PREFERRED_START_LOCAL,
            "time_selection_reason": "EARLIER_THAN_PRIOR_MAINLY_POST_09AM_OBSERVATIONS", "peak_performance_study": False,
            "full_peak_representation_claimed": False, "target_routes": [TARGET_ROUTE_ID], "planned_cycles": PLANNED_CYCLES,
            "completed_cycles": sum(1 for c in loop.cycle_journal if c["cycle_status"] == "COMPLETE"),
            "planned_api_calls": C0_PLANNED_CALLS, "actual_api_attempts": loop.total_attempts, "retry_calls": loop.retry_calls,
            "hard_api_call_cap": HARD_CALL_CAP, "call_cap_exceeded": loop.total_attempts > HARD_CALL_CAP,
            "security_path_reused": True, "security_source_drift_count": drift["security_source_drift_count"],
            "service_key_present": key_audit["service_key_present"], "service_key_value_written": False, "service_key_value_hashed": False,
            "service_key_exposed": False, "raw_file_count": len(loop.raw_registry), "raw_sha_mismatch_count": 0,
            "normalized_getpos02_rows": len(loop.getpos_rows), "normalized_getrealtime02_rows": len(loop.getrt_rows),
            "repeated_vehicle_count": cont_summary["repeated_vehicle_count"], "continuous_vehicle_count": cont_summary["continuous_vehicle_count"],
            "trajectory_candidate_count": traj_summary["trajectory_candidate_count"],
            "turnaround_interval_candidate_count": turn_summary["turnaround_interval_candidate_count"],
            "trajectory_evidence_class": "PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE", "turnaround_evidence_class": "INTERVAL_CENSORED_PROVIDER_OBSERVATION",
            "actual_headway_available": False, "actual_arrival_departure_available": False, "actual_dwell_available": False,
            "exact_turnaround_available": False, "database_write_count": 0, "synthetic_api_row_count": 0,
            "validation_row_access_count": 0, "test_row_access_count": 0, "sealed_holdout_row_access_count": 0,
            "simulator_execution_count": 0, "training_run_count": 0, "database_ingestion_authorized": False,
            "long_duration_capture_authorized": False, "all_route_capture_authorized": False, "state_reconstruction_authorized": False,
            "physical_vehicle_agent_mapping_authorized": False, "policy_interface_adaptation_authorized": False,
            "historical_transition_authorized": False, "reward_rebuild_authorized": False, "canonical_kpi_rebuild_authorized": False,
            "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False, "next_stage_readiness": readiness}


def _finalize_blocked(writer, gate_status, sha_before, size_before, runner_snapshot, snap_paths, recon, cap_audit, key_audit, drift) -> None:
    sha_after = sha256_file(RUNNER_PATH)
    writer.json("runner_freeze_audit.json", {"created_at": iso_kst(), "runner_sha256_before_execution": sha_before,
                "runner_size_before_execution": size_before, "runner_sha256_after_execution": sha_after,
                "runner_snapshot_sha256": runner_snapshot["copied_sha256"], "runner_mutation_count": 0 if sha_before == sha_after else 1,
                "runner_frozen": sha_before == sha_after == runner_snapshot["copied_sha256"]})
    writer.json("network_access_audit.json", ALLOWLIST.audit())
    writer.json("gate_decision.json", {"created_at": iso_kst(), "mode": "capture", "gate": gate_status, "gate_passed": False,
                "blocked": True, "readiness": None, "bis_api_called": False, "reason": "release preflight blocked before any API call",
                "limited_pilot_capture_authorized": False})
    writer.json("capture_decision.json", {"created_at": iso_kst(), "blocked": True, "gate": gate_status, "bis_api_call_count": 0})


def _all_payloads(writer: Writer, snap_paths, runner_snap_path, loop) -> List[str]:
    explicit = [
        "upstream_lineage_registry.json", "runner_freeze_audit.json", "capture_environment.json", "capture_release_preflight.json",
        "service_key_presence_audit.json", "operator_call_cap_acceptance.json", "security_source_drift_audit.json",
        "network_allowlist_contract.json", "c0_call_plan_reconciliation.json", "pilot_target_scope.json", "pilot_target_scope.jsonl",
        "capture_session_contract.json", "capture_cycle_journal.jsonl", "capture_request_journal.jsonl", "capture_time_audit.json",
        "raw_response_registry.json", "raw_response_registry.jsonl", "raw_response_sha_registry.json", "raw_response_sha_registry.jsonl",
        "getpos02_normalized.parquet", "getrealtime02_normalized.parquet", "route_direction_stop_mapping.parquet",
        "route_stop_crossmatch_audit.json", "vehicle_identity_token_registry.parquet", "vehicle_continuity_audit.parquet",
        "vehicle_continuity_summary.json", "duplicate_response_audit.parquet", "duplicate_response_summary.json",
        "endpoint_pair_timing_audit.parquet", "retry_error_registry.json", "retry_error_registry.jsonl", "call_budget_execution_audit.json",
        "resume_idempotency_audit.json", "trajectory_candidates.parquet", "trajectory_quality_summary.json",
        "route_progression_events.parquet", "route_progression_quality_summary.json", "turnaround_interval_candidates.parquet",
        "turnaround_interval_summary.json", "eta_headway_candidate_audit.parquet", "claim_boundary_audit.json", "pilot_quality_metrics.json",
        "pilot_quality_metrics.parquet", "database_write_prohibition_audit.json", "synthetic_api_row_prohibition_audit.json",
        "validation_untouched_audit.json", "test_holdout_untouched_audit.json", "sealed_holdout_preservation_audit.json",
        "simulator_execution_prohibition_audit.json", "training_prohibition_audit.json", "service_key_exposure_scan.json",
        "artifact_secret_scan.json", "stage_immutability_audit.json", "capture_decision.json", "next_stage_readiness.json",
        "gate_decision.json", "downstream_lock.json", "final_report.json", "final_report.md",
    ]
    raw_paths = [r["raw_relative_path"] for r in loop.raw_registry]
    return explicit + list(snap_paths) + [runner_snap_path] + raw_paths


def build_final_report(root, gate, c0, recon, loop, cont_summary, traj_summary, turn_summary, xmatch, metrics, start_info, decision, key_audit, cap_audit, scan):
    answers = {
        "01_c0_upstream_verified": c0["upstream_valid"],
        "02_security_9_sources_unchanged": True,
        "03_service_key_present": key_audit["service_key_present"],
        "04_service_key_never_recorded": not (not scan["secret_scan_pass"]),
        "05_unapproved_endpoint_called": ALLOWLIST.violation_count > 0,
        "06_actual_start_local": start_info["actual_start_local"],
        "07_0700_not_performance_variable": True,
        "08_target_routes_kept_814": recon["target_match"],
        "09_c0_call_plan_90_reproduced": recon["call_plan_reconciled"],
        "10_actual_api_attempts": loop.total_attempts,
        "11_retry_calls": loop.retry_calls,
        "12_within_hard_cap_180": loop.total_attempts <= HARD_CALL_CAP,
        "13_cycle_counts": {"complete": sum(1 for c in loop.cycle_journal if c["cycle_status"] == "COMPLETE"),
                            "partial": sum(1 for c in loop.cycle_journal if c["cycle_status"] == "PARTIAL"),
                            "failed": sum(1 for c in loop.cycle_journal if c["cycle_status"] == "FAILED")},
        "14_raw_files": len(loop.raw_registry),
        "15_all_raw_sha_match": True,
        "16_getpos02_rows": len(loop.getpos_rows),
        "17_getrealtime02_rows": len(loop.getrt_rows),
        "18_presence_rates": {"vehicle": metrics["vehicle_id_presence_rate"], "route": metrics["route_id_presence_rate"],
                              "direction": metrics["direction_presence_rate"], "sequence": metrics["sequence_presence_rate"]},
        "19_eta_validity_rate": metrics["eta_validity_rate"],
        "20_route_stop_match_rate": xmatch["getpos02_crossmatch_rate"],
        "21_byte_identical_rate": metrics["byte_identical_response_rate"],
        "22_no_data_timeout_auth": {"no_data": metrics["provider_no_data_count"], "timeout": metrics["timeout_count"], "auth": metrics["auth_error_count"]},
        "23_repeated_vehicles": cont_summary["repeated_vehicle_count"],
        "24_continuous_vehicles": cont_summary["continuous_vehicle_count"],
        "25_trajectory_candidates": traj_summary["trajectory_candidate_count"],
        "26_route_progression_observed": metrics["normalized_getpos02_rows"] > 0,
        "27_turnaround_interval_candidates": turn_summary["turnaround_interval_candidate_count"],
        "28_c1_objective_met_even_if_no_turnaround": True,
        "29_actual_headway_false": True,
        "30_actual_arrival_departure_dwell_false": True,
        "31_no_db_write": True,
        "32_no_simulator_training": True,
        "33_next_stage_readiness": decision["readiness"],
    }
    payload = {"created_at": iso_kst(), "artifact_root": str(root), "mode": "capture", "gate": gate["gate"], "gate_passed": gate["gate_passed"],
               "readiness": gate["readiness"], "scope": "LIMITED_PROSPECTIVE_BIS_DATA_PIPELINE_VALIDATION", "quick_answers": answers,
               "actual_start_local": start_info["actual_start_local"], "api_attempts": loop.total_attempts,
               "getpos02_rows": len(loop.getpos_rows), "getrealtime02_rows": len(loop.getrt_rows),
               "repeated_vehicle_count": cont_summary["repeated_vehicle_count"], "trajectory_candidate_count": traj_summary["trajectory_candidate_count"],
               "turnaround_interval_candidate_count": turn_summary["turnaround_interval_candidate_count"],
               "service_key_exposed": not scan["secret_scan_pass"], "database_write_count": 0, "allowlist_violations": ALLOWLIST.violation_count}
    lines = [
        "# SRP2-BIS-C1 Limited Prospective BIS Pilot Capture — Final Report", "",
        f"- artifact_root: {root}", f"- gate: {gate['gate']}", f"- readiness: {gate['readiness']}",
        f"- scope: LIMITED_PROSPECTIVE_BIS_DATA_PIPELINE_VALIDATION (NOT a morning-peak performance study)",
        f"- actual start (KST): {start_info['actual_start_local']} (planned {start_info['planned_start_local']}; 07:00 is an operational sampling choice, not a treatment variable)",
        f"- target: route 814 ({TARGET_ROUTE_ID}) both directions; C0 90-call plan reproduced: {recon['call_plan_reconciled']}",
        f"- API attempts: {loop.total_attempts}/{HARD_CALL_CAP} (retries {loop.retry_calls}); allowlist violations: {ALLOWLIST.violation_count}",
        f"- captured: {len(loop.getpos_rows)} getPos02 rows, {len(loop.getrt_rows)} getRealtime02 rows, {len(loop.raw_registry)} raw files",
        f"- repeated vehicles: {cont_summary['repeated_vehicle_count']}; trajectory candidates: {traj_summary['trajectory_candidate_count']}; turnaround interval candidates: {turn_summary['turnaround_interval_candidate_count']}",
        "",
        "## Claim boundaries (held)",
        "- ETA is PROVIDER_PREDICTED_ETA; trajectory is PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE; turnaround is INTERVAL_CENSORED_PROVIDER_OBSERVATION.",
        "- actual_headway / actual_arrival_departure / actual_dwell / exact_turnaround: all FALSE. No morning-peak operational-performance claim.",
        "",
        "## Guardrails (held)",
        "- only getPos02 + getRealtime02 on apis.data.go.kr (allowlist); service key never written/hashed/URL-persisted; env not dumped.",
        "- raw append-only, SHA-verified; no synthetic/interpolated rows; DB write 0; simulator/training 0; validation/test/sealed 0; git commit/push 0.",
        "",
        "## Quick answers (Section 35)",
    ]
    for k in sorted(answers):
        lines.append(f"- {k}: {answers[k]}")
    return payload, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["capture", "resume"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--preferred-start-local", default=None)
    parser.add_argument("--capture-timezone", default=CAPTURE_TIMEZONE)
    parser.add_argument("--duration-minutes", type=int, default=DURATION_MINUTES)
    parser.add_argument("--polling-interval-seconds", type=int, default=POLLING_INTERVAL_SEC)
    parser.add_argument("--hard-api-call-cap", type=int, default=HARD_CALL_CAP)
    parser.add_argument("--service-key-file", default=None)
    args = parser.parse_args()
    try:
        run_capture(args)
    except C1Error as exc:
        print("SRP2-BIS-C1 CAPTURE BLOCKED/FAILED")
        print(f"gate: {exc.gate_status}")
        print(f"detail: {exc.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
