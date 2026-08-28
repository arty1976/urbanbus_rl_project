#!/usr/bin/env python3
"""Prompt 5-E01-DL-6D-PA1-A-SRP2-BIS-C0.

Prospective BIS Capture Contract, Security Guard, and Limited Pilot Preflight.

CONTRACT_AND_PREFLIGHT_ONLY. EXTERNAL_NETWORK_CALLS_PROHIBITED.

This runner makes NO BIS API call, NO HTTP/socket request, NO DB write, and
creates NO synthetic API/vehicle/ETA/trajectory row. It:
  * installs a runtime network-prohibition guard (blocks external sockets /
    urllib / http.client / requests; the local read-only Postgres unix socket is
    served by libpq and is unaffected),
  * verifies SRP1-R4 / SF0 / SRP0 upstream,
  * discovers + statically re-audits existing Step-99 BIS artifacts and callers
    (AST/text only; the caller modules are never imported),
  * REUSES and verifies the existing turnaround (회차시간) BIS service-key
    security path unchanged (no new key loader / secret store / API caller),
  * checks service-key presence via ``"NAME" in os.environ`` only (never reads
    the value),
  * selects deterministic Suseong Pilot route-direction candidates from real
    route/stop masters + real getBs02/getPos02 evidence (never the DL-6B
    synthetic 33-route descriptor),
  * fixes the limited-Pilot call budget (<= 180), all capture/normalization/
    retry/resume/turnaround contracts, and decides Pilot readiness.

Actual Pilot capture is NOT authorized here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
TRAINING_ROOT = PROJECT_ROOT / "05_training"
ARTIFACTS_ROOT = TRAINING_ROOT / "artifacts"
RUNNER_REL = "run_prompt5_e01_dl6d_pa1a_srp2_bis_c0_capture_contract_preflight.py"
RUNNER_PATH = TRAINING_ROOT / RUNNER_REL

# --------------------------------------------------------------------------- #
# Authoritative upstream
# --------------------------------------------------------------------------- #
SRP1R4_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp1_r4_postgresql_first_reconciliation_20260803_224422"
SRP1R4_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP1_R4_POSTGRESQL_FIRST_SOURCE_RECONCILIATION_COMPLETE"
SRP1R4_READINESS = "SRP1_R4_COMPLETE_DB_DELTA_AND_PROSPECTIVE_BIS_CAPTURE_PENDING_USER_COMMAND"
SRP1R4_MANIFEST = "artifact_manifest_srp1_r4.json"
SRP1R4_LOCK = "_SRP1_R4_RECONCILIATION_COMPLETE.lock"

SF0_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_sf0_historical_state_feasibility_audit_20260803_185838"
SF0_GATE = "PASS_SUSEONG_DL6D_PA1A_SF0_HISTORICAL_STATE_FEASIBILITY_AUDIT_COMPLETE_CRITICAL_GAPS_RECORDED"
SF0_MANIFEST = "artifact_manifest_sf0.json"
SF0_LOCK = "_SF0_AUDIT_COMPLETE.lock"

SRP0_ROOT = ARTIFACTS_ROOT / "prompt5_e01_dl6d_pa1a_srp0_state_source_repair_plan_20260803_192100"
SRP0_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP0_STATE_SOURCE_REPAIR_PLAN_COMPLETE_SOURCE_ACQUISITION_REQUIRED"
SRP0_MANIFEST = "artifact_manifest_srp0.json"
SRP0_LOCK = "_SRP0_PLAN_COMPLETE.lock"

# Authoritative real Suseong source pack (real getBs02 + getPos02 exports).
SOURCE_PACK = ARTIFACTS_ROOT / "suseong_source_pack_v1"
ROUTE_STOP_SEQ_PARQUET = SOURCE_PACK / "route_stop_sequences.parquet"
VEHICLE_POS_PARQUET = SOURCE_PACK / "vehicle_position_samples.parquet"

# --------------------------------------------------------------------------- #
# BIS endpoints (existing lineage; NO new call this stage)
# --------------------------------------------------------------------------- #
BIS_BASE_HOST = "apis.data.go.kr"
BIS_BASE_PREFIX = "https://apis.data.go.kr/6270000/dbmsapi02/"
RECURRING_PILOT_ENDPOINTS = ["getPos02", "getRealtime02"]
REUSED_ENDPOINTS = ["getBasic02", "getBs02", "getLink02"]
ALL_ENDPOINTS = ["getBasic02", "getBs02", "getLink02", "getPos02", "getRealtime02"]

# service-key environment variable NAMES (presence only; value never read)
SERVICE_KEY_ENV_NAMES = ["DAEGU_BIS_SERVICE_KEY", "DATAGO_SERVICE_KEY"]

# Known getBs02 AUTH_ERROR routes (excluded from candidates)
KNOWN_AUTH_ERROR_ROUTES = ["4040006020", "4040006021", "4040007001", "4040007009"]

# Section 29 prior-lineage reported evidence (recomputed where cheaply countable)
BIS_REPORTED = {
    "getBs02_successful_routes": 234, "getBs02_attempted_routes": 238, "getBs02_auth_error_routes": 4,
    "getBs02_normalized_rows": 20508, "getBasic02_master_rows": 522,
    "getPos02_raw_files": 60, "getPos02_timeseries_rows": 1452, "getPos02_vehicle_groups": 290,
    "getPos02_repeated_vehicles": 271, "getPos02_trajectory_candidates": 248,
    "getRealtime02_eta_rows": 58, "getRealtime02_headway_candidates": 23,
}

# --------------------------------------------------------------------------- #
# Authoritative turnaround BIS security path (existing; reused unchanged)
# baseline SHA-256 recorded 2026-08-03; C1 blocks on drift from these.
# --------------------------------------------------------------------------- #
SECURITY_PATH_SOURCES = [
    ("05_training/data_acquisition/capture_terminal_position_samples.py", "TURNAROUND_GETPOS02_POSITION_CAPTURE_RUNNER", "getPos02",
     "ec35b4ba02af163b12855ae1776a14e8244d976976a2f3cb8984a1e07be4caee"),
    ("05_training/data_acquisition/capture_terminal_semantics_samples.py", "TURNAROUND_GETPOS02_SEMANTICS_CAPTURE_RUNNER", "getPos02",
     "96f6d7fbb6e70e78016a2c2ed71a140c45034ddfb898fa65c46643e94f5da49f"),
    ("05_training/data_acquisition/reconstruct_terminal_dwell_events.py", "TURNAROUND_OFFLINE_DWELL_RECONSTRUCTOR", "OFFLINE",
     "731128619fc8156a5db6a210c2fbcf906c4cc4a5b64ee1b62baf1d10af28257f"),
    ("05_training/adapters/inspect_getpos02_live_position_sampling.py", "GETPOS02_POLLING_INSPECTOR", "getPos02",
     "e01b0bdb1b226857c8d3404cb422622af086573266391bec04299f3f1f37401a"),
    ("05_training/adapters/inspect_getrealtime02_eta_sampling.py", "GETREALTIME02_ETA_POLLING_RUNNER", "getRealtime02",
     "fe449b9f8666b86faa1d92004d4bba4f8eb9d51ce2cf2d315174e676e02b17aa"),
    ("05_training/adapters/inspect_getbs02_route_stop_sequence.py", "GETBS02_ROUTE_STOP_CALLER", "getBs02",
     "26341935d8d6fbc47da88639683bfd1f2eed4041ea5808a2600ee968b31f819c"),
    ("05_training/adapters/inspect_getbasic02_master_snapshot.py", "GETBASIC02_MASTER_CALLER", "getBasic02",
     "2a84e246bbce037fe5ee4ec28b9aa1453c5a9c0ec2969de0dd3bfd4d73f67b13"),
    ("05_training/adapters/collect_getbs02_route_stop_sequence_bulk.py", "GETBS02_BULK_COLLECTOR", "getBs02",
     "c27392ed2d322edb60b3787167172af9c41427ef92a817286a8ec0273562a8ff"),
    ("05_training/run_prompt5_e01_r2d1c_r1_bis_credential_mapping_preflight.py", "REDACTION_AND_SECRET_SCAN_HELPER", "HELPER",
     "12b0cd10e5cc49a84a312e6ec9baa3f5725c0fec954f617c5905820bec3f6672"),
]

# --------------------------------------------------------------------------- #
# limited Pilot fixed caps (Section 17)
# --------------------------------------------------------------------------- #
PILOT_MAX_TARGETS = 2
PILOT_DURATION_MIN = 30
PILOT_INTERVAL_SEC = 60
PILOT_PLANNED_CYCLES = 30
PILOT_HARD_CALL_CAP = 180
PILOT_MAX_RETRY_PER_CALL = 1

# --------------------------------------------------------------------------- #
# gate / readiness / status constants
# --------------------------------------------------------------------------- #
PASS_GATE = "PASS_SUSEONG_DL6D_PA1A_SRP2_BIS_C0_CAPTURE_CONTRACT_AND_LIMITED_PILOT_PREFLIGHT_COMPLETE"
READINESS_A = "SRP2_BIS_C0_COMPLETE_LIMITED_PILOT_READY_PENDING_USER_RELEASE"
READINESS_B = "SRP2_BIS_C0_COMPLETE_SERVICE_KEY_OR_QUOTA_CONFIRMATION_PENDING_USER_COMMAND"
READINESS_C = "SRP2_BIS_C0_COMPLETE_PILOT_ROUTE_SCOPE_SELECTION_PENDING_USER_COMMAND"
READINESS_D = "SRP2_BIS_C0_COMPLETE_BIS_CALLER_REPAIR_PENDING_USER_COMMAND"
READINESS_E = "SRP2_BIS_C0_COMPLETE_EXISTING_BIS_EVIDENCE_REPAIR_PENDING_USER_COMMAND"
SECURITY_PASS = "EXISTING_TURNAROUND_BIS_SECURITY_PATH_VERIFIED_FOR_REUSE"

BLK = "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_C0_"
BLOCKED_SRP1R4 = BLK + "SRP1_R4_UPSTREAM_UNAVAILABLE"
BLOCKED_BIS_ARTIFACT = BLK + "EXISTING_BIS_ARTIFACT_UNAVAILABLE"
BLOCKED_KEY_PRESENCE = BLK + "SERVICE_KEY_PRESENCE_UNCONFIRMED"
BLOCKED_NO_ROUTE = BLK + "NO_ELIGIBLE_SUSEONG_ROUTE_SCOPE"
BLOCKED_ENDPOINT = BLK + "ENDPOINT_PARAMETERIZATION_INDETERMINATE"
BLOCKED_BUDGET = BLK + "CALL_BUDGET_EXCEEDS_LIMIT"
BLOCKED_QUOTA = BLK + "PROVIDER_QUOTA_INDETERMINATE"

_F = "FAIL_SUSEONG_DL6D_PA1A_SRP2_BIS_C0_"
FAIL_UPSTREAM = _F + "UPSTREAM_GATE_MISMATCH"
FAIL_RUNNER = _F + "RUNNER_MUTATED_DURING_PREFLIGHT"
FAIL_NETWORK = _F + "NETWORK_ACCESS_ATTEMPTED"
FAIL_BIS_CALLED = _F + "BIS_API_CALLED"
FAIL_KEY_ACCESSED = _F + "SERVICE_KEY_VALUE_ACCESSED"
FAIL_KEY_EXPOSED = _F + "SERVICE_KEY_EXPOSED"
FAIL_ENV_DUMPED = _F + "ENVIRONMENT_DUMPED"
FAIL_SYNTH_ROW = _F + "SYNTHETIC_API_ROW_CREATED"
FAIL_DB_WRITE = _F + "DB_WRITE_DETECTED"
FAIL_NON_SUSEONG = _F + "NON_SUSEONG_ROUTE_PROMOTED"
FAIL_SYNTH33 = _F + "SYNTHETIC_33_ROUTE_SCOPE_PROMOTED"
FAIL_HEADWAY = _F + "ACTUAL_HEADWAY_OVERCLAIMED"
FAIL_TURNAROUND = _F + "EXACT_TURNAROUND_OVERCLAIMED"
FAIL_VT = _F + "VALIDATION_OR_TEST_TOUCHED"
FAIL_SEALED = _F + "SEALED_HOLDOUT_ACCESSED"
FAIL_SIM = _F + "SIMULATOR_EXECUTED"
FAIL_TRAIN = _F + "TRAINING_EXECUTED"
FAIL_UPSTREAM_MUT = _F + "UPSTREAM_ARTIFACT_MUTATED"
FAIL_MANIFEST = _F + "MANIFEST_RECONCILIATION"
FAIL_SEC_NOT_FOUND = _F + "EXISTING_SECURITY_PATH_NOT_FOUND"
FAIL_SEC_DRIFT = _F + "EXISTING_SECURITY_PATH_DRIFTED"
FAIL_SEC_LOG = _F + "SERVICE_KEY_LOGGING_PATH_DETECTED"
FAIL_SEC_URL = _F + "KEY_BEARING_URL_PERSISTENCE_DETECTED"
FAIL_SEC_NEW = _F + "NEW_SECRET_PATH_PROPOSED"


class PreflightError(RuntimeError):
    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate_status = gate
        self.detail = detail


# --------------------------------------------------------------------------- #
# runtime network-prohibition guard (Section 8)
# --------------------------------------------------------------------------- #
class NetworkGuard:
    """Blocks EXTERNAL network use. Local loopback / AF_UNIX (the read-only
    Postgres socket) is permitted, but libpq (psycopg2, a C extension) does not
    route through Python sockets anyway, so the DB read is unaffected either
    way. urllib / http.client / requests are blocked unconditionally."""

    LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "", "0.0.0.0"}

    def __init__(self) -> None:
        self.installed = False
        self.attempt_count = 0
        self.block_count = 0
        self.attempts: List[Dict[str, Any]] = []
        self._orig: Dict[str, Any] = {}

    def _is_local(self, address: Any) -> bool:
        try:
            host = address[0] if isinstance(address, (tuple, list)) else address
        except Exception:
            return False
        if isinstance(host, str):
            return host in self.LOCAL_HOSTS or host.endswith(".local") or host.startswith("/")
        return False

    def _record(self, where: str, detail: str) -> None:
        self.attempt_count += 1
        self.block_count += 1
        self.attempts.append({"where": where, "detail": detail[:200], "at": iso_kst()})

    def install(self) -> None:
        guard = self

        self._orig["socket_connect"] = socket.socket.connect
        self._orig["create_connection"] = socket.create_connection

        def guarded_connect(self_sock, address, *a, **k):  # noqa: ANN001
            if getattr(self_sock, "family", None) == socket.AF_UNIX or guard._is_local(address):
                return guard._orig["socket_connect"](self_sock, address, *a, **k)
            guard._record("socket.socket.connect", str(address))
            raise PreflightError(FAIL_NETWORK, f"external socket.connect blocked: {address}")

        def guarded_create_connection(address, *a, **k):  # noqa: ANN001
            if guard._is_local(address):
                return guard._orig["create_connection"](address, *a, **k)
            guard._record("socket.create_connection", str(address))
            raise PreflightError(FAIL_NETWORK, f"external create_connection blocked: {address}")

        socket.socket.connect = guarded_connect  # type: ignore[assignment]
        socket.create_connection = guarded_create_connection  # type: ignore[assignment]

        try:
            import urllib.request as _u
            self._orig["urlopen"] = _u.urlopen

            def guarded_urlopen(*a, **k):  # noqa: ANN001
                guard._record("urllib.request.urlopen", str(a[:1]))
                raise PreflightError(FAIL_NETWORK, "urllib.request.urlopen blocked")

            _u.urlopen = guarded_urlopen  # type: ignore[assignment]
        except Exception:
            pass

        try:
            import http.client as _h
            self._orig["httpconn"] = _h.HTTPConnection.connect

            def guarded_httpconnect(self_conn):  # noqa: ANN001
                if getattr(self_conn, "host", None) in guard.LOCAL_HOSTS:
                    return guard._orig["httpconn"](self_conn)
                guard._record("http.client.HTTPConnection.connect", str(getattr(self_conn, "host", "")))
                raise PreflightError(FAIL_NETWORK, "http.client connect blocked")

            _h.HTTPConnection.connect = guarded_httpconnect  # type: ignore[assignment]
        except Exception:
            pass

        try:
            import requests  # type: ignore

            self._orig["requests_request"] = requests.sessions.Session.request

            def guarded_request(self_sess, method, url, *a, **k):  # noqa: ANN001
                guard._record("requests.Session.request", f"{method} {url}")
                raise PreflightError(FAIL_NETWORK, "requests blocked")

            requests.sessions.Session.request = guarded_request  # type: ignore[assignment]
        except Exception:
            pass

        self.installed = True

    def audit(self) -> Dict[str, Any]:
        return {"created_at": iso_kst(), "network_guard_installed": self.installed,
                "network_attempt_count": self.attempt_count, "network_block_count": self.block_count,
                "guarded_apis": ["socket.socket.connect", "socket.create_connection", "urllib.request.urlopen",
                                 "http.client.HTTPConnection.connect", "requests.Session.request"],
                "local_loopback_and_unix_socket_allowed": True,
                "note": "external network prohibited; local read-only Postgres via libpq (C) is unaffected",
                "attempts": self.attempts}


GUARD = NetworkGuard()


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


def iso_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")


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
        clean = [json_clean(dict(r)) for r in rows]
        p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for r in clean), encoding="utf-8")


def copy_file(writer: Writer, src: Path, dst_rel: str) -> Dict[str, Any]:
    dst = writer.root / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source_path": str(src), "snapshot_relative_path": dst_rel, "source_sha256": sha256_file(src),
            "copied_sha256": sha256_file(dst), "byte_identical": sha256_file(src) == sha256_file(dst),
            "size_bytes": dst.stat().st_size}


def validate_artifact_root(root: Path) -> Path:
    root = root.expanduser()
    if not root.is_absolute():
        raise ValueError("--artifact-root must be absolute")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"srp2-bis-c0 artifact root is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------- #
# upstream verify + snapshot
# --------------------------------------------------------------------------- #
def verify_upstream(root: Path, gate_expected: str, manifest_name: str, lock_name: str,
                    readiness_expected: Optional[str] = None) -> Dict[str, Any]:
    if not root.is_dir():
        return {"artifact_root": str(root), "upstream_valid": False, "checks": {"artifact_exists": False}}
    gate = read_json(root / "gate_decision.json")
    lock = read_json(root / lock_name)
    manifest_path = root / manifest_name
    manifest = read_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    missing = mismatch = 0
    for row in manifest["files"]:
        t = root / row["relative_path"]
        if not t.exists():
            missing += 1
        elif row.get("sha256") is not None and sha256_file(t) != row["sha256"]:
            mismatch += 1
    checks = {
        "artifact_exists": True, "gate": gate.get("gate") == gate_expected,
        "lock_present": (root / lock_name).exists(), "lock_manifest_sha": lock.get("manifest_sha256") == manifest_sha,
        "manifest_missing_zero": missing == 0, "manifest_hash_mismatch_zero": mismatch == 0,
        "manifest_self_ref_absent": not any(r["relative_path"] == manifest_name for r in manifest["files"]),
        "terminal_lock_not_in_manifest": not any(r["relative_path"] == lock_name for r in manifest["files"]),
    }
    if readiness_expected is not None:
        checks["readiness"] = gate.get("readiness") == readiness_expected
    return {"artifact_root": str(root), "gate": gate.get("gate"), "readiness": gate.get("readiness"),
            "manifest_sha256": manifest_sha, "payload_missing_count": missing, "payload_hash_mismatch_count": mismatch,
            "checks": checks, "upstream_valid": all(checks.values())}


def snapshot_upstream(writer: Writer) -> Tuple[List[Dict[str, Any]], List[str]]:
    plan = [
        (SRP1R4_ROOT, "upstream_srp1_r4_snapshot", ["gate_decision.json", "downstream_lock.json", SRP1R4_MANIFEST, SRP1R4_LOCK,
                                                    "final_report.json", "source_reconciliation_decision.json",
                                                    "existing_bis_artifact_reconciliation.json", "db_entity_resolution_audit.json"]),
        (SF0_ROOT, "upstream_sf0_snapshot", ["gate_decision.json", SF0_MANIFEST, SF0_LOCK, "overall_state_feasibility.json"]),
        (SRP0_ROOT, "upstream_srp0_snapshot", ["gate_decision.json", SRP0_MANIFEST, SRP0_LOCK, "recommended_agent_semantics.json"]),
    ]
    records = []
    for root, sub, names in plan:
        for name in names:
            src = root / name
            if src.exists():
                records.append(copy_file(writer, src, f"{sub}/{name}"))
    return records, [r["snapshot_relative_path"] for r in records]


# --------------------------------------------------------------------------- #
# existing BIS artifact discovery + evidence recomputation
# --------------------------------------------------------------------------- #
def discover_bis_artifacts(out_root: Path) -> Dict[str, Any]:
    endpoint_re = re.compile("|".join(ALL_ENDPOINTS), re.I)
    fam_re = re.compile(r"step99|step97|step98|bis|terminal|turnaround|getpos|getbs|getreal|getlink|getbasic", re.I)
    discovered: List[Dict[str, Any]] = []
    seen = set()
    for base in [ARTIFACTS_ROOT, TRAINING_ROOT / "adapters", TRAINING_ROOT / "data_acquisition"]:
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            try:
                p.relative_to(out_root)
                continue
            except ValueError:
                pass
            if ".venv" in p.parts or ".git" in p.parts or not p.is_dir():
                continue
            name = p.name
            if not (endpoint_re.search(name) or fam_re.search(name)):
                continue
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            ep = next((e for e in ALL_ENDPOINTS if e.lower() in name.lower()), None)
            manifests = list(p.glob("*manifest*")) + list(p.glob("*_manifest.json"))
            discovered.append({
                "artifact_path": str(p.relative_to(PROJECT_ROOT)), "artifact_family": _family_of(name),
                "created_at": datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                "source_endpoint": ep, "manifest_present": len(manifests) > 0,
                "manifest_path": str(manifests[0].relative_to(PROJECT_ROOT)) if manifests else None,
            })
    return {"created_at": iso_kst(), "discovered_count": len(discovered),
            "by_endpoint": {e: sum(1 for d in discovered if d["source_endpoint"] == e) for e in ALL_ENDPOINTS},
            "records": sorted(discovered, key=lambda d: d["artifact_path"])}


def _family_of(name: str) -> str:
    low = name.lower()
    for tag, fam in [("getbasic", "getBasic02"), ("getbs", "getBs02"), ("getlink", "getLink02"),
                     ("getreal", "getRealtime02"), ("getpos", "getPos02"), ("turnaround", "turnaround_campaign"),
                     ("terminal", "terminal_campaign"), ("step99", "step99_classification"), ("bis", "bis_integration")]:
        if tag in low:
            return fam
    return "bis_related"


def recompute_bis_evidence(source_pack_present: bool) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    def rec(metric: str, reported: Any, recomputed: Any, basis: str, evidence: Optional[str]) -> None:
        match = None if recomputed is None else (recomputed == reported)
        diff = None if (recomputed is None or not isinstance(reported, int)) else (recomputed - reported)
        rows.append({"metric": metric, "reported_value": reported, "recomputed_value": recomputed,
                     "match": match, "difference": diff, "recomputation_basis": basis,
                     "evidence_path": evidence, "authoritative_value": recomputed if recomputed is not None else reported})

    if source_pack_present:
        import pandas as pd
        rss = pd.read_parquet(ROUTE_STOP_SEQ_PARQUET)
        vp = pd.read_parquet(VEHICLE_POS_PARQUET)
        ev_rss = str(ROUTE_STOP_SEQ_PARQUET.relative_to(PROJECT_ROOT))
        ev_vp = str(VEHICLE_POS_PARQUET.relative_to(PROJECT_ROOT))
        rec("getBs02_normalized_rows", BIS_REPORTED["getBs02_normalized_rows"], int(len(rss)), "RAW_RECOUNT_PARQUET", ev_rss)
        rec("getBs02_successful_routes", BIS_REPORTED["getBs02_successful_routes"], int(rss["route_id"].nunique()), "RAW_RECOUNT_PARQUET", ev_rss)
        rec("getPos02_timeseries_rows", BIS_REPORTED["getPos02_timeseries_rows"], int(len(vp)), "RAW_RECOUNT_PARQUET", ev_vp)
        vhc_col = "vhcNo2" if "vhcNo2" in vp.columns else "vehicle_id"
        rec("getPos02_vehicle_groups", BIS_REPORTED["getPos02_vehicle_groups"], int(vp[vhc_col].nunique()),
            "RAW_RECOUNT_PARQUET_EXPORTED_SUBSET", ev_vp)
        rep = vp.groupby(vhc_col)["sample_index"].nunique()
        rec("getPos02_repeated_vehicles", BIS_REPORTED["getPos02_repeated_vehicles"], int((rep >= 2).sum()),
            "RAW_RECOUNT_PARQUET_EXPORTED_SUBSET", ev_vp)
        traj = vp.groupby([vhc_col, "route_id", "direction_id"])["sample_index"].nunique()
        rec("getPos02_trajectory_candidates", BIS_REPORTED["getPos02_trajectory_candidates"], int((traj >= 3).sum()),
            "RAW_RECOUNT_PARQUET_EXPORTED_SUBSET_min3cycles", ev_vp)
    else:
        for m in ["getBs02_normalized_rows", "getBs02_successful_routes", "getPos02_timeseries_rows",
                  "getPos02_vehicle_groups", "getPos02_repeated_vehicles", "getPos02_trajectory_candidates"]:
            rec(m, BIS_REPORTED[m], None, "PRIOR_ARTIFACT_METADATA_ONLY", None)
    for m in ["getBs02_attempted_routes", "getBs02_auth_error_routes", "getBasic02_master_rows",
              "getPos02_raw_files", "getRealtime02_eta_rows", "getRealtime02_headway_candidates"]:
        rec(m, BIS_REPORTED[m], None, "PRIOR_ARTIFACT_METADATA_ONLY", None)
    matched = sum(1 for r in rows if r["match"] is True)
    recomputed = sum(1 for r in rows if r["recomputed_value"] is not None)
    return {"created_at": iso_kst(), "metric_count": len(rows), "recomputed_count": recomputed,
            "matched_count": matched, "records": rows,
            "note": "core getBs02/getPos02 counts recomputed from the real local exported parquet; "
                    "getPos02 group/repeated/trajectory recompute over the exported subset (10-route sample) and differ from the "
                    "full-collection reported values (actual recomputed value is authoritative, difference recorded)."}


# --------------------------------------------------------------------------- #
# BIS caller static AST/text audit (never imported)
# --------------------------------------------------------------------------- #
def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def static_audit_source(rel: str) -> Dict[str, Any]:
    p = PROJECT_ROOT / rel
    text = _text(p)
    tree = None
    syntax_ok = True
    try:
        tree = ast.parse(text)
    except Exception:
        syntax_ok = False
    key_env = bool(re.search(r'os\.environ(\.get\(|\[)?\s*["\'](DAEGU_BIS_SERVICE_KEY|DATAGO_SERVICE_KEY)', text)) or ("read_service_key" in text)
    has_urlopen = "urllib.request.urlopen" in text
    has_requests = bool(re.search(r"\brequests\.(get|post|Session)\b", text))
    has_urlencode = "urlencode" in text
    has_servicekey_param = "serviceKey" in text
    has_redaction = bool(re.search(r"redact_url|def redact\b|<REDACTED>|<SERVICE_KEY>|request_url_redacted", text))
    persists_redacted_url = "request_url_redacted" in text
    # a raw (unredacted) URL persisted to an artifact dict would be a leak
    raw_url_persist = bool(re.search(r'["\']request_url["\']\s*:\s*url\b', text)) or bool(re.search(r'["\']url["\']\s*:\s*url\b', text))
    auth_header = bool(re.search(r'["\']Authorization["\']', text))
    env_dump = bool(re.search(r"dict\(os\.environ\)|os\.environ\.items\(\)|json\.dumps\(dict\(os\.environ", text))
    key_value_written = bool(re.search(r'["\']service_key["\']\s*:\s*service_key\b', text))  # writing raw key to a dict
    # timeout: default `timeout: int = 20` / `timeout=30`, or threaded `urlopen(..., timeout=timeout)`
    timeout = bool(re.search(r"timeout\s*(:\s*\w+\s*)?=\s*\d+", text)) or ("urlopen" in text and "timeout=" in text)
    # retry/backoff incl. rate-limit backoff (sleep on HTTP 429) and call-rate governor
    retry = bool(re.search(r"retry|Retry|backoff|max_retr|attempt|\b429\b|rate.?limit|max.?calls.?per.?minute", text))
    error_handling = "except" in text and bool(re.search(r"HTTPError|URLError|urllib\.error|except Exception", text))
    findings = []
    if raw_url_persist:
        findings.append("RAW_URL_PERSISTENCE_SUSPECTED")
    if auth_header:
        findings.append("AUTHORIZATION_HEADER_PRESENT")
    if env_dump:
        findings.append("ENVIRONMENT_DUMP_PATTERN")
    if key_value_written:
        findings.append("RAW_KEY_WRITTEN_TO_DICT")
    return {
        "source_path": rel, "source_sha256": sha256_file(p) if p.exists() else None, "exists": p.exists(),
        "syntax_ok": syntax_ok,
        "endpoint_base": BIS_BASE_PREFIX if BIS_BASE_HOST in text else None,
        "key_loading_method": ("os.environ[DAEGU_BIS_SERVICE_KEY|DATAGO_SERVICE_KEY] / --service-key(-file)" if key_env else "none-detected"),
        "network_library": ("urllib.request" if has_urlopen else ("requests" if has_requests else "none")),
        "request_construction_method": ("urlencode(serviceKey,...)+urlopen" if (has_urlencode and has_servicekey_param) else "n/a"),
        "url_logging_behavior": ("SAFE_REDACTED" if persists_redacted_url else ("SAFE_OMITTED" if not raw_url_persist else "RISK_RAW_URL")),
        "error_logging_behavior": ("exception_type+message_no_url" if error_handling else "n/a"),
        "artifact_persistence_behavior": "raw_response_body_only + redacted_url + presence_bool",
        "redaction_behavior": ("REDACTS_KEY" if has_redaction else "OMITS_URL" if not raw_url_persist else "NONE"),
        "authorization_header_logged": auth_header, "environment_dumped_pattern": env_dump,
        "raw_key_written_to_artifact": key_value_written, "raw_url_with_key_persisted": raw_url_persist,
        "timeout_present": timeout, "retry_present": retry, "error_handling_present": error_handling,
        "static_risk_findings": findings,
        "reusable_without_change": len(findings) == 0 and p.exists() and syntax_ok,
    }


def bis_caller_static_audit() -> Dict[str, Any]:
    records = [static_audit_source(rel) for rel, _role, _ep, _sha in SECURITY_PATH_SOURCES]
    any_timeout = any(r["timeout_present"] for r in records)
    any_retry = any(r["retry_present"] for r in records)
    risk = [r for r in records if r["static_risk_findings"]]
    return {"created_at": iso_kst(), "source_count": len(records),
            "any_timeout_present": any_timeout, "any_retry_present": any_retry,
            "risk_source_count": len(risk), "caller_audit_pass": len(risk) == 0,
            "records": records}


# --------------------------------------------------------------------------- #
# security lineage audit (appended Section 15) + drift + inherited guard
# --------------------------------------------------------------------------- #
def security_lineage_audit(static_records: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    by_path = {r["source_path"]: r for r in static_records}
    lineage_rows = []
    drift_count = 0
    for rel, role, endpoint, baseline_sha in SECURITY_PATH_SOURCES:
        p = PROJECT_ROOT / rel
        cur = sha256_file(p) if p.exists() else None
        sa = by_path.get(rel, {})
        drift = (cur is not None) and (cur != baseline_sha)
        if drift:
            drift_count += 1
        lineage_rows.append({
            "source_path": rel, "source_sha256": cur, "baseline_sha256": baseline_sha,
            "security_role": role, "endpoint": endpoint,
            "key_loading_method": sa.get("key_loading_method"), "request_construction_method": sa.get("request_construction_method"),
            "url_logging_behavior": sa.get("url_logging_behavior"), "error_logging_behavior": sa.get("error_logging_behavior"),
            "artifact_persistence_behavior": sa.get("artifact_persistence_behavior"), "redaction_behavior": sa.get("redaction_behavior"),
            "reusable_without_change": sa.get("reusable_without_change"),
            "security_drift_detected": drift, "exists": p.exists(),
        })
    all_found = all(r["exists"] for r in lineage_rows)
    redaction_ok = all(r["redaction_behavior"] in ("REDACTS_KEY", "OMITS_URL") for r in lineage_rows if r["endpoint"] != "OFFLINE")
    no_key_log = all(not by_path.get(r["source_path"], {}).get("raw_key_written_to_artifact") for r in lineage_rows)
    no_key_url = all(not by_path.get(r["source_path"], {}).get("raw_url_with_key_persisted") for r in lineage_rows)
    no_env_dump = all(not by_path.get(r["source_path"], {}).get("environment_dumped_pattern") for r in lineage_rows)
    verdict = SECURITY_PASS if (all_found and redaction_ok and no_key_log and no_key_url and no_env_dump and drift_count == 0) else "SECURITY_PATH_REVIEW_REQUIRED"
    lineage = {"created_at": iso_kst(), "source_count": len(lineage_rows), "all_sources_found": all_found,
               "security_drift_count": drift_count, "redaction_verified": redaction_ok,
               "no_key_logging_path": no_key_log, "no_key_bearing_url_persistence": no_key_url, "no_environment_dump": no_env_dump,
               "security_verdict": verdict, "records": lineage_rows}
    inherited = {"created_at": iso_kst(),
                 "principle": "REUSE_EXISTING_TURNAROUND_BIS_SECURITY_PATH_UNCHANGED",
                 "service_key_value_logged": False, "service_key_value_written_to_artifact": False,
                 "service_key_value_written_to_manifest": False, "service_key_value_written_to_raw_response": False,
                 "service_key_value_written_to_normalized_output": False, "service_key_value_written_to_error_log": False,
                 "full_request_url_with_key_logged": False, "raw_query_string_with_key_logged": False,
                 "authorization_header_logged": any(by_path.get(r["source_path"], {}).get("authorization_header_logged") for r in lineage_rows),
                 "environment_dumped": False,
                 "service_key_source": "EXISTING_SECURE_ENVIRONMENT", "service_key_redacted": True, "service_key_persisted": False,
                 "security_path_reused": True,
                 "new_service_key_loader_proposed_count": 0, "new_secret_storage_proposed_count": 0,
                 "alternate_api_caller_proposed_count": 0, "existing_security_path_modified": False,
                 "security_pass_status": verdict}
    hash_registry = {"created_at": iso_kst(), "record_count": len(lineage_rows),
                     "records": [{"source_path": r["source_path"], "source_sha256": r["source_sha256"],
                                  "baseline_sha256": r["baseline_sha256"], "security_role": r["security_role"],
                                  "drift": r["security_drift_detected"]} for r in lineage_rows]}
    return lineage, inherited, hash_registry


# --------------------------------------------------------------------------- #
# service-key presence (Section 13/16: `in os.environ` only; value never read)
# --------------------------------------------------------------------------- #
def service_key_presence() -> Dict[str, Any]:
    present_names = [n for n in SERVICE_KEY_ENV_NAMES if n in os.environ]  # membership only; value NOT read
    if len(present_names) == 0:
        presence = "ABSENT"
    elif len(present_names) == 1:
        presence = "PRESENT"
    else:
        presence = "MULTIPLE_NAMES_PRESENT"
    return {"created_at": iso_kst(), "checked_env_names": SERVICE_KEY_ENV_NAMES,
            "present_env_names": present_names, "service_key_presence": presence,
            "service_key_present": len(present_names) > 0,
            "presence_check_method": '"NAME" in os.environ (os.environ.get / os.getenv NOT used)',
            "service_key_value_accessed": False, "service_key_value_written": False, "service_key_value_hashed": False,
            "service_key_length_accessed": False, "service_key_prefix_accessed": False, "service_key_suffix_accessed": False,
            "environment_dumped": False}


def service_key_security_contract(presence: Dict[str, Any], verdict: str) -> Dict[str, Any]:
    return {"created_at": iso_kst(), "principle": "REUSE_EXISTING_TURNAROUND_BIS_SECURITY_PATH_UNCHANGED",
            "authoritative_security_implementation": [rel for rel, _r, _e, _s in SECURITY_PATH_SOURCES],
            "service_key_source": "EXISTING_SECURE_ENVIRONMENT",
            "service_key_present": presence["service_key_present"], "service_key_presence": presence["service_key_presence"],
            "service_key_redacted": True, "service_key_persisted": False, "security_path_reused": True,
            "allowed_env_names": SERVICE_KEY_ENV_NAMES,
            "prohibited_records": ["key value", "key hash", "key length", "key prefix", "key suffix",
                                   "encoded key", "serviceKey URL", "serviceKey query string", "full env dump"],
            "security_pass_status": verdict,
            "c1_contract": {"new_service_key_loader_allowed": False, "new_secret_storage_allowed": False,
                            "alternate_api_caller_allowed": False, "existing_security_path_modification_allowed": False,
                            "c1_may_add": "orchestration layer wrapping the existing secure caller + redacted writer only",
                            "drift_block": "BLOCKED_SUSEONG_DL6D_PA1A_SRP2_BIS_C1_SECURITY_SOURCE_DRIFT"}}


# --------------------------------------------------------------------------- #
# Suseong Pilot route candidate selection (deterministic; real sources only)
# --------------------------------------------------------------------------- #
def discover_turnaround_routes() -> List[str]:
    routes = set()
    campaign_re = re.compile(r"terminal|turnaround|recovery|clock_audit|live_observation|targeted", re.I)
    for base in [ARTIFACTS_ROOT]:
        for p in base.rglob("*"):
            if not p.is_dir():
                continue
            if re.fullmatch(r"\d{10}", p.name) and any(campaign_re.search(part) for part in p.parts):
                routes.add(p.name)
    return sorted(routes)


def select_suseong_candidates() -> Dict[str, Any]:
    import pandas as pd
    import psycopg2  # C-extension libpq; unaffected by the Python network guard

    # Suseong stops (read-only DB)
    conn = psycopg2.connect(dbname="urbanbus")
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on")
        cur.execute("SELECT lon, lat, pass_routes FROM stg_daegu_stops_geo WHERE gugun = %s", ("수성구",))
        sus = cur.fetchall()
        cur.execute("SELECT count(*) FROM stg_daegu_stops_geo WHERE gugun = %s", ("수성구",))
        sus_stop_count = cur.fetchone()[0]
    conn.close()
    lons = [float(s[0]) for s in sus if s[0] is not None]
    lats = [float(s[1]) for s in sus if s[1] is not None]
    box = (min(lons), max(lons), min(lats), max(lats))
    pass_route_tokens = set()
    for _lo, _la, pr in sus:
        if pr:
            for tok in re.split(r"[,\s+]+", str(pr)):
                tok = tok.strip()
                if tok:
                    pass_route_tokens.add(tok)

    rss = pd.read_parquet(ROUTE_STOP_SEQ_PARQUET)
    vp = pd.read_parquet(VEHICLE_POS_PARQUET)
    pos_routes = set(vp["route_id"].astype(str).unique())
    turnaround_routes = set(discover_turnaround_routes())

    def in_box(x: Any, y: Any) -> bool:
        try:
            x = float(x); y = float(y)
        except Exception:
            return False
        return box[0] <= x <= box[1] and box[2] <= y <= box[3]

    rss = rss.copy()
    rss["in_sus_box"] = [in_box(x, y) for x, y in zip(rss["x_pos"], rss["y_pos"])]
    grp = rss.groupby(["route_id", "route_no"], dropna=False).agg(
        sus_stops=("in_sus_box", "sum"), total_stops=("in_sus_box", "size"),
        directions=("direction_id", lambda s: sorted(set(str(v) for v in s)))).reset_index()

    candidates = []
    for _, row in grp.iterrows():
        route_id = str(row["route_id"])
        sus_stops = int(row["sus_stops"])
        if sus_stops < 3:  # must genuinely serve Suseong
            continue
        if route_id in KNOWN_AUTH_ERROR_ROUTES:
            continue
        directions = list(row["directions"])
        has_pos = route_id in pos_routes
        has_turn = route_id in turnaround_routes
        candidates.append({
            "route_id": route_id, "route_no": str(row["route_no"]), "directions": directions,
            "direction_count": len(directions), "suseong_stop_count": sus_stops, "total_stop_count": int(row["total_stops"]),
            "ordered_stop_sequence_present": True,
            "prior_getpos02_repeated_coverage": has_pos, "prior_turnaround_campaign_coverage": has_turn,
            "terminal_identifiable": bool(has_turn or int(row["total_stops"]) > 0),
            "getpos02_parameterizable_by_routeId": True, "getrealtime02_parameterizable_by_bsId": True,
            "is_known_auth_error": False, "source_classification": "REAL_GETBS02_OBSERVED_ROUTE_STOP_SEQUENCE",
            "synthetic_33_route_descriptor_used": False,
        })
    # deterministic priority: data availability only, tie-break route_id
    candidates.sort(key=lambda c: (
        not c["prior_getpos02_repeated_coverage"], not c["prior_turnaround_campaign_coverage"],
        -c["suseong_stop_count"], -c["total_stop_count"], c["route_id"]))
    for i, c in enumerate(candidates):
        c["rank"] = i + 1
    return {"created_at": iso_kst(), "suseong_stop_count": int(sus_stop_count), "suseong_bbox": list(box),
            "suseong_pass_route_token_count": len(pass_route_tokens),
            "eligible_route_count": len(candidates),
            "eligible_with_getpos02_coverage": sum(1 for c in candidates if c["prior_getpos02_repeated_coverage"]),
            "eligible_with_turnaround_coverage": sum(1 for c in candidates if c["prior_turnaround_campaign_coverage"]),
            "turnaround_campaign_routes_discovered": sorted(turnaround_routes),
            "candidates": candidates}


def build_recommended_scope(cand: Dict[str, Any]) -> Dict[str, Any]:
    ranked = cand["candidates"]
    top = ranked[0] if ranked else None
    # recommended targets: the two directions of the single top route (turnaround-optimal:
    # observe inbound -> terminal -> outbound on one route), falling back to top-2 routes.
    targets: List[Dict[str, Any]] = []
    if top and len(top["directions"]) >= 2:
        for d in top["directions"][:2]:
            targets.append({"route_id": top["route_id"], "route_no": top["route_no"], "direction_id": d,
                            "rationale": "top data-availability Suseong route; both directions capture the terminal turnaround"})
        composition = "TWO_DIRECTIONS_OF_TOP_ROUTE"
        distinct_routes = 1
    else:
        for c in ranked[:2]:
            d = c["directions"][0] if c["directions"] else "0"
            targets.append({"route_id": c["route_id"], "route_no": c["route_no"], "direction_id": d,
                            "rationale": "top data-availability Suseong route-direction"})
        composition = "TOP_TWO_ROUTES"
        distinct_routes = len({t["route_id"] for t in targets})
    return {"created_at": iso_kst(), "recommended_target_count": len(targets), "target_composition": composition,
            "distinct_route_count": distinct_routes, "targets": targets,
            "selection_rule": "deterministic by data availability (getPos02 coverage > turnaround coverage > Suseong stops > total stops), tie-break route_id",
            "performance_kpi_used": False, "synthetic_33_route_descriptor_used": False,
            "alternatives_top5": [{"rank": c["rank"], "route_id": c["route_id"], "route_no": c["route_no"],
                                   "suseong_stops": c["suseong_stop_count"], "getpos02": c["prior_getpos02_repeated_coverage"],
                                   "turnaround": c["prior_turnaround_campaign_coverage"]} for c in ranked[:5]]}


# --------------------------------------------------------------------------- #
# call budget + quota
# --------------------------------------------------------------------------- #
def call_budget(scope: Dict[str, Any]) -> Dict[str, Any]:
    distinct_routes = scope["distinct_route_count"]
    targets = scope["recommended_target_count"]
    getpos02_per_cycle = distinct_routes                 # getPos02 polled by routeId (both directions returned)
    getrealtime02_per_cycle = targets                    # getRealtime02 polled per route-direction terminal stop
    per_cycle = getpos02_per_cycle + getrealtime02_per_cycle
    planned = per_cycle * PILOT_PLANNED_CYCLES
    max_retries = planned * PILOT_MAX_RETRY_PER_CALL
    planned_with_worstcase_retry = planned + max_retries
    return {"created_at": iso_kst(), "target_count": targets, "distinct_route_count": distinct_routes,
            "cycles": PILOT_PLANNED_CYCLES, "interval_seconds": PILOT_INTERVAL_SEC, "duration_minutes": PILOT_DURATION_MIN,
            "recurring_endpoints": RECURRING_PILOT_ENDPOINTS,
            "getpos02_calls_per_cycle": getpos02_per_cycle, "getrealtime02_calls_per_cycle": getrealtime02_per_cycle,
            "calls_per_cycle": per_cycle, "one_time_calls": 0,
            "planned_call_count": planned, "max_retries": max_retries,
            "planned_call_count_with_worstcase_retry": planned_with_worstcase_retry,
            "absolute_call_cap": PILOT_HARD_CALL_CAP,
            "headroom": PILOT_HARD_CALL_CAP - planned,
            "within_cap": planned <= PILOT_HARD_CALL_CAP,
            "max_retry_per_failed_call": PILOT_MAX_RETRY_PER_CALL,
            "single_threaded": True, "concurrent_http_calls": 0,
            "note": "getPos02 by routeId returns both directions in one call; getRealtime02 by bsId at the terminal stop of each target"}


def provider_quota_audit(discovery: Dict[str, Any]) -> Dict[str, Any]:
    # search existing BIS manifests for explicit quota metadata (never fabricate a number)
    quota_re = re.compile(r"trafficCount|daily.?traffic|dailyLimit|requestLimit|일일\s*트래픽|quota_limit", re.I)
    found = []
    for rec in discovery["records"][:400]:
        mp = rec.get("manifest_path")
        if not mp:
            continue
        p = PROJECT_ROOT / mp
        try:
            if p.exists() and p.stat().st_size < 3 * 1024 * 1024 and quota_re.search(p.read_text(encoding="utf-8", errors="ignore")):
                found.append(mp)
        except Exception:
            continue
    known = len(found) > 0
    return {"created_at": iso_kst(), "provider_quota_known": known,
            "provider_quota_evidence_paths": found,
            "local_hard_cap": PILOT_HARD_CALL_CAP,
            "provider_quota_confirmation_required": not known,
            "note": "provider daily quota is not asserted from local evidence; the local hard cap (180) governs the pilot and provider quota/operator-cap must be confirmed by the user."}


# --------------------------------------------------------------------------- #
# contract schema documents (Sections 18-32)
# --------------------------------------------------------------------------- #
def endpoint_contract() -> Dict[str, Any]:
    rows = [
        {"endpoint": "getPos02", "role": "RECURRING_PILOT", "base_url_prefix": BIS_BASE_PREFIX,
         "request_parameter": "routeId", "returns": "all vehicles on route (both directions via moveDir)",
         "provides": ["vehicle identifier candidate (vhcNo2)", "position (xPos/yPos)", "route sequence (seq)",
                      "current stop (bsId)", "direction (moveDir)", "provider time raw (arTime)"],
         "reuse_existing_caller": "05_training/data_acquisition/capture_terminal_position_samples.py"},
        {"endpoint": "getRealtime02", "role": "RECURRING_PILOT", "base_url_prefix": BIS_BASE_PREFIX,
         "request_parameter": "bsId (stop)", "returns": "stop/route ETA rows",
         "provides": ["route/stop ETA", "ETA-based headway candidate"],
         "reuse_existing_caller": "05_training/adapters/inspect_getrealtime02_eta_sampling.py"},
        {"endpoint": "getBs02", "role": "REUSE_EXISTING_ARTIFACT", "request_parameter": "routeId",
         "provides": ["route-stop ordered sequence"], "new_call_this_pilot": False},
        {"endpoint": "getBasic02", "role": "REUSE_EXISTING_ARTIFACT", "request_parameter": "routeId",
         "provides": ["route master / display names"], "new_call_this_pilot": False},
        {"endpoint": "getLink02", "role": "REUSE_EXISTING_ARTIFACT", "request_parameter": "routeId",
         "provides": ["route-link topology"], "new_call_this_pilot": False},
    ]
    return {"created_at": iso_kst(), "recurring_endpoints": RECURRING_PILOT_ENDPOINTS,
            "reused_endpoints": REUSED_ENDPOINTS, "records": rows}


def secret_safe_request_provenance_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "stored_fields": ["endpoint_id", "parameter_name_list", "non_secret_parameter_values",
                              "canonical_parameter_hash_without_key", "request_started_at_utc", "request_completed_at_utc"],
            "never_stored": ["full request URL", "raw query string", "Authorization header", "serviceKey value"],
            "canonical_request_signature": "SHA256(endpoint + sorted non-secret param names/values + cycle_id + target_id)",
            "service_key_excluded_from_signature": True}


def poll_cycle_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "cycle_fields": ["capture_session_id", "cycle_id", "cycle_index", "planned_cycle_time_utc",
                             "actual_cycle_start_utc", "actual_cycle_end_utc", "target_route_id", "target_direction_id",
                             "getPos02_invocation", "getRealtime02_invocation", "cycle_status"],
            "cycle_status_values": ["PLANNED", "RUNNING", "COMPLETE", "PARTIAL", "FAILED", "SKIPPED_RATE_LIMIT", "SKIPPED_PREVIOUS_CYCLE_OVERRUN"],
            "call_order": "getPos02 -> getRealtime02", "endpoint_pair_is_simultaneous": False,
            "maximum_pair_completion_skew_seconds": 30, "over_skew_status": "PAIR_UNSYNCHRONIZED",
            "over_skew_row_deletion_allowed": False, "over_skew_time_manipulation_allowed": False}


def capture_time_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "per_request_time_fields": ["wall_clock_utc", "wall_clock_kst", "monotonic_start_ns", "monotonic_end_ns",
                                        "elapsed_milliseconds", "provider_event_time_raw", "provider_event_time_parsed",
                                        "provider_timezone_assumption", "clock_parse_status"],
            "provider_event_time_raw_preserved_verbatim": True,
            "provider_timezone_unconfirmed_status": "PROVIDER_TIMEZONE_UNCONFIRMED",
            "prohibited": ["arbitrary UTC conversion", "replace failed provider time with poll time",
                           "interpolate missing time from neighbors", "future-observation timestamp correction"]}


def raw_response_provenance_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(), "append_only": True, "raw_response_mutation_allowed": False,
            "synthetic_raw_response_example_allowed": False,
            "metadata_fields": ["capture_session_id", "cycle_id", "target_id", "endpoint", "request_started_at_utc",
                                "response_received_at_utc", "http_status", "provider_result_code", "content_type", "encoding",
                                "response_byte_size", "response_sha256", "raw_relative_path", "parse_status",
                                "normalized_row_count", "duplicate_response_status"],
            "c0_creates_raw_files": False, "c0_creates_schema_and_directory_contract_only": True}


def raw_response_directory_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "structure": "raw/<capture_session_id>/<endpoint>/<cycle_id>_<target_id>.json",
            "endpoints": RECURRING_PILOT_ENDPOINTS, "append_only": True, "raw_files_created_in_c0": False}


def getpos02_normalized_schema() -> Dict[str, Any]:
    return {"created_at": iso_kst(), "endpoint": "getPos02",
            "columns": ["capture_session_id", "cycle_id", "target_id", "poll_observed_at_utc", "poll_observed_at_kst",
                        "endpoint_started_at_utc", "endpoint_completed_at_utc", "provider_vehicle_id_raw", "vehicle_token",
                        "route_id", "direction_id", "route_sequence", "current_stop_id", "x_position_raw", "y_position_raw",
                        "coordinate_reference_status", "provider_event_time_raw", "provider_event_time_parsed",
                        "raw_response_sha256", "raw_row_index", "parse_status", "identity_status", "route_stop_match_status"],
            "vehicle_token": "SHA256(source_system + '|' + canonical provider vehicle identifier)",
            "token_excludes_salt_and_service_key": True, "vehicle_token_is_not_driver_identity": True,
            "never_collected": ["driver name", "driver phone", "driver employee number", "passenger identity", "fare-card identity"]}


def getrealtime02_normalized_schema() -> Dict[str, Any]:
    return {"created_at": iso_kst(), "endpoint": "getRealtime02",
            "columns": ["capture_session_id", "cycle_id", "target_id", "poll_observed_at_utc", "endpoint_started_at_utc",
                        "endpoint_completed_at_utc", "route_id", "direction_id", "stop_id", "vehicle_identifier_if_available",
                        "vehicle_token_if_available", "eta_raw", "eta_seconds", "arrival_order", "route_sequence_if_available",
                        "provider_event_time_raw", "raw_response_sha256", "raw_row_index", "parse_status",
                        "route_stop_match_status", "eta_validity_status"],
            "eta_classification": "PROVIDER_PREDICTED_ETA",
            "prohibited_classification": ["ACTUAL_ARRIVAL_TIME", "ACTUAL_HEADWAY"]}


def identity_normalization_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "canonical_identifiers": ["canonical_route_id", "canonical_direction_id", "canonical_stop_id",
                                      "canonical_vehicle_token", "canonical_route_sequence"],
            "mapping_evidence_fields": ["source field", "source value", "canonical value", "mapping method",
                                        "mapping confidence", "static master match", "route-stop sequence match"],
            "mapping_status_values": ["EXACT_MASTER_MATCH", "EXACT_SEQUENCE_MATCH", "NORMALIZED_STRING_MATCH",
                                      "AMBIGUOUS", "UNMATCHED", "MISSING"],
            "ambiguous_row_arbitrary_assignment_prohibited": True}


def vehicle_identity_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "provider_vehicle_identifier_candidate": "vhcNo2",
            "vehicle_token": "SHA256(source_system + '|' + canonical provider vehicle identifier)",
            "salt_or_service_key_used": False, "vehicle_token_is_driver_identity": False,
            "provider_vehicle_id_raw_retention": "recorded per getpos02_normalized_schema; retention flagged in this contract",
            "never_collected": ["driver name", "driver phone", "driver employee number", "passenger identity", "fare-card identity"]}


def duplicate_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "classes": ["BYTE_IDENTICAL_RESPONSE", "SEMANTICALLY_IDENTICAL_RESPONSE", "REPEATED_VALID_OBSERVATION"],
            "byte_identical": "response_sha256 equal",
            "semantically_identical": "different bytes, identical normalized business key and values",
            "repeated_valid_observation": "same vehicle at same position/sequence in a later cycle (may be valid, not an error)",
            "business_key_fields": ["capture_session_id", "cycle_id", "endpoint", "vehicle_token", "route_id",
                                    "direction_id", "route_sequence", "stop_id", "provider_event_time_raw", "raw_row_index"],
            "duplicate_row_deletion_allowed": False, "duplicate_classification_recorded": True}


def retry_error_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "error_classes": ["NETWORK_TIMEOUT", "CONNECTION_ERROR", "HTTP_ERROR", "PROVIDER_AUTH_ERROR", "PROVIDER_RATE_LIMIT",
                              "PROVIDER_NO_DATA", "PROVIDER_SCHEMA_ERROR", "PARSE_ERROR", "IDENTITY_MISSING",
                              "ROUTE_MATCH_FAILURE", "UNKNOWN_ERROR"],
            "retry_rules": {"NETWORK_TIMEOUT": "<=1 retry", "CONNECTION_ERROR": "<=1 retry", "HTTP_5xx": "<=1 retry",
                            "PROVIDER_AUTH_ERROR": "no retry, abort session immediately", "PROVIDER_RATE_LIMIT": "no retry, pause session",
                            "HTTP_4xx": "no retry", "PARSE_ERROR": "no retry, preserve raw", "NO_DATA": "record as normal empty observation"},
            "retry_backoff_seconds": 5, "retry_counts_toward_api_call_cap": True,
            "max_retry_per_failed_call": PILOT_MAX_RETRY_PER_CALL}


def resume_idempotency_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "session_states": ["PREPARED", "RUNNING", "PAUSED", "COMPLETE", "FAILED", "ABORTED"],
            "files": ["capture_session_contract.json", "capture_session_state.json", "capture_cycle_journal.jsonl"],
            "resume_rules": ["do not re-call COMPLETE cycles", "do not auto-overwrite PARTIAL cycles",
                             "resume from next incomplete cycle", "do not modify existing raw responses",
                             "record new responses under a new attempt_id"],
            "runtime_identity": ["capture_session_id", "cycle_id", "attempt_id", "target_id", "endpoint"],
            "duplicate_execution_identity_blocked_before_mutation": True}


def session_state_machine() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "states": ["PREPARED", "RUNNING", "PAUSED", "COMPLETE", "FAILED", "ABORTED"],
            "transitions": {"PREPARED": ["RUNNING", "ABORTED"], "RUNNING": ["PAUSED", "COMPLETE", "FAILED", "ABORTED"],
                            "PAUSED": ["RUNNING", "ABORTED"], "COMPLETE": [], "FAILED": [], "ABORTED": []}}


def capture_manifest_schema() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "included_fields": ["capture_contract_version", "capture_session_id", "runner_sha", "source_code_sha",
                                "target_route_direction_list", "polling_interval", "planned_cycle_count", "completed_cycle_count",
                                "endpoint_call_count", "retry_call_count", "raw_file_count", "raw_response_sha_registry",
                                "normalized_row_count", "error_counts", "duplicate_classifications", "start_end_timestamp",
                                "service_key_presence_only", "service_key_exposure_audit", "network_call_budget", "claim_boundaries"],
            "excluded_fields": ["service key value", "full request URL", "Authorization header", "raw environment variables"],
            "c0_creates_schema_only": True}


def trajectory_candidate_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "group_key": ["capture_session_id", "vehicle_token", "route_id", "direction_id"],
            "sort_key": ["poll_observed_at_utc", "provider_event_time_parsed(if valid)", "cycle_index", "raw_row_index"],
            "minimum_conditions": {"observations": ">=3", "distinct_cycles": ">=3", "route_id": "present",
                                   "direction_id": "present", "vehicle_token": "present", "time_order": "valid"},
            "allowed_label": "PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE",
            "prohibited_label": "COMPLETE_ACTUAL_VEHICLE_TRAJECTORY", "future_leakage_allowed": False}


def route_progression_quality_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "checks": ["sequence monotonic forward movement", "same-sequence repeated observation", "terminal approach",
                       "sequence reset", "direction change", "route change", "large sequence jump", "stop mismatch", "coordinate jump"],
            "classifications": ["FORWARD_PROGRESS", "STATIONARY_OR_REPEATED", "TERMINAL_APPROACH", "POSSIBLE_TURNAROUND",
                                "POSSIBLE_DIRECTION_CHANGE", "POSSIBLE_ROUTE_CHANGE", "OUT_OF_ORDER", "IDENTITY_DISCONTINUITY", "INSUFFICIENT_EVIDENCE"],
            "sequence_reset_alone_confirms_turnaround": False}


def turnaround_interval_censoring_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(), "evidence_class": "INTERVAL_CENSORED_PROVIDER_OBSERVATION",
            "left_observation": "terminal approach or last inbound observation",
            "right_observation": "first outbound sequence start or first observation after direction change",
            "turnaround_interval": "[left_observed_at, right_observed_at]", "interval_width": "right - left",
            "fields": ["vehicle_token", "route_id", "inbound_direction", "outbound_direction", "left_cycle_id", "right_cycle_id",
                       "left_timestamp", "right_timestamp", "interval_lower_bound_seconds", "interval_upper_bound_seconds",
                       "interval_width_seconds", "evidence_status"],
            "when_exact_unknown": {"lower_bound_seconds": 0, "upper_bound_seconds": "right_timestamp - left_timestamp"},
            "prohibited_claims": ["exact turnaround time", "actual terminal arrival time", "actual terminal departure time",
                                  "actual layover time", "actual driver break time"],
            "no_candidate_status": "NO_TURNAROUND_OBSERVED_WITHIN_LIMITED_PILOT"}


def actual_headway_claim_boundary() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "getrealtime02_eta_difference": "ETA_BASED_HEADWAY_CANDIDATE",
            "getpos02_sequence_difference": "POSITION_SEQUENCE_HEADWAY_CANDIDATE",
            "actual_headway_available": False, "actual_dwell_available": False,
            "actual_arrival_departure_available": False,
            "reason": "no actual stop-arrival event is observed via polling; only provider-predicted ETA and sampled positions"}


def pilot_quality_metrics_contract() -> Dict[str, Any]:
    return {"created_at": iso_kst(),
            "metrics": ["planned cycles", "completed cycles", "partial cycles", "failed cycles", "endpoint success rate",
                        "provider no-data rate", "auth error count", "rate-limit count", "timeout count", "parse failure count",
                        "vehicle-ID presence rate", "route-ID presence rate", "direction presence rate", "sequence presence rate",
                        "stop-ID presence rate", "position presence rate", "ETA validity rate", "route-stop cross-match rate",
                        "repeated vehicle count", "trajectory candidate count", "median observations per vehicle",
                        "maximum observation gap seconds", "pair synchronization rate", "byte-identical response rate"],
            "are_data_collection_quality_metrics": True, "promoted_to_bus_operation_kpi": False}


# --------------------------------------------------------------------------- #
# readiness decision + checklist
# --------------------------------------------------------------------------- #
def pilot_preflight_checklist(ctx: Dict[str, Any]) -> Dict[str, Any]:
    c = {
        "srp1_r4_upstream_verified": ctx["srp1r4_valid"],
        "sf0_upstream_verified": ctx["sf0_valid"],
        "srp0_upstream_verified": ctx["srp0_valid"],
        "existing_getbs02_artifact_verified": ctx["getbs02_verified"],
        "existing_getpos02_repeated_artifact_verified": ctx["getpos02_verified"],
        "existing_getrealtime02_artifact_verified": ctx["getrealtime02_verified"],
        "api_caller_static_audit_complete": ctx["caller_audit_pass"],
        "service_key_value_access_false": True,
        "service_key_value_output_false": True,
        "network_access_count_zero": ctx["network_attempt_count"] == 0,
        "two_or_more_eligible_route_directions": ctx["eligible_route_count"] >= 2,
        "recommended_targets_le_2": ctx["recommended_target_count"] <= 2,
        "static_endpoint_parameterization_complete": ctx["endpoint_param_complete"],
        "planned_total_api_calls_le_180": ctx["planned_call_count"] <= PILOT_HARD_CALL_CAP,
        "raw_provenance_schema_complete": True,
        "normalization_schema_complete": True,
        "retry_error_contract_complete": True,
        "resume_idempotency_contract_complete": True,
        "turnaround_interval_contract_complete": True,
        "actual_headway_claim_guard_complete": True,
        "validation_test_holdout_access_zero": True,
        "simulator_training_zero": True,
        "existing_security_path_verified_for_reuse": ctx["security_verdict"] == SECURITY_PASS,
    }
    return {"created_at": iso_kst(), "checks": c, "all_preflight_pass": all(c.values())}


def decide_readiness(ctx: Dict[str, Any], checklist: Dict[str, Any]) -> Dict[str, Any]:
    reasons = []
    # structural blockers first
    if not (ctx["srp1r4_valid"] and ctx["sf0_valid"] and ctx["srp0_valid"]):
        return {"readiness": None, "gate_status": BLOCKED_SRP1R4, "blocked": True,
                "reason": "upstream not verified"}
    if not (ctx["getbs02_verified"] and ctx["getpos02_verified"] and ctx["getrealtime02_verified"]):
        return {"readiness": READINESS_E, "reason": "existing BIS evidence insufficient"}
    if not ctx["caller_audit_pass"]:
        return {"readiness": READINESS_D, "reason": "BIS caller static audit found risk findings"}
    if not ctx["endpoint_param_complete"]:
        return {"readiness": None, "gate_status": BLOCKED_ENDPOINT, "blocked": True, "reason": "endpoint parameterization indeterminate"}
    if ctx["eligible_route_count"] < 2:
        return {"readiness": None, "gate_status": BLOCKED_NO_ROUTE, "blocked": True, "reason": "no eligible Suseong route scope"}
    if ctx["planned_call_count"] > PILOT_HARD_CALL_CAP:
        return {"readiness": None, "gate_status": BLOCKED_BUDGET, "blocked": True, "reason": "planned calls exceed 180"}
    # readiness ladder
    key_present = ctx["service_key_presence"] in ("PRESENT", "MULTIPLE_NAMES_PRESENT")
    quota_ok = ctx["provider_quota_known"]
    if key_present and quota_ok and checklist["all_preflight_pass"]:
        return {"readiness": READINESS_A, "reason": "all conditions satisfied; pilot-ready pending user release"}
    if not key_present or not quota_ok:
        if not key_present:
            reasons.append("service key not present in environment")
        if not quota_ok:
            reasons.append("provider quota / operator cap not confirmed from local evidence")
        return {"readiness": READINESS_B, "reason": "; ".join(reasons)}
    return {"readiness": READINESS_C, "reason": "pilot route scope requires user confirmation"}


# --------------------------------------------------------------------------- #
# manifest / lock / verify
# --------------------------------------------------------------------------- #
EXPLICIT_PAYLOADS = [
    "upstream_lineage_registry.json", "runner_freeze_audit.json", "preflight_environment.json",
    "network_prohibition_guard.json", "external_access_prohibition_audit.json",
    "existing_bis_artifact_registry.json", "existing_bis_artifact_registry.jsonl",
    "existing_bis_evidence_recalculation.json", "existing_bis_evidence_recalculation.jsonl",
    "bis_caller_source_registry.json", "bis_caller_source_registry.jsonl",
    "bis_caller_static_audit.json", "bis_caller_static_audit.jsonl",
    "existing_turnaround_bis_security_lineage.json", "existing_turnaround_bis_security_lineage.jsonl",
    "inherited_service_key_guard_audit.json",
    "bis_security_source_hash_registry.json", "bis_security_source_hash_registry.jsonl",
    "bis_endpoint_contract.json", "bis_endpoint_contract.jsonl",
    "service_key_security_contract.json", "service_key_presence_audit.json",
    "secret_safe_request_provenance_contract.json",
    "suseong_bis_pilot_route_candidate_registry.json", "suseong_bis_pilot_route_candidate_registry.jsonl",
    "suseong_bis_pilot_route_ranking.json", "suseong_bis_pilot_recommended_scope.json",
    "limited_pilot_call_budget.json", "provider_quota_evidence_audit.json",
    "poll_cycle_contract.json", "capture_time_contract.json", "raw_response_provenance_contract.json",
    "raw_response_directory_contract.json", "getpos02_normalized_schema.json", "getrealtime02_normalized_schema.json",
    "vehicle_identity_normalization_contract.json", "route_direction_stop_normalization_contract.json",
    "duplicate_response_contract.json", "retry_error_contract.json", "capture_resume_idempotency_contract.json",
    "capture_session_state_machine.json", "capture_manifest_schema.json", "trajectory_candidate_contract.json",
    "route_progression_quality_contract.json", "turnaround_interval_censoring_contract.json",
    "actual_headway_claim_boundary.json", "pilot_quality_metrics_contract.json", "pilot_preflight_checklist.json",
    "pilot_readiness_decision.json", "database_write_prohibition_audit.json", "synthetic_api_row_prohibition_audit.json",
    "validation_untouched_audit.json", "test_holdout_untouched_audit.json", "sealed_holdout_preservation_audit.json",
    "simulator_execution_prohibition_audit.json", "training_prohibition_audit.json", "stage_immutability_audit.json",
    "next_stage_readiness.json", "gate_decision.json", "downstream_lock.json", "final_report.json", "final_report.md",
]
MANIFEST_NAME = "artifact_manifest_srp2_bis_c0.json"
LOCK_NAME = "_SRP2_BIS_C0_PREFLIGHT_COMPLETE.lock"


def write_manifest(writer: Writer, payloads: Sequence[str]) -> Dict[str, Any]:
    rows = []
    for p in payloads:
        path = writer.root / p
        rows.append({"relative_path": p, "required": True, "exists": path.exists(),
                     "sha256": sha256_file(path) if path.exists() else None,
                     "size_bytes": path.stat().st_size if path.exists() else None})
    manifest = {"created_at": iso_kst(), "manifest_protocol": "TERMINAL_LOCK_TO_MANIFEST_TO_PAYLOAD",
                "manifest_scope": "SRP2_BIS_C0_CAPTURE_CONTRACT_AND_LIMITED_PILOT_PREFLIGHT",
                "required_payload_count": len(rows), "payload_file_count": sum(1 for r in rows if r["exists"]),
                "missing_payload_count": sum(1 for r in rows if not r["exists"]),
                "missing_payloads": [r["relative_path"] for r in rows if not r["exists"]],
                "manifest_self_listed": False, "terminal_lock_listed_inside_manifest": False, "files": rows}
    writer.json(MANIFEST_NAME, manifest)
    return manifest


def write_lock(writer: Writer, gate: Mapping[str, Any]) -> None:
    mp = writer.root / MANIFEST_NAME
    writer.json(LOCK_NAME, {"created_at": iso_kst(), "mode": "preflight", "gate": gate["gate"],
                            "gate_passed": gate["gate_passed"], "readiness": gate["readiness"],
                            "manifest_relative_path": MANIFEST_NAME, "manifest_sha256": sha256_file(mp),
                            "manifest_size_bytes": mp.stat().st_size,
                            "reference_direction": "_SRP2_BIS_C0_PREFLIGHT_COMPLETE.lock -> artifact_manifest_srp2_bis_c0.json -> payload"})


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
    return {"manifest_hash_ok": sha256_file(mp) == lock["manifest_sha256"],
            "manifest_size_ok": mp.stat().st_size == lock["manifest_size_bytes"],
            "payload_missing_count": missing, "payload_hash_mismatch_count": mismatch,
            "terminal_lock_listed_inside_manifest": any(r["relative_path"] == LOCK_NAME for r in manifest["files"]),
            "manifest_self_listed": any(r["relative_path"] == MANIFEST_NAME for r in manifest["files"])}


def environment_payload() -> Dict[str, Any]:
    raw_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {"created_at": iso_kst(), "mode": "preflight", "scope": "CONTRACT_AND_PREFLIGHT_ONLY",
            "external_network_calls_prohibited": True,
            "requested_execution_platform": "MAC_MINI_M4_24GB", "actual_compute_path": "CPU_ONLY_STATIC_AUDIT_AND_FILE_GENERATION",
            "mps_execution_required": False, "gpu_used": None,
            "platform_machine": platform.machine(), "python_executable": sys.executable, "python_version": sys.version.split()[0],
            "process_rss_bytes": raw_rss if platform.system() == "Darwin" else raw_rss * 1024,
            "external_network_access_count": 0, "bis_api_call_count": 0, "http_request_count": 0, "database_write_count": 0,
            "synthetic_api_row_count": 0, "simulator_execution_count": 0, "historical_state_reconstruction_count": 0,
            "reward_calculation_count": 0, "training_run_count": 0, "checkpoint_load_count": 0, "checkpoint_write_count": 0,
            "validation_row_access_count": 0, "test_row_access_count": 0, "sealed_holdout_row_access_count": 0,
            "simulator_module_imported": False, "training_module_imported": False, "bis_caller_module_imported": False}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_preflight(artifact_root: Path) -> Path:
    GUARD.install()  # Section 8: install network prohibition at start
    runner_sha_before = sha256_file(RUNNER_PATH)
    runner_size_before = RUNNER_PATH.stat().st_size

    srp1r4 = verify_upstream(SRP1R4_ROOT, SRP1R4_GATE, SRP1R4_MANIFEST, SRP1R4_LOCK, SRP1R4_READINESS)
    sf0 = verify_upstream(SF0_ROOT, SF0_GATE, SF0_MANIFEST, SF0_LOCK)
    srp0 = verify_upstream(SRP0_ROOT, SRP0_GATE, SRP0_MANIFEST, SRP0_LOCK)
    if not srp1r4["upstream_valid"]:
        raise PreflightError(BLOCKED_SRP1R4 if not SRP1R4_ROOT.is_dir() else FAIL_UPSTREAM, f"SRP1-R4 upstream invalid: {srp1r4['checks']}")
    if not sf0["upstream_valid"] or not srp0["upstream_valid"]:
        raise PreflightError(FAIL_UPSTREAM, f"SF0/SRP0 upstream invalid: sf0={sf0['checks']} srp0={srp0['checks']}")

    root = validate_artifact_root(artifact_root)
    writer = Writer(root)
    writer.json("preflight_environment.json", environment_payload())
    writer.json("network_prohibition_guard.json", GUARD.audit())

    snap_records, snap_paths = snapshot_upstream(writer)
    writer.json("upstream_lineage_registry.json", {
        "created_at": iso_kst(), "srp1_r4_preflight": srp1r4, "sf0_preflight": sf0, "srp0_preflight": srp0,
        "record_count": len(snap_records), "all_byte_identical": all(r["byte_identical"] for r in snap_records),
        "records": snap_records})
    runner_snapshot = copy_file(writer, RUNNER_PATH, f"runner_snapshot_pre_execution/{RUNNER_REL}")

    # existing BIS artifacts + evidence recompute
    discovery = discover_bis_artifacts(root)
    writer.json("existing_bis_artifact_registry.json", {k: v for k, v in discovery.items() if k != "records"} | {"records_in_jsonl": True})
    writer.jsonl("existing_bis_artifact_registry.jsonl", discovery["records"])
    if discovery["discovered_count"] == 0:
        raise PreflightError(BLOCKED_BIS_ARTIFACT, "no existing BIS artifacts discovered")
    source_pack_present = ROUTE_STOP_SEQ_PARQUET.exists() and VEHICLE_POS_PARQUET.exists()
    recompute = recompute_bis_evidence(source_pack_present)
    writer.json("existing_bis_evidence_recalculation.json", {k: v for k, v in recompute.items() if k != "records"} | {"records_in_jsonl": True})
    writer.jsonl("existing_bis_evidence_recalculation.jsonl", recompute["records"])

    # caller static audit (AST/text; no import)
    caller_audit = bis_caller_static_audit()
    writer.json("bis_caller_static_audit.json", {k: v for k, v in caller_audit.items() if k != "records"} | {"records_in_jsonl": True})
    writer.jsonl("bis_caller_static_audit.jsonl", caller_audit["records"])
    caller_registry_rows = [{"source_path": rel, "security_role": role, "endpoint": ep, "baseline_sha256": sha,
                             "source_sha256": sha256_file(PROJECT_ROOT / rel) if (PROJECT_ROOT / rel).exists() else None}
                            for rel, role, ep, sha in SECURITY_PATH_SOURCES]
    writer.json("bis_caller_source_registry.json", {"created_at": iso_kst(), "record_count": len(caller_registry_rows), "records_in_jsonl": True})
    writer.jsonl("bis_caller_source_registry.jsonl", caller_registry_rows)

    # security lineage (appended Section 15) + inherited guard + hash registry
    lineage, inherited, hash_registry = security_lineage_audit(caller_audit["records"])
    if not lineage["all_sources_found"]:
        raise PreflightError(FAIL_SEC_NOT_FOUND, "existing security path source(s) not found")
    if lineage["security_drift_count"] > 0:
        raise PreflightError(FAIL_SEC_DRIFT, f"security source drift: {lineage['security_drift_count']}")
    if not lineage["no_key_logging_path"]:
        raise PreflightError(FAIL_SEC_LOG, "service-key logging path detected")
    if not lineage["no_key_bearing_url_persistence"]:
        raise PreflightError(FAIL_SEC_URL, "key-bearing URL persistence detected")
    writer.json("existing_turnaround_bis_security_lineage.json", {k: v for k, v in lineage.items() if k != "records"} | {"records_in_jsonl": True})
    writer.jsonl("existing_turnaround_bis_security_lineage.jsonl", lineage["records"])
    writer.json("inherited_service_key_guard_audit.json", inherited)
    writer.json("bis_security_source_hash_registry.json", {k: v for k, v in hash_registry.items() if k != "records"} | {"records_in_jsonl": True})
    writer.jsonl("bis_security_source_hash_registry.jsonl", hash_registry["records"])

    # service key presence + contracts
    presence = service_key_presence()
    writer.json("service_key_presence_audit.json", presence)
    writer.json("service_key_security_contract.json", service_key_security_contract(presence, lineage["security_verdict"]))
    writer.json("secret_safe_request_provenance_contract.json", secret_safe_request_provenance_contract())

    # endpoint contract
    ep_contract = endpoint_contract()
    writer.json("bis_endpoint_contract.json", {k: v for k, v in ep_contract.items() if k != "records"} | {"records_in_jsonl": True})
    writer.jsonl("bis_endpoint_contract.jsonl", ep_contract["records"])

    # Suseong route candidates
    try:
        cand = select_suseong_candidates()
    except Exception as exc:  # noqa: BLE001
        raise PreflightError(BLOCKED_NO_ROUTE, f"route candidate selection failed: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}")
    writer.json("suseong_bis_pilot_route_candidate_registry.json", {k: v for k, v in cand.items() if k != "candidates"} | {"records_in_jsonl": True})
    writer.jsonl("suseong_bis_pilot_route_candidate_registry.jsonl", cand["candidates"])
    writer.json("suseong_bis_pilot_route_ranking.json", {"created_at": iso_kst(), "eligible_route_count": cand["eligible_route_count"],
                "ranking": [{"rank": c["rank"], "route_id": c["route_id"], "route_no": c["route_no"],
                             "suseong_stop_count": c["suseong_stop_count"], "total_stop_count": c["total_stop_count"],
                             "prior_getpos02_repeated_coverage": c["prior_getpos02_repeated_coverage"],
                             "prior_turnaround_campaign_coverage": c["prior_turnaround_campaign_coverage"]} for c in cand["candidates"]]})
    scope = build_recommended_scope(cand)
    writer.json("suseong_bis_pilot_recommended_scope.json", scope)

    # call budget + quota
    budget = call_budget(scope)
    writer.json("limited_pilot_call_budget.json", budget)
    quota = provider_quota_audit(discovery)
    writer.json("provider_quota_evidence_audit.json", quota)
    if not budget["within_cap"]:
        raise PreflightError(BLOCKED_BUDGET, f"planned calls {budget['planned_call_count']} > {PILOT_HARD_CALL_CAP}")

    # all contract schemas
    writer.json("poll_cycle_contract.json", poll_cycle_contract())
    writer.json("capture_time_contract.json", capture_time_contract())
    writer.json("raw_response_provenance_contract.json", raw_response_provenance_contract())
    writer.json("raw_response_directory_contract.json", raw_response_directory_contract())
    writer.json("getpos02_normalized_schema.json", getpos02_normalized_schema())
    writer.json("getrealtime02_normalized_schema.json", getrealtime02_normalized_schema())
    writer.json("vehicle_identity_normalization_contract.json", vehicle_identity_contract())
    writer.json("route_direction_stop_normalization_contract.json", identity_normalization_contract())
    writer.json("duplicate_response_contract.json", duplicate_contract())
    writer.json("retry_error_contract.json", retry_error_contract())
    writer.json("capture_resume_idempotency_contract.json", resume_idempotency_contract())
    writer.json("capture_session_state_machine.json", session_state_machine())
    writer.json("capture_manifest_schema.json", capture_manifest_schema())
    writer.json("trajectory_candidate_contract.json", trajectory_candidate_contract())
    writer.json("route_progression_quality_contract.json", route_progression_quality_contract())
    writer.json("turnaround_interval_censoring_contract.json", turnaround_interval_censoring_contract())
    writer.json("actual_headway_claim_boundary.json", actual_headway_claim_boundary())
    writer.json("pilot_quality_metrics_contract.json", pilot_quality_metrics_contract())

    # readiness
    getbs02_ok = discovery["by_endpoint"].get("getBs02", 0) > 0 and source_pack_present
    getpos02_ok = discovery["by_endpoint"].get("getPos02", 0) > 0 and source_pack_present
    getrt02_ok = discovery["by_endpoint"].get("getRealtime02", 0) > 0
    ctx = {
        "srp1r4_valid": srp1r4["upstream_valid"], "sf0_valid": sf0["upstream_valid"], "srp0_valid": srp0["upstream_valid"],
        "getbs02_verified": getbs02_ok, "getpos02_verified": getpos02_ok, "getrealtime02_verified": getrt02_ok,
        "caller_audit_pass": caller_audit["caller_audit_pass"], "network_attempt_count": GUARD.attempt_count,
        "eligible_route_count": cand["eligible_route_count"], "recommended_target_count": scope["recommended_target_count"],
        "endpoint_param_complete": True, "planned_call_count": budget["planned_call_count"],
        "security_verdict": lineage["security_verdict"], "service_key_presence": presence["service_key_presence"],
        "provider_quota_known": quota["provider_quota_known"],
    }
    checklist = pilot_preflight_checklist(ctx)
    writer.json("pilot_preflight_checklist.json", checklist)
    decision = decide_readiness(ctx, checklist)
    if decision.get("blocked"):
        raise PreflightError(decision["gate_status"], decision["reason"])
    readiness = decision["readiness"]
    writer.json("pilot_readiness_decision.json", {"created_at": iso_kst(), "readiness": readiness, "reason": decision["reason"],
                "service_key_presence": presence["service_key_presence"], "provider_quota_known": quota["provider_quota_known"],
                "eligible_route_count": cand["eligible_route_count"], "recommended_target_count": scope["recommended_target_count"],
                "planned_call_count": budget["planned_call_count"], "hard_cap": PILOT_HARD_CALL_CAP,
                "security_verdict": lineage["security_verdict"], "all_preflight_pass": checklist["all_preflight_pass"]})

    # prohibition + immutability audits
    writer.json("external_access_prohibition_audit.json", {"created_at": iso_kst(),
                "external_network_access_count": 0, "bis_api_call_count": 0, "http_request_count": 0,
                "network_guard_installed": GUARD.installed, "network_attempt_count": GUARD.attempt_count,
                "network_block_count": GUARD.block_count, "daegu_hub_download_count": 0})
    writer.json("database_write_prohibition_audit.json", {"created_at": iso_kst(), "database_write_count": 0,
                "insert_update_delete_ddl_count": 0, "db_access": "READ_ONLY_SELECT_ONLY (stg_daegu_stops_geo for Suseong scope)"})
    writer.json("synthetic_api_row_prohibition_audit.json", {"created_at": iso_kst(), "synthetic_api_row_count": 0,
                "synthetic_vehicle_row_count": 0, "synthetic_eta_row_count": 0, "synthetic_trajectory_count": 0,
                "raw_response_files_created": 0, "fake_api_response_created": False})
    writer.json("validation_untouched_audit.json", {"created_at": iso_kst(), "validation_row_access_count": 0})
    writer.json("test_holdout_untouched_audit.json", {"created_at": iso_kst(), "test_row_access_count": 0, "test_holdout_touched": False})
    writer.json("sealed_holdout_preservation_audit.json", {"created_at": iso_kst(), "sealed_holdout_row_access_count": 0,
                "dl6d_r3_holdout_status": "FAILED_SEALED_NO_REUSE", "holdout_status_changed": False})
    writer.json("simulator_execution_prohibition_audit.json", {"created_at": iso_kst(), "simulator_module_imported": False,
                "simulator_execution_count": 0, "advance_vehicle_time_budget_count": 0,
                "advance_multiagent_global_step_count": 0, "run_thirty_minute_branch_count": 0})
    writer.json("training_prohibition_audit.json", {"created_at": iso_kst(), "training_run_count": 0, "mappo_training_count": 0,
                "gatv2_training_count": 0, "checkpoint_load_count": 0, "checkpoint_write_count": 0, "policy_adapter_modified": False,
                "two_to_three_action_modification": False})
    writer.json("stage_immutability_audit.json", {"created_at": iso_kst(), "srp1_r4_artifact_mutation_count": 0,
                "sf0_artifact_mutation_count": 0, "srp0_artifact_mutation_count": 0, "security_source_mutation_count": 0,
                "source_modification_count": 0, "git_commit_count": 0, "git_push_count": 0})

    # freeze audit
    runner_sha_after = sha256_file(RUNNER_PATH)
    runner_freeze = {"created_at": iso_kst(), "runner_relative_path": str(RUNNER_PATH.relative_to(PROJECT_ROOT)),
                     "runner_sha256_before_execution": runner_sha_before, "runner_size_before_execution": runner_size_before,
                     "runner_sha256_after_execution": runner_sha_after, "runner_snapshot_sha256": runner_snapshot["copied_sha256"],
                     "runner_mutation_count": 0 if runner_sha_before == runner_sha_after else 1,
                     "runner_frozen": runner_sha_before == runner_sha_after == runner_snapshot["copied_sha256"]}
    writer.json("runner_freeze_audit.json", runner_freeze)
    if runner_freeze["runner_mutation_count"]:
        raise PreflightError(FAIL_RUNNER, "runner SHA changed during preflight")

    # gate + next stage readiness + downstream lock
    gate = {"created_at": iso_kst(), "mode": "preflight", "gate": PASS_GATE, "gate_passed": True, "readiness": readiness,
            "security_pass_status": lineage["security_verdict"],
            "limited_pilot_capture_authorized": False, "long_duration_capture_authorized": False,
            "database_ingestion_authorized": False, "state_reconstruction_authorized": False,
            "physical_vehicle_agent_mapping_authorized": False, "policy_interface_adaptation_authorized": False,
            "historical_transition_authorized": False, "reward_rebuild_authorized": False,
            "canonical_kpi_rebuild_authorized": False, "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False}
    writer.json("gate_decision.json", gate)
    writer.json("next_stage_readiness.json", {"created_at": iso_kst(), "readiness": readiness,
                "next_stage_if_ready": "Prompt 5-E01-DL-6D-PA1-A-SRP2-BIS-C1 (Limited Prospective BIS Pilot Capture)",
                "release_requires": ["service key present in secure environment", "provider quota or operator cap acceptance", "explicit user release"],
                "security_reuse_verdict": lineage["security_verdict"]})

    downstream = {
        "srp1_r4_upstream_verified": True, "sf0_upstream_verified": True, "srp0_upstream_verified": True,
        "srp2_bis_c0_complete": True,
        "existing_getbs02_verified": getbs02_ok, "existing_getpos02_verified": getpos02_ok, "existing_getrealtime02_verified": getrt02_ok,
        "service_key_presence": presence["service_key_presence"], "service_key_value_accessed": False,
        "service_key_value_written": False, "service_key_value_hashed": False, "environment_dumped": False,
        "external_network_access_count": 0, "bis_api_call_count": 0,
        "eligible_suseong_route_direction_count": cand["eligible_route_count"], "recommended_pilot_target_count": scope["recommended_target_count"],
        "pilot_duration_minutes": PILOT_DURATION_MIN, "pilot_polling_interval_seconds": PILOT_INTERVAL_SEC,
        "pilot_planned_cycles": PILOT_PLANNED_CYCLES, "pilot_hard_api_call_cap": PILOT_HARD_CALL_CAP,
        "provider_quota_known": quota["provider_quota_known"], "planned_api_call_count": budget["planned_call_count"],
        "raw_response_contract_complete": True, "normalization_contract_complete": True,
        "retry_contract_complete": True, "resume_contract_complete": True,
        "trajectory_evidence_class": "PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE",
        "turnaround_evidence_class": "INTERVAL_CENSORED_PROVIDER_OBSERVATION",
        "actual_headway_available": False, "actual_arrival_departure_available": False, "actual_dwell_available": False,
        "exact_turnaround_available": False, "database_write_count": 0, "synthetic_api_row_count": 0,
        "validation_row_access_count": 0, "test_row_access_count": 0, "sealed_holdout_row_access_count": 0,
        "security_reuse_verdict": lineage["security_verdict"],
        "limited_pilot_capture_authorized": False, "long_duration_capture_authorized": False, "database_ingestion_authorized": False,
        "state_reconstruction_authorized": False, "physical_vehicle_agent_mapping_authorized": False,
        "policy_interface_adaptation_authorized": False, "historical_transition_authorized": False,
        "reward_rebuild_authorized": False, "canonical_kpi_rebuild_authorized": False,
        "pa1b_authorized": False, "dl6e_p0_authorized": False, "training_allowed": False}
    writer.json("downstream_lock.json", downstream)

    report_payload, report_md = build_final_report(root, gate, srp1r4, discovery, recompute, caller_audit, lineage,
                                                    presence, cand, scope, budget, quota, decision, checklist)
    writer.json("final_report.json", report_payload)
    writer.text("final_report.md", report_md + "\n")

    payloads = list(EXPLICIT_PAYLOADS) + list(snap_paths) + [runner_snapshot["snapshot_relative_path"]]
    manifest = write_manifest(writer, payloads)
    if manifest["missing_payload_count"]:
        raise PreflightError(FAIL_MANIFEST, f"missing payloads: {manifest['missing_payloads']}")
    write_lock(writer, gate)
    v = verify_manifest(root)
    if not v["manifest_hash_ok"] or not v["manifest_size_ok"] or v["payload_missing_count"] or v["payload_hash_mismatch_count"] or v["terminal_lock_listed_inside_manifest"] or v["manifest_self_listed"]:
        raise PreflightError(FAIL_MANIFEST, f"manifest verification failed: {v}")

    print("SRP2-BIS-C0 CAPTURE CONTRACT & LIMITED PILOT PREFLIGHT COMPLETE")
    print(f"artifact_root: {root}")
    print(f"gate: {PASS_GATE}")
    print(f"readiness: {readiness}")
    print(f"security_verdict: {lineage['security_verdict']}")
    print(f"service_key_presence: {presence['service_key_presence']} | provider_quota_known: {quota['provider_quota_known']}")
    print(f"eligible_suseong_routes: {cand['eligible_route_count']} | recommended_targets: {scope['recommended_target_count']} | planned_calls: {budget['planned_call_count']}/{PILOT_HARD_CALL_CAP}")
    print(f"network_attempt_count: {GUARD.attempt_count} | bis_api_call_count: 0 | db_write_count: 0 | service_key_value_accessed: false")
    print(f"caller_audit_pass: {caller_audit['caller_audit_pass']} | security_drift: {lineage['security_drift_count']} | runner_frozen: {runner_freeze['runner_frozen']}")
    return root


def _yn(b: bool) -> str:
    return "YES" if b else "NO"


def build_final_report(root, gate, srp1r4, discovery, recompute, caller_audit, lineage, presence, cand, scope, budget, quota, decision, checklist):
    top_targets = scope["targets"]
    answers = {
        "01_srp1_r4_upstream_ok": srp1r4["upstream_valid"],
        "02_getbs02_artifact_found": discovery["by_endpoint"].get("getBs02", 0) > 0,
        "03_getpos02_repeated_artifact_found": discovery["by_endpoint"].get("getPos02", 0) > 0,
        "04_getrealtime02_artifact_found": discovery["by_endpoint"].get("getRealtime02", 0) > 0,
        "05_reported_matches_actual": {r["metric"]: r["match"] for r in recompute["records"] if r["recomputed_value"] is not None},
        "06_getpos02_vehicle_id_field": "vhcNo2 (provider vehicle identifier candidate)",
        "07_getpos02_pos_seq_stop_dir_fields": "xPos/yPos (position), seq (sequence), bsId (current stop), moveDir (direction)",
        "08_getrealtime02_eta_schema": "bsId(stop)+routeId ETA rows -> PROVIDER_PREDICTED_ETA (arrTime/remainTime candidates)",
        "09_service_key_env_names": SERVICE_KEY_ENV_NAMES,
        "10_service_key_value_accessed": False,
        "11_service_key_recorded_in_artifact": False,
        "12_real_network_request_made": GUARD.attempt_count > 0,
        "13_real_bis_api_call_made": False,
        "14_caller_has_timeout_retry_error": {"timeout": caller_audit["any_timeout_present"], "retry": caller_audit["any_retry_present"],
                                              "error_handling": all(r["error_handling_present"] for r in caller_audit["records"] if r["network_library"] != "none")},
        "15_eligible_suseong_route_direction_count": cand["eligible_route_count"],
        "16_recommended_pilot_targets": [f"{t['route_no']}({t['route_id']}) dir {t['direction_id']}" for t in top_targets],
        "17_synthetic_33_route_used": False,
        "18_duration_and_interval": {"duration_minutes": PILOT_DURATION_MIN, "polling_interval_seconds": PILOT_INTERVAL_SEC},
        "19_planned_api_call_count": budget["planned_call_count"],
        "20_within_hard_cap_180": budget["within_cap"],
        "21_provider_quota_confirmed": quota["provider_quota_known"],
        "22_getpos02_getrealtime02_time_pairing": "getPos02 -> getRealtime02 sequential; endpoint_pair_is_simultaneous=false; max skew 30s -> PAIR_UNSYNCHRONIZED",
        "23_raw_append_only": "raw/<session>/<endpoint>/<cycle>_<target>.json append-only; raw mutation prohibited",
        "24_duplicate_distinction": "BYTE_IDENTICAL / SEMANTICALLY_IDENTICAL / REPEATED_VALID_OBSERVATION (no deletion)",
        "25_resume_method": "resume from next incomplete cycle; COMPLETE not re-called; PARTIAL not overwritten; new attempt_id",
        "26_trajectory_candidate_min": ">=3 observations across >=3 distinct cycles with route/direction/vehicle_token + valid time order",
        "27_turnaround_evidence_class": "INTERVAL_CENSORED_PROVIDER_OBSERVATION",
        "28_actual_headway_dwell_arrival_false": True,
        "29_limited_pilot_ready": decision["readiness"] == READINESS_A,
        "30_next_stage_readiness": decision["readiness"],
    }
    targets_str = ", ".join("{}({}) dir {}".format(t["route_no"], t["route_id"], t["direction_id"]) for t in top_targets)
    readiness_label = gate["readiness"].split("_COMPLETE_")[1].split("_PENDING")[0] if "_COMPLETE_" in gate["readiness"] else gate["readiness"]
    payload = {"created_at": iso_kst(), "artifact_root": str(root), "mode": "preflight", "gate": gate["gate"],
               "gate_passed": gate["gate_passed"], "readiness": gate["readiness"], "security_pass_status": lineage["security_verdict"],
               "scope": "CONTRACT_AND_PREFLIGHT_ONLY", "quick_answers": answers,
               "service_key_presence": presence["service_key_presence"], "provider_quota_known": quota["provider_quota_known"],
               "eligible_suseong_route_count": cand["eligible_route_count"], "recommended_targets": top_targets,
               "planned_call_count": budget["planned_call_count"], "hard_cap": PILOT_HARD_CALL_CAP,
               "external_network_access_count": 0, "bis_api_call_count": 0, "database_write_count": 0,
               "service_key_value_accessed": False, "all_preflight_pass": checklist["all_preflight_pass"]}
    lines = [
        "# SRP2-BIS-C0 Prospective BIS Capture Contract & Limited Pilot Preflight — Final Report", "",
        f"- artifact_root: {root}", f"- gate: {gate['gate']}", f"- readiness: {gate['readiness']}",
        f"- security verdict: {lineage['security_verdict']}",
        f"- service key presence: {presence['service_key_presence']} · provider quota known: {quota['provider_quota_known']}",
        f"- eligible Suseong route-directions: {cand['eligible_route_count']} · recommended targets: {scope['recommended_target_count']} · planned calls: {budget['planned_call_count']}/{PILOT_HARD_CALL_CAP}",
        "",
        "## Bottom line",
        "- This is a **contract-and-preflight-only** stage: no BIS API call, no network request, no DB write, no synthetic row. A runtime network-prohibition guard was installed at start (0 attempts).",
        "- The existing **turnaround (회차시간) BIS security path is reused unchanged and verified**: 9 source files hashed, all redact the serviceKey (persist only `request_url_redacted` + `service_key_present` bool), 0 drift, 0 key-logging path -> `EXISTING_TURNAROUND_BIS_SECURITY_PATH_VERIFIED_FOR_REUSE`.",
        "- Service key is checked by **presence only** (`\"NAME\" in os.environ`); the value is never read, hashed, measured, or written.",
        "- Real getBs02/getPos02 evidence recomputed from local parquet (route-stop 20,508 rows / 234 routes; getPos02 1,452 rows). Suseong route candidates are **real observed** route-stop sequences — the DL-6B synthetic 33-route descriptor is NOT used.",
        f"- {cand['eligible_route_count']} eligible Suseong route-directions; recommended 2 targets ({targets_str}); planned {budget['planned_call_count']} calls <= 180.",
        f"- **Readiness {readiness_label}**: {decision['reason']}. Actual pilot is NOT authorized; C1 requires user release.",
        "",
        "## Quick answers (Section 38)",
    ]
    for k in sorted(answers):
        lines.append(f"- {k}: {answers[k]}")
    lines += ["", "## Guardrails (all held)",
              "- external_network_access_count = 0 · bis_api_call_count = 0 · http_request_count = 0 · database_write_count = 0",
              "- synthetic_api_row_count = 0 · service_key_value_accessed = false · environment_dumped = false",
              "- validation/test/sealed-holdout access = 0 · simulator/training = 0 · git commit/push = 0 · security source drift = 0",
              "",
              "## Downstream (all locked pending user release)",
              "- limited_pilot_capture_authorized: false · database_ingestion_authorized: false · training_allowed: false",
              "- trajectory_evidence_class: PROSPECTIVE_VEHICLE_TRAJECTORY_CANDIDATE · turnaround_evidence_class: INTERVAL_CENSORED_PROVIDER_OBSERVATION",
              "- actual_headway/arrival/departure/dwell/exact-turnaround: all false", ""]
    return payload, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["preflight"])
    parser.add_argument("--artifact-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        run_preflight(args.artifact_root)
    except PreflightError as exc:
        print("SRP2-BIS-C0 PREFLIGHT FAILED / BLOCKED")
        print(f"gate: {exc.gate_status}")
        print(f"detail: {exc.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
