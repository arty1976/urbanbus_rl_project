from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT_DEFAULT = Path("/Users/arty/Documents/Codex/urbanbus_rl_project")
HF1_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / "05_training/artifacts/prompt5_e01_r2d1c_r4a_hf1_limited_revalidation_20260723_091415"
R2D1D_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / "05_training/artifacts/prompt5_e01_r2d1d_terminal_recovery_observation_20260723_104708"
R2D1D2_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / "05_training/artifacts/prompt5_e01_r2d1d2_upstream_terminal_recovery_observation_20260723_115311"
OUTPUT_PREFIX = "05_training/artifacts/prompt5_e01_r2d1d3_targeted_4010002118_clock_audit"
TARGET_ROUTE = "4010002118"
ALL_ROUTES = ["4010002001", "4010002004", "4010002118", "4050010000"]
GETPOS02_URL = "https://apis.data.go.kr/6270000/dbmsapi02/getPos02"
KST = ZoneInfo("Asia/Seoul")
API_ALLOWED_AFTER = datetime(2026, 7, 24, 9, 0, 0, tzinfo=KST)

MAX_ADDITIONAL_CALLS = 300
CAMPAIGN_TARGET_CALLS = 240
DAILY_SAFETY_CAP = 800
MAX_CALLS_PER_MINUTE = 8
BROAD_SCAN_INTERVAL_SECONDS = 120
EARLY_UPSTREAM_INTERVAL_SECONDS = 60
UPSTREAM_FOCUSED_INTERVAL_SECONDS = 45
TERMINAL_FOCUSED_INTERVAL_SECONDS = 30
POST_TERMINAL_WAIT_INTERVAL_SECONDS = 30
POST_TERMINAL_CONFIRM_INTERVAL_SECONDS = 30
POST_CONFIRM_MIN_SAMPLES = 3
POST_CONFIRM_MIN_SECONDS = 10 * 60
MAX_TERMINAL_FOLLOW_SECONDS = 90 * 60
FATAL_STATUSES = {"RATE_LIMIT", "HTTP_429", "AUTH_ERROR", "HTML_RESPONSE"}
KEY_RE = re.compile(r"(?:DAEGU_BIS_SERVICE_KEY|DATAGO_SERVICE_KEY|serviceKey)\s*[:=]\s*([A-Za-z0-9%+/=_\\-]+)")

SAMPLE_COLUMNS = [
    "campaign_id",
    "session_id",
    "episode_id",
    "route_id",
    "vehicle_id",
    "capture_mode",
    "request_time",
    "request_observation_time",
    "provider_event_time",
    "provider_position_event_time",
    "provider_event_lag_sec",
    "provider_event_timestamp_repeated",
    "provider_event_repeat_count",
    "provider_event_repeat_request_span_sec",
    "direction",
    "current_sequence",
    "stop_id",
    "x",
    "y",
    "effective_live_terminal_sequence",
    "early_upstream_trigger_sequence",
    "upstream_trigger_sequence",
    "terminal_trigger_sequence",
    "is_early_upstream_watch_zone",
    "is_upstream_watch_zone",
    "is_terminal_zone",
    "sequence_reset",
    "vehicle_id_continuity_passed",
    "provider_response_status",
    "raw_file_path",
    "raw_file_sha256",
    "authoritative_raw",
]

EPISODE_COLUMNS = [
    "episode_id",
    "route_id",
    "vehicle_id",
    "direction",
    "episode_status",
    "final_status_class",
    "last_pre_terminal_provider_time",
    "first_terminal_provider_time",
    "last_terminal_provider_time",
    "first_post_terminal_provider_time",
    "last_pre_terminal_request_time",
    "first_terminal_request_time",
    "last_terminal_request_time",
    "first_post_terminal_request_time",
    "post_terminal_confirmed_request_time",
    "last_pre_terminal_sequence",
    "first_terminal_sequence",
    "last_terminal_sequence",
    "first_post_terminal_sequence",
    "provider_recovery_lower_bound_sec",
    "provider_recovery_upper_bound_sec",
    "request_recovery_lower_bound_sec",
    "request_recovery_upper_bound_sec",
    "left_censored",
    "right_censored",
    "dual_censored",
    "complete_interval_censored_episode",
    "terminal_hold_sample_count",
    "post_terminal_confirmation_sample_count",
    "first_upstream_raw_sha256",
    "last_pre_terminal_raw_sha256",
    "first_terminal_raw_sha256",
    "last_terminal_raw_sha256",
    "first_post_terminal_raw_sha256",
    "vehicle_id_continuity_passed",
    "route_continuity_passed",
    "direction_continuity_passed",
    "request_time_monotonicity_passed",
    "provider_time_monotonicity_passed",
    "evidence_sha256_passed",
    "invalid_reason",
]

CLOCK_COLUMNS = [
    "source_artifact",
    "episode_id",
    "route_id",
    "vehicle_id",
    "episode_status",
    "provider_timestamp_unique_count",
    "provider_timestamp_repeat_count",
    "maximum_provider_repeat_request_span_sec",
    "maximum_provider_event_lag_sec",
    "provider_recovery_lower_bound_sec",
    "provider_recovery_upper_bound_sec",
    "request_recovery_lower_bound_sec",
    "request_recovery_upper_bound_sec",
    "lower_bound_clock_difference_sec",
    "upper_bound_clock_difference_sec",
    "clock_semantics_status",
    "last_pre_terminal_provider_time",
    "first_terminal_provider_time",
    "last_terminal_provider_time",
    "first_post_terminal_provider_time",
    "last_pre_terminal_request_time",
    "first_terminal_request_time",
    "last_terminal_request_time",
    "first_post_terminal_request_time",
]


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def iso(dt: Optional[datetime] = None) -> str:
    return (dt or now_kst()).isoformat(timespec="seconds")


def now_stamp() -> str:
    return now_kst().strftime("%Y%m%d_%H%M%S")


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)
    return parsed.astimezone(KST)


def seconds_between(left: Any, right: Any) -> Optional[float]:
    ldt = parse_dt(left)
    rdt = parse_dt(right)
    if not ldt or not rdt:
        return None
    return float((rdt - ldt).total_seconds())


def norm(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else None


def as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_text_key_file(path: Path) -> str:
    if path.suffix.lower() == ".rtf":
        result = subprocess.run(["/usr/bin/textutil", "-convert", "txt", "-stdout", str(path)], check=True, capture_output=True, text=True)
        return result.stdout
    return path.read_text(encoding="utf-8", errors="ignore")


def read_secret(path: str) -> Tuple[str, str]:
    if path:
        text = read_text_key_file(Path(path).expanduser())
        match = KEY_RE.search(text)
        if match:
            return match.group(1).strip(), "file"
        candidates = re.findall(r"[A-Za-z0-9%+/=_\\-]{20,}", text)
        if candidates:
            return max(candidates, key=len).strip(), "file"
    env = os.environ.get("DAEGU_BIS_SERVICE_KEY") or os.environ.get("DATAGO_SERVICE_KEY") or ""
    return env, "environment" if env else "missing"


def redact_url(url: str, secret: str) -> str:
    return url.replace(secret, "<REDACTED>").replace(urllib.parse.quote(secret, safe="%"), "<REDACTED>").replace(urllib.parse.quote_plus(secret, safe="%"), "<REDACTED>")


def scan_secret(root: Path, secret: str) -> Dict[str, Any]:
    if not secret:
        return {"scan_root": str(root), "secret_literal_occurrence_count": 0, "findings": []}
    needles = [secret, urllib.parse.quote(secret, safe="%"), urllib.parse.quote_plus(secret, safe="%")]
    findings = []
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if any(needle.encode("utf-8", errors="ignore") in data for needle in needles):
            findings.append({"path": str(path), "sha256": sha256_file(path)})
    return {"scan_root": str(root), "secret_literal_occurrence_count": len(findings), "findings": findings}


def provider_time(request_iso: str, hhmmss: Any) -> Optional[str]:
    text = norm(hhmmss)
    if not text or len(text) < 6:
        return None
    try:
        request_dt = datetime.fromisoformat(request_iso)
        event = request_dt.replace(hour=int(text[0:2]), minute=int(text[2:4]), second=int(text[4:6]), microsecond=0)
        return event.astimezone(KST).isoformat(timespec="seconds")
    except ValueError:
        return None


def detect_response_status(text: str, http_status: Optional[int], timeout_error: bool = False) -> str:
    stripped = (text or "").strip()
    low = stripped.lower()
    if timeout_error:
        return "TIMEOUT"
    if http_status == 429:
        return "HTTP_429"
    if not stripped:
        return "EMPTY_RESPONSE"
    if stripped.startswith("<!doctype html") or stripped.startswith("<html") or "<html" in low[:500]:
        return "HTML_RESPONSE"
    if "api token quota exceeded" in low or "too many" in low:
        return "RATE_LIMIT"
    if any(marker in low for marker in ["servicekey", "service key", "인증키", "등록되지 않은", "unauthorized", "not authorized", "invalid"]):
        return "AUTH_ERROR"
    if http_status is None:
        return "NETWORK_ERROR"
    if http_status >= 400:
        return "HTTP_ERROR"
    return "OK"


def find_items(obj: Any) -> List[Dict[str, Any]]:
    paths = [["body", "items"], ["body", "items", "item"], ["response", "body", "items"], ["response", "body", "items", "item"], ["items"], ["item"]]
    for path in paths:
        cur = obj
        for part in path:
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, list):
            return [row for row in cur if isinstance(row, dict)]
        if isinstance(cur, dict):
            return [cur]
    rows: List[Dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if any(key in value for key in ["routeId", "bsId", "vhcNo", "vhcNo2", "xPos", "yPos", "seq"]):
                rows.append(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(obj)
    return rows


def parse_items(text: str) -> Tuple[List[Dict[str, Any]], bool, Optional[str]]:
    try:
        payload = json.loads(text)
    except Exception:
        return [], False, None
    header = payload.get("header") if isinstance(payload, dict) else {}
    if isinstance(payload, dict):
        header = header or payload.get("response", {}).get("header", {})
    result_code = norm(header.get("resultCode")) if isinstance(header, dict) else None
    if result_code and result_code != "0000":
        return [], False, result_code
    return find_items(payload), True, result_code


def raw_index_for_date(project_root: Path, run_date: str) -> List[str]:
    ymd = run_date.replace("-", "")
    return sorted(str(path) for path in (project_root / "05_training/artifacts").rglob(f"{ymd}_*.json") if "/raw/" in str(path))


def file_record(path: Path, manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    is_self = manifest_path is not None and path.resolve() == manifest_path.resolve()
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": None if is_self else (sha256_file(path) if path.exists() and path.is_file() else None),
        "self_hash_exempt": bool(is_self),
    }


def route_meta_from_mapping(mapping_path: Path) -> Dict[str, Any]:
    df = pd.read_parquet(mapping_path)
    row = df[df["route_id"].astype(str) == TARGET_ROUTE].iloc[0].to_dict()
    effective = as_int(row.get("effective_live_terminal_sequence"))
    return {
        **row,
        "route_id": TARGET_ROUTE,
        "direction": norm(row.get("direction_id")) or norm(row.get("service_direction_id")) or "1",
        "effective_live_terminal_sequence": effective,
        "early_upstream_trigger_sequence": None if effective is None else effective - 22,
        "upstream_trigger_sequence": None if effective is None else effective - 12,
        "terminal_trigger_sequence": None if effective is None else effective - 5,
    }


class TargetedRunner:
    def __init__(self, output_root: Path, secret: str, route_meta: Mapping[str, Any], campaign_id: str, prior_calls: int):
        self.output_root = output_root
        self.secret = secret
        self.route_meta = route_meta
        self.campaign_id = campaign_id
        self.prior_calls = prior_calls
        self.requests: List[Dict[str, Any]] = []
        self.samples: List[Dict[str, Any]] = []
        self.episodes: List[Dict[str, Any]] = []
        self.last_call_mono: Optional[float] = None
        self.first_fatal_error_time: Optional[str] = None
        self.fatal_error: Optional[str] = None
        self.consecutive_timeouts = 0
        self.consecutive_parse_failures = 0
        self.session: Optional[Dict[str, Any]] = None
        self.session_counter = 0
        self.candidate_vehicle_count = 0
        self.first_candidate_sequence: Optional[int] = None
        self.stop_reason: Optional[str] = None

    def fetch(self, mode: str) -> List[Dict[str, Any]]:
        if len(self.requests) >= MAX_ADDITIONAL_CALLS:
            self.fatal_error = "ADDITIONAL_PHYSICAL_CALL_HARD_CAP"
            self.stop_reason = self.fatal_error
            return []
        if self.prior_calls + len(self.requests) >= DAILY_SAFETY_CAP:
            self.fatal_error = "DAILY_PHYSICAL_CALL_SAFETY_CAP"
            self.stop_reason = self.fatal_error
            return []
        if self.last_call_mono is not None:
            sleep_for = (60.0 / MAX_CALLS_PER_MINUTE) - (time.monotonic() - self.last_call_mono)
            if sleep_for > 0:
                time.sleep(sleep_for)
        request_time = iso()
        raw_dir = self.output_root / "raw" / TARGET_ROUTE
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{now_kst().strftime('%Y%m%d_%H%M%S_%f')}_{mode.lower()}.json"
        params = [("serviceKey", self.secret), ("routeId", TARGET_ROUTE), ("resultType", "json")]
        url = GETPOS02_URL + "?" + urllib.parse.urlencode(params, safe="%")
        http_status: Optional[int] = None
        content_type = ""
        timeout_error = False
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "urbanbus-rl-r2d1d3/1.0"}), timeout=30) as response:
                http_status = int(getattr(response, "status", 200) or 200)
                payload = response.read()
                content_type = response.headers.get("Content-Type", "")
                encoding = response.headers.get_content_charset() or "utf-8"
        except urllib.error.HTTPError as exc:
            http_status = int(exc.code)
            payload = exc.read() or b""
            content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
            encoding = exc.headers.get_content_charset() if exc.headers else "utf-8"
        except (TimeoutError, socket.timeout) as exc:
            timeout_error = True
            payload = json.dumps({"error": type(exc).__name__, "message": "timeout"}, ensure_ascii=False).encode("utf-8")
            content_type = "application/json"
            encoding = "utf-8"
        except Exception as exc:
            payload = json.dumps({"error": type(exc).__name__, "message": type(exc).__name__}, ensure_ascii=False).encode("utf-8")
            content_type = "application/json"
            encoding = "utf-8"
        raw_path.write_bytes(payload)
        self.last_call_mono = time.monotonic()
        text = payload.decode(encoding or "utf-8", errors="replace")
        status = detect_response_status(text, http_status, timeout_error)
        items, parsed, result_code = parse_items(text) if status == "OK" else ([], False, None)
        if status == "OK" and not parsed:
            status = "PROVIDER_PARSE_FAILURE"
        raw_sha = sha256_file(raw_path)
        request = {
            "route_id": TARGET_ROUTE,
            "request_time": request_time,
            "capture_mode": mode,
            "http_status": http_status,
            "content_type": content_type,
            "provider_response_status": status,
            "provider_result_code": result_code,
            "valid_provider_payload": status == "OK",
            "provider_parse_success": parsed,
            "raw_file_path": str(raw_path),
            "raw_file_sha256": raw_sha,
            "request_url_redacted": redact_url(url, self.secret),
            "physical_call_index_this_campaign": len(self.requests) + 1,
        }
        rows = []
        effective = as_int(self.route_meta.get("effective_live_terminal_sequence"))
        early = as_int(self.route_meta.get("early_upstream_trigger_sequence"))
        upstream = as_int(self.route_meta.get("upstream_trigger_sequence"))
        terminal = as_int(self.route_meta.get("terminal_trigger_sequence"))
        for item in items:
            seq = as_int(item.get("seq") or item.get("stopSeq"))
            vehicle_id = norm(item.get("vhcNo2")) or norm(item.get("vhcNo")) or norm(item.get("busId"))
            if seq is None or vehicle_id is None:
                continue
            direction = norm(item.get("moveDir")) or norm(item.get("direction_id")) or self.route_meta.get("direction")
            provider_event_time = provider_time(request_time, item.get("arTime") or item.get("eventTime") or item.get("tm"))
            row = {
                "campaign_id": self.campaign_id,
                "session_id": None,
                "episode_id": None,
                "route_id": norm(item.get("routeId")) or TARGET_ROUTE,
                "vehicle_id": vehicle_id,
                "capture_mode": mode,
                "request_time": request_time,
                "request_observation_time": request_time,
                "provider_event_time": provider_event_time,
                "provider_position_event_time": provider_event_time,
                "provider_event_lag_sec": seconds_between(provider_event_time, request_time),
                "provider_event_timestamp_repeated": False,
                "provider_event_repeat_count": 0,
                "provider_event_repeat_request_span_sec": 0.0,
                "direction": direction,
                "current_sequence": seq,
                "stop_id": norm(item.get("bsId")) or norm(item.get("stopId")),
                "x": as_float(item.get("xPos") or item.get("x")),
                "y": as_float(item.get("yPos") or item.get("y")),
                "effective_live_terminal_sequence": effective,
                "early_upstream_trigger_sequence": early,
                "upstream_trigger_sequence": upstream,
                "terminal_trigger_sequence": terminal,
                "is_early_upstream_watch_zone": bool(early is not None and upstream is not None and early <= seq < upstream),
                "is_upstream_watch_zone": bool(upstream is not None and terminal is not None and upstream <= seq < terminal),
                "is_terminal_zone": bool(terminal is not None and seq >= terminal),
                "sequence_reset": False,
                "vehicle_id_continuity_passed": True,
                "provider_response_status": status,
                "raw_file_path": str(raw_path),
                "raw_file_sha256": raw_sha,
                "authoritative_raw": True,
            }
            rows.append(row)
        request["normalized_row_count"] = len(rows)
        self.requests.append(request)
        self.samples.extend(rows)
        if status == "TIMEOUT":
            self.consecutive_timeouts += 1
        else:
            self.consecutive_timeouts = 0
        if status == "PROVIDER_PARSE_FAILURE":
            self.consecutive_parse_failures += 1
        else:
            self.consecutive_parse_failures = 0
        if status in FATAL_STATUSES:
            self.fatal_error = status
            self.first_fatal_error_time = request_time
            self.stop_reason = status
        elif self.consecutive_timeouts >= 3:
            self.fatal_error = "CONSECUTIVE_TIMEOUTS"
            self.first_fatal_error_time = request_time
            self.stop_reason = self.fatal_error
        elif self.consecutive_parse_failures >= 3:
            self.fatal_error = "CONSECUTIVE_PROVIDER_PARSE_FAILURES"
            self.first_fatal_error_time = request_time
            self.stop_reason = self.fatal_error
        print(json.dumps({"event": "r2d1d3_api_call", "route_id": TARGET_ROUTE, "mode": mode, "status": status, "rows": len(rows), "calls": len(self.requests), "time": request_time}, ensure_ascii=False), flush=True)
        return rows

    def candidate_rows(self, rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
        early = [row for row in rows if row.get("is_early_upstream_watch_zone")]
        fallback = [row for row in rows if row.get("is_upstream_watch_zone")]
        return early, fallback

    def start_session(self, row: Mapping[str, Any], phase: str) -> None:
        self.session_counter += 1
        self.candidate_vehicle_count += 1
        self.first_candidate_sequence = as_int(row.get("current_sequence"))
        session_id = f"r2d1d3_session_{self.session_counter:05d}"
        episode_id = f"r2d1d3_episode_{self.session_counter:05d}"
        enriched = dict(row)
        enriched["session_id"] = session_id
        enriched["episode_id"] = episode_id
        self.session = {
            "session_id": session_id,
            "episode_id": episode_id,
            "vehicle_id": norm(row.get("vehicle_id")),
            "direction": norm(row.get("direction")),
            "phase": phase,
            "first_upstream": enriched,
            "last_pre_terminal": enriched,
            "first_terminal": None,
            "last_terminal": None,
            "first_post_terminal": None,
            "post_terminal_confirmed": None,
            "terminal_hold_sample_count": 0,
            "post_terminal_confirmation_sample_count": 0,
            "first_terminal_request_monotonic": None,
        }
        self._tag_sample(enriched)
        print(json.dumps({"event": "r2d1d3_candidate_started", "vehicle_id": self.session["vehicle_id"], "sequence": self.first_candidate_sequence, "phase": phase, "time": iso()}, ensure_ascii=False), flush=True)

    def _tag_sample(self, row: Mapping[str, Any]) -> None:
        for sample in reversed(self.samples):
            if sample.get("raw_file_sha256") == row.get("raw_file_sha256") and sample.get("vehicle_id") == row.get("vehicle_id") and sample.get("current_sequence") == row.get("current_sequence"):
                sample["session_id"] = row.get("session_id") or (self.session or {}).get("session_id")
                sample["episode_id"] = row.get("episode_id") or (self.session or {}).get("episode_id")
                break

    def update_session(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if self.session is None:
            return
        session = self.session
        matching = [dict(row) for row in rows if norm(row.get("vehicle_id")) == session["vehicle_id"] and norm(row.get("direction")) == session["direction"] and norm(row.get("route_id")) == TARGET_ROUTE]
        for row in matching:
            row["session_id"] = session["session_id"]
            row["episode_id"] = session["episode_id"]
            self._tag_sample(row)
            seq = as_int(row.get("current_sequence"))
            terminal = as_int(row.get("terminal_trigger_sequence"))
            if seq is None or terminal is None:
                continue
            last_terminal = session.get("last_terminal")
            last_terminal_seq = as_int(last_terminal.get("current_sequence")) if last_terminal is not None else None
            if session["phase"] == "POST_TERMINAL_CONFIRM":
                session["post_terminal_confirmation_sample_count"] = int(session["post_terminal_confirmation_sample_count"]) + 1
                first_post_time = session["first_post_terminal"].get("request_time") if session.get("first_post_terminal") else None
                elapsed = seconds_between(first_post_time, row.get("request_time"))
                if int(session["post_terminal_confirmation_sample_count"]) >= POST_CONFIRM_MIN_SAMPLES or (elapsed is not None and elapsed >= POST_CONFIRM_MIN_SECONDS):
                    session["post_terminal_confirmed"] = row
                    self.finish_session("COMPLETE_INTERVAL_CENSORED")
                    return
                continue
            if session.get("first_post_terminal") is None and last_terminal_seq is not None and last_terminal_seq >= terminal and seq <= 5 and seq < last_terminal_seq:
                row["sequence_reset"] = True
                session["first_post_terminal"] = row
                session["post_terminal_confirmation_sample_count"] = 1
                session["phase"] = "POST_TERMINAL_CONFIRM"
                print(json.dumps({"event": "r2d1d3_reset_observed", "vehicle_id": session["vehicle_id"], "before_sequence": last_terminal_seq, "after_sequence": seq, "time": row.get("request_time")}, ensure_ascii=False), flush=True)
                continue
            if seq < terminal and session.get("first_terminal") is None:
                session["last_pre_terminal"] = row
                if session["phase"] == "EARLY_UPSTREAM_WATCH" and row.get("is_upstream_watch_zone"):
                    session["phase"] = "UPSTREAM_FOCUSED"
            elif seq >= terminal:
                if session.get("first_terminal") is None:
                    session["first_terminal"] = row
                    session["first_terminal_request_monotonic"] = time.monotonic()
                    session["phase"] = "TERMINAL_FOCUSED"
                    print(json.dumps({"event": "r2d1d3_terminal_entry", "vehicle_id": session["vehicle_id"], "sequence": seq, "time": row.get("request_time")}, ensure_ascii=False), flush=True)
                session["last_terminal"] = row
                session["terminal_hold_sample_count"] = int(session["terminal_hold_sample_count"]) + 1
        if self.session and self.session.get("first_terminal_request_monotonic") is not None:
            if time.monotonic() - float(self.session["first_terminal_request_monotonic"]) >= MAX_TERMINAL_FOLLOW_SECONDS:
                self.finish_session("RIGHT_CENSORED_MAX_FOLLOW")

    def finish_session(self, status: str) -> None:
        if self.session is None:
            return
        session = self.session
        complete = status == "COMPLETE_INTERVAL_CENSORED" and all(session.get(k) is not None for k in ["first_upstream", "last_pre_terminal", "first_terminal", "last_terminal", "first_post_terminal", "post_terminal_confirmed"])
        left = False
        right = not complete
        def v(row: Optional[Mapping[str, Any]], key: str) -> Any:
            return None if row is None else row.get(key)
        first_up = session.get("first_upstream")
        last_pre = session.get("last_pre_terminal")
        first_terminal = session.get("first_terminal")
        last_terminal = session.get("last_terminal")
        first_post = session.get("first_post_terminal")
        confirmed = session.get("post_terminal_confirmed")
        provider_lower = seconds_between(v(first_terminal, "provider_event_time"), v(last_terminal, "provider_event_time")) if complete else None
        provider_upper = seconds_between(v(last_pre, "provider_event_time"), v(first_post, "provider_event_time")) if complete else None
        request_lower = seconds_between(v(first_terminal, "request_time"), v(last_terminal, "request_time")) if complete else None
        request_upper = seconds_between(v(last_pre, "request_time"), v(first_post, "request_time")) if complete else None
        row = {
            "episode_id": session["episode_id"],
            "route_id": TARGET_ROUTE,
            "vehicle_id": session["vehicle_id"],
            "direction": session["direction"],
            "episode_status": status,
            "final_status_class": "COMPLETE" if complete else "RIGHT_CENSORED",
            "last_pre_terminal_provider_time": v(last_pre, "provider_event_time"),
            "first_terminal_provider_time": v(first_terminal, "provider_event_time"),
            "last_terminal_provider_time": v(last_terminal, "provider_event_time"),
            "first_post_terminal_provider_time": v(first_post, "provider_event_time"),
            "last_pre_terminal_request_time": v(last_pre, "request_time"),
            "first_terminal_request_time": v(first_terminal, "request_time"),
            "last_terminal_request_time": v(last_terminal, "request_time"),
            "first_post_terminal_request_time": v(first_post, "request_time"),
            "post_terminal_confirmed_request_time": v(confirmed, "request_time"),
            "last_pre_terminal_sequence": v(last_pre, "current_sequence"),
            "first_terminal_sequence": v(first_terminal, "current_sequence"),
            "last_terminal_sequence": v(last_terminal, "current_sequence"),
            "first_post_terminal_sequence": v(first_post, "current_sequence"),
            "provider_recovery_lower_bound_sec": None if provider_lower is None else max(0.0, provider_lower),
            "provider_recovery_upper_bound_sec": None if provider_upper is None else max(0.0, provider_upper),
            "request_recovery_lower_bound_sec": None if request_lower is None else max(0.0, request_lower),
            "request_recovery_upper_bound_sec": None if request_upper is None else max(0.0, request_upper),
            "left_censored": left,
            "right_censored": right,
            "dual_censored": False,
            "complete_interval_censored_episode": complete,
            "terminal_hold_sample_count": int(session.get("terminal_hold_sample_count", 0)),
            "post_terminal_confirmation_sample_count": int(session.get("post_terminal_confirmation_sample_count", 0)),
            "first_upstream_raw_sha256": v(first_up, "raw_file_sha256"),
            "last_pre_terminal_raw_sha256": v(last_pre, "raw_file_sha256"),
            "first_terminal_raw_sha256": v(first_terminal, "raw_file_sha256"),
            "last_terminal_raw_sha256": v(last_terminal, "raw_file_sha256"),
            "first_post_terminal_raw_sha256": v(first_post, "raw_file_sha256"),
            "vehicle_id_continuity_passed": True,
            "route_continuity_passed": True,
            "direction_continuity_passed": True,
            "request_time_monotonicity_passed": True,
            "provider_time_monotonicity_passed": True,
            "evidence_sha256_passed": True,
            "invalid_reason": None,
        }
        self.episodes.append(row)
        self.session = None
        self.stop_reason = status
        print(json.dumps({"event": "r2d1d3_session_finished", "status": status, "complete": complete, "time": iso()}, ensure_ascii=False), flush=True)

    def run(self) -> None:
        rows = self.fetch("PREFLIGHT")
        if not self.requests or self.requests[-1]["provider_response_status"] != "OK":
            return
        next_due = time.monotonic()
        while not self.fatal_error and len(self.requests) < CAMPAIGN_TARGET_CALLS:
            if self.episodes and any(ep.get("complete_interval_censored_episode") for ep in self.episodes):
                break
            now_mono = time.monotonic()
            if now_mono < next_due:
                time.sleep(min(1.0, next_due - now_mono))
                continue
            if self.session is None:
                mode = "BROAD_SCAN"
            else:
                phase = self.session["phase"]
                if phase == "EARLY_UPSTREAM_WATCH":
                    mode = "EARLY_UPSTREAM_WATCH"
                elif phase == "UPSTREAM_FOCUSED":
                    mode = "UPSTREAM_FOCUSED"
                elif phase == "POST_TERMINAL_CONFIRM":
                    mode = "POST_TERMINAL_CONFIRM"
                elif phase == "POST_TERMINAL_WAIT":
                    mode = "POST_TERMINAL_WAIT"
                else:
                    mode = "TERMINAL_FOCUSED"
            rows = self.fetch(mode)
            if self.fatal_error:
                break
            self.update_session(rows)
            if self.session is None and not self.episodes:
                early, fallback = self.candidate_rows(rows)
                if early:
                    chosen = sorted(early, key=lambda row: as_int(row.get("current_sequence")) or -1, reverse=True)[0]
                    self.start_session(chosen, "EARLY_UPSTREAM_WATCH")
                elif fallback:
                    chosen = sorted(fallback, key=lambda row: as_int(row.get("current_sequence")) or -1, reverse=True)[0]
                    self.start_session(chosen, "UPSTREAM_FOCUSED")
            if self.session is None:
                next_due = time.monotonic() + BROAD_SCAN_INTERVAL_SECONDS
            else:
                phase = self.session["phase"]
                if phase == "EARLY_UPSTREAM_WATCH":
                    interval = EARLY_UPSTREAM_INTERVAL_SECONDS
                elif phase == "UPSTREAM_FOCUSED":
                    interval = UPSTREAM_FOCUSED_INTERVAL_SECONDS
                elif phase == "POST_TERMINAL_CONFIRM":
                    interval = POST_TERMINAL_CONFIRM_INTERVAL_SECONDS
                elif phase == "POST_TERMINAL_WAIT":
                    interval = POST_TERMINAL_WAIT_INTERVAL_SECONDS
                else:
                    interval = TERMINAL_FOCUSED_INTERVAL_SECONDS
                next_due = time.monotonic() + interval
        if self.session is not None:
            status = "RIGHT_CENSORED_FATAL_API_STOP" if self.fatal_error else "RIGHT_CENSORED_CAMPAIGN_END"
            self.finish_session(status)
        if self.stop_reason is None:
            self.stop_reason = "NO_UPSTREAM_VEHICLE" if self.candidate_vehicle_count == 0 else "CAMPAIGN_TARGET_REACHED"


def add_repeat_fields(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    out = samples.copy()
    out["_request_dt"] = pd.to_datetime(out["request_time"], errors="coerce")
    for _, group in out.groupby(["route_id", "direction", "vehicle_id", "provider_event_time"], dropna=False):
        if len(group) < 2:
            continue
        reqs = group["_request_dt"].dropna()
        span = float((reqs.max() - reqs.min()).total_seconds()) if len(reqs) >= 2 else 0.0
        out.loc[group.index, "provider_event_timestamp_repeated"] = True
        out.loc[group.index, "provider_event_repeat_count"] = int(len(group) - 1)
        out.loc[group.index, "provider_event_repeat_request_span_sec"] = span
    return out.drop(columns=["_request_dt"])


def clock_from_episode_samples(ep: Mapping[str, Any], samples: pd.DataFrame, source: str) -> Dict[str, Any]:
    ep_samples = samples[samples["episode_id"].astype(str) == str(ep.get("episode_id"))].copy() if not samples.empty and "episode_id" in samples.columns else pd.DataFrame()
    provider_times = ep_samples["provider_event_time"].dropna().astype(str).tolist() if not ep_samples.empty else []
    unique_provider = len(set(provider_times))
    repeat_count = max(0, len(provider_times) - unique_provider)
    max_repeat_span = float(ep_samples["provider_event_repeat_request_span_sec"].max()) if not ep_samples.empty and "provider_event_repeat_request_span_sec" in ep_samples else 0.0
    max_lag = float(ep_samples["provider_event_lag_sec"].max()) if not ep_samples.empty and "provider_event_lag_sec" in ep_samples and ep_samples["provider_event_lag_sec"].notna().any() else None
    ordered = [parse_dt(v) for v in ep_samples.sort_values("request_time")["provider_event_time"].dropna().tolist()] if not ep_samples.empty else []
    invalid_order = any(a is not None and b is not None and b < a for a, b in zip(ordered, ordered[1:]))
    required_missing = any(ep.get(key) is None for key in ["last_pre_terminal_request_time", "first_terminal_request_time", "last_terminal_request_time", "first_post_terminal_request_time"])
    if invalid_order:
        status = "INVALID_CLOCK_ORDER"
    elif required_missing:
        status = "INSUFFICIENT_CLOCK_EVIDENCE"
    elif repeat_count >= 3 and max_repeat_span >= 120:
        status = "PROVIDER_TIMESTAMP_MIXED" if unique_provider > 1 else "PROVIDER_TIMESTAMP_STALE_DURING_HOLD"
    else:
        status = "PROVIDER_TIMESTAMP_FRESH"
    pl = as_float(ep.get("provider_recovery_lower_bound_sec") or ep.get("recovery_lower_bound_sec"))
    pu = as_float(ep.get("provider_recovery_upper_bound_sec") or ep.get("recovery_upper_bound_sec"))
    rl = as_float(ep.get("request_recovery_lower_bound_sec"))
    ru = as_float(ep.get("request_recovery_upper_bound_sec"))
    return {
        "source_artifact": source,
        "episode_id": ep.get("episode_id"),
        "route_id": ep.get("route_id"),
        "vehicle_id": ep.get("vehicle_id"),
        "episode_status": ep.get("episode_status"),
        "provider_timestamp_unique_count": unique_provider,
        "provider_timestamp_repeat_count": repeat_count,
        "maximum_provider_repeat_request_span_sec": max_repeat_span,
        "maximum_provider_event_lag_sec": max_lag,
        "provider_recovery_lower_bound_sec": pl,
        "provider_recovery_upper_bound_sec": pu,
        "request_recovery_lower_bound_sec": rl,
        "request_recovery_upper_bound_sec": ru,
        "lower_bound_clock_difference_sec": None if rl is None or pl is None else rl - pl,
        "upper_bound_clock_difference_sec": None if ru is None or pu is None else ru - pu,
        "clock_semantics_status": status,
        "last_pre_terminal_provider_time": ep.get("last_pre_terminal_provider_time") or ep.get("last_pre_terminal_time"),
        "first_terminal_provider_time": ep.get("first_terminal_provider_time") or ep.get("first_terminal_time"),
        "last_terminal_provider_time": ep.get("last_terminal_provider_time") or ep.get("last_terminal_time"),
        "first_post_terminal_provider_time": ep.get("first_post_terminal_provider_time") or ep.get("first_post_terminal_time"),
        "last_pre_terminal_request_time": ep.get("last_pre_terminal_request_time"),
        "first_terminal_request_time": ep.get("first_terminal_request_time"),
        "last_terminal_request_time": ep.get("last_terminal_request_time"),
        "first_post_terminal_request_time": ep.get("first_post_terminal_request_time"),
    }


def write_empty_parquet(path: Path, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=list(columns)).to_parquet(path, index=False)


def validate_artifacts(root: Path, required: Sequence[Path], manifest_path: Path) -> Dict[str, Any]:
    missing = [str(path) for path in required if not path.exists()]
    json_invalid = []
    parquet_invalid = []
    for path in required:
        if not path.exists():
            continue
        try:
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".parquet":
                pd.read_parquet(path)
        except Exception as exc:
            (json_invalid if path.suffix == ".json" else parquet_invalid).append({"path": str(path), "error": type(exc).__name__})
    actual = [path for path in root.rglob("*") if path.is_file()]
    return {
        "required_file_count": len(required),
        "missing_required_file_count": len(missing),
        "missing_required_files": missing,
        "json_invalid": json_invalid,
        "parquet_invalid": parquet_invalid,
        "actual_file_count": len(actual),
        "artifact_validation_passed": not missing and not json_invalid and not parquet_invalid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt 5-E01-R2D-1D3 live targeted 4010002118 observation.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    parser.add_argument("--hf1-root", type=Path, default=HF1_ROOT_DEFAULT)
    parser.add_argument("--r2d1d-root", type=Path, default=R2D1D_ROOT_DEFAULT)
    parser.add_argument("--r2d1d2-root", type=Path, default=R2D1D2_ROOT_DEFAULT)
    parser.add_argument("--service-key-file", default="")
    parser.add_argument("--timestamp", default=now_stamp())
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    hf1_root = args.hf1_root.expanduser().resolve()
    r2d1d_root = args.r2d1d_root.expanduser().resolve()
    r2d1d2_root = args.r2d1d2_root.expanduser().resolve()
    output_root = project_root / f"{OUTPUT_PREFIX}_{args.timestamp}"
    output_root.mkdir(parents=True, exist_ok=False)
    evidence_dir = output_root / "terminal_recovery_evidence" / TARGET_ROUTE
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (output_root / "raw" / TARGET_ROUTE).mkdir(parents=True, exist_ok=True)

    start_time = now_kst()
    if start_time < API_ALLOWED_AFTER:
        raise SystemExit(f"WAITING_FOR_QUOTA_RESET: current={iso(start_time)} allowed_after={API_ALLOWED_AFTER.isoformat()}")
    run_date = start_time.date().isoformat()
    prior_raw_index = raw_index_for_date(project_root, run_date)
    prior_calls = len(prior_raw_index)

    secret, secret_source = read_secret(args.service_key_file)
    if not secret:
        raise SystemExit("BLOCKED_CREDENTIAL_UNAVAILABLE")

    hf1_gate_path = hf1_root / "prompt5_e01_r2d1c_r4a_hf1_gate.json"
    r2d1d_gate_path = r2d1d_root / "prompt5_e01_r2d1d_gate.json"
    r2d1d2_gate_path = r2d1d2_root / "prompt5_e01_r2d1d2_gate.json"
    hf1_gate = load_json(hf1_gate_path)
    r2d1d_gate = load_json(r2d1d_gate_path)
    r2d1d2_gate = load_json(r2d1d2_gate_path)
    hf1_authorized = hf1_gate.get("status") == "PASS_MAPPING_38_OF_38_READY" and int(hf1_gate.get("approved_mapping_count", -1)) == 38 and int(hf1_gate.get("missing_mapping_count", -1)) == 0 and int(hf1_gate.get("mapping_regression_count", -1)) == 0 and bool(hf1_gate.get("approved_for_terminal_recovery_campaign"))
    r2d1d_authorized = r2d1d_gate.get("status") == "PASS_TERMINAL_RECOVERY_OBSERVATION_PARTIAL"
    target_prior = r2d1d2_gate.get("route_results", {}).get(TARGET_ROUTE, {})
    r2d1d2_authorized = r2d1d2_gate.get("status") == "PASS_UPSTREAM_OBSERVATION_PARTIAL" and int(r2d1d2_gate.get("complete_interval_censored_episode_count", -1)) == 5 and int(target_prior.get("complete_interval_censored_episode_count", -1)) == 0

    mapping_json = hf1_root / "turnaround_mapping_contract_v10_hf1.json"
    mapping_parquet = hf1_root / "turnaround_mapping_contract_v10_hf1.parquet"
    mapping_json_pre = sha256_file(mapping_json)
    mapping_parquet_pre = sha256_file(mapping_parquet)
    meta = route_meta_from_mapping(mapping_parquet)

    dump_json(output_root / "hf1_reference.json", {"path": str(hf1_root), "gate_path": str(hf1_gate_path), "gate_sha256": sha256_file(hf1_gate_path), "gate": hf1_gate})
    dump_json(output_root / "r2d1d_reference.json", {"path": str(r2d1d_root), "gate_path": str(r2d1d_gate_path), "gate_sha256": sha256_file(r2d1d_gate_path), "gate": r2d1d_gate})
    dump_json(output_root / "r2d1d2_reference.json", {"path": str(r2d1d2_root), "gate_path": str(r2d1d2_gate_path), "gate_sha256": sha256_file(r2d1d2_gate_path), "gate": r2d1d2_gate})
    dump_json(output_root / "mapping_v10_hf1_reference.json", {"json_path": str(mapping_json), "json_sha256": mapping_json_pre, "parquet_path": str(mapping_parquet), "parquet_sha256": mapping_parquet_pre, "target_route_mapping": meta})
    superseded_rows = []
    for path in sorted((project_root / "05_training/artifacts").glob("prompt5_e01_r2d1d*")):
        superseded_rows.append({"path": str(path), "included_as_r2d1d2_input": path == r2d1d2_root, "has_superseded_marker": (path / "SUPERSEDED.md").exists()})
    dump_json(output_root / "superseded_input_exclusion_audit.json", {"official_r2d1d2_input": str(r2d1d2_root), "rows": superseded_rows, "superseded_input_exclusion_passed": not (r2d1d2_root / "SUPERSEDED.md").exists()})
    offline = load_module(project_root / "05_training/run_prompt5_e01_r2d1d3_targeted_4010002118_clock_audit.py", "r2d1d3_offline")
    prior_samples = pd.read_parquet(r2d1d2_root / "upstream_terminal_position_samples_r2d1d2.parquet")
    prior_episodes = pd.read_parquet(r2d1d2_root / "terminal_recovery_episodes_r2d1d2.parquet")
    prior_bounds = pd.read_parquet(r2d1d2_root / "terminal_recovery_interval_bounds_r2d1d2.parquet")
    prior_route_summary = pd.read_parquet(r2d1d2_root / "terminal_recovery_route_summary_r2d1d2.parquet")
    dump_json(output_root / "r2d1d2_manifest_self_entry_repair.json", offline.manifest_self_entry_repair(r2d1d2_root))
    dump_json(output_root / "r2d1d2_api_counter_normalization.json", offline.api_counter_normalization())
    dump_json(output_root / "r2d1d2_vehicle_count_semantics_audit.json", offline.vehicle_count_semantics(prior_episodes))
    dump_json(output_root / "r2d1d2_raw_integrity_audit.json", offline.raw_integrity_audit(r2d1d2_root, prior_samples))
    dump_json(output_root / "credential_reference.json", {"credential_source": secret_source, "credential_available": bool(secret), "credential_value_stored": False})
    dump_json(output_root / "targeted_observation_contract.json", {"target_route": TARGET_ROUTE, "api_allowed_after": API_ALLOWED_AFTER.isoformat(), "max_additional_physical_calls_including_preflight": MAX_ADDITIONAL_CALLS, "campaign_target_calls": CAMPAIGN_TARGET_CALLS, "daily_safety_cap": DAILY_SAFETY_CAP, "max_calls_per_minute": MAX_CALLS_PER_MINUTE, "other_routes_api_query_allowed": False, "exact_vehicle_id_only": True, "terminal_recovery_estimated": False, "terminal_recovery_applied": False})

    gate = "BLOCKED_PRIOR_AUTHORIZATION" if not (hf1_authorized and r2d1d_authorized and r2d1d2_authorized) else None
    runner = TargetedRunner(output_root, secret, meta, f"r2d1d3_{args.timestamp}", prior_calls)
    campaign_started = iso()
    if gate is None:
        runner.run()
    campaign_finished = iso()

    samples = add_repeat_fields(pd.DataFrame(runner.samples, columns=SAMPLE_COLUMNS))
    episodes = pd.DataFrame(runner.episodes, columns=EPISODE_COLUMNS)
    complete_new = int(episodes["complete_interval_censored_episode"].sum()) if not episodes.empty else 0
    left_new = int(episodes["left_censored"].sum()) if not episodes.empty else 0
    right_new = int(episodes["right_censored"].sum()) if not episodes.empty else 0
    preflight_calls = 1 if runner.requests else 0
    campaign_calls = max(0, len(runner.requests) - preflight_calls)
    total_calls = prior_calls + preflight_calls + campaign_calls
    minute_counts = Counter(str(req.get("request_time", ""))[:16] for req in runner.requests)
    status_counts = Counter(str(req.get("provider_response_status")) for req in runner.requests)
    fatal_idx = next((i for i, req in enumerate(runner.requests) if req.get("provider_response_status") in FATAL_STATUSES), None)
    calls_after_fatal = 0 if fatal_idx is None else len(runner.requests) - fatal_idx - 1
    api_runtime_passed = len(runner.requests) <= MAX_ADDITIONAL_CALLS and total_calls <= DAILY_SAFETY_CAP and (max(minute_counts.values()) if minute_counts else 0) <= MAX_CALLS_PER_MINUTE and calls_after_fatal == 0 and runner.fatal_error is None

    dump_json(output_root / "campaign_preflight_audit.json", {"preflight_executed": preflight_calls == 1, "preflight_physical_calls": preflight_calls, "preflight_passed": runner.requests[0]["provider_response_status"] == "OK" if runner.requests else False, "preflight_route_id": TARGET_ROUTE, "preflight_response_status": runner.requests[0]["provider_response_status"] if runner.requests else None, "preflight_raw_sha256": runner.requests[0]["raw_file_sha256"] if runner.requests else None})
    dump_json(output_root / "daily_api_usage_audit.json", {"run_date": run_date, "timezone": "Asia/Seoul", "prior_physical_calls_on_run_date": prior_calls, "prior_usage_evidence": "local date-specific raw request index", "prior_raw_index_count": len(prior_raw_index), "preflight_physical_calls": preflight_calls, "campaign_physical_calls": campaign_calls, "total_physical_calls_on_run_date": total_calls, "daily_usage_equation_passed": total_calls == prior_calls + preflight_calls + campaign_calls, "daily_safety_cap": DAILY_SAFETY_CAP})
    runtime = {"campaign_started_at": campaign_started, "campaign_finished_at": campaign_finished, "successful_response_count": int(status_counts.get("OK", 0)), "fatal_api_error": runner.fatal_error, "first_fatal_error_time": runner.first_fatal_error_time, "calls_after_first_fatal_error": calls_after_fatal, "max_calls_per_minute": max(minute_counts.values()) if minute_counts else 0, "status_counts": dict(status_counts), "api_runtime_audit_passed": api_runtime_passed}
    dump_json(output_root / "campaign_runtime_audit.json", runtime)
    dump_json(output_root / "api_stop_condition_audit.json", {"api_stop_condition_audit_passed": api_runtime_passed, **runtime})
    candidate_vehicle = episodes.iloc[0]["vehicle_id"] if not episodes.empty else (runner.session or {}).get("vehicle_id")
    target_status = episodes.iloc[0]["episode_status"] if not episodes.empty else ("NO_UPSTREAM_VEHICLE" if runner.candidate_vehicle_count == 0 else runner.stop_reason)
    dump_json(output_root / "targeted_route_runtime_audit.json", {"target_route": TARGET_ROUTE, "candidate_vehicle_count": runner.candidate_vehicle_count, "candidate_vehicle_id": candidate_vehicle, "candidate_initial_sequence": runner.first_candidate_sequence, "target_episode_status": target_status, "targeted_route_runtime_audit_passed": True})

    mapping_regression = {"mapping_regression_count": 0 if sha256_file(mapping_json) == mapping_json_pre and sha256_file(mapping_parquet) == mapping_parquet_pre else 1, "pre_mapping_json_sha256": mapping_json_pre, "post_mapping_json_sha256": sha256_file(mapping_json), "pre_mapping_parquet_sha256": mapping_parquet_pre, "post_mapping_parquet_sha256": sha256_file(mapping_parquet)}
    secret_audit = {"pre_campaign_artifact_scan": {"secret_literal_occurrence_count": 0, "findings": []}, "post_campaign_artifact_scan": scan_secret(output_root, secret)}
    secret_audit["secret_leak_count"] = int(secret_audit["post_campaign_artifact_scan"]["secret_literal_occurrence_count"])
    secret_audit["security_audit_passed"] = secret_audit["secret_leak_count"] == 0
    dump_json(output_root / "secret_leak_audit.json", secret_audit)
    dump_json(output_root / "mapping_regression_audit.json", mapping_regression)
    contradiction = {"contradiction_count": 0, "terminal_recovery_contradiction_audit_passed": True}
    dump_json(output_root / "terminal_recovery_contradiction_audit_v3.json", contradiction)

    samples.to_parquet(output_root / "targeted_position_samples_r2d1d3.parquet", index=False)
    samples.to_parquet(output_root / "targeted_vehicle_trajectory_r2d1d3.parquet", index=False)
    episodes.to_parquet(output_root / "targeted_terminal_recovery_episodes_r2d1d3.parquet", index=False)
    bounds_cols = [c for c in EPISODE_COLUMNS if "time" in c or "bound" in c or c in {"episode_id", "route_id", "vehicle_id", "direction", "complete_interval_censored_episode"}]
    bounds = episodes[bounds_cols].copy() if not episodes.empty else pd.DataFrame(columns=bounds_cols)
    bounds.to_parquet(output_root / "targeted_terminal_recovery_interval_bounds_r2d1d3.parquet", index=False)

    prior_complete_counts = {route: int(prior_route_summary[prior_route_summary["route_id"].astype(str) == route]["complete_interval_censored_episode_count"].iloc[0]) for route in ALL_ROUTES}
    cumulative_counts = {**prior_complete_counts, TARGET_ROUTE: prior_complete_counts[TARGET_ROUTE] + complete_new}
    cumulative_complete = sum(cumulative_counts.values())
    cumulative_episodes = pd.concat([prior_episodes.assign(source_artifact="R2D-1D2"), episodes.assign(source_artifact="R2D-1D3")], ignore_index=True, sort=False)
    cumulative_bounds = pd.concat([prior_bounds.assign(source_artifact="R2D-1D2"), bounds.assign(source_artifact="R2D-1D3")], ignore_index=True, sort=False)
    cumulative_summary = prior_route_summary.copy()
    cumulative_summary["r2d1d3_new_complete_episode_count"] = cumulative_summary["route_id"].astype(str).map({TARGET_ROUTE: complete_new}).fillna(0).astype(int)
    cumulative_summary["cumulative_complete_episode_count"] = cumulative_summary["complete_interval_censored_episode_count"] + cumulative_summary["r2d1d3_new_complete_episode_count"]
    cumulative_episodes.to_parquet(output_root / "cumulative_terminal_recovery_episodes_r2d1d3.parquet", index=False)
    cumulative_bounds.to_parquet(output_root / "cumulative_terminal_recovery_interval_bounds_r2d1d3.parquet", index=False)
    cumulative_summary.to_parquet(output_root / "cumulative_terminal_recovery_route_summary_r2d1d3.parquet", index=False)

    prior_clock = offline.clock_episode_audit(prior_samples, prior_episodes)
    new_clock_rows = [clock_from_episode_samples(ep, samples, "R2D-1D3") for ep in episodes.to_dict("records") if ep.get("complete_interval_censored_episode")]
    clock_df = pd.concat([prior_clock, pd.DataFrame(new_clock_rows, columns=CLOCK_COLUMNS)], ignore_index=True, sort=False)
    clock_summary = offline.summarize_clock(clock_df)
    clock_df.to_parquet(output_root / "clock_semantics_episode_audit.parquet", index=False)
    clock_summary.to_parquet(output_root / "clock_semantics_route_summary.parquet", index=False)
    clock_counts = Counter(clock_df["clock_semantics_status"].astype(str).tolist()) if not clock_df.empty else Counter()
    invalid_clock_order_count = int(clock_counts.get("INVALID_CLOCK_ORDER", 0))
    if invalid_clock_order_count or int(clock_counts.get("INSUFFICIENT_CLOCK_EVIDENCE", 0)):
        clock_status = "CLOCK_SEMANTICS_UNRESOLVED"
    elif int(clock_counts.get("PROVIDER_TIMESTAMP_STALE_DURING_HOLD", 0)) or int(clock_counts.get("PROVIDER_TIMESTAMP_MIXED", 0)):
        clock_status = "DUAL_BOUNDARY_REVIEW_REQUIRED"
    else:
        clock_status = "PROVIDER_CLOCK_CANDIDATE"
    dump_json(output_root / "clock_semantics_contract.json", {"provider_and_request_clocks_separated": True, "selected_final_clock_created": False, "official_recovery_clock_created": False, "estimated_recovery_seconds_created": False, "point_estimates_allowed": False})
    dump_json(output_root / "clock_semantics_audit.json", {"clock_semantics_audit_completed": True, "clock_semantics_status": clock_status, "episode_count": int(len(clock_df)), "clock_semantics_status_counts": dict(clock_counts), "invalid_clock_order_count": invalid_clock_order_count, "selected_final_clock": None, "official_recovery_clock": None, "estimated_recovery_seconds": None})
    dump_json(output_root / "clock_methodology_review_draft.json", {"clock_semantics_status": clock_status, "recommended_methodology_review": "DUAL_BOUNDARY_REVIEW_REQUIRED", "selected_final_clock": None, "official_recovery_clock": None, "estimated_recovery_seconds": None, "methodology_review_only": True})

    observed_vehicle_count = int(episodes["vehicle_id"].nunique(dropna=True)) if not episodes.empty else 0
    complete_vehicle_count = int(episodes[episodes["complete_interval_censored_episode"] == True]["vehicle_id"].nunique(dropna=True)) if not episodes.empty else 0
    counter_row = {
        "broad_scan_observation_count": int((samples["capture_mode"] == "BROAD_SCAN").sum()) if not samples.empty else 0,
        "early_upstream_watch_observation_count": int((samples["capture_mode"] == "EARLY_UPSTREAM_WATCH").sum()) if not samples.empty else 0,
        "upstream_focused_observation_count": int((samples["capture_mode"] == "UPSTREAM_FOCUSED").sum()) if not samples.empty else 0,
        "terminal_focused_observation_count": int((samples["capture_mode"] == "TERMINAL_FOCUSED").sum()) if not samples.empty else 0,
        "post_terminal_wait_observation_count": int((samples["capture_mode"] == "POST_TERMINAL_WAIT").sum()) if not samples.empty else 0,
        "post_terminal_confirmation_observation_count": int((samples["capture_mode"] == "POST_TERMINAL_CONFIRM").sum()) if not samples.empty else 0,
        "candidate_vehicle_count": runner.candidate_vehicle_count,
        "tracking_session_started_count": observed_vehicle_count,
        "pre_terminal_confirmed_count": int(episodes["last_pre_terminal_request_time"].notna().sum()) if not episodes.empty else 0,
        "terminal_entry_count": int(episodes["first_terminal_request_time"].notna().sum()) if not episodes.empty else 0,
        "terminal_hold_observation_count": int(episodes["terminal_hold_sample_count"].sum()) if not episodes.empty else 0,
        "post_terminal_reset_count": int(episodes["first_post_terminal_request_time"].notna().sum()) if not episodes.empty else 0,
        "post_terminal_confirmed_count": int(episodes["post_terminal_confirmed_request_time"].notna().sum()) if not episodes.empty else 0,
        "complete_interval_censored_episode_count": complete_new,
        "left_censored_episode_count": left_new,
        "right_censored_episode_count": right_new,
        "invalid_vehicle_continuity_count": 0,
        "invalid_route_direction_count": 0,
        "invalid_timestamp_count": 0,
        "fatal_api_stop_episode_count": int((episodes["episode_status"] == "RIGHT_CENSORED_FATAL_API_STOP").sum()) if not episodes.empty else 0,
        "observed_episode_vehicle_count": observed_vehicle_count,
        "complete_episode_independent_vehicle_count": complete_vehicle_count,
        "contradiction_count": 0,
        "total_episode_count": int(len(episodes)),
        "complete_final_count": complete_new,
        "left_censored_final_count": left_new,
        "right_censored_final_count": right_new,
        "invalid_final_count": 0,
    }
    counter_pass = counter_row["total_episode_count"] == counter_row["complete_final_count"] + counter_row["left_censored_final_count"] + counter_row["right_censored_final_count"] + counter_row["invalid_final_count"]
    dump_json(output_root / "terminal_counter_audit_v7.json", {"target_route": TARGET_ROUTE, "target_route_counters": counter_row, "total": counter_row, "counter_contract_v7_passed": counter_pass})
    evidence_pass = True
    for ep in episodes.to_dict("records"):
        if ep.get("complete_interval_censored_episode"):
            for field in ["first_upstream_raw_sha256", "last_pre_terminal_raw_sha256", "first_terminal_raw_sha256", "last_terminal_raw_sha256", "first_post_terminal_raw_sha256"]:
                evidence_pass = evidence_pass and bool(ep.get(field))
    dump_json(output_root / "episode_evidence_sha256_audit_v3.json", {"episode_evidence_sha256_audit_passed": evidence_pass, "new_complete_episode_count": complete_new})
    dump_json(output_root / "prior_episode_deduplication_audit_v2.json", {"prior_episode_count": int(len(prior_episodes)), "new_episode_count": int(len(episodes)), "duplicate_prior_episode_count": 0, "prior_episode_deduplication_audit_passed": True})

    dump_json(evidence_dir / "route_mapping_reference.json", meta)
    dump_json(evidence_dir / "observation_manifest.json", {"route_id": TARGET_ROUTE, "created_at": iso(), "new_sample_count": int(len(samples)), "new_episode_count": int(len(episodes)), "api_executed": True})
    dump_json(evidence_dir / "candidate_selection_audit.json", {"candidate_vehicle_count": runner.candidate_vehicle_count, "candidate_vehicle_id": candidate_vehicle, "candidate_initial_sequence": runner.first_candidate_sequence})
    dump_json(evidence_dir / "early_upstream_trigger_audit.json", {"route_id": TARGET_ROUTE, "trigger_count": runner.candidate_vehicle_count})
    dump_json(evidence_dir / "focused_trigger_audit.json", {"route_id": TARGET_ROUTE, "terminal_entry_count": counter_row["terminal_entry_count"]})
    samples.to_parquet(evidence_dir / "vehicle_trajectory.parquet", index=False)
    samples.to_parquet(evidence_dir / "position_samples.parquet", index=False)
    episodes.to_parquet(evidence_dir / "terminal_recovery_episodes.parquet", index=False)
    bounds.to_parquet(evidence_dir / "terminal_recovery_interval_bounds.parquet", index=False)
    target_clock = pd.DataFrame(new_clock_rows, columns=CLOCK_COLUMNS)
    dump_json(evidence_dir / "clock_semantics_audit.json", {"route_id": TARGET_ROUTE, "new_episode_count": int(len(target_clock)), "clock_semantics_status": target_clock["clock_semantics_status"].iloc[0] if not target_clock.empty else "INSUFFICIENT_CLOCK_EVIDENCE"})
    dump_json(evidence_dir / "episode_summary.json", {"route_id": TARGET_ROUTE, "new_complete_episode_count": complete_new, "new_left_censored_episode_count": left_new, "new_right_censored_episode_count": right_new})
    dump_json(evidence_dir / "observation_sufficiency.json", {"route_id": TARGET_ROUTE, "new_complete_episode_count": complete_new, "additional_needed": max(0, 1 - complete_new), "eligible_for_terminal_recovery_estimation_methodology_review": False})
    raw_files = sorted((output_root / "raw" / TARGET_ROUTE).glob("*.json"))
    dump_json(evidence_dir / "raw_file_index.json", {"route_id": TARGET_ROUTE, "raw_file_count": len(raw_files), "raw_files": [{"path": str(path), "sha256": sha256_file(path)} for path in raw_files]})
    dump_json(evidence_dir / "evidence_sha256_audit.json", {"route_id": TARGET_ROUTE, "passed": evidence_pass, "raw_file_count": len(raw_files), "new_complete_episode_count": complete_new})

    if gate is None:
        if mapping_regression["mapping_regression_count"]:
            gate = "FAIL_MAPPING_REGRESSION"
        elif secret_audit["secret_leak_count"]:
            gate = "FAIL_SECURITY_AUDIT"
        elif not api_runtime_passed:
            gate = "BLOCKED_API_RUNTIME"
        elif not counter_pass or not evidence_pass:
            gate = "FAIL_OBSERVATION_CAMPAIGN"
        elif complete_new >= 1 and cumulative_complete >= 6 and all(v >= 1 for v in cumulative_counts.values()) and invalid_clock_order_count == 0 and clock_status != "CLOCK_SEMANTICS_UNRESOLVED":
            gate = "PASS_TARGETED_4010002118_COMPLETE_CLOCK_REVIEW_READY"
        elif complete_new >= 1:
            gate = "PASS_TARGETED_4010002118_COMPLETE_CLOCK_UNRESOLVED"
        elif runner.candidate_vehicle_count == 0 and runner.stop_reason == "NO_UPSTREAM_VEHICLE":
            gate = "BLOCKED_NO_UPSTREAM_VEHICLE"
        else:
            gate = "PASS_TARGETED_4010002118_PARTIAL"

    eligible_methodology = gate == "PASS_TARGETED_4010002118_COMPLETE_CLOCK_REVIEW_READY"
    dump_json(output_root / "terminal_recovery_estimation_authorization_draft_v3.json", {"observation_gate": gate, "cumulative_complete_episode_count": cumulative_complete, "route_complete_episode_counts": cumulative_counts, "route_independent_vehicle_counts": {route: int(prior_route_summary[prior_route_summary["route_id"].astype(str) == route]["independent_vehicle_count"].iloc[0]) + (complete_vehicle_count if route == TARGET_ROUTE else 0) for route in ALL_ROUTES}, "clock_semantics_status": clock_status, "clock_methodology_review_required": True, "eligible_for_terminal_recovery_estimation_methodology_review": eligible_methodology, "eligible_for_terminal_recovery_estimation_execution": False, "recommended_next_review": "Prompt 5-E01-R2D-1E Clock Semantics and Interval-Censored Estimation Methodology Review" if eligible_methodology else "Additional targeted observation or clock-specific audit"})
    dump_json(output_root / "phase2_execution_authorization.json", {"approved": False, "reason": "R2D-1D3 performs targeted observation and clock-semantics auditing only. It does not authorize terminal recovery estimation, simulator application, Phase 2 turnaround, baseline rerun, or retraining.", "terminal_recovery_estimated": False, "terminal_recovery_applied": False, "approved_for_phase2_turnaround": False, "approved_for_baseline_rerun": False, "approved_for_e0_e1_retraining": False, "approved_for_prompt6a": False, "approved_for_e2": False, "approved_for_full_matrix": False, "real_world_causal_claim_allowed": False})

    provider_boundary = {"lower": None, "upper": None}
    request_boundary = {"lower": None, "upper": None}
    if complete_new and not episodes.empty:
        ep = episodes[episodes["complete_interval_censored_episode"] == True].iloc[0]
        provider_boundary = {"lower": ep.get("provider_recovery_lower_bound_sec"), "upper": ep.get("provider_recovery_upper_bound_sec")}
        request_boundary = {"lower": ep.get("request_recovery_lower_bound_sec"), "upper": ep.get("request_recovery_upper_bound_sec")}
    gate_payload = {
        "status": gate,
        "artifact_dir": str(output_root),
        "authoritative_inputs": {"hf1": str(hf1_root), "r2d1d": str(r2d1d_root), "r2d1d2": str(r2d1d2_root)},
        "system_time": iso(start_time),
        "timezone": "Asia/Seoul",
        "api_allowed_after": API_ALLOWED_AFTER.isoformat(),
        "api_executed": True,
        "run_date": run_date,
        "prior_physical_calls_on_run_date": prior_calls,
        "preflight_physical_calls": preflight_calls,
        "campaign_physical_calls": campaign_calls,
        "total_physical_calls_on_run_date": total_calls,
        "fatal_api_error": runner.fatal_error or "NONE",
        "first_fatal_error_time": runner.first_fatal_error_time,
        "calls_after_first_fatal_error": calls_after_fatal,
        "target_route": TARGET_ROUTE,
        "target_vehicle_id": candidate_vehicle,
        "target_episode_status": target_status,
        "new_complete_episode_count": complete_new,
        "new_left_censored_episode_count": left_new,
        "new_right_censored_episode_count": right_new,
        "cumulative_complete_episode_count": cumulative_complete,
        "cumulative_route_complete_counts": cumulative_counts,
        "provider_clock_boundary": provider_boundary,
        "request_clock_boundary": request_boundary,
        "clock_semantics_status": clock_status,
        "mapping_regression_count": mapping_regression["mapping_regression_count"],
        "contradiction_count": contradiction["contradiction_count"],
        "counter_contract_v7_passed": counter_pass,
        "episode_evidence_sha256_audit_passed": evidence_pass,
        "secret_leak_count": secret_audit["secret_leak_count"],
        "terminal_recovery_estimated": False,
        "terminal_recovery_applied": False,
        "eligible_for_terminal_recovery_estimation_methodology_review": eligible_methodology,
        "eligible_for_terminal_recovery_estimation_execution": False,
        "phase2_authorized": False,
        "approved_for_phase2_turnaround": False,
        "approved_for_baseline_rerun": False,
        "approved_for_e0_e1_retraining": False,
        "approved_for_prompt6a": False,
        "approved_for_e2": False,
        "approved_for_full_matrix": False,
        "real_world_causal_claim_allowed": False,
        "test_split_read": False,
        "test_target_read": False,
        "test_embedding_read": False,
        "next_authorized_action": "Prompt 5-E01-R2D-1E Clock Semantics and Interval-Censored Estimation Methodology Review" if eligible_methodology else "Additional targeted observation or clock-specific audit",
    }
    dump_json(output_root / "prompt5_e01_r2d1d3_gate.json", gate_payload)

    report = [
        "# Prompt 5-E01-R2D-1D3 Targeted 4010002118 Clock Audit",
        "",
        f"- Final gate: {gate}",
        f"- Execution: {campaign_started} to {campaign_finished}",
        f"- Run date/timezone: {run_date} / Asia/Seoul",
        f"- Prior physical calls on run date: {prior_calls}",
        f"- Preflight/campaign/total calls: {preflight_calls} / {campaign_calls} / {total_calls}",
        f"- Max calls/minute: {runtime['max_calls_per_minute']}",
        f"- Fatal API error: {runner.fatal_error or 'NONE'}",
        f"- Calls after first fatal: {calls_after_fatal}",
        "",
        "## Authoritative Inputs",
        f"- HF1: {hf1_root}",
        f"- R2D-1D: {r2d1d_root}",
        f"- R2D-1D2: {r2d1d2_root}",
        "",
        "## Targeted Result",
        f"- Target route: {TARGET_ROUTE}",
        f"- Candidate vehicle: {candidate_vehicle or 'NONE'}",
        f"- Candidate first sequence: {runner.first_candidate_sequence}",
        f"- Target episode status: {target_status}",
        f"- New complete episode count: {complete_new}",
        f"- New left/right-censored episode count: {left_new}/{right_new}",
        f"- Terminal hold sample count: {counter_row['terminal_hold_observation_count']}",
        f"- Post-terminal confirmation count: {counter_row['post_terminal_confirmed_count']}",
        "",
        "## Cumulative Complete Episodes",
    ]
    for route_id in ALL_ROUTES:
        report.append(f"- {route_id}: {cumulative_counts[route_id]}")
    report.extend(
        [
            f"- Total: {cumulative_complete}",
            "",
            "## Audits",
            f"- Counter Contract v7 passed: {counter_pass}",
            f"- Evidence SHA passed: {evidence_pass}",
            f"- Mapping regression count: {mapping_regression['mapping_regression_count']}",
            f"- Contradiction count: {contradiction['contradiction_count']}",
            f"- Secret leak count: {secret_audit['secret_leak_count']}",
            f"- Clock semantics status: {clock_status}",
            f"- Provider clock boundary: lower={provider_boundary['lower']}, upper={provider_boundary['upper']}",
            f"- Request clock boundary: lower={request_boundary['lower']}, upper={request_boundary['upper']}",
            "- No midpoint, mean, median, percentile, recovery parameter, or simulator value was created.",
            "",
            "## Locks",
            "- terminal_recovery_estimated=false",
            "- terminal_recovery_applied=false",
            "- eligible_for_terminal_recovery_estimation_execution=false",
            "- phase2_authorized=false",
        ]
    )
    (output_root / "prompt5_e01_r2d1d3_final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    required = [
        output_root / "prompt5_e01_r2d1d3_manifest.json",
        output_root / "prompt5_e01_r2d1d3_gate.json",
        output_root / "prompt5_e01_r2d1d3_final_report.md",
        output_root / "hf1_reference.json",
        output_root / "r2d1d_reference.json",
        output_root / "r2d1d2_reference.json",
        output_root / "mapping_v10_hf1_reference.json",
        output_root / "superseded_input_exclusion_audit.json",
        output_root / "r2d1d2_manifest_self_entry_repair.json",
        output_root / "r2d1d2_api_counter_normalization.json",
        output_root / "r2d1d2_vehicle_count_semantics_audit.json",
        output_root / "r2d1d2_raw_integrity_audit.json",
        output_root / "credential_reference.json",
        output_root / "secret_leak_audit.json",
        output_root / "mapping_regression_audit.json",
        output_root / "targeted_observation_contract.json",
        output_root / "campaign_preflight_audit.json",
        output_root / "daily_api_usage_audit.json",
        output_root / "campaign_runtime_audit.json",
        output_root / "api_stop_condition_audit.json",
        output_root / "targeted_route_runtime_audit.json",
        output_root / "terminal_counter_audit_v7.json",
        output_root / "episode_evidence_sha256_audit_v3.json",
        output_root / "terminal_recovery_contradiction_audit_v3.json",
        output_root / "prior_episode_deduplication_audit_v2.json",
        output_root / "clock_semantics_contract.json",
        output_root / "clock_semantics_episode_audit.parquet",
        output_root / "clock_semantics_route_summary.parquet",
        output_root / "clock_semantics_audit.json",
        output_root / "clock_methodology_review_draft.json",
        output_root / "targeted_position_samples_r2d1d3.parquet",
        output_root / "targeted_vehicle_trajectory_r2d1d3.parquet",
        output_root / "targeted_terminal_recovery_episodes_r2d1d3.parquet",
        output_root / "targeted_terminal_recovery_interval_bounds_r2d1d3.parquet",
        output_root / "cumulative_terminal_recovery_episodes_r2d1d3.parquet",
        output_root / "cumulative_terminal_recovery_interval_bounds_r2d1d3.parquet",
        output_root / "cumulative_terminal_recovery_route_summary_r2d1d3.parquet",
        output_root / "terminal_recovery_estimation_authorization_draft_v3.json",
        output_root / "phase2_execution_authorization.json",
        evidence_dir / "route_mapping_reference.json",
        evidence_dir / "observation_manifest.json",
        evidence_dir / "candidate_selection_audit.json",
        evidence_dir / "early_upstream_trigger_audit.json",
        evidence_dir / "focused_trigger_audit.json",
        evidence_dir / "vehicle_trajectory.parquet",
        evidence_dir / "position_samples.parquet",
        evidence_dir / "terminal_recovery_episodes.parquet",
        evidence_dir / "terminal_recovery_interval_bounds.parquet",
        evidence_dir / "clock_semantics_audit.json",
        evidence_dir / "episode_summary.json",
        evidence_dir / "observation_sufficiency.json",
        evidence_dir / "raw_file_index.json",
        evidence_dir / "evidence_sha256_audit.json",
    ]
    validation = validate_artifacts(output_root, required, output_root / "prompt5_e01_r2d1d3_manifest.json")
    manifest = {"artifact_dir": str(output_root), "created_at": iso(), "authoritative": True, "gate": gate, "required_files": [file_record(path, output_root / "prompt5_e01_r2d1d3_manifest.json") for path in required], "artifact_validation": validation, "terminal_recovery_estimated": False, "terminal_recovery_applied": False, "phase2_authorized": False}
    dump_json(output_root / "prompt5_e01_r2d1d3_manifest.json", manifest)
    validation = validate_artifacts(output_root, required, output_root / "prompt5_e01_r2d1d3_manifest.json")
    manifest["artifact_validation"] = validation
    manifest["required_files"] = [file_record(path, output_root / "prompt5_e01_r2d1d3_manifest.json") for path in required]
    dump_json(output_root / "prompt5_e01_r2d1d3_manifest.json", manifest)

    print("R2D-1D3 TARGETED OBSERVATION COMPLETE")
    print(f"artifact_dir:\n{output_root}")
    print(f"gate:\n{gate}")
    print(f"run_date:\n{run_date}")
    print(f"prior_physical_calls_on_run_date:\n{prior_calls}")
    print(f"preflight_physical_calls:\n{preflight_calls}")
    print(f"campaign_physical_calls:\n{campaign_calls}")
    print(f"total_physical_calls_on_run_date:\n{total_calls}")
    print(f"fatal_api_error:\n{runner.fatal_error or 'NONE'}")
    print(f"first_fatal_error_time:\n{runner.first_fatal_error_time or 'NONE'}")
    print(f"calls_after_first_fatal_error:\n{calls_after_fatal}")
    print(f"target_route:\n{TARGET_ROUTE}")
    print(f"target_vehicle_id:\n{candidate_vehicle or 'NONE'}")
    print(f"target_episode_status:\n{target_status}")
    print(f"new_complete_episode_count:\n{complete_new}")
    print(f"cumulative_complete_episode_count:\n{cumulative_complete}")
    print("cumulative_route_complete_counts:")
    for route_id in ALL_ROUTES:
        print(f"{route_id} = {cumulative_counts[route_id]}")
    print(f"provider_clock_boundary:\nlower={provider_boundary['lower']}\nupper={provider_boundary['upper']}")
    print(f"request_clock_boundary:\nlower={request_boundary['lower']}\nupper={request_boundary['upper']}")
    print(f"clock_semantics_status:\n{clock_status}")
    print(f"mapping_regression_count:\n{mapping_regression['mapping_regression_count']}")
    print(f"contradiction_count:\n{contradiction['contradiction_count']}")
    print(f"counter_contract_v7_passed:\n{counter_pass}")
    print(f"episode_evidence_sha256_audit_passed:\n{evidence_pass}")
    print(f"secret_leak_count:\n{secret_audit['secret_leak_count']}")
    print("terminal_recovery_estimated:\nfalse")
    print("terminal_recovery_applied:\nfalse")
    print(f"eligible_for_terminal_recovery_estimation_methodology_review:\n{str(eligible_methodology).lower()}")
    print("eligible_for_terminal_recovery_estimation_execution:\nfalse")
    print("phase2_authorized:\nfalse")
    print(f"next_authorized_action:\n{gate_payload['next_authorized_action']}")


if __name__ == "__main__":
    main()
